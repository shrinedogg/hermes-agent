"""``HostedRoomPolicyRepository`` on PostgreSQL (stage 4): the bounded room policy checkpoint in the
placement's schema, the reference's SQLite checkpoint ported to the PG dialect.

The append-only room log stays the source of truth; the checkpoint materializes only what the next active
discussion needs, by durable cursor. ``snapshot`` runs a fixed sequence: the cursor write, the cursor bound
read, then per log page one read of the page and one write applying it, then the snapshot read. Where main
raises a bare ``RuntimeError``, this raises ``PolicyProjectionError`` with the same message.

The checkpoint tables and the user-thread index come from the declarative DDL (the reference's lazy step is a
no-op here; its recorded replay capacity - the largest active-event budget any checkpoint on the placement has
seen - is kept in ``hosted_room_policy_capacity`` the same way).
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any, Callable

from persistence import JsonValue
from persistence.gateway.hosted_rooms import HostedPlacement, PolicyProjectionError, PolicySnapshot, RoomNotFoundError
from session_rules.hosted_rooms import MAX_LOG_LIMIT, compact_json
from session_rules.hosted_tasks import validate_task_input

from .._backend import PostgresStore, register_repository, translated
from ._hosted_common import fenced_update, jstr, placement_read, placement_write
from .hosted_rooms import _event_from_row, read_events_page

__all__ = ["PgHostedRoomPolicyRepository"]

# The projection bounds; a caller's active-event budget can raise the first, never lower it.
MAX_ACTIVE_POLICY_EVENTS = 64
# ``turn.deferred`` receipts come from explicit user retries, not from the configured reply
# budget, so they are bounded on their own: sizing the reply budget exactly must leave room.
MAX_ACTIVE_RETRY_RECEIPTS = 64
MAX_THREAD_TRANSCRIPT_EVENTS = 24
_TRANSCRIPT_SCHEMA_VERSION = 5  # 5: rebuild threads an old settle closed under a newer message
_TERMINAL_KINDS = frozenset({"turn.settled", "turn.failed", "turn.cancelled", "turn.deferred"})
# What a Continue restores into a reopened discussion's active projection (the cap's activity is not one).
_REOPENED_KINDS = (*sorted(_TERMINAL_KINDS), "message.member", "room.assignment", "room.routed", "room.continued")

# The room event columns every event read here selects.
_ROOM_EVENT_COLUMNS = "room_id, seq, event_id, kind, actor_json, authority_epoch, payload_json, created_at"
_DELETE_ACTIVE_EVENTS_SQL = "DELETE FROM hosted_room_policy_events WHERE room_id = %s AND discussion_event_id = %s"
_TRANSCRIPT_EVENTS_SQL = f"""WITH transcript_events(seq) AS (
        SELECT seq FROM hosted_room_policy_transcript WHERE room_id = %s AND thread_id = %s
        UNION ALL
        SELECT settled_seq FROM hosted_room_policy_transcript
         WHERE room_id = %s AND thread_id = %s AND settled_seq IS NOT NULL)
    SELECT {", ".join("events." + column for column in _ROOM_EVENT_COLUMNS.split(", "))}
    FROM transcript_events
    JOIN hosted_room_events AS events ON events.room_id = %s AND events.seq = transcript_events.seq
    ORDER BY events.seq LIMIT %s"""


def _event_from_room_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """The room event view of one ``hosted_room_events`` row."""
    return _event_from_row(row)


def _text(mapping: Mapping[str, Any], key: str) -> str:
    return str(mapping.get(key) or "")


def _require_room(conn, room_id: str) -> None:
    if conn.execute("SELECT 1 FROM hosted_rooms WHERE room_id = %s", (room_id,)).fetchone() is None:
        raise RoomNotFoundError("hosted room not found")


def _settled_message(
    conn, room_id: str, discussion_event_id: str, message_event_id: Any) -> dict[str, Any] | None:
    """Read the committed message even after its active discussion was compacted."""
    row = conn.execute(
        f"SELECT {_ROOM_EVENT_COLUMNS} FROM hosted_room_events WHERE room_id = %s AND event_id = %s",
        (room_id, message_event_id)).fetchone()
    if row is None:
        return None
    event = _event_from_room_row(row)
    if event["kind"] != "message.member" or _text(event["payload"], "discussion_event_id") != discussion_event_id:
        return None
    return event


def _settled_source_users(
    conn, *, room_id: str, thread_id: str, events: Mapping[int, Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """Load canonical user sources required by retained non-pass settlements."""
    source_ids = {
        _text(event["payload"], "discussion_event_id")
        for event in events.values()
        if event["kind"] == "turn.settled" and not event["payload"].get("passed")
    }
    source_ids.difference_update(_text(event, "event_id") for event in events.values())
    if not source_ids:
        return []
    sources = []
    ids = sorted(source_ids)
    for offset in range(0, len(ids), 500):
        batch = ids[offset:offset + 500]
        placeholders = ", ".join("%s" for _ in batch)
        rows = conn.execute(
            f"SELECT {_ROOM_EVENT_COLUMNS} FROM hosted_room_events "
            f"WHERE room_id = %s AND kind = 'message.user' "
            f"AND {jstr('payload_json', 'thread_id')} = %s AND event_id IN ({placeholders})",
            (room_id, thread_id, *batch),
        ).fetchall()
        sources.extend(_event_from_room_row(row) for row in rows)
    return sources


def _store_active_event(
    conn, *, event: Mapping[str, Any], thread_id: str, discussion_event_id: str) -> None:
    conn.execute("""INSERT INTO hosted_room_policy_events(
               room_id, thread_id, discussion_event_id, seq, event_json
           ) VALUES (%s, %s, %s, %s, %s)
           ON CONFLICT (room_id, seq) DO NOTHING""",
        (event["room_id"], thread_id, discussion_event_id, int(event["seq"]), compact_json(dict(event))))


def _store_transcript_event(
    conn, *, event: Mapping[str, Any], thread_id: str, settled_seq: int | None = None) -> None:
    conn.execute("""INSERT INTO hosted_room_policy_transcript(
               room_id, thread_id, seq, kind, settled_seq
           ) VALUES (%s, %s, %s, %s, %s)
           ON CONFLICT (room_id, thread_id, seq) DO UPDATE SET
               settled_seq = COALESCE(EXCLUDED.settled_seq, hosted_room_policy_transcript.settled_seq)""",
        (event["room_id"], thread_id, int(event["seq"]), str(event["kind"]), settled_seq))
    if event["kind"] in {"message.user", "message.member"}:
        cutoff = conn.execute("""SELECT seq FROM hosted_room_policy_transcript
               WHERE room_id = %s AND thread_id = %s AND kind IN ('message.user', 'message.member')
               ORDER BY seq DESC LIMIT 1 OFFSET %s""",
            (event["room_id"], thread_id, MAX_THREAD_TRANSCRIPT_EVENTS - 1)).fetchone()
        if cutoff is not None:
            conn.execute(
                "DELETE FROM hosted_room_policy_transcript WHERE room_id = %s AND thread_id = %s AND seq < %s",
                (event["room_id"], thread_id, int(cutoff["seq"])))


def _discussion_events(
    conn, *, room_id: str, thread_id: str, discussion_event_id: str, bound_error: str,
    max_active_events: int,
) -> list[dict[str, Any]]:
    """Merge the thread transcript with the active projection, ordered by seq."""
    active_rows = []
    for receipts, bound in ((False, max_active_events), (True, MAX_ACTIVE_RETRY_RECEIPTS)):
        rows = conn.execute(f"""SELECT event_json FROM hosted_room_policy_events
               WHERE room_id = %s AND discussion_event_id = %s
                 AND ({jstr('event_json', 'kind')} = 'turn.deferred') = %s
               ORDER BY seq LIMIT %s""",
            (room_id, discussion_event_id, receipts, bound + 1)).fetchall()
        if len(rows) > bound:
            raise PolicyProjectionError(bound_error)
        active_rows.extend(rows)
    # LIMIT also reads one overflow sentinel; keep its parameter inside the database's integer range.
    max_transcript_policy_events = min(
        2**63 - 2, MAX_THREAD_TRANSCRIPT_EVENTS * (max_active_events + MAX_ACTIVE_RETRY_RECEIPTS + 2))
    rows = conn.execute(_TRANSCRIPT_EVENTS_SQL,
                        (room_id, thread_id, room_id, thread_id, room_id,
                         max_transcript_policy_events + 1)).fetchall()
    if len(rows) > max_transcript_policy_events:
        raise PolicyProjectionError("thread policy transcript exceeded its bound")
    events_by_seq = {
        int(event["seq"]): event
        for event in (*map(_event_from_room_row, rows), *(json.loads(row["event_json"]) for row in active_rows))}
    for event in _settled_source_users(conn, room_id=room_id, thread_id=thread_id, events=events_by_seq):
        events_by_seq[int(event["seq"])] = event
    return [events_by_seq[seq] for seq in sorted(events_by_seq)]


# -- per-kind projection handlers (dispatched by _apply_event) -----------


def _apply_user_message(conn, event: Mapping[str, Any], payload: Mapping[str, Any]) -> None:
    room_id = str(event["room_id"])
    thread_id, event_id = _text(payload, "thread_id"), _text(event, "event_id")
    if not thread_id or not event_id:
        return
    conn.execute("""INSERT INTO hosted_room_policy_threads(
               room_id, thread_id, discussion_event_id, latest_user_seq, completed
           ) VALUES (%s, %s, %s, %s, 0)
           ON CONFLICT (room_id, thread_id) DO UPDATE SET
               discussion_event_id = EXCLUDED.discussion_event_id,
               latest_user_seq = EXCLUDED.latest_user_seq, completed = 0""",
        (room_id, thread_id, event_id, int(event["seq"])))
    _store_active_event(conn, event=event, thread_id=thread_id, discussion_event_id=event_id)
    _store_transcript_event(conn, event=event, thread_id=thread_id)


def _apply_discussion_event(conn, event: Mapping[str, Any], payload: Mapping[str, Any]) -> None:
    """Index member messages and terminal turn outcomes of a known discussion."""
    room_id, seq, kind = str(event["room_id"]), int(event["seq"]), _text(event, "kind")
    thread_id, discussion_event_id = _text(payload, "thread_id"), _text(payload, "discussion_event_id")
    if conn.execute(
        "SELECT 1 FROM hosted_room_policy_events WHERE room_id = %s AND discussion_event_id = %s LIMIT 1",
        (room_id, discussion_event_id)).fetchone() is not None:
        _store_active_event(conn, event=event, thread_id=thread_id, discussion_event_id=discussion_event_id)
    # Late outcomes still need publication receipts and transcript commits,
    # but must not resurrect a completed discussion's active projection.
    if kind not in _TERMINAL_KINDS:
        return
    task_id = _text(payload, "task_id")
    execution_generation = int(payload.get("execution_generation") or 0) if kind == "turn.deferred" else 0
    if task_id:
        conn.execute("""INSERT INTO hosted_room_policy_publications(
                   room_id, task_id, kind, execution_generation, seq
               ) VALUES (%s, %s, %s, %s, %s)
               ON CONFLICT (room_id, task_id, kind, execution_generation) DO NOTHING""",
            (room_id, task_id, kind, execution_generation, seq))
    member_id = _text(payload, "member_id")
    seen_through_seq = int(payload.get("seen_through_seq") or 0)
    if kind == "turn.settled" and payload.get("message_event_id"):
        committed = _settled_message(conn, room_id, discussion_event_id, payload["message_event_id"])
        if committed is not None:
            seen_through_seq = max(seen_through_seq, int(committed["seq"]))
            _store_transcript_event(conn, event=committed, thread_id=thread_id, settled_seq=seq)
    else:
        # Non-visible receipts still supply historical reconstruction watermarks.
        _store_transcript_event(conn, event=event, thread_id=thread_id)
    if member_id and seen_through_seq > 0:
        conn.execute("""INSERT INTO hosted_room_policy_watermarks(
                   room_id, thread_id, member_id, seen_through_seq
               ) VALUES (%s, %s, %s, %s)
               ON CONFLICT (room_id, thread_id, member_id) DO UPDATE SET
                   seen_through_seq = GREATEST(hosted_room_policy_watermarks.seen_through_seq, EXCLUDED.seen_through_seq)""",
            (room_id, thread_id, member_id, seen_through_seq))


def _apply_room_activity(conn, event: Mapping[str, Any], payload: Mapping[str, Any]) -> None:
    room_id, thread_id = str(event["room_id"]), _text(payload, "thread_id")
    discussion_event_id = _text(payload, "discussion_event_id")
    conn.execute(_DELETE_ACTIVE_EVENTS_SQL, (room_id, discussion_event_id))
    # A newer message on this thread may already have reopened it under its own discussion: the
    # old discussion's completion (planned before that message) must not close the new one.
    conn.execute("DELETE FROM hosted_room_policy_threads WHERE room_id = %s AND thread_id = %s AND discussion_event_id = %s",
                 (room_id, thread_id, discussion_event_id))


def _apply_continued(conn, event: Mapping[str, Any], payload: Mapping[str, Any]) -> None:
    """Continue reopens a discussion its cap completed: restore its thread row and active projection
    from the room log (the cap's ``room.activity`` dropped both)."""
    room_id, discussion_event_id = str(event["room_id"]), _text(payload, "discussion_event_id")
    source = conn.execute(f"SELECT {_ROOM_EVENT_COLUMNS} FROM hosted_room_events WHERE room_id = %s AND event_id = %s",
                          (room_id, discussion_event_id)).fetchone()
    if source is None or source["kind"] != "message.user":
        return
    source_event = _event_from_room_row(source)
    thread_id = _text(source_event["payload"], "thread_id")
    # A Continue written by an older build after a newer message on the thread (the router now
    # refuses those) must not hand the thread back: the newer discussion owns it.
    if conn.execute(f"""SELECT 1 FROM hosted_room_events WHERE room_id = %s AND seq > %s AND seq <= %s
            AND kind = 'message.user' AND {jstr('payload_json', 'thread_id')} = %s LIMIT 1""",
            (room_id, int(source_event["seq"]), int(event["seq"]), thread_id)).fetchone():
        return
    conn.execute("""INSERT INTO hosted_room_policy_threads(
               room_id, thread_id, discussion_event_id, latest_user_seq, completed
           ) VALUES (%s, %s, %s, %s, 0)
           ON CONFLICT (room_id, thread_id) DO UPDATE SET
               discussion_event_id = EXCLUDED.discussion_event_id,
               latest_user_seq = EXCLUDED.latest_user_seq, completed = 0""",
        (room_id, thread_id, discussion_event_id, int(source_event["seq"])))
    _store_active_event(conn, event=source_event, thread_id=thread_id, discussion_event_id=discussion_event_id)
    for row in conn.execute(f"""SELECT {_ROOM_EVENT_COLUMNS} FROM hosted_room_events
            WHERE room_id = %s AND seq > %s AND seq <= %s AND kind IN ({','.join('%s' for _ in _REOPENED_KINDS)})
              AND {jstr('payload_json', 'discussion_event_id')} = %s ORDER BY seq""",
            (room_id, int(source_event["seq"]), int(event["seq"]), *_REOPENED_KINDS,
             discussion_event_id)).fetchall():
        _store_active_event(conn, event=_event_from_room_row(row), thread_id=thread_id,
                            discussion_event_id=discussion_event_id)


def _apply_stop_requested(conn, event: Mapping[str, Any], payload: Mapping[str, Any]) -> None:
    conn.execute("""UPDATE hosted_room_policy_cursors
           SET stopped_through_seq = GREATEST(stopped_through_seq, %s) WHERE room_id = %s""",
        (int(event["seq"]), str(event["room_id"])))


_APPLY_BY_KIND: dict[str, Callable[..., None]] = {
    "message.user": _apply_user_message, "message.member": _apply_discussion_event,
    **dict.fromkeys(_TERMINAL_KINDS, _apply_discussion_event), "room.activity": _apply_room_activity,
    # The lead's assignment joins its discussion's projection so the planner sees who takes round 0.
    "room.assignment": _apply_discussion_event,
    "room.routed": _apply_discussion_event, "room.continued": _apply_continued,
    "room.stop_requested": _apply_stop_requested}


def _apply_event(conn, event: Mapping[str, Any]) -> None:
    handler = _APPLY_BY_KIND.get(_text(event, "kind"))
    if handler is not None:
        payload = event.get("payload")
        handler(conn, event, payload if isinstance(payload, Mapping) else {})


def _ensure_cursor_and_transcript(conn, room_id: str) -> int:
    """Create the cursor and migrate derived policy caches by bounded replay."""
    _require_room(conn, room_id)
    conn.execute("""INSERT INTO hosted_room_policy_cursors(room_id, through_seq, stopped_through_seq, updated_at)
           VALUES (%s, 0, 0, 0)
           ON CONFLICT (room_id) DO NOTHING""", (room_id,))
    cursor = int(
        conn.execute("SELECT through_seq FROM hosted_room_policy_cursors WHERE room_id = %s", (room_id,)).fetchone()[
            "through_seq"])
    transcript_state = conn.execute(
        "SELECT schema_version FROM hosted_room_policy_transcript_state WHERE room_id = %s", (room_id,)).fetchone()
    if transcript_state is None or int(transcript_state["schema_version"]) < _TRANSCRIPT_SCHEMA_VERSION:
        # Rebuild derived history without rewriting admissions or the room log.
        for table in ("hosted_room_policy_events", "hosted_room_policy_threads",
                      "hosted_room_policy_watermarks", "hosted_room_policy_transcript"):
            conn.execute(f"DELETE FROM {table} WHERE room_id = %s", (room_id,))
        conn.execute("UPDATE hosted_room_policy_cursors SET through_seq = 0, stopped_through_seq = 0 WHERE room_id = %s",
                     (room_id,))
        cursor = 0
        conn.execute("""INSERT INTO hosted_room_policy_transcript_state(room_id, schema_version)
               VALUES (%s, %s)
               ON CONFLICT (room_id) DO UPDATE SET schema_version = EXCLUDED.schema_version""",
            (room_id, _TRANSCRIPT_SCHEMA_VERSION))
    return cursor


# The message and terminal events published for ``task_id``; refuses one whose payload names another task.
def _published_task_events(conn, *, room_id: str, task_id: str | None) -> list[dict[str, Any]]:
    if not task_id:
        return []
    digest = task_id.removeprefix("dtask:")
    rows = conn.execute(
        """SELECT * FROM hosted_room_events
           WHERE room_id = %s AND event_id IN (%s, %s) ORDER BY seq""",
        (room_id, f"dmessage:{digest}", f"dterminal:{digest}"),
    ).fetchall()
    events = [_event_from_room_row(row) for row in rows]
    if any(event["payload"].get("task_id") != task_id for event in events):
        raise PolicyProjectionError("published task receipt identity changed")
    return events


class PgHostedRoomPolicyRepository:
    """The room policy checkpoint in the placement's schema."""

    def __init__(self, store: PostgresStore) -> None:
        self._store = store

    def _prepare(self, placement: HostedPlacement, max_active_events: int) -> int:
        """Validate the budget, then record the budget's capacity in the placement's capacity row (kept across
        restarts: the largest recorded budget wins) and return the placement's replay capacity as read then."""
        if isinstance(max_active_events, bool) or not isinstance(max_active_events, int) or max_active_events < 1:
            raise ValueError("max_active_events must be a positive integer")
        # LIMIT also reads one overflow sentinel; keep its parameter inside the database's integer range.
        budget = min(2**63 - 2, max(MAX_ACTIVE_POLICY_EVENTS, max_active_events))

        def _txn(conn) -> int:
            conn.execute("""INSERT INTO hosted_room_policy_capacity(singleton, max_active_events)
                   VALUES (1, %s)
                   ON CONFLICT (singleton) DO UPDATE SET
                       max_active_events = GREATEST(hosted_room_policy_capacity.max_active_events,
                                                   EXCLUDED.max_active_events)""", (budget,))
            return int(conn.execute(
                "SELECT max_active_events FROM hosted_room_policy_capacity WHERE singleton = 1").fetchone()[
                "max_active_events"])
        with translated():
            with placement_write(self._store, placement) as conn:
                return _txn(conn)

    def _sync(self, placement: HostedPlacement, *, room_id: str, latest_seq: int) -> int:
        """Materialize each unseen event exactly once by durable cursor."""
        def _cursor(conn) -> int:
            return _ensure_cursor_and_transcript(conn, room_id)
        with translated():
            with placement_write(self._store, placement) as conn:
                cursor = _cursor(conn)
        if cursor > latest_seq:
            def _bound(conn) -> None:
                room = conn.execute("SELECT next_seq FROM hosted_rooms WHERE room_id = %s", (room_id,)).fetchone()
                if room is None:
                    raise RoomNotFoundError("hosted room not found")
                if cursor >= int(room["next_seq"]):
                    raise PolicyProjectionError("room policy cursor is ahead of the durable log")
            with translated():
                with placement_read(self._store, placement) as conn:
                    _bound(conn)
        while cursor < latest_seq:
            def _read_page(conn) -> dict[str, Any]:
                return read_events_page(conn, room_id=room_id, since_seq=cursor, limit=MAX_LOG_LIMIT)
            with translated():
                with placement_read(self._store, placement) as conn:
                    page = _read_page(conn)
            rows = [event for event in page.get("events", []) if isinstance(event, Mapping)]
            next_cursor = int(page.get("cursor") or cursor)
            if not rows or next_cursor <= cursor:
                raise PolicyProjectionError("hosted room policy cursor did not advance")

            def _apply(conn) -> int | None:
                _require_room(conn, room_id)
                current = conn.execute("SELECT through_seq FROM hosted_room_policy_cursors WHERE room_id = %s",
                                       (room_id,)).fetchone()
                if current is None:
                    raise PolicyProjectionError("room policy cursor disappeared during replay")
                if int(current["through_seq"]) != cursor:
                    # Another worker applied the page: reload before any projection effects.
                    return int(current["through_seq"])
                for event in rows:
                    _apply_event(conn, event)
                fenced_update(
                    conn, "UPDATE hosted_room_policy_cursors SET through_seq = %s, updated_at = %s WHERE room_id = %s",
                    (next_cursor, float(rows[-1].get("created_at") or 0), room_id),
                    PolicyProjectionError("room policy cursor disappeared during replay"))
                return None
            with translated():
                with placement_write(self._store, placement) as conn:
                    moved = _apply(conn)
            cursor = next_cursor if moved is None else moved
        return cursor

    def snapshot(self, *, placement: HostedPlacement, room_id: str, latest_seq: int,
                 max_active_events: int = MAX_ACTIVE_POLICY_EVENTS) -> PolicySnapshot:
        """Return only the oldest active discussion and its watermark set."""
        def _read(conn) -> PolicySnapshot:
            cursor = conn.execute(
                "SELECT through_seq, stopped_through_seq FROM hosted_room_policy_cursors WHERE room_id = %s",
                (room_id,)).fetchone()
            if cursor is None:
                raise RoomNotFoundError("hosted room checkpoint not found")
            through_seq = int(cursor["through_seq"])
            stopped_through_seq = int(cursor["stopped_through_seq"])
            thread = conn.execute("""SELECT thread_id, discussion_event_id FROM hosted_room_policy_threads
                   WHERE room_id = %s AND completed = 0 AND latest_user_seq > %s
                   ORDER BY latest_user_seq, thread_id LIMIT 1""", (room_id, stopped_through_seq)).fetchone()
            if thread is None:
                return PolicySnapshot(
                    through_seq=through_seq, stopped_through_seq=stopped_through_seq, events=(), watermarks={})
            thread_id = str(thread["thread_id"])
            events = _discussion_events(
                conn, room_id=room_id, thread_id=thread_id, discussion_event_id=str(thread["discussion_event_id"]),
                bound_error="active room policy projection exceeded its bound", max_active_events=capacity)
            watermark_rows = conn.execute("""SELECT member_id, seen_through_seq FROM hosted_room_policy_watermarks
                   WHERE room_id = %s AND thread_id = %s""", (room_id, thread_id)).fetchall()
            return PolicySnapshot(
                through_seq=through_seq, stopped_through_seq=stopped_through_seq, events=tuple(events),
                watermarks={(thread_id, str(row["member_id"])): int(row["seen_through_seq"]) for row in watermark_rows})
        with translated():
            capacity = self._prepare(placement, max_active_events)
            self._sync(placement, room_id=room_id, latest_seq=latest_seq)
            with placement_read(self._store, placement) as conn:
                return _read(conn)

    def publication_exists(self, *, placement: HostedPlacement, room_id: str, task_id: str, status: str,
                           execution_generation: int, max_active_events: int = MAX_ACTIVE_POLICY_EVENTS) -> bool:
        """Return whether one exact driver outcome is already in the room log."""
        sql, params = (
            ("""SELECT 1 FROM hosted_room_policy_publications
                     WHERE room_id = %s AND task_id = %s AND kind = %s AND execution_generation = %s""",
             (room_id, task_id, f"turn.{status}", execution_generation))
            if status == "deferred" else
            ("""SELECT 1 FROM hosted_room_policy_publications
                     WHERE room_id = %s AND task_id = %s AND kind IN ('turn.settled', 'turn.failed', 'turn.cancelled')""",
             (room_id, task_id)))

        def _read(conn) -> bool:
            return conn.execute(sql, params).fetchone() is not None
        with translated():
            self._prepare(placement, max_active_events)
            with placement_read(self._store, placement) as conn:
                return _read(conn)

    def events_for_task(self, *, placement: HostedPlacement, room_id: str, source_event_seq: int,
                        input_context: Mapping[str, JsonValue] | None = None, task_id: str | None = None,
                        max_active_events: int = MAX_ACTIVE_POLICY_EVENTS) -> tuple[Mapping[str, JsonValue], ...]:
        """Load one bounded discussion projection for terminal reconstruction."""
        def _read(conn) -> list[dict[str, Any]]:
            published = _published_task_events(conn, room_id=room_id, task_id=task_id)
            if input_context is not None:
                context = validate_task_input(input_context)
                seqs = sorted({source_event_seq, *context["event_seqs"]})
                placeholders = ",".join("%s" for _ in seqs)
                rows = conn.execute(f"SELECT * FROM hosted_room_events WHERE room_id = %s AND seq IN ({placeholders}) ORDER BY seq",
                                    (room_id, *seqs)).fetchall()
                if len(rows) != len(seqs):
                    raise PolicyProjectionError("admitted task input event is missing")
                events = {int(row["seq"]): _event_from_room_row(row) for row in rows}
                source_event = events[source_event_seq]
                thread_id = _text(source_event["payload"], "thread_id")
                if source_event["kind"] != "message.user" or not thread_id:
                    raise PolicyProjectionError("admitted task source is not a user discussion")
                # Supersession remains authoritative after the display projection compacts.
                latest = conn.execute(f"""SELECT * FROM hosted_room_events
                    WHERE room_id = %s AND kind = 'message.user' AND {jstr('payload_json', 'thread_id')} = %s
                    ORDER BY seq DESC LIMIT 1""", (room_id, thread_id)).fetchone()
                if latest is not None:
                    events[int(latest["seq"])] = _event_from_room_row(latest)
                for event in published:
                    events[int(event["seq"])] = event
                return [events[seq] for seq in sorted(events)]
            row = conn.execute(
                f"SELECT {_ROOM_EVENT_COLUMNS} FROM hosted_room_events WHERE room_id = %s AND seq = %s",
                (room_id, source_event_seq)).fetchone()
            if row is None or row["kind"] != "message.user":
                return published
            source = _event_from_room_row(row)
            events = _discussion_events(
                conn, room_id=room_id, thread_id=_text(source["payload"], "thread_id"),
                discussion_event_id=str(source["event_id"]),
                bound_error="task policy projection exceeded its bound", max_active_events=capacity)
            # The source can age out of BOTH bounded projections while a
            # deferred task remains retryable. Its frozen prompt lives in the task.
            by_seq = {event["seq"]: event for event in (*events, *published)}
            by_seq[source_event_seq] = source
            return [by_seq[seq] for seq in sorted(by_seq)]
        with translated():
            capacity = self._prepare(placement, max_active_events)
            with placement_read(self._store, placement) as conn:
                return tuple(_read(conn))

    def compact_completed(self, *, placement: HostedPlacement, room_id: str,
                          max_active_events: int = MAX_ACTIVE_POLICY_EVENTS) -> None:
        """Drop any completed projections left by an interrupted sync."""
        def _txn(conn) -> None:
            for row in conn.execute(
                "SELECT discussion_event_id FROM hosted_room_policy_threads WHERE room_id = %s AND completed = 1",
                (room_id,)).fetchall():
                conn.execute(_DELETE_ACTIVE_EVENTS_SQL, (room_id, str(row["discussion_event_id"])))
            conn.execute("DELETE FROM hosted_room_policy_threads WHERE room_id = %s AND completed = 1", (room_id,))
        with translated():
            self._prepare(placement, max_active_events)
            with placement_write(self._store, placement) as conn:
                _txn(conn)


def _register() -> None:
    register_repository("hosted_room_policy", lambda store: PgHostedRoomPolicyRepository(store))


_register()

"""``HostedRoomRepository`` on PostgreSQL (stage 4): hosted room owners, transport bindings and member
sessions in the bound profile's schema through the store handle, and the rooms and their event log in
the schema ``placement`` names, each write one transaction - the SQLite room store's SQL ported to the
PG dialect.

The reference's two files map onto the store's own schema (the ``profile`` placement) and the backend's
one fixed shared schema (the ``shared`` placement); the hand-off's copy is a cross-schema read of the
shared schema into the store's schema (the reference's cross-file ``_copy_rows`` twin; both schemas carry
the same declarative DDL, so the copy is a plain per-table column copy and the legacy column mapping has
no work to do).
"""

from __future__ import annotations

import json
import time
from collections.abc import Collection, Mapping, Sequence
from typing import Any

from persistence import AuthorityRefusal, JsonValue, SessionId
from persistence.authority.vocabulary import OwnerRule
from persistence.conversation.sessions import TitleConflict, sanitize_title
from persistence.gateway.hosted_rooms import (
    AuthorityConflictError, EventConflictError, EventCursorConflictError, HostedMemberGuard, HostedOwnerClaim,
    HostedPlacement, HostedRoomError, HostedRoomRefusal, RoomConflictError, RoomHeldGrantsPendingError,
    RoomHistoryExpiredError, RoomNotFoundError,
)
from session_rules.hosted_rooms import (
    CONTEXT_EVENT_KIND, CONTROL_EVENT_BYTE_RESERVE, CONTROL_EVENT_COUNT_RESERVE, DISBANDED_ROOM_RETENTION_SECONDS,
    IDENTIFIER_RE, MAX_ACTIVE_ROOMS, MAX_DISBANDED_ROOM_TOMBSTONES, MAX_EVENTS_PER_ROOM, MAX_GATEWAY_EVENT_BYTES,
    MAX_LOG_PAGE_BYTES, MAX_ROOM_EVENT_BYTES, _CONTEXT_ACTOR, _CONTROL_EVENT_KINDS, _EMPTY_CONTEXT, _actor_id,
    _actor_json, _claim_payload_json, _control_event, _legacy_members_match, _payload_json, _room_id,
    _system_actor_json, _validate_members, _validate_room_name, clock as _now, context_from_payload, retry_fields,
    utf8_len, validate_context,
)
from session_rules.legacy_local import same_owner

from .._backend import PostgresStore, register_repository, translated
from ._hosted_common import (
    _epoch, fenced_update as _fenced_update, placement_all, placement_one, placement_read, placement_write,
    probe, table_exists,
)
from .hosted_room_replicas import _append_control_event

__all__ = ["PgHostedRoomRepository", "_event_from_row", "_member_authorized_locked", "_room_from_row",
           "read_events_page"]

# The ``state_meta`` key prefixes of room owners, transport bindings and member homes, as main's gateway writes them.
OWNER_PREFIX = "gateway.hosted.owner.v1:"
TRANSPORT_PREFIX = "gateway.hosted.transport.v1:"
MEMBER_HOMES_PREFIX = "gateway.hosted.member_homes.v1:"

_ROOM_SESSION_CONFLICT = "room session identity conflicts with existing data"
_ROOM_TITLE_CONFLICT = (
    "Another group already uses this room title on the target gateway. Rename or migrate that group before retrying.")

# --- SQL fragments (statement text must stay byte-stable after whitespace normalisation) ---
_EVENT_COLUMNS = ("room_id, seq, event_id, kind, actor_json, authority_epoch, payload_json, created_at")
_SELECT_EVENT = f"SELECT {_EVENT_COLUMNS} FROM hosted_room_events WHERE room_id = %s AND event_id = %s"
_INSERT_EVENT = (f"INSERT INTO hosted_room_events ({_EVENT_COLUMNS}) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)")
_ROOM_COLUMNS = (
    "room_id, name, members_json, authority_gateway_id, authority_epoch, next_seq, revision,"
    " created_at, updated_at, disbanded_at")
_ROOM_COLUMNS_WITH_BYTES = (
    "room_id, name, members_json, authority_gateway_id, authority_epoch, next_seq, event_bytes,"
    " revision, created_at, updated_at, disbanded_at")
_SELECT_ROOM = f"SELECT {_ROOM_COLUMNS} FROM hosted_rooms WHERE room_id = %s"
_SELECT_ROOM_WITH_BYTES = f"SELECT {_ROOM_COLUMNS_WITH_BYTES} FROM hosted_rooms WHERE room_id = %s"
_SUM_EVENT_BYTES = "SELECT COALESCE(SUM(event_bytes), 0) AS total FROM hosted_rooms"
_INSERT_RETIRED = ("INSERT INTO hosted_room_retired_ids (room_id, retired_at) VALUES (%s, %s) ON CONFLICT (room_id) DO NOTHING")
# The router-call claims, created by the first claim or release on the store. A claim outlives any router call; one
# a crashed worker left expires, and the next preparation asks again.
_ROUTER_CLAIMS_DDL = """CREATE TABLE IF NOT EXISTS hosted_room_router_claims (
    room_id TEXT NOT NULL, discussion_event_id TEXT NOT NULL, owner TEXT NOT NULL, expires_at DOUBLE PRECISION NOT NULL,
    PRIMARY KEY(room_id, discussion_event_id))"""
# The hand-off table, created by the first hand-off as the reference does (a write, as main's).
_HANDOFF_TABLE = "hosted_room_gateway_handoffs"
_HANDOFF_DDL = (
    f"CREATE TABLE IF NOT EXISTS {_HANDOFF_TABLE} (room_id TEXT PRIMARY KEY, handed_at DOUBLE PRECISION NOT NULL)")

# The room-scoped hosted tables the hand-off copy walks, parents first (the foreign keys run ahead of
# their children; the liveness tables a fresh driver claims instead stay out, as the reference's
# ``_SKIP_TABLES``). The copy is a plain per-table column copy: both schemas carry the same declarative
# DDL, so a column list read from the catalog always matches on both sides. The hand-off's own record
# table is not copied: the gateway store records its hand-offs itself, and its claim runs off the room
# rows and the retired ids.
_COPY_TABLES = (
    "hosted_rooms", "hosted_room_events", "hosted_room_retired_ids", "hosted_room_links",
    "hosted_room_remote_runs", "hosted_room_peer_reservations", "hosted_room_owed_revokes",
    "hosted_room_driver_tasks", "hosted_room_policy_cursors", "hosted_room_policy_threads",
    "hosted_room_policy_events", "hosted_room_policy_watermarks", "hosted_room_policy_publications",
    "hosted_room_policy_transcript", "hosted_room_policy_transcript_state",
    "hosted_room_replicas", "hosted_room_replica_events", "hosted_room_attachments",
    "hosted_room_work_records_source", "hosted_room_work_records_pending", "hosted_room_work_records_invalid")


# --- row helpers ------------------------------------------------------------------
def _is_retired(conn, room_id: str) -> bool:
    return conn.execute("SELECT 1 FROM hosted_room_retired_ids WHERE room_id = %s", (room_id,)).fetchone() is not None


def _room_row(conn, sql: str, params: tuple[Any, ...], room_id: str) -> dict[str, Any]:
    """Fetch one hosted_rooms row or raise the precise not-found/expired error."""
    row = conn.execute(sql, params).fetchone()
    if row is not None:
        return row
    # A retained disband tombstone still has replayable history; the caller did not opt into disbanded rooms.
    retained = conn.execute("SELECT 1 FROM hosted_rooms WHERE room_id = %s", (room_id,)).fetchone()
    if retained is None and _is_retired(conn, room_id):
        raise RoomHistoryExpiredError("Group Chat history expired; room_id remains permanently retired")
    raise RoomNotFoundError("hosted room not found")


# Refuses with ``message`` unless the room row names this authority and epoch.
def _require_authority(room: Mapping[str, Any], gateway_id: str, epoch: int, message: str) -> None:
    if str(room["authority_gateway_id"]) != gateway_id or int(room["authority_epoch"]) != epoch:
        raise AuthorityConflictError(message)


def _reload(conn, sql: str, params: tuple, missing: str) -> dict[str, Any]:
    """Re-read a row this transaction just wrote; a miss is an invariant violation."""
    row = conn.execute(sql, params).fetchone()
    if row is None:  # pragma: no cover - guarded by the write above
        raise RuntimeError(missing)
    return row


# The room view of a row; ``disbanded_at`` and ``latest_seq`` appear only when the row carries them.
def _room_from_row(row: Mapping[str, Any], *, idempotent: bool = False) -> dict[str, Any]:
    keys = row.keys()  # dict rows: the key check below is load-bearing.
    return {
        "room_id": row["room_id"], "name": row["name"], "members": json.loads(row["members_json"]),
        "authority_gateway_id": row["authority_gateway_id"], "authority_epoch": int(row["authority_epoch"]),
        "revision": int(row["revision"]), "created_at": float(row["created_at"]),
        "updated_at": float(row["updated_at"]), "idempotent": idempotent,
        **({"disbanded_at": float(row["disbanded_at"])} if "disbanded_at" in keys and row["disbanded_at"] is not None
           else {}),
        **({"latest_seq": int(row["next_seq"]) - 1} if "next_seq" in keys else {})}


# The event view of a row, with actor and payload decoded.
def _event_from_row(row: Mapping[str, Any], *, idempotent: bool = False) -> dict[str, Any]:
    epoch = row["authority_epoch"]
    return {
        "room_id": row["room_id"], "seq": int(row["seq"]), "event_id": row["event_id"], "kind": row["kind"],
        "actor": json.loads(row["actor_json"]), "authority_epoch": int(epoch) if epoch is not None else None,
        "payload": json.loads(row["payload_json"]), "created_at": float(row["created_at"]), "idempotent": idempotent}


# The event row bound to ``event_id`` in the room, or None.
def _load_event(conn, room_id: str, event_id: str) -> dict[str, Any] | None:
    return conn.execute(_SELECT_EVENT, (room_id, event_id)).fetchone()


# The payload of the room's newest event of ``kind``, of the newest below ``before_seq`` when given, or None.
def _latest_payload(conn, room_id: str, kind: str, before_seq: int | None) -> Any:
    sql = "SELECT payload_json FROM hosted_room_events WHERE room_id = %s AND kind = %s"
    params: tuple[Any, ...] = (room_id, kind)
    if before_seq is not None:
        sql, params = sql + " AND seq < %s", (*params, int(before_seq))
    row = conn.execute(sql + " ORDER BY seq DESC LIMIT 1", params).fetchone()
    return None if row is None else json.loads(row["payload_json"])


def _event_content(row: Mapping[str, Any]) -> tuple[Any, Any, Any, Any]:
    """The immutable (kind, actor_json, authority_epoch, payload_json) an event id is bound to."""
    return row["kind"], row["actor_json"], row["authority_epoch"], row["payload_json"]


# The event bytes of every room in the store.
def _gateway_event_bytes(conn) -> int:
    return int(conn.execute(_SUM_EVENT_BYTES).fetchone()["total"] or 0)


def _insert_event(
    conn, room: Mapping[str, Any], room_id: str, seq: int, event_id: str, kind: str, actor_json: str,
    epoch: int, payload_json: str, now: float, *, allow_control: bool = False) -> int:
    """Capacity-check then INSERT one event at ``seq``; returns its accounted bytes."""
    event_bytes = _prepare_event(conn, room, event_id, kind, actor_json, payload_json, allow_control=allow_control)
    conn.execute(_INSERT_EVENT, (room_id, seq, event_id, kind, actor_json, epoch, payload_json, now))
    return event_bytes


def _append_locked(
    conn, *, room_id: str, event_id: str, kind: str, actor: dict[str, Any], actor_json: str,
    payload_json: str, authority_gateway_id: str, authority_epoch: int, expected_latest_seq: int | None,
    control_reserve: bool, now: float) -> dict[str, Any]:
    """Append one event in the caller's write: a stored ``event_id`` replays or conflicts, then the room, authority,
    cursor, attachment and capacity checks, the insert and the sequence advance."""
    existing = _load_event(conn, room_id, event_id)
    if existing is not None:
        if _event_content(existing) != (kind, actor_json, authority_epoch, payload_json):
            raise EventConflictError("event_id already exists with different content")
        return _event_from_row(existing, idempotent=True)
    room = _room_row(
        conn, """SELECT next_seq, event_bytes, authority_gateway_id, authority_epoch
            FROM hosted_rooms WHERE room_id = %s AND disbanded_at IS NULL""", (room_id,), room_id)
    _require_authority(room, authority_gateway_id, authority_epoch, "stale hosted room authority")
    seq = int(room["next_seq"])
    if expected_latest_seq is not None and seq - 1 != expected_latest_seq:
        raise EventCursorConflictError("room changed before event publication")
    if kind in {"message.user", "message.member"}:
        from .hosted_room_attachments import retain_message_attachments
        retain_message_attachments(conn, room_id=room_id, event_id=event_id,
                                   manifest=json.loads(payload_json).get("attachments", []), now=now)
    event_bytes = _insert_event(
        conn, room, room_id, seq, event_id, kind, actor_json, authority_epoch, payload_json, now,
        allow_control=kind in _CONTROL_EVENT_KINDS or (control_reserve and actor["kind"] == "gateway"))
    _fenced_update(conn, """UPDATE hosted_rooms SET next_seq=%s, event_bytes=event_bytes+%s, updated_at=%s
            WHERE room_id=%s AND next_seq=%s""", (seq + 1, event_bytes, now, room_id, seq),
        RuntimeError("hosted room sequence advance lost its write fence"))
    row = _reload(
        conn, f"SELECT {_EVENT_COLUMNS} FROM hosted_room_events WHERE room_id = %s AND seq = %s", (room_id, seq),
        "appended event could not be reloaded")
    return {**_event_from_row(row), "actor": actor}


def _prepare_event(
    conn, room: Mapping[str, Any], event_id: str, kind: str, actor_json: str, payload_json: str, *,
    allow_control: bool = False) -> int:
    """Size one pending event and enforce per-room and gateway capacity; returns its bytes."""
    additional_bytes = utf8_len(event_id, kind, actor_json, payload_json)
    count_reserve, byte_reserve = (CONTROL_EVENT_COUNT_RESERVE, CONTROL_EVENT_BYTE_RESERVE) if allow_control else (0, 0)
    gateway_byte_limit = MAX_GATEWAY_EVENT_BYTES + byte_reserve
    if int(room["next_seq"]) - 1 >= MAX_EVENTS_PER_ROOM + count_reserve:
        raise HostedRoomError("This Group Chat reached its history limit. Start a new Group Chat to continue.")
    if int(room["event_bytes"]) + additional_bytes > MAX_ROOM_EVENT_BYTES + byte_reserve:
        raise HostedRoomError("This Group Chat reached its storage limit. Start a new Group Chat to continue.")
    gateway_bytes = _gateway_event_bytes(conn)
    if gateway_bytes + additional_bytes > gateway_byte_limit:
        _prune_disbanded_rooms_locked(
            conn, now=None, max_gateway_event_bytes=max(0, gateway_byte_limit - additional_bytes))
        gateway_bytes = _gateway_event_bytes(conn)
    if gateway_bytes + additional_bytes > gateway_byte_limit:
        raise HostedRoomError("Group Chat storage is full on this host. Delete an old Group Chat and try again.")
    return additional_bytes


# --- retention -------------------------------------------------------------------
# Deleted in this order when a disbanded room's payload is purged.
_DEPENDENT_TABLES = (
    "hosted_room_policy_transcript_state", "hosted_room_policy_transcript", "hosted_room_policy_publications",
    "hosted_room_policy_watermarks", "hosted_room_policy_events", "hosted_room_policy_threads",
    "hosted_room_policy_cursors", "hosted_room_driver_tasks", "hosted_room_remote_runs",
    "hosted_room_links", "hosted_room_peer_reservations", "hosted_room_events")


def _room_ids(conn, sql: str, params: tuple[Any, ...]) -> list[str]:
    return [str(row["room_id"]) for row in conn.execute(sql, params).fetchall()]


def _retire_from_rooms(conn, where: str, params: tuple[Any, ...]) -> None:
    """The reference's ``_RETIRE_FROM_ROOMS`` on the PG dialect: the purged ids land in the tombstone table."""
    conn.execute(f"""INSERT INTO hosted_room_retired_ids (room_id, retired_at)
        SELECT room_id, disbanded_at FROM hosted_rooms WHERE {where}
        ON CONFLICT (room_id) DO NOTHING""", params)


# Purges disbanded rooms past retention or beyond the tombstone cap, then the oldest while the store's event bytes
# exceed ``max_gateway_event_bytes``; each purged id stays retired.
def _prune_disbanded_rooms_locked(
    conn, *, now: float | None, max_gateway_event_bytes: int | None = None) -> int:
    candidates: set[str] = set()
    if now is not None:
        candidates.update(_room_ids(
            conn, """SELECT room_id FROM hosted_rooms
                     WHERE disbanded_at IS NOT NULL AND disbanded_at <= %s""", (now - DISBANDED_ROOM_RETENTION_SECONDS,)))
    candidates.update(_room_ids(
        conn, """SELECT room_id FROM hosted_rooms WHERE disbanded_at IS NOT NULL
                ORDER BY disbanded_at DESC, room_id ASC OFFSET %s""", (MAX_DISBANDED_ROOM_TOMBSTONES,)))
    if max_gateway_event_bytes is not None:
        retained_bytes = _gateway_event_bytes(conn)
        if retained_bytes > max_gateway_event_bytes:
            rows = conn.execute("""SELECT room_id, event_bytes FROM hosted_rooms WHERE disbanded_at IS NOT NULL
                    ORDER BY disbanded_at ASC, room_id ASC""").fetchall()
            for row in rows:
                candidates.add(str(row["room_id"]))
                retained_bytes -= int(row["event_bytes"])
                if retained_bytes <= max_gateway_event_bytes:
                    break
    if not candidates:
        return 0
    room_ids = tuple(sorted(candidates))
    _retire_from_rooms(conn, "room_id = ANY(%s) AND disbanded_at IS NOT NULL", (list(room_ids),))
    for table in _DEPENDENT_TABLES:
        if table_exists(conn, table):
            conn.execute(f"DELETE FROM {table} WHERE room_id = ANY(%s)", (list(room_ids),))
    conn.execute("DELETE FROM hosted_rooms WHERE room_id = ANY(%s)", (list(room_ids),))
    return len(room_ids)


# --- rooms and events -------------------------------------------------------------
def _adopt_legacy_room(
    conn, existing: Mapping[str, Any], *, room_id: str, members_json: str, authority_gateway_id: str,
    now: float) -> dict[str, Any]:
    """Claim a 'legacy'-authority room for a real gateway with a fenced claim event."""
    target_epoch = int(existing["authority_epoch"]) + 1
    seq = int(existing["next_seq"])
    claim_bytes = _insert_event(
        conn, existing, room_id, seq, "system:authority-adopted", "authority.claimed",
        _system_actor_json("authority-control"), target_epoch,
        _claim_payload_json("legacy", authority_gateway_id, target_epoch), now, allow_control=True)
    _fenced_update(conn, """UPDATE hosted_rooms
            SET members_json=%s, authority_gateway_id=%s, authority_epoch=%s,
                next_seq=next_seq+1, revision=revision+1, event_bytes=event_bytes+%s, updated_at=%s
            WHERE room_id=%s AND authority_gateway_id='legacy' AND authority_epoch=%s AND next_seq=%s
            AND disbanded_at IS NULL""",
        (
            members_json, authority_gateway_id, target_epoch, claim_bytes, now, room_id,
            int(existing["authority_epoch"]), seq), AuthorityConflictError("legacy room adoption lost its fence"))
    existing = _reload(conn, _SELECT_ROOM, (room_id,), "adopted room could not be reloaded")
    claim_event = _reload(
        conn, _SELECT_EVENT, (room_id, "system:authority-adopted"), "legacy adoption event could not be reloaded")
    return {**_room_from_row(existing, idempotent=True), "adopted": True, "claim_event": _event_from_row(claim_event)}


def _disband_replay(conn, room_id: str, room: dict[str, Any] | None) -> dict[str, Any] | None:
    """Idempotent replay for a retired or already-disbanded room; None when the room is live."""
    if room is None:
        retired = conn.execute("SELECT retired_at FROM hosted_room_retired_ids WHERE room_id = %s", (room_id,)).fetchone()
        if retired is None:
            raise RoomNotFoundError("hosted room not found")
        return {
            "room_id": room_id, "disbanded_at": float(retired["retired_at"]), "idempotent": True,
            "history_expired": True}
    if room["disbanded_at"] is None:
        return None
    conn.execute(_INSERT_RETIRED, (room_id, float(room["disbanded_at"])))
    event = _load_event(conn, room_id, "system:room-disbanded")
    return {
        "room_id": room_id, "disbanded_at": float(room["disbanded_at"]), "idempotent": True,
        **({"event": _event_from_row(event, idempotent=True)} if event is not None else {})}


# The caller validates the arguments first.
def read_events_page(
    conn, *, room_id: str, since_seq: int, limit: int, include_disbanded: bool = False
) -> dict[str, Any]:
    """Read a monotonic room-log delta after ``since_seq`` on the caller's connection: the largest prefix whose
    serialized page fits the page budget."""
    room = _room_row(
        conn, """SELECT next_seq, authority_gateway_id, authority_epoch FROM hosted_rooms
            WHERE room_id = %s AND (disbanded_at IS NULL OR %s)""", (room_id, include_disbanded), room_id)
    latest_seq = int(room["next_seq"]) - 1
    authority = {"gateway_id": str(room["authority_gateway_id"]), "epoch": int(room["authority_epoch"])}
    if since_seq > latest_seq:
        raise HostedRoomError("since_seq is ahead of the hosted room log")
    rows = conn.execute(
        f"""WITH candidates AS (
               SELECT {_EVENT_COLUMNS},
                      SUM(
                          octet_length(event_id) +
                          octet_length(kind) +
                          octet_length(actor_json) +
                          octet_length(payload_json)
                      ) OVER (ORDER BY seq ASC) AS cumulative_bytes
                 FROM hosted_room_events
                WHERE room_id = %s AND seq > %s
                ORDER BY seq ASC LIMIT %s
           )
           SELECT {_EVENT_COLUMNS}
             FROM candidates
            WHERE cumulative_bytes <= %s
            ORDER BY seq ASC""", (room_id, since_seq, limit, MAX_LOG_PAGE_BYTES)).fetchall()
    events = [_event_from_row(row) for row in rows]

    def build_page(page_events: list[dict[str, Any]]) -> dict[str, Any]:
        cursor = page_events[-1]["seq"] if page_events else since_seq
        return {"events": page_events, "cursor": cursor, "latest_seq": latest_seq, "has_more": cursor < latest_seq,
                "authority": authority}

    def fits(page_events: list[dict[str, Any]]) -> bool:
        page_json = json.dumps(build_page(page_events), ensure_ascii=False, separators=(",", ":"))
        return utf8_len(page_json) <= MAX_LOG_PAGE_BYTES

    if events and not fits(events):
        # Binary-search the largest prefix whose serialized page fits the budget.
        low, high = 1, len(events)
        while low < high:
            middle = (low + high + 1) // 2
            low, high = (middle, high) if fits(events[:middle]) else (low, middle - 1)
        events = events[:low]
        if not fits(events):
            raise HostedRoomError("hosted room event exceeds replay page limit")
    return build_page(events)


# --- owners and the member check ----------------------------------------------------
def _claim_owner_locked(conn, epoch: int, room_id: str, actor: str, rule: OwnerRule | None, *,
                        create: bool) -> None:
    """Check (or, with ``create``, claim) room ownership inside the caller's write on the store's schema."""
    if (not isinstance(actor, str) or not actor
            or not isinstance(room_id, str) or len(room_id) > 128
            or not IDENTIFIER_RE.fullmatch(room_id)):
        raise AuthorityRefusal('invalid_params')
    _epoch(conn, epoch)
    key = OWNER_PREFIX + room_id
    row = conn.execute('SELECT value FROM state_meta WHERE key = %s', (key,)).fetchone()
    if row is None and create:
        historical = conn.execute(
            'SELECT 1 FROM hosted_rooms WHERE room_id = %s UNION ALL '
            'SELECT 1 FROM hosted_room_retired_ids WHERE room_id = %s',
            (room_id, room_id)).fetchone()
        if historical is None:
            conn.execute('INSERT INTO state_meta(key, value) VALUES (%s, %s)', (key, actor))
            return
    if row is None or not same_owner(row["value"], actor, rule):
        raise AuthorityRefusal('permission_denied')


# The ``state_meta`` key of one member's recorded homes.
def _member_homes_key(room_id: str, member_id: str) -> str:
    return MEMBER_HOMES_PREFIX + json.dumps([room_id, member_id], separators=(',', ':'))


# The ``state_meta`` key of the home one attempt of a member was admitted at.
def _attempt_home_key(room_id: str, member_id: str, task_id: str, generation: int) -> str:
    return MEMBER_HOMES_PREFIX + json.dumps([room_id, member_id, task_id, int(generation)], separators=(',', ':'))


_LIVE_ATTEMPT = frozenset({'running', 'stopping', 'deferred', 'indeterminate'})
_STOPPING = frozenset({'stopping', 'deferred'})


def _attempt_homes(conn, room_id: str, member_id: str,
                   statuses: frozenset[str] = _LIVE_ATTEMPT) -> dict[str, tuple[str, bool]]:
    """``{key: (home, live)}`` for every attempt home recorded for this member; live while its task row is still
    that attempt (same generation) in one of *statuses*. A store without driver tasks holds no live attempt."""
    prefix, found = MEMBER_HOMES_PREFIX + json.dumps([room_id, member_id], separators=(',', ':'))[:-1] + ',', {}
    tasks = table_exists(conn, 'hosted_room_driver_tasks')
    for row in conn.execute('SELECT key, value FROM state_meta WHERE left(key, char_length(%s)) = %s',
                            (prefix, prefix)).fetchall():
        task_id, generation = json.loads(row["key"][len(MEMBER_HOMES_PREFIX):])[2:]
        task_row = conn.execute('SELECT status, execution_generation FROM hosted_room_driver_tasks '
                                'WHERE room_id = %s AND task_id = %s', (room_id, task_id)).fetchone() if tasks else None
        found[row["key"]] = (row["value"],
                             task_row is not None and task_row["status"] in statuses
                             and task_row["execution_generation"] == generation)
    return found


# Drops a disbanded room's recorded member homes in its tombstone's transaction; a store without ``state_meta`` has none.
def _forget_room_member_homes_locked(conn, room_id: str) -> None:
    if table_exists(conn, "state_meta"):
        prefix = MEMBER_HOMES_PREFIX + json.dumps([room_id], separators=(',', ':'))[:-1] + ','
        conn.execute('DELETE FROM state_meta WHERE left(key, char_length(%s)) = %s', (prefix, prefix))


# The caller runs the epoch fence first.
def _member_authorized_locked(conn, guard: HostedMemberGuard, *, operation: str) -> bool:
    """The hosted member check on the caller's store connection; False on any mismatch."""
    from .hosted_room_driver import _task_from_row

    owned = conn.execute('SELECT value FROM state_meta WHERE key = %s',
                         (OWNER_PREFIX + guard.room_id,)).fetchone()
    if owned is None or owned["value"] != guard.owner:
        return False
    stored = conn.execute('SELECT * FROM hosted_rooms WHERE room_id = %s',
                          (guard.room_id,)).fetchone()
    if stored is None or stored['disbanded_at'] is not None:
        return False
    room = _room_from_row(stored)
    if (room['authority_gateway_id'], room['authority_epoch']) != (guard.gateway_id, guard.authority_epoch):
        return False
    members = room['members']
    if not any(m.get('member_id') == guard.member_id and m.get('profile') == guard.profile
               and (m.get('target') is None or (
                   isinstance(m.get('target'), dict)
                   and m['target'].get('kind', 'local') == 'local'))
               for m in members):
        return False
    if not guard.home_current:
        return False
    if guard.task is not None:
        stored = conn.execute('SELECT * FROM hosted_room_driver_tasks WHERE room_id = %s AND task_id = %s',
                              (guard.room_id, guard.task.task_id)).fetchone()
        if stored is None:
            return False
        current = _task_from_row(stored)
        return (current['identity'] == guard.task and current['execution_generation'] == guard.generation
                and current['payload'].get('target_profile') == guard.profile
                and current['payload'].get('target_member_id', guard.profile) == guard.member_id
                and current['status'] in ({'running'} if operation in {'submit', 'execute'}
                                          else {'running', 'stopping'}))
    return True


# --- the hand-off of a standalone backend's rooms -----------------------------------
# The rooms a shared store handed to the gateway store: never offered again. Recorded by the first hand-off.
def _ensure_handoff_table(conn) -> None:
    conn.execute(_HANDOFF_DDL)


def _fenced_room_ids(conn, handoff_gateway_id: str) -> frozenset[str]:
    """Fenced live rooms not yet recorded as handed over."""
    _ensure_handoff_table(conn)
    return frozenset(str(row["room_id"]) for row in conn.execute(
        f"""SELECT room_id FROM hosted_rooms WHERE authority_gateway_id = %s AND disbanded_at IS NULL
            AND room_id NOT IN (SELECT room_id FROM {_HANDOFF_TABLE})""", (handoff_gateway_id,)).fetchall())


def _record_handoff(conn, room_ids: Collection[str], now: float) -> None:
    _ensure_handoff_table(conn)
    # ``executemany`` over the connection's cursor: one statement per parameter set.
    conn.cursor().executemany(f"INSERT INTO {_HANDOFF_TABLE} (room_id, handed_at) VALUES (%s, %s) ON CONFLICT (room_id) DO NOTHING",
                              [(room_id, now) for room_id in sorted(room_ids)])


def _handed_room_ids(conn, handoff_gateway_id: str) -> frozenset[str]:
    """Every room handed to the gateway store, recorded or still only fenced."""
    return frozenset(str(row["room_id"]) for row in conn.execute(
        "SELECT room_id FROM hosted_rooms WHERE authority_gateway_id = %s", (handoff_gateway_id,)).fetchall())


def _claim(conn, room_id: str, local_id: str, handoff_id: str, owner: str, now: float) -> None:
    """Claim a copied room for ``local_id`` at the next epoch and own it; an id that was not copied is left alone."""
    row = conn.execute("SELECT authority_epoch, next_seq FROM hosted_rooms WHERE room_id = %s", (room_id,)).fetchone()
    if row is None:
        return
    epoch = int(row["authority_epoch"]) + 1
    _append_control_event(conn, room_id, int(row["next_seq"]), epoch, _control_event("claimed", epoch, {
        "previous_gateway_id": handoff_id, "authority_gateway_id": local_id, "authority_epoch": epoch}), now)
    conn.execute("""UPDATE hosted_rooms SET authority_gateway_id=%s, authority_epoch=%s, next_seq=next_seq+1,
                 revision=revision+1, updated_at=%s WHERE room_id=%s""", (local_id, epoch, now, room_id))
    conn.execute("INSERT INTO state_meta(key, value) VALUES (%s, %s) ON CONFLICT (key) DO NOTHING",
                 (OWNER_PREFIX + room_id, owner))


def _table_columns(conn, table: str) -> list[str]:
    """The table's columns in declaration order (both hosted schemas carry the same DDL)."""
    return [str(row["column_name"]) for row in conn.execute(
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_schema = current_schema() AND table_name = %s ORDER BY ordinal_position", (table,)).fetchall()]


def _copy_rows(conn, shared_schema: str, *, only: frozenset[str]) -> int:
    """Copy the rooms this store has never known from the backend's shared schema (the reference's
    cross-file ``_copy_rows`` twin); returns rooms copied. A room id present in both stores is skipped as a
    unit; the room-scoped copy is a plain INSERT, so a refused row aborts the whole hand-off rather than
    being dropped in silence. The liveness tables a fresh driver claims instead stay out."""
    if not only:
        return 0
    copied = [str(row["room_id"]) for row in conn.execute(
        f'SELECT room_id FROM "{shared_schema}".hosted_rooms WHERE room_id = ANY(%s)', (sorted(only),)).fetchall()]
    if not copied:
        return 0
    ids = list(copied)
    for table in _COPY_TABLES:
        columns = _table_columns(conn, table)
        if "room_id" not in columns:
            continue
        # Quote the identifiers: the link and owed-revoke tables carry the reserved-word ``grant`` column.
        quoted = ", ".join(f'"{column}"' for column in columns)
        conn.execute(
            f'INSERT INTO "{table}" ({quoted}) SELECT {quoted} '
            f'FROM "{shared_schema}"."{table}" WHERE room_id = ANY(%s)', (ids,))
    return len(copied)


# The gateway store's claim of the hand-off's fenced rooms: copy from the shared schema, then claim each.
def _import_fenced_rooms(conn, shared_schema: str, *, epoch: int | None, room_ids: Collection[str],
                         owner: str, local_gateway_id: str, handoff_gateway_id: str, now: float) -> int:
    """Copy the fenced rooms this store has never known from the shared schema, then claim them; returns rooms copied."""
    if epoch is not None:
        _epoch(conn, epoch)
    # Every id this store has ever known: a disbanded room's row is pruned later, its id stays.
    known = {str(row["room_id"]) for row in conn.execute(
        "SELECT room_id FROM hosted_rooms UNION SELECT room_id FROM hosted_room_retired_ids").fetchall()}
    missing = frozenset(set(room_ids) - known)
    copied = _copy_rows(conn, shared_schema, only=missing)
    for room_id in sorted(missing):
        _claim(conn, room_id, local_gateway_id, handoff_gateway_id, owner, now)
    return copied


class PgHostedRoomRepository:
    """Hosted room owners, transport bindings, member sessions, rooms and the room event log."""

    def __init__(self, store: PostgresStore) -> None:
        self._store = store

    # ── writes on the store's schema ────────────────────────────────────────

    def claim_owner(self, *, epoch: int, room_id: str, actor: str, owner: OwnerRule | None) -> None:
        with translated():
            with self._store.tx() as conn:
                _claim_owner_locked(conn, epoch, room_id, actor, owner, create=False)

    # Binds the session's transport once; a different binding for the same session is refused.
    def bind_transport(self, *, epoch: int, session_id: str, binding: str) -> None:
        key = TRANSPORT_PREFIX + session_id

        def persist(conn) -> None:
            _epoch(conn, epoch)
            old = conn.execute('SELECT value FROM state_meta WHERE key = %s', (key,)).fetchone()
            if old is not None and old["value"] != binding:
                raise AuthorityRefusal('permission_denied')
            conn.execute('INSERT INTO state_meta(key, value) VALUES (%s, %s) ON CONFLICT (key) DO NOTHING', (key, binding))
        with translated():
            with self._store.tx() as conn:
                persist(conn)

    # Titles and hides a room session; a title another session holds raises ``TitleConflict``. The store's schema
    # carries no unique title index (the reference's index refusal), so the holder check runs before the update.
    def label_room_session(self, *, epoch: int, session_id: str, title: str) -> None:
        def label(conn) -> None:
            _epoch(conn, epoch)
            holder = conn.execute('SELECT id FROM sessions WHERE title = %s AND id != %s',
                                 (title, session_id)).fetchone()
            if holder is not None:
                raise TitleConflict(title, SessionId(holder["id"]))
            conn.execute('UPDATE sessions SET title = %s, hidden = 1 WHERE id = %s', (title, session_id))
        with translated():
            with self._store.tx() as conn:
                label(conn)

    # Creates the hidden ``bot_room`` session of a room member, or accepts the one there with the same title.
    def ensure_room_member_session(self, *, session_id: str, title: str) -> str:
        def atomic(conn) -> str:
            row = conn.execute("SELECT id, title, source FROM sessions WHERE id = %s", (session_id,)).fetchone()
            if row is not None:
                if row["title"] != title or row["source"] != "bot_room":
                    raise HostedRoomRefusal(_ROOM_SESSION_CONFLICT)
                return session_id
            clean_title = sanitize_title(title)
            conflict = conn.execute(
                "SELECT id FROM sessions WHERE title = %s AND id != %s", (clean_title, session_id)).fetchone()
            if conflict:
                raise HostedRoomRefusal(_ROOM_TITLE_CONFLICT)
            conn.execute(
                "INSERT INTO sessions(id, source, title, hidden, started_at) VALUES (%s, 'bot_room', %s, 1, %s)",
                (session_id, clean_title, time.time()))
            return session_id
        with translated():
            with self._store.tx() as conn:
                return atomic(conn)

    # Appends ``home`` to the member's recorded homes unless it is there.
    def record_member_home(self, *, room_id: str, member_id: str, home: str) -> None:
        key = _member_homes_key(room_id, member_id)

        def write(conn) -> None:
            row = conn.execute('SELECT value FROM state_meta WHERE key = %s', (key,)).fetchone()
            homes = json.loads(row["value"]) if row is not None else []
            if home not in homes:
                conn.execute('INSERT INTO state_meta(key, value) VALUES (%s, %s) ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value',
                             (key, json.dumps([*homes, home])))
        with translated():
            with self._store.tx() as conn:
                write(conn)

    # Drops this member's attempt homes whose attempt is over, then records ``home`` unless one is (first wins).
    def record_attempt_home(self, *, room_id: str, member_id: str, task_id: str, generation: int, home: str) -> None:
        key = _attempt_home_key(room_id, member_id, task_id, generation)

        def write(conn) -> None:
            for old, (_home, live) in _attempt_homes(conn, room_id, member_id).items():
                if not live and old != key:
                    conn.execute('DELETE FROM state_meta WHERE key = %s', (old,))
            conn.execute('INSERT INTO state_meta(key, value) VALUES (%s, %s) ON CONFLICT (key) DO NOTHING', (key, home))
        with translated():
            with self._store.tx() as conn:
                write(conn)

    # ── writes on the placement's schema ────────────────────────────────────

    # Creates a room or replays an identical one, adopting a room whose authority is ``legacy``; with ``owner``,
    # the owner claim commits in the same transaction.
    def create_room(self, *, placement: HostedPlacement, room_id: str, name: str, members: Sequence[Mapping[str, JsonValue]], authority_gateway_id: str, owner: HostedOwnerClaim | None = None, now: float | None = None) -> Mapping[str, JsonValue]:
        room_id = _room_id(room_id)
        name = _validate_room_name(name)
        normalized_members, members_json = _validate_members(members)
        authority_gateway_id = _actor_id(authority_gateway_id, "authority_gateway_id")
        now = _now(now)
        if owner is not None and placement != "profile":
            raise ValueError("an owner claim requires placement='profile'")

        def write(conn) -> dict[str, Any]:
            if owner is not None:
                _claim_owner_locked(conn, owner.epoch, room_id, owner.actor, owner.rule, create=True)
            if _is_retired(conn, room_id):
                raise RoomConflictError("room_id belongs to a disbanded room")
            existing = conn.execute(_SELECT_ROOM_WITH_BYTES, (room_id,)).fetchone()
            if existing is not None:
                if existing["disbanded_at"] is not None:
                    raise RoomConflictError("room_id belongs to a disbanded room")
                legacy_adoption = (existing["authority_gateway_id"] == "legacy" and authority_gateway_id != "legacy")
                members_match = existing["members_json"] == members_json or (
                    legacy_adoption and _legacy_members_match(existing["members_json"], normalized_members))
                if existing["name"] != name or not members_match:
                    raise RoomConflictError("room_id already exists with different state")
                if legacy_adoption:
                    return _adopt_legacy_room(
                        conn, existing, room_id=room_id, members_json=members_json,
                        authority_gateway_id=authority_gateway_id, now=now)
                if existing["authority_gateway_id"] != authority_gateway_id:
                    raise RoomConflictError("room_id already belongs to a different authority")
                return _room_from_row(existing, idempotent=True)
            active = conn.execute("SELECT COUNT(*) AS count FROM hosted_rooms WHERE disbanded_at IS NULL").fetchone()["count"]
            if int(active) >= MAX_ACTIVE_ROOMS:
                raise HostedRoomError("This host has too many active Group Chats. Delete one and try again.")
            conn.execute(
                f"""INSERT INTO hosted_rooms ({_ROOM_COLUMNS_WITH_BYTES})
                    VALUES (%s, %s, %s, %s, 1, 1, 0, 1, %s, %s, NULL)""",
                (room_id, name, members_json, authority_gateway_id, now, now))
            row = _reload(
                conn, """SELECT room_id, name, members_json, authority_gateway_id, authority_epoch, revision,
                    created_at, updated_at FROM hosted_rooms WHERE room_id = %s""", (room_id,),
                "created room could not be reloaded")
            return {**_room_from_row(row), "members": normalized_members}
        with translated():
            with placement_write(self._store, placement) as conn:
                return write(conn)

    # Renames the room and logs ``room.renamed``; the same event id with the same name replays.
    def rename_room(self, *, placement: HostedPlacement, room_id: str, event_id: str, name: str, now: float | None = None) -> Mapping[str, JsonValue]:
        now = _now(now)
        actor_json = _system_actor_json("room-control")
        payload_json = _payload_json({"name": name})

        def write(conn) -> dict[str, Any]:
            room = _room_row(conn, _SELECT_ROOM_WITH_BYTES, (room_id,), room_id)
            if room["disbanded_at"] is not None:
                raise RoomNotFoundError("hosted room not found")
            existing = _load_event(conn, room_id, event_id)
            if existing is not None:
                if existing["kind"] != "room.renamed" or existing["payload_json"] != payload_json:
                    raise EventConflictError("event_id already exists with different immutable content")
                return {**_room_from_row(room, idempotent=True), "event": _event_from_row(existing, idempotent=True)}
            seq = int(room["next_seq"])
            event_bytes = _prepare_event(conn, room, event_id, "room.renamed", actor_json, payload_json)
            # Rename updates the room row before inserting its event (order is load-bearing).
            conn.execute("""UPDATE hosted_rooms
                    SET name=%s, next_seq=%s, event_bytes=event_bytes+%s, revision=revision+1, updated_at=%s
                    WHERE room_id=%s""", (name, seq + 1, event_bytes, now, room_id))
            conn.execute(_INSERT_EVENT, (
                room_id, seq, event_id, "room.renamed", actor_json, int(room["authority_epoch"]), payload_json, now))
            updated = conn.execute(f"SELECT {_ROOM_COLUMNS} FROM hosted_rooms WHERE room_id = %s", (room_id,)).fetchone()
            return {**_room_from_row(updated), "event": _event_from_row(_load_event(conn, room_id, event_id))}
        with translated():
            with placement_write(self._store, placement) as conn:
                return write(conn)

    # The caller validates the event first. Appends it under the room's authority; the same id and content replays.
    def append_event(self, *, placement: HostedPlacement, room_id: str, event_id: str, kind: str, actor: Mapping[str, str], payload: Mapping[str, JsonValue], authority_gateway_id: str, authority_epoch: int, expected_latest_seq: int | None = None, control_reserve: bool = False, now: float | None = None) -> Mapping[str, JsonValue]:
        normalized_actor = dict(actor)
        actor_json = _actor_json(normalized_actor)
        payload_json = _payload_json(dict(payload))
        now = _now(now)
        with translated():
            with placement_write(self._store, placement) as conn:
                return _append_locked(
                    conn, room_id=room_id, event_id=event_id, kind=kind, actor=normalized_actor, actor_json=actor_json,
                    payload_json=payload_json, authority_gateway_id=authority_gateway_id, authority_epoch=authority_epoch,
                    expected_latest_seq=expected_latest_seq, control_reserve=control_reserve, now=now)

    # The caller validates the room, event id and authority first; the omitted routing, the record's rules, the lead
    # check and the retry comparison all resolve inside the append's write.
    def save_room_settings(self, *, placement: HostedPlacement, room_id: str, event_id: str, purpose: str, notes: str, routing: str | None, lead_member_id: str | None, seated_member_ids: Collection[str], authority_gateway_id: str, authority_epoch: int, now: float | None = None) -> Mapping[str, JsonValue]:
        actor = dict(_CONTEXT_ACTOR)
        actor_json = _actor_json(actor)
        seated = frozenset(seated_member_ids)
        now = _now(now)

        def write(conn) -> dict[str, Any]:
            first = _load_event(conn, room_id, event_id)
            original = (json.loads(first["payload_json"])
                        if first is not None and first["kind"] == CONTEXT_EVENT_KIND else None)
            current = ({**_EMPTY_CONTEXT, **original} if original is not None
                       else context_from_payload(_latest_payload(conn, room_id, CONTEXT_EVENT_KIND, None)))
            context = validate_context(
                purpose, notes, current["routing"] if routing is None else routing,
                current["lead_member_id"] if lead_member_id is None else lead_member_id)
            if first is None and context["routing"] == "lead" and context["lead_member_id"] not in seated:
                raise HostedRoomError("the lead must be a member of this Group Chat")
            payload = context if original is None else retry_fields(context, original)
            event = _append_locked(
                conn, room_id=room_id, event_id=event_id, kind=CONTEXT_EVENT_KIND, actor=actor, actor_json=actor_json,
                payload_json=_payload_json(payload), authority_gateway_id=authority_gateway_id,
                authority_epoch=authority_epoch, expected_latest_seq=None, control_reserve=False, now=now)
            return {"context": context, "event": event}
        with translated():
            with placement_write(self._store, placement) as conn:
                return write(conn)

    # The caller validates the arguments first. Disbands the room under its authority once no unexpired held grant
    # remains, retires its id, forgets its member homes and prunes; a room already disbanded or retired replays.
    def disband_room(self, *, placement: HostedPlacement, room_id: str, expected_gateway_id: str, expected_epoch: int, now: float | None = None) -> Mapping[str, JsonValue]:
        now = _now(now)

        def write(conn) -> dict[str, Any]:
            room = conn.execute("""SELECT authority_gateway_id, authority_epoch, next_seq, event_bytes, disbanded_at
                    FROM hosted_rooms WHERE room_id = %s""", (room_id,)).fetchone()
            if (replay := _disband_replay(conn, room_id, room)) is not None:
                return replay
            _require_authority(room, expected_gateway_id, expected_epoch, "stale hosted room authority")
            if conn.execute("""SELECT 1 FROM hosted_room_owed_revokes
                    WHERE room_id = %s AND held = 1 AND expires_at > %s LIMIT 1""", (room_id, now)).fetchone():
                raise RoomHeldGrantsPendingError("Group chat grants still need revoking; retry Disband.")
            conn.execute("DELETE FROM hosted_room_owed_revokes WHERE room_id = %s AND held = 1 AND expires_at <= %s",
                         (room_id, now))
            disband_bytes = _insert_event(
                conn, room, room_id, int(room["next_seq"]), "system:room-disbanded", "room.disbanded",
                _system_actor_json("room-control"), int(room["authority_epoch"]), _payload_json({"room_id": room_id}),
                now, allow_control=True)
            _fenced_update(conn, """UPDATE hosted_rooms
                    SET disbanded_at=%s, updated_at=%s, revision=revision+1,
                        next_seq=next_seq+1, event_bytes=event_bytes+%s
                    WHERE room_id=%s AND disbanded_at IS NULL AND authority_gateway_id=%s
                    AND authority_epoch=%s""",
                (now, now, disband_bytes, room_id, expected_gateway_id, expected_epoch),
                RoomConflictError("hosted room disband lost its fence"))
            conn.execute(_INSERT_RETIRED, (room_id, now))
            _forget_room_member_homes_locked(conn, room_id)
            event = _reload(
                conn, _SELECT_EVENT, (room_id, "system:room-disbanded"), "room disband event could not be reloaded")
            _prune_disbanded_rooms_locked(conn, now=now, max_gateway_event_bytes=MAX_GATEWAY_EVENT_BYTES)
            return {"room_id": room_id, "disbanded_at": now, "idempotent": False, "event": _event_from_row(event)}
        with translated():
            with placement_write(self._store, placement) as conn:
                return write(conn)

    # Inserts the claim, or takes over an expired one; true when ``owner`` holds it after the write.
    def claim_router_call(self, *, placement: HostedPlacement, room_id: str, discussion_event_id: str, owner: str, expires_at: float, now: float | None = None) -> bool:
        timestamp = _now(now)

        def write(conn) -> bool:
            conn.execute(_ROUTER_CLAIMS_DDL)
            row = conn.execute(
                """INSERT INTO hosted_room_router_claims(room_id, discussion_event_id, owner, expires_at)
                   VALUES (%s, %s, %s, %s) ON CONFLICT(room_id, discussion_event_id) DO UPDATE SET
                   owner = EXCLUDED.owner, expires_at = EXCLUDED.expires_at
                   WHERE hosted_room_router_claims.expires_at <= %s
                   RETURNING owner""",
                (room_id, discussion_event_id, owner, float(expires_at), timestamp)).fetchone()
            return row is not None and row["owner"] == owner
        with translated():
            with placement_write(self._store, placement) as conn:
                return write(conn)

    # Deletes the claim while ``owner`` holds it.
    def release_router_call(self, *, placement: HostedPlacement, room_id: str, discussion_event_id: str, owner: str) -> None:
        def write(conn) -> None:
            conn.execute(_ROUTER_CLAIMS_DDL)
            conn.execute("DELETE FROM hosted_room_router_claims WHERE room_id = %s AND discussion_event_id = %s AND owner = %s",
                         (room_id, discussion_event_id, owner))
        with translated():
            with placement_write(self._store, placement) as conn:
                write(conn)

    # Purges disbanded rooms past retention or beyond the tombstone cap.
    def prune_disbanded_rooms(self, *, placement: HostedPlacement, now: float | None = None) -> int:
        timestamp = _now(now)
        with translated():
            with placement_write(self._store, placement) as conn:
                return _prune_disbanded_rooms_locked(conn, now=timestamp)

    # ── reads on the store's schema ─────────────────────────────────────────

    # The room's recorded owner, or None.
    def owner(self, room_id: str) -> str | None:
        with translated():
            row = self._store.one('SELECT value FROM state_meta WHERE key = %s', OWNER_PREFIX + room_id)
        return None if row is None else row["value"]

    # The ids of every room with a recorded owner.
    def owned_rooms(self) -> frozenset[str]:
        with translated():
            rows = self._store.all('SELECT key FROM state_meta WHERE key LIKE %s', OWNER_PREFIX + '%')
        return frozenset(r["key"][len(OWNER_PREFIX):] for r in rows)

    # Probes the owner key with a short lock wait and no schema step.
    def probe_room_owner(self, room_id: str) -> bool:
        return probe(self._store, "profile", "state_meta",
                    "SELECT 1 FROM state_meta WHERE key = %s LIMIT 1",
                    (OWNER_PREFIX + room_id,), "native room ownership is temporarily unavailable")

    # The member's recorded homes in the order they were first recorded.
    def member_homes(self, *, room_id: str, member_id: str) -> tuple[str, ...]:
        with translated():
            row = self._store.one('SELECT value FROM state_meta WHERE key = %s',
                                  _member_homes_key(room_id, member_id))
        return tuple(json.loads(row["value"])) if row is not None else ()

    # The home recorded for one attempt, or None.
    def attempt_home(self, *, room_id: str, member_id: str, task_id: str, generation: int) -> str | None:
        with translated():
            row = self._store.one('SELECT value FROM state_meta WHERE key = %s',
                                  _attempt_home_key(room_id, member_id, task_id, generation))
        return None if row is None else row["value"]

    # The homes of this member's attempts still stopping or deferred.
    def stopping_attempt_homes(self, *, room_id: str, member_id: str) -> tuple[str, ...]:
        with translated():
            with placement_read(self._store, "profile") as conn:
                found = _attempt_homes(conn, room_id, member_id, _STOPPING)
        return tuple(dict.fromkeys(home for home, live in found.values() if live))

    # The session's transport binding, or None.
    def transport_binding(self, session_id: str) -> str | None:
        with translated():
            stored = self._store.one('SELECT value FROM state_meta WHERE key = %s',
                                     TRANSPORT_PREFIX + session_id)
        return None if stored is None else stored["value"]

    # The member check after the epoch fence, on one read of the store's schema.
    def member_authorized(self, guard: HostedMemberGuard, *, epoch: int, operation: str) -> bool:
        with translated():
            with placement_read(self._store, "profile") as conn:
                _epoch(conn, epoch)
                return _member_authorized_locked(conn, guard, operation=operation)

    # ── reads on the placement's schema ─────────────────────────────────────

    # The caller validates the arguments first; rooms by most recent update.
    def list_rooms(self, *, placement: HostedPlacement, include_disbanded: bool = False, limit: int = 500, offset: int = 0) -> list[Mapping[str, JsonValue]]:
        rows = placement_all(
            self._store, placement,
            f"""SELECT {_ROOM_COLUMNS} FROM hosted_rooms WHERE disbanded_at IS NULL OR %s
                ORDER BY updated_at DESC, room_id ASC LIMIT %s OFFSET %s""",
            include_disbanded, limit, offset)
        return [_room_from_row(row) for row in rows]

    # The caller validates the arguments first; the room view with the claim event of its current epoch, if any.
    def room_state(self, *, placement: HostedPlacement, room_id: str, include_disbanded: bool = False) -> Mapping[str, JsonValue]:
        with translated():
            row = placement_one(
                self._store, placement,
                f"""SELECT {_ROOM_COLUMNS} FROM hosted_rooms WHERE room_id = %s AND (disbanded_at IS NULL
                    OR %s)""", room_id, include_disbanded)
            if row is None:
                # The room row, if retained, or the tombstone: the precise not-found/expired error.
                retained = placement_one(
                    self._store, placement, "SELECT 1 AS one FROM hosted_rooms WHERE room_id = %s", room_id)
                if retained is None and placement_one(
                        self._store, placement,
                        "SELECT retired_at FROM hosted_room_retired_ids WHERE room_id = %s", room_id) is not None:
                    raise RoomHistoryExpiredError("Group Chat history expired; room_id remains permanently retired")
                raise RoomNotFoundError("hosted room not found")
            claim_row = placement_one(
                self._store, placement,
                f"""SELECT {_EVENT_COLUMNS} FROM hosted_room_events WHERE room_id = %s
                    AND kind = 'authority.claimed' AND authority_epoch = %s ORDER BY seq DESC LIMIT 1""",
                room_id, int(row["authority_epoch"]))
        return {**_room_from_row(row), **({"authority_claim": _event_from_row(claim_row)} if claim_row is not None else {})}

    def read_events(self, *, placement: HostedPlacement, room_id: str, since_seq: int = 0, limit: int = 100, include_disbanded: bool = False) -> Mapping[str, JsonValue]:
        with translated():
            with placement_read(self._store, placement) as conn:
                return read_events_page(
                    conn, room_id=room_id, since_seq=since_seq, limit=limit, include_disbanded=include_disbanded)

    # The caller validates the arguments first; probes for a live room with a short lock wait and no schema step.
    def probe_hosted_room(self, *, placement: HostedPlacement, room_id: str) -> bool:
        return probe(
            self._store, placement, "hosted_rooms",
            "SELECT 1 FROM hosted_rooms WHERE room_id = %s AND disbanded_at IS NULL LIMIT 1",
            (room_id,), "hosted room ownership is temporarily unavailable")

    # The caller validates the arguments first; one read of the log on the kind index.
    def latest_event_payload(self, *, placement: HostedPlacement, room_id: str, kind: str, before_seq: int | None = None) -> Mapping[str, JsonValue] | None:
        if before_seq is None:
            sql = "SELECT payload_json FROM hosted_room_events WHERE room_id = %s AND kind = %s ORDER BY seq DESC LIMIT 1"
            params: tuple[Any, ...] = (room_id, kind)
        else:
            sql = "SELECT payload_json FROM hosted_room_events WHERE room_id = %s AND kind = %s AND seq < %s ORDER BY seq DESC LIMIT 1"
            params = (room_id, kind, int(before_seq))
        with translated():
            row = placement_one(self._store, placement, sql, *params)
        return None if row is None else json.loads(row["payload_json"])

    # The caller validates the arguments first; the stored event, or None.
    def event_by_id(self, *, placement: HostedPlacement, room_id: str, event_id: str) -> Mapping[str, JsonValue] | None:
        with translated():
            row = placement_one(self._store, placement, _SELECT_EVENT, room_id, event_id)
        return None if row is None else _event_from_row(row)

    # ── the hand-off: the shared store, and the copy into the store's schema ─

    # A write, as main's: it creates the hand-off table before it reads.
    def fenced_room_ids(self, *, handoff_gateway_id: str) -> frozenset[str]:
        with translated():
            with placement_write(self._store, "shared") as conn:
                return _fenced_room_ids(conn, handoff_gateway_id)

    def record_room_handoff(self, *, room_ids: Collection[str], now: float | None = None) -> None:
        timestamp = _now(now)
        with translated():
            with placement_write(self._store, "shared") as conn:
                _record_handoff(conn, room_ids, timestamp)

    def handed_room_ids(self, *, handoff_gateway_id: str) -> frozenset[str]:
        with translated():
            rows = placement_all(self._store, "shared",
                                 "SELECT room_id FROM hosted_rooms WHERE authority_gateway_id = %s",
                                 handoff_gateway_id)
        return frozenset(str(row["room_id"]) for row in rows)

    def import_fenced_rooms(self, *, epoch: int | None, room_ids: Collection[str], owner: str, local_gateway_id: str, handoff_gateway_id: str, now: float | None = None) -> int:
        timestamp = _now(now)
        with translated():
            with placement_write(self._store, "profile") as conn:
                return _import_fenced_rooms(
                    conn, self._store.shared_schema, epoch=epoch, room_ids=room_ids, owner=owner,
                    local_gateway_id=local_gateway_id, handoff_gateway_id=handoff_gateway_id, now=timestamp)


def _register() -> None:
    register_repository("hosted_rooms", lambda store: PgHostedRoomRepository(store))


_register()

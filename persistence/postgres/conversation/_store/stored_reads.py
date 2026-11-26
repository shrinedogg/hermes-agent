"""The store reads behind ``session.transcript`` on PostgreSQL (the SQLite ``stored_reads`` port):
the lineage display set with its durable keys, one keyset page of a segment, the compaction handoffs a
page shows as rows (and those a display row before the page already holds), and the state check a
poller's cursor holds. The SQLite ``json_each`` id lists become ``= ANY(%s)`` arrays, the index hints
drop (the PG planner is cost-based), and ``active`` is ``NOT NULL`` so the legacy nullable-active copy
branch has no PG twin."""

from __future__ import annotations

from typing import Any, Iterable, Optional

from .common import (
    DISPLAY_VISIBLE_SQL, _DISPLAY_INDEX_MISSING_SQL, _display_rows_sql, _in_placeholders,
    not_a_later_tool_copy_sql, _payload_bytes, _encode_content, _row_to_message_dict,
)
from .reads import (
    _dedupe_display_generations, _fetch_conversation_rows, _rows_to_conversation,
)

__all__ = [
    "display_generations", "segment_page_messages", "transcript_cursor_state",
    "transcript_display_index_complete", "transcript_handoff_rows", "transcript_handoffs_shown_before",
    "transcript_tool_calls_before",
]

# A row that can hold a compaction handoff (the compressor's prefixes, the merged delimiter, or the
# stored flag); the bytea body is scanned as UTF-8 text.
_HANDOFF_PREDICATE = (
    "g.role IN ('user', 'assistant') AND (g._compressed_summary = 1"
    " OR hermes_nul_safe_text(g.content) LIKE '[CONTEXT COMPACTION%%'"
    " OR hermes_nul_safe_text(g.content) LIKE '[CONTEXT SUMMARY]%%')")

# The shown rows: their id/uid plus whether they can hold a handoff themselves (every other shown row is
# read for its id, uid and flag only, so an ordinary page never loads its bodies a second time).
_SHOWN_ROWS_SQL = (
    "SELECT g.id AS _shown_id, g.message_uid AS _shown_uid, "
    f"(g.session_id = ANY(%s) AND {_HANDOFF_PREDICATE}) AS _holds "
    "FROM messages g WHERE g.id = ANY(%s)")

# The earlier stored copies of the shown rows' uids that can hold a handoff: the live ones first, then
# the archived ones (``active`` is ``NOT NULL`` in the PG store, so the SQLite legacy NULL branch is
# gone). Each copy is paired with the shown rows after it in Python.
_HANDOFF_COPIES_SQL = (
    "SELECT g.* FROM messages g WHERE g.session_id = ANY(%s) AND g.active = 1"
    " AND g.message_uid = ANY(%s) AND g.id < %s AND " + _HANDOFF_PREDICATE
    + " UNION ALL "
    "SELECT g.* FROM messages g WHERE g.session_id = ANY(%s) AND g.active = 0"
    " AND g.message_uid = ANY(%s) AND g.id < %s AND " + _HANDOFF_PREDICATE)

# The handoff rows a shown row of *segment* before slot *before* holds: driven from the matching handoff
# copies, each joined to the rows that hold it, kept when one is its slot's chosen display row.
_HANDOFF_SHOWN_BEFORE_SQL = (
    "SELECT DISTINCT g.* FROM messages g JOIN messages w ON w.session_id = g.session_id AND (w.id = g.id OR ("
    "g.message_uid IS NOT NULL AND g.message_uid <> '' AND w.message_uid = g.message_uid AND w.id > g.id))"
    " WHERE g.session_id = %s AND ({match}) AND "
    + _HANDOFF_PREDICATE
    + " AND w.display_order < %s AND w.id = (SELECT candidate.id FROM messages AS candidate"
    " WHERE candidate.session_id = %s AND candidate.display_order = w.display_order"
    " AND (candidate.active = 1 OR candidate.compacted = 1)"
    + DISPLAY_VISIBLE_SQL
    + not_a_later_tool_copy_sql("candidate")
    + " ORDER BY candidate.active DESC, candidate.id DESC LIMIT 1) ORDER BY g.id")

# The current display slot of the message whose first durable row is *durable* (the SQLite
# ``_CURSOR_*_SQL`` set, ``%s``-bound).
_CURSOR_ROW_SQL = ("SELECT display_identity, display_order, active, compacted, message_uid FROM messages"
                  " WHERE id = %s AND session_id = %s")
_CURSOR_COPY_SQL = (
    "SELECT display_order FROM messages WHERE session_id = %s AND display_identity = %s"
    " AND (active = 1 OR compacted = 1) AND display_order IS NOT NULL"
    " ORDER BY active DESC, id DESC LIMIT 1")
_CURSOR_CARRIER_SQL = (
    "SELECT display_order FROM messages WHERE session_id = %s AND message_uid = %s AND active = 1"
    " AND _compressed_summary = 1 AND display_order IS NOT NULL ORDER BY id DESC LIMIT 1")


# -- session.transcript ------------------------------------------------------------------------------

def display_generations(store, session_id: str, lineage_ids: list[str]) -> list[tuple[dict[str, Any], int]]:
    """``[(replay dict with its row id, first durable id), ...]`` of the lineage's display set: the one
    display dedupe over every segment, keyed by each message's first durable row, a removed original's
    included (the SQLite ``display_generations`` port)."""
    first_ids: dict[int, int] = {}
    rows = _dedupe_display_generations(
        _fetch_conversation_rows(store, lineage_ids, "", with_session_id=False, with_compacted=True),
        drop_removed=True, first_ids=first_ids)
    messages = _rows_to_conversation(rows, session_id=session_id, include_ancestors=True,
                                    repair_alternation=False, include_row_ids=True)
    return [(message, first_ids.get(message["_row_id"], message["_row_id"])) for message in messages]


def transcript_cursor_state(store, session_id: str, through_id: int, lineage_ids: list[str]) -> tuple[int, int, int]:
    """``(live rows, inactive rows, transcript_revision)`` over *lineage_ids*: the session's own live rows
    with id <= *through_id* (only it is appended to) and every live row of its other segments; every
    inactive row; the summed revision (the SQLite port)."""
    ids = list(lineage_ids) or [session_id]
    marks = _in_placeholders(len(ids))
    row = store.one(
        f"SELECT (SELECT COUNT(*) FROM messages WHERE session_id IN ({marks}) AND active = 1"
        " AND (session_id <> %s OR id <= %s)) AS live,"
        f" (SELECT COUNT(*) FROM messages WHERE session_id IN ({marks}) AND active = 0) AS inactive,"
        f" (SELECT COALESCE(SUM(transcript_revision), 0) FROM sessions WHERE id IN ({marks})) AS revision",
        *ids, session_id, int(through_id), *ids, *ids)
    if row is None:
        return (0, 0, 0)
    return int(row["live"]), int(row["inactive"]), int(row["revision"])


def transcript_display_index_complete(store, segment: str) -> bool:
    """Whether every displayed row of *segment* has its display slot (the read-only twin of the display
    backfill; the PG DDL carries the display columns, so the check is the missing-slot probe alone)."""
    return store.one(_DISPLAY_INDEX_MISSING_SQL, segment) is None


def transcript_handoff_rows(store, shown_ids: list[int], segment_ids: list[str]) -> dict[int, list[dict[str, Any]]]:
    """``{shown row id: [candidate handoff message, ...]}`` in id order, for display rows *shown_ids* of
    the lineage *segment_ids*: the shown row when it can hold a handoff, and every earlier stored copy of
    its message that can, decoded as ``_row_to_message_dict`` does (the SQLite port; one indexed read per
    lookup class, never one per shown row)."""
    if not shown_ids:
        return {}
    segments = list(segment_ids)
    shown_ids = sorted(set(shown_ids))
    with store.tx() as conn:
        shown = conn.execute(_SHOWN_ROWS_SQL, (segments, shown_ids)).fetchall()
        uids = sorted({row["_shown_uid"] for row in shown if row["_shown_uid"]})
        copies = conn.execute(
            _HANDOFF_COPIES_SQL,
            (segments, uids, max(row["_shown_id"] for row in shown),
             segments, uids, max(row["_shown_id"] for row in shown))
        ).fetchall() if uids else []
        holder_ids = sorted(int(row["_shown_id"]) for row in shown if row["_holds"])
        holders = {int(row["id"]): row for row in conn.execute(
            "SELECT g.* FROM messages g WHERE g.id = ANY(%s)", (holder_ids,)).fetchall()} if holder_ids else {}
    by_uid: dict[str, list[Any]] = {}
    for row in sorted(copies, key=lambda row: row["id"]):
        by_uid.setdefault(row["message_uid"], []).append(row)
    found: dict[int, list[dict[str, Any]]] = {}
    for row in sorted(shown, key=lambda row: row["_shown_id"]):
        held = ([copy for copy in by_uid.get(row["_shown_uid"], ()) if copy["id"] < row["_shown_id"]]
                if row["_shown_uid"] else [])
        own = holders.get(int(row["_shown_id"])) if row["_holds"] else None
        held += [own] if own is not None else []
        for copy in held:
            message = _row_to_message_dict(copy, warn_context="session.transcript", summary_flag=True)
            found.setdefault(int(row["_shown_id"]), []).append(message)
    return found


def transcript_handoffs_shown_before(store, segment: str, before: int,
                                    handoffs: Iterable[tuple[Any, Any]]) -> list[dict[str, Any]]:
    """The stored copies of *handoffs* (``(message_uid or None, content)``) that a row of *segment*'s
    display set at a ``display_order`` below *before* holds (the SQLite port; the bytea body compares as
    its UTF-8 bytes)."""
    handoffs = list(handoffs)
    uids = sorted({uid for uid, _ in handoffs if uid})
    # Structured (content-block) carriers are matched by their stored encoding, as strings are.
    contents = sorted({_payload_bytes(_encode_content(content)) for uid, content in handoffs
                       if not uid and isinstance(content, (str, list, dict))})
    match: list[str] = []
    params: list = [segment]
    if uids:
        match.append("g.message_uid = ANY(%s)")
        params.append(uids)
    if contents:
        # The stored content column is BYTEA (a text payload's raw UTF-8 bytes; a structured payload
        # its sentinel-prefixed JSON, the same storage form the handoff content encodes to), so the
        # body compares bytea to bytea, not a decoded TEXT against bytea.
        match.append("(g.message_uid IS NULL OR g.message_uid = '') AND g.content = ANY(%s)")
        params.append(contents)
    if not match:
        return []
    params.extend([int(before), segment])
    sql = _HANDOFF_SHOWN_BEFORE_SQL.format(match=" OR ".join(f"({part})" for part in match))
    return [_row_to_message_dict(row, warn_context="session.transcript", summary_flag=True)
            for row in store.all(sql, *params)]


def _transcript_cursor_slot(conn, segment: str, durable: int) -> Optional[int]:
    """The display slot the cursor's durable row resolves to, or ``None`` (the message was rewound or
    deleted; a carried copy or its summary carrier stands in)."""
    row = conn.execute(_CURSOR_ROW_SQL, (durable, segment)).fetchone()
    if row is None:
        return None
    if row["display_identity"] is not None:
        copy = conn.execute(_CURSOR_COPY_SQL, (segment, row["display_identity"])).fetchone()
        if copy is not None:
            return copy["display_order"]
    if row["message_uid"]:
        carrier = conn.execute(_CURSOR_CARRIER_SQL, (segment, row["message_uid"])).fetchone()
        if carrier is not None:
            return carrier["display_order"]
    shown = (row["active"] == 1 or row["compacted"] == 1) and row["display_order"] is not None
    return row["display_order"] if shown else None


def _transcript_durable_firsts(conn, segment: str, raw) -> dict[int, int]:
    """``{row id: first durable id}`` for the page rows a compaction re-wrote: a carried copy (its removed
    original has the same identity) and a summary carrier (the first row with its ``message_uid``; only a
    carrier folds by uid, so a pruned tool result keeps its own identity) (the SQLite port, ``ANY`` lists)."""
    identities = sorted({row["display_identity"] for row in raw if row["display_identity"] is not None})
    stamps = sorted({row["timestamp"] for row in raw if row["timestamp"] is not None})
    by_identity: dict[Any, int] = {}
    if identities and stamps:
        rows = conn.execute(
            "SELECT display_identity, MIN(id) AS first FROM messages WHERE session_id = %s AND active = 0"
            " AND timestamp = ANY(%s) AND compacted = 0 AND display_identity = ANY(%s)"
            " GROUP BY display_identity",
            (segment, stamps, identities)).fetchall()
        by_identity = {row["display_identity"]: row["first"] for row in rows}
    carriers = [row for row in raw if row["_compressed_summary"] and row["message_uid"] and row["timestamp"] is not None]
    by_uid: dict[Any, int] = {}
    if carriers:
        uids = sorted({row["message_uid"] for row in carriers})
        carrier_stamps = sorted({row["timestamp"] for row in carriers})
        rows = conn.execute(
            "SELECT message_uid, MIN(id) AS first FROM messages WHERE session_id = %s AND active = 0"
            " AND timestamp = ANY(%s) AND message_uid = ANY(%s) GROUP BY message_uid",
            (segment, carrier_stamps, uids)).fetchall()
        by_uid = {row["message_uid"]: row["first"] for row in rows}
    return {row["id"]: min(row["display_order"], by_identity.get(row["display_identity"], row["id"]),
                          by_uid.get(row["message_uid"], row["id"]) if row["_compressed_summary"] else row["id"])
            for row in raw}


def segment_page_messages(store, segment: str, *, latest: bool, after: Optional[tuple[int, int]],
                         count: int, inclusive: bool = False):
    """``(messages with their (display_order, durable) key, full, edge key, first slot)`` of one segment
    page: the rows get the conversation normalization a resume applies, with the display row before the
    page as context (dropped) so a prompt/reply pair split by the page edge is still one pair (the
    SQLite ``segment_page_messages`` port, one read transaction)."""
    def _read(conn):
        position = None
        if after is not None:
            found = _transcript_cursor_slot(conn, segment, after[1])
            position = after[0] if found is None else found
            if inclusive:
                position += 1 if latest else -1
        raw_sql, raw_params = _display_rows_sql(segment, limit=count, latest=latest, after=position)
        raw = conn.execute(raw_sql, raw_params).fetchall()
        firsts = _transcript_durable_firsts(conn, segment, raw)
        if raw:
            before_sql, before_params = _display_rows_sql(
                segment, limit=1, latest=True, after=min(row["display_order"] for row in raw))
            before = conn.execute(before_sql, before_params).fetchall()
        else:
            before = []
        return raw, firsts, before

    with store.tx() as conn:
        raw, firsts, before = _read(conn)
    context_ids = {row["id"] for row in before}
    keys = {row["id"]: (row["display_order"], firsts[row["id"]]) for row in raw}
    messages = [message for message in _rows_to_conversation(
        [*before, *raw], session_id=segment, include_ancestors=False, repair_alternation=False,
        include_row_ids=True)
        if message["_row_id"] not in context_ids]
    edge = None if not raw else keys[raw[0]["id"] if latest else raw[-1]["id"]]
    start = min((row["display_order"] for row in raw), default=None)
    return ([(message, keys[message["_row_id"]]) for message in messages], len(raw) == count, edge, start)


def transcript_tool_calls_before(store, segment_ids: list[str], before_id: int, limit: int = 16) -> list[tuple[Any, Any]]:
    """``(tool_calls, tool_call_uids)`` JSON of the *limit* newest assistant rows with tool calls among
    *segment_ids* below row id *before_id*, nearest first (the SQLite port)."""
    rows = store.all(
        "SELECT tool_calls, tool_call_uids FROM messages WHERE session_id = ANY(%s)"
        " AND role = 'assistant' AND tool_calls IS NOT NULL AND id < %s ORDER BY id DESC LIMIT %s",
        list(segment_ids), int(before_id), int(limit))
    return [(row["tool_calls"], row["tool_call_uids"]) for row in rows]

"""Transcript read verbs on PostgreSQL (the SQLite store's read functions ported onto
``PostgresStore``): the ``get_messages`` family, the display projections (dedupe, legacy page,
counts), the conversation/OpenAI projections, the resume lineage and its safety guard, and the
single-row probes. Rows come back with the BYTEA payload columns decoded (the bytea payload
rule); every SQL is ``%s``-bound and JSON work goes through ``::jsonb``."""

from __future__ import annotations

from typing import Any, Mapping, Optional

from session_rules.turns import (
    _DB_PERSISTED_MARKER as _DB_PERSISTED_MARKER_KEY, index_tool_call_uids, resolve_tool_call_uid,
)
from persistence import StoreUnavailable
from persistence.sqlite._state.common import _json_or
from persistence.sqlite.conversation._store.compaction_display import later_tool_copy_ids
from persistence.sqlite.conversation._store.core import (
    _CONVERSATION_ROW_COLUMNS, SessionResumeTooLargeError, _strip_background_review_harness,
    _strip_stale_tool_call_markers,
)
from persistence.sqlite.conversation._store.identity import _restore_identity_columns, _uid_list
from persistence.sqlite.conversation._store.messages import (
    _canonical_replayed_user_content, _exact_replayed_user_clone_key, _find_duplicate_replayed_user_message,
)
from persistence.sqlite.conversation._store.messages_sql import logger
from persistence.sqlite.conversation._store.rules import DB_ROW_SNAPSHOT, TOOL_CALL_UID, lift_data_use

from .common import (
    DISPLAY_VISIBLE_SQL, not_a_later_tool_copy_sql,
    _ACTIVE_IDS_SQL, _DISPLAY_ACTIVE_CLAUSE, _display_dedupe_key, _display_identity, _display_rows,
    _decode_display_metadata, _decode_payloads, _decode_payloads_rows, _ensure_display_index,
    _fold_summary_carrier, _in_placeholders, _is_explicit_fork_child_row, _is_model_only_row,
    _json_value_or, _loaded_view_content, _decode_content,
)
from .repair import transcript_row_snapshot
from .sessions_pg import get_session

__all__ = [
    "assert_resume_safe", "conversation_revision", "display_message_count", "find_pr_url_messages",
    "get_active_message_ids", "get_active_message_watermark", "get_message_role", "get_messages",
    "get_messages_around", "get_messages_as_conversation", "get_resume_conversations",
    "has_gateway_input_owner", "has_platform_message_id", "latest_conversation_boundary",
    "latest_conversation_role", "latest_message_row_id", "message_count", "resolve_active_row_id",
]


# ── display-set plumbing ──

_REMOVED_IDENTITIES_SQL = (
    "SELECT session_id, id, role, timestamp, tool_call_id, tool_calls, tool_name, display_kind, display_metadata, "
    "message_uid, absorbed_message_uids, display_identity, active, compacted, "
    "CASE WHEN display_identity IS NULL THEN content END AS content "
    "FROM messages WHERE session_id IN ({ids}) AND active = 0 AND compacted = 0 ORDER BY id")


def _display_set(store, sql: str, session_ids: list[str]) -> list[Any]:
    """"*sql*'s rows (bound to *session_ids*) with the removed rows' identities, in ONE read transaction:
    a compaction committing between the two reads marks a row removed after it was read live, so the
    removed read would hide the message while the live read never saw its new copy (the SQLite
    ``_display_set`` port; an uninitialised read-only store degrades to empty, as its reads do)."""
    ids_sql = _REMOVED_IDENTITIES_SQL.format(ids=_in_placeholders(len(session_ids)))
    try:
        with store.tx() as conn:
            rows = conn.execute(sql, tuple(session_ids)).fetchall()
            removed = conn.execute(ids_sql, tuple(session_ids)).fetchall()
    except StoreUnavailable:
        return []
    return sorted([*rows, *removed], key=lambda row: row["id"]) if removed else list(rows)


def _dedupe_display_generations(rows, *, drop_removed: bool = False,
                               first_ids: Optional[dict] = None) -> list[dict[str, Any]]:
    """Collapse compaction generations so each logical message appears once (the protected tail is copied
    into each generation: same role/content/timestamp, different ``active``/id); prefer the live row, then
    the newest. The ONE definition every display projection shares; *rows* must be ordered by ``id``.
    (The SQLite ``_dedupe_display_generations`` port over dict rows: the PG display identity is a stored
    BYTEA.)"""
    seen: dict[Any, Any] = {}
    first_id: dict[Any, int] = {}
    uid_key: dict[Any, Any] = {}  # message_uid -> the display key of its first copy
    removed: dict[Any, bool] = {}  # copy identity -> its newest copy is removed (rows come in id order)
    identities: dict[Any, list[Any]] = {}
    removed_ts: set[Any] = set()
    removed_first: dict[Any, int] = {}  # copy identity -> its first removed row (rows come in id order)
    rows = rows if isinstance(rows, list) else list(rows)
    later_copies = later_tool_copy_ids(None, rows)  # a compacted tool result paints once, as its original
    if drop_removed:
        removed_ts = {row["timestamp"] for row in rows if not row["active"] and not row["compacted"]}
        drop_removed = bool(removed_ts)
    for row in rows:
        copies, key = None, None
        if drop_removed:
            gone = not row["active"] and not row["compacted"]
            stored = row["display_identity"] if gone else None
            key = None if stored else _display_dedupe_key(row)
            hashed = stored or (_display_identity(key) if key is not None and (
                gone or row["timestamp"] in removed_ts) else None)
            copies = [*((hashed,) if hashed else ()), *(
                ("uid", uid) for uid in (row["message_uid"], *_uid_list(row["absorbed_message_uids"])) if uid)]
            removed.update(dict.fromkeys(copies, gone))
            if gone:
                for copy in copies:
                    removed_first.setdefault(copy, row["id"])
                continue
        if _is_model_only_row(row) or row["id"] in later_copies:
            continue
        key = key or _display_dedupe_key(row)
        # Only the summary carrier folds by uid: other rewritten copies (a pruned tool stub) must keep
        # their archived original visible.
        key = _fold_summary_carrier(row, key, uid_key)
        cur = seen.get(key)
        if cur is None or (row["active"], row["id"]) > (cur["active"], cur["id"]):
            seen[key] = row
            identities[key] = copies
        first_id[key] = min(first_id.get(key, row["id"]), row["id"])
    keys = sorted(seen, key=first_id.__getitem__)
    if drop_removed:
        keys = [key for key in keys if not any(removed[copy] for copy in identities[key])]
    if first_ids is not None:
        for key in keys:
            first_ids[seen[key]["id"]] = min([first_id[key], *[
                removed_first[copy] for copy in identities[key] or () if copy in removed_first]])
    return [seen[key] for key in keys]


def _legacy_display_page(store, session_id: str, *, active_clause: str, limit: Optional[int], offset: int,
                         latest: bool) -> list[dict[str, Any]]:
    """Project a legacy display page without retaining transcript payloads (the SQLite
    ``_legacy_display_page`` port: fixed-width identities and representative ids while scanning, then
    fetch the selected payloads)."""
    representatives: dict[bytes, tuple[int, int]] = {}
    rows = store.all(
        "SELECT id, role, content, timestamp, tool_call_id, tool_calls, tool_name, active, "
        f"display_kind, display_metadata FROM messages WHERE session_id = %s{active_clause} ORDER BY id ASC",
        session_id)
    for row in _decode_payloads_rows(rows):
        if _is_model_only_row(row):
            continue
        identity = _display_identity(_display_dedupe_key(row))
        current = representatives.get(identity)
        candidate = (row["active"], row["id"])
        if current is None or candidate > current:
            representatives[identity] = candidate
    identities = list(representatives)
    identities = identities[::-1][offset:][:limit][::-1] if latest else identities[offset:][:limit]
    selected_ids = [representatives[identity][1] for identity in identities]
    selected: dict[int, dict[str, Any]] = {}
    for start in range(0, len(selected_ids), 900):
        chunk = selected_ids[start:start + 900]
        sel = store.all(
            f"SELECT * FROM messages WHERE session_id = %s{active_clause} "
            f"AND id IN ({_in_placeholders(len(chunk))})",
            [session_id, *chunk])
        selected.update({row["id"]: row for row in sel})
    return [selected[row_id] for row_id in selected_ids if row_id in selected]


def display_message_count(store, session_id: str) -> int:
    """Rows a display read of this segment paints: one per ``display_order`` group (the SQLite
    ``display_message_count`` port; the plain ``::jsonb`` is safe here, ``display_metadata`` is always
    valid JSON or NULL)."""
    n = store.scalar(
        "SELECT COUNT(*) FROM (SELECT DISTINCT display_order FROM messages WHERE session_id = %s"
        f"{_DISPLAY_ACTIVE_CLAUSE}{DISPLAY_VISIBLE_SQL}{not_a_later_tool_copy_sql('messages')})")
    return int(n or 0)


# ── the get_messages family ──

def _active_clause(include_inactive: bool, include_compacted: bool) -> str:
    """Audit: every row; display: active plus compaction-archived (never Undo/Rewind rows); default: live."""
    return "" if include_inactive else (_DISPLAY_ACTIVE_CLAUSE if include_compacted else " AND active = 1")


def get_messages(store, session_id: str, include_inactive: bool = False, include_compacted: bool = False,
                limit: Optional[int] = None, offset: int = 0, latest: bool = False,
                after_id: Optional[int] = None, include_ancestors: bool = False) -> list[dict[str, Any]]:
    """Load messages in insertion order (id, never timestamp: clocks regress). ``include_inactive``:
    rewind rows too; ``include_compacted``: compaction-archived display history (not rewind rows).
    ``latest`` pages back from the newest but returns chronological order; ``after_id``: keyset paging.
    ``include_ancestors``: also load the compression lineage (root -> tip, branch sessions exempt)
    (the SQLite ``get_messages`` port; ``LIMIT NULL`` is the PG no-limit form)."""
    if after_id is not None and (latest or offset):
        raise ValueError("after_id is incompatible with latest/offset paging")
    if after_id is not None and include_compacted:
        raise ValueError("after_id is incompatible with include_compacted (deduped display reads use offset paging)")
    if after_id is not None and include_ancestors:
        raise ValueError("after_id is incompatible with include_ancestors (merged-lineage reads use offset paging)")
    active_clause = _active_clause(include_inactive, include_compacted)
    session_ids = _resume_lineage_ids(store, session_id) if include_ancestors else [session_id]
    if len(session_ids) > 1:
        # Multi-segment lineage: dedupe-then-page on the merged display set (display_order is
        # per-segment and not comparable across segments).
        display = include_compacted and not include_inactive  # removed rows too: see drop_removed
        sql = (f"SELECT * FROM messages WHERE session_id IN ({_in_placeholders(len(session_ids))})"
               f"{active_clause} ORDER BY id ASC")
        rows = (_dedupe_display_generations(_display_set(store, sql, session_ids), drop_removed=display)
                if display else store.all(sql, *session_ids))
        rows = rows[::-1][offset:][:limit][::-1] if latest else rows[offset:][:limit]
    elif include_compacted and not include_inactive and _ensure_display_index(store, session_id):
        rows = _display_rows(store, session_id, limit=limit, offset=offset, latest=latest)
    elif include_compacted:
        # Read-only legacy stores cannot persist display identities; the Python projection instead.
        rows = _legacy_display_page(store, session_id, active_clause=active_clause, limit=limit,
                                   offset=offset, latest=latest)
    else:
        sql = (f"SELECT * FROM messages WHERE session_id = %s{active_clause}"
               f"{' AND id > %s' if after_id is not None else ''} ORDER BY id {'DESC' if latest else 'ASC'}")
        params: list = [session_id] if after_id is None else [session_id, after_id]
        if limit is not None or offset:
            sql += " LIMIT %s OFFSET %s"
            params.extend([None if limit is None else limit, offset])
        rows = store.all(sql, *params)
        if latest:
            rows.reverse()
    from .common import _row_to_message_dict
    return [_row_to_message_dict(row, warn_context="get_messages", summary_flag=True) for row in rows]


def find_pr_url_messages(store, session_ids: list[str]) -> list[dict[str, Any]]:
    """Tool results containing ``/pull/``: a deliberately loose scan, oldest-first so the caller takes the
    last (the SQLite ``find_pr_url_messages`` port; the bytea body is scanned as UTF-8 text)."""
    ids = [s for s in session_ids if s]
    out: list[dict[str, Any]] = []
    for start in range(0, len(ids), 900):  # bound-parameter ceiling
        chunk = ids[start:start + 900]
        rows = store.all(
            """SELECT session_id, content FROM messages
                    WHERE session_id IN (""" + _in_placeholders(len(chunk)) + """)
                      AND role = 'tool' AND convert_to(content, 'UTF8') LIKE '%/pull/%'
                    ORDER BY id ASC""",
            *chunk)
        out.extend({"session_id": row["session_id"], "content": _json_value_or(
            _decode_payloads(row)["content"], None, "pr-url hit")} for row in rows)
    return out


def get_messages_around(store, session_id: str, around_message_id: int, window: int = 5) -> dict[str, Any]:
    """Up to *window* messages either side of an anchor id (ascending); ``messages_before``/``_after`` count
    strictly around the anchor (the SQLite ``get_messages_around`` port)."""
    window = max(window, 0)
    if store.one("SELECT 1 FROM messages WHERE id = %s AND session_id = %s LIMIT 1",
                 around_message_id, session_id) is None:
        return {"window": [], "messages_before": 0, "messages_after": 0}
    before_rows = store.all(
        "SELECT * FROM messages WHERE session_id = %s AND id <= %s ORDER BY id DESC LIMIT %s",
        session_id, around_message_id, window + 1)
    after_rows = store.all(
        "SELECT * FROM messages WHERE session_id = %s AND id > %s ORDER BY id ASC LIMIT %s",
        session_id, around_message_id, window)
    from .common import _row_to_message_dict
    window_msgs = [_row_to_message_dict(r, warn_context="get_messages_around") for r in (*reversed(before_rows), *after_rows)]
    return {"window": window_msgs, "messages_before": max(0, len(before_rows) - 1), "messages_after": len(after_rows)}


# ── conversation (OpenAI) projection ──

def _dedupe_replayed_user(messages: list[dict[str, Any]], msg: dict[str, Any],
                         exact_user_clones: dict) -> tuple[bool, Any]:
    """Ancestor-lineage dedupe of one decoded user *msg* -> ``(skip, exact_clone_key)`` (the holder-less
    SQLite ``_dedupe_replayed_user``: its helpers are pure-Python over dict messages)."""
    canonical_content = _canonical_replayed_user_content(msg)[0]
    exact_clone_key = _exact_replayed_user_clone_key(msg.get("timestamp"), canonical_content)
    previous_exact = exact_user_clones.get(exact_clone_key) if exact_clone_key is not None else None
    duplicate = None
    if previous_exact is not None:
        previous_index = next((i for i, candidate in enumerate(messages) if candidate is previous_exact), None)
        if previous_index is not None:
            duplicate = (previous_index, True)
    if duplicate is None:
        duplicate = _find_duplicate_replayed_user_message(messages, msg)
    if duplicate is None:
        return False, exact_clone_key
    duplicate_index, prefer_current = duplicate
    if prefer_current:
        messages.pop(duplicate_index)
    return not prefer_current, exact_clone_key


def _rows_to_conversation(rows, *, session_id: str, include_ancestors: bool, repair_alternation: bool,
                          include_row_ids: bool = False,
                          include_summary_markers: bool = False) -> list[dict[str, Any]]:
    """Decode fetched rows (ordered by id, pre-filtered) into OpenAI format, stable key order (the SQLite
    ``_rows_to_conversation`` port; the PG payload columns decode before the projection, ``api_content``
    VERBATIM so replay keeps the provider prompt cache byte-stable)."""
    stamp_snapshot = repair_alternation and not include_row_ids
    messages: list[dict[str, Any]] = []
    exact_user_clones: dict[tuple[Any, str], dict[str, Any]] = {}
    tool_uid_index: dict[str, str] = {}
    unindexed_tool_owners: list[dict[str, Any]] = []
    for raw in rows:
        row = _decode_payloads(dict(raw))
        content = _loaded_view_content(row["role"], _decode_content(row["content"]))
        msg = {"role": row["role"], "content": content, _DB_PERSISTED_MARKER_KEY: True}
        if stamp_snapshot:
            msg[DB_ROW_SNAPSHOT] = transcript_row_snapshot(row)
        if include_row_ids and row["id"] is not None:
            msg["_row_id"] = row["id"]
        _restore_identity_columns(row, msg)
        msg.update((col, row[col]) for col in ("api_content", "display_kind") if row[col])
        if row["display_metadata"] and (decoded := _decode_display_metadata(row["display_metadata"])) is not None:
            msg["display_metadata"] = decoded
            lift_data_use(msg)
        if include_summary_markers and row["_compressed_summary"]:
            msg["_compressed_summary"] = True
        msg.update(
            (col, row[col]) for col in ("timestamp", "tool_call_id", "tool_name", "effect_disposition") if row[col])
        if row["tool_calls"]:
            msg["tool_calls"] = _json_or(
                row["tool_calls"], [], "Failed to deserialize tool_calls in conversation replay, falling back to []")
        if row["platform_message_id"]:
            msg["message_id"] = row["platform_message_id"]
        if row["observed"]:
            msg["observed"] = True
        if row["role"] == "assistant":
            msg.update((col, row[col]) for col in ("finish_reason", "reasoning") if row[col])
            if row["reasoning_content"] is not None:
                msg["reasoning_content"] = row["reasoning_content"]
            msg.update(
                (col, _json_or(row[col], None, f"Failed to deserialize {col}, falling back to None"))
                for col in ("reasoning_details", "codex_reasoning_items", "codex_message_items") if row[col])
            if msg.get("tool_calls"):
                unindexed_tool_owners.append(msg)
        elif row["role"] == "user":
            tool_uid_index.clear()  # a result never pairs across a user turn
            unindexed_tool_owners.clear()
        elif row["role"] == "tool" and not row["tool_call_uid"] and row["tool_call_id"]:
            for owner in unindexed_tool_owners:
                index_tool_call_uids(tool_uid_index, owner)
            unindexed_tool_owners.clear()
            if tool_uid := resolve_tool_call_uid(tool_uid_index, row["tool_call_id"]):
                msg[TOOL_CALL_UID] = tool_uid
        if include_ancestors:
            skip, exact_clone_key = _dedupe_replayed_user(messages, msg, exact_user_clones)
            if skip:
                continue
            if exact_clone_key is not None:
                exact_user_clones[exact_clone_key] = msg
        messages.append(msg)
    messages = _strip_stale_tool_call_markers(_strip_background_review_harness(messages))
    if repair_alternation and messages:
        from session_rules.turns import repair_message_sequence
        repaired = repair_message_sequence(None, messages)
        if repaired:
            logger.info("Repaired %d message-alternation violation(s) while "
                        "restoring session %s - durable transcript kept them, "
                        "see repair_message_sequence", repaired, session_id)
    return messages


def _fetch_conversation_rows(store, session_ids: list[str], active_clause: str, *, with_session_id: bool,
                             with_compacted: bool = False) -> list[Any]:
    """``_CONVERSATION_ROW_COLUMNS`` rows for *session_ids* ORDER BY id (timestamps are not monotonic
    and would break tool-call adjacency); ``with_compacted``: the display set with the ``compacted``
    column, plus the removed rows' identities (the SQLite port)."""
    sql = (f"SELECT {'session_id, ' if with_session_id or with_compacted else ''}"
           f"{'compacted, display_order, ' if with_compacted else ''}"
           f"{_CONVERSATION_ROW_COLUMNS} "
           f"FROM messages WHERE session_id IN ({_in_placeholders(len(session_ids))})"
           f"{_DISPLAY_ACTIVE_CLAUSE if with_compacted else active_clause} ORDER BY id")
    if with_compacted:
        return _display_set(store, sql, session_ids)
    return store.all(sql, *session_ids)


def get_messages_as_conversation(store, session_id: str, include_ancestors: bool = False,
                                 include_inactive: bool = False, repair_alternation: bool = False,
                                 include_row_ids: bool = False,
                                 include_compacted: bool = False) -> list[dict[str, Any]]:
    """Load messages in OpenAI format (the SQLite ``get_messages_as_conversation`` port)."""
    display = include_compacted and not include_inactive  # removed rows too: see drop_removed
    rows = _fetch_conversation_rows(
        store, _resume_lineage_ids(store, session_id) if include_ancestors else [session_id],
        _active_clause(include_inactive, include_compacted), with_session_id=False, with_compacted=display)
    if include_compacted:
        rows = _dedupe_display_generations(rows, drop_removed=display)
    return _rows_to_conversation(rows, session_id=session_id, include_ancestors=include_ancestors,
                                 repair_alternation=repair_alternation, include_row_ids=include_row_ids,
                                 include_summary_markers=repair_alternation)


def conversation_revision(store, session_id: str, include_ancestors: bool = False) -> tuple:
    """A cheap fingerprint of what ``get_messages_as_conversation`` (live rows) would load (the SQLite
    ``conversation_revision`` port)."""
    ids = _resume_lineage_ids(store, session_id) if include_ancestors else [session_id]
    marks = _in_placeholders(len(ids))
    row = store.one(
        f"SELECT COUNT(*) AS c, COALESCE(MAX(id), 0) AS m, "
        f"(SELECT COALESCE(SUM(transcript_revision), 0) FROM sessions WHERE id IN ({marks})) AS r "
        f"FROM messages WHERE session_id IN ({marks}) AND active = 1", *ids, *ids)
    if row is None:
        return (tuple(ids), 0, 0, 0)
    return (tuple(ids), int(row["c"]), int(row["m"]), int(row["r"]))


# ── resume lineage ──

def _is_compression_child(store, child: Mapping[str, Any]) -> bool:
    """Per-hop compression-lineage gate over the store (the SQLite ``_is_compression_child_row`` shape):
    an explicit branch/delegate/tool child (a reset fork included) is its own conversation, and the
    parent must have ended ``compression``."""
    parent_id = child.get("parent_session_id")
    if not parent_id or _is_explicit_fork_child_row(child, include_reset=True):
        return False
    parent = get_session(store, parent_id)
    return bool(parent and parent["end_reason"] == "compression")


def _resume_lineage_ids(store, session_id: str) -> list[str]:
    """Session ids a display resume materializes: the VERIFIED compression lineage (root -> tip; each hop's
    parent ended ``compression`` and the child is a genuine continuation), or the session alone for anything
    else (the SQLite ``_resume_lineage_ids`` port)."""
    if not session_id:
        return [session_id]
    session = get_session(store, session_id)
    if not session or _is_explicit_fork_child_row(session):
        return [session_id]
    chain = [session_id]
    current = session
    seen = {session_id}
    while len(chain) < 100:  # defensive bound
        if not _is_compression_child(store, current):
            break
        parent = get_session(store, current["parent_session_id"])
        if not parent or parent["id"] in seen:
            break
        seen.add(parent["id"])
        chain.append(parent["id"])
        current = parent
    return list(reversed(chain))


def _resume_count_scope(store, session_id: str, tip_only: bool) -> tuple[list[str], str]:
    """``tip_only``: the tip's ACTIVE rows (model restore); else the full-lineage DISPLAY set."""
    if tip_only:
        return [session_id], "active = 1"
    return _resume_lineage_ids(store, session_id), "(active = 1 OR compacted = 1)"


def assert_resume_safe(store, session_id: str, max_messages: int, *, tip_only: bool = False) -> int:
    """Resume row count, or raise ``SessionResumeTooLargeError`` (the SQLite ``assert_resume_safe``
    port; 0 disables the guard without counting)."""
    if max_messages < 0:
        raise ValueError("max_messages must be non-negative")
    if max_messages == 0:
        return 0
    session_ids, active_clause = _resume_count_scope(store, session_id, tip_only)
    count = store.scalar(
        f"SELECT COUNT(*) FROM (SELECT 1 FROM messages WHERE session_id IN ({_in_placeholders(len(session_ids))}) "
        f"AND {active_clause} LIMIT %s)", *session_ids, max_messages + 1)
    message_count = int(count or 0)
    if message_count > max_messages:
        raise SessionResumeTooLargeError(
            message_count, max_messages, scope="in its tip segment" if tip_only else "across its lineage")
    return message_count


def get_resume_conversations(store, session_id: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """``(model_history, display_history)`` for a resume from ONE read (the SQLite
    ``get_resume_conversations`` port: model = the tip's active rows, alternation-repaired; display =
    the full lineage, compaction-archived rows deduped)."""
    rows = _fetch_conversation_rows(store, _resume_lineage_ids(store, session_id), "",
                                   with_session_id=True, with_compacted=True)
    model_history = _rows_to_conversation(
        [r for r in rows if r["session_id"] == session_id and r["active"] and r["role"] != "session_meta"],
        session_id=session_id, include_ancestors=False, repair_alternation=True,
        include_row_ids=True, include_summary_markers=True)
    display_history = _rows_to_conversation(
        _dedupe_display_generations(rows, drop_removed=True), session_id=session_id,
        include_ancestors=True, repair_alternation=False, include_row_ids=True)
    return model_history, display_history


# ── single-row probes ──

def message_count(store, session_id: Optional[str] = None) -> int:
    """Count messages, optionally for a specific session (the SQLite ``message_count`` port)."""
    sql = "SELECT COUNT(*) FROM messages" + (" WHERE session_id = %s" if session_id else "")
    n = store.scalar(sql, session_id) if session_id else store.scalar(sql)
    return int(n or 0)


def get_active_message_ids(store, session_id: str) -> list[int]:
    """Ordered physical active ids for rewind CAS checks (the SQLite ``get_active_message_ids`` port)."""
    return [int(row["id"]) for row in store.all(_ACTIVE_IDS_SQL, session_id)]


def get_active_message_watermark(store, session_id: str) -> int:
    """MAX(id) of the active rows (0 if none), captured at compression START (the SQLite port)."""
    if not session_id:
        return 0
    n = store.scalar(
        "SELECT COALESCE(MAX(id), 0) FROM messages WHERE session_id = %s AND active = 1", session_id)
    return int(n or 0)


def latest_message_row_id(store, session_id: str, *, role: str = "user", offset: int = 0,
                         require_text: bool = True) -> Optional[int]:
    """Row id of the most recent active *role* message, or ``None`` (the SQLite port; the bytea body is
    trimmed as UTF-8 text)."""
    if not session_id or role not in {"user", "assistant"} or offset < 0:
        return None
    text_filter = "AND content IS NOT NULL AND TRIM(convert_to(content, 'UTF8')) <> '' " if require_text else ""
    row = store.one(
        "SELECT id FROM messages WHERE session_id = %s AND role = %s "
        f"AND active = 1 {text_filter}ORDER BY id DESC LIMIT 1 OFFSET %s",
        session_id, role, int(offset))
    return int(row["id"]) if row else None


def latest_conversation_role(store, session_id: str) -> Optional[str]:
    """Role of the newest active user/assistant/tool row, or ``None`` (the SQLite port)."""
    row = store.one(
        "SELECT role FROM messages WHERE session_id = %s AND active = 1 "
        "AND role NOT IN ('session_meta', 'system') ORDER BY id DESC LIMIT 1", session_id)
    return row["role"] if row else None


def get_message_role(store, session_id: str, row_id: int) -> Optional[str]:
    """Role of the active message at *row_id* in *session_id*, or ``None`` (the SQLite port)."""
    if not session_id:
        return None
    row = store.one("SELECT role FROM messages WHERE id = %s AND session_id = %s AND active = 1",
                    int(row_id), session_id)
    return row["role"] if row else None


def resolve_active_row_id(store, session_id: str, row_id: int) -> Optional[int]:
    """The active row that still carries *row_id*'s message (the SQLite ``resolve_active_row_id``
    port; the bytea ``content`` compares with PG's null-safe ``IS``)."""
    if not session_id or isinstance(row_id, bool) or not isinstance(row_id, int) or row_id <= 0:
        return None
    origin = store.one("SELECT active FROM messages WHERE id = %s AND session_id = %s", row_id, session_id)
    if origin is None:
        return None
    if origin["active"]:
        return row_id
    clones = store.all(
        "SELECT c.id FROM messages c JOIN messages o ON o.id = %s "
        "WHERE c.session_id = %s AND c.active = 1 AND c.id > o.id AND c.role = o.role "
        "AND c.content IS o.content AND c.timestamp = o.timestamp",
        row_id, session_id)
    return int(clones[0]["id"]) if len(clones) == 1 else None


def has_platform_message_id(store, session_id: str, platform_message_id: str) -> bool:
    """True when *platform_message_id* exists (the gateway's transient-failure dedupe probe, the SQLite
    port)."""
    return store.one(
        "SELECT 1 FROM messages WHERE session_id = %s AND platform_message_id = %s LIMIT 1",
        session_id, platform_message_id) is not None


def has_gateway_input_owner(store, session_id: str, owner: str) -> bool:
    """Probe the accepted-input marker without allocating message bodies or archives (the SQLite port:
    the ``json_valid`` guard is Python-side, since a plain ``::jsonb`` cast would throw on a malformed
    row)."""
    rows = store.all(
        "SELECT display_metadata FROM messages WHERE session_id = %s AND role = 'user' "
        "AND observed = 0 AND (active = 1 OR compacted = 1)", session_id)
    for row in rows:
        meta = _decode_display_metadata(row["display_metadata"])
        if isinstance(meta, dict) and meta.get("gateway_input_owner") == owner:
            return True
    return False


def latest_conversation_boundary(store, session_key: str, source: str) -> Optional[int]:
    """Conversation boundaries this peer has crossed, or ``None`` if never reset (the SQLite port, read
    from ``conversation_generations``)."""
    if not session_key or not source:
        return None
    row = store.one(
        "SELECT generation FROM conversation_generations WHERE source = %s AND session_key = %s",
        source, session_key)
    generation = int(row["generation"]) if row is not None and row["generation"] is not None else 0
    return generation if generation > 0 else None

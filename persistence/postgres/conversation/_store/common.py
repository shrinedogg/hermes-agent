"""Shared PG-dialect plumbing for the ported conversation store: ``%s`` placeholder expansion, the BYTEA
payload columns (``content``/``api_content``/``reasoning_content`` are UTF-8 bytes; text work goes through
``convert_to(…, 'UTF8')``), the ported SQL constants, and the display-index maintenance the DDL display
triggers leave to the write verbs (a content/display-kind update NULLs the row's identity; the verb
re-folds in-transaction)."""

from __future__ import annotations

import hashlib
import json
import re
import time
from typing import Any, Mapping, Optional

from process_identity import holder_process_is_dead
from session_rules.turns import (
    MODEL_ONLY_DISPLAY_METADATA_KEY, coalesce_tool_call_id, index_tool_call_uids, message_uid_or_none,
    mint_uid, resolve_tool_call_uid, sanitize_context, split_user_originated_turn,
)
from persistence import CompressionClosed, SessionRowMissing, SessionId, TurnLeaseLost
from persistence.sqlite._state.common import _ended_by_compression, _json_or
from persistence.sqlite._state.errors import (
    CompressionSessionClosedError, HeldRowRewrittenError, SessionCompressionInProgressError,
    SessionTurnLeaseLostError,
)
from persistence.sqlite.conversation._store.core import _CONTENT_JSON_PREFIX
from persistence.sqlite.conversation._store.identity import (
    _absorbed_uids_json, _restore_identity_columns, _stable_tool_key, _tool_call_uid_map,
    _tool_call_uid_or_none, _tool_call_uids_json,
)
from persistence.sqlite.conversation._store.messages_sql import (
    _coerce_timestamp, _parse_tool_calls, _scrub_surrogates, _tool_calls_count, _tool_calls_len,
)
from persistence.sqlite.conversation._store.rules import (
    CANONICAL_ROW, DATA_USE, DB_ROW_SNAPSHOT, TOOL_CALL_UID, _is_checkpoint_item, _newest_checkpoint_carrier,
    _sanitize_surrogates, coerce_epoch, lift_data_use, stamp_message_uid,
)

__all__ = [
    "API_CONTENT_TEXT", "CONTENT_TEXT", "DISPLAY_VISIBLE_SQL", "not_a_later_tool_copy_sql",
    "_ACTIVE_IDS_SQL", "_ARCHIVE_ACTIVE_SQL", "_BUMP_GENERATION_SQL", "_COMPRESSION_LOCK_ROW_SQL",
    "_DELETE_COMPRESSION_LOCK_SQL", "_DISPLAY_ACTIVE_CLAUSE", "_DISPLAY_INDEX_MISSING_SQL",
    "_DISPLAY_META_ROW_SQL", "_ENDED_ROW_SQL", "_INSERT_MESSAGE_SQL", "_LIVE_IDENTITY_SQL",
    "_MESSAGE_SCHEMA_KEYS", "_MESSAGE_WRITE_COLUMNS", "_RESET_COUNTERS_SQL", "_ROW_STATE_KEYS",
    "_SET_CODEX_REASONING_SQL", "_SET_COUNTERS_SQL", "_SET_DISPLAY_META_SQL",
    "_SHADOWED_CHECKPOINT_ROWS_SQL", "_TURN_LEASE_ROW_SQL",
    "_bump_session_counters", "_carry_parent_timestamps", "_check_transcript_write_guards",
    "_clone_message_rows", "_decode_display_metadata", "_decode_payloads", "_decode_content",
    "_display_dedupe_key", "_display_identity", "_display_rows", "_display_rows_sql", "_encode_content",
    "_encode_display_metadata", "_fold_summary_carrier", "_in_placeholders", "_is_explicit_fork_child_row",
    "_is_model_only_row", "_load_transcript_write", "_message_row_params", "_phs",
    "_payload_bytes", "_payload_str", "_prune_shadowed_checkpoints", "_reconcile_display_orders",
    "_reasoning_json_text", "_row_identity", "_row_to_message_dict", "_scrub", "_serialized_message_row",
    "_stale_holder", "_stamp_kept_live_prefix", "_transcript_write",
]


# ── PG placeholder plumbing ──

def _phs(count: int) -> str:
    """``%s, %s, %s`` for *count* bound parameters."""
    return ", ".join(["%s"] * count)


def _in_placeholders(items) -> str:
    """``(%s, %s, %s)`` for one bound parameter per element of *items*; an empty list is ``(NULL)``."""
    ids = list(items)
    return f"({', '.join(['%s'] * len(ids))})" if ids else "(NULL)"


# ── BYTEA payload columns: text work through convert_to ──

CONTENT_TEXT = "convert_to(content, 'UTF8')"
API_CONTENT_TEXT = "convert_to(api_content, 'UTF8')"


def _payload_bytes(value: Any) -> Optional[bytes]:
    """A payload column (``content``/``api_content``/``reasoning_content``) as its BYTEA storage form:
    UTF-8 bytes (PostgreSQL ``text`` cannot hold the NUL the ``\\x00json:`` sentinel encoding carries;
    the SQLite reference stores it in a BLOB/TEXT column that can). Non-string scalars are coerced
    through ``str`` first (the SQLite TEXT affinity)."""
    if value is None:
        return None
    if isinstance(value, (bytes, bytearray)):
        return bytes(value)
    if isinstance(value, str):
        return value.encode("utf-8")
    return str(value).encode("utf-8")


def _payload_str(value: Any) -> Optional[str]:
    """The inverse of :func:`_payload_bytes` for a read-back row: the payload column's stored UTF-8
    bytes decoded (None stays None)."""
    if value is None:
        return None
    if isinstance(value, (bytes, bytearray, memoryview)):
        return bytes(value).decode("utf-8")
    return value


def _decode_payloads(row: dict[str, Any]) -> dict[str, Any]:
    """Decode a fetched row's BYTEA payload columns to str in place (the store's ``SELECT *`` twin:
    the message dicts downstream see stored strings, never raw bytes)."""
    for column in ("content", "api_content", "reasoning_content"):
        if row.get(column) is not None:
            row[column] = _payload_str(row[column])
    return row


# ── ported SQL constants (PG dialect) ──

_INSERT_MESSAGE_SQL = """INSERT INTO messages (session_id, role, content, tool_call_id,
        tool_calls, tool_name, effect_disposition, timestamp, token_count, finish_reason,
        reasoning, reasoning_content, reasoning_details, codex_reasoning_items,
        codex_message_items, platform_message_id, observed, _compressed_summary, active,
        api_content, display_kind, display_metadata, display_identity, message_uid,
        absorbed_message_uids, tool_call_uids, tool_call_uid)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 1,
                %s, %s, %s, %s, %s, %s, %s, %s)
        RETURNING id"""


_MESSAGE_WRITE_COLUMNS = tuple(
    re.findall(r"\w+", _INSERT_MESSAGE_SQL.split("(", 1)[1].split(")", 1)[0])
)

_MESSAGE_SCHEMA_KEYS = frozenset(_MESSAGE_WRITE_COLUMNS) | {"id", "compacted", "display_order"}

_BUMP_GENERATION_SQL = """INSERT INTO conversation_generations (source, session_key, generation)
        VALUES (%s, %s, 1)
        ON CONFLICT (source, session_key) DO UPDATE
            SET generation = conversation_generations.generation + 1"""


_TURN_LEASE_ROW_SQL = "SELECT holder, expires_at FROM session_turn_leases WHERE conversation_id = %s"


_DELETE_COMPRESSION_LOCK_SQL = "DELETE FROM compression_locks WHERE session_id = %s AND holder = %s"


_COMPRESSION_LOCK_ROW_SQL = "SELECT holder, expires_at FROM compression_locks WHERE session_id = %s"


_DISPLAY_ACTIVE_CLAUSE = " AND (active = 1 OR compacted = 1)"


# ``display_metadata`` is always valid JSON or NULL in the PG store (the writers own the encoding), so a
# plain ``::jsonb`` is non-throwing here (the SQLite twin's ``json_valid`` guard has no PG twin).
DISPLAY_VISIBLE_SQL = (
    f" AND COALESCE((display_metadata::jsonb ->> '{MODEL_ONLY_DISPLAY_METADATA_KEY}')::int, 0) = 0")


_DISPLAY_META_ROW_SQL = ("SELECT display_metadata FROM messages WHERE id = %s AND session_id IN ({ids})"
                        + _DISPLAY_ACTIVE_CLAUSE)


def not_a_later_tool_copy_sql(alias: str) -> str:
    """Drop *alias* rows that are a later copy of an archived tool result (the SQLite
    ``not_a_later_tool_copy_sql`` port; a display read paints the compacted tool result once, as its
    archived original)."""
    original_visible = (
        f"COALESCE((original.display_metadata::jsonb ->> '{MODEL_ONLY_DISPLAY_METADATA_KEY}')::int, 0) = 0")
    return (f" AND NOT ({alias}.role = 'tool' AND {alias}.message_uid IS NOT NULL AND EXISTS ("
            f"SELECT 1 FROM messages AS original WHERE original.session_id = {alias}.session_id "
            f"AND original.timestamp = {alias}.timestamp AND original.message_uid = {alias}.message_uid "
            "AND original.role = 'tool' AND original.active = 0 AND original.compacted = 1 "
            f"AND original.display_order < {alias}.display_order AND {original_visible}))")


_DISPLAY_INDEX_MISSING_SQL = ("SELECT 1 FROM messages WHERE session_id = %s" + _DISPLAY_ACTIVE_CLAUSE
                              + " AND (display_order IS NULL OR display_identity IS NULL) LIMIT 1")


_ACTIVE_IDS_SQL = "SELECT id FROM messages WHERE session_id = %s AND active = 1 ORDER BY id"


_LIVE_IDENTITY_SQL = ("SELECT id, role, content, tool_call_id, tool_calls, message_uid FROM messages "
                      "WHERE session_id = %s AND active = 1 ORDER BY id LIMIT %s")


_SET_COUNTERS_SQL = "UPDATE sessions SET message_count = %s, tool_call_count = %s"


_RESET_COUNTERS_SQL = "UPDATE sessions SET message_count = 0, tool_call_count = 0 WHERE id = %s"


_SET_DISPLAY_META_SQL = "UPDATE messages SET display_metadata = %s WHERE id = %s"


_ARCHIVE_ACTIVE_SQL = "UPDATE messages SET active = 0, compacted = 1 WHERE session_id = %s AND active = 1"


_SHADOWED_CHECKPOINT_ROWS_SQL = ("SELECT id, codex_reasoning_items FROM messages WHERE session_id = %s "
                                "AND active = 1 AND role = 'assistant' AND id < %s "
                                "AND codex_reasoning_items LIKE '%\"compaction\"%'")


_SET_CODEX_REASONING_SQL = "UPDATE messages SET codex_reasoning_items = %s WHERE id = %s"


_ENDED_ROW_SQL = "SELECT ended_at, end_reason FROM sessions WHERE id = %s"


_ROW_STATE_KEYS = ("_row_id", DB_ROW_SNAPSHOT, "timestamp")


# ── ported pure helpers ──

def _scrub(value: Any) -> Any:
    return _sanitize_surrogates(value) if isinstance(value, str) else None


def _encode_content(content: Any) -> Any:
    """Serialize list/dict content (multimodal parts) as a sentinel-prefixed JSON string; a literal
    string carrying the reserved prefix is JSON-string-wrapped so it reads back verbatim."""
    if isinstance(content, str):
        content = _sanitize_surrogates(content)
        if not content.startswith(_CONTENT_JSON_PREFIX):
            return content
    elif content is None or isinstance(content, (bytes, int, float)):
        return content
    try:
        return _CONTENT_JSON_PREFIX + json.dumps(content, ensure_ascii=True)
    except (TypeError, ValueError):
        return _sanitize_surrogates(str(content))


def _decode_content(content: Any) -> Any:
    """Reverse ``_encode_content``: a reserved-prefix string that isn't valid JSON reads back raw."""
    if isinstance(content, str) and content.startswith(_CONTENT_JSON_PREFIX):
        return _json_value_or(content[len(_CONTENT_JSON_PREFIX):], content, "raw JSON content")
    return content


def _json_value_or(raw: Any, fallback: Any, warning: str) -> Any:
    """A stored JSON column decoded to *fallback* when absent or unreadable (the store's ``_json_or``)."""
    if raw in (None, ""):
        return fallback
    if isinstance(raw, (list, dict)):
        return raw
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        return fallback


def _encode_display_metadata(display_metadata: Any) -> Optional[str]:
    """Serialize ``display_metadata`` for its TEXT column; an already-serialized JSON string is not
    double-encoded; a non-object value is refused."""
    if not display_metadata:
        return None
    if isinstance(display_metadata, str):
        display_metadata = _json_value_or(display_metadata, _INVALID, "non-JSON display metadata")
        if display_metadata is _INVALID or not isinstance(display_metadata, dict):
            return None
    elif not isinstance(display_metadata, dict):
        return None
    return json.dumps(display_metadata)


_INVALID = object()


def _decode_display_metadata(raw: Any) -> Optional[dict[str, Any]]:
    """Decode a ``display_metadata`` column to a dict (never raw TEXT: the desktop does
    ``'task_count' in meta``); a double-encoded second string layer is unwrapped, and an invalid
    payload degrades to the empty dict (the store's ``_decode_display_metadata``)."""
    if raw is None:
        return None
    meta: Any = raw
    for _ in range(2):
        if isinstance(meta, str):
            try:
                meta = json.loads(meta)
            except ValueError:
                return None
    return dict(meta) if isinstance(meta, dict) else None


def _reasoning_json_text(value: Any) -> Optional[str]:
    """A structured reasoning field for its TEXT column: strings store as-is (re-dumping would
    double-encode), structured values dump as JSON."""
    return None if not value else (value if isinstance(value, str) else json.dumps(value))


def _stale_holder(row: Mapping[str, Any], now: float) -> bool:
    """A lock/lease row whose holder is expired or a provably dead local process."""
    return float(row["expires_at"]) <= now or holder_process_is_dead(row["holder"])


def _is_explicit_fork_child_row(session: Mapping[str, Any], *, include_reset: bool = False) -> bool:
    """True when the session row is a branch, delegate, or tool child of its parent (``include_reset``:
    also a reset fork); markers only count when they point at ``parent_session_id`` (the PG twin of the
    SQLite store's check, parsed in Python over the stored JSON)."""
    if session.get("source") == "tool":
        return True
    cfg = session.get("model_config")
    if isinstance(cfg, str):
        try:
            cfg = json.loads(cfg)
        except (json.JSONDecodeError, TypeError):
            return False
    if not isinstance(cfg, dict):
        return False
    markers = (cfg.get("_branched_from"), cfg.get("_delegate_from"))
    if include_reset:
        markers += (cfg.get("_reset_from"),)
    parent_id = session.get("parent_session_id")
    return parent_id in markers if parent_id else any(m is not None for m in markers)


def _session_turn_lease_key(store, conn, session_id: str) -> str:
    """Walk compression parents on the store's connection to the conversation lease key (the SQLite
    ``_session_turn_lease_key_on_conn`` shape: the same connection, markers bind to
    ``parent_session_id``)."""
    if not session_id:
        return session_id

    def _row(sid: str) -> Optional[dict[str, Any]]:
        row = conn.execute(
            "SELECT id, parent_session_id, source, model_config, end_reason FROM sessions WHERE id = %s",
            (sid,)).fetchone()
        return dict(row) if row else None

    current = _row(session_id)
    seen = {session_id}
    while current:
        parent_id = current.get("parent_session_id")
        if not parent_id or parent_id in seen or _is_explicit_fork_child_row(current, include_reset=True):
            break
        parent = _row(parent_id)
        if not parent or parent.get("end_reason") != "compression":
            break
        seen.add(parent_id)
        current = parent
    return str(current.get("id") or session_id) if current else session_id


def _check_transcript_write_guards(
        conn, session_id: str, compression_lock_holder: Optional[str] = None,
        turn_lease_holder: Optional[str] = None, turn_lease_ttl_seconds: float = 300.0,
        reject_active_turn_lease: bool = False, reject_active_compression_lock: bool = False,
        allow_closed_compression_parent: bool = False) -> None:
    """Transcript-write admission checks, run INSIDE the write transaction by every writer (the SQLite
    ``_check_transcript_write_guards`` port): an active unowned turn lease is reclaimed or refused, a
    named lease that expired is renewed, an active compression lock is refused to a non-holder, and a
    session ended by compression refuses (unless its parent's compression close is allowed)."""
    if reject_active_compression_lock:
        active_lock = conn.execute(_COMPRESSION_LOCK_ROW_SQL, (session_id,)).fetchone()
        if active_lock is not None:
            if _stale_holder(active_lock, time.time()):
                conn.execute(_DELETE_COMPRESSION_LOCK_SQL, (session_id, active_lock["holder"]))
            elif active_lock["holder"] != compression_lock_holder:
                raise SessionCompressionInProgressError(
                    f"Session {session_id!r} is being compressed by another writer")
    if turn_lease_holder or reject_active_turn_lease:
        conversation_id = _lease_key(conn, session_id)
        lease = conn.execute(_TURN_LEASE_ROW_SQL, (conversation_id,)).fetchone()
        now = time.time()
        if turn_lease_holder:
            if lease is None or lease["holder"] != turn_lease_holder:
                raise SessionTurnLeaseLostError(
                    f"Session turn lease lost; refusing transcript write for {session_id!r}")
            if float(lease["expires_at"]) <= now:
                conn.execute("UPDATE session_turn_leases SET expires_at = %s "
                             "WHERE conversation_id = %s AND holder = %s",
                             (now + max(0.1, float(turn_lease_ttl_seconds)), conversation_id,
                              turn_lease_holder))
        elif lease is not None:
            if not _stale_holder(lease, now):
                raise SessionTurnLeaseLostError(
                    f"Session has an active turn lease; refusing transcript mutation for {session_id!r}")
            conn.execute("DELETE FROM session_turn_leases WHERE conversation_id = %s AND holder = %s",
                         (conversation_id, lease["holder"]))
    if _ended_by_compression(conn.execute(_ENDED_ROW_SQL, (session_id,)).fetchone()) \
            and not allow_closed_compression_parent:
        raise CompressionSessionClosedError(session_id)
    # Last: the session row itself. In SQLite the insert's foreign key finds it; the DDL keeps the FK,
    # but its violation maps to a generic store error, so the writer names the refusal itself - after the
    # fences, so a lost lease or a compression-closed session still wins.
    if conn.execute("SELECT 1 FROM sessions WHERE id = %s", (session_id,)).fetchone() is None:
        raise SessionRowMissing(SessionId(session_id))


def _lease_key(conn, session_id: str) -> str:
    """The lease-key walk on a bare connection (``_session_turn_lease_key`` without the store)."""
    return _session_turn_lease_key(store=None, conn=conn, session_id=session_id)


def _message_row_params(session_id: str, role: str, msg: dict[str, Any], tool_calls: Any,
                       message_timestamp: float, *, keep_reasoning: bool = True) -> tuple:
    """The insert's bound values in ``_INSERT_MESSAGE_SQL`` column order (the SQLite
    ``_message_row_params`` port; payload columns bound as BYTEA)."""
    encoded_content = _encode_content(msg.get("content"))
    encoded_tool_calls = json.dumps(tool_calls) if tool_calls else None
    display_metadata = msg.get("display_metadata")
    if msg.get(DATA_USE) and isinstance(display_metadata, (dict, type(None))):
        display_metadata = {**(display_metadata or {}), DATA_USE: list(msg[DATA_USE])}
    display_metadata = _encode_display_metadata(display_metadata)
    identity_row = {
        "role": role, "content": encoded_content, "timestamp": message_timestamp,
        "tool_call_id": msg.get("tool_call_id"), "tool_calls": encoded_tool_calls,
        "tool_name": _scrub(msg.get("tool_name")), "display_kind": msg.get("display_kind"),
        "display_metadata": display_metadata,
    }
    params = [
        session_id, role, _payload_bytes(encoded_content), msg.get("tool_call_id"),
        encoded_tool_calls, _scrub(msg.get("tool_name")),
        msg.get("effect_disposition"), message_timestamp, msg.get("token_count"), msg.get("finish_reason"),
        _scrub(msg.get("reasoning")), _payload_bytes(_scrub(msg.get("reasoning_content"))),
        _reasoning_json_text(msg.get("reasoning_details")),
        _reasoning_json_text(msg.get("codex_reasoning_items")),
        _reasoning_json_text(msg.get("codex_message_items")),
        msg.get("platform_message_id") or msg.get("message_id"),
        1 if msg.get("observed") else 0, 1 if msg.get("_compressed_summary") else 0,
        _payload_bytes(_scrub(msg.get("api_content"))), _scrub(msg.get("display_kind")),
        display_metadata, _display_identity(_display_dedupe_key(identity_row)),
        message_uid_or_none(msg), _absorbed_uids_json(msg),
        _tool_call_uids_json(msg), _tool_call_uid_or_none(msg),
    ]
    if not keep_reasoning:
        for i in (10, 11, 12, 13, 14):
            params[i] = None
    return tuple(params)


def _serialized_message_row(session_id: str, msg: dict[str, Any], message_timestamp: float) -> dict[str, Any]:
    """Serialize one live message with the exact durable shape used by INSERT (the SQLite
    ``_serialized_message_row``)."""
    role = msg.get("role", "unknown")
    params = _message_row_params(session_id, role, msg, _parse_tool_calls(msg.get("tool_calls")),
                                 message_timestamp, keep_reasoning=role == "assistant")
    return dict(zip(_MESSAGE_WRITE_COLUMNS, params))


def _bump_session_counters(conn, session_id: str, inserted: int, tool_calls: int, *, unit: bool = False) -> None:
    """Bump sessions.* counters after an insert; *unit* bakes the ``+ 1`` literal into the SQL."""
    if tool_calls > 0:
        if unit:
            conn.execute(
                "UPDATE sessions SET message_count = message_count + 1, "
                "tool_call_count = tool_call_count + %s WHERE id = %s",
                (tool_calls, session_id))
        else:
            conn.execute(
                "UPDATE sessions SET message_count = message_count + %s, "
                "tool_call_count = tool_call_count + %s WHERE id = %s",
                (inserted, tool_calls, session_id))
    elif inserted > 0:
        conn.execute(
            "UPDATE sessions SET message_count = message_count + "
            + ("1" if unit else "%s") + " WHERE id = %s",
            (session_id,) if unit else (inserted, session_id))


def _stamp_tool_call_uids(msg: dict[str, Any], tool_calls: Any, batch_index: dict[str, str]) -> None:
    """Per-occurrence tool-call identity at insert (the SQLite ``_stamp_tool_call_uids``)."""
    role = msg.get("role")
    if role == "assistant" and tool_calls:
        uids = _tool_call_uid_map(msg)
        for tc in tool_calls:
            call_id = coalesce_tool_call_id(tc)
            if call_id and call_id not in uids:
                uids[call_id] = mint_uid()
        if uids:
            msg["tool_call_uids"] = uids
        index_tool_call_uids(batch_index, msg)
    elif role == "user":
        batch_index.clear()
    elif role == "tool" and _tool_call_uid_or_none(msg) is None:
        uid = resolve_tool_call_uid(batch_index, msg.get("tool_call_id"))
        if uid:
            msg[TOOL_CALL_UID] = uid


def _loaded_view_content(role: str, content: Any) -> Any:
    """Content as the conversation readers hand it to callers: user/assistant strings are sanitized
    and stripped on load, so a reloaded session's messages must be compared to rows through the
    same lens."""
    if role in {"user", "assistant"} and isinstance(content, str):
        return sanitize_context(content).strip()
    return content


def _row_identity(role: str, content: Any, tool_call_id: Any, tool_calls: Any) -> tuple:
    """The (role, content, tool_call_id, tool_calls) columns a message writes — the identity the
    kept-prefix match compares. *content* is passed through the loader's lens first."""
    return (role, _encode_content(_loaded_view_content(role, content)), tool_call_id,
            json.dumps(tool_calls) if tool_calls else None)


def _stamp_kept_live_prefix(live: list, messages: list[dict[str, Any]]) -> int:
    """Length of the in-order prefix of *messages* already present as the leading live rows; each
    matched message gets its existing ``_row_id`` (and ``message_uid``) stamped, as a fresh insert
    would set them."""
    kept = 0
    for row, msg in zip(live, messages):
        role = msg.get("role", "unknown")
        # Cheap scalars first; the content compare pays decode/sanitize/encode on both sides.
        if row["role"] != role or row["tool_call_id"] != msg.get("tool_call_id"):
            break
        stored_content = _payload_str(row["content"])
        identity = _row_identity(role, msg.get("content"), msg.get("tool_call_id"),
                                 _parse_tool_calls(msg.get("tool_calls")))
        if identity != _row_identity(row["role"], _decode_content(stored_content), row["tool_call_id"],
                                     _parse_tool_calls(row["tool_calls"])):
            break
        msg["_row_id"] = row["id"]
        if row["message_uid"]:
            msg["message_uid"] = row["message_uid"]
        kept += 1
    return kept


def _carry_parent_timestamps(conn, parent_session_id: str, messages: list[dict[str, Any]]) -> None:
    """Adopt the durable parent row's timestamp onto carried handoff rows so the re-inserted child row
    keeps the original's display identity (see ``_display_dedupe_key``). Idempotent; best-effort."""
    try:
        candidates = [i for i, msg in enumerate(messages)
            if isinstance(msg, dict) and coerce_epoch(msg.get("timestamp"), field="message timestamp") is None]
        if not candidates:
            return
        donors: dict[tuple[Any, ...], list[Any]] = {}
        for row in conn.execute(
                "SELECT id, role, content, timestamp, tool_call_id, tool_calls, tool_name FROM messages "
                "WHERE session_id = %s AND active = 1 ORDER BY id", (parent_session_id,)).fetchall():
            donors.setdefault((row["role"], _payload_str(row["content"]), row["tool_call_id"],
                               row["tool_calls"], row["tool_name"]), []).append(row["timestamp"])
        for i in candidates:
            msg = messages[i]
            tool_calls = _parse_tool_calls(msg.get("tool_calls"))
            key = (msg.get("role", "unknown"), _encode_content(msg.get("content")),
                   msg.get("tool_call_id"), json.dumps(tool_calls) if tool_calls else None,
                   _scrub_surrogates(msg.get("tool_name")))
            queue = donors.get(key)
            if queue:
                msg["timestamp"] = queue.pop(0)
    except Exception:  # health: allow BLE001 -- a failed carry keeps the publish-time stamps
        return  # Best-effort: a failed carry falls back to the publish-time stamp, never breaks publish.


def _display_dedupe_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    """Historical display identity, including normalized live content from user handoff carriers
    (the SQLite ``_display_dedupe_key`` port: the row's content is its stored encoded form)."""
    dedupe_content = row["content"]
    if row["role"] == "user":
        handoff, live_view = split_user_originated_turn({
            "role": "user", "content": _decode_content(row["content"]),
            "display_kind": row.get("display_kind"),
            "display_metadata": _decode_display_metadata(row.get("display_metadata"))})
        if handoff is not None and live_view is not None:
            dedupe_content = _encode_content(live_view.get("content"))
    return _stable_tool_key(row) or (
        row["role"], dedupe_content, row["timestamp"], row["tool_call_id"], row["tool_calls"],
        row["tool_name"])


def _is_model_only_row(row: Mapping[str, Any]) -> bool:
    """Python twin of :data:`DISPLAY_VISIBLE_SQL`."""
    return bool((_decode_display_metadata(row.get("display_metadata")) or {}).get(MODEL_ONLY_DISPLAY_METADATA_KEY))


def _display_identity(key: tuple[Any, ...]) -> bytes:
    """Fixed-width durable identity for indexed display-generation lookup (sha256 of the key's repr,
    the SQLite store's shape)."""
    return hashlib.sha256(repr(key).encode("utf-8", "surrogatepass")).digest()


def _fold_summary_carrier(row: Mapping[str, Any], key: Any, first_by_uid: dict) -> Any:
    """The display identity both display reads give *row*, scanning rows in ``id`` order: a compaction
    carrier keeps its message's ``message_uid`` but not its content, so it is a copy of the first row
    with that uid (shown once)."""
    uid = row.get("message_uid")
    if not uid:
        return key
    if row.get("_compressed_summary"):
        return first_by_uid.get(uid, key)
    first_by_uid.setdefault(uid, key)
    return key


def _reconcile_display_orders(conn, session_id: str) -> None:
    """Re-fold split display generations by the recomputed display key (the SQLite
    ``_reconcile_display_orders`` port): one logical message that a content update or a compaction
    generation left in two ``display_order`` groups is folded back onto the identity's first row; the
    live copy wins its group via the read path's ``ORDER BY candidate.active DESC, candidate.id DESC``.
    Writes only on drift."""
    first_id: dict[bytes, int] = {}
    uid_identity: dict[str, bytes] = {}  # message_uid -> its first copy's identity (the carrier rule)
    rows = conn.execute(
        "SELECT id, role, content, timestamp, tool_call_id, tool_calls, tool_name, "
        "display_kind, display_metadata, display_order, display_identity, message_uid, "
        "_compressed_summary FROM messages WHERE session_id = %s AND (active = 1 OR compacted = 1) "
        "ORDER BY id", (session_id,)).fetchall()
    updates = []
    for row in rows:
        row = _decode_payloads(dict(row))
        identity = _fold_summary_carrier(
            row, _display_identity(_display_dedupe_key(row)), uid_identity)
        # A row already holding this identity keeps its stored slot (an in-place prune changes a
        # row's content, not its place); otherwise the slot is the identity's first row.
        kept = row["display_order"] if row["display_identity"] == identity else None
        order = first_id.setdefault(identity, row["id"] if kept is None else kept)
        if order != row["display_order"] or identity != row["display_identity"]:
            updates.append((order, identity, row["id"]))
    for order, identity, row_id in updates:
        conn.execute("UPDATE messages SET display_order = %s, display_identity = %s WHERE id = %s",
                     (order, identity, row_id))


def _ensure_display_index(store, session_id: str) -> bool:
    """Backfill one session's display index (the PG twin of ``_ensure_display_order``): complete when no
    displayed row is missing a slot; otherwise fold the missing ones in a write and stay complete. A
    read-only store cannot backfill (False)."""
    if store.one(_DISPLAY_INDEX_MISSING_SQL, session_id) is None:
        return True
    if store.read_only:
        return False
    with store.tx() as conn:
        _reconcile_display_orders(conn, session_id)
    return True


def _display_rows_sql(session_id: str, *, limit: Optional[int] = None, offset: int = 0,
                     latest: bool = False, after: Optional[int] = None) -> tuple[str, list]:
    """The display-projection statement (``%s``-bound; run on any connection of the store's schema):
    representative rows per ``display_order`` group, keyset-paged past *after* (before it when *latest*)."""
    direction = "DESC" if latest else "ASC"
    keyset = "" if after is None else f" AND display_order {'<' if latest else '>'} %s"
    sql = f"""WITH page AS (
                SELECT display_order FROM messages
                WHERE session_id = %s AND (active = 1 OR compacted = 1){DISPLAY_VISIBLE_SQL}{not_a_later_tool_copy_sql("messages")}{keyset}
                GROUP BY display_order ORDER BY display_order {direction}
                LIMIT %s OFFSET %s
           )
           SELECT chosen.* FROM page
           JOIN messages AS chosen ON chosen.id = (
               SELECT candidate.id FROM messages AS candidate
               WHERE candidate.session_id = %s
                 AND candidate.display_order = page.display_order
                 AND (candidate.active = 1 OR candidate.compacted = 1){DISPLAY_VISIBLE_SQL}{not_a_later_tool_copy_sql("candidate")}
               ORDER BY candidate.active DESC, candidate.id DESC LIMIT 1
           )
           ORDER BY page.display_order ASC"""
    params: list = [session_id]
    if after is not None:
        params.append(int(after))
    params.extend([None if limit is None else int(limit), int(offset), session_id])
    return sql, params


def _display_rows(store, session_id: str, *, limit: Optional[int] = None, offset: int = 0,
                  latest: bool = False, after: Optional[int] = None) -> list[dict[str, Any]]:
    """One display-history projection (the SQLite ``_display_rows_from_conn`` port): representative rows
    per ``display_order`` group, keyset-paged past *after* (before it when *latest*). The index is
    backfilled first; the DDL triggers keep it complete, so the read is the pure indexed path."""
    _ensure_display_index(store, session_id)
    sql, params = _display_rows_sql(session_id, limit=limit, offset=offset, latest=latest, after=after)
    return _decode_payloads_rows(store.all(sql, *params))


def _decode_payloads_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [_decode_payloads(row) for row in rows]


def _prune_shadowed_checkpoints(conn, session_id: str, live_messages: list[dict[str, Any]]) -> None:
    """Keep only the newest checkpoint among *live_messages* (inserted rows carrying ``_row_id``)."""
    carrier = _newest_checkpoint_carrier(live_messages, "codex_reasoning_items")
    if carrier >= 0 and isinstance(live_messages[carrier].get("_row_id"), int):
        for row_id, raw in conn.execute(
                _SHADOWED_CHECKPOINT_ROWS_SQL, (session_id, live_messages[carrier]["_row_id"])).fetchall():
            items = _json_or(raw, None, "Ignoring malformed codex_reasoning_items on message row")
            if not isinstance(items, list) or not any(_is_checkpoint_item(item) for item in items):
                continue
            kept = [item for item in items if not _is_checkpoint_item(item)]
            conn.execute(_SET_CODEX_REASONING_SQL, (_reasoning_json_text(kept), row_id))


def _clone_message_rows(conn, tail_ids: list[int], *, session_id: Optional[str] = None) -> None:
    """Pure-SQL clone of *tail_ids* as fresh live rows (new id/display order, active=1, compacted=0;
    payload columns byte-exact), into *session_id* when given. The DDL insert triggers assign the
    clones' display order from their carried identity."""
    clone_cols = [c for c in _MESSAGE_WRITE_COLUMNS if c not in ("session_id", "active")]
    col_list = ", ".join(clone_cols)
    select_cols = ", ".join(clone_cols)
    if session_id is not None:
        sql = (f"INSERT INTO messages ({col_list}, session_id, active, compacted) "
               f"SELECT {select_cols}, %s, 1, 0 FROM messages WHERE id IN ({_in_placeholders(tail_ids)}) "
               "ORDER BY id")
        conn.execute(sql, (session_id, *tail_ids))
    else:
        sql = (f"INSERT INTO messages ({col_list}, active, compacted) "
               f"SELECT {select_cols}, 1, 0 FROM messages WHERE id IN ({_in_placeholders(tail_ids)}) "
               "ORDER BY id")
        conn.execute(sql, tuple(tail_ids))


def _row_to_message_dict(row: Mapping[str, Any], *, warn_context: str = "",
                         summary_flag: bool = False) -> dict[str, Any]:
    """``dict(row)`` with the payload columns decoded and the display-index columns dropped (the
    SQLite ``_row_to_message_dict`` port; unknown binary columns are dropped once so the JSON wire
    encoder never sees raw bytes)."""
    msg = dict(row)
    msg.pop("display_identity", None)
    msg.pop("display_order", None)
    for column in ("content", "api_content", "reasoning_content"):
        if msg.get(column) is not None:
            msg[column] = _payload_str(msg[column])
    if summary_flag and msg.pop("_compressed_summary", 0):
        msg["_compressed_summary"] = True
    msg["content"] = _decode_content(msg["content"])
    if msg.get("tool_calls"):
        msg["tool_calls"] = _json_or(msg["tool_calls"], [],
                                     f"Failed to deserialize tool_calls in {warn_context}, falling back to []")
    if msg.get("display_metadata") is not None:
        msg["display_metadata"] = _decode_display_metadata(msg["display_metadata"])
    for key, value in list(msg.items()):
        if key not in _MESSAGE_SCHEMA_KEYS and isinstance(value, (bytes, bytearray, memoryview)):
            msg.pop(key)
    return msg


def _transcript_write(store, messages: list[dict[str, Any]], fn) -> Any:
    """``store.tx()`` for callbacks that stamp row state onto the caller's *messages* (every
    ``_insert_message_rows`` caller that passes caller-owned dicts): each entry point restores the
    caller's ``_row_id`` / digest / timestamp so a rolled-back write never leaves a stale stamp a
    later flush would adopt (the SQLite ``_execute_transcript_write`` port)."""
    _absent = object()
    pre_state = [tuple(m.get(k, _absent) for k in _ROW_STATE_KEYS) for m in messages]

    def _restore() -> None:
        for msg, state in zip(messages, pre_state):
            msg.pop(CANONICAL_ROW, None)
            for key, value in zip(_ROW_STATE_KEYS, state):
                if value is _absent:
                    msg.pop(key, None)
                else:
                    msg[key] = value

    try:
        with store.tx() as conn:
            _restore()
            return fn(conn)
    except BaseException:
        _restore()
        raise


def _load_transcript_write(store, messages: list[dict[str, Any]], fn) -> Any:
    """``_transcript_write`` without the caller-dict stamp restore (fresh-copy callers)."""
    with store.tx() as conn:
        return fn(conn)

"""``MessageRepository`` on PostgreSQL: the slice verbs ported from the SQLite reference (``append``,
``read_transcript``); every other messages verb raises ``NotImplementedError`` naming itself (D8).

``append`` is the critical transcript write: the uid stamping (``message_uid`` minted once, the tool-call
uids resolved against the batch), the turn-lease guard (an expired lease still naming the holder is
renewed, otherwise ``TurnLeaseLost``), the closed-by-compression refusal (``CompressionClosed``), the
missing-session-row refusal (``SessionRowMissing``), the row insert and the session counter bump, all
in one transaction on the store's connection. ``read_transcript`` reads rows in insertion order
(id, never timestamp) through the transcript row projection.

Porting notes (recorded for the stage-1 review): the display identity here is the plain dedupe key
(role, content, timestamp, tool_call_id, tool_calls, tool_name, with the stable tool-call key for
assistant rows) without the user-turn handoff-carrier normalization the SQLite store applies before
keying - the display-generation parity suites (stage 1) decide whether that normalization must be
ported too. The turn-lease conversation key walks compression parents in Python over the store's
connection, the SQLite store's ``_session_turn_lease_key_on_conn`` shape.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from typing import Any, Mapping, Optional

from persistence import CompressionClosed, MessageId, RowVisibility, SessionRowMissing, SessionId, TurnLeaseLost
from persistence.conversation.messages import (
    MessageRepository, NewMessage, TranscriptRead, TranscriptRow, TranscriptView,
)
from persistence.sqlite.conversation._store.core import _CONTENT_JSON_PREFIX
from persistence.sqlite.conversation._store.identity import (
    _absorbed_uids_json, _stable_tool_key, _tool_call_uid_map, _tool_call_uid_or_none, _tool_call_uids_json,
)
from persistence.sqlite.conversation._store.messages_sql import (
    _coerce_timestamp, _parse_tool_calls, _tool_calls_count,
)
from persistence.sqlite.conversation._store.rules import (
    DATA_USE, _sanitize_surrogates, coerce_epoch, stamp_message_uid,
)
from session_rules.turns import (
    TOOL_CALL_UIDS, coalesce_tool_call_id, index_tool_call_uids, message_uid_or_none, mint_uid,
    resolve_tool_call_uid,
)
from psycopg import errors as pg_errors

from .._backend import PostgresStore, register_repository, translated
from .._stubs import add_stubs

__all__ = ["PgMessageRepository"]

_INVALID = object()
# The sessions row the transcript-write guard reads (the SQLite ``sql_common`` shape on the PG dialect).
_ENDED_ROW_SQL = "SELECT ended_at, end_reason FROM sessions WHERE id = %s"


def _decode_display_metadata(raw: Any) -> Optional[dict[str, Any]]:
    """Decode a ``display_metadata`` column to a dict (never raw TEXT: the desktop does ``'task_count'
    in meta``); a double-encoded second string layer is unwrapped, and an invalid payload degrades to
    the empty dict (the store's ``_decode_display_metadata``)."""
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


def _row_to_message_dict(row: dict[str, Any], *, summary_flag: bool = False) -> dict[str, Any]:
    """``dict(row)`` with content/tool_calls/display_metadata decoded (the store's
    ``_row_to_message_dict``: the display-index columns dropped, binary columns dropped,
    ``_compressed_summary`` kept only as ``True``)."""
    msg = dict(row)
    msg.pop("display_identity", None)
    msg.pop("display_order", None)
    if summary_flag and msg.pop("_compressed_summary", 0):
        msg["_compressed_summary"] = True
    msg["content"] = _decode_content(msg["content"])
    if msg.get("tool_calls"):
        try:
            msg["tool_calls"] = json.loads(msg["tool_calls"])
        except (TypeError, ValueError):
            msg["tool_calls"] = []
    if msg.get("display_metadata") is not None:
        msg["display_metadata"] = _decode_display_metadata(msg["display_metadata"])
    for key, value in list(msg.items()):
        if isinstance(value, (bytes, bytearray)) and key not in ("content", "display_identity"):
            msg.pop(key)
    return msg


def _transcript_row(row: Mapping[str, Any]) -> TranscriptRow:
    """A stored row dict (read back with the display columns decoded) as its record (the reference's
    ``_transcript_row`` projection: the wire fields plus the visibility and summary flag)."""
    from persistence.sqlite.conversation.messages import _WIRE_FIELDS

    def _visibility(active: Any, compacted: Any) -> RowVisibility:
        if active:
            return RowVisibility.ACTIVE
        return RowVisibility.COMPACTION_ARCHIVED if compacted else RowVisibility.REWOUND

    fields = {name: row.get(name) for name in _WIRE_FIELDS}
    return TranscriptRow(
        id=MessageId(int(row["id"])), session_id=SessionId(row["session_id"]),
        visibility=_visibility(row.get("active"), row.get("compacted")),
        observed=bool(row.get("observed")),
        compressed_summary=row.get("_compressed_summary") is True,
        **fields,
    )
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


def _reasoning_json_text(value: Any) -> Optional[str]:
    """A structured reasoning field for its TEXT column: strings store as-is (re-dumping would
    double-encode), structured values dump as JSON."""
    return None if not value else (value if isinstance(value, str) else json.dumps(value))


def _scrub(value: Any) -> Any:
    return _sanitize_surrogates(value) if isinstance(value, str) else None


def _display_dedupe_key(row: dict[str, Any]) -> tuple[Any, ...]:
    """The display identity key (stable tool key for complete assistant call sets, else the content
    key)."""
    return _stable_tool_key(row) or (
        row["role"], row["content"], row["timestamp"], row["tool_call_id"], row["tool_calls"], row["tool_name"]
    )


def _display_identity(key: tuple[Any, ...]) -> bytes:
    """The fixed-width durable identity for indexed display-generation lookup (the SQLite store's
    sha256 of the key's repr)."""
    return hashlib.sha256(repr(key).encode("utf-8", "surrogatepass")).digest()


def _is_explicit_fork_child_row(session: dict[str, Any], *, include_reset: bool = False) -> bool:
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


def _session_turn_lease_key(store: PostgresStore, conn, session_id: str) -> str:
    """Walk compression parents on the store's connection to the conversation lease key (the SQLite
    ``_session_turn_lease_key_on_conn`` shape: the same connection, markers bind to ``parent_session_id``)."""
    if not session_id:
        return session_id

    def _row(sid: str) -> Optional[dict[str, Any]]:
        row = conn.execute(
            "SELECT id, parent_session_id, source, model_config, end_reason FROM sessions WHERE id = %s", (sid,)
        ).fetchone()
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


class PgMessageRepository:
    """``MessageRepository`` verbs on the bound profile's store schema; unported verbs raise (D8)."""

    def __init__(self, store: PostgresStore) -> None:
        self._store = store

    # ── the slice verbs ──

    def append(self, session_id: SessionId, message: NewMessage, *,
               turn_lease_holder: Optional[str] = None, turn_lease_ttl_seconds: float = 300.0) -> MessageId:
        """Append one row and bump the session counters in one transaction; returns its id (the protocol
        docstring's refusals: ``TurnLeaseLost``, ``CompressionClosed``, ``SessionRowMissing``)."""
        store = self._store
        msg = self._message_dict(message)
        with store.tx() as conn:
            session_row = conn.execute(
                "SELECT id FROM sessions WHERE id = %s", (session_id,)
            ).fetchone()
            if session_row is None:
                raise SessionRowMissing(SessionId(session_id))
            self._check_transcript_write_guards(conn, session_id,
                                                turn_lease_holder=turn_lease_holder,
                                                turn_lease_ttl_seconds=turn_lease_ttl_seconds)
            # Encode outside the statement (display metadata first, log-order parity).
            msg["display_metadata"] = _encode_display_metadata(msg.get("display_metadata"))
            tool_calls = _parse_tool_calls(msg.get("tool_calls"))
            message_timestamp = _coerce_timestamp(message.timestamp, time.time())
            stamp_message_uid(msg)
            self._stamp_tool_call_uids(msg, tool_calls, {})
            params = self._message_row_params(session_id, message.role, msg, tool_calls, message_timestamp)
            row = conn.execute(
                """INSERT INTO messages (session_id, role, content, tool_call_id,
                        tool_calls, tool_name, effect_disposition, timestamp, token_count, finish_reason,
                        reasoning, reasoning_content, reasoning_details, codex_reasoning_items,
                        codex_message_items, platform_message_id, observed, _compressed_summary, active,
                        api_content, display_kind, display_metadata, display_identity, message_uid,
                        absorbed_message_uids, tool_call_uids, tool_call_uid)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 1,
                           %s, %s, %s, %s, %s, %s, %s, %s)
                   RETURNING id""",
                params,
            ).fetchone()
            self._bump_session_counters(conn, session_id, 1, _tool_calls_count(tool_calls))
        return MessageId(int(row["id"]))

    def read_transcript(self, session_id: SessionId, request: TranscriptRead = TranscriptRead()) -> tuple[TranscriptRow, ...]:
        """The session's transcript rows as records, in insertion order (id, never timestamp). The slice read is
        the ACTIVE view; ``request``'s paging and the other views come with the stage-1 port."""
        store = self._store
        rows = store.all("SELECT * FROM messages WHERE session_id = %s AND active = 1 ORDER BY id ASC", session_id)
        for row in rows:
            for column in ("content", "api_content", "reasoning_content"):
                row[column] = _payload_str(row.get(column))
        return tuple(_transcript_row(_row_to_message_dict(row, summary_flag=True)) for row in rows)

    # ── append's helpers (the SQLite store functions, ported) ──

    def _message_dict(self, message: NewMessage) -> dict[str, Any]:
        """One ``NewMessage`` as the message dict the writers take (the SQLite ``_message_dict``)."""
        from persistence.sqlite.conversation.messages import _MESSAGE_OPTIONALS

        msg: dict[str, Any] = {"role": message.role, "content": message.content}
        if message.timestamp is not None:
            msg["timestamp"] = message.timestamp
        if message.tool_calls:
            msg["tool_calls"] = list(message.tool_calls)
        for name in _MESSAGE_OPTIONALS:
            value = getattr(message, name)
            if value is not None:
                msg[name] = value
        if message.display_metadata is not None:
            msg["display_metadata"] = dict(message.display_metadata)
        if message.compressed_summary:
            msg["_compressed_summary"] = True
        if message.observed:
            msg["observed"] = True
        if message.repair_target is not None:
            msg["_row_id"] = int(message.repair_target)
        if message.db_row_snapshot is not None:
            msg["db_row_snapshot"] = message.db_row_snapshot
        if message.absorbed_message_uids:
            msg["_absorbed_message_uids"] = list(message.absorbed_message_uids)
        if message.tool_call_uids is not None:
            msg[TOOL_CALL_UIDS] = dict(message.tool_call_uids)
        return msg

    def _check_transcript_write_guards(self, conn, session_id: str, *,
                                      turn_lease_holder: Optional[str] = None,
                                      turn_lease_ttl_seconds: float = 300.0) -> None:
        """The transcript-write admission checks, run inside the write transaction: a named turn lease
        that no longer names the holder (or an active lease on an unfenced write) is a refusal, an
        expired lease still naming the holder is renewed; a session ended by compression refuses."""
        if turn_lease_holder:
            conversation_id = _session_turn_lease_key(self._store, conn, session_id)
            lease = conn.execute(
                "SELECT holder, expires_at FROM session_turn_leases WHERE conversation_id = %s",
                (conversation_id,),
            ).fetchone()
            now = time.time()
            if lease is None or lease["holder"] != turn_lease_holder:
                raise TurnLeaseLost(f"Session turn lease lost; refusing transcript write for {session_id!r}")
            if float(lease["expires_at"]) <= now:
                conn.execute(
                    "UPDATE session_turn_leases SET expires_at = %s WHERE conversation_id = %s AND holder = %s",
                    (now + max(0.1, float(turn_lease_ttl_seconds)), conversation_id, turn_lease_holder),
                )
        ended = conn.execute(_ENDED_ROW_SQL, (session_id,)).fetchone()
        if _ended_by_compression(ended):
            raise CompressionClosed(str(session_id))

    def _message_row_params(self, session_id: str, role: str, msg: dict[str, Any], tool_calls: Any,
                           message_timestamp: float) -> tuple:
        """The insert's bound values (the SQLite ``_message_row_params`` column order)."""
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
        return (
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
        )

    def _stamp_tool_call_uids(self, msg: dict[str, Any], tool_calls: Any, batch_index: dict[str, str]) -> None:
        """Per-occurrence tool-call identity at insert (the SQLite ``_stamp_tool_call_uids``)."""
        role = msg.get("role")
        if role == "assistant" and tool_calls:
            uids = _tool_call_uid_map(msg)
            for tc in tool_calls:
                call_id = coalesce_tool_call_id(tc)
                if call_id and call_id not in uids:
                    uids[call_id] = mint_uid()
            if uids:
                msg[TOOL_CALL_UIDS] = uids
            index_tool_call_uids(batch_index, msg)
        elif role == "user":
            batch_index.clear()
        elif role == "tool" and _tool_call_uid_or_none(msg) is None:
            # A tool-result row resolves its call id through the batch index (the same batch's
            # assistant rows), any id variant the SQLite ``resolve_tool_call_uid`` walks.
            uid = resolve_tool_call_uid(batch_index, msg.get("tool_call_id"))
            if uid:
                msg["_tool_call_uid"] = uid

    def _bump_session_counters(self, conn, session_id: str, inserted: int, tool_calls: int) -> None:
        """Bump the session's message and tool-call counters after an insert (the SQLite
        ``_bump_session_counters``)."""
        if tool_calls > 0:
            conn.execute(
                "UPDATE sessions SET message_count = message_count + 1, "
                "tool_call_count = tool_call_count + %s WHERE id = %s",
                (tool_calls, session_id),
            )
        elif inserted > 0:
            conn.execute(
                "UPDATE sessions SET message_count = message_count + 1 WHERE id = %s",
                (session_id,),
            )


def _ended_by_compression(row) -> bool:
    """A session row ended by a compression boundary (the shared read-side check)."""
    return row is not None and row["ended_at"] is not None and row["end_reason"] == "compression"


add_stubs(PgMessageRepository, MessageRepository, implemented=("append", "read_transcript"))


def _register() -> None:
    register_repository("messages", lambda store: PgMessageRepository(store))


_register()

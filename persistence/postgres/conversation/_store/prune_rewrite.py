"""Held-row staleness checks and the in-place prune rewrite (the SQLite ``prune_rewrite.py``
port, connection-first, ``%s`` placeholders, and the bytea payload rule: stored ``content`` is
bound as UTF-8 bytes and rows are payload-decoded before the CAS digest and identity helpers)."""

from __future__ import annotations

from typing import Any, Optional

from session_rules.turns import _DB_PERSISTED_MARKER as _DB_PERSISTED_MARKER_KEY, message_uid_or_none
from persistence.sqlite.conversation._store.messages_sql import _parse_tool_calls
from persistence.sqlite.conversation._store.rules import ABSORBED_ROW_IDS, DB_ROW_SNAPSHOT
from persistence.sqlite.conversation._store.sql_common import _id_chunks
from persistence.sqlite._state.errors import PruneRowStaleError, PruneRowUnresolvedError
from .common import (
    _MESSAGE_WRITE_COLUMNS, _check_transcript_write_guards, _decode_payloads, _display_dedupe_key,
    _display_identity, _ensure_display_index, _in_placeholders, _message_row_params, _row_identity,
    _transcript_write,
)
from .coverage import _matching_active_ids
from .repair import transcript_row_snapshot
from .sessions_pg import _merge_model_config_json

__all__ = ["held_row_mismatches", "rewrite_pruned_rows", "rewritten_held_row_ids"]

# Every messages column except the twin's fixed ones (the SQLite ``_message_column_names`` twin;
# Postgres has no PRAGMA table_info, so the DDL column list is inlined).
_MESSAGE_ROW_COLUMNS = _MESSAGE_WRITE_COLUMNS

_ALL_MESSAGE_COLUMNS = (
    "session_id", "role", "content", "tool_call_id", "tool_calls", "tool_name",
    "effect_disposition", "timestamp", "token_count", "finish_reason", "reasoning",
    "reasoning_content", "reasoning_details", "codex_reasoning_items", "codex_message_items",
    "platform_message_id", "observed", "_compressed_summary", "api_content", "display_kind",
    "display_metadata", "display_identity", "message_uid", "absorbed_message_uids",
    "tool_call_uids", "tool_call_uid",
)

_PRUNE_FIXED_COLUMNS = frozenset({"session_id", "role", "tool_call_id", "timestamp", "active", "display_identity"})


def rewritten_held_row_ids(store, session_id: str, held: list[dict[str, Any]]) -> list[int]:
    """Ids among *held* message dicts whose ACTIVE row no longer holds the payload the dict holds
    (the SQLite ``rewritten_held_row_ids`` port, store-first): an in-place commit (the proactive
    prune's ``rewrite_pruned_rows``) changes what a row says and keeps its id and ``active`` flag,
    so an id/activity staleness check cannot see it. A lease-less writer holding the pre-rewrite
    bodies must not publish them (#124102). The comparison is representation-safe: a dict's content
    passed through the loader lens, and the stored bytes through the same lens; the stored-row CAS
    digest wins when present."""
    if store.read_only:
        return _read_only_mismatches(store, session_id, held)
    with store.tx() as conn:
        return held_row_mismatches(conn, session_id, held)


def _read_only_mismatches(store, session_id: str, held: list[dict[str, Any]]) -> list[int]:
    """The same CAS compare over separate reads (a read-only store keeps no write transaction):
    the stored rows are fetched per chunk/uid and the compare is pure Python."""
    exact = [m for m in held if isinstance(m, dict) and m.get(_DB_PERSISTED_MARKER_KEY)
             and m.get("role") in ("user", "assistant", "tool")]
    by_id = {m.get("_row_id"): m for m in exact
            if isinstance(m.get("_row_id"), int) and not isinstance(m.get("_row_id"), bool) and m["_row_id"] > 0}
    by_uid = {}
    for m in exact:
        if m.get("_row_id") is None and isinstance(m.get(DB_ROW_SNAPSHOT), str):
            uid = message_uid_or_none(m)
            if uid is not None:
                by_uid[uid] = m
    if not (by_id or by_uid):
        return []
    stored: dict[int, Any] = {}
    for rid in by_id:
        row = store.one("SELECT * FROM messages WHERE session_id = %s AND id = %s", session_id, rid)
        if row is not None:
            stored[int(row["id"])] = row
    mismatches: list[int] = []
    for m, rid in list(by_id.items()):
        row = stored.get(rid)
        if row is not None and _mismatch_for(m, row):
            mismatches.append(int(row["id"]))
    for uid, m in by_uid.items():
        row = store.one(
            "SELECT * FROM messages WHERE session_id = %s AND active = 1 AND role = %s AND message_uid = %s "
            "ORDER BY id DESC LIMIT 1", session_id, m.get("role"), uid)
        if row is not None and _mismatch_for(m, row):
            mismatches.append(int(row["id"]))
    return mismatches


def _mismatch_for(message: dict[str, Any], row) -> bool:
    """The CAS compare for one held dict against its stored row (payload-decoded)."""
    if not int(row["active"] or 0):
        return False
    snapshot = message.get(DB_ROW_SNAPSHOT)
    if isinstance(snapshot, str):
        return transcript_row_snapshot(_decode_payloads(dict(row))) != snapshot
    from .common import _decode_content
    held_identity = _row_identity(
        message.get("role", "unknown"), message.get("content"), message.get("tool_call_id"),
        _parse_tool_calls(message.get("tool_calls")))
    stored_identity = _row_identity(
        row["role"], _decode_content(row["content"]), row["tool_call_id"],
        _parse_tool_calls(row["tool_calls"]))
    return held_identity != stored_identity


def held_row_mismatches(conn, session_id: str, held: list[dict[str, Any]]) -> list[int]:
    """``rewritten_held_row_ids`` against *conn*: the same CAS compare, on a connection the caller
    already holds (inside the full writer's commit window — a precheck outside that lock misses a
    write that lands between check and commit, #124102). Chunked like
    ``stamp_inserted_row_snapshots``: one ``id IN (...)`` per ``_id_chunks`` batch. Rows without an
    exact ``_row_id`` but with both the persist marker and a CAS snapshot are resolved by their
    logical ``message_uid`` (the newest active row, same rule as the loader)."""
    def _exact_id(message: dict[str, Any]) -> Optional[int]:
        rid = message.get("_row_id")
        if isinstance(rid, int) and not isinstance(rid, bool) and rid > 0:
            return rid
        return None

    exact = [
        (message, _exact_id(message))
        for message in held
        if isinstance(message, dict) and message.get(_DB_PERSISTED_MARKER_KEY)
        and message.get("role") in ("user", "assistant", "tool")]
    by_id = {row_id: message for message, row_id in exact if row_id is not None}
    by_uid = {}
    for message, row_id in exact:
        if row_id is None and isinstance(message.get(DB_ROW_SNAPSHOT), str):
            uid = message_uid_or_none(message)
            if uid is not None:
                by_uid[uid] = message
    if not session_id or not (by_id or by_uid):
        return []
    stored: dict[int, Any] = {}
    for chunk in _id_chunks(by_id):
        for row in conn.execute(
            f"SELECT * FROM messages WHERE session_id = %s AND id IN ({_in_placeholders(len(chunk))})",
            [session_id, *chunk]).fetchall():
            stored[int(row["id"])] = row
    by_uid_row: dict[str, Any] = {}
    for uid, message in by_uid.items():
        # Newest active row: a logical message_uid can have several active physical rows, and the
        # loader stamps the held dict from the newest. A bare fetchone() could return an older copy
        # whose payload still matches the held digest (#124102).
        row = conn.execute(
            "SELECT * FROM messages WHERE session_id = %s AND active = 1 AND role = %s AND message_uid = %s "
            "ORDER BY id DESC LIMIT 1",
            (session_id, message.get("role"), uid)).fetchone()
        if row is not None:
            by_uid_row[uid] = row
            stored[int(row["id"])] = row
    mismatches: list[int] = []
    for message, row_id in exact:
        row = stored.get(row_id) if row_id is not None else by_uid_row.get(message_uid_or_none(message) or "")
        if row is None or not int(row["active"] or 0):
            continue  # the activity staleness check owns absent/inactive rows
        if _mismatch_for(message, row):
            mismatches.append(int(row["id"]))
    return mismatches


def rewrite_pruned_rows(store, session_id: str, changes: list[tuple[dict[str, Any], dict[str, Any]]], *,
    model_config_patch: Optional[dict[str, Any]] = None, turn_lease_holder: Optional[str] = None,
    turn_lease_ttl_seconds: float = 300.0) -> int:
    """Proactive-prune commit that rewrites only the changed rows, in place (#124102). Each
    ``(original, replacement)`` in *changes* is a persisted message the prune edited. Its row is named
    by ``original["_row_id"]``, else by its tool call id (tool results) or tool-call ids (assistant
    turns), else by the single active row with the same role and stored content. That row gets one
    archived twin of its pre-prune self (``active=0, compacted=1``: still searchable), then takes
    the replacement's changed columns under its own id. Unchanged rows, model order, row ids and
    concurrently appended rows are untouched; the display-identity trigger keeps the indexes
    consistent; the counters are unchanged because no row enters or leaves the live set.

    Raises :class:`PruneRowUnresolvedError` before writing anything when a row cannot be named
    exactly; a named row whose stored payload moved on raises :class:`PruneRowStaleError` instead.
    ``turn_lease_holder`` refuses with ``SessionTurnLeaseLostError`` when another process owns the
    session's turn lease. Returns the rows rewritten; each rewritten dict is stamped with its row's
    new digest, so a second prune of the same transcript compares against what this commit wrote."""
    def _call_ids(tool_calls: Any) -> list[Any]:
        return [call.get("id") if isinstance(call, dict) else None for call in (_parse_tool_calls(tool_calls) or [])]

    def _resolve(conn, original: dict[str, Any]) -> Any:
        role = original.get("role")
        if role not in ("assistant", "tool") or original.get(ABSORBED_ROW_IDS):
            raise PruneRowUnresolvedError(f"a {role} message is not rewritten in place")
        row_id = original.get("_row_id")
        if isinstance(row_id, int) and not isinstance(row_id, bool) and row_id > 0:
            ids = [row_id]
        elif role == "tool" and original.get("tool_call_id"):
            ids = [int(row["id"]) for row in conn.execute(
                "SELECT id FROM messages WHERE session_id = %s AND active = 1 AND role = 'tool' AND tool_call_id = %s",
                (session_id, original["tool_call_id"])).fetchall()]
        elif role == "assistant" and _call_ids(original.get("tool_calls")):
            wanted = _call_ids(original.get("tool_calls"))
            ids = [int(row["id"]) for row in conn.execute(
                "SELECT id, tool_calls FROM messages WHERE session_id = %s AND active = 1 AND role = 'assistant' "
                "AND tool_calls IS NOT NULL", (session_id,)).fetchall() if _call_ids(row["tool_calls"]) == wanted]
        else:
            ids = _matching_active_ids(conn, session_id, original)
        if len(ids) != 1:
            raise PruneRowUnresolvedError(f"{len(ids)} live rows match a pruned {role} message")
        row = conn.execute(
            "SELECT * FROM messages WHERE id = %s AND session_id = %s AND active = 1", (ids[0], session_id)).fetchone()
        if row is None or row["role"] != role:
            raise PruneRowStaleError(f"row {ids[0]} no longer holds the pruned {role} message")
        snapshot = original.get(DB_ROW_SNAPSHOT)
        if isinstance(snapshot, str):
            # The stored-row CAS digest: covers every owned column (tool_calls arguments included).
            if transcript_row_snapshot(_decode_payloads(dict(row))) != snapshot:
                raise PruneRowStaleError(f"row {ids[0]} no longer holds the pruned {role} message")
        elif (_row_identity(role, original.get("content"), original.get("tool_call_id"),
                                 _parse_tool_calls(original.get("tool_calls")))
                != _row_identity(row["role"], _decoded_row_content(row),
                                 row["tool_call_id"], _parse_tool_calls(row["tool_calls"]))):
            # No digest (a resumed dict): compare through the loader lens instead.
            raise PruneRowStaleError(f"row {ids[0]} no longer holds the pruned {role} message")
        return row

    def _do(conn):
        if turn_lease_holder:
            # A writer inside a turn commits only while it still owns the turn, like its appends.
            _check_transcript_write_guards(conn, session_id, turn_lease_holder=turn_lease_holder,
                                          turn_lease_ttl_seconds=turn_lease_ttl_seconds)
        patch = model_config_patch is not None
        patched_model_config = _merge_model_config_json(conn, session_id, model_config_patch,
                                                       on_missing="raise") if patch else None
        updates: list[tuple[int, list[str], list[Any], Any, Any, Any]] = []
        for original, replacement in changes:
            row = _resolve(conn, original)
            role, timestamp = row["role"], row["timestamp"]
            if replacement.get("role") != role:
                raise PruneRowUnresolvedError("a prune replacement changed the message role")
            before = _message_row_params(session_id, role, original, _parse_tool_calls(
                original.get("tool_calls")), timestamp, keep_reasoning=role == "assistant")
            after = _message_row_params(session_id, role, replacement, _parse_tool_calls(
                replacement.get("tool_calls")), timestamp, keep_reasoning=role == "assistant")
            changed = [column for column, old, new in zip(_MESSAGE_ROW_COLUMNS, before, after)
                       if old != new and column not in _PRUNE_FIXED_COLUMNS]
            if changed:
                updates.append((int(row["id"]), changed, [
                    value for column, value in zip(_MESSAGE_ROW_COLUMNS, after) if column in changed],
                    row["display_identity"],
                    int(row["display_order"]) if row["display_order"] is not None else int(row["id"]),
                    replacement))
        twin_columns = ", ".join(c for c in _ALL_MESSAGE_COLUMNS
                                 if c not in ("id", "active", "compacted", "display_order"))
        for row_id, columns, values, identity, order, replacement in updates:
            # Twin: the pre-prune self, archived (the RETURNING id replaces lastrowid).
            twin_id = conn.execute(
                f"INSERT INTO messages ({twin_columns}, active, compacted) "
                f"SELECT {twin_columns}, 0, 1 FROM messages WHERE id = %s RETURNING id", (row_id,)).fetchone()["id"]
            # The content UPDATE fires the identity trigger, which nulls the slot of the live row and of
            # every visible row sharing its old identity: the twin and any earlier generation's copies.
            peers = conn.execute(
                "SELECT id, display_order FROM messages WHERE session_id = %s AND id <> %s "
                "AND display_identity = %s AND (active = 1 OR compacted = 1)",
                (session_id, row_id, identity)).fetchall() if identity is not None else [(twin_id, order)]
            conn.execute(f"UPDATE messages SET {', '.join(f'{column} = %s' for column in columns)} WHERE id = %s",
                         [*values, row_id])
            # The superseded bodies keep their identity and slot. The live row takes the identity of
            # what it now stores, so a later compaction's copy of it joins the same slot, and keeps
            # the slot. The page shows one row per display_order, preferring the live one.
            for peer_id, peer_order in peers:
                conn.execute(
                    "UPDATE messages SET display_identity = %s, display_order = %s WHERE id = %s",
                    (identity, order if peer_order is None else peer_order, peer_id))
            live = conn.execute(
                "SELECT role, content, timestamp, tool_call_id, tool_calls, tool_name, display_kind, "
                "display_metadata FROM messages WHERE id = %s", (row_id,)).fetchone()
            conn.execute("UPDATE messages SET display_identity = %s, display_order = %s WHERE id = %s",
                         (_display_identity(_display_dedupe_key(dict(live))), order, row_id))
            # Stamp the replacement with the stored row's new CAS digest (read back, not the bind values):
            # a second prune of this transcript must compare against what this commit wrote (#124102).
            final_row = conn.execute(
                "SELECT * FROM messages WHERE id = %s", (row_id,)).fetchone()
            replacement[DB_ROW_SNAPSHOT] = transcript_row_snapshot(_decode_payloads(dict(final_row)))
        if patch:
            conn.execute("UPDATE sessions SET model_config = %s WHERE id = %s", (patched_model_config, session_id))
        return len(updates)

    _ensure_display_index(store, session_id)  # every row holds its stored slot before one is carried over
    # _transcript_write restores each dict's row state (id/digest/timestamp) on rollback, so a
    # failed or retried commit never leaves the replacements stamped with an uncommitted digest.
    return _transcript_write(store, [replacement for _, replacement in changes], _do)


def _decoded_row_content(row) -> Any:
    """The payload-decoded stored content of a ``SELECT *`` row (the bytea payload rule)."""
    return _decode_payloads(dict(row))["content"]

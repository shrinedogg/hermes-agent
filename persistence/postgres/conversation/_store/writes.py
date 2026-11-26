"""Transcript write store verbs on PostgreSQL (the SQLite store's write functions ported onto
``PostgresStore``): batch append with repair, replace/clear, in-place compaction, single-row edits,
reactions, compression cooldowns, marker purge and the detached-result writers. Every write is one
``store.tx()``; the DDL display triggers keep the display index coherent, and the verbs that change a
row's display key re-fold it in the same transaction (``_reconcile_display_orders``)."""

from __future__ import annotations

import json
import time
from functools import partial
from typing import Any, Mapping, Optional, Sequence

from persistence.sqlite.conversation._store.rules import (
    DB_ROW_SNAPSHOT as _DB_SNAPSHOT_KEY, MAX_TODO_RESULT_CHARS, RESTORED_FROM_STORE, _sanitize_surrogates,
)
from persistence import CooldownRestoreRefused
from persistence.sqlite._state.common import _json_or
from persistence.sqlite._state.errors import HeldRowRewrittenError, SessionCompressionInProgressError
from persistence.sqlite.conversation._store.core import REACTIONS_METADATA_KEY, _STALE_TOOL_CALL_MARKER_RE
from persistence.sqlite.conversation._store.messages_sql import _parse_tool_calls, _tool_calls_len

from .common import (
    _ARCHIVE_ACTIVE_SQL, _BUMP_GENERATION_SQL, _COMPRESSION_LOCK_ROW_SQL, _DISPLAY_META_ROW_SQL,
    _INSERT_MESSAGE_SQL, _LIVE_IDENTITY_SQL, _RESET_COUNTERS_SQL, _SET_CODEX_REASONING_SQL,
    _SET_COUNTERS_SQL, _SET_DISPLAY_META_SQL, _SHADOWED_CHECKPOINT_ROWS_SQL, _in_placeholders,
    _carry_parent_timestamps, _check_transcript_write_guards, _clone_message_rows, _decode_display_metadata,
    _display_dedupe_key, _display_identity, _encode_content, _encode_display_metadata, _is_explicit_fork_child_row,
    _prune_shadowed_checkpoints, _reasoning_json_text, _reconcile_display_orders, _stamp_kept_live_prefix,
    _stamp_tool_call_uids, _tool_calls_count, _transcript_write,
)
from .repair import (
    _active_logical_message_row, resolve_and_repair_transcript_batch, stamp_inserted_row_snapshots,
    transcript_row_snapshot,
)
from .sessions_pg import _merge_model_config_json

__all__ = [
    "append_batch", "append_completion_note", "append_delegation_delivery", "archive_and_compact",
    "clear_transcript", "deactivate_message", "deactivate_messages_by_display_kind",
    "get_compression_failure_cooldown_row", "insert_message_rows", "latest_todo_result",
    "purge_stale_tool_call_markers", "record_compression_failure_cooldown",
    "clear_compression_failure_cooldown", "restore_compression_failure_cooldown_row",
    "replace_transcript", "set_latest_matching_message_display_kind", "set_latest_user_api_content",
    "set_message_api_content", "set_message_reaction", "set_user_message_content", "take_unseen_reactions",
]


# ── batch append ──

def _insert_message_rows(conn, session_id: str, messages: list[dict[str, Any]], *,
                         prune_checkpoints: bool = True) -> tuple[int, int]:
    """Insert *messages* as fresh active rows in the caller's txn -> ``(inserted, tool_call_count)``
    (the SQLite ``_insert_message_rows`` port: ``RETURNING id`` instead of ``lastrowid``, display
    order/identity left to the DDL insert triggers)."""
    now_ts = time.time()
    inserted = tool_calls_total = 0
    batch_tool_index: dict[str, str] = {}
    for msg in messages:
        role = msg.get("role", "unknown")
        tool_calls = _parse_tool_calls(msg.get("tool_calls"))
        message_timestamp = _coerce_ts(msg.get("timestamp"), now_ts)
        # The durable per-message id: kept when the dict carries one, minted and stamped otherwise.
        from persistence.sqlite.conversation._store.rules import stamp_message_uid
        stamp_message_uid(msg)
        _stamp_tool_call_uids(msg, tool_calls, batch_tool_index)
        row = conn.execute(
            _INSERT_MESSAGE_SQL,
            _message_row_params(session_id, role, msg, tool_calls, message_timestamp,
                               keep_reasoning=role == "assistant"),
        ).fetchone()
        msg["timestamp"] = message_timestamp
        if row is not None:
            msg["_row_id"] = row["id"]
        inserted += 1
        tool_calls_total += _tool_calls_count(tool_calls)
        now_ts = max(now_ts, message_timestamp) + 1e-6
    if prune_checkpoints:
        _prune_shadowed_checkpoints(conn, session_id, messages)
    # Every inserter gets the new rows' own stored-row digest (the versioned rewrite path's fence).
    stamp_inserted_row_snapshots(conn, session_id, messages)
    return inserted, tool_calls_total


def _coerce_ts(value: Any, default: float) -> float:
    from persistence.sqlite.conversation._store.messages_sql import _coerce_timestamp
    return _coerce_timestamp(value, default)


def _message_row_params(session_id: str, role: str, msg: dict[str, Any], tool_calls: Any,
                       message_timestamp: float, *, keep_reasoning: bool) -> tuple:
    from .common import _message_row_params as _params
    return _params(session_id, role, msg, tool_calls, message_timestamp, keep_reasoning=keep_reasoning)


def append_batch(store, session_id: str, messages: list[dict[str, Any]], *,
                 turn_lease_holder: Optional[str] = None,
                 turn_lease_ttl_seconds: float = 300.0) -> int:
    """One batch write transaction (the SQLite ``_append_chunk`` port): the guards, the repair-batch
    resolution, the insert and the counter bump, plus the restored-dict row-state stamp (a dict the
    repair left unwritten takes the newest active row of its role carrying its uid)."""
    if not messages:
        return 0

    def _do(conn) -> int:
        _check_transcript_write_guards(conn, session_id, None,
                                       turn_lease_holder=turn_lease_holder,
                                       turn_lease_ttl_seconds=turn_lease_ttl_seconds)
        inserted_rows = resolve_and_repair_transcript_batch(
            conn, session_id, messages,
            encode_content_fn=_encode_content,
            decode_content_fn=_decode_content,
            serialize_message_fn=partial(_serialize, session_id),
            decode_row_fn=_decoded_repair_row,
        )
        inserted, tool_calls_total = _insert_message_rows(conn, session_id, inserted_rows)
        _bump_counters(conn, session_id, inserted, tool_calls_total)
        for msg in messages:
            if msg.get(RESTORED_FROM_STORE) and msg.get("_db_persisted") and "_row_id" not in msg:
                from session_rules.turns import message_uid_or_none
                row = _active_logical_message_row(conn, session_id, msg["role"], message_uid_or_none(msg))
                if row is not None:
                    msg.update({"_row_id": int(row["id"]), "timestamp": row["timestamp"],
                                _DB_SNAPSHOT_KEY: transcript_row_snapshot(row)})
                    from persistence.sqlite.conversation._store.identity import _restore_row_identity
                    _restore_row_identity(row, msg)
        return inserted

    return _transcript_write(store, messages, _do)


def _serialize(session_id: str, msg: dict[str, Any], message_timestamp: float) -> dict[str, Any]:
    from .common import _serialized_message_row
    return _serialized_message_row(session_id, msg, message_timestamp)


def _decoded_repair_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Decode one durable row without replay dedupe, alternation repair, or content stripping (the
    SQLite ``_decoded_repair_row`` port)."""
    from .common import _decode_content
    msg: dict[str, Any] = {
        "role": row["role"],
        "content": _decode_content(row["content"]),
    }
    for column in ("tool_call_id", "tool_name", "effect_disposition", "token_count", "finish_reason"):
        if row[column] is not None:
            msg[column] = row[column]
    if row["tool_calls"]:
        msg["tool_calls"] = _json_or(row["tool_calls"], [],
                                     "Failed to deserialize repaired tool_calls, falling back to []")
    if row["platform_message_id"] is not None:
        msg["message_id"] = row["platform_message_id"]
        msg["platform_message_id"] = row["platform_message_id"]
    if row["observed"]:
        msg["observed"] = True
    if row["_compressed_summary"]:
        msg["_compressed_summary"] = True
    from persistence.sqlite.conversation._store.identity import _restore_identity_columns
    _restore_identity_columns(row, msg)
    if row["api_content"] is not None:
        msg["api_content"] = row["api_content"]
    if row["display_kind"] is not None:
        msg["display_kind"] = row["display_kind"]
    if row["display_metadata"] is not None:
        metadata = _decode_display_metadata(row["display_metadata"])
        if metadata is not None:
            msg["display_metadata"] = metadata
    if row["role"] == "assistant":
        for column in ("reasoning", "reasoning_content"):
            if row[column] is not None:
                msg[column] = row[column]
        for column in ("reasoning_details", "codex_reasoning_items", "codex_message_items"):
            if row[column]:
                msg[column] = _json_or(row[column], None, f"Failed to deserialize repaired {column}, "
                                                         f"falling back to None")
    return msg


def _bump_counters(conn, session_id: str, inserted: int, tool_calls: int) -> None:
    from .common import _bump_session_counters
    _bump_session_counters(conn, session_id, inserted, tool_calls, unit=False)


def _decode_content(value: Any) -> Any:
    from .common import _decode_content as _dc
    return _dc(value)


def _encode_content(value: Any) -> Any:
    from .common import _encode_content as _ec
    return _ec(value)


# ── replace / clear ──

def replace_transcript(store, session_id: str, messages: list[dict[str, Any]], *,
                       active_only: bool = False, archive_dropped: bool = False,
                       reject_active_turn_lease: bool = False) -> None:
    """Atomically replace a session's messages (the SQLite ``replace_messages`` port): the in-order
    prefix already present as leading live rows keeps its rows; only the divergent suffix is
    archived (or deleted) and the new suffix inserted, counters reconciled in the same txn."""
    def _do(conn) -> None:
        _replace_in_transaction(conn, session_id, messages, active_only=active_only,
                                archive_dropped=archive_dropped,
                                reject_active_turn_lease=reject_active_turn_lease)
    _transcript_write(store, messages, _do)


def _replace_in_transaction(conn, session_id: str, messages: list[dict[str, Any]], *,
                           active_only: bool = False, archive_dropped: bool = False,
                           reject_active_turn_lease: bool = False) -> None:
    from persistence.sqlite._state.errors import CompressionSessionClosedError
    if reject_active_turn_lease:
        _check_transcript_write_guards(conn, session_id, None,
                                       reject_active_turn_lease=True,
                                       reject_active_compression_lock=True)
    else:
        if _ended(conn, session_id):
            raise CompressionSessionClosedError(session_id)
        # The lease-fenced guard's last step, so a closed session still wins over a missing one.
        if conn.execute("SELECT 1 FROM sessions WHERE id = %s", (session_id,)).fetchone() is None:
            from persistence import SessionRowMissing, SessionId
            raise SessionRowMissing(SessionId(session_id))
    kept = kept_tool_calls = 0
    if archive_dropped:
        # Only the first len(messages)+1 live rows matter: the prefix to match plus the row whose id
        # anchors the archive UPDATE (which itself covers every later row via `id >= ?`).
        live = conn.execute(_LIVE_IDENTITY_SQL, (session_id, len(messages) + 1)).fetchall()
        kept = _stamp_kept_live_prefix(live, messages)
        kept_tool_calls = sum(_tool_calls_len(row["tool_calls"]) for row in live[:kept])
        if kept < len(live):
            conn.execute("UPDATE messages SET active = 0 WHERE session_id = %s AND active = 1 AND id >= %s",
                         (session_id, live[kept]["id"]))
    else:
        sql = "DELETE FROM messages WHERE session_id = %s" + (" AND active = 1" if active_only else "")
        conn.execute(sql, (session_id,))
    inserted, inserted_tool_calls = _insert_message_rows(conn, session_id, messages[kept:])
    conn.execute(f"{_SET_COUNTERS_SQL} WHERE id = %s",
                 (kept + inserted, kept_tool_calls + inserted_tool_calls, session_id))
    _reconcile_display_orders(conn, session_id)


def _ended(conn, session_id: str) -> bool:
    from .common import _ENDED_ROW_SQL, _ended_by_compression
    return _ended_by_compression(conn.execute(_ENDED_ROW_SQL, (session_id,)).fetchone())


def clear_transcript(store, session_id: str) -> None:
    """Delete every row of a session and zero its counters (the SQLite ``clear_messages`` port)."""
    def _do(conn) -> None:
        conn.execute("DELETE FROM messages WHERE session_id = %s", (session_id,))
        conn.execute(_RESET_COUNTERS_SQL, (session_id,))
    with store.tx() as conn:
        _do(conn)


# ── in-place compaction ──

def archive_and_compact(store, session_id: str, compacted_messages: list[dict[str, Any]], *,
                       model_config_patch: Optional[dict[str, Any]] = None,
                       watermark: Optional[int] = None, lock_holder: Optional[str] = None,
                       tail_count: int = 0, carried_messages: Optional[list[dict[str, Any]]] = None,
                       covered_ids: Optional[list[int]] = None,
                       unresolved_held: Optional[list[dict[str, Any]]] = None,
                       held_for_cas: Optional[list[dict[str, Any]]] = None) -> int:
    """Non-destructive in-place compaction under ONE session id (the SQLite ``archive_and_compact``
    port): soft-archive the active rows, insert the compacted set, clone the concurrent tail, all in
    one transaction; returns the new ACTIVE count."""
    from .coverage import _carried_tail_ids, _carried_tail_slots, _proved_coverage, _resolve_carried_row_ids, _tail_originals, _uncounted_merged_rows
    from .prune_rewrite import held_row_mismatches

    def _do(conn) -> int:
        if lock_holder is not None:
            lock_row = conn.execute(_COMPRESSION_LOCK_ROW_SQL, (session_id,)).fetchone()
            if lock_row is None or lock_row["holder"] != lock_holder \
                    or float(lock_row["expires_at"]) <= time.time():
                raise SessionCompressionInProgressError(
                    f"Compression lease for {session_id!r} lost before commit; "
                    "refusing to publish a stale compaction")
        patch = model_config_patch is not None
        patched_model_config = _merge_model_config_json(conn, session_id, model_config_patch,
                                                        on_missing="raise") if patch else None
        if held_for_cas:
            if held_row_mismatches(conn, session_id, held_for_cas):
                raise HeldRowRewrittenError(
                    f"a held row of session {session_id} was rewritten in place before the commit")
        proved = _proved_coverage(conn, session_id, covered_ids, unresolved_held, watermark)
        if proved is not None:
            return _archive_named_rows(conn, session_id, compacted_messages, proved[0],
                                      tail_count=tail_count, carried_messages=carried_messages,
                                      patched_model_config=patched_model_config, patch=patch,
                                      merged_away=proved[1])
        tail_ids: list[int] = [] if watermark is None else [int(row["id"]) for row in conn.execute(
            "SELECT id FROM messages WHERE session_id = %s AND active = 1 AND id > %s ORDER BY id",
            (session_id, int(watermark))).fetchall()]
        carried_tail, weak_claims = _carried_tail_ids(conn, tail_ids, compacted_messages)
        clone_ids = [row_id for row_id in tail_ids if row_id not in carried_tail]
        _ids, tail_tool_calls = ([], 0) if not clone_ids else (
            _tail_rows_after_watermark(conn, clone_ids))
        tail_below = tail_count - _carried_tail_slots(carried_tail, len(compacted_messages), tail_count)
        rewind_ids = _resolve_carried_row_ids(conn, session_id, carried_messages or [])
        if tail_below > 0:
            bound = watermark is not None
            rewind_ids += [int(row["id"]) for row in conn.execute(
                "SELECT id FROM messages WHERE session_id = %s AND active = 1"
                + (" AND id <= %s" if bound else "") + " ORDER BY id DESC LIMIT %s",
                (session_id, *((int(watermark),) if bound else ()),
                 int(tail_below) + _uncounted_merged_rows(compacted_messages[-int(tail_count):]))).fetchall()]
        rewind_ids += [row_id for row_id in tail_ids if row_id not in weak_claims]
        rewind_ids = list(dict.fromkeys(rewind_ids))
        if rewind_ids:
            marks = _in_placeholders(rewind_ids)
            conn.execute("UPDATE messages SET active = 0, compacted = 0 "
                         f"WHERE session_id = %s AND id IN ({marks})", [session_id, *rewind_ids])
            conn.execute(f"{_ARCHIVE_ACTIVE_SQL} AND id NOT IN ({marks})", [session_id, *rewind_ids])
        else:
            conn.execute(_ARCHIVE_ACTIVE_SQL, (session_id,))
        _carry_parent_timestamps(conn, session_id, compacted_messages)
        inserted, tool_calls_total = _insert_message_rows(conn, session_id, compacted_messages)
        if clone_ids:
            _clone_message_rows(conn, clone_ids)
            inserted += len(clone_ids)
            tool_calls_total += tail_tool_calls
        _reconcile_display_orders(conn, session_id)
        counters_sql = f"{_SET_COUNTERS_SQL}" + (", model_config = %s" if patch else "") + " WHERE id = %s"
        counter_params: list = [inserted, tool_calls_total]
        if patch:
            counter_params.append(patched_model_config)
        counter_params.append(session_id)
        conn.execute(counters_sql, tuple(counter_params))
        return inserted

    def _fn(conn):
        return _do(conn)

    return _transcript_write(store, compacted_messages, _fn)


def _tail_rows_after_watermark(conn, clone_ids: list[int]) -> tuple[list[int], int]:
    """``(ids, tool_call_count)`` of the concurrent-tail rows (the SQLite ``_tail_rows_after_watermark``)."""
    rows = conn.execute(
        f"SELECT id, tool_calls FROM messages WHERE id IN ({_in_placeholders(clone_ids)}) ORDER BY id",
        tuple(clone_ids)).fetchall()
    return [int(r["id"]) for r in rows], sum(_tool_calls_len(r["tool_calls"]) for r in rows)


def _archive_named_rows(conn, session_id: str, compacted_messages: list[dict[str, Any]],
                       covered: list[int], *, tail_count: int,
                       carried_messages: Optional[list[dict[str, Any]]],
                       patched_model_config: Any, patch: bool, merged_away: set[int]) -> int:
    """Archive *covered* as summarized and clone every other active row after the new set (the SQLite
    ``_archive_named_rows`` port)."""
    from .coverage import _carried_tail_ids, _carried_tail_slots, _resolve_carried_row_ids, _tail_originals
    active_ids = [int(row["id"]) for row in conn.execute(
        "SELECT id FROM messages WHERE session_id = %s AND active = 1 ORDER BY id", (session_id,)).fetchall()]
    covered_set = set(covered)
    carried_ids = _resolve_carried_row_ids(conn, session_id, carried_messages or [])
    covered_set.update(carried_ids)
    unseen = [row_id for row_id in active_ids if row_id not in covered_set]
    covered_active = [row_id for row_id in active_ids if row_id in covered_set]
    carried_unseen, weak_claims = _carried_tail_ids(conn, unseen, compacted_messages)
    clone_ids = [row_id for row_id in unseen if row_id not in carried_unseen]
    tail_below = tail_count - _carried_tail_slots(carried_unseen, len(compacted_messages), tail_count)
    rewind_ids = list(carried_ids)
    if tail_below > 0:
        rewind_ids += _tail_originals(covered_active, tail_below, merged_away)
    rewind_ids += [row_id for row_id in unseen if row_id not in weak_claims]
    rewind_ids = list(dict.fromkeys(rewind_ids))
    if rewind_ids:
        marks = _in_placeholders(rewind_ids)
        conn.execute("UPDATE messages SET active = 0, compacted = 0 "
                     f"WHERE session_id = %s AND id IN ({marks})", [session_id, *rewind_ids])
    conn.execute(_ARCHIVE_ACTIVE_SQL, (session_id,))
    _carry_parent_timestamps(conn, session_id, compacted_messages)
    inserted, tool_calls_total = _insert_message_rows(conn, session_id, compacted_messages)
    if clone_ids:
        _ids, unseen_tool_calls = _tail_rows_after_watermark(conn, clone_ids)
        _clone_message_rows(conn, clone_ids)
        inserted += len(clone_ids)
        tool_calls_total += unseen_tool_calls
    _reconcile_display_orders(conn, session_id)
    counters_sql = f"{_SET_COUNTERS_SQL}" + (", model_config = %s" if patch else "") + " WHERE id = %s"
    counter_params: list = [inserted, tool_calls_total]
    if patch:
        counter_params.append(patched_model_config)
    counter_params.append(session_id)
    conn.execute(counters_sql, tuple(counter_params))
    return inserted


# ── single-row edits ──

def set_user_message_content(store, session_id: str, row_id: int, content: Any,
                            display_metadata: Optional[dict[str, Any]] = None) -> bool:
    """Rewrite the content of ONE known active user row (the SQLite ``set_user_message_content``
    port); *display_metadata* is merge-patched in (reactions survive). The content change invalidates
    the display identity via the DDL trigger, so the fold runs in the same txn."""
    if not session_id or isinstance(row_id, bool) or not isinstance(row_id, int) or row_id <= 0:
        return False
    if display_metadata is None:
        def _do(conn) -> int:
            cur = conn.execute(
                "UPDATE messages SET content = %s WHERE id = %s AND session_id = %s "
                "AND role = 'user' AND active = 1",
                (_payload_bytes_param(_encode_content(content)), row_id, session_id))
            _reconcile_display_orders(conn, session_id)
            return cur.rowcount
    else:
        def _do(conn) -> int:
            stored = conn.execute(
                "SELECT display_metadata FROM messages WHERE id = %s AND session_id = %s "
                "AND role = 'user' AND active = 1", (row_id, session_id)).fetchone()
            # The PG twin of the SQLite ``json_patch`` merge (RFC 7386 object merge: a None value
            # deletes its key; everything else overwrites; reactions survive).
            base = _decode_display_metadata(stored["display_metadata"]) if stored else None
            merged_meta = dict(base) if base else {}
            for key, value in display_metadata.items():
                if value is None:
                    merged_meta.pop(key, None)
                else:
                    merged_meta[key] = value
            cur = conn.execute(
                "UPDATE messages SET content = %s, api_content = NULL, display_metadata = %s "
                "WHERE id = %s AND session_id = %s AND role = 'user' AND active = 1",
                (_payload_bytes_param(_encode_content(content)), _encode_display_metadata(merged_meta), row_id, session_id))
            _reconcile_display_orders(conn, session_id)
            return cur.rowcount
    with store.tx() as conn:
        return _do(conn) > 0


def deactivate_message(store, session_id: str, row_id: int) -> int:
    """Deactivate ONE known row (the SQLite ``deactivate_message`` port; the DDL visibility trigger
    re-slots the display group)."""
    if not session_id or isinstance(row_id, bool) or not isinstance(row_id, int) or row_id <= 0:
        return 0
    with store.tx() as conn:
        cur = conn.execute("UPDATE messages SET active = 0 WHERE id = %s AND session_id = %s",
                           (row_id, session_id))
        return cur.rowcount


def deactivate_messages_by_display_kind(store, session_id: str, display_kind: str) -> int:
    """Deactivate every live row of one ``display_kind`` (the SQLite port, surrogate-sanitized)."""
    if not session_id or not display_kind:
        return 0
    with store.tx() as conn:
        cur = conn.execute(
            "UPDATE messages SET active = 0 WHERE session_id = %s AND display_kind = %s AND active = 1",
            (session_id, _sanitize_surrogates(display_kind)))
        return cur.rowcount


def set_latest_matching_message_display_kind(store, session_id: str, *, role: str, content: str,
                                            display_kind: str,
                                            display_metadata: Optional[dict[str, Any]] = None) -> bool:
    """Stamp presentation metadata on the newest active row by content (the SQLite port). A
    display-kind change invalidates the display identity via the DDL trigger; the fold runs in-txn."""
    if not session_id or not content or not display_kind:
        return False

    def _do(conn) -> bool:
        row = conn.execute(
            "SELECT id FROM messages WHERE session_id = %s AND role = %s AND content = %s "
            "AND active = 1 ORDER BY id DESC LIMIT 1",
            (session_id, role, _payload_bytes_param(_encode_content(content)))).fetchone()
        if row is None:
            return False
        conn.execute("UPDATE messages SET display_kind = %s, display_metadata = %s WHERE id = %s",
                     (_sanitize_surrogates(display_kind), _encode_display_metadata(display_metadata),
                      row["id"]))
        _reconcile_display_orders(conn, session_id)
        return True

    with store.tx() as conn:
        return _do(conn)


def _payload_bytes_param(value: Any) -> Any:
    from .common import _payload_bytes
    return _payload_bytes(value)


# ── reactions ──

def _reaction_list(meta: Optional[dict[str, Any]]) -> list[dict[str, Any]]:
    """Well-formed (dict) reactions stored under ``REACTIONS_METADATA_KEY``."""
    reactions = (meta or {}).get(REACTIONS_METADATA_KEY)
    return [r for r in reactions if isinstance(r, dict)] if isinstance(reactions, list) else []


def set_message_reaction(store, session_id: str, message_row_id: int, emoji: Optional[str], *,
                        author: str = "user") -> Optional[list[dict[str, Any]]]:
    """Set (``emoji=None``: clear) *author*'s reaction (the SQLite port: tapback semantics, one per
    author per message; the row must be in the session's visible resume lineage)."""
    if not session_id or message_row_id is None:
        return None
    from .reads import _resume_lineage_ids
    lineage = _resume_lineage_ids(store, session_id)
    sql, params = _DISPLAY_META_ROW_SQL.format(ids=_in_placeholders(lineage)), (message_row_id, *lineage)

    def _do(conn) -> Optional[list[dict[str, Any]]]:
        row = conn.execute(sql, params).fetchone()
        if row is None:
            return None
        meta = _decode_display_metadata(row["display_metadata"]) or {}
        existing = _reaction_list(meta)
        reactions = [r for r in existing if r.get("author") != author]
        previous = next((r for r in existing if r.get("author") == author), None)
        if emoji and (previous is None or previous.get("emoji") != emoji):
            reactions.append({"emoji": _sanitize_surrogates(emoji), "author": author, "at": time.time()})
        if reactions:
            meta[REACTIONS_METADATA_KEY] = reactions
        else:
            meta.pop(REACTIONS_METADATA_KEY, None)
        conn.execute(_SET_DISPLAY_META_SQL,
                     (_encode_display_metadata(meta) if meta else None, message_row_id))
        return reactions

    with store.tx() as conn:
        return _do(conn)


def take_unseen_reactions(store, session_id: str, *, author: str = "user") -> list[dict[str, Any]]:
    """Return *author*'s not-yet-surfaced reactions and mark them seen (the SQLite port)."""
    if not session_id:
        return []
    from .reads import _resume_lineage_ids
    lineage = _resume_lineage_ids(store, session_id)

    def _do(conn) -> list[dict[str, Any]]:
        pending: list[dict[str, Any]] = []
        rows = conn.execute(
            "SELECT id, role, content, display_metadata FROM messages "
            f"WHERE session_id IN ({_in_placeholders(lineage)}) AND (active = 1 OR compacted = 1) "
            "AND (display_metadata::jsonb -> 'reactions') IS NOT NULL ORDER BY id",
            tuple(lineage)).fetchall()
        for row in rows:
            meta = _decode_display_metadata(row["display_metadata"])
            reactions = meta.get(REACTIONS_METADATA_KEY) if meta else None
            if not isinstance(reactions, list):
                continue
            changed = False
            for reaction in reactions:
                if not isinstance(reaction, dict) or reaction.get("author") != author or reaction.get("seen"):
                    continue
                reaction["seen"] = True
                changed = True
                content = _decode_content(row["content"])
                pending.append({
                    "row_id": row["id"], "role": row["role"], "emoji": reaction.get("emoji") or "",
                    "text": content if isinstance(content, str) else ""})
            if changed:
                conn.execute(_SET_DISPLAY_META_SQL, (_encode_display_metadata(meta), row["id"]))
        return pending

    with store.tx() as conn:
        return _do(conn)


# ── compression cooldowns ──

def get_compression_failure_cooldown_row(store, session_id: str) -> dict[str, Any]:
    """``{"session_exists", "cooldown_until", "error"}`` (the SQLite
    ``get_compression_failure_cooldown_row`` port)."""
    row = store.one("SELECT 1 FROM sessions WHERE id = %s", session_id)
    if row is None:
        return {"session_exists": False, "cooldown_until": None, "error": None}
    sess = store.one("SELECT compression_failure_cooldown_until, compression_failure_error "
                     "FROM sessions WHERE id = %s", session_id)
    return {"session_exists": True,
            "cooldown_until": float(sess["compression_failure_cooldown_until"])
            if sess and sess["compression_failure_cooldown_until"] is not None else None,
            "error": sess["compression_failure_error"] if sess else None}


def record_compression_failure_cooldown(store, session_id: str, until: float, error: Optional[str] = None) -> None:
    """Persist the active compression-failure cooldown (the SQLite port: merge-max with any longer live
    deadline so a later shorter write can't reopen the thrash window; the error always takes the latest
    diagnostic)."""
    def _do(conn) -> None:
        conn.execute("UPDATE sessions SET compression_failure_cooldown_until = CASE "
                     "WHEN compression_failure_cooldown_until IS NOT NULL "
                     "AND compression_failure_cooldown_until > %s "
                     "THEN compression_failure_cooldown_until ELSE %s END, "
                     "compression_failure_error = %s WHERE id = %s",
                     (float(until), float(until),
                      _sanitize_surrogates(error) if isinstance(error, str) else error,
                      session_id))
    with store.tx() as conn:
        _do(conn)


def clear_compression_failure_cooldown(store, session_id: str) -> None:
    def _do(conn) -> None:
        conn.execute("UPDATE sessions SET compression_failure_cooldown_until = NULL, "
                     "compression_failure_error = NULL WHERE id = %s", (session_id,))
    with store.tx() as conn:
        _do(conn)


def restore_compression_failure_cooldown_row(store, session_id: str, row: Mapping[str, Any]) -> None:
    """Restore and verify an exact cooldown-row snapshot (the SQLite port: unlike record/clear this
    rollback API propagates write and verification failures; the tolerated exception is a session row
    that vanished mid-attempt, since its cooldown died with it)."""
    def _exists(conn) -> bool:
        return conn.execute("SELECT 1 FROM sessions WHERE id = %s", (session_id,)).fetchone() is not None

    def _actual(conn) -> dict[str, Any]:
        sess = conn.execute("SELECT compression_failure_cooldown_until, compression_failure_error "
                            "FROM sessions WHERE id = %s", (session_id,)).fetchone()
        if sess is None:
            return {"session_exists": False, "cooldown_until": None, "error": None}
        return {"session_exists": True,
                "cooldown_until": float(sess["compression_failure_cooldown_until"])
                if sess["compression_failure_cooldown_until"] is not None else None,
                "error": sess["compression_failure_error"]}

    if not row.get("session_exists"):
        with store.tx() as conn:
            if _exists(conn):
                raise RuntimeError("cannot restore absent compression cooldown row: session now exists")
        return
    deadline = row.get("cooldown_until")
    error = row.get("error")
    with store.tx() as conn:
        written = conn.execute("UPDATE sessions SET compression_failure_cooldown_until = %s, "
                               "compression_failure_error = %s WHERE id = %s",
                               (float(deadline) if deadline is not None else None, error, session_id))
        if written.rowcount == 0:  # the session vanished before the compensating write: nothing to restore
            return
        actual = _actual(conn)
        expected = {"session_exists": True,
                    "cooldown_until": float(deadline) if deadline is not None else None,
                    "error": error}
        if actual != expected:
            if actual.get("session_exists"):
                raise RuntimeError(
                    f"compression cooldown rollback verification failed: expected={expected!r}, actual={actual!r}")


# ── marker purge ──

def purge_stale_tool_call_markers(store, *, dry_run: bool = False, backup: bool = True) -> dict[str, Any]:
    """Permanently clear bare tool-call marker content (the SQLite ``purge_stale_tool_call_markers``
    port; the marker regex is applied in Python over the decoded content). PostgreSQL has no
    ``VACUUM INTO`` physical backup: the report's ``backup_path`` is always ``None`` (recorded for the
    stage-1 review; a logical dump is a stage-6 maintenance concern)."""
    def _find_affected(conn) -> list[int]:
        ids = []
        for row in conn.execute(
                "SELECT id, content FROM messages WHERE role = 'assistant' "
                "AND tool_calls IS NOT NULL AND tool_calls != ''").fetchall():
            content = _payload_str_value(row["content"])
            if isinstance(content, str) and _STALE_TOOL_CALL_MARKER_RE.fullmatch(content.strip()):
                ids.append(int(row["id"]))
        return ids

    affected_ids = store.all("SELECT 1 FROM messages LIMIT 1")
    with store.tx() as conn:
        found = _find_affected(conn)
        if dry_run or not found:
            return {"dry_run": dry_run, "rows_affected": len(found), "row_ids": found, "backup_path": None}
        conn.execute(f"UPDATE messages SET content = %s WHERE id IN ({_in_placeholders(found)})",
                     [b""] + found)
        return {"dry_run": dry_run, "rows_affected": len(found), "row_ids": found, "backup_path": None}


def _payload_str_value(value: Any) -> Optional[str]:
    from .common import _payload_str
    return _payload_str(value)


# ── detached results (delegation delivery / completion notes) ──

def append_delegation_delivery(store, session_id: str, content: str, metadata: dict[str, Any]) -> int:
    """Persist one delegation delivery as a new assistant row (the SQLite
    ``append_delegation_delivery`` port)."""
    msg = {"role": "assistant", "content": content, "display_kind": "delegation_delivery",
           "display_metadata": {"delegation": metadata}}
    inserted = append_batch(store, session_id, [msg])
    return _last_row_id(msg)


def _last_row_id(msg: dict[str, Any]) -> int:
    return int(msg.get("_row_id") or 0)


def append_completion_note(store, session_id: str, admission_id: str, content: str,
                           display_kind: str, metadata: dict[str, Any]) -> int:
    """Persist an admission completion note as a new assistant row (the SQLite
    ``append_completion_note`` port)."""
    meta = dict(metadata)
    meta["admission_id"] = admission_id
    msg = {"role": "assistant", "content": content, "display_kind": display_kind,
           "display_metadata": meta}
    inserted = append_batch(store, session_id, [msg])
    return _last_row_id(msg)


# ── api_content / todo ──

def set_latest_user_api_content(store, session_id: str, content: str, api_content: str) -> bool:
    """Stamp *api_content* on the newest active user row when its content matches (the SQLite
    ``set_latest_user_api_content`` port; the stamp addresses the single newest row, never an older
    row that merely carries the same text)."""
    def _do(conn) -> bool:
        cur = conn.execute(
            "UPDATE messages SET api_content = %s WHERE id = (SELECT id FROM messages "
            "WHERE session_id = %s AND role = 'user' AND active = 1 ORDER BY id DESC LIMIT 1) "
            "AND content = %s",
            (_payload_bytes_param(api_content), session_id,
             _payload_bytes_param(_encode_content(content))))
        return cur.rowcount > 0
    with store.tx() as conn:
        return _do(conn)


def set_message_api_content(store, session_id: str, message_id: int, content: str,
                           api_content: str) -> bool:
    """Stamp *api_content* on one known user row whose content matches (the SQLite port: the ``role``
    and ``active`` guards stay, so an assistant row or a rewound one is refused)."""
    def _do(conn) -> bool:
        cur = conn.execute(
            "UPDATE messages SET api_content = %s WHERE id = %s AND session_id = %s "
            "AND role = 'user' AND active = 1 AND content = %s",
            (_payload_bytes_param(api_content), message_id, session_id,
             _payload_bytes_param(_encode_content(content))))
        return cur.rowcount > 0
    with store.tx() as conn:
        return _do(conn)


def latest_todo_result(store, session_id: str, *, max_chars: int) -> Optional[str]:
    """The newest active todo-result tool result, capped at *max_chars* (the SQLite
    ``get_latest_todo_result`` port)."""
    row = store.one(
        "SELECT content FROM messages WHERE session_id = %s AND active = 1 "
        "AND role = 'tool' AND tool_name = 'todo' ORDER BY id DESC LIMIT 1",
        session_id)
    if row is None or row["content"] is None:
        return None
    text = _payload_str_value(row["content"]) or ""
    return text[:max_chars]

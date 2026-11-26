"""``MessageRepository`` on PostgreSQL: every contract verb ported from the SQLite reference (D8: no
stubs remain - the DDL carries the schema, and the store layer under ``_store/`` carries the SQL).

Writes run on the handle's own pooled connection through the store writers (the guard, the row
stamp and the counter bump in one ``store.tx()``); reads take ``one``/``all``/``scalar``, which a
thread's open ``read_snapshot`` reroutes onto a second pooled connection's REPEATABLE READ
transaction, so the snapshot's reads see one generation while a write commits around them. The PG
DDL has no foreign keys, so the session-row refusal is the app-level check the write verbs make
(``SessionRowMissing``), and the store's ``RuntimeError``-family refusals map to the contract's
error family through ``refusals`` (plus the rewind's verbatim refusal texts to ``RewindRefused``).

Porting notes (recorded for the stage-1 review): the payload columns (``content``/``api_content``/
``reasoning_content``) are BYTEA (PostgreSQL ``text`` cannot hold the NUL the ``\\x00json:``
sentinel encoding carries); every SQL comparison of a payload body goes through
``hermes_nul_safe_text(...)`` (NUL-safe: a payload may carry the 0x00 JSON sentinel, which
``convert_from`` rejects) and Python work through ``_payload_str``/``_payload_bytes``. Display
identity is the plain dedupe key (role, content, timestamp, tool_call_id, tool_calls, tool_name,
with the stable tool-call key for assistant rows). The detached-result appends (completion notes,
delegation deliveries) and the latest-todo-result walk are ported here over the store's read and
write doors.
"""

from __future__ import annotations

import json
import time
from contextlib import contextmanager
from typing import Any, Iterator, Mapping, Optional, Sequence

from persistence import (
    CooldownRestoreRefused, MessageId, RowVisibility, RewindRefused, SessionId,
)
from persistence.conversation.messages import (
    AnchoredView, AnchoredWindow, BatchAppendProgress, BatchAppendResult, CooldownSnapshot,
    CompressionCooldown, CompactionResult, ConversationMessage, ConversationPeer, ConversationRequest,
    ConversationRevision, DelegationDelivery, HeldMessage, MessageRecord, MessageStorageState,
    MessageRepository, MessageId, NewMessage, PagedMessage, PruneRewriteResult, PrunedRowChange,
    PrUrlHit, PromptPage, Reaction, ReplaceMode, ReplaceResult, ResumeConversations, RowDisposition,
    RowOutcome, RowStamp, RowUids, SegmentPage, SessionTimeline, TimelineEntry, TranscriptRead,
    TranscriptRow, TranscriptView, UnseenReaction, UserTurnRewind, Watermark,
)
from persistence.conversation.messages import RewindPlan as _RewindPlanRecord
from persistence.sqlite.conversation._store.core import _CONTENT_JSON_PREFIX
from persistence.sqlite.conversation._store.rules import (
    ABSORBED_ROW_IDS, DB_ROW_SNAPSHOT, RESTORED_FROM_STORE,
)
from session_rules.turns import MERGED_DURABLE_ROWS, OWN_ROW, RETIRED_DURABLE_ROWS, RETIRED_ROW, UNNAMED_DURABLE_ROWS

from .._backend import PostgresStore, refusals, register_repository
from .._stubs import add_stubs
from ._store import reads as _reads
from ._store import timeline as _timeline
from ._store import writes as _writes
from ._store.common import _check_transcript_write_guards, _ensure_display_index, _in_placeholders
from ._store.common import CONTENT_TEXT_LEN
from ._store.common import _parse_tool_calls as _parse_tool_calls_local
from ._store.common import _row_to_message_dict as _store_row_to_message_dict
# Cross-repo surface: the lineage/usage/maintenance ports import these helpers from this module (the
# stage-0 base surface); they now live in ``._store.common``.
from ._store.common import (
    _display_dedupe_key,
    _encode_content,
    _is_explicit_fork_child_row,
    _payload_bytes,
    _row_to_message_dict,
    _scrub,
)
from ._store.common import _stamp_tool_call_uids as _stamp_tool_call_uids_local
from ._store.prune_rewrite import rewrite_pruned_rows as _store_rewrite_pruned_rows
from ._store.prune_rewrite import rewritten_held_row_ids as _store_rewritten_held_row_ids
from ._store.rewind import plan_user_turn_rewind as _store_plan_user_turn_rewind
from ._store.rewind import retain_refused_turn_for_display as _store_retain_refused_turn
from ._store.rewind import rewind_user_turn as _store_rewind_user_turn
from ._store.stored_reads import display_generations as _store_display_generations
from ._store.stored_reads import segment_page_messages as _store_segment_page_messages
from ._store.stored_reads import transcript_cursor_state as _store_transcript_cursor_state
from ._store.stored_reads import transcript_display_index_complete as _store_display_index_complete
from ._store.stored_reads import transcript_handoff_rows as _store_transcript_handoff_rows
from ._store.stored_reads import transcript_handoffs_shown_before as _store_transcript_handoffs_shown_before
from ._store.stored_reads import transcript_tool_calls_before as _store_transcript_tool_calls_before

__all__ = ["PgMessageRepository"]

# The legacy store's rewind refusal texts, by the contract reason each one is (the SQLite
# ``_REWIND_REFUSALS``; the PG ``rewind`` store module raises the same verbatim texts).
_REWIND_REFUSALS = {
    "session history changed before the rewind could be persisted": "history_changed",
    "active transcript changed before the rewind could be persisted": "history_changed",
    "rewind target has no durable row identity": "target_unidentified",
    "rewind target changed before it could be persisted": "target_changed",
    "rewind did not retain its compaction handoff": "handoff_lost",
}


@contextmanager
def _refusals(store: PostgresStore, session_id: Optional[SessionId] = None) -> Iterator[None]:
    """``refusals`` (the store-level refusals to the contract's classes) plus the transcript writes'
    own: the PG DDL has no foreign keys, so a missing session row is the app-level check the verbs
    make, and a rewind's verbatim ``RuntimeError`` is ``RewindRefused`` with its reason."""
    sid = str(session_id) if session_id is not None else None
    try:
        with refusals(store, session_id):
            yield
    except RuntimeError as exc:
        reason = _REWIND_REFUSALS.get(str(exc))
        if reason is None or sid is None:
            raise
        raise RewindRefused(SessionId(sid), reason, str(exc)) from exc


def _visibility(active: Any, compacted: Any) -> RowVisibility:
    if active:
        return RowVisibility.ACTIVE
    return RowVisibility.COMPACTION_ARCHIVED if compacted else RowVisibility.REWOUND


def _payload_str(value: Any) -> Optional[str]:
    """The payload column (``content``/``api_content``/``reasoning_content``) read back from its BYTEA
    storage form: the stored UTF-8 bytes decoded (None stays None; strings pass through)."""
    if value is None:
        return None
    if isinstance(value, (bytes, bytearray, memoryview)):
        return bytes(value).decode("utf-8")
    return value


def _sidecar(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except ValueError:
        return value


# ``NewMessage`` fields that travel under their own name when set (the SQLite reference's set, all of
# it: the PG store's writers stamp every one).
_MESSAGE_OPTIONALS = (
    "tool_name", "tool_call_id", "display_kind", "api_content", "platform_message_id", "effect_disposition",
    "finish_reason", "reasoning", "reasoning_content", "reasoning_details", "codex_reasoning_items",
    "codex_message_items", "token_count", "message_uid",
)


def _message_dict(message: NewMessage) -> dict[str, Any]:
    """One ``NewMessage`` as the message dict the store's writers take (the SQLite ``_message_dict``)."""
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
        msg[DB_ROW_SNAPSHOT] = message.db_row_snapshot
    if message.absorbed_message_uids:
        msg["_absorbed_message_uids"] = list(message.absorbed_message_uids)
    if message.tool_call_uids is not None:
        msg["_tool_call_uids"] = dict(message.tool_call_uids)
    if message.tool_call_uid is not None:
        msg["_tool_call_uid"] = message.tool_call_uid
    if message.data_use:
        msg["data_use"] = list(message.data_use)
    if message.merged_durable_rows:
        msg[MERGED_DURABLE_ROWS] = message.merged_durable_rows
    if message.unnamed_durable_rows:
        msg[UNNAMED_DURABLE_ROWS] = message.unnamed_durable_rows
    if message.absorbed_row_ids:
        msg[ABSORBED_ROW_IDS] = [int(i) for i in message.absorbed_row_ids]
    if message.restored_from_store:
        msg[RESTORED_FROM_STORE] = True
    return msg


def _held_dict(held: HeldMessage) -> dict[str, Any]:
    """One ``HeldMessage`` as the held message dict the store compares (the SQLite ``_held_dict``)."""
    msg: dict[str, Any] = {"role": held.role, "content": held.content}
    if held.tool_call_id is not None:
        msg["tool_call_id"] = held.tool_call_id
    if held.tool_calls:
        msg["tool_calls"] = list(held.tool_calls)
    if held.timestamp is not None:
        msg["timestamp"] = held.timestamp
    if held.row_id is not None:
        msg["_row_id"] = int(held.row_id)
    if held.persisted:
        msg["_db_persisted"] = True
    if held.db_row_snapshot is not None:
        msg[DB_ROW_SNAPSHOT] = held.db_row_snapshot
    if held.message_uid is not None:
        msg["message_uid"] = held.message_uid
    if held.absorbed_row_ids:
        msg[ABSORBED_ROW_IDS] = [int(i) for i in held.absorbed_row_ids]
    if held.merged_durable_rows:
        msg[MERGED_DURABLE_ROWS] = held.merged_durable_rows
    if held.retired_durable_rows:
        msg[RETIRED_DURABLE_ROWS] = [_held_dict(row) for row in held.retired_durable_rows]
    if held.tool_call_ids:
        msg["tool_call_ids"] = list(held.tool_call_ids)
    if held.own_row:
        msg[OWN_ROW] = True
    if held.retired_row:
        msg[RETIRED_ROW] = True
    return msg


def _pruned_original(change: PrunedRowChange) -> dict[str, Any]:
    """The message dict the in-place rewrite names, compares and diffs: ``before`` as written, with the
    held original's fields over it, so the row id and ownership version come from the original alone."""
    before = {key: value for key, value in _message_dict(change.before).items()
              if key not in ("_row_id", DB_ROW_SNAPSHOT)}
    return {**before, **_held_dict(change.original)}


def _transcript_row(row: Mapping[str, Any]) -> TranscriptRow:
    """A stored row dict (read back with the payload columns decoded) as its record (the reference's
    ``_transcript_row`` projection: the wire fields plus the visibility and summary flag)."""
    from persistence.sqlite.conversation.messages import _WIRE_FIELDS

    fields = {name: row.get(name) for name in _WIRE_FIELDS}
    return TranscriptRow(
        id=MessageId(int(row["id"])), session_id=SessionId(str(row["session_id"])),
        visibility=_visibility(row.get("active"), row.get("compacted")),
        observed=bool(row.get("observed")),
        compressed_summary=row.get("_compressed_summary") is True,
        **fields,
    )


def _message_record(row: Mapping[str, Any]) -> MessageRecord:
    """A stored row dict as its typed record (reactions are not part of the record's metadata)."""
    from persistence.conversation.messages import MessageRecord as _Record

    from persistence.sqlite.conversation.messages import _SIDECARS

    metadata = row.get("display_metadata")
    if isinstance(metadata, dict):
        metadata = {key: value for key, value in metadata.items() if key != "reactions"} or None
    return _Record(
        id=MessageId(int(row["id"])), role=row["role"], content=row.get("content"),
        timestamp=float(row["timestamp"]), tool_calls=tuple(row.get("tool_calls") or ()),
        tool_name=row.get("tool_name"), tool_call_id=row.get("tool_call_id"),
        display_kind=row.get("display_kind"), compressed_summary=bool(row.get("_compressed_summary")),
        visibility=_visibility(row.get("active"), row.get("compacted")), api_content=row.get("api_content"),
        display_metadata=metadata, platform_message_id=row.get("platform_message_id"),
        observed=bool(row.get("observed")), effect_disposition=row.get("effect_disposition"),
        finish_reason=row.get("finish_reason"), reasoning=row.get("reasoning"),
        reasoning_content=row.get("reasoning_content"),
        **{name: _sidecar(row.get(name)) for name in _SIDECARS}, message_uid=row.get("message_uid"))


def _stamp(msg: Mapping[str, Any], stored: Mapping[int, Mapping[str, Any]]) -> RowStamp:
    """The row state a write left on one message dict; a kept row the write did not restamp takes its
    stored timestamp (the SQLite ``_stamp``)."""
    row_id = int(msg["_row_id"])
    timestamp = msg.get("timestamp")
    if timestamp is None:
        timestamp = stored[row_id]["timestamp"]
    return RowStamp(
        id=MessageId(row_id), timestamp=float(timestamp),
        db_row_snapshot=msg.get(DB_ROW_SNAPSHOT),
        message_uid=msg.get("message_uid") or stored.get(row_id, {}).get("message_uid") or "",
        tool_call_uids=msg.get("_tool_call_uids"), tool_call_uid=msg.get("_tool_call_uid"))


def _row_outcomes(rows: Sequence[Mapping[str, Any]], count: int) -> tuple[RowOutcome, ...]:
    """The batch's outcomes from the row state the write stamped on each dict and its fresh-row
    ``count``: fresh ids are minted inside the write, so they are the ``count`` largest; any other row
    resolved to an existing target: ADOPTED when it took a canonical row other than lent metadata,
    REPAIRED otherwise (the SQLite ``_row_outcomes``)."""
    fresh = set(sorted({int(row["_row_id"]) for row in rows}, reverse=True)[:count])
    outcomes = []
    for row in rows:
        row_id, canonical = int(row["_row_id"]), row.get("_canonical_row")
        if row_id in fresh:
            disposition = RowDisposition.APPENDED
        elif canonical is not None and not canonical.get("_metadata_only"):
            disposition = RowDisposition.ADOPTED
        else:
            disposition = RowDisposition.REPAIRED
        outcomes.append(RowOutcome(
            id=MessageId(row_id), disposition=disposition, timestamp=float(row["timestamp"]),
            db_row_snapshot=row.get(DB_ROW_SNAPSHOT), canonical_row=canonical,
            message_uid=row.get("message_uid"), tool_call_uids=row.get("_tool_call_uids"),
            tool_call_uid=row.get("_tool_call_uid")))
    return tuple(outcomes)


class PgMessageRepository:
    """``MessageRepository`` on the bound profile's store schema (every verb ported; the store layer
    under ``_store/`` carries the SQL, the verbs here carry the contract mapping)."""

    def __init__(self, store: PostgresStore) -> None:
        self._store = store

    # ── writes ──

    def append(self, session_id: SessionId, message: NewMessage, *,
               turn_lease_holder: Optional[str] = None, turn_lease_ttl_seconds: float = 300.0) -> MessageId:
        """Append one row and bump the session counters in one transaction; returns its id (the protocol
        docstring's refusals: ``TurnLeaseLost``, ``CompressionClosed``, ``SessionRowMissing``). A single
        append drops ``data_use`` (only the batch door folds it into display_metadata) and the given
        tool-call uid (only the batch door stores it — the SQLite ``append`` field-based port)."""
        store = self._store
        msg = _message_dict(message)
        msg.pop("data_use", None)
        msg.pop("_tool_call_uid", None)
        with _refusals(store, session_id):
            self._append_one(store, session_id, msg, turn_lease_holder=turn_lease_holder,
                             turn_lease_ttl_seconds=turn_lease_ttl_seconds)
        return MessageId(int(msg["_row_id"]))

    def append_batch(self, session_id: SessionId, messages: Sequence[NewMessage], *,
                     turn_lease_holder: Optional[str] = None,
                     chunk_rows: Optional[int] = None, turn_lease_ttl_seconds: float = 300.0) -> BatchAppendResult:
        """Append a batch of rows, one write transaction per chunk; the outcomes name each row's stamp and
        disposition (the SQLite ``append_batch``)."""
        store = self._store
        dicts = [_message_dict(m) for m in messages]
        chunks = [dicts]
        if chunk_rows is not None and len(dicts) > chunk_rows:
            tool_uid_index: dict[str, str] = {}
            for msg in dicts:  # each chunk indexes only its own rows: pair results across chunk boundaries
                _stamp_tool_call_uids_local(msg, _parse_tool_calls_local(msg.get("tool_calls")), tool_uid_index)
            chunks = [dicts[start:start + chunk_rows] for start in range(0, len(dicts), chunk_rows)]
        committed: list[RowOutcome] = []
        try:
            with _refusals(store, session_id):
                for chunk in chunks:  # one write transaction each
                    inserted = _writes.append_batch(store, str(session_id), chunk,
                                                    turn_lease_holder=turn_lease_holder,
                                                    turn_lease_ttl_seconds=turn_lease_ttl_seconds)
                    committed.extend(_row_outcomes(chunk, inserted))
        except BaseException as exc:
            exc.batch_progress = BatchAppendProgress(outcomes=tuple(committed), uids=tuple(
                RowUids(message_uid=msg.get("message_uid") or None, tool_call_uids=msg.get("_tool_call_uids"),
                        tool_call_uid=msg.get("_tool_call_uid")) for msg in dicts))
            raise
        return BatchAppendResult(outcomes=tuple(committed))

    def append_delegation_delivery(self, session_id: SessionId, delivery: DelegationDelivery) -> MessageId:
        """Record a detached delegation result once, keyed by its ``delegation_id`` (the SQLite
        ``append_delegation_delivery``; a live turn or a closed-by-compression session refuses)."""
        store = self._store
        metadata = dict(delivery.metadata)
        delegation_id = metadata.get("delegation_id")
        if not delegation_id:
            raise ValueError("Delegation delivery requires a stable delegation_id")
        kind = "hidden" if metadata.get("presentation_suppressed") else "async_delegation_complete"
        match_sql = (
            "m.display_kind IN ('async_delegation_complete', 'hidden') "
            "AND COALESCE(((CASE WHEN m.display_metadata IS NULL OR m.display_metadata = '' "
            "THEN '{}'::jsonb ELSE m.display_metadata::jsonb END) ->> 'delegation_id'), '') = %s "
            "AND COALESCE(((CASE WHEN m.display_metadata IS NULL OR m.display_metadata = '' "
            "THEN '{}'::jsonb ELSE m.display_metadata::jsonb END) ->> 'delivery_notice'), '') = %s")
        with _refusals(store, session_id):
            return MessageId(self._append_detached_result(
                store, session_id, delivery.content, kind, metadata, match_sql,
                (str(delegation_id), str(metadata.get("delivery_notice", "")))))

    def append_completion_note(self, session_id: SessionId, admission_id: str, content: str,
                              display_kind: str, metadata: Mapping[str, Any]) -> MessageId:
        """Record an unviewed completion's note once, keyed by its admission (stored as
        ``completion_admission``; the SQLite ``append_completion_note``)."""
        store = self._store
        match_sql = (
            "COALESCE(((CASE WHEN m.display_metadata IS NULL OR m.display_metadata = '' "
            "THEN '{}'::jsonb ELSE m.display_metadata::jsonb END) ->> 'completion_admission'), '') = %s")
        with _refusals(store, session_id):
            return MessageId(self._append_detached_result(
                store, session_id, content, display_kind, {**dict(metadata), "completion_admission": admission_id},
                match_sql, (str(admission_id),)))

    def replace_transcript(self, session_id: SessionId, messages: Sequence[NewMessage], mode: ReplaceMode, *,
                           reject_active_turn_lease: bool = False) -> ReplaceResult:
        """Atomically replace the session's transcript (the SQLite ``replace_transcript``: the in-order
        prefix keeps its rows; the divergent suffix is archived or deleted; a kept row that the write
        did not restamp takes its stored timestamp for its stamp)."""
        store = self._store
        dicts = [_message_dict(m) for m in messages]
        for msg in dicts:  # a replace stamps no ownership version on a row it keeps
            msg.pop(DB_ROW_SNAPSHOT, None)
        with _refusals(store, session_id):
            _writes.replace_transcript(store, str(session_id), dicts,
                                       active_only=mode is ReplaceMode.PRESERVE_EXISTING_ARCHIVE,
                                       archive_dropped=mode is ReplaceMode.ARCHIVE_DROPPED,
                                       reject_active_turn_lease=reject_active_turn_lease)
            kept = any(m.get("timestamp") is None or not m.get("message_uid") for m in dicts)
            stored = {int(r["id"]): r for r in _reads.get_messages(store, str(session_id))} if kept else {}
        return ReplaceResult(rows=tuple(_stamp(m, stored) for m in dicts))

    def clear_transcript(self, session_id: SessionId) -> None:
        store = self._store
        with _refusals(store, session_id):
            _writes.clear_transcript(store, str(session_id))

    def rewind_user_turn(self, session_id: SessionId, user_ordinal: int, *,
                         warm_history: Optional[Sequence[Mapping[str, Any]]] = None, require_retryable: bool = False,
                         require_composite: bool = False, adopt_row_ids: bool = False) -> UserTurnRewind:
        store = self._store
        with _refusals(store, session_id):
            outcome = _store_rewind_user_turn(
                store, str(session_id), user_ordinal,
                warm_history=None if warm_history is None else [dict(m) for m in warm_history],
                require_retryable=require_retryable, require_composite=require_composite,
                adopt_row_ids=adopt_row_ids)
        return UserTurnRewind(prefix=tuple(outcome.prefix), live_view=outcome.live_view,
                              live_text=outcome.live_text, rewound_count=outcome.rewound_count,
                              turns_undone=outcome.turns_undone)

    def retain_refused_turn_for_display(self, session_id: SessionId, *, notice: str,
                                        boundary_row_id: Optional[MessageId] = None) -> bool:
        store = self._store
        with _refusals(store, session_id):
            return _store_retain_refused_turn(
                store, str(session_id), notice=notice,
                boundary_row_id=None if boundary_row_id is None else int(boundary_row_id))

    def compact(self, session_id: SessionId, replacement: Sequence[NewMessage], *,
                watermark: Optional[Watermark] = None, carried_tail_count: int = 0,
                lock_holder: Optional[str] = None, model_config_patch: Optional[Mapping[str, Any]] = None,
                carried: Sequence[HeldMessage] = (), coverage: Optional[Any] = None,
                held_for_cas: Optional[Sequence[HeldMessage]] = None) -> CompactionResult:
        store = self._store
        dicts = [_message_dict(m) for m in replacement]
        with _refusals(store, session_id):
            active = _writes.archive_and_compact(
                store, str(session_id), dicts,
                model_config_patch=None if model_config_patch is None else dict(model_config_patch),
                watermark=None if watermark is None else int(watermark.last_active or 0),
                lock_holder=lock_holder, tail_count=carried_tail_count,
                carried_messages=[_held_dict(m) for m in carried] or None,
                covered_ids=None if coverage is None else [int(i) for i in coverage.covered_ids],
                unresolved_held=None if coverage is None else [_held_dict(m) for m in coverage.unresolved],
                held_for_cas=None if held_for_cas is None else [_held_dict(m) for m in held_for_cas])
        return CompactionResult(active_count=active, rows=tuple(_stamp(m, {}) for m in dicts))

    def rewrite_pruned_rows(self, session_id: SessionId, changes: Sequence[PrunedRowChange], *,
                            model_config_patch: Optional[Mapping[str, Any]] = None,
                            turn_lease_holder: Optional[str] = None,
                            turn_lease_ttl_seconds: float = 300.0) -> PruneRewriteResult:
        store = self._store
        pairs = [(_pruned_original(c), _message_dict(c.replacement)) for c in changes]
        for _, replacement in pairs:
            replacement.pop(DB_ROW_SNAPSHOT, None)
        with _refusals(store, session_id):
            rewritten = _store_rewrite_pruned_rows(
                store, str(session_id), pairs,
                model_config_patch=None if model_config_patch is None else dict(model_config_patch),
                turn_lease_holder=turn_lease_holder, turn_lease_ttl_seconds=turn_lease_ttl_seconds)
        return PruneRewriteResult(rewritten=rewritten,
                                  snapshots=tuple(replacement.get(DB_ROW_SNAPSHOT) for _, replacement in pairs))

    def purge_stale_tool_call_markers(self, *, dry_run: bool = False, backup: bool = True) -> "MarkerPurgeReport":
        store = self._store
        with _refusals(store):
            report = _writes.purge_stale_tool_call_markers(store, dry_run=dry_run, backup=backup)
        from persistence.conversation.messages import MarkerPurgeReport
        return MarkerPurgeReport(dry_run=report["dry_run"], rows_affected=report["rows_affected"],
                                 row_ids=tuple(MessageId(i) for i in report["row_ids"]),
                                 backup_path=report["backup_path"])

    def set_user_message_content(self, session_id: SessionId, message_id: MessageId, content: Any,
                                display_metadata: Optional[Mapping[str, Any]] = None) -> bool:
        store = self._store
        metadata = dict(display_metadata) if display_metadata is not None else None
        with _refusals(store, session_id):
            return bool(_writes.set_user_message_content(store, str(session_id), int(message_id),
                                                         content, metadata))

    def deactivate_message(self, session_id: SessionId, message_id: MessageId) -> bool:
        store = self._store
        with _refusals(store, session_id):
            return int(_writes.deactivate_message(store, str(session_id), int(message_id))) > 0

    def deactivate_messages_by_display_kind(self, session_id: SessionId, display_kind: str) -> int:
        store = self._store
        with _refusals(store, session_id):
            return int(_writes.deactivate_messages_by_display_kind(store, str(session_id), display_kind))

    def set_latest_matching_message_display_kind(self, session_id: SessionId, *, role: str, content: str,
                                                 display_kind: str,
                                                 display_metadata: Optional[Mapping[str, Any]] = None) -> bool:
        store = self._store
        with _refusals(store, session_id):
            return bool(_writes.set_latest_matching_message_display_kind(
                store, str(session_id), role=role, content=content, display_kind=display_kind,
                display_metadata=None if display_metadata is None else dict(display_metadata)))

    def set_latest_user_api_content(self, session_id: SessionId, content: Any, api_content: str) -> bool:
        store = self._store
        with _refusals(store, session_id):
            return bool(_writes.set_latest_user_api_content(store, str(session_id), content, api_content))

    def set_message_api_content(self, session_id: SessionId, message_id: MessageId, content: Any,
                               api_content: str) -> bool:
        store = self._store
        row_id = int(message_id)
        if row_id <= 0:
            return False
        with _refusals(store, session_id):
            return bool(_writes.set_message_api_content(store, str(session_id), row_id, content, api_content))

    def record_compression_cooldown(self, session_id: SessionId, until: float, error: Optional[str] = None) -> None:
        store = self._store
        with _refusals(store, session_id):
            _writes.record_compression_failure_cooldown(store, str(session_id), until, error)

    def clear_compression_cooldown(self, session_id: SessionId) -> None:
        store = self._store
        with _refusals(store, session_id):
            _writes.clear_compression_failure_cooldown(store, str(session_id))

    def restore_compression_cooldown(self, session_id: SessionId, snapshot: CooldownSnapshot) -> None:
        store = self._store
        row = {"session_exists": snapshot.present, "cooldown_until": snapshot.until, "error": snapshot.error}
        try:
            with _refusals(store, session_id):
                _writes.restore_compression_failure_cooldown_row(store, str(session_id), row)
        except RuntimeError as exc:
            if isinstance(exc, (CooldownRestoreRefused,)):
                raise
            raise CooldownRestoreRefused(SessionId(str(session_id)), str(exc)) from exc

    def set_reaction(self, session_id: SessionId, message_id: MessageId, emoji: Optional[str], *,
                    author: str) -> Optional[tuple[Reaction, ...]]:
        store = self._store
        with _refusals(store, session_id):
            reactions = _writes.set_message_reaction(store, str(session_id), int(message_id), emoji, author=author)
        if reactions is None:
            return None
        return tuple(Reaction(emoji=r.get("emoji"), author=r.get("author"), at=float(r.get("at") or 0.0),
                              seen=bool(r.get("seen")))
                     for r in reactions)

    def take_unseen_reactions(self, session_id: SessionId, *, author: str = "user") -> tuple[UnseenReaction, ...]:
        store = self._store
        with _refusals(store, session_id):
            pending = _writes.take_unseen_reactions(store, str(session_id), author=author)
        return tuple(UnseenReaction(row_id=MessageId(p["row_id"]), role=p["role"], emoji=p["emoji"], text=p["text"])
                     for p in pending)

    # ── reads ──

    def read_transcript(self, session_id: SessionId, request: TranscriptRead = TranscriptRead()) -> tuple[TranscriptRow, ...]:
        """The session's transcript rows as records, in insertion order (id, never timestamp); ``request``
        selects the view, paging and the lineage (the SQLite ``read_transcript`` over ``get_messages``)."""
        store = self._store
        with _refusals(store, session_id):
            rows = _reads.get_messages(
                store, str(session_id),
                include_inactive=request.view is TranscriptView.ALL,
                include_compacted=request.view is TranscriptView.DISPLAY,
                limit=request.limit, offset=request.offset, latest=request.latest,
                after_id=None if request.after_id is None else int(request.after_id),
                include_ancestors=request.include_lineage)
        return tuple(_transcript_row(row) for row in rows)

    def active_message_ids(self, session_id: SessionId) -> tuple[MessageId, ...]:
        store = self._store
        with _refusals(store, session_id):
            return tuple(MessageId(i) for i in _reads.get_active_message_ids(store, str(session_id)))

    def active_watermark(self, session_id: SessionId) -> Watermark:
        store = self._store
        with _refusals(store, session_id):
            last = _reads.get_active_message_watermark(store, str(session_id))
        return Watermark(MessageId(last) if last else None)

    def count_all(self) -> int:
        store = self._store
        with _refusals(store):
            return _reads.message_count(store)

    def display_count(self, session_id: SessionId) -> int:
        store = self._store
        with _refusals(store, session_id):
            return _reads.display_message_count(store, str(session_id))

    def conversation_revision(self, session_id: SessionId, *, include_lineage: bool = False) -> ConversationRevision:
        store = self._store
        with _refusals(store, session_id):
            ids, count, max_id, revision = _reads.conversation_revision(
                store, str(session_id), include_ancestors=include_lineage)
        return ConversationRevision(lineage_ids=tuple(SessionId(i) for i in ids), active_count=count,
                                    max_active_id=max_id, transcript_revision=revision)

    def read_conversation(self, session_id: SessionId,
                         request: ConversationRequest = ConversationRequest()) -> tuple[ConversationMessage, ...]:
        store = self._store
        with _refusals(store, session_id):
            messages = _reads.get_messages_as_conversation(
                store, str(session_id),
                include_ancestors=request.include_lineage,
                include_inactive=request.view is TranscriptView.ALL,
                repair_alternation=request.repair_alternation,
                include_row_ids=request.include_row_ids,
                include_compacted=request.view is TranscriptView.DISPLAY)
        return tuple(ConversationMessage.from_openai_message(m) for m in messages)

    def read_resume_conversations(self, session_id: SessionId) -> ResumeConversations:
        store = self._store
        with _refusals(store, session_id):
            model, display = _reads.get_resume_conversations(store, str(session_id))
        return ResumeConversations(model=tuple(ConversationMessage.from_openai_message(m) for m in model),
                                   display=tuple(ConversationMessage.from_openai_message(m) for m in display))

    def assert_resume_safe(self, session_id: SessionId, *, max_messages: int, tip_only: bool = False) -> int:
        store = self._store
        with _refusals(store, session_id):
            return _reads.assert_resume_safe(store, str(session_id), max_messages, tip_only=tip_only)

    def read_around(self, session_id: SessionId, anchor: MessageId, *, window: int = 5) -> AnchoredWindow:
        store = self._store
        with _refusals(store, session_id):
            around = _reads.get_messages_around(store, str(session_id), int(anchor), window=window)
        return AnchoredWindow(window=tuple(_message_record(r) for r in around["window"]),
                              before=around["messages_before"], after=around["messages_after"])

    def read_anchored_view(self, session_id: SessionId, anchor: MessageId, *, window: int = 5, bookend: int = 3,
                           keep_roles: Optional[tuple[str, ...]] = ("user", "assistant")) -> AnchoredView:
        store = self._store
        with _refusals(store, session_id):
            bookend = max(bookend, 0)
            primitive = _reads.get_messages_around(store, str(session_id), int(anchor), window=window)
            window_rows = primitive["window"]
            if not window_rows:
                return AnchoredView(window=(), before=0, after=0, bookend_start=(), bookend_end=())
            filtered_window = window_rows
            if keep_roles is not None:
                keep_set = set(keep_roles)
                filtered_window = [m for m in window_rows if m.get("id") == int(anchor) or m.get("role") in keep_set]
            bookend_start_rows: list[dict[str, Any]] = []
            bookend_end_rows: list[dict[str, Any]] = []
            if bookend > 0:
                role_clause = "" if keep_roles is None else f" AND role IN ({_in_placeholders(len(keep_roles))})"
                role_params: tuple = () if keep_roles is None else tuple(keep_roles)

                def _bookend(op: str, boundary_id: int, order: str) -> list[dict[str, Any]]:
                    return store.all(
                        "SELECT * FROM messages WHERE session_id = %s AND id " + op + " %s" + role_clause +
                        f" AND content IS NOT NULL AND {CONTENT_TEXT_LEN} > 0 "
                        "ORDER BY id " + order + " LIMIT %s",
                        str(session_id), boundary_id, *role_params, bookend)

                bookend_start_rows = _bookend("<", window_rows[0]["id"], "ASC")
                bookend_end_rows = list(reversed(_bookend(">", window_rows[-1]["id"], "DESC")))
            return AnchoredView(
                window=tuple(_message_record(r) for r in filtered_window),
                before=primitive["messages_before"], after=primitive["messages_after"],
                bookend_start=tuple(_message_record(_store_row_to_message_dict(r, summary_flag=False))
                                   for r in bookend_start_rows),
                bookend_end=tuple(_message_record(_store_row_to_message_dict(r, summary_flag=False))
                                 for r in bookend_end_rows))

    def find_pr_url_hits(self, session_ids: Sequence[SessionId]) -> tuple[PrUrlHit, ...]:
        store = self._store
        with _refusals(store):
            hits = _reads.find_pr_url_messages(store, [str(s) for s in session_ids])
        return tuple(PrUrlHit(session_id=SessionId(str(h["session_id"])), content=h["content"]) for h in hits)

    def latest_todo_result(self, session_id: SessionId, *, max_chars: int) -> Optional[str]:
        store = self._store
        with _refusals(store, session_id):
            return self._latest_todo_result(store, str(session_id), max_chars)

    def read_session_timeline(self, session_id: SessionId, *, limit: int = 500,
                             after_row_id: int = 0) -> SessionTimeline:
        store = self._store
        with _refusals(store, session_id):
            result = _timeline.get_session_timeline(store, str(session_id), limit=limit, after_row_id=after_row_id)
        page = result["pagination"]
        return SessionTimeline(
            entries=tuple(TimelineEntry(row_id=MessageId(e["row_id"]), preview=e["preview"],
                                       timestamp=e["timestamp"]) for e in result["entries"]),
            limit=page["limit"], after_row_id=page["after_row_id"], returned=page["returned"],
            total=page["total"], has_more=page["has_more"], next_cursor=page["next_cursor"])

    def read_prompt_page(self, session_id: SessionId, row_id: MessageId, *, limit: int = 120) -> Optional[PromptPage]:
        store = self._store
        with _refusals(store, session_id):
            result = _timeline.get_session_messages_around(store, str(session_id), int(row_id), limit=limit)
        if result is None:
            return None
        page = result["pagination"]
        return PromptPage(
            messages=tuple(_transcript_row(r) for r in result["messages"]), row_id=MessageId(page["row_id"]),
            limit=page["limit"], returned=page["returned"], offset=page["offset"], total=page["total"],
            has_older=page["has_older"], has_newer=page["has_newer"])

    def latest_conversation_role(self, session_id: SessionId) -> Optional[str]:
        store = self._store
        with _refusals(store, session_id):
            return _reads.latest_conversation_role(store, str(session_id))

    def latest_dialogue_message(self, session_id: SessionId) -> Optional[str]:
        store = self._store
        with _refusals(store, session_id):
            row = store.one(
                "SELECT content FROM messages WHERE session_id = %s AND role IN ('user', 'assistant') "
                "AND active = 1 AND content IS NOT NULL AND TRIM(hermes_nul_safe_text(content)) <> '' "
                "ORDER BY id DESC LIMIT 1", str(session_id))
        return None if row is None else _payload_str(row["content"])

    def message_storage_state(self, message_id: MessageId) -> Optional[MessageStorageState]:
        if not message_id:
            return None
        store = self._store
        with _refusals(store):
            row = store.one("SELECT session_id, active, compacted FROM messages WHERE id = %s", int(message_id))
        if row is None:
            return None
        return MessageStorageState(session_id=SessionId(str(row["session_id"])),
                                   visibility=_visibility(row["active"], row["compacted"]))

    def latest_row_id(self, session_id: SessionId, *, role: str = "user", offset: int = 0,
                      require_text: bool = True) -> Optional[MessageId]:
        store = self._store
        with _refusals(store, session_id):
            row_id = _reads.latest_message_row_id(store, str(session_id), role=role, offset=offset,
                                                 require_text=require_text)
        return None if row_id is None else MessageId(row_id)

    def message_role(self, session_id: SessionId, message_id: MessageId) -> Optional[str]:
        store = self._store
        with _refusals(store, session_id):
            return _reads.get_message_role(store, str(session_id), int(message_id))

    def resolve_active_row_id(self, session_id: SessionId, message_id: MessageId) -> Optional[MessageId]:
        store = self._store
        with _refusals(store, session_id):
            row_id = _reads.resolve_active_row_id(store, str(session_id), int(message_id))
        return None if row_id is None else MessageId(row_id)

    def rewritten_held_row_ids(self, session_id: SessionId, held: Sequence[HeldMessage]) -> tuple[MessageId, ...]:
        store = self._store
        with _refusals(store, session_id):
            return tuple(MessageId(i) for i in _store_rewritten_held_row_ids(
                store, str(session_id), [_held_dict(m) for m in held]))

    def plan_user_turn_rewind(self, session_id: SessionId, user_ordinal: int, *,
                             warm_history: Optional[Sequence[Mapping[str, Any]]] = None,
                             require_retryable: bool = False, require_composite: bool = False) -> _RewindPlanRecord:
        store = self._store
        with _refusals(store, session_id):
            plan = _store_plan_user_turn_rewind(
                store, str(session_id), user_ordinal,
                warm_history=None if warm_history is None else [dict(m) for m in warm_history],
                require_retryable=require_retryable, require_composite=require_composite)
        return _RewindPlanRecord(
            target_row_id=MessageId(plan.target_row_id),
            expected_active_ids=tuple(MessageId(i) for i in plan.expected_active_ids),
            expected_target_content=plan.expected_target_content, scaffold=plan.scaffold,
            prefix=tuple(plan.prefix), durable_prefix=tuple(plan.durable_prefix),
            live_view=plan.live_view, live_text=plan.live_text,
            durable_user=tuple(plan.durable_user), user_ordinal=plan.user_ordinal)

    def has_platform_message_id(self, session_id: SessionId, platform_message_id: str) -> bool:
        store = self._store
        with _refusals(store, session_id):
            return _reads.has_platform_message_id(store, str(session_id), platform_message_id)

    def has_gateway_input_owner(self, session_id: SessionId, owner: str) -> bool:
        store = self._store
        with _refusals(store, session_id):
            return _reads.has_gateway_input_owner(store, str(session_id), owner)

    def latest_conversation_boundary(self, peer: ConversationPeer) -> Optional[int]:
        store = self._store
        with _refusals(store):
            return _reads.latest_conversation_boundary(store, peer.session_key, peer.source)

    def read_compression_cooldown(self, session_id: SessionId, at: float) -> Optional[CompressionCooldown]:
        snapshot = self.read_compression_cooldown_snapshot(session_id)
        if snapshot.until is None or float(snapshot.until) <= at:
            return None
        return CompressionCooldown(until=float(snapshot.until), error=snapshot.error)

    def read_compression_cooldown_snapshot(self, session_id: SessionId) -> CooldownSnapshot:
        store = self._store
        with _refusals(store, session_id):
            row = _writes.get_compression_failure_cooldown_row(store, str(session_id))
        if not row["session_exists"]:
            return CooldownSnapshot(present=False)
        return CooldownSnapshot(present=True, until=row["cooldown_until"], error=row["error"])

    def terminal_tool_call_payloads(self, *, since_ts: float) -> Iterator[str]:
        store = self._store
        with _refusals(store):
            for row in store.all(
                    "SELECT tool_calls FROM messages WHERE role = 'assistant' AND tool_calls IS NOT NULL "
                    "AND tool_calls LIKE '%%terminal%%' AND timestamp >= %s", since_ts):
                yield row["tool_calls"]

    def blocked_tool_result_payloads(self, *, since_ts: float) -> Iterator[tuple[str, Optional[str]]]:
        store = self._store
        with _refusals(store):
            # Text content only (a structured payload sits behind the 0x00 sentinel, where the SQLite
            # LIKE stops; the sentinel twin has length 0) and case-insensitive, as the store's scan.
            for row in store.all(
                    f"SELECT tool_call_id, content FROM messages WHERE role = 'tool' AND tool_call_id IS NOT NULL "
                    f"AND timestamp >= %s AND {CONTENT_TEXT_LEN} > 0 "
                    f"AND (hermes_nul_safe_text(content) ILIKE '%%BLOCKED%%' "
                    f"OR hermes_nul_safe_text(content) ILIKE '%%approval%%')", since_ts):
                yield (row["tool_call_id"], _payload_str(row["content"]))

    def has_any_message(self) -> bool:
        store = self._store
        with _refusals(store):
            return store.one("SELECT 1 FROM messages LIMIT 1") is not None

    def spillover_reference_rows(self) -> Iterator[tuple[Optional[str], Optional[str]]]:
        # No row cap: a capped read answered "unknown" past it and the sweep then kept every file for good.
        store = self._store
        with _refusals(store):
            for row in store.all(
                    "SELECT content, api_content FROM messages WHERE "
                    "(hermes_nul_safe_text(content) LIKE '%spillover%' OR hermes_nul_safe_text(api_content) LIKE '%spillover%' "
                    "OR hermes_nul_safe_text(content) LIKE '%cache/pruned/%' OR hermes_nul_safe_text(api_content) LIKE '%cache/pruned/%')"):
                yield (_payload_str(row["content"]), _payload_str(row["api_content"]))

    def read_snapshot(self):
        """Every read this thread makes through this store inside the block sees one generation (the store's
        second-connection REPEATABLE READ snapshot, re-entrant)."""
        return self._store.transcript_read_snapshot()

    def display_index_complete(self, session_id: SessionId) -> bool:
        store = self._store
        with _refusals(store, session_id):
            return _store_display_index_complete(store, str(session_id))

    def ensure_display_index(self, session_id: SessionId) -> bool:
        store = self._store
        with _refusals(store, session_id):
            return _ensure_display_index(store, str(session_id))

    def display_generations(self, session_id: SessionId, lineage_ids: Sequence[SessionId]) -> tuple["PagedMessage", ...]:
        store = self._store
        with _refusals(store, session_id):
            found = _store_display_generations(store, str(session_id), [str(s) for s in lineage_ids])
        return tuple(PagedMessage(message=ConversationMessage.from_openai_message(message),
                                  position=int(durable), durable_id=MessageId(int(durable)))
                     for message, durable in found)

    def transcript_segment_page(self, session_id: SessionId, *, latest: bool, after: Optional[tuple[int, int]],
                               count: int, inclusive: bool = False) -> SegmentPage:
        store = self._store
        with _refusals(store, session_id):
            found, full, edge, start = _store_segment_page_messages(
                store, str(session_id), latest=latest,
                after=None if after is None else (int(after[0]), int(after[1])),
                count=count, inclusive=inclusive)
        return SegmentPage(
            messages=tuple(PagedMessage(message=ConversationMessage.from_openai_message(message),
                                        position=int(position), durable_id=MessageId(int(durable)))
                           for message, (position, durable) in found),
            full=full, edge=None if edge is None else (int(edge[0]), int(edge[1])), start=start)

    def transcript_handoff_rows(self, shown_ids: Sequence[MessageId], segment_ids: Sequence[SessionId]) -> Mapping[MessageId, tuple[TranscriptRow, ...]]:
        store = self._store
        with _refusals(store):
            found = _store_transcript_handoff_rows(store, [int(i) for i in shown_ids],
                                                   [str(s) for s in segment_ids])
        return {MessageId(shown): tuple(_transcript_row(row) for row in rows)
                for shown, rows in found.items()}

    def transcript_handoffs_shown_before(self, session_id: SessionId, before: int,
                                          handoffs: Sequence[tuple[Optional[str], Any]]) -> tuple[TranscriptRow, ...]:
        store = self._store
        with _refusals(store, session_id):
            rows = _store_transcript_handoffs_shown_before(store, str(session_id), before, handoffs)
        return tuple(_transcript_row(row) for row in rows)

    def transcript_tool_calls_before(self, segment_ids: Sequence[SessionId], before_id: MessageId, *,
                                     limit: int = 16) -> tuple[tuple[Optional[str], Optional[str]], ...]:
        store = self._store
        with _refusals(store):
            return tuple(_store_transcript_tool_calls_before(
                store, [str(s) for s in segment_ids], int(before_id), limit))

    def transcript_cursor_state(self, session_id: SessionId, through_id: MessageId,
                                lineage_ids: Sequence[SessionId]) -> tuple[int, int, int]:
        store = self._store
        with _refusals(store, session_id):
            return _store_transcript_cursor_state(store, str(session_id), int(through_id),
                                                 [str(s) for s in lineage_ids])

    # ── the writers' shared doors ──

    # ── cross-repo helper surface: the lineage/maintenance ports (the etc branch) call these on the
    # repository as the stage-0 base surface exposed them; the implementations live in ``._store.common``.
    def _message_row_params(self, session_id: str, role: str, msg: dict[str, Any], tool_calls: Any,
                           message_timestamp: float) -> tuple:
        from ._store import common as _common

        return _common._message_row_params(session_id, role, msg, tool_calls, message_timestamp)

    def _stamp_tool_call_uids(self, msg: dict[str, Any], tool_calls: Any, batch_index: dict[str, str]) -> None:
        from ._store import common as _common

        _common._stamp_tool_call_uids(msg, tool_calls, batch_index)

    def _append_one(self, store: PostgresStore, session_id: SessionId, msg: dict[str, Any], *,
                    turn_lease_holder: Optional[str], turn_lease_ttl_seconds: float) -> None:
        """One transcript write: the guards (with their last session-row step), the insert and the
        counter bump in one transaction (the store's ``append_batch`` door, single-row form: the
        SQLite ``append_message`` keeps the reasoning columns for every role, unlike the copy doors)."""
        with _refusals(store, session_id):
            _writes.append_batch(store, str(session_id), [msg], turn_lease_holder=turn_lease_holder,
                                 turn_lease_ttl_seconds=turn_lease_ttl_seconds, keep_all_reasoning=True)

    def _append_detached_result(self, store: PostgresStore, session_id: SessionId, content: str,
                               display_kind: str, metadata: dict[str, Any], match_sql: str,
                               params: tuple) -> int:
        """Record a between-turns result as ONE user row, including replay after a compression rotation: a
        row ``match_sql`` finds in the compression lineage is returned instead of a fresh insert. Check and
        insert share the write transaction, so independent gateway processes cannot duplicate it; a live turn
        or a closed-by-compression session refuses (the SQLite ``_append_detached_result`` port; the
        session-row step is the guard's last check, after the live-turn fence)."""
        from persistence.sqlite.conversation._store.rules import stamp_message_uid
        from ._store.common import _bump_session_counters, _message_row_params as _pg_message_row_params
        msg = {"content": content, "display_kind": display_kind, "display_metadata": metadata}
        stamp_message_uid(msg)  # a fresh occurrence: mint its durable id
        now = time.time()
        row = _pg_message_row_params(str(session_id), "user", msg, None, now, keep_reasoning=True)

        # The compression lineage: root -> tip, each hop's parent ended compression (the SQLite
        # ``_LINEAGE_MATCH_SQL`` recursive CTE, precomputed over the read path and expanded as an IN list).
        lineage_ids = _reads._resume_lineage_ids(store, str(session_id))
        marks = _in_placeholders(len(lineage_ids))
        match_params = [str(i) for i in lineage_ids] + list(params)

        def _do(conn) -> int:
            existing = conn.execute(
                f"SELECT m.id FROM messages m WHERE m.session_id IN ({marks}) AND "
                f"({match_sql}) LIMIT 1", match_params).fetchone()
            if existing is not None:
                return int(existing["id"])
            _check_transcript_write_guards(conn, str(session_id), None, reject_active_turn_lease=True)
            from ._store.common import _INSERT_MESSAGE_SQL
            new_id = conn.execute(_INSERT_MESSAGE_SQL, row).fetchone()
            _bump_session_counters(conn, str(session_id), 1, 0, unit=True)
            return int(new_id["id"])

        from ._store.common import _transcript_write
        return int(_transcript_write(store, [msg], _do))

    def _latest_todo_result(self, store: PostgresStore, session_id: str, max_chars: int) -> Optional[str]:
        """Newest visible, paired and valid Todo result in the resume display scope (the SQLite
        ``get_latest_todo_result`` port, over the PG dialect: bytea content through
        ``hermes_nul_safe_text``, the
        removed-copy check through the display copy identities, the assistant-pair check in Python)."""
        from persistence.sqlite.conversation._store.identity import _uid_list
        from persistence.sqlite.conversation._store.rules import MAX_TODO_RESULT_CHARS
        from ._store.common import _display_dedupe_key, _display_identity
        if not session_id:
            return None
        if max_chars is None:
            max_chars = MAX_TODO_RESULT_CHARS
        session_ids = _reads._resume_lineage_ids(store, session_id)

        def _todo_copies(row: Mapping[str, Any], identity: Any = None) -> list[Any]:
            hashed = identity if identity is not None else _display_identity(_display_dedupe_key(dict(row)))
            return [hashed, *((("uid", uid) for uid in (row.get("message_uid"), *
                                                       _uid_list(row.get("absorbed_message_uids"))) if uid))]

        removed: dict[Any, int] = {}
        marks = _in_placeholders(len(session_ids))
        removed_rows = store.all(
            "SELECT id, role, timestamp, tool_call_id, tool_calls, tool_name, message_uid, "
            "absorbed_message_uids, display_identity, "
            "CASE WHEN display_identity IS NULL THEN content END AS content "
            f"FROM messages WHERE session_id IN ({marks}) AND active = 0 AND compacted = 0 "
            "AND role = 'tool' AND tool_name IN ('todo_list', 'todo')", *session_ids)
        for row in removed_rows:
            for copy in _todo_copies(row, row["display_identity"]):
                removed[copy] = max(removed.get(copy, 0), int(row["id"]))

        for sid in reversed(session_ids):
            before_id = 2**63 - 1
            while True:
                rows = store.all(
                    "SELECT id, role, content, timestamp, tool_call_id, tool_calls, tool_name, message_uid, "
                    f"absorbed_message_uids FROM messages WHERE session_id = %s AND id < %s "
                    f"AND role = 'tool' AND tool_name IN ('todo_list', 'todo') AND (active = 1 OR compacted = 1) "
                    f"AND {CONTENT_TEXT_LEN} <= %s ORDER BY id DESC LIMIT 100",
                    sid, before_id, max_chars)
                if not rows:
                    break
                for row in rows:
                    call_id = row["tool_call_id"]
                    if not call_id or any(removed.get(copy, 0) > int(row["id"])
                                         for copy in _todo_copies(row)):
                        continue
                    prior = store.one(
                        "SELECT role, tool_calls FROM messages WHERE session_id = %s AND id < %s "
                        "AND role IN ('assistant', 'user', 'system') AND (active = 1 OR compacted = 1) "
                        "ORDER BY id DESC LIMIT 1", sid, int(row["id"]))
                    if prior is None or prior["role"] != "assistant":
                        continue
                    try:
                        calls = json.loads(prior["tool_calls"]) if prior["tool_calls"] else []

                        def is_todo_call(call: Any) -> bool:
                            if not isinstance(call, dict) or call.get("id") != call_id:
                                return False
                            function = call.get("function")
                            if not isinstance(function, dict):
                                return False
                            if function.get("name") in ("todo_list", "todo"):
                                return True
                            if function.get("name") != "tool_call":
                                return False
                            arguments = function.get("arguments")
                            if isinstance(arguments, str):
                                arguments = json.loads(arguments)
                            if not isinstance(arguments, dict):
                                return False
                            inner_calls = arguments.get("calls")
                            return isinstance(inner_calls, list) and any(
                                isinstance(inner, dict) and inner.get("name") in ("todo_list", "todo")
                                for inner in inner_calls)

                        paired = isinstance(calls, list) and any(is_todo_call(call) for call in calls)
                        if not paired:
                            continue
                        data = json.loads(_payload_str(row["content"]))
                        if not isinstance(data, dict) or not isinstance(data.get("todos"), list):
                            continue
                        int(data.get("revision") or 0)
                    except (ValueError, TypeError):
                        continue
                    return _payload_str(row["content"])
                before_id = int(rows[-1]["id"])
        return None


add_stubs(PgMessageRepository, MessageRepository, implemented=tuple(sorted(MessageRepository.__protocol_attrs__)))


def _register() -> None:
    register_repository("messages", lambda store: PgMessageRepository(store))


_register()

"""``WorkerExecutionRepository`` on PostgreSQL (stage 2): the fifteen verbs of the core execution
protocol — ``register``, ``adopt``, ``reserve_for_admission``, ``lose_reservation``,
``reserve_child``, ``persist_operation``, ``finish``, ``record_usage``,
``record_auxiliary_usage``, ``acquire_turn_lease``, ``renew_turn_lease``, ``cleanup_turn_lease``,
``verify_claim``, ``execution_session`` and ``has_live_worker`` — and the one shared worker-receipt
transaction every ``sequence``-taking verb runs.

The receipt transaction ports the reference's one ``apply_worker_receipt``: the epoch fence, the
assignment, the exact-duplicate replay, the live status and the strict sequence are checked before
the operation handler runs on the transaction's connection, and the receipt, the execution's
``last_sequence``/status and the revision CAS move as one atomic transition. ``worker_receipts``
has no digest column, so the result row's ``record_json`` embeds the top-level ``payload_digest``
key (a late duplicate still conflicts by digest, a terminal execution's compacted read copy still
refuses ``stale_generation``). The CAS moves only when the handler changed durable state or the
execution closes: a read-only receipt must not make a client's mutation built on the pre-read
revision fail.

D8 note: the sibling worker protocols (the ``session.*`` verbs beyond this table's handlers,
compression, compaction, receipt reads and delegations) remain D8 stubs; their ``persist_operation``
entries are unreachable this stage (an operation outside the closed table is refused
``invalid_params`` before the store).
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import re
import secrets
import time
from functools import partial
from typing import Any, Iterator, Literal, Mapping, Optional, Sequence, Unpack

from persistence.authority.coordination import (
    COMPLETION_DELIVERIES, DELEGATION_COMPLETION_STATUSES, DELEGATION_EVENT_KEYS,
    DELEGATION_MAX_DELIVERY_ATTEMPTS, DELEGATION_ORIGIN_SOURCES, DELEGATION_ROUTE_KEYS, DELEGATION_TASK_KEYS,
)
from persistence.authority.vocabulary import OwnerRule, admission_fingerprint
from persistence.conversation.messages import ConversationPeer
from persistence.conversation.sessions import SessionLifecycleStatus, TitleSource
from persistence.core.canonical import canonical_json
from persistence.core.errors import AuthorityRefusal, CooldownRestoreRefused, CompressionPublishRefused
from persistence.core.fences import AuthorityEpoch, ExecutionGeneration
from persistence.core.identity import JsonValue, MessageId, SessionId
from persistence.sqlite._schema import _NUMBERED_TITLE_RE
from persistence.sqlite._state.common import _ended_by_compression
from persistence.sqlite.conversation._store.compression import _CHAIN_CAP
from persistence.sqlite.conversation._store.local import POLICY_PREFIX
from persistence.sqlite.conversation._store.sessions import classify_session_status
from persistence.sqlite.workers._runtime_workers import RETIRED_RECEIPT, descendant_range
from persistence.workers.compaction import (
    WorkerArchiveResult, WorkerCompaction, WorkerCompressionPublication, WorkerPublishResult,
)
from persistence.workers.compression import WorkerCooldown, WorkerCooldownSnapshot
from persistence.workers.compression_reads import WorkerLineageView
from persistence.workers.delegations import (
    WorkerDelegationChild, WorkerDelegationClaim, WorkerDelegationCompletion, WorkerDelegationDispatch,
    WorkerDelegationPayload, WorkerDelegationView, delegation_payload,
)
from persistence.workers.executions import (
    AdmissionWorkerScope, ChildExecution, ChildReservation, WorkerAuxiliaryUsage, WorkerExecutionId,
    WorkerExecutionRecord, WorkerExecutionRepository, WorkerKind, WorkerMainUsage, WorkerStatus,
    WorkerTurnLease,
)
from persistence.workers.sessions import (
    RowAnnotation, ToolPin, TranscriptAppendOptions, WorkerActivity, WorkerAppendResult, WorkerBillingRoute,
    WorkerMessage, WorkerSessionCreate, WorkerTitleResult,
)
from persistence.workers.vocabulary import (
    CHILD_SEPARATOR, HANDOFF_MESSAGE_FIELDS, SIDECAR_KEYS, WORKER_MESSAGE_FIELDS, row_annotations,
)
from process_identity import holder_process_is_dead
from session_rules.activity import ActivityProvenance, bound_activity_description, normalize_activity_provenance
from session_rules.automation import GoalState, HeartbeatState, LoopState
from session_rules.legacy_local import same_owner
from session_rules.session_ids import new_session_id
from session_rules.surfaces import created_surface

from .._backend import PostgresStore, register_repository
from .._stubs import add_stubs
from ..conversation._store.common import (
    _DISPLAY_ACTIVE_CLAUSE, _ENDED_ROW_SQL, _check_transcript_write_guards, _scrub_surrogates,
)
from ..conversation._store.reads import (
    _CONVERSATION_ROW_COLUMNS, _REMOVED_IDENTITIES_SQL, _active_clause, _dedupe_display_generations,
    _in_placeholders, _rows_to_conversation,
)
from ..conversation._store.repair import resolve_and_repair_transcript_batch
from ..conversation._store.sessions_pg import _merge_model_config_json
from ..conversation._store.writes import (
    _bump_counters, _decode_content, _decode_display_metadata, _encode_content,
    _encode_display_metadata, _insert_message_rows, _payload_bytes_param, _serialize,
    _decoded_repair_row, archive_and_compact_on_conn,
)
from ..conversation.lineage import PgLineageRepository
from ..conversation.lineage import _NON_CONTINUATION_CHILD_FILTER_SQL
from ..conversation.messages import _is_explicit_fork_child_row
from ..conversation.sessions import (
    _CHAIN_STEP_SQL, _delete_unreferenced_system_prompts,
    _end_and_bump, _parse_model_config, _session_turn_lease_key_on_conn,
    _set_session_title_in_transaction, _store_system_prompt, _walk_compression_chain,
    declared_scope_identity,
)
from ..conversation.usage import _MODEL_USAGE_FIELDS, _TOKEN_COUNTERS
from .admissions import (
    _append_messages_in_transaction, _compact_terminal_receipts, _epoch_check,
    _expand_compression_lineage, _local_physical_target, _session_row,
)
from .coordination import _claim_lease_row
from .runtime import _refused, _text

__all__ = ["PgWorkerExecutionsRepository"]

# The contract vocabulary keeps the one copy.
_MESSAGE_FIELDS = WORKER_MESSAGE_FIELDS
_MAIN_USAGE_KEYS = WorkerMainUsage.__required_keys__ | WorkerMainUsage.__optional_keys__
_AUXILIARY_USAGE_KEYS = WorkerAuxiliaryUsage.__required_keys__ | WorkerAuxiliaryUsage.__optional_keys__

# The closed ``session.create`` handler's payload (the reference's lifecycle key sets).
_IDENTITY_FIELDS = frozenset({
    'user_id', 'session_key', 'chat_id', 'chat_type', 'thread_id',
    'parent_session_id', 'cwd', 'profile_name', 'git_repo_root', 'origin_json', 'display_name',
})
_CREATE_FIELDS = _IDENTITY_FIELDS | {'source', 'model', 'model_config', 'system_prompt'}
_CONSTRUCTOR_CONFIG = frozenset({'max_iterations', 'max_tokens', 'reasoning_config', 'yolo_mode'})

_UNSET_CREATED_SOURCES = frozenset({None, '', 'unknown', 'local'})


def _key_set(keys: Any, allowed: frozenset[str], required: frozenset[str] = frozenset()) -> None:
    """A keyword payload's key set, checked before the store."""
    if set(keys) - allowed or required - set(keys):
        raise AuthorityRefusal('invalid_params')


def _record(public: Mapping[str, Any]) -> WorkerExecutionRecord:
    return WorkerExecutionRecord(
        execution_id=WorkerExecutionId(public['execution_id']), session_id=SessionId(public['session_id']),
        kind=WorkerKind(public['kind']), owner_epoch=AuthorityEpoch(public['owner_epoch']),
        generation=ExecutionGeneration(public['generation']), status=WorkerStatus(public['status']),
        last_sequence=public['last_sequence'],
    )


def _worker_public(row: Mapping[str, Any]) -> dict[str, Any]:
    public = dict(row)
    public.pop('adoption_digest', None)
    return public


def _messages(messages: Sequence[Any]) -> list[Any]:
    return [dict(message) if isinstance(message, Mapping) else message for message in messages]


def _annotations(result: Mapping[str, Any]) -> tuple[RowAnnotation, ...]:
    return tuple(RowAnnotation.from_wire(annotation) for annotation in result['annotations'])


_CREATE_KEYS = WorkerSessionCreate.__required_keys__ | WorkerSessionCreate.__optional_keys__
_APPEND_OPTION_KEYS = TranscriptAppendOptions.__required_keys__ | TranscriptAppendOptions.__optional_keys__


def _secret_digest(secret: str) -> str:
    _text(secret)
    return hashlib.sha256(secret.encode('utf-8')).hexdigest()


def _worker_assignment(conn, execution_id: str, session_id: str, generation: int):
    row = conn.execute("SELECT * FROM worker_executions WHERE execution_id = %s", (execution_id,)).fetchone()
    if row is None:
        raise AuthorityRefusal('not_found')
    if row["session_id"] != session_id:
        raise AuthorityRefusal('permission_denied')
    if (type(generation) is not int or row["generation"] != generation
            or _session_row(conn, session_id)["runtime_generation"] != generation):
        raise AuthorityRefusal('stale_generation')
    return row


def _linked_worker_admission(conn, session_id: str, generation: int, owner_epoch: int, statuses: Sequence[str]):
    """The reference's ``_linked_worker_admission`` (PG dialect): the live admission linked to this
    session's lineage at the worker's owner epoch; two links is a conflict."""
    placeholders = ",".join(["%s"] * len(statuses))
    rows = conn.execute(f"""SELECT * FROM session_admissions
        WHERE generation = %s AND owner_epoch = %s AND status IN ({placeholders})
          AND (target_session_id = %s OR EXISTS (
              SELECT 1 FROM jsonb_array_elements_text(lineage_json::jsonb) je WHERE je.value = %s
          )) ORDER BY seq""",
        (generation, owner_epoch, *statuses, session_id, session_id)).fetchall()
    if len(rows) > 1:
        raise AuthorityRefusal('admission_conflict')
    return rows[0] if rows else None


def _runtime_lineage(conn, session_id: str) -> set:
    """The reference's ``runtime_lineage``: the session, its local physical target and, through the
    compression-edge walk, the whole conversation it belongs to."""
    targets = {session_id, _local_physical_target(conn, session_id)}
    row = conn.execute("SELECT chat_id FROM sessions WHERE id = %s", (session_id,)).fetchone()
    if row and row["chat_id"]:
        saved = conn.execute("SELECT value FROM state_meta WHERE key = %s",
                             (POLICY_PREFIX + row["chat_id"],)).fetchone()
        if saved is not None and session_id in json.loads(saved["value"]).get('lineage', [row["chat_id"]]):
            targets.update({row["chat_id"], _local_physical_target(conn, row["chat_id"])})
    frontier = list(targets)
    for _ in range(_CHAIN_CAP):
        frontier = [sid for sid in _expand_compression_lineage(conn, frontier) if sid not in targets]
        if not frontier:
            return targets
        targets.update(frontier)
    raise AuthorityRefusal('storage_unavailable')


def _worker_states(conn, session_id: str) -> set:
    """The reference's ``worker_states``: the live statuses across the whole lineage (one probe)."""
    targets = _runtime_lineage(conn, session_id)
    return {row["status"] for row in conn.execute(
        "SELECT DISTINCT status FROM worker_executions WHERE session_id = ANY(%s) "
        "AND status IN ('registered','running','unknown')", (sorted(targets),))}


class _CountingConnection:
    """The receipt connection with a DML rowcount ledger (the reference's ``conn.total_changes``
    delta): the revision CAS moves only when the handler changed durable state."""

    def __init__(self, conn) -> None:
        self._conn = conn
        self.changes = 0

    def execute(self, sql: str, params=None):
        result = self._conn.execute(sql, params) if params is not None else self._conn.execute(sql)
        if re.match(r"\s*(INSERT|UPDATE|DELETE)\b", sql, re.I):
            self.changes += max(getattr(result, "rowcount", 0) or 0, 0)
        return result

    def __getattr__(self, name: str) -> Any:
        return getattr(self._conn, name)


def _escape_like(text: str) -> str:
    """Escape LIKE wildcards (``%``, ``_``) so derived text matches literally; pair with ``ESCAPE '\\''."""
    return text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _row_id(payload: Mapping[str, Any]) -> int:
    row_id = payload.get('row_id')
    if type(row_id) is not int or row_id <= 0:
        raise AuthorityRefusal('invalid_params')
    return row_id


def _session_lineage_root_to_tip_on(conn, session_id: str) -> list:
    """The raw ``parent_session_id`` walk from *session_id* up to its root, root-first (the
    reference's ``_session_lineage_root_to_tip`` shape, conn-bound)."""
    if not session_id:
        return [session_id]
    chain: list = []
    current = session_id
    while current and current not in chain and len(chain) < 100:
        chain.append(current)
        row = conn.execute("SELECT parent_session_id FROM sessions WHERE id = %s", (current,)).fetchone()
        current = row["parent_session_id"] if row is not None and row["parent_session_id"] else None
    return list(reversed(chain)) or [session_id]


def _authorize_on(conn, assigned: str, target: str) -> None:
    """The reference's ``_authorize``: attribution ancestry (root-to-tip) plus the compression
    segment of *assigned* is readable; unrelated siblings and foreign trees are not."""
    _text(target)
    allowed = set(_session_lineage_root_to_tip_on(conn, assigned))
    allowed.update(_walk_compression_chain(conn, assigned))
    if target not in allowed:
        raise AuthorityRefusal('permission_denied')


def _stamp_cache_generation(conn, session_id: str, clock: str) -> str:
    """A worker's prompt-cache event with the owner's durable context generation (the reference's
    ``stamp_cache_generation``): the count at commit is the count the request saw."""
    try:
        event = json.loads(clock)
    except ValueError:
        return clock
    if not isinstance(event, dict):
        return clock
    # The reference counts on the transaction connection (its own writes are visible); an aliased
    # COUNT because the store's rows are name-keyed dicts.
    row = conn.execute("SELECT COUNT(*) AS n FROM messages WHERE session_id = %s AND active = 0 AND compacted = 1",
                       (session_id,)).fetchone()
    event['generation'] = int(row["n"]) if row else 0
    return json.dumps(event, separators=(',', ':'))


def _stamp_compression_ancestors(conn, session_id: str) -> None:
    """The reference's ``stamp_compression_ancestors`` (PG dialect): copy the tip's
    ``created_source`` onto compression ancestors that have none; a different surface stops it.
    ``unknown`` and ``local`` count as unset, as in the creator upsert; an ancestor already on the
    same surface is walked through (a NULL root behind a stamped mid segment still gets stamped)."""
    row = conn.execute("SELECT parent_session_id, created_source FROM sessions WHERE id = %s",
                       (session_id,)).fetchone()
    surface = row["created_source"] if row else None
    if surface in _UNSET_CREATED_SOURCES:
        return
    current, parent_id = session_id, row["parent_session_id"] if row else None
    for _ in range(_CHAIN_CAP):
        parent = (conn.execute("SELECT parent_session_id, created_source FROM sessions WHERE id = %s",
                               (parent_id,)).fetchone() if parent_id else None)
        if not parent or parent["created_source"] not in (*_UNSET_CREATED_SOURCES, surface):
            return
        step = conn.execute(_CHAIN_STEP_SQL, (parent_id,)).fetchone()
        if step is None or step["id"] != current:
            return
        if parent["created_source"] != surface:
            conn.execute("UPDATE sessions SET created_source = %s WHERE id = %s", (surface, parent_id))
        current, parent_id = parent_id, parent["parent_session_id"]


# ── the closed operation handlers, called only inside the receipt transaction ───────────────────


def _worker_append(store, conn, session_id: str, payload: Any) -> dict:
    if set(payload) - {'messages', 'turn_lease_holder'} or 'messages' not in payload:
        raise AuthorityRefusal('invalid_params')
    messages = payload['messages']
    if not isinstance(messages, list) or len(messages) > 1000:
        raise AuthorityRefusal('invalid_params')
    for msg in messages:
        if (not isinstance(msg, dict) or set(msg) - _MESSAGE_FIELDS
                or msg.get('role') not in ('user', 'assistant', 'system', 'tool')):
            raise AuthorityRefusal('invalid_params')
        # Existing row annotations can adopt only rows in THIS transcript.
        if '_row_id' in msg and not conn.execute(
                "SELECT 1 FROM messages WHERE id = %s AND session_id = %s",
                (msg['_row_id'], session_id)).fetchone():
            raise AuthorityRefusal('permission_denied')
    lease_holder = payload.get('turn_lease_holder')
    if lease_holder is not None:
        _text(lease_holder)
    _check_transcript_write_guards(conn, session_id, None,
                                   turn_lease_holder=lease_holder, turn_lease_ttl_seconds=300.0)
    inserted_rows = resolve_and_repair_transcript_batch(
        conn, session_id, messages,
        encode_content_fn=_encode_content,
        decode_content_fn=_decode_content,
        serialize_message_fn=partial(_serialize, session_id),
        decode_row_fn=_decoded_repair_row,
    )
    inserted, tool_calls_total = _insert_message_rows(conn, session_id, inserted_rows)
    _bump_counters(conn, session_id, inserted, tool_calls_total)
    return {'count': inserted, 'annotations': row_annotations(messages)}


def _worker_turn(store, conn, session_id: str, payload: Any, operation: str) -> dict:
    if set(payload) != {'holder', 'ttl_seconds'}:
        raise AuthorityRefusal('invalid_params')
    lease_holder = _text(payload['holder'])
    ttl = payload.get('ttl_seconds', 300)
    if type(ttl) not in (float, int) or not 0.1 <= ttl <= 3600:
        raise AuthorityRefusal('invalid_params')
    key = _session_turn_lease_key_on_conn(conn, session_id)
    now = time.time()
    if operation == 'turn.acquire':
        value = _claim_lease_row(conn, 'session_turn_leases', 'conversation_id', key,
                                 lease_holder, now, now + ttl,
                                 lambda h, e: float(e) <= now or holder_process_is_dead(h))[0]
    else:
        value = conn.execute(
            "UPDATE session_turn_leases SET expires_at = %s WHERE conversation_id = %s AND holder = %s",
            (now + ttl, key, lease_holder)).rowcount > 0
    return {'value': value}


def _worker_turn_cleanup(store, conn, session_id: str, payload: Any) -> dict:
    if set(payload) != {'target', 'holder'}:
        raise AuthorityRefusal('invalid_params')
    target = _text(payload['target'])
    lease_holder = _text(payload['holder'])
    _authorize_on(conn, session_id, target)
    key = _session_turn_lease_key_on_conn(conn, session_id)
    if key != _session_turn_lease_key_on_conn(conn, target):
        raise AuthorityRefusal('permission_denied')
    conn.execute("DELETE FROM session_turn_leases WHERE conversation_id = %s AND holder = %s",
                 (key, lease_holder))
    return {'released': True}


def _worker_usage(store, conn, session_id: str, payload: Any, *, auxiliary: bool = False) -> dict:
    source = None
    if not auxiliary:
        # ``source`` is the worker platform (turn_usage.agent_session_source). Fill only an unset
        # column, as the creator upsert does.
        raw = payload.get('source')
        source = raw if isinstance(raw, str) else None
        payload = {k: v for k, v in payload.items() if k != 'source'}
    allowed = (_MODEL_USAGE_FIELDS - {'billing_mode', 'actual_cost_usd', 'cost_status', 'cost_source'} | {'task'}) \
        if auxiliary else (_MODEL_USAGE_FIELDS | {'pricing_version', 'absolute', 'task', 'prompt_cache_clock'})
    if set(payload) - allowed:
        raise AuthorityRefusal('invalid_params')
    for key, value in payload.items():
        if key in (*_TOKEN_COUNTERS, 'api_call_count'):
            valid = type(value) is int and 0 <= value <= 2 ** 53
        elif key in ('estimated_cost_usd', 'actual_cost_usd'):
            valid = value is None or (type(value) in (int, float) and 0 <= value <= 1e12)
        elif key == 'absolute':
            valid = type(value) is bool
        else:
            valid = value is None or (isinstance(value, str) and len(value) <= 4096)
        if not valid:
            raise AuthorityRefusal('invalid_params')
    if auxiliary:
        if not payload.get('task'):
            raise AuthorityRefusal('invalid_params')
        store.usage._record_model_usage(conn, session_id, **({'api_call_count': 1} | payload))
    else:
        if payload.get('prompt_cache_clock') is not None:
            payload = dict(payload, prompt_cache_clock=_stamp_cache_generation(conn, session_id,
                                                                             payload['prompt_cache_clock']))
        store.usage._update_token_counts_in_transaction(conn, session_id, **payload)
        surfaced = created_surface(source) if source is not None else None
        if surfaced:
            conn.execute(
                "UPDATE sessions SET created_source = %s WHERE id = %s "
                "AND COALESCE(created_source, '') IN ('', 'unknown', 'local')",
                (surfaced, session_id))
            _stamp_compression_ancestors(conn, session_id)
    return {'value': None}


def _worker_finish(store, conn, session_id: str, payload: Any) -> dict:
    if payload:
        raise AuthorityRefusal('invalid_params')
    return {'status': 'terminal'}


def _worker_create(store, conn, session_id: str, payload: Any) -> dict:
    if set(payload) - _CREATE_FIELDS or 'source' not in payload:
        raise AuthorityRefusal('invalid_params')
    row = conn.execute("SELECT * FROM sessions WHERE id = %s", (session_id,)).fetchone()
    # A worker cannot mint a row or supply unreserved routing, lineage or cwd.
    if row is None or payload['source'] != row["source"]:
        raise AuthorityRefusal('permission_denied')
    for key in _IDENTITY_FIELDS:
        value = payload.get(key)
        if value is not None and value != row[key]:
            raise AuthorityRefusal('permission_denied')
    for key in _CREATE_FIELDS - {'model_config'}:
        if payload.get(key) is not None and not isinstance(payload[key], str):
            raise AuthorityRefusal('invalid_params')
    config = payload.get('model_config')
    if config is not None and (not isinstance(config, dict) or set(config) - _CONSTRUCTOR_CONFIG):
        raise AuthorityRefusal('invalid_params')
    prompt_hash = _store_system_prompt(conn, payload.get('system_prompt'))
    # The keep-existing merge (the reference upsert's CASEs): every column keeps what an earlier
    # writer set, the constructor fills only the unset. The reference fills an empty profile from
    # the owning profile name, so mirror it (the store carries it).
    profile = payload.get('profile_name')
    if not (profile or "").strip():
        profile = store.owner_profile_name
    conn.execute(
        """UPDATE sessions SET
               model = COALESCE(sessions.model, %s),
               system_prompt_hash = COALESCE(sessions.system_prompt_hash, %s),
               user_id = COALESCE(sessions.user_id, %s),
               session_key = COALESCE(sessions.session_key, %s),
               chat_id = COALESCE(sessions.chat_id, %s),
               chat_type = COALESCE(sessions.chat_type, %s),
               thread_id = COALESCE(sessions.thread_id, %s),
               parent_session_id = COALESCE(sessions.parent_session_id, %s),
               cwd = COALESCE(sessions.cwd, %s),
               profile_name = COALESCE(sessions.profile_name, %s),
               git_repo_root = COALESCE(sessions.git_repo_root, %s),
               origin_json = COALESCE(sessions.origin_json, %s),
               display_name = COALESCE(sessions.display_name, %s)
           WHERE id = %s""",
        (payload.get('model'), prompt_hash, payload.get('user_id'), payload.get('session_key'),
         payload.get('chat_id'), payload.get('chat_type'), payload.get('thread_id'),
         payload.get('parent_session_id'), payload.get('cwd'), profile,
         payload.get('git_repo_root'), payload.get('origin_json'), payload.get('display_name'),
         session_id))
    cfg = dict(config) if config else {}
    stored_raw = row["model_config"]
    stored: dict[str, Any] = {}
    if stored_raw:
        try:
            parsed = json.loads(stored_raw)
            if isinstance(parsed, dict):
                stored = parsed
        except ValueError:
            pass
    # A newly inserted /new continuation must be visible immediately: a delegate marker riding the
    # incoming config would hide the reset chat from every list (#109073).
    if (payload.get('session_key') and cfg.get('_reset_from') is not None
            and cfg.get('_delegate_from') is not None):
        cfg = {key: value for key, value in cfg.items() if key != '_delegate_from'}
    final_raw = stored_raw
    if cfg and stored.get('_reset_from') is not None and not (set(stored) - {'_reset_from'}):
        # A reset row is the chat itself: keep the reset fact, merge beside it without the delegate
        # marker.
        merged = {key: value for key, value in cfg.items() if key != '_delegate_from'}
        merged['_reset_from'] = stored['_reset_from']
        final_raw = json.dumps(merged)
    elif cfg and stored_raw is None:
        final_raw = json.dumps(cfg) if cfg else None
    if final_raw != stored_raw:
        conn.execute("UPDATE sessions SET model_config = %s WHERE id = %s", (final_raw, session_id))
    if cfg:
        # An owner-reserved child row already holds its lineage marker, so the keep-existing upsert
        # would drop these facts; merge them beside it (RFC 7386: a None value deletes its key).
        final = json.loads(final_raw) if final_raw else {}
        if isinstance(final, dict) and '_delegate_from' in final:
            patched = dict(final)
            for key, value in cfg.items():
                if value is None:
                    patched.pop(key, None)
                else:
                    patched[key] = value
            conn.execute("UPDATE sessions SET model_config = %s WHERE id = %s",
                         (json.dumps(patched), session_id))
    return {'value': session_id}


def _worker_end(store, conn, session_id: str, payload: Any) -> dict:
    if set(payload) != {'end_reason'} or not isinstance(payload['end_reason'], str) or not payload['end_reason']:
        raise AuthorityRefusal('invalid_params')
    _end_and_bump(conn, session_id, payload['end_reason'])
    return {'value': None}


def _worker_lifecycle(store, conn, session_id: str, payload: Any) -> dict:
    if payload:
        raise AuthorityRefusal('invalid_params')
    row = conn.execute(
        "SELECT role, tool_calls IS NOT NULL AS has_tool_calls, finish_reason "
        "FROM messages WHERE session_id = %s ORDER BY id DESC LIMIT 1", (session_id,)).fetchone()
    value = (classify_session_status(role=row["role"], has_tool_calls=bool(row["has_tool_calls"]),
                                    finish_reason=row["finish_reason"]) if row else 'empty')
    return {'value': value}


def _worker_title(store, conn, session_id: str, payload: Any) -> dict:
    if (set(payload) != {'title', 'source'} or payload['source'] not in ('user', 'derived', 'llm')
            or not (payload['title'] is None or isinstance(payload['title'], str))):
        raise AuthorityRefusal('invalid_params')
    try:
        value = bool(_set_session_title_in_transaction(
            conn, session_id, payload['title'], source=payload['source']))
    except ValueError:
        # Expected validation/conflict outcomes must not poison the retry journal.
        return {'value': False, 'error': 'Title unavailable or invalid', 'title_candidate': payload['title']}
    return {'value': value}


def _worker_title_source(store, conn, session_id: str, payload: Any) -> dict:
    if set(payload) != {'source'} or payload['source'] not in ('user', 'derived', 'llm'):
        raise AuthorityRefusal('invalid_params')
    return {'value': conn.execute(
        "UPDATE sessions SET title_source = %s WHERE id = %s AND title IS NOT NULL",
        (payload['source'], session_id)).rowcount > 0}


def _worker_next_title(store, conn, session_id: str, payload: Any) -> dict:
    if set(payload) != {'base_title'} or not isinstance(payload['base_title'], str):
        raise AuthorityRefusal('invalid_params')
    title = conn.execute("SELECT title FROM sessions WHERE id = %s", (session_id,)).fetchone()["title"]
    # Only an assigned title or a receipted title attempt authorizes dedupe.
    candidate = payload['base_title']
    attempted = conn.execute(
        "SELECT 1 FROM worker_receipts r JOIN worker_executions w ON w.execution_id = r.execution_id "
        "WHERE w.session_id = %s AND w.status = 'running' "
        "AND json_text_valid(r.record_json) AND (r.record_json::jsonb)->>'title_candidate' = %s LIMIT 1",
        (session_id, candidate)).fetchone()
    if candidate != title and attempted is None:
        raise AuthorityRefusal('permission_denied')
    match = _NUMBERED_TITLE_RE.match(candidate)
    base = match.group(1) if match else candidate
    rows = conn.execute(
        # SQLite's LIKE folds ASCII case; the PG twin must keep that (the reference numbers titles
        # case-insensitively on the prefix, "MINE #4" counts toward "Mine").
        "SELECT title FROM sessions WHERE title = %s OR title ILIKE %s ESCAPE '\\'",
        (base, f'{_escape_like(base)} #%')).fetchall()
    if not rows:
        return {'value': base}
    numbers = [int(m.group(2)) for m in (_NUMBERED_TITLE_RE.match(row["title"]) for row in rows) if m]
    return {'value': f'{base} #{max([1, *numbers]) + 1}'}


def _worker_activity(store, conn, session_id: str, payload: Any) -> dict:
    if (set(payload) != {'ts', 'description', 'provenance'}
            or any(payload[k] is not None and not isinstance(payload[k], str)
                   for k in ('description', 'provenance'))):
        raise AuthorityRefusal('invalid_params')
    when = payload['ts']
    if when is not None and (type(when) not in (float, int) or not math.isfinite(when)):
        raise AuthorityRefusal('invalid_params')
    when = time.time() if when is None else float(when)
    conn.execute("UPDATE sessions SET last_activity_at = %s, last_activity_description = %s, "
                 "last_activity_provenance = %s WHERE id = %s AND (last_activity_at IS NULL OR last_activity_at < %s)",
                 (when, bound_activity_description(payload['description']),
                  normalize_activity_provenance(payload['provenance']).value, session_id, when))
    return {'value': None}


def _worker_activity_clear(store, conn, session_id: str, payload: Any) -> dict:
    if payload:
        raise AuthorityRefusal('invalid_params')
    conn.execute("UPDATE sessions SET last_activity_description = %s, last_activity_provenance = %s WHERE id = %s",
                 ('', ActivityProvenance.UNKNOWN.value, session_id))
    return {'value': None}


def _worker_billing_route(store, conn, session_id: str, payload: Any) -> dict:
    if (set(payload) != {'provider', 'base_url', 'billing_mode'}
            or any(payload[k] is not None and not isinstance(payload[k], str) for k in payload)):
        raise AuthorityRefusal('invalid_params')
    conn.execute("UPDATE sessions SET billing_provider = %s, billing_base_url = %s, "
                 "billing_mode = COALESCE(%s, billing_mode) WHERE id = %s",
                 (payload['provider'], payload['base_url'], payload['billing_mode'], session_id))
    return {'value': None}


def _worker_api_content(store, conn, session_id: str, payload: Any) -> dict:
    if set(payload) != {'content', 'api_content'} or not isinstance(payload['api_content'], str):
        raise AuthorityRefusal('invalid_params')
    value = conn.execute(
        "UPDATE messages SET api_content = %s WHERE id = (SELECT id FROM messages WHERE session_id = %s "
        "AND role = 'user' AND active = 1 ORDER BY id DESC LIMIT 1) AND content = %s",
        (_payload_bytes_param(_scrub_surrogates(payload['api_content'])), session_id,
         _payload_bytes_param(_encode_content(payload['content'])))).rowcount
    return {'value': value}


def _worker_message_api_content(store, conn, session_id: str, payload: Any) -> dict:
    if set(payload) != {'row_id', 'content', 'api_content'} or not isinstance(payload['api_content'], str):
        raise AuthorityRefusal('invalid_params')
    return {'value': conn.execute(
        "UPDATE messages SET api_content = %s WHERE id = %s AND session_id = %s AND role = 'user' "
        "AND active = 1 AND content = %s",
        (_payload_bytes_param(_scrub_surrogates(payload['api_content'])), _row_id(payload), session_id,
         _payload_bytes_param(_encode_content(payload['content'])))).rowcount}


def _worker_user_content(store, conn, session_id: str, payload: Any) -> dict:
    if set(payload) - {'display_metadata'} != {'row_id', 'content'}:
        raise AuthorityRefusal('invalid_params')
    content_param = _payload_bytes_param(_encode_content(payload['content']))
    if 'display_metadata' not in payload:
        return {'value': conn.execute(
            "UPDATE messages SET content = %s WHERE id = %s AND session_id = %s AND role = 'user' AND active = 1",
            (content_param, _row_id(payload), session_id)).rowcount}
    stored = conn.execute(
        "SELECT display_metadata FROM messages WHERE id = %s AND session_id = %s AND role = 'user' AND active = 1",
        (_row_id(payload), session_id)).fetchone()
    # The PG twin of the reference's ``json_patch`` merge (RFC 7386 object merge: a None value deletes
    # its key; everything else overwrites).
    base = _decode_display_metadata(stored["display_metadata"]) if stored else None
    merged = dict(base) if base else {}
    for key, value in dict(payload['display_metadata']).items():
        if value is None:
            merged.pop(key, None)
        else:
            merged[key] = value
    return {'value': conn.execute(
        "UPDATE messages SET content = %s, api_content = NULL, display_metadata = %s "
        "WHERE id = %s AND session_id = %s AND role = 'user' AND active = 1",
        (content_param, _encode_display_metadata(merged), _row_id(payload), session_id)).rowcount}


def _worker_prompt(store, conn, session_id: str, payload: Any) -> dict:
    if set(payload) != {'system_prompt'} or not (
            payload['system_prompt'] is None or isinstance(payload['system_prompt'], str)):
        raise AuthorityRefusal('invalid_params')
    conn.execute("UPDATE sessions SET system_prompt_hash = %s, system_prompt = NULL WHERE id = %s",
                 (_store_system_prompt(conn, payload['system_prompt']), session_id))
    _delete_unreferenced_system_prompts(conn)
    return {'value': None}


def _worker_sidecars(store, conn, session_id: str, payload: Any) -> dict:
    if set(payload) != {'patch'} or not isinstance(payload['patch'], dict) or set(payload['patch']) - SIDECAR_KEYS:
        raise AuthorityRefusal('invalid_params')
    merged = _merge_model_config_json(conn, session_id, payload['patch'], on_missing='raise')
    conn.execute("UPDATE sessions SET model_config = %s WHERE id = %s", (merged, session_id))
    return {'value': None}


def _worker_tool_names(store, conn, session_id: str, payload: Any) -> dict:
    """The tools[] pin (the reference's ``worker_tool_names``): any JSON pin content-addressed through
    ``system_prompts``, ``None`` clears; bounded so a worker cannot write an unbounded blob."""
    if set(payload) != {'tool_names'}:
        raise AuthorityRefusal('invalid_params')
    pin = payload['tool_names']
    if pin is not None:
        if not isinstance(pin, (list, dict)):
            raise AuthorityRefusal('invalid_params')
        if isinstance(pin, list) and not all(isinstance(n, str) and 0 < len(n) <= 256 for n in pin):
            raise AuthorityRefusal('invalid_params')
        backend = pin.get('delegation_backend', '') if isinstance(pin, dict) else ''
        if isinstance(pin, dict) and (not {'version', 'tools'} <= set(pin) <= {'version', 'tools', 'delegation_backend'}
                                      or not isinstance(pin['version'], str) or not isinstance(pin['tools'], list)
                                      or not isinstance(backend, str) or len(backend) > 64):
            raise AuthorityRefusal('invalid_params')
    encoded = None if pin is None else json.dumps(pin)
    if encoded is not None and len(encoded) > 4 * 1024 * 1024:
        raise AuthorityRefusal('invalid_params')
    conn.execute("UPDATE sessions SET tool_names = %s WHERE id = %s",
                 (_store_system_prompt(conn, encoded), session_id))
    _delete_unreferenced_system_prompts(conn)
    return {'value': None}


def _worker_tail_role(store, conn, session_id: str, payload: Any) -> dict:
    """The message store's ``latest_conversation_role`` on the receipt connection (failed-turn boundary)."""
    if payload:
        raise AuthorityRefusal('invalid_params')
    row = conn.execute(
        "SELECT role FROM messages WHERE session_id = %s AND active = 1 "
        "AND role NOT IN ('session_meta','system') ORDER BY id DESC LIMIT 1", (session_id,)).fetchone()
    return {'value': row["role"] if row else None}


def _worker_row_role(store, conn, session_id: str, payload: Any) -> dict:
    """The message store's ``get_message_role``: the role of an active row of THIS transcript, else None."""
    if set(payload) != {'row_id'} or type(payload['row_id']) is not int:
        raise AuthorityRefusal('invalid_params')
    row = conn.execute("SELECT role FROM messages WHERE id = %s AND session_id = %s AND active = 1",
                       (payload['row_id'], session_id)).fetchone()
    return {'value': row["role"] if row else None}


# ── the compression handlers (the reference's ``_compression`` worker table, PG dialect) ─────


def _w_fields(payload: Any, required, optional=()) -> None:
    if (not isinstance(payload, dict) or set(payload) - set(required) - set(optional)
            or set(required) - set(payload)):
        raise AuthorityRefusal('invalid_params')


def _w_number(value: Any) -> float:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise AuthorityRefusal('invalid_params')
    return value


def _handoff_messages(conn, sid: str, messages: Any) -> None:
    if not isinstance(messages, list) or len(messages) > 1000:
        raise AuthorityRefusal('invalid_params')
    for msg in messages:
        if (not isinstance(msg, dict) or set(msg) - HANDOFF_MESSAGE_FIELDS
                or msg.get('role') not in ('user', 'assistant', 'system', 'tool')):
            raise AuthorityRefusal('invalid_params')
        if '_row_id' in msg and not conn.execute(
                "SELECT 1 FROM messages WHERE session_id = %s AND id = %s",
                (sid, msg['_row_id'])).fetchone():
            raise AuthorityRefusal('permission_denied')


def _w_watermarks(payload: Mapping[str, Any]) -> None:
    for field in ('watermark', 'watermark_ceiling'):
        value = payload.get(field)
        if value is not None and (type(value) is not int or value < 0):
            raise AuthorityRefusal('invalid_params')


def _set_meta_on(conn, key: str, value: str) -> None:
    conn.execute("INSERT INTO state_meta (key, value) VALUES (%s, %s) "
                 "ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value", (key, value))


def _worker_compression_lock(store, conn, sid: str, payload: Any, *, action: str) -> dict:
    if action == 'holder':
        _w_fields(payload, ())
        row = conn.execute("SELECT holder FROM compression_locks WHERE session_id = %s AND expires_at > %s",
                           (sid, time.time())).fetchone()
        return {'value': row["holder"] if row else None}
    _w_fields(payload, ('holder',), ('ttl_seconds',))
    lease_holder = _text(payload['holder'])
    now = time.time()
    ttl = _w_number(payload.get('ttl_seconds', 300.0))
    if not 0.1 <= ttl <= 3600:
        raise AuthorityRefusal('invalid_params')
    if action == 'acquire':
        value = _claim_lease_row(conn, 'compression_locks', 'session_id', sid, lease_holder, now, now + ttl,
                                 lambda h, e: e < now or holder_process_is_dead(h))[0]
    else:
        value = conn.execute("UPDATE compression_locks SET expires_at = %s WHERE session_id = %s AND holder = %s",
                             (now + ttl, sid, lease_holder)).rowcount > 0
    return {'value': value}


def _worker_cooldown_record(store, conn, sid: str, payload: Any) -> dict:
    _w_fields(payload, ('cooldown_until', 'error'))
    deadline = _w_number(payload['cooldown_until'])
    error = payload['error']
    if error is not None and not isinstance(error, str):
        raise AuthorityRefusal('invalid_params')
    conn.execute("UPDATE sessions SET compression_failure_cooldown_until = CASE "
                 "WHEN compression_failure_cooldown_until IS NOT NULL AND compression_failure_cooldown_until > %s "
                 "THEN compression_failure_cooldown_until ELSE %s END, compression_failure_error = %s WHERE id = %s",
                 (deadline, deadline, error, sid))
    return {'value': None}


def _worker_cooldown_restore(store, conn, sid: str, payload: Any) -> dict:
    _w_fields(payload, ('snapshot',))
    snapshot = payload['snapshot']
    _w_fields(snapshot, ('session_exists', 'cooldown_until', 'error'))
    if type(snapshot['session_exists']) is not bool:
        raise AuthorityRefusal('invalid_params')
    if not snapshot['session_exists']:
        raise CooldownRestoreRefused(sid, 'cannot restore absent compression cooldown row: session now exists')
    deadline, error = snapshot['cooldown_until'], snapshot['error']
    if deadline is not None:
        _w_number(deadline)
    if error is not None and not isinstance(error, str):
        raise AuthorityRefusal('invalid_params')
    conn.execute("UPDATE sessions SET compression_failure_cooldown_until = %s, compression_failure_error = %s WHERE id = %s",
                 (deadline, error, sid))
    actual = conn.execute("SELECT compression_failure_cooldown_until, compression_failure_error FROM sessions WHERE id = %s",
                          (sid,)).fetchone()
    if actual is None or (actual["compression_failure_cooldown_until"], actual["compression_failure_error"]) != (deadline, error):
        raise CooldownRestoreRefused(sid, 'compression cooldown rollback verification failed')
    return {'value': None}


def _worker_cooldown_clear(store, conn, sid: str, payload: Any) -> dict:
    _w_fields(payload, ())
    return _worker_cooldown_restore(store, conn, sid, {'snapshot': {
        'session_exists': True, 'cooldown_until': None, 'error': None}})


def _worker_compression_counter(store, conn, sid: str, payload: Any, *, column: str) -> dict:
    _w_fields(payload, ('value',))
    value = payload['value']
    if value is not None:
        _w_number(value)
    if column != 'compression_recovery_deadline' and (type(value) is not int or value < 0):
        raise AuthorityRefusal('invalid_params')
    conn.execute(f"UPDATE sessions SET {column} = %s WHERE id = %s", (value, sid))
    return {'value': None}


def _worker_watermark(store, conn, sid: str, payload: Any) -> dict:
    _w_fields(payload, ())
    return {'value': conn.execute(
        "SELECT COALESCE(MAX(id), 0) AS c FROM messages WHERE session_id = %s AND active = 1",
        (sid,)).fetchone()["c"]}


def _worker_session_row(conn, session_id: str):
    """The reference's ``worker_context`` session read: the session row with its prompt-store
    resolution (the ``compression.context`` and ``compression.publish`` handlers share it)."""
    return conn.execute(
        "SELECT s.*, COALESCE(sp.prompt, s.system_prompt) AS _system_prompt_resolved, "
        "COALESCE(tp.prompt, s.tool_names) AS _tool_names_resolved "
        "FROM sessions s LEFT JOIN system_prompts sp ON sp.hash = s.system_prompt_hash "
        "LEFT JOIN system_prompts tp ON tp.hash = s.tool_names WHERE s.id = %s",
        (session_id,)).fetchone()


def _worker_reopen(store, conn, sid: str, payload: Any) -> dict:
    _w_fields(payload, ())
    if not _ended_by_compression(conn.execute(_ENDED_ROW_SQL, (sid,)).fetchone()):
        return {'value': False}
    child = conn.execute(
        "SELECT 1 FROM sessions WHERE parent_session_id = %s"
        + _NON_CONTINUATION_CHILD_FILTER_SQL + " LIMIT 1",
        (sid,) * 4).fetchone()
    if child is not None:
        return {'value': False}
    now = time.time()
    lock_row = conn.execute("SELECT holder, expires_at FROM compression_locks WHERE session_id = %s",
                            (sid,)).fetchone()
    if lock_row is not None:
        expires_at = lock_row["expires_at"]
        if expires_at is None or float(expires_at) >= now:
            return {'value': False}
        deleted = conn.execute("DELETE FROM compression_locks WHERE session_id = %s AND holder = %s AND expires_at = %s",
                               (sid, lock_row["holder"], expires_at))
        if deleted.rowcount != 1:
            return {'value': False}
    updated = conn.execute("UPDATE sessions SET ended_at = NULL, end_reason = NULL "
                           "WHERE id = %s AND ended_at IS NOT NULL AND end_reason = 'compression'", (sid,))
    return {'value': updated.rowcount == 1}


def _worker_compression_cleanup(store, conn, sid: str, payload: Any) -> dict:
    _w_fields(payload, ('target', 'holder'))
    target = _text(payload['target'])
    lease_holder = _text(payload['holder'])
    _authorize_on(conn, sid, target)
    if _session_turn_lease_key_on_conn(conn, sid) != _session_turn_lease_key_on_conn(conn, target):
        raise AuthorityRefusal('permission_denied')
    conn.execute("DELETE FROM compression_locks WHERE session_id = %s AND holder = %s", (target, lease_holder))
    return {'value': None}


def _worker_archive(store, conn, sid: str, payload: Any) -> dict:
    _w_fields(payload, ('messages', 'model_config_patch', 'watermark', 'lock_holder', 'tail_count'),
              optional=('carried_messages', 'covered_ids', 'unresolved_held', 'held_for_cas'))
    _handoff_messages(conn, sid, payload['messages'])
    carried = payload.get('carried_messages') or []
    if not isinstance(carried, list) or any(not isinstance(m, dict) for m in carried):
        raise AuthorityRefusal('invalid_params')
    covered = payload.get('covered_ids')
    if covered is not None and (not isinstance(covered, list)
                                or any(type(i) is not int or i <= 0 for i in covered)):
        raise AuthorityRefusal('invalid_params')
    unresolved = payload.get('unresolved_held')
    if unresolved is not None and (not isinstance(unresolved, list)
                                   or any(not isinstance(m, dict) for m in unresolved)):
        raise AuthorityRefusal('invalid_params')
    cas = payload.get('held_for_cas')
    if cas is not None and (not isinstance(cas, list)
                            or any(not isinstance(m, dict) for m in cas)):
        raise AuthorityRefusal('invalid_params')
    _w_watermarks(payload)
    if payload['lock_holder'] is not None:
        _text(payload['lock_holder'])
    patch = payload['model_config_patch']
    if patch is not None and (not isinstance(patch, dict) or set(patch) - SIDECAR_KEYS):
        raise AuthorityRefusal('invalid_params')
    if type(payload['tail_count']) is not int or payload['tail_count'] < 0:
        raise AuthorityRefusal('invalid_params')
    value = archive_and_compact_on_conn(conn, sid, payload['messages'], model_config_patch=patch,
                                       watermark=payload['watermark'], lock_holder=payload['lock_holder'],
                                       tail_count=payload['tail_count'], carried_messages=carried,
                                       covered_ids=covered, unresolved_held=unresolved, held_for_cas=cas)
    return {'value': value, 'annotations': row_annotations(payload['messages'])}


def _migrate_worker_automation(conn, parent: str, child: str) -> None:
    """The reference's ``_migrate_worker_automation``: move the parent's non-cleared automation state
    onto the compression child, clearing the parent's copy (the state is keyed per physical session)."""
    for family, state_type in (('goal', GoalState), ('heartbeat', HeartbeatState), ('loop', LoopState)):
        old_key, new_key = f'{family}:{parent}', f'{family}:{child}'
        row = conn.execute("SELECT value FROM state_meta WHERE key = %s", (old_key,)).fetchone()
        if row is None:
            continue
        state = state_type.from_json(row["value"])
        if state.status == 'cleared':
            continue
        prior = conn.execute("SELECT value FROM state_meta WHERE key = %s", (new_key,)).fetchone()
        if prior is not None and (family != 'heartbeat' or state_type.from_json(prior["value"]).status != 'cleared'):
            continue
        _set_meta_on(conn, new_key, state.to_json())
        state.status = 'cleared'
        _set_meta_on(conn, old_key, state.to_json())


def _worker_publish(store, conn, sid: str, payload: Any) -> dict:
    _w_fields(payload, ('child_session_id', 'source', 'messages', 'model', 'model_config', 'system_prompt',
                        'cwd', 'profile_name', 'compression_lock_holder', 'require_compression_lease',
                        'require_lease_refresh', 'lease_ttl_seconds', 'watermark', 'watermark_ceiling'))
    child = _text(payload['child_session_id'])
    _text(payload['compression_lock_holder'])
    _w_watermarks(payload)
    if payload['require_compression_lease'] is not True or type(payload['require_lease_refresh']) is not bool:
        raise AuthorityRefusal('invalid_params')
    if not 0.1 <= _w_number(payload['lease_ttl_seconds']) <= 3600:
        raise AuthorityRefusal('invalid_params')
    _handoff_messages(conn, sid, payload['messages'])
    parent = _worker_session_row(conn, sid)
    for key in ('source', 'cwd', 'profile_name'):
        if payload[key] is not None and payload[key] != parent[key]:
            raise AuthorityRefusal('permission_denied')
    for key in ('model', 'system_prompt'):
        if payload[key] is not None and not isinstance(payload[key], str):
            raise AuthorityRefusal('invalid_params')
    cfg = payload['model_config']
    if cfg is not None and not isinstance(cfg, dict):
        raise AuthorityRefusal('invalid_params')
    cfg = dict(cfg or {})
    prior = _parse_model_config(parent['model_config'])
    for key in ('_branched_from', '_delegate_from', '_reset_from'):
        if key in cfg and cfg[key] != prior.get(key):
            raise AuthorityRefusal('permission_denied')
        if key in prior:
            cfg[key] = prior[key]
    if conn.execute("SELECT 1 FROM sessions WHERE id = %s", (child,)).fetchone():
        raise AuthorityRefusal('permission_denied')
    worker = conn.execute(
        "SELECT * FROM worker_executions WHERE session_id = %s AND status IN ('registered','running')",
        (sid,)).fetchone()
    if worker is None:
        raise AuthorityRefusal('stale_generation')
    PgLineageRepository(store)._publish_compression_child_on_conn(
        conn, parent_session_id=sid, child_session_id=child, source=payload['source'],
        messages=payload['messages'], model=payload['model'], model_config=cfg,
        system_prompt=payload['system_prompt'], cwd=payload['cwd'], profile_name=payload['profile_name'],
        compression_lock_holder=payload['compression_lock_holder'],
        require_compression_lease=payload['require_compression_lease'],
        require_lease_refresh=payload['require_lease_refresh'], lease_ttl_seconds=payload['lease_ttl_seconds'],
        watermark=payload['watermark'], watermark_ceiling=payload['watermark_ceiling'])
    _migrate_worker_automation(conn, sid, child)
    conn.execute("UPDATE sessions SET runtime_generation = %s WHERE id = %s", (worker['generation'], child))
    conn.execute("UPDATE worker_executions SET session_id = %s WHERE execution_id = %s",
                 (child, worker['execution_id']))
    # The FIFO/retry key is immutable; only physical lineage follows rotation.
    conn.execute(
        "UPDATE session_admissions SET lineage_json = (lineage_json::jsonb || to_jsonb(%s::text))::text "
        "WHERE lineage_json::jsonb -> (jsonb_array_length(lineage_json::jsonb) - 1) = to_jsonb(%s::text) "
        "AND generation = %s AND owner_epoch = %s AND status IN ('started','unknown')",
        (child, sid, worker['generation'], worker['owner_epoch']))
    return {'value': None, 'worker_assignment': {'parent': sid, 'session_id': child},
            'annotations': row_annotations(payload['messages'])}


def _context_row_dict(row: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """The context row with the prompt-store joins resolved into ``system_prompt``/``tool_names`` and
    the fork markers lifted out of ``model_config`` (the reference's ``_session_row_dict``)."""
    if row is None:
        return None
    data = dict(row)
    for column in ("system_prompt", "tool_names"):
        if f"_{column}_resolved" in data:
            resolved = data.pop(f"_{column}_resolved")
            if column in data:
                data[column] = resolved
    if not (data.get("_reset_from") and data.get("_branched_from")):
        cfg = _parse_model_config(data.get("model_config"))
        if isinstance(cfg, dict):
            if not data.get("_reset_from"):
                value = cfg.get("_reset_from")
                if isinstance(value, str) and value.strip():
                    data["_reset_from"] = value.strip()
            if not data.get("_branched_from"):
                value = cfg.get("_branched_from")
                if isinstance(value, str) and value.strip():
                    data["_branched_from"] = value.strip()
    return data


def _worker_context(store, conn, sid: str, payload: Any) -> dict:
    _w_fields(payload, ('target',))
    target = str(payload['target'])
    _authorize_on(conn, sid, target)
    row = conn.execute(
        "SELECT s.*, COALESCE(sp.prompt, s.system_prompt) AS _system_prompt_resolved, "
        "COALESCE(tp.prompt, s.tool_names) AS _tool_names_resolved "
        "FROM sessions s "
        "LEFT JOIN system_prompts sp ON sp.hash = s.system_prompt_hash "
        "LEFT JOIN system_prompts tp ON tp.hash = s.tool_names "
        "WHERE s.id = %s", (target,)).fetchone()
    return {'session': _context_row_dict(row)}


def _worker_lineage(store, conn, sid: str, payload: Any) -> dict:
    _w_fields(payload, ('target',))
    target = str(payload['target'])
    _authorize_on(conn, sid, target)
    lineage = store.lineage
    return {
        'readable_ids': _session_lineage_root_to_tip_on(conn, sid) + _walk_compression_chain(conn, sid),
        'lineage': [str(s) for s in lineage.lineage(SessionId(target))],
        'root': str(lineage.conversation_root(SessionId(target))),
        'tip': str(lineage.tip(SessionId(target))),
        'resume': str(lineage.resolve_resume_target(SessionId(target))),
        'identity': declared_scope_identity(store, target),
        'tool_root': str(lineage.tool_rotation_root(SessionId(target))),
    }


def _worker_boundary(store, conn, sid: str, payload: Any) -> dict:
    _w_fields(payload, ('session_key', 'source'))
    row = _worker_session_row(conn, sid)
    if (row['session_key'], row['source']) != (payload['session_key'], payload['source']):
        raise AuthorityRefusal('permission_denied')
    return {'value': store.messages.latest_conversation_boundary(
        ConversationPeer(session_key=payload['session_key'], source=payload['source']))}


def _is_compression_child_on(conn, child):
    """The PG ``_is_compression_child`` per-hop gate on the receipt connection."""
    parent_id = child.get("parent_session_id")
    if not parent_id or _is_explicit_fork_child_row(dict(child), include_reset=True):
        return False
    parent = conn.execute("SELECT end_reason FROM sessions WHERE id = %s", (parent_id,)).fetchone()
    return parent is not None and parent["end_reason"] == "compression"


def _resume_lineage_ids_on(conn, session_id: str) -> list:
    """The PG ``_resume_lineage_ids`` walk on the receipt connection (the verified compression lineage,
    root first)."""
    if not session_id:
        return [session_id]
    session = conn.execute("SELECT * FROM sessions WHERE id = %s", (session_id,)).fetchone()
    if session is None or _is_explicit_fork_child_row(dict(session)):
        return [session_id]
    chain = [session_id]
    current = session
    seen = {session_id}
    while len(chain) < 100:
        if not _is_compression_child_on(conn, current):
            break
        parent = conn.execute("SELECT * FROM sessions WHERE id = %s", (current["parent_session_id"],)).fetchone()
        if parent is None or parent["id"] in seen:
            break
        seen.add(parent["id"])
        chain.append(parent["id"])
        current = parent
    return list(reversed(chain))


def _with_removed_identities_on(conn, rows: list, session_ids: list) -> list:
    """The reference's ``_with_removed_identities`` on the receipt connection: the display set merged
    with the removed rows' identities (a message whose newest copy was removed stays hidden)."""
    sql = _REMOVED_IDENTITIES_SQL.format(ids=_in_placeholders(len(session_ids)))
    removed = conn.execute(sql, tuple(session_ids)).fetchall()
    return sorted([*rows, *removed], key=lambda row: row["id"]) if removed else list(rows)


def _worker_history(store, conn, sid: str, payload: Any) -> dict:
    _w_fields(payload, ('target', 'include_ancestors', 'include_inactive', 'repair_alternation',
                        'include_row_ids', 'include_compacted'))
    if any(type(v) is not bool for k, v in payload.items() if k != 'target'):
        raise AuthorityRefusal('invalid_params')
    target = str(payload['target'])
    _authorize_on(conn, sid, target)
    ids = _resume_lineage_ids_on(conn, target) if payload['include_ancestors'] else [target]
    active = _active_clause(payload['include_inactive'], payload['include_compacted'])
    display = payload['include_compacted'] and not payload['include_inactive']
    rows = conn.execute(
        f"SELECT {'session_id, compacted, display_order, ' if display else ''}{_CONVERSATION_ROW_COLUMNS} "
        f"FROM messages WHERE session_id IN ({_in_placeholders(len(ids))})"
        f"{_DISPLAY_ACTIVE_CLAUSE if display else active} ORDER BY id",
        tuple(ids)).fetchall()
    if display:
        rows = _with_removed_identities_on(conn, rows, ids)
    if payload['include_compacted']:
        rows = _dedupe_display_generations(rows, drop_removed=display)
    return {'messages': _rows_to_conversation(rows, session_id=target,
                                              include_ancestors=payload['include_ancestors'],
                                              repair_alternation=payload['repair_alternation'],
                                              include_row_ids=payload['include_row_ids'])}


def _worker_compression_append(store, conn, sid: str, payload: Any) -> dict:
    _w_fields(payload, ('messages', 'compression_lock_holder', 'turn_lease_holder', 'turn_lease_ttl_seconds'))
    _handoff_messages(conn, sid, payload['messages'])
    for key in ('compression_lock_holder', 'turn_lease_holder'):
        if payload[key] is not None:
            _text(payload[key])
    if not 0.1 <= _w_number(payload['turn_lease_ttl_seconds']) <= 3600:
        raise AuthorityRefusal('invalid_params')
    count = _append_messages_in_transaction(
        conn, sid, payload['messages'],
        compression_lock_holder=payload['compression_lock_holder'],
        turn_lease_holder=payload['turn_lease_holder'],
        turn_lease_ttl_seconds=payload['turn_lease_ttl_seconds'])
    return {'count': count, 'annotations': row_annotations(payload['messages'])}


# ── the delegation-ledger handlers (the reference's ``_gateway`` worker table, PG dialect) ──────


def _d_shape(payload: Any, keys) -> None:
    if not isinstance(payload, dict) or set(payload) != set(keys):
        raise AuthorityRefusal('invalid_params')


def _d_text(value: Any) -> str:
    if not isinstance(value, str) or not value or len(value) > 512:
        raise AuthorityRefusal('invalid_params')
    return value


def _d_number(value: Any) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise AuthorityRefusal('invalid_params')
    return value


def _d_owned(conn, session_id: str, execution_id: str, delegation_id: Any):
    _d_text(delegation_id)
    row = conn.execute("SELECT * FROM async_delegations WHERE delegation_id = %s",
                       (delegation_id,)).fetchone()
    if row is None:
        raise AuthorityRefusal('not_found')
    if row['parent_session_id'] != session_id or row['owner_execution_id'] != execution_id:
        raise AuthorityRefusal('permission_denied')
    return row


def _d_dispatch(store, conn, session_id: str, payload: Any, *, execution_id: str, worker_pid: int,
                owner_started_at, probe_refused: str | None) -> dict:
    _d_shape(payload, {'delegation_id', 'task', 'dispatched_at', *DELEGATION_ROUTE_KEYS})
    _d_text(payload['delegation_id'])
    _d_number(payload['dispatched_at'])
    if not isinstance(payload['task'], dict) or set(payload['task']) - DELEGATION_TASK_KEYS:
        raise AuthorityRefusal('invalid_params')
    session = conn.execute("SELECT source, session_key FROM sessions WHERE id = %s", (session_id,)).fetchone()
    if session is None or session['source'] not in DELEGATION_ORIGIN_SOURCES:
        raise AuthorityRefusal('unsupported_producer')
    if (payload['parent_session_id'] != session_id
            or payload['session_key'] != (session['session_key'] or '')
            or payload['origin_ui_session_id'] not in ('', session_id)
            or payload['origin_session_id'] not in ('', session_id)):
        raise AuthorityRefusal('permission_denied')
    if conn.execute("SELECT 1 FROM async_delegations WHERE delegation_id = %s",
                    (payload['delegation_id'],)).fetchone() is not None:
        _d_owned(conn, session_id, execution_id, payload['delegation_id'])
        # Exact network retries are resolved by worker_receipts before this handler.
        raise AuthorityRefusal('admission_conflict')
    if probe_refused is not None:
        raise AuthorityRefusal(probe_refused)
    now = time.time()
    conn.execute(
        """INSERT INTO async_delegations
        (delegation_id,origin_session,origin_ui_session_id,parent_session_id,state,dispatched_at,
         updated_at,owner_pid,owner_started_at,task_json,origin_session_id,owner_execution_id)
        VALUES (%s, %s, %s, %s, 'running', %s, %s, %s, %s, %s, %s, %s)""",
        (payload['delegation_id'], payload['session_key'], payload['origin_ui_session_id'], session_id,
         payload['dispatched_at'], now, worker_pid, owner_started_at, json.dumps(payload['task']),
         payload['origin_session_id'], execution_id))
    return {'value': None}


def _d_complete(store, conn, session_id: str, payload: Any, *, execution_id: str) -> dict:
    _d_shape(payload, {'event', 'result'} | ({'delivery'} if 'delivery' in payload else set()))
    event, delivery = payload['event'], payload.get('delivery', 'pending')
    if (delivery not in COMPLETION_DELIVERIES or not isinstance(event, dict)
            or set(event) - DELEGATION_EVENT_KEYS
            or not isinstance(payload['result'], dict) or event.get('type') != 'async_delegation'
            or event.get('status') not in DELEGATION_COMPLETION_STATUSES):
        raise AuthorityRefusal('invalid_params')
    row = _d_owned(conn, session_id, execution_id, event.get('delegation_id'))
    if any(event.get(key, '') != (row[column] or '') for key, column in DELEGATION_ROUTE_KEYS.items()):
        raise AuthorityRefusal('permission_denied')
    _d_number(event.get('completed_at'))
    if row['state'] != 'running':
        raise AuthorityRefusal('admission_conflict')
    conn.execute(
        "UPDATE async_delegations SET state = %s, completed_at = %s, updated_at = %s, event_json = %s, "
        "result_json = %s, delivery_state = %s WHERE delegation_id = %s",
        (event['status'], event['completed_at'], time.time(), json.dumps(event),
         json.dumps(payload['result']), delivery, event['delegation_id']))
    return {'value': None}


def _d_child(store, conn, session_id: str, payload: Any, *, execution_id: str) -> dict:
    _d_shape(payload, {'delegation_id', 'entry'})
    row = _d_owned(conn, session_id, execution_id, payload['delegation_id'])
    entry = payload['entry']
    if not isinstance(entry, dict) or type(entry.get('task_index')) is not int or entry['task_index'] < 0:
        raise AuthorityRefusal('invalid_params')
    if row['state'] != 'running':
        return {'value': None}
    previous = json.loads(row['result_json'] or '{}')
    results = [item for item in previous.get('results', []) if item.get('task_index') != entry['task_index']]
    results.append(entry)
    conn.execute("UPDATE async_delegations SET result_json = %s, updated_at = %s WHERE delegation_id = %s",
                (json.dumps({'results': results, 'partial': True}), time.time(), payload['delegation_id']))
    return {'value': None}


def _d_read(store, conn, session_id: str, payload: Any, *, execution_id: str) -> dict:
    _d_shape(payload, {'delegation_id'})
    row = _d_owned(conn, session_id, execution_id, payload['delegation_id'])
    return {'value': {key: row[key] for key in ('delegation_id', 'origin_session', 'state', 'dispatched_at',
            'completed_at', 'delivery_state', 'delivery_attempts', 'origin_session_id')}
            | {'result': json.loads(row['result_json']) if row['result_json'] else None}}


def _d_claim(store, conn, session_id: str, payload: Any, *, execution_id: str) -> dict:
    _d_shape(payload, {'delegation_id', 'claim_id'})
    row = _d_owned(conn, session_id, execution_id, payload['delegation_id'])
    _d_text(payload['claim_id'])
    now = time.time()
    if row['state'] == 'running':
        return {'value': False}
    changed = conn.execute(
        "UPDATE async_delegations SET delivery_claim = %s, delivery_claimed_at = %s, "
        "delivery_attempts = delivery_attempts + 1, updated_at = %s WHERE delegation_id = %s "
        "AND delivery_state = 'pending' AND (delivery_claim IS NULL OR delivery_claimed_at < %s)",
        (payload['claim_id'], now, now, payload['delegation_id'], now - 300)).rowcount
    return {'value': changed == 1}


# The ``delegation.ack``, ``drop``, ``defer`` and ``release`` updates of a claimed delivery: the closed
# SET fragment and the parameters it consumes BEFORE the common tail (now, delegation_id, claim_id).
_DELIVERY_UPDATE_FRAGMENTS = {
    'ack': ("delivery_state = 'delivered', delivered_at = %s", ('now',)),
    'drop': ("delivery_state = 'dropped'", ()),
    'defer': ("delivery_attempts = CASE WHEN delivery_attempts > 1 THEN delivery_attempts - 1 ELSE 0 END", ()),
    'release': ("delivery_state = CASE WHEN delivery_attempts >= %s THEN 'dropped' ELSE delivery_state END",
                ('cap',)),
}


def _d_delivery(store, conn, session_id: str, payload: Any, *, execution_id: str, action: str) -> dict:
    _d_shape(payload, {'delegation_id', 'claim_id'})
    _d_owned(conn, session_id, execution_id, payload['delegation_id'])
    _d_text(payload['claim_id'])
    # Only constant, closed fragments enter this query.
    fragment, names = _DELIVERY_UPDATE_FRAGMENTS[action]
    now = time.time()
    values = {'now': now, 'cap': DELEGATION_MAX_DELIVERY_ATTEMPTS}
    changed = conn.execute(
        "UPDATE async_delegations SET " + fragment + ", updated_at = %s, delivery_claim = NULL, "
        "delivery_claimed_at = NULL WHERE delegation_id = %s AND delivery_state = 'pending' "
        "AND delivery_claim = %s",
        (*[values[n] for n in names], now, payload['delegation_id'], payload['claim_id'])).rowcount
    return {'value': changed == 1}


def _d_unscheduled(store, conn, session_id: str, payload: Any, *, execution_id: str) -> dict:
    _d_shape(payload, {'delegation_id'})
    row = _d_owned(conn, session_id, execution_id, payload['delegation_id'])
    if row['state'] != 'running' or row['result_json'] is not None:
        raise AuthorityRefusal('admission_conflict')
    conn.execute("DELETE FROM async_delegations WHERE delegation_id = %s", (payload['delegation_id'],))
    return {'value': None}


def _delegation_handlers(execution_id: str, worker_pid: int, worker_birth: float, owner_started_at,
                         probe_refused: str | None) -> dict:
    """The reference's ``worker_delegation_handlers``: register these only after gateway worker
    identity verification."""
    # The gateway's execution-id text check (non-empty, at most 512) — the store's identity ``_text``
    # is wider (1024) and would let an overlong id reach the ledger lookup as ``not_found``.
    if not isinstance(execution_id, str) or not execution_id or len(execution_id) > 512:
        raise AuthorityRefusal('invalid_params')
    if type(worker_pid) is not int or worker_pid <= 0:
        raise AuthorityRefusal('invalid_params')
    _d_number(worker_birth)
    handlers = {name: partial(fn, execution_id=execution_id) for name, fn in {
        'complete': _d_complete, 'child': _d_child, 'read': _d_read, 'claim': _d_claim,
        'unscheduled': _d_unscheduled}.items()}
    handlers['dispatch'] = partial(_d_dispatch, execution_id=execution_id, worker_pid=worker_pid,
                                  owner_started_at=owner_started_at, probe_refused=probe_refused)
    handlers.update({name: partial(_d_delivery, execution_id=execution_id, action=name)
                     for name in _DELIVERY_UPDATE_FRAGMENTS})
    return {'delegation.' + name: handler for name, handler in handlers.items()}


def _reserve_child_execution(store, conn, session_id: str, payload: Any, *, epoch: int, execution_id: str,
                            identity: Mapping[str, Any], principal: str, limits: tuple,
                            refused: str | None) -> dict:
    """The ``child.reserve`` handler: a new subagent session and running child execution under this
    worker, within the depth and live-children limits (a refusal is returned, not raised)."""
    secret = payload.get('secret') if isinstance(payload, dict) else None
    if set(payload) != {'secret'} or not isinstance(secret, str) or not 16 <= len(secret) <= 256:
        raise AuthorityRefusal('invalid_params')
    max_children, max_depth = limits
    prefix = execution_id + CHILD_SEPARATOR
    low, high = descendant_range(execution_id)
    live = conn.execute(
        "SELECT COUNT(*) AS live FROM worker_executions WHERE execution_id >= %s AND execution_id < %s "
        "AND strpos(substring(execution_id FROM %s), '/') = 0 AND status <> 'terminal'",
        (low, high, len(prefix) + 1)).fetchone()["live"]
    if refused is None and prefix.count(CHILD_SEPARATOR) > max_depth:
        refused = 'delegation_depth_exceeded'
    if refused is None and live >= max_children:
        refused = 'delegation_capacity'
    if refused is not None:
        return {'refused': refused}
    parent = conn.execute("SELECT profile_name FROM sessions WHERE id = %s", (session_id,)).fetchone()
    child_sid, child_eid = new_session_id(), prefix + secrets.token_hex(8)
    conn.execute(
        "INSERT INTO sessions (id, source, created_source, started_at, model_config, parent_session_id, profile_name) "
        "VALUES (%s, 'subagent', 'subagent', %s, %s, %s, %s)",
        (child_sid, time.time(), json.dumps({'_delegate_from': session_id}), session_id,
         parent["profile_name"] if parent else None))
    scope = dict(identity, session_id=child_sid, execution_id=child_eid, generation=0, secret=secret)
    claim = admission_fingerprint(canonical_target=child_sid, payload=scope | {'principal': principal})
    conn.execute(
        "INSERT INTO worker_executions (execution_id, session_id, kind, owner_epoch, generation, status, adoption_digest) "
        "VALUES (%s, %s, 'child', %s, 0, 'running', %s)",
        (child_eid, child_sid, epoch, _secret_digest(claim)))
    return {'child': {'session_id': child_sid, 'execution_id': child_eid, 'generation': 0}}


# ── the one worker receipt transaction ───────────────────────────────────────────────────────────


def _worker_receipt_assignment(conn, execution_id: str, session_id: str, generation: int, sequence: int,
                              digest: str):
    """Only an identical already-committed publication can retry its old assignment (the reference's
    ``worker_receipt_assignment``; the embedded-digest shape: the stored result row carries the
    digest beside its ``worker_assignment``)."""
    receipt = conn.execute(
        "SELECT record_json FROM worker_receipts WHERE execution_id = %s AND sequence = %s",
        (execution_id, sequence)).fetchone()
    if receipt is not None:
        stored = json.loads(receipt["record_json"])
        if stored.get('payload_digest') == digest:
            assignment = stored.get('worker_assignment')
            if assignment and assignment.get('parent') == session_id:
                current = conn.execute("SELECT session_id FROM worker_executions WHERE execution_id = %s",
                                       (execution_id,)).fetchone()
                return _worker_assignment(conn, execution_id, current["session_id"], generation)
    return _worker_assignment(conn, execution_id, session_id, generation)


def _apply_worker_receipt(store: PostgresStore, *, epoch: Any, execution_id: Any, session_id: Any,
                         generation: Any, sequence: Any, operation: str, payload: Any,
                         handlers: Mapping[str, Any]) -> Mapping[str, Any]:
    """The one worker receipt transaction every ``worker.persist`` family shares (the reference's
    ``apply_worker_receipt``). An exact (execution, sequence, digest) retry returns the stored
    result; otherwise the epoch, assignment, live status and next sequence are fenced,
    ``handlers[operation]`` runs on the transaction connection (it must not commit), and its result
    is stored as the receipt. ``execution.finish`` is the only operation that makes the execution
    terminal."""
    if type(sequence) is not int or sequence < 1 or not isinstance(operation, str) or operation not in handlers:
        raise AuthorityRefusal('invalid_params')
    encoded = canonical_json(payload)
    if len(encoded.encode('utf-8', errors='surrogatepass')) > 4 * 1024 * 1024:
        raise AuthorityRefusal('invalid_params')
    digest = admission_fingerprint(canonical_target=session_id,
                                   payload={'operation': operation, 'payload': json.loads(encoded)})

    def write(conn) -> Mapping[str, Any]:
        _epoch_check(conn, epoch)
        row = _worker_receipt_assignment(conn, execution_id, session_id, generation, sequence, digest)
        if row["owner_epoch"] != epoch:
            raise AuthorityRefusal('stale_epoch')
        old = conn.execute("SELECT record_json FROM worker_receipts WHERE execution_id = %s AND sequence = %s",
                           (execution_id, sequence)).fetchone()
        if old is not None:
            stored = json.loads(old["record_json"])
            if stored == RETIRED_RECEIPT:
                # A compacted read copy: its digest was discarded with it, so an exact replay is
                # refused ``stale_generation`` like the reference's post-retirement read copies
                # (a late duplicate with other content can no longer conflict by digest — the
                # marker carries none; only exact replays of read receipts are compacted here).
                raise AuthorityRefusal('stale_generation')
            if stored.get('payload_digest') != digest:
                raise AuthorityRefusal('admission_conflict')
            return {k: v for k, v in stored.items() if k != 'payload_digest'}
        if row["status"] not in ('registered', 'running'):
            raise AuthorityRefusal('stale_generation')
        if sequence != row["last_sequence"] + 1:
            raise AuthorityRefusal('invalid_params')
        # The CAS fence moves only when the handler changed durable state (or the execution closes):
        # a read-only receipt must not make a client's mutation built on the pre-read revision fail
        # revision_conflict.
        counter = _CountingConnection(conn)
        result = handlers[operation](store, counter, session_id, json.loads(encoded))
        changed = counter.changes > 0 or operation == 'execution.finish'
        now = time.time()
        record = dict(result)
        record['payload_digest'] = digest
        conn.execute("INSERT INTO worker_receipts (execution_id, sequence, record_json, created_at, updated_at) "
                     "VALUES (%s, %s, %s, %s, %s)",
                     (execution_id, sequence, json.dumps(record, ensure_ascii=True, allow_nan=False), now, now))
        conn.execute("UPDATE worker_executions SET last_sequence = %s, status = %s WHERE execution_id = %s",
                     (sequence, 'terminal' if operation == 'execution.finish' else 'running', execution_id))
        if changed:
            conn.execute("UPDATE sessions SET runtime_revision = runtime_revision + 1 WHERE id = %s", (session_id,))
        if operation == 'execution.finish':
            _compact_terminal_receipts(conn, execution_id)
        return result

    with store.tx() as conn:
        return write(conn)


def _worker_handlers() -> dict:
    """The closed table of receipted worker operations: the core execution verbs, the ``session.*``
    family, the compression family (the reference's ``WORKER_COMPRESSION_HANDLERS``) — an operation
    outside the table is refused ``invalid_params`` before the store. The delegation-ledger
    operations ride their own bound table (``_delegation_handlers``) on the same receipt stream."""
    return {
        'execution.finish': _worker_finish,
        'usage.main': _worker_usage,
        'usage.auxiliary': lambda store, conn, sid, p: _worker_usage(store, conn, sid, p, auxiliary=True),
        **{name: (lambda store, conn, sid, p, op=name: _worker_turn(store, conn, sid, p, op))
           for name in ('turn.acquire', 'turn.renew')},
        'turn.cleanup': _worker_turn_cleanup,
        'transcript.append': _worker_append,
        'session.title': _worker_title,
        'session.title_source': _worker_title_source,
        'session.next_title': _worker_next_title,
        'session.activity': _worker_activity,
        'session.activity_clear': _worker_activity_clear,
        'session.billing_route': _worker_billing_route,
        'session.api_content': _worker_api_content,
        'session.message_api_content': _worker_message_api_content,
        'session.user_content': _worker_user_content,
        'session.create': _worker_create,
        'session.end': _worker_end,
        'session.lifecycle': _worker_lifecycle,
        'session.prompt': _worker_prompt,
        'session.sidecars': _worker_sidecars,
        'session.tools': _worker_tool_names,
        'session.tail_role': _worker_tail_role,
        'session.row_role': _worker_row_role,
        'compression.append': _worker_compression_append,
        'compression.reopen': _worker_reopen,
        'compression.cleanup': _worker_compression_cleanup,
        'compression.watermark': _worker_watermark,
        'compression.archive': _worker_archive,
        'compression.publish': _worker_publish,
        'compression.context': _worker_context,
        'compression.lineage': _worker_lineage,
        'compression.boundary': _worker_boundary,
        'compression.history': _worker_history,
        **{'compression.lock.' + action: partial(_worker_compression_lock, action=action)
           for action in ('acquire', 'renew', 'holder')},
        'compression.cooldown.record': _worker_cooldown_record,
        'compression.cooldown.restore': _worker_cooldown_restore,
        'compression.cooldown.clear': _worker_cooldown_clear,
        **{'compression.' + name: partial(_worker_compression_counter, column=column)
           for name, column in (('fallback', 'compression_fallback_streak'),
                               ('ineffective', 'compression_ineffective_count'),
                               ('recovery', 'compression_recovery_deadline'))},
    }


class PgWorkerExecutionsRepository:
    """``WorkerExecutionRepository`` on PostgreSQL (stage 2): the fifteen verbs of the core
    execution protocol; every other worker-protocol verb raises ``NotImplementedError`` naming
    itself (D8)."""

    def __init__(self, store: PostgresStore) -> None:
        self._store = store

    # ── the shared receipted path ────────────────────────────────────────────

    def _persist(self, operation: str, payload: Any, *, epoch: Any, execution_id: Any, session_id: Any,
                 generation: Any, sequence: Any) -> Mapping[str, Any]:
        with _refused(self._store, session_id):
            return _apply_worker_receipt(self._store, epoch=epoch, execution_id=execution_id,
                                        session_id=session_id, generation=generation, sequence=sequence,
                                        operation=operation, payload=payload,
                                        handlers=_worker_handlers())

    def _value(self, operation: str, payload: Any, epoch: Any, execution_id: Any, session_id: Any, generation: Any,
               sequence: Any) -> Any:
        return self._persist(operation, payload, epoch=epoch, execution_id=execution_id, session_id=session_id,
                             generation=generation, sequence=sequence)['value']

    # ── WorkerExecutionRepository (stage 2) ─────────────────────────────────

    def register(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId,
                 generation: ExecutionGeneration, kind: WorkerKind, adoption_secret: str,
                 require_idle: bool = False) -> WorkerExecutionRecord:
        for value in (execution_id, session_id):
            _text(value)
        kind_value = getattr(kind, 'value', kind)
        if kind_value not in ('cron', 'child', 'compute', 'kanban'):
            raise AuthorityRefusal('invalid_params')
        digest = _secret_digest(adoption_secret)

        def write(conn):
            _epoch_check(conn, epoch)
            session = _session_row(conn, session_id)
            if type(generation) is not int or session["runtime_generation"] != generation:
                raise AuthorityRefusal('stale_generation')
            old = conn.execute("SELECT * FROM worker_executions WHERE execution_id = %s",
                               (execution_id,)).fetchone()
            if old is not None:
                if (old["session_id"], old["generation"], old["kind"], old["owner_epoch"], old["adoption_digest"]) != (
                        session_id, generation, kind_value, epoch, digest):
                    raise AuthorityRefusal('admission_conflict')
                return _worker_public(old)
            targets = _runtime_lineage(conn, session_id)
            if require_idle and conn.execute(
                    "SELECT 1 FROM session_admissions WHERE target_session_id = ANY(%s) "
                    "AND status IN ('queued','started','unknown') LIMIT 1",
                    (sorted(targets),)).fetchone():
                raise AuthorityRefusal('stale_generation')
            if _worker_states(conn, session_id):
                raise AuthorityRefusal('stale_generation')
            if conn.execute(
                    "SELECT 1 FROM session_admissions WHERE target_session_id = ANY(%s) AND status = 'unknown' LIMIT 1",
                    (sorted(targets),)).fetchone():
                raise AuthorityRefusal('unknown_execution')
            conn.execute(
                """INSERT INTO worker_executions (execution_id, session_id, kind, owner_epoch, generation, status, adoption_digest)
                VALUES (%s, %s, %s, %s, %s, 'registered', %s)""",
                (execution_id, session_id, kind_value, epoch, generation, digest))
            return _worker_public(_worker_assignment(conn, execution_id, session_id, generation))

        with _refused(self._store, session_id):
            with self._store.tx() as conn:
                row = write(conn)
        return _record(row)

    def adopt(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId,
              generation: ExecutionGeneration, adoption_secret: str) -> WorkerExecutionRecord:
        """Authority first verifies the original producer claim and live worker proof."""
        digest = _secret_digest(adoption_secret)

        def write(conn):
            _epoch_check(conn, epoch)
            row = _worker_assignment(conn, execution_id, session_id, generation)
            if row["status"] == 'terminal':
                raise AuthorityRefusal('stale_generation')
            if not hmac.compare_digest(row["adoption_digest"], digest):
                raise AuthorityRefusal('permission_denied')
            # An admission-reserved worker reports to the reserving owner over that owner's pipes and
            # dies with them, so only its same-epoch handshake adopts it.
            if str(execution_id).startswith('admission-worker:') and row["owner_epoch"] != epoch:
                raise AuthorityRefusal('stale_epoch')
            linked = _linked_worker_admission(
                conn, session_id, generation, row["owner_epoch"], ('started', 'unknown'))
            if linked is not None:
                changed = conn.execute(
                    "UPDATE session_admissions SET owner_epoch = %s, status = 'started' "
                    "WHERE admission_id = %s AND generation = %s AND owner_epoch = %s AND status IN ('started','unknown')",
                    (epoch, linked["admission_id"], generation, row["owner_epoch"]))
                if changed.rowcount != 1:
                    raise AuthorityRefusal('stale_generation')
            changed = conn.execute(
                "UPDATE worker_executions SET owner_epoch = %s, status = 'running' WHERE execution_id = %s",
                (epoch, execution_id))
            if changed.rowcount != 1:
                raise AuthorityRefusal('stale_generation')
            return _worker_public(_worker_assignment(conn, execution_id, session_id, generation))

        with _refused(self._store, session_id):
            with self._store.tx() as conn:
                row = write(conn)
        return _record(row)

    def reserve_for_admission(self, *, epoch: AuthorityEpoch, admission_id: str, principal_id: str,
                             profile_id: str, pid: int, birth: float, secret: str,
                             owner: OwnerRule | None,
                             worker_subject: str | None = None) -> AdmissionWorkerScope:
        """Return a private bootstrap scope; never accept caller-selected kind/target. The admission
        supplies the physical session and generation; its durable principal must match the producer
        under the owner rule *owner*."""
        def write(conn):
            _epoch_check(conn, epoch)
            admission = conn.execute("SELECT * FROM session_admissions WHERE admission_id = %s",
                                    (str(admission_id),)).fetchone()
            if admission is None:
                raise AuthorityRefusal('not_found')
            if admission["status"] != 'started' or admission["owner_epoch"] != epoch:
                raise AuthorityRefusal('producer_not_started')
            if not same_owner(admission["principal_id"], principal_id, owner):
                raise AuthorityRefusal('permission_denied')
            target_session, generation = admission["target_session_id"], admission["generation"]
            if _session_row(conn, target_session)["runtime_generation"] != generation:
                raise AuthorityRefusal('stale_generation')
            # The private worker fence follows its physical assignment; admission/FIFO generation and
            # identity remain on the logical owner.
            sid = _local_physical_target(conn, target_session)
            execution_id = 'admission-worker:' + str(admission_id)
            if conn.execute("SELECT 1 FROM worker_executions WHERE execution_id = %s",
                            (execution_id,)).fetchone():
                raise AuthorityRefusal('admission_conflict')
            if conn.execute("SELECT 1 FROM worker_executions WHERE session_id = %s AND status <> 'terminal'",
                            (sid,)).fetchone():
                raise AuthorityRefusal('stale_generation')
            conn.execute("UPDATE sessions SET runtime_generation = %s WHERE id = %s", (generation, sid))
            if sid != target_session:
                conn.execute(
                    "UPDATE session_admissions SET lineage_json = (lineage_json::jsonb || to_jsonb(%s::text))::text "
                    "WHERE admission_id = %s", (sid, str(admission_id)))
            scope = dict(profile_id=profile_id, session_id=sid, execution_id=execution_id,
                         generation=generation, pid=pid, birth=birth, secret=secret)
            claim = admission_fingerprint(canonical_target=sid,
                                         payload=scope | {'principal': worker_subject or principal_id})
            conn.execute(
                """INSERT INTO worker_executions (execution_id, session_id, kind, owner_epoch, generation, status, adoption_digest)
                VALUES (%s, %s, 'compute', %s, %s, 'registered', %s)""",
                (execution_id, sid, epoch, generation, _secret_digest(claim)))
            return scope | {'epoch': epoch}

        with _refused(self._store):
            with self._store.tx() as conn:
                scope = write(conn)
        return AdmissionWorkerScope(
            profile_id=scope['profile_id'], session_id=SessionId(scope['session_id']),
            execution_id=WorkerExecutionId(scope['execution_id']),
            generation=ExecutionGeneration(scope['generation']), pid=scope['pid'], birth=scope['birth'],
            secret=scope['secret'], epoch=AuthorityEpoch(scope['epoch']))

    def lose_reservation(self, *, epoch: AuthorityEpoch, admission_id: str,
                         execution_id: WorkerExecutionId, generation: ExecutionGeneration) -> None:
        """Revoke a private assignment and pause its FIFO; no inference replay. This also fences a
        result already in the pipe at loss/shutdown."""
        def write(conn):
            _epoch_check(conn, epoch)
            current = conn.execute("SELECT * FROM session_admissions WHERE admission_id = %s",
                                  (str(admission_id),)).fetchone()
            if current is None:
                raise AuthorityRefusal('not_found')
            if (current["status"] != 'started' or current["owner_epoch"] != epoch
                    or current["generation"] != generation):
                raise AuthorityRefusal('stale_generation')
            worker = conn.execute("SELECT * FROM worker_executions WHERE execution_id = %s",
                                 (execution_id,)).fetchone()
            if worker is None or worker["generation"] != generation or worker["owner_epoch"] != epoch:
                raise AuthorityRefusal('stale_generation')
            conn.execute("UPDATE worker_executions SET status = 'terminal' WHERE execution_id = %s",
                         (execution_id,))
            _compact_terminal_receipts(conn, execution_id)
            conn.execute("UPDATE session_admissions SET status = 'unknown' WHERE admission_id = %s",
                         (str(admission_id),))
            conn.execute("UPDATE sessions SET runtime_revision = runtime_revision + 1 WHERE id = %s",
                         (current["target_session_id"],))

        with _refused(self._store):
            with self._store.tx() as conn:
                write(conn)

    def reserve_child(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId,
                      generation: ExecutionGeneration, sequence: int, secret: str | WorkerDelegationPayload,
                      profile_id: str, pid: int, birth: float, principal: str,
                      max_concurrent_children: int, max_spawn_depth: int,
                      refused: str | None = None) -> ChildReservation:
        payload = secret.payload if isinstance(secret, WorkerDelegationPayload) else {'secret': secret}
        handler = partial(_reserve_child_execution, epoch=epoch, execution_id=execution_id,
                          identity={'profile_id': profile_id, 'pid': pid, 'birth': birth},
                          principal=principal, limits=(max_concurrent_children, max_spawn_depth),
                          refused=refused)
        with _refused(self._store, session_id):
            result = _apply_worker_receipt(self._store, epoch=epoch, execution_id=execution_id,
                                          session_id=session_id, generation=generation, sequence=sequence,
                                          operation='child.reserve', payload=payload,
                                          handlers={'child.reserve': handler})
        child = result.get('child')
        if child is None:
            return ChildReservation(refused=result['refused'])
        return ChildReservation(child=ChildExecution(
            session_id=SessionId(child['session_id']), execution_id=WorkerExecutionId(child['execution_id']),
            generation=ExecutionGeneration(child['generation'])))

    def persist_operation(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId,
                          generation: ExecutionGeneration, sequence: int, operation: str,
                          payload: Mapping[str, JsonValue]) -> Mapping[str, JsonValue]:
        try:
            return self._persist(operation, payload, epoch=epoch, execution_id=execution_id,  # type: ignore[no-any-return]
                                 session_id=session_id, generation=generation, sequence=sequence)
        except RuntimeError as exc:
            # The publication body's precondition refusals are plain RuntimeErrors; its typed refusals subclass it.
            if operation != 'compression.publish' or type(exc) is not RuntimeError:
                raise
            raise CompressionPublishRefused(session_id, str(exc)) from exc

    def finish(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId,
               generation: ExecutionGeneration, sequence: int) -> None:
        self._persist('execution.finish', {}, epoch=epoch, execution_id=execution_id, session_id=session_id,
                      generation=generation, sequence=sequence)

    def record_usage(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId,
                     generation: ExecutionGeneration, sequence: int,
                     **usage: Any) -> None:
        _key_set(usage, _MAIN_USAGE_KEYS)
        self._value('usage.main', dict(usage), epoch, execution_id, session_id, generation, sequence)

    def record_auxiliary_usage(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId,
                               session_id: SessionId, generation: ExecutionGeneration, sequence: int, task: str,
                               **usage: Any) -> None:
        _key_set(usage, _AUXILIARY_USAGE_KEYS)
        self._value('usage.auxiliary', dict(usage, task=task), epoch, execution_id, session_id, generation,
                    sequence)

    def acquire_turn_lease(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId,
                           generation: ExecutionGeneration, sequence: int, lease: WorkerTurnLease) -> bool:
        payload = {'holder': lease.holder, 'ttl_seconds': lease.ttl_seconds}
        return bool(self._value('turn.acquire', payload, epoch, execution_id, session_id, generation, sequence))

    def renew_turn_lease(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId,
                         generation: ExecutionGeneration, sequence: int, lease: WorkerTurnLease) -> bool:
        payload = {'holder': lease.holder, 'ttl_seconds': lease.ttl_seconds}
        return bool(self._value('turn.renew', payload, epoch, execution_id, session_id, generation, sequence))

    def cleanup_turn_lease(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId,
                           generation: ExecutionGeneration, sequence: int, target_session_id: SessionId,
                           holder: str) -> None:
        self._persist('turn.cleanup', {'target': target_session_id, 'holder': holder}, epoch=epoch,
                      execution_id=execution_id, session_id=session_id, generation=generation, sequence=sequence)

    def verify_claim(self, *, execution_id: WorkerExecutionId, session_id: SessionId,
                     generation: ExecutionGeneration, adoption_secret: str) -> WorkerExecutionRecord:
        """The claim check of a request already addressed to *session_id* (the caller resolves a
        publication replay's target first)."""
        with _refused(self._store, session_id):
            row = self._store.one("SELECT * FROM worker_executions WHERE execution_id = %s", str(execution_id))
            if row is None:
                raise AuthorityRefusal('not_found')
            if row["session_id"] != session_id:
                raise AuthorityRefusal('permission_denied')
            if type(generation) is not int or row["generation"] != generation:
                raise AuthorityRefusal('stale_generation')
            if not hmac.compare_digest(row["adoption_digest"], _secret_digest(adoption_secret)):
                raise AuthorityRefusal('permission_denied')
        return _record(_worker_public(row))

    def execution_session(self, *, execution_id: WorkerExecutionId) -> SessionId | None:
        """The session this worker execution last wrote to, as the owner's ledger records it."""
        if not execution_id:
            return None
        with _refused(self._store):
            row = self._store.one("SELECT session_id FROM worker_executions WHERE execution_id = %s",
                                  str(execution_id))
        return None if row is None else SessionId(row["session_id"])

    def has_live_worker(self) -> bool:
        """Whether any worker execution is still live."""
        with _refused(self._store):
            return self._store.scalar("SELECT 1 FROM worker_executions WHERE status <> 'terminal' LIMIT 1") is not None

    # ── WorkerSessionVerbs ──────────────────────────────────────────────────

    def set_session_title(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId, generation: ExecutionGeneration, sequence: int, title: str | None, source: TitleSource) -> WorkerTitleResult:
        result = self._persist('session.title', {'title': title, 'source': getattr(source, 'value', source)}, epoch=epoch,
                               execution_id=execution_id, session_id=session_id, generation=generation, sequence=sequence)
        return WorkerTitleResult(value=bool(result['value']), error=result.get('error'),
                                 title_candidate=result.get('title_candidate'))

    def set_session_title_source(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId, generation: ExecutionGeneration, sequence: int, source: TitleSource) -> bool:
        return bool(self._value('session.title_source', {'source': getattr(source, 'value', source)}, epoch,
                                execution_id, session_id, generation, sequence))

    def next_session_title(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId, generation: ExecutionGeneration, sequence: int, base_title: str) -> str:
        return str(self._value('session.next_title', {'base_title': base_title}, epoch, execution_id, session_id,
                               generation, sequence))

    def touch_session_activity(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId, generation: ExecutionGeneration, sequence: int, activity: WorkerActivity) -> None:
        payload = {'ts': activity.ts, 'description': activity.description, 'provenance': activity.provenance}
        self._value('session.activity', payload, epoch, execution_id, session_id, generation, sequence)

    def clear_session_activity(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId, generation: ExecutionGeneration, sequence: int) -> None:
        self._value('session.activity_clear', {}, epoch, execution_id, session_id, generation, sequence)

    def set_session_billing_route(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId, generation: ExecutionGeneration, sequence: int, route: WorkerBillingRoute) -> None:
        payload = {'provider': route.provider, 'base_url': route.base_url, 'billing_mode': route.billing_mode}
        self._value('session.billing_route', payload, epoch, execution_id, session_id, generation, sequence)

    def set_latest_user_api_content(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId, generation: ExecutionGeneration, sequence: int, content: JsonValue, api_content: str) -> int:
        return int(self._value('session.api_content', {'content': content, 'api_content': api_content}, epoch,
                               execution_id, session_id, generation, sequence))

    def set_message_api_content(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId, generation: ExecutionGeneration, sequence: int, row_id: MessageId, content: JsonValue, api_content: str) -> int:
        payload = {'row_id': row_id, 'content': content, 'api_content': api_content}
        return int(self._value('session.message_api_content', payload, epoch, execution_id, session_id, generation,
                               sequence))

    def set_user_message_content(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId, generation: ExecutionGeneration, sequence: int, row_id: MessageId, content: JsonValue, display_metadata: Optional[Mapping[str, JsonValue]] = None) -> int:
        payload: dict[str, Any] = {'row_id': row_id, 'content': content}
        if display_metadata is not None:
            payload['display_metadata'] = dict(display_metadata)
        return int(self._value('session.user_content', payload, epoch, execution_id, session_id, generation,
                               sequence))

    def create_session(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId, generation: ExecutionGeneration, sequence: int, **fields: Unpack[WorkerSessionCreate]) -> SessionId:
        _key_set(fields, _CREATE_KEYS, frozenset({'source'}))
        return SessionId(self._value('session.create', dict(fields), epoch, execution_id, session_id, generation,
                                     sequence))

    def end_session(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId, generation: ExecutionGeneration, sequence: int, end_reason: str) -> None:
        self._value('session.end', {'end_reason': end_reason}, epoch, execution_id, session_id, generation, sequence)

    def session_lifecycle(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId, generation: ExecutionGeneration, sequence: int) -> SessionLifecycleStatus:
        return SessionLifecycleStatus(self._value('session.lifecycle', {}, epoch, execution_id, session_id, generation,
                                                  sequence))

    def set_session_system_prompt(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId, generation: ExecutionGeneration, sequence: int, system_prompt: str | None) -> None:
        self._value('session.prompt', {'system_prompt': system_prompt}, epoch, execution_id, session_id, generation,
                    sequence)

    def patch_session_sidecars(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId, generation: ExecutionGeneration, sequence: int, patch: Mapping[str, JsonValue]) -> None:
        payload = {'patch': dict(patch) if isinstance(patch, Mapping) else patch}
        self._value('session.sidecars', payload, epoch, execution_id, session_id, generation, sequence)

    def set_session_tool_pin(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId, generation: ExecutionGeneration, sequence: int, tool_pin: ToolPin | None) -> None:
        payload = {'tool_names': dict(tool_pin) if isinstance(tool_pin, Mapping) else tool_pin}
        self._value('session.tools', payload, epoch, execution_id, session_id, generation, sequence)

    def latest_conversation_role(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId, generation: ExecutionGeneration, sequence: int) -> str | None:
        return self._value('session.tail_role', {}, epoch, execution_id, session_id, generation, sequence)  # type: ignore[no-any-return]

    def message_role(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId, generation: ExecutionGeneration, sequence: int, row_id: int) -> str | None:
        return self._value('session.row_role', {'row_id': row_id}, epoch, execution_id, session_id, generation, sequence)  # type: ignore[no-any-return]

    def append_transcript(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId, generation: ExecutionGeneration, sequence: int, messages: Sequence[WorkerMessage], **options: Unpack[TranscriptAppendOptions]) -> WorkerAppendResult:
        _key_set(options, _APPEND_OPTION_KEYS)
        result = self._persist('transcript.append', {'messages': _messages(messages), **options}, epoch=epoch,
                               execution_id=execution_id, session_id=session_id, generation=generation, sequence=sequence)
        return WorkerAppendResult(count=result['count'], annotations=_annotations(result))

    # ── WorkerCompressionRepository ─────────────────────────────────────────

    def acquire_compression_lock(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId, generation: ExecutionGeneration, sequence: int, lease: WorkerTurnLease) -> bool:
        payload = {'holder': lease.holder, 'ttl_seconds': lease.ttl_seconds}
        return bool(self._value('compression.lock.acquire', payload, epoch, execution_id, session_id, generation, sequence))

    def renew_compression_lock(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId, generation: ExecutionGeneration, sequence: int, lease: WorkerTurnLease) -> bool:
        payload = {'holder': lease.holder, 'ttl_seconds': lease.ttl_seconds}
        return bool(self._value('compression.lock.renew', payload, epoch, execution_id, session_id, generation, sequence))

    def compression_lock_holder(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId, generation: ExecutionGeneration, sequence: int) -> str | None:
        return self._value('compression.lock.holder', {}, epoch, execution_id, session_id, generation, sequence)  # type: ignore[no-any-return]

    def cleanup_compression_lock(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId, generation: ExecutionGeneration, sequence: int, target_session_id: SessionId, holder: str) -> None:
        self._value('compression.cleanup', {'target': target_session_id, 'holder': holder}, epoch, execution_id,
                    session_id, generation, sequence)

    def record_compression_cooldown(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId, generation: ExecutionGeneration, sequence: int, cooldown: WorkerCooldown) -> None:
        payload = {'cooldown_until': cooldown.cooldown_until, 'error': cooldown.error}
        self._value('compression.cooldown.record', payload, epoch, execution_id, session_id, generation, sequence)

    def restore_compression_cooldown(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId, generation: ExecutionGeneration, sequence: int, snapshot: WorkerCooldownSnapshot) -> None:
        payload = {'snapshot': {'session_exists': snapshot.session_exists, 'cooldown_until': snapshot.cooldown_until,
                                'error': snapshot.error}}
        self._value('compression.cooldown.restore', payload, epoch, execution_id, session_id, generation, sequence)

    def clear_compression_cooldown(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId, generation: ExecutionGeneration, sequence: int) -> None:
        self._value('compression.cooldown.clear', {}, epoch, execution_id, session_id, generation, sequence)

    def set_compression_fallback_streak(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId, generation: ExecutionGeneration, sequence: int, streak: int) -> None:
        self._value('compression.fallback', {'value': streak}, epoch, execution_id, session_id, generation, sequence)

    def set_compression_ineffective_count(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId, generation: ExecutionGeneration, sequence: int, count: int) -> None:
        self._value('compression.ineffective', {'value': count}, epoch, execution_id, session_id, generation, sequence)

    def set_compression_recovery_deadline(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId, generation: ExecutionGeneration, sequence: int, deadline: float | None) -> None:
        self._value('compression.recovery', {'value': deadline}, epoch, execution_id, session_id, generation, sequence)

    def compression_watermark(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId, generation: ExecutionGeneration, sequence: int) -> int:
        return int(self._value('compression.watermark', {}, epoch, execution_id, session_id, generation, sequence))

    def reopen_orphaned_compression(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId, generation: ExecutionGeneration, sequence: int) -> bool:
        return bool(self._value('compression.reopen', {}, epoch, execution_id, session_id, generation, sequence))

    def compression_boundary(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId, generation: ExecutionGeneration, sequence: int, peer: ConversationPeer) -> int | None:
        payload = {'session_key': peer.session_key, 'source': peer.source}
        return self._value('compression.boundary', payload, epoch, execution_id, session_id, generation, sequence)  # type: ignore[no-any-return]

    # ── WorkerCompressionReads ──────────────────────────────────────────────

    def compression_session_context(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId, generation: ExecutionGeneration, sequence: int, target_session_id: SessionId) -> Mapping[str, JsonValue]:
        return self._persist('compression.context', {'target': target_session_id}, epoch=epoch, execution_id=execution_id,  # type: ignore[no-any-return]
                             session_id=session_id, generation=generation, sequence=sequence)['session']

    def compression_lineage_view(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId, generation: ExecutionGeneration, sequence: int, target_session_id: SessionId) -> WorkerLineageView:
        view = self._persist('compression.lineage', {'target': target_session_id}, epoch=epoch, execution_id=execution_id,
                             session_id=session_id, generation=generation, sequence=sequence)
        return WorkerLineageView(
            readable_ids=tuple(SessionId(s) for s in view['readable_ids']), lineage=tuple(SessionId(s) for s in view['lineage']),
            root=SessionId(view['root']), tip=SessionId(view['tip']), resume=SessionId(view['resume']),
            identity=(bool(view['identity'][0]), str(view['identity'][1])), tool_root=SessionId(view['tool_root']))

    def compression_history(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId, generation: ExecutionGeneration, sequence: int, target_session_id: SessionId, include_ancestors: bool, include_inactive: bool, repair_alternation: bool, include_row_ids: bool, include_compacted: bool) -> tuple[Mapping[str, JsonValue], ...]:
        payload = {'target': target_session_id, 'include_ancestors': include_ancestors, 'include_inactive': include_inactive,
                   'repair_alternation': repair_alternation, 'include_row_ids': include_row_ids,
                   'include_compacted': include_compacted}
        return tuple(self._persist('compression.history', payload, epoch=epoch, execution_id=execution_id,
                                   session_id=session_id, generation=generation, sequence=sequence)['messages'])

    # ── WorkerCompactionVerbs ───────────────────────────────────────────────

    def compact_transcript(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId, generation: ExecutionGeneration, sequence: int, compaction: WorkerCompaction) -> WorkerArchiveResult:
        result = self._persist('compression.archive', compaction.payload(), epoch=epoch, execution_id=execution_id,
                               session_id=session_id, generation=generation, sequence=sequence)
        return WorkerArchiveResult(value=int(result['value']), annotations=_annotations(result))

    def publish_compression_child(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId, generation: ExecutionGeneration, sequence: int, publication: WorkerCompressionPublication) -> WorkerPublishResult:
        result = self.persist_operation(epoch=epoch, execution_id=execution_id, session_id=session_id,
                                        generation=generation, sequence=sequence, operation='compression.publish',
                                        payload=publication.payload())
        moved = result['worker_assignment']
        return WorkerPublishResult(parent=SessionId(moved['parent']), session_id=SessionId(moved['session_id']),
                                   annotations=_annotations(result))

    def append_compression_transcript(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId, generation: ExecutionGeneration, sequence: int, messages: Sequence[WorkerMessage], compression_lock_holder: str | None, turn_lease_holder: str | None, turn_lease_ttl_seconds: float) -> WorkerAppendResult:
        payload = {'messages': _messages(messages), 'compression_lock_holder': compression_lock_holder,
                   'turn_lease_holder': turn_lease_holder, 'turn_lease_ttl_seconds': turn_lease_ttl_seconds}
        result = self._persist('compression.append', payload, epoch=epoch, execution_id=execution_id,
                               session_id=session_id, generation=generation, sequence=sequence)
        return WorkerAppendResult(count=result['count'], annotations=_annotations(result))

    # ── WorkerReceiptReads ──────────────────────────────────────────────────

    def claim_origin(self, *, execution_id: WorkerExecutionId, session_id: SessionId) -> SessionId:
        with _refused(self._store):
            row = self._store.one(
                "SELECT record_json FROM worker_receipts WHERE execution_id = %s "
                "AND record_json::jsonb -> 'worker_assignment' ->> 'parent' IS NOT NULL ORDER BY sequence LIMIT 1",
                str(execution_id))
        return SessionId(json.loads(row["record_json"])['worker_assignment']['parent']) if row else SessionId(session_id)

    def publication_target(self, *, execution_id: WorkerExecutionId, session_id: SessionId, generation: ExecutionGeneration, sequence: int, operation: str, payload: Mapping[str, JsonValue]) -> SessionId:
        if operation != 'compression.publish':
            return SessionId(session_id)
        digest = admission_fingerprint(canonical_target=str(session_id),
                                       payload={'operation': operation, 'payload': dict(payload)})
        with _refused(self._store):
            with self._store.tx() as conn:
                row = _worker_receipt_assignment(conn, str(execution_id), str(session_id), generation, sequence, digest)
        return SessionId(row['session_id'])

    # ── WorkerDelegationLedger ───────────────────────────────────────────────

    def _ledger(self, operation: str, payload: Any, epoch: Any, execution_id: Any, session_id: Any, generation: Any,
                sequence: Any, worker_pid: Any, worker_birth: Any, owner_started_at: int | None = None,
                probe_refused: str | None = None) -> Any:
        with _refused(self._store, session_id):
            return _apply_worker_receipt(self._store, epoch=epoch, execution_id=execution_id, session_id=session_id,
                                        generation=generation, sequence=sequence, operation=operation, payload=payload,
                                        handlers=_delegation_handlers(execution_id, worker_pid, worker_birth,
                                                                     owner_started_at, probe_refused))['value']

    def dispatch_delegation(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId, generation: ExecutionGeneration, sequence: int, dispatch: WorkerDelegationDispatch | WorkerDelegationPayload, worker_pid: int, worker_birth: float, owner_started_at: int | None, probe_refused: Literal['permission_denied', 'worker_not_live'] | None = None) -> None:
        self._ledger('delegation.dispatch', delegation_payload(dispatch), epoch, execution_id, session_id, generation, sequence,
                     worker_pid, worker_birth, owner_started_at=owner_started_at, probe_refused=probe_refused)

    def complete_delegation(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId, generation: ExecutionGeneration, sequence: int, completion: WorkerDelegationCompletion | WorkerDelegationPayload, worker_pid: int, worker_birth: float) -> None:
        self._ledger('delegation.complete', delegation_payload(completion), epoch,
                     execution_id, session_id, generation, sequence, worker_pid, worker_birth)

    def record_delegation_child(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId, generation: ExecutionGeneration, sequence: int, child: WorkerDelegationChild | WorkerDelegationPayload, worker_pid: int, worker_birth: float) -> None:
        self._ledger('delegation.child', delegation_payload(child), epoch,
                     execution_id, session_id, generation, sequence, worker_pid, worker_birth)

    def read_delegation(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId, generation: ExecutionGeneration, sequence: int, delegation_id: str | WorkerDelegationPayload, worker_pid: int, worker_birth: float) -> WorkerDelegationView:
        return WorkerDelegationView(**self._ledger('delegation.read', delegation_payload(delegation_id), epoch,
                                                  execution_id, session_id, generation, sequence, worker_pid,
                                                  worker_birth))

    def _delivery(self, action: str, claim: WorkerDelegationClaim | WorkerDelegationPayload, *identity: Any) -> bool:
        return bool(self._ledger('delegation.' + action, delegation_payload(claim), *identity))

    def claim_delegation_delivery(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId, generation: ExecutionGeneration, sequence: int, claim: WorkerDelegationClaim | WorkerDelegationPayload, worker_pid: int, worker_birth: float) -> bool:
        return self._delivery('claim', claim, epoch, execution_id, session_id, generation, sequence, worker_pid, worker_birth)

    def ack_delegation_delivery(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId, generation: ExecutionGeneration, sequence: int, claim: WorkerDelegationClaim | WorkerDelegationPayload, worker_pid: int, worker_birth: float) -> bool:
        return self._delivery('ack', claim, epoch, execution_id, session_id, generation, sequence, worker_pid, worker_birth)

    def drop_delegation_delivery(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId, generation: ExecutionGeneration, sequence: int, claim: WorkerDelegationClaim | WorkerDelegationPayload, worker_pid: int, worker_birth: float) -> bool:
        return self._delivery('drop', claim, epoch, execution_id, session_id, generation, sequence, worker_pid, worker_birth)

    def defer_delegation_delivery(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId, generation: ExecutionGeneration, sequence: int, claim: WorkerDelegationClaim | WorkerDelegationPayload, worker_pid: int, worker_birth: float) -> bool:
        return self._delivery('defer', claim, epoch, execution_id, session_id, generation, sequence, worker_pid, worker_birth)

    def release_delegation_delivery(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId, generation: ExecutionGeneration, sequence: int, claim: WorkerDelegationClaim | WorkerDelegationPayload, worker_pid: int, worker_birth: float) -> bool:
        return self._delivery('release', claim, epoch, execution_id, session_id, generation, sequence, worker_pid, worker_birth)

    def withdraw_delegation(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId, generation: ExecutionGeneration, sequence: int, delegation_id: str | WorkerDelegationPayload, worker_pid: int, worker_birth: float) -> None:
        self._ledger('delegation.unscheduled', delegation_payload(delegation_id), epoch, execution_id, session_id,
                     generation, sequence, worker_pid, worker_birth)

    def pending_delegation_event(self, *, execution_id: WorkerExecutionId, session_id: SessionId, delegation_id: str) -> Mapping[str, JsonValue] | None:
        # The reference's ``_owned`` read: the id text check first, then ownership of the row.
        if not isinstance(delegation_id, str) or not delegation_id or len(delegation_id) > 512:
            raise AuthorityRefusal('invalid_params')
        with _refused(self._store):
            row = self._store.one("SELECT * FROM async_delegations WHERE delegation_id = %s", str(delegation_id))
        if row is None:
            raise AuthorityRefusal('not_found')
        if row['parent_session_id'] != session_id or row['owner_execution_id'] != execution_id:
            raise AuthorityRefusal('permission_denied')
        pending = row['delivery_state'] == 'pending' and row['event_json'] is not None
        return json.loads(row['event_json']) if pending else None


add_stubs(PgWorkerExecutionsRepository, WorkerExecutionRepository, implemented=(
    'register', 'adopt', 'reserve_for_admission', 'lose_reservation', 'reserve_child',
    'persist_operation', 'finish', 'record_usage', 'record_auxiliary_usage',
    'acquire_turn_lease', 'renew_turn_lease', 'cleanup_turn_lease', 'verify_claim',
    'execution_session', 'has_live_worker',
    'set_session_title', 'set_session_title_source', 'next_session_title', 'touch_session_activity',
    'clear_session_activity', 'set_session_billing_route', 'set_latest_user_api_content',
    'set_message_api_content', 'set_user_message_content', 'create_session', 'end_session',
    'session_lifecycle', 'set_session_system_prompt', 'patch_session_sidecars', 'set_session_tool_pin',
    'latest_conversation_role', 'message_role', 'append_transcript',
    'acquire_compression_lock', 'renew_compression_lock', 'compression_lock_holder',
    'cleanup_compression_lock', 'record_compression_cooldown', 'restore_compression_cooldown',
    'clear_compression_cooldown', 'set_compression_fallback_streak', 'set_compression_ineffective_count',
    'set_compression_recovery_deadline', 'compression_watermark', 'reopen_orphaned_compression',
    'compression_boundary', 'compression_session_context', 'compression_lineage_view',
    'compression_history', 'compact_transcript', 'publish_compression_child',
    'append_compression_transcript', 'claim_origin', 'publication_target',
    'dispatch_delegation', 'complete_delegation', 'record_delegation_child', 'read_delegation',
    'claim_delegation_delivery', 'ack_delegation_delivery', 'drop_delegation_delivery',
    'defer_delegation_delivery', 'release_delegation_delivery', 'withdraw_delegation',
    'pending_delegation_event',
))


def _register() -> None:
    # ``receipted_writes``: the stage-2 verbs land their writes through the one receipt transaction.
    # ``transactional_claims``: the receipt/turn-lease claim machinery (and the ported
    # coordination verbs) shares the ``session_turn_leases`` table one transaction at a time; the
    # receipts contract module is gated on it.
    register_repository("worker_executions", lambda store: PgWorkerExecutionsRepository(store),
                        {"receipted_writes": True})


_register()

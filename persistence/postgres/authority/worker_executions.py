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
from typing import Any, Iterator, Mapping, Optional, Sequence

from persistence.authority.vocabulary import OwnerRule, admission_fingerprint
from persistence.core.canonical import canonical_json
from persistence.core.errors import AuthorityRefusal, CompressionPublishRefused
from persistence.core.fences import AuthorityEpoch, ExecutionGeneration
from persistence.core.identity import JsonValue, SessionId
from persistence.sqlite._schema import _NUMBERED_TITLE_RE
from persistence.sqlite.conversation._store.compression import _CHAIN_CAP
from persistence.sqlite.conversation._store.local import POLICY_PREFIX
from persistence.sqlite.conversation._store.sessions import classify_session_status
from persistence.sqlite.workers._runtime_workers import RETIRED_RECEIPT, descendant_range
from persistence.workers.delegations import WorkerDelegationPayload
from persistence.workers.executions import (
    AdmissionWorkerScope, ChildExecution, ChildReservation, WorkerAuxiliaryUsage, WorkerExecutionId,
    WorkerExecutionRecord, WorkerExecutionRepository, WorkerKind, WorkerMainUsage, WorkerStatus,
    WorkerTurnLease,
)
from persistence.workers.vocabulary import (
    CHILD_SEPARATOR, SIDECAR_KEYS, WORKER_MESSAGE_FIELDS, row_annotations,
)
from process_identity import holder_process_is_dead
from session_rules.activity import ActivityProvenance, bound_activity_description, normalize_activity_provenance
from session_rules.legacy_local import same_owner
from session_rules.session_ids import new_session_id
from session_rules.surfaces import created_surface

from .._backend import PostgresStore, register_repository
from .._stubs import add_stubs
from ..conversation._store.common import _check_transcript_write_guards, _scrub_surrogates
from ..conversation._store.repair import resolve_and_repair_transcript_batch
from ..conversation._store.sessions_pg import _merge_model_config_json
from ..conversation._store.writes import (
    _bump_counters, _decode_content, _decode_display_metadata, _encode_content,
    _encode_display_metadata, _insert_message_rows, _payload_bytes_param, _serialize,
    _decoded_repair_row,
)
from ..conversation.sessions import (
    _CHAIN_STEP_SQL, _delete_unreferenced_system_prompts, _end_and_bump,
    _session_turn_lease_key_on_conn, _set_session_title_in_transaction, _store_system_prompt,
    _walk_compression_chain,
)
from ..conversation.usage import _MODEL_USAGE_FIELDS, _TOKEN_COUNTERS
from .admissions import (
    _compact_terminal_receipts, _epoch_check, _expand_compression_lineage, _local_physical_target,
    _session_row,
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
    # writer set, the constructor fills only the unset.
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
         payload.get('parent_session_id'), payload.get('cwd'), payload.get('profile_name'),
         payload.get('git_repo_root'), payload.get('origin_json'), payload.get('display_name'),
         session_id))
    if config:
        # An owner-reserved child row already holds its lineage marker, so the keep-existing upsert
        # would drop these facts; merge them beside it (keys are the constructor set only).
        conn.execute(
            "UPDATE sessions SET model_config = COALESCE(model_config::jsonb, '{}'::jsonb) || %s::jsonb "
            "WHERE id = %s AND json_text_valid(model_config) "
            "AND (model_config::jsonb) #> ARRAY['_delegate_from'] IS NOT NULL",
            (json.dumps(config), session_id))
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
    rows = conn.execute("SELECT title FROM sessions WHERE title = %s OR title LIKE %s ESCAPE '\\'",
                        (base, _escape_like(base))).fetchall()
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
        "AND role = 'user' AND active = 1 ORDER BY id DESC LIMIT 1) AND content IS %s",
        (_payload_bytes_param(_scrub_surrogates(payload['api_content'])), session_id,
         _payload_bytes_param(_encode_content(payload['content'])))).rowcount
    return {'value': value}


def _worker_message_api_content(store, conn, session_id: str, payload: Any) -> dict:
    if set(payload) != {'row_id', 'content', 'api_content'} or not isinstance(payload['api_content'], str):
        raise AuthorityRefusal('invalid_params')
    return {'value': conn.execute(
        "UPDATE messages SET api_content = %s WHERE id = %s AND session_id = %s AND role = 'user' "
        "AND active = 1 AND content IS %s",
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
    """The closed table of receipted worker operations (this stage's D8 scope: the core execution
    verbs' operations. The compression/delegation tables of the sibling protocols stay stubs; an
    operation outside the table is refused ``invalid_params`` before the store)."""
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
            if current is None or (current["status"] != 'started' or current["owner_epoch"] != epoch
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


add_stubs(PgWorkerExecutionsRepository, WorkerExecutionRepository, implemented=(
    'register', 'adopt', 'reserve_for_admission', 'lose_reservation', 'reserve_child',
    'persist_operation', 'finish', 'record_usage', 'record_auxiliary_usage',
    'acquire_turn_lease', 'renew_turn_lease', 'cleanup_turn_lease', 'verify_claim',
    'execution_session', 'has_live_worker',
))


def _register() -> None:
    # ``receipted_writes``: the stage-2 verbs land their writes through the one receipt transaction.
    # ``transactional_claims``: the receipt/turn-lease claim machinery (and the ported
    # coordination verbs) shares the ``session_turn_leases`` table one transaction at a time; the
    # receipts contract module is gated on it.
    register_repository("worker_executions", lambda store: PgWorkerExecutionsRepository(store),
                        {"receipted_writes": True})


_register()

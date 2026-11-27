"""``AdmissionRepository`` on PostgreSQL (stage 2): the bound profile's session admissions ledger in
``session_admissions`` and its ``state_meta`` terminal tombstones, the SQLite admission store's SQL
ported inline onto the PG dialect.

Every verb is one transaction (``store.tx()`` for writes and for the reads that need one snapshot;
the store's read doors for single probes). The reference's shared steps (the epoch fence, the row
reads, the row projection, the steering transfer, the replacement binding, the worker retirement
and the tombstone reads) are this module's own functions, mirroring
``persistence.sqlite.authority._admission_store`` / ``_admission_owner`` / ``_admission_media`` /
``_runtime``; the pure helpers they share (``admission_fingerprint``, ``retry_input``, the stored
result codec, the folded/compacted payload shapes) are imported from the shared modules, so the two
backends cannot drift.

D8 residuals this module documents: the ``admit`` hosted member check runs on the PG hosted-room
state (room rows, driver tasks), which lands with the stage-4 hosted rooms port; until that module
provides ``_member_authorized_locked`` on the PG schema the check names its gap (``NotImplementedError``),
and the gate's hosted case is blocked on that same stage-4 stub (``create_room``)."""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from contextlib import contextmanager
from dataclasses import fields
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Iterator, Literal, Mapping, Sequence

import psycopg

from persistence.authority.admissions import (
    ANY_HEAD,
    ActiveReplace,
    AdmissionGate,
    AdmissionRecord,
    AdmissionRepository,
    AdmissionResult,
    ApiTargetClaim,
    AnyHead,
    CapturedResult,
    CarrierRewind,
    ClaimedAdmission,
    QueuedFold,
    QueuedReplacement,
    RetryRewriteRefused,
    TranscriptRewrite,
    TurnBoundary,
    UnheldMedia,
    raise_gate,
)
from persistence.authority.vocabulary import (
    ADMISSION_QUEUE_MAX,
    ADMITTED_REQUEST_CONFLICT,
    STEERING,
    admission_fingerprint,
    retry_input,
)
from persistence.core.canonical import canonical_json
from persistence.core.errors import (
    AuthorityRefusal,
    FenceStale,
    PermanentStoreError,
    ProofRejected,
    StaleFenceKind,
    UnknownExecution,
)
from persistence.sqlite.conversation._store.local import POLICY_PREFIX
from persistence.sqlite.conversation._store.media import PREFIX as MEDIA_RETIRED_PREFIX
from persistence.sqlite.conversation._store.media import SPILLOVER_PREFIX as SPILLOVER_RETIRED_PREFIX
from persistence.sqlite.conversation._store.retirement import RETIRED_PREFIX
from persistence.sqlite.conversation._store.terminal import (
    ADMISSION_PREFIX,
    RESULT_PREFIX,
    encode_result,
    identity_key,
    load_result,
)
from persistence.sqlite.workers._runtime_workers import RETIRED_RECEIPT
from persistence.workers.vocabulary import CHILD_SEPARATOR
from session_rules.api_payloads import compacted_api_payload
from session_rules.automation import folded_payload
from session_rules.media import hosted_document_references
from session_rules.results import compact_result
from session_rules.turns import FAILED_TURN_DISPLAY_KIND, PARTIAL_FAILED_TURN_NOTICE

from .._backend import PostgresStore, refusals, register_repository
from .._lineage_common import (
    chain_step_sql,
    json_marker_absent_sql,
    reset_child_sql,
)
from .._stubs import add_stubs
from ..conversation._store.common import (
    _check_transcript_write_guards,
    _encode_content,
    _encode_display_metadata,
    _payload_bytes,
)
from ..conversation._store.repair import resolve_and_repair_transcript_batch
from ..conversation._store.rewind import rewind_in_transaction
from ..conversation._store.writes import (
    _bump_counters,
    _decode_content,
    _decoded_repair_row,
    _insert_message_rows,
    _replace_in_transaction,
    _serialize,
)

if TYPE_CHECKING:
    from persistence.gateway.hosted_rooms import HostedMemberGuard

__all__ = ["PgAdmissionsRepository"]

_RECORD_FIELDS = tuple(f.name for f in fields(AdmissionRecord))

# ── the reference's shared transaction steps (the SQLite ``_runtime`` port) ──

_FENCE_KINDS = {kind.value: kind for kind in StaleFenceKind}

# The state_meta key prefix of a queued replacement's binding (the reference's ``REPLACEMENT_PREFIX``).
_REPLACEMENT_PREFIX = "gateway.queued_replacement.v1."

# The state_meta key prefix of an API session's stored turn settings.
_SETTINGS_PREFIX = "gateway.api.settings.v1."

_CANCEL_QUEUED = "UPDATE session_admissions SET status = 'terminal', outcome = 'cancelled' WHERE admission_id = %s"


def _typed(exc: AuthorityRefusal) -> AuthorityRefusal:
    """The contract subclass for a fence, unknown-execution or proof refusal, ``exc`` itself for any other code."""
    if exc.reason in _FENCE_KINDS:
        typed: AuthorityRefusal = FenceStale(_FENCE_KINDS[exc.reason], "authority", "")
    elif exc.reason == "unknown_execution":
        typed = UnknownExecution("authority", "")
    elif exc.reason == "permission_denied":
        typed = ProofRejected("authority", "")
    else:
        return exc
    typed.field, typed.detail = exc.field, exc.detail
    return typed


@contextmanager
def _refused(store: PostgresStore) -> Iterator[None]:
    """The reference's ``refused`` port: the fence/unknown/proof refusals as the contract's typed
    refusal classes, and the store's failures (the PG ``translated`` + session-error mapping) as the
    contract's error family."""
    try:
        with refusals(store):
            yield
    except AuthorityRefusal as exc:
        if type(exc) is not AuthorityRefusal:
            raise
        typed = _typed(exc)
        if typed is exc:
            raise
        raise typed from exc


def _text_check(value: str) -> str:
    """The reference's ``_text``: the identity texts (str, non-empty, at most 1024 chars)."""
    if not isinstance(value, str) or not value or len(value) > 1024:
        raise AuthorityRefusal("invalid_params")
    return value


def _epoch_check(conn, epoch: int) -> None:
    """The epoch fence (the reference's ``_epoch``): the singleton row's stored epoch, else ``stale_epoch``."""
    row = conn.execute("SELECT epoch FROM runtime_epoch WHERE singleton = 1").fetchone()
    if type(epoch) is not int or row is None or row["epoch"] != epoch:
        raise AuthorityRefusal("stale_epoch")


def _session_row(conn, session_id: str) -> dict:
    """The reference's ``_session``: the session row, else ``not_found``."""
    row = conn.execute("SELECT * FROM sessions WHERE id = %s", (session_id,)).fetchone()
    if row is None:
        raise AuthorityRefusal("not_found")
    return row


def _terminal_admission_row(conn, admission_id: str) -> dict | None:
    """The reference's ``terminal_admission``: the terminal tombstone under the admission prefix."""
    saved = conn.execute("SELECT value FROM state_meta WHERE key = %s", (ADMISSION_PREFIX + admission_id,)).fetchone()
    return json.loads(saved["value"]) if saved else None


def _admission_row(conn, admission_id: str) -> dict:
    """The reference's ``_admission``: the live row, else its terminal tombstone, else ``not_found``."""
    row = conn.execute("SELECT * FROM session_admissions WHERE admission_id = %s", (admission_id,)).fetchone()
    if row is None:
        row = _terminal_admission_row(conn, admission_id)
    if row is None:
        raise AuthorityRefusal("not_found")
    return row


def _row(row: Mapping[str, Any] | None) -> dict | None:
    """The reference's ``_row``: the row's record shape (the decoded ``payload``; digest, lineage and
    the tombstone's retention stamp dropped)."""
    if row is None:
        return None
    result = dict(row)
    result["payload"] = json.loads(result.pop("payload_json"))
    result.pop("payload_digest", None)
    result.pop("lineage_json", None)
    result.pop("retired_at", None)  # tombstone retention stamp, not admission state
    return result


def _record(row: Mapping[str, Any] | None) -> AdmissionRecord | None:
    if row is None:
        return None
    return AdmissionRecord(**{name: row[name] for name in _RECORD_FIELDS})


def _claimed(row: Mapping[str, Any]) -> ClaimedAdmission:
    return ClaimedAdmission(**{name: row[name] for name in _RECORD_FIELDS},
                            transcript_mark=row["transcript_mark"],
                            queued_owner_epoch=row["queued_owner_epoch"])


def replacement_binding(principal_id: str, session_id: str, request_id: str) -> str:
    """The durable key binding a replacement's identity to the admission it retired (the reference's
    ``replacement_binding``: the same fingerprint under the replacement prefix)."""
    return _REPLACEMENT_PREFIX + admission_fingerprint(canonical_target=session_id,
                                                       payload={"principal": principal_id, "request": request_id})


# ── the compression-chain and physical-target walks (the reference's ``_canonical_chain`` /
# ``local_physical_target`` / ``runtime_lineage`` ports, on the PG continuation-child selector) ──


def _canonical_tip(conn, session_id: str) -> str:
    """The live tip of ``session_id``'s compression chain (the reference's ``_canonical_chain``: the
    same continuation-child selector; a cycle or an overlong chain is an admission conflict)."""
    _session_row(conn, session_id)
    chain = [session_id]
    for _ in range(100):
        child = conn.execute(chain_step_sql(), (chain[-1],)).fetchone()
        if child is None:
            return chain[-1]
        if child["id"] in chain:
            raise AuthorityRefusal("admission_conflict")
        chain.append(child["id"])
    raise AuthorityRefusal("admission_conflict")


def _local_physical_target(conn, session_id: str) -> str:
    """A local owner's current transcript (the reference's ``local_physical_target``): the session
    itself when it holds no local policy receipt; the receipt's validated lineage tip otherwise
    (an unverifiable receipt is ``storage_unavailable``, as the reference's
    ``validate_local_lineage``: the tip must have no continuation step of its own)."""
    saved = conn.execute("SELECT value FROM state_meta WHERE key = %s", (POLICY_PREFIX + session_id,)).fetchone()
    if saved is None:
        return session_id
    receipt = json.loads(saved["value"])
    if not isinstance(receipt, dict) or receipt.get("session_id") != session_id:
        raise AuthorityRefusal("storage_unavailable")
    entry = receipt.get("entry")
    if not isinstance(entry, dict) or entry.get("session_id") is None:
        raise AuthorityRefusal("storage_unavailable")
    lineage = receipt.get("lineage")
    if (not isinstance(lineage, list) or not lineage or lineage[-1] != entry["session_id"]
            or len(set(lineage)) != len(lineage)):
        raise AuthorityRefusal("storage_unavailable")
    for parent_id, child_id in zip(lineage, lineage[1:]):
        parent = conn.execute("SELECT end_reason FROM sessions WHERE id = %s", (parent_id,)).fetchone()
        child = conn.execute("SELECT parent_session_id, model_config FROM sessions WHERE id = %s", (child_id,)).fetchone()
        if parent is None or child is None or child["parent_session_id"] != parent_id:
            raise AuthorityRefusal("storage_unavailable")
        if parent["end_reason"] == "session_reset":
            if json.loads(child["model_config"] or "{}").get("_reset_from") != parent_id:
                raise AuthorityRefusal("storage_unavailable")
        else:
            tip = conn.execute(chain_step_sql(), (parent_id,)).fetchone()
            if not (parent["end_reason"] == "compression" and tip is not None and tip["id"] == child_id):
                raise AuthorityRefusal("storage_unavailable")
    if conn.execute(chain_step_sql(), (lineage[-1],)).fetchone() is not None:
        raise AuthorityRefusal("storage_unavailable")
    return lineage[-1]


def _runtime_lineage_targets(conn, session_id: str) -> set:
    """The logical conversation + physical transcript lineage the worker exclusion follows (the
    SQLite ``runtime_lineage`` port: the session and its physical target, the chat's local
    conversation when the session is one of its ids, every one's compression ancestors and
    descendants)."""
    targets = {session_id, _local_physical_target(conn, session_id)}
    row = conn.execute("SELECT chat_id FROM sessions WHERE id = %s", (session_id,)).fetchone()
    if row and row["chat_id"]:
        chat_id = row["chat_id"]
        saved = conn.execute("SELECT value FROM state_meta WHERE key = %s", (POLICY_PREFIX + chat_id,)).fetchone()
        if saved is not None:
            lineage = json.loads(saved["value"]).get("lineage", [chat_id])
            if session_id in (lineage if isinstance(lineage, list) else [chat_id]):
                targets.update({chat_id, _local_physical_target(conn, chat_id)})
    frontier = list(targets)
    for _ in range(1000):  # the reference's ``_CHAIN_CAP`` round cap on the expansion
        expanded = _expand_compression_lineage(conn, frontier)
        frontier = [sid for sid in expanded if sid not in targets]
        if not frontier:
            return targets
        targets.update(frontier)
    raise AuthorityRefusal("storage_unavailable")


def _expand_compression_lineage(conn, ids: Sequence[str]) -> list:
    """One both-direction compression-edge step from *ids* (the SQLite
    ``_expand_compression_lineage_ids`` port): forward through every continuation child and
    backward through the compression parent, the same edge definition as ``chain_step_sql``
    (branch/reset/delegate/tool children are not continuations)."""
    ids = [sid for sid in ids if sid]
    if not ids:
        return []
    edge = (
        f"parent.end_reason = 'compression' "
        f"AND {json_marker_absent_sql('child.model_config', '_branched_from')} "
        f"AND {json_marker_absent_sql('child.model_config', '_delegate_from')} "
        f"AND NOT ({reset_child_sql('child')}) "
        f"AND COALESCE(child.source, '') != 'tool'"
    )
    placeholders = ",".join(["%s"] * len(ids))
    out = [r["child_id"] for r in conn.execute(
        "SELECT child.id AS child_id FROM sessions parent "
        "JOIN sessions child ON child.parent_session_id = parent.id "
        f"WHERE parent.id IN ({placeholders}) AND {edge}", ids).fetchall()]
    out += [r["parent_id"] for r in conn.execute(
        "SELECT parent.id AS parent_id FROM sessions child "
        "JOIN sessions parent ON child.parent_session_id = parent.id "
        f"WHERE child.id IN ({placeholders}) AND {edge}", ids).fetchall()]
    return out


def _worker_states(conn, session_id: str) -> set:
    """The reference's ``worker_states``: the live worker statuses on the session's whole lineage."""
    targets = sorted(_runtime_lineage_targets(conn, session_id))
    if not targets:
        return set()
    placeholders = ",".join(["%s"] * len(targets))
    rows = conn.execute(
        f"SELECT DISTINCT status FROM worker_executions WHERE session_id IN ({placeholders}) "
        "AND status IN ('registered','running','unknown')", tuple(targets)).fetchall()
    return {row["status"] for row in rows}


def _terminal_api_admissions(conn, request_id: str) -> list:
    """Retired API admissions that carried *request_id* (the reference's ``terminal_api_admissions``:
    the tombstones' prefix scan, the ``api`` principal and the request id filtered on the stored row):
    a deleted session's tombstones spend its Idempotency-Key for every other target."""
    rows = conn.execute("SELECT value FROM state_meta WHERE key LIKE %s", (ADMISSION_PREFIX + "%",)).fetchall()
    out = []
    for row in rows:
        try:
            data = json.loads(row["value"])
        except (TypeError, ValueError):
            continue
        if isinstance(data, dict) and data.get("principal_id") == "api" and data.get("request_id") == request_id:
            out.append(data)
    return out


# ── worker retirement (the SQLite ``_runtime_workers`` steps the settlement and recovery run) ──


def _compact_receipts(conn, execution_id: str) -> None:
    """The reference's ``_compact``: the execution's read receipts (the rows carrying a transcript
    copy or a full session row) shrink to the retired marker; ``payload_digest``/``record_json`` stay
    shaped so a late duplicate is still recognised."""
    receipts = "CASE WHEN json_text_valid(record_json) THEN record_json::jsonb END"
    conn.execute(
        "UPDATE worker_receipts SET record_json = %s WHERE execution_id = %s "
        f"AND jsonb_typeof({receipts} #> '{{messages}}') IS NOT NULL",
        (json.dumps(RETIRED_RECEIPT), execution_id))
    conn.execute(
        "UPDATE worker_receipts SET record_json = %s WHERE execution_id = %s "
        f"AND jsonb_typeof({receipts} #> '{{session}}') IS NOT NULL",
        (json.dumps(RETIRED_RECEIPT), execution_id))


def _end_child_row(conn, execution_id: str) -> None:
    """The reference's ``_end_child_row``: the child's open session row ends (``agent_close``)."""
    conn.execute(
        "UPDATE sessions SET ended_at = %s, end_reason = 'agent_close' WHERE ended_at IS NULL "
        "AND id = (SELECT session_id FROM worker_executions WHERE execution_id = %s)",
        (time.time(), execution_id))


def _retire_one_child(conn, execution_id: str) -> None:
    """The reference's ``_retire_children`` step: terminal, receipts compacted, the child row ended."""
    conn.execute("UPDATE worker_executions SET status = 'terminal' WHERE execution_id = %s", (execution_id,))
    _compact_receipts(conn, execution_id)
    _end_child_row(conn, execution_id)


def _retire_stale_children(conn, epoch: int) -> None:
    """The reference's ``retire_stale_children`` (boot recovery): a child dies with its worker and its
    owner, so every live child of a foreign epoch is terminal and never adoptable."""
    rows = conn.execute(
        "SELECT execution_id FROM worker_executions WHERE kind = 'child' AND status <> 'terminal' "
        "AND owner_epoch != %s", (epoch,)).fetchall()
    for row in rows:
        _retire_one_child(conn, row["execution_id"])


def _compact_terminal_receipts(conn, execution_id: str) -> None:
    """The reference's ``compact_terminal_receipts``: the execution's live descendants (the
    ``/child:`` id range) go terminal with it, its read receipts shrink, and a child that finished
    itself ends its own row."""
    low = execution_id + CHILD_SEPARATOR
    high = execution_id + CHILD_SEPARATOR[:-1] + ";"
    rows = conn.execute(
        "SELECT execution_id FROM worker_executions WHERE execution_id >= %s AND execution_id < %s "
        "AND status <> 'terminal'", (low, high)).fetchall()
    for row in rows:
        _retire_one_child(conn, row["execution_id"])
    _compact_receipts(conn, execution_id)
    if CHILD_SEPARATOR in execution_id:
        _end_child_row(conn, execution_id)


def _retire_admission_workers(conn, row: Mapping[str, Any], owner_epoch: int | None) -> None:
    """The reference's ``_retire_admission_workers``: retire the worker rows this admission's
    generation owns on BOTH the logical FIFO owner and its current physical transcript (reset/
    compression move the private worker assignment to the child while target_session_id stays the
    logical id)."""
    lineage = json.loads(row["lineage_json"])
    targets = sorted({row["target_session_id"], lineage[-1] if lineage else row["target_session_id"]})
    placeholders = ",".join(["%s"] * len(targets))
    scope = f"session_id IN ({placeholders}) AND generation = %s AND owner_epoch = %s"
    # Settlement closes an execution that never wrote execution.finish: its read copies go too.
    rows = conn.execute(f"SELECT execution_id FROM worker_executions WHERE {scope} AND status <> 'terminal'",
                       (*targets, row["generation"], owner_epoch)).fetchall()
    for row_id in rows:
        _compact_terminal_receipts(conn, row_id["execution_id"])
    conn.execute(f"UPDATE worker_executions SET status = 'terminal' WHERE {scope}",
                 (*targets, row["generation"], owner_epoch))


def _requeue_transfer(conn, *, admission_id: str, epoch: int, generation: int) -> bool:
    """The reference's ``_requeue_transfer``: a transfer still bound to the claim goes back to its
    FIFO place (its ``seq``); False when the claim's settlement already decided it."""
    changed = conn.execute(
        "UPDATE session_admissions SET status = 'queued', outcome = NULL, generation = NULL "
        "WHERE admission_id = %s AND status = 'terminal' AND outcome = %s AND owner_epoch = %s "
        "AND generation = %s", (admission_id, STEERING, epoch, generation))
    return changed.rowcount == 1


def _settle_steer_transfers(conn, *, epoch: int, generation: int, transfers: Mapping[str, bool] | None, outcome: str) -> None:
    """The reference's ``settle_steer_transfers``: each transfer bound to the claim settles ``steered``
    only if its text reached the agent and the claim completed; a delivered text under any other
    ending is ``interrupted``; an undelivered one is queued again."""
    for admission_id, delivered in (transfers or {}).items():
        if not delivered:
            _requeue_transfer(conn, admission_id=admission_id, epoch=epoch, generation=generation)
            continue
        conn.execute(
            "UPDATE session_admissions SET outcome = %s WHERE admission_id = %s AND status = 'terminal' "
            "AND outcome = %s AND owner_epoch = %s AND generation = %s",
            ("steered" if outcome == "completed" else "interrupted", admission_id, STEERING, epoch, generation))


# ── the typed extras the admission write runs before its insert (the reference's ``_admission_owner`` port) ──


def _replace_queued(conn, admission_id: str, session_id: str) -> None:
    """The reference's ``_replace_queued``: cancel the replaced queued tick in the replacement's own
    transaction; refuse when that tick was claimed or settled first."""
    changed = conn.execute(
        "UPDATE session_admissions SET status = 'terminal', outcome = 'cancelled' "
        "WHERE admission_id = %s AND target_session_id = %s AND status = 'queued'",
        (admission_id, session_id))
    if changed.rowcount != 1:
        raise AuthorityRefusal("stale_generation")


def _fold_queued(conn, admission_id: str, session_id: str, skip: frozenset) -> None:
    """The reference's ``_fold_queued``: a completion fold that must still be the FIFO tail when it
    commits (the ``skip`` ids are the finishing process's ticks); the replaced row keeps only its own
    identity (its text becomes ``folded_payload``'s marker)."""
    row = conn.execute(
        "SELECT seq, payload_json FROM session_admissions WHERE admission_id = %s "
        "AND target_session_id = %s AND status = 'queued'", (admission_id, session_id)).fetchone()
    if row is None:
        raise AuthorityRefusal("stale_generation")
    later = conn.execute(
        "SELECT admission_id FROM session_admissions WHERE target_session_id = %s "
        "AND status = 'queued' AND seq > %s", (session_id, row["seq"])).fetchall()
    if any(other["admission_id"] not in skip for other in later):
        raise AuthorityRefusal("stale_generation")
    _replace_queued(conn, admission_id, session_id)
    payload = folded_payload(json.loads(row["payload_json"]))
    conn.execute("UPDATE session_admissions SET payload_json = %s WHERE admission_id = %s",
                 (json.dumps(payload, separators=(",", ":"), ensure_ascii=False), admission_id))


def _retire_replaced_input(conn, admission_id: str, binding: str) -> None:
    """The reference's ``retire_replaced_input``: inside the replacement's own admission write, the
    replaced row (still queued) is cancelled in the same transaction, bound to its successor's key."""
    row = _admission_row(conn, admission_id)
    if row["status"] != "queued":
        raise AuthorityRefusal("admission_started")
    conn.execute(_CANCEL_QUEUED, (admission_id,))
    conn.execute("INSERT INTO state_meta (key, value) VALUES (%s, %s) "
                 "ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value",
                 (binding, json.dumps(admission_id)))


def _check_replaced_input(conn, admission_id: str, binding: str) -> None:
    """The reference's ``check_replaced_input``: the exact replay of a replacement answers only the
    request that retired ``admission_id``."""
    saved = conn.execute("SELECT value FROM state_meta WHERE key = %s", (binding,)).fetchone()
    if saved is None or json.loads(saved["value"]) != admission_id:
        raise AuthorityRefusal("admission_conflict")


def _one_target(conn, session_id: str, request_id: str, client_key: bool) -> None:
    """The reference's ``_one_target``: one API request id names one admission. A deleted
    session's tombstone still spends a client key; a server-minted id skips that scan."""
    if conn.execute("SELECT 1 FROM session_admissions WHERE principal_id = 'api' AND request_id = %s "
                   "AND target_session_id != %s LIMIT 1", (request_id, session_id)).fetchone():
        raise AuthorityRefusal("admission_conflict")
    if not client_key:
        return
    if any(row["target_session_id"] != session_id for row in _terminal_api_admissions(conn, request_id)):
        raise AuthorityRefusal("admission_conflict")


def _admit_target(conn, session_id: str, request_id: str, settings: Mapping[str, Any], client_key: bool) -> None:
    """The reference's ``_admit_target``: the one-target guard, then the session's stored turn
    settings (committed with the admitted turn rather than by the drain)."""
    _one_target(conn, session_id, request_id, client_key=client_key)
    conn.execute("INSERT INTO state_meta (key, value) VALUES (%s, %s) "
                 "ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value",
                 (_SETTINGS_PREFIX + session_id, canonical_json(dict(settings))))


def _retry_rewrite(store: PostgresStore, conn, rewrite: TranscriptRewrite) -> None:
    """The reference's ``_retry_rewrite`` on the PG transcript doors: the carrier-aware rewind or the
    active-transcript replacement, each refusing when the transcript changed since the plan was read
    (a failure other than a refusal is ``RetryRewriteRefused``)."""
    try:
        if isinstance(rewrite, CarrierRewind):
            rewind_in_transaction(store, conn, rewrite.session_id, rewrite.target_row_id,
                                 preserve_compaction_handoff=True,
                                 expected_active_ids=list(rewrite.expected_active_ids),
                                 expected_target_content=rewrite.expected_target_content)
        else:  # ActiveReplace
            _check_transcript_write_guards(conn, rewrite.session_id, None,
                                          reject_active_turn_lease=True,
                                          reject_active_compression_lock=True)
            if rewrite.expected_active_ids is not None:
                active = [int(r["id"]) for r in conn.execute(
                    "SELECT id FROM messages WHERE session_id = %s AND active = 1 ORDER BY id",
                    (rewrite.session_id,)).fetchall()]
                if active != list(rewrite.expected_active_ids):
                    raise RuntimeError("active transcript changed before the rewrite could be persisted")
            _replace_in_transaction(conn, rewrite.session_id, [dict(m) for m in rewrite.messages],
                                    active_only=True, reject_active_turn_lease=True)
    except (AuthorityRefusal, psycopg.OperationalError, psycopg.errors.DeadlockDetected):
        raise  # gate refusals and writer contention keep their own handling
    except Exception as exc:
        raise RetryRewriteRefused(rewrite.session_id, str(exc)) from exc


def _hosted_authorize_write(conn, epoch: int, guard: "HostedMemberGuard") -> None:
    """The hosted member check inside the admission's own off-loop write (the reference's
    ``_hosted_authorize_write``): the epoch must be current and the member still allowed to submit;
    a guard that only returns False would be ignored wherever the admission hook signals refusal by
    raising, so the refusal is raised.

    The check runs on the PG hosted-room state, which lands with the stage-4 hosted rooms port;
    until that module provides ``_member_authorized_locked`` on the PG schema the check names its D8
    gap instead of guessing on a schema that has no hosted-room tables yet."""
    from persistence.postgres.gateway import hosted_rooms as _pg_hosted
    check = getattr(_pg_hosted, "_member_authorized_locked", None)
    if check is None:
        raise NotImplementedError(
            "postgres PgAdmissionsRepository hosted member check awaits the stage-4 hosted rooms port (D8)")
    _epoch_check(conn, epoch)
    if not check(conn, guard, operation="submit"):
        raise AuthorityRefusal("permission_denied")


def _authorize_admission(conn, *, epoch: int, principal_id: str, session_id: str, request_id: str,
                        replaces: QueuedReplacement | None, supersedes: str | None,
                        api_target: ApiTargetClaim | None, rewrite: TranscriptRewrite | None,
                        hosted: "HostedMemberGuard | None", fold: QueuedFold | None,
                        store: PostgresStore) -> None:
    """The typed extras of one admission, inside its write after the queue cap and gate and before
    the insert (the reference's ``authorize_admission`` order)."""
    if replaces is not None:
        _retire_replaced_input(conn, replaces.admission_id, replacement_binding(principal_id, session_id, request_id))
    if supersedes is not None:
        _replace_queued(conn, supersedes, session_id)
    if api_target is not None:
        _admit_target(conn, session_id, request_id, dict(api_target.settings), client_key=api_target.client_key)
    if rewrite is not None:
        _retry_rewrite(store, conn, rewrite)
    if hosted is not None:
        _hosted_authorize_write(conn, epoch, hosted)
    if fold is not None:
        _fold_queued(conn, fold.admission_id, session_id, fold.skip_ids)


# ── the discarded-turn closer (the reference's ``close_discarded_turn`` port, on the PG message door) ──


def _append_messages_in_transaction(conn, session_id: str, messages: list[dict[str, Any]]) -> int:
    """The reference's ``_append_messages_in_transaction`` on the PG message door: the transcript
    write guards, the repair-batch resolution, the insert (the DDL triggers mint ``message_uid``
    and ``display_order``) and the session counter bump, in the caller's transaction."""
    _check_transcript_write_guards(conn, session_id, None)
    inserted_rows = resolve_and_repair_transcript_batch(
        conn, session_id, messages,
        encode_content_fn=_encode_content, decode_content_fn=_decode_content,
        serialize_message_fn=partial(_serialize, session_id), decode_row_fn=_decoded_repair_row)
    inserted, tool_calls_total = _insert_message_rows(conn, session_id, inserted_rows)
    _bump_counters(conn, session_id, inserted, tool_calls_total)
    return inserted


def _close_discarded_turn(conn, row: Mapping[str, Any], notice: str | None, *,
                          replace_after: int | None = None, display_metadata: Mapping[str, Any] | None = None) -> bool:
    """The reference's ``close_discarded_turn``: close a discarded ``unknown`` turn the way a failed
    turn is closed, on the resolution's own transaction, on the CURRENT physical transcript.

    ``notice`` replaces the hedged copy (written even on a closed tail: nobody resolved the turn, so
    the transcript must say why); ``replace_after`` (the claim's transcript mark) makes the notice
    replace THIS turn's own failed-turn boundary instead of stacking a second one; ``display_metadata``
    (``{error, error_surface}``) rides the boundary so a reopened chat redraws the turn's error card.
    Idempotent on a closed tail: a user tail (or an assistant tail with no live tool calls) needs no
    boundary unless the notice says why the turn was lost."""
    target = _canonical_tip(conn, _local_physical_target(conn, row["target_session_id"]))
    tail = conn.execute(
        "SELECT role, tool_calls, id, display_kind FROM messages WHERE session_id = %s AND active = 1 "
        "AND role NOT IN ('session_meta','system') ORDER BY id DESC LIMIT 1", (target,)).fetchone()
    if (notice is not None and type(replace_after) is int and tail is not None
            and tail["role"] == "assistant" and tail["display_kind"] == FAILED_TURN_DISPLAY_KIND
            and tail["id"] > replace_after):
        conn.execute("UPDATE messages SET content = %s, display_metadata = %s WHERE id = %s",
                     (_payload_bytes(_encode_content(notice)), _encode_display_metadata(display_metadata), tail["id"]))
        return True
    if notice is None and (tail is None or not (
            tail["role"] == "user" or (tail["role"] == "assistant" and tail["tool_calls"] not in (None, "", "[]")))):
        return False
    boundary = {"role": "assistant", "content": notice or PARTIAL_FAILED_TURN_NOTICE, "timestamp": time.time(),
                "display_kind": FAILED_TURN_DISPLAY_KIND}
    if display_metadata:
        boundary["display_metadata"] = display_metadata
    _append_messages_in_transaction(conn, target, [boundary])
    return True


# ── the gateway's admission reads (the reference's SQL on the PG dialect) ──

# The JSONB-path guards (the SQLite ``json_type``/``json_extract`` shapes, non-throwing on the PG
# dialect: the ``CASE`` guard keeps the ``::jsonb`` cast out of reach of invalid JSON; ``jsonb_typeof``
# answers ``'null'`` for a JSON null but SQL NULL when the path is absent, as ``json_type``).
_JSONB = "CASE WHEN json_text_valid(payload_json) THEN payload_json::jsonb END"


def _JSONB_FOR(alias: str) -> str:
    """The :data:`_JSONB` guard on an aliased table column (the ``a.payload_json`` shapes)."""
    return f"CASE WHEN json_text_valid({alias}.payload_json) THEN {alias}.payload_json::jsonb END"

# The reference's ``_CANDIDATES`` (settled API admissions carrying an inline image beside committed
# media, or a client history outside the Responses API; LIKE is the ASCII case-insensitive match the
# reference relies on, so the PG dialect's ``ILIKE``). The ``#`` path operator answers the JSON value
# (SQL NULL when the path is absent, as the reference's ``json_type`` IS NOT NULL test) and
# ``jsonb_typeof`` answers the value's type name (``'array'`` for a stored history).
_API_CANDIDATE_WHERE = (
    "status = 'terminal' AND principal_id = 'api' AND ("
    f"jsonb_typeof({_JSONB} #> '{{api_turn_v1,media}}') IS NOT NULL AND payload_json ILIKE '%%data:image/%%' "
    f"OR (jsonb_typeof({_JSONB} #> '{{api_turn_v1,history}}') = 'array' AND request_id NOT ILIKE 'responses:%%'))")


def _history_mentions(conn, references: Sequence[Mapping[str, Any]], after_id: int = 0) -> set:
    """Candidate paths any transcript row (id > after_id) mentions raw or JSON-escaped, in ONE
    messages pass (the reference's ``_history_mentions``: the content-addressed tail with either
    separator, prefiltered in SQL on the literal digest prefixes so only rows that could name a
    candidate reach Python's exact substring test)."""
    needles = {}
    for reference in references:
        raw = reference["path"]
        for tail in (reference["sha256"] + sep + Path(raw).name for sep in ("/", "\\")):
            needles[tail] = needles[json.dumps(tail)[1:-1]] = raw
    prefixes = sorted({reference["sha256"] for reference in references})
    if not prefixes:
        return set()
    query = ("SELECT content FROM messages WHERE EXISTS "
             "(SELECT 1 FROM jsonb_array_elements_text(%s::jsonb) AS digest "
             "WHERE strpos(hermes_nul_safe_text(content), digest) > 0)"
             + (" AND id > %s" if after_id else ""))
    mentioned = set()
    rows = conn.execute(query, (json.dumps(prefixes),) + ((after_id,) if after_id else ())).fetchall()
    for (content,) in ((row["content"],) for row in rows):
        text = bytes(content).decode("utf-8", "replace") if content is not None else ""
        mentioned.update(raw for needle, raw in needles.items() if needle in text)
    return mentioned


def _held_media_paths(conn, media_root: str) -> set:
    """The reference's ``_held_media_paths``: each live admission's media and hosted documents, and
    every API turn's images (the terminal API rows stay held: their images are canonical history
    context). A live hosted row's prompt is projected too, because its documents are held by text."""
    rows = conn.execute(
        "SELECT status, request_id, payload_json FROM session_admissions WHERE status <> 'terminal' "
        f"OR jsonb_typeof({_JSONB} #> '{{api_turn_v1,media}}') IS NOT NULL").fetchall()
    held = set()
    for row in rows:
        payload = json.loads(row["payload_json"])
        api = (payload.get("api_turn_v1") or {}).get("media")
        references = list(api or ())
        if row["status"] != "terminal":
            references.extend((payload.get("attachments_v1") or {}).get("media", ()))
            references.extend((payload.get("native_text_v1") or {}).get("media", ()))
            if str(row["request_id"]).startswith("hosted:"):
                references.extend(hosted_document_references(row["request_id"],
                                                            {"text": payload.get("text")}, media_root))
        held.update(reference["path"] for reference in references)
    return held


def _settled_media(conn) -> list:
    """The reference's ``settled_media``: the ``attachments_v1`` and ``native_text_v1`` media lists of
    the terminal rows, one list per row (the JSONB-path guard selects the rows that carry either)."""
    rows = conn.execute(
        "SELECT payload_json FROM session_admissions WHERE status = 'terminal' AND ("
        f"jsonb_typeof({_JSONB} #> '{{attachments_v1,media}}') IS NOT NULL "
        f"OR jsonb_typeof({_JSONB} #> '{{native_text_v1,media}}') IS NOT NULL)").fetchall()
    media = []
    for row in rows:
        payload = json.loads(row["payload_json"])
        attachments = (payload.get("attachments_v1") or {}).get("media") or []
        native = (payload.get("native_text_v1") or {}).get("media") or []
        media.append(list(attachments) + list(native))
    return media


# ── the repository ─────────────────────────────────────────────────────────────────────────────────


class PgAdmissionsRepository:
    """The admission ledger of the bound profile's store schema (stage 2): every ``AdmissionRepository``
    verb ported; the one D8 residual is the ``admit`` hosted member check, which awaits the stage-4
    hosted rooms port (its state does not exist on the PG schema yet)."""

    def __init__(self, store: PostgresStore) -> None:
        self._store = store

    def _refused(self):
        """The reference's ``refused`` on the PG store: the typed fence/unknown/proof refusals, the
        store's failures as the contract's error family."""
        return _refused(self._store)

    # ── writes ──────────────────────────────────────────────────────────────

    def admit(self, *, epoch: int, principal_id: str, session_id: str, request_id: str, payload: Mapping[str, Any],
              intent: Literal["queue", "steer", "redirect"] = "queue", gate: AdmissionGate = None,
              replaces: QueuedReplacement | None = None, supersedes: str | None = None,
              api_target: ApiTargetClaim | None = None, rewrite: TranscriptRewrite | None = None,
              hosted: "HostedMemberGuard | None" = None, fold: QueuedFold | None = None) -> tuple[AdmissionRecord, bool]:
        """Admit one input; ``(record, created)`` (the reference's ``admit_session_input``: identity
        texts and the payload encoding first, the retired-target tombstone read, then one write with
        the epoch fence, the queue cap, the gate and the typed extras before the queued insert)."""
        for value in (principal_id, session_id, request_id):
            _text_check(value)
        if intent not in ("queue", "steer", "redirect"):
            raise AuthorityRefusal("invalid_params")
        encoded = canonical_json(payload)
        retired = self.retired_replay(epoch=epoch, principal_id=principal_id, session_id=session_id,
                                      request_id=request_id, payload=payload, intent=intent, _internal=True)
        if retired is not None:
            return retired, False
        digest = admission_fingerprint(canonical_target=session_id,
                                       payload={"input": retry_input(json.loads(encoded)), "intent": intent})
        with self._refused():
            with self._store.tx() as conn:
                _epoch_check(conn, epoch)
                _session_row(conn, session_id)
                old = conn.execute(
                    "SELECT * FROM session_admissions WHERE principal_id = %s AND target_session_id = %s "
                    "AND request_id = %s", (principal_id, session_id, request_id)).fetchone()
                if old is not None:
                    if old["payload_digest"] != digest:
                        raise AuthorityRefusal("admission_conflict", **ADMITTED_REQUEST_CONFLICT)
                    if replaces is not None:
                        _check_replaced_input(conn, replaces.admission_id,
                                              replacement_binding(principal_id, session_id, request_id))
                    return _record(_row(old)), False
                queued = conn.execute(
                    "SELECT COUNT(*) AS n FROM session_admissions WHERE target_session_id = %s "
                    "AND status = 'queued'", (session_id,)).fetchone()["n"]
                # Main's cap holds in both states: a FIFO paused behind an unknown turn runs every
                # accepted input after Discard, so it must not grow without bound while nobody resolves it.
                if queued >= ADMISSION_QUEUE_MAX:
                    raise AuthorityRefusal("queue_full")
                raise_gate(gate)
                _authorize_admission(conn, epoch=epoch, principal_id=principal_id, session_id=session_id,
                                     request_id=request_id, replaces=replaces, supersedes=supersedes,
                                     api_target=api_target, rewrite=rewrite, hosted=hosted, fold=fold,
                                     store=self._store)
                admission_id = uuid.uuid4().hex
                conn.execute(
                    "INSERT INTO session_admissions (admission_id, request_id, principal_id, target_session_id, "
                    "lineage_json, payload_json, payload_digest, intent, status, owner_epoch) "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, 'queued', %s)",
                    (admission_id, request_id, principal_id, session_id, json.dumps([session_id]),
                     encoded, digest, intent, epoch))
                return _record(_row(_admission_row(conn, admission_id))), True

    def claim(self, *, epoch: int, session_id: str, gate: AdmissionGate = None,
              head: str | None | AnyHead = ANY_HEAD) -> ClaimedAdmission | Literal[False] | None:
        """Claim the FIFO head under the next execution generation (the reference's
        ``claim_session_input``: the epoch fence, the gate, the unknown/started block, the worker
        exclusion, the head check, the generation bump and the claim, the transcript high-water mark)."""
        with self._refused():
            with self._store.tx() as conn:
                _epoch_check(conn, epoch)
                raise_gate(gate)
                session = _session_row(conn, session_id)
                blocked = conn.execute(
                    "SELECT status FROM session_admissions WHERE target_session_id = %s "
                    "AND status IN ('started','unknown')", (session_id,)).fetchall()
                if any(row["status"] == "unknown" for row in blocked):
                    raise AuthorityRefusal("unknown_execution")
                workers = _worker_states(conn, session_id)
                if "unknown" in workers:
                    raise AuthorityRefusal("unknown_execution")
                if blocked or workers:
                    return None
                row = conn.execute(
                    "SELECT * FROM session_admissions WHERE target_session_id = %s AND status = 'queued' "
                    "ORDER BY seq LIMIT 1", (session_id,)).fetchone()
                if head is not ANY_HEAD and (row["admission_id"] if row is not None else None) != head:
                    return False
                if row is None:
                    return None
                generation = session["runtime_generation"] + 1
                conn.execute("UPDATE sessions SET runtime_generation = %s WHERE id = %s", (generation, session_id))
                changed = conn.execute(
                    "UPDATE session_admissions SET status = 'started', owner_epoch = %s, generation = %s "
                    "WHERE admission_id = %s AND status = 'queued'",
                    (epoch, generation, row["admission_id"]))
                if changed.rowcount != 1:
                    raise AuthorityRefusal("stale_generation")
                claimed = _row(_admission_row(conn, row["admission_id"]))
                # The transcript high-water mark at the claim (``messages.id`` is the schema-wide
                # identity): a row above it on this conversation was written by this turn, so
                # settlement can tell this turn's own failed-turn boundary from an earlier turn's.
                claimed["transcript_mark"] = int(conn.execute(
                    "SELECT COALESCE(MAX(id), 0) AS mark FROM messages").fetchone()["mark"])
                claimed["queued_owner_epoch"] = row["owner_epoch"]
                return _claimed(claimed)

    def return_claim(self, *, epoch: int, admission_id: str, generation: int,
                     queued_owner_epoch: int | None) -> AdmissionRecord:
        """Undo a claim whose turn never started (the reference's ``return_session_claim``: the
        claim's two updates reversed, the session back at ``generation - 1``)."""
        with self._refused():
            with self._store.tx() as conn:
                _epoch_check(conn, epoch)
                row = _admission_row(conn, admission_id)
                if row["status"] == "queued" and row["generation"] is None:
                    return _record(_row(row))
                if (row["status"] != "started" or row["owner_epoch"] != epoch
                        or type(generation) is not int or row["generation"] != generation):
                    raise AuthorityRefusal("stale_generation")
                session = _session_row(conn, row["target_session_id"])
                lineage = json.loads(row["lineage_json"])
                targets = sorted({row["target_session_id"], lineage[-1] if lineage else row["target_session_id"]})
                placeholders = ",".join(["%s"] * len(targets))
                if (session["runtime_generation"] != generation
                        or conn.execute(f"SELECT 1 FROM worker_executions WHERE session_id IN ({placeholders}) "
                                        "AND owner_epoch = %s AND generation = %s LIMIT 1",
                                        (*targets, epoch, generation)).fetchone()
                        or conn.execute("SELECT 1 FROM session_admissions WHERE target_session_id = %s "
                                        "AND status = 'terminal' AND outcome = %s AND owner_epoch = %s "
                                        "AND generation = %s LIMIT 1",
                                        (row["target_session_id"], STEERING, epoch, generation)).fetchone()):
                    raise AuthorityRefusal("stale_generation")
                conn.execute("UPDATE session_admissions SET status = 'queued', owner_epoch = %s, generation = NULL "
                             "WHERE admission_id = %s AND status = 'started'",
                             (queued_owner_epoch, admission_id))
                conn.execute("UPDATE sessions SET runtime_generation = %s WHERE id = %s",
                             (generation - 1, row["target_session_id"]))
                return _record(_row(_admission_row(conn, admission_id)))

    def settle(self, *, epoch: int, admission_id: str, generation: int,
               outcome: Literal["completed", "interrupted", "rejected", "failed"],
               result: Mapping[str, Any] | None = None, transfers: Mapping[str, bool] | None = None,
               boundary: TurnBoundary | None = None) -> AdmissionRecord:
        """Settle a started admission (the reference's ``settle_session_input``: the composite fence,
        the worker retirement, the discarded-turn boundary, the compacted stored result, the terminal
        row, the steering transfers and the revision bump)."""
        if outcome not in ("completed", "interrupted", "rejected", "failed"):
            raise AuthorityRefusal("invalid_params")
        with self._refused():
            with self._store.tx() as conn:
                _epoch_check(conn, epoch)
                row = _admission_row(conn, admission_id)
                session = _session_row(conn, row["target_session_id"])
                if (row["status"] != "started" or row["owner_epoch"] != epoch
                        or type(generation) is not int or row["generation"] != generation
                        or session["runtime_generation"] != generation):
                    raise AuthorityRefusal("stale_generation")
                _retire_admission_workers(conn, row, epoch)
                if boundary is not None:
                    _close_discarded_turn(conn, row, boundary.notice, replace_after=boundary.replace_after,
                                          display_metadata=boundary.display_metadata)
                if result is not None:
                    encoded = canonical_json(encode_result(
                        compact_result(dict(result), user_message=json.loads(row["payload_json"]).get("text"))))
                    conn.execute("INSERT INTO state_meta (key, value) VALUES (%s, %s) "
                                 "ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value",
                                 (RESULT_PREFIX + admission_id, encoded))
                conn.execute("UPDATE session_admissions SET status = 'terminal', outcome = %s "
                             "WHERE admission_id = %s", (outcome, admission_id))
                _settle_steer_transfers(conn, epoch=epoch, generation=generation, transfers=transfers,
                                        outcome=outcome)
                conn.execute("UPDATE sessions SET runtime_revision = runtime_revision + 1 WHERE id = %s",
                             (row["target_session_id"],))
                return _record(_row(_admission_row(conn, admission_id)))

    def cancel(self, *, epoch: int, admission_id: str) -> AdmissionRecord:
        """Cancel a queued admission (the reference's ``cancel_session_input``: ``unknown`` refuses
        ``unknown_execution``; a started row or an in-flight steer refuses ``stale_generation``)."""
        with self._refused():
            with self._store.tx() as conn:
                _epoch_check(conn, epoch)
                row = _admission_row(conn, admission_id)
                if row["status"] == "unknown":
                    raise AuthorityRefusal("unknown_execution")
                # A steer in transfer reads as its running claim and settles with it: as started, not cancellable.
                if row["status"] == "started" or row["outcome"] == STEERING:
                    raise AuthorityRefusal("stale_generation")
                if row["status"] == "queued":
                    conn.execute(_CANCEL_QUEUED, (admission_id,))
                return _record(_row(_admission_row(conn, admission_id)))

    def steer(self, *, epoch: int, admission_id: str, generation: int) -> tuple[AdmissionRecord, bool]:
        """Hand a still-queued admission to its FIFO's running claim (the reference's
        ``steer_session_input``: queued -> terminal ``steering`` bound to ``generation``; a committed
        transfer replays ``(row, False)``)."""
        with self._refused():
            with self._store.tx() as conn:
                _epoch_check(conn, epoch)
                row = _admission_row(conn, admission_id)
                if row["status"] == "terminal" and row["outcome"] in (STEERING, "steered"):
                    return _record(_row(row)), False
                if row["status"] != "queued":
                    raise AuthorityRefusal("admission_started")
                if (type(generation) is not int or conn.execute(
                        "SELECT 1 FROM session_admissions WHERE target_session_id = %s AND status = 'started' "
                        "AND owner_epoch = %s AND generation = %s",
                        (row["target_session_id"], epoch, generation)).fetchone() is None):
                    raise AuthorityRefusal("session_idle")
                conn.execute("UPDATE session_admissions SET status = 'terminal', outcome = %s, "
                             "owner_epoch = %s, generation = %s WHERE admission_id = %s",
                             (STEERING, epoch, generation, admission_id))
                return _record(_row(_admission_row(conn, admission_id))), True

    def revert_steer(self, *, epoch: int, admission_id: str, generation: int) -> bool:
        """Queue a transfer whose text never reached the claim's agent again at its original ``seq``
        (the reference's ``revert_steer_input``); False when the claim's settlement already decided it."""
        with self._refused():
            with self._store.tx() as conn:
                _epoch_check(conn, epoch)
                return _requeue_transfer(conn, admission_id=admission_id, epoch=epoch, generation=generation)

    def mark_unknown(self, *, epoch: int, admission_id: str, generation: int,
                     transfers: Mapping[str, bool] | None = None) -> Literal["unknown", "terminal"]:
        """Mark this owner's started admission ``unknown`` after its settlement failed (the
        reference's ``_mark_unknown``: a terminal or unknown row keeps its status)."""
        with self._refused():
            with self._store.tx() as conn:
                _epoch_check(conn, epoch)
                current = _admission_row(conn, admission_id)
                if current["owner_epoch"] != epoch or current["generation"] != generation:
                    raise AuthorityRefusal("stale_generation")
                if current["status"] in {"terminal", "unknown"}:
                    return current["status"]
                if current["status"] != "started":
                    raise AuthorityRefusal("stale_generation")
                conn.execute("UPDATE session_admissions SET status = 'unknown' WHERE admission_id = %s",
                             (admission_id,))
                # A steer it carried is never replayed: interrupted if delivered, else queued again.
                _settle_steer_transfers(conn, epoch=epoch, generation=generation, transfers=transfers,
                                        outcome="unknown")
                conn.execute("UPDATE sessions SET runtime_revision = runtime_revision + 1 WHERE id = %s",
                             (current["target_session_id"],))
                return "unknown"

    def rehome_queued(self, *, epoch: int, old_session_id: str, new_session_id: str,
                     principal_prefix: str) -> list[str]:
        """Move the old owner's still-queued ``principal_prefix*`` admissions to ``new_session_id``
        (the reference's ``rehome_queued_inputs``: ids, request ids and ``seq`` kept, digests
        re-keyed to the new target, nothing moves behind an ``unknown`` row)."""
        for value in (old_session_id, new_session_id, principal_prefix):
            _text_check(value)
        with self._refused():
            with self._store.tx() as conn:
                _epoch_check(conn, epoch)
                _session_row(conn, new_session_id)
                if conn.execute("SELECT 1 FROM session_admissions WHERE target_session_id = %s "
                                "AND status = 'unknown'", (old_session_id,)).fetchone():
                    return []
                rows = conn.execute(
                    "SELECT * FROM session_admissions WHERE target_session_id = %s AND status = 'queued' "
                    "AND position(%s IN principal_id) = 1 ORDER BY seq",
                    (old_session_id, principal_prefix)).fetchall()
                for row in rows:
                    digest = admission_fingerprint(
                        canonical_target=new_session_id,
                        payload={"input": retry_input(json.loads(row["payload_json"])), "intent": row["intent"]})
                    conn.execute("UPDATE session_admissions SET target_session_id = %s, lineage_json = %s, "
                                 "payload_digest = %s WHERE admission_id = %s AND status = 'queued'",
                                 (new_session_id, json.dumps([new_session_id]), digest, row["admission_id"]))
                if rows:
                    for sid in (old_session_id, new_session_id):
                        conn.execute("UPDATE sessions SET runtime_revision = runtime_revision + 1 WHERE id = %s",
                                     (sid,))
                return [row["admission_id"] for row in rows]

    def retarget_local_automation(self, *, epoch: int, session_id: str, principal_id: str,
                                 old_targets: Iterable[str], new_target: str) -> list[str]:
        """Point the queued ``principal_id`` automation rows naming a pre-reset transcript at
        ``new_target`` (the reference's ``retarget_local_automation``: the descriptor's target
        retargeted and its digest re-keyed, nothing moves behind an ``unknown`` head)."""
        for value in (session_id, principal_id, new_target, *old_targets):
            _text_check(value)
        old_target_set = set(old_targets)
        with self._refused():
            with self._store.tx() as conn:
                _epoch_check(conn, epoch)
                _session_row(conn, new_target)
                if conn.execute("SELECT 1 FROM session_admissions WHERE target_session_id = %s "
                                "AND status = 'unknown'", (session_id,)).fetchone():
                    return []
                moved = []
                for row in conn.execute(
                        "SELECT * FROM session_admissions WHERE target_session_id = %s AND status = 'queued' "
                        "AND principal_id = %s ORDER BY seq", (session_id, principal_id)).fetchall():
                    payload = json.loads(row["payload_json"])
                    descriptor = payload.get("local_automation_v1") if isinstance(payload, dict) else None
                    if not isinstance(descriptor, dict) or descriptor.get("target") not in old_target_set:
                        continue
                    descriptor["target"] = new_target
                    digest = admission_fingerprint(
                        canonical_target=session_id,
                        payload={"input": retry_input(payload), "intent": row["intent"]})
                    conn.execute("UPDATE session_admissions SET payload_json = %s, payload_digest = %s "
                                 "WHERE admission_id = %s AND status = 'queued'",
                                 (canonical_json(payload), digest, row["admission_id"]))
                    moved.append(row["admission_id"])
                if moved:
                    conn.execute("UPDATE sessions SET runtime_revision = runtime_revision + 1 WHERE id = %s",
                                 (session_id,))
                return moved

    def recover(self, *, epoch: int) -> int:
        """Boot recovery (the reference's ``recover_session_inputs``): the stale child workers
        retire, the foreign-epoch live workers and in-flight steers settle, the foreign-epoch
        started admissions go ``unknown``; the count returned is the admissions marked."""
        with self._refused():
            with self._store.tx() as conn:
                _epoch_check(conn, epoch)
                _retire_stale_children(conn, epoch)
                conn.execute("UPDATE worker_executions SET status = 'unknown' "
                             "WHERE status IN ('registered','running') AND owner_epoch != %s", (epoch,))
                # A steer handed to a lost claim may or may not have reached its agent: never replayed.
                conn.execute("UPDATE session_admissions SET outcome = 'interrupted' WHERE status = 'terminal' "
                             "AND outcome = %s AND owner_epoch != %s", (STEERING, epoch))
                return conn.execute(
                    "UPDATE session_admissions SET status = 'unknown' WHERE status = 'started' "
                    "AND owner_epoch != %s", (epoch,)).rowcount

    def resolve_unknown(self, *, epoch: int, admission_id: str, generation: int, notice: str | None = None,
                       captured: CapturedResult | None = None) -> AdmissionRecord:
        """The operator's resolution of an ``unknown`` admission (the reference's
        ``resolve_unknown_session_input``: a terminal row at ``generation`` with an ``interrupted``
        outcome or a stored result replays; the captured result commits for this owner's row; the
        discarded-turn closer runs; the workers retire and the row goes terminal)."""
        with self._refused():
            with self._store.tx() as conn:
                _epoch_check(conn, epoch)
                row = _admission_row(conn, admission_id)
                if (type(generation) is int and row["generation"] == generation and row["status"] == "terminal"
                        and (row["outcome"] == "interrupted" or conn.execute(
                            "SELECT 1 FROM state_meta WHERE key = %s",
                            (RESULT_PREFIX + admission_id,)).fetchone())):
                    return _record(_row(row))
                if type(generation) is not int or row["status"] != "unknown" or row["generation"] != generation:
                    raise AuthorityRefusal("stale_generation")
                outcome = "interrupted"
                if captured is not None and row["owner_epoch"] == epoch:
                    if captured.outcome not in ("completed", "interrupted", "failed"):
                        raise AuthorityRefusal("invalid_params")
                    stored = canonical_json(encode_result(compact_result(
                        dict(captured.result), user_message=json.loads(row["payload_json"]).get("text"))))
                    conn.execute("INSERT INTO state_meta (key, value) VALUES (%s, %s) "
                                 "ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value",
                                 (RESULT_PREFIX + admission_id, stored))
                    outcome = captured.outcome
                # ``notice`` says the turn was interrupted: never beside a captured result this resolution commits.
                _close_discarded_turn(conn, row, None if (captured is not None and row["owner_epoch"] == epoch)
                                     else notice)
                _retire_admission_workers(conn, row, row["owner_epoch"])
                conn.execute("UPDATE session_admissions SET status = 'terminal', outcome = %s "
                             "WHERE admission_id = %s", (outcome, admission_id))
                conn.execute("UPDATE sessions SET runtime_revision = runtime_revision + 1 WHERE id = %s",
                             (row["target_session_id"],))
                return _record(_row(_admission_row(conn, admission_id)))

    def compact_api_payloads(self, *, admission_ids: Sequence[str]) -> int:
        """Each given row still matching the candidate predicate, re-read and rewritten by
        ``compacted_api_payload`` in one write (the reference's ``compact_api_payloads``); returns the
        rows written."""
        ids = list(admission_ids)
        if not ids:
            return 0
        with self._refused():
            with self._store.tx() as conn:
                compacted = 0
                for row_id in ids:
                    row = conn.execute(
                        f"SELECT payload_json, request_id FROM session_admissions "
                        f"WHERE {_API_CANDIDATE_WHERE} AND admission_id = %s", (row_id,)).fetchone()
                    if row is None:
                        continue
                    payload = compacted_api_payload(json.loads(row["payload_json"]), row["request_id"])
                    if payload is not None:
                        conn.execute("UPDATE session_admissions SET payload_json = %s WHERE admission_id = %s",
                                     (canonical_json(payload), row_id))
                        compacted += 1
                return compacted

    def unheld_media(self, references: Sequence[Mapping[str, Any]], *, after_id: int, retain_history: bool,
                     media_root: str) -> UnheldMedia:
        """The given references nothing holds (the reference's ``unheld_media`` under the writer: the
        held set is the live admissions' media, the API turns' images, the live hosted prompts'
        documents and the transcript mentions above ``after_id`` when ``retain_history``); a stored
        media reference without a path is a corrupt-store error, not a gate refusal."""
        refs = list(references)
        try:
            with self._refused():
                with self._store.tx() as conn:
                    held = _held_media_paths(conn, media_root)
                    if retain_history:
                        held |= _history_mentions(conn, refs, after_id=after_id)
                    candidates = [reference for reference in refs if reference["path"] not in held]
        except (KeyError, TypeError) as exc:  # a stored media reference without a path
            raise PermanentStoreError(f"admission media reference is corrupt: {exc!r}") from exc
        return UnheldMedia(candidates=tuple(candidates), held=frozenset(held))

    def record_retired_media(self, references: Sequence[Mapping[str, Any]]) -> None:
        """Record durable media retirement candidates keyed by path digest, first write wins (the
        reference's ``retire_media``: the reference dict stored under the path-digest key)."""
        refs = list(references)
        with self._refused():
            with self._store.tx() as conn:
                for reference in refs:
                    key = MEDIA_RETIRED_PREFIX + hashlib.sha256(reference["path"].encode()).hexdigest()
                    conn.execute("INSERT INTO state_meta (key, value) VALUES (%s, %s) "
                                 "ON CONFLICT (key) DO NOTHING", (key, json.dumps(reference)))

    def drop_retired_candidates(self, keys: Sequence[str]) -> None:
        """Delete collected candidate keys (the reference's ``drop_retired_candidates``)."""
        keys = list(keys)
        if not keys:
            return
        with self._refused():
            with self._store.tx() as conn:
                for start in range(0, len(keys), 500):
                    chunk = keys[start:start + 500]
                    placeholders = ",".join(["%s"] * len(chunk))
                    conn.execute(f"DELETE FROM state_meta WHERE key IN ({placeholders})", chunk)

    # ── reads ───────────────────────────────────────────────────────────────

    def find(self, *, principal_id: str, session_id: str, request_id: str) -> AdmissionRecord | None:
        """The live row at its exact (principal, target, request) identity (the UNIQUE key)."""
        with self._refused():
            row = self._store.one(
                "SELECT * FROM session_admissions WHERE principal_id = %s AND target_session_id = %s "
                "AND request_id = %s", principal_id, session_id, request_id)
        return _record(_row(row))

    def match(self, *, request_id: str | None = None, admission_id: str | None = None,
              principal_id: str | None = None, session_id: str | None = None,
              include_retired: bool = False) -> list[AdmissionRecord]:
        """Live rows matching every given filter in ``seq`` order, then (``include_retired`` with a
        request id) the retired API tombstones carrying that request id that match the other filters
        (the reference's ``match_admissions``)."""
        given = [(column, value) for column, value in (
            ("principal_id", principal_id), ("target_session_id", session_id),
            ("request_id", request_id), ("admission_id", admission_id)) if value is not None]
        where = " AND ".join(f"{column} = %s" for column, _ in given)
        with self._refused():
            rows = [
                _row(row) for row in self._store.all(
                    "SELECT * FROM session_admissions" + (f" WHERE {where}" if where else "") + " ORDER BY seq",
                    *[value for _, value in given])]
            if include_retired and request_id is not None:
                with self._store.tx() as conn:
                    rows.extend(_row(row) for row in _terminal_api_admissions(conn, request_id)
                                if all(row[column] == value for column, value in given))
        return [_record(row) for row in rows]

    def list(self, *, session_id: str, pending_only: bool = True) -> list[AdmissionRecord]:
        """The session's rows in ``seq`` order (the reference's ``list_session_admissions``: pending is
        the explicit status set, not a negation, so the index range bounds the scan)."""
        with self._refused():
            sql = ("SELECT * FROM session_admissions WHERE target_session_id = %s "
                   + ("AND status IN ('queued','started','unknown') " if pending_only else "")
                   + "ORDER BY seq")
            rows = self._store.all(sql, session_id)
        return [_record(_row(row)) for row in rows]

    def get(self, *, admission_id: str) -> AdmissionRecord | None:
        """The live row, else its terminal tombstone, else ``None`` (the reference's
        ``get_session_admission``)."""
        with self._refused():
            row = self._store.one("SELECT * FROM session_admissions WHERE admission_id = %s", admission_id)
            if row is None:
                with self._store.tx() as conn:
                    row = _terminal_admission_row(conn, admission_id)
        return _record(_row(row))

    def latest_admission(self, *, session_id: str, payload_marker: str) -> AdmissionRecord | None:
        """The session's newest row by ``seq`` whose decoded payload holds the top-level key
        ``payload_marker`` (the reference's ``latest_session_admission``: newest-first, stopping at
        the first match)."""
        with self._refused():
            rows = self._store.all(
                "SELECT * FROM session_admissions WHERE target_session_id = %s ORDER BY seq DESC", session_id)
            for row in rows:
                projected = _row(row)
                if isinstance(projected["payload"], dict) and payload_marker in projected["payload"]:
                    return _record(projected)
        return None

    def settled_result(self, *, admission_id: str, epoch: int | None = None,
                      generation: int | None = None) -> AdmissionResult | None:
        """The row and its stored result (the reference's ``settled_result``: without ``epoch`` the
        row unless terminal with a stored result; with ``epoch``/``generation`` the owner fence and
        the stored-result proof)."""
        if (epoch is None) != (generation is None):
            raise ValueError("settled_result takes epoch and generation together or not at all")
        with self._refused():
            if epoch is None:
                with self._store.tx() as conn:
                    row = _admission_row(conn, admission_id)
                    if row["status"] != "terminal":
                        return None
                    saved = conn.execute("SELECT value FROM state_meta WHERE key = %s",
                                        (RESULT_PREFIX + admission_id,)).fetchone()
                    found = (_row(row), load_result(saved["value"])) if saved else None
            else:
                with self._store.tx() as conn:
                    _epoch_check(conn, epoch)
                    current = _admission_row(conn, admission_id)
                    if current["owner_epoch"] != epoch or current["generation"] != generation:
                        raise AuthorityRefusal("stale_generation")
                    if current["status"] == "terminal":
                        saved = conn.execute("SELECT value FROM state_meta WHERE key = %s",
                                             (RESULT_PREFIX + admission_id,)).fetchone()
                        if saved is None:
                            raise AuthorityRefusal("storage_unavailable")
                        found = (_row(current), load_result(saved["value"]))
                    else:
                        found = (_row(current), None)
        return None if found is None else AdmissionResult(record=_record(found[0]), result=found[1])

    def retired_replay(self, *, epoch: int, principal_id: str, session_id: str, request_id: str,
                      payload: Mapping[str, Any],
                      intent: Literal["queue", "steer", "redirect"] = "queue",
                      _internal: bool = False) -> AdmissionRecord | None:
        """The reference's ``retry_terminal_admission``: ``None`` for a live target; on a retired
        target the tombstone of the exact retry, ``not_found`` for an unseen identity,
        ``admission_conflict`` (with :data:`ADMITTED_REQUEST_CONFLICT`) for another digest.

        ``_internal`` is the ``admit``'s pre-write call (the reference's ``admit_session_input``
        runs it before its own write); the public verb keeps the protocol signature."""
        for value in (principal_id, session_id, request_id):
            _text_check(value)
        digest = admission_fingerprint(canonical_target=session_id,
                                       payload={"input": retry_input(json.loads(canonical_json(payload))),
                                                "intent": intent})
        with self._refused():
            with self._store.tx() as conn:
                _epoch_check(conn, epoch)
                if conn.execute("SELECT 1 FROM state_meta WHERE key = %s",
                                (RETIRED_PREFIX + session_id,)).fetchone() is None:
                    return None
                saved = conn.execute("SELECT value FROM state_meta WHERE key = %s",
                                     (identity_key(principal_id, session_id, request_id),)).fetchone()
                if saved is None:
                    raise AuthorityRefusal("not_found")
                row = _terminal_admission_row(conn, json.loads(saved["value"]))
                if row["payload_digest"] != digest:
                    raise AuthorityRefusal("admission_conflict", **ADMITTED_REQUEST_CONFLICT)
                return _record(_row(row))

    def pending_targets(self, *, statuses: frozenset[str], principal_id: str | None = None) -> frozenset[str]:
        """The distinct targets holding an admission in ``statuses`` (and of ``principal_id`` when
        given) (the reference's ``pending_targets``)."""
        statuses = sorted(statuses)
        if not statuses:
            return frozenset()
        with self._refused():
            sql = ("SELECT DISTINCT target_session_id FROM session_admissions WHERE "
                   + ("principal_id = %s AND " if principal_id is not None else "")
                   + f"status IN ({','.join(['%s'] * len(statuses))})")
            params = ((principal_id,) if principal_id is not None else ()) + tuple(statuses)
            rows = self._store.all(sql, *params)
        return frozenset(row["target_session_id"] for row in rows)

    def list_unknown(self) -> list[AdmissionRecord]:
        """Every ``unknown`` admission in ``seq`` order (the reference's ``list_unknown``)."""
        with self._refused():
            rows = self._store.all(
                "SELECT * FROM session_admissions WHERE status = 'unknown' ORDER BY seq")
        return [_record(_row(row)) for row in rows]

    def has_pending(self, *, session_ids: Sequence[str] = (), statuses: frozenset[str] | None = None) -> bool:
        """Whether a row targets one of ``session_ids`` (any target when empty) with a status in
        ``statuses`` (any status when ``None``) (the reference's ``has_pending``)."""
        clauses, params = [], []
        if session_ids:
            clauses.append(f"target_session_id IN ({','.join(['%s'] * len(session_ids))})")
            params.extend(session_ids)
        if statuses is not None:
            statuses = sorted(statuses)
            if not statuses:
                return False
            clauses.append(f"status IN ({','.join(['%s'] * len(statuses))})")
            params.extend(statuses)
        with self._refused():
            row = self._store.scalar(
                "SELECT 1 FROM session_admissions" + (f" WHERE {' AND '.join(clauses)}" if clauses else "")
                + " LIMIT 1", *params)
        return row is not None

    def ledger_busy(self, session_id: str, target: str) -> bool:
        """A queued, started or unknown admission on ``session_id``, or a live worker on ``session_id``
        or ``target`` (the reference's ``_busy`` probe)."""
        with self._refused():
            row = self._store.one(
                "SELECT 1 FROM session_admissions WHERE target_session_id = %s "
                "AND status IN ('queued','started','unknown') "
                "UNION ALL SELECT 1 FROM worker_executions WHERE session_id IN (%s, %s) "
                "AND status IN ('registered','running','unknown') LIMIT 1",
                session_id, session_id, target)
        return row is not None

    def webhook_ingress_payloads(self, *, chat_id: str) -> list[Mapping[str, Any]]:
        """Payloads of the signed webhook ingress admissions for ``chat_id`` (the reference's
        ``webhook_ingress_payloads``: the ``native_text_v1.source`` names the chat on platform
        ``webhook`` and carries no ``automation``)."""
        with self._refused():
            rows = self._store.all(
                f"SELECT payload_json FROM session_admissions WHERE "
                f"({_JSONB}) #>> '{{native_text_v1,source,chat_id}}' = %s "
                f"AND ({_JSONB}) #>> '{{native_text_v1,source,platform}}' = 'webhook' "
                f"AND ((({_JSONB}) #>> '{{native_text_v1,automation}}') IS NULL OR ({_JSONB}) #>> '{{native_text_v1,automation}}' = 'null')", chat_id)
        return [json.loads(row["payload_json"]) for row in rows]

    def unfinalized_webhook_admissions(self) -> list[str]:
        """Ids of terminal webhook ingress admissions with a generation whose ``webhook`` session has
        not ended (the reference's ``unfinalized_webhook_admissions``)."""
        with self._refused():
            rows = self._store.all(
                "SELECT a.admission_id FROM sessions AS s "
                "JOIN session_admissions AS a ON a.target_session_id = s.id "
                "WHERE s.source = 'webhook' AND s.ended_at IS NULL "
                "AND a.status = 'terminal' AND a.generation IS NOT NULL "
                f"AND ({_JSONB_FOR('a')}) #>> '{{native_text_v1,source,platform}}' = 'webhook' "
                f"AND ((({_JSONB_FOR('a')}) #>> '{{native_text_v1,automation}}') IS NULL OR ({_JSONB_FOR('a')}) #>> '{{native_text_v1,automation}}' = 'null')")
        return [row["admission_id"] for row in rows]

    def terminal_ids(self, admission_ids: Sequence[str]) -> frozenset[str]:
        """The given ids that are terminal, live or tombstoned (the reference's ``terminal_ids``:
        500 ids a read, the tombstones' prefix-stripped keys)."""
        ids = [*admission_ids]
        terminal = []
        with self._refused():
            for start in range(0, len(ids), 500):
                chunk = ids[start:start + 500]
                placeholders = ",".join(["%s"] * len(chunk))
                terminal += [row["admission_id"] for row in self._store.all(
                    f"SELECT admission_id FROM session_admissions WHERE status = 'terminal' "
                    f"AND admission_id IN ({placeholders})", *chunk)]
                terminal += [row["key"][len(ADMISSION_PREFIX):] for row in self._store.all(
                    f"SELECT key FROM state_meta WHERE key IN ({placeholders})",
                    *(ADMISSION_PREFIX + admission_id for admission_id in chunk))]
        return frozenset(terminal)

    def api_compaction_candidates(self) -> list[str]:
        """The reference's ``api_compaction_candidates`` on the PG dialect: the terminal API
        admissions still carrying an inline image beside committed media, or a stored history
        outside the Responses API."""
        with self._refused():
            rows = self._store.all(
                f"SELECT admission_id FROM session_admissions WHERE {_API_CANDIDATE_WHERE}")
        return [row["admission_id"] for row in rows]

    def history_mentions(self, references: Sequence[Mapping[str, Any]]) -> tuple[int, frozenset[str]]:
        """``(watermark, mentioned)``: the highest ``messages.id`` and the reference paths a
        transcript row mentions by their ``<sha256>/<name>`` tail, raw or JSON-escaped (the
        reference's ``history_mentions``: one read snapshot so the watermark and the scan agree)."""
        refs = list(references)
        with self._refused():
            with self._store.tx() as conn:
                last_id = int(conn.execute("SELECT COALESCE(MAX(id), 0) AS mark FROM messages").fetchone()["mark"])
                seen = _history_mentions(conn, refs)
        return last_id, frozenset(seen)

    def settled_media(self) -> list[Sequence[Mapping[str, Any]]]:
        """The ``attachments_v1`` and ``native_text_v1`` media lists of terminal admissions, one list
        per row (the reference's ``settled_media``)."""
        with self._refused():
            with self._store.tx() as conn:
                return _settled_media(conn)

    def retired_candidates(self, kind: Literal["media", "spillover"]) -> list[tuple[str, Any]]:
        """``(key, decoded value)`` of the durable retirement candidates of ``kind`` in key order
        (the reference's ``retired_candidates``: the prefix range scan, the JSON values decoded)."""
        prefix = MEDIA_RETIRED_PREFIX if kind == "media" else SPILLOVER_RETIRED_PREFIX
        with self._refused():
            rows = self._store.all("SELECT key, value FROM state_meta WHERE key LIKE %s ORDER BY key", prefix + "%")
        return [(row["key"], json.loads(row["value"])) for row in rows]


_IMPLEMENTED = (
    "admit", "claim", "return_claim", "settle", "cancel", "steer", "revert_steer", "mark_unknown",
    "rehome_queued", "retarget_local_automation", "recover", "resolve_unknown", "compact_api_payloads",
    "unheld_media", "record_retired_media", "drop_retired_candidates",
    "find", "match", "list", "get", "latest_admission", "settled_result", "retired_replay",
    "pending_targets", "list_unknown", "has_pending", "ledger_busy", "webhook_ingress_payloads",
    "unfinalized_webhook_admissions", "terminal_ids", "api_compaction_candidates", "history_mentions",
    "settled_media", "retired_candidates",
)

# D8 inventory: every verb of the protocol is ported; the hosted member check inside ``admit`` is the
# module's single documented residual (it awaits the stage-4 hosted rooms port, see module docstring).
add_stubs(PgAdmissionsRepository, AdmissionRepository, implemented=_IMPLEMENTED)


def _register() -> None:
    # The ``durable_admissions`` capability is declared only once the gate is fully green; the hosted
    # admission check awaits the stage-4 hosted rooms port (its state does not exist on the PG schema
    # yet), so the flag stays off until that module lands.
    register_repository("admissions", lambda store: PgAdmissionsRepository(store))


_register()

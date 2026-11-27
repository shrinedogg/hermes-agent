"""``CoordinationRepository`` on PostgreSQL (stage 2): the compression lock, the turn lease, the hygiene
streak, session handoffs, the delegation ledger with its subagent records and the delivery-obligation
ledger of the bound profile's schema; each verb is one transaction of the store's single write
connection. The cross-process guarantees the SQLite reference got from its single writer rest on the
server here: one transaction per verb, the single-row compare-and-swap shapes and the ``SELECT``
confirms the reference makes, with contention surfacing as ``StoreContention`` through
``translated``.

The lease verbs resolve the conversation key and judge a holder's liveness with the conversation
store's bodies; the handoff settle check runs the runtime repository's idle check on the verb's own
connection; the subagent-record and delivery verbs are the SQLite ledger bodies ported onto the PG
dialect (``json_extract``/``json_set`` on ``record_json`` become guarded ``::jsonb`` reads and
``jsonb`` builds, the ``IS`` owner guards become ``IS NOT DISTINCT FROM``)."""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Mapping, Sequence
from typing import Any, Literal, Optional

from persistence import (
    AuthorityRefusal, DelegationId, JsonValue, PersistenceError, SessionId, StoreReplaced,
)
from persistence.authority.coordination import (
    COMPLETION_DELIVERIES, CoordinationRepository, DELEGATION_CLAIM_STALE_SECONDS,
    DELEGATION_FAILURE_WINDOW_SECONDS, DELEGATION_MAX_DELIVERY_ATTEMPTS, DELEGATION_MAX_DURABLE_PENDING,
    DELEGATION_MAX_RETAINED_TERMINAL, DELEGATION_ORPHAN_STALE_SECONDS, DELEGATION_REPLAY_MAX_AGE_SECONDS,
    DELEGATION_RETENTION_SECONDS, OBLIGATION_FLOOD_MARKER, OBLIGATION_MAX_ATTEMPTS, OBLIGATION_MAX_ROWS,
    OBLIGATION_RECONNECTED_MARKER, OBLIGATION_RETENTION_SECONDS, OBLIGATION_STALE_AFTER_SECONDS,
    TURN_LEASE_SWEEP_GRACE_SECONDS, ClaimedObligation, DelegationRecord, DelegationRecovery,
    DeliveryState, FailedObligationRow, FinishedDelegationRow, HandoffState, HandoffStatus,
    InflightDelegation, OrphanCompletion, OwnerStamp, PendingCompletion, PendingHandoff, RecordedOwner,
    RetryCandidate, RetryDue, flood_not_before, is_flood_error,
)
from persistence.authority.subagents import (
    AUTO_RESUME_CANCELLED, AUTO_RESUME_MAX_ATTEMPTS, GATEWAY_SHUTDOWN, SUBAGENT_EXPIRED_RETENTION_SECONDS,
    AutoRefusal, AutoResumeClaim, AutoStart, DelegationRoute, GaveUpCompletion, GiveUpOutcome,
    SubagentLimits, SubagentRecord, SubagentRecovery, SubagentState, SubagentWrite, parent_superseded_note,
)
from persistence.core.errors import HandoffUnsettled, SessionRowMissing
from persistence.sqlite._state.errors import SessionCompressionInProgressError, SessionTurnLeaseLostError
from persistence.sqlite.conversation._store.compression import _CHAIN_CAP
from process_identity import holder_process_is_dead

from .._backend import PostgresStore, register_repository, translated
from .._stubs import add_stubs
from ..conversation.sessions import (
    _CHAIN_STEP_SQL, _RESET_CHILD_SQL, _json_key_text, _session_turn_lease_key_on_conn,
)
from .runtime import _require_idle, _runtime_lineage

__all__ = ["PgCoordinationRepository"]

# The log records keep main's logger names: the store's and the delegation tool's.
logger = logging.getLogger("hermes_state")
_delegation_logger = logging.getLogger("tools.async_delegation")

# What the compression-lock writes log and answer as their empty value: a store failure (translated) or a torn-down
# backend. A replaced store is neither: ``StoreReplaced`` propagates so the caller re-acquires the store.
_LOCK_FAILURES = (PersistenceError, RuntimeError)

_HANDOFF_FAIL_SQL = "UPDATE sessions SET handoff_state = 'failed', handoff_error = %s WHERE "

# ── the resumable_subagents record (a TEXT column of valid JSON; the guarded jsonb read degrades to an
#    empty object, the DDL helpers' shape, so a write never throws on a malformed row) ──
_JR = "(CASE WHEN hermes_json_valid(record_json) THEN record_json::jsonb ELSE '{}'::jsonb END)"
# A promised record no give-up failure was delivered for: its parent is owed exactly one report.
_OWED_SQL = (
    f"COALESCE({_JR} -> 'auto_promised', 'false'::jsonb) IN ('true'::jsonb, '1'::jsonb) "
    f"AND COALESCE({_JR} -> 'auto_gave_up', 'null'::jsonb) = 'null'::jsonb")
# ``$.auto_claim`` set and not JSON null (the SQLite ``json_extract(...) IS NOT NULL`` / ``IS NULL``
# shapes: JSON null counts as absent).
_AUTO_CLAIM_SET = f"COALESCE({_JR} -> 'auto_claim', 'null'::jsonb) <> 'null'::jsonb"
_AUTO_CLAIM_ABSENT = f"COALESCE({_JR} -> 'auto_claim', 'null'::jsonb) = 'null'::jsonb"
# The ledger prune keeps the unit row a waiting child reports on: a silent unit is 'delivered', the first
# kind the count cap deletes, and the child's give-up failure is routed through that row.
_KEEP_WAITING_UNITS_SQL = (
    f" AND delegation_id NOT IN (SELECT {_JR} ->> 'unit_id' FROM resumable_subagents "
    f"WHERE state IN ('resumable','expired') AND {_OWED_SQL} AND {_JR} ->> 'unit_id' IS NOT NULL)")

_SUBAGENT_COLUMNS = ("subagent_id, state, pinned, stop_cause, owner_pid, owner_started_at, record_json, "
                     "created_at, updated_at, expires_at")
# A written run record never carries the columns or the marks of a claim no run has started yet.
_RUN_ONLY = ("state", "stop_cause", "auto_claim", "auto_cancel")
_CANCEL_STATES = (SubagentState.SUPERSEDED, SubagentState.STOPPED)


class _RouteMoved(Exception):
    """Rolls a give-up back when its unit row changed since the caller built the completion."""


def _require_settled(conn, session_id: str) -> None:
    """Refuse a handoff while durable work owns ``session_id``'s runtime lineage: a queued, started or unknown
    admission, a live worker, or a live turn lease or compression lock, checked on the caller's write."""
    try:
        _require_idle(conn, sorted(_runtime_lineage(conn, session_id)))
    except SessionRowMissing:
        return  # a session with no row owns no durable work: the reference settle check never meets an FK
    except (SessionTurnLeaseLostError, SessionCompressionInProgressError):
        raise HandoffUnsettled("session_busy") from None
    except AuthorityRefusal as exc:
        if exc.reason not in ("session_busy", "unknown_execution"):
            raise
        raise HandoffUnsettled(exc.reason) from None


def _claim_lease_row(conn, table: str, key_col: str, key: str, holder: str, now: float, expires_at: float,
                     stale) -> tuple[bool, Optional[str]]:
    """Single-transaction lease claim: DELETE a stale holder's row (``stale(holder, expires_at)``), INSERT ours
    (``ON CONFLICT DO NOTHING``: a concurrent claimant's row is the winner's, so the insert can only be a no-op),
    then SELECT to confirm ownership (the INSERT gives no rowcount signal). Returns
    ``(acquired, reclaimed_holder)``."""
    reclaimed_holder = None
    row = conn.execute(f"SELECT holder, expires_at FROM {table} WHERE {key_col} = %s", (key,)).fetchone()
    if row is not None and stale(row["holder"], row["expires_at"]):
        conn.execute(f"DELETE FROM {table} WHERE {key_col} = %s AND holder = %s", (key, row["holder"]))
        reclaimed_holder = row["holder"]
    conn.execute(
        f"INSERT INTO {table} ({key_col}, holder, acquired_at, expires_at) VALUES (%s, %s, %s, %s) "
        "ON CONFLICT DO NOTHING",
        (key, holder, now, expires_at))
    owner = conn.execute(f"SELECT holder FROM {table} WHERE {key_col} = %s", (key,)).fetchone()
    return owner is not None and owner["holder"] == holder, reclaimed_holder


def _turn_lease_key_read(store: PostgresStore, session_id: str) -> str:
    """The conversation's turn-lease key walked on the read door (the ``_session_turn_lease_key_on_conn``
    shape): the session's compression root, since the lease guards the whole conversation, never one segment."""
    current = session_id
    for _ in range(_CHAIN_CAP):
        row = store.one("SELECT parent_session_id FROM sessions WHERE id = %s", current)
        parent = row["parent_session_id"] if row is not None else None
        if not parent:
            break
        step = store.one(
            "SELECT 1 FROM sessions child JOIN sessions parent ON parent.id = child.parent_session_id "
            "WHERE child.id = %s AND parent.end_reason = 'compression' "
            f"AND COALESCE({_json_key_text('child.model_config', '_branched_from')}, '') <> child.parent_session_id "
            f"AND COALESCE({_json_key_text('child.model_config', '_delegate_from')}, '') <> child.parent_session_id "
            f"AND COALESCE({_json_key_text('child.model_config', '_reset_from')}, '') <> child.parent_session_id "
            "AND COALESCE(child.source, '') <> 'tool'",
            current)
        if step is None:
            break
        current = parent
    return current


def _superseded_reason_read(store: PostgresStore, parent_sid: str) -> Optional[str]:
    """Why the parent no longer wants its child resumed, from the live tip of its compression chain, walked on
    the read door (inside a write transaction nothing the parent does lands between this answer and the write
    acting on it)."""
    current = parent_sid
    chain = [current]
    seen = set(chain)
    for _ in range(_CHAIN_CAP):
        row = store.one(_CHAIN_STEP_SQL, current)
        child_id = row["id"] if row is not None else None
        if not child_id or child_id in seen:
            break
        seen.add(child_id)
        current = child_id
        chain.append(child_id)
    tip = store.one("SELECT ended_at, end_reason FROM sessions WHERE id = %s", chain[-1])
    if tip is None:
        return parent_superseded_note(None, tip_exists=False)
    return parent_superseded_note(tip["end_reason"] if tip["ended_at"] is not None else None, tip_exists=True)


# The in-memory offer of each returned completion stays with the caller.
def _replay_pending(conn, rows, now: float) -> tuple[PendingCompletion, ...]:
    """Each pending ``(delegation_id, event_json, completed_at, dispatched_at)`` row as a completion to offer, or
    terminally dropped past ``DELEGATION_REPLAY_MAX_AGE_SECONDS``."""
    offered = []
    for row in rows:
        delegation_id, payload, completed_at, dispatched_at = (
            row["delegation_id"], row["event_json"], row["completed_at"], row["dispatched_at"])
        age_basis = completed_at or dispatched_at
        if age_basis and (now - age_basis) > DELEGATION_REPLAY_MAX_AGE_SECONDS:
            conn.execute("""UPDATE async_delegations SET delivery_state='dropped',
                          delivery_claim=NULL, delivery_claimed_at=NULL,
                          updated_at=%s
                   WHERE delegation_id=%s AND delivery_state='pending'""", (now, delegation_id))
            _delegation_logger.warning("Async delegation %s: pending completion is %.1fh old "
                                       "(cap %.1fh); terminally dropping the replay (result remains queryable).",
                                       delegation_id, (now - age_basis) / 3600.0,
                                       DELEGATION_REPLAY_MAX_AGE_SECONDS / 3600.0)
            continue
        offered.append(PendingCompletion(delegation_id=DelegationId(delegation_id), event=json.loads(payload),
                                         completed_at=completed_at, dispatched_at=dispatched_at))
    return tuple(offered)


def _past_replay_age(now: float, completed_at: Optional[float], dispatched_at: Optional[float]) -> bool:
    age_basis = completed_at or dispatched_at
    return bool(age_basis) and (now - age_basis) > DELEGATION_REPLAY_MAX_AGE_SECONDS


# A JSON object column as a dict: ``{}`` when empty, undecodable or not an object.
def _json_object(raw: Optional[str]) -> dict[str, Any]:
    try:
        value = json.loads(raw) if raw else {}
    except ValueError:
        return {}
    return value if isinstance(value, dict) else {}


def _exhausted_warning(delegation_id: str) -> None:
    _delegation_logger.warning("Async delegation %s exhausted its %d delivery attempts; "
                               "marking terminally dropped (result remains queryable).",
                               delegation_id, DELEGATION_MAX_DELIVERY_ATTEMPTS)


# ── the delivery ledger's bodies (the SQLite ``_delivery_ledger`` port) ──────────────────────────────

def _prune_unlocked(conn, now: float) -> None:
    """Drops settled rows past retention, then the oldest rows over the row cap, settled ones first."""
    conn.execute(
        """DELETE FROM delivery_obligations
           WHERE state IN ('delivered', 'abandoned') AND updated_at < %s""", (now - OBLIGATION_RETENTION_SECONDS,))
    total = conn.execute("SELECT COUNT(*) FROM delivery_obligations").fetchone()["count"]
    if total > OBLIGATION_MAX_ROWS:
        conn.execute(
            """DELETE FROM delivery_obligations WHERE obligation_id IN (
                 SELECT obligation_id FROM delivery_obligations
                 ORDER BY CASE state
                            WHEN 'delivered' THEN 0
                            WHEN 'abandoned' THEN 1
                            ELSE 2
                          END, updated_at ASC
                 LIMIT %s)""", (total - OBLIGATION_MAX_ROWS,))


def _claimed_row(oid, session_key, platform, chat_id, thread_id, content, attempts, profile, *,
                 needs_marker: bool, runtime: bool = False, flood: bool = False,
                 last_error: Optional[str] = None) -> ClaimedObligation:
    """A claimed row for redelivery. A marked row names its own cause: ``flood`` gets the flood marker at boot or at
    runtime, a ``runtime`` reconnect replay the reconnect marker, and a boot-recovered crash the caller's restart
    marker default (``marker`` None). ``last_error`` is the row's pre-claim error."""
    marker = OBLIGATION_FLOOD_MARKER if flood else (OBLIGATION_RECONNECTED_MARKER if runtime else None)
    return ClaimedObligation(
        obligation_id=oid, session_key=session_key, platform=platform, chat_id=chat_id, thread_id=thread_id,
        content=content, needs_marker=needs_marker, marker=marker if needs_marker and marker else None,
        profile=profile, runtime_recovery=runtime, last_error=last_error or None, attempts=attempts + 1)


def _expected_profile(profile: Optional[str]) -> str:
    return "default" if not profile or profile == "default" else str(profile)


# ── the subagent record's bodies (the SQLite ``subagents`` port onto the PG dialect) ─────────────────

def _record(row) -> SubagentRecord:
    return SubagentRecord(subagent_id=row["subagent_id"], state=SubagentState(row["state"]),
                          pinned=bool(row["pinned"]), stop_cause=row["stop_cause"], owner_pid=row["owner_pid"],
                          owner_started_at=row["owner_started_at"],
                          document=json.loads(row["record_json"] or "{}"), created_at=row["created_at"],
                          updated_at=row["updated_at"], expires_at=row["expires_at"])


def _row(conn, sid: str) -> Optional[SubagentRecord]:
    got = conn.execute(f"SELECT {_SUBAGENT_COLUMNS} FROM resumable_subagents WHERE subagent_id = %s",
                       (sid,)).fetchone()
    return None if got is None else _record(got)


def _rows(conn, sql: str, params: Sequence[Any] = ()) -> tuple[SubagentRecord, ...]:
    return tuple(_record(row) for row in conn.execute(sql, list(params)).fetchall())


def _upsert(conn, sid: str, write: SubagentWrite, now: float) -> None:
    record = {k: v for k, v in write.document.items() if k not in _RUN_ONLY}
    conn.execute(
        """INSERT INTO resumable_subagents (subagent_id, state, pinned, stop_cause, owner_pid, owner_started_at,
               record_json, created_at, updated_at, expires_at)
           VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
           ON CONFLICT (subagent_id) DO UPDATE SET state=EXCLUDED.state, pinned=EXCLUDED.pinned,
               stop_cause=EXCLUDED.stop_cause, owner_pid=EXCLUDED.owner_pid,
               owner_started_at=EXCLUDED.owner_started_at, record_json=EXCLUDED.record_json,
               updated_at=EXCLUDED.updated_at, expires_at=EXCLUDED.expires_at""",
        (sid, SubagentState(write.state).value, int(write.pinned), write.stop_cause, write.owner_pid,
         write.owner_started_at, json.dumps(record, ensure_ascii=False, default=str), now, now,
         write.expires_at))


def _apply_limits(conn, limits: SubagentLimits, now: float) -> None:
    """TTL un-pins and retires a record; beyond the cap the oldest pins drop (the row stays continuable)."""
    conn.execute("UPDATE resumable_subagents SET pinned=0, state='expired' WHERE expires_at IS NOT NULL "
                 "AND expires_at <= %s AND state IN ('resumable','completed','stopped','superseded')", (now,))
    conn.execute(f"DELETE FROM resumable_subagents WHERE state='expired' AND updated_at < %s AND NOT ({_OWED_SQL})",
                 (now - SUBAGENT_EXPIRED_RETENTION_SECONDS - limits.ttl_hours * 3600,))
    conn.execute("""UPDATE resumable_subagents SET pinned=0 WHERE pinned=1 AND subagent_id NOT IN (
                        SELECT subagent_id FROM resumable_subagents WHERE pinned=1
                        ORDER BY updated_at DESC LIMIT %s)""", (max(limits.max_resumable, 0),))


def _note(conn, sid: str, *, state: Optional[SubagentState], note: str, now: float,
          from_states: Sequence[SubagentState] = (SubagentState.RESUMABLE,), unpin: bool = False,
          unstarted_claim: bool = False) -> bool:
    """Record why the scheduler will not (or no longer) run this record; *state* moves it when given (a cancel
    state also clears its debt). *unstarted_claim*: also a record an automatic continue claimed whose run has not
    started (``$.auto_claim``)."""
    if from_states:
        in_sql = f"state IN ({','.join(['%s'] * len(from_states))})"
    else:
        in_sql = "1=0"  # an empty from_states matches nothing (the SQLite ``IN ()`` is false; PG wants a literal)
    debt = ", 'auto_promised', 'false'::jsonb" if state in _CANCEL_STATES else ""
    sets = (f"record_json=(({_JR} || jsonb_build_object('auto_note', %s::text{debt})) - 'auto_claim'), updated_at=%s")
    if state:
        sets += ", state=%s" + (", pinned=0" if unpin else "")
    claimed = (f" OR (state='claimed' AND {_AUTO_CLAIM_SET})" if unstarted_claim else "")
    params = [note, now, *([SubagentState(state).value] if state else []), sid,
              *(SubagentState(s).value for s in from_states)]
    return conn.execute(
        f"UPDATE resumable_subagents SET {sets} WHERE subagent_id=%s AND ({in_sql}{claimed})",
        params).rowcount == 1


def _cancel_expired(conn, sid: str, note: str, now: float) -> bool:
    """A cancel that lands after the TTL moved an owed record to ``expired`` but before a tick reported it."""
    return conn.execute(
        f"UPDATE resumable_subagents SET record_json=({_JR} || jsonb_build_object('auto_note', %s::text, "
        f"'auto_promised', 'false'::jsonb)), updated_at=%s WHERE subagent_id=%s AND state='expired' AND {_OWED_SQL}",
        (note, now, sid)).rowcount == 1


def _cancel_manual_claim(conn, sid: str, *, state: SubagentState, note: str, now: float) -> bool:
    """A cancel that lands while the parent's own continue holds the row (``claimed``, no ``$.auto_claim``) before it
    ran: the decision rides on the row (``$.auto_cancel``) for the hand-back."""
    return conn.execute(
        f"UPDATE resumable_subagents SET record_json=({_JR} || jsonb_build_object('auto_cancel', "
        f"jsonb_build_object('state', %s::text, 'note', %s::text))), updated_at=%s WHERE subagent_id=%s AND state='claimed' "
        f"AND {_AUTO_CLAIM_ABSENT}",
        (SubagentState(state).value, note, now, sid)).rowcount == 1


def _superseded_reason(conn, parent_sid: str) -> Optional[str]:
    """Why the parent no longer wants its child resumed, from the live tip of its compression chain, read on *conn*
    (inside a write transaction nothing the parent does lands between this answer and the write acting on it)."""
    current = parent_sid
    chain = [current]
    seen = set(chain)
    for _ in range(_CHAIN_CAP):
        row = conn.execute(_CHAIN_STEP_SQL, (current,)).fetchone()
        child_id = row["id"] if row is not None else None
        if not child_id or child_id in seen:
            break
        seen.add(child_id)
        current = child_id
        chain.append(child_id)
    tip = conn.execute("SELECT ended_at, end_reason FROM sessions WHERE id = %s", (chain[-1],)).fetchone()
    if tip is None:
        return parent_superseded_note(None, tip_exists=False)
    return parent_superseded_note(tip["end_reason"] if tip["ended_at"] is not None else None, tip_exists=True)


def _unstarted_auto_claim(record: SubagentRecord) -> bool:
    return record.state == SubagentState.CLAIMED and bool(record.document.get("auto_claim"))


def _attempts(record: SubagentRecord) -> int:
    return int(record.document.get("auto_attempts") or 0)


def _refund(conn, sid: str) -> None:
    conn.execute(
        f"UPDATE resumable_subagents SET record_json=({_JR} || jsonb_build_object('auto_attempts', "
        f"GREATEST(COALESCE(({_JR} ->> 'auto_attempts')::numeric, 1) - 1, 0))) WHERE subagent_id=%s", (sid,))


def _insert_completion(conn, completion: GaveUpCompletion, now: float) -> None:
    """The give-up's failed completion on its own ledger row, pending, never replacing a row."""
    evt, result = completion.event, completion.result
    task = {k: evt.get(k) for k in ("goal", "goals", "context", "toolsets", "role", "model", "is_batch")}
    conn.execute(
        """INSERT INTO async_delegations (delegation_id, origin_session, origin_ui_session_id,
               parent_session_id, state, dispatched_at, completed_at, updated_at, event_json, result_json,
               delivery_state, delivery_attempts, owner_pid, owner_started_at, task_json, origin_session_id)
           VALUES (%s, %s, %s, %s, 'failed', %s, %s, %s, %s, %s, 'pending', 0, %s, %s, %s, %s)
           ON CONFLICT (delegation_id) DO NOTHING""",
        (evt["delegation_id"], evt.get("session_key") or "", evt.get("origin_ui_session_id") or "",
         evt.get("parent_session_id"), evt.get("dispatched_at") or now, evt.get("completed_at") or now, now,
         json.dumps(evt), json.dumps(result), completion.owner_pid, completion.owner_started_at,
         json.dumps(task), evt.get("origin_session_id") or ""))


def recover_subagents(conn, recoveries: Sequence[SubagentRecovery], limits: Optional[SubagentLimits],
                      now: float) -> None:
    """Boot recovery's subagent half, on the recovery's own transaction: each record still held by the same dead owner
    (``running`` or ``claimed``, read inside the write lock) is written as given, the limits applied after each."""
    for recovery in recoveries:
        got = conn.execute("SELECT state, owner_pid, owner_started_at FROM resumable_subagents WHERE subagent_id = %s",
                           (recovery.subagent_id,)).fetchone()
        if got is None or got["state"] not in ("running", "claimed") or (got["owner_pid"],
                                                                         got["owner_started_at"]) != (
                recovery.owner_pid, recovery.owner_started_at):
            continue
        _upsert(conn, recovery.subagent_id, recovery.write, now)
        if limits is not None:
            _apply_limits(conn, limits, now)


def _at(now: Optional[float]) -> float:
    return time.time() if now is None else now


class PgCoordinationRepository:
    """Compression lock, turn lease, hygiene streak, handoffs and the two ledgers of the bound profile's schema,
    the delegation ledger with its subagent records; the ledger verbs run one transaction of the store's write
    connection."""

    def __init__(self, store: PostgresStore) -> None:
        self._store = store

    def _ledger_write(self, fn) -> Any:
        """``fn(conn)`` in one transaction of the store's write connection (the ledgers' write door); the cross-process
        guarantee is the transaction's commit, a contender starting after the commit re-reads the committed state."""
        with translated():
            with self._store.tx() as conn:
                return fn(conn)

    def _ledger_read(self, read) -> Any:
        """``read(store)`` on the store's read door; an uninitialised store answers empty, as an absent SQLite file
        does, and a read never creates the store."""
        return read(self._store)

    def _subagent_write(self, fn, gone: Any = None) -> Any:
        """``fn(conn)`` in one ledger write (a profile's store is never removed here: the writer creates its schema,
        so nothing answers ``gone``)."""
        return self._ledger_write(fn)

    def _subagent_read(self, read, gone: Any = None) -> Any:
        return self._ledger_read(read)

    def _write_sql_logged(self, op: str, session_id: str, sql: str, params) -> None:
        """``_write_sql`` that logs a store failure instead of raising."""
        try:
            with translated():
                with self._store.tx() as conn:
                    conn.execute(sql, params)
        except StoreReplaced:
            raise
        except _LOCK_FAILURES as exc:
            logger.warning("%s(%s) failed: %s", op, session_id, exc)

    # ── compression lock ────────────────────────────────────────────────────

    def try_acquire_compression_lock(self, *, session_id: SessionId, holder: str, ttl_seconds: float = 300.0) -> bool:
        if not session_id:
            return False
        now = time.time()
        expires_at = now + ttl_seconds

        def _do(conn):
            return _claim_lease_row(
                conn, "compression_locks", "session_id", session_id, holder, now, expires_at,
                lambda h, e: e < now or holder_process_is_dead(h))

        try:
            with translated():
                with self._store.tx() as conn:
                    acquired, reclaimed_holder = _do(conn)
        except StoreReplaced:
            raise
        except _LOCK_FAILURES as exc:
            # False makes the caller skip compression: safe when the lock subsystem is broken.
            logger.warning("try_acquire_compression_lock(%s) failed: %s", session_id, exc)
            return False
        if reclaimed_holder:
            logger.warning("Reclaimed stale compression lock for session=%s (holder=%s)",
                           session_id, reclaimed_holder)
        return bool(acquired)

    def refresh_compression_lock(self, *, session_id: SessionId, holder: str, ttl_seconds: float = 300.0) -> bool:
        """Ownership is decided by ``holder`` alone, not ``expires_at``: a live owner whose refresher stalled past
        its TTL revives its still-unclaimed row; a reclaim cannot interleave because the write transaction
        serialises the claimants."""
        if not session_id or not holder:
            return False
        expires_at = time.time() + ttl_seconds
        try:
            with translated():
                with self._store.tx() as conn:
                    return conn.execute(
                        "UPDATE compression_locks SET expires_at = %s WHERE session_id = %s AND holder = %s",
                        (expires_at, session_id, holder)).rowcount > 0
        except StoreReplaced:
            raise
        except _LOCK_FAILURES as exc:
            logger.warning("refresh_compression_lock(%s) failed: %s", session_id, exc)
            return False

    def release_compression_lock(self, *, session_id: SessionId, holder: str) -> None:
        if not session_id:
            return
        self._write_sql_logged(
            "release_compression_lock", session_id,
            "DELETE FROM compression_locks WHERE session_id = %s AND holder = %s",
            (session_id, holder))

    def get_compression_lock_holder(self, *, session_id: SessionId) -> Optional[str]:
        if not session_id:
            return None
        with translated():
            return self._store.scalar(
                "SELECT holder FROM compression_locks WHERE session_id = %s AND expires_at >= %s",
                session_id, time.time())

    # ── turn lease ──────────────────────────────────────────────────────────

    def try_acquire_turn_lease(self, *, session_id: SessionId, holder: str, ttl_seconds: float = 300.0,
                              patience_s: Optional[float] = None) -> bool:
        # ``patience_s`` has no PG knob: the store's write connection has no lock-wait timeout, and a
        # single-row claim never blocks (a live claimant's row makes the insert a no-op); a lock wait that
        # did stall surfaces as ``StoreContention`` through ``translated``.
        if not session_id or not holder:
            return False
        now = time.time()
        expires_at = now + max(0.1, float(ttl_seconds))

        def _do(conn):
            # Sweep rows that expired long ago: a holder that died without releasing leaves its row behind, and
            # nothing else revisits a conversation nobody resumes. The grace keeps the recent expiries a live
            # owner can renew from a starved refresher.
            conn.execute("DELETE FROM session_turn_leases WHERE expires_at < %s",
                         (now - TURN_LEASE_SWEEP_GRACE_SECONDS,))
            conversation_id = _session_turn_lease_key_on_conn(conn, session_id)
            return _claim_lease_row(
                conn, "session_turn_leases", "conversation_id", conversation_id, holder, now, expires_at,
                lambda h, e: float(e) <= now or holder_process_is_dead(h),
            )[0]

        with translated():
            with self._store.tx() as conn:
                return bool(_do(conn))

    def refresh_turn_lease(self, *, session_id: SessionId, holder: str, ttl_seconds: float = 300.0,
                           patience_s: Optional[float] = None) -> bool:
        """``expires_at`` is stamped before the write, so a caller timestamp taken before this call never exceeds
        it."""
        if not session_id or not holder:
            return False
        expires_at = time.time() + max(0.1, float(ttl_seconds))

        def _do(conn):
            conversation_id = _session_turn_lease_key_on_conn(conn, session_id)
            return conn.execute(
                "UPDATE session_turn_leases SET expires_at = %s "
                "WHERE conversation_id = %s AND holder = %s", (expires_at, conversation_id, holder),
            ).rowcount > 0

        with translated():
            with self._store.tx() as conn:
                return bool(_do(conn))

    def release_turn_lease(self, *, session_id: SessionId, holder: str) -> None:
        if not session_id or not holder:
            return

        def _do(conn):
            conversation_id = _session_turn_lease_key_on_conn(conn, session_id)
            conn.execute(
                "DELETE FROM session_turn_leases WHERE conversation_id = %s AND holder = %s",
                (conversation_id, holder))

        with translated():
            with self._store.tx() as conn:
                _do(conn)

    def turn_lease_expires_at(self, *, session_id: SessionId, holder: str) -> Optional[float]:
        if not session_id or not holder:
            return None
        with translated():
            value = self._store.scalar(
                "SELECT expires_at FROM session_turn_leases WHERE conversation_id = %s AND holder = %s",
                _turn_lease_key_read(self._store, session_id), holder)
        return float(value) if value is not None else None

    # ── hygiene streak ──────────────────────────────────────────────────────

    def increment_hygiene_failure_streak(self, *, session_key: str) -> int:
        if not session_key:
            return 1

        def _do(conn):
            conn.execute(
                """INSERT INTO gateway_hygiene_state (session_key, failure_streak)
                   VALUES (%s, 1)
                   ON CONFLICT (session_key) DO UPDATE SET
                       failure_streak = gateway_hygiene_state.failure_streak + 1""",
                (session_key,))
            row = conn.execute(
                "SELECT failure_streak FROM gateway_hygiene_state WHERE session_key = %s", (session_key,))
            return int(row.fetchone()["failure_streak"])

        with translated():
            with self._store.tx() as conn:
                return _do(conn)

    def reset_hygiene_failure_streak(self, *, session_key: str) -> None:
        if not session_key:
            return
        with translated():
            with self._store.tx() as conn:
                conn.execute("DELETE FROM gateway_hygiene_state WHERE session_key = %s", (session_key,))

    # ── handoffs ────────────────────────────────────────────────────────────

    def request_handoff(self, *, session_id: SessionId, platform: str, require_settled: bool = False) -> bool:
        def _do(conn):
            if require_settled:
                _require_settled(conn, session_id)
            return conn.execute(
                "UPDATE sessions SET handoff_state = 'pending', handoff_platform = %s, handoff_error = NULL "
                "WHERE id = %s AND (handoff_state IS NULL OR handoff_state IN ('completed', 'failed'))",
                (platform, session_id)).rowcount

        with translated():
            with self._store.tx() as conn:
                return _do(conn) > 0

    def check_handoff_settled(self, *, session_id: SessionId) -> None:
        with translated():
            with self._store.tx() as conn:
                _require_settled(conn, session_id)

    def claim_handoff(self, *, session_id: SessionId) -> bool:
        with translated():
            with self._store.tx() as conn:
                return conn.execute(
                    "UPDATE sessions SET handoff_state = 'running' WHERE id = %s AND handoff_state = 'pending'",
                    (session_id,),
                ).rowcount > 0

    def complete_handoff(self, *, session_id: SessionId) -> None:
        with translated():
            with self._store.tx() as conn:
                conn.execute(
                    "UPDATE sessions SET handoff_state = 'completed', handoff_error = NULL WHERE id = %s",
                    (session_id,))

    def fail_handoff(self, *, session_id: SessionId, error: str,
                     only_states: Optional[tuple[HandoffStatus, ...]] = None) -> bool:
        """``only_states`` makes the write a compare-and-swap on ``handoff_state``: a waiter that gives up passes
        ``(pending,)``, since the watcher owns the terminal state of a row it claimed."""
        states = tuple(only_states) if only_states else ()
        sql = _HANDOFF_FAIL_SQL + "id = %s" + (
            f" AND handoff_state IN ({', '.join('%s' for _ in states)})" if states else "")
        with translated():
            with self._store.tx() as conn:
                return conn.execute(sql, (error[:500], session_id, *states)).rowcount > 0

    def reclaim_stale_running_handoffs(self, *, error: str) -> tuple[SessionId, ...]:
        """A ``running`` row at watcher startup belongs to a previous gateway that died mid-dispatch; it is failed,
        not re-queued, because the dead gateway may already have dispatched the turn."""
        def _do(conn):
            ids = [row["id"] for row in
                   conn.execute("SELECT id FROM sessions WHERE handoff_state = 'running'").fetchall()]
            if ids:
                conn.execute(_HANDOFF_FAIL_SQL + "handoff_state = 'running'", (error[:500],))
            return ids

        try:
            with translated():
                with self._store.tx() as conn:
                    ids = _do(conn) or []
        except Exception:
            # Swallow but never silently: a persistently failing reclaim leaves poisonous 'running' rows.
            logger.warning(
                "reclaim_stale_running_handoffs failed; stranded 'running' "
                "handoff rows (if any) were left in place", exc_info=True)
            return ()
        return tuple(SessionId(sid) for sid in ids)

    def get_handoff(self, *, session_id: SessionId) -> Optional[HandoffState]:
        try:
            row = self._store.one(
                "SELECT handoff_state, handoff_platform, handoff_error FROM sessions WHERE id = %s",
                session_id)
            if not row or row["handoff_state"] is None:
                return None
            return HandoffState(state=HandoffStatus(row["handoff_state"]), platform=row["handoff_platform"],
                                error=row["handoff_error"])
        except Exception:  # health: allow BLE001 -- handoff state is advisory; a failed read answers no handoff
            return None

    def list_pending_handoffs(self) -> tuple[PendingHandoff, ...]:
        try:
            rows = self._store.all(
                "SELECT id, handoff_platform, title, started_at FROM sessions "
                "WHERE handoff_state = 'pending' ORDER BY started_at ASC")
            return tuple(PendingHandoff(session_id=SessionId(row["id"]), platform=row["handoff_platform"],
                                       title=row["title"], started_at=float(row["started_at"])) for row in rows)
        except Exception:  # health: allow BLE001 -- the watcher polls again; a failed read answers none pending
            return ()

    def has_pending_handoffs(self) -> bool:
        with translated():
            return self._store.scalar(
                "SELECT 1 FROM sessions WHERE handoff_state = 'pending' LIMIT 1") is not None

    # ── delegation ledger ───────────────────────────────────────────────────

    # A dispatch replaces any earlier row of its id: running, its delivery pending, no attempts spent.
    def record_delegation_dispatch(self, *, delegation_id: DelegationId, origin_session: str,
                                   origin_ui_session_id: str, parent_session_id: Optional[SessionId],
                                   dispatched_at: float, owner: OwnerStamp, task: JsonValue,
                                   origin_session_id: str = '') -> None:
        now = time.time()

        def _do(conn):
            conn.execute("""INSERT INTO async_delegations
                   (delegation_id, origin_session, origin_ui_session_id,
                    parent_session_id, state, dispatched_at, updated_at,
                    delivery_state, delivery_attempts, owner_pid,
                    owner_started_at, task_json, origin_session_id)
                   VALUES (%s, %s, %s, %s, 'running', %s, %s, 'pending', 0, %s, %s, %s, %s)
                   ON CONFLICT (delegation_id) DO UPDATE SET
                     origin_session = EXCLUDED.origin_session,
                     origin_ui_session_id = EXCLUDED.origin_ui_session_id,
                     parent_session_id = EXCLUDED.parent_session_id,
                     state = EXCLUDED.state,
                     dispatched_at = EXCLUDED.dispatched_at,
                     updated_at = EXCLUDED.updated_at,
                     delivery_state = EXCLUDED.delivery_state,
                     delivery_attempts = EXCLUDED.delivery_attempts,
                     owner_pid = EXCLUDED.owner_pid,
                     owner_started_at = EXCLUDED.owner_started_at,
                     task_json = EXCLUDED.task_json,
                     origin_session_id = EXCLUDED.origin_session_id""",
                (delegation_id, origin_session, origin_ui_session_id,
                 parent_session_id, dispatched_at, now, owner.pid, owner.started_at,
                 json.dumps(task), origin_session_id))

        self._ledger_write(_do)

    def prune_delegation_ledger(self) -> None:
        """Bound terminal history, preferring delivered records for deletion. A unit a waiting child reports on is
        kept out of the retention and the terminal cap (``_KEEP_WAITING_UNITS_SQL``), not the pending cap."""
        cutoff = time.time() - DELEGATION_RETENTION_SECONDS
        keep = _KEEP_WAITING_UNITS_SQL

        def _do(conn):
            conn.execute(
                "DELETE FROM async_delegations WHERE delivery_state='delivered' AND updated_at < %s" + keep,
                (cutoff,))
            terminal_count = conn.execute(
                "SELECT COUNT(*) FROM async_delegations WHERE state != 'running'" + keep).fetchone()["count"]
            if terminal_count > DELEGATION_MAX_RETAINED_TERMINAL:
                conn.execute(f"""DELETE FROM async_delegations WHERE delegation_id IN (
                         SELECT delegation_id FROM async_delegations
                         WHERE state != 'running'{keep}
                         ORDER BY CASE delivery_state WHEN 'delivered' THEN 0 ELSE 1 END,
                                  updated_at ASC LIMIT %s
                       )""", (terminal_count - DELEGATION_MAX_RETAINED_TERMINAL,))
            pending_count = conn.execute("""SELECT COUNT(*) FROM async_delegations
                   WHERE state != 'running' AND delivery_state='pending'""").fetchone()["count"]
            if pending_count > DELEGATION_MAX_DURABLE_PENDING:
                conn.execute("""DELETE FROM async_delegations WHERE delegation_id IN (
                         SELECT delegation_id FROM async_delegations
                         WHERE state != 'running' AND delivery_state='pending'
                         ORDER BY updated_at ASC LIMIT %s
                       )""", (pending_count - DELEGATION_MAX_DURABLE_PENDING,))

        self._ledger_write(_do)

    # The finished state, event and result; a ``pending`` completion then waits for delivery, a ``delivered`` one
    # (every child of the unit waits on a promised automatic resume) is recorded but never offered.
    def record_delegation_completion(self, *, delegation_id: DelegationId, state: str, completed_at: float,
                                     event: JsonValue, result: JsonValue,
                                     delivery: Literal["pending", "delivered"] = "pending") -> None:
        if delivery not in COMPLETION_DELIVERIES:
            raise AuthorityRefusal("invalid_params", field="delivery")
        now = time.time()

        def _do(conn):
            conn.execute("""UPDATE async_delegations SET state=%s, completed_at=%s, updated_at=%s,
                   event_json=%s, result_json=%s, delivery_state=%s
                   WHERE delegation_id=%s""",
                (state, completed_at, now,
                 json.dumps(event), json.dumps(result), delivery, delegation_id))

        self._ledger_write(_do)

    def record_unit_child(self, *, delegation_id: DelegationId, entry: JsonValue) -> None:
        """Merge one finished child of a still-running unit into the stored partial by ``task_index``; the caller
        redacts ``entry``, and the stored partial holds only entries redacted that way."""

        def _do(conn):
            row = conn.execute("SELECT result_json FROM async_delegations WHERE delegation_id = %s "
                               "AND state = 'running'", (delegation_id,)).fetchone()
            if row is None:
                return
            partial = json.loads(row["result_json"] or "{}") or {}
            results = [r for r in partial.get("results") or []
                       if r.get("task_index") != entry.get("task_index")]
            results.append(entry)
            conn.execute("UPDATE async_delegations SET result_json=%s, updated_at=%s "
                         "WHERE delegation_id=%s AND state='running'",
                         (json.dumps({"results": results, "partial": True}), time.time(), delegation_id))

        self._ledger_write(_do)

    # Each recovery lands only while its row keeps the state and ``updated_at`` the caller read, after the subagent
    # records still held by their dead owners; the caller probes owners and builds each record and event outside the
    # transaction.
    def recover_abandoned_delegations(self, *, recoveries: Sequence[DelegationRecovery],
                                      subagent_recoveries: Sequence[SubagentRecovery] = (),
                                      limits: Optional[SubagentLimits] = None) -> int:
        now = time.time()

        def _do(conn):
            recover_subagents(conn, subagent_recoveries, limits, now)
            recovered = 0
            for recovery in recoveries:
                cursor = conn.execute("""UPDATE async_delegations SET state='unknown', completed_at=%s,
                       updated_at=%s, event_json=%s, result_json=%s, delivery_state=%s
                       WHERE delegation_id=%s AND state=%s AND updated_at=%s""",
                    (recovery.completed_at, now, json.dumps(recovery.event), json.dumps(recovery.result),
                     recovery.delivery, recovery.delegation_id, recovery.expected_state,
                     recovery.expected_updated_at))
                recovered += cursor.rowcount
            return recovered

        return self._ledger_write(_do)

    # Every finished, undelivered completion to replay, oldest first; one past the replay age cap is dropped.
    def take_replayable_completions(self) -> tuple[PendingCompletion, ...]:
        now = time.time()

        def _do(conn):
            rows = conn.execute("""SELECT delegation_id, event_json, completed_at, dispatched_at
                   FROM async_delegations
                   WHERE state != 'running' AND delivery_state='pending' AND event_json IS NOT NULL
                   ORDER BY completed_at, delegation_id""").fetchall()
            return _replay_pending(conn, rows, now)

        return self._ledger_write(_do)

    # The owner probe and the caller's lost copies arrive as ``dead_owners`` and ``lost_ids``; the offer runs in
    # the caller after the commit.
    def sweep_orphaned_completions(self, *, now: float, exclude_ids: frozenset[DelegationId] = frozenset(),
                                   dead_owners: frozenset[RecordedOwner],
                                   lost_ids: frozenset[DelegationId] = frozenset()) -> tuple[PendingCompletion, ...]:
        def _do(conn):
            rows = conn.execute("""SELECT delegation_id, event_json, completed_at, dispatched_at,
                          owner_pid, owner_started_at, delivery_attempts
                   FROM async_delegations
                   WHERE state != 'running' AND delivery_state='pending'
                     AND event_json IS NOT NULL AND updated_at < %s
                     AND (delivery_claim IS NULL OR delivery_claimed_at < %s)
                   ORDER BY completed_at, delegation_id""",
                (now - DELEGATION_ORPHAN_STALE_SECONDS, now - DELEGATION_CLAIM_STALE_SECONDS)).fetchall()
            orphans = []
            for row in rows:
                delegation_id, pid, started, attempts = (
                    row["delegation_id"], row["owner_pid"], row["owner_started_at"], row["delivery_attempts"])
                if delegation_id not in lost_ids and (
                        delegation_id in exclude_ids
                        or RecordedOwner(pid=pid, started_at=started) not in dead_owners):
                    continue
                if (attempts or 0) >= DELEGATION_MAX_DELIVERY_ATTEMPTS:
                    # Its last claimant died holding the final attempt; converge like release_delivery.
                    conn.execute("""UPDATE async_delegations SET delivery_state='dropped',
                                  delivery_claim=NULL, delivery_claimed_at=NULL, updated_at=%s
                           WHERE delegation_id=%s AND delivery_state='pending'""", (now, delegation_id))
                    _exhausted_warning(delegation_id)
                    continue
                orphans.append(row)
            return _replay_pending(conn, orphans, now)

        return self._ledger_write(_do)

    def _update_delivery(self, sql: str, params: tuple) -> bool:
        """Run one UPDATE on the ledger; True iff exactly one row changed."""
        def _do(conn):
            return conn.execute(sql, params).rowcount == 1

        return self._ledger_write(_do)

    def mark_delegation_delivered(self, *, delegation_id: DelegationId) -> bool:
        now = time.time()
        return self._update_delivery(
            """UPDATE async_delegations SET delivery_state='delivered', delivered_at=%s, updated_at=%s
               WHERE delegation_id=%s AND delivery_state != 'delivered'""", (now, now, delegation_id))

    # Claim a pending completion for delivery (a stale claim is taken over); each claim spends one attempt.
    def claim_delivery(self, *, delegation_id: DelegationId, claim_id: str) -> bool:
        now = time.time()

        def _do(conn):
            row = conn.execute(
                "SELECT delivery_state FROM async_delegations WHERE delegation_id = %s",
                (delegation_id,)).fetchone()
            if row is None:
                return True  # legacy event created before durable dispatch
            cur = conn.execute("""UPDATE async_delegations SET delivery_claim=%s, delivery_claimed_at=%s,
                          delivery_attempts=delivery_attempts+1, updated_at=%s
                   WHERE delegation_id=%s AND delivery_state='pending'
                     AND (delivery_claim IS NULL OR delivery_claimed_at < %s)""",
                (claim_id, now, now, delegation_id, now - DELEGATION_CLAIM_STALE_SECONDS))
            return cur.rowcount == 1

        return self._ledger_write(_do)

    def release_delivery(self, *, delegation_id: DelegationId, claim_id: str) -> bool:
        """Attempts are counted at claim time; once the budget is spent the row converges to terminal ``dropped``."""
        now = time.time()

        def _do(conn):
            capped = conn.execute("""UPDATE async_delegations SET delivery_state='dropped',
                          delivery_claim=NULL, delivery_claimed_at=NULL, updated_at=%s
                   WHERE delegation_id=%s AND delivery_state='pending'
                     AND delivery_claim=%s AND delivery_attempts >= %s""",
                (now, delegation_id, claim_id, DELEGATION_MAX_DELIVERY_ATTEMPTS))
            if capped.rowcount == 1:
                _exhausted_warning(delegation_id)
                return True
            cur = conn.execute("""UPDATE async_delegations SET delivery_claim=NULL,
                          delivery_claimed_at=NULL, updated_at=%s
                   WHERE delegation_id=%s AND delivery_state='pending'
                     AND delivery_claim=%s""", (now, delegation_id, claim_id))
            return cur.rowcount == 1

        return self._ledger_write(_do)

    # Give the claim back and refund the attempt it spent.
    def defer_delivery(self, *, delegation_id: DelegationId, claim_id: str) -> bool:
        return self._update_delivery("""UPDATE async_delegations SET delivery_claim=NULL,
                      delivery_claimed_at=NULL, delivery_attempts=GREATEST(COALESCE(delivery_attempts, 0) - 1, 0),
                      updated_at=%s
               WHERE delegation_id=%s AND delivery_state='pending' AND delivery_claim=%s""",
            (time.time(), delegation_id, claim_id))

    def drop_delivery(self, *, delegation_id: DelegationId, claim_id: str) -> bool:
        return self._update_delivery("""UPDATE async_delegations SET delivery_state='dropped',
                      updated_at=%s, delivery_claim=NULL,
                      delivery_claimed_at=NULL
               WHERE delegation_id=%s AND delivery_state='pending'
                 AND delivery_claim=%s""", (time.time(), delegation_id, claim_id))

    def complete_delivery(self, *, delegation_id: DelegationId, claim_id: str) -> bool:
        now = time.time()
        return self._update_delivery("""UPDATE async_delegations SET delivery_state='delivered',
                      delivered_at=%s, updated_at=%s, delivery_claim=NULL,
                      delivery_claimed_at=NULL
               WHERE delegation_id=%s AND delivery_state='pending'
                 AND delivery_claim=%s""", (now, now, delegation_id, claim_id))

    # A delivered completion goes back to pending (dropped once its attempts are spent); the caller offers the
    # returned completion after the commit.
    def reopen_admitted_completion(self, *, delegation_id: DelegationId) -> Optional[PendingCompletion]:
        now = time.time()

        def _do(conn):
            exhausted = conn.execute("""UPDATE async_delegations SET delivery_state='dropped', updated_at=%s
                   WHERE delegation_id=%s AND delivery_state='delivered' AND delivery_attempts >= %s""",
                (now, delegation_id, DELEGATION_MAX_DELIVERY_ATTEMPTS))
            if exhausted.rowcount == 1:
                _exhausted_warning(delegation_id)
                return None
            reopened = conn.execute("""UPDATE async_delegations SET delivery_state='pending',
                          delivered_at=NULL, updated_at=%s
                   WHERE delegation_id=%s AND delivery_state='delivered' AND event_json IS NOT NULL""",
                (now, delegation_id))
            if reopened.rowcount != 1:
                return None
            rows = conn.execute("""SELECT delegation_id, event_json, completed_at, dispatched_at
                   FROM async_delegations WHERE delegation_id=%s""", (delegation_id,)).fetchall()
            offered = _replay_pending(conn, rows, now)
            return offered[0] if offered else None

        return self._ledger_write(_do)

    # The cleanup of a dispatch whose submit was never scheduled.
    def delete_unscheduled_delegation(self, *, delegation_id: DelegationId) -> None:
        self._ledger_write(
            lambda conn: conn.execute("DELETE FROM async_delegations WHERE delegation_id = %s",
                                     (delegation_id,)))

    # The caller redacts ``result``.
    def get_delegation(self, *, delegation_id: DelegationId) -> Optional[DelegationRecord]:
        def _read(store):
            return store.one("""SELECT origin_session, state, dispatched_at, completed_at,
                          result_json, delivery_state, delivery_attempts,
                          origin_session_id
                   FROM async_delegations WHERE delegation_id = %s""", delegation_id)

        row = self._ledger_read(_read)
        return None if row is None else DelegationRecord(
            delegation_id=delegation_id, origin_session=row["origin_session"], state=row["state"],
            dispatched_at=row["dispatched_at"], completed_at=row["completed_at"],
            result=json.loads(row["result_json"]) if row["result_json"] else None,
            delivery_state=DeliveryState(row["delivery_state"]), delivery_attempts=row["delivery_attempts"],
            origin_session_id=row["origin_session_id"] or "")

    # A session's delegations finished inside the failure window, newest first; the caller expands failed
    # tasks.
    def list_finished_delegations(self, *, origin_ui_session_id: str = '', parent_session_id: str = '',
                                  limit: int = 20, now: Optional[float] = None) -> tuple[FinishedDelegationRow, ...]:
        selectors = [(col, val) for col, val in (
            ("origin_ui_session_id", origin_ui_session_id), ("parent_session_id", parent_session_id)) if val]
        if not selectors:
            return ()
        cutoff = (now if now is not None else time.time()) - DELEGATION_FAILURE_WINDOW_SECONDS
        owner_sql = " OR ".join(f"{col} = %s" for col, _ in selectors)

        def _read(store):
            return store.all(
                f"""SELECT delegation_id, state, dispatched_at, completed_at, task_json, result_json
                    FROM async_delegations
                    WHERE ({owner_sql}) AND state != 'running' AND completed_at >= %s
                    ORDER BY completed_at DESC LIMIT %s""",
                *([val for _, val in selectors] + [cutoff, limit]))

        rows = self._ledger_read(_read)
        return tuple(FinishedDelegationRow(
            delegation_id=DelegationId(row["delegation_id"]), state=row["state"],
            dispatched_at=row["dispatched_at"], completed_at=row["completed_at"],
            task=_json_object(row["task_json"]), result=_json_object(row["result_json"])) for row in rows)

    # The candidates of ``recover_abandoned_delegations``, with the ``updated_at`` its conditional update
    # compares.
    def list_inflight_delegations(self) -> tuple[InflightDelegation, ...]:
        def _read(store):
            return store.all("""SELECT delegation_id, origin_session, origin_ui_session_id,
                          parent_session_id, dispatched_at, owner_pid,
                          owner_started_at, task_json, origin_session_id, result_json, state, updated_at
                   FROM async_delegations WHERE state='running'""")

        rows = self._ledger_read(_read)
        return tuple(InflightDelegation(
            delegation_id=DelegationId(row["delegation_id"]), origin_session=row["origin_session"],
            origin_ui_session_id=row["origin_ui_session_id"], parent_session_id=row["parent_session_id"],
            dispatched_at=row["dispatched_at"], owner_pid=row["owner_pid"],
            owner_started_at=row["owner_started_at"], task=json.loads(row["task_json"] or "{}"),
            origin_session_id=row["origin_session_id"] or "",
            result=json.loads(row["result_json"]) if row["result_json"] else None, state=row["state"],
            updated_at=row["updated_at"]) for row in rows)

    def has_replayable_completions(self) -> bool:
        def _read(store):
            return store.scalar("""SELECT 1 FROM async_delegations
                   WHERE state != 'running' AND delivery_state='pending' AND event_json IS NOT NULL
                   LIMIT 1""")

        return self._ledger_read(_read) is not None

    # The owner stamps of the completions ``sweep_orphaned_completions`` would consider.
    def list_orphan_completion_owners(self, *, now: float) -> frozenset[RecordedOwner]:
        def _read(store):
            return store.all("""SELECT DISTINCT owner_pid, owner_started_at
                   FROM async_delegations
                   WHERE state != 'running' AND delivery_state='pending'
                     AND event_json IS NOT NULL AND updated_at < %s
                     AND (delivery_claim IS NULL OR delivery_claimed_at < %s)""",
                now - DELEGATION_ORPHAN_STALE_SECONDS, now - DELEGATION_CLAIM_STALE_SECONDS)

        rows = self._ledger_read(_read)
        return frozenset(RecordedOwner(pid=row["owner_pid"], started_at=row["owner_started_at"]) for row in rows)

    def list_orphan_completions(self, *, now: float) -> tuple[OrphanCompletion, ...]:
        def _read(store):
            return store.all("""SELECT delegation_id, owner_pid, owner_started_at, updated_at
                   FROM async_delegations
                   WHERE state != 'running' AND delivery_state='pending'
                     AND event_json IS NOT NULL AND updated_at < %s
                     AND (delivery_claim IS NULL OR delivery_claimed_at < %s)
                   ORDER BY completed_at, delegation_id""",
                now - DELEGATION_ORPHAN_STALE_SECONDS, now - DELEGATION_CLAIM_STALE_SECONDS)

        rows = self._ledger_read(_read)
        return tuple(OrphanCompletion(delegation_id=DelegationId(row["delegation_id"]),
                                      owner=RecordedOwner(pid=row["owner_pid"],
                                                         started_at=row["owner_started_at"]),
                                      updated_at=row["updated_at"]) for row in rows)

    # The completions ``sweep_orphaned_completions`` returns for the same arguments, read before its drops commit.
    def list_orphan_completion_offers(self, *, now: float, exclude_ids: frozenset[DelegationId] = frozenset(),
                                      dead_owners: frozenset[RecordedOwner],
                                      lost_ids: frozenset[DelegationId] = frozenset()) -> tuple[PendingCompletion, ...]:
        def _read(store):
            return store.all("""SELECT delegation_id, event_json, completed_at, dispatched_at,
                          owner_pid, owner_started_at, delivery_attempts
                   FROM async_delegations
                   WHERE state != 'running' AND delivery_state='pending'
                     AND event_json IS NOT NULL AND updated_at < %s
                     AND (delivery_claim IS NULL OR delivery_claimed_at < %s)
                   ORDER BY completed_at, delegation_id""",
                now - DELEGATION_ORPHAN_STALE_SECONDS, now - DELEGATION_CLAIM_STALE_SECONDS)

        rows = self._ledger_read(_read)
        offered = []
        for row in rows:
            delegation_id, payload, completed_at, dispatched_at = (
                row["delegation_id"], row["event_json"], row["completed_at"], row["dispatched_at"])
            owner = RecordedOwner(pid=row["owner_pid"], started_at=row["owner_started_at"])
            attempts = row["delivery_attempts"]
            if not ((delegation_id in lost_ids or (delegation_id not in exclude_ids
                                                   and owner in dead_owners))
                    and (attempts or 0) < DELEGATION_MAX_DELIVERY_ATTEMPTS
                    and not _past_replay_age(now, completed_at, dispatched_at)):
                continue
            offered.append(PendingCompletion(delegation_id=DelegationId(delegation_id), event=json.loads(payload),
                                             completed_at=completed_at, dispatched_at=dispatched_at))
        return tuple(offered)

    # The completion ``reopen_admitted_completion`` returns, read before the reopen commits.
    def get_admitted_completion(self, *, delegation_id: DelegationId) -> Optional[PendingCompletion]:
        def _read(store):
            return store.one("""SELECT event_json, completed_at, dispatched_at, delivery_attempts
                   FROM async_delegations
                   WHERE delegation_id = %s AND delivery_state='delivered' AND event_json IS NOT NULL""",
                   delegation_id)

        row = self._ledger_read(_read)
        if (row is None or row["delivery_attempts"] >= DELEGATION_MAX_DELIVERY_ATTEMPTS
                or _past_replay_age(time.time(), row["completed_at"], row["dispatched_at"])):
            return None
        return PendingCompletion(delegation_id=delegation_id, event=json.loads(row["event_json"]),
                                 completed_at=row["completed_at"], dispatched_at=row["dispatched_at"])

    # ── delivery obligations (the SQLite ``_delivery_ledger`` port) ─────────

    # Records a pending obligation owned by ``owner``, replacing a row with the same id, then prunes.
    def record_delivery_obligation(self, *, obligation_id: str, session_key: str, platform: str, chat_id: str,
                                  thread_id: Optional[str], content: str,
                                  adapter_profile: Optional[str] = None, owner: OwnerStamp) -> None:
        now, (pid, started) = time.time(), (owner.pid, owner.started_at)

        def _do(conn):
            conn.execute(
                """INSERT INTO delivery_obligations
                   (obligation_id, session_key, platform, chat_id, thread_id,
                    content, state, attempts, created_at, updated_at,
                    owner_pid, owner_started_at, adapter_profile)
                   VALUES (%s, %s, %s, %s, %s, %s, 'pending', 0, %s, %s, %s, %s, %s)
                   ON CONFLICT (obligation_id) DO UPDATE SET
                     session_key = EXCLUDED.session_key,
                     platform = EXCLUDED.platform,
                     chat_id = EXCLUDED.chat_id,
                     thread_id = EXCLUDED.thread_id,
                     content = EXCLUDED.content,
                     state = EXCLUDED.state,
                     attempts = EXCLUDED.attempts,
                     created_at = EXCLUDED.created_at,
                     updated_at = EXCLUDED.updated_at,
                     owner_pid = EXCLUDED.owner_pid,
                     owner_started_at = EXCLUDED.owner_started_at,
                     adapter_profile = EXCLUDED.adapter_profile""",
                (obligation_id, session_key, platform, str(chat_id), str(thread_id) if thread_id else None,
                 content, now, now, pid, started,
                 str(adapter_profile).strip() if adapter_profile else "default"))
            _prune_unlocked(conn, now)

        self._ledger_write(_do)

    def record_crash_left_reply(self, *, obligation_id: str, session_key: str, platform: str, chat_id: str,
                                thread_id: Optional[str], content: str, since: float,
                                adapter_profile: Optional[str] = None) -> None:
        """Adopt a reply a killed process persisted but never ledgered: unowned and ``attempting``, a no-op when the
        same reply was ledgered since ``since``."""
        now = time.time()

        def _do(conn):
            conn.execute(
                """INSERT INTO delivery_obligations
                   (obligation_id, session_key, platform, chat_id, thread_id,
                    content, state, attempts, created_at, updated_at,
                    owner_pid, owner_started_at, adapter_profile)
                   SELECT %s, %s, %s, %s, %s, %s, 'attempting', 0, %s, %s, NULL, NULL, %s
                   WHERE NOT EXISTS (SELECT 1 FROM delivery_obligations
                                     WHERE session_key = %s AND content = %s AND created_at >= %s)
                   ON CONFLICT (obligation_id) DO NOTHING""",
                (obligation_id, session_key, platform, str(chat_id), str(thread_id) if thread_id else None,
                 content, now, now, str(adapter_profile).strip() if adapter_profile else "default",
                 session_key, content, since))

        self._ledger_write(_do)

    # Sets one row's state and error whatever its owner; the error keeps its first 500 characters.
    def _update_obligation_state(self, obligation_id: str, state: str, error: str = "") -> None:
        def _do(conn):
            conn.execute(
                """UPDATE delivery_obligations
                   SET state=%s, updated_at=%s, last_error=%s
                   WHERE obligation_id=%s""",
                (state, time.time(), error[:500] if error else None, obligation_id))

        self._ledger_write(_do)

    def mark_obligation_attempting(self, *, obligation_id: str) -> None:
        self._update_obligation_state(obligation_id, "attempting")

    def mark_obligation_delivered(self, *, obligation_id: str) -> None:
        self._update_obligation_state(obligation_id, "delivered")

    def mark_obligation_failed(self, *, obligation_id: str, error: str = '') -> None:
        self._update_obligation_state(obligation_id, "failed", error=error)

    def release_runtime_obligation_claim(self, *, obligation_id: str, error: str = '',
                                        owner: OwnerStamp) -> bool:
        """Return an unsent runtime claim to ``failed`` without spending an attempt, fail-closed to the exact
        process instance and ``attempting`` state."""
        pid, started = owner.pid, owner.started_at
        if started is None:
            return False

        def _do(conn):
            cursor = conn.execute(
                """UPDATE delivery_obligations
                   SET state='failed', attempts=CASE
                           WHEN attempts > 0 THEN attempts - 1 ELSE 0 END,
                       updated_at=%s, last_error=%s
                   WHERE obligation_id=%s AND state='attempting'
                     AND owner_pid IS NOT DISTINCT FROM %s AND owner_started_at IS NOT DISTINCT FROM %s""",
                (time.time(), error[:500] if error else None, obligation_id, pid, started))
            return bool(cursor.rowcount)

        return self._ledger_write(_do)

    # The caller probes owner liveness and passes the dead owners as ``dead_owners``.
    def sweep_recoverable_obligations(self, *, now: Optional[float] = None,
                                     deliverable_platforms: Optional[frozenset[str]] = None,
                                     deliverable_targets: Optional[frozenset[tuple[str, str]]] = None,
                                     owner: OwnerStamp,
                                     dead_owners: frozenset[RecordedOwner]) -> tuple[ClaimedObligation, ...]:
        """Claim undelivered rows of dead owners: each claim re-stamps the owner, moves the row to ``attempting`` and
        spends an attempt, guarded on the previous owner stamp. Exhausted or stale rows become ``abandoned``; a
        flood row inside its wait is adopted without spending an attempt; rows of a platform or target the caller
        cannot send on are left alone."""
        now, (pid, started) = now if now is not None else time.time(), (owner.pid, owner.started_at)

        def _do(conn):
            claimed: list[ClaimedObligation] = []
            rows = conn.execute(
                """SELECT obligation_id, session_key, platform, chat_id, thread_id,
                          content, state, attempts, created_at,
                          owner_pid, owner_started_at, adapter_profile, last_error, updated_at
                   FROM delivery_obligations
                   WHERE state IN ('pending', 'attempting', 'failed')""").fetchall()
            for row in rows:
                oid, session_key, platform, chat_id, thread_id, content = (
                    row["obligation_id"], row["session_key"], row["platform"], row["chat_id"],
                    row["thread_id"], row["content"])
                state, attempts, created_at = row["state"], row["attempts"], row["created_at"]
                owner_pid, owner_started_at, adapter_profile = (
                    row["owner_pid"], row["owner_started_at"], row["adapter_profile"])
                last_error, updated_at = row["last_error"], row["updated_at"]
                if RecordedOwner(pid=owner_pid, started_at=owner_started_at) not in dead_owners:
                    continue  # a live gateway still owns this row
                if attempts >= OBLIGATION_MAX_ATTEMPTS or (now - created_at) > OBLIGATION_STALE_AFTER_SECONDS:
                    conn.execute(
                        """UPDATE delivery_obligations
                           SET state='abandoned', updated_at=%s WHERE obligation_id=%s""",
                        (now, oid))
                    continue
                if ((deliverable_platforms is not None and platform not in deliverable_platforms)
                        or (deliverable_targets is not None
                            and (platform, adapter_profile) not in deliverable_targets)):
                    continue  # no adapter this boot: claiming would spend an attempt on a no-op
                flood_row = state == "failed" and is_flood_error(last_error)
                if flood_row and now < flood_not_before(updated_at, last_error):
                    # Still inside the platform's wait: adopt the dead owner's row without spending an attempt.
                    cursor = conn.execute(
                        """UPDATE delivery_obligations
                           SET owner_pid=%s, owner_started_at=%s,
                               adapter_profile=COALESCE(adapter_profile, 'default')
                           WHERE obligation_id=%s AND owner_pid IS NOT DISTINCT FROM %s""",
                        (pid, started, oid, owner_pid))
                    if cursor.rowcount:
                        claimed.append(ClaimedObligation(
                            obligation_id=oid, session_key=session_key, platform=platform, chat_id=chat_id,
                            thread_id=thread_id, content=content, profile=adapter_profile or "default",
                            attempts=attempts, adopted=True,
                            not_before=flood_not_before(updated_at, last_error)))
                    continue
                # Every claim starts a send, so the row leaves 'pending'/'failed' for 'attempting' in the same CAS; a
                # claimed flood row also drops its stale refusal.
                cursor = conn.execute(
                    """UPDATE delivery_obligations
                       SET owner_pid=%s, owner_started_at=%s, attempts=attempts+1, updated_at=%s,
                           adapter_profile=COALESCE(adapter_profile, 'default'), state='attempting',
                           last_error=CASE WHEN %s THEN NULL ELSE last_error END
                       WHERE obligation_id=%s AND owner_pid IS NOT DISTINCT FROM %s""",
                    (pid, started, now, bool(flood_row), oid, owner_pid))
                if cursor.rowcount:
                    # A never-claimed pending row was never sent: redeliver plainly; anything else carries a marker.
                    claimed.append(_claimed_row(
                        oid, session_key, platform, chat_id, thread_id, content, attempts,
                        adapter_profile or "default",
                        needs_marker=state != "pending" or attempts > 0, flood=flood_row))
            return tuple(claimed)

        return self._ledger_write(_do)

    # The caller computes each row's deadline and passes it in ``due``; a row is claimed only while it is as the
    # candidate read saw it.
    def sweep_failed_obligations_for_runtime(self, platform: str, *, now: Optional[float] = None,
                                            profile: Optional[str] = None, owner: OwnerStamp,
                                            due: Sequence[RetryDue]) -> tuple[ClaimedObligation, ...]:
        """Claim this process's failed rows that are due for another send, for one adapter: only rows stamped to
        this exact process instance and profile, every update guarded by the prior owner stamp and ``failed``
        state. Claimed rows always carry a marker: the reconnect one, or the flood one."""
        now, (pid, started) = now if now is not None else time.time(), (owner.pid, owner.started_at)
        if started is None:  # PID alone cannot tell this process from a stale row left after PID reuse
            return ()
        expected_profile = _expected_profile(profile)

        def _do(conn):
            claimed: list[ClaimedObligation] = []
            rows = conn.execute(
                """SELECT obligation_id, session_key, platform, chat_id, thread_id,
                          content, attempts, created_at, owner_pid,
                          owner_started_at, last_error, adapter_profile, updated_at
                   FROM delivery_obligations
                   WHERE state='failed' AND platform=%s""", (platform,)).fetchall()
            current = {row["obligation_id"]: row for row in rows}
            for item in due:
                row = current.get(item.obligation_id)
                if row is None:
                    continue
                oid, session_key, row_platform = row["obligation_id"], row["session_key"], row["platform"]
                chat_id, thread_id, content = row["chat_id"], row["thread_id"], row["content"]
                attempts, created_at = row["attempts"], row["created_at"]
                owner_pid, owner_started_at = row["owner_pid"], row["owner_started_at"]
                last_error, adapter_profile, updated_at = (
                    row["last_error"], row["adapter_profile"], row["updated_at"])
                # Exact process-start matching prevents PID reuse from stealing work.
                if adapter_profile != expected_profile or owner_pid != pid or owner_started_at != started:
                    continue
                if (updated_at, last_error, attempts) != (item.updated_at, item.last_error, item.attempts):
                    continue  # changed since the candidate read
                owner_guard = (now, oid, owner_pid, owner_started_at)
                if attempts >= OBLIGATION_MAX_ATTEMPTS or (now - created_at) > OBLIGATION_STALE_AFTER_SECONDS:
                    conn.execute(
                        """UPDATE delivery_obligations
                           SET state='abandoned', updated_at=%s
                           WHERE obligation_id=%s AND state='failed'
                             AND owner_pid IS NOT DISTINCT FROM %s
                             AND owner_started_at IS NOT DISTINCT FROM %s""", owner_guard)
                    continue
                if now < item.not_before:
                    continue  # the platform's wait or the backoff has not passed
                # The claim clears the stale error: this is a fresh attempt.
                cursor = conn.execute(
                    """UPDATE delivery_obligations
                       SET state='attempting', attempts=attempts+1, updated_at=%s, last_error=NULL
                       WHERE obligation_id=%s AND state='failed'
                         AND owner_pid IS NOT DISTINCT FROM %s
                         AND owner_started_at IS NOT DISTINCT FROM %s""", owner_guard)
                if cursor.rowcount:
                    claimed.append(_claimed_row(
                        oid, session_key, row_platform, chat_id, thread_id, content,
                        attempts, adapter_profile, needs_marker=True, runtime=True,
                        flood=is_flood_error(last_error), last_error=last_error))
            return tuple(claimed)

        return self._ledger_write(_do)

    # The read behind the retry schedule; the caller aggregates it per adapter.
    def list_owned_failed_obligations(self, *, owner: OwnerStamp) -> tuple[FailedObligationRow, ...]:
        pid, started = owner.pid, owner.started_at
        if started is None:
            return ()

        def _read(store):
            return store.all(
                """SELECT platform, adapter_profile, updated_at, last_error, attempts, created_at
                   FROM delivery_obligations
                   WHERE state='failed'
                     AND owner_pid IS NOT DISTINCT FROM %s
                     AND owner_started_at IS NOT DISTINCT FROM %s""", pid, started)

        rows = self._ledger_read(_read)
        return tuple(FailedObligationRow(platform=row["platform"], adapter_profile=row["adapter_profile"],
                                         updated_at=row["updated_at"], last_error=row["last_error"],
                                         attempts=row["attempts"], created_at=row["created_at"])
                     for row in rows)

    # The owner stamps of rows still awaiting delivery: pending, attempting or failed.
    def list_obligation_owners(self) -> frozenset[RecordedOwner]:
        def _read(store):
            return store.all(
                """SELECT DISTINCT owner_pid, owner_started_at
                   FROM delivery_obligations
                   WHERE state IN ('pending', 'attempting', 'failed')""")

        rows = self._ledger_read(_read)
        return frozenset(RecordedOwner(pid=row["owner_pid"], started_at=row["owner_started_at"]) for row in rows)

    # The candidate read of ``sweep_failed_obligations_for_runtime``, kept to this process instance and profile.
    def list_runtime_retry_candidates(self, platform: str, *, profile: Optional[str] = None,
                                     owner: OwnerStamp) -> tuple[RetryCandidate, ...]:
        pid, started = owner.pid, owner.started_at
        if started is None:
            return ()
        expected_profile = _expected_profile(profile)

        def _read(store):
            return store.all(
                """SELECT obligation_id, session_key, platform, chat_id, thread_id,
                          content, attempts, created_at, owner_pid,
                          owner_started_at, last_error, adapter_profile, updated_at
                   FROM delivery_obligations
                   WHERE state='failed' AND platform=%s""", platform)

        rows = self._ledger_read(_read)
        return tuple(RetryCandidate(obligation_id=row["obligation_id"], updated_at=row["updated_at"],
                                   last_error=row["last_error"], attempts=row["attempts"])
                     for row in rows
                     if row["adapter_profile"] == expected_profile and row["owner_pid"] == pid
                     and row["owner_started_at"] == started)

    # ── subagent records (reads) ───────────────────────────────────────────

    def get_subagent(self, *, subagent_id: str) -> Optional[SubagentRecord]:
        def _read(store):
            return store.one(f"SELECT {_SUBAGENT_COLUMNS} FROM resumable_subagents WHERE subagent_id = %s",
                             subagent_id)

        row = self._ledger_read(_read)
        return None if row is None else _record(row)

    def list_inflight_subagents(self) -> tuple[SubagentRecord, ...]:
        def _read(store):
            return store.all(f"SELECT {_SUBAGENT_COLUMNS} FROM resumable_subagents "
                             "WHERE state IN ('running','claimed')")

        return tuple(_record(row) for row in self._ledger_read(_read))

    def list_continuable_subagents(self, *, unit_id: str) -> tuple[str, ...]:
        def _read(store):
            return store.all(
                "SELECT subagent_id FROM resumable_subagents WHERE state='resumable' "
                f"AND COALESCE({_JR} ->> 'unit_id', {_JR} ->> 'delegation_id') = %s ORDER BY created_at",
                unit_id)

        return tuple(row["subagent_id"] for row in self._ledger_read(_read))

    def list_pinned_subagents(self, *, max_resumable: int, now: Optional[float] = None) -> tuple[SubagentRecord, ...]:
        self._store.require_open()
        try:
            rows = self._ledger_read(
                lambda store: store.all(
                    f"SELECT {_SUBAGENT_COLUMNS} FROM resumable_subagents "
                    "WHERE pinned=1 AND state IN ('resumable','superseded','running','claimed') "
                    "AND (expires_at IS NULL OR expires_at > %s) ORDER BY updated_at DESC LIMIT %s",
                    _at(now), max(max_resumable, 0)))
            return tuple(_record(row) for row in rows)
        except PersistenceError:
            return ()  # no table yet: nothing pinned

    def list_auto_resume_candidates(self, *, owed_only: bool = False) -> tuple[SubagentRecord, ...]:
        def _read(store):
            return store.all(f"SELECT {_SUBAGENT_COLUMNS} FROM resumable_subagents WHERE "
                             f"(state='resumable' OR (state='expired' AND {_OWED_SQL}))"
                             + (f" AND {_OWED_SQL}" if owed_only else "")
                             + " ORDER BY updated_at")

        return tuple(_record(row) for row in self._ledger_read(_read))

    def has_owed_subagents(self) -> bool:
        self._store.require_open()
        try:
            return self._ledger_read(
                lambda store: store.scalar(
                    f"SELECT 1 FROM resumable_subagents WHERE state IN ('resumable','expired') "
                    f"AND {_OWED_SQL} LIMIT 1")) is not None
        except PersistenceError:
            return True  # unreadable here: the caller's writer pass decides

    def list_waiting_subagents(self, *, now: Optional[float] = None) -> tuple[SubagentRecord, ...]:
        def _read(store):
            return store.all(f"SELECT {_SUBAGENT_COLUMNS} FROM resumable_subagents "
                             f"WHERE state='resumable' AND {_OWED_SQL} "
                             "AND (expires_at IS NULL OR expires_at > %s)", _at(now))

        return tuple(_record(row) for row in self._ledger_read(_read))

    def subagent_parent_superseded(self, *, parent_session_id: SessionId) -> Optional[str]:
        with translated():
            return _superseded_reason_read(self._store, str(parent_session_id))

    def get_delegation_route(self, *, delegation_id: DelegationId) -> Optional[DelegationRoute]:
        def _read(store):
            return store.one("SELECT origin_session, origin_ui_session_id, parent_session_id, "
                            "origin_session_id, task_json, result_json, dispatched_at, updated_at "
                            "FROM async_delegations WHERE delegation_id = %s", str(delegation_id))

        got = self._ledger_read(_read)
        if got is None:
            return None
        return DelegationRoute(delegation_id=DelegationId(str(delegation_id)), origin_session=got["origin_session"],
                               origin_ui_session_id=got["origin_ui_session_id"],
                               parent_session_id=got["parent_session_id"],
                               origin_session_id=got["origin_session_id"] or "",
                               task=json.loads(got["task_json"]) if got["task_json"] else None,
                               result=json.loads(got["result_json"]) if got["result_json"] else None,
                               dispatched_at=got["dispatched_at"], updated_at=got["updated_at"])

    # ── subagent records (run records) ────────────────────────────────────

    def put_subagent(self, *, subagent_id: str, write: SubagentWrite, expected: Optional[SubagentRecord],
                     limits: Optional[SubagentLimits] = None, now: Optional[float] = None) -> bool:
        at = _at(now)

        def _do(conn):
            if _row(conn, subagent_id) != expected:
                return False
            _upsert(conn, subagent_id, write, at)
            if limits is not None:
                _apply_limits(conn, limits, at)
            return True

        return bool(self._subagent_write(_do, False))

    def start_auto_resumed_run(self, *, subagent_id: str, write: SubagentWrite, expected: SubagentRecord,
                               owner_pid: int, now: Optional[float] = None) -> AutoStart:
        """The run-start write of a run the scheduler started gives the parent its last word first: the parent's
        decision stays on the row and the attempt is refunded."""
        at = _at(now)

        def _do(conn):
            row = _row(conn, subagent_id)
            if row is not None and _unstarted_auto_claim(row) and row.owner_pid == owner_pid:
                reason = _superseded_reason(conn, str(row.document["auto_claim"]))
                if not reason:
                    if row != expected:
                        return AutoStart(written=False, cancelled=None)
                    _upsert(conn, subagent_id, write, at)
                    return AutoStart(written=True, cancelled=None)
                _note(conn, subagent_id, state=SubagentState.SUPERSEDED, note=reason, now=at, from_states=(),
                      unstarted_claim=True)
            elif row is not None and row.state in (SubagentState.STOPPED, SubagentState.SUPERSEDED):
                reason = row.document.get("auto_note") or "its parent stopped it"
            else:
                return AutoStart(written=False, cancelled=(
                    f"{AUTO_RESUME_CANCELLED}: the record moved ({row.state.value if row else 'gone'})"))
            _refund(conn, subagent_id)
            return AutoStart(written=False, cancelled=f"{AUTO_RESUME_CANCELLED}: {reason}")

        return self._subagent_write(_do, AutoStart(written=False, cancelled=None))

    def checkpoint_subagent(self, *, subagent_id: str, cwd: str, repo_baselines: Mapping[str, JsonValue],
                            now: Optional[float] = None) -> bool:
        at = _at(now)

        def _do(conn):
            return conn.execute(
                f"UPDATE resumable_subagents SET record_json=({_JR} || jsonb_build_object('cwd', %s::text, "
                f"'repo_baselines', %s::jsonb)), updated_at=%s WHERE subagent_id=%s AND state='running'",
                (cwd, json.dumps(repo_baselines), at, subagent_id)).rowcount == 1

        return bool(self._subagent_write(_do, False))

    def mark_subagents_gateway_shutdown(self, *, owner_pid: int) -> int:
        return int(self._subagent_write(
            lambda conn: conn.execute(
                "UPDATE resumable_subagents SET stop_cause=%s WHERE state='running' AND owner_pid=%s",
                (GATEWAY_SHUTDOWN, owner_pid)).rowcount, 0) or 0)

    def apply_subagent_limits(self, *, limits: SubagentLimits, subagent_id: Optional[str] = None,
                             now: Optional[float] = None) -> Optional[SubagentRecord]:
        at = _at(now)

        def _do(conn):
            _apply_limits(conn, limits, at)
            return None if subagent_id is None else _row(conn, subagent_id)

        return self._subagent_write(_do)

    # ── subagent records (continue) ───────────────────────────────────────

    def claim_subagent(self, *, subagent_id: str, expected_state: SubagentState, owner_pid: int,
                       owner_started_at: Optional[int], now: Optional[float] = None) -> bool:
        at = _at(now)
        return self._subagent_write(lambda conn: conn.execute(
            f"UPDATE resumable_subagents SET state='claimed', owner_pid=%s, owner_started_at=%s, "
            f"updated_at=%s, record_json=({_JR} || jsonb_build_object('auto_promised', 'false'::jsonb)) "
            f"WHERE subagent_id=%s AND state=%s",
            (owner_pid, owner_started_at, at, subagent_id, SubagentState(expected_state).value)).rowcount,
            0) == 1

    def hand_back_subagent(self, *, subagent_id: str, state: SubagentState, promised: Optional[bool] = None,
                           owner_pid: Optional[int] = None) -> None:
        """A stop that landed during the claim (``$.auto_cancel``) wins over handing a promise back."""
        def _do(conn):
            target = SubagentState(state).value
            hand_promised = promised
            if owner_pid is not None:
                where_sql, where = " AND owner_pid = %s", [subagent_id, owner_pid]
            else:
                where_sql, where = "", [subagent_id]
            got = conn.execute(
                f"SELECT ({_JR} -> 'auto_cancel' ->> 'state') AS cancel_state, "
                f"({_JR} -> 'auto_cancel' ->> 'note') AS cancel_note FROM resumable_subagents "
                f"WHERE subagent_id = %s AND state = 'claimed'{where_sql}", where).fetchone()
            if got is None:
                return
            json_expr, params, unpin = _JR, [], ""
            if got["cancel_state"]:  # the parent cancelled it during the claim: no automatic resume, no report
                hand_promised = False
                if target == "resumable":
                    target = got["cancel_state"]
                    json_expr = f"({json_expr} || jsonb_build_object('auto_note', %s::text))"
                    params.append(got["cancel_note"])
                    unpin = ", pinned=0" if target == "stopped" else ""
            if hand_promised is not None:
                json_expr = f"({json_expr} || jsonb_build_object('auto_promised', %s::bool))"
                params.append(bool(hand_promised))
            conn.execute(
                f"UPDATE resumable_subagents SET record_json=(({json_expr}) - 'auto_claim' - 'auto_cancel'), "
                f"state=%s{unpin} WHERE subagent_id=%s AND state='claimed'{where_sql}",
                (*params, target, *where))

        self._subagent_write(_do)

    # ── subagent records (automatic resume) ───────────────────────────────

    def note_subagent(self, *, subagent_id: str, note: str, state: Optional[SubagentState] = None,
                      from_states: Sequence[SubagentState] = (SubagentState.RESUMABLE,), unpin: bool = False,
                      unstarted_claim: bool = False, now: Optional[float] = None) -> bool:
        at = _at(now)
        return bool(self._subagent_write(lambda conn: _note(
            conn, subagent_id, state=state, note=note, now=at, from_states=from_states, unpin=unpin,
            unstarted_claim=unstarted_claim), False))

    def cancel_expired_subagent(self, *, subagent_id: str, note: str, now: Optional[float] = None) -> bool:
        at = _at(now)
        return bool(self._subagent_write(lambda conn: _cancel_expired(conn, subagent_id, note, at), False))

    def cancel_manual_subagent_claim(self, *, subagent_id: str,
                                     state: Literal[SubagentState.STOPPED, SubagentState.SUPERSEDED], note: str,
                                     now: Optional[float] = None) -> bool:
        at = _at(now)
        return bool(self._subagent_write(
            lambda conn: _cancel_manual_claim(conn, subagent_id, state=state, note=note, now=at), False))

    def claim_auto_resume(self, *, subagent_id: str, expected_attempts: int, parent_session_id: SessionId,
                          owner_pid: int, owner_started_at: Optional[int],
                          now: Optional[float] = None) -> AutoResumeClaim:
        """ONE write re-reads the row, re-runs supersession on that connection and, only while it is still
        ``resumable``, unsuperseded, uncancelled and under the cap, claims it and spends the attempt before it runs."""
        at = _at(now)
        parent_sid = str(parent_session_id)

        def _do(conn):
            row = _row(conn, subagent_id)
            if (row is None or row.state != SubagentState.RESUMABLE or row.document.get("auto_off")
                    or not row.document.get("auto_promised")
                    or _attempts(row) != expected_attempts
                    or expected_attempts >= AUTO_RESUME_MAX_ATTEMPTS):
                return AutoResumeClaim(record=None, superseded=None)
            if reason := _superseded_reason(conn, parent_sid):
                _note(conn, subagent_id, state=SubagentState.SUPERSEDED, note=reason, now=at)
                return AutoResumeClaim(record=None, superseded=reason)
            conn.execute(
                f"UPDATE resumable_subagents SET state='claimed', owner_pid=%s, owner_started_at=%s, "
                f"updated_at=%s, record_json=({_JR} || jsonb_build_object('auto_attempts', %s::int, "
                f"'auto_last_at', %s::double precision, 'auto_claim', %s::text)) WHERE subagent_id=%s AND state='resumable'",
                (owner_pid, owner_started_at, at, expected_attempts + 1, at, parent_sid, subagent_id))
            return AutoResumeClaim(record=_row(conn, subagent_id), superseded=None)

        return self._subagent_write(_do, AutoResumeClaim(record=None, superseded=None))

    def confirm_auto_start(self, *, subagent_id: str) -> Optional[str]:
        def _do(conn):
            row = _row(conn, subagent_id)
            if row is None or not _unstarted_auto_claim(row):
                return AUTO_RESUME_CANCELLED
            if reason := _superseded_reason(conn, str(row.document["auto_claim"])):
                _note(conn, subagent_id, state=SubagentState.SUPERSEDED, note=reason, now=time.time(),
                      from_states=(), unstarted_claim=True)
                return f"{AUTO_RESUME_CANCELLED}: {reason}"
            return None

        return self._subagent_write(_do, AUTO_RESUME_CANCELLED)

    def settle_auto_refusal(self, *, subagent_id: str, kind: AutoRefusal, attempts: int, note: str,
                            owner_pid: int) -> None:
        if kind == AutoRefusal.CANCELLED:  # the parent's decision already wrote the row: only the attempt is refunded
            sql, params = f"({_JR} || jsonb_build_object('auto_attempts', %s::int))", (attempts,)
        elif kind == AutoRefusal.TRANSIENT:
            sql, params = f"({_JR} || jsonb_build_object('auto_attempts', %s::int, 'auto_note', %s::text))", (attempts, note)
        else:
            sql, params = f"({_JR} || jsonb_build_object('auto_off', %s::text, 'auto_note', %s::text))", (note, note)
        self._subagent_write(lambda conn: conn.execute(
            f"UPDATE resumable_subagents SET record_json=(({sql}) - 'auto_claim'), "
            "state=CASE WHEN state='claimed' AND owner_pid=%s THEN 'resumable' ELSE state END "
            "WHERE subagent_id=%s",
            (*params, owner_pid, subagent_id)))

    def give_up_auto_resume(self, *, subagent_id: str, reason: str, completion: Optional[GaveUpCompletion],
                            now: Optional[float] = None) -> GiveUpOutcome:
        """ONE write marks the record and records its one failed completion as its own pending ledger row, so a crash
        on either side of the commit leaves the report replayable and never doubled."""
        at = _at(now)

        def _do(conn):
            if conn.execute(
                    f"UPDATE resumable_subagents SET record_json=({_JR} || jsonb_build_object("
                    f"'auto_gave_up', %s::text, 'auto_off', %s::text, 'auto_note', %s::text)), updated_at=%s "
                    "WHERE subagent_id=%s AND state IN ('resumable','expired') AND " + _OWED_SQL,
                    # a cancel since the caller read it cleared the debt: nothing to report
                    (reason, reason, reason, at, subagent_id)).rowcount != 1:
                return GiveUpOutcome.NOT_OWED
            if completion is None:
                return GiveUpOutcome.NO_ROUTE
            got = conn.execute("SELECT updated_at FROM async_delegations WHERE delegation_id = %s",
                               (str(completion.unit_id),)).fetchone()
            if got is None:  # pruning keeps a waiting child's unit row: only a nested or joined child has none
                return GiveUpOutcome.NO_ROUTE
            if got["updated_at"] != completion.unit_updated_at:
                raise _RouteMoved
            _insert_completion(conn, completion, at)
            return GiveUpOutcome.RECORDED

        try:
            return self._subagent_write(_do, GiveUpOutcome.NOT_OWED)
        except _RouteMoved:
            return GiveUpOutcome.ROUTE_MOVED

    def supersede_subagents_for_session(self, *, note: str, session_key: str = "", origin_ui_session_id: str = "",
                                       parent_session_id: str = "", promised_only: bool = False) -> int:
        selectors = [(col, val) for col, val in (("origin_session", session_key), ("origin_ui_session_id",
                    origin_ui_session_id), ("parent_session_id", parent_session_id)) if val]
        if not selectors:
            return 0
        owed_only = f" AND {_OWED_SQL}" if promised_only else ""

        def _do(conn):
            now = time.time()
            where = " OR ".join(f"{col} = %s" for col, _ in selectors)
            units = {row["delegation_id"] for row in conn.execute(
                f"SELECT delegation_id FROM async_delegations WHERE {where}",
                [val for _, val in selectors]).fetchall()}
            rows = conn.execute(
                "SELECT subagent_id, " + f"{_JR} ->> 'unit_id'" + " AS unit_id FROM resumable_subagents "
                f"WHERE state='claimed' OR (state='resumable'{owed_only}) "
                f"OR (state='expired' AND {_OWED_SQL})").fetchall()
            return sum(_note(conn, row["subagent_id"], state=SubagentState.SUPERSEDED, note=note, now=now,
                             unstarted_claim=True)
                       or _cancel_expired(conn, row["subagent_id"], note, now)
                       or _cancel_manual_claim(conn, row["subagent_id"], state=SubagentState.SUPERSEDED,
                                              note=note, now=now)
                       for row in rows if row["unit_id"] in units)

        return int(self._subagent_write(_do, 0) or 0)

    def dismiss_subagent(self, *, expected: SubagentRecord, note: str,
                         now: Optional[float] = None) -> Optional[Literal["dismissed", "recorded"]]:
        at = _at(now)
        sid = expected.subagent_id

        def _do(conn):
            record = _row(conn, sid)
            if record is None or record != expected:
                return None
            if record.state == SubagentState.CLAIMED and not record.document.get("auto_claim"):
                # The parent's continue goes on; if it is refused the hand-back ends the row ``stopped``.
                return ("recorded" if _cancel_manual_claim(conn, sid, state=SubagentState.STOPPED, note=note, now=at)
                        else None)
            done = _note(conn, sid, state=SubagentState.STOPPED, note=note, now=at,
                         from_states=(SubagentState.RESUMABLE, SubagentState.SUPERSEDED), unpin=True,
                         unstarted_claim=True) or _cancel_expired(conn, sid, note, at)
            return "dismissed" if done else None

        return self._subagent_write(_do)


_PORTED = tuple(sorted({
    "apply_subagent_limits", "cancel_expired_subagent", "cancel_manual_subagent_claim", "check_handoff_settled",
    "checkpoint_subagent", "claim_auto_resume", "claim_delivery", "claim_handoff", "claim_subagent",
    "complete_delivery", "complete_handoff", "confirm_auto_start", "defer_delivery",
    "delete_unscheduled_delegation", "dismiss_subagent", "drop_delivery", "fail_handoff", "get_admitted_completion",
    "get_compression_lock_holder", "get_delegation", "get_delegation_route", "get_handoff", "get_subagent",
    "give_up_auto_resume", "hand_back_subagent", "has_owed_subagents", "has_pending_handoffs",
    "has_replayable_completions", "increment_hygiene_failure_streak", "list_auto_resume_candidates",
    "list_continuable_subagents", "list_finished_delegations", "list_inflight_delegations",
    "list_inflight_subagents", "list_obligation_owners", "list_orphan_completion_offers",
    "list_orphan_completion_owners", "list_orphan_completions", "list_owned_failed_obligations",
    "list_pending_handoffs", "list_pinned_subagents", "list_runtime_retry_candidates", "list_waiting_subagents",
    "mark_delegation_delivered", "mark_obligation_attempting", "mark_obligation_delivered",
    "mark_obligation_failed", "mark_subagents_gateway_shutdown", "note_subagent", "prune_delegation_ledger",
    "put_subagent", "reclaim_stale_running_handoffs", "record_crash_left_reply", "record_delegation_completion",
    "record_delegation_dispatch", "record_delivery_obligation", "record_unit_child",
    "recover_abandoned_delegations", "refresh_compression_lock", "refresh_turn_lease",
    "release_compression_lock", "release_delivery", "release_runtime_obligation_claim", "release_turn_lease",
    "reopen_admitted_completion", "request_handoff", "reset_hygiene_failure_streak", "settle_auto_refusal",
    "start_auto_resumed_run", "subagent_parent_superseded", "supersede_subagents_for_session",
    "sweep_failed_obligations_for_runtime", "sweep_orphaned_completions", "sweep_recoverable_obligations",
    "take_replayable_completions", "try_acquire_compression_lock", "try_acquire_turn_lease",
    "turn_lease_expires_at",
}))

add_stubs(PgCoordinationRepository, CoordinationRepository, implemented=_PORTED)

register_repository("coordination", lambda store: PgCoordinationRepository(store),
                    {"transactional_claims": True})

"""``HostedRoomDriverRepository`` on PostgreSQL (stage 4): the room driver's fenced leases and task state
machine in the placement's schema, each verb one write transaction or one read on the placement - the
SQLite driver store's SQL ported to the PG dialect.

The driver tables and the status index are in the declarative DDL (the reference's ``DRIVER_STEP`` -
its migration, validation and first-touch - is a no-op here; an incompatible draft fails the same way:
the DDL is the contract). Every write that changes a task runs the work-record capture after its update
in the same transaction.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Collection, Mapping
from typing import Any, Callable

from persistence import JsonValue
from persistence.gateway.hosted_rooms import (
    SHOWN_TASK_STATUSES,
    TERMINAL_STATUSES,
    DriverLease,
    DriverStateError,
    DriverValidationError,
    HostedPlacement,
    HostedTaskIdentity,
    InvalidTaskTransitionError,
    LeaseHeldError,
    LeaseRelease,
    RoomChangeRows,
    RoomRecovery,
    RoomUnavailableError,
    StaleLeaseError,
    StaleTaskError,
    TaskAttempt,
    TaskConflictError,
    TaskStatus,
    TerminalStatus,
)
from session_rules.hosted_rooms import clock, compact_json
from session_rules.hosted_tasks import (
    MAX_RETAINED_TERMINAL_TASKS, MAX_TASK_PRUNE_BATCH, TERMINAL_TASK_RETENTION_SECONDS, normalize_task_payload,
)

from .._backend import PostgresStore, register_repository, translated
from ._hosted_common import fenced_update as _fenced_update, placement_all, placement_read, placement_write, table_exists
from ._hosted_work import capture_transition_locked

__all__ = ["PgHostedRoomDriverRepository", "_task_from_row"]

_TASK_COLUMN_ORDER = (
    "room_id", "task_id", "thread_id", "turn_id", "source_event_seq", "payload_json", "payload_digest", "status",
    "execution_generation", "cancel_generation", "run_gateway_id", "run_process_generation", "run_lease_generation",
    "cancel_id", "settlement_id", "settlement_status", "result_json", "created_at", "updated_at", "started_at",
    "terminal_at", "indeterminate_at")
_TASK_ORDER = "ORDER BY source_event_seq, created_at, task_id"
_SELECT_LEASE = "SELECT * FROM hosted_room_driver_leases WHERE room_id = %s"
_SELECT_TASK = "SELECT * FROM hosted_room_driver_tasks WHERE room_id = %s AND task_id = %s"

# --- Fenced task UPDATE statements (one per state-machine transition) ---------
# Every transition is "UPDATE ... SET <set> WHERE room_id=%s AND task_id=%s AND <fence>"; the fence names the
# expected status plus the generations that must not have moved.
_GENERATION_FENCE = "execution_generation=%s AND cancel_generation=%s"
_RUN_FENCE = "run_gateway_id=%s AND run_process_generation=%s AND run_lease_generation=%s"
_SETTLE_SET = "status=%s, settlement_id=%s, settlement_status=%s, result_json=%s, terminal_at=%s, updated_at=%s"
_REQUEUE_SET = "status='queued', run_gateway_id=NULL, run_process_generation=NULL, run_lease_generation=NULL"
_CANCEL_SET = "status='cancelled', cancel_generation=%s, cancel_id=%s, terminal_at=%s, updated_at=%s"


def _task_update(set_clause: str, fence: str) -> str:
    return f"UPDATE hosted_room_driver_tasks SET {set_clause} WHERE room_id = %s AND task_id = %s AND {fence}"


def _generation_update(set_clause: str, status: str) -> str:
    """Transition fenced on ``status`` + both generations (terminal settlements and the recovery family)."""
    return _task_update(set_clause, f"status='{status}' AND {_GENERATION_FENCE}")


_SETTLE_RUNNING_SQL = _generation_update(_SETTLE_SET, "running") + f" AND {_RUN_FENCE}"
_SETTLE_STOPPING_SQL = _generation_update(_SETTLE_SET, "stopping")
_REQUEUE_RUNNING_SQL = _task_update(
    f"{_REQUEUE_SET}, started_at=NULL, updated_at=%s", f"status='running' AND {_GENERATION_FENCE} AND {_RUN_FENCE}")
_CANCEL_QUEUED_SQL = _task_update(_CANCEL_SET, "status IN ('queued', 'deferred') AND cancel_generation=%s")
_BEGIN_STOP_SQL = _task_update(
    "status='stopping', cancel_generation=%s, cancel_id=%s, updated_at=%s",
    "status IN ('running', 'indeterminate') AND cancel_generation=%s")
_COMPLETE_STOP_SQL = _task_update(
    "status='cancelled', terminal_at=%s, updated_at=%s", "status='stopping' AND cancel_id=%s AND cancel_generation=%s")

# Lease-first recovery transitions: name -> (fenced status, SET clause, generation-guard stale message,
# row stale message); the UPDATE is _generation_update(set_clause, status).
_INDETERMINATE_STALE = "indeterminate task generation changed"
_GENERATION_TRANSITIONS = {
    "resolve": ("indeterminate", _SETTLE_SET, _INDETERMINATE_STALE, "indeterminate task changed during reconciliation"),
    "resolve_cancel": (
        "indeterminate", _CANCEL_SET, "indeterminate cancellation proof is stale",
        "indeterminate cancellation proof lost its fence"),
    "requeue": (
        "indeterminate", f"{_REQUEUE_SET}, started_at=NULL, indeterminate_at=NULL, updated_at=%s", _INDETERMINATE_STALE,
        "indeterminate task changed during requeue"),
    "defer": (
        "indeterminate", "status='deferred', result_json=%s, terminal_at=%s, updated_at=%s", _INDETERMINATE_STALE,
        "indeterminate task changed during deferral"),
    "defer_stopping": (
        "stopping", "status='deferred', result_json=%s, terminal_at=%s, updated_at=%s",
        "stopping task generation changed", "stopping task changed during deferral"),
    "reopen_deferred": (
        "deferred",
        "status='indeterminate', result_json=NULL, terminal_at=NULL, started_at=%s, indeterminate_at=%s, updated_at=%s",
        "deferred task generation changed", "deferred task changed during reopen"),
    "requeue_deferred": (
        "deferred",
        f"{_REQUEUE_SET}, result_json=NULL, started_at=NULL, terminal_at=NULL, indeterminate_at=NULL, updated_at=%s",
        "deferred task generation changed", "deferred task changed during requeue")}

Row = dict[str, Any]
Replay = Callable[[Row], "dict[str, Any] | None"]


# ── row helpers ───────────────────────────────────────────────────────────────


def _lease_from_row(row: Row, *, reclaimed: bool = False) -> DriverLease:
    return DriverLease(
        room_id=row["room_id"], gateway_id=row["gateway_id"], authority_epoch=int(row["authority_epoch"]),
        process_generation=row["process_generation"], lease_generation=int(row["lease_generation"]),
        expires_at=float(row["expires_at"]), reclaimed=reclaimed)


def _task_identity_from_row(row: Row) -> HostedTaskIdentity:
    return HostedTaskIdentity(
        room_id=row["room_id"], task_id=row["task_id"], thread_id=row["thread_id"], turn_id=row["turn_id"])


def _optional(cast: Callable[[Any], Any]) -> Callable[[Any], Any]:
    return lambda value: cast(value) if value is not None else None


# Task-view casts per row column (columns after payload_digest in _TASK_COLUMN_ORDER, same key order;
# result_json is exposed as "result"). Columns not listed are passed through untouched.
_TASK_VIEW_CASTS: dict[str, Callable[[Any], Any]] = {
    "execution_generation": int, "cancel_generation": int, "run_lease_generation": _optional(int),
    "result_json": _optional(json.loads), "created_at": float, "updated_at": float, "started_at": _optional(float),
    "terminal_at": _optional(float), "indeterminate_at": _optional(float)}


def _task_from_row(row: Row, *, idempotent: bool = False) -> dict[str, Any]:
    """The task view of one stored row; a stored payload that fails its normalization or integrity check refuses
    ``TaskConflictError``."""
    try:
        payload, encoded_payload, payload_digest = normalize_task_payload(json.loads(row["payload_json"]))
    except (TypeError, json.JSONDecodeError, DriverValidationError) as exc:
        raise TaskConflictError("stored task payload is invalid") from exc
    if (encoded_payload, payload_digest, payload["source_event_seq"]) != (
            row["payload_json"], row["payload_digest"], int(row["source_event_seq"])):
        raise TaskConflictError("stored task payload failed its integrity check")
    task: dict[str, Any] = {"identity": _task_identity_from_row(row), "payload": payload}
    for column in _TASK_COLUMN_ORDER[_TASK_COLUMN_ORDER.index("payload_digest"):]:
        task["result" if column == "result_json" else column] = _TASK_VIEW_CASTS.get(column, lambda v: v)(row[column])
    task["idempotent"] = idempotent
    return task


# The stored task of ``identity``; refuses a task id bound to another turn, and a missing task when ``required``.
def _load_task(conn, identity: HostedTaskIdentity, *, required: bool = True) -> Row | None:
    row = conn.execute(_SELECT_TASK, (identity.room_id, identity.task_id)).fetchone()
    if row is None:
        if required:
            raise TaskConflictError("task does not exist")
        return None
    if _task_identity_from_row(row) != identity:
        raise TaskConflictError("task_id is already bound to a different turn")
    return row


# The room's tasks, optionally of one status, in event order.
def _tasks_in_order(conn, room_id: str, status: str | None = None) -> list[Row]:
    where, params = ("", (room_id,)) if status is None else (" AND status = %s", (room_id, status))
    sql = f"SELECT * FROM hosted_room_driver_tasks WHERE room_id = %s{where} {_TASK_ORDER}"
    return conn.execute(sql, params).fetchall()


# The room row; refuses a missing or disbanded room.
def _load_active_room(conn, room_id: str) -> Row:
    row = conn.execute(
        "SELECT room_id, authority_gateway_id, authority_epoch, disbanded_at FROM hosted_rooms WHERE room_id = %s",
        (room_id,)).fetchone()
    if row is None or row["disbanded_at"] is not None:
        raise RoomUnavailableError("hosted room does not exist" if row is None else "hosted room is disbanded")
    return row


def _require_room_authority(conn, room_id: str, gateway_id: str, epoch: int) -> Row:
    room = _load_active_room(conn, room_id)
    if room["authority_gateway_id"] != gateway_id or int(room["authority_epoch"]) != epoch:
        raise StaleLeaseError("hosted room authority changed")
    return room


def _run_fence(lease: DriverLease) -> tuple[str, str, int]:
    """The ``_RUN_FENCE`` bind order: (gateway_id, process_generation, lease_generation)."""
    return lease.gateway_id, lease.process_generation, lease.lease_generation


def _lease_row_matches(row: Row | None, lease: DriverLease) -> bool:
    return row is not None and (
        row["gateway_id"], int(row["authority_epoch"]), row["process_generation"], int(row["lease_generation"])
    ) == (lease.gateway_id, lease.authority_epoch, lease.process_generation, lease.lease_generation)


# The lease row; refuses a changed room authority or a lease that is stale, released or (unless ``allow_expired``)
# expired.
def _require_active_lease(conn, lease: DriverLease, *, now: float, allow_expired: bool = False) -> Row:
    _require_room_authority(conn, lease.room_id, lease.gateway_id, lease.authority_epoch)
    row = conn.execute(_SELECT_LEASE, (lease.room_id,)).fetchone()
    if not _lease_row_matches(row, lease) or row["released_at"] is not None or (
        not allow_expired and float(row["expires_at"]) <= now
    ):
        raise StaleLeaseError("driver lease is stale or expired")
    return row


def _settlement(settlement_id: str, status: str, result: JsonValue, now: float) -> tuple[Replay, tuple[Any, ...]]:
    """(replay predicate, ``_SETTLE_SET`` params) of one settlement the caller has checked; the replay treats an
    identical committed settlement as idempotent and a different one as a conflict."""
    result_json = compact_json(result)

    def replay(row: Row) -> dict[str, Any] | None:
        if row["settlement_id"] is None:
            return None
        if (row["settlement_id"], row["settlement_status"], row["result_json"]) == (settlement_id, status, result_json):
            return _task_from_row(row, idempotent=True)
        raise TaskConflictError("task already has a different terminal settlement")
    return replay, (status, settlement_id, status, result_json, now, now)


def _cancel_replay(cancel_id: str, status: str = "cancelled") -> Replay:
    """Replay predicate: same cancel_id already committed in ``status``."""
    return lambda row: (
        _task_from_row(row, idempotent=True) if row["status"] == status and row["cancel_id"] == cancel_id else None)


def _generations_match(row: Row, status: str, execution_generation: int, cancel_generation: int) -> bool:
    return (row["status"], int(row["execution_generation"]), int(row["cancel_generation"])) == (
        status, execution_generation, cancel_generation)


def _require_cancel_generation(row: Row, expected_cancel_generation: int) -> None:
    if int(row["cancel_generation"]) != expected_cancel_generation:
        raise StaleTaskError("task cancellation generation changed")


class PgHostedRoomDriverRepository:
    """Driver leases, task transitions and task reads on the placement's schema."""

    def __init__(self, store: PostgresStore) -> None:
        self._store = store

    def _transition(
        self, placement: HostedPlacement, identity: HostedTaskIdentity, *, sql: str, set_params: tuple[Any, ...],
        fence_params: tuple[Any, ...], stale: str, now: float, lease: DriverLease | None = None,
        lease_first: bool = True, replay: Replay | None = None, guard: Callable[[Row], None] | None = None,
        allow_expired_lease: bool = False,
    ) -> dict[str, Any]:
        """Run one fenced task transition: load -> idempotent replay -> lease/fence guard -> UPDATE.

        ``sql`` binds ``(*set_params, room_id, task_id, *fence_params)`` and must hit exactly one row or ``stale``
        is raised. ``lease_first`` checks the lease before the row load (recovery paths) instead of after the
        replay (settlement paths: an identical replay still succeeds after the lease moved on);
        ``allow_expired_lease`` relaxes only that post-replay check and never the lease-first one.
        """
        params = (*set_params, identity.room_id, identity.task_id, *fence_params)

        def _txn(conn) -> dict[str, Any]:
            if lease is not None and lease_first:
                _require_active_lease(conn, lease, now=now)
            row = _load_task(conn, identity)
            if replay is not None and (replayed := replay(row)) is not None:
                return replayed
            if lease is not None and not lease_first:
                _require_active_lease(conn, lease, now=now, allow_expired=allow_expired_lease)
            if guard is not None:
                guard(row)
            _fenced_update(conn, sql, params, StaleTaskError(stale))
            capture_transition_locked(conn, identity.room_id)
            return _task_from_row(_load_task(conn, identity))
        with translated():
            with placement_write(self._store, placement) as conn:
                return _txn(conn)

    def _generation_transition(
        self, placement: HostedPlacement, identity: HostedTaskIdentity, lease: DriverLease, name: str,
        execution_generation: int, cancel_generation: int, *, now: float, set_params: tuple[Any, ...],
        replay: Replay | None = None,
    ) -> dict[str, Any]:
        """Lease-first transition from ``_GENERATION_TRANSITIONS`` fenced on status + both generations."""
        status, set_clause, generation_stale, stale = _GENERATION_TRANSITIONS[name]

        def guard(row: Row) -> None:
            if not _generations_match(row, status, execution_generation, cancel_generation):
                raise StaleTaskError(generation_stale)
        return self._transition(
            placement, identity, lease=lease, now=now, replay=replay, guard=guard,
            sql=_generation_update(set_clause, status), set_params=set_params,
            fence_params=(execution_generation, cancel_generation), stale=stale)

    def _run_fence_transition(
        self, placement: HostedPlacement, attempt: TaskAttempt, *, guard_stale: str,
        lease_generation: Callable[[Any], int] = int, **transition: Any,
    ) -> dict[str, Any]:
        """Transition fenced on this attempt's running generation under its exact lease (row guard + SQL fence).
        ``lease_generation`` casts the stored run_lease_generation: ``int`` raises on NULL, ``int(v or 0)`` reads 0."""
        lease = attempt.lease

        def guard(row: Row) -> None:
            if not _generations_match(row, "running", attempt.execution_generation, attempt.cancel_generation) or (
                row["run_gateway_id"], row["run_process_generation"], lease_generation(row["run_lease_generation"])
            ) != _run_fence(lease):
                raise StaleTaskError(guard_stale)
        return self._transition(
            placement, attempt.identity, lease=lease, guard=guard,
            fence_params=(attempt.execution_generation, attempt.cancel_generation, *_run_fence(lease)), **transition)

    # ── writes ──────────────────────────────────────────────────────────────

    # Takes or extends the room's lease for this process generation, reclaiming one that is expired, released or held
    # under another authority.
    def acquire_lease(self, *, placement: HostedPlacement, room_id: str, gateway_id: str, authority_epoch: int,
                      process_generation: str, ttl_seconds: float, now: float | None = None) -> DriverLease:
        now = clock(now)
        expires_at = now + float(ttl_seconds)

        def _txn(conn) -> DriverLease:
            _require_room_authority(conn, room_id, gateway_id, authority_epoch)
            row = conn.execute(_SELECT_LEASE, (room_id,)).fetchone()
            if row is None:
                conn.execute("""INSERT INTO hosted_room_driver_leases (
                           room_id, gateway_id, authority_epoch, process_generation, lease_generation,
                           expires_at, acquired_at, updated_at, released_at
                       ) VALUES (%s, %s, %s, %s, 1, %s, %s, %s, NULL)""",
                    (room_id, gateway_id, authority_epoch, process_generation, expires_at, now, now))
                return _lease_from_row(conn.execute(_SELECT_LEASE, (room_id,)).fetchone())
            same_authority = row["gateway_id"] == gateway_id and int(row["authority_epoch"]) == authority_epoch
            live = row["released_at"] is None and float(row["expires_at"]) > now
            if same_authority and row["process_generation"] == process_generation and live:
                renewed_expiry = max(float(row["expires_at"]), expires_at)
                conn.execute(
                    "UPDATE hosted_room_driver_leases SET expires_at = %s, updated_at = %s WHERE room_id = %s AND lease_generation = %s",
                    (renewed_expiry, now, room_id, int(row["lease_generation"])))
                return _lease_from_row({**row, "expires_at": renewed_expiry})
            if same_authority and live:
                raise LeaseHeldError("room driver lease is held by another generation")
            _fenced_update(conn, """UPDATE hosted_room_driver_leases
                   SET gateway_id=%s, authority_epoch=%s, process_generation=%s, lease_generation=lease_generation + 1,
                       expires_at=%s, acquired_at=%s, updated_at=%s, released_at=NULL
                   WHERE room_id=%s AND lease_generation=%s AND (
                       gateway_id != %s OR authority_epoch != %s OR released_at IS NOT NULL OR expires_at <= %s)""",
                (
                    gateway_id, authority_epoch, process_generation, expires_at, now, now, room_id,
                    int(row["lease_generation"]), gateway_id, authority_epoch, now),
                LeaseHeldError("room driver lease changed during acquisition"))
            return _lease_from_row(conn.execute(_SELECT_LEASE, (room_id,)).fetchone(), reclaimed=True)
        with translated():
            with placement_write(self._store, placement) as conn:
                return _txn(conn)

    # Extends a live lease; its expiry never moves backwards.
    def renew_lease(self, lease: DriverLease, *, placement: HostedPlacement, ttl_seconds: float,
                    now: float | None = None) -> DriverLease:
        now = clock(now)
        requested_expiry = now + float(ttl_seconds)

        def _txn(conn) -> DriverLease:
            current = _require_active_lease(conn, lease, now=now)
            expires_at = max(float(current["expires_at"]), requested_expiry)
            _fenced_update(conn, """UPDATE hosted_room_driver_leases SET expires_at=%s, updated_at=%s
                   WHERE room_id=%s AND gateway_id=%s AND process_generation=%s
                     AND lease_generation=%s AND released_at IS NULL AND expires_at > %s""",
                (expires_at, now, lease.room_id, *_run_fence(lease), now),
                StaleLeaseError("driver lease changed during renewal"))
            return dataclasses.replace(lease, expires_at=expires_at, reclaimed=False)
        with translated():
            with placement_write(self._store, placement) as conn:
                return _txn(conn)

    # Releases the lease unless a task is running; releasing it again replays.
    def release_lease(self, lease: DriverLease, *, placement: HostedPlacement, now: float | None = None) -> LeaseRelease:
        now = clock(now)

        def _txn(conn) -> LeaseRelease:
            _require_room_authority(conn, lease.room_id, lease.gateway_id, lease.authority_epoch)
            row = conn.execute(_SELECT_LEASE, (lease.room_id,)).fetchone()
            if not _lease_row_matches(row, lease):
                raise StaleLeaseError("driver lease is stale")
            if row["released_at"] is not None:
                return LeaseRelease(lease=_lease_from_row(row), idempotent=True)
            if float(row["expires_at"]) <= now:
                raise StaleLeaseError("driver lease expired before release")
            if conn.execute(
                "SELECT 1 FROM hosted_room_driver_tasks WHERE room_id = %s AND status = 'running' LIMIT 1",
                (lease.room_id,)).fetchone() is not None:
                raise InvalidTaskTransitionError("cannot release a room lease while tasks are running")
            conn.execute("""UPDATE hosted_room_driver_leases SET expires_at=%s, updated_at=%s, released_at=%s
                   WHERE room_id=%s AND lease_generation=%s""",
                (now, now, now, lease.room_id, lease.lease_generation))
            return LeaseRelease(
                lease=_lease_from_row({**row, "expires_at": now, "updated_at": now, "released_at": now}),
                idempotent=False)
        with translated():
            with placement_write(self._store, placement) as conn:
                return _txn(conn)

    # Queues a task in a live room; the same task with the same payload replays.
    def admit_task(self, identity: HostedTaskIdentity, *, placement: HostedPlacement, payload: Mapping[str, JsonValue],
                   now: float | None = None) -> Mapping[str, JsonValue]:
        normalized_payload, payload_json, payload_digest = normalize_task_payload(dict(payload))
        now = clock(now)

        def _txn(conn) -> dict[str, Any]:
            _load_active_room(conn, identity.room_id)
            existing = _load_task(conn, identity, required=False)
            if existing is not None:
                if existing["payload_digest"] != payload_digest or existing["payload_json"] != payload_json:
                    raise TaskConflictError("task_id is already bound to a different payload")
                return _task_from_row(existing, idempotent=True)
            if conn.execute(
                "SELECT * FROM hosted_room_driver_tasks WHERE room_id = %s AND thread_id = %s AND turn_id = %s",
                (identity.room_id, identity.thread_id, identity.turn_id)).fetchone() is not None:
                raise TaskConflictError("thread_id and turn_id are already bound to a task")
            conn.execute("""INSERT INTO hosted_room_driver_tasks (
                       room_id, task_id, thread_id, turn_id, source_event_seq, payload_json, payload_digest,
                       status, execution_generation, cancel_generation, created_at, updated_at
                   ) VALUES (%s, %s, %s, %s, %s, %s, %s, 'queued', 0, 0, %s, %s)""",
                (
                    *dataclasses.astuple(identity), normalized_payload["source_event_seq"], payload_json, payload_digest,
                    now, now))
            capture_transition_locked(conn, identity.room_id)
            return _task_from_row(_load_task(conn, identity))
        with translated():
            with placement_write(self._store, placement) as conn:
                return _txn(conn)

    # Starts the next queued task in event order under the lease, once no earlier task awaits recovery.
    def start_task(self, identity: HostedTaskIdentity, lease: DriverLease, *, placement: HostedPlacement,
                   expected_cancel_generation: int, now: float | None = None) -> TaskAttempt:
        now = clock(now)

        def _txn(conn) -> TaskAttempt:
            _require_active_lease(conn, lease, now=now)
            row = _load_task(conn, identity)
            _require_cancel_generation(row, expected_cancel_generation)
            if row["status"] != "queued":
                raise InvalidTaskTransitionError(f"cannot start task in state '{row['status']}'")
            if conn.execute(
                f"""SELECT task_id, status FROM hosted_room_driver_tasks
                   WHERE room_id = %s AND status IN ('running', 'indeterminate', 'stopping') {_TASK_ORDER} LIMIT 1""",
                (identity.room_id,)).fetchone() is not None:
                raise InvalidTaskTransitionError("room recovery must resolve the prior task before starting new work")
            next_queued = conn.execute(
                f"SELECT task_id FROM hosted_room_driver_tasks WHERE room_id = %s AND status = 'queued' {_TASK_ORDER} LIMIT 1",
                (identity.room_id,)).fetchone()
            if next_queued is None or next_queued["task_id"] != identity.task_id:
                raise InvalidTaskTransitionError("task is not next in the hosted room event order")
            execution_generation = int(row["execution_generation"]) + 1
            _fenced_update(conn, """UPDATE hosted_room_driver_tasks
                   SET status='running', execution_generation=%s, run_gateway_id=%s, run_process_generation=%s,
                       run_lease_generation=%s, started_at=%s, updated_at=%s
                   WHERE room_id=%s AND task_id=%s AND status='queued' AND cancel_generation=%s""",
                (
                    execution_generation, *_run_fence(lease), now, now, identity.room_id, identity.task_id,
                    expected_cancel_generation), StaleTaskError("task changed during start"))
            capture_transition_locked(conn, identity.room_id)
            return TaskAttempt(
                identity=identity, lease=lease, execution_generation=execution_generation,
                cancel_generation=expected_cancel_generation)
        with translated():
            with placement_write(self._store, placement) as conn:
                return _txn(conn)

    # Settles a running attempt; the replay check comes before the lease check, so an identical settlement replays.
    # ``allow_expired_lease`` (the live in-process settle) drops only that check's expiry term.
    def settle_task(self, attempt: TaskAttempt, *, placement: HostedPlacement, settlement_id: str,
                    status: TerminalStatus, result: JsonValue, now: float | None = None,
                    allow_expired_lease: bool = False) -> Mapping[str, JsonValue]:
        now = clock(now)
        replay, set_params = _settlement(settlement_id, status, result, now)
        return self._run_fence_transition(
            placement, attempt, guard_stale="task attempt is stale or cancelled", lease_first=False, now=now,
            replay=replay, sql=_SETTLE_RUNNING_SQL, set_params=set_params, stale="task changed during settlement",
            allow_expired_lease=allow_expired_lease)

    # Settles a stopping task; the replay check comes before the lease check.
    def settle_stopping_task(self, identity: HostedTaskIdentity, lease: DriverLease, *, placement: HostedPlacement,
                             expected_execution_generation: int, expected_cancel_generation: int, settlement_id: str,
                             status: TerminalStatus, result: JsonValue, now: float | None = None) -> Mapping[str, JsonValue]:
        now = clock(now)
        replay, set_params = _settlement(settlement_id, status, result, now)
        return self._transition(
            placement, identity, lease=lease, lease_first=False, now=now, replay=replay, sql=_SETTLE_STOPPING_SQL,
            set_params=set_params, fence_params=(expected_execution_generation, expected_cancel_generation),
            stale="task completion lost the stop race")

    # Settles an indeterminate task under the lease.
    def resolve_indeterminate_task(self, identity: HostedTaskIdentity, lease: DriverLease, *, placement: HostedPlacement,
                                   expected_execution_generation: int, expected_cancel_generation: int,
                                   settlement_id: str, status: TerminalStatus, result: JsonValue,
                                   now: float | None = None) -> Mapping[str, JsonValue]:
        now = clock(now)
        replay, set_params = _settlement(settlement_id, status, result, now)
        return self._generation_transition(
            placement, identity, lease, "resolve", expected_execution_generation, expected_cancel_generation, now=now,
            replay=replay, set_params=set_params)

    # Cancels an indeterminate task under the lease.
    def resolve_indeterminate_cancellation(self, identity: HostedTaskIdentity, lease: DriverLease, *,
                                           placement: HostedPlacement, expected_execution_generation: int,
                                           expected_cancel_generation: int, cancel_id: str,
                                           now: float | None = None) -> Mapping[str, JsonValue]:
        now = clock(now)
        return self._generation_transition(
            placement, identity, lease, "resolve_cancel", expected_execution_generation, expected_cancel_generation,
            now=now, replay=_cancel_replay(cancel_id), set_params=(expected_cancel_generation + 1, cancel_id, now, now))

    # Returns an indeterminate task to the queue under the lease.
    def requeue_indeterminate_task(self, identity: HostedTaskIdentity, lease: DriverLease, *, placement: HostedPlacement,
                                   expected_execution_generation: int, expected_cancel_generation: int,
                                   now: float | None = None) -> Mapping[str, JsonValue]:
        now = clock(now)
        return self._generation_transition(
            placement, identity, lease, "requeue", expected_execution_generation, expected_cancel_generation, now=now,
            set_params=(now,))

    # Defers an indeterminate task with a retryable reason; the same deferral replays.
    def defer_indeterminate_task(self, identity: HostedTaskIdentity, lease: DriverLease, *, placement: HostedPlacement,
                                 expected_execution_generation: int, expected_cancel_generation: int, reason: str,
                                 now: float | None = None) -> Mapping[str, JsonValue]:
        return self._defer(placement, identity, lease, "defer", expected_execution_generation,
                           expected_cancel_generation, reason=reason, now=now)

    # Defers a stopping task whose Stop its member never confirmed, with a retryable reason; the same deferral replays.
    def defer_unconfirmed_stop(self, identity: HostedTaskIdentity, lease: DriverLease, *, placement: HostedPlacement,
                               expected_execution_generation: int, expected_cancel_generation: int, reason: str,
                               now: float | None = None) -> Mapping[str, JsonValue]:
        return self._defer(placement, identity, lease, "defer_stopping", expected_execution_generation,
                           expected_cancel_generation, reason=reason, now=now)

    def _defer(self, placement: HostedPlacement, identity: HostedTaskIdentity, lease: DriverLease, name: str,
               expected_execution_generation: int, expected_cancel_generation: int, *, reason: str,
               now: float | None) -> dict[str, Any]:
        result_json = compact_json({"reason": reason, "retryable": True})
        now = clock(now)

        def replay(row: Row) -> dict[str, Any] | None:
            deferred = _generations_match(row, "deferred", expected_execution_generation, expected_cancel_generation)
            return _task_from_row(row, idempotent=True) if deferred and row["result_json"] == result_json else None
        return self._generation_transition(
            placement, identity, lease, name, expected_execution_generation, expected_cancel_generation, now=now,
            replay=replay, set_params=(result_json, now, now))

    # Returns a deferred task to the queue under the lease.
    def requeue_deferred_task(self, identity: HostedTaskIdentity, lease: DriverLease, *, placement: HostedPlacement,
                              expected_execution_generation: int, expected_cancel_generation: int,
                              now: float | None = None) -> Mapping[str, JsonValue]:
        now = clock(now)
        return self._generation_transition(
            placement, identity, lease, "requeue_deferred", expected_execution_generation, expected_cancel_generation,
            now=now, set_params=(now,))

    # Returns a deferred task to indeterminate at its same generations under the lease; its attempt clock restarts.
    def reopen_deferred_task(self, identity: HostedTaskIdentity, lease: DriverLease, *, placement: HostedPlacement,
                             expected_execution_generation: int, expected_cancel_generation: int,
                             now: float | None = None) -> Mapping[str, JsonValue]:
        now = clock(now)
        return self._generation_transition(
            placement, identity, lease, "reopen_deferred", expected_execution_generation, expected_cancel_generation,
            now=now, set_params=(now, now, now))

    # Requeues a running attempt that was not admitted, under its exact lease; a NULL run lease generation reads 0.
    def requeue_not_admitted_task(self, attempt: TaskAttempt, *, placement: HostedPlacement,
                                  now: float | None = None) -> Mapping[str, JsonValue]:
        now = clock(now)

        def replay(row: Row) -> dict[str, Any] | None:
            requeued = _generations_match(row, "queued", attempt.execution_generation, attempt.cancel_generation) and (
                row["run_gateway_id"], row["run_process_generation"], row["run_lease_generation"]) == (None, None, None)
            return _task_from_row(row, idempotent=True) if requeued else None
        return self._run_fence_transition(
            placement, attempt, guard_stale="not-admitted task attempt lost its fence",
            lease_generation=lambda value: int(value or 0), now=now, replay=replay, sql=_REQUEUE_RUNNING_SQL,
            set_params=(now,), stale="not-admitted task changed during requeue")

    # Cancels a queued or deferred task; running work goes through ``begin_task_cancel``.
    def cancel_task(self, identity: HostedTaskIdentity, *, placement: HostedPlacement, cancel_id: str,
                    expected_cancel_generation: int, now: float | None = None) -> Mapping[str, JsonValue]:
        now = clock(now)

        def guard(row: Row) -> None:
            if row["status"] in TERMINAL_STATUSES:
                raise InvalidTaskTransitionError(f"cannot cancel task in state '{row['status']}'")
            if row["status"] not in {"queued", "deferred"}:
                raise InvalidTaskTransitionError("running work requires acknowledged two-phase cancellation")
            _require_cancel_generation(row, expected_cancel_generation)
        return self._transition(
            placement, identity, now=now, replay=_cancel_replay(cancel_id), guard=guard, sql=_CANCEL_QUEUED_SQL,
            set_params=(expected_cancel_generation + 1, cancel_id, now, now), fence_params=(expected_cancel_generation,),
            stale="task changed during cancellation")

    # Moves a running or indeterminate task to ``stopping`` under the next cancel generation.
    def begin_task_cancel(self, identity: HostedTaskIdentity, *, placement: HostedPlacement, cancel_id: str,
                          expected_cancel_generation: int, now: float | None = None) -> Mapping[str, JsonValue]:
        now = clock(now)

        def guard(row: Row) -> None:
            if row["status"] in TERMINAL_STATUSES or row["status"] == "queued":
                raise InvalidTaskTransitionError(f"cannot request remote stop in state '{row['status']}'")
            _require_cancel_generation(row, expected_cancel_generation)
        return self._transition(
            placement, identity, now=now, replay=_cancel_replay(cancel_id, "stopping"), guard=guard, sql=_BEGIN_STOP_SQL,
            set_params=(expected_cancel_generation + 1, cancel_id, now), fence_params=(expected_cancel_generation,),
            stale="task changed during stop request")

    # Cancels a stopping task once its stop is acknowledged.
    def complete_task_cancel(self, identity: HostedTaskIdentity, *, placement: HostedPlacement, cancel_id: str,
                             expected_cancel_generation: int, now: float | None = None) -> Mapping[str, JsonValue]:
        now = clock(now)

        def guard(row: Row) -> None:
            if (row["status"], row["cancel_id"], int(row["cancel_generation"])) != (
                    "stopping", cancel_id, expected_cancel_generation):
                raise StaleTaskError("task stop acknowledgement is stale")
        return self._transition(
            placement, identity, now=now, replay=_cancel_replay(cancel_id), guard=guard, sql=_COMPLETE_STOP_SQL,
            set_params=(now, now), fence_params=(cancel_id, expected_cancel_generation),
            stale="task changed during stop acknowledgement")

    # Marks exactly this running attempt indeterminate under an active lease of the room, whichever lease ran it;
    # an attempt no longer running at its generations is left as it is.
    def mark_attempt_indeterminate(self, attempt: TaskAttempt, lease: DriverLease, *, placement: HostedPlacement,
                                   now: float | None = None) -> None:
        now = clock(now)
        identity = attempt.identity

        def _txn(conn) -> None:
            _require_active_lease(conn, lease, now=now)
            row = _load_task(conn, identity)
            if not _generations_match(row, "running", attempt.execution_generation, attempt.cancel_generation):
                return
            conn.execute(
                _generation_update("status='indeterminate', indeterminate_at=%s, updated_at=%s", "running"),
                (now, now, identity.room_id, identity.task_id, attempt.execution_generation, attempt.cancel_generation))
            capture_transition_locked(conn, lease.room_id)
        with translated():
            with placement_write(self._store, placement) as conn:
                _txn(conn)

    # Marks running tasks of other leases indeterminate and lists the room's queued and indeterminate tasks.
    def recover_room(self, lease: DriverLease, *, placement: HostedPlacement, now: float | None = None) -> RoomRecovery:
        now = clock(now)
        foreign_running = f"room_id = %s AND status = 'running' AND NOT ({_RUN_FENCE})"
        fence = (lease.room_id, *_run_fence(lease))

        def _txn(conn) -> RoomRecovery:
            _require_active_lease(conn, lease, now=now)
            stale_rows = conn.execute(
                f"SELECT * FROM hosted_room_driver_tasks WHERE {foreign_running} {_TASK_ORDER}", fence).fetchall()
            if stale_rows:
                conn.execute(
                    f"""UPDATE hosted_room_driver_tasks SET status='indeterminate', indeterminate_at=%s, updated_at=%s
                        WHERE {foreign_running}""", (now, now, *fence))
            capture_transition_locked(conn, lease.room_id)
            queued, indeterminate = (
                tuple(_task_identity_from_row(row) for row in _tasks_in_order(conn, lease.room_id, status))
                for status in ("queued", "indeterminate"))
            return RoomRecovery(queued=queued, indeterminate=indeterminate)
        with translated():
            with placement_write(self._store, placement) as conn:
                return _txn(conn)

    # Deletes published terminal tasks past retention or beyond the newest ``retain``, at most one batch per call.
    def prune_published_terminal_tasks(self, *, placement: HostedPlacement, room_id: str, now: float | None = None,
                                       retention_seconds: float = TERMINAL_TASK_RETENTION_SECONDS,
                                       retain: int = MAX_RETAINED_TERMINAL_TASKS) -> int:
        now = clock(now)

        def _txn(conn) -> int:
            # The publications table is in the declarative DDL; the reference's files-only check reads true on a
            # schema the DDL created, so the published-terminal filter is always live.
            if not table_exists(conn, "hosted_room_policy_publications"):
                return 0
            rows = conn.execute("""SELECT t.task_id, t.terminal_at FROM hosted_room_driver_tasks t
                    WHERE t.room_id = %s AND t.status IN ('settled', 'failed', 'cancelled')
                      AND EXISTS (SELECT 1 FROM hosted_room_policy_publications p
                                  WHERE p.room_id = t.room_id AND p.task_id = t.task_id
                                    AND p.kind IN ('turn.settled', 'turn.failed', 'turn.cancelled'))
                    ORDER BY t.terminal_at DESC, t.task_id ASC""", (room_id,)).fetchall()
            cutoff = now - float(retention_seconds)
            candidates = [
                str(row["task_id"]) for index, row in enumerate(rows)
                if index >= retain or (row["terminal_at"] is not None and float(row["terminal_at"]) <= cutoff)
            ][:MAX_TASK_PRUNE_BATCH]
            if not candidates:
                return 0
            deleted = conn.execute(
                f"DELETE FROM hosted_room_driver_tasks WHERE room_id = %s AND task_id = ANY(%s)",
                (room_id, list(candidates)))
            return max(0, int(deleted.rowcount))
        with translated():
            with placement_write(self._store, placement) as conn:
                return _txn(conn)

    # ── reads ───────────────────────────────────────────────────────────────

    def get_task(self, identity: HostedTaskIdentity, *, placement: HostedPlacement) -> Mapping[str, JsonValue]:
        def _read(conn) -> dict[str, Any]:
            return _task_from_row(_load_task(conn, identity))
        with translated():
            with placement_read(self._store, placement) as conn:
                return _read(conn)

    def list_tasks(self, *, placement: HostedPlacement, room_id: str,
                   status: TaskStatus | None = None) -> tuple[Mapping[str, JsonValue], ...]:
        def _read(conn) -> tuple[dict[str, Any], ...]:
            return tuple(_task_from_row(row) for row in _tasks_in_order(conn, room_id, status))
        with translated():
            with placement_read(self._store, placement) as conn:
                return _read(conn)

    def room_task_summary(self, *, placement: HostedPlacement, room_id: str,
                          statuses: Collection[str]) -> tuple[Mapping[str, int], tuple[Mapping[str, JsonValue], ...]]:
        statuses = tuple(statuses)

        def _read(conn) -> tuple[dict[str, int], tuple[dict[str, Any], ...]]:
            # One read transaction (the placement's read door) so both reads see one generation: a task moving
            # between them is neither doubled nor lost.
            counts = {str(row["status"]): int(row["count"]) for row in conn.execute(
                "SELECT status, COUNT(*) AS count FROM hosted_room_driver_tasks WHERE room_id = %s GROUP BY status",
                (room_id,))}
            rows = conn.execute(
                f"""SELECT * FROM hosted_room_driver_tasks
                    WHERE room_id = %s AND status IN ({','.join('%s' for _ in statuses)}) {_TASK_ORDER}""",
                (room_id, *statuses)).fetchall() if statuses else []
            return counts, tuple(_task_from_row(row) for row in rows)
        with translated():
            with placement_read(self._store, placement) as conn:
                return _read(conn)

    def room_change_rows(self, *, placement: HostedPlacement,
                         statuses: Collection[str] = SHOWN_TASK_STATUSES) -> Mapping[str, RoomChangeRows]:
        statuses = tuple(statuses)

        # Each room's rows are read through the status index (``room_id, status`` prefix) and sorted here: an
        # ``ORDER BY task_id`` would make the database walk the room's retained terminal tasks through the
        # primary key.
        def _read() -> dict[str, RoomChangeRows]:
            rooms = placement_all(
                self._store, placement,
                "SELECT room_id, next_seq, revision FROM hosted_rooms WHERE disbanded_at IS NULL")
            if not rooms:
                return {}
            query = f"""SELECT task_id, status, execution_generation, cancel_generation FROM hosted_room_driver_tasks
                        WHERE room_id = %s AND status IN ({','.join('%s' for _ in statuses)})"""
            return {str(row["room_id"]): RoomChangeRows(
                        int(row["next_seq"]), int(row["revision"]),
                        tuple((r["task_id"], r["status"], r["execution_generation"], r["cancel_generation"])
                              for r in placement_all(self._store, placement, query, row["room_id"], *statuses)))
                    for row in rooms}
        with translated():
            return _read()

    # The task bound to this thread and turn, or None.
    def get_task_for_turn(self, identity: HostedTaskIdentity, *,
                         placement: HostedPlacement) -> Mapping[str, JsonValue] | None:
        def _read(conn) -> dict[str, Any] | None:
            row = conn.execute(
                """SELECT * FROM hosted_room_driver_tasks
                   WHERE room_id = %s AND thread_id = %s AND turn_id = %s""",
                (identity.room_id, identity.thread_id, identity.turn_id),
            ).fetchone()
            return _task_from_row(row) if row is not None else None
        with translated():
            with placement_read(self._store, placement) as conn:
                return _read(conn)


def _register() -> None:
    register_repository("hosted_room_driver", lambda store: PgHostedRoomDriverRepository(store))


_register()

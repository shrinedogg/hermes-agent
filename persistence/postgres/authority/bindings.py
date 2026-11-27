"""``SessionBindingRepository`` on PostgreSQL (stage 2): local creation receipts, API bindings and the local
lineage reads.

Each verb converts its typed parameters to the reference body's arguments, runs one body (one
``store.tx()`` write, or one read) inside :func:`_refused`, and converts the result to the contract
record. The bodies are the SQLite authority store's SQL (``persistence.sqlite.authority``) ported
inline onto the PG dialect: the state_meta key prefixes, the pure guard/owner helpers and the receipt
codecs are imported from the shared modules, so the two backends cannot drift; the session-lineage
bodies (``_validate_local_lineage``, ``_advance_local_target``, ``_publish_child_session_row``,
``_commit_local_target``) and the gateway hand-off bodies (``_moved_receipts``,
``_hand_off_conversations``) reuse the PG conversation/gateway stores' own connection-level helpers,
so a reservation write composes over the same rows the conversation family writes.

The chat id a reservation stamps on the sessions row is spelled as the reference's SQLite driver
spells a bound value in a TEXT column (integers as their text, a REAL as the installed SQLite
library's spelling of it, a bool as ``1``/``0``); an out-of-range integer refuses with the driver's
``OverflowError`` and a non-scalar with the store's permanent error, so the reservation writes
nothing either way (the reference's bind failure).
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from contextlib import closing, contextmanager
from datetime import datetime
from functools import partial
from typing import Any, Iterator, Mapping, Optional

from persistence import (
    AuthorityRefusal,
    FenceStale,
    PermanentStoreError,
    ProofRejected,
    SessionId,
    StaleFenceKind,
    UnknownExecution,
)
from persistence.authority.admissions import AdmissionGate, raise_gate
from persistence.authority.bindings import (
    ApiBinding,
    BindingMarks,
    LegacyAdoptionScan,
    LocalLineage,
    NewApiBinding,
    NewSessionBinding,
    ReservedBinding,
    SessionBinding,
    SessionBindingRepository,
    api_generation_id,
)
from persistence.authority.runtime import HistoryActor
from persistence.authority.vocabulary import LEGACY_LOCAL_SOURCES, OwnerRule
from persistence.core.canonical import canonical_json
from persistence.sqlite.authority._binding_history import BINDING_PREFIX as HISTORY_BINDING_PREFIX
from persistence.sqlite.conversation._store.local import (
    API_BINDING_PREFIX,
    API_DECLARED_PREFIX,
    POLICY_PREFIX,
)
from persistence.sqlite.conversation._store.local_migration import LEGACY_PREFIX
from persistence.sqlite.conversation._store.retirement import RETIRED_PREFIX, VIEWER_CLOSE_PREFIX
from session_rules.legacy_local import (
    binding_names,
    client_history_row,
    legacy_account_label,
    legacy_history_row,
    login_label,
    not_client_labels,
    same_owner,
    validate_legacy_row,
)
from session_rules.local_create import same_create_retry

from .._backend import PostgresStore, refusals, register_repository
from .._stubs import add_stubs
from ..conversation._store.common import _check_transcript_write_guards
from ..conversation.sessions import (
    _CHAIN_STEP_SQL,
    _bump_conversation_generation,
    _end_and_bump,
    _get_compression_tip,
    _local_conversation_ids,
    _local_lineage_owner,
    _policy_target,
    _session_turn_lease_key_on_conn,
    _walk_compression_chain,
)
from ..gateway.routing import _hand_off_conversations, _moved_receipts
from .runtime import _validate_local_lineage

__all__ = ["PgBindingsRepository"]

# Main's logger name for API route warnings.
logger = logging.getLogger("gateway.session_api")

_FENCE_KINDS = {kind.value: kind for kind in StaleFenceKind}


def _typed(exc: AuthorityRefusal) -> AuthorityRefusal:
    """The contract subclass for a fence, unknown-execution or proof refusal, ``exc`` itself for any
    other code (the reference's ``_typed``)."""
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
    refusal classes, and the store's failures (the PG ``refusals``: ``translated`` plus the session
    error mapping - a live turn lease answers as the contract's ``TurnLeaseLost``) as the
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


# ── the reservation's shared steps (the reference's ``_local``) ─────────────────────────────────


def _require_epoch(conn, epoch: Any) -> None:
    """The epoch fence (the reference's ``_epoch``): the stored singleton epoch, else ``stale_epoch``
    - a non-int epoch (a bool or float included), a never-issued zero and any mismatch fence."""
    row = conn.execute("SELECT epoch FROM runtime_epoch WHERE singleton = 1").fetchone()
    if type(epoch) is not int or row is None or row["epoch"] != epoch:
        raise AuthorityRefusal("stale_epoch")


def _new_receipt(binding: NewSessionBinding) -> dict:
    """The receipt dict the reservation stores (``lineage`` and ``legacy_session_id`` only when set)."""
    receipt = {"profile_id": binding.profile_id, "principal_id": binding.principal_id, "request_id": binding.request_id,
               "session_id": binding.session_id, "route": binding.route, "entry": dict(binding.entry),
               "policy": dict(binding.policy)}
    if binding.legacy_session_id is not None:
        receipt["legacy_session_id"] = binding.legacy_session_id
    if binding.lineage is not None:
        receipt["lineage"] = list(binding.lineage)
    return receipt


def _binding_record(receipt: dict) -> SessionBinding:
    """The stored receipt as a detached ``SessionBinding``; a receipt missing an identity field is
    ``storage_unavailable`` (the reference's ``binding_record``)."""
    try:
        lineage = receipt.get("lineage", [receipt["session_id"]])
        return SessionBinding(
            profile_id=receipt["profile_id"], principal_id=receipt["principal_id"], request_id=receipt["request_id"],
            session_id=SessionId(receipt["session_id"]), route=receipt["route"], entry=receipt["entry"],
            policy=receipt["policy"], lineage=tuple(SessionId(sid) for sid in lineage),
            legacy_session_id=receipt.get("legacy_session_id"))
    except (AttributeError, TypeError, KeyError) as exc:
        raise AuthorityRefusal("storage_unavailable") from exc


def _receipt_of(binding: SessionBinding) -> dict:
    """The receipt dict a ``SessionBinding`` was read from (``lineage`` present unless it is the
    default) (the reference's ``receipt_of``)."""
    receipt = {"profile_id": binding.profile_id, "principal_id": binding.principal_id,
               "request_id": binding.request_id, "session_id": binding.session_id, "route": binding.route,
               "entry": dict(binding.entry), "policy": dict(binding.policy)}
    if binding.legacy_session_id is not None:
        receipt["legacy_session_id"] = binding.legacy_session_id
    if binding.lineage != (binding.session_id,):
        receipt["lineage"] = list(binding.lineage)
    return json.loads(json.dumps(receipt))


def _row_cwd(policy: dict):
    """The sessions-row ``cwd``: the create request's ``row_cwd`` (``""`` = none) when it sent one,
    else the execution cwd (the reference's ``_row_cwd``)."""
    try:
        request = json.loads(policy.get("request_json") or "{}")
    except (TypeError, ValueError):
        request = {}
    value = request.get("row_cwd") if isinstance(request, dict) else None
    return policy["cwd"] if not isinstance(value, str) else (value or None)


def _bindable_chat_id(value: Any) -> Optional[str]:
    """The row's ``chat_id`` cell as the reference's SQLite driver spells a bound value in a TEXT
    column: an INTEGER as its text (int64 in range; an out-of-range integer is the driver's
    ``OverflowError``), a REAL as the installed SQLite library's spelling of it (the parity the
    contract's ``real_as_text`` asks), a bool as ``1``/``0`` and a missing/NaN value as NULL; a
    non-scalar is the store's permanent error, so the reservation writes nothing either way."""
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, int):
        if -(2 ** 63) <= value < 2 ** 63:
            return str(value)
        raise OverflowError(f"Python int too large to convert to SQLite INTEGER: {value}")
    if isinstance(value, float):
        if value != value:  # NaN binds as NULL, as the reference's driver does
            return None
        with closing(sqlite3.connect(":memory:")) as conn:
            conn.execute("CREATE TABLE spelled (value TEXT)")
            conn.execute("INSERT INTO spelled (value) VALUES (?)", (value,))
            return conn.execute("SELECT value FROM spelled").fetchone()[0]
    raise PermanentStoreError(
        f"chat_id value of type {type(value).__name__} cannot be stored in a TEXT column")


def _require_not_executing(conn, session_ids) -> None:
    """Refuse while a turn is running (or its outcome is unknown) on any of *session_ids* (the
    reference's ``require_not_executing``): queued admissions are allowed, any non-terminal worker
    is executing."""
    for sid in session_ids:
        admissions = conn.execute(
            "SELECT status FROM session_admissions WHERE target_session_id = %s AND status IN ('started','unknown')",
            (sid,)).fetchall()
        workers = conn.execute(
            "SELECT status FROM worker_executions WHERE session_id = %s AND status IN ('registered','running','unknown')",
            (sid,)).fetchall()
        states = {row["status"] for row in [*admissions, *workers]}
        if "unknown" in states:
            raise AuthorityRefusal("unknown_execution")
        if states:
            raise AuthorityRefusal("session_busy")


def _require_transcript_unleased(conn, sid: str) -> None:
    """A logical owner closed by compression is an ancestor, not a transcript target; its live
    successor carries the lease/lock. A live row must be unleased (the reference's
    ``_require_transcript_unleased``): a live turn lease answers as the guard's ``TurnLeaseLost``,
    a live compression lock as the compression-in-progress error; stale guards are reclaimed."""
    row = conn.execute("SELECT ended_at, end_reason FROM sessions WHERE id = %s", (sid,)).fetchone()
    if row is not None and row["ended_at"] is not None and row["end_reason"] == "compression":
        return
    _check_transcript_write_guards(conn, sid, None,
                                   reject_active_turn_lease=True, reject_active_compression_lock=True)


def _require_idle(conn, session_ids) -> None:
    """Refuse while the ledger holds any unsettled work on *session_ids* (queued too), else as
    :func:`_require_transcript_unleased` (the reference's ``require_idle``)."""
    for sid in session_ids:
        admissions = conn.execute(
            "SELECT status FROM session_admissions WHERE target_session_id = %s AND status IN ('queued','started','unknown')",
            (sid,)).fetchall()
        workers = conn.execute(
            "SELECT status FROM worker_executions WHERE session_id = %s AND status IN ('registered','running','unknown')",
            (sid,)).fetchall()
        states = {row["status"] for row in [*admissions, *workers]}
        if "unknown" in states:
            raise AuthorityRefusal("unknown_execution")
        if states:
            raise AuthorityRefusal("session_busy")
        _require_transcript_unleased(conn, sid)


def _keep_passed_fences(conn, derive, session_id: str) -> bool:
    """Make permanent the retirement fences a generation walk passed on its way to *session_id*,
    inside the write that binds *session_id* (the reference's ``keep_passed_fences``): a passed
    generation that has lost its marker since the walk answers False, so the successor must not be
    bound."""
    generation = 0
    while (sid := derive(generation)) != session_id:
        if conn.execute("SELECT 1 FROM state_meta WHERE key = %s", (RETIRED_PREFIX + sid,)).fetchone() is None:
            return False
        # ``json_remove(value, '$.expires')`` on the PG dialect: drop the sweep half of the marker
        # (a permanent fence never ages out); the guard leaves an undecodable value untouched.
        conn.execute(
            "UPDATE state_meta SET value = (value::jsonb - 'expires')::text "
            "WHERE key = %s AND hermes_json_valid(value)", (RETIRED_PREFIX + sid,))
        generation += 1
    return True


def _advance_local_target(store: PostgresStore, conn, parent_session_id: str, child_session_id: str,
                         *, entry: Optional[dict] = None) -> None:
    """Advance the local creation receipt's target to the published child (the reference's
    ``advance_local_target``): the receipt's entry and the route's entry move to the child and the
    lineage appends it, in the store's commit; *entry* (the reset verb's) wins over the route's
    stored entry; a store without the receipt, or a receipt that no longer names this parent, is left
    alone and never blocks the publish."""
    parent = conn.execute("SELECT * FROM sessions WHERE id = %s", (parent_session_id,)).fetchone()
    logical_id = parent["chat_id"] if parent else None
    if not logical_id:
        return
    key = POLICY_PREFIX + logical_id
    raw = conn.execute("SELECT value FROM state_meta WHERE key = %s", (key,)).fetchone()
    if raw is None:
        legacy = conn.execute("SELECT 1 FROM state_meta WHERE key = %s",
                              (LEGACY_PREFIX + logical_id,)).fetchone()
        if logical_id.startswith("local-") or legacy:
            raise AuthorityRefusal("storage_unavailable")
        return
    try:
        receipt = json.loads(raw["value"])
        if (receipt["session_id"] != logical_id or receipt["entry"]["session_id"] != parent_session_id
                or receipt["principal_id"] != parent["user_id"] or receipt["route"] != parent["session_key"]):
            raise AuthorityRefusal("storage_unavailable")
        routed = _routed_entry_on(conn, receipt["route"], parent_session_id)
        if routed is None:
            raise AuthorityRefusal("admission_conflict")
        candidate = dict(entry or routed, session_id=child_session_id)
        receipt["entry"] = candidate
        receipt["lineage"] = [*receipt.get("lineage", [logical_id]), child_session_id]
        _validate_local_lineage(conn, receipt)
    except (KeyError, TypeError, ValueError) as exc:
        if isinstance(exc, AuthorityRefusal):
            raise
        raise AuthorityRefusal("storage_unavailable") from exc
    store.lineage._commit_local_target(conn, receipt, candidate)


def _heal_replacement(conn, caller: str, holder: str) -> bool:
    """Whether the route's holder is the stale-route heal's replacement (a session with no binding of
    its own): the caller takes the route back, used or not, and the replacement keeps its transcript
    under its own id (the reference's ``_heal_replacement``)."""
    if conn.execute("SELECT 1 FROM state_meta WHERE key = %s", (API_BINDING_PREFIX + holder,)).fetchone():
        return False
    logger.warning("API route of %s taken back from heal replacement %s", caller, holder)
    return True


def _routed_entry_on(conn, route: str, tip: str) -> Optional[dict]:
    """:data:`route`'s routing entry while it still targets *tip*, else None (the store's
    ``_routed_entry``)."""
    row = conn.execute("SELECT entry_json FROM gateway_routing WHERE scope = '' AND session_key = %s",
                      (route,)).fetchone()
    routed = json.loads(row["entry_json"]) if row is not None else None
    return routed if (isinstance(routed, dict) and routed.get("session_id") == tip) else None


def _untracked_steps(conn, tip: str, lineage: list) -> Optional[list]:
    """Every compression step past *tip*, in order; None when the chain leads back into itself (the
    reference's ``_untracked_steps``: one capped walk at a time, continuing from the last step)."""
    steps, seen, current = [], {*lineage, tip}, tip
    while True:
        batch = _walk_compression_chain(conn, current)[1:]
        if not batch:
            return steps
        if seen.intersection(batch):
            return None
        seen.update(batch)
        steps.extend(batch)
        current = batch[-1]


def _adoptable_compression(conn, receipt: dict) -> Optional[tuple]:
    """``(steps, routed)``: the untracked compression steps *receipt* would adopt and its route's
    entry, else None (the reference's ``adoptable_compression``): a step must carry the receipt's
    route, owner and principal, and the route must still target the tip."""
    tip = receipt["entry"]["session_id"]
    children = _untracked_steps(conn, tip, receipt.get("lineage", [receipt["session_id"]]))
    if not children:
        return None
    identity = (receipt["route"], receipt["session_id"], receipt["principal_id"])
    for child_id in children:
        child = conn.execute("SELECT session_key, chat_id, user_id FROM sessions WHERE id = %s",
                            (child_id,)).fetchone()
        if (child["session_key"], child["chat_id"], child["user_id"]) != identity:
            return None
    routed = _routed_entry_on(conn, receipt["route"], tip)
    return (children, routed) if routed is not None else None


def _with_adopted_compression(receipt: dict, children: list) -> dict:
    """"*receipt* as ``adopt_untracked_compression`` commits it: *children* appended to its lineage
    and its entry moved to the last one (the reference's ``with_adopted_compression``)."""
    return dict(receipt, entry=dict(receipt["entry"], session_id=children[-1]),
                lineage=[*receipt.get("lineage", [receipt["session_id"]]), *children])


# ── the history-control ownership (the reference's ``_binding_history``) ────────────────────────


def _history_row(row: dict, owner: Optional[OwnerRule]) -> bool:
    """Local surfaces' unrouted rows are owner history (the reference's ``_history_row``): every row
    the native resume adopts plus imports."""
    return row["source"] == "import" or legacy_history_row(dict(row), owner)


def _owner_user(row: dict):
    """The row's owner identity; main's Desktop Nous-account label is not one (the reference's
    ``_owner_user``)."""
    return None if legacy_account_label(dict(row)) else row["user_id"]


def _require_history_claim(conn, store: PostgresStore, actor: HistoryActor, session_id: str,
                          native_transport: bool, owner: Optional[OwnerRule]) -> None:
    """Import receipts confer ownership; merely knowing an old ID does not (the reference's
    ``require_history_claim``): *native_transport* is whether the owner's native transport registry
    holds the actor."""
    row = conn.execute("SELECT * FROM sessions WHERE id = %s", (session_id,)).fetchone()
    if row is None or row["session_key"] or row["chat_id"] or row["origin_json"]:
        return
    if conn.execute("SELECT 1 FROM state_meta WHERE key = %s",
                   (HISTORY_BINDING_PREFIX + session_id,)).fetchone():
        return  # the stored binding names the committed importer; the read re-checks it
    own_profile = row["profile_name"] in (None, "", store.owner_profile_name or "default")
    if own_profile and _owner_user(row) and same_owner(row["user_id"], actor.subject, owner):
        return  # its owner (a login stamp is the machine owner's); the read re-checks it
    if not native_transport:
        raise AuthorityRefusal("permission_denied")


def _authorize_history(conn, actor: HistoryActor, session_id: str, *, owner: Optional[OwnerRule] = None) -> bool:
    """The history authorization (the reference's ``authorize_history``): a routed or missing row is
    not history; a history-control record decides alone; otherwise the local-history rule, the
    stored owner test and the create capability."""
    row = conn.execute("SELECT * FROM sessions WHERE id = %s", (session_id,)).fetchone()
    if row is None:
        return False
    if row["session_key"] or row["chat_id"] or row["origin_json"]:
        return False
    key = HISTORY_BINDING_PREFIX + session_id
    binding = json.dumps([actor.profile_id, actor.subject], separators=(",", ":"))
    saved = conn.execute("SELECT value FROM state_meta WHERE key = %s", (key,)).fetchone()
    if saved is not None:
        if not binding_names(saved["value"], actor.subject, owner):
            raise AuthorityRefusal("permission_denied")
        return True
    if not _history_row(row, owner):
        return False
    if _owner_user(row) and not same_owner(row["user_id"], actor.subject, owner):
        raise AuthorityRefusal("permission_denied")
    if not actor.may_create:
        raise AuthorityRefusal("permission_denied")
    return True


# ── the legacy adoption arm (the reference's ``_local_migration``) ───────────────────────────────


def _bind_legacy_target(conn, store: PostgresStore, receipt: dict, owner: Optional[OwnerRule]) -> None:
    """The legacy adoption arm inside the reservation's write (the reference's ``bind_legacy_target``):
    the unrouted pre-authority row is bound to its new receipt - it keeps its source and end stamp and
    gains the route, chat id, principal and origin, and the legacy binding record is stored."""
    target = receipt["legacy_session_id"]
    if target != receipt["session_id"]:
        raise AuthorityRefusal("storage_unavailable")
    row = conn.execute("SELECT * FROM sessions WHERE id = %s", (target,)).fetchone()
    if row is None:
        raise AuthorityRefusal("not_found")
    validate_legacy_row(store.owner_profile_name, dict(row), subject=receipt["principal_id"], rule=owner)
    imported = conn.execute("SELECT value FROM state_meta WHERE key = %s",
                           (HISTORY_BINDING_PREFIX + target,)).fetchone()
    if imported is not None and not binding_names(imported["value"], receipt["principal_id"], owner):
        raise AuthorityRefusal("permission_denied")
    if conn.execute(_CHAIN_STEP_SQL, (target,)).fetchone() is not None:
        raise AuthorityRefusal("admission_conflict")
    _require_idle(conn, [target])
    root = _session_turn_lease_key_on_conn(conn, target)
    if conn.execute(
            "SELECT 1 FROM session_turn_leases WHERE conversation_id = %s AND expires_at > %s",
            (root, time.time())).fetchone():
        raise AuthorityRefusal("runtime_coordination_required")
    key = LEGACY_PREFIX + target
    if conn.execute("SELECT 1 FROM state_meta WHERE key = %s", (key,)).fetchone():
        raise AuthorityRefusal("storage_unavailable")
    source = receipt["entry"]["origin"]
    # Binding is a read-side claim: the row keeps main's ``source`` label and its end stamp. A client
    # app's own label is the exception: bound, the row has a route, so the label alone would read it as
    # a messaging chat; it becomes the Desktop chat it resumes as (``created_source`` keeps provenance).
    label = "desktop" if client_history_row(dict(row), owner) else row["source"]
    conn.execute(
        "UPDATE sessions SET source = %s, session_key = %s, chat_id = %s, user_id = %s, chat_type = 'dm', "
        "origin_json = %s WHERE id = %s",
        (label, receipt["route"], receipt["session_id"], receipt["principal_id"], canonical_json(source), target))
    conn.execute("INSERT INTO state_meta(key, value) VALUES (%s, %s)",
                 (key, canonical_json({
                     "session_id": receipt["session_id"], "profile_id": receipt["profile_id"],
                     "principal_id": receipt["principal_id"], "legacy_session_id": target})))


class PgBindingsRepository:
    """Local creation receipts, API bindings and the lineage reads of the bound profile's store schema;
    every protocol verb ported from the SQLite reference."""

    def __init__(self, store: PostgresStore) -> None:
        self._store = store

    # ── writes ────────────────────────────────────────────────────────────────

    def reserve(self, *, epoch: int, binding: NewSessionBinding, client_request: Mapping[str, Any] | None = None,
                owner: OwnerRule | None) -> ReservedBinding:
        """Reserve the row, the route and the receipt in one write (the reference's
        ``commit_local_session``): an exact duplicate replays the stored receipt with ``created``
        False, else the receipt this call inserted with True; the checks run in order and every
        refusal writes nothing."""
        request = None if client_request is None else dict(client_request)
        receipt = _new_receipt(binding)
        if request is None and receipt["policy"].get("client_env_json") is not None:
            # Lost (a copy): never judge a retry by this attempt's own filtered map instead.
            raise AuthorityRefusal("invalid_params")
        if request is None:
            request = {}
        encoded = canonical_json(receipt)  # encoding precedes the epoch fence
        receipt = json.loads(encoded)
        sid, route = receipt["session_id"], receipt["route"]
        key = POLICY_PREFIX + sid
        with _refused(self._store):
            with self._store.tx() as conn:
                _require_epoch(conn, epoch)
                old = conn.execute("SELECT value FROM state_meta WHERE key = %s", (key,)).fetchone()
                if old is not None:
                    try:
                        saved = json.loads(old["value"])
                        same = all(saved[k] == receipt[k] for k in (
                            "profile_id", "principal_id", "request_id", "session_id", "route"))
                        same = same and same_create_retry(saved["policy"], receipt["policy"]["request_json"], request)
                    except (ValueError, TypeError, KeyError) as exc:
                        raise AuthorityRefusal("storage_unavailable") from exc
                    if not same:
                        raise AuthorityRefusal("invalid_params")
                    return ReservedBinding(binding=_binding_record(saved), created=False)
                if conn.execute("SELECT 1 FROM state_meta WHERE key = %s", (RETIRED_PREFIX + sid,)).fetchone():
                    # Deleted: its receipt was retired with the transcript; a create/cron retry must not
                    # resurrect the row (and re-fire its prompt) under the same deterministic id.
                    raise AuthorityRefusal("not_found")
                if "legacy_session_id" not in receipt and conn.execute(
                        "SELECT 1 FROM sessions WHERE id = %s", (sid,)).fetchone():
                    raise AuthorityRefusal("storage_unavailable")
                if conn.execute("SELECT 1 FROM gateway_routing WHERE scope = '' AND session_key = %s",
                               (route,)).fetchone():
                    raise AuthorityRefusal("admission_conflict")
                if "legacy_session_id" in receipt:
                    _bind_legacy_target(conn, self._store, receipt, owner)
                policy, entry = receipt["policy"], receipt["entry"]
                started = datetime.fromisoformat(entry["created_at"]).timestamp()
                # The surface is stamped here: an auxiliary usage insert created the row as 'unknown' and
                # its DO NOTHING upsert does not repair it.
                created = binding.created_source or None
                conn.execute(
                    """INSERT INTO sessions(id, source, created_source, user_id, session_key, chat_id, chat_type,
                                            model, cwd, profile_name, origin_json, started_at)
                       VALUES (%s, %s, %s, %s, %s, %s, 'dm', %s, %s, %s, %s, %s)
                       ON CONFLICT (id) DO NOTHING""",
                    (sid, policy["source"], created, receipt["principal_id"], route,
                     _bindable_chat_id(entry["origin"]["chat_id"]), policy["model"], _row_cwd(policy),
                     self._store.owner_profile_name, canonical_json(entry["origin"]), started))
                conn.execute("INSERT INTO gateway_routing(scope, session_key, entry_json, updated_at) "
                             "VALUES ('', %s, %s, %s)", (route, canonical_json(entry), started))
                conn.execute("INSERT INTO state_meta(key, value) VALUES (%s, %s)", (key, encoded))
        return ReservedBinding(binding=_binding_record(receipt), created=True)

    def adopt_untracked_compression(self, *, epoch: int, session_id: SessionId, expected: SessionBinding) -> None:
        """Record, once, compression steps a receipt-unaware build published past the receipt's tip (the
        reference's ``adopt_untracked_compression``): the stored receipt must still be *expected* and the
        steps must carry its route, owner and principal; anything else is left, unchanged."""
        with _refused(self._store):
            with self._store.tx() as conn:
                _require_epoch(conn, epoch)
                saved = conn.execute("SELECT value FROM state_meta WHERE key = %s",
                                    (POLICY_PREFIX + str(session_id),)).fetchone()
                receipt = json.loads(saved["value"])
                try:
                    if _binding_record(receipt) != expected:
                        return
                except AuthorityRefusal:  # a receipt missing an identity field is not the expected one
                    return
                adoptable = _adoptable_compression(conn, receipt)
                if adoptable is None:
                    return
                children, routed = adoptable
                receipt = _with_adopted_compression(receipt, children)
                _validate_local_lineage(conn, receipt)
                self._store.lineage._commit_local_target(conn, receipt, dict(routed, session_id=children[-1]))

    def reset_target(self, *, epoch: int, parent_session_id: SessionId, entry: Mapping[str, Any]) -> None:
        """A local ``/new`` in one write (the reference's ``reset_local_target``): the child row is
        published ``_reset_from`` the parent, the parent ends ``session_reset``, the receipt's target
        and the routing entry move to the child, and the logical owner's generation bumps."""
        entry = dict(entry)
        with _refused(self._store):
            with self._store.tx() as conn:
                _require_epoch(conn, epoch)
                parent = conn.execute("SELECT * FROM sessions WHERE id = %s",
                                     (str(parent_session_id),)).fetchone()
                if parent is None or parent["end_reason"] == "compression":
                    raise AuthorityRefusal("admission_conflict")
                saved = conn.execute("SELECT value FROM state_meta WHERE key = %s",
                                     (POLICY_PREFIX + parent["chat_id"],)).fetchone()
                if saved is None:
                    raise AuthorityRefusal("storage_unavailable")
                receipt = json.loads(saved["value"])
                logical_id = receipt["session_id"]
                if _validate_local_lineage(conn, receipt) != str(parent_session_id):
                    raise AuthorityRefusal("admission_conflict")
                _require_not_executing(conn, list({logical_id, str(parent_session_id)}))
                policy = receipt["policy"]
                lineage = self._store.lineage
                lineage._publish_child_session_row(
                    conn, parent, parent_session_id=str(parent_session_id),
                    child_session_id=entry["session_id"], source=policy["source"], model=policy["model"],
                    model_config={"_reset_from": str(parent_session_id)}, system_prompt=None,
                    cwd=policy["cwd"], profile_name=parent["profile_name"])
                conn.execute("UPDATE sessions SET ended_at = %s, end_reason = 'session_reset' WHERE id = %s",
                             (time.time(), str(parent_session_id)))
                _bump_conversation_generation(conn, str(parent_session_id), "session_reset")
                _advance_local_target(self._store, conn, str(parent_session_id), entry["session_id"], entry=entry)
                conn.execute("UPDATE sessions SET runtime_generation = runtime_generation + 1 WHERE id = %s",
                             (logical_id,))

    def end_idle(self, *, epoch: int, session_id: SessionId, target_id: SessionId, reason: str) -> int:
        """Stamp ``ended_at`` on *target_id* only while the logical *session_id*'s FIFO is idle (the
        reference's ``end_idle_local_session``): no queued/started/unknown admission on it, no
        non-terminal worker on either id; returns the rows stamped (0 when nothing was stamped)."""
        with _refused(self._store):
            with self._store.tx() as conn:
                _require_epoch(conn, epoch)
                if conn.execute(
                        "SELECT 1 FROM session_admissions WHERE target_session_id = %s "
                        "AND status IN ('queued','started','unknown')", (str(session_id),)).fetchone():
                    return 0
                if conn.execute(
                        "SELECT 1 FROM worker_executions WHERE session_id IN (%s, %s) "
                        "AND status IN ('registered','running','unknown')",
                        (str(session_id), str(target_id))).fetchone():
                    return 0
                return _end_and_bump(conn, str(target_id), reason)

    def bind_api(self, *, epoch: int, binding: NewApiBinding, gate: AdmissionGate = None,
                 walked_from: SessionId | None = None) -> None:
        """Bind an API route in one write (the reference's ``bind_api``), in order: the epoch; the gate
        (a callable answered inside the write); a retired id; the passed-generation fences; the stored
        binding's source and declared key; the declared-key registry; the row's source and the
        bot-room's title; the legacy key adoption; the route's holder; then the row, its route
        columns, the routing entry and the receipt."""
        receipt = dict(binding.receipt)
        sid, route = str(binding.session_id), binding.route
        with _refused(self._store):
            with self._store.tx() as conn:
                _require_epoch(conn, epoch)
                raise_gate(gate)
                if conn.execute("SELECT 1 FROM state_meta WHERE key = %s",
                               (RETIRED_PREFIX + sid,)).fetchone():
                    raise AuthorityRefusal("not_found")
                # The walk ran before this write: a sweep marker it passed may have been reclaimed
                # since, and binding the successor then would split the chain.
                if walked_from is not None and not _keep_passed_fences(
                        conn, partial(api_generation_id, str(walked_from)), sid):
                    raise AuthorityRefusal("chain_moved")
                saved = conn.execute("SELECT value FROM state_meta WHERE key = %s",
                                    (API_BINDING_PREFIX + sid,)).fetchone()
                if saved is not None:
                    stored = json.loads(saved["value"])
                    if stored.get("storage_source", "api_server") != binding.storage_source:
                        raise AuthorityRefusal("permission_denied")
                    if binding.declared_key and stored.get("declared_key") != binding.declared_key:
                        raise AuthorityRefusal("admission_conflict")
                    return
                if binding.declared_key:
                    existing = conn.execute("SELECT value FROM state_meta WHERE key = %s",
                                           (API_DECLARED_PREFIX + binding.declared_key,)).fetchone()
                    if existing is not None and existing["value"] != sid:
                        raise AuthorityRefusal("admission_conflict")
                    conn.execute("INSERT INTO state_meta(key, value) VALUES (%s, %s) "
                                 "ON CONFLICT (key) DO NOTHING",
                                 (API_DECLARED_PREFIX + binding.declared_key, sid))
                row = conn.execute("SELECT source, session_key, title FROM sessions WHERE id = %s",
                                  (sid,)).fetchone()
                if row is not None and row["source"] != binding.storage_source:
                    raise AuthorityRefusal("permission_denied")
                if binding.storage_source == "bot_room":
                    if row is not None and row["title"] != binding.title:
                        raise AuthorityRefusal("admission_conflict")
                    if conn.execute("SELECT 1 FROM sessions WHERE title = %s AND id <> %s",
                                   (binding.title, sid)).fetchone():
                        raise AuthorityRefusal("admission_conflict")
                # Earlier API releases stored the caller-declared key directly on the session row:
                # a caller presenting that exact key adopts the row onto the canonical route, and
                # the registry keeps the key so a later request finds this row by it.
                legacy_key = binding.declared_key or binding.caller_key
                if row is not None and row["session_key"] not in (None, "", route):
                    if not legacy_key or row["session_key"] != legacy_key:
                        raise AuthorityRefusal("admission_conflict")
                    owner_row = conn.execute("SELECT value FROM state_meta WHERE key = %s",
                                            (API_DECLARED_PREFIX + legacy_key,)).fetchone()
                    if owner_row is not None and owner_row["value"] != sid:
                        raise AuthorityRefusal("admission_conflict")
                    conn.execute("INSERT INTO state_meta(key, value) VALUES (%s, %s) "
                                 "ON CONFLICT (key) DO NOTHING", (API_DECLARED_PREFIX + legacy_key, sid))
                    receipt["declared_key"] = legacy_key
                existing_entry = conn.execute(
                    "SELECT entry_json FROM gateway_routing WHERE scope = '' AND session_key = %s",
                    (route,)).fetchone()
                held = json.loads(existing_entry["entry_json"])["session_id"] if existing_entry is not None else sid
                if held != sid and not _heal_replacement(conn, sid, held):
                    raise AuthorityRefusal("admission_conflict")
                conn.execute("INSERT INTO sessions(id, source, title, hidden, started_at) "
                             "VALUES (%s, %s, %s, %s, %s) ON CONFLICT (id) DO NOTHING",
                             (sid, binding.storage_source, binding.title,
                              1 if binding.storage_source == "bot_room" else 0, binding.started_at))
                conn.execute("UPDATE sessions SET session_key = %s, chat_id = %s, user_id = %s, "
                             "chat_type = 'dm', origin_json = %s WHERE id = %s",
                             (route, sid, binding.origin["user_id"], canonical_json(dict(binding.origin)), sid))
                conn.execute("INSERT INTO gateway_routing(scope, session_key, entry_json, updated_at) "
                             "VALUES ('', %s, %s, %s) "
                             "ON CONFLICT (scope, session_key) DO UPDATE SET entry_json = EXCLUDED.entry_json",
                             (route, canonical_json(dict(binding.entry)), binding.started_at))
                conn.execute("INSERT INTO state_meta(key, value) VALUES (%s, %s)",
                             (API_BINDING_PREFIX + sid, canonical_json(receipt)))

    def release_handed_off_receipts(self, *, epoch: int, lanes: frozenset[str]) -> dict[str, tuple[str, str]]:
        """Retire every receipt whose current target row now carries another route and a source in
        *lanes* (the reference's ``release_handed_off_receipts``): its earlier segments take the
        lane's identity, the conversation's title follows its head, and the receipt and its legacy
        binding are deleted. Returns owner to (target, source) for the receipts handed off."""
        with _refused(self._store):
            with self._store.tx() as conn:
                _require_epoch(conn, epoch)
                moved = {owner: (target, source)
                         for owner, (target, source) in _moved_receipts(conn).items()
                         if source in lanes}
                return _hand_off_conversations(conn, moved)

    # ── reads ─────────────────────────────────────────────────────────────────

    def target_has_history(self, session_id: str) -> bool:
        """Whether *session_id* holds a transcript row, ledger rows or a child session, in one read
        (the reference's ``target_has_history``)."""
        with _refused(self._store):
            found = self._store.one(
                "SELECT 1 AS one FROM messages WHERE session_id = %s "
                "UNION ALL SELECT 1 FROM session_admissions WHERE target_session_id = %s "
                "UNION ALL SELECT 1 FROM worker_executions WHERE session_id = %s "
                "UNION ALL SELECT 1 FROM sessions WHERE parent_session_id = %s LIMIT 1",
                session_id, session_id, session_id, session_id)
        return found is not None

    def receipt(self, session_id: SessionId) -> SessionBinding:
        """The stored creation receipt; no receipt, an undecodable one or one that is not an object
        is ``storage_unavailable`` (the reference's ``local_receipt``)."""
        with _refused(self._store):
            row = self._store.one("SELECT value FROM state_meta WHERE key = %s",
                                  POLICY_PREFIX + str(session_id))
            if row is None:
                raise AuthorityRefusal("storage_unavailable")
            try:
                value = json.loads(row["value"])
                if not isinstance(value, dict):
                    raise ValueError("invalid receipt")
                return _binding_record(value)
            except (TypeError, ValueError) as exc:
                raise AuthorityRefusal("storage_unavailable") from exc

    def local_session_ids(self) -> list[SessionId]:
        """Every id holding a creation receipt, in store order (the reference's ``local_session_ids``)."""
        with _refused(self._store):
            rows = self._store.all("SELECT key FROM state_meta WHERE key LIKE %s", POLICY_PREFIX + "%")
        return [SessionId(row["key"][len(POLICY_PREFIX):]) for row in rows]

    def list_lineages(self) -> list[LocalLineage]:
        """Every receipt's owner and lineage; an undecodable receipt, a non-string owner or a
        non-list lineage is skipped, and non-string members are dropped (the reference's
        ``local_lineages``)."""
        with _refused(self._store):
            rows = self._store.all("SELECT value FROM state_meta WHERE key LIKE %s", POLICY_PREFIX + "%")
        found = []
        for (raw,) in [(row["value"],) for row in rows]:
            try:
                receipt = json.loads(raw)
            except (TypeError, ValueError):
                continue
            owner = receipt.get("session_id") if isinstance(receipt, dict) else None
            lineage = receipt.get("lineage") if isinstance(owner, str) else None
            if isinstance(lineage, list):
                found.append((owner, [physical for physical in lineage if isinstance(physical, str)]))
        return [LocalLineage(owner=SessionId(owner), lineage=tuple(SessionId(sid) for sid in lineage))
                for owner, lineage in found]

    def binding_marks(self, session_id: SessionId, *, lineage_of: SessionId | None = None) -> BindingMarks:
        """Whether a creation receipt and a legacy binding exist for *session_id*; with *lineage_of*
        and either present, also the lineage owner, in one read (the reference's ``binding_marks``)."""
        with _refused(self._store):
            with self._store.tx() as conn:
                policy = conn.execute("SELECT 1 FROM state_meta WHERE key = %s",
                                     (POLICY_PREFIX + str(session_id),)).fetchone()
                legacy = conn.execute("SELECT 1 FROM state_meta WHERE key = %s",
                                     (LEGACY_PREFIX + str(session_id),)).fetchone()
                owner = _local_lineage_owner(conn, str(lineage_of)) if lineage_of and (policy or legacy) else None
        return BindingMarks(policy=policy is not None, legacy=legacy is not None,
                           lineage_owner=None if owner is None else SessionId(owner))

    def lineage_owner(self, session_id: SessionId) -> SessionId:
        """The creation id whose receipt lineage holds the segment *session_id* (named by the row's
        ``chat_id``), else *session_id* (the reference's ``lineage_owner``)."""
        with _refused(self._store):
            with self._store.tx() as conn:
                return SessionId(_local_lineage_owner(conn, str(session_id)))

    def conversation_ids(self, session_id: SessionId) -> list[SessionId]:
        """Every row of the local conversation *session_id* belongs to: its creation id first, then
        each lineage segment, deduplicated; ``[session_id]`` without a receipt lineage (the
        reference's ``conversation_ids``)."""
        with _refused(self._store):
            with self._store.tx() as conn:
                return [SessionId(sid) for sid in _local_conversation_ids(conn, str(session_id))]

    def validate_lineage(self, receipt: SessionBinding) -> SessionId:
        """Validate the passed receipt against the rows and return its tip: the id must be the local
        identity (or the legacy binding must match), the lineage must run from the root to the target
        through real reset/compression hops, and the tip must carry no compression step; anything
        else is ``storage_unavailable`` (the reference's ``validate_lineage``)."""
        receipt_dict = _receipt_of(receipt)
        with _refused(self._store):
            with self._store.tx() as conn:
                tip = _validate_local_lineage(conn, receipt_dict)
        return SessionId(tip)

    def compression_adoptable(self, receipt: SessionBinding) -> bool:
        """Whether ``adopt_untracked_compression`` would adopt steps for this receipt, in one read
        (the reference's ``compression_adoptable``)."""
        receipt_dict = _receipt_of(receipt)
        with _refused(self._store):
            with self._store.tx() as conn:
                return _adoptable_compression(conn, receipt_dict) is not None

    def restorable(self, receipt: SessionBinding) -> SessionBinding:
        """The passed receipt as a restore would commit it (any adoptable compression adopted),
        validated, in one read that writes nothing (the reference's ``restorable``); it refuses as
        ``validate_lineage`` does."""
        receipt_dict = _receipt_of(receipt)
        with _refused(self._store):
            with self._store.tx() as conn:
                adoptable = _adoptable_compression(conn, receipt_dict)
                if adoptable is not None:
                    receipt_dict = _with_adopted_compression(receipt_dict, adoptable[0])
                _validate_local_lineage(conn, receipt_dict)
        return _binding_record(receipt_dict)

    def authorize_history(self, *, session_id: SessionId, profile_id: str, subject: str, may_create: bool,
                         native_transport: bool | None = None, owner: OwnerRule | None) -> bool:
        """One read (the reference's ``authorize_history_read``): the claim check (a boolean
        *native_transport* runs it, None skips it - the listing's read) and the authorization, with
        the gateway's answer, then the owner and create-capability rules."""
        actor = HistoryActor(profile_id, subject, login_label(subject), may_create)
        with _refused(self._store):
            with self._store.tx() as conn:
                if native_transport is not None:
                    _require_history_claim(conn, self._store, actor, str(session_id), native_transport, owner)
                return _authorize_history(conn, actor, str(session_id), owner=owner)

    def api_binding(self, session_id: SessionId) -> Optional[ApiBinding]:
        """The stored API binding receipt, parsed (``storage_source`` ``api_server`` and
        ``room_generation`` 0 when absent, a ``room_identity`` as a tuple), or None (the reference's
        ``api_binding_receipt``)."""
        with _refused(self._store):
            row = self._store.one("SELECT value FROM state_meta WHERE key = %s",
                                  API_BINDING_PREFIX + str(session_id))
        if row is None:
            return None
        receipt = json.loads(row["value"])
        identity = receipt.get("room_identity")
        return ApiBinding(
            profile_id=receipt["profile_id"], session_id=SessionId(receipt["session_id"]), route=receipt["route"],
            entry=receipt["entry"], declared_key=receipt.get("declared_key"),
            storage_source=receipt.get("storage_source", "api_server"),
            room_identity=tuple(identity) if identity is not None else None,
            room_generation=int(receipt.get("room_generation", 0)))

    def declared_session(self, declared_key: str) -> Optional[SessionId]:
        """The session a declared conversation key is registered to, or None (the registry read only,
        the reference's ``declared_session``)."""
        with _refused(self._store):
            row = self._store.one("SELECT value FROM state_meta WHERE key = %s",
                                  API_DECLARED_PREFIX + declared_key)
        return None if row is None else SessionId(row["value"])

    def legacy_adoption_scan(self, owner: OwnerRule) -> LegacyAdoptionScan:
        """The legacy local rows ``hermes sessions adopt`` walks (the reference's
        ``legacy_adoption_scan``): the candidate rows (a legacy local source, or a routeless row whose
        label ``not_client_labels`` does not refuse under *owner*) ordered by start time then id, and
        the ids already carrying a local creation receipt or a legacy binding."""
        local = tuple(sorted(LEGACY_LOCAL_SOURCES))
        refused_labels = tuple(sorted(not_client_labels(owner)))
        with _refused(self._store):
            ids = [row["id"] for row in self._store.all(
                f"SELECT id FROM sessions WHERE source IN ({', '.join('%s' for _ in local)})"
                f" OR (source NOT IN ({', '.join('%s' for _ in refused_labels)}) "
                "AND COALESCE(session_key, '') = '' AND COALESCE(chat_id, '') = '' "
                "AND COALESCE(origin_json, '') = '') ORDER BY started_at, id",
                *local, *refused_labels)]
            bound_rows = self._store.all(
                "SELECT key FROM state_meta WHERE key LIKE %s OR key LIKE %s",
                POLICY_PREFIX + "%", LEGACY_PREFIX + "%")
        marked = {key[len(prefix):] for key in (row["key"] for row in bound_rows)
                  for prefix in (POLICY_PREFIX, LEGACY_PREFIX) if key.startswith(prefix)}
        return LegacyAdoptionScan(session_ids=tuple(SessionId(sid) for sid in ids),
                                  marked=frozenset(marked))

    def history_binding(self, session_id: SessionId) -> Optional[str]:
        """The stored import-binding text of *session_id* exactly as written (compact JSON
        ``[profile_id, subject]``), or None (the reference's ``history_binding``)."""
        with _refused(self._store):
            row = self._store.one("SELECT value FROM state_meta WHERE key = %s",
                                  HISTORY_BINDING_PREFIX + str(session_id))
        return row["value"] if row is not None else None

    def handoff_head(self, session_id: SessionId) -> tuple[Optional[SessionId], SessionId]:
        """``(local owner or None, head row)`` a ``/handoff`` of *session_id* moves (the reference's
        ``handoff_head``): the lineage owner and its receipt's target, then that target's compression
        tip; a receipt without a target names the row itself, and a missing row answers
        ``(None, session_id)``."""
        with _refused(self._store):
            with self._store.tx() as conn:
                owner = _local_lineage_owner(conn, str(session_id))
                saved = conn.execute("SELECT value FROM state_meta WHERE key = %s",
                                    (POLICY_PREFIX + owner,)).fetchone()
        target = _policy_target(saved["value"]) if saved is not None else None
        start = target or str(session_id)
        tip = _get_compression_tip(self._store, start)
        return (SessionId(owner) if target else None), SessionId(tip if tip is not None else start)

    def boot_session_ids(self) -> list[SessionId]:
        """The ids of the receipts boot restores, in one read (the reference's ``local_receipt_ids``):
        the target row is open, missing or ended by compression, the owner has unsettled input, or a
        stored viewer close; a receipt that does not decode names no target row, so it is kept."""
        start = len(POLICY_PREFIX) + 1
        sql = (
            "SELECT m.key FROM state_meta m "
            "LEFT JOIN sessions s ON s.id = CASE WHEN hermes_json_valid(m.value) "
            "THEN m.value::jsonb -> 'entry' ->> 'session_id' END "
            "WHERE m.key LIKE %s AND (s.ended_at IS NULL OR s.end_reason = 'compression' "
            "OR EXISTS (SELECT 1 FROM session_admissions a "
            "           WHERE a.target_session_id = substring(m.key, %s) "
            "             AND a.status IN ('queued','started','unknown')) "
            "OR EXISTS (SELECT 1 FROM state_meta c WHERE c.key = %s || substring(m.key, %s)))"
        )
        with _refused(self._store):
            rows = self._store.all(sql, POLICY_PREFIX + "%", start, VIEWER_CLOSE_PREFIX, start)
        return [SessionId(row["key"][len(POLICY_PREFIX):]) for row in rows]


add_stubs(
    PgBindingsRepository,
    SessionBindingRepository,
    implemented=(
        "adopt_untracked_compression",
        "api_binding",
        "bind_api",
        "boot_session_ids",
        "binding_marks",
        "compression_adoptable",
        "conversation_ids",
        "declared_session",
        "end_idle",
        "handoff_head",
        "history_binding",
        "lineage_owner",
        "list_lineages",
        "legacy_adoption_scan",
        "local_session_ids",
        "receipt",
        "release_handed_off_receipts",
        "reset_target",
        "reserve",
        "restorable",
        "target_has_history",
        "validate_lineage",
    ),
)


def _register() -> None:
    register_repository("bindings", lambda store: PgBindingsRepository(store))


_register()

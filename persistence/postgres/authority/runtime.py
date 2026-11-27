"""``SessionRuntimeRepository`` on PostgreSQL (stage 2): the store-wide epoch, the receipted session
mutations and their prepare phase, the deferred and chat picks, the frozen-policy refresh, the viewer
closes, the retirement reads and the ledger-gated ends.

Each verb converts its typed parameters to its body's arguments, runs one body (one ``store.tx()``
write, or one snapshot-pinned read) inside :func:`_refused`, and converts the result to the contract
record. The bodies are the SQLite authority store's SQL (``persistence.sqlite.authority``) ported
inline onto the PG dialect: the state_meta key prefixes, the receipt codecs, the pure guard/owner
helpers and the title rules are imported from the shared modules, so the two backends cannot drift;
the session-lineage, compression, titles and transcript bodies reuse the PG conversation stores' own
connection-level helpers (``_set_lineage_column_in_transaction``, ``_retire_sessions``, the lineage
repository's child publish and message rows, ``rewind_in_transaction``), so a mutation write composes
over the same rows the conversation family writes.

The store has no profile directory, so ``begin_epoch`` never fences out a profile lock (the contract's
no-home case): every bump is granted. The two fixture seams (``seed_local_receipt``,
``seed_retirement``) are carried on the repository itself, so a ported repository always provides
them (the test suite's seam attach skips a repository that already has a verb).

D8 note: the in-place ``compress`` body archives the active rows and inserts the compacted set
inline; the store's fuller ``archive_and_compact`` coverage/held-CAS body stays reachable through the
non-receipted conversation path. The branch op's unnumbered-title walk reuses the PG store's
``next_title_in_lineage`` (a snapshot read), which sees committed titles only.
"""

from __future__ import annotations

import json
import time
import uuid
from contextlib import contextmanager
from typing import Any, Iterator, Mapping, Optional, Sequence

from persistence.authority.bindings import (
    api_generation_id,
    generation_request_id,
    local_session_id,
    room_session_id,
)
from persistence.authority.runtime import (
    ApiGenerations,
    ChatPickParams,
    ClaimHistory,
    COMPACTED_TARGET_DETAIL,
    DeferredPick,
    DeleteFootprint,
    GenerationChain,
    GenerationPick,
    GenerationRow,
    HistoryWrite,
    LocalGenerations,
    LocalSnapshot,
    MutationCommand,
    MutationOutcome,
    PinnedIdentity,
    PreparedMutation,
    RoomGenerations,
    SessionRuntimeRepository,
    SessionRuntimeState,
    StoredDeferredPick,
    mutation_receipt_key,
    parse_mutation,
)
from persistence.authority.vocabulary import OwnerRule
from persistence.core.canonical import canonical_json
from persistence.core.errors import (
    AuthorityRefusal,
    FenceStale,
    ProofRejected,
    StaleFenceKind,
    UnknownExecution,
)
from persistence.core.identity import SessionId
from persistence.sqlite._schema import TITLE_SOURCE_DERIVED, TITLE_SOURCE_USER
from persistence.sqlite.conversation._store.compression import _CHAIN_CAP
from persistence.sqlite.conversation._store.local import POLICY_PREFIX
from persistence.sqlite.conversation._store.local_migration import LEGACY_PREFIX
from persistence.sqlite.conversation._store.retirement import RETIRED_PREFIX, VIEWER_CLOSE_PREFIX
from persistence.sqlite.conversation._store.terminal import identity_key
from session_rules.legacy_local import (
    binding_names,
    bindings_agree,
    legacy_account_label,
    legacy_history_row,
    owners_agree,
    same_owner,
)
from session_rules.turns import split_user_originated_turn, typed_prompt_content

from .._backend import PostgresStore, refusals, register_repository, translated
from .._stubs import add_stubs
from ..conversation._store.common import _check_transcript_write_guards
from ..conversation._store.rewind import _active_transcript_counts, rewind_in_transaction
from ..conversation.messages import _row_to_message_dict
from ..conversation.sessions import (
    _CHAIN_STEP_SQL,
    _LINEAGE_CTE_SQL,
    _bump_conversation_generation,
    _compression_tip_on,
    _collect_delegate_child_ids,
    _delete_unreferenced_system_prompts,
    _end_and_bump,
    _local_conversation_ids,
    _local_lineage_owner,
    _mark_retired,
    _non_continuation_child_sql,
    _parse_model_config,
    _retire_canonical_bot_chat_in_transaction,
    _retire_sessions,
    _set_lineage_column_in_transaction,
    _set_session_title_in_transaction,
    next_title_in_lineage,
)
from ..gateway.routing import _title_follows_head

__all__ = ["PgRuntimeRepository"]

# The state_meta key prefixes (the reference's module constants; the contract package keeps the
# receipt-key copy, the binding-control copy lives beside its checks here).
DEFERRED_PICK_PREFIX = "gateway.deferred_pick.v1."
BINDING_PREFIX = "gateway.history_control.v1."
_MUTATION_PREFIX = "gateway.mutation.v1."

_FENCE_KINDS = {kind.value: kind for kind in StaleFenceKind}

# The compression-continuation edge (the PG store's rule): the children a parent's compression ended
# that keep the conversation, never by marker presence (a continuation copies its parent's
# ``model_config``, so an inherited fork marker does not make it a fork).
_CONTINUATION_EDGE = _non_continuation_child_sql("child.", "parent.id")
_FORWARD_CONTINUATION_SQL = (
    "SELECT child.id FROM sessions parent JOIN sessions child ON child.parent_session_id = parent.id "
    "WHERE parent.id = %s AND parent.end_reason = 'compression'" + _CONTINUATION_EDGE
)
_BACKWARD_CONTINUATION_SQL = (
    "SELECT parent.id FROM sessions child JOIN sessions parent ON parent.id = child.parent_session_id "
    "WHERE child.id = %s AND parent.end_reason = 'compression'" + _CONTINUATION_EDGE
)
_FORWARD_CONTINUATION_MANY_SQL = (
    "SELECT child.id FROM sessions parent JOIN sessions child ON child.parent_session_id = parent.id "
    "WHERE parent.id = ANY(%s) AND parent.end_reason = 'compression'" + _CONTINUATION_EDGE
)
_BACKWARD_CONTINUATION_MANY_SQL = (
    "SELECT parent.id FROM sessions child JOIN sessions parent ON parent.id = child.parent_session_id "
    "WHERE child.id = ANY(%s) AND parent.end_reason = 'compression'" + _CONTINUATION_EDGE
)

# The gateway ledger's first unsettled admission or worker on one session id (the SQLite
# ``_CANONICAL_PENDING_SQL``; which row answers when several of one kind match is unspecified).
_CANONICAL_PENDING_SQL = (
    "SELECT status FROM session_admissions WHERE target_session_id = %s AND status <> 'terminal' "
    "UNION ALL SELECT status FROM worker_executions WHERE session_id = %s AND status <> 'terminal' "
    "LIMIT 1"
)


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
def _refused(store: PostgresStore, session_id: Optional[str] = None) -> Iterator[None]:
    """Raise these steps' refusals as the contract's typed refusals, and the store's failures as the
    contract's error family (the reference's ``refused``; the PG ``refusals`` maps the store's errors)."""
    try:
        with refusals(store, session_id):
            yield
    except AuthorityRefusal as exc:
        if type(exc) is not AuthorityRefusal:
            raise
        typed = _typed(exc)
        if typed is exc:
            raise
        raise typed from exc


def _text(value: Any) -> str:
    """The identity texts (str, non-empty, at most 1024 chars)."""
    if not isinstance(value, str) or not value or len(value) > 1024:
        raise AuthorityRefusal("invalid_params")
    return value


def _epoch(conn, epoch: int) -> None:
    """The epoch fence (the reference's ``_epoch``): the singleton row's stored epoch, else ``stale_epoch``."""
    row = conn.execute("SELECT epoch FROM runtime_epoch WHERE singleton = 1").fetchone()
    if type(epoch) is not int or row is None or row["epoch"] != epoch:
        raise AuthorityRefusal("stale_epoch")


def _session_row(conn, session_id: str) -> dict:
    """The session row (the reference's ``_session``), else ``not_found``."""
    row = conn.execute("SELECT * FROM sessions WHERE id = %s", (session_id,)).fetchone()
    if row is None:
        raise AuthorityRefusal("not_found")
    return row


def _replay_mutation_receipt(conn, key: str, digest: str) -> Optional[dict]:
    """The stored result of an exact retry, ``None`` when unseen; a different request refuses."""
    old = conn.execute("SELECT value FROM state_meta WHERE key = %s", (key,)).fetchone()
    if old is None:
        return None
    receipt = json.loads(old["value"])
    if receipt["digest"] != digest:
        raise AuthorityRefusal("admission_conflict")
    return receipt["result"]


def _validate_local_lineage(conn, receipt: dict) -> str:
    """The receipt's lineage is a valid root-to-tip walk (the store's ``validate_local_lineage``), else
    ``storage_unavailable``; returns the tip. The PG dialect walks the shared chain-step rule."""
    if ("legacy_session_id" not in receipt
            and receipt["session_id"] != str(local_session_id(receipt["profile_id"], receipt["principal_id"],
                                                             receipt["request_id"]))):
        raise AuthorityRefusal("storage_unavailable")
    root = receipt.get("legacy_session_id", receipt["session_id"])
    if "legacy_session_id" in receipt:
        if root != receipt["session_id"]:
            raise AuthorityRefusal("storage_unavailable")
        row = conn.execute("SELECT value FROM state_meta WHERE key = %s", (LEGACY_PREFIX + root,)).fetchone()
        expected = {k: receipt[k] for k in ("session_id", "profile_id", "principal_id", "legacy_session_id")}
        if row is None or json.loads(row["value"]) != expected:
            raise AuthorityRefusal("storage_unavailable")
    lineage = receipt.get("lineage", [root])
    if (not isinstance(lineage, list) or not lineage or len(set(lineage)) != len(lineage)
            or lineage[0] != root or lineage[-1] != receipt["entry"]["session_id"]):
        raise AuthorityRefusal("storage_unavailable")
    for parent_id, child_id in zip(lineage, lineage[1:]):
        parent = conn.execute("SELECT end_reason FROM sessions WHERE id = %s", (parent_id,)).fetchone()
        child = conn.execute("SELECT parent_session_id, model_config FROM sessions WHERE id = %s", (child_id,)).fetchone()
        if parent is None or child is None or child["parent_session_id"] != parent_id:
            raise AuthorityRefusal("storage_unavailable")
        if parent["end_reason"] == "session_reset":
            valid = _parse_model_config(child["model_config"]).get("_reset_from") == parent_id
        else:
            tip = conn.execute(_CHAIN_STEP_SQL, (parent_id,)).fetchone()
            valid = parent["end_reason"] == "compression" and tip is not None and tip["id"] == child_id
        if not valid:
            raise AuthorityRefusal("storage_unavailable")
    if conn.execute(_CHAIN_STEP_SQL, (lineage[-1],)).fetchone() is not None:
        raise AuthorityRefusal("storage_unavailable")
    return lineage[-1]


def _local_physical_target(conn, session_id: str) -> str:
    """Resolve a local owner's current transcript on the caller's transaction (the SQLite
    ``local_physical_target``)."""
    saved = conn.execute("SELECT value FROM state_meta WHERE key = %s", (POLICY_PREFIX + session_id,)).fetchone()
    if saved is None:
        return session_id
    receipt = json.loads(saved["value"])
    if receipt["session_id"] != session_id:
        raise AuthorityRefusal("storage_unavailable")
    return _validate_local_lineage(conn, receipt)


def _local_snapshot(conn, session_id: str) -> dict:
    """The local snapshot (the reference's ``local_snapshot``): the receipt, its validated physical
    target and that target's revision; ``runtime_coordination_required`` without a receipt."""
    row = conn.execute("SELECT value FROM state_meta WHERE key = %s", (POLICY_PREFIX + session_id,)).fetchone()
    if row is None:
        raise AuthorityRefusal("runtime_coordination_required")
    receipt = json.loads(row["value"])
    target = _validate_local_lineage(conn, receipt)
    _require_target_advanceable(conn, list({session_id, target}))
    revision = conn.execute("SELECT runtime_revision FROM sessions WHERE id = %s", (target,)).fetchone()["runtime_revision"]
    return {"receipt": receipt, "target": target, "target_revision": revision}


def _expand_compression_lineage_ids(conn, frontier: list) -> list:
    """One both-direction compression-edge step from *frontier* (the SQLite ``_expand_compression_lineage_ids``
    over the PG chain-step rule)."""
    ids = [sid for sid in frontier if sid]
    if not ids:
        return []
    out = [row["id"] for row in conn.execute(_FORWARD_CONTINUATION_MANY_SQL, (ids,)).fetchall()]
    out += [row["id"] for row in conn.execute(_BACKWARD_CONTINUATION_MANY_SQL, (ids,)).fetchall()]
    return out


def _continuation_edges(conn, sid: str) -> list:
    """One both-direction compression step from *sid* (the reference's ``_continuation_edges``)."""
    rows = conn.execute(
        _FORWARD_CONTINUATION_SQL + " UNION " + _BACKWARD_CONTINUATION_SQL, (sid, sid)).fetchall()
    return [row["id"] for row in rows]


# ── the transcript guards (the reference's ``_runtime_guards``) ─────────────────────────────────


def _require_not_executing(conn, session_ids) -> None:
    """Refuse while a turn is running (or its outcome is unknown) on any of ``session_ids``."""
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
    successor (also in the ids) carries the lease/lock. A live row must be unleased."""
    row = conn.execute("SELECT ended_at, end_reason FROM sessions WHERE id = %s", (sid,)).fetchone()
    if row is not None and row["ended_at"] is not None and row["end_reason"] == "compression":
        return
    _check_transcript_write_guards(conn, sid, None,
                                   reject_active_turn_lease=True, reject_active_compression_lock=True)


def _require_target_advanceable(conn, session_ids) -> None:
    """Fence for writes that publish a new physical target: refuse executing/unknown work and live
    transcript leases, never a queued follower."""
    _require_not_executing(conn, session_ids)
    for sid in session_ids:
        _require_transcript_unleased(conn, sid)


def _require_idle(conn, session_ids) -> None:
    """Refuse while the ledger holds ANY unsettled work on ``session_ids`` (queued too), else as
    :func:`_require_target_advanceable`."""
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


# ── the history-control ownership (the reference's ``_binding_history``) ─────────────────────────


def _history_row(row: dict, owner: Optional[OwnerRule]) -> bool:
    """Local surfaces' unrouted rows are owner history: every row the native resume adopts plus imports."""
    return row["source"] == "import" or legacy_history_row(dict(row), owner)


def _owner_user(row: dict) -> Optional[str]:
    """The row's owner identity; a native Desktop Nous-account label is not one."""
    return None if legacy_account_label(dict(row)) else row["user_id"]


def _history_binding(conn, session_id: str) -> Optional[str]:
    saved = conn.execute("SELECT value FROM state_meta WHERE key = %s", (BINDING_PREFIX + session_id,)).fetchone()
    return saved["value"] if saved else None


def _import_history_control(conn, actor, session_ids) -> None:
    binding = json.dumps([actor.profile_id, actor.subject], separators=(",", ":"))
    for sid in session_ids:
        if conn.execute("SELECT 1 FROM sessions WHERE id = %s", (sid,)).fetchone() is None:
            conn.execute("INSERT INTO state_meta (key, value) VALUES (%s, %s)", (BINDING_PREFIX + sid, binding))


def _authorize_history(conn, actor, session_id: str, *, claim: bool = False,
                       owner: Optional[OwnerRule] = None) -> bool:
    row = conn.execute("SELECT * FROM sessions WHERE id = %s", (session_id,)).fetchone()
    if row is None:
        return False
    if row["session_key"] or row["chat_id"] or row["origin_json"]:
        return False
    key = BINDING_PREFIX + session_id
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
    if claim:
        conn.execute("INSERT INTO state_meta (key, value) VALUES (%s, %s)", (key, binding))
    return True


def _authorize_write(conn, history: HistoryWrite, session_id: str, owner: Optional[OwnerRule] = None) -> None:
    """The history-control write: claim the session's unbound legacy history, or bind an import's new ids."""
    if not isinstance(history, ClaimHistory):
        _import_history_control(conn, history.actor, history.session_ids)
        return
    if not _authorize_history(conn, history.actor, session_id, claim=True, owner=owner):
        raise AuthorityRefusal("permission_denied")


def _same_history_owner(conn, anchor_id: str, session_id: str, owner: Optional[OwnerRule] = None) -> bool:
    """Whether chain member ``session_id`` is history the authorizer of ``anchor_id`` controls
    (the reference's ``same_history_owner``; routed members must carry the anchor's exact route
    columns, routeless ones pass the binding/local rule under *owner*)."""
    columns = ("SELECT source, user_id, session_key, chat_id, origin_json FROM sessions WHERE id = %s")
    anchor, row = (conn.execute(columns, (sid,)).fetchone() for sid in (anchor_id, session_id))
    if anchor is None or row is None:
        return False
    keys = ("source", "user_id", "session_key", "chat_id", "origin_json")
    anchor_t = tuple(anchor[k] for k in keys)
    row_t = tuple(row[k] for k in keys)
    if row["session_key"] or row["chat_id"] or row["origin_json"]:
        return row_t[1:4] == anchor_t[1:4]
    binding = _history_binding(conn, session_id)
    if not bindings_agree(binding, _history_binding(conn, anchor_id), owner):
        return False
    routeless_peer = row["source"] == anchor["source"] and not any(anchor_t[2:5])
    return binding is not None or (
        (_history_row(row, owner) or routeless_peer)
        and (_owner_user(row) in (None, "") or owners_agree(_owner_user(row), anchor["user_id"], owner)))


# ── the delete target walk (the reference's ``delete_targets``) ─────────────────────────────────


def _delete_targets(conn, session_id: str, rule: Optional[OwnerRule] = None) -> list:
    """The delete walk of *session_id* under *rule*: the id, its local conversation's segments, its
    compression chain in both directions within one history owner, the delegate subtrees. A missing
    row's targets are the id alone; a walk whose local receipt lineage does not validate refuses
    ``storage_unavailable``."""
    targets = {session_id}
    owner = _local_lineage_owner(conn, session_id)
    targets.add(owner)
    saved = conn.execute("SELECT value FROM state_meta WHERE key = %s", (POLICY_PREFIX + owner,)).fetchone()
    if saved is not None:
        receipt = json.loads(saved["value"])
        _validate_local_lineage(conn, receipt)
        targets.update(receipt.get("lineage", [session_id]))
    anchor = dict.fromkeys(targets, session_id)
    frontier = list(targets)
    for _ in range(_CHAIN_CAP):
        found: dict[str, str] = {}
        for sid in frontier:
            for nxt in _continuation_edges(conn, sid):
                if nxt not in targets and nxt not in found and _same_history_owner(conn, anchor[sid], nxt, rule):
                    found[nxt] = anchor[sid]
        for sid in _collect_delegate_child_ids(conn, frontier):
            if sid not in targets:
                found.setdefault(sid, sid)
        if not found:
            break
        anchor.update(found)
        targets.update(found)
        frontier = list(found)
    return [session_id, *sorted(targets - {session_id})]


def _delete_in_transaction(conn, session_id: str, *, media_root, owner: Optional[OwnerRule]) -> tuple[set, dict]:
    """The delete a mutation runs (the reference's ``delete_in_transaction``): the fence, the
    generation bumps and the row removals, one transaction."""
    targets = _delete_targets(conn, session_id, owner)
    _require_idle(conn, targets)
    _retire_sessions(conn, targets, media_root=media_root)
    for sid in targets:
        _bump_conversation_generation(conn, sid, "session_reset")
        conn.execute(
            "UPDATE sessions SET parent_session_id = NULL, runtime_revision = runtime_revision + 1 "
            "WHERE parent_session_id = %s", (sid,))
        conn.execute("DELETE FROM messages WHERE session_id = %s", (sid,))
    if targets:
        conn.execute(
            f"DELETE FROM sessions WHERE id = ANY(%s)", (targets,))
    _delete_unreferenced_system_prompts(conn)
    return set(), {"deleted_ids": targets}


# ── the rewind body helpers (the reference's ``_rewind``) ───────────────────────────────────────


def _summarized_by_compaction(conn, physical: str, message_id: int) -> bool:
    """Whether *message_id* is a user turn compaction took out of *physical*'s live transcript, as
    opposed to one a rewind/undo deactivated, an assistant row, or a row on no displayed row of the
    chain (the reference's ``_summarized_by_compaction`` over the PG chain-step rule)."""
    row = conn.execute(
        "SELECT session_id, role, active, compacted FROM messages WHERE id = %s", (message_id,)).fetchone()
    if row is None or row["role"] != "user":
        return False
    if row["session_id"] == physical:
        return row["active"] == 0 and row["compacted"] == 1
    if not (row["active"] == 1 or row["compacted"] == 1):
        return False
    lineage, frontier = {physical}, [physical]
    for _ in range(_CHAIN_CAP):
        frontier = [sid for sid in _expand_compression_lineage_ids(conn, frontier) if sid not in lineage]
        if not frontier:
            break
        lineage.update(frontier)
    return row["session_id"] in lineage


def _carrier_live_view(conn, session_id: str, message_id: int) -> Optional[dict]:
    """The composite compaction carrier's live ask (the reference's ``_carrier_live_view``), else None."""
    row = conn.execute(
        "SELECT * FROM messages WHERE id = %s AND session_id = %s AND active = 1", (message_id, session_id)).fetchone()
    message = _row_to_message_dict(row, warn_context="rewind target", summary_flag=True) if row else None
    handoff, live = split_user_originated_turn(message) if message else (None, None)
    return live if handoff is not None else None


def _rewind_in_transaction(conn, store: PostgresStore, session_id: str, payload: dict) -> tuple[set, dict]:
    """The rewind command (the reference's ``_rewind``): the guards, the target's live view, the
    soft-delete through the store's rewind body, and the generation bump."""
    physical = _local_physical_target(conn, session_id)
    _require_idle(conn, list({session_id, physical}))
    row = conn.execute(
        "SELECT role FROM messages WHERE id = %s AND session_id = %s AND active = 1",
        (payload["target_message_id"], physical)).fetchone()
    if row is None or row["role"] != "user":
        if _summarized_by_compaction(conn, physical, payload["target_message_id"]):
            raise AuthorityRefusal("invalid_params", field="target_message_id", detail=COMPACTED_TARGET_DETAIL)
        raise AuthorityRefusal("invalid_params", field="target_message_id",
                              detail="target user message is no longer in session history")
    live = _carrier_live_view(conn, physical, payload["target_message_id"])
    target, ids, head, replacement = rewind_in_transaction(
        store, conn, physical, payload["target_message_id"],
        preserve_compaction_handoff=live is not None or payload.get("preserve_compaction_handoff", False))
    # Receipts are JSON: publish the same public message shape get_messages() returns.
    target = _row_to_message_dict(target, warn_context="rewind receipt", summary_flag=True)
    if live is not None:
        target["content"] = live.get("content")
    target["content"] = typed_prompt_content(target.get("content"), target)
    conn.execute("UPDATE sessions SET runtime_generation = runtime_generation + 1 WHERE id = %s", (session_id,))
    return {session_id, physical}, {"rewound_count": len(ids), "target_message": target,
                                   "new_head_id": head, "replacement_message_id": replacement}


# ── the picks and the frozen-policy refresh (the reference's ``_runtime_picks``) ────────────────


def _commit_policy(conn, session_id: str, prepared: dict) -> tuple[dict, int]:
    """Write the prepared policy over the snapshot it was derived from and bump the generation."""
    current = _validate_prepared(conn, session_id, prepared)
    snapshot = current
    saved = snapshot["receipt"]
    saved["policy"] = prepared["policy"]
    conn.execute("UPDATE state_meta SET value = %s WHERE key = %s", (canonical_json(saved), POLICY_PREFIX + session_id))
    conn.execute("UPDATE sessions SET runtime_generation = runtime_generation + 1 WHERE id = %s", (session_id,))
    generation = conn.execute(
        "SELECT runtime_generation FROM sessions WHERE id = %s", (session_id,)).fetchone()["runtime_generation"]
    return snapshot, generation


def _validate_prepared(conn, session_id: str, prepared) -> dict:
    if not isinstance(prepared, dict) or "snapshot" not in prepared:
        raise AuthorityRefusal("runtime_coordination_required")
    current = _local_snapshot(conn, session_id)
    if current != prepared["snapshot"]:
        raise AuthorityRefusal("revision_conflict")
    return current


def _policy_target_row(conn, session_id: str) -> Optional[dict]:
    return conn.execute("SELECT * FROM sessions WHERE id = %s", (session_id,)).fetchone()


def _model_in_transaction(conn, store: PostgresStore, session_id: str, payload: dict, prepared: dict) -> tuple[set, dict]:
    snapshot, generation = _commit_policy(conn, session_id, prepared)
    model = prepared["policy"]["model"]
    config = json.loads(prepared["policy"]["config_json"])["model"]
    target = snapshot["target"]
    row = conn.execute("SELECT model_config FROM sessions WHERE id = %s", (target,)).fetchone()
    model_config = _parse_model_config(row["model_config"])
    model_config["provider"] = config["provider"]
    conn.execute("UPDATE sessions SET model = %s, model_config = %s WHERE id = %s",
                 (model, canonical_json(model_config), target))
    # This selection is newer than any model pick still waiting for a turn boundary: that pick is
    # superseded. A level deferred beside it stays unless this selection carries its own.
    key = DEFERRED_PICK_PREFIX + session_id
    row = conn.execute("SELECT value FROM state_meta WHERE key = %s", (key,)).fetchone()
    level = json.loads(row["value"]).get("reasoning") if row is not None and "reasoning" not in payload else None
    if level is None:
        conn.execute("DELETE FROM state_meta WHERE key = %s", (key,))
    else:
        conn.execute("UPDATE state_meta SET value = %s WHERE key = %s", (canonical_json({"reasoning": level}), key))
    return {session_id, target}, {"model": model, "provider": config["provider"],
                                 "execution_generation": generation,
                                 **({"effort": payload["reasoning"]} if "reasoning" in payload else {})}


def _reasoning_in_transaction(conn, session_id: str, payload: dict, prepared: dict) -> tuple[set, dict]:
    snapshot, generation = _commit_policy(conn, session_id, prepared)
    key = DEFERRED_PICK_PREFIX + session_id
    row = conn.execute("SELECT value FROM state_meta WHERE key = %s", (key,)).fetchone()
    if row is not None:
        pick = json.loads(row["value"])
        pick.pop("reasoning", None)
        pick.get("payload", {}).pop("reasoning", None)
        if "application_id" in pick:
            conn.execute("UPDATE state_meta SET value = %s WHERE key = %s", (canonical_json(pick), key))
        else:
            conn.execute("DELETE FROM state_meta WHERE key = %s", (key,))
    return {session_id, snapshot["target"]}, {"effort": payload["effort"], "execution_generation": generation}


def _workspace_in_transaction(conn, session_id: str, payload: dict, prepared: dict) -> tuple[set, dict]:
    """The policy (execution cwd + TERMINAL_CWD) and every row of the conversation move together."""
    snapshot, generation = _commit_policy(conn, session_id, prepared)
    cwd, git = prepared["policy"]["cwd"], prepared["git"]
    rows = list(dict.fromkeys([*_local_conversation_ids(conn, session_id), snapshot["target"]]))
    for sid in rows:
        conn.execute(
            "UPDATE sessions SET cwd = %s, git_branch = %s, git_repo_root = %s, "
            "git_metadata_generation = COALESCE(git_metadata_generation, 0) + 1 WHERE id = %s",
            (cwd, git["branch"] or None, git["root"] or None, sid))
    return {session_id, snapshot["target"]}, {"cwd": cwd, "branch": git["branch"] or None,
                                             "git_repo_root": git["root"] or None,
                                             "execution_generation": generation}


def _editor_mcp_in_transaction(conn, session_id: str, payload: dict, prepared: dict) -> tuple[set, dict]:
    snapshot, generation = _commit_policy(conn, session_id, prepared)
    return {session_id, snapshot["target"]}, {"tools_changed": prepared["tools_changed"],
                                             "mcp_servers": prepared["mcp_servers"],
                                             "execution_generation": generation}


def _edit_policy_in_transaction(conn, session_id: str, payload: dict, prepared: dict) -> tuple[set, dict]:
    snapshot, generation = _commit_policy(conn, session_id, prepared)
    return {session_id, snapshot["target"]}, {"edit_approval_policy": payload["policy"],
                                              "execution_generation": generation}


_POLICY_HANDLERS = {
    "model": "model", "reasoning": "reasoning", "workspace": "workspace",
    "editor_mcp": "editor_mcp", "edit_policy": "edit_policy",
}


# The receipts' durable picks (the reference's ``picks_on`` over the PG dialect: state_meta's JSON
# validated by ``hermes_json_valid``, read with ``::jsonb`` casts; the LIKE prefix is a bound
# parameter, never a literal -- psycopg's placeholder scan does not look inside string literals).
_RECEIPTS_SQL = (
    "FROM state_meta WHERE key LIKE %s AND hermes_json_valid(value) "
    "AND (value::jsonb -> 'result' ->> 'operation') IN ('model','reasoning','workspace') "
    "AND COALESCE(value::jsonb -> 'result' ->> 'status', '') <> 'deferred' "
    "AND value::jsonb -> 'result' ->> 'session_id' = %s"
)


def _picks_on(store: PostgresStore, session_id: str, before: int | None = None) -> set:
    """The policy edits a ``session.mutate`` committed for *session_id* (below revision *before*):
    their receipts, the stored policy's ``picks``, or its branch parent's picks below the branch's
    revision."""
    params: list = [_MUTATION_PREFIX + "%", session_id]
    sql = ("SELECT (value::jsonb -> 'result' ->> 'operation') AS op " + _RECEIPTS_SQL)
    if before is not None:
        sql += " AND (value::jsonb -> 'result' -> 'revision') < to_jsonb(%s)"
        params.append(before)
    picked = {row["op"] for row in store.all(sql, *params)}
    saved = store.one("SELECT value FROM state_meta WHERE key = %s", POLICY_PREFIX + session_id)
    if saved is None:
        return picked
    receipt = json.loads(saved["value"])
    if "picks" in receipt:
        return picked | set(receipt["picks"])
    branched = store.one(
        "SELECT value::jsonb -> 'result' ->> 'session_id' AS s, "
        "(value::jsonb -> 'result' -> 'revision')::int AS r "
        "FROM state_meta WHERE key LIKE %s AND hermes_json_valid(value) "
        "AND (value::jsonb -> 'result' ->> 'operation') = 'branch' "
        "AND value::jsonb -> 'result' ->> 'branched_session_id' = %s",
        _MUTATION_PREFIX + "%", session_id)
    if branched is not None:
        return picked | _picks_on(store, branched["s"], branched["r"])
    return picked


# ── the publish-side commands (the reference's ``_mutation_publish``) ───────────────────────────


def _child_policy(policy: dict, child: str, prepared: Optional[dict]) -> dict:
    """The parent's frozen policy, except editor MCP servers: the gateway prepared the child's own
    selection under the child's own scope. A branch starts at Default (``ask``)."""
    if "edit_approval_policy" in policy:
        policy = {**policy, "edit_approval_policy": "ask"}
    if prepared is None:
        if policy.get("editor_mcp_json"):
            raise AuthorityRefusal("runtime_coordination_required")
        return policy
    if child != prepared["child"] or policy.get("editor_mcp_json") != prepared["parent_editor_mcp_json"]:
        raise AuthorityRefusal("revision_conflict")
    return {**policy, "editor_mcp_json": prepared["editor_mcp_json"], "toolsets": prepared["toolsets"],
            "config_secret_ref": prepared["config_secret_ref"]}


def _branch_in_transaction(self, conn, session_id: str, payload: dict,
                           prepared: Optional[dict]) -> tuple[set, dict]:
    """An independent local route copying the active transcript through ``through_message_id``
    (the reference's ``branch_in_transaction``; the child row and its rows publish through the
    lineage repository's connection-level bodies)."""
    from session_rules.route_entries import Platform, SessionEntry, SessionSource, _now, build_session_key
    row = conn.execute("SELECT value FROM state_meta WHERE key = %s", (POLICY_PREFIX + session_id,)).fetchone()
    if row is None:
        raise AuthorityRefusal("runtime_coordination_required")
    saved = json.loads(row["value"])
    target = _validate_local_lineage(conn, saved)
    _require_idle(conn, list({session_id, target}))
    request_id = prepared["request_id"] if prepared else "branch:" + uuid.uuid4().hex
    child = str(local_session_id(saved["profile_id"], saved["principal_id"], request_id))
    policy = _child_policy(saved["policy"], child, prepared)
    # The child keeps the parent's owning profile: a served secondary's origin names it.
    source = SessionSource(Platform.LOCAL, child, user_id=saved["principal_id"], chat_type="dm",
                          profile=saved["entry"]["origin"].get("profile"))
    route = build_session_key(source, profile=_profile_from_session_key(saved["route"]))
    now = _now()
    entry = SessionEntry(route, child, now, now, origin=source, platform=Platform.LOCAL)
    parent = conn.execute("SELECT * FROM sessions WHERE id = %s", (target,)).fetchone()
    self._store.lineage._publish_child_session_row(
        conn, parent, parent_session_id=target, child_session_id=child, source=policy["source"],
        model=policy["model"], model_config={"_branched_from": target}, system_prompt=None,
        cwd=policy["cwd"], profile_name=parent["profile_name"])
    conn.execute("UPDATE sessions SET session_key = %s, chat_id = %s, origin_json = %s, system_prompt_hash = %s WHERE id = %s",
                 (route, child, canonical_json(source.to_dict()), parent["system_prompt_hash"], child))
    boundary = _branch_boundary(conn, target, payload.get("through_message_id"))
    ids, tools = self._store.lineage._tail_rows_after_watermark(
        conn, "SELECT id, tool_calls FROM messages WHERE session_id = %s AND active = 1 AND id <= %s ORDER BY id",
        [target, boundary])
    self._store.lineage._clone_message_rows(conn, ids, session_id=child)
    conn.execute("UPDATE sessions SET message_count = %s, tool_call_count = %s WHERE id = %s",
                 (len(ids), tools, child))
    if payload.get("title"):
        _set_session_title_in_transaction(conn, child, payload["title"], source=TITLE_SOURCE_USER)
    else:
        # Untitled branch: the next name in the parent's lineage ("my session" -> "my session #2").
        _set_session_title_in_transaction(conn, child,
                                          next_title_in_lineage(self._store, parent["title"] or "branch"),
                                          source=TITLE_SOURCE_DERIVED)
    receipt = dict(profile_id=saved["profile_id"], principal_id=saved["principal_id"],
                   request_id=request_id, session_id=child, route=route, entry=entry.to_dict(), policy=policy)
    # The copied policy holds the parent's picks; the child's refresh must keep them as picks.
    receipt["picks"] = sorted(_picks_on(self._store, session_id))
    conn.execute("INSERT INTO gateway_routing (scope, session_key, entry_json, updated_at) VALUES ('', %s, %s, %s)",
                 (route, canonical_json(entry.to_dict()), now.timestamp()))
    conn.execute("INSERT INTO state_meta (key, value) VALUES (%s, %s)",
                 (POLICY_PREFIX + child, canonical_json(receipt)))
    return {session_id}, {"branched_session_id": child, "copied_messages": len(ids)}


def _branch_boundary(conn, target: str, through: Optional[int]) -> int:
    """Highest row id the branch copies (the reference's ``_branch_boundary``); an assistant
    boundary keeps the tool results that directly answer its calls."""
    if through is None:
        return conn.execute(
            "SELECT COALESCE(MAX(id), 0) FROM messages WHERE session_id = %s AND active = 1",
            (target,)).fetchone()[0]
    row = conn.execute("SELECT role FROM messages WHERE id = %s AND session_id = %s AND active = 1",
                      (through, target)).fetchone()
    if row is None:
        raise AuthorityRefusal("invalid_params")
    if row["role"] == "assistant":
        for later in conn.execute(
                "SELECT id, role FROM messages WHERE session_id = %s AND active = 1 AND id > %s ORDER BY id",
                (target, through)).fetchall():
            if later["role"] != "tool":
                break
            through = later["id"]
    return through


def _profile_from_session_key(session_key: str) -> Optional[str]:
    """Profile namespace encoded in an ``agent:<ns>:...`` gateway session key, else None (the
    reference's ``_profile_from_session_key``)."""
    if not session_key:
        return None
    parts = str(session_key).split(":")
    if len(parts) < 2 or parts[0] != "agent":
        return None
    namespace = parts[1] or "main"
    if namespace == "main":
        return "default"
    return "main" if namespace == "main~" else namespace


def _compress_in_transaction(self, conn, session_id: str, payload: dict, prepared: dict) -> tuple[set, dict]:
    """Commit a prepared summary (the reference's ``compress_in_transaction``); a rotation moves
    the conversation's name to the new segment through the store's publish bodies."""
    snapshot = _validate_prepared(conn, session_id, prepared)
    policy = snapshot["receipt"]["policy"]
    target = snapshot["target"]
    if prepared["in_place"]:
        child = target
        # The in-place body on this receipt's connection: the active rows archive, the compacted
        # set inserts under the same id (the store's fuller coverage body stays on the
        # non-receipted path, documented in the module's D8 note).
        conn.execute("UPDATE messages SET active = 0, compacted = 1 WHERE session_id = %s AND active = 1",
                     (target,))
        if prepared["messages"]:
            self._store.lineage._insert_message_rows(conn, target, prepared["messages"])
        message_count, tool_call_count = _active_transcript_counts(conn, target)
        conn.execute("UPDATE sessions SET message_count = %s, tool_call_count = %s WHERE id = %s",
                     (message_count, tool_call_count, target))
        affected = {session_id, target}
    else:
        child = uuid.uuid4().hex
        self._store.lineage._publish_compression_child_on_conn(
            conn, parent_session_id=target, child_session_id=child, source=policy["source"],
            messages=prepared["messages"], model=policy["model"],
            model_config={"_compressed_from": target}, system_prompt=None, cwd=policy["cwd"],
            profile_name=None, compression_lock_holder=None, require_compression_lease=False,
            require_lease_refresh=False, lease_ttl_seconds=300.0, watermark=None, watermark_ceiling=None)
        # The conversation's name moves to the live segment, as an automatic rotation's does.
        _title_follows_head(conn, target, child)
        affected = set()  # Rotation publication already advances the logical revision.
    conn.execute("UPDATE sessions SET runtime_generation = runtime_generation + 1 WHERE id = %s", (session_id,))
    generation = conn.execute(
        "SELECT runtime_generation FROM sessions WHERE id = %s", (session_id,)).fetchone()["runtime_generation"]
    return affected, {"target_session_id": child, "previous_target_session_id": target,
                      "execution_generation": generation, "message_count": len(prepared["messages"]),
                      "summary": prepared.get("summary")}


def _reset_in_transaction(conn, session_id: str, payload: dict, store: PostgresStore) -> tuple[set, dict]:
    """Publish a fresh physical target for a local conversation (the reference's
    ``reset_in_transaction``; the worker-discard half runs through the ledger's own rows)."""
    saved = conn.execute("SELECT value FROM state_meta WHERE key = %s", (POLICY_PREFIX + session_id,)).fetchone()
    if saved is None:
        raise AuthorityRefusal("runtime_coordination_required")
    receipt = json.loads(saved["value"])
    target = _validate_local_lineage(conn, receipt)
    _discard_orphan_workers(conn, _runtime_lineage(conn, session_id))
    _require_target_advanceable(conn, list({session_id, target}))
    parent = conn.execute("SELECT * FROM sessions WHERE id = %s", (target,)).fetchone()
    policy = receipt["policy"]
    previous = receipt["entry"]
    child_id = uuid.uuid4().hex
    routed = dict(previous, session_id=child_id)
    store.lineage._publish_child_session_row(
        conn, parent, parent_session_id=target, child_session_id=child_id, source=policy["source"],
        model=policy["model"], model_config={"_reset_from": target}, system_prompt=None,
        cwd=policy["cwd"], profile_name=parent["profile_name"])
    conn.execute("UPDATE sessions SET ended_at = %s, end_reason = 'session_reset' WHERE id = %s",
                 (time.time(), target))
    _bump_conversation_generation(conn, target, "session_reset")
    store.lineage._advance_local_target(conn, target, child_id)
    conn.execute("UPDATE sessions SET runtime_generation = runtime_generation + 1 WHERE id = %s", (session_id,))
    generation = conn.execute(
        "SELECT runtime_generation FROM sessions WHERE id = %s", (session_id,)).fetchone()["runtime_generation"]
    # advance_local_target already advances the logical owner's revision.
    return set(), {"target_session_id": child_id, "previous_target_session_id": target,
                  "execution_generation": generation}


def _runtime_lineage(conn, session_id: str) -> set:
    """The session's worker-facing lineage (the reference's ``runtime_lineage`` over the PG
    chain-step rule): the physical target, the receipt's segments and their compression chains."""
    targets = {session_id, _local_physical_target(conn, session_id)}
    row = conn.execute("SELECT chat_id FROM sessions WHERE id = %s", (session_id,)).fetchone()
    if row and row["chat_id"]:
        saved = conn.execute("SELECT value FROM state_meta WHERE key = %s", (POLICY_PREFIX + row["chat_id"],)).fetchone()
        if saved is not None and session_id in json.loads(saved["value"]).get("lineage", [row["chat_id"]]):
            targets.update({row["chat_id"], _local_physical_target(conn, row["chat_id"])})
    frontier = list(targets)
    for _ in range(_CHAIN_CAP):
        frontier = [sid for sid in _expand_compression_lineage_ids(conn, frontier) if sid not in targets]
        if not frontier:
            return targets
        targets.update(frontier)
    raise AuthorityRefusal("storage_unavailable")


def _discard_orphan_workers(conn, session_ids) -> None:
    """An explicit reset revokes unlinked unknown compute; it never replays its side effects
    (the reference's ``discard_orphan_workers_on_reset``; admission-linked work still requires
    its own discard receipt, a running/registered worker still refuses)."""
    from ..conversation.sessions import _retire_terminal_receipts as _retire
    for sid in session_ids:
        orphans = [row["execution_id"] for row in conn.execute(
            "SELECT execution_id FROM worker_executions WHERE session_id = %s AND status = 'unknown' "
            "AND execution_id NOT LIKE 'admission-worker:%%'", (sid,)).fetchall()]
        for execution_id in orphans:
            conn.execute("UPDATE worker_executions SET status = 'terminal' WHERE execution_id = %s", (execution_id,))


# ── the import command (the reference's ``_import`` over the PG dialect) ────────────────────────


def _attach_import_parents(conn, parent_updates: list) -> int:
    """Re-attach imported children whose parent exists without creating a cycle (the reference's
    ``_attach_import_parents``; only the closing edge of a cycle is dropped)."""
    parent_by_child = dict(parent_updates)

    def _would_create_cycle(session_id: str, parent_id: str) -> bool:
        seen = {session_id}
        current = parent_id
        while current:
            if current in seen:
                return True
            seen.add(current)
            if current in parent_by_child:
                current = parent_by_child[current]
                continue
            row = conn.execute(
                "SELECT parent_session_id FROM sessions WHERE id = %s LIMIT 1", (current,)).fetchone()
            if row is None:
                return False
            current = row["parent_session_id"]
        return False

    detached = 0
    for session_id, parent_id in parent_updates:
        parent_exists = conn.execute(
            "SELECT 1 FROM sessions WHERE id = %s LIMIT 1", (parent_id,)).fetchone()
        if parent_exists and not _would_create_cycle(session_id, parent_id):
            conn.execute("UPDATE sessions SET parent_session_id = %s WHERE id = %s", (parent_id, session_id))
        else:
            parent_by_child.pop(session_id, None)
            detached += 1
    return detached


def _import_session_row(conn, store: PostgresStore, session_id: str, raw: dict,
                        messages: list) -> None:
    """INSERT one normalized session + its messages (the reference's ``_import_session_row`` on the
    PG dialect: the session row through this store's columns, the rows through the lineage
    repository's inserter)."""
    from persistence.sqlite.conversation._store.messages_sql import _parse_tool_calls, _tool_calls_count
    from persistence.sqlite.conversation._store.rules import coerce_epoch
    from persistence.sqlite.housekeeping._portability import (
        _IMPORT_FLAG_COLS, _IMPORT_FLOAT_COLS, _IMPORT_INT_COLS, _IMPORT_MESSAGE_JSON_FIELDS,
        _IMPORT_PASSTHROUGH_COLS, _coerce_or, _import_float_or_finite, _import_int_or_bounded)
    from utils import safe_json_loads

    started_at = coerce_epoch(raw.get("started_at"), session_id=session_id, field="started_at")
    params = {
        "id": session_id, "source": str(raw.get("source") or "import"),
        "system_prompt_hash": store.lineage._store_system_prompt(conn, raw.get("system_prompt")),
        "started_at": time.time() if started_at is None else started_at,
        **{col: 1 if _coerce_or(raw.get(col), int, 0) else 0 for col in _IMPORT_FLAG_COLS},
        **{col: raw.get(col) for col in _IMPORT_PASSTHROUGH_COLS},
        **{col: _import_float_or_finite(raw.get(col), col) for col in _IMPORT_FLOAT_COLS},
        **{col: _import_int_or_bounded(raw.get(col), col, 0) for col in _IMPORT_INT_COLS},
    }
    params["auto_archived"] &= params["archived"]  # only the sweep sets it, and only on rows it archives
    conn.execute(
        """INSERT INTO sessions (
               id, source, user_id, model, model_config, system_prompt_hash,
               started_at, ended_at, end_reason, input_tokens, output_tokens,
               cache_read_tokens, cache_write_tokens, reasoning_tokens,
               cwd, git_branch, git_repo_root, billing_provider, billing_base_url,
               billing_mode, estimated_cost_usd, actual_cost_usd, cost_status,
               cost_source, pricing_version, title, api_call_count, archived,
               auto_archived, pinned, hidden)
           VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                   %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
        (params["id"], params["source"], params["user_id"], params["model"], params["model_config"],
         params["system_prompt_hash"], params["started_at"], params["ended_at"], params["end_reason"],
         params["input_tokens"], params["output_tokens"], params["cache_read_tokens"],
         params["cache_write_tokens"], params["reasoning_tokens"], params["cwd"], params["git_branch"],
         params["git_repo_root"], params["billing_provider"], params["billing_base_url"],
         params["billing_mode"], params["estimated_cost_usd"], params["actual_cost_usd"],
         params["cost_status"], params["cost_source"], params["pricing_version"], params["title"],
         params["api_call_count"], params["archived"], params["auto_archived"], params["pinned"],
         params["hidden"]))

    def _json_value(value):
        return safe_json_loads(value, default=value) if isinstance(value, str) else value

    sanitized_messages = [
        {**msg, **{key: _json_value(msg.get(key)) for key in _IMPORT_MESSAGE_JSON_FIELDS}} for msg in messages]
    # A row exported archived (``include_inactive``) must stay archived.
    live: list[dict] = []
    archived: list[dict] = []
    for msg in sanitized_messages:
        (live if _coerce_or(msg.get("active"), int, 1) else archived).append(msg)
    store.lineage._insert_message_rows(conn, session_id, sanitized_messages, prune_checkpoints=False)
    for msg in archived:
        conn.execute("UPDATE messages SET active = 0, compacted = %s WHERE id = %s",
                     (1 if _coerce_or(msg.get("compacted"), int, 0) else 0, msg["_row_id"]))
    store.lineage._prune_shadowed_checkpoints(conn, session_id, live)
    # Session counters count live rows only.
    conn.execute("UPDATE sessions SET message_count = %s, tool_call_count = %s WHERE id = %s",
                 (len(live),
                  sum(_tool_calls_count(_parse_tool_calls(msg.get("tool_calls"))) for msg in live),
                  session_id))


def _import_op(conn, store: PostgresStore, session_id: str, payload: dict,
               owner: Optional[OwnerRule]) -> tuple[set, dict]:
    from persistence.sqlite.conversation._store.core import _IMPORT_MAX_SESSIONS
    from persistence.sqlite.housekeeping._portability import _validate_import_payload

    rows = payload["sessions"]
    if len(rows) > _IMPORT_MAX_SESSIONS:
        raise AuthorityRefusal("invalid_params")
    normalized, errors = _validate_import_payload(rows)
    if errors or not normalized or session_id != normalized[0]["session"]["id"]:
        raise AuthorityRefusal("invalid_params")
    imported, skipped, parents = [], [], []
    for item in normalized:
        raw = item["session"]
        sid = raw["id"]
        if conn.execute("SELECT 1 FROM state_meta WHERE key = %s", (RETIRED_PREFIX + sid,)).fetchone():
            # Refused by name at import time: a row restored beside the fence would refuse every turn.
            raise AuthorityRefusal("admission_conflict", field="sessions",
                                  detail=f"session {sid} was deleted here; a deleted session is not restored")
        if conn.execute("SELECT 1 FROM sessions WHERE id = %s", (sid,)).fetchone():
            skipped.append(sid)
            continue
        _import_session_row(conn, store, sid, raw, item["messages"])
        if raw.get("parent_session_id"):
            parents.append((sid, raw["parent_session_id"]))
        imported.append(sid)
    # A parent outside this payload is kept only when it is history the importer already controls.
    fresh = set(imported)
    foreign = [(sid, parent) for sid, parent in parents if parent not in fresh and (
        _history_binding(conn, parent) is None
        or not bindings_agree(_history_binding(conn, parent), _history_binding(conn, sid), owner))]
    parents = [link for link in parents if link not in foreign]
    detached = _attach_import_parents(conn, parents) + len(foreign)
    return set(imported), {"ok": True, "imported": len(imported), "skipped": len(skipped),
                           "imported_ids": imported, "skipped_ids": skipped,
                           "detached": detached, "errors": []}


class PgRuntimeRepository:
    """The runtime authority of the bound profile's PostgreSQL schema."""

    def __init__(self, store: PostgresStore) -> None:
        self._store = store

    # ── shared steps the verbs share ─────────────────────────────────────────

    def _apply_action(self, conn, session_id: str, operation: str, payload: dict, *,
                      prepared: Optional[dict], media_root: Optional[str],
                      owner: Optional[OwnerRule]) -> tuple[set, dict]:
        """The command handlers that run inside the receipt write (the reference's ``apply_action``)."""
        from persistence.sqlite.conversation._store.titles import sanitize_title
        if operation == "compress":
            return _compress_in_transaction(self, conn, session_id, payload, prepared)
        if operation == "model":
            return _model_in_transaction(conn, self._store, session_id, payload, prepared)
        if operation == "reasoning":
            return _reasoning_in_transaction(conn, session_id, payload, prepared)
        if operation == "workspace":
            return _workspace_in_transaction(conn, session_id, payload, prepared)
        if operation == "editor_mcp":
            return _editor_mcp_in_transaction(conn, session_id, payload, prepared)
        if operation == "edit_policy":
            return _edit_policy_in_transaction(conn, session_id, payload, prepared)
        if operation == "branch":
            return _branch_in_transaction(self, conn, session_id, payload, prepared)
        if operation == "delete":
            return _delete_in_transaction(conn, session_id, media_root=media_root, owner=owner)
        if operation == "import":
            return _import_op(conn, self._store, session_id, payload, owner)
        if operation == "rewind":
            return _rewind_in_transaction(conn, self._store, session_id, payload)
        if operation == "reset":
            return _reset_in_transaction(conn, session_id, payload, self._store)
        # The metadata edits: rename, archive and the sidebar flags.
        affected: set = set()
        result: dict = {}
        for key, value in payload.items():
            if key == "title":
                # The listing shows a compressed conversation under its live tip's title, so a
                # rename addressed to an ended segment lands there; a committed rename advances
                # the whole lineage's revision.
                tip = _compression_tip_on(conn, session_id)
                written = _set_session_title_in_transaction(conn, tip, value, source=TITLE_SOURCE_USER)
                affected.update(written)
                if written:
                    lineage = [row["id"] for row in conn.execute(
                        _LINEAGE_CTE_SQL + "SELECT id FROM lineage", (session_id, session_id)).fetchall()]
                    affected.update(lineage)
                    if not sanitize_title(value):
                        # An untitled tip lists under an ancestor's title, so a clear clears the
                        # lineage, each row through the same title rules.
                        for sid in lineage:
                            affected.update(_set_session_title_in_transaction(
                                conn, sid, value, source=TITLE_SOURCE_USER))
                result[key] = conn.execute("SELECT title FROM sessions WHERE id = %s", (tip,)).fetchone()["title"]
            else:
                column = "last_read_at" if key == "unread" else key
                stored = (0.0 if value else time.time()) if key == "unread" else int(value)
                affected.update(_set_lineage_column_in_transaction(conn, column, session_id, stored))
                if key == "archived" and value:
                    # Desktop's Archive is this sidebar write: it retires the canonical Bot Chat.
                    affected.update(_retire_canonical_bot_chat_in_transaction(conn, session_id))
                result[key] = value
        return affected, result

    # ── writes ──────────────────────────────────────────────────────────────

    def begin_epoch(self, *, instance_id: str) -> int:
        """Advance the store's authority epoch (1 on first acquisition), record ``instance_id`` as
        holder, return it. This store has no profile directory, so there is no profile lock to be
        fenced out of: every bump is granted (the contract's no-home case)."""
        _text(instance_id)
        with translated():
            with self._store.tx() as conn:
                conn.execute(
                    """INSERT INTO runtime_epoch (singleton, epoch, instance_id) VALUES (1, 1, %s)
                       ON CONFLICT (singleton) DO UPDATE SET epoch = runtime_epoch.epoch + 1,
                          instance_id = excluded.instance_id""",
                    (instance_id,))
                return conn.execute("SELECT epoch FROM runtime_epoch WHERE singleton = 1").fetchone()["epoch"]

    def prepare_mutation(self, epoch: int, *, principal_id: str, session_id: str, request_id: str,
                         expected_revision: int, command: MutationCommand,
                         expected_generation: int | None = None,
                         live_guard: Any = None) -> LocalSnapshot | MutationOutcome:
        """``mutate``'s checks up to ``live_guard``, then the local snapshot; an exact retry
        returns its stored ``MutationOutcome``."""
        operation = getattr(command, "operation", None)
        payload = command.payload() if hasattr(command, "payload") else None
        payload = json.loads(canonical_json(payload))
        key, digest = _mutation_key(principal_id, session_id, request_id, operation, payload,
                                    expected_revision, expected_generation)

        def write(conn):
            _epoch(conn, epoch)
            replay = _replay_mutation_receipt(conn, key, digest)
            if replay is not None:
                return replay, None, True
            session = conn.execute("SELECT * FROM sessions WHERE id = %s", (session_id,)).fetchone()
            if session is None and operation != "import":
                raise AuthorityRefusal("not_found")
            revision = session["runtime_revision"] if session else 0
            if revision != expected_revision:
                raise AuthorityRefusal("revision_conflict")
            if expected_generation is not None and (session is None or session["runtime_generation"] != expected_generation):
                raise AuthorityRefusal("stale_generation")
            if live_guard is not None:
                # The owner's verdict refuses after the delete target walk, which may refuse first.
                if operation == "delete":
                    _delete_targets(conn, session_id, None)
                raise live_guard.refusal()
            snapshot = _local_snapshot(conn, session_id)
            if operation == "branch":
                # The branch commit's own idle rule (queued input too), before the owner starts servers.
                _require_idle(conn, list({session_id, snapshot["target"]}))
            return {"snapshot": snapshot}, None, False

        with _refused(self._store, session_id):
            with self._store.tx() as conn:
                result, _gone, replayed = write(conn)
        if replayed:
            return MutationOutcome(result=result, replayed=True, gone=None)
        snapshot = result["snapshot"]
        return LocalSnapshot(receipt=snapshot["receipt"], target=snapshot["target"],
                             target_revision=snapshot["target_revision"])

    def mutate(self, epoch: int, *, principal_id: str, session_id: str, request_id: str,
               expected_revision: int, command: MutationCommand, expected_generation: int | None = None,
               live_guard: Any = None, history: HistoryWrite | None = None,
               prepared: PreparedMutation | None = None, media_root: str | None = None,
               owner: OwnerRule | None) -> MutationOutcome:
        """Commit one closed mutation and its retry receipt (the reference's
        ``mutate_runtime_session``: the input checks, the fence order, the command handler, the
        affected-row bumps and the receipt, one transaction)."""
        for value in (principal_id, session_id, request_id):
            _text(value)
        if type(expected_revision) is not int or expected_revision < 0:
            raise AuthorityRefusal("invalid_params")
        operation = getattr(command, "operation", None)
        payload = command.payload() if hasattr(command, "payload") else None
        parse_mutation(operation, payload)
        payload = json.loads(canonical_json(payload))
        if expected_generation is not None and (type(expected_generation) is not int or expected_generation < 0):
            raise AuthorityRefusal("invalid_params")
        if operation in {"delete", "rewind", "reset", "branch", "model", "compress"} and expected_generation is None:
            raise AuthorityRefusal("invalid_params")
        # Snapshot caller data before waiting for the writer lock.
        key, digest = _mutation_key(principal_id, session_id, request_id, operation, payload,
                                   expected_revision, expected_generation)
        legacy_prepared = None
        if prepared is not None:
            legacy_prepared = {"snapshot": {"receipt": prepared.snapshot.receipt,
                                           "target": prepared.snapshot.target,
                                           "target_revision": prepared.snapshot.target_revision},
                              **prepared.values}

        from persistence.authority.runtime import SessionGone
        def write(conn):
            _epoch(conn, epoch)
            replay = _replay_mutation_receipt(conn, key, digest)
            if replay is not None:
                return replay, None, True
            session = conn.execute("SELECT * FROM sessions WHERE id = %s", (session_id,)).fetchone()
            if session is None and operation != "import":
                raise AuthorityRefusal("not_found")
            revision = session["runtime_revision"] if session else 0
            if revision != expected_revision:
                raise AuthorityRefusal("revision_conflict")
            if expected_generation is not None and (session is None or session["runtime_generation"] != expected_generation):
                raise AuthorityRefusal("stale_generation")
            if live_guard is not None:
                # The owner's verdict refuses after the delete target walk, which may refuse first.
                if operation == "delete":
                    _delete_targets(conn, session_id, owner)
                raise live_guard.refusal()
            if history is not None:
                _authorize_write(conn, history, session_id, owner)
            affected, projection = self._apply_action(conn, session_id, operation, payload,
                                                      prepared=legacy_prepared, media_root=media_root, owner=owner)
            gone = None
            if operation == "delete":
                gone = SessionGone(tuple(projection.get("deleted_ids") or (session_id,)), "delete")
            elif payload.get("archived") is True and session is not None and not session["archived"]:
                gone = SessionGone(tuple(affected) or (session_id,), "archive")
            for target in affected:
                conn.execute("UPDATE sessions SET runtime_revision = runtime_revision + 1 WHERE id = %s",
                             (target,))
            updated = conn.execute("SELECT * FROM sessions WHERE id = %s", (session_id,)).fetchone()
            result = {"session_id": session_id,
                      "revision": updated["runtime_revision"] if updated else expected_revision + 1,
                      "execution_generation": updated["runtime_generation"] if updated else expected_generation,
                      "operation": operation, **projection}
            conn.execute("INSERT INTO state_meta (key, value) VALUES (%s, %s)",
                         (key, canonical_json({"digest": digest, "result": result})))
            return result, gone, False

        with _refused(self._store, session_id):
            with self._store.tx() as conn:
                result, gone, replayed = write(conn)
        return MutationOutcome(result=result, replayed=replayed, gone=gone)

    def defer_model_pick(self, epoch: int, *, key: str, digest: str, session_id: str,
                        pick: DeferredPick) -> dict[str, Any]:
        """Keep ``pick`` in the session's deferred slot with its receipt (the reference's
        ``defer_pick``: an exact retry returns its receipt; a reasoning pick drops the slot model
        pick's own level, a model pick keeps an older level only when it carries none)."""
        parse_mutation(pick.operation, pick.payload)
        payload = dict(pick.payload)
        slot_pick = {"principal_id": pick.principal_id, "request_id": pick.request_id,
                     "payload": payload, "application_id": "deferred-pick:" + pick.request_id}

        def write(conn):
            _epoch(conn, epoch)
            replay = _replay_mutation_receipt(conn, key, digest)
            if replay is not None:
                return replay
            row = conn.execute(
                "SELECT runtime_revision, runtime_generation FROM sessions WHERE id = %s",
                (session_id,)).fetchone()
            if row is None:
                raise AuthorityRefusal("not_found")
            stored = conn.execute("SELECT value FROM state_meta WHERE key = %s",
                                 (DEFERRED_PICK_PREFIX + session_id,)).fetchone()
            current = json.loads(stored["value"]) if stored else {}
            if pick.operation == "reasoning":
                current.get("payload", {}).pop("reasoning", None)  # this level is newer than the one picked with the model
                value = {**current, "reasoning": slot_pick}
                picked = {"effort": payload["effort"]}
            else:
                older = "reasoning" in current and "reasoning" not in payload
                value = {**slot_pick, **({"reasoning": current["reasoning"]} if older else {})}
                picked = {"model": pick.shown or payload["model"]}
            result = {"session_id": session_id, "operation": pick.operation, "status": "deferred", **picked,
                      "revision": row["runtime_revision"], "execution_generation": row["runtime_generation"]}
            conn.execute("INSERT INTO state_meta (key, value) VALUES (%s, %s) "
                         "ON CONFLICT (key) DO UPDATE SET value = excluded.value",
                         (DEFERRED_PICK_PREFIX + session_id, canonical_json(value)))
            conn.execute("INSERT INTO state_meta (key, value) VALUES (%s, %s)",
                         (key, canonical_json({"digest": digest, "result": result})))
            return result

        with _refused(self._store, session_id):
            with self._store.tx() as conn:
                return write(conn)

    def forget_deferred_pick(self, session_id: str, *, stored: str) -> None:
        """Delete the deferred slot while it still holds exactly ``stored``; no epoch fence."""
        with _refused(self._store, session_id):
            with self._store.tx() as conn:
                conn.execute("DELETE FROM state_meta WHERE key = %s AND value = %s",
                             (DEFERRED_PICK_PREFIX + session_id, stored))

    def mark_deferred_pick_asked(self, session_id: str, *, stored: str, confirm: str) -> None:
        """Store the slot as asked (``awaiting_confirm``) while it still holds exactly ``stored``."""
        try:
            pick = json.loads(stored)
        except (TypeError, ValueError) as exc:
            raise AuthorityRefusal("invalid_params") from exc
        if not isinstance(pick, dict):
            raise AuthorityRefusal("invalid_params")
        value = canonical_json({**pick, "awaiting_confirm": confirm})
        with _refused(self._store, session_id):
            with self._store.tx() as conn:
                conn.execute("UPDATE state_meta SET value = %s WHERE key = %s AND value = %s",
                             (value, DEFERRED_PICK_PREFIX + session_id, stored))

    def replace_deferred_pick(self, session_id: str, *, stored: str,
                             pick: Mapping[str, Any]) -> str:
        """Store ``pick`` in canonical JSON as the slot while it still holds exactly ``stored``, and
        return that text whether or not it was stored; an unencodable pick is ``invalid_params``."""
        rest = canonical_json(dict(pick))
        with _refused(self._store, session_id):
            with self._store.tx() as conn:
                conn.execute("UPDATE state_meta SET value = %s WHERE key = %s AND value = %s",
                             (rest, DEFERRED_PICK_PREFIX + session_id, stored))
        return rest

    def commit_chat_pick(self, epoch: int, *, key: str, digest: str, session_id: str,
                        params: ChatPickParams, result: Mapping[str, Any]) -> dict[str, Any]:
        """Bump the mount's revision and store ``result`` plus that revision as the receipt,
        re-checking the fences (the reference's ``commit_chat_pick``); an exact retry returns its
        receipt."""
        params_dict = {"request_id": params.request_id, "operation": params.operation,
                       "payload": dict(params.payload), "expected_revision": params.expected_revision,
                       "expected_generation": params.expected_generation}

        def write(conn):
            _epoch(conn, epoch)
            replay = _replay_mutation_receipt(conn, key, digest)
            if replay is not None:
                return replay
            _fenced_row(conn, session_id, params_dict)
            conn.execute("UPDATE sessions SET runtime_revision = runtime_revision + 1 WHERE id = %s",
                         (session_id,))
            revision = conn.execute(
                "SELECT runtime_revision FROM sessions WHERE id = %s", (session_id,)).fetchone()["runtime_revision"]
            committed = {**result, "revision": revision}
            conn.execute("INSERT INTO state_meta (key, value) VALUES (%s, %s)",
                         (key, canonical_json({"digest": digest, "result": committed})))
            return committed

        with _refused(self._store, session_id):
            with self._store.tx() as conn:
                return write(conn)

    def commit_refreshed_policy(self, epoch: int, *, session_id: str, old: Mapping[str, Any],
                               new: Mapping[str, Any], generation: int) -> int:
        """The owner's config refresh of an idle session (the reference's ``commit_refreshed_policy``):
        the epoch, the local snapshot, ``require_idle``; the stored policy must be exactly ``old``
        at ``generation``, else ``revision_conflict``; ``new`` is written and the generation bumps."""
        def write(conn):
            _epoch(conn, epoch)
            snapshot = _local_snapshot(conn, session_id)
            _require_idle(conn, list({session_id, snapshot["target"]}))
            current = conn.execute("SELECT runtime_generation FROM sessions WHERE id = %s", (session_id,)).fetchone()
            if snapshot["receipt"]["policy"] != dict(old) or current is None or current["runtime_generation"] != generation:
                raise AuthorityRefusal("revision_conflict")
            return _commit_policy(conn, session_id, {"snapshot": snapshot, "policy": dict(new)})[1]

        with _refused(self._store, session_id):
            with self._store.tx() as conn:
                return write(conn)

    def store_viewer_close(self, epoch: int, *, session_id: str, record: str) -> None:
        """Store the pending viewer close under the epoch fence."""
        with _refused(self._store, session_id):
            with self._store.tx() as conn:
                _epoch(conn, epoch)
                conn.execute(
                    "INSERT INTO state_meta (key, value) VALUES (%s, %s) "
                    "ON CONFLICT (key) DO UPDATE SET value = excluded.value",
                    (VIEWER_CLOSE_PREFIX + session_id, record))

    def forget_viewer_close(self, epoch: int, *, session_id: str, record: str) -> bool:
        """Delete the stored close while it is exactly ``record``, under the epoch fence; whether a
        row went."""
        with _refused(self._store, session_id):
            with self._store.tx() as conn:
                _epoch(conn, epoch)
                cur = conn.execute("DELETE FROM state_meta WHERE key = %s AND value = %s",
                                  (VIEWER_CLOSE_PREFIX + session_id, record))
                return bool(cur.rowcount)

    def end_unless_ledger_busy(self, session_id: str, *, session_key: str | None, end_reason: str) -> bool:
        """End the row unless the ledger holds work for the session, its key, its ``local-`` chat id or
        its lineage owner (the reference's ``_end_unless_ledger_busy``); True when ended now or
        already ended. No epoch fence."""
        with _refused(self._store, session_id):
            with self._store.tx() as conn:
                row = conn.execute("SELECT chat_id FROM sessions WHERE id = %s", (session_id,)).fetchone()
                ids = {session_id, str(session_key or "")}
                if row is not None and str(row["chat_id"] or "").startswith("local-"):
                    ids.add(str(row["chat_id"]))
                ids.add(_local_lineage_owner(conn, session_id))
                for sid in filter(None, ids):
                    if conn.execute(_CANONICAL_PENDING_SQL, (sid, sid)).fetchone() is not None:
                        return False
                _end_and_bump(conn, session_id, end_reason)
                return True

    def end_after_settlement(self, epoch: int, *, session_id: str, admission_id: str,
                             generation: int, admission_epoch: int, reason: str) -> bool:
        """One conditional end (the reference's ``finalize_webhook``): the epoch is current,
        ``runtime_generation`` is ``generation``, the admission is terminal on this session at
        ``(generation, admission_epoch)``, nothing on the session is pending, the row is open.
        Every miss is False."""
        with _refused(self._store, session_id):
            with self._store.tx() as conn:
                cur = conn.execute(
                    """UPDATE sessions SET ended_at = %s, end_reason = %s
                       WHERE id = %s AND ended_at IS NULL AND runtime_generation = %s
                         AND EXISTS (SELECT 1 FROM runtime_epoch WHERE singleton = 1 AND epoch = %s)
                         AND EXISTS (SELECT 1 FROM session_admissions WHERE admission_id = %s
                                     AND target_session_id = %s AND status = 'terminal'
                                     AND generation = %s AND owner_epoch = %s)
                         AND NOT EXISTS (SELECT 1 FROM session_admissions
                                         WHERE target_session_id = %s AND status <> 'terminal')""",
                    (time.time(), reason, session_id, generation, epoch, admission_id,
                     session_id, generation, admission_epoch, session_id))
                if cur.rowcount:
                    _bump_conversation_generation(conn, session_id, reason)
                return bool(cur.rowcount)

    # ── reads ───────────────────────────────────────────────────────────────

    def current_epoch(self) -> int | None:
        """The stored epoch, ``None`` before the first ``begin_epoch``."""
        return self._store.scalar("SELECT epoch FROM runtime_epoch WHERE singleton = 1")

    def has_mutation_receipt(self, principal_id: str, session_id: str, request_id: str) -> bool:
        """Whether a mutation receipt exists for the identity, whatever its digest."""
        for value in (principal_id, session_id, request_id):
            _text(value)
        key, _digest = _mutation_key(principal_id, session_id, request_id, "model", {}, 0, None)
        return self._store.scalar("SELECT 1 FROM state_meta WHERE key = %s", key) is not None

    def deferred_pick(self, session_id: str) -> StoredDeferredPick | None:
        """The session's deferred slot."""
        stored = self._store.scalar("SELECT value FROM state_meta WHERE key = %s",
                                    DEFERRED_PICK_PREFIX + session_id)
        return None if stored is None else StoredDeferredPick(pick=json.loads(stored), stored=stored)

    def prepare_chat_pick(self, epoch: int, *, key: str, digest: str, session_id: str,
                         params: ChatPickParams) -> dict[str, Any] | SessionRuntimeState:
        """The stored receipt of an exact retry, else the fenced mount's counters: the epoch, the
        receipt (``admission_conflict``), the session (``not_found``), the revision
        (``revision_conflict``), the generation (``stale_generation``)."""
        params_dict = {"request_id": params.request_id, "operation": params.operation,
                       "payload": dict(params.payload), "expected_revision": params.expected_revision,
                       "expected_generation": params.expected_generation}
        with _refused(self._store, session_id):
            stored_epoch = self._store.scalar("SELECT epoch FROM runtime_epoch WHERE singleton = 1")
            if type(epoch) is not int or stored_epoch is None or stored_epoch != epoch:
                raise AuthorityRefusal("stale_epoch")
            stored = self._store.one("SELECT value FROM state_meta WHERE key = %s", key)
            if stored is not None:
                receipt = json.loads(stored["value"])
                if receipt["digest"] != digest:
                    raise AuthorityRefusal("admission_conflict")
                return receipt["result"]
            row = self._store.one("SELECT * FROM sessions WHERE id = %s", session_id)
            if row is None:
                raise AuthorityRefusal("not_found")
            if row["runtime_revision"] != params_dict["expected_revision"]:
                raise AuthorityRefusal("revision_conflict")
            if row["runtime_generation"] != params_dict["expected_generation"]:
                raise AuthorityRefusal("stale_generation")
            return SessionRuntimeState(revision=row["runtime_revision"],
                                      generation=row["runtime_generation"])

    def policy_picks(self, session_id: str) -> set[str]:
        """The policy operations the session's non-deferred receipts committed, its receipt's
        ``picks``, or its branch parent's picks below the branch's revision."""
        with _refused(self._store):
            return _picks_on(self._store, session_id)

    def retired_session(self, session_id: str) -> bool:
        """Whether the id carries a retirement marker."""
        return self._store.scalar("SELECT 1 FROM state_meta WHERE key = %s",
                                  RETIRED_PREFIX + session_id) is not None

    def first_generation(self, chain: GenerationChain, *, pinned: PinnedIdentity | None = None,
                         start: int = 0) -> GenerationPick:
        """From ``start``, the first generation whose id is not retired, or is retired and holds the
        ``pinned`` identity's tombstone, with its row's title, source and archived flag."""
        derive = _derive(chain)

        with _refused(self._store):
            generation = start
            while True:
                sid = derive(generation)
                retired = self._store.scalar(
                    "SELECT 1 FROM state_meta WHERE key = %s", RETIRED_PREFIX + sid) is not None
                if retired:
                    if pinned is not None and self._store.scalar(
                            "SELECT 1 FROM state_meta WHERE key = %s",
                            identity_key(pinned.principal_id, sid, pinned.request_id)) is not None:
                        return GenerationPick(generation=generation, session_id=sid, row=None)
                else:
                    row = self._store.one(
                        "SELECT title, source, archived FROM sessions WHERE id = %s", sid)
                    return GenerationPick(
                        generation=generation, session_id=sid,
                        row=None if row is None else GenerationRow(
                            title=row["title"], source=row["source"], archived=bool(row["archived"])))
                generation += 1

    def expiring_retirements(self, session_ids: Sequence[str]) -> set[str]:
        """The ids among *session_ids* whose own marker is a sweep's (it ages out at the horizon)."""
        rows = self._store.all(
            "SELECT key FROM state_meta WHERE hermes_json_valid(value) "
            "AND (value::jsonb ->> 'expires') = 'true' "
            "AND key IN (SELECT %s || u FROM unnest(%s::text[]) AS t(u))",
            RETIRED_PREFIX, list(session_ids))
        return {row["key"][len(RETIRED_PREFIX):] for row in rows}

    def viewer_close(self, session_id: str) -> str | None:
        """The stored close record."""
        return self._store.scalar("SELECT value FROM state_meta WHERE key = %s",
                                  VIEWER_CLOSE_PREFIX + session_id)

    def viewer_closes(self) -> list[tuple[str, str]]:
        """``(session_id, record)`` of every stored close, sorted."""
        rows = self._store.all("SELECT key, value FROM state_meta WHERE key LIKE %s",
                               VIEWER_CLOSE_PREFIX + "%")
        return sorted((key[len(VIEWER_CLOSE_PREFIX):], record) for key, record in
                      ((row["key"], row["value"]) for row in rows))

    def pending_ledger_status(self, session_key: str) -> str | None:
        """The status of a non-terminal admission targeted at ``session_key``, else of a
        non-terminal worker execution on it, else None, in one read."""
        return self._store.scalar(_CANONICAL_PENDING_SQL, session_key, session_key)

    def delete_footprint(self, session_id: SessionId, owner: OwnerRule | None) -> DeleteFootprint:
        """In one read, the target walk ``mutate`` with ``Delete`` runs under ``owner`` and every
        row whose parent is one of the targets; a missing row's targets are the id alone. Refuses
        ``storage_unavailable`` when the walk's local receipt lineage does not validate."""
        with _refused(self._store):
            # An uninitialised store reads empty (an absent SQLite file's answer); the walk's
            # helpers work on one read-only transaction of the handle's connection.
            if self._store._read_conn() is None:
                return DeleteFootprint(targets=(session_id,), children=())
            with self._store.tx() as conn:
                targets = _delete_targets(conn, session_id, owner)
                children = [row["id"] for row in conn.execute(
                    "SELECT id FROM sessions WHERE parent_session_id = ANY(%s)", (targets,)).fetchall()]
        return DeleteFootprint(targets=tuple(SessionId(sid) for sid in targets),
                               children=tuple(SessionId(sid) for sid in children))

    # ── the fixture seams (the suite's seeding verbs, on the ported repository) ──

    def seed_local_receipt(self, session_id: str, receipt: Mapping[str, Any]) -> None:
        """Store ``receipt`` where main keeps a local policy receipt (canonical JSON under
        ``POLICY_PREFIX + session_id``)."""
        with translated():
            with self._store.tx() as conn:
                conn.execute(
                    "INSERT INTO state_meta (key, value) VALUES (%s, %s) "
                    "ON CONFLICT (key) DO UPDATE SET value = excluded.value",
                    (POLICY_PREFIX + session_id, canonical_json(dict(receipt))))

    def seed_retirement(self, session_id: str, *, expires: bool = False) -> None:
        """Record the retirement marker of ``session_id`` alone (the store's ``_mark_retired``;
        ``expires`` makes it a sweep's marker)."""
        with translated():
            with self._store.tx() as conn:
                _mark_retired(conn, RETIRED_PREFIX + session_id, expires=expires, own=True)


def _fenced_row(conn, session_id: str, params: dict) -> dict:
    """The fenced mount row (the reference's ``_fenced_row``): ``not_found``, ``revision_conflict``,
    ``stale_generation``."""
    row = conn.execute("SELECT * FROM sessions WHERE id = %s", (session_id,)).fetchone()
    if row is None:
        raise AuthorityRefusal("not_found")
    if row["runtime_revision"] != params["expected_revision"]:
        raise AuthorityRefusal("revision_conflict")
    if row["runtime_generation"] != params["expected_generation"]:
        raise AuthorityRefusal("stale_generation")
    return row


def _derive(chain: GenerationChain):
    """The id of each generation of ``chain`` (the contract package's pure derivations)."""
    if isinstance(chain, ApiGenerations):
        return lambda n: str(api_generation_id(chain.session_id, n))
    if isinstance(chain, RoomGenerations):
        return lambda n: str(room_session_id(chain.identity, n))
    return lambda n: str(local_session_id(chain.profile_id, chain.principal_id,
                                          generation_request_id(chain.base_request_id, n)))


def _mutation_key(principal_id: str, session_id: str, request_id: str, operation: str,
                  payload: dict, expected_revision: int,
                  expected_generation: int | None) -> tuple[str, str]:
    """``(state_meta key, digest)`` of a mutation request's exact-retry receipt."""
    return mutation_receipt_key(principal_id, session_id, request_id, operation=operation,
                                payload=payload, expected_revision=expected_revision,
                                expected_generation=expected_generation)


add_stubs(PgRuntimeRepository, SessionRuntimeRepository, implemented=(
    "begin_epoch", "prepare_mutation", "mutate", "defer_model_pick", "forget_deferred_pick",
    "mark_deferred_pick_asked", "replace_deferred_pick", "commit_chat_pick",
    "commit_refreshed_policy", "store_viewer_close", "forget_viewer_close", "end_unless_ledger_busy",
    "end_after_settlement", "current_epoch", "has_mutation_receipt", "deferred_pick",
    "prepare_chat_pick", "policy_picks", "retired_session", "first_generation",
    "expiring_retirements", "viewer_close", "viewer_closes", "pending_ledger_status",
    "delete_footprint",
))


def _register() -> None:
    register_repository("runtime", lambda store: PgRuntimeRepository(store), {"authority_fences": True})


_register()

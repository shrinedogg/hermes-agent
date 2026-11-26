"""User-turn rewind for the ported store: the transaction body, the plan reads and the rewind write
(the SQLite ``rewind.py`` + ``mutation_transcript.py`` ports, connection-first, ``%s`` placeholders,
and the bytea payload rule: stored ``content`` is decoded before the turn-view helpers touch it)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from session_rules.turns import sanitize_context, split_user_originated_turn
from persistence.conversation.messages import CONTENT_REFUSED_DISPLAY_KIND, CONTENT_REFUSED_METADATA_KEY
from persistence.sqlite.conversation._store.rules import DB_ROW_SNAPSHOT
from persistence.sqlite.conversation._store.messages_sql import _tool_calls_len
from .common import (
    _ACTIVE_IDS_SQL, _SET_COUNTERS_SQL, _check_transcript_write_guards, _decode_content,
    _decode_display_metadata, _decode_payloads, _payload_str, _transcript_write,
)

__all__ = [
    "RewindPlan", "RewindOutcome", "RewindTargetUnavailableError", "plan_user_turn_rewind",
    "retain_refused_turn_for_display", "rewind_in_transaction", "rewind_user_turn",
]

_HISTORY_CHANGED = "session history changed before the rewind could be persisted"


class RewindTargetUnavailableError(ValueError):
    """The requested user turn is not a rewindable target of the active transcript: no user turns, an
    ordinal past the newest one, a row that is not user-originated, or a plain turn where the caller
    required a compaction carrier. Surfaces map this to their own "nothing to undo" message."""


@dataclass
class RewindPlan:
    target_row_id: int
    expected_active_ids: list[int]
    expected_target_content: Any
    scaffold: Optional[dict[str, Any]]
    prefix: list[dict[str, Any]]
    durable_prefix: list[dict[str, Any]]
    live_view: dict[str, Any]
    live_text: Optional[str]
    durable_user: list[int]
    user_ordinal: int


@dataclass
class RewindOutcome:
    prefix: list[dict[str, Any]]
    live_view: dict[str, Any]
    live_text: str
    rewound_count: int
    turns_undone: int


def _user_indices(messages: list[dict[str, Any]]) -> list[int]:
    from session_rules.turns import user_originated_turn_view
    return [i for i, m in enumerate(messages) if user_originated_turn_view(m) is not None]


def _comparison_content(message: dict[str, Any]) -> Any:
    """Project content the way the durable row stores it (flush projection, then the read-side
    sanitize) so a warm row and its durable twin compare equal."""
    from session_rules.turns import _durable_content
    from .common import _loaded_view_content
    return _loaded_view_content(message.get("role"), _durable_content(message.get("content")))


def rewind_in_transaction(store, conn, session_id: str, target_message_id: int, *,
                          preserve_compaction_handoff: bool = False, expected_active_ids: Optional[list[int]] = None,
                          expected_target_content: Any = None):
    """The rewind body run inside the caller's write transaction (the SQLite
    ``mutation_transcript.rewind_in_transaction`` port): the guards, the active-id pin, the target
    validation, the soft-delete of the target and every later active row, the handoff scaffold insert
    (the replacement's id comes from the insert's ``_row_id`` stamp, not ``last_insert_rowid``), the
    ``rewind_count`` bump and the counter reset."""
    from .writes import _insert_message_rows
    _check_transcript_write_guards(conn, session_id, None,
                                   reject_active_turn_lease=True, reject_active_compression_lock=True)
    if expected_active_ids is not None:
        active_rows = conn.execute(_ACTIVE_IDS_SQL, (session_id,)).fetchall()
        if [int(r["id"]) for r in active_rows] != expected_active_ids:
            raise RuntimeError("active transcript changed before the rewind could be persisted")
    row = conn.execute(
        "SELECT * FROM messages WHERE id = %s AND session_id = %s", (target_message_id, session_id)).fetchone()
    if row is None:
        raise ValueError(f"message {target_message_id} not found in session {session_id}")
    target_row = dict(row)
    if target_row.get("role") != "user":
        raise ValueError(
            f"rewind target must be a 'user' message (got role={target_row.get('role')!r}, id={target_message_id})")
    replacement_message_id = replacement = None
    if preserve_compaction_handoff or expected_target_content is not None:
        replacement = _split_rewind_target(target_row, expected_target_content, preserve_compaction_handoff)
    ids = [r["id"] for r in conn.execute(
        "SELECT id FROM messages WHERE session_id = %s AND id >= %s AND active = 1",
        (session_id, target_message_id)).fetchall()]
    if ids:
        from .common import _in_placeholders
        conn.execute(f"UPDATE messages SET active = 0 WHERE id IN ({_in_placeholders(len(ids))})", ids)
    if replacement is not None:
        _insert_message_rows(conn, session_id, [replacement])
        replacement_message_id = int(replacement["_row_id"])
    conn.execute(
        "UPDATE sessions SET rewind_count = COALESCE(rewind_count, 0) + 1 WHERE id = %s", (session_id,))
    message_count, tool_call_count = _active_transcript_counts(conn, session_id)
    conn.execute(f"{_SET_COUNTERS_SQL} WHERE id = %s", (message_count, tool_call_count, session_id))
    head = conn.execute(
        "SELECT MAX(id) AS head FROM messages WHERE session_id = %s AND active = 1", (session_id,)).fetchone()
    head_id = int(head["head"]) if head is not None and head["head"] is not None else None
    return target_row, ids, head_id, replacement_message_id


def _active_transcript_counts(conn, session_id: str) -> tuple[int, int]:
    """The active message/tool-call counts inside the caller's txn."""
    rows = conn.execute(
        "SELECT tool_calls FROM messages WHERE session_id = %s AND active = 1", (session_id,)).fetchall()
    return len(rows), sum(_tool_calls_len(row["tool_calls"], scalar=1) for row in rows)


def rewind_to_message(store, session_id: str, target_message_id: int, *,
                      preserve_compaction_handoff: bool = False,
                      expected_active_ids: Optional[list[int]] = None,
                      expected_target_content: Any = None) -> dict[str, Any]:
    """Soft-delete (``active=0``) every message with id >= *target_message_id*, target included (the caller
    pre-fills it as the next prompt); returns ``{"rewound_count", "target_message", "new_head_id"}`` plus
    ``replacement_message_id``/``replacement_message_uid`` with ``preserve_compaction_handoff`` (the SQLite
    ``rewind_to_message`` port; the payload-decoded target row comes back for the prompt-buffer prefill)."""
    def _do(conn):
        target_row, ids, head_id, replacement_id = rewind_in_transaction(
            store, conn, session_id, target_message_id,
            preserve_compaction_handoff=preserve_compaction_handoff,
            expected_active_ids=expected_active_ids, expected_target_content=expected_target_content)
        # The scaffold insert stamped a ``message_uid``; read it back in the same txn so the caller can
        # address the new head by its durable identity.
        replacement_uid = None
        if replacement_id:
            row = conn.execute("SELECT message_uid FROM messages WHERE id = %s", (replacement_id,)).fetchone()
            replacement_uid = row["message_uid"] if row is not None else None
        return target_row, ids, head_id, replacement_id, replacement_uid

    target_row, rewound, new_head_id, replacement_message_id, replacement_message_uid = \
        _transcript_write(store, [], lambda conn: _do(conn))
    # Decode for the prompt-buffer prefill without a second fallible DB operation (the bytea payload rule).
    target_row["content"] = _decode_content(_payload_str(target_row.get("content")))
    return {"rewound_count": len(rewound), "target_message": target_row, "new_head_id": new_head_id,
            **({"replacement_message_id": replacement_message_id,
                "replacement_message_uid": replacement_message_uid}
               if preserve_compaction_handoff else {})}


def _split_rewind_target(target_row: dict[str, Any], expected_target_content: Any,
                         preserve_compaction_handoff: bool) -> Optional[dict[str, Any]]:
    """Validate an active rewind target; return its handoff scaffold (or None). ``ValueError``:
    inactive / non-user-originated / missing composite carrier; ``RuntimeError``: live payload changed
    (the payload columns are payload-decoded first: the bytea payload rule)."""
    row = _decode_payloads(dict(target_row))
    if not row.get("active"):
        raise ValueError("rewind target is not active")
    content = _decode_content(_payload_str(row.get("content")))
    handoff, live_view = split_user_originated_turn({
        **row, "content": content,
        "display_metadata": _decode_display_metadata(row.get("display_metadata"))})
    if live_view is None:
        raise ValueError("rewind target is not a user-originated turn")
    live_content = live_view.get("content")
    if isinstance(live_content, str):
        live_content = sanitize_context(live_content).strip()
    if expected_target_content is not None and live_content != expected_target_content:
        raise RuntimeError("rewind target changed before it could be persisted")
    if preserve_compaction_handoff and handoff is None:
        raise ValueError("preserve_compaction_handoff requires an active composite carrier")
    return handoff if preserve_compaction_handoff else None


def retain_refused_turn_for_display(
    store, session_id: str, *, notice: str, boundary_row_id: Optional[int] = None,
) -> bool:
    """Take a just-safety-refused turn out of the model context, keeping the user's message on screen
    (the SQLite ``retain_refused_turn_for_display`` port): the whole turn is soft-deleted as ``/undo``
    does, and in the SAME write transaction the target row is re-surfaced into the display projection
    (``compacted = 1``) tagged ``content_refused`` with *notice* in its display metadata."""
    plan = _plan_refused_retention(store, session_id, boundary_row_id)
    if plan is None:
        return False
    target_row_id, preserve_handoff, expected_active_ids = plan

    def _do(conn) -> bool:
        rewind_in_transaction(
            store, conn, session_id, target_row_id,
            preserve_compaction_handoff=preserve_handoff, expected_active_ids=expected_active_ids)
        row = conn.execute(
            "SELECT display_metadata FROM messages WHERE id = %s AND session_id = %s",
            (target_row_id, session_id)).fetchone()
        if row is None:
            return False
        meta = _decode_display_metadata(row["display_metadata"]) or {}
        meta[CONTENT_REFUSED_METADATA_KEY] = {"notice": notice}
        from .common import _encode_display_metadata
        conn.execute(
            "UPDATE messages SET compacted = 1, display_kind = %s, display_metadata = %s "
            "WHERE id = %s AND session_id = %s",
            (CONTENT_REFUSED_DISPLAY_KIND, _encode_display_metadata(meta), target_row_id, session_id))
        return True

    return bool(_transcript_write(store, [], _do))


def _plan_refused_retention(store, session_id: str, boundary_row_id: Optional[int]):
    """``(target_row_id, preserve_compaction_handoff, expected_active_ids)`` for the rewind that retains
    the refused turn, or ``None`` when there is nothing to retain."""
    from .reads import get_active_message_ids
    if isinstance(boundary_row_id, int) and boundary_row_id > 0:
        row = store.one(
            "SELECT role, active, content, display_metadata FROM messages "
            "WHERE id = %s AND session_id = %s", (boundary_row_id, session_id))
        if row is not None and row["role"] == "user" and row["active"]:
            decoded = {"role": "user", "content": _decode_content(_payload_str(row["content"])),
                       "display_metadata": _decode_display_metadata(row["display_metadata"])}
            handoff, _live_view = split_user_originated_turn(decoded)
            return boundary_row_id, handoff is not None, list(get_active_message_ids(store, session_id))
    try:
        plan = plan_user_turn_rewind(store, session_id, -1)
    except RewindTargetUnavailableError:
        return None
    return plan.target_row_id, plan.scaffold is not None, plan.expected_active_ids


def rewind_user_turn(
    store, session_id: str, user_ordinal: int, *, warm_history: Optional[list[dict[str, Any]]] = None,
    require_retryable: bool = False, require_composite: bool = False, adopt_row_ids: bool = False,
) -> RewindOutcome:
    """Rewind the active transcript to just before user turn ``user_ordinal`` (0 = oldest; negative
    counts back from the newest, so ``-n`` is ``/undo n``) (the SQLite ``rewind_user_turn`` port).
    ``warm_history``: the in-memory view must have the same user turns and the same live target text
    as the durable transcript, else ``RuntimeError`` and nothing changes; its richer prefix is what
    gets installed. Out-of-range / wrong-shape targets raise
    :class:`RewindTargetUnavailableError`."""
    from session_rules.turns import _DB_PERSISTED_MARKER, flatten_message_text, message_uid_or_none, MESSAGE_UID
    plan = plan_user_turn_rewind(store, session_id, user_ordinal, warm_history=warm_history,
                                require_retryable=require_retryable, require_composite=require_composite)
    (prefix, durable_prefix, live_view, live_text, scaffold, durable_user,
     user_ordinal) = (plan.prefix, plan.durable_prefix, plan.live_view, plan.live_text, plan.scaffold,
                      plan.durable_user, plan.user_ordinal)
    try:
        result = rewind_to_message(store,
            session_id, plan.target_row_id, preserve_compaction_handoff=scaffold is not None,
            expected_active_ids=plan.expected_active_ids, expected_target_content=plan.expected_target_content)
    except ValueError as exc:  # target vanished / changed role under us: same class of failure as out-of-range
        raise RewindTargetUnavailableError(str(exc)) from exc
    if scaffold is not None:
        replacement_id = result.get("replacement_message_id")
        if not isinstance(replacement_id, int) or not durable_prefix:
            raise RuntimeError("rewind did not retain its compaction handoff")
        # The installed scaffold IS the replacement row: carry its identity, not just its row id.
        durable_prefix[-1].update({"_row_id": replacement_id, _DB_PERSISTED_MARKER: True})
        if replacement_uid := result.get("replacement_message_uid"):
            durable_prefix[-1][MESSAGE_UID] = replacement_uid
        prefix[-1] = durable_prefix[-1]
    if adopt_row_ids and prefix is not durable_prefix and len(prefix) == len(durable_prefix) and all(
        warm.get("role") == durable_message.get("role")
        and bool(warm.get("display_kind")) == bool(durable_message.get("display_kind"))
        and _comparison_content(warm) == _comparison_content(durable_message)
        for warm, durable_message in zip(prefix, durable_prefix)
    ):
        # Clients address follow-ups by durable row id: keep the richer warm content, adopt the identities.
        for warm, durable_message in zip(prefix, durable_prefix):
            if isinstance(row_id := durable_message.get("_row_id"), int):
                warm["_row_id"] = row_id
            if uid := message_uid_or_none(durable_message):
                warm[MESSAGE_UID] = uid
    return RewindOutcome(
        prefix=prefix, live_view=live_view,
        live_text=live_text if live_text is not None else flatten_message_text(live_view.get("content")),
        rewound_count=int(result.get("rewound_count", 0)), turns_undone=len(durable_user) - user_ordinal)


def plan_user_turn_rewind(
    store, session_id: str, user_ordinal: int, *, warm_history: Optional[list[dict[str, Any]]] = None,
    require_retryable: bool = False, require_composite: bool = False,
) -> RewindPlan:
    """Every read and check :func:`rewind_user_turn` makes before its write, with no write: the gateway
    ``/retry`` commits the planned rewind inside its re-send's admission transaction, so a refused
    re-send leaves it unapplied (the SQLite ``plan_user_turn_rewind`` port)."""
    from session_rules.turns import (
        history_before_user_originated_turn, retryable_user_text, split_user_originated_turn,
        user_originated_turn_view, _is_ephemeral_scaffolding,
    )
    from .reads import get_active_message_ids, get_messages_as_conversation

    expected_active_ids = list(get_active_message_ids(store, session_id))
    stored = get_messages_as_conversation(store, session_id, include_row_ids=True)
    # Live replay (the pre-request repair, a resume) merges a stored ``user;user`` pair into ONE turn
    # while both rows stay stored. Address turns on that same repaired projection.
    durable = get_messages_as_conversation(store, session_id, include_row_ids=True, repair_alternation=True)
    durable_user = _user_indices(durable)
    if user_ordinal < 0:
        user_ordinal = max(len(durable_user) + user_ordinal, 0)
    if user_ordinal >= len(durable_user):
        raise RewindTargetUnavailableError("target user message is no longer in session history")
    target_index = durable_user[user_ordinal]
    target = durable[target_index]
    durable_prefix, live_view = history_before_user_originated_turn(durable, target_index)
    scaffold, _ = split_user_originated_turn(target)
    if require_composite and scaffold is None:
        raise RewindTargetUnavailableError("target user message is not a compaction carrier")

    prefix = durable_prefix
    if warm_history is not None:
        warm = [m for m in warm_history if not _is_ephemeral_scaffolding(m)]
        warm_user = _user_indices(warm)
        if len(warm_user) != len(durable_user):
            raise RuntimeError(_HISTORY_CHANGED)
        prefix, warm_live_view = history_before_user_originated_turn(warm, warm_user[user_ordinal])
        if _comparison_content(live_view) != _comparison_content(warm_live_view):
            raise RuntimeError(_HISTORY_CHANGED)
    # Retry re-sends the stored bytes: ``""`.join`` of the text parts, never the "\n"-joined display
    # flattening (wire bytes == stored bytes).
    live_text = retryable_user_text(live_view.get("content")) if require_retryable else None
    target_row_id = target.get("_row_id")
    if not isinstance(target_row_id, int):
        raise RuntimeError("rewind target has no durable row identity")
    # The in-txn payload pin compares against the STORED row, which for a merged turn holds only the
    # first ask, never the merged text the live views carry.
    stored_view = next(
        (user_originated_turn_view(m) for m in stored if m.get("_row_id") == target_row_id), None)
    if stored_view is None:
        raise RuntimeError(_HISTORY_CHANGED)
    return RewindPlan(
        target_row_id=target_row_id, expected_active_ids=expected_active_ids,
        expected_target_content=stored_view.get("content"), scaffold=scaffold, prefix=prefix,
        durable_prefix=durable_prefix, live_view=live_view, live_text=live_text, durable_user=durable_user,
        user_ordinal=user_ordinal)

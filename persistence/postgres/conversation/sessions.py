"""``SessionRepository`` on PostgreSQL: the slice verbs ported from the SQLite reference (``create``,
``get``); every other sessions verb raises ``NotImplementedError`` naming itself (D8).

``create`` is the keep-existing constructor upsert: it never overwrites what an earlier writer set,
stamps a missing ``profile_name`` with the store's own, stores the system prompt content-addressed in
``system_prompts``, backfills a child's cwd/git/profile from its parent and, for compression forks,
the gateway routing columns too. ``get`` reads the row with the prompt-store joins and maps it through
the shared :func:`session_record` projection (degraded-instant rules included).
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from typing import Any, Iterator, Mapping, Optional, Sequence

from persistence import AuthorityRefusal, RetiredSessionError, SessionId
from persistence.authority.vocabulary import OwnerRule
from persistence.conversation.messages import QUEUED_PROMPT_METADATA_KEY
from persistence.conversation.sessions import (
    ONBOARDING_RUN_KEY, ArchiveChange, CanonicalTitleLocked, CronRunEntry, CronRunIdentity,
    DeleteManyOutcome, DeclaredScopeIdentity, LiveListingFacts, LiveListingRow, NewSession,
    OnboardingAdvance, PendingAdmissionRef, PickerFacts, PickerRoot, PickerSegment,
    RecentBrowseQuery, RecentBrowseTimeout, RecentSessionEntry, ReplaceSessionConfig,
    RosterSession, RosterSessions, RosterWorker, RuntimeLock, SessionCountQuery, SessionCwdUsage,
    SessionEdge, SessionExists, SessionFacetEntry, SessionListEntry, SessionListRow,
    SessionLifecycleStatus, SessionLineageKind, SessionOrder, SessionQuery,
    SessionRecencyEntry, SessionRepository, SessionRecord, SessionWindow,
    SkillScaffoldedSession, TitleConflict, TitleSource, bound_activity_description,
    is_automatic_end_reason, is_corrupt_model_config, normalize_activity_provenance,
    sanitize_title,
)
from persistence.core.canonical import canonical_json as _json
from persistence.core.conventions import UNTRUSTED_INSTANT, stored_instant
from persistence.sqlite._state.common import _BOUNDARY_END_REASONS, _shape_preview
from persistence.sqlite._state.errors import SessionActiveWriteGuardError
from persistence.sqlite._state.keys import (
    CRON_PRINCIPAL_ID, admission_fingerprint, cron_request_range, cron_run_from_request,
    legacy_cron_run_job, prefix_range,
)
from persistence.sqlite._schema import (
    TITLE_SOURCE_DERIVED, TITLE_SOURCE_LLM, TITLE_SOURCE_USER, _NUMBERED_TITLE_RE,
    _TITLE_SOURCE_RANK, _title_rank,
)
from persistence.sqlite._schema_ddl import _RESET_END_REASONS
from persistence.sqlite.conversation._store.compression import _CHAIN_CAP
from persistence.sqlite.conversation._store.core import CANONICAL_BOT_CHAT_TITLE, MAX_TITLE_LENGTH
from persistence.sqlite.conversation._store.local import (
    API_BINDING_PREFIX, API_DECLARED_PREFIX, POLICY_PREFIX, local_conversation_ids,
    local_lineage_owner, owned_lineage_ids, retire_local_receipts,
)
from persistence.sqlite.conversation._store.local_migration import LEGACY_PREFIX
from persistence.sqlite.conversation._store.model_config import session_ids_with_model_config_key
from persistence.sqlite.conversation._store.portability import (
    distinct_session_cwds, list_cron_job_runs, list_skill_scaffolded_sessions,
)
from persistence.sqlite.conversation._store.prompt_refresh import (
    PROFILE_KEY as _PROMPT_REFRESH_PROFILE_KEY,
    SESSION_KEY_PREFIX as _PROMPT_REFRESH_SESSION_PREFIX,
    count_sessions_with_stored_prompt,
)
from persistence.sqlite.conversation._store.retirement import (
    RETIRED_PREFIX, VIEWER_CLOSE_PREFIX, entry_session_id, retire_mutation_receipts,
    retire_prunable, retire_routes,
)
from persistence.sqlite.conversation._store.rules import (
    AUTO_LOAD_SCAFFOLD_SQL_LIKE, EPOCH_MAX, EPOCH_MIN, SKILL_SCAFFOLD_SQL_LIKE,
)
from persistence.sqlite.conversation._store.sessions import INTERNAL_LISTING_SOURCES
from persistence.sqlite.conversation._store.terminal import ADMISSION_PREFIX, WORKER_PREFIX, identity_key
from session_rules import onboarding_flow
from session_rules.activity import ActivityProvenance
from session_rules.legacy_local import same_owner
from session_rules.turns import (
    LEGACY_SUMMARY_PREFIX, SKILL_EXCERPT_JOINT, SUMMARY_PREFIX, _MERGED_PRIOR_CONTEXT_HEADER,
    _MERGED_SUMMARY_DELIMITER, _SUMMARY_END_MARKER, flatten_message_text,
    project_compaction_message_for_display,
)
from process_identity import holder_process_is_dead

from .._backend import PostgresStore, register_repository
from .._stubs import add_stubs

__all__ = ["PgSessionsRepository", "session_record"]

# The fork markers a child carries in ``model_config``, in the order the first non-blank one names its kind.
_MARKERS = (
    ("_delegate_from", SessionLineageKind.DELEGATE),
    ("_branched_from", SessionLineageKind.BRANCH),
    ("_reset_from", SessionLineageKind.RESET),
)
_MARKER_OF_KIND = {kind: key for key, kind in _MARKERS}
_TITLE_SOURCES = {source.value: source for source in TitleSource}
# The content hash a listing row's ``tool_names`` holds when the row was read without the prompt-store join.
_PROMPT_HASH = re.compile(r"[0-9a-f]{64}")


def _stripped(value: Any) -> Optional[str]:
    return value.strip() if isinstance(value, str) and value.strip() else None


def _json_value(text: Any) -> Any:
    """A stored JSON column decoded, ``None`` when unset or unreadable."""
    if not isinstance(text, str) or not text:
        return None
    try:
        return json.loads(text)
    except ValueError:
        return None


def _tool_pin(text: Any) -> Any:
    """The decoded ``tools[]`` pin; ``None`` for an unresolved content hash."""
    return None if isinstance(text, str) and _PROMPT_HASH.fullmatch(text) else _json_value(text)


def _parse_model_config(raw: Any) -> dict[str, Any]:
    """Tolerant ``model_config`` decode: JSON text or dict -> dict copy; anything else -> {}."""
    if isinstance(raw, str) and raw.strip():
        try:
            raw = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return {}
    return dict(raw) if isinstance(raw, dict) else {}


def session_record(row: Mapping[str, Any]) -> SessionRecord:
    """The record of one stored session row (the SQLite ``session_record`` projection, verbatim
    semantics: lineage markers lifted from ``model_config``, degraded stored instants)."""
    config = _parse_model_config(row.get("model_config"))
    kind, origin_id = SessionLineageKind.NONE, None
    for key, marker_kind in _MARKERS:
        if _stripped(config.get(key)):
            kind, origin_id = marker_kind, config[key]
            break
    origin = _json_value(row.get("origin_json"))
    started = stored_instant(row.get("started_at"), session_id=row["id"], field="started_at")
    ended = stored_instant(row.get("ended_at"), session_id=row["id"], field="ended_at")
    return SessionRecord(
        session_id=SessionId(row["id"]),
        source=row.get("source") or "",
        created_source=row.get("created_source"),
        user_id=row.get("user_id"),
        session_key=row.get("session_key"),
        chat_id=row.get("chat_id"),
        chat_type=row.get("chat_type"),
        thread_id=row.get("thread_id"),
        display_name=row.get("display_name"),
        origin=dict(origin) if isinstance(origin, Mapping) else None,
        expiry_finalized=bool(row.get("expiry_finalized")),
        model=row.get("model"),
        model_config=config,
        system_prompt=row.get("system_prompt"),
        system_prompt_hash=row.get("system_prompt_hash"),
        parent_session_id=row.get("parent_session_id"),
        lineage_kind=kind,
        reset_from=SessionId(_stripped(config.get("_reset_from"))) if _stripped(config.get("_reset_from")) else None,
        branched_from=SessionId(_stripped(config.get("_branched_from"))) if _stripped(config.get("_branched_from")) else None,
        started_at=(started or UNTRUSTED_INSTANT).timestamp(),
        ended_at=None if ended is None else ended.timestamp(),
        end_reason=row.get("end_reason"),
        message_count=row.get("message_count", 0),
        tool_call_count=row.get("tool_call_count", 0),
        input_tokens=row.get("input_tokens", 0),
        output_tokens=row.get("output_tokens", 0),
        cache_read_tokens=row.get("cache_read_tokens", 0),
        cache_write_tokens=row.get("cache_write_tokens", 0),
        reasoning_tokens=row.get("reasoning_tokens", 0),
        api_call_count=row.get("api_call_count", 0),
        estimated_cost_usd=row.get("estimated_cost_usd"),
        actual_cost_usd=row.get("actual_cost_usd"),
        billing_provider=row.get("billing_provider"),
        billing_base_url=row.get("billing_base_url"),
        billing_mode=row.get("billing_mode"),
        cost_status=row.get("cost_status"),
        cost_source=row.get("cost_source"),
        pricing_version=row.get("pricing_version"),
        cwd=row.get("cwd"),
        git_branch=row.get("git_branch"),
        git_repo_root=row.get("git_repo_root"),
        git_metadata_generation=int(row.get("git_metadata_generation") or 0),
        title=row.get("title"),
        title_source=_TITLE_SOURCES.get(row.get("title_source")),
        last_activity_at=row.get("last_activity_at"),
        last_activity_description=row.get("last_activity_description"),
        last_activity_provenance=row.get("last_activity_provenance"),
        handoff_state=row.get("handoff_state"),
        handoff_platform=row.get("handoff_platform"),
        handoff_error=row.get("handoff_error"),
        compression_failure_cooldown_until=row.get("compression_failure_cooldown_until"),
        compression_failure_error=row.get("compression_failure_error"),
        compression_fallback_streak=int(row.get("compression_fallback_streak") or 0),
        compression_ineffective_count=int(row.get("compression_ineffective_count") or 0),
        compression_recovery_deadline=row.get("compression_recovery_deadline"),
        compression_overload_streak=int(row.get("compression_overload_streak") or 0),
        profile_name=row.get("profile_name"),
        transport_profile=row.get("transport_profile"),
        rewind_count=int(row.get("rewind_count") or 0),
        archived=bool(row.get("archived")),
        auto_archived=bool(row.get("auto_archived")),
        pinned=bool(row.get("pinned")),
        hidden=bool(row.get("hidden")),
        last_read_at=row.get("last_read_at"),
        tool_pin=_tool_pin(row.get("tool_names")),
        transcript_revision=int(row.get("transcript_revision") or 0),
        runtime_revision=int(row.get("runtime_revision") or 0),
        runtime_generation=int(row.get("runtime_generation") or 0),
        lineage_origin_id=origin_id,
        model_config_corrupt=is_corrupt_model_config(row.get("model_config")),
        model_config_json=row.get("model_config"),
        origin_json=row.get("origin_json"),
        tool_names=row.get("tool_names"),
        prompt_cache_clock=row.get("prompt_cache_clock"),
    )


def _row_to_record(row: Mapping[str, Any]) -> Mapping[str, Any]:
    """The stored row with the prompt joins resolved and the fork markers lifted (the SQLite
    ``_session_row_dict`` projection)."""
    data = dict(row)
    for column in ("system_prompt", "tool_names"):
        if f"_{column}_resolved" in data:
            resolved = data.pop(f"_{column}_resolved")
            if column in data:
                data[column] = resolved
    if not (data.get("_reset_from") and data.get("_branched_from")):
        raw = data.get("model_config")
        cfg = _parse_model_config(raw)
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


# The SQLite upsert's keep-existing column set (the gateway creates a bare row before the real
# creator carries model/prompt): each COALESCEs the existing value over the incoming one.
_KEEP_EXISTING_COLUMNS = (
    "session_key", "chat_id", "chat_type", "thread_id", "parent_session_id", "cwd", "profile_name",
    "transport_profile", "git_repo_root", "origin_json", "display_name",
)


# ── dialect helpers: the SQLite store's SQL constructs, translated to the PG store's primitives ──


def _ph(values) -> str:
    """``%s,%s,...`` placeholders for *values* (a count or the values themselves)."""
    n = values if isinstance(values, int) else len(values)
    return ",".join("%s" for _ in range(n))


def _escape_like(text: str) -> str:
    """Escape LIKE wildcards (``%``, ``_``, ``\\``) so derived text matches literally (ESCAPE '\\\\')."""
    return text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _sql_literal(text: str) -> str:
    return "'" + text.replace("'", "''") + "'"


def _json_key(col: str, key: str) -> str:
    """The SQLite ``_sql_json_extract(col, '$.' + key)``: a non-throwing scalar lookup (text or NULL):
    the column's JSON is valid (the schema's ``hermes_json_valid``), else NULL, else the jsonb text form."""
    return f"(CASE WHEN {col} IS NULL OR NOT hermes_json_valid({col}) THEN NULL ELSE {col}::jsonb END ->> {key!r})"


def _legacy_reset_child_sql(alias: str, reasons: str) -> str:
    """Pre-marker reset-continuation heuristic: the child rides its parent's exact routing key and the
    parent ended at a reset boundary (shared by the list predicate and ``reopen``)."""
    return (f"EXISTS (SELECT 1 FROM sessions p WHERE p.id = {alias}.parent_session_id"
            f" AND p.end_reason IN ({reasons}) AND {alias}.session_key IS NOT NULL"
            f" AND {alias}.session_key <> '' AND {alias}.session_key = p.session_key)")


_RESET_END_REASONS_SQL = ", ".join(f"'{reason}'" for reason in _RESET_END_REASONS)
_RECOVERABLE_END_REASONS = ("agent_close", "ws_orphan_reap", "superseded_by_resume", "startup_orphan_reap")
_RECOVERABLE_END_REASONS_SQL = ", ".join(f"'{reason}'" for reason in _RECOVERABLE_END_REASONS)


def _branch_child_sql(alias: str) -> str:
    return (f"({_json_key(f'{alias}.model_config', '_branched_from')} IS NOT NULL"
            f" OR EXISTS (SELECT 1 FROM sessions p WHERE p.id = {alias}.parent_session_id"
            f" AND p.end_reason = 'branched' AND {alias}.started_at >= p.ended_at))")


def _reset_child_sql(alias: str) -> str:
    return (f"({_json_key(f'{alias}.model_config', '_reset_from')} IS NOT NULL"
            f" OR {_legacy_reset_child_sql(alias, _RESET_END_REASONS_SQL)})")


# Template predicates for a named alias (the SQLite ``_*_CHILD_SQL.format(a=...)`` shapes).
_BRANCH_CHILD_SQL = (
    f"({_json_key('{a}.model_config', '_branched_from')} IS NOT NULL"
    f" OR EXISTS (SELECT 1 FROM sessions p WHERE p.id = {{a}}.parent_session_id"
    f" AND p.end_reason = 'branched' AND {{a}}.started_at >= p.ended_at))"
)
_RESET_CHILD_SQL = (
    f"({_json_key('{a}.model_config', '_reset_from')} IS NOT NULL"
    f" OR {_legacy_reset_child_sql('{a}', _RESET_END_REASONS_SQL)}"
)
_COMPRESSION_CHILD_SQL = (
    "EXISTS (SELECT 1 FROM sessions p WHERE p.id = {a}.parent_session_id AND p.end_reason = 'compression')"
)


def _non_continuation_child_sql(child: str = "", parent: str = "parent.id") -> str:
    """``AND`` clauses rejecting children that are NOT compression continuations: branch/delegate/reset
    forks (markers bound to *parent*) and tool sessions (the SQLite ``_non_continuation_child_sql``)."""
    out = "".join(
        f" AND COALESCE({_json_key(f'{child}model_config', marker)}, '') <> {parent}"
        for marker in ("_branched_from", "_delegate_from", "_reset_from")
    )
    return out + f" AND COALESCE({child}source, '') <> 'tool'"


_LISTABLE_CHILD_SQL = (
    f"(s.parent_session_id IS NULL OR {_branch_child_sql('s')}"
    f" OR {_reset_child_sql('s')})"
)


def _where_sql(clauses: list, lead: str = "") -> str:
    """``WHERE a AND b`` (with *lead* prefix) or '' when there are no clauses."""
    return f"{lead}WHERE {' AND '.join(clauses)}" if clauses else ""


def _cwd_prefix_clause(cwd_prefix: str) -> tuple[str, list]:
    """``s.cwd`` at/under *cwd_prefix* (``_``/``%`` are LIKE wildcards but ordinary path characters)."""
    prefix = cwd_prefix.rstrip("/\\") or cwd_prefix
    esc = _escape_like(prefix)
    return (
        "(s.cwd = %s OR s.cwd LIKE %s ESCAPE '\\\\' OR s.cwd LIKE %s ESCAPE '\\\\')",
        [prefix, f"{esc}/%", f"{esc}\\\\%"],
    )


def _workspace_key_clause(key: str) -> tuple[str, list]:
    """WHERE for ``workspace_key(row) == key``: git_repo_root equals ``key``, or (rows predating
    per-session git metadata) cwd is at/under ``key``."""
    prefix = key.rstrip("/\\") or key
    cwd_clause, cwd_params = _cwd_prefix_clause(prefix)
    return (
        f"(s.git_repo_root = %s OR (COALESCE(s.git_repo_root, '') = '' AND {cwd_clause}))",
        [prefix, *cwd_params],
    )


def _source_match_clause(sources, local_created_sources=None) -> tuple[str, list]:
    """``sessions s`` source predicate; with ``local_created_sources`` a ``source='local'`` row matches
    only when ``created_source`` is one of those surfaces (the SQLite ``_source_match_clause``)."""
    labels = list(sources or [])
    if not labels:
        return "", []
    surfaces = list(local_created_sources or ())
    if not surfaces:
        return f"s.source IN ({_ph(labels)})", labels
    non_local = [item for item in labels if item != "local"]
    local_sql = f"(s.source = 'local' AND s.created_source IN ({_ph(surfaces)}))"
    if not non_local:
        return local_sql, list(surfaces)
    return (
        f"(s.source IN ({_ph(non_local)}) OR {local_sql})",
        [*non_local, *surfaces],
    )


def _session_filter_where(
    *, exclude_children: bool = False, source: Optional[str] = None, sources: Optional[list] = None,
    session_key: Optional[str] = None, exclude_sources: Optional[list] = None,
    cwd_prefix: Optional[str] = None, min_message_count: int = 0, archived_only: bool = False,
    include_archived: bool = False, include_subagents: bool = False,
    local_created_sources: Optional[Sequence[str]] = None, workspace_key: Optional[str] = None,
) -> tuple[list, list]:
    """The shared ``sessions s`` WHERE builder so counts line up with listed rows
    (the SQLite ``_session_filter_where``; clause order is part of the SQL text contract)."""
    where: list = []
    params: list = []
    if exclude_children and include_subagents:
        where.append(f"({_LISTABLE_CHILD_SQL} OR {_json_key('s.model_config', '_delegate_from')} IS NOT NULL)")
    elif exclude_children:
        where += [_LISTABLE_CHILD_SQL, f"{_json_key('s.model_config', '_delegate_from')} IS NULL"]
    include_sources = [source] if source else list(sources or [])
    source_clause, source_params = _source_match_clause(include_sources, local_created_sources)
    for clause, values in (
        (source_clause, source_params),
        ("s.session_key = %s", [session_key] if session_key else []),
        (f"s.source NOT IN ({_ph(len(exclude_sources or ()))})", exclude_sources or []),
        (_cwd_prefix_clause(cwd_prefix) if cwd_prefix else ("", [])),
        (_workspace_key_clause(workspace_key) if workspace_key else ("", [])),
        ("s.message_count >= %s", [min_message_count] if min_message_count > 0 else []),
    ):
        if values:
            where.append(clause)
            params.extend(values)
    if archived_only:
        where.append("s.archived = 1")
    elif not include_archived:
        where.append("s.archived = 0")
    return where, params


# The epoch-coercion window (a cell outside it is not a trusted instant; the same values the
# SQLite store pins): started/ended/activity/message timestamps read through it.
_EPOCH_WINDOW = f"BETWEEN {EPOCH_MIN!r} AND {EPOCH_MAX!r}"


def _in_window(expr: str) -> str:
    """*expr* when it is inside the ``coerce_epoch`` window, else NULL."""
    return f"(SELECT _w.v FROM (SELECT {expr} AS v) _w WHERE _w.v {_EPOCH_WINDOW})"


def _msg_max_ts(session_id_expr: str) -> str:
    return (f"(SELECT MAX(m.timestamp) FROM messages m WHERE m.session_id = {session_id_expr}"
            f" AND m.timestamp {_EPOCH_WINDOW})")


def _freshest_of(activity: str, session_id_expr: str, started: str) -> str:
    """Freshest of *activity* and the latest trusted message timestamp, else *started*; cells outside
    the epoch window are skipped (a session with no trusted cell at all is NULL)."""
    return (
        f"COALESCE((SELECT MAX(_v.v) FROM (SELECT {activity} AS v UNION ALL SELECT {_msg_max_ts(session_id_expr)}) _v"
        f" WHERE _v.v {_EPOCH_WINDOW}), {_in_window(started)})"
    )


def _last_active(alias: str = "s") -> str:
    """Session recency expression for a ``sessions {alias}`` row (the SQLite ``_sql_session_last_active``)."""
    return _freshest_of(f"{alias}.last_activity_at", f"{alias}.id", f"{alias}.started_at")


def _last_active_by_id(session_id_expr: str) -> str:
    """The same freshest-of expression keyed by a session-id SQL expression."""
    return _freshest_of(
        f"(SELECT last_activity_at FROM sessions _act_s WHERE _act_s.id = {session_id_expr})",
        session_id_expr,
        f"(SELECT started_at FROM sessions _act_s WHERE _act_s.id = {session_id_expr})",
    )


# ── the listing preview (first eligible user message, shaped in SQL; the SQLite preview, translated) ──

_PREVIEW_HEAD_CHARS = 63
_PREVIEW_SCAFFOLD_WINDOW = 400
# The SQLite CHAR(9) || CHAR(10) || CHAR(13) || CHAR(32) trim set, as a regexp class.
_WS = r"[ \t\n\r]"
_SUMMARY_PREFIXES = (SUMMARY_PREFIX, LEGACY_SUMMARY_PREFIX)


def _ltrim_ws(expr: str) -> str:
    return f"regexp_replace({expr}, '^{_WS}+', '')"


def _trim_ws(expr: str) -> str:
    return f"regexp_replace(regexp_replace({expr}, '^{_WS}+', ''), '{_WS}+$', '')"


def _starts_with(expr: str, prefixes: tuple) -> str:
    trimmed = _ltrim_ws(expr)
    return "(" + " OR ".join(
        f"substring({trimmed}, 1, {len(p)}) = {_sql_literal(p)}" for p in prefixes) + ")"


def _after_marker(expr: str, marker: str) -> str:
    """*expr* after the first occurrence of *marker* (``strpos`` mirrors ``INSTR``)."""
    return f"substring({expr}, strpos({expr}, {_sql_literal(marker)}) + {len(marker)})"


_CONTENT_SQL = "replace(replace(m.content, chr(10), ' '), chr(13), ' ')"
_SCAFFOLDED_SQL = (
    f"(m.content LIKE {_sql_literal(SKILL_SCAFFOLD_SQL_LIKE)}"
    f" OR m.content LIKE {_sql_literal(AUTO_LOAD_SCAFFOLD_SQL_LIKE)})")
_STANDALONE_SUMMARY_SQL = _starts_with("m.content", _SUMMARY_PREFIXES)
_MERGED_AFTER_SQL = _after_marker("m.content", _MERGED_SUMMARY_DELIMITER)
_MERGED_SUMMARY_SQL = (
    f"(strpos(m.content, {_sql_literal(_MERGED_SUMMARY_DELIMITER)}) > 0 AND "
    f"{_starts_with(_MERGED_AFTER_SQL, _SUMMARY_PREFIXES)})")
_MERGED_PRIOR_SQL = _trim_ws(
    f"substring(m.content, 1, strpos(m.content, {_sql_literal(_MERGED_SUMMARY_DELIMITER)}) - 1)")
_MERGED_PRIOR_LTRIMMED_SQL = _ltrim_ws(_MERGED_PRIOR_SQL)
_MERGED_PRIOR_UNWRAPPED_SQL = (
    f"CASE WHEN substring({_MERGED_PRIOR_LTRIMMED_SQL}, 1, {len(_MERGED_PRIOR_CONTEXT_HEADER)})"
    f" = {_sql_literal(_MERGED_PRIOR_CONTEXT_HEADER)}"
    f" THEN {_ltrim_ws(f'substring({_MERGED_PRIOR_LTRIMMED_SQL}, {len(_MERGED_PRIOR_CONTEXT_HEADER) + 1})')}"
    f" ELSE {_MERGED_PRIOR_SQL} END")
_FORCE_USER_REMAINDER_SQL = _after_marker("m.content", _SUMMARY_END_MARKER)
_PREVIEW_ELIGIBLE_SQL = (
    f"(COALESCE(m.display_kind, '') <> 'hidden'"
    f" AND ((NOT ({_STANDALONE_SUMMARY_SQL}) AND NOT ({_MERGED_SUMMARY_SQL}))"
    f" OR ({_STANDALONE_SUMMARY_SQL} AND strpos(m.content, {_sql_literal(_SUMMARY_END_MARKER)}) > 0"
    f" AND char_length({_trim_ws(_FORCE_USER_REMAINDER_SQL)}) > 0)"
    f" OR ({_MERGED_SUMMARY_SQL}"
    f" AND char_length({_trim_ws(_MERGED_PRIOR_UNWRAPPED_SQL)}) > 0)))")


def _preview_raw_select() -> str:
    """The preview SELECT over ``m`` (scaffolded rows: head + tail around the skill excerpt join)."""
    return (
        f"CASE WHEN {_STANDALONE_SUMMARY_SQL} THEN {_FORCE_USER_REMAINDER_SQL}"
        f" WHEN {_MERGED_SUMMARY_SQL} THEN {_MERGED_PRIOR_UNWRAPPED_SQL}"
        f" WHEN {_SCAFFOLDED_SQL} AND char_length(m.content) > {_PREVIEW_SCAFFOLD_WINDOW * 2}"
        f" THEN substring({_CONTENT_SQL}, 1, {_PREVIEW_SCAFFOLD_WINDOW})"
        f" || {_sql_literal(SKILL_EXCERPT_JOINT)}"
        f" || right({_CONTENT_SQL}, {_PREVIEW_SCAFFOLD_WINDOW})"
        f" WHEN {_SCAFFOLDED_SQL} THEN substring({_CONTENT_SQL}, 1, {_PREVIEW_SCAFFOLD_WINDOW * 2})"
        f" ELSE substring({_ltrim_ws(_CONTENT_SQL)}, 1, {_PREVIEW_HEAD_CHARS}) END")


def _preview_raw(session_id_expr: str = "s.id") -> str:
    """The correlated ``_preview_raw`` column: the first eligible user message of *session_id_expr*
    (the SQLite ``_sql_preview_raw``, without the INDEXED BY hint PG has no analogue for)."""
    return (
        f"COALESCE((SELECT {_preview_raw_select()} FROM messages m"
        f" WHERE m.session_id = {session_id_expr} AND m.role = 'user' AND m.content IS NOT NULL"
        f" AND {_PREVIEW_ELIGIBLE_SQL} ORDER BY m.timestamp, m.id LIMIT 1), '') AS _preview_raw")


# A local chat: minted by this owner (``local-*``), or a pre-upgrade chat it adopted under a legacy binding.
_LOCAL_CHAT = ("(chat_id LIKE 'local-%' OR EXISTS "
               "(SELECT 1 FROM state_meta WHERE key = %s || sessions.chat_id))")

# The descendants walk: continuation edges only (same predicate as the session list's chain CTE): a
# subagent run, a /branch fork, a /new reset child or a tool-owned row is its own conversation.
_DESCENDANTS_SQL = f"""
            WITH RECURSIVE descendants(id, parent_session_id, started_at) AS (
                SELECT id, parent_session_id, started_at FROM sessions WHERE id = %s
                UNION
                SELECT s.id, s.parent_session_id, s.started_at
                FROM sessions s
                JOIN descendants d ON s.parent_session_id = d.id
                WHERE {_json_key('s.model_config', '_delegate_from')} IS NULL
                  AND {_json_key('s.model_config', '_branched_from')} IS NULL
                  AND NOT ({_RESET_CHILD_SQL.format(a='s')})
                  AND COALESCE(s.source, '') <> 'tool'
            )
            SELECT id, parent_session_id, started_at FROM descendants
            """

# The lineage CTE (set-archived / pinned / hidden / read across a whole compression lineage).
_LINEAGE_CTE_SQL = f"""
            WITH RECURSIVE
              ancestors(id) AS (
                SELECT %s
                UNION
                SELECT parent.id
                FROM ancestors a
                JOIN sessions child ON child.id = a.id
                JOIN sessions parent ON parent.id = child.parent_session_id
                WHERE parent.end_reason = 'compression'
{_non_continuation_child_sql('child.', 'parent.id')}
              ),
              descendants(id) AS (
                SELECT %s
                UNION
                SELECT child.id
                FROM descendants d
                JOIN sessions parent ON parent.id = d.id
                JOIN sessions child ON child.parent_session_id = parent.id
                WHERE parent.end_reason = 'compression'
{_non_continuation_child_sql('child.', 'parent.id')}
              ),
              lineage(id) AS (
                SELECT id FROM ancestors
                UNION
                SELECT id FROM descendants
              )"""

_EMPTY_SESSION_WHERE = (
    "message_count = 0 AND ended_at IS NOT NULL AND archived = 0 AND NOT EXISTS ("
    "SELECT 1 FROM messages WHERE messages.session_id = sessions.id)")

# One compression-continuation step (the SQLite ``_CHAIN_STEP_SQL``; PG forbids LIMIT inside a
# recursive CTE, so the walk is bounded in Python with the caller's cap).
_CHAIN_STEP_SQL = f"""
                    SELECT child.id
                    FROM sessions parent
                    JOIN sessions child ON child.parent_session_id = parent.id
                    WHERE parent.id = %s
                      AND parent.end_reason = 'compression'
                      AND {_json_key('child.model_config', '_branched_from')} IS NULL
                      AND {_json_key('child.model_config', '_delegate_from')} IS NULL
                      AND NOT ({_RESET_CHILD_SQL.format(a='child')})
                      AND COALESCE(child.source, '') <> 'tool'
                    ORDER BY
                      CASE
                        WHEN child.end_reason = 'compression' THEN 0
                        WHEN child.ended_at IS NULL THEN 1
                        ELSE 2
                      END,
                      {_last_active('child')} DESC,
                      child.started_at DESC,
                      child.id DESC
                    LIMIT 1"""


def _session_row_dict(row: Mapping[str, Any]) -> dict[str, Any]:
    """A stored row dict with the resolved prompt/tool columns folded in and the /new vs /branch
    markers lifted out of model_config (the SQLite ``_session_row_dict``, over the PG row shape)."""
    data = dict(row)
    for column in ("system_prompt", "tool_names"):
        if f"_{column}_resolved" in data:
            resolved = data.pop(f"_{column}_resolved")
            if column in data:
                data[column] = resolved
    if not (data.get("_reset_from") and data.get("_branched_from")):
        cfg = _parse_model_config(data.get("model_config"))
        if not data.get("_reset_from"):
            value = cfg.get("_reset_from")
            if isinstance(value, str) and value.strip():
                data["_reset_from"] = value.strip()
        if not data.get("_branched_from"):
            value = cfg.get("_branched_from")
            if isinstance(value, str) and value.strip():
                data["_branched_from"] = value.strip()
    return data


def _session_unread(session_row: Mapping[str, Any]) -> bool:
    """Unread = activity postdates the ``last_read_at`` watermark (NULL = read)."""
    last_read = session_row.get("last_read_at")
    if last_read is None:
        return False
    last_active = session_row.get("last_active") or session_row.get("started_at")
    return float(last_active or 0) > float(last_read)


def _ids(values: Sequence[str]) -> tuple[SessionId, ...]:
    return tuple(SessionId(value) for value in values)


def _removed(values: Sequence[str]) -> tuple[SessionId, ...]:
    """Removed or archived ids, each once (a write retried on a busy store reports its ids again)."""
    return _ids(tuple(dict.fromkeys(values)))


def _listed(values: Optional[Sequence[str]]) -> Optional[list]:
    """A filter sequence as the store's list argument; empty or absent is no filter."""
    return list(values) if values else None


def _optional_list(values: Optional[Sequence[str]]) -> Optional[list]:
    return list(values) if values is not None else None


def _sql_limit(window) -> int:
    """The window's ``LIMIT`` operand (the SQLite ``-1`` no-limit is a plain PG LIMIT of all rows)."""
    return 10**9 if window.limit is None else window.limit


def _last_active_value(row: Mapping[str, Any]) -> Optional[float]:
    value = row.get("last_active")
    return None if value is None else float(value)


def _list_entry(row: Mapping[str, Any], *, compact: bool = False) -> SessionListEntry:
    """One listing entry of a rich row; a row the store did not mark ``unread`` takes the store's unread rule."""
    session = session_record(_row_to_record(row))
    return SessionListEntry(
        session=session,
        preview=row.get("preview") or "",
        unread=bool(row["unread"]) if "unread" in row else _session_unread(dict(row)),
        last_active=_last_active_value(row),
        lineage_root_id=row.get("_lineage_root_id"),
        lineage_ids=_ids(row.get("_lineage_ids") or ()),
        continuation_kind=row.get("continuation_kind"),
        compact=compact,
    )


def _recency_entry(row: Mapping[str, Any]) -> SessionRecencyEntry:
    return SessionRecencyEntry(session=session_record(_row_to_record(row)), last_active=_last_active_value(row))


def _row_to_record_compat(row: Mapping[str, Any]) -> Mapping[str, Any]:
    return _row_to_record(row)


class PgSessionsRepository:
    """``SessionRepository`` verbs on the bound profile's store schema; unported verbs raise (D8)."""

    def __init__(self, store: PostgresStore) -> None:
        self._store = store

    # ── the slice verbs ──

    def create(self, spec: Any) -> SessionId:
        """Create (upsert) a session record, keeping what an earlier writer set; returns the session id."""
        store = self._store
        sid = spec.session_id
        config = dict(spec.model_config or {})
        if marker := _MARKER_OF_KIND.get(spec.lineage_kind):
            config[marker] = spec.parent_session_id
        with store.tx() as conn:
            # A delayed constructor must not recreate a deleted session beside its tombstone.
            if store.one("SELECT 1 FROM sessions WHERE id = %s", sid) is None and store.one(
                    "SELECT 1 FROM state_meta WHERE key = %s", RETIRED_PREFIX + sid) is not None:
                raise RetiredSessionError(sid, "not_found")
            profile_name = spec.profile_name
            if not (profile_name or "").strip():
                profile_name = store.owner_profile_name
            # A newly inserted /new continuation is visible immediately: a delegate marker riding the
            # incoming config would hide the reset chat from every list (#109073).
            if (spec.session_key and isinstance(config, dict)
                    and config.get("_reset_from") is not None
                    and config.get("_delegate_from") is not None):
                config = {key: value for key, value in config.items() if key != "_delegate_from"}
            model_config_json = json.dumps(config) if config else None
            prompt_hash = self._store_system_prompt(conn, spec.system_prompt)
            existing = store.one("SELECT * FROM sessions WHERE id = %s", sid)
            if existing is None:
                conn.execute(
                    """INSERT INTO sessions (
                       id, source, created_source, user_id, session_key, chat_id, chat_type, thread_id,
                       model, model_config, system_prompt, system_prompt_hash,
                       parent_session_id, cwd, profile_name, transport_profile, git_repo_root,
                       origin_json, display_name, started_at
                   ) VALUES (
                       %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                   )""",
                    (
                        sid, spec.source, spec.source, spec.user_id, spec.session_key, spec.chat_id,
                        spec.chat_type, spec.thread_id, spec.model, model_config_json, spec.system_prompt,
                        prompt_hash, spec.parent_session_id, spec.cwd, profile_name, spec.transport_profile,
                        spec.git_repo_root, json.dumps(dict(spec.origin)) if spec.origin else None,
                        spec.display_name, time.time(),
                    ),
                )
            else:
                # Keep-existing merge (the SQLite upsert's CASEs, computed in Python over the stored row).
                source = spec.source if existing["source"] in (None, "unknown") else existing["source"]
                model = existing["model"] if existing["model"] is not None else spec.model
                if model_config_json is not None:
                    stored_cfg = _parse_model_config(existing["model_config"])
                    reset_from = stored_cfg.get("_reset_from")
                    remainder = {k: v for k, v in stored_cfg.items() if k != "_reset_from"}
                    if reset_from is not None and remainder == {}:
                        merged = json.loads(model_config_json)
                        merged.pop("_delegate_from", None)
                        merged["_reset_from"] = reset_from
                        model_config = json.dumps(merged)
                    else:
                        model_config = existing["model_config"] or model_config_json
                else:
                    model_config = existing["model_config"]
                sp_hash = existing["system_prompt_hash"] if existing["system_prompt_hash"] is not None else prompt_hash
                sp_text = (
                    None
                    if existing["system_prompt_hash"] is None and prompt_hash is not None
                    else existing["system_prompt"]
                )
                merged = {
                    col: (existing[col] if existing[col] is not None else value)
                    for col, value in {
                        "session_key": spec.session_key, "chat_id": spec.chat_id, "chat_type": spec.chat_type,
                        "thread_id": spec.thread_id, "parent_session_id": spec.parent_session_id,
                        "cwd": spec.cwd, "profile_name": profile_name,
                        "transport_profile": spec.transport_profile, "git_repo_root": spec.git_repo_root,
                        "origin_json": json.dumps(dict(spec.origin)) if spec.origin else None,
                        "display_name": spec.display_name,
                    }.items()
                }
                # created_source: a later creator replaces the accounting placeholders (and 'local'
                # never replaces anything); a stamped real source is preserved (#56439).
                unset = (existing["created_source"] or "") in ("", "unknown", "local")
                created_source = existing["created_source"]
                if unset and spec.source and spec.source != "local":
                    created_source = spec.source
                conn.execute(
                    """UPDATE sessions SET
                       source = %s, model = %s, model_config = %s, system_prompt = %s,
                       system_prompt_hash = %s, created_source = %s,
                       session_key = %s, chat_id = %s, chat_type = %s, thread_id = %s,
                       parent_session_id = %s, cwd = %s, profile_name = %s,
                       transport_profile = %s, git_repo_root = %s, origin_json = %s, display_name = %s
                     WHERE id = %s""",
                    (
                        source, model, model_config, sp_text, sp_hash, created_source,
                        merged["session_key"], merged["chat_id"], merged["chat_type"], merged["thread_id"],
                        merged["parent_session_id"], merged["cwd"], merged["profile_name"],
                        merged["transport_profile"], merged["git_repo_root"], merged["origin_json"],
                        merged["display_name"], sid,
                    ),
                )
            if prompt_hash is not None:
                self._delete_unreferenced_system_prompts(conn)
            if spec.parent_session_id:
                self._inherit_parent_session_metadata(conn, sid)
            self._stamp_compression_ancestors(conn, sid)
        return SessionId(sid)

    def get(self, session_id: SessionId) -> Optional[SessionRecord]:
        """The session's record, or ``None``: the row with the prompt-store joins, through the shared
        record projection."""
        store = self._store
        row = store.one(
            """SELECT s.*, COALESCE(sp.prompt, s.system_prompt) AS _system_prompt_resolved,
                      COALESCE(tp.prompt, s.tool_names) AS _tool_names_resolved
               FROM sessions s
               LEFT JOIN system_prompts sp ON sp.hash = s.system_prompt_hash
               LEFT JOIN system_prompts tp ON tp.hash = s.tool_names
               WHERE s.id = %s""",
            session_id,
        )
        return session_record(_row_to_record(row)) if row else None

    # ── create's helpers (the SQLite store functions, ported) ──

    def _store_system_prompt(self, conn, system_prompt: Optional[str]) -> Optional[str]:
        if system_prompt is None:
            return None
        prompt_hash = hashlib.sha256(system_prompt.encode("utf-8")).hexdigest()
        conn.execute(
            "INSERT INTO system_prompts (hash, prompt) VALUES (%s, %s) ON CONFLICT (hash) DO NOTHING",
            (prompt_hash, system_prompt),
        )
        return prompt_hash

    def _delete_unreferenced_system_prompts(self, conn) -> None:
        conn.execute(
            """DELETE FROM system_prompts
               WHERE NOT EXISTS (SELECT 1 FROM sessions WHERE sessions.system_prompt_hash = system_prompts.hash)
                 AND NOT EXISTS (SELECT 1 FROM sessions WHERE sessions.tool_names = system_prompts.hash)"""
        )

    def _inherit_parent_session_metadata(self, conn, session_id: str) -> None:
        """Backfill a child's NULL cwd/git/profile from its parent; for compression forks (excluding
        fork edges), the gateway routing columns too (#64709, #59527)."""
        for col in ("cwd", "git_repo_root", "git_branch"):
            conn.execute(
                f"""UPDATE sessions SET {col} = COALESCE(sessions.{col},
                        (SELECT p.{col} FROM sessions p
                          WHERE p.id = sessions.parent_session_id))
                    WHERE id = %s AND parent_session_id IS NOT NULL""",
                (session_id,),
            )
        # Same-namespace profile backfill (the SQLite same-key check, expressed without the SQLite
        # instr() helper: position() mirrors it, 0 when the namespace separator is absent).
        conn.execute(
            """UPDATE sessions SET profile_name = COALESCE(sessions.profile_name,
                        (SELECT p.profile_name FROM sessions p
                          WHERE p.id = sessions.parent_session_id
                            AND (p.session_key IS NULL OR sessions.session_key IS NULL
                                 OR substr(p.session_key, 1, position(':' in substr(p.session_key, 7)) + 6)
                                      = substr(sessions.session_key, 1, position(':' in substr(sessions.session_key, 7)) + 6))))
               WHERE id = %s AND parent_session_id IS NOT NULL""",
            (session_id,),
        )
        # Gateway routing inheritance, compression forks only, skipping fork edges (a child whose
        # marker names its parent is its own conversation, not a crash-recovery continuation).
        child = conn.execute(
            "SELECT model_config, parent_session_id FROM sessions WHERE id = %s", (session_id,)
        ).fetchone()
        if child is None or child["parent_session_id"] is None:
            return
        cfg = _parse_model_config(child["model_config"])
        markers = (cfg.get("_branched_from"), cfg.get("_delegate_from"), cfg.get("_reset_from"))
        parent_id = child["parent_session_id"]
        if any(str(marker or "") == str(parent_id) for marker in markers):
            return
        parent = conn.execute(
            "SELECT end_reason FROM sessions WHERE id = %s", (parent_id,)
        ).fetchone()
        if parent is None or parent["end_reason"] != "compression":
            return
        for col in ("user_id", "session_key", "chat_id", "chat_type", "thread_id",
                    "display_name", "origin_json", "transport_profile"):
            conn.execute(
                f"""UPDATE sessions SET {col} = (SELECT p.{col} FROM sessions p
                                                 WHERE p.id = %s)
                    WHERE id = %s AND sessions.{col} IS NULL""",
                (parent_id, session_id),
            )

    def _stamp_compression_ancestors(self, conn, session_id: str) -> None:
        """Copy the tip's ``created_source`` onto compression ancestors that have none (the SQLite
        ``stamp_compression_ancestors``, same chain walk)."""
        row = conn.execute(
            "SELECT parent_session_id, created_source FROM sessions WHERE id = %s", (session_id,)
        ).fetchone()
        if row is None:
            return
        surface = row["created_source"]
        if surface in (None, "", "unknown", "local"):
            return
        current, parent_id = session_id, row["parent_session_id"]
        for _ in range(1000):
            if not parent_id:
                return
            parent = conn.execute(
                "SELECT parent_session_id, created_source FROM sessions WHERE id = %s", (parent_id,)
            ).fetchone()
            if parent is None or parent["created_source"] not in (None, "", "unknown", "local", surface):
                return
            # The chain step: the parent's preferred continuation must be ``current``.
            step = conn.execute(
                """SELECT child.id FROM sessions parent
                   JOIN sessions child ON child.parent_session_id = parent.id
                   WHERE parent.id = %s AND parent.end_reason = 'compression'
                     AND COALESCE(child.source, '') != 'tool'
                   ORDER BY child.id LIMIT 1""",
                (parent_id,),
            ).fetchone()
            if step is None or step["id"] != current:
                return
            if parent["created_source"] != surface:
                conn.execute("UPDATE sessions SET created_source = %s WHERE id = %s", (surface, parent_id))
            current, parent_id = parent_id, parent["parent_session_id"]


add_stubs(PgSessionsRepository, SessionRepository, implemented=("create", "get"))


def _register() -> None:
    register_repository("sessions", lambda store: PgSessionsRepository(store))


_register()

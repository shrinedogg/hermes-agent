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
from contextlib import contextmanager
from typing import Any, Callable, Iterator, Mapping, Optional, Sequence

from persistence import (
    AuthorityRefusal, CompressionClosed, CompressionInProgress, RetiredSessionError,
    SessionId, SessionWriteGuarded, TurnLeaseLost, UnknownExecution,
)
from persistence.authority.vocabulary import OwnerRule
from persistence.conversation.messages import QUEUED_PROMPT_METADATA_KEY
from persistence.conversation.sessions import (
    END_REASON_SESSION_RESET, ONBOARDING_RUN_KEY, ArchiveChange, CanonicalTitleLocked, CronRunEntry,
    CronRunIdentity,
    DeleteManyOutcome, DeclaredScopeIdentity, LiveListingFacts, LiveListingRow, NewSession,
    OnboardingAdvance, PendingAdmissionRef, PickerFacts, PickerRoot, PickerSegment,
    RecentBrowseQuery, RecentBrowseTimeout, RecentSessionEntry, ReplaceSessionConfig,
    RosterSession, RosterSessions, RosterWorker, RuntimeLock, SessionCountQuery, SessionCwdUsage,
    SessionEdge, SessionExists, SessionFacetEntry, SessionListEntry, SessionListRow,
    SessionLifecycleStatus, SessionLineageKind, SessionOrder, SessionQuery,
    SessionRecencyEntry, SessionRepository, SessionRecord, SessionWindow,
    SkillScaffoldedSession, TitleConflict, TitleSource, bound_activity_description,
    is_automatic_end_reason, is_corrupt_model_config,
    sanitize_title,
)
from persistence.core.canonical import canonical_json as _json
from persistence.core.conventions import UNTRUSTED_INSTANT, stored_instant
from persistence.sqlite._state.common import _BOUNDARY_END_REASONS, _shape_preview
from persistence.sqlite._state.errors import (
    CompressionSessionClosedError, SessionActiveWriteGuardError,
    SessionCompressionInProgressError, SessionTurnLeaseLostError,
)
from persistence.sqlite._state.keys import (
    CRON_PRINCIPAL_ID, CRON_REQUEST_PREFIX, admission_fingerprint, cron_request_range,
    cron_run_from_request, legacy_cron_run_job, prefix_range,
)
from persistence.sqlite._schema import (
    TITLE_SOURCE_DERIVED, TITLE_SOURCE_LLM, TITLE_SOURCE_USER, _NUMBERED_TITLE_RE,
    _TITLE_SOURCE_RANK, _title_rank,
)
from persistence.sqlite._schema_ddl import _RESET_END_REASONS
from persistence.sqlite.conversation._store.compression import _CHAIN_CAP
from persistence.sqlite.conversation._store.core import CANONICAL_BOT_CHAT_TITLE, MAX_TITLE_LENGTH
from persistence.sqlite.conversation._store.local import (
    API_BINDING_PREFIX, API_DECLARED_PREFIX, POLICY_PREFIX,
)
from persistence.sqlite.conversation._store.local_migration import LEGACY_PREFIX
from persistence.sqlite.conversation._store.prompt_refresh import (
    PROFILE_KEY as _PROMPT_REFRESH_PROFILE_KEY,
    SESSION_KEY_PREFIX as _PROMPT_REFRESH_SESSION_PREFIX,
)
from persistence.sqlite.conversation._store.retirement import (
    RETIRED_PREFIX, VIEWER_CLOSE_PREFIX, _TRANSCRIPT_FIELDS, entry_session_id,
)
from persistence.sqlite.conversation._store.rules import (
    AUTO_LOAD_SCAFFOLD_SQL_LIKE, EPOCH_MAX, EPOCH_MIN, SKILL_SCAFFOLD_SQL_LIKE,
)
from persistence.sqlite.conversation._store.sessions import INTERNAL_LISTING_SOURCES, classify_session_status
from persistence.sqlite.conversation._store.terminal import (
    ADMISSION_PREFIX, RESULT_PREFIX, WORKER_PREFIX, encode_result, identity_key, load_result,
)
from session_rules import onboarding_flow
from session_rules.activity import ActivityProvenance, normalize_activity_provenance
from session_rules.legacy_local import same_owner
from session_rules.turns import (
    LEGACY_SUMMARY_PREFIX, MODEL_ONLY_DISPLAY_METADATA_KEY, SKILL_EXCERPT_JOINT, SUMMARY_PREFIX,
    _MERGED_PRIOR_CONTEXT_HEADER, _MERGED_SUMMARY_DELIMITER, _SUMMARY_END_MARKER,
    flatten_message_text, project_compaction_message_for_display,
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
    """Escape LIKE wildcards (``%``, ``_``, ``\\``) so derived text matches literally (ESCAPE '\\')."""
    return text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _sql_literal(text: str) -> str:
    return "'" + text.replace("'", "''") + "'"


def _json_key_text(col: str, key: str) -> str:
    """Tolerant jsonb text lookup (the SQLite ``_sql_json_extract``; NULL or invalid JSON → NULL)."""
    return _json_key(col, key)


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
    f" OR {_legacy_reset_child_sql('{a}', _RESET_END_REASONS_SQL)})"
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
        "(s.cwd = %s OR s.cwd LIKE %s ESCAPE '\\' OR s.cwd LIKE %s ESCAPE '\\')",
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


def _content_text() -> str:
    """``m.content`` as NUL-safe text: the BYTEA payload converted to UTF-8 with NUL bytes shown as
    spaces (PostgreSQL ``text`` cannot hold a NUL; the SQLite reference stores it in a column that can,
    and its preview SQL replaces NUL bytes in its trim/replace steps the same way)."""
    return "hermes_nul_safe_text(m.content)"


_MTXT = _content_text()
_CONTENT_SQL = "replace(replace(" + _MTXT + ", chr(10), ' '), chr(13), ' ')"
# LIKE prefix patterns with the wildcard appended at runtime (``|| CHR(37)``): a raw ``%`` in the SQL
# text would trip psycopg's client-side placeholder scan, and a ``%%`` escape only survives when the
# query is executed with parameters (it would remain a literal ``%%`` on a bare ``execute``).
assert SKILL_SCAFFOLD_SQL_LIKE.endswith("%") and AUTO_LOAD_SCAFFOLD_SQL_LIKE.endswith("%")
_SCAFFOLDED_SQL = (
    f"({_MTXT} LIKE {_sql_literal(SKILL_SCAFFOLD_SQL_LIKE[:-1])} || CHR(37)"
    f" OR {_MTXT} LIKE {_sql_literal(AUTO_LOAD_SCAFFOLD_SQL_LIKE[:-1])} || CHR(37))")
_STANDALONE_SUMMARY_SQL = _starts_with(_MTXT, _SUMMARY_PREFIXES)
_MERGED_AFTER_SQL = _after_marker(_MTXT, _MERGED_SUMMARY_DELIMITER)
_MERGED_SUMMARY_SQL = (
    f"(strpos({_MTXT}, {_sql_literal(_MERGED_SUMMARY_DELIMITER)}) > 0 AND "
    f"{_starts_with(_MERGED_AFTER_SQL, _SUMMARY_PREFIXES)})")
_MERGED_PRIOR_SQL = _trim_ws(
    f"substring({_MTXT}, 1, strpos({_MTXT}, {_sql_literal(_MERGED_SUMMARY_DELIMITER)}) - 1)")
_MERGED_PRIOR_LTRIMMED_SQL = _ltrim_ws(_MERGED_PRIOR_SQL)
_MERGED_PRIOR_UNWRAPPED_SQL = (
    f"CASE WHEN substring({_MERGED_PRIOR_LTRIMMED_SQL}, 1, {len(_MERGED_PRIOR_CONTEXT_HEADER)})"
    f" = {_sql_literal(_MERGED_PRIOR_CONTEXT_HEADER)}"
    f" THEN {_ltrim_ws(f'substring({_MERGED_PRIOR_LTRIMMED_SQL}, {len(_MERGED_PRIOR_CONTEXT_HEADER) + 1})')}"
    f" ELSE {_MERGED_PRIOR_SQL} END")
_FORCE_USER_REMAINDER_SQL = _after_marker(_MTXT, _SUMMARY_END_MARKER)
_PREVIEW_ELIGIBLE_SQL = (
    f"(COALESCE(m.display_kind, '') <> 'hidden'"
    f" AND ((NOT ({_STANDALONE_SUMMARY_SQL}) AND NOT ({_MERGED_SUMMARY_SQL}))"
    f" OR ({_STANDALONE_SUMMARY_SQL} AND strpos({_MTXT}, {_sql_literal(_SUMMARY_END_MARKER)}) > 0"
    f" AND char_length({_trim_ws(_FORCE_USER_REMAINDER_SQL)}) > 0)"
    f" OR ({_MERGED_SUMMARY_SQL}"
    f" AND char_length({_trim_ws(_MERGED_PRIOR_UNWRAPPED_SQL)}) > 0)))")


def _preview_raw_select() -> str:
    """The preview SELECT over ``m`` (scaffolded rows: head + tail around the skill excerpt join)."""
    return (
        f"CASE WHEN {_STANDALONE_SUMMARY_SQL} THEN {_FORCE_USER_REMAINDER_SQL}"
        f" WHEN {_MERGED_SUMMARY_SQL} THEN {_MERGED_PRIOR_UNWRAPPED_SQL}"
        f" WHEN {_SCAFFOLDED_SQL} AND char_length({_MTXT}) > {_PREVIEW_SCAFFOLD_WINDOW * 2}"
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
_LOCAL_CHAT = ("(chat_id LIKE 'local-' || CHR(37) OR EXISTS "
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

_MEDIA_PREFIX = 'gateway.media.retired.v1.'
_SPILLOVER_PREFIX = 'gateway.spillover.retired.v1.'
_MUTATION_PREFIX = 'gateway.mutation.v1.'
_MUTATION_TRANSCRIPT_FIELDS = ('target_message', 'summary', 'title')
_RECEIPT_SESSION_PATHS = ('result.session_id', 'result.target_message.session_id',
                          'result.target_session_id', 'result.previous_target_session_id')
_DERIVED_LOCAL_PREFIXES = ('bot-chat:', 'a2a:')
_DERIVED_API_SQL = "(source = 'bot_room' OR (source = 'api_server' AND id ~ 'api-[0-9a-f]{16}'))"
_LIVE_LEDGER_SQL = (
    "SELECT 1 FROM session_admissions WHERE target_session_id = %s AND status IN ('queued', 'started', 'unknown')"
    " UNION ALL SELECT 1 FROM worker_executions WHERE session_id = %s"
    " AND status IN ('registered', 'running', 'unknown') LIMIT 1")


def _retired_in(conn, session_id: str) -> bool:
    return conn.execute("SELECT 1 FROM state_meta WHERE key = %s", (RETIRED_PREFIX + session_id,)).fetchone() is not None


def _create_conflict(conn, session_id: str):
    """``(code, message)`` refusing a create of the explicit *session_id*, else None: a live row, or a
    deleted (retired) id."""
    if conn.execute("SELECT 1 FROM sessions WHERE id = %s", (session_id,)).fetchone():
        return 'session_exists', 'Session already exists'
    if _retired_in(conn, session_id):
        return 'session_deleted', 'Session was deleted; its id cannot be reused'
    return None


def _compact_result(conn, admission_id: str) -> None:
    """A retired result keeps the exact-retry outcome but not the transcript copies (the SQLite
    ``_compact_result``, reusing the pure terminal result codec)."""
    from persistence.sqlite.conversation._store.terminal import encode_result, load_result
    saved = conn.execute("SELECT value FROM state_meta WHERE key = %s", (RESULT_PREFIX + admission_id,)).fetchone()
    if saved is None:
        return
    value = load_result(saved["value"])
    result = value.get('result')
    if isinstance(result, dict):
        value['result'] = {**{k: v for k, v in result.items() if k not in _TRANSCRIPT_FIELDS}, 'messages': []}
    conn.execute("UPDATE state_meta SET value = %s WHERE key = %s", (_json(encode_result(value)), RESULT_PREFIX + admission_id))


def _retire_media(conn, payload, request_id: str, *, media_root) -> None:
    """Record the media references the admission payload names (the SQLite ``retire_media``; file
    removal happens at sweep time, the fence only records)."""
    from session_rules.media import admission_media_references, hosted_document_references
    references = admission_media_references(payload)
    if media_root is not None:
        references = [*references, *hosted_document_references(request_id, payload, media_root)]
    references += (payload.get('api_turn_v1', {}) or {}).get('media', []) or []
    for reference in references:
        key = _MEDIA_PREFIX + hashlib.sha256(reference['path'].encode()).hexdigest()
        conn.execute("INSERT INTO state_meta (key, value) VALUES (%s, %s) ON CONFLICT (key) DO NOTHING",
                     (key, json.dumps(reference)))


def _retire_spillover(conn, session_ids) -> None:
    """Record the spillover archives the doomed messages point at, inside the delete transaction."""
    from session_rules.media import retirement_candidates_in_text
    if not session_ids:
        return
    ids = list(session_ids)
    rows = conn.execute(
        "SELECT content, api_content FROM messages WHERE session_id = ANY(%s) AND ("
        "strpos(hermes_nul_safe_text(content), 'spillover') > 0"
        " OR strpos(hermes_nul_safe_text(api_content), 'spillover') > 0)",
        (ids,)).fetchall()
    for row in rows:
        content = row["content"].decode("utf-8") if isinstance(row["content"], (bytes, memoryview)) else (row["content"] or "")
        api_content = row["api_content"].decode("utf-8") if isinstance(row["api_content"], (bytes, memoryview)) else (row["api_content"] or "")
        for name in retirement_candidates_in_text(content, api_content):
            conn.execute("INSERT INTO state_meta (key, value) VALUES (%s, %s) ON CONFLICT (key) DO NOTHING",
                         (_SPILLOVER_PREFIX + name, json.dumps(name)))


def _retire_terminal_receipts(conn, session_ids, *, media_root) -> None:
    """Retire the terminal admission/worker receipts of *session_ids* (the SQLite
    ``retire_terminal_receipts``): busy states refuse, terminal rows move to state_meta tombstones."""
    from persistence.sqlite.conversation._store.terminal import RESULT_PREFIX as _RESULT_PREFIX
    for sid in session_ids:
        admissions = conn.execute("SELECT * FROM session_admissions WHERE target_session_id = %s", (sid,)).fetchall()
        workers = conn.execute("SELECT * FROM worker_executions WHERE session_id = %s", (sid,)).fetchall()
        states = {row['status'] for row in [*admissions, *workers]} - {'terminal'}
        if states:
            raise AuthorityRefusal('unknown_execution' if 'unknown' in states else 'session_busy')
        for raw in admissions:
            row = dict(raw)
            _retire_media(conn, json.loads(row['payload_json']) if row['payload_json'] else {}, row['request_id'], media_root=media_root)
            row['payload_json'] = '{}'
            row['lineage_json'] = '[]'
            row['retired_at'] = time.time()
            _compact_result(conn, row['admission_id'])
            conn.execute("INSERT INTO state_meta (key, value) VALUES (%s, %s)",
                         (ADMISSION_PREFIX + row['admission_id'], _json(row)))
            conn.execute("INSERT INTO state_meta (key, value) VALUES (%s, %s)",
                         (identity_key(row['principal_id'], sid, row['request_id']), json.dumps(row['admission_id'])))
        for raw in workers:
            row = dict(raw)
            receipts = [dict(r) for r in conn.execute(
                "SELECT sequence, payload_digest, result_json FROM worker_receipts WHERE execution_id = %s ORDER BY sequence",
                (row['execution_id'],))]
            closing = admission_fingerprint(canonical_target=sid, payload={'operation': 'execution.finish', 'payload': {}})
            for receipt in receipts:
                if receipt['payload_digest'] != closing:
                    receipt['result_json'] = None
            row['receipts'] = receipts
            row['retired_at'] = time.time()
            conn.execute("INSERT INTO state_meta (key, value) VALUES (%s, %s)",
                         (WORKER_PREFIX + row['execution_id'], _json(row)))
            conn.execute("DELETE FROM worker_receipts WHERE execution_id = %s", (row['execution_id'],))
        conn.execute("DELETE FROM worker_executions WHERE session_id = %s", (sid,))
        conn.execute("DELETE FROM session_admissions WHERE target_session_id = %s", (sid,))


def _retire_mutation_receipts(conn, session_ids) -> None:
    """Keep exact-retry mutation receipts but drop the user content they carry (the SQLite
    ``retire_mutation_receipts``; the GLOB prefix scan is a LIKE, the json paths are jsonb)."""
    if not session_ids:
        return
    ids = list(session_ids)
    rows = conn.execute(
        "SELECT key, value FROM state_meta WHERE key LIKE %s AND hermes_json_valid(value) AND ("
        f"((value::jsonb -> 'result' ->> 'session_id') = ANY(%s))"
        f" OR ((value::jsonb -> 'result' -> 'target_message' ->> 'session_id') = ANY(%s))"
        f" OR ((value::jsonb -> 'result' ->> 'target_session_id') = ANY(%s))"
        f" OR ((value::jsonb -> 'result' ->> 'previous_target_session_id') = ANY(%s)))",
        [_MUTATION_PREFIX + '%', ids, ids, ids, ids]).fetchall()
    for key, raw in rows:
        receipt = json.loads(raw)
        result = receipt.get('result')
        if isinstance(result, dict) and any(f in result for f in _MUTATION_TRANSCRIPT_FIELDS):
            receipt['result'] = {k: (None if k in _MUTATION_TRANSCRIPT_FIELDS else v) for k, v in result.items()}
            conn.execute("UPDATE state_meta SET value = %s WHERE key = %s", (_json(receipt), key))


def _local_conversations(conn):
    """``(members, derived)``: member id -> every id of its local conversation, and the ids of every
    derived (Bot Chat, A2A) conversation (the SQLite ``_local_conversations``)."""
    members, derived = {}, set()
    rows = conn.execute("SELECT value FROM state_meta WHERE key LIKE %s", (POLICY_PREFIX + '*',)).fetchall()
    for (raw,) in [(r['value'],) for r in rows]:
        try:
            receipt = json.loads(raw)
        except (TypeError, ValueError):
            continue
        if not isinstance(receipt, dict):
            continue
        entry, lineage = receipt.get('entry'), receipt.get('lineage')
        ids = tuple(x for x in [receipt.get('session_id'), *(lineage if isinstance(lineage, list) else []),
                                entry.get('session_id') if isinstance(entry, dict) else None] if isinstance(x, str))
        members.update(dict.fromkeys(ids, ids))
        if str(receipt.get('request_id') or '').startswith(_DERIVED_LOCAL_PREFIXES):
            derived.update(ids)
    return members, derived


def _derived_api_sessions(conn, session_ids):
    """The ids among *session_ids* of derived API sessions (Bot Chat / api_server mints)."""
    rows = conn.execute(
        f"SELECT id FROM sessions WHERE {_DERIVED_API_SQL} AND id = ANY(%s)", (list(session_ids),)).fetchall()
    return {row['id'] for row in rows}


def _mark_retired(conn, key: str, *, expires, **fields) -> None:
    """Write the RETIRED fence marker (the SQLite ``_mark_retired``: replace semantics, sweep markers
    age out, a permanent unscoped marker of an earlier release stays unscoped)."""
    value = {'retired_at': time.time(), **fields}
    if expires:
        conn.execute(
            "INSERT INTO state_meta (key, value) VALUES (%s, %s) ON CONFLICT (key) DO UPDATE"
            " SET value = excluded.value WHERE hermes_json_valid(state_meta.value)"
            " AND (state_meta.value::jsonb -> 'expires') IS NOT NULL",
            (key, json.dumps({**value, 'expires': True})))
    else:
        conn.execute(
            "INSERT INTO state_meta (key, value) VALUES (%s, %s) ON CONFLICT (key) DO UPDATE"
            " SET value = CASE"
            "   WHEN NOT hermes_json_valid(state_meta.value) THEN (excluded.value::jsonb - 'own')::text"
            "   WHEN (state_meta.value::jsonb -> 'own') IS NOT NULL"
            "      OR (state_meta.value::jsonb -> 'expires') IS NOT NULL"
            "     THEN excluded.value"
            "   ELSE (excluded.value::jsonb - 'own')::text END",
            (key, json.dumps(value)))


def _retire_routes(conn, session_ids, *, expiring=()) -> None:
    """Retire the routing/receipt rows of *session_ids* (the SQLite ``retire_routes``)."""
    expiring = set(expiring)
    for sid in session_ids:
        _mark_retired(conn, RETIRED_PREFIX + sid, expires=sid in expiring, own=True)
        conn.execute("DELETE FROM state_meta WHERE key = %s", ('gateway.api.settings.v1.' + sid,))
        conn.execute("DELETE FROM state_meta WHERE key = %s", (VIEWER_CLOSE_PREFIX + sid,))
    for row in conn.execute("SELECT scope, session_key, entry_json FROM gateway_routing").fetchall():
        if entry_session_id(row['entry_json']) in set(session_ids):
            conn.execute("DELETE FROM gateway_routing WHERE scope = %s AND session_key = %s",
                         (row['scope'], row['session_key']))


def _retire_sessions(conn, session_ids, *, expiring=(), media_root) -> None:
    """The complete deletion fence for the ids deleted in this transaction (the SQLite
    ``retire_sessions``)."""
    _retire_spillover(conn, session_ids)
    _retire_terminal_receipts(conn, session_ids, media_root=media_root)
    _retire_mutation_receipts(conn, session_ids)
    _retire_routes(conn, session_ids, expiring=expiring)
    _retire_local_receipts(conn, session_ids)


def _retire_prunable(conn, session_ids, *, media_root) -> list:
    """Sweep variant of :func:`_retire_sessions`: fence only the idle sessions (the SQLite
    ``retire_prunable``); a local conversation is swept whole."""
    swept, (conversations, derived) = set(session_ids), _local_conversations(conn)

    def removable(sid):
        whole = conversations.get(sid, (sid,))
        return swept.issuperset(whole) and not any(
            conn.execute(_LIVE_LEDGER_SQL, (x, x)).fetchone() for x in whole)

    quiet = [sid for sid in session_ids if removable(sid)]
    derived |= _derived_api_sessions(conn, quiet)
    _retire_sessions(conn, quiet, expiring=[sid for sid in quiet if sid not in derived], media_root=media_root)
    return quiet


def _retire_local_receipts(conn, session_ids) -> None:
    """Deletion fence half for owner receipts (the SQLite ``retire_local_receipts``)."""
    targets = set(session_ids)
    if not targets:
        return
    gone = [row['key'] for row in conn.execute(
        "SELECT key, value FROM state_meta WHERE key LIKE %s", (POLICY_PREFIX + '*',)).fetchall()
        if _policy_target(row['value']) in targets]
    gone += [row['key'] for row in conn.execute(
        "SELECT key, value FROM state_meta WHERE key LIKE %s", (API_DECLARED_PREFIX + '*',)).fetchall()
        if row['value'] in targets]
    gone += [API_BINDING_PREFIX + sid for sid in targets]
    for key in gone:
        conn.execute("DELETE FROM state_meta WHERE key = %s", (key,))


def _policy_target(raw):
    try:
        value = json.loads(raw)
    except (TypeError, ValueError):
        return None
    entry = value.get('entry') if isinstance(value, dict) else None
    return entry.get('session_id') if isinstance(entry, dict) else None


# ── local conversation lineages (the SQLite local store, cache-less: a PG read is one query) ──

def _local_lineage_owner(conn, session_id: str) -> str:
    """The creation id whose receipt lineage holds physical segment *session_id*, else itself."""
    row = conn.execute("SELECT chat_id FROM sessions WHERE id = %s", (session_id,)).fetchone()
    owner = row['chat_id'] if row else None
    if not owner or owner == session_id:
        return session_id
    saved = conn.execute("SELECT value FROM state_meta WHERE key = %s", (POLICY_PREFIX + owner,)).fetchone()
    try:
        lineage = json.loads(saved['value']).get('lineage') if saved else None
    except (TypeError, ValueError, AttributeError):
        return session_id
    return owner if isinstance(lineage, list) and session_id in lineage else session_id


def _local_conversation_ids(conn, session_id: str) -> list:
    """Every row of the local conversation *session_id* belongs to (the SQLite ``local_conversation_ids``)."""
    owner = _local_lineage_owner(conn, session_id)
    saved = conn.execute("SELECT value FROM state_meta WHERE key = %s", (POLICY_PREFIX + owner,)).fetchone()
    try:
        lineage = json.loads(saved['value']).get('lineage') if saved else None
    except (TypeError, ValueError, AttributeError):
        lineage = None
    if not isinstance(lineage, list):
        return [session_id]
    return list(dict.fromkeys([owner, *(sid for sid in lineage if isinstance(sid, str))]))


def _owned_lineage_ids(conn, session_id: str) -> list:
    """[session_id] outside a local reset/compression lineage, else the whole conversation."""
    return list(dict.fromkeys([session_id, *_local_conversation_ids(conn, session_id)]))


def _local_lineage_index(conn):
    """``(superseded, heads)`` for listing a local conversation as one row (the SQLite
    ``local_lineage_index``, without its connection cache)."""
    superseded, heads = set(), {}
    rows = conn.execute(
        "SELECT value FROM state_meta WHERE key LIKE %s AND hermes_json_valid(value)"
        " AND jsonb_array_length(value::jsonb -> 'lineage') > 1", (POLICY_PREFIX + '*',)).fetchall()
    for row in rows:
        receipt = json.loads(row['value'])
        owner, lineage = receipt.get('session_id'), receipt.get('lineage')
        if not isinstance(owner, str) or not isinstance(lineage, list) or not all(isinstance(sid, str) for sid in lineage):
            continue
        ended = {r['id']: r['end_reason'] for r in conn.execute(
            f"SELECT id, end_reason FROM sessions WHERE id = ANY(%s)", (lineage,)).fetchall()}
        starts = [sid for i, sid in enumerate(lineage) if i == 0 or ended.get(lineage[i - 1]) == 'session_reset']
        superseded.update(starts[:-1])
        heads[starts[-1]] = (owner, lineage)
    return superseded, heads


def _exclude_superseded_segments(conn, where_clauses, params) -> dict:
    """Drop earlier reset segments from a listing/count (one conversation, one row); returns ``heads``."""
    superseded, heads = _local_lineage_index(conn)
    if superseded:
        where_clauses.append("s.id <> ALL(%s)")
        params.append(sorted(superseded))
    return heads


def _annotate_local_lineages(sessions, heads) -> list:
    """Name every segment on the conversation's representative row (the SQLite ``annotate_local_lineages``)."""
    for row in sessions:
        found = heads.get(row.get('_lineage_root_id') or row['id'])
        if found is not None:
            row['_lineage_ids'] = list(dict.fromkeys([*found[1], *(row.get('_lineage_ids') or ())]))
    return sessions


# ── the compression chain walk (the SQLite compression store, conn-bound) ──

_CHAIN_STEP_SQL = (
    "SELECT child.id FROM sessions parent"
    " JOIN sessions child ON child.parent_session_id = parent.id"
    " WHERE parent.id = %s AND parent.end_reason = 'compression'"
    f" AND {_json_key('child.model_config', '_branched_from')} IS NULL"
    f" AND {_json_key('child.model_config', '_delegate_from')} IS NULL"
    f" AND NOT ({_RESET_CHILD_SQL.format(a='child')})"
    " AND COALESCE(child.source, '') <> 'tool'"
    " ORDER BY CASE WHEN child.end_reason = 'compression' THEN 0"
    " WHEN child.ended_at IS NULL THEN 1 ELSE 2 END,"
    f" {_last_active_by_id('child.id')} DESC, child.started_at DESC, child.id DESC"
    " LIMIT 1"
)


def _walk_compression_chain(conn, session_id: str) -> list:
    """Root-first ids from *session_id* through its tip (the SQLite ``_walk_compression_chain``)."""
    current = session_id
    chain = [current] if current else []
    seen = set(chain)
    for _ in range(_CHAIN_CAP):
        row = conn.execute(_CHAIN_STEP_SQL, (current,)).fetchone()
        child_id = row["id"] if row is not None else None
        if not child_id or child_id in seen:
            return chain
        seen.add(child_id)
        current = child_id
        chain.append(child_id)
    return chain


def _compression_tip_on(conn, session_id: str) -> str:
    """``get_compression_tip`` read on the writer's own connection."""
    return _walk_compression_chain(conn, session_id)[-1]


def _get_compression_tip(store: PostgresStore, session_id: str) -> Optional[str]:
    """Live tip of a compression chain (the SQLite ``get_compression_tip``)."""
    row = store.one("SELECT id FROM sessions WHERE id = %s", session_id)
    if row is None:
        return session_id if not session_id else None
    chain = _walk_compression_chain(store.connection(), session_id)
    return chain[-1] if chain else session_id


# ── titles (the SQLite titles store, ported to the PG store) ──

def _bot_chat_title_released(archived, auto_archived, tip_end_reason) -> bool:
    return bool(archived) and not auto_archived and (tip_end_reason or "") not in _RECOVERABLE_END_REASONS


def _is_compression_ancestor(conn, *, ancestor_id: str, descendant_id: str) -> bool:
    """True if *ancestor_id* is a compression predecessor of *descendant_id* (the SQLite helper)."""
    if not ancestor_id or not descendant_id or ancestor_id == descendant_id:
        return False
    edge = _COMPRESSION_CHILD_SQL.format(a="child") + _non_continuation_child_sql("child.", "parent.id")
    return conn.execute(f"""
            WITH RECURSIVE ancestors(id) AS (
                SELECT %s
                UNION
                SELECT parent.id
                FROM ancestors a
                JOIN sessions child ON child.id = a.id
                JOIN sessions parent ON parent.id = child.parent_session_id
                WHERE {edge}
            )
            SELECT 1 FROM ancestors WHERE id = %s AND id <> %s LIMIT 1
            """, (descendant_id, ancestor_id, descendant_id)).fetchone() is not None


def _resolve_title_conflict(conn, session_id: str, title: str) -> Optional[str]:
    """Free ``title`` for ``session_id`` in the caller's write transaction (the SQLite
    ``_resolve_title_conflict``; raises ValueError when another session legitimately holds it)."""
    conflict = conn.execute(
        f"SELECT id, archived, auto_archived, hidden, (NOT ({_EMPTY_SESSION_WHERE.replace('sessions.', 's.')})) AS ghost"
        " FROM sessions s WHERE s.title = %s AND s.id <> %s", (title, session_id),
    ).fetchone()
    if not conflict:
        return None
    conflict_id = conflict["id"]
    if _is_compression_ancestor(conn, ancestor_id=conflict_id, descendant_id=session_id):
        conn.execute("UPDATE sessions SET title = NULL WHERE id = %s", (conflict_id,))
        return conflict_id
    retired = title == CANONICAL_BOT_CHAT_TITLE and _bot_chat_title_released(
        conflict["archived"], conflict["auto_archived"], conn.execute(
            "SELECT end_reason FROM sessions WHERE id = %s",
            (_compression_tip_on(conn, conflict_id),)).fetchone()["end_reason"])
    ghost = not conflict["ghost"]  # the NOT() flip above: a live empty ended row
    if retired or (ghost and not conflict["hidden"]):
        conn.execute("UPDATE sessions SET title = NULL, title_source = NULL WHERE id = %s", (conflict_id,))
        return conflict_id
    raise ValueError(f"Title '{title}' is already in use by session {conflict_id}")


def _set_session_title_in_transaction(conn, session_id: str, title, *, source: str) -> list:
    """Write a title with provenance precedence (the SQLite ``_set_session_title_in_transaction``)."""
    title = sanitize_title(title)
    is_user = source == TITLE_SOURCE_USER
    new_rank = _title_rank(source) if not is_user else None
    affected: list = []
    current = conn.execute(
        "SELECT title, title_source, hidden FROM sessions WHERE id = %s", (session_id,),
    ).fetchone()
    if current is None:
        return []
    if ((current["title"] or "") == CANONICAL_BOT_CHAT_TITLE and bool(current["hidden"])
            and title != CANONICAL_BOT_CHAT_TITLE):
        if is_user:
            raise ValueError("This is the bot's canonical Bot Chat — its name is its "
                             "identity, and renaming it would orphan the conversation. "
                             "To start fresh, create a new bot instead.")
        return []
    if not is_user and current["title"] is not None and _title_rank(current["title_source"] or "") >= new_rank:
        return []
    if title:
        cleared = _resolve_title_conflict(conn, session_id, title)
        if cleared:
            affected.append(cleared)
    changed = conn.execute(
        "UPDATE sessions SET title = %s, title_source = %s WHERE id = %s AND title IS NOT DISTINCT FROM %s"
        " AND title_source IS NOT DISTINCT FROM %s",
        (title, source if title else None, session_id, current["title"], current["title_source"]),
    ).rowcount
    return [*affected, session_id] if changed else affected


def _retire_canonical_bot_chat_in_transaction(conn, session_id: str) -> list:
    """Retire the canonical Bot Chat's name on a deliberate archive (the SQLite helper)."""
    lineage = (session_id, session_id)
    held = [row['id'] for row in conn.execute(
        _LINEAGE_CTE_SQL + "SELECT id FROM sessions WHERE id IN (SELECT id FROM lineage) AND title = %s",
        (*lineage, CANONICAL_BOT_CHAT_TITLE)).fetchall()]
    if not held:
        return []
    conn.execute(
        _LINEAGE_CTE_SQL + "UPDATE sessions SET end_reason = CASE WHEN end_reason IN "
        f"({_RECOVERABLE_END_REASONS_SQL}) THEN NULL ELSE end_reason END, hidden = 0 "
        "WHERE id IN (SELECT id FROM lineage)", lineage)
    for sid in held:
        conn.execute("UPDATE sessions SET title = NULL, title_source = NULL WHERE id = %s", (sid,))
    return held


# ── model-config merge and the conversation generation (the SQLite messages store, ported) ──

def _merge_model_config_json(conn, session_id: str, patch: dict) -> Optional[str]:
    """SELECT + tolerant-parse + merge *patch* into model_config; ``None`` removes a key. Returns
    serialized JSON (``None`` when empty) or ``None``-ish missing marker via the sentinel return."""
    row = conn.execute("SELECT model_config FROM sessions WHERE id = %s", (session_id,)).fetchone()
    if row is None:
        return _MODEL_CONFIG_ROW_MISSING
    config = _parse_model_config(row["model_config"])
    for key, value in patch.items():
        if value is None:
            config.pop(key, None)
        else:
            config[key] = value
    return json.dumps(config) if config else None


_MODEL_CONFIG_ROW_MISSING = object()

_BUMP_GENERATION_SQL = (
    "INSERT INTO conversation_generations (source, session_key, generation) VALUES (%s, %s, 1)"
    " ON CONFLICT (source, session_key) DO UPDATE SET generation = conversation_generations.generation + 1")


def _bump_conversation_generation(conn, session_id: str, end_reason: str) -> None:
    """Advance the peer's conversation generation past a boundary (the SQLite
    ``_bump_conversation_generation``; only reset reasons count)."""
    if end_reason not in _RESET_END_REASONS:
        return
    row = conn.execute("SELECT source, session_key FROM sessions WHERE id = %s", (session_id,)).fetchone()
    if row is None:
        return
    source, session_key = (str(row[k] or "").strip() for k in ("source", "session_key"))
    if source and session_key:
        conn.execute(_BUMP_GENERATION_SQL, (source, session_key))


# ── session lifecycle writes (the SQLite sessions store, ported to the PG store) ──

def _jsonb_key(col: str, key: str) -> str:
    """Tolerant jsonb scalar lookup (the SQLite ``_sql_json_extract``; NULL or invalid JSON → NULL)."""
    return f"CASE WHEN {col} IS NULL OR NOT hermes_json_valid({col}) THEN NULL ELSE ({col}::jsonb -> {_sql_literal(key)}) END"


def _jsonb_key_text(col: str, key: str) -> str:
    return f"CASE WHEN {col} IS NULL OR NOT hermes_json_valid({col}) THEN NULL ELSE ({col}::jsonb ->> {_sql_literal(key)}) END"


def _in_list(values) -> str:
    """``= ANY(%s)`` clause with one %s for the whole sequence."""
    return "= ANY(%s)"


def _end_and_bump(conn, session_id: str, reason: str) -> int:
    """Run an end-stamp UPDATE; only a boundary this call actually wrote advances the conversation
    generation (a no-op must not rotate the peer). Returns rowcount (the SQLite ``_end_and_bump``)."""
    changed = conn.execute(
        "UPDATE sessions SET ended_at = %s, end_reason = %s WHERE id = %s AND ended_at IS NULL",
        (time.time(), reason, session_id)).rowcount
    if changed:
        _bump_conversation_generation(conn, session_id, reason)
    return changed


def end_session(store: "PostgresStore", session_id: str, reason: str) -> None:
    """Mark a session ended; the first end_reason wins (the SQLite ``end_session``)."""
    with store.tx() as conn:
        _end_and_bump(conn, session_id, reason)


def _retire_undrained_queue_rows_conn(conn, session_id: str) -> int:
    """The still-marked == never-drained accept-row UPDATE (#125577), on a caller's connection."""
    return conn.execute(
        "UPDATE messages SET active = 0 "
        "WHERE session_id = %s AND role = 'user' AND active = 1 AND hermes_json_valid(display_metadata)"
        " AND COALESCE((display_metadata::jsonb -> %s)::text, '0') IN ('1', 'true')",
        (session_id, QUEUED_PROMPT_METADATA_KEY)).rowcount


def retire_undrained_queue_rows(store: "PostgresStore", session_id: str) -> int:
    """Deactivate busy-queue accept rows that never drained (#125577; the SQLite
    ``retire_undrained_queue_rows``). Deactivated, never deleted."""
    with store.tx() as conn:
        return _retire_undrained_queue_rows_conn(conn, session_id)


def reopen_session(store: "PostgresStore", session_id: str) -> None:
    """Clear ended_at/end_reason so a session can be resumed; first freeze markerless legacy reset
    children, skipping explicit fork/delegate provenance and children that predate the parent itself
    (the SQLite ``reopen_session``)."""
    with store.tx() as conn:
        _retire_undrained_queue_rows_conn(conn, session_id)
        conn.execute(
            "UPDATE sessions AS child SET model_config = ("
            "CASE WHEN hermes_json_valid(child.model_config) THEN child.model_config::jsonb"
            " ELSE '{}'::jsonb END || jsonb_build_object('_reset_from', child.parent_session_id))::text "
            f"WHERE child.parent_session_id = %s AND {_jsonb_key_text('child.model_config', '_reset_from')} IS NULL "
            f"AND {_jsonb_key_text('child.model_config', '_branched_from')} IS NULL "
            f"AND {_jsonb_key_text('child.model_config', '_delegate_from')} IS NULL "
            "AND COALESCE(child.source, '') != 'tool' "
            "AND child.started_at >= (SELECT p.started_at FROM sessions p WHERE p.id = child.parent_session_id) "
            f"AND {_legacy_reset_child_sql('child', _RESET_END_REASONS_SQL)}",
            (session_id,),
        )
        conn.execute(
            "UPDATE sessions SET ended_at = NULL, end_reason = NULL WHERE id = %s", (session_id,),
        )
        _unarchive_auto_archived_lineage(conn, session_id)


def promote_to_session_reset(store: "PostgresStore", session_id: str, reason: str) -> bool:
    """Durably mark an intentional reset boundary on live or recoverable rows (the SQLite
    ``promote_to_session_reset``); a failed write answers False."""
    if not session_id:
        return False
    try:
        with store.tx() as conn:
            return bool(_end_and_bump_recoverable(conn, session_id, reason))
    except Exception:
        return False


def _end_and_bump_recoverable(conn, session_id: str, reason: str) -> int:
    changed = conn.execute(
        "UPDATE sessions SET ended_at = %s, end_reason = %s WHERE id = %s AND (ended_at IS NULL "
        f"OR end_reason IN ({_RECOVERABLE_END_REASONS_SQL}))",
        (time.time(), reason, session_id)).rowcount
    if changed:
        _bump_conversation_generation(conn, session_id, reason)
    return changed


def update_session_cwd(
    store: "PostgresStore", session_id: str, cwd: str, git_branch: Optional[str] = None,
    git_repo_root: Optional[str] = None, replace_git_meta: bool = False,
) -> Optional[int]:
    """Persist the authoritative cwd and claim a Git metadata generation (the SQLite
    ``update_session_cwd``). git fields are written only when non-empty, except under
    ``replace_git_meta``."""
    if not session_id or not cwd:
        return None
    branch = (git_branch or "").strip()
    repo_root = (git_repo_root or "").strip()
    with store.tx() as conn:
        current = conn.execute("SELECT cwd FROM sessions WHERE id = %s", (session_id,)).fetchone()
        if current is None:
            return None
        sets = ["cwd = %s", "git_metadata_generation = COALESCE(git_metadata_generation, 0) + 1"]
        params: list[Any] = [cwd]
        if current["cwd"] != cwd or replace_git_meta:
            sets.extend(("git_branch = %s", "git_repo_root = %s"))
            params.extend((branch or None, repo_root or None))
        else:  # same cwd: only overwrite with captured (non-empty) values
            for col, val in (("git_branch", branch), ("git_repo_root", repo_root)):
                if val:
                    sets.append(f"{col} = %s")
                    params.append(val)
        conn.execute(f"UPDATE sessions SET {', '.join(sets)} WHERE id = %s", [*params, session_id])
        row = conn.execute(
            "SELECT git_metadata_generation FROM sessions WHERE id = %s", (session_id,),
        ).fetchone()
        return None if row is None else int(row["git_metadata_generation"])


def publish_session_git_metadata(
    store: "PostgresStore", session_id: str, cwd: str, generation: int,
    git_branch: Optional[str] = None, git_repo_root: Optional[str] = None,
) -> bool:
    """Publish async Git enrichment only while its cwd claim is current (the SQLite
    ``publish_session_git_metadata``)."""
    valid_generation = isinstance(generation, int) and not isinstance(generation, bool) and generation >= 1
    if not session_id or not cwd or not valid_generation:
        return False
    fields = [
        (col, val) for col, val in (
            ("git_branch", (git_branch or "").strip()), ("git_repo_root", (git_repo_root or "").strip()),
        ) if val
    ]
    if not fields:
        return False
    with store.tx() as conn:
        cur = conn.execute(
            f"UPDATE sessions SET {', '.join(f'{col} = %s' for col, _ in fields)} "
            "WHERE id = %s AND cwd = %s AND git_metadata_generation = %s",
            [val for _, val in fields] + [session_id, cwd, generation],
        )
        return cur.rowcount == 1


def backfill_repo_roots(store: "PostgresStore", cwd_to_root: dict) -> None:
    """Backfill git repo roots for cwds without one; never clobbers a recorded root (the SQLite
    ``backfill_repo_roots``)."""
    pairs = [(root, cwd) for cwd, root in cwd_to_root.items() if root and cwd]
    if pairs:
        with store.tx() as conn:
            for root, cwd in pairs:
                conn.execute(
                    "UPDATE sessions SET git_repo_root = %s WHERE cwd = %s AND COALESCE(git_repo_root, '') = ''",
                    (root, cwd),
                )


def touch_session_activity(
    store: "PostgresStore", session_id: str, ts: Optional[float] = None, *,
    description: Optional[str] = None, provenance: Optional[Any] = None,
) -> None:
    """Stamp durable mid-turn activity (the SQLite ``touch_session_activity``); never moves
    ``last_activity_at`` backwards."""
    if not session_id:
        return
    when = float(ts if ts is not None else time.time())
    with store.tx() as conn:
        conn.execute(
            "UPDATE sessions SET last_activity_at = %s, "
            "last_activity_description = %s, last_activity_provenance = %s "
            "WHERE id = %s AND (last_activity_at IS NULL OR last_activity_at < %s)",
            (
                when, bound_activity_description(description),
                normalize_activity_provenance(provenance).value, session_id, when,
            ),
        )


def clear_session_activity_labels(store: "PostgresStore", session_id: str) -> None:
    """Clear activity labels after a turn, keeping ``last_activity_at`` (the SQLite
    ``clear_session_activity_labels``); a no-op clear skips the write."""
    if not session_id:
        return
    row = store.one(
        "SELECT last_activity_description, last_activity_provenance FROM sessions WHERE id = %s",
        session_id)
    if row is not None and not row["last_activity_description"] and (
            not row["last_activity_provenance"] or row["last_activity_provenance"] == ActivityProvenance.UNKNOWN.value):
        return
    with store.tx() as conn:
        conn.execute(
            "UPDATE sessions SET last_activity_description = %s, last_activity_provenance = %s WHERE id = %s",
            ("", ActivityProvenance.UNKNOWN.value, session_id),
        )


def update_session_meta(
    store: "PostgresStore", session_id: str, model_config_json: str, model: Optional[str] = None,
) -> None:
    """Update model_config and (COALESCE) optionally model (the SQLite ``update_session_meta``)."""
    with store.tx() as conn:
        conn.execute(
            "UPDATE sessions SET model_config = %s, model = COALESCE(%s, model) WHERE id = %s",
            (model_config_json, model, session_id),
        )


def update_system_prompt(store: "PostgresStore", session_id: str, system_prompt: Optional[str]) -> None:
    """Store the full assembled system prompt snapshot (the SQLite ``update_system_prompt``)."""
    with store.tx() as conn:
        prompt_hash = _store_system_prompt(conn, system_prompt)
        conn.execute(
            "UPDATE sessions SET system_prompt_hash = %s, system_prompt = NULL WHERE id = %s",
            (prompt_hash, session_id),
        )
        _delete_unreferenced_system_prompts(conn)


def update_session_tool_names(store: "PostgresStore", session_id: str, pin: Any) -> None:
    """Persist the session's tools[] pin in the content-addressed prompt store (the SQLite
    ``update_session_tool_names``); ``None`` clears."""
    payload = json.dumps(pin) if pin is not None else None
    with store.tx() as conn:
        conn.execute(
            "UPDATE sessions SET tool_names = %s WHERE id = %s",
            (_store_system_prompt(conn, payload), session_id),
        )
        _delete_unreferenced_system_prompts(conn)


def update_session_model(
    store: "PostgresStore", session_id: str, model: str, provider: Optional[str] = None, *,
    base_url: Optional[str] = None, api_mode: Optional[str] = None,
) -> None:
    """Set the model after a mid-session /model switch (the SQLite ``update_session_model``);
    drops any Browser runtime lock (lineage markers survive)."""
    patch: dict[str, Any] = {"browser_model_lock": None}
    if model:
        patch["model"] = model
    if provider:
        route = {"provider": provider, "base_url": base_url or None, "api_mode": api_mode or None}
        patch.update(route, gateway_runtime={**route, "model": model or None})
    _write_model_config_patch(store, session_id, patch,
                             "UPDATE sessions SET model = %s, model_config = %s WHERE id = %s",
                             lambda merged: (model, merged, session_id))


def _write_model_config_patch(
    store: "PostgresStore", session_id: str, patch: dict,
    sql: str = "UPDATE sessions SET model_config = %s WHERE id = %s",
    params: Optional[Callable[[Optional[str]], tuple]] = None,
) -> None:
    """Merge ``patch`` into model_config then run ``sql`` with ``params(merged)`` in one write
    transaction; no-op when the row doesn't exist (the SQLite ``_write_model_config_patch``)."""
    with store.tx() as conn:
        merged = _merge_model_config_json(conn, session_id, patch)
        if merged is _MODEL_CONFIG_ROW_MISSING:
            return
        conn.execute(sql, params(merged) if params else (merged, session_id))


def patch_session_model_config(store: "PostgresStore", session_id: str, patch: dict) -> None:
    """Merge ``patch`` into model_config atomically (the SQLite ``patch_session_model_config``)."""
    if not session_id or not patch:
        return
    _write_model_config_patch(store, session_id, dict(patch))


def get_session_model_config_value(
    store: "PostgresStore", session_id: str, key: str, default: Any = None,
) -> Any:
    """Read one key out of a session's model_config JSON (the SQLite ``get_session_model_config_value``)."""
    row = store.one("SELECT model_config FROM sessions WHERE id = %s", session_id)
    if row is None:
        return default
    return _parse_model_config(row["model_config"]).get(key, default)


def update_session_runtime_lock(
    store: "PostgresStore", session_id: str, *, model: Optional[str] = None, provider: Optional[str] = None,
    model_options: Optional[dict] = None, route_source: Optional[str] = None, confirmed: bool = False,
) -> None:
    """Persist a Browser / API-client runtime lock into model_config (the SQLite
    ``update_session_runtime_lock``; lineage markers survive)."""
    lock = {
        "provider": provider or "", "model": model or "", "model_options": model_options or {},
        "route_source": route_source or "", "confirmed": bool(confirmed), "updated_at": time.time(),
    }
    _write_model_config_patch(
        store, session_id, {"browser_model_lock": lock},
        "UPDATE sessions SET model_config = %s, model = COALESCE(%s, model) WHERE id = %s",
        lambda merged: (merged, model, session_id),
    )


def set_session_yolo(store: "PostgresStore", session_id: str, enabled: bool) -> None:
    """Persist the per-session YOLO flag (the SQLite ``set_session_yolo``); no-op without a row."""
    if not session_id:
        return
    _write_model_config_patch(store, session_id, {"yolo_mode": bool(enabled)})


def _store_system_prompt(conn, system_prompt: Optional[str]) -> Optional[str]:
    """Store a prompt (or tool pin) in the content-addressed ``system_prompts`` table; the hash."""
    if system_prompt is None:
        return None
    prompt_hash = hashlib.sha256(system_prompt.encode("utf-8")).hexdigest()
    conn.execute(
        "INSERT INTO system_prompts (hash, prompt) VALUES (%s, %s) ON CONFLICT (hash) DO NOTHING",
        (prompt_hash, system_prompt),
    )
    return prompt_hash


def _delete_unreferenced_system_prompts(conn) -> None:
    conn.execute(
        """DELETE FROM system_prompts
           WHERE NOT EXISTS (SELECT 1 FROM sessions WHERE sessions.system_prompt_hash = system_prompts.hash)
             AND NOT EXISTS (SELECT 1 FROM sessions WHERE sessions.tool_names = system_prompts.hash)""")

# ── reads, backfills and lineage-column writes (the SQLite sessions store, ported) ──

def get_session_row(store: "PostgresStore", session_id: str) -> Optional[dict]:
    """The session row with prompt-store resolution (the SQLite ``get_session`` read shape)."""
    row = store.one(
        """SELECT s.*, COALESCE(sp.prompt, s.system_prompt) AS _system_prompt_resolved,
                  COALESCE(tp.prompt, s.tool_names) AS _tool_names_resolved
           FROM sessions s
           LEFT JOIN system_prompts sp ON sp.hash = s.system_prompt_hash
           LEFT JOIN system_prompts tp ON tp.hash = s.tool_names
           WHERE s.id = %s""",
        session_id,
    )
    return dict(row) if row else None


def get_session_list_row(store: "PostgresStore", session_id: str) -> Optional[dict]:
    """Picker columns only: the live listing skips the prompt/tool JOIN (the SQLite
    ``get_session_list_row``)."""
    row = store.one(
        "SELECT id, source, title, started_at, ended_at, message_count, model, hidden, archived "
        "FROM sessions WHERE id = %s",
        session_id,
    )
    return dict(row) if row else None


def resolve_session_id(store: "PostgresStore", session_id_or_prefix: str) -> Optional[str]:
    """Exact id, else the single unambiguous prefix match, else None (the SQLite
    ``resolve_session_id``)."""
    if store.one("SELECT 1 FROM sessions WHERE id = %s", session_id_or_prefix) is not None:
        return session_id_or_prefix
    matches = store.all(
        "SELECT id FROM sessions WHERE id LIKE %s ESCAPE '\\' ORDER BY started_at DESC LIMIT 2",
        f"{_escape_like(session_id_or_prefix)}%",
    )
    return matches[0]["id"] if len(matches) == 1 else None


def backfill_null_session_profiles(store: "PostgresStore", profile_name: str) -> int:
    """Stamp the store's own profile onto legacy NULL-owner rows (the SQLite
    ``backfill_null_session_profiles``); never overwrites a non-NULL owner."""
    stamp = (profile_name or "").strip()
    if not stamp:
        return 0
    with store.tx() as conn:
        cur = conn.execute(
            "UPDATE sessions SET profile_name = %s "
            "WHERE profile_name IS NULL OR TRIM(profile_name) = ''",
            (stamp,),
        )
        return int(cur.rowcount or 0)


def backfill_acp_session_cwd(store: "PostgresStore") -> int:
    """Promote ``model_config.cwd`` into the cwd column for ACP rows lacking one (the SQLite
    ``backfill_acp_session_cwd``); only fills NULL/empty, explicit column values win."""
    with store.tx() as conn:
        cur = conn.execute(
            "UPDATE sessions SET cwd = (model_config::jsonb ->> 'cwd') "
            "WHERE source = 'acp' AND COALESCE(cwd, '') = '' "
            " AND hermes_json_valid(model_config) "
            " AND COALESCE(model_config::jsonb ->> 'cwd', '') != ''")
        return int(cur.rowcount or 0)


def _set_lineage_column_in_transaction(
    conn, column: str, session_id: str, value: Any, *, extra_set_sql: str = "",
):
    """Set one ``sessions`` column across a whole compression lineage, returning the affected ids
    (the SQLite ``_set_lineage_column_in_transaction``). A plain ``archived`` write clears the
    sweep's ``auto_archived`` provenance."""
    if column == "archived" and not extra_set_sql:
        extra_set_sql = ", auto_archived = 0"
    rows = conn.execute(
        _LINEAGE_CTE_SQL + f"""
            UPDATE sessions
            SET {column} = %s{extra_set_sql}
            WHERE id IN (SELECT id FROM lineage)
            RETURNING id
            """,
        (session_id, session_id, value),
    ).fetchall()
    return [row["id"] for row in rows]


def _set_lineage_column(
    store: "PostgresStore", column: str, session_id: str, value: Any, *, extra_set_sql: str = "",
) -> bool:
    """Set one column across a compression lineage (the SQLite ``_set_lineage_column``)."""
    with store.tx() as conn:
        return bool(_set_lineage_column_in_transaction(conn, column, session_id, value,
                                                       extra_set_sql=extra_set_sql))


def set_session_archived(
    store: "PostgresStore", session_id: str, archived: bool, *, notify: bool = True,
) -> tuple[bool, tuple]:
    """Soft-hide (or unhide) a session and its compression lineage (the SQLite
    ``set_session_archived``); the deliberate archive clears the sweep's provenance and retires
    the canonical Bot Chat's name in the same write."""
    def _do(conn):
        row = conn.execute("SELECT archived FROM sessions WHERE id = %s", (session_id,)).fetchone()
        ids = _set_lineage_column_in_transaction(conn, "archived", session_id, int(archived),
                                                extra_set_sql=", auto_archived = 0")
        if archived and ids:
            _retire_canonical_bot_chat_in_transaction(conn, session_id)
        return ids, bool(archived and ids and row is not None and not row["archived"])
    with store.tx() as conn:
        ids, newly_archived = _do(conn)
    return bool(ids), tuple(ids) if newly_archived and notify else ()


def _auto_archive_lineage(store: "PostgresStore", session_id: str) -> bool:
    """The idle sweep's archive: like ``set_session_archived`` but stamps ``auto_archived`` on the
    rows it hides (the SQLite ``_auto_archive_lineage``); a row already archived keeps its provenance."""
    return _set_lineage_column(
        store, "archived", session_id, 1,
        extra_set_sql=", auto_archived = CASE WHEN archived = 0 THEN 1 ELSE auto_archived END",
    )


def _unarchive_auto_archived_lineage(conn, session_id: str) -> bool:
    """Un-hide a lineage the idle sweep archived, on *conn* (the SQLite
    ``_unarchive_auto_archived_lineage``); a lineage with any deliberately archived row is left alone."""
    params = (session_id, session_id)
    manual = conn.execute(
        _LINEAGE_CTE_SQL + """
            SELECT 1 FROM sessions
            WHERE id IN (SELECT id FROM lineage) AND archived <> 0 AND COALESCE(auto_archived, 0) = 0
            LIMIT 1
            """, params).fetchone()
    if manual is not None:
        return False
    return conn.execute(
        _LINEAGE_CTE_SQL + """
            UPDATE sessions SET archived = 0, auto_archived = 0
            WHERE id IN (SELECT id FROM lineage) AND archived <> 0
            """, params).rowcount > 0


def unarchive_recoverable_session(store: "PostgresStore", session_id: str) -> bool:
    """Un-archive a session archived by a recoverable accident (the SQLite
    ``unarchive_recoverable_session``); deliberate archives are left alone."""
    if not session_id:
        return False
    try:
        row = get_session_row(store, session_id)
    except Exception:
        return False
    if not row or not row.get("archived"):
        return False
    tip = row
    try:
        tip_id = _get_compression_tip(store, session_id) or session_id
        if tip_id != session_id:
            tip = get_session_row(store, tip_id) or row
    except Exception:
        pass
    if (tip.get("end_reason") or "") not in _RECOVERABLE_END_REASONS:
        return False
    if not set_session_archived(store, session_id, False, notify=False)[0]:
        return False
    # Clear the accidental end stamp, or a LATER deliberate archive would auto-resurrect on the
    # next lookup.
    with store.tx() as conn:
        conn.execute("UPDATE sessions SET ended_at = NULL, end_reason = NULL WHERE id = %s",
                     (tip["id"],))
    return True


def set_session_pinned(store: "PostgresStore", session_id: str, pinned: bool) -> bool:
    """Pin/unpin a session and its compression lineage (the SQLite ``set_session_pinned``);
    pinning also clears ``hidden`` except for the canonical Bot Chat."""
    result = _set_lineage_column(store, "pinned", session_id, int(pinned))
    if pinned:
        row = get_session_row(store, session_id)
        is_canonical_bot_chat = bool(row) and bool(row.get("hidden")) and (
            (row.get("title") or "") == CANONICAL_BOT_CHAT_TITLE)
        if not is_canonical_bot_chat:
            _set_lineage_column(store, "hidden", session_id, 0)
    return result


def set_session_hidden(store: "PostgresStore", session_id: str, hidden: bool) -> bool:
    """Hide/unhide a session and its compression lineage (the SQLite ``set_session_hidden``)."""
    return _set_lineage_column(store, "hidden", session_id, int(hidden))


def set_session_read(store: "PostgresStore", session_id: str, read: bool = True) -> bool:
    """Mark read/unread across the compression lineage (the SQLite ``set_session_read``);
    ``last_read_at`` is a watermark, NULL = never tracked = read, 0 = unread."""
    return _set_lineage_column(store, "last_read_at", session_id, time.time() if read else 0.0)


def session_unread(session_row: Mapping) -> bool:
    """Unread = activity postdates the ``last_read_at`` watermark (NULL = read)."""
    last_read = session_row.get("last_read_at")
    if last_read is None:
        return False
    last_active = session_row.get("last_active") or session_row.get("started_at")
    return float(last_active or 0) > float(last_read)


def session_lifecycle_statuses(store: "PostgresStore", session_ids: list) -> dict:
    """``{session_id: status}`` from each session's LAST message row (``'empty'`` when none) (the
    SQLite ``session_lifecycle_statuses``); one query, MAX(id) per session — never scans transcripts."""
    ids = [sid for sid in (session_ids or []) if sid]
    if not ids:
        return {}
    statuses: dict = {sid: "empty" for sid in ids}
    rows = store.all(
        """SELECT m.session_id, m.role,
                  (m.tool_calls IS NOT NULL) AS has_tool_calls,
                  m.finish_reason
           FROM messages m
           JOIN (
               SELECT session_id, MAX(id) AS max_id
               FROM messages
               WHERE session_id = ANY(%s)
               GROUP BY session_id
           ) latest ON m.id = latest.max_id""",
        ids,
    )
    for row in rows:
        statuses[row["session_id"]] = classify_session_status(
            role=row["role"], has_tool_calls=bool(row["has_tool_calls"]),
            finish_reason=row["finish_reason"],
        )
    return statuses


# ── the list / search / count machinery (the SQLite sessions store, ported) ──

# Drop empty launch stubs (a CLI crash or a failed resume before the first turn). The port's DDL
# always has ``session_admissions``, so the full predicate applies (the SQLite ``launch_stub_filter_sql``).
_NOT_A_LAUNCH_STUB_SQL = (
    "(s.message_count > 0 OR EXISTS (SELECT 1 FROM session_admissions a WHERE a.target_session_id = s.id))")

_SESSION_COMPACT_EXCLUDED = frozenset({"system_prompt", "system_prompt_hash", "git_metadata_generation"})

_SESSION_ALL_COLS = (
    "id, runtime_revision, runtime_generation, source, created_source, user_id, session_key, chat_id,"
    " chat_type, thread_id, display_name, origin_json, expiry_finalized, model, model_config,"
    " system_prompt, system_prompt_hash, parent_session_id, cwd, git_branch, git_repo_root,"
    " git_metadata_generation, billing_provider, billing_base_url, billing_mode, estimated_cost_usd,"
    " actual_cost_usd, cost_status, cost_source, pricing_version, title, title_source,"
    " last_activity_at, last_activity_description, last_activity_provenance, api_call_count,"
    " handoff_state, handoff_platform, handoff_error, compression_failure_cooldown_until,"
    " compression_failure_error, compression_fallback_streak, compression_ineffective_count,"
    " compression_recovery_deadline, compression_overload_streak, profile_name, transport_profile,"
    " rewind_count, archived, auto_archived, pinned, hidden, last_read_at, tool_names,"
    " transcript_revision, prompt_cache_clock, started_at, ended_at, end_reason, message_count,"
    " tool_call_count, input_tokens, output_tokens, cache_read_tokens, cache_write_tokens,"
    " reasoning_tokens",
)


def _compact_session_cols() -> str:
    """``s.``-prefixed SELECT list of every sessions column except the prompt storage internals
    (the SQLite ``_compact_session_cols``)."""
    return ", ".join(
        f"s.{c}" for c in _SESSION_ALL_COLS.split(", ")
        if c.strip() not in _SESSION_COMPACT_EXCLUDED)


_PROMPT_RESOLVED_SQL = "COALESCE(sp.prompt, s.system_prompt) AS _system_prompt_resolved"


def _chain_search_where(where_sql: str, id_needle: str, search_needle: str) -> tuple:
    """Extend ``where_sql`` with the id_query / search_query filters: a row is admitted when its own
    id or any id in its forward compression chain matches (the SQLite ``_chain_search_where``;
    search also matches titles and a punctuation-stripped form, so ``an94`` finds ``AN-94``)."""
    params: list = []
    clauses: list = []
    esc = "ESCAPE '\\'"  # one backslash in the SQL text: the LIKE escape character

    def like(needle: str) -> str:
        return f"%{_escape_like(needle)}%"

    if id_needle:
        clauses.append(
            "EXISTS (SELECT 1 FROM chain cq        WHERE cq.root_id = s.id"
            f"          AND LOWER(cq.cur_id) LIKE %s {esc})")
        params.append(like(id_needle))
    if search_needle:
        compact_needle = re.sub(r"[\W_]+", "", search_needle)
        search_clause = (
            "EXISTS (SELECT 1 FROM chain cq JOIN sessions cs ON cs.id = cq.cur_id"
            f" WHERE cq.root_id = s.id AND (LOWER(COALESCE(cs.title, '')) LIKE %s {esc}"
            f" OR LOWER(cq.cur_id) LIKE %s {esc}")
        params.extend([like(search_needle)] * 2)
        if compact_needle:
            search_clause += (
                " OR REPLACE(REPLACE(REPLACE(REPLACE(LOWER(COALESCE(cs.title, '')),"
                f" '-', ''), '_', ''), '.', ''), ' ', '') LIKE %s {esc}")
            params.append(like(compact_needle))
        clauses.append(search_clause + "))")
    if not clauses:
        return where_sql, params
    combined = " AND ".join(clauses)
    return (f"{where_sql} AND {combined}" if where_sql else f"WHERE {combined}"), params


def _list_row(row: Mapping) -> dict:
    """Project a list_sessions_rich row: shape the preview, drop internal ordering columns."""
    s = dict(row)
    s["preview"] = _shape_preview(s.pop("_preview_raw", ""))
    s.pop("_effective_last_active", None)
    return s


def _rich_row(row: Mapping) -> dict:
    """Session row dict with ``_preview_raw`` shaped into ``preview`` (the SQLite ``_rich_row``)."""
    s = dict(row)
    s["preview"] = _shape_preview(s.pop("_preview_raw", ""))
    return s


def list_sessions_rich(
    store: "PostgresStore", source: Optional[str] = None, sources: Optional[list] = None,
    exclude_sources: Optional[list] = None, cwd_prefix: Optional[str] = None, limit: int = 20,
    offset: int = 0, include_children: bool = False, min_message_count: int = 0,
    project_compression_tips: bool = True, order_by_last_active: bool = False,
    include_archived: bool = False, archived_only: bool = False, id_query: Optional[str] = None,
    search_query: Optional[str] = None, compact_rows: bool = False, include_pinned: bool = False,
    session_key: Optional[str] = None, include_hidden: bool = False, include_subagents: bool = False,
    local_created_sources: Optional[Sequence[str]] = None, workspace_key: Optional[str] = None,
) -> list:
    """List sessions with preview and ``last_active`` in one query (the SQLite
    ``list_sessions_rich``; the chain walk is depth-capped, so a corrupt cycle cannot loop forever)."""
    where_clauses, params = _session_filter_where(
        exclude_children=not include_children, source=source, sources=sources, session_key=session_key,
        exclude_sources=exclude_sources, cwd_prefix=cwd_prefix, min_message_count=min_message_count,
        archived_only=archived_only, include_archived=include_archived, include_subagents=include_subagents,
        local_created_sources=local_created_sources, workspace_key=workspace_key,
    )
    if not include_hidden and not archived_only:
        where_clauses.append("s.hidden = 0")
    heads = {}
    if not include_children:
        heads = _exclude_superseded_segments(store.connection(), where_clauses, params)
    where_sql = _where_sql(where_clauses)
    select_head = (
        f"SELECT {_compact_session_cols() if compact_rows else 's.*'}"
        + ("" if compact_rows else f", {_PROMPT_RESOLVED_SQL}")
        + f",\n                    {_preview_raw()},\n                    "
    )
    prompt_join = (
        "" if compact_rows else "LEFT JOIN system_prompts sp ON sp.hash = s.system_prompt_hash")
    from_sessions = f"FROM sessions s\n                {prompt_join}"
    if order_by_last_active:
        id_needle = (id_query or "").strip().lower()
        search_needle = (search_query or "").strip().lower()
        seed_where = where_sql if (id_needle or search_needle) else _where_sql(where_clauses + [
            "EXISTS (SELECT 1 FROM sessions _seed_c WHERE _seed_c.parent_session_id = s.id)"])
        outer_where, id_params = _chain_search_where(where_sql, id_needle, search_needle)
        query = f"""
                WITH RECURSIVE chain(root_id, cur_id, depth) AS (
                    SELECT s.id, s.id, 0 FROM sessions s {seed_where}
                    UNION ALL
                    SELECT c.root_id, child.id, c.depth + 1
                    FROM chain c
                    JOIN sessions parent ON parent.id = c.cur_id
                    JOIN sessions child ON child.parent_session_id = c.cur_id
                    WHERE parent.end_reason = 'compression'
                      AND c.depth < {_CHAIN_CAP}
                      AND {_json_key_text('child.model_config', '_branched_from')} IS NULL
                      AND {_json_key_text('child.model_config', '_delegate_from')} IS NULL
                      AND NOT ({_RESET_CHILD_SQL.format(a='child')})
                      AND COALESCE(child.source, '') != 'tool'
                ),
                chain_max AS (
                    SELECT
                        root_id,
                        MAX({_last_active_by_id("cur_id")}) AS effective_last_active
                    FROM chain
                    GROUP BY root_id
                ),
                page AS (
                    SELECT s.id AS page_id,
                        COALESCE(cm.effective_last_active, {_last_active("s")},
                                 {_in_window("s.started_at")}) AS effective_last_active
                    FROM sessions s
                    LEFT JOIN chain_max cm ON cm.root_id = s.id
                    {outer_where}
                    ORDER BY effective_last_active DESC, s.started_at DESC, s.id DESC
                    LIMIT %s OFFSET %s
                )
                {select_head}{_last_active("s")} AS last_active,
                    page.effective_last_active AS _effective_last_active
                FROM page
                JOIN sessions s ON s.id = page.page_id
                {prompt_join}
                ORDER BY _effective_last_active DESC, s.started_at DESC, s.id DESC
            """
        # WHERE binds twice (seed + outer), each on its own connection call.
        rows = store.all(query, *(params + params + id_params + [limit, offset]))
    else:
        query = f"""
                {select_head}{_last_active("s")} AS last_active
                {from_sessions}
                {where_sql}
                ORDER BY s.started_at DESC
                LIMIT %s OFFSET %s
            """
        rows = store.all(query, *(params + [limit, offset]))
    sessions = [_list_row(row) for row in rows]
    if include_pinned:
        seen_ids = {s["id"] for s in sessions}
        pinned_clauses, pinned_params = _session_filter_where(
            exclude_children=not include_children, source=source, sources=sources,
            session_key=session_key, exclude_sources=exclude_sources, cwd_prefix=cwd_prefix,
            min_message_count=min_message_count, archived_only=False, include_archived=True,
            include_subagents=include_subagents, local_created_sources=local_created_sources,
            workspace_key=workspace_key,
        )
        if not include_hidden and not archived_only:
            pinned_clauses.append("s.hidden = 0")
        if heads:
            _exclude_superseded_segments(store.connection(), pinned_clauses, pinned_params)
        pinned_clauses.append("s.pinned = 1")
        pinned_where = _where_sql(pinned_clauses)
        pinned_query = f"""
                {select_head}{_last_active("s")} AS last_active
                {from_sessions}
                {pinned_where}
                ORDER BY s.started_at DESC
            """
        for row in store.all(pinned_query, *pinned_params):
            s = _list_row(row)
            if s["id"] not in seen_ids:
                seen_ids.add(s["id"])
                sessions.append(s)
    if project_compression_tips and not include_children:
        sessions = _project_compression_tips(store, sessions, compact_rows)
    if heads:
        sessions = _annotate_local_lineages(sessions, heads)
    # last_read_at is lineage-stamped, so root and tip watermarks agree.
    for s in sessions:
        s["unread"] = session_unread(s)
    return sessions


def _project_compression_tips(store: "PostgresStore", sessions: list, compact_rows: bool) -> list:
    """Replace each compression root's surfaced fields with its live tip's (the SQLite
    ``project_compression_tips``; one batched query for the tip rows)."""
    chain_by_root: dict = {}
    conn = store.connection()
    for s in sessions:
        if s.get("end_reason") == "compression":
            chain = _walk_compression_chain(conn, s["id"])
            if chain and chain[-1] != s["id"]:
                chain_by_root[s["id"]] = chain
    tip_rows = (
        _get_session_rich_rows_batch(store, {chain[-1] for chain in chain_by_root.values()},
                                     compact_rows=compact_rows)
        if chain_by_root else {})
    projected = []
    for s in sessions:
        chain = chain_by_root.get(s["id"])
        tip_row = tip_rows.get(chain[-1]) if chain else None
        if not tip_row:
            projected.append(s)
            continue
        merged = dict(s)
        for key in (
            "id", "ended_at", "end_reason", "message_count", "tool_call_count", "title", "last_active",
            "preview", "model", "system_prompt", "cwd", "git_branch", "git_repo_root",
        ):
            if key in tip_row:
                merged[key] = tip_row[key]
        if merged.get("title") is None:
            merged["title"] = s.get("title")
        merged["_lineage_root_id"] = s["id"]
        merged["_lineage_ids"] = chain
        merged["continuation_kind"] = "compression"
        projected.append(merged)
    return projected


def _get_session_rich_rows_batch(
    store: "PostgresStore", session_ids, compact_rows: bool = False,
) -> dict:
    """Enriched rows for many sessions in one query, keyed by id (the SQLite
    ``_get_session_rich_rows_batch``; chunked so a large page stays within one statement)."""
    ids = [sid for sid in session_ids if sid]
    if not ids:
        return {}
    _CHUNK = 900
    if len(ids) > _CHUNK:
        result: dict = {}
        for start in range(0, len(ids), _CHUNK):
            result.update(_get_session_rich_rows_batch(store, ids[start:start + _CHUNK],
                                                      compact_rows=compact_rows))
        return result
    ph = ",".join(" %s" for _ in ids)
    query = (
        f"SELECT {_compact_session_cols() if compact_rows else 's.*'}"
        + ("" if compact_rows else f", {_PROMPT_RESOLVED_SQL}")
        + f", {_preview_raw()}, {_last_active('s')} AS last_active FROM sessions s"
        + ("" if compact_rows else " LEFT JOIN system_prompts sp ON sp.hash = s.system_prompt_hash")
        + f" WHERE s.id IN ({ph})"
    )
    rows = store.all(query, *ids)
    return {s["id"]: _rich_row(s) for s in rows}


def list_recent_sessions_bounded(
    store: "PostgresStore", *, limit: int = 20, exclude_sources: Optional[list] = None,
    timeout_seconds: float = 3.0, candidate_limit: Optional[int] = None,
    lineage_limit: Optional[int] = None,
) -> list:
    """Latency-bounded recent-conversation browse (the SQLite ``list_recent_sessions_bounded``):
    preselect a small candidate set from the indexed durable activity timestamp, resolve only those
    across compression ancestry/chains (bounded: PostgreSQL cannot LIMIT a recursive CTE, so the
    walk is Python-bounded by the same lineage ceiling), then hydrate that bounded set in one query.
    A deadline breach raises ``TimeoutError``."""
    limit = max(1, int(limit))
    timeout_seconds = max(0.0, float(timeout_seconds))
    deadline = time.monotonic() + timeout_seconds
    if candidate_limit is None:
        candidate_limit = max(128, limit * 8)
    candidate_limit = max(limit, min(int(candidate_limit), 2048))
    if lineage_limit is None:
        lineage_limit = min(8192, candidate_limit * 8)
    lineage_limit = max(candidate_limit, min(int(lineage_limit), 8192))

    conn = store.connection()
    # A compression continuation is an implementation edge, unlike /new reset and /branch children
    # which are independent user-visible conversations (the same predicate in both directions).
    compression_edge = (
        "parent.end_reason = 'compression'"
        " AND child.parent_session_id = parent.id"
        f" AND {_json_key_text('child.model_config', '_branched_from')} IS NULL"
        f" AND {_json_key_text('child.model_config', '_delegate_from')} IS NULL"
        " AND COALESCE(child.source, '') != 'tool'")

    candidate_clauses = ["archived = 0", "hidden = 0",
                         f"{_json_key_text('model_config', '_delegate_from')} IS NULL"]
    candidate_params: list = []
    if exclude_sources:
        candidate_clauses.append(f"source NOT IN ({','.join(' %s' for _ in exclude_sources)})")
        candidate_params.extend(exclude_sources)
    candidate_where = " AND ".join(candidate_clauses)
    candidates = [row["id"] for row in conn.execute(
        f"SELECT id FROM sessions WHERE {candidate_where} "
        "ORDER BY COALESCE(last_activity_at, started_at) DESC, started_at DESC, id DESC LIMIT %s",
        [*candidate_params, candidate_limit]).fetchall()]
    if not candidates or time.monotonic() >= deadline:
        raise TimeoutError(f"recent-session browse exceeded {timeout_seconds:g}s deadline")

    def compression_parents(ids: list) -> list:
        """The compression parent of each *ids* member (upward edge, one bounded query)."""
        return [row["id"] for row in conn.execute(
            "SELECT parent.id FROM sessions child JOIN sessions parent ON parent.id = child.parent_session_id"
            f" WHERE child.id = ANY(%s) AND {compression_edge}", ids).fetchall()]

    # Roots: each candidate's logical root, walking compression parents up. The lineage ceiling is a
    # total-row bound like the SQLite recursive-CTE LIMIT: the seed row counts, so at most
    # lineage_limit - 1 upward steps are taken; a lineage that reaches the ceiling before a terminal
    # root is omitted, not expanded (the SQLite candidate_roots NOT EXISTS, mirrored).
    def _parent_compression_edge(cur: str) -> Optional[str]:
        """The *cur* row's compression parent id, or None when the parent edge is not a continuation."""
        row = conn.execute(
            "SELECT c.parent_session_id FROM sessions c WHERE c.id = %s", (cur,)).fetchone()
        pid = row["parent_session_id"] if row else None
        if not pid:
            return None
        edge = conn.execute(
            "SELECT 1 FROM sessions child JOIN sessions parent ON parent.id = child.parent_session_id"
            f" WHERE child.id = %s AND {compression_edge}", (cur,)).fetchone()
        return pid if edge is not None else None

    def root_of(sid: str) -> Optional[str]:
        cur = sid
        for _ in range(max(0, lineage_limit - 1)):
            pid = _parent_compression_edge(cur)
            if pid is None:
                return cur
            cur = pid
        # At the ceiling: the lineage is admitted only when *cur* is already a terminal root.
        return cur if _parent_compression_edge(cur) is None else None

    roots = list(dict.fromkeys(r for r in (root_of(sid) for sid in candidates) if r is not None))

    # Forward chains from every root, collecting the live tips.
    tips_by_root: dict = {}
    for root in roots:
        chain = _walk_compression_chain(conn, root)
        tips_by_root[root] = chain[-1] if chain else root
    if time.monotonic() >= deadline:
        raise TimeoutError(f"recent-session browse exceeded {timeout_seconds:g}s deadline")

    sessions = []
    for root in roots:
        tip_id = tips_by_root[root]
        row = conn.execute(
            "SELECT tip.id, tip.source, tip.model, COALESCE(tip.title, s.title) AS title, "
            " s.started_at AS started_at, tip.ended_at, tip.end_reason, "
            " tip.message_count, tip.tool_call_count, "
            f" {_last_active_by_id('tip.id')} AS last_active, {_preview_raw('tip.id')},"
            " CASE WHEN s.id <> tip.id THEN s.id ELSE NULL END AS _lineage_root_id"
            f" FROM sessions s JOIN sessions tip ON tip.id = %s WHERE s.id = %s AND ({_LISTABLE_CHILD_SQL})",
            (tip_id, root)).fetchone()
        if row is None:
            continue
        s = dict(row)
        s["id"] = tip_id
        s["preview"] = _shape_preview(s.pop("_preview_raw", ""))
        s["unread"] = session_unread(s)
        sessions.append(s)
    sessions.sort(key=lambda s: (-(s["last_active"] or 0.0), -(s["started_at"] or 0.0), s["id"]))
    return sessions[:limit]


def search_sessions(
    store: "PostgresStore", source: Optional[Any] = None, limit: int = 20, offset: int = 0,
    workspace_key: Optional[str] = None, local_created_sources: Optional[Sequence[str]] = None,
    skip_launch_stubs: bool = False,
) -> list:
    """Sessions MRU-first with a computed ``last_active`` (the SQLite ``search_sessions``);
    ``workspace_key`` scopes to one workspace, ``skip_launch_stubs`` drops empty launch stubs."""
    where_clauses = [_NOT_A_LAUNCH_STUB_SQL] if skip_launch_stubs else []
    params: list = []
    if source:
        sources = [source] if isinstance(source, str) else list(source)
        clause, bound = _source_match_clause(sources, local_created_sources)
        where_clauses.append(clause)
        params.extend(bound)
    if workspace_key:
        ws_clause, ws_params = _workspace_key_clause(workspace_key)
        where_clauses.append(ws_clause)
        params.extend(ws_params)
    return [dict(row) for row in store.all(
        "SELECT s.*, COALESCE(sp.prompt, s.system_prompt) AS _system_prompt_resolved, "
        f"{_last_active('s')} AS last_active "
        "FROM sessions s LEFT JOIN system_prompts sp ON sp.hash = s.system_prompt_hash "
        f"{_where_sql(where_clauses, ' ')} "
        "ORDER BY last_active DESC, s.started_at DESC, s.id DESC LIMIT %s OFFSET %s",
        *params, limit, offset)]


def session_count(
    store: "PostgresStore", source: Optional[str] = None, sources: Optional[list] = None,
    cwd_prefix: Optional[str] = None, min_message_count: int = 0, include_archived: bool = False,
    archived_only: bool = False, exclude_children: bool = False, exclude_sources: Optional[list] = None,
    include_subagents: bool = False, local_created_sources: Optional[Sequence[str]] = None,
) -> int:
    """Count sessions with list_sessions_rich's filters (the SQLite ``session_count``)."""
    where_clauses, params = _session_filter_where(
        exclude_children=exclude_children, source=source, sources=sources,
        exclude_sources=exclude_sources, cwd_prefix=cwd_prefix, min_message_count=min_message_count,
        archived_only=archived_only, include_archived=include_archived, include_subagents=include_subagents,
        local_created_sources=local_created_sources,
    )
    if exclude_children:
        _exclude_superseded_segments(store.connection(), where_clauses, params)
    return int(store.scalar(
        f"SELECT COUNT(*) FROM sessions s{_where_sql(where_clauses, ' ')}", *params) or 0)


def session_count_ge(store: "PostgresStore", n: int = 1) -> bool:
    """At least N sessions exist (archived included; the SQLite ``session_count_ge``)."""
    return len(store.all("SELECT 1 FROM sessions LIMIT %s", n)) >= n


def session_count_by_source(
    store: "PostgresStore", *, include_archived: bool = False, archived_only: bool = False,
    exclude_children: bool = False,
) -> dict:
    """``{source: count}`` via one GROUP BY (the SQLite ``session_count_by_source``)."""
    where_clauses, params = _session_filter_where(
        exclude_children=exclude_children, archived_only=archived_only,
        include_archived=include_archived,
    )
    if exclude_children:
        _exclude_superseded_segments(store.connection(), where_clauses, params)
    rows = store.all(
        "SELECT COALESCE(NULLIF(s.source, ''), 'cli') AS source, COUNT(*) AS count "
        f"FROM sessions s{_where_sql(where_clauses, ' ')} "
        "GROUP BY COALESCE(NULLIF(s.source, ''), 'cli') ORDER BY count DESC", *params)
    return {str(row["source"]): int(row["count"] or 0) for row in rows}


def _is_explicit_fork_child_row(row: Mapping, *, include_reset: bool = False) -> bool:
    """True when *row* is a branch, delegate, or tool child of its parent (the SQLite
    ``_is_explicit_fork_child_row``; markers only count when they point at parent_session_id)."""
    if row.get("source") == "tool":
        return True
    cfg = row.get("model_config")
    if isinstance(cfg, str):
        cfg = _parse_model_config(cfg)
    if not isinstance(cfg, dict):
        return False
    markers = (cfg.get("_branched_from"), cfg.get("_delegate_from"))
    if include_reset:
        markers += (cfg.get("_reset_from"),)
    parent_id = row.get("parent_session_id")
    return parent_id in markers if parent_id else any(m is not None for m in markers)


def declared_scope_identity(store: "PostgresStore", session_id: str) -> tuple:
    """(is_fork_child, source) in ONE read (the SQLite ``declared_scope_identity``)."""
    row = store.one("SELECT source, parent_session_id, model_config FROM sessions WHERE id = %s",
                    session_id)
    if row is None:
        return False, ""
    return _is_explicit_fork_child_row(dict(row)), str(row.get("source") or "").strip()


def is_kanban_owned_session(store: "PostgresStore", session_id: str) -> bool:
    """True when the verified compression lineage a resume would materialize is Kanban-owned (the
    SQLite ``is_kanban_owned_session``): any segment with ``source``/``created_source`` = 'kanban'."""
    conn = store.connection()
    lineage = [session_id]
    seen = set(lineage)
    current = session_id
    for _ in range(_CHAIN_CAP):
        row = conn.execute("SELECT parent_session_id, end_reason FROM sessions WHERE id = %s",
                           (current,)).fetchone()
        if row is None or not row["parent_session_id"] or row["parent_session_id"] in seen:
            break
        if row["end_reason"] != "compression":
            break
        seen.add(row["parent_session_id"])
        lineage.append(row["parent_session_id"])
        current = row["parent_session_id"]
    if not lineage:
        return False
    ph = ",".join(" %s" for _ in lineage)
    row = conn.execute(
        f"SELECT 1 FROM sessions WHERE id IN ({ph}) AND (source = 'kanban' OR created_source = 'kanban') "
        "LIMIT 1", lineage).fetchone()
    return row is not None


# ── the display fence read (the SQLite messages display projection, ported) ──

_DISPLAY_ACTIVE_CLAUSE = " AND (active = 1 OR compacted = 1)"
_DISPLAY_VISIBLE_SQL = (
    f" AND COALESCE({_json_key('display_metadata', MODEL_ONLY_DISPLAY_METADATA_KEY)}, 0) = 0")
_NOT_A_LATER_TOOL_COPY_SQL = (
    " AND NOT (messages.role = 'tool' AND messages.message_uid IS NOT NULL AND EXISTS ("
    "SELECT 1 FROM messages AS original WHERE original.session_id = messages.session_id"
    " AND original.timestamp = messages.timestamp AND original.message_uid = messages.message_uid"
    " AND original.role = 'tool' AND original.active = 0 AND original.compacted = 1"
    " AND original.display_order < messages.display_order"
    f" {_DISPLAY_VISIBLE_SQL.replace('display_metadata', 'original.display_metadata')}))")


def _display_rows_from_conn(conn, session_id: str):
    """One display-history projection (the SQLite ``_display_rows_from_conn``, unbounded)."""
    return conn.execute(
        """WITH page AS (
                   SELECT display_order FROM messages
                   WHERE session_id = %s AND (active = 1 OR compacted = 1)
                   """ + _DISPLAY_VISIBLE_SQL.replace("display_metadata", "messages.display_metadata") + _NOT_A_LATER_TOOL_COPY_SQL + """
                   GROUP BY display_order ORDER BY display_order
               )
               SELECT chosen.* FROM page
               JOIN messages AS chosen ON chosen.id = (
                   SELECT candidate.id FROM messages AS candidate
                   WHERE candidate.session_id = %s
                     AND candidate.display_order = page.display_order
                     AND (candidate.active = 1 OR candidate.compacted = 1)
                     """ + _DISPLAY_VISIBLE_SQL.replace("display_metadata", "candidate.display_metadata") + """
                     AND NOT (candidate.role = 'tool' AND candidate.message_uid IS NOT NULL AND EXISTS (
                         SELECT 1 FROM messages AS original WHERE original.session_id = candidate.session_id
                         AND original.timestamp = candidate.timestamp AND original.message_uid = candidate.message_uid
                         AND original.role = 'tool' AND original.active = 0 AND original.compacted = 1
                         AND original.display_order < candidate.display_order
                         """ + _DISPLAY_VISIBLE_SQL.replace("display_metadata", "original.display_metadata") + """
                     ))
                   ORDER BY candidate.active DESC, candidate.id DESC LIMIT 1
               )
               ORDER BY page.display_order""",
        (session_id, session_id),
    ).fetchall()


def _display_messages_from_conn(conn, session_id: str) -> Optional[list]:
    """Exact display snapshot on the held transaction; None fails the fence closed."""
    if conn.execute("SELECT 1 FROM sessions WHERE id = %s LIMIT 1", (session_id,)).fetchone() is None:
        return None
    if conn.execute(
        "SELECT 1 FROM messages WHERE session_id = %s AND (active = 1 OR compacted = 1)"
        " AND (display_order IS NULL OR display_identity IS NULL) LIMIT 1", (session_id,)).fetchone():
        return None
    from .messages import _row_to_message_dict
    return [
        _row_to_message_dict(row, summary_flag=True)
        for row in _display_rows_from_conn(conn, session_id)
    ]


# ── the delete pipeline (the SQLite delete store, ported) ──

def _collect_delegate_child_ids(conn, parent_ids: list) -> list:
    """Delegate-subagent ids (``_delegate_from`` marker, walked recursively) to cascade-delete with
    *parent_ids*; untagged children stay orphaned, not deleted (the SQLite ``_collect_delegate_child_ids``)."""
    df = _json_key_text("model_config", "_delegate_from")
    found = {sid for sid in parent_ids if sid}
    frontier = list(found)
    while frontier:
        next_frontier = []
        rows = conn.execute(
            f"SELECT id, parent_session_id FROM sessions WHERE {df} = ANY(%s) "
            f"OR (parent_session_id = ANY(%s) AND ({df} IS NOT NULL))",
            (frontier, frontier)).fetchall()
        for row in rows:
            if row["id"] not in found:
                found.add(row["id"])
                next_frontier.append(row["id"])
        frontier = next_frontier
    return [sid for sid in found if sid not in set(parent_ids)]


def _delete_delegate_children(conn, parent_ids: list, *, media_root: str) -> list:
    """Cascade-delete the delegate children of *parent_ids* (the SQLite ``_delete_delegate_children``)."""
    ids = _collect_delegate_child_ids(conn, parent_ids)
    if ids:
        _retire_sessions(conn, ids, media_root=media_root)
        for sid in ids:
            conn.execute("DELETE FROM messages WHERE session_id = %s", (sid,))
            # FK safety: orphan any untagged stragglers pointing at a doomed row.
            conn.execute("UPDATE sessions SET parent_session_id = NULL WHERE parent_session_id = %s", (sid,))
            conn.execute("DELETE FROM sessions WHERE id = %s", (sid,))
    return ids


def _expand_compression_lineage(conn, session_ids: list) -> list:
    """Expand *session_ids* to every member of their compression chains in BOTH directions (the SQLite
    ``_expand_compression_lineage``; branch/reset/delegate children are deliberately not pulled in), plus
    each member's local-conversation scope: a local reset lineage is one listed conversation."""
    seen = set(session_ids)
    frontier = list(session_ids)
    for _ in range(_CHAIN_CAP):
        if not frontier:
            break
        up = [row["parent_session_id"] for row in conn.execute(
            "SELECT parent_session_id FROM sessions WHERE id = ANY(%s) AND end_reason = 'compression'",
            (frontier,)).fetchall() if row["parent_session_id"]]
        down = [row["id"] for row in conn.execute(
            f"SELECT child.id FROM sessions parent JOIN sessions child ON child.parent_session_id = parent.id "
            "WHERE parent.id = ANY(%s) AND parent.end_reason = 'compression' "
            f"AND {_json_key_text('child.model_config', '_branched_from')} IS NULL "
            f"AND {_json_key_text('child.model_config', '_delegate_from')} IS NULL "
            f"AND NOT ({_RESET_CHILD_SQL.format(a='child')}) AND COALESCE(child.source, '') <> 'tool'",
            (frontier,)).fetchall()]
        frontier = [sid for sid in [*up, *down] if sid not in seen]
        seen.update(frontier)
    for sid in list(seen):
        seen.update(_local_conversation_ids(conn, sid))
    return list(seen)


def _lease_stale(row, now: float) -> bool:
    """A lock/lease row whose holder is expired or a provably dead local process (the SQLite
    ``_stale_holder``)."""
    return float(row["expires_at"]) <= now or holder_process_is_dead(row["holder"])


def _session_turn_lease_key_on_conn(conn, session_id: str) -> str:
    """The conversation's turn-lease key (the SQLite ``_session_turn_lease_key_on_conn`` shape): the
    session's compression root, since the lease guards the whole conversation, never one segment."""
    if not session_id:
        return session_id
    current = session_id
    for _ in range(_CHAIN_CAP):
        row = conn.execute(
            "SELECT parent_session_id FROM sessions WHERE id = %s", (current,)).fetchone()
        parent = row["parent_session_id"] if row else None
        if not parent:
            break
        step = conn.execute(
            "SELECT 1 FROM sessions child JOIN sessions parent ON parent.id = child.parent_session_id "
            "WHERE child.id = %s AND parent.end_reason = 'compression' "
            f"AND {_json_key_text('child.model_config', '_branched_from')} IS NULL "
            f"AND {_json_key_text('child.model_config', '_delegate_from')} IS NULL "
            f"AND NOT ({_RESET_CHILD_SQL.format(a='child')}) AND COALESCE(child.source, '') <> 'tool'",
            (current,)).fetchone()
        if step is None:
            break
        current = parent
    return current


def _guarded_ids(conn, ids) -> list:
    """Ids in *ids* protected by a live turn lease / compression lock (the SQLite ``_guarded_ids``;
    expired or dead-holder guards are reclaimed as a side effect)."""
    guard_pool = [sid for sid in (ids or []) if sid]
    if not guard_pool:
        return []
    now = time.time()
    locked = set()
    lock_rows = conn.execute(
        f"SELECT session_id, holder, expires_at FROM compression_locks WHERE session_id = ANY(%s)",
        (guard_pool,)).fetchall()
    for row in lock_rows:
        if _lease_stale(row, now):
            conn.execute("DELETE FROM compression_locks WHERE session_id = %s", (row["session_id"],))
        else:
            locked.add(row["session_id"])
    active = [sid for sid in guard_pool if sid in locked]
    # The lease guards the conversation, so key by each id's compression root.
    keys = list(dict.fromkeys(_session_turn_lease_key_on_conn(conn, sid) for sid in active))
    lease_rows = conn.execute(
        f"SELECT conversation_id, holder, expires_at FROM session_turn_leases WHERE conversation_id = ANY(%s)",
        (keys,)).fetchall()
    leased: set = set()
    for row in lease_rows:
        if _lease_stale(row, now):
            conn.execute("DELETE FROM session_turn_leases WHERE conversation_id = %s", (row["conversation_id"],))
        else:
            leased.add(row["conversation_id"])
    return [sid for sid in active if _session_turn_lease_key_on_conn(conn, sid) in leased]


def get_session_delete_targets(store: "PostgresStore", session_id: str) -> list:
    """Rows :func:`delete_session` would remove: the session, its local-conversation scope and its
    recursive delegate children (the SQLite ``get_session_delete_targets``)."""
    conn = store.connection()
    if store.one("SELECT 1 FROM sessions WHERE id = %s", session_id) is None:
        return []
    scope = _owned_lineage_ids(conn, session_id)
    delegate_ids = _collect_delegate_child_ids(conn, scope)
    return [session_id, *sorted(set(scope) - {session_id}), *sorted(delegate_ids)]


def delete_session(
    store: "PostgresStore", session_id: str,
    expected_delete_ids: Optional[list] = None,
    expected_display_messages: Optional[dict] = None,
    exclude_active_write_guards: bool = False,
    include_compression_chain: bool = False, *, media_root: str,
) -> tuple:
    """Delete a session and its messages; delegate children cascade, branch/compression children are
    orphaned (the SQLite ``delete_session``; expected ids / display snapshots fence drift in-transaction,
    ``exclude_active_write_guards`` refuses live turn-lease or compression-lock guards)."""
    if include_compression_chain:
        return delete_sessions(
            store, [session_id], exclude_active_write_guards=exclude_active_write_guards,
            include_compression_chain=True, media_root=media_root)[1]
    removed_ids: list = []
    expected_ids = set(expected_delete_ids) if expected_delete_ids is not None else None
    with store.tx() as conn:
        if conn.execute("SELECT 1 FROM sessions WHERE id = %s", (session_id,)).fetchone() is None:
            return tuple(removed_ids)
        scope = _owned_lineage_ids(conn, session_id)
        target_ids = (
            [*scope, *_collect_delegate_child_ids(conn, scope)]
            if exclude_active_write_guards or expected_ids is not None else None)
        if exclude_active_write_guards and _guarded_ids(conn, target_ids):
            raise SessionActiveWriteGuardError(
                f"session '{session_id}' (or a delegate child) has an active turn lease or compression lock")
        if expected_ids is not None and expected_ids != set(target_ids):
            return tuple(removed_ids)
        if expected_display_messages is not None and any(
                _display_messages_from_conn(conn, covered_id) != expected
                for covered_id, expected in expected_display_messages.items()):
            return tuple(removed_ids)
        _retire_sessions(conn, scope, media_root=media_root)
        delegate_ids = _delete_delegate_children(conn, scope, media_root=media_root)
        for sid in scope:  # orphan remaining children (branches) so FK is satisfied
            conn.execute("UPDATE sessions SET parent_session_id = NULL WHERE parent_session_id = %s", (sid,))
            conn.execute("DELETE FROM messages WHERE session_id = %s", (sid,))
        for sid in scope:
            conn.execute("DELETE FROM sessions WHERE id = %s", (sid,))
        _delete_unreferenced_system_prompts(conn)
        removed_ids.extend([session_id, *sorted(set(scope) - {session_id}), *sorted(delegate_ids)])
    return tuple(removed_ids)


def discard_unadmitted_session(store: "PostgresStore", session_id: str, *, media_root: str) -> tuple:
    """Abort a row this same flow just created and nobody has been admitted against (the SQLite
    ``discard_unadmitted_session``); a row with any admission or worker receipt takes the fenced
    :func:`delete_session` path instead."""
    def _do(conn):
        row = conn.execute(
            "SELECT 1 FROM session_admissions WHERE target_session_id = %s UNION ALL "
            "SELECT 1 FROM worker_executions WHERE session_id = %s LIMIT 1", (session_id, session_id)).fetchone()
        if row is not None:
            return None
        conn.execute("DELETE FROM messages WHERE session_id = %s", (session_id,))
        cur = conn.execute("DELETE FROM sessions WHERE id = %s", (session_id,))
        return (session_id,) if cur.rowcount > 0 else ()
    with store.tx() as conn:
        result = _do(conn)
    if result is None:
        return delete_session(store, session_id, media_root=media_root)
    return result


def delete_session_if_empty(store: "PostgresStore", session_id: str, *, media_root: str) -> tuple:
    """Delete *session_id* only if it has no messages, no title and no children; check and delete share
    one transaction (the SQLite ``delete_session_if_empty``; a guarded row is simply not deleted)."""
    def _do(conn):
        if _guarded_ids(conn, [session_id]) or _owned_lineage_ids(conn, session_id) != [session_id]:
            return False
        eligible = conn.execute(
            "SELECT 1 FROM sessions WHERE id = %s AND title IS NULL "
            " AND NOT EXISTS (SELECT 1 FROM messages WHERE messages.session_id = sessions.id) "
            " AND NOT EXISTS (SELECT 1 FROM sessions child WHERE child.parent_session_id = sessions.id)",
            (session_id,)).fetchone()
        if eligible is None:
            return False
        _retire_sessions(conn, [session_id], expiring=[session_id], media_root=media_root)
        conn.execute("DELETE FROM sessions WHERE id = %s", (session_id,))
        _delete_unreferenced_system_prompts(conn)
        return True
    with store.tx() as conn:
        did = _do(conn)
    return (session_id,) if did else ()


def count_empty_sessions(store: "PostgresStore") -> int:
    """Count of empty, ended, non-archived sessions; ended_at guards a fresh session's first message."""
    return int(store.scalar(f"SELECT COUNT(*) FROM sessions WHERE {_EMPTY_SESSION_WHERE}") or 0)


def delete_empty_sessions(store: "PostgresStore", *, media_root: str) -> tuple:
    """Delete every empty, ended, non-archived session in one transaction, orphaning children (the
    SQLite ``delete_empty_sessions``)."""
    removed_ids: list = []
    with store.tx() as conn:
        ids = {row["id"] for row in conn.execute(
            f"SELECT id FROM sessions WHERE {_EMPTY_SESSION_WHERE}").fetchall()}
        ids = set(_retire_prunable(conn, sorted(ids), media_root=media_root))
        if not ids:
            return tuple(removed_ids)
        for sid in sorted(ids):
            conn.execute("UPDATE sessions SET parent_session_id = NULL WHERE parent_session_id = %s", (sid,))
            conn.execute("DELETE FROM messages WHERE session_id = %s", (sid,))
            conn.execute("DELETE FROM sessions WHERE id = %s", (sid,))
            removed_ids.append(sid)
        _delete_unreferenced_system_prompts(conn)
    return tuple(removed_ids)


def delete_sessions(
    store: "PostgresStore", session_ids: list,
    exclude_active_write_guards: bool = False, skipped_ids: Optional[list] = None,
    include_compression_chain: bool = False, *, media_root: str,
) -> tuple:
    """Bulk delete with :func:`delete_session` semantics per row, in ONE transaction (the SQLite
    ``delete_sessions``; the returned count reflects the *requested* ids, not expanded links)."""
    unique_ids = list({sid for sid in (session_ids or []) if isinstance(sid, str) and sid})
    if not unique_ids:
        return 0, ()
    removed_ids: list = []
    with store.tx() as conn:
        existing = [row["id"] for row in conn.execute(
            f"SELECT id FROM sessions WHERE id = ANY(%s)", (unique_ids,)).fetchall()]
        if not existing:
            return 0, tuple(removed_ids)

        def expand(ids: list) -> list:
            # A local reset/compression segment deletes as its whole conversation (delete_session).
            if include_compression_chain:
                return _expand_compression_lineage(conn, ids)
            out: list = []
            for sid in ids:
                for member in _owned_lineage_ids(conn, sid):
                    if member not in out:
                        out.append(member)
            return out

        expanded = expand(existing)
        guard_pool = [*expanded, *_collect_delegate_child_ids(conn, expanded)]
        if exclude_active_write_guards:
            active_ids: set = set()
            if _guarded_ids(conn, guard_pool):
                if include_compression_chain:
                    for sid in existing:
                        member_lineage = set(_expand_compression_lineage(conn, [sid]))
                        member_lineage.add(sid)
                        if _guarded_ids(conn, [*member_lineage,
                                *_collect_delegate_child_ids(conn, list(member_lineage))]):
                            active_ids.add(sid)
                else:
                    active_ids = {
                        sid for sid in existing
                        if _guarded_ids(conn, [*(members := expand([sid])),
                                              *_collect_delegate_child_ids(conn, members)])
                    }
            existing = [sid for sid in existing if sid not in active_ids]
            if skipped_ids is not None:
                skipped_ids.extend(sorted(active_ids))
            if not existing:
                return 0, tuple(removed_ids)
            expanded = expand(existing)
        _retire_sessions(conn, expanded, media_root=media_root)
        delegate_ids = _delete_delegate_children(conn, expanded, media_root=media_root)
        for sid in expanded:  # orphan children whose parent is in the kill list (FK)
            conn.execute("UPDATE sessions SET parent_session_id = NULL WHERE parent_session_id = %s", (sid,))
            conn.execute("DELETE FROM messages WHERE session_id = %s", (sid,))
            conn.execute("DELETE FROM sessions WHERE id = %s", (sid,))
        _delete_unreferenced_system_prompts(conn)
        removed_ids.extend([*expanded, *delegate_ids])
        count = len(existing)
    return count, tuple(removed_ids)


# ── the title store (the SQLite titles store, ported) ──

def set_session_title(store: "PostgresStore", session_id: str, title: Optional[str]) -> bool:
    """Set a title on the user's behalf (``user`` provenance); empty clears it. Raises ValueError on
    conflict or validation failure (the SQLite ``set_session_title``)."""
    with store.tx() as conn:
        return bool(_set_session_title_in_transaction(conn, session_id, title, source=TITLE_SOURCE_USER))


def set_auto_title(store: "PostgresStore", session_id: str, title: str, *, source: str) -> bool:
    """Set an automatic title; False (untouched) when a higher-authority title holds the row."""
    if source not in (TITLE_SOURCE_DERIVED, TITLE_SOURCE_LLM):
        raise ValueError(f"invalid automatic title source: {source!r}")
    with store.tx() as conn:
        return bool(_set_session_title_in_transaction(conn, session_id, title, source=source))


def get_session_title(store: "PostgresStore", session_id: str) -> Optional[str]:
    row = store.one("SELECT title FROM sessions WHERE id = %s", session_id)
    return row["title"] if row else None


def get_session_title_source(store: "PostgresStore", session_id: str) -> Optional[str]:
    row = store.one("SELECT title, title_source FROM sessions WHERE id = %s", session_id)
    return row["title_source"] if row and row["title"] is not None else None


def set_session_title_source(store: "PostgresStore", session_id: str, source: str) -> bool:
    """Overwrite a title's provenance without touching the text."""
    if source not in _TITLE_SOURCE_RANK:
        raise ValueError(f"invalid title source: {source!r}")
    with store.tx() as conn:
        cur = conn.execute(
            "UPDATE sessions SET title_source = %s WHERE id = %s AND title IS NOT NULL",
            (source, session_id))
        return cur.rowcount > 0


def get_session_by_title(store: "PostgresStore", title: str) -> Optional[dict]:
    """Look up a session by exact title; a deliberately archived canonical "Bot Chat" is a retired name:
    skipped, so Bot Mode mints a fresh one (the SQLite ``get_session_by_title``)."""
    row = store.one(
        "SELECT s.*, " + _PROMPT_RESOLVED_SQL + " FROM sessions s "
        "LEFT JOIN system_prompts sp ON sp.hash = s.system_prompt_hash WHERE s.title = %s", title)
    if row is None:
        return None
    session = _list_row(row)
    if title != CANONICAL_BOT_CHAT_TITLE or not session.get("archived"):
        return session
    tip_id = _get_compression_tip(store, session["id"]) or session["id"]
    if tip_id != session["id"]:
        tip = get_session_row(store, tip_id) or session
    else:
        tip = session
    released = _bot_chat_title_released(session.get("archived"), session.get("auto_archived"), tip.get("end_reason"))
    return None if released else session


def resolve_session_by_title(store: "PostgresStore", title: str) -> Optional[str]:
    """Resolve a title to a session ID, preferring the latest "title #N" continuation (the SQLite
    ``resolve_session_by_title``; the canonical Bot Chat never falls through to a numbered row)."""
    exact = get_session_by_title(store, title)
    if title == CANONICAL_BOT_CHAT_TITLE:
        return exact["id"] if exact is not None else None
    rows = store.all(
        "SELECT id, title, started_at FROM sessions WHERE title LIKE %s ESCAPE '\\' "
        "ORDER BY started_at DESC", f"{_escape_like(title)} #%")
    return rows[0]["id"] if rows else (exact["id"] if exact else None)


def next_title_in_lineage(store: "PostgresStore", base_title: str) -> str:
    """Next title in a lineage ("my session" -> "my session #2") as seen by this store (the SQLite
    ``next_title_in_lineage``; the unnumbered original counts as #1)."""
    match = _NUMBERED_TITLE_RE.match(base_title)
    base = match.group(1) if match else base_title
    rows = store.all(
        "SELECT title FROM sessions WHERE title = %s OR title LIKE %s ESCAPE '\\'",
        base, f"{_escape_like(base)} #%")
    if not rows:
        return base
    numbers = [int(m.group(2)) for m in (_NUMBERED_TITLE_RE.match(row["title"]) for row in rows) if m]
    return f"{base} #{max([1, *numbers]) + 1}"


def retitle_canonical_bot_chat(store: "PostgresStore", session_id: str, title: str) -> bool:
    """Rename a hidden canonical "Bot Chat" row (the SQLite ``retitle_canonical_bot_chat``): the UPDATE
    is conditional on the title still being the canonical name."""
    with store.tx() as conn:
        cur = conn.execute(
            "UPDATE sessions SET title = %s, title_source = %s, hidden = 0 WHERE id = %s AND title = %s",
            (title, TITLE_SOURCE_USER, session_id, CANONICAL_BOT_CHAT_TITLE))
        return cur.rowcount > 0


# ── the model-config value update (the SQLite model_config store, ported) ──

def update_conversation_model_config_value(
    store: "PostgresStore", session_id: str, key: str, update,
) -> Optional[str]:
    """Read a conversation-wide ``key`` at the session's compression tip and write ``update(value)``
    back in one write transaction (the SQLite ``update_conversation_model_config_value``; returning
    None removes the key; None when the row does not exist)."""
    with store.tx() as conn:
        tip = _compression_tip_on(conn, session_id)
        row = conn.execute("SELECT model_config FROM sessions WHERE id = %s", (tip,)).fetchone()
        if row is None:
            return None
        config = _parse_model_config(row["model_config"])
        value = update(config.get(key))
        if value is None:
            config.pop(key, None)
        else:
            config[key] = value
        conn.execute("UPDATE sessions SET model_config = %s WHERE id = %s", (json.dumps(config), tip))
        return tip


def session_ids_with_model_config_key(store: "PostgresStore", key: str) -> list:
    """Ids of the sessions whose model_config carries ``key``, newest first."""
    rows = store.all(
        f"SELECT id FROM sessions WHERE {_json_key_text('model_config', key)} IS NOT NULL "
        "ORDER BY started_at DESC")
    return [row["id"] for row in rows]


# ── the prompt-refresh epochs (the SQLite prompt_refresh store, ported) ──

def request_prompt_refresh(store: "PostgresStore", session_id: Optional[str] = None) -> int:
    """Record a refresh request for every session (or just *session_id*); returns the epoch."""
    key = _PROMPT_REFRESH_PROFILE_KEY if session_id is None else _PROMPT_REFRESH_SESSION_PREFIX + session_id
    with store.tx() as conn:
        row = conn.execute(
            "SELECT MAX(CAST(value AS INTEGER)) AS max_epoch FROM state_meta "
            "WHERE key = %s OR left(key, %s) = %s",
            (_PROMPT_REFRESH_PROFILE_KEY, len(_PROMPT_REFRESH_SESSION_PREFIX),
             _PROMPT_REFRESH_SESSION_PREFIX)).fetchone()
        epoch = int(row["max_epoch"] or 0) + 1
        conn.execute(
            "INSERT INTO state_meta (key, value) VALUES (%s, %s) "
            "ON CONFLICT (key) DO UPDATE SET value = excluded.value",
            (key, str(epoch)))
        return epoch


def prompt_refresh_epoch(store: "PostgresStore", session_id: str) -> int:
    """The newest refresh epoch that applies to *session_id* (0 = never requested)."""
    value = store.scalar(
        "SELECT MAX(CAST(value AS INTEGER)) FROM state_meta WHERE key IN (%s, %s)",
        _PROMPT_REFRESH_PROFILE_KEY, _PROMPT_REFRESH_SESSION_PREFIX + session_id)
    return int(value or 0)


def count_sessions_with_stored_prompt(store: "PostgresStore") -> int:
    """Sessions whose next turn would replay a stored prompt (the ones a refresh rebuilds)."""
    return int(store.scalar(
        "SELECT COUNT(*) FROM sessions WHERE system_prompt_hash IS NOT NULL OR system_prompt IS NOT NULL") or 0)


# ── the kanban worker retag (the SQLite core store, ported) ──

def retag_kanban_worker_sessions(store: "PostgresStore", workspaces_root: str) -> int:
    """Retag legacy kanban worker rows from ``cli`` to ``kanban`` by cwd under the board's workspaces
    root; gated once per root via state_meta (the SQLite ``retag_kanban_worker_sessions``)."""
    prefix = str(workspaces_root).rstrip("/\\")
    if not prefix:
        return 0
    gate = f"kanban_worker_source_retagged:{prefix}"
    row = store.one("SELECT value FROM state_meta WHERE key = %s", gate)
    if row is not None and row["value"] == "1":
        return 0
    with store.tx() as conn:
        esc = _escape_like(prefix)
        cur = conn.execute(
            "UPDATE sessions SET source = 'kanban' WHERE source = 'cli' "
            "AND (cwd = %s OR cwd LIKE %s ESCAPE '\\' OR cwd LIKE %s ESCAPE '\\')",
            (prefix, f"{esc}/%", f"{esc}\\%"))
        retagged = cur.rowcount or 0
        # rowcount BEFORE the gate upsert (the SQLite set_meta is an upsert: a pre-existing gate
        # holding another value is written "1" here, not left alone by DO NOTHING).
        conn.execute(
            "INSERT INTO state_meta (key, value) VALUES (%s, '1') ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value",
            (gate,))
        return retagged


# ── the gateway listing (the SQLite gateway store, ported) ──

def list_gateway_sessions(
    store: "PostgresStore", *, platform: Optional[str] = None, active_only: bool = True,
) -> list:
    """List gateway sessions (rows with a session_key): newest row per key, one live mapping per
    routing key (the SQLite ``list_gateway_sessions``; ``platform`` filters on ``source``)."""
    query = (
        "SELECT s.*, " + _PROMPT_RESOLVED_SQL + f", {_last_active('s')} AS last_active "
        "FROM sessions s LEFT JOIN system_prompts sp ON sp.hash = s.system_prompt_hash "
        "WHERE s.session_key IS NOT NULL "
        " AND s.started_at = (SELECT MAX(s2.started_at) FROM sessions s2 WHERE s2.session_key = s.session_key) ")
    params: list = []
    if platform:
        query += " AND LOWER(s.source) = LOWER(%s)"
        params.append(platform)
    if active_only:
        query += " AND s.ended_at IS NULL"
    query += " ORDER BY last_active DESC"
    return [_list_row(row) for row in store.all(query, *params)]


# ── the stored-read facts (the SQLite stored_reads store, ported) ──

def sessions_in_workspace(store: "PostgresStore", session_ids, workspace: str) -> set:
    """The ids among *session_ids* whose row is in *workspace* (the SQLite ``sessions_in_workspace``:
    the same repo root, or at/under the folder)."""
    ids = [sid for sid in (session_ids or []) if sid]
    if not ids:
        return set()
    clause, params = _workspace_key_clause(workspace)
    rows = store.all(f"SELECT s.id FROM sessions s WHERE s.id = ANY(%s) AND {clause}", ids, *params)
    return {row["id"] for row in rows}


def _last_reply(candidates) -> Optional[str]:
    """The newest of a segment's replies that shows text, projected as a display read projects it (the
    SQLite ``_last_reply``; a pure handoff the compressor wrote as an assistant row shows nothing)."""
    for content, summary in candidates:
        shown = project_compaction_message_for_display(
            {"role": "assistant", "content": content, "_compressed_summary": summary})
        text = _shape_preview(flatten_message_text(shown["content"])) if shown is not None else ""
        if text:
            return text
    return None


_LAST_REPLIES_SQL = (
    "SELECT " + _content_text() + " AS content, m._compressed_summary FROM messages m "
    "WHERE m.session_id = %s AND m.role = 'assistant' AND (m.active = 1 OR m.compacted = 1) "
    " AND m.content IS NOT NULL AND m.content <> ''::bytea "
    " AND COALESCE(m.display_kind, '') NOT IN ('hidden', 'failed_turn')" + _DISPLAY_VISIBLE_SQL
    + " ORDER BY m.id DESC LIMIT 8")


def picker_session_facts(store: "PostgresStore", root_ids, unlisted_ids, segment_ids) -> tuple:
    """``(roots, unlisted, last replies)`` in one store checkout (the SQLite ``picker_session_facts``)."""
    root_ids = [sid for sid in (root_ids or []) if sid]
    unlisted_ids = [sid for sid in (unlisted_ids or []) if sid]
    roots = {row["id"]: dict(row) for row in store.all(
        "SELECT id, cwd, git_repo_root FROM sessions WHERE id = ANY(%s)", root_ids) if root_ids} if root_ids else {}
    unlisted = {}
    if unlisted_ids:
        unlisted = {row["id"]: dict(row) for row in store.all(
            f"SELECT s.id, s.git_branch, s.pinned, {_preview_raw('s.id')}, "
            f"{_last_active_by_id('s.id')} AS last_active FROM sessions s WHERE s.id = ANY(%s)",
            unlisted_ids)}
        for row in unlisted.values():
            row["preview"] = _shape_preview(row.pop("_preview_raw", ""))
    replies = {}
    for segment in set(segment_ids or []):
        candidates = [(row["content"], bool(row["_compressed_summary"])) for row in store.all(
            _LAST_REPLIES_SQL, segment)]
        replies[segment] = _last_reply(candidates)
    return roots, unlisted, replies


def subagent_session_ids(store: "PostgresStore", subagent_id: str, marker_key: str) -> list:
    """Ids of the ``source='subagent'`` sessions whose model_config carries *subagent_id* under
    *marker_key*, newest first (the SQLite ``subagent_session_ids``; the caller proves ownership)."""
    rows = store.all(
        f"SELECT id FROM sessions WHERE source = 'subagent' AND {_json_key_text('model_config', marker_key)} = %s "
        "ORDER BY started_at DESC, id DESC", subagent_id)
    return [row["id"] for row in rows]


# ── the cron-run listing (the SQLite portability store, ported) ──

def list_cron_job_runs(store: "PostgresStore", job_id: str, limit: int = 20, offset: int = 0) -> list:
    """Run sessions of one cron job, newest first, in the ``list_sessions_rich`` row shape, stamped
    with ``cron_job_id`` and ``cron_execution_id`` (the SQLite ``list_cron_job_runs``; pre-unified runs
    are the flat ``cron_{job}_{ts}`` id range, unified runs the scheduler principal's admission range)."""
    prefix, prefix_hi = prefix_range(f"cron_{job_id}_")
    request_lo, request_hi = cron_request_range(job_id)
    query = (
        "SELECT s.*, " + _CRON_EXECUTION_SQL + " AS cron_execution_id, " + _PROMPT_RESOLVED_SQL
        + f", {_last_active('s')} AS last_active "
        "FROM sessions s LEFT JOIN system_prompts sp ON sp.hash = s.system_prompt_hash "
        "WHERE s.source = 'cron' AND s.id IN (" + _CRON_RUN_IDS_SQL + ") "
        "ORDER BY s.started_at DESC, s.id DESC LIMIT %s OFFSET %s")
    rows = store.all(
        query,
        # _CRON_EXECUTION_SQL binds first (it is the SELECT-list subquery): the substr anchor, then
        # the admission scan's params.
        request_lo, request_lo, request_hi, request_lo, request_lo, request_lo, job_id,
        # _CRON_RUN_IDS_SQL: legacy range (prefix, hi, then the prefix again for the anchored
        # stamp length), then the admission scan's params.
        prefix, prefix_hi, prefix,
        request_lo, request_hi, request_lo, request_lo, request_lo, job_id,
        limit, offset)
    return [{**_list_row(row), "cron_job_id": job_id} for row in rows]


def list_skill_scaffolded_sessions(store: "PostgresStore", limit: int = 200) -> list:
    """Titled sessions whose first user turn was a /skill invocation or a gateway auto-load scaffold (the
    SQLite ``list_skill_scaffolded_sessions``; returns id, title and the first-turn content)."""
    rows = store.all(
        "SELECT s.id, s.title, " + _content_text() + " AS content "
        "FROM sessions s JOIN messages m ON m.id = ("
        " SELECT m2.id FROM messages m2 WHERE m2.session_id = s.id AND m2.role = 'user' "
        " AND m2.content IS NOT NULL ORDER BY m2.timestamp, m2.id LIMIT 1) "
        "WHERE s.title IS NOT NULL AND (" + _content_text() + " LIKE %s OR " + _content_text() + " LIKE %s) "
        "ORDER BY s.started_at DESC LIMIT %s",
        SKILL_SCAFFOLD_SQL_LIKE, AUTO_LOAD_SCAFFOLD_SQL_LIKE, int(limit))
    return [dict(row) for row in rows]


def distinct_session_cwds(store: "PostgresStore", include_archived: bool = False) -> list:
    """Distinct non-empty session cwds with usage stats, for repo discovery (the SQLite
    ``distinct_session_cwds``; children/branches count)."""
    where = "cwd IS NOT NULL AND TRIM(cwd) <> ''"
    if not include_archived:
        where += " AND archived = 0"
    rows = store.all(
        "SELECT cwd AS cwd, COUNT(*) AS sessions, MAX(COALESCE(ended_at, started_at, 0)) AS last_active "
        f"FROM sessions WHERE {where} GROUP BY cwd ORDER BY cwd")
    return [{"cwd": r["cwd"], "sessions": int(r["sessions"] or 0), "last_active": float(r["last_active"] or 0)}
            for r in rows]


def cron_run_identity(store: "PostgresStore", session_id: str) -> Optional[dict]:
    """``{"cron_job_id", "cron_execution_id"}`` of a cron run session, else None (the SQLite
    ``cron_run_identity``: a legacy id names its job, a unified run is its scheduler admission probe)."""
    legacy_job = legacy_cron_run_job(session_id)
    if legacy_job is not None:
        return {"cron_job_id": legacy_job, "cron_execution_id": None}
    row = store.one(
        "SELECT request_id, " + _CRON_POLICY_JOB_SQL + " AS policy_job FROM session_admissions "
        "WHERE principal_id = %s AND target_session_id = %s ORDER BY seq LIMIT 1",
        CRON_PRINCIPAL_ID, session_id)
    run = None
    if row is not None:
        run = cron_run_from_request(row["request_id"])
        if run is None and isinstance(row["policy_job"], str):
            run = cron_run_from_request(row["request_id"], row["policy_job"])
    return {"cron_job_id": run[0], "cron_execution_id": run[1]} if run else None


def get_session_rich_row(store: "PostgresStore", session_id: str, compact_rows: bool = False) -> Optional[dict]:
    """One session with the ``list_sessions_rich`` enriched columns, or None."""
    return _get_session_rich_rows_batch(store, [session_id], compact_rows=compact_rows).get(session_id)


# ── the id search (the SQLite search store, ported) ──

def search_sessions_by_id(
    store: "PostgresStore", query: str, limit: int = 20, include_archived: bool = True,
    source: Optional[str] = None, sources: Optional[list] = None, exclude_sources: Optional[list] = None,
) -> list:
    """Search surfaced sessions by exact/prefix/substring session id, matching ``_lineage_root_id`` too
    (the SQLite ``search_sessions_by_id``; over-fetch so the in-Python ranking has candidates)."""
    needle = (query or "").strip().lower()
    if not needle or limit <= 0:
        return []
    candidates = list_sessions_rich(
        store, source=source, sources=sources, exclude_sources=exclude_sources,
        limit=max(limit * 4, limit), offset=0, include_archived=include_archived,
        order_by_last_active=True, id_query=needle)

    def score(row: dict) -> int:
        normalized = [str(v).lower() for v in (row.get("id") or "", row.get("_lineage_root_id") or "") if v]
        if any(value == needle for value in normalized):
            return 0
        return 1 if any(value.startswith(needle) for value in normalized) else 2

    ranked = sorted(enumerate(candidates), key=lambda item: (score(item[1]), item[0]))
    return [row for _, row in ranked[:limit]]


# ── the telegram listing (the SQLite telegram store, ported) ──

def list_unlinked_telegram_sessions_for_user(
    store: "PostgresStore", *, chat_id: str, user_id: str, profile_name: str = "default", limit: int = 10,
) -> list:
    """This user's Telegram sessions not bound to a topic (the SQLite
    ``list_unlinked_telegram_sessions_for_user``; the topic-bindings table is not in the port's DDL,
    so the profile-unscoped query is used: every unlinked telegram row of the user, any owner)."""
    rows = store.all(
        "SELECT s.*, " + _PROMPT_RESOLVED_SQL + f", {_preview_raw('s.id')}, "
        f"{_last_active('s')} AS last_active "
        "FROM sessions s LEFT JOIN system_prompts sp ON sp.hash = s.system_prompt_hash "
        "WHERE s.source = 'telegram' AND s.user_id = %s "
        "ORDER BY last_active DESC, s.started_at DESC LIMIT %s",
        str(user_id), int(limit))
    return [_list_row(row) for row in rows]


# ── the reopen guard (the SQLite compression store, ported) ──

def reopen_if_explicitly_closed(
    store: "PostgresStore", session_id: str, *, provenance: str, patience_s: Optional[float] = None,
) -> Optional[str]:
    """Clear an explicit-close stamp from a session a HOST has just proven is still routed to it (the
    SQLite ``reopen_if_explicitly_closed``; automatic, compression and boundary stamps are never
    touched, a published continuation owns the lineage elsewhere)."""
    if not session_id:
        return None
    with store.tx() as conn:
        row = conn.execute("SELECT ended_at, end_reason FROM sessions WHERE id = %s", (session_id,)).fetchone()
        if row is None or row["ended_at"] is None:
            return None
        reason = row["end_reason"]
        if is_automatic_end_reason(reason) or reason == "compression" or reason in _BOUNDARY_END_REASONS:
            return None
        # The child predicates compare each marker against the session's own id (the SQLite filter binds
        # every ``?`` to *session_id*; there is no ``parent`` alias in this one-table query).
        superseded = conn.execute(
            "SELECT 1 FROM sessions WHERE parent_session_id = %s"
            + _non_continuation_child_sql(child="", parent="%s") + " LIMIT 1",
            (session_id,) * 4).fetchone()
        if superseded is not None:
            return None
        changed = conn.execute(
            "UPDATE sessions SET ended_at = NULL, end_reason = NULL WHERE id = %s AND ended_at = %s AND end_reason = %s",
            (session_id, row["ended_at"], reason)).rowcount
        return str(reason) if changed else None


# ── the Bots roster (the SQLite facade's _read_bot_sessions, ported) ──

# A local chat: minted by this owner (``local-*``), or a pre-upgrade chat adopted under a legacy binding.
_LOCAL_CHAT = ("(chat_id LIKE 'local-' || CHR(37) OR EXISTS "
               "(SELECT 1 FROM state_meta WHERE key = %s || sessions.chat_id))")

_DESCENDANTS_SQL = f"""
            WITH RECURSIVE descendants(id, parent_session_id, started_at) AS (
                SELECT id, parent_session_id, started_at FROM sessions WHERE id = %s
                UNION
                SELECT s.id, s.parent_session_id, s.started_at
                FROM sessions s
                JOIN descendants d ON s.parent_session_id = d.id
                -- Continuation edges only (same predicate as the session list's chain CTE): a subagent run,
                -- a /branch fork, a /new reset child or a tool-owned row is its own conversation (#115092).
                WHERE {_json_key_text('s.model_config', '_delegate_from')} IS NULL
                  AND {_json_key_text('s.model_config', '_branched_from')} IS NULL
                  AND NOT ({_RESET_CHILD_SQL.format(a='s')})
                  AND COALESCE(s.source, '') <> 'tool'
            )
            SELECT id, parent_session_id, started_at FROM descendants
            """


def _read_bot_sessions(store: "PostgresStore", subject, operator, rule) -> dict:
    """The Bots roster's session fields of the store, in one read (the SQLite ``_read_bot_sessions``)."""
    hidden_sources = ",".join(" %s" for _ in INTERNAL_LISTING_SOURCES)
    owner_filter, owner_params = ("", ()) if operator else ("user_id = %s AND ", (subject,))
    conn = store.connection()

    def summary(root):
        tip = _compression_tip_on(conn, root["id"])
        target = conn.execute("SELECT * FROM sessions WHERE id = %s", (tip,)).fetchone() or root
        message = conn.execute(
            "SELECT " + _content_text() + " AS content FROM messages m "
            "WHERE session_id = %s AND active = 1 AND role IN ('user', 'assistant') "
            "AND TRIM(COALESCE(" + _content_text() + ", '')) <> '' "
            "ORDER BY id DESC LIMIT 1", (tip,)).fetchone()
        text = " ".join(str(message["content"] or "").split()) if message else ""
        return {"id": root["id"], "resolved_id": tip, "title": target["title"] or "",
                "root_title": root["title"] or "",
                "preview": text[:80] + "..." if len(text) > 80 else text,
                "started_at": target["started_at"] or 0,
                "last_active": target["last_activity_at"] or target["started_at"] or 0,
                "message_count": target["message_count"] or 0}

    fields: dict = {"last_session": None, "worker_session": None, "canonical_session": None}
    canonical = conn.execute(
        "SELECT * FROM sessions WHERE title = %s AND " + _LOCAL_CHAT,
        (CANONICAL_BOT_CHAT_TITLE, LEGACY_PREFIX)).fetchone()
    if canonical is not None and (same_owner(canonical["user_id"], subject, rule) or operator) \
            and not canonical["archived"]:
        fields["canonical_session"] = summary(canonical)
    latest = conn.execute(
        "SELECT * FROM sessions WHERE " + owner_filter + _LOCAL_CHAT +
        " AND archived = 0 AND hidden = 0 AND COALESCE(source, '') NOT IN (" + hidden_sources + ") "
        "ORDER BY COALESCE(last_activity_at, started_at) DESC LIMIT 1",
        (*owner_params, LEGACY_PREFIX, *INTERNAL_LISTING_SOURCES)).fetchone()
    if latest is not None and fields["canonical_session"] and latest["id"] == canonical["id"]:
        fields["last_session"] = dict(fields["canonical_session"])
    elif latest is not None:
        fields["last_session"] = summary(latest)
    worker = conn.execute(
        "SELECT id, source, title, COALESCE(last_activity_at, started_at) AS last_active FROM sessions "
        "WHERE COALESCE(source, '') IN (" + hidden_sources + ") AND archived = 0 AND hidden = 0 "
        "ORDER BY last_active DESC LIMIT 1", tuple(INTERNAL_LISTING_SOURCES)).fetchone()
    if worker is not None:
        fields["worker_session"] = {"id": worker["id"], "source": (worker["source"] or "").strip().lower(),
                                   "title": worker["title"] or "", "last_active": worker["last_active"] or 0}
    return fields


def _roster_session(fields: Optional[Mapping]) -> Optional[RosterSession]:
    if fields is None:
        return None
    return RosterSession(**{**fields, "id": SessionId(fields["id"]), "resolved_id": SessionId(fields["resolved_id"])})


def live_listing_facts(store: "PostgresStore", session_ids) -> LiveListingFacts:
    """The listing's live facts for a set of session ids (the SQLite ``live_listing_facts``): the rows
    with their target-ended stamp, plus each session's pending admissions."""
    ids = [sid for sid in (session_ids or []) if sid]
    if not ids:
        return LiveListingFacts(rows={}, pending={})
    conn = store.connection()
    rows = conn.execute(
        "SELECT s.id, s.source, s.title, s.started_at, s.ended_at, s.message_count, s.model, s.hidden, "
        "s.archived, s.runtime_revision, s.runtime_generation, "
        "(SELECT t.ended_at FROM state_meta m JOIN sessions t ON t.id = "
        "CASE WHEN hermes_json_valid(m.value) THEN m.value::jsonb -> 'entry' ->> 'session_id' END "
        "WHERE m.key = %s || s.id) AS target_ended_at "
        "FROM sessions s WHERE s.id = ANY(%s)",
        (POLICY_PREFIX, ids)).fetchall()
    pending = conn.execute(
        "SELECT target_session_id, admission_id, seq, status FROM session_admissions "
        "WHERE target_session_id = ANY(%s) AND status IN ('queued', 'started', 'unknown') ORDER BY seq",
        (ids,)).fetchall()
    by_session: dict = {}
    for row in pending:
        by_session.setdefault(SessionId(row["target_session_id"]), []).append(
            PendingAdmissionRef(admission_id=row["admission_id"], seq=int(row["seq"]), status=row["status"]))
    return LiveListingFacts(
        rows={
            SessionId(row["id"]): LiveListingRow(
                id=SessionId(row["id"]), source=row["source"] or "", title=row["title"],
                started_at=float(row["started_at"] or 0.0), ended_at=row["ended_at"],
                message_count=row["message_count"], model=row["model"], hidden=bool(row["hidden"]),
                archived=bool(row["archived"]), runtime_revision=int(row["runtime_revision"] or 0),
                runtime_generation=int(row["runtime_generation"] or 0), target_ended_at=row["target_ended_at"],
            )
            for row in rows
        },
        pending={sid: tuple(refs) for sid, refs in by_session.items()},
    )


# ── the facade's shared helpers (the SQLite facade's module-level helpers, ported) ──

def _ids(values) -> tuple:
    return tuple(SessionId(value) for value in values)


def _removed(values) -> tuple:
    """Removed or archived ids, each once (a write retried on a busy store reports its ids again)."""
    return _ids(tuple(dict.fromkeys(values)))


def _listed(values) -> Optional[list]:
    """A filter sequence as the store's list argument; empty or absent is no filter."""
    return list(values) if values else None


def _optional_list(values) -> Optional[list]:
    return list(values) if values is not None else None


def _sql_limit(window) -> int:
    """The window's ``LIMIT`` operand; ``-1`` as no limit."""
    return -1 if window.limit is None else window.limit


def _last_active_value(row: Mapping) -> Optional[float]:
    value = row.get("last_active")
    return None if value is None else float(value)


def _list_entry(row: Mapping, *, compact: bool = False) -> SessionListEntry:
    """One listing entry of a rich row; a row the store did not mark ``unread`` takes the store's
    unread rule (the SQLite ``_list_entry``)."""
    return SessionListEntry(
        session=session_record(row),
        preview=row.get("preview") or "",
        unread=bool(row["unread"]) if "unread" in row else session_unread(dict(row)),
        last_active=_last_active_value(row),
        lineage_root_id=row.get("_lineage_root_id"),
        lineage_ids=_ids(row.get("_lineage_ids") or ()),
        continuation_kind=row.get("continuation_kind"),
        compact=compact,
    )


def _recency_entry(row: Mapping) -> SessionRecencyEntry:
    return SessionRecencyEntry(session=session_record(row), last_active=_last_active_value(row))


_TITLE_HELD = re.compile(r"Title '(?P<title>.*)' is already in use by session (?P<holder>.+)", re.DOTALL)
_CANONICAL_TITLE_LOCKED = "This is the bot's canonical Bot Chat"


@contextmanager
def _title_refusals(session_id: SessionId) -> Iterator[None]:
    """The store's title ``ValueError`` refusals as :class:`TitleConflict` and
    :class:`CanonicalTitleLocked` (the SQLite ``_title_refusals``)."""
    try:
        yield
    except ValueError as exc:
        text = str(exc)
        if held := _TITLE_HELD.fullmatch(text):
            raise TitleConflict(held["title"], SessionId(held["holder"])) from exc
        if text.startswith(_CANONICAL_TITLE_LOCKED):
            raise CanonicalTitleLocked(session_id) from exc
        raise


@contextmanager
def refusals(store: Optional["PostgresStore"] = None, session_id: Optional[str] = None) -> Iterator[None]:
    """Raise the store's failures (``translated``) and its typed refusals as the contract's classes,
    chaining the original (the SQLite ``refusals``; the Postgres ``tx`` doors already translate the
    driver errors, so this maps the contract-typed store refusals only)."""
    sid = session_id or ""
    try:
        yield
    except AuthorityRefusal as exc:
        # A bare unknown-execution refusal becomes the contract's typed UnknownExecution (the SQLite
        # ``refusals``); the session names it.
        if type(exc) is AuthorityRefusal and exc.reason == "unknown_execution":
            raise UnknownExecution("session", sid) from exc
        raise
    except SessionCompressionInProgressError as exc:
        raise CompressionInProgress(sid, str(exc)) from exc
    except SessionActiveWriteGuardError as exc:
        raise SessionWriteGuarded(sid, str(exc)) from exc
    except CompressionSessionClosedError as exc:
        tip = _get_compression_tip(store, exc.session_id) if store is not None else exc.session_id
        raise CompressionClosed(exc.session_id, tip) from exc
    except SessionTurnLeaseLostError as exc:
        raise TurnLeaseLost(sid, str(exc)) from exc




# ── the cron-run listing (the SQLite portability store, ported) ──

_CRON_POLICY_JOB_SQL = (
    "(SELECT CASE WHEN hermes_json_valid(r) THEN r::jsonb -> 'cron_job' ->> 'id' END FROM"
    " (SELECT CASE WHEN hermes_json_valid(value) THEN (value::jsonb -> 'policy' -> 'request_json')::text END AS r"
    f" FROM state_meta WHERE key = '{POLICY_PREFIX}' || target_session_id))")

_CRON_ADMISSION_SQL = (
    f"principal_id = '{CRON_PRINCIPAL_ID}' AND request_id >= %s AND request_id < %s"
    f" AND ((strpos(substr(request_id, char_length(%s) + 1), ':') = char_length(%s) - {len(CRON_REQUEST_PREFIX)}"
    " AND strpos(substr(request_id, char_length(%s) + 1), ':') = 0)"
    f" OR {_CRON_POLICY_JOB_SQL} = %s)")

_CRON_RUN_IDS_SQL = (
    "SELECT id FROM sessions WHERE source = 'cron' AND id >= %s AND id < %s"
    " AND substr(id, char_length(%s) + 1) ~ '^[0-9]{8}_[0-9]{6}$'"
    f" UNION ALL SELECT target_session_id FROM session_admissions WHERE {_CRON_ADMISSION_SQL}")

_CRON_EXECUTION_SQL = (
    "(SELECT substr(request_id, char_length(%s) + 1) FROM session_admissions"
    f" WHERE target_session_id = s.id AND {_CRON_ADMISSION_SQL} LIMIT 1)")


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

    # ── writes ──

    def end(self, session_id: SessionId, reason: str) -> None:
        store = self._store
        with refusals(store, session_id):
            end_session(store, session_id, reason)

    def set_title(self, session_id: SessionId, title: Optional[str]) -> bool:
        store = self._store
        with refusals(store, session_id), _title_refusals(session_id):
            return bool(set_session_title(store, session_id, title))

    def set_auto_title(self, session_id: SessionId, title: str, *, source: TitleSource) -> bool:
        store = self._store
        with refusals(store, session_id), _title_refusals(session_id):
            return bool(set_auto_title(store, session_id, title, source=source.value))

    def delete(
        self,
        session_id: SessionId,
        *,
        expected_ids: Optional[Sequence[SessionId]] = None,
        expected_display: Optional[Mapping[SessionId, Sequence[Mapping[str, Any]]]] = None,
        refuse_guarded: bool = False,
        include_compression_chain: bool = False,
        media_root: str,
    ) -> tuple[SessionId, ...]:
        store = self._store
        with refusals(store, session_id):
            removed = delete_session(
                store,
                session_id,
                expected_delete_ids=list(expected_ids) if expected_ids is not None else None,
                expected_display_messages=(
                    {sid: [dict(message) for message in messages] for sid, messages in expected_display.items()}
                    if expected_display is not None else None
                ),
                exclude_active_write_guards=refuse_guarded,
                include_compression_chain=include_compression_chain,
                media_root=media_root,
            )
        return _removed(removed)

    def delete_many(
        self,
        session_ids: Sequence[SessionId],
        *,
        skip_guarded: bool = False,
        include_compression_chain: bool = False,
        media_root: str,
    ) -> DeleteManyOutcome:
        store = self._store
        skipped: list = []
        with refusals(store):
            deleted, removed = delete_sessions(
                store, list(session_ids), exclude_active_write_guards=skip_guarded, skipped_ids=skipped,
                include_compression_chain=include_compression_chain, media_root=media_root,
            )
        return DeleteManyOutcome(deleted=deleted, skipped_guarded=_ids(skipped), removed_ids=_removed(removed))

    def delete_if_empty(self, session_id: SessionId, *, media_root: str) -> tuple[SessionId, ...]:
        store = self._store
        with refusals(store, session_id):
            return _ids(delete_session_if_empty(store, session_id, media_root=media_root))

    def delete_empty(self, *, media_root: str) -> tuple[SessionId, ...]:
        store = self._store
        with refusals(store):
            return _removed(delete_empty_sessions(store, media_root=media_root))

    def create_exclusive(
        self,
        session_id: SessionId,
        *,
        source: str,
        model: Optional[str] = None,
        model_config: Optional[Mapping[str, Any]] = None,
        system_prompt: Optional[str] = None,
        title: Optional[str] = None,
    ) -> SessionRecord:
        store = self._store
        model_config = dict(model_config) if model_config else None

        def _atomic(conn):
            if conflict := _create_conflict(conn, session_id):  # one write transaction: a racing create sees the row
                return None, conflict
            conn.execute(
                """INSERT INTO sessions (
                       id, source, model, model_config, system_prompt, started_at
                   ) VALUES (%s, %s, %s, %s, %s, %s)""",
                (session_id, source, model, json.dumps(model_config) if model_config else None,
                 system_prompt, time.time()))
            if title is not None:
                clean_title = sanitize_title(str(title))
                if clean_title:
                    try:
                        _resolve_title_conflict(conn, session_id, clean_title)
                    except ValueError as exc:  # the DB's uniqueness rule; undo the INSERT
                        conn.execute("DELETE FROM sessions WHERE id = %s", (session_id,))
                        return None, ("title", str(exc))
                conn.execute("UPDATE sessions SET title = %s WHERE id = %s", (clean_title, session_id))
            return dict(conn.execute("SELECT * FROM sessions WHERE id = %s", (session_id,)).fetchone()), None

        with refusals(store, session_id):
            with store.tx() as conn:
                row, conflict = _atomic(conn)
            if conflict is None:
                return session_record(row)
            code, message = conflict
            if code == "title":
                with _title_refusals(session_id):
                    raise ValueError(message)
            existing = get_session_row(store, session_id) if code == "session_exists" else None
        if existing is not None:
            raise SessionExists(session_record(existing))
        raise RetiredSessionError(session_id, "session_deleted")

    def reopen(self, session_id: SessionId) -> None:
        store = self._store
        with refusals(store, session_id):
            reopen_session(store, session_id)

    def retire_undrained_queue_rows(self, session_id: SessionId) -> int:
        store = self._store
        with refusals(store, session_id):
            return int(retire_undrained_queue_rows(store, session_id) or 0)

    def promote_to_reset(self, session_id: SessionId, reason: str = END_REASON_SESSION_RESET) -> bool:
        store = self._store
        with refusals(store, session_id):
            return promote_to_session_reset(store, session_id, reason)

    def reopen_if_explicitly_closed(
        self, session_id: SessionId, *, provenance: str, patience_s: Optional[float] = None
    ) -> Optional[str]:
        store = self._store
        with refusals(store, session_id):
            return reopen_if_explicitly_closed(store, session_id, provenance=provenance, patience_s=patience_s)

    def discard_unadmitted(self, session_id: SessionId, *, media_root: str) -> tuple[SessionId, ...]:
        store = self._store
        with refusals(store, session_id):
            return _removed(discard_unadmitted_session(store, session_id, media_root=media_root))

    def set_title_source(self, session_id: SessionId, source: TitleSource) -> bool:
        store = self._store
        with refusals(store, session_id):
            return set_session_title_source(store, session_id, source.value)

    def retitle_canonical_bot_chat(self, session_id: SessionId, title: str) -> bool:
        store = self._store
        with refusals(store, session_id):
            return retitle_canonical_bot_chat(store, session_id, title)

    def replace_session_config(self, session_id: SessionId, command: ReplaceSessionConfig) -> None:
        store = self._store
        with refusals(store, session_id):
            update_session_meta(store, session_id, json.dumps(dict(command.model_config)), command.model)

    def patch_model_config(self, session_id: SessionId, patch: Mapping[str, Any]) -> None:
        store = self._store
        with refusals(store, session_id):
            patch_session_model_config(store, session_id, dict(patch))

    def advance_onboarding_run(self, session_id: SessionId, event: Any, *, now: float) -> OnboardingAdvance:
        store = self._store
        outcome: dict = {}

        def update(current: Any) -> Any:
            outcome.clear()
            if not isinstance(current, dict):
                return current
            outcome["was_active"] = onboarding_flow.is_active(current)
            outcome["step"] = onboarding_flow.apply(
                current, event, onboarding_flow.Facts.from_stored(current["facts"]), now=now)
            return outcome["step"].record
        with refusals(store, session_id):
            tip = update_conversation_model_config_value(store, session_id, ONBOARDING_RUN_KEY, update)
        return OnboardingAdvance(session_id=None if tip is None else SessionId(tip),
                                 was_active=outcome.get("was_active", False), step=outcome.get("step"))

    def update_model(
        self,
        session_id: SessionId,
        model: str,
        provider: Optional[str] = None,
        *,
        base_url: Optional[str] = None,
        api_mode: Optional[str] = None,
    ) -> None:
        store = self._store
        with refusals(store, session_id):
            update_session_model(store, session_id, model, provider, base_url=base_url, api_mode=api_mode)

    def set_runtime_lock(self, session_id: SessionId, lock: RuntimeLock) -> None:
        store = self._store
        with refusals(store, session_id):
            update_session_runtime_lock(
                store, session_id, model=lock.model, provider=lock.provider,
                model_options=dict(lock.model_options) if lock.model_options is not None else None,
                route_source=lock.route_source, confirmed=lock.confirmed,
            )

    def set_yolo(self, session_id: SessionId, enabled: bool) -> None:
        store = self._store
        with refusals(store, session_id):
            set_session_yolo(store, session_id, enabled)

    def update_system_prompt(self, session_id: SessionId, system_prompt: Optional[str]) -> None:
        store = self._store
        with refusals(store, session_id):
            update_system_prompt(store, session_id, system_prompt)

    def request_prompt_refresh(self, session_id: Optional[SessionId] = None) -> int:
        store = self._store
        with refusals(store, session_id):
            return request_prompt_refresh(store, session_id)

    def update_tool_pin(self, session_id: SessionId, tool_pin: Any) -> None:
        store = self._store
        with refusals(store, session_id):
            update_session_tool_names(store, session_id, tool_pin)

    def update_cwd(
        self,
        session_id: SessionId,
        cwd: str,
        git_branch: Optional[str] = None,
        git_repo_root: Optional[str] = None,
        *,
        replace_git_meta: bool = False,
    ) -> Optional[int]:
        store = self._store
        with refusals(store, session_id):
            return update_session_cwd(store, session_id, cwd, git_branch, git_repo_root, replace_git_meta=replace_git_meta)

    def publish_git_metadata(
        self,
        session_id: SessionId,
        cwd: str,
        generation: int,
        git_branch: Optional[str] = None,
        git_repo_root: Optional[str] = None,
    ) -> bool:
        store = self._store
        with refusals(store, session_id):
            return bool(publish_session_git_metadata(store, session_id, cwd, generation, git_branch, git_repo_root))

    def set_archived(self, session_id: SessionId, archived: bool, *, notify: bool = True) -> ArchiveChange:
        store = self._store
        with refusals(store, session_id):
            changed, archived_ids = set_session_archived(store, session_id, archived, notify=notify)
        return ArchiveChange(changed=changed, archived_ids=_removed(archived_ids))

    def set_pinned(self, session_id: SessionId, pinned: bool) -> bool:
        store = self._store
        with refusals(store, session_id):
            return bool(set_session_pinned(store, session_id, pinned))

    def set_hidden(self, session_id: SessionId, hidden: bool) -> bool:
        store = self._store
        with refusals(store, session_id):
            return bool(set_session_hidden(store, session_id, hidden))

    def set_read(self, session_id: SessionId, read: bool = True) -> bool:
        store = self._store
        with refusals(store, session_id):
            return bool(set_session_read(store, session_id, read))

    def unarchive_recoverable(self, session_id: SessionId) -> bool:
        store = self._store
        with refusals(store, session_id):
            return unarchive_recoverable_session(store, session_id)

    def backfill_repo_roots(self, cwd_to_root: Mapping[str, str]) -> None:
        store = self._store
        with refusals(store):
            backfill_repo_roots(store, dict(cwd_to_root))

    def backfill_session_owners(self, *, stamp: Optional[str] = None) -> int:
        store = self._store
        owner = store.owner_profile_name if stamp is None else stamp
        if not owner:
            return 0
        with refusals(store):
            return int(backfill_null_session_profiles(store, owner) or 0)

    def backfill_acp_session_cwds(self) -> int:
        store = self._store
        with refusals(store):
            return int(backfill_acp_session_cwd(store) or 0)

    def retag_worker_sessions(self, workspaces_root: str) -> int:
        store = self._store
        with refusals(store):
            return int(retag_kanban_worker_sessions(store, workspaces_root) or 0)

    def touch_activity(
        self,
        session_id: SessionId,
        at: Optional[float] = None,
        *,
        description: Optional[str] = None,
        provenance: Optional[str] = None,
    ) -> None:
        store = self._store
        with refusals(store, session_id):
            touch_session_activity(store, session_id, at, description=description, provenance=provenance)

    def clear_activity_labels(self, session_id: SessionId) -> None:
        store = self._store
        with refusals(store, session_id):
            clear_session_activity_labels(store, session_id)

    # ── reads ──

    def owns_session(self, session_id: SessionId) -> bool:
        store = self._store
        owner = store.owner_profile_name
        with refusals(store, session_id):
            row = store.one("SELECT profile_name FROM sessions WHERE id = %s", session_id)
        return row is not None and row["profile_name"] in (None, owner)

    def store_owner(self) -> Optional[str]:
        # The owner is derived from the store's scope alone (never a guess: None outside the profile tree).
        return self._store.owner_profile_name

    def resolve_id(self, id_or_prefix: str) -> Optional[SessionId]:
        store = self._store
        with refusals(store):
            found = resolve_session_id(store, id_or_prefix)
        return SessionId(found) if found else None

    def is_kanban_owned(self, session_id: SessionId) -> bool:
        store = self._store
        with refusals(store, session_id):
            return is_kanban_owned_session(store, session_id)

    def declared_scope_identity(self, session_id: SessionId) -> DeclaredScopeIdentity:
        store = self._store
        with refusals(store, session_id):
            is_fork_child, source = declared_scope_identity(store, session_id)
        return DeclaredScopeIdentity(is_fork_child=is_fork_child, source=source)

    def get_title(self, session_id: SessionId) -> Optional[str]:
        store = self._store
        with refusals(store, session_id):
            return get_session_title(store, session_id)

    def get_title_source(self, session_id: SessionId) -> Optional[TitleSource]:
        store = self._store
        with refusals(store, session_id):
            return _TITLE_SOURCES.get(get_session_title_source(store, session_id))

    def get_by_title(self, title: str) -> Optional[SessionRecord]:
        store = self._store
        with refusals(store):
            row = get_session_by_title(store, title)
        return session_record(row) if row else None

    def resolve_by_title(self, title: str) -> Optional[SessionId]:
        store = self._store
        with refusals(store):
            found = resolve_session_by_title(store, title)
        return SessionId(found) if found else None

    def next_title_in_lineage(self, base_title: str) -> str:
        store = self._store
        with refusals(store):
            return next_title_in_lineage(store, base_title)

    def title_holder(self, title: str, *, excluding: SessionId) -> Optional[SessionId]:
        if not title:
            return None
        store = self._store
        with refusals(store):
            row = store.one("SELECT id FROM sessions WHERE title = %s AND id <> %s", title, excluding)
        return SessionId(row["id"]) if row else None

    def get_model_config_value(self, session_id: SessionId, key: str, default: Any = None) -> Any:
        store = self._store
        with refusals(store, session_id):
            return get_session_model_config_value(store, session_id, key, default)

    def session_ids_with_model_config_key(self, key: str) -> tuple[SessionId, ...]:
        store = self._store
        with refusals(store):
            return _ids(session_ids_with_model_config_key(store, key))

    def prompt_refresh_epoch(self, session_id: SessionId) -> int:
        store = self._store
        with refusals(store, session_id):
            return prompt_refresh_epoch(store, session_id)

    def count_with_stored_prompt(self) -> int:
        store = self._store
        with refusals(store):
            return count_sessions_with_stored_prompt(store)

    def delete_targets(self, session_id: SessionId) -> tuple[SessionId, ...]:
        store = self._store
        with refusals(store, session_id):
            return _ids(get_session_delete_targets(store, session_id))

    def is_empty_candidate(self, session_id: SessionId) -> bool:
        store = self._store
        with refusals(store, session_id):
            row = store.one(
                "SELECT 1 FROM sessions WHERE id = %s AND title IS NULL "
                " AND NOT EXISTS (SELECT 1 FROM messages WHERE messages.session_id = sessions.id) "
                " AND NOT EXISTS (SELECT 1 FROM sessions child WHERE child.parent_session_id = sessions.id)",
                session_id)
        return row is not None

    def count_empty(self) -> int:
        store = self._store
        with refusals(store):
            return int(count_empty_sessions(store))

    def lifecycle_statuses(self, session_ids: Sequence[SessionId]) -> Mapping[SessionId, SessionLifecycleStatus]:
        store = self._store
        with refusals(store):
            statuses = session_lifecycle_statuses(store, list(session_ids))
        return {SessionId(sid): SessionLifecycleStatus(status) for sid, status in statuses.items()}

    def descendants(self, session_id: SessionId) -> tuple[SessionEdge, ...]:
        store = self._store
        with refusals(store, session_id):
            rows = store.all(_DESCENDANTS_SQL, session_id)
        return tuple(SessionEdge(session_id=SessionId(row["id"]), parent_session_id=row["parent_session_id"],
                                 started_at=row["started_at"]) for row in rows)

    def first_session_started_at(self) -> Optional[float]:
        store = self._store
        with refusals(store):
            value = store.scalar("SELECT MIN(started_at) FROM sessions")
        return float(value) if value is not None else None

    def longest_session_id(self) -> Optional[SessionId]:
        store = self._store
        with refusals(store):
            row = store.one(
                "SELECT id FROM sessions s ORDER BY "
                "(SELECT COUNT(*) FROM messages m WHERE m.session_id = s.id) DESC LIMIT 1")
        return SessionId(row["id"]) if row else None

    def sessions_in_workspace(self, session_ids: Sequence[SessionId], workspace: str) -> frozenset[SessionId]:
        store = self._store
        with refusals(store):
            return frozenset(SessionId(sid) for sid in sessions_in_workspace(store, session_ids, workspace))

    def picker_session_facts(self, root_ids: Sequence[SessionId], segment_ids: Sequence[SessionId],
                             reply_ids: Sequence[SessionId]) -> PickerFacts:
        store = self._store
        with refusals(store):
            roots, segments, replies = picker_session_facts(store, root_ids, segment_ids, reply_ids)
        return PickerFacts(
            roots={SessionId(sid): PickerRoot(cwd=row["cwd"], git_repo_root=row["git_repo_root"]) for sid, row in roots.items()},
            segments={SessionId(sid): PickerSegment(preview=row["preview"], git_branch=row["git_branch"],
                                                    pinned=bool(row["pinned"]), last_active=row["last_active"])
                      for sid, row in segments.items()},
            last_replies={SessionId(sid): reply for sid, reply in replies.items()})

    def subagent_session_ids(self, subagent_id: str, marker_key: str) -> tuple[SessionId, ...]:
        store = self._store
        with refusals(store):
            return tuple(SessionId(sid) for sid in subagent_session_ids(store, subagent_id, marker_key))

    def records_created_surface(self) -> bool:
        store = self._store
        with refusals(store):
            row = store.one(
                "SELECT 1 FROM information_schema.columns "
                "WHERE table_schema = current_schema() AND table_name = 'sessions' AND column_name = 'created_source'")
        return row is not None

    def last_active(self, session_id: SessionId) -> Optional[float]:
        store = self._store
        with refusals(store, session_id):
            row = store.one(f"SELECT {_last_active('s')} AS last_active FROM sessions s WHERE s.id = %s", session_id)
        return row["last_active"] if row else None

    def list_sessions(
        self,
        query: SessionQuery,
        window: SessionWindow = SessionWindow(limit=20),
        *,
        compact_rows: bool = False,
    ) -> tuple[SessionListEntry, ...]:
        store = self._store
        with refusals(store):
            rows = list_sessions_rich(
                store,
                sources=_listed(query.sources), exclude_sources=_listed(query.exclude_sources),
                cwd_prefix=query.cwd_prefix, limit=_sql_limit(window), offset=window.offset,
                include_children=query.include_children, min_message_count=query.min_message_count,
                project_compression_tips=query.project_compression_tips,
                order_by_last_active=query.order is SessionOrder.LAST_ACTIVE, include_archived=query.include_archived,
                archived_only=query.archived_only, id_query=query.id_query, search_query=query.search_text,
                compact_rows=compact_rows, include_pinned=query.include_pinned, session_key=query.session_key,
                include_hidden=query.include_hidden, include_subagents=query.include_subagents,
                local_created_sources=_optional_list(query.local_created_sources), workspace_key=query.workspace_key,
            )
        return tuple(_list_entry(row, compact=compact_rows) for row in rows)

    def search_sessions(
        self,
        *,
        sources: Optional[Sequence[str]] = None,
        workspace_key: Optional[str] = None,
        local_created_sources: Optional[Sequence[str]] = None,
        skip_launch_stubs: bool = False,
        window: SessionWindow = SessionWindow(limit=20),
    ) -> tuple[SessionRecencyEntry, ...]:
        store = self._store
        with refusals(store):
            rows = search_sessions(
                store,
                source=_listed(sources), limit=_sql_limit(window), offset=window.offset, workspace_key=workspace_key,
                local_created_sources=_optional_list(local_created_sources), skip_launch_stubs=skip_launch_stubs,
            )
        return tuple(_recency_entry(row) for row in rows)

    def list_row(self, session_id: SessionId) -> Optional[SessionListRow]:
        store = self._store
        with refusals(store, session_id):
            row = get_session_list_row(store, session_id)
        if row is None:
            return None
        return SessionListRow(
            session_id=SessionId(row["id"]), source=row["source"] or "", title=row["title"],
            started_at=float(row["started_at"] or 0.0), ended_at=row["ended_at"],
            message_count=row["message_count"], model=row["model"], hidden=bool(row["hidden"]),
            archived=bool(row["archived"]),
        )

    def summary(self, session_id: SessionId, *, compact_rows: bool = False) -> Optional[SessionListEntry]:
        store = self._store
        with refusals(store, session_id):
            row = get_session_rich_row(store, session_id, compact_rows=compact_rows)
        return _list_entry(row, compact=compact_rows) if row else None

    def search_by_id(
        self,
        text: str,
        *,
        limit: int = 20,
        include_archived: bool = True,
        sources: Optional[Sequence[str]] = None,
        exclude_sources: Sequence[str] = (),
    ) -> tuple[SessionListEntry, ...]:
        store = self._store
        with refusals(store):
            rows = search_sessions_by_id(
                store, text, limit=limit, include_archived=include_archived, sources=_listed(sources),
                exclude_sources=_listed(exclude_sources))
        return tuple(_list_entry(row) for row in rows)

    def count_sessions(self, query: SessionCountQuery = SessionCountQuery()) -> int:
        store = self._store
        with refusals(store):
            return int(session_count(
                store,
                sources=_listed(query.sources), cwd_prefix=query.cwd_prefix, min_message_count=query.min_message_count,
                include_archived=query.include_archived, archived_only=query.archived_only,
                exclude_children=query.exclude_children, exclude_sources=_listed(query.exclude_sources),
                include_subagents=query.include_subagents,
                local_created_sources=_optional_list(query.local_created_sources)))

    def count_at_least(self, minimum: int = 1) -> bool:
        store = self._store
        with refusals(store):
            return session_count_ge(store, minimum)

    def count_by_source(
        self, *, include_archived: bool = False, archived_only: bool = False, exclude_children: bool = False
    ) -> Mapping[str, int]:
        store = self._store
        with refusals(store):
            return session_count_by_source(
                store,
                include_archived=include_archived, archived_only=archived_only, exclude_children=exclude_children,
            )

    def list_gateway_sessions(
        self, *, platform: Optional[str] = None, active_only: bool = True
    ) -> tuple[SessionRecencyEntry, ...]:
        store = self._store
        with refusals(store):
            rows = list_gateway_sessions(store, platform=platform, active_only=active_only)
        return tuple(_recency_entry(row) for row in rows)

    def list_recent_sessions(self, query: RecentBrowseQuery) -> tuple[RecentSessionEntry, ...]:
        store = self._store
        with refusals(store):
            try:
                rows = list_recent_sessions_bounded(
                    store, limit=query.limit, exclude_sources=_listed(query.exclude_sources),
                    timeout_seconds=query.timeout_seconds, candidate_limit=query.candidate_limit,
                    lineage_limit=query.lineage_limit,
                )
            except TimeoutError as exc:
                raise RecentBrowseTimeout(str(exc)) from exc
        return tuple(
            RecentSessionEntry(
                session_id=SessionId(row["id"]), source=row["source"] or "", model=row.get("model"),
                title=row.get("title"), started_at=float(row["started_at"] or 0.0), ended_at=row["ended_at"],
                end_reason=row["end_reason"], message_count=row["message_count"],
                tool_call_count=row["tool_call_count"], last_active=_last_active_value(row),
                preview=row.get("preview") or "", lineage_root_id=row.get("_lineage_root_id"),
                unread=bool(row.get("unread")),
            )
            for row in rows
        )

    def list_cron_job_runs(
        self, job_id: str, window: SessionWindow = SessionWindow(limit=20)
    ) -> tuple[CronRunEntry, ...]:
        store = self._store
        with refusals(store):
            rows = list_cron_job_runs(store, job_id, limit=_sql_limit(window), offset=window.offset)
        return tuple(
            CronRunEntry(entry=_list_entry(row), cron_job_id=row["cron_job_id"],
                         cron_execution_id=row.get("cron_execution_id"))
            for row in rows
        )

    def cron_run_identity(self, session_id: SessionId) -> Optional[CronRunIdentity]:
        store = self._store
        with refusals(store, session_id):
            found = cron_run_identity(store, session_id)
        if not found:
            return None
        return CronRunIdentity(cron_job_id=found["cron_job_id"], cron_execution_id=found["cron_execution_id"])

    def list_skill_scaffolded_sessions(self, *, limit: int = 200) -> tuple[SkillScaffoldedSession, ...]:
        store = self._store
        with refusals(store):
            rows = list_skill_scaffolded_sessions(store, limit)
        return tuple(SkillScaffoldedSession(session_id=SessionId(row["id"]), title=row["title"], content=row["content"])
                     for row in rows)

    def list_unlinked_telegram_sessions(
        self, *, user_id: str, profile_name: Optional[str] = None, limit: int = 10,
    ) -> tuple[SessionFacetEntry, ...]:
        store = self._store
        profile = (store.owner_profile_name or "default") if profile_name is None else profile_name
        with refusals(store):
            rows = list_unlinked_telegram_sessions_for_user(
                store, chat_id="", user_id=user_id, profile_name=profile, limit=limit,
            )
        return tuple(
            SessionFacetEntry(session=session_record(row), last_active=_last_active_value(row),
                             preview=row.get("preview") or "")
            for row in rows
        )

    def distinct_session_cwds(self, *, include_archived: bool = False) -> tuple[SessionCwdUsage, ...]:
        store = self._store
        with refusals(store):
            rows = distinct_session_cwds(store, include_archived=include_archived)
        return tuple(SessionCwdUsage(cwd=row["cwd"], sessions=row["sessions"], last_active=row["last_active"])
                     for row in rows)

    def roster_sessions(self, *, owner_user_id: Optional[str], owner: OwnerRule | None) -> RosterSessions:
        store = self._store
        with refusals(store):
            fields = _read_bot_sessions(store, owner_user_id, owner_user_id is None, owner)
        worker = fields['worker_session']
        return RosterSessions(
            canonical_session=_roster_session(fields['canonical_session']),
            last_session=_roster_session(fields['last_session']),
            worker_session=RosterWorker(
                id=SessionId(worker['id']), source=worker['source'], title=worker['title'],
                last_active=worker['last_active'],
            ) if worker else None,
        )

    def live_listing_facts(self, session_ids: Sequence[SessionId]) -> LiveListingFacts:
        store = self._store
        with refusals(store):
            return live_listing_facts(store, list(session_ids))


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

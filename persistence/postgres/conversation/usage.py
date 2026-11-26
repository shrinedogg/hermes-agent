"""``UsageRepository`` and ``UsageAnalytics`` on PostgreSQL (stage 1): token and cost accounting and the
dashboard/insights reads, the SQLite usage store's SQL ported inline.

The SQLite store applies queued deltas on a background accounting queue; the PG store has no such queue, so
``queue_usage`` applies the delta synchronously through the same counters transaction (the session row's
counters, the first-accounted-route fill, the prompt-cache clock and the per-model ``session_model_usage``
upsert) and ``flush_usage`` is a no-op barrier. The pure delta-translation helper is imported from the
SQLite repository module; the SQL is this module's own.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Mapping, Optional

from persistence import AuthorityRefusal, SessionId
from persistence.conversation.usage import (
    AuxiliaryTaskUsage, BillingRoute, PromptCacheClockRecord, RouteUsage, TokenCounts, UsageDelta,
    UsageRepository, UsageTotals, UsageTotalsQuery,
)
from persistence.conversation.usage import UsageAnalytics
from persistence.sqlite.conversation._store.retirement import RETIRED_PREFIX
from persistence.sqlite.conversation.usage import _delta_kwargs
from psycopg import errors as pg_errors

from .._backend import PostgresStore, register_repository, translated
from .._lineage_common import json_marker_absent_sql, session_lineage_root_to_tip
from .._stubs import add_stubs

__all__ = ["PgUsageRepository"]

logger = logging.getLogger(__name__)

_TOKEN_COUNTERS = ("input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens", "reasoning_tokens")
_MODEL_USAGE_FIELDS = frozenset((
    "model", "billing_provider", "billing_base_url", "billing_mode", "input_tokens", "output_tokens",
    "cache_read_tokens", "cache_write_tokens", "reasoning_tokens", "estimated_cost_usd",
    "actual_cost_usd", "cost_status", "cost_source", "api_call_count"))


def _token_update_sql(delta: bool) -> str:
    """``UPDATE sessions`` for one usage report (the SQLite store's statement shape on the PG dialect;
    *delta* adds to the stored counters, otherwise sets them)."""
    def add(col: str) -> str:
        return f"{col} + %s" if delta else "%s"
    def add0(col: str) -> str:
        return f"COALESCE({col}, 0) + %s" if delta else "%s"
    counters = "".join(f"                   {c} = {add(c)},\n" for c in _TOKEN_COUNTERS)
    estimated = "COALESCE(estimated_cost_usd, 0) + COALESCE(%s, 0)" if delta else "COALESCE(%s, 0)"
    return (
        "UPDATE sessions SET\n" + counters
        + f"""                   estimated_cost_usd = {estimated},
                   actual_cost_usd = CASE
                       WHEN %s::double precision IS NULL THEN actual_cost_usd
                       ELSE {add0("actual_cost_usd")}
                   END,
                   cost_status = COALESCE(%s, cost_status),
                   cost_source = COALESCE(%s, cost_source),
                   pricing_version = COALESCE(%s, pricing_version),
                   billing_provider = COALESCE(billing_provider, %s),
                   billing_base_url = COALESCE(billing_base_url, %s),
                   billing_mode = COALESCE(billing_mode, %s),
                   model = COALESCE(model, %s),
                   api_call_count = {add0("api_call_count")}
                   WHERE id = %s"""
    )


_TOKEN_UPDATE_ABSOLUTE_SQL = _token_update_sql(delta=False)
_TOKEN_UPDATE_DELTA_SQL = _token_update_sql(delta=True)

# Rows a compaction or prune archived: the prompt-cache clock's durable context generation.
_ARCHIVED_ROWS_SQL = "SELECT COUNT(*) FROM messages WHERE session_id = %s AND active = 0 AND compacted = 1"

_MODEL_USAGE_UPSERT_SQL = """INSERT INTO session_model_usage (
                   session_id, model, billing_provider, billing_base_url, billing_mode,
                   task, api_call_count, input_tokens, output_tokens,
                   cache_read_tokens, cache_write_tokens, reasoning_tokens,
                   estimated_cost_usd, actual_cost_usd, cost_status, cost_source,
                   first_seen, last_seen
               ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
               ON CONFLICT(session_id, model, billing_provider, billing_base_url, billing_mode, task)
               DO UPDATE SET
                   api_call_count = session_model_usage.api_call_count + EXCLUDED.api_call_count,
                   input_tokens = session_model_usage.input_tokens + EXCLUDED.input_tokens,
                   output_tokens = session_model_usage.output_tokens + EXCLUDED.output_tokens,
                   cache_read_tokens = session_model_usage.cache_read_tokens + EXCLUDED.cache_read_tokens,
                   cache_write_tokens = session_model_usage.cache_write_tokens + EXCLUDED.cache_write_tokens,
                   reasoning_tokens = session_model_usage.reasoning_tokens + EXCLUDED.reasoning_tokens,
                   estimated_cost_usd = session_model_usage.estimated_cost_usd + EXCLUDED.estimated_cost_usd,
                   actual_cost_usd = session_model_usage.actual_cost_usd + EXCLUDED.actual_cost_usd,
                   cost_status = COALESCE(EXCLUDED.cost_status, session_model_usage.cost_status),
                   cost_source = COALESCE(EXCLUDED.cost_source, session_model_usage.cost_source),
                   last_seen = EXCLUDED.last_seen"""


# The source-filter arm the SQLite module builds with ``?`` placeholders cannot be reused here: psycopg
# only binds ``%s``, so the pair is rebuilt on the PG dialect. The shape is the store's, not the
# database's: a source filter also matches the gateway's ``local`` rows that surface created
# (the ``created_source`` arm).
_PG_LOCAL_SURFACE = " OR ({a}source = 'local' AND {a}created_source = %s)"


def _pg_scoped(before: str, after: str = "", *, alias: str = "s.") -> tuple[str, str]:
    """(unfiltered, source-filtered) query pair sharing one body; the filtered arm takes two
    source parameters (the explicit source, then the ``created_source`` arm's)."""
    src = f" AND ({alias}source = %s{_PG_LOCAL_SURFACE.format(a=alias)})"
    return before + after, before + src + after


class PgUsageRepository:
    """Token/cost accounting and the usage reads on the bound profile's store schema; one class
    carries both conversation protocols (as the SQLite pair's single class does), unported verbs
    raise naming themselves (D8)."""

    def __init__(self, store: PostgresStore) -> None:
        self._store = store

    # ── writes ──────────────────────────────────────────────────────────────

    def record_auxiliary_usage(
        self, session_id: SessionId, task: str, *, model: Optional[str] = None, billing_provider: Optional[str] = None,
        billing_base_url: Optional[str] = None, tokens: TokenCounts = TokenCounts(),
        estimated_cost_usd: Optional[float] = None, api_call_count: int = 1,
    ) -> None:
        store = self._store
        if not str(session_id) or not task:
            return
        usage = {k: v for k, v in {
            "model": model, "billing_provider": billing_provider, "billing_base_url": billing_base_url,
            "billing_mode": None, "input_tokens": tokens.input_tokens, "output_tokens": tokens.output_tokens,
            "cache_read_tokens": tokens.cache_read_tokens, "cache_write_tokens": tokens.cache_write_tokens,
            "reasoning_tokens": tokens.reasoning_tokens, "estimated_cost_usd": estimated_cost_usd,
            "actual_cost_usd": None, "cost_status": None, "cost_source": None,
            "api_call_count": 1 if api_call_count is None else int(api_call_count),
        }.items() if k in _MODEL_USAGE_FIELDS}
        with translated():
            with store.tx() as conn:
                # FK to sessions.id: a missing row is inserted as the ``unknown`` placeholder; an
                # existing row keeps its surface, only a NULL profile_name is filled (never ``unknown``).
                exists = conn.execute("SELECT 1 FROM sessions WHERE id = %s", (str(session_id),)).fetchone()
                if exists is None:
                    conn.execute(
                        "INSERT INTO sessions (id, source, created_source, started_at) VALUES (%s, %s, %s, %s) "
                        "ON CONFLICT (id) DO NOTHING",
                        (str(session_id), "unknown", "unknown", time.time()))
                elif own_profile := store.owner_profile_name:
                    conn.execute(
                        "UPDATE sessions SET profile_name = %s WHERE id = %s AND profile_name IS NULL",
                        (own_profile, str(session_id)))
                self._record_model_usage(conn, str(session_id), task=task, **usage)

    def set_billing_route(self, session_id: SessionId, *, provider: str, base_url: str,
                          billing_mode: Optional[str] = None) -> None:
        store = self._store
        with translated():
            with store.tx() as conn:
                conn.execute(
                    """UPDATE sessions SET
                           billing_provider = %s,
                           billing_base_url = %s,
                           billing_mode = COALESCE(%s, billing_mode)
                       WHERE id = %s""",
                    (provider, base_url, billing_mode, str(session_id)))

    # ── reads ───────────────────────────────────────────────────────────────

    def recent_route(self, session_id: SessionId) -> Optional[RouteUsage]:
        store = self._store
        row = store.one(
            """SELECT model, billing_provider, billing_base_url, billing_mode, api_call_count
               FROM session_model_usage
              WHERE session_id = %s AND task = '' AND model <> 'unknown' AND billing_provider <> ''
              ORDER BY last_seen DESC, ctid DESC
              LIMIT 1""", str(session_id))
        if row is None:
            return None
        route = BillingRoute(model=row["model"], billing_provider=row["billing_provider"] or None,
                             billing_base_url=row["billing_base_url"] or None,
                             billing_mode=row["billing_mode"] or None)
        return RouteUsage(route=route, api_call_count=int(row["api_call_count"] or 0))

    def usage_totals(self, query: UsageTotalsQuery = UsageTotalsQuery()) -> UsageTotals:
        store = self._store
        where = ["parent_session_id IS NULL", "message_count >= %s"]
        params: list[Any] = [query.min_message_count]
        if not query.include_archived:
            where.append("COALESCE(archived, 0) = 0")
        row = store.one(
            f"""SELECT COALESCE(SUM(COALESCE(input_tokens, 0) + COALESCE(output_tokens, 0)), 0) AS tokens,
                       COALESCE(SUM(COALESCE(actual_cost_usd, estimated_cost_usd, 0)), 0) AS cost_usd
                FROM sessions
                WHERE {' AND '.join(where)}""", *params)
        if not row:
            return UsageTotals(tokens=0, cost_usd=0.0)
        return UsageTotals(tokens=int(row["tokens"] or 0),
                          cost_usd=float(row["cost_usd"] or 0.0))

    def auxiliary_usage_by_task(self, session_id: SessionId) -> Mapping[str, AuxiliaryTaskUsage]:
        store = self._store
        sid = str(session_id)
        if not sid:
            return {}
        chain = session_lineage_root_to_tip(store, sid)
        # An explicit /branch copy owns its spend: cut the walk at the nearest branch node.
        for i in range(len(chain) - 1, -1, -1):
            if self._is_explicit_branch_session(chain[i]):
                chain = chain[i:]
                break
        ph = ", ".join(["%s"] * len(chain))
        rows = store.all(
            f"""SELECT task,
                       COALESCE(SUM(api_call_count), 0) AS api_calls,
                       COALESCE(SUM(input_tokens), 0) AS input_tokens,
                       COALESCE(SUM(output_tokens), 0) AS output_tokens,
                       COALESCE(SUM(cache_read_tokens), 0) AS cache_read_tokens,
                       COALESCE(SUM(cache_write_tokens), 0) AS cache_write_tokens,
                       COALESCE(SUM(reasoning_tokens), 0) AS reasoning_tokens,
                       COALESCE(SUM(estimated_cost_usd), 0) AS estimated_cost_usd
                FROM session_model_usage
                WHERE session_id IN ({ph}) AND task != ''
                GROUP BY task""", *chain)
        return {
            str(row["task"]): AuxiliaryTaskUsage(
                task=str(row["task"]), api_call_count=int(row["api_calls"]),
                tokens=TokenCounts(
                    input_tokens=int(row["input_tokens"]), output_tokens=int(row["output_tokens"]),
                    cache_read_tokens=int(row["cache_read_tokens"]), cache_write_tokens=int(row["cache_write_tokens"]),
                    reasoning_tokens=int(row["reasoning_tokens"])),
                estimated_cost_usd=float(row["estimated_cost_usd"] or 0.0))
            for row in rows
        }

    def prompt_cache_clock(self, session_id: SessionId) -> Optional[PromptCacheClockRecord]:
        store = self._store
        row = store.one(
            """SELECT s.prompt_cache_clock,
                      CASE WHEN COALESCE(s.cache_read_tokens, 0) + COALESCE(s.cache_write_tokens, 0) > 0
                           THEN (SELECT MAX(m.timestamp) FROM messages m
                                 WHERE m.session_id = s.id AND m.role = 'assistant' AND m.active = 1) END,
                      (SELECT COUNT(*) FROM messages WHERE session_id = s.id AND active = 0 AND compacted = 1)
               FROM sessions s WHERE s.id = %s""", str(session_id))
        if not row:
            return None
        return PromptCacheClockRecord(clock=row["prompt_cache_clock"], last_assistant_at=row["case"],
                                     generation=int(row["count"]))

    def prompt_cache_generation(self, session_id: SessionId) -> int:
        row = self._store.scalar(_ARCHIVED_ROWS_SQL, str(session_id))
        return int(row) if row else 0

    # ── queue and barrier (synchronous on PG) ───────────────────────────────

    def queue_usage(self, session_id: SessionId, delta: UsageDelta) -> None:
        """Apply one usage delta (the SQLite queue's item); the PG store applies it straight through
        the counters transaction - no background writer to queue for. A failed apply is logged at
        WARNING and dropped, never raised into the caller (the SQLite queue's failure contract: a
        delta beside a session's deletion tombstone is the standing case)."""
        kwargs = _delta_kwargs(delta)
        try:
            self._update_token_counts(str(session_id), absolute=False, **kwargs)
        except Exception as exc:  # health: allow BLE001 -- accounting loss is logged, never raised into a turn
            logger.warning("async token accounting: apply failed (session=%s): %s", session_id, exc)

    def flush_usage(self, timeout: float = 5.0) -> bool:
        """The read barrier: nothing is queued on the PG store, so the barrier is already met."""
        return True

    # ── the counters transaction (the SQLite ``update_token_counts``) ──────

    def _update_token_counts(self, session_id: str, *, absolute: bool = False, **values) -> None:
        store = self._store
        source = values.pop("source", None)
        with translated():
            with store.tx() as conn:
                # A delayed accounting backfill must not recreate a deleted session beside its durable
                # tombstone (the SQLite ``insert_session_row`` guard): the apply fails and the queue's
                # failure path logs and drops the delta.
                if conn.execute("SELECT 1 FROM sessions WHERE id = %s", (session_id,)).fetchone() is None \
                        and conn.execute(
                            "SELECT 1 FROM state_meta WHERE key = %s",
                            (RETIRED_PREFIX + session_id,)).fetchone():
                    raise AuthorityRefusal("not_found")
                # Ensure the row exists: the UPDATE would otherwise silently affect 0 rows when the
                # creator's insert raced; the first backfill carries the real source, and a later
                # real source replaces the token-accounting guard's ``unknown`` placeholder
                # (the reference upsert's #111999 rule) - every other surface is kept. A routed delta
                # projects its model onto an unset session model (the reference upsert's
                # ``model = COALESCE(sessions.model, excluded.model)``).
                conn.execute(
                    "INSERT INTO sessions (id, source, created_source, started_at, model) "
                    "VALUES (%s, %s, %s, %s, %s) "
                    "ON CONFLICT (id) DO UPDATE SET "
                    "source = CASE WHEN sessions.source = 'unknown' "
                    "THEN COALESCE(EXCLUDED.source, 'unknown') ELSE sessions.source END, "
                    "model = COALESCE(sessions.model, EXCLUDED.model)",
                    (session_id, source or "unknown", source or "unknown", time.time(), values.get("model")))
                self._update_token_counts_in_transaction(conn, session_id, absolute=absolute, **values)

    def _update_token_counts_in_transaction(self, conn, session_id: str, *, absolute: bool = False,
                                            task: str = "", prompt_cache_clock: Optional[str] = None,
                                            **values) -> None:
        usage = {k: v for k, v in values.items() if k in _MODEL_USAGE_FIELDS}
        sql = _TOKEN_UPDATE_ABSOLUTE_SQL if absolute else _TOKEN_UPDATE_DELTA_SQL
        has_usage = bool(values.get("input_tokens") or values.get("output_tokens") or values.get("cache_read_tokens")
                         or values.get("cache_write_tokens") or values.get("reasoning_tokens")
                         or values.get("api_call_count") or values.get("estimated_cost_usd"))
        has_accounted_usage = bool(has_usage or values.get("actual_cost_usd"))
        model = values.get("model")
        billing_provider = values.get("billing_provider")
        billing_base_url = values.get("billing_base_url")
        billing_mode = values.get("billing_mode")
        estimated = values.get("estimated_cost_usd")
        actual = values.get("actual_cost_usd")
        params = (
            values.get("input_tokens") or 0, values.get("output_tokens") or 0,
            values.get("cache_read_tokens") or 0, values.get("cache_write_tokens") or 0,
            values.get("reasoning_tokens") or 0,
            estimated, actual, actual,
            values.get("cost_status"), values.get("cost_source"), values.get("pricing_version"),
            billing_provider if has_accounted_usage else None,
            billing_base_url if has_accounted_usage else None,
            billing_mode if has_accounted_usage else None, model if has_accounted_usage else None,
            values.get("api_call_count") or 0, session_id)
        record_model_usage = (not absolute) and has_usage
        row = conn.execute(
            "SELECT model, billing_provider, api_call_count FROM sessions WHERE id = %s",
            (session_id,)).fetchone()
        existing = dict(row) if row is not None else {}
        # The first accounted usage is the authoritative route; after that the row is kept as is.
        first_accounted_route = (
            not task and int(existing.get("api_call_count") or 0) == 0 and has_accounted_usage
            and bool(model) and bool(billing_provider)
            and (existing.get("model") != model or existing.get("billing_provider") != billing_provider))
        if first_accounted_route:
            conn.execute(
                """UPDATE sessions SET model = %s, billing_provider = %s,
                       billing_base_url = %s, billing_mode = %s WHERE id = %s""",
                (model, billing_provider, billing_base_url, billing_mode, session_id))
        conn.execute(sql, params)
        if prompt_cache_clock is not None:
            conn.execute("UPDATE sessions SET prompt_cache_clock = %s WHERE id = %s",
                         (prompt_cache_clock, session_id))
        if record_model_usage:
            self._record_model_usage(conn, session_id, task=task, **usage)

    def _record_model_usage(self, conn, session_id: str, *, task: str = "",
                            model: Optional[str] = None, billing_provider: Optional[str] = None,
                            billing_base_url: Optional[str] = None, billing_mode: Optional[str] = None,
                            input_tokens: int = 0, output_tokens: int = 0, cache_read_tokens: int = 0,
                            cache_write_tokens: int = 0, reasoning_tokens: int = 0,
                            estimated_cost_usd: Optional[float] = None, actual_cost_usd: Optional[float] = None,
                            cost_status: Optional[str] = None, cost_source: Optional[str] = None,
                            api_call_count: int = 0) -> None:
        """Accumulate a per-API-call delta into ``session_model_usage`` (the store's
        ``_record_model_usage``): a missing model/provider falls back to the session row, except
        aux rows (``task`` set), which never inherit the main-loop route."""
        row = conn.execute(
            "SELECT model, billing_provider, billing_base_url, billing_mode FROM sessions WHERE id = %s",
            (session_id,)).fetchone()
        sess = dict(row) if (row is not None and not task) else {}
        counts = [v or 0 for v in (input_tokens, output_tokens, cache_read_tokens,
                                  cache_write_tokens, reasoning_tokens)]
        now = time.time()
        conn.execute(_MODEL_USAGE_UPSERT_SQL, (
            session_id, model or sess.get("model") or "unknown",
            billing_provider or sess.get("billing_provider") or "",
            billing_base_url or sess.get("billing_base_url") or "",
            billing_mode or sess.get("billing_mode") or "", task or "", api_call_count or 0, *counts,
            float(estimated_cost_usd or 0.0), float(actual_cost_usd or 0.0), cost_status, cost_source, now, now))

    def _is_explicit_branch_session(self, session_id: str) -> bool:
        """A copied branch (the ``_branched_from`` marker) - the store's ``_is_explicit_branch_session``."""
        from .sessions import _parse_model_config
        if not session_id:
            return False
        row = self._store.one("SELECT model_config FROM sessions WHERE id = %s", session_id)
        return row is not None and bool(_parse_model_config(row["model_config"]).get("_branched_from"))

    # ── analytics reads (the SQLite ``UsageAnalytics`` SQL on the PG dialect) ──

    _GET_SESSIONS_ALL, _GET_SESSIONS_WITH_SOURCE = _pg_scoped(
        "SELECT id, source, model, started_at, ended_at, "
        "message_count, tool_call_count, input_tokens, output_tokens, "
        "cache_read_tokens, cache_write_tokens, billing_provider, "
        "billing_base_url, billing_mode, estimated_cost_usd, "
        "actual_cost_usd, cost_status, cost_source, api_call_count"
        " FROM sessions WHERE started_at >= %s",
        " ORDER BY started_at DESC",
        alias="")

    _GET_TOOL_CALLS_ALL, _GET_TOOL_CALLS_WITH_SOURCE = _pg_scoped(
        "SELECT m.session_id, m.tool_calls"
        " FROM messages m"
        " JOIN sessions s ON s.id = m.session_id"
        " WHERE s.started_at >= %s",
        " AND m.role = 'assistant' AND m.tool_calls IS NOT NULL",
    )
    _GET_SKILL_CALLS_ALL, _GET_SKILL_CALLS_WITH_SOURCE = _pg_scoped(
        "SELECT m.tool_calls, m.timestamp"
        " FROM messages m"
        " JOIN sessions s ON s.id = m.session_id"
        " WHERE s.started_at >= %s",
        " AND m.role = 'assistant' AND m.tool_calls IS NOT NULL"
        " AND (position('skill_view' IN m.tool_calls) > 0"
        " OR position('skill_manage' IN m.tool_calls) > 0)",
    )
    _GET_TOOL_NAMES_ALL, _GET_TOOL_NAMES_WITH_SOURCE = _pg_scoped(
        """SELECT m.session_id, m.tool_name, COUNT(*) AS count
             FROM messages m
             JOIN sessions s ON s.id = m.session_id
            WHERE s.started_at >= %s""",
        """  AND m.role = 'tool' AND m.tool_name IS NOT NULL
             GROUP BY m.session_id, m.tool_name""",
    )
    _GET_MESSAGE_STATS_ALL, _GET_MESSAGE_STATS_WITH_SOURCE = _pg_scoped(
        """SELECT COUNT(*) AS total_messages,
                  SUM(CASE WHEN m.role = 'user' THEN 1 ELSE 0 END) AS user_messages,
                  SUM(CASE WHEN m.role = 'assistant' THEN 1 ELSE 0 END) AS assistant_messages,
                  SUM(CASE WHEN m.role = 'tool' THEN 1 ELSE 0 END) AS tool_messages
             FROM messages m
             JOIN sessions s ON s.id = m.session_id
            WHERE s.started_at >= %s""",
    )
    _GET_MODEL_USAGE_ALL, _GET_MODEL_USAGE_WITH_SOURCE = _pg_scoped(
        "SELECT u.session_id, u.model, u.billing_provider, u.billing_base_url,"
        " u.api_call_count, u.input_tokens, u.output_tokens,"
        " u.cache_read_tokens, u.cache_write_tokens, u.reasoning_tokens,"
        " u.estimated_cost_usd, u.actual_cost_usd, u.cost_status,"
        " u.cost_source, u.billing_mode"
        " FROM session_model_usage u"
        " JOIN sessions s ON s.id = u.session_id"
        " WHERE s.started_at >= %s",
    )

    def _insight_rows(self, base: str, started_since: float, source: Optional[str]) -> list[dict[str, Any]]:
        """Rows of ``<base>_WITH_SOURCE`` or ``<base>_ALL``; a source filter also matches the
        gateway's ``local`` rows that surface created (the ``created_source`` arm)."""
        store = self._store
        if not source:
            return store.all(getattr(self, f"_{base}_ALL"), started_since)
        return store.all(getattr(self, f"_{base}_WITH_SOURCE"), started_since, source, source)

    def daily_usage(self, *, started_after: float) -> list[dict[str, Any]]:
        store = self._store
        rows = store.all(
            """SELECT to_char((to_timestamp(started_at) AT TIME ZONE 'UTC')::timestamp, 'YYYY-MM-DD') AS day,
                      SUM(input_tokens) AS input_tokens,
                      SUM(output_tokens) AS output_tokens,
                      SUM(cache_read_tokens) AS cache_read_tokens,
                      SUM(reasoning_tokens) AS reasoning_tokens,
                      COALESCE(SUM(estimated_cost_usd), 0) AS estimated_cost,
                      COALESCE(SUM(actual_cost_usd), 0) AS actual_cost,
                      COUNT(*) AS sessions,
                      SUM(COALESCE(api_call_count, 0)) AS api_calls
               FROM sessions WHERE started_at > %s
               GROUP BY day ORDER BY day""", started_after)
        return [dict(r) for r in rows]

    def session_usage_by_model(self, *, started_after: float) -> list[dict[str, Any]]:
        store = self._store
        rows = store.all(
            """SELECT model,
                      SUM(input_tokens) AS input_tokens,
                      SUM(output_tokens) AS output_tokens,
                      COALESCE(SUM(estimated_cost_usd), 0) AS estimated_cost,
                      COUNT(*) AS sessions,
                      SUM(COALESCE(api_call_count, 0)) AS api_calls
               FROM sessions WHERE started_at > %s AND model IS NOT NULL
               GROUP BY model ORDER BY SUM(input_tokens) + SUM(output_tokens) DESC""",
            started_after)
        return [dict(r) for r in rows]

    def session_usage_totals(self, *, started_after: float, model_rows_only: bool = False) -> dict[str, Any]:
        store = self._store
        extra = " AND model IS NOT NULL AND model != ''" if model_rows_only else ""
        row = store.one(
            """SELECT SUM(input_tokens) AS total_input,
                     SUM(output_tokens) AS total_output,
                     SUM(cache_read_tokens) AS total_cache_read,
                     SUM(reasoning_tokens) AS total_reasoning,
                     COALESCE(SUM(estimated_cost_usd), 0) AS total_estimated_cost,
                     COALESCE(SUM(actual_cost_usd), 0) AS total_actual_cost,
                     COUNT(*) AS total_sessions,
                     SUM(COALESCE(api_call_count, 0)) AS total_api_calls
               FROM sessions WHERE started_at > %s""" + extra, started_after)
        return dict(row) if row else {}

    def session_usage_by_route(self, *, started_after: float) -> list[dict[str, Any]]:
        store = self._store
        rows = store.all(
            """SELECT model,
                      billing_provider,
                      SUM(input_tokens) AS input_tokens,
                      SUM(output_tokens) AS output_tokens,
                      SUM(cache_read_tokens) AS cache_read_tokens,
                      SUM(reasoning_tokens) AS reasoning_tokens,
                      COALESCE(SUM(estimated_cost_usd), 0) AS estimated_cost,
                      COALESCE(SUM(actual_cost_usd), 0) AS actual_cost,
                      COUNT(*) AS sessions,
                      SUM(COALESCE(api_call_count, 0)) AS api_calls,
                      SUM(tool_call_count) AS tool_calls,
                      MAX(started_at) AS last_used_at,
                      AVG(input_tokens + output_tokens) AS avg_tokens_per_session
               FROM sessions WHERE started_at > %s AND model IS NOT NULL AND model != ''
               GROUP BY model, billing_provider
               ORDER BY SUM(input_tokens) + SUM(output_tokens) DESC""",
            started_after)
        return [dict(r) for r in rows]

    def session_tool_calls_by_route(self, *, started_after: float) -> list[dict[str, Any]]:
        store = self._store
        rows = store.all(
            """SELECT model, billing_provider, SUM(COALESCE(tool_call_count, 0)) AS tool_calls
               FROM sessions WHERE started_at > %s AND model IS NOT NULL AND model != ''
               GROUP BY model, billing_provider""",
            started_after)
        return [dict(r) for r in rows]

    def call_usage_by_route(self, *, started_after: float) -> list[dict[str, Any]]:
        store = self._store
        rows = store.all(
            """SELECT u.model,
                      u.billing_provider,
                      SUM(u.input_tokens) AS input_tokens,
                      SUM(u.output_tokens) AS output_tokens,
                      SUM(u.cache_read_tokens) AS cache_read_tokens,
                      SUM(u.reasoning_tokens) AS reasoning_tokens,
                      COALESCE(SUM(u.estimated_cost_usd), 0) AS estimated_cost,
                      COALESCE(SUM(u.actual_cost_usd), 0) AS actual_cost,
                      COUNT(DISTINCT u.session_id) AS sessions,
                      SUM(COALESCE(u.api_call_count, 0)) AS api_calls,
                      MAX(u.last_seen) AS last_used_at,
                      AVG(u.input_tokens + u.output_tokens) AS avg_tokens_per_session
               FROM session_model_usage u
               JOIN sessions s ON s.id = u.session_id
               WHERE s.started_at > %s AND u.model IS NOT NULL AND u.model != ''
                 AND u.task = ''
               GROUP BY u.model, u.billing_provider
               ORDER BY SUM(u.input_tokens) + SUM(u.output_tokens) DESC""",
            started_after)
        return [
            {**dict(r), "estimated_cost": float(r["estimated_cost"] or 0.0), "actual_cost": float(r["actual_cost"] or 0.0),
             "avg_tokens_per_session": float(r["avg_tokens_per_session"] or 0.0)}
            for r in rows]

    def auxiliary_usage_by_model_task(self, *, started_after: float) -> list[dict[str, Any]]:
        store = self._store
        rows = store.all(
            """SELECT u.model,
                      u.task,
                      u.billing_provider,
                      SUM(u.input_tokens) AS input_tokens,
                      SUM(u.output_tokens) AS output_tokens,
                      SUM(u.cache_read_tokens) AS cache_read_tokens,
                      SUM(u.reasoning_tokens) AS reasoning_tokens,
                      COALESCE(SUM(u.estimated_cost_usd), 0) AS estimated_cost,
                      COUNT(DISTINCT u.session_id) AS sessions,
                      SUM(COALESCE(u.api_call_count, 0)) AS api_calls,
                      MAX(u.last_seen) AS last_used_at
               FROM session_model_usage u
               JOIN sessions s ON s.id = u.session_id
               WHERE s.started_at > %s AND u.task != ''
               GROUP BY u.model, u.task, u.billing_provider
               ORDER BY SUM(u.input_tokens) + SUM(u.output_tokens) DESC""",
            started_after)
        return [
            {**dict(r), "estimated_cost": float(r["estimated_cost"] or 0.0)}
            for r in rows]

    def insight_sessions(self, *, started_since: float, source: Optional[str] = None) -> list[dict[str, Any]]:
        return self._insight_rows("GET_SESSIONS", started_since, source)

    def insight_tool_result_counts(self, *, started_since: float, source: Optional[str] = None) -> list[dict[str, Any]]:
        return self._insight_rows("GET_TOOL_NAMES", started_since, source)

    def insight_tool_call_payloads(self, *, started_since: float, source: Optional[str] = None) -> list[dict[str, Any]]:
        return self._insight_rows("GET_TOOL_CALLS", started_since, source)

    def insight_skill_call_payloads(self, *, started_since: float, source: Optional[str] = None) -> list[dict[str, Any]]:
        return self._insight_rows("GET_SKILL_CALLS", started_since, source)

    def insight_message_stats(self, *, started_since: float, source: Optional[str] = None) -> dict[str, Any]:
        rows = self._insight_rows("GET_MESSAGE_STATS", started_since, source)
        return rows[0] if rows else {}

    def insight_model_usage(self, *, started_since: float, source: Optional[str] = None) -> list[dict[str, Any]]:
        return self._insight_rows("GET_MODEL_USAGE", started_since, source)


add_stubs(PgUsageRepository, UsageRepository, implemented=(
    "record_auxiliary_usage", "set_billing_route", "recent_route", "usage_totals",
    "auxiliary_usage_by_task", "prompt_cache_clock", "prompt_cache_generation", "queue_usage", "flush_usage"))
add_stubs(PgUsageRepository, UsageAnalytics, implemented=(
    "daily_usage", "session_usage_by_model", "session_usage_totals", "session_usage_by_route",
    "session_tool_calls_by_route", "call_usage_by_route", "auxiliary_usage_by_model_task",
    "insight_sessions", "insight_tool_result_counts", "insight_tool_call_payloads",
    "insight_skill_call_payloads", "insight_message_stats", "insight_model_usage"))


def _register() -> None:
    register_repository("usage", lambda store: PgUsageRepository(store))
    register_repository("usage_analytics", lambda store: PgUsageRepository(store))


_register()

"""Shared session-lineage SQL fragments for the PostgreSQL repositories (stage 1).

The PostgreSQL twins of the SQLite store's lineage predicates (``persistence.sqlite._schema_ddl`` /
``_store/sql_common``): JSON marker lookups on the TEXT ``model_config`` column through
``json_valid``/``json_extract_path`` (never throwing), the reset/branch child predicates, the
compression-continuation edge, the recency expression and the lineage CTE. One module so the
lineage, usage and housekeeping repositories cannot drift apart.
"""

from __future__ import annotations

from persistence.sqlite.conversation._store.rules import EPOCH_MAX, EPOCH_MIN

__all__ = [
    "json_marker_sql",
    "json_marker_absent_sql",
    "legacy_reset_child_sql",
    "reset_child_sql",
    "non_continuation_child_sql",
    "continuation_edge_sql",
    "session_last_active_sql",
    "chain_step_sql",
    "lineage_cte_sql",
    "session_lineage_root_to_tip",
]

_RESET_END_REASONS = ("session_reset", "session_switch", "idle", "daily", "suspended", "resume_pending_expired")
_RESET_END_REASONS_SQL = ", ".join(f"'{reason}'" for reason in _RESET_END_REASONS)


def json_marker_sql(expression: str, key: str) -> str:
    """A non-throwing lookup of marker *key* in a JSON TEXT column: the marker's text value, NULL
    when the column is invalid JSON or the key is absent (the SQLite ``_sql_json_extract`` shape on
    the PG dialect; a JSON string value reads back bare). ``::jsonb ->>`` yields the bare text
    (``json_extract_path(...)::text`` would yield the quoted JSON form, ``'"value"'``)."""
    return (
        f"(CASE WHEN json_text_valid({expression}) "
        f"THEN {expression}::jsonb ->> '{key}' END)"
    )


def json_marker_absent_sql(expression: str, key: str) -> str:
    """True when the marker is not set (the SQLite ``<marker> IS NULL`` predicate)."""
    return f"{json_marker_sql(expression, key)} IS NULL"


def legacy_reset_child_sql(alias: str = "s") -> str:
    """Pre-marker reset-continuation heuristic: *alias* rides its parent's exact routing key and
    the parent ended at a reset boundary (the SQLite ``_legacy_reset_child_sql`` shape)."""
    return (
        f"EXISTS (SELECT 1 FROM sessions p WHERE p.id = {alias}.parent_session_id "
        f"AND p.end_reason IN ({_RESET_END_REASONS_SQL}) "
        f"AND {alias}.session_key IS NOT NULL AND {alias}.session_key != '' "
        f"AND {alias}.session_key = p.session_key)"
    )


def reset_child_sql(alias: str = "s") -> str:
    """A /new reset child: the ``_reset_from`` marker, or the legacy same-key heuristic (the
    SQLite ``_RESET_CHILD_SQL`` predicate)."""
    return (
        f"({json_marker_sql(alias + '.model_config', '_reset_from')} IS NOT NULL "
        f"OR {legacy_reset_child_sql(alias)})"
    )


def non_continuation_child_sql(child: str = "", parent: str = "?") -> str:
    """The ``  AND ...`` clauses rejecting children that are NOT compression continuations of
    *parent* (branch/delegate/reset forks, tool sessions). Markers are bound to the parent id:
    continuations inherit ``model_config`` verbatim, so a marker naming another row is inherited,
    not a fork (the SQLite ``_non_continuation_child_sql`` shape; *child* is the column prefix,
    *parent* the parent-id SQL expression)."""
    clauses = "".join(
        f"  AND COALESCE({json_marker_sql(f'{child}model_config', marker)}, '') != {parent}\n"
        for marker in ("_branched_from", "_delegate_from", "_reset_from")
    )
    return clauses + f"  AND COALESCE({child}source, '') != 'tool'\n"


def continuation_edge_sql(child: str = "c", parent: str = "p") -> str:
    """*child* continues compression-ended *parent* (same predicate as the compression chain step)."""
    return f"{parent}.end_reason = 'compression'\n" + non_continuation_child_sql(f"{child}.", f"{parent}.id")


def session_last_active_sql(alias: str = "s") -> str:
    """Session recency: the freshest of ``last_activity_at``, the newest in-window message
    timestamp and ``started_at`` (the SQLite ``_sql_session_last_active`` shape, in-window clamp)."""
    in_window = f"BETWEEN {EPOCH_MIN} AND {EPOCH_MAX}"
    return (
        f"COALESCE((SELECT MAX(v) FROM (SELECT {alias}.last_activity_at AS v UNION ALL "
        f"(SELECT MAX(m.timestamp) FROM messages m WHERE m.session_id = {alias}.id "
        f"AND m.timestamp {in_window})) _lv "
        f"WHERE _lv.v {in_window}), "
        f"(CASE WHEN {alias}.started_at {in_window} THEN {alias}.started_at END))"
    )


def chain_step_sql() -> str:
    """The preferred continuation child of a compression-ended parent (the SQLite
    ``_CHAIN_STEP_SQL`` shape): exclude branch/delegate/reset/tool children; prefer a child that
    continues the chain, then a live child, over stale closed siblings."""
    child = "child"
    return f"""
                SELECT child.id
                FROM sessions parent
                JOIN sessions child ON child.parent_session_id = parent.id
                WHERE parent.id = %s
                  AND parent.end_reason = 'compression'
                  AND {json_marker_absent_sql('child.model_config', '_branched_from')}
                  AND {json_marker_absent_sql('child.model_config', '_delegate_from')}
                  AND NOT ({reset_child_sql(child)})
                  AND COALESCE(child.source, '') != 'tool'
                ORDER BY
                  CASE
                    WHEN child.end_reason = 'compression' THEN 0
                    WHEN child.ended_at IS NULL THEN 1
                    ELSE 2
                  END,
                  {session_last_active_sql('child')} DESC,
                  child.started_at DESC,
                  child.id DESC
                LIMIT 1
                """


def lineage_cte_sql() -> str:
    """``WITH RECURSIVE`` over the compression lineage of two ids (first ``%s`` = the session,
    second ``%s`` = its conversation root seed): ancestors through compression edges above the
    session, and its compression descendants (the SQLite ``_LINEAGE_CTE_SQL`` shape; the CTE body
    binds both seeds in that order)."""
    edge = continuation_edge_sql("child", "parent")
    return f"""
            WITH RECURSIVE
              ancestors(id) AS (
                SELECT %s
                UNION
                SELECT parent.id
                FROM ancestors a
                JOIN sessions child ON child.id = a.id
                JOIN sessions parent ON parent.id = child.parent_session_id
                WHERE {edge}
              ),
              descendants(id) AS (
                SELECT %s
                UNION
                SELECT child.id
                FROM descendants d
                JOIN sessions child ON child.parent_session_id = d.id
                WHERE {edge}
              ),
              lineage(id) AS (
                SELECT id FROM ancestors UNION SELECT id FROM descendants
              )
            """


def session_lineage_root_to_tip(store, session_id: str, cap: int = 100) -> list[str]:
    """The raw ``parent_session_id`` walk from *session_id* up to its root, root-first
    (the SQLite ``_session_lineage_root_to_tip`` shape)."""
    if not session_id:
        return [session_id]
    chain: list[str] = []
    current = session_id
    while current and current not in chain and len(chain) < cap:
        chain.append(current)
        row = store.one("SELECT parent_session_id FROM sessions WHERE id = %s", current)
        if row is None:
            break
        current = row["parent_session_id"]
    return list(reversed(chain)) or [session_id]

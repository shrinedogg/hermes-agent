"""The session timeline and prompt pages on PostgreSQL (the SQLite ``timeline`` port). The PG DDL
triggers keep the display index complete for every display row, so the reads always take the indexed
projection (the SQLite legacy window-function branch, which registered Python identity functions, has
no PG twin); the prompt previews, which SQLite computes in a registered ``timeline_preview`` function,
are computed in Python instead (the statement fetches the page's rows, the previews filter in Python)."""

from __future__ import annotations

import re
from typing import Any, Optional

from session_rules.turns import project_compaction_message_for_display, user_originated_turn_view

from .common import (
    DISPLAY_VISIBLE_SQL, not_a_later_tool_copy_sql, _decode_content, _decode_payloads, _row_to_message_dict,
)

__all__ = ["get_session_messages_around", "get_session_timeline"]

_SYNTHETIC_PROMPT = re.compile(
    r"^\s*(?:\[IMPORTANT: Background process |\[ASYNC (?:DELEGATION )?(?:BATCH )?COMPLETE\b|"
    r"A background fan-out of \d+ subagent\(s\) you dispatched earlier has finished\.|"
    r"A background subagent you dispatched earlier has finished\.)",
    re.IGNORECASE,
)


def _prompt_preview(row: dict[str, Any]) -> str:
    """The timeline preview of one user display row (the SQLite ``timeline_preview`` UDF port): the
    compaction projection's live text, trimmed to 120 chars; empty for scaffolding and synthetic
    background notices (the row's payload columns already decoded)."""
    message = project_compaction_message_for_display({
        "role": "user", "content": _decode_content(row["content"]),
        "display_kind": row["display_kind"], "_compressed_summary": bool(row["_compressed_summary"]),
    })
    if message is None or user_originated_turn_view(message) is None:
        return ""
    content = message.get("content")
    if isinstance(content, list):
        content = " ".join(
            part if isinstance(part, str) else part.get("text", "")
            for part in content if isinstance(part, (str, dict)))
    if not isinstance(content, str):
        return ""
    text = " ".join(content.split())
    if not text or _SYNTHETIC_PROMPT.match(text):
        return ""
    return text if len(text) <= 120 else text[:119].rstrip() + "…"


def _display_rows_cte(session_id: str, *, users_only: bool = False) -> tuple[str, list]:
    """``WITH display_rows AS (...)``: representative ids and their stable first-row order, never bodies
    (the indexed branch of the SQLite ``_display_rows_sql``, ``%s``-bound)."""
    filters = (" AND role = 'user'" if users_only else "") + DISPLAY_VISIBLE_SQL
    sql = (
        f"""WITH display_rows AS (
            SELECT (SELECT candidate.id FROM messages candidate
                    WHERE candidate.session_id = %s
                      AND candidate.display_order = m.display_order
                      AND (candidate.active = 1 OR candidate.compacted = 1){DISPLAY_VISIBLE_SQL}{not_a_later_tool_copy_sql("candidate")}
                    ORDER BY candidate.active DESC, candidate.id DESC LIMIT 1) AS row_id,
                   m.display_order AS sort_id
            FROM messages m WHERE m.session_id = %s AND (m.active = 1 OR m.compacted = 1){filters}{not_a_later_tool_copy_sql("m")}
            GROUP BY m.display_order
        )""")
    return sql, [session_id, session_id]


def get_session_messages_around(store, session_id: str, row_id: int, *, limit: int = 120) -> Optional[dict[str, Any]]:
    """Read at most *limit* display rows starting at an exact human prompt (the SQLite
    ``get_session_messages_around`` port; existence probes/counts carry ids only, full payloads are
    fetched only for the selected bounded page)."""
    cte, cte_params = _display_rows_cte(session_id)
    anchor = store.one(
        "SELECT content, display_kind, _compressed_summary FROM messages "
        "WHERE session_id = %s AND id = %s AND role = 'user' AND (active = 1 OR compacted = 1)",
        session_id, row_id)
    if anchor is None or not _prompt_preview(_decode_payloads(anchor)):
        return None
    start_row = store.one(f"{cte} SELECT sort_id FROM display_rows WHERE row_id = %s", *cte_params, row_id)
    if start_row is None:
        return None
    start = start_row["sort_id"]
    counts = store.one(
        f"{cte} SELECT COUNT(*) AS total, COUNT(*) FILTER (WHERE sort_id < %s) AS offset FROM display_rows",
        *cte_params, start)
    total = int(counts["total"]) if counts is not None else 0
    offset = int(counts["offset"]) if counts is not None else 0
    rows = store.all(
        f"{cte} SELECT m.* FROM (SELECT row_id, sort_id FROM display_rows"
        f" WHERE sort_id >= %s ORDER BY sort_id LIMIT %s) AS page"
        f" JOIN messages m ON m.id = page.row_id ORDER BY page.sort_id",
        *cte_params, start, limit)
    messages = [_row_to_message_dict(row, warn_context="timeline jump", summary_flag=True) for row in rows]
    return {"messages": messages, "pagination": {
        "row_id": row_id, "limit": limit, "returned": len(messages), "order": "oldest",
        "offset": offset, "total": total,
        "has_older": offset > 0,
        "has_newer": offset + len(messages) < total,
    }}


def get_session_timeline(store, session_id: str, *, limit: int = 500, after_row_id: int = 0) -> dict[str, Any]:
    """Chronological prompts (the SQLite ``get_session_timeline`` port; the cursor is the first physical
    row id of a logical turn, the previews filter in Python over the segment's display rows)."""
    cte, cte_params = _display_rows_cte(session_id, users_only=True)
    rows = store.all(
        f"{cte} SELECT m.id AS row_id, dr.sort_id, m.timestamp, m.content, m.display_kind, m._compressed_summary"
        f" FROM display_rows AS dr JOIN messages m ON m.id = dr.row_id ORDER BY dr.sort_id",
        *cte_params)
    eligible: list[dict[str, Any]] = []
    for row in rows:
        preview = _prompt_preview(_decode_payloads(row))
        if preview:
            eligible.append({"row_id": row["row_id"], "sort_id": row["sort_id"],
                            "timestamp": row["timestamp"], "preview": preview})
    total = len(eligible)
    filtered = [entry for entry in eligible if entry["sort_id"] > after_row_id]
    page = filtered[:limit]
    has_more = len(filtered) > limit
    return {
        "entries": [{"row_id": int(entry["row_id"]), "preview": entry["preview"], "timestamp": entry["timestamp"]}
                    for entry in page],
        "pagination": {"limit": limit, "after_row_id": after_row_id, "returned": len(page),
                       "total": total, "has_more": has_more,
                       "next_cursor": page[-1]["sort_id"] if has_more else None},
    }

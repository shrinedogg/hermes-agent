"""``SessionMaintenance`` on PostgreSQL: the prune selection and delete, the ghost and orphan sweeps, the idle
archive, backend heartbeats, the export bound, the session projection digest and the retained-worktree reads,
each one write transaction or read through the store. The export and import verbs come from the
``SessionPortability`` contract (ported inline here, the SQLite ``_portability.py`` twin).

Post-commit effects of a delete (the delete hook, retired media, transcript files) are not run here: callers run
them over the ids ``prune_sessions`` and ``prune_empty_ghost_sessions`` return. The stage-1 deletion fence
publishes the retirement markers and tombstones the ledger rows (the contract's import-refusal and child
orphaning semantics); the receipt scrubs that depend on repos still unported on PostgreSQL (the terminal
result store, the local creation ledger's deep receipts, the media spillover) are recorded but not run."""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

from persistence.authority.vocabulary import RETRY_HORIZON_S
from persistence.conversation.sessions import CANONICAL_BOT_CHAT_TITLE
from persistence.core.errors import AuthorityRefusal, Conflict, ExportTooLarge, PermanentStoreError
from persistence.core.identity import SessionId
from persistence.housekeeping.maintenance import (
    IMPORT_MAX_MESSAGES_PER_SESSION, IMPORT_MAX_SESSIONS, IMPORT_MAX_SESSION_BYTES,
    IMPORT_MAX_TOTAL_BYTES, IMPORT_MAX_TOTAL_MESSAGES, SESSION_PROJECTION_FIELDS,
    ForeignImportResult, ForeignOrigin, ForeignTurn, ImportResult, PruneCandidate, PruneCriteria,
    PruneReclaimEstimate, PruneResult, SessionExport, SessionExportRow, SessionLineageExport,
    SessionMaintenance, TreeServeFacts,
)
from session_rules.session_ids import new_session_id
from utils import safe_json_loads
from psycopg import errors as pg_errors

from .._backend import PostgresStore, register_repository, translated
from .._lineage_common import json_marker_sql, session_last_active_sql
from ..conversation.lineage import _INSERT_MESSAGE_SQL
from ..conversation.messages import _payload_str, _row_to_message_dict, _transcript_row
from .._stubs import add_stubs
from persistence.sqlite.conversation._store.local import (
    API_BINDING_PREFIX, API_DECLARED_PREFIX, POLICY_PREFIX, _policy_target)
from persistence.sqlite.conversation._store.messages_sql import _parse_tool_calls, _tool_calls_count
from persistence.sqlite.conversation._store.portability import _SQLITE_INT_MAX, _SQLITE_INT_MIN
from persistence.sqlite.conversation._store.rules import coerce_epoch
from persistence.sqlite.conversation._store.retirement import RETIRED_PREFIX
from persistence.sqlite.conversation._store.terminal import (
    ADMISSION_PREFIX, RESULT_PREFIX, WORKER_PREFIX, identity_key)
from persistence.sqlite.gateway._routes import ROUTE_FENCE_PREFIX

__all__ = ["PgSessionMaintenance"]

# The prune and archive selection SQL (the SQLite ``_LAST_ACTIVE_SQL`` / token / cost fragments on the
# PG dialect; the recency read carries the window filter, so the store's out-of-window rows match nothing).
_LAST_ACTIVE_SQL = session_last_active_sql("s")
_TOKENS_SQL = "(COALESCE(s.input_tokens, 0) + COALESCE(s.output_tokens, 0))"
_COST_SQL = "COALESCE(s.actual_cost_usd, s.estimated_cost_usd, 0)"


def _id_chunks(ids, size: int = 900):
    """Yield *ids* (any iterable) as lists of at most *size* elements (the SQLite bulk-prune chunking)."""
    ids = list(ids)
    for start in range(0, len(ids), size):
        yield ids[start:start + size]


def _placeholders(count: int) -> str:
    """``%s,%s,...`` for *count* bound parameters."""
    return ",".join("%s" for _ in range(count))


def _escape_like(text: str) -> str:
    """Escape LIKE wildcards (``%``, ``_``) so derived text matches literally; pair with ``ESCAPE '\\\\'``."""
    return text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _like(value: str) -> str:
    return f"%{_escape_like(value.lower())}%"


def _export_message_dict(row: Mapping[str, Any], *, summary_flag: bool = False) -> dict[str, Any]:
    """``_row_to_message_dict`` over the export's raw ``SELECT *`` rows: the BYTEA payload columns decode
    to the reference's text shape before the row dict is built (``content`` stays a str; the other
    binary columns still drop)."""
    return _row_to_message_dict({**row, "content": _payload_str(row.get("content"))}, summary_flag=summary_flag)


def _cwd_prefix_filter(value: str):
    """The SQLite ``_cwd_prefix_clause`` on the PG dialect: the raw ``=`` arm plus the escaped ``/`` and
    ``\\`` separator arms."""
    prefix = value.rstrip("/\\") or value
    esc = _escape_like(prefix)
    return ("(s.cwd = %s OR s.cwd LIKE %s ESCAPE '\\' OR s.cwd LIKE %s ESCAPE '\\')",
            [prefix, f"{esc}/%", f"{esc}\\%"])


def _one(clause: str, conv=None):
    return lambda v: ([clause], [conv(v) if conv else v])


# Prune/archive filters in evaluation order: (kwarg, applies-when, builder -> (clauses, params)).
# ``applies-when``: "notnone" (numeric/time bounds; 0 is a real bound) or "truthy" ("" = unset).
_PRUNE_FILTERS = (
    # Orphan-swept rows age from the sweep, not old activity, or the next prune deletes them before recovery.
    ("last_active_before", "notnone", lambda v: (
        [_LAST_ACTIVE_SQL + " < %s",
         "(COALESCE(s.end_reason, '') != 'startup_orphan_reap' OR s.ended_at < %s)"],
        [v, v])),
    ("last_active_after", "notnone", _one(_LAST_ACTIVE_SQL + " >= %s")),
    ("started_before", "notnone", _one("s.started_at < %s")),
    ("started_after", "notnone", _one("s.started_at >= %s")),
    ("source", "truthy", _one("s.source = %s")),
    ("title_like", "truthy", _one("LOWER(COALESCE(s.title, '')) LIKE %s ESCAPE '\\'", _like)),
    ("end_reason", "truthy", _one("s.end_reason = %s")),
    ("cwd_prefix", "truthy", _cwd_prefix_filter),
    ("min_messages", "notnone", _one("s.message_count >= %s")),
    ("max_messages", "notnone", _one("s.message_count <= %s")),
    ("model_like", "truthy", _one("LOWER(COALESCE(s.model, '')) LIKE %s ESCAPE '\\'", _like)),
    ("provider", "truthy", _one("LOWER(COALESCE(s.billing_provider, '')) = %s", str.lower)),
    ("user_id", "truthy", _one("s.user_id = %s")),
    ("chat_id", "truthy", _one("s.chat_id = %s")),
    ("chat_type", "truthy", _one("s.chat_type = %s")),
    ("branch_like", "truthy", _one("LOWER(COALESCE(s.git_branch, '')) LIKE %s ESCAPE '\\'", _like)),
    ("min_tokens", "notnone", _one(_TOKENS_SQL + " >= %s")),
    ("max_tokens", "notnone", _one(_TOKENS_SQL + " <= %s")),
    ("min_cost", "notnone", _one(_COST_SQL + " >= %s")),
    ("max_cost", "notnone", _one(_COST_SQL + " <= %s")),
    ("min_tool_calls", "notnone", _one("COALESCE(s.tool_call_count, 0) >= %s")),
    ("max_tool_calls", "notnone", _one("COALESCE(s.tool_call_count, 0) <= %s")),
)

# Child ``c`` continues compression-ended ``p`` (same predicate as compression's child lookup; the
# JSON marker reads degrade to NULL on invalid ``model_config`` instead of raising).
_CONTINUATION_EDGE_SQL = (
    "p.end_reason = 'compression'\n"
    + "".join(
        f"  AND COALESCE({json_marker_sql(f'c.model_config', marker)}, '') != p.id\n"
        for marker in ("_branched_from", "_delegate_from", "_reset_from"))
    + "  AND COALESCE(c.source, '') != 'tool'\n")


def _continued_ancestors_sql(candidates_where: str) -> str:
    """Compression ancestors of every row *candidates_where* (alias ``s``) does not select."""
    return ("WITH RECURSIVE kept(id) AS ("
            " SELECT p.id FROM sessions c JOIN sessions p ON p.id = c.parent_session_id"
            f" WHERE {_CONTINUATION_EDGE_SQL}"
            f" AND NOT EXISTS (SELECT 1 FROM sessions s WHERE s.id = c.id AND {candidates_where})"
            " UNION"
            " SELECT p.id FROM kept k JOIN sessions c ON c.id = k.id JOIN sessions p ON p.id = c.parent_session_id"
            f" WHERE {_CONTINUATION_EDGE_SQL}"
            ") SELECT id FROM kept")


# A pin covers the whole conversation, but a store can hold a pinned segment whose later
# continuations were published unpinned; those still belong to the pinned chat.
_PINNED_TAIL_SQL = ("WITH RECURSIVE tail(id) AS ("
                    " SELECT c.id FROM sessions c JOIN sessions p ON p.id = c.parent_session_id"
                    f" WHERE COALESCE(p.pinned, 0) = 1 AND {_CONTINUATION_EDGE_SQL}"
                    " UNION"
                    " SELECT c.id FROM tail t JOIN sessions p ON p.id = t.id JOIN sessions c ON c.parent_session_id = p.id"
                    f" WHERE {_CONTINUATION_EDGE_SQL}"
                    ") SELECT id FROM tail")


def _not_pinned_sql(alias: str = "s") -> str:
    """Predicate sparing pinned rows and the unpinned continuations a pinned segment covers."""
    return f"COALESCE({alias}.pinned, 0) = 0 AND {alias}.id NOT IN ({_PINNED_TAIL_SQL})"


_PRUNE_FILTER_NAMES = frozenset(name for name, _, _ in _PRUNE_FILTERS) | {"archived", "include_pinned", "lineage_tips_only"}


def _prune_filter_where(*, archived: Optional[bool] = None, include_pinned: bool = False,
                       lineage_tips_only: bool = False, include_open: bool = False,
                       **filters) -> tuple[str, list]:
    """Shared WHERE clause for bulk prune/archive selection (alias ``s``): ``_PRUNE_FILTERS``
    AND together, only ended sessions are ever candidates, ``archived`` is tri-state
    (None = both), ``*_like`` are case-insensitive substrings, the rest exact.
    ``lineage_tips_only`` (bulk archive) drops compression ancestors: they are archived with
    their tip, never on their own age, and the canonical Bot Chat (the idle sweep's exemption).
    ``include_open`` lifts the ended guard for reversible archive only, never for prune/delete."""
    unknown = set(filters) - _PRUNE_FILTER_NAMES
    if unknown:
        raise TypeError("SessionMaintenance._prune_filter_where() got an unexpected "
                        f"keyword argument {min(unknown)!r}")
    clauses = [] if include_open else ["s.ended_at IS NOT NULL"]
    if lineage_tips_only:
        clauses.append("COALESCE(s.end_reason, '') <> 'compression'")
        # Bulk archive is housekeeping, like the idle sweep: it never retires the canonical Bot Chat (archiving
        # releases its name). Only a per-chat Archive may. Matched by title alone: compression moves the title
        # onto a visible continuation tip.
        clauses.append("COALESCE(s.title, '') <> 'Bot Chat'")
    params: list = []
    for name, applies, build in _PRUNE_FILTERS:
        value = filters.get(name)
        if (value is not None) if applies == "notnone" else bool(value):
            new_clauses, new_params = build(value)
            clauses.extend(new_clauses)
            params.extend(new_params)
    if isinstance(archived, bool):
        clauses.append(f"s.archived = {int(archived)}")
    # Pinned is a durable "keep" flag: bulk prune/delete/archive exclude pinned rows unless opted in.
    if not include_pinned:
        clauses.append(_not_pinned_sql())
    return " AND ".join(clauses), params


def _prune_where(older_than_days, source, filters, *, whole_lineages: bool = False) -> tuple[str, list]:
    """Translate the legacy age window into the shared activity filter, then build WHERE.
    ``whole_lineages`` (prune) keeps a compression ancestor while any continuation after it
    is unmatched."""
    if (older_than_days is not None and filters.get("last_active_before") is None
            and filters.get("started_before") is None):
        if older_than_days < 0:
            raise ValueError(
                f"older_than_days must be >= 0, got {older_than_days!r}: a negative "
                "retention builds a future cutoff that matches every ended session.")
        filters["last_active_before"] = time.time() - (older_than_days * 86400)
    where, params = _prune_filter_where(source=source, **filters)
    if not whole_lineages:
        return where, params
    # A compressed-away segment ages with its conversation, not on its own: while any later
    # segment stays, deleting it would cut the start off a chat that is still in use.
    return f"{where} AND s.id NOT IN ({_continued_ancestors_sql(where)})", [*params, *params]


def _row_bytes_sql(conn) -> str:
    """Stored size of one ``messages`` row: the byte length of every text/blob column (read from
    the live schema, so a column added later is counted without touching this)."""
    cols = [row["column_name"] for row in conn.execute(
        "SELECT column_name, data_type FROM information_schema.columns"
        " WHERE table_schema = current_schema() AND table_name = 'messages'").fetchall()
        if row["data_type"] in ("text", "bytea")]
    return " + ".join(f'COALESCE(LENGTH(m."{c}"), 0)' for c in cols) or "0"


# The worktree reads wait at most this long on a busy store (less than the store's default read wait).
_WORKTREE_READ_BUSY_S = 2.0


def _inside_tree(worktree_path: str):
    """``inside(path) -> 0/1``: *path* is the tree or under it (native path identity: Windows is
    case-insensitive, either separator; the tree's own spelling and its resolved one)."""
    def key(value) -> str:
        return os.path.normcase(str(value)).replace("\\", "/").rstrip("/")

    roots = {key(Path(worktree_path)), key(Path(worktree_path).resolve())}

    def inside(cwd) -> int:
        return int(bool(cwd) and any(key(cwd) == r or key(cwd).startswith(r + "/") for r in roots))
    return inside


# ── the deletion fence (the SQLite ``conversation._store.retirement`` / ``housekeeping._retirement``
#    twins on the PG dialect) ────────────────────────────────────────────────────────
#
# The stage-1 fence publishes the marker + ledger tombstones the contract reads: the retired-session
# marker (a user delete's, permanent; a sweep's, expiring at the retry horizon), the routing rows
# whose entries name the deleted ids, the terminal admission/worker tombstones in ``state_meta``,
# the mutation-receipt content scrub, and the owner-receipt deletion. The scrubs that need repos
# still unported on PostgreSQL (the terminal result store, the media spillover, the local creation
# ledger's deep receipts) are left to the porting of those repos; their absence is a no-op here
# because those repos have not written any such rows yet.

_LIVE_LEDGER_SQL = (
    "SELECT 1 FROM session_admissions WHERE target_session_id = %s AND status IN ('queued','started','unknown') "
    "UNION ALL SELECT 1 FROM worker_executions WHERE session_id = %s "
    "AND status IN ('registered','running','unknown') LIMIT 1")

# A deleted session's receipt result keeps the exact-retry outcome, not the transcript copies.
_TRANSCRIPT_FIELDS = frozenset({'messages', 'last_reasoning', 'tools', '_turn_user_row'})

_MUTATION_TRANSCRIPT_FIELDS = ('target_message', 'summary', 'title')

_DERIVED_LOCAL_PREFIXES = ('bot-chat:', 'a2a:')

_DERIVED_API_SQL = ("(source = 'bot_room' OR (source = 'api_server' AND id ~ '^api-[0-9a-f]{16}$'))")


def _retired_in(conn, session_id: str) -> bool:
    """The deletion fence's read: the retired-session marker for *session_id*."""
    return conn.execute(
        "SELECT 1 FROM state_meta WHERE key = %s", (RETIRED_PREFIX + session_id,)).fetchone() is not None


def _local_conversations(conn):
    """``(members, derived)``: member id -> every id of its local conversation (the creation id
    (policy, FIFO, generation), each receipt lineage segment and the current target), and the ids
    of every derived (Bot Chat, A2A) conversation. One prefix scan, not a lookup per swept id.
    Malformed receipts name nothing."""
    members, derived = {}, set()
    for (raw,) in [(row["value"],) for row in conn.execute(
            "SELECT value FROM state_meta WHERE key LIKE %s", (POLICY_PREFIX + "*",)).fetchall()]:
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
    """The ids among *session_ids* of derived API sessions."""
    if not session_ids:
        return set()
    rows = conn.execute(
        f"SELECT id FROM sessions WHERE {_DERIVED_API_SQL} AND id = ANY(%s)",
        (list(session_ids),)).fetchall()
    return {row["id"] for row in rows}


def _mark_retired(conn, key: str, *, expires: bool, **fields) -> None:
    """Publish (or replace) the deletion marker at *key*: the permanent user-delete marker holds only
    ``retired_at`` (+ ``own``), the sweep marker adds ``expires`` and ages out at the retry horizon.
    Re-retiring an id recreated beside its tombstone never aborts; a sweep marker never shortens a
    permanent one, and a permanent marker without ``own`` stays unscoped (the SQLite ``_mark_retired``)."""
    value = {'retired_at': time.time(), **fields}
    if expires:
        conn.execute(
            "INSERT INTO state_meta(key, value) VALUES (%s, %s) "
            "ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value "
            "WHERE json_text_valid(state_meta.value) AND state_meta.value::jsonb ? 'expires'",
            (key, json.dumps({**value, 'expires': True})))
    else:
        conn.execute(
            "INSERT INTO state_meta(key, value) VALUES (%s, %s) "
            "ON CONFLICT (key) DO UPDATE SET value = CASE "
            "WHEN NOT json_text_valid(state_meta.value) THEN (EXCLUDED.value::jsonb - 'own')::text "
            "WHEN (state_meta.value::jsonb -> 'own') IS NOT NULL OR state_meta.value::jsonb ? 'expires' "
            "THEN EXCLUDED.value ELSE (EXCLUDED.value::jsonb - 'own')::text END",
            (key, json.dumps(value)))


def _retire_terminal_receipts(conn, session_ids) -> None:
    """Tombstone the session's terminal ledger rows in ``state_meta`` (the SQLite
    ``retire_terminal_receipts``): an id with live or unknown work is refused (``session_busy`` /
    ``unknown_execution``), a terminal admission keeps its identity pointer and a compacted
    payload, a terminal worker keeps only its receipt sequences (results must not outlive the
    delete), and the ledger rows are deleted so the sessions' own delete is unguarded."""
    for sid in session_ids:
        admissions = conn.execute(
            "SELECT * FROM session_admissions WHERE target_session_id = %s", (sid,)).fetchall()
        workers = conn.execute(
            "SELECT * FROM worker_executions WHERE session_id = %s", (sid,)).fetchall()
        states = {row["status"] for row in [*admissions, *workers]} - {'terminal'}
        if states:
            raise AuthorityRefusal('unknown_execution' if 'unknown' in states else 'session_busy')
        for row in admissions:
            rec = dict(row)
            # Keep the digest for exact retries, not another copy of user input/history.
            rec['payload_json'] = '{}'
            rec['lineage_json'] = '[]'
            rec['retired_at'] = time.time()
            conn.execute("INSERT INTO state_meta(key, value) VALUES (%s, %s)",
                         (ADMISSION_PREFIX + rec['admission_id'], json.dumps(rec)))
            conn.execute("INSERT INTO state_meta(key, value) VALUES (%s, %s)",
                         (identity_key(rec['principal_id'], sid, rec['request_id']),
                          json.dumps(rec['admission_id'])))
        for row in workers:
            rec = dict(row)
            receipts = [dict(r) for r in conn.execute(
                "SELECT sequence, record_json FROM worker_receipts WHERE execution_id = %s ORDER BY sequence",
                (rec['execution_id'],)).fetchall()]
            # A terminal worker can only replay an explicit ``execution.finish`` receipt. Every
            # other result (history/context reads) is user data that must not outlive the delete,
            # even when it is the last one: the result payload is dropped, the sequence kept.
            for receipt in receipts:
                receipt['record_json'] = None
            rec['receipts'] = receipts
            rec['retired_at'] = time.time()
            conn.execute("INSERT INTO state_meta(key, value) VALUES (%s, %s)",
                         (WORKER_PREFIX + rec['execution_id'], json.dumps(rec)))
            conn.execute("DELETE FROM worker_receipts WHERE execution_id = %s", (rec['execution_id'],))
        conn.execute("DELETE FROM worker_executions WHERE session_id = %s", (sid,))
        conn.execute("DELETE FROM session_admissions WHERE target_session_id = %s", (sid,))


def _retire_mutation_receipts(conn, session_ids) -> None:
    """Keep each exact-retry mutation receipt (digest, ids, revision) but drop the user content
    a rewind/compress/rename result carries, so a retry after the delete cannot read it back."""
    if not session_ids:
        return
    targets = set(session_ids)
    for row in conn.execute(
            "SELECT key, value FROM state_meta WHERE key LIKE 'gateway.mutation.v1.*'").fetchall():
        try:
            receipt = json.loads(row["value"])
        except (TypeError, ValueError):
            continue
        result = receipt.get('result') if isinstance(receipt, dict) else None
        if not isinstance(result, dict):
            continue
        named = (result.get('session_id'), result.get('target_session_id'),
                 result.get('previous_target_session_id'))
        tm = result.get('target_message')
        if isinstance(tm, dict):
            named = (*named, tm.get('session_id'))
        if any(name in targets for name in named if isinstance(name, str)):
            receipt['result'] = {k: (None if k in _MUTATION_TRANSCRIPT_FIELDS else v)
                                for k, v in result.items()}
            conn.execute("UPDATE state_meta SET value = %s WHERE key = %s",
                         (json.dumps(receipt), row["key"]))


def _entry_session_id(raw):
    """The ``session_id`` a stored routing JSON object names, else None (unrelated/malformed rows
    name no target; the routing loader skips them too)."""
    try:
        value = json.loads(raw)
    except (TypeError, ValueError):
        return None
    return value.get('session_id') if isinstance(value, dict) else None


def _retire_routes(conn, session_ids, *, expiring=()) -> None:
    expiring = set(expiring)
    for sid in session_ids:
        _mark_retired(conn, RETIRED_PREFIX + sid, expires=sid in expiring, own=True)
        conn.execute("DELETE FROM state_meta WHERE key = %s", ("gateway.api.settings.v1." + sid,))
        conn.execute("DELETE FROM state_meta WHERE key = %s", ("gateway.viewer_close.v1:" + sid,))
    targets = set(session_ids)
    for row in conn.execute("SELECT scope, session_key, entry_json FROM gateway_routing").fetchall():
        if _entry_session_id(row["entry_json"]) in targets:
            conn.execute("DELETE FROM gateway_routing WHERE scope = %s AND session_key = %s",
                         (row["scope"], row["session_key"]))


def _retire_local_receipts(conn, session_ids) -> None:
    """The deletion-fence half for owner receipts: the local creation policy and the declared
    conversation key go with the deleted ids (the SQLite ``retire_local_receipts``)."""
    targets = set(session_ids)
    if not targets:
        return
    gone = [row["key"] for row in conn.execute(
        "SELECT key, value FROM state_meta WHERE key LIKE %s", (POLICY_PREFIX + '*',))
        if _policy_target(row["value"]) in targets]
    gone += [row["key"] for row in conn.execute(
        "SELECT key, value FROM state_meta WHERE key LIKE %s", (API_DECLARED_PREFIX + '*',))
        if row["value"] in targets]
    gone += [API_BINDING_PREFIX + sid for sid in targets]
    for key in gone:
        conn.execute("DELETE FROM state_meta WHERE key = %s", (key,))


def _retire_sessions(conn, session_ids, *, expiring=(), media_root=None) -> None:
    """The complete deletion fence for the ids deleted in this transaction (the SQLite
    ``retire_sessions``): terminal receipts, mutation receipts, routes and owner receipts. The
    media spillover and the terminal result store have no PG tables in stage 1, so their scrubs
    are no-ops until those repos are ported."""
    _retire_terminal_receipts(conn, session_ids)
    _retire_mutation_receipts(conn, session_ids)
    _retire_routes(conn, session_ids, expiring=expiring)
    _retire_local_receipts(conn, session_ids)


def _retire_prunable(conn, session_ids, *, media_root=None) -> list:
    """Sweep variant of :func:`_retire_sessions`: fence the idle sessions and return only those
    ids. A session with live or unknown work is skipped, so one busy row cannot abort a whole
    prune/empty-session sweep. A local reset/compression conversation is swept only whole: while
    any of its ids is kept or has live work, every one is skipped. A sweep's markers (non-derived
    ids) age out at the retry horizon; a derived id's stays (the SQLite ``retire_prunable``)."""
    swept, (conversations, derived) = set(session_ids), _local_conversations(conn)

    def removable(sid):
        whole = conversations.get(sid, (sid,))
        return swept.issuperset(whole) and not any(
            conn.execute(_LIVE_LEDGER_SQL, (x, x)).fetchone() for x in whole)

    quiet = [sid for sid in session_ids if removable(sid)]
    derived |= _derived_api_sessions(conn, quiet)
    _retire_sessions(conn, quiet, expiring=[sid for sid in quiet if sid not in derived],
                     media_root=media_root)
    return quiet


def _reclaim_expired(conn, prefix: str, cutoff: float) -> None:
    """Drop the sweep (``expires``) markers under *prefix* retired before *cutoff* (the SQLite
    ``_reclaim_expired``; the expired predicate is the non-throwing number read)."""
    conn.execute(
        "DELETE FROM state_meta WHERE key LIKE %s AND json_text_valid(value) "
        "AND json_text_field_number(value, 'retired_at') < %s AND value::jsonb ? 'expires'",
        (prefix + '*', cutoff))


def _prune_retired_tombstones(conn, older_than: float = RETRY_HORIZON_S) -> None:
    """Reclaim deletion tombstones retired more than *older_than* seconds ago: the terminal
    admission with its identity pointer, terminal workers and a sweep's (``expires``)
    retired-session marker or shared routing fence. A user delete's marker is the deletion fence
    itself and is never reclaimed. Tombstones written before ``retired_at`` existed are stamped
    now and age from this pass (the SQLite ``prune_retired_tombstones``)."""
    now = time.time()
    for prefix in (ADMISSION_PREFIX, WORKER_PREFIX):
        conn.execute(
            "UPDATE state_meta SET value = (jsonb_set(value::jsonb, '{retired_at}', "
            "to_jsonb(%s::float8)))::text WHERE key LIKE %s AND json_text_valid(value) "
            "AND json_text_field_number(value, 'retired_at') IS NULL",
            (now, prefix + '*'))
    cutoff = now - older_than
    for row in conn.execute(
            "SELECT key, value FROM state_meta WHERE key LIKE %s AND json_text_valid(value) "
            "AND json_text_field_number(value, 'retired_at') < %s",
            (ADMISSION_PREFIX + '*', cutoff)).fetchall():
        rec = json.loads(row["value"])
        conn.execute("DELETE FROM state_meta WHERE key IN (%s, %s, %s)",
                     (row["key"], RESULT_PREFIX + rec['admission_id'],
                      identity_key(rec['principal_id'], rec['target_session_id'], rec['request_id'])))
    conn.execute(
        "DELETE FROM state_meta WHERE key LIKE %s AND json_text_valid(value) "
        "AND json_text_field_number(value, 'retired_at') < %s",
        (WORKER_PREFIX + '*', cutoff))
    for prefix in (RETIRED_PREFIX, ROUTE_FENCE_PREFIX):
        _reclaim_expired(conn, prefix, cutoff)


def _write_guards_reject(conn, sid: str) -> bool:
    """True when a live turn lease or compression lock protects ``sid`` (the SQLite
    ``_write_guards_reject`` read-only shape: the guards' own reclamation lives in the
    transcript writers, which this store's prune path never calls)."""
    row = conn.execute(
        "SELECT 1 FROM session_turn_leases WHERE conversation_id = %s AND expires_at > EXTRACT(EPOCH FROM now()) "
        "UNION ALL SELECT 1 FROM compression_locks WHERE session_id = %s "
        "AND expires_at > EXTRACT(EPOCH FROM now()) LIMIT 1", (sid, sid)).fetchone()
    return row is not None


# ── the import payload validation (the SQLite ``_portability`` module functions, pure Python) ──
_IMPORT_SESSION_TEXT_FIELDS = (
    "source", "user_id", "model", "system_prompt", "end_reason", "cwd", "git_branch", "git_repo_root",
    "billing_provider", "billing_base_url", "billing_mode", "cost_status", "cost_source", "pricing_version", "title",
)
# ``role`` is validated separately (non-empty string).
_IMPORT_MESSAGE_TEXT_FIELDS = (
    "tool_call_id", "tool_name", "effect_disposition", "finish_reason",
    "reasoning", "reasoning_content", "platform_message_id", "message_id",
)
_IMPORT_MESSAGE_JSON_FIELDS = (
    "reasoning_details", "codex_reasoning_items", "codex_message_items", "absorbed_message_uids", "tool_call_uids",
)
_IMPORT_PASSTHROUGH_COLS = (
    "user_id", "model", "model_config", "end_reason", "cwd", "git_branch", "git_repo_root", "billing_provider",
    "billing_base_url", "billing_mode", "cost_status", "cost_source", "pricing_version", "title",
)
_IMPORT_INT_COLS = (
    "input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens", "reasoning_tokens", "api_call_count",
)
_IMPORT_FLOAT_COLS = ("ended_at", "estimated_cost_usd", "actual_cost_usd")
# Flags a restore must keep: pinned/hidden/auto_archived drive the prune, listings and resume-unarchive.
_IMPORT_FLAG_COLS = ("archived", "auto_archived", "pinned", "hidden")

# The session insert: the payload's columns (the SQLite ``_IMPORT_SESSION_INSERT_SQL`` on the PG
# dialect; ``system_prompt`` is stored inline here, ``parent_session_id`` re-attached after).
_IMPORT_SESSION_INSERT_SQL = """INSERT INTO sessions (
                           id, source, user_id, model, model_config,
                           system_prompt_hash, parent_session_id, started_at, ended_at,
                           end_reason, message_count, tool_call_count, input_tokens, output_tokens,
                           cache_read_tokens, cache_write_tokens, reasoning_tokens,
                           cwd, git_branch, git_repo_root,
                           billing_provider, billing_base_url, billing_mode,
                           estimated_cost_usd, actual_cost_usd, cost_status, cost_source,
                           pricing_version, title, api_call_count, archived, auto_archived, pinned, hidden
                       )
                       VALUES (
                           %s, %s, %s, %s, %s,
                           %s, NULL, %s, %s,
                           %s, 0, 0, %s, %s,
                           %s, %s, %s, %s, %s, %s,
                           %s, %s, %s,
                           %s, %s, %s, %s,
                           %s, %s, %s, %s, %s, %s, %s)"""


def _export_columns() -> tuple:
    """The export record's session columns (``SessionExportRow`` minus ``last_active`` /
    ``reset_from`` / ``branched_from``) — exactly the live ``sessions`` columns."""
    return tuple(field.name for field in __import__("dataclasses").fields(SessionExportRow)[:-3])


def _export_row(session: dict[str, Any], last_active: Optional[float]) -> SessionExportRow:
    """The contract record of one session row dict (the PG ``sessions`` shape plus the lifted
    ``model_config`` markers and the recency read)."""
    model_config = session.get("model_config")
    if isinstance(model_config, str):
        try:
            model_config = json.loads(model_config)
        except (ValueError, TypeError):
            model_config = None
    model_config = model_config if isinstance(model_config, dict) else {}
    return SessionExportRow(
        **{column: session[column] for column in _export_columns()}, last_active=last_active,
        reset_from=model_config.get("_reset_from"), branched_from=model_config.get("_branched_from"))


def _import_text_or_none(value: Any, field: str) -> Optional[str]:
    if value is None or isinstance(value, str):
        return value
    raise ValueError(f"{field} must be a string")


def _import_int_or_none(value: Any, field: str) -> Optional[int]:
    try:
        return None if value is None else int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be an integer") from exc


def _import_non_finite_path(value: Any, path: str) -> Optional[str]:
    """First ``path`` of a NaN/inf number anywhere in decoded JSON data, else None."""
    pending = [(value, path)]
    while pending:
        current, current_path = pending.pop()
        if isinstance(current, float) and not math.isfinite(current):
            return current_path
        if isinstance(current, dict):
            pending.extend((child, f"{current_path}.{key}") for key, child in current.items())
        elif isinstance(current, list):
            pending.extend((child, f"{current_path}[{key}]") for key, child in enumerate(current))
    return None


def _import_load_json(value: str, path: str) -> Any:
    """``json.loads`` that refuses NaN/Infinity anywhere in the decoded value."""
    parsed = json.loads(value)
    non_finite = _import_non_finite_path(parsed, path)
    if non_finite is not None:
        raise ValueError(f"non-finite number at {non_finite}")
    return parsed


def _import_json_object_or_none(value: Any, field: str) -> Optional[str]:
    if value is None:
        return None
    path = f"session.{field}"
    if isinstance(value, str):
        try:
            parsed = _import_load_json(value, path)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{field} must be valid JSON") from exc
        if not isinstance(parsed, dict):
            raise ValueError(f"{field} must be a JSON object")
        return value
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be a JSON object")
    if _import_non_finite_path(value, path) is not None:
        raise ValueError(f"non-finite number at {path}")
    try:
        return json.dumps(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be JSON serializable") from exc


def _coerce_or(value: Any, cast, default):
    """``cast(value)``; *default* for None or an unparsable value."""
    try:
        return default if value is None else cast(value)
    except (TypeError, ValueError):
        return default


def _import_int_or_bounded(value: Any, field: str, default: int) -> int:
    """Import an int column: SQL-int-range or *default* (never fails the import)."""
    number = _coerce_or(value, int, default)
    if number is None:
        return default
    if isinstance(number, bool) or not (_SQLITE_INT_MIN <= number <= _SQLITE_INT_MAX):
        return default
    return number


def _normalize_import_message(message: dict[str, Any], message_index: int) -> dict[str, Any]:
    """Type-check one payload message; raises ValueError naming the offending field."""
    clean_message = dict(message)
    role = clean_message.get("role")
    if not isinstance(role, str) or not role:
        raise ValueError(f"messages[{message_index}].role must be a non-empty string")
    for field in _IMPORT_MESSAGE_TEXT_FIELDS:
        clean_message[field] = _import_text_or_none(clean_message.get(field), field)
    token_count = _import_int_or_none(clean_message.get("token_count"), "token_count")
    if token_count is not None and not (_SQLITE_INT_MIN <= token_count <= _SQLITE_INT_MAX):
        raise ValueError("token_count is outside SQLite's integer range")
    clean_message["token_count"] = token_count
    message_path = f"session.messages[{message_index}]"
    timestamp = clean_message.get("timestamp")
    if timestamp is not None and coerce_epoch(timestamp, field="messages[].timestamp") is None:
        raise ValueError(f"{message_path}.timestamp must be a supported Unix timestamp")
    _validate_import_message_content(clean_message.get("content"), message_path)
    for field in _IMPORT_MESSAGE_JSON_FIELDS:
        json_value = clean_message.get(field)
        if isinstance(json_value, str):
            # Malformed JSON text persists verbatim (readers fall back via _json_or);
            # only a PARSED non-finite number is rejected: it would break strict
            # JSON serializers once decoded back out of the row.
            try:
                _import_load_json(json_value, f"{message_path}.{field}")
            except json.JSONDecodeError:
                pass
            except ValueError:
                raise
        elif json_value is not None:
            non_finite = _import_non_finite_path(json_value, f"{message_path}.{field}")
            if non_finite is not None:
                raise ValueError(f"non-finite number at {non_finite}")
    return clean_message


def _validate_import_message_content(content: Any, message_path: str) -> None:
    """Scalar message content must be bindable and later serializable."""
    if isinstance(content, int) and not isinstance(content, bool) and not (
            _SQLITE_INT_MIN <= content <= _SQLITE_INT_MAX):
        raise ValueError(f"{message_path}.content is outside SQLite's integer range")
    if isinstance(content, float) and not math.isfinite(content):
        raise ValueError(f"non-finite number at {message_path}.content")
    if isinstance(content, (bytes, bytearray)):
        raise ValueError(f"{message_path}.content must be a string, list, object, finite number, or null")


def _import_float_or_finite(value: Any, field: str) -> Optional[float]:
    """Import a float column: finite float or None (never fails the import)."""
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _normalize_import_session(raw: dict[str, Any], session_id: str, messages: list) -> dict[str, Any]:
    """Type-check one payload session + its messages; raises ValueError."""
    clean_session = dict(raw)
    clean_session["id"] = session_id
    clean_session["model_config"] = _import_json_object_or_none(clean_session.get("model_config"), "model_config")
    for field in ("parent_session_id", *_IMPORT_SESSION_TEXT_FIELDS):
        clean_session[field] = _import_text_or_none(clean_session.get(field), field)
    started_at = clean_session.get("started_at")
    if started_at is not None and coerce_epoch(started_at, field="started_at") is None:
        raise ValueError("session.started_at must be a supported Unix timestamp")
    for field in _IMPORT_FLOAT_COLS:
        value = clean_session.get(field)
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"non-finite number at session.{field}")
    for field in _IMPORT_INT_COLS:
        value = clean_session.get(field)
        if isinstance(value, int) and not isinstance(value, bool) and not (
                _SQLITE_INT_MIN <= value <= _SQLITE_INT_MAX):
            raise ValueError(f"{field} is outside SQLite's integer range")
    clean_messages: list[dict[str, Any]] = []
    for message_index, message in enumerate(messages):
        clean_messages.append(_normalize_import_message(message, message_index))
    return {"session": clean_session, "messages": clean_messages}


def _validate_import_session(raw: Any, session_id: str, seen_ids: set, totals: dict[str, int]) -> dict[str, Any]:
    """One payload session -> normalized item; ValueError(message) on rejection. *totals*
    accumulate before their limit check (a rejected oversize entry still counts)."""
    if not isinstance(raw, dict):
        raise ValueError("session must be an object")
    if not session_id:
        raise ValueError("session id is required")
    if session_id in seen_ids:
        raise ValueError("duplicate session id")
    messages = raw.get("messages") or []
    if not isinstance(messages, list):
        raise ValueError("messages must be a list")
    if len(messages) > IMPORT_MAX_MESSAGES_PER_SESSION:
        raise ValueError("messages exceeds the per-session import limit")
    if any(not isinstance(msg, dict) for msg in messages):
        raise ValueError("messages must contain only objects")
    try:
        # `timings` is derived from the messages at export time and rebuilt on the next export;
        # it must not eat into the size budget of the content it merely describes.
        measured = {k: v for k, v in raw.items() if k != "timings"}
        session_bytes = len(json.dumps(measured, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
    except (TypeError, ValueError):
        raise ValueError("session must be JSON serializable") from None
    if session_bytes > IMPORT_MAX_SESSION_BYTES:
        raise ValueError("session exceeds the import size limit")
    totals["bytes"] += session_bytes
    if totals["bytes"] > IMPORT_MAX_TOTAL_BYTES:
        raise ValueError("import exceeds the total size limit")
    item = _normalize_import_session(raw, session_id, messages)
    totals["messages"] += len(item["messages"])
    if totals["messages"] > IMPORT_MAX_TOTAL_MESSAGES:
        raise ValueError("messages exceeds the total import limit")
    return item


def _validate_import_payload(sessions: list[dict[str, Any]]) -> tuple:
    """Size/shape/type validation of the whole payload; returns ``(normalized_items,
    errors)``. Every rejected entry is reported."""
    normalized: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    totals = {"messages": 0, "bytes": 0}
    for index, raw in enumerate(sessions):
        session_id = str(raw.get("id") or "").strip() if isinstance(raw, dict) else ""
        try:
            item = _validate_import_session(raw, session_id, seen_ids, totals)
        except ValueError as exc:
            item = {"index": index, "error": str(exc)}
            if session_id:
                item["session_id"] = session_id
            errors.append(item)
            continue
        seen_ids.add(session_id)
        normalized.append({"index": index, **item})
    return normalized, errors


def _attach_import_parents(conn, parent_updates: list) -> int:
    """Re-attach imported children whose parent exists (in the store or the same payload)
    without creating a cycle; returns the detached count (the SQLite ``_attach_import_parents``)."""
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
        parent_exists = conn.execute("SELECT 1 FROM sessions WHERE id = %s LIMIT 1", (parent_id,)).fetchone()
        if parent_exists and not _would_create_cycle(session_id, parent_id):
            conn.execute("UPDATE sessions SET parent_session_id = %s WHERE id = %s", (parent_id, session_id))
        else:
            parent_by_child.pop(session_id, None)
            detached += 1
    return detached


def _find_foreign_import_on_txn(conn, origin: dict) -> Optional[str]:
    """The txn arm of the foreign-import match (runs on the write connection in
    ``import_foreign_history``); the store-read arm is the module-level
    :func:`_find_foreign_import_on_conn`."""
    rows = conn.execute(
        "SELECT id, origin_json FROM sessions WHERE source = %s AND origin_json IS NOT NULL",
        (origin["tool"],)).fetchall()
    for row in rows:
        imported = (safe_json_loads(row["origin_json"], default={}) or {}).get("imported_from", {})
        if imported.get("tool") != origin["tool"]:
            continue
        foreign_id = origin.get("foreign_session_id")
        if ((foreign_id and foreign_id == imported.get("foreign_session_id"))
                or (not foreign_id and imported.get("path") == origin["path"])):
            return row["id"]
    return None


class PgSessionMaintenance:
    """Maintenance verbs on the bound profile's store schema."""

    def __init__(self, store: PostgresStore) -> None:
        self._store = store

    # ── prune and sweeps ──────────────────────────────────────────────────

    def prune_sessions(self, older_than_days: Optional[float] = 90, source: Optional[str] = None, *,
                       criteria: PruneCriteria = PruneCriteria(), exclude_active_write_guards: bool = False,
                       media_root: str) -> PruneResult:
        filters = asdict(criteria)
        if filters.get("include_open"):
            # Deletion is not reversible, so the ended-session guard is structural here: the
            # archive-only escape hatch must never reach the delete path.
            raise ValueError("prune_sessions() never deletes open sessions: include_open is archive-only")
        where, where_params = _prune_where(older_than_days, source, filters, whole_lineages=True)
        removed_ids: list[str] = []

        def _do(conn):
            cursor = conn.execute(f"SELECT s.id FROM sessions s WHERE {where}", where_params)
            session_ids = {row["id"] for row in cursor.fetchall()}
            if exclude_active_write_guards:
                session_ids = {sid for sid in session_ids
                               if not _write_guards_reject(conn, sid)}
            session_ids = _retire_prunable(conn, sorted(session_ids), media_root=media_root)
            _prune_retired_tombstones(conn)  # every prune pass also reclaims expired deletion tombstones
            if not session_ids:
                return 0
            # Batched: a cron-heavy store prunes tens of thousands of ids in one call.
            for chunk in _id_chunks(session_ids):
                ph = _placeholders(len(chunk))
                conn.execute(f"UPDATE sessions SET parent_session_id = NULL WHERE parent_session_id IN ({ph})", chunk)
                conn.execute(f"DELETE FROM messages WHERE session_id IN ({ph})", chunk)
                conn.execute(f"DELETE FROM sessions WHERE id IN ({ph})", chunk)
                removed_ids.extend(chunk)
            self._delete_unreferenced_system_prompts(conn)
            return len(session_ids)

        with translated():
            with self._store.tx() as conn:
                count = _do(conn)
        return PruneResult(count=count, deleted_ids=tuple(SessionId(sid) for sid in removed_ids))

    def prune_empty_ghost_sessions(self, *, media_root: str) -> tuple[SessionId, ...]:
        cutoff = time.time() - 86400

        def _do(conn):
            ids = [row["id"] for row in conn.execute("""
                SELECT id FROM sessions
                WHERE source = 'tui'
                  AND title IS NULL
                  AND ended_at IS NOT NULL
                  AND started_at < %s
                  AND NOT EXISTS (
                      SELECT 1 FROM messages WHERE messages.session_id = sessions.id
                  )
            """, (cutoff,)).fetchall()]
            ids = _retire_prunable(conn, ids, media_root=media_root)
            # A concurrent prune that committed first deletes the same rows; RETURNING keeps the
            # contenders' result sets disjoint (only the ids this DELETE removed).
            removed: list[str] = []
            for chunk in _id_chunks(ids):
                deleted = conn.execute(
                    f"DELETE FROM sessions WHERE id IN ({_placeholders(len(chunk))}) RETURNING id",
                    chunk).fetchall()
                removed.extend(row["id"] for row in deleted)
            if ids:
                self._delete_unreferenced_system_prompts(conn)
            return removed

        with translated():
            with self._store.tx() as conn:
                removed_ids = _do(conn) or []
        return tuple(SessionId(sid) for sid in removed_ids)

    def sweep_orphaned_sessions(
        self, *, max_idle_seconds: float, sources: tuple[str, ...] = ("tui", "desktop", "subagent"),
        exclude_ids: tuple[str, ...] = (), exclude_pinned: bool = False,
        heartbeat_staleness_seconds: Optional[float] = None,
        heartbeat_ownership_grace_seconds: Optional[float] = None, respect_gateway_heartbeats: bool = True,
    ) -> tuple[SessionId, ...]:
        srcs = tuple(s for s in sources if s)
        if max_idle_seconds <= 0 or not srcs:
            return ()
        hb_staleness, hb_grace = heartbeat_staleness_seconds, heartbeat_ownership_grace_seconds
        if not (hb_staleness and hb_staleness > 0):
            hb_staleness = max_idle_seconds * 2
        if not (hb_grace is not None and hb_grace >= 0):
            hb_grace = hb_staleness
        cutoff = (now := time.time()) - max_idle_seconds
        pin_scope = f" AND {_not_pinned_sql('sessions')}" if exclude_pinned else ""
        orphan_predicate = f"started_at < %s AND {session_last_active_sql('sessions')} < %s"
        heartbeat_params: tuple = ()
        if respect_gateway_heartbeats:
            orphan_predicate += (" AND NOT EXISTS (SELECT 1 FROM gateway_heartbeats h WHERE"
                                 " h.last_heartbeat >= %s AND h.started_at <= sessions.started_at + %s)")
            heartbeat_params = (now - hb_staleness, hb_grace)
        scope_sql = f" AND source IN ({_placeholders(len(srcs))}){pin_scope} AND {orphan_predicate}"
        scope_params = (*srcs, cutoff, cutoff, *heartbeat_params)

        def _do(conn):
            rows = conn.execute(f"SELECT id FROM sessions WHERE ended_at IS NULL{scope_sql}",
                                scope_params).fetchall()
            excluded = {str(x) for x in exclude_ids if x}
            victims = [sid for sid in (row["id"] for row in rows)
                       if sid not in excluded and not _write_guards_reject(conn, sid)]
            if not victims:
                return []
            # Re-apply every predicate under the write lock; a concurrent sweep that committed first
            # leaves those rows ended, so return exactly the ids this UPDATE ended - the contenders'
            # result sets stay disjoint.
            changed = conn.execute(
                f"UPDATE sessions SET ended_at = %s, end_reason = 'startup_orphan_reap'"
                f" WHERE id IN ({_placeholders(len(victims))}) AND ended_at IS NULL{scope_sql} RETURNING id",
                (time.time(), *victims, *scope_params)).fetchall()
            return [row["id"] for row in changed]

        with translated():
            with self._store.tx() as conn:
                removed = _do(conn) or []
        return tuple(SessionId(sid) for sid in removed)

    # ── selection reads and the idle archive ──────────────────────────────

    def list_prune_candidates(self, older_than_days: Optional[float] = None, source: Optional[str] = None, *,
                              criteria: PruneCriteria = PruneCriteria(), whole_lineages: bool = False,
                              ) -> tuple[PruneCandidate, ...]:
        where, params = _prune_where(older_than_days, source, asdict(criteria), whole_lineages=whole_lineages)
        with translated():
            rows = self._store.all(
                f"""SELECT s.id, s.source, s.title, s.model, s.started_at,
                           {_LAST_ACTIVE_SQL} AS last_active,
                           s.ended_at, s.message_count, s.archived
                    FROM sessions s WHERE {where}
                    ORDER BY last_active ASC, s.started_at ASC""", *params)
        return tuple(PruneCandidate(
            session_id=SessionId(row["id"]), source=row["source"], title=row["title"], model=row["model"],
            started_at=row["started_at"], last_active=row["last_active"], ended_at=row["ended_at"],
            message_count=row["message_count"], archived=row["archived"]) for row in rows)

    def count_prune_matches(self, older_than_days: Optional[float] = None, source: Optional[str] = None, *,
                            criteria: PruneCriteria = PruneCriteria(), pinned_only: bool = False) -> int:
        filters = asdict(criteria)
        if pinned_only:
            filters["include_pinned"] = True
        where, params = _prune_where(older_than_days, source, filters)
        if pinned_only:
            where += " AND COALESCE(s.pinned, 0) = 1"
        with translated():
            return int(self._store.scalar(
                f"SELECT COUNT(*) FROM sessions s WHERE {where}", *params) or 0)

    def count_open_prune_matches(self, older_than_days: Optional[float] = None, source: Optional[str] = None, *,
                                 criteria: PruneCriteria = PruneCriteria()) -> int:
        filters = asdict(criteria)
        if filters.get("include_open"):
            return 0
        where, params = _prune_where(older_than_days, source, filters)
        ended_guard = "s.ended_at IS NOT NULL"
        if not where.startswith(ended_guard):
            raise RuntimeError("prune filter lost its ended-session safety guard")
        open_where = f"s.ended_at IS NULL{where[len(ended_guard):]}"
        with translated():
            return int(self._store.scalar(
                f"SELECT COUNT(*) FROM sessions s WHERE {open_where}", *params) or 0)

    def prune_reclaim_estimate(self, older_than_days: Optional[float], source: Optional[str] = None, *,
                               criteria: PruneCriteria = PruneCriteria(), whole_lineages: bool = True,
                               measure_total_bytes: bool = True) -> PruneReclaimEstimate:
        where, params = _prune_where(older_than_days, source, asdict(criteria), whole_lineages=whole_lineages)
        prunable_bytes, total_bytes = 0, None
        with translated():
            prunable, total = self._store.scalar(
                f"SELECT (SELECT COUNT(*) FROM sessions s WHERE {where})", *params), None
            prunable = int(prunable or 0)
            total = int(self._store.scalar("SELECT COUNT(*) FROM sessions") or 0)
            if prunable:
                # Weight by stored bytes, not message_count: old messages are far smaller than recent
                # ones, and message_count skips the soft-archived rows a prune also deletes.
                with self._store.tx() as conn:
                    row_bytes = _row_bytes_sql(conn)
                prunable_bytes = int(self._store.scalar(
                    f"SELECT COALESCE(SUM({row_bytes}), 0) FROM messages m WHERE m.session_id IN "
                    f"(SELECT s.id FROM sessions s WHERE {where})", *params) or 0)
                if measure_total_bytes:
                    total_bytes = int(self._store.scalar(
                        f"SELECT COALESCE(SUM({row_bytes}), 0) FROM messages m") or 0)
        return PruneReclaimEstimate(prunable=prunable, total=total, prunable_bytes=prunable_bytes,
                                    total_bytes=None if total_bytes is None else total_bytes)

    def list_idle_archive_candidates(self, idle_days: float, *, exclude_pinned: bool = True) -> tuple[SessionId, ...]:
        if idle_days is None or idle_days < 0:
            return ()
        cutoff = time.time() - float(idle_days) * 86400.0
        pin_clause = f"AND {_not_pinned_sql()}" if exclude_pinned else ""
        with translated():
            rows = self._store.all(
                f"""
                SELECT s.id FROM sessions s
                WHERE s.archived = 0
                  AND COALESCE(s.end_reason, '') <> 'compression'
                  {pin_clause}
                  AND NOT (COALESCE(s.hidden, 0) <> 0 AND COALESCE(s.title, '') = %s)
                  AND {_LAST_ACTIVE_SQL} < %s
                ORDER BY s.started_at ASC
                """, CANONICAL_BOT_CHAT_TITLE, cutoff)
        return tuple(SessionId(row["id"]) for row in rows)

    def auto_archive_lineage(self, session_id: SessionId) -> bool:
        # A row already archived keeps its provenance, so a deliberately archived ancestor is never
        # relabelled as sweep-owned (a plain ``archived`` write clears ``auto_archived``).
        with translated():
            with self._store.tx() as conn:
                row = conn.execute(
                    f"""{_LINEAGE_CTE_SQL}
                        UPDATE sessions
                        SET archived = 1, auto_archived = CASE WHEN archived = 0 THEN 1 ELSE auto_archived END
                        WHERE id IN (SELECT id FROM lineage)
                        RETURNING id""",
                    (str(session_id), str(session_id)),
                ).fetchone()
            return row is not None

    # ── backend heartbeats ────────────────────────────────────────────────

    def register_backend_heartbeat(self, *, backend_id: str, pid: int, started_at: float,
                                   last_heartbeat: Optional[float] = None, profile: str = "", host: str = "") -> None:
        if not backend_id:
            return
        ts = time.time() if last_heartbeat is None else float(last_heartbeat)
        with translated():
            with self._store.tx() as conn:
                conn.execute(
                    "INSERT INTO gateway_heartbeats (backend_id, pid, started_at, last_heartbeat, profile, host)"
                    " VALUES (%s, %s, %s, %s, %s, %s) "
                    "ON CONFLICT (backend_id) DO UPDATE SET pid = EXCLUDED.pid,"
                    " started_at = EXCLUDED.started_at, last_heartbeat = EXCLUDED.last_heartbeat,"
                    " profile = EXCLUDED.profile, host = EXCLUDED.host",
                    (str(backend_id), int(pid), float(started_at), ts, str(profile), str(host)))

    def clear_backend_heartbeat(self, backend_id: str) -> bool:
        if not backend_id:
            return False
        with translated():
            with self._store.tx() as conn:
                return (conn.execute(
                    "DELETE FROM gateway_heartbeats WHERE backend_id = %s", (str(backend_id),)).rowcount or 0) > 0

    # ── read-only probes ──────────────────────────────────────────────────

    def assert_export_safe(self, session_id: SessionId, max_messages: int) -> int:
        if max_messages < 0:
            raise ValueError("max_messages must be non-negative")
        if max_messages == 0:
            return 0
        with translated():
            row = self._store.one(
                "SELECT COUNT(*) AS n FROM (SELECT 1 FROM messages WHERE session_id = %s LIMIT %s) t",
                str(session_id), max_messages + 1)
        message_count = int(row["n"] if row else 0)
        if message_count > max_messages:
            raise ExportTooLarge(session_id, message_count, max_messages)
        return message_count

    def session_projection_digest(self, *, busy_timeout_s: float = 0.05) -> Optional[tuple[tuple[str, ...], bytes]]:
        with translated():
            available = {row["column_name"] for row in self._store.all(
                "SELECT column_name FROM information_schema.columns"
                " WHERE table_schema = current_schema() AND table_name = 'sessions'")}
            fields = tuple(field for field in SESSION_PROJECTION_FIELDS if field in available)
            if not fields:
                return None
            order = " ORDER BY id" if "id" in available else ""
            rows = self._store.all(f"SELECT {', '.join(fields)} FROM sessions{order}")
            digest = hashlib.blake2b(digest_size=16)
            for row in rows:
                digest.update(repr(tuple(row.values())).encode("utf-8", "backslashreplace"))
                digest.update(b"\0")
            return fields, digest.digest()

    def retained_tree_is_live(self, worktree_path: str, *, pending_only: bool = False) -> bool:
        """Whether the tree still serves a session: a session under the tree with work left — its own
        messages (never for ``pending_only``) or a non-terminal admission (the SQLite
        ``retained_tree_is_live``; the tree predicate runs in Python, the UDF's PG stand-in)."""
        inside = _inside_tree(worktree_path)
        with translated():
            rows = self._store.all("SELECT id, cwd, message_count FROM sessions")
            if pending_only:
                ids = [row["id"] for row in rows if inside(row["cwd"])]
                if not ids:
                    return False
                return bool(self._store.one(
                    "SELECT 1 FROM session_admissions WHERE status <> 'terminal' "
                    "AND target_session_id = ANY(%s) LIMIT 1", ids))
            live = [row["id"] for row in rows
                    if inside(row["cwd"]) and (int(row["message_count"] or 0) > 0)]
            pending = [row["id"] for row in rows if inside(row["cwd"])
                       and int(row["message_count"] or 0) == 0]
            pending_live = None
            if pending:
                pending_live = self._store.scalar(
                    "SELECT 1 FROM session_admissions WHERE status <> 'terminal' "
                    "AND target_session_id = ANY(%s)", pending)
                if pending_live:
                    return True
        return bool(live)

    def tree_served_sessions(self, worktree_path: str) -> TreeServeFacts:
        """The gateway's own records that it will still act in a session after its last viewer quits:
        an open session under the tree, or a state_meta loop/heartbeat/goal row (status ``active``
        when the field is absent) one the session's close did not end (the SQLite
        ``_SERVED_TREE_SQL`` with the tree predicate in Python)."""
        inside = _inside_tree(worktree_path)
        with translated():
            rows = self._store.all("SELECT id, cwd, ended_at, chat_id FROM sessions WHERE cwd IS NOT NULL")
            inside_rows = [row for row in rows if inside(row["cwd"])]
            open_ids = [row["id"] for row in inside_rows if row["ended_at"] is None]
            closed_ids = [row["id"] for row in inside_rows if row["ended_at"] is not None]
            served = bool(open_ids)
            if not served and closed_ids:
                keys = [f"{prefix}{sid}" for sid in closed_ids for prefix in ("loop:", "heartbeat:", "goal:")]
                meta = {row["key"]: row["value"] for row in self._store.all(
                    "SELECT key, value FROM state_meta WHERE key = ANY(%s)", keys)}
                for sid in closed_ids:
                    for prefix in ("loop:", "heartbeat:", "goal:"):
                        raw = meta.get(f"{prefix}{sid}")
                        if raw is None:
                            continue
                        try:
                            payload = json.loads(raw)
                        except (ValueError, TypeError):
                            continue
                        status = payload.get("status", "active") if isinstance(payload, dict) else "active"
                        if status == "active":
                            served = True
                            break
                    if served:
                        break
            if not served:
                chat_ids = frozenset(row["chat_id"] for row in inside_rows if row["chat_id"] is not None)
                return TreeServeFacts(served=False, chat_ids=chat_ids)
        return TreeServeFacts(served=True)

    # ── portability: export and import ────────────────────────────────────

    def _get_session_dict(self, store, session_id: str) -> Optional[dict[str, Any]]:
        """One ``sessions`` row as the export's session dict: every column plus the recency read
        and the prompt-store-resolved ``system_prompt`` / ``tool_names`` (the SQLite ``get_session``
        shape on the PG schema, where the prompt is stored inline and ``tool_names`` holds a
        prompt-store hash)."""
        row = store.one(
            """SELECT s.*,
                     COALESCE(s.system_prompt, sp.prompt) AS _system_prompt_resolved,
                     COALESCE(tp.prompt, s.tool_names) AS _tool_names_resolved
                FROM sessions s
                LEFT JOIN system_prompts sp ON sp.hash = s.system_prompt_hash
                LEFT JOIN system_prompts tp ON tp.hash = s.tool_names
                WHERE s.id = %s""", session_id)
        if row is None:
            return None
        session = dict(row)
        session["system_prompt"] = session.pop("_system_prompt_resolved", None)
        session["tool_names"] = session.pop("_tool_names_resolved", None)
        return session

    @staticmethod
    def _session_messages(store, session_id: str, include_compacted: bool = False,
                          include_inactive: bool = False) -> list[dict[str, Any]]:
        """One session's rows in storage order, the audit/display/live selection of the SQLite
        ``get_messages`` export flags (the display-generation dedupe of the native reader is a
        display-read concern; exports carry the raw rows)."""
        if include_inactive:
            clause = ""
        elif include_compacted:
            clause = " AND (active = 1 OR compacted = 1)"
        else:
            clause = " AND active = 1"
        rows = store.all(f"SELECT * FROM messages WHERE session_id = %s{clause} ORDER BY id", session_id)
        return [_export_message_dict(row, summary_flag=True) for row in rows]

    @staticmethod
    def _with_messages(store, session: dict[str, Any], include_compacted: bool = False,
                       include_inactive: bool = False) -> SessionExport:
        messages = PgSessionMaintenance._session_messages(store, session["id"], include_compacted, include_inactive)
        # Contract: a single-session export row leaves ``last_active`` unset (``export_all`` sets it);
        # an unset extra has no key in the row dict.
        return SessionExport(session=_export_row(session, None),
                             messages=tuple(map(_transcript_row, messages)))

    def _export_session(self, store, session_id: SessionId, include_compacted: bool = False,
                       include_inactive: bool = False) -> Optional[SessionExport]:
        session = self._get_session_dict(store, str(session_id))
        return self._with_messages(store, session, include_compacted, include_inactive) if session else None

    def export_session(self, session_id: SessionId, include_compacted: bool = False,
                      include_inactive: bool = False) -> Optional[SessionExport]:
        with translated():
            return self._export_session(self._store, session_id, include_compacted, include_inactive)

    def export_session_lineage(self, session_id: SessionId, include_compacted: bool = False,
                               include_inactive: bool = False) -> Optional[SessionLineageExport]:
        store = self._store
        with translated():
            lineage_ids = self._compression_lineage_ids(store, str(session_id))
            if not lineage_ids:
                return None
            segments = tuple(seg for sid in lineage_ids
                             if (seg := self._export_session(store, SessionId(sid),
                                                            include_compacted, include_inactive)) is not None)
        if not segments:
            return None
        messages = tuple(msg for seg in segments for msg in seg.messages)
        return SessionLineageExport(segments=segments,
                                    lineage_session_ids=tuple(seg.session.id for seg in segments),
                                    message_count=len(messages), messages=messages)

    def _compression_lineage_ids(self, store, session_id: str) -> list[str]:
        """Compression ancestors through the tip in chronological order (the SQLite
        ``get_compression_lineage`` on the PG schema; a branch/delegate/reset/tool child of a
        compression parent is an explicit fork and stands alone)."""
        session = self._get_session_dict(store, session_id)
        if not session:
            return []
        parent_id = session.get("parent_session_id")
        parent = self._get_session_dict(store, parent_id) if parent_id else None
        if parent and not self._is_compression_child(session, parent):
            return [session_id]
        # Climb to the root through unmarked compression continuations.
        root = session
        seen = {root["id"]}
        while (pid := root.get("parent_session_id")) and pid not in seen:
            par = self._get_session_dict(store, pid)
            if not par or not self._is_compression_child(root, par):
                break
            root = par
            seen.add(pid)
        lineage = [root["id"]]
        seen = {root["id"]}
        current = root
        while (current.get("end_reason") == "compression"):
            children = store.all(
                "SELECT * FROM sessions WHERE parent_session_id = %s ORDER BY started_at ASC",
                current["id"])
            next_child = next((dict(row) for row in children
                              if self._is_compression_child(dict(row), current)), None)
            if not next_child or next_child["id"] in seen:
                break
            lineage.append(next_child["id"])
            seen.add(next_child["id"])
            current = next_child
        # Later tips are included only when the requested session itself was compacted.
        return lineage if session_id in lineage else [session_id]

    @staticmethod
    def _is_compression_child(child: dict, parent: dict) -> bool:
        """Whether *child* continues compression-ended *parent* (the shared continuation predicate:
        the ``model_config`` fork markers naming *parent*'s id and the ``tool`` source reject)."""
        if parent.get("end_reason") != "compression":
            return False
        config = child.get("model_config")
        if isinstance(config, str):
            try:
                config = json.loads(config)
            except (ValueError, TypeError):
                config = {}
        config = config if isinstance(config, dict) else {}
        for marker in ("_branched_from", "_delegate_from", "_reset_from"):
            if str(config.get(marker) or "") == str(parent.get("id")):
                return False
        return (child.get("source") or "") != "tool"

    @classmethod
    def _export_all(cls, store, source: Optional[str] = None, include_compacted: bool = False,
                    include_inactive: bool = False) -> tuple[SessionExport, ...]:
        # Order by the store recency read (the SQLite ``export_all`` ordering): the newest activity of
        # last_activity_at, in-window message timestamps and started_at - not a two-column coalesce.
        recency = session_last_active_sql("s")
        sessions = store.all(
            f"SELECT * FROM sessions s WHERE s.source = %s "
            f"ORDER BY {recency} DESC, s.started_at DESC, s.id DESC LIMIT 100000", source) \
            if source else store.all(
            f"SELECT * FROM sessions s ORDER BY {recency} DESC, s.started_at DESC, s.id DESC LIMIT 100000")
        session_dicts = []
        for row in sessions:
            session = dict(row)
            session["last_active"] = store.scalar(
                f"SELECT {session_last_active_sql('s')} AS v FROM sessions s WHERE s.id = %s",
                row["id"])
            session_dicts.append(session)
        if include_compacted:
            return tuple(cls._with_messages(store, session, True, include_inactive)
                        for session in session_dicts)
        messages_by_session = {session["id"]: [] for session in session_dicts}
        session_ids = list(messages_by_session)
        active_clause = "" if include_inactive else (
            " AND (active = 1 OR compacted = 1)" if include_compacted else " AND active = 1")
        # Chunked reads keep the ``IN`` list small (the SQLite 900-id chunking).
        for chunk in _id_chunks(session_ids):
            ph = _placeholders(len(chunk))
            rows = store.all(
                f"SELECT * FROM messages WHERE session_id IN ({ph}) {active_clause} "
                f"ORDER BY session_id, id", *chunk)
            for row in rows:
                messages_by_session[row["session_id"]].append(
                    _export_message_dict(row, summary_flag=True))
        return tuple(SessionExport(session=_export_row(session, session.get("last_active")),
                                   messages=tuple(map(_transcript_row, messages_by_session[session["id"]])))
                     for session in session_dicts)

    def export_all(self, source: Optional[str] = None, include_compacted: bool = False,
                   include_inactive: bool = False) -> tuple[SessionExport, ...]:
        with translated():
            return self._export_all(self._store, source, include_compacted, include_inactive)

    def _store_system_prompt(self, conn, system_prompt: Optional[str]) -> Optional[str]:
        """The prompt store's upsert on ``conn`` (the holder's ``_store_system_prompt``)."""
        import hashlib as _hashlib
        if system_prompt is None:
            return None
        prompt_hash = _hashlib.sha256(system_prompt.encode("utf-8")).hexdigest()
        conn.execute(
            "INSERT INTO system_prompts (hash, prompt) VALUES (%s, %s) ON CONFLICT (hash) DO NOTHING",
            (prompt_hash, system_prompt))
        return prompt_hash

    def _delete_unreferenced_system_prompts(self, conn) -> None:
        conn.execute(
            "DELETE FROM system_prompts WHERE NOT EXISTS ("
            "SELECT 1 FROM sessions WHERE sessions.system_prompt_hash = system_prompts.hash) "
            "AND NOT EXISTS ("
            "SELECT 1 FROM sessions WHERE sessions.tool_names = system_prompts.hash)")

    def _import_message_rows(self, conn, session_id: str, messages: list[dict[str, Any]]) -> tuple[int, int]:
        """Insert *messages* as fresh active rows in the caller's txn -> ``(inserted, tool_call_count)``
        (the PG ``_insert_message_rows`` twin: the PG messages repo's row params, the ids read back
        via ``RETURNING``)."""
        inserted = tool_calls_total = 0
        now_ts = time.time()
        for msg in messages:
            role = str(msg.get("role") or "unknown")
            tool_calls = _parse_tool_calls(msg.get("tool_calls"))
            message_timestamp = coerce_epoch(msg.get("timestamp"), field="message timestamp")
            if message_timestamp is None:
                message_timestamp = now_ts
            if msg.get("active") is False:
                pass  # the caller re-marks archived rows after the insert
            params = self._store.messages._message_row_params(session_id, role, msg, tool_calls, message_timestamp)
            row = conn.execute(_INSERT_MESSAGE_SQL, params).fetchone()
            msg["timestamp"] = message_timestamp
            if row is not None:
                msg["_row_id"] = int(row["id"])
            inserted += 1
            tool_calls_total += _tool_calls_count(tool_calls)
            now_ts = max(now_ts, message_timestamp) + 1e-6
        return inserted, tool_calls_total

    def _import_session_row(self, conn, raw: dict[str, Any], messages: list[dict[str, Any]], session_id: str) -> None:
        """INSERT one normalized session + its messages; counts fixed up after (the SQLite
        ``_import_session_row`` on the PG dialect; the ``parent_session_id`` and the archived-row
        flags are applied by the caller and :meth:`_attach_import_parents`)."""
        started_at = coerce_epoch(raw.get("started_at"), session_id=session_id, field="started_at")
        params = {
            "id": session_id, "source": str(raw.get("source") or "import"),
            "system_prompt_hash": self._store_system_prompt(conn, raw.get("system_prompt")),
            "started_at": time.time() if started_at is None else started_at,
            **{col: 1 if _coerce_or(raw.get(col), int, 0) else 0 for col in _IMPORT_FLAG_COLS},
            **{col: raw.get(col) for col in _IMPORT_PASSTHROUGH_COLS},
            **{col: _import_float_or_finite(raw.get(col), col) for col in _IMPORT_FLOAT_COLS},
            **{col: _import_int_or_bounded(raw.get(col), col, 0) for col in _IMPORT_INT_COLS},
        }
        params["auto_archived"] &= params["archived"]  # only the sweep sets it, and only on rows it archives
        conn.execute(_IMPORT_SESSION_INSERT_SQL, (
            params["id"], params["source"], params["user_id"], params["model"], params["model_config"],
            params["system_prompt_hash"], params["started_at"], params["ended_at"],
            params["end_reason"], params["input_tokens"], params["output_tokens"],
            params["cache_read_tokens"], params["cache_write_tokens"], params["reasoning_tokens"],
            params["cwd"], params["git_branch"], params["git_repo_root"],
            params["billing_provider"], params["billing_base_url"], params["billing_mode"],
            params["estimated_cost_usd"], params["actual_cost_usd"], params["cost_status"],
            params["cost_source"], params["pricing_version"], params["title"],
            params["api_call_count"], params["archived"], params["auto_archived"],
            params["pinned"], params["hidden"]))
        def _json_value(value: Any) -> Any:
            return safe_json_loads(value, default=value) if isinstance(value, str) else value
        sanitized_messages = [
            {**msg, **{key: _json_value(msg.get(key)) for key in _IMPORT_MESSAGE_JSON_FIELDS}} for msg in messages
        ]
        # A row exported archived (``include_inactive``) must stay archived: inserted live, compacted or
        # rewound turns would re-enter model context. Flags are coerced like the int session columns, so a
        # hand-edited "0" archives and a missing/null/unparsable flag imports live (older exports have none).
        live: list[dict[str, Any]] = []
        archived: list[dict[str, Any]] = []
        for msg in sanitized_messages:
            (live if _coerce_or(msg.get("active"), int, 1) else archived).append(msg)
        self._import_message_rows(conn, session_id, sanitized_messages)
        for msg in archived:
            if msg.get("_row_id") is not None:
                conn.execute("UPDATE messages SET active = 0, compacted = %s WHERE id = %s",
                             (1 if _coerce_or(msg.get("compacted"), int, 0) else 0, msg["_row_id"]))
        # Session counters count live rows only.
        conn.execute("UPDATE sessions SET message_count = %s, tool_call_count = %s WHERE id = %s",
                     (len(live),
                      sum(_tool_calls_count(_parse_tool_calls(msg.get("tool_calls"))) for msg in live),
                      session_id))

    def import_sessions(self, sessions: Sequence[Mapping[str, Any]]) -> ImportResult:
        if not isinstance(sessions, (list, tuple)):
            raise ValueError("sessions must be a list")
        if len(sessions) > IMPORT_MAX_SESSIONS:
            raise ValueError(f"sessions must contain at most {IMPORT_MAX_SESSIONS} entries")
        normalized, errors = _validate_import_payload(list(sessions))
        if errors:
            return ImportResult(ok=False, imported=0, skipped=0, detached=0, errors=tuple(errors))

        def _do(conn):
            # A user-deleted id stays fenced for good: its row restored beside the fence would list
            # fine and refuse every turn, so the import is refused by name and writes nothing.
            ids = [str(item["session"].get("id") or "").strip() for item in normalized]
            if deleted := [sid for sid in ids if _retired_in(conn, sid)]:
                return ImportResult(ok=False, imported=0, skipped=0, detached=0, errors=tuple(
                    f"session {sid} was deleted here; a deleted session is not restored" for sid in deleted))
            imported_ids: list[SessionId] = []
            skipped_ids: list[SessionId] = []
            parent_updates: list[tuple[str, str]] = []
            for item in normalized:
                raw = item["session"]
                session_id = str(raw.get("id") or "").strip()
                if conn.execute("SELECT 1 FROM sessions WHERE id = %s LIMIT 1", (session_id,)).fetchone():
                    skipped_ids.append(SessionId(session_id))
                    continue
                self._import_session_row(conn, raw, item["messages"], session_id)
                parent_id = str(raw.get("parent_session_id") or "").strip()
                if parent_id:
                    parent_updates.append((session_id, parent_id))
                imported_ids.append(SessionId(session_id))
            return ImportResult(
                ok=True, imported=len(imported_ids), skipped=len(skipped_ids),
                detached=_attach_import_parents(conn, parent_updates),
                imported_ids=tuple(imported_ids), skipped_ids=tuple(skipped_ids),
            )

        with translated():
            with self._store.tx() as conn:
                return _do(conn)

    def find_foreign_import(self, origin: ForeignOrigin) -> Optional[SessionId]:
        """The id of a foreign snapshot already imported for *origin* (the SQLite
        ``find_foreign_import``; the match is the stored ``imported_from`` provenance)."""
        with translated():
            found = _find_foreign_import_on_conn(self._store, origin.as_dict())
        return None if found is None else SessionId(found)

    def import_foreign_history(self, origin: ForeignOrigin, messages: Sequence[ForeignTurn], *, title: str,
                               cwd: Optional[str], profile: str) -> ForeignImportResult:
        """Adopt or mint a foreign snapshot in one transaction, including its provenance.
        Reuse the portability validator and message writer so counters obey the same contract as
        ordinary transcript imports; a taken minted id fails the insert and rolls back."""
        origin_dict = origin.as_dict()
        turns = [{"role": turn.role, "content": turn.content} for turn in messages]
        session_id = new_session_id(hex_len=12)
        normalized, errors = _validate_import_payload([
            {"id": session_id, "source": origin_dict["tool"], "title": title,
             "cwd": cwd, "messages": turns}])
        if errors:
            raise ValueError(errors[0]["error"])

        def _do(conn):
            # Two concurrent imports of one origin race the check-then-mint; the transaction-level
            # advisory lock (keyed on the origin identity) serialises them, so the loser re-reads after
            # the winner commits and reports ``already_imported``.
            conn.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", (
                f"foreign-import:{origin_dict['tool']}:"
                f"{origin_dict.get('foreign_session_id') or origin_dict.get('path') or ''}",))
            existing = self._find_foreign_import_on_txn(conn, origin_dict)
            if existing:
                return ForeignImportResult(session_id=SessionId(existing), already_imported=True)
            # Titles are globally unique within a profile. Preserve a readable
            # title while giving unrelated conversations with the same text room.
            item = normalized[0]
            if conn.execute("SELECT 1 FROM sessions WHERE title = %s", (title,)).fetchone():
                item["session"]["title"] = f"{title} ({session_id[-12:]})"
            self._import_session_row(conn, item["session"], item["messages"], session_id)
            conn.execute("UPDATE sessions SET origin_json = %s, profile_name = %s WHERE id = %s",
                         (json.dumps({"imported_from": origin_dict}), profile, session_id))
            return ForeignImportResult(session_id=SessionId(session_id), already_imported=False)

        with translated():
            try:
                with self._store.tx() as conn:
                    return _do(conn)
            except pg_errors.UniqueViolation as exc:
                # The minted id is the primary key: a taken one fails the insert and the write rolls back.
                raise Conflict("session", session_id) from exc
            except PermanentStoreError as exc:
                # The transaction's translation turns the raw unique violation into a permanent store error
                # before this handler runs, so rewrap just that case as the contract's conflict.
                if isinstance(exc.__cause__, pg_errors.UniqueViolation):
                    raise Conflict("session", session_id) from exc
                raise

    def _find_foreign_import_on_txn(self, conn, origin: dict) -> Optional[str]:
        for row in conn.execute(
                "SELECT id, origin_json FROM sessions WHERE source = %s AND origin_json IS NOT NULL",
                (origin["tool"],)).fetchall():
            imported = (safe_json_loads(row["origin_json"], default={}) or {}).get("imported_from", {})
            if imported.get("tool") != origin["tool"]:
                continue
            foreign_id = origin.get("foreign_session_id")
            if ((foreign_id and foreign_id == imported.get("foreign_session_id"))
                    or (not foreign_id and imported.get("path") == origin["path"])):
                return row["id"]
        return None


def _find_foreign_import_on_conn(store, origin: dict) -> Optional[str]:
    """The store-read arm of the foreign-import match (``find_foreign_import`` through the store's
    read connection; the txn arm runs on the write connection in ``import_foreign_history``)."""
    for row in store.all(
            "SELECT id, origin_json FROM sessions WHERE source = %s AND origin_json IS NOT NULL",
            origin["tool"]):
        imported = (safe_json_loads(row["origin_json"], default={}) or {}).get("imported_from", {})
        if imported.get("tool") != origin["tool"]:
            continue
        foreign_id = origin.get("foreign_session_id")
        if ((foreign_id and foreign_id == imported.get("foreign_session_id"))
                or (not foreign_id and imported.get("path") == origin["path"])):
            return row["id"]
    return None


# The lineage column set for the archive walk (the SQLite ``_LINEAGE_CTE_SQL`` on the PG dialect:
# compression ancestors and descendants through unmarked continuations).
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
                  AND {json_marker_sql('child.model_config', '_branched_from')} IS NULL
                  AND {json_marker_sql('child.model_config', '_delegate_from')} IS NULL
                  AND {json_marker_sql('child.model_config', '_reset_from')} IS NULL
                  AND COALESCE(child.source, '') != 'tool'
              ),
              descendants(id) AS (
                SELECT %s
                UNION
                SELECT child.id
                FROM descendants d
                JOIN sessions parent ON parent.id = d.id
                JOIN sessions child ON child.parent_session_id = parent.id
                WHERE parent.end_reason = 'compression'
                  AND {json_marker_sql('child.model_config', '_branched_from')} IS NULL
                  AND {json_marker_sql('child.model_config', '_delegate_from')} IS NULL
                  AND {json_marker_sql('child.model_config', '_reset_from')} IS NULL
                  AND COALESCE(child.source, '') != 'tool'
              ),
              lineage(id) AS (
                SELECT id FROM ancestors
                UNION
                SELECT id FROM descendants
              )"""


add_stubs(PgSessionMaintenance, SessionMaintenance,
          implemented=(
              "assert_export_safe", "auto_archive_lineage", "clear_backend_heartbeat",
              "count_open_prune_matches", "count_prune_matches", "export_all", "export_session",
              "export_session_lineage", "find_foreign_import", "import_foreign_history",
              "import_sessions", "list_idle_archive_candidates", "list_prune_candidates",
              "prune_empty_ghost_sessions", "prune_reclaim_estimate", "prune_sessions",
              "register_backend_heartbeat", "retained_tree_is_live", "session_projection_digest",
              "sweep_orphaned_sessions", "tree_served_sessions"))


def _register() -> None:
    register_repository("maintenance", lambda store: PgSessionMaintenance(store))


_register()

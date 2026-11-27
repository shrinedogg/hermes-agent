"""``RoutingRepository`` on PostgreSQL (gateway routing stage 4): the gateway routing index, the routing
columns of the bound store's ``sessions`` rows, peer/orphan recovery, never-active pruning and the
Telegram DM topic tables, each verb one write transaction (or a fixed sequence of them) through the
store handle - the SQLite routing store's SQL ported to the PG dialect.

The retirement predicates read the entry's ``session_id`` out of the TEXT ``entry_json`` (the SQLite
``json_extract`` twin); a malformed entry text raises the store's ``malformed JSON`` refusal exactly
where the reference's ``json_extract`` raised it. Topic tables are created on explicit opt-in
(``prepare_topic_storage``) inside the store's own write; a store without them reads empty and its
cleanups are no-ops, as on the reference.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterable
from typing import Any, Mapping, Optional

from persistence import PermanentStoreError, SessionArtifacts, SessionId, SessionRowMissing
from persistence.authority.vocabulary import RETRY_HORIZON_S
from persistence.conversation.sessions import RECOVERABLE_END_REASONS
from persistence.gateway.routing import (
    ORPHAN_ADOPTION_MAX_GAP_S,
    PEER_RESET_END_REASONS,
    TOPIC_SCHEMA_META_KEY,
    NeverActivePrune,
    NeverActiveSession,
    OrphanReport,
    RecoveredPeerSession,
    RoutingRepository,
    TopicAlreadyLinked,
    TopicBinding,
    normalize_topic_profile_name,
    route_namespace,
)
from persistence.sqlite._schema import _title_rank
from persistence.sqlite.conversation._store.local import POLICY_PREFIX
from persistence.sqlite.conversation._store.local_migration import LEGACY_PREFIX
from persistence.sqlite.conversation._store.retirement import RETIRED_PREFIX, entry_session_id
from persistence.sqlite.gateway._routes import ROUTE_FENCE_PREFIX

from .._backend import PostgresStore, register_repository, translated
from .._lineage_common import (
    json_marker_absent_sql,
    json_marker_sql,
    reset_child_sql,
    session_last_active_sql,
)
from .._stubs import add_stubs
from ..conversation.sessions import _mark_retired, _row_to_record, session_record

__all__ = ["PgRoutingRepository"]

# ── shared SQL (the SQLite ``_routes`` shapes on the PG dialect) ──────────────

_RECOVERABLE_END_REASONS_SQL = ", ".join(f"'{reason}'" for reason in sorted(RECOVERABLE_END_REASONS))
_RESET_END_REASONS_SQL = ", ".join(f"'{reason}'" for reason in sorted(PEER_RESET_END_REASONS))

# The ASCII-only source folding of the reference's SQLite ``LOWER`` (SQLite's built-in
# ``lower`` folds only the ASCII letters; PostgreSQL's ``lower`` is locale-dependent, so the
# fold is an explicit ``translate`` over the ASCII range).
_ASCII_UPPER = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
_ASCII_LOWER = "abcdefghijklmnopqrstuvwxyz"

# A route retirement is judged against the profile the route belongs to: this store's own
# ``RETIRED_PREFIX`` marker for a ``main`` route, and the ``ROUTE_FENCE_PREFIX`` fence (keyed
# (namespace, id)) for the key's namespace - a ``main`` fence only while this store holds no
# session of the id. A sweep (``expires``) fence ages out at the retry horizon; a permanent one
# never does (its age test is NULL and reads as not expired). Every JSON read sits in a
# ``hermes_json_valid`` guard (a bad ``::jsonb`` cast would raise).
_ROUTE_RETIRED_SQL = f"""(
    EXISTS (SELECT 1 FROM state_meta m
            WHERE m.key = '{RETIRED_PREFIX}' || %s
              AND (%s = 'main' OR NOT CASE WHEN hermes_json_valid(m.value)
                                          THEN (m.value::jsonb -> 'own') IS NOT NULL
                                          ELSE false END))
    OR EXISTS (SELECT 1 FROM state_meta f
               WHERE f.key = '{ROUTE_FENCE_PREFIX}' || %s || ':' || %s
                 AND NOT CASE WHEN hermes_json_valid(f.value)
                              THEN COALESCE((f.value::jsonb ->> 'expires') = 'true'
                                            AND ((f.value::jsonb ->> 'retired_at')::double precision
                                                 < %s - %s),
                                            false)
                              ELSE false END
                 AND (%s <> 'main' OR NOT EXISTS (
                         SELECT 1 FROM sessions s WHERE s.id = %s)))
)"""


def _route_retired_params(session_id: Optional[str], namespace: str, now: float) -> tuple:
    """The named parameters of :data:`_ROUTE_RETIRED_SQL` for one routing entry saved at *now*."""
    return (session_id, namespace, namespace, session_id, now, RETRY_HORIZON_S, namespace, session_id)


_UPSERT_ROUTING_SQL = (
    "INSERT INTO gateway_routing (scope, session_key, entry_json, updated_at)"
    " SELECT %s, %s, %s, %s WHERE NOT " + _ROUTE_RETIRED_SQL +
    " ON CONFLICT (scope, session_key) DO UPDATE SET"
    " entry_json = EXCLUDED.entry_json,"
    " updated_at = EXCLUDED.updated_at"
)

# Recursive CTE naming a session plus its compression ancestors (rows a resume must keep on one
# routing peer); branch/delegate/tool/reset rows stop the walk (the SQLite
# ``_COMPRESSION_LINEAGE_CTE`` membership rule).
_COMPRESSION_LINEAGE_CTE = (
    "WITH RECURSIVE compression_lineage(id) AS ("
    " SELECT %s"
    " UNION"
    " SELECT parent.id"
    " FROM compression_lineage lineage"
    " JOIN sessions child ON child.id = lineage.id"
    " JOIN sessions parent ON parent.id = child.parent_session_id"
    " WHERE parent.end_reason = 'compression'"
    f" AND {json_marker_absent_sql('child.model_config', '_branched_from')}"
    f" AND {json_marker_absent_sql('child.model_config', '_delegate_from')}"
    f" AND NOT ({reset_child_sql('child')})"
    " AND COALESCE(child.source, '') <> 'tool')"
)
_COMPRESSION_LINEAGE_UPDATE_WHERE = " WHERE id IN (SELECT id FROM compression_lineage)"

# Projection shared by both peer-recovery queries (exact key, then peer tuple).
_PEER_SELECT_HEAD = (
    " SELECT s.*,"
    " COALESCE(sp.prompt, s.system_prompt) AS _system_prompt_resolved,"
    " (COALESCE(s.message_count, 0) > 0 OR EXISTS ("
    "  SELECT 1 FROM messages WHERE messages.session_id = s.id LIMIT 1"
    "  )) AS _has_messages"
    " FROM sessions s"
    " LEFT JOIN system_prompts sp ON sp.hash = s.system_prompt_hash"
)
# Gateway routing recovery must never land on a delegate child (the SQLite
# ``_DELEGATE_CHILD_EXCLUSION_SQL``: anchored on the ``_delegate_from`` marker, kept for a
# ``_reset_from`` or ``_chat_adopted`` row that is the conversation itself).
_DELEGATE_CHILD_EXCLUSION_SQL = (
    f" AND (COALESCE({json_marker_sql('s.model_config', '_delegate_from')}, '') = ''"
    f"  OR {json_marker_sql('s.model_config', '_reset_from')} IS NOT NULL"
    f"  OR {json_marker_sql('s.model_config', '_chat_adopted')} IS NOT NULL)"
)
# A row a completed handoff owns stays recoverable whatever end_reason it carries (the
# SQLite ``_HANDOFF_OWNED_ROW_SQL``).
_PEER_BY_KEY_SQL = (
    _PEER_SELECT_HEAD +
    " WHERE s.session_key = %s"
    "  AND s.source = %s"
    f" {_DELEGATE_CHILD_EXCLUSION_SQL}"
    "  AND (s.ended_at IS NULL OR s.end_reason IN (" + _RECOVERABLE_END_REASONS_SQL + ")"
    "   OR s.handoff_state = 'completed')"
    "  AND NOT EXISTS ("
    "   SELECT 1 FROM sessions b"
    "   WHERE b.session_key = s.session_key"
    "    AND b.source = s.source"
    "    AND b.ended_at IS NOT NULL"
    f"    AND b.end_reason IN ({_RESET_END_REASONS_SQL})"
    "    AND b.ended_at > COALESCE(s.last_activity_at, s.started_at)"
    "  )"
    " ORDER BY _has_messages DESC, COALESCE(s.last_activity_at, s.started_at) DESC"
    " LIMIT 1"
)
# Tuple arm: the profile fence keeps a sibling profile's row (in a shared legacy store) from
# being adopted - a row is ours when its profile_name is the store owner's or NULL (the SQLite
# ``? IS NULL OR COALESCE(s.profile_name, ?) = ?`` guard with the owner derived, or none).
_PEER_BY_TUPLE_SQL = (
    _PEER_SELECT_HEAD +
    " WHERE s.source = %s"
    "  AND COALESCE(s.user_id, '') = COALESCE(%s, '')"
    "  AND COALESCE(s.chat_id, '') = COALESCE(%s, '')"
    "  AND COALESCE(s.chat_type, '') = COALESCE(%s, '')"
    "  AND COALESCE(s.thread_id, '') = COALESCE(%s, '')"
    f" {_DELEGATE_CHILD_EXCLUSION_SQL}"
    "  AND (%s::text IS NULL OR COALESCE(s.profile_name, %s::text) = %s::text)"
    "  AND (s.ended_at IS NULL OR s.end_reason IN (" + _RECOVERABLE_END_REASONS_SQL + ")"
    "   OR s.handoff_state = 'completed')"
    "  AND (COALESCE(s.message_count, 0) > 0 OR EXISTS ("
    "   SELECT 1 FROM messages WHERE messages.session_id = s.id LIMIT 1"
    "  ))"
    "  AND NOT EXISTS ("
    "   SELECT 1 FROM sessions b"
    "   WHERE b.source = s.source"
    "    AND COALESCE(b.user_id, '') = COALESCE(s.user_id, '')"
    "    AND COALESCE(b.chat_id, '') = COALESCE(s.chat_id, '')"
    "    AND COALESCE(b.chat_type, '') = COALESCE(s.chat_type, '')"
    "    AND COALESCE(b.thread_id, '') = COALESCE(s.thread_id, '')"
    "    AND (%s::text IS NULL OR COALESCE(b.profile_name, %s::text) = %s::text)"
    "    AND b.ended_at IS NOT NULL"
    f"    AND b.end_reason IN ({_RESET_END_REASONS_SQL})"
    "    AND b.ended_at > COALESCE(s.last_activity_at, s.started_at)"
    "  )"
    " ORDER BY COALESCE(s.last_activity_at, s.started_at) DESC"
    " LIMIT 1"
)

_ORPHAN_DONOR_COLUMNS = (
    "d.id, d.session_key, d.chat_id, d.chat_type, d.thread_id, "
    "d.user_id, d.origin_json, d.display_name, d.end_reason"
)
# Donor profile fence: the donor's effective profile (NULL reads as the store owner) must equal
# the orphan's (the SQLite ``_ORPHAN_DONOR_PROFILE_FENCE_SQL``; stores outside the profile tree
# derive no owner and keep the historical unfenced behavior).
_ORPHAN_DONOR_PROFILE_FENCE_SQL = " AND (%s::text IS NULL OR COALESCE(d.profile_name, %s::text) = %s::text)"
_ORPHANS_SQL = (
    " SELECT o.id, o.source, o.user_id, o.started_at,"
    "  o.parent_session_id, o.profile_name,"
    f"  {session_last_active_sql('o')} AS last_active,"
    "  (SELECT COUNT(*) FROM messages m WHERE m.session_id = o.id) AS message_count"
    " FROM sessions o"
    " WHERE o.session_key IS NULL"
    "  AND EXISTS (SELECT 1 FROM messages m WHERE m.session_id = o.id)"
    "  AND COALESCE(o.source, '') <> 'tool'"
    f"  AND {json_marker_absent_sql('o.model_config', '_branched_from')}"
    f"  AND {json_marker_absent_sql('o.model_config', '_delegate_from')}"
    " ORDER BY o.started_at ASC"
)
_ORPHAN_LINEAGE_DONOR_SQL = (
    f" SELECT {_ORPHAN_DONOR_COLUMNS}"
    "  FROM sessions d"
    "  WHERE d.id = %s"
    "   AND d.session_key IS NOT NULL"
    "   AND COALESCE(d.source, '') = COALESCE(%s, '')"
    + _ORPHAN_DONOR_PROFILE_FENCE_SQL
)
_ORPHAN_CONTIGUITY_DONORS_SQL = (
    f" SELECT {_ORPHAN_DONOR_COLUMNS}, {session_last_active_sql('d')} AS last_active"
    "  FROM sessions d"
    "  WHERE d.session_key IS NOT NULL"
    "   AND d.id <> %s"
    "   AND COALESCE(d.source, '') = COALESCE(%s, '')"
    "   AND (COALESCE(d.user_id, '') = ''"
    "        OR COALESCE(%s, '') = ''"
    "        OR d.user_id = %s)"
    f"   AND {session_last_active_sql('d')} BETWEEN %s AND %s"
    f"   AND {session_last_active_sql('d')} < %s"
    "  AND (%s::text IS NULL OR COALESCE(d.profile_name, %s::text) = %s::text)"
    "  ORDER BY last_active DESC"
    "  LIMIT 2"
)

# ── Telegram DM topic tables (created by ``prepare_topic_storage`` on opt-in) ──

_TOPIC_TABLE_MODE_SQL = """CREATE TABLE IF NOT EXISTS telegram_dm_topic_mode (
    profile_name TEXT NOT NULL DEFAULT 'default',
    chat_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    enabled BIGINT NOT NULL DEFAULT 1,
    activated_at DOUBLE PRECISION NOT NULL,
    updated_at DOUBLE PRECISION NOT NULL,
    has_topics_enabled BIGINT,
    allows_users_to_create_topics BIGINT,
    capability_checked_at DOUBLE PRECISION,
    intro_message_id TEXT,
    pinned_message_id TEXT,
    PRIMARY KEY (profile_name, chat_id)
)"""
_TOPIC_TABLE_BINDINGS_SQL = """CREATE TABLE IF NOT EXISTS telegram_dm_topic_bindings (
    profile_name TEXT NOT NULL DEFAULT 'default',
    chat_id TEXT NOT NULL,
    thread_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    session_key TEXT NOT NULL,
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    managed_mode TEXT NOT NULL DEFAULT 'auto',
    linked_at DOUBLE PRECISION NOT NULL,
    updated_at DOUBLE PRECISION NOT NULL,
    PRIMARY KEY (profile_name, chat_id, thread_id)
)"""


# ── the local-conversation hand-off (the SQLite ``_local_lineage`` twins) ────


def _receipt_json(raw: Optional[str]) -> Optional[dict]:
    if raw is None:
        return None
    try:
        value = json.loads(raw)
    except (TypeError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _moved_receipts(conn, target_ids: Optional[set] = None) -> dict:
    """``{owner id: (moved row id, its source)}`` of local receipts whose current physical target
    row now carries another lane's route (the SQLite ``moved_receipts``): only the current target
    counts, and *target_ids* narrows the scan to the rows one repoint rewrote."""
    rows = conn.execute(
        """SELECT substring(m.key, %s) AS owner, s.id AS target, s.source AS source
           FROM state_meta m
           JOIN sessions s ON s.id = CASE WHEN hermes_json_valid(m.value)
                                          THEN m.value::jsonb -> 'entry' ->> 'session_id' END
           WHERE m.key LIKE %s
             AND s.session_key IS NOT NULL
             AND s.session_key IS DISTINCT FROM (CASE WHEN hermes_json_valid(m.value)
                                                     THEN m.value::jsonb ->> 'route' END)""",
        (len(POLICY_PREFIX) + 1, POLICY_PREFIX + "%")).fetchall()
    return {row["owner"]: (row["target"], row["source"]) for row in rows
            if target_ids is None or row["target"] in target_ids}


_LANES_IDENTITY = "session_key, source, user_id, chat_id, chat_type, thread_id, display_name, origin_json, transport_profile"


def _hand_off_conversations(conn, moved: dict) -> dict:
    """Retire the receipts in *moved* and move each conversation's other segments onto the lane
    (the SQLite ``hand_off_conversations``); a receipt whose lineage is malformed stays as it is."""
    handed: dict = {}
    for owner, (target, source) in moved.items():
        saved = conn.execute("SELECT value FROM state_meta WHERE key = %s", (POLICY_PREFIX + owner,)).fetchone()
        receipt = _receipt_json(saved["value"]) if saved else None
        lineage = (receipt or {}).get("lineage", [owner])
        if not isinstance(lineage, list) or not all(isinstance(sid, str) for sid in lineage):
            continue
        segments = list(dict.fromkeys([owner, *lineage]))
        conn.execute(
            f" UPDATE sessions SET ({_LANES_IDENTITY}) = "
            f" (SELECT {_LANES_IDENTITY} FROM sessions WHERE id = %s)"
            " WHERE id = ANY(%s) AND id <> %s",
            (target, segments, target))
        _title_follows_head(conn, owner, _segment_root(conn, segments, target))
        handed[owner] = (target, source)
    for owner in handed:
        for key in (POLICY_PREFIX + owner, LEGACY_PREFIX + owner):
            conn.execute("DELETE FROM state_meta WHERE key = %s", (key,))
    return handed


def _segment_root(conn, segments: list, target: str) -> str:
    """The first row of *target*'s ``/new`` segment: back through the compressions behind it."""
    if target not in segments:
        return target
    index = segments.index(target)
    while index > 0:
        row = conn.execute(
            "SELECT end_reason FROM sessions WHERE id = %s", (segments[index - 1],)).fetchone()
        if row is None or row["end_reason"] != "compression":
            break
        index -= 1
    return segments[index]


def _title_follows_head(conn, owner: str, head: str) -> None:
    """The conversation's name moves to its head, over a title of lower provenance there (a
    generated one); a name of equal rank the head carries stays (the SQLite ``_title_follows_head``)."""
    row = conn.execute(
        "SELECT title, title_source, hidden FROM sessions WHERE id = %s", (owner,)).fetchone()
    if head == owner or row is None or not row["title"]:
        return
    kept = conn.execute(
        "SELECT title, title_source FROM sessions WHERE id = %s", (head,)).fetchone()
    if kept is None or (kept["title"] and _title_rank(kept["title_source"]) >= _title_rank(row["title_source"])):
        return
    conn.execute("UPDATE sessions SET title = NULL WHERE id IN (%s, %s)", (owner, head))
    conn.execute(
        "UPDATE sessions SET title = %s, title_source = %s, hidden = %s WHERE id = %s",
        (row["title"], row["title_source"], row["hidden"], head))


class PgRoutingRepository:
    """Routing index, routing peers and Telegram DM topic state of the bound profile's store
    schema; every protocol verb ported from the SQLite reference, unported verbs raise naming
    themselves (D8)."""

    def __init__(self, store: PostgresStore) -> None:
        self._store = store

    # ── entry parsing (the reference's ``json_extract`` twins) ──────────────

    @staticmethod
    def _entry_session_id_strict(entry_json: str) -> Optional[str]:
        """The entry's ``session_id`` as the row id it fences on (the SQLite ``json_extract`` twin:
        a JSON number reads back as its text form), a string id as-is; a malformed text raises the
        store's ``malformed JSON`` refusal, which the reference's ``json_extract`` raised too and
        which gates the write (an entry that names nothing can be stored, one that is not JSON
        cannot)."""
        try:
            entry = json.loads(entry_json)
        except (ValueError, RecursionError):
            raise PermanentStoreError("malformed JSON") from None
        if not isinstance(entry, dict):
            return None
        session_id = entry.get("session_id")
        if isinstance(session_id, str):
            return session_id
        if isinstance(session_id, (int, float)) and not isinstance(session_id, bool):
            return str(session_id)
        return None

    def _routing_entries_for_sessions(self, session_ids: set) -> list[tuple[str, str, str]]:
        """``(scope, session_key, session_id)`` of ``gateway_routing`` rows pointing at any of
        *session_ids*; the target id lives only inside ``entry_json``, so matching is in Python."""
        store = self._store
        found: list[tuple[str, str, str]] = []
        if not session_ids:
            return found
        for row in store.all("SELECT scope, session_key, entry_json FROM gateway_routing"):
            target = entry_session_id(row["entry_json"])
            if target in session_ids:
                found.append((row["scope"], row["session_key"], target))
        return found

    def _local_lineage_owner(self, session_id: str) -> str:
        """The creation id whose receipt lineage holds physical segment *session_id* (a reset or
        compression child, which carries the owner as ``chat_id``); *session_id* itself otherwise
        (the SQLite ``local_lineage_owner`` twin through the store's read door)."""
        store = self._store
        row = store.one("SELECT chat_id FROM sessions WHERE id = %s", session_id)
        owner = row["chat_id"] if row is not None else None
        if not owner or owner == session_id:
            return session_id
        saved = store.one("SELECT value FROM state_meta WHERE key = %s", POLICY_PREFIX + owner)
        receipt = _receipt_json(saved["value"] if saved is not None else None)
        lineage = receipt.get("lineage") if receipt is not None else None
        return owner if isinstance(lineage, list) and session_id in lineage else session_id

    def _local_conversation_ids(self, session_id: str) -> list:
        """Every row of the local conversation *session_id* belongs to (the SQLite
        ``local_conversation_ids`` twin)."""
        store = self._store
        owner = self._local_lineage_owner(session_id)
        saved = store.one("SELECT value FROM state_meta WHERE key = %s", POLICY_PREFIX + owner)
        receipt = _receipt_json(saved["value"] if saved is not None else None)
        lineage = receipt.get("lineage") if receipt is not None else None
        if not isinstance(lineage, list):
            return [session_id]
        return list(dict.fromkeys([owner, *(sid for sid in lineage if isinstance(sid, str))]))

    def _owned_lineage_ids(self, session_id: str) -> list:
        """``[session_id]`` for any row outside a local reset/compression lineage, else the whole
        conversation (*session_id* first) (the SQLite ``owned_lineage_ids`` twin)."""
        return list(dict.fromkeys([session_id, *self._local_conversation_ids(session_id)]))

    # ── writes ──────────────────────────────────────────────────────────────

    def upsert_routing_entry(self, session_key: str, entry_json: str, *, routing_scope: str = "") -> None:
        if not session_key or not entry_json:
            return
        store = self._store
        session_id = self._entry_session_id_strict(entry_json)
        now = time.time()
        with translated():
            with store.tx() as conn:
                conn.execute(
                    _UPSERT_ROUTING_SQL,
                    (routing_scope, session_key, entry_json, now,
                     *_route_retired_params(session_id, route_namespace(session_key), now)))

    def replace_routing_entries(self, entries: Mapping[str, str], *, routing_scope: str = "") -> set[str]:
        store = self._store
        now = time.time()
        rows = [
            (key, entry, self._entry_session_id_strict(entry), route_namespace(key))
            for key, entry in entries.items() if key and entry
        ]
        skipped: set[str] = set()
        with translated():
            with store.tx() as conn:
                conn.execute("DELETE FROM gateway_routing WHERE scope = %s", (routing_scope,))
                for key, entry, session_id, namespace in rows:
                    retired = conn.execute(
                        f"SELECT {_ROUTE_RETIRED_SQL} AS retired",
                        _route_retired_params(session_id, namespace, now)).fetchone()
                    if retired is not None and bool(retired["retired"]):
                        skipped.add(key)
                        continue
                    conn.execute(
                        "INSERT INTO gateway_routing (scope, session_key, entry_json, updated_at)"
                        " VALUES (%s, %s, %s, %s)",
                        (routing_scope, key, entry, now))
        return skipped

    def fence_retired_routes(
        self, session_ids: Iterable[str], *, namespace: str, expiring: Iterable[str] = ()
    ) -> None:
        store = self._store
        ids = sorted({str(sid) for sid in session_ids if sid})
        if not ids:
            return
        expiring = set(expiring)
        with translated():
            with store.tx() as conn:
                self._reclaim_expired_fences(conn, time.time() - RETRY_HORIZON_S)
                owned: dict[str, set] = {}
                for sid in ids:
                    names = {namespace}
                    if conn.execute("SELECT 1 FROM sessions WHERE id = %s", (sid,)).fetchone() is None:
                        names.add("main")
                    for name in sorted(names):
                        _mark_retired(conn, f"{ROUTE_FENCE_PREFIX}{name}:{sid}", expires=sid in expiring)
                    owned[sid] = names
                for row in conn.execute("SELECT scope, session_key, entry_json FROM gateway_routing").fetchall():
                    if route_namespace(row["session_key"]) in owned.get(entry_session_id(row["entry_json"]), ()):
                        conn.execute(
                            "DELETE FROM gateway_routing WHERE scope = %s AND session_key = %s",
                            (row["scope"], row["session_key"]))

    @staticmethod
    def _reclaim_expired_fences(conn, cutoff: float) -> None:
        """Drop the sweep (``expires``) fence markers under :data:`ROUTE_FENCE_PREFIX` retired
        before *cutoff* (the SQLite ``_reclaim_expired``); a permanent marker never does."""
        conn.execute(
            """DELETE FROM state_meta
               WHERE key LIKE %s
                 AND CASE WHEN hermes_json_valid(value)
                          THEN (value::jsonb ->> 'retired_at')::double precision < %s
                               AND (value::jsonb ->> 'expires') = 'true'
                          ELSE false END""",
            (ROUTE_FENCE_PREFIX + "%", cutoff))

    def record_session_peer(
        self, session_id: str, *, source: str, user_id: str | None = None, session_key: str | None = None,
        chat_id: str | None = None, chat_type: str | None = None, thread_id: str | None = None,
        display_name: str | None = None, origin_json: str | None = None,
        include_compression_ancestors: bool = False, messaging_lane: bool = False,
        transport_profile: str | None = None,
    ) -> None:
        if not session_id or not session_key:
            return
        store = self._store
        ancestors = include_compression_ancestors
        identity = (session_key, source, user_id, chat_id, chat_type, thread_id,
                    display_name, origin_json, transport_profile)
        with translated():
            with store.tx() as conn:
                if ancestors:
                    conn.execute(
                        _COMPRESSION_LINEAGE_CTE +
                        """ UPDATE sessions
                               SET session_key = %s, source = %s, user_id = %s, chat_id = %s,
                                   chat_type = %s, thread_id = %s,
                                   display_name = COALESCE(%s, display_name),
                                   origin_json = COALESCE(%s, origin_json),
                                   transport_profile = COALESCE(%s, transport_profile)
                           """ + _COMPRESSION_LINEAGE_UPDATE_WHERE,
                        (session_id, *identity))
                    if messaging_lane:
                        # An explicit repoint to a messaging lane moved these rows: the local
                        # conversation whose receipt names one of them goes with it in the same
                        # write, or it would match neither lane (the SQLite hand-off).
                        moved = {row["id"] for row in conn.execute(
                            _COMPRESSION_LINEAGE_CTE + " SELECT id FROM compression_lineage",
                            (session_id,)).fetchall()}
                        _hand_off_conversations(conn, _moved_receipts(conn, moved))
                    return
                # The UPDATE silently no-ops on a missing row: insert it with the full identity.
                conn.execute(
                    """UPDATE sessions
                           SET session_key = %s, source = %s, user_id = %s, chat_id = %s,
                               chat_type = %s, thread_id = %s,
                               display_name = COALESCE(%s, display_name),
                               origin_json = COALESCE(%s, origin_json),
                               transport_profile = COALESCE(%s, transport_profile)
                         WHERE id = %s""",
                    (*identity, session_id))
                if conn.execute("SELECT 1 FROM sessions WHERE id = %s", (session_id,)).fetchone() is None:
                    # Same ownership stamp as the sessions create: an unowned (NULL) row
                    # vanishes from profile-keyed consumers. The self-heal insert runs no
                    # retirement check: a retired id's row comes back.
                    conn.execute(
                        """INSERT INTO sessions (
                                   id, source, created_source, user_id, session_key, chat_id,
                                   chat_type, thread_id, display_name, origin_json,
                                   profile_name, transport_profile, started_at
                               )
                               VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                               ON CONFLICT (id) DO UPDATE SET
                                   session_key = COALESCE(sessions.session_key, EXCLUDED.session_key),
                                   chat_id = COALESCE(sessions.chat_id, EXCLUDED.chat_id),
                                   chat_type = COALESCE(sessions.chat_type, EXCLUDED.chat_type),
                                   thread_id = COALESCE(sessions.thread_id, EXCLUDED.thread_id),
                                   display_name = COALESCE(sessions.display_name, EXCLUDED.display_name),
                                   origin_json = COALESCE(sessions.origin_json, EXCLUDED.origin_json),
                                   transport_profile = COALESCE(sessions.transport_profile, EXCLUDED.transport_profile),
                                   created_source = COALESCE(sessions.created_source, EXCLUDED.created_source)""",
                        (session_id, source, source, user_id, session_key, chat_id, chat_type,
                         thread_id, display_name, origin_json, store.owner_profile_name,
                         transport_profile, time.time()))

    def adopt_orphan(self, orphan_id: str, donor_id: str) -> bool:
        if not orphan_id or not donor_id or orphan_id == donor_id:
            return False
        owner = self._store.owner_profile_name
        with translated():
            with self._store.tx() as conn:
                donor = conn.execute(
                    """SELECT session_key, chat_id, chat_type, thread_id, user_id,
                              origin_json, display_name, source, profile_name
                       FROM sessions WHERE id = %s""", (donor_id,)).fetchone()
                orphan = conn.execute(
                    "SELECT session_key, source, profile_name FROM sessions WHERE id = %s",
                    (orphan_id,)).fetchone()
                if (donor is None or orphan is None or not donor["session_key"] or orphan["session_key"]
                        or (donor["source"] or "") != (orphan["source"] or "")
                        or (owner is not None
                            and (donor["profile_name"] or owner) != (orphan["profile_name"] or owner))):
                    return False
                conn.execute(
                    """UPDATE sessions
                          SET session_key = %s,
                              chat_id = COALESCE(chat_id, %s),
                              chat_type = COALESCE(chat_type, %s),
                              thread_id = COALESCE(thread_id, %s),
                              user_id = COALESCE(user_id, %s),
                              origin_json = COALESCE(origin_json, %s),
                              display_name = COALESCE(display_name, %s),
                              parent_session_id = COALESCE(parent_session_id, %s)
                        WHERE id = %s AND session_key IS NULL""",
                    (donor["session_key"], donor["chat_id"], donor["chat_type"], donor["thread_id"],
                     donor["user_id"], donor["origin_json"], donor["display_name"], donor_id, orphan_id))
                # Retire under a reason recovery does NOT treat as resumable: an 'agent_close' /
                # 'ws_orphan_reap' stamp would keep the donor in the running and the orphan could
                # lose the chat again.
                conn.execute(
                    """UPDATE sessions SET ended_at = COALESCE(ended_at, %s),
                                          end_reason = 'superseded_by_repair' WHERE id = %s""",
                    (time.time(), donor_id))
                return True

    def prune_never_active(
        self, *, older_than_days: float, artifacts: SessionArtifacts | None = None, media_root: str,
    ) -> NeverActivePrune:
        store = self._store
        candidates = self.never_active_keyed_sessions(older_than_days=older_than_days)
        # A fresh local reset tip is its conversation's current target, not a never-active chat:
        # the conversation holds the earlier segments' history and the owner's queued work.
        with translated():
            ids = {str(row.session_id) for row in candidates
                   if self._owned_lineage_ids(str(row.session_id)) == [str(row.session_id)]}
            if not ids:
                return NeverActivePrune(0, 0, 0, (), ())
            # The delete retires the deleted rows' routing entries inside its own transaction, so
            # count what pointed at the candidates before it commits; the caller's sweep then only
            # catches entries a concurrent writer re-added.
            routed_before = self._routing_entries_for_sessions(ids)
        outcome = store.sessions.delete_many(
            [SessionId(sid) for sid in sorted(ids)], skip_guarded=True, media_root=media_root)
        if artifacts is not None:
            artifacts.remove_for(outcome.removed_ids)
        kept = {str(sid) for sid in outcome.skipped_guarded}
        routing_deleted = sum(1 for _, _, sid in routed_before if sid not in kept)
        return NeverActivePrune(
            outcome.deleted, routing_deleted, len(kept),
            tuple(sorted(ids - kept)), tuple(sorted(outcome.removed_ids)))

    def sweep_routing_entries_for_sessions(self, session_ids: Iterable[str]) -> int:
        store = self._store
        doomed = [(scope, key) for scope, key, _ in
                  self._routing_entries_for_sessions({str(sid) for sid in session_ids})]
        if not doomed:
            return 0
        with translated():
            with store.tx() as conn:
                for scope, key in doomed:
                    conn.execute(
                        "DELETE FROM gateway_routing WHERE scope = %s AND session_key = %s",
                        (scope, key))
        return len(doomed)

    def prepare_topic_storage(self) -> None:
        store = self._store
        with translated():
            with store.tx() as conn:
                conn.execute(_TOPIC_TABLE_MODE_SQL)
                conn.execute(_TOPIC_TABLE_BINDINGS_SQL)
                # Indexes after any table create: the user index needs profile_name.
                conn.execute(
                    "CREATE UNIQUE INDEX IF NOT EXISTS idx_telegram_dm_topic_bindings_session "
                    "ON telegram_dm_topic_bindings(session_id)")
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_telegram_dm_topic_bindings_user "
                    "ON telegram_dm_topic_bindings(profile_name, user_id, chat_id)")
                conn.execute(
                    """INSERT INTO state_meta (key, value) VALUES (%s, %s)
                       ON CONFLICT (key) DO UPDATE SET value = excluded.value""",
                    (TOPIC_SCHEMA_META_KEY, "3"))

    def enable_topic_mode(
        self, *, chat_id: str, user_id: str, profile_name: str = "default",
        has_topics_enabled: bool | None = None, allows_users_to_create_topics: bool | None = None,
    ) -> None:
        store = self._store
        self.prepare_topic_storage()
        now = time.time()
        profile_name = normalize_topic_profile_name(profile_name)

        def _to_int(value: Optional[bool]) -> Optional[int]:
            return None if value is None else (1 if value else 0)

        with translated():
            with store.tx() as conn:
                conn.execute(
                    """INSERT INTO telegram_dm_topic_mode (
                           profile_name, chat_id, user_id, enabled, activated_at, updated_at,
                           has_topics_enabled, allows_users_to_create_topics,
                           capability_checked_at
                       ) VALUES (%s, %s, %s, 1, %s, %s, %s, %s, %s)
                       ON CONFLICT (profile_name, chat_id) DO UPDATE SET
                           user_id = EXCLUDED.user_id,
                           enabled = 1,
                           updated_at = EXCLUDED.updated_at,
                           has_topics_enabled = EXCLUDED.has_topics_enabled,
                           allows_users_to_create_topics = EXCLUDED.allows_users_to_create_topics,
                           capability_checked_at = EXCLUDED.capability_checked_at""",
                    (profile_name, str(chat_id), str(user_id), now, now,
                     _to_int(has_topics_enabled), _to_int(allows_users_to_create_topics), now))

    def disable_topic_mode(self, *, chat_id: str, profile_name: str = "default", clear_bindings: bool = True) -> None:
        if not self._topic_tables_ready():
            return
        store = self._store
        profile_name = normalize_topic_profile_name(profile_name)
        with translated():
            with store.tx() as conn:
                conn.execute(
                    "UPDATE telegram_dm_topic_mode SET enabled = 0, updated_at = %s "
                    "WHERE profile_name = %s AND chat_id = %s",
                    (time.time(), profile_name, str(chat_id)))
                if clear_bindings:
                    conn.execute(
                        "DELETE FROM telegram_dm_topic_bindings WHERE profile_name = %s AND chat_id = %s",
                        (profile_name, str(chat_id)))

    def delete_topic_binding(self, *, chat_id: str, thread_id: str, profile_name: str = "default") -> int:
        if not self._topic_tables_ready():
            return 0
        store = self._store
        chat_id, thread_id = str(chat_id), str(thread_id)
        profile_name = normalize_topic_profile_name(profile_name)
        with translated():
            with store.tx() as conn:
                result = conn.execute(
                    """DELETE FROM telegram_dm_topic_bindings
                       WHERE profile_name = %s AND chat_id = %s AND thread_id = %s""",
                    (profile_name, chat_id, thread_id))
                deleted = result.rowcount or 0
                if deleted:
                    # Last binding gone: disable topic mode in the same transaction (no
                    # read-after-prune race).
                    remaining = conn.execute(
                        """SELECT 1 FROM telegram_dm_topic_bindings
                           WHERE profile_name = %s AND chat_id = %s LIMIT 1""",
                        (profile_name, chat_id)).fetchone()
                    if remaining is None:
                        conn.execute(
                            "UPDATE telegram_dm_topic_mode SET enabled = 0, updated_at = %s "
                            "WHERE profile_name = %s AND chat_id = %s",
                            (time.time(), profile_name, chat_id))
                return deleted

    def bind_topic(
        self, *, chat_id: str, thread_id: str, user_id: str, session_key: str, session_id: str,
        managed_mode: str = "auto", profile_name: str = "default", expected_session_id: str | None = None,
        move_session: bool = False,
    ) -> bool:
        store = self._store
        self.prepare_topic_storage()
        now = time.time()
        chat_id, thread_id, user_id = str(chat_id), str(thread_id), str(user_id)
        session_key, session_id = str(session_key), str(session_id)
        profile_name = normalize_topic_profile_name(profile_name)
        with translated():
            with store.tx() as conn:
                if expected_session_id is not None:
                    current = conn.execute(
                        """SELECT session_id FROM telegram_dm_topic_bindings
                           WHERE profile_name = %s AND chat_id = %s AND thread_id = %s""",
                        (profile_name, chat_id, thread_id)).fetchone()
                    if current is None or str(current["session_id"]) != str(expected_session_id):
                        return False
                existing_session = conn.execute(
                    """SELECT profile_name, chat_id, thread_id
                       FROM telegram_dm_topic_bindings
                       WHERE session_id = %s""", (session_id,)).fetchone()
                if existing_session is not None:
                    linked = (str(existing_session["profile_name"]),
                              str(existing_session["chat_id"]), str(existing_session["thread_id"]))
                    if linked != (profile_name, chat_id, thread_id):
                        if not move_session or linked[0] != profile_name:  # never across profiles
                            raise TopicAlreadyLinked(session_id)
                        conn.execute("DELETE FROM telegram_dm_topic_bindings WHERE session_id = %s", (session_id,))
                # The binding's session_id references sessions(id): a missing row refuses the
                # bind (the reference's FK violation), mapping to the named refusal.
                if conn.execute("SELECT 1 FROM sessions WHERE id = %s", (session_id,)).fetchone() is None:
                    raise SessionRowMissing(session_id)
                conn.execute(
                    """INSERT INTO telegram_dm_topic_bindings (
                           profile_name, chat_id, thread_id, user_id, session_key, session_id,
                           managed_mode, linked_at, updated_at
                       ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                       ON CONFLICT (profile_name, chat_id, thread_id) DO UPDATE SET
                           user_id = EXCLUDED.user_id,
                           session_key = EXCLUDED.session_key,
                           session_id = EXCLUDED.session_id,
                           managed_mode = EXCLUDED.managed_mode,
                           updated_at = EXCLUDED.updated_at""",
                    (profile_name, chat_id, thread_id, user_id, session_key, session_id,
                     managed_mode, now, now))
                return True

    # ── reads ───────────────────────────────────────────────────────────────

    def route_retired(self, session_key: str, entry_json: str) -> bool:
        store = self._store
        session_id = self._entry_session_id_strict(entry_json)
        with translated():
            row = store.one(
                f"SELECT {_ROUTE_RETIRED_SQL} AS retired",
                *_route_retired_params(session_id, route_namespace(session_key), time.time()))
        return bool(row["retired"]) if row is not None else False

    def load_routing_entries(self, *, routing_scope: str = "") -> dict[str, str]:
        store = self._store
        with translated():
            rows = store.all(
                "SELECT session_key, entry_json FROM gateway_routing WHERE scope = %s",
                routing_scope)
        return {row["session_key"]: row["entry_json"] for row in rows}

    def routing_entry_for_session(self, session_id: str) -> Mapping[str, Any] | None:
        store = self._store
        with translated():
            rows = store.all("SELECT entry_json FROM gateway_routing")
        for row in rows:
            try:
                entry = json.loads(row["entry_json"] or "{}")
            except (ValueError, TypeError, RecursionError):  # malformed / non-text / over-nested entry
                continue
            if isinstance(entry, dict) and entry.get("session_id") == session_id:
                return entry
        return None

    @staticmethod
    def _recovered(row: Mapping[str, Any]) -> RecoveredPeerSession:
        data = _row_to_record(dict(row))
        return RecoveredPeerSession(session=session_record(data),
                                    has_messages=bool(data.get("_has_messages")))

    def latest_session_for_peer(
        self, *, source: str, user_id: str | None = None, session_key: str | None = None,
        chat_id: str | None = None, chat_type: str | None = None, thread_id: str | None = None,
    ) -> RecoveredPeerSession | None:
        if not session_key:
            return None
        store = self._store
        with translated():
            row = store.one(_PEER_BY_KEY_SQL, session_key, source)
            if row is not None:
                return self._recovered(row)
            if chat_id is None or chat_type is None:
                return None
            # Profile fence: a Telegram DM's peer tuple is identical for every bot (chat_id ==
            # user_id, no thread), so a sibling profile's legacy row in this store would otherwise
            # be adopted. A row is ours when its profile_name is the owner's or NULL; stores
            # outside the profile tree derive no owner and keep the unfenced behavior.
            owner = store.owner_profile_name
            row = store.one(
                _PEER_BY_TUPLE_SQL,
                source, user_id, chat_id, chat_type, thread_id, owner, owner, owner, owner, owner, owner)
        return self._recovered(row) if row is not None else None

    def session_for_origin(
        self, *, platform: str, chat_id: str, thread_id: str | None = None, user_id: str | None = None,
    ) -> str | None:
        if not platform or chat_id in (None, ""):
            return None
        store = self._store
        query = (
            "SELECT id, user_id, started_at FROM sessions"
            " WHERE translate(source, %s, %s) = translate(%s, %s, %s)"
            "  AND session_key IS NOT NULL"
            "  AND chat_id = %s"
            "  AND ended_at IS NULL")
        params: list = [_ASCII_UPPER, _ASCII_LOWER, platform, _ASCII_UPPER, _ASCII_LOWER, str(chat_id)]
        if thread_id is not None:
            query += " AND COALESCE(thread_id, '') = %s"
            params.append(str(thread_id))
        with translated():
            rows = store.all(query + " ORDER BY started_at DESC", *params)
        if not rows:
            return None
        if user_id:
            exact = [r for r in rows if str(r.get("user_id") or "") == str(user_id)]
            if exact:
                return str(exact[0]["id"])
            if len(rows) > 1:
                return None
        elif len({u for u in (str(r.get("user_id") or "").strip() for r in rows) if u}) > 1:
            return None
        return str(rows[0]["id"])

    def orphaned_sessions(self, *, max_gap_s: float | None = None) -> tuple[OrphanReport, ...]:
        store = self._store
        gap = ORPHAN_ADOPTION_MAX_GAP_S if max_gap_s is None else float(max_gap_s)
        owner = store.owner_profile_name
        records: list[dict[str, Any]] = []
        with translated():
            for orphan in store.all(_ORPHANS_SQL):
                donor = None
                reason = ""
                orphan_owner = orphan["profile_name"] or owner
                if orphan["parent_session_id"]:
                    evidence = "lineage"
                    donor = store.one(
                        _ORPHAN_LINEAGE_DONOR_SQL,
                        orphan["parent_session_id"], orphan["source"], owner, owner, orphan_owner)
                    if donor is None:
                        reason = "parent session carries no gateway identity of this source"
                else:
                    evidence = "contiguity"
                    started = orphan["started_at"] or 0
                    candidates = store.all(
                        _ORPHAN_CONTIGUITY_DONORS_SQL,
                        orphan["id"], orphan["source"], orphan["user_id"], orphan["user_id"],
                        started - gap, started + gap, orphan["last_active"],
                        owner, owner, orphan_owner)
                    if not candidates:
                        reason = f"no keyed predecessor fell quiet within {gap:.0f}s of this session's start"
                    elif len(candidates) > 1:
                        reason = "ambiguous: more than one keyed predecessor matches this window"
                    else:
                        donor = candidates[0]
                records.append({
                    "orphan_id": orphan["id"], "source": orphan["source"],
                    "message_count": orphan["message_count"], "started_at": orphan["started_at"],
                    "last_active": orphan["last_active"],
                    "donor_id": donor["id"] if donor else None,
                    "session_key": donor["session_key"] if donor else None,
                    "evidence": evidence if donor else "", "adoptable": donor is not None, "reason": reason})
        # Two unkeyed successors claiming one predecessor: at most one continues that chat.
        contested = {
            r["donor_id"] for r in records
            if r["adoptable"] and sum(1 for x in records if x["donor_id"] == r["donor_id"]) > 1}
        for record in records:
            if record["donor_id"] in contested:
                record["adoptable"] = False
                record["reason"] = "ambiguous: more than one unkeyed session claims this predecessor"
        return tuple(OrphanReport(**record) for record in records)

    def never_active_keyed_sessions(self, *, older_than_days: float) -> tuple[NeverActiveSession, ...]:
        if older_than_days < 0:
            raise ValueError(
                f"older_than_days must be >= 0, got {older_than_days!r}: a negative "
                "retention builds a future cutoff that matches every never-active keyed row.")
        cutoff = time.time() - (float(older_than_days) * 86400.0)
        store = self._store
        with translated():
            rows = store.all(
                """SELECT s.id, s.session_key, s.source, s.chat_id,
                          s.chat_type, s.user_id, s.started_at
                   FROM sessions s
                   WHERE s.session_key IS NOT NULL
                     AND s.ended_at IS NULL
                     AND s.title IS NULL
                     AND s.last_activity_at IS NULL
                     AND COALESCE(s.message_count, 0) = 0
                     AND COALESCE(s.tool_call_count, 0) = 0
                     AND COALESCE(s.api_call_count, 0) = 0
                     AND COALESCE(s.input_tokens, 0) = 0
                     AND COALESCE(s.output_tokens, 0) = 0
                     AND COALESCE(s.pinned, 0) = 0
                     AND COALESCE(s.archived, 0) = 0
                     AND s.started_at IS NOT NULL
                     AND s.started_at < %s
                     AND NOT EXISTS (
                             SELECT 1 FROM messages m WHERE m.session_id = s.id
                         )
                   ORDER BY s.started_at""",
                   cutoff)
        return tuple(
            NeverActiveSession(
                session_id=row["id"], session_key=row["session_key"], source=row["source"],
                chat_id=row["chat_id"], chat_type=row["chat_type"], user_id=row["user_id"],
                started_at=row["started_at"])
            for row in rows)

    def _topic_tables_ready(self) -> bool:
        """Whether the topic tables exist in this store's schema (a store without them reads
        empty and its cleanups are no-ops, as on the reference)."""
        row = self._store.one(
            """SELECT COUNT(*) AS n FROM information_schema.tables
               WHERE table_schema = current_schema()
                 AND table_name IN ('telegram_dm_topic_mode', 'telegram_dm_topic_bindings')""")
        return row is not None and int(row["n"]) == 2

    def is_topic_mode_enabled(self, *, chat_id: str, user_id: str, profile_name: str = "default") -> bool:
        if not self._topic_tables_ready():
            return False
        store = self._store
        profile_name = normalize_topic_profile_name(profile_name)
        with translated():
            row = store.one(
                """SELECT enabled FROM telegram_dm_topic_mode
                   WHERE profile_name = %s AND chat_id = %s AND user_id = %s""",
                profile_name, str(chat_id), str(user_id))
        return bool(row["enabled"]) if row is not None else False

    def get_topic_binding(self, *, chat_id: str, thread_id: str, profile_name: str = "default") -> TopicBinding | None:
        if not self._topic_tables_ready():
            return None
        store = self._store
        profile_name = normalize_topic_profile_name(profile_name)
        with translated():
            row = store.one(
                """SELECT * FROM telegram_dm_topic_bindings
                   WHERE profile_name = %s AND chat_id = %s AND thread_id = %s""",
                profile_name, str(chat_id), str(thread_id))
        return TopicBinding(**dict(row)) if row is not None else None

    def list_topic_bindings_for_chat(
        self, *, chat_id: str, profile_name: str = "default"
    ) -> tuple[TopicBinding, ...]:
        if not self._topic_tables_ready():
            return ()
        store = self._store
        profile_name = normalize_topic_profile_name(profile_name)
        with translated():
            rows = store.all(
                """SELECT * FROM telegram_dm_topic_bindings
                   WHERE profile_name = %s AND chat_id = %s
                   ORDER BY updated_at DESC""",
                profile_name, str(chat_id))
        return tuple(TopicBinding(**dict(row)) for row in rows)

    def get_topic_binding_for_session(self, *, session_id: str) -> TopicBinding | None:
        if not self._topic_tables_ready():
            return None
        store = self._store
        with translated():
            row = store.one(
                """SELECT * FROM telegram_dm_topic_bindings
                   WHERE session_id = %s""", str(session_id))
        return TopicBinding(**dict(row)) if row is not None else None

    def is_session_linked_to_topic(self, *, session_id: str) -> bool:
        if not self._topic_tables_ready():
            return False
        store = self._store
        with translated():
            row = store.one(
                """SELECT 1 FROM telegram_dm_topic_bindings
                   WHERE session_id = %s
                   LIMIT 1""", str(session_id))
        return row is not None


add_stubs(PgRoutingRepository, RoutingRepository, implemented=(
    "upsert_routing_entry", "replace_routing_entries", "fence_retired_routes", "record_session_peer",
    "adopt_orphan", "prune_never_active", "sweep_routing_entries_for_sessions", "prepare_topic_storage",
    "enable_topic_mode", "disable_topic_mode", "delete_topic_binding", "bind_topic",
    "route_retired", "load_routing_entries", "routing_entry_for_session", "latest_session_for_peer",
    "session_for_origin", "orphaned_sessions", "never_active_keyed_sessions", "is_topic_mode_enabled",
    "get_topic_binding", "list_topic_bindings_for_chat", "get_topic_binding_for_session",
    "is_session_linked_to_topic"))


def _register() -> None:
    register_repository("routing", lambda store: PgRoutingRepository(store))


_register()

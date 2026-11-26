"""``ProfileIdentityRepository`` on PostgreSQL: the profile rekey and purge of the bound store schema, each one
write transaction; a facet whose lazily created table or column the store lacks is skipped and counted
``None``. The PG schema never creates the lazily-migrated telegram topic tables or the delivery ledger, so
those facets count ``None`` here; the rekey also moves the API binding and local policy receipts in
``state_meta``."""

from __future__ import annotations

import json
import time
from typing import Optional

from persistence.housekeeping.profile_identity import (
    ProfileIdentityCollision, ProfileIdentityRepository, ProfilePurgeCounts, ProfileRekeyCounts,
)

from .._backend import PostgresStore, register_repository, translated
from .._stubs import add_stubs
from persistence.sqlite.conversation._store.local import API_BINDING_PREFIX, POLICY_PREFIX
from persistence.sqlite.gateway._local_lineage import _receipt

__all__ = ["PgProfileIdentityRepository"]

# The lazily created tables the settlement gates on (the SQLite ``_optional_table_columns`` set).
_OPTIONAL_TABLE_NAMES = (
    "telegram_dm_topic_mode", "telegram_dm_topic_bindings", "delivery_obligations")


def _optional_table_columns(conn) -> dict[str, set[str]]:
    """Live column sets of the lazily-created tables that exist (``{}`` when none do) — read from
    ``information_schema`` on the store's pinned search path (the SQLite ``PRAGMA table_info`` twin)."""
    rows = conn.execute(
        "SELECT table_name, column_name FROM information_schema.columns"
        " WHERE table_schema = current_schema() AND table_name IN ('telegram_dm_topic_mode',"
        " 'telegram_dm_topic_bindings', 'delivery_obligations')").fetchall()
    out: dict[str, set[str]] = {}
    for row in rows:
        out.setdefault(row["table_name"], set()).add(row["column_name"])
    return out


def _moved_key(key: object, old: str, new: str) -> Optional[str]:
    """*key* moved from the ``agent:<old>:`` session-key namespace to ``agent:<new>:``, or None when
    it is not a string in that namespace."""
    old_ns = f"agent:{old}:"
    if isinstance(key, str) and key.startswith(old_ns):
        return f"agent:{new}:" + key[len(old_ns):]
    return None


def _rekey_entry(entry: dict, old: str, new: str) -> bool:
    """Move one serialized ``SessionEntry`` (a routing row, a session receipt) from profile *old*
    to *new*: its ``agent:<old>:`` session-key namespace and its ``origin.profile``. True when
    either changed."""
    changed = False
    if (session_key := _moved_key(entry.get("session_key"), old, new)) is not None:
        entry["session_key"] = session_key
        changed = True
    origin = entry.get("origin")
    if isinstance(origin, dict) and origin.get("profile") == old:
        origin["profile"] = new
        changed = True
    return changed


def _rekey_session_receipts(conn, prefix: str, old: str, new: str) -> int:
    """Rewrite the session receipts under *prefix* a profile rename would strand; returns how many
    changed (the SQLite ``_rekey_session_receipts``; the GLOB scan becomes a LIKE prefix)."""
    rewritten = 0
    rows = conn.execute("SELECT key, value FROM state_meta WHERE key LIKE %s", (prefix + "%",)).fetchall()
    for row in rows:
        key, raw = row["key"], row["value"]
        if (payload := _receipt(raw)) is None:
            continue
        changed = False
        if (route := _moved_key(payload.get("route"), old, new)) is not None:
            payload["route"] = route
            changed = True
        entry = payload.get("entry")
        if isinstance(entry, dict) and _rekey_entry(entry, old, new):
            changed = True
        if changed:
            conn.execute("UPDATE state_meta SET value = %s WHERE key = %s",
                         (json.dumps(payload, ensure_ascii=False), key))
            rewritten += 1
    return rewritten


class PgProfileIdentityRepository:
    """Profile identity rewrite and purge inside the bound profile's store schema."""

    def __init__(self, store: PostgresStore) -> None:
        self._store = store

    def rekey_profile(self, old_name: str, new_name: str) -> ProfileRekeyCounts:
        old, new = (old_name or "").strip(), (new_name or "").strip()
        counts: dict[str, int] = {}
        if not old or not new or old == new:
            return ProfileRekeyCounts()
        old_ns, new_ns = f"agent:{old}:", f"agent:{new}:"
        ns_len = len(old_ns)

        def _do(conn):
            existing = {row["table_name"] for row in conn.execute(
                "SELECT table_name FROM information_schema.tables"
                " WHERE table_schema = current_schema()").fetchall()}
            topic_columns = _optional_table_columns(conn)
            # A routing entry already stored under the target key in the same scope: the key's
            # ``agent:<old>:`` prefix re-homed onto an existing ``agent:<new>:`` row.
            collision = conn.execute(
                "SELECT old.scope AS scope, %s || substring(old.session_key from %s) AS target_key "
                "FROM gateway_routing AS old JOIN gateway_routing AS target "
                "ON target.scope = old.scope "
                "AND target.session_key = %s || substring(old.session_key from %s) "
                "WHERE left(old.session_key, %s) = %s LIMIT 1",
                (new_ns, ns_len + 1, new_ns, ns_len + 1, ns_len, old_ns),
            ).fetchone()
            if collision is not None:
                raise ProfileIdentityCollision(
                    new, f"profile routing collision in scope {collision['scope']!r}: "
                         f"{collision['target_key']!r}")
            for table, columns in (
                ("telegram_dm_topic_mode", ("chat_id",)),
                ("telegram_dm_topic_bindings", ("chat_id", "thread_id")),
            ):
                if "profile_name" not in topic_columns.get(table, set()):
                    continue
                equality = " AND ".join(
                    f"target.{column} = old.{column}" for column in columns)
                collision = conn.execute(
                    f"SELECT 1 FROM {table} AS old JOIN {table} AS target "
                    f"ON target.profile_name = %s AND {equality} "
                    "WHERE old.profile_name = %s LIMIT 1", (new, old)).fetchone()
                if collision is not None:
                    raise ProfileIdentityCollision(new, f"profile identity collision in {table}")

            counts["sessions_profile_name"] = conn.execute(
                "UPDATE sessions SET profile_name = %s WHERE profile_name = %s", (new, old)).rowcount
            counts["gateway_heartbeats_profile"] = conn.execute(
                "UPDATE gateway_heartbeats SET profile = %s WHERE profile = %s", (new, old)).rowcount
            counts["sessions_session_key"] = conn.execute(
                "UPDATE sessions SET session_key = %s || substring(session_key from %s) "
                "WHERE left(session_key, %s) = %s",
                (new_ns, ns_len + 1, ns_len, old_ns)).rowcount

            origin_count = 0
            for row in conn.execute(
                    "SELECT id, origin_json FROM sessions WHERE origin_json IS NOT NULL").fetchall():
                try:
                    payload = json.loads(row["origin_json"])
                except (ValueError, TypeError):
                    continue
                if isinstance(payload, dict) and payload.get("profile") == old:
                    payload["profile"] = new
                    conn.execute("UPDATE sessions SET origin_json = %s WHERE id = %s",
                                 (json.dumps(payload, ensure_ascii=False), row["id"]))
                    origin_count += 1
            counts["sessions_origin_json"] = origin_count

            if "delivery_obligations" in existing:
                if "adapter_profile" in topic_columns.get("delivery_obligations", set()):
                    counts["delivery_obligations_adapter_profile"] = conn.execute(
                        "UPDATE delivery_obligations SET adapter_profile = %s WHERE adapter_profile = %s",
                        (new, old)).rowcount
                counts["delivery_obligations_session_key"] = conn.execute(
                    "UPDATE delivery_obligations SET session_key = %s || substring(session_key from %s) "
                    "WHERE left(session_key, %s) = %s",
                    (new_ns, ns_len + 1, ns_len, old_ns)).rowcount
            for table in ("telegram_dm_topic_mode", "telegram_dm_topic_bindings"):
                if "profile_name" in topic_columns.get(table, set()):
                    counts[f"{table}_profile_name"] = conn.execute(
                        f"UPDATE {table} SET profile_name = %s WHERE profile_name = %s",
                        (new, old)).rowcount
            if "session_key" in topic_columns.get("telegram_dm_topic_bindings", set()):
                counts["telegram_dm_topic_bindings_session_key"] = conn.execute(
                    "UPDATE telegram_dm_topic_bindings "
                    "SET session_key = %s || substring(session_key from %s) "
                    "WHERE left(session_key, %s) = %s",
                    (new_ns, ns_len + 1, ns_len, old_ns)).rowcount

            routing = conn.execute(
                "SELECT scope, session_key, entry_json FROM gateway_routing "
                "WHERE left(session_key, %s) = %s", (ns_len, old_ns)).fetchall()
            for row in routing:
                new_session_key = new_ns + row["session_key"][ns_len:]
                new_json = row["entry_json"]
                if row["entry_json"]:
                    try:
                        payload = json.loads(row["entry_json"])
                    except (ValueError, TypeError):
                        payload = None
                    if isinstance(payload, dict):
                        _rekey_entry(payload, old, new)
                        new_json = json.dumps(payload, ensure_ascii=False)
                conn.execute(
                    "UPDATE gateway_routing SET session_key = %s, entry_json = %s "
                    "WHERE scope = %s AND session_key = %s",
                    (new_session_key, new_json, row["scope"], row["session_key"]))
            counts["gateway_routing"] = len(routing)
            counts["api_binding_receipts"] = _rekey_session_receipts(conn, API_BINDING_PREFIX, old, new)
            counts["local_policy_receipts"] = _rekey_session_receipts(conn, POLICY_PREFIX, old, new)

        with translated():
            with self._store.tx() as conn:
                _do(conn)
        return ProfileRekeyCounts(**counts)

    def purge_profile(self, profile: str) -> ProfilePurgeCounts:
        name = (profile or "").strip()
        counts: dict[str, int] = {}
        if not name:
            return ProfilePurgeCounts()
        ns, ns_len = f"agent:{name}:", len(f"agent:{name}:")

        def _do(conn):
            topic_columns = _optional_table_columns(conn)
            counts["gateway_routing"] = conn.execute(
                "DELETE FROM gateway_routing WHERE left(session_key, %s) = %s",
                (ns_len, ns)).rowcount
            counts["gateway_heartbeats"] = conn.execute(
                "DELETE FROM gateway_heartbeats WHERE profile = %s", (name,)).rowcount
            if "delivery_obligations" in topic_columns:
                # Terminalize, never hard-delete: a pending obligation is delivery state someone may
                # still care about, and the ledger's own retention prunes abandoned rows. Only
                # non-terminal rows are touched; delivered history is left exactly as it was.
                # A ledger created before ``adapter_profile`` existed matches on namespace alone.
                by_profile = ("adapter_profile = %s OR "
                              if "adapter_profile" in topic_columns["delivery_obligations"]
                              else "")
                params = (time.time(), name, ns_len, ns) if by_profile else (time.time(), ns_len, ns)
                counts["delivery_obligations"] = conn.execute(
                    "UPDATE delivery_obligations SET state = 'abandoned', updated_at = %s "
                    f"WHERE ({by_profile}left(session_key, %s) = %s) "
                    "AND state NOT IN ('delivered', 'abandoned')", params).rowcount
            if "profile_name" in topic_columns.get("telegram_dm_topic_mode", set()):
                counts["telegram_dm_topic_mode"] = conn.execute(
                    "DELETE FROM telegram_dm_topic_mode WHERE profile_name = %s", (name,)).rowcount
            binding_columns = topic_columns.get("telegram_dm_topic_bindings", set())
            if "session_key" in binding_columns:
                # A rename rewrites a binding's session_key namespace as well as its profile_name, so
                # matching on one alone leaves the other behind. Legacy v1/v2 bindings have no
                # profile_name but keep the ``agent:<name>:`` namespace in session_key, so the namespace
                # match alone is the exact cleanup there.
                by_profile = "profile_name = %s OR " if "profile_name" in binding_columns else ""
                params = (name, ns_len, ns) if by_profile else (ns_len, ns)
                counts["telegram_dm_topic_bindings"] = conn.execute(
                    "DELETE FROM telegram_dm_topic_bindings "
                    f"WHERE {by_profile}left(session_key, %s) = %s", params).rowcount

        with translated():
            with self._store.tx() as conn:
                _do(conn)
        return ProfilePurgeCounts(**counts)


add_stubs(PgProfileIdentityRepository, ProfileIdentityRepository,
          implemented=("rekey_profile", "purge_profile"))


def _register() -> None:
    register_repository("profile_identity", lambda store: PgProfileIdentityRepository(store))


_register()

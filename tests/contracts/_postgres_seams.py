"""Test seams for the PostgreSQL backend: the fixture hooks the contract suites call on the in-memory
reference, given their PostgreSQL meaning.

``ContractPostgresBackend`` serves any profile the suites name, each served by its own schema in the
database ``HERMES_DATABASE_URL`` names (created on first write, dropped when the backend closes), and
attaches to every store handle it opens the seeding seams of the reference repositories:
``search.seed_session``, ``search.seed_message``, ``search.seed_deferred_rebuild``, ``usage.seed_session``,
``usage.set_session_archived``, ``sessions.seed_ledger_admission``, ``runtime.seed_local_receipt`` and
``runtime.seed_retirement``. Each seeds the rows main stores, through the contract verbs where one
exists and otherwise through direct SQL on the store's schema. Every handle also carries the reference
handle's ``owner_profile``, the profile the backend stamps as the store's owner. ``session_db(scope)``
is one profile's store handle (the suites' SQL reads; their ``?``-style SQL translated to the PG
dialect), ``scope_store(scope)["sessions"][id]`` writes raw cells of one stored session row,
``begin_epoch(scope, instance_id)`` begins an authority epoch with main's runtime epoch write,
``replace_store(scope)`` puts a replacement store where the profile's schema lives, and
``close_store_under(scope)`` force-closes the store under its handles as a profile delete's sweep does.
These are test plumbing; nothing here is product code.
"""

from __future__ import annotations

import json
import time
import uuid
from datetime import datetime
from typing import Any, Mapping, Optional, Sequence

import psycopg.rows
from persistence import PersistenceError, ProfileScope, RowVisibility, SessionId
from persistence.core.canonical import canonical_json
from persistence.conversation.messages import NewMessage
from persistence.conversation.sessions import NewSession
from persistence.postgres import PostgresBackend, PostgresStore
from persistence.postgres._backend import _dsn, schema_for_home
from tests.contracts._scopes import CONTRACT_ROOT


def _pg_write(store: PostgresStore, sql: str, params: Sequence[Any]) -> int:
    """Run one statement in a write transaction on the store's schema; returns the changed row count."""
    with store.tx() as conn:
        result = conn.execute(sql, tuple(params))
        return result.rowcount


# ── search: sessions and rows the filters read ──────────────────────────────


def _search_seed_session(store: PostgresStore, session_id: str, *, source: Optional[str] = None, model: Optional[str] = None,
                         started_at: Optional[float] = None, delegate_child: bool = False) -> SessionId:
    """A stored session row with the start time and source the filters read; a delegate child carries
    ``main``'s ``_delegate_from`` creation marker."""
    store.sessions.create(NewSession(SessionId(session_id), source, model=model))
    if started_at is not None:
        _pg_write(store, "UPDATE sessions SET started_at = %s WHERE id = %s", (started_at, session_id))
    if delegate_child:
        _pg_write(store, "UPDATE sessions SET model_config = %s WHERE id = %s",
                  (json.dumps({"_delegate_from": "seed-parent"}), session_id))
    return SessionId(session_id)


_VISIBILITY_FLAGS = {RowVisibility.REWOUND: (0, 0), RowVisibility.COMPACTION_ARCHIVED: (0, 1)}


def _search_seed_message(store: PostgresStore, session_id: str, role: str, content: Any, *, timestamp: float,
                         tool_name: Optional[str] = None, tool_calls: Sequence[Mapping[str, Any]] = (),
                         display_kind: Optional[str] = None, visibility: RowVisibility = RowVisibility.ACTIVE) -> Any:
    """One transcript row appended through the messages verb, then given the stored ``active``/``compacted``
    flags of ``visibility``."""
    message = NewMessage(role, content, float(timestamp), tool_name=tool_name, tool_calls=tuple(tool_calls),
                         display_kind=display_kind)
    row_id = store.messages.append(SessionId(session_id), message)
    if visibility is not RowVisibility.ACTIVE:
        _pg_write(store, "UPDATE messages SET active = %s, compacted = %s WHERE id = %s",
                  (*_VISIBILITY_FLAGS[visibility], int(row_id)))
    return row_id


def _search_seed_deferred_rebuild(store: PostgresStore, total: int, indexed: int) -> None:
    """The deferred rebuild markers ``main`` keeps in ``state_meta``: the high-water id (``total``) and the
    backfill progress (``indexed``)."""
    store.state_meta.set("fts_rebuild_high_water", str(int(total)))
    store.state_meta.set("fts_rebuild_progress", str(int(indexed)))


# ── usage: the session columns the totals filter reads ──────────────────────


def _usage_seed_session(store: PostgresStore, session_id: str, *, message_count: int = 0,
                       parent_session_id: Optional[str] = None, archived: bool = False) -> SessionId:
    """A ``cli`` session row with the stored message count, parent link and archived flag the totals filter
    reads."""
    parent = SessionId(parent_session_id) if parent_session_id else None
    store.sessions.create(NewSession(SessionId(session_id), "cli", parent_session_id=parent))
    _pg_write(store, "UPDATE sessions SET message_count = %s, archived = %s WHERE id = %s",
              (message_count, int(archived), session_id))
    return SessionId(session_id)


def _usage_set_session_archived(store: PostgresStore, session_id: str, archived: bool) -> None:
    if not _pg_write(store, "UPDATE sessions SET archived = %s WHERE id = %s", (int(archived), session_id)):
        raise KeyError(session_id)


# ── sessions: the admissions ledger the delete fence reads ──────────────────


def _seed_ledger_admission(store: PostgresStore, session_id: SessionId, *, status: str = "queued") -> None:
    """One ``session_admissions`` row targeting ``session_id`` in ``status``; a started row carries the owner
    epoch and generation the schema requires."""
    def write(conn: Any) -> None:
        row = conn.execute("SELECT COUNT(*) AS count FROM session_admissions").fetchone()
        admission_id = f"seed-adm-{int(row['count'])}"
        conn.execute(
            """INSERT INTO session_admissions (admission_id, request_id, principal_id, target_session_id,
                                              lineage_json, payload_json, payload_digest, intent, status,
                                              owner_epoch, generation)
               VALUES (%s, %s, 'contract-seed', %s, %s, '{}', '', 'queue', %s, %s, %s)""",
            (admission_id, admission_id, str(session_id), json.dumps([str(session_id)]), status,
             0 if status == "started" else None, 0 if status == "started" else None),
        )
    with store.tx() as conn:
        write(conn)


# ── runtime: the local policy receipt and the retirement marker ────────────


def _seed_local_receipt(store: PostgresStore, session_id: str, receipt: Mapping[str, Any]) -> None:
    """``receipt`` stored as ``main`` keeps a local policy receipt: canonical JSON under ``POLICY_PREFIX +
    session_id``."""
    from persistence.sqlite.conversation._store.local import POLICY_PREFIX

    key = POLICY_PREFIX + session_id
    _pg_write(store,
              """INSERT INTO state_meta (key, value) VALUES (%s, %s)
                 ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value""",
              (key, canonical_json(dict(receipt))))


def _seed_retirement(store: PostgresStore, session_id: str, *, expires: bool = False) -> None:
    """The retirement marker ``main``'s ``retire_routes`` writes for ``session_id``, alone; ``expires`` makes
    it a sweep's marker (the SQLite ``_mark_retired`` CASEs on the PG dialect; ``jsonb_valid`` guards the
    stored JSON, ``jsonb_set(..., 'null', true)`` is ``json_remove``)."""
    from persistence.sqlite.conversation._store.retirement import RETIRED_PREFIX

    key = RETIRED_PREFIX + session_id
    if expires:
        payload = json.dumps({"retired_at": time.time(), "expires": True})
        _pg_write(
            store,
            """INSERT INTO state_meta (key, value) VALUES (%s, %s)
               ON CONFLICT (key) DO UPDATE SET value = excluded.value
               WHERE jsonb_valid(state_meta.value) AND (state_meta.value::jsonb ? 'expires')""",
            (key, payload),
        )
    else:
        payload = json.dumps({"retired_at": time.time(), "own": True})
        _pg_write(
            store,
            """INSERT INTO state_meta (key, value) VALUES (%s, %s)
               ON CONFLICT (key) DO UPDATE SET value = CASE
                   WHEN NOT jsonb_valid(state_meta.value) THEN jsonb_set(excluded.value::jsonb, '{own}', 'null', true)::text
                   WHEN (state_meta.value::jsonb ? 'own') OR (state_meta.value::jsonb ? 'expires') THEN excluded.value
                   ELSE jsonb_set(excluded.value::jsonb, '{own}', 'null', true)::text
               END""",
            (key, payload),
        )


# ── raw session cells (the reference's row-dict seam) ───────────────────────


def _cell(value: Any) -> Any:
    """A datetime is stored as ``main`` stores instants, epoch seconds; anything else lands as given."""
    return value.timestamp() if isinstance(value, datetime) else value


def _session_columns(store: PostgresStore) -> dict[str, str]:
    rows = store.all("SELECT column_name, data_type FROM information_schema.columns WHERE table_name = 'sessions'")
    return {row["column_name"]: row["data_type"] for row in rows}


class _SessionCells:
    """One stored session row: assigning a cell writes that column as given, bypassing every typed write path."""

    def __init__(self, store: PostgresStore, session_id: str) -> None:
        self._store = store
        self._session_id = session_id

    def __setitem__(self, column: str, value: Any) -> None:
        self.update({column: value})

    def update(self, cells: Optional[Mapping[str, Any]] = None, **more: Any) -> None:
        cells = {**(cells or {}), **more}
        columns = _session_columns(self._store)
        unknown = set(cells) - set(columns)
        if unknown:
            raise KeyError(f"sessions has no column {sorted(unknown)}")
        assignments = ", ".join(f"{column} = %s" for column in cells)
        params = []
        for column, value in cells.items():
            value = _cell(value)
            # A corrupted cell: the SQLite reference's REAL affinity stores a text value verbatim and the
            # stored read degrades it (untrusted -> the trust-window floor); a double precision column
            # holds numbers only, so a non-numeric text is planted as the window ceiling, which reads
            # back untrusted the same way (the ``stored_instant`` convention's damaged-page case).
            if columns[column] == "double precision" and isinstance(value, str):
                try:
                    value = float(value)
                except ValueError:
                    from persistence.sqlite.conversation._store.rules import EPOCH_MAX
                    value = EPOCH_MAX + 1.0
            params.append(value)
        if not _pg_write(self._store, f"UPDATE sessions SET {assignments} WHERE id = %s", (*params, self._session_id)):
            raise KeyError(self._session_id)


class _SessionRows:
    """The stored session rows of one profile, by id."""

    def __init__(self, store: PostgresStore) -> None:
        self._store = store

    def __getitem__(self, session_id: str) -> _SessionCells:
        if self._store.one("SELECT 1 AS one FROM sessions WHERE id = %s", session_id) is None:
            raise KeyError(session_id)
        return _SessionCells(self._store, session_id)


# ── the suites' SQL reads (their ``?``-style statements, translated) ────────


class _SessionDb:
    """One profile's store handle for the suites' SQL reads: their SQLite ``?`` statements translated to the
    PG dialect, rows as position- and name-addressable rows."""

    def __init__(self, store: PostgresStore) -> None:
        self._store = store

    def _translate(self, sql: str) -> str:
        return sql.replace("?", "%s")

    def _read(self, sql: str, params: Sequence[Any]) -> list[Any]:
        store = self._store
        with store.tx() as conn:
            conn.row_factory = psycopg.rows.tuple_row
            try:
                return conn.execute(self._translate(sql), tuple(params)).fetchall()
            finally:
                conn.row_factory = psycopg.rows.dict_row

    def _read_one(self, sql: str, params: Sequence[Any]) -> Optional[Any]:
        rows = self._read(sql, params)
        return rows[0] if rows else None

    def _read_all(self, sql: str, params: Sequence[Any]) -> list[Any]:
        return self._read(sql, params)


# ── the backend ──────────────────────────────────────────────────────────────


class ContractPostgresBackend(PostgresBackend):
    """The PostgreSQL backend as the contract suites drive it: every profile they name is served, and every
    handle carries the seeding seams."""

    def __init__(self, stores: Mapping[str, Any], *, root: Optional[Any] = None) -> None:
        # A fresh assembly gets its own schema namespace: the suites' factory contract is that every
        # call starts from empty stores (the reference assembly's per-assembly state), while handles
        # of one assembly still share their profile's schema.
        super().__init__(stores, schema_namespace=uuid.uuid4().hex[:8])
        self.root = root
        self._opened_scopes: list[ProfileScope] = []

    def open(self, scope: ProfileScope) -> PostgresStore:
        # PostgresBackend has no public hook to serve one more profile, so this fixture class, and nothing
        # else in the suite, registers the store location in its private map.
        if scope.hermes_home_key not in self._stores:
            from pathlib import Path
            from tests.contracts._scopes import CONTRACT_ROOT
            try:
                # Mirror the scope's place in the suites' fake root under the fixture root: the owner
                # derivation (profile tree or not) then answers from the location, as the reference
                # derives it from the store's own path.
                rel = Path(scope.profile_dir).relative_to(CONTRACT_ROOT)
            except ValueError:  # a scope home outside the suites' root keeps its own name
                rel = Path(scope.name)
            self._stores[scope.hermes_home_key] = self.root / rel / "state.db" \
                if self.root else scope.hermes_home_key
        store = super().open(scope)
        self._opened_scopes.append(scope)
        return _attach_seams(store)

    def close(self) -> None:
        """The base close drops only the schemas this backend created; a schema left by a killed run (or
        another process) pre-existed it and would otherwise outlive the fixture - its rows would seed the
        next backend's store. Drop every served scope's schema, best effort, after the base close has
        released the handles."""
        super().close()
        for scope in self._opened_scopes:
            schema = schema_for_home(scope, self.schema_override)
            if schema in self._created_schemas:
                continue  # the base close already dropped it
            try:
                with psycopg.connect(_dsn(), autocommit=True, connect_timeout=5) as conn:
                    conn.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
            except PersistenceError:
                pass  # a dead server leaves the schema for the next run; the base close did its part

    def owner_profile_name_of(self, scope: ProfileScope) -> Optional[str]:
        return self.open(scope).owner_profile_name

    def session_db(self, scope: ProfileScope) -> _SessionDb:
        """The store handle of ``scope``, for the suites' SQL reads."""
        return _SessionDb(self.open(scope))

    def scope_store(self, scope: ProfileScope) -> dict[str, Any]:
        """The reference's namespace shape over ``scope``'s store: ``["sessions"][id]`` writes raw cells of
        one stored row."""
        return {"sessions": _SessionRows(self.open(scope))}

    def begin_epoch(self, scope: ProfileScope, instance_id: str) -> int:
        """Begin an authority epoch on ``scope``'s store through its write door, as the reference's authority
        state does on its own."""
        return PostgresBackend.begin_epoch(self, scope, instance_id)


def _attach_seams(store: PostgresStore) -> PostgresStore:
    """The reference handle's ``owner_profile`` plus the seeding seams, on the handle's repositories."""
    from functools import partial

    store.owner_profile = store.owner_profile_name  # the profile the backend stamps as the store's owner
    for attribute, name, seam in (
        ("search", "seed_session", _search_seed_session),
        ("search", "seed_message", _search_seed_message),
        ("search", "seed_deferred_rebuild", _search_seed_deferred_rebuild),
        ("usage", "seed_session", _usage_seed_session),
        ("usage", "set_session_archived", _usage_set_session_archived),
        ("sessions", "seed_ledger_admission", _seed_ledger_admission),
        ("runtime", "seed_local_receipt", _seed_local_receipt),
        ("runtime", "seed_retirement", _seed_retirement),
    ):
        target = getattr(store, attribute)
        if hasattr(target, name):
            continue  # a ported verb already carries the seam
        setattr(target, name, partial(seam, store))
    return store

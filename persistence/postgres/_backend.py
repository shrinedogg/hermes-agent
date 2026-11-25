"""The assembled PostgreSQL backend.

One ``PostgresBackend`` holds the stores one assembly was built for, each profile's ``hermes_home_key``
served by one PG schema in the database named by ``HERMES_DATABASE_URL`` (D10: the backend registers only
when that variable is set; a missing variable fails closed with ``StoreUnavailable`` at first use). One
process-wide connection pool per DSN serves every backend in the process (plan 5.2 speed decision); a
store handle checks a connection out of the pool, pins it to the store's schema through ``search_path``
and hands it back on backend close. The reader's ``open`` touches nothing; the writer creates its schema
on first write (``CREATE SCHEMA`` + the declarative DDL of ``_ddl``), so a store that is never written is
"not initialized" to the health probe, as on SQLite.

Constructor arguments that are SQLite ideas, accepted once and said what they mean here (plan 5.2):
``read_patience_s`` has no file-lock equivalent on PostgreSQL (row locks wait server-side); it is
accepted and ignored. ``check_drift`` probes legacy SQLite schema drift; a PG schema is declarative
(``_ddl`` + the ``schema_version`` row), so it too is accepted and ignored.

Repository modules in this package call ``register_repository`` at import time with a factory that
receives the bound handle; the package ``__init__`` loads every public module, so a family registers
without touching the assembly. An unported verb of a repository raises ``NotImplementedError`` naming
itself (D8). PostgreSQL failures cross the repository boundary only as the contract's error family,
through ``translated``: deadlocks and lock waits are ``StoreContention``, connection and server
failures ``StoreUnavailable``, structural damage ``MalformedSchema`` (latched by ``PostgresStorageHealth``),
integrity violations ``PermanentStoreError``, resource exhaustion ``DiskFull``, everything else
``TransientStoreError``.
"""

from __future__ import annotations

import dataclasses
import atexit
import hashlib
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterator, Mapping, Optional

import psycopg
import psycopg_pool
import psycopg.rows
from psycopg_pool import ConnectionPool

from persistence import (
    AwaitableStore, DiskFull, MalformedSchema, PermanentStoreError, PersistenceCapabilities,
    PersistenceError, ProfileScope, StoreContention, StoreQuarantined, StoreReplaced, StoreUnavailable,
    TransientStoreError,
)
from persistence.core.admin import AdminUnsupported, ProfileRepairAdmin, StoreAdmin
from persistence.core.health import StorageProbe, StorageState

from . import _ddl

if TYPE_CHECKING:
    from psycopg import Connection

__all__ = [
    "CAPABILITY_ATTRIBUTES",
    "PostgresBackend",
    "PostgresStore",
    "PostgresStorageHealth",
    "register_repository",
    "schema_for_home",
    "translated",
]

# The repository attributes a store handle can carry, each once its repository registers (same set as
# the SQLite backend: a postgres handle presents the declared door).
CAPABILITY_ATTRIBUTES: tuple[str, ...] = (
    "sessions",
    "messages",
    "lineage",
    "search",
    "usage",
    "usage_analytics",
    "admissions",
    "runtime",
    "bindings",
    "coordination",
    "worker_executions",
    "maintenance",
    "state_meta",
    "profile_identity",
    "routing",
    "hosted_rooms",
    "hosted_room_peers",
    "hosted_room_replicas",
    "hosted_room_driver",
    "hosted_room_policy",
    "hosted_room_attachments",
    "health",
)

# Facts of a postgres store. Cross-process and reopen durability rest on the server: committed writes
# are visible to every connection and survive process exit (D9: physical_admin stays False; a
# PostgreSQL database is backed up by its operator, not the agent). remote stays False until a remote
# deployment case demands it.
_BASE_CAPABILITIES = PersistenceCapabilities(
    full_text_search=False,
    substring_search=False,
    transactional_claims=False,
    cross_process=True,
    remote=False,
    reopen_durability=True,
    authority_fences=False,
    durable_admissions=False,
    receipted_writes=False,
    physical_admin=False,
)

RepositoryFactory = Callable[["PostgresStore"], Any]

_REPOSITORY_FACTORIES: dict[str, RepositoryFactory] = {}
_CAPABILITY_OVERRIDES: dict[str, bool] = {}


def register_repository(
    attribute: str, factory: RepositoryFactory, capabilities: Optional[Mapping[str, bool]] = None,
) -> None:
    """Register the repository assembled as ``attribute`` on every new store handle (``factory`` receives the bound
    handle), with the capability flags its behaviour earns. Registering an attribute twice fails."""
    if attribute not in CAPABILITY_ATTRIBUTES:
        raise ValueError(f"unknown capability attribute {attribute!r}")
    if attribute in _REPOSITORY_FACTORIES:
        raise ValueError(f"postgres repository {attribute!r} already registered")
    _REPOSITORY_FACTORIES[attribute] = factory
    for name, value in (capabilities or {}).items():
        if not hasattr(_BASE_CAPABILITIES, name):
            raise ValueError(f"unknown capability field {name!r}")
        _CAPABILITY_OVERRIDES[name] = value


# ── error translation ─────────────────────────────────────────────────────────


def _pg_error(exc: Exception) -> PersistenceError:
    """The contract error for one PostgreSQL failure the store raised."""
    sqlstate = getattr(exc, "sqlstate", None)
    if isinstance(exc, psycopg.errors.DeadlockDetected) or sqlstate == "40P01":
        return StoreContention(str(exc))
    if isinstance(exc, psycopg.errors.LockNotAvailable) or sqlstate == "55P03":
        return StoreContention(str(exc))
    if isinstance(exc, psycopg.errors.QueryCanceled) or sqlstate == "57014":
        return StoreContention(str(exc))
    if isinstance(exc, psycopg.errors.DiskFull) or sqlstate in ("58037", "53100"):
        return DiskFull(str(exc))
    if isinstance(exc, psycopg.errors.ReadOnlySqlTransaction):
        return PermanentStoreError(f"read-only store refuses the write: {exc}")
    if sqlstate in ("23503", "23505", "23514", "23502", "23506"):
        return PermanentStoreError(str(exc))
    if sqlstate in ("42P01", "42P07", "42703", "42809", "42883", "42601", "42804"):
        return MalformedSchema(str(exc))
    if isinstance(exc, psycopg.OperationalError) or sqlstate in (
        "08000", "08001", "08003", "08006", "57P03", "57P01", "57000", "58000", "28000", "0A000",
    ):
        return StoreUnavailable(str(exc))
    if isinstance(exc, (psycopg.InterfaceError, OSError)):
        return StoreUnavailable(str(exc))
    if isinstance(exc, psycopg_pool.PoolTimeout):
        return StoreContention(str(exc))
    if isinstance(exc, psycopg_pool.PoolClosed):
        return StoreUnavailable(str(exc))
    return TransientStoreError(str(exc))


@contextmanager
def translated() -> Iterator[None]:
    """Raise the store's failures as the contract's error family, chaining the original."""
    try:
        yield
    except PersistenceError:
        raise
    except psycopg.Error as exc:
        raise _pg_error(exc) from exc
    except (psycopg_pool.PoolTimeout, psycopg_pool.PoolClosed) as exc:
        raise _pg_error(exc) from exc


# ── connection pool (one per DSN, per process; plan 5.2 speed decision) ──────

_POOLS: dict[str, ConnectionPool] = {}
_POOL_LOCK = threading.Lock()


def _dsn() -> str:
    url = os.environ.get("HERMES_DATABASE_URL")
    if not url:
        raise StoreUnavailable(
            "HERMES_DATABASE_URL is not set; the postgres persistence provider needs a database URL "
            "(plan D10: the backend registers only when it is set)"
        )
    return url


def _pool_for(dsn: str) -> ConnectionPool:
    with _POOL_LOCK:
        pool = _POOLS.get(dsn)
        if pool is None:
            pool = ConnectionPool(conninfo=dsn, min_size=1, max_size=16, timeout=30)
            try:
                pool.open(wait=True)
            except Exception as exc:
                raise StoreUnavailable(f"cannot reach the postgres database: {exc}") from exc
            _POOLS[dsn] = pool
        return pool


def _close_pools() -> None:
    """Interpreter-shutdown hook: close every pool so its threads finish before finalization (the driver's
    ``ConnectionPool.__del__`` cannot join them at shutdown and warns)."""
    for pool in list(_POOLS.values()):
        try:
            pool.close()
        except Exception:
            pass
    _POOLS.clear()


atexit.register(_close_pools)


_EXECUTOR_WORKERS = min(32, (os.cpu_count() or 1) + 4)
_executor: Optional[ThreadPoolExecutor] = None
_executor_lock = threading.Lock()


def _store_executor() -> ThreadPoolExecutor:
    """The awaitable form's executor, shared by every backend in the process (the SQLite backend's pool shape)."""
    global _executor
    with _executor_lock:
        if _executor is None:
            _executor = ThreadPoolExecutor(max_workers=_EXECUTOR_WORKERS, thread_name_prefix="hermes-postgres-store")
        return _executor


# ── schema naming ──────────────────────────────────────────────────────────────


def _valid_schema_name(name: str) -> None:
    if not re.fullmatch(r"[a-z_][a-z0-9_]{0,62}", name) or name.startswith("pg_"):
        raise ValueError(f"invalid postgres schema name {name!r}")


def schema_for_home(scope: ProfileScope, override: Optional[str] = None) -> str:
    """The store schema of ``scope`` in the shared database: ``hermes_<profile>_<12 hex of the profile
    path>`` (the old rule: a valid PG identifier, no ``pg_`` prefix, per-profile). ``override`` is the
    configured ``persistence.schema`` (plan 2.5): a single-profile host names its schema directly and
    every scope served by that backend shares it."""
    if override:
        _valid_schema_name(override)
        return override
    digest = hashlib.md5(str(Path(scope.profile_dir)).encode("utf-8")).hexdigest()[:12]
    name = re.sub(r"[^a-z0-9_]", "_", (scope.name or "default").lower()) or "default"
    if name.startswith("pg_"):
        name = f"_{name}"
    return f"hermes_{name}_{digest}"


# ── health (the corruption latch and the readiness probe) ─────────────────────

_LATCHES: dict[tuple[str, str], StorageState] = {}
_LATCH_LOCK = threading.Lock()


class PostgresStorageHealth:
    """``StorageHealth`` for one store schema: the process-local latch answers without store access; the
    probe is one read that never creates, migrates or repairs the schema (plan 5.2: a missing schema is
    "not initialized", a latched or structurally damaged one degrades to "corrupt")."""

    def __init__(self, schema: str, dsn: Optional[str] = None) -> None:
        _valid_schema_name(schema)
        self._schema = schema
        self._dsn = dsn

    def _key(self) -> tuple[str, str]:
        return (self._dsn or _dsn(), self._schema)

    def storage_state(self) -> StorageState:
        with _LATCH_LOCK:
            return _LATCHES.get(self._key(), StorageState.OK)

    def probe(self) -> StorageProbe:
        if self.storage_state() is StorageState.CORRUPT:
            return StorageProbe(status="degraded", detail="corrupt")
        try:
            dsn = self._dsn or _dsn()
        except PersistenceError as exc:
            return StorageProbe(status="degraded", detail=type(exc).__name__)
        try:
            with psycopg.connect(dsn, autocommit=True, connect_timeout=5) as conn:
                present = conn.execute(
                    "SELECT 1 FROM information_schema.schemata WHERE schema_name = %s", (self._schema,)
                ).fetchone()
                if not present:
                    return StorageProbe(status="ok", detail="not initialized")
                conn.execute(f'SET search_path TO "{self._schema}"')
                conn.execute("SELECT version FROM schema_version LIMIT 1")
        except PersistenceError as exc:
            return StorageProbe(status="degraded", detail=type(exc).__name__)
        except psycopg.Error as exc:
            sqlstate = getattr(exc, "sqlstate", None)
            if sqlstate in ("42P01", "42703", "42601"):  # structural damage latches, as on SQLite
                with _LATCH_LOCK:
                    _LATCHES[self._key()] = StorageState.CORRUPT
                return StorageProbe(status="degraded", detail="corrupt")
            return StorageProbe(status="degraded", detail=type(exc).__name__)
        except Exception as exc:  # noqa: BLE001 - a probe reports its failure's class, never raises
            return StorageProbe(status="degraded", detail=type(exc).__name__)
        return StorageProbe(status="ok", detail=None)


def register_health() -> None:
    """The handle's ``health`` attribute: this store schema's latch and probe, bound on the handle's open."""
    register_repository("health", lambda store: PostgresStorageHealth(store.schema))


# ── the backend and its store handles ─────────────────────────────────────────


class PostgresBackend:
    """Every store one assembly was built for; ``open(scope)`` binds a handle to one of them. See the module
    docstring for the store model (schema per profile, pool per process, create-on-write)."""

    def __init__(
        self, stores: Mapping[str, Path], *, schema: Optional[str] = None, read_only: bool = False,
        read_patience_s: Optional[float] = None, check_drift: bool = False,
    ) -> None:
        """``stores`` maps each served ``hermes_home_key`` to its state location; on PostgreSQL the path is the
        SQLite location the assembly still computes and the store itself is the schema named by
        :func:`schema_for_home` in the ``HERMES_DATABASE_URL`` database. ``read_patience_s`` and ``check_drift``
        are the SQLite constructor's ideas, accepted and ignored here (see the module docstring)."""
        self._stores = {key: Path(path) for key, path in stores.items()}
        self.schema_override = schema
        self.read_only = read_only
        self.read_patience_s = read_patience_s  # accepted, ignored (no file-lock patience on PostgreSQL)
        self.check_drift = check_drift  # accepted, ignored (declarative schema, no legacy drift)
        self.executor = _store_executor()
        self._lock = threading.Lock()
        self._closed = False
        self._open_handles: list[PostgresStore] = []
        self._generations: dict[str, int] = {}
        self._generation_lock = threading.Lock()
        # key -> the schema serving that profile's store; ``replace_store`` swaps the entry so new handles
        # open the replacement while open handles stay on the generation they hold.
        self._schema_by_key: dict[str, str] = {}
        # A replacement retires the generation the replaced schema's handles held; writes through them
        # raise ``StoreReplaced`` (reads still answer from the store they opened).
        self._replaced_schemas: set[str] = set()
        # A force-close under a handle (a profile delete's sweep) latches the schema; every later use
        # of a handle bound to it refuses instead of opening the store again.
        self._closed_schemas: set[str] = set()
        # Schemas this backend created (writer only); dropped on close so the shared test database
        # does not accumulate one schema per fresh store.
        self._created_schemas: set[str] = set()
        self._repl_counter = 0
        self.capabilities = dataclasses.replace(_BASE_CAPABILITIES, **_CAPABILITY_OVERRIDES)

    def open(self, scope: ProfileScope) -> "PostgresStore":
        """A handle bound to ``scope``'s store; opening touches no database. A profile this backend holds no
        store for is refused with ``ValueError`` (the suites' backend registers every profile it serves)."""
        if scope.hermes_home_key not in self._stores:
            raise ValueError(f"this postgres backend holds no store for {scope.profile_dir}")
        self._require_open()
        with self._lock:
            key = scope.hermes_home_key
            schema = self._schema_by_key.setdefault(key, schema_for_home(scope, self.schema_override))
        store = PostgresStore(self, scope, schema)
        with self._lock:
            self._open_handles.append(store)
        return store

    def _require_open(self) -> None:
        with self._lock:
            if self._closed:
                raise RuntimeError("postgres backend used after close")

    def _generation_for(self, schema: str) -> int:
        with self._generation_lock:
            return self._generations.get(schema, 0)

    def _bump_generation(self, schema: str) -> int:
        """A replacement of the schema (the suites' ``replace_store`` seam) retires the generation every
        handle opened before it held."""
        with self._generation_lock:
            self._generations[schema] = self._generations.get(schema, 0) + 1
            return self._generations[schema]

    def _release_connections(self) -> None:
        for store in list(self._open_handles):
            store.close()

    def reopen(self) -> None:
        """Drop every held connection; the next verb checks one out of the pool again (committed data is
        server-side, so nothing is lost - ``reopen_durability``)."""
        self._release_connections()

    def close(self) -> None:
        """Release every connection this backend checked out and refuse further use; the schemas this
        backend created are dropped (a fresh store's teardown), best effort."""
        with self._lock:
            self._closed = True
            created = list(self._created_schemas)
        self._release_connections()
        for schema in created:
            try:
                with translated():
                    with psycopg.connect(_dsn(), autocommit=True, connect_timeout=5) as conn:
                        conn.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
            except PersistenceError:
                pass  # a dropped database or unreachable server leaves the schema for the operator

    def close_store_under(self, scope: ProfileScope) -> None:
        """Force-close ``scope``'s store under its open handles, as a profile delete's sweep does: every
        handle bound to that store refuses on every use instead of opening the store again."""
        key = scope.hermes_home_key
        schema = None
        with self._lock:
            schema = self._schema_by_key.get(key) or schema_for_home(scope, self.schema_override)
            self._closed_schemas.add(schema)
        for store in list(self._open_handles):
            if store.schema == schema:
                store.close()

    def replace_store(self, scope: ProfileScope) -> None:
        """Put a replacement store where ``scope``'s store lives, so new handles open the replacement while
        open handles stay on the generation they hold (their writes now raise ``StoreReplaced``)."""
        key = scope.hermes_home_key
        with self._lock:
            old = self._schema_by_key.get(key) or schema_for_home(scope, self.schema_override)
            self._repl_counter += 1
            new = f"{old}_r{self._repl_counter}"[:63]
        _valid_schema_name(new)
        with translated():
            with psycopg.connect(_dsn(), autocommit=True, connect_timeout=30) as conn:
                conn.execute(f'CREATE SCHEMA "{new}"')
                conn.execute(f'SET search_path TO "{new}"')
                _ddl.apply(conn)
                conn.execute(f'SET search_path TO "{old}"')
                tables = [
                    row["table_name"]
                    for row in conn.execute(
                        "SELECT table_name FROM information_schema.tables "
                        "WHERE table_schema = %s AND table_type = 'BASE TABLE'", (old,)
                    )
                ]
                conn.execute(f'SET search_path TO "{new}"')
                for table in tables:
                    conn.execute(f'INSERT INTO {table} SELECT * FROM "{old}".{table}')
                # Identity sequences keep minting past the copied ids.
                for row in conn.execute(
                    "SELECT table_name, column_name FROM information_schema.columns "
                    "WHERE table_schema = %s AND is_identity = 'YES'", (new,)
                ):
                    table, column = row["table_name"], row["column_name"]
                    conn.execute(
                        f"""SELECT setval(
                                  (SELECT pg_get_serial_sequence('{new}', '{column}')),
                                  COALESCE((SELECT MAX({column}) FROM {table}), 1))"""
                    )
        with self._lock:
            self._schema_by_key[key] = new
            self._replaced_schemas.add(old)
            self._created_schemas.add(new)
        self._bump_generation(new)

    def open_store_admin(self, target: ProfileScope | Path, *, read_only: bool = True) -> StoreAdmin | AdminUnsupported:
        """PostgreSQL declares no physical admin capability (D9): the admin doors answer unsupported."""
        return AdminUnsupported("physical_admin", "postgres")

    def open_profile_repair(self, scope: ProfileScope, *, read_only: bool) -> ProfileRepairAdmin | AdminUnsupported:
        return AdminUnsupported("physical_admin", "postgres")

    def begin_epoch(self, scope: ProfileScope, instance_id: str) -> int:
        """Begin an authority epoch on ``scope``'s store and return it (the suites' seam: the same
        singleton-row advance and ``instance_id`` stamp the reference's runtime epoch write makes)."""
        store = self.open(scope)
        with store.tx() as conn:
            conn.execute(
                """INSERT INTO runtime_epoch (singleton, epoch, instance_id) VALUES (1, 1, %s)
                   ON CONFLICT (singleton) DO UPDATE SET epoch = runtime_epoch.epoch + 1,
                              instance_id = excluded.instance_id""",
                (instance_id,),
            )
            epoch = conn.execute("SELECT epoch FROM runtime_epoch WHERE singleton = 1").fetchone()
        return int(epoch["epoch"])


class PostgresStore:
    """One store handle bound to one profile: its schema, repositories, capability facts and the awaitable
    view ``aio``. The writer creates its schema on first write; the reader verifies the schema exists and
    never creates it. A store closed under its handle (or replaced out from under it) refuses at the
    connection doors: reads of a closed store raise, writes of a replaced store raise ``StoreReplaced``."""

    def __init__(self, backend: PostgresBackend, scope: ProfileScope, schema: str) -> None:
        self.scope = scope
        self.read_only = backend.read_only
        self.capabilities = backend.capabilities
        self.executor = backend.executor
        self._backend = backend
        self._schema = schema  # fixed at open: a replacement of the store binds to new handles only
        self._conn: Optional[Connection] = None
        self._schema_ready = False
        self._generation: Optional[int] = None
        self.aio = AwaitableStore(self)
        for attribute in CAPABILITY_ATTRIBUTES:
            factory = _REPOSITORY_FACTORIES.get(attribute)
            if factory is not None:
                setattr(self, attribute, factory(self))

    @property
    def schema(self) -> str:
        return self._schema

    # ── connection ──

    def _refuse_write(self) -> None:
        """A closed-under store refuses at every door; a replaced store refuses writes (its reads still
        answer from the store the handle opened)."""
        if self._schema in self._backend._closed_schemas:
            raise StoreQuarantined(f"the postgres store {self._schema!r} was closed under its handle; it will not be opened again")
        if self._schema in self._backend._replaced_schemas:
            raise StoreReplaced(f"the postgres store {self._schema!r} was replaced under this handle")

    def _refuse_read(self) -> None:
        if self._schema in self._backend._closed_schemas:
            raise StoreQuarantined(f"the postgres store {self._schema!r} was closed under its handle; it will not be opened again")

    def _schema_exists(self, conn: Connection) -> bool:
        return bool(conn.execute(
            "SELECT 1 FROM information_schema.schemata WHERE schema_name = %s", (self.schema,)
        ).fetchone())

    def _checkout(self, conn) -> None:
        """Pin a pooled connection to this store's schema (session ``search_path``; the pool serves
        every backend in the process, so the pin is per checkout, never global)."""
        conn.row_factory = psycopg.rows.dict_row
        if self.read_only:
            conn.set_read_only(True)
        conn.execute(f'SET search_path TO "{self.schema}"')

    def _read_conn(self) -> Optional[Connection]:
        """The handle's read connection, or ``None`` when the store has never been written (no schema
        yet): an uninitialised store reads empty, as an absent SQLite file does, and is never created
        by a read."""
        self._refuse_read()
        if self._conn is not None:
            return self._conn
        self._backend._require_open()
        dsn = _dsn()
        with translated():
            conn = _pool_for(dsn).getconn()
            self._checkout(conn)
            if not self._schema_exists(conn):
                conn.close()  # discard: the pool creates a replacement on its next checkout
                return None
        self._conn = conn
        return conn

    def connection(self) -> Connection:
        """The handle's write connection, checked out on first use and pinned to this store's schema;
        a read-only handle whose store was never written refuses (the reader never creates)."""
        self._refuse_write()
        if self._conn is not None:
            return self._conn
        self._backend._require_open()
        dsn = _dsn()
        with translated():
            conn = _pool_for(dsn).getconn()
            self._checkout(conn)
            if self.read_only and not self._schema_exists(conn):
                conn.close()  # discard: the pool creates a replacement on its next checkout
                raise StoreUnavailable(f"postgres store for schema {self.schema!r} is not initialized")
        self._conn = conn
        return conn

    def ensure_schema(self) -> None:
        """Create the store's schema and declarative DDL if the store has never been written (writer only):
        idempotent, one shot per handle."""
        if self.read_only:
            return
        if self._schema_ready:
            return
        if not self._schema_exists(self.connection()):
            with translated():
                with psycopg.connect(_dsn(), autocommit=True) as ddl:
                    try:
                        ddl.execute(f'CREATE SCHEMA "{self.schema}"')
                    except psycopg.errors.DuplicateSchema:
                        pass  # a concurrent creator won the race; the DDL below is idempotent
                    ddl.execute(f'SET search_path TO "{self.schema}"')
                    _ddl.apply(ddl)
                with self._backend._lock:
                    self._backend._created_schemas.add(self.schema)
        self._schema_ready = True
        self._generation = self._backend._generation_for(self.schema)

    # ── verbs' doors ──

    def ready(self) -> None:
        """Open the store now, so an unopenable store raises its typed error here instead of on the first verb."""
        self.connection()
        self.ensure_schema()

    def require_open(self) -> None:
        """Raise ``RuntimeError`` once the backend has closed, as a verb through the handle does."""
        self._backend._require_open()

    @contextmanager
    def tx(self) -> Iterator[Connection]:
        """One write transaction: the schema exists, the work runs on the handle's connection and commits, or
        rolls back and raises the contract error family."""
        self.require_open()
        with translated():
            self.ensure_schema()
            conn = self.connection()
            try:
                yield conn
                conn.commit()
            except BaseException:
                conn.rollback()
                raise

    # ── SQL ──

    def one(self, sql: str, *params: Any) -> Optional[dict[str, Any]]:
        """One row of a read, as a dict; ``None`` when the read is empty or the store is uninitialised."""
        conn = self._read_conn()
        if conn is None:
            return None
        with translated():
            return conn.execute(sql, params or None).fetchone()

    def all(self, sql: str, *params: Any) -> list[dict[str, Any]]:
        """Every row of a read, as dicts, in the statement's order; empty when the store is uninitialised."""
        conn = self._read_conn()
        if conn is None:
            return []
        with translated():
            return conn.execute(sql, params or None).fetchall()

    def scalar(self, sql: str, *params: Any) -> Any:
        """The single value of a single-column, single-row read; ``None`` when the read is empty."""
        row = self.one(sql, *params)
        if row is None:
            return None
        return next(iter(row.values()))

    def close(self) -> None:
        """Hand the handle's connection back to the pool (``putconn`` closes it only when the pool is full)."""
        conn, self._conn = self._conn, None
        if conn is not None:
            try:
                with translated():
                    _pool_for(_dsn()).putconn(conn)
            except PersistenceError:
                pass

    def serves_same_store(self, other: object) -> bool:
        """Whether ``other`` reads and writes through this handle's store: the same schema, and the same
        generation of it (a replacement of the schema retires the generation the older handle held)."""
        if not isinstance(other, PostgresStore) or self._backend is not other._backend:
            return False
        if self.schema != other.schema:
            return False
        if self is other:
            return True
        current = self._backend._generation_for(self.schema)
        return self._generation in (None, current) and other._generation in (None, current)

    @property
    def owner_profile_name(self) -> Optional[str]:
        """The profile owning this store: the scope's name (``default`` for the default root, the profile's
        name for a named profile home), None for a folder outside the profile tree - the SQLite store's
        own-profile derivation, never a guess."""
        from persistence_assembly import store_owner_name

        return store_owner_name(Path(self.scope.profile_dir))


register_health()

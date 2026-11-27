"""The shared helpers of the hosted-room repositories on PostgreSQL (stage 4): the placement doors
(the profile store's schema vs. the backend's fixed shared schema), the PG-dialect twins of the
reference's SQL shapes (its ``json_extract`` and ``length(CAST ... AS BLOB)`` reads), the epoch
fence of the owner verbs, the fenced updates, and the non-blocking probes.

The reference's two files (``state.db`` and the install's ``shared-state.db``) map onto the store's
own schema and the backend's ONE fixed shared schema: a per-profile schema could not be shared, so
the shared placement of the hosted rooms takes the install's shared place as one schema every
profile of the install reaches. The reference's schema steps and legacy migrations are the
declarative DDL here: every hosted table is created with the schema, and the optional
files-only tables a files-only source carries are simply absent.
"""

from __future__ import annotations

from contextlib import closing, contextmanager
from pathlib import Path
from typing import Any, Optional

import psycopg

from persistence import FenceStale, PermanentStoreError
from persistence.core.errors import StaleFenceKind
from persistence.gateway.hosted_rooms import HostedPlacement, RoomProbeUnavailableError

from .._backend import PostgresStore, _dsn

__all__ = [
    "attachment_root",
    "byte_len",
    "fenced_update",
    "jstr",
    "placement_all",
    "placement_one",
    "placement_read",
    "placement_scalar",
    "placement_write",
    "probe",
    "table_exists",
]

ATTACHMENT_DIR_NAME = "hosted-room-attachments"


def _check_placement(placement: HostedPlacement) -> None:
    if placement not in ("profile", "shared"):
        raise ValueError(f"unknown hosted-room placement {placement!r}")


@contextmanager
def placement_write(store: PostgresStore, placement: HostedPlacement):
    """One write transaction on the placement's store: the profile store's schema for ``'profile'``,
    the backend's fixed shared schema for ``'shared'``. A read-only handle refuses the write before
    anything opens (the reference's ``refuse_read_only``), which the profile write door would never
    do on its own (a written store's connection is still writable SQL-wise, and the shared schema
    may be missing, which the shared write door answers with ``StoreUnavailable``)."""
    if store.read_only:
        raise PermanentStoreError("hosted-room write refused: the store handle is read-only")
    _check_placement(placement)
    if placement == "shared":
        with store.shared_tx() as conn:
            yield conn
    else:
        with store.tx() as conn:
            yield conn


def placement_one(store: PostgresStore, placement: HostedPlacement, sql: str, *params: Any) -> Optional[dict[str, Any]]:
    """One row of a read on the placement's store, as a dict; ``None`` when the read is empty or the
    placement's store has never been written (the reference's missing file reads empty)."""
    _check_placement(placement)
    if placement == "shared":
        return store.shared_one(sql, *params)
    return store.one(sql, *params)


def placement_all(store: PostgresStore, placement: HostedPlacement, sql: str, *params: Any) -> list[dict[str, Any]]:
    """Every row of a read on the placement's store, as dicts, in the statement's order; empty when the
    placement's store has never been written."""
    _check_placement(placement)
    if placement == "shared":
        return store.shared_all(sql, *params)
    return store.all(sql, *params)


def placement_scalar(store: PostgresStore, placement: HostedPlacement, sql: str, *params: Any) -> Any:
    """The single value of a single-column, single-row read on the placement's store; ``None`` when the
    read is empty."""
    _check_placement(placement)
    if placement == "shared":
        return store.shared_scalar(sql, *params)
    return store.scalar(sql, *params)


class _StatementRows:
    """The read doors' rows in cursor shape: ``fetchone``/``fetchall`` over the door's dict rows."""

    def __init__(self, rows: list) -> None:
        self._rows = list(rows)
        self._i = 0

    def fetchone(self):
        if self._i < len(self._rows):
            row = self._rows[self._i]
            self._i += 1
            return row
        return None

    def fetchall(self):
        rows, self._i = self._rows[self._i:], 0
        return rows


class _StatementRead:
    """The placement's read doors in cursor shape: ``execute`` returns a result over the door's rows,
    so the reference's read shapes (their ``conn.execute(...).fetchall()``) run unchanged on a
    read-only handle (a read-only handle never creates, and an uninitialised store reads empty)."""

    def __init__(self, store: PostgresStore, placement: HostedPlacement) -> None:
        self._store = store
        self._placement = placement

    def execute(self, sql: str, params: tuple = ()) -> "_StatementRows":
        if self._placement == "shared":
            return _StatementRows(self._store.shared_all(sql, *params))
        return _StatementRows(self._store.all(sql, *params))


@contextmanager
def placement_read(store: PostgresStore, placement: HostedPlacement):
    """One read on the placement's store: a read transaction when the handle may write (the reference's
    first-touch read), the statement doors otherwise (the read-only twin: reads only, never creates)."""
    _check_placement(placement)
    if store.read_only:
        yield _StatementRead(store, placement)
    else:
        with placement_write(store, placement) as conn:
            yield conn


def _epoch(conn, epoch: int) -> None:
    """The epoch fence of the owner verbs: the singleton row's stored epoch, else the contract's
    ``FenceStale`` (the reference's ``_epoch``)."""
    row = conn.execute("SELECT epoch FROM runtime_epoch WHERE singleton = 1").fetchone()
    if type(epoch) is not int or row is None or row["epoch"] != epoch:
        raise FenceStale(StaleFenceKind.STALE_EPOCH, "runtime_epoch", "singleton")


def fenced_update(conn, sql: str, params: tuple, error: Exception) -> None:
    """Run a compare-and-swap UPDATE; anything but exactly one row means the fence was lost."""
    if conn.execute(sql, params).rowcount != 1:
        raise error


def table_exists(conn, table: str) -> bool:
    """Whether ``table`` is in the pinned schema (the declarative DDL carries every table the hosted
    verbs need; the optional safety/retirement tables a files-only source carries are not installed
    and read absent, as the reference's files-only check does)."""
    row = conn.execute(
        "SELECT 1 FROM information_schema.tables WHERE table_schema = current_schema() AND table_name = %s",
        (table,)).fetchone()
    return row is not None


def jstr(expr: str, key: str) -> str:
    """The SQLite ``json_extract(expr, '$.key')`` twin on the PG dialect, as text: ``NULL`` when the
    stored text is not JSON (the reference's ``json_extract`` returns NULL there), through the DDL's
    ``hermes_json_valid`` guard (a bare ``::jsonb`` cast would raise on a malformed page)."""
    return f"CASE WHEN hermes_json_valid({expr}) THEN {expr}::jsonb ->> '{key}' END"


def byte_len(expr: str) -> str:
    """The SQLite ``length(CAST(expr AS BLOB))`` twin on the PG dialect: the value's UTF-8 byte length for
    every column type (an integer cast to ``octet_length`` does not exist; text/JSON go through the
    same cast, so a UTF-8 multibyte character charges as bytes, as the BLOB twin did)."""
    return f"octet_length(convert_to({expr}::text, 'UTF8'))"


def probe(store: PostgresStore, placement: HostedPlacement, table: str, sql: str,
          params: tuple[Any, ...], unavailable: str) -> bool:
    """The non-blocking existence probe (the reference's 0.05 s-timeout twin): a short-lived
    connection with a short connect timeout, no schema creation or migration. ``False`` when the
    placement's store - or the part of it that keeps the probed rows - does not exist yet; a store
    failure raises ``RoomProbeUnavailableError`` with the placement's message."""
    _check_placement(placement)
    schema = store.shared_schema if placement == "shared" else store.schema
    try:
        with closing(psycopg.connect(_dsn(), autocommit=True, connect_timeout=2)) as conn:
            if not conn.execute(
                "SELECT 1 FROM information_schema.schemata WHERE schema_name = %s", (schema,)
            ).fetchone():
                return False
            conn.execute(f'SET search_path TO "{schema}"')
            if not conn.execute(
                "SELECT 1 FROM information_schema.tables WHERE table_schema = %s AND table_name = %s",
                (schema, table)).fetchone():
                return False
            return conn.execute(sql, params).fetchone() is not None
    except psycopg.Error as exc:
        raise RoomProbeUnavailableError(unavailable) from exc


def attachment_root(store: PostgresStore, placement: HostedPlacement) -> Path:
    """The attachment directory of the placement (the reference's ``HostedFiles.attachment_root``):
    beside the placement's store - the profile's registered ``state.db`` location for ``'profile'``,
    the install root the location belongs to for ``'shared'`` (the reference's ``shared_state_path``
    twin over the registered location, never a guess from the scope)."""
    _check_placement(placement)
    location = store._backend._stores.get(store.scope.hermes_home_key)
    if location is None:  # a suite serving a scope the backend registered nowhere: the folder itself
        location = Path(store.scope.profile_dir)
    location = Path(location)
    if placement == "shared":
        # The shared placement's store is the install's shared-state place: the install root the
        # location belongs to - the parent of the location's ``profiles`` directory (one shared
        # root for every profile of the assembly; a location not under a profiles directory keeps
        # its own parent as the root).
        for ancestor in (location.parent, *location.parent.parents):
            if ancestor.name == "profiles":
                return ancestor.parent / ATTACHMENT_DIR_NAME
        return location.parent / ATTACHMENT_DIR_NAME
    return location.parent / ATTACHMENT_DIR_NAME

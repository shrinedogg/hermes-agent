"""The PostgreSQL backend of the contract suites, registered as ``postgres`` when ``HERMES_DATABASE_URL`` is set
and the ``persistence.postgres`` package is importable in this tree (plan D10: without a configured server the
backend registers nowhere, so no suite collects cases it cannot run)."""

from __future__ import annotations

import importlib.util
import os
import tempfile
from contextlib import contextmanager
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Iterator, Mapping, Sequence

from persistence import ProfileScope

if TYPE_CHECKING:
    from tests.contracts._postgres_seams import ContractPostgresBackend


def _postgres_backend_available() -> bool:
    """Whether the PostgreSQL backend registers in this process: a database URL is configured and the
    backend package is importable in this tree (D10)."""
    if not os.environ.get("HERMES_DATABASE_URL"):
        return False
    try:
        return importlib.util.find_spec("persistence.postgres") is not None
    except ImportError:
        return False


@contextmanager
def _open_postgres_backend(scopes: Sequence[ProfileScope]) -> Iterator[ContractPostgresBackend]:
    """The PostgreSQL backend over the test scopes: each profile's store is a schema (``hermes_<profile>_<hash>``)
    in the one shared database ``HERMES_DATABASE_URL`` names, created on its first write and dropped when the
    backend closes, so every factory call starts from empty stores. The backend's folder is the Hermes root
    while it is open (the store derives its owner profile from its place under that root, as the reference
    derives it from its own); every handle carries the seeding seams of ``tests/contracts/_postgres_seams.py``."""
    from hermes_constants import get_hermes_home
    from tests.contracts._postgres_seams import ContractPostgresBackend

    root = Path(tempfile.mkdtemp(prefix="postgres-backend-", dir=get_hermes_home()))
    stores: dict[str, Path] = {}
    for scope in scopes:
        home = root / "profiles" / scope.name
        home.mkdir(parents=True, exist_ok=True)
        stores[scope.hermes_home_key] = home / "state.db"
    # The backend's folder is the Hermes root while it is open: the store derives its owner profile from its
    # path under that root, as the reference derives it from its own.
    previous = os.environ.get("HERMES_HOME")
    os.environ["HERMES_HOME"] = str(root)
    backend = ContractPostgresBackend(stores, root=root)
    try:
        yield backend
    finally:
        backend.close()
        if previous is None:
            os.environ.pop("HERMES_HOME", None)
        else:
            os.environ["HERMES_HOME"] = previous


def register(register_backend: Callable[..., None], scopes: Sequence[ProfileScope]) -> None:
    """Register ``postgres`` over ``scopes`` when the backend package is importable and a database URL is
    configured (D10)."""
    if _postgres_backend_available():
        register_backend("postgres", partial(_open_postgres_backend, tuple(scopes)))

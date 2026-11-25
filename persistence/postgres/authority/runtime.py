"""``SessionRuntimeRepository`` on PostgreSQL: not ported yet (the local runtime epoch and receipts (stage 2)); every verb raises ``NotImplementedError``
naming itself (D8)."""

from __future__ import annotations

from persistence.authority.runtime import SessionRuntimeRepository

from .._backend import register_repository
from .._stubs import add_stubs

__all__ = ["PgRuntimeRepository"]


class PgRuntimeRepository:
    """Unported on PostgreSQL (the local runtime epoch and receipts (stage 2)): every verb raises naming itself (D8)."""


add_stubs(PgRuntimeRepository, SessionRuntimeRepository)


def _register() -> None:
    register_repository("runtime", lambda store: PgRuntimeRepository())


_register()

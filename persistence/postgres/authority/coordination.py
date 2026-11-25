"""``CoordinationRepository`` on PostgreSQL: not ported yet (cross-process coordination (stage 2)); every verb raises ``NotImplementedError``
naming itself (D8)."""

from __future__ import annotations

from persistence.authority.coordination import CoordinationRepository

from .._backend import register_repository
from .._stubs import add_stubs

__all__ = ["PgCoordinationRepository"]


class PgCoordinationRepository:
    """Unported on PostgreSQL (cross-process coordination (stage 2)): every verb raises naming itself (D8)."""


add_stubs(PgCoordinationRepository, CoordinationRepository)


def _register() -> None:
    register_repository("coordination", lambda store: PgCoordinationRepository())


_register()

"""``SessionBindingRepository`` on PostgreSQL: not ported yet (session bindings (stage 2)); every verb raises ``NotImplementedError``
naming itself (D8)."""

from __future__ import annotations

from persistence.authority.bindings import SessionBindingRepository

from .._backend import register_repository
from .._stubs import add_stubs

__all__ = ["PgBindingsRepository"]


class PgBindingsRepository:
    """Unported on PostgreSQL (session bindings (stage 2)): every verb raises naming itself (D8)."""


add_stubs(PgBindingsRepository, SessionBindingRepository)


def _register() -> None:
    register_repository("bindings", lambda store: PgBindingsRepository())


_register()

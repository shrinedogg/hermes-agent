"""``RoutingRepository`` on PostgreSQL: not ported yet (gateway routing (stage 4)); every verb raises ``NotImplementedError``
naming itself (D8)."""

from __future__ import annotations

from persistence.gateway.routing import RoutingRepository

from .._backend import register_repository
from .._stubs import add_stubs

__all__ = ["PgRoutingRepository"]


class PgRoutingRepository:
    """Unported on PostgreSQL (gateway routing (stage 4)): every verb raises naming itself (D8)."""


add_stubs(PgRoutingRepository, RoutingRepository)


def _register() -> None:
    register_repository("routing", lambda store: PgRoutingRepository())


_register()

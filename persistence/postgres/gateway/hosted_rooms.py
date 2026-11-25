"""``HostedRoomRepository`` on PostgreSQL: not ported yet (hosted rooms (stage 4)); every verb raises ``NotImplementedError``
naming itself (D8)."""

from __future__ import annotations

from persistence.gateway.hosted_rooms import HostedRoomRepository

from .._backend import register_repository
from .._stubs import add_stubs

__all__ = ["PgHostedRoomRepository"]


class PgHostedRoomRepository:
    """Unported on PostgreSQL (hosted rooms (stage 4)): every verb raises naming itself (D8)."""


add_stubs(PgHostedRoomRepository, HostedRoomRepository)


def _register() -> None:
    register_repository("hosted_rooms", lambda store: PgHostedRoomRepository())


_register()

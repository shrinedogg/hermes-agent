"""``HostedRoomDriverRepository`` on PostgreSQL: not ported yet (hosted-room driver (stage 4)); every verb raises ``NotImplementedError``
naming itself (D8)."""

from __future__ import annotations

from persistence.gateway.hosted_rooms import HostedRoomDriverRepository

from .._backend import register_repository
from .._stubs import add_stubs

__all__ = ["PgHostedRoomDriverRepository"]


class PgHostedRoomDriverRepository:
    """Unported on PostgreSQL (hosted-room driver (stage 4)): every verb raises naming itself (D8)."""


add_stubs(PgHostedRoomDriverRepository, HostedRoomDriverRepository)


def _register() -> None:
    register_repository("hosted_room_driver", lambda store: PgHostedRoomDriverRepository())


_register()

"""``HostedRoomPeerRepository`` on PostgreSQL: not ported yet (hosted-room peers (stage 4)); every verb raises ``NotImplementedError``
naming itself (D8)."""

from __future__ import annotations

from persistence.gateway.hosted_rooms import HostedRoomPeerRepository

from .._backend import register_repository
from .._stubs import add_stubs

__all__ = ["PgHostedRoomPeerRepository"]


class PgHostedRoomPeerRepository:
    """Unported on PostgreSQL (hosted-room peers (stage 4)): every verb raises naming itself (D8)."""


add_stubs(PgHostedRoomPeerRepository, HostedRoomPeerRepository)


def _register() -> None:
    register_repository("hosted_room_peers", lambda store: PgHostedRoomPeerRepository())


_register()

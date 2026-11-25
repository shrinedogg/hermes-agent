"""``HostedRoomReplicaRepository`` on PostgreSQL: not ported yet (hosted-room replicas (stage 4)); every verb raises ``NotImplementedError``
naming itself (D8)."""

from __future__ import annotations

from persistence.gateway.hosted_rooms import HostedRoomReplicaRepository

from .._backend import register_repository
from .._stubs import add_stubs

__all__ = ["PgHostedRoomReplicaRepository"]


class PgHostedRoomReplicaRepository:
    """Unported on PostgreSQL (hosted-room replicas (stage 4)): every verb raises naming itself (D8)."""


add_stubs(PgHostedRoomReplicaRepository, HostedRoomReplicaRepository)


def _register() -> None:
    register_repository("hosted_room_replicas", lambda store: PgHostedRoomReplicaRepository())


_register()

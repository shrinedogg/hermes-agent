"""``HostedRoomPolicyRepository`` on PostgreSQL: not ported yet (hosted-room policy (stage 4)); every verb raises ``NotImplementedError``
naming itself (D8)."""

from __future__ import annotations

from persistence.gateway.hosted_rooms import HostedRoomPolicyRepository

from .._backend import register_repository
from .._stubs import add_stubs

__all__ = ["PgHostedRoomPolicyRepository"]


class PgHostedRoomPolicyRepository:
    """Unported on PostgreSQL (hosted-room policy (stage 4)): every verb raises naming itself (D8)."""


add_stubs(PgHostedRoomPolicyRepository, HostedRoomPolicyRepository)


def _register() -> None:
    register_repository("hosted_room_policy", lambda store: PgHostedRoomPolicyRepository())


_register()

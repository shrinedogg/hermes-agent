"""``HostedRoomAttachmentRepository`` on PostgreSQL: not ported yet (hosted-room attachments (stage 4)); every verb raises ``NotImplementedError``
naming itself (D8)."""

from __future__ import annotations

from persistence.gateway.hosted_rooms import HostedRoomAttachmentRepository

from .._backend import register_repository
from .._stubs import add_stubs

__all__ = ["PgHostedRoomAttachmentRepository"]


class PgHostedRoomAttachmentRepository:
    """Unported on PostgreSQL (hosted-room attachments (stage 4)): every verb raises naming itself (D8)."""


add_stubs(PgHostedRoomAttachmentRepository, HostedRoomAttachmentRepository)


def _register() -> None:
    register_repository("hosted_room_attachments", lambda store: PgHostedRoomAttachmentRepository())


_register()

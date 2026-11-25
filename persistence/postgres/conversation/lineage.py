"""``LineageRepository`` on PostgreSQL: not ported yet (compression-chain and resume-lineage reads (stage 1)); every verb raises ``NotImplementedError``
naming itself (D8)."""

from __future__ import annotations

from persistence.conversation.lineage import LineageRepository

from .._backend import register_repository
from .._stubs import add_stubs

__all__ = ["PgLineageRepository"]


class PgLineageRepository:
    """Unported on PostgreSQL (compression-chain and resume-lineage reads (stage 1)): every verb raises naming itself (D8)."""


add_stubs(PgLineageRepository, LineageRepository)


def _register() -> None:
    register_repository("lineage", lambda store: PgLineageRepository())


_register()

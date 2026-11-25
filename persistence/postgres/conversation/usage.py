"""``UsageRepository`` and ``UsageAnalytics`` on PostgreSQL: not ported yet (stage 1); every verb raises
``NotImplementedError`` naming itself (D8)."""

from __future__ import annotations

from persistence.conversation.usage import UsageAnalytics, UsageRepository

from .._backend import register_repository
from .._stubs import add_stubs

__all__ = ["PgUsageRepository"]


class PgUsageRepository:
    """Unported on PostgreSQL (stage 1): every verb raises naming itself (D8)."""


add_stubs(PgUsageRepository, UsageRepository)
add_stubs(PgUsageRepository, UsageAnalytics)  # one stub class carries both conversation protocols


def _register() -> None:
    register_repository("usage", lambda store: PgUsageRepository())
    register_repository("usage_analytics", lambda store: PgUsageRepository())


_register()

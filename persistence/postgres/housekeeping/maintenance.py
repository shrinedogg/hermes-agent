"""``SessionMaintenance`` on PostgreSQL: not ported yet (prune/archive/export maintenance (stage 1)); every verb raises ``NotImplementedError``
naming itself (D8)."""

from __future__ import annotations

from persistence.housekeeping.maintenance import SessionMaintenance

from .._backend import register_repository
from .._stubs import add_stubs

__all__ = ["PgMaintenanceRepository"]


class PgMaintenanceRepository:
    """Unported on PostgreSQL (prune/archive/export maintenance (stage 1)): every verb raises naming itself (D8)."""


add_stubs(PgMaintenanceRepository, SessionMaintenance)


def _register() -> None:
    register_repository("maintenance", lambda store: PgMaintenanceRepository())


_register()

"""``WorkerExecutionRepository`` on PostgreSQL: not ported yet (worker execution receipts (stage 2)); every verb raises ``NotImplementedError``
naming itself (D8)."""

from __future__ import annotations

from persistence.workers.executions import WorkerExecutionRepository

from .._backend import register_repository
from .._stubs import add_stubs

__all__ = ["PgWorkerExecutionsRepository"]


class PgWorkerExecutionsRepository:
    """Unported on PostgreSQL (worker execution receipts (stage 2)): every verb raises naming itself (D8)."""


add_stubs(PgWorkerExecutionsRepository, WorkerExecutionRepository)


def _register() -> None:
    register_repository("worker_executions", lambda store: PgWorkerExecutionsRepository())


_register()

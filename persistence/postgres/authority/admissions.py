"""``AdmissionRepository`` on PostgreSQL: not ported yet (the session admissions ledger (stage 2)); every verb raises ``NotImplementedError``
naming itself (D8)."""

from __future__ import annotations

from persistence.authority.admissions import AdmissionRepository

from .._backend import register_repository
from .._stubs import add_stubs

__all__ = ["PgAdmissionsRepository"]


class PgAdmissionsRepository:
    """Unported on PostgreSQL (the session admissions ledger (stage 2)): every verb raises naming itself (D8)."""


add_stubs(PgAdmissionsRepository, AdmissionRepository)


def _register() -> None:
    register_repository("admissions", lambda store: PgAdmissionsRepository())


_register()

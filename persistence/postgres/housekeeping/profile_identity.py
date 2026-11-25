"""``ProfileIdentityRepository`` on PostgreSQL: not ported yet (profile rekey and purge (stage 1)); every verb raises ``NotImplementedError``
naming itself (D8)."""

from __future__ import annotations

from persistence.housekeeping.profile_identity import ProfileIdentityRepository

from .._backend import register_repository
from .._stubs import add_stubs

__all__ = ["PgProfileIdentityRepository"]


class PgProfileIdentityRepository:
    """Unported on PostgreSQL (profile rekey and purge (stage 1)): every verb raises naming itself (D8)."""


add_stubs(PgProfileIdentityRepository, ProfileIdentityRepository)


def _register() -> None:
    register_repository("profile_identity", lambda store: PgProfileIdentityRepository())


_register()

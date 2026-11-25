"""``SessionSearch`` on PostgreSQL: not ported yet (full-text search (stage 3)); every verb raises ``NotImplementedError``
naming itself (D8)."""

from __future__ import annotations

from persistence.conversation.search import SessionSearch

from .._backend import register_repository
from .._stubs import add_stubs

__all__ = ["PgSearchRepository"]


class PgSearchRepository:
    """Unported on PostgreSQL (full-text search (stage 3)): every verb raises naming itself (D8)."""


add_stubs(PgSearchRepository, SessionSearch)


def _register() -> None:
    register_repository("search", lambda store: PgSearchRepository())


_register()

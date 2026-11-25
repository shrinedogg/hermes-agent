"""``StateMetaRepository`` on PostgreSQL: the ``state_meta`` key/value rows, read and written on the
store's own schema (``seq`` mirrors the SQLite rowid: listings come back in first-stored order, and an
upsert keeps the key's original place)."""

from __future__ import annotations

from typing import Optional

from persistence.housekeeping.state_meta import StateMetaEntry

from .._backend import PostgresStore, register_repository, translated

__all__ = ["PgStateMetaRepository"]


def _prefix_match(key: str, prefix: str) -> bool:
    """The SQLite ``LIKE`` prefix semantics the store applies (wildcards escaped): ASCII letters
    match either case, every other character - including ``_`` and ``%`` - matches only itself."""
    if len(key) < len(prefix):
        return False
    for key_char, prefix_char in zip(key[: len(prefix)], prefix):
        if key_char == prefix_char:
            continue
        if prefix_char.isascii() and prefix_char.isalpha() and key_char.upper() == prefix_char.upper():
            continue
        return False
    return True


class PgStateMetaRepository:
    """Core-owned bookkeeping keys of the bound profile's state store."""

    def __init__(self, store: PostgresStore) -> None:
        self._store = store

    def get(self, key: str) -> Optional[str]:
        if not key:
            raise ValueError("state_meta key must be non-empty")
        row = self._store.one("SELECT value FROM state_meta WHERE key = %s", key)
        return None if row is None else row["value"]

    def set(self, key: str, value: str) -> None:
        if not key:
            raise ValueError("state_meta key must be non-empty")
        with self._store.tx() as conn:
            conn.execute(
                """INSERT INTO state_meta (key, value) VALUES (%s, %s)
                   ON CONFLICT (key) DO UPDATE SET value = excluded.value""",
                (key, value),
            )

    def list_prefix(self, prefix: str) -> tuple[StateMetaEntry, ...]:
        if not prefix:
            raise ValueError("state_meta prefix must be non-empty")
        rows = self._store.all("SELECT key, value FROM state_meta ORDER BY seq")
        # The listing is the store's table order (first-stored first); the prefix is literal with the
        # LIKE case fold, applied in Python so no locale changes the match (the store's LIKE does the
        # same for ASCII).
        return tuple(
            StateMetaEntry(row["key"], row["value"]) for row in rows if _prefix_match(row["key"], prefix)
        )


register_repository("state_meta", PgStateMetaRepository)

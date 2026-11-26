"""Session-row helpers the ported message verbs need (the SQLite sessions-store functions in
``%s`` form, connection-first like the rest of the PG store): the model-config merge, a session
row as a dict, and the compression-child gates. ``model_config`` is a TEXT JSON column; every
consumer accepts the stored string (the shared ``_is_explicit_fork_child_row`` parses it in
Python)."""

from __future__ import annotations

import json
from typing import Any, Mapping, Optional

from ..sessions import _parse_model_config
from .common import _is_explicit_fork_child_row

__all__ = [
    "_is_compression_child_row", "_merge_model_config_json", "get_session",
]

_MODEL_CONFIG_ROW_MISSING = object()


def get_session(store, session_id: str) -> Optional[dict[str, Any]]:
    """The session row as a dict (``None`` when absent); the conversation verbs use only the
    identity/parent/lineage fields, so the resolved-prompt joins of the SQLite reader are dropped."""
    return store.one("SELECT * FROM sessions WHERE id = %s", session_id)


def _merge_model_config_json(conn, session_id: str, patch: dict[str, Any], *,
                            on_missing: str = "skip") -> Any:
    """SELECT + tolerant-parse + merge *patch* into ``model_config`` on the caller's connection (the
    SQLite ``_merge_model_config_json`` port, connection-first; ``None`` deletes a key). Returns the
    serialized JSON (``None`` when empty) or ``_MODEL_CONFIG_ROW_MISSING``; ``on_missing="raise"``
    raises ``ValueError`` instead (a compaction never commits against a vanished session row)."""
    row = conn.execute("SELECT model_config FROM sessions WHERE id = %s", (session_id,)).fetchone()
    if row is None:
        if on_missing == "raise":
            raise ValueError(f"Session not found: {session_id}")
        return _MODEL_CONFIG_ROW_MISSING
    config = _parse_model_config(row["model_config"])
    for key, value in patch.items():
        if value is None:
            config.pop(key, None)
        else:
            config[key] = value
    return json.dumps(config) if config else None


def _is_compression_child_row(conn, child: Mapping[str, Any]) -> bool:
    """Whether *child* continues a compression lineage (the SQLite ``_is_compression_child_row``
    port, per-hop gate over the caller's connection): an explicit branch/delegate/tool child (a reset
    fork included) is its own conversation, and its parent must have ended ``compression``."""
    parent_id = child.get("parent_session_id")
    if not parent_id or _is_explicit_fork_child_row(child, include_reset=True):
        return False
    parent = conn.execute("SELECT end_reason FROM sessions WHERE id = %s", (parent_id,)).fetchone()
    return bool(parent and parent["end_reason"] == "compression")

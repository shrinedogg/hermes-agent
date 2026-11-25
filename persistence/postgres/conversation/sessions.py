"""``SessionRepository`` on PostgreSQL: the slice verbs ported from the SQLite reference (``create``,
``get``); every other sessions verb raises ``NotImplementedError`` naming itself (D8).

``create`` is the keep-existing constructor upsert: it never overwrites what an earlier writer set,
stamps a missing ``profile_name`` with the store's own, stores the system prompt content-addressed in
``system_prompts``, backfills a child's cwd/git/profile from its parent and, for compression forks,
the gateway routing columns too. ``get`` reads the row with the prompt-store joins and maps it through
the shared :func:`session_record` projection (degraded-instant rules included).
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from typing import Any, Mapping, Optional

from persistence import AuthorityRefusal, RetiredSessionError, SessionId
from persistence.conversation.sessions import (
    SessionLineageKind, SessionRecord, SessionRepository, TitleSource, is_corrupt_model_config,
)
from persistence.core.conventions import UNTRUSTED_INSTANT, stored_instant
from persistence.sqlite.conversation._store.retirement import RETIRED_PREFIX

from .._backend import PostgresStore, register_repository
from .._stubs import add_stubs

__all__ = ["PgSessionsRepository", "session_record"]

# The fork markers a child carries in ``model_config``, in the order the first non-blank one names its kind.
_MARKERS = (
    ("_delegate_from", SessionLineageKind.DELEGATE),
    ("_branched_from", SessionLineageKind.BRANCH),
    ("_reset_from", SessionLineageKind.RESET),
)
_MARKER_OF_KIND = {kind: key for key, kind in _MARKERS}
_TITLE_SOURCES = {source.value: source for source in TitleSource}
# The content hash a listing row's ``tool_names`` holds when the row was read without the prompt-store join.
_PROMPT_HASH = re.compile(r"[0-9a-f]{64}")


def _stripped(value: Any) -> Optional[str]:
    return value.strip() if isinstance(value, str) and value.strip() else None


def _json_value(text: Any) -> Any:
    """A stored JSON column decoded, ``None`` when unset or unreadable."""
    if not isinstance(text, str) or not text:
        return None
    try:
        return json.loads(text)
    except ValueError:
        return None


def _tool_pin(text: Any) -> Any:
    """The decoded ``tools[]`` pin; ``None`` for an unresolved content hash."""
    return None if isinstance(text, str) and _PROMPT_HASH.fullmatch(text) else _json_value(text)


def _parse_model_config(raw: Any) -> dict[str, Any]:
    """Tolerant ``model_config`` decode: JSON text or dict -> dict copy; anything else -> {}."""
    if isinstance(raw, str) and raw.strip():
        try:
            raw = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return {}
    return dict(raw) if isinstance(raw, dict) else {}


def session_record(row: Mapping[str, Any]) -> SessionRecord:
    """The record of one stored session row (the SQLite ``session_record`` projection, verbatim
    semantics: lineage markers lifted from ``model_config``, degraded stored instants)."""
    config = _parse_model_config(row.get("model_config"))
    kind, origin_id = SessionLineageKind.NONE, None
    for key, marker_kind in _MARKERS:
        if _stripped(config.get(key)):
            kind, origin_id = marker_kind, config[key]
            break
    origin = _json_value(row.get("origin_json"))
    started = stored_instant(row.get("started_at"), session_id=row["id"], field="started_at")
    ended = stored_instant(row.get("ended_at"), session_id=row["id"], field="ended_at")
    return SessionRecord(
        session_id=SessionId(row["id"]),
        source=row.get("source") or "",
        created_source=row.get("created_source"),
        user_id=row.get("user_id"),
        session_key=row.get("session_key"),
        chat_id=row.get("chat_id"),
        chat_type=row.get("chat_type"),
        thread_id=row.get("thread_id"),
        display_name=row.get("display_name"),
        origin=dict(origin) if isinstance(origin, Mapping) else None,
        expiry_finalized=bool(row.get("expiry_finalized")),
        model=row.get("model"),
        model_config=config,
        system_prompt=row.get("system_prompt"),
        system_prompt_hash=row.get("system_prompt_hash"),
        parent_session_id=row.get("parent_session_id"),
        lineage_kind=kind,
        reset_from=SessionId(_stripped(config.get("_reset_from"))) if _stripped(config.get("_reset_from")) else None,
        branched_from=SessionId(_stripped(config.get("_branched_from"))) if _stripped(config.get("_branched_from")) else None,
        started_at=(started or UNTRUSTED_INSTANT).timestamp(),
        ended_at=None if ended is None else ended.timestamp(),
        end_reason=row.get("end_reason"),
        message_count=row.get("message_count", 0),
        tool_call_count=row.get("tool_call_count", 0),
        input_tokens=row.get("input_tokens", 0),
        output_tokens=row.get("output_tokens", 0),
        cache_read_tokens=row.get("cache_read_tokens", 0),
        cache_write_tokens=row.get("cache_write_tokens", 0),
        reasoning_tokens=row.get("reasoning_tokens", 0),
        api_call_count=row.get("api_call_count", 0),
        estimated_cost_usd=row.get("estimated_cost_usd"),
        actual_cost_usd=row.get("actual_cost_usd"),
        billing_provider=row.get("billing_provider"),
        billing_base_url=row.get("billing_base_url"),
        billing_mode=row.get("billing_mode"),
        cost_status=row.get("cost_status"),
        cost_source=row.get("cost_source"),
        pricing_version=row.get("pricing_version"),
        cwd=row.get("cwd"),
        git_branch=row.get("git_branch"),
        git_repo_root=row.get("git_repo_root"),
        git_metadata_generation=int(row.get("git_metadata_generation") or 0),
        title=row.get("title"),
        title_source=_TITLE_SOURCES.get(row.get("title_source")),
        last_activity_at=row.get("last_activity_at"),
        last_activity_description=row.get("last_activity_description"),
        last_activity_provenance=row.get("last_activity_provenance"),
        handoff_state=row.get("handoff_state"),
        handoff_platform=row.get("handoff_platform"),
        handoff_error=row.get("handoff_error"),
        compression_failure_cooldown_until=row.get("compression_failure_cooldown_until"),
        compression_failure_error=row.get("compression_failure_error"),
        compression_fallback_streak=int(row.get("compression_fallback_streak") or 0),
        compression_ineffective_count=int(row.get("compression_ineffective_count") or 0),
        compression_recovery_deadline=row.get("compression_recovery_deadline"),
        compression_overload_streak=int(row.get("compression_overload_streak") or 0),
        profile_name=row.get("profile_name"),
        transport_profile=row.get("transport_profile"),
        rewind_count=int(row.get("rewind_count") or 0),
        archived=bool(row.get("archived")),
        auto_archived=bool(row.get("auto_archived")),
        pinned=bool(row.get("pinned")),
        hidden=bool(row.get("hidden")),
        last_read_at=row.get("last_read_at"),
        tool_pin=_tool_pin(row.get("tool_names")),
        transcript_revision=int(row.get("transcript_revision") or 0),
        runtime_revision=int(row.get("runtime_revision") or 0),
        runtime_generation=int(row.get("runtime_generation") or 0),
        lineage_origin_id=origin_id,
        model_config_corrupt=is_corrupt_model_config(row.get("model_config")),
        model_config_json=row.get("model_config"),
        origin_json=row.get("origin_json"),
        tool_names=row.get("tool_names"),
        prompt_cache_clock=row.get("prompt_cache_clock"),
    )


def _row_to_record(row: Mapping[str, Any]) -> Mapping[str, Any]:
    """The stored row with the prompt joins resolved and the fork markers lifted (the SQLite
    ``_session_row_dict`` projection)."""
    data = dict(row)
    for column in ("system_prompt", "tool_names"):
        if f"_{column}_resolved" in data:
            resolved = data.pop(f"_{column}_resolved")
            if column in data:
                data[column] = resolved
    if not (data.get("_reset_from") and data.get("_branched_from")):
        raw = data.get("model_config")
        cfg = _parse_model_config(raw)
        if isinstance(cfg, dict):
            if not data.get("_reset_from"):
                value = cfg.get("_reset_from")
                if isinstance(value, str) and value.strip():
                    data["_reset_from"] = value.strip()
            if not data.get("_branched_from"):
                value = cfg.get("_branched_from")
                if isinstance(value, str) and value.strip():
                    data["_branched_from"] = value.strip()
    return data


# The SQLite upsert's keep-existing column set (the gateway creates a bare row before the real
# creator carries model/prompt): each COALESCEs the existing value over the incoming one.
_KEEP_EXISTING_COLUMNS = (
    "session_key", "chat_id", "chat_type", "thread_id", "parent_session_id", "cwd", "profile_name",
    "transport_profile", "git_repo_root", "origin_json", "display_name",
)


class PgSessionsRepository:
    """``SessionRepository`` verbs on the bound profile's store schema; unported verbs raise (D8)."""

    def __init__(self, store: PostgresStore) -> None:
        self._store = store

    # ── the slice verbs ──

    def create(self, spec: Any) -> SessionId:
        """Create (upsert) a session record, keeping what an earlier writer set; returns the session id."""
        store = self._store
        sid = spec.session_id
        config = dict(spec.model_config or {})
        if marker := _MARKER_OF_KIND.get(spec.lineage_kind):
            config[marker] = spec.parent_session_id
        with store.tx() as conn:
            # A delayed constructor must not recreate a deleted session beside its tombstone.
            if store.one("SELECT 1 FROM sessions WHERE id = %s", sid) is None and store.one(
                    "SELECT 1 FROM state_meta WHERE key = %s", RETIRED_PREFIX + sid) is not None:
                raise RetiredSessionError(sid, "not_found")
            profile_name = spec.profile_name
            if not (profile_name or "").strip():
                profile_name = store.owner_profile_name
            # A newly inserted /new continuation is visible immediately: a delegate marker riding the
            # incoming config would hide the reset chat from every list (#109073).
            if (spec.session_key and isinstance(config, dict)
                    and config.get("_reset_from") is not None
                    and config.get("_delegate_from") is not None):
                config = {key: value for key, value in config.items() if key != "_delegate_from"}
            model_config_json = json.dumps(config) if config else None
            prompt_hash = self._store_system_prompt(conn, spec.system_prompt)
            existing = store.one("SELECT * FROM sessions WHERE id = %s", sid)
            if existing is None:
                conn.execute(
                    """INSERT INTO sessions (
                       id, source, created_source, user_id, session_key, chat_id, chat_type, thread_id,
                       model, model_config, system_prompt, system_prompt_hash,
                       parent_session_id, cwd, profile_name, transport_profile, git_repo_root,
                       origin_json, display_name, started_at
                   ) VALUES (
                       %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                   )""",
                    (
                        sid, spec.source, spec.source, spec.user_id, spec.session_key, spec.chat_id,
                        spec.chat_type, spec.thread_id, spec.model, model_config_json, spec.system_prompt,
                        prompt_hash, spec.parent_session_id, spec.cwd, profile_name, spec.transport_profile,
                        spec.git_repo_root, json.dumps(dict(spec.origin)) if spec.origin else None,
                        spec.display_name, time.time(),
                    ),
                )
            else:
                # Keep-existing merge (the SQLite upsert's CASEs, computed in Python over the stored row).
                source = spec.source if existing["source"] in (None, "unknown") else existing["source"]
                model = existing["model"] if existing["model"] is not None else spec.model
                if model_config_json is not None:
                    stored_cfg = _parse_model_config(existing["model_config"])
                    reset_from = stored_cfg.get("_reset_from")
                    remainder = {k: v for k, v in stored_cfg.items() if k != "_reset_from"}
                    if reset_from is not None and remainder == {}:
                        merged = json.loads(model_config_json)
                        merged.pop("_delegate_from", None)
                        merged["_reset_from"] = reset_from
                        model_config = json.dumps(merged)
                    else:
                        model_config = existing["model_config"] or model_config_json
                else:
                    model_config = existing["model_config"]
                sp_hash = existing["system_prompt_hash"] if existing["system_prompt_hash"] is not None else prompt_hash
                sp_text = (
                    None
                    if existing["system_prompt_hash"] is None and prompt_hash is not None
                    else existing["system_prompt"]
                )
                merged = {
                    col: (existing[col] if existing[col] is not None else value)
                    for col, value in {
                        "session_key": spec.session_key, "chat_id": spec.chat_id, "chat_type": spec.chat_type,
                        "thread_id": spec.thread_id, "parent_session_id": spec.parent_session_id,
                        "cwd": spec.cwd, "profile_name": profile_name,
                        "transport_profile": spec.transport_profile, "git_repo_root": spec.git_repo_root,
                        "origin_json": json.dumps(dict(spec.origin)) if spec.origin else None,
                        "display_name": spec.display_name,
                    }.items()
                }
                # created_source: a later creator replaces the accounting placeholders (and 'local'
                # never replaces anything); a stamped real source is preserved (#56439).
                unset = (existing["created_source"] or "") in ("", "unknown", "local")
                created_source = existing["created_source"]
                if unset and spec.source and spec.source != "local":
                    created_source = spec.source
                conn.execute(
                    """UPDATE sessions SET
                       source = %s, model = %s, model_config = %s, system_prompt = %s,
                       system_prompt_hash = %s, created_source = %s,
                       session_key = %s, chat_id = %s, chat_type = %s, thread_id = %s,
                       parent_session_id = %s, cwd = %s, profile_name = %s,
                       transport_profile = %s, git_repo_root = %s, origin_json = %s, display_name = %s
                     WHERE id = %s""",
                    (
                        source, model, model_config, sp_text, sp_hash, created_source,
                        merged["session_key"], merged["chat_id"], merged["chat_type"], merged["thread_id"],
                        merged["parent_session_id"], merged["cwd"], merged["profile_name"],
                        merged["transport_profile"], merged["git_repo_root"], merged["origin_json"],
                        merged["display_name"], sid,
                    ),
                )
            if prompt_hash is not None:
                self._delete_unreferenced_system_prompts(conn)
            if spec.parent_session_id:
                self._inherit_parent_session_metadata(conn, sid)
            self._stamp_compression_ancestors(conn, sid)
        return SessionId(sid)

    def get(self, session_id: SessionId) -> Optional[SessionRecord]:
        """The session's record, or ``None``: the row with the prompt-store joins, through the shared
        record projection."""
        store = self._store
        row = store.one(
            """SELECT s.*, COALESCE(sp.prompt, s.system_prompt) AS _system_prompt_resolved,
                      COALESCE(tp.prompt, s.tool_names) AS _tool_names_resolved
               FROM sessions s
               LEFT JOIN system_prompts sp ON sp.hash = s.system_prompt_hash
               LEFT JOIN system_prompts tp ON tp.hash = s.tool_names
               WHERE s.id = %s""",
            session_id,
        )
        return session_record(_row_to_record(row)) if row else None

    # ── create's helpers (the SQLite store functions, ported) ──

    def _store_system_prompt(self, conn, system_prompt: Optional[str]) -> Optional[str]:
        if system_prompt is None:
            return None
        prompt_hash = hashlib.sha256(system_prompt.encode("utf-8")).hexdigest()
        conn.execute(
            "INSERT INTO system_prompts (hash, prompt) VALUES (%s, %s) ON CONFLICT (hash) DO NOTHING",
            (prompt_hash, system_prompt),
        )
        return prompt_hash

    def _delete_unreferenced_system_prompts(self, conn) -> None:
        conn.execute(
            """DELETE FROM system_prompts
               WHERE NOT EXISTS (SELECT 1 FROM sessions WHERE sessions.system_prompt_hash = system_prompts.hash)
                 AND NOT EXISTS (SELECT 1 FROM sessions WHERE sessions.tool_names = system_prompts.hash)"""
        )

    def _inherit_parent_session_metadata(self, conn, session_id: str) -> None:
        """Backfill a child's NULL cwd/git/profile from its parent; for compression forks (excluding
        fork edges), the gateway routing columns too (#64709, #59527)."""
        for col in ("cwd", "git_repo_root", "git_branch"):
            conn.execute(
                f"""UPDATE sessions SET {col} = COALESCE(sessions.{col},
                        (SELECT p.{col} FROM sessions p
                          WHERE p.id = sessions.parent_session_id))
                    WHERE id = %s AND parent_session_id IS NOT NULL""",
                (session_id,),
            )
        # Same-namespace profile backfill (the SQLite same-key check, expressed without the SQLite
        # instr() helper: position() mirrors it, 0 when the namespace separator is absent).
        conn.execute(
            """UPDATE sessions SET profile_name = COALESCE(sessions.profile_name,
                        (SELECT p.profile_name FROM sessions p
                          WHERE p.id = sessions.parent_session_id
                            AND (p.session_key IS NULL OR sessions.session_key IS NULL
                                 OR substr(p.session_key, 1, position(':' in substr(p.session_key, 7)) + 6)
                                      = substr(sessions.session_key, 1, position(':' in substr(sessions.session_key, 7)) + 6))))
               WHERE id = %s AND parent_session_id IS NOT NULL""",
            (session_id,),
        )
        # Gateway routing inheritance, compression forks only, skipping fork edges (a child whose
        # marker names its parent is its own conversation, not a crash-recovery continuation).
        child = conn.execute(
            "SELECT model_config, parent_session_id FROM sessions WHERE id = %s", (session_id,)
        ).fetchone()
        if child is None or child["parent_session_id"] is None:
            return
        cfg = _parse_model_config(child["model_config"])
        markers = (cfg.get("_branched_from"), cfg.get("_delegate_from"), cfg.get("_reset_from"))
        parent_id = child["parent_session_id"]
        if any(str(marker or "") == str(parent_id) for marker in markers):
            return
        parent = conn.execute(
            "SELECT end_reason FROM sessions WHERE id = %s", (parent_id,)
        ).fetchone()
        if parent is None or parent["end_reason"] != "compression":
            return
        for col in ("user_id", "session_key", "chat_id", "chat_type", "thread_id",
                    "display_name", "origin_json", "transport_profile"):
            conn.execute(
                f"""UPDATE sessions SET {col} = (SELECT p.{col} FROM sessions p
                                                 WHERE p.id = %s)
                    WHERE id = %s AND sessions.{col} IS NULL""",
                (parent_id, session_id),
            )

    def _stamp_compression_ancestors(self, conn, session_id: str) -> None:
        """Copy the tip's ``created_source`` onto compression ancestors that have none (the SQLite
        ``stamp_compression_ancestors``, same chain walk)."""
        row = conn.execute(
            "SELECT parent_session_id, created_source FROM sessions WHERE id = %s", (session_id,)
        ).fetchone()
        if row is None:
            return
        surface = row["created_source"]
        if surface in (None, "", "unknown", "local"):
            return
        current, parent_id = session_id, row["parent_session_id"]
        for _ in range(1000):
            if not parent_id:
                return
            parent = conn.execute(
                "SELECT parent_session_id, created_source FROM sessions WHERE id = %s", (parent_id,)
            ).fetchone()
            if parent is None or parent["created_source"] not in (None, "", "unknown", "local", surface):
                return
            # The chain step: the parent's preferred continuation must be ``current``.
            step = conn.execute(
                """SELECT child.id FROM sessions parent
                   JOIN sessions child ON child.parent_session_id = parent.id
                   WHERE parent.id = %s AND parent.end_reason = 'compression'
                     AND COALESCE(child.source, '') != 'tool'
                   ORDER BY child.id LIMIT 1""",
                (parent_id,),
            ).fetchone()
            if step is None or step["id"] != current:
                return
            if parent["created_source"] != surface:
                conn.execute("UPDATE sessions SET created_source = %s WHERE id = %s", (surface, parent_id))
            current, parent_id = parent_id, parent["parent_session_id"]


add_stubs(PgSessionsRepository, SessionRepository, implemented=("create", "get"))


def _register() -> None:
    register_repository("sessions", lambda store: PgSessionsRepository(store))


_register()

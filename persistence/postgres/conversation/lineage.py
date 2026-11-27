"""``LineageRepository`` on PostgreSQL (stage 1): the compression chain walks, the compression write side and the
durable compressor counters, the SQLite store's lineage SQL ported inline over the store's connection.

The read side walks compression edges through the shared chain-step query (JSON markers parsed with
``json_valid``/``json_extract_path``, never throwing); the write side publishes a compression child in one
transaction (closure, child row, handoff commit, ancestor stamping, local-target advance) with the same
preconditions and refusals as the SQLite store. The pure-Python walk and identity helpers are imported from
the SQLite tree; the SQL is this module's own. The message-row shape reuses the stage-0 repository's bind
values, so a compression handoff row is byte-identical to an ``append``-ed one.
"""

from __future__ import annotations

import json
import time
from typing import Mapping, Optional, Sequence

from persistence import CompressionPublishRefused, Conflict, MessageId, SessionId
from persistence.conversation.lineage import CompressionChild, LineageRepository
from persistence.conversation.messages import ConversationMessage, NewMessage, RowStamp
from persistence.conversation.sessions import SessionRecord
from persistence.core.errors import AuthorityRefusal, PermanentStoreError
from persistence.sqlite._state.common import _ended_by_compression, is_automatic_end_reason
from persistence.sqlite.conversation._store.compression import (
    _CHAIN_CAP, _UNSET_CREATED_SOURCES, _walk_compression_chain)
from persistence.sqlite.conversation._store.local import POLICY_PREFIX
from persistence.sqlite.conversation._store.local_migration import LEGACY_PREFIX
from persistence.sqlite.conversation._store.messages_sql import (
    _coerce_timestamp, _parse_tool_calls, _tool_calls_count)
from persistence.sqlite.conversation._store.rules import (
    DB_ROW_SNAPSHOT, _is_checkpoint_item, _newest_checkpoint_carrier, coerce_epoch, stamp_message_uid)
from persistence.sqlite.conversation._store.transcript_repair import transcript_row_snapshot
from persistence.sqlite.conversation.messages import _message_dict, _stamp
from session_rules.surfaces import created_surface
from psycopg import errors as pg_errors

from .._backend import PostgresStore, register_repository, translated
from .._lineage_common import (
    chain_step_sql, json_marker_absent_sql, json_marker_sql, lineage_cte_sql,
    legacy_reset_child_sql, reset_child_sql, session_lineage_root_to_tip,
)
from .._stubs import add_stubs
from .messages import (
    PgMessageRepository, _payload_bytes, _payload_str, _row_to_message_dict, _scrub,
)
from .sessions import _parse_model_config

__all__ = ["PgLineageRepository"]

_CHAIN_STEP_SQL = chain_step_sql()

# Unset-ancestor precheck for the stamp walk (the SQLite ``_UNSET_ANCESTOR_SQL`` shape): through stamped
# (surface) ancestors, is there an unset compression ancestor to repair?
_UNSET_ANCESTOR_SQL = """
    WITH RECURSIVE up(pid, n) AS (
        SELECT parent_session_id, 1 FROM sessions WHERE id = %s
        UNION ALL
        SELECT s.parent_session_id, up.n + 1 FROM up JOIN sessions s ON s.id = up.pid
        WHERE s.end_reason = 'compression' AND s.created_source = %s AND up.n < %s
    )
    SELECT 1 FROM up JOIN sessions s ON s.id = up.pid
    WHERE s.end_reason = 'compression' AND COALESCE(s.created_source, '') IN ('', 'unknown', 'local')
    LIMIT 1
"""

# A stamped child whose parent still carries no ``created_source`` (the SQLite ``_UNSTAMPED_EDGE_CHILD_SQL``
# shape; the non-continuation filter keeps the edge a genuine continuation).
_UNSTAMPED_EDGE_CHILD_SQL = (
    "SELECT child.id FROM sessions parent JOIN sessions child ON child.parent_session_id = parent.id\n"
    "WHERE parent.end_reason = 'compression' AND COALESCE(parent.created_source, '') IN ('', 'unknown', 'local')\n"
    "  AND NOT (COALESCE(child.created_source, '') IN ('', 'unknown', 'local'))\n"
    "  AND COALESCE(" + json_marker_sql("child.model_config", "_branched_from") + ", '') != parent.id\n"
    "  AND COALESCE(" + json_marker_sql("child.model_config", "_delegate_from") + ", '') != parent.id\n"
    "  AND COALESCE(" + json_marker_sql("child.model_config", "_reset_from") + ", '') != parent.id\n"
    "  AND COALESCE(child.source, '') != 'tool'\n"
)

# A non-continuation child of ``parent_session_id = %s`` (branch/delegate/reset/tool) - the reopen
# predicate, the parent id bound for each marker comparison (the SQLite ``_NON_CONTINUATION_CHILD_FILTER_SQL``).
_NON_CONTINUATION_CHILD_FILTER_SQL = (
    "  AND COALESCE(" + json_marker_sql("model_config", "_branched_from") + ", '') != %s\n"
    "  AND COALESCE(" + json_marker_sql("model_config", "_delegate_from") + ", '') != %s\n"
    "  AND COALESCE(" + json_marker_sql("model_config", "_reset_from") + ", '') != %s\n"
    "  AND COALESCE(source, '') != 'tool'\n"
)

# The stage-0 repository's insert (same shape, so handoff rows match ``append`` rows): the ``active``
# column is the literal 1, 26 bound values.
_INSERT_MESSAGE_SQL = """INSERT INTO messages (session_id, role, content, tool_call_id,
                        tool_calls, tool_name, effect_disposition, timestamp, token_count, finish_reason,
                        reasoning, reasoning_content, reasoning_details, codex_reasoning_items,
                        codex_message_items, platform_message_id, observed, _compressed_summary, active,
                        api_content, display_kind, display_metadata, display_identity, message_uid,
                        absorbed_message_uids, tool_call_uids, tool_call_uid)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 1,
                           %s, %s, %s, %s, %s, %s, %s, %s)
                   RETURNING id"""

_SHADOWED_CHECKPOINT_ROWS_SQL = (
    "SELECT id, codex_reasoning_items FROM messages WHERE session_id = %s AND active = 1 "
    "AND role = 'assistant' AND id < %s AND codex_reasoning_items LIKE '%%\"compaction\"%%'"
)
_SET_CODEX_REASONING_SQL = "UPDATE messages SET codex_reasoning_items = %s WHERE id = %s"

# The child session row's carried model-config keys (the SQLite ``_CONVERSATION_MODEL_CONFIG_KEYS``).
_CONVERSATION_MODEL_CONFIG_KEYS = ("onboarding_run", "prompt_refresh_epoch", "_chat_adopted")


def _id_chunks(ids, size: int = 900):
    ids = list(ids)
    for start in range(0, len(ids), size):
        yield ids[start:start + size]


def _json_or(raw, default):
    """A stored JSON text column: the parsed value, *default* when NULL or malformed (the store's
    ``_json_or`` for the checkpoint items; readers never throw on a malformed row)."""
    if raw is None:
        return default
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        return default


class PgLineageRepository:
    """Lineage walks and the compression write side on the bound profile's store schema; unported
    verbs raise naming themselves (D8)."""

    def __init__(self, store: PostgresStore) -> None:
        self._store = store
        # The stage-0 repository carries the message-row bind shape; a handoff row stays byte-identical
        # to an ``append``-ed one.
        self._messages = PgMessageRepository(store)

    # ── shared walks ────────────────────────────────────────────────────────

    def _session_row(self, session_id: str) -> Optional[dict[str, object]]:
        if not session_id:
            return None
        return self._store.one("SELECT * FROM sessions WHERE id = %s", session_id)

    def _is_compression_child_row(self, child: Mapping[str, object]) -> bool:
        """A compression continuation (not a branch/delegate/reset/tool child) of a compression-ended
        parent (the store's ``_is_compression_child_row``)."""
        from .messages import _is_explicit_fork_child_row
        parent_id = child.get("parent_session_id")
        if not parent_id or _is_explicit_fork_child_row(dict(child), include_reset=True):
            return False
        parent = self._session_row(str(parent_id))
        return bool(parent and parent.get("end_reason") == "compression")

    def _compression_chain(self, session_id: str) -> list[str]:
        """Root-first ids through the compression tip (the store's ``get_compression_chain``)."""
        store = self._store

        def step(current: str) -> Optional[dict[str, object]]:
            return store.one(_CHAIN_STEP_SQL, current)

        return _walk_compression_chain(session_id, step)

    def _compression_tip(self, session_id: str) -> str:
        chain = self._compression_chain(session_id)
        return chain[-1] if chain else session_id

    def _resume_lineage_ids(self, session_id: str) -> list[str]:
        """The verified compression lineage root -> session (the store's ``_resume_lineage_ids``): the
        upward-only walk; a fork child's lineage is itself alone."""
        if not session_id:
            return [session_id]
        from .messages import _is_explicit_fork_child_row
        session = self._session_row(session_id)
        if not session or _is_explicit_fork_child_row(dict(session)):
            return [session_id]
        chain = [session_id]
        current = session
        seen = {session_id}
        while len(chain) < _CHAIN_CAP:
            if not self._is_compression_child_row(current):
                break
            parent = self._session_row(str(current["parent_session_id"])) if current.get("parent_session_id") else None
            if not parent or parent["id"] in seen:
                break
            seen.add(parent["id"])
            chain.append(parent["id"])
            current = parent
        return list(reversed(chain))

    # ── writes ──────────────────────────────────────────────────────────────

    def publish_compression_child(
        self, parent_session_id: SessionId, child: CompressionChild, handoff: Sequence[NewMessage], *,
        compression_lock_holder: Optional[str] = None, require_compression_lease: bool = True,
        require_lease_refresh: bool = False, lease_ttl_seconds: float = 300.0, watermark: Optional[MessageId] = None,
        watermark_ceiling: Optional[MessageId] = None,
    ) -> tuple[RowStamp, ...]:
        store = self._store
        messages = [_message_dict(message) for message in handoff]
        with translated():
            try:
                with store.tx() as conn:
                    self._publish_compression_child_on_conn(
                        conn, parent_session_id=str(parent_session_id), child_session_id=str(child.session_id),
                        source=child.source, messages=messages, model=child.model,
                        model_config=dict(child.model_config) if child.model_config is not None else None,
                        system_prompt=child.system_prompt, cwd=child.cwd, profile_name=child.profile_name,
                        compression_lock_holder=compression_lock_holder,
                        require_compression_lease=require_compression_lease,
                        require_lease_refresh=require_lease_refresh, lease_ttl_seconds=lease_ttl_seconds,
                        watermark=int(watermark) if watermark is not None else None,
                        watermark_ceiling=int(watermark_ceiling) if watermark_ceiling is not None else None)
            except pg_errors.UniqueViolation as exc:
                raise Conflict("session", str(child.session_id)) from exc
            except PermanentStoreError as exc:
                # A taken child id fails the insert; the transaction's translation turns the raw unique
                # violation into a permanent store error before this handler runs, so rewrap just that
                # case as the contract's conflict.
                if isinstance(exc.__cause__, pg_errors.UniqueViolation):
                    raise Conflict("session", str(child.session_id)) from exc
                raise
            except RuntimeError as exc:
                # The preconditions are plain RuntimeErrors; the typed refusals subclass it.
                if type(exc) is not RuntimeError:
                    raise
                raise CompressionPublishRefused(parent_session_id, str(exc)) from exc
        return tuple(_stamp(message, {}) for message in messages)

    def _publish_compression_child_on_conn(
        self, conn, *, parent_session_id: str, child_session_id: str, source: str, messages: list[dict[str, object]],
        model: Optional[str], model_config: Optional[dict], system_prompt: Optional[str], cwd: Optional[str],
        profile_name: Optional[str], compression_lock_holder: Optional[str], require_compression_lease: bool,
        require_lease_refresh: bool, lease_ttl_seconds: float, watermark: Optional[int], watermark_ceiling: Optional[int],
    ) -> None:
        """One publish on the caller's write transaction (the store's ``_publish_compression_child_on_conn``)."""
        if require_lease_refresh and compression_lock_holder:
            conn.execute(
                "UPDATE compression_locks SET expires_at = %s WHERE session_id = %s AND holder = %s",
                (time.time() + lease_ttl_seconds, parent_session_id, compression_lock_holder))
        lock_row = conn.execute(
            "SELECT holder, expires_at FROM compression_locks WHERE session_id = %s", (parent_session_id,)
        ).fetchone()
        if require_compression_lease and (
            lock_row is None or not compression_lock_holder
            or lock_row["holder"] != compression_lock_holder
            or float(lock_row["expires_at"]) <= time.time()
        ):
            from persistence.sqlite._state.errors import CompressionSessionBusyError
            raise CompressionSessionBusyError(
                f"Compression lease lost before publication: {parent_session_id}")
        parent = conn.execute(
            """SELECT ended_at, end_reason, cwd, git_branch, git_repo_root,
                      user_id, session_key, chat_id, chat_type,
                      thread_id, display_name, origin_json, profile_name, tool_names,
                      archived, auto_archived, pinned, model_config, created_source
               FROM sessions WHERE id = %s""",
            (parent_session_id,),
        ).fetchone()
        if parent is None:
            raise RuntimeError(f"Compression parent not found: {parent_session_id}")
        if parent["ended_at"] is not None:
            # An AUTOMATIC end stamp is stale by construction (the lease holder is still continuing);
            # deliberate boundaries fail closed.
            if not is_automatic_end_reason(parent["end_reason"]):
                raise RuntimeError(f"Compression parent already ended: {parent_session_id}")
            conn.execute(
                "UPDATE sessions SET ended_at = NULL, end_reason = NULL WHERE id = %s",
                (parent_session_id,))
        if not messages:
            raise RuntimeError("Compression child handoff must not be empty")
        self._publish_child_session_row(
            conn, parent, parent_session_id=parent_session_id, child_session_id=child_session_id, source=source,
            model=model, model_config=model_config, system_prompt=system_prompt, cwd=cwd, profile_name=profile_name)
        # Carried handoff tail rows arrive without a timestamp and would otherwise be stamped now,
        # breaking their display identity with the parent's durable originals.
        self._carry_parent_timestamps(conn, parent_session_id, messages)
        total_messages, total_tool_calls = self._insert_message_rows(conn, child_session_id, messages)
        if watermark is not None:
            # Clone the parent's concurrent tail into the child after the handoff; the originals stay in
            # the closed parent for lineage recovery.
            bounded = watermark_ceiling is not None
            sql = ("SELECT id, tool_calls FROM messages WHERE session_id = %s AND active = 1 AND id > %s"
                   + (" AND id <= %s" if bounded else "") + " ORDER BY id")
            params = [parent_session_id, int(watermark)] + ([int(watermark_ceiling)] if bounded else [])
            tail_ids, tail_tool_calls = self._tail_rows_after_watermark(conn, sql, params)
            if tail_ids:
                self._clone_message_rows(conn, tail_ids, session_id=child_session_id)
                total_messages += len(tail_ids)
                total_tool_calls += tail_tool_calls
        conn.execute(
            "UPDATE sessions SET message_count = %s, tool_call_count = %s WHERE id = %s",
            (total_messages, total_tool_calls, child_session_id))
        updated = conn.execute(
            "UPDATE sessions SET ended_at = %s, end_reason = 'compression' "
            "WHERE id = %s AND ended_at IS NULL",
            (time.time(), parent_session_id))
        if updated.rowcount != 1:
            raise RuntimeError(f"Compression parent changed during publication: {parent_session_id}")
        # A child on a real surface stamps the chain's unset ancestors, so compressing a conversation
        # whose root predates write-time stamping lists it again under its surface.
        self._stamp_compression_ancestors(conn, child_session_id)
        self._advance_local_target(conn, parent_session_id, child_session_id)
        if parent["archived"]:
            # A live continuation under an idle-sweep archive re-activates the chat; after the closure
            # above the child is linked into the lineage walk.
            self._unarchive_auto_archived_lineage(conn, child_session_id)

    def _publish_child_session_row(self, conn, parent, *, parent_session_id: str, child_session_id: str,
                                   source: str, model: Optional[str], model_config: Optional[dict],
                                   system_prompt: Optional[str], cwd: Optional[str],
                                   profile_name: Optional[str]) -> None:
        """INSERT the compression child's ``sessions`` row copied from the parent (the store's
        ``_publish_child_session_row``): a continuation keeps the conversation keys, the pin and the
        archive state; a branch/reset child is its own conversation."""
        continuation = not any(k in (model_config or {}) for k in ("_branched_from", "_reset_from"))
        carried = {key: value for key, value in _parse_model_config(parent["model_config"]).items()
                   if continuation and key in _CONVERSATION_MODEL_CONFIG_KEYS}
        config = {**carried, **(model_config or {})}
        prompt_hash = self._store_system_prompt(conn, system_prompt)
        # Copy a real creating surface: automatic compression passes the parent's live label; ``local``
        # and ``unknown`` are not surfaces and stay NULL (the turn's usage write fills the NULL).
        created_source = created_surface(parent["created_source"]) or created_surface(source)
        conn.execute(
            """INSERT INTO sessions (
                   id, source, created_source, model, model_config, system_prompt,
                   system_prompt_hash, tool_names,
                   parent_session_id, cwd, git_branch, git_repo_root,
                   profile_name, user_id, session_key, chat_id, chat_type,
                   thread_id, display_name, origin_json, pinned, started_at,
                   archived, auto_archived
               ) VALUES (%s, %s, %s, %s, %s, NULL, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
            (
                child_session_id, source, created_source, model, json.dumps(config) if config else None,
                prompt_hash, parent["tool_names"], parent_session_id,
                cwd or parent["cwd"], parent["git_branch"], parent["git_repo_root"],
                profile_name or parent["profile_name"] or self._store.owner_profile_name,
                parent["user_id"], parent["session_key"], parent["chat_id"], parent["chat_type"],
                parent["thread_id"], parent["display_name"], parent["origin_json"],
                int(parent["pinned"] or 0) if continuation else 0, time.time(),
                (parent["archived"] or 0) if continuation else 0,
                (parent["auto_archived"] or 0) if continuation else 0),
        )

    def _store_system_prompt(self, conn, system_prompt: Optional[str]) -> Optional[str]:
        """The prompt store's upsert on ``conn`` (the holder's ``_store_system_prompt``)."""
        import hashlib
        if system_prompt is None:
            return None
        prompt_hash = hashlib.sha256(system_prompt.encode("utf-8")).hexdigest()
        conn.execute(
            "INSERT INTO system_prompts (hash, prompt) VALUES (%s, %s) ON CONFLICT (hash) DO NOTHING",
            (prompt_hash, system_prompt))
        return prompt_hash

    def _carry_parent_timestamps(self, conn, parent_session_id: str, messages: list[dict[str, object]]) -> None:
        """Adopt the parent's durable timestamps onto carried handoff rows (the store's
        ``_carry_parent_timestamps``; best-effort, a failed carry keeps the publish-time stamps)."""
        try:
            candidates = [i for i, msg in enumerate(messages)
                          if isinstance(msg, dict) and coerce_epoch(msg.get("timestamp"), field="message timestamp") is None]
            if not candidates:
                return
            donors: dict[tuple, list] = {}
            for row in conn.execute(
                    "SELECT role, content, timestamp, tool_call_id, tool_calls, tool_name FROM messages "
                    "WHERE session_id = %s AND active = 1 ORDER BY id", (parent_session_id,)).fetchall():
                key = (row["role"], _payload_str(row["content"]), row["tool_call_id"], row["tool_calls"],
                       row["tool_name"])
                donors.setdefault(key, []).append(row["timestamp"])
            for i in candidates:
                msg = messages[i]
                tool_calls = _parse_tool_calls(msg.get("tool_calls"))
                key = (msg.get("role", "unknown"), _encode_content_for_key(msg),
                       msg.get("tool_call_id"), json.dumps(tool_calls) if tool_calls else None,
                       _scrub(msg.get("tool_name")))
                queue = donors.get(key)
                if queue:
                    msg["timestamp"] = queue.pop(0)
        except Exception:  # health: a failed carry keeps the publish-time stamps
            return

    def _insert_message_rows(self, conn, session_id: str, messages: list[dict[str, object]],
                             prune_checkpoints: bool = True) -> tuple[int, int]:
        """Insert *messages* as fresh active rows in the caller's txn -> ``(inserted, tool_calls)``
        (the store's ``_insert_message_rows``; never touches the session counters)."""
        now_ts = time.time()
        inserted = tool_calls_total = 0
        batch_tool_index: dict[str, str] = {}
        for msg in messages:
            role = str(msg.get("role", "unknown"))
            tool_calls = _parse_tool_calls(msg.get("tool_calls"))
            message_timestamp = _coerce_timestamp(msg.get("timestamp"), now_ts)
            stamp_message_uid(msg)
            self._messages._stamp_tool_call_uids(msg, tool_calls, batch_tool_index)
            params = list(self._messages._message_row_params(session_id, role, msg, tool_calls, message_timestamp))
            if role != "assistant":
                # keep_reasoning=False: a non-assistant handoff row never carries reasoning content.
                for i in (10, 11, 12, 13, 14):
                    params[i] = None
            cur = conn.execute(_INSERT_MESSAGE_SQL, tuple(params))
            msg["timestamp"] = message_timestamp
            if (row := cur.fetchone()) is not None:
                msg["_row_id"] = int(row["id"])
            inserted += 1
            tool_calls_total += _tool_calls_count(tool_calls)
            now_ts = max(now_ts, message_timestamp) + 1e-6
        if prune_checkpoints:
            self._prune_shadowed_checkpoints(conn, session_id, messages)
        # Every inserter gets the new rows' stored-row digest, so a later re-flush takes the versioned
        # rewrite path instead of the legacy one.
        self._stamp_inserted_row_snapshots(conn, session_id, messages)
        return inserted, tool_calls_total

    def _prune_shadowed_checkpoints(self, conn, session_id: str, live_messages: list[dict[str, object]]) -> None:
        """Keep only the newest checkpoint among *live_messages* (the store's ``_prune_shadowed_checkpoints``)."""
        carrier = _newest_checkpoint_carrier(live_messages, "codex_reasoning_items")
        if carrier >= 0 and isinstance(live_messages[carrier].get("_row_id"), int):
            self._drop_shadowed_checkpoint_rows(conn, session_id, int(live_messages[carrier]["_row_id"]))

    def _drop_shadowed_checkpoint_rows(self, conn, session_id: str, carrier_row_id: int) -> None:
        for row in conn.execute(_SHADOWED_CHECKPOINT_ROWS_SQL, (session_id, carrier_row_id)).fetchall():
            row_id, raw = row["id"], row["codex_reasoning_items"]
            items = _json_or(raw, None)
            if not isinstance(items, list) or not any(_is_checkpoint_item(item) for item in items):
                continue
            kept = [item for item in items if not _is_checkpoint_item(item)]
            conn.execute(_SET_CODEX_REASONING_SQL, (json.dumps(kept), row_id))

    def _stamp_inserted_row_snapshots(self, conn, session_id: str, messages: list[dict[str, object]]) -> None:
        """The stored-row digests of the rows just inserted (the store's ``stamp_inserted_row_snapshots``)."""
        by_id: dict[int, dict[str, object]] = {}
        for msg in messages:
            if isinstance(msg, dict) and isinstance(msg.get("_row_id"), int):
                by_id[msg["_row_id"]] = msg
        if not by_id:
            return
        ph = ", ".join(["%s"] * len(by_id))
        for row in conn.execute(
                f"SELECT * FROM messages WHERE session_id = %s AND id IN ({ph}) ORDER BY id",
                (session_id, *by_id)).fetchall():
            for column in ("content", "api_content", "reasoning_content"):
                row[column] = _payload_str(row.get(column))
            by_id[int(row["id"])][DB_ROW_SNAPSHOT] = transcript_row_snapshot(row)

    def _tail_rows_after_watermark(self, conn, sql: str, params: list) -> tuple[list[int], int]:
        """The parent's tail rows after the watermark, as ids + their tool-call count (the store's
        ``_tail_rows_after_watermark``)."""
        rows = conn.execute(sql, tuple(params)).fetchall()
        tail_ids = [int(r["id"]) for r in rows]
        return tail_ids, sum(_tool_calls_count(_parse_tool_calls(r["tool_calls"])) for r in rows)

    def _clone_message_rows(self, conn, tail_ids: list[int], *, session_id: Optional[str] = None) -> None:
        """Pure-SQL clone of *tail_ids* into *session_id* as fresh live rows (the store's
        ``_clone_message_rows``: payload columns byte-exact, new row ids, ``active = 1``)."""
        if not tail_ids:
            return
        ph = ", ".join(["%s"] * len(tail_ids))
        conn.execute(
            f"""INSERT INTO messages (session_id, role, content, tool_call_id, tool_calls, tool_name,
                        effect_disposition, timestamp, token_count, finish_reason,
                        reasoning, reasoning_content, reasoning_details, codex_reasoning_items,
                        codex_message_items, platform_message_id, observed, _compressed_summary,
                        active, compacted, api_content, display_kind, display_metadata,
                        message_uid, absorbed_message_uids, tool_call_uids, tool_call_uid)
               SELECT %s, role, content, tool_call_id, tool_calls, tool_name,
                       effect_disposition, timestamp, token_count, finish_reason,
                       reasoning, reasoning_content, reasoning_details, codex_reasoning_items,
                       codex_message_items, platform_message_id, observed, _compressed_summary,
                       1, 0, api_content, display_kind, display_metadata,
                       message_uid, absorbed_message_uids, tool_call_uids, tool_call_uid
                 FROM messages WHERE id IN ({ph}) ORDER BY id""",
            (session_id, *tail_ids))

    def _stamp_compression_ancestors(self, conn, session_id: str) -> int:
        """Copy *session_id*'s ``created_source`` onto compression ancestors that have none, through
        each hop's preferred edge (the store's ``stamp_compression_ancestors``); returns stamped."""
        row = conn.execute("SELECT parent_session_id, created_source FROM sessions WHERE id = %s",
                           (session_id,)).fetchone()
        if row is None:
            return 0
        surface = row["created_source"]
        if surface in _UNSET_CREATED_SOURCES:
            return 0
        if conn.execute(_UNSET_ANCESTOR_SQL, (session_id, surface, _CHAIN_CAP)).fetchone() is None:
            return 0
        current, parent_id = session_id, row["parent_session_id"]
        stamped = 0
        for _ in range(_CHAIN_CAP):
            parent = (conn.execute("SELECT parent_session_id, created_source FROM sessions WHERE id = %s",
                                  (parent_id,)).fetchone() if parent_id else None)
            if not parent or parent["created_source"] not in (*_UNSET_CREATED_SOURCES, surface):
                return stamped
            step = conn.execute(_CHAIN_STEP_SQL, (parent_id,)).fetchone()
            if step is None or step["id"] != current:
                return stamped
            if parent["created_source"] != surface:
                conn.execute("UPDATE sessions SET created_source = %s WHERE id = %s", (surface, parent_id))
                stamped += 1
            current, parent_id = parent_id, parent["parent_session_id"]
        return stamped

    def _advance_local_target(self, conn, parent_session_id: str, child_session_id: str) -> None:
        """Advance the local creation receipt's target to the published child (the store's
        ``advance_local_target`` over the store's own receipt; a store without the receipt, or a
        receipt that no longer names this parent, is left alone and never blocks the publish)."""
        parent = conn.execute("SELECT * FROM sessions WHERE id = %s", (parent_session_id,)).fetchone()
        logical_id = parent["chat_id"] if parent else None
        if not logical_id:
            return
        key = POLICY_PREFIX + logical_id
        raw = conn.execute("SELECT value FROM state_meta WHERE key = %s", (key,)).fetchone()
        if raw is None:
            legacy = conn.execute("SELECT 1 FROM state_meta WHERE key = %s",
                                  (LEGACY_PREFIX + logical_id,)).fetchone()
            if logical_id.startswith("local-") or legacy:
                raise AuthorityRefusal("storage_unavailable")
            return
        try:
            receipt = json.loads(raw["value"])
            if (receipt["session_id"] != logical_id or receipt["entry"]["session_id"] != parent_session_id
                    or receipt["principal_id"] != parent["user_id"] or receipt["route"] != parent["session_key"]):
                raise AuthorityRefusal("storage_unavailable")
            routed = self._routed_entry(conn, receipt["route"], parent_session_id)
            if routed is None:
                raise AuthorityRefusal("admission_conflict")
            candidate = {**routed, "session_id": child_session_id}
            receipt["entry"] = candidate
            receipt["lineage"] = [*receipt.get("lineage", [logical_id]), child_session_id]
            self._validate_local_lineage(conn, receipt)
        except (KeyError, TypeError, ValueError) as exc:
            if isinstance(exc, AuthorityRefusal):
                raise
            raise AuthorityRefusal("storage_unavailable") from exc
        self._commit_local_target(conn, receipt, candidate)

    def _routed_entry(self, conn, route: str, tip: str) -> Optional[dict[str, object]]:
        """*route*'s routing entry while it still targets *tip*, else None (the store's ``_routed_entry``)."""
        row = conn.execute("SELECT entry_json FROM gateway_routing WHERE scope = '' AND session_key = %s",
                           (route,)).fetchone()
        routed = json.loads(row["entry_json"]) if row else None
        return routed if (isinstance(routed, dict) and routed.get("session_id") == tip) else None

    def _validate_local_lineage(self, conn, receipt: dict) -> None:
        """The receipt's lineage is a valid root-to-tip walk (the store's ``validate_local_lineage``)."""
        from persistence.authority.bindings import local_session_id
        from persistence.sqlite.conversation._store.local_migration import LEGACY_PREFIX
        if ("legacy_session_id" not in receipt
                and receipt["session_id"] != local_session_id(receipt["profile_id"], receipt["principal_id"],
                                                              receipt["request_id"])):
            raise AuthorityRefusal("storage_unavailable")
        # The PG-dialect ``legacy_lineage_root``: a migrated receipt's root is the legacy id, proven by
        # its durable migration marker.
        root = receipt.get("legacy_session_id", receipt["session_id"])
        if "legacy_session_id" in receipt:
            if root != receipt["session_id"]:
                raise AuthorityRefusal("storage_unavailable")
            row = conn.execute("SELECT value FROM state_meta WHERE key = %s", (LEGACY_PREFIX + root,)).fetchone()
            expected = {k: receipt[k] for k in ("session_id", "profile_id", "principal_id", "legacy_session_id")}
            if row is None or json.loads(row["value"]) != expected:
                raise AuthorityRefusal("storage_unavailable")
        lineage = receipt.get("lineage", [root])
        if (not isinstance(lineage, list) or not lineage or len(set(lineage)) != len(lineage)
                or lineage[0] != root or lineage[-1] != receipt["entry"]["session_id"]):
            raise AuthorityRefusal("storage_unavailable")
        for parent_id, child_id in zip(lineage, lineage[1:]):
            parent = conn.execute("SELECT end_reason FROM sessions WHERE id = %s", (parent_id,)).fetchone()
            child = conn.execute("SELECT parent_session_id, model_config FROM sessions WHERE id = %s",
                                (child_id,)).fetchone()
            if parent is None or child is None or child["parent_session_id"] != parent_id:
                raise AuthorityRefusal("storage_unavailable")
            if parent["end_reason"] == "session_reset":
                valid = _parse_model_config(child["model_config"]).get("_reset_from") == parent_id
            else:
                tip = conn.execute(_CHAIN_STEP_SQL, (parent_id,)).fetchone()
                valid = (parent["end_reason"] == "compression" and tip is not None and tip["id"] == child_id)
            if not valid:
                raise AuthorityRefusal("storage_unavailable")
        if conn.execute(_CHAIN_STEP_SQL, (lineage[-1],)).fetchone() is not None:
            raise AuthorityRefusal("storage_unavailable")

    def _commit_local_target(self, conn, receipt: dict, routed: dict) -> None:
        """Store the receipt and its route's entry, bumping the logical revision, in one transaction
        (the store's ``_commit_local_target``)."""
        conn.execute("UPDATE gateway_routing SET entry_json = %s WHERE scope = '' AND session_key = %s",
                     (json.dumps(routed, sort_keys=True, separators=(",", ":")), receipt["route"]))
        conn.execute("UPDATE state_meta SET value = %s WHERE key = %s",
                     (json.dumps(receipt, sort_keys=True, separators=(",", ":")), POLICY_PREFIX + receipt["session_id"]))
        conn.execute("UPDATE sessions SET runtime_revision = runtime_revision + 1 WHERE id = %s",
                     (receipt["session_id"],))

    def _unarchive_auto_archived_lineage(self, conn, session_id: str) -> bool:
        """Un-hide a lineage the idle sweep archived (the store's ``_unarchive_auto_archived_lineage``)."""
        cte = lineage_cte_sql()
        params = (session_id, session_id)
        manual = conn.execute(
            cte + """
                SELECT 1 FROM sessions
                WHERE id IN (SELECT id FROM lineage) AND archived <> 0 AND COALESCE(auto_archived, 0) = 0
                LIMIT 1
                """, params).fetchone()
        if manual is not None:
            return False
        return conn.execute(
            cte + """
                UPDATE sessions SET archived = 0, auto_archived = 0
                WHERE id IN (SELECT id FROM lineage) AND archived <> 0
                """, params).rowcount > 0

    def reopen_orphaned_compression(self, session_id: SessionId) -> bool:
        """Reopen a compression parent only when no continuation was published (the store's
        ``reopen_orphaned_compression_session``)."""
        store = self._store
        sid = str(session_id)
        if not sid:
            return False
        with translated():
            with store.tx() as conn:
                row = conn.execute("SELECT ended_at, end_reason FROM sessions WHERE id = %s", (sid,)).fetchone()
                if not _ended_by_compression(row):
                    return False
                # Any non-branch/non-delegate/non-reset/non-tool child is a continuation, ended or not.
                child = conn.execute(
                    "SELECT 1 FROM sessions WHERE parent_session_id = %s"
                    + _NON_CONTINUATION_CHILD_FILTER_SQL + " LIMIT 1",
                    (sid,) * 4,
                ).fetchone()
                if child is not None:
                    return False
                now = time.time()
                lock_row = conn.execute(
                    "SELECT holder, expires_at FROM compression_locks WHERE session_id = %s", (sid,)).fetchone()
                if lock_row is not None:
                    expires_at = lock_row["expires_at"]
                    if expires_at is None or float(expires_at) >= now:
                        return False
                    deleted = conn.execute(
                        "DELETE FROM compression_locks WHERE session_id = %s AND holder = %s AND expires_at = %s",
                        (sid, lock_row["holder"], expires_at))
                    if deleted.rowcount != 1:
                        return False
                updated = conn.execute(
                    "UPDATE sessions SET ended_at = NULL, end_reason = NULL "
                    "WHERE id = %s AND ended_at IS NOT NULL AND end_reason = 'compression'",
                    (sid,))
                return updated.rowcount == 1

    def finalize_orphaned_compression_children(self) -> int:
        """Mark orphaned compression continuations as ``orphaned_compression`` (the store's
        ``finalize_orphaned_compression_sessions``)."""
        store = self._store
        cutoff = time.time() - 604800  # 7 days
        with translated():
            with store.tx() as conn:
                cur = conn.execute(
                    """
                    UPDATE sessions
                    SET ended_at = %s,
                        end_reason = 'orphaned_compression'
                    WHERE api_call_count = 0
                      AND end_reason IS NULL
                      AND ended_at IS NULL
                      AND started_at < %s
                      AND parent_session_id IS NOT NULL
                      AND EXISTS (
                          SELECT 1 FROM sessions p
                          WHERE p.id = sessions.parent_session_id
                            AND p.end_reason = 'compression'
                            AND p.ended_at IS NOT NULL
                      )
                      AND EXISTS (
                          SELECT 1 FROM messages m
                          WHERE m.session_id = sessions.id
                      )
                    """,
                    (time.time(), cutoff))
                return cur.rowcount or 0

    def stamp_legacy_compression_chains(self) -> int:
        """Stamp the unset ancestors of pre-stamping chains from their stamped segments (the store's
        ``stamp_legacy_compression_chains``); one write, idempotent."""
        store = self._store
        with translated():
            with store.tx() as conn:
                total = 0
                for row in conn.execute(_UNSTAMPED_EDGE_CHILD_SQL).fetchall():
                    total += self._stamp_compression_ancestors(conn, row["id"])
                return total

    def set_compression_fallback_streak(self, session_id: SessionId, value: int) -> None:
        self._write_session_column("compression_fallback_streak", session_id, max(0, int(value)))

    def set_compression_ineffective_count(self, session_id: SessionId, value: int) -> None:
        self._write_session_column("compression_ineffective_count", session_id, max(0, int(value)))

    def set_compression_overload_streak(self, session_id: SessionId, value: int) -> None:
        self._write_session_column("compression_overload_streak", session_id, max(0, int(value)))

    def increment_compression_overload_streak(self, session_id: SessionId) -> Optional[int]:
        """Atomically bump the overload streak and return the new value (one ``UPDATE ... RETURNING``)."""
        store = self._store
        sid = str(session_id)
        if not sid:
            return None
        with translated():
            with store.tx() as conn:
                row = conn.execute(
                    "UPDATE sessions SET compression_overload_streak = compression_overload_streak + 1 "
                    "WHERE id = %s RETURNING compression_overload_streak", (sid,)).fetchone()
                return None if row is None else int(row["compression_overload_streak"])

    def set_compression_recovery_deadline(self, session_id: SessionId, deadline: Optional[float]) -> None:
        sid = str(session_id)
        if not sid:
            return
        try:
            normalized = max(0.0, float(deadline or 0.0))
        except (TypeError, ValueError):
            normalized = 0.0
        self._write_session_column("compression_recovery_deadline", sid, normalized or None)

    def _write_session_column(self, column: str, session_id: SessionId, value) -> None:
        store = self._store
        sid = str(session_id)
        if not sid:
            return
        with translated():
            with store.tx() as conn:
                conn.execute(f"UPDATE sessions SET {column} = %s WHERE id = %s", (value, sid))

    # ── reads ───────────────────────────────────────────────────────────────

    def tip(self, session_id: SessionId) -> SessionId:
        return SessionId(self._compression_tip(str(session_id)) or str(session_id))

    def lineage(self, session_id: SessionId) -> tuple[SessionId, ...]:
        """Compression ancestors through the tip, root first (the store's ``get_compression_lineage``)."""
        from .messages import _is_explicit_fork_child_row
        store = self._store
        sid = str(session_id)
        session = self._session_row(sid)
        if not session or _is_explicit_fork_child_row(dict(session)):
            return (SessionId(sid),) if session else ()
        root = session
        ancestors = {root["id"]}
        while self._is_compression_child_row(root):
            parent = self._session_row(str(root["parent_session_id"])) if root.get("parent_session_id") else None
            if not parent or parent["id"] in ancestors:
                break
            root = parent
            ancestors.add(root["id"])
        lineage = [root["id"]]
        seen = {root["id"]}
        current = root
        while current.get("end_reason") == "compression":
            children = store.all(
                "SELECT * FROM sessions WHERE parent_session_id = %s ORDER BY started_at ASC",
                current["id"])
            next_child = next((dict(row) for row in children if self._is_compression_child_row(row)), None)
            if not next_child or next_child["id"] in seen:
                break
            lineage.append(next_child["id"])
            seen.add(next_child["id"])
            current = next_child
        # Later tips are included only when the requested session itself was compacted.
        return tuple(SessionId(x) for x in (lineage if sid in lineage else [sid]))

    def chain(self, session_id: SessionId) -> tuple[SessionId, ...]:
        return tuple(SessionId(x) for x in self._compression_chain(str(session_id)))

    def conversation_root(self, session_id: SessionId) -> SessionId:
        """ROOT id of the lineage (the store's ``get_conversation_root``)."""
        sid = str(session_id)
        chain = session_lineage_root_to_tip(self._store, sid)
        return SessionId(chain[0] if chain and chain[0] else sid)

    def tool_rotation_root(self, session_id: SessionId) -> SessionId:
        """Root of a ``--source tool`` run's compression chain (the store's ``get_tool_rotation_root``)."""
        sid = str(session_id)
        row, seen = self._session_row(sid), {sid}
        while row and row.get("source") == "tool" and (parent_id := row.get("parent_session_id")) and parent_id not in seen:
            parent = self._session_row(str(parent_id)) if parent_id else None
            if not parent or parent.get("source") != "tool" or not self._is_compression_child_row(
                    {**row, "source": None}):
                break
            seen.add(parent_id)
            row = parent
        return SessionId(row["id"] if row else sid)

    def is_fork_child(self, session_id: SessionId) -> bool:
        from .messages import _is_explicit_fork_child_row
        session = self._session_row(str(session_id))
        return bool(session and _is_explicit_fork_child_row(dict(session)))

    def is_compression_child(self, child: SessionRecord) -> bool:
        row = {"source": child.source, "model_config": child.model_config,
               "parent_session_id": child.parent_session_id}
        return self._is_compression_child_row(row)

    def resolve_resume_target(self, session_id: SessionId) -> SessionId:
        """Redirect a resume to the descendant holding the messages (the store's
        ``resolve_resume_session_id``: compression tip first, then the deepest child with messages)."""
        store = self._store
        sid = str(session_id)
        if not sid:
            return SessionId(sid)
        try:
            sid = self._compression_tip(sid) or sid
        except Exception:  # health: a failed tip lookup resumes the id reached
            pass
        current = sid
        seen = {current}
        best = None
        for _ in range(32):
            if store.one("SELECT 1 FROM messages WHERE session_id = %s LIMIT 1", current) is not None:
                best = current
            child = store.one(
                "SELECT id FROM sessions AS child WHERE child.parent_session_id = %s "
                f"AND {json_marker_absent_sql('child.model_config', '_branched_from')} "
                f"AND {json_marker_absent_sql('child.model_config', '_delegate_from')} "
                f"AND {json_marker_absent_sql('child.model_config', '_reset_from')} "
                f"AND NOT ({legacy_reset_child_sql('child')}) "
                "AND COALESCE(child.source, '') != 'tool' "
                "ORDER BY child.started_at DESC, child.id DESC LIMIT 1", current)
            if child is None or not child["id"] or child["id"] in seen:
                break
            current = child["id"]
            seen.add(current)
        return SessionId(best if best is not None else sid)

    def resume_lineage_ids(self, session_id: SessionId) -> tuple[SessionId, ...]:
        return tuple(SessionId(x) for x in self._resume_lineage_ids(str(session_id)))

    def delegated_child_lineage_ids(self, session_id: SessionId) -> tuple[SessionId, ...]:
        """The compression lineage a delegated child's transcript displays (the store's
        ``transcript_lineage_ids``: the resume lineage once the row carries ``_delegate_from``)."""
        sid = str(session_id)
        row = self._session_row(sid)
        cfg = row.get("model_config") if row else None
        if isinstance(cfg, str):
            cfg = _parse_model_config(cfg)
        if not isinstance(cfg, dict) or not cfg.get("_delegate_from"):
            return (SessionId(sid),)
        return self.resume_lineage_ids(sid)

    def delegate_descendant_ids(self, session_id: SessionId) -> tuple[SessionId, ...]:
        """Marked delegate subagents (recursive, with their compression continuations) of the
        session and the compression segments it continues (the store's ``delegate_descendant_ids``)."""
        sid = str(session_id)
        chain = session_lineage_root_to_tip(self._store, sid)
        seeds: list[str] = []
        for s in reversed(chain):
            row = self._store.one("SELECT model_config FROM sessions WHERE id = %s", s)
            config = _parse_model_config(row["model_config"]) if row is not None else {}
            if config.get("_delegate_from"):
                break
            seeds.append(s)
            if config.get("_branched_from"):
                break
        found: list[str] = []
        frontier = seeds
        while frontier:
            children = [c for c in self._collect_delegate_child_ids(frontier) if c not in found]
            segments = [seg for c in children for seg in self._compression_chain(c)]
            frontier = [seg for seg in dict.fromkeys(segments) if seg not in found]
            found.extend(frontier)
        return tuple(SessionId(x) for x in found)

    def _collect_delegate_child_ids(self, parent_ids: list[str]) -> list[str]:
        """Delegate-subagent ids (the ``_delegate_from`` marker, walked recursively) under
        *parent_ids* (the store's ``_collect_delegate_child_ids``, chunked under the PG bind limit)."""
        store = self._store
        marker = json_marker_sql("model_config", "_delegate_from")
        seeds = {sid for sid in parent_ids if sid}
        found: set[str] = set(seeds)
        frontier = list(seeds)
        while frontier:
            next_frontier: list[str] = []
            for chunk in _id_chunks(frontier, 450):  # each id binds twice below
                ph = ", ".join(["%s"] * len(chunk))
                rows = store.all(
                    f"SELECT id FROM sessions WHERE ({marker}) IN ({ph}) "
                    f"OR (parent_session_id IN ({ph}) AND {marker} IS NOT NULL)",
                    *chunk, *chunk)
                for row in rows:
                    if row["id"] not in found:
                        found.add(row["id"])
                        next_frontier.append(row["id"])
            frontier = next_frontier
        return [sid for sid in found if sid not in seeds]

    def current_segments(self, session_ids: Sequence[SessionId]) -> Mapping[SessionId, SessionId]:
        """``{logical id: current physical segment}``: a local receipt's entry, else the compression
        tip of a compression-ended row (the store's ``session_current_segments``)."""
        store = self._store
        ids = [str(sid) for sid in session_ids]
        if not ids:
            return {}
        ph = ", ".join(["%s"] * len(ids))
        rows = store.all(
            f"""SELECT s.id, s.end_reason,
                       (SELECT json_extract_path_text(m.value::json, 'entry', 'session_id') FROM state_meta m
                        WHERE m.key = '{POLICY_PREFIX}' || s.id AND json_text_valid(m.value)) AS entry
                FROM sessions s WHERE s.id IN ({ph})""", *ids)
        out: dict[str, str] = {}
        for row in rows:
            if row["entry"]:
                out[row["id"]] = row["entry"]
            elif row["end_reason"] == "compression":
                out[row["id"]] = self._compression_tip(row["id"])
            else:
                out[row["id"]] = row["id"]
        return {SessionId(k): SessionId(v) for k, v in out.items()}

    def ancestor_display_prefix(self, session_id: SessionId) -> tuple[ConversationMessage, ...]:
        """Ancestor-only display messages of a lineage (the store's ``get_ancestor_display_prefix``):
        the display rows (active plus compaction-archived) of the verified compression lineage
        root -> session, the session's own segment excluded, a key a later segment carries forward
        shown where it lives now."""
        sid = str(session_id)
        ids = self._resume_lineage_ids(sid)
        if len(ids) <= 1:
            return ()
        store = self._store
        # The shared display read (the SQLite ``get_ancestor_display_prefix`` port): one dedupe over all
        # segments' display rows, projected through the conversation reader so the dicts carry the wire
        # surface (a raw row's ``id``/``session_id``/``active`` keys are not replay-projection keys).
        from ._store.reads import _dedupe_display_generations, _fetch_conversation_rows, _rows_to_conversation
        rows = _dedupe_display_generations(
            _fetch_conversation_rows(store, ids, "", with_session_id=True, with_compacted=True),
            drop_removed=True)
        ancestor_ids = {int(row["id"]) for row in rows if row["session_id"] != sid and row["id"] is not None}
        if not ancestor_ids:
            return ()
        lineage = _rows_to_conversation(rows, session_id=sid, include_ancestors=True,
                                        repair_alternation=False, include_row_ids=True)
        return tuple(ConversationMessage.from_openai_message(
                       {k: v for k, v in message.items() if k != "_row_id"})
                     for message in lineage if message.get("_row_id") in ancestor_ids)

    def compression_fallback_streak(self, session_id: SessionId) -> int:
        return self._read_session_number("compression_fallback_streak", session_id, int, 0)

    def compression_ineffective_count(self, session_id: SessionId) -> int:
        return self._read_session_number("compression_ineffective_count", session_id, int, 0)

    def compression_overload_streak(self, session_id: SessionId) -> int:
        return self._read_session_number("compression_overload_streak", session_id, int, 0)

    def compression_recovery_deadline(self, session_id: SessionId) -> float:
        return self._read_session_number("compression_recovery_deadline", session_id, float, 0.0)

    def _read_session_number(self, column: str, session_id: SessionId, cast: type, zero) -> object:
        """One numeric ``sessions`` column clamped at *zero*; a missing session or NULL reads *zero*."""
        sid = str(session_id)
        row = self._store.one(f"SELECT {column} FROM sessions WHERE id = %s", sid) if sid else None
        try:
            return zero if row is None else max(zero, cast(row[column] or zero))
        except (TypeError, ValueError):
            return zero


def _encode_content_for_key(msg: Mapping[str, object]):
    """The handoff row's display content in stored form (the store's ``_carry_parent_timestamps``
    donor key): the ``_encode_content`` of the live content."""
    from .messages import _encode_content
    return _encode_content(msg.get("content"))


add_stubs(PgLineageRepository, LineageRepository, implemented=(
    "publish_compression_child", "reopen_orphaned_compression", "finalize_orphaned_compression_children",
    "stamp_legacy_compression_chains", "set_compression_fallback_streak", "set_compression_ineffective_count",
    "set_compression_overload_streak", "increment_compression_overload_streak", "set_compression_recovery_deadline",
    "tip", "lineage", "chain", "conversation_root", "tool_rotation_root", "is_fork_child",
    "is_compression_child", "resolve_resume_target", "resume_lineage_ids", "delegated_child_lineage_ids",
    "delegate_descendant_ids", "current_segments", "ancestor_display_prefix",
    "compression_fallback_streak", "compression_ineffective_count", "compression_overload_streak",
    "compression_recovery_deadline",
))


def _register() -> None:
    register_repository("lineage", lambda store: PgLineageRepository(store))


_register()

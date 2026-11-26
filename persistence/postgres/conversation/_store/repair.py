"""Transcript batch repair and inserted-row snapshots on PostgreSQL (the SQLite ``transcript_repair``
port): the row-addressed rewrite paths (a durable ``_row_id`` or a logical ``message_uid`` + stored-row
digest) resolve against the store's connection with ``%s`` placeholders; the owned-column digest
(``transcript_row_snapshot``) hashes the row's stored values after the BYTEA payload columns are
decoded, so a rewrite's compare-and-swap version matches what this backend stored."""

from __future__ import annotations

import hashlib
from functools import partial
from typing import Any, Callable, Mapping, Optional

from session_rules.turns import (
    ABSORBED_MESSAGE_UIDS, MESSAGE_UID, TYPED_ASK, _DB_PERSISTED_MARKER, is_content_blank,
    message_uid_or_none,
)
from persistence.sqlite.conversation._store.identity import _fill_missing_tool_call_uids, _restore_row_identity
from persistence.sqlite.conversation._store.rules import CANONICAL_ROW, DB_ROW_SNAPSHOT, RESTORED_FROM_STORE

from .common import (
    _MESSAGE_WRITE_COLUMNS, _decode_payloads, _in_placeholders, _loaded_view_content,
)

_NON_PAYLOAD_COLUMNS = frozenset(
    {"session_id", "role", "timestamp", "active", "display_identity", MESSAGE_UID, "tool_call_uid"})


_REPAIR_COLUMNS = tuple(c for c in _MESSAGE_WRITE_COLUMNS if c not in _NON_PAYLOAD_COLUMNS)


_METADATA_COLUMNS = frozenset(
    {"display_kind", "display_metadata", "api_content", "codex_reasoning_items", "platform_message_id"})


_OWNED_COLUMNS = tuple(c for c in _REPAIR_COLUMNS if c not in _METADATA_COLUMNS)


_REPLAY_COLUMNS = _OWNED_COLUMNS + ("api_content", "codex_reasoning_items")


_LIVE_MISSING_METADATA = ("display_kind", "display_metadata")


_METADATA_ONLY = "_metadata_only"


_CONTENT_ONLY = "_content_only"


def transcript_row_snapshot(row: Mapping[str, Any]) -> str:
    """Fixed-size digest of the owned (non-metadata) columns (the CAS version) of a ``SELECT *`` messages
    row, over the decoded stored values (the SQLite reference hashes the stored strings; the PG
    payload columns decode to the same strings, so the digest stays a stored-version marker)."""
    digest = hashlib.blake2b(digest_size=16)
    for column in _OWNED_COLUMNS:
        value = row[column]
        if value is None:
            digest.update(b"N")
            continue
        if isinstance(value, (bytes, bytearray, memoryview)):
            tag, data = b"B", bytes(value)
        elif isinstance(value, str):
            tag, data = b"S", value.encode("utf-8", "surrogatepass")
        else:
            tag, data = (b"F" if isinstance(value, float) else b"I"), repr(value).encode("ascii")
        digest.update(tag + len(data).to_bytes(8, "big") + data)
    return digest.hexdigest()


def stamp_inserted_row_snapshots(conn, session_id: str, messages: list[dict[str, Any]]) -> None:
    """Stamp the stored-row digest on freshly inserted dicts (every ``_insert_message_rows`` caller;
    one SELECT per 1000-row chunk)."""
    by_id = {msg["_row_id"]: msg for msg in messages if isinstance(msg.get("_row_id"), int)}
    for chunk in _id_chunks(by_id):
        for row in conn.execute(
            f"SELECT * FROM messages WHERE session_id = %s AND id IN ({_in_placeholders(chunk)})",
            (session_id, *chunk)).fetchall():
            by_id[int(row["id"])][DB_ROW_SNAPSHOT] = transcript_row_snapshot(_decode_payloads(dict(row)))


def _active_logical_message_row(conn, session_id: str, role: str, message_uid: Optional[str]):
    """Newest active physical row for one durable logical message (a ``message_uid`` names a logical
    message, not a physical row: compaction/copy paths keep it while re-issuing row ids)."""
    if message_uid is None:
        return None
    return conn.execute(
        "SELECT * FROM messages WHERE session_id = %s AND active = 1 AND role = %s AND message_uid = %s "
        "ORDER BY id DESC LIMIT 1",
        (session_id, role, message_uid),
    ).fetchone()


def _active_uid_dedupe(conn, session_id: str, messages: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """ACTIVE durable rows by uid among *messages*' restored, no-snapshot, no-``_row_id`` rows — the
    re-insert guard (the SQLite ``_active_uid_dedupe`` port)."""
    candidate_uids = sorted({
        uid
        for msg in messages
        if isinstance(msg, dict)
        and msg.get(RESTORED_FROM_STORE) is True
        and not isinstance(msg.get("_row_id"), int)
        and not isinstance(msg.get(DB_ROW_SNAPSHOT), str)
        and (uid := message_uid_or_none(msg))
    })
    durable: dict[str, list[dict[str, Any]]] = {}
    for chunk in _id_chunks(candidate_uids):
        for row in conn.execute(
            f"SELECT * FROM messages WHERE session_id = %s AND active = 1 AND message_uid IN ({_in_placeholders(chunk)})",
            (session_id, *chunk)).fetchall():
            durable.setdefault(row["message_uid"], []).append(row)
    return durable


_RESTORE_MAY_OMIT = (
    ("token_count", "token_count"), ("_compressed_summary", "_compressed_summary"),
    ("tool_call_uids", "tool_call_uids"), ("absorbed_message_uids", ABSORBED_MESSAGE_UIDS))


def _same_as_stored(
        msg: dict[str, Any], rows: list,
        serialize_message_fn: Callable[[dict[str, Any], float], Mapping[str, Any]],
        encode_content_fn: Callable[[Any], Any], decode_content_fn: Callable[[Any], Any],
) -> bool:
    """*msg* serializes to the replayed columns of one of its uid's active *rows*: the re-handed copy,
    not an edit (the row is seen through the restore lens)."""
    for row in rows:
        serialized = dict(serialize_message_fn(msg, float(row["timestamp"])))
        for column, key in _RESTORE_MAY_OMIT:
            if not msg.get(key):
                serialized[column] = row[column]
        if serialized["content"] == encode_content_fn(
            _loaded_view(row["role"], decode_content_fn(row["content"]))
        ):
            serialized["content"] = row["content"]
        if all(row[column] == serialized[column] for column in _REPLAY_COLUMNS):
            return True
    return False


def _loaded_view(role: str, content: Any) -> Any:
    return _loaded_view_content(role, content)


def resolve_and_repair_transcript_batch(
        conn, session_id: str, messages: list[dict[str, Any]],
        encode_content_fn: Callable[[Any], Any], decode_content_fn: Callable[[Any], Any],
        serialize_message_fn: Callable[[dict[str, Any], float], Mapping[str, Any]],
        decode_row_fn: Callable[[Mapping[str, Any]], dict[str, Any]],
) -> list[dict[str, Any]]:
    """Resolve row-addressed rewrites without appending duplicates or replacing concurrent winners
    (the SQLite ``resolve_and_repair_transcript_batch`` port: same CAS semantics over the store's
    connection)."""
    # Late import: agent.session_persistence imports this module at load time.
    from persistence.sqlite.conversation._store.rules import _PERSIST_AFTER_ADMISSION_INTERRUPT

    inserted_rows: list[dict[str, Any]] = []
    durable_uids = _active_uid_dedupe(conn, session_id, messages)
    for msg in messages:
        existing_row_id = msg.get("_row_id") if isinstance(msg, dict) else None
        role = msg.get("role", "unknown") if isinstance(msg, dict) else "unknown"
        expected = msg.get(DB_ROW_SNAPSHOT) if isinstance(msg, dict) else None
        target_row = None
        if isinstance(existing_row_id, int):
            target_row = _active_message_row(conn, session_id, existing_row_id, role)
        elif isinstance(expected, str):
            target_row = _active_logical_message_row(conn, session_id, role, message_uid_or_none(msg))
        if target_row is None:
            if (
                isinstance(msg, dict)
                and (stored := durable_uids.get(message_uid_or_none(msg) or ""))
                and not msg.get(_PERSIST_AFTER_ADMISSION_INTERRUPT)
                and _same_as_stored(msg, stored, serialize_message_fn, encode_content_fn, decode_content_fn)
            ):
                msg[_DB_PERSISTED_MARKER] = True
                continue
            inserted_rows.append(msg)
            continue

        target_id = int(target_row["id"])
        msg["_row_id"] = target_id
        _fill_missing_tool_call_uids(target_row, msg)  # a dict that lost its uids must not rewrite them away
        canonical = None
        adopt = wrote = hand_back = False
        if isinstance(expected, str):
            adopt = transcript_row_snapshot(_decode_payloads(dict(target_row))) != expected
            if not adopt:
                serialized = serialize_message_fn(msg, float(target_row["timestamp"]))
                if serialized["token_count"] is None:
                    serialized = {**serialized, "token_count": target_row["token_count"]}
                if any(target_row[column] != serialized[column] for column in _OWNED_COLUMNS):
                    _rewrite_row(conn, session_id, target_row, serialized)
                    wrote = True
                hand_back = True
        elif role == "assistant" and is_content_blank(decode_content_fn(target_row["content"])):
            # Legacy dict (no digest) over a blank assistant row: the interrupted-stream content repair.
            written = conn.execute(
                "UPDATE messages SET content = %s WHERE id = %s AND session_id = %s AND content IS NULL",
                (encode_content_fn(msg.get("content")), target_id, session_id)) \
                if target_row["content"] is None else \
                conn.execute(
                    "UPDATE messages SET content = %s WHERE id = %s AND session_id = %s AND content IS NOT DISTINCT FROM %s",
                    (encode_content_fn(msg.get("content")), target_id, session_id, target_row["content"]))
            wrote = written.rowcount > 0
        else:
            # Legacy dict over a non-blank assistant row: adopt its content only, never the whole row.
            if role == "assistant":
                canonical = {"content": decode_content_fn(target_row["content"]), _CONTENT_ONLY: True}

        final_row = conn.execute(
            "SELECT * FROM messages WHERE id = %s AND session_id = %s", (target_id, session_id)
        ).fetchone() if wrote else target_row
        final_row = _decode_payloads(dict(final_row))
        if hand_back:  # metadata as written: the rewrite may have removed a stamp the content outlived
            missing = {c: final_row[c] for c in _LIVE_MISSING_METADATA if msg.get(c) is None}
            if any(value is not None for value in missing.values()):
                decoded = decode_row_fn(final_row)
                canonical = {c: decoded[c] for c in missing if c in decoded}
                canonical[_METADATA_ONLY] = True
        msg["timestamp"] = final_row["timestamp"]
        # A row-addressed rewrite keeps the row's identity: the stored uid wins over whatever the live
        # dict carried (a restored dict without one, or a dict stamped before a rolled-back insert).
        _restore_row_identity(final_row, msg)
        msg[DB_ROW_SNAPSHOT] = transcript_row_snapshot(final_row)
        if adopt:
            canonical = decode_row_fn(final_row)
        if canonical:
            msg[CANONICAL_ROW] = canonical
        else:
            msg.pop(CANONICAL_ROW, None)
    return inserted_rows


def _rewrite_row(conn, session_id: str, target_row: Mapping[str, Any], serialized: Mapping[str, Any]) -> None:
    """Rewrite one durable payload whose digest matched the live dict's last committed version (the
    SQLite ``_rewrite_row`` port: owned columns + the typed-ask stamp's content-outlived removal, the
    display index restored by the caller's verb)."""
    old_identity = target_row["display_identity"]
    old_peer_ids = [
        int(row["id"])
        for row in conn.execute(
            "SELECT id FROM messages WHERE session_id = %s AND id != %s "
            "AND (active = 1 OR compacted = 1) AND display_identity = %s",
            (session_id, target_row["id"], old_identity),
        ).fetchall()
    ] if old_identity is not None else []

    # Owned columns only: same-process metadata writes (a reaction, a backfill) stay as committed.
    # The typed-ask stamp names the content it was taken from: new content that no longer carries it
    # (a fold dropped it) removes it, or a reload would read the old ask for the merged row.
    assignments = ", ".join(f"{column} = %s" for column in _OWNED_COLUMNS)
    conn.execute(
        f"UPDATE messages SET {assignments}, display_metadata = "
        f"CASE WHEN display_metadata IS NOT NULL AND (display_metadata::jsonb -> '{TYPED_ASK}') IS NOT NULL "
        f"AND content IS DISTINCT FROM %s AND (COALESCE(%s, '{{}}')::jsonb -> '{TYPED_ASK}') IS NULL "
        f"THEN (display_metadata::jsonb - '{TYPED_ASK}')::text ELSE display_metadata END "
        f"WHERE id = %s AND session_id = %s",
        [*(serialized[column] for column in _OWNED_COLUMNS), serialized["content"],
         serialized["display_metadata"], int(target_row["id"]), session_id],
    )
    _restore_display_index(conn, session_id, target_row, serialized, old_identity, old_peer_ids)


def _restore_display_index(conn, session_id, target_row, serialized, old_identity, old_peer_ids) -> None:
    """Restore display identities/orders invalidated by the payload-update trigger (the SQLite
    ``_restore_display_index`` port, bytea identity parameters)."""
    if old_peer_ids:
        old_order = min(old_peer_ids)
        conn.execute(
            f"UPDATE messages SET display_identity = %s, display_order = %s "
            f"WHERE session_id = %s AND id IN ({_in_placeholders(old_peer_ids)})",
            (old_identity, old_order, session_id, *old_peer_ids),
        )

    target_id = int(target_row["id"])
    new_identity = serialized["display_identity"]
    visible = bool(target_row["active"] or target_row["compacted"])
    if not visible:
        conn.execute(
            "UPDATE messages SET display_identity = %s, display_order = %s WHERE id = %s AND session_id = %s",
            (new_identity, target_id, target_id, session_id))
        return

    peers = conn.execute(
        "SELECT id, display_order FROM messages WHERE session_id = %s AND id != %s "
        "AND (active = 1 OR compacted = 1) AND display_identity = %s",
        (session_id, target_id, new_identity),
    ).fetchall()
    new_order = min(
        [target_id]
        + [int(peer["display_order"] if peer["display_order"] is not None else peer["id"]) for peer in peers]
    )
    conn.execute(
        "UPDATE messages SET display_identity = %s, display_order = %s WHERE id = %s AND session_id = %s",
        (new_identity, new_order, target_id, session_id))
    for peer in peers:
        conn.execute("UPDATE messages SET display_order = %s WHERE id = %s AND session_id = %s",
                     (new_order, int(peer["id"]), session_id))


def _active_message_row(conn, session_id: str, row_id: int, role: str):
    """The same-role active clone for ``row_id``, or the addressed inactive row when no clone exists
    (the SQLite ``_active_message_row`` port: payload predicates over the BYTEA columns use ``IS NOT DISTINCT FROM``)."""
    row = conn.execute(
        "SELECT * FROM messages WHERE id = %s AND session_id = %s", (row_id, session_id)
    ).fetchone()
    if row is None or row["role"] != role:
        return None
    if int(row["active"] or 0) == 1:
        return row
    if row["display_identity"] is None:
        return row
    predicates = []
    params: list = []
    for column in _REPAIR_COLUMNS:
        value = row[column]
        if value is None:
            predicates.append(f"{column} IS NULL")
        elif isinstance(value, (bytes, bytearray, memoryview)):
            predicates.append(f"{column} IS NOT DISTINCT FROM %s")
            params.append(bytes(value))
        else:
            predicates.append(f"{column} IS NOT DISTINCT FROM %s")
            params.append(value)
    payload_predicates = " AND ".join(predicates)
    clones = conn.execute(
        "SELECT * FROM messages WHERE session_id = %s AND active = 1 AND role = %s "
        "AND display_identity = %s AND timestamp = %s AND id != %s AND "
        f"{payload_predicates} ORDER BY id DESC LIMIT 2",
        (session_id, role, row["display_identity"], row["timestamp"], row["id"], *params),
    ).fetchall()
    return clones[0] if len(clones) == 1 else row


def _id_chunks(ids, size: int = 1000):
    """Yield *ids* (any iterable) as lists of at most *size* elements."""
    ids = list(ids)
    for start in range(0, len(ids), size):
        yield ids[start:start + size]

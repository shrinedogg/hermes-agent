"""The hosted-room work-record capture on PostgreSQL (stage 4): the bounded room-scoped evidence of
driver tasks and remote-run receipts, recorded inside the transaction that changed them, never
executable driver state or authority (the reference's ``_hosted_work`` ported to the PG dialect).

The storage tables come from the declarative DDL (no step, no legacy rebuild, no triggers: the
reference's budget guard runs as the explicit ``_budget`` check, its immutability and replace guards
as the explicit ``save_locked`` checks, and its authority-transition guard as the DDL's
``hosted_room_work_authority_transition`` trigger). ``capture_transition_locked`` swallows every
work-record failure under its savepoint; ``AuthorityHistoryError`` from the lineage read and
``HostedRoomError`` from the roster check escape and roll the caller's transaction back.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import psycopg

from persistence.gateway.hosted_rooms import AuthorityHistoryError, DriverStateError
from session_rules.hosted_rooms import (
    INVALID_TABLE, LIMITATIONS, MAX_AUTHORITY_SPANS, MAX_BYTES, MAX_RECEIPTS, MAX_STORE_BYTES, MAX_STORE_ROWS,
    MAX_TASKS, PENDING_TABLE, SOURCE_TABLE, TARGET_TABLE, _RECEIPT_FIELDS, _TASK_FIELDS, AuthoritySpan,
    InvalidStoredWorkRecord, WorkRecordCapacityError, WorkRecordError, WorkRecordPrefixError, _gateway,
    _validate_roster, at_sequence, digest, encode, retained_fields, roster_digest, scope, validate,
    validate_history, validate_provenance, validate_stored,
)
from session_rules.hosted_tasks import normalize_task_payload

from ._hosted_common import byte_len

__all__ = ["capture_locked", "capture_transition_locked"]

# Main's logger name for work-record warnings.
logger = logging.getLogger("gateway.hosted_room_work_records")

# The kinds the authority history reads, inlined for the PG dialect (the reference's ``IN {TRANSFER_KINDS}``).
_TRANSFER_KINDS_SQL = ", ".join(f"'{kind}'" for kind in ("authority.claimed", "authority.lost"))


# ── storage ───────────────────────────────────────────────────────────────────


def row_size_sql(table, prefix=""):
    """Reserve the longest bounded disposition up front so freezing/cleanup can still run at capacity;
    every original variable column is charged as bytes (the SQLite ``length(CAST ... AS BLOB)`` twin)."""
    return "20+" + "+".join(f"COALESCE({byte_len(f'{prefix}{k}')},0)" for k in retained_fields(table))


def usage_sql(tables):
    return (" + ".join(f"(SELECT COALESCE(SUM({row_size_sql(t)}),0) FROM {t})" for t in tables),
            " + ".join(f"(SELECT COUNT(*) FROM {t})" for t in tables))


def initialize(conn) -> None:
    """No-op on PostgreSQL: the work-record tables are in the declarative DDL (no legacy rebuild, no
    triggers to install; the guards run as the explicit checks below and the DDL's authority trigger)."""


def scope_disposition(conn, table, row: dict) -> str:
    """Read-only authority projection, also used when inserting migrated rows."""
    if row["disposition"] != "current":
        return row["disposition"]
    current = conn.execute(
        "SELECT authority_gateway_id, authority_epoch FROM hosted_rooms WHERE room_id = %s",
        (row["room_id"],)).fetchone()
    if current is None or (current["authority_gateway_id"], current["authority_epoch"]) == (
            row["producer_gateway_id"], row["producer_epoch"]):
        return "current"
    return "superseded_authority" if table == PENDING_TABLE and row["status"] != "acked" else "historical"


def validate_stored_locked(conn, table, row: dict) -> dict:
    """Validate the consumed row; invalidate only its unchanged current version.

    The savepoint joins the caller's write transaction. Historical rows are never rewritten.
    """
    try:
        return validate_stored(row)
    except InvalidStoredWorkRecord:
        if row["disposition"] == "current":
            # Match all examined values, not merely a key that may since have been replaced. IS preserves
            # the NULL/BLOB distinctions (the PG twin: IS NOT DISTINCT FROM).
            fields = list(row.keys())
            where = " AND ".join(f'"{k}" IS NOT DISTINCT FROM %s' for k in fields)
            conn.execute("SAVEPOINT work_record_invalidate")
            try:
                conn.execute(
                    f"UPDATE {table} SET disposition = 'invalid' WHERE {where}",
                    tuple(row[k] for k in fields))
            except BaseException:
                conn.execute("ROLLBACK TO work_record_invalidate")
                conn.execute("RELEASE work_record_invalidate")
                raise
            conn.execute("RELEASE work_record_invalidate")
        raise


# Upserts the current record of its scope after the budget check; a delivery row starts ``pending``. The
# reference's immutability and replace triggers are the explicit checks here: a non-current row is
# refused both before the upsert and in the upsert's own WHERE, and only its unchanged fields move.
def save_locked(conn, table, record, *, target_install_id=None, route_generation=None):
    data = encode(record)
    fields = ["room_id", "producer_gateway_id", "producer_epoch", "revision", "digest", "record_json", "disposition"]
    values = [*scope(record), record["revision"], record["digest"], data, "current"]
    keys = fields[:3]
    if target_install_id is not None:
        fields += ["target_install_id", "route_generation", "status"]
        values += [target_install_id, route_generation, "pending"]
        keys += ["target_install_id"]
    _budget(conn, table, dict(zip(fields, values)))
    proposed = dict(zip(fields, values))
    key_where = " AND ".join(f"{k} = %s" for k in keys)
    existing = conn.execute(
        f"SELECT disposition FROM {table} WHERE {key_where}",
        tuple(proposed[k] for k in keys)).fetchone()
    if existing is not None and existing["disposition"] != "current":
        raise WorkRecordError("historical work record is immutable")
    updates = ",".join(f"{k} = EXCLUDED.{k}" for k in fields if k not in keys)
    conn.execute(
        f"INSERT INTO {table} ({','.join(fields)}) VALUES ({','.join('%s' for _ in fields)}) "
        f"ON CONFLICT ({','.join(keys)}) DO UPDATE SET {updates} WHERE {table}.disposition = 'current'",
        values)


# ── authority history ─────────────────────────────────────────────────────────


# Committed authority transfers, each the first event of its span. ``authority.claimed`` is the new
# owner's record; ``authority.lost`` is the record the outgoing owner commits when it is fenced to a
# proven newer epoch. Both carry the same previous/next/epoch payload and are checked alike, so a store
# whose log holds its own fence followed by a later claim reads as one unbroken chain.
TRANSFER_KINDS = ("authority.claimed", "authority.lost")


def read_history_locked(conn, room_id: str, *, gateway_id: str, epoch: int) -> list[dict[str, Any]] | None:
    """Read committed authority transfers within the caller's room-state transaction.

    Legacy adoption is left on its existing path; it is not retroactively certified as a history of
    physical-host transfers.
    """
    rows = conn.execute(
        f"""SELECT seq, authority_epoch, actor_json, payload_json FROM hosted_room_events
            WHERE room_id = %s AND kind IN ({_TRANSFER_KINDS_SQL}) ORDER BY seq LIMIT %s""",
        (room_id, MAX_AUTHORITY_SPANS)).fetchall()
    if len(rows) >= MAX_AUTHORITY_SPANS:
        raise AuthorityHistoryError("authority history exceeds its bound")
    if not rows:
        if epoch == 1:
            return None
        raise AuthorityHistoryError("current authority has no committed lineage")
    first = json.loads(rows[0]["payload_json"])
    if isinstance(first, dict) and first.get("previous_gateway_id") == "legacy":
        return None
    if not isinstance(first, dict) or "previous_gateway_id" not in first:
        raise AuthorityHistoryError("authority history has no previous owner")
    history = [AuthoritySpan(_gateway(first["previous_gateway_id"]), 1, 0)]
    for row in rows:
        actor, payload = json.loads(row["actor_json"]), json.loads(row["payload_json"])
        if actor != {"kind": "system", "id": "authority-control"} or not isinstance(payload, dict):
            raise AuthorityHistoryError("authority claim provenance is invalid")
        expected = history[-1].epoch + 1
        if (payload.get("previous_gateway_id") != history[-1].gateway_id
                or type(payload.get("authority_epoch")) is not int or payload["authority_epoch"] != expected
                or row["authority_epoch"] != expected):
            raise AuthorityHistoryError("authority claim does not extend its predecessor")
        history.append(AuthoritySpan(_gateway(payload.get("authority_gateway_id")), expected, int(row["seq"])))
    result = [span.as_mapping() for span in history]
    validate_history(result, gateway_id=gateway_id, epoch=epoch)
    return result


# ── lineage ───────────────────────────────────────────────────────────────────


# Verifies the room log up to ``seq`` against the authority history; returns the spans and the history digest.
def source_prefix_locked(conn, room_id, authority, seq):
    history = read_history_locked(conn, room_id, gateway_id=authority["gateway_id"], epoch=authority["epoch"])
    if history is None:
        raise WorkRecordPrefixError("successor work evidence has no verified lineage")
    spans = validate_history(history, gateway_id=authority["gateway_id"], epoch=authority["epoch"])
    count = 0
    events = conn.execute(
        "SELECT seq, authority_epoch, actor_json, kind FROM hosted_room_events "
        "WHERE room_id = %s AND seq <= %s ORDER BY seq", (room_id, seq)).fetchall()
    for event in events:
        count += 1
        span = at_sequence(spans, count)
        actor = json.loads(event["actor_json"])
        boundary = span.epoch > 1 and count == span.from_seq
        if (event["seq"] != count or event["authority_epoch"] != span.epoch
                or (actor.get("kind") == "gateway" and actor.get("id") != span.gateway_id)
                or (event["kind"] in TRANSFER_KINDS) != boundary):
            raise WorkRecordPrefixError("work record history prefix is not verified")
    if count != seq:
        raise WorkRecordPrefixError("work record history prefix is unavailable")
    return spans, digest(history)


# ── capture ───────────────────────────────────────────────────────────────────


# Refuses a write that would push the work-record tables past their byte or row budget.
def _budget(conn, table, proposed: dict) -> None:
    total_sql, count_sql = usage_sql((SOURCE_TABLE, PENDING_TABLE, INVALID_TABLE))
    row = conn.execute(f"SELECT {total_sql} AS total, {count_sql} AS count").fetchone()
    total, count = row["total"], row["count"]
    keys = ["room_id", "producer_gateway_id", "producer_epoch"]
    if table == PENDING_TABLE:
        keys.append("target_install_id")
    where = " AND ".join(f"{k} = %s" for k in keys)
    old_row = conn.execute(
        f"SELECT {row_size_sql(table)} AS size FROM {table} WHERE {where}",
        tuple(proposed[k] for k in keys)).fetchone()
    old = old_row["size"] if old_row else 0
    size_row = conn.execute(
        f"SELECT {row_size_sql(table)} AS size FROM (SELECT "
        + ",".join(f"%s AS {k}" for k in proposed) + ")",
        tuple(proposed.values())).fetchone()
    size = size_row["size"]
    if total - old + size > MAX_STORE_BYTES or count - bool(old_row) + 1 > MAX_STORE_ROWS:
        raise WorkRecordCapacityError("work record storage is full")


# The digest of the room event at ``seq``, or of an empty history when ``seq`` is 0.
def history_anchor(conn, table: str, room_id: str, seq: int) -> str:
    if seq == 0:
        return digest([])
    row = conn.execute(
        f"""SELECT seq, event_id, kind, actor_json, authority_epoch, payload_json, created_at
            FROM {table} WHERE room_id = %s AND seq = %s""", (room_id, seq)).fetchone()
    if row is None:
        raise WorkRecordPrefixError("work record history prefix is unavailable")
    return digest([row["seq"], row["event_id"], row["kind"], json.loads(row["actor_json"]),
                   row["authority_epoch"], json.loads(row["payload_json"]), float(row["created_at"])])


# The room's outstanding tasks and receipts within their bounds, or empty lists and why they are unavailable.
def _capture_tasks(conn, room_id):
    # Both tables are in the declarative DDL, so the published-terminal filter is always live (the
    # reference's table_exists checks read true on a store the DDL created).
    published = """NOT EXISTS (SELECT 1 FROM hosted_room_policy_publications AS p
        WHERE p.room_id = t.room_id AND p.task_id = t.task_id
          AND p.kind IN ('turn.settled', 'turn.failed', 'turn.cancelled'))"""
    task_filter = f" AND (status NOT IN ('settled', 'failed', 'cancelled') OR {published})"
    rows = conn.execute(
        """SELECT task_id, thread_id, turn_id, source_event_seq, payload_json, payload_digest,
            status, execution_generation, cancel_generation, settlement_id, cancel_id
            FROM hosted_room_driver_tasks AS t WHERE room_id = %s""" + task_filter + " ORDER BY task_id LIMIT %s",
        (room_id, MAX_TASKS + 1)).fetchall()
    receipts = [dict(row) for row in conn.execute(
        f"SELECT {','.join(sorted(_RECEIPT_FIELDS))} FROM hosted_room_remote_runs AS t WHERE room_id = %s"
        f" AND {published} ORDER BY task_id, execution_generation, member_id LIMIT %s",
        (room_id, MAX_RECEIPTS + 1))]
    if len(rows) > MAX_TASKS or len(receipts) > MAX_RECEIPTS:
        return [], [], "bounds_exceeded"
    tasks = []
    for row in rows:
        try:
            payload, encoded, payload_digest = normalize_task_payload(json.loads(row["payload_json"]))
        except (ValueError, DriverStateError):
            return [], [], "unsupported_task"
        if payload_digest != row["payload_digest"] or encoded != row["payload_json"] \
                or payload["source_event_seq"] != row["source_event_seq"]:
            return [], [], "unsupported_task"
        tasks.append({**{key: row[key] for key in _TASK_FIELDS - {"payload_sha256", "phase", "member_id", "profile"}},
                      "payload_sha256": payload_digest, "phase": row["status"], "profile": payload["target_profile"],
                      "member_id": payload.get("target_member_id", payload["target_profile"])})
    return tasks, receipts, None


def capture_transition_locked(conn, room_id: str) -> None:
    """Observe committed driver facts, without making evidence an execution gate.

    The caller's transaction owns both facts and evidence. A rejected evidence write rolls back only
    its savepoint; malformed retained bytes are classified in place, never replaced with a fabricated
    current snapshot.
    """
    owner = conn.execute(
        "SELECT authority_gateway_id FROM hosted_rooms WHERE room_id = %s AND disbanded_at IS NULL",
        (room_id,)).fetchone()
    if owner is None:
        return
    conn.execute("SAVEPOINT work_transition_capture")
    try:
        capture_locked(conn, room_id=room_id, local_gateway_id=owner["authority_gateway_id"])
    except InvalidStoredWorkRecord:
        logger.warning("Hosted work evidence invalid for room %s", room_id)
    except (WorkRecordError, psycopg.Error):
        conn.execute("ROLLBACK TO work_transition_capture")
        logger.warning("Hosted work evidence unavailable for room %s", room_id, exc_info=True)
    finally:
        conn.execute("RELEASE work_transition_capture")


def capture_locked(conn, *, room_id, local_gateway_id, through_seq=None):
    """Commit a new revision only when one consistent source view changes."""
    initialize(conn)
    room = conn.execute("SELECT * FROM hosted_rooms WHERE room_id = %s", (room_id,)).fetchone()
    if (room is None or room["authority_gateway_id"] != local_gateway_id or room["disbanded_at"] is not None):
        raise WorkRecordError("work record source is unavailable")
    seq = room["next_seq"] - 1
    if through_seq is not None and seq > through_seq:
        raise WorkRecordPrefixError("history delivery must precede work records")
    tasks, receipts, reason = _capture_tasks(conn, room_id)
    fence = None  # This runtime has no durable peer revocation completion proof.
    stop = conn.execute(
        """SELECT seq, payload_json FROM hosted_room_events
           WHERE room_id = %s AND kind = 'room.stop_requested' ORDER BY seq DESC LIMIT 1""",
        (room_id,)).fetchone()
    content = {
        "version": 1 if room["authority_epoch"] == 1 else 2, "room_id": room_id, "home_install_id": local_gateway_id,
        "authority": {"gateway_id": local_gateway_id, "epoch": room["authority_epoch"]},
        "roster_sha256": roster_digest(json.loads(room["members_json"])),
        "history": {"seq": seq, "event_sha256": history_anchor(conn, "hosted_room_events", room_id, seq)},
        "availability": "unavailable" if reason else "available", "reason": reason,
        "tasks": tasks, "receipts": receipts, "limitations": LIMITATIONS,
        "stop": {"closing": fence is not None, "revocation_complete": fence is not None and fence[0] is not None,
                 "seq": stop["seq"] if stop else 0,
                 "cancel_id": json.loads(stop["payload_json"]).get("cancel_id") if stop else None},
    }
    spans = None
    if content["version"] == 2:
        spans, content["lineage_sha256"] = source_prefix_locked(conn, room_id, content["authority"], seq)
        content["incompleteness"] = ["prior_authority_work_unknown"]
    previous = conn.execute(
        f"SELECT * FROM {SOURCE_TABLE} WHERE room_id = %s AND producer_gateway_id = %s AND producer_epoch = %s",
        scope(content)).fetchone()
    previous_record = validate_stored_locked(conn, SOURCE_TABLE, previous) if previous is not None else None
    revision = previous_record["revision"] + 1 if previous_record else 1
    record = {**content, "revision": revision, "digest": digest(content)}
    if len(encode(record).encode("utf-8")) > MAX_BYTES:
        content.update(tasks=[], receipts=[], availability="unavailable", reason="bounds_exceeded")
        record = {**content, "revision": revision, "digest": digest(content)}
    try:
        validate(record)
        _validate_roster(record, json.loads(room["members_json"]))
        if spans is not None:
            validate_provenance(record, spans)
    except WorkRecordError:
        if reason is not None:
            raise
        content.update(tasks=[], receipts=[], availability="unavailable", reason="unsupported_task")
        record = validate({**content, "revision": revision, "digest": digest(content)})
    if previous_record is not None and previous_record["digest"] == record["digest"]:
        return previous_record
    save_locked(conn, SOURCE_TABLE, record)
    return record

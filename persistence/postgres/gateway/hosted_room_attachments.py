"""``HostedRoomAttachmentRepository`` on PostgreSQL (stage 4): room attachment rows in the placement's
schema and their content-deduplicated blob files beside the placement's store, each write one write
transaction on the placement and each read one read - the reference's SQLite attachment store ported
to the PG dialect.

Blob bytes are files under ``<attachment root>/blobs/`` (``attachment_root`` beside the placement's
registered store location). A blob is written or verified inside the write transaction that records it,
so a committed row always names a verified file; released blobs are unlinked after the commit, and the
orphan sweep renames ``.tmp-`` staging files, ``.prune-`` quarantine files and unreferenced ``blob_*``
files inside its own write and unlinks them after it. The attachment tables come from the declarative
DDL (the reference's first-touch schema step is a no-op; its reconciliation of the commitments the
durable events name and its prune run once per placement per process). On a read-only handle the read
verbs skip that setup and only read: a placement without the tables holds no attachment, a missing
blob reads as on a writable handle, and a commitment its durable event names is never read as expired
(the reconciliation's rule, applied on the read). The quotas are the ``session_rules.hosted_attachments``
constants.
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import stat
import threading
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, TypeVar

from hermes_dir_acl import make_user_dir
from persistence import JsonValue
from persistence.gateway.hosted_rooms import (
    AttachmentConflictError,
    AttachmentData,
    AttachmentError,
    AttachmentIntegrityError,
    AttachmentNotFoundError,
    AttachmentQuotaError,
    EventAttachmentConflictError,
    HostedPlacement,
)
from session_rules.hosted_attachments import (
    DISBANDED_GRACE_SECONDS, MAX_ATTACHMENT_BYTES, MAX_GATEWAY_ATTACHMENT_COUNT, MAX_GATEWAY_BLOB_BYTES,
    MAX_ROOM_ATTACHMENT_BYTES, MAX_ROOM_ATTACHMENT_COUNT, MAX_ROOM_UNCOMMITTED_BYTES, MAX_ROOM_UNCOMMITTED_COUNT,
    UNCOMMITTED_TTL_SECONDS, _BLOB_ID_RE, validate_manifest,
)
from session_rules.hosted_rooms import clock

from .._backend import PostgresStore, register_repository, translated
from ._hosted_common import attachment_root, placement_read, placement_write, table_exists

__all__ = ["PgHostedRoomAttachmentRepository", "retain_message_attachments"]

_T = TypeVar("_T")

# (placement, blob root) identities whose reconciliation and prune already ran in this process.
_STARTED: set[tuple[str, str]] = set()
_STARTED_LOCK = threading.Lock()


# ── blob files ────────────────────────────────────────────────────────────────


def _blob_path(blob_root: Path, blob_id: str) -> Path:
    if _BLOB_ID_RE.fullmatch(blob_id) is None:
        raise AttachmentIntegrityError("stored blob id is invalid")
    return blob_root / blob_id


def _write_blob(target: Path, data: bytes) -> None:
    from utils import atomic_write_bytes
    # ``.tmp-`` staging names are what the orphan sweep reclaims after a crash.
    atomic_write_bytes(target, data, tmp_prefix=".tmp-", mode=0o600, fsync_dir=True)


def _open_blob(blob_root: Path, *, blob_id: str, size: int) -> int:
    """Open the regular blob file and fence its size against the durable row."""
    path = _blob_path(blob_root, blob_id)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise AttachmentIntegrityError("canonical attachment blob is unavailable") from exc
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_size != size:
            raise AttachmentIntegrityError("canonical attachment blob size changed")
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def _read_blob(blob_root: Path, *, blob_id: str, size: int, sha256: str) -> bytes:
    descriptor = _open_blob(blob_root, blob_id=blob_id, size=size)
    try:
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            data = handle.read(MAX_ATTACHMENT_BYTES + 1)
    finally:
        os.close(descriptor)
    if len(data) != size or hashlib.sha256(data).hexdigest() != sha256:
        raise AttachmentIntegrityError("canonical attachment blob failed SHA-256 validation")
    return data


def _read_blob_range(blob_root: Path, *, blob_id: str, size: int, offset: int, length: int) -> bytes:
    """One slice of the blob; the row's SHA-256 is the caller's whole-file digest and the
    receiver verifies the reassembled bytes against it, so no slice is hashed here."""
    descriptor = _open_blob(blob_root, blob_id=blob_id, size=size)
    try:
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            handle.seek(offset)
            data = handle.read(length)
    finally:
        os.close(descriptor)
    if len(data) != min(length, size - offset):
        raise AttachmentIntegrityError("canonical attachment blob size changed")
    return data


# ── row steps ─────────────────────────────────────────────────────────────────


# The metadata dict of an attachment row: its identity, content facts and state, ``event_id`` only once one is bound.
def _metadata(row: Mapping[str, Any], *, idempotent: bool = False) -> dict[str, Any]:
    result = {
        "attachment_id": str(row["attachment_id"]),
        "kind": str(row["kind"]),
        "name": str(row["name"]),
        "size": int(row["size"]),
        "mime": str(row["mime"]),
        "sha256": str(row["sha256"]),
        "state": str(row["state"]),
        "created_at": float(row["created_at"]),
        "idempotent": idempotent,
    }
    if row["event_id"] is not None:
        result["event_id"] = str(row["event_id"])
    return result


# Refuses a viewer read unless the room is live under the caller's authority and neither quarantined nor fenced.
def _require_viewer_room(
    conn,
    *,
    room_id: str,
    authority_gateway_id: str,
    authority_epoch: int,
) -> None:
    if not table_exists(conn, "hosted_rooms"):
        raise AttachmentNotFoundError("Group Chat is unavailable to viewers")
    room = conn.execute(
        """SELECT authority_gateway_id, authority_epoch, disbanded_at
           FROM hosted_rooms WHERE room_id = %s""",
        (room_id,),
    ).fetchone()
    if (
        room is None
        or room["disbanded_at"] is not None
        or room["authority_gateway_id"] != authority_gateway_id
        or room["authority_epoch"] != authority_epoch
    ):
        raise AttachmentNotFoundError("Group Chat viewer authority changed")
    # The optional safety/retirement tables a files-only source carries are not installed by the DDL and read absent.
    for table in ("hosted_room_quarantine", "hosted_room_disband_fences"):
        if table_exists(conn, table) and conn.execute(
            f"SELECT 1 FROM {table} WHERE room_id = %s", (room_id,),
        ).fetchone() is not None:
            raise AttachmentNotFoundError("Group Chat is unavailable to viewers")


# The committed, unexpired row this recipient or viewer may read; a room-visible file must be published in its event.
def _read_committed_row(
    conn,
    *,
    room_id: str,
    attachment_id: str,
    recipient_member_id: str,
    normalized_event: str | None,
    viewer: bool,
    now: float,
    reconciled: bool = True,
) -> Mapping[str, Any]:
    # A read-only handle never creates the tables, so a placement without them holds no attachment.
    row = conn.execute(
        "SELECT * FROM hosted_room_attachments WHERE attachment_id = %s AND room_id = %s",
        (attachment_id, room_id),
    ).fetchone() if table_exists(conn, "hosted_room_attachments") else None
    if row is None or str(row["state"]) != "committed":
        raise AttachmentNotFoundError("attachment is not committed for this room")
    # A read-only handle never reconciles, so it applies the reconciliation's rule here.
    if row["expires_at"] is not None and float(row["expires_at"]) <= now and (reconciled or not _named_by_owner_event(conn, row)):
        raise AttachmentNotFoundError("attachment has expired")
    if normalized_event is not None and str(row["event_id"] or "") != normalized_event:
        raise AttachmentNotFoundError("attachment is not owned by this room event")
    if int(row["viewer_access"] or 0):
        # Room-visible files need a published owner even for member reads.
        # A pre-event commitment is staging, not authority to expose bytes.
        if not table_exists(conn, "hosted_room_events"):
            raise AttachmentNotFoundError("attachment is not published in this room")
        owner = conn.execute(
            """SELECT payload_json FROM hosted_room_events
                   WHERE room_id = %s AND event_id = %s AND kind IN ('message.user', 'message.member')""",
            (room_id, str(row["event_id"] or "")),
        ).fetchone()
        manifest = {
            key: row[key]
            for key in ("attachment_id", "kind", "name", "size", "mime")
        }
        payload = (
            json.loads(str(owner["payload_json"])) if owner is not None else None
        )
        published = (
            payload.get("attachments") if isinstance(payload, Mapping) else None
        )
        if not isinstance(published, list) or manifest not in published:
            raise AttachmentNotFoundError(
                "attachment is not published in this room"
            )
    recipients = json.loads(str(row["recipient_member_ids_json"]))
    if viewer:
        if int(row["viewer_access"] or 0) != 1:
            raise AttachmentNotFoundError(
                "attachment is unavailable to Group Chat viewers"
            )
    elif recipient_member_id not in recipients:
        raise AttachmentNotFoundError("attachment is unavailable to this recipient")
    return row


# The event that owns an attachment, or None when the placement has no room log or no such event.
def _owner_event(
    conn, room_id: str, event_id: str
) -> Mapping[str, Any] | None:
    # The standalone private-transport placement need not have a canonical room log (a placement its
    # DDL never installed reads absent here).
    if not table_exists(conn, "hosted_room_events"):
        return None
    return conn.execute(
        "SELECT payload_json FROM hosted_room_events WHERE room_id = %s AND event_id = %s",
        (room_id, event_id),
    ).fetchone()


# ``_reconcile_room_events``'s rule for one row: its durable event payload names it, so the commitment is retained.
def _named_by_owner_event(conn, row: Mapping[str, Any]) -> bool:
    owner = _owner_event(conn, str(row["room_id"]), str(row["event_id"] or ""))
    if owner is None:
        return False
    try:
        payload = json.loads(str(owner["payload_json"]))
        return str(row["attachment_id"]) in {
            str(item.get("attachment_id") or "") for item in payload.get("attachments", []) if isinstance(item, Mapping)
        }
    # Malformed row: the reconciliation skips it too.
    except (ValueError, TypeError, AttributeError, RecursionError):
        return False


def retain_message_attachments(
    conn,
    *,
    room_id: str,
    event_id: str,
    manifest: Any,
    now: float,
) -> None:
    """Validate and retain exact commitments inside the canonical append transaction."""
    normalized = validate_manifest(manifest)
    if not normalized:
        return
    if normalized != manifest:
        raise AttachmentError("published attachment metadata must be canonical")
    if not table_exists(conn, "hosted_room_attachments"):
        raise EventAttachmentConflictError("message attachments are not committed")
    for entry in normalized:
        row = conn.execute(
            "SELECT * FROM hosted_room_attachments WHERE attachment_id = %s",
            (entry["attachment_id"],),
        ).fetchone()
        if (
            row is None
            or row["room_id"] != room_id
            or row["event_id"] != event_id
            or row["state"] != "committed"
            or int(row["viewer_access"] or 0) != 1
            or (row["expires_at"] is not None and float(row["expires_at"]) <= now)
            or any(row[key] != value for key, value in entry.items())
        ):
            raise EventAttachmentConflictError(
                "attachment commitment changed before event publication"
            )
    # Rollback and expiry use the same SQL write lock. No byte or peer I/O is
    # needed here; import verified the bytes before staging these commitments.
    conn.execute(
        """UPDATE hosted_room_attachments SET expires_at = NULL, updated_at = %s
           WHERE attachment_id = ANY(%s)""",
        (now, [entry["attachment_id"] for entry in normalized]),
    )


class PgHostedRoomAttachmentRepository:
    """Attachment rows in the placement's schema and blob files beside the store."""

    def __init__(self, store: PostgresStore) -> None:
        self._store = store

    def _blob_root(self, placement: HostedPlacement) -> Path:
        return attachment_root(self._store, placement) / "blobs"

    def _read_root(self, placement: HostedPlacement, now: float) -> Path:
        """The blob directory for a read verb: a writable handle runs ``_open`` first; a read-only handle only reads, so it creates, prunes and reconciles nothing."""
        return self._blob_root(placement) if self._store.read_only else self._open(placement, now)

    def _read_rows(self, placement: HostedPlacement, fn: Callable[[Any], _T]) -> _T:
        """``fn`` on a read of the placement; the tables are the declarative DDL, so a read never migrates them."""
        with translated():
            with placement_read(self._store, placement) as conn:
                return fn(conn)

    def _open(self, placement: HostedPlacement, now: float) -> Path:
        """Prepare the private root, run the first-touch reconciliation and prune of this placement once per
        process, and return the blob directory (the attachment tables are the declarative DDL)."""
        root = attachment_root(self._store, placement)
        blob_root = root / "blobs"
        make_user_dir(root, parents=True, exist_ok=True, mode=0o700)
        make_user_dir(blob_root, parents=True, exist_ok=True, mode=0o700)
        for path in (root, blob_root):
            try:
                os.chmod(path, 0o700)
            except OSError:
                pass
        with _STARTED_LOCK:
            key = (placement, str(blob_root))
            if key in _STARTED:
                return blob_root
            self._reconcile_room_events(placement, now)
            self._prune(placement, now)
            _STARTED.add(key)
        return blob_root

    def _reconcile_room_events(self, placement: HostedPlacement, now: float) -> int:
        """Recover only attachment commitments named by the durable event payload."""
        def _txn(conn) -> int:
            changed = 0
            rows = conn.execute(
                """SELECT attachment.attachment_id, attachment.event_id,
                          event.payload_json
                     FROM hosted_room_attachments AS attachment
                     JOIN hosted_room_events AS event
                       ON event.room_id = attachment.room_id
                      AND event.event_id = attachment.event_id
                    WHERE attachment.state = 'committed'
                      AND attachment.expires_at IS NOT NULL"""
            ).fetchall()
            for row in rows:
                try:
                    payload = json.loads(str(row["payload_json"]))
                    ids = {
                        str(item.get("attachment_id") or "")
                        for item in payload.get("attachments", [])
                        if isinstance(item, Mapping)
                    }
                # Malformed row: bad/over-nested JSON, a non-object payload, or non-iterable attachments.
                except (ValueError, TypeError, AttributeError, RecursionError):
                    continue
                if str(row["attachment_id"]) not in ids:
                    continue
                changed += conn.execute(
                    """UPDATE hosted_room_attachments
                          SET expires_at = NULL, updated_at = %s
                        WHERE attachment_id = %s AND state = 'committed'""",
                    (now, str(row["attachment_id"])),
                ).rowcount
            return int(changed)
        with translated():
            with placement_write(self._store, placement) as conn:
                return _txn(conn)

    def _prune(self, placement: HostedPlacement, now: float) -> int:
        """Internal step ``prune_expired_attachments``: mark disbanded rooms' rows, delete expired rows and their last
        blob references, then unlink the released blobs after the commit and sweep orphans."""
        def _txn(conn) -> tuple[int, list[str]]:
            removed_blob_ids: list[str] = []
            removed = 0
            conn.execute(
                """UPDATE hosted_room_attachments
                      SET state = 'disbanded',
                          expires_at = (
                            SELECT room.disbanded_at + %s FROM hosted_rooms AS room
                             WHERE room.room_id = hosted_room_attachments.room_id
                          ),
                          updated_at = %s
                    WHERE state = 'committed' AND EXISTS (
                        SELECT 1 FROM hosted_rooms AS room
                         WHERE room.room_id = hosted_room_attachments.room_id
                           AND room.disbanded_at IS NOT NULL)""",
                (DISBANDED_GRACE_SECONDS, now),
            )
            rows = conn.execute(
                """SELECT attachment_id, blob_id FROM hosted_room_attachments
                    WHERE expires_at IS NOT NULL AND expires_at <= %s""",
                (now,),
            ).fetchall()
            released = Counter(str(row["blob_id"]) for row in rows)
            for row in rows:
                conn.execute(
                    "DELETE FROM hosted_room_attachments WHERE attachment_id = %s",
                    (str(row["attachment_id"]),),
                )
                removed += 1
            for blob_id, count in released.items():
                blob = conn.execute(
                    "SELECT ref_count FROM hosted_room_attachment_blobs WHERE blob_id = %s",
                    (blob_id,),
                ).fetchone()
                if blob is None:
                    continue
                if int(blob["ref_count"]) <= count:
                    removed_blob_ids.append(blob_id)
                else:
                    conn.execute(
                        """UPDATE hosted_room_attachment_blobs
                              SET ref_count = ref_count - %s
                            WHERE blob_id = %s""",
                        (count, blob_id),
                    )
            if removed_blob_ids:
                conn.execute(
                    "DELETE FROM hosted_room_attachment_blobs WHERE blob_id = ANY(%s)",
                    (list(removed_blob_ids),),
                )
            return removed, removed_blob_ids
        with translated():
            with placement_write(self._store, placement) as conn:
                removed, removed_blob_ids = _txn(conn)
        blob_root = self._blob_root(placement)
        for blob_id in removed_blob_ids:
            _blob_path(blob_root, blob_id).unlink(missing_ok=True)
        self._sweep_orphans(placement)
        return removed

    # Moves stray staging, quarantine and unreferenced blob files aside inside a write, then unlinks them.
    def _sweep_orphans(self, placement: HostedPlacement) -> None:
        """Internal step ``sweep_orphan_blobs``."""
        blob_root = self._blob_root(placement)

        # The listing runs inside the write transaction, so a concurrent put's blob and row land
        # wholly before or wholly after it.
        def _txn(conn) -> list[Path]:
            try:
                candidates = tuple(blob_root.iterdir())
            except OSError:
                return []
            quarantined: list[Path] = []
            live = {
                str(row["blob_id"])
                for row in conn.execute(
                    "SELECT blob_id FROM hosted_room_attachment_blobs"
                )
            }
            for index, path in enumerate(candidates):
                if not (
                    path.name.startswith((".tmp-", ".prune-"))
                    or (
                        _BLOB_ID_RE.fullmatch(path.name) is not None
                        and path.name not in live
                    )
                ):
                    continue
                quarantine = blob_root / (
                    f".prune-{os.getpid()}-{threading.get_ident()}-"
                    f"{time.time_ns()}-{index}"
                )
                try:
                    os.replace(path, quarantine)
                except OSError:
                    continue
                quarantined.append(quarantine)
            return quarantined
        # Renaming while the write transaction is active is the atomic
        # reference fence; unlinking the private quarantine can be slow and
        # therefore happens only after the database has committed.
        with translated():
            with placement_write(self._store, placement) as conn:
                for path in _txn(conn):
                    path.unlink(missing_ok=True)

    # The placement's first-touch setup on its own, for a caller that runs it before any other verb.
    def prepare(self, *, placement: HostedPlacement, now: float | None = None) -> None:
        with translated():
            self._open(placement, clock(now))

    # ── writes ──────────────────────────────────────────────────────────────

    # Stores an upload under the room and gateway quotas, sharing a blob with the same SHA-256; a repeated
    # ``upload_id`` with the same content returns the stored row.
    def put(self, *, placement: HostedPlacement, room_id: str, upload_id: str, kind: str, name: str, mime: str,
            data: bytes, now: float | None = None) -> Mapping[str, JsonValue]:
        digest = hashlib.sha256(data).hexdigest()
        now = clock(now)
        with translated():
            blob_root = self._open(placement, now)
            # Long-lived gateways keep one store instance. Reclaim abandoned
            # uploads before applying quotas so an expired failed send cannot
            # permanently exhaust the room without a process restart.
            self._prune(placement, now)

            def _txn(conn) -> dict[str, Any]:
                existing = conn.execute(
                    "SELECT * FROM hosted_room_attachments WHERE room_id = %s AND upload_id = %s",
                    (room_id, upload_id),
                ).fetchone()
                if existing is not None:
                    if (
                        str(existing["kind"]) != kind
                        or str(existing["name"]) != name
                        or str(existing["mime"]) != mime
                        or int(existing["size"]) != len(data)
                        or str(existing["sha256"]) != digest
                    ):
                        raise AttachmentConflictError(
                            "upload_id was already used for different attachment content"
                        )
                    _read_blob(
                        blob_root,
                        blob_id=str(existing["blob_id"]),
                        size=int(existing["size"]),
                        sha256=str(existing["sha256"]),
                    )
                    return _metadata(existing, idempotent=True)

                room_totals = conn.execute(
                    """SELECT COALESCE(SUM(size), 0) AS bytes, COUNT(*) AS count
                         FROM hosted_room_attachments
                        WHERE room_id = %s AND state != 'disbanded'""",
                    (room_id,),
                ).fetchone()
                uncommitted = conn.execute(
                    """SELECT COALESCE(SUM(size), 0) AS bytes, COUNT(*) AS count
                         FROM hosted_room_attachments
                        WHERE room_id = %s AND state = 'uploaded'""",
                    (room_id,),
                ).fetchone()
                if int(room_totals["bytes"]) + len(data) > MAX_ROOM_ATTACHMENT_BYTES:
                    raise AttachmentQuotaError("room attachment quota exceeded")
                if int(room_totals["count"]) >= MAX_ROOM_ATTACHMENT_COUNT:
                    raise AttachmentQuotaError("room attachment count quota exceeded")
                total_count = int(conn.execute(
                    "SELECT COUNT(*) AS count FROM hosted_room_attachments").fetchone()["count"])
                if total_count >= MAX_GATEWAY_ATTACHMENT_COUNT:
                    raise AttachmentQuotaError("gateway attachment count quota exceeded")
                if (
                    int(uncommitted["bytes"]) + len(data) > MAX_ROOM_UNCOMMITTED_BYTES
                    or int(uncommitted["count"]) + 1 > MAX_ROOM_UNCOMMITTED_COUNT
                ):
                    raise AttachmentQuotaError("room uncommitted upload quota exceeded")

                blob = conn.execute(
                    "SELECT * FROM hosted_room_attachment_blobs WHERE sha256 = %s",
                    (digest,),
                ).fetchone()
                if blob is None:
                    physical = int(
                        conn.execute(
                            "SELECT COALESCE(SUM(size), 0) AS bytes FROM hosted_room_attachment_blobs"
                        ).fetchone()["bytes"]
                    )
                    if physical + len(data) > MAX_GATEWAY_BLOB_BYTES:
                        raise AttachmentQuotaError("gateway attachment quota exceeded")
                    blob_id = f"blob_{secrets.token_hex(16)}"
                    target = _blob_path(blob_root, blob_id)
                    _write_blob(target, data)
                    conn.execute(
                        """INSERT INTO hosted_room_attachment_blobs
                           (blob_id, sha256, size, ref_count, created_at)
                           VALUES (%s, %s, %s, 1, %s)""",
                        (blob_id, digest, len(data), now),
                    )
                else:
                    blob_id = str(blob["blob_id"])
                    _read_blob(blob_root, blob_id=blob_id, size=int(blob["size"]), sha256=digest)
                    conn.execute(
                        "UPDATE hosted_room_attachment_blobs SET ref_count = ref_count + 1 WHERE blob_id = %s",
                        (blob_id,),
                    )

                attachment_id = f"att_{secrets.token_hex(16)}"
                conn.execute(
                    """INSERT INTO hosted_room_attachments
                       (attachment_id, upload_id, room_id, event_id, kind, name, size,
                        mime, sha256, blob_id, recipient_member_ids_json, state,
                        created_at, updated_at, expires_at)
                       VALUES (%s, %s, %s, NULL, %s, %s, %s, %s, %s, %s, '[]', 'uploaded', %s, %s, %s)""",
                    (
                        attachment_id,
                        upload_id,
                        room_id,
                        kind,
                        name,
                        len(data),
                        mime,
                        digest,
                        blob_id,
                        now,
                        now,
                        now + UNCOMMITTED_TTL_SECONDS,
                    ),
                )
                row = conn.execute(
                    "SELECT * FROM hosted_room_attachments WHERE attachment_id = %s",
                    (attachment_id,),
                ).fetchone()
                if row is None:  # pragma: no cover - guarded by insert
                    raise RuntimeError("stored attachment could not be reloaded")
                return _metadata(row)
            with placement_write(self._store, placement) as conn:
                return _txn(conn)

    def commit_message_with_receipt(
        self, *, placement: HostedPlacement, room_id: str, event_id: str, manifest: Sequence[Mapping[str, JsonValue]],
        recipient_member_ids: Sequence[str], viewer_access: bool = False, retention_seconds: float | None = None,
        hold_until_event: bool = False, now: float | None = None,
    ) -> tuple[list[Mapping[str, JsonValue]], tuple[str, ...]]:
        """Commit a manifest and report only rows transitioned by this call."""
        normalized = [dict(entry) for entry in manifest]
        recipients_json = json.dumps(sorted(recipient_member_ids), separators=(",", ":"))
        now = clock(now)
        expires_at = (
            now + UNCOMMITTED_TTL_SECONDS
            if hold_until_event
            else None
            if retention_seconds is None
            else now + float(retention_seconds)
        )

        def _txn(conn) -> tuple[str, ...]:
            transitioned: list[str] = []
            for entry in normalized:
                row = conn.execute(
                    "SELECT * FROM hosted_room_attachments WHERE attachment_id = %s",
                    (entry["attachment_id"],),
                ).fetchone()
                if row is None or str(row["room_id"]) != room_id:
                    raise AttachmentNotFoundError("attachment is not uploaded for this room")
                durable = {
                    "attachment_id": str(row["attachment_id"]),
                    "kind": str(row["kind"]),
                    "name": str(row["name"]),
                    "size": int(row["size"]),
                    "mime": str(row["mime"]),
                }
                if durable != entry:
                    raise AttachmentConflictError(
                        "attachment manifest metadata does not match the uploaded bytes"
                    )
                if row["expires_at"] is not None and float(row["expires_at"]) <= now:
                    raise AttachmentNotFoundError("attachment has expired")
                state = str(row["state"])
                if state == "disbanded":
                    raise AttachmentNotFoundError("attachment belongs to a disbanded room")
                if state == "committed" and (
                    str(row["event_id"] or "") != event_id
                    or str(row["recipient_member_ids_json"]) != recipients_json
                    or bool(row["viewer_access"]) != bool(viewer_access)
                ):
                    raise AttachmentConflictError(
                        "attachment is already owned by a different room event"
                    )
                if state == "uploaded":
                    transitioned.append(entry["attachment_id"])
                _read_blob(
                    blob_root,
                    blob_id=str(row["blob_id"]),
                    size=int(row["size"]),
                    sha256=str(row["sha256"]),
                )
            for entry in normalized:
                if entry["attachment_id"] not in transitioned:
                    continue
                updated = conn.execute(
                    """UPDATE hosted_room_attachments
                          SET event_id = %s, recipient_member_ids_json = %s, viewer_access = %s, state = 'committed',
                              updated_at = %s, expires_at = %s
                        WHERE attachment_id = %s AND state = 'uploaded'""",
                    (
                        event_id,
                        recipients_json,
                        1 if viewer_access else 0,
                        now,
                        expires_at,
                        entry["attachment_id"],
                    ),
                )
                if updated.rowcount != 1:
                    raise AttachmentConflictError(
                        "attachment changed during message commitment"
                    )
            return tuple(transitioned)
        with translated():
            blob_root = self._open(placement, now)
            with placement_write(self._store, placement) as conn:
                return normalized, _txn(conn)

    def abort_message_commit(self, *, placement: HostedPlacement, room_id: str, event_id: str,
                             attachment_ids: Sequence[str], now: float | None = None) -> int:
        """Return a failed pre-event commitment to its bounded upload state."""
        attachment_ids = list(attachment_ids)
        if not attachment_ids:
            return 0
        now = clock(now)

        def _txn(conn) -> int:
            ids = attachment_ids
            owner = _owner_event(conn, room_id, event_id)
            if owner is not None:
                published_ids = {
                    item.get("attachment_id")
                    for item in json.loads(str(owner["payload_json"])).get("attachments", [])
                    if isinstance(item, Mapping)
                }
                ids = [
                    value for value in ids if value not in published_ids
                ]
            if not ids:
                return 0
            placeholders = ",".join("%s" for _ in ids)
            changed = conn.execute(
                f"""UPDATE hosted_room_attachments
                        SET event_id = NULL, recipient_member_ids_json = '[]', viewer_access = 0,
                            state = 'uploaded', updated_at = %s, expires_at = %s
                      WHERE room_id = %s AND event_id = %s AND state = 'committed'
                        AND attachment_id IN ({placeholders})""",
                (
                    now,
                    now + UNCOMMITTED_TTL_SECONDS,
                    room_id,
                    event_id,
                    *ids,
                ),
            )
            return int(changed.rowcount)
        with translated():
            self._open(placement, now)
            with placement_write(self._store, placement) as conn:
                return _txn(conn)

    # ── reads ───────────────────────────────────────────────────────────────

    # A committed attachment's metadata and verified bytes, for one recipient or for a viewer.
    def read(self, *, placement: HostedPlacement, room_id: str, attachment_id: str, recipient_member_id: str | None,
             event_id: str | None = None, viewer: bool = False, now: float | None = None) -> AttachmentData:
        now = clock(now)

        def _read(conn) -> Mapping[str, Any]:
            return _read_committed_row(
                conn, room_id=room_id, attachment_id=attachment_id,
                recipient_member_id=recipient_member_id or "",
                normalized_event=event_id, viewer=viewer, now=now, reconciled=not self._store.read_only,
            )
        with translated():
            blob_root = self._read_root(placement, now)
            row = self._read_rows(placement, _read)
            data = _read_blob(
                blob_root,
                blob_id=str(row["blob_id"]),
                size=int(row["size"]),
                sha256=str(row["sha256"]),
            )
            return AttachmentData(_metadata(row), data)

    def describe(self, *, placement: HostedPlacement, room_id: str, attachment_id: str, recipient_member_id: str,
                 event_id: str, now: float | None = None) -> Mapping[str, JsonValue]:
        """``read`` without bytes: the metadata (including the upload-verified SHA-256) of a
        committed attachment this recipient may read for this event."""
        now = clock(now)

        def _read(conn) -> dict[str, Any]:
            row = _read_committed_row(
                conn, room_id=room_id, attachment_id=attachment_id,
                recipient_member_id=recipient_member_id,
                normalized_event=event_id, viewer=False, now=now, reconciled=not self._store.read_only,
            )
            return _metadata(row)
        with translated():
            self._read_root(placement, now)
            return self._read_rows(placement, _read)

    def read_range(self, *, placement: HostedPlacement, room_id: str, attachment_id: str, recipient_member_id: str,
                   event_id: str, offset: int, length: int, now: float | None = None) -> AttachmentData:
        """``read`` for one slice: same authorization, metadata carries the stored whole-file
        SHA-256 and the receiver verifies the reassembled bytes against it."""
        now = clock(now)

        def _read(conn) -> Mapping[str, Any]:
            row = _read_committed_row(
                conn, room_id=room_id, attachment_id=attachment_id,
                recipient_member_id=recipient_member_id,
                normalized_event=event_id, viewer=False, now=now, reconciled=not self._store.read_only,
            )
            if offset >= int(row["size"]):
                raise AttachmentError("attachment range starts past the end")
            return row
        with translated():
            blob_root = self._read_root(placement, now)
            row = self._read_rows(placement, _read)
            data = _read_blob_range(
                blob_root, blob_id=str(row["blob_id"]), size=int(row["size"]), offset=offset, length=length)
            return AttachmentData(_metadata(row), data)

    def read_viewer(self, *, placement: HostedPlacement, room_id: str, attachment_id: str, event_id: str | None = None,
                    recipient_member_id: str | None = None, authority_gateway_id: str, authority_epoch: int,
                    now: float | None = None) -> AttachmentData:
        """Read for a live hosted viewer, fencing revocation across byte I/O."""
        scope = {
            "room_id": room_id,
            "authority_gateway_id": authority_gateway_id,
            "authority_epoch": authority_epoch,
        }
        selection = {
            "room_id": room_id,
            "attachment_id": attachment_id,
            "recipient_member_id": recipient_member_id or "",
            "normalized_event": event_id,
            "viewer": True,
            "reconciled": not self._store.read_only,
        }

        # Each viewer read is one read transaction: the room check and the row read see one snapshot.
        def _first(conn) -> Mapping[str, Any]:
            _require_viewer_room(conn, **scope)
            return _read_committed_row(conn, **selection, now=clock(now))

        def _second(conn) -> Mapping[str, Any]:
            _require_viewer_room(conn, **scope)
            current = _read_committed_row(conn, **selection, now=clock(now))
            if current["blob_id"] != row["blob_id"] or _metadata(current) != _metadata(row):
                raise AttachmentNotFoundError("attachment changed during viewer read")
            return current
        with translated():
            blob_root = self._read_root(placement, clock(now))
            row = self._read_rows(placement, _first)
            # No execution lock or database snapshot spans filesystem/hash work.
            data = _read_blob(
                blob_root, blob_id=str(row["blob_id"]), size=int(row["size"]), sha256=str(row["sha256"]),
            )
            current = self._read_rows(placement, _second)
            return AttachmentData(_metadata(current), data)


def _register() -> None:
    register_repository("hosted_room_attachments", lambda store: PgHostedRoomAttachmentRepository(store))


_register()

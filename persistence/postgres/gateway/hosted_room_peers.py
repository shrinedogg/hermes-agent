"""``HostedRoomPeerRepository`` on PostgreSQL (stage 4): peer room links, grant revocations, peer reservations
and remote-run receipts in the placement's schema, each write one write transaction and each read one read on
the placement - the reference's SQLite peer store ported to the PG dialect.

The peer tables come from the declarative DDL (no step, no first-touch). The ``grant`` column is a reserved
word in the PG dialect and is quoted wherever the reference's SQL names it. The reference's file-mode
management (owner-only grant files) has no twin here.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from dataclasses import asdict
from typing import Any

from persistence import JsonValue
from persistence.gateway import hosted_rooms as _bounds  # the owed-revoke bounds, read at each call
from persistence.gateway.hosted_rooms import (
    AuthorityConflictError, HostedPlacement,
    HostedRoomError, OwedRevokesFullError, RemoteRunIdentity, RemoteRunReceipt, RoomGrantClaims, RoomLinkRecord,
    grant_digest, room_grant_hard_expiry,
)
from session_rules.hosted_rooms import _claim_values, clock as _now, room_grant_scope_key

from .._backend import PostgresStore, register_repository, translated
from ._hosted_common import placement_read, placement_write, probe
from .hosted_rooms import _room_row

__all__ = ["PgHostedRoomPeerRepository"]

# The remote-run receipt key, the room-link columns in table order, and the SQL built from them.
_REMOTE_RUN_IDENTITY_COLUMNS = (
    "room_id", "home_install_id", "authority_gateway_id", "authority_epoch", "member_id", "target_install_id",
    "target_profile", "task_id", "execution_generation")
_LINK_COLUMNS = (
    "room_id", "member_id", "target_url", "target_profile", "grant", "catalog_json", "cancellation_scope_id",
    "trace_id", "transport_security", "status", "updated_at")
_REMOTE_RUN_WHERE = " AND ".join(f"{column} = %s" for column in _REMOTE_RUN_IDENTITY_COLUMNS)
_SELECT_REMOTE_RUN = f"SELECT * FROM hosted_room_remote_runs WHERE {_REMOTE_RUN_WHERE}"
_LIVE_RESERVATION_WHERE = ("WHERE room_id = %s AND target_profile = %s AND expires_at > %s AND revoked_at IS NULL")
_SELECT_LIVE_RESERVATION = (f"SELECT 1 FROM hosted_room_peer_reservations {_LIVE_RESERVATION_WHERE} LIMIT 1")
_OWE_REVOKE = """INSERT INTO hosted_room_owed_revokes(
        room_id, member_id, grant_digest, target_url, "grant", expires_at, next_attempt_at, created_at, held)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s) ON CONFLICT (room_id, member_id, grant_digest) DO NOTHING"""
_DUE_OWED_REVOKES = """SELECT room_id, member_id, grant_digest, target_url, "grant", attempts
    FROM hosted_room_owed_revokes WHERE held = 0 AND next_attempt_at <= %s AND expires_at > %s
    ORDER BY next_attempt_at, created_at LIMIT %s"""


# Records one owed revoke inside the caller's write: an existing key stays, then the member bound, then the table
# ceiling, where expired rows make room first. A held row past the member bound refuses as a held-grant refusal.
def _owe_revoke_locked(
    conn, *, room_id: str, member_id: str, target_url: str, grant: str,
    expires_at: float, now: float, held: bool = False) -> None:
    expires_at = min(float(expires_at), now + _bounds.MAX_OWED_REVOKE_SECONDS)
    digest = grant_digest(grant)
    member = (room_id, member_id)
    if conn.execute("SELECT 1 FROM hosted_room_owed_revokes WHERE room_id = %s AND member_id = %s AND grant_digest = %s",
                    (*member, digest)).fetchone() is not None:
        return
    owed = int(conn.execute("SELECT COUNT(*) FROM hosted_room_owed_revokes WHERE room_id = %s AND member_id = %s",
                            member).fetchone()["count"])
    if owed >= _bounds.MAX_OWED_REVOKES_PER_MEMBER:
        if held:
            raise HostedRoomError("too many held group chat grant revokes")
        raise OwedRevokesFullError("too many owed revokes for this group chat member")
    if int(conn.execute("SELECT COUNT(*) AS count FROM hosted_room_owed_revokes").fetchone()["count"]) >= _bounds.MAX_OWED_REVOKES:
        # A grant past its expiry passes nowhere: its row is the one to give up.
        conn.execute("DELETE FROM hosted_room_owed_revokes WHERE expires_at <= %s", (now,))
        if int(conn.execute("SELECT COUNT(*) AS count FROM hosted_room_owed_revokes").fetchone()["count"]) >= _bounds.MAX_OWED_REVOKES:
            raise OwedRevokesFullError("too many owed room grant revokes")
    conn.execute(_OWE_REVOKE, (
        room_id, member_id, digest, target_url, grant, expires_at, now, now, int(held)))


def _reservation_superseded(row: Mapping[str, Any], gateway_id: str, epoch: int) -> bool:
    """A newer epoch, or the same epoch under another gateway, outranks this claim."""
    row_epoch = int(row["authority_epoch"])
    return row_epoch > epoch or (row_epoch == epoch and str(row["authority_gateway_id"]) != gateway_id)


def _reservation_values(claims: RoomGrantClaims) -> tuple[str, str, str, str, int]:
    """(room_id, member_id, target_profile, authority_gateway_id, authority_epoch) of claims the caller validated."""
    return (claims.room_id, claims.member_id, claims.target_profile, claims.authority_gateway_id,
            int(claims.authority_epoch or 0))


# The receipt key of a typed record or a stored row.
def _remote_run_identity(record: RemoteRunIdentity | RemoteRunReceipt | Mapping[str, Any]) -> tuple[Any, ...]:
    if isinstance(record, Mapping):
        return tuple(record[column] for column in _REMOTE_RUN_IDENTITY_COLUMNS)
    return tuple(getattr(record, column) for column in _REMOTE_RUN_IDENTITY_COLUMNS)


class PgHostedRoomPeerRepository:
    """Peer room links, grant revocations, peer reservations and remote-run receipts on the placement's schema."""

    def __init__(self, store: PostgresStore) -> None:
        self._store = store

    # ── writes ──────────────────────────────────────────────────────────────

    # Inserts or replaces the member's link, refusing a new link once ``max_links`` are stored; with ``hold_replaced``
    # a replaced grant that outlives the new one is held as an owed revoke. The new grant's held row goes.
    def upsert_room_link_record(self, *, placement: HostedPlacement, record: RoomLinkRecord, max_links: int, hold_replaced: bool = False) -> None:
        def write(conn) -> None:
            existing = conn.execute(
                "SELECT target_url, \"grant\" FROM hosted_room_links WHERE room_id = %s AND member_id = %s",
                (record.room_id, record.member_id)).fetchone()
            if existing is None and int(conn.execute("SELECT COUNT(*) AS count FROM hosted_room_links").fetchone()["count"]) >= max_links:
                raise HostedRoomError("too many stored room links")
            if (existing is not None and hold_replaced and existing["grant"] != record.grant
                    and room_grant_hard_expiry(str(existing["grant"])) > room_grant_hard_expiry(record.grant)):
                _owe_revoke_locked(
                    conn, room_id=record.room_id, member_id=record.member_id, target_url=str(existing["target_url"]),
                    grant=str(existing["grant"]), expires_at=room_grant_hard_expiry(str(existing["grant"])),
                    now=_now(None), held=True)
            conn.execute("""INSERT INTO hosted_room_links(
                       room_id, member_id, target_url, target_profile, "grant",
                       catalog_json, cancellation_scope_id, trace_id,
                       transport_security, status, updated_at
                   ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                   ON CONFLICT (room_id, member_id) DO UPDATE SET
                       target_url = EXCLUDED.target_url,
                       target_profile = EXCLUDED.target_profile,
                       "grant" = EXCLUDED."grant",
                       catalog_json = EXCLUDED.catalog_json,
                       cancellation_scope_id = EXCLUDED.cancellation_scope_id,
                       trace_id = EXCLUDED.trace_id,
                       transport_security = EXCLUDED.transport_security,
                       status = EXCLUDED.status,
                       updated_at = EXCLUDED.updated_at""",
                tuple(getattr(record, column) for column in _LINK_COLUMNS))
            conn.execute("""DELETE FROM hosted_room_owed_revokes
                    WHERE room_id = %s AND member_id = %s AND grant_digest = %s AND held = 1""",
                (record.room_id, record.member_id, grant_digest(record.grant)))
        with translated():
            with placement_write(self._store, placement) as conn:
                write(conn)

    # True when the link existed and took the new status.
    def update_room_link_status(self, *, placement: HostedPlacement, room_id: str, member_id: str, status: str, now: float | None = None) -> bool:
        def write(conn) -> bool:
            return conn.execute(
                "UPDATE hosted_room_links SET status = %s, updated_at = %s WHERE room_id = %s AND member_id = %s",
                (status, _now(now), room_id, member_id)).rowcount == 1
        with translated():
            with placement_write(self._store, placement) as conn:
                return write(conn)

    # Deletes the room's links and returns how many.
    def delete_room_link_records(self, *, placement: HostedPlacement, room_id: str) -> int:
        def write(conn) -> int:
            return conn.execute("DELETE FROM hosted_room_links WHERE room_id = %s", (room_id,)).rowcount
        with translated():
            with placement_write(self._store, placement) as conn:
                return write(conn)

    # Revokes the scope's grants issued until now and the matching reservation; expired revocations go first.
    def revoke_room_grant_scope(self, *, placement: HostedPlacement, claims: RoomGrantClaims, expires_at: float, now: float | None = None) -> None:
        fields = asdict(claims)
        scope_key = room_grant_scope_key(fields)
        timestamp = _now(now)
        expiry = float(expires_at)
        if expiry <= timestamp:
            return

        def write(conn) -> None:
            conn.execute("DELETE FROM hosted_room_revoked_grants WHERE expires_at <= %s", (timestamp,))
            conn.execute("""INSERT INTO hosted_room_revoked_grants(
                       scope_key, expires_at, revoked_before
                   ) VALUES (%s, %s, %s)
                   ON CONFLICT (scope_key) DO UPDATE SET
                       expires_at = GREATEST(hosted_room_revoked_grants.expires_at, EXCLUDED.expires_at),
                       revoked_before = GREATEST(hosted_room_revoked_grants.revoked_before, EXCLUDED.revoked_before)""", (scope_key, expiry, timestamp))
            conn.execute("""UPDATE hosted_room_peer_reservations SET revoked_at = %s, updated_at = %s WHERE room_id = %s
                    AND member_id = %s AND target_profile = %s AND authority_gateway_id = %s
                    AND authority_epoch = %s""",
                (
                    timestamp, timestamp,
                    *_claim_values(fields, ("room_id", "member_id", "target_profile", "authority_gateway_id")).values(),
                    int(fields.get("authority_epoch") or 0)))
        with translated():
            with placement_write(self._store, placement) as conn:
                write(conn)

    # The caller validates the claims first. Refuses a reservation a live one outranks, revokes older epochs, upserts.
    def reserve_peer_room(self, *, placement: HostedPlacement, claims: RoomGrantClaims, expires_at: float, now: float | None = None) -> None:
        timestamp = _now(now)
        expiry = float(expires_at)
        if expiry <= timestamp:
            raise HostedRoomError("peer room reservation must expire in the future")
        values = _reservation_values(claims)
        room_id, _, target_profile, gateway_id, epoch = values

        def write(conn) -> None:
            conn.execute("DELETE FROM hosted_room_peer_reservations WHERE expires_at <= %s", (timestamp,))
            authority_rows = conn.execute(
                f"""SELECT authority_gateway_id, authority_epoch
                    FROM hosted_room_peer_reservations {_LIVE_RESERVATION_WHERE}""", (room_id, target_profile, timestamp)
            ).fetchall()
            if any(_reservation_superseded(row, gateway_id, epoch) for row in authority_rows):
                raise AuthorityConflictError("peer room reservation authority changed")
            conn.execute("""UPDATE hosted_room_peer_reservations SET revoked_at = %s, updated_at = %s WHERE room_id = %s
                    AND target_profile = %s AND authority_epoch < %s AND revoked_at IS NULL""",
                (timestamp, timestamp, room_id, target_profile, epoch))
            existing = conn.execute("""SELECT authority_gateway_id, authority_epoch FROM hosted_room_peer_reservations
                    WHERE room_id = %s AND member_id = %s AND target_profile = %s""", values[:3]).fetchone()
            if existing is not None and _reservation_superseded(existing, gateway_id, epoch):
                raise AuthorityConflictError("peer room reservation authority changed")
            conn.execute("""INSERT INTO hosted_room_peer_reservations(
                       room_id, member_id, target_profile, authority_gateway_id,
                       authority_epoch, expires_at, revoked_at, created_at, updated_at
                   ) VALUES (%s, %s, %s, %s, %s, %s, NULL, %s, %s)
                   ON CONFLICT (room_id, member_id, target_profile) DO UPDATE SET
                       authority_gateway_id = EXCLUDED.authority_gateway_id,
                       authority_epoch = EXCLUDED.authority_epoch,
                       expires_at = GREATEST(hosted_room_peer_reservations.expires_at, EXCLUDED.expires_at),
                       revoked_at = NULL,
                       updated_at = EXCLUDED.updated_at""", (*values, expiry, timestamp, timestamp))
        with translated():
            with placement_write(self._store, placement) as conn:
                write(conn)

    # Records a remote run once per receipt key; a repeat with the same run and session only moves ``updated_at``.
    def upsert_remote_run_receipt(self, *, placement: HostedPlacement, receipt: RemoteRunReceipt, now: float | None = None) -> None:
        timestamp = _now(now)
        identity = _remote_run_identity(receipt)
        immutable = (*identity, receipt.run_id, receipt.session_id)

        def write(conn) -> None:
            existing = conn.execute(_SELECT_REMOTE_RUN, identity).fetchone()
            if existing is not None:
                if (*_remote_run_identity(existing), existing["run_id"], existing["session_id"]) != immutable:
                    raise HostedRoomError("remote run receipt conflicts with its logical task")
                conn.execute(
                    f"UPDATE hosted_room_remote_runs SET updated_at = %s WHERE {_REMOTE_RUN_WHERE}", (timestamp, *identity))
                return
            conn.execute("""INSERT INTO hosted_room_remote_runs(
                       room_id, home_install_id, authority_gateway_id,
                       authority_epoch, member_id, target_install_id,
                       target_profile, task_id, execution_generation, run_id,
                       session_id, created_at, updated_at
                   ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""", (*immutable, timestamp, timestamp))
            from ._hosted_work import capture_transition_locked
            capture_transition_locked(conn, receipt.room_id)
        with translated():
            with placement_write(self._store, placement) as conn:
                write(conn)

    # Drops the room's links; with ``owe_routes`` each link's grant is owed and the held rows become due, else the
    # held rows (those named in ``revoked_digests`` when given) are forgotten.
    def retire_room_link_records(self, *, placement: HostedPlacement, room_id: str, owe_routes: bool = False, revoked_digests: Collection[str] | None = None, now: float | None = None) -> int:
        timestamp = _now(now)

        def write(conn) -> int:
            rows = conn.execute(
                "SELECT member_id, target_url, \"grant\" FROM hosted_room_links WHERE room_id = %s", (room_id,)).fetchall()
            for row in rows if owe_routes else ():
                grant = str(row["grant"])
                _owe_revoke_locked(conn, room_id=room_id, member_id=str(row["member_id"]),
                                   target_url=str(row["target_url"]), grant=grant,
                                   expires_at=room_grant_hard_expiry(grant), now=timestamp)
            conn.execute("DELETE FROM hosted_room_links WHERE room_id = %s", (room_id,))
            if not owe_routes and revoked_digests is not None:
                if list(revoked_digests):
                    conn.execute("""DELETE FROM hosted_room_owed_revokes
                            WHERE room_id = %s AND held = 1 AND grant_digest = ANY(%s)""",
                        (room_id, list(revoked_digests)))
            elif not owe_routes:
                conn.execute("DELETE FROM hosted_room_owed_revokes WHERE room_id = %s AND held = 1", (room_id,))
            else:
                conn.execute("UPDATE hosted_room_owed_revokes SET held = 0, next_attempt_at = %s WHERE room_id = %s AND held = 1",
                             (timestamp, room_id))
            return len(rows)
        with translated():
            with placement_write(self._store, placement) as conn:
                return write(conn)

    # Owes one revoke, within the member and table bounds.
    def owe_room_grant_revoke(self, *, placement: HostedPlacement, room_id: str, member_id: str, target_url: str, grant: str, expires_at: float, now: float | None = None) -> None:
        timestamp = _now(now)
        with translated():
            with placement_write(self._store, placement) as conn:
                _owe_revoke_locked(
                    conn, room_id=room_id, member_id=member_id, target_url=target_url, grant=grant,
                    expires_at=expires_at, now=timestamp)

    # Holds a grant while the room is live; on a tombstoned room it is owed at once.
    def hold_room_grant_revoke(self, *, placement: HostedPlacement, room_id: str, member_id: str, target_url: str, grant: str, expires_at: float, now: float | None = None) -> None:
        timestamp = _now(now)

        def write(conn) -> None:
            room = _room_row(conn, "SELECT disbanded_at FROM hosted_rooms WHERE room_id = %s", (room_id,), room_id)
            try:
                _owe_revoke_locked(conn, room_id=room_id, member_id=member_id, target_url=target_url, grant=grant,
                                   expires_at=expires_at, now=timestamp, held=room["disbanded_at"] is None)
            except OwedRevokesFullError as exc:
                if room["disbanded_at"] is None:
                    raise HostedRoomError("too many held group chat grant revokes") from exc
                raise
        with translated():
            with placement_write(self._store, placement) as conn:
                write(conn)

    # Forgets a settled revoke, or counts the attempt and schedules the next.
    def settle_room_grant_revoke(self, *, placement: HostedPlacement, room_id: str, member_id: str, grant_digest: str, retry_at: float | None = None) -> None:
        key = (room_id, member_id, grant_digest)

        def write(conn) -> None:
            if retry_at is None:
                conn.execute("DELETE FROM hosted_room_owed_revokes WHERE room_id = %s AND member_id = %s AND grant_digest = %s", key)
            else:
                conn.execute("""UPDATE hosted_room_owed_revokes SET attempts = attempts + 1, next_attempt_at = %s
                                 WHERE room_id = %s AND member_id = %s AND grant_digest = %s""", (float(retry_at), *key))
        with translated():
            with placement_write(self._store, placement) as conn:
                write(conn)

    # Deletes the attempt's receipt only while it names ``run_id``; a dropped row runs the work-record capture.
    def drop_remote_run_receipt(self, *, placement: HostedPlacement, identity: RemoteRunIdentity, run_id: str) -> bool:
        def write(conn) -> bool:
            dropped = conn.execute(
                f"DELETE FROM hosted_room_remote_runs WHERE {_REMOTE_RUN_WHERE} AND run_id = %s",
                (*_remote_run_identity(identity), run_id)).rowcount
            if dropped:
                from ._hosted_work import capture_transition_locked
                capture_transition_locked(conn, identity.room_id)
            return bool(dropped)
        with translated():
            with placement_write(self._store, placement) as conn:
                return write(conn)

    # ── reads ───────────────────────────────────────────────────────────────

    # True while the room has a link or an unexpired owed revoke.
    def room_has_peer_obligations(self, *, placement: HostedPlacement, room_id: str, now: float | None = None) -> bool:
        timestamp = _now(now)

        def read(conn) -> bool:
            return conn.execute(
                """SELECT 1 FROM hosted_room_links WHERE room_id = %s
                UNION ALL SELECT 1 FROM hosted_room_owed_revokes WHERE room_id = %s AND expires_at > %s LIMIT 1""",
                (room_id, room_id, timestamp)).fetchone() is not None
        with translated():
            with placement_read(self._store, placement) as conn:
                return read(conn)

    # True while any owed revoke is recorded.
    def owes_room_grant_revokes(self, *, placement: HostedPlacement) -> bool:
        def read(conn) -> bool:
            return conn.execute("SELECT 1 FROM hosted_room_owed_revokes LIMIT 1").fetchone() is not None
        with translated():
            with placement_read(self._store, placement) as conn:
                return read(conn)

    # The due unheld revokes, oldest first; expired rows are deleted in a write only when one exists.
    def due_room_grant_revokes(self, *, placement: HostedPlacement, limit: int, now: float | None = None) -> Sequence[Mapping[str, JsonValue]]:
        timestamp = _now(now)

        def read(conn) -> tuple[list[Mapping[str, Any]], bool]:
            rows = conn.execute(_DUE_OWED_REVOKES, (timestamp, timestamp, int(limit))).fetchall()
            expired = conn.execute(
                "SELECT 1 FROM hosted_room_owed_revokes WHERE expires_at <= %s LIMIT 1", (timestamp,)).fetchone()
            return rows, expired is not None
        with translated():
            with placement_read(self._store, placement) as conn:
                rows, expired = read(conn)
            if expired:
                with placement_write(self._store, placement) as conn:
                    conn.execute(
                        "DELETE FROM hosted_room_owed_revokes WHERE expires_at <= %s", (timestamp,))
        return list(rows)

    # Deletes the room's expired held rows, then returns the unexpired ones.
    def held_room_grant_revokes(self, *, placement: HostedPlacement, room_id: str, now: float | None = None) -> Sequence[Mapping[str, JsonValue]]:
        timestamp = _now(now)

        def write(conn) -> list[Mapping[str, Any]]:
            conn.execute("DELETE FROM hosted_room_owed_revokes WHERE room_id = %s AND held = 1 AND expires_at <= %s",
                         (room_id, timestamp))
            return conn.execute(
                """SELECT target_url, "grant" FROM hosted_room_owed_revokes
                    WHERE room_id = %s AND held = 1 AND expires_at > %s""", (room_id, timestamp)).fetchall()
        with translated():
            with placement_write(self._store, placement) as conn:
                return write(conn)

    # Every stored link, by room and member.
    def list_room_link_records(self, *, placement: HostedPlacement) -> Sequence[Mapping[str, JsonValue]]:
        def read(conn) -> list[Mapping[str, Any]]:
            return conn.execute(
                """SELECT room_id, member_id, target_url, target_profile, "grant",
                          catalog_json, cancellation_scope_id, trace_id,
                          transport_security, status, updated_at
                     FROM hosted_room_links
                 ORDER BY room_id, member_id""").fetchall()
        with translated():
            with placement_read(self._store, placement) as conn:
                return read(conn)

    # True when an unexpired revocation of the claims' scope covers their issue time.
    def room_grant_is_revoked(self, *, placement: HostedPlacement, claims: RoomGrantClaims, now: float | None = None) -> bool:
        timestamp = _now(now)
        scope_key = room_grant_scope_key(asdict(claims))
        issued_at = float(claims.issued_at or 0)

        def read(conn) -> Mapping[str, Any] | None:
            return conn.execute(
                """SELECT revoked_before FROM hosted_room_revoked_grants
                WHERE scope_key = %s AND expires_at > %s""", (scope_key, timestamp)).fetchone()
        with translated():
            with placement_read(self._store, placement) as conn:
                row = read(conn)
        return row is not None and issued_at <= float(row["revoked_before"])

    # The caller validates the claims first; true when their reservation is live and unrevoked.
    def peer_room_grant_is_current(self, *, placement: HostedPlacement, claims: RoomGrantClaims, now: float | None = None) -> bool:
        timestamp = _now(now)
        values = _reservation_values(claims)

        def read(conn) -> bool:
            return conn.execute(
                """SELECT 1 FROM hosted_room_peer_reservations WHERE room_id = %s AND member_id = %s
                AND target_profile = %s AND authority_gateway_id = %s AND authority_epoch = %s
                AND expires_at > %s AND revoked_at IS NULL LIMIT 1""", (*values, timestamp)).fetchone() is not None
        with translated():
            with placement_read(self._store, placement) as conn:
                return read(conn)

    # The caller validates the arguments first; probes for a live reservation with a short lock wait and no schema step.
    def probe_peer_room_reservation(self, *, placement: HostedPlacement, room_id: str, target_profile: str, now: float | None = None) -> bool:
        params = (room_id, target_profile, _now(now))
        return probe(
            self._store, placement, "hosted_room_peer_reservations", _SELECT_LIVE_RESERVATION, params,
            "peer room ownership is temporarily unavailable")

    # The stored receipt for this key, or None.
    def remote_run_receipt(self, *, placement: HostedPlacement, identity: RemoteRunIdentity) -> Mapping[str, JsonValue] | None:
        def read(conn) -> Mapping[str, Any] | None:
            return conn.execute(_SELECT_REMOTE_RUN, _remote_run_identity(identity)).fetchone()
        with translated():
            with placement_read(self._store, placement) as conn:
                return read(conn)


def _register() -> None:
    register_repository("hosted_room_peers", lambda store: PgHostedRoomPeerRepository(store))


_register()

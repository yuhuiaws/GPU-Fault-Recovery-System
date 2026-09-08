from __future__ import annotations

import json
import time
from datetime import datetime, timedelta, timezone
from threading import Event as ThreadEvent
from typing import Any, Callable

from gpu_fault.store.shared.telemetry_models import (
    TELEMETRY_SPOOL_MAX_ATTEMPTS,
    SpooledTelemetry,
)


class PostgresTelemetrySpoolMixin:
    # Attributes supplied by the composed concrete implementation.
    _db: Any
    url: Any

    TELEMETRY_SPOOL_MAX_ATTEMPTS = TELEMETRY_SPOOL_MAX_ATTEMPTS

    # How long one process trusts its last depth reading. Twelve ingress
    # processes at 0.25 s each read the table 48 times a second, and the
    # reading was ``telemetry_spool_stats`` -- a ``payload::text`` of every
    # row -- so the cost peaked exactly when the spool was backed up (E-2).
    # The projection only needs counts, and admitting a few rows past the
    # cap during one second is the trade the cache already makes.
    _TELEMETRY_SPOOL_DEPTH_TTL_SECONDS = 1.0

    _telemetry_spool_depth_cache: dict | None = None

    def listen_telemetry_spool_notifications(
        self,
        stop_event: ThreadEvent,
        on_notification: Callable[[str], None],
        on_state: Callable[[bool], None],
        *,
        timeout_seconds: float = 1.0,
        writer_check_seconds: float = 5.0,
    ) -> None:
        """Wake spool consumers without polling the processor channel.

        Once per ``writer_check_seconds`` the loop asks the server whether it
        is still the writer (G-11): after a failover a LISTEN connection left
        on the demoted instance stays open but never receives a NOTIFY.
        """

        import psycopg

        from gpu_fault.store.postgres.pool import _reject_reader

        while not stop_event.is_set():
            try:
                with psycopg.connect(
                    self.url,
                    autocommit=True,
                    connect_timeout=5,
                ) as connection:
                    connection.execute("LISTEN gpu_fault_telemetry_spool")
                    on_state(True)
                    last_writer_check = time.monotonic()
                    while not stop_event.is_set():
                        monotonic = time.monotonic()
                        if monotonic - last_writer_check >= writer_check_seconds:
                            last_writer_check = monotonic
                            _reject_reader(
                                connection,
                                "telemetry spool LISTEN connection is on a "
                                "read-only replica; NOTIFY is not forwarded there",
                            )
                        payloads = set()
                        for notification in connection.notifies(
                            timeout=timeout_seconds,
                            stop_after=256,
                        ):
                            if notification.payload not in payloads:
                                on_notification(notification.payload)
                                payloads.add(notification.payload)
                            if stop_event.is_set():
                                break
            except Exception:
                on_state(False)
                if stop_event.wait(1.0):
                    return
        on_state(False)

    def _projected_telemetry_spool_depths(self, *, now: datetime) -> dict:
        cache = self._telemetry_spool_depth_cache
        monotonic = time.monotonic()
        if (
            cache is None
            or monotonic - cache["read_at"] >= self._TELEMETRY_SPOOL_DEPTH_TTL_SECONDS
        ):
            depths = self.telemetry_spool_depths()
            cache = {
                "read_at": monotonic,
                "depth": depths["depth"],
                "by_cluster": dict(depths["by_cluster"]),
                "admitted": 0,
                "admitted_by_cluster": {},
            }
            # Two threads refreshing at once both compute the same
            # reading and the loser's admissions are folded into the
            # winner's counters on its next batch, so this needs no lock.
            self._telemetry_spool_depth_cache = cache
        return cache

    def try_spool_telemetry_requests(
        self,
        requests,
        *,
        max_depth: int,
        max_cluster_depth: int,
        now: datetime | None = None,
    ) -> list[tuple[object, str | None]]:
        """One statement admits and coalesces a whole batch.

        No per-cluster split, unlike ``try_enqueue_processor_requests_batch``
        above: that split exists because the counter trigger serialises
        clusters against each other, and there is no counter row here, so a
        batch spanning fifty clusters is still one statement and one
        commit rather than fifty.
        """

        if not requests:
            return []
        observed_at = now or datetime.now(timezone.utc)
        results: list[tuple[object, str | None] | None] = [None] * len(requests)
        # ``ON CONFLICT DO UPDATE`` refuses to see the same key twice in
        # one statement, so a batch carrying two samples for one node has
        # to be collapsed here. The last one wins because it is the newer
        # one; the earlier ones are reported coalesced, which is what they
        # would have been had they arrived in separate batches.
        winner_by_key: dict[str, int] = {}
        for index, request in enumerate(requests):
            winner_by_key[request.spool_key()] = index
        cache = self._projected_telemetry_spool_depths(now=observed_at)
        depth = cache["depth"] + cache["admitted"]
        by_cluster = dict(cache["by_cluster"])
        for scope, admitted in cache["admitted_by_cluster"].items():
            by_cluster[scope] = by_cluster.get(scope, 0) + admitted
        admitted_rows: list[tuple[object, ...]] = []
        admitted_indexes: list[int] = []
        for index, request in enumerate(requests):
            key = request.spool_key()
            if winner_by_key[key] != index:
                results[index] = (request, "coalesced")
                continue
            scope = request.cluster_id or "__unscoped__"
            # The projection cannot tell an insert from a coalesce, so a
            # coalescing sample is charged depth it will not use. It only
            # costs admissions the burst was about to be rejected for
            # anyway, and erring the other way would let the spool grow
            # past the cap.
            if depth >= max_depth:
                results[index] = (None, "global")
                continue
            if by_cluster.get(scope, 0) >= max_cluster_depth:
                results[index] = (None, "cluster")
                continue
            depth += 1
            by_cluster[scope] = by_cluster.get(scope, 0) + 1
            cache["admitted"] += 1
            cache["admitted_by_cluster"][scope] = (
                cache["admitted_by_cluster"].get(scope, 0) + 1
            )
            admitted_rows.append(
                (
                    key,
                    request.cluster_id,
                    request.path,
                    request.request_id,
                    observed_at,
                    observed_at,
                    observed_at,
                    request.body().decode(),
                )
            )
            admitted_indexes.append(index)
        if admitted_rows:
            placeholder = "(%s, %s, %s, %s, 0, 0, NULL, %s, %s, %s, %s::jsonb)"
            values = ", ".join([placeholder] * len(admitted_rows))
            parameters: list[object] = []
            for row in admitted_rows:
                parameters.extend(row)
            with self._db.cursor() as cursor:
                cursor.execute(
                    f"""
                    INSERT INTO gpu_fault_telemetry_spool (
                        spool_key, cluster_id, path, request_id,
                        revision, attempts, lease_owner, available_at,
                        created_at, updated_at, payload
                    )
                    VALUES {values}
                    ON CONFLICT(spool_key) DO UPDATE SET
                        cluster_id=excluded.cluster_id,
                        path=excluded.path,
                        request_id=excluded.request_id,
                        revision=
                            gpu_fault_telemetry_spool.revision + 1,
                        -- ``attempts`` is deliberately kept (E-4): the
                        -- budget belongs to the lane, not the payload. A
                        -- node whose sample the endpoint cannot execute
                        -- re-sends every cycle, and resetting here let it
                        -- retry forever on Postgres while the in-memory
                        -- store -- and the test pinning the intent --
                        -- dropped it after TELEMETRY_SPOOL_MAX_ATTEMPTS.
                        -- A drop deletes the row, so the next sample
                        -- starts from zero anyway.
                        lease_owner=NULL,
                        -- Pull a leased row back into the claim window.
                        -- Whoever holds it is carrying the payload this
                        -- sample just replaced, and its completion is
                        -- fenced on the revision above, so the newer
                        -- payload is not waiting out a lease it has
                        -- already invalidated.
                        available_at=least(
                            gpu_fault_telemetry_spool.available_at,
                            excluded.available_at
                        ),
                        updated_at=excluded.updated_at,
                        payload=excluded.payload
                    RETURNING spool_key, revision
                    """,
                    parameters,
                )
                revisions = dict(cursor.fetchall())
        else:
            revisions = {}
        for index in admitted_indexes:
            request = requests[index]
            results[index] = (
                request,
                "coalesced" if revisions.get(request.spool_key(), 0) > 0 else None,
            )
        return results

    def claim_telemetry_spool(
        self,
        owner_id: str,
        *,
        now: datetime,
        lease_duration: timedelta,
        limit: int,
        max_bytes: int | None = None,
        path: str | None = None,
    ) -> list[SpooledTelemetry]:
        if limit <= 0:
            return []
        byte_limit = 2**63 - 1 if max_bytes is None else max_bytes
        if byte_limit <= 0:
            return []
        path_filter = "" if path is None else "AND path=%s"
        path_parameters = () if path is None else (path,)
        with self._db.cursor() as cursor:
            cursor.execute(
                f"""
                WITH candidates AS (
                    SELECT
                        spool_key,
                        available_at,
                        octet_length(payload::text) AS payload_bytes
                    FROM gpu_fault_telemetry_spool
                    WHERE available_at <= %s
                      {path_filter}
                    ORDER BY available_at, spool_key
                    LIMIT %s
                    FOR UPDATE SKIP LOCKED
                ),
                ranked AS (
                    SELECT
                        spool_key,
                        payload_bytes,
                        row_number() OVER (
                            ORDER BY available_at, spool_key
                        ) AS row_number,
                        sum(payload_bytes) OVER (
                            ORDER BY available_at, spool_key
                        ) AS cumulative_bytes
                    FROM candidates
                ),
                selected AS (
                    SELECT spool_key, payload_bytes
                    FROM ranked
                    WHERE cumulative_bytes <= %s
                       OR row_number=1
                )
                UPDATE gpu_fault_telemetry_spool AS spool
                SET lease_owner=%s,
                    available_at=%s,
                    attempts=spool.attempts + 1,
                    updated_at=%s
                FROM selected
                WHERE spool.spool_key=selected.spool_key
                RETURNING
                    spool.spool_key,
                    spool.revision,
                    spool.cluster_id,
                    spool.path,
                    spool.request_id,
                    spool.attempts,
                    spool.created_at,
                    spool.payload,
                    selected.payload_bytes
                """,
                (
                    now,
                    *path_parameters,
                    limit,
                    byte_limit,
                    owner_id,
                    now + lease_duration,
                    now,
                ),
            )
            rows = cursor.fetchall()
        return [
            SpooledTelemetry(
                spool_key=row[0],
                revision=row[1],
                cluster_id=row[2],
                path=row[3],
                request_id=row[4],
                attempts=row[5],
                created_at=row[6],
                payload=(
                    json.loads(row[7]) if isinstance(row[7], (str, bytes)) else row[7]
                ),
                payload_bytes=row[8],
            )
            for row in rows
        ]

    def complete_telemetry_spool(self, items: list[SpooledTelemetry]) -> int:
        if not items:
            return 0
        with self._db.cursor() as cursor:
            cursor.execute(
                """
                DELETE FROM gpu_fault_telemetry_spool AS spool
                USING (
                    SELECT
                        unnest(%s::text[]) AS spool_key,
                        unnest(%s::bigint[]) AS revision
                ) AS fence
                WHERE spool.spool_key=fence.spool_key
                  AND spool.revision=fence.revision
                """,
                (
                    [item.spool_key for item in items],
                    [item.revision for item in items],
                ),
            )
            return cursor.rowcount

    def abandon_telemetry_spool_claims(
        self,
        items: list[SpooledTelemetry],
        *,
        now: datetime,
    ) -> int:
        if not items:
            return 0
        with self._db.cursor() as cursor:
            cursor.execute(
                """
                UPDATE gpu_fault_telemetry_spool AS spool
                SET lease_owner=NULL,
                    available_at=%s,
                    attempts=greatest(0, spool.attempts - 1),
                    updated_at=%s
                FROM (
                    SELECT
                        unnest(%s::text[]) AS spool_key,
                        unnest(%s::bigint[]) AS revision
                ) AS fence
                WHERE spool.spool_key=fence.spool_key
                  AND spool.revision=fence.revision
                """,
                (
                    now,
                    now,
                    [item.spool_key for item in items],
                    [item.revision for item in items],
                ),
            )
            return cursor.rowcount

    def release_telemetry_spool(
        self,
        items: list[SpooledTelemetry],
        *,
        now: datetime,
        backoff: timedelta = timedelta(0),
        max_attempts: int | None = None,
    ) -> tuple[int, int]:
        if not items:
            return 0, 0
        limit = (
            self.TELEMETRY_SPOOL_MAX_ATTEMPTS if max_attempts is None else max_attempts
        )
        keys = [item.spool_key for item in items]
        revisions = [item.revision for item in items]
        with self._db.transaction():
            with self._db.cursor() as cursor:
                # Dropped first, so the release below only sees rows that
                # still have budget left. Both statements carry the
                # revision fence: a sample that arrived while this replay
                # was failing has already reset the row's availability and
                # its attempt count, and neither the backoff nor the drop
                # is its to inherit.
                cursor.execute(
                    """
                    DELETE FROM gpu_fault_telemetry_spool AS spool
                    USING (
                        SELECT
                            unnest(%s::text[]) AS spool_key,
                            unnest(%s::bigint[]) AS revision
                    ) AS fence
                    WHERE spool.spool_key=fence.spool_key
                      AND spool.revision=fence.revision
                      AND spool.attempts >= %s
                    """,
                    (keys, revisions, limit),
                )
                dropped = cursor.rowcount
                cursor.execute(
                    """
                    UPDATE gpu_fault_telemetry_spool AS spool
                    SET lease_owner=NULL,
                        available_at=%s,
                        updated_at=%s
                    FROM (
                        SELECT
                            unnest(%s::text[]) AS spool_key,
                            unnest(%s::bigint[]) AS revision
                    ) AS fence
                    WHERE spool.spool_key=fence.spool_key
                      AND spool.revision=fence.revision
                    """,
                    (now + backoff, now, keys, revisions),
                )
                released = cursor.rowcount
        return released, dropped

    def telemetry_spool_depths(self) -> dict[str, Any]:
        """Row counts by cluster, for the admission projection (E-2).

        Index-only on ``gpu_fault_telemetry_spool_cluster``; never touches
        ``payload``. Byte totals stay in :meth:`telemetry_spool_stats` for
        ``/metrics``.
        """

        with self._db.cursor() as cursor:
            cursor.execute(
                """
                SELECT coalesce(cluster_id, '__unscoped__'), count(*)
                FROM gpu_fault_telemetry_spool
                GROUP BY 1
                """
            )
            rows = cursor.fetchall()
        return {
            "depth": sum(row[1] for row in rows),
            "by_cluster": {row[0]: row[1] for row in rows},
        }

    def telemetry_spool_stats(self, *, now: datetime | None = None) -> dict:
        observed_at = now or datetime.now(timezone.utc)
        with self._db.cursor() as cursor:
            cursor.execute(
                """
                SELECT
                    coalesce(cluster_id, '__unscoped__'),
                    count(*),
                    count(*) FILTER (WHERE available_at > %s),
                    min(created_at),
                    coalesce(sum(octet_length(payload::text)), 0),
                    coalesce(
                        sum(octet_length(payload::text))
                            FILTER (WHERE available_at > %s),
                        0
                    )
                FROM gpu_fault_telemetry_spool
                GROUP BY 1
                """,
                (observed_at, observed_at),
            )
            rows = cursor.fetchall()
        oldest = min(
            (row[3] for row in rows if row[3] is not None),
            default=None,
        )
        return {
            "depth": sum(row[1] for row in rows),
            "leased": sum(row[2] for row in rows),
            "oldest_age_seconds": (
                0.0
                if oldest is None
                else max(
                    0.0,
                    (observed_at - oldest).total_seconds(),
                )
            ),
            "by_cluster": {row[0]: row[1] for row in rows},
            "payload_bytes": sum(row[4] for row in rows),
            "leased_bytes": sum(row[5] for row in rows),
        }

from __future__ import annotations

import hashlib
import time
from contextlib import contextmanager
from datetime import datetime
from threading import Event as ThreadEvent
from typing import Any, Callable

from gpu_fault.store.contracts import ProcessorQueueCountStatus
from gpu_fault.store.shared.cleanup_log import log_cleanup


class PostgresProcessorAdminMixin:
    # Attributes supplied by the composed concrete implementation.
    _processor_counter_mode: Callable[..., Any]
    _processor_counter_source: Callable[..., Any]

    _db: Any
    _state_transaction: Callable[..., Any]
    url: Any

    def listen_processor_queue_notifications(
        self,
        stop_event: ThreadEvent,
        owner_id: str,
        shard_count: int,
        on_notification: Callable[[str], None],
        on_state: Callable[[bool, int | None], None],
        *,
        timeout_seconds: float = 1.0,
        on_progress: Callable[[], int] | None = None,
        stall_seconds: float = 5.0,
    ) -> None:
        """Wake a consumer when a queue row becomes pending.

        Notifications are hints, not state. The processor retains a
        polling fallback, so reconnects and PostgreSQL's best-effort
        delivery cannot strand work.

        Shard ownership is tied to consumption (F-D11). The shard lock is
        session-level on this LISTEN connection, so a process whose
        consumer loop is wedged but whose listener is healthy would keep
        the shard and swallow every wakeup for it. ``on_progress`` is a
        counter the consumer advances; when payloads have been forwarded
        for ``stall_seconds`` without it moving, the shard is unlocked and
        reported as lost so a healthy process claims it within its next
        attempt. No forwarded payloads means nothing was ignored, so an
        idle owner keeps its shard. After giving a shard up this process
        waits ``stall_seconds`` before competing for one again.
        """

        import psycopg

        if shard_count <= 0:
            raise ValueError("processor notification shard count must be positive")
        if stall_seconds <= 0:
            raise ValueError("processor notification stall window must be positive")
        shard_order = sorted(
            range(shard_count),
            key=lambda shard: hashlib.blake2b(
                f"{owner_id}:{shard}".encode(),
                digest_size=8,
            ).digest(),
            reverse=True,
        )
        while not stop_event.is_set():
            try:
                with psycopg.connect(
                    self.url,
                    autocommit=True,
                    connect_timeout=5,
                ) as connection:
                    connection.execute("LISTEN gpu_fault_processor_queue")
                    owned_shard: int | None = None
                    last_claim_attempt = 0.0
                    claim_not_before = 0.0
                    progress_seen: int | None = None
                    first_forward_at: float | None = None
                    while not stop_event.is_set():
                        monotonic = time.monotonic()
                        if (
                            owned_shard is None
                            and monotonic >= claim_not_before
                            and monotonic - last_claim_attempt >= 2.0
                        ):
                            last_claim_attempt = monotonic
                            for shard in shard_order:
                                row = connection.execute(
                                    """
                                    SELECT pg_try_advisory_lock(
                                        hashtextextended(%s, 0)
                                    )
                                    """,
                                    (f"gpu_fault_processor_notify/{shard}",),
                                ).fetchone()
                                if row[0]:
                                    owned_shard = shard
                                    break
                            progress_seen = (
                                on_progress() if on_progress is not None else None
                            )
                            first_forward_at = None
                            on_state(True, owned_shard)
                        payloads = set()
                        for notification in connection.notifies(
                            timeout=timeout_seconds,
                            stop_after=256,
                        ):
                            if (
                                owned_shard is not None
                                and notification.payload not in payloads
                            ):
                                on_notification(notification.payload)
                                payloads.add(notification.payload)
                                if first_forward_at is None:
                                    first_forward_at = time.monotonic()
                            if stop_event.is_set():
                                break
                        if owned_shard is None or on_progress is None:
                            continue
                        progress = on_progress()
                        if progress != progress_seen:
                            progress_seen = progress
                            first_forward_at = None
                            continue
                        if (
                            first_forward_at is None
                            or time.monotonic() - first_forward_at < stall_seconds
                        ):
                            continue
                        # Payloads were forwarded ``stall_seconds`` ago and
                        # the consumer has not moved since: give the shard
                        # to a process that will act on it.
                        connection.execute(
                            """
                            SELECT pg_advisory_unlock(
                                hashtextextended(%s, 0)
                            )
                            """,
                            (f"gpu_fault_processor_notify/{owned_shard}",),
                        )
                        owned_shard = None
                        first_forward_at = None
                        claim_not_before = time.monotonic() + stall_seconds
                        on_state(True, None)
            except Exception:
                on_state(False, None)
                if stop_event.wait(1.0):
                    return
        on_state(False, None)

    @contextmanager
    def processor_batch_transaction(self):
        with self._db.transaction():
            yield

    def backfill_processor_queue_state_columns(self) -> int:
        with self._state_transaction("processor_queue_state/backfill"):
            with self._db.cursor() as cursor:
                cursor.execute(
                    """
                    UPDATE gpu_fault_processor_queue
                    SET
                        response_status=coalesce(
                            response_status,
                            nullif(
                                payload->>'response_status', ''
                            )::integer
                        ),
                        response_content_type=coalesce(
                            response_content_type,
                            payload->>'response_content_type'
                        ),
                        response_body_base64=coalesce(
                            response_body_base64,
                            payload->>'response_body_base64'
                        ),
                        not_before=coalesce(
                            not_before,
                            nullif(payload->>'not_before', '')::timestamptz
                        ),
                        retry_count=coalesce(
                            nullif(payload->>'retry_count', '')::integer,
                            retry_count
                        ),
                        lane_policy=coalesce(
                            nullif(payload->>'lane_policy', ''),
                            lane_policy
                        )
                    WHERE
                        response_status IS NULL
                        OR response_content_type IS NULL
                        OR response_body_base64 IS NULL
                        OR (
                            not_before IS NULL
                            AND nullif(payload->>'not_before', '') IS NOT NULL
                        )
                        OR nullif(payload->>'retry_count', '')::integer
                           IS DISTINCT FROM retry_count
                        OR coalesce(payload->>'lane_policy', 'STRICT')
                           IS DISTINCT FROM lane_policy
                    """
                )
                return cursor.rowcount

    def processor_queue_state_status(
        self,
    ) -> dict[str, int | bool]:
        with self._db.cursor() as cursor:
            cursor.execute(
                """
                SELECT
                    count(*) AS total,
                    count(*) FILTER (
                        WHERE payload->>'status'
                              IS DISTINCT FROM status
                    ) AS status_mismatch,
                    count(*) FILTER (
                        WHERE payload->>'lease_owner'
                              IS DISTINCT FROM lease_owner
                    ) AS lease_owner_mismatch,
                    count(*) FILTER (
                        WHERE nullif(
                                  payload->>'leader_epoch', ''
                              )::bigint
                              IS DISTINCT FROM leader_epoch
                    ) AS leader_epoch_mismatch,
                    count(*) FILTER (
                        WHERE payload->>'lease_token'
                              IS DISTINCT FROM lease_token
                    ) AS lease_token_mismatch,
                    count(*) FILTER (
                        WHERE nullif(
                                  payload->>'lease_expires_at', ''
                              )::timestamptz
                              IS DISTINCT FROM lease_expires_at
                    ) AS lease_expiry_mismatch,
                    count(*) FILTER (
                        WHERE nullif(
                                  payload->>'response_status', ''
                              )::integer
                              IS NOT NULL
                          AND response_status IS NULL
                    ) AS response_missing,
                    count(*) FILTER (
                        WHERE response_status IS NOT NULL
                          AND nullif(
                                  payload->>'response_status', ''
                              )::integer
                              IS DISTINCT FROM response_status
                    ) AS response_status_mismatch,
                    count(*) FILTER (
                        WHERE response_content_type IS NOT NULL
                          AND payload->>'response_content_type'
                              IS DISTINCT FROM
                                  response_content_type
                    ) AS response_content_type_mismatch,
                    count(*) FILTER (
                        WHERE response_body_base64 IS NOT NULL
                          AND payload->>'response_body_base64'
                              IS DISTINCT FROM
                                  response_body_base64
                    ) AS response_body_mismatch,
                    count(*) FILTER (
                        WHERE nullif(payload->>'not_before', '')::timestamptz
                              IS DISTINCT FROM not_before
                    ) AS not_before_mismatch,
                    count(*) FILTER (
                        WHERE coalesce(
                                  nullif(payload->>'retry_count', '')::integer,
                                  0
                              )
                              IS DISTINCT FROM retry_count
                    ) AS retry_count_mismatch,
                    count(*) FILTER (
                        WHERE coalesce(payload->>'lane_policy', 'STRICT')
                              IS DISTINCT FROM lane_policy
                    ) AS lane_policy_mismatch
                FROM gpu_fault_processor_queue
                """
            )
            row = cursor.fetchone()
        names = [
            "total",
            "status_mismatch",
            "lease_owner_mismatch",
            "leader_epoch_mismatch",
            "lease_token_mismatch",
            "lease_expiry_mismatch",
            "response_missing",
            "response_status_mismatch",
            "response_content_type_mismatch",
            "response_body_mismatch",
            "not_before_mismatch",
            "retry_count_mismatch",
            "lane_policy_mismatch",
        ]
        result = {name: int(value) for name, value in zip(names, row, strict=True)}
        result["ready"] = all(result[name] == 0 for name in names[1:])
        return result

    def processor_counter_mode(self) -> str:
        """Which table admission reads the per-cluster depth from.

        ``dual`` (the shipped default) keeps the single-row
        ``gpu_fault_processor_queue_counts`` authoritative and writes the 16
        priority shards alongside, so every enqueue still locks one row per
        cluster and the shards buy no contention relief. Only
        ``partitioned`` reads the shards. The mode is invisible from
        outside otherwise -- production sat in ``dual`` unnoticed (store
        review 2026-09-07, item I) -- so /metrics exports this. Served from
        the 0.25 s cache the admission path uses.
        """
        return str(self._processor_counter_mode())

    def processor_queue_count_status(
        self,
    ) -> ProcessorQueueCountStatus:
        with self._db.cursor() as cursor:
            source = self._processor_counter_source(
                self._processor_counter_mode(cursor)
            )
            cursor.execute(
                f"""
                WITH expected AS (
                    SELECT
                        coalesce(cluster_id, '__unscoped__')
                            AS cluster_id,
                        count(*) AS incomplete_count
                    FROM gpu_fault_processor_queue
                    WHERE status IN ('PENDING', 'LEASED')
                    GROUP BY coalesce(
                        cluster_id, '__unscoped__'
                    )
                ),
                actual AS (
                    SELECT cluster_id, sum(incomplete_count)
                        AS incomplete_count
                    FROM ({source}) AS counter_rows
                    GROUP BY cluster_id
                ),
                compared AS (
                    SELECT
                        coalesce(
                            expected.cluster_id,
                            actual.cluster_id
                        ) AS cluster_id,
                        coalesce(
                            expected.incomplete_count, 0
                        ) AS expected_count,
                        coalesce(
                            actual.incomplete_count, 0
                        ) AS actual_count
                    FROM expected
                    FULL OUTER JOIN actual
                    USING(cluster_id)
                )
                SELECT
                    coalesce(sum(expected_count), 0),
                    coalesce(sum(actual_count), 0),
                    count(*) FILTER (
                        WHERE expected_count <> actual_count
                    )
                FROM compared
                """
            )
            expected, actual, mismatched = cursor.fetchone()
        return {
            "expected_total": int(expected),
            "counter_total": int(actual),
            "mismatched_clusters": int(mismatched),
            "ready": int(mismatched) == 0,
        }

    def _switch_processor_counter_mode(
        self,
        mode: str,
        *,
        require_empty: bool,
    ) -> dict:
        if mode not in {"dual", "partitioned"}:
            raise ValueError("invalid processor counter mode")
        with self._db.transaction():
            with self._db.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT pg_advisory_xact_lock(
                        hashtextextended(%s, 0)
                    )
                    """,
                    ("gpu_fault_processor_counter_mode",),
                )
                cursor.execute(
                    """
                    LOCK TABLE gpu_fault_processor_queue
                    IN SHARE ROW EXCLUSIVE MODE
                    """
                )
                cursor.execute(
                    """
                    SELECT count(*)
                    FROM gpu_fault_processor_queue
                    WHERE status IN ('PENDING', 'LEASED')
                    """
                )
                incomplete = int(cursor.fetchone()[0])
                if require_empty and incomplete:
                    raise RuntimeError(
                        "processor counter mode switch requires an "
                        f"empty queue; found {incomplete} incomplete rows"
                    )
                cursor.execute(
                    """
                    INSERT INTO gpu_fault_processor_counter_mode(
                        singleton, mode, updated_at
                    ) VALUES (TRUE, %s, now())
                    ON CONFLICT(singleton) DO UPDATE SET
                        mode=excluded.mode,
                        updated_at=excluded.updated_at
                    """,
                    (mode,),
                )
                # Replayed in both directions. A process whose 0.25 s mode
                # cache still says ``dual`` after the switch to partitioned
                # reads this table; left empty it answered depth 0 and that
                # process admitted without limit for the cache window
                # (F-D10). Filled, it holds the depth that was true at the
                # switch -- stale, never zero.
                cursor.execute("TRUNCATE gpu_fault_processor_queue_counts")
                cursor.execute(
                    """
                    INSERT INTO gpu_fault_processor_queue_counts(
                        cluster_id, incomplete_count, updated_at
                    )
                    SELECT
                        coalesce(cluster_id, '__unscoped__'),
                        count(*),
                        now()
                    FROM gpu_fault_processor_queue
                    WHERE status IN ('PENDING', 'LEASED')
                    GROUP BY coalesce(
                        cluster_id, '__unscoped__'
                    )
                    """
                )
                cursor.execute("TRUNCATE gpu_fault_processor_priority_count_shards")
                cursor.execute(
                    """
                    INSERT INTO
                        gpu_fault_processor_priority_count_shards(
                            cluster_id,
                            priority_bucket,
                            shard_id,
                            incomplete_count,
                            updated_at
                        )
                    SELECT
                        coalesce(cluster_id, '__unscoped__'),
                        (
                            CASE
                                WHEN priority <= 10 THEN 0
                                WHEN priority <= 50 THEN 50
                                ELSE 100
                            END
                        )::smallint,
                        mod(
                            (
                                hashtextextended(request_id, 0)
                                & 9223372036854775807
                            ),
                            16
                        )::smallint,
                        count(*),
                        now()
                    FROM gpu_fault_processor_queue
                    WHERE status IN ('PENDING', 'LEASED')
                    GROUP BY 1, 2, 3
                    """
                )
        self._processor_counter_mode_cache = mode
        self._processor_counter_mode_checked_at = time.monotonic()
        return {
            "mode": mode,
            "incomplete_rows": incomplete,
            **self.processor_queue_count_status(),
        }

    def finalize_processor_counter_shards(self) -> dict:
        return self._switch_processor_counter_mode(
            "partitioned",
            require_empty=True,
        )

    def restore_legacy_processor_counters(self) -> dict:
        return self._switch_processor_counter_mode(
            "dual",
            require_empty=False,
        )

    def cleanup_completed_processor_requests(
        self,
        *,
        older_than: datetime,
        limit: int,
    ) -> int:
        with self._state_transaction("processor_request/cleanup"):
            with self._db.cursor() as cursor:
                cursor.execute(
                    """
                    WITH victims AS (
                        SELECT request_id
                        FROM gpu_fault_processor_queue
                        WHERE status='COMPLETED'
                          AND updated_at <= %s
                        ORDER BY updated_at, request_id
                        LIMIT %s
                        FOR UPDATE SKIP LOCKED
                    ),
                    deleted AS (
                        DELETE FROM gpu_fault_processor_queue AS queue
                        USING victims
                        WHERE queue.request_id=victims.request_id
                        RETURNING queue.request_id
                    )
                    SELECT request_id FROM deleted
                    """,
                    (older_than, limit),
                )
                request_ids = [row[0] for row in cursor.fetchall()]
                if request_ids:
                    cursor.execute(
                        """
                        DELETE FROM gpu_fault_objects
                        WHERE kind='processor_request'
                          AND key = ANY(%s)
                        """,
                        (request_ids,),
                    )
            return log_cleanup("processor_request", request_ids)

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any, Callable

from gpu_fault.processor.models import RESERVED_TIER_MAX_PRIORITY
from gpu_fault.store.contracts import ProcessorQueueStats
from gpu_fault.store.shared.errors import NotFoundError


class PostgresProcessorStorageMixin:
    # Attributes supplied by the composed concrete implementation.
    processor_queue_state_mode: Any

    _db: Any
    _decode: Callable[..., Any]
    _state_transaction: Callable[..., Any]
    _PROCESSOR_QUEUE_COLUMNS = (
        "request_id",
        "status",
        "cluster_id",
        "correlation_key",
        "ordering_key",
        "priority",
        "lease_owner",
        "leader_epoch",
        "lease_token",
        "lease_expires_at",
        "response_status",
        "response_content_type",
        "response_body_base64",
        "not_before",
        "retry_count",
        "lane_policy",
        "created_at",
        "updated_at",
        "payload",
    )

    @staticmethod
    def _processor_queue_row(request) -> tuple[object, ...]:
        return (
            request.request_id,
            request.status.value,
            request.cluster_id,
            request.correlation_key,
            request.ordering_key(),
            request.queue_priority(),
            request.lease_owner,
            request.leader_epoch,
            request.lease_token,
            request.lease_expires_at,
            request.response_status,
            request.response_content_type,
            request.response_body_base64,
            request.not_before,
            request.retry_count,
            request.lane_policy.value,
            request.created_at,
            request.updated_at,
            request.model_dump_json(),
        )

    def _put_processor_queue_many(
        self, requests, *, update_payload: bool = True
    ) -> None:
        """Insert a whole admission group in one statement.

        One statement, not one per request. The count trigger is
        ``FOR EACH STATEMENT`` and takes ``FOR UPDATE`` on the cluster's
        counter row, and that lock is held until this transaction
        commits - so a group that inserted its rows one at a time paid
        the trigger, eight partial index updates and a round trip per
        row while every other admission and completion for the same
        cluster queued behind it. A 50-cluster burst had ~200 backends
        blocked on ``gpu_fault_processor_queue_counts`` tuple locks with
        the longest waiter at 43s, at 6% database CPU.

        Request ids are unique within a group (the caller keys them by
        base id), which ON CONFLICT DO UPDATE requires: the same key
        twice in one statement is an error, not a second update.
        """

        rows = sorted(
            (self._processor_queue_row(request) for request in requests),
            key=lambda row: row[0],
        )
        if not rows:
            return
        if len({row[0] for row in rows}) != len(rows):
            raise ValueError("processor queue batch has duplicate request ids")
        columns = ", ".join(self._PROCESSOR_QUEUE_COLUMNS)
        placeholder = "(" + ", ".join(["%s"] * 18) + ", %s::jsonb)"
        values = ", ".join([placeholder] * len(rows))
        payload_update = ", payload=excluded.payload" if update_payload else ""
        parameters: list[object] = []
        for row in rows:
            parameters.extend(row)
        with self._db.cursor() as cursor:
            cursor.execute(
                f"""
                INSERT INTO gpu_fault_processor_queue (
                    {columns}
                )
                VALUES {values}
                ON CONFLICT(request_id) DO UPDATE SET
                    status=excluded.status,
                    cluster_id=excluded.cluster_id,
                    correlation_key=excluded.correlation_key,
                    ordering_key=excluded.ordering_key,
                    priority=excluded.priority,
                    lease_owner=excluded.lease_owner,
                    leader_epoch=excluded.leader_epoch,
                    lease_token=excluded.lease_token,
                    lease_expires_at=excluded.lease_expires_at,
                    response_status=excluded.response_status,
                    response_content_type=excluded.response_content_type,
                    response_body_base64=
                        excluded.response_body_base64,
                    not_before=excluded.not_before,
                    retry_count=excluded.retry_count,
                    lane_policy=excluded.lane_policy,
                    created_at=excluded.created_at,
                    updated_at=excluded.updated_at
                    {payload_update}
                WHERE excluded.updated_at
                      >= gpu_fault_processor_queue.updated_at
                """,
                parameters,
            )

    def _put_processor_queue(self, request, *, update_payload: bool = True) -> None:
        self._put_processor_queue_many([request], update_payload=update_payload)

    @staticmethod
    def _processor_queue_effective_payload(
        alias: str = "gpu_fault_processor_queue",
    ) -> str:
        prefix = f"{alias}."
        return f"""
            {prefix}payload || jsonb_build_object(
                'status', {prefix}status,
                'lease_owner', to_jsonb({prefix}lease_owner),
                'leader_epoch', to_jsonb({prefix}leader_epoch),
                'lease_token', to_jsonb({prefix}lease_token),
                'lease_expires_at',
                    to_jsonb({prefix}lease_expires_at),
                'response_status',
                    coalesce(
                        to_jsonb({prefix}response_status),
                        {prefix}payload->'response_status',
                        'null'::jsonb
                    ),
                'response_content_type',
                    coalesce(
                        to_jsonb({prefix}response_content_type),
                        {prefix}payload->'response_content_type',
                        'null'::jsonb
                    ),
                'response_body_base64',
                    coalesce(
                        to_jsonb({prefix}response_body_base64),
                        {prefix}payload->'response_body_base64',
                        'null'::jsonb
                    ),
                'not_before', to_jsonb({prefix}not_before),
                'retry_count', to_jsonb({prefix}retry_count),
                'lane_policy', to_jsonb({prefix}lane_policy),
                'updated_at', to_jsonb({prefix}updated_at)
            )
        """

    def _persist_processor_request(
        self, request, *, update_payload: bool = True
    ) -> None:
        self._put_processor_queue(request, update_payload=update_payload)

    def _persist_processor_state(self, request) -> None:
        self._persist_processor_request(
            request,
            update_payload=(self.processor_queue_state_mode != "dedicated"),
        )

    def enqueue_processor_request(self, request):
        with self._state_transaction(f"processor_request/{request.request_id}"):
            try:
                return self.get_processor_request(request.request_id)
            except NotFoundError:
                self._persist_processor_request(request)
                return request

    def _processor_counter_mode(self, cursor=None) -> str:
        monotonic = time.monotonic()
        if monotonic - self._processor_counter_mode_checked_at < 0.25:
            return self._processor_counter_mode_cache
        if cursor is None:
            with self._db.cursor() as owned:
                return self._processor_counter_mode(owned)
        cursor.execute(
            """
            SELECT mode
            FROM gpu_fault_processor_counter_mode
            WHERE singleton=TRUE
            """
        )
        row = cursor.fetchone()
        mode = row[0] if row is not None else "dual"
        self._processor_counter_mode_cache = mode
        self._processor_counter_mode_checked_at = monotonic
        return mode

    @staticmethod
    def _processor_counter_source(mode: str) -> str:
        legacy = (
            "SELECT cluster_id, incomplete_count FROM gpu_fault_processor_queue_counts"
        )
        if mode == "dual":
            return legacy
        return (
            "SELECT cluster_id, incomplete_count "
            "FROM gpu_fault_processor_priority_count_shards"
        )

    def _processor_counter_depths(
        self,
        cursor,
        scopes: list[str],
    ) -> tuple[dict[str, int], int]:
        mode = self._processor_counter_mode(cursor)
        source = self._processor_counter_source(mode)
        cursor.execute(
            f"""
            WITH counts AS ({source})
            SELECT cluster_id, coalesce(sum(incomplete_count), 0)
            FROM counts
            WHERE cluster_id=ANY(%s)
            GROUP BY cluster_id
            """,
            (scopes,),
        )
        depths = {scope: 0 for scope in scopes}
        depths.update(
            {cluster_id: int(count) for cluster_id, count in cursor.fetchall()}
        )
        cursor.execute(
            f"""
            WITH counts AS ({source})
            SELECT coalesce(sum(incomplete_count), 0)
            FROM counts
            """
        )
        return depths, int(cursor.fetchone()[0])

    def processor_queue_stats(
        self, *, now: datetime | None = None
    ) -> ProcessorQueueStats:
        observed_at = now or datetime.now(timezone.utc)
        with self._db.cursor() as cursor:
            source = self._processor_counter_source(
                self._processor_counter_mode(cursor)
            )
            cursor.execute(
                f"""
                WITH counts AS ({source})
                SELECT cluster_id, sum(incomplete_count)
                FROM counts
                GROUP BY cluster_id
                HAVING sum(incomplete_count) > 0
                """
            )
            rows = cursor.fetchall()
            # One round trip for every cluster's oldest incomplete request; the
            # region-wide age is the max of these, so it needs no query of its
            # own. NULL cluster ids are grouped under the same key the counter
            # tables and the in-memory backend use for unscoped requests.
            cursor.execute(
                """
                SELECT coalesce(cluster_id, '__unscoped__'), min(created_at)
                FROM gpu_fault_processor_queue
                WHERE status IN ('PENDING', 'LEASED')
                GROUP BY coalesce(cluster_id, '__unscoped__')
                """
            )
            oldest_rows = cursor.fetchall()
        by_cluster = {row[0]: row[1] for row in rows}
        oldest_age_by_cluster: dict[str, float] = {
            str(cluster_id): max(0.0, (observed_at - created_at).total_seconds())
            for cluster_id, created_at in oldest_rows
            if created_at is not None
        }
        return {
            "depth": sum(by_cluster.values()),
            "oldest_age_seconds": max(oldest_age_by_cluster.values(), default=0.0),
            "by_cluster": by_cluster,
            "oldest_age_by_cluster": oldest_age_by_cluster,
        }

    def processor_fault_backlog_depth(self) -> int:
        with self._db.cursor() as cursor:
            cursor.execute(
                """
                SELECT count(*)
                FROM gpu_fault_processor_queue
                WHERE status IN ('PENDING', 'LEASED')
                  AND priority <= %s
                """,
                (RESERVED_TIER_MAX_PRIORITY,),
            )
            return cursor.fetchone()[0]

    def get_processor_request(self, request_id: str):
        with self._db.cursor() as cursor:
            cursor.execute(
                f"""
                SELECT {
                    self._processor_queue_effective_payload("gpu_fault_processor_queue")
                }
                FROM gpu_fault_processor_queue
                WHERE request_id=%s
                """,
                (request_id,),
            )
            row = cursor.fetchone()
        if row is not None:
            return self._decode("processor_request", row[0])
        raise NotFoundError(request_id)

    def has_incomplete_processor_requests(self, cluster_id: str) -> bool:
        with self._db.cursor() as cursor:
            cursor.execute(
                """
                SELECT EXISTS (
                    SELECT 1
                    FROM gpu_fault_processor_queue
                    WHERE cluster_id=%s
                      AND status IN ('PENDING', 'LEASED')
                )
                """,
                (cluster_id,),
            )
            return cursor.fetchone()[0]

    def has_incomplete_processor_requests_for_scopes(
        self, cluster_id: str, scope_keys: set[str]
    ) -> bool:
        if not scope_keys:
            return self.has_incomplete_processor_requests(cluster_id)
        with self._db.cursor() as cursor:
            cursor.execute(
                """
                SELECT EXISTS (
                    SELECT 1
                    FROM gpu_fault_processor_queue
                    WHERE cluster_id=%s
                      AND status IN ('PENDING', 'LEASED')
                      AND payload->'correlation_scope_keys' ?| %s
                )
                """,
                (cluster_id, sorted(scope_keys)),
            )
            return cursor.fetchone()[0]

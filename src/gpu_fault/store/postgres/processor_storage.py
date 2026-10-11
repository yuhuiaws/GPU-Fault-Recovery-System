from __future__ import annotations

import base64
import time
from datetime import datetime, timezone
from typing import Any, Callable

from gpu_fault.processor.models import RESERVED_TIER_MAX_PRIORITY
from gpu_fault.store.contracts import ProcessorQueueStats
from gpu_fault.store.postgres.ddl_processor_layout import (
    LAYOUT_DETACHED,
    PAYLOADS_TABLE,
    REPLAYS_TABLE,
)
from gpu_fault.store.postgres.hot_layout import (
    read_layout_mode,
    read_layout_mode_for_write,
)
from gpu_fault.store.shared.errors import NotFoundError

#: The payload fields that leave the queue row in the detached layout: the
#: request body lives in ``gpu_fault_processor_payloads``, the response body
#: in the replay row.
DETACHED_PAYLOAD_EXCLUDE = frozenset({"body_base64", "response_body_base64"})
#: What the detached queue row's payload still carries: the fields that have
#: no column of their own. Every other field is merged back from its column
#: by ``_processor_queue_state_json`` (the previous release's readers never
#: see a detached row: the layout is switched only after they are gone), so
#: a queue row is about half a kilobyte and three versions of it fit the
#: fillfactor room of one page.
DETACHED_PAYLOAD_FIELDS = frozenset(
    {
        "request_id",
        "method",
        "path",
        "query",
        "content_type",
        "execution_authorized",
        "correlation_scope_keys",
    }
)


def detached_payload_json(request: Any) -> str:
    return str(request.model_dump_json(include=set(DETACHED_PAYLOAD_FIELDS)))


_BODY_B64_SQL = "replace(encode({alias}.body, 'base64'), E'\\n', '')"


def _b64_bytes(value: str | None) -> bytes:
    return base64.b64decode(value) if value else b""


class PostgresProcessorStorageMixin:
    # Attributes supplied by the composed concrete implementation.
    processor_queue_state_mode: Any

    _db: Any
    _decode: Callable[..., Any]
    _state_transaction: Callable[..., Any]
    _hot_layout_mode_cache: str = "inline"
    _hot_layout_mode_checked_at: float = 0.0
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
        # Schema v23: the two payload fields the hot statements read, as
        # columns, so no index has to reference the payload.
        "path",
        "scope_keys",
    )

    # -- layout mode -------------------------------------------------------

    def _hot_layout_mode_for_write(self, cursor: Any) -> str:
        """The layout this write transaction uses (shared lock on the mode key;
        read before any row lock so the switch cannot deadlock a writer)."""

        mode = read_layout_mode_for_write(cursor)
        self._hot_layout_mode_cache = mode
        self._hot_layout_mode_checked_at = time.monotonic()
        return mode

    def _hot_layout_mode(self, cursor: Any = None) -> str:
        """The cached mode for readers and reports (every read path reads both
        layouts, so a 0.25 s stale answer costs nothing)."""

        monotonic = time.monotonic()
        if monotonic - self._hot_layout_mode_checked_at < 0.25:
            return self._hot_layout_mode_cache
        if cursor is None:
            with self._db.cursor() as owned:
                return self._hot_layout_mode(owned)
        mode = read_layout_mode(cursor)
        self._hot_layout_mode_cache = mode
        self._hot_layout_mode_checked_at = monotonic
        return mode

    def hot_layout_mode(self) -> str:
        """``inline`` (the v22 row layout) or ``detached`` (schema v23's)."""

        return str(self._hot_layout_mode())

    # -- rows --------------------------------------------------------------

    @staticmethod
    def _processor_queue_row(request: Any) -> tuple[object, ...]:
        """One queue row in the inline layout (the full payload); the writer
        swaps in the slim payload for the detached layout."""

        payload = request.model_dump_json()
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
            payload,
            request.path,
            list(request.correlation_scope_keys),
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

        In the detached layout (schema v23) the same statement also writes
        the request bodies into ``gpu_fault_processor_payloads`` through a
        data-modifying CTE, so the group is still one round trip and the
        queue row carries no body.

        Request ids are unique within a group (the caller keys them by
        base id), which ON CONFLICT DO UPDATE requires: the same key
        twice in one statement is an error, not a second update.
        """

        if not requests:
            return
        with self._db.cursor() as cursor:
            detached = self._hot_layout_mode_for_write(cursor) == LAYOUT_DETACHED
            rows = sorted(
                (self._processor_queue_row(request) for request in requests),
                key=lambda row: row[0],
            )
            if detached:
                by_request = {request.request_id: request for request in requests}
                rows = [
                    (
                        *row[:18],
                        detached_payload_json(by_request[str(row[0])]),
                        *row[19:],
                    )
                    for row in rows
                ]
            if len({row[0] for row in rows}) != len(rows):
                raise ValueError("processor queue batch has duplicate request ids")
            columns = ", ".join(self._PROCESSOR_QUEUE_COLUMNS)
            placeholder = "(" + ", ".join(["%s"] * 18) + ", %s::jsonb, %s, %s::text[])"
            values = ", ".join([placeholder] * len(rows))
            payload_update = ", payload=excluded.payload" if update_payload else ""
            parameters: list[object] = []
            for row in rows:
                parameters.extend(row)
            upsert = f"""
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
                    updated_at=excluded.updated_at,
                    path=excluded.path,
                    scope_keys=excluded.scope_keys
                    {payload_update}
                WHERE excluded.updated_at
                      >= gpu_fault_processor_queue.updated_at
            """
            if not (detached and update_payload):
                cursor.execute(upsert, parameters)
                return
            by_id = {request.request_id: request for request in requests}
            ids = [row[0] for row in rows]
            bodies = [_b64_bytes(by_id[request_id].body_base64) for request_id in ids]
            cursor.execute(
                f"""
                WITH queued AS (
                    {upsert}
                    RETURNING request_id
                )
                INSERT INTO {PAYLOADS_TABLE} (request_id, body)
                SELECT bodies.request_id, bodies.body
                FROM unnest(%s::text[], %s::bytea[]) AS bodies(request_id, body)
                JOIN queued ON queued.request_id=bodies.request_id
                ON CONFLICT(request_id) DO UPDATE SET body=excluded.body
                """,
                [*parameters, ids, bodies],
            )

    def _put_processor_queue(self, request, *, update_payload: bool = True) -> None:
        self._put_processor_queue_many([request], update_payload=update_payload)

    @staticmethod
    def _processor_queue_state_json(alias: str) -> str:
        """The column-authoritative fields merged over the stored payload."""

        prefix = f"{alias}."
        return f"""jsonb_build_object(
                'status', {prefix}status,
                'cluster_id', to_jsonb({prefix}cluster_id),
                'correlation_key', to_jsonb({prefix}correlation_key),
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
                'created_at', to_jsonb({prefix}created_at),
                'updated_at', to_jsonb({prefix}updated_at),
                'path', coalesce(to_jsonb({prefix}path), {prefix}payload->'path'),
                'correlation_scope_keys',
                    coalesce(
                        to_jsonb({prefix}scope_keys),
                        {prefix}payload->'correlation_scope_keys',
                        '[]'::jsonb
                    )
            )"""

    @classmethod
    def _processor_queue_effective_payload(
        cls,
        alias: str = "gpu_fault_processor_queue",
        payload_alias: str | None = "gpu_fault_processor_payloads",
    ) -> str:
        """The JSON a ``ProcessorRequest`` is decoded from: stored payload,
        column-authoritative state, and the body from the payloads row (detached
        layout) or from the payload itself (inline layout).

        Every statement that renders this must ``LEFT JOIN
        gpu_fault_processor_payloads`` under ``payload_alias`` on the row's
        request id; ``None`` is for statements over rows that provably carry
        their body inline.
        """

        prefix = f"{alias}."
        body = f"{prefix}payload->'body_base64'"
        if payload_alias is not None:
            encoded = _BODY_B64_SQL.format(alias=payload_alias)
            body = f"coalesce(to_jsonb({encoded}), {body})"
        return f"""
            {prefix}payload || {cls._processor_queue_state_json(alias)}
            || jsonb_build_object(
                'body_base64', coalesce({body}, '""'::jsonb)
            )
        """

    @staticmethod
    def _processor_replay_effective_payload(
        alias: str = "gpu_fault_processor_replays",
    ) -> str:
        prefix = f"{alias}."
        body = _BODY_B64_SQL.format(alias=alias)
        response = "replace(encode({p}response_body, 'base64'), E'\\n', '')".format(
            p=prefix
        )
        return f"""
            {prefix}envelope || jsonb_build_object(
                'body_base64', coalesce(to_jsonb({body}), '""'::jsonb),
                'response_status', to_jsonb({prefix}response_status),
                'response_content_type', to_jsonb({prefix}response_content_type),
                'response_body_base64', to_jsonb({response})
            )
        """

    @staticmethod
    def _processor_payloads_join(
        alias: str, payload_alias: str = "gpu_fault_processor_payloads"
    ) -> str:
        return (
            f"LEFT JOIN {PAYLOADS_TABLE} AS {payload_alias} "
            f"ON {payload_alias}.request_id={alias}.request_id"
        )

    def _persist_processor_request(
        self, request, *, update_payload: bool = True
    ) -> None:
        self._put_processor_queue(request, update_payload=update_payload)

    def _persist_processor_state(self, request) -> None:
        from gpu_fault.processor import ProcessorRequestStatus

        if request.status is ProcessorRequestStatus.COMPLETED:
            with self._db.cursor() as cursor:
                if self._hot_layout_mode_for_write(cursor) == LAYOUT_DETACHED:
                    self._move_processor_request_to_replays(cursor, request)
                    return
        self._persist_processor_request(
            request,
            update_payload=(self.processor_queue_state_mode != "dedicated"),
        )

    def _move_processor_request_to_replays(self, cursor: Any, request: Any) -> None:
        """A completion in the detached layout: the queue row and its payload
        row go, the replay row (envelope, bodies, response) comes.

        The queue DELETE fires the statement-level count trigger exactly as
        the sweep's DELETE did, so a LEASED row leaving the queue releases
        its admission slot.
        """

        envelope = request.model_dump_json(exclude=set(DETACHED_PAYLOAD_EXCLUDE))
        cursor.execute(
            f"""
            WITH gone AS (
                DELETE FROM gpu_fault_processor_queue
                WHERE request_id=%s
                RETURNING request_id
            ),
            bodies AS (
                DELETE FROM {PAYLOADS_TABLE}
                WHERE request_id=%s
                RETURNING body
            )
            INSERT INTO {REPLAYS_TABLE} (
                request_id, cluster_id, completed_at, envelope, body,
                response_status, response_content_type, response_body
            )
            SELECT %s, %s, %s, %s::jsonb,
                   coalesce((SELECT body FROM bodies), %s::bytea),
                   %s, %s, %s::bytea
            ON CONFLICT (request_id, completed_at) DO UPDATE SET
                envelope=excluded.envelope,
                body=excluded.body,
                response_status=excluded.response_status,
                response_content_type=excluded.response_content_type,
                response_body=excluded.response_body
            """,
            (
                request.request_id,
                request.request_id,
                request.request_id,
                request.cluster_id,
                request.updated_at,
                envelope,
                _b64_bytes(request.body_base64),
                request.response_status,
                request.response_content_type,
                _b64_bytes(request.response_body_base64),
            ),
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
        """The queue row first, then the replay row a detached-layout
        completion left behind (same ``request_id``, same response, for the
        completed-request retention window)."""

        with self._db.cursor() as cursor:
            cursor.execute(
                f"""
                SELECT {
                    self._processor_queue_effective_payload(
                        "gpu_fault_processor_queue", "payloads"
                    )
                }
                FROM gpu_fault_processor_queue
                {self._processor_payloads_join("gpu_fault_processor_queue", "payloads")}
                WHERE gpu_fault_processor_queue.request_id=%s
                """,
                (request_id,),
            )
            row = cursor.fetchone()
            if row is None:
                cursor.execute(
                    f"""
                    SELECT {self._processor_replay_effective_payload()}
                    FROM {REPLAYS_TABLE}
                    WHERE request_id=%s
                    ORDER BY completed_at DESC
                    LIMIT 1
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
                      AND cardinality(scope_keys) > 0
                      AND scope_keys && %s::text[]
                )
                """,
                (cluster_id, sorted(scope_keys)),
            )
            return cursor.fetchone()[0]

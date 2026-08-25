from __future__ import annotations

from typing import Any, Callable, TypedDict

from datetime import datetime, timezone
from threading import Event as ThreadEvent

from gpu_fault.store.shared.time import (
    utc_text as _utc_text,
)


class _CompletionEntry(TypedDict):
    args: tuple[str, str, int, str]
    kwargs: dict[str, Any]
    event: ThreadEvent
    result: Any
    error: BaseException | None


class PostgresProcessorCompletionMixin:
    # Attributes supplied by the composed concrete implementation.
    get_processor_request: Callable[..., Any]

    _db: Any
    _decode: Callable[..., Any]
    _lock_processor_queue_row: Callable[..., Any]
    _persist_processor_state: Callable[..., Any]
    _processor_completion_condition: Any
    _processor_completion_queue: list[_CompletionEntry]
    _processor_queue_effective_payload: Callable[..., Any]
    _state_transaction: Callable[..., Any]
    get_processor_leadership: Callable[..., Any]
    processor_queue_state_mode: Any

    def complete_active_processor_request(
        self,
        request_id: str,
        owner_id: str,
        lane_epoch: int,
        lease_token: str,
        *,
        response_status: int,
        response_content_type: str | None,
        response_body_base64: str,
        path: str | None = None,
    ):
        # Only observation completions are worth group-committing, and the
        # caller already knows the path. Reading the row here just to
        # learn it costs an extra commit on every other request, because
        # the pool is autocommit.
        if path is None:
            path = self.get_processor_request(request_id).path
        if path != "/v1/workload-observations":
            return self._complete_active_processor_request_now(
                request_id,
                owner_id,
                lane_epoch,
                lease_token,
                response_status=response_status,
                response_content_type=response_content_type,
                response_body_base64=response_body_base64,
            )
        entry: _CompletionEntry = {
            "args": (
                request_id,
                owner_id,
                lane_epoch,
                lease_token,
            ),
            "kwargs": {
                "response_status": response_status,
                "response_content_type": response_content_type,
                "response_body_base64": response_body_base64,
            },
            "event": ThreadEvent(),
            "result": None,
            "error": None,
        }
        with self._processor_completion_condition:
            leader = not self._processor_completion_queue
            self._processor_completion_queue.append(entry)
            if not leader:
                self._processor_completion_condition.notify()
        if leader:
            with self._processor_completion_condition:
                self._processor_completion_condition.wait(timeout=0.01)
                batch = self._processor_completion_queue[:64]
                del self._processor_completion_queue[: len(batch)]
            try:
                completed = self._complete_active_processor_requests_batch(batch)
                for item in batch:
                    request_id = item["args"][0]
                    if request_id in completed:
                        item["result"] = completed[request_id]
                    else:
                        item["error"] = ValueError("stale processor lane fencing token")
            except Exception as exc:
                for item in batch:
                    item["error"] = exc
            for item in batch:
                item["event"].set()
        if not entry["event"].wait(timeout=30):
            raise TimeoutError("processor completion batch did not flush")
        if entry["error"] is not None:
            raise entry["error"]
        return entry["result"]

    def _completions_by_counter_scope(self, batch) -> dict:
        """Split a completion batch the way the count trigger locks.

        A claim window is filled from the whole region, so one telemetry
        batch of 64 requests carries work for as many clusters as it
        found - and the count trigger locks the counter row of every one
        of them until the statement's transaction commits. Every other
        completion statement in the fleet spans the same clusters, so no
        two of them could ever run at the same time.
        """
        request_ids = [item["args"][0] for item in batch]
        with self._db.cursor() as cursor:
            cursor.execute(
                """
                SELECT
                    request_id,
                    coalesce(cluster_id, '__unscoped__')
                FROM gpu_fault_processor_queue
                WHERE request_id=ANY(%s)
                """,
                (request_ids,),
            )
            scopes = dict(cursor.fetchall())
        grouped: dict[str, list] = {}
        for item in batch:
            # A request the queue no longer has cannot be completed at
            # all; it goes in a group of its own scope so the statement
            # below reports it missing exactly as it did before.
            scope = scopes.get(item["args"][0], "__unscoped__")
            grouped.setdefault(scope, []).append(item)
        return grouped

    def _complete_active_processor_requests_batch(self, batch) -> dict[str, object]:
        if not batch:
            return {}
        groups = self._completions_by_counter_scope(batch)
        if len(groups) == 1:
            return self._complete_processor_batch_scope(next(iter(groups.values())))
        completed: dict[str, object] = {}
        failure: Exception | None = None
        # One statement per cluster, each in its own transaction. Sharing
        # a transaction would accumulate every group's counter row lock
        # until the last one committed, which is the shape being fixed.
        # A 50-cluster burst drained at ~57 rows/s with 115 backends
        # blocked on this table and the longest waiter at 40.1s, while
        # Aurora sat at 3.8% CPU and 28.5 ACU - the queue was waiting on
        # itself, not on the database.
        for scope in sorted(groups):
            try:
                completed.update(self._complete_processor_batch_scope(groups[scope]))
            except Exception as exc:
                if failure is None:
                    failure = exc
        if failure is not None:
            raise failure
        return completed

    def _complete_processor_batch_scope(self, batch) -> dict[str, object]:
        now = datetime.now(timezone.utc)
        request_ids = [item["args"][0] for item in batch]
        owner_ids = [item["args"][1] for item in batch]
        epochs = [item["args"][2] for item in batch]
        tokens = [item["args"][3] for item in batch]
        response_statuses = [item["kwargs"]["response_status"] for item in batch]
        content_types = [item["kwargs"]["response_content_type"] for item in batch]
        response_bodies = [item["kwargs"]["response_body_base64"] for item in batch]
        payload_update = ""
        payload_parameters: tuple[object, ...] = ()
        if self.processor_queue_state_mode != "dedicated":
            payload_update = """
                            , payload=queue.payload ||
                                jsonb_build_object(
                                    'status', 'COMPLETED'::text,
                                    'response_status',
                                        valid.response_status,
                                    'response_content_type',
                                        valid.content_type,
                                    'response_body_base64',
                                        valid.response_body,
                                    'lease_expires_at', NULL,
                                    'updated_at', %s::text
                                )
            """
            payload_parameters = (_utc_text(now),)
        with self._db.transaction():
            with self._db.cursor() as cursor:
                cursor.execute(
                    f"""
                    WITH input AS (
                        SELECT *
                        FROM unnest(
                            %s::text[], %s::text[], %s::bigint[],
                            %s::text[], %s::integer[], %s::text[],
                            %s::text[]
                        ) AS value(
                            request_id, owner_id, epoch, lease_token,
                            response_status, content_type, response_body
                        )
                    ),
                    -- The claim path locks queue rows first and lane
                    -- rows second. Completing in the opposite order
                    -- inverts that and deadlocks: a claim that picked
                    -- up a request whose lease looks expired holds the
                    -- queue row and waits for the lane, while this
                    -- statement holds the lane and waits for the same
                    -- queue row. Taking the queue rows here, in
                    -- request_id order, keeps both paths on one
                    -- ordering.
                    locked_queue AS (
                        SELECT queue.request_id
                        FROM gpu_fault_processor_queue AS queue
                        WHERE queue.request_id IN (
                            SELECT input.request_id FROM input
                        )
                        ORDER BY queue.request_id
                        FOR UPDATE
                    ),
                    valid AS (
                        SELECT
                            queue.request_id,
                            queue.ordering_key,
                            input.response_status,
                            input.content_type,
                            input.response_body
                        FROM input
                        JOIN locked_queue
                          ON locked_queue.request_id=input.request_id
                        JOIN gpu_fault_processor_queue AS queue
                          ON queue.request_id=input.request_id
                         AND queue.status='LEASED'
                         AND queue.lease_owner=input.owner_id
                         AND queue.leader_epoch=input.epoch
                         AND queue.lease_token=input.lease_token
                        JOIN gpu_fault_processor_lanes AS lane
                          ON lane.ordering_key=queue.ordering_key
                         AND lane.owner_id=input.owner_id
                         AND lane.epoch=input.epoch
                         AND lane.lease_token=input.lease_token
                         AND lane.lease_expires_at > %s
                    ),
                    -- An UPDATE ... FROM takes its row locks in scan
                    -- order, which for a batch of completions is
                    -- whatever order the join produces. Two workers
                    -- completing overlapping ordering keys then
                    -- deadlock against each other and against the
                    -- claim path's lane insert, so the rows are
                    -- locked here first, in key order.
                    locked_lanes AS (
                        SELECT lane.ordering_key
                        FROM gpu_fault_processor_lanes AS lane
                        WHERE lane.ordering_key IN (
                            SELECT valid.ordering_key FROM valid
                        )
                        ORDER BY lane.ordering_key
                        FOR UPDATE
                    ),
                    released AS (
                        UPDATE gpu_fault_processor_lanes AS lane
                        SET lease_expires_at=%s, updated_at=%s
                        FROM valid
                        JOIN locked_lanes
                          ON locked_lanes.ordering_key=
                             valid.ordering_key
                        WHERE lane.ordering_key=valid.ordering_key
                        RETURNING lane.ordering_key
                    ),
                    completed AS (
                        UPDATE gpu_fault_processor_queue AS queue
                        SET
                            status='COMPLETED',
                            lease_expires_at=NULL,
                            response_status=valid.response_status,
                            response_content_type=valid.content_type,
                            response_body_base64=valid.response_body,
                            updated_at=%s
                            {payload_update}
                        FROM valid
                        JOIN released
                          ON released.ordering_key=
                             valid.ordering_key
                        WHERE queue.request_id=valid.request_id
                        RETURNING
                            queue.request_id,
                            queue.payload,
                            queue.status,
                            queue.lease_owner,
                            queue.leader_epoch,
                            queue.lease_token,
                            queue.lease_expires_at,
                            queue.response_status,
                            queue.response_content_type,
                            queue.response_body_base64,
                            queue.updated_at
                    )
                    SELECT
                        completed.request_id,
                        {self._processor_queue_effective_payload("completed")}
                    FROM completed
                    """,
                    (
                        request_ids,
                        owner_ids,
                        epochs,
                        tokens,
                        response_statuses,
                        content_types,
                        response_bodies,
                        now,
                        now,
                        now,
                        now,
                    )
                    + payload_parameters,
                )
                rows = cursor.fetchall()
        return {
            request_id: self._decode("processor_request", payload)
            for request_id, payload in rows
        }

    def complete_active_processor_requests_batch(self, completions):
        batch = [
            {
                "args": (
                    item["request_id"],
                    item["owner_id"],
                    item["lane_epoch"],
                    item["lease_token"],
                ),
                "kwargs": {
                    "response_status": item["response_status"],
                    "response_content_type": item["response_content_type"],
                    "response_body_base64": item["response_body_base64"],
                },
            }
            for item in completions
        ]
        completed = self._complete_active_processor_requests_batch(batch)
        return [completed.get(item["request_id"]) for item in completions]

    def _complete_active_processor_request_now(
        self,
        request_id: str,
        owner_id: str,
        lane_epoch: int,
        lease_token: str,
        *,
        response_status: int,
        response_content_type: str | None,
        response_body_base64: str,
    ):
        from gpu_fault.processor import (
            ProcessorRequestStatus,
        )

        with self._state_transaction(f"processor_request/{request_id}"):
            self._lock_processor_queue_row(request_id)
            current = self.get_processor_request(request_id)
            now = datetime.now(timezone.utc)
            if (
                current.lease_owner != owner_id
                or current.leader_epoch != lane_epoch
                or current.lease_token != lease_token
            ):
                raise ValueError("stale processor lane fencing token")
            with self._db.cursor() as cursor:
                cursor.execute(
                    """
                    UPDATE gpu_fault_processor_lanes
                    SET lease_expires_at=%s, updated_at=%s
                    WHERE ordering_key=%s
                      AND owner_id=%s
                      AND epoch=%s
                      AND lease_token=%s
                      AND lease_expires_at > %s
                    """,
                    (
                        now,
                        now,
                        current.ordering_key(),
                        owner_id,
                        lane_epoch,
                        lease_token,
                        now,
                    ),
                )
                completed_lane = cursor.rowcount == 1
            if not completed_lane:
                raise ValueError("stale processor lane fencing token")
            completed = current.model_copy(
                update={
                    "status": ProcessorRequestStatus.COMPLETED,
                    "response_status": response_status,
                    "response_content_type": response_content_type,
                    "response_body_base64": response_body_base64,
                    "lease_expires_at": None,
                    "updated_at": now,
                }
            )
            self._persist_processor_state(completed)
            return completed

    def complete_processor_request(
        self,
        request_id: str,
        owner_id: str,
        leader_epoch: int,
        lease_token: str,
        *,
        response_status: int,
        response_content_type: str | None,
        response_body_base64: str,
    ):
        from gpu_fault.processor import (
            ProcessorRequestStatus,
        )

        with self._state_transaction(f"processor_request/{request_id}"):
            leadership = self.get_processor_leadership()
            current = self.get_processor_request(request_id)
            now = datetime.now(timezone.utc)
            if (
                leadership is None
                or leadership.owner_id != owner_id
                or leadership.epoch != leader_epoch
                or leadership.lease_expires_at <= now
                or current.lease_owner != owner_id
                or current.leader_epoch != leader_epoch
                or current.lease_token != lease_token
                or current.lease_expires_at is None
                or current.lease_expires_at <= now
            ):
                raise ValueError("stale processor fencing token")
            completed = current.model_copy(
                update={
                    "status": ProcessorRequestStatus.COMPLETED,
                    "response_status": response_status,
                    "response_content_type": response_content_type,
                    "response_body_base64": response_body_base64,
                    "lease_expires_at": None,
                    "updated_at": now,
                }
            )
            self._persist_processor_state(completed)
            return completed

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from gpu_fault.processor import ProcessorRequestStatus
from gpu_fault.processor.models import ROUTINE_PRIORITY
from gpu_fault.store.shared.processor_helpers import (
    PartialEnqueueError,
    coalesce_routine_sample,
)


@dataclass
class _ProcessorAdmissionPlan:
    results: list[tuple[object | None, str | None] | None]
    final_by_base: dict[str, object]
    base_by_index: dict[int, str]
    reason_by_index: dict[int, str | None]
    new_base_ids: set[str]
    new_requests: list[object]
    # Coalesced base id -> the newest raw sample merged into it. If the
    # conditional merge write finds the row no longer PENDING, this sample
    # is admitted as a row of its own instead of being dropped.
    fallback_by_base: dict[str, object]


class PostgresProcessorAdmissionMixin:
    # Attributes supplied by the composed concrete implementation.
    _db: Any
    _decode: Callable[..., Any]
    _persist_processor_request: Callable[..., Any]
    _processor_counter_depths: Callable[..., Any]
    _processor_counter_mode: Callable[..., Any]
    _processor_counter_source: Callable[..., Any]
    _processor_queue_effective_payload: Callable[..., Any]
    _put_processor_queue_many: Callable[..., Any]

    def try_enqueue_processor_request(
        self,
        request,
        *,
        max_depth: int,
        max_cluster_depth: int,
        reserved_fault_depth: int = 0,
        reserved_cluster_fault_depth: int = 0,
        global_admission_guard: int = 256,
    ):
        with self._db.transaction():
            with self._db.cursor() as cursor:
                cursor.execute(
                    f"""
                    SELECT {
                        self._processor_queue_effective_payload(
                            "gpu_fault_processor_queue"
                        )
                    }
                    FROM gpu_fault_processor_queue
                    WHERE request_id=%s
                    FOR UPDATE
                    """,
                    (request.request_id,),
                )
                existing = cursor.fetchone()
            if existing is not None:
                return (
                    self._decode("processor_request", existing[0]),
                    None,
                )
            # ``coalescable()`` matches ``try_enqueue_processor_requests_
            # batch`` below. Asking the tier instead let a supersedable
            # sample overwrite whatever else was pending on the same lane.
            if request.coalescable():
                with self._db.cursor() as cursor:
                    cursor.execute(
                        f"""
                        SELECT {
                            self._processor_queue_effective_payload(
                                "gpu_fault_processor_queue"
                            )
                        }
                        FROM gpu_fault_processor_queue
                        WHERE status='PENDING'
                          AND ordering_key=%s
                          AND priority=%s
                          AND payload->>'path'=%s
                        ORDER BY created_at, request_id
                        LIMIT 1
                        FOR UPDATE
                        """,
                        (request.ordering_key(), ROUTINE_PRIORITY, request.path),
                    )
                    row = cursor.fetchone()
                if row is not None:
                    pending = self._decode("processor_request", row[0])
                    coalesced = coalesce_routine_sample(pending, request)
                    if self._update_pending_processor_row(coalesced):
                        return coalesced, "coalesced"
            counter_scope = request.cluster_id or "__unscoped__"
            with self._db.cursor() as cursor:
                depths, global_depth = self._processor_counter_depths(
                    cursor, [counter_scope]
                )
                cluster_depth = depths[counter_scope]
                # Near the cluster cap, unlocked depths would let
                # concurrent admissions each fit individually and
                # together overshoot - which for a fault means eating
                # into depth another cluster's faults are entitled to.
                # The guard band is the one the global boundary uses and
                # is far wider than the in-flight admission count, so
                # the locking read only runs when the cluster is
                # genuinely close to full.
                cluster_boundary = max(
                    0,
                    max_cluster_depth
                    - max(reserved_cluster_fault_depth, 0)
                    - max(global_admission_guard, 0),
                )
                if cluster_depth >= cluster_boundary:
                    cursor.execute(
                        """
                        SELECT pg_advisory_xact_lock(
                            hashtextextended(%s, 0)
                        )
                        """,
                        (f"processor_request/admission:cluster/{counter_scope}",),
                    )
                    depths, global_depth = self._processor_counter_depths(
                        cursor, [counter_scope]
                    )
                    cluster_depth = depths[counter_scope]
            global_limit = (
                max_depth
                if request.is_reserved_tier()
                else max_depth - reserved_fault_depth
            )
            if global_admission_guard > 0 and global_depth >= max(
                0, global_limit - global_admission_guard
            ):
                with self._db.cursor() as cursor:
                    cursor.execute(
                        """
                        SELECT pg_advisory_xact_lock(
                            hashtextextended(%s, 0)
                        )
                        """,
                        ("processor_request/admission:global-boundary",),
                    )
                    cursor.execute(
                        f"""
                        WITH counts AS (
                            {
                            self._processor_counter_source(
                                self._processor_counter_mode(cursor)
                            )
                        }
                        )
                        SELECT coalesce(sum(incomplete_count), 0)
                        FROM counts
                        """
                    )
                    global_depth = cursor.fetchone()[0]
            if global_depth >= max_depth:
                return None, "global"
            if (
                not request.is_reserved_tier()
                and global_depth >= max_depth - reserved_fault_depth
            ):
                return None, "global_reserved"
            if cluster_depth >= max_cluster_depth:
                return None, "cluster"
            if (
                not request.is_reserved_tier()
                and cluster_depth >= max_cluster_depth - reserved_cluster_fault_depth
            ):
                return None, "cluster_reserved"
            self._persist_processor_request(request)
            return request, None

    @staticmethod
    def _processor_admission_scope_indexes(requests):
        indexes_by_scope: dict[str, list[int]] = {}
        for index, request in enumerate(requests):
            indexes_by_scope.setdefault(
                request.cluster_id or "__unscoped__", []
            ).append(index)
        return indexes_by_scope

    def _enqueue_processor_scope_groups(
        self,
        requests,
        indexes_by_scope,
        *,
        max_depth: int,
        max_cluster_depth: int,
        reserved_fault_depth: int,
        reserved_cluster_fault_depth: int,
        global_admission_guard: int,
    ):
        # One transaction per cluster scope. The only caller today groups by
        # the same key, so a mixed batch never reaches this loop -- but the
        # signature allows one, and a group that fails after an earlier group
        # committed must say so rather than look like nothing landed (F-D9).
        combined: list[tuple[object | None, str | None] | None] = [None] * len(requests)
        committed: list[str] = []
        for indexes in indexes_by_scope.values():
            group = [requests[index] for index in indexes]
            try:
                group_results = self.try_enqueue_processor_requests_batch(
                    group,
                    max_depth=max_depth,
                    max_cluster_depth=max_cluster_depth,
                    reserved_fault_depth=reserved_fault_depth,
                    reserved_cluster_fault_depth=(reserved_cluster_fault_depth),
                    global_admission_guard=global_admission_guard,
                )
            except Exception as exc:
                raise PartialEnqueueError(committed=committed, cause=exc) from exc
            for index, result in zip(indexes, group_results, strict=True):
                combined[index] = result
                if result[0] is not None:
                    committed.append(result[0].request_id)
        return combined

    def _load_processor_admission_rows(self, requests):
        request_ids = sorted({request.request_id for request in requests})
        with self._db.cursor() as cursor:
            cursor.execute(
                """
                SELECT pg_advisory_xact_lock(
                    hashtextextended(request_id, 0)
                )
                FROM (
                    SELECT unnest(%s::text[]) AS request_id
                    ORDER BY request_id
                ) AS ordered
                """,
                (request_ids,),
            )
            # ``FOR UPDATE`` so this path and ``try_enqueue_processor_request``
            # (which locks the row, not the advisory key) exclude each
            # other on the same request id (F-D10). The advisory locks
            # above already fix the order, so the row locks add no new
            # deadlock face.
            cursor.execute(
                f"""
                SELECT request_id, {
                    self._processor_queue_effective_payload("gpu_fault_processor_queue")
                }
                FROM gpu_fault_processor_queue
                WHERE request_id=ANY(%s)
                ORDER BY request_id
                FOR UPDATE
                """,
                (request_ids,),
            )
            existing_by_id = {
                request_id: self._decode("processor_request", payload)
                for request_id, payload in cursor.fetchall()
            }
        ordering_keys = sorted(
            {request.ordering_key() for request in requests if request.coalescable()}
        )
        pending_by_ordering: dict[tuple[str, str], object] = {}
        if not ordering_keys:
            return existing_by_id, pending_by_ordering
        # Keyed by (lane, path): a lane is shared by every channel that
        # resolves to the node, and only the previous routine sample of the
        # same path may be superseded (F-D8).
        with self._db.cursor() as cursor:
            cursor.execute(
                f"""
                WITH candidates AS MATERIALIZED (
                    SELECT DISTINCT ON (ordering_key, payload->>'path')
                        request_id
                    FROM gpu_fault_processor_queue
                    WHERE status='PENDING'
                      AND priority=%s
                      AND ordering_key=ANY(%s)
                    ORDER BY
                        ordering_key,
                        payload->>'path',
                        created_at,
                        request_id
                ),
                locked AS MATERIALIZED (
                    SELECT queue.*
                    FROM gpu_fault_processor_queue AS queue
                    JOIN candidates
                      ON candidates.request_id=queue.request_id
                    ORDER BY queue.request_id
                    FOR UPDATE OF queue
                )
                SELECT ordering_key, {self._processor_queue_effective_payload("locked")}
                FROM locked
                ORDER BY ordering_key
                """,
                (ROUTINE_PRIORITY, ordering_keys),
            )
            for ordering_key, payload in cursor.fetchall():
                pending = self._decode("processor_request", payload)
                if pending.status is not ProcessorRequestStatus.PENDING:
                    continue
                pending_by_ordering.setdefault((ordering_key, pending.path), pending)
        return existing_by_id, pending_by_ordering

    def _update_pending_processor_row(self, coalesced: Any) -> bool:
        """Write a merged sample over its PENDING row, and only over that.

        A conditional UPDATE keyed on ``status='PENDING'`` rather than the
        blind ``ON CONFLICT`` upsert: the upsert overwrote ``lease_owner``,
        ``leader_epoch``, ``lease_token`` and ``lease_expires_at`` with the
        fresh sample's ``None`` whenever its ``updated_at`` was newer, which
        is how a claimed row went back to PENDING under its owner (B-1).
        The lease columns are not written at all here. Returns ``False``
        when the row is no longer PENDING, so the caller admits the sample
        as a row of its own instead of silently dropping it.
        """

        with self._db.cursor() as cursor:
            cursor.execute(
                """
                UPDATE gpu_fault_processor_queue
                SET
                    cluster_id=%s,
                    correlation_key=%s,
                    ordering_key=%s,
                    priority=%s,
                    not_before=%s,
                    retry_count=%s,
                    lane_policy=%s,
                    updated_at=%s,
                    payload=%s::jsonb
                WHERE request_id=%s
                  AND status='PENDING'
                """,
                (
                    coalesced.cluster_id,
                    coalesced.correlation_key,
                    coalesced.ordering_key(),
                    coalesced.queue_priority(),
                    coalesced.not_before,
                    coalesced.retry_count,
                    coalesced.lane_policy.value,
                    coalesced.updated_at,
                    coalesced.model_dump_json(),
                    coalesced.request_id,
                ),
            )
            return bool(cursor.rowcount == 1)

    @staticmethod
    def _plan_processor_admission(
        requests,
        existing_by_id,
        pending_by_ordering,
    ) -> _ProcessorAdmissionPlan:
        results: list[tuple[object | None, str | None] | None] = [None] * len(requests)
        final_by_base = {}
        base_by_index: dict[int, str] = {}
        reason_by_index: dict[int, str | None] = {}
        new_base_ids: set[str] = set()
        fallback_by_base: dict[str, object] = {}
        for index, request in enumerate(requests):
            existing = existing_by_id.get(request.request_id)
            if existing is not None:
                results[index] = (existing, None)
                continue
            ordering_key = (request.ordering_key(), request.path)
            coalescable = request.coalescable()
            pending = pending_by_ordering.get(ordering_key) if coalescable else None
            if pending is not None:
                coalesced = coalesce_routine_sample(pending, request)
                pending_by_ordering[ordering_key] = coalesced
                final_by_base[pending.request_id] = coalesced
                base_by_index[index] = pending.request_id
                reason_by_index[index] = "coalesced"
                if pending.request_id not in new_base_ids:
                    fallback_by_base[pending.request_id] = request
                continue
            final_by_base[request.request_id] = request
            base_by_index[index] = request.request_id
            reason_by_index[index] = None
            new_base_ids.add(request.request_id)
            if coalescable:
                pending_by_ordering[ordering_key] = request
        new_requests = [
            final_by_base[base_id]
            for base_id in dict.fromkeys(
                base_by_index[index]
                for index in range(len(requests))
                if (index in base_by_index and base_by_index[index] in new_base_ids)
            )
        ]
        return _ProcessorAdmissionPlan(
            results=results,
            final_by_base=final_by_base,
            base_by_index=base_by_index,
            reason_by_index=reason_by_index,
            new_base_ids=new_base_ids,
            new_requests=new_requests,
            fallback_by_base=fallback_by_base,
        )

    def _lock_processor_admission_capacity(
        self,
        requests,
        *,
        max_depth: int,
        max_cluster_depth: int,
        reserved_fault_depth: int,
        reserved_cluster_fault_depth: int,
        global_admission_guard: int,
    ) -> tuple[dict[str, int], int]:
        counter_scopes = sorted(
            {request.cluster_id or "__unscoped__" for request in requests}
        )
        if not counter_scopes:
            return {}, 0
        with self._db.cursor() as cursor:
            depths, global_depth = self._processor_counter_depths(
                cursor, counter_scopes
            )
            boundary = max(
                0,
                max_cluster_depth
                - max(reserved_cluster_fault_depth, 0)
                - max(global_admission_guard, 0),
            )
            if any(depth >= boundary for depth in depths.values()):
                cursor.execute(
                    """
                    SELECT pg_advisory_xact_lock(
                        hashtextextended(
                            'processor_request/'
                            'admission:cluster/' || scope,
                            0
                        )
                    )
                    FROM unnest(%s::text[]) AS scope
                    ORDER BY scope
                    """,
                    (counter_scopes,),
                )
                depths, global_depth = self._processor_counter_depths(
                    cursor, counter_scopes
                )
        if not (
            global_admission_guard > 0
            and global_depth
            >= max(
                0,
                min(
                    (
                        max_depth
                        if request.is_reserved_tier()
                        else max_depth - reserved_fault_depth
                    )
                    for request in requests
                )
                - global_admission_guard,
            )
        ):
            return depths, global_depth
        with self._db.cursor() as cursor:
            cursor.execute(
                """
                SELECT pg_advisory_xact_lock(
                    hashtextextended(%s, 0)
                )
                """,
                ("processor_request/admission:global-boundary",),
            )
            cursor.execute(
                f"""
                WITH counts AS (
                    {
                    self._processor_counter_source(self._processor_counter_mode(cursor))
                }
                )
                SELECT coalesce(sum(incomplete_count), 0)
                FROM counts
                """
            )
            global_depth = cursor.fetchone()[0]
        return depths, global_depth

    @staticmethod
    def _select_processor_admission_requests(
        requests,
        depths,
        global_depth: int,
        *,
        max_depth: int,
        max_cluster_depth: int,
        reserved_fault_depth: int,
        reserved_cluster_fault_depth: int,
    ) -> tuple[dict[str, str], set[str]]:
        rejected_by_base: dict[str, str] = {}
        accepted_new_ids: set[str] = set()
        for request in requests:
            scope = request.cluster_id or "__unscoped__"
            global_limit = (
                max_depth
                if request.is_reserved_tier()
                else max_depth - reserved_fault_depth
            )
            cluster_limit = (
                max_cluster_depth
                if request.is_reserved_tier()
                else max_cluster_depth - reserved_cluster_fault_depth
            )
            if global_depth >= max_depth:
                rejected_by_base[request.request_id] = "global"
                continue
            if not request.is_reserved_tier() and global_depth >= global_limit:
                rejected_by_base[request.request_id] = "global_reserved"
                continue
            if depths[scope] >= max_cluster_depth:
                rejected_by_base[request.request_id] = "cluster"
                continue
            if not request.is_reserved_tier() and depths[scope] >= cluster_limit:
                rejected_by_base[request.request_id] = "cluster_reserved"
                continue
            accepted_new_ids.add(request.request_id)
            global_depth += 1
            depths[scope] += 1
        return rejected_by_base, accepted_new_ids

    def _persist_processor_admission_plan(
        self,
        plan: _ProcessorAdmissionPlan,
        rejected_by_base: dict[str, str],
        accepted_new_ids: set[str],
    ):
        new_rows = [
            request
            for base_id, request in plan.final_by_base.items()
            if base_id in plan.new_base_ids and base_id in accepted_new_ids
        ]
        # Merges into rows that were already queued go through the
        # conditional UPDATE, never the upsert (B-1). The rows are locked
        # FOR UPDATE by ``_load_processor_admission_rows`` in this same
        # transaction, so a miss here means the lock and the status check
        # disagreed - the sample is then admitted as a row of its own
        # rather than answered "coalesced" and lost.
        for base_id, request in plan.final_by_base.items():
            if base_id in plan.new_base_ids:
                continue
            if self._update_pending_processor_row(request):
                continue
            fallback = plan.fallback_by_base[base_id]
            plan.final_by_base[base_id] = fallback
            for index, mapped in plan.base_by_index.items():
                if mapped == base_id:
                    plan.reason_by_index[index] = None
            new_rows.append(fallback)
        self._put_processor_queue_many(new_rows)
        for index in range(len(plan.results)):
            if plan.results[index] is not None:
                continue
            base_id = plan.base_by_index[index]
            rejection = rejected_by_base.get(base_id)
            if rejection is not None:
                plan.results[index] = (None, rejection)
            else:
                plan.results[index] = (
                    plan.final_by_base[base_id],
                    plan.reason_by_index[index],
                )
        return plan.results

    def try_enqueue_processor_requests_batch(
        self,
        requests,
        *,
        max_depth: int,
        max_cluster_depth: int,
        reserved_fault_depth: int = 0,
        reserved_cluster_fault_depth: int = 0,
        global_admission_guard: int = 256,
    ):
        if not requests:
            return []
        indexes_by_scope = self._processor_admission_scope_indexes(requests)
        if len(indexes_by_scope) > 1:
            return self._enqueue_processor_scope_groups(
                requests,
                indexes_by_scope,
                max_depth=max_depth,
                max_cluster_depth=max_cluster_depth,
                reserved_fault_depth=reserved_fault_depth,
                reserved_cluster_fault_depth=(reserved_cluster_fault_depth),
                global_admission_guard=global_admission_guard,
            )
        with self._db.transaction():
            existing, pending = self._load_processor_admission_rows(requests)
            plan = self._plan_processor_admission(requests, existing, pending)
            depths, global_depth = self._lock_processor_admission_capacity(
                plan.new_requests,
                max_depth=max_depth,
                max_cluster_depth=max_cluster_depth,
                reserved_fault_depth=reserved_fault_depth,
                reserved_cluster_fault_depth=(reserved_cluster_fault_depth),
                global_admission_guard=global_admission_guard,
            )
            rejected, accepted = self._select_processor_admission_requests(
                plan.new_requests,
                depths,
                global_depth,
                max_depth=max_depth,
                max_cluster_depth=max_cluster_depth,
                reserved_fault_depth=reserved_fault_depth,
                reserved_cluster_fault_depth=(reserved_cluster_fault_depth),
            )
            return self._persist_processor_admission_plan(plan, rejected, accepted)

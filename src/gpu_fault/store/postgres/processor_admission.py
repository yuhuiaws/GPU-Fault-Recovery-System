from __future__ import annotations

from typing import Any, Callable

from dataclasses import dataclass

from gpu_fault.processor import ProcessorRequestStatus


@dataclass
class _ProcessorAdmissionPlan:
    results: list[tuple[object | None, str | None] | None]
    final_by_base: dict[str, object]
    base_by_index: dict[int, str]
    reason_by_index: dict[int, str | None]
    new_base_ids: set[str]
    new_requests: list[object]


class PostgresProcessorAdmissionMixin:
    # Attributes supplied by the composed concrete implementation.
    _db: Any
    _decode: Callable[..., Any]
    _maybe_reconcile_legacy_processor_requests: Callable[..., Any]
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
        self._maybe_reconcile_legacy_processor_requests()
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
                        ORDER BY created_at, request_id
                        LIMIT 1
                        FOR UPDATE
                        """,
                        (request.ordering_key(),),
                    )
                    row = cursor.fetchone()
                if row is not None:
                    pending = self._decode("processor_request", row[0])
                    coalesced = request.model_copy(
                        update={
                            "request_id": pending.request_id,
                            "created_at": pending.created_at,
                            "status": ProcessorRequestStatus.PENDING,
                        }
                    )
                    self._persist_processor_request(coalesced)
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
                if request.queue_priority() == 0
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
                request.queue_priority() != 0
                and global_depth >= max_depth - reserved_fault_depth
            ):
                return None, "global_reserved"
            if cluster_depth >= max_cluster_depth:
                return None, "cluster"
            if (
                request.queue_priority() != 0
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
        combined = [None] * len(requests)
        for indexes in indexes_by_scope.values():
            group = [requests[index] for index in indexes]
            group_results = self.try_enqueue_processor_requests_batch(
                group,
                max_depth=max_depth,
                max_cluster_depth=max_cluster_depth,
                reserved_fault_depth=reserved_fault_depth,
                reserved_cluster_fault_depth=(reserved_cluster_fault_depth),
                global_admission_guard=global_admission_guard,
            )
            for index, result in zip(indexes, group_results, strict=True):
                combined[index] = result
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
            cursor.execute(
                f"""
                SELECT request_id, {
                    self._processor_queue_effective_payload("gpu_fault_processor_queue")
                }
                FROM gpu_fault_processor_queue
                WHERE request_id=ANY(%s)
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
        pending_by_ordering = {}
        if not ordering_keys:
            return existing_by_id, pending_by_ordering
        with self._db.cursor() as cursor:
            cursor.execute(
                f"""
                WITH candidates AS MATERIALIZED (
                    SELECT DISTINCT ON (ordering_key)
                        request_id
                    FROM gpu_fault_processor_queue
                    WHERE status='PENDING'
                      AND ordering_key=ANY(%s)
                    ORDER BY
                        ordering_key,
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
                (ordering_keys,),
            )
            for ordering_key, payload in cursor.fetchall():
                pending_by_ordering.setdefault(
                    ordering_key,
                    self._decode("processor_request", payload),
                )
        return existing_by_id, pending_by_ordering

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
        for index, request in enumerate(requests):
            existing = existing_by_id.get(request.request_id)
            if existing is not None:
                results[index] = (existing, None)
                continue
            ordering_key = request.ordering_key()
            coalescable = request.coalescable()
            pending = pending_by_ordering.get(ordering_key) if coalescable else None
            if pending is not None:
                coalesced = request.model_copy(
                    update={
                        "request_id": pending.request_id,
                        "created_at": pending.created_at,
                        "status": ProcessorRequestStatus.PENDING,
                    }
                )
                pending_by_ordering[ordering_key] = coalesced
                final_by_base[pending.request_id] = coalesced
                base_by_index[index] = pending.request_id
                reason_by_index[index] = "coalesced"
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
                        if request.queue_priority() == 0
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
                if request.queue_priority() == 0
                else max_depth - reserved_fault_depth
            )
            cluster_limit = (
                max_cluster_depth
                if request.queue_priority() == 0
                else max_cluster_depth - reserved_cluster_fault_depth
            )
            if global_depth >= max_depth:
                rejected_by_base[request.request_id] = "global"
                continue
            if request.queue_priority() != 0 and global_depth >= global_limit:
                rejected_by_base[request.request_id] = "global_reserved"
                continue
            if depths[scope] >= max_cluster_depth:
                rejected_by_base[request.request_id] = "cluster"
                continue
            if request.queue_priority() != 0 and depths[scope] >= cluster_limit:
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
        self._put_processor_queue_many(
            [
                request
                for base_id, request in plan.final_by_base.items()
                if not (
                    base_id in plan.new_base_ids and base_id not in accepted_new_ids
                )
            ]
        )
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
        self._maybe_reconcile_legacy_processor_requests()
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

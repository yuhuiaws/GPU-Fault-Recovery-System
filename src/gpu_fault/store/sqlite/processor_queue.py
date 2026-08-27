from __future__ import annotations

from typing import Any, Callable

import secrets
from datetime import datetime, timedelta, timezone

from gpu_fault.processor import (
    ProcessorLaneLease,
    ProcessorRequestStatus,
    deferred_strict_processor_lanes,
    processor_request_claimable,
)
from gpu_fault.store.contracts import ProcessorQueueStats
from gpu_fault.store.shared.processor_helpers import (
    incomplete_observation_scope_keys as _incomplete_observation_scope_keys,
    pending_fault_scope_keys as _pending_fault_scope_keys,
)


class SqliteProcessorQueueMixin:
    # Attributes supplied by the composed concrete implementation.
    get_processor_leadership: Callable[..., Any]

    _delete: Callable[..., Any]
    _get: Callable[..., Any]
    _get_optional: Callable[..., Any]
    _list: Callable[..., Any]
    _put: Callable[..., Any]
    _state_transaction: Callable[..., Any]

    def enqueue_processor_request(self, request):
        with self._state_transaction(f"processor_request/{request.request_id}"):
            existing = self._get_optional("processor_request", request.request_id)
            if existing is not None:
                return existing
            self._put(
                "processor_request",
                request.request_id,
                request,
            )
            return request

    def try_enqueue_processor_request(
        self,
        request,
        *,
        max_depth: int,
        max_cluster_depth: int,
        reserved_fault_depth: int = 0,
        reserved_cluster_fault_depth: int = 0,
        global_admission_guard: int = 0,
    ):
        with self._state_transaction("processor_request/admission"):
            existing = self._get_optional("processor_request", request.request_id)
            if existing is not None:
                return existing, None
            # See ``InMemoryStore.try_enqueue_processor_request``: the tier
            # says the sample is supersedable, only the lane says what it
            # would supersede.
            if request.coalescable():
                pending_match = next(
                    (
                        item
                        for item in self._list("processor_request")
                        if item.status is ProcessorRequestStatus.PENDING
                        and item.ordering_key() == request.ordering_key()
                    ),
                    None,
                )
                if pending_match is not None:
                    coalesced = request.model_copy(
                        update={
                            "request_id": pending_match.request_id,
                            "created_at": pending_match.created_at,
                        }
                    )
                    self._put(
                        "processor_request",
                        pending_match.request_id,
                        coalesced,
                    )
                    return coalesced, "coalesced"
            incomplete = [
                item
                for item in self._list("processor_request")
                if item.status
                in {
                    ProcessorRequestStatus.PENDING,
                    ProcessorRequestStatus.LEASED,
                }
            ]
            if len(incomplete) >= max_depth:
                return None, "global"
            if (
                request.queue_priority() != 0
                and len(incomplete) >= max_depth - reserved_fault_depth
            ):
                return None, "global_reserved"
            cluster_depth = sum(
                item.cluster_id == request.cluster_id for item in incomplete
            )
            if cluster_depth >= max_cluster_depth:
                return None, "cluster"
            if (
                request.queue_priority() != 0
                and cluster_depth >= max_cluster_depth - reserved_cluster_fault_depth
            ):
                return None, "cluster_reserved"
            self._put(
                "processor_request",
                request.request_id,
                request,
            )
            return request, None

    def processor_queue_stats(
        self, *, now: datetime | None = None
    ) -> ProcessorQueueStats:
        observed_at = now or datetime.now(timezone.utc)
        incomplete = [
            item
            for item in self._list("processor_request")
            if item.status
            in {
                ProcessorRequestStatus.PENDING,
                ProcessorRequestStatus.LEASED,
            }
        ]
        by_cluster: dict[str, int] = {}
        for item in incomplete:
            key = item.cluster_id or "__unscoped__"
            by_cluster[key] = by_cluster.get(key, 0) + 1
        oldest_age = max(
            (
                max(
                    0.0,
                    (observed_at - item.created_at).total_seconds(),
                )
                for item in incomplete
            ),
            default=0.0,
        )
        return {
            "depth": len(incomplete),
            "oldest_age_seconds": oldest_age,
            "by_cluster": by_cluster,
        }

    def processor_fault_backlog_depth(self) -> int:
        return sum(
            item.status
            in {
                ProcessorRequestStatus.PENDING,
                ProcessorRequestStatus.LEASED,
            }
            and item.queue_priority() == 0
            for item in self._list("processor_request")
        )

    def get_processor_request(self, request_id: str):
        return self._get("processor_request", request_id)

    def has_incomplete_processor_requests(self, cluster_id: str) -> bool:
        return any(
            item.cluster_id == cluster_id
            and item.status
            in {
                ProcessorRequestStatus.PENDING,
                ProcessorRequestStatus.LEASED,
            }
            for item in self._list("processor_request")
        )

    def has_incomplete_processor_requests_for_scopes(
        self, cluster_id: str, scope_keys: set[str]
    ) -> bool:
        if not scope_keys:
            return self.has_incomplete_processor_requests(cluster_id)
        return any(
            item.cluster_id == cluster_id
            and item.status
            in {
                ProcessorRequestStatus.PENDING,
                ProcessorRequestStatus.LEASED,
            }
            and not scope_keys.isdisjoint(item.correlation_scope_keys)
            for item in self._list("processor_request")
        )

    def claim_processor_requests(
        self,
        owner_id: str,
        leader_epoch: int,
        *,
        now: datetime,
        lease_duration: timedelta,
        limit: int,
    ):
        with self._state_transaction("processor_request/claims"):
            leadership = self.get_processor_leadership()
            if (
                leadership is None
                or leadership.owner_id != owner_id
                or leadership.epoch != leader_epoch
                or leadership.lease_expires_at <= now
            ):
                return []
            items = self._list("processor_request")
            deferred_strict_lanes = deferred_strict_processor_lanes(items, now)
            busy_clusters = {
                item.ordering_key()
                for item in items
                if item.status is ProcessorRequestStatus.LEASED
                and item.lease_expires_at is not None
                and item.lease_expires_at > now
            }
            eligible_items = [
                item
                for item in items
                if processor_request_claimable(
                    item,
                    now=now,
                    deferred_strict_lanes=deferred_strict_lanes,
                )
            ]
            observation_scope_keys = _incomplete_observation_scope_keys(items)
            eligible_items = [
                item
                for item in eligible_items
                if not item.waits_for_observation(observation_scope_keys)
            ]
            pending_fault_keys = _pending_fault_scope_keys(eligible_items)
            eligible = sorted(
                eligible_items,
                key=lambda item: (
                    item.claim_priority(pending_fault_keys),
                    item.created_at,
                    item.request_id,
                ),
            )
            candidates = []
            selected_clusters = set(busy_clusters)
            for item in eligible:
                cluster = item.ordering_key()
                if cluster in selected_clusters:
                    continue
                candidates.append(item)
                selected_clusters.add(cluster)
                if len(candidates) >= limit:
                    break
            claimed = []
            for item in candidates:
                value = item.model_copy(
                    update={
                        "status": ProcessorRequestStatus.LEASED,
                        "lease_owner": owner_id,
                        "leader_epoch": leader_epoch,
                        "lease_token": secrets.token_urlsafe(32),
                        "lease_expires_at": now + lease_duration,
                        "updated_at": now,
                    }
                )
                self._put(
                    "processor_request",
                    item.request_id,
                    value,
                )
                claimed.append(value)
            return claimed

    def claim_active_processor_requests(
        self,
        owner_id: str,
        *,
        now: datetime,
        lease_duration: timedelta,
        limit: int,
        include_paths: set[str] | None = None,
        exclude_paths: set[str] | None = None,
        routine_starvation_seconds: float = 30.0,
    ):
        with self._state_transaction("processor_lane/claims"):
            items = self._list("processor_request")
            lanes = {item.ordering_key: item for item in self._list("processor_lane")}
            deferred_strict_lanes = deferred_strict_processor_lanes(items, now)
            eligible_items = [
                item
                for item in items
                if processor_request_claimable(
                    item,
                    now=now,
                    deferred_strict_lanes=deferred_strict_lanes,
                )
            ]
            observation_scope_keys = _incomplete_observation_scope_keys(items)
            eligible_items = [
                item
                for item in eligible_items
                if not item.waits_for_observation(observation_scope_keys)
            ]
            if include_paths is not None:
                eligible_items = [
                    item for item in eligible_items if item.path in include_paths
                ]
            if exclude_paths is not None:
                eligible_items = [
                    item for item in eligible_items if item.path not in exclude_paths
                ]
            pending_fault_keys = _pending_fault_scope_keys(eligible_items)
            routine_starvation_before = now - timedelta(
                seconds=routine_starvation_seconds
            )
            eligible = sorted(
                eligible_items,
                key=lambda item: (
                    item.claim_priority(
                        pending_fault_keys,
                        routine_starvation_before,
                    ),
                    item.created_at,
                    item.request_id,
                ),
            )
            claimed = []
            selected_lanes = set()
            for item in eligible:
                key = item.ordering_key()
                lane = lanes.get(key)
                if key in selected_lanes or (
                    lane is not None and lane.lease_expires_at > now
                ):
                    continue
                epoch = lane.epoch + 1 if lane is not None else 1
                token = secrets.token_urlsafe(32)
                lane = ProcessorLaneLease(
                    ordering_key=key,
                    owner_id=owner_id,
                    epoch=epoch,
                    lease_token=token,
                    lease_expires_at=now + lease_duration,
                    updated_at=now,
                )
                self._put("processor_lane", key, lane)
                value = item.model_copy(
                    update={
                        "status": ProcessorRequestStatus.LEASED,
                        "lease_owner": owner_id,
                        "leader_epoch": epoch,
                        "lease_token": token,
                        "lease_expires_at": now + lease_duration,
                        "updated_at": now,
                    }
                )
                self._put(
                    "processor_request",
                    item.request_id,
                    value,
                )
                claimed.append(value)
                selected_lanes.add(key)
                if len(claimed) >= limit:
                    break
            return claimed

    def cleanup_completed_processor_requests(
        self,
        *,
        older_than: datetime,
        limit: int,
    ) -> int:
        with self._state_transaction("processor_request/cleanup"):
            request_ids = [
                item.request_id
                for item in sorted(
                    self._list("processor_request"),
                    key=lambda item: (
                        item.updated_at,
                        item.request_id,
                    ),
                )
                if item.status is ProcessorRequestStatus.COMPLETED
                and item.updated_at <= older_than
            ][:limit]
            for request_id in request_ids:
                self._delete("processor_request", request_id)
            return len(request_ids)

    def active_backlog_is_lane_blocked(
        self,
        *,
        now: datetime,
        include_paths: set[str] | None = None,
        exclude_paths: set[str] | None = None,
    ) -> bool:
        lanes = {item.ordering_key: item for item in self._list("processor_lane")}
        items = self._list("processor_request")
        deferred_strict_lanes = deferred_strict_processor_lanes(items, now)
        for item in items:
            if not processor_request_claimable(
                item,
                now=now,
                deferred_strict_lanes=deferred_strict_lanes,
            ):
                continue
            if include_paths is not None and item.path not in include_paths:
                continue
            if exclude_paths is not None and item.path in exclude_paths:
                continue
            lane = lanes.get(item.ordering_key())
            if lane is not None and lane.lease_expires_at > now:
                return True
        return False

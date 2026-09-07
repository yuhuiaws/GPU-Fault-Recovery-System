from __future__ import annotations

from typing import Any

import secrets
from datetime import datetime, timedelta, timezone

from gpu_fault.processor import (
    ProcessorLaneLease,
    ProcessorRequestStatus,
    deferred_strict_processor_lanes,
    processor_request_claimable,
)
from gpu_fault.store.contracts import (
    ProcessorQueueStats,
)
from gpu_fault.store.shared.errors import NotFoundError
from gpu_fault.store.shared.processor_helpers import (
    fault_rows_blocked_by_observation as _fault_rows_blocked_by_observation,
    incomplete_observation_scope_keys as _incomplete_observation_scope_keys,
    pending_fault_scope_keys as _pending_fault_scope_keys,
)


class MemoryProcessorQueueMixin:
    # Attributes supplied by the composed concrete implementation.
    _processor_requests: Any

    _lock: Any
    _processor_lanes: Any
    _processor_leadership: Any

    def enqueue_processor_request(self, request):
        with self._lock:
            existing = self._processor_requests.get(request.request_id)
            if existing is not None:
                return existing
            self._processor_requests[request.request_id] = request
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
        with self._lock:
            existing = self._processor_requests.get(request.request_id)
            if existing is not None:
                return existing, None
            # A lane is shared by every channel that resolves to the same
            # node, so "the pending row on this lane" may be a fault, a node
            # log batch, or an earlier breach whose confirmation the
            # detector is still counting. What a latest-wins sample may
            # supersede is the previous routine sample of its *own* path,
            # nothing else (F-D8).
            if request.coalescable():
                pending_match = next(
                    (
                        item
                        for item in self._processor_requests.values()
                        if item.status is ProcessorRequestStatus.PENDING
                        and item.ordering_key() == request.ordering_key()
                        and item.path == request.path
                        and item.coalescable()
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
                    self._processor_requests[pending_match.request_id] = coalesced
                    return coalesced, "coalesced"
            incomplete = [
                item
                for item in self._processor_requests.values()
                if item.status
                in {
                    ProcessorRequestStatus.PENDING,
                    ProcessorRequestStatus.LEASED,
                }
            ]
            if len(incomplete) >= max_depth:
                return None, "global"
            if (
                not request.is_reserved_tier()
                and len(incomplete) >= max_depth - reserved_fault_depth
            ):
                return None, "global_reserved"
            cluster_depth = sum(
                item.cluster_id == request.cluster_id for item in incomplete
            )
            if cluster_depth >= max_cluster_depth:
                return None, "cluster"
            if (
                not request.is_reserved_tier()
                and cluster_depth >= max_cluster_depth - reserved_cluster_fault_depth
            ):
                return None, "cluster_reserved"
            self._processor_requests[request.request_id] = request
            return request, None

    def processor_queue_stats(
        self, *, now: datetime | None = None
    ) -> ProcessorQueueStats:
        observed_at = now or datetime.now(timezone.utc)
        with self._lock:
            incomplete = [
                item
                for item in self._processor_requests.values()
                if item.status
                in {
                    ProcessorRequestStatus.PENDING,
                    ProcessorRequestStatus.LEASED,
                }
            ]
        by_cluster: dict[str, int] = {}
        oldest_age_by_cluster: dict[str, float] = {}
        for item in incomplete:
            key = item.cluster_id or "__unscoped__"
            by_cluster[key] = by_cluster.get(key, 0) + 1
            age = max(0.0, (observed_at - item.created_at).total_seconds())
            oldest_age_by_cluster[key] = max(oldest_age_by_cluster.get(key, 0.0), age)
        return {
            "depth": len(incomplete),
            "oldest_age_seconds": max(oldest_age_by_cluster.values(), default=0.0),
            "by_cluster": by_cluster,
            "oldest_age_by_cluster": oldest_age_by_cluster,
        }

    def processor_fault_backlog_depth(self) -> int:
        with self._lock:
            return sum(
                item.status
                in {
                    ProcessorRequestStatus.PENDING,
                    ProcessorRequestStatus.LEASED,
                }
                and item.is_reserved_tier()
                for item in self._processor_requests.values()
            )

    def get_processor_request(self, request_id: str):
        with self._lock:
            request = self._processor_requests.get(request_id)
            if request is None:
                raise NotFoundError(request_id)
            return request

    def has_incomplete_processor_requests(self, cluster_id: str) -> bool:
        with self._lock:
            return any(
                item.cluster_id == cluster_id
                and item.status
                in {
                    ProcessorRequestStatus.PENDING,
                    ProcessorRequestStatus.LEASED,
                }
                for item in self._processor_requests.values()
            )

    def has_incomplete_processor_requests_for_scopes(
        self, cluster_id: str, scope_keys: set[str]
    ) -> bool:
        if not scope_keys:
            return self.has_incomplete_processor_requests(cluster_id)
        with self._lock:
            return any(
                item.cluster_id == cluster_id
                and item.status
                in {
                    ProcessorRequestStatus.PENDING,
                    ProcessorRequestStatus.LEASED,
                }
                and not scope_keys.isdisjoint(item.correlation_scope_keys)
                for item in self._processor_requests.values()
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
        with self._lock:
            leadership = self._processor_leadership
            if (
                leadership is None
                or leadership.owner_id != owner_id
                or leadership.epoch != leader_epoch
                or leadership.lease_expires_at <= now
            ):
                return []
            items = list(self._processor_requests.values())
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
            observation_scope_keys = _incomplete_observation_scope_keys(items, now)
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
                self._processor_requests[item.request_id] = value
                claimed.append(value)
            return claimed

    def count_fault_rows_blocked_by_observation(self, *, now: datetime) -> int:
        with self._lock:
            items = list(self._processor_requests.values())
        return _fault_rows_blocked_by_observation(items, now)

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
        with self._lock:
            items = list(self._processor_requests.values())
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
            observation_scope_keys = _incomplete_observation_scope_keys(items, now)
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
                lane_key = item.ordering_key()
                lane = self._processor_lanes.get(lane_key)
                if lane_key in selected_lanes or (
                    lane is not None and lane.lease_expires_at > now
                ):
                    continue
                epoch = lane.epoch + 1 if lane is not None else 1
                token = secrets.token_urlsafe(32)
                self._processor_lanes[lane_key] = ProcessorLaneLease(
                    ordering_key=lane_key,
                    owner_id=owner_id,
                    epoch=epoch,
                    lease_token=token,
                    lease_expires_at=now + lease_duration,
                    updated_at=now,
                )
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
                self._processor_requests[item.request_id] = value
                claimed.append(value)
                selected_lanes.add(lane_key)
                if len(claimed) >= limit:
                    break
            return claimed

    def cleanup_completed_processor_requests(
        self,
        *,
        older_than: datetime,
        limit: int,
    ) -> int:
        with self._lock:
            request_ids = [
                item.request_id
                for item in sorted(
                    self._processor_requests.values(),
                    key=lambda item: (
                        item.updated_at,
                        item.request_id,
                    ),
                )
                if item.status is ProcessorRequestStatus.COMPLETED
                and item.updated_at <= older_than
            ][:limit]
            for request_id in request_ids:
                del self._processor_requests[request_id]
            return len(request_ids)

    def active_backlog_is_lane_blocked(
        self,
        *,
        now: datetime,
        include_paths: set[str] | None = None,
        exclude_paths: set[str] | None = None,
    ) -> bool:
        """Is there claimable work whose lane is currently leased?

        ``claim_active_processor_requests`` skips any request whose lane
        still has a live lease, so an empty claim carries two very
        different meanings: the queue is drained, or the queue is deep on
        a handful of lanes that somebody else holds. The consumer needs
        to tell them apart before it backs off for seconds.
        """

        with self._lock:
            items = list(self._processor_requests.values())
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
                lane = self._processor_lanes.get(item.ordering_key())
                if lane is not None and lane.lease_expires_at > now:
                    return True
            return False

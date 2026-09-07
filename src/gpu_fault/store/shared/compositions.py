"""Store methods with one implementation for every backend.

Each method here is written over the store's own public contract -- it calls
other ``ControlPlaneStore`` methods and touches no rows, dicts or connections
itself -- so all three stores compose it. A backend with a better statement
(PostgreSQL's multi-row batches, its single-query scope scan) overrides the
entry it improves; the method here is the definition the override must match.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Callable, Iterable, Iterator, Sequence

from gpu_fault.gpu_metrics import GpuInventorySnapshot
from gpu_fault.models import CompletionDecision
from gpu_fault.policy import XidEvent
from gpu_fault.remote_command_models import RemoteCommandStatus
from gpu_fault.store.contracts import ProcessorQueueCountStatus
from gpu_fault.store.shared.errors import NotFoundError
from gpu_fault.store.shared.processor_helpers import PartialEnqueueError
from gpu_fault.telemetry import CollectorStatus
from gpu_fault.telemetry_models import TelemetryMetricLatest
from gpu_fault.watcher import AttemptObservation


class SharedCompositionMixin:
    # Attributes supplied by the composed concrete implementation.
    _open_remote_command_candidates: Callable[..., Any]
    complete_active_processor_request: Callable[..., Any]
    get_decision_by_event: Callable[..., Any]
    get_event_by_attempt: Callable[..., Any]
    get_gpu_inventory_snapshot: Callable[..., Any]
    get_xid_event: Callable[..., Any]
    list_xid_events: Callable[..., Any]
    observe_gpu_inventory_snapshots: Callable[..., Any]
    observe_telemetry_metrics: Callable[..., Any]
    processor_queue_stats: Callable[..., Any]
    save_attempt_observation: Callable[..., Any]
    save_collector_statuses_batch: Callable[..., Any]
    try_enqueue_processor_request: Callable[..., Any]

    def get_decision_by_attempt(
        self, cluster_id: str, attempt_id: str
    ) -> CompletionDecision:
        event = self.get_event_by_attempt(cluster_id, attempt_id)
        decision = self.get_decision_by_event(event.event_key)
        if decision is None:
            raise NotFoundError(f"{cluster_id}/{attempt_id}")
        return decision

    def remote_command_cluster_health(
        self,
        cluster_id: str,
        *,
        now: datetime | None = None,
    ) -> dict:
        """What one cluster's executor needs to prove it is useful.

        Readiness for an executor is not "the process is up" -- it is
        "the backlog this cluster has is claimable by me". The two ways
        that fails silently are an executor that advertises fewer
        execution owners than the queued steps require, and an executor
        that is running but never claims. Both are visible only by
        comparing the advertised owners against the open backlog, which
        is what this returns.
        """

        observed_at = now or datetime.now(timezone.utc)
        commands = self._open_remote_command_candidates(cluster_id, None)
        pending_owner_counts: dict[str, int] = {}
        leased = 0
        oldest_unclaimed = 0.0
        oldest_unclaimed_owner: str | None = None
        for command in commands:
            if command.status is RemoteCommandStatus.LEASED:
                leased += 1
                continue
            if command.status is not RemoteCommandStatus.PENDING:
                continue
            owner = command.step.execution_owner
            pending_owner_counts[owner] = pending_owner_counts.get(owner, 0) + 1
            age = max(
                0.0,
                (observed_at - command.created_at).total_seconds(),
            )
            if age > oldest_unclaimed:
                oldest_unclaimed = age
                oldest_unclaimed_owner = owner
        return {
            "cluster_id": cluster_id,
            "open_total": len(commands),
            "leased_total": leased,
            "pending_total": sum(pending_owner_counts.values()),
            "pending_owner_counts": pending_owner_counts,
            "oldest_unclaimed_age_seconds": oldest_unclaimed,
            "oldest_unclaimed_execution_owner": oldest_unclaimed_owner,
        }

    def get_xid_events(self, event_ids: Iterable[str]) -> dict[str, XidEvent]:
        """Load the events behind a whole claimed batch at once.

        Correlation claims up to ``batch_size`` correlations and then
        needs the event behind each one. One round trip per correlation
        put the pass's latency at ``batch_size × RTT`` before any policy
        ran, which is what made a 100-deep batch miss its poll interval.
        Missing ids are absent from the result rather than raising, so
        the caller can log and skip them the same way it did before.
        """
        found: dict[str, XidEvent] = {}
        for event_id in event_ids:
            try:
                found[event_id] = self.get_xid_event(event_id)
            except NotFoundError:
                continue
        return found

    def list_xid_events_for_scopes(
        self,
        scopes: Iterable[tuple[str, str, datetime | None, datetime | None]],
    ) -> dict[tuple[str, str], list[XidEvent]]:
        """Companion candidates for several node windows in one call.

        ``scopes`` is an iterable of ``(cluster_id, node_id,
        observed_after, observed_before)``. The result is keyed by
        ``(cluster_id, node_id)`` and holds the union of the matching
        events for that node - the caller still narrows to each event's
        own window, because two events on the same node have different
        ones. Stores with a query planner override this with a single
        statement; here the loop is the whole point of the default.
        """
        grouped: dict[tuple[str, str], dict[str, XidEvent]] = {}
        for cluster_id, node_id, after, before in scopes:
            bucket = grouped.setdefault((cluster_id, node_id), {})
            for item in self.list_xid_events(
                cluster_id,
                node_id,
                observed_after=after,
                observed_before=before,
            ):
                bucket[item.event_id] = item
        return {scope: list(bucket.values()) for scope, bucket in grouped.items()}

    def complete_active_processor_requests_batch(self, completions):
        return [
            self.complete_active_processor_request(**completion)
            for completion in completions
        ]

    @contextmanager
    def processor_batch_transaction(self) -> Iterator[None]:
        yield

    def processor_queue_count_status(
        self,
    ) -> ProcessorQueueCountStatus:
        depth = int(self.processor_queue_stats()["depth"])
        return {
            "expected_total": depth,
            "counter_total": depth,
            "mismatched_clusters": 0,
            "ready": True,
        }

    def try_enqueue_processor_requests_batch(
        self,
        requests,
        *,
        max_depth: int,
        max_cluster_depth: int,
        reserved_fault_depth: int = 0,
        reserved_cluster_fault_depth: int = 0,
        global_admission_guard: int = 0,
    ):
        results: list[tuple[Any, str | None]] = []
        committed: list[str] = []
        for request in requests:
            try:
                result = self.try_enqueue_processor_request(
                    request,
                    max_depth=max_depth,
                    max_cluster_depth=max_cluster_depth,
                    reserved_fault_depth=reserved_fault_depth,
                    reserved_cluster_fault_depth=(reserved_cluster_fault_depth),
                    global_admission_guard=global_admission_guard,
                )
            except Exception as exc:
                # Rows admitted before the failure stay admitted; tell the
                # caller which ones (F-D9), as the Postgres batch does.
                raise PartialEnqueueError(committed=committed, cause=exc) from exc
            results.append(result)
            if result[0] is not None:
                committed.append(result[0].request_id)
        return results

    def save_attempt_observations_batch(
        self, observations: Sequence[AttemptObservation]
    ) -> list[bool]:
        return [
            self.save_attempt_observation(observation) for observation in observations
        ]

    def observe_telemetry_metric(self, latest: TelemetryMetricLatest) -> bool:
        return self.observe_telemetry_metrics([latest])[0]

    def save_collector_status(self, status: CollectorStatus) -> bool:
        return self.save_collector_statuses_batch([status])[0]

    def save_gpu_inventory_snapshot(
        self, snapshot: GpuInventorySnapshot
    ) -> GpuInventorySnapshot | None:
        previous = self.observe_gpu_inventory_snapshots([snapshot])[0]
        if previous is False:
            return self.get_gpu_inventory_snapshot(
                snapshot.cluster_id, snapshot.node_id
            )
        return snapshot

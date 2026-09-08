from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime
from typing import Any, Iterator, Literal, Sequence

from gpu_fault.attempt_observation_state import (
    attempt_observation_state_is_terminal,
    terminal_attempt_observation_state,
)
from gpu_fault.gpu_metric_models import GpuFindingState, GpuHealthFinding
from gpu_fault.gpu_metrics import (
    GpuInventorySnapshot,
    GpuMetricLatest,
    GpuMetricsIngestionResult,
)
from gpu_fault.models import TerminalEvent
from gpu_fault.store.shared.attempt_observation_support import (
    MemoryAttemptEventState,
    bound_attempt_observation_states,
    memory_attempt_terminal_event,
    memory_terminal_events,
)
from gpu_fault.store.shared.telemetry_models import (
    GpuFindingKey,
    GpuMetricKey,
    GpuMetricsBatchKey,
)
from gpu_fault.telemetry import (
    CollectorMetricsSnapshotRecord,
    CollectorStatus,
    WorkloadCoverageHeartbeat,
    coverage_heartbeat_supersedes,
)
from gpu_fault.telemetry_models import (
    TelemetryMetricLatest,
    WorkloadObservationState,
)
from gpu_fault.training_models import TrainingProgressHeartbeat, TrainingProgressState
from gpu_fault.watcher import AttemptObservation


class MemoryTelemetryMixin(MemoryAttemptEventState):
    # Attributes supplied by the composed concrete implementation. They are
    # spelled out rather than left as `Any` because every read below feeds a
    # typed return: an `Any` here would silently make the whole mixin
    # unchecked while still satisfying the annotations.
    _attempt_observations: dict[tuple[str, str], WorkloadObservationState]
    _collector_metrics_snapshot: CollectorMetricsSnapshotRecord | None
    _collector_statuses: dict[tuple[str, str, str], CollectorStatus]
    _gpu_finding_states: dict[GpuFindingKey, GpuFindingState]
    _gpu_inventory_snapshots: dict[tuple[str, str], GpuInventorySnapshot]
    _gpu_metric_latest: dict[GpuMetricKey, GpuMetricLatest]
    _gpu_metrics_batches: dict[GpuMetricsBatchKey, GpuMetricsIngestionResult]
    _telemetry_metric_latest: dict[tuple[str, str, str, str], TelemetryMetricLatest]
    _training_progress: dict[tuple[str, str, int], TrainingProgressState]
    _workload_coverage_heartbeats: dict[str, WorkloadCoverageHeartbeat]

    _gpu_finding_history: dict[str, GpuHealthFinding]
    _lock: Any

    def save_collector_metrics_snapshot(
        self, record: CollectorMetricsSnapshotRecord
    ) -> CollectorMetricsSnapshotRecord:
        with self._lock:
            self._collector_metrics_snapshot = record
        return record

    def get_collector_metrics_snapshot(self) -> CollectorMetricsSnapshotRecord | None:
        with self._lock:
            return self._collector_metrics_snapshot

    def get_gpu_metrics_batch(
        self, key: GpuMetricsBatchKey
    ) -> GpuMetricsIngestionResult | None:
        with self._lock:
            return self._gpu_metrics_batches.get(key)

    def save_gpu_metrics_batch(
        self,
        key: GpuMetricsBatchKey,
        result: GpuMetricsIngestionResult,
    ) -> GpuMetricsIngestionResult:
        with self._lock:
            existing = self._gpu_metrics_batches.get(key)
            if existing is not None:
                return existing.model_copy(update={"duplicate": True})
            self._gpu_metrics_batches[key] = result
            return result

    def observe_gpu_metrics(
        self, items: Sequence[tuple[GpuMetricKey, GpuMetricLatest]]
    ) -> list[GpuMetricLatest | Literal[False] | None]:
        """Save one batch and return each candidate's previous value."""
        with self._lock:
            results: list[GpuMetricLatest | Literal[False] | None] = []
            for key, latest in items:
                previous = self._gpu_metric_latest.get(key)
                if previous is not None and latest.observed_at <= previous.observed_at:
                    results.append(False)
                    continue
                self._gpu_metric_latest[key] = latest
                results.append(previous)
            return results

    def observe_gpu_metric(
        self, key: GpuMetricKey, latest: GpuMetricLatest
    ) -> GpuMetricLatest | Literal[False] | None:
        """Save a newer GPU sample and return its previous value.

        False means the candidate was stale or an exact retry. None means
        that the candidate established the first baseline.
        """
        return self.observe_gpu_metrics([(key, latest)])[0]

    def list_gpu_metrics_latest(
        self, cluster_id: str, node_id: str
    ) -> list[GpuMetricLatest]:
        with self._lock:
            return [
                item
                for key, item in self._gpu_metric_latest.items()
                if key[0] == cluster_id and key[1] == node_id
            ]

    def observe_gpu_inventory_snapshots(
        self, snapshots: Sequence[GpuInventorySnapshot]
    ) -> list[GpuInventorySnapshot | Literal[False] | None]:
        with self._lock:
            results: list[GpuInventorySnapshot | Literal[False] | None] = []
            for snapshot in snapshots:
                key = (
                    snapshot.cluster_id,
                    snapshot.node_id,
                )
                previous = self._gpu_inventory_snapshots.get(key)
                if (
                    previous is not None
                    and snapshot.observed_at <= previous.observed_at
                ):
                    results.append(False)
                    continue
                self._gpu_inventory_snapshots[key] = snapshot
                results.append(previous)
            return results

    def save_gpu_inventory_snapshot(
        self, snapshot: GpuInventorySnapshot
    ) -> GpuInventorySnapshot | None:
        previous = self.observe_gpu_inventory_snapshots([snapshot])[0]
        if previous is False:
            return self.get_gpu_inventory_snapshot(
                snapshot.cluster_id, snapshot.node_id
            )
        return snapshot

    def get_gpu_inventory_snapshot(
        self, cluster_id: str, node_id: str
    ) -> GpuInventorySnapshot | None:
        with self._lock:
            return self._gpu_inventory_snapshots.get((cluster_id, node_id))

    def get_gpu_finding_states(
        self, keys: Sequence[GpuFindingKey]
    ) -> list[GpuFindingState | None]:
        with self._lock:
            return [self._gpu_finding_states.get(key) for key in keys]

    def update_gpu_findings(
        self,
        items: Sequence[tuple[GpuFindingKey, GpuHealthFinding | None, datetime]],
    ) -> list[bool]:
        with self._lock:
            activated_values: list[bool] = []
            for key, finding, observed_at in items:
                previous = self._gpu_finding_states.get(key)
                if previous is not None and observed_at <= previous.observed_at:
                    activated_values.append(False)
                    continue
                previous_finding = previous.finding if previous is not None else None
                consecutive_breaches = (
                    previous.consecutive_breaches + 1
                    if finding is not None
                    and previous is not None
                    and previous.finding is not None
                    else 1
                    if finding is not None
                    else 0
                )
                state = GpuFindingState(
                    observed_at=observed_at,
                    finding=finding,
                    consecutive_breaches=consecutive_breaches,
                )
                self._gpu_finding_states[key] = state
                activated = finding is not None and (
                    previous_finding is None
                    or finding.severity != previous_finding.severity
                    or finding.automatic_action != previous_finding.automatic_action
                )
                activated_values.append(activated)
                if activated and finding is not None:
                    self._gpu_finding_history[finding.finding_id] = finding
            return activated_values

    def update_gpu_finding(
        self,
        key: GpuFindingKey,
        finding: GpuHealthFinding | None,
        observed_at: datetime,
    ) -> bool:
        return self.update_gpu_findings([(key, finding, observed_at)])[0]

    def get_gpu_finding_state(self, key: GpuFindingKey) -> GpuFindingState | None:
        return self.get_gpu_finding_states([key])[0]

    def list_gpu_findings(
        self,
        cluster_id: str,
        node_id: str,
        *,
        active_only: bool,
    ) -> list[GpuHealthFinding]:
        with self._lock:
            source = (
                [
                    state.finding
                    for state in self._gpu_finding_states.values()
                    if state.finding is not None
                ]
                if active_only
                else list(self._gpu_finding_history.values())
            )
            return [
                item
                for item in source
                if item.cluster_id == cluster_id and item.node_id == node_id
            ]

    def save_collector_statuses_batch(
        self, statuses: Sequence[CollectorStatus]
    ) -> list[bool]:
        with self._lock:
            results: list[bool] = []
            for status in statuses:
                key = (
                    status.cluster_id,
                    status.node_id,
                    status.collector.value,
                )
                previous = self._collector_statuses.get(key)
                if previous is not None and status.observed_at < previous.observed_at:
                    results.append(False)
                    continue
                if previous is not None:
                    status = status.model_copy(
                        update={
                            "last_success_at": (
                                status.last_success_at or previous.last_success_at
                            ),
                            "last_error_at": (
                                status.last_error_at or previous.last_error_at
                            ),
                        }
                    )
                self._collector_statuses[key] = status
                results.append(True)
            return results

    def save_collector_status(self, status: CollectorStatus) -> bool:
        return self.save_collector_statuses_batch([status])[0]

    def list_collector_statuses(
        self, cluster_id: str, node_id: str | None = None
    ) -> list[CollectorStatus]:
        with self._lock:
            return [
                item
                for key, item in self._collector_statuses.items()
                if key[0] == cluster_id and (node_id is None or key[1] == node_id)
            ]

    def observe_telemetry_metrics(
        self, items: Sequence[TelemetryMetricLatest]
    ) -> list[bool]:
        with self._lock:
            results: list[bool] = []
            for latest in items:
                key = (
                    latest.cluster_id,
                    latest.node_id,
                    latest.device or "node",
                    latest.name,
                )
                previous = self._telemetry_metric_latest.get(key)
                if previous is not None and latest.observed_at <= previous.observed_at:
                    results.append(False)
                    continue
                self._telemetry_metric_latest[key] = latest
                results.append(True)
            return results

    def observe_telemetry_metric(self, latest: TelemetryMetricLatest) -> bool:
        return self.observe_telemetry_metrics([latest])[0]

    def list_telemetry_metrics_latest(
        self, cluster_id: str, node_id: str
    ) -> list[TelemetryMetricLatest]:
        with self._lock:
            return [
                item
                for key, item in self._telemetry_metric_latest.items()
                if key[0] == cluster_id and key[1] == node_id
            ]

    def save_attempt_observation(self, observation: AttemptObservation) -> bool:
        key = (
            observation.cluster_id,
            observation.attempt_id,
        )
        with self._lock:
            event = memory_attempt_terminal_event(self, *key)
            if event is not None:
                self._terminalize_attempt_observation(event)
                return False
            previous = self._attempt_observations.get(key)
            if (
                previous is not None
                and observation.observed_at < previous.observation.observed_at
            ):
                return False
            self._attempt_observations[key] = WorkloadObservationState(
                first_observed_at=(
                    previous.first_observed_at
                    if previous is not None
                    else observation.observed_at
                ),
                observation=observation,
            )
            return True

    def _terminalize_attempt_observation(self, event: TerminalEvent) -> bool:
        key = (event.cluster_id, event.attempt_id)
        previous = self._attempt_observations.get(key)
        terminal = terminal_attempt_observation_state(event, previous)
        if terminal == previous:
            return False
        self._attempt_observations[key] = terminal
        return True

    def _reconcile_terminal_attempt_observations(self, limit: int) -> int:
        if limit < 1:
            return 0
        candidates = [
            item
            for item in sorted(
                memory_terminal_events(self),
                key=lambda value: (value.ended_at, value.event_key),
            )
            if not attempt_observation_state_is_terminal(
                self._attempt_observations.get((item.cluster_id, item.attempt_id))
            )
        ][:limit]
        return sum(
            int(self._terminalize_attempt_observation(event)) for event in candidates
        )

    def list_attempt_observations(self, cluster_id: str) -> list[AttemptObservation]:
        with self._lock:
            states = [
                item
                for key, item in self._attempt_observations.items()
                if key[0] == cluster_id
            ]
        return [item.observation for item in states]

    def list_attempt_observation_states(
        self,
        cluster_id: str | None = None,
        *,
        limit: int | None = None,
        newest_first: bool = False,
    ) -> list[WorkloadObservationState]:
        with self._lock:
            states = [
                item
                for key, item in self._attempt_observations.items()
                if cluster_id is None or key[0] == cluster_id
            ]
        return bound_attempt_observation_states(
            states,
            limit=limit,
            newest_first=newest_first,
        )

    def save_workload_coverage_heartbeat(
        self, heartbeat: WorkloadCoverageHeartbeat
    ) -> bool:
        """Keep one heartbeat per cluster; ``False`` when an older one arrives."""

        with self._lock:
            previous = self._workload_coverage_heartbeats.get(heartbeat.cluster_id)
            if previous is not None and not coverage_heartbeat_supersedes(
                heartbeat, previous
            ):
                return False
            self._workload_coverage_heartbeats[heartbeat.cluster_id] = heartbeat
            return True

    def get_workload_coverage_heartbeat(
        self, cluster_id: str
    ) -> WorkloadCoverageHeartbeat | None:
        with self._lock:
            return self._workload_coverage_heartbeats.get(cluster_id)

    def observe_training_progress(
        self, progress: TrainingProgressHeartbeat
    ) -> TrainingProgressHeartbeat | Literal[False] | None:
        key = (
            progress.cluster_id,
            progress.attempt_id,
            progress.rank,
        )
        with self._lock:
            previous = self._training_progress.get(key)
            if (
                previous is not None
                and progress.observed_at <= previous.heartbeat.observed_at
            ):
                return False
            advanced = (
                previous is None
                or progress.step is None
                or previous.heartbeat.step is None
                or progress.step > previous.heartbeat.step
            )
            self._training_progress[key] = TrainingProgressState(
                heartbeat=progress,
                last_progress_at=(
                    progress.observed_at
                    if previous is None or advanced
                    else previous.last_progress_at
                ),
            )
            return previous.heartbeat if previous is not None else None

    def list_training_progress(
        self, cluster_id: str, attempt_id: str | None = None
    ) -> list[TrainingProgressHeartbeat]:
        with self._lock:
            states = [
                item
                for key, item in self._training_progress.items()
                if key[0] == cluster_id and (attempt_id is None or key[1] == attempt_id)
            ]
        return [item.heartbeat for item in states]

    def list_training_progress_states(
        self, cluster_id: str, attempt_id: str | None = None
    ) -> list[TrainingProgressState]:
        with self._lock:
            return [
                item
                for key, item in self._training_progress.items()
                if key[0] == cluster_id and (attempt_id is None or key[1] == attempt_id)
            ]

    @contextmanager
    def collector_ingestion_transaction(
        self,
        _cluster_id: str,
        _node_id: str,
        _batch_id: str,
    ) -> Iterator[None]:
        yield

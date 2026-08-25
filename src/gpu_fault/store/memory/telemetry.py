from __future__ import annotations

from typing import Any, Callable

from contextlib import contextmanager

from gpu_fault.gpu_metric_models import GpuFindingState
from gpu_fault.telemetry_models import (
    WorkloadObservationState,
)
from gpu_fault.training_models import TrainingProgressState
from gpu_fault.store.shared.errors import NotFoundError


class MemoryTelemetryMixin:
    # Attributes supplied by the composed concrete implementation.
    _attempt_observations: Any
    _collector_statuses: Any
    _gpu_finding_states: Any
    _gpu_inventory_snapshots: Any
    _gpu_metric_latest: Any
    _gpu_metrics_batches: Any
    _telemetry_metric_latest: Any
    _training_progress: Any

    _get: Callable[..., Any]
    _gpu_finding_history: Any
    _lock: Any
    _put: Callable[..., Any]

    def save_collector_metrics_snapshot(self, record):
        if hasattr(self, "_put"):
            self._put(
                "collector_metrics_snapshot",
                "current",
                record,
            )
            return record
        with self._lock:
            self._collector_metrics_snapshot = record
        return record

    def get_collector_metrics_snapshot(self):
        if hasattr(self, "_get"):
            try:
                return self._get("collector_metrics_snapshot", "current")
            except NotFoundError:
                return None
        with self._lock:
            return self._collector_metrics_snapshot

    def get_gpu_metrics_batch(self, key):
        with self._lock:
            return self._gpu_metrics_batches.get(key)

    def save_gpu_metrics_batch(self, key, result):
        with self._lock:
            existing = self._gpu_metrics_batches.get(key)
            if existing is not None:
                return existing.model_copy(update={"duplicate": True})
            self._gpu_metrics_batches[key] = result
            return result

    def observe_gpu_metrics(self, items):
        """Save one batch and return each candidate's previous value."""
        with self._lock:
            results = []
            for key, latest in items:
                previous = self._gpu_metric_latest.get(key)
                if previous is not None and latest.observed_at <= previous.observed_at:
                    results.append(False)
                    continue
                self._gpu_metric_latest[key] = latest
                results.append(previous)
            return results

    def observe_gpu_metric(self, key, latest):
        """Save a newer GPU sample and return its previous value.

        False means the candidate was stale or an exact retry. None means
        that the candidate established the first baseline.
        """
        return self.observe_gpu_metrics([(key, latest)])[0]

    def list_gpu_metrics_latest(self, cluster_id: str, node_id: str):
        with self._lock:
            return [
                item
                for key, item in self._gpu_metric_latest.items()
                if key[0] == cluster_id and key[1] == node_id
            ]

    def observe_gpu_inventory_snapshots(self, snapshots):
        with self._lock:
            results = []
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

    def save_gpu_inventory_snapshot(self, snapshot):
        previous = self.observe_gpu_inventory_snapshots([snapshot])[0]
        if previous is False:
            return self.get_gpu_inventory_snapshot(
                snapshot.cluster_id, snapshot.node_id
            )
        return snapshot

    def get_gpu_inventory_snapshot(self, cluster_id: str, node_id: str):
        with self._lock:
            return self._gpu_inventory_snapshots.get((cluster_id, node_id))

    def get_gpu_finding_states(self, keys):
        with self._lock:
            return [self._gpu_finding_states.get(key) for key in keys]

    def update_gpu_findings(self, items) -> list[bool]:
        with self._lock:
            activated_values = []
            for key, finding, observed_at in items:
                previous = self._gpu_finding_states.get(key)
                if previous is not None and observed_at <= previous.observed_at:
                    activated_values.append(False)
                    continue
                previous_finding = previous.finding if previous is not None else None
                consecutive_breaches = (
                    previous.consecutive_breaches + 1
                    if finding is not None and previous_finding is not None
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
                if activated:
                    self._gpu_finding_history[finding.finding_id] = finding
            return activated_values

    def update_gpu_finding(self, key, finding, observed_at) -> bool:
        return self.update_gpu_findings([(key, finding, observed_at)])[0]

    def get_gpu_finding_state(self, key):
        return self.get_gpu_finding_states([key])[0]

    def list_gpu_findings(
        self,
        cluster_id: str,
        node_id: str,
        *,
        active_only: bool,
    ):
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

    def save_collector_statuses_batch(self, statuses) -> list[bool]:
        with self._lock:
            results = []
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

    def save_collector_status(self, status) -> bool:
        return self.save_collector_statuses_batch([status])[0]

    def list_collector_statuses(self, cluster_id: str, node_id: str | None = None):
        with self._lock:
            return [
                item
                for key, item in self._collector_statuses.items()
                if key[0] == cluster_id and (node_id is None or key[1] == node_id)
            ]

    def observe_telemetry_metrics(self, items) -> list[bool]:
        with self._lock:
            results = []
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

    def observe_telemetry_metric(self, latest) -> bool:
        return self.observe_telemetry_metrics([latest])[0]

    def list_telemetry_metrics_latest(self, cluster_id: str, node_id: str):
        with self._lock:
            return [
                item
                for key, item in self._telemetry_metric_latest.items()
                if key[0] == cluster_id and key[1] == node_id
            ]

    def save_attempt_observation(self, observation) -> bool:
        key = (
            observation.cluster_id,
            observation.attempt_id,
        )
        with self._lock:
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

    def save_attempt_observations_batch(self, observations):
        return [
            self.save_attempt_observation(observation) for observation in observations
        ]

    def list_attempt_observations(self, cluster_id: str):
        with self._lock:
            states = [
                item
                for key, item in self._attempt_observations.items()
                if key[0] == cluster_id
            ]
        return [item.observation for item in states]

    def list_attempt_observation_states(self, cluster_id: str | None = None):
        with self._lock:
            return [
                item
                for key, item in self._attempt_observations.items()
                if cluster_id is None or key[0] == cluster_id
            ]

    def observe_training_progress(self, progress):
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
                    progress.observed_at if advanced else previous.last_progress_at
                ),
            )
            return previous.heartbeat if previous is not None else None

    def list_training_progress(self, cluster_id: str, attempt_id: str | None = None):
        with self._lock:
            states = [
                item
                for key, item in self._training_progress.items()
                if key[0] == cluster_id and (attempt_id is None or key[1] == attempt_id)
            ]
        return [item.heartbeat for item in states]

    def list_training_progress_states(
        self, cluster_id: str, attempt_id: str | None = None
    ):
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
    ):
        yield

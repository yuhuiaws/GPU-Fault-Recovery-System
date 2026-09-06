from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Callable, ContextManager, Literal, Sequence, cast

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
    bound_attempt_observation_states,
)
from gpu_fault.store.shared.telemetry_models import (
    GpuFindingKey,
    GpuMetricKey,
    GpuMetricsBatchKey,
)
from gpu_fault.telemetry import CollectorStatus
from gpu_fault.telemetry_models import (
    TelemetryMetricLatest,
    WorkloadObservationState,
)
from gpu_fault.training_models import TrainingProgressHeartbeat, TrainingProgressState
from gpu_fault.watcher import AttemptObservation


class SqliteTelemetryMixin:
    # Attributes supplied by the composed concrete implementation. The row
    # accessors stay `Any`-returning because `SqliteCoreMixin` resolves the
    # model class from a string kind at run time; each method below casts the
    # result to the kind it asked for.
    _get_optional: Callable[[str, str], Any]
    _list: Callable[[str], list[Any]]
    _put: Callable[..., Any]
    _state_key: Callable[..., str]
    _state_transaction: Callable[[str], ContextManager[None]]

    @contextmanager
    def collector_ingestion_transaction(
        self, cluster_id: str, node_id: str, batch_id: str
    ) -> Iterator[None]:
        # One transaction for the whole ingestion of a batch (F-M1): the
        # finding state, the health-signal claim, the incident and the workflow
        # the store writes inside become savepoints of it and commit or roll
        # back together, as they do on PostgreSQL.
        with self._state_transaction(
            f"collector_ingestion/{cluster_id}/{node_id}/{batch_id}"
        ):
            yield

    def get_gpu_metrics_batch(
        self, key: GpuMetricsBatchKey
    ) -> GpuMetricsIngestionResult | None:
        return cast(
            "GpuMetricsIngestionResult | None",
            self._get_optional("gpu_metrics_batch", self._state_key(key)),
        )

    def save_gpu_metrics_batch(
        self,
        key: GpuMetricsBatchKey,
        result: GpuMetricsIngestionResult,
    ) -> GpuMetricsIngestionResult:
        storage_key = self._state_key(key)
        with self._state_transaction(f"gpu_metrics_batch/{storage_key}"):
            existing = cast(
                "GpuMetricsIngestionResult | None",
                self._get_optional("gpu_metrics_batch", storage_key),
            )
            if existing is not None:
                return existing.model_copy(update={"duplicate": True})
            self._put("gpu_metrics_batch", storage_key, result)
            return result

    def observe_gpu_metrics(
        self, items: Sequence[tuple[GpuMetricKey, GpuMetricLatest]]
    ) -> list[GpuMetricLatest | Literal[False] | None]:
        if not items:
            return []
        lock_key = self._state_key(items[0][0][:2])
        with self._state_transaction(f"gpu_metric_latest/batch/{lock_key}"):
            results: list[GpuMetricLatest | Literal[False] | None] = []
            for key, latest in items:
                storage_key = self._state_key(key)
                previous = cast(
                    "GpuMetricLatest | None",
                    self._get_optional("gpu_metric_latest", storage_key),
                )
                if previous is not None and latest.observed_at <= previous.observed_at:
                    results.append(False)
                    continue
                self._put("gpu_metric_latest", storage_key, latest)
                results.append(previous)
            return results

    def observe_gpu_metric(
        self, key: GpuMetricKey, latest: GpuMetricLatest
    ) -> GpuMetricLatest | Literal[False] | None:
        return self.observe_gpu_metrics([(key, latest)])[0]

    def list_gpu_metrics_latest(
        self, cluster_id: str, node_id: str
    ) -> list[GpuMetricLatest]:
        return [
            item
            for item in cast("list[GpuMetricLatest]", self._list("gpu_metric_latest"))
            if item.cluster_id == cluster_id and item.node_id == node_id
        ]

    def observe_gpu_inventory_snapshots(
        self, snapshots: Sequence[GpuInventorySnapshot]
    ) -> list[GpuInventorySnapshot | Literal[False] | None]:
        if not snapshots:
            return []
        storage_keys = [
            self._state_key((snapshot.cluster_id, snapshot.node_id))
            for snapshot in snapshots
        ]
        with self._state_transaction(
            "gpu_inventory_snapshot/batch/"
            + self._state_key(
                sorted(
                    {
                        (
                            snapshot.cluster_id,
                            snapshot.node_id,
                        )
                        for snapshot in snapshots
                    }
                )
            )
        ):
            current_by_key: dict[str, GpuInventorySnapshot | None] = {
                key: cast(
                    "GpuInventorySnapshot | None",
                    self._get_optional("gpu_inventory_snapshot", key),
                )
                for key in set(storage_keys)
            }
            results: list[GpuInventorySnapshot | Literal[False] | None] = []
            final_by_key: dict[str, GpuInventorySnapshot] = {}
            for storage_key, snapshot in zip(storage_keys, snapshots, strict=True):
                previous = current_by_key.get(storage_key)
                if (
                    previous is not None
                    and snapshot.observed_at <= previous.observed_at
                ):
                    results.append(False)
                    continue
                results.append(previous)
                current_by_key[storage_key] = snapshot
                final_by_key[storage_key] = snapshot
            for (
                storage_key,
                snapshot,
            ) in final_by_key.items():
                self._put(
                    "gpu_inventory_snapshot",
                    storage_key,
                    snapshot,
                )
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
        return cast(
            "GpuInventorySnapshot | None",
            self._get_optional(
                "gpu_inventory_snapshot",
                self._state_key((cluster_id, node_id)),
            ),
        )

    def get_gpu_finding_states(
        self, keys: Sequence[GpuFindingKey]
    ) -> list[GpuFindingState | None]:
        return [
            cast(
                "GpuFindingState | None",
                self._get_optional("gpu_finding_state", self._state_key(key)),
            )
            for key in keys
        ]

    def update_gpu_findings(
        self,
        items: Sequence[tuple[GpuFindingKey, GpuHealthFinding | None, datetime]],
    ) -> list[bool]:
        if not items:
            return []
        lock_key = self._state_key(items[0][0][:2])
        with self._state_transaction(f"gpu_finding_state/batch/{lock_key}"):
            activated_values: list[bool] = []
            for key, finding, observed_at in items:
                storage_key = self._state_key(key)
                previous = cast(
                    "GpuFindingState | None",
                    self._get_optional("gpu_finding_state", storage_key),
                )
                if previous is not None and observed_at <= previous.observed_at:
                    activated_values.append(False)
                    continue
                activated = finding is not None and (
                    previous is None
                    or previous.finding is None
                    or finding.severity != previous.finding.severity
                    or finding.automatic_action != previous.finding.automatic_action
                )
                consecutive_breaches = (
                    previous.consecutive_breaches + 1
                    if finding is not None
                    and previous is not None
                    and previous.finding is not None
                    else 1
                    if finding is not None
                    else 0
                )
                self._put(
                    "gpu_finding_state",
                    storage_key,
                    GpuFindingState(
                        observed_at=observed_at,
                        finding=finding,
                        consecutive_breaches=consecutive_breaches,
                    ),
                )
                activated_values.append(activated)
                if activated and finding is not None:
                    self._put(
                        "gpu_finding_history",
                        finding.finding_id,
                        finding,
                    )
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
        source = (
            [
                state.finding
                for state in cast(
                    "list[GpuFindingState]", self._list("gpu_finding_state")
                )
                if state.finding is not None
            ]
            if active_only
            else cast("list[GpuHealthFinding]", self._list("gpu_finding_history"))
        )
        return [
            item
            for item in source
            if item.cluster_id == cluster_id and item.node_id == node_id
        ]

    def save_collector_statuses_batch(
        self, statuses: Sequence[CollectorStatus]
    ) -> list[bool]:
        if not statuses:
            return []
        storage_keys = [
            self._state_key(
                (
                    status.cluster_id,
                    status.node_id,
                    status.collector.value,
                )
            )
            for status in statuses
        ]
        with self._state_transaction(
            "collector_status/batch/" + self._state_key(sorted(set(storage_keys)))
        ):
            current_by_key: dict[str, CollectorStatus | None] = {
                key: cast(
                    "CollectorStatus | None",
                    self._get_optional("collector_status", key),
                )
                for key in set(storage_keys)
            }
            results: list[bool] = []
            final_by_key: dict[str, CollectorStatus] = {}
            for storage_key, status in zip(storage_keys, statuses, strict=True):
                previous = current_by_key.get(storage_key)
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
                current_by_key[storage_key] = status
                final_by_key[storage_key] = status
                results.append(True)
            for storage_key, status in final_by_key.items():
                self._put("collector_status", storage_key, status)
            return results

    def save_collector_status(self, status: CollectorStatus) -> bool:
        return self.save_collector_statuses_batch([status])[0]

    def list_collector_statuses(
        self, cluster_id: str, node_id: str | None = None
    ) -> list[CollectorStatus]:
        return [
            item
            for item in cast("list[CollectorStatus]", self._list("collector_status"))
            if item.cluster_id == cluster_id
            and (node_id is None or item.node_id == node_id)
        ]

    def observe_telemetry_metrics(
        self, items: Sequence[TelemetryMetricLatest]
    ) -> list[bool]:
        if not items:
            return []
        storage_keys = [
            self._state_key(
                (
                    latest.cluster_id,
                    latest.node_id,
                    latest.device or "node",
                    latest.name,
                )
            )
            for latest in items
        ]
        with self._state_transaction(
            "telemetry_metric_latest/batch/"
            + self._state_key(sorted(set(storage_keys)))
        ):
            current_by_key: dict[str, TelemetryMetricLatest | None] = {
                key: cast(
                    "TelemetryMetricLatest | None",
                    self._get_optional("telemetry_metric_latest", key),
                )
                for key in set(storage_keys)
            }
            results: list[bool] = []
            final_by_key: dict[str, TelemetryMetricLatest] = {}
            for storage_key, latest in zip(storage_keys, items, strict=True):
                previous = current_by_key.get(storage_key)
                if previous is not None and latest.observed_at <= previous.observed_at:
                    results.append(False)
                    continue
                current_by_key[storage_key] = latest
                final_by_key[storage_key] = latest
                results.append(True)
            for storage_key, latest in final_by_key.items():
                self._put(
                    "telemetry_metric_latest",
                    storage_key,
                    latest,
                )
            return results

    def observe_telemetry_metric(self, latest: TelemetryMetricLatest) -> bool:
        return self.observe_telemetry_metrics([latest])[0]

    def list_telemetry_metrics_latest(
        self, cluster_id: str, node_id: str
    ) -> list[TelemetryMetricLatest]:
        return [
            item
            for item in cast(
                "list[TelemetryMetricLatest]", self._list("telemetry_metric_latest")
            )
            if item.cluster_id == cluster_id and item.node_id == node_id
        ]

    def save_attempt_observation(self, observation: AttemptObservation) -> bool:
        storage_key = self._state_key((observation.cluster_id, observation.attempt_id))
        with self._state_transaction(f"attempt_observation/{storage_key}"):
            event = cast(
                "TerminalEvent | None",
                self._get_optional(
                    "event",
                    (
                        f"{observation.cluster_id}/{observation.attempt_id}/"
                        "TrainingAttemptTerminal"
                    ),
                ),
            )
            if event is not None:
                self._terminalize_attempt_observation(event)
                return False
            previous = cast(
                "WorkloadObservationState | None",
                self._get_optional("attempt_observation", storage_key),
            )
            if (
                previous is not None
                and observation.observed_at < previous.observation.observed_at
            ):
                return False
            self._put(
                "attempt_observation",
                storage_key,
                WorkloadObservationState(
                    first_observed_at=(
                        previous.first_observed_at
                        if previous is not None
                        else observation.observed_at
                    ),
                    observation=observation,
                ),
            )
            return True

    def _terminalize_attempt_observation(self, event: TerminalEvent) -> bool:
        storage_key = self._state_key((event.cluster_id, event.attempt_id))
        previous = cast(
            "WorkloadObservationState | None",
            self._get_optional("attempt_observation", storage_key),
        )
        terminal = terminal_attempt_observation_state(event, previous)
        if terminal == previous:
            return False
        self._put("attempt_observation", storage_key, terminal)
        return True

    def _reconcile_terminal_attempt_observations(self, limit: int) -> int:
        if limit < 1:
            return 0
        candidates = [
            item
            for item in sorted(
                cast("list[TerminalEvent]", self._list("event")),
                key=lambda value: (value.ended_at, value.event_key),
            )
            if not attempt_observation_state_is_terminal(
                self._get_optional(
                    "attempt_observation",
                    self._state_key((item.cluster_id, item.attempt_id)),
                )
            )
        ][:limit]
        return sum(
            int(self._terminalize_attempt_observation(event)) for event in candidates
        )

    def list_attempt_observations(self, cluster_id: str) -> list[AttemptObservation]:
        return [
            item.observation
            for item in cast(
                "list[WorkloadObservationState]", self._list("attempt_observation")
            )
            if item.observation.cluster_id == cluster_id
        ]

    def list_attempt_observation_states(
        self,
        cluster_id: str | None = None,
        *,
        limit: int | None = None,
        newest_first: bool = False,
    ) -> list[WorkloadObservationState]:
        return bound_attempt_observation_states(
            [
                item
                for item in cast(
                    "list[WorkloadObservationState]", self._list("attempt_observation")
                )
                if cluster_id is None or item.observation.cluster_id == cluster_id
            ],
            limit=limit,
            newest_first=newest_first,
        )

    def observe_training_progress(
        self, progress: TrainingProgressHeartbeat
    ) -> TrainingProgressHeartbeat | Literal[False] | None:
        storage_key = self._state_key(
            (
                progress.cluster_id,
                progress.attempt_id,
                progress.rank,
            )
        )
        with self._state_transaction(f"training_progress/{storage_key}"):
            previous = cast(
                "TrainingProgressState | None",
                self._get_optional("training_progress", storage_key),
            )
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
            self._put(
                "training_progress",
                storage_key,
                TrainingProgressState(
                    heartbeat=progress,
                    last_progress_at=(
                        progress.observed_at
                        if previous is None or advanced
                        else previous.last_progress_at
                    ),
                ),
            )
            return previous.heartbeat if previous is not None else None

    def list_training_progress(
        self, cluster_id: str, attempt_id: str | None = None
    ) -> list[TrainingProgressHeartbeat]:
        return [
            item.heartbeat
            for item in cast(
                "list[TrainingProgressState]", self._list("training_progress")
            )
            if item.heartbeat.cluster_id == cluster_id
            and (attempt_id is None or item.heartbeat.attempt_id == attempt_id)
        ]

    def list_training_progress_states(
        self, cluster_id: str, attempt_id: str | None = None
    ) -> list[TrainingProgressState]:
        return [
            item
            for item in cast(
                "list[TrainingProgressState]", self._list("training_progress")
            )
            if item.heartbeat.cluster_id == cluster_id
            and (attempt_id is None or item.heartbeat.attempt_id == attempt_id)
        ]

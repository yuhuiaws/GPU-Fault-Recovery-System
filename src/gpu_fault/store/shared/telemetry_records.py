"""Telemetry rows kept as whole JSON records, shared by the key/value stores.

SQLite stores all of its telemetry this way. PostgreSQL moved the hot rows --
GPU metric batches and latest samples, attempt observations, training
progress -- into dedicated tables, and its overrides fall back to these
templates with ``super()`` when ``hot_state_mode == "legacy"``; that is why
this mixin must follow the Postgres telemetry mixins in ``PostgresStore``'s
MRO. The collector metrics snapshot (one row, ``current``) and a node's GPU
inventory snapshot are read this way on both backends.
"""

from __future__ import annotations

from typing import Literal, Sequence, cast

from pydantic import ValidationError

from gpu_fault.attempt_observation_state import (
    attempt_observation_state_is_terminal,
    terminal_attempt_observation_state,
)
from gpu_fault.gpu_metrics import (
    GpuInventorySnapshot,
    GpuMetricLatest,
    GpuMetricsIngestionResult,
)
from gpu_fault.models import TerminalEvent
from gpu_fault.store.shared.attempt_observation_support import (
    bound_attempt_observation_states,
)
from gpu_fault.store.shared.errors import NotFoundError
from gpu_fault.store.shared.primitives import (
    GetOptionalRecord,
    GetRecord,
    ListRecords,
    PutRecord,
    StateKey,
    StateTransaction,
)
from gpu_fault.store.shared.telemetry_models import (
    GpuMetricKey,
    GpuMetricsBatchKey,
)
from gpu_fault.telemetry import (
    CollectorMetricsSnapshotRecord,
    WorkloadCoverageHeartbeat,
    coverage_heartbeat_supersedes,
    note_usable_coverage_heartbeat,
    warn_unusable_coverage_heartbeat,
)
from gpu_fault.telemetry_models import WorkloadObservationState
from gpu_fault.training_models import TrainingProgressHeartbeat, TrainingProgressState
from gpu_fault.watcher import AttemptObservation


class SharedTelemetryRecordMixin:
    # Attributes supplied by the composed concrete implementation.
    _get: GetRecord
    _get_optional: GetOptionalRecord
    _list: ListRecords
    _put: PutRecord
    _state_key: StateKey
    _state_transaction: StateTransaction

    def save_collector_metrics_snapshot(
        self, record: CollectorMetricsSnapshotRecord
    ) -> CollectorMetricsSnapshotRecord:
        self._put(
            "collector_metrics_snapshot",
            "current",
            record,
        )
        return record

    def get_collector_metrics_snapshot(self) -> CollectorMetricsSnapshotRecord | None:
        try:
            snapshot: CollectorMetricsSnapshotRecord = self._get(
                "collector_metrics_snapshot", "current"
            )
            return snapshot
        except NotFoundError:
            return None

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

    def list_gpu_metrics_latest(
        self, cluster_id: str, node_id: str
    ) -> list[GpuMetricLatest]:
        return [
            item
            for item in cast("list[GpuMetricLatest]", self._list("gpu_metric_latest"))
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

    def save_workload_coverage_heartbeat(
        self, heartbeat: WorkloadCoverageHeartbeat
    ) -> bool:
        """Keep one heartbeat per cluster; ``False`` when an older one arrives.

        The row is coverage evidence, so the newest ``observed_at`` has to win
        outright: after a watcher rollout the replaced Pod's last in-flight
        heartbeat can land behind the new Pod's first one, and letting it
        overwrite the row would age coverage backwards.
        """

        storage_key = self._state_key((heartbeat.cluster_id,))
        with self._state_transaction(f"workload_coverage_heartbeat/{storage_key}"):
            previous = self._read_coverage_heartbeat(heartbeat.cluster_id, storage_key)
            if previous is not None and not coverage_heartbeat_supersedes(
                heartbeat, previous
            ):
                return False
            self._put("workload_coverage_heartbeat", storage_key, heartbeat)
            # The row is readable again, so the next unusable one is news.
            note_usable_coverage_heartbeat(heartbeat.cluster_id)
            return True

    def get_workload_coverage_heartbeat(
        self, cluster_id: str
    ) -> WorkloadCoverageHeartbeat | None:
        return self._read_coverage_heartbeat(cluster_id, self._state_key((cluster_id,)))

    def _read_coverage_heartbeat(
        self, cluster_id: str, storage_key: str
    ) -> WorkloadCoverageHeartbeat | None:
        """The stored row, or ``None`` when it cannot be read as one.

        A payload an older release wrote -- a naive ``observed_at``, before the
        model required a timezone -- no longer validates. Coverage is read on
        the fault ingest path, where raising would fail the ingest of every
        fault on the cluster, and it is read again by the writer, where
        answering ``None`` is what lets the next heartbeat replace the row
        instead of leaving it poisoned for ever.

        Decoding fails in more ways than validation: ``_get`` ends in
        ``self._models[kind].model_validate_json(payload)``, so a build whose
        ``record_models()`` does not carry this kind raises ``KeyError`` and a
        payload that is not a JSON object can raise before pydantic wraps it.
        Every one of those is the same fact -- there is no readable coverage --
        and answering it is what keeps this row out of the ingest path's
        failure modes. A store or connection error is *not* caught: that is a
        broken store, not a broken row, and the caller must see it.
        """

        try:
            heartbeat = cast(
                "WorkloadCoverageHeartbeat | None",
                self._get_optional("workload_coverage_heartbeat", storage_key),
            )
        except (ValidationError, TypeError, ValueError, KeyError):
            warn_unusable_coverage_heartbeat(
                cluster_id, "the stored payload cannot be read as a coverage heartbeat"
            )
            return None
        if heartbeat is not None:
            note_usable_coverage_heartbeat(cluster_id)
        return heartbeat

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

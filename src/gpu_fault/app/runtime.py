from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class EventLoopLag:
    count: int = 0
    total_seconds: float = 0.0
    max_seconds: float = 0.0

    def observe(self, lag_seconds: float) -> None:
        self.count += 1
        self.total_seconds += lag_seconds
        self.max_seconds = max(self.max_seconds, lag_seconds)

    def snapshot(self) -> tuple[int, float, float]:
        return (
            self.count,
            self.total_seconds,
            self.max_seconds,
        )


@dataclass
class ProcessorDispatchState:
    telemetry_coalesced: int = 0
    oversize_rejections: int = 0
    telemetry_spool_admitted: int = 0
    telemetry_spool_coalesced: int = 0


@dataclass
class AppRuntime:
    context: Any
    processor: Any | None
    processor_replay_tracker: Any
    store_io: Any
    decode_io: Any
    fault_store_io: Any
    evidence_store_io: Any
    fault_decode_io: Any
    telemetry_spool_store_io: Any
    processor_admission_batcher: Any
    fault_admission_batcher: Any
    evidence_admission_batcher: Any
    telemetry_spool_batcher: Any
    background_services_enabled: bool
    telemetry_spool_enabled: bool
    processor_queue_bypass_enabled: bool
    processor_queue_bypass_paths: set[str]
    processor_admission_rejections: dict[str, int]
    processor_admission_rejections_by_path: dict[str, int]
    processor_queue_bypasses_by_path: dict[str, int]
    telemetry_spool_admitted_by_path: dict[str, int]
    telemetry_spool_rejections: dict[str, int]
    ingress_backpressure_rejections: dict[str, int]
    event_loop_lag_snapshot: Callable[[], tuple[int, float, float]]
    dispatch_state: ProcessorDispatchState
    collector_metrics_snapshot: Any
    # Shared, bounded store scans for the /metrics render path. Left optional so
    # a plugin contributor or a test that builds this runtime directly still
    # renders; the accessor falls back to an unshared read.
    metric_scan_cache: Any = None

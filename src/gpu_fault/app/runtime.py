from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

# Upper bounds of the lag histogram. The monitor samples twice a second, so
# the buckets have to resolve a few milliseconds of ordinary scheduling, the
# 0.15-0.6 s of a gen-2 garbage collection, and the multi-second stall a
# liveness probe would die on (timeoutSeconds 10): a running max alone cannot
# tell "one GC pause an hour ago" from "the loop is stalling every second".
EVENT_LOOP_LAG_BUCKETS: tuple[float, ...] = (
    0.005,
    0.01,
    0.025,
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.5,
    5.0,
    10.0,
)


@dataclass
class EventLoopLag:
    count: int = 0
    total_seconds: float = 0.0
    max_seconds: float = 0.0
    bucket_counts: list[int] = field(
        default_factory=lambda: [0] * len(EVENT_LOOP_LAG_BUCKETS)
    )

    def observe(self, lag_seconds: float) -> None:
        self.count += 1
        self.total_seconds += lag_seconds
        self.max_seconds = max(self.max_seconds, lag_seconds)
        for index, bound in enumerate(EVENT_LOOP_LAG_BUCKETS):
            if lag_seconds <= bound:
                self.bucket_counts[index] += 1
                break

    def snapshot(self) -> tuple[int, float, float]:
        return (
            self.count,
            self.total_seconds,
            self.max_seconds,
        )

    def histogram(self) -> list[tuple[float, int]]:
        """Cumulative ``(le, count)`` pairs, ``+Inf`` excluded (it is ``count``)."""

        cumulative = 0
        rows: list[tuple[float, int]] = []
        for bound, observed in zip(
            EVENT_LOOP_LAG_BUCKETS, self.bucket_counts, strict=True
        ):
            cumulative += observed
            rows.append((bound, cumulative))
        return rows


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
    # ``{"nul": n}`` from AdmissionRuntime (E-3); rendered as
    # gpu_fault_ingress_decode_rejections_total{reason}.
    decode_rejections: Any = None
    # ``EventLoopLag.histogram``; optional so a runtime built without it (a
    # plugin test, an older assembly) still renders the summary fields.
    event_loop_lag_histogram: Callable[[], list[tuple[float, int]]] | None = None

"""Persistent producer edges must carry the same temperature actions as ingestion."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from gpu_fault.channel_registry import GPU_METRICS_PATH
from gpu_fault.collectors.gpu.dcgm import DCGM_METRICS, DcgmMetricsCollector
from gpu_fault.collectors.models import CollectorContext
from gpu_fault.collectors.scheduling import next_stable_phase
from gpu_fault.collectors.sinks import CollectorError
from gpu_fault.gpu_metrics import (
    GpuMetricBatch,
    GpuMetricSample,
    GpuMetricsIngestionResult,
    GpuMetricsService,
)

CLUSTER = "alignment-dcgm"
NODE = "alignment-node"
SUMMARY_SECONDS = 86_400
START = next_stable_phase(
    datetime(2026, 9, 15, tzinfo=timezone.utc),
    cluster_id=CLUSTER,
    node_id=NODE,
    channel="gpu-metrics",
    interval_seconds=SUMMARY_SECONDS,
) + timedelta(seconds=1)


def text(temperatures: dict[str, dict[str, float]]) -> str:
    defaults = {
        "gpu_temperature_c": 40,
        "memory_temperature_c": 45,
        "power_limit_w": 700,
    }
    return "".join(
        f'{name}{{UUID="{device}"}} '
        f"{values.get(canonical, defaults.get(canonical, 0))}\n"
        for device, values in temperatures.items()
        for name, (canonical, _unit) in DCGM_METRICS.items()
    )


def limit(name: str, value: float, device: str = "GPU-a") -> GpuMetricSample:
    return GpuMetricSample(
        metric_name=name, canonical_name=name, value=value, gpu_uuid=device
    )


class IngestingSink:
    def __init__(self) -> None:
        self.service = GpuMetricsService()
        self.batches: list[GpuMetricBatch] = []
        self.results: list[GpuMetricsIngestionResult] = []
        self.rejection: CollectorError | None = None

    def post(self, path: str, payload: dict) -> dict:
        assert path == GPU_METRICS_PATH, "temperature replay must stay on GPU metrics"
        if self.rejection is not None:
            raise self.rejection
        batch = GpuMetricBatch.model_validate(payload)
        assert not batch.collection_errors, "the fixture must be a complete scrape"
        self.batches.append(batch)
        result = self.service.ingest(batch)
        self.results.append(result)
        return result.model_dump(mode="json")


class Series:
    def __init__(self) -> None:
        self.sink = IngestingSink()
        self.collector = DcgmMetricsCollector(
            self.sink,
            CollectorContext(cluster_id=CLUSTER),
            node_id=NODE,
            health_summary_seconds=SUMMARY_SECONDS,
            edge_confirmation_samples=3,
        )
        self.next_at = START

    def tick(
        self,
        values: dict[str, float],
        *,
        limits: list[GpuMetricSample] | None = None,
        other_devices: dict[str, dict[str, float]] | None = None,
    ) -> GpuMetricBatch:
        at = self.next_at
        self.next_at += timedelta(seconds=15)
        return self.collector.collect_text(
            text({"GPU-a": values, **(other_devices or {})}),
            observed_at=at,
            extra_samples=limits,
        )

    def confirm(
        self, values: dict[str, float], *, limits: list[GpuMetricSample] | None = None
    ) -> None:
        before = len(self.sink.batches)
        for _ in range(2):
            self.tick(values, limits=limits)
            assert len(self.sink.batches) == before, (
                "a new action must not borrow the preceding action's confirmations"
            )
        self.tick(values, limits=limits)
        assert len(self.sink.batches) == before + 1, (
            "a confirmed semantic transition must reach the control plane"
        )
        assert self.sink.batches[-1].edge_filter_reasons == ["candidate-confirmed"]


@pytest.mark.parametrize(
    "metric,warning,critical",
    [("gpu_temperature_c", 86, 91), ("memory_temperature_c", 91, 96)],
)
def test_persistent_warning_then_critical_delivers_each_action_once(
    metric: str, warning: float, critical: float
) -> None:
    series = Series()
    series.tick({})
    series.confirm({metric: warning})
    warning_finding = series.sink.results[-1].new_findings
    assert [(item.severity, item.automatic_action) for item in warning_finding] == [
        ("WARNING", "RUN_DIAGNOSTICS")
    ]
    for _ in range(4):
        series.tick({metric: warning + 0.1})
    assert len(series.sink.batches) == 2, "stable warnings must remain filtered"
    series.confirm({metric: critical})
    critical_finding = series.sink.results[-1].new_findings
    assert [(item.severity, item.automatic_action) for item in critical_finding] == [
        ("CRITICAL", "DRAIN")
    ]
    for _ in range(4):
        series.tick({metric: critical + 0.1})
    assert len(series.sink.batches) == 3, "stable critical findings must not flood"
    series.tick({})
    assert series.sink.batches[-1].edge_filter_reasons == ["candidate-recovered"]
    assert series.sink.service.findings(CLUSTER, NODE) == []


@pytest.mark.parametrize(
    "metric,limits,warning,critical",
    [
        ("gpu_temperature_c", [limit("gpu_slowdown_temperature_c", 70)], 65, 70),
        ("gpu_temperature_c", [limit("gpu_max_operating_temperature_c", 72)], 67, 72),
        ("gpu_temperature_c", [limit("gpu_shutdown_temperature_c", 78)], 70, 75),
        (
            "gpu_temperature_c",
            [
                limit("gpu_slowdown_temperature_c", 78),
                limit("gpu_max_operating_temperature_c", 68),
            ],
            68,
            78,
        ),
        (
            "memory_temperature_c",
            [limit("memory_max_operating_temperature_c", 80)],
            75,
            80,
        ),
    ],
    ids=["slowdown", "max-operating", "shutdown-margin", "lower-max", "memory-max"],
)
def test_device_thresholds_below_fallback_share_producer_and_ingestion_actions(
    metric: str, limits: list[GpuMetricSample], warning: float, critical: float
) -> None:
    series = Series()
    series.tick({}, limits=limits)
    series.confirm({metric: warning}, limits=limits)
    finding = series.sink.results[-1].new_findings[0]
    assert (finding.severity, finding.automatic_action, finding.threshold_value) == (
        "WARNING",
        "RUN_DIAGNOSTICS",
        warning,
    )
    assert finding.threshold_source == "NVIDIA_DEVICE_LIMIT"
    series.confirm({metric: critical}, limits=limits)
    finding = series.sink.results[-1].new_findings[0]
    assert (finding.severity, finding.automatic_action, finding.threshold_value) == (
        "CRITICAL",
        "DRAIN",
        critical,
    )


def test_device_limits_cannot_bleed_into_another_gpu() -> None:
    series = Series()
    limits = [
        limit("gpu_slowdown_temperature_c", 100),
        limit("gpu_slowdown_temperature_c", 70, "GPU-b"),
    ]
    series.tick({}, limits=limits, other_devices={"GPU-b": {}})
    for _ in range(3):
        series.tick(
            {"gpu_temperature_c": 76},
            limits=limits,
            other_devices={"GPU-b": {"gpu_temperature_c": 76}},
        )
    assert len(series.sink.batches) == 2
    assert [
        (item.gpu_uuid, item.severity, item.automatic_action)
        for item in series.sink.results[-1].new_findings
    ] == [("GPU-b", "CRITICAL", "DRAIN")], "limits must bind to the sample's device"


def test_newly_discovered_limit_creates_an_edge_without_temperature_change() -> None:
    series = Series()
    series.tick({"gpu_temperature_c": 76})
    for _ in range(3):
        series.tick({"gpu_temperature_c": 76})
    assert len(series.sink.batches) == 1
    series.confirm(
        {"gpu_temperature_c": 76}, limits=[limit("gpu_slowdown_temperature_c", 70)]
    )
    assert series.sink.results[-1].new_findings[0].automatic_action == "DRAIN"


def test_brief_critical_transition_is_not_confirmation_or_false_recovery() -> None:
    series = Series()
    series.tick({})
    series.confirm({"gpu_temperature_c": 86})
    series.tick({"gpu_temperature_c": 91})
    series.tick({"gpu_temperature_c": 86})
    assert len(series.sink.batches) == 2, "a one-scrape action change is not confirmed"
    series.tick({})
    assert len(series.sink.batches) == 3, (
        "an unconfirmed transition must not erase the last delivered warning"
    )
    assert series.sink.batches[-1].edge_filter_reasons == ["candidate-recovered"]
    assert series.sink.service.findings(CLUSTER, NODE) == []


def test_warning_samples_cannot_confirm_a_subsequent_critical_action() -> None:
    series = Series()
    series.tick({})
    series.tick({"gpu_temperature_c": 86})
    series.tick({"gpu_temperature_c": 86})
    series.confirm({"gpu_temperature_c": 91})
    assert [
        item.automatic_action
        for result in series.sink.results
        for item in result.new_findings
    ] == ["DRAIN"]


@pytest.mark.parametrize("transition", ["critical", "recovered"])
def test_rejected_semantic_edge_is_retried_on_the_next_unchanged_scrape(
    transition: str,
) -> None:
    series = Series()
    series.tick({})
    series.confirm({"gpu_temperature_c": 86})
    values = {"gpu_temperature_c": 91} if transition == "critical" else {}
    if transition == "critical":
        series.tick(values)
        series.tick(values)
    series.sink.rejection = CollectorError("private sink rejected the test batch")
    with pytest.raises(CollectorError):
        series.tick(values)
    assert len(series.sink.batches) == 2
    series.sink.rejection = None
    series.tick(values)
    assert len(series.sink.batches) == 3, "unaccepted semantic edges must remain open"
    assert series.sink.batches[-1].edge_filter_reasons == [
        "candidate-confirmed" if transition == "critical" else "candidate-recovered"
    ]

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from threading import Barrier

import pytest
from pydantic import ValidationError

from gpu_fault.gpu_metrics import (
    GpuHealthFinding,
    GpuHealthSeverity,
    GpuInventoryDevice,
    GpuInventorySnapshot,
    GpuMetricBatch,
    GpuMetricSample,
    GpuMetricSource,
    GpuMetricsService,
    GpuMetricsThresholds,
)
from gpu_fault.store import InMemoryStore, SqliteStore
from tests._builders import (
    asgi_client,
    build_context,
    build_store,
    copy_model,
    gpu_metric_batch,
)

NOW = datetime(2026, 7, 20, 13, 0, tzinfo=timezone.utc)


def sample(name: str, value: float) -> GpuMetricSample:
    return GpuMetricSample(
        metric_name=f"DCGM_FI_DEV_{name.upper()}",
        canonical_name=name,
        value=value,
        gpu_index="0",
        gpu_uuid="GPU-a",
        pci_bdf="0000:b9:00",
        labels={"modelName": "H100"},
    )


def _finding(finding_id: str, observed_at: datetime) -> GpuHealthFinding:
    return GpuHealthFinding(
        finding_id=finding_id,
        cluster_id="cluster-a",
        node_id="node-a",
        observed_at=observed_at,
        severity=GpuHealthSeverity.WARNING,
        reason=finding_id,
        canonical_name="gpu_temperature_c",
        value=86,
    )


def test_gpu_finding_history_retention_is_bounded(tmp_path) -> None:
    stores = [build_store(), SqliteStore(str(tmp_path / "finding-history.db"))]
    try:
        for store in stores:
            old = _finding("finding-old", NOW - timedelta(days=31))
            recent = _finding("finding-recent", NOW - timedelta(days=1))
            store.update_gpu_finding(
                ("cluster-a", "node-a", "old"), old, old.observed_at
            )
            store.update_gpu_finding(
                ("cluster-a", "node-a", "recent"), recent, recent.observed_at
            )

            deleted = store.cleanup_hot_state(
                now=NOW, finding_history_retention=timedelta(days=30), limit=100
            )

            assert deleted["gpu_finding_history"] == 1
            assert [
                item.finding_id
                for item in store.list_gpu_findings(
                    "cluster-a", "node-a", active_only=False
                )
            ] == ["finding-recent"]
            assert {
                item.finding_id
                for item in store.list_gpu_findings(
                    "cluster-a", "node-a", active_only=True
                )
            } == {"finding-old", "finding-recent"}
    finally:
        stores[1].close()


def test_gpu_inventory_snapshot_persists_latest_per_node(tmp_path) -> None:
    path = tmp_path / "gpu-inventory.db"
    first = SqliteStore(str(path))
    older = GpuInventorySnapshot(
        snapshot_id="inventory-old",
        cluster_id="cluster-a",
        node_id="node-a",
        observed_at=NOW,
        source=GpuMetricSource.NVIDIA_SMI,
        source_boot_id="boot-a",
        devices=[
            GpuInventoryDevice(
                gpu_index=0, gpu_uuid="GPU-a", pci_bdf="0000:b9:00.0", product="H100"
            )
        ],
    )
    newer = copy_model(
        older, snapshot_id="inventory-new", observed_at=NOW + timedelta(minutes=1)
    )
    first.save_gpu_inventory_snapshot(older)
    first.save_gpu_inventory_snapshot(newer)
    first.save_gpu_inventory_snapshot(older)
    first.close()

    reopened = SqliteStore(str(path))
    loaded = reopened.get_gpu_inventory_snapshot("cluster-a", "node-a")
    reopened.close()

    assert loaded == newer


def batch(
    batch_id: str, samples: list[GpuMetricSample], *, observed_at: datetime = NOW
) -> GpuMetricBatch:
    return gpu_metric_batch(
        batch_id,
        observed_at,
        GpuMetricSource.DCGM_EXPORTER,
        samples,
        cluster_id="hp-cluster",
        node_id="worker-1",
        product="H100",
        driver_branch=575,
        cuda_version="12.9",
        runtime_profile_version="simulated-v1",
        workload_state="ACTIVE",
        affected_workload_ids=["training-job-1"],
        evidence_ref="prometheus://worker-1:9400/metrics",
    )


def test_gpu_batches_reject_blank_node_identity() -> None:
    identity = {"cluster_id": "cluster-a", "node_id": "node-a"}
    for field in ("cluster_id", "node_id"):
        with pytest.raises(ValidationError):
            GpuMetricBatch(
                batch_id="blank-identity",
                **{**identity, field: ""},
                observed_at=NOW,
                source=GpuMetricSource.DCGM_EXPORTER,
                samples=[sample("gpu_temperature_c", 40)],
            )
        with pytest.raises(ValidationError):
            GpuInventorySnapshot(
                snapshot_id="blank-identity",
                **{**identity, field: ""},
                observed_at=NOW,
                source=GpuMetricSource.NVIDIA_SMI,
                source_boot_id="boot-a",
                devices=[
                    GpuInventoryDevice(
                        gpu_index=0, gpu_uuid="GPU-a", pci_bdf="0000:b9:00.0"
                    )
                ],
            )


def test_metrics_service_writes_one_store_batch() -> None:
    class RecordingStore(InMemoryStore):
        def __init__(self) -> None:
            super().__init__()
            self.batch_sizes = []

        def observe_gpu_metrics(self, items):
            self.batch_sizes.append(len(items))
            return super().observe_gpu_metrics(items)

    store = RecordingStore()
    service = GpuMetricsService(store=store)
    samples = [sample(f"custom_metric_{index}", index) for index in range(264)]

    result = service.ingest(batch("batch-write", samples))

    assert result.accepted_samples == 264
    assert store.batch_sizes == [264]


def test_metrics_service_allows_different_nodes_to_write_in_parallel() -> None:
    entered = Barrier(2, timeout=5)

    class BlockingStore(InMemoryStore):
        def observe_gpu_metrics(self, items):
            entered.wait()
            return super().observe_gpu_metrics(items)

    store = BlockingStore()
    service = GpuMetricsService(store=store)
    first = batch("node-a", [sample("gpu_temperature_c", 40)])
    second = copy_model(first, batch_id="node-b", node_id="worker-2")
    while hash((first.cluster_id, first.node_id)) % len(service._node_locks) == hash(
        (second.cluster_id, second.node_id)
    ) % len(service._node_locks):
        second = copy_model(second, node_id=second.node_id + "x")

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(service.ingest, (first, second)))

    assert [item.accepted_samples for item in results] == [1, 1]


def test_metric_thresholds_are_configurable(monkeypatch) -> None:
    monkeypatch.setenv("GPU_FAULT_PCIE_REPLAY_RATE_WARNING_PER_MINUTE", "12.5")
    monkeypatch.setenv("GPU_FAULT_NVLINK_ERROR_DELTA_CRITICAL", "2")
    monkeypatch.setenv("GPU_FAULT_GPU_TEMP_WARNING_C", "82")
    monkeypatch.setenv("GPU_FAULT_GPU_TEMP_CRITICAL_C", "88")
    monkeypatch.setenv("GPU_FAULT_GPU_TEMP_WARNING_MARGIN_C", "7")
    monkeypatch.setenv("GPU_FAULT_THERMAL_VIOLATION_DRAIN_CONSECUTIVE_SAMPLES", "3")
    monkeypatch.setenv("GPU_FAULT_DCGM_CORRELATION_WINDOW_SECONDS", "60")
    monkeypatch.setenv("GPU_FAULT_DCGM_COMPOSITE_CONSECUTIVE_SAMPLES", "3")
    monkeypatch.setenv("GPU_FAULT_DCGM_POWER_LIMIT_RATIO", "0.9")
    monkeypatch.setenv("GPU_FAULT_DCGM_POWER_CORRELATION_MIN_UTILIZATION_PERCENT", "75")
    monkeypatch.setenv("GPU_FAULT_DCGM_ECC_SBE_DELTA_WARNING", "2")
    monkeypatch.setenv("GPU_FAULT_DCGM_RETIRED_PAGES_SBE_DELTA_WARNING", "2")
    monkeypatch.setenv("GPU_FAULT_DCGM_RETIRED_PAGES_DBE_DELTA_CRITICAL", "1")
    monkeypatch.setenv("GPU_FAULT_DCGM_ROW_REMAP_CORRECTABLE_DELTA_WARNING", "3")
    monkeypatch.setenv(
        "GPU_FAULT_DCGM_CORRECTABLE_MEMORY_DRAIN_CONSECUTIVE_SAMPLES", "4"
    )

    thresholds = GpuMetricsThresholds.from_environment()

    assert thresholds.pcie_replay_rate_warning_per_minute == 12.5
    assert thresholds.nvlink_error_delta_critical == 2
    assert thresholds.gpu_temperature_warning_c == 82
    assert thresholds.gpu_temperature_critical_c == 88
    assert thresholds.gpu_temperature_warning_margin_c == 7
    assert thresholds.thermal_violation_drain_consecutive_samples == 3
    assert thresholds.correlation_window_seconds == 60
    assert thresholds.composite_consecutive_samples == 3
    assert thresholds.power_limit_ratio == 0.9
    assert thresholds.power_correlation_min_utilization_percent == 75
    assert thresholds.ecc_sbe_delta_warning == 2
    assert thresholds.retired_pages_sbe_delta_warning == 2
    assert thresholds.retired_pages_dbe_delta_critical == 1
    assert thresholds.row_remap_correctable_delta_warning == 3
    assert thresholds.correctable_memory_drain_consecutive_samples == 4


def test_metrics_service_evaluates_absolute_health_rules() -> None:
    service = GpuMetricsService()

    result = service.ingest(
        batch(
            "batch-1",
            [
                sample("gpu_temperature_c", 91),
                sample("ecc_dbe_volatile_total", 1),
                sample("row_remap_failure", 1),
                sample("row_remap_pending", 1),
            ],
        )
    )

    assert result.accepted_samples == 4
    assert {item.canonical_name for item in result.findings} == {
        "gpu_temperature_c",
        "ecc_dbe_volatile_total",
        "row_remap_failure",
        "row_remap_pending",
    }
    actions = {item.canonical_name: item.automatic_action for item in result.findings}
    assert actions == {
        "gpu_temperature_c": "DRAIN",
        "ecc_dbe_volatile_total": "DRAIN",
        "row_remap_failure": "DRAIN",
        "row_remap_pending": "RESET_GPU",
    }
    assert len(service.latest("hp-cluster", "worker-1")) == 4
    assert len(service.findings("hp-cluster", "worker-1")) == 5
    assert result.new_composite_findings[0].correlation_rule_id == (
        "GPU_MEMORY_DEGRADATION"
    )
    assert result.suppressed_finding_ids, (
        "expected result.suppressed_finding_ids to be truthy"
    )

    service.ingest(
        batch(
            "batch-recovered",
            [sample("gpu_temperature_c", 70), sample("row_remap_pending", 0)],
            observed_at=NOW + timedelta(seconds=15),
        )
    )
    active_names = {
        item.canonical_name for item in service.findings("hp-cluster", "worker-1")
    }
    assert "gpu_temperature_c" not in active_names
    assert "row_remap_pending" not in active_names
    assert len(service.findings("hp-cluster", "worker-1", active_only=False)) == 5


def test_metrics_service_uses_counter_delta_and_handles_reset() -> None:
    service = GpuMetricsService()

    baseline = service.ingest(batch("batch-1", [sample("pcie_replay_total", 1000)]))
    increased = service.ingest(
        batch(
            "batch-2",
            [sample("pcie_replay_total", 1200)],
            observed_at=NOW + timedelta(seconds=15),
        )
    )
    reset = service.ingest(
        batch(
            "batch-3",
            [sample("pcie_replay_total", 10)],
            observed_at=NOW + timedelta(seconds=30),
        )
    )

    assert not baseline.findings, "expected baseline.findings to be falsy"
    assert increased.findings[0].delta == 200
    assert not reset.findings, "expected reset.findings to be falsy"


def test_pcie_replay_uses_official_per_minute_rate() -> None:
    service = GpuMetricsService()
    service.ingest(batch("pcie-baseline", [sample("pcie_replay_total", 1000)]))
    at_threshold = service.ingest(
        batch(
            "pcie-at-threshold",
            [sample("pcie_replay_total", 1002)],
            observed_at=NOW + timedelta(seconds=15),
        )
    )
    exceeded = service.ingest(
        batch(
            "pcie-exceeded",
            [sample("pcie_replay_total", 1005)],
            observed_at=NOW + timedelta(seconds=30),
        )
    )

    assert not at_threshold.findings, "expected at_threshold.findings to be falsy"
    finding = exceeded.findings[0]
    assert finding.rate_per_minute == 12
    assert finding.policy_source == "NVIDIA_DCGM_HEALTH"
    assert finding.automatic_action == "RUN_DIAGNOSTICS"
    assert finding.official_action == "EXAMINE_GPU_HEALTH"


def test_temperature_uses_device_limits_and_escalates() -> None:
    service = GpuMetricsService()
    limits = [
        sample("gpu_slowdown_temperature_c", 92),
        sample("gpu_shutdown_temperature_c", 95),
        sample("gpu_max_operating_temperature_c", 87),
    ]
    warning = service.ingest(
        batch("temperature-warning", [*limits, sample("gpu_temperature_c", 88)])
    )
    critical = service.ingest(
        batch(
            "temperature-critical",
            [*limits, sample("gpu_temperature_c", 93)],
            observed_at=NOW + timedelta(seconds=15),
        )
    )

    warning_finding = warning.new_findings[0]
    assert warning_finding.severity == "WARNING"
    assert warning_finding.automatic_action == "RUN_DIAGNOSTICS"
    assert warning_finding.threshold_value == 87
    assert warning_finding.threshold_source == "NVIDIA_DEVICE_LIMIT"
    assert warning_finding.policy_source == ("SITE_NVIDIA_DEVICE_LIMIT_DERIVED")
    critical_finding = critical.new_findings[0]
    assert critical_finding.severity == "CRITICAL"
    assert critical_finding.automatic_action == "DRAIN"
    assert critical_finding.threshold_value == 92


def test_memory_temperature_uses_device_limit() -> None:
    service = GpuMetricsService()
    result = service.ingest(
        batch(
            "memory-temperature",
            [
                sample("memory_max_operating_temperature_c", 90),
                sample("memory_temperature_c", 90),
            ],
        )
    )

    finding = result.new_findings[0]
    assert finding.severity == "CRITICAL"
    assert finding.automatic_action == "DRAIN"
    assert finding.threshold_value == 90
    assert finding.threshold_source == "NVIDIA_DEVICE_LIMIT"


def test_thermal_violation_escalates_after_consecutive_growth() -> None:
    service = GpuMetricsService()
    service.ingest(batch("thermal-baseline", [sample("thermal_violation_total_us", 0)]))
    warning = service.ingest(
        batch(
            "thermal-warning",
            [sample("thermal_violation_total_us", 1)],
            observed_at=NOW + timedelta(seconds=15),
        )
    )
    critical = service.ingest(
        batch(
            "thermal-critical",
            [sample("thermal_violation_total_us", 2)],
            observed_at=NOW + timedelta(seconds=30),
        )
    )
    unchanged = service.ingest(
        batch(
            "thermal-unchanged",
            [sample("thermal_violation_total_us", 2)],
            observed_at=NOW + timedelta(seconds=45),
        )
    )

    assert warning.new_findings[0].severity == "WARNING"
    assert warning.new_findings[0].automatic_action == ("RUN_DIAGNOSTICS")
    assert critical.new_findings[0].severity == "CRITICAL"
    assert critical.new_findings[0].automatic_action == "DRAIN"
    assert not unchanged.findings, "expected unchanged.findings to be falsy"


def test_thermal_violation_faster_than_the_clock_is_not_graded() -> None:
    """A counter claiming more violation time than elapsed must not DRAIN a node.

    Live H200 nodes advance the power violation counter slightly faster than
    wall clock while completely idle. The same shape on the thermal counter
    would reach ``DRAIN`` after two samples, evicting a healthy node on the
    strength of a counter that cannot be holding microseconds.
    """

    service = GpuMetricsService()
    # 16.5 s of claimed violation per 15 s interval, in microseconds.
    per_sample = 16_500_000
    service.ingest(batch("outrun-baseline", [sample("thermal_violation_total_us", 0)]))
    results = [
        service.ingest(
            batch(
                f"outrun-{index}",
                [sample("thermal_violation_total_us", per_sample * index)],
                observed_at=NOW + timedelta(seconds=15 * index),
            )
        )
        for index in (1, 2, 3)
    ]

    assert [result.new_findings for result in results] == [[], [], []]


def test_thermal_clock_throttling_composite_escalates() -> None:
    service = GpuMetricsService()
    thermal_samples = [
        sample("gpu_slowdown_temperature_c", 92),
        sample("gpu_max_operating_temperature_c", 87),
        sample("gpu_temperature_c", 88),
        sample("clock_throttle_reasons", 0x20),
        sample("sm_clock_mhz", 900),
        sample("memory_clock_mhz", 1200),
    ]

    warning = service.ingest(batch("thermal-throttle-warning", thermal_samples))
    critical = service.ingest(
        batch(
            "thermal-throttle-critical",
            thermal_samples,
            observed_at=NOW + timedelta(seconds=15),
        )
    )

    first = warning.new_composite_findings[0]
    second = critical.new_composite_findings[0]
    assert first.correlation_rule_id == "THERMAL_STRESS"
    assert first.severity == "WARNING"
    assert first.automatic_action == "RUN_DIAGNOSTICS"
    assert "sm_clock_mhz" in first.component_metrics
    assert second.severity == "CRITICAL"
    assert second.automatic_action == "DRAIN"
    assert critical.new_findings == [second]
    assert {"gpu_temperature_c", "clock_throttle_reasons"}.issubset(
        set(second.component_metrics)
    ), (
        'expected { "gpu_temperature_c", "clock_throttle_reasons", }.issubset(set(second.component_metrics)) to be truthy'
    )


def test_low_clock_without_thermal_reason_is_not_a_fault() -> None:
    service = GpuMetricsService()

    result = service.ingest(
        batch(
            "low-clock-idle",
            [
                sample("gpu_temperature_c", 55),
                sample("gpu_utilization_percent", 0),
                sample("sm_clock_mhz", 300),
                sample("memory_clock_mhz", 400),
                sample("clock_throttle_reasons", 0x1),
            ],
        )
    )

    assert not result.findings, "expected result.findings to be falsy"
    assert not result.composite_findings, (
        "expected result.composite_findings to be falsy"
    )


def test_power_limit_metrics_correlate_without_thermal_fault() -> None:
    service = GpuMetricsService()
    service.ingest(batch("power-baseline", [sample("power_violation_total_us", 0)]))

    result = service.ingest(
        batch(
            "power-correlated",
            [
                sample("power_violation_total_us", 1),
                sample("power_usage_w", 680),
                sample("power_limit_w", 700),
                sample("gpu_utilization_percent", 95),
                sample("gpu_temperature_c", 70),
            ],
            observed_at=NOW + timedelta(seconds=15),
        )
    )

    finding = result.new_composite_findings[0]
    assert finding.correlation_rule_id == ("POWER_LIMIT_THROTTLING")
    assert finding.severity == "WARNING"
    assert finding.automatic_action == "RUN_DIAGNOSTICS"
    assert finding.confidence == "MEDIUM"


def test_idle_power_violation_counter_does_not_stay_active() -> None:
    service = GpuMetricsService()
    service.ingest(
        batch(
            "idle-power-baseline",
            [
                sample("power_violation_total_us", 100),
                sample("power_usage_w", 70),
                sample("power_limit_w", 700),
                sample("gpu_utilization_percent", 0),
            ],
        )
    )

    result = service.ingest(
        batch(
            "idle-power-next",
            [
                sample("power_violation_total_us", 15_000_100),
                sample("power_usage_w", 70),
                sample("power_limit_w", 700),
                sample("gpu_utilization_percent", 0),
            ],
            observed_at=NOW + timedelta(seconds=15),
        )
    )

    assert not result.findings, "expected result.findings to be falsy"
    assert not service.findings("cluster-a", "worker-1", active_only=True), (
        'expected service.findings("cluster-a", "worker-1", active_only=True) to be falsy'
    )


def test_pcie_replay_and_xid_create_link_composite() -> None:
    service = GpuMetricsService()
    service.ingest(
        batch(
            "pcie-xid-baseline",
            [sample("pcie_replay_total", 1000), sample("xid_last_error", 0)],
        )
    )

    result = service.ingest(
        batch(
            "pcie-xid-failure",
            [sample("pcie_replay_total", 1005), sample("xid_last_error", 79)],
            observed_at=NOW + timedelta(seconds=15),
        )
    )

    finding = result.new_composite_findings[0]
    assert finding.correlation_rule_id == ("PCIE_XID_LINK_FAILURE")
    assert finding.automatic_action == "DRAIN"
    assert set(finding.component_metrics) == {"pcie_replay_total", "xid_last_error"}
    assert result.xid_events[0].xid == 79


def test_multi_gpu_nvlink_errors_create_one_fabric_composite() -> None:
    service = GpuMetricsService()

    def gpu_metric(name: str, value: float, gpu_index: int) -> GpuMetricSample:
        return copy_model(
            sample(name, value),
            gpu_index=str(gpu_index),
            gpu_uuid=f"GPU-{gpu_index}",
            pci_bdf=f"0000:{gpu_index + 1:02x}:00",
        )

    baseline = [
        gpu_metric("nvlink_crc_aggregate_error_total", 0, 0),
        gpu_metric("nvlink_replay_aggregate_error_total", 0, 0),
        gpu_metric("nvlink_crc_aggregate_error_total", 0, 1),
        gpu_metric("nvlink_replay_aggregate_error_total", 0, 1),
    ]
    service.ingest(batch("nvlink-multi-baseline", baseline))
    result = service.ingest(
        batch(
            "nvlink-multi-failure",
            [copy_model(item, value=1) for item in baseline],
            observed_at=NOW + timedelta(seconds=15),
        )
    )

    assert len(result.new_composite_findings) == 1
    finding = result.new_composite_findings[0]
    assert finding.correlation_rule_id == ("MULTI_GPU_NVLINK_FABRIC_FAILURE")
    assert finding.affected_gpu_uuids == ["GPU-0", "GPU-1"]
    assert finding.automatic_action == "DRAIN"
    assert result.new_findings == [finding]


def test_composite_state_survives_service_restart(tmp_path) -> None:
    path = tmp_path / "composite-state.db"
    samples = [
        sample("gpu_slowdown_temperature_c", 92),
        sample("gpu_temperature_c", 88),
        sample("clock_throttle_reasons", 0x40),
        sample("sm_clock_mhz", 800),
    ]
    first_store = SqliteStore(str(path))
    first = GpuMetricsService(store=first_store).ingest(
        batch("thermal-before-restart", samples)
    )
    first_store.close()

    second_store = SqliteStore(str(path))
    try:
        second = GpuMetricsService(store=second_store).ingest(
            batch(
                "thermal-after-restart",
                samples,
                observed_at=NOW + timedelta(seconds=15),
            )
        )
        assert first.new_composite_findings[0].severity == ("WARNING")
        assert second.new_composite_findings[0].severity == ("CRITICAL")
    finally:
        second_store.close()


def test_correctable_memory_trend_escalates_after_three_samples() -> None:
    service = GpuMetricsService()
    names = [
        "ecc_sbe_volatile_total",
        "retired_pages_sbe_total",
        "row_remap_correctable_total",
    ]
    service.ingest(
        batch("correctable-memory-baseline", [sample(name, 0) for name in names])
    )

    results = [
        service.ingest(
            batch(
                f"correctable-memory-{index}",
                [sample(name, index) for name in names],
                observed_at=NOW + timedelta(seconds=15 * index),
            )
        )
        for index in range(1, 4)
    ]

    first = results[0].new_composite_findings[0]
    second = results[1].composite_findings[0]
    third = results[2].new_composite_findings[0]
    assert first.correlation_rule_id == ("CORRECTABLE_MEMORY_DEGRADATION")
    assert first.severity == "WARNING"
    assert first.automatic_action == "RUN_DIAGNOSTICS"
    assert second.severity == "WARNING"
    assert not results[1].new_findings, "expected results[1].new_findings to be falsy"
    assert third.severity == "CRITICAL"
    assert third.automatic_action == "DRAIN"
    assert results[2].new_findings == [third]
    assert set(third.component_metrics) == set(names)


def test_new_retired_dbe_page_is_critical() -> None:
    service = GpuMetricsService()
    service.ingest(
        batch("retired-dbe-baseline", [sample("retired_pages_dbe_total", 4)])
    )

    result = service.ingest(
        batch(
            "retired-dbe-increased",
            [sample("retired_pages_dbe_total", 5)],
            observed_at=NOW + timedelta(seconds=15),
        )
    )

    finding = result.new_findings[0]
    assert finding.canonical_name == "retired_pages_dbe_total"
    assert finding.severity == "CRITICAL"
    assert finding.automatic_action == "DRAIN"
    assert finding.delta == 1
    assert finding.policy_source == "SITE_DCGM_METRIC"


def test_nvlink_error_uses_official_critical_health_response() -> None:
    service = GpuMetricsService()
    service.ingest(
        batch("nvlink-baseline", [sample("nvlink_crc_aggregate_error_total", 0)])
    )
    result = service.ingest(
        batch(
            "nvlink-error",
            [sample("nvlink_crc_aggregate_error_total", 1)],
            observed_at=NOW + timedelta(seconds=15),
        )
    )

    finding = result.findings[0]
    assert finding.severity == "CRITICAL"
    assert finding.automatic_action == "DRAIN"
    assert finding.policy_source == "NVIDIA_DCGM_HEALTH"
    assert finding.official_action == ("TERMINATE_JOB_AND_ANALYZE_GPU_HEALTH")


def test_xid_metric_emits_only_on_nonzero_state_change() -> None:
    service = GpuMetricsService()

    first = service.ingest(batch("batch-1", [sample("xid_last_error", 94)]))
    duplicate = service.ingest(
        batch(
            "batch-2",
            [sample("xid_last_error", 94)],
            observed_at=NOW + timedelta(seconds=15),
        )
    )
    cleared = service.ingest(
        batch(
            "batch-3",
            [sample("xid_last_error", 0)],
            observed_at=NOW + timedelta(seconds=30),
        )
    )
    repeated = service.ingest(
        batch(
            "batch-4",
            [sample("xid_last_error", 94)],
            observed_at=NOW + timedelta(seconds=45),
        )
    )

    assert not first.xid_events, "expected first.xid_events to be falsy"
    assert not duplicate.xid_events, "expected duplicate.xid_events to be falsy"
    assert not cleared.xid_events, "expected cleared.xid_events to be falsy"
    assert repeated.xid_events[0].xid == 94


def test_xid_metric_baseline_is_shared_across_service_replicas() -> None:
    store = build_store()
    first_replica = GpuMetricsService(store=store)
    second_replica = GpuMetricsService(store=store)

    baseline = first_replica.ingest(batch("batch-1", [sample("xid_last_error", 0)]))
    changed = second_replica.ingest(
        batch(
            "batch-2",
            [sample("xid_last_error", 94)],
            observed_at=NOW + timedelta(seconds=15),
        )
    )
    observed_again = first_replica.ingest(
        batch(
            "batch-3",
            [sample("xid_last_error", 94)],
            observed_at=NOW + timedelta(seconds=30),
        )
    )

    assert not baseline.xid_events, "expected baseline.xid_events to be falsy"
    assert changed.xid_events[0].xid == 94
    assert not observed_again.xid_events, (
        "expected observed_again.xid_events to be falsy"
    )


def test_xid_metric_baseline_survives_service_restart(tmp_path) -> None:
    path = tmp_path / "xid-baseline.db"
    first_store = SqliteStore(str(path))
    first_service = GpuMetricsService(store=first_store)
    first_service.ingest(batch("batch-1", [sample("xid_last_error", 0)]))
    first_store.close()

    second_store = SqliteStore(str(path))
    try:
        changed = GpuMetricsService(store=second_store).ingest(
            batch(
                "batch-2",
                [sample("xid_last_error", 94)],
                observed_at=NOW + timedelta(seconds=15),
            )
        )
        assert changed.xid_events[0].xid == 94
    finally:
        second_store.close()


def test_gpu_latest_and_counter_survive_service_restart(tmp_path) -> None:
    path = tmp_path / "gpu-metrics.db"
    first_store = SqliteStore(str(path))
    GpuMetricsService(store=first_store).ingest(
        batch("counter-baseline", [sample("pcie_replay_total", 1000)])
    )
    first_store.close()

    second_store = SqliteStore(str(path))
    try:
        result = GpuMetricsService(store=second_store).ingest(
            batch(
                "counter-increase",
                [sample("pcie_replay_total", 1200)],
                observed_at=NOW + timedelta(seconds=15),
            )
        )
        assert result.findings[0].delta == 200
        assert result.new_findings[0].canonical_name == ("pcie_replay_total")
        assert (
            GpuMetricsService(store=second_store)
            .latest("hp-cluster", "worker-1")[0]
            .sample.value
            == 1200
        )
    finally:
        second_store.close()


def test_stale_xid_metric_cannot_roll_back_shared_baseline() -> None:
    store = build_store()
    service = GpuMetricsService(store=store)

    service.ingest(batch("batch-1", [sample("xid_last_error", 0)]))
    changed = service.ingest(
        batch(
            "batch-2",
            [sample("xid_last_error", 94)],
            observed_at=NOW + timedelta(seconds=30),
        )
    )
    stale = GpuMetricsService(store=store).ingest(
        batch(
            "batch-stale",
            [sample("xid_last_error", 0)],
            observed_at=NOW + timedelta(seconds=15),
        )
    )
    unchanged = GpuMetricsService(store=store).ingest(
        batch(
            "batch-3",
            [sample("xid_last_error", 94)],
            observed_at=NOW + timedelta(seconds=45),
        )
    )

    assert changed.xid_events[0].xid == 94
    assert not stale.xid_events, "expected stale.xid_events to be falsy"
    assert not unchanged.xid_events, "expected unchanged.xid_events to be falsy"


def test_metrics_batch_retry_and_stale_sample_are_idempotent() -> None:
    service = GpuMetricsService()
    current_batch = batch(
        "batch-current",
        [sample("pcie_replay_total", 1200)],
        observed_at=NOW + timedelta(seconds=30),
    )

    first = service.ingest(current_batch)
    duplicate = service.ingest(current_batch)
    stale = service.ingest(
        batch("batch-stale", [sample("pcie_replay_total", 1000)], observed_at=NOW)
    )

    assert first.accepted_samples == 1
    assert duplicate.duplicate, "expected duplicate.duplicate to be truthy"
    assert stale.accepted_samples == 0
    assert service.latest("hp-cluster", "worker-1")[0].sample.value == (1200)


def test_gpu_metrics_endpoint_bridges_xid94_to_policy() -> None:
    context = build_context()

    async def scenario() -> None:
        async with asgi_client(context) as client:
            baseline = await client.post(
                "/v1/collector-events/gpu-metrics",
                json=batch(
                    "dcgm-xid-baseline", [sample("xid_last_error", 0)]
                ).model_dump(mode="json"),
            )
            response = await client.post(
                "/v1/collector-events/gpu-metrics",
                json=batch(
                    "dcgm-xid94",
                    [sample("gpu_temperature_c", 91), sample("xid_last_error", 94)],
                    observed_at=NOW + timedelta(seconds=15),
                ).model_dump(mode="json"),
            )
            latest = await client.get("/v1/gpu-metrics/hp-cluster/worker-1/latest")
            findings = await client.get("/v1/gpu-health-findings/hp-cluster/worker-1")

        assert baseline.status_code == 200
        assert not baseline.json()["xid_events"]
        assert response.status_code == 200
        body = response.json()
        assert body["accepted_samples"] == 2
        assert body["findings"][0]["severity"] == "CRITICAL"
        assert body["decisions"][0]["official_action"] == "RESTART_APP"
        assert body["decisions"][0]["containment"] == "APPLICATION"
        assert latest.status_code == 200
        assert len(latest.json()) == 2
        assert findings.status_code == 200
        assert findings.json()[0]["canonical_name"] == ("gpu_temperature_c")

    asyncio.run(scenario())


def test_non_xid_gpu_finding_creates_site_safety_workflow() -> None:
    """The DRAIN chain now ends in ESCALATE_SUPPORT: the node stays held, so
    the operator hand-off is an explicit step (logic item 8)."""
    context = build_context()

    async def scenario() -> None:
        async with asgi_client(context) as client:
            response = await client.post(
                "/v1/collector-events/gpu-metrics",
                json=batch(
                    "gpu-temperature-critical", [sample("gpu_temperature_c", 91)]
                ).model_dump(mode="json"),
            )
            statuses = await client.get(
                "/v1/collector-status/hp-cluster", params={"node_id": "worker-1"}
            )

        assert response.status_code == 200
        finding = response.json()["new_findings"][0]
        incident = context.store.get_incident_by_event(f"gpu-{finding['finding_id']}")
        assert incident is not None
        workflow = context.store.get_workflow(incident.workflow_request_id)
        assert [item.operation.value for item in workflow.official_steps] == [
            "FREEZE_EVIDENCE",
            "MARK_UNSCHEDULABLE",
            "STOP_WORKLOADS",
            "QUARANTINE",
            "COLLECT_DIAGNOSTIC_BUNDLE",
            "VALIDATE_GPU",
            "ESCALATE_SUPPORT",
        ]
        assert statuses.status_code == 200
        assert statuses.json()[0]["collector"] == "GPU_METRICS"
        assert statuses.json()[0]["last_success_at"] is not None

    asyncio.run(scenario())


def test_composite_finding_suppresses_duplicate_component_workflows() -> None:
    context = build_context()

    async def scenario() -> None:
        async with asgi_client(context) as client:
            response = await client.post(
                "/v1/collector-events/gpu-metrics",
                json=batch(
                    "memory-composite",
                    [
                        sample("ecc_dbe_volatile_total", 1),
                        sample("row_remap_failure", 1),
                    ],
                ).model_dump(mode="json"),
            )

        assert response.status_code == 200
        body = response.json()
        assert len(body["findings"]) == 2
        assert len(body["new_findings"]) == 1
        composite = body["new_findings"][0]
        assert composite["finding_kind"] == "COMPOSITE"
        assert composite["correlation_rule_id"] == ("GPU_MEMORY_DEGRADATION")
        incident = context.store.get_incident_by_event(f"gpu-{composite['finding_id']}")
        assert incident is not None
        assert "correlation_rule=GPU_MEMORY_DEGRADATION" in (incident.reasons[0])
        for component_id in composite["component_finding_ids"]:
            assert context.store.get_incident_by_event(f"gpu-{component_id}") is None

    asyncio.run(scenario())


def test_row_remap_failure_preserves_nvidia_policy_provenance() -> None:
    """The chain now ends in ESCALATE_SUPPORT: an RMA-class DRAIN keeps the
    node held, so the operator hand-off must be explicit (logic item 8)."""
    context = build_context()

    async def scenario() -> None:
        async with asgi_client(context) as client:
            response = await client.post(
                "/v1/collector-events/gpu-metrics",
                json=batch(
                    "row-remap-failure", [sample("row_remap_failure", 1)]
                ).model_dump(mode="json"),
            )

        assert response.status_code == 200
        finding = response.json()["new_findings"][0]
        incident = context.store.get_incident_by_event(f"gpu-{finding['finding_id']}")
        workflow = context.store.get_workflow(incident.workflow_request_id)
        assert incident.policy_source == ("NVIDIA_GPU_MEMORY_ERROR_MANAGEMENT")
        assert incident.official_action == ("RUN_FIELD_DIAGNOSTIC_FOR_RMA")
        assert incident.effective_action.value == "DRAIN"
        assert incident.policy_reference.endswith(
            "rma-policy-thresholds-for-row-remapping.html"
        ), (
            'expected incident.policy_reference.endswith( "rma-policy-thresholds-for-row-remapping.html" ) to be truthy'
        )
        assert [step.operation.value for step in workflow.official_steps] == [
            "FREEZE_EVIDENCE",
            "MARK_UNSCHEDULABLE",
            "STOP_WORKLOADS",
            "QUARANTINE",
            "COLLECT_DIAGNOSTIC_BUNDLE",
            "RUN_FIELD_DIAGNOSTIC",
            "VALIDATE_GPU",
            "ESCALATE_SUPPORT",
        ]

    asyncio.run(scenario())

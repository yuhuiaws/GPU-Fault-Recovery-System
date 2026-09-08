from __future__ import annotations

import logging

from ._support import (
    NOW,
    NVIDIA_SMI_CORE_FIELDS,
    NVIDIA_SMI_REMAP_FIELDS,
    BufferingSink,
    CollectorError,
    DcgmMetricsCollector,
    GpuMetricSample,
    HostTelemetryCollector,
    NvidiaSmiMetricsCollector,
    RecordingSink,
    completed_nvidia_smi,
    context,
    discover_gpu_product,
    discover_gpu_software_versions,
    next_stable_phase,
    normalize_gpu_product,
    pytest,
    query_nvidia_temperature_limits,
    rank_liveness_collector,
    rank_liveness_cycle,
    subprocess,
    timedelta,
    write_fake_rank,
)


def test_dcgm_summary_uses_stable_channel_phase() -> None:
    sink = RecordingSink()
    collector = DcgmMetricsCollector(
        sink, context(), node_id="worker-1", now=lambda: NOW, health_summary_seconds=300
    )
    text = 'DCGM_FI_DEV_GPU_TEMP{gpu="0",UUID="GPU-a"} 70\n'

    collector.collect_text(text)

    assert collector._next_health_summary_at == next_stable_phase(
        NOW,
        cluster_id=collector.context.cluster_id,
        node_id="worker-1",
        channel="gpu-metrics",
        interval_seconds=300,
    )


def test_dcgm_reports_missing_required_fields() -> None:
    collector = DcgmMetricsCollector(
        RecordingSink(), context(), node_id="worker-1", now=lambda: NOW
    )
    batch = collector.collect_text('DCGM_FI_DEV_GPU_TEMP{gpu="0",UUID="GPU-a"} 70\n')
    assert batch.collection_errors
    assert "pcie_replay_total" in batch.collection_errors[0]
    assert "nvlink_recovery" in batch.collection_errors[0]


def test_force_snapshot_bypasses_dcgm_startup_phase(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    request_path = tmp_path / "gpu.request"
    request_path.write_text("trigger\n", encoding="ascii")
    collector = DcgmMetricsCollector(
        RecordingSink(),
        context(),
        node_id="worker-1",
        force_snapshot_path=str(request_path),
    )
    calls = []

    def collect_once():
        calls.append("collect")
        raise KeyboardInterrupt

    monkeypatch.setattr(collector, "collect_once", collect_once)
    monkeypatch.setattr(
        "gpu_fault.collectors.gpu.dcgm.time.sleep",
        lambda _seconds: pytest.fail(
            "forced health snapshot must not wait for startup phase"
        ),
    )

    with pytest.raises(KeyboardInterrupt):
        collector.run()
    assert calls == ["collect"]


def test_nvidia_smi_discovers_one_product_from_all_gpus() -> None:
    calls = []

    def runner(command, **kwargs):
        calls.append((command, kwargs))
        return completed_nvidia_smi(
            "\n".join(f"{index}, GPU-{index}, NVIDIA H200" for index in range(8))
        )

    product = discover_gpu_product(runner)

    assert product == "H200"
    assert calls[0][0] == [
        "nvidia-smi",
        "--query-gpu=index,uuid,name",
        "--format=csv,noheader",
    ]
    assert calls[0][1]["timeout"] == 15


def test_nvidia_smi_discovers_driver_branch_and_cuda_version() -> None:
    calls = []

    def runner(command, **kwargs):
        calls.append((command, kwargs))
        if "--query-gpu=driver_version" in command:
            return completed_nvidia_smi("\n".join("575.57.08" for _ in range(8)))
        return completed_nvidia_smi(
            "| NVIDIA-SMI 575.57.08 Driver Version: 575.57.08 CUDA Version: 12.9 |"
        )

    driver_branch, cuda_version = discover_gpu_software_versions(runner)

    assert driver_branch == 575
    assert cuda_version == "12.9"
    assert calls[0][0] == [
        "nvidia-smi",
        "--query-gpu=driver_version",
        "--format=csv,noheader",
    ]
    assert calls[1][0] == ["nvidia-smi"]


def test_nvidia_smi_rejects_mixed_driver_branches() -> None:
    def runner(*_args, **_kwargs):
        return completed_nvidia_smi("570.1\n575.2\n")

    with pytest.raises(CollectorError, match="mixed NVIDIA driver"):
        discover_gpu_software_versions(runner)


def test_nvidia_smi_normalizes_b_series_product() -> None:
    assert normalize_gpu_product("NVIDIA B200") == "B200"
    assert normalize_gpu_product("NVIDIA H200 NVL") == "H200"
    assert normalize_gpu_product("NVIDIA L40S") == "L40S"
    assert normalize_gpu_product("Tesla L4") == "L4"
    assert normalize_gpu_product("Tesla T4") == "T4"
    assert normalize_gpu_product("Tesla V100-SXM2-32GB") == "V100"


def test_nvidia_smi_rejects_mixed_gpu_products() -> None:
    def runner(*_args, **_kwargs):
        return completed_nvidia_smi("0, GPU-a, NVIDIA H200\n1, GPU-b, NVIDIA B200\n")

    with pytest.raises(CollectorError, match="mixed GPU products"):
        discover_gpu_product(runner)


def test_dcgm_collector_uses_prometheus_parser() -> None:
    sink = RecordingSink()
    collector = DcgmMetricsCollector(
        sink, context(), node_id="worker-1", now=lambda: NOW
    )
    text = """
# HELP DCGM_FI_DEV_GPU_TEMP GPU temperature
# TYPE DCGM_FI_DEV_GPU_TEMP gauge
DCGM_FI_DEV_GPU_TEMP{gpu="0",UUID="GPU-a",modelName="H100"} 91
# HELP DCGM_FI_DEV_XID_ERRORS Last XID error
# TYPE DCGM_FI_DEV_XID_ERRORS gauge
DCGM_FI_DEV_XID_ERRORS{gpu="0",UUID="GPU-a",modelName="H100"} 94
DCGM_FI_DEV_MEMORY_TEMP{gpu="1",UUID="GPU-b"} 9.223372036854776e+18
# TYPE DCGM_FI_DEV_PCIE_REPLAY_COUNTER counter
DCGM_FI_DEV_PCIE_REPLAY_COUNTER{gpu="0",UUID="GPU-a"} 100
# TYPE DCGM_FI_DEV_CORRECTABLE_REMAPPED_ROWS counter
DCGM_FI_DEV_CORRECTABLE_REMAPPED_ROWS{gpu="0",UUID="GPU-a"} 2
# TYPE DCGM_FI_DEV_NVLINK_ERROR_DL_CRC counter
DCGM_FI_DEV_NVLINK_ERROR_DL_CRC{gpu="0",UUID="GPU-a"} 0
unrelated_metric{gpu="0"} 123
"""

    batch = collector.collect_text(text)

    assert len(batch.samples) == 5
    assert {item.canonical_name for item in batch.samples} == {
        "gpu_temperature_c",
        "xid_last_error",
        "pcie_replay_total",
        "row_remap_correctable_total",
        "nvlink_crc_aggregate_error_total",
    }
    assert batch.samples[0].gpu_uuid == "GPU-a"
    assert sink.requests[0][0] == ("/v1/collector-events/gpu-metrics")


def test_dcgm_edge_filter_suppresses_unchanged_healthy_samples() -> None:
    sink = RecordingSink()
    times = iter([NOW, NOW + timedelta(seconds=15), NOW + timedelta(seconds=300)])
    collector = DcgmMetricsCollector(
        sink,
        context(),
        node_id="worker-1",
        now=lambda: next(times),
        health_summary_seconds=300,
    )
    text = (
        'DCGM_FI_DEV_GPU_TEMP{gpu="0",UUID="GPU-a"} 70\n'
        "DCGM_FI_DEV_ECC_DBE_VOL_TOTAL"
        '{gpu="0",UUID="GPU-a"} 0\n'
    )

    collector.collect_text(text)
    collector.collect_text(text)
    collector.collect_text(text)

    assert len(sink.requests) == 2
    assert sink.requests[1][1]["context_history"] == []


def test_dcgm_edge_filter_ignores_micro_violation_creep() -> None:
    sink = RecordingSink()
    times = iter([NOW + timedelta(seconds=15 * index) for index in range(5)])
    collector = DcgmMetricsCollector(
        sink,
        context(),
        node_id="worker-1",
        now=lambda: next(times),
        health_summary_seconds=300,
    )

    def text(power: int, thermal: int) -> str:
        return (
            'DCGM_FI_DEV_GPU_TEMP{gpu="0",UUID="GPU-a"} 70\n'
            "DCGM_FI_DEV_POWER_VIOLATION"
            f'{{gpu="0",UUID="GPU-a"}} {power}\n'
            "DCGM_FI_DEV_THERMAL_VIOLATION"
            f'{{gpu="0",UUID="GPU-a"}} {thermal}\n'
        )

    # DCGM exports nanoseconds. 200 us per 15 s interval is a
    # 0.0013% duty cycle after collector normalization.
    collector.collect_text(text(1_000_000, 500_000))
    collector.collect_text(text(1_200_000, 700_000))
    collector.collect_text(text(1_400_000, 900_000))
    collector.collect_text(text(1_600_000, 1_100_000))
    collector.collect_text(text(1_800_000, 1_300_000))

    assert len(sink.requests) == 1
    assert sink.requests[0][1]["edge_filter_reasons"] == ["initial-baseline"]


def test_dcgm_edge_filter_confirms_sustained_power_throttling() -> None:
    sink = RecordingSink()
    times = iter([NOW + timedelta(seconds=15 * index) for index in range(6)])
    collector = DcgmMetricsCollector(
        sink,
        context(),
        node_id="worker-1",
        now=lambda: next(times),
        health_summary_seconds=300,
        edge_confirmation_samples=3,
        violation_duty_cycle_threshold=0.05,
    )

    def text(power: int) -> str:
        return (
            'DCGM_FI_DEV_GPU_TEMP{gpu="0",UUID="GPU-a"} 70\n'
            "DCGM_FI_DEV_POWER_VIOLATION"
            f'{{gpu="0",UUID="GPU-a"}} {power}\n'
        )

    # 3 s of throttling per 15 s interval is a 20% duty cycle.
    collector.collect_text(text(0))
    collector.collect_text(text(3_000_000_000))
    collector.collect_text(text(6_000_000_000))
    collector.collect_text(text(9_000_000_000))
    collector.collect_text(text(9_000_000_000))
    collector.collect_text(text(9_000_000_000))

    reasons = [payload["edge_filter_reasons"] for _path, payload in sink.requests]
    assert reasons == [
        ["initial-baseline"],
        ["candidate-confirmed"],
        ["candidate-recovered"],
    ]


def test_dcgm_edge_filter_ignores_a_violation_counter_that_outruns_the_clock() -> None:
    sink = RecordingSink()
    times = iter([NOW + timedelta(seconds=15 * index) for index in range(8)])
    collector = DcgmMetricsCollector(
        sink,
        context(),
        node_id="worker-1",
        now=lambda: next(times),
        health_summary_seconds=300,
        edge_confirmation_samples=3,
        violation_duty_cycle_threshold=0.05,
    )

    def text(violation: int, power: float, limit: float, utilization: int) -> str:
        return (
            'DCGM_FI_DEV_GPU_TEMP{gpu="0",UUID="GPU-a"} 70\n'
            "DCGM_FI_DEV_POWER_VIOLATION"
            f'{{gpu="0",UUID="GPU-a"}} {violation}\n'
            "DCGM_FI_DEV_POWER_USAGE"
            f'{{gpu="0",UUID="GPU-a"}} {power}\n'
            "DCGM_FI_DEV_POWER_MGMT_LIMIT"
            f'{{gpu="0",UUID="GPU-a"}} {limit}\n'
            "DCGM_FI_DEV_GPU_UTIL"
            f'{{gpu="0",UUID="GPU-a"}} {utilization}\n'
        )

    # Live H200 nodes advance DCGM_FI_DEV_POWER_VIOLATION by about 1.1e9 ns per
    # second on a completely idle GPU: 16.5 s of claimed throttling per 15 s
    # interval. Grading that as a 110% duty cycle latched every GPU as a
    # confirmed candidate for the life of the collector.
    idle = 1_080_000_000_000_000
    for index in range(4):
        collector.collect_text(text(idle + 16_500_000_000 * index, 128.0, 700.0, 0))

    assert [payload["edge_filter_reasons"] for _path, payload in sink.requests] == [
        ["initial-baseline"]
    ]

    # The real condition COLLECT-002 injects: power pinned at a lowered limit
    # under full utilization. It must still produce a delivery edge.
    for index in range(4, 7):
        collector.collect_text(text(idle + 16_500_000_000 * index, 210.0, 200.0, 99))

    reasons = [payload["edge_filter_reasons"] for _path, payload in sink.requests]
    assert len(reasons) == 2
    assert reasons[0] == ["initial-baseline"]
    assert "candidate-confirmed" in reasons[1]


DCGM_LOGGER = "gpu_fault.collectors.gpu.dcgm"


def _power_violation_text(nanoseconds: int) -> str:
    return (
        'DCGM_FI_DEV_GPU_TEMP{gpu="0",UUID="GPU-a"} 70\n'
        "DCGM_FI_DEV_POWER_VIOLATION"
        f'{{gpu="0",UUID="GPU-a"}} {nanoseconds}\n'
    )


def _duty_cycle_warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [
        record.getMessage()
        for record in caplog.records
        if record.name == DCGM_LOGGER and "duty cycle" in record.getMessage()
    ]


def test_duty_cycle_uses_the_counters_own_previous_observation(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The exporter's refresh rate must not double the graded duty cycle.

    A 30 s exporter collect interval scraped every 15 s repeats the identical
    counter value on every other tick, so a real 18 s-per-30 s throttle was
    divided by a 15 s wall clock, graded 1.20, and written off as a counter
    that does not hold microseconds. Grading from the counter's own previous
    observation restores the true 0.60.
    """

    sink = RecordingSink()
    collector = DcgmMetricsCollector(
        sink,
        context(),
        node_id="worker-1",
        interval_seconds=15,
        health_summary_seconds=3600,
        edge_confirmation_samples=1,
        violation_duty_cycle_threshold=0.05,
    )

    with caplog.at_level(logging.WARNING, logger=DCGM_LOGGER):
        collector.collect_text(_power_violation_text(0), observed_at=NOW)
        collector.collect_text(
            _power_violation_text(0), observed_at=NOW + timedelta(seconds=15)
        )
        collector.collect_text(
            _power_violation_text(18_000_000_000),
            observed_at=NOW + timedelta(seconds=30),
        )

    reasons = [payload["edge_filter_reasons"] for _path, payload in sink.requests]
    assert reasons == [["initial-baseline"], ["candidate-confirmed"]], (
        "a 60% throttle spanning two scrapes of one exporter sample was not graded"
    )
    assert _duty_cycle_warnings(caplog) == [], (
        "the true 60% duty cycle was rejected as implausible"
    )


def test_implausible_duty_cycle_blacklist_expires(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A counter that is not a duration stays ignored, but not silently forever.

    The warning that a device's violation counter outruns the clock was emitted
    once per process lifetime, so an operator who joined later never saw why the
    counter was ignored. It re-arms after ten intervals -- and not sooner, so a
    permanently broken counter cannot warn every tick.
    """

    sink = RecordingSink()
    collector = DcgmMetricsCollector(
        sink,
        context(),
        node_id="worker-1",
        interval_seconds=15,
        health_summary_seconds=3600,
        edge_confirmation_samples=1,
        violation_duty_cycle_threshold=0.05,
    )

    # 1.1e9 ns of claimed throttling per wall-clock second: implausible at
    # every spacing, so the duty cycle can never be graded.
    def text(seconds: int) -> str:
        return _power_violation_text(1_100_000_000 * seconds)

    with caplog.at_level(logging.WARNING, logger=DCGM_LOGGER):
        collector.collect_text(text(0), observed_at=NOW)
        collector.collect_text(text(15), observed_at=NOW + timedelta(seconds=15))
        collector.collect_text(text(30), observed_at=NOW + timedelta(seconds=30))
        warned_within_the_window = _duty_cycle_warnings(caplog)
        collector.collect_text(text(180), observed_at=NOW + timedelta(seconds=180))

    assert len(warned_within_the_window) == 1, (
        "a broken counter must warn once per expiry window, not every tick"
    )
    assert len(_duty_cycle_warnings(caplog)) == 2, (
        "the implausible-counter warning never re-armed after ten intervals"
    )
    reasons = [payload["edge_filter_reasons"] for _path, payload in sink.requests]
    assert reasons == [["initial-baseline"]], (
        "an implausible duty cycle must not become a candidate after expiry"
    )


def test_dcgm_violation_duty_cycle_threshold_is_validated() -> None:
    with pytest.raises(ValueError, match="duty cycle threshold"):
        DcgmMetricsCollector(
            RecordingSink(),
            context(),
            node_id="worker-1",
            violation_duty_cycle_threshold=0,
        )
    with pytest.raises(ValueError, match="duty cycle threshold"):
        DcgmMetricsCollector(
            RecordingSink(),
            context(),
            node_id="worker-1",
            violation_duty_cycle_threshold=1.5,
        )


def test_dcgm_inventory_delivery_is_independent_of_edge_filter(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return (
                b'DCGM_FI_DEV_GPU_TEMP{gpu="0",UUID="GPU-a",'
                b'pci_bus_id="00000000:B9:00.0"} 70\n'
            )

    monkeypatch.setattr(
        "gpu_fault.collectors.gpu.dcgm.urlopen", lambda *_args, **_kwargs: Response()
    )
    boot_id = tmp_path / "boot_id"
    boot_id.write_text("boot-a\n")
    monkeypatch.setenv("GPU_FAULT_BOOT_ID_PATH", str(boot_id))

    def runner(command, **_kwargs):
        assert "--query-gpu=index,uuid,pci.bus_id,name" in command
        return subprocess.CompletedProcess(
            command, 0, stdout=("0, GPU-a, 00000000:B9:00.0, NVIDIA H100\n"), stderr=""
        )

    times = iter([NOW, NOW + timedelta(seconds=15), NOW + timedelta(seconds=60)])
    sink = RecordingSink()
    collector = DcgmMetricsCollector(
        sink,
        context(),
        node_id="worker-1",
        now=lambda: next(times),
        runner=runner,
        health_summary_seconds=300,
        inventory_interval_seconds=60,
    )
    collector._temperature_limit_samples = []

    collector.collect_once()
    collector.collect_once()
    collector.collect_once()

    inventory = [
        payload
        for path, payload in sink.requests
        if path == "/v1/collector-events/gpu-inventory"
    ]
    metrics = [
        payload
        for path, payload in sink.requests
        if path == "/v1/collector-events/gpu-metrics"
    ]
    assert len(inventory) == 2
    assert len(metrics) == 1
    assert inventory[0]["source_boot_id"] == "boot-a"
    assert inventory[0]["devices"] == [
        {
            "gpu_index": 0,
            "gpu_uuid": "GPU-a",
            "pci_bdf": "00000000:b9:00.0",
            "product": "H100",
        }
    ]


def test_dcgm_edge_filter_emits_counter_delta_and_candidate_recovery():
    sink = RecordingSink()
    collector = DcgmMetricsCollector(
        sink, context(), node_id="worker-1", edge_confirmation_samples=2
    )

    def text(temperature: int, errors: int) -> str:
        return (
            "DCGM_FI_DEV_GPU_TEMP"
            f'{{gpu="0",UUID="GPU-a"}} {temperature}\n'
            "DCGM_FI_DEV_ECC_DBE_VOL_TOTAL"
            f'{{gpu="0",UUID="GPU-a"}} {errors}\n'
        )

    collector.collect_text(text(70, 0), observed_at=NOW)
    collector.collect_text(text(91, 1), observed_at=NOW + timedelta(seconds=15))
    collector.collect_text(text(91, 1), observed_at=NOW + timedelta(seconds=30))
    collector.collect_text(text(91, 1), observed_at=NOW + timedelta(seconds=45))
    collector.collect_text(text(70, 1), observed_at=NOW + timedelta(seconds=60))

    assert len(sink.requests) == 4
    assert sink.requests[1][1]["context_history"] == []
    assert sink.requests[-1][1]["context_history"] == []


def test_dcgm_batch_carries_edge_filter_reasons_to_control_plane():
    """The DCGM reasons must survive the process boundary.

    Live check on 2026-08-08 found every ingested GPU_METRICS batch
    with ``edge_filter_reasons`` absent while HOST_TELEMETRY carried
    ``health-summary`` / ``threshold:...`` / ``recovered``: the DCGM
    collector computed the reasons and then dropped them, because
    ``GpuMetricBatch`` had no such field. Without them the control
    plane cannot tell a 300s health summary apart from a real anomaly
    edge, which is exactly the distinction COLLECT-001/002 rest on.
    """

    sink = RecordingSink()
    collector = DcgmMetricsCollector(
        sink,
        context(),
        node_id="worker-1",
        health_summary_seconds=300,
        edge_confirmation_samples=2,
    )

    def text(temperature: int) -> str:
        return f'DCGM_FI_DEV_GPU_TEMP{{gpu="0",UUID="GPU-a"}} {temperature}\n'

    collector.collect_text(text(70), observed_at=NOW)
    collector.collect_text(text(70), observed_at=NOW + timedelta(seconds=300))
    collector.collect_text(text(91), observed_at=NOW + timedelta(seconds=315))
    collector.collect_text(text(91), observed_at=NOW + timedelta(seconds=330))

    reasons = [request[1]["edge_filter_reasons"] for request in sink.requests]
    assert reasons == [
        ["initial-baseline"],
        ["health-summary"],
        ["candidate-confirmed"],
    ]


def test_dcgm_transient_candidate_is_fully_suppressed() -> None:
    sink = RecordingSink()
    collector = DcgmMetricsCollector(
        sink, context(), node_id="worker-1", edge_confirmation_samples=2
    )

    def text(temperature: int) -> str:
        return f'DCGM_FI_DEV_GPU_TEMP{{gpu="0",UUID="GPU-a"}} {temperature}\n'

    collector.collect_text(text(70), observed_at=NOW)
    collector.collect_text(text(91), observed_at=NOW + timedelta(seconds=15))
    collector.collect_text(text(70), observed_at=NOW + timedelta(seconds=30))

    assert len(sink.requests) == 1
    assert sink.requests[0][1]["edge_filter_reasons"] == ["initial-baseline"]


@pytest.mark.parametrize(("token", "expected"), [("1", True), ("off", False)])
def test_dcgm_edge_filter_switch_reads_every_token(
    monkeypatch: pytest.MonkeyPatch, token: str, expected: bool
) -> None:
    """``GPU_FAULT_DCGM_EDGE_FILTER_ENABLED=1`` used to switch the filter off."""

    monkeypatch.setenv("GPU_FAULT_DCGM_EDGE_FILTER_ENABLED", token)

    collector = DcgmMetricsCollector(RecordingSink(), context(), node_id="worker-1")

    assert collector.edge_filter_enabled is expected


def test_dcgm_edge_state_has_bounded_lru() -> None:
    collector = DcgmMetricsCollector(
        RecordingSink(),
        context(),
        node_id="worker-1",
        edge_filter_enabled=False,
        state_max_keys=2,
    )

    for index in range(4):
        collector.collect_text(
            f'DCGM_FI_DEV_GPU_TEMP{{gpu="{index}",UUID="GPU-{index}"}} 70\n',
            observed_at=NOW + timedelta(seconds=index),
        )

    assert list(collector._previous_values) == [
        "GPU-2/gpu_temperature_c",
        "GPU-3/gpu_temperature_c",
    ]


def test_dcgm_run_recovers_from_unexpected_exception(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    force_snapshot_path = tmp_path / "gpu.request"
    force_snapshot_path.write_text("trigger\n", encoding="ascii")
    collector = DcgmMetricsCollector(
        RecordingSink(),
        context(),
        node_id="worker-1",
        force_snapshot_path=str(force_snapshot_path),
    )
    attempts = []

    def collect_once():
        attempts.append(True)
        raise ValueError("unexpected parser failure")

    monkeypatch.setattr(collector, "collect_once", collect_once)
    monkeypatch.setattr(
        "gpu_fault.collectors.gpu.dcgm.time.sleep",
        lambda _seconds: (_ for _ in ()).throw(StopIteration),
    )

    with pytest.raises(StopIteration):
        collector.run()

    assert attempts == [True]


def test_dcgm_edge_filter_can_be_disabled() -> None:
    sink = RecordingSink()
    collector = DcgmMetricsCollector(
        sink, context(), node_id="worker-1", edge_filter_enabled=False
    )
    text = 'DCGM_FI_DEV_GPU_TEMP{gpu="0",UUID="GPU-a"} 70\n'

    collector.collect_text(text, observed_at=NOW)
    collector.collect_text(text, observed_at=NOW + timedelta(seconds=15))

    assert len(sink.requests) == 2


def test_dcgm_batch_includes_device_temperature_limits() -> None:
    collector = DcgmMetricsCollector(
        RecordingSink(), context(), node_id="worker-1", now=lambda: NOW
    )
    limit = GpuMetricSample(
        metric_name="nvidia_smi_gpu_slowdown_temperature_c",
        canonical_name="gpu_slowdown_temperature_c",
        value=92,
        unit="celsius",
        gpu_index="0",
        gpu_uuid="GPU-a",
    )

    batch = collector.collect_text(
        'DCGM_FI_DEV_GPU_TEMP{gpu="0",UUID="GPU-a"} 80\n', extra_samples=[limit]
    )

    assert {item.canonical_name for item in batch.samples} == {
        "gpu_temperature_c",
        "gpu_slowdown_temperature_c",
    }


def test_dcgm_collection_continues_when_temperature_limits_fail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return b'DCGM_FI_DEV_GPU_TEMP{gpu="0",UUID="GPU-a"} 80\n'

    monkeypatch.setattr(
        "gpu_fault.collectors.gpu.dcgm.urlopen", lambda *_args, **_kwargs: Response()
    )

    def runner(command, **_kwargs):
        return subprocess.CompletedProcess(command, 1, stdout="", stderr="unsupported")

    collector = DcgmMetricsCollector(
        RecordingSink(), context(), node_id="worker-1", now=lambda: NOW, runner=runner
    )

    batch = collector.collect_once()

    assert [item.canonical_name for item in batch.samples] == ["gpu_temperature_c"]


def test_dcgm_collector_rejects_empty_supported_metric_set() -> None:
    collector = DcgmMetricsCollector(
        RecordingSink(), context(), node_id="worker-1", now=lambda: NOW
    )

    with pytest.raises(CollectorError, match="no supported GPU metrics"):
        collector.collect_text("unrelated_metric 1\n")


def test_host_collector_reports_gpu_utilization_by_uuid() -> None:
    def runner(argv, **_kwargs):
        return subprocess.CompletedProcess(
            argv, 0, stdout="GPU-a, 0\nGPU-b, 97\n", stderr=""
        )

    collector = HostTelemetryCollector(
        RecordingSink(), context(), node_id="worker-1", runner=runner
    )

    samples = collector._gpu_utilization(NOW)

    assert [(item.device, item.value) for item in samples] == [
        ("GPU-a", 0),
        ("GPU-b", 97),
    ]
    assert all(item.name == "host_gpu_utilization_percent" for item in samples)


def test_rank_liveness_treats_idle_gpu_cpu_work_as_progress(tmp_path) -> None:
    # Graph compilation is the mirror image: host CPU saturated while
    # the GPUs sit idle, which a stuck collective cannot look like.
    collector = rank_liveness_collector(tmp_path, gpu_utilization=0)
    for pid in (1234, 5678):
        write_fake_rank(tmp_path, pid, cpu_ticks=100, write_bytes=0, wchar=0)
    rank_liveness_cycle(collector, NOW)
    for pid in (1234, 5678):
        write_fake_rank(tmp_path, pid, cpu_ticks=1_600, write_bytes=0, wchar=0)
    values = rank_liveness_cycle(collector, NOW + timedelta(seconds=15))

    assert values["training_rank_advancing_count"] == 2
    assert values["training_rank_seconds_since_progress"] == 0


def test_rank_liveness_is_silent_without_gpu_processes(tmp_path) -> None:
    collector = HostTelemetryCollector(
        RecordingSink(),
        context(),
        node_id="worker-1",
        proc_root=str(tmp_path),
        runner=lambda argv, **_kwargs: (
            subprocess.CompletedProcess(argv, 0, stdout="", stderr="")
        ),
    )

    assert collector._rank_liveness(NOW) == []


def test_host_collector_reports_persistent_gpu_and_efa_card_loss(tmp_path) -> None:
    infiniband_root = tmp_path / "infiniband"
    for index in range(16):
        port = infiniband_root / f"efa_{index}" / "ports" / "1"
        port.mkdir(parents=True)
        device = port.parents[1] / "device"
        device.mkdir()
        (device / "uevent").write_text("DRIVER=efa\n")
        (port / "state").write_text("4: ACTIVE\n" if index < 15 else "1: DOWN\n")
        (port / "phys_state").write_text(
            "5: LinkUp\n" if index < 15 else "3: Disabled\n"
        )

    def runner(argv, **_kwargs):
        assert "--query-gpu=uuid,utilization.gpu" in argv, argv
        return subprocess.CompletedProcess(
            argv, 0, stdout="\n".join(f"GPU-{index}" for index in range(7)), stderr=""
        )

    collector = HostTelemetryCollector(
        RecordingSink(),
        context(),
        node_id="worker-1",
        node_instance_type="ml.p5en.48xlarge",
        expected_gpu_count=8,
        expected_efa_device_count=16,
        inventory_mismatch_consecutive_samples=2,
        infiniband_root=str(infiniband_root),
        runner=runner,
    )

    first = {item.name: item for item in collector._accelerator_inventory(NOW)}
    second = {
        item.name: item
        for item in collector._accelerator_inventory(NOW + timedelta(seconds=15))
    }

    assert first["gpu_inventory_active_count"].value == 7
    assert first["gpu_inventory_missing_count"].value == 1
    assert first["gpu_inventory_mismatch"].value == 0
    assert second["gpu_inventory_mismatch"].value == 1
    assert second["gpu_inventory_mismatch"].labels["expected_count"] == "8"
    assert (
        second["gpu_inventory_mismatch"].labels["node_instance_type"]
        == "ml.p5en.48xlarge"
    )
    assert second["gpu_inventory_mismatch"].labels["expected_gpu_count"] == "8"
    assert second["gpu_inventory_mismatch"].labels["expected_efa_device_count"] == "16"
    assert second["efa_inventory_discovered_count"].value == 16
    assert second["efa_inventory_driver_bound_count"].value == 16
    assert second["efa_inventory_active_count"].value == 15
    assert second["efa_inventory_missing_count"].value == 1
    assert second["efa_inventory_mismatch"].value == 1
    assert second["efa_inventory_mismatch"].labels["failure_mode"] == "LINK_INACTIVE"


def test_gpu_inventory_failure_does_not_hide_efa_inventory(tmp_path) -> None:
    port = tmp_path / "efa_0" / "ports" / "1"
    port.mkdir(parents=True)
    device = port.parents[1] / "device"
    device.mkdir()
    (device / "uevent").write_text("DRIVER=efa\n")
    (port / "state").write_text("4: ACTIVE\n")
    (port / "phys_state").write_text("5: LinkUp\n")

    def runner(argv, **_kwargs):
        return subprocess.CompletedProcess(
            argv, 1, stdout="", stderr="driver unavailable"
        )

    collector = HostTelemetryCollector(
        RecordingSink(),
        context(),
        node_id="worker-1",
        expected_gpu_count=8,
        expected_efa_device_count=1,
        infiniband_root=str(tmp_path),
        runner=runner,
    )

    batch = collector.collect_once()
    by_name = {item.name: item for item in batch.samples}

    assert any(
        "_gpu_inventory: CollectorError: GPU inventory query failed" in error
        for error in batch.collection_errors
    )
    assert by_name["efa_inventory_active_count"].value == 1
    assert by_name["efa_inventory_mismatch"].value == 0


def test_nvidia_smi_csv_fallback_normalizes_core_metrics() -> None:
    fields = [*NvidiaSmiMetricsCollector._IDENTITY_FIELDS, *NVIDIA_SMI_CORE_FIELDS]
    text = "0, GPU-a, NVIDIA H100, 00000000:B9:00.0, 75, 350.5, 98, 72, 40000, 41000\n"

    samples = NvidiaSmiMetricsCollector.parse_csv(text, fields, NVIDIA_SMI_CORE_FIELDS)

    assert len(samples) == 6
    by_name = {item.canonical_name: item for item in samples}
    assert by_name["gpu_temperature_c"].value == 75
    assert by_name["power_usage_w"].value == 350.5
    assert by_name["framebuffer_used_mib"].gpu_uuid == "GPU-a"


def test_nvidia_smi_xml_parses_device_temperature_limits() -> None:
    xml = """<?xml version="1.0"?>
<nvidia_smi_log>
  <gpu>
    <product_name>NVIDIA H200</product_name>
    <minor_number>0</minor_number>
    <uuid>GPU-a</uuid>
    <pci><pci_bus_id>00000000:B9:00.0</pci_bus_id></pci>
    <temperature>
      <gpu_temp_max_threshold>95 C</gpu_temp_max_threshold>
      <gpu_temp_slow_threshold>92 C</gpu_temp_slow_threshold>
      <gpu_temp_max_gpu_threshold>87 C</gpu_temp_max_gpu_threshold>
      <gpu_temp_max_mem_threshold>90 C</gpu_temp_max_mem_threshold>
    </temperature>
  </gpu>
</nvidia_smi_log>
"""

    def runner(command, **_kwargs):
        assert command == ["nvidia-smi", "-q", "-x"]
        return subprocess.CompletedProcess(command, 0, stdout=xml, stderr="")

    samples = query_nvidia_temperature_limits(runner)
    values = {item.canonical_name: item.value for item in samples}

    assert values == {
        "gpu_slowdown_temperature_c": 92,
        "gpu_shutdown_temperature_c": 95,
        "gpu_max_operating_temperature_c": 87,
        "memory_max_operating_temperature_c": 90,
    }
    assert all(item.gpu_uuid == "GPU-a" for item in samples)
    assert all(item.labels["threshold_source"] == "nvidia-smi-xml" for item in samples)


def test_nvidia_smi_csv_parses_row_remap_booleans() -> None:
    fields = [*NvidiaSmiMetricsCollector._IDENTITY_FIELDS, *NVIDIA_SMI_REMAP_FIELDS]
    text = "0, GPU-a, NVIDIA H100, 00000000:B9:00.0, 2, 1, Yes, No\n"

    samples = NvidiaSmiMetricsCollector.parse_csv(text, fields, NVIDIA_SMI_REMAP_FIELDS)

    by_name = {item.canonical_name: item.value for item in samples}
    assert by_name["row_remap_pending"] == 1
    assert by_name["row_remap_failure"] == 0


GPU_INVENTORY_CHANNEL = "/v1/collector-events/gpu-inventory"
GPU_METRICS_CHANNEL = "/v1/collector-events/gpu-metrics"


_NVIDIA_SMI_VALUES = {
    "index": "0",
    "uuid": "GPU-a",
    "name": "NVIDIA H100",
    "pci.bus_id": "00000000:B9:00.0",
    "temperature.gpu": "75",
    "power.draw": "350.5",
    "utilization.gpu": "98",
    "utilization.memory": "72",
    "memory.used": "40000",
    "memory.free": "41000",
    "remapped_rows.pending": "Yes",
    "remapped_rows.failure": "No",
}


def _nvidia_smi_csv(command) -> str:
    """One CSV row answering exactly the fields the command asked for."""

    prefix = "--query-gpu="
    requested = next(
        str(item)[len(prefix) :] for item in command if str(item).startswith(prefix)
    )
    return (
        ", ".join(_NVIDIA_SMI_VALUES.get(field, "0") for field in requested.split(","))
        + "\n"
    )


def _nvidia_smi_runner(command, **_kwargs) -> subprocess.CompletedProcess[str]:
    """Answer the inventory and metric queries; refuse everything else.

    The XML temperature-limit query is allowed to fail: the collector treats it
    as best effort.
    """

    joined = " ".join(str(item) for item in command)
    if "--query-gpu=index,uuid,pci.bus_id,name" in joined:
        return completed_nvidia_smi("0, GPU-a, 00000000:B9:00.0, NVIDIA H100\n")
    if "temperature.gpu" in joined:
        return completed_nvidia_smi(_nvidia_smi_csv(command))
    return completed_nvidia_smi("", returncode=1, stderr="unsupported query")


def _nvidia_smi_environment(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.delenv("GPU_FAULT_EXPECTED_GPU_COUNT", raising=False)
    monkeypatch.delenv("GPU_FAULT_NODE_INSTANCE_TYPE", raising=False)
    boot_id = tmp_path / "boot_id"
    boot_id.write_text("boot-a\n", encoding="ascii")
    monkeypatch.setenv("GPU_FAULT_BOOT_ID_PATH", str(boot_id))


def test_nvidia_smi_round_issues_one_query(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """Four ``--query-gpu`` calls per round cost four 15 s timeouts.

    With a slow driver a round took up to 90 s and ``observed_at``, stamped
    before the first call, predated the samples by that much -- which is the
    spacing the control plane divides its counter rates by.
    """

    _nvidia_smi_environment(monkeypatch, tmp_path)
    queries: list[str] = []

    def runner(command, **kwargs) -> subprocess.CompletedProcess[str]:
        joined = " ".join(str(item) for item in command)
        if "temperature.gpu" in joined:
            queries.append(joined)
        return _nvidia_smi_runner(command, **kwargs)

    times = iter([NOW, NOW + timedelta(seconds=3)])
    collector = NvidiaSmiMetricsCollector(
        RecordingSink(),
        context(),
        node_id="worker-1",
        now=lambda: next(times),
        runner=runner,
    )

    batch = collector.collect_once()

    assert len(queries) == 1, f"one round must issue one --query-gpu: {queries}"
    canonical = {sample.canonical_name for sample in batch.samples}
    assert {
        "gpu_temperature_c",
        "ecc_sbe_volatile_total",
        "retired_pages_pending",
        "row_remap_failure",
    } <= canonical, canonical
    assert batch.observed_at == NOW + timedelta(seconds=3), (
        "observed_at must be stamped after the query returned"
    )


def test_nvidia_smi_round_splits_the_query_only_when_a_field_is_refused(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """One unsupported field costs its group, never the whole round."""

    _nvidia_smi_environment(monkeypatch, tmp_path)
    queries: list[str] = []

    def runner(command, **kwargs) -> subprocess.CompletedProcess[str]:
        joined = " ".join(str(item) for item in command)
        if "--query-gpu=index,uuid,pci.bus_id,name" in joined:
            return _nvidia_smi_runner(command, **kwargs)
        queries.append(joined)
        if "retired_pages.pending" in joined:
            return completed_nvidia_smi(
                "",
                returncode=1,
                stderr='Field "retired_pages.pending" is not a valid field to query.',
            )
        if "--query-gpu=" not in joined:
            return completed_nvidia_smi("", returncode=1, stderr="no xml")
        return completed_nvidia_smi(_nvidia_smi_csv(command))

    times = iter([NOW, NOW + timedelta(seconds=3)])
    collector = NvidiaSmiMetricsCollector(
        RecordingSink(),
        context(),
        node_id="worker-1",
        now=lambda: next(times),
        runner=runner,
    )

    batch = collector.collect_once()

    canonical = {sample.canonical_name for sample in batch.samples}
    assert "gpu_temperature_c" in canonical, (
        "a refused optional field blanked the required core metrics"
    )
    assert "ecc_sbe_volatile_total" in canonical, (
        "a refused optional field blanked an unrelated optional group"
    )
    assert "retired_pages_pending" not in canonical, canonical
    assert len([item for item in queries if "--query-gpu=" in item]) == 5, (
        f"the refused merged query must split into the four groups: {queries}"
    )


def test_nvidia_smi_inventory_the_outbox_took_is_not_re_sent_every_round(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """A buffered inventory snapshot must advance the inventory schedule.

    ``deliver_gpu_inventory`` raised the sink's buffered ``CollectorError``
    straight out of ``collect_once``, so the inventory schedule never moved and
    the run loop reported the round as a collection failure -- posting a second,
    sample-less error batch for a round whose records were all safely buffered.
    """

    _nvidia_smi_environment(monkeypatch, tmp_path)
    sink = BufferingSink()
    # Two stamps per round: the schedule check, then the batch's own
    # ``observed_at`` once the merged query has returned.
    times = iter(
        [
            NOW,
            NOW + timedelta(seconds=1),
            NOW + timedelta(seconds=30),
            NOW + timedelta(seconds=31),
        ]
    )
    collector = NvidiaSmiMetricsCollector(
        sink,
        context(),
        node_id="worker-1",
        now=lambda: next(times),
        runner=_nvidia_smi_runner,
        inventory_interval_seconds=3600,
    )

    collector.collect_once()
    collector.collect_once()

    assert [path for path, _payload in sink.requests] == [
        GPU_INVENTORY_CHANNEL,
        GPU_METRICS_CHANNEL,
        GPU_METRICS_CHANNEL,
    ], "a buffered inventory snapshot was re-sent on the next round"


def test_dcgm_batches_the_outbox_took_advance_both_schedules(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """Buffered DCGM inventory and metrics both count as delivered (ARCH-G3).

    With the live path down, the inventory was re-posted every round and the
    metrics batch re-buffered with a fresh ``batch_id`` every round, because
    both bookkeeping updates sat after a ``post`` that raises once the outbox
    has taken the record.
    """

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return (
                b'DCGM_FI_DEV_GPU_TEMP{gpu="0",UUID="GPU-a",'
                b'pci_bus_id="00000000:B9:00.0"} 70\n'
            )

    monkeypatch.setattr(
        "gpu_fault.collectors.gpu.dcgm.urlopen", lambda *_args, **_kwargs: Response()
    )
    monkeypatch.delenv("GPU_FAULT_EXPECTED_GPU_COUNT", raising=False)
    monkeypatch.delenv("GPU_FAULT_NODE_INSTANCE_TYPE", raising=False)
    boot_id = tmp_path / "boot_id"
    boot_id.write_text("boot-a\n", encoding="ascii")
    monkeypatch.setenv("GPU_FAULT_BOOT_ID_PATH", str(boot_id))
    times = iter([NOW, NOW + timedelta(seconds=15), NOW + timedelta(seconds=60)])
    sink = BufferingSink()
    collector = DcgmMetricsCollector(
        sink,
        context(),
        node_id="worker-1",
        now=lambda: next(times),
        runner=_nvidia_smi_runner,
        health_summary_seconds=300,
        inventory_interval_seconds=60,
    )

    for _ in range(3):
        collector.collect_once()

    paths = [path for path, _payload in sink.requests]
    assert paths.count(GPU_INVENTORY_CHANNEL) == 2, (
        "a buffered inventory snapshot was re-sent every round"
    )
    assert paths.count(GPU_METRICS_CHANNEL) == 1, (
        "an unchanged healthy batch was re-buffered after the outbox took it"
    )

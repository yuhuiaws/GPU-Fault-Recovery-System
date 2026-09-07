"""GPU collectors survive a hung driver and report a vanished device (ARCH-G5/G6).

A hung ``nvidia-smi`` is the symptom of a driver hang. The runner raises
``TimeoutExpired`` for it, which the nvidia-smi collector's run loop did not
catch, so the process exited into a systemd restart loop and the node's GPU
telemetry went silent at exactly the moment it mattered. A GPU that vanished
between two DCGM scrapes used to surface only as ``candidate-recovered`` --
the device's confirmed candidate stopped appearing -- which reads as good news.
"""

from __future__ import annotations

import subprocess

import pytest

from gpu_fault.channel_registry import GPU_INVENTORY_PATH, GPU_METRICS_PATH
from gpu_fault.collectors.gpu.discovery import deliver_gpu_inventory

from ._support import (
    NOW,
    DcgmMetricsCollector,
    NvidiaSmiMetricsCollector,
    RecordingSink,
    context,
    timedelta,
)


def _stop_after(monkeypatch: pytest.MonkeyPatch, count: int) -> list[float]:
    sleeps: list[float] = []

    def sleep(seconds: float) -> None:
        sleeps.append(seconds)
        if len(sleeps) >= count:
            raise StopIteration

    monkeypatch.setattr("gpu_fault.collectors.gpu.nvidia_smi.time.sleep", sleep)
    return sleeps


def test_nvidia_smi_run_survives_hung_nvidia_smi_and_reports_the_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    sink = RecordingSink()
    force_snapshot_path = tmp_path / "gpu.request"
    force_snapshot_path.write_text("trigger\n", encoding="ascii")
    collector = NvidiaSmiMetricsCollector(
        sink,
        context(),
        node_id="worker-1",
        interval_seconds=30,
        force_snapshot_path=str(force_snapshot_path),
        now=lambda: NOW,
    )
    monkeypatch.setattr(
        collector,
        "collect_once",
        lambda: (_ for _ in ()).throw(
            subprocess.TimeoutExpired(["nvidia-smi"], timeout=15)
        ),
    )
    _stop_after(monkeypatch, 1)

    with pytest.raises(StopIteration):
        collector.run()

    assert len(sink.requests) == 1, "a hung nvidia-smi must still report a batch"
    path, payload = sink.requests[0]
    assert path == GPU_METRICS_PATH, "the error report goes on the metrics channel"
    assert payload["samples"] == [], "an erroring round carries no metric samples"
    assert any("TimeoutExpired" in item for item in payload["collection_errors"]), (
        payload["collection_errors"]
    )
    assert force_snapshot_path.exists(), "a failed round must not consume the request"


def test_nvidia_smi_run_backs_off_after_consecutive_failures(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    sink = RecordingSink()
    collector = NvidiaSmiMetricsCollector(
        sink,
        context(),
        node_id="worker-1",
        interval_seconds=30,
        force_snapshot_path=str(tmp_path / "gpu.request"),
        startup_spread_seconds=1,
        now=lambda: NOW,
    )
    monkeypatch.setattr(
        collector,
        "collect_once",
        lambda: (_ for _ in ()).throw(
            subprocess.TimeoutExpired(["nvidia-smi"], timeout=15)
        ),
    )
    sleeps = _stop_after(monkeypatch, 6)

    with pytest.raises(StopIteration):
        collector.run()

    # The first sleep is the startup spread; then two plain intervals, then
    # doubling up to the 4x cap while nvidia-smi keeps hanging.
    assert sleeps[1:] == [30, 30, 60, 120, 120], sleeps


def test_nvidia_smi_run_resets_backoff_after_a_success(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    sink = RecordingSink()
    collector = NvidiaSmiMetricsCollector(
        sink,
        context(),
        node_id="worker-1",
        interval_seconds=30,
        force_snapshot_path=str(tmp_path / "gpu.request"),
        startup_spread_seconds=1,
        now=lambda: NOW,
    )
    outcomes = iter([False, False, False, True, False])

    def collect_once():
        if not next(outcomes):
            raise RuntimeError("driver wedged")
        return None

    monkeypatch.setattr(collector, "collect_once", collect_once)
    sleeps = _stop_after(monkeypatch, 6)

    with pytest.raises(StopIteration):
        collector.run()

    assert sleeps[1:] == [30, 30, 60, 30, 30], sleeps


def test_nvidia_smi_error_report_delivery_failure_does_not_exit(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    class FailingSink:
        def post(self, *_args, **_kwargs):
            raise OSError("control plane unreachable")

    collector = NvidiaSmiMetricsCollector(
        FailingSink(),
        context(),
        node_id="worker-1",
        interval_seconds=30,
        force_snapshot_path=str(tmp_path / "gpu.request"),
        startup_spread_seconds=1,
        now=lambda: NOW,
    )
    monkeypatch.setattr(
        collector,
        "collect_once",
        lambda: (_ for _ in ()).throw(RuntimeError("driver wedged")),
    )
    sleeps = _stop_after(monkeypatch, 3)

    with pytest.raises(StopIteration):
        collector.run()

    assert len(sleeps) == 3, "the loop must keep running when the report fails"


def _dcgm_text(*devices: tuple[str, int, int]) -> str:
    return "".join(
        "DCGM_FI_DEV_GPU_TEMP"
        f'{{gpu="{index}",UUID="{uuid}"}} {temperature}\n'
        "DCGM_FI_DEV_ECC_DBE_VOL_TOTAL"
        f'{{gpu="{index}",UUID="{uuid}"}} {errors}\n'
        for index, (uuid, temperature, errors) in enumerate(devices)
    )


def test_dcgm_vanished_device_is_reported_as_device_lost() -> None:
    sink = RecordingSink()
    collector = DcgmMetricsCollector(
        sink, context(), node_id="worker-1", edge_confirmation_samples=2
    )

    collector.collect_text(
        _dcgm_text(("GPU-a", 70, 0), ("GPU-b", 70, 0)), observed_at=NOW
    )
    collector.collect_text(
        _dcgm_text(("GPU-a", 70, 0)), observed_at=NOW + timedelta(seconds=15)
    )
    collector.collect_text(
        _dcgm_text(("GPU-a", 70, 0)), observed_at=NOW + timedelta(seconds=30)
    )

    reasons = [payload["edge_filter_reasons"] for _, payload in sink.requests]
    assert reasons == [["initial-baseline"], ["device-lost"]], reasons


def test_dcgm_vanished_device_with_confirmed_candidate_is_not_recovered() -> None:
    sink = RecordingSink()
    collector = DcgmMetricsCollector(
        sink, context(), node_id="worker-1", edge_confirmation_samples=2
    )

    collector.collect_text(
        _dcgm_text(("GPU-a", 70, 0), ("GPU-b", 70, 0)), observed_at=NOW
    )
    collector.collect_text(
        _dcgm_text(("GPU-a", 70, 0), ("GPU-b", 91, 0)),
        observed_at=NOW + timedelta(seconds=15),
    )
    collector.collect_text(
        _dcgm_text(("GPU-a", 70, 0), ("GPU-b", 91, 0)),
        observed_at=NOW + timedelta(seconds=30),
    )
    collector.collect_text(
        _dcgm_text(("GPU-a", 70, 0)), observed_at=NOW + timedelta(seconds=45)
    )

    last_reasons = sink.requests[-1][1]["edge_filter_reasons"]
    assert "device-lost" in last_reasons, last_reasons
    assert "candidate-recovered" not in last_reasons, (
        "a device that vanished must not read as its candidate recovering"
    )


def _inventory_runner(count: int):
    def runner(argv, **_kwargs):
        if "-q" in argv:
            return subprocess.CompletedProcess(argv, 1, stdout="", stderr="no xml")
        rows = "".join(
            f"{index}, GPU-{index}, 00000000:{index:02x}:00.0, NVIDIA H100\n"
            for index in range(count)
        )
        return subprocess.CompletedProcess(argv, 0, stdout=rows, stderr="")

    return runner


def test_gpu_inventory_expected_count_defaults_from_instance_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GPU_FAULT_EXPECTED_GPU_COUNT", raising=False)
    monkeypatch.setenv("GPU_FAULT_NODE_INSTANCE_TYPE", "ml.p5en.48xlarge")
    sink = RecordingSink()

    snapshot = deliver_gpu_inventory(
        sink,
        context(),
        node_id="worker-1",
        observed_at=NOW,
        runner=_inventory_runner(8),
    )

    assert snapshot.expected_gpu_count == 8, "p5en.48xlarge carries eight GPUs"
    assert sink.requests[0][0] == GPU_INVENTORY_PATH
    assert sink.requests[0][1]["expected_gpu_count"] == 8


def test_gpu_inventory_explicit_expected_count_wins_over_instance_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GPU_FAULT_EXPECTED_GPU_COUNT", "4")
    monkeypatch.setenv("GPU_FAULT_NODE_INSTANCE_TYPE", "ml.p5en.48xlarge")

    snapshot = deliver_gpu_inventory(
        RecordingSink(),
        context(),
        node_id="worker-1",
        observed_at=NOW,
        runner=_inventory_runner(4),
    )

    assert snapshot.expected_gpu_count == 4, "the explicit count is authoritative"


def test_gpu_inventory_unknown_instance_type_leaves_expected_count_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GPU_FAULT_EXPECTED_GPU_COUNT", raising=False)
    monkeypatch.setenv("GPU_FAULT_NODE_INSTANCE_TYPE", "ml.unknown.large")

    snapshot = deliver_gpu_inventory(
        RecordingSink(),
        context(),
        node_id="worker-1",
        observed_at=NOW,
        runner=_inventory_runner(2),
    )

    assert snapshot.expected_gpu_count is None, (
        "an unknown instance type must not invent a count"
    )

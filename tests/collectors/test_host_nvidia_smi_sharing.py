"""The host collector shells out to nvidia-smi once per round (ARCH-G5/G6).

GPU utilization, rank liveness and GPU inventory each ran their own
``nvidia-smi`` with a 30 s timeout, so a hung driver stretched one 15 s
sampling round to 90 s and the collector never said so. The expected GPU
count also had to be configured by hand although the instance type already
names it.
"""

from __future__ import annotations

import logging
import subprocess

import pytest

from ._support import NOW, HostTelemetryCollector, RecordingSink, context, timedelta


def _quiet_non_gpu_contributors(
    monkeypatch: pytest.MonkeyPatch, collector: HostTelemetryCollector
) -> None:
    gpu_contributors = {"_gpu_utilization", "_rank_liveness", "_gpu_inventory"}
    for name in collector.CONTRIBUTORS:
        if name not in gpu_contributors:
            monkeypatch.setattr(collector, name, lambda _observed_at: [])


def _nvidia_smi_calls(calls: list[list[str]]) -> list[list[str]]:
    return [argv for argv in calls if argv and argv[0] == "nvidia-smi"]


def test_host_collector_runs_one_query_gpu_call_per_round(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []

    def runner(argv, **_kwargs):
        calls.append(list(argv))
        if any("compute-apps" in item for item in argv):
            return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")
        return subprocess.CompletedProcess(
            argv, 0, stdout="GPU-a, 12\nGPU-b, 97\n", stderr=""
        )

    collector = HostTelemetryCollector(
        RecordingSink(),
        context(),
        node_id="worker-1",
        expected_gpu_count=2,
        now=lambda: NOW,
        runner=runner,
    )
    _quiet_non_gpu_contributors(monkeypatch, collector)

    batch = collector.collect_once()

    query_gpu = [
        argv
        for argv in _nvidia_smi_calls(calls)
        if any(item.startswith("--query-gpu=") for item in argv)
    ]
    assert len(query_gpu) == 1, f"expected one --query-gpu call per round: {calls}"
    by_name = {(item.name, item.device): item.value for item in batch.samples}
    assert by_name[("host_gpu_utilization_percent", "GPU-b")] == 97
    assert by_name[("gpu_inventory_active_count", None)] == 2
    assert by_name[("gpu_inventory_mismatch", None)] == 0
    assert batch.collection_errors == [], batch.collection_errors


def test_host_collector_opens_breaker_after_consecutive_timeouts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []

    def runner(argv, **kwargs):
        calls.append(list(argv))
        raise subprocess.TimeoutExpired(argv, timeout=kwargs.get("timeout", 0))

    collector = HostTelemetryCollector(
        RecordingSink(),
        context(),
        node_id="worker-1",
        expected_gpu_count=8,
        now=lambda: NOW,
        runner=runner,
        nvidia_smi_breaker_rounds=3,
        nvidia_smi_breaker_cooldown_rounds=2,
    )
    _quiet_non_gpu_contributors(monkeypatch, collector)

    per_round: list[int] = []
    batches = []
    for round_index in range(7):
        before = len(_nvidia_smi_calls(calls))
        batches.append(collector.collect_once())
        per_round.append(len(_nvidia_smi_calls(calls)) - before)

    # Rounds 1-3: the hung query is attempted once per round (never three
    # times), plus the separate compute-apps probe. Rounds 4-5: the breaker
    # is open and nothing shells out. Round 6 probes again and fails, so
    # round 7 is skipped again.
    assert all(count <= 2 for count in per_round[:3]), per_round
    assert all(count >= 1 for count in per_round[:3]), per_round
    assert per_round[3:5] == [0, 0], per_round
    assert per_round[5] >= 1, per_round
    assert per_round[6] == 0, per_round
    assert any(
        "circuit breaker open (3 consecutive timeouts)" in error
        for error in batches[3].collection_errors
    ), batches[3].collection_errors
    assert any("timed out" in error for error in batches[0].collection_errors), batches[
        0
    ].collection_errors


def test_host_collector_bounds_nvidia_smi_timeout_per_call() -> None:
    timeouts: list[float] = []

    def runner(argv, **kwargs):
        timeouts.append(kwargs["timeout"])
        return subprocess.CompletedProcess(argv, 0, stdout="GPU-a, 1\n", stderr="")

    collector = HostTelemetryCollector(
        RecordingSink(),
        context(),
        node_id="worker-1",
        expected_gpu_count=1,
        now=lambda: NOW,
        runner=runner,
    )

    collector._gpu_utilization(NOW)
    collector._gpu_inventory(NOW + timedelta(seconds=1))

    assert timeouts and all(value <= 15 for value in timeouts), timeouts


def test_host_collector_defaults_expected_counts_from_instance_type(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO, logger="gpu_fault.collectors.host.collector"):
        collector = HostTelemetryCollector(
            RecordingSink(),
            context(),
            node_id="worker-1",
            node_instance_type="ml.p5en.48xlarge",
        )

    assert collector.expected_gpu_count == 8, "p5en.48xlarge carries eight GPUs"
    assert collector.expected_efa_device_count == 16, "p5en.48xlarge has 16 EFA"
    assert any("p5en.48xlarge" in record.getMessage() for record in caplog.records), (
        "the derived default must name the instance type in the log"
    )


def test_host_collector_explicit_expected_counts_win_over_instance_type() -> None:
    collector = HostTelemetryCollector(
        RecordingSink(),
        context(),
        node_id="worker-1",
        node_instance_type="ml.p5en.48xlarge",
        expected_gpu_count=4,
        expected_efa_device_count=2,
    )

    assert collector.expected_gpu_count == 4
    assert collector.expected_efa_device_count == 2


def test_host_collector_warns_when_instance_type_is_unknown(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING, logger="gpu_fault.collectors.host.collector"):
        collector = HostTelemetryCollector(
            RecordingSink(),
            context(),
            node_id="worker-1",
            node_instance_type="ml.unknown.large",
        )

    assert collector.expected_gpu_count is None, "an unknown type invents nothing"
    assert any(
        record.levelno == logging.WARNING and "ml.unknown.large" in record.getMessage()
        for record in caplog.records
    ), [record.getMessage() for record in caplog.records]

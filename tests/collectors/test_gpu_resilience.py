"""GPU collectors survive a hung driver and report a vanished device (ARCH-G5/G6).

A hung ``nvidia-smi`` is the symptom of a driver hang. The runner raises
``TimeoutExpired`` for it, which the nvidia-smi collector's run loop did not
catch, so the process exited into a systemd restart loop and the node's GPU
telemetry went silent at exactly the moment it mattered. A GPU that vanished
between two DCGM scrapes used to surface only as ``candidate-recovered`` --
the device's confirmed candidate stopped appearing -- which reads as good news.
"""

from __future__ import annotations

import logging
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

DCGM_LOGGER = "gpu_fault.collectors.gpu.dcgm"


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


class _DcgmExporterResponse:
    """The keep-alive response object ``dcgm.urlopen`` returns."""

    def __init__(self, text: str) -> None:
        self._text = text

    def __enter__(self) -> _DcgmExporterResponse:
        return self

    def __exit__(self, *_args) -> bool:
        return False

    def read(self) -> bytes:
        return self._text.encode("utf-8")


_DCGM_SCRAPE = 'DCGM_FI_DEV_GPU_TEMP{gpu="0",UUID="GPU-a"} 80\n'


def _dcgm_environment(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.delenv("GPU_FAULT_EXPECTED_GPU_COUNT", raising=False)
    monkeypatch.delenv("GPU_FAULT_NODE_INSTANCE_TYPE", raising=False)
    boot_id = tmp_path / "boot_id"
    boot_id.write_text("boot-a\n", encoding="ascii")
    monkeypatch.setenv("GPU_FAULT_BOOT_ID_PATH", str(boot_id))


def _stop_dcgm_after(monkeypatch: pytest.MonkeyPatch, count: int) -> list[float]:
    sleeps: list[float] = []

    def sleep(seconds: float) -> None:
        sleeps.append(seconds)
        if len(sleeps) >= count:
            raise StopIteration

    monkeypatch.setattr("gpu_fault.collectors.gpu.dcgm.time.sleep", sleep)
    return sleeps


def test_dcgm_scrape_is_delivered_when_temperature_limit_query_hangs(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """A wedged driver hangs ``nvidia-smi -q -x``; the exporter still answers.

    ``query_nvidia_temperature_limits`` let ``TimeoutExpired`` escape the
    ``except CollectorError`` guard *after* the scrape had already succeeded, so
    the node's DCGM telemetry went silent for the whole hang.
    """

    _dcgm_environment(monkeypatch, tmp_path)
    monkeypatch.setattr(
        "gpu_fault.collectors.gpu.dcgm.urlopen",
        lambda *_args, **_kwargs: _DcgmExporterResponse(_DCGM_SCRAPE),
    )
    hangs: list[list[str]] = []

    def runner(command, **_kwargs) -> subprocess.CompletedProcess[str]:
        if "-q" in command:
            hangs.append(list(command))
            raise subprocess.TimeoutExpired(list(command), timeout=15)
        return subprocess.CompletedProcess(
            command, 0, stdout="0, GPU-a, 00000000:B9:00.0, NVIDIA H100\n", stderr=""
        )

    sink = RecordingSink()
    times = iter([NOW, NOW + timedelta(seconds=15)])
    collector = DcgmMetricsCollector(
        sink,
        context(),
        node_id="worker-1",
        now=lambda: next(times),
        runner=runner,
        inventory_interval_seconds=3600,
    )

    collector.collect_once()
    collector.collect_once()

    metrics = [payload for path, payload in sink.requests if path == GPU_METRICS_PATH]
    assert len(metrics) == 1, (
        "a hung nvidia-smi -q -x discarded a successful DCGM scrape"
    )
    assert hangs, "the temperature-limit query was never attempted"


def test_dcgm_metrics_survive_inventory_validation_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """``--expected-gpu-count 4`` on an 8-GPU node must not silence metrics.

    ``GpuInventorySnapshot`` raises pydantic's ``ValidationError`` -- a
    ``ValueError``, not a ``CollectorError`` -- so the tick died before the
    scrape, and because the inventory schedule never advanced it died the same
    way on every following tick.
    """

    _dcgm_environment(monkeypatch, tmp_path)
    monkeypatch.setenv("GPU_FAULT_EXPECTED_GPU_COUNT", "4")
    monkeypatch.setattr(
        "gpu_fault.collectors.gpu.dcgm.urlopen",
        lambda *_args, **_kwargs: _DcgmExporterResponse(_DCGM_SCRAPE),
    )
    sink = RecordingSink()
    ticks = iter(NOW + timedelta(seconds=15 * step) for step in range(7))
    attempts: list[list[str]] = []
    inventory_runner = _inventory_runner(8)

    def runner(command, **kwargs) -> subprocess.CompletedProcess[str]:
        if any(item.startswith("--query-gpu=index,uuid") for item in command):
            attempts.append(list(command))
        return inventory_runner(command, **kwargs)

    collector = DcgmMetricsCollector(
        sink,
        context(),
        node_id="worker-1",
        interval_seconds=15,
        health_summary_seconds=3600,
        inventory_interval_seconds=60,
        now=lambda: next(ticks),
        runner=runner,
    )

    for _ in range(3):
        collector.collect_once()

    assert len(attempts) == 3, "the first three ticks each retry the inventory"

    for _ in range(3):
        collector.collect_once()

    assert len(attempts) == 3, (
        "a permanently unusable inventory must back off to the inventory "
        f"interval instead of paying a subprocess every tick: {len(attempts)}"
    )

    collector.collect_once()

    assert len(attempts) == 4, "the inventory retry resumes after the interval"
    paths = [path for path, _payload in sink.requests]
    assert paths.count(GPU_METRICS_PATH) == 1, (
        "an unusable expected GPU count blocked every metrics tick"
    )
    assert paths.count(GPU_INVENTORY_PATH) == 0, (
        "an invalid inventory snapshot must not be delivered"
    )


def test_dcgm_temperature_limit_query_stops_probing_when_no_thresholds_exist(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """A driver that reports no temperature thresholds is not a success.

    A vGPU or MIG device answers ``nvidia-smi -q -x`` with valid XML that
    carries no threshold tags at all. An empty result left the samples unset
    *and* cleared the failure counter, so the probe ran on every single tick
    forever and no backoff could ever engage.
    """

    _dcgm_environment(monkeypatch, tmp_path)
    monkeypatch.setattr(
        "gpu_fault.collectors.gpu.dcgm.urlopen",
        lambda *_args, **_kwargs: _DcgmExporterResponse(_DCGM_SCRAPE),
    )
    probes: list[list[str]] = []

    def runner(command, **_kwargs) -> subprocess.CompletedProcess[str]:
        if "-q" in command:
            probes.append(list(command))
            return subprocess.CompletedProcess(
                command,
                0,
                stdout=(
                    "<?xml version='1.0' ?><nvidia_smi_log><gpu>"
                    "<minor_number>0</minor_number></gpu></nvidia_smi_log>"
                ),
                stderr="",
            )
        return subprocess.CompletedProcess(
            command, 0, stdout="0, GPU-a, 00000000:B9:00.0, NVIDIA H100\n", stderr=""
        )

    ticks = iter(NOW + timedelta(seconds=15 * step) for step in range(7))
    collector = DcgmMetricsCollector(
        RecordingSink(),
        context(),
        node_id="worker-1",
        interval_seconds=15,
        inventory_interval_seconds=60,
        now=lambda: next(ticks),
        runner=runner,
    )

    for _ in range(6):
        collector.collect_once()

    assert len(probes) == 3, (
        "a limits query that reports no thresholds must count as a failure and "
        f"back off, not run on every tick: {len(probes)} probes"
    )

    collector.collect_once()

    assert len(probes) == 4, "the probe resumes once the inventory interval elapses"


def test_dcgm_temperature_limit_query_backs_off_after_repeated_failures(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """A driver that stays wedged must not cost one 15 s probe per tick.

    ``nvidia-smi -q -x`` hangs for its full timeout while the driver is wedged.
    Probing it every 15 s tick keeps a hung subprocess permanently in flight, so
    after three consecutive failures the retry drops to the inventory cadence.
    """

    _dcgm_environment(monkeypatch, tmp_path)
    monkeypatch.setattr(
        "gpu_fault.collectors.gpu.dcgm.urlopen",
        lambda *_args, **_kwargs: _DcgmExporterResponse(_DCGM_SCRAPE),
    )
    probes: list[list[str]] = []

    def runner(command, **_kwargs) -> subprocess.CompletedProcess[str]:
        if "-q" in command:
            probes.append(list(command))
            raise subprocess.TimeoutExpired(list(command), timeout=15)
        return subprocess.CompletedProcess(
            command, 0, stdout="0, GPU-a, 00000000:B9:00.0, NVIDIA H100\n", stderr=""
        )

    ticks = iter(NOW + timedelta(seconds=15 * step) for step in range(7))
    collector = DcgmMetricsCollector(
        RecordingSink(),
        context(),
        node_id="worker-1",
        interval_seconds=15,
        inventory_interval_seconds=60,
        now=lambda: next(ticks),
        runner=runner,
    )

    for _ in range(3):
        collector.collect_once()

    assert len(probes) == 3, "the first three ticks each probe the limits once"

    for _ in range(3):
        collector.collect_once()

    assert len(probes) == 3, (
        "a wedged limits query must back off to the inventory interval, "
        f"not run on every tick: {len(probes)} probes"
    )

    collector.collect_once()

    assert len(probes) == 4, "the probe resumes once the inventory interval elapses"


def test_dcgm_run_reports_scrape_failure_as_error_batch(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """A dead exporter must reach the control plane as an error, not as silence.

    DCGM mode logged the failure and slept the same interval, so the control
    plane learned of it only from the 420 s GPU_METRICS silence threshold and
    could not tell a dead exporter from a dead collector.
    """

    _dcgm_environment(monkeypatch, tmp_path)

    def refuse(*_args, **_kwargs):
        raise OSError("connection refused")

    monkeypatch.setattr("gpu_fault.collectors.gpu.dcgm.urlopen", refuse)
    sink = RecordingSink()
    collector = DcgmMetricsCollector(
        sink,
        context(),
        node_id="worker-1",
        interval_seconds=15,
        startup_spread_seconds=1,
        # The exporter is dead, not still coming up: no startup grace here.
        startup_grace_seconds=0,
        force_snapshot_path=str(tmp_path / "gpu.request"),
        now=lambda: NOW,
        runner=_inventory_runner(8),
    )
    sleeps = _stop_dcgm_after(monkeypatch, 4)

    with pytest.raises(StopIteration):
        collector.run()

    errors = [
        payload
        for path, payload in sink.requests
        if path == GPU_METRICS_PATH and payload["collection_errors"]
    ]
    assert len(errors) == 3, (
        f"every failed DCGM round must report an error batch: {sink.requests}"
    )
    assert errors[0]["samples"] == [], "an erroring round carries no metric samples"
    assert errors[0]["edge_filter_reasons"] == ["collection-error"], errors[0]
    assert any(
        "cannot scrape DCGM exporter" in item for item in errors[0]["collection_errors"]
    ), errors[0]["collection_errors"]
    # The startup spread, then two plain intervals, then the doubling backoff.
    assert sleeps[1:] == [15, 15, 30], sleeps


def test_dcgm_run_keeps_early_scrape_failures_inside_the_startup_grace(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """A freshly booted node starts the collector before the exporter Pod
    listens; those first refusals are the exporter coming up, not a fault.
    Inside the grace they are retried and not reported (live: every reboot
    risked a dcgm-fields incident and a diagnostic workflow); once the grace
    is over the same refusal is an error batch again."""

    _dcgm_environment(monkeypatch, tmp_path)
    clock = {"now": NOW}

    def refuse(*_args, **_kwargs):
        raise OSError("connection refused")

    monkeypatch.setattr("gpu_fault.collectors.gpu.dcgm.urlopen", refuse)
    sink = RecordingSink()
    collector = DcgmMetricsCollector(
        sink,
        context(),
        node_id="worker-1",
        interval_seconds=15,
        startup_spread_seconds=1,
        startup_grace_seconds=120,
        force_snapshot_path=str(tmp_path / "gpu.request"),
        now=lambda: clock["now"],
        runner=_inventory_runner(8),
    )
    ticks = iter([0, 30, 60, 90, 150, 200])

    def advance(_seconds):
        try:
            clock["now"] = NOW + timedelta(seconds=next(ticks))
        except StopIteration:
            raise StopIteration from None

    monkeypatch.setattr("gpu_fault.collectors.gpu.dcgm.time.sleep", advance)

    with pytest.raises(StopIteration):
        collector.run()

    errors = [
        payload
        for path, payload in sink.requests
        if path == GPU_METRICS_PATH and payload["collection_errors"]
    ]
    # Rounds at 0, 30, 60 and 90 s are inside the 120 s grace; 150 and 200 s
    # are the exporter genuinely dead and are reported.
    assert len(errors) == 2, sink.requests
    assert all(
        "cannot scrape DCGM exporter" in item
        for payload in errors
        for item in payload["collection_errors"]
    ), errors


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


def test_an_exporter_slower_than_the_carry_over_window_says_so_at_startup(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The carry-over window is an invariant, not a cure for any exporter.

    An unrefreshed counter keeps its streak for
    ``DUTY_CYCLE_STALE_CARRY_OVER_INTERVALS`` collector intervals -- 8 x 15 s at
    the defaults. An exporter refreshing slower than that brings back the exact
    F2 symptom the carry-over exists to remove (a sustained throttle whose
    streak is dropped between two refreshes), and nothing in the metrics text
    says the exporter is slow. So the one place the ratio is knowable -- a
    configured exporter interval -- has to say it out loud.
    """

    with caplog.at_level(logging.WARNING, logger=DCGM_LOGGER):
        DcgmMetricsCollector(
            RecordingSink(),
            context(),
            node_id="worker-1",
            interval_seconds=15,
            exporter_interval_seconds=180,
        )
        violating = [
            record.getMessage()
            for record in caplog.records
            if record.name == DCGM_LOGGER and "exporter" in record.getMessage()
        ]
        caplog.clear()
        DcgmMetricsCollector(
            RecordingSink(),
            context(),
            node_id="worker-1",
            interval_seconds=15,
            exporter_interval_seconds=120,
        )
        at_the_limit = [
            record.getMessage()
            for record in caplog.records
            if record.name == DCGM_LOGGER and "exporter" in record.getMessage()
        ]

    assert len(violating) == 1, (
        "an exporter refreshing slower than 8 x the collector interval must be "
        f"reported once at startup: {violating}"
    )
    assert "180" in violating[0] and "120" in violating[0], (
        f"the warning must name the configured period and the bound: {violating[0]}"
    )
    assert at_the_limit == [], (
        "exactly 8 x the collector interval still satisfies the invariant and "
        f"must stay quiet: {at_the_limit}"
    )


def test_a_carried_over_streak_is_dropped_before_a_real_candidate() -> None:
    """The streak bound must evict what this tick told us nothing about.

    The confirmation streaks are truncated to ``state_max_keys`` entries.
    Carried-over keys -- duty-cycle counters the exporter has not refreshed --
    were merged in last, so the truncation kept them and dropped the keys the
    batch had just observed breaching: on a node with more breaching keys than
    the bound, a real sustained fault could never accumulate the streak that
    produces its delivery edge. Observed through the edge, not the table: the
    fault has to be announced.
    """

    sink = RecordingSink()
    collector = DcgmMetricsCollector(
        sink,
        context(),
        node_id="worker-1",
        interval_seconds=15,
        health_summary_seconds=3600,
        edge_confirmation_samples=3,
        violation_duty_cycle_threshold=0.05,
        state_max_keys=2,
    )

    def text(violation: int) -> str:
        # The two hot temperatures are candidates without needing a previous
        # value, so the LRU that keeps only the last two remembered values
        # cannot starve them; the violation counter is last in the text, so it
        # is one of the two that survives and can be graded.
        return (
            'DCGM_FI_DEV_GPU_TEMP{gpu="0",UUID="GPU-a"} 95\n'
            'DCGM_FI_DEV_GPU_TEMP{gpu="1",UUID="GPU-b"} 95\n'
            "DCGM_FI_DEV_POWER_VIOLATION"
            f'{{gpu="2",UUID="GPU-c"}} {violation}\n'
        )

    # Tick 1 advances the counter by 3 s in 15 s, so GPU-c breaches too and owns
    # a streak; from tick 2 on the exporter never refreshes it again, so it
    # carries no news and must not hold a slot against the two hot GPUs.
    violations = (0, 3_000_000_000, 3_000_000_000, 3_000_000_000, 3_000_000_000)
    for tick, violation in enumerate(violations):
        collector.collect_text(
            text(violation), observed_at=NOW + timedelta(seconds=15 * tick)
        )

    confirmations = [
        index
        for index, (_, payload) in enumerate(sink.requests)
        if "candidate-confirmed" in payload["edge_filter_reasons"]
    ]

    assert len(confirmations) == 2, (
        "both hot GPUs must reach the confirmation streak and be announced; a "
        "carried-over counter that keeps its slot starves the second one for "
        f"ever: {[payload['edge_filter_reasons'] for _, payload in sink.requests]}"
    )


def test_the_implausible_counter_write_off_is_bounded(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A per-key write-off must not grow without limit.

    The keys are remembered only to rate-limit the warning, and they were never
    evicted: a node whose GPU UUIDs change (a replaced device, a driver reload
    that renumbers) added one entry per broken counter for the life of the
    process, in a collector that is meant to run for months. Eviction is
    observed through the warning it rate-limits -- a key that has been evicted
    warns again -- because that is all the table is for.
    """

    collector = DcgmMetricsCollector(
        RecordingSink(),
        context(),
        node_id="worker-1",
        interval_seconds=15,
        health_summary_seconds=3600,
        edge_confirmation_samples=1,
        violation_duty_cycle_threshold=0.05,
        state_max_keys=2,
    )

    def text(uuids: tuple[str, str], nanoseconds: int) -> str:
        return "".join(
            "DCGM_FI_DEV_POWER_VIOLATION"
            f'{{gpu="{index}",UUID="{uuid}"}} {nanoseconds}\n'
            for index, uuid in enumerate(uuids)
        )

    # 1.1e9 ns of claimed throttling per wall-clock second is implausible at
    # every spacing, so both devices of each pair are written off. The first
    # pair is seen again 60 s later, well inside the 150 s (10 x 15 s) the
    # write-off suppresses a repeat warning for.
    pairs = (("GPU-a", "GPU-b"), ("GPU-c", "GPU-d"), ("GPU-a", "GPU-b"))
    with caplog.at_level(logging.WARNING, logger=DCGM_LOGGER):
        for index, pair in enumerate(pairs):
            collector.collect_text(
                text(pair, 0), observed_at=NOW + timedelta(seconds=30 * index)
            )
            collector.collect_text(
                text(pair, 16_500_000_000),
                observed_at=NOW + timedelta(seconds=30 * index + 15),
            )

    written_off = [
        record.getMessage()
        for record in caplog.records
        if record.name == DCGM_LOGGER and "duty cycle" in record.getMessage()
    ]

    assert len(written_off) == 6, (
        "the write-off table is unbounded: the first pair's entries survived "
        "the second pair and suppressed their own repeat warning, so a node "
        "whose GPU UUIDs change grows one entry per broken counter for the "
        f"life of the process: {written_off}"
    )


def test_a_non_numeric_exporter_interval_is_refused_by_name(monkeypatch) -> None:
    """A bad exporter period must say which variable to fix.

    ``-c`` is milliseconds and this variable is seconds, so "15000" and "15s"
    are both plausible mistakes; a bare ``float()`` answered either with
    ``could not convert string to float: '15s'`` in a systemd restart loop,
    naming neither the variable nor the unit.
    """

    monkeypatch.setenv("GPU_FAULT_DCGM_EXPORTER_INTERVAL_SECONDS", "15s")

    with pytest.raises(ValueError) as refusal:
        DcgmMetricsCollector(RecordingSink(), context(), node_id="worker-1")

    assert "GPU_FAULT_DCGM_EXPORTER_INTERVAL_SECONDS" in str(refusal.value), (
        f"the refusal did not name the variable to fix: {refusal.value}"
    )
    assert "15s" in str(refusal.value), (
        f"the refusal did not quote the value it could not read: {refusal.value}"
    )


def _fallback_mode_runner(gpu_count: int, *, limits_xml: str | None):
    """``nvidia-smi`` for the fallback collector: inventory, metrics and limits.

    ``--query-gpu`` calls answer one CSV row per GPU with a value for every
    requested column, so the same runner serves the inventory query and the
    merged metrics query. ``-q -x`` answers ``limits_xml`` (or a non-zero exit
    when it is ``None``) and is recorded so a test can count the probes.
    """

    probes: list[list[str]] = []

    def runner(argv, **_kwargs) -> subprocess.CompletedProcess[str]:
        if "-q" in argv:
            probes.append(list(argv))
            if limits_xml is None:
                return subprocess.CompletedProcess(argv, 1, stdout="", stderr="no xml")
            return subprocess.CompletedProcess(argv, 0, stdout=limits_xml, stderr="")
        fields = next(
            item for item in argv if item.startswith("--query-gpu=")
        ).removeprefix("--query-gpu=")
        rows = []
        for index in range(gpu_count):
            identity = {
                "index": str(index),
                "uuid": f"GPU-{index}",
                "pci.bus_id": f"00000000:{index:02x}:00.0",
                "name": "NVIDIA H100",
            }
            rows.append(
                ", ".join(identity.get(field, "0") for field in fields.split(","))
            )
        return subprocess.CompletedProcess(
            argv, 0, stdout="\n".join(rows) + "\n", stderr=""
        )

    runner.probes = probes  # type: ignore[attr-defined]
    return runner


def test_nvidia_smi_metrics_survive_inventory_validation_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """Fallback mode: a wrong ``GPU_FAULT_EXPECTED_GPU_COUNT`` must not silence GPU_METRICS.

    DCGM mode guards its inventory delivery (F3); the nvidia-smi collector ran
    ``deliver_gpu_inventory`` outside any guard, so a misconfigured expected
    count made every round raise, ``run()`` only ever posted sample-less error
    batches, and the node never emitted one real GPU_METRICS sample. The
    failure also has to back off to the inventory cadence, as in DCGM mode.
    """

    _dcgm_environment(monkeypatch, tmp_path)
    monkeypatch.setenv("GPU_FAULT_EXPECTED_GPU_COUNT", "4")
    sink = RecordingSink()
    attempts: list[list[str]] = []
    fallback_runner = _fallback_mode_runner(8, limits_xml=None)

    def runner(argv, **kwargs) -> subprocess.CompletedProcess[str]:
        if any(item.startswith("--query-gpu=index,uuid,pci.bus_id") for item in argv):
            attempts.append(list(argv))
        return fallback_runner(argv, **kwargs)

    clock = [NOW]
    collector = NvidiaSmiMetricsCollector(
        sink,
        context(),
        node_id="worker-1",
        interval_seconds=30,
        inventory_interval_seconds=120,
        now=lambda: clock[0],
        runner=runner,
    )

    def rounds(count: int) -> None:
        for _ in range(count):
            collector.collect_once()
            clock[0] += timedelta(seconds=30)

    rounds(3)

    assert len(attempts) == 3, "the first three rounds each retry the inventory"

    rounds(3)

    assert len(attempts) == 3, (
        "a permanently unusable inventory must back off to the inventory "
        f"interval instead of paying a subprocess every round: {len(attempts)}"
    )

    rounds(1)

    assert len(attempts) == 4, "the inventory retry resumes after the interval"
    paths = [path for path, _payload in sink.requests]
    assert paths.count(GPU_METRICS_PATH) == 7, (
        "an unusable expected GPU count blocked the fallback collector's metrics "
        f"rounds: {paths}"
    )
    assert paths.count(GPU_INVENTORY_PATH) == 0, (
        "an invalid inventory snapshot must not be delivered"
    )
    assert all(
        payload["samples"]
        for path, payload in sink.requests
        if path == GPU_METRICS_PATH
    ), "a round whose inventory failed must still carry its real samples"


def test_nvidia_smi_temperature_limit_query_stops_probing_when_no_thresholds_exist(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """Fallback mode on vGPU/MIG: an empty limits answer must back off, as in DCGM mode.

    Valid XML with no threshold tags left the cached samples unset, so the
    nvidia-smi collector re-ran ``nvidia-smi -q -x`` on every round for the
    process's whole lifetime (the DCGM side was fixed as I3 of Task 12).
    """

    _dcgm_environment(monkeypatch, tmp_path)
    runner = _fallback_mode_runner(
        1,
        limits_xml=(
            "<?xml version='1.0' ?><nvidia_smi_log><gpu>"
            "<minor_number>0</minor_number></gpu></nvidia_smi_log>"
        ),
    )
    clock = [NOW]
    collector = NvidiaSmiMetricsCollector(
        RecordingSink(),
        context(),
        node_id="worker-1",
        interval_seconds=30,
        inventory_interval_seconds=120,
        now=lambda: clock[0],
        runner=runner,
    )

    for _ in range(6):
        collector.collect_once()
        clock[0] += timedelta(seconds=30)

    assert len(runner.probes) == 3, (
        "a limits query that reports no thresholds must count as a failure and "
        f"back off, not run on every round: {len(runner.probes)} probes"
    )

    collector.collect_once()

    assert len(runner.probes) == 4, (
        "the probe resumes once the inventory interval elapses"
    )

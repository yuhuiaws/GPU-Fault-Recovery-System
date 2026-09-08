"""The host collector cannot wedge, and a bad progress file is not a hang.

Every case here is a blocking syscall the collector used to wait on forever
(F-H1/F-H2), an interface tree that changes under it (F-H4), a driver query
whose failure hid the inventory invariant (F-H6), or a progress file the
application wrote badly (F-H3) -- the last of which the control plane read as
a rank that stopped answering and escalated as a CRITICAL hang.
"""

from __future__ import annotations

import json
import logging
import os
import socket
import subprocess
import threading
import time
from collections.abc import Iterator
from pathlib import Path

import pytest

from gpu_fault.collectors.host.collector import BoundedProcessRunner

from ._support import (
    NOW,
    HostTelemetryCollector,
    RecordingSink,
    StopTheLoop,
    TrainingProgressCollector,
    context,
    timedelta,
)


class _WedgedChild:
    """A ``Popen`` whose child is in uninterruptible sleep in the driver.

    ``communicate`` never answers, ``kill()`` is delivered but not acted on,
    and ``wait()`` blocks -- which is exactly why ``subprocess.run``'s own
    timeout handling (kill, then a *blocking* wait) could not return.
    """

    def __init__(self, argv: list[str], released: threading.Event) -> None:
        self.args = argv
        # A real ``Popen`` always has a pid, and the reaper logs it.
        self.pid = 424242
        self.returncode = None
        self.stdout = None
        self.stderr = None
        self.stdin = None
        self.kills = 0
        self._released = released

    def communicate(self, timeout: float | None = None) -> tuple[str, str]:
        if self._released.wait(timeout):
            return "", ""
        raise subprocess.TimeoutExpired(self.args, timeout or 0)

    def kill(self) -> None:
        self.kills += 1

    def wait(self, timeout: float | None = None) -> int:
        if self._released.wait(timeout):
            self.returncode = -9
            return -9
        raise subprocess.TimeoutExpired(self.args, timeout or 0)


class _WedgedChildFactory:
    """Hands out wedged children and releases them all on teardown."""

    def __init__(self) -> None:
        self.released = threading.Event()
        self.children: list[_WedgedChild] = []

    def __call__(self, argv, **_kwargs) -> _WedgedChild:
        child = _WedgedChild(list(argv), self.released)
        self.children.append(child)
        return child


@pytest.fixture
def wedged_children() -> Iterator[_WedgedChildFactory]:
    """Fake D-state children, released so no reaper outlives the test."""

    factory = _WedgedChildFactory()
    try:
        yield factory
    finally:
        factory.released.set()


def _blocking_statvfs(released: threading.Event):
    """``os.statvfs`` that never answers for one mount (a hard NFS/Lustre hang)."""

    def statvfs(mount: str) -> os.statvfs_result:
        if mount.endswith("hung") or mount == "/fsx":
            released.wait()
        return os.statvfs_result((4096, 4096, 1_000, 500, 400, 0, 0, 0, 0, 255))

    return statvfs


def _fake_mountinfo(monkeypatch: pytest.MonkeyPatch, text: str) -> None:
    original_read_text = Path.read_text

    def read_text(path, *args, **kwargs):
        if str(path) == "/proc/self/mountinfo":
            return text
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read_text)


def _write_interface(
    root: Path, name: str, *, physical: bool, operstate: str | None
) -> Path:
    interface = root / name
    statistics = interface / "statistics"
    statistics.mkdir(parents=True)
    for counter in ("rx_errors", "tx_errors", "rx_dropped", "tx_dropped"):
        (statistics / counter).write_text("1\n")
    if physical:
        (interface / "device").mkdir()
    if operstate is not None:
        (interface / "operstate").write_text(f"{operstate}\n")
    return interface


def _set_counters(interface: Path, value: int) -> None:
    """Advance every error/drop counter of one interface."""

    for counter in ("rx_errors", "tx_errors", "rx_dropped", "tx_dropped"):
        (interface / "statistics" / counter).write_text(f"{value}\n")


def _remove_interface(interface: Path) -> None:
    for path in sorted(interface.rglob("*"), reverse=True):
        path.unlink() if path.is_file() else path.rmdir()
    interface.rmdir()


def test_a_d_state_nvidia_smi_cannot_outlive_its_own_timeout(
    monkeypatch: pytest.MonkeyPatch, wedged_children: _WedgedChildFactory
) -> None:
    """The breaker never engaged because ``TimeoutExpired`` never arrived.

    ``subprocess.run`` kills the child and then waits for it without a bound,
    so nvidia-smi wedged in the driver held ``collect_once`` forever: no host
    telemetry at all from the node that most needed it (F-H1).
    """

    monkeypatch.setattr(HostTelemetryCollector, "CONTRIBUTORS", ("_gpu_utilization",))
    runner = BoundedProcessRunner(popen=wedged_children, kill_grace_seconds=0.25)
    collector = HostTelemetryCollector(
        RecordingSink(),
        context(),
        node_id="worker-1",
        now=lambda: NOW,
        runner=runner,
        nvidia_smi_timeout_seconds=0.5,
        nvidia_smi_breaker_rounds=3,
        nvidia_smi_breaker_cooldown_rounds=2,
    )

    rounds: list[tuple[float, list[str]]] = []
    for _ in range(4):
        started = time.monotonic()
        batch = collector.collect_once()
        rounds.append((time.monotonic() - started, list(batch.collection_errors)))

    for elapsed, errors in rounds:
        assert elapsed < 2.5, f"a round outlived nvidia-smi's timeout: {elapsed:.1f}s"
        assert errors, "a hung nvidia-smi round reported no collection error"
    assert any("timed out" in error for error in rounds[0][1]), rounds[0][1]
    assert any("circuit breaker open" in error for error in rounds[3][1]), rounds[3][1]
    assert all(child.kills == 1 for child in wedged_children.children), (
        "the timed-out child was not killed"
    )
    assert all(
        thread.daemon
        for thread in threading.enumerate()
        if thread.name.startswith("gpu-fault-process-reaper")
    ), "a reaper thread would keep the collector process alive"


def test_a_timed_out_driver_query_does_not_fabricate_an_inventory_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A timeout says nothing about how many GPUs are on the node.

    The PCI fallback below reports ``observed`` from a *partial* answer; a
    query that never answered has no partial answer, and reporting zero
    active GPUs would mint a CRITICAL REBOOT_NODE finding for a driver that
    was merely slow.
    """

    monkeypatch.setattr(HostTelemetryCollector, "CONTRIBUTORS", ("_gpu_inventory",))

    def runner(argv, **kwargs):
        raise subprocess.TimeoutExpired(argv, kwargs.get("timeout", 0))

    collector = HostTelemetryCollector(
        RecordingSink(),
        context(),
        node_id="worker-1",
        now=lambda: NOW,
        runner=runner,
        expected_gpu_count=8,
        inventory_mismatch_consecutive_samples=1,
    )

    batch = collector.collect_once()

    assert not [item for item in batch.samples if item.name.startswith("gpu_inventory")]
    assert any("timed out" in error for error in batch.collection_errors), (
        batch.collection_errors
    )


def test_a_failed_driver_query_still_reports_the_inventory_mismatch(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """A GPU that falls off the bus makes ``nvidia-smi`` exit non-zero.

    The raise skipped the inventory samples entirely, so
    ``gpu_inventory_mismatch`` never reached the consumer and the
    CRITICAL/REBOOT_NODE rule could not fire (F-H6).
    """

    monkeypatch.setattr(HostTelemetryCollector, "CONTRIBUTORS", ("_gpu_inventory",))
    gpus = tmp_path / "driver" / "nvidia" / "gpus"
    for index in range(8):
        (gpus / f"0000:0{index}:00.0").mkdir(parents=True)
    stdout = "".join(f"GPU-{index}, 10\n" for index in range(7))

    def runner(argv, **_kwargs):
        return subprocess.CompletedProcess(
            argv,
            255,
            stdout=stdout,
            stderr="Unable to determine the device handle for GPU 0000:07:00.0",
        )

    collector = HostTelemetryCollector(
        RecordingSink(),
        context(),
        node_id="worker-1",
        now=lambda: NOW,
        runner=runner,
        proc_root=str(tmp_path),
        expected_gpu_count=8,
        inventory_mismatch_consecutive_samples=2,
    )

    first = {item.name: item for item in collector.collect_once().samples}
    second = {item.name: item for item in collector.collect_once().samples}

    assert first["gpu_inventory_mismatch"].value == 0, (
        "one sample is not persistent yet"
    )
    assert second["gpu_inventory_mismatch"].value == 1, (
        "the mismatch never reached the consumer"
    )
    assert second["gpu_inventory_active_count"].value == 7, "7 of 8 UUIDs were parsed"
    assert second["gpu_inventory_discovered_count"].value == 8, (
        "the PCI/driver enumeration did not see the eight installed GPUs"
    )
    assert second["gpu_inventory_mismatch"].labels["failure_mode"] == (
        "DRIVER_QUERY_FAILED"
    )


def test_a_hung_shared_mount_does_not_wedge_the_tick(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``statvfs`` blocks in D state on a hard NFS mount or a recovering MDS.

    The detector deadlocked on the condition it detects: no batch was posted
    again, so the CRITICAL ``shared_filesystem_unavailable`` rule never fired
    (F-H2). One hung mount must also not make its healthy neighbour look
    unavailable.
    """

    monkeypatch.setattr(
        HostTelemetryCollector, "CONTRIBUTORS", ("_shared_filesystems",)
    )
    _fake_mountinfo(
        monkeypatch,
        "36 25 259:1 / / rw,relatime - xfs /dev/nvme0n1p1 rw\n"
        "40 36 0:42 / /fsx rw,relatime - lustre 10.0.0.1@tcp:/fsx rw\n"
        "41 36 0:43 / /shared rw,relatime - nfs4 10.0.0.2:/shared rw\n",
    )
    released = threading.Event()
    collector = HostTelemetryCollector(
        RecordingSink(),
        context(),
        node_id="worker-1",
        now=lambda: NOW,
        statvfs=_blocking_statvfs(released),
        statvfs_timeout_seconds=0.5,
    )

    try:
        started = time.monotonic()
        batch = collector.collect_once()
        elapsed = time.monotonic() - started
    finally:
        released.set()

    by_mount = {(item.name, item.device): item.value for item in batch.samples}
    assert elapsed < 3, f"the tick waited on the hung mount: {elapsed:.1f}s"
    assert by_mount[("shared_filesystem_unavailable", "/fsx")] == 1, by_mount
    assert ("shared_filesystem_used_percent", "/fsx") not in by_mount, (
        "used-percent was reported for a mount that never answered"
    )
    assert by_mount[("shared_filesystem_unavailable", "/shared")] == 0, (
        "a hung mount poisoned the probe for a healthy one"
    )
    assert ("shared_filesystem_used_percent", "/shared") in by_mount, by_mount


def test_a_hung_configured_filesystem_is_reported_not_skipped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``GPU_FAULT_FILESYSTEMS`` may name an FSx or NFS path too."""

    monkeypatch.setattr(HostTelemetryCollector, "CONTRIBUTORS", ("_filesystems",))
    _fake_mountinfo(
        monkeypatch, "36 25 259:1 / / rw,relatime - xfs /dev/nvme0n1p1 rw\n"
    )
    released = threading.Event()
    collector = HostTelemetryCollector(
        RecordingSink(),
        context(),
        node_id="worker-1",
        now=lambda: NOW,
        filesystems=["/hung", "/"],
        statvfs=_blocking_statvfs(released),
        statvfs_timeout_seconds=0.5,
    )

    try:
        started = time.monotonic()
        batch = collector.collect_once()
        elapsed = time.monotonic() - started
    finally:
        released.set()

    by_mount = {(item.name, item.device): item.value for item in batch.samples}
    assert elapsed < 3, f"the tick waited on the hung mount: {elapsed:.1f}s"
    assert by_mount[("shared_filesystem_unavailable", "/hung")] == 1, by_mount
    assert by_mount[("filesystem_used_percent", "/")] == 60, by_mount


def test_network_survives_a_vanishing_interface_and_ignores_virtual_ones(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """Pod churn deletes veth pairs between ``iterdir`` and ``read_text``.

    One deletion discarded every network sample for the tick, including
    ``network_link_down`` for a required interface, and every veth was
    reported as a device of its own (F-H4).
    """

    monkeypatch.setattr(HostTelemetryCollector, "CONTRIBUTORS", ("_network",))
    _write_interface(tmp_path, "eth0", physical=True, operstate="down")
    _write_interface(tmp_path, "vanishing0", physical=True, operstate=None)
    _write_interface(tmp_path, "veth1234", physical=False, operstate="up")
    _write_interface(tmp_path, "lo", physical=False, operstate="up")
    collector = HostTelemetryCollector(
        RecordingSink(),
        context(),
        node_id="worker-1",
        now=lambda: NOW,
        required_interfaces=["eth0"],
        net_class_root=str(tmp_path),
    )

    batch = collector.collect_once()
    devices = {item.device for item in batch.samples}
    by_name = {(item.name, item.device): item.value for item in batch.samples}

    assert by_name[("network_link_down", "eth0")] == 1, (
        "the required interface's link state was lost with the vanished one"
    )
    assert devices == {"eth0"}, devices
    assert batch.collection_errors == [], batch.collection_errors


def test_network_prunes_counters_of_departed_interfaces(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """``net/<if>/...`` baselines grew for every interface that ever existed.

    A node runs this collector for weeks across thousands of pods, so the
    baseline map grew without bound. Pruning is visible from the outside: an
    interface that comes back is a first sighting again, and a first sighting
    has no delta to report -- whereas a kept baseline would mint a
    ``network_errors_delta`` spanning the whole absence, which is what the
    consumer's error-rate rule reads (F-H4).
    """

    monkeypatch.setattr(HostTelemetryCollector, "CONTRIBUTORS", ("_network",))
    eth0 = _write_interface(tmp_path, "eth0", physical=True, operstate="up")
    eth1 = _write_interface(tmp_path, "eth1", physical=True, operstate="up")
    times = iter([NOW + timedelta(seconds=15 * index) for index in range(4)])
    collector = HostTelemetryCollector(
        RecordingSink(),
        context(),
        node_id="worker-1",
        now=lambda: next(times),
        net_class_root=str(tmp_path),
    )

    collector.collect_once()
    _set_counters(eth0, 5)
    _set_counters(eth1, 5)
    second = collector.collect_once()
    _remove_interface(eth1)
    collector.collect_once()
    returned = _write_interface(tmp_path, "eth1", physical=True, operstate="up")
    _set_counters(returned, 9)
    fourth = collector.collect_once()

    assert ("network_errors_delta", "eth1") in {
        (item.name, item.device) for item in second.samples
    }, "a second sighting of an interface must report its counter delta"
    fourth_names = {(item.name, item.device) for item in fourth.samples}
    assert ("network_errors_delta", "eth1") not in fourth_names, (
        "the departed interface kept its baseline, so its return reported a "
        "delta spanning the absence"
    )
    assert ("network_errors_delta", "eth0") in fourth_names, (
        "the interface that never left lost its baseline"
    )


@pytest.mark.parametrize(
    ("payload", "expected_error"),
    [
        ('{"labels": "phase-1"}', "ValueError"),
        ('{"step": -1}', "ValidationError"),
        ('{"step": 4', "JSONDecodeError"),
        ('{"loss": {"value": 1}}', "ValidationError"),
    ],
)
def test_a_malformed_progress_file_still_reports_the_rank_as_alive(
    tmp_path, payload: str, expected_error: str
) -> None:
    """An application-written file must not be able to mint a hang.

    Every parse and validation error escaped before ``sink.post``, so the
    cycle posted nothing; 120 s later the control plane emitted a CRITICAL
    "training rank heartbeat timed out" finding for a live rank (F-H3).
    """

    progress = tmp_path / "progress.json"
    progress.write_text(payload, encoding="utf-8")
    sink = RecordingSink()
    collector = TrainingProgressCollector(
        sink,
        cluster_id="cluster-a",
        attempt_id="attempt-a",
        rank=3,
        progress_path=str(progress),
        node_id="node-a",
        now=lambda: NOW,
    )

    heartbeat = collector.collect_once()

    assert [path for path, _payload in sink.requests] == ["/v1/training-progress"], (
        "a malformed progress file suppressed the liveness heartbeat"
    )
    assert heartbeat.step is None, "a malformed file must not invent progress"
    assert heartbeat.labels["progress_error"] == expected_error, heartbeat.labels


def test_a_malformed_progress_file_is_warned_once(
    tmp_path, caplog: pytest.LogCaptureFixture
) -> None:
    """The loop used to log a traceback every 15 s for the same bad file."""

    progress = tmp_path / "progress.json"
    progress.write_text('{"step": -1}', encoding="utf-8")
    collector = TrainingProgressCollector(
        RecordingSink(),
        cluster_id="cluster-a",
        attempt_id="attempt-a",
        rank=3,
        progress_path=str(progress),
        now=lambda: NOW,
    )

    logger = "gpu_fault.collectors.training_progress"
    with caplog.at_level(logging.WARNING, logger=logger):
        for _ in range(3):
            collector.collect_once()

    warnings = [record for record in caplog.records if record.name == logger]
    assert len(warnings) == 1, [record.getMessage() for record in warnings]


def test_a_good_progress_file_after_a_bad_one_clears_the_error(tmp_path) -> None:
    progress = tmp_path / "progress.json"
    progress.write_text('{"step": 4', encoding="utf-8")
    collector = TrainingProgressCollector(
        RecordingSink(),
        cluster_id="cluster-a",
        attempt_id="attempt-a",
        rank=3,
        progress_path=str(progress),
        now=lambda: NOW,
    )
    collector.collect_once()

    progress.write_text(json.dumps({"step": 42, "labels": {"phase": "warmup"}}))
    heartbeat = collector.collect_once()

    assert heartbeat.step == 42, "a repaired file must be read again"
    assert heartbeat.labels == {"phase": "warmup"}, heartbeat.labels


def test_run_tells_systemd_it_is_ready_and_pings_the_watchdog_each_tick(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """``Type=notify`` plus ``WatchdogSec`` is what turns a wedge into a restart.

    With ``Type=simple`` and no watchdog, a collector blocked forever in a
    driver or on a hung mount looked like a healthy service to systemd, so
    ``Restart=always`` never fired (F-H1/F-H2).
    """

    monkeypatch.setattr(HostTelemetryCollector, "CONTRIBUTORS", ())
    notify_path = tmp_path / "notify.sock"
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    listener.bind(str(notify_path))
    listener.settimeout(0.5)
    monkeypatch.setenv("NOTIFY_SOCKET", str(notify_path))
    snapshot_request = tmp_path / "host.request"
    snapshot_request.write_text("now\n")
    sleeps: list[float] = []

    def sleep(seconds: float) -> None:
        sleeps.append(seconds)
        if len(sleeps) == 2:
            raise StopTheLoop("two ticks are enough")

    monkeypatch.setattr("gpu_fault.collectors.host.collector.time.sleep", sleep)
    times = iter([NOW + timedelta(seconds=15 * index) for index in range(8)])
    collector = HostTelemetryCollector(
        RecordingSink(),
        context(),
        node_id="worker-1",
        now=lambda: next(times),
        force_snapshot_path=str(snapshot_request),
    )

    try:
        with pytest.raises(StopTheLoop):
            collector.run()
        messages = []
        while True:
            try:
                messages.append(listener.recv(4096).decode())
            except TimeoutError:
                break
    finally:
        listener.close()

    assert messages == ["READY=1", "WATCHDOG=1", "WATCHDOG=1"], messages
    assert sleeps == [collector.interval_seconds] * 2, (
        "the force-snapshot path must not also wait out the startup spread"
    )


def test_the_watchdog_notification_is_a_no_op_without_systemd(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A collector run by hand has no ``NOTIFY_SOCKET``."""

    from gpu_fault.collectors.host.collector import sd_notify

    monkeypatch.delenv("NOTIFY_SOCKET", raising=False)

    assert sd_notify("READY=1") is False, "no socket, no notification"


def test_the_statvfs_worker_a_hung_mount_captured_is_replaced(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One wedged mount costs one leaked daemon thread, not one per tick.

    ``statvfs`` cannot be cancelled, so the worker the hung mount captured is
    abandoned; a shared pool would have been poisoned by it and every later
    mount, on every later tick, would have timed out too.
    """

    from gpu_fault.collectors.host.system_metrics import BoundedStatvfs

    released = threading.Event()
    probe = BoundedStatvfs(_blocking_statvfs(released), timeout_seconds=0.25)

    try:
        for _ in range(3):
            with pytest.raises(TimeoutError):
                probe("/hung")
        answered = probe("/healthy")
    finally:
        released.set()

    assert answered.f_blocks == 1_000, "the replacement worker did not answer"
    workers = [
        thread
        for thread in threading.enumerate()
        if thread.name.startswith("gpu-fault-statvfs")
    ]
    assert all(thread.daemon for thread in workers), (
        "a captured statvfs worker would keep the process alive"
    )


def test_the_tick_does_not_reprobe_a_mount_that_already_timed_out(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``/hung`` appears in both the configured and the shared mount lists."""

    monkeypatch.setattr(
        HostTelemetryCollector, "CONTRIBUTORS", ("_filesystems", "_shared_filesystems")
    )
    _fake_mountinfo(
        monkeypatch,
        "36 25 259:1 / / rw,relatime - xfs /dev/nvme0n1p1 rw\n"
        "40 36 0:42 / /hung rw,relatime - lustre 10.0.0.1@tcp:/hung rw\n",
    )
    released = threading.Event()
    probes: list[str] = []

    def statvfs(mount: str) -> os.statvfs_result:
        probes.append(mount)
        if mount == "/hung":
            released.wait()
        return os.statvfs_result((4096, 4096, 1_000, 500, 400, 0, 0, 0, 0, 255))

    collector = HostTelemetryCollector(
        RecordingSink(),
        context(),
        node_id="worker-1",
        now=lambda: NOW,
        filesystems=["/hung"],
        statvfs=statvfs,
        statvfs_timeout_seconds=0.5,
    )

    try:
        started = time.monotonic()
        batch = collector.collect_once()
        elapsed = time.monotonic() - started
    finally:
        released.set()

    unavailable = [
        item
        for item in batch.samples
        if item.name == "shared_filesystem_unavailable" and item.device == "/hung"
    ]
    assert elapsed < 3, f"the tick paid the deadline twice: {elapsed:.1f}s"
    assert probes.count("/hung") == 1, probes
    assert [item.value for item in unavailable] == [1, 1], unavailable


def test_a_bounded_runner_returns_the_child_output(tmp_path) -> None:
    """The default runner is still ``subprocess.run`` for everything healthy."""

    runner = BoundedProcessRunner()

    completed = runner(
        ["/bin/sh", "-c", "printf out; printf err >&2; exit 3"],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )

    assert completed.returncode == 3, completed
    assert completed.stdout == "out", completed.stdout
    assert completed.stderr == "err", completed.stderr


def test_a_bounded_runner_kills_a_child_that_outlives_its_timeout() -> None:
    runner = BoundedProcessRunner()
    started = time.monotonic()

    with pytest.raises(subprocess.TimeoutExpired):
        runner(
            ["/bin/sh", "-c", "sleep 30"],
            capture_output=True,
            text=True,
            timeout=0.5,
            check=False,
        )

    assert time.monotonic() - started < 3, "the kill path was not bounded"

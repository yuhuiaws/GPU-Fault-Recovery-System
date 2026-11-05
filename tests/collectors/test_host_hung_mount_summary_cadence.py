"""The host tick never touches a hung mount outside its probe worker.

GF-GENERAL-LIVE-005 ``healthy_other_mount`` (live 2026-10-26, HyperPod lane
hpb): one NFS export was SIGSTOPped for nine minutes and the harness saw no
second host batch while it hung, which it read as "the collector's next
scheduled delivery never arrived" -- a tick wedged in a syscall on the hung
mount outside the deadline-bounded probe. This module is the executable audit
of that claim against the real contributors: every ``os``/``io`` path call
under the hung mount is trapped, parks the caller exactly as a hard NFS hang
does, and records which thread made it. Only the ``gpu-fault-mount-probe``
worker may park there; a call from the tick's own thread is the wedge, named
by function, and the tick's wall-clock bound catches it. Across two health
summary intervals the tick keeps delivering: the unavailable reading for the
hung mount, the healthy neighbour, the configured filesystem, and the summary
itself -- one parked worker for the hung mount the whole time, not one per
tick.

Deliberately not on ``isolated_runtime``: that fixture answers every
``statvfs`` with ENOENT, and this audit needs the collector's real syscalls on
real directories, with only the hung one trapped. Real syscalls, not the
host's numbers, though: the directories sit on the host's own filesystem, so
every ``statvfs`` under the test's root reports a fixed capacity once it has
answered (:data:`CAPACITY`) -- the host's disk usage must not choose the edges.
"""

from __future__ import annotations

import builtins
import io
import os
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from gpu_fault.collectors.host import system_metrics
from gpu_fault.collectors.host.collector import HostTelemetryCollector
from tests.collectors import _cov95_runtime_collect as common

ROOT = "28 1 259:1 / / rw,relatime shared:1 - ext4 /dev/root rw"
PROBE_WORKER = "gpu-fault-mount-probe"
UNAVAILABLE = "shared_filesystem_unavailable"
#: The tick interval and the summary cadence the test drives, in seconds.
TICK_SECONDS = 15
SUMMARY_SECONDS = 60
#: A tick that waits on the hung mount is caught here; a bounded one is
#: milliseconds plus one ``statvfs_timeout_seconds`` on the detection tick.
TICK_BOUND_SECONDS = 5.0
#: What every ``statvfs`` under the test's root answers once the real call has
#: returned: ``f_blocks``, ``f_bavail`` (and ``f_bfree``), ``f_files``,
#: ``f_ffree`` (and ``f_favail``) -- 60 % of the bytes and 10 % of the inodes
#: in use, under every capacity rule the edge filter reads (the 90 %
#: ``local_filesystem_used_percent`` warning, the 98 % ``*_used_percent``
#: thresholds). The mounts are directories on the host's root filesystem, so
#: the real numbers are the host's: with its ``/`` 90 % used (the deploy's
#: release gate, 2026-10-27) the baseline carried
#: ``sustained:local_filesystem_used_percent`` next to ``baseline``.
CAPACITY = (1000, 400, 1000, 900)
USED_PERCENT = 60.0


class HungFilesystem:
    """Every path syscall under ``root`` parks its caller -- a hard NFS hang.

    Installed over the ``os`` and ``io`` entry points a collector could reach
    a mount point through (``statvfs``, ``stat``/``lstat``, ``open``,
    ``listdir``/``scandir``, ``access``, ``readlink``, ``io.open``). While
    ``hung`` a trapped call blocks until :meth:`release`, then goes through to
    the real filesystem so a recovered mount reads healthy again. Each call
    records its function and thread; one from any thread but the probe worker
    is a violation -- on the node that call wedges the tick.
    """

    def __init__(self, root: str, monkeypatch: pytest.MonkeyPatch) -> None:
        self.root = root
        self.hung = False
        self.released = threading.Event()
        self.calls: list[tuple[str, str, str]] = []
        self.violations: list[tuple[str, str, str]] = []
        for module, name in (
            (os, "statvfs"),
            (os, "stat"),
            (os, "lstat"),
            (os, "open"),
            (os, "listdir"),
            (os, "scandir"),
            (os, "access"),
            (os, "readlink"),
            (io, "open"),
            (builtins, "open"),
        ):
            monkeypatch.setattr(module, name, self._trap(name, getattr(module, name)))

    def release(self) -> None:
        self.hung = False
        self.released.set()

    def _under_root(self, candidate: Any) -> str | None:
        if candidate is None or isinstance(candidate, int):
            return None
        try:
            path = os.fspath(candidate)
        except TypeError:
            return None
        if isinstance(path, bytes):
            path = path.decode(errors="surrogateescape")
        if path == self.root or path.startswith(self.root + "/"):
            return path
        return None

    def _trap(self, name: str, original: Callable[..., Any]) -> Callable[..., Any]:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            path = self._under_root(args[0] if args else kwargs.get("path"))
            if path is not None and self.hung:
                thread = threading.current_thread().name
                record = (name, path, thread)
                self.calls.append(record)
                if not thread.startswith(PROBE_WORKER):
                    self.violations.append(record)
                self.released.wait()
            return original(*args, **kwargs)

        return wrapper


def _fixed_capacity(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``os.statvfs`` under ``root``: the real syscall, then :data:`CAPACITY`.

    The call still reaches the kernel -- a missing directory is still ENOENT,
    and :class:`HungFilesystem`, installed over this, still parks it -- but the
    blocks and inodes it reports are the test's, not the host's.
    """

    real = os.statvfs
    prefix = str(root) + os.sep

    def statvfs(path: Any, *args: Any, **kwargs: Any) -> os.statvfs_result:
        result = real(path, *args, **kwargs)
        if isinstance(path, int) or not os.fsdecode(path).startswith(prefix):
            return result
        blocks, free, files, files_free = CAPACITY
        return os.statvfs_result(
            (
                result.f_bsize,
                result.f_frsize,
                blocks,
                free,
                free,
                files,
                files_free,
                files_free,
                result.f_flag,
                result.f_namemax,
            )
        )

    monkeypatch.setattr(os, "statvfs", statvfs)


def _nfs_line(mount_id: int, mount_point: str, export: str) -> str:
    return (
        f"{mount_id} 28 0:{mount_id} / {mount_point} rw,relatime shared:{mount_id} "
        f"- nfs4 {export} rw,vers=4.1,hard,proto=tcp"
    )


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def _probe_workers() -> int:
    return sum(
        1 for thread in threading.enumerate() if thread.name.startswith(PROBE_WORKER)
    )


def _tick(collector: HostTelemetryCollector, hung: HungFilesystem, bound: float):
    """One ``collect_once`` under a wall-clock bound, on its own thread.

    A tick that is still running at the bound is the wedge the module exists
    to catch: the test fails naming the calls that parked outside the worker,
    and releases the hang so the thread can finish.
    """

    outcome: dict[str, Any] = {}

    def run() -> None:
        try:
            outcome["batch"] = collector.collect_once()
        except BaseException as exc:  # pragma: no cover - reported below
            outcome["error"] = exc

    thread = threading.Thread(target=run, name="host-tick", daemon=True)
    started = time.monotonic()
    thread.start()
    thread.join(bound)
    if thread.is_alive():
        hung.release()
        thread.join(bound)
        raise AssertionError(
            f"the tick did not finish within {bound:g}s while the mount hung; "
            f"calls on the hung mount outside the probe worker: {hung.violations}"
        )
    if "error" in outcome:
        raise outcome["error"]
    return outcome["batch"], time.monotonic() - started


def _gauges(batch) -> dict[str | None, tuple[float, dict[str, str]]]:
    return {
        sample.device: (sample.value, dict(sample.labels))
        for sample in batch.samples
        if sample.name == UNAVAILABLE
    }


def _values(batch) -> dict[tuple[str, str | None], float]:
    return {(sample.name, sample.device): sample.value for sample in batch.samples}


def _collector(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    sink: common.RecordingSink,
    clock: common.Clock,
    *,
    tables: str,
    configured: Path,
) -> HostTelemetryCollector:
    """The real ``_filesystems`` and ``_shared_filesystems`` over a private ``/proc``.

    Only the mount tables are faked (``/proc/self/mountinfo`` and PID 1's,
    both listing the same mounts so the sandbox check sees them); the mount
    points are real directories and the probes are the collector's own
    ``os.statvfs`` (trapped, and given :data:`CAPACITY`, by the caller) and its
    directory-entry read.
    """

    proc = tmp_path / "proc"
    _write(proc / "1/mountinfo", tables)
    _write(proc / "self/mountinfo", tables)

    def mapped_path(path: str | os.PathLike[str]) -> Path:
        raw = os.fspath(path)
        if raw == "/proc" or raw.startswith("/proc/"):
            return proc / raw.removeprefix("/proc").lstrip("/")
        return Path(path)

    monkeypatch.setattr(system_metrics, "Path", mapped_path)
    monkeypatch.setattr(
        HostTelemetryCollector, "CONTRIBUTORS", ("_filesystems", "_shared_filesystems")
    )
    # A deterministic summary phase: the jitter would otherwise move each
    # summary by a random fraction of the interval.
    monkeypatch.setenv("GPU_FAULT_COLLECTOR_CADENCE_JITTER_RATIO", "0")
    return HostTelemetryCollector(
        sink,
        common.collector_context(workload_state="ACTIVE"),
        node_id="node-a",
        proc_root=str(proc),
        filesystems=[str(configured)],
        now=clock.now,
        statvfs=os.statvfs,
        statvfs_timeout_seconds=0.2,
        shared_filesystem_state_path=str(tmp_path / "memory.json"),
        edge_filter_enabled=True,
        health_summary_seconds=SUMMARY_SECONDS,
        force_snapshot_path=str(tmp_path / "host.request"),
    )


def test_the_summary_keeps_arriving_while_one_mount_hangs_for_two_intervals(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Two summary periods under a hung export: every tick bounded, all delivered.

    Mirrors the live chronology: a healthy baseline, the export stops answering,
    the detection edge goes out once, then the tick runs on for two summary
    intervals with the mount still hung and must deliver each summary with the
    hung mount unavailable (``PROBE_TIMEOUT``), the neighbour at 0 and the
    configured filesystem read -- then the resume is one ``recovered`` edge.
    """

    hung_mount = tmp_path / "hung"
    other_mount = tmp_path / "other"
    configured = tmp_path / "local"
    for mount in (hung_mount, other_mount, configured):
        mount.mkdir()
        (mount / "checkpoint").write_text("")
    tables = "\n".join(
        (
            ROOT,
            _nfs_line(41, str(hung_mount), "10.0.0.2:/hung"),
            _nfs_line(42, str(other_mount), "10.0.0.3:/other"),
        )
    )
    sink = common.RecordingSink()
    clock = common.Clock()
    # Installed first, so the trap wraps it: the collector still calls the
    # trap, and the capacity it reads is the test's, whatever the host's disk
    # holds.
    _fixed_capacity(tmp_path, monkeypatch)
    # Trapped before the collector binds ``os.statvfs``: the audit is of the
    # collector's own syscalls, not of a scripted probe.
    hung = HungFilesystem(str(hung_mount), monkeypatch)
    collector = _collector(
        monkeypatch, tmp_path, sink, clock, tables=tables + "\n", configured=configured
    )
    workers_before = _probe_workers()

    try:
        baseline, _ = _tick(collector, hung, TICK_BOUND_SECONDS)
        assert _gauges(baseline)[str(hung_mount)][0] == 0, _gauges(baseline)
        assert baseline.edge_filter_reasons == ["baseline"], (
            baseline.edge_filter_reasons
        )

        hung.hung = True
        clock.seconds += TICK_SECONDS
        detection, elapsed = _tick(collector, hung, TICK_BOUND_SECONDS)
        assert elapsed < TICK_BOUND_SECONDS, elapsed
        assert _gauges(detection)[str(hung_mount)] == (
            1,
            {
                "failure_mode": "PROBE_TIMEOUT",
                "source": "10.0.0.2:/hung",
                "fstype": "nfs4",
            },
        ), _gauges(detection)
        assert f"threshold:{UNAVAILABLE}" in detection.edge_filter_reasons, (
            detection.edge_filter_reasons
        )
        parked = _probe_workers() - workers_before
        assert parked >= 1, "no worker was parked for the hung mount"

        summaries = []
        delivered_before = len(sink.requests)
        ticks = (2 * SUMMARY_SECONDS) // TICK_SECONDS + 2
        for _ in range(ticks):
            clock.seconds += TICK_SECONDS
            batch, elapsed = _tick(collector, hung, TICK_BOUND_SECONDS)
            assert elapsed < 2.0, (
                f"a tick waited {elapsed:.1f}s on a mount already known to hang"
            )
            gauges = _gauges(batch)
            assert gauges[str(hung_mount)][0] == 1, gauges
            assert gauges[str(hung_mount)][1]["failure_mode"] == "PROBE_TIMEOUT", gauges
            assert gauges[str(other_mount)] == (
                0,
                {"source": "10.0.0.3:/other", "fstype": "nfs4"},
            ), "a hung mount poisoned the probe of its healthy neighbour"
            values = _values(batch)
            assert values[("filesystem_unavailable", str(configured))] == 0, values
            # Read through the collector's statvfs, at the test's capacity.
            assert values[("filesystem_used_percent", str(configured))] == (
                USED_PERCENT
            ), values
            assert _probe_workers() - workers_before <= parked + 1, (
                "the hung mount parked a new worker on a later tick"
            )
            if "health-summary" in batch.edge_filter_reasons:
                summaries.append(batch)
        delivered = sink.requests[delivered_before:]
        assert len(summaries) >= 2, (
            f"{len(summaries)} health summaries were delivered in "
            f"{ticks * TICK_SECONDS}s of hang (expected two {SUMMARY_SECONDS}s periods)"
        )
        assert len(delivered) == len(summaries), (
            "a batch other than the summaries was delivered under an unchanged hang"
        )
        for payload in (request[1] for request in delivered):
            sample = next(
                item
                for item in payload["samples"]
                if item["name"] == UNAVAILABLE and item["device"] == str(hung_mount)
            )
            assert sample["value"] == 1, payload["batch_id"]

        hung.release()
        # The parked worker answers only once the scheduler runs it after the
        # release; until then the collector rightly still reports the earlier
        # probe as running. A slow host can tick first, so tick until it has.
        deadline = time.monotonic() + TICK_BOUND_SECONDS
        while True:
            clock.seconds += TICK_SECONDS
            recovered, _ = _tick(collector, hung, TICK_BOUND_SECONDS)
            if (
                _gauges(recovered)[str(hung_mount)][0] == 0
                or time.monotonic() >= deadline
            ):
                break
            time.sleep(0.05)
        assert _gauges(recovered)[str(hung_mount)][0] == 0, _gauges(recovered)
        assert "recovered" in recovered.edge_filter_reasons, (
            recovered.edge_filter_reasons
        )
    finally:
        hung.release()

    assert hung.violations == [], (
        "the tick touched the hung mount outside the probe worker (function, "
        f"path, thread): {hung.violations}"
    )
    assert hung.calls, "nothing probed the hung mount at all; the audit is vacuous"
    assert {name for name, _, _ in hung.calls} == {"statvfs"}, (
        f"calls parked on the hung mount: {sorted({name for name, _, _ in hung.calls})}"
    )

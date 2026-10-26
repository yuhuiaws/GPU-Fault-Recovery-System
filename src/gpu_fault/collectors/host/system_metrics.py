from __future__ import annotations

import json
import logging
import os
import queue
import shutil
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, NamedTuple, TypeVar

from gpu_fault.collectors.host.mount_table import (
    MountEntry,
    host_mount,
    read_mount_table,
)
from gpu_fault.collectors.host.observation import HostObservationError
from gpu_fault.host_capability_markers import SMART_STATUS_UNSUPPORTED
from gpu_fault.host_device_signals import DEVICE_STATE_GONE, DEVICE_STATE_LABEL
from gpu_fault.host_health import (
    HostMetricSample,
)

LOGGER = logging.getLogger(__name__)

_T = TypeVar("_T")

#: One request to a probe worker: the mount and the syscall to make on it.
#: None retires an abandoned worker once its mount answers, so a mount that
#: recovers does not leave the thread behind.
_ProbeRequest = tuple[str, Callable[[str], Any]] | None
_ProbeResponse = tuple[Any, OSError | None]


class ProbeStillRunning(TimeoutError):
    """A probe of this mount from an earlier tick has not returned yet.

    Carries the syscall that is parked, so the caller can keep reporting the
    condition under the failure mode that first described it: a directory read
    that hung must not be relabelled a ``statvfs`` timeout on the next tick
    merely because ``statvfs`` was the call that found the mount busy.
    """

    def __init__(self, mount: str, operation: Callable[[str], Any]) -> None:
        super().__init__(
            f"{_operation_name(operation)}({mount}) from an earlier tick has not "
            "returned"
        )
        self.operation = operation


def _operation_name(operation: Callable[[str], Any]) -> str:
    return getattr(operation, "__name__", "probe")


#: How long a SMART health verdict is reused. ``smartctl -H`` is one fork per
#: drive with a 20 s timeout -- nine forks and a 180 s worst case on a p5 with
#: eight NVMe, inside a 15 s tick -- while the signal it reads flips on the
#: order of days (F-H7). This is therefore also the *whole* staleness the cache
#: may add to a health *change*: a drive that starts failing inside the window
#: is reported when the window ends, one cache period late and never more.
_SMART_CACHE_SECONDS = 300.0


class _SmartHealth(NamedTuple):
    """The last SMART verdicts, and the device set they were read from.

    The device set is part of the key, so a hot-added NVMe (or one that
    vanished) is checked on the next tick rather than at the end of the window.
    """

    observed_at: datetime | None
    devices: tuple[str, ...]
    samples: tuple[HostMetricSample, ...]


_NO_SMART_HEALTH = _SmartHealth(None, (), ())

_NO_FILESYSTEM_WARNINGS: frozenset[str] = frozenset()

# ``disk_io_await_ms`` is iostat's ``await``; its definition is vendor-documented
# policy, so the sources are cited here verbatim and pinned by a test.
#
# Linux kernel, ``Documentation/admin-guide/iostats.rst``, section "I/O
# statistics fields" (mainline text as rendered at docs.kernel.org, read
# 2026-10-25). In ``/proc/diskstats`` the fields follow the major and minor
# device numbers and the device name, so kernel field *n* is column ``n + 2``.
KERNEL_IOSTATS_REFERENCE = "https://docs.kernel.org/admin-guide/iostats.html"
#: The field headings the arithmetic below rests on, by kernel field number.
KERNEL_IOSTATS_FIELDS: dict[int, str] = {
    1: "# of reads completed",
    4: "# of milliseconds spent reading",
    5: "# of writes completed",
    8: "# of milliseconds spent writing",
    10: "# of milliseconds spent doing I/Os",
    11: "weighted # of milliseconds spent doing I/Os",
}
#: Field 11, the column the collector divided by the completions before batch
#: 172: an in-flight-count-weighted time (its rate is iostat's ``aqu-sz``), not
#: a per-request time.
KERNEL_IOSTATS_FIELD_11_QUOTE = (
    "This field is incremented at each I/O start, I/O completion, I/O merge, "
    "or read of these stats by the number of I/Os in progress (field 9) times "
    "the number of milliseconds spent doing I/O since the last update of this "
    "field. This can provide an easy measure of both I/O completion time and "
    "the backlog that may be accumulating."
)
KERNEL_IOSTATS_RESET_QUOTE = (
    "All fields are cumulative, monotonic counters, except for field 9, which "
    "resets to zero as I/Os complete. The remaining fields reset at boot, on "
    "device reattachment or reinitialization, or when the underlying counter "
    "overflows. Applications reading these counters should detect and handle "
    "resets when comparing stat snapshots."
)
# sysstat iostat(1), section "Device Utilization Report" (man7.org rendering of
# the sysstat Git page dated AUGUST 2026; the sysstat 12.6.1 page carries the
# same sentences). sysstat's ``iostat.c`` computes each await as the class's
# request milliseconds over its completions in the interval
# (``r_await = (rd_ticks - rd_ticks') / (rd_ios - rd_ios')``) and ``aqu-sz`` as
# field 11's rate (``S_VALUE(rq_ticks', rq_ticks, itv) / 1000``).
SYSSTAT_IOSTAT_REFERENCE = "https://man7.org/linux/man-pages/man1/iostat.1.html"
SYSSTAT_IOSTAT_AWAIT_QUOTE = (
    "The average time (in milliseconds) for I/O requests issued to the device "
    "to be served. This includes the time spent by the requests in queue and "
    "the time spent servicing them."
)
SYSSTAT_IOSTAT_AQU_SZ_QUOTE = (
    "The average queue length of the requests that were issued to the device. "
    "Note: In previous versions, this field was known as avgqu-sz."
)
#: The ``/proc/diskstats`` counters the collector rates, by kernel field number
#: (``KERNEL_IOSTATS_FIELDS``). Field 9 (I/Os in progress) is the one field
#: that is not monotonic, so it takes no part in the restart check.
_DISK_COUNTER_FIELDS: dict[str, int] = {
    "reads": 1,
    "read_ms": 4,
    "writes": 5,
    "write_ms": 8,
    "io_ms": 10,
}


class _SmartScan(NamedTuple):
    """What ``smartctl --scan-open`` could and could not enumerate.

    ``failure`` set means the scan itself did not answer, so the drive set is
    unknown -- which is not the same as a node without drives.
    """

    devices: tuple[str, ...]
    unopened: tuple[tuple[str, str], ...]
    failure: str | None = None


#: smartctl's ``--scan-open`` prints a platform scan failure as this comment on
#: stdout and still exits 0, so the exit status alone cannot tell it from a
#: node with no drives.
_SMART_SCAN_ERROR = "scan_smart_devices:"
_SMART_OPEN_FAILED = "open failed"


class _InFlightProbe:
    """A mount syscall that missed its deadline and is still running.

    The syscall cannot be cancelled, so the worker it captured is abandoned --
    but the *mount* is remembered, across ticks: until this worker answers,
    that mount is unavailable by definition and probing it again would only
    abandon another thread and re-pay another deadline. One wedged mount
    therefore costs one thread and one deadline in total, not one per tick.
    """

    def __init__(
        self,
        operation: Callable[[str], Any],
        requests: queue.SimpleQueue[_ProbeRequest],
        responses: queue.SimpleQueue[_ProbeResponse],
    ) -> None:
        self.operation = operation
        self._requests = requests
        self._responses = responses

    def has_answered(self) -> bool:
        """True once the abandoned syscall returned, so the mount is probeable."""

        try:
            self._responses.get_nowait()
        except queue.Empty:
            return False
        # The worker is idle again and nothing else holds its queues: retire it
        # rather than keep a thread per mount that ever hung.
        self._requests.put(None)
        return True


class BoundedStatvfs:
    """Mount syscalls a hung mount cannot wedge the caller with.

    ``statvfs`` on a hard NFS mount or a Lustre client in MDS recovery blocks
    in uninterruptible sleep, and the collector calls it once per configured
    mount and once per shared mount per tick -- so the very condition
    ``shared_filesystem_unavailable`` exists to report stopped every batch,
    including the batch that would have reported it. The syscall cannot be
    cancelled, so it runs on a worker thread and the caller gives up after
    ``timeout_seconds`` with ``TimeoutError`` (an ``OSError``, which the
    callers already read as "this mount is unavailable"). :meth:`run` makes
    any other per-mount syscall under the same deadline -- the shared mounts'
    directory read -- and the parking below is per mount, whichever call hung
    it: a mount whose read is still parked is not asked for a ``statvfs``.

    A worker a hung mount captured is abandoned, and the mount is remembered
    until that worker answers: a dead FSx server that is N hung bind mounts
    costs N threads once, not N threads every tick until ``TasksMax`` is
    exhausted. Unlike a shared pool, one wedged mount also cannot make every
    healthy mount behind it look unavailable.

    Those parked workers do still accumulate -- one daemon thread and one dict
    entry per mount that is hung right now, held until the syscall returns,
    which for a file server that never comes back is never. ``max_in_flight``
    keeps that ceiling here rather than in the unit's ``TasksMax``, where
    hitting it means ``Thread.start()`` raises and breaks every *other* probe:
    past the cap a mount is reported unavailable without a worker at all.
    """

    def __init__(
        self,
        statvfs: Callable[[str], os.statvfs_result],
        *,
        timeout_seconds: float = 5.0,
        max_in_flight: int = 64,
    ) -> None:
        self._statvfs = statvfs
        self._timeout_seconds = timeout_seconds
        self._max_in_flight = max_in_flight
        self._requests: queue.SimpleQueue[_ProbeRequest] | None = None
        self._responses: queue.SimpleQueue[_ProbeResponse] | None = None
        self._in_flight: dict[str, _InFlightProbe] = {}

    def __call__(self, mount: str) -> os.statvfs_result:
        return self.run(mount, self._statvfs)

    def run(self, mount: str, operation: Callable[[str], _T]) -> _T:
        """``operation(mount)`` on the worker, or ``TimeoutError`` past the deadline."""

        what = _operation_name(operation)
        blocked = self._in_flight.get(mount)
        if blocked is not None:
            if not blocked.has_answered():
                raise ProbeStillRunning(mount, blocked.operation)
            del self._in_flight[mount]
        self._retire_answered_probes()
        if len(self._in_flight) >= self._max_in_flight:
            raise TimeoutError(
                f"{len(self._in_flight)} mount probes are already parked in the "
                f"kernel; not starting {what} for {mount}"
            )
        if self._requests is None or self._responses is None:
            self._start_worker()
        requests = self._requests
        responses = self._responses
        if requests is None or responses is None:  # pragma: no cover - defensive
            raise OSError(f"no probe worker for {mount}")
        requests.put((mount, operation))
        try:
            value, error = responses.get(timeout=self._timeout_seconds)
        except queue.Empty:
            self._in_flight[mount] = _InFlightProbe(operation, requests, responses)
            self._requests = None
            self._responses = None
            raise TimeoutError(
                f"{what}({mount}) did not answer within {self._timeout_seconds:g}s"
            ) from None
        if error is not None:
            raise error
        return value  # type: ignore[no-any-return]

    def _retire_answered_probes(self) -> None:
        """Free the slots of mounts whose abandoned syscall has since returned.

        Without this the cap would be reached by mounts that recovered long ago
        and are no longer probed -- an unmounted pod volume, say -- and a real
        hang would then be refused a worker on their account.
        """

        answered = [
            mount for mount, probe in self._in_flight.items() if probe.has_answered()
        ]
        for mount in answered:
            del self._in_flight[mount]

    def _start_worker(self) -> None:
        requests: queue.SimpleQueue[_ProbeRequest] = queue.SimpleQueue()
        responses: queue.SimpleQueue[_ProbeResponse] = queue.SimpleQueue()
        threading.Thread(
            target=self._serve,
            args=(requests, responses),
            name="gpu-fault-mount-probe",
            daemon=True,
        ).start()
        self._requests = requests
        self._responses = responses

    @staticmethod
    def _serve(
        requests: queue.SimpleQueue[_ProbeRequest],
        responses: queue.SimpleQueue[_ProbeResponse],
    ) -> None:
        while True:
            request = requests.get()
            if request is None:
                return
            mount, operation = request
            try:
                responses.put((operation(mount), None))
            except OSError as exc:
                responses.put((None, exc))
            except Exception as exc:  # pragma: no cover - defensive
                responses.put((None, OSError(str(exc))))


class HostSystemMetricsMixin:
    # Attributes supplied by the composed concrete implementation.
    filesystems: Any
    proc_root: Path
    statvfs_budget_seconds: float
    statvfs_timeout_seconds: float

    _delta: Callable[..., Any]
    _diskstats_devices: set[str]
    _diskstats_gone: set[str]
    _previous: Any
    _sample: Callable[..., Any]
    _statvfs_probe: BoundedStatvfs
    _statvfs_spent_seconds: float
    _statvfs_tick: datetime | None
    _unresponsive_mounts: set[str]
    runner: Callable[..., Any]

    #: Not part of the contract above: the composed class supplies nothing
    #: here, so the SMART cache starts from an immutable module-level default
    #: owned by the code that reads it.
    _smart_health = _NO_SMART_HEALTH
    #: The same for the conditions ``_filesystems`` has already logged: a
    #: configured mount the collector's sandbox hides, or an unreadable host
    #: mount table, stays so for as long as the configuration stands, and a
    #: warning per tick would flood the journal.
    _filesystem_warnings = _NO_FILESYSTEM_WARNINGS

    def _probe_mount(
        self,
        mount: str,
        observed_at: datetime,
        operation: Callable[[str], Any] | None = None,
    ) -> Any:
        """A mount syscall with a deadline, at most once per hung mount per tick.

        ``operation`` is ``statvfs`` by default, or the shared mounts' directory
        read; both share the deadline, the budget and the parking below. A
        mount that missed its deadline is not asked again in the same tick:
        the configured filesystem list and ``/proc/self/mountinfo`` overlap on
        an FSx or NFS path, and paying the deadline once per contributor would
        multiply the tick's worst case by the number of contributors that name
        the mount.

        The tick also has a total budget for waiting on mounts. One dead file
        server is not one mount but every bind mount served from it, and paying
        the per-mount deadline for each of them pushed the tick past the
        systemd watchdog -- so the batch carrying
        ``shared_filesystem_unavailable`` was killed before it was posted. Past
        the budget the remaining mounts are reported unavailable without being
        probed at all.
        """

        if self._statvfs_tick != observed_at:
            self._statvfs_tick = observed_at
            self._statvfs_spent_seconds = 0.0
            self._unresponsive_mounts.clear()
        if mount in self._unresponsive_mounts:
            raise TimeoutError(f"{mount} already missed its probe deadline this tick")
        if self._statvfs_spent_seconds >= self.statvfs_budget_seconds:
            self._unresponsive_mounts.add(mount)
            LOGGER.warning(
                "this tick already spent %gs of its %gs mount probe budget; "
                "reporting %s as unavailable without probing it",
                self._statvfs_spent_seconds,
                self.statvfs_budget_seconds,
                mount,
            )
            raise TimeoutError(
                f"the tick's mount probe budget was spent before {mount}"
            )
        started = time.monotonic()
        try:
            if operation is None:
                return self._statvfs_probe(mount)
            return self._statvfs_probe.run(mount, operation)
        except TimeoutError as error:
            self._unresponsive_mounts.add(mount)
            LOGGER.warning(
                "%s; reporting the mount as unavailable",
                error,
            )
            raise
        finally:
            self._statvfs_spent_seconds += time.monotonic() - started

    def _cpu(self, observed_at: datetime) -> list[HostMetricSample]:
        fields = [
            float(value)
            for value in Path("/proc/stat")
            .read_text(encoding="ascii")
            .splitlines()[0]
            .split()[1:]
        ]
        idle = fields[3] + (fields[4] if len(fields) > 4 else 0)
        # ``guest``/``guest_nice`` (fields 8 and 9) are already counted inside
        # ``user``/``nice``, so summing every field double counts them and
        # deflates the busy fraction on a node that runs VMs (F-H9).
        total = sum(fields[:8])
        result = []
        previous_total = self._previous.get("cpu-total")
        previous_idle = self._previous.get("cpu-idle")
        self._previous["cpu-total"] = (total, observed_at)
        self._previous["cpu-idle"] = (idle, observed_at)
        if previous_total and previous_idle and total > previous_total[0]:
            busy = 1 - ((idle - previous_idle[0]) / (total - previous_total[0]))
            result.append(
                self._sample(
                    "cpu_usage_percent",
                    max(0.0, min(100.0, busy * 100)),
                    "percent",
                )
            )
        load1 = float(Path("/proc/loadavg").read_text().split()[0])
        cpus = max(1, os.cpu_count() or 1)
        result.extend(
            [
                self._sample("load1", load1),
                self._sample("load1_per_cpu", load1 / cpus),
            ]
        )
        return result

    def _memory(self, _: datetime) -> list[HostMetricSample]:
        values = {}
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, raw = line.split(":", 1)
            values[key] = float(raw.strip().split()[0]) * 1024
        total = values["MemTotal"]
        free = values.get("MemFree", 0)
        available = values.get("MemAvailable", values.get("MemFree", 0))
        page_cache = max(
            0.0,
            values.get("Cached", 0)
            + values.get("SReclaimable", 0)
            - values.get("Shmem", 0),
        )
        swap_total = values.get("SwapTotal", 0)
        swap_free = values.get("SwapFree", 0)
        return [
            self._sample(
                "memory_used_percent",
                100 * (total - available) / total,
                "percent",
            ),
            self._sample("memory_available_bytes", available, "bytes"),
            self._sample(
                "memory_available_percent",
                100 * available / total,
                "percent",
            ),
            self._sample("memory_free_bytes", free, "bytes"),
            self._sample(
                "memory_free_percent",
                100 * free / total,
                "percent",
            ),
            self._sample("page_cache_bytes", page_cache, "bytes"),
            self._sample(
                "page_cache_percent",
                100 * page_cache / total,
                "percent",
            ),
            self._sample(
                "swap_used_percent",
                (100 * (swap_total - swap_free) / swap_total if swap_total else 0),
                "percent",
            ),
        ]

    def _mount_tables(
        self,
    ) -> tuple[tuple[MountEntry, ...] | None, tuple[MountEntry, ...] | None]:
        """The collector's own mount table and the host's (PID 1's).

        Either is None when it cannot be read. The one place both tables are
        read, for ``_filesystems`` and the shared filesystem contributor alike.
        """

        return (
            read_mount_table(Path("/proc/self/mountinfo")),
            read_mount_table(self.proc_root / "1" / "mountinfo"),
        )

    def _filesystems(self, observed_at: datetime) -> list[HostMetricSample]:
        result = []
        # This process runs in its unit's sandbox (``mount_table``): its own
        # ``statvfs`` flags call every healthy mount read-only, so the
        # read-only verdict comes from PID 1's -- the host's -- mount table.
        own, host = self._mount_tables()
        host_table = self.proc_root / "1" / "mountinfo"
        if host is None:
            self._warn_filesystems_once(
                "\0host-mount-table",
                "the host mount table %s cannot be read; a read-only configured "
                "filesystem is not reported",
                host_table,
            )
        for mount in dict.fromkeys(self.filesystems):
            view = host_mount(mount, own, host)
            if not view.same_filesystem:
                # ``ProtectHome=`` (or another sandbox mount) covers the path:
                # ``statvfs`` here would read that mount, not the host's
                # filesystem. A path the collector cannot observe is missing
                # evidence -- never a usable, read-only or missing reading.
                self._warn_filesystems_once(
                    mount,
                    "configured filesystem %s is hidden by the collector's sandbox "
                    "(the host holds it on %s); it is not reported",
                    mount,
                    view.entry.mount_point if view.entry is not None else "?",
                )
                continue
            # No ``os.path.exists`` guard: ``stat`` blocks on a dead mount for
            # the same reason ``statvfs`` does, and ``statvfs`` already reports
            # a missing path as ENOENT.
            try:
                stat = self._probe_mount(mount, observed_at)
            except TimeoutError:
                # ``GPU_FAULT_FILESYSTEMS`` may name an FSx or NFS path, and a
                # local mount that stops answering is worse news still; either
                # way the consumer's rule for "this mount is not usable" is
                # ``shared_filesystem_unavailable``. The same failure mode as
                # ``_shared_filesystems`` gives a hung ``statvfs``, so a mount
                # both contributors name carries one meaning, not two.
                result.append(
                    HostMetricSample(
                        name="shared_filesystem_unavailable",
                        value=1.0,
                        device=mount,
                        labels={"failure_mode": "PROBE_TIMEOUT"},
                    )
                )
                continue
            except OSError as error:
                if isinstance(error, FileNotFoundError) and not view.same_directory:
                    # Missing from the collector's private copy of the
                    # directory (``PrivateTmp=``), which says nothing about
                    # the host's: not a vanished mount.
                    self._warn_filesystems_once(
                        mount,
                        "configured filesystem %s is under a directory private to "
                        "the collector's sandbox; it is not reported",
                        mount,
                    )
                    continue
                # A configured mount whose ``statvfs`` fails is unusable, not
                # absent-and-fine: ENOENT is a required mount that vanished
                # (GF-GENERAL-LIVE-004 ``required_mount_missing``), and any
                # other error -- EIO, EACCES, ESTALE -- a mount whose status
                # cannot be read (``probe_error``). Reported, never skipped: an
                # unreadable status reads unusable (UNKNOWN), never healthy.
                # Before this branch existed the mount was dropped from the
                # batch, so a required mount that had gone looked exactly like
                # one the node was never meant to have.
                result.append(
                    self._filesystem_unavailable(
                        mount,
                        "MOUNT_MISSING"
                        if isinstance(error, FileNotFoundError)
                        else "PROBE_FAILED",
                    )
                )
                continue
            # A read-only remount keeps its blocks and inodes, so the capacity
            # signals below read healthy while the mount refuses every write:
            # read-only is its own "not usable" signal (004 ``read_only``), and
            # the gauge reports 0 for a usable read-write mount so the signal
            # clears when the mount is remounted read-write. Read-only is the
            # host mount's verdict, per mount or per superblock -- never this
            # process's ``ST_RDONLY``, which ``ProtectSystem=strict`` sets on
            # every healthy mount -- and an unreadable host table is no
            # evidence of a read-only filesystem.
            read_only = view.entry is not None and view.entry.read_only
            result.append(
                self._filesystem_unavailable(mount, "READ_ONLY" if read_only else None)
            )
            total = stat.f_blocks * stat.f_frsize
            available = stat.f_bavail * stat.f_frsize
            if total:
                used_percent = 100 * (total - available) / total
                result.append(
                    self._sample(
                        "filesystem_used_percent",
                        used_percent,
                        "percent",
                        mount,
                    )
                )
                filesystem_type = self._filesystem_type(mount)
                if filesystem_type not in {
                    "nfs",
                    "nfs4",
                    "lustre",
                    "cifs",
                    "ceph",
                    "efs",
                    "fuse.s3fs",
                    "fuse.goofys",
                }:
                    result.append(
                        self._sample(
                            "local_filesystem_used_percent",
                            used_percent,
                            "percent",
                            mount,
                        )
                    )
            # Inodes run out independently of bytes: a table of small files
            # refuses every new file with ENOSPC while ``f_bavail`` still
            # shows free space, so the byte reading alone read such a mount as
            # healthy (GF-GENERAL-LIVE-004 inode_full). ``f_files == 0`` is a
            # filesystem that keeps no fixed inode table (btrfs, many FUSE
            # and NFS exports) -- not graded, rather than 0 % or 100 %.
            if stat.f_files:
                inodes_used = 100 * (stat.f_files - stat.f_ffree) / stat.f_files
                result.append(
                    self._sample(
                        "filesystem_inodes_used_percent",
                        max(0.0, min(100.0, inodes_used)),
                        "percent",
                        mount,
                    )
                )
        return result

    @staticmethod
    def _filesystem_unavailable(
        mount: str, failure_mode: str | None
    ) -> HostMetricSample:
        """Availability gauge for one configured mount (GF-GENERAL-LIVE-004).

        ``failure_mode`` is None for a usable read-write mount (value 0) and
        the reason otherwise (value 1): ``READ_ONLY`` for a mount the host's
        mount table shows read-only (per mount or per superblock),
        ``MOUNT_MISSING`` for a configured mount whose ``statvfs`` is ENOENT,
        ``PROBE_FAILED`` for one whose ``statvfs`` raised any other error.
        Kept distinct from ``filesystem_used_percent``: free bytes say nothing
        about a mount that is read-only, gone, or unreadable, so the capacity
        reading must not be read as proof the mount is healthy.
        """

        return HostMetricSample(
            name="filesystem_unavailable",
            value=0.0 if failure_mode is None else 1.0,
            device=mount,
            labels={} if failure_mode is None else {"failure_mode": failure_mode},
        )

    def _warn_filesystems_once(self, key: str, message: str, *args: object) -> None:
        if key in self._filesystem_warnings:
            return
        self._filesystem_warnings = self._filesystem_warnings | {key}
        LOGGER.warning(message, *args)

    @staticmethod
    def _filesystem_type(path: str) -> str | None:
        mountinfo = Path("/proc/self/mountinfo")
        if not mountinfo.exists():
            return None
        target = os.path.abspath(path)
        candidates: list[tuple[int, str]] = []
        for line in mountinfo.read_text().splitlines():
            fields = line.split()
            try:
                separator = fields.index("-")
            except ValueError:
                continue
            if len(fields) <= separator + 1 or len(fields) <= 4:
                continue
            mount_point = fields[4].replace("\\040", " ")
            prefix = mount_point.rstrip("/") + "/"
            if target == mount_point or target.startswith(prefix) or mount_point == "/":
                candidates.append(
                    (
                        len(mount_point),
                        fields[separator + 1].lower(),
                    )
                )
        return max(candidates)[1] if candidates else None

    def _lustre(self, observed_at: datetime) -> list[HostMetricSample]:
        result = []
        metric_names = {
            "read_bytes": "lustre_read_bytes_delta",
            "write_bytes": "lustre_write_bytes_delta",
            "dirty_pages_hits": "lustre_dirty_page_hits_delta",
            "dirty_pages_misses": ("lustre_dirty_page_misses_delta"),
        }
        for path in Path("/proc/fs/lustre/llite").glob("*/stats"):
            device = path.parent.name
            for line in path.read_text().splitlines():
                fields = line.split()
                if not fields or fields[0] not in metric_names:
                    continue
                numeric = []
                for value in fields[1:]:
                    try:
                        numeric.append(float(value))
                    except ValueError:
                        continue
                if not numeric:
                    continue
                current = numeric[-1]
                change = self._delta(
                    f"lustre/{device}/{fields[0]}",
                    current,
                    observed_at,
                )
                if change:
                    result.append(
                        self._sample(
                            metric_names[fields[0]],
                            change[0],
                            ("bytes" if fields[0].endswith("_bytes") else "events"),
                            device,
                        )
                    )
        return result

    def _diskstats(self, observed_at: datetime) -> list[HostMetricSample]:
        """``disk_io_util_percent`` and ``disk_io_await_ms`` per block device.

        ``disk_io_await_ms`` is iostat's ``await`` (``SYSSTAT_IOSTAT_REFERENCE``:
        "The average time (in milliseconds) for I/O requests issued to the
        device to be served. This includes the time spent by the requests in
        queue and the time spent servicing them."): the request milliseconds
        completed in the interval over the requests completed in it, which for
        reads and writes together is ``/proc/diskstats`` kernel fields (4 + 8)
        / (1 + 5) -- "# of milliseconds spent reading" plus "# of milliseconds
        spent writing" over "# of reads completed" plus "# of writes completed"
        (``KERNEL_IOSTATS_REFERENCE``, section "I/O statistics fields").

        Until batch 172 the numerator was field 11, "weighted # of milliseconds
        spent doing I/Os": the in-flight count integrated over time
        (``KERNEL_IOSTATS_FIELD_11_QUOTE``), whose rate is iostat's ``aqu-sz``
        (``SYSSTAT_IOSTAT_AQU_SZ_QUOTE``), not a per-request time. Under a
        steady load the two ratios agree (Little's law: queue length = arrival
        rate x time in system), so the 100 ms rule keeps its meaning; they
        diverge whenever the backlog changes inside the interval, and 5.0+
        kernels print field 11 as the summed request time of reads, writes,
        discards and flushes (``block/genhd.c``, ``diskstats_show``), so an
        fsync-heavy writer inflated the product's await with flush time that
        no completed read or write accounts for -- the GF-GENERAL-LIVE-006
        ``reference_iostat`` divergence.

        Discards (fields 12-15) and flushes (16-17) stay out of the await on
        purpose: they are what ``d_await``/``f_await`` report, the rule reads
        the latency a job's reads and writes see, and the 14-column layout of
        older kernels has no such fields. ``disk_io_util_percent`` keeps field
        10 ("# of milliseconds spent doing I/Os"), which grows while a request
        of any class is in flight.

        Counter restarts: the fields "reset at boot, on device reattachment or
        reinitialization, or when the underlying counter overflows", and
        "Applications reading these counters should detect and handle resets
        when comparing stat snapshots" (``KERNEL_IOSTATS_RESET_QUOTE``). A
        device whose monotonic counters went backwards -- a dm mapping recreated
        under the same name, a re-attached NVMe -- reports nothing for the
        interval spanning the restart and is rated from the restarted baseline
        on the next tick. Clamping each counter's delta to zero on its own, as
        before, mixed a clamped numerator with a live denominator (or the
        reverse) into an await of 0 ms, or of hundreds, that no request had.

        A device that left ``/proc/diskstats`` since the previous read, and a
        restarted one (a new device instance under the old name), are reported
        as ``device_state=GONE`` for both metrics until a batch is delivered
        (:meth:`_device_gone_samples`), and a vanished device's baseline is
        forgotten so a same-named successor is rated from its own first read.
        """

        result = []
        present: set[str] = set()
        for line in Path("/proc/diskstats").read_text().splitlines():
            fields = line.split()
            if len(fields) < 14:
                continue
            device = fields[2]
            if device.startswith(("loop", "ram")):
                continue
            present.add(device)
            restarted = False
            deltas: dict[str, float] = {}
            elapsed = 0.0
            for name, field in _DISK_COUNTER_FIELDS.items():
                key = f"disk/{device}/{name}"
                value = float(fields[field + 2])
                last = self._previous.get(key)
                if last is not None and value < last[0]:
                    restarted = True
                change = self._delta(key, value, observed_at)
                if change is not None:
                    deltas[name], elapsed = change
            if restarted:
                # The device instance the control plane knew is gone and this
                # is its successor; the new baseline is already recorded.
                self._diskstats_gone.add(device)
            if restarted or len(deltas) < len(_DISK_COUNTER_FIELDS) or elapsed <= 0:
                continue
            result.append(
                self._sample(
                    "disk_io_util_percent",
                    min(100.0, 100 * deltas["io_ms"] / (elapsed * 1000)),
                    "percent",
                    device,
                )
            )
            completed = deltas["reads"] + deltas["writes"]
            if completed > 0:
                result.append(
                    self._sample(
                        "disk_io_await_ms",
                        (deltas["read_ms"] + deltas["write_ms"]) / completed,
                        "milliseconds",
                        device,
                    )
                )
        for device in self._diskstats_devices - present:
            self._diskstats_gone.add(device)
            for name in _DISK_COUNTER_FIELDS:
                self._previous.pop(f"disk/{device}/{name}", None)
        self._diskstats_devices = present
        return [*self._device_gone_samples(), *result]

    def _device_gone_samples(self) -> list[HostMetricSample]:
        """One ``device_state=GONE`` sample per ``disk_io_*`` metric for every
        block device that left ``/proc/diskstats`` (or restarted its counters)
        since the last delivered batch (batch 160).

        The control plane grades such a sample as a clear of the device's
        signals and never as a reading (``host_device_signals.device_gone``):
        a device that is not there can neither sustain a fault nor measure
        0 ms. Without it the latched ``disk_io_await_ms`` signal of a torn-down
        dm device stayed notified and silenced a same-named successor
        (GF-GENERAL-LIVE-006, live 2026-10-25). The samples come first in the
        tick, so a successor rated in the same tick is judged after the clear
        and opens a new episode. They are repeated every tick until a batch is
        delivered (``collect_once`` empties the set then): the tick a device
        vanishes is usually an edge of its own -- its threshold reason drops
        out, ``recovered`` -- and otherwise the next health summary carries
        them.
        """

        labels = {DEVICE_STATE_LABEL: DEVICE_STATE_GONE}
        return [
            HostMetricSample(
                name=name, value=0.0, unit=unit, device=device, labels=labels
            )
            for device in sorted(self._diskstats_gone)
            for name, unit in (
                ("disk_io_util_percent", "percent"),
                ("disk_io_await_ms", "milliseconds"),
            )
        ]

    def _smart(self, observed_at: datetime) -> list[HostMetricSample]:
        """SMART health per drive, re-read at most once per cache period.

        The ``--scan-open`` fork still runs every tick -- one fork against one
        per drive -- because it is what makes a drive that appeared or vanished
        rescan immediately instead of waiting out the window. Only the per-drive
        ``-H`` checks are cached, and the cached verdicts are still reported on
        every tick: the consumer reads ``smart_health_failed`` as a latest
        value, so a tick that omitted it would read as a drive that stopped
        being watched.

        What SMART could not say is reported as such, never as a verdict
        (GF-GENERAL-LIVE-007). A scan or a drive query that did not answer --
        a non-zero ``--scan-open`` or its ``scan_smart_devices`` error line, a
        drive the scan could not open, a ``-H`` that timed out or returned no
        usable verdict -- is UNKNOWN: ``smart_query_failed`` plus a collection
        error, which holds host validation and grades nothing. A failed scan
        used to parse as an empty drive set, i.e. a node with nothing to
        check. SMART the node cannot provide -- no ``smartctl``, no drive, a
        drive without a SMART status -- is NOT_SUPPORTED:
        ``smart_status_unsupported`` on every tick and no collection error,
        because the gap is permanent and an error would hold every host
        validation of the node until its deadline. Neither one is a passed
        drive, and neither quarantines one.
        """

        if not shutil.which("smartctl"):
            return [self._smart_unsupported(None, "TOOL_MISSING")]
        scan = self._smart_devices()
        if scan.failure is not None:
            raise HostObservationError(
                "smartctl --scan-open failed, so the drive set is unknown: "
                + scan.failure,
                [self._smart_query_failed(None, "SCAN_FAILED")],
            )
        if not scan.devices and not scan.unopened:
            return [self._smart_unsupported(None, "NO_DEVICES")]
        samples, failures = self._smart_verdicts(observed_at, scan.devices)
        for device, reason in scan.unopened:
            samples.append(self._smart_query_failed(device, "OPEN_FAILED"))
            failures.append(f"{device}: open failed: {reason}")
        if failures:
            raise HostObservationError(
                "SMART health is unknown for " + "; ".join(failures), samples
            )
        return samples

    def _smart_verdicts(
        self, observed_at: datetime, devices: tuple[str, ...]
    ) -> tuple[list[HostMetricSample], list[str]]:
        """The ``-H`` answers for ``devices``, and the drives that gave none."""

        cached = self._smart_health
        if (
            cached.observed_at is not None
            and cached.devices == devices
            and (observed_at - cached.observed_at).total_seconds()
            < _SMART_CACHE_SECONDS
        ):
            return list(cached.samples), []
        samples: list[HostMetricSample] = []
        failures: list[str] = []
        for device in devices:
            verdict, failure = self._smart_verdict(device)
            samples.extend(verdict)
            if failure is not None:
                failures.append(f"{device}: {failure}")
        # A failed query must not start a cache window. Caching it the way a
        # verdict is cached would hide a drive that has begun refusing SMART
        # queries -- a real dying-NVMe failure mode -- for a whole period,
        # which is exactly what the cache may not do to a health change. With
        # no timestamp the next tick is a miss and re-queries.
        self._smart_health = _SmartHealth(
            None if failures else observed_at, devices, tuple(samples)
        )
        return samples, failures

    def _smart_devices(self) -> _SmartScan:
        """The drives ``smartctl`` can open, sorted so the key is stable.

        ``--scan-open`` lists a drive it could not open only as a comment,
        ``# /dev/nvme1 -d nvme # /dev/nvme1, NVMe device open failed: <why>``,
        which used to drop the drive from the set silently. It is kept as
        unopened. A scan that did not answer at all -- an exit status, a
        timeout, or the ``# scan_smart_devices: <error>`` line smartctl prints
        before exiting 0 -- returns ``failure`` and no drives.
        """

        try:
            scan = self.runner(
                ["smartctl", "--scan-open"],
                capture_output=True,
                text=True,
                timeout=20,
                check=False,
            )
        except (subprocess.TimeoutExpired, OSError) as exc:
            return _SmartScan((), (), f"{type(exc).__name__}: {exc}")
        output = scan.stdout or ""
        if scan.returncode != 0:
            detail = (scan.stderr or output).strip()[:200]
            return _SmartScan((), (), f"exit status {scan.returncode}: {detail}")
        devices: set[str] = set()
        unopened: dict[str, str] = {}
        for line in output.splitlines():
            fields = line.split(maxsplit=1)
            if fields and fields[0].startswith("/dev/"):
                devices.add(fields[0])
                continue
            comment = line.strip()
            if not comment.startswith("#"):
                continue
            body = comment.lstrip("#").strip()
            if body.startswith(_SMART_SCAN_ERROR):
                return _SmartScan((), (), body[:200])
            named = body.split(maxsplit=1)
            if named and named[0].startswith("/dev/") and _SMART_OPEN_FAILED in body:
                reason = body.split(_SMART_OPEN_FAILED, 1)[1].lstrip(": ")
                unopened[named[0]] = reason.strip()[:200]
        return _SmartScan(
            tuple(sorted(devices)),
            tuple(
                sorted(
                    (device, reason)
                    for device, reason in unopened.items()
                    if device not in devices
                )
            ),
        )

    def _smart_verdict(self, device: str) -> tuple[list[HostMetricSample], str | None]:
        """One drive's verdict, and why the query gave none if it did not.

        A refused, hung or malformed query is unknown, not a failed device
        self-assessment: report ``smart_query_failed`` plus a collection
        error, and retry next cycle instead of caching it or authorizing
        hardware quarantine. A hung query is caught here, per drive: raised,
        it discarded the verdicts of every other drive in the tick, including
        one that had just reported FAILED.

        A verdict-less answer from a *clean* ``smartctl`` is a different thing:
        that is a device ``--scan-open`` can open but which implements no SMART
        status at all. It has no health to report and is not a failure, so it
        is NOT_SUPPORTED and stays cacheable -- calling it failed would hold
        every node that carries such a device in an observation error.
        """

        try:
            check = self.runner(
                ["smartctl", "-H", "-j", device],
                capture_output=True,
                text=True,
                timeout=20,
                check=False,
            )
        except (subprocess.TimeoutExpired, OSError) as exc:
            return [self._smart_query_failed(device)], f"{type(exc).__name__}: {exc}"
        try:
            payload = json.loads(check.stdout)
        except json.JSONDecodeError:
            return [self._smart_query_failed(device)], (
                f"exit status {check.returncode}, output is not JSON"
            )
        if not isinstance(payload, dict):
            return [self._smart_query_failed(device)], "output is not a JSON object"
        status = payload.get("smart_status")
        passed = status.get("passed") if isinstance(status, dict) else None
        if passed is None:
            reported = payload.get("smartctl")
            exit_status = (
                reported.get("exit_status", 0) if isinstance(reported, dict) else 0
            )
            if check.returncode or exit_status:
                return [self._smart_query_failed(device)], (
                    f"no verdict, exit status {check.returncode or exit_status}"
                )
            return [self._smart_unsupported(device, "NO_SMART_STATUS")], None
        if not isinstance(passed, bool):
            # ``"false"`` is truthy and ``0`` falsy: either would have been read
            # as a verdict the drive never gave.
            return [self._smart_query_failed(device)], (
                f"smart_status.passed is not a boolean: {passed!r}"
            )
        return [
            self._sample(
                "smart_health_failed",
                0 if passed else 1,
                None,
                device,
            )
        ], None

    @staticmethod
    def _smart_query_failed(
        device: str | None, failure_mode: str = "QUERY_FAILED"
    ) -> HostMetricSample:
        """The marker for a drive whose SMART query could not be answered.

        Kept separate from ``smart_health_failed``: unreadable health is not
        proof of either a healthy drive or a hardware failure. ``device`` is
        None for a scan that failed (``SCAN_FAILED``) and so names no drive;
        ``OPEN_FAILED`` is a drive the scan listed but could not open.
        """

        return HostMetricSample(
            name="smart_query_failed",
            value=1,
            device=device,
            labels={"failure_mode": failure_mode},
        )

    @staticmethod
    def _smart_unsupported(device: str | None, reason: str) -> HostMetricSample:
        """The marker for SMART this node or drive cannot provide.

        ``TOOL_MISSING`` and ``NO_DEVICES`` name no drive; ``NO_SMART_STATUS``
        is a drive that answers cleanly with no PASSED/FAILED verdict. Not a
        rule input and not a collection error: a capability gap is neither a
        healthy scan nor an unknown one.
        """

        return HostMetricSample(
            name=SMART_STATUS_UNSUPPORTED,
            value=1,
            device=device,
            labels={"reason": reason},
        )

from __future__ import annotations

from typing import Any, Callable, NamedTuple

import json
import logging
import os
import queue
import shutil
import threading
import time
from datetime import datetime
from pathlib import Path


from gpu_fault.host_health import (
    HostMetricSample,
)


LOGGER = logging.getLogger(__name__)

#: Sentinel that retires an abandoned ``statvfs`` worker once its mount
#: answers, so a mount that recovers does not leave the thread behind.
_RETIRE = "\0retire"

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


class _InFlightProbe:
    """A ``statvfs`` call that missed its deadline and is still running.

    The syscall cannot be cancelled, so the worker it captured is abandoned --
    but the *mount* is remembered, across ticks: until this worker answers,
    that mount is unavailable by definition and probing it again would only
    abandon another thread and re-pay another deadline. One wedged mount
    therefore costs one thread and one deadline in total, not one per tick.
    """

    def __init__(
        self,
        requests: queue.SimpleQueue[str],
        responses: queue.SimpleQueue[tuple[os.statvfs_result | None, OSError | None]],
    ) -> None:
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
        self._requests.put(_RETIRE)
        return True


class BoundedStatvfs:
    """``os.statvfs`` a hung mount cannot wedge the caller with.

    ``statvfs`` on a hard NFS mount or a Lustre client in MDS recovery blocks
    in uninterruptible sleep, and the collector calls it once per configured
    mount and once per shared mount per tick -- so the very condition
    ``shared_filesystem_unavailable`` exists to report stopped every batch,
    including the batch that would have reported it. The syscall cannot be
    cancelled, so it runs on a worker thread and the caller gives up after
    ``timeout_seconds`` with ``TimeoutError`` (an ``OSError``, which the
    callers already read as "this mount is unavailable").

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
        self._requests: queue.SimpleQueue[str] | None = None
        self._responses: (
            queue.SimpleQueue[tuple[os.statvfs_result | None, OSError | None]] | None
        ) = None
        self._in_flight: dict[str, _InFlightProbe] = {}

    def __call__(self, mount: str) -> os.statvfs_result:
        blocked = self._in_flight.get(mount)
        if blocked is not None:
            if not blocked.has_answered():
                raise TimeoutError(
                    f"statvfs({mount}) from an earlier tick has not returned"
                )
            del self._in_flight[mount]
        self._retire_answered_probes()
        if len(self._in_flight) >= self._max_in_flight:
            raise TimeoutError(
                f"{len(self._in_flight)} statvfs calls are already parked in the "
                f"kernel; not starting one for {mount}"
            )
        if self._requests is None or self._responses is None:
            self._start_worker()
        requests = self._requests
        responses = self._responses
        if requests is None or responses is None:  # pragma: no cover - defensive
            raise OSError(f"no statvfs worker for {mount}")
        requests.put(mount)
        try:
            stat, error = responses.get(timeout=self._timeout_seconds)
        except queue.Empty:
            self._in_flight[mount] = _InFlightProbe(requests, responses)
            self._requests = None
            self._responses = None
            raise TimeoutError(
                f"statvfs({mount}) did not answer within {self._timeout_seconds:g}s"
            ) from None
        if error is not None:
            raise error
        if stat is None:  # pragma: no cover - defensive
            raise OSError(f"statvfs({mount}) returned nothing")
        return stat

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
        requests: queue.SimpleQueue[str] = queue.SimpleQueue()
        responses: queue.SimpleQueue[
            tuple[os.statvfs_result | None, OSError | None]
        ] = queue.SimpleQueue()
        threading.Thread(
            target=self._serve,
            args=(requests, responses),
            name="gpu-fault-statvfs",
            daemon=True,
        ).start()
        self._requests = requests
        self._responses = responses

    def _serve(
        self,
        requests: queue.SimpleQueue[str],
        responses: queue.SimpleQueue[tuple[os.statvfs_result | None, OSError | None]],
    ) -> None:
        while True:
            mount = requests.get()
            if mount == _RETIRE:
                return
            try:
                responses.put((self._statvfs(mount), None))
            except OSError as exc:
                responses.put((None, exc))
            except Exception as exc:  # pragma: no cover - defensive
                responses.put((None, OSError(str(exc))))


class HostSystemMetricsMixin:
    # Attributes supplied by the composed concrete implementation.
    filesystems: Any
    statvfs_budget_seconds: float
    statvfs_timeout_seconds: float

    _delta: Callable[..., Any]
    _previous: Any
    _sample: Callable[..., Any]
    _statvfs_probe: Callable[[str], os.statvfs_result]
    _statvfs_spent_seconds: float
    _statvfs_tick: datetime | None
    _unresponsive_mounts: set[str]
    runner: Callable[..., Any]

    #: Not part of the contract above: the composed class supplies nothing
    #: here, so the SMART cache starts from an immutable module-level default
    #: owned by the code that reads it.
    _smart_health = _NO_SMART_HEALTH

    def _probe_mount(self, mount: str, observed_at: datetime) -> os.statvfs_result:
        """``statvfs`` with a deadline, at most once per mount per tick.

        A mount that missed its deadline is not asked again in the same tick:
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
            raise TimeoutError(f"{mount} already missed its statvfs deadline this tick")
        if self._statvfs_spent_seconds >= self.statvfs_budget_seconds:
            self._unresponsive_mounts.add(mount)
            LOGGER.warning(
                "this tick already spent %gs of its %gs statvfs budget; reporting "
                "%s as unavailable without probing it",
                self._statvfs_spent_seconds,
                self.statvfs_budget_seconds,
                mount,
            )
            raise TimeoutError(f"the tick's statvfs budget was spent before {mount}")
        started = time.monotonic()
        try:
            return self._statvfs_probe(mount)
        except TimeoutError:
            self._unresponsive_mounts.add(mount)
            LOGGER.warning(
                "statvfs(%s) did not answer within %gs; reporting the mount as "
                "unavailable",
                mount,
                self.statvfs_timeout_seconds,
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

    def _filesystems(self, observed_at: datetime) -> list[HostMetricSample]:
        result = []
        for mount in dict.fromkeys(self.filesystems):
            # No ``os.path.exists`` guard: ``stat`` blocks on a dead mount for
            # the same reason ``statvfs`` does, and ``statvfs`` already reports
            # a missing path as ENOENT.
            try:
                stat = self._probe_mount(mount, observed_at)
            except TimeoutError:
                # ``GPU_FAULT_FILESYSTEMS`` may name an FSx or NFS path, and a
                # local mount that stops answering is worse news still; either
                # way the consumer's rule for "this mount is not usable" is
                # ``shared_filesystem_unavailable``.
                result.append(
                    self._sample("shared_filesystem_unavailable", 1, None, mount)
                )
                continue
            except OSError:
                # A configured mount that this node does not have is not a
                # fault; it was skipped before this guard existed too.
                continue
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
        return result

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

    def _shared_filesystems(self, observed_at: datetime) -> list[HostMetricSample]:
        result = []
        mountinfo = Path("/proc/self/mountinfo")
        if not mountinfo.exists():
            return result
        for line in mountinfo.read_text().splitlines():
            fields = line.split()
            if " - " not in line:
                continue
            separator = fields.index("-")
            fs_type = fields[separator + 1]
            if fs_type not in {"lustre", "nfs", "nfs4"}:
                continue
            mount = fields[4].replace("\\040", " ")
            try:
                # ``TimeoutError`` is an ``OSError``: a mount that never
                # answers lands in the same "unavailable" branch as one that
                # answers ESTALE, which is what the CRITICAL rule wants.
                stat = self._probe_mount(mount, observed_at)
                total = stat.f_blocks * stat.f_frsize
                available = stat.f_bavail * stat.f_frsize
                result.append(
                    self._sample(
                        "shared_filesystem_unavailable",
                        0,
                        None,
                        mount,
                    )
                )
                if total:
                    result.append(
                        self._sample(
                            "shared_filesystem_used_percent",
                            100 * (total - available) / total,
                            "percent",
                            mount,
                        )
                    )
            except OSError:
                result.append(
                    self._sample(
                        "shared_filesystem_unavailable",
                        1,
                        None,
                        mount,
                    )
                )
        return result

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
        result = []
        for line in Path("/proc/diskstats").read_text().splitlines():
            fields = line.split()
            if len(fields) < 14:
                continue
            device = fields[2]
            if device.startswith(("loop", "ram")):
                continue
            io_ms = float(fields[12])
            completed_ios = float(fields[3]) + float(fields[7])
            weighted_io_ms = float(fields[13])
            change = self._delta(f"disk/{device}/io_ms", io_ms, observed_at)
            if change and change[1] > 0:
                result.append(
                    self._sample(
                        "disk_io_util_percent",
                        min(
                            100.0,
                            100 * change[0] / (change[1] * 1000),
                        ),
                        "percent",
                        device,
                    )
                )
            io_change = self._delta(
                f"disk/{device}/completed_ios",
                completed_ios,
                observed_at,
            )
            weighted_change = self._delta(
                f"disk/{device}/weighted_io_ms",
                weighted_io_ms,
                observed_at,
            )
            if io_change and weighted_change and io_change[0] > 0:
                result.append(
                    self._sample(
                        "disk_io_await_ms",
                        weighted_change[0] / io_change[0],
                        "milliseconds",
                        device,
                    )
                )
        return result

    def _smart(self, observed_at: datetime) -> list[HostMetricSample]:
        """SMART health per drive, re-read at most once per cache period.

        The ``--scan-open`` fork still runs every tick -- one fork against one
        per drive -- because it is what makes a drive that appeared or vanished
        rescan immediately instead of waiting out the window. Only the per-drive
        ``-H`` checks are cached, and the cached verdicts are still reported on
        every tick: the consumer reads ``smart_health_failed`` as a latest
        value, so a tick that omitted it would read as a drive that stopped
        being watched.
        """

        if not shutil.which("smartctl"):
            return []
        devices = self._smart_devices()
        cached = self._smart_health
        if (
            cached.observed_at is not None
            and cached.devices == devices
            and (observed_at - cached.observed_at).total_seconds()
            < _SMART_CACHE_SECONDS
        ):
            return list(cached.samples)
        samples: list[HostMetricSample] = []
        answered = True
        for device in devices:
            verdict, queried = self._smart_verdict(device)
            samples.extend(verdict)
            answered = answered and queried
        # A failed query must not start a cache window. Caching it the way a
        # verdict is cached would hide a drive that has begun refusing SMART
        # queries -- a real dying-NVMe failure mode -- for a whole period,
        # which is exactly what the cache may not do to a health change. With
        # no timestamp the next tick is a miss and re-queries.
        self._smart_health = _SmartHealth(
            observed_at if answered else None, devices, tuple(samples)
        )
        return samples

    def _smart_devices(self) -> tuple[str, ...]:
        """The drives ``smartctl`` can open, sorted so the key is stable."""

        scan = self.runner(
            ["smartctl", "--scan-open"],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
        devices = set()
        for line in scan.stdout.splitlines():
            fields = line.split(maxsplit=1)
            if fields and fields[0].startswith("/dev/"):
                devices.add(fields[0])
        return tuple(sorted(devices))

    def _smart_verdict(self, device: str) -> tuple[list[HostMetricSample], bool]:
        """One drive's verdict, and whether the query itself answered.

        A drive that has begun refusing SMART queries is a real dying-NVMe
        failure mode, and it used to read exactly like a healthy one: no
        sample at all. It is now reported as a failure of its own, tagged
        ``failure_mode=QUERY_FAILED`` so an operator can tell it from a drive
        whose own self-assessment failed, and the caller does not cache it.

        A verdict-less answer from a *clean* ``smartctl`` is a different thing:
        that is a device ``--scan-open`` can open but which implements no SMART
        status at all. It has no health to report and is not a failure, so it
        stays cacheable -- calling it failed would quarantine every node that
        carries such a device.
        """

        check = self.runner(
            ["smartctl", "-H", "-j", device],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
        try:
            payload = json.loads(check.stdout)
        except json.JSONDecodeError:
            return [self._smart_query_failed(device)], False
        if not isinstance(payload, dict):
            return [self._smart_query_failed(device)], False
        status = payload.get("smart_status")
        passed = status.get("passed") if isinstance(status, dict) else None
        if passed is None:
            reported = payload.get("smartctl")
            exit_status = (
                reported.get("exit_status", 0) if isinstance(reported, dict) else 0
            )
            if check.returncode or exit_status:
                return [self._smart_query_failed(device)], False
            return [], True
        return [
            self._sample(
                "smart_health_failed",
                0 if passed else 1,
                None,
                device,
            )
        ], True

    @staticmethod
    def _smart_query_failed(device: str) -> HostMetricSample:
        """The marker for a drive whose SMART query could not be answered.

        Same metric as a failed self-assessment -- a drive that cannot be
        asked is not a drive that is known good -- with the reason in the
        labels, which the consumer carries into the finding's diagnostic
        parameters.
        """

        return HostMetricSample(
            name="smart_health_failed",
            value=1,
            device=device,
            labels={"failure_mode": "QUERY_FAILED"},
        )

    def _bmc(self, _: datetime) -> list[HostMetricSample]:
        if not shutil.which("ipmitool"):
            return []
        completed = self.runner(
            ["ipmitool", "sensor"],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
        if completed.returncode != 0:
            return []
        critical = 0
        for line in completed.stdout.splitlines():
            columns = [item.strip().lower() for item in line.split("|")]
            status = columns[2] if len(columns) > 2 else ""
            if status in {
                "cr",
                "critical",
                "nr",
                "non-recoverable",
            }:
                critical += 1
        return [self._sample("bmc_critical_sensor", critical, "sensors")]

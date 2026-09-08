from __future__ import annotations

import json
import logging
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from gpu_fault.host_health import (
    HostMetricSample,
)

LOGGER = logging.getLogger(__name__)


class HostSystemMetricsMixin:
    # Attributes supplied by the composed concrete implementation.
    filesystems: Any

    _delta: Callable[..., Any]
    _previous: Any
    _sample: Callable[..., Any]
    runner: Callable[..., Any]

    def _cpu(self, observed_at: datetime) -> list[HostMetricSample]:
        fields = [
            float(value)
            for value in Path("/proc/stat")
            .read_text(encoding="ascii")
            .splitlines()[0]
            .split()[1:]
        ]
        idle = fields[3] + (fields[4] if len(fields) > 4 else 0)
        total = sum(fields)
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

    def _filesystems(self, _: datetime) -> list[HostMetricSample]:
        result = []
        for mount in dict.fromkeys(self.filesystems):
            if not os.path.exists(mount):
                continue
            stat = os.statvfs(mount)
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

    def _shared_filesystems(self, _: datetime) -> list[HostMetricSample]:
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
                stat = os.statvfs(mount)
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

    def _smart(self, _: datetime) -> list[HostMetricSample]:
        if not shutil.which("smartctl"):
            return []
        scan = self.runner(
            ["smartctl", "--scan-open"],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
        result = []
        for line in scan.stdout.splitlines():
            fields = line.split(maxsplit=1)
            if not fields:
                continue
            device = fields[0]
            if not device.startswith("/dev/"):
                continue
            check = self.runner(
                ["smartctl", "-H", "-j", device],
                capture_output=True,
                text=True,
                timeout=20,
                check=False,
            )
            try:
                payload = json.loads(check.stdout)
                passed = payload.get("smart_status", {}).get("passed")
            except json.JSONDecodeError:
                continue
            if passed is not None:
                result.append(
                    self._sample(
                        "smart_health_failed",
                        0 if passed else 1,
                        None,
                        device,
                    )
                )
        return result

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

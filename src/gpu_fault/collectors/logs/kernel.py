from __future__ import annotations

import hashlib
import io
import logging
import os
import re
import select
import time
from collections import deque
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

from gpu_fault.channel_registry import (
    COLLECTOR_HEALTH_PATH,
    NVIDIA_KERNEL_PATH,
)

from gpu_fault.collectors.scheduling import next_stable_phase
from gpu_fault.collectors.logs.fabric_manager import (
    SXID_SUMMARY_PATTERN,
)
from gpu_fault.collectors.models import CollectorContext, CollectorStats
from gpu_fault.collectors.sinks import CollectorError, EventSink
from gpu_fault.telemetry import CollectorKind

LOGGER = logging.getLogger(__name__)

NVIDIA_EVENT_PATTERN = re.compile(r"(?:\bNVRM\b.*\bXid\b|\bSXid\b)", re.IGNORECASE)

SXID_PATTERN = re.compile(r"\bSXid\b", re.IGNORECASE)

KMSG_RECORD_PATTERN = re.compile(
    r"^(?P<priority>\d+),(?P<sequence>\d+),"
    r"(?P<monotonic_us>\d+),(?P<flags>[^;]*);"
    r"(?P<message>.*)$"
)


class KernelLogCollector:
    def __init__(
        self,
        sink: EventSink,
        context: CollectorContext,
        *,
        node_id: str,
        kmsg_path: str = "/dev/kmsg",
        boot_id: str | None = None,
        now: Callable[[], datetime] | None = None,
        deduplication_window: int = 4096,
        start_at_end: bool = True,
        reopen_delay_seconds: float = 1.0,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if reopen_delay_seconds <= 0:
            raise ValueError("kernel log reopen delay must be positive")
        self.sink = sink
        self.context = context
        self.node_id = node_id
        self.kmsg_path = kmsg_path
        self.boot_id = boot_id or self._read_boot_id()
        self.now = now or (lambda: datetime.now(timezone.utc))
        self.start_at_end = start_at_end
        self.reopen_delay_seconds = reopen_delay_seconds
        self.sleep = sleep
        self._seen_order: deque[str] = deque()
        self._seen: set[str] = set()
        self._deduplication_window = deduplication_window
        self._boot_time: datetime | None = None
        self._fallback_sequence = 0
        self.health_summary_seconds = int(
            os.getenv("GPU_FAULT_KERNEL_HEALTH_SUMMARY_SECONDS", "300")
        )
        self._next_health_summary = next_stable_phase(
            self.now(),
            cluster_id=self.context.cluster_id,
            node_id=self.node_id,
            channel=CollectorKind.NVIDIA_KERNEL.value,
            interval_seconds=self.health_summary_seconds,
        )
        self._last_read_at: datetime | None = None

    def collect_lines(
        self, lines: Iterable[str], *, limit: int | None = None
    ) -> CollectorStats:
        stats = CollectorStats()
        for line in lines:
            if limit is not None and stats.observed >= limit:
                break
            stats = stats.model_copy(update={"observed": stats.observed + 1})
            parsed = self._parse_record(line.rstrip("\n"))
            if not NVIDIA_EVENT_PATTERN.search(parsed["message"]):
                stats = stats.model_copy(update={"skipped": stats.skipped + 1})
                continue
            if SXID_PATTERN.search(
                parsed["message"]
            ) and not SXID_SUMMARY_PATTERN.search(parsed["message"]):
                stats = stats.model_copy(update={"skipped": stats.skipped + 1})
                continue
            record_id = self._record_id(parsed)
            if record_id in self._seen:
                stats = stats.model_copy(update={"duplicates": stats.duplicates + 1})
                continue
            collected_at = self.now()
            monotonic = parsed.get("monotonic_us")
            observed_at = self._observed_at(parsed, collected_at=collected_at)
            self.sink.post(
                NVIDIA_KERNEL_PATH,
                {
                    **self.context.model_dump(mode="json"),
                    "node_id": self.node_id,
                    "record_id": record_id,
                    "observed_at": observed_at.isoformat(),
                    "source_monotonic_us": (int(monotonic) if monotonic else None),
                    "source_boot_id": self.boot_id,
                    "collected_at": collected_at.isoformat(),
                    "message": parsed["message"],
                    "evidence_ref": (
                        f"kmsg://{self.node_id}/{self.boot_id}/"
                        f"{parsed.get('sequence') or record_id}"
                    ),
                },
            )
            self._remember(record_id)
            stats = stats.model_copy(update={"delivered": stats.delivered + 1})
        return stats

    def run(self) -> None:
        while True:
            try:
                with open(
                    self.kmsg_path,
                    "r",
                    encoding="utf-8",
                    errors="replace",
                    buffering=1,
                ) as stream:
                    if self.start_at_end:
                        try:
                            self._seek_to_live_tail(stream)
                        except OSError as exc:
                            raise CollectorError(
                                "cannot seek kernel message device to "
                                "the live tail; refusing to replay the "
                                "ring buffer"
                            ) from exc
                    self._boot_time = self._estimate_boot_time()
                    self._collect_live_stream(stream)
                    LOGGER.warning(
                        "kernel message stream ended; reopening %s",
                        self.kmsg_path,
                    )
            except PermissionError as exc:
                raise CollectorError(
                    f"cannot read {self.kmsg_path}; CAP_SYSLOG or an "
                    "equivalent privileged device mount is required"
                ) from exc
            except FileNotFoundError as exc:
                raise CollectorError(
                    f"kernel message device not found: {self.kmsg_path}"
                ) from exc
            except Exception:
                LOGGER.exception(
                    "kernel log collection failed; reopening %s",
                    self.kmsg_path,
                )
            self.sleep(self.reopen_delay_seconds)

    def _collect_live_stream(self, stream: Any) -> None:
        try:
            descriptor = stream.fileno()
        except (AttributeError, io.UnsupportedOperation):
            self.collect_lines(stream)
            self._last_read_at = self.now()
            self._maybe_health_summary_without_interrupting_stream(self._last_read_at)
            return
        while True:
            now = self.now()
            timeout = min(
                1.0,
                max(
                    0.0,
                    (self._next_health_summary - now).total_seconds(),
                ),
            )
            ready, _, _ = select.select([descriptor], [], [], timeout)
            self._last_read_at = self.now()
            if ready:
                line = stream.readline()
                if not line:
                    return
                self.collect_lines([line])
            self._maybe_health_summary_without_interrupting_stream(self._last_read_at)

    def _maybe_health_summary_without_interrupting_stream(
        self,
        observed_at: datetime,
    ) -> None:
        try:
            self._maybe_health_summary(observed_at)
        except CollectorError as exc:
            LOGGER.warning(
                "kernel health summary delivery failed; continuing live stream: %s",
                exc,
            )

    def _maybe_health_summary(self, observed_at: datetime) -> None:
        if observed_at < self._next_health_summary:
            return
        self._next_health_summary = next_stable_phase(
            observed_at,
            cluster_id=self.context.cluster_id,
            node_id=self.node_id,
            channel=CollectorKind.NVIDIA_KERNEL.value,
            interval_seconds=self.health_summary_seconds,
        )
        self._send_health_summary(observed_at)

    def _send_health_summary(self, observed_at: datetime) -> None:
        self.sink.post(
            COLLECTOR_HEALTH_PATH,
            {
                "summary_id": (
                    f"kernel-health-{self.node_id}-{int(observed_at.timestamp())}"
                ),
                "cluster_id": self.context.cluster_id,
                "node_id": self.node_id,
                "collector": CollectorKind.NVIDIA_KERNEL.value,
                "observed_at": observed_at.isoformat(),
                "edge_filter_reasons": ["health-summary"],
            },
        )

    @staticmethod
    def _seek_to_live_tail(stream: Any) -> None:
        try:
            descriptor = stream.fileno()
        except (AttributeError, io.UnsupportedOperation):
            # StringIO and other test streams have no file descriptor.
            stream.seek(0, os.SEEK_END)
            return
        os.lseek(descriptor, 0, os.SEEK_END)

    @staticmethod
    def _parse_record(line: str) -> dict[str, str | None]:
        match = KMSG_RECORD_PATTERN.match(line)
        if not match:
            return {
                "priority": None,
                "sequence": None,
                "monotonic_us": None,
                "flags": None,
                "message": line,
            }
        return match.groupdict()

    def _record_id(self, parsed: dict[str, str | None]) -> str:
        sequence = parsed.get("sequence")
        if sequence:
            suffix = sequence
        elif parsed.get("monotonic_us"):
            digest = hashlib.sha256((parsed.get("message") or "").encode()).hexdigest()[
                :12
            ]
            suffix = f"{parsed['monotonic_us']}-{digest}"
        else:
            self._fallback_sequence += 1
            digest = hashlib.sha256((parsed.get("message") or "").encode()).hexdigest()[
                :12
            ]
            suffix = f"fallback-{self._fallback_sequence}-{digest}"
        return f"kmsg-{self.boot_id}-{suffix}"

    def _observed_at(
        self,
        parsed: dict[str, str | None],
        *,
        collected_at: datetime,
    ) -> datetime:
        monotonic = parsed.get("monotonic_us")
        if monotonic is None or self._boot_time is None:
            return collected_at
        try:
            observed_at = self._boot_time + timedelta(microseconds=int(monotonic))
        except (TypeError, ValueError, OverflowError):
            return collected_at
        if observed_at > collected_at + timedelta(seconds=5):
            LOGGER.warning(
                "kernel event monotonic timestamp is in the future; "
                "using collection time record=%s",
                parsed.get("sequence"),
            )
            return collected_at
        return observed_at

    def _estimate_boot_time(self) -> datetime | None:
        try:
            uptime = float(
                Path("/proc/uptime").read_text(encoding="ascii").split(maxsplit=1)[0]
            )
        except (OSError, ValueError, IndexError):
            LOGGER.warning(
                "cannot read host uptime; kernel event timestamps "
                "will use collection time"
            )
            return None
        return self.now() - timedelta(seconds=uptime)

    def _remember(self, record_id: str) -> None:
        self._seen.add(record_id)
        self._seen_order.append(record_id)
        while len(self._seen_order) > self._deduplication_window:
            self._seen.discard(self._seen_order.popleft())

    @staticmethod
    def _read_boot_id() -> str:
        try:
            return (
                Path("/proc/sys/kernel/random/boot_id")
                .read_text(encoding="ascii")
                .strip()
            )
        except OSError:
            return "unknown-boot"

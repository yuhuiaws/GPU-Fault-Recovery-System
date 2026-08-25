from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import subprocess
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

from gpu_fault.channel_registry import NODE_LOG_PATH
from gpu_fault.collectors.scheduling import next_stable_phase
from gpu_fault.host_health import (
    NodeLogBatch,
    NodeLogEntry,
)
from gpu_fault.log_rules import (
    log_signal_priority,
    matching_log_rules,
)


from gpu_fault.collectors.models import CollectorContext
from gpu_fault.collectors.sinks import EventSink

LOGGER = logging.getLogger(__name__)


class NodeLogCollector:
    """Polls journald/dmesg and configured training logs with stable IDs."""

    def __init__(
        self,
        sink: EventSink,
        context: CollectorContext,
        *,
        node_id: str,
        interval_seconds: float = 10,
        training_log_paths: list[str] | None = None,
        state_path: str | None = None,
        initial_tail_bytes: int | None = None,
        max_entries_per_batch: int | None = None,
        max_batch_bytes: int | None = None,
        max_entry_bytes: int | None = None,
        now: Callable[[], datetime] | None = None,
        runner: Callable[..., subprocess.CompletedProcess[str]] = (subprocess.run),
    ) -> None:
        self.sink = sink
        self.context = context
        self.node_id = node_id
        self.interval_seconds = interval_seconds
        self.training_log_paths = training_log_paths or []
        self.now = now or (lambda: datetime.now(timezone.utc))
        self.runner = runner
        self._journal_since: datetime | None = None
        self._offsets: dict[str, int] = {}
        self.state_path = Path(state_path) if state_path else None
        self.initial_tail_bytes = (
            initial_tail_bytes
            if initial_tail_bytes is not None
            else int(os.getenv("GPU_FAULT_NODE_LOG_INITIAL_TAIL_BYTES", "0"))
        )
        if self.initial_tail_bytes < 0:
            raise ValueError("node log initial tail bytes must be non-negative")
        self.max_entries_per_batch = (
            max_entries_per_batch
            if max_entries_per_batch is not None
            else int(
                os.getenv(
                    "GPU_FAULT_NODE_LOG_MAX_ENTRIES_PER_BATCH",
                    "1000",
                )
            )
        )
        self.max_batch_bytes = (
            max_batch_bytes
            if max_batch_bytes is not None
            else int(
                os.getenv(
                    "GPU_FAULT_NODE_LOG_MAX_BATCH_BYTES",
                    str(4 * 1024 * 1024),
                )
            )
        )
        self.max_entry_bytes = (
            max_entry_bytes
            if max_entry_bytes is not None
            else int(
                os.getenv(
                    "GPU_FAULT_NODE_LOG_MAX_ENTRY_BYTES",
                    str(64 * 1024),
                )
            )
        )
        if (
            self.max_entries_per_batch < 1
            or self.max_batch_bytes < 1024
            or self.max_entry_bytes < 256
            or self.max_entry_bytes > self.max_batch_bytes
        ):
            raise ValueError("invalid node log batch limits")
        self._load_state()
        self.health_summary_seconds = int(
            os.getenv("GPU_FAULT_NODE_LOG_HEALTH_SUMMARY_SECONDS", "300")
        )
        self._next_health_summary = next_stable_phase(
            self.now(),
            cluster_id=self.context.cluster_id,
            node_id=self.node_id,
            channel="NODE_LOGS",
            interval_seconds=self.health_summary_seconds,
        )

    def collect_once(self) -> NodeLogBatch:
        collected_at = self.now()
        previous_since = self._journal_since
        previous_offsets = dict(self._offsets)
        try:
            entries = [
                entry
                for entry in self._journal(collected_at)
                if matching_log_rules(entry.message)
            ]
            entries.extend(
                entry
                for entry in self._training_logs(
                    collected_at,
                    max_entries=self.max_entries_per_batch,
                    max_bytes=self.max_batch_bytes,
                )
                if matching_log_rules(entry.message)
            )
            entries = self._limit_entries(
                entries,
                max_entries=self.max_entries_per_batch,
                max_bytes=self.max_batch_bytes,
            )
        except Exception:
            self._journal_since = previous_since
            self._offsets = previous_offsets
            raise
        health_summary = not entries and collected_at >= self._next_health_summary
        batch = NodeLogBatch(
            batch_id=(
                f"logs-{self.node_id}-{int(collected_at.timestamp() * 1_000_000)}"
            ),
            cluster_id=self.context.cluster_id,
            node_id=self.node_id,
            collected_at=collected_at,
            entries=entries,
            runtime_profile_version=(self.context.runtime_profile_version),
            workload_state=self.context.workload_state,
            affected_workload_ids=(self.context.affected_workload_ids),
            edge_filter_reasons=(
                ["candidate-confirmed"]
                if entries
                else ["health-summary"]
                if health_summary
                else []
            ),
        )
        try:
            if entries or health_summary:
                self.sink.post(
                    NODE_LOG_PATH,
                    batch.model_dump(mode="json"),
                )
            if health_summary:
                self._next_health_summary = next_stable_phase(
                    collected_at,
                    cluster_id=self.context.cluster_id,
                    node_id=self.node_id,
                    channel="NODE_LOGS",
                    interval_seconds=self.health_summary_seconds,
                )
            self._journal_since = collected_at
            self._save_state()
        except Exception:
            self._journal_since = previous_since
            self._offsets = previous_offsets
            raise
        return batch

    def run(self) -> None:
        while True:
            try:
                self.collect_once()
            except Exception:
                LOGGER.exception("node log collection failed")
            time.sleep(self.interval_seconds)

    def _journal(self, until: datetime) -> list[NodeLogEntry]:
        if not shutil.which("journalctl"):
            return []
        since = self._journal_since or (until - timedelta(minutes=5))
        completed = self.runner(
            [
                "journalctl",
                "--since",
                f"@{since.timestamp()}",
                "--until",
                f"@{until.timestamp()}",
                "--output=json",
                "--no-pager",
                f"--lines={self.max_entries_per_batch}",
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        entries = []
        for line in completed.stdout.splitlines():
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            message = self._bounded_message(str(item.get("MESSAGE") or ""))
            cursor = str(
                item.get("__CURSOR") or hashlib.sha256(line.encode()).hexdigest()
            )
            timestamp = datetime.fromtimestamp(
                int(item.get("__REALTIME_TIMESTAMP", "0")) / 1_000_000,
                tz=timezone.utc,
            )
            transport = str(item.get("_TRANSPORT") or "journal")
            entries.append(
                NodeLogEntry(
                    entry_id=cursor,
                    source=("dmesg" if transport == "kernel" else "journal"),
                    observed_at=timestamp,
                    message=message,
                    priority=(
                        int(item["PRIORITY"])
                        if str(item.get("PRIORITY", "")).isdigit()
                        else None
                    ),
                    unit=item.get("_SYSTEMD_UNIT"),
                )
            )
        return entries

    def _bounded_message(self, message: str) -> str:
        raw = message.encode("utf-8", errors="replace")
        if len(raw) <= self.max_entry_bytes:
            return message
        marker = b"\n...[node-log-entry-truncated]...\n"
        available = max(0, self.max_entry_bytes - len(marker))
        prefix = available // 2
        suffix = available - prefix
        bounded = raw[:prefix] + marker + raw[-suffix:]
        return bounded.decode("utf-8", errors="replace")

    @staticmethod
    def _entry_size(entry: NodeLogEntry) -> int:
        return len(entry.model_dump_json().encode("utf-8"))

    def _limit_entries(
        self,
        entries: list[NodeLogEntry],
        *,
        max_entries: int,
        max_bytes: int,
    ) -> list[NodeLogEntry]:
        selected = []
        used = 0
        ordered = sorted(
            entries,
            key=lambda entry: (
                log_signal_priority(entry.message),
                entry.observed_at,
            ),
            reverse=True,
        )
        for entry in ordered:
            size = self._entry_size(entry)
            if len(selected) >= max_entries:
                break
            if used + size > max_bytes:
                continue
            selected.append(entry)
            used += size
        return sorted(selected, key=lambda entry: entry.observed_at)

    def _load_state(self) -> None:
        if self.state_path is None or not self.state_path.exists():
            return
        try:
            value = json.loads(self.state_path.read_text())
            since = value.get("journal_since")
            self._journal_since = datetime.fromisoformat(since) if since else None
            self._offsets = {
                str(key): int(offset)
                for key, offset in (value.get("training_log_offsets") or {}).items()
                if int(offset) >= 0
            }
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            LOGGER.exception("cannot load node log collector state")
            self._journal_since = None
            self._offsets = {}

    def _save_state(self) -> None:
        if self.state_path is None:
            return
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.state_path.with_suffix(self.state_path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(
                {
                    "journal_since": (
                        self._journal_since.isoformat() if self._journal_since else None
                    ),
                    "training_log_offsets": self._offsets,
                },
                separators=(",", ":"),
            ),
            encoding="utf-8",
        )
        os.replace(temporary, self.state_path)

    def _training_logs(
        self,
        observed_at: datetime,
        *,
        max_entries: int | None = None,
        max_bytes: int | None = None,
    ) -> list[NodeLogEntry]:
        entries = []
        entry_limit = self.max_entries_per_batch if max_entries is None else max_entries
        byte_limit = self.max_batch_bytes if max_bytes is None else max_bytes
        used_bytes = 0
        if entry_limit <= 0 or byte_limit <= 0:
            return entries
        for configured in self.training_log_paths:
            for path in sorted(Path("/").glob(configured.lstrip("/"))):
                if not path.is_file():
                    continue
                key = str(path)
                size = path.stat().st_size
                offset = min(
                    self._offsets.get(
                        key,
                        max(0, size - self.initial_tail_bytes),
                    ),
                    size,
                )
                with path.open("r", encoding="utf-8", errors="replace") as stream:
                    stream.seek(offset)
                    while True:
                        start = stream.tell()
                        line = stream.readline()
                        if not line:
                            break
                        entry_id = hashlib.sha256(f"{key}:{start}".encode()).hexdigest()
                        entry = NodeLogEntry(
                            entry_id=entry_id,
                            source="training-log",
                            observed_at=observed_at,
                            message=self._bounded_message(line.rstrip("\n")),
                            fields={"path": key},
                        )
                        entry_size = self._entry_size(entry)
                        if (
                            len(entries) >= entry_limit
                            or used_bytes + entry_size > byte_limit
                        ):
                            stream.seek(start)
                            break
                        entries.append(entry)
                        used_bytes += entry_size
                    self._offsets[key] = stream.tell()
                if len(entries) >= entry_limit or used_bytes >= byte_limit:
                    return entries
        return entries

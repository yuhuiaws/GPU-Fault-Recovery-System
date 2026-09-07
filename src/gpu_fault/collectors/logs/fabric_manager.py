from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

from gpu_fault.channel_registry import (
    COLLECTOR_HEALTH_PATH,
    FABRIC_MANAGER_PATH,
)

from gpu_fault.collectors.models import CollectorContext, CollectorStats
from gpu_fault.collectors.scheduling import next_stable_phase
from gpu_fault.collectors.sinks import CollectorError, EventSink
from gpu_fault.env import env_bool
from gpu_fault.telemetry import CollectorKind

LOGGER = logging.getLogger(__name__)

SXID_SUMMARY_PATTERN = re.compile(
    r"\bSXid\b\s*\(PCI:[0-9a-fA-F:.]+\)\s*:\s*\d+\s*,\s*"
    r"(?:Non-fatal|Nonfatal|Fatal)\b",
    re.IGNORECASE,
)

FABRIC_MANAGER_SXID_PATTERN = SXID_SUMMARY_PATTERN


class FabricManagerLogCollector:
    """Collects SXIDs from Fabric Manager journald and file outputs."""

    def __init__(
        self,
        sink: EventSink,
        context: CollectorContext,
        *,
        node_id: str,
        interval_seconds: float = 5,
        journal_enabled: bool = True,
        journal_identifiers: tuple[str, ...] = (
            "nvidia-fabricmanager",
            "nv-fabricmanager",
        ),
        log_paths: list[str] | None = None,
        state_path: str | None = None,
        now: Callable[[], datetime] | None = None,
        runner: Callable[..., subprocess.CompletedProcess[str]] = (subprocess.run),
    ) -> None:
        self.sink = sink
        self.context = context
        self.node_id = node_id
        self.interval_seconds = interval_seconds
        self.journal_enabled = journal_enabled
        self.journal_identifiers = tuple(
            item.lower() for item in journal_identifiers if item
        )
        self.log_paths = log_paths or []
        self.state_path = Path(state_path) if state_path else None
        self.now = now or (lambda: datetime.now(timezone.utc))
        self.runner = runner
        self._journal_cursor: str | None = None
        self._files: dict[str, dict[str, int]] = {}
        self._state_dirty = False
        self.health_summary_seconds = int(
            os.getenv(
                "GPU_FAULT_FABRIC_MANAGER_HEALTH_SUMMARY_SECONDS",
                "300",
            )
        )
        self._next_health_summary = next_stable_phase(
            self.now(),
            cluster_id=self.context.cluster_id,
            node_id=self.node_id,
            channel=CollectorKind.FABRIC_MANAGER_LOG.value,
            interval_seconds=self.health_summary_seconds,
        )
        self._load_state()

    def collect_once(self) -> CollectorStats:
        collected_at = self.now()
        try:
            records = [
                *self._journal_records(collected_at),
                *self._file_records(collected_at),
            ]
        except Exception:
            self._flush_state()
            raise
        stats = CollectorStats()
        try:
            for record in records:
                eligible = bool(record.pop("_eligible", True))
                if eligible:
                    stats = stats.model_copy(update={"observed": stats.observed + 1})
                if not eligible or not FABRIC_MANAGER_SXID_PATTERN.search(
                    record.get("message", "")
                ):
                    if eligible:
                        stats = stats.model_copy(update={"skipped": stats.skipped + 1})
                    self._commit_record(record)
                    continue
                checkpoint = record.pop("_checkpoint", None)
                try:
                    self.sink.post(
                        FABRIC_MANAGER_PATH,
                        {
                            **self.context.model_dump(mode="json"),
                            "node_id": self.node_id,
                            **record,
                            "collected_at": collected_at.isoformat(),
                        },
                    )
                except Exception:
                    if checkpoint is not None:
                        record["_checkpoint"] = checkpoint
                    raise
                stats = stats.model_copy(update={"delivered": stats.delivered + 1})
                if checkpoint is not None:
                    record["_checkpoint"] = checkpoint
                self._commit_record(record)
        finally:
            self._flush_state()
        self._maybe_health_summary(collected_at)
        return stats

    def _maybe_health_summary(self, observed_at: datetime) -> None:
        if observed_at < self._next_health_summary:
            return
        self._next_health_summary = next_stable_phase(
            observed_at,
            cluster_id=self.context.cluster_id,
            node_id=self.node_id,
            channel=CollectorKind.FABRIC_MANAGER_LOG.value,
            interval_seconds=self.health_summary_seconds,
        )
        self.sink.post(
            COLLECTOR_HEALTH_PATH,
            {
                "summary_id": (
                    f"fabric-health-{self.node_id}-{int(observed_at.timestamp())}"
                ),
                "cluster_id": self.context.cluster_id,
                "node_id": self.node_id,
                "collector": CollectorKind.FABRIC_MANAGER_LOG.value,
                "observed_at": observed_at.isoformat(),
                "edge_filter_reasons": ["health-summary"],
            },
        )

    def run(self) -> None:
        while True:
            try:
                self.collect_once()
            except Exception:
                LOGGER.exception("Fabric Manager log collection failed")
            time.sleep(self.interval_seconds)

    def _journal_records(self, collected_at: datetime) -> list[dict[str, Any]]:
        if not self.journal_enabled or not shutil.which("journalctl"):
            return []
        completed = self._run_journal(collected_at, cursor=self._journal_cursor)
        if (
            completed.returncode != 0
            and self._journal_cursor
            and self._is_cursor_rejection(completed.stderr)
        ):
            LOGGER.warning(
                "Fabric Manager journal cursor is no longer "
                "available; resuming from the recent window"
            )
            self._journal_cursor = None
            self._state_dirty = True
            completed = self._run_journal(collected_at, cursor=None)
        if completed.returncode != 0:
            raise CollectorError(
                "Fabric Manager journal query failed: "
                + (completed.stderr.strip() or "unknown error")
            )
        records = []
        for line in completed.stdout.splitlines():
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            cursor = str(item.get("__CURSOR") or "")
            unit = str(item.get("_SYSTEMD_UNIT") or "")
            identifier = str(item.get("SYSLOG_IDENTIFIER") or item.get("_COMM") or "")
            eligible = self._is_fabric_manager(unit, identifier)
            message = str(item.get("MESSAGE") or "")
            timestamp = self._journal_timestamp(item, collected_at)
            stable = cursor or hashlib.sha256(line.encode()).hexdigest()
            records.append(
                {
                    "record_id": f"fm-journal-{stable}",
                    "observed_at": timestamp.isoformat(),
                    "message": message,
                    "source": "journal",
                    "_eligible": eligible,
                    "_checkpoint": {
                        "kind": "journal",
                        "cursor": cursor,
                    },
                    "unit": unit or None,
                    "fields": {
                        "journal_cursor": cursor,
                        "syslog_identifier": identifier,
                    },
                    "evidence_ref": (f"journal://{self.node_id}/{stable}"),
                }
            )
        return records

    def _run_journal(
        self, collected_at: datetime, *, cursor: str | None
    ) -> subprocess.CompletedProcess[str]:
        command = [
            "journalctl",
            "--output=json",
            "--no-pager",
            "--until",
            f"@{collected_at.timestamp()}",
        ]
        if cursor:
            command.extend(["--after-cursor", cursor])
        else:
            command.extend(
                [
                    "--since",
                    f"@{(collected_at - timedelta(minutes=5)).timestamp()}",
                ]
            )
        command.extend(self._journal_matches())
        return self.runner(
            command,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )

    def _journal_matches(self) -> list[str]:
        """Field matches restricting journald reads to Fabric Manager.

        journald ANDs matches inside a group and ORs the groups
        separated by ``+``; every value below is therefore its own
        group. ``_COMM`` is truncated to 15 characters by the kernel,
        so the comparison value has to be truncated as well.
        """

        values: list[str] = []
        for identifier in self.journal_identifiers:
            for value in (
                f"_SYSTEMD_UNIT={identifier}.service",
                f"SYSLOG_IDENTIFIER={identifier}",
                f"_COMM={identifier[:15]}",
            ):
                if value not in values:
                    values.append(value)
        matches: list[str] = []
        for value in values:
            if matches:
                matches.append("+")
            matches.append(value)
        return matches

    @staticmethod
    def _is_cursor_rejection(stderr: str | None) -> bool:
        text = (stderr or "").lower()
        return "cursor" in text and (
            "seek" in text or "invalid" in text or "not found" in text
        )

    def _file_records(self, collected_at: datetime) -> list[dict[str, Any]]:
        records = []
        for configured in self.log_paths:
            for path in sorted(Path("/").glob(configured.lstrip("/"))):
                if not path.is_file():
                    continue
                stat = path.stat()
                key = str(path)
                previous = self._files.get(key)
                identity_changed = previous is not None and (
                    previous.get("device") != stat.st_dev
                    or previous.get("inode") != stat.st_ino
                )
                truncated = previous is not None and stat.st_size < previous.get(
                    "offset", 0
                )
                if previous is None:
                    # Historical file content has no trustworthy node-generation
                    # boundary when the durable checkpoint is absent.
                    self._files[key] = {
                        "device": stat.st_dev,
                        "inode": stat.st_ino,
                        "offset": stat.st_size,
                    }
                    self._state_dirty = True
                    continue
                elif identity_changed or truncated:
                    offset = 0
                else:
                    offset = min(previous.get("offset", 0), stat.st_size)
                with path.open("r", encoding="utf-8", errors="replace") as stream:
                    if offset > 0:
                        stream.seek(offset - 1)
                        previous_character = stream.read(1)
                        stream.seek(offset)
                        if previous_character != "\n":
                            stream.readline()
                            records.append(
                                {
                                    "message": "",
                                    "_eligible": False,
                                    "_checkpoint": {
                                        "kind": "file",
                                        "path": key,
                                        "device": stat.st_dev,
                                        "inode": stat.st_ino,
                                        "offset": stream.tell(),
                                    },
                                }
                            )
                    while True:
                        start = stream.tell()
                        line = stream.readline()
                        if not line:
                            break
                        if not line.endswith("\n"):
                            stream.seek(start)
                            break
                        stable = f"{stat.st_dev}:{stat.st_ino}:{start}"
                        records.append(
                            {
                                "record_id": (
                                    "fm-file-"
                                    + hashlib.sha256(stable.encode()).hexdigest()
                                ),
                                "observed_at": self._file_timestamp(
                                    line, collected_at
                                ).isoformat(),
                                "message": line.rstrip("\n"),
                                "source": "file",
                                "_eligible": True,
                                "_checkpoint": {
                                    "kind": "file",
                                    "path": key,
                                    "device": stat.st_dev,
                                    "inode": stat.st_ino,
                                    "offset": stream.tell(),
                                },
                                "fields": {
                                    "path": key,
                                    "device": str(stat.st_dev),
                                    "inode": str(stat.st_ino),
                                    "offset": str(start),
                                },
                                "evidence_ref": (f"file://{key}#{stable}"),
                            }
                        )
        return records

    def _is_fabric_manager(self, unit: str, identifier: str) -> bool:
        values = {
            unit.lower().removesuffix(".service"),
            identifier.lower(),
        }
        return any(
            value == expected
            or value.startswith(expected + "-")
            or value.startswith(expected + "@")
            for value in values
            for expected in self.journal_identifiers
        )

    @staticmethod
    def _file_timestamp(line: str, fallback: datetime) -> datetime:
        match = re.search(
            r"(?<!\d)(\d{4}-\d{2}-\d{2}[T ]"
            r"\d{2}:\d{2}:\d{2}(?:\.\d+)?"
            r"(?:Z|[+-]\d{2}:?\d{2})?)",
            line,
        )
        if match is None:
            return fallback
        raw = match.group(1).replace(" ", "T")
        if raw.endswith("Z"):
            raw = raw[:-1] + "+00:00"
        if re.search(r"[+-]\d{4}$", raw):
            raw = raw[:-5] + raw[-5:-2] + ":" + raw[-2:]
        try:
            parsed = datetime.fromisoformat(raw)
        except ValueError:
            return fallback
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)

    @staticmethod
    def _journal_timestamp(item: dict[str, Any], fallback: datetime) -> datetime:
        try:
            return datetime.fromtimestamp(
                int(item["__REALTIME_TIMESTAMP"]) / 1_000_000,
                tz=timezone.utc,
            )
        except (KeyError, TypeError, ValueError, OSError):
            return fallback

    def _load_state(self) -> None:
        if self.state_path is None or not self.state_path.exists():
            return
        try:
            value = json.loads(self.state_path.read_text(encoding="utf-8"))
            self._journal_cursor = value.get("journal_cursor")
            raw_files = value.get("files") or {}
            self._files = {
                str(path): {
                    "device": int(state["device"]),
                    "inode": int(state["inode"]),
                    "offset": int(state["offset"]),
                }
                for path, state in raw_files.items()
                if int(state["offset"]) >= 0
            }
        except (
            KeyError,
            OSError,
            ValueError,
            TypeError,
            json.JSONDecodeError,
        ):
            LOGGER.exception("cannot load Fabric Manager collector state")
            self._journal_cursor = None
            self._files = {}

    def _save_state(self) -> None:
        if self.state_path is None:
            return
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.state_path.with_suffix(self.state_path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as stream:
            stream.write(
                json.dumps(
                    {
                        "journal_cursor": self._journal_cursor,
                        "files": self._files,
                    },
                    separators=(",", ":"),
                )
            )
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, self.state_path)
        directory_fd = os.open(
            self.state_path.parent,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
        )
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)

    def _commit_record(self, record: dict[str, Any]) -> None:
        checkpoint = record.pop("_checkpoint", None)
        if not checkpoint:
            return
        if checkpoint["kind"] == "journal":
            cursor = str(checkpoint.get("cursor") or "")
            if cursor:
                self._journal_cursor = cursor
        elif checkpoint["kind"] == "file":
            self._files[str(checkpoint["path"])] = {
                "device": int(checkpoint["device"]),
                "inode": int(checkpoint["inode"]),
                "offset": int(checkpoint["offset"]),
            }
        else:
            raise ValueError(
                "unknown Fabric Manager checkpoint kind: " + str(checkpoint["kind"])
            )
        self._state_dirty = True

    def _flush_state(self) -> None:
        """Persist checkpoints once per batch instead of per record.

        Every ``_save_state`` costs two fsyncs on the shared node
        volume, so the durability point is moved to the end of the
        batch. Redelivery after a crash stays bounded by the batch,
        and the sink is idempotent on ``record_id``.
        """

        if not self._state_dirty:
            return
        self._save_state()
        self._state_dirty = False


def build_from_environment(
    sink: EventSink, context: CollectorContext, arguments: argparse.Namespace
) -> FabricManagerLogCollector:
    """The ``gpu-fault-collector fabric-manager`` factory named by the registry."""

    if not arguments.node_id:
        raise SystemExit("--node-id, NODE_NAME, or HOSTNAME is required")
    return FabricManagerLogCollector(
        sink,
        context,
        node_id=arguments.node_id,
        interval_seconds=arguments.interval_seconds,
        journal_enabled=env_bool("GPU_FAULT_FABRIC_MANAGER_JOURNAL", True),
        journal_identifiers=tuple(
            item
            for item in os.getenv(
                "GPU_FAULT_FABRIC_MANAGER_IDENTIFIERS",
                "nvidia-fabricmanager,nv-fabricmanager",
            ).split(",")
            if item
        ),
        log_paths=[
            item
            for item in os.getenv("GPU_FAULT_FABRIC_MANAGER_LOG_PATHS", "").split(",")
            if item
        ],
        state_path=os.getenv(
            "GPU_FAULT_FABRIC_MANAGER_STATE_PATH",
            "/var/lib/gpu-fault/fabric-manager-collector-state.json",
        ),
    )

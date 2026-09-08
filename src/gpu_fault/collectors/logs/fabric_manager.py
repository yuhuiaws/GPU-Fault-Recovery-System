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
from gpu_fault.collectors.sinks import CollectorError, EventSink, deliver_event
from gpu_fault.env import env_bool
from gpu_fault.telemetry import CollectorKind

LOGGER = logging.getLogger(__name__)

#: How many log files the collector keeps offsets for. A glob over rotated
#: files grows the persisted table forever otherwise (ARCH-G9).
DEFAULT_MAX_TRACKED_FILES = 256

SXID_SUMMARY_PATTERN = re.compile(
    r"\bSXid\b\s*\(PCI:[0-9a-fA-F:.]+\)\s*:\s*\d+\s*,\s*"
    r"(?:Non-fatal|Nonfatal|Fatal)\b",
    re.IGNORECASE,
)

FABRIC_MANAGER_SXID_PATTERN = SXID_SUMMARY_PATTERN


def _read_boot_id() -> str:
    """The kernel's boot id, or a marker that says it could not be read.

    It scopes every file record id this collector mints, the same way
    ``node.py`` scopes the ids of its training-log lines: a reused inode after
    a reboot must not restart the offsets under ids that were already used. The
    marker keeps ids unique per node and per file, it only stops distinguishing
    boots, so a missing ``/proc`` must not stop log collection.
    """

    try:
        return (
            Path("/proc/sys/kernel/random/boot_id").read_text(encoding="ascii").strip()
            or "unknown-boot"
        )
    except OSError:
        return "unknown-boot"


class FabricManagerLogCollector:
    """Collects SXIDs from Fabric Manager journald and file outputs."""

    def __init__(
        self,
        sink: EventSink,
        context: CollectorContext,
        *,
        node_id: str,
        boot_id: str | None = None,
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
        max_tracked_files: int = DEFAULT_MAX_TRACKED_FILES,
    ) -> None:
        if max_tracked_files < 1:
            raise ValueError("Fabric Manager tracked file limit must be at least 1")
        self.sink = sink
        self.context = context
        self.node_id = node_id
        self.boot_id = boot_id or _read_boot_id()
        self.interval_seconds = interval_seconds
        self.max_tracked_files = max_tracked_files
        self._files_bound_warned = False
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
        self._resync_warned = False
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
                    result = deliver_event(
                        self.sink,
                        FABRIC_MANAGER_PATH,
                        {
                            **self.context.model_dump(mode="json"),
                            "node_id": self.node_id,
                            **record,
                            "collected_at": collected_at.isoformat(),
                        },
                    )
                    # Only a record that went nowhere pins the checkpoint; one
                    # the outbox took is replayed from there (ARCH-G3).
                    result.raise_for_failure()
                except Exception:
                    if checkpoint is not None:
                        record["_checkpoint"] = checkpoint
                    raise
                if result.buffered:
                    LOGGER.warning(
                        "Fabric Manager record persisted to the collector "
                        "outbox; advancing the checkpoint: record=%s error=%s",
                        record.get("record_id"),
                        result.error,
                    )
                else:
                    stats = stats.model_copy(update={"delivered": stats.delivered + 1})
                if checkpoint is not None:
                    record["_checkpoint"] = checkpoint
                self._commit_record(record)
        finally:
            self._flush_state()
        self._deliver_health_summary_without_failing_the_round(collected_at)
        return stats

    def _deliver_health_summary_without_failing_the_round(
        self, observed_at: datetime
    ) -> None:
        """A lost health summary is not a failed collection round.

        The summary used to be posted with a bare ``sink.post`` whose
        ``CollectorError`` escaped ``collect_once``, so the run loop logged
        "Fabric Manager log collection failed" for a round in which every SXID
        committed and the summary schedule had already advanced. The kernel
        collector answers the same way for its own summary.
        """

        try:
            self._maybe_health_summary(observed_at)
        except CollectorError as exc:
            LOGGER.warning(
                "Fabric Manager health summary delivery failed; "
                "the collection round is unaffected: %s",
                exc,
            )

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
        result = deliver_event(
            self.sink,
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
        # A summary the outbox took is on its way; only one that went nowhere
        # is reported (ARCH-G3).
        result.raise_for_failure()

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
            # Without --all journalctl encodes any field over 4096 bytes as a
            # null value, so a long Fabric Manager line arrived as MESSAGE:
            # null, matched no SXID pattern and was dropped without a trace.
            # The collector bounds its own records.
            "--all",
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
        """Every full line appended to the configured logs since the last poll.

        The three steps are kept apart on purpose: what exists now, where each
        file resumes, and only then what it says. Resolving every offset before
        any read is what lets a rotated inode take its predecessor's checkpoint
        whatever order the glob returns the two names in.
        """

        live = self._live_files()
        records: list[dict[str, Any]] = []
        for key, stat, offset in self._resume_offsets(live):
            try:
                records.extend(self._records_from_file(key, stat, offset, collected_at))
            except OSError as exc:
                # A 0600 file under ProtectSystem=strict, or one that vanished
                # between the stat and the open. The read used to raise out of
                # the round and take the journal SXIDs of that round with it,
                # every round, permanently (ARCH-G4).
                LOGGER.warning(
                    "Fabric Manager log file unreadable this round: %s (%s)",
                    key,
                    exc,
                )
                continue
        self._bound_tracked_files({key for key, _stat in live})
        return records

    def _live_files(self) -> list[tuple[str, os.stat_result]]:
        """The configured logs that exist and are readable enough to stat."""

        live: list[tuple[str, os.stat_result]] = []
        seen: set[str] = set()
        for configured in self.log_paths:
            for path in sorted(Path("/").glob(configured.lstrip("/"))):
                key = str(path)
                if key in seen:
                    continue
                try:
                    if not path.is_file():
                        continue
                    stat = path.stat()
                except OSError as exc:
                    # Rotated away between the glob and the stat: the next
                    # round sees the successor. One vanished sibling must
                    # not abort the whole round (ARCH-G9).
                    LOGGER.warning(
                        "Fabric Manager log file unreadable this round: %s (%s)",
                        path,
                        exc,
                    )
                    continue
                seen.add(key)
                live.append((key, stat))
        return live

    def _resume_offsets(
        self, live: list[tuple[str, os.stat_result]]
    ) -> list[tuple[str, os.stat_result, int]]:
        plan: list[tuple[str, os.stat_result, int]] = []
        for key, stat in live:
            previous = self._files.get(key)
            if previous is None:
                inherited = self._inherit_rotated_offset(key, stat, live=live)
                if inherited is None:
                    # Historical file content has no trustworthy node-generation
                    # boundary when the durable checkpoint is absent.
                    self._files[key] = {
                        "device": stat.st_dev,
                        "inode": stat.st_ino,
                        "offset": stat.st_size,
                    }
                    self._state_dirty = True
                    continue
                plan.append((key, stat, inherited))
                continue
            recorded = int(previous.get("offset", 0))
            identity_changed = (
                previous.get("device") != stat.st_dev
                or previous.get("inode") != stat.st_ino
            )
            if identity_changed or stat.st_size < recorded:
                plan.append((key, stat, 0))
                continue
            plan.append((key, stat, min(recorded, stat.st_size)))
        return plan

    def _inherit_rotated_offset(
        self,
        key: str,
        stat: os.stat_result,
        *,
        live: list[tuple[str, os.stat_result]],
    ) -> int | None:
        """The checkpoint of this inode under the name it had before rotation.

        logrotate renames ``fabricmanager.log`` to ``fabricmanager.1.log``, so
        the same inode reappears under a name that has no checkpoint. Baselining
        it at EOF discarded every line written between the last poll and the
        rename -- including the fatal SXID that made Fabric Manager rotate. The
        old name keeps a checkpoint at the start of its replacement, or loses it
        altogether when nothing is there any more.
        """

        current = {name: (item.st_dev, item.st_ino) for name, item in live}
        identity = (stat.st_dev, stat.st_ino)
        for other, record in list(self._files.items()):
            if other == key:
                continue
            if (record.get("device"), record.get("inode")) != identity:
                continue
            if current.get(other) == identity:
                # Two names for one inode (a hard link, or two globs matching
                # the same file): its checkpoint is not this key's to take.
                continue
            offset = min(int(record.get("offset", 0)), stat.st_size)
            replacement = current.get(other)
            if replacement is None:
                del self._files[other]
            else:
                self._files[other] = {
                    "device": replacement[0],
                    "inode": replacement[1],
                    "offset": 0,
                }
            self._files[key] = {
                "device": stat.st_dev,
                "inode": stat.st_ino,
                "offset": offset,
            }
            self._state_dirty = True
            LOGGER.warning(
                "Fabric Manager log %s was rotated to %s; resuming its tail "
                "at byte %d instead of baselining at the end",
                other,
                key,
                offset,
            )
            return offset
        return None

    def _records_from_file(
        self,
        key: str,
        stat: os.stat_result,
        offset: int,
        collected_at: datetime,
    ) -> list[dict[str, Any]]:
        """Read one file from ``offset`` in binary and decode line by line.

        The offsets are byte offsets. They used to be ``TextIOWrapper.tell()``
        cookies, which are opaque: they cannot be compared with ``st_size``, and
        seeking one byte back from one could land inside a multibyte character.
        """

        records: list[dict[str, Any]] = []
        with Path(key).open("rb") as stream:
            if offset > 0:
                stream.seek(offset - 1)
                boundary = stream.read(1)
                stream.seek(offset)
                if boundary != b"\n":
                    skipped = stream.readline()
                    if skipped:
                        self._warn_resumed_inside_a_line(key, offset)
                        records.append(
                            {
                                "message": "",
                                "_eligible": False,
                                "_checkpoint": self._file_checkpoint(
                                    key, stat, stream.tell()
                                ),
                            }
                        )
            while True:
                start = stream.tell()
                raw = stream.readline()
                if not raw:
                    break
                if not raw.endswith(b"\n"):
                    # A line still being written: its checkpoint stays at
                    # ``start`` so the next round reads it whole.
                    break
                message = raw.decode("utf-8", errors="replace").rstrip("\n")
                stable = f"{stat.st_dev}:{stat.st_ino}:{start}"
                records.append(
                    {
                        "record_id": (
                            "fm-file-"
                            + self._record_identity(stat.st_dev, stat.st_ino, start)
                        ),
                        "observed_at": self._file_timestamp(
                            message, collected_at
                        ).isoformat(),
                        "message": message,
                        "source": "file",
                        "_eligible": True,
                        "_checkpoint": self._file_checkpoint(key, stat, stream.tell()),
                        "fields": {
                            "path": key,
                            "device": str(stat.st_dev),
                            "inode": str(stat.st_ino),
                            "offset": str(start),
                        },
                        "evidence_ref": (f"file://{self.node_id}{key}#{stable}"),
                    }
                )
        return records

    @staticmethod
    def _file_checkpoint(key: str, stat: os.stat_result, offset: int) -> dict[str, Any]:
        return {
            "kind": "file",
            "path": key,
            "device": stat.st_dev,
            "inode": stat.st_ino,
            "offset": offset,
        }

    def _record_identity(self, *parts: object) -> str:
        """A record id that says which cluster, node and boot, and then what.

        The control plane derives ``event_id`` from ``cluster_id`` and the
        record id alone (``hma.py``), and treats a repeated ``event_id`` as a
        duplicate. ``dev:ino:offset`` is not node-unique -- identically imaged
        nodes share the device, and the same inode and offset for
        ``/var/log/fabricmanager.log`` is plausible on one AMI -- so node B's
        SXID was folded into node A's event and never recovered. This mirrors
        ``node.py:_entry_identity``; the parts are JSON-encoded so no two part
        lists can spell one string. The path is deliberately not a part: a
        rotation renames the file and the line must keep its id.
        """

        return hashlib.sha256(
            json.dumps(
                [self.context.cluster_id, self.node_id, self.boot_id, *parts],
                ensure_ascii=True,
                separators=(",", ":"),
                default=str,
            ).encode("utf-8")
        ).hexdigest()

    def _warn_resumed_inside_a_line(self, key: str, offset: int) -> None:
        """Say once that a checkpoint did not land on a line boundary.

        A checkpoint written before the offsets were bytes is a text cookie, and
        one written by a crash mid-append can point anywhere. Either way the read
        re-syncs to the next newline rather than deliver half a line, and that is
        worth exactly one warning per process, not one per line.
        """

        if self._resync_warned:
            return
        self._resync_warned = True
        LOGGER.warning(
            "Fabric Manager log %s resumed at byte %d, which is not a line "
            "boundary; re-syncing to the next line (an offset from before the "
            "checkpoints were byte counts, or a torn append)",
            key,
            offset,
        )

    def _bound_tracked_files(self, seen: set[str]) -> None:
        """Forget the oldest offsets of files the glob no longer finds.

        Files still present are never evicted: dropping a live file's offset
        would re-baseline it at EOF next round and skip whatever it gained.
        """

        excess = len(self._files) - self.max_tracked_files
        if excess <= 0:
            return
        stale = [key for key in self._files if key not in seen]
        evicted = stale[:excess]
        for key in evicted:
            del self._files[key]
        if evicted:
            self._state_dirty = True
        if not self._files_bound_warned:
            self._files_bound_warned = True
            LOGGER.warning(
                "Fabric Manager tracked file table exceeded %d entries; "
                "forgot %d stale offset(s), %d live file(s) kept",
                self.max_tracked_files,
                len(evicted),
                len(self._files),
            )

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

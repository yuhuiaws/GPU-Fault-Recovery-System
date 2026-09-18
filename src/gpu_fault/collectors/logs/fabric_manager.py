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
from threading import RLock
from typing import Any, Callable

from gpu_fault.channel_registry import (
    COLLECTOR_HEALTH_PATH,
    FABRIC_MANAGER_PATH,
)
from gpu_fault.collectors.models import CollectorContext, CollectorStats
from gpu_fault.collectors.logs.fabric_manager_cursor import FabricManagerCursorState
from gpu_fault.collectors.logs.fabric_manager_receipts import (
    FabricManagerReceiptLog,
    RoundReceipt,
    identity_sha256,
)
from gpu_fault.collectors.scheduling import next_stable_phase
from gpu_fault.collectors.sinks import CollectorError, EventSink, deliver_event
from gpu_fault.env import env_bool
from gpu_fault.telemetry import CollectorKind

LOGGER = logging.getLogger(__name__)

#: How often to repeat the warning that a resume offset sits inside a record the
#: daemon has not finished writing. The condition recurs every round while it
#: lasts, so it is reported on a schedule rather than once per line or never.
PARTIAL_RESUME_WARN_INTERVAL_SECONDS = 300.0

#: How many log files the collector keeps offsets for. A glob over rotated
#: files grows the persisted table forever otherwise (ARCH-G9).
DEFAULT_MAX_TRACKED_FILES = 256

#: A file line whose own timestamp is older than this, measured from the
#: collector's read time, advances the checkpoint but is not delivered. The
#: file is history the moment the checkpoint is lost (a fresh state file, a
#: rewritten log); replaying it opened a four-day-old Always-Fatal SXID as a new
#: incident after a node reboot and quarantined a healthy node.
DEFAULT_MAX_LINE_AGE_SECONDS = 900.0

SXID_SUMMARY_PATTERN = re.compile(
    r"\bSXid\b\s*\(PCI:[0-9a-fA-F:.]+\)\s*:\s*\d+\s*,\s*"
    r"(?:Non-fatal|Nonfatal|Fatal)\b",
    re.IGNORECASE,
)

FABRIC_MANAGER_SXID_PATTERN = SXID_SUMMARY_PATTERN


def file_record_id(
    cluster_id: str,
    node_id: str,
    boot_id: str,
    device: int,
    inode: int,
    offset: int,
    generation: int,
) -> str:
    # Generation zero preserves IDs from checkpoints written before this field.
    parts: list[object] = [cluster_id, node_id, boot_id, device, inode, offset]
    if generation:
        parts.extend(("generation", generation))
    digest = hashlib.sha256(
        json.dumps(parts, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return "fm-file-" + digest


def file_evidence_ref(
    node_id: str, path: str, device: int, inode: int, offset: int, generation: int
) -> str:
    position = f"{device}:{inode}:{offset}"
    if generation:
        position += f":generation={generation}"
    return f"file://{node_id}{path}#{position}"


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
        max_line_age_seconds: float = DEFAULT_MAX_LINE_AGE_SECONDS,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        if max_tracked_files < 1:
            raise ValueError("Fabric Manager tracked file limit must be at least 1")
        if max_line_age_seconds <= 0:
            raise ValueError("Fabric Manager max line age must be positive")
        self.max_line_age = timedelta(seconds=max_line_age_seconds)
        #: Lines skipped as older than ``max_line_age``, over the collector's life.
        self.stale_lines_skipped = 0
        self.sink = sink
        self.context = context
        self.node_id = node_id
        self.boot_id = boot_id or _read_boot_id()
        self.interval_seconds = interval_seconds
        self.max_tracked_files = max_tracked_files
        self.journal_enabled = journal_enabled
        self.journal_identifiers = tuple(
            item.lower() for item in journal_identifiers if item
        )
        self.log_paths = log_paths or []
        self.state_path = Path(state_path) if state_path else None
        self.now = now or (lambda: datetime.now(timezone.utc))
        self.runner = runner
        self._resync_warned = False
        # (path, offset, monotonic) of the last "still being written" warning,
        # so a daemon that stalls mid-line says so once instead of every round.
        self._partial_resume_warned: tuple[str, int, float] | None = None
        self.health_summary_seconds = int(
            os.getenv(
                "GPU_FAULT_FABRIC_MANAGER_HEALTH_SUMMARY_SECONDS",
                "300",
            )
        )
        if self.health_summary_seconds <= 0:
            raise ValueError("Fabric Manager health summary interval must be positive")
        self._collection_lock = RLock()
        self._receipt_round_failed = False
        self._receipts = FabricManagerReceiptLog(
            LOGGER,
            cluster_id=context.cluster_id,
            node_id=node_id,
            boot_id=self.boot_id,
            source_configuration=json.dumps(
                {
                    "journal_enabled": self.journal_enabled,
                    "journal_identifiers": list(self.journal_identifiers),
                    "log_paths": self.log_paths,
                    "state_path": str(self.state_path) if self.state_path else None,
                    "max_line_age_seconds": self.max_line_age.total_seconds(),
                    "max_tracked_files": self.max_tracked_files,
                },
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            ),
            summary_seconds=self.health_summary_seconds,
            monotonic=monotonic,
        )
        # First summary on the first collection (see kernel.py); later ones
        # keep their stable phase.
        self._next_health_summary = self.now()
        self._cursor = FabricManagerCursorState(
            self.state_path, max_tracked_files=self.max_tracked_files, logger=LOGGER
        )

    def collect_once(self) -> CollectorStats:
        with self._collection_lock:
            round_receipt = self._receipts.begin_round()
            self._receipt_round_failed = False
            complete = False
            try:
                stats = self._collect_round(round_receipt)
                complete = not self._receipt_round_failed
                return stats
            finally:
                self._receipts.complete_round(round_receipt, complete=complete)

    def _collect_round(self, round_receipt: RoundReceipt) -> CollectorStats:
        collected_at = self.now()
        try:
            records = [
                *self._journal_records(collected_at),
                *self._file_records(collected_at),
            ]
        except BaseException:
            self._cursor.flush()
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
                    self._cursor.commit_record(record)
                    continue
                checkpoint = record.pop("_checkpoint", None)
                attempt = self._receipts.begin_attempt(
                    round_receipt,
                    record_id=str(record.get("record_id") or ""),
                    source=record.get("source"),
                )
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
                    if not (result.delivered or result.buffered):
                        raise CollectorError("Fabric Manager sink outcome is unknown")
                except BaseException:
                    self._receipts.complete_attempt(attempt, "FAILED")
                    if checkpoint is not None:
                        record["_checkpoint"] = checkpoint
                    raise
                self._receipts.complete_attempt(
                    attempt, "BUFFERED" if result.buffered else "DELIVERED"
                )
                if result.buffered:
                    stats = stats.model_copy(update={"buffered": stats.buffered + 1})
                else:
                    stats = stats.model_copy(update={"delivered": stats.delivered + 1})
                if checkpoint is not None:
                    record["_checkpoint"] = checkpoint
                self._cursor.commit_record(record)
        finally:
            self._cursor.flush()
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
        except CollectorError:
            LOGGER.warning(
                "Fabric Manager health summary delivery failed; "
                "the collection round is unaffected"
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
                LOGGER.error("Fabric Manager log collection failed")
            time.sleep(self.interval_seconds)

    def _journal_records(self, collected_at: datetime) -> list[dict[str, Any]]:
        if not self.journal_enabled:
            return []
        if not shutil.which("journalctl"):
            self._receipt_round_failed = True
            return []
        completed = self._run_journal(collected_at, cursor=self._cursor.journal_cursor)
        if (
            completed.returncode != 0
            and self._cursor.journal_cursor
            and self._is_cursor_rejection(completed.stderr)
        ):
            LOGGER.warning(
                "Fabric Manager journal cursor is no longer "
                "available; resuming from the recent window"
            )
            self._cursor.clear_journal_cursor()
            completed = self._run_journal(collected_at, cursor=None)
        if completed.returncode != 0:
            LOGGER.error("Fabric Manager journal query failed")
            raise CollectorError(
                "Fabric Manager journal query failed: "
                + (completed.stderr.strip() or "unknown error")
            )
        records = []
        for line in completed.stdout.splitlines():
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                self._receipt_round_failed = True
                continue
            if not isinstance(item, dict):
                self._receipt_round_failed = True
                continue
            cursor = str(item.get("__CURSOR") or "")
            unit = str(item.get("_SYSTEMD_UNIT") or "")
            identifier = str(item.get("SYSLOG_IDENTIFIER") or item.get("_COMM") or "")
            eligible = self._is_fabric_manager(unit, identifier)
            message = self._message_text(item.get("MESSAGE"))
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

    @staticmethod
    def _message_text(value: object) -> str:
        """The MESSAGE field as text.

        ``journalctl --output=json`` emits a JSON array of byte values for any
        field that is not valid UTF-8, and ``--all`` makes that reachable for
        the long lines it used to null out instead. ``str()`` of that list is
        ``"[110, 118, ...]"``, which matches no SXID pattern, so the line was
        dropped without a trace (mirrors ``node.py:_message_text``).
        """

        if isinstance(value, list) and all(
            isinstance(item, int) and 0 <= item <= 255 for item in value
        ):
            return bytes(value).decode("utf-8", errors="replace")
        return str(value or "")

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
        plan = self._cursor.resume_offsets(live)
        # Persist a new file generation before any sink can acknowledge its IDs.
        # A crash/retry must not mint a second identity for the same new record.
        self._cursor.flush()
        for key, stat, offset in plan:
            try:
                records.extend(self._records_from_file(key, stat, offset, collected_at))
            except OSError:
                # A 0600 file under ProtectSystem=strict, or one that vanished
                # between the stat and the open. The read used to raise out of
                # the round and take the journal SXIDs of that round with it,
                # every round, permanently (ARCH-G4).
                self._receipt_round_failed = True
                LOGGER.warning(
                    "Fabric Manager log file unreadable this round: file_sha256=%s",
                    identity_sha256(key),
                )
                continue
        self._cursor.bound_files({key for key, _stat in live})
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
                except OSError:
                    # Rotated away between the glob and the stat: the next
                    # round sees the successor. One vanished sibling must
                    # not abort the whole round (ARCH-G9).
                    self._receipt_round_failed = True
                    LOGGER.warning(
                        "Fabric Manager log file unreadable this round: file_sha256=%s",
                        identity_sha256(str(path)),
                    )
                    continue
                seen.add(key)
                live.append((key, stat))
        return live

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
        generation = self._cursor.generation(key)
        stale_in_round = 0
        oldest_stale: datetime | None = None
        with Path(key).open("rb") as stream:
            opened = os.fstat(stream.fileno())
            if (opened.st_dev, opened.st_ino) != (stat.st_dev, stat.st_ino):
                raise OSError("Fabric Manager file identity changed before read")
            if offset > 0:
                stream.seek(offset - 1)
                boundary = stream.read(1)
                stream.seek(offset)
                if boundary != b"\n":
                    skipped = stream.readline()
                    if not skipped.endswith(b"\n"):
                        self._warn_resume_inside_an_unfinished_line(key, offset)
                        # The daemon is still writing that line. Consuming its
                        # tail would commit a checkpoint inside an incomplete
                        # record, and the next round would skip the completed
                        # line as "another" partial one -- silently, since the
                        # warning fires once per collector. The offset stays
                        # where it is until the line has a terminator, which is
                        # the invariant the read loop below already holds.
                        return records
                    self._warn_resumed_inside_a_line(key, offset)
                    records.append(
                        {
                            "message": "",
                            "_eligible": False,
                            "_checkpoint": self._cursor.file_checkpoint(
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
                observed_at = self._file_timestamp(message, collected_at)
                stale = collected_at - observed_at > self.max_line_age
                if stale:
                    self.stale_lines_skipped += 1
                    stale_in_round += 1
                    oldest_stale = min(oldest_stale or observed_at, observed_at)
                records.append(
                    {
                        "record_id": file_record_id(
                            self.context.cluster_id,
                            self.node_id,
                            self.boot_id,
                            stat.st_dev,
                            stat.st_ino,
                            start,
                            generation,
                        ),
                        "observed_at": observed_at.isoformat(),
                        "message": message,
                        "source": "file",
                        "_eligible": not stale,
                        "_checkpoint": self._cursor.file_checkpoint(
                            key, stat, stream.tell()
                        ),
                        "fields": {
                            "path": key,
                            "device": str(stat.st_dev),
                            "inode": str(stat.st_ino),
                            "offset": str(start),
                            "generation": str(generation),
                        },
                        "evidence_ref": file_evidence_ref(
                            self.node_id,
                            key,
                            stat.st_dev,
                            stat.st_ino,
                            start,
                            generation,
                        ),
                    }
                )
        if stale_in_round:
            LOGGER.warning(
                "Fabric Manager log file_sha256=%s: %d line(s) older than %ss (oldest %s) "
                "advanced the checkpoint without being reported",
                identity_sha256(key),
                stale_in_round,
                self.max_line_age.total_seconds(),
                oldest_stale.isoformat() if oldest_stale else "?",
            )
        return records

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
            "Fabric Manager log file_sha256=%s resumed at byte %d, which is not a line "
            "boundary; re-syncing to the next line (an offset from before the "
            "checkpoints were byte counts, or a torn append)",
            identity_sha256(key),
            offset,
        )

    def _warn_resume_inside_an_unfinished_line(self, key: str, offset: int) -> None:
        """Say that the resume offset sits inside a line nobody has finished.

        The read makes no progress until the line has its terminator, which is
        the point -- but doing that in silence is indistinguishable from a
        healthy collector, and a log whose last line never completes (a killed
        daemon, a truncated file) would never be read again with nothing said.
        Rate limited to one line per file and offset per interval, because the
        condition repeats every collection round while it lasts.
        """

        now = time.monotonic()
        last = self._partial_resume_warned
        if (
            last is not None
            and last[0] == key
            and last[1] == offset
            and now - last[2] < PARTIAL_RESUME_WARN_INTERVAL_SECONDS
        ):
            return
        self._partial_resume_warned = (key, offset, now)
        LOGGER.warning(
            "Fabric Manager log file_sha256=%s resumed at byte %d, inside a record the "
            "daemon has not finished writing; holding the offset there until "
            "the record is complete (nothing is read from this file meanwhile)",
            identity_sha256(key),
            offset,
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
        max_line_age_seconds=float(
            os.getenv("GPU_FAULT_FABRIC_MANAGER_MAX_LINE_AGE_SECONDS", "900")
        ),
    )

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

from gpu_fault.channel_registry import NODE_LOG_PATH
from gpu_fault.collector_requirements import COLLECTOR_SYSTEMD_UNITS
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
from gpu_fault.collectors.sinks import EventSink, deliver_event

LOGGER = logging.getLogger(__name__)


def excluded_journal_units() -> frozenset[str]:
    """The units whose journal lines are this system talking about itself.

    ``_record_collection_error`` warns to the local journal on purpose, and the
    other collectors log their own failures there too. Those lines are then read
    back by this collector, matched against the same log rules, and posted as
    node evidence -- so "the log collector cannot reach the control plane" turns
    into a finding about the node, and a collector stuck in a restart loop
    manufactures a fault out of its own complaint.

    Every unit excluded here has its own reporting channel: they all publish a
    ``CollectorStatus``, which is where their failures belong and where
    ``GpuFaultCollectorCollectionErrors`` reads them. The list is derived from
    ``COLLECTOR_SYSTEMD_UNITS`` rather than restated so a new collector cannot
    arrive without one. Units of this system that have no such channel --
    ``gpu-fault-gpu-persistence``, the node installer -- are deliberately left
    visible: for those, the journal is the only evidence there is.
    """

    override = os.getenv("GPU_FAULT_NODE_LOG_EXCLUDED_UNITS")
    if override is not None:
        # An empty value means "exclude nothing", which is how an operator gets
        # the feedback loop back for one debugging session.
        return frozenset(item.strip() for item in override.split(",") if item.strip())
    return frozenset(
        f"{unit}.service" for unit in set(COLLECTOR_SYSTEMD_UNITS.values())
    )


def _read_boot_id() -> str:
    """The kernel's boot id, or a marker that says it could not be read.

    It goes into every entry id this collector mints (see ``_entry_identity``),
    so a missing ``/proc`` must not stop log collection: the marker keeps the
    ids unique per node and per file, it only stops distinguishing boots.
    """

    try:
        return (
            Path("/proc/sys/kernel/random/boot_id").read_text(encoding="ascii").strip()
            or "unknown-boot"
        )
    except OSError:
        return "unknown-boot"


@dataclass(frozen=True)
class _JournalRead:
    """What one `journalctl` invocation produced, and what may be committed.

    `consumed_until` is `None` when the window was not read at all, which is the
    difference between a quiet node and a broken one: the cursor stays where it
    was so the next poll asks for the same span again, instead of stepping over
    logs nobody has seen.
    """

    entries: list[NodeLogEntry] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    consumed_until: datetime | None = None


class NodeLogCollector:
    """Polls journald/dmesg and configured training logs with stable IDs."""

    def __init__(
        self,
        sink: EventSink,
        context: CollectorContext,
        *,
        node_id: str,
        boot_id: str | None = None,
        interval_seconds: float = 10,
        training_log_paths: list[str] | None = None,
        state_path: str | None = None,
        initial_tail_bytes: int | None = None,
        max_entries_per_batch: int | None = None,
        max_batch_bytes: int | None = None,
        max_entry_bytes: int | None = None,
        context_lines: int | None = None,
        now: Callable[[], datetime] | None = None,
        runner: Callable[..., subprocess.CompletedProcess[str]] = (subprocess.run),
    ) -> None:
        self.sink = sink
        self.context = context
        self.node_id = node_id
        self.boot_id = boot_id or _read_boot_id()
        self.interval_seconds = interval_seconds
        self.training_log_paths = training_log_paths or []
        self.now = now or (lambda: datetime.now(timezone.utc))
        self.runner = runner
        self._journal_since: datetime | None = None
        # Per training log: `device`, `inode` and `offset`. The offset alone was
        # not enough -- a rotated file kept the old offset and the collector read
        # from the middle of the new one.
        self._files: dict[str, dict[str, int]] = {}
        # Which expanded training log path the last poll stopped on, so the next
        # one resumes after it instead of starting from the front of the list.
        self._resume_after: str | None = None
        self._collection_errors: list[str] = []
        self._suppressed_errors = 0
        # Cumulative per-reason loss, kept for the life of the process: one batch
        # cannot tell "this happened once" from "this happens every poll", and
        # the second one is the one that needs a human.
        self._discarded: dict[str, int] = {}
        # journald hands back a byte array for a MESSAGE that is not valid
        # UTF-8; those are decoded with replacement and counted here so a
        # driver line with one stray byte does not vanish from the batch.
        self.binary_messages_total = 0
        self._cold_start_reason: str | None = "no state was loaded"
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
        # How many neighbouring lines to keep around a line that matched a rule.
        # A matched line is rarely self-contained -- an Xid is preceded by the
        # driver saying what it was doing, a NCCL abort by the rank that timed out
        # -- and only the matched line used to survive, so the operator had to go
        # back to the node for the two lines that explained it.
        self.context_lines = (
            context_lines
            if context_lines is not None
            else int(os.getenv("GPU_FAULT_NODE_LOG_CONTEXT_LINES", "2"))
        )
        if self.context_lines < 0:
            raise ValueError("node log context lines must be non-negative")
        # `collection_errors` is carried in every batch until the node recovers,
        # so a node with a thousand rotating training logs would post a thousand
        # strings per poll. The count that was suppressed is reported instead.
        self.max_collection_errors = int(
            os.getenv("GPU_FAULT_NODE_LOG_MAX_COLLECTION_ERRORS", "20")
        )
        if self.max_collection_errors < 1:
            raise ValueError("node log collection error limit must be at least 1")
        # The offset table is persisted, so a glob over dated filenames grows it
        # forever: months of `train-2026-*.log` stay in the state file long after
        # the files are gone.
        self.max_tracked_files = int(
            os.getenv("GPU_FAULT_NODE_LOG_MAX_TRACKED_FILES", "512")
        )
        if self.max_tracked_files < 1:
            raise ValueError("node log tracked file limit must be at least 1")
        self.excluded_units = excluded_journal_units()
        # How far back a journal window may stretch after failed polls. Keeping
        # the cursor pinned while journalctl is down is what stops silent loss,
        # but an unbounded window is its own failure: this caps it and says in
        # `collection_errors` exactly which span was given up on.
        self.max_journal_window_seconds = int(
            os.getenv("GPU_FAULT_NODE_LOG_MAX_JOURNAL_WINDOW_SECONDS", "900")
        )
        if self.max_journal_window_seconds < 60:
            raise ValueError("node log journal window must be at least 60 seconds")
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
        previous_files = {key: dict(value) for key, value in self._files.items()}
        previous_resume_after = self._resume_after
        self._collection_errors = []
        self._suppressed_errors = 0
        try:
            journal = self._journal(collected_at)
            for message in journal.errors:
                self._record_collection_error(message)
            # Each source keeps its own context: neighbouring lines only mean
            # anything within the stream they were written to.
            entries = self._with_context(journal.entries)
            entries.extend(
                self._with_context(
                    self._training_logs(
                        collected_at,
                        max_entries=self.max_entries_per_batch,
                        max_bytes=self.max_batch_bytes,
                    )
                )
            )
            candidates = len(entries)
            entries = self._limit_entries(
                entries,
                max_entries=self.max_entries_per_batch,
                max_bytes=self.max_batch_bytes,
            )
            if len(entries) < candidates:
                # These are dropped on purpose, lowest log-signal priority first,
                # and re-reading them would drop the same ones again. Saying how
                # many is the only way an operator can tell a quiet batch from a
                # batch that did not fit.
                dropped = candidates - len(entries)
                self._record_discard("batch-limit-entries", dropped)
                self._record_collection_error(
                    f"{dropped} matching entries dropped by the "
                    f"batch limits ({self.max_entries_per_batch} entries, "
                    f"{self.max_batch_bytes} bytes)"
                )
        except Exception:
            self._journal_since = previous_since
            self._files = previous_files
            self._resume_after = previous_resume_after
            raise
        errors = self._batch_errors()
        # One periodic slot carries whichever of the two an empty batch means: a
        # health summary when nothing failed, the collection errors when
        # something did. Both are reported on the same schedule so a node whose
        # journalctl is broken does not post every interval.
        periodic_due = not entries and collected_at >= self._next_health_summary
        health_summary = periodic_due and not errors
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
                else ["collection-error"]
                if periodic_due
                else []
            ),
            collection_errors=errors,
        )
        try:
            if entries or periodic_due:
                result = deliver_event(
                    self.sink,
                    NODE_LOG_PATH,
                    batch.model_dump(mode="json"),
                )
                # A batch the outbox took is as good as delivered for the
                # cursor (ARCH-G3): rolling back re-read the same window and
                # re-buffered the same entries every poll. Only a batch that
                # went nowhere pins the cursor.
                result.raise_for_failure()
                if result.buffered:
                    LOGGER.warning(
                        "node log batch persisted to the collector outbox; "
                        "advancing the cursor: batch=%s error=%s",
                        batch.batch_id,
                        result.error,
                    )
            if periodic_due:
                self._next_health_summary = next_stable_phase(
                    collected_at,
                    cluster_id=self.context.cluster_id,
                    node_id=self.node_id,
                    channel="NODE_LOGS",
                    interval_seconds=self.health_summary_seconds,
                )
            if journal.consumed_until is not None:
                self._journal_since = journal.consumed_until
            self._save_state()
        except Exception:
            self._journal_since = previous_since
            self._files = previous_files
            self._resume_after = previous_resume_after
            raise
        return batch

    def run(self) -> None:
        while True:
            try:
                self.collect_once()
            except Exception:
                LOGGER.exception("node log collection failed")
            time.sleep(self.interval_seconds)

    def _record_collection_error(self, message: str) -> None:
        """Notes something this batch could not read.

        Every caller has already decided to carry on with a partial batch, so the
        note is the only trace left; it goes to the local log as well because a
        node whose sink is unreachable is exactly the case this exists for.

        Past the limit only the count survives. The list is carried in every batch
        and held in memory until the node recovers, so a node whose whole training
        log directory rotated at once would otherwise post one string per file and
        keep growing the batch it is already struggling to deliver.
        """

        if len(self._collection_errors) >= self.max_collection_errors:
            self._suppressed_errors += 1
            LOGGER.warning("node log collection incomplete (suppressed): %s", message)
            return
        self._collection_errors.append(message)
        LOGGER.warning("node log collection incomplete: %s", message)

    def _record_discard(self, reason: str, amount: int = 1) -> None:
        """Counts what was given up on, by reason, for the life of the process.

        A batch says what this poll could not read; it cannot say whether that is
        new. A node that drops entries to the batch limit once during a burst and
        a node that has been dropping them every ten seconds for a day produce the
        same string, and only the second one is losing evidence continuously. The
        unit is part of the reason name because these are not all entries -- some
        are files, some are seconds of journal.
        """

        if amount > 0:
            self._discarded[reason] = self._discarded.get(reason, 0) + amount

    def _batch_errors(self) -> list[str]:
        """What this batch reports it could not read.

        The running totals are attached only to a batch that already carries a
        failure. ``collection_errors`` is what the control plane reads to decide a
        batch was not a success -- an error-only batch does not move
        ``last_success_at``, and the node lands in
        ``gpu_fault_collector_erroring_nodes`` -- so a summary on every batch
        would pin the node in that count forever and there would be nothing left
        that could clear it.
        """

        errors = list(self._collection_errors)
        if self._suppressed_errors:
            errors.append(
                f"{self._suppressed_errors} further collection errors suppressed "
                f"(limit {self.max_collection_errors})"
            )
        if errors and self._discarded:
            totals = ", ".join(
                f"{reason}={count}" for reason, count in sorted(self._discarded.items())
            )
            errors.append(f"cumulative loss since collector start: {totals}")
        return errors

    def _with_context(self, entries: list[NodeLogEntry]) -> list[NodeLogEntry]:
        """Keeps the lines around a match, not just the match.

        Only the matched line used to survive, and a matched line is rarely
        self-contained: an Xid is preceded by the driver saying what it was doing,
        a NCCL abort by the rank that timed out. Those neighbours match no rule of
        their own, so they used to be dropped here and the operator had to go back
        to the node -- which after a reinstall no longer has them.

        ``entries`` must be in stream order, because that is what "neighbouring"
        means. A context line is marked so the control plane can tell it apart,
        and it is safe to carry: matching no rule is exactly what makes it
        incapable of producing a finding of its own, and it sorts to the bottom of
        ``_limit_entries`` (log signal priority -1), so context is what the batch
        limits give up first.
        """

        matched = [
            index
            for index, entry in enumerate(entries)
            if matching_log_rules(entry.message)
        ]
        if not matched or not self.context_lines:
            return [entries[index] for index in matched]
        keep: set[int] = set()
        for index in matched:
            keep.update(
                range(
                    max(0, index - self.context_lines),
                    min(len(entries), index + self.context_lines + 1),
                )
            )
        selected = set(matched)
        return [
            entries[index]
            if index in selected
            else entries[index].model_copy(
                update={"fields": {**entries[index].fields, "log_context": "true"}}
            )
            for index in sorted(keep)
        ]

    def _journal_window(self, until: datetime) -> tuple[datetime, list[str]]:
        """Where this poll starts reading, and what it had to give up on.

        The cursor stays put across a failed poll, so a long journalctl outage
        would otherwise ask for a window of hours and time out forever. Capping
        it trades the oldest logs for making progress, and says which span went.

        With no cursor there is nothing to resume from and the five minute default
        is a guess, so it is reported as a gap. That covers both the first start
        on a node and a restart that lost its state -- a reinstall, a wiped state
        directory, an unreadable state file -- and those are the cases where the
        collector was silently starting five minutes ago and nobody could tell how
        long it had actually been down.
        """

        errors = []
        since = self._journal_since
        if since is None:
            since = until - timedelta(minutes=5)
            errors.append(
                f"no journal cursor ({self._cold_start_reason}), reading from the "
                f"5 minute default at {since.isoformat()}: anything older than that "
                "was never read by this collector"
            )
        oldest = until - timedelta(seconds=self.max_journal_window_seconds)
        if since >= oldest:
            return since, errors
        self._record_discard(
            "journal-window-capped-seconds",
            int((oldest - since).total_seconds()),
        )
        errors.append(
            f"journal window capped at {self.max_journal_window_seconds}s: "
            f"gave up on {since.isoformat()}..{oldest.isoformat()}"
        )
        return oldest, errors

    @staticmethod
    def _stderr_detail(stderr: str | None) -> str:
        lines = [line.strip() for line in (stderr or "").splitlines() if line.strip()]
        if not lines:
            return "no stderr"
        # The last line is the one that says why it stopped, and it is bounded
        # because this string is carried in every batch until the node recovers.
        return lines[-1][:200]

    def _journal(self, until: datetime) -> _JournalRead:
        if not shutil.which("journalctl"):
            # There is no journal on this node and there never will be, so the
            # cursor advances: pinning it would grow a window nothing can read.
            return _JournalRead(consumed_until=until)
        since, errors = self._journal_window(until)
        completed = self.runner(
            [
                "journalctl",
                "--since",
                f"@{since.timestamp()}",
                "--until",
                f"@{until.timestamp()}",
                "--output=json",
                "--no-pager",
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        parsed = self._parse_journal(completed.stdout)
        entries = sorted(
            (entry for entry in parsed if entry.unit not in self.excluded_units),
            key=lambda entry: entry.observed_at,
        )
        self._record_discard("self-unit-entries", len(parsed) - len(entries))
        if completed.returncode != 0:
            # Whatever it printed before failing is still worth posting, but the
            # window was not read: leaving `consumed_until` unset holds the cursor
            # so the next poll asks for the same span instead of stepping over it.
            errors.append(
                f"journalctl exited {completed.returncode}: "
                f"{self._stderr_detail(completed.stderr)}"
            )
            self._record_discard("journalctl-failures")
            return _JournalRead(entries=entries, errors=errors)
        if (completed.stderr or "").strip():
            # A zero exit read the window, so the cursor may advance; the
            # complaint still belongs in the batch.
            errors.append(f"journalctl warned: {self._stderr_detail(completed.stderr)}")
        if len(entries) > self.max_entries_per_batch:
            # Keep the oldest. `--lines` used to make journalctl hand back the
            # newest N of the window and the cursor then jumped past everything
            # older, which is loss that looked like a quiet node. The boundary
            # entry is read again next poll and de-duplicated by its cursor id.
            kept = entries[: self.max_entries_per_batch]
            errors.append(
                f"journal window {since.isoformat()}..{until.isoformat()} produced "
                f"{len(entries)} entries, took the oldest {len(kept)} and left the "
                "rest for the next poll"
            )
            return _JournalRead(
                entries=kept, errors=errors, consumed_until=kept[-1].observed_at
            )
        return _JournalRead(entries=entries, errors=errors, consumed_until=until)

    def _parse_journal(self, stdout: str) -> list[NodeLogEntry]:
        entries = []
        for line in stdout.splitlines():
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            message = self._bounded_message(self._message_text(item.get("MESSAGE")))
            # A journal cursor carries the boot id and the journal file id, so
            # it is unique across nodes on its own. The fallback for a line
            # without one used to hash the raw line, and two nodes printing the
            # same ``NVRM: Xid`` text verbatim then shared one id (P0-38A).
            cursor = str(item.get("__CURSOR") or self._entry_identity(line))
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

    def _entry_identity(self, *parts: object) -> str:
        """An entry id that says which node, which boot, and then what.

        Every id this collector mints starts from the node's identity. A
        training-log line used to be ``sha256(path:offset)`` -- every rank of a
        distributed job writes the same path on its own node, so the same
        offset on two nodes was one id, and the second node's fault was folded
        into the first node's incident and never recovered (P0-38A). The boot
        id and the file identity keep a rotated file, or a reused inode after a
        reboot, from restarting the offsets under ids that were already used.
        The parts are JSON-encoded so no two part lists can spell one string.
        """

        return hashlib.sha256(
            json.dumps(
                [self.context.cluster_id, self.node_id, self.boot_id, *parts],
                ensure_ascii=True,
                separators=(",", ":"),
                default=str,
            ).encode("utf-8")
        ).hexdigest()

    def _message_text(self, value: object) -> str:
        """The MESSAGE field as text.

        ``journalctl --output=json`` emits a JSON array of byte values when the
        field is not valid UTF-8. ``str()`` of that list is ``"[78, 86, ...]"``,
        which matches no rule, so the line was silently dropped (ARCH-G9).
        """

        if isinstance(value, list) and all(
            isinstance(item, int) and 0 <= item <= 255 for item in value
        ):
            self.binary_messages_total += 1
            return bytes(value).decode("utf-8", errors="replace")
        return str(value or "")

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
        if self.state_path is None:
            self._cold_start_reason = "this collector keeps no state file"
            return
        if not self.state_path.exists():
            self._cold_start_reason = f"no state file at {self.state_path}"
            return
        try:
            value = json.loads(self.state_path.read_text())
            since = value.get("journal_since")
            self._journal_since = datetime.fromisoformat(since) if since else None
            self._cold_start_reason = (
                None
                if self._journal_since is not None
                else "the state file held no journal cursor"
            )
            self._files = self._decode_files(
                value.get("training_log_files")
                or value.get("training_log_offsets")
                or {}
            )
            resume_after = value.get("training_log_resume_after")
            self._resume_after = str(resume_after) if resume_after else None
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            LOGGER.exception("cannot load node log collector state")
            self._journal_since = None
            self._files = {}
            self._resume_after = None
            self._cold_start_reason = "the state file could not be read"

    @staticmethod
    def _decode_files(stored: dict[str, Any]) -> dict[str, dict[str, int]]:
        """Reads both the offset-only and the file-identity state form.

        An agent that is already running has `{path: offset}` on disk under the
        old key, and it keeps it: rejecting that would send the collector back to
        `initial_tail_bytes` on the upgrade, which at the default of 0 means
        skipping every line written since the last poll. A legacy record is kept
        without a device or inode rather than with a placeholder, so the first
        poll afterwards adopts the file it finds instead of reporting that every
        training log rotated.
        """

        files: dict[str, dict[str, int]] = {}
        for key, item in stored.items():
            record = (
                {
                    "device": int(item.get("device", 0)),
                    "inode": int(item.get("inode", 0)),
                    "offset": int(item.get("offset", 0)),
                }
                if isinstance(item, dict)
                else {"offset": int(item)}
            )
            if record["offset"] >= 0:
                files[str(key)] = record
        return files

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
                    # A new key, because the value shape changed: an agent rolled
                    # back to the offset-only reader finds no `training_log_offsets`
                    # and tails, instead of failing to parse the whole file and
                    # losing the journal cursor with it.
                    "training_log_files": self._files,
                    # Where the rotation resumes. Persisted so a restart does not
                    # send every poll back to the front of the path list, which is
                    # how the last paths starved in the first place.
                    "training_log_resume_after": self._resume_after,
                },
                separators=(",", ":"),
            ),
            encoding="utf-8",
        )
        os.replace(temporary, self.state_path)

    def _training_log_offset(self, key: str, stat: os.stat_result) -> int:
        """Where to resume reading one training log.

        The offset alone cannot tell "the file grew" from "the file was replaced":
        both leave a name at a byte count. Comparing the recorded device and inode
        is what catches a rotation, whose old offset would otherwise land in the
        middle of a line of the new file and skip everything before it.
        """

        previous = self._files.get(key)
        if previous is None:
            return max(0, stat.st_size - self.initial_tail_bytes)
        recorded = int(previous.get("offset", 0))
        device = previous.get("device")
        if device is not None and (
            device != stat.st_dev or previous.get("inode") != stat.st_ino
        ):
            self._record_discard("rotated-training-logs")
            self._record_collection_error(
                f"{key} was rotated (device/inode changed) with {recorded} bytes "
                "read, reading the replacement from the start"
            )
            return 0
        if stat.st_size < recorded:
            self._record_discard("truncated-training-logs")
            self._record_collection_error(
                f"{key} was truncated to {stat.st_size} bytes below the {recorded} "
                "already read, reading it from the start"
            )
            return 0
        return min(recorded, stat.st_size)

    def _expanded_training_paths(self) -> list[str]:
        """Every configured training log that exists now, in a fair order.

        The list is rotated to start after the path the previous poll stopped on.
        A poll that spent its whole budget on the first path used to return there
        and the next poll started at the front again, so with one busy log first in
        the configuration every path after it was never read at all -- not late,
        never -- for as long as the busy one stayed busy. Rotating costs nothing
        when the budget is not exhausted, because then every path is visited
        either way.
        """

        ordered = list(
            dict.fromkeys(
                str(path)
                for configured in self.training_log_paths
                for path in sorted(Path("/").glob(configured.lstrip("/")))
                if path.is_file()
            )
        )
        if self._resume_after is None or self._resume_after not in ordered:
            return ordered
        cut = ordered.index(self._resume_after) + 1
        return ordered[cut:] + ordered[:cut]

    def _bound_tracked_files(self, visited: set[str]) -> None:
        """Keeps the persisted offset table from growing without end.

        A glob over dated filenames -- ``train-2026-*.log`` -- adds a record per
        file and nothing ever removed one, so months of vanished files stayed in
        the state file and in memory.

        A path is only dropped on identity when it is actually gone. One that
        exists but was not visited this poll still holds a real offset, and that is
        what the rotation above depends on: forgetting those would make every
        skipped file look new and tail it, losing whatever was written meanwhile.
        Past the cap the earliest recorded paths go first and that is reported,
        because forgetting an offset while the file is still there is loss.
        """

        for key in [
            key for key in self._files if key not in visited and not Path(key).exists()
        ]:
            del self._files[key]
        surplus = len(self._files) - self.max_tracked_files
        if surplus <= 0:
            return
        stale = [key for key in self._files if key not in visited][:surplus]
        for key in stale:
            del self._files[key]
        self._record_discard("forgotten-training-log-offsets", len(stale))
        self._record_collection_error(
            f"tracking {len(self._files) + len(stale)} training logs over the "
            f"{self.max_tracked_files} limit, forgot the offsets of {len(stale)}: "
            "they will be tailed rather than resumed if they are read again"
        )

    def _training_logs(
        self,
        observed_at: datetime,
        *,
        max_entries: int | None = None,
        max_bytes: int | None = None,
    ) -> list[NodeLogEntry]:
        entries: list[NodeLogEntry] = []
        entry_limit = self.max_entries_per_batch if max_entries is None else max_entries
        byte_limit = self.max_batch_bytes if max_bytes is None else max_bytes
        used_bytes = 0
        if entry_limit <= 0 or byte_limit <= 0:
            return entries
        ordered = self._expanded_training_paths()
        visited: list[str] = []
        budget_spent = False
        for key in ordered:
            path = Path(key)
            try:
                stat = path.stat()
            except OSError:
                # It was there when the glob ran and is gone now: a rotation that
                # landed in between. `_bound_tracked_files` drops what it left.
                continue
            visited.append(key)
            offset = self._training_log_offset(key, stat)
            with path.open("r", encoding="utf-8", errors="replace") as stream:
                stream.seek(offset)
                while True:
                    start = stream.tell()
                    line = stream.readline()
                    if not line:
                        break
                    entry_id = self._entry_identity(
                        stat.st_dev, stat.st_ino, key, start
                    )
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
                        budget_spent = True
                        break
                    entries.append(entry)
                    used_bytes += entry_size
                self._files[key] = {
                    "device": stat.st_dev,
                    "inode": stat.st_ino,
                    "offset": stream.tell(),
                }
            if budget_spent:
                # Stopping here rather than trying the remaining paths with an
                # empty budget is what makes the rotation fair: the next poll
                # starts at the path after this one.
                break
        self._resume_after = visited[-1] if visited else None
        unread = len(ordered) - len(visited)
        if unread:
            self._record_discard("deferred-training-log-reads", unread)
            self._record_collection_error(
                f"{unread} of {len(ordered)} training logs were not read this poll "
                f"(batch limits reached at {visited[-1]}), the next poll resumes "
                "after that one"
            )
        self._bound_tracked_files(set(visited))
        return entries


def build_from_environment(
    sink: EventSink, context: CollectorContext, arguments: argparse.Namespace
) -> NodeLogCollector:
    """The ``gpu-fault-collector logs`` factory named by the registry."""

    if not arguments.node_id:
        raise SystemExit("--node-id, NODE_NAME, or HOSTNAME is required")
    return NodeLogCollector(
        sink,
        context,
        node_id=arguments.node_id,
        interval_seconds=arguments.interval_seconds,
        training_log_paths=[
            item
            for item in os.getenv("GPU_FAULT_TRAINING_LOG_PATHS", "").split(",")
            if item
        ],
        state_path=os.getenv(
            "GPU_FAULT_LOG_STATE_PATH",
            "/var/lib/gpu-fault/log-collector-state.json",
        ),
    )

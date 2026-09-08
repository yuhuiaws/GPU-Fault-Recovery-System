from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
import subprocess
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

from gpu_fault.channel_registry import NODE_LOG_PATH
from gpu_fault.collector_requirements import COLLECTOR_SYSTEMD_UNITS
from gpu_fault.collectors.logs.node_sources import (
    SOURCE_SCAN_BYTE_BUDGET,
    SOURCE_SCAN_SECONDS_BUDGET,
    _bounded_message,
    _ScanBudget,
    _TrainingLogs,
)
from gpu_fault.collectors.models import CollectorContext
from gpu_fault.collectors.scheduling import next_stable_phase
from gpu_fault.collectors.sinks import EventSink, deliver_event
from gpu_fault.host_health import NodeLogBatch, NodeLogEntry
from gpu_fault.log_rules import log_signal_priority, matching_log_rules

LOGGER = logging.getLogger(__name__)

#: How long a streamed ``journalctl`` may take to finish its window before it
#: is killed: what ``subprocess.run(timeout=30)`` used to bound.
JOURNAL_READ_TIMEOUT_SECONDS = 30.0

#: How long to wait for a signalled ``journalctl`` to exit before killing it.
JOURNAL_REAP_SECONDS = 5.0


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


def _setting(value: int | None, name: str, default: int) -> int:
    """A limit from the constructor, the environment, or the reviewed default."""

    if value is not None:
        return value
    return int(os.getenv(f"GPU_FAULT_NODE_LOG_{name}", str(default)))


def _stderr_detail(stderr: str | None) -> str:
    lines = [line.strip() for line in (stderr or "").splitlines() if line.strip()]
    if not lines:
        return "no stderr"
    # The last line is the one that says why it stopped, and it is bounded
    # because this string is carried in every batch until the node recovers.
    return lines[-1][:200]


def _entry_size(entry: NodeLogEntry) -> int:
    return len(entry.model_dump_json().encode("utf-8"))


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


def _with_context(
    entries: list[NodeLogEntry], context_lines: int
) -> list[NodeLogEntry]:
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
    if not matched or not context_lines:
        return [entries[index] for index in matched]
    keep: set[int] = set()
    for index in matched:
        keep.update(
            range(
                max(0, index - context_lines),
                min(len(entries), index + context_lines + 1),
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


def _limit_entries(
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
        size = _entry_size(entry)
        if len(selected) >= max_entries:
            break
        if used + size > max_bytes:
            continue
        selected.append(entry)
        used += size
    return sorted(selected, key=lambda entry: entry.observed_at)


def _journal_command(since: datetime, until: datetime) -> list[str]:
    return [
        "journalctl",
        "--since",
        f"@{since.timestamp()}",
        "--until",
        f"@{until.timestamp()}",
        "--output=json",
        # Without --all journald replaces any field over 4096 bytes with a
        # null value, so a long driver line arrived as MESSAGE: null, became
        # "", matched no rule and was dropped without a trace -- and the long
        # lines are the ones this collector exists to catch. The collector
        # bounds entries itself (`max_entry_bytes`).
        "--all",
        "--no-pager",
    ]


def _budget_error(
    budget: _ScanBudget, since: datetime, until: datetime, *, kept: int
) -> str:
    window = f"journal window {since.isoformat()}..{until.isoformat()}"
    if budget.stop_reason == "entries":
        # `--lines=N` used to make journalctl hand back the newest N of the
        # window and the cursor then jumped past everything older, which is
        # loss that looked like a quiet node. The boundary entry is read
        # again next poll and de-duplicated by its cursor id.
        return (
            f"{window} filled the batch, read the oldest {kept} entries and "
            "left the rest for the next poll"
        )
    return (
        f"{window} hit the {budget.stop_reason} scan budget after "
        f"{budget.bytes_read} bytes, took the oldest {kept} entries and left "
        "the rest for the next poll"
    )


def _reap_journal(
    child: Any, scan: _JournalScan, *, streamed: bool, stopped_early: bool
) -> None:
    """Leave nothing running, and read how the child ended.

    A scan that stopped at its budget leaves journalctl writing into a pipe
    nobody reads: it has to be signalled and then drained by ``communicate``, or
    it blocks on its own write and never exits.

    A scan that read the window to its end must *not* be signalled. Closing
    stdout is not exiting: the child is still un-waited, so ``poll()`` is None,
    and a SIGTERM there made a healthy poll exit -15 -- reported as
    ``journalctl exited -15``, counted as a `journalctl-failures`, and the cursor
    held on a window that had in fact been read completely.
    """

    if not streamed:
        scan.returncode = int(getattr(child, "returncode", 0) or 0)
        scan.stderr = str(getattr(child, "stderr", "") or "")
        return
    try:
        if stopped_early and child.poll() is None:
            child.terminate()
        _, stderr = child.communicate(timeout=JOURNAL_REAP_SECONDS)
    except subprocess.TimeoutExpired:
        child.kill()
        _, stderr = child.communicate()
    scan.stderr = str(stderr or "")
    scan.returncode = int(child.returncode or 0)


@dataclass(frozen=True)
class _JournalRead:
    """What one `journalctl` invocation produced, and what may be committed.

    `consumed_until` is `None` when the window was not read at all, which is
    the difference between a quiet node and a broken one: the cursor stays put so
    the next poll asks for the same span. A budget stop sets it to the last entry
    actually read, for the same reason.
    """

    entries: list[NodeLogEntry] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    consumed_until: datetime | None = None


@dataclass
class _JournalScan:
    """What one scan of the journal produced, and how the child ended.

    The exit status belongs here rather than in a type of its own because it is
    only readable after the stream has been drained: ``_reap_journal`` fills it
    from a ``finally``, on the same object, whatever ended the scan.
    """

    entries: list[NodeLogEntry] = field(default_factory=list)
    unparseable: int = 0
    excluded: int = 0
    consumed_at: datetime | None = None
    returncode: int = 0
    stderr: str = ""


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
        scan_bytes: int | None = None,
        scan_seconds: float | None = None,
        journal_timeout_seconds: float | None = None,
        now: Callable[[], datetime] | None = None,
        monotonic: Callable[[], float] = time.monotonic,
        runner: Callable[..., Any] = subprocess.Popen,
    ) -> None:
        self.sink = sink
        self.context = context
        self.node_id = node_id
        self.boot_id = boot_id or _read_boot_id()
        self.interval_seconds = interval_seconds
        self.training_log_paths = training_log_paths or []
        self.now = now or (lambda: datetime.now(timezone.utc))
        # The scan budgets read the clock through this, so a test can show that
        # a child's start-up latency does not consume the budget.
        self.monotonic = monotonic
        self.runner = runner
        self._journal_since: datetime | None = None
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
        # Lines of this system's own units, dropped on purpose. Counted apart
        # from the loss totals: excluding them is a policy, and adding them to
        # "cumulative loss" made a healthy node look like it was losing evidence
        # every time this collector warned about itself.
        self.self_unit_entries_total = 0
        self._journal_timed_out = False
        self._cold_start_reason: str | None = "no state was loaded"
        self.state_path = Path(state_path) if state_path else None
        self.initial_tail_bytes = _setting(initial_tail_bytes, "INITIAL_TAIL_BYTES", 0)
        if self.initial_tail_bytes < 0:
            raise ValueError("node log initial tail bytes must be non-negative")
        self.max_entries_per_batch = _setting(
            max_entries_per_batch, "MAX_ENTRIES_PER_BATCH", 1000
        )
        self.max_batch_bytes = _setting(
            max_batch_bytes, "MAX_BATCH_BYTES", 4 * 1024 * 1024
        )
        self.max_entry_bytes = _setting(max_entry_bytes, "MAX_ENTRY_BYTES", 64 * 1024)
        if (
            self.max_entries_per_batch < 1
            or self.max_batch_bytes < 1024
            or self.max_entry_bytes < 256
            or self.max_entry_bytes > self.max_batch_bytes
        ):
            raise ValueError("invalid node log batch limits")
        # How many neighbouring lines to keep around a matched one (see
        # ``_with_context``).
        self.context_lines = _setting(context_lines, "CONTEXT_LINES", 2)
        if self.context_lines < 0:
            raise ValueError("node log context lines must be non-negative")
        # What one poll may read from one source before it stops and leaves the
        # rest behind the cursor. Not configurable through the environment: an
        # operator who raises these is raising the OOM risk they exist to remove.
        self.scan_bytes = SOURCE_SCAN_BYTE_BUDGET if scan_bytes is None else scan_bytes
        self.scan_seconds = (
            SOURCE_SCAN_SECONDS_BUDGET if scan_seconds is None else scan_seconds
        )
        self.journal_timeout_seconds = (
            JOURNAL_READ_TIMEOUT_SECONDS
            if journal_timeout_seconds is None
            else journal_timeout_seconds
        )
        if (
            self.scan_bytes < 1024
            or self.scan_seconds < 0
            or (self.journal_timeout_seconds <= 0)
        ):
            raise ValueError("invalid node log scan budget")
        # `collection_errors` is carried in every batch until the node recovers,
        # so a node with a thousand rotating training logs would post a thousand
        # strings per poll. The count that was suppressed is reported instead.
        self.max_collection_errors = _setting(None, "MAX_COLLECTION_ERRORS", 20)
        if self.max_collection_errors < 1:
            raise ValueError("node log collection error limit must be at least 1")
        # The offset table is persisted, so a glob over dated filenames grows it
        # forever: months of `train-2026-*.log` stay in the state file long after
        # the files are gone.
        self.max_tracked_files = _setting(None, "MAX_TRACKED_FILES", 512)
        if self.max_tracked_files < 1:
            raise ValueError("node log tracked file limit must be at least 1")
        self.excluded_units = excluded_journal_units()
        # How far back a journal window may stretch after failed polls. Keeping
        # the cursor pinned while journalctl is down is what stops silent loss,
        # but an unbounded window is its own failure: this caps it and says in
        # `collection_errors` exactly which span was given up on.
        self.max_journal_window_seconds = _setting(
            None, "MAX_JOURNAL_WINDOW_SECONDS", 900
        )
        if self.max_journal_window_seconds < 60:
            raise ValueError("node log journal window must be at least 60 seconds")
        self._training = _TrainingLogs(
            paths=self.training_log_paths,
            initial_tail_bytes=self.initial_tail_bytes,
            max_tracked_files=self.max_tracked_files,
            max_entry_bytes=self.max_entry_bytes,
            entry_identity=self._entry_identity,
            record_discard=self._record_discard,
            record_error=self._record_collection_error,
        )
        self._load_state()
        self.health_summary_seconds = _setting(None, "HEALTH_SUMMARY_SECONDS", 300)
        self._next_health_summary = next_stable_phase(
            self.now(),
            cluster_id=self.context.cluster_id,
            node_id=self.node_id,
            channel="NODE_LOGS",
            interval_seconds=self.health_summary_seconds,
        )

    def collect_once(self) -> NodeLogBatch:
        collected_at = self.now()
        snapshot = self._snapshot()
        self._collection_errors = []
        self._suppressed_errors = 0
        try:
            journal = self._journal(collected_at)
            for message in journal.errors:
                self._record_collection_error(message)
            # Each source keeps its own context: neighbouring lines only mean
            # anything within the stream they were written to.
            entries = _with_context(journal.entries, self.context_lines)
            # What the journal already put in the batch is what the training
            # logs may no longer read: one entry allowance, one batch.
            entries.extend(
                _with_context(
                    self._training_logs(
                        collected_at,
                        allowance=max(0, self.max_entries_per_batch - len(entries)),
                    ),
                    self.context_lines,
                )
            )
            candidates = len(entries)
            entries = _limit_entries(
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
            self._restore(snapshot)
            raise
        errors = self._batch_errors()
        # One periodic slot carries whichever of the two an empty batch means: a
        # health summary when nothing failed, the collection errors when
        # something did. Both are reported on the same schedule so a node whose
        # journalctl is broken does not post every interval.
        periodic_due = not entries and collected_at >= self._next_health_summary
        health_summary = periodic_due and not errors
        batch = NodeLogBatch(
            batch_id=f"logs-{self.node_id}-{int(collected_at.timestamp() * 1_000_000)}",
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
                    self.sink, NODE_LOG_PATH, batch.model_dump(mode="json")
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
            self._restore(snapshot)
            raise
        return batch

    def run(self) -> None:
        while True:
            try:
                self.collect_once()
            except Exception:
                LOGGER.exception("node log collection failed")
            time.sleep(self.interval_seconds)

    def _training_logs(
        self, observed_at: datetime, *, allowance: int | None = None
    ) -> list[NodeLogEntry]:
        """Read the configured training logs under one shared scan budget.

        The budget is shared across the paths on purpose: that is what makes the
        fair rotation mean anything, because a poll stops at the path where the
        budget ran out and the next one starts after it. ``allowance`` is how
        many entries the batch has left after the journal, so the offsets cannot
        move past lines this batch has no room for.
        """

        return self._training.read(
            observed_at, budget=self._scan_budget(max_entries=allowance)
        )

    def _snapshot(self) -> dict[str, Any]:
        """Everything a poll that goes nowhere has to put back.

        The cursor and the offsets were rolled back and the running totals were
        not, so a retried poll -- which re-reads the same window and re-detects
        the same rotation -- counted the same loss twice.
        """

        return {
            "journal_since": self._journal_since,
            "files": {key: dict(value) for key, value in self._training.files.items()},
            "resume_after": self._training.resume_after,
            "discarded": dict(self._discarded),
            "self_unit_entries_total": self.self_unit_entries_total,
            "binary_messages_total": self.binary_messages_total,
        }

    def _restore(self, snapshot: dict[str, Any]) -> None:
        self._journal_since = snapshot["journal_since"]
        self._training.files = snapshot["files"]
        self._training.resume_after = snapshot["resume_after"]
        self._discarded = snapshot["discarded"]
        self.self_unit_entries_total = snapshot["self_unit_entries_total"]
        self.binary_messages_total = snapshot["binary_messages_total"]

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

        Only loss belongs here. Lines this collector excludes on purpose do not:
        they are counted in ``self_unit_entries_total`` instead.
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

    def _scan_budget(self, *, max_entries: int | None = None) -> _ScanBudget:
        """One source's budget for one poll.

        The bytes and the wall clock are per source -- a busy journal must not
        take the training logs' turn away entirely -- but the entry allowance is
        shared, because ``_limit_entries`` caps the *batch* and anything read
        past that cap is dropped after the cursor has already moved past it.
        """

        return _ScanBudget(
            max_bytes=self.scan_bytes,
            max_seconds=self.scan_seconds,
            max_entries=(
                self.max_entries_per_batch if max_entries is None else max_entries
            ),
            context_lines=self.context_lines,
            monotonic=self.monotonic,
        )

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

    def _journal(self, until: datetime) -> _JournalRead:
        """Read the journal window as a stream, under a budget.

        The window stretches to 900 s after failed polls and used to be read
        with ``capture_output=True``: at ~1 KiB per entry a node logging 1000
        entries/s produced hundreds of MB against ``MemoryMax=768M``, and the
        OOM kill landed before the cursor was saved, so the next start read the
        same window again. The child is streamed and stopped at the first of
        three bounds -- entries the batch would keep, bytes, wall clock -- with
        the cursor left at the last entry actually read.
        """

        if not shutil.which("journalctl"):
            # There is no journal on this node and there never will be, so the
            # cursor advances: pinning it would grow a window nothing can read.
            return _JournalRead(consumed_until=until)
        since, errors = self._journal_window(until)
        self._journal_timed_out = False
        child = self.runner(
            _journal_command(since, until),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        # A ``Popen`` hands back a pipe to iterate; a ``subprocess.run``-shaped
        # result hands back the whole window as one string. Both are scanned by
        # the same budget, so the bound does not depend on which one it is.
        stdout = getattr(child, "stdout", None)
        streamed = stdout is not None and not isinstance(stdout, str)
        source: Any = stdout if streamed else (stdout or "").splitlines()
        lines: Iterator[str] = iter(source)
        budget = self._scan_budget()
        watchdog: threading.Timer | None = None
        if streamed:
            watchdog = threading.Timer(
                self.journal_timeout_seconds, self._kill_journal, (child,)
            )
            watchdog.daemon = True
            watchdog.start()
        scan = _JournalScan()
        try:
            self._scan_journal(scan, lines, budget)
        finally:
            if watchdog is not None:
                watchdog.cancel()
            # Every exit path, including a raise out of the scan: a child left
            # running holds a pipe nobody reads and accumulates as a zombie.
            _reap_journal(
                child,
                scan,
                streamed=streamed,
                # Only a scan that walked away from a child still writing may
                # signal it. Anything else is a complete read.
                stopped_early=(
                    budget.stop_reason is not None or self._journal_timed_out
                ),
            )
        entries = sorted(scan.entries, key=lambda entry: entry.observed_at)
        self.self_unit_entries_total += scan.excluded
        consumed_until: datetime | None
        if self._journal_timed_out:
            self._record_discard("journal-read-timeouts")
            errors.append(
                f"journalctl did not finish the window in "
                f"{self.journal_timeout_seconds}s and was killed (read timeout); "
                f"kept the {len(entries)} entries it had produced"
            )
            consumed_until = scan.consumed_at
        elif budget.stop_reason is not None:
            errors.append(_budget_error(budget, since, until, kept=len(entries)))
            consumed_until = scan.consumed_at
        elif scan.returncode != 0:
            # Whatever it printed before failing is still worth posting, but the
            # window was not read: leaving `consumed_until` unset holds the cursor
            # so the next poll asks for the same span instead of stepping over it.
            errors.append(
                f"journalctl exited {scan.returncode}: {_stderr_detail(scan.stderr)}"
            )
            self._record_discard("journalctl-failures")
            consumed_until = None
        else:
            if scan.stderr.strip():
                # A zero exit read the window, so the cursor may advance; the
                # complaint still belongs in the batch.
                errors.append(f"journalctl warned: {_stderr_detail(scan.stderr)}")
            consumed_until = until
        if scan.unparseable:
            # Whether they are gone depends on the branch above: a held cursor
            # asks for the same window again, and saying they were stepped over
            # when they were not is how an operator stops looking.
            self._record_discard("unparseable-journal-entries", scan.unparseable)
            errors.append(
                f"{scan.unparseable} journal entries could not be parsed and were "
                + (
                    "skipped, so the cursor moved past them"
                    if consumed_until is not None
                    else "skipped; the cursor holds, so they are read again"
                )
            )
        return _JournalRead(entries, errors, consumed_until=consumed_until)

    def _scan_journal(
        self, scan: _JournalScan, lines: Iterator[str], budget: _ScanBudget
    ) -> None:
        """Read journal lines one at a time until the budget is spent.

        Nothing here may raise on the content of a line: an unreadable entry
        used to come out of ``collect_once``, which rolls the cursor back, so
        every poll failed on the same line until the 900 s cap threw the lot
        away.
        """

        for line in lines:
            entry = self._parse_journal_line(line)
            matched = False
            if entry is None:
                scan.unparseable += 1
            else:
                # Read is read: the cursor may move past an entry this batch does
                # not carry, or the next poll would read it again for ever.
                scan.consumed_at = entry.observed_at
                if entry.unit in self.excluded_units:
                    scan.excluded += 1
                else:
                    matched = bool(matching_log_rules(entry.message))
                    scan.entries.append(entry)
            budget.account(len(line.encode("utf-8", errors="replace")), matched=matched)
            if budget.spent():
                break

    def _kill_journal(self, child: Any) -> None:
        """The watchdog: end a child that stopped producing and has not exited.

        A streamed read blocks in one place, so a journalctl wedged on a corrupt
        journal file would hold the poll for ever with nothing to restart. What
        did arrive is kept and the cursor stops at its last entry.
        """

        self._journal_timed_out = True
        try:
            child.kill()
        except OSError as exc:
            LOGGER.debug("journalctl child could not be killed: %s", exc)

    def _parse_journal_line(self, line: str) -> NodeLogEntry | None:
        """One journal entry, or ``None`` when the line does not hold one.

        ``json.loads("1")`` succeeds and ``item.get`` on the result raises, and a
        truncated line raises too. Both used to come out of ``collect_once`` and
        pin the cursor on the same line for ever. A line that cannot be read is
        counted by the caller and stepped over.
        """

        try:
            item = json.loads(line)
            if not isinstance(item, dict):
                return None
            message = _bounded_message(
                self._message_text(item.get("MESSAGE")), self.max_entry_bytes
            )
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
            return NodeLogEntry(
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
        except (ValueError, TypeError, AttributeError, OverflowError, OSError) as exc:
            LOGGER.debug("unreadable journal entry skipped: %s", exc)
            return None

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
            self._training.files = _decode_files(
                value.get("training_log_files")
                or value.get("training_log_offsets")
                or {}
            )
            resume_after = value.get("training_log_resume_after")
            self._training.resume_after = str(resume_after) if resume_after else None
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            LOGGER.exception("cannot load node log collector state")
            self._journal_since = None
            self._training.files = {}
            self._training.resume_after = None
            self._cold_start_reason = "the state file could not be read"

    def _save_state(self) -> None:
        """Write the cursor and the offsets so a crash cannot lose them.

        ``os.replace`` alone only orders the rename against the write in the page
        cache, not against a power loss: the next start could find a truncated
        file, read no cursor (the five minute default) and tail every training
        log. The file and its directory are both fsynced.
        """

        if self.state_path is None:
            return
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.state_path.with_suffix(self.state_path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as stream:
            stream.write(
                json.dumps(
                    {
                        "journal_since": (
                            self._journal_since.isoformat()
                            if self._journal_since
                            else None
                        ),
                        # A new key, because the value shape changed: an agent
                        # rolled back to the offset-only reader finds no
                        # `training_log_offsets` and tails, instead of failing to
                        # parse the whole file and losing the cursor with it.
                        "training_log_files": self._training.files,
                        # Where the rotation resumes. Persisted so a restart does
                        # not send every poll back to the front of the path list,
                        # which is how the last paths starved in the first place.
                        "training_log_resume_after": self._training.resume_after,
                    },
                    separators=(",", ":"),
                )
            )
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, self.state_path)
        directory = os.open(
            self.state_path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        )
        try:
            os.fsync(directory)
        finally:
            os.close(directory)


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

"""What one poll of the node log collector may read, and how a file is tailed.

Split out of ``node.py``, which holds the collector itself: the window, the
cursor, the batch limits and the delivery. What lives here is the reading -- the
budget that bounds one poll of one source, and the training-log reader that owns
the offset table, the rotation rules and the byte-accurate tail. The two halves
meet at ``_ScanBudget``, which the collector shares between its sources, and at
``_bounded_message``, which caps one entry however it was read.

Every rule here is fail-closed in the same direction: a source that could not be
read leaves its position alone, and a position that has been recorded is a
position nothing will skip.
"""

from __future__ import annotations

import logging
import copy
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from gpu_fault.host_health import NodeLogEntry
from gpu_fault.log_rules import matching_log_rules
from .node_records import decode_record, pod_identity, read_line

LOGGER = logging.getLogger(__name__)

#: How many bytes one poll may read from one source: the journal is one, the
#: training logs together are the other. What is left unread stays behind the
#: cursor, so the bound costs latency and never evidence.
SOURCE_SCAN_BYTE_BUDGET = 4 * 1024 * 1024

#: How long one poll may spend reading one source. The interval is 10 s and
#: there are two sources, so 200 ms each keeps the poll far inside it while
#: still covering thousands of lines.
SOURCE_SCAN_SECONDS_BUDGET = 0.2

#: How often to repeat the warning that a resume offset sits inside a line the
#: writer has not finished: the condition recurs every poll while it lasts.
PARTIAL_RESUME_WARN_INTERVAL_SECONDS = 300.0

#: How many consecutive polls may defer the training logs before it is reported.
#: One busy poll loses nothing -- the positions are recorded and the next poll
#: resumes -- so reporting it lit `GpuFaultCollectorCollectionErrors` on a node
#: that was merely busy. Three in a row is a node that is not keeping up, where
#: the backlog can outlive the file it is waiting in.
DEFERRED_ROUNDS_BEFORE_ERROR = 3


def _bounded_message(message: str, max_bytes: int) -> str:
    raw = message.encode("utf-8", errors="replace")
    if len(raw) <= max_bytes:
        return message
    marker = b"\n...[node-log-entry-truncated]...\n"
    available = max(0, max_bytes - len(marker))
    prefix = available // 2
    suffix = available - prefix
    bounded = raw[:prefix] + marker + raw[-suffix:]
    return bounded.decode("utf-8", errors="replace")


class _ScanBudget:
    """How much of one source one poll may read, and how much it would keep.

    The entry cap used to be applied to raw lines, which is not what a batch
    carries: 1000 lines of kubelet chatter filled it on their own and were then
    dropped by ``_with_context``, so at a 10 s interval a source could not
    deliver more than 100 lines/s of anything and the machine check behind the
    chatter waited for the 900 s window cap.

    ``kept`` counts what would survive ``_with_context`` instead: a matched
    line, the uncounted ``context_lines`` before it and the ones after it, with
    the overlap of two neighbours counted once.

    The wall clock starts at the first line, not at construction: journalctl
    opening the journal files on a cold page cache takes longer than the whole
    200 ms budget (332 ms measured), and a deadline that included it stopped the
    scan after one entry -- so the cursor advanced one entry per poll and the
    window grew to the 900 s cap, which is loss on a node whose journal is
    merely cold. What the budget bounds is the reading, which is the part that
    costs this process memory and the poll its interval.
    """

    def __init__(
        self,
        *,
        max_bytes: int,
        max_seconds: float,
        max_entries: int,
        context_lines: int,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self.max_bytes = max_bytes
        self.max_seconds = max_seconds
        self.max_entries = max_entries
        self.context_lines = context_lines
        self.bytes_read = 0
        self.kept = 0
        self.stop_reason: str | None = None
        self._monotonic = monotonic
        self._deadline: float | None = None
        self._leading = 0
        self._trailing = 0

    def account(self, raw_bytes: int, *, matched: bool) -> None:
        """Count one raw line that has been read, and what it costs the batch."""

        self.consume_bytes(raw_bytes)
        if matched:
            self.kept += 1 + min(self._leading, self.context_lines)
            self._leading = 0
            self._trailing = self.context_lines
        elif self._trailing:
            # Trailing context of an earlier match: already kept, so it cannot
            # also be counted as leading context of a later one.
            self._trailing -= 1
            self.kept += 1
            self._leading = 0
        else:
            self._leading = min(self._leading + 1, self.context_lines)

    def consume_bytes(self, raw_bytes: int) -> None:
        if self._deadline is None:
            self._deadline = self._monotonic() + self.max_seconds
        self.bytes_read += raw_bytes

    def spent(self) -> str | None:
        """Why the scan must stop, or ``None``.

        Checked after a line has been read, never before: a budget that stopped
        a poll before its first line would leave the cursor where it was and the
        collector would make no progress at all, for ever.
        """

        if self.stop_reason is None:
            if self.kept >= self.max_entries:
                self.stop_reason = "entries"
            elif self.bytes_read >= self.max_bytes:
                self.stop_reason = "byte"
            elif self._deadline is not None and self._monotonic() >= self._deadline:
                self.stop_reason = "wall-clock"
        return self.stop_reason


@dataclass(frozen=True)
class _TrainingRead:
    """One training log, its identity this poll, and where to resume reading."""

    key: str
    stat: os.stat_result
    offset: int


class _TrainingLogs:
    """Which training logs are read, and where the reading resumes in each.

    The whole of one source: the offset table, the rotation rules that decide
    whether a name still holds the file it held, and the byte-accurate reader that
    will not deliver half a line. The two reporting channels and the entry
    identity are not this reader's to define, so the collector hands them over.
    """

    def __init__(
        self,
        *,
        paths: list[str],
        initial_tail_bytes: int,
        max_tracked_files: int,
        max_entry_bytes: int,
        entry_identity: Callable[..., str],
        record_discard: Callable[..., None],
        record_error: Callable[[str], None],
        pod_log_root: str = "/var/log/pods",
    ) -> None:
        self.paths = paths
        self.initial_tail_bytes = initial_tail_bytes
        self.max_tracked_files = max_tracked_files
        self.max_entry_bytes = max_entry_bytes
        self.entry_identity = entry_identity
        self.record_discard = record_discard
        self.record_error = record_error
        self.pod_log_root = pod_log_root
        self.pending: dict[str, dict[str, Any]] = {}
        # Per training log: `device`, `inode` and `offset` in bytes. The offset
        # alone was not enough -- a rotated file kept the old offset and the
        # collector read from the middle of the new one.
        self.files: dict[str, dict[str, int]] = {}
        # Which expanded path the last poll stopped on, so the next one resumes
        # after it instead of starting from the front of the list.
        self.resume_after: str | None = None
        # Consecutive polls that read nothing here because the journal spent the
        # shared budget. Counted so the report waits for a pattern (see
        # `DEFERRED_ROUNDS_BEFORE_ERROR`); reset by any poll that reads.
        self.deferred_rounds = 0
        self._resync_warned = False
        self._partial_resume_warned: tuple[str, int, float] | None = None

    def _ordered_paths(self) -> list[str]:
        """Every configured training log that exists now, in a fair order.

        The list is rotated to start after the path the previous poll stopped on.
        A poll that spent its whole budget on the first path used to return there
        and the next poll started at the front again, so with one busy log first in
        the configuration every path after it was never read at all -- not late,
        never -- for as long as the busy one stayed busy. Rotating costs nothing
        when the budget is not exhausted, because then every path is visited
        either way.
        """

        found: dict[str, None] = {}
        for configured in self.paths:
            for path in Path("/").glob(configured.lstrip("/")):
                if path.is_file():
                    found[str(path)] = None
                if len(found) > self.max_tracked_files:
                    self.record_error(
                        f"training log discovery exceeds {self.max_tracked_files} "
                        "files; coverage is incomplete, narrow configured paths"
                    )
                    break
            if len(found) > self.max_tracked_files:
                break
        # Prefer already tracked names so excess files cannot repeatedly reset
        # a baseline. Discovery and state both have a hard cardinality bound.
        ordered = sorted(found, key=lambda key: (key not in self.files, key))[
            : self.max_tracked_files
        ]
        ordered.sort()
        if self.resume_after is None or self.resume_after not in ordered:
            return ordered
        cut = ordered.index(self.resume_after) + 1
        return ordered[cut:] + ordered[:cut]

    def _resume_plan(self, ordered: list[str]) -> list[_TrainingRead]:
        """Where every training log resumes, decided before any of them is read.

        The offsets are all resolved first because one rotation makes two names
        share one inode for one poll: the glob sorts ``training.log`` before
        ``training.log.1``, so reading in order would commit the replacement's
        identity under the old name and the renamed file could not inherit it.
        """

        live: list[tuple[str, os.stat_result]] = []
        for key in ordered:
            try:
                live.append((key, Path(key).stat()))
            except OSError:
                # There when the glob ran and gone now: a rotation that landed in
                # between. `_bound_tracked_files` drops what it left behind.
                continue
        identities = {key: (stat.st_dev, stat.st_ino) for key, stat in live}
        return [
            _TrainingRead(key, stat, self._offset_for(key, stat, identities))
            for key, stat in live
        ]

    def _offset_for(
        self,
        key: str,
        stat: os.stat_result,
        identities: dict[str, tuple[int, int]],
    ) -> int:
        """Where to resume reading one training log, in bytes.

        The offset alone cannot tell "the file grew" from "the file was
        replaced": both leave a name at a byte count. The recorded device and
        inode are what catch a rotation, whose old offset would otherwise land
        in the middle of the new file.
        """

        previous = self.files.get(key)
        if previous is None:
            inherited = self._inherit_rotated_offset(key, stat, identities=identities)
            if inherited is not None:
                return inherited
            return max(0, stat.st_size - self.initial_tail_bytes)
        recorded = int(previous.get("offset", 0))
        device = previous.get("device")
        if device is not None and (
            device != stat.st_dev or previous.get("inode") != stat.st_ino
        ):
            self._report_rotation(key, previous, recorded, identities)
            return 0
        if stat.st_size < recorded:
            self.record_discard("truncated-training-logs")
            self.record_error(
                f"{key} was truncated to {stat.st_size} bytes below the {recorded} "
                "already read, reading it from the start"
            )
            return 0
        return min(recorded, stat.st_size)

    def _report_rotation(
        self,
        key: str,
        previous: dict[str, int],
        recorded: int,
        identities: dict[str, tuple[int, int]],
    ) -> None:
        """Say that this name holds a different file now, and whether that is loss.

        It is not loss when the file it held is still on disk under another
        name: a rename, whose tail is read there. Counting that as loss made
        every logrotate cycle look like dropped evidence.

        A matching inode number is not proof of a rename: unlinking a file frees
        its inode and the next file created in that directory can be given the
        same number. A name already tracked under an identity of its own is
        therefore rejected (see ``_identity_is_free``), or a lost tail would be
        reported as readable elsewhere.
        """

        identity = (previous.get("device"), previous.get("inode"))
        successor = next(
            (
                other
                for other, item in identities.items()
                if other != key
                and item == identity
                and self._identity_is_free(other, identity)
            ),
            None,
        )
        if successor is not None:
            self.record_error(
                f"{key} was rotated to {successor} with {recorded} bytes read; its "
                "tail is read under that name and the replacement from the start"
            )
            return
        self.record_discard("rotated-training-logs")
        self.record_error(
            f"{key} was rotated (device/inode changed) with {recorded} bytes "
            "read, reading the replacement from the start"
        )

    def _identity_is_free(self, key: str, identity: tuple[Any, Any]) -> bool:
        """Whether ``key`` could be a name this identity moved to.

        It could not if ``key`` was already being read as a file of its own: then
        both names changed inode and the match is a recycled number.
        """

        tracked = self.files.get(key)
        if tracked is None:
            return True
        return (tracked.get("device"), tracked.get("inode")) == identity

    def _inherit_rotated_offset(
        self,
        key: str,
        stat: os.stat_result,
        *,
        identities: dict[str, tuple[int, int]],
    ) -> int | None:
        """The offset of this inode under the name it had before the rotation.

        logrotate renames ``training.log`` to ``training.log.1``, so the same
        inode reappears under a name with no offset of its own. Baselining it at
        EOF discarded every line written between the last poll and the rename --
        the lines that made the log rotate. The old name keeps an offset at the
        start of its replacement, or loses it when nothing is there any more.
        """

        identity = (stat.st_dev, stat.st_ino)
        for other, record in list(self.files.items()):
            if other == key:
                continue
            if (record.get("device"), record.get("inode")) != identity:
                continue
            if identities.get(other) == identity:
                # Two names for one inode (a hard link, or two globs matching the
                # same file): its offset is not this key's to take.
                continue
            if stat.st_size < int(record.get("offset", 0)):
                # A renamed file still holds everything that was read from it, so
                # one that is now shorter than the offset is a different file that
                # was given a recycled inode number. Inheriting would seek past
                # its first lines.
                continue
            offset = min(int(record.get("offset", 0)), stat.st_size)
            replacement = identities.get(other)
            if replacement is None:
                del self.files[other]
            else:
                self.files[other] = {
                    "device": replacement[0],
                    "inode": replacement[1],
                    "offset": 0,
                }
            self.files[key] = {
                "device": stat.st_dev,
                "inode": stat.st_ino,
                "offset": offset,
            }
            if other in self.pending:
                self.pending[key] = self.pending.pop(other)
            LOGGER.warning(
                "training log %s was rotated to %s; resuming its tail at byte %d "
                "instead of baselining at the end",
                other,
                key,
                offset,
            )
            return offset
        return None

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
            key for key in self.files if key not in visited and not Path(key).exists()
        ]:
            del self.files[key]
        surplus = len(self.files) - self.max_tracked_files
        if surplus <= 0:
            return
        stale = sorted(self.files, key=lambda key: key in visited)[:surplus]
        for key in stale:
            del self.files[key]
        self.record_discard("forgotten-training-log-offsets", len(stale))
        self.record_error(
            f"tracking {len(self.files) + len(stale)} training logs over the "
            f"{self.max_tracked_files} limit, forgot the offsets of {len(stale)}: "
            "they will be tailed rather than resumed if they are read again"
        )

    def read(self, observed_at: datetime, *, budget: _ScanBudget) -> list[NodeLogEntry]:
        """Read the configured training logs under one shared scan budget.

        The budget is shared across the paths on purpose: that is what makes the
        fair rotation mean anything, because a poll stops at the path where the
        budget ran out and the next one starts after it.
        """

        entries: list[NodeLogEntry] = []
        ordered = self._ordered_paths()
        plan = self._resume_plan(ordered)
        for read in plan:
            previous = self.files.get(read.key, {})
            if (
                previous.get("device") != read.stat.st_dev
                or previous.get("inode") != read.stat.st_ino
                or read.offset < int(previous.get("offset", 0))
            ):
                self.pending.pop(read.key, None)
            self._remember(read, read.offset)
        if budget.spent() is not None:
            # The journal used what this poll had, on the entry axis or the
            # byte axis. Reading a line here would move an offset past an entry
            # `_limit_entries` then drops, so nothing is read -- but every
            # position is still recorded, or a file not yet in the table would
            # look new next poll and be baselined at its larger size, skipping
            # everything appended in between.
            self._defer_reads(plan)
            return entries
        # This poll reads, so whatever streak of deferrals came before it ended.
        self.deferred_rounds = 0
        visited: list[str] = []
        for read in plan:
            visited.append(read.key)
            entries.extend(self._entries_from_file(read, observed_at, budget))
            if budget.spent():
                # Stopping here rather than trying the remaining paths with an
                # empty budget is what makes the rotation fair: the next poll
                # starts at the path after this one.
                break
        self.resume_after = visited[-1] if visited else None
        unread = len(ordered) - len(visited)
        if unread:
            stopped = visited[-1] if visited else "no readable path"
            self.record_discard("deferred-training-log-reads", unread)
            self.record_error(
                f"{unread} of {len(ordered)} training logs were not read this poll "
                f"(batch limits reached at {stopped}), the next poll resumes "
                "after that one"
            )
        self._bound_tracked_files(set(visited))
        self.pending = {
            key: item for key, item in self.pending.items() if key in self.files
        }
        return entries

    def _defer_reads(self, plan: list[_TrainingRead]) -> None:
        """Record where every training log stands without reading any of it.

        ``resume_after`` is deliberately left alone: no path took its turn, so
        the next poll starts where this one would have. The streak is counted
        here because one deferral is not a fault and a run of them is: see
        ``DEFERRED_ROUNDS_BEFORE_ERROR``.
        """

        if not plan:
            # Nothing is configured, or nothing exists: a poll cannot defer what
            # it does not have, and reporting it would put an error on every
            # batch of a node with no training logs at all.
            return
        for read in plan:
            self._remember(read, read.offset)
        self.deferred_rounds += 1
        # Counted every time, reported only when it persists: the count is what
        # says how often a node runs out of budget, and one poll of it is not a
        # loss -- the positions above are recorded, so nothing is skipped.
        self.record_discard("deferred-training-log-reads", len(plan))
        if self.deferred_rounds >= DEFERRED_ROUNDS_BEFORE_ERROR:
            self.record_error(
                f"{len(plan)} training logs were not read this poll for the "
                f"{self.deferred_rounds}th poll in a row (the journal spends the "
                "shared budget); their positions are recorded, but a backlog this "
                "long can outlive a rotation"
            )
        self._bound_tracked_files({read.key for read in plan})

    def _remember(self, read: _TrainingRead, offset: int) -> None:
        """Persist where one training log stands, in bytes.

        Called on every path out of a read, including the ones that delivered
        nothing: an unrecorded file is a *new* file next poll, and a new file is
        baselined at its size, which skips whatever was appended meanwhile.
        """

        self.files[read.key] = {
            "device": read.stat.st_dev,
            "inode": read.stat.st_ino,
            "offset": offset,
        }

    def _entries_from_file(
        self, read: _TrainingRead, observed_at: datetime, budget: _ScanBudget
    ) -> list[NodeLogEntry]:
        """Read one training log from its byte offset, line by whole line.

        The file is opened in binary because the offsets are byte counts: a
        ``TextIOWrapper.tell()`` cookie cannot be compared with ``st_size`` and
        can point inside a multibyte character.
        """

        entries: list[NodeLogEntry] = []
        pending_before = copy.deepcopy(self.pending.get(read.key, {}))
        try:
            return self._read_from_offset(read, observed_at, budget)
        except OSError as error:
            # A 0600 log, or one rotated away between the stat and the open.
            # This used to leave `collect_once`, which rolls the cursor back, so
            # one unreadable file cost the poll its journal entries as well --
            # and did so again every poll, for a permissions error for ever. A
            # rotation that happens while it is unreadable is still caught: the
            # device and inode recorded here are compared on the next poll.
            LOGGER.warning("training log %s could not be read: %s", read.key, error)
            self.record_discard("unreadable-training-logs")
            # The stat succeeded, so where this file stands is known -- and
            # recording it is what keeps the next poll from calling an untracked
            # file new and baselining it at its larger size, which silently
            # skipped everything appended while it could not be read.
            self._remember(read, read.offset)
            self.pending[read.key] = pending_before
            self.record_error(
                f"{read.key} could not be read ({error}); its lines are not in "
                "this batch and the next poll resumes at byte "
                f"{read.offset}, where it stands now"
            )
            return entries

    def _read_from_offset(
        self, read: _TrainingRead, observed_at: datetime, budget: _ScanBudget
    ) -> list[NodeLogEntry]:
        """Read one training log's new whole lines, from `read.offset`."""

        entries: list[NodeLogEntry] = []
        with Path(read.key).open("rb") as stream:
            pending = self.pending.setdefault(read.key, {})
            physical = pending.setdefault("physical", {})
            stream.seek(read.offset)
            if not physical and read.offset:
                stream.seek(read.offset - 1)
                if stream.read(1) != b"\n":
                    physical.update(start=read.offset, hex="", discard=True)
                    self._warn_resumed_inside_a_line(read.key, read.offset)
            while True:
                discard = physical.get("discard", False)
                line = read_line(stream, physical, budget, self.max_entry_bytes)
                if line is None:
                    if discard and not budget.spent():
                        self._warn_resume_inside_an_unfinished_line(
                            read.key, int(physical.get("start", read.offset))
                        )
                    break
                if discard:
                    continue
                start, raw = line
                record = decode_record(
                    raw,
                    start,
                    observed_at,
                    pending.setdefault("cri", {}),
                    self.max_entry_bytes,
                )
                if record is None:
                    if budget.spent():
                        break
                    continue
                start, event_at, message, fields = record
                if fields["timestamp_source"] == "cri":
                    fields.update(pod_identity(read.key, self.pod_log_root))
                entries.append(
                    NodeLogEntry(
                        entry_id=self.entry_identity(
                            read.stat.st_dev, read.stat.st_ino, read.key, start
                        ),
                        source="training-log",
                        observed_at=event_at,
                        message=message,
                        fields={"path": read.key, **fields},
                    )
                )
                budget.account(0, matched=bool(matching_log_rules(message)))
                if budget.spent():
                    break
            self._remember(read, stream.tell())
            if budget.spent() and stream.tell() < read.stat.st_size:
                self.record_error(
                    f"{read.key}: {budget.stop_reason} scan budget reached; "
                    "unread bytes and partial records resume next poll"
                )
        return entries

    def _warn_resumed_inside_a_line(self, key: str, offset: int) -> None:
        """Say once that an offset did not land on a line boundary.

        An offset written by an older agent is a text cookie, and one written by
        a crash mid-append can point anywhere. Either way the read re-syncs to
        the next newline, which is worth one warning per process, not one per
        line.
        """

        if self._resync_warned:
            return
        self._resync_warned = True
        LOGGER.warning(
            "training log %s resumed at byte %d, which is not a line boundary; "
            "re-syncing to the next line (an offset from before the offsets were "
            "byte counts, or a torn append)",
            key,
            offset,
        )

    def _warn_resume_inside_an_unfinished_line(self, key: str, offset: int) -> None:
        """Say that the resume offset sits inside a line nobody has finished.

        The read makes no progress until the line has its terminator, which is
        the point -- but in silence that is indistinguishable from a quiet file,
        and a log whose last line never completes would never be read again with
        nothing said. Rate limited per file and offset: the condition repeats.
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
            "training log %s resumed at byte %d, inside a line the job has not "
            "finished writing; retaining a bounded continuation until the line "
            "is complete (no record is emitted meanwhile)",
            key,
            offset,
        )

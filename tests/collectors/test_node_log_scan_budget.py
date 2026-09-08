"""What one poll of the node log collector is allowed to read (F1 / F7).

Two defects met here. The batch entry cap was applied to *raw* lines, so 1000
lines of kubelet chatter spent the whole budget of a poll and the machine check
behind them waited for the 900 s window cap -- or a rotation -- to throw it
away; at a 10 s interval that is a ceiling of 100 lines/s per source. And the
window was read with ``capture_output=True`` over a span of up to 900 s: at
~1 KiB per JSON entry a node logging 1000 entries/s produced hundreds of MB
against ``MemoryMax=768M``, and the OOM kill landed before the cursor was saved,
so the same window was read again after every ``RestartSec=5``.

The fix is a budget on the raw scan (bytes and wall clock) with the entry cap
counting only what the batch would keep, and a streamed child instead of a
buffered one. Every case below therefore checks the pair that matters: what the
batch carries, and where the cursor was left. A cursor at the end of a window
that was never read is silent loss, so a scan that stops early must leave it at
the last entry it actually read.
"""

from __future__ import annotations

import threading
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import gpu_fault.collectors.logs.node as node_module
from gpu_fault.collectors.logs.node import NodeLogCollector

from ._support import NOW, RecordingSink, context, json, timedelta

MATCHING = "machine check hardware error"
NCCL = "NCCL WARN collective timeout error"


def _journal_line(cursor: str, message: str, at: Any) -> str:
    return json.dumps(
        {
            "__CURSOR": cursor,
            "__REALTIME_TIMESTAMP": str(int(at.timestamp() * 1_000_000)),
            "MESSAGE": message,
            "_TRANSPORT": "journal",
        }
    )


class _StreamingJournalctl:
    """A ``journalctl`` child whose stdout is consumed lazily, like Popen's.

    ``consumed`` is how many lines the collector actually pulled out of the
    pipe, which is the only honest way to assert the memory bound: a test can
    hand this 50 000 lines and see how many of them the poll ever touched.
    """

    def __init__(
        self, lines: Iterable[str], *, returncode: int = 0, stderr: str = ""
    ) -> None:
        self.consumed = 0
        self.calls: list[list[str]] = []
        self.terminated = 0
        self.killed = 0
        self.communicated = 0
        self._pending = iter(lines)
        self._stderr = stderr
        self._final_returncode = returncode
        self.returncode: int | None = None
        self.stdout = self._stream()

    def __call__(
        self, command: Iterable[str], **_kwargs: object
    ) -> _StreamingJournalctl:
        self.calls.append(list(command))
        return self

    def _stream(self) -> Iterator[str]:
        for line in self._pending:
            self.consumed += 1
            yield line
        self.returncode = self._final_returncode

    def poll(self) -> int | None:
        return self.returncode

    def terminate(self) -> None:
        self.terminated += 1
        if self.returncode is None:
            self.returncode = -15

    def kill(self) -> None:
        self.killed += 1
        if self.returncode is None:
            self.returncode = -9

    def communicate(self, timeout: float | None = None) -> tuple[str, str]:
        self.communicated += 1
        if self.returncode is None:
            self.returncode = self._final_returncode
        return "", self._stderr


class _HangingJournalctl(_StreamingJournalctl):
    """A child that stops writing and never exits, until it is killed."""

    def __init__(self, lines: Iterable[str]) -> None:
        super().__init__(lines)
        self.released = threading.Event()

    def _stream(self) -> Iterator[str]:
        for line in self._pending:
            self.consumed += 1
            yield line
        # journalctl wedged on a corrupt journal file: nothing more arrives and
        # the process does not exit. Only the watchdog can end this poll.
        self.released.wait(timeout=30)

    def kill(self) -> None:
        super().kill()
        self.released.set()

    def terminate(self) -> None:
        super().terminate()
        self.released.set()


def _present_journalctl(monkeypatch: Any) -> None:
    monkeypatch.setattr(
        node_module.shutil, "which", lambda _name: "/usr/bin/journalctl"
    )


def _seeded_state(tmp_path: Path) -> Path:
    """A state file with a cursor, so a cold start is not what is under test."""

    state = tmp_path / "state.json"
    state.write_text(
        json.dumps({"journal_since": (NOW - timedelta(seconds=30)).isoformat()}),
        encoding="utf-8",
    )
    return state


def _collector(
    state: Path,
    *,
    runner: Any = None,
    training_log_paths: list[str] | None = None,
    **overrides: Any,
) -> NodeLogCollector:
    return NodeLogCollector(
        RecordingSink(),
        context(),
        node_id="worker-1",
        training_log_paths=training_log_paths or [],
        state_path=str(state),
        now=lambda: NOW,
        runner=runner or _StreamingJournalctl([]),
        **overrides,
    )


def test_three_thousand_quiet_journal_lines_no_longer_hide_the_one_that_matters(
    tmp_path, monkeypatch
) -> None:
    """The entry cap counts what the batch keeps, not what the scan reads.

    3000 lines of kubelet chatter used to fill ``max_entries_per_batch`` on
    their own -- they are dropped later, by ``_with_context``, so the cap was
    spent on lines nobody would ever see -- and the cursor then stopped at the
    thousandth of them. The machine check behind them arrived one poll later at
    best, and after 900 s of that, never.

    The time budget is raised here on purpose: this case is about the entry cap,
    and the wall clock has its own case below.
    """

    _present_journalctl(monkeypatch)
    state = _seeded_state(tmp_path)
    quiet = [
        _journal_line(
            f"c-{index}",
            f"kubelet: pod status update for pod-{index} in namespace default",
            NOW - timedelta(seconds=60) + timedelta(milliseconds=index),
        )
        for index in range(3000)
    ]
    runner = _StreamingJournalctl(
        [*quiet, _journal_line("c-mce", MATCHING, NOW - timedelta(seconds=1))]
    )
    collector = _collector(state, runner=runner, scan_seconds=30.0)

    batch = collector.collect_once()

    assert MATCHING in [item.message for item in batch.entries], (
        f"the matched line was starved by 3000 quiet ones: {batch.entries}"
    )
    assert len(batch.entries) == 3, (
        f"only the match and its two context lines belong here: {batch.entries}"
    )
    assert batch.collection_errors == [], (
        f"nothing was given up on, so nothing is reported: {batch.collection_errors}"
    )
    assert runner.consumed == 3001, "the whole window fits in the scan budget"
    assert json.loads(state.read_text())["journal_since"] == NOW.isoformat(), (
        "the window was read to its end, so the cursor is at its end"
    )


def test_three_thousand_quiet_training_log_lines_no_longer_hide_the_one_that_matters(
    tmp_path, monkeypatch
) -> None:
    """The same cap, on the other source: a per-step training log.

    Every step line is appended by the job and matches no rule, so the raw cap
    made the NCCL abort at the end of the file wait for as many polls as the job
    had steps.
    """

    _present_journalctl(monkeypatch)
    state = _seeded_state(tmp_path)
    log = tmp_path / "training.log"
    log.write_text(
        "".join(f"step {index} loss 0.5\n" for index in range(3000)) + f"{NCCL}\n",
        encoding="utf-8",
    )
    collector = _collector(
        state,
        training_log_paths=[str(log)],
        initial_tail_bytes=1_000_000,
        scan_seconds=30.0,
    )

    batch = collector.collect_once()

    assert NCCL in [item.message for item in batch.entries], (
        f"the abort was starved by 3000 step lines: {batch.entries}"
    )
    assert len(batch.entries) == 3, (
        f"only the match and its two context lines belong here: {batch.entries}"
    )
    stored = json.loads(state.read_text())["training_log_files"][str(log)]
    assert stored["offset"] == log.stat().st_size, (
        f"the whole file was scanned, so nothing is left to re-read: {stored}"
    )


def test_a_journal_flood_stops_at_the_entry_cap_without_being_read_into_memory(
    tmp_path, monkeypatch
) -> None:
    """A storm must not be buffered whole before the first entry is kept.

    ``capture_output=True`` read the entire window into one string, so a node
    logging thousands of entries a second was OOM-killed before ``_save_state``
    and re-read the same window after every restart. The stream stops at the
    cap, the child is reaped, and the cursor is left at the last entry read --
    the next poll continues from there rather than skipping to the window end.
    """

    _present_journalctl(monkeypatch)
    state = _seeded_state(tmp_path)
    first = NOW - timedelta(seconds=600)
    runner = _StreamingJournalctl(
        _journal_line(
            f"c-{index}", f"{MATCHING} {index}", first + timedelta(seconds=index)
        )
        for index in range(50_000)
    )
    collector = _collector(state, runner=runner, max_entries_per_batch=50)

    batch = collector.collect_once()

    assert runner.consumed == 50, (
        f"the scan read {runner.consumed} lines to keep 50: the stream was buffered"
    )
    assert len(batch.entries) == 50, f"the cap is the batch size: {len(batch.entries)}"
    assert runner.terminated == 1, "the journalctl child was left running"
    assert runner.communicated == 1, (
        "a terminated child whose pipe is never drained can block on its write"
    )
    assert json.loads(state.read_text())["journal_since"] == (
        (first + timedelta(seconds=49)).isoformat()
    ), "the cursor must stop at the last entry read, never at the window end"
    assert any(
        "left the rest for the next poll" in item for item in batch.collection_errors
    ), f"a poll that stopped early has to say so: {batch.collection_errors}"


def test_the_journal_child_is_reaped_when_the_scan_reads_the_whole_window(
    tmp_path, monkeypatch
) -> None:
    """No zombie journalctl accumulates behind a healthy poll."""

    _present_journalctl(monkeypatch)
    state = _seeded_state(tmp_path)
    runner = _StreamingJournalctl([_journal_line("c-1", MATCHING, NOW)])
    collector = _collector(state, runner=runner)

    batch = collector.collect_once()

    assert len(batch.entries) == 1, f"the match was not batched: {batch.entries}"
    assert runner.communicated == 1, "the child was never waited for"
    assert runner.terminated == 0, "a child that already exited must not be signalled"


def test_a_scan_that_runs_out_of_time_stops_at_the_last_entry_it_read(
    tmp_path, monkeypatch
) -> None:
    """The wall-clock budget, and the progress guarantee that goes with it.

    One line is always read before the budget is consulted: a zero budget that
    stopped before the first entry would leave the cursor where it was and the
    poll would make no progress at all, for ever.
    """

    _present_journalctl(monkeypatch)
    state = _seeded_state(tmp_path)
    first = NOW - timedelta(seconds=30)
    runner = _StreamingJournalctl(
        [
            _journal_line("c-1", f"{MATCHING} one", first),
            _journal_line("c-2", f"{MATCHING} two", NOW - timedelta(seconds=20)),
            _journal_line("c-3", f"{MATCHING} three", NOW - timedelta(seconds=10)),
        ]
    )
    collector = _collector(state, runner=runner, scan_seconds=0.0)

    batch = collector.collect_once()

    assert runner.consumed == 1, (
        f"the budget must not stop the scan before its first line: {runner.consumed}"
    )
    assert [item.entry_id for item in batch.entries] == ["c-1"], (
        f"the oldest entry is the one that was read: {batch.entries}"
    )
    assert json.loads(state.read_text())["journal_since"] == first.isoformat(), (
        "the two unread entries would be skipped by a cursor at the window end"
    )
    assert any("scan budget" in item for item in batch.collection_errors), (
        f"the batch has to name the budget it hit: {batch.collection_errors}"
    )


def test_a_scan_that_runs_out_of_bytes_stops_at_the_last_entry_it_read(
    tmp_path, monkeypatch
) -> None:
    """The byte budget is what bounds the memory of one poll."""

    _present_journalctl(monkeypatch)
    state = _seeded_state(tmp_path)
    first = NOW - timedelta(seconds=30)
    lines = [
        _journal_line(
            f"c-{index}", f"{MATCHING} " + "x" * 400, first + timedelta(seconds=index)
        )
        for index in range(20)
    ]
    # A budget of exactly three lines, measured rather than guessed: the point is
    # where the scan stops, not how long a journal entry happens to be.
    line_bytes = len(lines[0].encode("utf-8"))
    runner = _StreamingJournalctl(lines)
    collector = _collector(state, runner=runner, scan_bytes=3 * line_bytes)

    batch = collector.collect_once()

    assert runner.consumed == 3, (
        f"a budget of three lines is spent by three lines: {runner.consumed}"
    )
    assert len(batch.entries) == 3, f"everything read is kept: {batch.entries}"
    assert json.loads(state.read_text())["journal_since"] == (
        (first + timedelta(seconds=2)).isoformat()
    ), "the cursor stops at the third entry, which is the last one read"


def test_the_scan_budget_is_four_mebibytes_and_two_hundred_milliseconds() -> None:
    """The reviewed values, and the collector that uses them by default."""

    collector = NodeLogCollector(
        RecordingSink(), context(), node_id="worker-1", now=lambda: NOW
    )

    assert node_module.SOURCE_SCAN_BYTE_BUDGET == 4 * 1024 * 1024, (
        "the reviewed byte budget is 4 MiB per source per poll"
    )
    assert node_module.SOURCE_SCAN_SECONDS_BUDGET == 0.2, (
        "the reviewed time budget is 200 ms per source per poll"
    )
    assert collector.scan_bytes == 4 * 1024 * 1024, "the default is the constant"
    assert collector.scan_seconds == 0.2, "the default is the constant"


def test_a_journal_child_that_stops_writing_is_killed_and_the_poll_returns(
    tmp_path, monkeypatch
) -> None:
    """A streamed read has no ``subprocess.run(timeout=30)`` to save it.

    Reading line by line means one blocking read on a child that wedged on a
    corrupt journal file would hold the collector for ever, with nothing to
    restart because the process is alive. The watchdog kills it, the prefix that
    did arrive is kept, and the cursor stops at the last entry of that prefix.
    """

    _present_journalctl(monkeypatch)
    state = _seeded_state(tmp_path)
    first = NOW - timedelta(seconds=30)
    runner = _HangingJournalctl([_journal_line("c-1", MATCHING, first)])
    collector = _collector(state, runner=runner, journal_timeout_seconds=0.2)

    batch = collector.collect_once()

    assert runner.killed == 1, "the wedged child was never killed"
    assert [item.entry_id for item in batch.entries] == ["c-1"], (
        f"what did arrive before the hang is still evidence: {batch.entries}"
    )
    assert json.loads(state.read_text())["journal_since"] == first.isoformat(), (
        "the cursor stops at the last entry read, so nothing is skipped"
    )
    assert any("timeout" in item for item in batch.collection_errors), (
        f"a killed child has to be reported: {batch.collection_errors}"
    )


def test_the_journal_command_asks_for_untruncated_fields(tmp_path, monkeypatch) -> None:
    """Without ``--all`` journald nulls out every field over 4096 bytes.

    ``MESSAGE: null`` becomes ``""``, matches no rule and is dropped without a
    trace, which is exactly what happens to the long driver lines this collector
    exists to catch (F3). The collector bounds entries itself.
    """

    _present_journalctl(monkeypatch)
    state = _seeded_state(tmp_path)
    runner = _StreamingJournalctl([])
    collector = _collector(state, runner=runner)

    collector.collect_once()

    assert "--all" in runner.calls[-1], (
        f"a line over 4096 bytes arrives as null without it: {runner.calls[-1]}"
    )


def test_a_streamed_journalctl_failure_still_holds_the_cursor(
    tmp_path, monkeypatch
) -> None:
    """The exit status of a streamed child is read after the stream ends."""

    _present_journalctl(monkeypatch)
    state = _seeded_state(tmp_path)
    cursor = json.loads(state.read_text())["journal_since"]
    runner = _StreamingJournalctl(
        [], returncode=1, stderr="Failed to open journal: Permission denied\n"
    )
    collector = _collector(state, runner=runner)

    batch = collector.collect_once()

    assert batch.collection_errors[0] == (
        "journalctl exited 1: Failed to open journal: Permission denied"
    ), f"the exit code and its reason belong in the batch: {batch.collection_errors}"
    assert json.loads(state.read_text())["journal_since"] == cursor, (
        "a window that was not read must not move the cursor"
    )

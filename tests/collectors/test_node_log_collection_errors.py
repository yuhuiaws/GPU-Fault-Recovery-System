"""What a node log batch says when the collector could not read everything.

An empty batch used to mean "the node is quiet" and "log collection is broken"
equally well, and the second one was silent. These cases drive the collector
through `collect_once` and read the durable state file, because the pair that
matters is what the batch reports and where the cursor was left: a batch that
claims a window it never read is loss that looks like silence.
"""

from __future__ import annotations

from pathlib import Path

from ._support import (
    NOW,
    NodeLogCollector,
    RecordingSink,
    context,
    json,
    os,
    subprocess,
    timedelta,
)

MATCHING = "machine check hardware error"


def _journal_line(cursor: str, message: str, at: object) -> str:
    return json.dumps(
        {
            "__CURSOR": cursor,
            "__REALTIME_TIMESTAMP": str(int(at.timestamp() * 1_000_000)),
            "MESSAGE": message,
            "_TRANSPORT": "journal",
        }
    )


class _Journalctl:
    """A journalctl stand-in that records the arguments it was called with."""

    def __init__(
        self, *, stdout: str = "", stderr: str = "", returncode: int = 0
    ) -> None:
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode
        self.calls: list[list[str]] = []

    def __call__(self, command, **_kwargs) -> subprocess.CompletedProcess[str]:
        self.calls.append(list(command))
        return subprocess.CompletedProcess(
            list(command), self.returncode, stdout=self.stdout, stderr=self.stderr
        )

    def argument(self, name: str) -> str | None:
        for index, item in enumerate(self.calls[-1]):
            if item == name:
                return self.calls[-1][index + 1]
            if item.startswith(f"{name}="):
                return item.split("=", 1)[1]
        return None


def _collector(
    tmp_path: Path,
    runner: _Journalctl,
    *,
    clock: list,
    state: Path | None = None,
    training_log_paths: list[str] | None = None,
    max_entries_per_batch: int = 1000,
) -> NodeLogCollector:
    return NodeLogCollector(
        RecordingSink(),
        context(),
        node_id="worker-1",
        training_log_paths=training_log_paths or [],
        state_path=str(state) if state else str(tmp_path / "state.json"),
        max_entries_per_batch=max_entries_per_batch,
        now=lambda: clock[0],
        runner=runner,
    )


def _present_journalctl(monkeypatch) -> None:
    """Pretends journald exists, whatever the machine running the tests has."""

    from gpu_fault.collectors.logs import node as module

    monkeypatch.setattr(module.shutil, "which", lambda _name: "/usr/bin/journalctl")


def test_a_failed_journalctl_holds_the_cursor_and_names_the_exit_code(
    tmp_path, monkeypatch
) -> None:
    """A poll that could not read its window must not step over it.

    Advancing the cursor after a non-zero exit is unrecoverable loss: nothing
    else remembers that span. The batch reports the failure on the periodic slot
    so a node whose journalctl is broken does not post every interval.
    """

    _present_journalctl(monkeypatch)
    state = tmp_path / "state.json"
    cursor = NOW - timedelta(seconds=120)
    state.write_text(
        json.dumps({"journal_since": cursor.isoformat()}), encoding="utf-8"
    )
    runner = _Journalctl(
        returncode=1, stderr="Failed to open journal: Permission denied\n"
    )
    clock = [NOW]
    collector = _collector(tmp_path, runner, clock=clock, state=state)
    clock[0] = NOW + timedelta(seconds=600)

    batch = collector.collect_once()

    assert batch.entries == [], "a failed read has no entries to report"
    assert batch.collection_errors == [
        "journalctl exited 1: Failed to open journal: Permission denied",
        "cumulative loss since collector start: journalctl-failures=1",
    ], "the batch has to carry the exit code, the reason, and the running total"
    assert batch.edge_filter_reasons == ["collection-error"], (
        "an error-only batch must not be reported as a health summary"
    )
    assert json.loads(state.read_text())["journal_since"] == cursor.isoformat(), (
        "the cursor stays where it was so the next poll asks for the same span"
    )
    assert collector.sink.requests, "an error-only batch is still posted"


def test_a_quiet_journal_still_reports_a_health_summary(tmp_path, monkeypatch) -> None:
    """The periodic slot is shared, so the silent case has to keep working.

    The cursor is seeded because a collector that has never read anything reports
    that gap, which is a different case -- see
    ``test_a_start_without_a_cursor_reports_the_gap_it_cannot_see``.
    """

    _present_journalctl(monkeypatch)
    state = tmp_path / "state.json"
    state.write_text(
        json.dumps({"journal_since": (NOW - timedelta(seconds=30)).isoformat()}),
        encoding="utf-8",
    )
    clock = [NOW]
    collector = _collector(tmp_path, _Journalctl(), clock=clock, state=state)
    clock[0] = NOW + timedelta(seconds=600)

    batch = collector.collect_once()

    assert batch.collection_errors == [], "nothing failed on a quiet node"
    assert batch.edge_filter_reasons == ["health-summary"], (
        "a quiet node reports the summary it always did"
    )


def test_an_overflowing_window_takes_the_oldest_and_leaves_the_rest(
    tmp_path, monkeypatch
) -> None:
    """journalctl must not be asked to pick which entries survive.

    `--lines=N` returns the newest N of the window and the cursor then jumped
    past everything older, so a burst silently dropped its own beginning. Taking
    the oldest N and holding the cursor at the last one read costs one duplicate
    entry, which the cursor id de-duplicates.
    """

    _present_journalctl(monkeypatch)
    first = NOW - timedelta(seconds=30)
    second = NOW - timedelta(seconds=20)
    third = NOW - timedelta(seconds=10)
    # In the order journalctl writes them, which is the only order a streamed
    # read can see: the scan stops at the cap where it is, so what the batch
    # keeps and where the cursor lands are decided by the stream, not by a sort
    # over a window that was buffered whole.
    runner = _Journalctl(
        stdout="\n".join(
            [
                _journal_line("cursor-1", f"{MATCHING} one", first),
                _journal_line("cursor-2", f"{MATCHING} two", second),
                _journal_line("cursor-3", f"{MATCHING} three", third),
            ]
        )
    )
    state = tmp_path / "state.json"
    state.write_text(
        json.dumps({"journal_since": (NOW - timedelta(seconds=300)).isoformat()}),
        encoding="utf-8",
    )
    clock = [NOW]
    collector = _collector(
        tmp_path, runner, clock=clock, state=state, max_entries_per_batch=2
    )

    batch = collector.collect_once()

    assert [item.entry_id for item in batch.entries] == ["cursor-1", "cursor-2"], (
        "the oldest entries are the ones that would otherwise be lost"
    )
    assert not any(item.startswith("--lines") for item in runner.calls[-1]), (
        "journalctl must not be the one dropping entries"
    )
    assert len(batch.collection_errors) == 1, "one report for one truncated window"
    assert "read the oldest 2" in batch.collection_errors[0], (
        "the batch says how the window was cut"
    )
    assert json.loads(state.read_text())["journal_since"] == second.isoformat(), (
        "the cursor stops at the last entry read, not at the end of the window"
    )


def test_a_long_outage_caps_the_window_and_says_what_it_gave_up(
    tmp_path, monkeypatch
) -> None:
    """Holding the cursor cannot mean an unbounded window.

    After a long journalctl outage the pinned cursor would ask for hours and time
    out forever. The cap gives up the oldest span on purpose and names it, which
    is the difference between a decision and a silent gap.
    """

    _present_journalctl(monkeypatch)
    state = tmp_path / "state.json"
    stale = NOW - timedelta(hours=2)
    state.write_text(json.dumps({"journal_since": stale.isoformat()}), encoding="utf-8")
    runner = _Journalctl()
    clock = [NOW]
    collector = _collector(tmp_path, runner, clock=clock, state=state)
    clock[0] = NOW + timedelta(seconds=600)
    collected_at = clock[0]
    oldest = collected_at - timedelta(seconds=collector.max_journal_window_seconds)

    batch = collector.collect_once()

    assert runner.argument("--since") == f"@{oldest.timestamp()}", (
        "the window starts at the cap, not at the pinned cursor"
    )
    assert batch.collection_errors == [
        f"journal window capped at {collector.max_journal_window_seconds}s: "
        f"gave up on {stale.isoformat()}..{oldest.isoformat()}",
        "cumulative loss since collector start: journal-window-capped-seconds="
        f"{int((oldest - stale).total_seconds())}",
    ], "the span that was given up on has to be named, and added to the running total"
    assert json.loads(state.read_text())["journal_since"] == collected_at.isoformat(), (
        "a capped window was still read, so the cursor advances"
    )


def test_a_rotated_training_log_is_read_from_the_start_and_reported(
    tmp_path, monkeypatch
) -> None:
    """The name is the same, the file is not.

    Resuming at the old offset inside a fresh file starts mid-line and skips
    everything the rotated-away file still held.
    """

    _present_journalctl(monkeypatch)
    log = tmp_path / "training.log"
    log.write_text(f"{MATCHING} first\n", encoding="utf-8")
    clock = [NOW]
    collector = _collector(
        tmp_path, _Journalctl(), clock=clock, training_log_paths=[str(log)]
    )

    first = collector.collect_once()
    # What logrotate does: a new file is created and moved into place, so the
    # inode changes while the old one is still open. Deleting and recreating in
    # place would not do, because the filesystem may hand back the same inode.
    replacement = tmp_path / "training.log.next"
    replacement.write_text(f"{MATCHING} after rotation\n", encoding="utf-8")
    os.replace(replacement, log)
    clock[0] = NOW + timedelta(seconds=10)
    second = collector.collect_once()

    assert first.entries == [], (
        "a file seen for the first time is tailed, so its history is not resent"
    )
    assert [item.message for item in second.entries] == [
        f"{MATCHING} after rotation"
    ], "the replacement file is read from its first byte"
    assert len(second.collection_errors) == 2, "one report and one running total"
    assert "was rotated" in second.collection_errors[0], (
        "a rotation is a gap in what was collected, so it is reported"
    )
    assert second.collection_errors[1] == (
        "cumulative loss since collector start: rotated-training-logs=1"
    ), "the total is what says whether this file has been rotating all day"


def test_a_truncated_training_log_is_re_read_from_the_start(
    tmp_path, monkeypatch
) -> None:
    """Same file, fewer bytes: the offset now points past the end."""

    _present_journalctl(monkeypatch)
    log = tmp_path / "training.log"
    log.write_text(f"{MATCHING} first\n{MATCHING} second\n", encoding="utf-8")
    clock = [NOW]
    collector = _collector(
        tmp_path, _Journalctl(), clock=clock, training_log_paths=[str(log)]
    )

    collector.collect_once()
    with log.open("w", encoding="utf-8") as stream:
        stream.write(f"{MATCHING} restarted\n")
    clock[0] = NOW + timedelta(seconds=10)
    batch = collector.collect_once()

    assert [item.message for item in batch.entries] == [f"{MATCHING} restarted"], (
        "a truncated file is read from the start rather than skipped"
    )
    assert any("was truncated" in item for item in batch.collection_errors), (
        "the batch says the file it was reading went backwards"
    )


def test_the_offset_only_state_of_a_running_agent_is_adopted(
    tmp_path, monkeypatch
) -> None:
    """An upgrade must not resend, nor skip, what the old state already read.

    A live agent has `{path: offset}` on disk with no file identity. Reading it
    as a rotation would replay the file; discarding it would tail and skip
    everything written since the last poll. It is adopted instead, and the new
    state is written under its own key so a rollback tails rather than failing to
    parse the journal cursor along with it.
    """

    _present_journalctl(monkeypatch)
    log = tmp_path / "training.log"
    already_read = f"{MATCHING} old\n"
    log.write_text(already_read + f"{MATCHING} new\n", encoding="utf-8")
    state = tmp_path / "state.json"
    state.write_text(
        json.dumps({"training_log_offsets": {str(log): len(already_read)}}),
        encoding="utf-8",
    )
    clock = [NOW]
    collector = _collector(
        tmp_path, _Journalctl(), clock=clock, state=state, training_log_paths=[str(log)]
    )

    batch = collector.collect_once()

    assert [item.message for item in batch.entries] == [f"{MATCHING} new"], (
        "the legacy offset is where reading resumes"
    )
    assert not [item for item in batch.collection_errors if "rotated" in item], (
        f"a state file without file identity is not a rotation: "
        f"{batch.collection_errors}"
    )
    written = json.loads(state.read_text())["training_log_files"][str(log)]
    assert written["inode"] == log.stat().st_ino, (
        "the identity is recorded now, so the next rotation is detectable"
    )
    assert written["offset"] == log.stat().st_size, "the whole file has been read"

"""What survives a bad line, a bad offset, a rotation and a rejected batch.

Five defects of the same family: the collector reported something it had not
lost, lost something it did not report, or gave up a poll over one line.

* A malformed journal entry raised out of ``collect_once``, which pinned the
  cursor -- so every poll failed on the same line until the 900 s window cap
  stepped over it, taking every entry behind it (F10).
* The collector's own journal lines, which it excludes on purpose, were counted
  as "cumulative loss", and the loss counters were not rolled back with the
  cursor, so a retried poll counted the same loss twice (F11).
* A rotation renamed the file under the glob and the new name was baselined at
  EOF, discarding the tail written between the last poll and the rename -- the
  lines that explain why the log rotated (F4).
* The persisted offsets were ``TextIOWrapper`` cookies, so a stored value could
  land inside a multibyte character and the resume read half a line (F12).
* ``_save_state`` never fsynced, so a crash could leave an unreadable state file
  and the next start read from the five minute default and tailed every training
  log (F14).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pytest

import gpu_fault.collectors.logs.node as node_module
import gpu_fault.collectors.logs.node_sources as node_sources_module
from gpu_fault.collectors.logs.node import NodeLogCollector

from ._support import (
    NOW,
    CollectorError,
    RecordingSink,
    context,
    json,
    os,
    subprocess,
    timedelta,
)

MATCHING = "machine check hardware error"


def _journal_line(cursor: str, message: str, at: Any, unit: str | None = None) -> str:
    item = {
        "__CURSOR": cursor,
        "__REALTIME_TIMESTAMP": str(int(at.timestamp() * 1_000_000)),
        "MESSAGE": message,
        "_TRANSPORT": "journal",
    }
    if unit is not None:
        item["_SYSTEMD_UNIT"] = unit
    return json.dumps(item)


class _Journalctl:
    """A buffered ``journalctl`` stand-in that records its arguments."""

    def __init__(
        self, *, stdout: str = "", stderr: str = "", returncode: int = 0
    ) -> None:
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode
        self.calls: list[list[str]] = []

    def __call__(self, command: Any, **_kwargs: object) -> subprocess.CompletedProcess:
        self.calls.append(list(command))
        return subprocess.CompletedProcess(
            list(command), self.returncode, stdout=self.stdout, stderr=self.stderr
        )

    def argument(self, name: str) -> str | None:
        for index, item in enumerate(self.calls[-1]):
            if item == name:
                return self.calls[-1][index + 1]
        return None


class _RejectsTheFirstBatchWithEntries:
    """A control plane that rejected the batch: it went nowhere (not buffered)."""

    def __init__(self) -> None:
        self.requests: list[tuple[str, dict[str, Any]]] = []
        self.rejected = 0

    def post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        self.requests.append((path, payload))
        if payload["entries"] and not self.rejected:
            self.rejected += 1
            raise CollectorError("rejected (422)", status_code=422)
        return {"accepted": True}


def _present_journalctl(monkeypatch: Any) -> None:
    monkeypatch.setattr(
        node_module.shutil, "which", lambda _name: "/usr/bin/journalctl"
    )


def _absent_journalctl(monkeypatch: Any) -> None:
    """No journal on this node, so a case about files is only about files."""

    monkeypatch.setattr(node_module.shutil, "which", lambda _name: None)


def _seeded_state(tmp_path: Path) -> Path:
    state = tmp_path / "state.json"
    state.write_text(
        json.dumps({"journal_since": (NOW - timedelta(seconds=30)).isoformat()}),
        encoding="utf-8",
    )
    return state


def _collector(
    state: Path,
    *,
    sink: Any = None,
    runner: Any = None,
    training_log_paths: list[str] | None = None,
    **overrides: Any,
) -> NodeLogCollector:
    return NodeLogCollector(
        sink if sink is not None else RecordingSink(),
        context(),
        node_id="worker-1",
        training_log_paths=training_log_paths or [],
        state_path=str(state),
        now=lambda: NOW,
        runner=runner or _Journalctl(),
        **overrides,
    )


def _write_file_state(state: Path, log: Path, *, offset: int) -> None:
    stat = log.stat()
    state.write_text(
        json.dumps(
            {
                "journal_since": (NOW - timedelta(seconds=30)).isoformat(),
                "training_log_files": {
                    str(log): {
                        "device": stat.st_dev,
                        "inode": stat.st_ino,
                        "offset": offset,
                    }
                },
            }
        ),
        encoding="utf-8",
    )


def test_one_malformed_journal_entry_does_not_stop_the_poll(
    tmp_path, monkeypatch
) -> None:
    """A bare ``1`` is valid JSON and not an entry, and it used to be fatal.

    ``item.get`` on an int raises ``AttributeError`` out of ``collect_once``,
    which rolls the cursor back: the next poll asks for the same window, reads
    the same line and fails again, so nothing behind it is ever delivered until
    the 900 s cap steps over the lot.
    """

    _present_journalctl(monkeypatch)
    state = _seeded_state(tmp_path)
    runner = _Journalctl(
        stdout="\n".join(
            [
                "1",
                "{not json",
                _journal_line("c-1", MATCHING, NOW - timedelta(seconds=10)),
            ]
        )
    )
    collector = _collector(state, runner=runner)

    batch = collector.collect_once()

    assert [item.entry_id for item in batch.entries] == ["c-1"], (
        f"the entries behind the malformed line were lost: {batch.entries}"
    )
    assert json.loads(state.read_text())["journal_since"] == NOW.isoformat(), (
        "a malformed entry that is counted has been dealt with, so the cursor moves"
    )
    assert any("could not be parsed" in item for item in batch.collection_errors), (
        f"a skipped entry is loss and has to be reported: {batch.collection_errors}"
    )
    assert batch.collection_errors[-1].endswith("unparseable-journal-entries=2"), (
        f"both malformed lines are counted, not silently dropped: "
        f"{batch.collection_errors}"
    )


def test_the_collectors_own_journal_lines_are_not_counted_as_loss(
    tmp_path, monkeypatch
) -> None:
    """Excluding our own complaints is a policy, not a gap in the evidence.

    They were added to the same per-reason totals as dropped entries, so the
    running "cumulative loss" of a healthy node grew by one for every warning
    this system wrote about itself -- and an operator reading the batch could not
    tell that from evidence actually going missing.
    """

    _present_journalctl(monkeypatch)
    state = _seeded_state(tmp_path)
    runner = _Journalctl(
        stdout=_journal_line(
            "c-1",
            f"node log collection incomplete: {MATCHING}",
            NOW,
            "gpu-fault-log-collector.service",
        ),
        returncode=1,
        stderr="Failed to open journal\n",
    )
    collector = _collector(state, runner=runner)

    batch = collector.collect_once()

    assert batch.entries == [], "the collector's own line is not node evidence"
    assert batch.collection_errors[-1] == (
        "cumulative loss since collector start: journalctl-failures=1"
    ), f"an excluded unit is not loss: {batch.collection_errors}"
    assert collector.self_unit_entries_total == 1, (
        "the exclusions still need a count of their own, separate from loss"
    )


def test_a_rejected_batch_rolls_back_the_cursor_and_the_loss_counters(
    tmp_path, monkeypatch
) -> None:
    """A batch that went nowhere must leave the poll exactly as it found it.

    The cursor and the file offsets were rolled back but ``_discarded`` was not,
    so the retried poll -- which re-reads the same window and re-detects the same
    rotation -- added the same loss a second time, and the totals said the node
    was losing evidence twice as fast as it was.

    A buffered batch is the opposite case (the outbox has it, so the cursor
    advances); that one is covered in ``test_delivery_result.py``.
    """

    _absent_journalctl(monkeypatch)
    state = _seeded_state(tmp_path)
    log = tmp_path / "training.log"
    log.write_text(f"{MATCHING} before\n", encoding="utf-8")
    sink = _RejectsTheFirstBatchWithEntries()
    collector = _collector(
        state, sink=sink, training_log_paths=[str(log)], context_lines=0
    )

    collector.collect_once()
    replacement = tmp_path / "training.log.next"
    replacement.write_text(f"{MATCHING} after rotation\n", encoding="utf-8")
    os.replace(replacement, log)
    with pytest.raises(CollectorError, match="rejected"):
        collector.collect_once()
    recovered = collector.collect_once()

    assert [item.message for item in recovered.entries] == [
        f"{MATCHING} after rotation"
    ], f"the rejected batch was not re-read: {recovered.entries}"
    assert recovered.collection_errors[-1] == (
        "cumulative loss since collector start: rotated-training-logs=1"
    ), f"the one rotation was counted twice: {recovered.collection_errors}"


def test_a_rejected_batch_asks_for_the_same_journal_window_again(
    tmp_path, monkeypatch
) -> None:
    """The cursor half of the same rollback, read through the next command."""

    _present_journalctl(monkeypatch)
    state = _seeded_state(tmp_path)
    since = NOW - timedelta(seconds=30)
    runner = _Journalctl(stdout=_journal_line("c-1", MATCHING, NOW))
    sink = _RejectsTheFirstBatchWithEntries()
    collector = _collector(state, sink=sink, runner=runner)

    with pytest.raises(CollectorError, match="rejected"):
        collector.collect_once()
    retried = collector.collect_once()

    assert runner.argument("--since") == f"@{since.timestamp()}", (
        f"the rejected window has to be asked for again: {runner.calls[-1]}"
    )
    assert [item.entry_id for item in retried.entries] == ["c-1"], (
        f"the rejected entry was dropped instead of re-read: {retried.entries}"
    )


def test_the_state_file_is_fsynced_with_its_directory(tmp_path, monkeypatch) -> None:
    """A state file that survives the crash it was written for.

    Without the two fsyncs a power loss can leave a truncated file: the next
    start then has no cursor (five minute default, reported as a gap) and tails
    every training log, losing whatever was written meanwhile.
    """

    _absent_journalctl(monkeypatch)
    state = _seeded_state(tmp_path)
    fsynced: list[int] = []
    monkeypatch.setattr(node_module.os, "fsync", fsynced.append)
    collector = _collector(state)

    collector.collect_once()

    assert len(fsynced) == 2, (
        f"the state file and its directory both need an fsync: {fsynced}"
    )


def test_a_renamed_training_log_keeps_the_tail_written_before_the_rotation(
    tmp_path, monkeypatch
) -> None:
    """logrotate renames the file; the inode reappears under a new name.

    That name has no offset of its own, so it was baselined at EOF and every
    line appended between the last poll and the rename was discarded -- exactly
    the lines that made the log rotate. The offset is inherited by
    ``(st_dev, st_ino)`` instead, and it has to be resolved before anything is
    read, because the old name is glob-sorted first and its record is about to
    be overwritten with the replacement's identity.
    """

    _absent_journalctl(monkeypatch)
    state = _seeded_state(tmp_path)
    log = tmp_path / "training.log"
    log.write_text("step 0 loss 0.5\n", encoding="utf-8")
    paths = [str(tmp_path / "training.log*")]

    _collector(state, training_log_paths=paths).collect_once()
    with log.open("a", encoding="utf-8") as stream:
        stream.write(f"{MATCHING} written before the rotation\n")
    log.rename(tmp_path / "training.log.1")
    log.write_text("", encoding="utf-8")
    batch = _collector(state, training_log_paths=paths).collect_once()

    assert [item.message for item in batch.entries] == [
        f"{MATCHING} written before the rotation"
    ], f"the tail written before the rename was baselined away: {batch.entries}"
    stored = json.loads(state.read_text())["training_log_files"]
    assert str(tmp_path / "training.log.1") in stored, (
        f"the rotated file kept no offset of its own: {sorted(stored)}"
    )


def test_a_stored_offset_inside_a_character_resyncs_to_the_next_line(
    tmp_path, monkeypatch, caplog
) -> None:
    """The offsets are byte offsets now, and a legacy cookie is recovered from.

    A ``TextIOWrapper.tell()`` cookie cannot be compared with ``st_size`` and can
    point inside a multibyte character, where ``seek(offset - 1)`` reads a
    replacement character instead of the newline and the line that follows is
    read as half a line. The read re-syncs to the next line boundary, says so
    once per process, and delivers the following line exactly once.
    """

    _absent_journalctl(monkeypatch)
    log = tmp_path / "training.log"
    first = "训练启动 rank 0 ready\n"
    log.write_bytes((first + f"{MATCHING} on cpu 3\n").encode("utf-8"))
    state = tmp_path / "state.json"
    # Two bytes into the first character: a cookie, not a line boundary.
    _write_file_state(state, log, offset=2)
    paths = [str(log)]

    with caplog.at_level(logging.WARNING, logger=node_sources_module.LOGGER.name):
        batch = _collector(state, training_log_paths=paths).collect_once()
    repeated = _collector(state, training_log_paths=paths).collect_once()

    assert [item.message for item in batch.entries] == [f"{MATCHING} on cpu 3"], (
        f"the line after the resync was lost or torn: {batch.entries}"
    )
    assert "line boundary" in caplog.text, (
        "resuming inside a line has to be said once, not silently repaired"
    )
    assert json.loads(state.read_text())["training_log_files"][str(log)]["offset"] == (
        log.stat().st_size
    ), "the committed offset is not a byte count"
    assert repeated.entries == [], f"the same line was read twice: {repeated.entries}"


def test_a_line_the_job_is_still_writing_is_not_delivered_half(
    tmp_path, monkeypatch
) -> None:
    """A trailing line without its terminator is not a line yet.

    Consuming it committed an offset inside an incomplete record, so the rest of
    that line -- the half that says which rank aborted -- was never read.
    """

    _absent_journalctl(monkeypatch)
    log = tmp_path / "training.log"
    complete = f"{MATCHING} first\n"
    log.write_text(complete + f"{MATCHING} second half w", encoding="utf-8")
    state = tmp_path / "state.json"
    _write_file_state(state, log, offset=0)
    paths = [str(log)]

    first = _collector(state, training_log_paths=paths).collect_once()
    held = json.loads(state.read_text())["training_log_files"][str(log)]["offset"]
    with log.open("a", encoding="utf-8") as stream:
        stream.write("ritten later\n")
    second = _collector(state, training_log_paths=paths).collect_once()

    assert [item.message for item in first.entries] == [f"{MATCHING} first"], (
        f"the unfinished line must wait for its terminator: {first.entries}"
    )
    assert held == len(complete.encode("utf-8")), (
        "the offset was committed inside a line nobody had finished writing"
    )
    assert [item.message for item in second.entries] == [
        f"{MATCHING} second half written later"
    ], f"the completed line was delivered whole, once: {second.entries}"


def test_a_resume_offset_inside_an_unfinished_line_holds_and_says_so(
    tmp_path, monkeypatch, caplog
) -> None:
    """Holding the offset is right; doing it silently is not.

    A stored offset can land inside a line that never got its terminator -- a
    killed job, a truncated write. Nothing can be read from that file until the
    line completes, and a collector that says nothing about it looks exactly like
    a healthy one reading a quiet file.
    """

    _absent_journalctl(monkeypatch)
    log = tmp_path / "training.log"
    complete = f"{MATCHING} first\n"
    log.write_text(complete + "partial line the job never fini", encoding="utf-8")
    state = tmp_path / "state.json"
    resume = len(complete.encode("utf-8")) + 3
    _write_file_state(state, log, offset=resume)
    paths = [str(log)]

    with caplog.at_level(logging.WARNING, logger=node_sources_module.LOGGER.name):
        batch = _collector(state, training_log_paths=paths).collect_once()

    assert batch.entries == [], f"half a line is not evidence: {batch.entries}"
    assert "has not finished writing" in caplog.text, (
        "a file nothing can be read from has to be reported"
    )
    assert json.loads(state.read_text())["training_log_files"][str(log)]["offset"] == (
        resume
    ), "the offset moved past a line that was never complete"


def test_a_new_training_log_with_an_unfinished_tail_records_its_position(
    tmp_path, monkeypatch
) -> None:
    """A file that yields nothing this poll must still be remembered.

    With ``initial_tail_bytes=0`` a newly discovered log resumes at its size,
    which is normally inside the line the job is still writing: the read
    correctly delivers nothing, but it also used to persist nothing, so the next
    poll saw the file as new again and re-baselined at the *larger* size --
    everything appended in between was skipped, with no discard counted and no
    collection error, only a rate-limited warning. Recording the position is
    what makes "nothing readable yet" different from "nothing there".
    """

    _absent_journalctl(monkeypatch)
    state = _seeded_state(tmp_path)
    log = tmp_path / "rank0.log"
    log.write_text(f"{MATCHING} first\nhalf a li", encoding="utf-8")
    held = log.stat().st_size
    collector = _collector(state, training_log_paths=[str(log)])

    first = collector.collect_once()

    assert first.entries == [], (
        f"an unfinished line must not be delivered half: {first.entries}"
    )
    assert json.loads(state.read_text())["training_log_files"][str(log)]["offset"] == (
        held
    ), "the position of a file that yielded nothing was not recorded"

    with log.open("a", encoding="utf-8") as handle:
        handle.write(f"ne finished\n{MATCHING} second\n")

    second = collector.collect_once()

    assert any("second" in entry.message for entry in second.entries), (
        f"the entry written after the unfinished line was skipped: {second.entries}"
    )


def test_a_training_log_that_could_not_be_opened_is_not_baselined_at_its_end(
    tmp_path, monkeypatch
) -> None:
    """An unreadable file that is not in the offset table yet must still be placed.

    The read error left the position alone, which is right for a file already
    tracked and silent loss for one that is not: with no entry in the table the
    next poll calls it new and baselines it at its *current* size, so everything
    appended while it could not be read (0600 until the job fixes its umask, an
    SELinux denial, an NFS EIO) is skipped and never counted. The stat succeeded,
    so where it stood is known and is recorded.
    """

    _absent_journalctl(monkeypatch)
    log = tmp_path / "rank0.log"
    log.write_text("starting up\n", encoding="utf-8")
    state = tmp_path / "state.json"
    paths = [str(log)]
    log.chmod(0o000)

    blocked = _collector(state, training_log_paths=paths).collect_once()

    log.chmod(0o644)
    with log.open("a", encoding="utf-8") as handle:
        handle.write(f"{MATCHING} appended after the denial\n")

    recovered = _collector(state, training_log_paths=paths).collect_once()

    assert any("could not be read" in item for item in blocked.collection_errors), (
        f"the unreadable file was not reported: {blocked.collection_errors}"
    )
    assert [item.message for item in recovered.entries] == [
        f"{MATCHING} appended after the denial"
    ], f"the lines appended while the file was unreadable were skipped: {recovered}"


def test_an_unreadable_training_log_does_not_lose_the_journal_entries(
    tmp_path, monkeypatch, caplog
) -> None:
    """A file this collector may not open costs its own lines, nothing else.

    ``stat`` was guarded and ``open`` was not, so a 0600 training log -- or one
    rotated away between the stat and the open -- raised ``OSError`` out of
    ``collect_once``, which rolls the cursor back: the journal entries of the
    same poll were dropped and the next poll failed on the same file, for ever.
    """

    _present_journalctl(monkeypatch)
    state = _seeded_state(tmp_path)
    log = tmp_path / "rank0.log"
    log.write_text(f"{MATCHING} unreadable\n", encoding="utf-8")
    _write_file_state(state, log, offset=0)
    log.chmod(0o000)
    runner = _Journalctl(stdout=_journal_line("c-1", MATCHING, NOW) + "\n")
    collector = _collector(state, runner=runner, training_log_paths=[str(log)])

    with caplog.at_level(logging.WARNING, logger=node_sources_module.LOGGER.name):
        batch = collector.collect_once()

    assert [entry.source for entry in batch.entries] == ["journal"], (
        f"one unreadable file cost the poll its journal entries: {batch.entries}"
    )
    assert any(
        "unreadable-training-logs=1" in item for item in batch.collection_errors
    ), f"a file that could not be read must be counted: {batch.collection_errors}"
    assert any(str(log) in record.message for record in caplog.records), (
        "the unreadable path was never named in the log"
    )

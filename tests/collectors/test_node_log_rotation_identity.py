"""Training-log rotation cases decided by file identity, not by name.

``logrotate`` renames a file, a job hard-links one, an operator truncates one,
and the kernel hands a freed inode number to the next file created. Each case
has to come out as either "the tail is read under its new name" or "this is
loss", and a mid-line resume offset is reported once rather than per line or
per poll.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import pytest

import gpu_fault.collectors.logs.node as node_module
import gpu_fault.collectors.logs.node_sources as node_sources_module
from gpu_fault.collectors.logs.node import NodeLogCollector
from tests.collectors._support import NOW, RecordingSink, context, timedelta

MATCHING = "machine check hardware error"


@pytest.fixture(autouse=True)
def _no_journal(monkeypatch: pytest.MonkeyPatch) -> None:
    """These cases are about files; a node without a journal keeps them so."""

    monkeypatch.setattr(node_module.shutil, "which", lambda _name: None)


def _state(tmp_path: Path, files: dict[str, dict[str, int]] | None = None) -> Path:
    state = tmp_path / "state.json"
    document: dict[str, Any] = {
        "journal_since": (NOW - timedelta(seconds=30)).isoformat()
    }
    if files is not None:
        document["training_log_files"] = files
    state.write_text(json.dumps(document), encoding="utf-8")
    return state


def _identity(path: Path, *, offset: int) -> dict[str, int]:
    stat = path.stat()
    return {"device": stat.st_dev, "inode": stat.st_ino, "offset": offset}


def _collector(state: Path, paths: list[str], **overrides: Any) -> NodeLogCollector:
    return NodeLogCollector(
        RecordingSink(),
        context(),
        node_id="worker-1",
        training_log_paths=paths,
        state_path=str(state),
        now=lambda: NOW,
        runner=lambda *_args, **_kwargs: None,
        **overrides,
    )


def _stored(state: Path) -> dict[str, dict[str, int]]:
    return json.loads(state.read_text())["training_log_files"]


def test_a_rename_seen_from_the_old_name_first_is_not_counted_as_loss(
    tmp_path: Path,
) -> None:
    """The old name is resolved before the new one: its inode is found under the
    rotated name, so the rotation is reported as a rename and the replacement's
    tail is still inherited by identity."""

    state = _state(tmp_path)
    log = tmp_path / "training.log"
    log.write_text("step 0\n", encoding="utf-8")
    # A path that sorts after every rotated name makes the next poll start at
    # ``training.log`` again instead of at ``training.log.1``.
    tail = tmp_path / "zz-other.log"
    tail.write_text("other 0\n", encoding="utf-8")
    paths = [str(tmp_path / "training.log*"), str(tail)]
    _collector(state, paths).collect_once()

    with log.open("a", encoding="utf-8") as stream:
        stream.write(f"{MATCHING} before the rename\n")
    log.rename(tmp_path / "training.log.1")
    log.write_text(f"{MATCHING} in the replacement\n", encoding="utf-8")
    batch = _collector(state, paths).collect_once()

    assert sorted(item.message for item in batch.entries) == [
        f"{MATCHING} before the rename",
        f"{MATCHING} in the replacement",
    ], f"a rename lost lines on one side of it: {batch.entries}"
    assert not any(
        "rotated-training-logs" in error for error in batch.collection_errors
    ), f"a rename whose tail was read was reported as loss: {batch.collection_errors}"
    assert any(
        "was rotated to" in error and "training.log.1" in error
        for error in batch.collection_errors
    ), f"the rename was not named: {batch.collection_errors}"
    stored = _stored(state)
    assert (
        stored[str(tmp_path / "training.log.1")]["offset"]
        == (tmp_path / "training.log.1").stat().st_size
    )


def test_a_name_already_tracked_under_its_own_inode_is_not_a_rename_target(
    tmp_path: Path,
) -> None:
    """``b.log`` now carries the inode ``a.log`` had, but ``b.log`` was being read
    as a file of its own: the earlier tracked identity says so, and the match is
    a renamed-over file, which is loss for whatever ``b.log`` held."""

    state = _state(tmp_path)
    first = tmp_path / "a.log"
    second = tmp_path / "b.log"
    first.write_text("a 0\n", encoding="utf-8")
    second.write_text("b 0\n", encoding="utf-8")
    paths = [str(first), str(second)]
    _collector(state, paths).collect_once()

    with first.open("a", encoding="utf-8") as stream:
        stream.write(f"{MATCHING} written to a\n")
    os.replace(first, second)
    first.write_text("a replacement\n", encoding="utf-8")
    batch = _collector(state, paths).collect_once()

    assert any("rotated-training-logs" in error for error in batch.collection_errors), (
        f"a renamed-over file was not reported as loss: {batch.collection_errors}"
    )
    assert not any("was rotated to" in error for error in batch.collection_errors), (
        f"a tracked name was accepted as a rename target: {batch.collection_errors}"
    )


def test_a_hard_link_does_not_take_the_offset_of_the_name_it_shares_an_inode_with(
    tmp_path: Path,
) -> None:
    state = _state(tmp_path)
    log = tmp_path / "a.log"
    log.write_text("a 0\n", encoding="utf-8")
    paths = [str(tmp_path / "*.log")]
    _collector(state, paths).collect_once()

    with log.open("a", encoding="utf-8") as stream:
        stream.write(f"{MATCHING} appended once\n")
    os.link(log, tmp_path / "b.log")
    batch = _collector(state, paths).collect_once()

    assert [item.message for item in batch.entries] == [f"{MATCHING} appended once"], (
        f"one inode under two names was read twice or not at all: {batch.entries}"
    )
    stored = _stored(state)
    assert stored[str(log)]["offset"] == log.stat().st_size
    assert stored[str(tmp_path / "b.log")]["offset"] == log.stat().st_size, (
        "the second name of the same inode was not baselined at its end"
    )


def test_a_shorter_file_with_a_recycled_inode_number_does_not_inherit_an_offset(
    tmp_path: Path,
) -> None:
    """A real rename still holds every byte that was read from it; one that is
    shorter than the recorded offset is a different file given the same number,
    so seeking to the old offset would skip its first lines."""

    state = _state(tmp_path)
    old = tmp_path / "a.log"
    new = tmp_path / "b.log"
    old.write_text("a 0\n" * 20, encoding="utf-8")
    paths = [str(tmp_path / "*.log")]
    _collector(state, paths).collect_once()

    os.replace(old, new)
    new.write_text(f"{MATCHING} short\n", encoding="utf-8")
    batch = _collector(state, paths).collect_once()

    assert batch.entries == [], (
        f"a file not yet polled under its name was tailed from the start: "
        f"{batch.entries}"
    )
    stored = _stored(state)
    assert stored[str(new)]["offset"] == new.stat().st_size
    assert str(old) not in stored, "a vanished path kept its offset record"


def test_a_rename_with_no_replacement_moves_the_offset_record_to_the_new_name(
    tmp_path: Path,
) -> None:
    state = _state(tmp_path)
    old = tmp_path / "a.log"
    new = tmp_path / "b.log"
    old.write_text("a 0\n", encoding="utf-8")
    paths = [str(tmp_path / "*.log")]
    _collector(state, paths).collect_once()

    with old.open("a", encoding="utf-8") as stream:
        stream.write(f"{MATCHING} before the move\n")
    os.replace(old, new)
    batch = _collector(state, paths).collect_once()

    assert [item.message for item in batch.entries] == [f"{MATCHING} before the move"]
    stored = _stored(state)
    assert set(stored) == {str(new)}, (
        f"the old name kept a record although nothing replaced it: {sorted(stored)}"
    )


def test_a_mid_line_resume_is_warned_about_once_per_process(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    first = tmp_path / "a.log"
    second = tmp_path / "b.log"
    for log in (first, second):
        log.write_text(f"{MATCHING} line one\n{MATCHING} line two\n", encoding="utf-8")
    state = _state(
        tmp_path,
        {
            str(first): _identity(first, offset=3),
            str(second): _identity(second, offset=3),
        },
    )

    with caplog.at_level(logging.WARNING, logger=node_sources_module.LOGGER.name):
        batch = _collector(state, [str(first), str(second)]).collect_once()

    assert [item.message for item in batch.entries] == [
        f"{MATCHING} line two",
        f"{MATCHING} line two",
    ], f"a resume inside a line did not re-sync to the next one: {batch.entries}"
    resyncs = [
        record for record in caplog.records if "not a line boundary" in record.message
    ]
    assert len(resyncs) == 1, (
        f"the re-sync warning fired once per file instead of once: {len(resyncs)}"
    )


def test_a_resume_inside_an_unfinished_line_is_not_reported_on_every_poll(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    log = tmp_path / "a.log"
    complete = f"{MATCHING} first\n"
    log.write_text(complete + "a line the job has not fini", encoding="utf-8")
    resume = len(complete.encode("utf-8")) + 3
    state = _state(tmp_path, {str(log): _identity(log, offset=resume)})
    collector = _collector(state, [str(log)])

    with caplog.at_level(logging.WARNING, logger=node_sources_module.LOGGER.name):
        first = collector.collect_once()
        second = collector.collect_once()

    assert first.entries == [] and second.entries == []
    warnings = [
        record
        for record in caplog.records
        if "has not finished writing" in record.message
    ]
    assert len(warnings) == 1, (
        f"the same stalled line was reported on every poll: {len(warnings)}"
    )
    assert _stored(state)[str(log)]["offset"] == resume

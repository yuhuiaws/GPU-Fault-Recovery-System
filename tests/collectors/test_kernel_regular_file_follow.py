"""A regular ``GPU_FAULT_KMSG_PATH`` is followed like ``tail -f``, not reopened.

``/dev/kmsg`` blocks in ``select`` until the kernel writes the next record and
never reports EOF. A regular file -- the kernel source replay's reviewed
fixture -- is always "ready" and reports EOF at once, and the collector used
to treat that EOF as an ended stream: it closed the file, slept the reopen
delay, reopened it and sought to its end again, so a record appended between
two reopens was skipped. Live (release R13g, 2026-10-26) every kmsg-mode replay
on the plain EKS cluster -- 024 ``independent_xid`` (XID 92), 032
``unknown_xid`` (XID 987) and 032 ``missing_topology`` (SXID 11001) -- wrote its
reviewed line into the fixture the collector was pointed at and no XID/SXID
event ever reached the control plane. These tests drive the real reader over a
real file the way the probe's ``arm`` writes it.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from gpu_fault.channel_registry import NVIDIA_KERNEL_PATH
from gpu_fault.collectors.logs.kernel import KernelLogCollector
from tests.collectors._support import RecordingSink, context

POLL_SECONDS = 0.02
DEADLINE_SECONDS = 5.0


def _record(sequence: int, message: str) -> str:
    return f"3,{sequence},{int(time.monotonic() * 1_000_000)},-;{message}\n"


def _append(path: Path, text: str) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())


def _events(sink: RecordingSink) -> list[dict[str, Any]]:
    return [payload for path, payload in sink.requests if path == NVIDIA_KERNEL_PATH]


def _wait_for(predicate: Any, *, what: str) -> None:
    deadline = time.monotonic() + DEADLINE_SECONDS
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(POLL_SECONDS)
    raise AssertionError(f"timed out waiting for {what}")


class _Reader:
    """The collector's ``run`` on its own thread over a real fixture file."""

    def __init__(self, fixture: Path, **options: Any) -> None:
        self.sink = RecordingSink()
        self.collector = KernelLogCollector(
            self.sink,
            context(),
            node_id="worker-1",
            boot_id="boot-123",
            kmsg_path=str(fixture),
            reopen_delay_seconds=POLL_SECONDS,
            **options,
        )
        # Keep the periodic summary out of the way; the start-up one is posted.
        self.collector.health_summary_seconds = 3600
        self.thread = threading.Thread(target=self.collector.run, daemon=True)

    def __enter__(self) -> _Reader:
        self.thread.start()
        _wait_for(
            lambda: any("summary_id" in p for _p, p in self.sink.requests),
            what="the collector to open the fixture and post its start-up summary",
        )
        return self

    def __exit__(self, *_args: Any) -> None:
        self.collector._shutdown.set()
        self.thread.join(timeout=DEADLINE_SECONDS)
        assert not self.thread.is_alive(), "the reader did not stop"


def test_a_record_appended_to_a_regular_file_is_read_without_a_reopen(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The live sequence: open-window (empty fixture, collector restarted on it),
    then ``arm`` appends the reviewed line once a second, a few times."""

    fixture = tmp_path / "kmsg.fixture"
    fixture.write_text(
        _record(
            7, "NVRM: Xid (PCI:0000:53:00): 31, written before the collector started"
        )
    )
    caplog.set_level(logging.WARNING, logger="gpu_fault.collectors.logs.kernel")
    with _Reader(fixture) as reader:
        for _attempt in range(3):
            _append(
                fixture,
                _record(
                    900_000_123,
                    "NVRM: Xid (PCI:0000:53:00): 987, acceptance unknown-code replay",
                ),
            )
            time.sleep(5 * POLL_SECONDS)
        _wait_for(lambda: _events(reader.sink), what="the appended record to be posted")
        time.sleep(5 * POLL_SECONDS)
    events = _events(reader.sink)
    assert [event["record_id"] for event in events] == ["kmsg-boot-123-900000123"], (
        "exactly one event: the appended line, deduped across its copies, and not "
        f"the line that was in the file before the collector started: {events}"
    )
    assert events[0]["message"].endswith("acceptance unknown-code replay"), events
    assert "stream ended" not in caplog.text, (
        "a regular file at its end is followed, not reopened at its new end "
        f"(every reopen skips what was appended meanwhile): {caplog.text}"
    )


def test_a_line_still_being_written_is_held_until_its_newline_arrives(
    tmp_path: Path,
) -> None:
    fixture = tmp_path / "kmsg.fixture"
    fixture.write_text("")
    with _Reader(fixture) as reader:
        head = f"3,900000200,{int(time.monotonic() * 1_000_000)},-;NVRM: Xid (PCI:0000:53:00): 9"
        _append(fixture, head)
        time.sleep(5 * POLL_SECONDS)
        assert _events(reader.sink) == [], (
            "a partial line (no newline yet) must not be parsed as a record"
        )
        _append(fixture, "87, the rest of the line\n")
        _wait_for(lambda: _events(reader.sink), what="the completed line to be posted")
    events = _events(reader.sink)
    assert len(events) == 1, events
    assert events[0]["record_id"] == "kmsg-boot-123-900000200", events
    assert events[0]["message"] == (
        "NVRM: Xid (PCI:0000:53:00): 987, the rest of the line"
    ), "the two halves are one record, not a truncated one plus a skipped tail"


def test_a_replaced_file_is_reopened_and_followed_again(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    fixture = tmp_path / "kmsg.fixture"
    fixture.write_text("")
    caplog.set_level(logging.WARNING, logger="gpu_fault.collectors.logs.kernel")
    with _Reader(fixture) as reader:
        _append(
            fixture, _record(900_000_300, "NVRM: Xid (PCI:0000:53:00): 987, first file")
        )
        _wait_for(lambda: _events(reader.sink), what="the first record")
        replacement = tmp_path / "kmsg.fixture.new"
        replacement.write_text(
            _record(
                900_000_301, "NVRM: Xid (PCI:0000:53:00): 987, already in the new file"
            )
        )
        os.replace(replacement, fixture)
        _wait_for(
            lambda: "truncated or replaced" in caplog.text,
            what="the collector to notice the replaced file",
        )
        _wait_for(
            lambda: "stream ended" in caplog.text,
            what="the reader to reopen the new file",
        )
        time.sleep(5 * POLL_SECONDS)
        _append(
            fixture,
            _record(
                900_000_302, "NVRM: Xid (PCI:0000:53:00): 987, appended to the new file"
            ),
        )
        _wait_for(
            lambda: len(_events(reader.sink)) >= 2,
            what="the record appended to the new file",
        )
    sequences = [event["record_id"] for event in _events(reader.sink)]
    assert sequences == ["kmsg-boot-123-900000300", "kmsg-boot-123-900000302"], (
        "the new file is opened at its end like every open (its backlog is not "
        f"replayed) and then followed: {sequences}"
    )


def test_a_truncated_file_is_reopened(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    fixture = tmp_path / "kmsg.fixture"
    fixture.write_text("")
    caplog.set_level(logging.WARNING, logger="gpu_fault.collectors.logs.kernel")
    with _Reader(fixture) as reader:
        _append(
            fixture,
            _record(900_000_400, "NVRM: Xid (PCI:0000:53:00): 987, before truncation"),
        )
        _wait_for(lambda: _events(reader.sink), what="the first record")
        os.truncate(fixture, 0)
        _wait_for(
            lambda: "truncated or replaced" in caplog.text,
            what="the collector to notice the truncation",
        )
        _wait_for(lambda: "stream ended" in caplog.text, what="the reopen")
        time.sleep(5 * POLL_SECONDS)
        _append(
            fixture,
            _record(900_000_401, "NVRM: Xid (PCI:0000:53:00): 987, after truncation"),
        )
        _wait_for(
            lambda: len(_events(reader.sink)) >= 2, what="the record after truncation"
        )
    assert [event["record_id"] for event in _events(reader.sink)] == [
        "kmsg-boot-123-900000400",
        "kmsg-boot-123-900000401",
    ]


def test_the_start_of_a_regular_file_is_read_when_not_starting_at_the_end(
    tmp_path: Path,
) -> None:
    """Replay mode (``start_at_end=False``) still drains the file's content
    first and then keeps following it; nothing is read twice."""

    fixture = tmp_path / "kmsg.fixture"
    fixture.write_text(_record(900_000_500, "NVRM: Xid (PCI:0000:53:00): 987, backlog"))
    with _Reader(fixture, start_at_end=False) as reader:
        _wait_for(lambda: _events(reader.sink), what="the backlog record")
        _append(fixture, _record(900_000_501, "NVRM: Xid (PCI:0000:53:00): 987, live"))
        _wait_for(lambda: len(_events(reader.sink)) >= 2, what="the live record")
        time.sleep(5 * POLL_SECONDS)
    assert [event["record_id"] for event in _events(reader.sink)] == [
        "kmsg-boot-123-900000500",
        "kmsg-boot-123-900000501",
    ]

"""journalctl is positioned by its cursor or by a time window, never both.

systemd's ``journalctl`` refuses ``--since`` next to ``--cursor``/``--after-cursor``
before it reads anything (``parse_argv`` in ``src/journal/journalctl.c``, v252)::

    Please specify only one of --since=, --cursor=, and --after-cursor=.

The collector passed both as soon as it knew a cursor, so every resumed poll was
refused; the refusal mentions "cursor", was taken for a lost cursor, and the
cursor was dropped -- the next poll read by timestamp, learned a cursor, and was
refused again. Every other poll read nothing (``journalctl-failures=17`` on a
live AL2023 node). These cases drive the collector through ``collect_once``
against a stand-in with systemd 252's argument validation and seek rules, and
pin the positioning rules, the no-gap/no-duplicate resume, and how a failing
journalctl is reported to the control plane.
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

import gpu_fault.collectors.logs.node as node_module
from gpu_fault.collectors.logs.node import NodeLogCollector
from gpu_fault.collectors.logs.node_journal import (
    FAILURE_CLASSES,
    POSITIONING_REFUSAL,
    journal_command,
    journalctl_failure_class,
)

from ._support import RecordingSink, context

NOW = datetime(2026, 10, 26, 6, 0, tzinfo=timezone.utc)
#: The kernel's form of the boot id (``/proc/sys/kernel/random/boot_id``).
BOOT = "51dda8ef-6251-465b-93e0-e16ec1d9ee46"
SEQNUM_ID = "8af3cc0fef5540b8b6da183db30ee343"
MATCHING = "machine check hardware error"
REFUSAL = "Please specify only one of --since=, --cursor=, and --after-cursor=."
CURSOR_REJECTED = "Failed to seek to cursor: Invalid argument"
#: What getopt says for an option the binary does not know.
UNKNOWN_OPTION = "journalctl: unrecognized option '--after-cursor'"
VALUED = {"--since", "--until", "--cursor", "--after-cursor", "--output"}


def record(index: int, at: datetime, message: str = MATCHING) -> dict[str, str]:
    """One journal entry as ``journalctl --output=json`` prints it.

    The cursor has journald's shape (``s=;i=;b=;m=;t=;x=``, see
    ``journalctl --show-cursor``) so the stand-in can apply the same seek rules
    the real binary does.
    """

    realtime = int(at.timestamp() * 1_000_000)
    boot = BOOT.replace("-", "")
    return {
        "__CURSOR": (
            f"s={SEQNUM_ID};i={index:x};b={boot};m={index * 1000:x};"
            f"t={realtime:x};x={index * 7919:x}"
        ),
        "__REALTIME_TIMESTAMP": str(realtime),
        "_TRANSPORT": "kernel",
        "_BOOT_ID": boot,
        "MESSAGE": message,
    }


def options(argv: list[str]) -> dict[str, str]:
    """The options of a journalctl argv, ``--name=value`` and ``--name value`` alike."""

    parsed: dict[str, str] = {}
    items = iter(argv[1:])
    for item in items:
        name, equals, value = item.partition("=")
        parsed[name] = value if equals or name not in VALUED else next(items)
    return parsed


def positioning(argv: list[str]) -> list[str]:
    return [
        name
        for name in ("--since", "--cursor", "--after-cursor")
        if name in options(argv)
    ]


def _microseconds(stamp: str) -> int:
    assert stamp.startswith("@"), f"the collector positions by epoch stamps: {stamp}"
    return int(round(float(stamp[1:]) * 1_000_000))


def _seekable(cursor: str) -> bool:
    """``sd_journal_seek_cursor`` (systemd v252): ``k=hex`` words, one full location.

    A word without ``=`` at index 1 is ``-EINVAL``; unknown keys are ignored;
    the cursor needs seqnum+seqnum id, monotonic+boot id, or a realtime stamp.
    """

    fields: dict[str, str] = {}
    for word in cursor.split(";"):
        if len(word) < 2 or word[1] != "=":
            return False
        fields[word[0]] = word[2:]
    try:
        for key, value in fields.items():
            if key in "sibmtx":
                int(value, 16)
    except ValueError:
        return False
    return {"s", "i"} <= fields.keys() or {"b", "m"} <= fields.keys() or "t" in fields


class Systemd252Journalctl:
    """journalctl as systemd 252 argues it: one positioning option, or nothing is read.

    ``--until`` is not part of that check and bounds both modes. A cursor the
    library cannot parse fails the seek with ``Invalid argument``; a parseable
    cursor whose entry is gone seeks to the closest entry in time and
    ``--after-cursor`` steps past it (``sd_journal_seek_head(3)``). The journal
    is the list of records in journal order; ``failure`` makes every call fail
    with that stderr, ``refuse_after_cursor`` models a binary without the option.
    """

    def __init__(self, records: list[dict[str, str]] | None = None) -> None:
        self.records = list(records or [])
        self.calls: list[list[str]] = []
        self.failure: str | None = None
        self.refuse_after_cursor = False

    def __call__(
        self, argv: list[str], **_kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        argv = list(argv)
        self.calls.append(argv)
        parsed = options(argv)
        if len(positioning(argv)) > 1:
            return self._failed(argv, REFUSAL)
        if self.failure is not None:
            return self._failed(argv, self.failure)
        if "--after-cursor" in parsed:
            if self.refuse_after_cursor:
                return self._failed(argv, UNKNOWN_OPTION)
            start = self._after(parsed["--after-cursor"])
            if start is None:
                return self._failed(argv, CURSOR_REJECTED)
            selected = self.records[start:]
        else:
            since = _microseconds(parsed["--since"])
            selected = [
                item
                for item in self.records
                if int(item["__REALTIME_TIMESTAMP"]) >= since
            ]
        until = _microseconds(parsed["--until"])
        selected = [
            item for item in selected if int(item["__REALTIME_TIMESTAMP"]) <= until
        ]
        return subprocess.CompletedProcess(
            argv, 0, stdout="\n".join(json.dumps(item) for item in selected), stderr=""
        )

    def _after(self, cursor: str) -> int | None:
        if not _seekable(cursor):
            return None
        for index, item in enumerate(self.records):
            if item["__CURSOR"] == cursor:
                return index + 1
        if not self.records:
            return 0
        realtime = int(dict(word.split("=", 1) for word in cursor.split(";"))["t"], 16)
        closest = min(
            range(len(self.records)),
            key=lambda index: abs(
                int(self.records[index]["__REALTIME_TIMESTAMP"]) - realtime
            ),
        )
        return closest + 1

    @staticmethod
    def _failed(argv: list[str], stderr: str) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(argv, 1, stdout="", stderr=stderr + "\n")


def collector(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    journal: Systemd252Journalctl,
    *,
    clock: list[datetime],
    state: dict[str, object] | None = None,
    boot_id: str = BOOT,
    **settings: object,
) -> NodeLogCollector:
    monkeypatch.setattr(
        node_module.shutil, "which", lambda _name: "/usr/bin/journalctl"
    )
    # The periodic slot every 300 s exactly: the cadence jitter is somebody
    # else's case and would make the due times depend on the node id.
    monkeypatch.setattr(
        node_module,
        "jittered_next_phase",
        lambda collected_at, **_kwargs: collected_at + timedelta(seconds=300),
    )
    path = tmp_path / "state.json"
    if state is not None and not path.exists():
        path.write_text(json.dumps(state), encoding="utf-8")
    return NodeLogCollector(
        RecordingSink(),
        context(),
        node_id="node-a",
        boot_id=boot_id,
        now=lambda: clock[0],
        state_path=str(path),
        runner=journal,
        context_lines=0,
        **settings,  # type: ignore[arg-type]
    )


def _state(path: Path) -> dict[str, object]:
    return json.loads((path / "state.json").read_text(encoding="utf-8"))


def _seeded(cursor: str | None, *, boot: object = BOOT) -> dict[str, object]:
    state: dict[str, object] = {
        "journal_since": (NOW - timedelta(seconds=60)).isoformat(),
        "journal_cursor": cursor,
    }
    if boot is not None:
        state["boot_id"] = boot
    return state


# --------------------------------------------------------------------------- #
# positioning
# --------------------------------------------------------------------------- #
def test_a_resumed_poll_is_positioned_by_its_cursor_alone(
    tmp_path, monkeypatch
) -> None:
    """The live defect: the second poll of every pair was refused by journalctl."""

    journal = Systemd252Journalctl(
        [record(1, NOW - timedelta(seconds=20)), record(2, NOW - timedelta(seconds=15))]
    )
    clock = [NOW - timedelta(seconds=10)]
    value = collector(tmp_path, monkeypatch, journal, clock=clock)

    first = value.collect_once()
    journal.records.append(record(3, NOW - timedelta(seconds=5)))
    clock[0] = NOW
    second = value.collect_once()

    assert [item.entry_id for item in first.entries] == [
        journal.records[0]["__CURSOR"],
        journal.records[1]["__CURSOR"],
    ], f"the first read takes the window: {first.entries}"
    assert positioning(journal.calls[0]) == ["--since"], (
        f"with no cursor the first read is a time window: {journal.calls[0]}"
    )
    assert positioning(journal.calls[1]) == ["--after-cursor"], (
        f"a resumed poll is positioned by the cursor and nothing else: {journal.calls[1]}"
    )
    assert (
        options(journal.calls[1])["--after-cursor"] == journal.records[1]["__CURSOR"]
    ), "the resume point is the last entry read"
    assert "--until" in options(journal.calls[1]), (
        "the window end still bounds a resumed read (journalctl allows it)"
    )
    assert second.collection_errors == [], (
        f"systemd 252 must accept the resumed poll: {second.collection_errors}"
    )
    assert [item.entry_id for item in second.entries] == [
        journal.records[2]["__CURSOR"]
    ], f"only what came after the cursor, and all of it: {second.entries}"


def test_no_poll_is_wasted_on_a_refusal_any_more(tmp_path, monkeypatch) -> None:
    """Four polls, four reads: the alternating failure pattern is gone."""

    journal = Systemd252Journalctl([record(1, NOW - timedelta(seconds=30))])
    clock = [NOW]
    value = collector(tmp_path, monkeypatch, journal, clock=clock)

    batches = []
    for step in range(1, 5):
        batches.append(value.collect_once())
        journal.records.append(record(step + 1, clock[0] + timedelta(seconds=5)))
        clock[0] += timedelta(seconds=10)

    assert all(len(positioning(call)) == 1 for call in journal.calls), (
        f"every invocation carries exactly one positioning option: {journal.calls}"
    )
    assert not any(
        "journalctl" in error for batch in batches for error in batch.collection_errors
    ), f"nothing about journalctl is left to report: {batches}"
    delivered = [item.entry_id for batch in batches for item in batch.entries]
    assert delivered == [item["__CURSOR"] for item in journal.records[:4]], (
        f"every entry once, in journal order, no gap and no duplicate: {delivered}"
    )
    assert value.sink.requests, "the entries were posted"


def test_a_budget_stop_resumes_after_the_last_entry_read(tmp_path, monkeypatch) -> None:
    """Timestamp ties are where a time window cannot say where it stopped.

    Three entries share one microsecond; with one entry per batch the cursor is
    the only position that reads each exactly once. The time window would read
    the boundary entry again on every poll.
    """

    tie = NOW - timedelta(seconds=30)
    journal = Systemd252Journalctl([record(index, tie) for index in range(1, 4)])
    clock = [NOW]
    value = collector(
        tmp_path, monkeypatch, journal, clock=clock, max_entries_per_batch=1
    )

    seen = []
    for _poll in range(4):
        seen.extend(item.entry_id for item in value.collect_once().entries)
        clock[0] += timedelta(seconds=10)

    assert seen == [item["__CURSOR"] for item in journal.records], (
        f"each tied entry exactly once: {seen}"
    )
    assert _state(tmp_path)["journal_cursor"] == journal.records[-1]["__CURSOR"], (
        "the saved resume point is the last entry read"
    )
    assert [positioning(call) for call in journal.calls] == [
        ["--since"],
        ["--after-cursor"],
        ["--after-cursor"],
        ["--after-cursor"],
    ], f"a budget stop resumes by cursor like a complete read: {journal.calls}"


def test_the_window_cap_reads_by_time_without_the_cursor(tmp_path, monkeypatch) -> None:
    """A cursor older than the cap would replay the whole outage; the cap wins."""

    journal = Systemd252Journalctl([record(1, NOW - timedelta(seconds=30))])
    clock = [NOW]
    value = collector(
        tmp_path,
        monkeypatch,
        journal,
        clock=clock,
        state={
            "journal_since": (NOW - timedelta(hours=2)).isoformat(),
            "journal_cursor": record(0, NOW - timedelta(hours=2))["__CURSOR"],
            "boot_id": BOOT,
        },
    )

    batch = value.collect_once()

    assert positioning(journal.calls[0]) == ["--since"], journal.calls[0]
    assert options(journal.calls[0])["--since"] == (
        f"@{(NOW - timedelta(seconds=900)).timestamp()}"
    ), "the capped window starts 900 s back"
    assert any("journal window capped" in error for error in batch.collection_errors), (
        batch.collection_errors
    )
    assert [item.entry_id for item in batch.entries] == [journal.records[0]["__CURSOR"]]


# --------------------------------------------------------------------------- #
# cursor invalidation
# --------------------------------------------------------------------------- #
def test_a_cursor_journalctl_cannot_seek_to_is_dropped_once(
    tmp_path, monkeypatch
) -> None:
    """``Failed to seek to cursor`` costs one poll and no entry."""

    journal = Systemd252Journalctl(
        [record(1, NOW - timedelta(seconds=40)), record(2, NOW - timedelta(seconds=20))]
    )
    clock = [NOW]
    value = collector(
        tmp_path, monkeypatch, journal, clock=clock, state=_seeded("gfrs-replay-0-boot")
    )

    first = value.collect_once()
    clock[0] += timedelta(seconds=10)
    second = value.collect_once()
    clock[0] += timedelta(seconds=10)
    value.collect_once()

    assert first.collection_errors == [
        f"journalctl exited 1: {CURSOR_REJECTED}",
        "journal cursor unavailable; next poll replays timestamp window",
        "cumulative loss since collector start: journalctl-failures=1, "
        "journalctl-failures-cursor-rejected=1",
    ], f"the rejected cursor is named, classed and counted: {first.collection_errors}"
    assert first.entries == [], "a refused read has nothing to post"
    assert positioning(journal.calls[1]) == ["--since"], journal.calls[1]
    assert options(journal.calls[1])["--since"] == (
        f"@{(NOW - timedelta(seconds=60)).timestamp()}"
    ), "the fallback window starts where the last complete read ended"
    assert [item.entry_id for item in second.entries] == [
        item["__CURSOR"] for item in journal.records
    ], f"nothing behind the dropped cursor is skipped: {second.entries}"
    assert positioning(journal.calls[2]) == ["--after-cursor"], (
        f"the cursor learned by the fallback read is resumed from: {journal.calls[2]}"
    )


@pytest.mark.parametrize(
    ("boot", "resumed"),
    [(BOOT, True), ("0f0f0f0f-0f0f-0f0f-0f0f-0f0f0f0f0f0f", False), (None, False)],
    ids=["same-boot", "other-boot", "no-boot-recorded"],
)
def test_a_cursor_is_resumed_only_by_the_boot_that_saved_it(
    tmp_path, monkeypatch, boot, resumed
) -> None:
    """A reboot invalidates the position; the time window is the safe read.

    With a volatile journal the cursor's entry is gone and ``--after-cursor``
    seeks to the closest entry in time and steps past it -- the first line of
    the new boot. The time window from ``journal_since`` reads it. A state file
    written before the boot was recorded cannot prove the boot either.
    """

    # The saved cursor's entry is still in the journal (persistent storage);
    # the time window from journal_since starts after it either way.
    saved = record(1, NOW - timedelta(seconds=90))
    journal = Systemd252Journalctl([saved, record(2, NOW - timedelta(seconds=30))])
    clock = [NOW]
    value = collector(
        tmp_path,
        monkeypatch,
        journal,
        clock=clock,
        state=_seeded(saved["__CURSOR"], boot=boot),
    )

    batch = value.collect_once()

    assert positioning(journal.calls[0]) == (
        ["--after-cursor"] if resumed else ["--since"]
    ), f"boot recorded as {boot!r}: {journal.calls[0]}"
    assert [item.entry_id for item in batch.entries] == [
        journal.records[1]["__CURSOR"]
    ], "what came after the saved position is read either way, once"
    assert not any("journal cursor" in error for error in batch.collection_errors), (
        f"a boot change is not a collection error, nothing was lost: {batch.collection_errors}"
    )
    assert _state(tmp_path)["boot_id"] == BOOT, (
        "the state file names the boot its cursor belongs to"
    )


def test_a_binary_that_refuses_the_cursor_option_is_not_asked_again(
    tmp_path, monkeypatch
) -> None:
    """A refusal of the argv is a product fault, not a lost cursor.

    Dropping the cursor and resuming it on the next poll is what hid the live
    defect. A refusal while positioned by cursor switches to the time window
    for the life of the process, says so, and keeps saying so on every periodic
    report until the collector is restarted with a fix.
    """

    journal = Systemd252Journalctl([record(1, NOW - timedelta(seconds=30))])
    journal.refuse_after_cursor = True
    clock = [NOW]
    value = collector(
        tmp_path,
        monkeypatch,
        journal,
        clock=clock,
        state=_seeded(record(0, NOW)["__CURSOR"]),
    )

    refused = value.collect_once()
    clock[0] += timedelta(seconds=10)
    fallback = value.collect_once()
    for step in range(2, 5):
        journal.records.append(record(step, clock[0] - timedelta(seconds=5)))
        clock[0] += timedelta(seconds=10)
        value.collect_once()
    clock[0] = NOW + timedelta(seconds=300)
    periodic = value.collect_once()

    refusal = (
        f"journalctl refused the cursor arguments at {NOW.isoformat()} "
        f"({UNKNOWN_OPTION}); positioning by time window from now on: nothing is "
        "skipped, an entry on a window boundary can be read twice under its own id"
    )
    assert refused.collection_errors == [
        f"journalctl exited 1: {UNKNOWN_OPTION}",
        refusal,
        "cumulative loss since collector start: journalctl-failures=1, "
        "journalctl-failures-refused-arguments=1",
    ], refused.collection_errors
    assert [item.entry_id for item in fallback.entries] == [
        journal.records[0]["__CURSOR"]
    ], "the fallback read is complete"
    assert [positioning(call) for call in journal.calls[1:]] == [["--since"]] * 5, (
        f"no later poll resumes by cursor: {journal.calls[1:]}"
    )
    assert periodic.edge_filter_reasons == ["collection-error"], (
        f"the periodic report is not a health summary while degraded: {periodic}"
    )
    assert refusal in periodic.collection_errors, periodic.collection_errors


# --------------------------------------------------------------------------- #
# failures reach the control plane
# --------------------------------------------------------------------------- #
def test_failures_between_periodic_reports_are_carried_by_the_next_one(
    tmp_path, monkeypatch
) -> None:
    """Intermittent failures used to be invisible when the slot hit a good poll.

    A quiet poll is not posted, so its errors reached the control plane only if
    the periodic slot landed on it. Every other poll failing meant a coin toss
    per report, so the collector status flipped between the failure and a clean
    health summary and the erroring state never lasted.
    """

    journal = Systemd252Journalctl()
    clock = [NOW]
    value = collector(tmp_path, monkeypatch, journal, clock=clock, state=_seeded(None))
    startup = value.collect_once()

    failure = "Failed to open journal: Permission denied"
    for seconds, failing in ((10, True), (20, False), (30, True), (40, False)):
        clock[0] = NOW + timedelta(seconds=seconds)
        journal.failure = failure if failing else None
        quiet = value.collect_once()
        assert quiet.edge_filter_reasons == [], f"a quiet poll is not posted: {quiet}"
    clock[0] = NOW + timedelta(seconds=300)
    report = value.collect_once()
    clock[0] = NOW + timedelta(seconds=600)
    recovered = value.collect_once()

    assert startup.edge_filter_reasons == ["health-summary"], startup
    assert report.edge_filter_reasons == ["collection-error"], (
        f"the report after failed polls is not a health summary: {report}"
    )
    assert report.collection_errors == [
        "2 of the 4 polls since the last periodic report ended in collection "
        f"errors that were not posted; last: journalctl exited 1: {failure}",
        "cumulative loss since collector start: journalctl-failures=2, "
        "journalctl-failures-permission-denied=2",
    ], report.collection_errors
    assert recovered.edge_filter_reasons == ["health-summary"], (
        f"a clean period clears the state: {recovered}"
    )
    assert recovered.collection_errors == [], recovered.collection_errors
    posted = [request[1]["edge_filter_reasons"] for request in value.sink.requests]
    assert posted == [["health-summary"], ["collection-error"], ["health-summary"]], (
        posted
    )


def test_a_failure_on_the_slot_itself_still_counts_the_polls_before_it(
    tmp_path, monkeypatch
) -> None:
    """Consecutive failures: the report says how many, not just that one happened."""

    journal = Systemd252Journalctl()
    clock = [NOW]
    value = collector(tmp_path, monkeypatch, journal, clock=clock, state=_seeded(None))
    value.collect_once()
    journal.failure = "No journal files were found."
    for seconds in (10, 20):
        clock[0] = NOW + timedelta(seconds=seconds)
        value.collect_once()
    clock[0] = NOW + timedelta(seconds=300)

    report = value.collect_once()

    assert report.collection_errors == [
        "journalctl exited 1: No journal files were found.",
        "2 of the 2 polls since the last periodic report ended in collection "
        "errors that were not posted; last: journalctl exited 1: No journal files "
        "were found.",
        "cumulative loss since collector start: journalctl-failures=3, "
        "journalctl-failures-journal-unavailable=3",
    ], report.collection_errors


# --------------------------------------------------------------------------- #
# the helpers on their own
# --------------------------------------------------------------------------- #
def test_the_command_carries_one_positioning_option() -> None:
    since, until = NOW - timedelta(seconds=10), NOW
    cursor = record(1, since)["__CURSOR"]

    resumed = journal_command(since, until, cursor)
    first = journal_command(since, until, None)

    assert REFUSAL == POSITIONING_REFUSAL, (
        "the refusal the stand-in mirrors is the one recorded"
    )
    assert positioning(resumed) == ["--after-cursor"], resumed
    assert options(resumed)["--after-cursor"] == cursor, resumed
    assert positioning(first) == ["--since"], first
    assert options(first)["--since"] == f"@{since.timestamp()}", first
    for command in (resumed, first):
        assert options(command)["--until"] == f"@{until.timestamp()}", command
        assert {"--output", "--all", "--no-pager"} <= options(command).keys(), (
            f"the output shape and the untruncated fields are mode-independent: {command}"
        )


@pytest.mark.parametrize(
    ("returncode", "stderr", "expected"),
    [
        (1, REFUSAL, "refused-arguments"),
        (1, "--since= must be before --until=.", "refused-arguments"),
        (1, UNKNOWN_OPTION, "refused-arguments"),
        (1, CURSOR_REJECTED, "cursor-rejected"),
        (1, "Failed to open journal: Permission denied", "permission-denied"),
        (1, "Failed to open journal: No such file or directory", "journal-unavailable"),
        (1, "No journal files were found.", "journal-unavailable"),
        (-9, "", "killed"),
        (2, "something else entirely", "other"),
    ],
)
def test_failure_classes_follow_journalctl_s_own_words(
    returncode, stderr, expected
) -> None:
    assert journalctl_failure_class(returncode, stderr) == expected, (
        returncode,
        stderr,
    )
    assert expected in FAILURE_CLASSES, "every class is a documented counter"

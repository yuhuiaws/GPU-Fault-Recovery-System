"""What the node log collector keeps, what it refuses to keep, and in what order.

The durability cases live in ``test_node_log_collection_errors``; these are about
the four ways a batch could be technically correct and still useless: a matched
line with nothing around it to explain it, the collector reading its own
complaints back as node evidence, state that grows until the poll is the problem,
and a first configured path that starves every path after it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

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


class _Journalctl:
    def __init__(
        self, *, stdout: str = "", stderr: str = "", returncode: int = 0
    ) -> None:
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode

    def __call__(self, command, **_kwargs) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            list(command), self.returncode, stdout=self.stdout, stderr=self.stderr
        )


def _journal_line(cursor: str, message: str, at, *, unit: str | None = None) -> str:
    item = {
        "__CURSOR": cursor,
        "__REALTIME_TIMESTAMP": str(int(at.timestamp() * 1_000_000)),
        "MESSAGE": message,
        "_TRANSPORT": "journal",
    }
    if unit is not None:
        item["_SYSTEMD_UNIT"] = unit
    return json.dumps(item)


def _present_journalctl(monkeypatch) -> None:
    from gpu_fault.collectors.logs import node as module

    monkeypatch.setattr(module.shutil, "which", lambda _name: "/usr/bin/journalctl")


def _seeded_state(tmp_path: Path) -> Path:
    """A state file with a cursor, so a cold start is not what is under test."""

    state = tmp_path / "state.json"
    state.write_text(
        json.dumps({"journal_since": (NOW - timedelta(seconds=30)).isoformat()}),
        encoding="utf-8",
    )
    return state


def _collector(
    tmp_path: Path,
    *,
    runner: _Journalctl | None = None,
    clock: list | None = None,
    state: Path | None = None,
    training_log_paths: list[str] | None = None,
    **overrides,
) -> NodeLogCollector:
    ticker = clock if clock is not None else [NOW]
    return NodeLogCollector(
        RecordingSink(),
        context(),
        node_id="worker-1",
        training_log_paths=training_log_paths or [],
        state_path=str(state) if state else str(_seeded_state(tmp_path)),
        now=lambda: ticker[0],
        runner=runner or _Journalctl(),
        **overrides,
    )


def test_the_lines_around_a_match_are_kept_and_marked(tmp_path, monkeypatch) -> None:
    """A matched line alone does not say what the node was doing.

    Only lines matching a rule used to survive, so the driver message that
    precedes an Xid and the abort that follows it were dropped at the node -- and
    after a reinstall the node no longer has them either.
    """

    _present_journalctl(monkeypatch)
    messages = [
        "nvidia: loading driver",
        "nvidia: bound device 0000:03:00.0",
        MATCHING,
        "nvidia: resetting device",
        "kubelet: pod evicted",
        "sshd: accepted publickey",
    ]
    runner = _Journalctl(
        stdout="\n".join(
            _journal_line(
                f"cursor-{index}", message, NOW - timedelta(seconds=60 - index)
            )
            for index, message in enumerate(messages)
        )
    )
    collector = _collector(tmp_path, runner=runner)

    batch = collector.collect_once()

    assert [item.message for item in batch.entries] == messages[:5], (
        "two lines either side of the match are what explain it"
    )
    marked = {item.message: item.fields.get("log_context") for item in batch.entries}
    assert marked[MATCHING] is None, "the match itself is not context"
    assert all(marked[message] == "true" for message in messages[:2] + messages[3:5]), (
        f"a neighbour has to be distinguishable from a match: {marked}"
    )
    assert batch.collection_errors == [], "keeping context is not a collection failure"


def test_a_context_line_cannot_create_a_finding_of_its_own(
    tmp_path, monkeypatch
) -> None:
    """The safety property that makes carrying context free.

    A neighbour is kept precisely because it matched no rule, so nothing on the
    control plane can turn it into evidence: there is no rule for it to trip.
    """

    from gpu_fault.log_rules import matching_log_rules

    _present_journalctl(monkeypatch)
    runner = _Journalctl(
        stdout="\n".join(
            [
                _journal_line("cursor-0", "nvidia: loading driver", NOW),
                _journal_line("cursor-1", MATCHING, NOW + timedelta(seconds=1)),
            ]
        )
    )
    collector = _collector(tmp_path, runner=runner)

    batch = collector.collect_once()

    context_lines = [
        item for item in batch.entries if item.fields.get("log_context") == "true"
    ]
    assert context_lines, "the neighbour is in the batch"
    assert all(matching_log_rules(item.message) == [] for item in context_lines), (
        "a context line that matched a rule would be a finding nobody asked for"
    )


def test_context_lines_are_the_first_thing_the_batch_limits_drop(
    tmp_path, monkeypatch
) -> None:
    """Under pressure the match survives and its context does not.

    `_limit_entries` orders by log signal priority, which is -1 for a line no rule
    matches, so context is given up before any evidence is.
    """

    _present_journalctl(monkeypatch)
    filler = "nvidia: " + "x" * 300
    messages = [filler, filler, MATCHING + " y" * 150, filler, filler]
    runner = _Journalctl(
        stdout="\n".join(
            _journal_line(
                f"cursor-{index}", message, NOW - timedelta(seconds=60 - index)
            )
            for index, message in enumerate(messages)
        )
    )
    collector = _collector(
        tmp_path, runner=runner, max_batch_bytes=1024, max_entry_bytes=512
    )

    batch = collector.collect_once()

    kept = [item.message for item in batch.entries]
    assert messages[2] in kept, "the matched line is the one that must survive"
    assert len(kept) < len(messages), "the batch limit was reached"
    assert any(
        "dropped by the batch limits" in item for item in batch.collection_errors
    ), f"a batch that did not fit has to say so: {batch.collection_errors}"
    assert any("batch-limit-entries=" in item for item in batch.collection_errors), (
        "the running total is what says this is happening every poll"
    )


def test_the_collectors_own_journal_lines_are_not_node_evidence(
    tmp_path, monkeypatch
) -> None:
    """The feedback loop: our warning, read back, posted as a node fault.

    `_record_collection_error` warns to the journal on purpose, and the other
    collectors log their failures there too. Those lines match the same rules as
    anything else, so "the log collector cannot reach the control plane" arrived as
    a finding about the node, and a collector in a restart loop manufactured one
    out of its own complaint.
    """

    _present_journalctl(monkeypatch)
    runner = _Journalctl(
        stdout="\n".join(
            [
                _journal_line(
                    "cursor-0",
                    f"node log collection incomplete: {MATCHING}",
                    NOW,
                    unit="gpu-fault-log-collector.service",
                ),
                _journal_line(
                    "cursor-1",
                    f"nvidia-smi failed: {MATCHING}",
                    NOW + timedelta(seconds=1),
                    unit="gpu-fault-metrics-collector.service",
                ),
                _journal_line(
                    "cursor-2",
                    MATCHING,
                    NOW + timedelta(seconds=2),
                    unit="kubelet.service",
                ),
            ]
        )
    )
    collector = _collector(tmp_path, runner=runner)

    batch = collector.collect_once()

    assert [item.unit for item in batch.entries] == ["kubelet.service"], (
        "every excluded unit reports its own failures through CollectorStatus"
    )
    assert "gpu-fault-log-collector.service" in collector.excluded_units, (
        "the unit doing the reading is the one that must not be read"
    )
    assert "gpu-fault-gpu-persistence.service" not in collector.excluded_units, (
        "a unit of this system with no status channel keeps the journal as its "
        "only evidence"
    )


def test_an_operator_can_put_the_collectors_own_lines_back(
    tmp_path, monkeypatch
) -> None:
    """Debugging the collector itself needs the loop back for one session."""

    _present_journalctl(monkeypatch)
    monkeypatch.setenv("GPU_FAULT_NODE_LOG_EXCLUDED_UNITS", "")
    runner = _Journalctl(
        stdout=_journal_line(
            "cursor-0", MATCHING, NOW, unit="gpu-fault-log-collector.service"
        )
    )
    collector = _collector(tmp_path, runner=runner)

    batch = collector.collect_once()

    assert collector.excluded_units == frozenset(), "an empty value excludes nothing"
    assert [item.unit for item in batch.entries] == [
        "gpu-fault-log-collector.service"
    ], "the override is the only way the collector sees its own lines"


@pytest.mark.parametrize(
    ("state", "fragment", "why"),
    [
        (None, "no state file at", "a node installed for the first time"),
        ("{not json", "could not be read", "a truncated or corrupt state file"),
        (
            '{"training_log_files": {}}',
            "held no journal cursor",
            "state without a cursor",
        ),
    ],
)
def test_a_start_without_a_cursor_reports_the_gap_it_cannot_see(
    tmp_path, monkeypatch, state: str | None, fragment: str, why: str
) -> None:
    """Five minutes back is a guess, and it used to be a silent one.

    With no cursor the collector cannot know how long it was down: a reinstall, a
    wiped state directory or an unreadable state file all resume five minutes ago
    and everything older is simply never read. Saying so is what lets an operator
    tell a fresh node from one that has been restarting all night, and the note
    clears itself as soon as the next poll has a cursor.
    """

    _present_journalctl(monkeypatch)
    path = tmp_path / "state.json"
    if state is not None:
        path.write_text(state, encoding="utf-8")
    clock = [NOW]
    collector = _collector(tmp_path, clock=clock, state=path)
    clock[0] = NOW + timedelta(seconds=600)

    batch = collector.collect_once()

    assert len(batch.collection_errors) == 1, batch.collection_errors
    reported = batch.collection_errors[0]
    assert fragment in reported, f"{why}: {reported}"
    assert "was never read by this collector" in reported, (
        f"the note has to say what the gap means, not just that there is one: {reported}"
    )
    assert collector.sink.requests, "the gap has to reach the control plane"
    assert batch.edge_filter_reasons == ["collection-error"], (
        "a batch reporting a gap is not a health summary"
    )
    clock[0] = NOW + timedelta(seconds=1200)

    assert collector.collect_once().collection_errors == [], (
        "the poll that has a cursor has nothing to report"
    )


def test_the_running_totals_stay_off_a_healthy_batch(tmp_path, monkeypatch) -> None:
    """A total on every batch would pin the node in the erroring count forever.

    `collection_errors` is what the control plane reads to decide a batch was not
    a success, so a batch that carries a summary of past losses is a batch that
    never clears -- and nothing else can clear it either.
    """

    _present_journalctl(monkeypatch)
    clock = [NOW]
    failing = _Journalctl(returncode=1, stderr="Failed to open journal\n")
    collector = _collector(
        tmp_path, runner=failing, clock=clock, state=_seeded_state(tmp_path)
    )
    clock[0] = NOW + timedelta(seconds=300)

    first = collector.collect_once()
    clock[0] = NOW + timedelta(seconds=600)
    second = collector.collect_once()
    # A failed poll holds the cursor, so the clock stays inside the window cap:
    # a capped window is its own report and this case is about the totals.
    failing.returncode = 0
    failing.stderr = ""
    clock[0] = NOW + timedelta(seconds=800)
    third = collector.collect_once()

    assert first.collection_errors[-1].endswith("journalctl-failures=1"), (
        f"the first failure starts the total: {first.collection_errors}"
    )
    assert second.collection_errors[-1].endswith("journalctl-failures=2"), (
        "the total is what distinguishes one bad poll from a node that is down"
    )
    assert third.collection_errors == [], (
        "a poll that read its window reports nothing, so the node stops counting "
        "as erroring"
    )


def test_the_collection_error_list_is_capped_and_says_how_many_it_dropped(
    tmp_path, monkeypatch
) -> None:
    """The list is carried in every batch, so it needs a ceiling of its own.

    A node whose whole training log directory rotates at once produced one string
    per file, in a batch it was already struggling to deliver.
    """

    _present_journalctl(monkeypatch)
    monkeypatch.setenv("GPU_FAULT_NODE_LOG_MAX_COLLECTION_ERRORS", "2")
    logs = [tmp_path / f"train-{index}.log" for index in range(4)]
    for log in logs:
        log.write_text(f"{MATCHING} before\n", encoding="utf-8")
    collector = _collector(tmp_path, training_log_paths=[str(tmp_path / "train-*.log")])
    collector.collect_once()
    for index, log in enumerate(logs):
        replacement = tmp_path / f"replacement-{index}"
        replacement.write_text(f"{MATCHING} after\n", encoding="utf-8")
        os.replace(replacement, log)

    batch = collector.collect_once()

    rotations = [item for item in batch.collection_errors if "was rotated" in item]
    assert len(rotations) == 2, (
        f"the cap is what bounds the batch: {batch.collection_errors}"
    )
    assert (
        "2 further collection errors suppressed (limit 2)" in batch.collection_errors
    ), batch.collection_errors
    assert batch.collection_errors[-1].endswith("rotated-training-logs=4"), (
        "the total counts what the list stopped naming"
    )


def _training_log(path: Path, lines: int) -> Path:
    path.write_text(
        "".join(f"{MATCHING} {path.name} line {index}\n" for index in range(lines)),
        encoding="utf-8",
    )
    return path


def test_a_busy_first_training_log_cannot_starve_the_rest(
    tmp_path, monkeypatch
) -> None:
    """The starvation was permanent, not a delay.

    `_training_logs` returned as soon as the batch budget was spent and the next
    poll started at the front of the configured list again, so with one busy log
    first every path after it was never read -- for as long as the busy one stayed
    busy, which is exactly the incident where the others matter.
    """

    _present_journalctl(monkeypatch)
    busy = _training_log(tmp_path / "a-busy.log", 6)
    quiet = _training_log(tmp_path / "b-quiet.log", 1)
    collector = _collector(
        tmp_path,
        training_log_paths=[str(busy), str(quiet)],
        initial_tail_bytes=8192,
        max_entries_per_batch=2,
    )

    first = collector.collect_once()
    second = collector.collect_once()

    assert {item.fields["path"] for item in first.entries} == {str(busy)}, (
        "the first poll spends its whole budget on the busy log, as it always did"
    )
    assert any(
        "1 of 2 training logs were not read this poll" in item
        for item in first.collection_errors
    ), first.collection_errors
    assert str(quiet) in {item.fields["path"] for item in second.entries}, (
        "the second poll resumes after the path the first one stopped on"
    )


def test_the_resume_point_survives_a_restart(tmp_path, monkeypatch) -> None:
    """Restarting must not send every poll back to the front of the list."""

    _present_journalctl(monkeypatch)
    state = _seeded_state(tmp_path)
    busy = _training_log(tmp_path / "a-busy.log", 6)
    quiet = _training_log(tmp_path / "b-quiet.log", 1)
    arguments = {
        "training_log_paths": [str(busy), str(quiet)],
        "initial_tail_bytes": 8192,
        "max_entries_per_batch": 2,
    }
    first = _collector(tmp_path, state=state, **arguments)

    first.collect_once()
    stored = json.loads(state.read_text())["training_log_resume_after"]
    resumed = _collector(tmp_path, state=state, **arguments).collect_once()

    assert stored == str(busy), "the state file names where to carry on"
    assert str(quiet) in {item.fields["path"] for item in resumed.entries}, (
        "a collector that restarted every poll would read the busy log forever"
    )


def test_a_vanished_training_log_is_forgotten_and_a_skipped_one_is_not(
    tmp_path, monkeypatch
) -> None:
    """Two reasons a path is not read this poll, with opposite conclusions.

    A path that is gone leaves an offset nothing will ever use again, and a glob
    over dated filenames adds one per day. A path that merely did not fit in this
    poll's budget still holds a real offset, and forgetting that would make it look
    new and tail it -- losing everything written since the last poll, which is the
    starvation this rotation exists to fix.
    """

    _present_journalctl(monkeypatch)
    state = _seeded_state(tmp_path)
    busy = _training_log(tmp_path / "a-busy.log", 1)
    doomed = _training_log(tmp_path / "b-doomed.log", 1)
    skipped = _training_log(tmp_path / "c-skipped.log", 1)
    collector = _collector(
        tmp_path,
        state=state,
        training_log_paths=[str(tmp_path / "*.log")],
        max_entries_per_batch=1,
    )

    collector.collect_once()
    first_offsets = json.loads(state.read_text())["training_log_files"]
    doomed.unlink()
    _training_log(busy, 4)
    collector.collect_once()
    offsets = json.loads(state.read_text())["training_log_files"]

    assert set(first_offsets) == {str(busy), str(doomed), str(skipped)}, (
        f"the first poll records an offset for every path: {sorted(first_offsets)}"
    )
    assert str(doomed) not in offsets, (
        "an offset into a file that is gone is dead state"
    )
    assert offsets[str(skipped)] == first_offsets[str(skipped)], (
        "the skipped file keeps the offset it will resume from"
    )


def test_the_tracked_file_table_is_bounded_and_reports_what_it_forgot(
    tmp_path, monkeypatch
) -> None:
    """Files that all still exist can outnumber the table on their own.

    The existence prune cannot help when every dated file is still on disk, so
    past the cap the least recently visited offsets go -- and that is reported,
    because a forgotten offset means the file is tailed rather than resumed.
    """

    _present_journalctl(monkeypatch)
    monkeypatch.setenv("GPU_FAULT_NODE_LOG_MAX_TRACKED_FILES", "2")
    state = _seeded_state(tmp_path)
    logs = [_training_log(tmp_path / f"train-{index}.log", 2) for index in range(4)]
    collector = _collector(
        tmp_path,
        state=state,
        training_log_paths=[str(tmp_path / "train-*.log")],
        initial_tail_bytes=8192,
        max_entries_per_batch=2,
    )

    collector.collect_once()
    second = collector.collect_once()
    offsets = json.loads(state.read_text())["training_log_files"]

    assert set(offsets) == {str(logs[2]), str(logs[3])}, (
        f"the two least recently visited offsets go first: {sorted(offsets)}"
    )
    assert any(
        "forgot the offsets of 2" in item for item in second.collection_errors
    ), second.collection_errors
    assert second.collection_errors[-1].endswith("forgotten-training-log-offsets=2"), (
        f"forgetting an offset is loss, so it is counted: {second.collection_errors}"
    )

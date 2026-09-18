"""Durable log checkpoints and malformed records through real collector entrypoints."""

from __future__ import annotations

import json
import subprocess
from datetime import timedelta
from types import SimpleNamespace

import pytest

from gpu_fault.channel_registry import (
    COLLECTOR_HEALTH_PATH,
    FABRIC_MANAGER_PATH,
    NODE_LOG_PATH,
)
from gpu_fault.collectors.logs import fabric_manager as fabric
from gpu_fault.collectors.logs import node
from gpu_fault.collectors.sinks import CollectorError
from tests.collectors import _cov95_runtime_collect as support

isolated_runtime = support.isolated_runtime
SXID = "SXid (PCI:0000:01:00.0): 11001, Fatal, test fault\n"
TRAINING = "machine check hardware error\n"


def empty_journal(argv, **kwargs):
    return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")


def make_log_collector(kind, sink, clock, log_path, state_path):
    if kind == "fabric":
        return fabric.FabricManagerLogCollector(
            sink,
            support.collector_context(),
            node_id="node-a",
            boot_id="boot-a",
            log_paths=[str(log_path)],
            state_path=str(state_path),
            now=clock.now,
            journal_enabled=False,
            runner=empty_journal,
        )
    return node.NodeLogCollector(
        sink,
        support.collector_context(),
        node_id="node-a",
        boot_id="boot-a",
        training_log_paths=[str(log_path)],
        state_path=str(state_path),
        now=clock.now,
        monotonic=clock.monotonic,
        runner=empty_journal,
    )


@pytest.mark.parametrize("kind", ["fabric", "node"])
@pytest.mark.parametrize("corruption", ["array", "null", "file-table"])
def test_malformed_checkpoint_cold_starts_without_replaying_historical_faults(
    tmp_path, kind, corruption, caplog
):
    clock = support.Clock()
    sink = support.RecordingSink()
    log = tmp_path / "private.log"
    line = SXID if kind == "fabric" else TRAINING
    log.write_text(line)
    state = tmp_path / "checkpoint.json"
    field = "files" if kind == "fabric" else "training_log_files"
    payload = (
        []
        if corruption == "array"
        else None
        if corruption == "null"
        else {field: ["bad"]}
    )
    state.write_text(json.dumps(payload))

    collector = make_log_collector(kind, sink, clock, log, state)
    collector.collect_once()
    assert len(sink.requests) == 1, "the first round must report its collection status"
    path, report = sink.requests[0]
    assert path == (COLLECTOR_HEALTH_PATH if kind == "fabric" else NODE_LOG_PATH), (
        "a cold start must report its status on the registered channel"
    )
    if kind == "node":
        assert report["entries"] == [], "historical training faults must not replay"
        assert report["edge_filter_reasons"] == ["collection-error"], (
            "a lost journal cursor must report the unknown collection window"
        )
        assert len(report["collection_errors"]) == 1, (
            "the cold-start report must retain its journal coverage gap"
        )
        assert "no journal cursor" in report["collection_errors"][0], (
            "the report must identify the missing cursor"
        )
        assert "state file could not be read" in report["collection_errors"][0], (
            "the report must retain checkpoint corruption as the gap's cause"
        )
    else:
        assert report["edge_filter_reasons"] == ["health-summary"], (
            "a Fabric health summary must not be counted as a historical SXID"
        )
    stored = json.loads(state.read_text())
    assert stored[field][str(log)]["offset"] == log.stat().st_size, (
        "corrupt state must baseline historical files at EOF"
    )
    clock.sleep(1)
    with log.open("a") as stream:
        stream.write(line)
    collector.collect_once()

    messages = (
        [
            payload["message"]
            for path, payload in sink.requests
            if path == FABRIC_MANAGER_PATH
        ]
        if kind == "fabric"
        else [
            entry["message"]
            for path, payload in sink.requests
            if path == NODE_LOG_PATH
            for entry in payload["entries"]
        ]
    )
    assert messages == [line.rstrip("\n")], (
        "only the fault appended after the cold-start checkpoint must be delivered"
    )
    assert "cannot load" in caplog.text, "checkpoint corruption must remain observable"
    stored = json.loads(state.read_text())
    assert stored[field][str(log)]["offset"] == log.stat().st_size, (
        "the delivered fault must advance its durable checkpoint"
    )


@pytest.mark.parametrize("malformed", [None, [], 42])
def test_non_object_journal_line_cannot_block_a_following_sxid(monkeypatch, malformed):
    valid = {
        "__CURSOR": "cursor-next",
        "_SYSTEMD_UNIT": "nvidia-fabricmanager.service",
        "MESSAGE": SXID,
        "__REALTIME_TIMESTAMP": "invalid",
    }
    calls = []

    def runner(argv, **kwargs):
        calls.append(argv)
        return subprocess.CompletedProcess(
            argv, 0, stdout=json.dumps(malformed) + "\n" + json.dumps(valid), stderr=""
        )

    monkeypatch.setattr(fabric.shutil, "which", lambda name: "/private/journalctl")
    sink = support.RecordingSink()
    collector = fabric.FabricManagerLogCollector(
        sink,
        support.collector_context(),
        node_id="node-a",
        now=lambda: support.NOW,
        runner=runner,
    )
    stats = collector.collect_once()
    assert stats.delivered == 1
    assert sink.requests[0][1]["record_id"] == "fm-journal-cursor-next"
    assert sink.requests[0][1]["observed_at"] == support.NOW.isoformat()
    assert "--all" in calls[0]


def test_journal_cursor_reset_is_flushed_even_when_the_window_retry_fails(
    monkeypatch, tmp_path, caplog
):
    state = tmp_path / "checkpoint.json"
    state.write_text(json.dumps({"journal_cursor": "expired", "files": {}}))
    calls = []

    def runner(argv, **kwargs):
        calls.append(argv)
        return subprocess.CompletedProcess(
            argv, 1, stdout="", stderr="invalid cursor" if len(calls) == 1 else ""
        )

    def sleep(seconds):
        raise support.StopLoop

    monkeypatch.setattr(fabric.shutil, "which", lambda name: "/private/journalctl")
    monkeypatch.setattr(
        fabric, "time", SimpleNamespace(sleep=sleep, monotonic=lambda: 0)
    )
    collector = fabric.FabricManagerLogCollector(
        support.RecordingSink(),
        support.collector_context(),
        node_id="node-a",
        state_path=str(state),
        runner=runner,
    )
    with pytest.raises(support.StopLoop):
        collector.run()
    assert "--after-cursor" in calls[0]
    assert "--since" in calls[1]
    assert json.loads(state.read_text())["journal_cursor"] is None
    assert "Fabric Manager journal query failed" in caplog.text, (
        "the fixed journal failure category must remain observable"
    )
    assert "unknown error" not in caplog.text, (
        "failure logging must not forward the exception body"
    )


@pytest.mark.parametrize(
    ("stamp", "expected"),
    [
        ("2026-09-12T13:00:00+0100", support.NOW),
        ("2026-09-12T12:00:00Z", support.NOW),
        ("2026-09-12 11:55:00", support.NOW - timedelta(minutes=5)),
        ("2026-99-12T12:00:00", support.NOW),
    ],
)
def test_fabric_file_timestamp_conversion_is_applied_to_delivered_records(
    tmp_path, stamp, expected
):
    clock = support.Clock()
    log, state = tmp_path / "fabric.log", tmp_path / "state.json"
    log.write_text("")
    sink = support.RecordingSink()
    collector = make_log_collector("fabric", sink, clock, log, state)
    collector.collect_once()
    log.write_text(stamp + " " + SXID)
    assert collector.collect_once().delivered == 1, (
        "a fresh SXID must be delivered after timestamp conversion"
    )
    records = [
        payload for path, payload in sink.requests if path == FABRIC_MANAGER_PATH
    ]
    assert len(records) == 1, "timestamp assertions must inspect the hardware record"
    assert records[0]["observed_at"] == expected.isoformat(), (
        "the delivered SXID must carry the converted source timestamp"
    )


@pytest.mark.parametrize(
    ("age_seconds", "delivered"), [(900, True), (901, False), (3600, False)]
)
def test_fabric_file_age_fence_advances_checkpoint_without_replaying_history(
    tmp_path, age_seconds, delivered
):
    clock = support.Clock()
    log, state = tmp_path / "fabric.log", tmp_path / "state.json"
    log.write_text("")
    sink = support.RecordingSink()
    collector = make_log_collector("fabric", sink, clock, log, state)
    collector.collect_once()
    observed_at = support.NOW - timedelta(seconds=age_seconds)
    log.write_text(observed_at.isoformat() + " " + SXID)

    stats = collector.collect_once()
    records = [
        payload for path, payload in sink.requests if path == FABRIC_MANAGER_PATH
    ]
    assert (stats.observed, stats.delivered, len(records)) == (
        int(delivered),
        int(delivered),
        int(delivered),
    ), "only lines inside the default 900-second age fence are eligible faults"
    assert collector.stale_lines_skipped == int(not delivered), (
        "stale suppression must remain observable without reporting a new fault"
    )
    if delivered:
        assert records[0]["observed_at"] == observed_at.isoformat(), (
            "a line exactly at the age boundary must preserve its timestamp"
        )
    assert json.loads(state.read_text())["files"][str(log)]["offset"] == (
        log.stat().st_size
    ), "both delivered and stale complete lines must advance the checkpoint"
    requests = list(sink.requests)
    assert collector.collect_once().delivered == 0, (
        "a checkpointed line must not be delivered again"
    )
    assert sink.requests == requests, "a repeated scan must not replay stale evidence"


def test_health_summary_failure_does_not_revoke_successful_fabric_round(
    tmp_path, caplog
):
    clock = support.Clock()
    sink = support.RecordingSink(CollectorError("private summary rejected"))
    collector = fabric.FabricManagerLogCollector(
        sink,
        support.collector_context(),
        node_id="node-a",
        now=clock.now,
        journal_enabled=False,
        runner=empty_journal,
    )
    clock.sleep(301)
    stats = collector.collect_once()
    assert stats.delivered == 0
    assert len(sink.requests) == 1
    assert "the collection round is unaffected" in caplog.text


def test_duplicate_paths_directories_and_stat_errors_do_not_hide_readable_log(
    tmp_path, monkeypatch, caplog
):
    log, bad, directory = (
        tmp_path / "good.log",
        tmp_path / "bad.log",
        tmp_path / "directory",
    )
    log.write_text("")
    bad.write_text("")
    directory.mkdir()
    original = type(log).stat

    def stat(path, *args, **kwargs):
        if path == bad:
            raise OSError("private stat unavailable")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(type(log), "stat", stat)
    sink = support.RecordingSink()
    collector = fabric.FabricManagerLogCollector(
        sink,
        support.collector_context(),
        node_id="node-a",
        journal_enabled=False,
        runner=empty_journal,
        log_paths=[str(log), str(log), str(directory), str(bad)],
        now=lambda: support.NOW,
    )
    collector.collect_once()
    log.write_text(SXID)
    assert collector.collect_once().delivered == 1, (
        "an unreadable sibling must not hide the readable SXID"
    )
    records = [
        payload for path, payload in sink.requests if path == FABRIC_MANAGER_PATH
    ]
    assert len(records) == 1, "duplicate paths must not duplicate the hardware record"
    assert records[0]["message"] == SXID.rstrip("\n"), (
        "the readable file's actual SXID must be delivered"
    )
    summaries = [
        payload for path, payload in sink.requests if path == COLLECTOR_HEALTH_PATH
    ]
    assert len(summaries) == 1, "the first-round health report must remain present"
    assert "Fabric Manager log file unreadable this round" in caplog.text, (
        "the unreadable sibling must remain observable"
    )
    assert f"file_sha256={fabric.identity_sha256(str(bad))}" in caplog.text, (
        "the diagnostic must bind the exact unreadable sibling"
    )
    assert "private stat unavailable" not in caplog.text, (
        "failure logging must not forward free exception text"
    )


def test_fabric_truncation_reads_new_content_and_retains_live_offset_entries(
    tmp_path, caplog
):
    log, second, state = (
        tmp_path / "first.log",
        tmp_path / "second.log",
        tmp_path / "state.json",
    )
    log.write_text("old nonfault data " * 20 + "\n")
    state.write_text(
        json.dumps(
            {
                "files": {
                    str(tmp_path / "gone-a"): {"device": 1, "inode": 1, "offset": 0},
                    str(tmp_path / "gone-b"): {"device": 1, "inode": 2, "offset": 0},
                }
            }
        )
    )
    sink = support.RecordingSink()
    collector = fabric.FabricManagerLogCollector(
        sink,
        support.collector_context(),
        node_id="node-a",
        journal_enabled=False,
        runner=empty_journal,
        log_paths=[str(tmp_path / "*.log")],
        state_path=str(state),
        max_tracked_files=1,
    )
    collector.collect_once()
    log.write_text(SXID)
    second.write_text("")
    assert collector.collect_once().delivered == 1
    collector.collect_once()
    assert set(json.loads(state.read_text())["files"]) == {str(log), str(second)}
    assert caplog.text.count("tracked file table exceeded") == 1


def test_two_resumed_partial_lines_warn_once_and_commit_only_complete_records(
    tmp_path, caplog
):
    paths = [tmp_path / "a.log", tmp_path / "b.log"]
    states = {}
    for path in paths:
        path.write_text("old-prefix\n" + SXID)
        stat = path.stat()
        states[str(path)] = {"device": stat.st_dev, "inode": stat.st_ino, "offset": 1}
    state = tmp_path / "state.json"
    state.write_text(json.dumps({"files": states}))
    collector = fabric.FabricManagerLogCollector(
        support.RecordingSink(),
        support.collector_context(),
        node_id="node-a",
        journal_enabled=False,
        runner=empty_journal,
        state_path=str(state),
        log_paths=[str(path) for path in paths],
    )
    assert collector.collect_once().delivered == 2
    assert caplog.text.count("re-syncing to the next line") == 1
    assert all(
        entry["offset"] == paths[index].stat().st_size
        for index, entry in enumerate(json.loads(state.read_text())["files"].values())
    ), "both checkpoints must reach complete line boundaries"


@pytest.mark.parametrize(
    "options",
    [
        {"initial_tail_bytes": -1},
        {"max_entries_per_batch": 0},
        {"max_batch_bytes": 100},
        {"max_entry_bytes": 100},
        {"context_lines": -1},
        {"scan_bytes": 100},
        {"scan_seconds": -1},
        {"journal_timeout_seconds": 0},
    ],
)
def test_node_log_limits_are_checked_before_journal_start(options):
    with pytest.raises(ValueError, match="node log"):
        node.NodeLogCollector(
            support.RecordingSink(),
            support.collector_context(),
            node_id="node-a",
            boot_id="boot-a",
            runner=empty_journal,
            **options,
        )


@pytest.mark.parametrize(
    ("variable", "value"),
    [
        ("GPU_FAULT_NODE_LOG_MAX_COLLECTION_ERRORS", "0"),
        ("GPU_FAULT_NODE_LOG_MAX_TRACKED_FILES", "0"),
        ("GPU_FAULT_NODE_LOG_MAX_JOURNAL_WINDOW_SECONDS", "59"),
    ],
)
def test_node_log_environment_bounds_fail_before_journal_start(
    monkeypatch, variable, value
):
    monkeypatch.setenv(variable, value)
    with pytest.raises(ValueError, match="node log"):
        node.NodeLogCollector(
            support.RecordingSink(),
            support.collector_context(),
            node_id="node-a",
            boot_id="boot-a",
            runner=empty_journal,
        )

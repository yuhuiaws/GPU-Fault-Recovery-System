"""Raw-source regressions: real pagination, CRI, source identity and bounded I/O."""

import json
import subprocess
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

import gpu_fault.collectors.logs.node as node_module
from gpu_fault.collectors.logs.node import NodeLogCollector
from gpu_fault.collectors.models import CollectorContext
from gpu_fault.collectors.sinks import CollectorError
from gpu_fault.fleet import AgentRecord
from gpu_fault.host_health import NodeHealthPolicy, NodeLogBatch, NodeLogEntry
from gpu_fault.models import RecoveryAction
from tests._builders import attempt_observation, build_store, container_observation
from tests.execution._health_policy_e2e_support import HealthRig

NOW = datetime(2026, 10, 23, 10, tzinfo=timezone.utc)


class Sink:
    def post(self, path, payload):
        return {}


class Journal:
    def __init__(self, records):
        self.records = list(records)
        self.commands = []

    def __call__(self, argv, **_):
        self.commands.append(argv)
        since = float(argv[argv.index("--since") + 1].removeprefix("@"))
        until = float(argv[argv.index("--until") + 1].removeprefix("@"))
        cursor = next(
            (arg.split("=", 1)[1] for arg in argv if arg.startswith("--after-cursor=")),
            None,
        )
        first = 0
        if cursor is not None:
            positions = [
                i for i, item in enumerate(self.records) if item["__CURSOR"] == cursor
            ]
            if not positions:
                return subprocess.CompletedProcess(
                    argv, 1, "", "Failed to seek to cursor"
                )
            first = positions[0] + 1
        records = [
            item
            for item in self.records[first:]
            if since <= int(item["__REALTIME_TIMESTAMP"]) / 1_000_000 <= until
        ]
        return subprocess.CompletedProcess(
            argv, 0, "\n".join(map(json.dumps, records)), ""
        )


def entry(
    cursor, at=NOW, message="mce: uncorrectable hardware error", boot="boot-current"
):
    return {
        "__CURSOR": cursor,
        "__REALTIME_TIMESTAMP": str(int(at.timestamp() * 1_000_000)),
        "_TRANSPORT": "kernel",
        "_BOOT_ID": boot,
        "MESSAGE": message,
    }


def collector(tmp_path, monkeypatch, *, records=(), paths=(), state=None, **options):
    monkeypatch.setattr(node_module.shutil, "which", lambda _: "/usr/bin/journalctl")
    path = tmp_path / "state.json"
    if not path.exists():
        path.write_text(
            json.dumps(
                state or {"journal_since": (NOW - timedelta(seconds=60)).isoformat()}
            )
        )
    return NodeLogCollector(
        Sink(),
        CollectorContext(
            cluster_id="cluster-a", runtime_profile_version="simulated-v1"
        ),
        node_id="node-a",
        boot_id="boot-current",
        now=lambda: NOW,
        state_path=str(path),
        runner=Journal(records),
        training_log_paths=list(paths),
        context_lines=0,
        **options,
    )


@pytest.mark.parametrize("limit", [1, 2])
def test_timestamp_ties_resume_by_cursor_across_restart(tmp_path, monkeypatch, limit):
    records = [entry(f"cursor-{i}", NOW - timedelta(seconds=5)) for i in range(3)]
    seen = []
    for _ in range(4):
        value = collector(
            tmp_path, monkeypatch, records=records, max_entries_per_batch=limit
        )
        seen.extend(item.entry_id for item in value.collect_once().entries)
    assert seen == ["cursor-0", "cursor-1", "cursor-2"], seen


def test_vacuumed_cursor_replays_timestamp_window(tmp_path, monkeypatch):
    value = collector(
        tmp_path,
        monkeypatch,
        records=[entry("new")],
        state={
            "journal_since": (NOW - timedelta(seconds=60)).isoformat(),
            "journal_cursor": "vacuumed",
        },
    )
    first = value.collect_once()
    assert any("cursor unavailable" in error for error in first.collection_errors), (
        first
    )
    assert [item.entry_id for item in value.collect_once().entries] == ["new"]


def test_deferred_new_file_keeps_its_first_baseline(tmp_path, monkeypatch):
    busy, quiet = tmp_path / "a.log", tmp_path / "b.log"
    busy.write_text("NCCL timeout error\nNCCL timeout error\n")
    quiet.write_text("")
    stat = busy.stat()
    value = collector(
        tmp_path,
        monkeypatch,
        paths=[str(busy), str(quiet)],
        state={
            "journal_since": (NOW - timedelta(seconds=60)).isoformat(),
            "training_log_files": {
                str(busy): {"device": stat.st_dev, "inode": stat.st_ino, "offset": 0}
            },
        },
        max_entries_per_batch=1,
    )
    value.collect_once()
    quiet.write_text("NCCL timeout on newly observed file\n")
    second = value.collect_once()
    assert any(item.fields.get("path") == str(quiet) for item in second.entries), second


def test_missing_journal_is_unknown(tmp_path, monkeypatch):
    value = collector(tmp_path, monkeypatch)
    monkeypatch.setattr(node_module.shutil, "which", lambda _: None)
    batch = value.collect_once()
    assert batch.collection_errors, batch
    assert "health-summary" not in batch.edge_filter_reasons, batch


def test_oversize_line_resumes_with_bounded_raw_reads(tmp_path, monkeypatch):
    path = tmp_path / "rank.log"
    path.write_bytes(b"NCCL timeout error " + b"x" * 8000 + b"\n")
    seen = []
    for _ in range(9):
        value = collector(
            tmp_path,
            monkeypatch,
            paths=[str(path)],
            initial_tail_bytes=16384,
            scan_bytes=1024,
            max_entry_bytes=256,
        )
        seen.extend(value.collect_once().entries)
        state = json.loads((tmp_path / "state.json").read_text())
        assert state["training_log_files"][str(path)]["offset"] <= (_ + 1) * 1024
    assert len(seen) == 1, seen
    assert "truncated" in seen[0].message, seen


@pytest.mark.parametrize(
    "message,action",
    [
        (
            "mce: uncorrectable hardware error; corrected errors were also logged",
            RecoveryAction.QUARANTINE,
        ),
        ("mce: corrected hardware error", RecoveryAction.RUN_DIAGNOSTICS),
        ("mce: corrected hardware error; efa fatal error", RecoveryAction.QUARANTINE),
    ],
)
def test_corrected_context_never_masks_a_fatal_rule(message, action):
    result = NodeHealthPolicy(build_store()).evaluate_logs(
        NodeLogBatch(
            cluster_id="cluster-a",
            node_id="node-a",
            collected_at=NOW,
            entries=[
                NodeLogEntry(
                    entry_id="mce", source="dmesg", observed_at=NOW, message=message
                )
            ],
        )
    )
    assert result[0].recommended_action is action, result


def test_exited_rank_keeps_identity_without_live_process_capture():
    store = build_store()
    store.save_attempt_observation(
        attempt_observation(
            "train",
            "attempt-a",
            NOW,
            expected_critical_ranks=1,
            workload_ids=["training/job/train"],
            containers=[
                container_observation(
                    "pod-a",
                    "trainer",
                    0,
                    "node-a",
                    host_pid=123,
                    terminated=True,
                    exit_code=1,
                )
            ],
        )
    )
    result = NodeHealthPolicy(store).evaluate_logs(
        NodeLogBatch(
            cluster_id="cluster-a",
            node_id="node-a",
            collected_at=NOW,
            workload_state="ACTIVE",
            affected_workload_ids=["training/job/train"],
            entries=[
                NodeLogEntry(
                    entry_id="exited",
                    source="journal",
                    observed_at=NOW,
                    message="NCCL timeout error",
                    fields={"_PID": "123"},
                )
            ],
        )
    )
    assert result[0].diagnostic_parameters["process_state"] == "EXITED", result
    assert not result[0].diagnostic_parameters["capture_process_state"], result


def test_old_boot_evidence_cannot_quarantine_current_boot(tmp_path, monkeypatch):
    rig = HealthRig(tmp_path, monkeypatch)
    try:
        now = rig.clock
        rig.store.save_agent(
            AgentRecord(
                cluster_id="cluster-a",
                node_id="node-a",
                endpoint="https://127.0.0.1",
                agent_version="test",
                artifact_sha256="a" * 64,
                policy_version="test",
                runtime_profile_version="simulated-v1",
                config_digest="b" * 64,
                allowed_operations=[],
                boot_id="boot-current",
                first_seen_at=now - timedelta(seconds=30),
                last_seen_at=now,
                lease_expires_at=now + timedelta(minutes=1),
            )
        )

        class ApiSink:
            def post(self, path, payload):
                return rig.post(path, payload)

        monkeypatch.setattr(
            node_module.shutil, "which", lambda _: "/usr/bin/journalctl"
        )
        value = NodeLogCollector(
            ApiSink(),
            CollectorContext(
                cluster_id="cluster-a", runtime_profile_version="simulated-v1"
            ),
            node_id="node-a",
            boot_id="boot-current",
            now=lambda: now,
            runner=Journal(
                [entry("old", now - timedelta(seconds=60), boot="boot-old")]
            ),
        )
        batch = value.collect_once()
        assert batch.entries[0].fields["_BOOT_ID"] == "boot-old", batch
        assert rig.store.list_workflows() == [], (
            "old evidence must not dispatch recovery"
        )
        assert rig.core.node["spec"]["unschedulable"] is False, rig.core.node
    finally:
        rig.store.close()


def test_cri_partial_join_survives_restart_and_preserves_identity(
    tmp_path, monkeypatch
):
    path = tmp_path / "pods" / "ns_train_pod-a" / "trainer" / "0.log"
    path.parent.mkdir(parents=True)
    path.write_text("2026-10-23T09:59:59.000000001Z stdout P NCCL WARN collective \n")
    options = dict(
        paths=[str(path)], initial_tail_bytes=4096, pod_log_root=str(tmp_path / "pods")
    )
    first = collector(tmp_path, monkeypatch, **options).collect_once()
    assert first.entries == [], first
    with path.open("a") as stream:
        stream.write("2026-10-23T09:59:59.000000002Z stdout F operation timeout\n")
    batch = collector(tmp_path, monkeypatch, **options).collect_once()
    assert len(batch.entries) == 1, batch
    item = batch.entries[0]
    assert item.message == "NCCL WARN collective operation timeout"
    assert item.fields["pod_uid"] == "pod-a", item
    assert item.observed_at == NOW - timedelta(seconds=1)


def test_cri_continuation_follows_inode_rotation(tmp_path, monkeypatch):
    path = tmp_path / "0.log"
    path.write_text("2026-10-23T09:59:59Z stdout P NCCL collective \n")
    options = dict(paths=[str(tmp_path / "*.log*")], initial_tail_bytes=4096)
    collector(tmp_path, monkeypatch, **options).collect_once()
    rotated = tmp_path / "0.log.1"
    path.rename(rotated)
    path.write_text("")
    with rotated.open("a") as stream:
        stream.write("2026-10-23T09:59:59Z stdout F timeout\n")
    batch = collector(tmp_path, monkeypatch, **options).collect_once()
    assert [item.message for item in batch.entries] == ["NCCL collective timeout"]
    assert batch.entries[0].fields["path"] == str(rotated)


def test_rejected_cri_batch_preserves_the_durable_fragment(tmp_path, monkeypatch):
    path = tmp_path / "0.log"
    path.write_text("2026-10-23T09:59:59Z stdout P NCCL collective \n")
    options = dict(paths=[str(path)], initial_tail_bytes=4096)
    collector(tmp_path, monkeypatch, **options).collect_once()
    before = (tmp_path / "state.json").read_text()
    with path.open("a") as stream:
        stream.write("2026-10-23T09:59:59Z stdout F timeout\n")

    class Rejected:
        def post(self, *_):
            raise CollectorError("not delivered", status_code=503)

    value = collector(tmp_path, monkeypatch, **options)
    value.sink = Rejected()
    with pytest.raises(CollectorError):
        value.collect_once()
    assert (tmp_path / "state.json").read_text() == before
    batch = collector(tmp_path, monkeypatch, **options).collect_once()
    assert [item.message for item in batch.entries] == ["NCCL collective timeout"]


def test_file_read_error_restores_partial_line_before_retry(tmp_path, monkeypatch):
    path = tmp_path / "0.log"
    path.write_text("NCCL collec")
    options = dict(paths=[str(path)], initial_tail_bytes=4096)
    value = collector(tmp_path, monkeypatch, **options)
    value.collect_once()
    with path.open("a") as stream:
        stream.write("tive timeout\nanother line\n")
    original = Path.open

    class Interrupted:
        def __init__(self, stream):
            self.stream, self.reads = stream, 0

        def __enter__(self):
            return self

        def __exit__(self, *_):
            self.stream.close()

        def __getattr__(self, name):
            return getattr(self.stream, name)

        def readline(self, size):
            self.reads += 1
            if self.reads == 2:
                raise OSError("injected read failure after a completed record")
            return self.stream.readline(size)

    def open_file(target, mode="r", *args, **kwargs):
        stream = original(target, mode, *args, **kwargs)
        return Interrupted(stream) if target == path and mode == "rb" else stream

    with monkeypatch.context() as patch:
        patch.setattr(Path, "open", open_file)
        failed = value.collect_once()
    assert failed.collection_errors, failed
    recovered = value.collect_once()
    assert [item.message for item in recovered.entries] == ["NCCL collective timeout"]


def test_old_or_unbound_training_log_cannot_target_current_attempt(
    tmp_path, monkeypatch
):
    path = tmp_path / "rank.log"
    path.write_text("2026-10-22T10:00:00.000000000Z stdout F NCCL timeout error\n")
    value = collector(tmp_path, monkeypatch, paths=[str(path)], initial_tail_bytes=4096)
    batch = value.collect_once().model_copy(
        update={"affected_workload_ids": ["training/job/train"]}
    )
    store = build_store()
    store.save_attempt_observation(
        attempt_observation(
            "train",
            "new-attempt",
            NOW,
            expected_critical_ranks=1,
            started_at=NOW - timedelta(seconds=10),
            workload_ids=["training/job/train"],
            containers=[
                container_observation("new-pod", "trainer", 0, "node-a", host_pid=456)
            ],
        )
    )
    findings = NodeHealthPolicy(store).evaluate_logs(batch)
    assert findings[0].attempt_id is None, findings
    assert batch.entries[0].observed_at == NOW - timedelta(days=1)


def test_tracking_cap_reports_incomplete_coverage(tmp_path, monkeypatch):
    monkeypatch.setenv("GPU_FAULT_NODE_LOG_MAX_TRACKED_FILES", "2")
    paths = [tmp_path / f"quiet-{i}.log" for i in range(3)]
    for path in paths:
        path.write_text("")
    value = collector(tmp_path, monkeypatch, paths=list(map(str, paths)))
    batch = value.collect_once()
    tracked = json.loads((tmp_path / "state.json").read_text())["training_log_files"]
    assert len(tracked) <= 2, tracked
    assert any(
        "coverage is incomplete" in error for error in batch.collection_errors
    ), batch


def test_stalled_file_io_does_not_block_journal_or_spawn_more_workers(
    tmp_path, monkeypatch
):
    entered, release = threading.Event(), threading.Event()
    original = Path.glob

    def blocked_glob(path, pattern):
        entered.set()
        release.wait(5)
        yield from original(path, pattern)

    monkeypatch.setattr(Path, "glob", blocked_glob)
    value = collector(
        tmp_path,
        monkeypatch,
        paths=[str(tmp_path / "*.log")],
        records=[entry("kernel")],
        training_io_timeout_seconds=0.01,
    )
    workers = []
    try:
        first = value.collect_once()
        assert entered.wait(1), "the file reader must enter the blocked I/O"
        workers = [
            item
            for item in threading.enumerate()
            if item.name == "node-log-file-reader"
        ]
        second = value.collect_once()
        assert [
            item
            for item in threading.enumerate()
            if item.name == "node-log-file-reader"
        ] == workers, "do not accumulate stuck readers"
        assert [item.entry_id for item in first.entries] == ["kernel"]
        assert any("coverage UNKNOWN" in error for error in second.collection_errors), (
            second
        )
    finally:
        release.set()
        for worker in workers:
            worker.join(2)

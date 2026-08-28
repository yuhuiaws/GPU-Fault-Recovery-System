from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

from gpu_fault.hma import FabricManagerLogEvent
from gpu_fault.training_models import TrainingProgressHeartbeat
from gpu_fault.watcher import AttemptObservation
from scripts.perf import benchmark_synchronized_burst as burst
from scripts.perf import regional_capacity_suite as suite

ROOT = Path(__file__).resolve().parents[2]


def test_drain_targets_preserve_clean_environment_threshold() -> None:
    metrics = {
        "pod-a": (
            "gpu_fault_processor_queue_depth 0\ngpu_fault_telemetry_spool_depth 1\n"
        )
    }

    assert suite.drain_targets(metrics) == (1.0, 1.0)


def test_drain_targets_allow_online_steady_state() -> None:
    metrics = {
        "pod-a": (
            "gpu_fault_processor_queue_depth 8\ngpu_fault_telemetry_spool_depth 0\n"
        ),
        "pod-b": (
            "gpu_fault_processor_queue_depth 10\ngpu_fault_telemetry_spool_depth 2\n"
        ),
    }

    assert suite.drain_targets(metrics) == (14.0, 6.0)


def test_artifact_directory_uses_case_release_and_utc(tmp_path: Path) -> None:
    path = suite.artifact_dir(tmp_path, "burst 32c", "release/abc123")

    assert path.parent.parent.name == "burst-32c"
    assert path.parent.name == "release-abc123"
    assert path.name.endswith("Z")


def test_status_and_aborted_directory_are_explicit(tmp_path: Path) -> None:
    path = suite.artifact_dir(tmp_path, "burst", "abc123")
    suite.write_status(path, status="aborted", reason="probe failed")

    status = json.loads((path / "status.json").read_text())
    assert status["status"] == "aborted"
    assert status["reason"] == "probe failed"

    moved = suite.move_to_aborted(tmp_path, path)
    assert moved.is_dir()
    assert moved.relative_to(tmp_path).parts[0] == "_aborted"


def test_dataplane_context_must_be_explicit(monkeypatch) -> None:
    monkeypatch.setattr(suite, "DATAPLANE_CONTEXT", "")
    monkeypatch.setattr(sys, "argv", ["regional-capacity", "register"])

    with pytest.raises(SystemExit, match="2"):
        suite.main()


def test_synthetic_registry_uses_placeholder_aws_account() -> None:
    entry = suite.perf_cluster_entries(1)[0]

    assert ":000000000000:cluster/" in entry["eks_cluster_arn"]


def test_registry_baseline_artifact_carries_digests_not_tokens() -> None:
    # baseline 行是从生产 registry Secret 里原样读出来的，带真 token。
    # 每次 register 都把它整行写进 artifacts/，77 份历史证据因此全都
    # 明文存着同一个区域集群 token。工件只需要回答「保留了哪几行、
    # 有没有放回同样的行」，摘要足够。
    token = "3f9c1a04be27d5610872ef4bc93d0a6f5e18720b4dcaf3961e05b8d2740cae63"
    entries = [{"cluster_id": "hp-a", "region": "us-east-2", "token": token}]

    redacted = suite.redacted_registry_entries(entries)

    assert "token" not in redacted[0]
    assert redacted[0]["token_sha256"] == hashlib.sha256(token.encode()).hexdigest()
    assert redacted[0]["cluster_id"] == "hp-a"
    # 原始入参不能被就地改写：同一批 entries 随后要写回 registry Secret。
    assert entries[0]["token"] == token


def test_capacity_cleanup_removes_action_workflows_without_cluster_payload() -> None:
    statements = {
        name: (sql, pattern) for name, sql, pattern in suite.AUDIT_PURGE_STATEMENTS
    }

    sql, pattern = statements["gpu_fault_action_workflows"]

    assert "kind='workflow'" in sql
    assert "key LIKE %s" in sql
    assert pattern == "action_workflow"


def test_repository_registry_baselines_are_redacted() -> None:
    offenders = [
        path.relative_to(ROOT).as_posix()
        for path in (ROOT / "artifacts").rglob("registry-baseline.json")
        if any(
            "token" in entry for entry in json.loads(path.read_text(encoding="utf-8"))
        )
    ]

    assert offenders == []


def test_burst_job_keeps_indexed_start_gate_contract() -> None:
    manifest = suite.build_job(
        "burst",
        clusters=32,
        nodes_per_cluster=256,
        xid_total=500,
        sxid_total=500,
        gpu_evidence_total=0,
        host_evidence_total=0,
        training_heartbeat_total=0,
        workload_observation_total=0,
        correlate_attempt_faults=False,
        start_epoch=1234.0,
        workers=128,
        duration_seconds=60,
        cpu_request="2",
        cpu_limit="4",
        include_telemetry=True,
        prewarm_connections=False,
    )

    spec = manifest["spec"]
    container = spec["template"]["spec"]["containers"][0]
    environment = {item["name"]: item for item in container["env"]}

    assert spec["completionMode"] == "Indexed"
    assert spec["completions"] == 32
    assert container["readinessProbe"]["exec"]["command"][-1] == (
        "test -f /tmp/gpu-fault-load-ready"
    )
    assert environment["START_GATE_NAME"]["value"] == suite.START_GATE_CONFIGMAP
    assert environment["CLUSTER_OFFSET"]["valueFrom"]["fieldRef"]["fieldPath"] == (
        "metadata.annotations['batch.kubernetes.io/job-completion-index']"
    )


def test_complete_single_cluster_burst_covers_attempt_context() -> None:
    events = burst.build_events(
        cluster_count=1,
        cluster_offset=0,
        nodes_per_cluster=1000,
        xid_total=500,
        sxid_total=500,
        gpu_evidence_total=500,
        host_evidence_total=500,
        include_telemetry=True,
        training_heartbeat_total=1000,
        workload_observation_total=500,
        correlate_attempt_faults=True,
    )
    counts = {
        kind: sum(item[0] == kind for item in events)
        for kind in {item[0] for item in events}
    }

    assert counts == {
        "FABRIC_MANAGER_LOG": 500,
        "GPU_INVENTORY": 1000,
        "GPU_METRICS": 1000,
        "GPU_METRICS_EVIDENCE": 500,
        "HOST_TELEMETRY": 1000,
        "HOST_TELEMETRY_EVIDENCE": 500,
        "NVIDIA_KERNEL": 500,
        "TRAINING_PROGRESS": 1000,
        "WORKLOAD_OBSERVATION": 500,
    }
    xid_nodes = [node for kind, _template, node in events if kind == "NVIDIA_KERNEL"]
    sxid_nodes = [
        node for kind, _template, node in events if kind == "FABRIC_MANAGER_LOG"
    ]
    assert xid_nodes == list(range(0, 1000, 2))
    assert sxid_nodes == list(range(1, 1000, 2))


def test_complete_burst_payloads_share_two_node_attempt_identity() -> None:
    templates = json.loads(
        (ROOT / "scripts/perf/payload-templates.json").read_text(encoding="utf-8")
    )
    observation = burst.build_event_payload(
        templates=templates,
        kind="WORKLOAD_OBSERVATION",
        template_kind="WORKLOAD_OBSERVATION",
        cluster_id="cluster-a",
        node_index=24,
        sequence=1,
        correlate_attempt_faults=True,
    )
    heartbeat = burst.build_event_payload(
        templates=templates,
        kind="TRAINING_PROGRESS",
        template_kind="TRAINING_PROGRESS",
        cluster_id="cluster-a",
        node_index=25,
        sequence=2,
        correlate_attempt_faults=True,
    )
    fault = burst.build_event_payload(
        templates=templates,
        kind="FABRIC_MANAGER_LOG",
        template_kind="FABRIC_MANAGER_LOG",
        cluster_id="cluster-a",
        node_index=25,
        sequence=3,
        correlate_attempt_faults=True,
    )

    assert observation["attempt_id"] == "burst-attempt-0012"
    assert [item["node_id"] for item in observation["containers"]] == [
        "burst-node-0024",
        "burst-node-0025",
    ]
    assert heartbeat["attempt_id"] == observation["attempt_id"]
    assert heartbeat["rank"] == 1
    assert fault["affected_workload_ids"] == observation["workload_ids"]
    assert fault["workload_state"] == "ACTIVE"
    AttemptObservation(**observation)
    TrainingProgressHeartbeat(**heartbeat)
    FabricManagerLogEvent(**fault)


def test_result_aggregation_preserves_fault_latency_summary() -> None:
    summary = suite.aggregate(
        [
            {
                "events": 2,
                "wall_seconds": 1.0,
                "start_lag_seconds": 0.1,
                "client_cpu_cores": 0.5,
                "paths": {
                    "NVIDIA_KERNEL": {
                        "raw_latencies_ms": [10.0, 20.0],
                        "status_counts": {"202": 2},
                    }
                },
            }
        ]
    )

    assert summary["requests"] == 2
    assert summary["throughput_req_s"] == 2.0
    assert summary["fault_p50_ms"] == 10.0
    assert summary["fault_p99_ms"] == 10.0

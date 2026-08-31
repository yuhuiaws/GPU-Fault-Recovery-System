from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from gpu_fault.hma import FabricManagerLogEvent
from gpu_fault.regional import RegionalClusterRegistration
from gpu_fault.training_models import TrainingProgressHeartbeat
from gpu_fault.watcher import AttemptObservation
from scripts.perf import benchmark_synchronized_burst as burst
from scripts.perf import regional_capacity_registry as registry_module
from scripts.perf import regional_capacity_suite as suite

ROOT = Path(__file__).resolve().parents[2]
NOW = datetime(2026, 8, 30, tzinfo=timezone.utc)


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


def test_capacity_run_refuses_drill_email_delivery(monkeypatch) -> None:
    monkeypatch.setattr(registry_module, "control_pods", lambda: ["pod-a", "pod-b"])

    def fake_control(*args, **_kwargs):
        pod = args[1]
        return "true\n" if pod == "pod-b" else "false\n"

    monkeypatch.setattr(registry_module, "control", fake_control)

    with pytest.raises(
        RuntimeError, match="capacity runs must not deliver drill notifications"
    ):
        registry_module.validate_notification_safety()


def test_capacity_run_allows_suppressed_drills(monkeypatch) -> None:
    monkeypatch.setattr(registry_module, "control_pods", lambda: ["pod-a", "pod-b"])
    monkeypatch.setattr(registry_module, "control", lambda *_args, **_kwargs: "false\n")

    registry_module.validate_notification_safety()


def test_synthetic_registry_uses_placeholder_aws_account() -> None:
    entry = registry_module.perf_cluster_entries(
        1, run_id="run-a", expires_at=NOW + timedelta(days=1)
    )[0]

    assert ":000000000000:cluster/" in entry["eks_cluster_arn"]
    assert entry["synthetic"] is True
    assert entry["synthetic_run_id"] == "run-a"
    assert entry["synthetic_expires_at"] == (NOW + timedelta(days=1)).isoformat()
    assert entry["agent_endpoint_allowed_cidrs"] == ["127.0.0.1/32"]


def test_synthetic_registry_metadata_passes_runtime_model() -> None:
    observed_at = datetime.now(timezone.utc)
    entry = registry_module.perf_cluster_entries(
        1, run_id="run-a", expires_at=observed_at + timedelta(days=1)
    )[0]
    token = entry.pop("token")
    entry["token_sha256"] = hashlib.sha256(token.encode()).hexdigest()

    registration = RegionalClusterRegistration(**entry)

    assert registration.is_active(observed_at), (
        "synthetic registration was inactive before TTL"
    )
    assert registration.authenticates(token), (
        "active synthetic registration rejected its token"
    )


def test_live_registry_requires_double_confirmation(monkeypatch) -> None:
    monkeypatch.setattr(
        registry_module,
        "CONTROL_NAMESPACE",
        registry_module.PRODUCTION_CONTROL_NAMESPACE,
    )

    with pytest.raises(RuntimeError, match="production control plane"):
        registry_module.validate_registry_target(
            allow_live_registry=False, confirmation=None
        )

    assert (
        registry_module.validate_registry_target(
            allow_live_registry=True,
            confirmation=registry_module.LIVE_REGISTRY_CONFIRMATION,
        )
        == "live"
    )


def test_preflight_cleanup_removes_legacy_synthetic_entries(
    tmp_path: Path, monkeypatch
) -> None:
    entries = [{"cluster_id": "production"}, {"cluster_id": "perf-cap-000"}]
    restarts = []
    monkeypatch.setattr(registry_module, "load_registry", lambda: list(entries))
    monkeypatch.setattr(
        registry_module,
        "write_registry",
        lambda values: entries.__setitem__(slice(None), values),
    )
    monkeypatch.setattr(
        registry_module, "restart_control_plane", lambda: restarts.append(True)
    )

    removed = registry_module.cleanup_registry_residuals(
        scope="isolated", artifacts=tmp_path, phase="preflight", force=False
    )

    assert removed == 1
    assert entries == [{"cluster_id": "production"}]
    assert restarts == [True]
    audit = json.loads((tmp_path / "registry-preflight.json").read_text())
    assert audit["removed"] == 1
    assert audit["synthetic_entries"][0]["legacy_prefix_only"] is True


def test_preflight_refuses_another_active_synthetic_run(monkeypatch) -> None:
    monkeypatch.setattr(
        registry_module,
        "load_registry",
        lambda: [
            {
                "cluster_id": "perf-cap-000",
                "synthetic": True,
                "synthetic_run_id": "other-run",
                "synthetic_expires_at": (NOW + timedelta(hours=1)).isoformat(),
            }
        ],
    )
    with pytest.raises(RuntimeError, match="other-run"):
        registry_module.cleanup_registry_residuals(
            scope="isolated", artifacts=None, phase="preflight", force=False, now=NOW
        )


def test_teardown_retries_idempotently(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(suite, "validate_registry_target", lambda **_kwargs: "isolated")
    monkeypatch.setattr(suite.time, "sleep", lambda _seconds: None)

    def teardown_once(**_kwargs):
        calls.append(True)
        if len(calls) < 3:
            raise RuntimeError("transient")

    monkeypatch.setattr(suite, "_teardown_once", teardown_once)

    suite.teardown(purge=True, deregister_clusters=True)

    assert len(calls) == 3


def test_capacity_exception_still_runs_teardown(tmp_path: Path, monkeypatch) -> None:
    calls = []
    args = SimpleNamespace(
        command="all",
        clusters=1,
        keep_registration=False,
        allow_live_registry=False,
        confirm_live_registry=None,
        no_purge=False,
        case="burst",
        nodes_per_cluster=1,
        gpu_evidence_total=0,
        host_evidence_total=0,
        training_heartbeat_total=0,
        workload_observation_total=0,
        correlate_attempt_faults=False,
        workers=1,
        duration_seconds=1,
        lead_seconds=1,
        cpu_request="1",
        cpu_limit="1",
        label="test",
        fault_only=False,
        prewarm_connections=False,
        artifact_root=tmp_path,
    )
    monkeypatch.setattr(suite, "register", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(
        suite,
        "execute_case",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("failed")),
    )
    monkeypatch.setattr(suite, "teardown", lambda **kwargs: calls.append(kwargs))
    monkeypatch.setattr(suite, "move_to_aborted", lambda _root, path: path)

    with pytest.raises(RuntimeError, match="failed"):
        suite.execute_capacity_command(
            args,
            artifacts=tmp_path,
            suite_id="run-a",
            expires_at=NOW + timedelta(hours=1),
            scope="isolated",
            xid_total=1,
            sxid_total=1,
        )

    assert calls[0]["run_id"] == "run-a"
    assert calls[0]["deregister_clusters"] is True


def test_registry_baseline_artifact_carries_digests_not_tokens() -> None:
    # baseline 行是从生产 registry Secret 里原样读出来的，带真 token。
    # 每次 register 都把它整行写进 artifacts/，77 份历史证据因此全都
    # 明文存着同一个区域集群 token。工件只需要回答「保留了哪几行、
    # 有没有放回同样的行」，摘要足够。
    token = "3f9c1a04be27d5610872ef4bc93d0a6f5e18720b4dcaf3961e05b8d2740cae63"
    entries = [{"cluster_id": "hp-a", "region": "us-east-2", "token": token}]

    redacted = registry_module.redacted_registry_entries(entries)

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


def test_capacity_cleanup_removes_synthetic_notification_chain() -> None:
    statements = {
        name: (sql, pattern) for name, sql, pattern in suite.AUDIT_PURGE_STATEMENTS
    }

    for name in (
        "gpu_fault_notification_results",
        "gpu_fault_notification_deliveries",
        "gpu_fault_notifications",
    ):
        sql, pattern = statements[name]
        assert "cluster_name" in sql
        assert pattern == "cluster"


def test_capacity_cleanup_removes_synthetic_links() -> None:
    statements = {
        name: (sql, pattern) for name, sql, pattern in suite.AUDIT_PURGE_STATEMENTS
    }

    sql, pattern = statements["gpu_fault_links"]

    assert "DELETE FROM gpu_fault_links" in sql
    assert "strpos(link.key, pattern.prefix)" in sql
    assert "strpos(link.value, pattern.prefix)" in sql
    assert pattern == "cluster"


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

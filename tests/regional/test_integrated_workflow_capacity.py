from __future__ import annotations

import json
import subprocess
from types import SimpleNamespace

import pytest

from gpu_fault.operation_registry import (
    OPERATION_REGISTRY,
    OperationAdapter,
    OperationScope,
)
from gpu_fault.policy import SxidEvent, XidEvent
from scripts.perf import benchmark_regional_action_executor as executor
from scripts.perf import benchmark_synchronized_burst as burst
from scripts.perf import regional_action_capacity_suite as action_suite
from scripts.perf import regional_integrated_workflow_capacity_suite as suite
from scripts.perf import seed_regional_action_workflows as seed
from tests._builders import build_context


def test_fixed_matrix_keeps_26576_requests_and_selects_four_xids_per_cluster() -> None:
    batches = [
        burst.build_events(
            cluster_count=32,
            cluster_offset=offset,
            nodes_per_cluster=256,
            xid_total=500,
            sxid_total=500,
            gpu_evidence_total=500,
            host_evidence_total=500,
            include_telemetry=True,
        )
        for offset in range(32)
    ]

    assert sum(len(events) for events in batches) == suite.FORMAL_MODELS[32]["total"]
    assert (
        sum(event[0] == "NVIDIA_KERNEL" for events in batches for event in events)
        == 500
    )
    assert (
        sum(event[0] == "FABRIC_MANAGER_LOG" for events in batches for event in events)
        == 500
    )
    assert all(
        sum(event[0] == "NVIDIA_KERNEL" for event in events) >= 3
        and sum(event[0] == "FABRIC_MANAGER_LOG" for event in events) >= 1
        for events in batches
    ), "a cluster lacks the four action-bearing P0 candidates"


def test_integrated_runner_executable_can_render_help() -> None:
    completed = subprocess.run(
        [
            str(suite.PERF_DIR / "regional_integrated_workflow_capacity_suite.py"),
            "--help",
        ],
        cwd=suite.ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--spool-mode" in completed.stdout, completed.stdout


def test_action_bearing_xid_has_complete_causal_identity() -> None:
    identity = burst.integrated_action_identity("run-a", 3, 0)
    payload = burst.build_event_payload(
        templates={},
        kind="NVIDIA_KERNEL",
        template_kind="NVIDIA_KERNEL",
        cluster_id="perf-cap-003",
        node_index=2,
        sequence=1,
        correlate_attempt_faults=False,
        action_identity=identity,
        runtime_profile_version="profile-a",
    )

    event = XidEvent.model_validate(payload)

    assert event.event_id == "integrated-run-a-c003-w00"
    assert event.xid == 48
    assert event.node_id == seed.integrated_node_id("run-a", 3, 0)
    assert event.gpu_uuid == "GPU-integrated-003-00"
    assert event.affected_workload_ids == [
        "training/PyTorchJob/integrated-run-a-c003-w00"
    ]


def test_aurora_preflight_requires_declared_and_observed_124_acu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_aws_json(*arguments: str, **_kwargs):
        operation = arguments[1]
        if operation == "describe-db-clusters":
            return {
                "DBClusters": [
                    {
                        "Status": "available",
                        "ServerlessV2ScalingConfiguration": {
                            "MinCapacity": 124.0,
                            "MaxCapacity": 128.0,
                        },
                        "DBClusterMembers": [
                            {
                                "DBInstanceIdentifier": "writer-a",
                                "IsClusterWriter": True,
                            },
                            {
                                "DBInstanceIdentifier": "reader-a",
                                "IsClusterWriter": False,
                            },
                        ],
                    }
                ]
            }
        if operation == "describe-db-instances":
            return {
                "DBInstances": [
                    {
                        "DBInstanceStatus": "available",
                        "DBInstanceClass": "db.serverless",
                    }
                ]
            }
        return {"Datapoints": [{"Timestamp": "2026-08-31T17:00:00Z", "Maximum": 124.0}]}

    monkeypatch.setattr(suite, "aws_json", fake_aws_json)

    result = suite.aurora_capacity_preflight("aurora-a", timeout_seconds=1)

    assert result["min_acu"] == 124.0
    assert result["max_acu"] == 128.0
    assert {item["actual_acu"] for item in result["instances"].values()} == {124.0}


def test_aurora_fixture_sets_124_128_before_capacity_check(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    modified = []

    def fake_shell_run(arguments, **_kwargs):
        modified.append(arguments)
        return SimpleNamespace(stdout=b"{}")

    def fake_aws_json(*arguments: str, **_kwargs):
        operation = arguments[1]
        if operation == "describe-db-clusters":
            target = bool(modified)
            return {
                "DBClusters": [
                    {
                        "Status": "available",
                        "ServerlessV2ScalingConfiguration": {
                            "MinCapacity": 124.0 if target else 0.5,
                            "MaxCapacity": 128.0 if target else 8.0,
                        },
                        "DBClusterMembers": [
                            {
                                "DBInstanceIdentifier": "writer-a",
                                "IsClusterWriter": True,
                            },
                            {
                                "DBInstanceIdentifier": "reader-a",
                                "IsClusterWriter": False,
                            },
                        ],
                    }
                ]
            }
        if operation == "describe-db-instances":
            return {
                "DBInstances": [
                    {
                        "DBInstanceStatus": "available",
                        "DBInstanceClass": "db.serverless",
                    }
                ]
            }
        return {"Datapoints": [{"Timestamp": "2026-08-31T17:00:00Z", "Maximum": 124.0}]}

    monkeypatch.setattr(suite, "shell_run", fake_shell_run)
    monkeypatch.setattr(suite, "aws_json", fake_aws_json)

    result = suite.ensure_aurora_capacity("aurora-a", timeout_seconds=1, configure=True)

    assert modified, "Aurora modify command was not issued before validation"
    assert "MinCapacity=124,MaxCapacity=128" in modified[0]
    assert result["initial_scaling"] == {"min_acu": 0.5, "max_acu": 8.0}
    assert result["scaling_modified"] is True


def test_remediation_budget_preflight_reads_every_live_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = {
        "GPU_FAULT_REMEDIATION_MAX_ACTIVE_REGION": "128",
        "GPU_FAULT_REMEDIATION_MAX_ACTIVE_PER_CLUSTER": "4",
        "GPU_FAULT_REMEDIATION_MAX_ACTIVE_PER_RESOURCE_CLASS": "4",
        "GPU_FAULT_REMEDIATION_MAX_ACTIVE_PER_NODE": "1",
        "GPU_FAULT_REMEDIATION_MAX_ACTIVE_PER_FAILURE_DOMAIN": "1",
    }

    def fake_control(*arguments: str, **_kwargs):
        if arguments[0] == "get":
            return "worker-a worker-b"
        return json.dumps(expected)

    monkeypatch.setattr(suite, "control", fake_control)

    result = suite.remediation_budget_preflight(32)

    assert result["expected"] == expected
    assert result["pods"] == {"worker-a": expected, "worker-b": expected}


def test_action_mix_covers_reboot_and_diagnostic_fabric_reset() -> None:
    reboot_identity = burst.integrated_action_identity("run-a", 0, 2)
    reboot = XidEvent.model_validate(
        burst.build_event_payload(
            templates={},
            kind="NVIDIA_KERNEL",
            template_kind="NVIDIA_KERNEL",
            cluster_id="perf-cap-000",
            node_index=2,
            sequence=2,
            correlate_attempt_faults=False,
            action_identity=reboot_identity,
            runtime_profile_version="profile-a",
        )
    )
    fabric_identity = burst.integrated_action_identity("run-a", 0, 3)
    fabric = SxidEvent.model_validate(
        burst.build_event_payload(
            templates={},
            kind="FABRIC_MANAGER_LOG",
            template_kind="FABRIC_MANAGER_LOG",
            cluster_id="perf-cap-000",
            node_index=3,
            sequence=3,
            correlate_attempt_faults=False,
            action_identity=fabric_identity,
            runtime_profile_version="profile-a",
        )
    )

    assert reboot.xid == 79
    assert fabric.sxid == 11001
    assert fabric.classification.value == "FATAL"
    assert fabric.fabric_partition.endswith("/local-nvswitch"), fabric.fabric_partition


def test_action_mix_compiles_current_online_dags() -> None:
    operations = []
    lengths = []
    remote_counts = []
    for workflow_index in range(4):
        identity = burst.integrated_action_identity("run-a", 0, workflow_index)
        payload = burst.build_event_payload(
            templates={},
            kind=(
                "FABRIC_MANAGER_LOG"
                if identity["action_kind"] == "FABRIC_RESET"
                else "NVIDIA_KERNEL"
            ),
            template_kind=(
                "FABRIC_MANAGER_LOG"
                if identity["action_kind"] == "FABRIC_RESET"
                else "NVIDIA_KERNEL"
            ),
            cluster_id="perf-cap-000",
            node_index=workflow_index,
            sequence=workflow_index,
            correlate_attempt_faults=False,
            action_identity=identity,
            runtime_profile_version="simulated-v1",
        )
        context = build_context()
        if identity["action_kind"] == "FABRIC_RESET":
            event = SxidEvent.model_validate(payload)
            decision = context.policy.evaluate_sxid(event)
        else:
            event = XidEvent.model_validate(payload)
            decision = context.policy.evaluate_xid(event)
        _, workflow = context.orchestrator.ingest(event, decision)
        current = [step.operation.value for step in workflow.official_steps]
        operations.extend(current)
        lengths.append(len(current))
        remote_counts.append(
            sum(
                OPERATION_REGISTRY[step.operation].scope
                is not OperationScope.CONTROL_PLANE
                and OperationAdapter.GPU_VALIDATION
                not in OPERATION_REGISTRY[step.operation].adapters
                for step in workflow.official_steps
            )
        )

    assert lengths == [10, 10, 9, 12]
    assert remote_counts == [8, 8, 5, 9]
    assert sum(remote_counts) == suite.FORMAL_COMMANDS_PER_CLUSTER
    assert {
        "MARK_UNSCHEDULABLE",
        "STOP_WORKLOADS",
        "COLLECT_DIAGNOSTIC_BUNDLE",
        "RESET_GPU",
        "RESET_ALL_GPUS_NVSWITCHES",
        "RESTART_NODE",
        "RESTART_WORKLOAD",
        "RESTORE_SCHEDULING",
        "VALIDATE_HOST",
    }.issubset(operations), operations


def test_action_setup_adds_context_outside_measured_matrix() -> None:
    payloads = burst.action_context_payloads(
        cluster_id="perf-cap-000",
        run_id="run-a",
        cluster_offset=0,
        workflows_per_cluster=4,
        runtime_profile_version="profile-a",
    )

    assert len(payloads) == 8
    assert [path for path, _ in payloads].count("/v1/workload-observations") == 4
    assert [path for path, _ in payloads].count("/v1/training-progress") == 4
    observations = [
        payload for path, payload in payloads if path == "/v1/workload-observations"
    ]
    assert all(item["restart_budget"] == 1 for item in observations), observations
    assert len({item["attempt_id"] for item in observations}) == 4


def test_setup_context_waits_through_processor_202(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Response:
        def __init__(self, status: int, payload: dict) -> None:
            self.status = status
            self.payload = payload

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self) -> bytes:
            return json.dumps(self.payload).encode()

    responses = iter(
        [
            Response(200, {"processor_request_id": "request-a"}),
            Response(202, {}),
            Response(200, {}),
        ]
    )
    calls = []

    def fake_urlopen(request, **_kwargs):
        calls.append(request)
        return next(responses)

    monkeypatch.setattr(burst.urllib_request, "urlopen", fake_urlopen)
    monkeypatch.setattr(burst.time, "sleep", lambda _seconds: None)

    burst.submit_setup_context(
        base_url="https://control.example",
        registration={"cluster_id": "perf-cap-000", "token": "t" * 32},
        ssl_context=object(),  # type: ignore[arg-type]
        path="/v1/workload-observations",
        payload={"cluster_id": "perf-cap-000"},
    )

    assert len(calls) == 3


def test_integrated_action_selection_reuses_three_xids_and_one_sxid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events = burst.build_events(
        cluster_count=32,
        cluster_offset=0,
        nodes_per_cluster=256,
        xid_total=500,
        sxid_total=500,
        gpu_evidence_total=500,
        host_evidence_total=500,
        include_telemetry=True,
    )
    submitted = []
    monkeypatch.setenv("ACTION_RUN_ID", "run-a")
    monkeypatch.setenv("ACTION_WORKFLOWS_PER_CLUSTER", "4")
    monkeypatch.setenv("RUNTIME_PROFILE_VERSION", "profile-a")
    monkeypatch.setattr(
        burst, "submit_setup_context", lambda **kwargs: submitted.append(kwargs)
    )

    run_id, profile, indexes, setup_count = burst.prepare_integrated_actions(
        base_url="https://control.example",
        registration={"cluster_id": "perf-cap-000", "token": "t" * 32},
        ssl_context=object(),  # type: ignore[arg-type]
        events=events,
        cluster_offset=0,
    )

    assert run_id == "run-a"
    assert profile == "profile-a"
    assert [events[index][0] for index in sorted(indexes)] == [
        "NVIDIA_KERNEL",
        "NVIDIA_KERNEL",
        "NVIDIA_KERNEL",
        "FABRIC_MANAGER_LOG",
    ]
    assert sorted(indexes.values()) == [0, 1, 2, 3]
    assert setup_count == 8
    assert len(submitted) == 8


def test_integrated_load_manifest_carries_action_contract() -> None:
    manifest = suite.build_load_manifest(
        run_id="run-a",
        clusters=32,
        workflows_per_cluster=4,
        runtime_profile_version="profile-a",
        workers=256,
        lead_seconds=20,
    )
    environment = {
        item["name"]: item.get("value")
        for item in manifest["spec"]["template"]["spec"]["containers"][0]["env"]
    }

    assert environment["XID_TOTAL"] == "500"
    assert environment["SXID_TOTAL"] == "500"
    assert environment["GPU_EVIDENCE_TOTAL"] == "500"
    assert environment["HOST_EVIDENCE_TOTAL"] == "500"
    assert environment["ACTION_WORKFLOWS_PER_CLUSTER"] == "4"
    assert environment["ACTION_RUN_ID"] == "run-a"
    assert environment["RUNTIME_PROFILE_VERSION"] == "profile-a"
    assert environment["PREWARM_CONNECTIONS"] == "false"


def test_integrated_executor_uses_idle_drain_and_realistic_simulated_delays() -> None:
    manifest = suite.build_executor_manifest(
        clusters=32,
        executor_workers=8,
        max_seconds=1800,
        idle_exit_seconds=300,
        identity={
            "executor_protocol_version": 2,
            "executor_artifact_sha256": "a" * 64,
            "executor_compatibility_digest": "b" * 64,
        },
    )
    environment = {
        item["name"]: item.get("value")
        for item in manifest["spec"]["template"]["spec"]["containers"][0]["env"]
    }

    assert environment["EXPECTED_COMMANDS"] == "30"
    assert environment["ACTION_IDLE_EXIT_SECONDS"] == "300"
    assert environment["ACTION_MAX_SECONDS"] == "1800"
    assert environment["ACTION_MIN_CONCURRENT_COMMANDS"] == "1"
    assert executor.DELAYS["RESET_GPU"] == 8.0
    assert executor.DELAYS["RESTART_NODE"] == 120.0


def test_executor_log_collection_accepts_integrated_job_name(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = []

    def fake_dataplane(*args, **_kwargs):
        calls.append(args)
        if args[0] == "get":
            return "pod-a 0\n"
        return '{"cluster_id":"perf-cap-000","completed_commands":10}'

    monkeypatch.setattr(action_suite, "dataplane", fake_dataplane)

    documents = action_suite.collect_executor_logs(
        tmp_path, job_name=suite.EXECUTOR_JOB
    )

    assert documents[0]["completed_commands"] == 10
    assert f"batch.kubernetes.io/job-name={suite.EXECUTOR_JOB}" in calls[0]


def test_integrated_verdict_requires_causal_terminal_simulated_workflows() -> None:
    expected = 128
    summary = {
        "clusters": 32,
        "fixed_matrix": {**suite.FORMAL_MODELS[32], "p0": 1000, "p50": 1000},
        "ingress": {
            "requests": 26576,
            "job_status": "Complete",
            "paths": {
                "NVIDIA_KERNEL": {"count": 500, "status_counts": {"200": 500}},
                "FABRIC_MANAGER_LOG": {"count": 500, "status_counts": {"200": 500}},
            },
        },
        "workflow": {
            "incident_count": expected,
            "context_incident_count": expected,
            "workflow_count": expected,
            "succeeded_workflow_count": expected,
            "terminal_command_count": 1024,
            "command_count": 1024,
            "simulated_command_count": 1024,
            "step_count_min": 8,
            "step_count_max": 12,
            "operation_counts": {
                "MARK_UNSCHEDULABLE": 128,
                "STOP_WORKLOADS": 128,
                "COLLECT_DIAGNOSTIC_BUNDLE": 32,
                "RESET_GPU": 64,
                "RESET_ALL_GPUS_NVSWITCHES": 32,
                "RESTART_NODE": 32,
                "RESTORE_SCHEDULING": 128,
            },
            "duplicate_idempotency_keys": 0,
            "duplicate_workflow_steps": 0,
            "fencing_mismatches": 0,
            "permanent_budget_waiters": 0,
        },
        "executor": {
            "expected_commands": 960,
            "completed_commands": 960,
            "duplicate_claims": 0,
            "claim_errors": 0,
            "result_errors": 0,
            "long_commands": 32,
            "renewals": 32,
        },
    }

    status, errors = suite.verdict(summary, expected_workflows=expected)

    assert status == "PASS", errors
    assert errors == []


@pytest.mark.parametrize("status_code", ("429", "503"))
def test_integrated_verdict_rejects_ingress_http_errors(status_code: str) -> None:
    expected = 128
    summary = {
        "clusters": 32,
        "fixed_matrix": {**suite.FORMAL_MODELS[32], "p0": 1000, "p50": 1000},
        "ingress": {
            "requests": 26576,
            "job_status": "Complete",
            "paths": {
                "NVIDIA_KERNEL": {"count": 500, "status_counts": {"202": 500}},
                "FABRIC_MANAGER_LOG": {"count": 500, "status_counts": {"202": 500}},
                "GPU_METRICS": {
                    "count": 8192,
                    "status_counts": {"202": 8191, status_code: 1},
                },
            },
        },
        "workflow": {
            "incident_count": expected,
            "context_incident_count": expected,
            "workflow_count": expected,
            "succeeded_workflow_count": expected,
            "terminal_command_count": 960,
            "command_count": 960,
            "simulated_command_count": 960,
            "step_count_min": 8,
            "step_count_max": 12,
            "operation_counts": {
                "MARK_UNSCHEDULABLE": 128,
                "STOP_WORKLOADS": 128,
                "COLLECT_DIAGNOSTIC_BUNDLE": 32,
                "RESET_GPU": 64,
                "RESET_ALL_GPUS_NVSWITCHES": 32,
                "RESTART_NODE": 32,
                "RESTORE_SCHEDULING": 128,
            },
            "duplicate_idempotency_keys": 0,
            "duplicate_workflow_steps": 0,
            "fencing_mismatches": 0,
            "permanent_budget_waiters": 0,
        },
        "executor": {
            "expected_commands": 960,
            "completed_commands": 960,
            "duplicate_claims": 0,
            "claim_errors": 0,
            "result_errors": 0,
            "long_commands": 32,
            "renewals": 32,
        },
    }

    status, errors = suite.verdict(summary, expected_workflows=expected)

    assert status == "FAIL"
    assert f"GPU_METRICS returned HTTP {status_code}" in errors


def test_50_cluster_manifest_uses_formal_41524_matrix() -> None:
    manifest = suite.build_load_manifest(
        run_id="run-a",
        clusters=50,
        workflows_per_cluster=4,
        runtime_profile_version="profile-a",
        workers=256,
        lead_seconds=20,
    )
    environment = {
        item["name"]: item.get("value")
        for item in manifest["spec"]["template"]["spec"]["containers"][0]["env"]
    }

    assert environment["XID_TOTAL"] == "781"
    assert environment["SXID_TOTAL"] == "781"
    assert environment["GPU_EVIDENCE_TOTAL"] == "781"
    assert environment["HOST_EVIDENCE_TOTAL"] == "781"
    assert suite.FORMAL_MODELS[50]["total"] == 41524


@pytest.mark.parametrize(
    ("mode", "values", "replicas"),
    [("disabled", ["false", "false"], "0"), ("enabled", ["true", "true"], "3")],
)
def test_spool_mode_requires_consistent_live_ingress(
    mode: str, values: list[str], replicas: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    ingress_pods = ["ingress-a", "ingress-b"]
    spool_pods = [f"spool-{index}" for index in range(int(replicas))]

    def fake_control(*args, **_kwargs):
        if args[:2] == ("get", "pod"):
            selector = args[args.index("-l") + 1]
            return (
                " ".join(spool_pods)
                if "telemetry-spool-worker" in selector
                else " ".join(ingress_pods)
            )
        if args[:2] == ("get", "deployment"):
            return replicas
        if args[0] == "exec":
            if args[1] in ingress_pods:
                return values[ingress_pods.index(args[1])]
            return "true"
        raise AssertionError(args)

    monkeypatch.setattr(suite, "control", fake_control)

    report = suite.control_spool_mode(mode)

    assert report["expected"] == mode
    assert report["spool_worker_replicas"] == int(replicas)
    assert len(report["spool_worker_values"]) == int(replicas)


def test_integrated_embedded_database_scripts_compile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scripts = []

    def fake_control(*args, **_kwargs):
        if "-c" in args:
            scripts.append(args[args.index("-c") + 1])
        return "{}"

    monkeypatch.setattr(suite, "control", fake_control)

    assert suite.workflow_audit("run-a") == {}
    suite.purge_integrated_rows("run-a")
    assert len(scripts) == 2
    for index, script in enumerate(scripts):
        compile(script, f"integrated-script-{index}", "exec")


def test_integrated_heartbeat_refresher_is_throttled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []
    times = iter((0.0, 10.0, 30.0))
    monkeypatch.setattr(suite.time, "monotonic", lambda: next(times))
    monkeypatch.setattr(
        suite, "refresh_integrated_agents", lambda **kwargs: calls.append(kwargs) or {}
    )
    refresh = suite.integrated_agent_heartbeat_refresher(
        run_id="run-a",
        clusters=32,
        workflows_per_cluster=4,
        lease_seconds=2700,
        interval_seconds=30,
    )

    refresh()
    refresh()
    refresh()

    assert len(calls) == 2
    assert calls[0]["run_id"] == "run-a"
    assert calls[1]["lease_seconds"] == 2700


def test_integrated_cleanup_signal_guard_ignores_and_restores_signals(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    guarded = tuple(
        getattr(suite.signal, name) for name in ("SIGINT", "SIGTERM", "SIGHUP")
    )
    original = {value: object() for value in guarded}
    active = dict(original)

    monkeypatch.setattr(suite.signal, "getsignal", lambda value: active[value])
    monkeypatch.setattr(
        suite.signal,
        "signal",
        lambda value, handler: active.__setitem__(value, handler),
    )

    with suite.cleanup_signal_guard():
        assert all(active[value] is suite.signal.SIG_IGN for value in guarded), (
            "cleanup did not ignore termination signals"
        )

    assert active == original


def test_integrated_workload_cleanup_waits_for_jobs_and_pods(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    samples = iter(
        ({"jobs": ["job/load"], "pods": ["pod/load"]}, {"jobs": [], "pods": []})
    )
    clock = iter((0.0, 0.1))
    sleeps = []
    monkeypatch.setattr(suite, "integrated_workload_residuals", lambda: next(samples))
    monkeypatch.setattr(suite.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(suite.time, "sleep", lambda seconds: sleeps.append(seconds))

    result = suite.wait_for_integrated_workload_cleanup(
        timeout_seconds=1, sample_seconds=2
    )

    assert result == {"jobs": [], "pods": []}
    assert sleeps == [2]


def test_integrated_runner_refuses_non_live_registry(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(suite, "validate_registry_target", lambda **_kwargs: "isolated")
    args = SimpleNamespace(
        clusters=32,
        workflows_per_cluster=4,
        allow_live_registry=True,
        confirm_live_registry="ALLOW_PERF_CAPACITY_LIVE_REGISTRY",
        suite_id="run-a",
        artifact_root=tmp_path,
        label=None,
    )

    with pytest.raises(ValueError, match="live registry"):
        suite.run(args)

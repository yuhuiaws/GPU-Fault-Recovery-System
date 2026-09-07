from __future__ import annotations

import importlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import audit_destr013_replacement_invariant as destr013
from scripts.e2e.regional import regional_live_fixture as live_fixture_module
from scripts.e2e.regional import run_collector_destructive as collector_destructive
from scripts.e2e.regional import run_destr001_gpu_reset as destr001
from scripts.e2e.regional import run_destr002_hyperpod_reboot as destr002
from scripts.e2e.regional import run_destr003_warm_spare_failover as destr003
from scripts.e2e.regional import run_destr008_warm_spare_shortage as destr008
from scripts.e2e.regional import run_destr009_workload_restart as destr009
from scripts.e2e.regional import run_destr012_managed_recovery_guard as destr012
from scripts.e2e.regional import run_destr014_branch_exhaustion as destr014
from scripts.e2e.regional import run_destr015_parallel_branch_join as destr015
from scripts.e2e.regional import run_destr016_preempting_reboot as destr016
from scripts.e2e.regional import run_destr017_out_of_band_reboot_fence as destr017
from scripts.e2e.regional import run_destr018_lifetime_deadline as destr018
from scripts.e2e.regional import run_workload_acceptance as workload_acceptance
from scripts.e2e.regional.acceptance_scope import (
    EXECUTION_SCOPE_ENV,
    SELECTION_REFERENCE_ENV,
)
from scripts.e2e.regional.managed_workload_fixture import (
    TRAINING_IMAGE,
    ImagePrewarmFixture,
    ManagedWorkloadFixture,
    ManagedWorkloadSettings,
    render_node_pinned_manifest,
)
from scripts.e2e.regional.regional_live_fixture import (
    RegionalLiveFixture,
    RegionalLiveSettings,
    predecessor_evidence,
    provider_event_actor_matches_role,
    runtime_identity_errors,
)
from scripts.e2e.regional.warm_spare_fixture import (
    GpuHolderFixture,
    WarmSpareLiveFixture,
)

yaml = importlib.import_module("yaml")


ROOT = Path(__file__).resolve().parents[2]
REGIONAL = ROOT / "scripts/e2e/regional"


def _regional(tmp_path: Path) -> RegionalLiveFixture:
    cpu = tmp_path / "cpu.kubeconfig"
    gpu = tmp_path / "gpu.kubeconfig"
    cpu.write_text("apiVersion: v1\n", encoding="utf-8")
    gpu.write_text("apiVersion: v1\n", encoding="utf-8")
    return RegionalLiveFixture(
        RegionalLiveSettings(
            cpu_kubeconfig=cpu,
            gpu_kubeconfig=gpu,
            gpu_context="gpu-context",
            namespace="gpu-fault-system",
            cluster_id="cluster-a",
            region="us-west-2",
        )
    )


def test_destructive_live_drivers_are_promoted_and_plan_only() -> None:
    expected = {
        "run_destr001_gpu_reset.py": destr001.CONFIRMATION,
        "run_destr002_hyperpod_reboot.py": destr002.CONFIRMATION,
        "run_destr003_warm_spare_failover.py": destr003.CONFIRMATION,
        "run_destr008_warm_spare_shortage.py": destr008.CONFIRMATION,
        "run_destr009_workload_restart.py": destr009.CONFIRMATION,
        "run_destr012_managed_recovery_guard.py": destr012.CONFIRMATION,
        "run_destr014_branch_exhaustion.py": destr014.CONFIRMATION,
        "run_destr015_parallel_branch_join.py": destr015.CONFIRMATION,
        "run_destr016_preempting_reboot.py": destr016.CONFIRMATION,
        "run_destr017_out_of_band_reboot_fence.py": destr017.CONFIRMATION,
        "run_destr018_lifetime_deadline.py": destr018.CONFIRMATION,
    }

    for name, confirmation in expected.items():
        path = REGIONAL / name
        assert path.is_file(), f"missing destructive live driver: {name}"
        source = path.read_text(encoding="utf-8")
        assert "/secure/gpu-fault-bootstrap" not in source, name
        assert "514385905925" not in source, name
        module = {
            "run_destr001_gpu_reset.py": destr001,
            "run_destr002_hyperpod_reboot.py": destr002,
            "run_destr003_warm_spare_failover.py": destr003,
            "run_destr008_warm_spare_shortage.py": destr008,
            "run_destr009_workload_restart.py": destr009,
            "run_destr012_managed_recovery_guard.py": destr012,
            "run_destr014_branch_exhaustion.py": destr014,
            "run_destr015_parallel_branch_join.py": destr015,
            "run_destr016_preempting_reboot.py": destr016,
            "run_destr017_out_of_band_reboot_fence.py": destr017,
            "run_destr018_lifetime_deadline.py": destr018,
        }[name]
        arguments = module.parser().parse_args(["--run-dir", "/tmp/test-run"])
        assert arguments.execute is False, name
        assert confirmation.startswith("DESTR"), confirmation
        assert hasattr(arguments, "predecessor_evidence"), name


def test_predecessor_evidence_requires_exact_pass(tmp_path: Path) -> None:
    path = tmp_path / "predecessor.json"

    missing = predecessor_evidence(path, "GF-REGIONAL-DESTR-010")
    assert missing["valid"] is False, missing
    path.write_text(
        json.dumps({"case_id": "GF-REGIONAL-DESTR-010", "verdict": "FAIL"}),
        encoding="utf-8",
    )
    failed = predecessor_evidence(path, "GF-REGIONAL-DESTR-010")
    assert failed["valid"] is False, failed
    path.write_text(
        json.dumps({"case_id": "GF-REGIONAL-DESTR-010", "verdict": "PASS"}),
        encoding="utf-8",
    )
    passed = predecessor_evidence(path, "GF-REGIONAL-DESTR-010")
    assert passed["valid"] is True, passed


def test_selective_predecessor_is_explicitly_skipped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(EXECUTION_SCOPE_ENV, "selective")
    monkeypatch.setenv(SELECTION_REFERENCE_ENV, "CHG-DESTR001-SELECTIVE")

    result = predecessor_evidence(tmp_path / "missing.json", "GF-REGIONAL-DESTR-010")

    assert result["valid"] is True, result
    assert result["execution_allowed"] is True, result
    assert result["evidence_valid"] is False, result
    assert result["verdict"] == "SKIPPED_BY_OPERATOR", result
    assert result["execution_scope"] == "selective", result
    assert result["formal_sequence_satisfied"] is False, result
    assert result["selection_reference"] == "CHG-DESTR001-SELECTIVE", result


def test_formal_predecessor_rejects_selective_pass(tmp_path: Path) -> None:
    path = tmp_path / "predecessor.json"
    path.write_text(
        json.dumps(
            {
                "case_id": "GF-REGIONAL-DESTR-010",
                "verdict": "PASS",
                "execution_scope": "selective",
                "selection_reference": "CHG-OLD-SELECTIVE",
                "formal_sequence_satisfied": False,
            }
        ),
        encoding="utf-8",
    )

    result = predecessor_evidence(path, "GF-REGIONAL-DESTR-010")

    assert result["valid"] is False, result
    assert result["evidence_valid"] is True, result
    assert result["formal_sequence_satisfied"] is False, result
    assert "selective evidence" in result["error"], result


def test_image_prewarm_is_node_pinned_without_gpu_request(tmp_path: Path) -> None:
    fixture = ImagePrewarmFixture(
        _regional(tmp_path), case_id="GF-REGIONAL-DESTR-009", run_id="run-a"
    )

    manifest = fixture.manifest("node-a", 0)
    container = manifest["spec"]["containers"][0]

    assert manifest["spec"]["nodeName"] == "node-a", manifest
    assert manifest["spec"]["activeDeadlineSeconds"] == 1800, manifest
    assert container["image"] == TRAINING_IMAGE, container
    assert "nvidia.com/gpu" not in container["resources"]["requests"], container
    assert "nvidia.com/gpu" not in container["resources"]["limits"], container


def test_idle_node_check_keeps_gpu_workloads_in_system_namespace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    regional = _regional(tmp_path)
    payload = {
        "items": [
            {
                "metadata": {
                    "namespace": "gpu-fault-system",
                    "name": "gpu-fault-cluster-executor-a",
                },
                "spec": {"containers": [{"resources": {}}]},
            },
            {
                "metadata": {"namespace": "gpu-fault-system", "name": "training-a"},
                "spec": {
                    "containers": [{"resources": {"requests": {"nvidia.com/gpu": "8"}}}]
                },
            },
        ]
    }
    monkeypatch.setattr(
        regional, "kubectl", lambda *_args, **_kwargs: json.dumps(payload)
    )

    assert regional.business_workloads("node-a") == [
        {"namespace": "gpu-fault-system", "name": "training-a"}
    ]


def test_all_namespaces_flag_follows_kubectl_verb(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    regional = _regional(tmp_path)
    commands: list[list[str]] = []

    def fake_run(
        command: list[str], **_kwargs: Any
    ) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="{}", stderr="")

    monkeypatch.setattr(regional, "run", fake_run)

    regional.kubectl("gpu", "get", "pod", "-o", "json", all_namespaces=True)

    assert commands[0].index("get") < commands[0].index("--all-namespaces"), commands


def test_store_snapshot_falls_back_to_release_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    regional = _regional(tmp_path)
    monkeypatch.setattr(
        regional, "cpu_python", lambda *_args, **_kwargs: {"release_id": None}
    )
    monkeypatch.setattr(regional, "release_id", lambda: "release-a")

    result = regional.store_snapshot(node="node-a")

    assert result["release_id"] == "release-a", result


def test_provider_events_use_cloudtrail_session_issuer_role(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    regional = _regional(tmp_path)
    expected_role = "arn:aws:iam::123456789012:role/path/executor-role"
    cloudtrail_event = json.dumps(
        {
            "userIdentity": {
                "type": "AssumedRole",
                "sessionContext": {"sessionIssuer": {"arn": expected_role}},
            }
        }
    )
    payload = {
        "Events": [
            {
                "EventName": "BatchRebootClusterNodes",
                "EventTime": "2026-09-01T00:00:00Z",
                "Username": "pod-session",
                "CloudTrailEvent": cloudtrail_event,
            }
        ]
    }
    monkeypatch.setattr(
        regional,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            [], 0, stdout=json.dumps(payload), stderr=""
        ),
    )

    events = regional.provider_events(
        datetime.now(timezone.utc), datetime.now(timezone.utc)
    )

    assert events[0]["session_issuer_role_name"] == "executor-role", events
    assert provider_event_actor_matches_role(events[0], expected_role), events
    assert not provider_event_actor_matches_role(
        events[0], "arn:aws:iam::123456789012:role/other-role"
    ), events


def test_destr002_executor_restart_does_not_wait_for_deleted_pod(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    regional = _regional(tmp_path)
    snapshots = iter(
        (
            [
                {"name": "executor-a", "uid": "uid-a", "node": "node-a"},
                {"name": "executor-b", "uid": "uid-b", "node": "node-b"},
            ],
            [
                {"name": "executor-b", "uid": "uid-b", "node": "node-b"},
                {"name": "executor-c", "uid": "uid-c", "node": "node-c"},
            ],
        )
    )
    delete_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    monkeypatch.setattr(regional, "ready_pods", lambda *_args: next(snapshots))
    monkeypatch.setattr(
        regional,
        "kubectl",
        lambda *args, **kwargs: delete_calls.append((args, kwargs)) or "",
    )

    result = destr002.restart_executor(
        regional, {"result_details": {"executor_id": "lease/executor-a"}}
    )

    assert result["deleted"]["uid"] == "uid-a", result
    assert delete_calls == [
        (("gpu", "delete", "pod", "executor-a", "--wait=false"), {"timeout": 30})
    ], delete_calls


def test_wait_for_workflow_preserves_observed_waiting_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    regional = _regional(tmp_path)
    snapshots = iter(
        (
            {
                "event": {"event_id": "event-a"},
                "decision": {"disposition": "EXECUTABLE"},
                "workflow": {
                    "status": "RUNNING",
                    "completed_operations": [],
                    "step_executions": [
                        {
                            "step_index": 2,
                            "operation": "RESET_GPU",
                            "status": "WAITING",
                            "adapter_operation_id": "remote/command-a",
                            "details": {"mutation_submitted_by_control_plane": False},
                        }
                    ],
                },
                "commands": [{"status": "WAITING"}],
            },
            {
                "event": {"event_id": "event-a"},
                "decision": {"disposition": "EXECUTABLE"},
                "workflow": {
                    "status": "SUCCEEDED",
                    "completed_operations": ["RESET_GPU"],
                    "step_executions": [
                        {
                            "step_index": 2,
                            "operation": "RESET_GPU",
                            "status": "SUCCEEDED",
                            "adapter_operation_id": "remote/command-a",
                            "details": {"reset_attempts": 1},
                        }
                    ],
                },
                "commands": [{"status": "SUCCEEDED"}],
            },
        )
    )
    monkeypatch.setattr(regional, "store_snapshot", lambda **_kwargs: next(snapshots))
    monkeypatch.setattr(live_fixture_module.time, "sleep", lambda _seconds: None)

    result = regional.wait_for_workflow(
        node="node-a",
        marker="marker-a",
        observed_after=datetime.now(timezone.utc),
        case_dir=tmp_path,
        timeout_seconds=5,
    )

    assert result["workflow"]["status"] == "SUCCEEDED", result
    assert result["observed_waiting_step_executions"] == [
        {
            "step_index": 2,
            "operation": "RESET_GPU",
            "status": "WAITING",
            "adapter_operation_id": "remote/command-a",
            "details": {"mutation_submitted_by_control_plane": False},
            "error": None,
        }
    ], result
    timeline = json.loads((tmp_path / "timeline.json").read_text(encoding="utf-8"))
    assert (
        timeline["observed_waiting_step_executions"]
        == result["observed_waiting_step_executions"]
    ), timeline


def test_managed_workload_fixture_uses_source_manifest_identity(tmp_path: Path) -> None:
    site = tmp_path / "site.yaml"
    site.write_text("schemaVersion: 1\n", encoding="utf-8")
    fixture = ManagedWorkloadFixture(
        _regional(tmp_path),
        ManagedWorkloadSettings(
            manifest=(REGIONAL / "manifests/training/xid11-single-node-job.yaml"),
            site_file=site,
            job_id="job-a",
            attempt_id="job-a-a001",
            restart_budget=1,
            expected_pods=1,
            expected_gpu_count=1,
        ),
    )

    assert fixture.kind == "Job", fixture.kind
    assert fixture.name == "gpu-fault-xid11-auto-resume-guard", fixture.name
    assert fixture.resource == "job", fixture.resource


def test_node_pinned_manifest_updates_every_pytorch_template(tmp_path: Path) -> None:
    source = REGIONAL / "manifests/training/single-node-warm-spare-pytorchjob.yaml"
    destination = tmp_path / "pinned.yaml"

    render_node_pinned_manifest(source, destination, node="node-a")
    document = yaml.safe_load(destination.read_text(encoding="utf-8"))
    templates = document["spec"]["pytorchReplicaSpecs"]

    assert templates, document
    assert all(
        item["template"]["spec"]["nodeName"] == "node-a" for item in templates.values()
    ), templates
    assert destination.stat().st_mode & 0o777 == 0o600, destination.stat()


def test_warm_spare_gpu_holder_is_node_pinned_and_bounded(tmp_path: Path) -> None:
    warm = WarmSpareLiveFixture(_regional(tmp_path), "hp-cluster")
    holder = GpuHolderFixture(warm, node="spare-a", run_id="run-a")

    manifest = holder.manifest()
    container = manifest["spec"]["containers"][0]

    assert manifest["spec"]["nodeName"] == "spare-a", manifest
    assert manifest["spec"]["activeDeadlineSeconds"] == 900, manifest
    assert container["resources"]["requests"]["nvidia.com/gpu"] == "1", container
    assert manifest["spec"]["tolerations"] == [{"operator": "Exists"}], manifest


def _reset_state() -> dict[str, Any]:
    waiting = []
    for operation in (
        "QUIESCE_GPU_SERVICES",
        "VERIFY_NO_GPU_CLIENTS",
        "RESET_GPU",
        "RESTORE_GPU_SERVICES",
    ):
        waiting.append(
            {
                "operation": operation,
                "status": "WAITING",
                "details": {"mutation_submitted_by_control_plane": False},
            }
        )
    return {
        "event": {"xid": 46, "evidence_ref": "kmsg://node/boot/1"},
        "decision": {"official_action": "RESET_GPU"},
        "workflow": {
            "status": "SUCCEEDED",
            "official_steps": [
                {"operation": operation} for operation in destr001.EXPECTED_STEPS
            ],
            "completed_operations": list(destr001.EXPECTED_STEPS),
            "step_executions": [
                {
                    "operation": operation,
                    "status": "SUCCEEDED",
                    "adapter_operation_id": f"remote/{operation.lower()}",
                }
                for operation in (
                    "QUIESCE_GPU_SERVICES",
                    "VERIFY_NO_GPU_CLIENTS",
                    "RESET_GPU",
                    "RESTORE_GPU_SERVICES",
                )
            ],
        },
        "observed_waiting_step_executions": waiting,
        "commands": [
            {"status": "SUCCEEDED", "step": {"operation": operation}}
            for operation in (
                "MARK_UNSCHEDULABLE",
                "QUIESCE_GPU_SERVICES",
                "VERIFY_NO_GPU_CLIENTS",
                "RESET_GPU",
                "RESTORE_GPU_SERVICES",
                "RESTORE_SCHEDULING",
            )
        ],
    }


def test_destr001_requires_the_exact_reset_contract() -> None:
    state = _reset_state()

    assert destr001.workflow_errors(state) == [], state
    # A step that finished inside one poll is never seen WAITING; its terminal
    # `remote/` adapter id is equivalent evidence that the GPU-side executor,
    # not the control plane, performed the mutation (COLLECT-016 D, QUIESCE).
    state["observed_waiting_step_executions"] = state[
        "observed_waiting_step_executions"
    ][:-1]
    assert destr001.workflow_errors(state) == [], state
    # Without either form of evidence the contract still fails.
    for execution in state["workflow"]["step_executions"]:
        if execution["operation"] == "RESTORE_GPU_SERVICES":
            execution["adapter_operation_id"] = "workflow-local/6"
    errors = destr001.workflow_errors(state)
    assert any("WAITING evidence" in error for error in errors), errors


def test_destr001_reset_contract_takes_the_workload_step_sequence() -> None:
    """COLLECT-016 D resets a node that runs the managed workload: the
    compiler wraps the idle-node sequence in STOP_WORKLOADS/RESTART_WORKLOAD,
    and the contract must be told so instead of failing on the extra steps."""

    from scripts.e2e.regional import run_collect016_training_recovery as collect016

    steps = collect016.WORKLOAD_RESET_STEPS
    assert [
        s for s in steps if s not in {"STOP_WORKLOADS", "RESTART_WORKLOAD"}
    ] == list(destr001.EXPECTED_STEPS)
    assert steps.index("STOP_WORKLOADS") < steps.index("QUIESCE_GPU_SERVICES")
    assert steps[-1] == "RESTART_WORKLOAD"

    state = _reset_state()
    state["event"]["xid"] = 109
    state["workflow"]["official_steps"] = [{"operation": s} for s in steps]
    state["workflow"]["completed_operations"] = list(steps)
    assert any(
        "reset contract" in error for error in destr001.workflow_errors(state, xid=109)
    ), "an XID 109 workflow with reset steps must name the reset contract"
    assert destr001.workflow_errors(state, xid=109, expected_steps=steps) == [], state


def test_destr002_preflight_and_reboot_contract() -> None:
    safe = {"safe_to_submit": True, "node_recovery": "None", "gate_failures": []}
    missing = {
        "safe_to_submit": False,
        "gate_failures": [
            "trusted scheduler isolation evidence is missing for: node-a"
        ],
    }
    disabled = {
        "safe_to_submit": False,
        "gate_failures": ["HyperPod REBOOT mutation is disabled by configuration"],
    }
    probe = {
        "configured_cluster_name": "hp-a",
        "positive": safe,
        "missing_isolation": missing,
        "wrong_isolation": missing,
        "disabled": disabled,
    }
    state: dict[str, Any] = {
        "event": {"xid": 79},
        "decision": {"action": "REBOOT_NODE"},
        "workflow": {
            "status": "SUCCEEDED",
            "official_steps": [
                {"operation": operation}
                for operation in (
                    "MARK_UNSCHEDULABLE",
                    "RESTART_NODE",
                    "VALIDATE_GPU",
                    "VALIDATE_HOST",
                    "VALIDATE_FABRIC",
                    "RESTORE_SCHEDULING",
                )
            ],
        },
        "submission": {"state": "SUBMITTED", "action": "REBOOT"},
        "agent": {
            "lifecycle_state": "ACTIVE",
            "artifact_sha256": "a" * 64,
            "boot_id": "boot-new",
        },
    }

    assert destr002.preflight_probe_errors(probe, "hp-a") == [], probe
    assert (
        destr002.workflow_errors(
            state, expected_artifact="a" * 64, expected_boot_id="boot-old"
        )
        == []
    ), state


def test_destr002_allows_transient_zero_gpu_capacity_before_validation() -> None:
    baseline = {"uid": "uid-a", "boot_id": "boot-old", "gpu_allocatable": "8"}
    first_ready = {"uid": "uid-a", "boot_id": "boot-new", "gpu_allocatable": "0"}
    final = {"uid": "uid-a", "boot_id": "boot-new", "gpu_allocatable": "8"}

    assert destr002.node_recovery_errors(baseline, first_ready, final) == [], final


@pytest.mark.parametrize(
    ("first_ready", "final", "expected_error"),
    [
        (
            {"uid": "uid-new", "boot_id": "boot-new", "gpu_allocatable": "0"},
            {"uid": "uid-new", "boot_id": "boot-new", "gpu_allocatable": "8"},
            "Kubernetes Node UID changed across reboot",
        ),
        (
            {"uid": "uid-a", "boot_id": "boot-new", "gpu_allocatable": "0"},
            {"uid": "uid-a", "boot_id": "boot-new", "gpu_allocatable": "0"},
            "target GPU capacity did not return to baseline after validation",
        ),
    ],
)
def test_destr002_rejects_identity_or_final_gpu_capacity_drift(
    first_ready: dict[str, Any], final: dict[str, Any], expected_error: str
) -> None:
    baseline = {"uid": "uid-a", "boot_id": "boot-old", "gpu_allocatable": "8"}

    errors = destr002.node_recovery_errors(baseline, first_ready, final)

    assert expected_error in errors, errors


def test_destr002_wait_stops_on_terminal_workflow_without_submission(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    regional = _regional(tmp_path)
    state = {
        "workflow": {"status": "SUCCEEDED", "completed_operations": ["RESTART_NODE"]},
        "submission": None,
    }
    monkeypatch.setattr(regional, "store_snapshot", lambda **_kwargs: state)
    ticks = iter([0.0, 0.0])
    monkeypatch.setattr(destr002.time, "monotonic", lambda: next(ticks))
    settings = destr002.Settings(
        regional=regional.settings,
        node="node-a",
        host_probe_image="registry.example/probe@sha256:" + "a" * 64,
        hyperpod_cluster="hp-cluster",
        executor_role_arn="arn:aws:iam::123456789012:role/executor",
        predecessor_path=tmp_path / "predecessor.json",
    )

    observed = destr002.wait_for_submission(
        regional,
        settings,
        marker="marker-a",
        observed_after=datetime.now(timezone.utc),
        case_dir=tmp_path,
        timeout_seconds=5,
    )

    assert observed is state


def test_destr002_duplicate_replay_requires_exact_submitted_record() -> None:
    """The replay is a store read from the replacement executor: the record
    under the command's recorded key must be SUBMITTED with a result and the
    same request identity. It never submits and never recomputes the key."""

    source = destr002.STORE_REPLAY_PROBE

    assert "get_hyperpod_submission" in source
    assert 'record.state == "SUBMITTED"' in source
    assert "record.result is not None" in source
    assert "record.request_identity == expected_identity" in source
    assert 'command["result_details"]["submission_idempotency_key"]' in source
    assert ".submit(" not in source
    assert "hyperpod_submission_idempotency_key" not in source
    assert not hasattr(destr002, "DIRECT_DUPLICATE_REPLAY"), (
        "DESTR-002 no longer ships the direct duplicate-replay probe"
    )
    assert "hyperpod_submission_idempotency_key" in live_fixture_module.STORE_PROBE


def test_destr002_redacts_lease_tokens_from_evidence() -> None:
    source = {
        "commands": [
            {"command_id": "command-a", "lease_token": "sensitive-lease-token-value"}
        ]
    }

    redacted = destr002.redact_lease_tokens(source)

    assert "lease_token" not in redacted["commands"][0], redacted
    assert redacted["commands"][0]["lease_token_length"] == 27, redacted
    assert len(redacted["commands"][0]["lease_token_sha256"]) == 64, redacted
    assert source["commands"][0]["lease_token"] == "sensitive-lease-token-value", source


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("NVIDIA H200", "H200"),
        ("ml.p5.48xlarge", "H100"),
        ("p5.48xlarge", "H100"),
        ("ml.p5e.48xlarge", "H200"),
        ("ml.p5en.48xlarge", "H200"),
        ("ml.p6-b200.48xlarge", "B200"),
        ("ml.p6-b300.48xlarge", "B300"),
    ],
)
def test_destr009_normalizes_gpu_product_or_instance_type(
    value: str, expected: str
) -> None:
    assert destr009.normalize_product(value) == expected, value


def test_destr009_rejects_unknown_gpu_product() -> None:
    with pytest.raises(
        destr009.RegionalFixtureError, match="cannot normalize GPU product"
    ):
        destr009.normalize_product("ml.unknown.48xlarge")


@pytest.mark.parametrize(
    "source",
    [
        live_fixture_module.EXECUTOR_XID_POST,
        workload_acceptance.OBSERVATION_POST_PROBE,
        collector_destructive.FABRIC_POST,
    ],
)
def test_executor_event_probes_scope_private_ca_to_the_probe_process(
    source: str,
) -> None:
    ca_index = source.index(
        'os.environ["SSL_CERT_FILE"] = os.environ["GPU_FAULT_CONTROL_PLANE_CA_FILE"]'
    )
    sink_index = source.index("sink = HttpEventSink(")

    assert ca_index < sink_index, source


def _destr003_settings(tmp_path: Path) -> destr003.Settings:
    return destr003.Settings(
        regional=_regional(tmp_path).settings,
        site_file=tmp_path / "site.yaml",
        manifest=REGIONAL / "manifests/training/single-node-warm-spare-pytorchjob.yaml",
        hyperpod_cluster="hp-cluster",
        fault_node="node-a",
        spare_node="node-b",
        job_id="job-a",
        attempt_id="job-a-a001",
        predecessor_path=tmp_path / "predecessor.json",
    )


def test_destr003_requires_local_warm_spare_rebinding_and_notification(
    tmp_path: Path,
) -> None:
    settings = _destr003_settings(tmp_path)
    state: dict[str, Any] = {
        "workflow": {
            "status": "SUCCEEDED",
            "official_steps": [
                {
                    "operation": operation,
                    "parameters": (
                        {"replacement_strategy": "HEALTHY_WARM_SPARE_ONLY"}
                        if operation == "REPLACE_NODE"
                        else {}
                    ),
                }
                for operation in destr003.EXPECTED_OPERATIONS
            ],
            "step_executions": [
                {
                    "operation": "REPLACE_NODE",
                    "status": "SUCCEEDED",
                    "details": {
                        "action": "SPARE_FAILOVER",
                        "activated_spare_nodes": ["node-b"],
                        "node_rebindings": {"node-a": "node-b"},
                        "provider_mutation_submitted": False,
                        "notification_id": "notification-a",
                    },
                },
                {
                    "operation": "RESTART_WORKLOAD",
                    "status": "SUCCEEDED",
                    # Nested exactly as `KubernetesWorkloadOperations` emits it.
                    # A flat fixture made this test and the runner agree with
                    # each other and both disagree with the product: the live
                    # DESTR-003 run failed on three counts that were in fact
                    # correct, one level down.
                    "details": {
                        "notification_context": {
                            "source_gpu_count": 8,
                            "target_gpu_count": 8,
                            "restart_count": 1,
                        }
                    },
                },
            ],
        },
        "notifications": [{"notification_id": "notification-a"}],
    }

    assert destr003.workflow_errors(state, settings) == [], state
    state["workflow"]["step_executions"][0]["details"][
        "provider_mutation_submitted"
    ] = True
    errors = destr003.workflow_errors(state, settings)
    assert any("provider mutation" in error for error in errors), errors


def test_destr003_rejects_a_restart_that_reported_no_gpu_counts(tmp_path: Path) -> None:
    # The adapter emits `notification_context` only once the restart produced a
    # restarted attempt, so its absence means the workload did not come back --
    # which must read as one clear failure, not as three "is not 8" complaints
    # about keys that were never there.
    settings = _destr003_settings(tmp_path)
    state: dict[str, Any] = {
        "workflow": {
            "status": "SUCCEEDED",
            "official_steps": [
                {"operation": operation, "parameters": {}}
                for operation in destr003.EXPECTED_OPERATIONS
            ],
            "step_executions": [
                {
                    "operation": "RESTART_WORKLOAD",
                    "status": "SUCCEEDED",
                    "details": {"workloads": ["ns/pytorchjob/job"], "suspended": False},
                }
            ],
        },
        "notifications": [],
    }

    errors = destr003.workflow_errors(state, settings)

    assert "RESTART_WORKLOAD reported no restart notification context" in errors, errors
    assert not [item for item in errors if "count is not" in item], errors


def _restart_state(gpu_count: int) -> dict[str, Any]:
    waiting: list[dict[str, Any]] = []
    executions: list[dict[str, Any]] = []
    for operation in ("STOP_WORKLOADS", "RESTART_WORKLOAD"):
        waiting.append(
            {
                "operation": operation,
                "status": "WAITING",
                "details": {"mutation_submitted_by_control_plane": False},
            }
        )
        details: dict[str, Any] = {}
        if operation == "RESTART_WORKLOAD":
            details = {
                "notification_context": {
                    "source_gpu_count": gpu_count,
                    "target_gpu_count": gpu_count,
                    "restart_count": 1,
                }
            }
        executions.append(
            {
                "operation": operation,
                "status": "SUCCEEDED",
                "adapter_operation_id": f"remote/{operation.lower()}",
                "details": details,
            }
        )
    return {
        "event": {"xid": 11},
        "decision": {"official_action": "RESTART_APP"},
        "workflow": {
            "status": "SUCCEEDED",
            "official_steps": [
                {
                    "operation": "FREEZE_EVIDENCE",
                    "execution_owner": "gpu-fault-control-plane",
                },
                {
                    "operation": "STOP_WORKLOADS",
                    "execution_owner": "gpu-fault-kubernetes-adapter",
                },
                {
                    "operation": "RESTART_WORKLOAD",
                    "execution_owner": "gpu-fault-kubernetes-adapter",
                },
            ],
            "step_executions": executions,
        },
        "observed_waiting_step_executions": waiting,
        "commands": [
            {"status": "SUCCEEDED", "step": {"operation": operation}}
            for operation in ("STOP_WORKLOADS", "RESTART_WORKLOAD")
        ],
        "restart_budget": {"budget": 1, "restart_count": 1},
    }


def test_destr009_workflow_contract_scales_to_expected_gpu_count() -> None:
    state = _restart_state(24)

    assert destr009.workflow_errors(state, expected_gpu_count=24) == [], state
    # COLLECT-016 A reaches the same RESTART_APP contract from a real kmsg
    # XID 31, so the event the contract names is a parameter.
    state["event"]["xid"] = 31
    assert any(
        "not XID 11" in error
        for error in destr009.workflow_errors(state, expected_gpu_count=24)
    ), "an XID 31 event fails the XID 11 contract by name"
    assert destr009.workflow_errors(state, expected_gpu_count=24, xid=31) == [], state
    state["event"]["xid"] = 11
    errors = destr009.workflow_errors(state, expected_gpu_count=1)
    assert any("source GPU count is not 1" in error for error in errors), errors


def test_destr009_accepts_fast_remote_step_terminal_evidence() -> None:
    state = _restart_state(24)
    state["observed_waiting_step_executions"] = [
        item
        for item in state["observed_waiting_step_executions"]
        if item["operation"] != "RESTART_WORKLOAD"
    ]

    assert destr009.workflow_errors(state) == [], state


def test_destr009_rejects_fast_step_without_remote_command_evidence() -> None:
    state = _restart_state(24)
    state["observed_waiting_step_executions"] = []
    state["commands"] = [
        item
        for item in state["commands"]
        if item["step"]["operation"] != "RESTART_WORKLOAD"
    ]

    errors = destr009.workflow_errors(state)

    assert any(
        "RESTART_WORKLOAD lacks remote execution evidence" in item for item in errors
    ), errors


def test_destr009_cleanup_waits_for_late_remote_command_quiescence(
    tmp_path: Path,
) -> None:
    snapshots = [
        {
            "event": {"event_id": "event-a"},
            "workflow": None,
            "commands": [],
            "observations": [{"workload_phase": "RUNNING"}],
        },
        {
            "event": {"event_id": "event-a"},
            "workflow": {"request_id": "workflow-a", "status": "RUNNING"},
            "observations": [{"workload_phase": "RUNNING"}],
            "commands": [
                {
                    "step": {"operation": "RESTART_WORKLOAD"},
                    "status": "PENDING",
                    "lease_token": "sensitive-lease-token",
                }
            ],
        },
        {
            "event": {"event_id": "event-a"},
            "workflow": {"request_id": "workflow-a", "status": "SUCCEEDED"},
            "observations": [{"workload_phase": "FAILED"}],
            "commands": [
                {
                    "step": {"operation": "RESTART_WORKLOAD"},
                    "status": "SUCCEEDED",
                    "updated_at": "2026-09-01T15:01:15Z",
                }
            ],
        },
    ]

    class Regional:
        calls = 0

        def store_snapshot(self, **_kwargs):
            index = min(self.calls, len(snapshots) - 1)
            self.calls += 1
            return snapshots[index]

    regional = Regional()
    report = destr009.wait_for_cleanup_quiescence(
        regional=regional,
        node="node-a",
        marker="marker-a",
        observed_after=datetime.now(timezone.utc),
        job_id="job-a",
        attempt_id="attempt-a",
        case_dir=tmp_path,
        timeout_seconds=1,
        quiet_seconds=0,
        poll_seconds=0,
    )

    assert regional.calls == 3
    assert report["safe_to_delete"] is True
    persisted = json.loads(
        (tmp_path / "cleanup-quiescence.json").read_text(encoding="utf-8")
    )
    assert "sensitive-lease-token" not in json.dumps(persisted, sort_keys=True)
    assert persisted["entries"][1]["commands"][0]["status"] == "PENDING"
    assert persisted["entries"][-1]["commands"][0]["status"] == "SUCCEEDED"
    assert persisted["entries"][-1]["observation_terminal"] is True


def test_destr009_cleanup_rejects_active_observation_after_terminal_workflow(
    tmp_path: Path,
) -> None:
    class Regional:
        def store_snapshot(self, **_kwargs):
            return {
                "event": {"event_id": "event-a"},
                "workflow": {"request_id": "workflow-a", "status": "SUCCEEDED"},
                "observations": [{"workload_phase": "RUNNING"}],
                "commands": [
                    {"step": {"operation": "RESTART_WORKLOAD"}, "status": "SUCCEEDED"}
                ],
            }

    with pytest.raises(destr009.RegionalFixtureError, match="cleanup could not prove"):
        destr009.wait_for_cleanup_quiescence(
            regional=Regional(),
            node="node-a",
            marker="marker-a",
            observed_after=datetime.now(timezone.utc),
            job_id="job-a",
            attempt_id="attempt-a",
            case_dir=tmp_path,
            timeout_seconds=0,
            quiet_seconds=0,
            poll_seconds=0,
        )


def test_destr009_cleanup_waits_before_deleting_workload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []

    class Regional:
        def kubectl(self, *_args, **_kwargs):
            return ""

        def verify_runtime_identity(self, *_args, **_kwargs):
            calls.append("identity")

    class Workload:
        resource = "pytorchjob"
        name = "training-a"

        def delete(self):
            calls.append("delete")

    class Prewarm:
        def cleanup(self):
            return {}

    monkeypatch.setattr(
        destr009,
        "wait_for_cleanup_quiescence",
        lambda **_kwargs: calls.append("quiescence") or {"safe_to_delete": True},
    )
    result: dict[str, Any] = {"verdict": "PASS"}

    destr009.cleanup_case(
        regional=Regional(),
        workload=Workload(),
        prewarm=Prewarm(),
        case_dir=tmp_path,
        runtime_identity={},
        result=result,
        node="node-a",
        marker="marker-a",
        observed_after=datetime.now(timezone.utc),
        job_id="job-a",
        attempt_id="attempt-a",
    )

    assert calls[:2] == ["quiescence", "delete"]
    assert result["workload_residual"] is False


def test_destr009_cleanup_defers_delete_without_quiescence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    deleted = False

    class Regional:
        def verify_runtime_identity(self, *_args, **_kwargs):
            return None

    class Workload:
        resource = "pytorchjob"
        name = "training-a"

        def delete(self):
            nonlocal deleted
            deleted = True

    class Prewarm:
        def cleanup(self):
            return {}

    def fail_quiescence(**_kwargs):
        raise destr009.RegionalFixtureError("remote command is still PENDING")

    monkeypatch.setattr(destr009, "wait_for_cleanup_quiescence", fail_quiescence)
    result: dict[str, Any] = {"verdict": "PASS"}

    destr009.cleanup_case(
        regional=Regional(),
        workload=Workload(),
        prewarm=Prewarm(),
        case_dir=tmp_path,
        runtime_identity={},
        result=result,
        node="node-a",
        marker="marker-a",
        observed_after=datetime.now(timezone.utc),
        job_id="job-a",
        attempt_id="attempt-a",
    )

    assert deleted is False
    assert result["workload_cleanup_deferred"] is True
    assert result["workload_residual"] is True
    assert result["verdict"] == "FAIL"


def test_destr012_group_d_requires_prewrite_failure_details() -> None:
    state = {
        "workflow": {
            "status": "FAILED",
            "step_executions": [
                {
                    "operation": "STOP_WORKLOADS",
                    "status": "FAILED",
                    "error": "enable-job-auto-resume is enabled",
                    "details": {
                        "managed_job_recovery_workloads": [
                            "gpu-fault-system/job/guard-job"
                        ],
                        "required_annotation": (
                            "sagemaker.amazonaws.com/enable-job-auto-resume"
                        ),
                        "required_annotation_value": "absent or false",
                        "remediation_commands": [
                            "kubectl annotate job guard-job "
                            "-n gpu-fault-system "
                            "sagemaker.amazonaws.com/"
                            "enable-job-auto-resume-"
                        ],
                    },
                }
            ],
        }
    }

    assert (
        destr012.group_d_failure_errors(
            state, expected_workload_id="gpu-fault-system/job/guard-job"
        )
        == []
    ), state


def test_destr012_keeps_group_c_optional_and_isolated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        destr012,
        "record_focused_tests",
        lambda details, result: details.__setitem__("focused_tests", result),
    )
    settings = destr012.Settings(
        regional=_regional(tmp_path).settings,
        site_file=tmp_path / "site.yaml",
        a_manifest=REGIONAL / "manifests/training/xid11-three-node-pytorchjob.yaml",
        d_manifest=REGIONAL / "manifests/training/xid11-single-node-job.yaml",
        a_job_id="a-job",
        a_attempt_id="a-job-a001",
        d_job_id="d-job",
        d_attempt_id="d-job-a001",
        predecessor_path=tmp_path / "predecessor.json",
    )
    preflight = {
        "predecessor": {"valid": True},
        "candidate_nodes": [{"name": "node-a", "uid": "uid-a"}],
        "release_id": "release-a",
        "store": {"profile": {"profile_version": "profile-a"}},
        "group_b": {"profile": {"profile_sha256s": ["a" * 64]}},
        "runtime_identity": {"release_state": {}, "deployments": {}},
        "focused_tests": {"passed": True},
    }

    details = destr012.plan_details(settings, preflight)

    assert details["groups"] == ["B", "A", "D", "C"], details
    assert details["group_c"]["optional"] is True, details
    assert details["group_c"]["planned_status"] == "NOT_RUN", details
    assert details["rollback"]["production_Runtime_Profile_is_never_modified"] is True
    # Group workloads are deleted only once DESTR-009's quiescence gate passes,
    # and the plan says so; group A defaults to the DESTR-009 evidence.
    assert (
        details["rollback"]["runner_finally_deletes_both_test_workloads"]
        == "only_after_quiescence"
    )
    assert details["rollback"]["group_workloads_are_deleted_only_after_quiescence"]
    assert details["group_a"]["source"] == destr012.GROUP_A_EVIDENCE_SOURCE
    assert details["focused_tests"] == {"passed": True}
    assert (
        details["preflight_identity"]["runtime_identity"]
        == preflight["runtime_identity"]
    )


def _runtime_identity(*, phase: str = "complete") -> dict[str, Any]:
    deployments = {}
    for plane, names in live_fixture_module.RUNTIME_IDENTITY_DEPLOYMENTS.items():
        deployments[plane] = {
            name: {
                "generation": 1,
                "desired_replicas": 2,
                "observed_generation": 1,
                "updated_replicas": 2,
                "ready_replicas": 2,
                "available_replicas": 2,
                "template_sha256": "a" * 64,
                "images": ["registry.example/runtime@sha256:" + "b" * 64],
            }
            for name in names
        }
    return {
        "release_state": {"release_id": "release-a", "phase": phase},
        "deployments": deployments,
    }


def test_runtime_identity_rejects_active_release_rollback() -> None:
    errors = runtime_identity_errors(
        _runtime_identity(phase="rollback-controller-staged")
    )

    assert any("rollback is active" in error for error in errors), errors


def test_runtime_identity_rejects_incomplete_runtime_rollout() -> None:
    identity = _runtime_identity()
    identity["deployments"]["gpu"]["gpu-fault-cluster-executor"]["ready_replicas"] = 1

    errors = runtime_identity_errors(identity)

    assert any(
        "gpu/gpu-fault-cluster-executor is not fully rolled out" in error
        for error in errors
    ), errors


def test_runtime_identity_verification_records_and_rejects_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _regional(tmp_path)
    expected = _runtime_identity()
    current = _runtime_identity()
    current["release_state"]["updated_at_epoch"] = 2
    monkeypatch.setattr(fixture, "runtime_identity", lambda: current)
    evidence = tmp_path / "runtime-identity.json"

    with pytest.raises(
        live_fixture_module.RegionalFixtureError,
        match="release/runtime deployment identity drifted",
    ):
        fixture.verify_runtime_identity(
            expected, evidence_path=evidence, stage="after group A"
        )

    assert json.loads(evidence.read_text(encoding="utf-8")) == current


def test_destr013_manifest_audit_keeps_provider_replace_disabled() -> None:
    result = destr013.manifest_invariants()

    assert result["violations"] == [], result
    assert result["observed_replace_settings"], result


def test_a_signal_abort_is_not_swallowed_by_the_probe_retry_loop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A ^C during a destructive case has to land on the first retry.

    ``pod_python`` retries three times on a bare ``except Exception``, and the
    wait loops retry for their whole timeout. When the signal handler raised a
    ``RegionalFixtureError`` -- a ``RuntimeError`` -- those loops caught the
    abort and kept going, so the operator's interrupt did nothing until the
    timeout expired. That is why this abort derives from ``BaseException``.
    """

    regional = _regional(tmp_path)
    attempts: list[int] = []

    def interrupted(*_args: Any, **_kwargs: Any) -> str:
        attempts.append(1)
        live_fixture_module.abort_on_signal(2, None)
        raise AssertionError("unreachable")

    monkeypatch.setattr(regional, "ready_pod", lambda *_a, **_k: "pod-a")
    monkeypatch.setattr(regional, "kubectl", interrupted)

    with pytest.raises(live_fixture_module.RegionalFixtureAbort) as raised:
        regional.pod_python("cpu", "gpu-fault-api-ha", "print()")

    assert attempts == [1], (
        f"the abort was retried {len(attempts)} times instead of propagating"
    )
    assert raised.value.signum == 2
    assert not isinstance(raised.value, Exception), (
        "an abort that is an Exception is catchable by the loops it must escape"
    )


def test_run_case_main_reports_an_abort_distinctly_from_a_fail_verdict() -> None:
    """Exit 1 already means ``verdict: FAIL``, so an abort needs its own code."""

    def aborted() -> int:
        live_fixture_module.abort_on_signal(15, None)
        raise AssertionError("unreachable")

    assert live_fixture_module.run_case_main(aborted) == 143
    assert live_fixture_module.run_case_main(lambda: 1) == 1
    assert live_fixture_module.run_case_main(lambda: 0) == 0


def test_every_regional_runner_installs_the_shared_abort_handler() -> None:
    """The handler was copy-pasted into 19 runners; one copy is enough.

    A runner that keeps a local ``abort_on_signal`` re-introduces the
    swallowed-abort defect silently, so the absence is asserted rather than
    left to review.
    """

    offenders = []
    for path in sorted(REGIONAL.glob("run_*.py")):
        source = path.read_text(encoding="utf-8")
        if "install_abort_signals" not in source:
            continue
        if "def abort_on_signal" in source:
            offenders.append(f"{path.name}: keeps a local abort_on_signal")
        if "raise SystemExit(main())" in source:
            offenders.append(f"{path.name}: entry point bypasses run_case_main")

    assert offenders == [], offenders


def test_collect004_finds_restart_node_anywhere_in_the_nodes_workflow_chain() -> None:
    """The reboot lives in the mismatch workflow, not in the escalation after it."""

    mismatch = {
        "request_id": "workflow-mismatch",
        "status": "FAILED",
        "official_steps": [
            {"operation": "FREEZE_EVIDENCE"},
            {"operation": "RESTART_NODE"},
            {"operation": "VALIDATE_GPU"},
        ],
    }
    replace_after = {
        "request_id": "workflow-replace-after",
        "status": "FAILED",
        "official_steps": [
            {"operation": "FREEZE_EVIDENCE"},
            {"operation": "REPLACE_NODE"},
        ],
    }
    state = {
        "workflow": replace_after,
        "matches": [{"workflow": replace_after}, {"workflow": mismatch}],
    }

    planned = collector_destructive.workflow_planning(state, "RESTART_NODE")
    assert planned is not None and planned["request_id"] == "workflow-mismatch", planned
    assert collector_destructive.workflow_planning(state, "REPLACE_NODE")[
        "request_id"
    ] == ("workflow-replace-after")
    assert (
        collector_destructive.workflow_planning(
            {"workflow": replace_after}, "RESTART_NODE"
        )
        is None
    ), "a state without the chain still answers from the one workflow it has"


def test_collect004_restores_the_collector_env_through_a_fresh_probe_after_a_reboot() -> (
    None
):
    calls: list[str] = []

    class Collector:
        def __init__(self) -> None:
            self.alive = False

        def execute(self, *arguments: str, timeout: int = 180) -> dict:
            calls.append("execute" if self.alive else "execute-dead")
            if not self.alive:
                raise (
                    live_fixture_module.HostProbeError(
                        "cannot exec into a container in a completed pod"
                    )
                    if hasattr(live_fixture_module, "HostProbeError")
                    else (
                        collector_destructive.HostProbeError(
                            "cannot exec into a container in a completed pod"
                        )
                    )
                )
            return {"restored": True, "arguments": arguments}

        def recreate(self) -> None:
            calls.append("recreate")
            self.alive = True

    result = collector_destructive.restore_collector_env(Collector(), "c004-1")

    assert result["restored"] is True, result
    assert calls == ["execute-dead", "recreate", "execute"], (
        "the dead Pod is replaced exactly once and the restore then goes through it"
    )


def test_reset_contract_is_parametrised_for_the_xid48_companion_drill() -> None:
    state = _reset_state()
    state["event"]["xid"] = 48
    state["decision"]["official_action"] = "DRAIN_AND_RESET"

    assert (
        destr001.workflow_errors(state, xid=48, official_action="DRAIN_AND_RESET") == []
    )
    errors = destr001.workflow_errors(state)
    assert "matched event is not XID 46" in errors, errors
    assert "policy did not finalize XID 46 as RESET_GPU" in errors, errors


def _reset_host_pair(
    *, minimum: int, last: int, journal_resets: int = 0
) -> tuple[dict, dict]:
    ledger_before = [
        {"command_id": "cmd-old", "operation": "QUIESCE_GPU_SERVICES", "attempt": 1}
    ]
    baseline = {
        "gpu_inventory": [{"pci_bdf": f"0000:{index:02x}:00.0"} for index in range(8)],
        "ledger": list(ledger_before),
        "services": {"kubelet.service": {"ActiveState": "active"}},
        "gpu_fault_timers": ["gpu-fault-certificate-check.timer"],
    }
    after = {
        **baseline,
        "compute_clients": [],
        "quiesce_states": [],
        "ledger": ledger_before
        + [{"command_id": "cmd-reset", "operation": "RESET_GPU", "attempt": 1}],
        "kernel_reset_journal": {"target_reset_count": journal_resets},
        "sampler": {
            "sample_count": 40,
            "min_gpu_count": minimum,
            "last": {"gpu_count": last},
        },
    }
    return baseline, after


def test_physical_reset_is_proven_by_the_sampler_dip_not_by_journal_text() -> None:
    baseline, after = _reset_host_pair(minimum=7, last=8)
    assert (
        destr001.host_errors(
            baseline, after, expected_gpu_count=8, target_bdf="0000:59:00"
        )
        == []
    )

    baseline, flat = _reset_host_pair(minimum=8, last=8)
    errors = destr001.host_errors(
        baseline, flat, expected_gpu_count=8, target_bdf="0000:59:00"
    )
    assert any("leave and return" in error for error in errors), errors

    baseline, twice = _reset_host_pair(minimum=7, last=8, journal_resets=2)
    errors = destr001.host_errors(
        baseline, twice, expected_gpu_count=8, target_bdf="0000:59:00"
    )
    assert any("more than one reset" in error for error in errors), errors

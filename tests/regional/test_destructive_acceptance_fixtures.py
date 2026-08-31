from __future__ import annotations

import importlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import audit_destr013_replacement_invariant as destr013
from scripts.e2e.regional import run_destr001_gpu_reset as destr001
from scripts.e2e.regional import run_destr002_hyperpod_reboot as destr002
from scripts.e2e.regional import run_destr003_warm_spare_failover as destr003
from scripts.e2e.regional import run_destr008_warm_spare_shortage as destr008
from scripts.e2e.regional import run_destr009_workload_restart as destr009
from scripts.e2e.regional import run_destr012_managed_recovery_guard as destr012
from scripts.e2e.regional import run_ha003_aurora_failover_reset as ha003
from scripts.e2e.regional import run_ha004_waiting_reclaim_reset as ha004
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
    step_executions = []
    for operation in (
        "QUIESCE_GPU_SERVICES",
        "VERIFY_NO_GPU_CLIENTS",
        "RESET_GPU",
        "RESTORE_GPU_SERVICES",
    ):
        step_executions.append(
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
            "step_executions": step_executions,
        },
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
    state["workflow"]["step_executions"] = state["workflow"]["step_executions"][:-1]
    errors = destr001.workflow_errors(state)
    assert any("WAITING evidence" in error for error in errors), errors


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
                    "details": {
                        "source_gpu_count": 8,
                        "target_gpu_count": 8,
                        "restart_count": 1,
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


def _destr008_settings(tmp_path: Path) -> destr008.Settings:
    return destr008.Settings(
        regional=_regional(tmp_path).settings,
        site_file=tmp_path / "site.yaml",
        manifest=REGIONAL / "manifests/training/single-node-warm-spare-pytorchjob.yaml",
        hyperpod_cluster="hp-cluster",
        fault_node="node-a",
        spare_node="node-b",
        host_probe_image="registry.example/probe@sha256:" + "a" * 64,
        scenarios=destr008.SCENARIOS,
        predecessor_path=tmp_path / "predecessor.json",
    )


def _shortage_state(scenario: str, *, alert: bool) -> dict[str, Any]:
    incident_id = "incident-a"
    return {
        "incident": {"incident_id": incident_id},
        "workflow": {
            "status": "FAILED",
            "official_steps": [
                {
                    "operation": "REPLACE_NODE",
                    "parameters": {"replacement_strategy": "HEALTHY_WARM_SPARE_ONLY"},
                }
            ],
            "step_executions": [
                {"operation": "STOP_WORKLOADS", "status": "SUCCEEDED"},
                {
                    "operation": "REPLACE_NODE",
                    "status": "FAILED",
                    "error": destr008.EXPECTED_REASON[scenario],
                },
            ],
        },
        "notifications": (
            [{"deduplication_key": (f"{incident_id}/hyperpod-spare-insufficient")}]
            if alert
            else []
        ),
        "markers": [],
        "fault_node": {
            "unschedulable": True,
            "taints": [{"key": "gpu-fault.io/quarantined"}],
        },
        "spare_node": {
            "annotations": {
                "gpu-fault.io/spare-reservation": None,
                "gpu-fault.io/spare-pool-state": "AVAILABLE",
            }
        },
    }


def test_destr008_notification_semantics_distinguish_no_pool_from_shortage(
    tmp_path: Path,
) -> None:
    settings = _destr008_settings(tmp_path)

    no_pool = destr008.scenario_errors(
        _shortage_state("no-spare", alert=False), settings, "no-spare"
    )
    shortage = destr008.scenario_errors(
        _shortage_state("topology-mismatch", alert=True), settings, "topology-mismatch"
    )

    assert no_pool == [], no_pool
    assert shortage == [], shortage


def _restart_state(gpu_count: int) -> dict[str, Any]:
    executions: list[dict[str, Any]] = []
    for operation in ("STOP_WORKLOADS", "RESTART_WORKLOAD"):
        executions.append(
            {
                "operation": operation,
                "status": "WAITING",
                "details": {"mutation_submitted_by_control_plane": False},
            }
        )
        details: dict[str, Any] = {}
        if operation == "RESTART_WORKLOAD":
            details = {
                "source_gpu_count": gpu_count,
                "target_gpu_count": gpu_count,
                "restart_count": 1,
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
    }


def test_destr009_workflow_contract_scales_to_expected_gpu_count() -> None:
    state = _restart_state(24)

    assert destr009.workflow_errors(state, expected_gpu_count=24) == [], state
    errors = destr009.workflow_errors(state, expected_gpu_count=1)
    assert any("source GPU count is not 1" in error for error in errors), errors


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


def test_destr012_keeps_group_c_optional_and_isolated(tmp_path: Path) -> None:
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
    }

    details = destr012.plan_details(settings, preflight)

    assert details["groups"] == ["B", "A", "D", "C"], details
    assert details["group_c"]["optional"] is True, details
    assert details["group_c"]["planned_status"] == "NOT_RUN", details
    assert details["rollback"]["production_Runtime_Profile_is_never_modified"] is True


def _ha003_settings(tmp_path: Path) -> ha003.Settings:
    return ha003.Settings(
        regional=_regional(tmp_path).settings,
        node="node-a",
        host_probe_image="registry.example/probe@sha256:" + "a" * 64,
        rds_cluster_id="aurora-a",
        predecessor_path=tmp_path / "predecessor.json",
    )


def test_ha003_waits_for_reset_claim_before_failover(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    regional = _regional(tmp_path)
    snapshots = iter(
        [
            {"commands": [{"step": {"operation": "RESET_GPU"}, "status": "PENDING"}]},
            {
                "commands": [
                    {
                        "command_id": "remote-a",
                        "step": {"operation": "RESET_GPU"},
                        "status": "LEASED",
                    }
                ]
            },
        ]
    )
    monkeypatch.setattr(regional, "store_snapshot", lambda **_kwargs: next(snapshots))
    state, command = ha003.wait_reset_claim(
        regional,
        _ha003_settings(tmp_path),
        marker="marker-a",
        observed_after=datetime.now(timezone.utc),
        timeout_seconds=5,
    )

    assert command["command_id"] == "remote-a", command
    assert state["commands"][0]["status"] == "LEASED", state


def _ha004_settings(tmp_path: Path) -> ha004.Settings:
    return ha004.Settings(
        regional=_regional(tmp_path).settings,
        node="node-a",
        host_probe_image="registry.example/probe@sha256:" + "a" * 64,
        predecessor_path=tmp_path / "predecessor.json",
    )


def test_ha004_reclaim_timeline_keeps_one_command_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    regional = _regional(tmp_path)
    snapshots = iter(
        [
            {
                "commands": [
                    {
                        "command_id": "remote-a",
                        "status": "LEASED",
                        "lease_owner": "cluster/pod-a",
                        "lease_token": "token-a",
                        "step": {"operation": "RESET_GPU"},
                    }
                ]
            },
            {
                "commands": [
                    {
                        "command_id": "remote-a",
                        "status": "LEASED",
                        "lease_owner": "cluster/pod-b",
                        "lease_token": "token-b",
                        "step": {"operation": "RESET_GPU"},
                    }
                ]
            },
            {
                "commands": [
                    {
                        "command_id": "remote-a",
                        "status": "SUCCEEDED",
                        "last_lease_owner": "cluster/pod-b",
                        "step": {"operation": "RESET_GPU"},
                    }
                ]
            },
        ]
    )
    killed: list[str] = []
    monkeypatch.setattr(regional, "store_snapshot", lambda **_kwargs: next(snapshots))
    state, timeline = ha004.command_timeline(
        regional,
        _ha004_settings(tmp_path),
        marker="marker-a",
        observed_after=datetime.now(timezone.utc),
        timeout_seconds=5,
        kill_owner=killed.append,
    )

    assert killed == ["cluster/pod-a"], killed
    assert {item["command_id"] for item in timeline} == {"remote-a"}, timeline
    assert state["commands"][0]["status"] == "SUCCEEDED", state


def test_destr013_manifest_audit_keeps_provider_replace_disabled() -> None:
    result = destr013.manifest_invariants()

    assert result["violations"] == [], result
    assert result["observed_replace_settings"], result

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts.e2e.regional import audit_destr011_provider_replace as destr011
from scripts.e2e.regional import audit_destr013_replacement_invariant as destr013
from scripts.e2e.regional import audit_warm_spare_guardrails as warm_spare
from scripts.e2e.regional import run_destr001_gpu_reset as destr001
from scripts.e2e.regional import run_destr002_hyperpod_reboot as destr002
from scripts.e2e.regional import run_destr003_warm_spare_failover as destr003
from scripts.e2e.regional import run_destr008_warm_spare_shortage as destr008
from scripts.e2e.regional import run_destr009_workload_restart as destr009
from scripts.e2e.regional import run_destr010_fabric_manager_restart as destr010
from scripts.e2e.regional import run_destr012_managed_recovery_guard as destr012
from scripts.e2e.regional import run_destr015_parallel_branch_join as destr015
from scripts.e2e.regional import run_destr016_preempting_reboot as destr016
from scripts.e2e.regional import run_destr017_out_of_band_reboot_fence as destr017
from scripts.e2e.regional import run_destr018_lifetime_deadline as destr018
from scripts.e2e.regional import run_ha003_aurora_failover_reset as ha003
from scripts.e2e.regional import run_ha004_waiting_reclaim_reset as ha004
from scripts.e2e.regional import run_ha009_aurora_credential_rotation as ha009
from scripts.e2e.regional.host_probe_fixture import HostProbeFixture, HostProbeSettings

ROOT = Path(__file__).resolve().parents[2]
REGIONAL = ROOT / "scripts/e2e/regional"


def test_promoted_ha_and_destructive_helpers_exist() -> None:
    expected = {
        "run_ha007_control_worker_shutdown.py",
        "run_ha008_processor_exit_acceptance.py",
        "run_ha009_aurora_credential_rotation.py",
        "run_ha003_aurora_failover_reset.py",
        "run_ha004_waiting_reclaim_reset.py",
        "audit_destr013_replacement_invariant.py",
        "audit_destr011_provider_replace.py",
        "audit_warm_spare_guardrails.py",
        "run_destr001_gpu_reset.py",
        "run_destr002_hyperpod_reboot.py",
        "run_destr003_warm_spare_failover.py",
        "run_destr008_warm_spare_shortage.py",
        "run_destr009_workload_restart.py",
        "run_destr010_fabric_manager_restart.py",
        "run_destr012_managed_recovery_guard.py",
        "run_destr015_parallel_branch_join.py",
        "destr015_verdicts.py",
        "run_destr016_preempting_reboot.py",
        "destr016_verdicts.py",
        "run_destr017_out_of_band_reboot_fence.py",
        "destr017_verdicts.py",
        "run_destr018_lifetime_deadline.py",
        "destr018_verdicts.py",
        "control_plane_env_window.py",
        "host_probe_fixture.py",
        "regional_live_fixture.py",
        "managed_workload_fixture.py",
        "warm_spare_fixture.py",
        "collector_acceptance_fixture.py",
        "run_collector_acceptance.py",
        "run_collector_destructive.py",
        "run_collect016_training_recovery.py",
        "run_collect017_efa_plugin.py",
        "multi_cluster_fixture.py",
        "run_iso006_cluster_offline.py",
        "run_e2e002_multicluster_fault.py",
    }

    assert expected <= {path.name for path in REGIONAL.glob("*.py")}


def test_promoted_helpers_contain_no_site_specific_topology() -> None:
    forbidden = (
        "/secure/" + "gpu-fault-bootstrap",
        "hypd" + "-1127",
        "514385" + "905925",
        "gpu-fault-" + "us-west-2-control-plane",
    )
    paths = (
        REGIONAL / "run_ha009_aurora_credential_rotation.py",
        REGIONAL / "run_ha003_aurora_failover_reset.py",
        REGIONAL / "run_ha004_waiting_reclaim_reset.py",
        REGIONAL / "audit_destr013_replacement_invariant.py",
        REGIONAL / "audit_destr011_provider_replace.py",
        REGIONAL / "audit_warm_spare_guardrails.py",
        REGIONAL / "run_destr001_gpu_reset.py",
        REGIONAL / "run_destr002_hyperpod_reboot.py",
        REGIONAL / "run_destr003_warm_spare_failover.py",
        REGIONAL / "run_destr008_warm_spare_shortage.py",
        REGIONAL / "run_destr009_workload_restart.py",
        REGIONAL / "run_destr010_fabric_manager_restart.py",
        REGIONAL / "run_destr012_managed_recovery_guard.py",
        REGIONAL / "run_destr015_parallel_branch_join.py",
        REGIONAL / "destr015_verdicts.py",
        REGIONAL / "run_destr016_preempting_reboot.py",
        REGIONAL / "destr016_verdicts.py",
        REGIONAL / "run_destr017_out_of_band_reboot_fence.py",
        REGIONAL / "destr017_verdicts.py",
        REGIONAL / "run_destr018_lifetime_deadline.py",
        REGIONAL / "destr018_verdicts.py",
        REGIONAL / "control_plane_env_window.py",
        REGIONAL / "host_probe_fixture.py",
        REGIONAL / "regional_live_fixture.py",
        REGIONAL / "managed_workload_fixture.py",
        REGIONAL / "warm_spare_fixture.py",
        REGIONAL / "collector_acceptance_fixture.py",
        REGIONAL / "run_collector_acceptance.py",
        REGIONAL / "run_collector_destructive.py",
        REGIONAL / "run_collect016_training_recovery.py",
        REGIONAL / "run_collect017_efa_plugin.py",
        REGIONAL / "multi_cluster_fixture.py",
        REGIONAL / "run_iso006_cluster_offline.py",
        REGIONAL / "run_e2e002_multicluster_fault.py",
    )

    for path in paths:
        source = path.read_text(encoding="utf-8")
        assert all(value not in source for value in forbidden), path


def test_ha009_default_invocation_is_plan_only(tmp_path: Path) -> None:
    env = {
        **os.environ,
        "PYTHONPATH": str(ROOT / "src"),
        "GPU_FAULT_CONTROL_KUBECONFIG": "/secure/cpu.kubeconfig",
        "KUBECONFIG": "/secure/gpu.kubeconfig",
        "GPU_FAULT_DATAPLANE_CONTEXT": "gpu-context",
        "GPU_FAULT_PERF_AWS_REGION": "us-west-2",
        "GPU_FAULT_PERF_CONTROL_NAMESPACE": "gpu-fault-system",
        "GPU_FAULT_PERF_DATAPLANE_NAMESPACE": "gpu-fault-system",
    }
    completed = subprocess.run(
        [
            sys.executable,
            str(REGIONAL / "run_ha009_aurora_credential_rotation.py"),
            "--run-dir",
            str(tmp_path),
            "--rds-cluster-id",
            "aurora-test",
        ],
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )

    plan_path = tmp_path / "cases/GF-REGIONAL-HA-009/plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    assert plan["mutation_performed"] is False
    assert plan["confirmation"] == "HA009_ROTATE_AURORA_CREDENTIALS"
    assert plan["details"]["rds_cluster_id"] == "aurora-test"
    assert plan_path.stat().st_mode & 0o777 == 0o600
    assert '"mutation_performed": false' in completed.stdout


def test_ha009_accepts_rds_current_pending_alias() -> None:
    complete = {
        "stages": {
            "AWSCURRENT": "version-new",
            "AWSPENDING": "version-new",
            "AWSPREVIOUS": "version-old",
        }
    }
    still_pending = {
        "stages": {"AWSCURRENT": "version-old", "AWSPENDING": "version-new"}
    }

    assert ha009.managed_rotation_complete(
        complete, old_current="version-old", cluster_status="available"
    ), "completed managed rotation was not recognized"
    assert not ha009.managed_rotation_complete(
        still_pending, old_current="version-old", cluster_status="available"
    ), "stale AWSCURRENT was accepted as a completed rotation"


def test_destructive_audits_are_parameterized(tmp_path: Path) -> None:
    kubeconfig = tmp_path / "gpu.kubeconfig"
    kubeconfig.write_text("apiVersion: v1\n", encoding="utf-8")
    cpu_kubeconfig = tmp_path / "cpu.kubeconfig"
    cpu_kubeconfig.write_text("apiVersion: v1\n", encoding="utf-8")
    common = {
        "gpu_kubeconfig": str(kubeconfig),
        "gpu_context": "gpu-context",
        "namespace": "gpu-fault-system",
        "region": "us-east-2",
    }
    destr011.configure(
        SimpleNamespace(
            **common,
            executor_role_arn="arn:aws:iam::123456789012:role/executor",
            hyperpod_cluster_name="gpu-hyperpod",
        )
    )
    warm_spare.configure(
        SimpleNamespace(
            **common,
            cpu_kubeconfig=str(cpu_kubeconfig),
            cpu_context="cpu-context",
            managed_gpu_cluster_name="gpu-hyperpod",
            automatic_negative_cluster_name="automatic-test",
        ),
        set(warm_spare.CASE_IDS),
    )

    assert destr011.GPU_KUBECONFIG == kubeconfig.resolve()
    assert destr011.HYPERPOD_CLUSTER == "gpu-hyperpod"
    assert destr011.AWS_REGION == "us-east-2"
    assert warm_spare.CPU_KUBECONFIG == cpu_kubeconfig.resolve()
    assert warm_spare.CPU_CONTEXT == "cpu-context"
    assert warm_spare.GPU_KUBECONFIG == kubeconfig.resolve()
    assert warm_spare.MANAGED_GPU_CLUSTER == "gpu-hyperpod"
    assert warm_spare.AUTOMATIC_NEGATIVE_CLUSTER == "automatic-test"


def test_host_probe_fixture_is_node_pinned_and_time_bounded(tmp_path: Path) -> None:
    kubeconfig = tmp_path / "gpu.kubeconfig"
    kubeconfig.write_text("apiVersion: v1\n", encoding="utf-8")
    probe = tmp_path / "probe.py"
    probe.write_text("print('{}')\n", encoding="utf-8")
    fixture = HostProbeFixture(
        HostProbeSettings(
            kubeconfig=kubeconfig,
            context="gpu-context",
            namespace="gpu-fault-system",
            node="node-a",
            image="registry.example/probe@sha256:" + "a" * 64,
            case_id="GF-REGIONAL-DESTR-010",
            run_id="run-a",
            probe_script=probe,
        )
    )

    configmap, pod = fixture.manifests()

    assert configmap["metadata"]["namespace"] == "gpu-fault-system"
    assert pod["spec"]["nodeName"] == "node-a"
    assert pod["spec"]["activeDeadlineSeconds"] == 1800
    assert pod["spec"]["containers"][0]["securityContext"] == {"privileged": True}
    assert pod["spec"]["volumes"][0]["hostPath"]["path"] == "/"


def test_destr010_exact_workflow_contract() -> None:
    state: dict[str, Any] = {
        "event": {"xid": 45, "evidence_ref": "kmsg://node/boot/1"},
        "decision": {"official_action": "RESTART_FM", "disposition": "EXECUTABLE"},
        "workflow": {
            "status": "SUCCEEDED",
            "official_steps": [
                {
                    "operation": "FREEZE_EVIDENCE",
                    "execution_owner": "gpu-fault-control-plane",
                },
                {
                    "operation": "RESTART_FABRIC_MANAGER",
                    "execution_owner": "gpu-fault-node-agent",
                },
            ],
            "completed_operations": ["FREEZE_EVIDENCE", "RESTART_FABRIC_MANAGER"],
        },
        "commands": [{"status": "SUCCEEDED"}],
        "evidence": [{"record_id": "evidence-a"}],
        "notifications": [
            {
                "notification_id": "notification-a",
                "category": "ACTION_COMPLETED",
                "subject": "[DRILL:test] [GPU Fabric Manager 已自动重启]",
                "status": "SKIPPED",
            }
        ],
    }

    assert destr010.workflow_errors(state) == []
    state["workflow"]["completed_operations"].append("MARK_UNSCHEDULABLE")
    assert "forbidden isolation/quiesce" in " ".join(destr010.workflow_errors(state))


def test_destr010_parser_is_plan_only_by_default(tmp_path: Path) -> None:
    arguments = destr010.parser().parse_args(["--run-dir", str(tmp_path)])

    assert arguments.execute is False
    assert arguments.plan is False
    assert destr010.CONFIRMATION == "DESTR010_RESTART_FABRIC_MANAGER"


def test_destructive_sequence_parsers_are_plan_only(tmp_path: Path) -> None:
    modules = (
        destr001,
        destr002,
        destr003,
        destr008,
        destr009,
        destr012,
        destr015,
        destr016,
        destr017,
        destr018,
        ha003,
        ha004,
    )

    for module in modules:
        arguments = module.parser().parse_args(["--run-dir", str(tmp_path)])
        assert arguments.execute is False, module.CASE_ID
        assert arguments.predecessor_evidence == "", module.CASE_ID


def test_destr013_parser_requires_an_explicit_window(tmp_path: Path) -> None:
    arguments = destr013.parser().parse_args(
        [
            "--run-dir",
            str(tmp_path),
            "--window-start",
            "2026-08-30T00:00:00Z",
            "--window-end",
            "2026-08-31T00:00:00Z",
        ]
    )

    assert arguments.window_start.endswith("Z"), arguments.window_start
    assert arguments.window_end.endswith("Z"), arguments.window_end


def _gpu_node(
    name: str,
    *,
    unschedulable: bool = False,
    taints: list[dict[str, Any]] | None = None,
    incident_id: str | None = None,
) -> dict[str, Any]:
    return {
        "name": name,
        "uid": f"uid-{name}",
        "ready": "True",
        "gpu_allocatable": 8,
        "unschedulable": unschedulable,
        "taints": taints or [],
        "ownership_annotations": {
            "gpu-fault.io/incident-id": incident_id,
            "gpu-fault.io/fencing-token": "1" if incident_id else None,
            "gpu-fault.io/previous-unschedulable": ("false" if incident_id else None),
        },
    }


def test_warm_spare_audit_blocks_preexisting_quarantine() -> None:
    nodes = [
        _gpu_node("node-a"),
        _gpu_node(
            "node-b",
            unschedulable=True,
            taints=[
                {
                    "key": "gpu-fault.io/quarantined",
                    "value": "incident-deadbeef",
                    "effect": "NoSchedule",
                }
            ],
            incident_id="incident-a",
        ),
    ]

    errors = warm_spare.node_preflight_errors(nodes)

    assert errors == ["node-b has pre-existing gpu-fault quarantine ownership"]


def test_warm_spare_audit_requires_exact_node_postflight() -> None:
    baseline = [_gpu_node("node-a"), _gpu_node("node-b")]
    postflight = [
        _gpu_node("node-a"),
        _gpu_node(
            "node-b",
            unschedulable=True,
            taints=[
                {
                    "key": "gpu-fault.io/quarantined",
                    "value": "incident-deadbeef",
                    "effect": "NoSchedule",
                }
            ],
            incident_id="incident-a",
        ),
    ]

    errors = warm_spare.node_state_drift(baseline, postflight)

    assert any("node-b changed unschedulable" in error for error in errors), (
        "postflight did not detect the cordon drift"
    )
    assert any("node-b changed taints" in error for error in errors), (
        "postflight did not detect the taint drift"
    )
    assert any("node-b changed ownership_annotations" in error for error in errors), (
        "postflight did not detect the ownership drift"
    )


_AUTOMATIC_GUARD_ERROR = warm_spare.AUTOMATIC_GUARD_ERROR
_AUTOMATIC_GATE_ERROR = (
    "HyperPod preflight failed: HyperPod automatic node recovery is "
    "enabled; direct batch mutation would create a second lifecycle trigger"
)


def _automatic_probe(**overrides: Any) -> dict[str, Any]:
    probe = {
        "cluster_name": "automatic-test",
        "observed_node_recovery": "Automatic",
        "probed_node_status": "Running",
        "warm_spare_guard": {"status": "FAILED", "error": _AUTOMATIC_GUARD_ERROR},
        "control_without_warm_spare_strategy": {
            "status": "FAILED",
            "error": _AUTOMATIC_GATE_ERROR,
        },
    }
    probe.update(overrides)
    return probe


def test_destr006_accepts_only_a_guard_that_fired_on_an_observed_automatic() -> None:
    probe = _automatic_probe()

    assert warm_spare.probe_errors("GF-REGIONAL-DESTR-006", None, None, probe) == [], (
        probe
    )


def test_destr006_rejects_a_guard_that_never_saw_an_automatic_cluster() -> None:
    # The guard refusing is only evidence if what it refused was a real
    # Automatic cluster. A stubbed preflight produces exactly this error while
    # proving nothing about the deployed adapter's view of the provider.
    probe = _automatic_probe(observed_node_recovery="None")

    errors = warm_spare.probe_errors("GF-REGIONAL-DESTR-006", None, None, probe)

    assert any("did not observe NodeRecovery=Automatic" in item for item in errors), (
        errors
    )


def test_destr006_rejects_a_guard_that_failed_for_another_reason() -> None:
    # An AccessDenied on DescribeCluster also ends in FAILED, and reporting that
    # as the NodeRecovery guard would record a missing IAM permission as proof
    # the guard works.
    probe = _automatic_probe(
        warm_spare_guard={
            "status": "FAILED",
            "error": "HyperPod preflight failed: AccessDeniedException",
        }
    )

    errors = warm_spare.probe_errors("GF-REGIONAL-DESTR-006", None, None, probe)

    assert errors == [
        "deployed automatic-recovery guard returned an unexpected result"
    ], errors


def test_destr006_rejects_a_control_arm_that_refused_for_no_stated_reason() -> None:
    # If the step fails the same way with and without the warm-spare strategy,
    # or fails without the preflight naming the automatic recovery it read,
    # the recorded refusal cannot be attributed to the observed NodeRecovery.
    for control in (
        {"status": "FAILED", "error": _AUTOMATIC_GUARD_ERROR},
        {"status": "FAILED", "error": "HyperPod preflight failed: unrelated gate"},
        {"status": "SUCCEEDED", "error": None},
    ):
        errors = warm_spare.probe_errors(
            "GF-REGIONAL-DESTR-006",
            None,
            None,
            _automatic_probe(control_without_warm_spare_strategy=control),
        )

        assert errors == [
            "deployed control probe did not attribute the refusal to "
            "the observed NodeRecovery"
        ], control


def test_destr006_requires_the_deployed_probe_to_have_run() -> None:
    errors = warm_spare.probe_errors("GF-REGIONAL-DESTR-006", None, None, None)

    assert errors == ["deployed automatic-recovery guard probe did not run"], errors


def test_warm_spare_audit_allows_unchanged_preexisting_business_taint() -> None:
    taints = [{"key": "workload.example/dedicated", "effect": "NoSchedule"}]
    baseline = [_gpu_node("node-a", unschedulable=True, taints=taints)]

    assert warm_spare.node_preflight_errors(baseline) == []
    assert warm_spare.node_state_drift(baseline, baseline) == []


def test_warm_spare_audit_records_postflight_after_probe_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    baseline = [_gpu_node("node-a")]
    snapshots = iter([baseline, baseline])
    monkeypatch.setattr(warm_spare, "node_snapshot", lambda: next(snapshots))
    monkeypatch.setattr(warm_spare, "run_pytest", lambda _dir, _nodeids: True)
    monkeypatch.setattr(
        warm_spare,
        "cluster_recovery",
        lambda _name: (_ for _ in ()).throw(RuntimeError("probe unavailable")),
    )
    monkeypatch.setattr(warm_spare, "replace_events", lambda _start, _end: [])

    exit_code = warm_spare.run_audit(tmp_path, ["GF-REGIONAL-DESTR-005"])

    result = json.loads(
        (tmp_path / "cases/GF-REGIONAL-DESTR-005/GF-REGIONAL-DESTR-005.json").read_text(
            encoding="utf-8"
        )
    )
    postflight = json.loads(
        (tmp_path / "gpu-node-postflight.json").read_text(encoding="utf-8")
    )
    assert exit_code == 1
    assert result["verdict"] == "FAIL"
    assert result["node_state_identical"] is True
    assert "RuntimeError: probe unavailable" in result["errors"]
    # A live probe blowing up says nothing about the focused pytest, which ran
    # and passed. Reporting it as a pytest failure sent the last DESTR-006 run
    # looking for a repo-side regression that did not exist.
    assert "focused pytest failed" not in result["errors"], result["errors"]
    assert postflight == baseline


def test_warm_spare_audit_separates_an_unrun_pytest_from_a_failed_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    baseline = [_gpu_node("node-a")]
    snapshots = iter([baseline, baseline])
    monkeypatch.setattr(warm_spare, "node_snapshot", lambda: next(snapshots))
    monkeypatch.setattr(
        warm_spare,
        "run_pytest",
        lambda _dir, _nodeids: (_ for _ in ()).throw(
            RuntimeError("pytest unavailable")
        ),
    )
    monkeypatch.setattr(warm_spare, "replace_events", lambda _start, _end: [])

    exit_code = warm_spare.run_audit(tmp_path, ["GF-REGIONAL-DESTR-005"])

    result = json.loads(
        (tmp_path / "cases/GF-REGIONAL-DESTR-005/GF-REGIONAL-DESTR-005.json").read_text(
            encoding="utf-8"
        )
    )
    assert exit_code == 1
    assert "focused pytest did not run" in result["errors"], result["errors"]
    assert "focused pytest failed" not in result["errors"], result["errors"]


def test_host_probe_create_deletes_a_leftover_pod_before_applying(
    tmp_path: Path,
) -> None:
    """The pod name is a digest of (case, run, node), so a rerun after an
    operator abort meets the previous pod, by then Failed on its
    activeDeadlineSeconds; `apply` onto it is a no-op and the Ready wait can
    only time out. create() has to delete whatever carries the name first."""

    kubeconfig = tmp_path / "gpu.kubeconfig"
    kubeconfig.write_text("apiVersion: v1\n", encoding="utf-8")
    probe = tmp_path / "probe.py"
    probe.write_text("print('{}')\n", encoding="utf-8")
    fixture = HostProbeFixture(
        HostProbeSettings(
            kubeconfig=kubeconfig,
            context="gpu-context",
            namespace="gpu-fault-system",
            node="node-a",
            image="registry.example/probe@sha256:" + "a" * 64,
            case_id="GF-REGIONAL-COLLECT-014",
            run_id="run-a",
            probe_script=probe,
        )
    )
    calls: list[tuple[str, ...]] = []

    def fake_kubectl(*arguments: str, **_kwargs: Any) -> Any:
        calls.append(arguments)
        return None

    fixture._kubectl = fake_kubectl  # type: ignore[method-assign]
    fixture.create()

    verbs = [call[0] for call in calls]
    assert verbs == ["delete", "apply", "apply", "wait"], verbs
    assert calls[0][1:3] == ("pod", fixture.pod), calls[0]
    assert "--ignore-not-found" in calls[0], calls[0]


def _ha009_snapshot(generation: int, pods: list[tuple[str, str]]) -> dict:
    return {
        name: {
            "generation": generation,
            "observed_generation": generation,
            "replicas": len(pods),
            "ready": len(pods),
            "refresh_annotation": None,
            "pods": [
                (pod, {"uid": uid, "ready": True, "restarts": 0}) for pod, uid in pods
            ],
        }
        for name in ha009.DEPLOYMENTS
    }


def test_ha009_rollout_wait_outlives_draining_old_pods(monkeypatch) -> None:
    """A Terminating control-worker Pod stays Ready for up to its 240 s grace.

    The wait must not return while any pre-rotation Pod uid is still listed,
    or _rotation_result reports "did not replace every old Pod" for a
    rollout that merely had not finished draining.
    """
    before = _ha009_snapshot(1, [("old-a", "uid-a"), ("old-b", "uid-b")])
    snapshots = iter(
        [
            _ha009_snapshot(2, [("old-a", "uid-a"), ("new-b", "uid-nb")]),
            _ha009_snapshot(2, [("new-a", "uid-na"), ("new-b", "uid-nb")]),
        ]
    )
    seen = []
    monkeypatch.setattr(ha009, "deployment_snapshot", lambda: next(snapshots))
    monkeypatch.setattr(ha009.time, "sleep", lambda _seconds: seen.append("slept"))

    after = ha009.wait_deployments(before, timeout_seconds=30)

    assert seen == ["slept"], "wait must poll past the still-draining old Pod"
    for name in ha009.DEPLOYMENTS:
        uids = {value["uid"] for _pod, value in after[name]["pods"]}
        assert uids == {"uid-na", "uid-nb"}, name

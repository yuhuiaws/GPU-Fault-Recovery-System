#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
for _path in (ROOT, ROOT / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from gpu_fault.admin.deadlines import (  # noqa: E402
    DeploymentDeadlineExceeded,
    deadline_scope,
)
from gpu_fault.models import Environment, WorkflowOperation  # noqa: E402
from gpu_fault.watcher import AttemptObservation, WorkloadPhase  # noqa: E402
from scripts.e2e.regional import run_collector_destructive as base  # noqa: E402
from scripts.e2e.regional import (  # noqa: E402
    run_destr009_workload_restart as workload_case,
)
from scripts.e2e.regional.acceptance_runner_common import (  # noqa: E402
    write_json_atomic,
)
from scripts.e2e.regional.collector_acceptance_fixture import (  # noqa: E402
    CollectorAcceptanceFixture,
)
from scripts.e2e.regional.collector_action_guard import (  # noqa: E402
    WINDOW,
    bounded_collector_case,
    finite_seconds,
    require_action_time,
)
from scripts.e2e.regional.host_probe_fixture import (  # noqa: E402
    HostProbeFixture,
    HostProbeSettings,
)
from scripts.e2e.regional.live_driver_guard import (  # noqa: E402
    CaseRunner,
    add_live_arguments,
    run_standard_case,
)
from scripts.e2e.regional.managed_workload_fixture import (  # noqa: E402
    ImagePrewarmFixture,
    ManagedWorkloadFixture,
    ManagedWorkloadSettings,
)
from scripts.e2e.regional.regional_commands import RegionalFixtureError  # noqa: E402
from scripts.e2e.regional.regional_live_fixture import (  # noqa: E402
    RegionalLiveFixture,
    RegionalLiveSettings,
    predecessor_evidence,
    required,
    run_case_main,
    settings_from_arguments,
)
from tools.pytest_result_identity import (  # noqa: E402
    normalized_pytest_nodeid,
)
from tools.run_fault_test_cases import (  # noqa: E402
    build_isolated_environment,
    run_reported_pytest,
)

CASE_ID = "GF-REGIONAL-COLLECT-016"
PREDECESSOR_CASE_ID = "GF-REGIONAL-COLLECT-014"
CONFIRMATION = "COLLECT016_EXECUTE"
# Missing attempts are refreshed for the watcher's default 300-second grace.
A_WORKLOAD_DRAIN_TIMEOUT_SECONDS = 600
# D section: RESET_GPU on a node that is running the managed workload. The
# compiler wraps the idle-node reset (run_destr001_gpu_reset.EXPECTED_STEPS)
# in STOP_WORKLOADS before the quiesce and RESTART_WORKLOAD after scheduling
# is restored (observed live 2026-09-06 09:29Z, workflow-bec004a7).
WORKLOAD_RESET_STEPS = [
    "FREEZE_EVIDENCE",
    "MARK_UNSCHEDULABLE",
    "STOP_WORKLOADS",
    "QUIESCE_GPU_SERVICES",
    "VERIFY_NO_GPU_CLIENTS",
    "RESET_GPU",
    "RESTORE_GPU_SERVICES",
    "VALIDATE_GPU",
    "RESTORE_SCHEDULING",
    "RESTART_WORKLOAD",
]
LOCAL_CONTRACT_SELECTORS = {
    "reset_and_restart": (
        "tests/hyperpod/test_e2e_distributed_xid_reset.py::"
        "test_three_node_job_stops_before_two_fault_gpu_resets",
        "tests/execution/test_restart_safety.py::"
        "test_restart_budget_blocks_second_restart_for_same_job",
    ),
    "dcgm_classification": (
        "tests/host_health/test_dcgm_diagnostic_analysis.py::"
        "test_config_severity_failures_are_configuration_only",
        "tests/host_health/test_dcgm_diagnostic_analysis.py::"
        "test_isolate_severity_alongside_config_still_drains",
        "tests/host_health/test_dcgm_diagnostic_analysis.py::"
        "test_missing_severity_is_not_configuration_only",
        "tests/node_agent/test_diagnostics.py::"
        "test_quick_dcgm_config_severity_failure_is_warn",
        "tests/node_agent/test_diagnostics.py::"
        "test_quick_dcgm_diagnostic_returns_structured_outcome",
        "tests/node_agent/test_diagnostics.py::"
        "test_quick_dcgm_diagnostic_invalid_json_is_inconclusive",
    ),
    "dcgm_control_action": (
        "tests/execution/test_misc.py::"
        "test_dcgm_config_severity_failure_does_not_drain",
        "tests/execution/test_node_action.py::"
        "test_node_action_adapter_interprets_dcgm_diagnostic_outcome",
        "tests/orchestration/test_misc.py::"
        "test_gpu_warning_runs_dcgm_diagnostic_and_failure_drains",
        "tests/orchestration/test_misc.py::"
        "test_active_workload_dcgm_execution_review_does_not_drain",
    ),
    "client_intersection": (
        "tests/node_agent/test_misc.py::"
        "test_verify_ignores_respawning_short_lived_device_clients",
        "tests/node_agent/test_misc.py::"
        "test_verify_still_fails_closed_on_persistent_device_client",
    ),
}


@dataclass(frozen=True)
class Settings:
    regional: RegionalLiveSettings
    site_file: Path
    host_probe_image: str
    predecessor_path: Path

    def environment(self) -> dict[str, str]:
        return {
            **self.regional.environment(),
            "GPU_FAULT_SITE_FILE": str(self.site_file),
            "GPU_FAULT_HOST_PROBE_IMAGE": self.host_probe_image,
            "GPU_FAULT_PREDECESSOR_EVIDENCE": str(self.predecessor_path),
        }


def configure(arguments: argparse.Namespace) -> Settings:
    predecessor = (
        Path(arguments.predecessor_evidence).expanduser().resolve()
        if arguments.predecessor_evidence
        else (
            arguments.run_dir
            / "cases"
            / PREDECESSOR_CASE_ID
            / f"{PREDECESSOR_CASE_ID}.json"
        ).resolve()
    )
    return Settings(
        regional=settings_from_arguments(arguments),
        site_file=Path(
            required(
                arguments.site_file or os.getenv("GPU_FAULT_SITE_FILE", ""),
                "regional site file",
            )
        )
        .expanduser()
        .resolve(),
        host_probe_image=required(
            arguments.host_probe_image or os.getenv("GPU_FAULT_HOST_PROBE_IMAGE", ""),
            "host probe image",
        ),
        predecessor_path=predecessor,
    )


def local_contract_proof(case_dir: Path) -> dict[str, Any]:
    selectors = [
        selector for group in LOCAL_CONTRACT_SELECTORS.values() for selector in group
    ]
    command = [sys.executable, "-m", "pytest", "-q", *selectors, "-n", "0"]
    completed, receipt, receipt_error = run_reported_pytest(
        command,
        environment=build_isolated_environment(),
        timeout_seconds=300,
    )
    errors = receipt.complete_selection_errors(selectors, root=ROOT)
    if receipt_error is not None:
        errors.append(receipt_error)
    if completed.returncode:
        errors.append("focused pytest did not finish successfully")
    log = case_dir / "focused-tests.log"
    log.write_text(
        (completed.stdout or "") + (completed.stderr or ""), encoding="utf-8"
    )
    log.chmod(0o600)
    result = {
        "proof_scope": "local-contract",
        "validation_level": "LOCAL",
        "live_negative_actions_performed": False,
        "verdict": "PASS" if not errors else "FAIL",
        "source_identity": (receipt.session or {}).get("source_identity"),
        "pytest_session": receipt.session,
        "pytest_records": receipt.records,
        "receipt_errors": errors,
        "groups": {
            name: {
                "verdict": "PASS" if not errors else "FAIL",
                "selectors": list(group),
                "executed_variants": sum(
                    len(receipt.selection(normalized_pytest_nodeid(item, root=ROOT))[0])
                    for item in group
                ),
            }
            for name, group in LOCAL_CONTRACT_SELECTORS.items()
        },
    }
    write_json_atomic(case_dir / "local-contracts.json", result)
    return result


def read_only_preflight(
    settings: Settings,
    case_dir: Path,
) -> dict[str, Any]:
    regional = RegionalLiveFixture(settings.regional)
    candidates = [
        item
        for item in regional.gpu_nodes()
        if item["ready"] == "True" and not item["unschedulable"] and not item["taints"]
    ]
    identity = regional.evidence_identity()
    predecessor = predecessor_evidence(
        settings.predecessor_path,
        PREDECESSOR_CASE_ID,
        **identity,
    )
    local_proof = local_contract_proof(case_dir)
    errors = []
    if not predecessor["valid"]:
        errors.append("COLLECT-014 predecessor evidence is not PASS")
    if not settings.site_file.is_file():
        errors.append("regional site file does not exist")
    if len(candidates) < 3:
        errors.append("fewer than three idle Ready GPU nodes")
    if regional.gpu_workloads():
        errors.append("GPU cluster already has active GPU workloads")
    if local_proof["verdict"] != "PASS":
        errors.append("focused regression tests failed")
    result = {
        **identity,
        "candidate_nodes": candidates,
        "predecessor": predecessor,
        "focused_tests_passed": local_proof["verdict"] == "PASS",
        "local_contracts": local_proof,
        "cpu_blast": regional.cpu_blast_snapshot(),
        "errors": errors,
    }
    write_json_atomic(case_dir / "preflight.json", result)
    return result


def workload_settings(
    settings: Settings,
    *,
    manifest: Path,
    job_id: str,
    attempt_id: str,
) -> workload_case.Settings:
    return workload_case.Settings(
        regional=settings.regional,
        site_file=settings.site_file,
        manifest=manifest,
        job_id=job_id,
        attempt_id=attempt_id,
        predecessor_path=settings.predecessor_path,
    )


def managed_fixture(
    settings: Settings,
    *,
    manifest: Path,
    job_id: str,
    attempt_id: str,
) -> ManagedWorkloadFixture:
    return ManagedWorkloadFixture(
        RegionalLiveFixture(settings.regional),
        ManagedWorkloadSettings(
            manifest=manifest,
            site_file=settings.site_file,
            job_id=job_id,
            attempt_id=attempt_id,
            restart_budget=1,
            expected_pods=3,
            expected_gpu_count=24,
        ),
    )


def restart_workload_job_id(workflow: dict[str, Any]) -> str | None:
    """The job the RESTART_WORKLOAD step was compiled for, from its parameters."""

    for step in workflow.get("official_steps") or []:
        if step.get("operation") != "RESTART_WORKLOAD":
            continue
        parameters = step.get("parameters") or {}
        value = parameters.get("job_id")
        return str(value) if value else None
    return None


def restart_budget_section_errors(
    state_a: dict[str, Any],
    state_b: dict[str, Any],
    *,
    job_id: str,
) -> list[str]:
    """What is unique to COLLECT-016 A/B beyond the DESTR-009 restart contract.

    A: the incident identified the workload (``workload_identity_source``), the
    workflow was never BLOCKED, and RESTART_WORKLOAD was compiled for *this*
    job. B: the second RESTART_APP failed for ``RESTART_BUDGET_EXHAUSTED`` and
    the budget still shows the one restart A spent -- not two, not zero.
    """

    errors: list[str] = []
    incident_a = state_a.get("incident") or {}
    workflow_a = state_a.get("workflow") or {}
    if incident_a.get("workload_identity_source") != "SOLE_ACTIVE_ATTEMPT_ON_NODE":
        errors.append("A: workload_identity_source is not SOLE_ACTIVE_ATTEMPT_ON_NODE")
    if list(workflow_a.get("blocked_reasons") or []):
        errors.append(
            f"A: RESTART_APP workflow was blocked: {workflow_a.get('blocked_reasons')}"
        )
    compiled_job = restart_workload_job_id(workflow_a)
    if compiled_job != job_id:
        errors.append(
            f"A: RESTART_WORKLOAD was compiled for job {compiled_job!r}, not {job_id!r}"
        )
    workflow_b = state_b.get("workflow") or {}
    if workflow_b.get("status") != "FAILED":
        errors.append("B: second RESTART_APP was not budget rejected")
    # withhold_exhausted_restart keeps the stop/repair steps, never the restart.
    completed = workflow_b.get("completed_operations", [])
    if not isinstance(completed, list) or any(
        not isinstance(item, str) or item not in WorkflowOperation for item in completed
    ):
        errors.append("B: completed workflow operations are malformed")
    elif "RESTART_WORKLOAD" in completed:
        errors.append("B: budget rejection completed RESTART_WORKLOAD")
    commands = state_b.get("commands")
    if not isinstance(commands, list) or any(
        not isinstance(item, dict)
        or not isinstance(item.get("step"), dict)
        or not isinstance(item["step"].get("operation"), str)
        or item["step"].get("operation") not in WorkflowOperation
        for item in commands
    ):
        errors.append("B: remote command report is missing or malformed")
    elif any(item["step"]["operation"] == "RESTART_WORKLOAD" for item in commands):
        errors.append(
            "B: budget rejection dispatched a RESTART_WORKLOAD remote command"
        )
    reasons = {
        str((item.get("details") or {}).get("reason") or "")
        for item in workflow_b.get("step_executions") or []
    }
    if "RESTART_BUDGET_EXHAUSTED" not in reasons:
        errors.append(
            f"B: no step execution failed for RESTART_BUDGET_EXHAUSTED: {sorted(reasons)}"
        )
    budget = state_b.get("restart_budget") or {}
    if budget.get("restart_count") != 1:
        errors.append(
            f"B: restart budget shows restart_count={budget.get('restart_count')!r}, "
            "expected the single restart A spent"
        )
    return errors


def run_restart_budget_sections(
    settings: Settings,
    regional: RegionalLiveFixture,
    case_dir: Path,
    suffix: str,
    *,
    workloads: list[ManagedWorkloadFixture],
    fixtures: list[CollectorAcceptanceFixture | HostProbeFixture],
    cleanup: base.CaseCleanup | None = None,
) -> dict[str, Any]:
    """A and B. Resources land in ``workloads``/``fixtures`` the moment they exist.

    They used to be returned and appended by the caller, so a timeout inside
    this helper (a 900s wait_running, a workflow wait) leaked a 24-GPU
    PyTorchJob and a privileged probe Pod that no finally block knew about.
    """

    cleanup = cleanup or base.CaseCleanup()
    job_id = f"c016-a-{suffix}"
    attempt_id = f"{job_id}-a001"
    manifest = base.render_named_training_manifest(
        case_dir / "a-workload.yaml",
        name=f"gpu-fault-c016-a-{suffix}",
    )
    workload = managed_fixture(
        settings,
        manifest=manifest,
        job_id=job_id,
        attempt_id=attempt_id,
    )
    workloads.append(workload)
    require_action_time(180)
    workload.submit()
    source = workload.wait_running(timeout_seconds=900)
    source_uids = {str(item["uid"]) for item in source["pods"]}
    target_node = str(source["pods"][0]["node"])
    case_settings = workload_settings(
        settings,
        manifest=manifest,
        job_id=job_id,
        attempt_id=attempt_id,
    )
    workload_case.wait_observation(
        regional,
        case_settings,
        node=target_node,
        expected_gpu_count=24,
    )
    collector = CollectorAcceptanceFixture(
        regional,
        node=target_node,
        image=settings.host_probe_image,
        case_id=CASE_ID,
        run_id=f"c016-a-{suffix}",
        case_dir=case_dir,
    )
    fixtures.append(collector)
    collector.create()
    bdf = str(collector.snapshot()["gpu_inventory"][0]["pci_bdf"])
    marker_a = f"c016-a-{int(time.time())}"
    injected_a = datetime.now(timezone.utc)
    cleanup.register_seed(collector, marker_a)
    collector.execute(
        "write-xid",
        "--xid",
        "31",
        "--marker",
        marker_a,
        "--pci-bdf",
        bdf,
        "--message",
        "MMU Fault RESTART_APP",
    )
    state_a = regional.wait_for_workflow(
        node=target_node,
        marker=marker_a,
        observed_after=injected_a,
        case_dir=case_dir / "a-restart",
        timeout_seconds=1200,
        job_id=job_id,
        attempt_id=attempt_id,
    )
    errors = workload_case.workflow_errors(state_a, expected_gpu_count=24, xid=31)
    if errors:
        raise RegionalFixtureError(
            "restart phase failed; refusing the budget injection"
        )
    target = capture_post_restart_workload(
        regional,
        workload,
        case_settings,
        state=state_a,
        node=target_node,
        source_uids=source_uids,
    )
    target_attempt_id = target["observation"]["attempt_id"]
    marker_b = f"c016-b-{int(time.time())}"
    injected_b = datetime.now(timezone.utc)
    cleanup.register_seed(collector, marker_b)
    collector.execute(
        "write-xid",
        "--xid",
        "31",
        "--marker",
        marker_b,
        "--pci-bdf",
        bdf,
        "--message",
        "MMU Fault RESTART_APP budget exhausted",
    )
    state_b = regional.wait_for_workflow(
        node=target_node,
        marker=marker_b,
        observed_after=injected_b,
        case_dir=case_dir / "b-budget",
        timeout_seconds=600,
        job_id=job_id,
        attempt_id=target_attempt_id,
    )
    workflow_b = state_b.get("workflow") or {}
    errors.extend(restart_budget_section_errors(state_a, state_b, job_id=job_id))
    return {
        "errors": errors,
        "a": {
            "marker": marker_a,
            "job_id": job_id,
            "source_attempt_id": attempt_id,
            "target_attempt_id": target_attempt_id,
            "source_pod_uids": sorted(source_uids),
            "target_pod_uids": sorted(str(item["uid"]) for item in target["pods"]),
            "workflow": state_a.get("workflow"),
            "incident": state_a.get("incident"),
        },
        "b": {
            "marker": marker_b,
            "workflow": workflow_b,
            "restart_budget": state_b.get("restart_budget"),
        },
    }


def capture_post_restart_workload(
    regional: RegionalLiveFixture,
    workload: ManagedWorkloadFixture,
    source_settings: workload_case.Settings,
    *,
    state: dict[str, Any],
    node: str,
    source_uids: set[str],
) -> dict[str, Any]:
    workload.authorize_restart(state)
    target = workload.wait_restarted(source_uids, timeout_seconds=900)
    pods = target.get("pods") or []
    uids = [pod.get("uid") for pod in pods]
    attempts = {pod.get("attempt_id") for pod in pods}
    if (
        not source_uids
        or len(pods) != len(source_uids)
        or any(not isinstance(uid, str) or not uid for uid in uids)
        or len(set(uids)) != len(uids)
        or set(uids).intersection(source_uids)
        or len(attempts) != 1
        or source_settings.attempt_id in attempts
        or any(
            pod.get("ready") is not True
            or pod.get("phase") != "Running"
            or not pod.get("node")
            for pod in pods
        )
    ):
        raise RegionalFixtureError(
            "replacement Pods do not prove a complete Ready new attempt"
        )
    new_attempt = next(iter(attempts))
    if not isinstance(new_attempt, str) or not new_attempt:
        raise RegionalFixtureError("replacement attempt ID is unknown")
    observation = workload_case.wait_observation(
        regional,
        replace(source_settings, attempt_id=new_attempt),
        node=node,
        expected_gpu_count=24,
    )
    try:
        observed_at = datetime.fromisoformat(
            str(observation.get("observed_at") or "").replace("Z", "+00:00")
        )
        age = (datetime.now(timezone.utc) - observed_at).total_seconds()
    except (TypeError, ValueError):
        raise RegionalFixtureError(
            "replacement observation timestamp is unknown"
        ) from None
    if (
        not 0 <= age <= 120
        or observation.get("workload_phase") != "RUNNING"
        or observation.get("cluster_id") != source_settings.regional.cluster_id
        or observation.get("job_id") != source_settings.job_id
        or observation.get("attempt_id") != new_attempt
    ):
        raise RegionalFixtureError(
            "replacement RUNNING observation identity or freshness is unproven"
        )
    return {
        "cluster_id": source_settings.regional.cluster_id,
        "job_id": source_settings.job_id,
        "node": node,
        "source_attempt_id": source_settings.attempt_id,
        "source_pod_uids": sorted(source_uids),
        "pods": pods,
        "observation": observation,
    }


def run_reset_section(
    settings: Settings,
    regional: RegionalLiveFixture,
    case_dir: Path,
    suffix: str,
    *,
    workloads: list[ManagedWorkloadFixture],
    fixtures: list[CollectorAcceptanceFixture | HostProbeFixture],
    cleanup: base.CaseCleanup | None = None,
) -> dict[str, Any]:
    """D. Like A/B, the workload and probe are registered as soon as they exist."""

    cleanup = cleanup or base.CaseCleanup()
    job_id = f"c016-d-{suffix}"
    attempt_id = f"{job_id}-a001"
    manifest = base.render_named_training_manifest(
        case_dir / "d-workload.yaml",
        name=f"gpu-fault-c016-d-{suffix}",
    )
    workload = managed_fixture(
        settings,
        manifest=manifest,
        job_id=job_id,
        attempt_id=attempt_id,
    )
    workloads.append(workload)
    require_action_time(180)
    workload.submit()
    source = workload.wait_running(timeout_seconds=900)
    source_uids = {str(item["uid"]) for item in source["pods"]}
    target_node = str(source["pods"][0]["node"])
    case_settings = workload_settings(
        settings,
        manifest=manifest,
        job_id=job_id,
        attempt_id=attempt_id,
    )
    workload_case.wait_observation(
        regional,
        case_settings,
        node=target_node,
        expected_gpu_count=24,
    )
    reset_host = HostProbeFixture(
        HostProbeSettings(
            kubeconfig=settings.regional.gpu_kubeconfig,
            context=settings.regional.gpu_context,
            namespace=settings.regional.namespace,
            node=target_node,
            image=settings.host_probe_image,
            case_id=CASE_ID,
            run_id=f"c016-d-{suffix}",
            probe_script=base.DESTRUCTIVE_PROBE,
            state_directory=case_dir / "host-probes",
            active_deadline_seconds=3600,
        )
    )
    fixtures.append(reset_host)
    require_action_time(180)
    reset_host.create()
    collector = CollectorAcceptanceFixture(
        regional,
        node=target_node,
        image=settings.host_probe_image,
        case_id=CASE_ID,
        run_id=f"c016-d-audit-{suffix}",
        case_dir=case_dir,
    )
    fixtures.append(collector)
    collector.create()
    marker = f"c016-d-{int(time.time())}"
    shim = base.Settings(
        regional=settings.regional,
        case_id=CASE_ID,
        node=target_node,
        second_node=None,
        host_probe_image=settings.host_probe_image,
        hyperpod_cluster="",
        executor_role_arn="",
        site_file=settings.site_file,
        predecessor_path=settings.predecessor_path,
    )
    state, errors = base.run_single_reset(
        shim,
        regional,
        reset_host,
        case_dir / "d-reset",
        collector=collector,
        cleanup=cleanup,
        xid=109,
        marker=marker,
        run_id=f"c016-d-{suffix}",
        node=target_node,
        expected_steps=WORKLOAD_RESET_STEPS,
        observe_post_restart_workload=lambda state: capture_post_restart_workload(
            regional,
            workload,
            case_settings,
            state=state,
            node=target_node,
            source_uids=source_uids,
        ),
    )
    target = state["post_restart_workload"]
    return {
        "errors": errors,
        "d": {
            "marker": marker,
            "workflow": state.get("workflow"),
            "source_pod_uids": sorted(source_uids),
            "target_pod_uids": sorted(str(item["uid"]) for item in target["pods"]),
        },
    }


def plan_details(settings: Settings, preflight: dict[str, Any]) -> dict[str, Any]:
    return {
        "risk": "destructive",
        "predecessor": preflight["predecessor"],
        "mutation": (
            "submit managed 24-GPU workloads, inject real kmsg XID31 twice "
            "and XID109 once, then verify restart budget and reset ordering"
        ),
        "proof_scopes": {
            "D": "live-positive-XID109-reset",
            "dcgm_and_client_negative_contracts": "source-bound LOCAL pytest only",
            "negative_gpu_holder_injection": False,
        },
        "preflight_identity": {
            "release_id": preflight["release_id"],
            "candidate_node_uids": sorted(
                str(item["uid"]) for item in preflight["candidate_nodes"]
            ),
        },
        "stop_conditions": [
            "COLLECT-014 predecessor evidence is not PASS",
            "fewer than three idle GPU nodes or image prewarm fails",
            "workload observation lacks 24 GPU UUIDs",
            "restart budget or real reset workflow differs from contract",
            "workload/probe cleanup leaves residuals",
        ],
        "rollback": {
            "delete_both_unique_workloads": True,
            "delete_prewarm_pods": True,
            "restore_quiesce_and_scheduling": True,
        },
        "preflight": preflight,
    }


def a_drain_observations(
    state: object,
    *,
    cluster_id: str,
    job_id: str,
    expected_attempt_ids: set[str],
) -> list[AttemptObservation]:
    if not isinstance(state, dict) or not isinstance(state.get("observations"), list):
        raise RegionalFixtureError(
            "A-phase workload drain observation report is missing"
        )
    observations: list[AttemptObservation] = []
    seen: set[str] = set()
    for value in state["observations"]:
        try:
            observation = AttemptObservation.model_validate_json(
                json.dumps(value), strict=True
            )
        except (TypeError, ValueError):
            raise RegionalFixtureError(
                "A-phase workload drain observation is malformed"
            ) from None
        if (
            observation.cluster_id != cluster_id
            or observation.job_id != job_id
            or not observation.attempt_id.strip()
            or observation.attempt_id in seen
            or observation.environment
            not in {Environment.KUBERNETES, Environment.EKS, Environment.HYPERPOD_EKS}
            or observation.observed_at.utcoffset() is None
            or not observation.runtime_profile_version.strip()
        ):
            raise RegionalFixtureError("A-phase workload drain observation is unbound")
        containers = observation.containers
        if (
            "containers" not in observation.model_fields_set
            or any(
                not {"terminated", "critical"} <= item.model_fields_set
                or not all(
                    value.strip()
                    for value in (
                        item.pod_uid,
                        item.pod_name,
                        item.container_name,
                        item.role,
                    )
                )
                for item in containers
            )
            or len({item.observation_key for item in containers}) != len(containers)
        ):
            raise RegionalFixtureError(
                "A-phase workload drain containers are incomplete"
            )
        if (
            observation.workload_phase in {WorkloadPhase.PENDING, WorkloadPhase.RUNNING}
            and all(item.terminated for item in containers)
            and len({item.rank for item in containers if item.critical})
            < observation.expected_critical_ranks
        ):
            raise RegionalFixtureError(
                "A-phase workload drain lacks a complete terminated rank set"
            )
        seen.add(observation.attempt_id)
        observations.append(observation)
    if not expected_attempt_ids <= seen:
        raise RegionalFixtureError(
            "A-phase workload drain report omits an expected attempt"
        )
    return observations


def wait_a_workload_drained(
    regional: RegionalLiveFixture,
    *,
    job_id: str,
    expected_attempt_ids: set[str],
    case_dir: Path,
    timeout_seconds: int = A_WORKLOAD_DRAIN_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Require durable non-correlation for every known A attempt before D."""

    cluster_id = regional.settings.cluster_id
    if (
        not isinstance(job_id, str)
        or not job_id.strip()
        or not isinstance(cluster_id, str)
        or not cluster_id.strip()
        or not expected_attempt_ids
        or any(
            not isinstance(item, str) or not item.strip()
            for item in expected_attempt_ids
        )
    ):
        raise RegionalFixtureError("A-phase workload drain identity is missing")
    timeout = finite_seconds(timeout_seconds)
    if (window := WINDOW.get()) is not None:
        timeout = finite_seconds(min(timeout, window.remaining()))
    result: dict[str, Any] = {
        "cluster_id": cluster_id,
        "job_id": job_id,
        "expected_attempt_ids": sorted(expected_attempt_ids),
        "drained": False,
        "entries": [],
    }
    path = case_dir / "a-drain.json"
    write_json_atomic(path, result)
    try:
        # Scope all reads/retries and sleeps together; leave cleanup's budget intact.
        with deadline_scope("COLLECT-016 A workload drain", timeout) as deadline:
            while True:
                deadline.remaining()
                require_action_time()
                state = regional.store_snapshot(job_id=job_id, queue_attempts=1)
                deadline.remaining()
                require_action_time()
                observations = a_drain_observations(
                    state,
                    cluster_id=cluster_id,
                    job_id=job_id,
                    expected_attempt_ids=expected_attempt_ids,
                )
                deadline.remaining()
                require_action_time()
                live = [
                    {
                        "attempt_id": item.attempt_id,
                        "workload_phase": item.workload_phase.value,
                        "live_nodes": sorted(
                            {
                                c.node_id
                                for c in item.containers
                                if not c.terminated and c.node_id
                            }
                        ),
                    }
                    for item in observations
                    if item.workload_phase
                    in {WorkloadPhase.PENDING, WorkloadPhase.RUNNING}
                    and any(not container.terminated for container in item.containers)
                ]
                result["entries"].append(
                    {
                        "observed_at": datetime.now(timezone.utc).isoformat(),
                        "observation_count": len(observations),
                        "live": live,
                    }
                )
                result["drained"] = not live
                write_json_atomic(path, result)
                if not live:
                    return result
                time.sleep(min(5, deadline.remaining()))
    except Exception as exc:
        result.update(drained=False, error_type=type(exc).__name__)
        write_json_atomic(path, result)
        if isinstance(exc, DeploymentDeadlineExceeded):
            raise RegionalFixtureError(
                "A-phase workload did not drain within its total deadline"
            ) from None
        raise


@bounded_collector_case
def execute_case(
    settings: Settings,
    run_dir: Path,
    attempt: int,
    maintenance_window_end: datetime,
) -> int:
    case_dir = run_dir / "cases" / CASE_ID
    case_dir.mkdir(parents=True, exist_ok=True)
    preflight = read_only_preflight(settings, case_dir)
    if preflight["errors"]:
        raise RegionalFixtureError(
            "preflight failed: " + "; ".join(preflight["errors"])
        )
    if datetime.now(timezone.utc) >= maintenance_window_end:
        raise RegionalFixtureError("approved maintenance window has ended")
    regional = RegionalLiveFixture(settings.regional)
    candidates = [str(item["name"]) for item in preflight["candidate_nodes"]]
    prewarm = ImagePrewarmFixture(
        regional,
        case_id=CASE_ID,
        run_id=f"c016-{attempt}",
    )
    workloads: list[ManagedWorkloadFixture] = []
    fixtures: list[CollectorAcceptanceFixture | HostProbeFixture] = []
    result: dict[str, Any] = {
        "case_id": CASE_ID,
        "attempt": attempt,
        "verdict": "FAIL",
        "errors": [],
        "local_contracts": preflight.get("local_contracts"),
        **regional.evidence_identity(),
    }
    cleanup = base.CaseCleanup()
    try:
        require_action_time(180)
        prewarm.create(candidates)
        suffix = f"{int(time.time())}-{attempt}"
        ab = run_restart_budget_sections(
            settings,
            regional,
            case_dir,
            suffix,
            workloads=workloads,
            fixtures=fixtures,
            cleanup=cleanup,
        )
        result["errors"].extend(ab["errors"])
        result.update({"a": ab["a"], "b": ab["b"]})
        if result["errors"]:
            raise RegionalFixtureError("A/B failed; refusing the reset phase")
        a = ab["a"]
        job_id = a.get("job_id")
        source_attempt = a.get("source_attempt_id")
        target_attempt = a.get("target_attempt_id")
        if (
            len(workloads) != 1
            or not isinstance(job_id, str)
            or not job_id.strip()
            or job_id != workloads[0].settings.job_id
            or source_attempt != workloads[0].settings.attempt_id
            or not isinstance(source_attempt, str)
            or not source_attempt.strip()
            or not isinstance(target_attempt, str)
            or not target_attempt.strip()
            or target_attempt == source_attempt
        ):
            raise RegionalFixtureError("A-phase workload identity is unbound")
        workloads[0].delete()
        result["a_drain"] = wait_a_workload_drained(
            regional,
            job_id=job_id,
            expected_attempt_ids={source_attempt, target_attempt},
            case_dir=case_dir,
        )
        drain = result["a_drain"]
        if (
            not isinstance(drain, dict)
            or drain.get("drained") is not True
            or drain.get("job_id") != job_id
            or drain.get("cluster_id") != settings.regional.cluster_id
            or drain.get("expected_attempt_ids")
            != sorted({source_attempt, target_attempt})
        ):
            raise RegionalFixtureError(
                "A-phase workload drain proof is missing or unbound"
            )
        if datetime.now(timezone.utc) >= maintenance_window_end:
            raise RegionalFixtureError("approved maintenance window has ended")
        d = run_reset_section(
            settings,
            regional,
            case_dir,
            suffix,
            workloads=workloads,
            fixtures=fixtures,
            cleanup=cleanup,
        )
        result["errors"].extend(d["errors"])
        result["d"] = d["d"]
        if regional.cpu_blast_snapshot() != preflight["cpu_blast"]:
            result["errors"].append("control-plane EKS state differs from baseline")
        result["verdict"] = "PASS" if not result["errors"] else "FAIL"
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["verdict"] = "FAIL"
    finally:
        if cleanup.seed_markers or cleanup.incident_states:
            try:
                state = regional.store_snapshot()
                released = cleanup.finish(
                    profile_version=str(
                        (state.get("profile") or {}).get("profile_version") or ""
                    ),
                    reason="COLLECT-016 validated cleanup",
                )
                result["errors"].extend(released["errors"])
            except Exception as exc:
                result["errors"].append(
                    f"incident cleanup failed: {type(exc).__name__}: {exc}"
                )
        for workload in workloads:
            try:
                workload.delete()
            except Exception as exc:
                result["errors"].append(
                    f"workload cleanup failed: {type(exc).__name__}: {exc}"
                )
        for fixture in fixtures:
            try:
                residuals = fixture.cleanup()
                if isinstance(residuals, dict) and any(residuals.values()):
                    result["errors"].append("probe resources remain")
            except Exception as exc:
                result["errors"].append(
                    f"probe cleanup failed: {type(exc).__name__}: {exc}"
                )
        try:
            residuals = prewarm.cleanup()
            if any(residuals.values()):
                result["errors"].append("prewarm Pods remain")
        except Exception as exc:
            result["errors"].append(
                f"prewarm cleanup failed: {type(exc).__name__}: {exc}"
            )
        if result["errors"]:
            result["verdict"] = "FAIL"
    write_json_atomic(case_dir / f"{CASE_ID}.json", result)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["verdict"] == "PASS" else 1


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        description="Run COLLECT-016 managed training recovery acceptance."
    )
    add_live_arguments(value, confirmation=CONFIRMATION)
    value.add_argument("--cpu-kubeconfig", default="")
    value.add_argument("--gpu-kubeconfig", default="")
    value.add_argument("--gpu-context", default="")
    value.add_argument("--namespace", default="gpu-fault-system")
    value.add_argument("--cluster-id", default="")
    value.add_argument("--region", default="")
    value.add_argument("--site-file", default="")
    value.add_argument("--host-probe-image", default="")
    value.add_argument("--predecessor-evidence", default="")
    return value


CASE = CaseRunner(
    case_id=CASE_ID,
    confirmation=CONFIRMATION,
    parser=parser,
    configure=configure,
    read_only_preflight=read_only_preflight,
    plan_details=plan_details,
    execute_case=execute_case,
)


def main() -> int:
    return run_standard_case(CASE)


if __name__ == "__main__":
    raise SystemExit(run_case_main(main))

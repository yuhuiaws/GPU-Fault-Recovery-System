#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.e2e.regional.acceptance_runner_common import (  # noqa: E402
    write_json_atomic,
)
from scripts.e2e.regional.live_driver_guard import (  # noqa: E402
    add_live_arguments,
    authorize_execution,
    build_plan,
    install_site_profile,
)
from scripts.e2e.regional.managed_workload_fixture import (  # noqa: E402
    ImagePrewarmFixture,
    ManagedWorkloadFixture,
    ManagedWorkloadSettings,
    render_node_pinned_manifest,
)
from scripts.e2e.regional.regional_live_fixture import (  # noqa: E402
    RegionalFixtureError,
    RegionalLiveFixture,
    RegionalLiveSettings,
    install_abort_signals,
    predecessor_evidence,
    required,
    run_case_main,
    settings_from_arguments,
)
from scripts.e2e.regional.warm_spare_fixture import (  # noqa: E402
    HYPERPOD_HEALTH_LABEL,
    INSTANCE_GROUP_LABEL,
    INSTANCE_TYPE_LABELS,
    OWNERSHIP_ANNOTATIONS,
    PROVIDER_REPLACE_EVENTS,
    QUARANTINE_TAINT,
    SPARE_LABEL,
    SPARE_POOL_STATE_ANNOTATION,
    SPARE_RESERVATION_ANNOTATION,
    WarmSpareLiveFixture,
)

DEFAULT_MANIFEST = (
    Path(__file__).with_name("manifests")
    / "training"
    / "single-node-warm-spare-pytorchjob.yaml"
)
CASE_ID = "GF-REGIONAL-DESTR-003"
PREDECESSOR_CASE_ID = "GF-REGIONAL-DESTR-012"
CONFIRMATION = "DESTR003_ACTIVATE_HEALTHY_WARM_SPARE"
EXPECTED_OPERATIONS = [
    "FREEZE_EVIDENCE",
    "MARK_UNSCHEDULABLE",
    "QUARANTINE",
    "STOP_WORKLOADS",
    "REPLACE_NODE",
    "VALIDATE_GPU",
    "VALIDATE_FABRIC",
    "RESTORE_SCHEDULING",
    "RESTART_WORKLOAD",
]


@dataclass(frozen=True)
class Settings:
    regional: RegionalLiveSettings
    site_file: Path
    manifest: Path
    hyperpod_cluster: str
    fault_node: str
    spare_node: str
    job_id: str
    attempt_id: str
    predecessor_path: Path

    def environment(self) -> dict[str, str]:
        return {
            **self.regional.environment(),
            "GPU_FAULT_SITE_FILE": str(self.site_file),
            "GPU_FAULT_TRAINING_MANIFEST": str(self.manifest),
            "GPU_FAULT_HYPERPOD_CLUSTER_NAME": self.hyperpod_cluster,
            "GPU_FAULT_FAULT_NODE": self.fault_node,
            "GPU_FAULT_SPARE_NODE": self.spare_node,
            "GPU_FAULT_TEST_JOB_ID": self.job_id,
            "GPU_FAULT_TEST_ATTEMPT_ID": self.attempt_id,
            "GPU_FAULT_PREDECESSOR_EVIDENCE": str(self.predecessor_path),
        }


def derived_identity(run_dir: Path, attempt: int) -> tuple[str, str]:
    suffix = hashlib.sha256(
        f"{run_dir.resolve()}\0{attempt}\0{CASE_ID}".encode()
    ).hexdigest()[:12]
    job_id = f"destr003-{suffix}"
    return job_id, f"{job_id}-a001"


def configure(arguments: argparse.Namespace) -> Settings:
    default_job, _default_attempt = derived_identity(
        arguments.run_dir,
        arguments.attempt,
    )
    job_id = arguments.job_id.strip() or default_job
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
        manifest=Path(arguments.manifest).expanduser().resolve(),
        hyperpod_cluster=required(
            arguments.hyperpod_cluster
            or os.getenv("GPU_FAULT_HYPERPOD_CLUSTER_NAME", ""),
            "HyperPod cluster name",
        ),
        fault_node=required(
            arguments.fault_node or os.getenv("GPU_FAULT_FAULT_NODE", ""),
            "fault node",
        ),
        spare_node=required(
            arguments.spare_node or os.getenv("GPU_FAULT_SPARE_NODE", ""),
            "spare node",
        ),
        job_id=job_id,
        attempt_id=arguments.attempt_id.strip() or f"{job_id}-a001",
        predecessor_path=predecessor,
    )


def focused_tests(case_dir: Path) -> dict[str, Any]:
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "tests/hyperpod/test_hyperpod_spares.py::"
        "test_local_only_allocation_uses_kubernetes_node_inventory",
        "tests/execution/test_node_action.py::"
        "test_hyperpod_replace_records_activated_spare_nodes",
        "tests/execution/test_node_action.py::"
        "test_hyperpod_spare_checker_uses_node_agent_for_each_phase",
        "tests/regional/test_api.py::"
        "test_synthetic_replacement_requires_flag_and_execution_token",
    ]
    completed = RegionalLiveFixture.run(
        command,
        cwd=ROOT,
        check=False,
        timeout=300,
    )
    path = case_dir / "focused-tests.log"
    path.write_text(completed.stdout + completed.stderr, encoding="utf-8")
    path.chmod(0o600)
    return {
        "passed": completed.returncode == 0,
        "returncode": completed.returncode,
        "command": command,
    }


def capability(profile: dict[str, Any] | None, name: str) -> dict[str, Any] | None:
    for item in (profile or {}).get("capabilities", []):
        if isinstance(item, dict) and item.get("capability") == name:
            return cast(dict[str, Any], item)
    return None


def agent_by_node(state: dict[str, Any], node: str) -> dict[str, Any] | None:
    matches = [
        item for item in state.get("agents") or [] if item.get("node_id") == node
    ]
    return cast(dict[str, Any], matches[0]) if len(matches) == 1 else None


def instance_type(snapshot: dict[str, Any]) -> str | None:
    return next(
        (
            snapshot["labels"].get(key)
            for key in INSTANCE_TYPE_LABELS
            if snapshot["labels"].get(key)
        ),
        None,
    )


def profile_errors(profile: dict[str, Any] | None) -> list[str]:
    expected = {
        "nodeReplace": ("gpu-fault-hyperpod-adapter", "regional-cluster-executor"),
        "workloadStop": (
            "gpu-fault-kubernetes-adapter",
            "regional-cluster-executor",
        ),
        "workloadRestart": (
            "gpu-fault-kubernetes-adapter",
            "regional-cluster-executor",
        ),
    }
    errors = []
    for name, (owner, adapter) in expected.items():
        item = capability(profile, name)
        if item is None:
            errors.append(f"runtime profile has no {name} capability")
        elif (
            item.get("mode") != "OWN"
            or item.get("owner") != owner
            or item.get("adapter") != adapter
        ):
            errors.append(f"{name} has the wrong owner or adapter")
    return errors


def preflight_errors(
    settings: Settings,
    *,
    fault: dict[str, Any],
    spare: dict[str, Any],
    state: dict[str, Any],
    spare_nodes: list[str],
    cluster: dict[str, Any],
    executor_env: list[dict[str, str | None]],
    synthetic_gates: list[dict[str, str | None]],
    gpu_workloads: list[dict[str, Any]],
    predecessor: dict[str, Any],
    tests: dict[str, Any],
) -> list[str]:
    errors = []
    if not predecessor["valid"]:
        errors.append("DESTR-012 predecessor evidence is not PASS")
    if settings.fault_node == settings.spare_node:
        errors.append("fault and spare nodes are identical")
    if fault["ready"] != "True" or fault["unschedulable"]:
        errors.append("fault node is not Ready and schedulable")
    if fault["labels"].get(SPARE_LABEL) == "true":
        errors.append("fault node is labeled as a spare")
    if any(item.get("key") == QUARANTINE_TAINT for item in fault["taints"]) or any(
        fault["annotations"].get(key) for key in OWNERSHIP_ANNOTATIONS
    ):
        errors.append("fault node has pre-existing quarantine ownership")
    if spare["ready"] != "True" or not spare["unschedulable"]:
        errors.append("spare node is not Ready and cordoned")
    if spare["labels"].get(SPARE_LABEL) != "true":
        errors.append("spare node is not declared by the spare label")
    if spare["labels"].get(HYPERPOD_HEALTH_LABEL) != "Schedulable":
        errors.append("spare HyperPod health label is not Schedulable")
    if spare["annotations"].get(SPARE_RESERVATION_ANNOTATION):
        errors.append("spare node already has a reservation")
    if spare["annotations"].get(SPARE_POOL_STATE_ANNOTATION) not in {
        None,
        "AVAILABLE",
    }:
        errors.append("spare pool state is not AVAILABLE")
    if spare_nodes != [settings.spare_node]:
        errors.append("declared spare set does not exactly match --spare-node")
    if fault["labels"].get(INSTANCE_GROUP_LABEL) != spare["labels"].get(
        INSTANCE_GROUP_LABEL
    ) or instance_type(fault) != instance_type(spare):
        errors.append("fault and spare topology do not match")
    active_on_targets = [
        item
        for item in gpu_workloads
        if item.get("node") in {settings.fault_node, settings.spare_node}
    ]
    if active_on_targets:
        errors.append("fault or spare node already has a GPU workload")
    for node in (settings.fault_node, settings.spare_node):
        agent = agent_by_node(state, node)
        if agent is None or agent.get("lifecycle_state") != "ACTIVE":
            errors.append(f"{node} does not have exactly one ACTIVE Agent")
    errors.extend(profile_errors(state.get("profile")))
    if cluster.get("status") != "InService" or cluster.get("node_recovery") != "None":
        errors.append("HyperPod cluster is not InService with NodeRecovery=None")
    if len(executor_env) < 1 or any(
        item.get("spare_failover") != "true"
        or item.get("remote_state") != "true"
        or item.get("allow_replace") != "false"
        or item.get("spare_label") not in {None, SPARE_LABEL}
        for item in executor_env
    ):
        errors.append("executor warm-spare safety environment is inconsistent")
    if len(synthetic_gates) < 1 or any(
        str(item.get("enabled") or "").lower() != "true" for item in synthetic_gates
    ):
        errors.append(
            "synthetic replacement test route is not enabled on every API Pod"
        )
    if not tests["passed"]:
        errors.append("focused regression tests failed")
    return errors


def read_only_preflight(
    settings: Settings,
    case_dir: Path,
) -> dict[str, Any]:
    if not settings.site_file.is_file() or not settings.manifest.is_file():
        raise RegionalFixtureError("site file or training manifest does not exist")
    regional = RegionalLiveFixture(settings.regional)
    warm = WarmSpareLiveFixture(regional, settings.hyperpod_cluster)
    fault = warm.node_snapshot(settings.fault_node)
    spare = warm.node_snapshot(settings.spare_node)
    state = warm.store_snapshot()
    fault_state = regional.store_snapshot(node=settings.fault_node)
    state["profile"] = fault_state.get("profile")
    state["release_id"] = fault_state.get("release_id")
    spare_nodes = warm.spare_nodes()
    cluster = warm.cluster_recovery()
    executor_env = warm.executor_environment()
    synthetic_gates = warm.synthetic_replacement_gates()
    gpu_workloads = regional.gpu_workloads()
    predecessor = predecessor_evidence(
        settings.predecessor_path,
        PREDECESSOR_CASE_ID,
    )
    tests = focused_tests(case_dir)
    result = {
        "release_id": state.get("release_id"),
        "fault_node": fault,
        "spare_node": spare,
        "declared_spares": spare_nodes,
        "cluster": cluster,
        "provider_inventory": warm.provider_inventory(),
        "store": state,
        "executor_environment": executor_env,
        "synthetic_replacement_gates": synthetic_gates,
        "gpu_workloads": gpu_workloads,
        "predecessor": predecessor,
        "focused_tests": tests,
        "cpu_blast": regional.cpu_blast_snapshot(),
    }
    result["errors"] = preflight_errors(
        settings,
        fault=fault,
        spare=spare,
        state=state,
        spare_nodes=spare_nodes,
        cluster=cluster,
        executor_env=executor_env,
        synthetic_gates=synthetic_gates,
        gpu_workloads=gpu_workloads,
        predecessor=predecessor,
        tests=tests,
    )
    write_json_atomic(case_dir / "preflight.json", result)
    return result


def observation_gpu_uuids(observation: dict[str, Any]) -> list[str]:
    return sorted(
        {
            str(gpu_uuid)
            for container in observation.get("containers", [])
            for gpu_uuid in container.get("gpu_uuids", [])
        }
    )


def wait_observation(
    warm: WarmSpareLiveFixture,
    settings: Settings,
    *,
    timeout_seconds: int = 180,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    last: list[dict[str, Any]] = []
    while time.monotonic() < deadline:
        state = warm.store_snapshot(
            job_id=settings.job_id,
            attempt_id=settings.attempt_id,
        )
        last = state.get("observations") or []
        if len(last) == 1:
            observation = last[0]
            if (
                observation.get("workload_phase") == "RUNNING"
                and len(observation_gpu_uuids(observation)) == 8
            ):
                return cast(dict[str, Any], observation)
        time.sleep(5)
    raise RegionalFixtureError(f"8-GPU attempt observation did not appear: {last}")


def replacement_payload(
    settings: Settings,
    *,
    event_id: str,
    observation: dict[str, Any],
    observed_at: datetime,
) -> dict[str, Any]:
    return {
        "event_id": event_id,
        "cluster_id": settings.regional.cluster_id,
        "node_id": settings.fault_node,
        "observed_at": observed_at.isoformat(),
        "runtime_profile_version": observation["runtime_profile_version"],
        "job_id": settings.job_id,
        "attempt_id": settings.attempt_id,
        "affected_workload_ids": observation["workload_ids"],
        "gpu_uuids": observation_gpu_uuids(observation),
        "reason": "synthetic warm-spare acceptance trigger",
        "replacement_strategy": "HEALTHY_WARM_SPARE_ONLY",
        "synthetic": True,
    }


def terminal_execution(
    workflow: dict[str, Any],
    operation: str,
) -> dict[str, Any] | None:
    matches = [
        item
        for item in workflow.get("step_executions", [])
        if item.get("operation") == operation
        and item.get("status") in {"SUCCEEDED", "FAILED"}
    ]
    return cast(dict[str, Any], matches[-1]) if matches else None


def workflow_errors(
    state: dict[str, Any],
    settings: Settings,
) -> list[str]:
    workflow = state.get("workflow") or {}
    errors = []
    operations = [item.get("operation") for item in workflow.get("official_steps", [])]
    if workflow.get("status") != "SUCCEEDED":
        errors.append("warm-spare workflow is not SUCCEEDED")
    if operations != EXPECTED_OPERATIONS:
        errors.append("warm-spare workflow operations differ from the contract")
    replace_step = next(
        (
            item
            for item in workflow.get("official_steps", [])
            if item.get("operation") == "REPLACE_NODE"
        ),
        None,
    )
    if (replace_step or {}).get("parameters") != {
        "replacement_strategy": "HEALTHY_WARM_SPARE_ONLY"
    }:
        errors.append("REPLACE_NODE lacks HEALTHY_WARM_SPARE_ONLY")
    execution = terminal_execution(workflow, "REPLACE_NODE")
    details = (execution or {}).get("details") or {}
    if execution is None or execution.get("status") != "SUCCEEDED":
        errors.append("REPLACE_NODE did not reach SUCCEEDED")
    if details.get("action") != "SPARE_FAILOVER":
        errors.append("REPLACE_NODE did not report SPARE_FAILOVER")
    if details.get("activated_spare_nodes") != [settings.spare_node]:
        errors.append("REPLACE_NODE activated an unexpected spare")
    if details.get("provider_mutation_submitted") is not False:
        errors.append("REPLACE_NODE did not deny provider mutation")
    rebindings = details.get("node_rebindings") or {}
    if rebindings.get(settings.fault_node) != settings.spare_node:
        errors.append("REPLACE_NODE did not rebind fault node to spare")
    restart = terminal_execution(workflow, "RESTART_WORKLOAD") or {}
    restart_details = restart.get("details") or {}
    if restart.get("status") != "SUCCEEDED":
        errors.append("RESTART_WORKLOAD did not reach SUCCEEDED")
    if restart_details.get("source_gpu_count") != 8:
        errors.append("source GPU count is not 8")
    if restart_details.get("target_gpu_count") != 8:
        errors.append("target GPU count is not 8")
    if restart_details.get("restart_count") != 1:
        errors.append("restart count is not one")
    notification_id = details.get("notification_id")
    if not notification_id:
        errors.append("warm-spare success notification ID is missing")
    elif not any(
        item.get("notification_id") == notification_id
        for item in state.get("notifications") or []
    ):
        errors.append("warm-spare success notification was not persisted")
    return errors


def plan_details(settings: Settings, preflight: dict[str, Any]) -> dict[str, Any]:
    state = preflight["store"]
    return {
        "risk": "destructive-warm-spare",
        "predecessor": preflight["predecessor"],
        "fault_node": settings.fault_node,
        "spare_node": settings.spare_node,
        "job_id": settings.job_id,
        "attempt_id": settings.attempt_id,
        "manifest": str(settings.manifest),
        "synthetic_trigger": True,
        "mutation": (
            "submit one node-pinned 8-GPU managed PyTorchJob and use the "
            "execution-token protected synthetic replacement endpoint to drive "
            "the real warm-spare workflow; no hardware fault is claimed"
        ),
        "preflight_identity": {
            "release_id": preflight["release_id"],
            "fault_node_uid": preflight["fault_node"]["uid"],
            "spare_node_uid": preflight["spare_node"]["uid"],
            "runtime_profile_version": (state.get("profile") or {}).get(
                "profile_version"
            ),
            "provider_inventory_sha256": preflight["provider_inventory"]["sha256"],
        },
        "stop_conditions": [
            "DESTR-012 predecessor evidence is not PASS",
            "fault/spare topology, labels, Agent or release drift",
            "NodeRecovery is not None or provider replace is enabled",
            "training image, scheduling, NCCL or observation failure",
            "workflow does not use HEALTHY_WARM_SPARE_ONLY",
            "unexpected spare allocation or provider mutation",
            "restart budget or GPU count differs from one/8",
            "cleanup cannot release the spare or restore the fault node",
        ],
        "rollback": {
            "delete_test_workload": True,
            "release_only_the_incident_spare_reservation": True,
            "reactivate_only_the_revoked_fault_agent": True,
            "restore_fault_node_via_validation_first_workflow": True,
            "never_call_provider_replace": True,
        },
        "preflight": preflight,
    }


def verify_plan_identity(
    case_dir: Path,
    preflight: dict[str, Any],
) -> None:
    plan = json.loads((case_dir / "plan.json").read_text(encoding="utf-8"))
    planned = plan["details"]["preflight_identity"]
    current = {
        "release_id": preflight["release_id"],
        "fault_node_uid": preflight["fault_node"]["uid"],
        "spare_node_uid": preflight["spare_node"]["uid"],
        "runtime_profile_version": (preflight["store"].get("profile") or {}).get(
            "profile_version"
        ),
        "provider_inventory_sha256": preflight["provider_inventory"]["sha256"],
    }
    if current != planned:
        raise RegionalFixtureError(f"DESTR-003 plan drifted: {planned} != {current}")


def cleanup_case(
    *,
    warm: WarmSpareLiveFixture,
    regional: RegionalLiveFixture,
    workload: ManagedWorkloadFixture,
    prewarm: ImagePrewarmFixture,
    settings: Settings,
    incident_id: str,
    profile_version: str,
) -> dict[str, Any]:
    result: dict[str, Any] = {"errors": []}
    try:
        workload.delete()
    except Exception as exc:
        result["errors"].append(f"workload cleanup: {type(exc).__name__}: {exc}")
    if incident_id:
        try:
            result["spare_release"] = warm.release_spares(
                [settings.spare_node],
                incident_id,
            )
        except Exception as exc:
            result["errors"].append(f"spare release: {type(exc).__name__}: {exc}")
        try:
            result["agent_reactivation"] = warm.reactivate_agent(settings.fault_node)
            warm.wait_agent_active(settings.fault_node)
        except Exception as exc:
            result["errors"].append(f"agent reactivation: {type(exc).__name__}: {exc}")
        try:
            fault = warm.node_snapshot(settings.fault_node)
            if fault["annotations"].get("gpu-fault.io/incident-id") == incident_id:
                created = warm.create_restore_workflow(
                    incident_id=incident_id,
                    node=settings.fault_node,
                    profile_version=profile_version,
                    reason="DESTR-003 validated cleanup",
                )
                result["restore_workflow"] = warm.wait_workflow_id(
                    str(created["workflow_request_id"])
                )
                if result["restore_workflow"].get("status") != "SUCCEEDED":
                    result["errors"].append("fault-node restore workflow failed")
        except Exception as exc:
            result["errors"].append(f"fault-node restore: {type(exc).__name__}: {exc}")
    try:
        residuals = prewarm.cleanup()
        result["prewarm_residuals"] = residuals
        if any(residuals.values()):
            result["errors"].append("image prewarm Pods remain")
    except Exception as exc:
        result["errors"].append(f"prewarm cleanup: {type(exc).__name__}: {exc}")
    try:
        result["fault_node"] = warm.node_snapshot(settings.fault_node)
        result["spare_node"] = warm.node_snapshot(settings.spare_node)
        fault = result["fault_node"]
        spare = result["spare_node"]
        if (
            fault["ready"] != "True"
            or fault["unschedulable"]
            or any(item.get("key") == QUARANTINE_TAINT for item in fault["taints"])
            or any(fault["annotations"].get(key) for key in OWNERSHIP_ANNOTATIONS)
        ):
            result["errors"].append(
                "fault node did not return to Ready/schedulable/unowned"
            )
        if (
            spare["ready"] != "True"
            or not spare["unschedulable"]
            or spare["labels"].get(SPARE_LABEL) != "true"
            or spare["annotations"].get(SPARE_RESERVATION_ANNOTATION)
            or spare["annotations"].get(SPARE_POOL_STATE_ANNOTATION)
            not in {None, "AVAILABLE"}
        ):
            result["errors"].append(
                "spare node did not return to the available cordoned pool"
            )
    except Exception as exc:
        result["errors"].append(f"postflight snapshot: {type(exc).__name__}: {exc}")
    return result


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
    verify_plan_identity(case_dir, preflight)
    regional = RegionalLiveFixture(settings.regional)
    warm = WarmSpareLiveFixture(regional, settings.hyperpod_cluster)
    run_id = f"destr003-{run_dir.name.rsplit('-', 1)[-1].lower()}-a{attempt}"
    pinned_manifest = render_node_pinned_manifest(
        settings.manifest,
        case_dir / "pinned-workload.yaml",
        node=settings.fault_node,
    )
    workload = ManagedWorkloadFixture(
        regional,
        ManagedWorkloadSettings(
            manifest=pinned_manifest,
            site_file=settings.site_file,
            job_id=settings.job_id,
            attempt_id=settings.attempt_id,
            restart_budget=1,
            expected_pods=1,
            expected_gpu_count=8,
        ),
    )
    prewarm = ImagePrewarmFixture(regional, case_id=CASE_ID, run_id=run_id)
    result: dict[str, Any] = {
        "case_id": CASE_ID,
        "attempt": attempt,
        "verdict": "FAIL",
        "synthetic_trigger": True,
        "maintenance_window_end": maintenance_window_end.isoformat(),
    }
    incident_id = ""
    try:
        prewarm.create([settings.fault_node, settings.spare_node])
        submission = workload.submit()
        write_json_atomic(case_dir / "submission.json", submission)
        source = workload.wait_running(timeout_seconds=900)
        write_json_atomic(case_dir / "workload-source.json", source)
        source_uids = {str(item["uid"]) for item in source["pods"]}
        observation = wait_observation(warm, settings)
        write_json_atomic(case_dir / "source-observation.json", observation)
        if datetime.now(timezone.utc) >= maintenance_window_end:
            raise RegionalFixtureError(
                "approved maintenance window ended before replacement trigger"
            )
        event_id = f"destr003-{int(time.time())}-a{attempt}"
        started_at = datetime.now(timezone.utc)
        injection = warm.post_synthetic_replacement(
            replacement_payload(
                settings,
                event_id=event_id,
                observation=observation,
                observed_at=started_at,
            )
        )
        write_json_atomic(case_dir / "injection.json", injection)
        if injection.get("status") != 200:
            raise RegionalFixtureError(
                f"synthetic replacement endpoint failed: {injection}"
            )
        state = warm.wait_for_workflow(
            event_id=event_id,
            job_id=settings.job_id,
            attempt_id=settings.attempt_id,
            case_dir=case_dir,
            timeout_seconds=1500,
        )
        write_json_atomic(case_dir / "workflow-state.json", state)
        incident_id = str((state.get("incident") or {}).get("incident_id") or "")
        errors = workflow_errors(state, settings)
        budget = state.get("restart_budget") or {}
        if budget.get("budget") != 1 or budget.get("restart_count") != 1:
            errors.append("restart budget did not advance exactly once")
        target = workload.wait_restarted(source_uids, timeout_seconds=900)
        write_json_atomic(case_dir / "workload-target.json", target)
        if {item["node"] for item in target["pods"]} != {settings.spare_node}:
            errors.append("restarted workload is not on the warm spare")
        fault_after = warm.node_snapshot(settings.fault_node)
        spare_after = warm.node_snapshot(settings.spare_node)
        write_json_atomic(
            case_dir / "nodes-after.json",
            {
                "fault": fault_after,
                "spare": spare_after,
            },
        )
        if not fault_after["unschedulable"] or not any(
            item.get("key") == QUARANTINE_TAINT for item in fault_after["taints"]
        ):
            errors.append("fault node is not quarantined after failover")
        if fault_after["annotations"].get("gpu-fault.io/incident-id") != incident_id:
            errors.append("fault node quarantine ownership is incorrect")
        if spare_after["annotations"].get(SPARE_RESERVATION_ANNOTATION) != incident_id:
            errors.append("spare reservation does not reference the incident")
        if spare_after["annotations"].get(SPARE_POOL_STATE_ANNOTATION) != "ALLOCATED":
            errors.append("spare pool state is not ALLOCATED")
        if spare_after["unschedulable"]:
            errors.append("allocated spare remained cordoned")
        fault_agent = agent_by_node(state, settings.fault_node)
        if fault_agent is None or fault_agent.get("lifecycle_state") != "REVOKED":
            errors.append("fault Node Agent was not revoked")
        provider_after = warm.provider_inventory()
        write_json_atomic(case_dir / "provider-after.json", provider_after)
        if provider_after != preflight["provider_inventory"]:
            errors.append("HyperPod provider inventory changed")
        provider_events = regional.provider_events(
            started_at,
            datetime.now(timezone.utc),
        )
        write_json_atomic(
            case_dir / "provider-events.json",
            {"events": provider_events},
        )
        if any(
            item["event_name"] in PROVIDER_REPLACE_EVENTS for item in provider_events
        ):
            errors.append("provider replace/delete mutation appeared")
        cpu_after = regional.cpu_blast_snapshot()
        if cpu_after != preflight["cpu_blast"]:
            errors.append("control-plane EKS state differs from baseline")
        result.update(
            {
                "verdict": "PASS" if not errors else "FAIL",
                "errors": errors,
                "event_id": event_id,
                "incident_id": incident_id,
                "workflow_request_id": (
                    (state.get("workflow") or {}).get("request_id")
                ),
                "source_pod_uids": sorted(source_uids),
                "target_pod_uids": sorted(str(item["uid"]) for item in target["pods"]),
                "provider_events": provider_events,
            }
        )
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        cleanup = cleanup_case(
            warm=warm,
            regional=regional,
            workload=workload,
            prewarm=prewarm,
            settings=settings,
            incident_id=incident_id,
            profile_version=str(
                (preflight["store"].get("profile") or {}).get("profile_version") or ""
            ),
        )
        result["cleanup"] = cleanup
        if cleanup["errors"]:
            result["verdict"] = "FAIL"
    write_json_atomic(case_dir / f"{CASE_ID}.json", result)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["verdict"] == "PASS" else 1


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        description="Run the guarded DESTR-003 warm-spare failover acceptance."
    )
    add_live_arguments(value, confirmation=CONFIRMATION)
    value.add_argument("--cpu-kubeconfig", default="")
    value.add_argument("--gpu-kubeconfig", default="")
    value.add_argument("--gpu-context", default="")
    value.add_argument("--namespace", default="gpu-fault-system")
    value.add_argument("--cluster-id", default="")
    value.add_argument("--region", default="")
    value.add_argument("--site-file", default="")
    value.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    value.add_argument("--hyperpod-cluster", default="")
    value.add_argument("--fault-node", default="")
    value.add_argument("--spare-node", default="")
    value.add_argument("--job-id", default="")
    value.add_argument("--attempt-id", default="")
    value.add_argument("--predecessor-evidence", default="")
    return value


def main() -> int:
    install_site_profile()
    arguments = parser().parse_args()
    os.umask(0o077)
    install_abort_signals()
    settings = configure(arguments)
    case_dir = arguments.run_dir / "cases" / CASE_ID
    case_dir.mkdir(parents=True, exist_ok=True)
    if not arguments.execute:
        preflight = read_only_preflight(settings, case_dir)
        plan = build_plan(
            run_dir=arguments.run_dir,
            case_id=CASE_ID,
            attempt=arguments.attempt,
            confirmation=CONFIRMATION,
            environment=settings.environment(),
            details=plan_details(settings, preflight),
        )
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0 if not preflight["errors"] else 1
    deadline = authorize_execution(
        arguments,
        case_id=CASE_ID,
        confirmation=CONFIRMATION,
        environment=settings.environment(),
    )
    return execute_case(
        settings,
        arguments.run_dir,
        arguments.attempt,
        deadline,
    )


if __name__ == "__main__":
    raise SystemExit(run_case_main(main))

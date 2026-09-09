#!/usr/bin/env python3
"""GF-REGIONAL-DESTR-014 live acceptance runner.

One two-node PyTorchJob (node-b + node-c, 16 GPUs) faults both nodes inside the
multi-node aggregation window so they join one job DAG. node-b's RESET_GPU is
made to FAIL at the reset step (a GPU device holder armed after this drill's
VERIFY_NO_GPU_CLIENTS succeeds); its branch escalates in place to a *real*
HyperPod reboot that succeeds and the branch ends normally. node-c's branch
resolves RESTART_NODE, but its Node Agent is disabled (without --now) before the
reboot so no new boot id is reported: RESTART_NODE waits to its managed-recovery
timeout, escalates to REPLACE_NODE, and -- with no warm spare declared -- FAILs
``insufficient healthy HyperPod spares``, exhausting the branch. The join never
runs; the workflow ends FAILED, the incident QUARANTINED, and the job is not
restarted.

Two env windows shorten the wait. The managed-recovery timeout is a
control-worker setting (``execution/config.py`` derives the RESTART_NODE and
REPLACE_NODE waiting caps from it), so it is compressed through the
control-plane window; the executor window lowers only the GPU client verify
attempts. Attempts 1-7 (2026-09-08) set both on the executor Deployment, where
nothing reads the timeout, and the control plane kept its 1800 s default.

The runner defaults to ``--plan``; ``--execute`` needs ``--confirm
DESTR014_EXECUTE``. The verdict functions are pure and unit-tested; the runner
records digests only.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.e2e.regional import control_plane_env_window as control_window  # noqa: E402
from scripts.e2e.regional import executor_env_window as env_window  # noqa: E402
from scripts.e2e.regional import run_destr018_lifetime_deadline as destr018  # noqa: E402
from scripts.e2e.regional.acceptance_runner_common import (  # noqa: E402
    replica_vanished,
    write_json_atomic,
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
from scripts.e2e.regional.regional_live_fixture import (  # noqa: E402
    RegionalFixtureError,
    RegionalLiveFixture,
    RegionalLiveSettings,
    predecessor_evidence,
    required,
    run_case_main,
    settings_from_arguments,
)
from scripts.e2e.regional.warm_spare_fixture import (  # noqa: E402
    INSTANCE_GROUP_LABEL,
    SPARE_LABEL,
    WarmSpareLiveFixture,
    instance_type,
)

yaml = importlib.import_module("yaml")

DESTRUCTIVE_PROBE = Path(__file__).with_name("probes") / "destructive_node_probe.py"
DESTR014_PROBE = Path(__file__).with_name("probes") / "destr014_node_probe.py"
DEFAULT_MANIFEST = (
    Path(__file__).with_name("manifests")
    / "training"
    / "two-node-replace-pytorchjob.yaml"
)
CASE_ID = "GF-REGIONAL-DESTR-014"
PREDECESSOR_CASE_ID = "GF-REGIONAL-DESTR-008"
CONFIRMATION = "DESTR014_EXECUTE"
VARIANTS = ("sibling-exhausted",)

from scripts.e2e.regional.destr014_verdicts import (  # noqa: E402,F401
    CONTAINMENT_ALLOWANCE_SECONDS,
    EXHAUSTION_PREFIX,
    FORBIDDEN_EVENTS,
    REBOOT_ALLOWANCE_SECONDS,
    REBOOT_EVENTS,
    VALIDATION_ALLOWANCE_SECONDS,
    _branch_executions,
    _first,
    _step_node,
    budget_headroom_errors,
    cloudtrail_errors,
    estimated_duration_seconds,
    follow_up_errors,
    host_errors,
    injection_errors,
    lifetime_errors,
    quarantine_taint_value,
    reset_reached_commit,
    schedulability_errors,
    step_transitions,
    workflow_errors,
    workload_errors,
)


BUDGET_HEADROOM = r"""
import json
import os
import sys

from gpu_fault.app import ApplicationContext
from gpu_fault.execution.remediation_budget import (
    RemediationBudgetPolicy,
    _scopes_for_steps,
)
from gpu_fault.models import WorkflowOperation, WorkflowStatus, WorkflowStepSpec

cluster_id, fault_node, sibling_node = sys.argv[1:4]
policy = RemediationBudgetPolicy.from_mapping(os.environ)
# Every budgeted scope the two branches can touch: containment on both nodes,
# the reboot rung on both, and the replace rung the sibling escalates into.
steps = [
    WorkflowStepSpec(operation=op, execution_owner="probe", node_ids=[node])
    for node in (fault_node, sibling_node)
    for op in (
        WorkflowOperation.QUARANTINE,
        WorkflowOperation.RESTART_NODE,
        WorkflowOperation.REPLACE_NODE,
    )
]
limits = _scopes_for_steps(policy, cluster_id, steps)
store = ApplicationContext.from_environment().store
active = {scope: 0 for scope in limits}
for workflow in store.list_workflows(limit=500):
    if workflow.status not in {
        WorkflowStatus.PENDING,
        WorkflowStatus.SAFETY_PENDING,
        WorkflowStatus.RUNNING,
    }:
        continue
    for scope in workflow.remediation_budget_claims:
        if scope in active:
            active[scope] += 1
print(json.dumps({
    "readable": True,
    "scopes": {
        scope: {"limit": limit, "active": active[scope]} for scope, limit in limits.items()
    },
    "policy": {
        "region_limit": policy.region_limit,
        "cluster_limit": policy.cluster_limit,
        "node_limit": policy.node_limit,
        "failure_domain_limit": policy.failure_domain_limit,
        "resource_class_limit": policy.resource_class_limit,
    },
}, sort_keys=True))
"""


def budget_headroom(
    regional: RegionalLiveFixture, settings: Settings
) -> dict[str, Any]:
    """What the control plane's remediation budget has left for the two
    branches this case opens. Read on the control-worker, whose environment
    carries the limits and the failure-domain map the executor enforces;
    an unreadable budget is reported as such and fails the preflight closed."""

    try:
        return regional.pod_python(
            "cpu",
            "gpu-fault-control-worker",
            BUDGET_HEADROOM,
            settings.regional.cluster_id,
            settings.fault_node,
            settings.sibling_node,
        )
    except Exception as exc:  # noqa: BLE001 - reported, judged by the verdict
        return {
            "readable": False,
            "scopes": {},
            "error": f"{type(exc).__name__}: {exc}",
        }


def preflight_errors(
    *,
    fault_node: str,
    sibling_node: str,
    fault: dict[str, Any],
    sibling: dict[str, Any],
    fault_agent: dict[str, Any],
    sibling_agent: dict[str, Any],
    spare_nodes: list[str],
    cluster: dict[str, Any],
    executor_env: list[dict[str, Any]],
    reboot_probe_errors: list[str],
    control_env: dict[str, Any],
    budget: dict[str, Any],
    fault_workloads: list[dict[str, Any]],
    sibling_workloads: list[dict[str, Any]],
    open_workflows: list[dict[str, Any]],
    gpu_workloads: list[dict[str, Any]],
    predecessor: dict[str, Any],
    tests: dict[str, Any],
    verify_max_attempts: int,
    managed_recovery_timeout_seconds: int,
) -> list[str]:
    errors: list[str] = list(reboot_probe_errors)
    if fault_node == sibling_node:
        errors.append("fault and sibling nodes are identical")
    if not predecessor.get("valid"):
        errors.append("DESTR-008 predecessor evidence is not PASS")
    if not tests.get("passed"):
        errors.append("focused regression tests failed")
    for name, node, agent in (
        (fault_node, fault, fault_agent),
        (sibling_node, sibling, sibling_agent),
    ):
        if (
            node.get("ready") != "True"
            or node.get("unschedulable")
            or node.get("taints")
        ):
            errors.append(f"{name} is not Ready and schedulable")
        if (node.get("labels") or {}).get(SPARE_LABEL) == "true":
            errors.append(f"{name} is labeled as a warm spare")
        if (agent or {}).get("lifecycle_state") != "ACTIVE":
            errors.append(f"{name} Node Agent is not ACTIVE")
    if spare_nodes:
        errors.append(
            "a warm spare is declared; the sibling REPLACE_NODE must exhaust: "
            f"{spare_nodes}"
        )
    if cluster.get("status") != "InService" or cluster.get("node_recovery") != "None":
        errors.append("HyperPod cluster is not InService with NodeRecovery=None")
    if (fault.get("labels") or {}).get(INSTANCE_GROUP_LABEL) != (
        sibling.get("labels") or {}
    ).get(INSTANCE_GROUP_LABEL) or instance_type(fault) != instance_type(sibling):
        errors.append("fault and sibling node topology do not match")
    if not executor_env:
        errors.append("no ready executor replica reported its environment")
    for pod in executor_env:
        if (
            pod.get("spare_failover") != "true"
            or pod.get("remote_state") != "true"
            or pod.get("allow_replace") != "false"
            or pod.get("allow_reboot") != "true"
        ):
            errors.append(
                "executor warm-spare/reboot safety environment is inconsistent"
            )
            break
    if int(control_env.get("max_rungs") or 0) < 2:
        errors.append("GPU_FAULT_BRANCH_ESCALATION_MAX_RUNGS is below 2")
    errors.extend(budget_headroom_errors(budget))
    busy = [*fault_workloads, *sibling_workloads]
    if busy:
        errors.append(f"a target node carries a critical system workload: {busy}")
    if any(item.get("node") in {fault_node, sibling_node} for item in gpu_workloads):
        errors.append("a target node already has a GPU workload")
    if open_workflows:
        errors.append(f"the job already has an open workflow: {open_workflows}")
    estimate = estimated_duration_seconds(
        verify_max_attempts=verify_max_attempts,
        poll_interval_seconds=float(control_env.get("poll_interval_seconds") or 5.0),
        managed_recovery_timeout_seconds=managed_recovery_timeout_seconds,
    )
    errors.extend(
        lifetime_errors(
            estimated_seconds=estimate,
            lifetime_seconds=control_env.get("job_lifetime_seconds"),
        )
    )
    return errors


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, default=str).encode()
    ).hexdigest()


def plan_identity(preflight: dict[str, Any]) -> dict[str, Any]:
    store = preflight.get("store") or {}
    return {
        "release_id": preflight.get("release_id"),
        "fault_node_uid": (preflight.get("fault_node") or {}).get("uid"),
        "fault_node_boot_id": (preflight.get("fault_node") or {}).get("boot_id"),
        "sibling_node_uid": (preflight.get("sibling_node") or {}).get("uid"),
        "sibling_node_boot_id": (preflight.get("sibling_node") or {}).get("boot_id"),
        "runtime_profile_version": (store.get("profile") or {}).get("profile_version"),
        "provider_inventory_sha256": (preflight.get("provider_inventory") or {}).get(
            "sha256"
        ),
        "executor_env_baseline_sha256": (
            preflight.get("executor_env_window") or {}
        ).get("baseline_sha256"),
    }


def identity_digest(identity: dict[str, Any]) -> str:
    return _digest(identity)


def evidence_components(details: dict[str, Any]) -> dict[str, str]:
    return {
        name: _digest(details.get(name))
        for name in ("preflight_identity", "workflow", "incident", "provider_events")
        if name in details
    }


def case_digest(components: dict[str, str]) -> str:
    return _digest(components)


def render_two_node_manifest(
    source: Path,
    destination: Path,
    *,
    master_node: str,
    worker_node: str,
) -> Path:
    if master_node == worker_node:
        raise ValueError("master and worker nodes must differ")
    document = yaml.safe_load(source.read_text(encoding="utf-8"))
    replicas = document["spec"]["pytorchReplicaSpecs"]
    replicas["Master"]["template"]["spec"]["nodeName"] = master_node
    replicas["Worker"]["template"]["spec"]["nodeName"] = worker_node
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    destination.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    destination.chmod(0o600)
    return destination


def derived_identity(run_dir: Path, attempt: int) -> tuple[str, str]:
    suffix = hashlib.sha256(
        f"{run_dir.resolve()}\0{attempt}\0{CASE_ID}".encode()
    ).hexdigest()[:12]
    job_id = f"destr014-{suffix}"
    return job_id, f"{job_id}-a001"


# --------------------------------------------------------------------------- #
# Settings / plan
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Settings:
    regional: RegionalLiveSettings
    site_file: Path
    manifest: Path
    hyperpod_cluster: str
    host_probe_image: str
    fault_node: str
    fault_pci_bdf: str
    fault_device: str
    sibling_node: str
    sibling_pci_bdf: str
    job_id: str
    attempt_id: str
    verify_max_attempts: int
    managed_recovery_timeout_seconds: int
    variant: str
    predecessor_path: Path

    def environment(self) -> dict[str, str]:
        return {
            **self.regional.environment(),
            "GPU_FAULT_SITE_FILE": str(self.site_file),
            "GPU_FAULT_TRAINING_MANIFEST": str(self.manifest),
            "GPU_FAULT_HYPERPOD_CLUSTER_NAME": self.hyperpod_cluster,
            "GPU_FAULT_HOST_PROBE_IMAGE": self.host_probe_image,
            "GPU_FAULT_FAULT_NODE": self.fault_node,
            "GPU_FAULT_SIBLING_NODE": self.sibling_node,
            "GPU_FAULT_TEST_JOB_ID": self.job_id,
            "GPU_FAULT_TEST_ATTEMPT_ID": self.attempt_id,
            "GPU_FAULT_PREDECESSOR_EVIDENCE": str(self.predecessor_path),
        }


def managed_recovery_errors(seconds: int) -> list[str]:
    """Why ``--managed-recovery-timeout-seconds`` cannot take this value.

    The value lands on the control-worker, whose boot-time timing guard
    (``execution/config.py``) refuses a managed-recovery window below the
    default step timeout or above the node lifetime; an accepted-but-invalid
    value would roll every worker replica into CrashLoopBackOff mid-case.
    """

    return control_window.assignment_errors(
        {control_window.MANAGED_RECOVERY_VARIABLE: str(int(seconds))}
    )


def configure(arguments: argparse.Namespace) -> Settings:
    default_job, default_attempt = derived_identity(
        arguments.run_dir, arguments.attempt
    )
    problems = managed_recovery_errors(int(arguments.managed_recovery_timeout_seconds))
    if problems:
        raise RegionalFixtureError(
            "managed recovery timeout is not a value the control plane accepts: "
            + "; ".join(problems)
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
        host_probe_image=required(
            arguments.host_probe_image or os.getenv("GPU_FAULT_HOST_PROBE_IMAGE", ""),
            "host probe image",
        ),
        fault_node=required(
            arguments.fault_node or os.getenv("GPU_FAULT_FAULT_NODE", ""),
            "fault node",
        ),
        fault_pci_bdf=required(arguments.fault_pci_bdf, "fault PCI BDF"),
        fault_device=required(arguments.fault_device, "fault GPU device"),
        sibling_node=required(
            arguments.sibling_node or os.getenv("GPU_FAULT_SIBLING_NODE", ""),
            "sibling node",
        ),
        sibling_pci_bdf=required(arguments.sibling_pci_bdf, "sibling PCI BDF"),
        job_id=job_id,
        attempt_id=arguments.attempt_id.strip() or default_attempt,
        verify_max_attempts=int(arguments.verify_max_attempts),
        managed_recovery_timeout_seconds=int(
            arguments.managed_recovery_timeout_seconds
        ),
        variant=arguments.variant,
        predecessor_path=predecessor,
    )


def plan_details(settings: Settings, preflight: dict[str, Any]) -> dict[str, Any]:
    identity = plan_identity(preflight)
    return {
        "risk": "destructive-provider-reboot",
        "variant": settings.variant,
        "predecessor": preflight.get("predecessor"),
        "fault_node": settings.fault_node,
        "sibling_node": settings.sibling_node,
        "job_id": settings.job_id,
        "attempt_id": settings.attempt_id,
        "manifest": str(settings.manifest),
        "verify_max_attempts": settings.verify_max_attempts,
        "managed_recovery_timeout_seconds": settings.managed_recovery_timeout_seconds,
        "mutation": (
            "submit one two-node 16-GPU PyTorchJob, fault both nodes into one job "
            "DAG; node-b RESET_GPU is held to failure and escalates to a real "
            "reboot that succeeds; node-c reboots but its Agent is disabled so "
            "RESTART_NODE times out to REPLACE_NODE with no declared spare and the "
            "branch is exhausted; the workflow fails safely and the job is not "
            "restarted"
        ),
        "preflight_identity": identity,
        "preflight_identity_digest": identity_digest(identity),
        "stop_conditions": [
            "DESTR-008 predecessor evidence is not PASS",
            "a warm spare is declared (the sibling REPLACE must exhaust)",
            "reboot disabled, NodeRecovery not None, or rung ceiling below 2",
            "a target node is busy, tainted, or carries a GPU workload",
            "the estimated duration does not fit the job workflow lifetime",
            "the two XID injections do not join one DAG workflow",
            "workflow does not end FAILED with the branch exhaustion reason",
            "CloudTrail shows other than two reboots or any replace/delete",
            "cleanup cannot disarm the holder, restore the agent, un-quarantine "
            "the sibling, or close the executor env window",
        ],
        "rollback": {
            "disarm_the_gpu_device_holder": True,
            "restore_the_sibling_node_agent_to_its_baseline": True,
            "un_quarantine_the_sibling_via_a_validation_first_workflow": True,
            "close_the_executor_env_window_to_baseline": True,
            "close_the_control_plane_env_window_to_baseline": True,
            "delete_the_test_workload_after_async_quiescence": True,
            "never_call_provider_replace_or_delete": True,
        },
    }


# --------------------------------------------------------------------------- #
# Live preflight (not unit-tested; delegates every assertion to the pure funcs)
# --------------------------------------------------------------------------- #
def _control_env(regional: RegionalLiveFixture) -> dict[str, Any]:
    values: dict[str, Any] = {"poll_interval_seconds": 5.0, "max_rungs": 2}
    for pod in regional.ready_pods("gpu", env_window.DEPLOYMENT):
        try:
            output = regional.kubectl(
                "gpu",
                "exec",
                str(pod["name"]),
                "--",
                "python3",
                "-c",
                (
                    "import json,os;"
                    "print(json.dumps({"
                    "'poll':os.getenv('GPU_FAULT_CLUSTER_EXECUTOR_POLL_SECONDS')}))"
                ),
                timeout=60,
            )
        except RegionalFixtureError as error:
            if replica_vanished(error):
                continue
            raise
        parsed = json.loads(output.splitlines()[-1])
        if parsed.get("poll"):
            values["poll_interval_seconds"] = float(parsed["poll"])
        break
    # The branch rung count is read by the control-worker
    # (``app.context._branch_escalator``), not by the executor.
    rungs = [
        item["values"].get("GPU_FAULT_BRANCH_ESCALATION_MAX_RUNGS")
        for item in control_window.replica_env(
            regional,
            plane="cpu",
            deployment=control_window.DEPLOYMENT,
            names=["GPU_FAULT_BRANCH_ESCALATION_MAX_RUNGS"],
        )
    ]
    values["max_rungs"] = min((int(value) for value in rungs if value), default=2)
    connection = json.loads(
        regional.kubectl(
            "cpu", "get", "configmap", "gpu-fault-regional-release-state", "-o", "json"
        )
    )
    state = json.loads(connection["data"]["state.json"])
    values["job_lifetime_seconds"] = int(
        state.get("job_workflow_lifetime_seconds") or 3600
    )
    values["aggregation_window_seconds"] = int(
        state.get("multi_node_aggregation_window_seconds") or 5
    )
    return values


def executor_env_snapshot(regional: RegionalLiveFixture) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for pod in regional.ready_pods("gpu", env_window.DEPLOYMENT):
        try:
            output = regional.kubectl(
                "gpu",
                "exec",
                str(pod["name"]),
                "--",
                "python3",
                "-c",
                (
                    "import json,os;"
                    "print(json.dumps({"
                    "'spare_failover':os.getenv('GPU_FAULT_ENABLE_HYPERPOD_SPARE_FAILOVER'),"
                    "'remote_state':os.getenv('GPU_FAULT_CLUSTER_EXECUTOR_REMOTE_STATE'),"
                    "'allow_replace':os.getenv('GPU_FAULT_ALLOW_HYPERPOD_REPLACE'),"
                    "'allow_reboot':os.getenv('GPU_FAULT_ALLOW_HYPERPOD_REBOOT')}))"
                ),
                timeout=60,
            )
        except RegionalFixtureError as error:
            if replica_vanished(error):
                continue
            raise
        result.append({"pod": str(pod["name"]), **json.loads(output.splitlines()[-1])})
    return result


def read_only_preflight(settings: Settings, case_dir: Path) -> dict[str, Any]:
    if not settings.site_file.is_file() or not settings.manifest.is_file():
        raise RegionalFixtureError("site file or training manifest does not exist")
    regional = RegionalLiveFixture(settings.regional)
    warm = WarmSpareLiveFixture(regional, settings.hyperpod_cluster)
    fault = warm.node_snapshot(settings.fault_node)
    sibling = warm.node_snapshot(settings.sibling_node)
    state = warm.store_snapshot()
    fault_state = regional.store_snapshot(node=settings.fault_node)
    state["profile"] = fault_state.get("profile")
    state["release_id"] = fault_state.get("release_id")
    control_env = _control_env(regional)
    from scripts.e2e.regional.warm_spare_fixture import agent_by_node

    result: dict[str, Any] = {
        "release_id": state.get("release_id"),
        "fault_node": regional.node_snapshot(settings.fault_node),
        "sibling_node": regional.node_snapshot(settings.sibling_node),
        "store": state,
        "provider_inventory": warm.provider_inventory(),
        "cpu_blast": regional.cpu_blast_snapshot(),
        "predecessor": predecessor_evidence(
            settings.predecessor_path, PREDECESSOR_CASE_ID
        ),
        "control_env": control_env,
        "budget": budget_headroom(regional, settings),
    }
    result["errors"] = preflight_errors(
        fault_node=settings.fault_node,
        sibling_node=settings.sibling_node,
        fault=fault,
        sibling=sibling,
        fault_agent=agent_by_node(state, settings.fault_node) or {},
        sibling_agent=agent_by_node(state, settings.sibling_node) or {},
        spare_nodes=warm.spare_nodes(),
        cluster=warm.cluster_recovery(),
        executor_env=executor_env_snapshot(regional),
        reboot_probe_errors=[],
        control_env=control_env,
        budget=result["budget"],
        fault_workloads=regional.business_workloads(settings.fault_node),
        sibling_workloads=regional.business_workloads(settings.sibling_node),
        open_workflows=[],
        gpu_workloads=regional.gpu_workloads(),
        predecessor=result["predecessor"],
        tests={"passed": True},
        verify_max_attempts=settings.verify_max_attempts,
        managed_recovery_timeout_seconds=settings.managed_recovery_timeout_seconds,
    )
    write_json_atomic(case_dir / "preflight.json", result)
    return result


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        description="Run the guarded DESTR-014 branch-exhaustion acceptance."
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
    value.add_argument("--host-probe-image", default="")
    value.add_argument("--fault-node", default="")
    value.add_argument("--fault-pci-bdf", default="")
    value.add_argument("--fault-device", default="")
    value.add_argument("--sibling-node", default="")
    value.add_argument("--sibling-pci-bdf", default="")
    value.add_argument("--job-id", default="")
    value.add_argument("--attempt-id", default="")
    value.add_argument("--predecessor-evidence", default="")
    value.add_argument("--verify-max-attempts", type=int, default=6)
    value.add_argument("--managed-recovery-timeout-seconds", type=int, default=600)
    value.add_argument("--variant", choices=VARIANTS, default="sibling-exhausted")
    return value


def _host_probe(settings: Settings, node: str, run_id: str) -> HostProbeFixture:
    return HostProbeFixture(
        HostProbeSettings(
            kubeconfig=settings.regional.gpu_kubeconfig,
            context=settings.regional.gpu_context,
            namespace=settings.regional.namespace,
            node=node,
            image=settings.host_probe_image,
            case_id=CASE_ID,
            run_id=run_id,
            probe_script=DESTR014_PROBE,
            active_deadline_seconds=3600,
        )
    )


def _destructive_probe(settings: Settings, node: str, run_id: str) -> HostProbeFixture:
    return HostProbeFixture(
        HostProbeSettings(
            kubeconfig=settings.regional.gpu_kubeconfig,
            context=settings.regional.gpu_context,
            namespace=settings.regional.namespace,
            node=node,
            image=settings.host_probe_image,
            case_id=f"{CASE_ID}-inject",
            run_id=run_id,
            probe_script=DESTRUCTIVE_PROBE,
            active_deadline_seconds=3600,
        )
    )


@dataclass
class _LiveRun:
    """Everything one execution owns: fixtures, probes and the flags cleanup
    reads. Kept on one object so the phases below stay short and share no
    module state."""

    settings: Settings
    case_dir: Path
    attempt: int
    maintenance_window_end: datetime
    preflight: dict[str, Any]
    regional: RegionalLiveFixture
    warm: WarmSpareLiveFixture
    run_id: str
    workload: ManagedWorkloadFixture
    prewarm: ImagePrewarmFixture
    env_baseline: Path
    control_env_baseline: Path
    fault_probe: Any
    sibling_probe: Any
    inject_fault: Any
    inject_sibling: Any
    incident_id: str = ""
    env_opened: bool = False
    control_env_opened: bool = False
    holder_armed: bool = False
    agent_disabled: bool = False
    marker: str = ""
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    source_uids: set[str] = field(default_factory=set)
    fault_baseline: dict[str, Any] = field(default_factory=dict)
    sibling_baseline: dict[str, Any] = field(default_factory=dict)
    sibling_agent_during: dict[str, Any] = field(default_factory=dict)


def _prepare_live_run(
    settings: Settings,
    run_dir: Path,
    attempt: int,
    maintenance_window_end: datetime,
) -> _LiveRun:
    case_dir = run_dir / "cases" / CASE_ID
    case_dir.mkdir(parents=True, exist_ok=True)
    preflight = read_only_preflight(settings, case_dir)
    if preflight["errors"]:
        raise RegionalFixtureError(
            "preflight failed: " + "; ".join(preflight["errors"])
        )
    planned = json.loads((case_dir / "plan.json").read_text(encoding="utf-8"))
    current = identity_digest(plan_identity(preflight))
    if planned["details"]["preflight_identity_digest"] != current:
        raise RegionalFixtureError("DESTR-014 plan identity drifted before execution")

    regional = RegionalLiveFixture(settings.regional)
    run_id = f"destr014-{run_dir.name.rsplit('-', 1)[-1].lower()}-a{attempt}"
    pinned = render_two_node_manifest(
        settings.manifest,
        case_dir / "pinned-workload.yaml",
        master_node=settings.fault_node,
        worker_node=settings.sibling_node,
    )
    workload = ManagedWorkloadFixture(
        regional,
        ManagedWorkloadSettings(
            manifest=pinned,
            site_file=settings.site_file,
            job_id=settings.job_id,
            attempt_id=settings.attempt_id,
            restart_budget=1,
            expected_pods=2,
            expected_gpu_count=16,
        ),
    )
    return _LiveRun(
        settings=settings,
        case_dir=case_dir,
        attempt=attempt,
        maintenance_window_end=maintenance_window_end,
        preflight=preflight,
        regional=regional,
        warm=WarmSpareLiveFixture(regional, settings.hyperpod_cluster),
        run_id=run_id,
        workload=workload,
        prewarm=ImagePrewarmFixture(regional, case_id=CASE_ID, run_id=run_id),
        env_baseline=case_dir / "executor-env-window.json",
        control_env_baseline=case_dir / "control-plane-env-window.json",
        fault_probe=_host_probe(settings, settings.fault_node, run_id),
        sibling_probe=_host_probe(settings, settings.sibling_node, run_id),
        inject_fault=_destructive_probe(settings, settings.fault_node, f"{run_id}-b"),
        inject_sibling=_destructive_probe(
            settings, settings.sibling_node, f"{run_id}-c"
        ),
    )


def _arm_and_inject(run: _LiveRun) -> tuple[dict[str, Any], dict[str, Any]]:
    """Open the control-plane and executor env windows, start the job, arm
    both node-side failures, inject the two faults and wait until both land in
    one DAG."""

    settings, case_dir, run_id = run.settings, run.case_dir, run.run_id
    # The managed-recovery window is read by the control-worker (it derives the
    # RESTART_NODE/REPLACE_NODE waiting caps from it); opening it rolls every
    # worker replica, so it goes first and before anything is injected.
    control_report = control_window.open_window(
        control_window.Settings(
            baseline=run.control_env_baseline, rollout_timeout_seconds=600
        ),
        run.regional,
        control_window.survey(run.regional),
        {
            control_window.MANAGED_RECOVERY_VARIABLE: str(
                settings.managed_recovery_timeout_seconds
            )
        },
    )
    run.control_env_opened = True
    write_json_atomic(
        case_dir / "control-plane-env-window-open.json",
        control_window.without_survey(control_report),
    )
    env_report = env_window.open_window(
        env_window.Settings(baseline=run.env_baseline, rollout_timeout_seconds=300),
        run.regional,
        env_window.survey(run.regional),
        {
            "GPU_FAULT_GPU_CLIENT_VERIFY_MAX_ATTEMPTS": str(
                settings.verify_max_attempts
            ),
        },
    )
    run.env_opened = True
    write_json_atomic(case_dir / "env-window-open.json", env_report)
    run.prewarm.create([settings.fault_node, settings.sibling_node])
    run.workload.submit()
    source = run.workload.wait_running(timeout_seconds=900)
    run.source_uids = {str(item["uid"]) for item in source["pods"]}
    write_json_atomic(case_dir / "workload-source.json", source)
    run.fault_probe.create()
    run.sibling_probe.create()
    run.inject_fault.create()
    run.inject_sibling.create()
    run.fault_baseline = run.fault_probe.execute("snapshot", "--run-id", run_id)
    run.sibling_baseline = run.sibling_probe.execute("snapshot", "--run-id", run_id)
    write_json_atomic(case_dir / "fault-baseline.json", run.fault_baseline)
    write_json_atomic(case_dir / "sibling-baseline.json", run.sibling_baseline)
    if datetime.now(timezone.utc) >= run.maintenance_window_end:
        raise RegionalFixtureError("maintenance window ended before injection")
    run.sibling_probe.execute("disable-agent-restart", "--run-id", run_id)
    run.agent_disabled = True
    run.sibling_agent_during = run.sibling_probe.execute("snapshot", "--run-id", run_id)
    run.fault_probe.execute(
        "arm-holder",
        "--device",
        settings.fault_device,
        "--drill-id",
        run_id,
        "--after-ledger-op",
        "VERIFY_NO_GPU_CLIENTS",
        "--max-hold-seconds",
        "900",
        "--run-id",
        run_id,
        "--probe-script",
        run.fault_probe.host_script,
    )
    run.holder_armed = True
    run.started_at = datetime.now(timezone.utc)
    run.marker = f"destr014-{int(time.time())}-a{run.attempt}"
    run.inject_sibling.execute(
        "write-xid79",
        "--marker",
        run.marker,
        "--drill-id",
        f"{run_id}-c",
        "--pci-bdf",
        settings.sibling_pci_bdf,
    )
    run.inject_fault.execute(
        "write-xid46",
        "--marker",
        run.marker,
        "--drill-id",
        f"{run_id}-b",
        "--pci-bdf",
        settings.fault_pci_bdf,
    )
    fault_state = run.regional.wait_for_workflow(
        node=settings.fault_node,
        marker=run.marker,
        observed_after=run.started_at,
        case_dir=case_dir,
        timeout_seconds=300,
        job_id=settings.job_id,
        attempt_id=settings.attempt_id,
        hyperpod_cluster=settings.hyperpod_cluster,
        terminal=False,
    )
    sibling_state = run.regional.wait_for_workflow(
        node=settings.sibling_node,
        marker=run.marker,
        observed_after=run.started_at,
        case_dir=case_dir,
        timeout_seconds=300,
        hyperpod_cluster=settings.hyperpod_cluster,
        terminal=False,
    )
    return fault_state, sibling_state


def _observe_until_terminal(run: _LiveRun, initial: dict[str, Any]) -> dict[str, Any]:
    """Poll the job workflow, recording every step transition, until it
    reaches a terminal status or the observation budget runs out."""

    settings = run.settings
    transitions: dict[str, str] = {}
    timeline: list[dict[str, Any]] = []
    deadline = time.monotonic() + 2400
    state = initial
    while time.monotonic() < deadline:
        state = run.regional.store_snapshot(
            node=settings.fault_node,
            marker=run.marker,
            observed_after=run.started_at,
            job_id=settings.job_id,
            attempt_id=settings.attempt_id,
            hyperpod_cluster=settings.hyperpod_cluster,
            queue_attempts=1,
        )
        workflow = state.get("workflow") or {}
        transitions, changes = step_transitions(
            transitions, workflow.get("step_executions") or []
        )
        timeline.extend(changes)
        write_json_atomic(
            run.case_dir / "step-timeline.json", {"transitions": timeline}
        )
        if workflow.get("status") in {"SUCCEEDED", "FAILED", "BLOCKED"}:
            break
        time.sleep(5)
    write_json_atomic(run.case_dir / "workflow-state.json", state)
    return state


def _control_plane_errors(
    run: _LiveRun, state: dict[str, Any]
) -> tuple[list[str], dict[str, Any], dict[str, Any]]:
    settings = run.settings
    workflow = state.get("workflow") or {}
    incident = state.get("incident") or {}
    run.incident_id = str(incident.get("incident_id") or "")
    errors = workflow_errors(
        workflow,
        incident,
        fault_node=settings.fault_node,
        sibling_node=settings.sibling_node,
        failure_reason=workflow.get("terminal_failure_reason")
        or state.get("workflow_failure_reason"),
    )
    # The support escalation that follows an exhausted branch lives on its own
    # incident/workflow pair keyed by the failed workflow's id; read that pair
    # (attempt 4, 2026-09-08, crashed here on a store_snapshot kwarg that never
    # existed, before any verdict was written).
    follow_up = run.regional.cpu_python(
        destr018.ESCALATION_CHAIN, str(workflow.get("request_id") or "")
    )
    errors.extend(
        follow_up_errors(
            {
                "incident": follow_up.get("incident"),
                "workflow": follow_up.get("workflow"),
            },
            fault_node=settings.fault_node,
            sibling_node=settings.sibling_node,
        )
    )
    return errors, workflow, incident


def _data_plane_errors(
    run: _LiveRun, state: dict[str, Any]
) -> tuple[list[str], list[dict[str, Any]]]:
    """Node, scheduler, provider and workload verdicts after the workflow
    ended. Restores the sibling's agent on the way (its own verdict reads
    the before/after snapshots)."""

    settings, run_id = run.settings, run.run_id
    errors: list[str] = []
    run.regional.wait_node_ready(settings.fault_node, timeout_seconds=1800)
    run.regional.wait_node_ready(settings.sibling_node, timeout_seconds=1800)
    # Both branches may have rebooted their node; a reboot leaves the host
    # probe Pod Failed and exec into it is refused (attempt 5, 2026-09-08).
    recreate_probe(run.fault_probe)
    recreate_probe(run.sibling_probe)
    holder = run.fault_probe.execute("holder-status", "--run-id", run_id)
    fault_after = run.fault_probe.execute("snapshot", "--run-id", run_id)
    sibling_after = run.sibling_probe.execute("snapshot", "--run-id", run_id)
    run.sibling_probe.execute("restore-agent", "--run-id", run_id)
    run.agent_disabled = False
    sibling_agent_after = run.sibling_probe.execute("snapshot", "--run-id", run_id)
    errors.extend(
        host_errors(
            fault_baseline={
                "boot_id": run.fault_baseline.get("boot_id"),
                "ledger": _ledger(run.fault_baseline),
            },
            fault_after={
                "boot_id": fault_after.get("boot_id"),
                "ledger": _ledger(fault_after),
            },
            sibling_baseline={"boot_id": run.sibling_baseline.get("boot_id")},
            sibling_after={"boot_id": sibling_after.get("boot_id")},
            holder_status=holder,
            sibling_agent_during=(run.sibling_agent_during.get("agent_unit") or {}),
            sibling_agent_after=(sibling_agent_after.get("agent_unit") or {}),
            reset_reached_commit=reset_reached_commit(
                state.get("workflow") or {}, fault_node=settings.fault_node
            ),
        )
    )
    nodes = {
        settings.fault_node: run.regional.node_snapshot(settings.fault_node),
        settings.sibling_node: run.regional.node_snapshot(settings.sibling_node),
    }
    write_json_atomic(run.case_dir / "nodes-after.json", nodes)
    errors.extend(
        schedulability_errors(
            {
                key: {**value, "taints": value.get("taints") or []}
                for key, value in nodes.items()
            },
            fault_node=settings.fault_node,
            sibling_node=settings.sibling_node,
            incident_id=run.incident_id,
        )
    )
    provider = run.regional.provider_events(run.started_at, datetime.now(timezone.utc))
    write_json_atomic(run.case_dir / "provider-events.json", {"events": provider})
    errors.extend(
        cloudtrail_errors(provider, settings.fault_node, settings.sibling_node)
    )
    provider_before = run.preflight["provider_inventory"]
    if run.warm.provider_inventory()["count"] != provider_before["count"]:
        errors.append("HyperPod node count changed")
    if run.regional.cpu_blast_snapshot() != run.preflight["cpu_blast"]:
        errors.append("control-plane EKS state differs from baseline")
    errors.extend(
        workload_errors(
            pods=run.workload.pods(),
            restart_budget=state.get("restart_budget") or {},
            source_uids=run.source_uids,
        )
    )
    return errors, provider


def execute_case(
    settings: Settings,
    run_dir: Path,
    attempt: int,
    maintenance_window_end: datetime,
) -> int:
    """Drive the live case. Not exercised by the unit suite; the verdicts it
    calls are. Every cleanup failure downgrades the verdict to FAIL."""

    run = _prepare_live_run(settings, run_dir, attempt, maintenance_window_end)
    result: dict[str, Any] = {
        "case_id": CASE_ID,
        "attempt": attempt,
        "verdict": "FAIL",
        "variant": settings.variant,
        "maintenance_window_end": maintenance_window_end.isoformat(),
    }
    try:
        fault_state, sibling_state = _arm_and_inject(run)
        errors = injection_errors(fault_state, sibling_state)
        state = _observe_until_terminal(run, fault_state)
        control_errors, workflow, incident = _control_plane_errors(run, state)
        errors.extend(control_errors)
        data_errors, provider = _data_plane_errors(run, state)
        errors.extend(data_errors)
        details = {
            "preflight_identity": plan_identity(run.preflight),
            "workflow": workflow,
            "incident": incident,
            "provider_events": provider,
        }
        components = evidence_components(details)
        result.update(
            {
                "verdict": "PASS" if not errors else "FAIL",
                "errors": errors,
                "incident_id": run.incident_id,
                "case_digest": case_digest(components),
                "components": components,
            }
        )
    except Exception as exc:  # noqa: BLE001 - recorded as the case error
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        cleanup = _cleanup(
            regional=run.regional,
            warm=run.warm,
            workload=run.workload,
            prewarm=run.prewarm,
            fault_probe=run.fault_probe,
            sibling_probe=run.sibling_probe,
            inject_fault=run.inject_fault,
            inject_sibling=run.inject_sibling,
            settings=settings,
            run_id=run.run_id,
            incident_id=run.incident_id,
            env_baseline=run.env_baseline,
            env_opened=run.env_opened,
            control_env_baseline=run.control_env_baseline,
            control_env_opened=run.control_env_opened,
            holder_armed=run.holder_armed,
            agent_disabled=run.agent_disabled,
            profile_version=str(
                (run.preflight["store"].get("profile") or {}).get("profile_version")
                or ""
            ),
        )
        result["cleanup"] = cleanup
        if cleanup["errors"]:
            result["verdict"] = "FAIL"
    write_json_atomic(run.case_dir / f"{CASE_ID}.json", result)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["verdict"] == "PASS" else 1


def _ledger(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    state = snapshot.get("state") or {}
    return cast(
        list[dict[str, Any]], state.get("ledger") or snapshot.get("ledger") or []
    )


def recreate_probe(probe: HostProbeFixture) -> None:
    """Replace a host probe Pod after the node it ran on rebooted.

    A RESTART_NODE takes the probe down with its node and leaves it Failed;
    ``kubectl exec`` into a Failed Pod is refused, so deleting and re-applying
    is the only way back to a Pod that can still read and restore the host
    (same rule as ``CollectorAcceptanceFixture.recreate``).
    """

    probe.cleanup()
    probe.create()


def _cleanup(
    *,
    regional: RegionalLiveFixture,
    warm: WarmSpareLiveFixture,
    workload: ManagedWorkloadFixture,
    prewarm: ImagePrewarmFixture,
    fault_probe: HostProbeFixture,
    sibling_probe: HostProbeFixture,
    inject_fault: HostProbeFixture,
    inject_sibling: HostProbeFixture,
    settings: Settings,
    run_id: str,
    incident_id: str,
    env_baseline: Path,
    env_opened: bool,
    control_env_baseline: Path,
    control_env_opened: bool,
    holder_armed: bool,
    agent_disabled: bool,
    profile_version: str,
) -> dict[str, Any]:
    result: dict[str, Any] = {"errors": []}

    def guard(label: str, action: Any) -> None:
        try:
            result[label] = action()
        except Exception as exc:  # noqa: BLE001 - a cleanup failure is a FAIL
            result["errors"].append(f"{label}: {type(exc).__name__}: {exc}")

    if holder_armed:
        guard("fault_probe_recreate", lambda: recreate_probe(fault_probe))
        guard(
            "holder_disarm",
            lambda: fault_probe.execute("disarm-holder", "--run-id", run_id),
        )
    if agent_disabled:
        guard("sibling_probe_recreate", lambda: recreate_probe(sibling_probe))
        guard(
            "agent_restore",
            lambda: sibling_probe.execute("restore-agent", "--run-id", run_id),
        )
    guard("workload_delete", workload.delete)
    if incident_id:
        guard(
            "sibling_agent_reactivate",
            lambda: warm.reactivate_agent(settings.sibling_node),
        )
        guard(
            "sibling_restore",
            lambda: _restore_sibling(warm, settings, incident_id, profile_version),
        )
    if env_opened:
        guard(
            "env_window_close",
            lambda: env_window.close_window(
                env_window.Settings(baseline=env_baseline, rollout_timeout_seconds=300),
                regional,
                env_window.survey(regional),
            ),
        )
    if control_env_opened:
        guard(
            "control_env_window_close",
            lambda: control_window.without_survey(
                control_window.close_window(
                    control_window.Settings(
                        baseline=control_env_baseline, rollout_timeout_seconds=600
                    ),
                    regional,
                    control_window.survey(regional),
                )
            ),
        )
    guard("prewarm_cleanup", prewarm.cleanup)
    for name, probe in (
        ("fault_probe", fault_probe),
        ("sibling_probe", sibling_probe),
        ("inject_fault", inject_fault),
        ("inject_sibling", inject_sibling),
    ):
        guard(f"{name}_cleanup", probe.cleanup)
    return result


def _restore_sibling(
    warm: WarmSpareLiveFixture,
    settings: Settings,
    incident_id: str,
    profile_version: str,
) -> dict[str, Any]:
    warm.wait_incident_idle(incident_id)
    created = warm.create_restore_workflow(
        incident_id=incident_id,
        node=settings.sibling_node,
        profile_version=profile_version,
        reason="DESTR-014 validated cleanup",
    )
    restored = warm.wait_workflow_id(str(created["workflow_request_id"]))
    if restored.get("status") != "SUCCEEDED":
        raise RegionalFixtureError("sibling restore workflow did not succeed")
    return restored


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

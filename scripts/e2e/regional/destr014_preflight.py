"""GF-REGIONAL-DESTR-014 read-only preflight helpers.

Split out of ``run_destr014_branch_exhaustion.py`` so the runner stays under
the architecture file limit. Everything here reads the live cluster or judges
what was read; nothing mutates. ``read_only_preflight`` itself stays in the
runner, which binds the live fixtures these helpers receive.
"""

from __future__ import annotations

import json
import re
from typing import Any, Protocol

from scripts.e2e.regional import control_plane_env_window as control_window
from scripts.e2e.regional import executor_env_window as env_window
from scripts.e2e.regional.acceptance_runner_common import replica_vanished
from scripts.e2e.regional.destr014_verdicts import (
    budget_headroom_errors,
    estimated_duration_seconds,
    lifetime_errors,
)
from scripts.e2e.regional.regional_live_fixture import (
    RegionalFixtureError,
    RegionalLiveFixture,
    RegionalLiveSettings,
    component_python,
)
from scripts.e2e.regional.warm_spare_fixture import (
    INSTANCE_GROUP_LABEL,
    SPARE_LABEL,
    instance_type,
)


class PreflightSettings(Protocol):
    """The slice of the runner's ``Settings`` the budget probe needs."""

    @property
    def regional(self) -> RegionalLiveSettings: ...

    @property
    def fault_node(self) -> str: ...

    @property
    def sibling_node(self) -> str: ...


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
    regional: RegionalLiveFixture, settings: PreflightSettings
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
    arbiter_pods: dict[str, list[str]] | None = None,
    dns_nodes: list[str] | None = None,
) -> list[str]:
    errors: list[str] = list(reboot_probe_errors)
    # Precondition 2 of the spec: neither target node may host a controller or
    # executor arbiter replica -- QUIESCE stops kubelet on the fault node and
    # both nodes really reboot, so a replica placed there is lost mid-case --
    # and a pair holding every kube-dns endpoint takes cluster DNS down with
    # it (the rule GF-REGIONAL-DESTR-015 learned live). Read-only: the
    # placement is what ``cluster_arbiter_placement`` listed before the plan.
    for node in (fault_node, sibling_node):
        hosted = (arbiter_pods or {}).get(node) or []
        if hosted:
            errors.append(f"{node} hosts controller/executor arbiter Pods: {hosted}")
    if (
        dns_nodes is not None
        and dns_nodes
        and set(dns_nodes)
        <= {
            fault_node,
            sibling_node,
        }
    ):
        errors.append(
            "the two target nodes hold every kube-dns endpoint; rebooting both "
            "would take cluster DNS down"
        )
    if (
        not sibling.get("uid")
        or sibling_agent.get("node_instance_id") != sibling["uid"]
        or not sibling_agent.get("runtime_profile_version")
        or any(
            not re.fullmatch(r"[0-9a-f]{64}", str(sibling_agent.get(key) or ""))
            for key in ("artifact_sha256", "installer_bundle_sha256")
        )
    ):
        errors.append(
            "sibling Node Agent recovery identity is incomplete or mismatched"
        )
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


def _control_env(regional: RegionalLiveFixture) -> dict[str, Any]:
    values: dict[str, Any] = {"poll_interval_seconds": 5.0, "max_rungs": 2}
    for pod in regional.ready_pods("gpu", env_window.DEPLOYMENT):
        try:
            output = regional.kubectl(
                "gpu",
                "exec",
                str(pod["name"]),
                "--",
                component_python("gpu"),
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
                component_python("gpu"),
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

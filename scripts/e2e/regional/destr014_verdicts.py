"""Pure verdict functions and constants of GF-REGIONAL-DESTR-014.

Split out of ``run_destr014_branch_exhaustion.py`` so the runner stays a
driver: everything here is unit-tested against synthetic control-plane,
node and CloudTrail snapshots and touches no cluster.
"""

from __future__ import annotations

import hashlib
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.e2e.regional.warm_spare_fixture import QUARANTINE_TAINT  # noqa: E402

REBOOT_EVENTS = {"BatchRebootClusterNodes", "RebootClusterNodes"}
FORBIDDEN_EVENTS = {
    "BatchDeleteClusterNodes",
    "BatchReplaceClusterNodes",
    "DeleteClusterNodes",
    "ReplaceClusterNodes",
}

# Wall-clock allowances (seconds) for the lifetime arithmetic. A branch takes a
# real reboot plus a validation-and-release tail; the shared containment runs
# once. Deliberately generous so a passing estimate is a real safety margin.
REBOOT_ALLOWANCE_SECONDS = 600
VALIDATION_ALLOWANCE_SECONDS = 300
CONTAINMENT_ALLOWANCE_SECONDS = 120

EXHAUSTION_PREFIX = "node branch escalation exhausted"

# What the shipped adapter says when the strategy is HEALTHY_WARM_SPARE_ONLY and
# no spare pool is declared at all. ``hyperpod_spares`` reports the allocation as
# not *applicable* (there is no pool to be short of), and the lifecycle adapter
# then refuses because the provider fallback is disabled -- a different sentence
# from the "insufficient healthy HyperPod spares" a declared-but-short pool
# produces, which is DESTR-008's topology-mismatch scenario, not this one.
EXPECTED_REPLACE_FAILURE = (
    "warm-spare replacement is required; provider node replacement API "
    "fallback is disabled"
)

# The remediation budget scopes this drill claims: the region, the cluster, one
# node scope per faulted node, and one resource class per budgeted operation.
# Failure-domain scopes come from an operator-declared map the runner cannot
# read from outside the executor, so they are not modelled here; the preflight
# reads the region/cluster/node/class tiers, which is where the drill's own
# concurrency lands.
BUDGET_LIMIT_ENV = {
    "region": ("GPU_FAULT_REMEDIATION_MAX_ACTIVE_REGION", 20),
    "cluster": ("GPU_FAULT_REMEDIATION_MAX_ACTIVE_PER_CLUSTER", 5),
    "node": ("GPU_FAULT_REMEDIATION_MAX_ACTIVE_PER_NODE", 1),
    "resource_class": ("GPU_FAULT_REMEDIATION_MAX_ACTIVE_PER_RESOURCE_CLASS", 2),
}
BUDGETED_OPERATIONS = ("RESET_GPU", "RESTART_NODE", "REPLACE_NODE")


# --------------------------------------------------------------------------- #
# Pure verdict functions (unit-tested)
# --------------------------------------------------------------------------- #
def quarantine_taint_value(incident_id: str) -> str:
    return hashlib.sha256(incident_id.encode()).hexdigest()[:24]


def _step_node(step: dict[str, Any]) -> list[str]:
    return list(step.get("branch_node_ids") or step.get("node_ids") or [])


def _branch_executions(
    steps: list[dict[str, Any]],
    executions: list[dict[str, Any]],
    node: str,
) -> list[dict[str, Any]]:
    result = []
    for item in executions:
        index = item.get("step_index")
        if isinstance(index, int) and 0 <= index < len(steps):
            if _step_node(steps[index]) == [node]:
                result.append(item)
    return result


def _first(
    executions: list[dict[str, Any]],
    operation: str,
    status: str,
) -> dict[str, Any] | None:
    return next(
        (
            item
            for item in executions
            if item.get("operation") == operation and item.get("status") == status
        ),
        None,
    )


def workflow_errors(
    workflow: dict[str, Any],
    incident: dict[str, Any],
    *,
    fault_node: str,
    sibling_node: str,
    failure_reason: str | None,
) -> list[str]:
    errors: list[str] = []
    if workflow.get("status") != "FAILED":
        errors.append("workflow status is not FAILED")
    if incident.get("state") != "QUARANTINED":
        errors.append("incident state is not QUARANTINED")
    if not workflow.get("dag_enabled"):
        errors.append("workflow is not dag_enabled")
    counts = workflow.get("branch_escalation_counts") or {}
    if counts != {fault_node: 1, sibling_node: 1}:
        errors.append(
            "branch_escalation_counts is not "
            f"{{{fault_node}: 1, {sibling_node}: 1}}: {counts}"
        )
    exhausted = workflow.get("exhausted_branch_ids") or []
    if len(exhausted) != 1 or not str(exhausted[0]).startswith(
        f"branch:{sibling_node}"
    ):
        errors.append(
            "exhausted_branch_ids is not exactly one "
            f"branch:{sibling_node} entry: {exhausted}"
        )
    if not str(failure_reason or "").startswith(EXHAUSTION_PREFIX):
        errors.append(
            f"failure reason does not start with {EXHAUSTION_PREFIX!r}: "
            f"{failure_reason!r}"
        )

    steps = workflow.get("official_steps") or []
    executions = workflow.get("step_executions") or []
    if any(item.get("operation") == "RESTART_WORKLOAD" for item in executions):
        errors.append("RESTART_WORKLOAD has an execution record; the join ran")

    resolved = set(workflow.get("completed_step_indexes") or []) | set(
        workflow.get("superseded_step_indexes") or []
    )
    for index, step in enumerate(steps):
        if step.get("branch_id") == "join":
            continue
        if index not in resolved:
            errors.append(
                f"non-join step {index}/{step.get('operation')} is not resolved"
            )

    fault_execs = _branch_executions(steps, executions, fault_node)
    sibling_execs = _branch_executions(steps, executions, sibling_node)

    reset = next(
        (
            item
            for item in fault_execs
            if item.get("operation") == "RESET_GPU" and item.get("status") == "FAILED"
        ),
        None,
    )
    if reset is None:
        errors.append(f"{fault_node} has no FAILED RESET_GPU execution")
    else:
        if "clients are still active" not in str(reset.get("error") or ""):
            errors.append(
                f"{fault_node} RESET_GPU did not fail with 'clients are still active'"
            )
        if "gpu_reset_commit_attempt" not in (reset.get("details") or {}):
            errors.append(
                f"{fault_node} RESET_GPU failure lacks gpu_reset_commit_attempt"
            )

    if _first(fault_execs, "RESTART_NODE", "SUCCEEDED") is None:
        errors.append(f"{fault_node} in-place RESTART_NODE did not SUCCEED")
    for operation in (
        "VALIDATE_GPU",
        "VALIDATE_HOST",
        "VALIDATE_FABRIC",
        "RESTORE_SCHEDULING",
    ):
        if _first(fault_execs, operation, "SUCCEEDED") is None:
            errors.append(f"{fault_node} branch did not complete {operation}")

    sibling_reboot = next(
        (
            item
            for item in sibling_execs
            if item.get("operation") == "RESTART_NODE"
            and item.get("status") == "FAILED"
        ),
        None,
    )
    if sibling_reboot is None:
        errors.append(f"{sibling_node} has no FAILED RESTART_NODE execution")
    elif "step_waiting_timeout_seconds" not in (sibling_reboot.get("details") or {}):
        errors.append(
            f"{sibling_node} RESTART_NODE did not fail by bounded waiting "
            "(step_waiting_timeout_seconds absent)"
        )

    replace_step = next(
        (
            step
            for step in steps
            if step.get("operation") == "REPLACE_NODE"
            and _step_node(step) == [sibling_node]
        ),
        None,
    )
    if replace_step is None:
        errors.append(f"{sibling_node} branch has no REPLACE_NODE step")
    elif (replace_step.get("parameters") or {}).get("replacement_strategy") != (
        "HEALTHY_WARM_SPARE_ONLY"
    ):
        errors.append(f"{sibling_node} REPLACE_NODE is not HEALTHY_WARM_SPARE_ONLY")
    replace_exec = next(
        (
            item
            for item in sibling_execs
            if item.get("operation") == "REPLACE_NODE"
            and item.get("status") == "FAILED"
        ),
        None,
    )
    if replace_exec is None:
        errors.append(f"{sibling_node} has no FAILED REPLACE_NODE execution")
    elif EXPECTED_REPLACE_FAILURE not in str(replace_exec.get("error") or ""):
        errors.append(
            f"{sibling_node} REPLACE_NODE did not fail with "
            f"{EXPECTED_REPLACE_FAILURE!r}: {replace_exec.get('error')!r}"
        )
    return errors


def host_errors(
    *,
    fault_baseline: dict[str, Any],
    fault_after: dict[str, Any],
    sibling_baseline: dict[str, Any],
    sibling_after: dict[str, Any],
    holder_status: dict[str, Any],
    sibling_agent_during: dict[str, Any],
    sibling_agent_after: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    if fault_after.get("boot_id") == fault_baseline.get("boot_id"):
        errors.append("fault node boot id did not change across its reboot")
    if sibling_after.get("boot_id") == sibling_baseline.get("boot_id"):
        errors.append("sibling node boot id did not change across its reboot")

    baseline_ids = {
        (row.get("command_id"), row.get("operation"))
        for row in fault_baseline.get("ledger") or []
    }
    new_reset_rows = [
        row
        for row in fault_after.get("ledger") or []
        if row.get("operation") == "RESET_GPU"
        and (row.get("command_id"), row.get("operation")) not in baseline_ids
    ]
    if not new_reset_rows:
        errors.append("fault node ledger has no new RESET_GPU row; the reset never ran")
    if any(row.get("state") == "SUCCEEDED" for row in new_reset_rows):
        errors.append("fault node RESET_GPU succeeded; the holder did not break it")

    if not holder_status.get("matched_row") or not holder_status.get("hold_started_at"):
        errors.append("GPU device holder never armed against a VERIFY ledger row")

    if str(sibling_agent_during.get("UnitFileState") or "") != "disabled":
        errors.append("sibling Node Agent was not disabled during the run")
    if (
        str(sibling_agent_after.get("UnitFileState") or "") != "enabled"
        or sibling_agent_after.get("ActiveState") != "active"
    ):
        errors.append("sibling Node Agent was not restored after the run")
    return errors


def cloudtrail_errors(
    events: list[dict[str, Any]],
    fault_node: str,
    sibling_node: str,
) -> list[str]:
    errors: list[str] = []
    reboots = [item for item in events if item.get("event_name") in REBOOT_EVENTS]
    if len(reboots) != 2:
        errors.append(
            "CloudTrail does not contain exactly two BatchRebootClusterNodes "
            f"(one per node): got {len(reboots)}"
        )
    for item in events:
        if item.get("event_name") in FORBIDDEN_EVENTS:
            errors.append(
                f"CloudTrail contains a forbidden mutation: {item['event_name']}"
            )
    return errors


def schedulability_errors(
    snapshots: dict[str, dict[str, Any]],
    *,
    fault_node: str,
    sibling_node: str,
    incident_id: str,
) -> list[str]:
    errors: list[str] = []
    fault = snapshots.get(fault_node) or {}
    sibling = snapshots.get(sibling_node) or {}
    if fault.get("unschedulable"):
        errors.append(f"fault node {fault_node} is still cordoned")
    if any(item.get("key") == QUARANTINE_TAINT for item in fault.get("taints") or []):
        errors.append(f"fault node {fault_node} still carries the quarantine taint")
    if not sibling.get("unschedulable"):
        errors.append(f"sibling node {sibling_node} is not cordoned")
    taint = next(
        (
            item
            for item in sibling.get("taints") or []
            if item.get("key") == QUARANTINE_TAINT
        ),
        None,
    )
    if taint is None:
        errors.append(f"sibling node {sibling_node} is missing the quarantine taint")
    elif taint.get("value") != quarantine_taint_value(incident_id):
        errors.append(
            f"sibling node {sibling_node} quarantine taint value is not the "
            "incident digest"
        )
    return errors


def injection_errors(
    fault_state: dict[str, Any],
    sibling_state: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    fault_workflow = fault_state.get("workflow") or {}
    sibling_workflow = sibling_state.get("workflow") or {}
    request = fault_workflow.get("request_id")
    if not request or request != sibling_workflow.get("request_id"):
        errors.append("the two XID injections did not join one workflow_request_id")
    if not (fault_workflow.get("dag_enabled") and sibling_workflow.get("dag_enabled")):
        errors.append("the joined workflow is not dag_enabled")
    return errors


def follow_up_errors(
    follow_up: dict[str, Any],
    *,
    fault_node: str,
    sibling_node: str,
) -> list[str]:
    errors: list[str] = []
    incident = follow_up.get("incident")
    workflow = follow_up.get("workflow")
    if not incident or not workflow:
        errors.append("hardware escalation follow-up is missing for the incident")
        return errors
    operations = [
        step.get("operation") for step in workflow.get("official_steps") or []
    ]
    if "ESCALATE_SUPPORT" not in operations:
        errors.append("follow-up workflow has no ESCALATE_SUPPORT step")
    nodes = incident.get("node_ids") or []
    if nodes != [sibling_node]:
        errors.append(
            f"follow-up incident is not scoped to {sibling_node} alone: {nodes}"
        )
    return errors


def workload_errors(
    *,
    pods: list[dict[str, Any]],
    restart_budget: dict[str, Any],
    source_uids: set[str],
) -> list[str]:
    errors: list[str] = []
    live = [item for item in pods if item.get("phase") not in {"Succeeded", "Failed"}]
    if live:
        errors.append(f"the job still has {len(live)} Pod(s) after the failed workflow")
    if not restart_budget:
        # Fail-safe, but for the right reason: a budget row the store never
        # returned proves nothing either way, and calling it "advanced" sent the
        # reader looking for a restart that may never have happened.
        errors.append(
            "restart budget unreadable; cannot prove the job was not restarted"
        )
    elif restart_budget.get("restart_count") != 0:
        errors.append("restart budget advanced; the job was restarted")
    return errors


def step_transitions(
    previous: dict[str, str],
    executions: list[dict[str, Any]],
) -> tuple[dict[str, str], list[dict[str, Any]]]:
    current = dict(previous)
    changes: list[dict[str, Any]] = []
    for item in executions:
        key = f"{item.get('step_index')}:{item.get('operation')}"
        status = str(item.get("status") or "")
        if current.get(key) != status:
            changes.append(
                {
                    "step_index": item.get("step_index"),
                    "operation": item.get("operation"),
                    "from": current.get(key),
                    "to": status,
                    "observed_at": datetime.now(timezone.utc).isoformat(),
                }
            )
            current[key] = status
    return current, changes


def estimated_duration_seconds(
    *,
    verify_max_attempts: int,
    poll_interval_seconds: float,
    managed_recovery_timeout_seconds: int,
) -> int:
    return int(
        round(verify_max_attempts * poll_interval_seconds)
        + managed_recovery_timeout_seconds
        + 2 * REBOOT_ALLOWANCE_SECONDS
        + VALIDATION_ALLOWANCE_SECONDS
        + CONTAINMENT_ALLOWANCE_SECONDS
    )


def lifetime_errors(
    *,
    estimated_seconds: int,
    lifetime_seconds: int | None,
) -> list[str]:
    if lifetime_seconds is None:
        return ["workflow lifetime is unknown; refusing to run fail-closed"]
    if estimated_seconds >= lifetime_seconds:
        return [
            f"estimated duration {estimated_seconds}s meets or exceeds the job "
            f"workflow lifetime {lifetime_seconds}s"
        ]
    return []


def budget_limits(environment: dict[str, Any]) -> dict[str, int]:
    """The executor's remediation concurrency limits, from its environment.

    Unset variables take the product's own defaults; a value that is not a
    positive integer is refused rather than defaulted, because the executor
    would refuse it too and a preflight that read past it would be grading a
    configuration the executor does not run.
    """

    limits: dict[str, int] = {}
    for tier, (name, default) in BUDGET_LIMIT_ENV.items():
        raw = environment.get(name)
        try:
            value = int(str(raw)) if raw not in (None, "") else default
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} is not an integer: {raw!r}") from exc
        if value < 1:
            raise ValueError(f"{name} must be positive: {value}")
        limits[tier] = value
    return limits


def planned_budget_scopes(
    *,
    cluster_id: str,
    nodes: list[str],
    limits: dict[str, int],
    resource_classes: dict[str, list[str]],
) -> dict[str, int]:
    """Scope name -> limit for the claims this drill's workflow will make.

    ``resource_classes`` maps each budgeted operation to the resource classes
    the operation registry declares for it, so the class tier follows the
    product's registry rather than a list kept here.
    """

    scopes = {
        "region": limits["region"],
        f"cluster:{cluster_id}": limits["cluster"],
    }
    for node in sorted(set(nodes)):
        scopes[f"node:{cluster_id}:{node}"] = limits["node"]
    classes: set[str] = set()
    for operation in BUDGETED_OPERATIONS:
        declared = resource_classes.get(operation)
        if declared is None:
            raise ValueError(f"{operation} has no resource classes in the registry")
        classes.update(declared)
    for resource_class in sorted(classes):
        scopes[f"class:{cluster_id}:{resource_class}"] = limits["resource_class"]
    return scopes


def budget_headroom(
    scopes: dict[str, int],
    active_workflows: list[dict[str, Any]],
) -> dict[str, Any]:
    """The ``budget`` ``budget_headroom_errors`` grades, from live store rows.

    ``active_workflows`` are the RUNNING, lease-holding workflows and the
    scopes each holds; a scope is as busy as the number of such rows naming it.
    """

    counted: dict[str, dict[str, int]] = {}
    for name, limit in scopes.items():
        active = sum(
            1 for item in active_workflows if name in (item.get("claims") or [])
        )
        counted[name] = {"limit": int(limit), "active": active}
    return {"readable": True, "scopes": counted}


def budget_headroom_errors(budget: dict[str, Any]) -> list[str]:
    if not budget.get("readable"):
        return ["budget_headroom_unknown: remediation budget could not be read"]
    scopes = budget.get("scopes") or {}
    if not scopes:
        return ["budget_headroom_unknown: no remediation budget scopes were computed"]
    errors: list[str] = []
    for name, scope in scopes.items():
        if int(scope.get("active") or 0) >= int(scope.get("limit") or 0):
            errors.append(f"remediation budget scope is full: {name}")
    return errors

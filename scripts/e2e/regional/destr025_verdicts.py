"""Pure verdict functions and constants of GF-REGIONAL-DESTR-025.

A single-node ``RESET_GPU`` that fails with a *known* outcome (the GPU has a
live client, so the reset is refused) must climb the hardware ladder inside
its own workflow record: the flat plan is grown into a one-branch DAG, a
``RESTART_NODE`` rung with its validation tail is appended, the node really
reboots and comes back, and the record ends ``SUCCEEDED`` with the incident
``RECOVERED``. No ``workflow-reboot-after-<id>`` successor may appear.

Split out of ``run_destr025_single_node_reset_escalation.py`` so the runner
stays a driver: everything here is unit-tested against synthetic
control-plane and host snapshots and touches no cluster.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gpu_fault.recovery_safety import unresolved_details  # noqa: E402
from scripts.e2e.regional.acceptance_runner_common import (  # noqa: E402
    open_incident_errors,
    processor_queue_backlog,
)
from scripts.e2e.regional.destr014_verdicts import (  # noqa: E402
    RESET_FAILURE_DETAIL_KEYS,
    budget_headroom_errors,
    canonical_digest,
)
from scripts.e2e.regional.destr016_verdicts import (  # noqa: E402
    AGENT_OPERATIONS,
    holder_errors,
    lifetime_errors,
)
from scripts.e2e.regional.run_destr002_hyperpod_reboot import (  # noqa: E402
    capability,
)

CASE_ID = "GF-REGIONAL-DESTR-025"
PREDECESSOR_CASE_ID = "GF-REGIONAL-DESTR-002"
CONFIRMATION = "DESTR025_EXECUTE"

RESET_OPERATION = "RESET_GPU"
REBOOT_OPERATION = "RESTART_NODE"
REPLACE_OPERATION = "REPLACE_NODE"
RUNG_EVENT_CODE = "BRANCH_ESCALATED"
EXHAUSTED_EVENT_CODE = "BRANCH_EXHAUSTED"
BARRIER_OPERATION = "VERIFY_NO_GPU_CLIENTS"
CLIENTS_ACTIVE_REASON = "clients are still active"
VALIDATION_TAIL = (
    "VALIDATE_GPU",
    "VALIDATE_HOST",
    "VALIDATE_FABRIC",
    "RESTORE_SCHEDULING",
)
SUCCESSOR_PREFIXES = (
    "workflow-reboot-after-",
    "workflow-replace-after-",
    "workflow-support-after-",
)
FORBIDDEN_LEDGER_OPERATIONS = ("RESET_GPU", "RESET_ALL_GPUS_NVSWITCHES")
GPU_HOLD_SECONDS = 900
MIN_VERIFY_ATTEMPTS = 1
MAX_VERIFY_ATTEMPTS = 20

# Wall-clock allowances (seconds) for the lifetime arithmetic: containment
# and the barrier, the refused reset, one real reboot, the validation tail.
CONTAINMENT_ALLOWANCE_SECONDS = 300
REBOOT_ALLOWANCE_SECONDS = 1200
VALIDATION_ALLOWANCE_SECONDS = 600


# --------------------------------------------------------------------------- #
# Identity and digests
# --------------------------------------------------------------------------- #
def plan_identity(preflight: dict[str, Any], *, node: str) -> dict[str, Any]:
    snapshot = preflight.get("node") or {}
    store = preflight.get("store") or {}
    return {
        "release_id": preflight.get("release_id"),
        "node": node,
        "node_uid": snapshot.get("uid"),
        "node_boot_id": snapshot.get("boot_id"),
        "runtime_profile_version": (store.get("profile") or {}).get("profile_version"),
        "runtime_identity": preflight.get("runtime_identity"),
    }


def identity_digest(identity: dict[str, Any]) -> str:
    return canonical_digest(identity)


def evidence_components(details: dict[str, Any]) -> dict[str, str]:
    return {name: canonical_digest(value) for name, value in details.items()}


def case_digest(components: dict[str, str]) -> str:
    return canonical_digest(components)


# --------------------------------------------------------------------------- #
# Workflow shape
# --------------------------------------------------------------------------- #
def executions_of(workflow: dict[str, Any], operation: str) -> list[dict[str, Any]]:
    return [
        item
        for item in workflow.get("step_executions") or []
        if item.get("operation") == operation
    ]


def known_failure(execution: dict[str, Any]) -> bool:
    """A FAILED step whose receipt says what happened, not one an operator
    still has to confirm (the same flags the CPU restore gate reads)."""

    return execution.get("status") == "FAILED" and not unresolved_details(
        execution.get("details") or {}
    )


def rung_events(workflow: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        event
        for event in workflow.get("events") or []
        if event.get("code") == RUNG_EVENT_CODE
    ]


def _index(value: Any) -> int:
    return int(value) if isinstance(value, int) else -1


def _reset_failure(workflow: dict[str, Any]) -> dict[str, Any] | None:
    failed = [
        item
        for item in executions_of(workflow, RESET_OPERATION)
        if item.get("status") == "FAILED"
    ]
    return failed[0] if len(failed) == 1 else None


def _terminal_errors(workflow: dict[str, Any], incident: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if workflow.get("status") != "SUCCEEDED":
        errors.append(f"the workflow is not SUCCEEDED: {workflow.get('status')}")
    if incident.get("state") != "RECOVERED":
        errors.append(f"the incident is not RECOVERED: {incident.get('state')}")
    if workflow.get("terminal_failure_reason"):
        errors.append(
            f"the record carries a failure reason: {workflow.get('terminal_failure_reason')}"
        )
    if not workflow.get("dag_enabled"):
        errors.append("the record is not dag_enabled; the flat plan never grew a rung")
    if workflow.get("exhausted_branch_ids"):
        errors.append(f"a branch was exhausted: {workflow.get('exhausted_branch_ids')}")
    return errors


def _rung_errors(workflow: dict[str, Any], *, node: str) -> list[str]:
    errors: list[str] = []
    counts = workflow.get("branch_escalation_counts") or {}
    if counts != {node: 1}:
        errors.append(f"branch_escalation_counts is not {{{node!r}: 1}}: {counts}")
    rungs = rung_events(workflow)
    if len(rungs) != 1:
        errors.append(f"the record does not carry exactly one {RUNG_EVENT_CODE}")
    for event in rungs:
        details = event.get("details") or {}
        if details.get("from_operation") != RESET_OPERATION:
            errors.append(
                f"the rung did not escalate from {RESET_OPERATION}: "
                f"{details.get('from_operation')}"
            )
        if details.get("to_operation") == REPLACE_OPERATION:
            errors.append(f"a rung to {REPLACE_OPERATION} was taken")
        elif details.get("to_operation") != REBOOT_OPERATION:
            errors.append(
                f"the rung did not escalate to {REBOOT_OPERATION}: "
                f"{details.get('to_operation')}"
            )
    if any(
        event.get("code") == EXHAUSTED_EVENT_CODE
        for event in workflow.get("events") or []
    ):
        errors.append("the record carries an exhausted-branch event")
    return errors


def _reset_errors(reset: dict[str, Any] | None) -> list[str]:
    if reset is None:
        return [f"the record has no single FAILED {RESET_OPERATION} execution"]
    errors: list[str] = []
    if not known_failure(reset):
        errors.append(
            f"the {RESET_OPERATION} failure is not a known outcome; an operator "
            "confirmation hold is not this case"
        )
    if CLIENTS_ACTIVE_REASON not in str(reset.get("error") or ""):
        errors.append(
            f"{RESET_OPERATION} did not fail with {CLIENTS_ACTIVE_REASON!r}: "
            f"{reset.get('error')!r}"
        )
    if not (RESET_FAILURE_DETAIL_KEYS & set(reset.get("details") or {})):
        errors.append(
            f"{RESET_OPERATION} failure carries none of "
            f"{sorted(RESET_FAILURE_DETAIL_KEYS)}"
        )
    return errors


def _ladder_errors(workflow: dict[str, Any], reset_index: int) -> list[str]:
    """The reboot rung sits behind the failed reset; every validation and the
    release sit behind the reboot; nothing of the flat tail ran."""

    errors: list[str] = []
    restarts = [
        item
        for item in executions_of(workflow, REBOOT_OPERATION)
        if item.get("status") == "SUCCEEDED"
    ]
    if len(restarts) != 1:
        errors.append(
            f"the record does not have exactly one SUCCEEDED {REBOOT_OPERATION}: "
            f"{len(restarts)}"
        )
        return errors
    restart_index = _index(restarts[0].get("step_index"))
    if restart_index <= reset_index:
        errors.append(
            f"{REBOOT_OPERATION} at step {restart_index} is not behind the failed "
            f"{RESET_OPERATION} at step {reset_index}"
        )
    if any(
        item.get("status") == "FAILED"
        for item in executions_of(workflow, REBOOT_OPERATION)
    ):
        errors.append(f"{REBOOT_OPERATION} has a FAILED execution")
    for operation in VALIDATION_TAIL:
        rows = executions_of(workflow, operation)
        behind = [
            item
            for item in rows
            if _index(item.get("step_index")) > restart_index
            and item.get("status") == "SUCCEEDED"
        ]
        if len(behind) != 1:
            errors.append(
                f"{operation} did not SUCCEED exactly once after the reboot rung: "
                f"{[(item.get('step_index'), item.get('status')) for item in rows]}"
            )
        ahead = [
            item
            for item in rows
            if _index(item.get("step_index")) < restart_index
            and item.get("status") not in {None, "WAITING"}
        ]
        if ahead:
            errors.append(
                f"{operation} ran before the reboot rung at step {restart_index}: "
                f"{[item.get('step_index') for item in ahead]}"
            )
    return errors


def _retirement_errors(workflow: dict[str, Any], reset_index: int) -> list[str]:
    steps = workflow.get("official_steps") or []
    if not 0 <= reset_index < len(steps):
        return [f"the failed reset index {reset_index} is not a planned step"]
    superseded = set(workflow.get("superseded_step_indexes") or [])
    completed = set(workflow.get("completed_step_indexes") or [])
    errors: list[str] = []
    branch = steps[reset_index].get("branch_id")
    tail = [
        index
        for index, step in enumerate(steps)
        if index >= reset_index and step.get("branch_id") == branch
    ]
    missing = [index for index in tail if index not in superseded]
    if missing:
        errors.append(
            "the failed reset and the flat tail behind it are not retired "
            f"(superseded_step_indexes): {missing}"
        )
    unresolved = [
        f"{index}/{step.get('operation')}"
        for index, step in enumerate(steps)
        if index not in superseded and index not in completed
    ]
    if unresolved:
        errors.append(f"planned steps neither completed nor retired: {unresolved}")
    return errors


def workflow_errors(
    workflow: dict[str, Any],
    incident: dict[str, Any],
    *,
    node: str,
) -> list[str]:
    """The in-record ladder contract, judged from one workflow read."""

    errors = _terminal_errors(workflow, incident)
    errors.extend(_rung_errors(workflow, node=node))
    reset = _reset_failure(workflow)
    errors.extend(_reset_errors(reset))
    if reset is not None:
        reset_index = _index(reset.get("step_index"))
        errors.extend(_ladder_errors(workflow, reset_index))
        errors.extend(_retirement_errors(workflow, reset_index))
    return errors


def successor_errors(
    successors: list[dict[str, Any]],
    *,
    request_id: str,
    workflow: dict[str, Any],
) -> list[str]:
    """No other record may have taken the reboot over from this one."""

    errors: list[str] = []
    for item in successors:
        other = str(item.get("request_id") or "")
        if any(other == f"{prefix}{request_id}" for prefix in SUCCESSOR_PREFIXES):
            errors.append(
                f"a whole-workflow escalation successor exists: {other} "
                "(the reboot left the record)"
            )
        elif item.get("predecessor_workflow_id") == request_id:
            errors.append(f"a record names this workflow as its predecessor: {other}")
    if workflow.get("preempted_by_workflow_id"):
        errors.append(
            "the record was preempted; it did not finish its own ladder: "
            f"{workflow.get('preempted_by_workflow_id')}"
        )
    return errors


def command_errors(
    commands: list[dict[str, Any]], workflow: dict[str, Any]
) -> list[str]:
    """Every remote command of the record was minted under its fencing token."""

    if not commands:
        return ["the record has no remote commands; nothing ran on the node"]
    errors: list[str] = []
    token = workflow.get("fencing_token")
    request_id = workflow.get("request_id")
    for item in commands:
        if item.get("workflow_request_id") not in {None, request_id}:
            errors.append(
                f"command {item.get('command_id')} belongs to another workflow"
            )
        if item.get("fencing_token") != token:
            errors.append(
                f"command {item.get('command_id')} carries fencing token "
                f"{item.get('fencing_token')}, not {token}"
            )
    return errors


# --------------------------------------------------------------------------- #
# Host evidence
# --------------------------------------------------------------------------- #
def _added_rows(
    baseline: list[dict[str, Any]], after: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    seen = {(row.get("command_id"), row.get("operation")) for row in baseline}
    return [
        row
        for row in after
        if (row.get("command_id"), row.get("operation")) not in seen
    ]


def host_errors(
    baseline: dict[str, Any],
    after: dict[str, Any],
    *,
    holder_status: dict[str, Any],
    expected_gpu_count: int,
) -> list[str]:
    """The node's own record: a real reboot, a refused reset (never a
    successful one, on either boot), a client verification that ran, the GPU
    inventory back, no quiesce or client left, and the holder gone."""

    errors: list[str] = []
    if not baseline.get("boot_id") or not after.get("boot_id"):
        errors.append("a host boot id snapshot is missing")
    elif baseline.get("boot_id") == after.get("boot_id"):
        errors.append("the host boot id did not change; the node did not reboot")
    added = _added_rows(
        list(baseline.get("ledger") or []), list(after.get("ledger") or [])
    )
    for operation in FORBIDDEN_LEDGER_OPERATIONS:
        succeeded = [
            row.get("command_id")
            for row in added
            if row.get("operation") == operation and row.get("state") == "SUCCEEDED"
        ]
        if succeeded:
            errors.append(
                f"the Node Agent ledger shows a successful {operation}; the holder "
                f"did not break it: {succeeded}"
            )
    if not any(row.get("operation") == BARRIER_OPERATION for row in added):
        errors.append(f"the Node Agent ledger has no {BARRIER_OPERATION} row")
    if len(after.get("gpu_inventory") or []) != expected_gpu_count:
        errors.append(
            f"the host GPU inventory is not {expected_gpu_count} after the reboot: "
            f"{len(after.get('gpu_inventory') or [])}"
        )
    if after.get("quiesce_states"):
        errors.append(
            f"a GPU quiesce state survives the reboot: {after['quiesce_states']}"
        )
    if after.get("compute_clients"):
        errors.append(
            f"the node still has NVIDIA compute clients: {after['compute_clients']}"
        )
    for unit, state in (baseline.get("services") or {}).items():
        if state.get("ActiveState") != "active":
            continue
        current = (after.get("services") or {}).get(unit) or {}
        if current.get("ActiveState") != "active":
            errors.append(f"a service did not return active after the reboot: {unit}")
    errors.extend(holder_errors(holder_status))
    return errors


# --------------------------------------------------------------------------- #
# Arithmetic and preflight
# --------------------------------------------------------------------------- #
def estimated_duration_seconds(
    *, verify_max_attempts: int, poll_interval_seconds: float
) -> int:
    return int(
        CONTAINMENT_ALLOWANCE_SECONDS
        + round(verify_max_attempts * poll_interval_seconds)
        + REBOOT_ALLOWANCE_SECONDS
        + VALIDATION_ALLOWANCE_SECONDS
    )


def verify_attempt_errors(verify_max_attempts: int) -> list[str]:
    if not MIN_VERIFY_ATTEMPTS <= verify_max_attempts <= MAX_VERIFY_ATTEMPTS:
        return [
            f"verify attempts {verify_max_attempts} is outside "
            f"{MIN_VERIFY_ATTEMPTS}..{MAX_VERIFY_ATTEMPTS}"
        ]
    return []


def _node_errors(
    node: str, node_snapshot: dict[str, Any], agent: dict[str, Any]
) -> list[str]:
    errors: list[str] = []
    if (
        node_snapshot.get("ready") != "True"
        or node_snapshot.get("unschedulable")
        or node_snapshot.get("taints")
        or node_snapshot.get("ownership_annotations")
    ):
        errors.append(f"{node} is not Ready, schedulable, untainted and unowned")
    if int(node_snapshot.get("gpu_allocatable") or 0) < 1:
        errors.append(f"{node} allocates no GPU")
    if agent.get("lifecycle_state") != "ACTIVE":
        errors.append(f"{node} Node Agent is not ACTIVE")
    missing = sorted(set(AGENT_OPERATIONS) - set(agent.get("allowed_operations") or []))
    if missing:
        errors.append(f"{node} Node Agent does not allow {missing}")
    return errors


def _profile_errors(profile: dict[str, Any] | None) -> list[str]:
    errors: list[str] = []
    reset = capability(profile, "gpuReset")
    if reset is None:
        errors.append("the runtime profile has no gpuReset capability")
    elif reset.get("mode") != "OWN" or reset.get("owner") != "gpu-fault-node-agent":
        errors.append("gpuReset is not OWN by the Node Agent")
    reboot = capability(profile, "nodeReboot")
    if reboot is None:
        errors.append("the runtime profile has no nodeReboot capability")
    elif (
        reboot.get("mode") != "OWN"
        or reboot.get("owner") != "gpu-fault-hyperpod-adapter"
        or reboot.get("adapter") != "regional-cluster-executor"
    ):
        errors.append("nodeReboot is not OWN by the HyperPod adapter")
    if (profile or {}).get("warnings"):
        errors.append("the runtime profile has warnings")
    return errors


def _executor_errors(executor_env: list[dict[str, Any]]) -> list[str]:
    if not executor_env:
        return ["no ready executor replica reported its environment"]
    for pod in executor_env:
        if pod.get("allow_reboot") != "true" or pod.get("allow_replace") != "false":
            return [
                "executor reboot safety environment is inconsistent (allow_reboot "
                "must be true and allow_replace false)"
            ]
    return []


def preflight_errors(
    *,
    node: str,
    node_snapshot: dict[str, Any],
    agent: dict[str, Any],
    profile: dict[str, Any] | None,
    queue: dict[str, Any],
    remote_commands: dict[str, Any],
    gpu_workloads: list[dict[str, Any]],
    business_workloads: list[dict[str, Any]],
    event: dict[str, Any] | None,
    open_incidents: list[dict[str, Any]],
    predecessor: dict[str, Any],
    tests: dict[str, Any],
    control_env: dict[str, Any],
    executor_env: list[dict[str, Any]],
    budget: dict[str, Any],
    identity_errors: list[str],
    verify_max_attempts: int,
) -> list[str]:
    """Everything that must hold before a real reboot is authorized."""

    errors: list[str] = list(identity_errors)
    if not predecessor.get("valid"):
        errors.append(f"{PREDECESSOR_CASE_ID} predecessor evidence is not PASS")
    if not tests.get("passed"):
        errors.append("focused regression tests failed")
    errors.extend(_node_errors(node, node_snapshot, agent))
    if business_workloads:
        errors.append(f"{node} carries a non-system workload: {business_workloads}")
    if gpu_workloads:
        errors.append(f"the GPU cluster already has a GPU workload: {gpu_workloads}")
    if event is not None:
        errors.append(f"{node} has a recent XID event")
    errors.extend(open_incident_errors(node, open_incidents))
    errors.extend(_profile_errors(profile))
    if processor_queue_backlog(queue):
        errors.append("the processor queue is not empty")
    if remote_commands.get("open_by_cluster"):
        errors.append("the remote command queue is not empty")
    errors.extend(_executor_errors(executor_env))
    if int(control_env.get("max_rungs") or 0) < 1:
        errors.append("GPU_FAULT_BRANCH_ESCALATION_MAX_RUNGS is below 1")
    errors.extend(budget_headroom_errors(budget))
    errors.extend(verify_attempt_errors(verify_max_attempts))
    errors.extend(
        lifetime_errors(
            estimated_seconds=estimated_duration_seconds(
                verify_max_attempts=verify_max_attempts,
                poll_interval_seconds=float(
                    control_env.get("poll_interval_seconds") or 5.0
                ),
            ),
            lifetime_seconds=control_env.get("node_lifetime_seconds"),
        )
    )
    return errors


def window_identity_errors(
    before: dict[str, Any],
    after: dict[str, Any],
    *,
    deployment: str,
    generation_delta: int,
    allow_template_change: bool = False,
) -> list[str]:
    """Judge a runtime identity across the executor env window's rollouts.

    ``verify_runtime_identity`` compares the whole document, generation
    counters included, so it cannot span a window that rolls ``deployment`` on
    purpose (DESTR-018 carries the same rule for the control worker). Every
    other Deployment and the release state must be identical. While the window
    is open (``allow_template_change``) the managed Deployment's template digest
    is expected to differ; after the close it must be back to its pre-window
    value, and its generation may have moved only by ``generation_delta``.
    """

    errors: list[str] = []
    if after.get("release_state") != before.get("release_state"):
        errors.append("regional release state identity drifted")
    counters = ("generation", "observed_generation")
    volatile = (*counters, "template_sha256") if allow_template_change else counters
    for plane, deployments in (before.get("deployments") or {}).items():
        current_plane = (after.get("deployments") or {}).get(plane) or {}
        for name, expected in deployments.items():
            observed = current_plane.get(name)
            if observed is None:
                errors.append(f"{plane} deployment {name} disappeared")
                continue
            if name != deployment:
                if observed != expected:
                    errors.append(f"{plane} deployment {name} identity drifted")
                continue
            stable = {k: v for k, v in observed.items() if k not in volatile}
            if stable != {k: v for k, v in expected.items() if k not in volatile}:
                errors.append(
                    f"{deployment} differs from the pre-window baseline in "
                    "something other than its generation"
                )
            expected_generation = int(expected["generation"]) + generation_delta
            for counter in counters:
                if int(observed.get(counter, -1)) != expected_generation:
                    errors.append(
                        f"{deployment} {counter} is {observed.get(counter)}, "
                        f"expected {expected_generation} after the window's rollouts"
                    )
    return errors

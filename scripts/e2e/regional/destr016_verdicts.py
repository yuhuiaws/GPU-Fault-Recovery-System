"""Pure verdict functions and constants of GF-REGIONAL-DESTR-016.

Split out of ``run_destr016_preempting_reboot.py`` so the runner stays a
driver: everything here is unit-tested against synthetic control-plane, node
and CloudTrail snapshots and touches no cluster.

The case proves the escalation boundary on one node, in the middle of a
physical step:

* a RESET_GPU workflow is parked at ``VERIFY_NO_GPU_CLIENTS`` WAITING with the
  quiesce applied and not yet restored (a GPU device holder keeps a client
  alive), which is the *dirty* boundary;
* a second XID 46 of the same rank is absorbed onto the same incident and the
  same workflow -- no second workflow, no extra steps;
* an XID 79 of a strictly stronger rank (RESTART_NODE, recovery rank 50, over
  RESET_GPU's 30) preempts it: the reset workflow goes SUPERSEDED, its WAITING
  remote command is cancelled, and the successor reboot workflow adopts the
  quiesce handoff, reboots the node for real and restores services after the
  reboot.

Two contracts encoded here that prose could drift from, both derived from the
code rather than from the plan:

* the successor's step graph is *rewired* at claim time, not appended to.
  ``executor._adopt_quiesce_handoff_from_predecessor`` appends one
  ``RESTORE_GPU_SERVICES`` step carrying
  ``preemption_quiesce_handoff_after_reboot``, points it at ``RESTART_NODE``
  and repoints ``VALIDATE_GPU`` at *it*, so the last index of the successor is
  the restore and the first validation depends on the last step. Asserting the
  operation list in plain order would fail on a correct run.
* the predecessor's ``VERIFY_NO_GPU_CLIENTS`` *step execution* stays WAITING
  for ever; it is the *remote command* that is cancelled
  (``status_source="workflow-preempted"``). A verdict that expected the step
  record to become FAILED would fail on a correct run.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

# The reset workflow the first XID 46 must build, in order, exactly once.
RESET_OPERATIONS = (
    "FREEZE_EVIDENCE",
    "MARK_UNSCHEDULABLE",
    "QUIESCE_GPU_SERVICES",
    "VERIFY_NO_GPU_CLIENTS",
    "RESET_GPU",
    "RESTORE_GPU_SERVICES",
    "VALIDATE_GPU",
    "RESTORE_SCHEDULING",
)
# The successor reboot workflow, in the index order the claim-time handoff
# leaves behind: the appended restore is the last index, and VALIDATE_GPU
# depends on it rather than on RESTART_NODE.
SUCCESSOR_OPERATIONS = (
    "FREEZE_EVIDENCE",
    "MARK_UNSCHEDULABLE",
    "RESTART_NODE",
    "VALIDATE_GPU",
    "VALIDATE_HOST",
    "VALIDATE_FABRIC",
    "RESTORE_SCHEDULING",
    "RESTORE_GPU_SERVICES",
)
# Steps the successor inherits as already done instead of running again.
INHERITED_OPERATIONS = ("MARK_UNSCHEDULABLE",)
HANDOFF_PARAMETER = "preemption_quiesce_handoff_after_reboot"
# The step that must be parked WAITING when the escalation arrives, and the
# step that must never have succeeded on the pre-reboot boot.
BARRIER_OPERATION = "VERIFY_NO_GPU_CLIENTS"
FORBIDDEN_LEDGER_OPERATIONS = ("RESET_GPU", "RESET_ALL_GPUS_NVSWITCHES")
# Node Agent operations the node must advertise before the case starts.
# Node Agent operations the case dispatches to the node. VALIDATE_GPU is not
# one of them: it runs through the GPU_VALIDATION adapter, so an Agent's
# allowed_operations never lists it (live 2026-09-08, the preflight refused
# every node for that reason).
AGENT_OPERATIONS = (
    "QUIESCE_GPU_SERVICES",
    "VERIFY_NO_GPU_CLIENTS",
    "RESTORE_GPU_SERVICES",
)
VALIDATION_OPERATION = "VALIDATE_GPU"
RESET_ACTION = "RESET_GPU"
REBOOT_ACTION = "RESTART_NODE"
# The policy and the incident speak in RecoveryAction terms: XID 79 resolves to
# RESTART_BM, which the planner realises as the RESTART_NODE operation.
REBOOT_RECOVERY_ACTION = "RESTART_BM"
# Containment the successor may carry over as already done. MARK_UNSCHEDULABLE
# is required (the cordon is inherited, not redone); FREEZE_EVIDENCE is fine;
# anything else -- a quiesce, a reset -- would mean the handoff was skipped.
ALLOWED_INHERITED_OPERATIONS = ("FREEZE_EVIDENCE", "MARK_UNSCHEDULABLE")
FIRST_XID = 46
ESCALATION_XID = 79
CANCELLED_STATUS_SOURCE = "workflow-preempted"
PREEMPTION_REASON_SUBSTRING = "strictly stronger recovery action"
WAITING_REASON_SUBSTRING = "clients are still active"
QUARANTINE_TAINT_PREFIX = "gpu-fault.io/"
REBOOT_EVENTS = ("BatchRebootClusterNodes", "RebootClusterNodes")
FORBIDDEN_EVENTS = (
    "BatchDeleteClusterNodes",
    "BatchReplaceClusterNodes",
    "DeleteClusterNodes",
    "ReplaceClusterNodes",
)

# Wall-clock allowances (seconds) for the lifetime arithmetic. The successor
# inherits the predecessor's ``lifetime_deadline_at`` (F-N1 is per incident,
# not per workflow), so containment, the WAITING park in which both later
# faults are injected, the real reboot and the validation tail all have to fit
# inside ONE node workflow lifetime. Deliberately generous.
CONTAINMENT_ALLOWANCE_SECONDS = 300
WAITING_PARK_ALLOWANCE_SECONDS = 240
REBOOT_ALLOWANCE_SECONDS = 1200
VALIDATION_ALLOWANCE_SECONDS = 600
# The barrier step's own ceiling is ``GPU_FAULT_WORKFLOW_STEP_TIMEOUT_SECONDS``
# (600s by default, and VERIFY_NO_GPU_CLIENTS takes no managed-recovery
# override). Both escalation injections have to land inside it, so a deployed
# ceiling below this leaves no room for the two writes plus collector latency.
# Kept above ``WAITING_PARK_ALLOWANCE_SECONDS`` on purpose: the park the case
# plans for has to fit inside the smallest ceiling it will accept, which is why
# ``step_timeout_errors`` needs only the one comparison.
MIN_STEP_TIMEOUT_SECONDS = 300


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def _parse(value: Any) -> datetime | None:
    if not value:
        return None
    text = str(value).replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def operations(workflow: dict[str, Any]) -> list[str]:
    return [str(step.get("operation")) for step in workflow.get("official_steps") or []]


def _step_index(workflow: dict[str, Any], operation: str) -> int | None:
    indexes = [
        index for index, name in enumerate(operations(workflow)) if name == operation
    ]
    return indexes[0] if len(indexes) == 1 else None


def executions_of(workflow: dict[str, Any], operation: str) -> list[dict[str, Any]]:
    return [
        item
        for item in workflow.get("step_executions") or []
        if item.get("operation") == operation
    ]


def _statuses(workflow: dict[str, Any], operation: str) -> list[str]:
    return [str(item.get("status")) for item in executions_of(workflow, operation)]


def _succeeded(workflow: dict[str, Any], operation: str) -> bool:
    return "SUCCEEDED" in _statuses(workflow, operation)


def added_ledger_rows(
    baseline: list[dict[str, Any]],
    after: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Ledger rows present after the case that the baseline did not have.

    Keyed by ``(command_id, attempt)`` rather than ``command_id`` alone: the
    barrier step retries under one command id, so each attempt is its own row
    and dropping the attempt would hide every retry but the first.
    """

    seen = {(row.get("command_id"), row.get("attempt")) for row in baseline}
    return [
        row for row in after if (row.get("command_id"), row.get("attempt")) not in seen
    ]


# --------------------------------------------------------------------------- #
# Phase 1: the reset workflow parks at the dirty boundary
# --------------------------------------------------------------------------- #
def reset_workflow_errors(
    workflow: dict[str, Any],
    incident: dict[str, Any],
    decision: dict[str, Any],
) -> list[str]:
    """The state the escalation must arrive into: a reset workflow whose
    quiesce is applied and whose barrier step is parked WAITING."""

    errors: list[str] = []
    if decision.get("official_action", decision.get("action")) != RESET_ACTION:
        errors.append(
            "policy did not resolve the first XID 46 to RESET_GPU: "
            f"{decision.get('official_action', decision.get('action'))}"
        )
    if incident.get("official_action") != RESET_ACTION:
        errors.append(
            f"incident official action is not RESET_GPU: {incident.get('official_action')}"
        )
    if list(operations(workflow)) != list(RESET_OPERATIONS):
        errors.append(f"reset workflow steps are not {list(RESET_OPERATIONS)}")
    if workflow.get("status") not in {"RUNNING", "PENDING"}:
        errors.append(
            f"reset workflow is not still running at the boundary: {workflow.get('status')}"
        )
    errors.extend(waiting_boundary_errors(workflow))
    return errors


def waiting_boundary_errors(workflow: dict[str, Any]) -> list[str]:
    """The dirty boundary itself, asserted only where the code guarantees it.

    ``QUIESCE_GPU_SERVICES`` has succeeded, ``RESTORE_GPU_SERVICES`` has not
    run, and the barrier step is WAITING on a remote command that reports
    clients are still active. RESET_GPU has not run at all: a single-node reset
    cannot itself park WAITING (the client-verification retry loop only exists
    on the multi-node reset path), so the boundary this case preempts is the
    barrier's, not the reset's.
    """

    errors: list[str] = []
    if not _succeeded(workflow, "QUIESCE_GPU_SERVICES"):
        errors.append("QUIESCE_GPU_SERVICES has not succeeded; the boundary is clean")
    if executions_of(workflow, "RESTORE_GPU_SERVICES"):
        errors.append("RESTORE_GPU_SERVICES already ran; the quiesce is not unrestored")
    barrier = executions_of(workflow, BARRIER_OPERATION)
    if len(barrier) != 1:
        errors.append(
            f"{BARRIER_OPERATION} does not have exactly one execution: {len(barrier)}"
        )
    elif barrier[0].get("status") != "WAITING":
        errors.append(f"{BARRIER_OPERATION} is not WAITING: {barrier[0].get('status')}")
    else:
        details = barrier[0].get("details") or {}
        if details.get("remote_status") != "WAITING":
            errors.append(
                "the barrier step's remote command is not WAITING: "
                f"{details.get('remote_status')}"
            )
        if not details.get("remote_command_id"):
            errors.append("the barrier step carries no remote command id")
    if executions_of(workflow, "RESET_GPU"):
        errors.append("RESET_GPU already ran; the holder did not park the barrier")
    return errors


def barrier_commands(
    commands: list[dict[str, Any]], workflow: dict[str, Any] | None = None
) -> list[dict[str, Any]]:
    """The remote command that carries the client-verification barrier.

    Since 6e0248c the executor runs the agent steps of a reset as one compound
    command labelled by its first step (QUIESCE_GPU_SERVICES, step_index 2),
    and the VERIFY_NO_GPU_CLIENTS hold lives in that command's
    ``result_details`` (``gpu_client_quiesce_attempt``, the reason). Selecting
    by operation therefore found nothing on 2026-09-14 while the barrier stood
    for four minutes. The barrier step execution names the command it waits on
    (``details.remote_command_id``); that id wins, then a command whose hold
    details record the verification attempt, then the pre-batching per-step
    operation match.
    """

    matches: list[dict[str, Any]] = []
    remote_command_id = None
    for execution in executions_of(workflow or {}, BARRIER_OPERATION):
        remote_command_id = (execution.get("details") or {}).get("remote_command_id")
    for item in commands:
        operation = (item.get("step") or {}).get("operation") or item.get("operation")
        details = item.get("result_details") or {}
        if remote_command_id and item.get("command_id") == remote_command_id:
            return [item]
        if operation == BARRIER_OPERATION or "gpu_client_quiesce_attempt" in details:
            matches.append(item)
    return matches


def barrier_reason_errors(
    commands: list[dict[str, Any]], workflow: dict[str, Any] | None = None
) -> list[str]:
    """The WAITING command must say *why* it waits, and it must be the holder.

    Recorded from ``result_details`` rather than inferred: an empty reason
    would mean the barrier parked for some other cause and the case would be
    proving a boundary it did not create.
    """

    barrier = barrier_commands(commands, workflow)
    if len(barrier) != 1:
        return [f"there is not exactly one barrier remote command: {len(barrier)}"]
    details = barrier[0].get("result_details") or {}
    reason = str(details.get("reason") or "")
    errors: list[str] = []
    if barrier[0].get("status") != "WAITING":
        errors.append(
            f"barrier remote command is not WAITING: {barrier[0].get('status')}"
        )
    if WAITING_REASON_SUBSTRING not in reason:
        errors.append(
            f"barrier command does not report {WAITING_REASON_SUBSTRING!r}: {reason!r}"
        )
    if int(details.get("gpu_client_quiesce_attempt") or 0) < 1:
        errors.append("barrier command recorded no client-verification attempt")
    return errors


# --------------------------------------------------------------------------- #
# Phase A: the same-rank second fault is absorbed
# --------------------------------------------------------------------------- #
def absorb_errors(
    before: dict[str, Any],
    after: dict[str, Any],
    *,
    node: str,
) -> list[str]:
    """``before``/``after`` are store snapshots taken around the second XID 46.

    What the merge code guarantees is the same for both dispositions it can
    reach here -- ``ABSORB`` when the second event carries the same GPU UUID,
    ``WIDEN_IN_PLACE`` when it carries none or another GPU -- so the verdict
    asserts the observables they share and never the label: one incident, one
    workflow row, one request id, an unchanged operation list, no new step
    execution, and one more recorded reason.
    """

    errors: list[str] = []
    first_workflow = before.get("workflow") or {}
    second_workflow = after.get("workflow") or {}
    first_incident = before.get("incident") or {}
    second_incident = after.get("incident") or {}
    if not first_workflow.get("request_id"):
        errors.append("no workflow was observed before the absorbed fault")
    if second_workflow.get("request_id") != first_workflow.get("request_id"):
        errors.append(
            "the absorbed XID 46 created a second workflow: "
            f"{first_workflow.get('request_id')} -> {second_workflow.get('request_id')}"
        )
    if second_incident.get("incident_id") != first_incident.get("incident_id"):
        errors.append(
            "the absorbed XID 46 created a second incident: "
            f"{first_incident.get('incident_id')} -> {second_incident.get('incident_id')}"
        )
    if operations(second_workflow) != operations(first_workflow):
        errors.append(
            "the absorbed XID 46 changed the step list: "
            f"{operations(first_workflow)} -> {operations(second_workflow)}"
        )
    if second_incident.get("official_action") != RESET_ACTION:
        errors.append(
            "the absorbed XID 46 changed the official action: "
            f"{second_incident.get('official_action')}"
        )
    before_executions = len(first_workflow.get("step_executions") or [])
    after_executions = len(second_workflow.get("step_executions") or [])
    if after_executions != before_executions:
        errors.append(
            "the absorbed XID 46 added a step execution: "
            f"{before_executions} -> {after_executions}"
        )
    errors.extend(
        incident_reason_errors(
            first_incident,
            second_incident,
            node=node,
            xid=FIRST_XID,
        )
    )
    event = after.get("event") or {}
    if event.get("xid") != FIRST_XID:
        errors.append(f"the absorbed event is not XID {FIRST_XID}: {event.get('xid')}")
    return errors


def incident_reason_errors(
    before: dict[str, Any],
    after: dict[str, Any],
    *,
    node: str,
    xid: int,
) -> list[str]:
    """A merged fault has to leave a trace on the incident it joined."""

    first = list(before.get("reasons") or [])
    second = list(after.get("reasons") or [])
    if len(second) <= len(first):
        errors = [f"incident reasons did not grow for the XID {xid} fault: {second}"]
        return errors
    added = second[len(first) :]
    if not any(f"XID {xid}" in str(item) and node in str(item) for item in added):
        return [f"no incident reason records the XID {xid} fault on {node}: {added}"]
    return []


# --------------------------------------------------------------------------- #
# Phase B: the stronger fault preempts and hands off the quiesce
# --------------------------------------------------------------------------- #
def escalation_errors(
    predecessor: dict[str, Any],
    successor: dict[str, Any],
    incident: dict[str, Any],
    *,
    decision: dict[str, Any],
) -> list[str]:
    """The preemption itself: one incident, two workflows, one direction."""

    errors: list[str] = []
    if decision.get("official_action", decision.get("action")) not in {
        REBOOT_RECOVERY_ACTION,
        REBOOT_ACTION,
    }:
        errors.append(
            f"policy did not resolve XID 79 to {REBOOT_RECOVERY_ACTION}: "
            f"{decision.get('official_action', decision.get('action'))}"
        )
    predecessor_id = str(predecessor.get("request_id") or "")
    successor_id = str(successor.get("request_id") or "")
    if not predecessor_id or not successor_id:
        return errors + ["the predecessor or successor workflow is unknown"]
    if predecessor_id == successor_id:
        return errors + ["the escalation reused the reset workflow row"]
    if predecessor.get("status") != "SUPERSEDED":
        errors.append(
            f"the reset workflow is not SUPERSEDED: {predecessor.get('status')}"
        )
    if predecessor.get("preempted_by_workflow_id") != successor_id:
        errors.append(
            "the reset workflow does not point at its successor: "
            f"{predecessor.get('preempted_by_workflow_id')}"
        )
    if successor.get("predecessor_workflow_id") != predecessor_id:
        errors.append(
            "the successor does not point back at the reset workflow: "
            f"{successor.get('predecessor_workflow_id')}"
        )
    if successor.get("incident_id") != predecessor.get("incident_id"):
        errors.append("the successor belongs to another incident")
    if incident.get("workflow_request_id") != successor_id:
        errors.append(
            "the incident still points at the superseded workflow: "
            f"{incident.get('workflow_request_id')}"
        )
    if incident.get("official_action") not in {REBOOT_RECOVERY_ACTION, REBOOT_ACTION}:
        errors.append(
            f"the incident action is not {REBOOT_RECOVERY_ACTION}: "
            f"{incident.get('official_action')}"
        )
    reason = str(successor.get("preemption_reason") or "")
    if PREEMPTION_REASON_SUBSTRING not in reason:
        errors.append(
            f"the successor does not record {PREEMPTION_REASON_SUBSTRING!r}: {reason!r}"
        )
    inherited = sorted(successor.get("completed_operations") or [])
    if not set(INHERITED_OPERATIONS) <= set(inherited) or not set(inherited) <= set(
        ALLOWED_INHERITED_OPERATIONS
    ):
        errors.append(
            f"the successor did not inherit {list(INHERITED_OPERATIONS)} (and only "
            f"containment from {list(ALLOWED_INHERITED_OPERATIONS)}): {inherited}"
        )
    if not successor.get("inherited_step_indexes"):
        errors.append("the successor records no inherited step indexes")
    return errors


def successor_step_graph_errors(successor: dict[str, Any]) -> list[str]:
    """The handoff wiring the claim-time adoption leaves behind."""

    errors: list[str] = []
    if list(operations(successor)) != list(SUCCESSOR_OPERATIONS):
        errors.append(f"successor steps are not {list(SUCCESSOR_OPERATIONS)}")
        return errors
    if not successor.get("dag_enabled"):
        errors.append("the successor is not dag_enabled; the handoff was not rewired")
    if successor.get("quiesce_handoff_from_workflow_id") != successor.get(
        "predecessor_workflow_id"
    ):
        errors.append(
            "the successor does not record the quiesce handoff source: "
            f"{successor.get('quiesce_handoff_from_workflow_id')}"
        )
    steps = successor.get("official_steps") or []
    restore_index = _step_index(successor, "RESTORE_GPU_SERVICES")
    restart_index = _step_index(successor, "RESTART_NODE")
    validate_index = _step_index(successor, "VALIDATE_GPU")
    if restore_index is None or restart_index is None or validate_index is None:
        return errors + ["the successor does not have a unique handoff step trio"]
    if restore_index != len(steps) - 1:
        errors.append(
            f"the handoff restore is not the last step: index {restore_index}"
        )
    parameters = (steps[restore_index].get("parameters") or {}) if steps else {}
    if not parameters.get(HANDOFF_PARAMETER):
        errors.append(f"the handoff restore does not carry {HANDOFF_PARAMETER}")
    if not parameters.get("handoff_from_workflow_id"):
        errors.append("the handoff restore does not name the workflow it inherits from")
    restore_dependencies = list(
        steps[restore_index].get("depends_on_step_indexes") or []
    )
    if restore_dependencies != [restart_index]:
        errors.append(
            "the handoff restore does not depend on RESTART_NODE alone: "
            f"{restore_dependencies}"
        )
    validate_dependencies = list(
        steps[validate_index].get("depends_on_step_indexes") or []
    )
    if validate_dependencies != [restore_index]:
        errors.append(
            "VALIDATE_GPU was not repointed at the handoff restore: "
            f"{validate_dependencies}"
        )
    return errors


def superseded_predecessor_errors(predecessor: dict[str, Any]) -> list[str]:
    """What the superseded reset must and must not have done.

    Its barrier step record stays WAITING for ever -- nothing rewrites a
    WAITING execution on supersession -- and no reset or restore ever ran on
    it. Asserting a FAILED barrier record here would fail a correct run.
    """

    errors: list[str] = []
    barrier = executions_of(predecessor, BARRIER_OPERATION)
    if len(barrier) != 1 or barrier[0].get("status") != "WAITING":
        errors.append(
            "the superseded reset's barrier record is not a single WAITING row: "
            f"{[item.get('status') for item in barrier]}"
        )
    for operation in ("RESET_GPU", "RESTORE_GPU_SERVICES", "RESTORE_SCHEDULING"):
        if executions_of(predecessor, operation):
            errors.append(f"the superseded reset ran {operation}")
    if not _succeeded(predecessor, "QUIESCE_GPU_SERVICES"):
        errors.append("the superseded reset has no successful QUIESCE_GPU_SERVICES")
    return errors


def cancelled_command_errors(
    commands: list[dict[str, Any]],
    *,
    successor_request_id: str,
    workflow: dict[str, Any] | None = None,
) -> list[str]:
    """The predecessor's WAITING remote command must be cancelled, once, by
    the preemption -- and no reset command may have been issued at all."""

    errors: list[str] = []
    for item in commands:
        operation = (item.get("step") or {}).get("operation")
        if operation in FORBIDDEN_LEDGER_OPERATIONS:
            errors.append(f"the superseded reset issued a {operation} command")
    barrier = barrier_commands(commands, workflow)
    if len(barrier) != 1:
        return errors + [
            f"there is not exactly one barrier remote command: {len(barrier)}"
        ]
    command = barrier[0]
    if command.get("status") != "FAILED":
        errors.append(
            f"the barrier remote command is not FAILED: {command.get('status')}"
        )
    if command.get("status_source") != CANCELLED_STATUS_SOURCE:
        errors.append(
            "the barrier remote command was not cancelled by the preemption: "
            f"{command.get('status_source')}"
        )
    error = str(command.get("error") or "")
    if successor_request_id and successor_request_id not in error:
        errors.append(
            f"the cancellation does not name the successor workflow: {error!r}"
        )
    return errors


# --------------------------------------------------------------------------- #
# Terminal state
# --------------------------------------------------------------------------- #
def terminal_errors(
    state: dict[str, Any],
    *,
    expected_boot_id: str | None,
    expected_artifact: str | None,
) -> list[str]:
    """The successor reached a real, validated, restored recovery."""

    errors: list[str] = []
    workflow = state.get("workflow") or {}
    incident = state.get("incident") or {}
    submission = state.get("submission") or {}
    agent = state.get("agent") or {}
    if workflow.get("status") != "SUCCEEDED":
        errors.append(
            f"the successor workflow is not SUCCEEDED: {workflow.get('status')}"
        )
    if incident.get("state") != "RECOVERED":
        errors.append(f"the incident is not RECOVERED: {incident.get('state')}")
    if workflow.get("terminal_failure_reason"):
        errors.append(
            f"the successor carries a failure reason: {workflow.get('terminal_failure_reason')}"
        )
    for operation in (
        "RESTART_NODE",
        "RESTORE_GPU_SERVICES",
        "VALIDATE_GPU",
        "VALIDATE_HOST",
        "VALIDATE_FABRIC",
        "RESTORE_SCHEDULING",
    ):
        statuses = _statuses(workflow, operation)
        if statuses[-1:] != ["SUCCEEDED"]:
            errors.append(f"{operation} did not end SUCCEEDED: {statuses}")
    if submission.get("state") != "SUBMITTED":
        errors.append(
            f"the HyperPod submission is not SUBMITTED: {submission.get('state')}"
        )
    if submission.get("action") != "REBOOT":
        errors.append(
            f"the HyperPod submission action is not REBOOT: {submission.get('action')}"
        )
    if expected_boot_id and agent.get("boot_id") == expected_boot_id:
        errors.append("the Node Agent boot id did not change; no real reboot happened")
    if expected_artifact and agent.get("artifact_sha256") != expected_artifact:
        errors.append("the Node Agent artifact changed across the reboot")
    if agent.get("lifecycle_state") != "ACTIVE":
        errors.append(
            f"the Node Agent did not return ACTIVE: {agent.get('lifecycle_state')}"
        )
    return errors


def _succeeded_executions(
    workflow: dict[str, Any],
    operation: str,
) -> list[dict[str, Any]]:
    return [
        item
        for item in executions_of(workflow, operation)
        if item.get("status") == "SUCCEEDED"
    ]


def restore_after_reboot_errors(successor: dict[str, Any]) -> list[str]:
    """The handoff restore has to run *after* the reboot, not before it.

    Only the SUCCEEDED rows are compared: a managed recovery parks WAITING once
    per poll before it completes, so ``RESTART_NODE`` legitimately has several
    executions and requiring exactly one would fail every real reboot.
    """

    restart = _succeeded_executions(successor, "RESTART_NODE")
    restore = _succeeded_executions(successor, "RESTORE_GPU_SERVICES")
    if len(restart) != 1 or len(restore) != 1:
        return [
            "the successor does not have exactly one successful RESTART_NODE and "
            f"one successful RESTORE_GPU_SERVICES: {len(restart)}, {len(restore)}"
        ]
    ended = _parse(restart[0].get("updated_at"))
    started = _parse(restore[0].get("started_at"))
    if ended is None or started is None:
        return ["the reboot/restore timestamps are incomplete"]
    if started < ended:
        return [
            "the handoff restore started before the reboot finished: "
            f"{started.isoformat()} < {ended.isoformat()}"
        ]
    return []


# --------------------------------------------------------------------------- #
# Data-plane verdicts
# --------------------------------------------------------------------------- #
def host_errors(
    baseline: dict[str, Any],
    after: dict[str, Any],
    *,
    expected_gpu_count: int,
) -> list[str]:
    """The node's own record of what happened, from the Agent ledger and
    ``/proc``: a real reboot, a quiesce, a post-reboot restore -- and no reset
    success anywhere, on either boot."""

    errors: list[str] = []
    if not baseline.get("boot_id") or not after.get("boot_id"):
        errors.append("a host boot id snapshot is missing")
    elif baseline.get("boot_id") == after.get("boot_id"):
        errors.append("the host boot id did not change; the node did not reboot")
    added = added_ledger_rows(
        list(baseline.get("ledger") or []), list(after.get("ledger") or [])
    )
    for operation in FORBIDDEN_LEDGER_OPERATIONS:
        succeeded = [
            row
            for row in added
            if row.get("operation") == operation and row.get("state") == "SUCCEEDED"
        ]
        if succeeded:
            errors.append(
                f"the Node Agent ledger shows a successful {operation}: "
                f"{[row.get('command_id') for row in succeeded]}"
            )
    # VALIDATE_GPU runs through the GPU_VALIDATION adapter, not the Node Agent,
    # so it never appears in the ledger; its success is a workflow step check.
    for operation in ("QUIESCE_GPU_SERVICES", "RESTORE_GPU_SERVICES"):
        succeeded = [
            row
            for row in added
            if row.get("operation") == operation and row.get("state") == "SUCCEEDED"
        ]
        if not succeeded:
            errors.append(f"the Node Agent ledger has no successful {operation}")
    barrier = [row for row in added if row.get("operation") == BARRIER_OPERATION]
    if not barrier:
        errors.append(f"the Node Agent ledger has no {BARRIER_OPERATION} row")
    elif all(row.get("state") == "SUCCEEDED" for row in barrier):
        errors.append(
            "every client-verification attempt succeeded; the holder never held"
        )
    if len(after.get("gpu_inventory") or []) != expected_gpu_count:
        errors.append(
            f"the host GPU inventory is not {expected_gpu_count} after the reboot: "
            f"{len(after.get('gpu_inventory') or [])}"
        )
    if after.get("quiesce_states"):
        errors.append(
            f"a GPU quiesce state survives the handoff restore: {after['quiesce_states']}"
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
    return errors


def holder_errors(status: dict[str, Any]) -> list[str]:
    """The holder must have held, must not have lost the arming race, and must
    be gone after the reboot took it with it."""

    errors: list[str] = []
    if status.get("holder_error"):
        errors.append(f"the GPU holder reported an error: {status.get('holder_error')}")
    if status.get("arm_race_lost"):
        errors.append(
            "the client verification succeeded before the holder started; the "
            "reset committed and there was no dirty boundary to preempt"
        )
    if not status.get("hold_started_at"):
        errors.append("the GPU holder never started")
    if not status.get("matched_row"):
        errors.append("the GPU holder never matched a ledger row")
    active = str((status.get("unit_state") or {}).get("ActiveState") or "")
    if active == "active":
        errors.append("the GPU holder is still active after the reboot")
    return errors


def schedulability_errors(snapshot: dict[str, Any], *, node: str) -> list[str]:
    errors: list[str] = []
    if snapshot.get("ready") != "True":
        errors.append(f"{node} is not Ready after the case")
    if snapshot.get("unschedulable"):
        errors.append(f"{node} is not schedulable after RESTORE_SCHEDULING")
    taints = [
        taint.get("key")
        for taint in snapshot.get("taints") or []
        if str(taint.get("key") or "").startswith(QUARANTINE_TAINT_PREFIX)
    ]
    if taints:
        errors.append(f"{node} still carries a gpu-fault taint: {taints}")
    if snapshot.get("ownership_annotations"):
        errors.append(
            f"{node} still carries an ownership annotation: "
            f"{sorted(snapshot['ownership_annotations'])}"
        )
    return errors


def node_recovery_errors(
    baseline: dict[str, Any],
    first_ready: dict[str, Any],
    final: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    if first_ready.get("boot_id") == baseline.get("boot_id"):
        errors.append("the Kubernetes Node boot id did not change")
    if first_ready.get("uid") != baseline.get("uid") or final.get(
        "uid"
    ) != baseline.get("uid"):
        errors.append("the Kubernetes Node UID changed across the reboot")
    if final.get("gpu_allocatable") != baseline.get("gpu_allocatable"):
        errors.append("GPU capacity did not return to baseline after validation")
    return errors


def reboot_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [item for item in events if item.get("event_name") in REBOOT_EVENTS]


def forbidden_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [item for item in events if item.get("event_name") in FORBIDDEN_EVENTS]


def provider_errors(
    events: list[dict[str, Any]],
    *,
    actor_matches_role: bool | None,
) -> list[str]:
    """Exactly one reboot call, made by the executor role, and nothing else.

    ``actor_matches_role`` is supplied by the caller (the shared
    ``provider_event_actor_matches_role`` helper needs the role ARN, which is
    site topology this module must not carry) and is ``None`` when there is no
    unique event to attribute.
    """

    errors: list[str] = []
    reboots = reboot_events(events)
    if len(reboots) != 1:
        errors.append(
            "CloudTrail does not contain exactly one reboot event: "
            f"{[item.get('event_name') for item in reboots]}"
        )
    elif actor_matches_role is not True:
        errors.append("the CloudTrail reboot actor is not the executor role")
    forbidden = forbidden_events(events)
    if forbidden:
        errors.append(
            "CloudTrail contains a replace/delete mutation: "
            f"{sorted({str(item.get('event_name')) for item in forbidden})}"
        )
    return errors


# --------------------------------------------------------------------------- #
# Timeline and arithmetic
# --------------------------------------------------------------------------- #
def step_transitions(
    previous: dict[str, str],
    executions: list[dict[str, Any]],
) -> tuple[dict[str, str], list[dict[str, Any]]]:
    """Fold step executions into ``{index/operation#occurrence: status}``;
    return the new state and only the entries whose status changed.

    The occurrence ordinal is part of the key on purpose. A step that parks
    WAITING before it succeeds keeps *both* rows in ``step_executions``, and a
    key of index/operation alone would see the pair flip status on every poll
    and append the same two "changes" for ever.
    """

    state = dict(previous)
    changes: list[dict[str, Any]] = []
    now = datetime.now(timezone.utc).isoformat()
    seen: dict[str, int] = {}
    for item in executions:
        step = f"{item.get('step_index')}/{item.get('operation')}"
        occurrence = seen.get(step, 0)
        seen[step] = occurrence + 1
        key = f"{step}#{occurrence}"
        status = str(item.get("status") or "")
        if state.get(key) == status:
            continue
        state[key] = status
        changes.append(
            {
                "observed_at": now,
                "step": step,
                "occurrence": occurrence,
                "status": status,
                "error": item.get("error"),
                "started_at": item.get("started_at"),
                "updated_at": item.get("updated_at"),
            }
        )
    return state, changes


def estimated_duration_seconds() -> int:
    return (
        CONTAINMENT_ALLOWANCE_SECONDS
        + WAITING_PARK_ALLOWANCE_SECONDS
        + REBOOT_ALLOWANCE_SECONDS
        + VALIDATION_ALLOWANCE_SECONDS
    )


def lifetime_errors(
    *,
    estimated_seconds: int,
    lifetime_seconds: int | None,
) -> list[str]:
    """The successor inherits the predecessor's lifetime deadline, so the whole
    case has to fit one node workflow lifetime rather than two."""

    if lifetime_seconds is None:
        return ["node workflow lifetime is unknown; cannot bound the case duration"]
    if estimated_seconds >= lifetime_seconds:
        return [
            f"estimated duration {estimated_seconds}s does not fit the "
            f"{lifetime_seconds}s node workflow lifetime the successor inherits"
        ]
    return []


def step_timeout_errors(*, step_timeout_seconds: int | None) -> list[str]:
    """Both escalation injections have to land inside the barrier step's own
    waiting ceiling, which is the plain step timeout for this operation."""

    if step_timeout_seconds is None:
        return ["the deployed workflow step timeout is unknown"]
    if step_timeout_seconds < MIN_STEP_TIMEOUT_SECONDS:
        return [
            f"the workflow step timeout {step_timeout_seconds}s is below "
            f"{MIN_STEP_TIMEOUT_SECONDS}s; the barrier cannot stay WAITING long "
            "enough for both escalation injections"
        ]
    return []

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from gpu_fault.models import (
    FaultIncident,
    IncidentState,
    PlanStatus,
    RecoveryPlan,
    WorkflowOperation,
    WorkflowRequest,
    WorkflowStatus,
    WorkflowStepStatus,
    bounded_reasons,
)
from gpu_fault.operation_registry import (
    CONTAINMENT_ONLY_OPERATIONS,
    NODE_MUTATING_OPERATIONS,
)
from gpu_fault.retired_generation import (
    OPEN_REMOTE_STATUSES,
    RETIRED_GENERATION_STATUSES,
    retired_generation_audit,
    retired_generation_reasons,
    retired_generation_records,
    retired_generation_successor,
)
from gpu_fault.store import NotFoundError
from gpu_fault.store.contracts import WorkflowStore
from gpu_fault.store.shared.errors import StaleWriteError

if TYPE_CHECKING:
    from gpu_fault.regional import RemoteActionCommand

# Re-exported so callers keep one import site for workflow-resolution rules.
# ``gpu_fault.retired_generation`` is a separate module only because the admin
# CLI ships its source into a Pod running the previously deployed image, which
# constrains what it may import; see that module's docstring.
__all__ = [
    "OPEN_REMOTE_STATUSES",
    "RETIRED_GENERATION_STATUSES",
    "SETTLED_INCIDENT_STATES",
    "abandoned_generation_successor",
    "completed_containment_operations",
    "reconciled_restore_records",
    "restore_reconciliation_reasons",
    "retired_generation_audit",
    "retired_generation_reasons",
    "retired_generation_records",
    "retired_generation_successor",
    "retirement_fences_out_dispatch",
    "verified_restore_successor",
    "workflow_never_changed_a_node",
]

# Incident states in which nobody is waiting on the incident's workflow to act:
# it recovered, or it was handed to an operator. A BLOCKED workflow puts its
# incident in ESCALATED (``coordinator._incident_state_for_workflow``), which is
# the state the operator running ``workflow-reconcile`` is answering.
SETTLED_INCIDENT_STATES = frozenset({IncidentState.RECOVERED, IncidentState.ESCALATED})


def verified_restore_successor(
    store: WorkflowStore,
    workflow: WorkflowRequest,
) -> WorkflowRequest | None:
    if workflow.status is not WorkflowStatus.BLOCKED:
        return None
    try:
        incident = store.get_incident(workflow.incident_id)
    except (KeyError, NotFoundError):
        return None
    successor_id = incident.workflow_request_id
    if (
        incident.state is not IncidentState.RECOVERED
        or not successor_id
        or successor_id == workflow.request_id
    ):
        return None
    try:
        successor = store.get_workflow(successor_id)
    except (KeyError, NotFoundError):
        return None
    if (
        successor.incident_id != workflow.incident_id
        or successor.status is not WorkflowStatus.SUCCEEDED
        or successor.fencing_token != workflow.fencing_token
        or incident.fencing_token != workflow.fencing_token
        or WorkflowOperation.RESTORE_SCHEDULING not in successor.completed_operations
    ):
        return None
    return successor


def abandoned_generation_successor(
    store: WorkflowStore,
    workflow: WorkflowRequest,
    *,
    now: datetime,
) -> WorkflowRequest | None:
    """The current workflow that replaced ``workflow``, if it was left behind.

    When a family re-plans an incident it usually links the new workflow as a
    preempting successor, and ``executor`` then supersedes the predecessor at a
    step boundary. Some re-plan paths do not: ``families/health.py`` sets
    ``predecessor_workflow_id`` only in its "serialized behind a node-exclusive
    incumbent" branch, so an incident that escalates for an unrelated reason
    bumps its generation and points ``workflow_request_id`` at a new workflow
    while the old one keeps every field a live workflow has.

    Nothing else closes that record. ``_validate_fencing`` only rejects a stale
    token when the incident still names *this* workflow -- once the incident has
    moved on, the mismatch is read as a preemption that the (absent) successor
    link was supposed to resolve. So the abandoned workflow is dispatched
    forever, and because ``WorkflowDispatcher.run_once`` admits one workflow per
    incident, it also starves the successor the incident is actually waiting on.
    On 2026-09-04 a ``RESTART_APP`` workflow at generation 1 did both for four
    and a half hours while its incident sat ``ACTION_PENDING`` at generation 4.

    Every condition below is required, and together they leave no reading under
    which the workflow is still going to run:

    * ``PENDING`` with no step execution, no completed step and no completed
      operation -- no adapter has ever been handed any of its steps.
    * No execution owner and no unexpired lease -- nobody is driving it.
    * No remediation budget claim -- it is holding no fleet-wide allowance.
    * Its incident names a *different* workflow, at a *strictly* higher
      generation, and that workflow really belongs to the same incident at the
      incident's current generation.

    Deliberately not keyed on age. A workflow can legitimately sit at
    ``PENDING`` behind an aggregation deadline or a fence, and a staleness
    threshold would eventually terminalize one of those instead.

    ``RUNNING`` and ``SAFETY_PENDING`` are excluded rather than examined: a
    workflow past ``PENDING`` may be mid-step, and there is no cheap proof
    otherwise. Everything unproven keeps its blocking behaviour.
    """

    if (
        workflow.status is not WorkflowStatus.PENDING
        or workflow.step_executions
        or workflow.completed_step_indexes
        or workflow.completed_operations
        or workflow.remediation_budget_claims
        or workflow.execution_owner_id is not None
        or (
            workflow.execution_lease_expires_at is not None
            and workflow.execution_lease_expires_at > now
        )
    ):
        return None
    successor = retired_generation_successor(store, workflow)
    if successor is not None:
        return successor
    return _same_generation_twin_successor(store, workflow)


def _same_generation_twin_successor(
    store: WorkflowStore,
    workflow: WorkflowRequest,
) -> WorkflowRequest | None:
    """The fourth shape (P1-80A): a twin at the *same* generation.

    ``retired_generation_successor`` requires a strictly later generation
    because a legitimate preemption or queued successor shares the token -- and
    links back to its predecessor. A twin created by two families racing the
    same incident shares the token too, but nothing links it: the incident
    names the other record, nobody names this one as a predecessor. Nothing
    dispatches, fences or sweeps it, so it is closed here. The predecessor link
    check is what keeps a queued successor's pending predecessor safe.
    """

    if workflow.predecessor_workflow_id is not None:
        # A queued successor is waiting on its own predecessor; it is that
        # predecessor's business, not a twin.
        return None
    try:
        incident = store.get_incident(workflow.incident_id)
    except (KeyError, NotFoundError):
        return None
    successor_id = incident.workflow_request_id
    if (
        not successor_id
        or successor_id == workflow.request_id
        or workflow.fencing_token != incident.fencing_token
    ):
        return None
    try:
        successor: WorkflowRequest = store.get_workflow(successor_id)
    except (KeyError, NotFoundError):
        return None
    if (
        successor.incident_id != workflow.incident_id
        or successor.fencing_token != incident.fencing_token
        or successor.predecessor_workflow_id == workflow.request_id
        or store.has_workflow_successor(workflow.request_id)
    ):
        return None
    return successor


def retirement_fences_out_dispatch(
    store: WorkflowStore,
    workflow: WorkflowRequest,
    incident: FaultIncident,
) -> bool:
    """Whether the executor must refuse to run ``workflow`` at all.

    ``_validate_fencing`` used to reject a stale token only while the incident
    still named *this* workflow. Once it named another, the mismatch was read as a
    preemption for the successor link to resolve -- and when no link was ever
    written, nothing rejected anything: on 2026-09-04 a generation-1
    ``STOP_WORKLOADS`` stayed dispatchable for four and a half hours after its
    incident recovered at generation 4.

    The successor lookup is what tells the two shapes apart, and it is why this
    reads the Store rather than being a pure comparison. When the link *was*
    written the record is somebody's business and dispatch has to continue: with
    ``preempt_predecessor`` the executor's own ``_supersede_if_safe`` closes it at
    a step boundary, and only that path compensates an unrestored
    ``QUIESCE_GPU_SERVICES``; without it the successor is queued behind this
    workflow and refusing to dispatch would deadlock the pair. The read only
    happens in the rare shape where the incident has already moved past this
    workflow, so the common dispatch stays a comparison.

    This is a belt, not the cure. ``WorkflowDispatcher`` revokes such records once
    per loop; this stops the tick that beats it from handing another destructive
    step to an adapter.
    """

    if incident.workflow_request_id in {None, workflow.request_id}:
        return False
    if workflow.fencing_token >= incident.fencing_token:
        return False
    return retired_generation_successor(store, workflow) is not None


def workflow_never_changed_a_node(workflow: WorkflowRequest) -> bool:
    """Whether the workflow completed no node-mutating operation (F-B4 (4)).

    Containment-only operations -- a cordon, a quarantine taint -- do not count:
    they change what the scheduler may place on the node, not the node, and the
    incident that owns the isolation is what lifts it. A record that only ever
    isolated, or did nothing at all before it blocked, has left nothing on the
    node that a restore successor would have to undo.
    """

    return not (set(workflow.completed_operations) & NODE_MUTATING_OPERATIONS)


def completed_containment_operations(workflow: WorkflowRequest) -> list[str]:
    return sorted(
        operation.value
        for operation in set(workflow.completed_operations)
        & CONTAINMENT_ONLY_OPERATIONS
    )


def restore_reconciliation_reasons(
    workflow: WorkflowRequest,
    incident: FaultIncident | None,
    successor: WorkflowRequest | None,
    source_plan: RecoveryPlan | None,
    remote_commands: list[RemoteActionCommand],
    *,
    evaluated_at: datetime,
) -> list[str]:
    reasons: list[str] = []
    if workflow.status is not WorkflowStatus.BLOCKED:
        reasons.append(f"workflow status is {workflow.status.value}, not BLOCKED")
    if workflow.execution_owner_id is not None:
        reasons.append("workflow still has an execution owner")
    if (
        workflow.execution_lease_expires_at is not None
        and workflow.execution_lease_expires_at > evaluated_at
    ):
        reasons.append("workflow execution lease has not expired")
    open_commands = [
        item.command_id
        for item in remote_commands
        if item.workflow_request_id == workflow.request_id
        and item.status in OPEN_REMOTE_STATUSES
    ]
    if open_commands:
        reasons.append("workflow has open remote commands")
    if any(
        item.status is WorkflowStepStatus.WAITING for item in workflow.step_executions
    ):
        reasons.append("workflow has an unknown provider action")
    if incident is None:
        reasons.append("incident is missing")
    # The second eligible path (F-B4 (4)): a record that never changed a node
    # needs no restore successor, because there is nothing to restore. It still
    # needs its incident settled -- recovered elsewhere, or escalated to the
    # operator now closing it -- and every gate below this block still applies,
    # including the source-plan gate: a workflow that was never plan-driven has
    # no plan to carry the reconciliation audit, and stays ineligible by design.
    never_changed = successor is None and workflow_never_changed_a_node(workflow)
    if incident is not None and never_changed:
        if incident.state not in SETTLED_INCIDENT_STATES:
            reasons.append(
                f"incident is {incident.state.value}, still waiting on a workflow"
            )
    elif incident is not None and incident.state is not IncidentState.RECOVERED:
        reasons.append("incident is not RECOVERED")
    if successor is None:
        if not never_changed:
            reasons.append("workflow has no verified restore successor")
    elif (
        incident is None
        or incident.workflow_request_id != successor.request_id
        or successor.incident_id != workflow.incident_id
        or successor.status is not WorkflowStatus.SUCCEEDED
        or successor.fencing_token != workflow.fencing_token
        or incident.fencing_token != workflow.fencing_token
        or WorkflowOperation.RESTORE_SCHEDULING not in successor.completed_operations
    ):
        reasons.append("workflow restore successor is no longer valid")
    if not workflow.source_plan_id:
        reasons.append("workflow has no source recovery plan")
    elif source_plan is None:
        reasons.append("source recovery plan is missing")
    elif (
        source_plan.plan_id != workflow.source_plan_id
        or source_plan.incident_id != workflow.incident_id
        or source_plan.workflow_request_id != workflow.request_id
    ):
        reasons.append("source recovery plan linkage is invalid")
    elif source_plan.status is not PlanStatus.FAILED:
        reasons.append("source recovery plan is not FAILED")
    return reasons


def reconciled_restore_records(
    workflow: WorkflowRequest,
    incident: FaultIncident,
    successor: WorkflowRequest | None,
    source_plan: RecoveryPlan,
    remote_commands: list[RemoteActionCommand],
    *,
    expected_fencing_token: int,
    expected_workflow_updated_at: datetime | None,
    expected_execution_epoch: int | None = None,
    reference: str,
    reconciled_at: datetime,
) -> tuple[WorkflowRequest, FaultIncident, RecoveryPlan]:
    """Terminalize a BLOCKED record, re-verifying every condition first.

    The compare-and-set names the field that moved and both values, so a
    refusal reads as "the generation changed" or "somebody claimed it" rather
    than a generic "plan changed". ``fencing_token`` and ``execution_epoch`` are
    the keys a re-plan or a claim moves and a merge does not; ``updated_at`` is
    what the Store contract still hands in today and is honoured when given.

    ``blocked_reasons`` is deliberately not appended to (P1-61D): it records why
    the workflow blocked, the audit lives in ``preemption_reason`` and on the
    incident, and the retired-generation sibling refuses to touch it for the
    same reason.
    """

    reasons = restore_reconciliation_reasons(
        workflow,
        incident,
        successor,
        source_plan,
        remote_commands,
        evaluated_at=reconciled_at,
    )
    stale: list[str] = []
    if workflow.fencing_token != expected_fencing_token:
        stale.append(
            "workflow fencing token changed: expected "
            f"{expected_fencing_token}, found {workflow.fencing_token}"
        )
    if (
        expected_execution_epoch is not None
        and workflow.execution_epoch != expected_execution_epoch
    ):
        stale.append(
            "workflow execution epoch changed: expected "
            f"{expected_execution_epoch}, found {workflow.execution_epoch}"
        )
    if (
        expected_workflow_updated_at is not None
        and workflow.updated_at != expected_workflow_updated_at
    ):
        stale.append(
            "workflow updated_at changed after reconcile validation: expected "
            f"{expected_workflow_updated_at.isoformat()}, found "
            f"{workflow.updated_at.isoformat()}"
        )
    if stale:
        # The record moved under the caller: a ``StaleWriteError`` (still a
        # ``ValueError``) so a retry can tell it from an ineligible record.
        raise StaleWriteError(
            "workflow reconcile rejected: " + "; ".join([*reasons, *stale])
        )
    if reasons:
        raise ValueError("workflow reconcile rejected: " + "; ".join(reasons))
    if successor is not None:
        audit = (
            f"operator reconciliation {reference}: superseded {workflow.request_id} "
            f"after verified restore {successor.request_id}"
        )
    else:
        audit = (
            f"operator reconciliation {reference}: closed {workflow.request_id}, "
            "which completed no node-mutating operation, with its incident "
            f"{incident.state.value}"
        )
    updated_workflow = workflow.model_copy(
        update={
            "status": WorkflowStatus.SUPERSEDED,
            "preempted_by_workflow_id": (
                successor.request_id if successor is not None else None
            ),
            "preemption_reason": audit,
            "superseded_at": reconciled_at,
            "updated_at": reconciled_at,
        }
    )
    updated_incident = incident.model_copy(
        update={
            "reasons": bounded_reasons([*incident.reasons, audit]),
            "updated_at": reconciled_at,
        }
    )
    updated_plan = source_plan.model_copy(
        update={
            "resolved_by_restore_workflow_id": (
                successor.request_id if successor is not None else None
            ),
            "reconciliation_reference": reference,
            "reconciled_at": reconciled_at,
        }
    )
    return updated_workflow, updated_incident, updated_plan

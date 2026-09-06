"""Workflow state-machine invariants, checked after every write (review item 6).

The invariants below are what the executor, dispatcher and merge paths each
assume about a ``WorkflowRequest`` and each enforce only for the corner they
touch. Checking them in one place after a write turns "the scenario ended in
the right status" into "the record was well-formed at every step". The mode
is configuration: ``off`` costs nothing, ``log`` reports a violation and
keeps going, ``raise`` is for tests and staging.

Every check is one sentence and reads only the record: no store, no clock.
Where a rule comes from one specific writer, that writer is named so a
violation can be traced back to the path that broke it.
"""

from __future__ import annotations

import logging
from enum import StrEnum

from gpu_fault.models import (
    WORKFLOW_EVENTS_LIMIT,
    WorkflowOperation,
    WorkflowRequest,
    WorkflowStatus,
    WorkflowStepSpec,
)

LOGGER = logging.getLogger(__name__)

# ``dag_branching._reconnect_join`` stamps this on every RESTART_WORKLOAD of a
# DAG and on nothing else; ``node_branch_step_indexes`` relies on it.
JOIN_BRANCH_ID = "join"

# A record in one of these states will never execute again on its own, so the
# executor's terminalizing writes release the owner and the lease together.
TERMINAL_STATUSES = frozenset(
    {
        WorkflowStatus.SUCCEEDED,
        WorkflowStatus.FAILED,
        WorkflowStatus.BLOCKED,
        WorkflowStatus.SUPERSEDED,
    }
)


class InvariantMode(StrEnum):
    OFF = "off"
    LOG = "log"
    RAISE = "raise"


class WorkflowInvariantError(RuntimeError):
    pass


def _index_list_violations(workflow: WorkflowRequest, count: int) -> list[str]:
    violations: list[str] = []
    completed_list = list(workflow.completed_step_indexes)
    superseded_list = list(workflow.superseded_step_indexes)
    # Order is execution order, not index order: a DAG batch appends the step
    # that finished first, so [0, 2, 4, 3] is legitimate. Only duplicates are
    # a defect (a step counted twice by the coverage check).
    if len(completed_list) != len(set(completed_list)):
        violations.append(
            f"completed_step_indexes contain duplicates: {completed_list}"
        )
    if len(superseded_list) != len(set(superseded_list)):
        violations.append(
            f"superseded_step_indexes contain duplicates: {superseded_list}"
        )
    completed = set(completed_list)
    superseded = set(superseded_list)
    overlap = completed & superseded
    if overlap:
        violations.append(
            f"step indexes both completed and superseded: {sorted(overlap)}"
        )
    for name, indexes in (("completed", completed), ("superseded", superseded)):
        outside = sorted(index for index in indexes if index >= count)
        if outside:
            violations.append(f"{name} step indexes outside the step list: {outside}")
    return violations


def _dag_violations(
    workflow: WorkflowRequest, steps: list[WorkflowStepSpec], count: int
) -> list[str]:
    violations: list[str] = []
    for index, step in enumerate(steps):
        # The join may depend on a *later* branch tail
        # (``dag_branching._reconnect_join``), so only existence and
        # self-reference are checked, never the direction.
        bad = sorted(
            d for d in step.depends_on_step_indexes if d < 0 or d >= count or d == index
        )
        if bad:
            violations.append(f"step {index} depends on invalid indexes: {bad}")
    if not workflow.dag_enabled:
        return violations
    joins = [
        index
        for index, step in enumerate(steps)
        if step.operation is WorkflowOperation.RESTART_WORKLOAD
    ]
    if len(joins) > 1:
        violations.append(f"DAG has more than one RESTART_WORKLOAD join: {joins}")
    # The join label is stamped by ``_reconnect_join`` when a branch is
    # appended; a DAG whose join was never re-wired legitimately carries the
    # compiler's label, so only the converse (a non-restart step labelled as
    # the join) is a defect.
    for index, step in enumerate(steps):
        is_join = step.operation is WorkflowOperation.RESTART_WORKLOAD
        if step.branch_id == JOIN_BRANCH_ID and not is_join:
            violations.append(
                f"DAG step {index} ({step.operation.value}) carries branch_id "
                f'"{JOIN_BRANCH_ID}" but is not the RESTART_WORKLOAD join'
            )
    return violations


def _branch_bookkeeping_violations(workflow: WorkflowRequest) -> list[str]:
    """``exhausted_branch_ids`` and ``branch_escalation_counts`` are written by
    ``BranchEscalator`` against the official steps; both must point at
    something the plan still contains."""

    violations: list[str] = []
    # Branch ids are minted by ``dag_branching._candidate_branch`` as
    # ``branch:<nodes>[:successor:N][:M]``; the exhausted list must quote one.
    branch_ids = {step.branch_id for step in workflow.official_steps}
    unknown_branches = sorted(
        branch for branch in workflow.exhausted_branch_ids if branch not in branch_ids
    )
    if unknown_branches:
        violations.append(
            "exhausted_branch_ids name branches absent from official_steps: "
            f"{unknown_branches}"
        )
    nodes = {
        node
        for step in (*workflow.official_steps, *workflow.safety_steps)
        for node in (*step.node_ids, *step.branch_node_ids)
    }
    unknown_nodes = sorted(
        node for node in workflow.branch_escalation_counts if node not in nodes
    )
    if unknown_nodes:
        violations.append(
            "branch_escalation_counts name nodes absent from every step: "
            f"{unknown_nodes}"
        )
    return violations


def _ownership_violations(workflow: WorkflowRequest) -> list[str]:
    violations: list[str] = []
    status = workflow.status.value
    if workflow.status in TERMINAL_STATUSES:
        if workflow.execution_owner_id is not None:
            violations.append(f"{status} workflow still has an execution owner")
        if workflow.execution_lease_expires_at is not None:
            violations.append(f"{status} workflow still holds an execution lease")
    elif (
        workflow.status is WorkflowStatus.RUNNING
        and workflow.execution_owner_id is None
    ):
        violations.append("RUNNING workflow has no execution owner")
    if (
        workflow.preempted_by_workflow_id is not None
        and workflow.status is not WorkflowStatus.SUPERSEDED
        and workflow.preemption_pending_by_workflow_id is None
    ):
        violations.append(
            "preempted_by_workflow_id is set but the workflow is neither "
            "SUPERSEDED nor preemption-pending"
        )
    if workflow.workload_withdrawn_at is not None and not (
        workflow.workload_withdrawn_reason
    ):
        violations.append(
            "workload_withdrawn_at is set without workload_withdrawn_reason"
        )
    if (
        workflow.lifetime_deadline_at is not None
        and workflow.lifetime_deadline_at < workflow.created_at
    ):
        violations.append("lifetime_deadline_at precedes created_at")
    return violations


def _record_violations(workflow: WorkflowRequest) -> list[str]:
    """Step execution records and the audit trail."""

    violations: list[str] = []
    # A record stamped for either phase is legitimate: a safety-only record
    # keeps the official-phase attempts it made before it was demoted (F-C2).
    lengths = {
        "official": len(workflow.official_steps),
        "safety": len(workflow.safety_steps),
    }
    for record in workflow.step_executions:
        limit = max(lengths.values()) if record.phase is None else lengths[record.phase]
        if record.step_index >= limit:
            violations.append(
                f"step execution {record.phase or 'unphased'}/{record.step_index}/"
                f"{record.operation.value} points outside its phase's step list "
                f"({limit} steps)"
            )
    events = workflow.events
    if len(events) > WORKFLOW_EVENTS_LIMIT:
        violations.append(
            f"events exceed WORKFLOW_EVENTS_LIMIT: {len(events)} > "
            f"{WORKFLOW_EVENTS_LIMIT}"
        )
    for position in range(1, len(events)):
        try:
            out_of_order = events[position].at < events[position - 1].at
        except TypeError:
            out_of_order = True  # naive and aware timestamps mixed
        if out_of_order:
            violations.append(
                f"events are not in chronological order at position {position}"
            )
            break
    return violations


def workflow_invariant_violations(workflow: WorkflowRequest) -> list[str]:
    """Every invariant the record breaks, as one sentence each; empty is good."""

    steps = (
        workflow.safety_steps
        if workflow.executes_safety_steps
        else (workflow.official_steps)
    )
    count = len(steps)
    violations = _index_list_violations(workflow, count)
    violations.extend(_dag_violations(workflow, steps, count))
    resolved = set(workflow.completed_step_indexes) | set(
        workflow.superseded_step_indexes
    )
    if workflow.status is WorkflowStatus.SUCCEEDED and not resolved >= set(
        range(count)
    ):
        violations.append("SUCCEEDED workflow has unresolved steps")
    violations.extend(_branch_bookkeeping_violations(workflow))
    violations.extend(_ownership_violations(workflow))
    violations.extend(_record_violations(workflow))
    return violations


def check_workflow_invariants(workflow: WorkflowRequest, mode: InvariantMode) -> None:
    if mode is InvariantMode.OFF:
        return
    violations = workflow_invariant_violations(workflow)
    if not violations:
        return
    message = f"workflow {workflow.request_id} violates invariants: " + "; ".join(
        violations
    )
    if mode is InvariantMode.RAISE:
        raise WorkflowInvariantError(message)
    LOGGER.error(message)

"""A read-only NCCL hung diagnostic always ends at its support hand-off.

The hung DAG (``orchestration/hung_dag.py``) only collects evidence --
FREEZE_EVIDENCE, two whole-task summaries, two triage waves, the diagnostic
bundle, VALIDATE_FABRIC -- and ends with ESCALATE_SUPPORT, the step that drafts
the ticket and tells a human that a training attempt looks hung. A failed
evidence step says nothing about the node, but the DAG loop treated it like a
failed repair: the workflow terminalized at once and the hand-off never ran.
Live (HyperPod, GF-GENERAL-LIVE-011.active_to_zero, 2026-10-24): VALIDATE_FABRIC
waited out its 600 s per-step cap, the workflow ended FAILED with step 7
ESCALATE_SUPPORT never executed; the incident read ESCALATED
(``terminal_state.failure_incident_state``) but no ticket was drafted and
nobody was told while the job stayed hung.

``support_handoff`` keeps such a workflow going instead: the failed step is
resolved as it stands (its FAILED record stays), the steps that need its output
are retired, its error becomes the record's ``terminal_failure_reason`` -- the
workflow still ends FAILED and the incident ESCALATED
(``ProductionWorkflowExecutor._fail_with_reason``) -- independent evidence keeps
running and the hand-off runs once its dependencies are resolved. After the
execution deadline nothing new starts but the hand-off (``deadline_failure``).
Only the read-only hung DAG takes this path: a plan that also touches a node, an
ordinary diagnostic (design 6.3: RECOVERED, ``diagnostic inconclusive``) and a
workflow past its lifetime -- which the hardware escalation hands to an
operator -- keep the established failure path.

A step whose verdict the hand-off carries may also resolve to an explicit
UNKNOWN (``adapters/gpu_validation``) instead of waiting for evidence nothing
will provide; without a hand-off an UNKNOWN must never complete a diagnostic.

Control-plane only: the node runtime and cluster executor wheels carry
``step_bounds`` but run no hung DAG, so nothing they import depends on this.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from datetime import datetime, timezone
from typing import Any

from gpu_fault.execution import step_bounds
from gpu_fault.execution.branch_escalation import BranchEscalation
from gpu_fault.execution.models import WorkflowStepOutcome
from gpu_fault.models import (
    WorkflowEventKind,
    WorkflowOperation,
    WorkflowRequest,
    WorkflowStepSpec,
    WorkflowStepStatus,
    record_workflow_event,
    resolved_step_indexes,
)
from gpu_fault.orchestration.hung_episode import diagnostic_only_hung

LOGGER = logging.getLogger(__name__)
HANDOFF = WorkflowOperation.ESCALATE_SUPPORT
HUNG_DIAGNOSTIC = "HUNG_DIAGNOSTIC"
#: The ``PLAN_REWRITE`` event code of a continued hung DAG. A writer-local code
#: (``WorkflowEvent.code`` is a string so older rows and releases load it):
#: ``models.py`` is a shared contract whose change selects the full regional
#: plan, and this event has one writer.
HUNG_SUPPORT_HANDOFF = "HUNG_SUPPORT_HANDOFF"


def _mode(step: WorkflowStepSpec) -> object:
    contract = step.parameters.get("health_recovery_contract")
    return contract.get("mode") if isinstance(contract, dict) else None


def hung_handoff_workflow(workflow: WorkflowRequest) -> bool:
    """The read-only hung DAG: hung evidence steps and a hand-off, nothing else."""

    steps = workflow.official_steps
    return bool(
        workflow.dag_enabled
        and not workflow.executes_safety_steps
        and diagnostic_only_hung(workflow)
        and any(step.operation is HANDOFF for step in steps)
        and all(_mode(step) == HUNG_DIAGNOSTIC for step in steps)
    )


def dependent_step_indexes(steps: Sequence[WorkflowStepSpec], index: int) -> set[int]:
    """Every step that waits on step ``index``, directly or through another."""

    found: set[int] = set()
    frontier = [index]
    while frontier:
        current = frontier.pop()
        for position, step in enumerate(steps):
            if position not in found and current in step.depends_on_step_indexes:
                found.add(position)
                frontier.append(position)
    return found


def support_handoff_follows(workflow: WorkflowRequest, index: int) -> bool:
    """Whether a pending ESCALATE_SUPPORT of this record runs after step ``index``."""

    steps = (
        workflow.safety_steps
        if workflow.executes_safety_steps
        else workflow.official_steps
    )
    if not 0 <= index < len(steps):
        return False
    later: Iterable[int] = (
        dependent_step_indexes(steps, index)
        if workflow.dag_enabled
        else range(index + 1, len(steps))
    )
    resolved = resolved_step_indexes(workflow)
    return any(
        steps[position].operation is HANDOFF and position not in resolved
        for position in later
    )


def support_handoff(
    workflow: WorkflowRequest, index: int, outcome: WorkflowStepOutcome
) -> BranchEscalation | None:
    """The hung DAG rewritten to go on to its hand-off after step ``index`` FAILED.

    ``None`` keeps the established failure path: not the read-only hung DAG, a
    parked failure already owns the verdict, the failed step is the hand-off
    itself, no hand-off is left, or the workflow's lifetime ran out. The
    rewrite is returned in the shape the DAG loop applies to a rewritten
    branch; the caller persists it.
    """

    steps = workflow.official_steps
    if (
        not hung_handoff_workflow(workflow)
        or workflow.pending_failure_step_index is not None
        or not 0 <= index < len(steps)
        or steps[index].operation is HANDOFF
    ):
        return None
    details = outcome.details or {}
    resolved = resolved_step_indexes(workflow)
    handoffs = [
        position
        for position, step in enumerate(steps)
        if step.operation is HANDOFF and position not in resolved
    ]
    if not handoffs or details.get("workflow_lifetime_exceeded"):
        return None
    if "workflow_execution_deadline" in details:
        # Past the deadline every other step would only fail on it: nothing
        # new starts; a step still in flight fails on its own next attempt.
        in_flight = {
            item.step_index
            for item in workflow.step_executions
            if item.status is WorkflowStepStatus.WAITING
        }
        retired = set(range(len(steps))) - in_flight
    else:
        retired = dependent_step_indexes(steps, index)
    retired -= {*resolved, *handoffs, index}
    step = steps[index]
    error = outcome.error or f"step {index}/{step.operation.value} failed"
    reason = (
        f"hung diagnostic step {index}/{step.operation.value} failed; "
        f"{len(retired)} dependent step(s) retired, the support hand-off "
        f"still runs: {error}"
    )
    continued = workflow.model_copy(
        update={
            "superseded_step_indexes": sorted(
                {*workflow.superseded_step_indexes, *retired, index}
            ),
            "terminal_failure_reason": workflow.terminal_failure_reason or error,
            "updated_at": datetime.now(timezone.utc),
        }
    )
    continued = record_workflow_event(
        continued,
        WorkflowEventKind.PLAN_REWRITE,
        code=HUNG_SUPPORT_HANDOFF,
        reason=reason,
        step_index=index,
        operation=step.operation,
        details={
            "failed_step_index": index,
            "retired_step_indexes": sorted(retired),
            "handoff_step_indexes": handoffs,
        },
    )
    LOGGER.warning("workflow %s: %s", workflow.request_id, reason)
    return BranchEscalation(
        workflow=continued,
        outcome="handoff",
        node_id=",".join(step.node_ids),
        branch_id=step.branch_id or "",
        next_operation=HANDOFF,
        reason=reason,
    )


def deadline_failure(
    executor: Any, workflow: WorkflowRequest, step: WorkflowStepSpec, index: int
) -> WorkflowStepOutcome | None:
    """``step_bounds.workflow_deadline_failure``, except for the hung hand-off.

    The read-only hung DAG's ESCALATE_SUPPORT is its last word to a human --
    one control-plane write -- and is owed even past the execution deadline,
    when ``support_handoff`` has retired every other step that had not
    started. A spent lifetime is not exempt: the hardware escalation hands that
    record to an operator.
    """

    lifetime = workflow.lifetime_deadline_at
    if (
        step.operation is HANDOFF
        and hung_handoff_workflow(workflow)
        and (lifetime is None or datetime.now(timezone.utc) < lifetime)
    ):
        return None
    return step_bounds.workflow_deadline_failure(executor, workflow, step, index)

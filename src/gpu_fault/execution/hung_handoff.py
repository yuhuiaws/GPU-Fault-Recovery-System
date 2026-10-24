"""Whether a read-only NCCL hung diagnostic hands its verdict to a human.

The hung DAG (``orchestration/hung_dag.py``) only collects evidence --
FREEZE_EVIDENCE, two whole-task summaries, two triage waves, the diagnostic
bundle, VALIDATE_FABRIC -- and ends with ESCALATE_SUPPORT, the step that drafts
the ticket and tells a human that a training attempt looks hung. A step whose
verdict that hand-off carries may resolve to an explicit UNKNOWN
(``adapters/gpu_validation``) instead of waiting for evidence nothing will
provide; without a hand-off an UNKNOWN must never complete a diagnostic, which
would end RECOVERED (design 6.3).

The predicates import only the models and ``orchestration.hung_episode``.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from gpu_fault.models import (
    WorkflowOperation,
    WorkflowRequest,
    WorkflowStepSpec,
    resolved_step_indexes,
)
from gpu_fault.orchestration.hung_episode import diagnostic_only_hung

HANDOFF = WorkflowOperation.ESCALATE_SUPPORT
HUNG_DIAGNOSTIC = "HUNG_DIAGNOSTIC"


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

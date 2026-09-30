"""A flat (non-DAG) record's FAILED step: escalate in place or end the record.

The flat executor loop used to end its record FAILED on the first failed step
and leave the hardware ladder to the whole-workflow escalation (a
``workflow-reboot-after-<id>`` successor), while the same failure on a node
branch of a job DAG escalated in place (F-N1). A single-node record now takes
the same ladder: the branch escalator grows it into a one-branch DAG with its
next rung appended, and the DAG loop carries it from there. The flat failure
semantics stay for everything the escalator declines (several nodes, unknown
outcome, no rung, a rung the remediation budget refused).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING

from gpu_fault.execution.models import WorkflowStepOutcome
from gpu_fault.models import (
    FaultIncident,
    WorkflowExecutionRequest,
    WorkflowExecutionResult,
    WorkflowOperation,
    WorkflowRequest,
    WorkflowStatus,
    WorkflowStepSpec,
)

if TYPE_CHECKING:
    from gpu_fault.execution.executor import ProductionWorkflowExecutor


def fail_flat_step(
    executor: "ProductionWorkflowExecutor",
    workflow: WorkflowRequest,
    incident: FaultIncident,
    request: WorkflowExecutionRequest,
    *,
    index: int,
    step: WorkflowStepSpec,
    steps: list[WorkflowStepSpec],
    outcome: WorkflowStepOutcome,
    is_safety: bool,
    execution_epoch: int,
) -> WorkflowExecutionResult:
    if not is_safety:
        escalation = executor._escalate_failed_branch(
            workflow, incident, index, outcome
        )
        if escalation is not None:
            workflow = escalation.workflow
            steps = workflow.official_steps
            if escalation.outcome == "escalated":
                executor._save_leased(workflow, execution_epoch)
                return executor._execute_dag(
                    workflow,
                    incident,
                    request,
                    is_safety=is_safety,
                    execution_epoch=execution_epoch,
                )
            # Exhausted (the budget refused the rung): the record fails as
            # before, with the refusal on its trail.
    if (
        step.operation is not WorkflowOperation.RESTORE_GPU_SERVICES
        and executor._has_unrestored_quiesce(workflow, steps)
    ):
        workflow = workflow.model_copy(
            update={
                "pending_failure_step_index": index,
                "pending_failure_error": (outcome.error or "workflow step failed"),
                "updated_at": datetime.now(timezone.utc),
            }
        )
        executor._save_leased(workflow, execution_epoch)
        return executor._resume_failure_compensation(
            workflow,
            incident,
            request,
            execution_epoch,
            is_safety=is_safety,
        )
    return executor._terminalize(
        workflow,
        incident,
        WorkflowStatus.FAILED,
        execution_epoch,
        reason=outcome.error,
    )

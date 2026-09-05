from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import logging
from typing import Any, Sequence

from gpu_fault.models import (
    FaultIncident,
    IncidentState,
    WorkflowOperation,
    WorkflowExecutionRequest,
    WorkflowRequest,
    WorkflowStatus,
    WorkflowStepSpec,
)
from gpu_fault.store import NotFoundError
from gpu_fault.execution.models import WorkflowStepOutcome
from gpu_fault.execution.remediation_budget import remediation_budget_claims
import gpu_fault.execution.step_bounds as step_bounds


LOGGER = logging.getLogger(__name__)
_REQUIRED_RESTART_PARAMETERS = frozenset(
    {
        "cluster_id",
        "job_id",
        "source_attempt_id",
        "source_gpu_count",
        "restart_budget",
    }
)


@dataclass(frozen=True)
class RestartBudgetPreflightFailure:
    step_index: int
    step: WorkflowStepSpec
    outcome: WorkflowStepOutcome


@dataclass(frozen=True)
class ClaimedWorkflowPreparation:
    workflow: WorkflowRequest
    execution_epoch: int
    result: Any | None = None


def _reservation_id(
    workflow: WorkflowRequest,
    step_index: int,
) -> str:
    return (
        f"{workflow.request_id}/{step_index}/{WorkflowOperation.RESTART_WORKLOAD.value}"
    )


def reserve_restart_budgets(
    store: Any,
    workflow: WorkflowRequest,
    incident: FaultIncident,
    steps: Sequence[WorkflowStepSpec],
) -> RestartBudgetPreflightFailure | None:
    """Reserve every pending restart before any workflow adapter runs."""

    completed = set(workflow.completed_step_indexes)
    for step_index, step in enumerate(steps):
        if (
            step.operation is not WorkflowOperation.RESTART_WORKLOAD
            or step_index in completed
        ):
            continue
        parameters = step.parameters
        missing = _REQUIRED_RESTART_PARAMETERS - set(parameters)
        if missing:
            return RestartBudgetPreflightFailure(
                step_index=step_index,
                step=step,
                outcome=WorkflowStepOutcome.failed(
                    "restart safety context is missing: " + ", ".join(sorted(missing)),
                    details={
                        "reason": "RESTART_SAFETY_CONTEXT_MISSING",
                        "missing_parameters": sorted(missing),
                    },
                ),
            )
        cluster_id = str(parameters["cluster_id"])
        job_id = str(parameters["job_id"])
        if cluster_id != incident.cluster_id:
            return RestartBudgetPreflightFailure(
                step_index=step_index,
                step=step,
                outcome=WorkflowStepOutcome.failed(
                    "restart safety context cluster does not match "
                    f"incident: {cluster_id} != {incident.cluster_id}",
                    details={
                        "reason": "RESTART_CLUSTER_MISMATCH",
                        "restart_cluster_id": cluster_id,
                        "incident_cluster_id": incident.cluster_id,
                    },
                ),
            )
        try:
            budget = int(parameters["restart_budget"])
            state, reserved = store.reserve_job_restart(
                cluster_id,
                job_id,
                budget,
                _reservation_id(workflow, step_index),
            )
        except (TypeError, ValueError) as exc:
            return RestartBudgetPreflightFailure(
                step_index=step_index,
                step=step,
                outcome=WorkflowStepOutcome.failed(
                    f"restart safety context is invalid: {exc}",
                    details={
                        "reason": "RESTART_SAFETY_CONTEXT_INVALID",
                    },
                ),
            )
        if reserved:
            continue
        return RestartBudgetPreflightFailure(
            step_index=step_index,
            step=step,
            outcome=WorkflowStepOutcome.failed(
                "restart budget exhausted for "
                f"{state.cluster_id}/{state.job_id}: "
                f"{state.restart_count}/{state.budget}",
                details={
                    "reason": "RESTART_BUDGET_EXHAUSTED",
                    "restart_count": state.restart_count,
                    "restart_budget": state.budget,
                },
            ),
        )
    return None


def prepare_claimed_workflow(
    executor: Any,
    request_id: str,
    request: WorkflowExecutionRequest,
    workflow: WorkflowRequest,
    incident: FaultIncident,
    *,
    is_safety: bool,
) -> ClaimedWorkflowPreparation:
    """Claim a workflow and finish every no-mutation preflight gate."""

    steps = workflow.safety_steps if is_safety else workflow.official_steps
    workflow = executor.store.claim_workflow(
        request_id,
        executor.config.executor_id,
        request.expected_fencing_token,
        lease_duration=executor._lease_duration,
        remediation_budget_claims=remediation_budget_claims(
            executor.config.remediation_budget,
            workflow,
            incident,
            steps,
        ),
    ).model_copy(
        update={
            "status": WorkflowStatus.RUNNING,
            "execution_deadline": (
                workflow.execution_deadline
                or datetime.now(timezone.utc)
                + timedelta(seconds=executor.config.workflow_execution_timeout_seconds)
            ),
            "updated_at": datetime.now(timezone.utc),
        }
    )
    execution_epoch = workflow.execution_epoch
    workflow = executor._adopt_quiesce_handoff_from_predecessor(workflow, incident)
    executor._save_leased(workflow, execution_epoch)
    if workflow.pending_failure_step_index is not None:
        return ClaimedWorkflowPreparation(
            workflow=workflow,
            execution_epoch=execution_epoch,
            result=executor._resume_failure_compensation(
                workflow,
                incident,
                request,
                execution_epoch,
                is_safety=is_safety,
            ),
        )
    steps = workflow.safety_steps if is_safety else workflow.official_steps
    failure = reserve_restart_budgets(
        executor.store,
        workflow,
        incident,
        steps,
    )
    return ClaimedWorkflowPreparation(
        workflow=workflow,
        execution_epoch=execution_epoch,
        result=(
            fail_restart_preflight(
                executor,
                workflow,
                incident,
                execution_epoch,
                failure,
            )
            if failure is not None
            else None
        ),
    )


def fail_restart_preflight(
    executor: Any,
    workflow: WorkflowRequest,
    incident: FaultIncident,
    execution_epoch: int,
    failure: RestartBudgetPreflightFailure,
) -> Any:
    """Persist a restart preflight failure without invoking an adapter."""

    workflow = step_bounds.record_attempt(
        workflow,
        failure.step,
        failure.step_index,
        failure.outcome,
    )
    now = datetime.now(timezone.utc)
    workflow = workflow.model_copy(
        update={
            "status": WorkflowStatus.FAILED,
            "execution_owner_id": None,
            "execution_lease_expires_at": None,
            "updated_at": now,
        }
    )
    incident = incident.model_copy(
        update={
            "state": IncidentState.ESCALATED,
            "updated_at": now,
        }
    )
    executor._save_terminal(workflow, incident, execution_epoch)
    LOGGER.warning(
        "workflow failed restart budget preflight before adapter "
        "execution: workflow=%s step=%s error=%s",
        workflow.request_id,
        failure.step_index,
        failure.outcome.error,
    )
    return executor._result(
        workflow,
        incident,
        error=failure.outcome.error,
    )


def release_unattempted_restart_reservations(
    store: Any,
    workflow: WorkflowRequest,
    steps: Sequence[WorkflowStepSpec] | None = None,
    *,
    release_step_indexes: set[int] | None = None,
) -> None:
    """Release reservations for restart steps no adapter ever attempted."""

    selected = (
        list(steps)
        if steps is not None
        else (
            workflow.safety_steps
            if workflow.blocked_reasons
            else workflow.official_steps
        )
    )
    attempted = {item.step_index for item in workflow.step_executions}
    completed = set(workflow.completed_step_indexes)
    forced = release_step_indexes or set()
    for step_index, step in enumerate(selected):
        if (
            step.operation is not WorkflowOperation.RESTART_WORKLOAD
            or (step_index in attempted and step_index not in forced)
            or step_index in completed
        ):
            continue
        parameters = step.parameters
        if not {"cluster_id", "job_id"}.issubset(parameters):
            continue
        try:
            store.release_job_restart(
                str(parameters["cluster_id"]),
                str(parameters["job_id"]),
                _reservation_id(workflow, step_index),
            )
        except NotFoundError:
            continue

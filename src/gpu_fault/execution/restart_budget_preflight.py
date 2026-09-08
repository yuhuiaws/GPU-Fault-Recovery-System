from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Literal, Sequence

import gpu_fault.execution.step_bounds as step_bounds
from gpu_fault.execution.config import OPERATOR_ACKNOWLEDGEMENT_OPERATIONS
from gpu_fault.execution.models import WorkflowStepOutcome
from gpu_fault.execution.remediation_budget import remediation_budget_claims
from gpu_fault.models import (
    FaultIncident,
    IncidentState,
    WorkflowExecutionRequest,
    WorkflowOperation,
    WorkflowRequest,
    WorkflowStatus,
    WorkflowStepExecution,
    WorkflowStepSpec,
    WorkflowStepStatus,
    resolved_step_indexes,
)
from gpu_fault.store import NotFoundError

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


# Which step list a reservation belongs to. Safety and official steps share
# indexes (P0-62D), so a restart at index N in each phase is two different
# steps and must hold two different reservations (F-C9).
ReservationPhase = Literal["official", "safety"]


def reservation_phase(workflow: WorkflowRequest) -> ReservationPhase:
    """The phase whose steps this record executes, as the release sees it."""

    return "safety" if workflow.executes_safety_steps else "official"


def reservation_id(
    workflow: WorkflowRequest,
    step_index: int,
    *,
    phase: ReservationPhase = "official",
) -> str:
    """The reservation a RESTART_WORKLOAD step holds on its job's budget.

    Also the adapter's idempotency key for that step: the restart adapter
    reserves under its key, and the two have to agree or a step would take a
    second reservation. The official form is the historical
    ``<workflow>/<index>/RESTART_WORKLOAD`` so rows reserved before the phase
    existed still match; only the safety phase carries a discriminator.
    """

    operation = WorkflowOperation.RESTART_WORKLOAD.value
    if phase == "safety":
        return f"{workflow.request_id}/safety/{step_index}/{operation}"
    return f"{workflow.request_id}/{step_index}/{operation}"


_reservation_id = reservation_id


def claim_deadlines(
    workflow: WorkflowRequest,
    now: datetime,
    *,
    timeout_seconds: float,
    job_lifetime_seconds: float,
    node_lifetime_seconds: float,
    operator_acknowledgement_seconds: float | None = None,
) -> tuple[datetime, datetime]:
    """``(execution_deadline, lifetime_deadline_at)`` for a claim (F-N1).

    The lifetime is stamped once, at the first claim, from the workflow's kind
    (job vs single node) and inherited afterwards. The execution deadline is
    the usual per-claim budget capped by the lifetime. A workflow that contains
    an operator-acknowledgement step (CHECK_MECHANICALS) waits on a human, so
    both deadlines are floored at ``now + operator_acknowledgement_seconds``:
    a one-hour lifetime would otherwise fail the very step whose meaning is
    "wait for the inspection".
    """

    lifetime = workflow.lifetime_deadline_at
    if lifetime is None:
        is_job = workflow.dag_enabled or any(
            step.operation is WorkflowOperation.RESTART_WORKLOAD
            for step in workflow.official_steps
        )
        lifetime = now + timedelta(
            seconds=job_lifetime_seconds if is_job else node_lifetime_seconds
        )
    budget = timedelta(seconds=timeout_seconds)
    existing = workflow.execution_deadline
    fresh_execution = workflow.dag_enabled or existing is None
    execution = now + budget if (workflow.dag_enabled or existing is None) else existing
    if operator_acknowledgement_seconds is not None and any(
        step.operation in OPERATOR_ACKNOWLEDGEMENT_OPERATIONS
        for step in workflow.official_steps
    ):
        floor = now + timedelta(seconds=operator_acknowledgement_seconds)
        lifetime = max(lifetime, floor)
        # The floor belongs to the claim that stamps the deadline. Re-applying
        # it on every later claim slid the deadline forward each dispatcher
        # cycle, so the inspection step could never reach its own ceiling and
        # ``step_bounds`` measured its wait from the latest claim (observed
        # live as step_waiting_seconds = -84599, then 0).
        if fresh_execution:
            execution = max(execution, floor)
    return min(execution, lifetime), lifetime


def reserve_restart_budgets(
    store: Any,
    workflow: WorkflowRequest,
    incident: FaultIncident,
    steps: Sequence[WorkflowStepSpec],
    *,
    phase: ReservationPhase | None = None,
) -> RestartBudgetPreflightFailure | None:
    """Reserve every pending restart before any workflow adapter runs.

    ``phase`` names the list ``steps`` came from; it defaults to the one the
    record executes.
    """

    if phase is None:
        phase = reservation_phase(workflow)
    # A superseded restart will never run, so it needs no reservation (F-C2).
    completed = set(resolved_step_indexes(workflow))
    # A restart whose adapter already ran holds its reservation -- the adapter
    # reserves under the same id -- and is only released by a terminal write,
    # so re-reserving it on every claim was one budget write per tick for
    # nothing (D-10).
    already_attempted = {
        item.step_index
        for item in workflow.step_executions
        if item.operation is WorkflowOperation.RESTART_WORKLOAD
        and item.status is WorkflowStepStatus.WAITING
        and item.phase in (None, phase)
    }
    for step_index, step in enumerate(steps):
        if (
            step.operation is not WorkflowOperation.RESTART_WORKLOAD
            or step_index in completed
            or step_index in already_attempted
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
                reservation_id(workflow, step_index, phase=phase),
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
    )
    deadlines = claim_deadlines(
        workflow,
        datetime.now(timezone.utc),
        timeout_seconds=executor.config.workflow_execution_timeout_seconds,
        job_lifetime_seconds=executor.config.job_workflow_lifetime_seconds,
        node_lifetime_seconds=executor.config.node_workflow_lifetime_seconds,
        operator_acknowledgement_seconds=(
            executor.config.operator_acknowledgement_timeout_seconds
        ),
    )
    workflow = workflow.model_copy(
        update={
            "status": WorkflowStatus.RUNNING,
            "execution_deadline": deadlines[0],
            "lifetime_deadline_at": deadlines[1],
            "updated_at": datetime.now(timezone.utc),
        }
    )
    execution_epoch = workflow.execution_epoch
    workflow = executor._adopt_quiesce_handoff_from_predecessor(workflow, incident)
    executor._save_leased(workflow, execution_epoch)
    if (
        workflow.pending_failure_step_index is not None
        and not executor._job_failure_still_deferred(workflow)
    ):
        return ClaimedWorkflowPreparation(
            workflow=workflow,
            execution_epoch=execution_epoch,
            result=executor._land_pending_failure(
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
        phase="safety" if is_safety else "official",
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
    result = executor._terminalize(
        workflow,
        incident,
        WorkflowStatus.FAILED,
        execution_epoch,
        reason=failure.outcome.error,
        incident_state=IncidentState.ESCALATED,
    )
    LOGGER.warning(
        "workflow failed restart budget preflight before adapter "
        "execution: workflow=%s step=%s error=%s",
        workflow.request_id,
        failure.step_index,
        failure.outcome.error,
    )
    return result


# The bounds in ``step_bounds`` stamp these when they, not the adapter, end a
# step; a RESTART_WORKLOAD record carrying one never reached a submission.
_WAITING_CAP_DETAIL = "step_waiting_timeout_seconds"


def _restart_never_left_the_gate(
    record: WorkflowStepExecution,
    *,
    now: datetime,
    waiting_ttl: timedelta | None,
) -> bool:
    """Does this RESTART_WORKLOAD record show a restart that never happened?

    The restart adapter answers WAITING only before it submits anything -- an
    approval is pending, or the incident it depends on is not recovered -- so a
    wait that outlived the step's cap, or that the cap already turned into a
    failure, holds budget for a restart nobody made (F-C9). A remote command
    that a cluster executor is running is the one shape that says nothing
    about submission, and keeps its reservation.
    """

    if str(record.details.get("remote_status") or "") == "RUNNING":
        return False
    if record.status is WorkflowStepStatus.WAITING:
        return waiting_ttl is not None and now - record.started_at >= waiting_ttl
    if record.status is WorkflowStepStatus.FAILED:
        return _WAITING_CAP_DETAIL in record.details
    return False


def release_unattempted_restart_reservations(
    store: Any,
    workflow: WorkflowRequest,
    steps: Sequence[WorkflowStepSpec] | None = None,
    *,
    release_step_indexes: set[int] | None = None,
    waiting_ttl: timedelta | None = None,
    now: datetime | None = None,
) -> None:
    """Release reservations for restart steps no adapter ever attempted.

    A step whose only record is a wait older than ``waiting_ttl`` -- or one the
    waiting cap already failed -- counts as unattempted too: the adapter never
    submitted its restart (see ``_restart_never_left_the_gate``). Without a
    ``waiting_ttl`` a WAITING record keeps its reservation.
    """

    selected = (
        list(steps)
        if steps is not None
        else (
            workflow.safety_steps
            if workflow.executes_safety_steps
            else workflow.official_steps
        )
    )
    phase = reservation_phase(workflow)
    moment = now if now is not None else datetime.now(timezone.utc)
    # Only a RESTART_WORKLOAD record counts as "attempted": a safety-phase
    # step at the same index is a different step (P0-62D). The last record at
    # an index is the live one (see ``step_bounds.previous_execution``).
    latest: dict[int, WorkflowStepExecution] = {}
    for item in workflow.step_executions:
        if item.operation is WorkflowOperation.RESTART_WORKLOAD:
            latest[item.step_index] = item
    completed = set(workflow.completed_step_indexes)
    forced = release_step_indexes or set()
    for step_index, step in enumerate(selected):
        if (
            step.operation is not WorkflowOperation.RESTART_WORKLOAD
            or step_index in completed
        ):
            continue
        record = latest.get(step_index)
        if (
            record is not None
            and step_index not in forced
            and not _restart_never_left_the_gate(
                record, now=moment, waiting_ttl=waiting_ttl
            )
        ):
            continue
        parameters = step.parameters
        if not {"cluster_id", "job_id"}.issubset(parameters):
            continue
        try:
            store.release_job_restart(
                str(parameters["cluster_id"]),
                str(parameters["job_id"]),
                reservation_id(workflow, step_index, phase=phase),
            )
        except NotFoundError:
            continue

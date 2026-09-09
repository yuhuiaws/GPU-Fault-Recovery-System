from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from typing import Any, Literal, Sequence

import gpu_fault.execution.step_bounds as step_bounds
from gpu_fault.execution.config import OPERATOR_ACKNOWLEDGEMENT_OPERATIONS
from gpu_fault.execution.models import WorkflowStepOutcome
from gpu_fault.execution.remediation_budget import remediation_budget_claims
from gpu_fault.models import (
    FaultIncident,
    IncidentState,
    RestartAuthorization,
    WorkflowExecutionRequest,
    WorkflowOperation,
    WorkflowRequest,
    WorkflowStatus,
    WorkflowStepExecution,
    WorkflowStepSpec,
    WorkflowStepStatus,
    resolved_step_indexes,
)
from gpu_fault.remote_command_models import RemoteCommandStatus
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
    compares its authorization's ``reservation_id`` with that key, so the two
    have to agree or the restart is refused. The official form is the historical
    ``<workflow>/<index>/RESTART_WORKLOAD`` so rows reserved before the phase
    existed still match; only the safety phase carries a discriminator.
    """

    operation = WorkflowOperation.RESTART_WORKLOAD.value
    if phase == "safety":
        return f"{workflow.request_id}/safety/{step_index}/{operation}"
    return f"{workflow.request_id}/{step_index}/{operation}"


def issue_restart_authorization(
    store: Any,
    incident: FaultIncident,
    step: WorkflowStepSpec,
    reservation_id: str,
) -> RestartAuthorization | WorkflowStepOutcome:
    """The proof a data-plane restart carries: the preflight's reservation.

    Only ``reserve_restart_budgets`` reserves; this reads that reservation
    back and signs it. A step without one is a step the preflight never
    admitted (or whose reservation was released as unattempted) and fails
    closed rather than reserving here.
    """

    parameters = step.parameters
    missing = _REQUIRED_RESTART_PARAMETERS - set(parameters)
    if missing:
        return WorkflowStepOutcome.failed(
            "restart safety context is missing: " + ", ".join(sorted(missing)),
            details={
                "reason": "RESTART_SAFETY_CONTEXT_MISSING",
                "missing_parameters": sorted(missing),
            },
        )
    cluster_id = str(parameters["cluster_id"])
    job_id = str(parameters["job_id"])
    # The same gates the claim preflight applied, in the same words: the gate
    # trusts the incident over the plan, and a malformed plan is a FAILED step
    # with a reason rather than an adapter traceback.
    if cluster_id != incident.cluster_id:
        return WorkflowStepOutcome.failed(
            "restart safety context cluster does not match "
            f"incident: {cluster_id} != {incident.cluster_id}",
            details={
                "reason": "RESTART_CLUSTER_MISMATCH",
                "restart_cluster_id": cluster_id,
                "incident_cluster_id": incident.cluster_id,
            },
        )
    try:
        source_gpu_count = int(parameters["source_gpu_count"])
        restart_budget = int(parameters["restart_budget"])
    except (TypeError, ValueError) as exc:
        return WorkflowStepOutcome.failed(
            f"restart safety context is invalid: {exc}",
            details={"reason": "RESTART_SAFETY_CONTEXT_INVALID"},
        )
    try:
        state = store.get_restart_budget(cluster_id, job_id)
    except NotFoundError:
        state = None
    if state is None or reservation_id not in state.reservation_ids:
        return WorkflowStepOutcome.failed(
            f"restart reservation missing for {cluster_id}/{job_id}: {reservation_id}",
            details={
                "reason": "RESTART_RESERVATION_MISSING",
                "reservation_id": reservation_id,
                "restart_count": state.restart_count if state is not None else 0,
                "restart_budget": state.budget if state is not None else restart_budget,
            },
        )
    return RestartAuthorization(
        cluster_id=cluster_id,
        job_id=job_id,
        source_attempt_id=str(parameters["source_attempt_id"]),
        source_gpu_count=source_gpu_count,
        restart_budget=state.budget,
        restart_count=state.restart_count,
        reservation_id=reservation_id,
    )


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
    # A restart whose adapter already ran still holds the reservation made on
    # its first claim (only a terminal write releases it), so re-reserving it
    # on every claim was one budget write per tick for nothing (D-10).
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
    """Persist a restart preflight failure without invoking an adapter.

    An exhausted budget is also the operator's business: the preflight is the
    only site that refuses a restart for it, so the preflight sends the mail
    (the restart adapter no longer reserves, and so never learns the budget
    is gone).
    """

    outcome = failure.outcome
    details = outcome.details or {}
    if details.get("reason") == "RESTART_BUDGET_EXHAUSTED":
        parameters = failure.step.parameters
        notification = executor.restart_email_builder.build_budget_exhausted(
            cluster_id=incident.cluster_id,
            incident_id=incident.incident_id,
            job_id=str(parameters["job_id"]),
            attempt_id=str(parameters["source_attempt_id"]),
            restart_count=int(details["restart_count"]),
            restart_budget=int(details["restart_budget"]),
        )
        notification = executor.store.save_notification_if_absent(notification)
        if executor.notification_sender is not None:
            executor.notification_sender(notification.notification_id)
        outcome = replace(
            outcome,
            details={**details, "notification_id": notification.notification_id},
        )
    workflow = step_bounds.record_attempt(
        workflow,
        failure.step,
        failure.step_index,
        outcome,
    )
    result = executor._terminalize(
        workflow,
        incident,
        WorkflowStatus.FAILED,
        execution_epoch,
        reason=outcome.error,
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
# The restart adapter stamps this on every refusal, and every hold, it makes
# before submitting anything; the regional path carries it verbatim from the
# data plane's ``RemoteCommandResult.details`` into the step record.
_NOT_SUBMITTED_DETAIL = "restart_submitted"
# ``remote_status_source`` values of a FAILED remote command no adapter ever
# ran: the cluster executor refused it before dispatch (its checks are
# deterministic per command, so a later lease would have refused it too), or it
# aged out PENDING because no executor ever claimed it.
_NEVER_DISPATCHED_STATUS_SOURCES = frozenset(
    {"executor-rejected", "unclaimed-deadline-exceeded"}
)
# ``remote_status_source`` values of a command the workflow cancelled while it
# was PENDING or WAITING. Cancellation keeps the command's last report as its
# ``result_details``, so those details say what the node was doing.
_CANCELLED_STATUS_SOURCES = frozenset({"workflow-timeout", "workflow-preempted"})


def _wait_was_before_submission(details: dict[str, Any]) -> bool:
    """Does a WAITING report (or the cap failure copied from one) prove that
    the restart adapter had not submitted anything?

    The adapter's own holds carry ``restart_submitted: False``. Two shapes
    override or outrank that: a retryable adapter error is a WAITING the
    *executor* answered for an exception that may have been raised after
    ``create_namespaced_job`` (a 5xx on the second workload, a client
    timeout after the API server persisted the Job), and a remote command
    that is LEASED is on a data-plane executor right now -- its last report
    may be a hold the executor has since moved past. A PENDING remote command
    was never leased, so no adapter ran. Anything else says nothing, and
    nothing means keep.
    """

    if details.get("retryable_adapter_error"):
        return False
    remote_status = details.get("remote_status")
    if remote_status == RemoteCommandStatus.LEASED.value:
        return False
    if remote_status == RemoteCommandStatus.PENDING.value:
        return True
    return details.get(_NOT_SUBMITTED_DETAIL) is False


def _restart_never_left_the_gate(
    record: WorkflowStepExecution,
    *,
    now: datetime,
    waiting_ttl: timedelta | None,
) -> bool:
    """Does this RESTART_WORKLOAD record show a restart that never happened?

    Release needs positive evidence that nothing was submitted; a record that
    says nothing keeps the budget spent, by decision (F-C9, Task 13 C-1).

    A WAITING record older than ``waiting_ttl``, and a FAILED record the
    waiting cap produced from one, release only when the wait was one of the
    adapter's own pre-submission holds (``restart_submitted: False``) or the
    remote command was never leased (``remote_status`` PENDING); a wait the
    executor answered for a retryable adapter error, or whose remote command
    is LEASED, may hide a Job that exists (``_wait_was_before_submission``).

    A FAILED record otherwise releases on the adapter's own marker (its
    refusals carry it; the regional path copies it verbatim from the data
    plane's ``RemoteCommandResult.details``), on a remote command no adapter
    ever ran (executor-rejected, unclaimed), or on a cancelled command whose
    last report carries the marker or that never left PENDING (no report at
    all). A cancellation the node settled afterwards, an executor-internal or
    configuration error, a stale-fence settle and a plain node failure keep
    the budget spent.
    """

    details = record.details
    if record.status is WorkflowStepStatus.WAITING:
        if waiting_ttl is None or now - record.started_at < waiting_ttl:
            return False
        return _wait_was_before_submission(details)
    if record.status is not WorkflowStepStatus.FAILED:
        return False
    if _WAITING_CAP_DETAIL in details:
        # ``bounded_waiting_outcome`` copies the last WAITING's details into
        # this failure, so the same evidence rule applies.
        return _wait_was_before_submission(details)
    if details.get(_NOT_SUBMITTED_DETAIL) is False:
        return True
    source = details.get("remote_status_source")
    if source in _NEVER_DISPATCHED_STATUS_SOURCES:
        return True
    if source in _CANCELLED_STATUS_SOURCES:
        # ``RemoteActionCommand.result_details`` starts empty and nothing ever
        # returns a command to PENDING: no report means no lease, no adapter.
        return set(details) <= {"remote_status_source"}
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

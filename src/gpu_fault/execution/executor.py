from __future__ import annotations

import logging
import os
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from threading import RLock
from typing import Any
from uuid import uuid4

from pydantic import ValidationError

import gpu_fault.execution.restart_budget_preflight as restart_preflight
import gpu_fault.execution.step_bounds as step_bounds
from gpu_fault.execution.branch_escalation import BranchEscalation, BranchEscalator
from gpu_fault.execution import node_rebinding
from gpu_fault.execution.config import (
    ProductionExecutorConfig,
)
from gpu_fault.execution.fleet_preflight import (
    held_workflow_result,
)
from gpu_fault.execution.hung_classification import (
    classify_hung_signals,
)
from gpu_fault.execution.invariants import check_workflow_invariants
from gpu_fault.execution.models import (
    WorkflowExecutionError,
    WorkflowRecordInvalidError,
    WorkflowStepAdapter,
    WorkflowStepContext,
    WorkflowStepOutcome,
    WorkflowStructureError,
    _failure_details,
)
from gpu_fault.execution.remediation_budget import escalation_budget_claims
from gpu_fault.execution.transient_errors import (
    retryable_adapter_error,
    transient_store_error,
)
from gpu_fault.markers import retire_markers_for_incident
from gpu_fault.models import (
    BlockedKind,
    FaultIncident,
    IncidentState,
    WorkflowEventCode,
    WorkflowEventKind,
    WorkflowExecutionRequest,
    WorkflowExecutionResult,
    WorkflowOperation,
    WorkflowRequest,
    WorkflowStatus,
    WorkflowStepExecution,
    WorkflowStepSpec,
    WorkflowStepStatus,
    bounded_reasons,
    record_workflow_event,
    resolved_step_indexes,
)
from gpu_fault.notifications import (
    DiagnosticInconclusiveEmailBuilder,
    RestartGuardEmailBuilder,
    WarmSpareReplacementEmailBuilder,
)
from gpu_fault.operation_registry import (
    DESTRUCTIVE_OPERATIONS,
    NODE_WIDE_RECOVERY_OPERATIONS,
    SAFE_REMOTE_WAITING_PREEMPT_OPERATIONS,
)
from gpu_fault.orchestration import WorkflowFencingError
from gpu_fault.orchestration.preemption_boundary import (
    PreemptionBoundary,
    WaitingRecord,
    preemption_boundary,
)
from gpu_fault.store import NotFoundError
from gpu_fault.store.contracts import ControlPlaneStore
from gpu_fault.store.shared.errors import RemediationBudgetError
from gpu_fault.workflow_resolution import retirement_fences_out_dispatch

LOGGER = logging.getLogger(__name__)

#: Reason an incident carries when its only workflow observed the node and
#: failed to conclude; the TERMINAL event and the retired markers say the same.
DIAGNOSTIC_INCONCLUSIVE_REASON = "diagnostic inconclusive"
# Upper bound on the steps one DAG may carry. Appended branches and in-place
# escalation rungs grow the graph; the validator and the ready-set scan are
# both quadratic in its size, so a graph with no bound is a runaway
# workflow's way of pinning a dispatcher worker (F-C6).
MAX_DAG_STEPS = 256

# One vocabulary for "this workflow is waiting on a node someone else is
# repairing" (review item 4). The dispatcher's busy-node hold and the restart
# adapter's ``requires_incident_state`` premise are different product
# decisions and stay separate mechanisms; they share these codes in
# ``details["reason"]`` and the ``HOLD`` audit event below so an operator can
# follow either wait to the remediation workflow it is waiting on. The codes
# are ``WorkflowEventCode`` values so the Kubernetes restart adapter, which
# must not depend on this module, spells them from ``gpu_fault.models``.
HOLD_REASON_NODE_UNDER_REMEDIATION = WorkflowEventCode.NODE_UNDER_REMEDIATION.value
HOLD_REASON_NODE_REMEDIATION_TIMEOUT = WorkflowEventCode.NODE_REMEDIATION_TIMEOUT.value
# No event is written for this one: the step fails and TERMINAL follows.
HOLD_REASON_INCIDENT_NOT_RECOVERABLE = "INCIDENT_NOT_RECOVERABLE"

# Actors stamped on TERMINAL events written from outside the executor's own
# ``execute()``; the executor itself signs with its ``executor_id``.
DISPATCHER_WATCHDOG_ACTOR = "dispatcher-watchdog"
DISPATCHER_INTERNAL_ERROR_ACTOR = "dispatcher-internal-error"
DISPATCHER_ACTOR = "dispatcher"

# D-8: ``status_source`` values a remote command carries when the workflow
# itself cancelled it. A step failed for one of these is not a node failure.
WORKFLOW_CANCELLATION_STATUS_SOURCES = frozenset(
    {"workflow-preempted", "workflow-timeout"}
)

# ARCH-B1: ``details["reason"]`` of a step that is waiting only because its
# adapter hit a retryable error (a Kubernetes 5xx, a torn connection). Not a
# hold: nothing is being waited on but the next dispatcher tick.
STEP_RETRY_REASON_ADAPTER_ERROR = "RETRYABLE_ADAPTER_ERROR"

# ARCH-B3: called from the one place every terminal write lands, with the
# workflow as written, the incident as written (``None`` when the incident row
# is gone or belongs to a successor) and the steps the workflow executed.
TerminalHook = Callable[
    [WorkflowRequest, FaultIncident | None, list[WorkflowStepSpec]], None
]


def configured_step_transient_retry_limit(values: Mapping[str, str]) -> int:
    """How many consecutive retryable adapter errors one step may absorb.

    Past the limit the last error fails the step on the established path. Zero
    disables the retry, which is the pre-ARCH-B1 behaviour.
    """

    limit = int(values.get("GPU_FAULT_WORKFLOW_STEP_TRANSIENT_RETRY_LIMIT", "5"))
    if limit < 0:
        raise WorkflowExecutionError(
            "GPU_FAULT_WORKFLOW_STEP_TRANSIENT_RETRY_LIMIT must not be negative"
        )
    return limit


def record_hold_event(
    workflow: WorkflowRequest,
    *,
    reason: str,
    remediation_workflow_id: str | None,
    actor: str,
    step_index: int | None = None,
    operation: WorkflowOperation | None = None,
    details: Mapping[str, object] | None = None,
) -> WorkflowRequest | None:
    """Append one ``HOLD`` event, or ``None`` when the last HOLD already says it.

    A hold is polled -- every dispatcher tick, every re-dispatch of a WAITING
    step -- so recording each observation would write sixty events for one
    five-minute wait. The key is (reason, remediation workflow): a new event
    is worth having only when the wait changed what it waits on.
    """

    last = next(
        (
            event
            for event in reversed(workflow.events)
            if event.kind is WorkflowEventKind.HOLD
        ),
        None,
    )
    if (
        last is not None
        and last.details.get("reason") == reason
        and last.details.get("remediation_workflow_id") == remediation_workflow_id
    ):
        return None
    return record_workflow_event(
        workflow,
        WorkflowEventKind.HOLD,
        code=reason,
        reason=reason,
        actor=actor,
        step_index=step_index,
        operation=operation,
        details={
            **(dict(details) if details else {}),
            "reason": reason,
            "remediation_workflow_id": remediation_workflow_id,
        },
    )


class ProductionWorkflowExecutor:
    """Execute owned workflow steps with durable, fail-closed gates."""

    def __init__(
        self,
        store: ControlPlaneStore,
        adapters: list[WorkflowStepAdapter],
        config: ProductionExecutorConfig,
        *,
        notification_sender=None,
        step_transient_retry_limit: int | None = None,
    ) -> None:
        if config.workflow_execution_timeout_seconds <= 0:
            raise WorkflowExecutionError("workflow execution timeout must be positive")
        self.store = store
        self.adapters = adapters
        self.config = config
        # ARCH-B1: consecutive retryable adapter errors one step may absorb
        # before the last one fails it; from the environment unless given.
        self.step_transient_retry_limit = (
            step_transient_retry_limit
            if step_transient_retry_limit is not None
            else configured_step_transient_retry_limit(os.environ)
        )
        if self.step_transient_retry_limit < 0:
            raise WorkflowExecutionError(
                "step transient retry limit must not be negative"
            )
        # ARCH-B3: release hooks for state that outlives the workflow record
        # (a warm-spare reservation, a Kubernetes-side lease). The application
        # context registers them; every terminal write calls each one.
        self.on_terminal: list[TerminalHook] = []
        # F-N1: set by the application context; ``None`` keeps the
        # whole-workflow failure semantics for a failed node branch.
        self.branch_escalator: BranchEscalator | None = None
        # F-N1: workflows failed because their hard lifetime passed.
        self.lifetime_exceeded_total = 0
        self.branch_escalation_budget_refusals_total = 0
        self.notification_sender = notification_sender
        self.warm_spare_email_builder = WarmSpareReplacementEmailBuilder()
        self.diagnostic_inconclusive_email_builder = (
            DiagnosticInconclusiveEmailBuilder()
        )
        # The budget-exhausted mail: sent by the restart preflight, which is
        # the only site that refuses a restart for a spent budget, using the
        # same template the restart adapters carry rather than a second one.
        self.restart_email_builder = RestartGuardEmailBuilder()
        # D-7: set by the dispatcher; the lease a WAITING row keeps until the
        # next tick. ``None`` keeps the full ``lease_duration_seconds``.
        self.waiting_lease_duration: timedelta | None = None
        self._lock = RLock()
        # One lock per workflow: the dispatcher's worker pool shares this
        # instance, and a single lock around execute() ran eight workers one
        # workflow at a time (F-C7). Entries are dropped when nobody holds them.
        self._workflow_locks: dict[str, tuple[RLock, int]] = {}

    @contextmanager
    def _workflow_lock(self, request_id: str) -> Iterator[None]:
        with self._lock:
            lock, holders = self._workflow_locks.get(request_id, (RLock(), 0))
            self._workflow_locks[request_id] = (lock, holders + 1)
        try:
            with lock:
                yield
        finally:
            with self._lock:
                lock, holders = self._workflow_locks[request_id]
                if holders <= 1:
                    del self._workflow_locks[request_id]
                else:
                    self._workflow_locks[request_id] = (lock, holders - 1)

    def execute(
        self,
        request_id: str,
        request: WorkflowExecutionRequest,
    ) -> WorkflowExecutionResult:
        with self._workflow_lock(request_id):
            if not self.config.enabled:
                raise WorkflowExecutionError("production workflow executor is disabled")
            try:
                workflow = self.store.get_workflow(request_id)
            except ValidationError as exc:
                # The one decode failure that is about this record; the
                # dispatcher blocks on this class, not on ValidationError.
                raise WorkflowRecordInvalidError(
                    f"workflow {request_id} cannot be decoded: {exc}"
                ) from exc
            incident = self.store.get_incident(workflow.incident_id)
            self._validate_fencing(workflow, incident, request)

            if workflow.status in {
                WorkflowStatus.SUCCEEDED,
                WorkflowStatus.BLOCKED,
                WorkflowStatus.SUPERSEDED,
            }:
                return self._result(workflow, incident)
            if workflow.status is WorkflowStatus.FAILED:
                return self._result(
                    workflow,
                    incident,
                    error="workflow is FAILED and requires a new fencing token",
                )
            if workflow.status not in {
                WorkflowStatus.PENDING,
                WorkflowStatus.SAFETY_PENDING,
                WorkflowStatus.RUNNING,
            }:
                raise WorkflowExecutionError(
                    f"workflow is not executable from {workflow.status.value}"
                )

            is_safety = workflow.executes_safety_steps
            steps = workflow.safety_steps if is_safety else workflow.official_steps
            if not steps:
                raise WorkflowExecutionError("workflow has no executable steps")
            if held := held_workflow_result(self, workflow, incident, steps):
                return held
            prepared = restart_preflight.prepare_claimed_workflow(
                self,
                request_id,
                request,
                workflow,
                incident,
                is_safety=is_safety,
            )
            if prepared.result is not None:
                return prepared.result
            execution_epoch = prepared.execution_epoch
            if execution_epoch != workflow.execution_epoch:
                # A new execution epoch: this executor took the row over (or
                # took it first). The claim itself happens inside the
                # preflight, which is not this module's to edit; the event is
                # written here, in its own leased save, because the loops
                # below re-read the row on their first lease renewal and
                # would drop an unsaved event. A redispatch of a WAITING row
                # by the same holder opens no epoch and writes no event and
                # no second row (D-10).
                workflow = record_workflow_event(
                    prepared.workflow,
                    WorkflowEventKind.CLAIM,
                    code=WorkflowEventCode.CLAIMED.value,
                    actor=self.config.executor_id,
                    status=prepared.workflow.status.value,
                    details={"execution_epoch": execution_epoch},
                )
                self._save_leased(workflow, execution_epoch)
            else:
                workflow = prepared.workflow
            steps = workflow.safety_steps if is_safety else workflow.official_steps
            if workflow.dag_enabled:
                return self._execute_dag(
                    workflow,
                    incident,
                    request,
                    is_safety=is_safety,
                    execution_epoch=execution_epoch,
                )

            if workflow.workload_withdrawn_at is not None:
                trimmed = self._apply_workload_withdrawal(workflow, steps)
                if trimmed is not workflow:
                    workflow = trimmed
                    self._save_leased(workflow, execution_epoch)
            for index in range(len(steps)):
                step = steps[index]
                # A superseded step never runs; skip it like a completed one
                # (F-C2) -- the DAG loop already did.
                if index in resolved_step_indexes(workflow):
                    continue
                workflow = self.store.renew_workflow_lease(
                    request_id,
                    self.config.executor_id,
                    execution_epoch,
                    lease_duration=self._lease_duration,
                )
                if workflow.dag_enabled:
                    self._save_leased(workflow, execution_epoch)
                    return self._result(workflow, incident)
                if workflow.workload_withdrawn_at is not None:
                    # Re-read on every step, as the DAG loop does (F-N1 §7): a
                    # stop that landed while the previous step ran must keep
                    # this one -- the job restart included -- from starting
                    # (control-plane review 2026-09-08, D-2).
                    trimmed = self._apply_workload_withdrawal(workflow, steps)
                    if trimmed is not workflow:
                        workflow = trimmed
                        self._save_leased(workflow, execution_epoch)
                    if index in resolved_step_indexes(workflow):
                        continue
                preempted = self._supersede_if_safe(
                    workflow,
                    incident,
                    request,
                    step,
                    index,
                    execution_epoch,
                )
                if preempted is not None:
                    return preempted
                outcome = self._execute_step(workflow, incident, request, step, index)
                outcome = self._persist_step_completion_notification(
                    workflow, incident, step, index, outcome
                )
                workflow = self.store.renew_workflow_lease(
                    request_id,
                    self.config.executor_id,
                    execution_epoch,
                    lease_duration=self._lease_duration,
                )
                workflow = step_bounds.record_attempt(workflow, step, index, outcome)
                if outcome.status is WorkflowStepStatus.WAITING:
                    workflow = self._record_hold(workflow, step, index, outcome)
                    self._save_waiting(workflow, execution_epoch)
                    return self._result(
                        workflow,
                        incident,
                        waiting_step_index=index,
                    )
                if outcome.status is WorkflowStepStatus.FAILED:
                    if (
                        step.operation is not WorkflowOperation.RESTORE_GPU_SERVICES
                        and self._has_unrestored_quiesce(workflow, steps)
                    ):
                        workflow = workflow.model_copy(
                            update={
                                "pending_failure_step_index": index,
                                "pending_failure_error": (
                                    outcome.error or "workflow step failed"
                                ),
                                "updated_at": datetime.now(timezone.utc),
                            }
                        )
                        self._save_leased(workflow, execution_epoch)
                        return self._resume_failure_compensation(
                            workflow,
                            incident,
                            request,
                            execution_epoch,
                            is_safety=is_safety,
                        )
                    return self._terminalize(
                        workflow,
                        incident,
                        WorkflowStatus.FAILED,
                        execution_epoch,
                        reason=outcome.error,
                    )

                rebindings = (outcome.details or {}).get("node_rebindings", {})
                incident_rebound = bool(rebindings)
                if rebindings:
                    workflow, incident = self._rebind_nodes(
                        workflow,
                        incident,
                        rebindings,
                        is_safety=is_safety,
                        after_index=index,
                    )
                    steps = (
                        workflow.safety_steps if is_safety else workflow.official_steps
                    )

                completed_indexes = [
                    *workflow.completed_step_indexes,
                    index,
                ]
                completed_operations = [
                    *workflow.completed_operations,
                    step.operation,
                ]
                workflow = workflow.model_copy(
                    update={
                        "completed_step_indexes": completed_indexes,
                        "completed_operations": completed_operations,
                        "updated_at": datetime.now(timezone.utc),
                    }
                )
                if incident_rebound:
                    self._save_leased_state(workflow, incident, execution_epoch)
                else:
                    self._save_leased(workflow, execution_epoch)

            if workflow.workload_withdrawn_at is not None:
                return self._finish_withdrawn(workflow, incident, execution_epoch)
            return self._complete_claimed_workflow(
                workflow,
                incident,
                is_safety=is_safety,
                execution_epoch=execution_epoch,
            )

    def _execute_dag(
        self,
        workflow: WorkflowRequest,
        incident: FaultIncident,
        request: WorkflowExecutionRequest,
        *,
        is_safety: bool,
        execution_epoch: int,
    ) -> WorkflowExecutionResult:
        attempted: set[int] = set()
        validated_revision: int | None = None
        while True:
            workflow = self.store.renew_workflow_lease(
                workflow.request_id,
                self.config.executor_id,
                execution_epoch,
                lease_duration=self._lease_duration,
            )
            steps = workflow.safety_steps if is_safety else workflow.official_steps
            # Validate the graph once per shape, not once per step: it only
            # changes when a branch is appended, replaced or retired, which
            # bumps ``dag_revision`` (F-C6).
            if validated_revision != workflow.dag_revision:
                self._validate_dag(steps)
                validated_revision = workflow.dag_revision
            if workflow.workload_withdrawn_at is not None:
                trimmed = self._apply_workload_withdrawal(workflow, steps)
                if trimmed is not workflow:
                    workflow = trimmed
                    self._save_leased(workflow, execution_epoch)
            resolved = set(resolved_step_indexes(workflow))
            # Set coverage, not a count: an index outside the step list (a
            # stale entry after a DAG rewrite) must not end the workflow with
            # a real step still pending (P0-65B).
            if resolved >= set(range(len(steps))):
                if workflow.pending_failure_step_index is not None:
                    # The in-flight siblings a job-level failure waited for
                    # have settled (D-3): the parked verdict lands now.
                    return self._land_pending_failure(
                        workflow,
                        incident,
                        request,
                        execution_epoch,
                        is_safety=is_safety,
                    )
                if workflow.workload_withdrawn_at is not None:
                    return self._finish_withdrawn(workflow, incident, execution_epoch)
                if workflow.exhausted_branch_ids:
                    return self._fail_exhausted_branches(
                        workflow, incident, execution_epoch
                    )
                return self._complete_claimed_workflow(
                    workflow,
                    incident,
                    is_safety=is_safety,
                    execution_epoch=execution_epoch,
                )
            ready = [
                index
                for index, step in enumerate(steps)
                if index not in resolved
                and index not in attempted
                and set(step.depends_on_step_indexes) <= resolved
            ]
            if workflow.exhausted_branch_ids:
                # A node's ladder is exhausted: the job must not restart on
                # it. The join waits out the other branches, then the
                # workflow fails (F-N1).
                ready = [index for index in ready if steps[index].branch_id != "join"]
                if not ready and all(
                    index in resolved
                    for index, step in enumerate(steps)
                    if step.branch_id != "join"
                ):
                    return self._fail_exhausted_branches(
                        workflow, incident, execution_epoch
                    )
            if not ready:
                self._save_waiting(workflow, execution_epoch)
                waiting = min(
                    (
                        item.step_index
                        for item in workflow.step_executions
                        if item.status is WorkflowStepStatus.WAITING
                    ),
                    default=None,
                )
                return self._result(
                    workflow,
                    incident,
                    waiting_step_index=waiting,
                )
            batch_failure: tuple[int, WorkflowStepOutcome] | None = None
            for index in ready:
                attempted.add(index)
                step = steps[index]
                preempted = self._supersede_if_safe(
                    workflow,
                    incident,
                    request,
                    step,
                    index,
                    execution_epoch,
                )
                if preempted is not None:
                    return preempted
                outcome = self._execute_step(workflow, incident, request, step, index)
                outcome = self._persist_step_completion_notification(
                    workflow, incident, step, index, outcome
                )
                workflow = self.store.renew_workflow_lease(
                    workflow.request_id,
                    self.config.executor_id,
                    execution_epoch,
                    lease_duration=self._lease_duration,
                )
                workflow = step_bounds.record_attempt(workflow, step, index, outcome)
                if (
                    outcome.status is WorkflowStepStatus.SUCCEEDED
                    and step.operation is WorkflowOperation.COLLECT_HUNG_TRIAGE
                ):
                    # Keep the triage execution record and the DAG
                    # rewrite in memory until the single leased save
                    # below. That transaction is the serialization
                    # point across control-plane replicas.
                    workflow = self._rewrite_hung_triage_bundle(
                        workflow,
                        incident,
                        triage_index=index,
                        triage_details=outcome.details,
                    )
                    steps = (
                        workflow.safety_steps if is_safety else workflow.official_steps
                    )
                if outcome.status is WorkflowStepStatus.WAITING:
                    workflow = self._record_hold(workflow, step, index, outcome)
                    self._save_leased(workflow, execution_epoch)
                    continue
                if outcome.status is WorkflowStepStatus.FAILED:
                    escalation = self._escalate_failed_branch(
                        workflow, incident, index, outcome
                    )
                    if escalation is None:
                        # A job-level failure (or no escalator): the rest of
                        # the batch keeps running -- a sibling node's repair
                        # is still wanted -- and the first failure is the one
                        # reported when the batch ends.
                        batch_failure = batch_failure or (
                            index,
                            outcome,
                        )
                        self._save_leased(workflow, execution_epoch)
                        continue
                    # F-N1: only this node's branch failed. It was rewritten
                    # (next rung appended, or marked exhausted); the loop
                    # re-reads the DAG on its next pass.
                    workflow = escalation.workflow
                    steps = (
                        workflow.safety_steps if is_safety else workflow.official_steps
                    )
                    self._save_leased(workflow, execution_epoch)
                    continue
                rebindings = (outcome.details or {}).get("node_rebindings", {})
                incident_rebound = bool(rebindings)
                if rebindings:
                    workflow, incident = self._rebind_nodes(
                        workflow,
                        incident,
                        rebindings,
                        is_safety=is_safety,
                        after_index=index,
                    )
                    # The rebind rewrote later steps; keep reading the
                    # current list, not the one captured before it (F-C6).
                    steps = (
                        workflow.safety_steps if is_safety else workflow.official_steps
                    )
                workflow = workflow.model_copy(
                    update={
                        "completed_step_indexes": [
                            *workflow.completed_step_indexes,
                            index,
                        ],
                        "completed_operations": [
                            *workflow.completed_operations,
                            step.operation,
                        ],
                        "updated_at": datetime.now(timezone.utc),
                    }
                )
                if incident_rebound:
                    self._save_leased_state(workflow, incident, execution_epoch)
                else:
                    self._save_leased(workflow, execution_epoch)
            if batch_failure is not None:
                failure_index, failure_outcome = batch_failure
                if workflow.pending_failure_step_index is not None:
                    # A parked job-level failure owns the verdict; this batch's
                    # failure is one of the siblings it was waiting for.
                    failure_index = workflow.pending_failure_step_index
                    failure_outcome = WorkflowStepOutcome.failed(
                        workflow.pending_failure_error or "workflow DAG step failed"
                    )
                deferred = self._defer_job_failure(
                    workflow,
                    incident,
                    steps,
                    failure_index,
                    failure_outcome,
                    execution_epoch,
                )
                if deferred is not None:
                    return deferred
                if workflow.pending_failure_step_index is not None:
                    return self._land_pending_failure(
                        workflow,
                        incident,
                        request,
                        execution_epoch,
                        is_safety=is_safety,
                    )
                if (
                    steps[failure_index].operation
                    is not WorkflowOperation.RESTORE_GPU_SERVICES
                    and self._has_unrestored_quiesce(workflow, steps)
                ):
                    workflow = workflow.model_copy(
                        update={
                            "pending_failure_step_index": (failure_index),
                            "pending_failure_error": (
                                failure_outcome.error or "workflow DAG step failed"
                            ),
                            "updated_at": datetime.now(timezone.utc),
                        }
                    )
                    self._save_leased(workflow, execution_epoch)
                    return self._resume_failure_compensation(
                        workflow,
                        incident,
                        request,
                        execution_epoch,
                        is_safety=is_safety,
                    )
                return self._terminalize(
                    workflow,
                    incident,
                    WorkflowStatus.FAILED,
                    execution_epoch,
                    reason=failure_outcome.error,
                )

    def _defer_job_failure(
        self,
        workflow: WorkflowRequest,
        incident: FaultIncident,
        steps: list[WorkflowStepSpec],
        failure_index: int,
        failure_outcome: WorkflowStepOutcome,
        execution_epoch: int,
    ) -> WorkflowExecutionResult | None:
        """Wait out sibling commands still on the nodes before failing (D-3).

        A job-level DAG failure used to terminalize at the end of the batch
        while a sibling branch's remote command was LEASED on its node: nobody
        cancelled it, nobody collected it, and the hardware escalation that
        followed could plan a second action on that node. The preemption
        boundary already knows which waits can be stopped: cancellable
        commands (PENDING, or a WAITING collection) are cancelled here; if
        anything in flight remains, the untouched steps are retired, the
        failure is parked in ``pending_failure_*`` and the workflow stays
        RUNNING -- the loop keeps polling only the in-flight steps and lands
        the verdict once they settle. Returns the WAITING result in that case,
        ``None`` when the failure may land now.
        """

        boundary = preemption_boundary(workflow)
        for cancellation in boundary.cancellations:
            self.store.cancel_remote_command(
                cancellation.remote_command_id or "",
                reason=(
                    f"workflow step {failure_index} failed: "
                    f"{failure_outcome.error or 'workflow DAG step failed'}"
                ),
            )
        in_flight = (boundary.blocking | boundary.malformed) - {failure_index}
        if not in_flight:
            return None
        resolved = set(resolved_step_indexes(workflow))
        retired = {
            index
            for index, step in enumerate(steps)
            if index not in resolved
            and index not in in_flight
            # The compensation step stays runnable: ``_resume_failure_compensation``
            # needs it once the siblings settle.
            and step.operation is not WorkflowOperation.RESTORE_GPU_SERVICES
        }
        now = datetime.now(timezone.utc)
        parked = workflow.model_copy(
            update={
                "superseded_step_indexes": sorted(
                    set(workflow.superseded_step_indexes) | retired
                ),
                "pending_failure_step_index": (
                    workflow.pending_failure_step_index
                    if workflow.pending_failure_step_index is not None
                    else failure_index
                ),
                "pending_failure_error": (
                    workflow.pending_failure_error
                    or failure_outcome.error
                    or "workflow DAG step failed"
                ),
                "updated_at": now,
            }
        )
        if workflow.pending_failure_step_index is None:
            LOGGER.warning(
                "workflow %s step %s failed; the failure is parked until the "
                "in-flight sibling steps %s settle",
                workflow.request_id,
                failure_index,
                sorted(in_flight),
            )
        self._save_waiting(parked, execution_epoch)
        return self._result(parked, incident, waiting_step_index=min(in_flight))

    def _land_pending_failure(
        self,
        workflow: WorkflowRequest,
        incident: FaultIncident,
        request: WorkflowExecutionRequest,
        execution_epoch: int,
        *,
        is_safety: bool,
    ) -> WorkflowExecutionResult:
        """End a record whose ``pending_failure_*`` verdict may land now.

        The same fork the batch failure takes when nothing is in flight: a
        quiesce nobody restored gets its compensation first
        (``_resume_failure_compensation``); otherwise the parked error ends
        the workflow FAILED as it stands.
        """

        if self._has_unrestored_quiesce(workflow):
            return self._resume_failure_compensation(
                workflow,
                incident,
                request,
                execution_epoch,
                is_safety=is_safety,
            )
        return self._finalize_compensated_failure(
            workflow, incident, execution_epoch, compensation_error=None
        )

    def _job_failure_still_deferred(self, workflow: WorkflowRequest) -> bool:
        """Does a parked job-level failure still wait on in-flight siblings?

        Asked by the claim preflight before it resumes the compensation: while
        a DAG sibling's command is still executing the record goes back into
        the DAG loop, which polls it and lands the verdict when it settles.
        """

        if not workflow.dag_enabled or workflow.pending_failure_step_index is None:
            return False
        boundary = preemption_boundary(workflow)
        return bool(
            (boundary.blocking | boundary.malformed)
            - {workflow.pending_failure_step_index}
        )

    def _escalate_failed_branch(
        self,
        workflow: WorkflowRequest,
        incident: FaultIncident,
        index: int,
        outcome: WorkflowStepOutcome,
    ) -> BranchEscalation | None:
        if self.branch_escalator is None or not workflow.dag_enabled:
            return None
        if workflow.pending_failure_step_index is not None:
            # A job-level failure is already parked on this record (D-3): the
            # siblings are only being waited out, no rung is planned for them.
            return None
        if (outcome.details or {}).get(
            "remote_status_source"
        ) in WORKFLOW_CANCELLATION_STATUS_SOURCES:
            # The step failed because this workflow cancelled its own command
            # (a step boundary, the deadline): nothing is wrong with the node,
            # and a rung for it spends cluster budget on nothing (D-8).
            return None
        if (outcome.details or {}).get("workflow_lifetime_exceeded"):
            # The remediation's lifetime is over; no rung is planned for
            # anyone, the whole workflow fails to an operator (F-N1).
            return None
        escalation = self.branch_escalator.escalate_branch(
            workflow, index, outcome.error
        )
        if escalation is not None and escalation.outcome == "escalated":
            escalation = self._settle_escalation_budget(workflow, incident, escalation)
        if escalation is not None:
            LOGGER.warning(
                "workflow %s branch %s (%s): %s",
                workflow.request_id,
                escalation.branch_id,
                escalation.outcome,
                escalation.reason,
            )
        return escalation

    def _settle_escalation_budget(
        self,
        workflow: WorkflowRequest,
        incident: FaultIncident,
        escalation: BranchEscalation,
    ) -> BranchEscalation:
        """Take the budget the new rung needs, or retire the branch (F-N1).

        The workflow's concurrency budget was claimed for its original plan;
        a reboot or replacement appended later is a resource class that
        budget never counted. The extra scopes are taken here under the
        store's budget lock. If the cluster cannot take another one the
        branch is exhausted and goes to an operator -- fail closed rather
        than exceed the limit."""

        appended = escalation.workflow.official_steps[len(workflow.official_steps) :]
        claims = escalation_budget_claims(
            self.config.remediation_budget, workflow, incident, appended
        )
        if not claims:
            return escalation
        assert self.branch_escalator is not None
        try:
            extended = self.store.extend_remediation_budget(
                workflow.request_id, self.config.executor_id, claims
            )
        except RemediationBudgetError as exc:
            self.branch_escalation_budget_refusals_total += 1
            return self.branch_escalator.exhaust_branch(
                workflow,
                escalation.node_id,
                escalation.branch_id,
                reason=(
                    f"{escalation.reason}; the cluster remediation budget "
                    f"cannot take the next rung ({exc})"
                ),
            )
        return replace(
            escalation,
            workflow=escalation.workflow.model_copy(
                update={
                    "remediation_budget_claims": extended.remediation_budget_claims,
                    "remediation_budget_limits": extended.remediation_budget_limits,
                }
            ),
        )

    def _fail_exhausted_branches(
        self,
        workflow: WorkflowRequest,
        incident: FaultIncident,
        execution_epoch: int,
    ) -> WorkflowExecutionResult:
        reason = "node branch escalation exhausted: " + ", ".join(
            workflow.exhausted_branch_ids
        )
        # The reason is the record's, not only the audit event's: the field is
        # what the API, the acceptance verdicts and the failure handler read
        # (live 2026-09-09 the exhausted DESTR-014 workflow ended FAILED with
        # ``terminal_failure_reason`` None and the reason only in TERMINAL).
        return self._terminalize(
            workflow,
            incident,
            WorkflowStatus.FAILED,
            execution_epoch,
            reason=reason,
            updates={"terminal_failure_reason": reason},
        )

    # Steps that undo what an earlier step did to a node; they still run for a
    # withdrawn workflow so a stopped job does not leave its nodes cordoned.
    _WITHDRAWAL_RELEASE_OPERATIONS = frozenset(
        {
            WorkflowOperation.RESTORE_SCHEDULING,
            WorkflowOperation.RESTORE_GPU_SERVICES,
        }
    )
    _QUIESCE_SETTLING_OPERATIONS = frozenset(
        {
            WorkflowOperation.RESTART_NODE,
            WorkflowOperation.REPLACE_NODE,
            WorkflowOperation.RESTART_VM,
        }
    )
    # A release is only owed to a node whose containment took place.
    _WITHDRAWAL_RELEASE_COUNTERPARTS = {
        WorkflowOperation.RESTORE_SCHEDULING: WorkflowOperation.MARK_UNSCHEDULABLE,
        WorkflowOperation.RESTORE_GPU_SERVICES: WorkflowOperation.QUIESCE_GPU_SERVICES,
    }

    def _apply_workload_withdrawal(
        self,
        workflow: WorkflowRequest,
        steps: list[WorkflowStepSpec],
    ) -> WorkflowRequest:
        """Retire every step that has not started and is not a needed release (F-N1 §7).

        In-flight steps (WAITING records) finish; a release step runs once its
        dependencies resolve, but only for a node the workflow actually
        touched -- one whose cordon or quiesce completed or is in flight.
        Everything else -- untouched node repairs, releases for nodes never
        cordoned, the job restart -- is superseded.
        """

        resolved = set(resolved_step_indexes(workflow))
        waiting = {
            item.step_index
            for item in workflow.step_executions
            if item.status is WorkflowStepStatus.WAITING
        }
        touched = set(workflow.completed_step_indexes) | waiting

        def release_needed(step: WorkflowStepSpec) -> bool:
            counterpart = self._WITHDRAWAL_RELEASE_COUNTERPARTS.get(step.operation)
            return any(
                index in touched
                and other.operation is counterpart
                and set(other.node_ids) & set(step.node_ids)
                for index, other in enumerate(steps)
            )

        skip = {
            index
            for index, step in enumerate(steps)
            if index not in resolved
            and index not in waiting
            and not (
                step.operation in self._WITHDRAWAL_RELEASE_OPERATIONS
                and release_needed(step)
            )
        }
        if not skip:
            return workflow
        return workflow.model_copy(
            update={
                "superseded_step_indexes": sorted(
                    set(workflow.superseded_step_indexes) | skip
                ),
                "updated_at": datetime.now(timezone.utc),
            }
        )

    def _finish_withdrawn(
        self,
        workflow: WorkflowRequest,
        incident: FaultIncident,
        execution_epoch: int,
    ) -> WorkflowExecutionResult:
        result = self._terminalize(
            workflow,
            incident,
            WorkflowStatus.SUPERSEDED,
            execution_epoch,
            updates={
                "preemption_reason": (
                    "workload stopped externally: "
                    + (workflow.workload_withdrawn_reason or "no reason recorded")
                ),
                "superseded_at": datetime.now(timezone.utc),
            },
        )
        LOGGER.warning(
            "workflow %s wound down after its job was stopped externally: %s",
            workflow.request_id,
            workflow.workload_withdrawn_reason,
        )
        return result

    def _fail_with_reason(
        self,
        workflow: WorkflowRequest,
        incident: FaultIncident,
        execution_epoch: int,
    ) -> WorkflowExecutionResult:
        """End FAILED once the plan ran with a reason set ahead of time.

        The rule A stop rewrite (F-N1 §8) and the restart withheld for an
        exhausted budget (逻辑 4) both set ``terminal_failure_reason`` and let
        the remaining steps run. The incident then follows the failure rule:
        a node this workflow cordoned and never released stays QUARANTINED;
        otherwise the job needs an operator, ESCALATED. The stop rewrite keeps
        only STOP_WORKLOADS, so it lands on ESCALATED as before.
        """

        completed = set(workflow.completed_operations)
        still_isolated = (
            bool(
                {WorkflowOperation.MARK_UNSCHEDULABLE, WorkflowOperation.QUARANTINE}
                & completed
            )
            and WorkflowOperation.RESTORE_SCHEDULING not in completed
        )
        return self._terminalize(
            workflow,
            incident,
            WorkflowStatus.FAILED,
            execution_epoch,
            reason=workflow.terminal_failure_reason,
            incident_state=(
                IncidentState.QUARANTINED if still_isolated else IncidentState.ESCALATED
            ),
        )

    @staticmethod
    def _validate_dag(
        steps: list[WorkflowStepSpec],
    ) -> None:
        count = len(steps)
        if count > MAX_DAG_STEPS:
            raise WorkflowStructureError(
                f"workflow DAG has {count} steps, more than the "
                f"{MAX_DAG_STEPS} one workflow may carry"
            )
        dependencies = {
            index: set(step.depends_on_step_indexes) for index, step in enumerate(steps)
        }
        for index, values in dependencies.items():
            if index in values or any(value < 0 or value >= count for value in values):
                raise WorkflowStructureError(
                    f"invalid DAG dependencies for step {index}"
                )
        resolved: set[int] = set()
        while len(resolved) < count:
            ready = {
                index
                for index, values in dependencies.items()
                if index not in resolved and values <= resolved
            }
            if not ready:
                raise WorkflowStructureError(
                    "workflow step dependency graph contains a cycle"
                )
            resolved.update(ready)

    def _resume_failure_compensation(
        self,
        workflow: WorkflowRequest,
        incident: FaultIncident,
        request: WorkflowExecutionRequest,
        execution_epoch: int,
        *,
        is_safety: bool,
    ) -> WorkflowExecutionResult:
        steps = workflow.safety_steps if is_safety else workflow.official_steps
        restore_match = next(
            (
                (index, step)
                for index, step in enumerate(steps)
                if step.operation is WorkflowOperation.RESTORE_GPU_SERVICES
                and index not in resolved_step_indexes(workflow)
            ),
            None,
        )
        if restore_match is None:
            return self._finalize_compensated_failure(
                workflow,
                incident,
                execution_epoch,
                compensation_error=("no RESTORE_GPU_SERVICES compensation step"),
            )
        restore_index, restore_step = restore_match
        outcome = self._execute_step(
            workflow,
            incident,
            request,
            restore_step,
            restore_index,
        )
        workflow = self.store.renew_workflow_lease(
            workflow.request_id,
            self.config.executor_id,
            execution_epoch,
            lease_duration=self._lease_duration,
        )
        workflow = step_bounds.record_attempt(
            workflow, restore_step, restore_index, outcome
        )
        if outcome.status is WorkflowStepStatus.WAITING:
            self._save_waiting(workflow, execution_epoch)
            return self._result(
                workflow,
                incident,
                waiting_step_index=restore_index,
                error=workflow.pending_failure_error,
            )
        if outcome.status is WorkflowStepStatus.SUCCEEDED:
            workflow = workflow.model_copy(
                update={
                    "completed_step_indexes": sorted(
                        set(workflow.completed_step_indexes) | {restore_index}
                    ),
                    "completed_operations": list(
                        dict.fromkeys(
                            [
                                *workflow.completed_operations,
                                WorkflowOperation.RESTORE_GPU_SERVICES,
                            ]
                        )
                    ),
                    "updated_at": datetime.now(timezone.utc),
                }
            )
        return self._finalize_compensated_failure(
            workflow,
            incident,
            execution_epoch,
            compensation_error=(
                outcome.error if outcome.status is WorkflowStepStatus.FAILED else None
            ),
        )

    def _finalize_compensated_failure(
        self,
        workflow: WorkflowRequest,
        incident: FaultIncident,
        execution_epoch: int,
        *,
        compensation_error: str | None,
    ) -> WorkflowExecutionResult:
        original_error = workflow.pending_failure_error or "workflow step failed"
        error = (
            original_error
            if compensation_error is None
            else f"{original_error}; restore compensation failed: {compensation_error}"
        )
        return self._terminalize(
            workflow,
            incident,
            WorkflowStatus.FAILED,
            execution_epoch,
            reason=error,
            updates={
                "pending_failure_step_index": None,
                "pending_failure_error": None,
            },
        )

    def _persist_step_completion_notification(
        self,
        workflow: WorkflowRequest,
        incident: FaultIncident,
        step: WorkflowStepSpec,
        step_index: int,
        outcome: WorkflowStepOutcome,
    ) -> WorkflowStepOutcome:
        if (
            outcome.status is not WorkflowStepStatus.SUCCEEDED
            or step.operation is not WorkflowOperation.REPLACE_NODE
            or (outcome.details or {}).get("action") != "SPARE_FAILOVER"
        ):
            return outcome
        operation_id = f"{workflow.request_id}/{step_index}/REPLACE_NODE"
        spare_nodes = list(outcome.details.get("activated_spare_nodes") or [])
        rebindings = {
            str(key): str(value)
            for key, value in (outcome.details.get("node_rebindings") or {}).items()
        }
        notification = self.warm_spare_email_builder.build(
            cluster_id=incident.cluster_id,
            incident_id=incident.incident_id,
            workflow_id=workflow.request_id,
            event_id=incident.event_id,
            policy_source=incident.policy_source,
            official_action=incident.official_action,
            effective_action=(
                incident.effective_action.value if incident.effective_action else None
            ),
            reasons=incident.reasons,
            operation_id=operation_id,
            fault_node_ids=step.node_ids,
            spare_node_ids=spare_nodes,
            node_rebindings=rebindings,
            confirmation_source=str(
                outcome.details.get(
                    "confirmation_source",
                    "healthy-running-warm-spare",
                )
            ),
            provider_mutation_submitted=bool(
                outcome.details.get("provider_mutation_submitted", False)
            ),
        )
        notification = self.store.save_notification_if_absent(notification)
        if self.notification_sender is not None:
            self.notification_sender(notification.notification_id)
        return replace(
            outcome,
            details={
                **outcome.details,
                "notification_id": (
                    outcome.details.get("notification_id")
                    or notification.notification_id
                ),
            },
        )

    def _complete_claimed_workflow(
        self,
        workflow: WorkflowRequest,
        incident: FaultIncident,
        *,
        is_safety: bool,
        execution_epoch: int,
    ) -> WorkflowExecutionResult:
        if workflow.terminal_failure_reason:
            return self._fail_with_reason(workflow, incident, execution_epoch)
        final_status = WorkflowStatus.BLOCKED if is_safety else WorkflowStatus.SUCCEEDED
        return self._terminalize(
            workflow,
            incident,
            final_status,
            execution_epoch,
            updates={
                "blocked_kind": (
                    BlockedKind.SAFETY_SETTLED
                    if final_status is WorkflowStatus.BLOCKED
                    else None
                ),
            },
        )

    @staticmethod
    def _has_unrestored_quiesce(
        workflow: WorkflowRequest,
        steps: list[WorkflowStepSpec] | None = None,
    ) -> bool:
        """Is there a completed quiesce whose nodes no later completed restore
        covers? Judged per node (F-C4): a restore on another node -- or a
        restore that ran *before* the quiesce -- does not undo it. The step
        set defaults to the one this record executes."""

        if steps is None:
            steps = (
                workflow.safety_steps
                if workflow.executes_safety_steps
                else workflow.official_steps
            )
        completed = set(workflow.completed_step_indexes)
        for quiesce_index, quiesce in enumerate(steps):
            if (
                quiesce_index not in completed
                or quiesce.operation is not WorkflowOperation.QUIESCE_GPU_SERVICES
            ):
                continue
            restored: set[str] = set()
            for restore_index, restore in enumerate(steps):
                if (
                    restore_index > quiesce_index
                    and restore_index in completed
                    and (
                        restore.operation is WorkflowOperation.RESTORE_GPU_SERVICES
                        # A rebooted or replaced node has no quiesced services
                        # left to restore (F-N1: the rung that replaced the
                        # branch's RESTORE_GPU_SERVICES settles it).
                        or restore.operation
                        in ProductionWorkflowExecutor._QUIESCE_SETTLING_OPERATIONS
                    )
                ):
                    restored.update(restore.node_ids)
            if set(quiesce.node_ids) - restored:
                return True
        return False

    def _supersede_if_safe(
        self,
        workflow: WorkflowRequest,
        incident: FaultIncident,
        request: WorkflowExecutionRequest,
        next_step: WorkflowStepSpec,
        next_index: int,
        execution_epoch: int,
    ) -> WorkflowExecutionResult | None:
        if not self.config.workflow_preemption_enabled:
            return None
        successor = self.store.get_preempting_successor(workflow.request_id)
        if successor is None:
            return None
        # The boundary is the whole workflow, not the one step about to run:
        # every WAITING record -- this step's and any other branch's -- must be
        # something we can stop, or the successor would abandon a command that
        # is executing on the node with nobody left to collect it (F-C1). The
        # merge asks the same function at planning time; judged whole before
        # anything is cancelled, so a closed boundary cancels nothing.
        boundary = preemption_boundary(workflow)
        if not boundary.open:
            LOGGER.info(
                "workflow not superseded at step boundary: workflow=%s "
                "next_step=%s successor=%s reason=%s",
                workflow.request_id,
                next_index,
                successor.request_id,
                boundary.reason,
            )
            return None
        # Judged live before anything is cancelled (D-8): the record says what
        # the command was on the last poll, and a command the data plane leased
        # since cannot be cancelled -- cancelling its siblings first and then
        # finding that out left a FAILED sibling the next tick escalated.
        stale = self._uncancellable_live(boundary)
        if stale is not None:
            LOGGER.info(
                "workflow not superseded at step boundary: workflow=%s "
                "next_step=%s successor=%s reason=%s",
                workflow.request_id,
                next_index,
                successor.request_id,
                stale,
            )
            return None
        for cancellation in boundary.cancellations:
            if not self.store.cancel_remote_command(
                cancellation.remote_command_id or "",
                reason=(
                    "remote command cancelled by stronger "
                    f"workflow {successor.request_id}"
                ),
            ):
                return None
        if self._has_unrestored_quiesce(workflow):
            if self._can_handoff_quiesce_for_preemption(workflow, incident, successor):
                pass
            else:
                compensation = self._restore_for_preemption(
                    workflow,
                    incident,
                    request,
                    execution_epoch,
                    successor,
                )
                if isinstance(compensation, WorkflowExecutionResult):
                    return compensation
                if compensation is None:
                    return None
                workflow = compensation
        now = datetime.now(timezone.utc)
        superseded = workflow.model_copy(
            update={
                "status": WorkflowStatus.SUPERSEDED,
                "preempted_by_workflow_id": successor.request_id,
                "preemption_reason": successor.preemption_reason,
                "superseded_at": now,
                "execution_owner_id": None,
                "execution_lease_expires_at": None,
                "updated_at": now,
            }
        )
        superseded = record_workflow_event(
            superseded,
            WorkflowEventKind.TERMINAL,
            code=WorkflowEventCode.TERMINALIZED.value,
            reason=successor.preemption_reason,
            actor=self.config.executor_id,
            status=WorkflowStatus.SUPERSEDED.value,
            step_index=next_index,
            operation=next_step.operation,
            details={
                "execution_epoch": execution_epoch,
                "preempted_by_workflow_id": successor.request_id,
            },
        )
        # The incident already names the successor, so nothing is written for
        # it here (``incident=None``); the workflow itself ends through the
        # same terminal funnel as every other ending (F-C9 release of the
        # restart reservations, ARCH-B3 terminal hooks). The step about to
        # run is released too: it never left the gate.
        self._save_terminal(
            superseded,
            None,
            execution_epoch,
            release_step_indexes={next_index},
        )
        LOGGER.info(
            "workflow safely superseded at step boundary: "
            "workflow=%s next_step=%s/%s successor=%s",
            workflow.request_id,
            next_index,
            next_step.operation.value,
            successor.request_id,
        )
        return self._result(superseded, incident)

    def _uncancellable_live(self, boundary: PreemptionBoundary) -> str | None:
        """The first cancellable record whose command can no longer be
        cancelled *now*, as a reason, or ``None`` when every one still can."""

        for record in boundary.cancellations:
            try:
                command = self.store.get_remote_command(record.remote_command_id or "")
            except NotFoundError:
                # Nothing is running for it; the cancel below says so itself.
                continue
            if not self._cancellable_now(record, command):
                return (
                    f"step {record.step_index} {record.operation.value} remote "
                    f"command is {command.status.value} now and cannot be cancelled"
                )
        return None

    @staticmethod
    def _cancellable_now(record: WaitingRecord, command: Any) -> bool:
        """The store's own cancel rule (PENDING, or a WAITING collection),
        applied to the command as it is rather than as it was recorded."""

        from gpu_fault.remote_command_models import RemoteCommandStatus

        if command.status is RemoteCommandStatus.PENDING:
            return True
        return (
            command.status is RemoteCommandStatus.WAITING
            and record.operation in SAFE_REMOTE_WAITING_PREEMPT_OPERATIONS
        )

    def _quiesce_handoff_evidence(
        self,
        workflow: WorkflowRequest,
    ) -> tuple[int, WorkflowStepSpec, WorkflowStepExecution] | None:
        completed = set(workflow.completed_step_indexes)
        quiesce_match = next(
            (
                (index, step)
                for index, step in reversed(list(enumerate(workflow.official_steps)))
                if index in completed
                and step.operation is WorkflowOperation.QUIESCE_GPU_SERVICES
            ),
            None,
        )
        if quiesce_match is None:
            return None
        quiesce_index, quiesce_step = quiesce_match
        quiesce_execution = next(
            (
                item
                for item in workflow.step_executions
                if item.step_index == quiesce_index
                and item.operation is quiesce_step.operation
                and item.status is WorkflowStepStatus.SUCCEEDED
            ),
            None,
        )
        if quiesce_execution is None:
            return None
        return (
            quiesce_index,
            quiesce_step,
            quiesce_execution,
        )

    def _can_handoff_quiesce_for_preemption(
        self,
        workflow: WorkflowRequest,
        incident: FaultIncident,
        successor: WorkflowRequest,
    ) -> bool:
        if successor.incident_id != incident.incident_id:
            return False
        evidence = self._quiesce_handoff_evidence(workflow)
        if evidence is None:
            return False
        _, quiesce_step, _ = evidence
        successor_operations = {step.operation for step in successor.official_steps}
        if successor_operations.intersection(
            {
                WorkflowOperation.RESET_GPU,
                WorkflowOperation.RESET_ALL_GPUS_NVSWITCHES,
            }
        ):
            successor_quiesce = next(
                (
                    (index, step)
                    for index, step in enumerate(successor.official_steps)
                    if step.operation is WorkflowOperation.QUIESCE_GPU_SERVICES
                    and set(step.node_ids) <= set(quiesce_step.node_ids)
                ),
                None,
            )
            return successor_quiesce is not None
        if WorkflowOperation.RESTART_NODE not in successor_operations:
            return False
        restart_index = next(
            index
            for index, step in enumerate(successor.official_steps)
            if step.operation is WorkflowOperation.RESTART_NODE
        )
        restart_step = successor.official_steps[restart_index]
        if not set(restart_step.node_ids) <= set(quiesce_step.node_ids):
            return False
        restore_step = next(
            (
                step
                for step in workflow.official_steps
                if step.operation is WorkflowOperation.RESTORE_GPU_SERVICES
            ),
            None,
        )
        if restore_step is None:
            return False
        return True

    def _adopt_quiesce_handoff_from_predecessor(
        self,
        workflow: WorkflowRequest,
        incident: FaultIncident,
    ) -> WorkflowRequest:
        if (
            workflow.quiesce_handoff_from_workflow_id is not None
            or workflow.predecessor_workflow_id is None
        ):
            return workflow
        try:
            predecessor = self.store.get_workflow(workflow.predecessor_workflow_id)
        except NotFoundError:
            return workflow
        if (
            predecessor.status is not WorkflowStatus.SUPERSEDED
            or predecessor.preempted_by_workflow_id != workflow.request_id
            or predecessor.incident_id != incident.incident_id
        ):
            return workflow
        evidence = self._quiesce_handoff_evidence(predecessor)
        if evidence is None:
            return workflow
        _, quiesce_step, quiesce_execution = evidence
        operations = {step.operation for step in workflow.official_steps}
        if operations.intersection(
            {
                WorkflowOperation.RESET_GPU,
                WorkflowOperation.RESET_ALL_GPUS_NVSWITCHES,
            }
        ):
            match = next(
                (
                    (index, step)
                    for index, step in enumerate(workflow.official_steps)
                    if step.operation is WorkflowOperation.QUIESCE_GPU_SERVICES
                    and set(step.node_ids) <= set(quiesce_step.node_ids)
                ),
                None,
            )
            if match is None:
                return workflow
            quiesce_index, _ = match
            executions = [
                item
                for item in workflow.step_executions
                if item.step_index != quiesce_index
                or item.operation is not WorkflowOperation.QUIESCE_GPU_SERVICES
            ]
            executions.append(
                WorkflowStepExecution(
                    step_index=quiesce_index,
                    operation=(WorkflowOperation.QUIESCE_GPU_SERVICES),
                    status=WorkflowStepStatus.SUCCEEDED,
                    phase="official",
                    adapter_operation_id=(quiesce_execution.adapter_operation_id),
                    details={
                        **quiesce_execution.details,
                        "inherited_from_workflow_id": (predecessor.request_id),
                        "preemption_quiesce_handoff": True,
                    },
                )
            )
            return workflow.model_copy(
                update={
                    "completed_step_indexes": sorted(
                        set(workflow.completed_step_indexes) | {quiesce_index}
                    ),
                    "completed_operations": list(
                        dict.fromkeys(
                            [
                                *workflow.completed_operations,
                                WorkflowOperation.QUIESCE_GPU_SERVICES,
                            ]
                        )
                    ),
                    "step_executions": sorted(
                        executions,
                        key=lambda item: item.step_index,
                    ),
                    "inherited_step_indexes": sorted(
                        set(workflow.inherited_step_indexes) | {quiesce_index}
                    ),
                    "quiesce_handoff_from_workflow_id": (predecessor.request_id),
                    "updated_at": datetime.now(timezone.utc),
                }
            )
        if WorkflowOperation.RESTART_NODE not in operations:
            return workflow
        restart_index = next(
            index
            for index, step in enumerate(workflow.official_steps)
            if step.operation is WorkflowOperation.RESTART_NODE
        )
        restart_step = workflow.official_steps[restart_index]
        if not set(restart_step.node_ids) <= set(quiesce_step.node_ids):
            return workflow
        restore_step = next(
            (
                step
                for step in predecessor.official_steps
                if step.operation is WorkflowOperation.RESTORE_GPU_SERVICES
            ),
            None,
        )
        if restore_step is None:
            return workflow
        cleanup_step = restore_step.model_copy(
            update={
                "node_ids": list(restart_step.node_ids),
                "branch_id": restart_step.branch_id,
                "parameters": {
                    **restore_step.parameters,
                    "preemption_quiesce_handoff_after_reboot": True,
                    "handoff_from_workflow_id": (predecessor.request_id),
                },
            }
        )
        steps = list(workflow.official_steps)
        if not workflow.dag_enabled:
            steps = [
                step.model_copy(
                    update={"depends_on_step_indexes": ([index - 1] if index else [])}
                )
                for index, step in enumerate(steps)
            ]
        cleanup_index = len(steps)
        cleanup_step = cleanup_step.model_copy(
            update={"depends_on_step_indexes": [restart_index]}
        )
        for index, step in enumerate(steps):
            if index == restart_index:
                continue
            dependencies = list(step.depends_on_step_indexes)
            if restart_index not in dependencies:
                continue
            steps[index] = step.model_copy(
                update={
                    "depends_on_step_indexes": [
                        cleanup_index if value == restart_index else value
                        for value in dependencies
                    ]
                }
            )
        steps.append(cleanup_step)
        return workflow.model_copy(
            update={
                "official_steps": steps,
                "dag_enabled": True,
                "dag_revision": workflow.dag_revision + 1,
                "quiesce_handoff_from_workflow_id": (predecessor.request_id),
                "updated_at": datetime.now(timezone.utc),
            }
        )

    def _restore_for_preemption(
        self,
        workflow: WorkflowRequest,
        incident: FaultIncident,
        request: WorkflowExecutionRequest,
        execution_epoch: int,
        successor: WorkflowRequest,
    ) -> WorkflowRequest | WorkflowExecutionResult | None:
        restore_match = next(
            (
                (index, step)
                for index, step in enumerate(workflow.official_steps)
                if step.operation is WorkflowOperation.RESTORE_GPU_SERVICES
                and index not in resolved_step_indexes(workflow)
            ),
            None,
        )
        if restore_match is None:
            return None
        restore_index, restore_step = restore_match
        outcome = self._execute_step(
            workflow,
            incident,
            request,
            restore_step,
            restore_index,
        )
        workflow = self.store.renew_workflow_lease(
            workflow.request_id,
            self.config.executor_id,
            execution_epoch,
            lease_duration=self._lease_duration,
        )
        workflow = step_bounds.record_attempt(
            workflow, restore_step, restore_index, outcome
        )
        if outcome.status is WorkflowStepStatus.WAITING:
            self._save_waiting(workflow, execution_epoch)
            return self._result(
                workflow,
                incident,
                waiting_step_index=restore_index,
            )
        if outcome.status is WorkflowStepStatus.FAILED:
            # The incident already names the successor, so its state is the
            # successor's to set; this record only reports its own end.
            return self._terminalize(
                workflow,
                incident,
                WorkflowStatus.FAILED,
                execution_epoch,
                reason=outcome.error,
                incident_state=incident.state,
                updates={
                    "preempted_by_workflow_id": successor.request_id,
                    "preemption_reason": (
                        "preemption compensation failed: "
                        f"{outcome.error or 'restore failed'}"
                    ),
                },
            )
        workflow = workflow.model_copy(
            update={
                "completed_step_indexes": [
                    *workflow.completed_step_indexes,
                    restore_index,
                ],
                "completed_operations": [
                    *workflow.completed_operations,
                    WorkflowOperation.RESTORE_GPU_SERVICES,
                ],
                "updated_at": datetime.now(timezone.utc),
            }
        )
        self._save_leased(workflow, execution_epoch)
        return workflow

    def _rewrite_hung_triage_bundle(
        self,
        workflow: WorkflowRequest,
        incident: FaultIncident,
        *,
        triage_index: int,
        triage_details: dict[str, Any],
    ) -> WorkflowRequest:
        bundle_index = next(
            (
                index
                for index, step in enumerate(workflow.official_steps)
                if step.operation is WorkflowOperation.COLLECT_DIAGNOSTIC_BUNDLE
                and triage_index in step.depends_on_step_indexes
            ),
            None,
        )
        if bundle_index is None:
            return workflow
        rank_context = self._hung_rank_context(incident)
        signals = []
        for node_id, details in (triage_details.get("node_results") or {}).items():
            for raw in details.get("ranks") or []:
                if not isinstance(raw, dict):
                    continue
                signal = {**raw, "node_id": node_id}
                pid = signal.get("pid")
                context = rank_context.get(
                    (node_id, int(pid)) if isinstance(pid, int) else None,
                    {},
                )
                for key in ("rank", "gpu_uuid"):
                    if signal.get(key) is None and context.get(key) is not None:
                        signal[key] = context[key]
                signals.append(signal)
        undetermined_nodes = sorted(set(triage_details.get("undetermined_nodes") or []))
        observed_ranks = {
            signal.get("rank")
            for signal in signals
            if isinstance(signal.get("rank"), int)
        }
        for (node_id, pid), context in rank_context.items():
            rank = context.get("rank")
            if (
                node_id not in undetermined_nodes
                or not isinstance(rank, int)
                or rank in observed_ranks
            ):
                continue
            signals.append(
                {
                    "node_id": node_id,
                    "pid": pid,
                    "rank": rank,
                    "gpu_uuid": context.get("gpu_uuid"),
                    "flight_recorder": {"status": "node_unavailable"},
                    "python_stack": {},
                    "proc": {},
                    "gpu": {},
                }
            )
        decision = self._classify_hung_signals(
            signals,
            undetermined_nodes=undetermined_nodes,
            efa_zero_pending_at_by_node=(
                self._hung_efa_zero_pending_times(
                    incident,
                    {
                        str(signal.get("node_id"))
                        for signal in signals
                        if signal.get("node_id")
                    },
                )
            ),
        )
        not_sampled_nodes = sorted(
            {
                str(node_id)
                for node_id in (triage_details.get("not_sampled_nodes") or [])
                if node_id
            }
        )
        if not_sampled_nodes:
            # An inconclusive verdict over a partial sample means "not
            # localised here", not "the whole fabric is suspect", and an
            # operator reading the decision needs to see which nodes were
            # never attached to.
            decision = {
                **decision,
                "not_sampled_nodes": not_sampled_nodes,
            }
        selected_ranks = set(decision.get("culprit_ranks") or [])
        control_ranks = set(decision.get("control_ranks") or [])
        selected = [
            signal
            for signal in signals
            if signal.get("rank") in selected_ranks | control_ranks
        ]
        steps = list(workflow.official_steps)
        bundle = steps[bundle_index]
        actionable = (
            decision["classification"] in {"CONFIRMED", "PLAUSIBLE", "WEAK"}
            and 0 < len(selected_ranks) <= 3
        )
        parameters = {
            **bundle.parameters,
            "hung_triage_decision": decision,
            "hung_triage_target_pending": False,
        }
        if actionable:
            target_pids_by_node: dict[str, list[int]] = {}
            target_gpu_uuids_by_pid_by_node: dict[str, dict[str, str]] = {}
            gpu_uuids = []
            for signal in selected:
                pid = signal.get("pid")
                node_id = signal.get("node_id")
                if isinstance(pid, int) and isinstance(node_id, str):
                    target_pids_by_node.setdefault(node_id, []).append(pid)
                    if signal.get("gpu_uuid"):
                        target_gpu_uuids_by_pid_by_node.setdefault(node_id, {})[
                            str(pid)
                        ] = str(signal["gpu_uuid"])
                if signal.get("gpu_uuid"):
                    gpu_uuids.append(str(signal["gpu_uuid"]))
            node_ids = sorted(target_pids_by_node)
            parameters.update(
                {
                    "capture_process_state": True,
                    "target_pids_by_node": {
                        node_id: sorted(set(pids))
                        for node_id, pids in target_pids_by_node.items()
                    },
                    "target_gpu_uuids_by_pid_by_node": (
                        target_gpu_uuids_by_pid_by_node
                    ),
                    "max_processes": len(selected),
                    "expand_python_cgroup_processes": False,
                    "strace_sample_count": (
                        1 if decision["classification"] == "WEAK" else 3
                    ),
                }
            )
            steps[bundle_index] = bundle.model_copy(
                update={
                    "node_ids": node_ids,
                    "gpu_uuids": sorted(set(gpu_uuids)),
                    "parameters": parameters,
                }
            )
        else:
            parameters.update(
                {
                    "capture_process_state": False,
                    "target_pids_by_node": {},
                    "target_gpu_uuids_by_pid_by_node": {},
                    "max_processes": 0,
                    "expand_python_cgroup_processes": False,
                    "strace_sample_count": 0,
                    "operator_escalation_required": True,
                }
            )
            steps[bundle_index] = bundle.model_copy(update={"parameters": parameters})
        executions = [
            execution.model_copy(
                update={
                    "details": {
                        **execution.details,
                        "hung_triage_decision": decision,
                    }
                }
            )
            if execution.step_index == triage_index
            and execution.operation is WorkflowOperation.COLLECT_HUNG_TRIAGE
            else execution
            for execution in workflow.step_executions
        ]
        return workflow.model_copy(
            update={
                "official_steps": steps,
                "step_executions": executions,
                "dag_revision": workflow.dag_revision + 1,
                "updated_at": datetime.now(timezone.utc),
            }
        )

    def _hung_rank_context(
        self, incident: FaultIncident
    ) -> dict[tuple[str, int], dict[str, Any]]:
        if not incident.attempt_id:
            return {}
        states = self.store.list_attempt_observation_states(incident.cluster_id)
        observation = next(
            (
                state.observation
                for state in reversed(states)
                if state.observation.attempt_id == incident.attempt_id
            ),
            None,
        )
        if observation is None:
            return {}
        return {
            (container.node_id, container.host_pid): {
                "rank": container.rank,
                "gpu_uuid": (
                    container.gpu_uuids[0] if len(container.gpu_uuids) == 1 else None
                ),
            }
            for container in observation.containers
            if container.node_id and container.host_pid is not None
        }

    def _hung_efa_zero_pending_times(
        self,
        incident: FaultIncident,
        node_ids: set[str],
    ) -> dict[str, datetime]:
        if not incident.job_id or not incident.attempt_id:
            return {}
        result = {}
        for node_id in node_ids:
            key = self.store.efa_traffic_state_key(
                incident.cluster_id,
                node_id,
                incident.job_id,
                incident.attempt_id,
            )
            try:
                state = self.store.get_efa_traffic_state(key)
            except NotFoundError:
                continue
            if state.zero_since is not None:
                result[node_id] = state.zero_since
        return result

    @staticmethod
    def _classify_hung_signals(
        signals: list[dict[str, Any]],
        *,
        undetermined_nodes: list[str],
        efa_zero_pending_at_by_node: dict[str, datetime] | None = None,
    ) -> dict[str, Any]:
        return classify_hung_signals(
            signals,
            undetermined_nodes=undetermined_nodes,
            efa_zero_pending_at_by_node=efa_zero_pending_at_by_node,
        )

    def _rebind_nodes(
        self,
        workflow: WorkflowRequest,
        incident: FaultIncident,
        rebindings: dict[str, str],
        *,
        is_safety: bool,
        after_index: int,
    ) -> tuple[WorkflowRequest, FaultIncident]:
        return node_rebinding.rebind_nodes(
            self.store,
            workflow,
            incident,
            rebindings,
            is_safety=is_safety,
            after_index=after_index,
        )

    @property
    def _lease_duration(self) -> timedelta:
        return timedelta(seconds=self.config.lease_duration_seconds)

    @property
    def _restart_waiting_ttl(self) -> timedelta:
        """How long a RESTART_WORKLOAD may wait before its reservation is
        treated as never used (F-C9): the step's own waiting cap."""

        return timedelta(
            seconds=self.config.step_waiting_limit(WorkflowOperation.RESTART_WORKLOAD)
        )

    def _check_invariants(self, workflow: WorkflowRequest) -> None:
        """Every write the executor owns passes here first (review item 6)."""

        check_workflow_invariants(workflow, self.config.workflow_invariant_mode)

    def _save_leased(
        self,
        workflow: WorkflowRequest,
        execution_epoch: int,
    ) -> None:
        self._check_invariants(workflow)
        self.store.save_workflow_if_leased(
            workflow,
            self.config.executor_id,
            execution_epoch,
        )

    def _save_waiting(
        self,
        workflow: WorkflowRequest,
        execution_epoch: int,
    ) -> None:
        """The leased save of a row this tick is done with (D-7).

        The row keeps its owner and epoch -- the next tick's claim by the same
        executor is a no-op re-lease -- but its lease is cut to
        ``waiting_lease_duration`` so another process can take it soon after a
        dispatch-lease handover, instead of after the full lease.
        """

        shorter = self.waiting_lease_duration
        if shorter is not None and workflow.execution_lease_expires_at is not None:
            expires = datetime.now(timezone.utc) + shorter
            if expires < workflow.execution_lease_expires_at:
                workflow = workflow.model_copy(
                    update={"execution_lease_expires_at": expires}
                )
        self._save_leased(workflow, execution_epoch)

    def _save_leased_state(
        self,
        workflow: WorkflowRequest,
        incident: FaultIncident,
        execution_epoch: int,
    ) -> None:
        self._check_invariants(workflow)
        current_incident = self.store.get_incident(incident.incident_id)
        if (
            current_incident.workflow_request_id is not None
            and current_incident.workflow_request_id != workflow.request_id
        ):
            self.store.save_workflow_if_leased(
                workflow,
                self.config.executor_id,
                execution_epoch,
            )
            return
        self.store.save_workflow_and_incident_if_leased(
            workflow,
            incident,
            self.config.executor_id,
            execution_epoch,
        )

    def _record_hold(
        self,
        workflow: WorkflowRequest,
        step: WorkflowStepSpec,
        index: int,
        outcome: WorkflowStepOutcome,
    ) -> WorkflowRequest:
        """Turn an adapter's "node under remediation" WAITING into a HOLD event.

        Only the shared hold code is recorded here; a step waiting on its own
        remote command is a STEP_ATTEMPT, not a hold.
        """

        details = outcome.details or {}
        if details.get("reason") != HOLD_REASON_NODE_UNDER_REMEDIATION:
            return workflow
        remediation = details.get("remediation_workflow_id")
        held = record_hold_event(
            workflow,
            reason=HOLD_REASON_NODE_UNDER_REMEDIATION,
            remediation_workflow_id=(
                str(remediation) if remediation is not None else None
            ),
            actor=self.config.executor_id,
            step_index=index,
            operation=step.operation,
            details={
                key: value
                for key, value in details.items()
                if key in {"premise_reason", "incident_id", "incident_state"}
            },
        )
        return workflow if held is None else held

    def terminalize_claimed(
        self,
        workflow: WorkflowRequest,
        incident: FaultIncident | None,
        status: WorkflowStatus,
        execution_epoch: int,
        *,
        reason: str | None,
        actor: str,
        incident_state: IncidentState | None = None,
        updates: Mapping[str, object] | None = None,
    ) -> WorkflowExecutionResult:
        """End a workflow this executor's id already holds the lease on.

        The dispatcher's deadline watchdog and its internal-error BLOCK claim
        the record as ``config.executor_id`` and used to write their own
        terminal copies -- the reap never touched the incident, so it stayed
        ACTION_PENDING behind a FAILED workflow. They now come through the
        same funnel as the executor's own endings; ``actor`` names them on
        the TERMINAL event. ``incident`` may be ``None`` when the incident row
        is gone: the workflow still ends, and nothing is written for it.
        """

        return self._terminalize(
            workflow,
            incident,
            status,
            execution_epoch,
            reason=reason,
            incident_state=incident_state,
            updates=updates,
            actor=actor,
        )

    def _terminalize(
        self,
        workflow: WorkflowRequest,
        incident: FaultIncident | None,
        status: WorkflowStatus,
        execution_epoch: int,
        *,
        reason: str | None = None,
        incident_state: IncidentState | None = None,
        updates: Mapping[str, object] | None = None,
        actor: str | None = None,
    ) -> WorkflowExecutionResult:
        """The one way a claimed workflow ends (F-C9).

        Every terminal path used to write its own copy of the same four
        fields and its own incident state, and each drift between them was a
        bug in the release of restart reservations. This writes the terminal
        ``status``, drops the owner and lease, applies the path's own
        ``updates`` (a blocked kind, a preemption reason, cleared compensation
        markers), derives the incident state from the status unless the path
        names one, appends the one ``TERMINAL`` audit event, and saves
        through ``_save_terminal`` -- which is where the unattempted restart
        reservations are released. ``reason`` is the error the result
        carries; ``actor`` defaults to this executor.
        """

        now = datetime.now(timezone.utc)
        ended = workflow.model_copy(
            update={
                **(dict(updates) if updates else {}),
                "status": status,
                "execution_owner_id": None,
                "execution_lease_expires_at": None,
                "updated_at": now,
            }
        )
        derived_state = (
            incident_state
            if incident_state is not None
            else self._terminal_incident_state(ended, status)
        )
        # A diagnostic-only workflow that failed did not conclude anything
        # about the node; the derivation closes the incident RECOVERED and the
        # verdict is spelled out on the incident, the event and the markers.
        inconclusive = (
            status is WorkflowStatus.FAILED
            and incident_state is None
            and self._diagnostic_only(ended)
        )
        if incident is not None:
            update: dict[str, Any] = {"state": derived_state, "updated_at": now}
            if inconclusive:
                update["reasons"] = bounded_reasons(
                    [*incident.reasons, self._inconclusive_reason(reason)]
                )
            incident = incident.model_copy(update=update)
        details: dict[str, Any] = {
            "execution_epoch": execution_epoch,
            "incident_state": derived_state.value,
        }
        if inconclusive:
            details["diagnostic_inconclusive"] = True
        ended = record_workflow_event(
            ended,
            WorkflowEventKind.TERMINAL,
            code=WorkflowEventCode.TERMINALIZED.value,
            reason=self._inconclusive_reason(reason) if inconclusive else reason,
            actor=actor or self.config.executor_id,
            status=status.value,
            details=details,
            at=now,
        )
        incident_written = self._save_terminal(ended, incident, execution_epoch)
        if inconclusive and incident is not None and incident_written:
            self._close_inconclusive_diagnostic(ended, incident, reason)
        return self._result(ended, incident, error=reason)

    @staticmethod
    def _inconclusive_reason(error: str | None) -> str:
        return (
            f"{DIAGNOSTIC_INCONCLUSIVE_REASON}: {error}"
            if error
            else DIAGNOSTIC_INCONCLUSIVE_REASON
        )

    @staticmethod
    def _diagnostic_only(workflow: WorkflowRequest) -> bool:
        """Whether every step this workflow planned or ran only observed the node.

        Judged on the plan as well as on what ran: a ``VALIDATE_HOST`` that
        gates a later reboot is a gate, not a diagnostic, and its failure must
        keep escalating. The classes come from the operation registry --
        destructive (including containment) and node-wide operations -- so a
        new operation cannot be mutating there and diagnostic here.
        """

        steps = (
            workflow.safety_steps
            if workflow.executes_safety_steps
            else workflow.official_steps
        )
        operations = (
            {step.operation for step in steps}
            | set(workflow.completed_operations)
            | {item.operation for item in workflow.step_executions}
        )
        if not operations:
            return False
        return not (
            operations & (DESTRUCTIVE_OPERATIONS | NODE_WIDE_RECOVERY_OPERATIONS)
        )

    def _close_inconclusive_diagnostic(
        self,
        workflow: WorkflowRequest,
        incident: FaultIncident,
        error: str | None,
    ) -> None:
        """Retire the incident's markers and file one advisory notification.

        Runs after the terminal write landed and only when this workflow was
        still the incident's plan. Each half is isolated like an ``on_terminal``
        hook: a marker or notification failure is logged and must not unwind
        the terminal state that already landed.
        """

        try:
            retire_markers_for_incident(
                self.store,
                incident.incident_id,
                reason=self._inconclusive_reason(error),
                retired_by=self.config.executor_id,
            )
        except Exception:  # noqa: BLE001 - the terminal write already landed
            LOGGER.exception(
                "retiring markers of incident %s after inconclusive workflow %s",
                incident.incident_id,
                workflow.request_id,
            )
        steps = (
            workflow.safety_steps
            if workflow.executes_safety_steps
            else workflow.official_steps
        )
        failed = [
            item
            for item in workflow.step_executions
            if item.status is WorkflowStepStatus.FAILED
        ]
        try:
            notification = self.diagnostic_inconclusive_email_builder.build(
                cluster_id=incident.cluster_id,
                incident_id=incident.incident_id,
                workflow_id=workflow.request_id,
                event_id=incident.event_id,
                node_ids=sorted(
                    {node for step in steps for node in step.node_ids}
                    or set(incident.node_ids)
                ),
                operations=[step.operation.value for step in steps],
                failed_operation=failed[-1].operation.value if failed else None,
                error=error,
                policy_source=incident.policy_source,
                official_action=incident.official_action,
                reasons=incident.reasons,
            )
            notification = self.store.save_notification_if_absent(notification)
            if self.notification_sender is not None:
                self.notification_sender(notification.notification_id)
        except Exception:  # noqa: BLE001 - the terminal write already landed
            LOGGER.exception(
                "notifying inconclusive diagnostic for incident %s workflow %s",
                incident.incident_id,
                workflow.request_id,
            )

    @classmethod
    def _terminal_incident_state(
        cls,
        workflow: WorkflowRequest,
        status: WorkflowStatus,
    ) -> IncidentState:
        """What the incident becomes when its workflow ends in ``status``.

        FAILED keeps the containment verdict (``_failure_incident_state``),
        except that a diagnostic-only workflow leaves it RECOVERED;
        BLOCKED is a settled safety phase, so the node stays quarantined; a
        finished or withdrawn workflow that isolated a node and never released
        it leaves the incident QUARANTINED, a completed one that escalated to
        support leaves it ESCALATED, and anything else is RECOVERED.
        """

        if status is WorkflowStatus.FAILED:
            return cls._failure_incident_state(workflow)
        if status is WorkflowStatus.BLOCKED:
            return IncidentState.QUARANTINED
        completed = workflow.completed_operations
        if (
            status is WorkflowStatus.SUCCEEDED
            and WorkflowOperation.ESCALATE_SUPPORT in completed
        ):
            return IncidentState.ESCALATED
        isolated = (
            WorkflowOperation.QUARANTINE in completed
            and WorkflowOperation.RESTORE_SCHEDULING not in completed
        )
        return IncidentState.QUARANTINED if isolated else IncidentState.RECOVERED

    def _save_terminal(
        self,
        workflow: WorkflowRequest,
        incident: FaultIncident | None,
        execution_epoch: int,
        *,
        release_step_indexes: set[int] | None = None,
    ) -> bool:
        """The one write every terminal transition lands through.

        ``_terminalize`` (and so the dispatcher's ``terminalize_claimed``) and
        the step-boundary preemption in ``_supersede_if_safe`` both end here:
        the unattempted restart reservations are released (F-C9), the record
        -- and the incident, when this workflow is still its plan -- is saved,
        and the ``on_terminal`` hooks run once the write has landed (ARCH-B3).
        Returns whether the incident was written alongside the workflow.
        """

        self._check_invariants(workflow)
        restart_preflight.release_unattempted_restart_reservations(
            self.store,
            workflow,
            release_step_indexes=release_step_indexes,
            waiting_ttl=self._restart_waiting_ttl,
        )
        current_incident = (
            None if incident is None else self.store.get_incident(incident.incident_id)
        )
        if incident is None or (
            current_incident is not None
            and current_incident.workflow_request_id is not None
            and current_incident.workflow_request_id != workflow.request_id
        ):
            # No incident row to end, or a stronger successor became the
            # incident's current plan while this predecessor was executing.
            # Persist only the predecessor terminal state; a stale incident
            # snapshot must not steal the pointer or mark the successor's
            # incident recovered/failed.
            self.store.save_workflow_if_leased(
                workflow,
                self.config.executor_id,
                execution_epoch,
            )
            self._notify_terminal(workflow, None)
            return False
        self.store.save_workflow_and_incident_if_leased(
            workflow,
            incident,
            self.config.executor_id,
            execution_epoch,
        )
        self._notify_terminal(workflow, incident)
        return True

    def _notify_terminal(
        self,
        workflow: WorkflowRequest,
        incident: FaultIncident | None,
    ) -> None:
        """Run every ``on_terminal`` hook, each isolated from the others.

        A hook releases state the workflow held outside the store; a failure
        there is logged and must neither stop the next hook nor unwind the
        terminal write that already landed (ARCH-B3).
        """

        if not self.on_terminal:
            return
        steps = (
            workflow.safety_steps
            if workflow.executes_safety_steps
            else workflow.official_steps
        )
        for hook in list(self.on_terminal):
            try:
                hook(workflow, incident, list(steps))
            except Exception:  # noqa: BLE001 - one hook must not stop the rest
                LOGGER.exception(
                    "on_terminal hook failed: workflow=%s status=%s hook=%s",
                    workflow.request_id,
                    workflow.status.value,
                    getattr(hook, "__qualname__", repr(hook)),
                )

    def _execute_step(
        self,
        workflow: WorkflowRequest,
        incident: FaultIncident,
        request: WorkflowExecutionRequest,
        step: WorkflowStepSpec,
        index: int,
    ) -> WorkflowStepOutcome:
        """Dispatch one step, bounded by the workflow and per-step deadlines.

        Both bounds are applied here rather than in the two loops above because
        this is the single funnel every step outcome passes through, including
        the compensation and preemption-restore steps. Why the lease holder is
        the one enforcing the workflow deadline is in ``step_bounds``.
        """

        expired = step_bounds.workflow_deadline_failure(self, workflow, step, index)
        if expired is not None:
            return expired
        outcome = self._dispatch_step(workflow, incident, request, step, index)
        return step_bounds.bounded_waiting_outcome(self, workflow, step, index, outcome)

    def _dispatch_step(
        self,
        workflow: WorkflowRequest,
        incident: FaultIncident,
        request: WorkflowExecutionRequest,
        step: WorkflowStepSpec,
        index: int,
    ) -> WorkflowStepOutcome:
        if step.operation not in self.config.allowed_operations:
            return WorkflowStepOutcome.failed(
                f"operation {step.operation.value} is not in "
                "GPU_FAULT_ALLOWED_OPERATIONS"
            )
        matches = [adapter for adapter in self.adapters if adapter.supports(step)]
        if len(matches) != 1:
            return WorkflowStepOutcome.failed(
                f"expected exactly one adapter for "
                f"{step.execution_owner}/{step.operation.value}; "
                f"found {len(matches)}"
            )
        idempotency_key = f"{workflow.request_id}/{index}/{step.operation.value}"
        if step.operation is WorkflowOperation.RESTART_WORKLOAD:
            # The adapter compares its authorization's reservation_id with this
            # key, so it has to be the id the preflight reserved under, phase
            # included, or a safety-phase restart would be refused (F-C9).
            idempotency_key = restart_preflight.reservation_id(
                workflow,
                index,
                phase=restart_preflight.reservation_phase(workflow),
            )
            if request.restart_authorization is None:
                # The preflight already reserved under that id; dispatch signs
                # the reservation for the adapter rather than reserving again,
                # and fails closed when it is gone.
                issued = restart_preflight.issue_restart_authorization(
                    self.store, incident, step, idempotency_key
                )
                if isinstance(issued, WorkflowStepOutcome):
                    return issued
                request = request.model_copy(update={"restart_authorization": issued})
        context = WorkflowStepContext(
            workflow=workflow,
            incident=incident,
            step=step,
            step_index=index,
            request=request,
            idempotency_key=idempotency_key,
        )
        adapter = matches[0]
        try:
            return adapter.execute(context)
        except Exception as exc:
            if transient_store_error(exc):
                # A store hiccup inside the adapter says nothing about the
                # step; the dispatcher retries the workflow on a later tick
                # instead of this becoming a FAILED step and a hardware
                # escalation (F-C7 / F-J1).
                LOGGER.warning(
                    "workflow step hit a transient store error and will be "
                    "retried: workflow=%s step=%s/%s error=%s: %s",
                    workflow.request_id,
                    index,
                    step.operation.value,
                    type(exc).__name__,
                    exc,
                )
                raise
            retryable = retryable_adapter_error(exc)
            attempt = self._adapter_error_attempt(workflow, step, index)
            if retryable and attempt <= self.step_transient_retry_limit:
                # Symmetric with the store case (ARCH-B1): a Kubernetes 5xx or
                # a torn connection says nothing about the node. The step
                # waits for the next tick instead of failing into isolation;
                # the per-step waiting cap (``step_bounds``) still bounds the
                # wall-clock, this counter bounds the attempts.
                LOGGER.warning(
                    "workflow step hit a retryable adapter error and will be "
                    "retried: workflow=%s step=%s/%s adapter=%s attempt=%s/%s "
                    "error=%s: %s",
                    workflow.request_id,
                    index,
                    step.operation.value,
                    type(adapter).__name__,
                    attempt,
                    self.step_transient_retry_limit,
                    type(exc).__name__,
                    exc,
                )
                return self._retry_adapter_error_outcome(
                    workflow, step, index, exc, attempt
                )
            LOGGER.exception(
                "workflow step raised: workflow=%s step=%s/%s "
                "adapter=%s incident=%s nodes=%s",
                workflow.request_id,
                index,
                step.operation.value,
                type(adapter).__name__,
                incident.incident_id,
                ",".join(step.node_ids),
            )
            details = _failure_details(adapter, exc)
            if retryable:
                # The bound is exhausted: the last error fails the step on the
                # established path, and the record says how many it absorbed.
                details.update(
                    {
                        "retryable_adapter_error": True,
                        "error_class": type(exc).__name__,
                        "attempt": attempt,
                        "retry_limit": self.step_transient_retry_limit,
                        "reason": STEP_RETRY_REASON_ADAPTER_ERROR,
                    }
                )
            return WorkflowStepOutcome.failed(
                f"{type(exc).__name__}: {exc}",
                details=details,
            )

    @staticmethod
    def _adapter_error_attempt(
        workflow: WorkflowRequest,
        step: WorkflowStepSpec,
        index: int,
    ) -> int:
        """Which consecutive retryable-error attempt this one is (ARCH-B1).

        Read off the step's own last record: the count continues only while
        that record is a retryable-error wait, so a run of errors after a
        legitimate adapter wait starts again at one.
        """

        previous = step_bounds.previous_execution(workflow, step, index)
        if (
            previous is None
            or previous.status is not WorkflowStepStatus.WAITING
            or not previous.details.get("retryable_adapter_error")
        ):
            return 1
        prior = previous.details.get("attempt")
        return prior + 1 if isinstance(prior, int) and prior > 0 else 1

    @staticmethod
    def _retry_adapter_error_outcome(
        workflow: WorkflowRequest,
        step: WorkflowStepSpec,
        index: int,
        exc: BaseException,
        attempt: int,
    ) -> WorkflowStepOutcome:
        """The WAITING outcome a retryable adapter error turns into (ARCH-B1).

        ``record_attempt`` replaces the step's record with this outcome, so the
        pointers a previous wait carried -- a remote command id and status, a
        node-action command -- are kept: the adapter resumes polling them on
        the next tick and the preemption boundary still sees the in-flight
        command. A ``reason`` a previous hold recorded is kept for the same
        reason; otherwise the record says why it waits.
        """

        previous = step_bounds.previous_execution(workflow, step, index)
        inherited: dict[str, Any] = {}
        operation_id: str | None = None
        if previous is not None and previous.status is WorkflowStepStatus.WAITING:
            inherited = dict(previous.details)
            operation_id = previous.adapter_operation_id
            # A hold's "nothing submitted yet" does not survive a retry: this
            # error may have been raised after the adapter created the Job,
            # and the reservation release must not read the stale marker.
            inherited.pop("restart_submitted", None)
        details: dict[str, Any] = {
            **inherited,
            "retryable_adapter_error": True,
            "error_class": type(exc).__name__,
            "attempt": attempt,
            "adapter_error": f"{type(exc).__name__}: {exc}"[:300],
        }
        details.setdefault("reason", STEP_RETRY_REASON_ADAPTER_ERROR)
        return WorkflowStepOutcome(
            status=WorkflowStepStatus.WAITING,
            adapter_operation_id=operation_id,
            error=f"{type(exc).__name__}: {exc}",
            details=details,
        )

    def _validate_fencing(
        self,
        workflow: WorkflowRequest,
        incident: FaultIncident,
        request: WorkflowExecutionRequest,
    ) -> None:
        expected = request.expected_fencing_token
        current = incident.workflow_request_id in {None, workflow.request_id}
        # No longer a pure comparison: an incident that names someone else and has
        # moved on has retired this workflow, and only a Store read distinguishes
        # that from a preemption somebody else is already resolving.
        retired = retirement_fences_out_dispatch(self.store, workflow, incident)
        if (
            expected != workflow.fencing_token
            or (current and expected != incident.fencing_token)
            or retired
        ):
            raise WorkflowFencingError(
                "stale fencing token: workflow="
                f"{workflow.fencing_token}, incident="
                f"{incident.fencing_token}, got={expected}"
                + (f", retired by {incident.workflow_request_id}" if retired else "")
            )

    @classmethod
    def _failure_incident_state(
        cls,
        workflow: WorkflowRequest,
    ) -> IncidentState:
        """The incident's state behind a FAILED workflow.

        A completed isolation keeps the node QUARANTINED. A workflow that
        planned nothing but evidence, diagnostics and validation -- the
        node-health ``RUN_DIAGNOSTICS`` shape -- did not change the node, so
        its failure is a diagnostic that did not conclude: the incident ends
        RECOVERED (``_terminalize`` records the reason, retires the markers
        and notifies), instead of ESCALATED with no follow-up step while its
        marker keeps the node "under remediation" until the TTL. Everything
        else is ESCALATED.
        """

        isolated = {
            WorkflowOperation.MARK_UNSCHEDULABLE,
            WorkflowOperation.QUARANTINE,
        }.intersection(workflow.completed_operations)
        if isolated:
            return IncidentState.QUARANTINED
        if cls._diagnostic_only(workflow):
            return IncidentState.RECOVERED
        return IncidentState.ESCALATED

    @staticmethod
    def _result(
        workflow: WorkflowRequest,
        incident: FaultIncident | None,
        *,
        waiting_step_index: int | None = None,
        error: str | None = None,
    ) -> WorkflowExecutionResult:
        return WorkflowExecutionResult(
            operation_id=f"workflow-op-{uuid4()}",
            workflow_request_id=workflow.request_id,
            incident_id=(
                workflow.incident_id if incident is None else incident.incident_id
            ),
            status=workflow.status,
            completed_operations=workflow.completed_operations,
            simulation_only=False,
            waiting_step_index=waiting_step_index,
            error=error,
        )

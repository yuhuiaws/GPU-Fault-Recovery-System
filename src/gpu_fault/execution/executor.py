from __future__ import annotations

import logging
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from threading import RLock
from typing import Any
from uuid import uuid4

from gpu_fault.models import (
    FaultIncident,
    IncidentState,
    WorkflowExecutionRequest,
    WorkflowExecutionResult,
    WorkflowOperation,
    WorkflowRequest,
    WorkflowStatus,
    WorkflowStepExecution,
    WorkflowStepSpec,
    WorkflowStepStatus,
)
from gpu_fault.operation_registry import (
    SAFE_REMOTE_WAITING_PREEMPT_OPERATIONS,
    SAFE_WAITING_PREEMPT_OPERATIONS,
)
from gpu_fault.notifications import (
    WarmSpareReplacementEmailBuilder,
)
from gpu_fault.orchestrator import WorkflowFencingError
from gpu_fault.store import NotFoundError
from gpu_fault.store.contracts import ControlPlaneStore
from gpu_fault.workflow_resolution import retirement_fences_out_dispatch
from gpu_fault.execution.config import (
    ProductionExecutorConfig,
)
from gpu_fault.execution.fleet_preflight import (
    held_workflow_result,
)
from gpu_fault.execution.hung_classification import (
    classify_hung_signals,
)
from gpu_fault.execution.models import (
    WorkflowExecutionError,
    WorkflowStepAdapter,
    WorkflowStepContext,
    WorkflowStepOutcome,
    _failure_details,
)
import gpu_fault.execution.restart_budget_preflight as restart_preflight
import gpu_fault.execution.step_bounds as step_bounds

LOGGER = logging.getLogger(__name__)


class ProductionWorkflowExecutor:
    """Execute owned workflow steps with durable, fail-closed gates."""

    def __init__(
        self,
        store: ControlPlaneStore,
        adapters: list[WorkflowStepAdapter],
        config: ProductionExecutorConfig,
        *,
        notification_sender=None,
    ) -> None:
        if config.workflow_execution_timeout_seconds <= 0:
            raise WorkflowExecutionError("workflow execution timeout must be positive")
        self.store = store
        self.adapters = adapters
        self.config = config
        self.notification_sender = notification_sender
        self.warm_spare_email_builder = WarmSpareReplacementEmailBuilder()
        self._lock = RLock()

    def execute(
        self,
        request_id: str,
        request: WorkflowExecutionRequest,
    ) -> WorkflowExecutionResult:
        with self._lock:
            if not self.config.enabled:
                raise WorkflowExecutionError("production workflow executor is disabled")
            workflow = self.store.get_workflow(request_id)
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

            is_safety = bool(workflow.blocked_reasons)
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
            workflow = prepared.workflow
            execution_epoch = prepared.execution_epoch
            steps = workflow.safety_steps if is_safety else workflow.official_steps
            if workflow.dag_enabled:
                return self._execute_dag(
                    workflow,
                    incident,
                    request,
                    is_safety=is_safety,
                    execution_epoch=execution_epoch,
                )

            for index in range(len(steps)):
                step = steps[index]
                if index in workflow.completed_step_indexes:
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
                    self._save_leased(workflow, execution_epoch)
                    return self._result(
                        workflow,
                        incident,
                        waiting_step_index=index,
                    )
                if outcome.status is WorkflowStepStatus.FAILED:
                    if (
                        step.operation is not WorkflowOperation.RESTORE_GPU_SERVICES
                        and self._has_unrestored_quiesce(workflow)
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
                    workflow = workflow.model_copy(
                        update={
                            "status": WorkflowStatus.FAILED,
                            "execution_owner_id": None,
                            "execution_lease_expires_at": None,
                            "updated_at": datetime.now(timezone.utc),
                        }
                    )
                    incident = incident.model_copy(
                        update={
                            "state": self._failure_incident_state(workflow),
                            "updated_at": datetime.now(timezone.utc),
                        }
                    )
                    self._save_terminal(workflow, incident, execution_epoch)
                    return self._result(
                        workflow,
                        incident,
                        error=outcome.error,
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

            final_status = (
                WorkflowStatus.BLOCKED if is_safety else WorkflowStatus.SUCCEEDED
            )
            final_incident_state = (
                IncidentState.QUARANTINED
                if is_safety
                else IncidentState.ESCALATED
                if WorkflowOperation.ESCALATE_SUPPORT in workflow.completed_operations
                else IncidentState.QUARANTINED
                if (
                    WorkflowOperation.QUARANTINE in workflow.completed_operations
                    and WorkflowOperation.RESTORE_SCHEDULING
                    not in workflow.completed_operations
                )
                else IncidentState.RECOVERED
            )
            now = datetime.now(timezone.utc)
            workflow = workflow.model_copy(
                update={
                    "status": final_status,
                    "execution_owner_id": None,
                    "execution_lease_expires_at": None,
                    "updated_at": now,
                }
            )
            incident = incident.model_copy(
                update={
                    "state": final_incident_state,
                    "updated_at": now,
                }
            )
            self._save_terminal(workflow, incident, execution_epoch)
            return self._result(workflow, incident)

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
        while True:
            workflow = self.store.renew_workflow_lease(
                workflow.request_id,
                self.config.executor_id,
                execution_epoch,
                lease_duration=self._lease_duration,
            )
            steps = workflow.safety_steps if is_safety else workflow.official_steps
            self._validate_dag(steps)
            completed = set(workflow.completed_step_indexes)
            resolved = completed | set(workflow.superseded_step_indexes)
            if len(resolved) == len(steps):
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
            if not ready:
                self._save_leased(workflow, execution_epoch)
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
                    self._save_leased(workflow, execution_epoch)
                    continue
                if outcome.status is WorkflowStepStatus.FAILED:
                    batch_failure = batch_failure or (
                        index,
                        outcome,
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
                if (
                    steps[failure_index].operation
                    is not WorkflowOperation.RESTORE_GPU_SERVICES
                    and self._has_unrestored_quiesce(workflow)
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
                        "state": self._failure_incident_state(workflow),
                        "updated_at": now,
                    }
                )
                self._save_terminal(workflow, incident, execution_epoch)
                return self._result(
                    workflow,
                    incident,
                    error=failure_outcome.error,
                )

    @staticmethod
    def _validate_dag(
        steps: list[WorkflowStepSpec],
    ) -> None:
        count = len(steps)
        dependencies = {
            index: set(step.depends_on_step_indexes) for index, step in enumerate(steps)
        }
        for index, values in dependencies.items():
            if index in values or any(value < 0 or value >= count for value in values):
                raise WorkflowExecutionError(
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
                raise WorkflowExecutionError(
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
                and index not in workflow.completed_step_indexes
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
            self._save_leased(workflow, execution_epoch)
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
        now = datetime.now(timezone.utc)
        failed = workflow.model_copy(
            update={
                "status": WorkflowStatus.FAILED,
                "pending_failure_step_index": None,
                "pending_failure_error": None,
                "execution_owner_id": None,
                "execution_lease_expires_at": None,
                "updated_at": now,
            }
        )
        incident = incident.model_copy(
            update={
                "state": self._failure_incident_state(failed),
                "updated_at": now,
            }
        )
        self._save_terminal(failed, incident, execution_epoch)
        return self._result(failed, incident, error=error)

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
        final_status = WorkflowStatus.BLOCKED if is_safety else WorkflowStatus.SUCCEEDED
        final_incident_state = (
            IncidentState.QUARANTINED
            if is_safety
            else IncidentState.ESCALATED
            if WorkflowOperation.ESCALATE_SUPPORT in workflow.completed_operations
            else IncidentState.QUARANTINED
            if (
                WorkflowOperation.QUARANTINE in workflow.completed_operations
                and WorkflowOperation.RESTORE_SCHEDULING
                not in workflow.completed_operations
            )
            else IncidentState.RECOVERED
        )
        now = datetime.now(timezone.utc)
        workflow = workflow.model_copy(
            update={
                "status": final_status,
                "execution_owner_id": None,
                "execution_lease_expires_at": None,
                "updated_at": now,
            }
        )
        incident = incident.model_copy(
            update={
                "state": final_incident_state,
                "updated_at": now,
            }
        )
        self._save_terminal(workflow, incident, execution_epoch)
        return self._result(workflow, incident)

    _SAFE_WAITING_PREEMPT_OPERATIONS = SAFE_WAITING_PREEMPT_OPERATIONS
    _SAFE_REMOTE_WAITING_PREEMPT_OPERATIONS = SAFE_REMOTE_WAITING_PREEMPT_OPERATIONS

    @staticmethod
    def _has_unrestored_quiesce(
        workflow: WorkflowRequest,
    ) -> bool:
        completed = set(workflow.completed_step_indexes)
        quiesce_indexes = [
            index
            for index, step in enumerate(workflow.official_steps)
            if index in completed
            and step.operation is WorkflowOperation.QUIESCE_GPU_SERVICES
        ]
        if not quiesce_indexes:
            return False
        restore_indexes = [
            index
            for index, step in enumerate(workflow.official_steps)
            if index in completed
            and step.operation is WorkflowOperation.RESTORE_GPU_SERVICES
        ]
        return max(quiesce_indexes) > max(restore_indexes, default=-1)

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
        waiting = next(
            (
                item
                for item in workflow.step_executions
                if item.step_index == next_index
                and item.status is WorkflowStepStatus.WAITING
            ),
            None,
        )
        if waiting is not None:
            operation_id = waiting.adapter_operation_id or ""
            if operation_id.startswith("remote/"):
                remote_status = str(waiting.details.get("remote_status") or "")
                can_cancel = remote_status == "PENDING" or (
                    remote_status == "WAITING"
                    and next_step.operation
                    in self._SAFE_REMOTE_WAITING_PREEMPT_OPERATIONS
                )
                command_id = waiting.details.get(
                    "remote_command_id"
                ) or operation_id.removeprefix("remote/")
                if (
                    not can_cancel
                    or not isinstance(command_id, str)
                    or not self.store.cancel_remote_command(
                        command_id,
                        reason=(
                            "remote command cancelled by stronger "
                            f"workflow {successor.request_id}"
                        ),
                    )
                ):
                    return None
            elif next_step.operation not in self._SAFE_WAITING_PREEMPT_OPERATIONS:
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
        restart_preflight.release_unattempted_restart_reservations(
            self.store,
            superseded,
            release_step_indexes={next_index},
        )
        self._save_leased(superseded, execution_epoch)
        LOGGER.info(
            "workflow safely superseded at step boundary: "
            "workflow=%s next_step=%s/%s successor=%s",
            workflow.request_id,
            next_index,
            next_step.operation.value,
            successor.request_id,
        )
        return self._result(superseded, incident)

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
            ]
            executions.append(
                WorkflowStepExecution(
                    step_index=quiesce_index,
                    operation=(WorkflowOperation.QUIESCE_GPU_SERVICES),
                    status=WorkflowStepStatus.SUCCEEDED,
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
                and index not in workflow.completed_step_indexes
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
            self._save_leased(workflow, execution_epoch)
            return self._result(
                workflow,
                incident,
                waiting_step_index=restore_index,
            )
        if outcome.status is WorkflowStepStatus.FAILED:
            now = datetime.now(timezone.utc)
            failed = workflow.model_copy(
                update={
                    "status": WorkflowStatus.FAILED,
                    "preempted_by_workflow_id": successor.request_id,
                    "preemption_reason": (
                        "preemption compensation failed: "
                        f"{outcome.error or 'restore failed'}"
                    ),
                    "execution_owner_id": None,
                    "execution_lease_expires_at": None,
                    "updated_at": now,
                }
            )
            self._save_terminal(failed, incident, execution_epoch)
            return self._result(failed, incident, error=outcome.error)
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

    @staticmethod
    def _rebind_nodes(
        workflow: WorkflowRequest,
        incident: FaultIncident,
        rebindings: dict[str, str],
        *,
        is_safety: bool,
        after_index: int,
    ) -> tuple[WorkflowRequest, FaultIncident]:
        def replace(node_ids: list[str]) -> list[str]:
            return list(
                dict.fromkeys(rebindings.get(node_id, node_id) for node_id in node_ids)
            )

        field = "safety_steps" if is_safety else "official_steps"
        steps = list(getattr(workflow, field))
        for index in range(after_index + 1, len(steps)):
            steps[index] = steps[index].model_copy(
                update={"node_ids": replace(steps[index].node_ids)}
            )
        now = datetime.now(timezone.utc)
        return (
            workflow.model_copy(update={field: steps, "updated_at": now}),
            incident.model_copy(
                update={
                    "node_ids": replace(incident.node_ids),
                    "updated_at": now,
                }
            ),
        )

    @property
    def _lease_duration(self) -> timedelta:
        return timedelta(seconds=self.config.lease_duration_seconds)

    def _save_leased(
        self,
        workflow: WorkflowRequest,
        execution_epoch: int,
    ) -> None:
        self.store.save_workflow_if_leased(
            workflow,
            self.config.executor_id,
            execution_epoch,
        )

    def _save_leased_state(
        self,
        workflow: WorkflowRequest,
        incident: FaultIncident,
        execution_epoch: int,
    ) -> None:
        current_incident = self.store.get_incident(incident.incident_id)
        if (
            current_incident.workflow_request_id is not None
            and current_incident.workflow_request_id != workflow.request_id
        ):
            self._save_leased(workflow, execution_epoch)
            return
        self.store.save_workflow_and_incident_if_leased(
            workflow,
            incident,
            self.config.executor_id,
            execution_epoch,
        )

    def _save_terminal(
        self,
        workflow: WorkflowRequest,
        incident: FaultIncident,
        execution_epoch: int,
    ) -> None:
        restart_preflight.release_unattempted_restart_reservations(self.store, workflow)
        current_incident = self.store.get_incident(incident.incident_id)
        if (
            current_incident.workflow_request_id is not None
            and current_incident.workflow_request_id != workflow.request_id
        ):
            # A stronger successor became the incident's current plan
            # while this predecessor was executing. Persist only the
            # predecessor terminal state; its stale incident snapshot
            # must not steal the pointer or mark the successor's
            # incident recovered/failed.
            self._save_leased(workflow, execution_epoch)
            return
        self.store.save_workflow_and_incident_if_leased(
            workflow,
            incident,
            self.config.executor_id,
            execution_epoch,
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
        context = WorkflowStepContext(
            workflow=workflow,
            incident=incident,
            step=step,
            step_index=index,
            request=request,
            idempotency_key=(f"{workflow.request_id}/{index}/{step.operation.value}"),
        )
        adapter = matches[0]
        try:
            return adapter.execute(context)
        except Exception as exc:
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
            return WorkflowStepOutcome.failed(
                f"{type(exc).__name__}: {exc}",
                details=_failure_details(adapter, exc),
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

    @staticmethod
    def _failure_incident_state(
        workflow: WorkflowRequest,
    ) -> IncidentState:
        isolated = {
            WorkflowOperation.MARK_UNSCHEDULABLE,
            WorkflowOperation.QUARANTINE,
        }.intersection(workflow.completed_operations)
        return IncidentState.QUARANTINED if isolated else IncidentState.ESCALATED

    @staticmethod
    def _result(
        workflow: WorkflowRequest,
        incident: FaultIncident,
        *,
        waiting_step_index: int | None = None,
        error: str | None = None,
    ) -> WorkflowExecutionResult:
        return WorkflowExecutionResult(
            operation_id=f"workflow-op-{uuid4()}",
            workflow_request_id=workflow.request_id,
            incident_id=incident.incident_id,
            status=workflow.status,
            completed_operations=workflow.completed_operations,
            simulation_only=False,
            waiting_step_index=waiting_step_index,
            error=error,
        )

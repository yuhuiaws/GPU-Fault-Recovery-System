from __future__ import annotations

import hashlib
from datetime import datetime, timezone

from gpu_fault.models import (
    FaultIncident,
    IncidentState,
    RecoveryAction,
    RecoveryPlan,
    TerminalEvent,
    WorkflowOperation,
    WorkflowRequest,
    WorkflowStatus,
    WorkflowStepSpec,
    recovery_action_sort_key,
)
from gpu_fault.recovery_actions import RECOVERY_ACTION_PROFILES
from gpu_fault.store import NotFoundError

#: Derived from the one recovery-action table shared with the spare-blocking
#: set in :mod:`gpu_fault.markers`; see :mod:`gpu_fault.recovery_actions`.
ACTION_OPERATION: dict[RecoveryAction, WorkflowOperation] = {
    action: profile.operation
    for action, profile in RECOVERY_ACTION_PROFILES.items()
    if profile.operation is not None
}


class PassiveCompileError(ValueError):
    """A recovery plan names an action the passive compiler cannot execute."""


def operation_for_action(action: RecoveryAction) -> WorkflowOperation:
    """The workflow operation for a plan action, or a named error.

    ``ACTION_OPERATION[action]`` raised ``KeyError`` for the four actions the
    planner can emit that were missing from the table, and it raised it inside
    the window between the event row and the decision row being written -- the
    permanent poisoning of P0-62A / P0-62B. The table is now complete for
    every actionable ``RecoveryAction`` (pinned by a test), and an unmapped
    action is reported by name instead of as a bare ``KeyError``.
    """

    try:
        return ACTION_OPERATION[action]
    except KeyError:
        raise PassiveCompileError(
            f"recovery action {action.value} has no workflow operation"
        ) from None


class PassiveWorkflowCompiler:
    """Converts a passive RecoveryPlan into the fenced workflow engine."""

    def __init__(
        self,
        store,
        *,
        evidence_owner: str = "gpu-fault-control-plane",
    ) -> None:
        self.store = store
        self.evidence_owner = evidence_owner

    def compile(
        self,
        plan: RecoveryPlan,
        event: TerminalEvent,
        *,
        predecessor_workflow_id: str | None = None,
    ) -> RecoveryPlan:
        """Compile ``plan`` into an incident and a PENDING workflow.

        ``predecessor_workflow_id`` names the attempt's passive containment
        workflow when the completion service knows one: the dispatcher holds
        the recovery while that predecessor is open, which replaced the HTTP
        conflict the terminal used to be answered with.
        """

        if plan.workflow_request_id:
            return plan
        now = datetime.now(timezone.utc)
        incident_id = plan.incident_id
        try:
            existing_incident = self.store.get_incident(incident_id)
        except NotFoundError:
            existing_incident = None
        else:
            incident_id = (
                f"{incident_id}-completion-{event.attempt_id}-"
                + hashlib.sha256(plan.plan_id.encode()).hexdigest()[:12]
            )
            try:
                derived = self.store.get_incident(incident_id)
            except NotFoundError:
                pass
            else:
                if not derived.workflow_request_id:
                    raise RuntimeError("derived incident has no workflow pointer")
                workflow = self.store.get_workflow(derived.workflow_request_id)
                if workflow.source_plan_id != plan.plan_id:
                    raise RuntimeError("derived incident ID belongs to another plan")
                return plan.model_copy(
                    update={
                        "incident_id": derived.incident_id,
                        "workflow_request_id": workflow.request_id,
                    }
                )
        primary_action = (
            max(
                plan.steps,
                key=lambda step: recovery_action_sort_key(step.action),
            ).action
            if plan.steps
            else None
        )
        incident = FaultIncident(
            incident_id=incident_id,
            event_id=(
                event.event_key
                if existing_incident is None
                else f"{event.event_key}/{plan.plan_id}"
            ),
            event_type="TRAINING_ATTEMPT_TERMINAL",
            cluster_id=event.cluster_id,
            # ``_withdraw_job_workflows`` finds the workflows of a stopped job
            # through the incident's ``job_id``; without it a pending passive
            # restart survived the user's stop and ran anyway (F-N1 §7).
            job_id=event.job_id,
            attempt_id=event.attempt_id,
            node_ids=sorted({item.node_id for item in event.allocation}),
            gpu_uuids=sorted(
                {gpu for item in event.allocation for gpu in item.gpu_uuids}
            ),
            policy_version="passive-recovery-v1",
            policy_source="completion-handler",
            official_action=(primary_action.value if primary_action else None),
            effective_action=primary_action,
            state=IncidentState.ACTION_PENDING,
            drill_id=plan.drill_id,
            reasons=[f"trigger={plan.trigger}"],
            created_at=now,
            updated_at=now,
        )
        steps = self._steps(plan, event)
        workflow = WorkflowRequest(
            incident_id=incident.incident_id,
            source_plan_id=plan.plan_id,
            predecessor_workflow_id=predecessor_workflow_id,
            runtime_profile_version=plan.runtime_profile_version,
            status=WorkflowStatus.PENDING,
            official_action=incident.official_action,
            fencing_token=1,
            official_steps=steps,
            created_at=now,
            updated_at=now,
        )
        incident = incident.model_copy(
            update={"workflow_request_id": workflow.request_id}
        )
        self.store.save_incident_and_workflow(incident, workflow)
        return plan.model_copy(
            update={
                "incident_id": incident.incident_id,
                "workflow_request_id": workflow.request_id,
            }
        )

    def _steps(
        self, plan: RecoveryPlan, event: TerminalEvent
    ) -> list[WorkflowStepSpec]:
        steps = []
        validates_node_lifecycle = any(
            item.action
            in {
                RecoveryAction.REBOOT_NODE,
                RecoveryAction.REPLACE_NODE,
            }
            for item in plan.steps
        )
        for item in plan.steps:
            if item.action is RecoveryAction.VALIDATE_NODE and validates_node_lifecycle:
                for operation in (
                    WorkflowOperation.VALIDATE_GPU,
                    WorkflowOperation.VALIDATE_HOST,
                    WorkflowOperation.VALIDATE_FABRIC,
                ):
                    steps.append(
                        WorkflowStepSpec(
                            operation=operation,
                            execution_owner=item.execution_owner,
                            node_ids=item.node_ids,
                            gpu_uuids=item.gpu_uuids,
                        )
                    )
                continue
            operation = operation_for_action(item.action)
            owner = (
                self.evidence_owner
                if operation is WorkflowOperation.FREEZE_EVIDENCE
                else item.execution_owner
            )
            if operation is WorkflowOperation.RESET_GPU:
                for preparation in (
                    WorkflowOperation.QUIESCE_GPU_SERVICES,
                    WorkflowOperation.VERIFY_NO_GPU_CLIENTS,
                ):
                    steps.append(
                        WorkflowStepSpec(
                            operation=preparation,
                            execution_owner=owner,
                            node_ids=item.node_ids,
                            gpu_uuids=item.gpu_uuids,
                        )
                    )
            workload_ids = (
                event.workload_ids
                if operation
                in {
                    WorkflowOperation.STOP_WORKLOADS,
                    WorkflowOperation.RESTART_WORKLOAD,
                }
                else []
            )
            parameters = dict(item.parameters)
            if operation is WorkflowOperation.RESTART_WORKLOAD:
                parameters.update(
                    {
                        "cluster_id": event.cluster_id,
                        "job_id": event.job_id,
                        "source_attempt_id": event.attempt_id,
                        "source_gpu_count": event.gpu_count,
                        "restart_budget": event.restart_budget,
                    }
                )
                if plan.avoid_node_ids:
                    # The plan's only isolation lever for a node it does not
                    # trust; the Kubernetes restart adapter reads it (F-G5).
                    parameters["avoid_node_ids"] = sorted(set(plan.avoid_node_ids))
                if plan.restart_after_incident_id:
                    # The planner only names the incident the restart waits on;
                    # deriving the premise here keeps the adapter reading the
                    # incident's state at execution time, not plan time (F-G5).
                    premise = self.store.get_incident(plan.restart_after_incident_id)
                    parameters.update(
                        {
                            "requires_incident_state": IncidentState.RECOVERED.value,
                            "incident_id": premise.incident_id,
                            "incident_node_ids": sorted(premise.node_ids),
                        }
                    )
            steps.append(
                WorkflowStepSpec(
                    operation=operation,
                    execution_owner=owner,
                    node_ids=item.node_ids,
                    gpu_uuids=item.gpu_uuids,
                    workload_ids=workload_ids,
                    parameters=parameters,
                )
            )
            if operation is WorkflowOperation.RESET_GPU:
                steps.append(
                    WorkflowStepSpec(
                        operation=WorkflowOperation.RESTORE_GPU_SERVICES,
                        execution_owner=owner,
                        node_ids=item.node_ids,
                        gpu_uuids=item.gpu_uuids,
                    )
                )
        return steps

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
from gpu_fault.store import NotFoundError


ACTION_OPERATION = {
    RecoveryAction.MARK_UNSCHEDULABLE: (WorkflowOperation.MARK_UNSCHEDULABLE),
    RecoveryAction.DRAIN: WorkflowOperation.MARK_UNSCHEDULABLE,
    RecoveryAction.STOP_WORKLOAD: WorkflowOperation.STOP_WORKLOADS,
    RecoveryAction.COLLECT_EVIDENCE: (WorkflowOperation.FREEZE_EVIDENCE),
    RecoveryAction.RESTART_WORKLOAD: (WorkflowOperation.RESTART_WORKLOAD),
    RecoveryAction.RESET_GPU: WorkflowOperation.RESET_GPU,
    RecoveryAction.REBOOT_NODE: WorkflowOperation.RESTART_NODE,
    RecoveryAction.REPLACE_NODE: WorkflowOperation.REPLACE_NODE,
    RecoveryAction.RUN_DIAGNOSTICS: WorkflowOperation.VALIDATE_GPU,
    RecoveryAction.VALIDATE_NODE: WorkflowOperation.VALIDATE_GPU,
    RecoveryAction.RESTORE_SCHEDULING: (WorkflowOperation.RESTORE_SCHEDULING),
    RecoveryAction.QUARANTINE: WorkflowOperation.QUARANTINE,
    RecoveryAction.ESCALATE_OPERATOR: (WorkflowOperation.ESCALATE_SUPPORT),
}


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

    def compile(self, plan: RecoveryPlan, event: TerminalEvent) -> RecoveryPlan:
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
            operation = ACTION_OPERATION[item.action]
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

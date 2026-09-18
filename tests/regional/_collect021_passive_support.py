"""Typed persisted-state fixtures for the current passive completion contract."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from gpu_fault.models import (
    AllocationEntry,
    CompletionDecision,
    DecisionStatus,
    Environment,
    IncidentState,
    PlanStatus,
    PlanStep,
    RecoveryAction,
    RecoveryPlan,
    TerminalEvent,
    TerminalStatus,
    WorkflowOperation,
    WorkflowStatus,
)
from gpu_fault.regional import RemoteActionCommand, RemoteCommandStatus
from gpu_fault.watcher import AttemptObservation, ContainerObservation, WorkloadPhase
from tests._builders import (
    fault_incident,
    workflow_request,
    workflow_step,
    workflow_step_execution,
)


def completed_passive_state(
    *,
    cluster_id: str = "cluster-a",
    job_id: str = "job",
    attempt_id: str = "attempt-a",
    new_attempt_id: str = "attempt-b",
    pods: list[dict[str, Any]] | None = None,
    with_predecessor: bool = False,
) -> dict[str, Any]:
    now = datetime.now(timezone.utc)
    pods = pods or [
        {
            "uid": f"new-pod-{index}",
            "name": f"trainer-{index}",
            "node": f"node-{index}",
            "attempt_id": new_attempt_id,
            "phase": "Running",
        }
        for index in range(3)
    ]
    event = TerminalEvent(
        cluster_id=cluster_id,
        environment=Environment.HYPERPOD_EKS,
        job_id=job_id,
        attempt_id=attempt_id,
        terminal_status=TerminalStatus.FAILED,
        ended_at=now,
        runtime_profile_version="profile",
        allocation=[
            AllocationEntry(node_id=pod["node"], rank=index, gpu_count=8)
            for index, pod in enumerate(pods)
        ],
    )
    incident = fault_incident(
        "recovery-incident",
        event.event_key,
        cluster_id=cluster_id,
        job_id=job_id,
        attempt_id=attempt_id,
        workflow_request_id="recovery-workflow",
        state=IncidentState.RECOVERED,
    )
    plan = RecoveryPlan(
        plan_id="recovery-plan",
        incident_id=incident.incident_id,
        attempt_id=attempt_id,
        trigger="no-hardware-evidence:RESTART",
        runtime_profile_version="profile",
        status=PlanStatus.SUCCEEDED,
        workflow_request_id=incident.workflow_request_id,
        steps=[
            PlanStep(
                action=RecoveryAction.RESTART_WORKLOAD,
                execution_owner="gpu-fault-kubernetes-adapter",
            )
        ],
        restart_after_incident_id="containment-incident" if with_predecessor else None,
    )
    workflow = workflow_request(
        incident.workflow_request_id,
        incident.incident_id,
        status=WorkflowStatus.SUCCEEDED,
        source_plan_id=plan.plan_id,
        official_steps=[
            workflow_step(
                WorkflowOperation.RESTART_WORKLOAD,
                "gpu-fault-kubernetes-adapter",
                workload_ids=["gpu-system/pytorchjob/source"],
                parameters={
                    "cluster_id": cluster_id,
                    "job_id": job_id,
                    "source_attempt_id": attempt_id,
                },
            )
        ],
        completed_operations=[WorkflowOperation.RESTART_WORKLOAD],
        completed_step_indexes=[0],
        step_executions=[
            workflow_step_execution(
                0,
                WorkflowOperation.RESTART_WORKLOAD,
                adapter_operation_id="remote/recovery-command",
                details={"restart_attempt_id": new_attempt_id},
            )
        ],
        predecessor_workflow_id="containment-workflow" if with_predecessor else None,
    )
    command = RemoteActionCommand(
        command_id="recovery-command",
        cluster_id=cluster_id,
        workflow_request_id=workflow.request_id,
        incident_id=incident.incident_id,
        step_index=0,
        fencing_token=workflow.fencing_token,
        idempotency_key="restart-operation",
        step=workflow.official_steps[0],
        workflow=workflow,
        incident=incident,
        status=RemoteCommandStatus.SUCCEEDED,
        last_lease_owner="fixture-executor",
        result_details={"restart_attempt_id": new_attempt_id},
    )
    decision = CompletionDecision(
        cluster_id=cluster_id,
        attempt_id=attempt_id,
        event_key=event.event_key,
        status=DecisionStatus.PLAN_CREATED,
        reason="no hardware evidence",
        recovery_plan_id=plan.plan_id,
    )
    observation = AttemptObservation(
        cluster_id=cluster_id,
        environment=Environment.HYPERPOD_EKS,
        job_id=job_id,
        attempt_id=new_attempt_id,
        workload_phase=WorkloadPhase.RUNNING,
        observed_at=now,
        runtime_profile_version="profile",
        expected_critical_ranks=len(pods),
        containers=[
            ContainerObservation(
                pod_uid=pod["uid"],
                pod_name=pod.get("name") or f"trainer-{index}",
                container_name="trainer",
                role="worker",
                rank=index,
                node_id=pod["node"],
                gpu_count=8,
            )
            for index, pod in enumerate(pods)
        ],
    )
    predecessor = workflow_request(
        "containment-workflow", "containment-incident", status=WorkflowStatus.SUCCEEDED
    )
    predecessor_incident = fault_incident(
        predecessor.incident_id,
        "containment-event",
        cluster_id=cluster_id,
        job_id=job_id,
        attempt_id=attempt_id,
        workflow_request_id=predecessor.request_id,
        state=IncidentState.RECOVERED,
    )
    return {
        "captured_at": now.isoformat(),
        "completion_event": {
            **event.model_dump(mode="json"),
            "event_key": event.event_key,
        },
        "completion_decision": decision.model_dump(mode="json"),
        "recovery_plan": plan.model_dump(mode="json"),
        "recovery_workflow": workflow.model_dump(mode="json"),
        "recovery_incident": incident.model_dump(mode="json"),
        "commands": [command.model_dump(mode="json", exclude={"lease_token"})],
        "predecessor_workflow": (
            predecessor.model_dump(mode="json") if with_predecessor else None
        ),
        "predecessor_incident": (
            predecessor_incident.model_dump(mode="json") if with_predecessor else None
        ),
        "observations": [observation.model_dump(mode="json")],
        "restart_budget": {"budget": 1, "restart_count": 1},
    }

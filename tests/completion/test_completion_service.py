from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from gpu_fault.app import ApplicationContext
from gpu_fault.models import (
    DecisionStatus,
    IncidentState,
    MarkerScope,
    NodeMarker,
    RecoveryAction,
    Severity,
    TerminalEvent,
    TerminalStatus,
    WorkflowOperation,
    WorkflowStatus,
)
from gpu_fault.passive import PassiveWorkflowCompiler
from gpu_fault.service import CompletionPendingError, CompletionService
from gpu_fault.telemetry import EvidenceKind
from gpu_fault.watcher import (
    AllocationCompleteness,
    FailureDetectedEvent,
    failure_containment_ids,
)
from tests._builders import copy_model, fault_incident, workflow_request, workflow_step


def marker(
    ended_at: datetime,
    *,
    node_id: str,
    action: RecoveryAction,
    marker_id: str = "marker-1",
) -> NodeMarker:
    return NodeMarker(
        marker_id=marker_id,
        source="test-agent",
        trusted=True,
        incident_id="inc-existing",
        observed_at=ended_at - timedelta(seconds=10),
        expires_at=ended_at + timedelta(minutes=30),
        scope=MarkerScope(node_ids=[node_id]),
        severity=Severity.CRITICAL,
        recommended_action=action,
        action_owner="simulated-runtime",
        mapping_version="test-v1",
    )


def test_unrelated_marker_does_not_match(
    context: ApplicationContext, failed_event: TerminalEvent, ended_at: datetime
) -> None:
    context.completion.add_marker(
        marker(
            ended_at,
            node_id="node-not-in-allocation",
            action=RecoveryAction.REBOOT_NODE,
        )
    )

    decision = context.completion.handle_terminal(failed_event)
    plan = context.store.get_plan(decision.recovery_plan_id)

    assert decision.status is DecisionStatus.PLAN_CREATED
    assert decision.matched_marker_ids == []
    assert plan.trigger == "no-hardware-evidence:RESTART"


def test_failure_detection_creates_idempotent_containment_workflow(
    context: ApplicationContext, ended_at: datetime
) -> None:
    event = FailureDetectedEvent(
        cluster_id="cluster-a",
        job_id="train-123",
        attempt_id="train-123-a1",
        detected_at=ended_at,
        runtime_profile_version="simulated-v1",
        workload_ids=["training/pytorchjob/distributed-training"],
        node_ids=["node-a", "node-b"],
        gpu_uuids=["GPU-a", "GPU-b"],
        first_failed_rank=0,
        node_id="node-a",
        exit_code=1,
        reason="critical container exited non-zero",
        allocation_completeness=AllocationCompleteness.COMPLETE,
    )

    first = context.completion.handle_failure_detected(event)
    second = context.completion.handle_failure_detected(event)
    workflow = context.store.get_workflow(first.workflow_request_id)
    incident = context.store.get_incident(first.incident_id)

    assert not first.duplicate
    assert second.duplicate
    assert second.workflow_request_id == first.workflow_request_id
    assert incident.event_type == ("TRAINING_ATTEMPT_FAILURE_DETECTED")
    assert [step.operation for step in workflow.official_steps] == [
        WorkflowOperation.FREEZE_EVIDENCE,
        WorkflowOperation.STOP_WORKLOADS,
    ]


def test_failure_detection_persists_workload_log_snapshot(
    context: ApplicationContext, ended_at: datetime
) -> None:
    snapshot = {
        "record_id": "workload-log/failure-snapshot",
        "node_id": "node-a",
        "captured_at": ended_at.isoformat(),
        "pod_name": "worker-0",
        "tail": "training output",
        "tail_bytes": 15,
        "truncated": False,
        "s3_uri": "s3://bucket/log.gz",
    }
    event = FailureDetectedEvent(
        cluster_id="cluster-a",
        job_id="train-log",
        attempt_id="train-log-a1",
        detected_at=ended_at,
        runtime_profile_version="simulated-v1",
        workload_ids=["training/pytorchjob/train-log"],
        node_ids=["node-a"],
        first_failed_rank=0,
        node_id="node-a",
        exit_code=1,
        reason="critical container exited non-zero",
        allocation_completeness=AllocationCompleteness.COMPLETE,
        workload_log_snapshots=[snapshot],
    )

    context.completion.handle_failure_detected(event)

    evidence = context.store.list_raw_evidence(
        "cluster-a", attempt_id="train-log-a1", kind=EvidenceKind.WORKLOAD_LOG
    )
    assert len(evidence) == 1
    assert evidence[0].payload["tail"] == "training output"


def test_passive_containment_terminal_continues_to_recovery(
    context: ApplicationContext, failed_event: TerminalEvent, ended_at: datetime
) -> None:
    containment = context.completion.handle_failure_detected(
        FailureDetectedEvent(
            cluster_id=failed_event.cluster_id,
            job_id=failed_event.job_id,
            attempt_id=failed_event.attempt_id,
            detected_at=ended_at,
            runtime_profile_version=(failed_event.runtime_profile_version),
            workload_ids=["training/pytorchjob/distributed-training"],
            node_ids=["node-a", "node-b"],
            gpu_uuids=["GPU-a", "GPU-b"],
            first_failed_rank=0,
            node_id="node-a",
            exit_code=1,
            reason="critical container exited non-zero",
            allocation_completeness=(AllocationCompleteness.COMPLETE),
        )
    )
    terminal = copy_model(
        failed_event, termination_initiator_incident_id=containment.incident_id
    )
    workflow = context.store.get_workflow(containment.workflow_request_id)
    context.store.save_workflow(copy_model(workflow, status=WorkflowStatus.SUCCEEDED))

    decision = context.completion.handle_terminal(terminal)
    plan = context.store.get_plan(decision.recovery_plan_id)

    assert decision.status is DecisionStatus.PLAN_CREATED
    assert [step.action for step in plan.steps] == [RecoveryAction.RESTART_WORKLOAD]


def test_emergency_stopped_terminal_continues_to_recovery(
    context: ApplicationContext, failed_event: TerminalEvent, ended_at: datetime
) -> None:
    containment = context.completion.handle_failure_detected(
        FailureDetectedEvent(
            cluster_id=failed_event.cluster_id,
            job_id=failed_event.job_id,
            attempt_id=failed_event.attempt_id,
            detected_at=ended_at,
            runtime_profile_version=(failed_event.runtime_profile_version),
            workload_ids=["training/pytorchjob/distributed-training"],
            node_ids=["node-a", "node-b"],
            gpu_uuids=["GPU-a", "GPU-b"],
            first_failed_rank=0,
            node_id="node-a",
            exit_code=143,
            reason="emergency fallback stopped the attempt",
            allocation_completeness=(AllocationCompleteness.COMPLETE),
        )
    )
    workflow = context.store.get_workflow(containment.workflow_request_id)
    context.store.save_workflow(copy_model(workflow, status=WorkflowStatus.SUCCEEDED))
    terminal = copy_model(
        failed_event,
        terminal_status=TerminalStatus.STOPPED,
        termination_initiator_incident_id=containment.incident_id,
    )

    decision = context.completion.handle_terminal(terminal)
    plan = context.store.get_plan(decision.recovery_plan_id)

    assert decision.status is DecisionStatus.PLAN_CREATED
    assert [step.action for step in plan.steps] == [RecoveryAction.RESTART_WORKLOAD]


def test_passive_terminal_waits_for_incident_creation(
    context: ApplicationContext, failed_event: TerminalEvent
) -> None:
    attempt_id = "train-123-passive-race"
    incident_id, _ = failure_containment_ids(
        (f"{failed_event.cluster_id}/{attempt_id}/TrainingAttemptFailureDetected")
    )
    terminal = copy_model(
        failed_event,
        attempt_id=attempt_id,
        terminal_status=TerminalStatus.STOPPED,
        termination_initiator_incident_id=incident_id,
    )

    with pytest.raises(CompletionPendingError, match="incident is not persisted yet"):
        context.completion.handle_terminal(terminal)

    assert context.store.get_decision_by_event(terminal.event_key) is None


def test_terminal_waits_for_passive_containment(
    context: ApplicationContext, failed_event: TerminalEvent, ended_at: datetime
) -> None:
    context.completion.handle_failure_detected(
        FailureDetectedEvent(
            cluster_id=failed_event.cluster_id,
            job_id=failed_event.job_id,
            attempt_id=failed_event.attempt_id,
            detected_at=ended_at,
            runtime_profile_version=(failed_event.runtime_profile_version),
            workload_ids=["training/pytorchjob/distributed-training"],
            node_ids=["node-a", "node-b"],
            gpu_uuids=["GPU-a", "GPU-b"],
            first_failed_rank=0,
            node_id="node-a",
            exit_code=1,
            reason="critical container exited non-zero",
            allocation_completeness=(AllocationCompleteness.COMPLETE),
        )
    )

    with pytest.raises(
        CompletionPendingError, match="containment workflow is not complete"
    ):
        context.completion.handle_terminal(failed_event)

    assert context.store.get_decision_by_event(failed_event.event_key) is None


def test_matching_marker_reuses_incident_and_is_idempotent(
    context: ApplicationContext, failed_event: TerminalEvent, ended_at: datetime
) -> None:
    context.completion.add_marker(
        marker(ended_at, node_id="node-a", action=RecoveryAction.REBOOT_NODE)
    )

    first = context.completion.handle_terminal(failed_event)
    second = context.completion.handle_terminal(failed_event)

    assert first.status is DecisionStatus.PLAN_CREATED
    assert not first.duplicate
    assert second.duplicate
    assert second.recovery_plan_id == first.recovery_plan_id

    plan = context.store.get_plan(first.recovery_plan_id)
    assert plan.incident_id == "inc-existing"
    assert [step.action for step in plan.steps] == [
        RecoveryAction.MARK_UNSCHEDULABLE,
        RecoveryAction.COLLECT_EVIDENCE,
        RecoveryAction.REBOOT_NODE,
        RecoveryAction.VALIDATE_NODE,
        RecoveryAction.RESTORE_SCHEDULING,
    ]


def test_no_marker_restarts_once_within_budget(
    context: ApplicationContext, failed_event: TerminalEvent
) -> None:
    service = CompletionService(
        context.store, workflow_compiler=PassiveWorkflowCompiler(context.store)
    )

    decision = service.handle_terminal(failed_event)

    assert decision.status is DecisionStatus.PLAN_CREATED
    assert "restart budget" in decision.reason
    plan = context.store.get_plan(decision.recovery_plan_id)
    assert plan.trigger == "no-hardware-evidence:RESTART"
    assert [step.action for step in plan.steps] == [RecoveryAction.RESTART_WORKLOAD]
    workflow = context.store.get_workflow(plan.workflow_request_id)
    restart = workflow.official_steps[-1]
    assert restart.operation is WorkflowOperation.RESTART_WORKLOAD
    assert restart.parameters["restart_budget"] == failed_event.restart_budget
    assert restart.parameters["job_id"] == failed_event.job_id


def test_stopped_attempt_does_not_restart(
    context: ApplicationContext, failed_event: TerminalEvent
) -> None:
    stopped = copy_model(
        failed_event,
        attempt_id="train-123-a-stopped",
        terminal_status=TerminalStatus.STOPPED,
        termination_initiator_incident_id="inc-controller",
    )

    decision = context.completion.handle_terminal(stopped)

    assert decision.status is DecisionStatus.NO_ACTION
    assert decision.recovery_plan_id is None


def test_controller_initiated_failed_exit_does_not_recurse(
    context: ApplicationContext, failed_event: TerminalEvent
) -> None:
    incident = fault_incident(
        "inc-xid-95",
        "xid-95",
        cluster_id=failed_event.cluster_id,
        job_id=failed_event.job_id,
        attempt_id="train-123-controlled",
        policy_version="test",
        policy_source="PREEMPT-025",
        state=IncidentState.ACTION_PENDING,
        fencing_token=1,
    )
    workflow = workflow_request(
        "workflow-xid-95",
        incident.incident_id,
        WorkflowStatus.RUNNING,
        1,
        official_action="RESET_GPU",
        official_steps=[
            workflow_step(
                WorkflowOperation.STOP_WORKLOADS,
                "gpu-fault-kubernetes-adapter",
                workload_ids=["training/pytorchjob/distributed-training"],
                parameters={"termination_initiator_incident_id": incident.incident_id},
            )
        ],
    )
    incident = copy_model(incident, workflow_request_id=workflow.request_id)
    context.store.save_incident_and_workflow(incident, workflow)
    context.store.reserve_job_restart(
        failed_event.cluster_id, failed_event.job_id, 3, "existing-restart"
    )
    before_counts = (
        len(context.store._incidents),
        len(context.store.list_workflows()),
        len(context.store.list_markers()),
    )
    before_budget = context.store.get_restart_budget(
        failed_event.cluster_id, failed_event.job_id
    )
    before_workflow = context.store.get_workflow(workflow.request_id)
    controlled = copy_model(
        failed_event,
        attempt_id="train-123-controlled",
        termination_initiator_incident_id="inc-xid-95",
    )

    decision = context.completion.handle_terminal(controlled)

    assert decision.status is DecisionStatus.NO_ACTION
    assert decision.recovery_plan_id is None
    assert "without recursive recovery" in decision.reason
    assert decision.matched_marker_ids == []
    assert (
        len(context.store._incidents),
        len(context.store.list_workflows()),
        len(context.store.list_markers()),
    ) == before_counts
    assert (
        context.store.get_restart_budget(failed_event.cluster_id, failed_event.job_id)
        == before_budget
    )
    assert context.store.get_workflow(workflow.request_id) == before_workflow


def test_missing_allocation_blocks_automatic_restart(
    context: ApplicationContext, failed_event: TerminalEvent
) -> None:
    event = copy_model(
        failed_event, attempt_id="train-123-no-allocation", allocation=[]
    )

    decision = context.completion.handle_terminal(event)
    plan = context.store.get_plan(decision.recovery_plan_id)

    assert decision.status is DecisionStatus.PLAN_CREATED
    assert decision.diagnostic_request_id is None
    assert plan.trigger == "allocation-missing:INCONCLUSIVE"
    assert plan.avoid_node_ids == []
    assert [step.action for step in plan.steps] == [
        RecoveryAction.COLLECT_EVIDENCE,
        RecoveryAction.ESCALATE_OPERATOR,
    ]
    assert all(not step.node_ids for step in plan.steps)


def test_diagnostic_marker_creates_diagnostic_plan(
    context: ApplicationContext, failed_event: TerminalEvent, ended_at: datetime
) -> None:
    context.completion.add_marker(
        marker(ended_at, node_id="node-a", action=RecoveryAction.RUN_DIAGNOSTICS)
    )

    decision = context.completion.handle_terminal(failed_event)
    plan = context.store.get_plan(decision.recovery_plan_id)

    assert [step.action for step in plan.steps] == [
        RecoveryAction.COLLECT_EVIDENCE,
        RecoveryAction.RUN_DIAGNOSTICS,
    ]
    assert plan.avoid_node_ids == ["node-a"]

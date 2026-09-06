"""A remediation that ran out of lifetime hands the node to an operator; the
support workflow that does the handing must itself be reachable.

DESTR-018 judged this red by design: the support successor inherited the
predecessor's already-expired ``lifetime_deadline_at`` ("the chain shares one
lifetime"), and the first claim only stamps a lifetime when none is set, so
the successor was failed as lifetime-exceeded before FREEZE_EVIDENCE ran. The
shared lifetime bounds *automatic* rungs (reboot, replacement); an operator
hand-off (support ticket, drain) starts its own clock.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from gpu_fault.execution.models import WorkflowExecutionRequest
from gpu_fault.execution.restart_budget_preflight import claim_deadlines
from gpu_fault.host_health import NodeHealthCategory
from gpu_fault.models import (
    RecoveryAction,
    WorkflowOperation,
    WorkflowStatus,
    WorkflowStepStatus,
    WorkloadState,
    lifetime_exceeded,
)
from tests._builders import (
    active_workflow_executor,
    build_context,
    copy_model,
    node_health_finding,
    workflow_step_execution,
)
from tests.execution._support import FakeAdapter, WorkflowStepOutcome

NOW = datetime(2026, 9, 6, 8, 0, tzinfo=timezone.utc)
SUPPORT_STEPS = [
    WorkflowOperation.FREEZE_EVIDENCE,
    WorkflowOperation.MARK_UNSCHEDULABLE,
    WorkflowOperation.QUARANTINE,
    WorkflowOperation.ESCALATE_SUPPORT,
]


class _AnyOwnerAdapter(FakeAdapter):
    """The compiled support steps carry the profile's execution owner, not
    the fake's ``owner-a``; the fake still answers only for the operations
    it was given outcomes for."""

    def supports(self, step) -> bool:
        return step.operation in self.outcomes


def _expired_reset_workflow(context, *, lifetime_ago: timedelta):
    finding = node_health_finding(
        "finding-lifetime",
        "lifetime",
        observed_at=NOW,
        category=NodeHealthCategory.GPU,
        severity="critical",
        reason="uncorrectable ECC errors detected",
        recommended_action=RecoveryAction.RESET_GPU,
        gpu_uuids=["GPU-a"],
        runtime_profile_version="simulated-v1",
        workload_state=WorkloadState.IDLE,
    )
    _, reset = context.orchestrator.ingest_node_health(finding)
    assert reset is not None, "the finding must compile a RESET_GPU workflow"
    verify_index = next(
        index
        for index, step in enumerate(reset.official_steps)
        if step.operation is WorkflowOperation.VERIFY_NO_GPU_CLIENTS
    )
    expired = datetime.now(timezone.utc) - lifetime_ago
    failed = copy_model(
        reset,
        status=WorkflowStatus.FAILED,
        lifetime_deadline_at=expired,
        completed_operations=[
            WorkflowOperation.FREEZE_EVIDENCE,
            WorkflowOperation.MARK_UNSCHEDULABLE,
            WorkflowOperation.QUIESCE_GPU_SERVICES,
        ],
        step_executions=[
            workflow_step_execution(
                verify_index,
                WorkflowOperation.VERIFY_NO_GPU_CLIENTS,
                WorkflowStepStatus.FAILED,
                error="workflow lifetime exceeded before step",
                details={
                    "workflow_lifetime_exceeded": True,
                    "workflow_execution_deadline": expired.isoformat(),
                },
            )
        ],
    )
    context.store.save_workflow(failed)
    return failed


def test_the_support_successor_of_an_expired_remediation_gets_its_own_lifetime():
    context = build_context()
    failed = _expired_reset_workflow(context, lifetime_ago=timedelta(minutes=5))

    escalated = context.orchestrator.escalate_failed_hardware_remediation(failed)

    assert escalated is not None, "lifetime_exceeded must still hand off to support"
    incident, support = escalated
    assert [step.operation for step in support.official_steps] == SUPPORT_STEPS
    assert support.status is WorkflowStatus.PENDING
    assert support.lifetime_deadline_at is None, (
        "an operator hand-off must not inherit the expired automatic lifetime"
    )
    assert not lifetime_exceeded(support), "the successor is not born expired"
    execution, lifetime = claim_deadlines(
        support,
        datetime.now(timezone.utc),
        timeout_seconds=600,
        job_lifetime_seconds=3600,
        node_lifetime_seconds=3600,
    )
    assert lifetime > datetime.now(timezone.utc), (
        "the first claim stamps a fresh lifetime"
    )
    assert execution <= lifetime


def test_the_support_successor_reaches_its_escalation_step():
    context = build_context()
    failed = _expired_reset_workflow(context, lifetime_ago=timedelta(minutes=5))
    incident, support = context.orchestrator.escalate_failed_hardware_remediation(
        failed
    )
    adapter = _AnyOwnerAdapter(
        {op: WorkflowStepOutcome.succeeded() for op in SUPPORT_STEPS}
    )
    executor = active_workflow_executor(context.store, [adapter], set(SUPPORT_STEPS))

    result = executor.execute(
        support.request_id,
        WorkflowExecutionRequest(expected_fencing_token=support.fencing_token),
    )

    assert result.status is WorkflowStatus.SUCCEEDED, result.error
    assert [call.rsplit("/", 1)[1] for call in adapter.calls] == [
        op.value for op in SUPPORT_STEPS
    ]
    current = context.store.get_workflow(support.request_id)
    assert not any(
        execution.details.get("workflow_lifetime_exceeded")
        for execution in current.step_executions
    ), current.step_executions
    assert context.orchestrator.escalate_failed_hardware_remediation(current) is None, (
        "a completed support hand-off never escalates again"
    )


def test_an_automatic_rung_still_shares_a_live_lifetime():
    """A reboot successor emitted while the lifetime is still ahead keeps the
    chain's single clock; only operator hand-offs start their own."""

    context = build_context()
    finding = node_health_finding(
        "finding-reset-fail",
        "reset-fail",
        observed_at=NOW,
        category=NodeHealthCategory.GPU,
        severity="critical",
        reason="uncorrectable ECC errors detected",
        recommended_action=RecoveryAction.RESET_GPU,
        gpu_uuids=["GPU-a"],
        runtime_profile_version="simulated-v1",
        workload_state=WorkloadState.IDLE,
    )
    _, reset = context.orchestrator.ingest_node_health(finding)
    assert reset is not None, "the finding must compile a RESET_GPU workflow"
    reset_index = next(
        index
        for index, step in enumerate(reset.official_steps)
        if step.operation is WorkflowOperation.RESET_GPU
    )
    ahead = datetime.now(timezone.utc) + timedelta(minutes=30)
    failed = copy_model(
        reset,
        status=WorkflowStatus.FAILED,
        lifetime_deadline_at=ahead,
        completed_operations=[
            WorkflowOperation.FREEZE_EVIDENCE,
            WorkflowOperation.MARK_UNSCHEDULABLE,
        ],
        step_executions=[
            workflow_step_execution(
                reset_index,
                WorkflowOperation.RESET_GPU,
                WorkflowStepStatus.FAILED,
                error="reset refused",
            )
        ],
    )
    context.store.save_workflow(failed)

    escalated = context.orchestrator.escalate_failed_hardware_remediation(failed)

    assert escalated is not None, "a failed reset escalates to a reboot"
    _, reboot = escalated
    assert WorkflowOperation.RESTART_NODE in {
        step.operation for step in reboot.official_steps
    }
    assert reboot.lifetime_deadline_at == ahead

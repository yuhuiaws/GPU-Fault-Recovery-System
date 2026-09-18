"""A spare's execution target must not replace its branch's stable lineage."""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from gpu_fault.execution import WorkflowStepContext, WorkflowStepOutcome
from gpu_fault.execution.branch_escalation import BranchEscalator
from gpu_fault.models import (
    WorkflowOperation,
    WorkflowRequest,
    WorkflowStatus,
    WorkflowStepSpec,
)
from gpu_fault.orchestration.arbitration import RecoveryArbiter
from gpu_fault.orchestration.dag_branching import DagBrancher, branch_node_ids
from tests._builders import (
    active_workflow_executor,
    workflow_request,
    workflow_step,
    workflow_step_execution,
)
from tests.execution._cov95_runtime_workflows import FlowHarness

OP = WorkflowOperation


def compile_steps(
    workflow: WorkflowRequest, operations: Sequence[OP], node: str, gpus: Sequence[str]
) -> list[WorkflowStepSpec]:
    return [
        workflow_step(operation, node_ids=[node], gpu_uuids=list(gpus))
        for operation in operations
    ]


@pytest.mark.parametrize(
    "failure",
    [OP.VALIDATE_GPU, OP.VALIDATE_HOST, OP.VALIDATE_FABRIC, OP.RESTORE_SCHEDULING],
)
@pytest.mark.parametrize("late_hold", [False, True])
def test_failed_spare_branch_settles_without_retiring_other_work(
    failure: OP, late_hold: bool
) -> None:
    flow = FlowHarness([OP.STOP_WORKLOADS, OP.RESTART_WORKLOAD])
    flow.amend(
        completed_step_indexes=[0],
        completed_operations=[OP.STOP_WORKLOADS],
        step_executions=[
            workflow_step_execution(0, OP.STOP_WORKLOADS, phase="official")
        ],
    )
    brancher = DagBrancher(RecoveryArbiter())
    replacement = workflow_request(
        "replacement",
        flow.incident.incident_id,
        official_steps=[
            workflow_step(operation)
            for operation in (
                OP.QUARANTINE,
                OP.REPLACE_NODE,
                OP.VALIDATE_GPU,
                OP.VALIDATE_HOST,
                OP.VALIDATE_FABRIC,
                OP.RESTORE_SCHEDULING,
            )
        ],
    )
    sibling = workflow_request(
        "sibling",
        flow.incident.incident_id,
        official_steps=[
            workflow_step(OP.VALIDATE_HOST, node_ids=["node-b"]),
            workflow_step(OP.RESTORE_SCHEDULING, node_ids=["node-b"]),
        ],
    )
    plan = brancher.append_parallel_job_branch(flow.workflow, replacement)
    plan = brancher.append_parallel_job_branch(plan, sibling)
    if late_hold:
        plan = brancher.append_parallel_job_branch_successor(
            plan,
            workflow_request(
                "late-hold",
                flow.incident.incident_id,
                official_steps=[workflow_step(OP.QUARANTINE)],
            ),
            "node-a",
        )
    flow.store.save_workflow(plan, expected=flow.workflow)
    flow.workflow = plan
    calls: list[WorkflowStepContext] = []

    class Boundary:
        def supports(self, step: WorkflowStepSpec) -> bool:
            return step.execution_owner == "owner-a"

        def execute(self, context: WorkflowStepContext) -> WorkflowStepOutcome:
            calls.append(context)
            if context.step.operation is OP.REPLACE_NODE:
                return WorkflowStepOutcome.succeeded(
                    details={
                        "action": "SPARE_FAILOVER",
                        "node_rebindings": {"node-a": "node-spare"},
                    }
                )
            if context.step.operation is failure and context.step.node_ids == [
                "node-spare"
            ]:
                return WorkflowStepOutcome.failed("post-replacement validation refused")
            return WorkflowStepOutcome.succeeded()

    flow.executor = active_workflow_executor(
        flow.store, [Boundary()], [step.operation for step in plan.official_steps]
    )
    flow.executor.branch_escalator = BranchEscalator(brancher, compile_steps)

    result = flow.execute()
    saved = flow.store.get_workflow(plan.request_id)
    count = len(calls)

    assert result.status is WorkflowStatus.FAILED, (
        "an exhausted spare branch must settle instead of polling failed validation"
    )
    assert len(saved.exhausted_branch_ids) == 1
    assert any(
        call.step.operation is OP.RESTORE_SCHEDULING
        and call.step.node_ids == ["node-b"]
        for call in calls
    ), "an independent healthy branch still owes its readmission"
    assert not any(call.step.operation is OP.RESTART_WORKLOAD for call in calls), (
        "the job join must not run after a branch exhausts"
    )
    if late_hold:
        assert sum(call.step.operation is OP.QUARANTINE for call in calls) == 2, (
            "a later quarantine for the failed original is not part of the exhausted tail"
        )
    failed = next(
        call
        for call in calls
        if call.step.operation is failure and call.step.node_ids == ["node-spare"]
    )
    assert failed.step.branch_node_ids == ["node-a"]
    assert failed.step_index in saved.superseded_step_indexes
    assert flow.execute().status is WorkflowStatus.FAILED
    assert len(calls) == count, (
        "terminal redispatch must not repeat failed spare validation"
    )


def rebound_plan(*, rungs: int = 0) -> WorkflowRequest:
    return workflow_request(
        "rebound",
        "incident",
        dag_enabled=True,
        dag_revision=1,
        branch_escalation_counts={"node-a": rungs} if rungs else {},
        official_steps=[
            workflow_step(
                OP.RESET_GPU,
                node_ids=["node-spare"],
                gpu_uuids=["GPU-spare"],
                branch_id="branch:node-a",
                branch_node_ids=["node-a"],
            )
        ],
    )


def test_exhaustion_is_idempotent_and_uses_the_fixed_branch_identity() -> None:
    escalator = BranchEscalator(DagBrancher(RecoveryArbiter()), compile_steps)
    first = escalator.exhaust_branch(
        rebound_plan(), "node-spare", "branch:node-a", reason="no further rung"
    )
    second = escalator.exhaust_branch(
        first.workflow, "node-spare", "branch:node-a", reason="no further rung"
    )

    assert first.workflow.superseded_step_indexes == [0]
    assert second.workflow == first.workflow, (
        "repeated exhaustion must add neither IDs nor events"
    )
    assert second.node_id == "node-spare", (
        "the diagnostic target remains the physical spare"
    )


def test_rebinding_does_not_reset_the_lineage_rung_budget() -> None:
    escalator = BranchEscalator(
        DagBrancher(RecoveryArbiter()), compile_steps, max_rungs=2
    )
    result = escalator.escalate_branch(rebound_plan(rungs=2), 0, "reset failed")

    assert result is not None and result.outcome == "exhausted"
    assert result.workflow.branch_escalation_counts == {"node-a": 2}
    assert result.node_id == "node-spare"


def test_a_remaining_rung_keeps_lineage_but_targets_the_physical_spare() -> None:
    escalator = BranchEscalator(
        DagBrancher(RecoveryArbiter()), compile_steps, max_rungs=2
    )
    result = escalator.escalate_branch(rebound_plan(rungs=1), 0, "reset failed")

    assert result is not None and result.outcome == "escalated"
    assert result.workflow.branch_escalation_counts == {"node-a": 2}
    assert result.workflow.superseded_step_indexes == [0]
    reboot = next(
        step
        for step in result.workflow.official_steps
        if step.operation is OP.RESTART_NODE
    )
    assert reboot.node_ids == ["node-spare"]
    assert reboot.branch_node_ids == ["node-a"]
    assert branch_node_ids(reboot.branch_id) == ("node-a",)

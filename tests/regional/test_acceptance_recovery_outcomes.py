"""PREEMPT-019: distinct known-failure, exhausted and unknown branch obligations."""

from __future__ import annotations

import pytest

from gpu_fault.execution.branch_escalation import BranchEscalator
from gpu_fault.execution.models import WorkflowStepOutcome
from gpu_fault.models import BlockedKind, WorkflowStatus
from gpu_fault.models import WorkflowOperation as Op
from gpu_fault.orchestration.arbitration import RecoveryArbiter
from gpu_fault.orchestration.dag_branching import DagBrancher
from tests._builders import (
    active_workflow_executor,
    build_store,
    execute_workflow,
    fault_incident,
    workflow_request,
    workflow_step,
    workflow_step_execution,
)
from tests.execution._support import RESTART_PARAMETERS


def branch_plan(store):
    operations = [
        Op.STOP_WORKLOADS,
        Op.RESET_GPU,
        Op.RESTART_NODE,
        Op.RESTORE_SCHEDULING,
        Op.RESTART_WORKLOAD,
    ]
    incident = fault_incident(
        "incident-alignment",
        "event-alignment",
        workflow_request_id="workflow-alignment",
        fencing_token=3,
    )
    steps = [
        workflow_step(
            operation,
            node_ids=["node-a", "node-b"]
            if index in {0, 4}
            else ["node-a"]
            if index == 1
            else ["node-b"],
            branch_id="shared"
            if index == 0
            else "join"
            if index == 4
            else "branch:node-a"
            if index == 1
            else "branch:node-b",
            branch_node_ids=[]
            if index in {0, 4}
            else ["node-a"]
            if index == 1
            else ["node-b"],
            depends_on_step_indexes={0: [], 1: [0], 2: [0], 3: [2], 4: [1, 3]}[index],
            parameters=dict(RESTART_PARAMETERS) if index == 4 else {},
        )
        for index, operation in enumerate(operations)
    ]
    workflow = workflow_request(
        "workflow-alignment",
        incident.incident_id,
        runtime_profile_version="simulated-v1",
        official_action="RESET_GPU",
        dag_enabled=True,
        dag_revision=1,
        official_steps=steps,
        completed_step_indexes=[0],
        completed_operations=[Op.STOP_WORKLOADS],
        step_executions=[workflow_step_execution(0, Op.STOP_WORKLOADS)],
    )
    store.save_incident_and_workflow(incident, workflow)
    return workflow


@pytest.mark.parametrize("scenario", ["recoverable", "exhausted", "unknown"])
def test_preempt019_branch_outcome_does_not_conflate_repair_failure_and_operator_hold(
    scenario,
):
    store = build_store()
    workflow = branch_plan(store)
    calls = []

    class Adapter:
        def supports(self, step):
            return step.execution_owner == "owner-a"

        def execute(self, context):
            operation, nodes = context.step.operation, context.step.node_ids
            calls.append((operation, tuple(nodes)))
            if nodes == ["node-a"] and operation is Op.RESET_GPU:
                return WorkflowStepOutcome.failed(
                    "controlled reset refusal",
                    details={
                        "outcome_unknown": True,
                        "manual_confirmation_required": True,
                    }
                    if scenario == "unknown"
                    else {"node_action_not_started": True},
                )
            if (
                scenario == "exhausted"
                and nodes == ["node-a"]
                and operation in {Op.RESTART_NODE, Op.REPLACE_NODE}
            ):
                return WorkflowStepOutcome.failed(
                    "confirmed no-start refusal",
                    details={"node_action_not_started": True},
                )
            return WorkflowStepOutcome.succeeded()

    executor = active_workflow_executor(store, [Adapter()], set(Op))
    executor.branch_escalator = BranchEscalator(
        DagBrancher(RecoveryArbiter()),
        lambda _workflow, operations, node, gpus: [
            workflow_step(operation, node_ids=[node], gpu_uuids=list(gpus))
            for operation in operations
        ],
    )
    result = execute_workflow(
        executor, workflow.request_id, expected_fencing_token=workflow.fencing_token
    )
    saved = store.get_workflow(workflow.request_id)
    assert (Op.RESTORE_SCHEDULING, ("node-b",)) in calls
    restarted = [item for item in calls if item[0] is Op.RESTART_WORKLOAD]
    if scenario == "recoverable":
        assert result.status is WorkflowStatus.SUCCEEDED
        assert saved.branch_escalation_counts == {"node-a": 1}
        assert len(restarted) == 1
        assert calls.index((Op.RESTORE_SCHEDULING, ("node-a",))) < calls.index(
            restarted[0]
        )
        assert calls.index((Op.RESTORE_SCHEDULING, ("node-b",))) < calls.index(
            restarted[0]
        )
    elif scenario == "exhausted":
        assert result.status is WorkflowStatus.FAILED
        assert saved.branch_escalation_counts == {"node-a": 2}
        assert saved.exhausted_branch_ids and not restarted
    else:
        assert result.status is WorkflowStatus.BLOCKED
        assert saved.blocked_kind is BlockedKind.NEEDS_OPERATOR
        assert saved.branch_escalation_counts == {} and not restarted
        assert (Op.RESTART_NODE, ("node-a",)) not in calls
        assert (Op.REPLACE_NODE, ("node-a",)) not in calls
        assert (Op.RESTORE_SCHEDULING, ("node-a",)) not in calls

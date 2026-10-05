"""A command the cluster executor refused before dispatch climbs no ladder.

Live (gpu-eks-cluster, XID 46 reset): a successor's plan rewrite moved the
incident to generation 2 while the reset chain held 1. The chain's RESTART_NODE
rung was refused "ClusterExecutorError: command fencing token does not match
workflow/incident" (``remote_status_source: executor-rejected``), the branch
ladder appended REPLACE_NODE, that was refused the same way, and so was the
RESTORE_GPU_SERVICES compensation -- three rungs spent on a record whose
incident had moved on, nothing learned about the node. Both ladders now stop
on such a failure: the branch is exhausted with a reason naming the refusal
and the record's generation, and the whole-workflow classifier opens no
successor for it.
"""

from __future__ import annotations

from gpu_fault.execution.branch_escalation import BranchEscalator
from gpu_fault.models import WorkflowOperation, WorkflowStatus, WorkflowStepStatus
from gpu_fault.orchestration.arbitration import RecoveryArbiter
from gpu_fault.orchestration.dag_branching import DagBrancher
from gpu_fault.orchestration.escalation import (
    EXECUTOR_REJECTED_STATUS_SOURCE,
    HardwareEscalationService,
    executor_rejected_failure,
)
from tests._builders import (
    copy_model,
    workflow_request,
    workflow_step,
    workflow_step_execution,
)

OP = WorkflowOperation
REJECTION = (
    "ClusterExecutorError: command fencing token does not match workflow/incident"
)


def _compile_steps(workflow, operations, node, gpus):
    return [
        workflow_step(operation, node_ids=[node], gpu_uuids=list(gpus))
        for operation in operations
    ]


def _reset_chain(**execution_details) -> tuple[object, int]:
    steps = [
        workflow_step(OP.MARK_UNSCHEDULABLE, node_ids=["node-a"], gpu_uuids=["GPU-a"]),
        workflow_step(
            OP.QUIESCE_GPU_SERVICES, node_ids=["node-a"], gpu_uuids=["GPU-a"]
        ),
        workflow_step(OP.RESET_GPU, node_ids=["node-a"], gpu_uuids=["GPU-a"]),
        workflow_step(
            OP.RESTORE_GPU_SERVICES, node_ids=["node-a"], gpu_uuids=["GPU-a"]
        ),
        workflow_step(OP.VALIDATE_GPU, node_ids=["node-a"], gpu_uuids=["GPU-a"]),
    ]
    workflow = workflow_request(
        "wf-reset",
        "inc-reset",
        status=WorkflowStatus.RUNNING,
        fencing_token=1,
        official_steps=steps,
        completed_step_indexes=[0, 1],
        completed_operations=[OP.MARK_UNSCHEDULABLE, OP.QUIESCE_GPU_SERVICES],
        step_executions=[
            workflow_step_execution(0, OP.MARK_UNSCHEDULABLE),
            workflow_step_execution(1, OP.QUIESCE_GPU_SERVICES),
            workflow_step_execution(
                2,
                OP.RESET_GPU,
                WorkflowStepStatus.FAILED,
                error=REJECTION,
                details=execution_details,
            ),
        ],
    )
    return workflow, 2


def test_the_predicate_reads_only_the_executor_rejection_source() -> None:
    assert executor_rejected_failure(
        {"remote_status_source": EXECUTOR_REJECTED_STATUS_SOURCE}
    ), "the executor's own refusal source"
    assert not executor_rejected_failure(
        {"remote_status_source": "workflow-timeout"}
    ), "a cancellation is classified elsewhere"
    assert not executor_rejected_failure({"reset_failed": ["GPU-a"]}), (
        "a node failure keeps its ladder"
    )
    assert not executor_rejected_failure(None), "no details: not a rejection"


def test_the_branch_ladder_exhausts_a_rejected_step_instead_of_climbing() -> None:
    workflow, failed_index = _reset_chain(
        remote_status_source=EXECUTOR_REJECTED_STATUS_SOURCE
    )
    escalator = BranchEscalator(DagBrancher(RecoveryArbiter()), _compile_steps)

    escalation = escalator.escalate_branch(
        workflow,
        failed_index,
        REJECTION,
        details={"remote_status_source": "executor-rejected"},
    )

    assert escalation is not None
    assert escalation.outcome == "exhausted", escalation.reason
    assert "refused the command before dispatch" in escalation.reason
    assert "generation 1" in escalation.reason, (
        "the reason names the generation the record held, so the operator can "
        "see the incident moved past it"
    )
    assert OP.RESTART_NODE not in {
        step.operation for step in escalation.workflow.official_steps
    }, "no reboot rung for a node the refused command never touched"


def test_the_same_failure_without_the_rejection_source_still_climbs() -> None:
    workflow, failed_index = _reset_chain(reset_failed=["GPU-a"])
    escalator = BranchEscalator(DagBrancher(RecoveryArbiter()), _compile_steps)

    escalation = escalator.escalate_branch(
        workflow, failed_index, "GPU 0: In use by another client", details={}
    )

    assert escalation is not None and escalation.outcome == "escalated"
    assert OP.RESTART_NODE in {
        step.operation for step in escalation.workflow.official_steps
    }, "a real node failure keeps its reboot rung"


def test_the_whole_workflow_classifier_opens_no_successor_for_a_rejection() -> None:
    workflow, _ = _reset_chain(remote_status_source=EXECUTOR_REJECTED_STATUS_SOURCE)
    failed = copy_model(workflow, status=WorkflowStatus.FAILED)

    assert HardwareEscalationService.classify(failed) is None, (
        "the node was never touched: no reboot-after, no support ticket"
    )

    real_failure, _ = _reset_chain(reset_failed=["GPU-a"])
    classified = HardwareEscalationService.classify(
        copy_model(real_failure, status=WorkflowStatus.FAILED)
    )
    assert classified is not None, "a real reset failure still escalates"

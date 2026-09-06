"""Branch identity, idempotency and the "no branch for this node" case.

FINAL-建议汇总 F-C3 (P0-55B, P0-63B, P1-64C). A branch id was spelled from
its node set alone, so two different branches on the same nodes collapsed
into one for every ``node_branch_step_indexes`` consumer. A branch that had
already *completed* still counted as "already present" and blocked a new
identical branch. And appending a branch successor for a node that had no
branch yet returned the existing workflow unchanged -- the candidate was
silently dropped.
"""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

from gpu_fault.models import WorkflowOperation, WorkflowRequest, WorkflowStatus
from gpu_fault.orchestration.arbitration import RecoveryArbiter
from gpu_fault.orchestration.dag_branching import DagBrancher
from gpu_fault.orchestration.families.node_lifecycle import (
    NodeLifecycleOperationService,
)
from tests._builders import copy_model, workflow_request, workflow_step

ALL_NODES = ["node-a", "node-b", "node-c"]


def _job_workflow(
    node_id: str,
    request_id: str,
    action: WorkflowOperation = WorkflowOperation.RESET_GPU,
) -> WorkflowRequest:
    return workflow_request(
        request_id,
        "incident-dag",
        fencing_token=1,
        runtime_profile_version="simulated-v1",
        official_action=action.value,
        official_steps=[
            workflow_step(
                WorkflowOperation.STOP_WORKLOADS,
                "owner",
                node_ids=ALL_NODES,
                workload_ids=["training/job/job-a"],
            ),
            workflow_step(action, "owner", node_ids=[node_id]),
            workflow_step(
                WorkflowOperation.RESTORE_SCHEDULING, "owner", node_ids=[node_id]
            ),
            workflow_step(
                WorkflowOperation.RESTART_WORKLOAD,
                "owner",
                node_ids=ALL_NODES,
                workload_ids=["training/job/job-a"],
            ),
        ],
    )


def _running_dag() -> WorkflowRequest:
    return copy_model(
        _job_workflow("node-a", "workflow-node-a"),
        status=WorkflowStatus.RUNNING,
        completed_step_indexes=[0],
        completed_operations=[WorkflowOperation.STOP_WORKLOADS],
    )


def _branch_ids(workflow: WorkflowRequest, node_id: str) -> set[str | None]:
    return {
        step.branch_id
        for step in workflow.official_steps
        if node_id in step.node_ids and step.branch_id not in {None, "shared", "join"}
    }


def test_two_different_branches_on_the_same_nodes_keep_distinct_ids():
    brancher = DagBrancher(RecoveryArbiter())
    dag = brancher.append_parallel_job_branch(
        _running_dag(), _job_workflow("node-b", "wf-reset-b")
    )

    dag = brancher.append_parallel_job_branch(
        dag, _job_workflow("node-b", "wf-reboot-b", WorkflowOperation.RESTART_NODE)
    )

    assert len(_branch_ids(dag, "node-b")) == 2


def test_a_completed_identical_branch_does_not_block_a_new_one():
    brancher = DagBrancher(RecoveryArbiter())
    dag = brancher.append_parallel_job_branch(
        _running_dag(), _job_workflow("node-b", "wf-reset-b")
    )
    branch_indexes = [
        index
        for index, step in enumerate(dag.official_steps)
        if "node-b" in step.node_ids and step.branch_id not in {None, "shared", "join"}
    ]
    done = copy_model(
        dag,
        completed_step_indexes=sorted({*dag.completed_step_indexes, *branch_indexes}),
    )

    again = brancher.append_parallel_job_branch(
        done, _job_workflow("node-b", "wf-reset-b-again")
    )

    assert again.dag_revision == done.dag_revision + 1
    assert len(again.official_steps) > len(done.official_steps)


def test_a_branch_successor_for_a_node_without_a_branch_is_appended_not_dropped():
    brancher = DagBrancher(RecoveryArbiter())
    dag = brancher.append_parallel_job_branch(
        _running_dag(), _job_workflow("node-b", "wf-reset-b")
    )

    merged = brancher.append_parallel_job_branch_successor(
        dag, _job_workflow("node-c", "wf-reset-c"), "node-c"
    )

    assert merged is not dag
    assert any(
        step.operation is WorkflowOperation.RESET_GPU and "node-c" in step.node_ids
        for step in merged.official_steps
    ), (
        'expected any( step.operation is WorkflowOperation.RESET_GPU and "node-c" in step.node_ids for step in merged.official_steps ) to be true'
    )


def test_node_lifecycle_does_not_branch_while_serialized_behind_an_incumbent():
    """P0-55B: with an in-flight node-exclusive incumbent the candidate must
    queue behind it, not be appended as a parallel branch of the active job
    workflow as well."""

    appended: list[str] = []
    fake_self = SimpleNamespace(
        callbacks=SimpleNamespace(
            active_job_recovery_workflow=lambda observation: (
                SimpleNamespace(incident_id="inc-job"),
                copy_model(_running_dag(), not_before=None),
            ),
            can_append_parallel_job_branch=lambda active, candidate: True,
        ),
        brancher=SimpleNamespace(
            append_parallel_job_branch=lambda *args: appended.append("branch")
        ),
        arbiter=RecoveryArbiter(),
    )
    state = SimpleNamespace(merge_existing=False)
    candidate = _job_workflow("node-b", "wf-candidate")

    result = NodeLifecycleOperationService._parallel_branch(
        fake_self,
        SimpleNamespace(observation=None),
        state,
        SimpleNamespace(incident_id="inc-candidate"),
        candidate,
        datetime.now(timezone.utc),
        incumbent=copy_model(_running_dag(), request_id="wf-incumbent"),
    )

    assert result is None
    assert appended == []


def test_widening_a_step_onto_another_node_does_not_move_it_to_that_nodes_branch():
    """F-B6 (2): identity is fixed at creation; scope may grow later."""
    from gpu_fault.orchestration.arbitration import RecoveryArbiter as _Arbiter
    from gpu_fault.orchestration.dag_branching import DagBrancher as _Brancher
    from tests._builders import copy_model as _copy
    from tests._builders import workflow_request as _wr
    from tests._builders import workflow_step as _ws

    brancher = _Brancher(_Arbiter())
    nodes = ["node-a", "node-b"]

    def job(request_id: str, node_id: str):
        return _wr(
            request_id,
            "inc-dag",
            fencing_token=1,
            official_steps=[
                _ws(
                    WorkflowOperation.STOP_WORKLOADS,
                    node_ids=nodes,
                    workload_ids=["training/job/j"],
                ),
                _ws(
                    WorkflowOperation.MARK_UNSCHEDULABLE,
                    node_ids=[node_id],
                    depends_on_step_indexes=[0],
                ),
                _ws(
                    WorkflowOperation.RESET_GPU,
                    node_ids=[node_id],
                    depends_on_step_indexes=[1],
                ),
                _ws(
                    WorkflowOperation.RESTART_WORKLOAD,
                    node_ids=nodes,
                    workload_ids=["training/job/j"],
                    depends_on_step_indexes=[2],
                ),
            ],
        )

    dag = brancher.append_parallel_job_branch(
        _copy(
            job("wf-job", "node-a"),
            status=WorkflowStatus.RUNNING,
            completed_step_indexes=[0],
        ),
        job("wf-b", "node-b"),
    )
    node_b_indexes = brancher.node_branch_step_indexes(dag, "node-b")
    cordon_b = next(
        index
        for index in node_b_indexes
        if dag.official_steps[index].operation is WorkflowOperation.MARK_UNSCHEDULABLE
    )
    # A merge widens node-b's cordon onto node-a as well.
    widened = _copy(
        dag,
        official_steps=[
            step.model_copy(update={"node_ids": ["node-a", "node-b"]})
            if index == cordon_b
            else step
            for index, step in enumerate(dag.official_steps)
        ],
    )

    assert brancher.node_branch_step_indexes(widened, "node-b") == node_b_indexes, (
        "node-b keeps its branch"
    )
    assert cordon_b not in brancher.node_branch_step_indexes(widened, "node-a"), (
        "node-a's lookup must not swallow node-b's branch through the widened cordon"
    )

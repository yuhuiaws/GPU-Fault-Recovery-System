"""An isolating candidate retires only what it owns, and isolation is budgeted.

F-C5 remainder (docs/review/FINAL-建议汇总.md). A QUARANTINE candidate for
one node of a multi-node job superseded the whole job's RESTART_WORKLOAD
join, so the other nodes were repaired and the job never restarted; a
support escalation did not count as terminal at all; and QUARANTINE claimed
no remediation budget, so a false-positive cascade could isolate nodes
without limit.
"""

from __future__ import annotations

from gpu_fault.execution.remediation_budget import (
    RemediationBudgetPolicy,
    remediation_budget_claims,
)
from gpu_fault.models import WorkflowOperation, WorkflowRequest, WorkflowStatus
from gpu_fault.orchestration.arbitration import RecoveryArbiter
from gpu_fault.orchestration.dag_branching import DagBrancher
from tests._builders import (
    copy_model,
    fault_incident,
    workflow_request,
    workflow_step,
    workflow_step_execution,
)

STOP = WorkflowOperation.STOP_WORKLOADS
RESET = WorkflowOperation.RESET_GPU
RESTORE = WorkflowOperation.RESTORE_SCHEDULING
RESTART_JOB = WorkflowOperation.RESTART_WORKLOAD
QUARANTINE = WorkflowOperation.QUARANTINE
SUPPORT = WorkflowOperation.ESCALATE_SUPPORT
JOB_NODES = ["node-a", "node-b"]


def _job_dag() -> WorkflowRequest:
    brancher = DagBrancher(RecoveryArbiter())
    base = workflow_request(
        "wf-job",
        "inc-job",
        status=WorkflowStatus.RUNNING,
        fencing_token=1,
        completed_step_indexes=[0],
        official_steps=[
            workflow_step(STOP, node_ids=JOB_NODES, workload_ids=["training/job/j"]),
            workflow_step(
                RESET,
                node_ids=["node-a"],
                gpu_uuids=["GPU-a"],
                depends_on_step_indexes=[0],
            ),
            workflow_step(RESTORE, node_ids=["node-a"], depends_on_step_indexes=[1]),
            workflow_step(
                RESTART_JOB,
                node_ids=JOB_NODES,
                workload_ids=["training/job/j"],
                depends_on_step_indexes=[2],
            ),
        ],
    )
    node_b = workflow_request(
        "wf-b",
        "inc-job",
        fencing_token=1,
        official_steps=[
            workflow_step(STOP, node_ids=JOB_NODES, workload_ids=["training/job/j"]),
            workflow_step(
                RESET,
                node_ids=["node-b"],
                gpu_uuids=["GPU-b"],
                depends_on_step_indexes=[0],
            ),
            workflow_step(RESTORE, node_ids=["node-b"], depends_on_step_indexes=[1]),
            workflow_step(
                RESTART_JOB,
                node_ids=JOB_NODES,
                workload_ids=["training/job/j"],
                depends_on_step_indexes=[2],
            ),
        ],
    )
    return brancher.append_parallel_job_branch(base, node_b)


def _candidate(operation: WorkflowOperation, node_id: str) -> WorkflowRequest:
    return workflow_request(
        "wf-cand",
        "inc-job",
        fencing_token=1,
        official_steps=[
            workflow_step(WorkflowOperation.MARK_UNSCHEDULABLE, node_ids=[node_id]),
            workflow_step(operation, node_ids=[node_id], depends_on_step_indexes=[0]),
        ],
    )


def _by_operation(
    workflow: WorkflowRequest, operation: WorkflowOperation, node_id: str
) -> int:
    return next(
        index
        for index, step in enumerate(workflow.official_steps)
        if step.operation is operation and node_id in step.node_ids
    )


def test_isolating_one_node_keeps_the_job_restart_the_other_nodes_still_need():
    dag = _job_dag()
    brancher = DagBrancher(RecoveryArbiter())

    replaced = brancher.replace_parallel_job_branch(
        dag, _candidate(QUARANTINE, "node-b"), "node-b"
    )

    restore_b = _by_operation(dag, RESTORE, "node-b")
    restart = _by_operation(dag, RESTART_JOB, "node-a")
    assert (
        restore_b in replaced.superseded_step_indexes
    )  # the isolated node owes no release
    assert (
        restart not in replaced.superseded_step_indexes
    )  # node-a's job still restarts
    assert _by_operation(dag, RESTORE, "node-a") not in replaced.superseded_step_indexes


def test_isolating_the_only_node_of_a_job_retires_its_restart():
    brancher = DagBrancher(RecoveryArbiter())
    single = workflow_request(
        "wf-one",
        "inc-one",
        status=WorkflowStatus.RUNNING,
        fencing_token=1,
        dag_enabled=True,
        completed_step_indexes=[0],
        official_steps=[
            workflow_step(
                STOP,
                node_ids=["node-a"],
                workload_ids=["training/job/j"],
                branch_id="shared",
            ),
            workflow_step(
                RESET,
                node_ids=["node-a"],
                depends_on_step_indexes=[0],
                branch_id="branch:1:node-a",
            ),
            workflow_step(
                RESTORE,
                node_ids=["node-a"],
                depends_on_step_indexes=[1],
                branch_id="branch:1:node-a",
            ),
            workflow_step(
                RESTART_JOB,
                node_ids=["node-a"],
                workload_ids=["training/job/j"],
                depends_on_step_indexes=[2],
                branch_id="join",
            ),
        ],
    )

    replaced = brancher.replace_parallel_job_branch(
        single, _candidate(QUARANTINE, "node-a"), "node-a"
    )

    assert {2, 3} <= set(replaced.superseded_step_indexes)


def test_a_restart_that_already_ran_is_never_retired_by_a_later_isolation():
    dag = _job_dag()
    restart = _by_operation(dag, RESTART_JOB, "node-a")
    started = copy_model(
        dag,
        completed_step_indexes=[0, restart],
        step_executions=[workflow_step_execution(restart, RESTART_JOB)],
    )
    brancher = DagBrancher(RecoveryArbiter())

    replaced = brancher.replace_parallel_job_branch(
        copy_model(
            started,
            official_steps=[
                step.model_copy(update={"node_ids": ["node-b"]})
                if step.operation is RESTART_JOB
                else step
                for step in started.official_steps
            ],
        ),
        _candidate(QUARANTINE, "node-b"),
        "node-b",
    )

    assert restart not in replaced.superseded_step_indexes


def test_a_support_escalation_is_terminal_for_its_node():
    dag = _job_dag()
    brancher = DagBrancher(RecoveryArbiter())

    replaced = brancher.replace_parallel_job_branch(
        dag, _candidate(SUPPORT, "node-b"), "node-b"
    )

    assert _by_operation(dag, RESTORE, "node-b") in replaced.superseded_step_indexes
    assert (
        _by_operation(dag, RESTART_JOB, "node-a")
        not in replaced.superseded_step_indexes
    )


def test_quarantine_claims_the_isolation_budget():
    policy = RemediationBudgetPolicy()
    incident = fault_incident("inc-q", "event-q", cluster_id="cluster-a")
    steps = [
        workflow_step(WorkflowOperation.MARK_UNSCHEDULABLE, node_ids=["node-a"]),
        workflow_step(QUARANTINE, node_ids=["node-a"], depends_on_step_indexes=[0]),
    ]
    workflow = workflow_request("wf-q", "inc-q", official_steps=steps)

    claims = remediation_budget_claims(policy, workflow, incident, steps)

    assert claims["node:cluster-a:node-a"] == policy.node_limit
    assert claims["class:cluster-a:NODE_ISOLATION"] == policy.resource_class_limit
    # Plain containment (cordon alone) is still free.
    cordon_only = [
        workflow_step(WorkflowOperation.MARK_UNSCHEDULABLE, node_ids=["node-a"])
    ]
    assert remediation_budget_claims(policy, workflow, incident, cordon_only) == {}

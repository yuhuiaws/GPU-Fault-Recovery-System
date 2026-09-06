"""Preemption guards fail closed (F-C1 remainder).

The node-overlap gate between a predecessor and its preempting successor was
skipped whenever either side had no node-exclusive step, so a candidate that
touched no node at all could still preempt. Inherited containment looked
only at ``completed_operations``, so a STOP_WORKLOADS was inherited as done
while the predecessor's RESTART_WORKLOAD was already executing. The
dispatcher's one-per-incident pick fell back to the oldest row when the
incident pointer did not resolve, which could be the predecessor of a
flagged successor.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from gpu_fault.execution.config import WorkflowDispatcherConfig
from gpu_fault.execution.dispatcher import WorkflowDispatcher
from gpu_fault.models import WorkflowOperation, WorkflowStatus, WorkflowStepStatus
from gpu_fault.operation_registry import NODE_EXCLUSIVE_OPERATIONS
from gpu_fault.orchestration.arbitration import RecoveryArbiter
from gpu_fault.orchestration.dag_branching import DagBrancher
from gpu_fault.orchestration.workflow_merge import WorkflowMergeService
from tests._builders import (
    active_workflow_executor,
    build_store,
    workflow_request,
    workflow_step,
    workflow_step_execution,
)

STOP = WorkflowOperation.STOP_WORKLOADS
RESET = WorkflowOperation.RESET_GPU
REBOOT = WorkflowOperation.RESTART_NODE
RESTART_JOB = WorkflowOperation.RESTART_WORKLOAD
FREEZE = WorkflowOperation.FREEZE_EVIDENCE


def _service() -> WorkflowMergeService:
    return WorkflowMergeService(
        RecoveryArbiter(),
        DagBrancher(RecoveryArbiter()),
        preemption_enabled=True,
        workload_scoped_operations=set(),
        node_exclusive_operations=set(NODE_EXCLUSIVE_OPERATIONS),
        workflow_resource_claims_by_node=lambda _workflow: {},
    )


def test_a_candidate_that_holds_no_node_cannot_preempt():
    existing = workflow_request(
        "existing",
        "inc",
        status=WorkflowStatus.RUNNING,
        fencing_token=1,
        official_steps=[workflow_step(FREEZE, node_ids=["node-a"])],
    )
    candidate = workflow_request(
        "candidate",
        "inc",
        fencing_token=1,
        official_steps=[
            workflow_step(
                RESTART_JOB, node_ids=["node-a"], workload_ids=["training/job/j"]
            )
        ],
    )

    prepared = _service().prepare_preempting_successor(existing, candidate)

    assert prepared.preempt_predecessor is False, (
        "a workflow with no node-exclusive step must queue, not preempt"
    )


def test_a_stronger_node_action_still_preempts_a_predecessor_without_nodes():
    existing = workflow_request(
        "existing",
        "inc",
        status=WorkflowStatus.RUNNING,
        fencing_token=1,
        official_steps=[workflow_step(FREEZE, node_ids=["node-a"])],
    )
    candidate = workflow_request(
        "candidate",
        "inc",
        fencing_token=1,
        official_steps=[workflow_step(REBOOT, node_ids=["node-a"])],
    )

    assert (
        _service().prepare_preempting_successor(existing, candidate).preempt_predecessor
        is True
    )


def test_a_stop_is_not_inherited_while_the_predecessor_is_restarting_the_job():
    existing = workflow_request(
        "existing",
        "inc",
        status=WorkflowStatus.RUNNING,
        fencing_token=1,
        official_steps=[
            workflow_step(STOP, node_ids=["node-a"], workload_ids=["training/job/j"]),
            workflow_step(RESET, node_ids=["node-a"], gpu_uuids=["GPU-a"]),
            workflow_step(
                RESTART_JOB, node_ids=["node-a"], workload_ids=["training/job/j"]
            ),
        ],
        completed_step_indexes=[0, 1],
        completed_operations=[STOP, RESET],
        step_executions=[
            workflow_step_execution(0, STOP),
            workflow_step_execution(1, RESET),
            workflow_step_execution(2, RESTART_JOB, WorkflowStepStatus.WAITING),
        ],
    )
    candidate = workflow_request(
        "candidate",
        "inc",
        fencing_token=1,
        official_steps=[
            workflow_step(STOP, node_ids=["node-a"], workload_ids=["training/job/j"]),
            workflow_step(REBOOT, node_ids=["node-a"]),
        ],
    )

    prepared = _service().prepare_preempting_successor(existing, candidate)

    assert prepared.preempt_predecessor is True
    assert prepared.inherited_step_indexes == [], (
        "the job is being restarted; its stop is not a fact any more"
    )


def test_the_dispatcher_prefers_the_flagged_successor_when_the_pointer_is_dangling():
    store = build_store()
    dispatcher = WorkflowDispatcher(
        store,
        active_workflow_executor(store, [], frozenset()),
        WorkflowDispatcherConfig(enabled=True),
    )
    now = datetime.now(timezone.utc)
    predecessor = workflow_request(
        "wf-old",
        "inc-missing",
        official_steps=[workflow_step(RESET)],
        created_at=now - timedelta(minutes=5),
        updated_at=now - timedelta(minutes=5),
    )
    successor = workflow_request(
        "wf-new",
        "inc-missing",
        official_steps=[workflow_step(REBOOT)],
        predecessor_workflow_id="wf-old",
        preempt_predecessor=True,
        created_at=now,
        updated_at=now,
    )

    chosen = dispatcher._one_per_incident([predecessor, successor])

    assert [item.request_id for item in chosen] == ["wf-new"]

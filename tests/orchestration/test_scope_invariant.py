"""Invariant: after any merge, every (node, GPU) the fault names has a live step.

F-B6 (7). Whatever disposition the merge service picks, the workflow that
comes out must still hold an unresolved node-mutating step for each node and
GPU the candidate fault named -- otherwise the event was recorded and the
hardware left alone. Record-only verdicts are the explicit exception and are
covered by their own tests.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from gpu_fault.models import (
    WorkflowOperation,
    WorkflowRequest,
    WorkflowStatus,
    resolved_step_indexes,
)
from gpu_fault.operation_registry import (
    NODE_MUTATING_OPERATIONS,
    NODE_WIDE_RECOVERY_OPERATIONS,
)
from gpu_fault.orchestration.arbitration import RecoveryArbiter
from gpu_fault.orchestration.dag_branching import DagBrancher
from gpu_fault.orchestration.disposition import Disposition, DispositionApplier
from gpu_fault.orchestration.families.conflicts import NodeConflictService
from gpu_fault.orchestration.workflow_merge import WorkflowMergeService
from tests._builders import copy_model, fault_incident, workflow_request, workflow_step

NOW = datetime(2026, 9, 6, 10, 0, tzinfo=timezone.utc)
ALL_NODES = ["node-a", "node-b", "node-c"]


def _live_step_covers(
    workflow: WorkflowRequest, node_id: str, gpu_uuid: str | None
) -> bool:
    resolved = resolved_step_indexes(workflow)
    for index, step in enumerate(workflow.official_steps):
        if index in resolved or node_id not in step.node_ids:
            continue
        if step.operation in NODE_WIDE_RECOVERY_OPERATIONS:
            return True
        if step.operation not in NODE_MUTATING_OPERATIONS:
            continue
        if gpu_uuid is None:
            return True
        mapping = step.parameters.get("gpu_uuids_by_node")
        named = (
            set(mapping.get(node_id, []))
            if isinstance(mapping, dict)
            else set(step.gpu_uuids)
        )
        if gpu_uuid in named:
            return True
    return False


def _services() -> tuple[WorkflowMergeService, DispositionApplier]:
    arbiter = RecoveryArbiter()
    brancher = DagBrancher(arbiter)
    merger = WorkflowMergeService(
        arbiter,
        brancher,
        preemption_enabled=True,
        workload_scoped_operations=set(),
        node_exclusive_operations=set(),
        workflow_resource_claims_by_node=(
            NodeConflictService.workflow_resource_claims_by_node
        ),
    )
    applier = DispositionApplier(
        arbiter=arbiter,
        brancher=brancher,
        aggregation_deadlines=lambda now, _w: (
            now + timedelta(seconds=5),
            now + timedelta(seconds=30),
        ),
        prepare_preempting_successor=lambda _existing, successor: successor,
        preempt_parallel_job_branch=lambda existing, _c, _n: existing,
        workflow_preemption_enabled=True,
    )
    return merger, applier


def _node_workflow(
    request_id: str,
    node_id: str,
    gpus: list[str],
    operation=WorkflowOperation.RESET_GPU,
    **updates,
):
    return workflow_request(
        request_id,
        "inc-a",
        official_steps=[
            workflow_step(WorkflowOperation.MARK_UNSCHEDULABLE, node_ids=[node_id]),
            workflow_step(
                operation,
                node_ids=[node_id],
                gpu_uuids=gpus,
                parameters={"gpu_uuids_by_node": {node_id: list(gpus)}} if gpus else {},
                depends_on_step_indexes=[0],
            ),
            workflow_step(
                WorkflowOperation.VALIDATE_GPU,
                node_ids=[node_id],
                gpu_uuids=gpus,
                depends_on_step_indexes=[1],
            ),
            workflow_step(
                WorkflowOperation.RESTORE_SCHEDULING,
                node_ids=[node_id],
                depends_on_step_indexes=[2],
            ),
        ],
        created_at=NOW - timedelta(minutes=1),
        updated_at=NOW - timedelta(minutes=1),
        **updates,
    )


def _job_workflow(request_id: str, node_id: str, gpus: list[str], **updates):
    return workflow_request(
        request_id,
        "incident-dag",
        fencing_token=1,
        official_steps=[
            workflow_step(
                WorkflowOperation.STOP_WORKLOADS,
                node_ids=ALL_NODES,
                workload_ids=["training/job/job-a"],
            ),
            workflow_step(
                WorkflowOperation.RESET_GPU,
                node_ids=[node_id],
                gpu_uuids=gpus,
                depends_on_step_indexes=[0],
            ),
            workflow_step(
                WorkflowOperation.RESTORE_SCHEDULING,
                node_ids=[node_id],
                depends_on_step_indexes=[1],
            ),
            workflow_step(
                WorkflowOperation.RESTART_WORKLOAD,
                node_ids=ALL_NODES,
                workload_ids=["training/job/job-a"],
                depends_on_step_indexes=[2],
            ),
        ],
        created_at=NOW - timedelta(minutes=1),
        updated_at=NOW - timedelta(minutes=1),
        **updates,
    )


def _running_dag() -> WorkflowRequest:
    brancher = DagBrancher(RecoveryArbiter())
    running = copy_model(
        _job_workflow("wf-job", "node-a", ["GPU-a"]),
        status=WorkflowStatus.RUNNING,
        completed_step_indexes=[0],
    )
    return brancher.append_parallel_job_branch(
        running, _job_workflow("wf-b", "node-b", ["GPU-b1"])
    )


CASES = {
    # same node, second GPU, reset already running for the first
    "widen_in_place": (
        lambda: copy_model(
            _node_workflow("wf-a", "node-a", ["GPU-1"]),
            status=WorkflowStatus.RUNNING,
            completed_step_indexes=[0],
        ),
        lambda: _node_workflow("wf-new", "node-a", ["GPU-2"]),
        "node-a",
        {"GPU-2"},
        Disposition.WIDEN_IN_PLACE,
    ),
    # a stronger plan takes over a pending one
    "replace_in_place": (
        lambda: _node_workflow("wf-a", "node-a", ["GPU-1"]),
        lambda: _node_workflow(
            "wf-new", "node-a", [], operation=WorkflowOperation.RESTART_NODE
        ),
        "node-a",
        set(),
        Disposition.REPLACE_IN_PLACE,
    ),
    # a third node joins the job workflow
    "parallel_branch": (
        _running_dag,
        lambda: _job_workflow("wf-new", "node-c", ["GPU-c1"]),
        "node-c",
        {"GPU-c1"},
        Disposition.PARALLEL_BRANCH,
    ),
    # a second GPU on a node whose branch has not started
    "widen_branch": (
        _running_dag,
        lambda: _job_workflow("wf-new", "node-b", ["GPU-b2"]),
        "node-b",
        {"GPU-b2"},
        Disposition.WIDEN_BRANCH,
    ),
}


@pytest.mark.parametrize("name", sorted(CASES))
def test_every_named_node_and_gpu_keeps_a_live_step_after_the_merge(name):
    build_existing, build_candidate, node_id, gpu_uuids, expected = CASES[name]
    existing, candidate = build_existing(), build_candidate()
    merger, applier = _services()

    verdict = merger.disposition(existing, candidate, node_id, set(gpu_uuids))
    workflow, _ = applier.apply(
        verdict,
        node_id=node_id,
        candidate=fault_incident("inc-new", "event-new"),
        candidate_workflow=candidate,
        existing_incident=fault_incident("inc-a", "event-a"),
        existing_workflow=existing,
        gpu_uuids=set(gpu_uuids),
        mutable=existing.status is WorkflowStatus.PENDING,
        now=NOW,
    )

    assert verdict is expected
    for gpu_uuid in gpu_uuids or {None}:
        assert _live_step_covers(workflow, node_id, gpu_uuid), (name, node_id, gpu_uuid)
    # Nothing the existing workflow already covered was lost either.
    for step in existing.official_steps:
        if (
            step.operation in NODE_MUTATING_OPERATIONS
            and step.operation is not WorkflowOperation.STOP_WORKLOADS
        ):
            for node in step.node_ids:
                for gpu in step.gpu_uuids or [None]:
                    assert _live_step_covers(workflow, node, gpu), (name, node, gpu)

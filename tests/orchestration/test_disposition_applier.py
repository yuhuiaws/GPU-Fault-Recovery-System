"""One disposition vocabulary, one application of it, across every family.

F-B5 (docs/review/FINAL-建议汇总.md). The merge service returned nine bare
strings and three families re-implemented how to apply them, drifting apart:
only the grouped-faults family let a stronger QUEUE_BRANCH_SUCCESSOR preempt
the weaker branch; REPLACE_IN_PLACE dropped the predecessor pointer that the
serialization chain relies on; WIDEN_IN_PLACE was a no-op in all three, so a
second GPU on the same node was never added to the pending reset; and an
unknown value silently fell through to "queue a successor". The vocabulary is
now an enum, the application is one shared object, and the WIDEN_BRANCH gate
leaves finished and in-flight steps alone when widening a branch.
"""

from __future__ import annotations

import dataclasses
from datetime import datetime, timedelta, timezone

import pytest

from gpu_fault.models import (
    WorkflowOperation,
    WorkflowRequest,
    WorkflowStatus,
    WorkflowStepStatus,
)
from gpu_fault.operation_registry import (
    NODE_ACTION_SCOPE_OPERATIONS,
    WORKLOAD_SCOPED_OPERATIONS,
)
from gpu_fault.orchestration.arbitration import RecoveryArbiter
from gpu_fault.orchestration.coordinator import IncidentOrchestrator
from gpu_fault.orchestration.dag_branching import DagBrancher
from gpu_fault.orchestration.disposition import Disposition, DispositionApplier
from gpu_fault.orchestration.families.conflicts import NodeConflictService
from gpu_fault.orchestration.families.faults import NodeScopedFaultCallbacks
from gpu_fault.orchestration.families.grouped_faults import GroupedFaultCallbacks
from gpu_fault.orchestration.families.grouped_health import GroupedHealthCallbacks
from gpu_fault.orchestration.workflow_merge import WorkflowMergeService
from tests._builders import (
    copy_model,
    fault_incident,
    workflow_request,
    workflow_step,
    workflow_step_execution,
)

NOW = datetime(2026, 9, 6, 9, 0, tzinfo=timezone.utc)
ALL_NODES = ["node-a", "node-b", "node-c"]
RESET = WorkflowOperation.RESET_GPU


def _merger(brancher: DagBrancher | None = None) -> WorkflowMergeService:
    return WorkflowMergeService(
        RecoveryArbiter(),
        brancher or DagBrancher(RecoveryArbiter()),
        preemption_enabled=True,
        workload_scoped_operations=set(),
        node_exclusive_operations=set(),
        workflow_resource_claims_by_node=(
            NodeConflictService.workflow_resource_claims_by_node
        ),
    )


def _applier(preempt=None, *, preemption_enabled: bool = True) -> DispositionApplier:
    arbiter = RecoveryArbiter()
    return DispositionApplier(
        arbiter=arbiter,
        brancher=DagBrancher(arbiter),
        aggregation_deadlines=lambda now, _workflow: (
            now + timedelta(seconds=5),
            now + timedelta(seconds=30),
        ),
        prepare_preempting_successor=lambda _existing, successor: successor,
        preempt_parallel_job_branch=preempt
        or (lambda existing, _candidate, _node_id: existing),
        workflow_preemption_enabled=preemption_enabled,
    )


def _reset_workflow(
    request_id: str,
    node_id: str,
    gpu_uuids: list[str],
    *,
    completed: list[int] | None = None,
    status: WorkflowStatus = WorkflowStatus.RUNNING,
    **updates,
) -> WorkflowRequest:
    return workflow_request(
        request_id,
        "inc-a",
        status=status,
        official_steps=[
            workflow_step(
                WorkflowOperation.MARK_UNSCHEDULABLE,
                node_ids=[node_id],
                gpu_uuids=gpu_uuids,
            ),
            workflow_step(
                RESET,
                node_ids=[node_id],
                gpu_uuids=gpu_uuids,
                parameters={"gpu_uuids_by_node": {node_id: list(gpu_uuids)}},
                depends_on_step_indexes=[0],
            ),
            workflow_step(
                WorkflowOperation.VALIDATE_GPU,
                node_ids=[node_id],
                gpu_uuids=gpu_uuids,
                depends_on_step_indexes=[1],
            ),
            workflow_step(
                WorkflowOperation.RESTORE_SCHEDULING,
                node_ids=[node_id],
                depends_on_step_indexes=[2],
            ),
        ],
        completed_step_indexes=completed or [],
        created_at=NOW - timedelta(minutes=2),
        updated_at=NOW - timedelta(minutes=2),
        **updates,
    )


def _job_workflow(
    node_id: str, request_id: str, gpu_uuids: list[str] | None = None, **updates
) -> WorkflowRequest:
    gpus = gpu_uuids or []
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
                WorkflowOperation.FREEZE_EVIDENCE, node_ids=[node_id], gpu_uuids=gpus
            ),
            workflow_step(
                RESET, node_ids=[node_id], gpu_uuids=gpus, depends_on_step_indexes=[1]
            ),
            workflow_step(
                WorkflowOperation.RESTORE_SCHEDULING,
                node_ids=[node_id],
                depends_on_step_indexes=[2],
            ),
            workflow_step(
                WorkflowOperation.RESTART_WORKLOAD,
                node_ids=ALL_NODES,
                workload_ids=["training/job/job-a"],
                depends_on_step_indexes=[3],
            ),
        ],
        created_at=NOW - timedelta(minutes=1),
        updated_at=NOW - timedelta(minutes=1),
        **updates,
    )


# ------------------------------------------------------------ vocabulary


def test_the_merge_service_speaks_the_enum_and_the_enum_is_closed():
    assert {member.value for member in Disposition} == {
        "ABSORB",
        "ABSORB_RECORD_ONLY",
        "WIDEN_IN_PLACE",
        "WIDEN_BRANCH",
        "REPLACE_IN_PLACE",
        "REPLACE_BRANCH",
        "PARALLEL_BRANCH",
        "QUEUE_BRANCH_SUCCESSOR",
        "QUEUE_SUCCESSOR",
    }
    existing = _reset_workflow(
        "wf-a", "node-a", ["GPU-1"], status=WorkflowStatus.PENDING
    )
    candidate = _reset_workflow(
        "wf-b", "node-a", ["GPU-1"], status=WorkflowStatus.PENDING
    )

    verdict = _merger().disposition(existing, candidate, "node-a", {"GPU-1"})

    assert isinstance(verdict, Disposition), (
        "expected isinstance(verdict, Disposition) to be true"
    )
    # String equality is preserved for every caller that still compares text.
    assert verdict == "ABSORB"


def test_an_unknown_disposition_is_rejected_instead_of_queuing_a_successor():
    existing = _reset_workflow("wf-a", "node-a", ["GPU-1"])
    with pytest.raises(ValueError):
        _applier().apply(
            "SOMETHING_NEW",
            node_id="node-a",
            candidate=fault_incident("inc-b", "event-b"),
            candidate_workflow=_reset_workflow("wf-b", "node-a", ["GPU-2"]),
            existing_incident=fault_incident("inc-a", "event-a"),
            existing_workflow=existing,
            gpu_uuids={"GPU-2"},
            mutable=False,
            now=NOW,
        )


# ------------------------------------------------------------ WIDEN_IN_PLACE


def test_widen_in_place_adds_the_new_gpu_to_pending_steps_and_leaves_finished_ones():
    existing = _reset_workflow("wf-a", "node-a", ["GPU-1"], completed=[0])
    candidate = _reset_workflow(
        "wf-b", "node-a", ["GPU-2"], status=WorkflowStatus.PENDING
    )
    existing_incident = fault_incident("inc-a", "event-a")

    workflow, winner = _applier().apply(
        Disposition.WIDEN_IN_PLACE,
        node_id="node-a",
        candidate=fault_incident("inc-b", "event-b"),
        candidate_workflow=candidate,
        existing_incident=existing_incident,
        existing_workflow=existing,
        gpu_uuids={"GPU-2"},
        mutable=False,
        now=NOW,
    )

    assert winner is existing_incident
    assert workflow.request_id == "wf-a"
    # The completed cordon is history; it is not rewritten.
    assert workflow.official_steps[0].gpu_uuids == ["GPU-1"]
    # The pending reset and its validation now cover both GPUs.
    assert workflow.official_steps[1].gpu_uuids == ["GPU-1", "GPU-2"]
    assert workflow.official_steps[1].parameters["gpu_uuids_by_node"] == {
        "node-a": ["GPU-1", "GPU-2"]
    }
    assert workflow.official_steps[2].gpu_uuids == ["GPU-1", "GPU-2"]
    assert workflow.updated_at == NOW


def test_widen_in_place_does_not_touch_a_step_the_agent_is_already_running():
    existing = _reset_workflow(
        "wf-a",
        "node-a",
        ["GPU-1"],
        completed=[0],
        step_executions=[workflow_step_execution(1, RESET, WorkflowStepStatus.WAITING)],
    )
    candidate = _reset_workflow(
        "wf-b", "node-a", ["GPU-2"], status=WorkflowStatus.PENDING
    )

    workflow, _ = _applier().apply(
        Disposition.WIDEN_IN_PLACE,
        node_id="node-a",
        candidate=fault_incident("inc-b", "event-b"),
        candidate_workflow=candidate,
        existing_incident=fault_incident("inc-a", "event-a"),
        existing_workflow=existing,
        gpu_uuids={"GPU-2"},
        mutable=False,
        now=NOW,
    )

    assert workflow.official_steps[1].gpu_uuids == ["GPU-1"]
    assert workflow.official_steps[2].gpu_uuids == ["GPU-1", "GPU-2"]


# ------------------------------------------------------------ REPLACE_IN_PLACE


def test_replace_in_place_keeps_the_predecessor_pointer_and_the_lifetime():
    lifetime = NOW + timedelta(minutes=40)
    existing = _reset_workflow(
        "wf-a",
        "node-a",
        ["GPU-1"],
        status=WorkflowStatus.PENDING,
        predecessor_workflow_id="wf-0",
        lifetime_deadline_at=lifetime,
    )
    candidate = workflow_request(
        "wf-b",
        "inc-b",
        official_steps=[
            workflow_step(WorkflowOperation.RESTART_NODE, node_ids=["node-a"])
        ],
    )
    candidate_incident = fault_incident("inc-b", "event-b")

    workflow, winner = _applier().apply(
        Disposition.REPLACE_IN_PLACE,
        node_id="node-a",
        candidate=candidate_incident,
        candidate_workflow=candidate,
        existing_incident=fault_incident("inc-a", "event-a"),
        existing_workflow=existing,
        gpu_uuids={"GPU-1"},
        mutable=True,
        now=NOW,
    )

    assert winner is candidate_incident
    assert workflow.request_id == "wf-a"
    assert workflow.incident_id == "inc-a"
    assert workflow.fencing_token == existing.fencing_token + 1
    assert workflow.predecessor_workflow_id == "wf-0"
    assert workflow.lifetime_deadline_at == lifetime
    assert workflow.official_steps[0].operation is WorkflowOperation.RESTART_NODE


# ------------------------------------------------------------ QUEUE_BRANCH_SUCCESSOR


def test_a_stronger_branch_successor_preempts_in_the_shared_applier():
    calls: list[tuple[str, str, str]] = []

    def preempt(existing, candidate, node_id):
        calls.append((existing.request_id, candidate.request_id, node_id))
        return existing

    brancher = DagBrancher(RecoveryArbiter())
    running = copy_model(
        _job_workflow("node-a", "wf-job"),
        status=WorkflowStatus.RUNNING,
        completed_step_indexes=[0],
    )
    dag = brancher.append_parallel_job_branch(running, _job_workflow("node-b", "wf-b"))
    stronger = workflow_request(
        "wf-c",
        "incident-dag",
        fencing_token=1,
        official_steps=[
            workflow_step(WorkflowOperation.RESTART_NODE, node_ids=["node-b"]),
            workflow_step(
                WorkflowOperation.RESTORE_SCHEDULING,
                node_ids=["node-b"],
                depends_on_step_indexes=[0],
            ),
        ],
    )

    _applier(preempt).apply(
        Disposition.QUEUE_BRANCH_SUCCESSOR,
        node_id="node-b",
        candidate=fault_incident("incident-dag", "event-c"),
        candidate_workflow=stronger,
        existing_incident=fault_incident("incident-dag", "event-a"),
        existing_workflow=dag,
        gpu_uuids=set(),
        mutable=False,
        now=NOW,
    )

    assert calls == [("wf-job", "wf-c", "node-b")]


def test_every_family_receives_the_preemption_callback():
    for callbacks in (
        NodeScopedFaultCallbacks,
        GroupedFaultCallbacks,
        GroupedHealthCallbacks,
    ):
        names = {field.name for field in dataclasses.fields(callbacks)}
        assert "preempt_parallel_job_branch" in names, callbacks.__name__


# ------------------------------------------------------------ WIDEN_BRANCH


def test_widen_branch_widens_pending_steps_and_leaves_the_started_evidence_alone():
    brancher = DagBrancher(RecoveryArbiter())
    running = copy_model(
        _job_workflow("node-a", "wf-job", ["GPU-a"]),
        status=WorkflowStatus.RUNNING,
        completed_step_indexes=[0],
    )
    dag = brancher.append_parallel_job_branch(
        running, _job_workflow("node-b", "wf-b", ["GPU-1"])
    )
    indexes = brancher.node_branch_step_indexes(dag, "node-b")
    by_operation = {dag.official_steps[index].operation: index for index in indexes}
    freeze_index = by_operation[WorkflowOperation.FREEZE_EVIDENCE]
    dag = copy_model(
        dag,
        completed_step_indexes=[0, freeze_index],
        step_executions=[
            workflow_step_execution(freeze_index, WorkflowOperation.FREEZE_EVIDENCE)
        ],
    )
    candidate = _job_workflow(
        "node-b", "wf-c", ["GPU-2"], status=WorkflowStatus.PENDING
    )

    verdict = _merger(brancher).disposition(dag, candidate, "node-b", {"GPU-2"})
    widened, _ = _applier().apply(
        verdict,
        node_id="node-b",
        candidate=fault_incident("incident-dag", "event-c"),
        candidate_workflow=candidate,
        existing_incident=fault_incident("incident-dag", "event-a"),
        existing_workflow=dag,
        gpu_uuids={"GPU-2"},
        mutable=False,
        now=NOW,
    )

    assert verdict is Disposition.WIDEN_BRANCH
    # The evidence already captured for GPU-1 is not rewritten as if it
    # had covered GPU-2; the pending reset is.
    assert widened.official_steps[freeze_index].gpu_uuids == ["GPU-1"]
    assert widened.official_steps[by_operation[RESET]].gpu_uuids == ["GPU-1", "GPU-2"]
    # The other node's branch is untouched.
    assert widened.official_steps[2].gpu_uuids == ["GPU-a"]


# ------------------------------------------------------------ F-B6(1) scope widening


def test_widen_node_action_scope_unions_gpus_and_skips_finished_or_running_steps():
    existing = _reset_workflow(
        "wf-a",
        "node-a",
        ["GPU-1"],
        completed=[0],
        step_executions=[workflow_step_execution(1, RESET, WorkflowStepStatus.WAITING)],
    )
    fake_self = type(
        "Self",
        (),
        {
            "_WORKLOAD_SCOPED_OPERATIONS": WORKLOAD_SCOPED_OPERATIONS,
            "_NODE_ACTION_OPERATIONS": NODE_ACTION_SCOPE_OPERATIONS,
        },
    )()

    widened = IncidentOrchestrator._widen_node_action_scope(
        fake_self, existing, {"node-a": ["GPU-2"], "node-b": ["GPU-3"]}
    )

    # Finished and in-flight steps are left exactly as they were.
    assert widened.official_steps[0] == existing.official_steps[0]
    assert widened.official_steps[1] == existing.official_steps[1]
    # Pending steps cover the union, never a rewrite that drops GPU-1.
    assert widened.official_steps[2].node_ids == ["node-a", "node-b"]
    assert widened.official_steps[2].gpu_uuids == ["GPU-1", "GPU-2", "GPU-3"]
    assert widened.official_steps[3].node_ids == ["node-a", "node-b"]

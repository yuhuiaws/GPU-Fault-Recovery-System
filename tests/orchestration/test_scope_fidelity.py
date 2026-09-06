"""Coverage and widening answer for the steps that will actually run.

F-B6 (docs/review/FINAL-建议汇总.md). "Is this fault already covered?" used to
say yes when a node-wide step existed anywhere in the workflow, when the GPU
was only ever named by an evidence step, when the covering step had already
been superseded, or when the event carried no GPU identity at all -- and each
yes absorbed a fault that nobody then repaired. Widening dropped nodes that
had no GPU, so a host-level fault merged into a GPU workflow never entered a
single step. A terminal branch superseded every RESTORE_SCHEDULING that so
much as touched one of its nodes, leaving the other node cordoned.
"""

from __future__ import annotations

from gpu_fault.models import WorkflowOperation, WorkflowRequest, WorkflowStatus
from gpu_fault.operation_registry import (
    NODE_ACTION_SCOPE_OPERATIONS,
    WORKLOAD_SCOPED_OPERATIONS,
)
from gpu_fault.orchestration.arbitration import RecoveryArbiter
from gpu_fault.orchestration.coordinator import IncidentOrchestrator
from gpu_fault.orchestration.dag_branching import DagBrancher
from tests._builders import workflow_request, workflow_step

RESET = WorkflowOperation.RESET_GPU


def _workflow(*steps, **updates) -> WorkflowRequest:
    return workflow_request(
        "wf-a",
        "inc-a",
        status=WorkflowStatus.RUNNING,
        official_steps=list(steps),
        **updates,
    )


def _candidate(operation: WorkflowOperation, gpu_uuids: list[str]) -> WorkflowRequest:
    return workflow_request(
        "wf-b",
        "inc-b",
        official_steps=[
            workflow_step(operation, node_ids=["node-a"], gpu_uuids=gpu_uuids)
        ],
    )


# ------------------------------------------------------------ coverage


def test_a_gpu_fault_with_an_unresolved_gpu_is_not_covered_by_a_reset_of_another_gpu():
    existing = _workflow(workflow_step(RESET, node_ids=["node-a"], gpu_uuids=["GPU-1"]))
    arbiter = RecoveryArbiter()

    unresolved = arbiter.workflow_covers_fault_scope(
        existing, "node-a", set(), candidate=_candidate(RESET, [])
    )
    node_level = arbiter.workflow_covers_fault_scope(
        existing,
        "node-a",
        set(),
        candidate=_candidate(WorkflowOperation.RESTART_NODE, []),
    )

    assert unresolved is False
    assert node_level is True


def test_evidence_only_steps_do_not_count_as_gpu_coverage():
    existing = _workflow(
        workflow_step(
            WorkflowOperation.FREEZE_EVIDENCE,
            node_ids=["node-a"],
            gpu_uuids=["GPU-1", "GPU-2"],
        ),
        workflow_step(
            RESET, node_ids=["node-a"], gpu_uuids=["GPU-1"], depends_on_step_indexes=[0]
        ),
    )

    assert RecoveryArbiter().workflow_covers_fault_scope(
        existing, "node-a", {"GPU-1"}
    ), (
        'expected RecoveryArbiter().workflow_covers_fault_scope(existing, "node-a", {"GPU-1"}) to be true'
    )
    assert not RecoveryArbiter().workflow_covers_fault_scope(
        existing, "node-a", {"GPU-2"}
    ), (
        'expected RecoveryArbiter().workflow_covers_fault_scope(existing, "node-a", {"GPU-2"}) to be false'
    )


def test_a_superseded_reset_no_longer_covers_its_gpu():
    existing = _workflow(
        workflow_step(RESET, node_ids=["node-a"], gpu_uuids=["GPU-2"]),
        workflow_step(RESET, node_ids=["node-a"], gpu_uuids=["GPU-1"]),
        superseded_step_indexes=[0],
    )

    assert not RecoveryArbiter().workflow_covers_fault_scope(
        existing, "node-a", {"GPU-2"}
    ), (
        'expected RecoveryArbiter().workflow_covers_fault_scope(existing, "node-a", {"GPU-2"}) to be false'
    )


def test_a_node_wide_step_covers_only_the_node_it_names():
    existing = _workflow(
        workflow_step(WorkflowOperation.RESTART_NODE, node_ids=["node-b"]),
        workflow_step(RESET, node_ids=["node-a"], gpu_uuids=["GPU-1"]),
    )
    arbiter = RecoveryArbiter()

    assert arbiter.workflow_covers_fault_scope(existing, "node-b", {"GPU-9"}), (
        'expected arbiter.workflow_covers_fault_scope(existing, "node-b", {"GPU-9"}) to be true'
    )
    assert not arbiter.workflow_covers_fault_scope(existing, "node-a", {"GPU-2"}), (
        'expected arbiter.workflow_covers_fault_scope(existing, "node-a", {"GPU-2"}) to be false'
    )


def test_the_branch_variant_applies_the_same_rules():
    brancher = DagBrancher(RecoveryArbiter())
    existing = _workflow(
        workflow_step(
            WorkflowOperation.FREEZE_EVIDENCE,
            node_ids=["node-a"],
            gpu_uuids=["GPU-1", "GPU-2"],
            branch_id="branch:1:node-a",
        ),
        workflow_step(
            RESET,
            node_ids=["node-a"],
            gpu_uuids=["GPU-1"],
            branch_id="branch:1:node-a",
            depends_on_step_indexes=[0],
        ),
    )

    assert brancher.node_branch_covers_fault_scope(
        existing, [0, 1], "node-a", {"GPU-1"}
    ), (
        'expected brancher.node_branch_covers_fault_scope(existing, [0, 1], "node-a", {"GPU-1"}) to be true'
    )
    assert not brancher.node_branch_covers_fault_scope(
        existing, [0, 1], "node-a", {"GPU-2"}
    ), (
        'expected brancher.node_branch_covers_fault_scope(existing, [0, 1], "node-a", {"GPU-2"}) to be false'
    )
    assert not brancher.node_branch_covers_fault_scope(
        existing, [0, 1], "node-a", set(), candidate=_candidate(RESET, [])
    ), (
        'expected brancher.node_branch_covers_fault_scope( existing, [0, 1], "node-a", set(), candidate=_candidate(RESET, []) ) to be false'
    )


# ------------------------------------------------------------ widening


def test_a_node_without_gpus_is_kept_in_scope_and_widened_onto_host_steps_only():
    existing = _workflow(
        workflow_step(WorkflowOperation.MARK_UNSCHEDULABLE, node_ids=["node-a"]),
        workflow_step(
            RESET,
            node_ids=["node-a"],
            gpu_uuids=["GPU-1"],
            parameters={"gpu_uuids_by_node": {"node-a": ["GPU-1"]}},
            depends_on_step_indexes=[0],
        ),
        workflow_step(
            WorkflowOperation.RESTORE_SCHEDULING,
            node_ids=["node-a"],
            depends_on_step_indexes=[1],
        ),
    )
    scope = RecoveryArbiter().merged_gpu_scope(existing, "node-b", set())
    fake_self = type(
        "Self",
        (),
        {
            "_WORKLOAD_SCOPED_OPERATIONS": WORKLOAD_SCOPED_OPERATIONS,
            "_NODE_ACTION_OPERATIONS": NODE_ACTION_SCOPE_OPERATIONS,
        },
    )()

    widened = IncidentOrchestrator._widen_node_action_scope(fake_self, existing, scope)

    assert scope == {"node-a": ["GPU-1"], "node-b": []}
    # Host-level steps take the new node ...
    assert widened.official_steps[0].node_ids == ["node-a", "node-b"]
    assert widened.official_steps[2].node_ids == ["node-a", "node-b"]
    # ... the GPU action does not: the agent refuses a node without GPUs.
    assert widened.official_steps[1].node_ids == ["node-a"]
    assert widened.official_steps[1].parameters["gpu_uuids_by_node"] == {
        "node-a": ["GPU-1"]
    }


# ------------------------------------------------------------ terminal branch


def test_a_terminal_branch_only_supersedes_the_restore_it_wholly_owns():
    steps = [
        workflow_step(
            WorkflowOperation.RESTORE_SCHEDULING, node_ids=["node-a", "node-b"]
        ),
        workflow_step(WorkflowOperation.RESTORE_SCHEDULING, node_ids=["node-b"]),
        workflow_step(
            WorkflowOperation.RESTART_WORKLOAD, node_ids=["node-a", "node-b"]
        ),
    ]
    existing = _workflow(*steps)

    superseded = DagBrancher._terminal_superseded(
        existing, steps, ["node-b"], frozenset(), True
    )

    # Only node-b's own release goes; the shared release and the job restart
    # that node-a still needs stay (F-B6 (5), F-C5).
    assert superseded == {1}

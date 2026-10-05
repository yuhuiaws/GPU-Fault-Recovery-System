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


def _widen(existing: WorkflowRequest, scope: dict[str, list[str]]) -> WorkflowRequest:
    """The orchestrator's widening with only the class attributes it reads."""

    fake_self = type(
        "Self",
        (),
        {
            "_WORKLOAD_SCOPED_OPERATIONS": WORKLOAD_SCOPED_OPERATIONS,
            "_NODE_ACTION_OPERATIONS": NODE_ACTION_SCOPE_OPERATIONS,
        },
    )()
    return IncidentOrchestrator._widen_node_action_scope(fake_self, existing, scope)


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

    widened = _widen(existing, scope)

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


# ------------------------------------------- widening keeps workload nodes out of isolation
# A merge widens isolation to the faulted nodes only: the job's STOP_WORKLOADS
# participants (healthy siblings that only host the workload) never enter a
# workflow's MARK_UNSCHEDULABLE / QUARANTINE (asked live, GF-REGIONAL-COLLECT-016).

FAULT_NODE = "node-a"
SECOND_FAULT = "node-b"
HEALTHY_SIBLING = "node-c"
JOB_NODES = [FAULT_NODE, SECOND_FAULT, HEALTHY_SIBLING]
JOB = "training/job/c016"


def _reset_workflow() -> WorkflowRequest:
    one = [FAULT_NODE]
    return workflow_request(
        "wf-reset",
        "inc-reset",
        status=WorkflowStatus.PENDING,
        official_steps=[
            workflow_step(WorkflowOperation.FREEZE_EVIDENCE, node_ids=one),
            workflow_step(WorkflowOperation.MARK_UNSCHEDULABLE, node_ids=one),
            workflow_step(
                WorkflowOperation.STOP_WORKLOADS, node_ids=JOB_NODES, workload_ids=[JOB]
            ),
            workflow_step(WorkflowOperation.QUIESCE_GPU_SERVICES, node_ids=one),
            workflow_step(
                WorkflowOperation.RESET_GPU,
                node_ids=one,
                gpu_uuids=["GPU-a"],
                parameters={"gpu_uuids_by_node": {FAULT_NODE: ["GPU-a"]}},
            ),
            workflow_step(WorkflowOperation.RESTORE_GPU_SERVICES, node_ids=one),
            workflow_step(
                WorkflowOperation.VALIDATE_GPU, node_ids=one, gpu_uuids=["GPU-a"]
            ),
            workflow_step(WorkflowOperation.RESTORE_SCHEDULING, node_ids=one),
            workflow_step(
                WorkflowOperation.RESTART_WORKLOAD,
                node_ids=JOB_NODES,
                workload_ids=[JOB],
            ),
        ],
    )


def _by_operation(workflow: WorkflowRequest) -> dict[WorkflowOperation, list[str]]:
    return {step.operation: list(step.node_ids) for step in workflow.official_steps}


def test_the_merged_scope_names_faulted_nodes_never_workload_participants() -> None:
    existing = _reset_workflow()

    scope = RecoveryArbiter().merged_gpu_scope(existing, SECOND_FAULT, {"GPU-b"})

    assert scope == {FAULT_NODE: ["GPU-a"], SECOND_FAULT: ["GPU-b"]}
    assert HEALTHY_SIBLING not in scope, (
        "a node that only hosts the stopped job is not a faulted node"
    )


def test_widening_isolates_the_second_fault_and_leaves_the_sibling_alone() -> None:
    existing = _reset_workflow()
    scope = RecoveryArbiter().merged_gpu_scope(existing, SECOND_FAULT, {"GPU-b"})

    nodes = _by_operation(_widen(existing, scope))

    faulted = [FAULT_NODE, SECOND_FAULT]
    assert nodes[WorkflowOperation.MARK_UNSCHEDULABLE] == faulted
    assert nodes[WorkflowOperation.RESTORE_SCHEDULING] == faulted
    assert nodes[WorkflowOperation.VALIDATE_GPU] == faulted
    assert nodes[WorkflowOperation.RESET_GPU] == faulted, (
        "GPU actions widen to the faulted nodes that carry GPUs"
    )
    assert nodes[WorkflowOperation.STOP_WORKLOADS] == JOB_NODES
    assert nodes[WorkflowOperation.RESTART_WORKLOAD] == JOB_NODES
    for operation, node_ids in nodes.items():
        if operation not in WORKLOAD_SCOPED_OPERATIONS:
            assert HEALTHY_SIBLING not in node_ids, (operation, node_ids)


def test_a_sibling_fault_merged_without_gpus_joins_host_steps_only() -> None:
    """A node-health fault with no GPU identity on a job sibling: the sibling
    is cordoned and validated, the GPU reset stays on the node whose GPU it is."""

    existing = _reset_workflow()
    scope = RecoveryArbiter().merged_gpu_scope(existing, SECOND_FAULT, set())

    nodes = _by_operation(_widen(existing, scope))

    assert nodes[WorkflowOperation.MARK_UNSCHEDULABLE] == [FAULT_NODE, SECOND_FAULT]
    assert nodes[WorkflowOperation.RESET_GPU] == [FAULT_NODE]
    assert nodes[WorkflowOperation.STOP_WORKLOADS] == JOB_NODES
    assert HEALTHY_SIBLING not in nodes[WorkflowOperation.MARK_UNSCHEDULABLE]

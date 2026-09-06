"""RF-7: one identity for a node branch, whichever field spells it.

``branch_escalation_counts`` is keyed by node id and ``exhausted_branch_ids``
holds branch ids, so nothing could join the two without guessing the branch
id's shape. The shape is minted in one place (``dag_branching.mint_branch_id``)
and parsed by its inverse next to it; ``exhausted_node_ids`` derives node ids
from the branch ids, and every BRANCH_ESCALATION event carries both.
"""

from __future__ import annotations

import pytest

from gpu_fault.execution.branch_escalation import BranchEscalator, exhausted_node_ids
from gpu_fault.models import (
    WorkflowEventCode,
    WorkflowEventKind,
    WorkflowOperation,
    WorkflowStepStatus,
)
from gpu_fault.orchestration.arbitration import RecoveryArbiter
from gpu_fault.orchestration.dag_branching import (
    DagBrancher,
    branch_node_id,
    branch_node_ids,
    mint_branch_id,
)
from tests._builders import (
    copy_model,
    workflow_request,
    workflow_step,
    workflow_step_execution,
)

STOP = WorkflowOperation.STOP_WORKLOADS
RESET = WorkflowOperation.RESET_GPU
REBOOT = WorkflowOperation.RESTART_NODE
REPLACE = WorkflowOperation.REPLACE_NODE
RESTART_JOB = WorkflowOperation.RESTART_WORKLOAD


@pytest.mark.parametrize(
    ("branch_id", "expected"),
    [
        ("branch:node-b", "node-b"),
        ("branch:node-b:successor:3", "node-b"),
        ("branch:node-b:4", "node-b"),
        ("branch:node-b:successor:3:5", "node-b"),
        ("branch:node-b,node-c", None),
        ("branch:initial", None),
        ("shared", None),
        ("join", None),
        (None, None),
    ],
)
def test_branch_node_id_inverts_the_minted_shape(branch_id, expected):
    assert branch_node_id(branch_id) == expected


def test_minting_and_parsing_are_inverses():
    assert branch_node_ids(mint_branch_id(["node-c", "node-b"])) == ("node-b", "node-c")
    assert branch_node_ids(mint_branch_id(["node-b"], successor_revision=7)) == (
        "node-b",
    )
    assert mint_branch_id(["node-b"], successor_revision=7).endswith(":successor:7"), (
        "a successor branch id ends with its revision suffix"
    )
    assert branch_node_ids("branch:initial") == ()


def _dag():
    """STOP (shared) -> {node-b RESET, node-c RESET} -> RESTART (join)."""

    return workflow_request(
        "workflow-rf7",
        "incident-rf7",
        dag_enabled=True,
        dag_revision=1,
        official_steps=[
            workflow_step(STOP, node_ids=["node-b", "node-c"], branch_id="shared"),
            workflow_step(
                RESET,
                node_ids=["node-b"],
                depends_on_step_indexes=[0],
                branch_id="branch:node-b",
            ),
            workflow_step(
                RESET,
                node_ids=["node-c"],
                depends_on_step_indexes=[0],
                branch_id="branch:node-c",
            ),
            workflow_step(
                RESTART_JOB,
                node_ids=["node-b", "node-c"],
                depends_on_step_indexes=[1, 2],
                branch_id="join",
            ),
        ],
        completed_step_indexes=[0],
        completed_operations=[STOP],
        step_executions=[workflow_step_execution(0, STOP)],
    )


def _escalator(max_rungs: int = 2) -> BranchEscalator:
    def compile_steps(_workflow, operations, node_id, gpu_uuids):
        return [
            workflow_step(operation, node_ids=[node_id], gpu_uuids=list(gpu_uuids))
            for operation in operations
        ]

    return BranchEscalator(
        DagBrancher(RecoveryArbiter()), compile_steps, max_rungs=max_rungs
    )


def _fail(workflow, node_id, escalator):
    """Fail ``node_id``'s current rung and let the escalator rewrite."""

    index = max(
        index
        for index, step in enumerate(workflow.official_steps)
        if step.node_ids == [node_id]
        and step.operation in {RESET, REBOOT, REPLACE}
        and index not in workflow.superseded_step_indexes
    )
    workflow = copy_model(
        workflow,
        step_executions=[
            *workflow.step_executions,
            workflow_step_execution(
                index,
                workflow.official_steps[index].operation,
                WorkflowStepStatus.FAILED,
            ),
        ],
    )
    escalation = escalator.escalate_branch(workflow, index, "refused")
    assert escalation is not None, "a single-node branch step must escalate"
    return escalation


def test_counts_keys_are_the_exhausted_nodes_plus_the_escalated_nodes():
    escalator = _escalator(max_rungs=2)
    workflow = _dag()
    # node-b walks the whole ladder and exhausts it; node-c takes one rung.
    for _ in range(3):
        workflow = _fail(workflow, "node-b", escalator).workflow
    workflow = _fail(workflow, "node-c", escalator).workflow

    assert exhausted_node_ids(workflow) == frozenset({"node-b"})
    escalated = {
        event.details["node_id"]
        for event in workflow.events
        if event.kind is WorkflowEventKind.BRANCH_ESCALATION
        and event.code == WorkflowEventCode.BRANCH_ESCALATED
    }
    assert escalated == {"node-b", "node-c"}
    assert set(workflow.branch_escalation_counts) == (
        exhausted_node_ids(workflow) | escalated
    )
    assert workflow.branch_escalation_counts == {"node-b": 2, "node-c": 1}
    # The exhausted branch id is a successor spelling; it still maps back.
    (exhausted_id,) = workflow.exhausted_branch_ids
    assert exhausted_id != "branch:node-b", exhausted_id
    assert branch_node_id(exhausted_id) == "node-b"


def test_every_branch_escalation_event_names_both_identities():
    escalator = _escalator(max_rungs=1)
    workflow = _dag()
    first = _fail(workflow, "node-b", escalator)
    second = _fail(first.workflow, "node-b", escalator)

    events = [
        event
        for event in second.workflow.events
        if event.kind is WorkflowEventKind.BRANCH_ESCALATION
    ]
    assert [event.code for event in events] == [
        WorkflowEventCode.BRANCH_ESCALATED,
        WorkflowEventCode.BRANCH_EXHAUSTED,
    ]
    for event, escalation in zip(events, (first, second), strict=True):
        assert event.details["node_id"] == escalation.node_id == "node-b"
        assert event.details["branch_id"] == escalation.branch_id
        assert branch_node_id(event.details["branch_id"]) == event.details["node_id"]
    assert events[0].details["to_branch_id"] == events[1].details["branch_id"]


def test_exhaust_branch_from_the_caller_records_the_callers_reason():
    escalator = _escalator()
    workflow = _dag()

    exhausted = escalator.exhaust_branch(
        workflow, "node-c", "branch:node-c", reason="budget refused the next rung"
    )

    assert exhausted_node_ids(exhausted.workflow) == frozenset({"node-c"})
    (event,) = exhausted.workflow.events
    assert event.kind is WorkflowEventKind.BRANCH_ESCALATION
    assert event.code == WorkflowEventCode.BRANCH_EXHAUSTED
    assert event.reason == "budget refused the next rung"
    assert event.details["rung_count"] == 0
    assert event.details["superseded_indexes"] == [2]

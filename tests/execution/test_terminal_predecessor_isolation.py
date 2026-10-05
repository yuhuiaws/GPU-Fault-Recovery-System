"""A successor's verdict carries the isolation its predecessor left on the node.

Live 2026-10-05 (gpu-eks-cluster DESTR-001): a RESET_GPU chain cordoned and
tainted the node (MARK_UNSCHEDULABLE), then FAILED; a RESTART_GPU_DEVICE_PLUGIN
successor queued on the same incident had taken the incident pointer, so the
reset's QUARANTINED verdict was never written, and when the successor SUCCEEDED
the incident read RECOVERED while the node stayed cordoned, tainted and
annotated -- with no operator exit (``restore`` refused "RECOVERED"). The
isolation belongs to the incident, so a successor whose predecessor still holds
it ends the incident QUARANTINED, whatever its own steps say.
"""

from __future__ import annotations

from datetime import datetime, timezone

from gpu_fault.execution.models import WorkflowExecutionRequest
from gpu_fault.execution.terminal_state import (
    predecessor_holds_isolation,
    terminal_decision,
)
from gpu_fault.models import (
    IncidentState,
    WorkflowOperation,
    WorkflowStatus,
    WorkflowStepStatus,
)
from tests._builders import (
    active_workflow_executor,
    build_store,
    copy_model,
    fault_incident,
    workflow_request,
    workflow_step,
    workflow_step_execution,
)
from tests.execution._support import FakeAdapter, WorkflowStepOutcome

NOW = datetime(2026, 10, 5, 15, 14, 45, tzinfo=timezone.utc)
MARK = WorkflowOperation.MARK_UNSCHEDULABLE
RESET = WorkflowOperation.RESET_GPU
RELEASE = WorkflowOperation.RESTORE_SCHEDULING
PLUGIN = WorkflowOperation.RESTART_GPU_DEVICE_PLUGIN


def _failed_reset(*, released: bool = False, incident_id: str = "inc-a"):
    operations = [MARK, RESET, RELEASE]
    completed = [0, 2] if released else [0]
    return workflow_request(
        "wf-reset",
        incident_id,
        status=WorkflowStatus.FAILED,
        official_steps=[
            workflow_step(operation, node_ids=["node-a"]) for operation in operations
        ],
        completed_step_indexes=completed,
        completed_operations=[operations[index] for index in completed],
        step_executions=[
            workflow_step_execution(0, MARK),
            workflow_step_execution(1, RESET, WorkflowStepStatus.FAILED),
            *([workflow_step_execution(2, RELEASE)] if released else []),
        ],
        terminal_failure_reason="quiesce maintenance window expired",
    )


def _successor(*, released: bool = False):
    operations = [PLUGIN] + ([RELEASE] if released else [])
    return workflow_request(
        "wf-plugin",
        "inc-a",
        status=WorkflowStatus.PENDING,
        predecessor_workflow_id="wf-reset",
        official_steps=[
            workflow_step(operation, node_ids=["node-a"]) for operation in operations
        ],
        completed_step_indexes=list(range(len(operations))) if released else [],
        completed_operations=operations if released else [],
    )


def test_the_predecessors_unreleased_cordon_is_held_isolation() -> None:
    assert predecessor_holds_isolation(_failed_reset(), _successor()) is True
    assert predecessor_holds_isolation(_failed_reset(released=True), _successor()) is (
        False
    ), "the predecessor released the node itself"
    assert predecessor_holds_isolation(_failed_reset(), _successor(released=True)) is (
        False
    ), "the successor's own RESTORE_SCHEDULING released it"
    assert predecessor_holds_isolation(None, _successor()) is False
    assert (
        predecessor_holds_isolation(
            _failed_reset(incident_id="inc-other"), _successor()
        )
        is False
    ), "another incident's isolation is its own business"


def test_a_successor_verdict_keeps_the_incident_quarantined() -> None:
    incident = fault_incident(
        "inc-a", "event-a", node_ids=["node-a"], workflow_request_id="wf-plugin"
    )
    successor = _successor()

    with_predecessor = terminal_decision(
        successor,
        incident,
        WorkflowStatus.SUCCEEDED,
        1,
        now=NOW,
        actor="executor-a",
        predecessor=_failed_reset(),
    )
    alone = terminal_decision(
        successor, incident, WorkflowStatus.SUCCEEDED, 1, now=NOW, actor="executor-a"
    )
    explicit = terminal_decision(
        successor,
        incident,
        WorkflowStatus.FAILED,
        1,
        now=NOW,
        actor="executor-a",
        incident_state=IncidentState.ESCALATED,
        predecessor=_failed_reset(),
    )

    assert with_predecessor.incident is not None
    assert with_predecessor.incident.state is IncidentState.QUARANTINED, (
        "the node is still cordoned by the failed reset: the validated restore applies"
    )
    assert with_predecessor.workflow.events[-1].details["incident_state"] == (
        "QUARANTINED"
    )
    assert alone.incident is not None
    assert alone.incident.state is IncidentState.RECOVERED
    assert explicit.incident is not None
    assert explicit.incident.state is IncidentState.ESCALATED, (
        "a path that names the state keeps its say"
    )


def test_the_executor_hands_the_predecessor_to_the_terminal_verdict() -> None:
    store = build_store()
    reset = _failed_reset()
    successor = _successor()
    incident = fault_incident(
        "inc-a",
        "event-a",
        node_ids=["node-a"],
        state=IncidentState.ACTION_PENDING,
        workflow_request_id="wf-plugin",
        fencing_token=successor.fencing_token,
    )
    store.save_incident(incident)
    store.save_workflow(reset)
    store.save_workflow(successor)
    adapter = FakeAdapter({PLUGIN: WorkflowStepOutcome.succeeded()})
    executor = active_workflow_executor(
        store, [adapter], {PLUGIN, MARK, RESET, RELEASE}
    )

    result = executor.execute(
        "wf-plugin",
        WorkflowExecutionRequest(expected_fencing_token=successor.fencing_token),
    )

    assert result.status is WorkflowStatus.SUCCEEDED, result
    assert store.get_workflow("wf-plugin").status is WorkflowStatus.SUCCEEDED
    assert store.get_incident("inc-a").state is IncidentState.QUARANTINED, (
        "a RECOVERED incident on a node its predecessor still cordons is a trap "
        "without an exit; QUARANTINED keeps the validated restore available"
    )
    assert copy_model(reset).status is WorkflowStatus.FAILED, (
        "the predecessor is untouched"
    )

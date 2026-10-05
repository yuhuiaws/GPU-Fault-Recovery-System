"""A successor's restore closes the QUARANTINED predecessors of its chain.

A ``reboot-after`` / ``replace-after`` successor takes the node over from the
workflow it escalates (``node_operations.escalation_continues``). Once the
successor's RESTORE_SCHEDULING succeeded, the predecessor -- parked QUARANTINED
with its FAILED workflow -- holds nothing on the node any more: the node
carried the successor's ownership when it was released. The terminal hook
closes such predecessors like the ESCALATED incidents it already closed, so
a ladder that recovered the node leaves no record waiting for an operator.
"""

from __future__ import annotations

from gpu_fault.models import IncidentState, WorkflowOperation, WorkflowStatus
from gpu_fault.orchestration.incident_closure import IncidentClosureService
from tests._builders import (
    build_store,
    fault_incident,
    workflow_request,
    workflow_step,
    workflow_step_execution,
)

NODE = "node-a"
PRED_WF = "wf-reset"
REBOOT_WF = f"workflow-reboot-after-{PRED_WF}"
REPLACE_WF = f"workflow-replace-after-{REBOOT_WF}"


def _predecessor(store, *, nodes=(NODE,), state=IncidentState.QUARANTINED, status=None):
    steps = [
        workflow_step(WorkflowOperation.MARK_UNSCHEDULABLE, node_ids=list(nodes)),
        workflow_step(WorkflowOperation.RESET_GPU, node_ids=list(nodes)),
        workflow_step(WorkflowOperation.RESTORE_SCHEDULING, node_ids=list(nodes)),
    ]
    workflow = workflow_request(
        PRED_WF,
        "inc-reset",
        status=status or WorkflowStatus.FAILED,
        official_steps=steps,
        completed_step_indexes=[0, 1],
        completed_operations=[steps[0].operation, steps[1].operation],
    )
    incident = fault_incident(
        "inc-reset",
        "event-reset",
        state=state,
        node_ids=list(nodes),
        workflow_request_id=PRED_WF,
    )
    store.save_incident(incident)
    store.save_workflow(workflow)
    return incident


def _successor(store, request_id: str, event_id: str, *, state=IncidentState.RECOVERED):
    """A SUCCEEDED successor that cordoned, rebooted, validated and restored."""

    operations = (
        WorkflowOperation.MARK_UNSCHEDULABLE,
        WorkflowOperation.QUARANTINE,
        WorkflowOperation.RESTART_NODE,
        WorkflowOperation.VALIDATE_GPU,
        WorkflowOperation.RESTORE_SCHEDULING,
    )
    incident_id = f"inc-{event_id}"
    workflow = workflow_request(
        request_id,
        incident_id,
        status=WorkflowStatus.SUCCEEDED,
        official_steps=[workflow_step(op, node_ids=[NODE]) for op in operations],
        completed_step_indexes=list(range(len(operations))),
        completed_operations=list(operations),
        step_executions=[
            workflow_step_execution(index, op, phase="official")
            for index, op in enumerate(operations)
        ],
    )
    incident = fault_incident(
        incident_id,
        event_id,
        state=state,
        node_ids=[NODE],
        workflow_request_id=request_id,
    )
    store.save_incident(incident)
    store.save_workflow(workflow)
    return incident, workflow


def test_the_reboot_successors_restore_closes_the_quarantined_reset_incident() -> None:
    store = build_store()
    _predecessor(store)
    incident, workflow = _successor(store, REBOOT_WF, f"reboot-after-{PRED_WF}")

    closed = IncidentClosureService(store).on_terminal(
        workflow, incident, workflow.official_steps
    )

    assert closed == ["inc-reset"]
    predecessor = store.get_incident("inc-reset")
    assert predecessor.state is IncidentState.RECOVERED
    assert predecessor.reasons[-1] == (
        f"node restored via escalation successor {incident.incident_id}"
    )
    assert store.get_workflow(PRED_WF).events[-1].details[
        "restored_by_workflow_id"
    ] == (REBOOT_WF)


def test_a_two_rung_chain_closes_every_quarantined_predecessor() -> None:
    store = build_store()
    _predecessor(store)
    # The reboot rung also failed past its cordon and was escalated again.
    reboot_incident, reboot_workflow = _successor(
        store, REBOOT_WF, f"reboot-after-{PRED_WF}", state=IncidentState.QUARANTINED
    )
    store.save_workflow(
        reboot_workflow.model_copy(
            update={
                "status": WorkflowStatus.FAILED,
                "completed_step_indexes": [0, 1],
                "completed_operations": [
                    WorkflowOperation.MARK_UNSCHEDULABLE,
                    WorkflowOperation.QUARANTINE,
                ],
                "step_executions": reboot_workflow.step_executions[:2],
            }
        )
    )
    incident, workflow = _successor(store, REPLACE_WF, f"replace-after-{REBOOT_WF}")

    closed = IncidentClosureService(store).on_terminal(
        workflow, incident, workflow.official_steps
    )

    assert closed == [reboot_incident.incident_id, "inc-reset"]
    assert store.get_incident("inc-reset").state is IncidentState.RECOVERED
    assert store.get_incident(reboot_incident.incident_id).state is (
        IncidentState.RECOVERED
    )


def test_predecessors_the_restore_did_not_cover_or_that_still_run_stay() -> None:
    wider = build_store()
    _predecessor(wider, nodes=(NODE, "node-b"))
    incident, workflow = _successor(wider, REBOOT_WF, f"reboot-after-{PRED_WF}")
    assert (
        IncidentClosureService(wider).on_terminal(
            workflow, incident, workflow.official_steps
        )
        == []
    ), "a predecessor with a node the restore never touched is not closed"
    assert wider.get_incident("inc-reset").state is IncidentState.QUARANTINED

    running = build_store()
    _predecessor(running, status=WorkflowStatus.RUNNING)
    incident, workflow = _successor(running, REBOOT_WF, f"reboot-after-{PRED_WF}")
    assert (
        IncidentClosureService(running).on_terminal(
            workflow, incident, workflow.official_steps
        )
        == []
    ), "an open predecessor workflow still owns its node"

    unrelated = build_store()
    _predecessor(unrelated)
    incident, workflow = _successor(unrelated, "workflow-fresh", "event-fresh")
    assert (
        IncidentClosureService(unrelated).on_terminal(
            workflow, incident, workflow.official_steps
        )
        == []
    ), "a workflow that escalates nothing closes no predecessor"
    assert unrelated.get_incident("inc-reset").state is IncidentState.QUARANTINED


def test_a_failed_or_unrestored_successor_closes_nothing() -> None:
    store = build_store()
    _predecessor(store)
    incident, workflow = _successor(store, REBOOT_WF, f"reboot-after-{PRED_WF}")
    failed = workflow.model_copy(update={"status": WorkflowStatus.FAILED})

    assert (
        IncidentClosureService(store).on_terminal(
            failed,
            incident.model_copy(update={"state": IncidentState.ESCALATED}),
            failed.official_steps,
        )
        == []
    )
    assert store.get_incident("inc-reset").state is IncidentState.QUARANTINED

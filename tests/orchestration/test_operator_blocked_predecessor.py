"""A never-executed BLOCKED(NEEDS_OPERATOR) record does not hold its successor.

Control-plane review 2026-09-08, C-03 (CP-8 second layer; user decision
2026-09-08). An idle cluster reads as workload state UNKNOWN, so every XID plan
that mutates the node compiles BLOCKED(NEEDS_OPERATOR) by design (memory
``idle-cluster-workload-state-unknown``). F-A4 then makes that record occupy
its node: ``list_active_workflow_incidents`` lists it, ``disposition`` queued
the next same-node fault behind it as a successor with
``predecessor_workflow_id`` set, and the dispatcher held that successor for
as long as the predecessor stayed open -- which, for an operator block, is
until someone runs ``compile-blocked``. The real fault's recovery waited on a
record from the idle period that had never touched the node.

The rule now: a BLOCKED(NEEDS_OPERATOR) record with no step execution, no
completed and no superseded step is *replaced in place* (same ``request_id``,
``fencing_token + 1``) when the next fault arrives through its group, and is
not named as a node incumbent when the next fault arrives from outside it.
Either way the successor is dispatchable.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from gpu_fault.models import BlockedKind, IncidentState, WorkflowStatus
from gpu_fault.orchestration.coordinator import IncidentOrchestrator
from tests._builders import copy_model

from ._support import WorkloadState, _node_event, ingest

UNKNOWN_REASON = "node workload state is UNKNOWN"


def _blocked_by_idle_cluster(context, *, event_id: str, gpu_uuid: str):
    """An idle-period XID 79 whose plan compiled BLOCKED(NEEDS_OPERATOR).

    The simulated profile owns the safety steps, so the coordinator would
    compile SAFETY_PENDING; the review's production shape (no safety plan)
    is reproduced by pinning the persisted record to BLOCKED under the same
    node group key the fault family merges through.
    """

    _, incident, workflow = ingest(
        context,
        _node_event(79, event_id=event_id, gpu_uuid=gpu_uuid).model_copy(
            update={"workload_state": WorkloadState.UNKNOWN}
        ),
    )
    assert UNKNOWN_REASON in workflow.blocked_reasons
    blocked = copy_model(
        workflow,
        status=WorkflowStatus.BLOCKED,
        blocked_kind=BlockedKind.NEEDS_OPERATOR,
        safety_steps=[],
        safety_only=False,
    )
    context.store.save_workflow(blocked, expected=workflow)
    context.store.save_incident(
        copy_model(incident, state=IncidentState.ESCALATED), expected=incident
    )
    return context.store.get_incident(incident.incident_id), blocked


def _dispatchable_ids(context) -> set[str]:
    return {
        workflow.request_id
        for workflow in context.store.list_workflows(
            {WorkflowStatus.PENDING},
            dispatchable_at=datetime.now(timezone.utc) + timedelta(hours=1),
        )
    }


def test_the_next_fault_through_the_group_recompiles_the_block_in_place(context):
    incident, blocked = _blocked_by_idle_cluster(
        context, event_id="idle-xid79", gpu_uuid="GPU-a"
    )

    _, again_incident, successor = ingest(
        context, _node_event(79, event_id="live-xid79", gpu_uuid="GPU-b")
    )

    assert successor.request_id == blocked.request_id, "replaced in place"
    assert successor.status is WorkflowStatus.PENDING
    assert successor.blocked_kind is None and successor.blocked_reasons == []
    assert successor.predecessor_workflow_id is None
    assert successor.fencing_token == blocked.fencing_token + 1
    assert again_incident.incident_id == incident.incident_id
    assert again_incident.workflow_request_id == successor.request_id
    assert again_incident.fencing_token == successor.fencing_token
    assert again_incident.state is IncidentState.ACTION_PENDING
    assert set(again_incident.gpu_uuids) == {"GPU-a", "GPU-b"}
    assert _dispatchable_ids(context) == {successor.request_id}
    assert (
        context.store.count_held_workflows(
            {WorkflowStatus.PENDING},
            dispatchable_at=datetime.now(timezone.utc) + timedelta(hours=1),
        )
        == {}
    )
    assert len(context.store.list_workflows(limit=10)) == 1, "no chain grew"


def test_a_block_that_ran_a_step_still_holds_its_successor(context):
    """The boundary of the rule: once the executor ran anything, an operator
    has to look before the node is reused, exactly as F-A4 intends."""

    from gpu_fault.models import WorkflowOperation
    from tests._builders import workflow_step_execution

    _, blocked = _blocked_by_idle_cluster(
        context, event_id="idle-xid79-ran", gpu_uuid="GPU-a"
    )
    touched = copy_model(
        blocked,
        step_executions=[workflow_step_execution(0, WorkflowOperation.FREEZE_EVIDENCE)],
    )
    context.store.save_workflow(touched, expected=blocked)

    _, _, successor = ingest(
        context, _node_event(79, event_id="live-xid79-ran", gpu_uuid="GPU-b")
    )

    assert successor.request_id != blocked.request_id
    assert successor.predecessor_workflow_id == blocked.request_id
    assert _dispatchable_ids(context) == set()


def test_a_block_outside_the_group_is_not_named_as_the_incumbent(context):
    """Route two of the review: the block was created under no node group
    link (escalation, reset, the independent path) and the next fault's
    family looks for a node incumbent to serialize behind."""

    from gpu_fault.models import WorkflowOperation
    from tests._builders import fault_incident, workflow_request, workflow_step

    blocked = workflow_request(
        "wf-idle-block",
        "inc-idle-block",
        status=WorkflowStatus.BLOCKED,
        blocked_kind=BlockedKind.NEEDS_OPERATOR,
        fencing_token=1,
        blocked_reasons=[UNKNOWN_REASON],
        official_steps=[
            workflow_step(WorkflowOperation.RESTART_NODE, node_ids=["node-a"])
        ],
    )
    context.store.save_incident_and_workflow(
        fault_incident(
            "inc-idle-block",
            "idle-block-event",
            state=IncidentState.ESCALATED,
            workflow_request_id="wf-idle-block",
            node_ids=["node-a"],
            fencing_token=1,
        ),
        blocked,
    )

    _, _, successor = ingest(
        context, _node_event(79, event_id="live-xid79-outside", gpu_uuid="GPU-b")
    )

    assert successor.status is WorkflowStatus.PENDING
    assert successor.predecessor_workflow_id is None
    assert successor.request_id in _dispatchable_ids(context)
    assert context.store.get_workflow("wf-idle-block").status is WorkflowStatus.BLOCKED


def test_group_key_shape_is_the_node_scope_key():
    assert IncidentOrchestrator._node_group_key("cluster-a", "node-a") == (
        '["node-scope","cluster-a","node-a"]'
    )

"""``departed_node_cleanup_actions``: the records a proven departure cannot end
the usual way, each reported with its reason.

``tests/departed_nodes/test_departed_node_cleanup.py`` drives the live shapes.
These cases run the same cleanup cycle (``DepartedNodeCleanup.run_once`` on
the in-memory store) on the rest: an incident whose workflow is gone or that
has none, one an operator closes mid-cycle, a closure that refuses or loses a
race, a live or parked workflow, a write another writer beat, and a plan that
moved before its apply.
"""

from __future__ import annotations

import pytest

from gpu_fault.departed_node_cleanup_actions import (
    INCIDENT_NOT_CLOSABLE,
    OTHER,
    RECORD_MOVED,
    WORKFLOW_CHANGED_NODE,
    WORKFLOW_LIVE,
    Blocker,
    incident_command,
    workflow_command,
)
from gpu_fault.departed_nodes import DepartedNodeCleanupStatus
from gpu_fault.models import IncidentState, WorkflowStatus
from tests._builders import build_store
from tests.departed_nodes._cov95_departure_edges import (
    ALIAS,
    ConcurrentWorkflowWrite,
    OperatorClosesMidCycle,
    ReplannedBeforeApply,
    ScriptedClosure,
    accepted_alias,
    cleanup,
    hyperpod_departed,
    incident,
    parked,
)
from tests.departed_nodes._support import (
    EKS_CLUSTER,
    HP_CLUSTER,
    HP_NODE,
    blocked_never_changed,
    pending_never_dispatched,
)

RETIRED = f"agent-retired:{HP_NODE}"
CLEANED = DepartedNodeCleanupStatus.CLEANED
PENDING = DepartedNodeCleanupStatus.PENDING
REFUSED = DepartedNodeCleanupStatus.REFUSED


def test_a_blocker_kind_outside_the_metric_labels_is_refused() -> None:
    with pytest.raises(ValueError, match="unknown departed-node blocker kind: oops"):
        Blocker("oops", "a reason nobody can count")
    assert Blocker(OTHER, "counted").kind == OTHER


def test_an_incident_whose_workflow_is_gone_is_closed_on_the_departure() -> None:
    store = build_store()
    world_ = hyperpod_departed(store)
    closed = incident(store, "dangling", workflow_request_id="workflow-archived")

    [record] = world_.cleanup.run_once()

    assert record.status is CLEANED, record.blockers
    assert record.actions == [RETIRED, f"incident-closed:{closed}"], record.actions
    assert store.get_incident(closed).state is IncidentState.RECOVERED


def test_an_incident_still_waiting_on_a_workflow_is_left_for_it() -> None:
    store = build_store()
    world_ = hyperpod_departed(store)
    waiting = incident(store, "detected", state=IncidentState.ACTION_PENDING)

    [record] = world_.cleanup.run_once()

    assert record.status is PENDING, record.blockers
    assert record.blocker_classes == [WORKFLOW_LIVE]
    assert record.blockers == [
        f"incident {waiting} is ACTION_PENDING; it is closed once its workflow ends"
    ]
    assert record.admin_commands == [], "a waiting incident names no manual path"
    assert store.get_incident(waiting).state is IncidentState.ACTION_PENDING


def test_an_incident_an_operator_closed_mid_cycle_is_left_as_closed() -> None:
    store = build_store()
    world_ = hyperpod_departed(store)
    closed = incident(store, "operator")
    closure = ScriptedClosure(store, {closed: []})

    [record] = cleanup(
        world_, store=OperatorClosesMidCycle(store, closed), closure=closure
    ).run_once()

    assert record.status is CLEANED, record.blockers
    assert record.actions == [RETIRED], "the cleanup closed nothing itself"
    assert closure.calls == [], "a RECOVERED incident is not closed again"
    assert store.get_incident(closed).reasons[-1] == "closed by an operator"


@pytest.mark.parametrize(
    ("answer", "kind", "status", "command"),
    [
        (
            "refuse:incident incident-held still has an open workflow wf-x "
            "(RUNNING); wait for it to end",
            WORKFLOW_LIVE,
            PENDING,
            True,
        ),
        (
            "refuse:incident incident-held is DETECTED, not ESCALATED",
            INCIDENT_NOT_CLOSABLE,
            REFUSED,
            True,
        ),
        ("stale", RECORD_MOVED, PENDING, False),
    ],
    ids=["open-workflow", "not-closable", "lost-race"],
)
def test_a_close_the_closure_service_refuses_is_reported_with_its_reason(
    answer: str, kind: str, status: DepartedNodeCleanupStatus, command: bool
) -> None:
    store = build_store()
    world_ = hyperpod_departed(store)
    held = incident(store, "held")
    closure = ScriptedClosure(store, {held: [answer]})

    [record] = cleanup(world_, closure=closure).run_once()

    assert record.status is status, record.blockers
    assert record.blocker_classes == [kind]
    assert record.actions == [RETIRED], "nothing but the Agent was ended"
    [(called, operator, reason)] = closure.calls
    assert (called, operator) == (held, "departed-node-cleanup")
    assert reason.startswith(f"node departed: {HP_NODE} absent from Kubernetes"), reason
    if answer == "stale":
        assert record.blockers == [
            f"incident {held} moved: incident/{held} changed since it was read"
        ]
    else:
        assert record.blockers == [answer.removeprefix("refuse:")]
    assert record.admin_commands == ([incident_command(held)] if command else [])
    assert store.get_incident(held).state is IncidentState.ESCALATED


def test_a_close_that_changed_nothing_claims_nothing() -> None:
    store = build_store()
    world_ = hyperpod_departed(store)
    unchanged = incident(store, "unchanged")
    closure = ScriptedClosure(store, {unchanged: ["unchanged"]})

    [record] = cleanup(
        world_, closure=closure, notifications=world_.notifications
    ).run_once()

    assert record.status is CLEANED, record.blockers
    assert record.actions == [RETIRED], "an unchanged close is not an action"
    [notice] = [store.get_notification(item) for item in world_.notifications.sent]
    assert "No action is needed" in notice.body_text, notice.body_text


def test_a_running_workflow_is_left_to_its_executor() -> None:
    store = build_store()
    world_ = hyperpod_departed(store)
    request_id, incident_id = parked(store, "running", status=WorkflowStatus.RUNNING)

    [record] = world_.cleanup.run_once()

    assert record.status is PENDING, record.blockers
    assert record.blockers == [
        f"workflow {request_id} is RUNNING; the executor ends it"
    ]
    assert store.get_workflow(request_id).status is WorkflowStatus.RUNNING
    assert store.get_incident(incident_id).state is IncidentState.ESCALATED, (
        "an incident with a live workflow is not closed"
    )


def test_a_parked_record_an_operator_confirmed_ends_with_its_incident() -> None:
    store = build_store()
    world_ = hyperpod_departed(store)
    request_id, incident_id = parked(store, "confirmed", confirmed=True)

    [record] = world_.cleanup.run_once()

    assert record.status is CLEANED, record.blockers
    assert record.actions == [
        RETIRED,
        f"incident-closed:{incident_id}",
        f"workflow-closed:{request_id}",
    ], record.actions
    assert store.get_incident(incident_id).state is IncidentState.RECOVERED
    assert store.get_workflow(request_id).status is WorkflowStatus.SUPERSEDED


def test_an_accepted_departure_never_answers_what_an_action_on_the_node_did() -> None:
    store = build_store()
    world_ = accepted_alias(store)
    request_id, incident_id = parked(
        store, "alias-reset", cluster_id=EKS_CLUSTER, node_id=ALIAS
    )

    [record] = world_.cleanup.run_once()

    assert record.status is REFUSED, record.blockers
    assert record.blocker_classes == [WORKFLOW_CHANGED_NODE]
    [blocker] = record.blockers
    assert blocker.startswith(
        f"workflow {request_id} (BLOCKED) is not closed: workflow completed a "
        "node-mutating operation; unresolved node action not answered by a "
        f"departure: step 1 RESET_GPU on {ALIAS}"
    ), blocker
    assert record.admin_commands == [workflow_command(request_id)]
    assert store.get_workflow(request_id).status is WorkflowStatus.BLOCKED
    assert store.get_incident(incident_id).state is IncidentState.ESCALATED


def test_a_never_dispatched_close_another_writer_beat_waits_for_the_next_cycle() -> (
    None
):
    store = build_store()
    world_ = hyperpod_departed(store)
    request_id, incident_id = pending_never_dispatched(
        store, HP_CLUSTER, HP_NODE, suffix="raced"
    )
    view = ConcurrentWorkflowWrite(store, request_id)

    [record] = cleanup(world_, store=view).run_once()

    assert view.refused_writes == 1, "the close reached the write and lost it"
    assert record.status is PENDING, record.blockers
    assert record.blocker_classes == [RECORD_MOVED]
    assert record.blockers == [
        f"workflow {request_id} (PENDING) is not closed: workflow/{request_id} "
        "changed since it was read"
    ]
    assert record.admin_commands == [], "a lost race is retried, not handed over"
    assert store.get_workflow(request_id).status is WorkflowStatus.PENDING
    assert store.get_incident(incident_id).state is IncidentState.ACTION_PENDING


def test_a_never_changed_record_replanned_before_its_apply_waits() -> None:
    store = build_store()
    world_ = hyperpod_departed(store)
    request_id, incident_id = blocked_never_changed(
        store, HP_CLUSTER, HP_NODE, suffix="replanned"
    )
    view = ReplannedBeforeApply(store, request_id)

    [record] = cleanup(world_, store=view).run_once()

    assert view.command_reads >= 2, "the apply rebuilt the plan"
    assert record.status is PENDING, record.blockers
    assert record.blocker_classes == [RECORD_MOVED]
    assert record.blockers == [
        f"workflow {request_id} (BLOCKED) is not closed: workflow record changed "
        "before the close: workflow reconcile plan changed before apply"
    ]
    assert store.get_workflow(request_id).status is WorkflowStatus.BLOCKED
    assert store.get_incident(incident_id).state is IncidentState.ESCALATED

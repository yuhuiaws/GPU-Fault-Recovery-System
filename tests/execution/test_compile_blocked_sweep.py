"""The dispatcher closes a workflow that BLOCKED at compile time and never ran.

Until 2026-09-08 this close was ``gpu-fault-admin workflow-reconcile --mode
compile-blocked``: an operator reviewed a plan and applied it, every time to
unblock a release preflight that counted the record as active destructive work.
Every condition the operator's plan checked is a Store predicate, so the
dispatcher now runs the same predicate on ``sweep_stuck_records`` and closes the
record with the proofs the manual path had: re-derived from a fresh read, never
by age, written through a compare-and-set ``save_workflow`` with an attributed
``OPERATOR_RECONCILED`` event in the same write, nothing deleted.

The cases below pin what the sweep closes, everything it must keep its hands
off -- in particular a record the dispatcher itself BLOCKED for an internal
error, which a claim distinguishes from a compile-time refusal -- and that a
second pass changes nothing.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from gpu_fault.compile_blocked import (
    CLOSE_MARKER,
    DISPATCHER_ACTOR,
    SETTLED_CLOSE_MARKER,
    close_compile_blocked_workflows,
)
from gpu_fault.execution.config import WorkflowDispatcherConfig
from gpu_fault.execution.dispatcher import WorkflowDispatcher
from gpu_fault.models import (
    BlockedKind,
    IncidentState,
    WorkflowEventKind,
    WorkflowOperation,
    WorkflowStatus,
)
from gpu_fault.regional import RemoteActionCommand
from gpu_fault.remote_command_models import RemoteCommandStatus
from gpu_fault.store.shared.errors import StaleWriteError
from tests._builders import (
    active_workflow_executor,
    build_store,
    fault_incident,
    workflow_request,
    workflow_step,
    workflow_step_execution,
)

WORKFLOW = "workflow-924abfb5-be1d-481d-a0f6-72801da4007c"
INCIDENT = "inc-host-node-a-efa_inventory_mismatch-node"
BLOCKED_REASON = "no executable owner for efaDriverRemediation"
STEPS = [
    workflow_step(WorkflowOperation.MARK_UNSCHEDULABLE),
    workflow_step(WorkflowOperation.REMEDIATE_EFA_DRIVER),
    workflow_step(WorkflowOperation.RESTORE_SCHEDULING),
]


def compile_blocked_pair(
    store,
    *,
    request_id: str = WORKFLOW,
    incident_id: str = INCIDENT,
    incident_state=IncidentState.ESCALATED,
    **overrides,
):
    """The live 2026-09-04 shape: one compiler refusal under a settled incident."""

    now = datetime.now(timezone.utc)
    incident = fault_incident(
        incident_id,
        f"event-{incident_id}",
        event_type="HOST",
        node_ids=["node-a"],
        state=incident_state,
        fencing_token=3,
        # The incident moved on to a (failed) validated restore; the BLOCKED
        # record is no longer the one it names.
        workflow_request_id="workflow-validated-restore-failed",
    )
    values = {
        "official_action": "REMEDIATE_EFA_DRIVER",
        "official_steps": STEPS,
        "blocked_reasons": [BLOCKED_REASON],
        "blocked_kind": BlockedKind.NEEDS_OPERATOR,
        # Seconds old: the predicate never reads age.
        "created_at": now,
        "updated_at": now,
    }
    values.update(overrides)
    workflow = workflow_request(
        request_id, incident_id, status=WorkflowStatus.BLOCKED, **values
    )
    store.save_incident(incident)
    store.save_workflow(workflow)
    return incident, workflow


def dispatcher(store) -> WorkflowDispatcher:
    return WorkflowDispatcher(
        store,
        active_workflow_executor(store, [], frozenset()),
        WorkflowDispatcherConfig(enabled=True, batch_size=100),
    )


def reconcile_events(store, request_id: str = WORKFLOW):
    return [
        event
        for event in store.get_workflow(request_id).events
        if event.kind is WorkflowEventKind.OPERATOR_RECONCILED
    ]


def all_workflow_ids(store) -> list[str]:
    return sorted(
        workflow.request_id
        for workflow in store.list_workflows(set(WorkflowStatus), limit=1000)
    )


def test_a_compile_time_blocked_no_op_is_closed_on_the_dispatch_tick() -> None:
    store = build_store()
    incident, _ = compile_blocked_pair(store)
    before = all_workflow_ids(store)

    dispatcher(store).run_once()

    closed = store.get_workflow(WORKFLOW)
    assert closed.status is WorkflowStatus.SUPERSEDED
    assert closed.superseded_at is not None, "the close must stamp superseded_at"
    reason = closed.preemption_reason or ""
    assert f"{DISPATCHER_ACTOR} reconciliation" in reason
    assert CLOSE_MARKER in reason
    assert BLOCKED_REASON in reason, "the original blocked_reasons must survive"
    assert all_workflow_ids(store) == before, "the sweep must never delete a record"
    # The incident is not written: a compile-time record owns no node state.
    assert store.get_incident(INCIDENT) == incident


def test_the_close_carries_an_event_attributed_to_the_dispatcher() -> None:
    store = build_store()
    compile_blocked_pair(store)

    dispatcher(store).run_once()

    (event,) = reconcile_events(store)
    assert event.actor == DISPATCHER_ACTOR
    assert event.status == WorkflowStatus.SUPERSEDED.value
    assert event.details["previous_status"] == WorkflowStatus.BLOCKED.value
    assert event.details["terminalization"] == CLOSE_MARKER
    assert event.details["blocked_reasons"] == [BLOCKED_REASON]


def test_a_second_tick_leaves_the_closed_record_alone() -> None:
    store = build_store()
    compile_blocked_pair(store)
    sweep = dispatcher(store)

    sweep.run_once()
    first = store.get_workflow(WORKFLOW)
    sweep.run_once()

    assert store.get_workflow(WORKFLOW) == first, "a rerun must be a no-op"
    assert len(reconcile_events(store)) == 1, "a rerun must not append an event"


@pytest.mark.parametrize(
    ("name", "overrides", "incident_state"),
    [
        (
            "dispatched",
            {
                "step_executions": [
                    workflow_step_execution(0, WorkflowOperation.MARK_UNSCHEDULABLE)
                ]
            },
            IncidentState.ESCALATED,
        ),
        (
            "completed an operation",
            {"completed_operations": [WorkflowOperation.MARK_UNSCHEDULABLE]},
            IncidentState.ESCALATED,
        ),
        (
            "holds an owner",
            {"execution_owner_id": "executor-b"},
            IncidentState.ESCALATED,
        ),
        # The dispatcher's own internal-error BLOCKED: claimed, so the epoch
        # moved. It is an operator matter the alert names, not a compiler no-op.
        (
            "was claimed",
            {"execution_epoch": 1, "blocked_kind": BlockedKind.INTERNAL_ERROR},
            IncidentState.ESCALATED,
        ),
        ("is plan-driven", {"source_plan_id": "plan-a"}, IncidentState.ESCALATED),
        (
            "holds budget claims",
            {"remediation_budget_claims": ["cluster-a/efa"]},
            IncidentState.ESCALATED,
        ),
        ("has no blocked_reasons", {"blocked_reasons": []}, IncidentState.ESCALATED),
        ("incident still waits", {}, IncidentState.ACTION_PENDING),
    ],
)
def test_anything_that_ran_or_is_still_waited_on_is_left_blocked(
    name: str, overrides: dict, incident_state: IncidentState
) -> None:
    store = build_store()
    compile_blocked_pair(store, incident_state=incident_state, **overrides)

    dispatcher(store).run_once()

    assert store.get_workflow(WORKFLOW).status is WorkflowStatus.BLOCKED, name
    assert reconcile_events(store) == [], f"{name}: a refusal writes no event"


def test_an_open_remote_command_keeps_the_record_blocked() -> None:
    store = build_store()
    incident, workflow = compile_blocked_pair(store)
    store.ensure_remote_command(
        RemoteActionCommand(
            command_id="remote-open",
            cluster_id=incident.cluster_id,
            workflow_request_id=WORKFLOW,
            incident_id=INCIDENT,
            step_index=1,
            fencing_token=workflow.fencing_token,
            idempotency_key=f"{WORKFLOW}/1/REMEDIATE_EFA_DRIVER",
            step=STEPS[1],
            workflow=workflow,
            incident=incident,
            status=RemoteCommandStatus.WAITING,
        )
    )

    closed = close_compile_blocked_workflows(store, now=datetime.now(timezone.utc))

    assert closed == []
    assert store.get_workflow(WORKFLOW).status is WorkflowStatus.BLOCKED
    assert (
        store.get_remote_command("remote-open").status is RemoteCommandStatus.WAITING
    ), "this sweep cancels nothing; an open command is a reason to refuse"

    # On a full tick the orphaned-command sweep settles that command (BLOCKED is
    # terminal for it), so the record closes on the *next* tick, commands first.
    sweep = dispatcher(store)
    sweep.run_once()
    assert store.get_remote_command("remote-open").status is RemoteCommandStatus.FAILED
    assert store.get_workflow(WORKFLOW).status is WorkflowStatus.BLOCKED
    sweep.run_once()
    assert store.get_workflow(WORKFLOW).status is WorkflowStatus.SUPERSEDED


def test_a_record_whose_incident_is_missing_is_left_blocked() -> None:
    store = build_store()
    now = datetime.now(timezone.utc)
    store.save_workflow(
        workflow_request(
            WORKFLOW,
            "inc-gone",
            status=WorkflowStatus.BLOCKED,
            official_steps=STEPS,
            blocked_reasons=[BLOCKED_REASON],
            created_at=now - timedelta(days=30),
            updated_at=now - timedelta(days=30),
        )
    )

    dispatcher(store).run_once()

    assert store.get_workflow(WORKFLOW).status is WorkflowStatus.BLOCKED, (
        "a month of age proves nothing; a missing incident fails closed"
    )


def test_a_row_that_moved_since_the_read_is_refused_and_the_sweep_goes_on() -> None:
    """Compare-and-set, per record, with the others unaffected."""

    store = build_store()
    compile_blocked_pair(store)
    compile_blocked_pair(store, request_id="workflow-second", incident_id="inc-second")
    genuine_save = store.save_workflow
    refused: list[str] = []

    def contested_save(workflow, *, expected=None):
        if workflow.request_id == WORKFLOW and expected is not None and not refused:
            refused.append(workflow.request_id)
            raise StaleWriteError("workflow moved since it was read")
        return genuine_save(workflow, expected=expected)

    store.save_workflow = contested_save  # type: ignore[method-assign]

    closed = close_compile_blocked_workflows(store, now=datetime.now(timezone.utc))

    assert refused == [WORKFLOW]
    assert closed == ["workflow-second"], "the second record must still be closed"
    assert store.get_workflow(WORKFLOW).status is WorkflowStatus.BLOCKED, (
        "a refused compare-and-set must leave the row exactly as it was"
    )
    assert store.get_workflow("workflow-second").status is WorkflowStatus.SUPERSEDED


# --------------------------------------------------------------------------- #
# The second shape: a BLOCKED record whose incident is already RECOVERED.
# --------------------------------------------------------------------------- #
def settled_incident_pair(
    store, *, incident_state=IncidentState.RECOVERED, **overrides
):
    """The live 2026-09-11 shape: an Always-Fatal SXID's RESTART_BM workflow,
    FREEZE_EVIDENCE already run, BLOCKED SAFETY_SETTLED, its incident closed by
    an operator after a validated restore of another incident freed the node."""

    now = datetime.now(timezone.utc)
    incident = fault_incident(
        INCIDENT,
        f"event-{INCIDENT}",
        event_type="SXID",
        node_ids=["node-a"],
        state=incident_state,
        fencing_token=1,
        workflow_request_id=WORKFLOW,
    )
    values = {
        "official_action": "RESTART_BM",
        "official_steps": [
            workflow_step(WorkflowOperation.FREEZE_EVIDENCE),
            workflow_step(WorkflowOperation.RESTART_NODE),
        ],
        "blocked_reasons": ["NVIDIA Table 23 classifies SXID 23001 as Always Fatal"],
        "blocked_kind": BlockedKind.SAFETY_SETTLED,
        "completed_operations": [WorkflowOperation.FREEZE_EVIDENCE],
        "completed_step_indexes": [0],
        "created_at": now,
        "updated_at": now,
    }
    values.update(overrides)
    workflow = workflow_request(
        WORKFLOW, INCIDENT, status=WorkflowStatus.BLOCKED, **values
    )
    store.save_incident(incident)
    store.save_workflow(workflow)
    return incident, workflow


def test_a_blocked_workflow_of_a_recovered_incident_is_closed_on_the_tick() -> None:
    store = build_store()
    incident, _ = settled_incident_pair(store)
    before = all_workflow_ids(store)

    dispatcher(store).run_once()

    closed = store.get_workflow(WORKFLOW)
    assert closed.status is WorkflowStatus.SUPERSEDED, closed.status
    assert SETTLED_CLOSE_MARKER in (closed.preemption_reason or "")
    assert "23001" in (closed.preemption_reason or ""), "blocked_reasons must survive"
    (event,) = reconcile_events(store)
    assert event.actor == DISPATCHER_ACTOR
    assert event.details["completed_operations"] == ["FREEZE_EVIDENCE"], event.details
    assert all_workflow_ids(store) == before, "the sweep must never delete a record"
    assert store.get_incident(INCIDENT) == incident, "the incident is not rewritten"


def test_an_owned_blocked_record_is_not_the_settled_shape() -> None:
    store = build_store()
    settled_incident_pair(store, execution_owner_id="executor-a")

    dispatcher(store).run_once()

    assert store.get_workflow(WORKFLOW).status is WorkflowStatus.BLOCKED


def test_stale_budget_claims_are_released_with_the_settled_close() -> None:
    """Budget occupancy counts RUNNING rows with a live lease, so claims left on
    a BLOCKED record hold nothing. Refusing on them kept three SAFETY_SETTLED
    records of a RECOVERED incident open for five days (live 2026-09-11); the
    close now drops them and says which it dropped."""

    store = build_store()
    settled_incident_pair(
        store, remediation_budget_claims=["node:cluster-a:node-a", "region"]
    )

    dispatcher(store).run_once()

    closed = store.get_workflow(WORKFLOW)
    assert closed.status is WorkflowStatus.SUPERSEDED
    assert closed.remediation_budget_claims == []
    (event,) = reconcile_events(store)
    assert event.details["released_budget_claims"] == [
        "node:cluster-a:node-a",
        "region",
    ]


def test_an_escalated_incidents_blocked_record_waits_for_the_operator() -> None:
    """ESCALATED is not settled: the operator may still act on that record."""

    store = build_store()
    settled_incident_pair(store, incident_state=IncidentState.ESCALATED)

    dispatcher(store).run_once()

    assert store.get_workflow(WORKFLOW).status is WorkflowStatus.BLOCKED

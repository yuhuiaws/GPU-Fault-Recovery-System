"""Operator writes leave an attributed ``WorkflowEvent`` in the same transaction.

Architecture review 2026-09-07 (audit trail, I1). ``amend_workflow`` bumped the
merge revision and wrote the row, the reconcile transactions rewrote status and
``preemption_reason``, and none of them appended to ``events`` -- so the one
append-only history on the record had no entry for the one write a human made.
The event now lands in the same transaction as the write it describes, carrying
who (STS ARN), which approval (plan digests), and the status transition.

The memory store keeps its own copies of these three writes; this file covers
them alongside the shared transactional mixin the SQLite and PostgreSQL stores run.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

import pytest

from gpu_fault.models import (
    IncidentState,
    PlanStatus,
    RecoveryPlan,
    WorkflowEvent,
    WorkflowEventKind,
    WorkflowOperation,
    WorkflowStatus,
    build_operator_event,
)
from gpu_fault.regional import RemoteActionCommand
from gpu_fault.remote_command_models import RemoteCommandStatus
from gpu_fault.store import SqliteStore
from tests._builders import build_store, fault_incident, workflow_request, workflow_step
from tests.store._postgres_processor_claim_support import (
    _truncate,
    postgres_store_instance,
)

NOW = datetime(2026, 9, 7, 9, 0, tzinfo=timezone.utc)
ACTOR = "arn:aws:sts::123456789012:assumed-role/Admin/alice"
ADMIN_DIGEST = "c" * 64
INCIDENT_ID = "inc-op"
BLOCKED_ID = "wf-blocked"
SUCCESSOR_ID = "wf-restored"
PLAN_ID = "plan-op"
RETIRED_ID = "wf-retired"
CURRENT_ID = "wf-current"
NODES = ["node-a", "node-b"]


@pytest.fixture(params=["memory", "sqlite", "postgres"])
def store(request, tmp_path):
    if request.param == "memory":
        yield build_store()
        return
    if request.param == "sqlite":
        sqlite = SqliteStore(str(tmp_path / "operator-events.db"))
        try:
            yield sqlite
        finally:
            sqlite.close()
        return
    if not os.getenv("GPU_FAULT_TEST_POSTGRES_URL"):
        pytest.skip("GPU_FAULT_TEST_POSTGRES_URL is required")
    for postgres in postgres_store_instance():
        yield postgres
    _truncate()


def _operator_events(store, request_id: str) -> list[WorkflowEvent]:
    return [
        event
        for event in store.get_workflow(request_id).events
        if event.kind
        in (
            WorkflowEventKind.OPERATOR_RECONCILED,
            WorkflowEventKind.OPERATOR_RETIRED_GENERATION,
        )
    ]


def _restored_state(store) -> None:
    store.save_plan(
        RecoveryPlan(
            plan_id=PLAN_ID,
            incident_id=INCIDENT_ID,
            attempt_id="attempt-a",
            trigger="test",
            runtime_profile_version="profile-v1",
            steps=[],
            workflow_request_id=BLOCKED_ID,
            status=PlanStatus.FAILED,
            created_at=NOW - timedelta(hours=2),
        )
    )
    store.save_workflow(
        workflow_request(
            BLOCKED_ID,
            INCIDENT_ID,
            status=WorkflowStatus.BLOCKED,
            fencing_token=7,
            execution_epoch=2,
            source_plan_id=PLAN_ID,
            official_action=WorkflowOperation.QUARANTINE.value,
            official_steps=[workflow_step(WorkflowOperation.QUARANTINE)],
            created_at=NOW - timedelta(hours=2),
            updated_at=NOW - timedelta(hours=1),
        )
    )
    store.save_workflow(
        workflow_request(
            SUCCESSOR_ID,
            INCIDENT_ID,
            status=WorkflowStatus.SUCCEEDED,
            fencing_token=7,
            predecessor_workflow_id=BLOCKED_ID,
            completed_operations=[WorkflowOperation.RESTORE_SCHEDULING],
            created_at=NOW - timedelta(hours=1),
            updated_at=NOW - timedelta(minutes=30),
        )
    )
    store.save_incident(
        fault_incident(
            INCIDENT_ID,
            "event-op",
            state=IncidentState.RECOVERED,
            workflow_request_id=SUCCESSOR_ID,
            fencing_token=7,
            created_at=NOW - timedelta(hours=2),
            updated_at=NOW - timedelta(minutes=30),
        )
    )


def _retired_pair(store, *, with_command: bool = False) -> None:
    steps = [
        workflow_step(WorkflowOperation.FREEZE_EVIDENCE, node_ids=NODES),
        workflow_step(WorkflowOperation.STOP_WORKLOADS, node_ids=NODES),
    ]
    incident = fault_incident(
        INCIDENT_ID,
        "event-retired",
        cluster_id="cluster-a",
        node_ids=list(NODES),
        state=IncidentState.ACTION_PENDING,
        fencing_token=4,
        workflow_request_id=CURRENT_ID,
    )
    store.save_incident(incident)
    retired = workflow_request(
        RETIRED_ID,
        INCIDENT_ID,
        status=WorkflowStatus.RUNNING,
        fencing_token=1,
        official_action="RESTART_APP",
        official_steps=steps,
        execution_owner_id="executor-b",
        execution_lease_expires_at=NOW + timedelta(minutes=3),
        completed_step_indexes=[0],
        completed_operations=[WorkflowOperation.FREEZE_EVIDENCE],
        updated_at=NOW - timedelta(minutes=1),
    )
    store.save_workflow(retired)
    store.save_workflow(
        workflow_request(
            CURRENT_ID,
            INCIDENT_ID,
            fencing_token=4,
            official_action="RUN_DIAGNOSTICS",
            official_steps=[workflow_step(WorkflowOperation.VALIDATE_GPU)],
        )
    )
    if with_command:
        store.ensure_remote_command(
            RemoteActionCommand(
                command_id="command-retired",
                cluster_id="cluster-a",
                workflow_request_id=RETIRED_ID,
                incident_id=INCIDENT_ID,
                step_index=1,
                fencing_token=1,
                idempotency_key=f"{RETIRED_ID}/1/STOP_WORKLOADS",
                step=steps[1],
                workflow=retired,
                incident=incident,
                status=RemoteCommandStatus.WAITING,
            )
        )


def test_amend_workflow_lands_the_event_with_the_update(store) -> None:
    store.save_workflow(
        workflow_request("wf-a", INCIDENT_ID, status=WorkflowStatus.BLOCKED)
    )
    before = store.get_workflow("wf-a")
    # Built the way the never-changed close will build it: from the copy that
    # already carries the new status, naming the status it moved from.
    event = build_operator_event(
        before.model_copy(update={"status": WorkflowStatus.SUPERSEDED}),
        WorkflowEventKind.OPERATOR_RECONCILED,
        actor=ACTOR,
        reference="CHG-1",
        previous_status=before.status,
        at=NOW,
    )
    assert isinstance(event, WorkflowEvent), "build_operator_event returns the event"

    amended = store.amend_workflow(
        "wf-a", {"status": WorkflowStatus.SUPERSEDED}, event=event
    )

    saved = store.get_workflow("wf-a")
    assert saved == amended, "the returned row must be the row that was written"
    assert saved.status is WorkflowStatus.SUPERSEDED
    assert saved.merge_revision == before.merge_revision + 1, (
        "recording the event must not cost a second merge-revision bump"
    )
    assert [item.kind for item in saved.events[len(before.events) :]] == [
        WorkflowEventKind.OPERATOR_RECONCILED
    ], f"the operator event did not land with the amend: {saved.events}"
    assert saved.events[-1].actor == ACTOR
    assert saved.events[-1].details["reference"] == "CHG-1"
    assert saved.events[-1].details["previous_status"] == WorkflowStatus.BLOCKED.value
    assert saved.events[-1].status == WorkflowStatus.SUPERSEDED.value


def test_amend_workflow_without_an_event_appends_nothing(store) -> None:
    store.save_workflow(
        workflow_request("wf-b", INCIDENT_ID, status=WorkflowStatus.BLOCKED)
    )
    before = store.get_workflow("wf-b")

    store.amend_workflow("wf-b", {"status": WorkflowStatus.SUPERSEDED})

    assert store.get_workflow("wf-b").events == before.events, (
        "an amend that names no event must not invent one"
    )


def test_the_restore_reconcile_records_who_closed_the_record(store) -> None:
    _restored_state(store)

    updated, _incident, _plan = store.reconcile_restored_workflow(
        BLOCKED_ID,
        SUCCESSOR_ID,
        expected_fencing_token=7,
        expected_execution_epoch=2,
        reference="CHG-1",
        reconciled_at=NOW,
        actor=ACTOR,
        approval={"admin_plan_sha256": ADMIN_DIGEST, "plan_sha256": "b" * 64},
    )

    events = _operator_events(store, BLOCKED_ID)
    assert len(events) == 1, f"expected one operator event, found {events}"
    event = events[0]
    assert event.kind is WorkflowEventKind.OPERATOR_RECONCILED
    assert event.code == "OPERATOR_RECONCILED"
    assert event.actor == ACTOR, "the STS identity must be the event actor"
    assert event.at == NOW
    assert event.status == WorkflowStatus.SUPERSEDED.value
    assert event.details["previous_status"] == WorkflowStatus.BLOCKED.value
    assert event.details["new_status"] == WorkflowStatus.SUPERSEDED.value
    assert event.details["reference"] == "CHG-1"
    assert event.details["successor_workflow_id"] == SUCCESSOR_ID
    assert event.details["admin_plan_sha256"] == ADMIN_DIGEST
    assert event.details["plan_sha256"] == "b" * 64
    assert updated.events == store.get_workflow(BLOCKED_ID).events


def test_an_unknown_actor_is_named_not_omitted(store) -> None:
    _restored_state(store)

    store.reconcile_restored_workflow(
        BLOCKED_ID,
        SUCCESSOR_ID,
        expected_fencing_token=7,
        expected_execution_epoch=2,
        reference="CHG-2",
        reconciled_at=NOW,
    )

    (event,) = _operator_events(store, BLOCKED_ID)
    assert event.actor == "unknown-identity", (
        "a write whose identity could not be resolved must still say so"
    )


def test_a_refused_restore_reconcile_records_no_event(store) -> None:
    _restored_state(store)

    with pytest.raises(ValueError, match="execution epoch changed"):
        store.reconcile_restored_workflow(
            BLOCKED_ID,
            SUCCESSOR_ID,
            expected_fencing_token=7,
            expected_execution_epoch=1,
            reference="CHG-STALE",
            reconciled_at=NOW,
            actor=ACTOR,
        )

    assert _operator_events(store, BLOCKED_ID) == [], (
        "a refused write must not leave an event claiming it happened"
    )


def test_the_retired_generation_reconcile_records_who_revoked_it(store) -> None:
    _retired_pair(store)

    revoked, _incident = store.reconcile_retired_generation_workflow(
        RETIRED_ID,
        CURRENT_ID,
        expected_fencing_token=1,
        reference="pre-deploy-1",
        reconciled_at=NOW,
        actor=ACTOR,
        approval={"admin_plan_sha256": ADMIN_DIGEST},
    )

    assert revoked.status is WorkflowStatus.SUPERSEDED
    events = _operator_events(store, RETIRED_ID)
    assert len(events) == 1, f"expected one operator event, found {events}"
    event = events[0]
    assert event.kind is WorkflowEventKind.OPERATOR_RETIRED_GENERATION
    assert event.actor == ACTOR
    assert event.at == NOW
    assert event.details["previous_status"] == WorkflowStatus.RUNNING.value
    assert event.details["new_status"] == WorkflowStatus.SUPERSEDED.value
    assert event.details["reference"] == "pre-deploy-1"
    assert event.details["successor_workflow_id"] == CURRENT_ID
    assert event.details["admin_plan_sha256"] == ADMIN_DIGEST


def test_the_dispatcher_sweep_is_not_recorded_as_an_operator(store) -> None:
    """``reference=None`` is the self-healing sweep; it is not a human write."""

    _retired_pair(store)

    store.reconcile_retired_generation_workflow(
        RETIRED_ID,
        CURRENT_ID,
        expected_fencing_token=1,
        reference=None,
        reconciled_at=NOW,
    )

    assert _operator_events(store, RETIRED_ID) == [], (
        "the automatic sweep must not masquerade as an operator reconciliation"
    )

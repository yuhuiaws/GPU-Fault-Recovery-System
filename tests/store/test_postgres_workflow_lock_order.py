"""Every multi-row workflow transaction locks the incident row first.

Store review 2026-09-07, item C. The families use different advisory keys, so
the row locks serialize them, and two families taking the same two rows in
opposite orders were a deadlock pair: ``save_incident_and_workflow`` and the
executor's ``save_workflow_and_incident_if_leased`` went workflow -> incident
while every merge went incident -> workflow; PostgreSQL aborted one side after
``deadlock_timeout`` with 40P01, a one-second stall and a 503. The rule is now
incident, then workflows in ascending key, then the plan.

Checked from the outside: a side connection holds the incident row, the store
call blocks on it, and while it is blocked every workflow row the call will
touch is still free (``FOR UPDATE NOWAIT`` succeeds) -- the call has taken no
workflow lock ahead of the incident. Released, the call completes.
"""

from __future__ import annotations

import os
import threading
from datetime import datetime, timedelta, timezone
from typing import Callable

import pytest

from gpu_fault.models import (
    IncidentState,
    PlanStatus,
    RecoveryPlan,
    WorkflowOperation,
    WorkflowStatus,
)
from gpu_fault.store.shared.errors import StaleWriteError
from tests._builders import copy_model, fault_incident, workflow_request, workflow_step
from tests.store._postgres_processor_claim_support import (
    POSTGRES_URL,
    _truncate,
    postgres_store_instance,
)

pytestmark = pytest.mark.skipif(
    not os.getenv("GPU_FAULT_TEST_POSTGRES_URL"),
    reason="GPU_FAULT_TEST_POSTGRES_URL is not configured",
)


@pytest.fixture
def store():
    yield from postgres_store_instance()
    _truncate()


NOW = datetime(2026, 9, 7, 9, 0, tzinfo=timezone.utc)
STEPS = [workflow_step(WorkflowOperation.RESET_GPU, node_ids=["node-a"])]


def _hold_incident_row(incident_id: str):
    import psycopg

    connection = psycopg.connect(POSTGRES_URL, autocommit=False)
    with connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT 1 FROM gpu_fault_objects
            WHERE kind='incident' AND key=%s
            FOR UPDATE
            """,
            (incident_id,),
        )
        assert cursor.fetchone() is not None, f"incident {incident_id} not seeded"
    return connection


def _workflow_rows_are_free(connection, request_ids: list[str]) -> None:
    with connection.cursor() as cursor:
        for request_id in request_ids:
            cursor.execute(
                """
                SELECT 1 FROM gpu_fault_objects
                WHERE kind='workflow' AND key=%s
                FOR UPDATE NOWAIT
                """,
                (request_id,),
            )
            assert cursor.fetchone() is not None, f"workflow {request_id} not seeded"


def _blocked_on_incident_then_completes(
    action: Callable[[], object],
    incident_id: str,
    touched_workflows: list[str],
    *,
    settle_seconds: float = 0.5,
):
    connection = _hold_incident_row(incident_id)
    outcome: dict[str, object] = {}

    def run() -> None:
        try:
            outcome["result"] = action()
        except Exception as exc:  # noqa: BLE001 - reported to the test thread
            outcome["error"] = exc

    worker = threading.Thread(target=run)
    worker.start()
    try:
        worker.join(settle_seconds)
        assert worker.is_alive(), "the call did not wait for the incident row"
        # Blocked on the incident, it must hold none of its workflow rows yet.
        _workflow_rows_are_free(connection, touched_workflows)
    finally:
        connection.rollback()
        connection.close()
    worker.join(30)
    assert not worker.is_alive(), "the call never finished after the release"
    if "error" in outcome:
        raise outcome["error"]  # type: ignore[misc]
    return outcome["result"]


def _seed_pair(store, *, fencing_token: int = 1):
    incident = fault_incident(
        "inc-lo",
        "event-lo",
        state=IncidentState.ACTION_PENDING,
        workflow_request_id="wf-lo",
        fencing_token=fencing_token,
        node_ids=["node-a"],
        created_at=NOW,
        updated_at=NOW,
    )
    workflow = workflow_request(
        "wf-lo",
        "inc-lo",
        fencing_token=fencing_token,
        official_steps=STEPS,
        created_at=NOW,
        updated_at=NOW,
    )
    store.save_incident_and_workflow(incident, workflow)
    return store.get_incident("inc-lo"), store.get_workflow("wf-lo")


def test_save_incident_and_workflow_locks_the_incident_first(store):
    incident, workflow = _seed_pair(store)

    _blocked_on_incident_then_completes(
        lambda: store.save_incident_and_workflow(
            copy_model(incident, node_ids=["node-a", "node-b"]),
            copy_model(workflow, blocked_reasons=["widened"]),
        ),
        "inc-lo",
        ["wf-lo"],
    )

    assert store.get_incident("inc-lo").node_ids == ["node-a", "node-b"]
    current = store.get_workflow("wf-lo")
    assert current.blocked_reasons == ["widened"]
    assert current.merge_revision == 1


def test_leased_save_of_workflow_and_incident_locks_the_incident_first(store):
    incident, _ = _seed_pair(store)
    claimed = store.claim_workflow(
        "wf-lo", "executor-a", 1, lease_duration=timedelta(minutes=5)
    )

    _blocked_on_incident_then_completes(
        lambda: store.save_workflow_and_incident_if_leased(
            copy_model(claimed, completed_step_indexes=[0]),
            copy_model(incident, reasons=["step done"]),
            "executor-a",
            claimed.execution_epoch,
        ),
        "inc-lo",
        ["wf-lo"],
    )

    assert store.get_workflow("wf-lo").completed_step_indexes == [0]
    assert store.get_incident("inc-lo").reasons == ["step done"]


def _create(existing_incident, existing_workflow):
    assert existing_incident is None and existing_workflow is None
    incident = fault_incident(
        "inc-lo",
        "event-1",
        state=IncidentState.ACTION_PENDING,
        workflow_request_id="wf-lo",
        fencing_token=1,
        node_ids=["node-a"],
        created_at=NOW,
        updated_at=NOW,
    )
    workflow = workflow_request(
        "wf-lo",
        "inc-lo",
        fencing_token=1,
        official_steps=STEPS,
        created_at=NOW,
        updated_at=NOW,
    )
    return incident, workflow


def _widen(existing_incident, existing_workflow):
    assert existing_incident is not None and existing_workflow is not None
    return (
        copy_model(existing_incident, node_ids=["node-a", "node-b"]),
        copy_model(
            existing_workflow,
            official_steps=[
                copy_model(step, node_ids=["node-a", "node-b"])
                for step in existing_workflow.official_steps
            ],
        ),
    )


def test_merge_replacement_workflow_locks_the_incident_first(store):
    store.merge_replacement_workflow("group-lo", "event-1", _create)

    _blocked_on_incident_then_completes(
        lambda: store.merge_replacement_workflow("group-lo", "event-2", _widen),
        "inc-lo",
        ["wf-lo"],
    )

    assert store.get_workflow("wf-lo").official_steps[0].node_ids == [
        "node-a",
        "node-b",
    ]
    assert store.get_workflow("wf-lo").merge_revision == 1


BLOCKED_ID = "wf-lo-blocked"
SUCCESSOR_ID = "wf-lo-restored"
PLAN_ID = "plan-lo"


def _restored_state(store) -> None:
    store.save_plan(
        RecoveryPlan(
            plan_id=PLAN_ID,
            incident_id="inc-lo",
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
            "inc-lo",
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
            "inc-lo",
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
            "inc-lo",
            "event-lo",
            state=IncidentState.RECOVERED,
            workflow_request_id=SUCCESSOR_ID,
            fencing_token=7,
            created_at=NOW - timedelta(hours=2),
            updated_at=NOW - timedelta(minutes=30),
        )
    )


def test_reconcile_restored_workflow_locks_the_incident_first(store):
    _restored_state(store)

    updated, incident, plan = _blocked_on_incident_then_completes(
        lambda: store.reconcile_restored_workflow(
            BLOCKED_ID,
            SUCCESSOR_ID,
            expected_fencing_token=7,
            expected_execution_epoch=2,
            reference="CHG-LOCK-ORDER",
            reconciled_at=NOW,
        ),
        "inc-lo",
        [BLOCKED_ID, SUCCESSOR_ID],
    )

    assert updated.status is WorkflowStatus.SUPERSEDED
    assert plan.reconciliation_reference == "CHG-LOCK-ORDER"
    assert store.get_workflow(BLOCKED_ID).status is WorkflowStatus.SUPERSEDED


def _retired_state(store) -> None:
    incident = fault_incident(
        "inc-lo",
        "event-lo",
        state=IncidentState.ACTION_PENDING,
        workflow_request_id="wf-lo-current",
        fencing_token=4,
        created_at=NOW,
        updated_at=NOW,
    )
    retired = workflow_request(
        "wf-lo-retired",
        "inc-lo",
        status=WorkflowStatus.PENDING,
        fencing_token=1,
        official_steps=STEPS,
        created_at=NOW,
        updated_at=NOW,
    )
    current = workflow_request(
        "wf-lo-current",
        "inc-lo",
        status=WorkflowStatus.PENDING,
        fencing_token=4,
        official_steps=STEPS,
        created_at=NOW + timedelta(minutes=1),
        updated_at=NOW + timedelta(minutes=1),
    )
    store.save_incident_and_workflow(incident, current)
    store.save_workflow(retired)


def test_reconcile_retired_generation_locks_the_incident_first(store):
    _retired_state(store)

    revoked, _ = _blocked_on_incident_then_completes(
        lambda: store.reconcile_retired_generation_workflow(
            "wf-lo-retired",
            "wf-lo-current",
            expected_fencing_token=1,
            reference="CHG-RETIRED",
            reconciled_at=NOW + timedelta(minutes=2),
        ),
        "inc-lo",
        ["wf-lo-retired", "wf-lo-current"],
    )

    assert revoked.status is WorkflowStatus.SUPERSEDED
    assert store.get_workflow("wf-lo-retired").preempted_by_workflow_id == (
        "wf-lo-current"
    )


def test_a_workflow_reparented_while_its_incident_was_locked_is_a_stale_write(store):
    """The reconcile learns the incident from an unlocked read; if the row is
    moved under another incident before the workflow lock lands, the pointer
    check refuses rather than reconciling against the wrong incident."""

    _restored_state(store)
    store.save_incident(
        fault_incident(
            "inc-other",
            "event-other",
            state=IncidentState.RECOVERED,
            workflow_request_id=SUCCESSOR_ID,
            fencing_token=7,
        )
    )
    connection = _hold_incident_row("inc-lo")
    outcome: dict[str, object] = {}

    def run() -> None:
        try:
            store.reconcile_restored_workflow(
                BLOCKED_ID,
                SUCCESSOR_ID,
                expected_fencing_token=7,
                expected_execution_epoch=2,
                reference="CHG-REPARENT",
                reconciled_at=NOW,
            )
        except Exception as exc:  # noqa: BLE001 - reported to the test thread
            outcome["error"] = exc

    worker = threading.Thread(target=run)
    worker.start()
    try:
        worker.join(0.5)
        assert worker.is_alive(), "the reconcile did not wait for the incident row"
        with connection.cursor() as cursor:
            cursor.execute(
                """
                UPDATE gpu_fault_objects
                SET payload=jsonb_set(payload, '{incident_id}', to_jsonb(%s::text))
                WHERE kind='workflow' AND key=%s
                """,
                ("inc-other", BLOCKED_ID),
            )
        connection.commit()
    finally:
        connection.close()
    worker.join(30)
    assert not worker.is_alive(), (
        "the store call never finished after the incident lock was released"
    )

    assert isinstance(outcome.get("error"), StaleWriteError), outcome
    assert "inc-other" in str(outcome["error"])
    assert store.get_workflow(BLOCKED_ID).status is WorkflowStatus.BLOCKED
    assert store.get_plan(PLAN_ID).reconciliation_reference is None

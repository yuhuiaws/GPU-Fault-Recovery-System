"""``reconcile_restored_workflow`` compares on the execution epoch (F-K1).

The approved plan digest covers ``fencing_token`` and ``execution_epoch`` and
deliberately not ``updated_at`` (P0-72A): a heartbeat or a merge restamps the
latter without changing anything the verdict read. The Store's compare-and-set
must therefore be on the epoch the approval bound, so that a stale epoch is
refused and nothing is written, while a moved ``updated_at`` alone is not a
reason to refuse.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

import pytest

from gpu_fault.models import (
    IncidentState,
    PlanStatus,
    RecoveryPlan,
    WorkflowOperation,
    WorkflowStatus,
)
from gpu_fault.store import SqliteStore
from gpu_fault.store.shared.errors import StaleWriteError
from tests._builders import build_store, fault_incident, workflow_request, workflow_step
from tests.store._postgres_processor_claim_support import (
    _truncate,
    postgres_store_instance,
)

NOW = datetime(2026, 9, 6, 10, 0, tzinfo=timezone.utc)
INCIDENT_ID = "incident-epoch-cas"
BLOCKED_ID = "workflow-epoch-cas-blocked"
SUCCESSOR_ID = "workflow-epoch-cas-restored"
PLAN_ID = "plan-epoch-cas"


@pytest.fixture(params=["memory", "sqlite", "postgres"])
def store(request, tmp_path):
    if request.param == "memory":
        yield build_store()
        return
    if request.param == "sqlite":
        sqlite = SqliteStore(str(tmp_path / "epoch-cas.db"))
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
            "event-epoch-cas",
            state=IncidentState.RECOVERED,
            workflow_request_id=SUCCESSOR_ID,
            fencing_token=7,
            created_at=NOW - timedelta(hours=2),
            updated_at=NOW - timedelta(minutes=30),
        )
    )


def test_a_stale_execution_epoch_is_refused_and_writes_nothing(store) -> None:
    _restored_state(store)
    before = store.get_workflow(BLOCKED_ID)

    with pytest.raises(
        ValueError, match=r"execution epoch changed.*expected 1.*found 2"
    ):
        store.reconcile_restored_workflow(
            BLOCKED_ID,
            SUCCESSOR_ID,
            expected_fencing_token=7,
            expected_execution_epoch=1,
            reference="CHG-STALE",
            reconciled_at=NOW,
        )

    assert store.get_workflow(BLOCKED_ID) == before
    assert store.get_workflow(BLOCKED_ID).status is WorkflowStatus.BLOCKED
    assert store.get_plan(PLAN_ID).reconciliation_reference is None
    assert "CHG-STALE" not in " ".join(store.get_incident(INCIDENT_ID).reasons)


def test_the_approved_epoch_succeeds_after_a_heartbeat_moved_updated_at(store) -> None:
    _restored_state(store)
    # A heartbeat (or a merge) touched the row after the plan was reviewed:
    # ``updated_at`` moved, the epoch the approval bound did not.
    heartbeat = store.get_workflow(BLOCKED_ID).model_copy(
        update={"updated_at": NOW - timedelta(seconds=5)}
    )
    store.save_workflow(heartbeat)

    updated, incident, plan = store.reconcile_restored_workflow(
        BLOCKED_ID,
        SUCCESSOR_ID,
        expected_fencing_token=7,
        expected_execution_epoch=2,
        reference="CHG-EPOCH",
        reconciled_at=NOW,
    )

    assert updated.status is WorkflowStatus.SUPERSEDED
    assert updated.preempted_by_workflow_id == SUCCESSOR_ID
    assert store.get_workflow(BLOCKED_ID).status is WorkflowStatus.SUPERSEDED
    assert plan.reconciliation_reference == "CHG-EPOCH"
    assert incident.workflow_request_id == SUCCESSOR_ID


def test_an_explicit_updated_at_is_still_honoured_when_given(store) -> None:
    """The optional ``updated_at`` compare stays available for callers that
    hold a fresh read; it defaults to off because the digest never covered it."""

    _restored_state(store)
    stale_read = NOW - timedelta(days=1)

    with pytest.raises(ValueError, match="updated_at changed"):
        store.reconcile_restored_workflow(
            BLOCKED_ID,
            SUCCESSOR_ID,
            expected_fencing_token=7,
            expected_execution_epoch=2,
            expected_workflow_updated_at=stale_read,
            reference="CHG-UPDATED-AT",
            reconciled_at=NOW,
        )

    assert store.get_workflow(BLOCKED_ID).status is WorkflowStatus.BLOCKED


def test_a_stale_epoch_is_a_stale_write_error(store) -> None:
    """The compare-and-set names its failure class (log §70): a caller that
    retries on ``StaleWriteError`` can tell a moved generation from a record
    that is simply ineligible."""

    _restored_state(store)

    with pytest.raises(StaleWriteError, match="execution epoch changed"):
        store.reconcile_restored_workflow(
            BLOCKED_ID,
            SUCCESSOR_ID,
            expected_fencing_token=7,
            expected_execution_epoch=1,
            reference="CHG-STALE-CLASS",
            reconciled_at=NOW,
        )
    with pytest.raises(StaleWriteError, match="fencing token changed"):
        store.reconcile_restored_workflow(
            BLOCKED_ID,
            SUCCESSOR_ID,
            expected_fencing_token=6,
            expected_execution_epoch=2,
            reference="CHG-STALE-TOKEN",
            reconciled_at=NOW,
        )
    with pytest.raises(StaleWriteError, match="updated_at changed"):
        store.reconcile_restored_workflow(
            BLOCKED_ID,
            SUCCESSOR_ID,
            expected_fencing_token=7,
            expected_execution_epoch=2,
            expected_workflow_updated_at=NOW - timedelta(days=1),
            reference="CHG-STALE-UPDATED-AT",
            reconciled_at=NOW,
        )


def test_an_ineligible_record_is_a_plain_value_error_not_a_stale_write(store) -> None:
    _restored_state(store)
    store.save_incident(
        store.get_incident(INCIDENT_ID).model_copy(
            update={"state": IncidentState.ACTION_PENDING}
        )
    )

    with pytest.raises(ValueError, match="incident is not RECOVERED") as raised:
        store.reconcile_restored_workflow(
            BLOCKED_ID,
            SUCCESSOR_ID,
            expected_fencing_token=7,
            expected_execution_epoch=2,
            reference="CHG-INELIGIBLE",
            reconciled_at=NOW,
        )
    assert not isinstance(raised.value, StaleWriteError), (
        "a validation refusal is not a stale write"
    )

"""``renew_workflow_lease`` only writes once less than half the lease is left.

Store review 2026-09-07, item F1. The executor renews before and after every
step, seconds into a 3-minute lease, and each renewal re-serialized the whole
record (step executions, events, steps) into a new heap tuple, its TOAST chunks
and every partial index entry. The validation and the read stay -- the executor
picks merges up through the row this returns -- only the write is skipped while
more than half of ``lease_duration`` remains.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from gpu_fault.models import IncidentState, WorkflowOperation
from gpu_fault.store import SqliteStore
from gpu_fault.store.shared.errors import WorkflowLeaseError
from tests._builders import build_store, fault_incident, workflow_request, workflow_step
from tests.store._postgres_processor_claim_support import (
    POSTGRES_URL,
    _truncate,
    postgres_store_instance,
)

NOW = datetime(2026, 9, 7, 12, 0, tzinfo=timezone.utc)
LEASE = timedelta(minutes=3)


@pytest.fixture(params=["memory", "sqlite", "postgres"])
def store(request, tmp_path):
    if request.param == "memory":
        yield build_store()
        return
    if request.param == "sqlite":
        sqlite = SqliteStore(str(tmp_path / "renew.db"))
        try:
            yield sqlite
        finally:
            sqlite.close()
        return
    if not POSTGRES_URL:
        pytest.skip("GPU_FAULT_TEST_POSTGRES_URL is required")
    for postgres in postgres_store_instance():
        yield postgres
    _truncate()


@pytest.fixture
def postgres_store():
    """Only the Postgres backend: the tuple-header assertion has no memory or
    SQLite counterpart, and a parametrized skip would count against the CAP-005
    zero-skip gate."""

    if not POSTGRES_URL:
        pytest.skip("GPU_FAULT_TEST_POSTGRES_URL is required")
    for postgres in postgres_store_instance():
        yield postgres
    _truncate()


def _claimed(store):
    incident = fault_incident(
        "inc-r",
        "event-r",
        state=IncidentState.ACTION_PENDING,
        workflow_request_id="wf-r",
        fencing_token=3,
    )
    workflow = workflow_request(
        "wf-r",
        "inc-r",
        fencing_token=3,
        official_steps=[workflow_step(WorkflowOperation.RESET_GPU)],
    )
    store.save_incident_and_workflow(incident, workflow)
    return store.claim_workflow("wf-r", "executor-a", 3, now=NOW, lease_duration=LEASE)


def _renew(store, claimed, at: datetime):
    return store.renew_workflow_lease(
        "wf-r", "executor-a", claimed.execution_epoch, now=at, lease_duration=LEASE
    )


def test_an_early_renewal_returns_the_row_and_writes_nothing(store) -> None:
    claimed = _claimed(store)

    renewed = _renew(store, claimed, NOW + timedelta(seconds=30))

    assert renewed == claimed
    assert renewed.execution_lease_expires_at == NOW + LEASE
    assert store.get_workflow("wf-r") == claimed


def test_a_renewal_past_the_half_life_extends(store) -> None:
    claimed = _claimed(store)
    at = NOW + timedelta(seconds=100)  # 80 s left of 180

    renewed = _renew(store, claimed, at)

    assert renewed.execution_lease_expires_at == at + LEASE
    assert store.get_workflow("wf-r").execution_lease_expires_at == at + LEASE


def test_exactly_half_remaining_extends(store) -> None:
    claimed = _claimed(store)
    at = NOW + LEASE / 2

    renewed = _renew(store, claimed, at)

    assert renewed.execution_lease_expires_at == at + LEASE


def test_an_early_renewal_still_validates_the_lease(store) -> None:
    claimed = _claimed(store)
    early = NOW + timedelta(seconds=30)

    with pytest.raises(WorkflowLeaseError):
        store.renew_workflow_lease(
            "wf-r", "executor-b", claimed.execution_epoch, now=early
        )
    with pytest.raises(WorkflowLeaseError):
        store.renew_workflow_lease(
            "wf-r", "executor-a", claimed.execution_epoch + 1, now=early
        )
    with pytest.raises(WorkflowLeaseError):
        _renew(store, claimed, NOW + LEASE)


def test_an_early_renewal_still_returns_a_merge_made_since(store) -> None:
    claimed = _claimed(store)
    store.amend_workflow("wf-r", {"blocked_reasons": ["merged"]})

    renewed = _renew(store, claimed, NOW + timedelta(seconds=30))

    assert renewed.merge_revision == claimed.merge_revision + 1
    assert renewed.blocked_reasons == ["merged"]


def _xmin(request_id: str) -> str:
    import psycopg

    with psycopg.connect(POSTGRES_URL, autocommit=True) as connection:
        row = connection.execute(
            """
            SELECT xmin::text FROM gpu_fault_objects
            WHERE kind='workflow' AND key=%s
            """,
            (request_id,),
        ).fetchone()
    assert row is not None
    return str(row[0])


def test_postgres_early_renewal_leaves_the_tuple_untouched(postgres_store) -> None:
    store = postgres_store
    claimed = _claimed(store)
    before = _xmin("wf-r")

    _renew(store, claimed, NOW + timedelta(seconds=30))
    assert _xmin("wf-r") == before, "an early renewal produced a new tuple"

    _renew(store, claimed, NOW + timedelta(seconds=100))
    assert _xmin("wf-r") != before

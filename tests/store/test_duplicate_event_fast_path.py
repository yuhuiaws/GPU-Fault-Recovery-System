"""The duplicate-event fast path repairs a stale link instead of raising (F-B7).

All three transactional ingestion entry points first ask ``incident_by_event``
whether this ``event_id`` was already handled. That branch used an unguarded
``_get`` on the incident and on its ``workflow_request_id`` pointer, so once
the pointer dangled -- the workflow row gone, the pointer never set, the
incident itself deleted -- every re-post of that event failed inside the
transaction. A re-post is exactly the retry path, so the event became a poison
message (P1-57F, P1-69D). The link is now treated as dirty: the builder runs
again and re-links the event, which is the idempotent repair.

Postgres-capable: parametrised over sqlite and Postgres (skipped without
``GPU_FAULT_TEST_POSTGRES_URL``). The in-memory store keeps its own copy of
these three methods and is out of this change's scope.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone

import pytest

from gpu_fault.models import FaultIncident, WorkflowRequest
from gpu_fault.store import SqliteStore
from tests._builders import fault_incident, workflow_request
from tests.store._postgres_processor_claim_support import (
    _truncate,
    postgres_store_instance,
)

NOW = datetime(2026, 9, 6, 10, 0, tzinfo=timezone.utc)
EVENT = "log-cluster-a-node-a-0123456789abcdef"
FIRST_EVENT = "log-cluster-a-node-a-fedcba9876543210"


@pytest.fixture(params=["sqlite", "postgres"])
def store(request, tmp_path):
    if request.param == "sqlite":
        sqlite = SqliteStore(str(tmp_path / "fast-path.db"))
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


def _pair(suffix: str, event_id: str = EVENT) -> tuple[FaultIncident, WorkflowRequest]:
    incident = fault_incident(
        f"inc-{suffix}",
        event_id,
        workflow_request_id=f"wf-{suffix}",
        created_at=NOW,
        updated_at=NOW,
    )
    workflow = workflow_request(
        f"wf-{suffix}",
        incident.incident_id,
        fencing_token=1,
        created_at=NOW,
        updated_at=NOW,
    )
    return incident, workflow


class _Builder:
    """Counts how often ingestion had to build, and with what."""

    def __init__(self, suffix: str, event_id: str = EVENT) -> None:
        self.suffix = suffix
        self.event_id = event_id
        self.calls: list[tuple[FaultIncident | None, WorkflowRequest | None]] = []

    def plain(self) -> tuple[FaultIncident, WorkflowRequest]:
        self.calls.append((None, None))
        return _pair(self.suffix, self.event_id)

    def merge(
        self, incident: FaultIncident | None, workflow: WorkflowRequest | None
    ) -> tuple[FaultIncident, WorkflowRequest]:
        self.calls.append((incident, workflow))
        return _pair(self.suffix, self.event_id)


def _seed_incident_without_workflow(store) -> FaultIncident:
    incident, _ = _pair("stale")
    incident = incident.model_copy(update={"workflow_request_id": None})
    store.save_incident(incident)
    store.link_event_to_incident(EVENT, incident.incident_id)
    return incident


def _seed_incident_with_missing_workflow(store) -> FaultIncident:
    incident, _ = _pair("stale")
    store.save_incident(incident)  # ``wf-stale`` is never written
    store.link_event_to_incident(EVENT, incident.incident_id)
    return incident


def _dangle(store, incident: FaultIncident) -> FaultIncident:
    """Points the incident at a workflow that does not exist.

    The shape a cross-group rewrite or a failed ``_put`` leaves behind
    (P0-57A); the workflow row itself may or may not still be there.
    """

    dangling = incident.model_copy(update={"workflow_request_id": "wf-vanished"})
    store.save_incident(dangling)
    return dangling


# --- create_incident_workflow_if_absent ------------------------------------


def test_create_if_absent_rebuilds_when_the_incident_has_no_workflow_pointer(
    store,
) -> None:
    _seed_incident_without_workflow(store)
    builder = _Builder("fresh")

    incident, workflow, created = store.create_incident_workflow_if_absent(
        EVENT, builder.plain
    )

    assert created is True
    assert len(builder.calls) == 1
    assert (incident.incident_id, workflow.request_id) == ("inc-fresh", "wf-fresh")
    assert store.get_incident_by_event(EVENT) == incident
    assert store.get_workflow("wf-fresh") == workflow
    assert store.stale_event_link_repairs == 1


def test_create_if_absent_rebuilds_when_the_workflow_row_is_missing(store) -> None:
    _seed_incident_with_missing_workflow(store)
    builder = _Builder("fresh")

    incident, workflow, created = store.create_incident_workflow_if_absent(
        EVENT, builder.plain
    )

    assert created is True
    assert store.get_incident_by_event(EVENT) == incident
    assert store.get_workflow(workflow.request_id) == workflow


def test_create_if_absent_returns_a_healthy_duplicate_without_building(store) -> None:
    incident, workflow = _pair("stored")
    store.save_incident_and_workflow(incident, workflow)
    builder = _Builder("fresh")

    result = store.create_incident_workflow_if_absent(EVENT, builder.plain)

    assert result == (incident, workflow, False)
    assert builder.calls == []
    assert store.stale_event_link_repairs == 0


# --- merge_attempt_fault_workflow -------------------------------------------


def test_merge_attempt_fault_rebuilds_when_the_workflow_row_is_missing(store) -> None:
    _seed_incident_with_missing_workflow(store)
    builder = _Builder("fresh")

    incident, workflow = store.merge_attempt_fault_workflow(
        "group-a", EVENT, builder.merge
    )

    # No group link yet, so the builder starts from nothing and its result is
    # what gets stored and linked -- the same thing a first ingestion does.
    assert builder.calls == [(None, None)]
    assert (incident.incident_id, workflow.request_id) == ("inc-fresh", "wf-fresh")
    assert store.get_incident_by_event(EVENT) == incident
    assert store.get_workflow("wf-fresh") == workflow
    assert store.stale_event_link_repairs == 1


def test_merge_attempt_fault_hands_the_group_incident_to_the_builder_when_its_workflow_is_gone(
    store,
) -> None:
    # The stale duplicate *is* the group's incident: the builder must receive
    # it (with ``workflow=None``) so the fault stays in its group instead of
    # opening a parallel one. The group is established by a real first
    # ingestion, then its incident is left pointing at nothing.
    first = _Builder("stale", FIRST_EVENT)
    stale, _ = store.merge_attempt_fault_workflow("group-a", FIRST_EVENT, first.merge)
    store.link_event_to_incident(EVENT, stale.incident_id)
    dangling = _dangle(store, stale)
    builder = _Builder("fresh")

    store.merge_attempt_fault_workflow("group-a", EVENT, builder.merge)

    assert builder.calls == [(dangling, None)]


def test_merge_attempt_fault_returns_a_healthy_duplicate_without_building(
    store,
) -> None:
    incident, workflow = _pair("stored")
    store.save_incident_and_workflow(incident, workflow)
    builder = _Builder("fresh")

    result = store.merge_attempt_fault_workflow("group-a", EVENT, builder.merge)

    assert result == (incident, workflow)
    assert builder.calls == []


# --- merge_replacement_workflow ---------------------------------------------


def test_merge_replacement_rebuilds_when_the_workflow_pointer_dangles(store) -> None:
    incident, workflow = _pair("stale")
    store.save_incident_and_workflow(incident, workflow)
    _dangle(store, incident)
    builder = _Builder("fresh")

    rebuilt_incident, rebuilt_workflow = store.merge_replacement_workflow(
        "group-a", EVENT, builder.merge
    )

    assert builder.calls == [(None, None)]
    assert store.get_incident_by_event(EVENT) == rebuilt_incident
    assert store.get_workflow(rebuilt_workflow.request_id) == rebuilt_workflow
    assert store.stale_event_link_repairs == 1


def test_merge_replacement_rebuilds_when_the_incident_has_no_workflow_pointer(
    store,
) -> None:
    _seed_incident_without_workflow(store)
    builder = _Builder("fresh")

    incident, workflow = store.merge_replacement_workflow(
        "group-a", EVENT, builder.merge
    )

    assert len(builder.calls) == 1
    assert store.get_incident_by_event(EVENT) == incident


def test_merge_replacement_returns_a_healthy_duplicate_without_building(store) -> None:
    incident, workflow = _pair("stored")
    store.save_incident_and_workflow(incident, workflow)
    builder = _Builder("fresh")

    result = store.merge_replacement_workflow("group-a", EVENT, builder.merge)

    assert result == (incident, workflow)
    assert builder.calls == []


def test_a_re_posted_event_is_no_longer_a_poison_message(store) -> None:
    # The sequence the review described: first post succeeds, the pointer
    # dangles, and from then on every re-post failed. Two re-posts must both
    # succeed and agree.
    builder = _Builder("first")
    incident, _ = store.merge_attempt_fault_workflow("group-a", EVENT, builder.merge)
    _dangle(store, incident)

    repaired = _Builder("second")
    first_retry = store.merge_attempt_fault_workflow("group-a", EVENT, repaired.merge)
    second_retry = store.merge_attempt_fault_workflow(
        "group-a", EVENT, _Builder("third").merge
    )

    assert first_retry == second_retry
    assert first_retry[1].request_id == "wf-second"
    assert store.stale_event_link_repairs == 1

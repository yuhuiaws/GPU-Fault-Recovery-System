"""The duplicate-event fast path is one contract across the three backends.

Control-plane review 2026-09-08, C-05 (closing F-B7 "未做 1"). The shared
``TransactionalWorkflowMixin`` repaired a dangling ``incident_by_event`` link
by rebuilding (P1-57F, P1-69D), but the in-memory store kept its own copies of
the three fast paths: ``create_incident_workflow_if_absent`` raised
``RuntimeError`` on an incident without a workflow pointer, and both
``merge_*`` duplicate branches read the workflow unguarded. Every family test
that runs on ``InMemoryStore`` -- most of them -- therefore could not pin the
behaviour PostgreSQL has, and the ``--dry-run`` adapter path saw a poison
message. This file parametrises the same scenarios over memory, SQLite and
PostgreSQL so the three answers cannot drift again.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone

import pytest

from gpu_fault.models import FaultIncident, WorkflowRequest
from gpu_fault.store import SqliteStore
from tests._builders import build_store, fault_incident, workflow_request
from tests.store._postgres_processor_claim_support import (
    _truncate,
    postgres_store_instance,
)

NOW = datetime(2026, 9, 8, 10, 0, tzinfo=timezone.utc)
EVENT = "log-cluster-a-node-a-0123456789abcdef"


@pytest.fixture(params=["memory", "sqlite", "postgres"])
def store(request, tmp_path):
    if request.param == "memory":
        yield build_store()
        return
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


def _pair(suffix: str) -> tuple[FaultIncident, WorkflowRequest]:
    incident = fault_incident(
        f"inc-{suffix}",
        EVENT,
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
    def __init__(self, suffix: str) -> None:
        self.suffix = suffix
        self.calls: list[object] = []

    def plain(self) -> tuple[FaultIncident, WorkflowRequest]:
        self.calls.append((None, None))
        return _pair(self.suffix)

    def merge(self, incident, workflow) -> tuple[FaultIncident, WorkflowRequest]:
        self.calls.append((incident, workflow))
        return _pair(self.suffix)


def _dangling_pointer(store) -> None:
    incident, _ = _pair("stale")
    store.save_incident(incident)  # ``wf-stale`` is never written
    store.link_event_to_incident(EVENT, incident.incident_id)


def _no_pointer(store) -> None:
    incident, _ = _pair("stale")
    store.save_incident(incident.model_copy(update={"workflow_request_id": None}))
    store.link_event_to_incident(EVENT, incident.incident_id)


ENTRY_POINTS = [
    pytest.param(
        lambda store, builder: store.create_incident_workflow_if_absent(
            EVENT, builder.plain
        )[:2],
        id="create_if_absent",
    ),
    pytest.param(
        lambda store, builder: store.merge_attempt_fault_workflow(
            "group-a", EVENT, builder.merge
        ),
        id="merge_attempt_fault",
    ),
    pytest.param(
        lambda store, builder: store.merge_replacement_workflow(
            "group-a", EVENT, builder.merge
        ),
        id="merge_replacement",
    ),
]


@pytest.mark.parametrize("entry", ENTRY_POINTS)
@pytest.mark.parametrize("seed", [_dangling_pointer, _no_pointer])
def test_a_dirty_link_is_rebuilt_and_counted_on_every_backend(
    store, entry, seed
) -> None:
    seed(store)
    builder = _Builder("fresh")

    incident, workflow = entry(store, builder)

    assert len(builder.calls) == 1, "the builder ran: the link was treated as dirty"
    assert (incident.incident_id, workflow.request_id) == ("inc-fresh", "wf-fresh")
    assert store.get_incident_by_event(EVENT) == incident
    assert store.get_workflow("wf-fresh") == workflow
    assert store.stale_event_link_repairs == 1


@pytest.mark.parametrize("entry", ENTRY_POINTS)
def test_a_healthy_duplicate_is_returned_without_building_on_every_backend(
    store, entry
) -> None:
    incident, workflow = _pair("stored")
    store.save_incident_and_workflow(incident, workflow)
    builder = _Builder("fresh")

    result = entry(store, builder)

    assert tuple(result) == (incident, workflow)
    assert builder.calls == []
    assert store.stale_event_link_repairs == 0


@pytest.mark.parametrize("entry", ENTRY_POINTS)
def test_a_re_posted_event_is_never_a_poison_message_on_any_backend(
    store, entry
) -> None:
    first = _Builder("first")
    incident, _ = entry(store, first)
    store.save_incident(incident.model_copy(update={"workflow_request_id": "gone"}))

    once = entry(store, _Builder("second"))
    twice = entry(store, _Builder("third"))

    assert tuple(once) == tuple(twice)
    assert once[1].request_id == "wf-second"
    assert store.stale_event_link_repairs == 1

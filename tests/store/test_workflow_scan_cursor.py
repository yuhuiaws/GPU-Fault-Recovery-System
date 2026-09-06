"""The dispatch scan orders by when a row became eligible and walks in pages.

FINAL-建议汇总 F-A2 (a) and (c) (P1-73D, P2-59F, P2-39F). ``updated_at`` was
the sort key, and every merge into a row -- ABSORB, WIDEN_IN_PLACE, a budget
refusal -- bumped it, so the node with the densest faults sorted last. The
dispatch-mode scan now orders by ``dispatch_eligible_at`` = max(``created_at``,
``not_before``), which no merge rewrites, and takes ``after`` (the last row of
the previous page) so a horizon is walked page by page instead of being
re-read from the start with a larger LIMIT.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

import pytest

from gpu_fault.models import IncidentState, WorkflowOperation, WorkflowStatus
from gpu_fault.store import SqliteStore
from tests._builders import build_store, fault_incident, workflow_request, workflow_step
from tests.store._postgres_processor_claim_support import (
    _truncate,
    postgres_store_instance,
)

NOW = datetime(2026, 9, 6, 12, 0, tzinfo=timezone.utc)
EXECUTABLE = {
    WorkflowStatus.PENDING,
    WorkflowStatus.SAFETY_PENDING,
    WorkflowStatus.RUNNING,
}


@pytest.fixture(params=["memory", "sqlite", "postgres"])
def store(request, tmp_path):
    if request.param == "memory":
        yield build_store()
        return
    if request.param == "sqlite":
        sqlite = SqliteStore(str(tmp_path / "cursor.db"))
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


def _workflow(store, request_id: str, *, created_at: datetime, **updates):
    incident = fault_incident(
        f"inc-{request_id}",
        f"event-{request_id}",
        state=IncidentState.ACTION_PENDING,
        workflow_request_id=request_id,
        created_at=created_at,
        updated_at=created_at,
    )
    workflow = workflow_request(
        request_id,
        f"inc-{request_id}",
        status=updates.pop("status", WorkflowStatus.PENDING),
        official_steps=[
            workflow_step(WorkflowOperation.FREEZE_EVIDENCE, node_ids=["node-a"])
        ],
        created_at=created_at,
        updated_at=updates.pop("updated_at", created_at),
        **updates,
    )
    store.save_incident_and_workflow(incident, workflow)
    return workflow


def _seed(store) -> None:
    # Oldest fault, but merged into again and again: updated_at is NOW.
    _workflow(store, "wf-merged", created_at=NOW - timedelta(hours=3), updated_at=NOW)
    # Untouched since it was written.
    _workflow(store, "wf-quiet", created_at=NOW - timedelta(hours=2))
    # Written first of all, but deferred: eligible only an hour ago.
    _workflow(
        store,
        "wf-deferred",
        created_at=NOW - timedelta(hours=4),
        not_before=NOW - timedelta(hours=1),
    )
    _workflow(store, "wf-young", created_at=NOW - timedelta(minutes=30))
    # Not part of the dispatch order at all.
    _workflow(
        store,
        "wf-future",
        created_at=NOW - timedelta(days=1),
        not_before=NOW + timedelta(hours=1),
    )
    _workflow(
        store,
        "wf-done",
        created_at=NOW - timedelta(days=1),
        status=WorkflowStatus.SUCCEEDED,
    )


def test_dispatch_scan_orders_by_eligibility_not_by_the_last_merge(store):
    _seed(store)

    rows = store.list_workflows(EXECUTABLE, limit=100, dispatchable_at=NOW)

    assert [item.request_id for item in rows] == [
        "wf-merged",
        "wf-quiet",
        "wf-deferred",
        "wf-young",
    ]


def test_dispatch_scan_walks_the_horizon_in_pages(store):
    _seed(store)

    first = store.list_workflows(EXECUTABLE, limit=2, dispatchable_at=NOW)
    second = store.list_workflows(
        EXECUTABLE, limit=2, dispatchable_at=NOW, after=first[-1]
    )
    third = store.list_workflows(
        EXECUTABLE, limit=2, dispatchable_at=NOW, after=second[-1]
    )

    assert [item.request_id for item in first] == ["wf-merged", "wf-quiet"]
    assert [item.request_id for item in second] == ["wf-deferred", "wf-young"]
    assert third == []


def test_the_cursor_breaks_ties_on_request_id(store):
    same = NOW - timedelta(hours=1)
    _workflow(store, "wf-b", created_at=same)
    _workflow(store, "wf-a", created_at=same)
    _workflow(store, "wf-c", created_at=same)

    first = store.list_workflows(EXECUTABLE, limit=1, dispatchable_at=NOW)
    rest = store.list_workflows(
        EXECUTABLE, limit=10, dispatchable_at=NOW, after=first[0]
    )

    assert [item.request_id for item in first] == ["wf-a"]
    assert [item.request_id for item in rest] == ["wf-b", "wf-c"]


def test_the_cursor_is_only_defined_over_the_dispatch_order(store):
    _seed(store)
    anchor = store.get_workflow("wf-quiet")

    with pytest.raises(ValueError, match="dispatchable_at"):
        store.list_workflows(EXECUTABLE, limit=10, after=anchor)


def test_the_plain_listing_still_orders_by_updated_at(store):
    """Admin listings and the other sweeps keep the historical order; only
    the dispatch-mode scan changed its key."""

    _seed(store)

    rows = store.list_workflows(EXECUTABLE, limit=100)

    assert [item.request_id for item in rows] == [
        "wf-future",
        "wf-deferred",
        "wf-quiet",
        "wf-young",
        "wf-merged",
    ]

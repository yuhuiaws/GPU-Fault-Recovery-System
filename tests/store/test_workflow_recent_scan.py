"""The ``/metrics`` detail scan reads what is in flight plus recent history,
and the recency filter runs in the store.

``MetricScanCache`` used to take the newest N terminal rows, so the scan --
and the truncation gauge the GpuFaultWorkflowMetricScanTruncated alert reads
-- grew with audit history: a removed rule's 19 699 SUCCEEDED rows pushed a
quiet fleet past the budget. ``list_recent_workflows`` returns every open
workflow whatever its age plus the terminal workflows updated inside a window,
open rows first, each half newest first, capped at ``limit``; terminal rows
older than the window are never read, on every backend.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

import pytest

from gpu_fault.models import IncidentState, WorkflowOperation, WorkflowStatus
from gpu_fault.store import SqliteStore
from gpu_fault.store.postgres.workflows import PostgresWorkflowMixin
from tests._builders import build_store, fault_incident, workflow_request, workflow_step
from tests.store._postgres_processor_claim_support import (
    _truncate,
    postgres_store_instance,
)

NOW = datetime(2026, 9, 10, 12, 0, tzinfo=timezone.utc)
OPEN = {
    WorkflowStatus.PENDING,
    WorkflowStatus.RUNNING,
    WorkflowStatus.SAFETY_PENDING,
    WorkflowStatus.BLOCKED,
}
WINDOW_EDGE = NOW - timedelta(days=7)


@pytest.fixture(params=["memory", "sqlite", "postgres"])
def store(request, tmp_path):
    if request.param == "memory":
        yield build_store()
        return
    if request.param == "sqlite":
        sqlite = SqliteStore(str(tmp_path / "recent-scan.db"))
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


def _workflow(store, request_id: str, status: WorkflowStatus, updated_at: datetime):
    steps = [workflow_step(WorkflowOperation.FREEZE_EVIDENCE, node_ids=["node-a"])]
    incident = fault_incident(
        f"inc-{request_id}",
        f"event-{request_id}",
        state=IncidentState.ACTION_PENDING,
        workflow_request_id=request_id,
        created_at=updated_at - timedelta(minutes=5),
        updated_at=updated_at,
    )
    workflow = workflow_request(
        request_id,
        f"inc-{request_id}",
        status=status,
        official_steps=steps,
        created_at=updated_at - timedelta(minutes=5),
        updated_at=updated_at,
    )
    store.save_incident_and_workflow(incident, workflow)
    return workflow


def _ids(rows) -> list[str]:
    return [row.request_id for row in rows]


def _seed(store) -> None:
    # Terminal residue well outside the window, an open row older still, and
    # recent terminal rows on both sides of the window edge.
    for index in range(5):
        _workflow(
            store,
            f"old-{index}",
            WorkflowStatus.SUCCEEDED,
            NOW - timedelta(days=20, minutes=index),
        )
    _workflow(store, "stuck", WorkflowStatus.RUNNING, NOW - timedelta(days=40))
    _workflow(store, "held", WorkflowStatus.BLOCKED, NOW - timedelta(days=9))
    _workflow(store, "edge", WorkflowStatus.FAILED, WINDOW_EDGE)
    _workflow(
        store, "just-out", WorkflowStatus.SUCCEEDED, WINDOW_EDGE - timedelta(seconds=1)
    )
    _workflow(store, "fresh-a", WorkflowStatus.SUCCEEDED, NOW - timedelta(hours=2))
    _workflow(store, "fresh-b", WorkflowStatus.SUPERSEDED, NOW - timedelta(hours=1))


def test_open_rows_of_any_age_then_terminal_rows_inside_the_window(store) -> None:
    _seed(store)

    rows = store.list_recent_workflows(OPEN, updated_since=WINDOW_EDGE, limit=100)

    # Open half first (newest first), then the windowed terminal half newest
    # first; the edge is inclusive, one second before it is out, and the
    # twenty-day-old residue never appears.
    assert _ids(rows) == ["held", "stuck", "fresh-b", "fresh-a", "edge"]


def test_window_none_reads_the_newest_rows_after_the_open_ones(store) -> None:
    _seed(store)

    rows = store.list_recent_workflows(OPEN, updated_since=None, limit=6)

    assert _ids(rows) == ["held", "stuck", "fresh-b", "fresh-a", "edge", "just-out"]


def test_the_cap_falls_on_terminal_rows_before_open_ones(store) -> None:
    _seed(store)

    rows = store.list_recent_workflows(OPEN, updated_since=WINDOW_EDGE, limit=3)

    assert _ids(rows) == ["held", "stuck", "fresh-b"]
    assert store.list_recent_workflows(OPEN, updated_since=WINDOW_EDGE, limit=0) == []


def test_an_empty_open_set_is_purely_the_window(store) -> None:
    _seed(store)

    rows = store.list_recent_workflows(set(), updated_since=WINDOW_EDGE, limit=100)

    assert _ids(rows) == ["fresh-b", "fresh-a", "edge"]


def test_postgres_scan_is_two_status_bounded_range_scans() -> None:
    """G-2: never ``statuses=None``; the recent half carries the window edge as
    a text bound on ``payload->>'updated_at'`` so ``gpu_fault_workflow_updated_all``
    can serve it as a range scan that stops at the edge."""

    sql, parameters = PostgresWorkflowMixin.workflow_scan_query(
        {WorkflowStatus.SUCCEEDED, WorkflowStatus.FAILED},
        limit=7,
        newest_first=True,
        updated_since=WINDOW_EDGE,
    )

    assert "w.payload->>'status' IN ('FAILED', 'SUCCEEDED')" in sql
    assert "w.payload->>'updated_at' >= %s" in sql
    assert "ORDER BY w.payload->>'updated_at' DESC, w.key DESC LIMIT %s" in sql
    assert parameters == ("2026-09-03T12:00:00.000000Z", 7)

    without_window, parameters = PostgresWorkflowMixin.workflow_scan_query(
        {WorkflowStatus.SUCCEEDED}, limit=7, newest_first=True
    )
    assert "updated_at' >=" not in without_window
    assert parameters == (7,)

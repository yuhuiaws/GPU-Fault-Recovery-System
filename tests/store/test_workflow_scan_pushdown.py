"""The dispatcher's permanent filters run in the store, not after the LIMIT.

FINAL-建议汇总 F-A2(b) (P0-79B, P0-39B, P1-70G, P2-80C). ``list_workflows``
returned the oldest N executable rows and the dispatcher then discarded the
ones with a future ``not_before``, an open predecessor, or a retired
generation -- so those rows occupied the *front* of every window. The three
filters are now optional pushdowns on the scan itself, and the reasons the
dispatch report needs come from a separate aggregate that is only asked for
when the tick came up short.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

import pytest

from gpu_fault.models import (
    BlockedKind,
    IncidentState,
    WorkflowOperation,
    WorkflowStatus,
)
from gpu_fault.store import SqliteStore
from tests._builders import build_store, fault_incident, workflow_request, workflow_step
from tests.store._postgres_processor_claim_support import (
    _truncate,
    postgres_store_instance,
)

NOW = datetime(2026, 9, 6, 0, 0, tzinfo=timezone.utc)
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
        sqlite = SqliteStore(str(tmp_path / "pushdown.db"))
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


def _workflow(store, request_id: str, index: int, **updates):
    steps = [workflow_step(WorkflowOperation.FREEZE_EVIDENCE, node_ids=["node-a"])]
    at = NOW + timedelta(minutes=index)
    incident = fault_incident(
        f"inc-{request_id}",
        f"event-{request_id}",
        state=IncidentState.ACTION_PENDING,
        workflow_request_id=request_id,
        created_at=at,
        updated_at=at,
    )
    workflow = workflow_request(
        request_id,
        f"inc-{request_id}",
        status=updates.pop("status", WorkflowStatus.PENDING),
        official_steps=steps,
        created_at=at,
        updated_at=at,
        **updates,
    )
    store.save_incident_and_workflow(incident, workflow)
    return workflow


def _seed(store) -> None:
    _workflow(store, "wf-plain", 0)
    _workflow(store, "wf-deferred", 1, not_before=NOW + timedelta(hours=1))
    _workflow(store, "wf-due", 2, not_before=NOW - timedelta(seconds=1))
    _workflow(store, "wf-open-pred", 3, status=WorkflowStatus.RUNNING)
    _workflow(store, "wf-behind-open", 4, predecessor_workflow_id="wf-open-pred")
    _workflow(store, "wf-done-pred", 5, status=WorkflowStatus.SUCCEEDED)
    _workflow(store, "wf-behind-done", 6, predecessor_workflow_id="wf-done-pred")
    _workflow(store, "wf-retired", 7)
    _workflow(store, "wf-old-terminal", 8, status=WorkflowStatus.FAILED)


def test_scan_pushes_the_permanent_filters_below_the_limit(store):
    _seed(store)

    rows = store.list_workflows(
        EXECUTABLE, limit=100, dispatchable_at=NOW, exclude_request_ids={"wf-retired"}
    )

    assert [item.request_id for item in rows] == [
        "wf-plain",
        "wf-due",
        "wf-open-pred",
        "wf-behind-done",
    ]


def test_scan_limit_applies_after_the_pushdown(store):
    _seed(store)

    rows = store.list_workflows(
        EXECUTABLE, limit=2, dispatchable_at=NOW, exclude_request_ids={"wf-retired"}
    )

    # Without the pushdown the window of two would have been the plain row and
    # the deferred row, and the due row would have been invisible.
    assert [item.request_id for item in rows] == ["wf-plain", "wf-due"]


def test_scan_without_pushdown_is_unchanged(store):
    _seed(store)

    rows = store.list_workflows(EXECUTABLE, limit=100)

    assert len(rows) == 7
    assert {item.request_id for item in rows} >= {"wf-deferred", "wf-behind-open"}


def test_held_counts_name_each_pushed_down_reason(store):
    _seed(store)

    held = store.count_held_workflows(
        EXECUTABLE, dispatchable_at=NOW, exclude_request_ids={"wf-retired"}
    )

    assert held == {"not_before": 1, "predecessor": 1, "retired": 1}


# ---------------------------------------------------------------- F-A4
# A BLOCKED predecessor is only "out of the way" when its safety plan
# settled. One that is waiting for an operator (or hit an internal error)
# still owns its node: the successor must wait, exactly as behind RUNNING.


def _seed_blocked_predecessors(store) -> None:
    _workflow(
        store,
        "wf-settled",
        0,
        status=WorkflowStatus.BLOCKED,
        blocked_kind=BlockedKind.SAFETY_SETTLED,
    )
    _workflow(store, "wf-behind-settled", 1, predecessor_workflow_id="wf-settled")
    _workflow(
        store,
        "wf-operator",
        2,
        status=WorkflowStatus.BLOCKED,
        blocked_kind=BlockedKind.NEEDS_OPERATOR,
    )
    _workflow(store, "wf-behind-operator", 3, predecessor_workflow_id="wf-operator")
    _workflow(
        store,
        "wf-internal",
        4,
        status=WorkflowStatus.BLOCKED,
        blocked_kind=BlockedKind.INTERNAL_ERROR,
    )
    _workflow(store, "wf-behind-internal", 5, predecessor_workflow_id="wf-internal")
    # A legacy BLOCKED row without a kind keeps the old semantics: it lets
    # its successor through, so the four production rows written before
    # ``blocked_kind`` existed cannot pin anything forever.
    _workflow(store, "wf-legacy", 6, status=WorkflowStatus.BLOCKED)
    _workflow(store, "wf-behind-legacy", 7, predecessor_workflow_id="wf-legacy")


def test_only_a_settled_blocked_predecessor_releases_its_successor(store):
    _seed_blocked_predecessors(store)

    rows = store.list_workflows(EXECUTABLE, limit=100, dispatchable_at=NOW)

    assert [item.request_id for item in rows] == [
        "wf-behind-settled",
        "wf-behind-legacy",
    ]
    assert store.count_held_workflows(EXECUTABLE, dispatchable_at=NOW) == {
        "predecessor": 2
    }

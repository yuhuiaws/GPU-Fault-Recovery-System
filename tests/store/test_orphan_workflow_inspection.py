"""Read-only orphan inspection across the three backends (F-B3 (4) / Q-ORPHAN).

An orphan is a PENDING / SAFETY_PENDING workflow that nothing will ever act on:
its incident is gone or names a different workflow, and no other workflow names
it as predecessor. The dispatcher hands one workflow per incident, the fences
and both sweepers require a strictly higher generation, so such a record sits
forever -- the "aggregated but never handled" shape from the causal chain 4 of
FINAL-建议汇总. The mirror image is an incident whose ``workflow_request_id``
names a workflow row that does not exist. Both are pure reads; the gauge in
``closed_loop_metric_lines`` publishes their counts.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from gpu_fault.app import ApplicationContext
from gpu_fault.app.builtin_metric_contributors import closed_loop_metric_lines
from gpu_fault.models import IncidentState, WorkflowStatus
from gpu_fault.store import SqliteStore
from tests._builders import build_store, fault_incident, workflow_request
from tests.store._postgres_processor_claim_support import (
    _truncate,
    postgres_store_instance,
)

NOW = datetime(2026, 9, 5, 15, 0, tzinfo=timezone.utc)
HOUR_AGO = NOW - timedelta(hours=1)
TWO_HOURS_AGO = NOW - timedelta(hours=2)
CUTOFF = NOW - timedelta(minutes=10)


@pytest.fixture(params=["memory", "sqlite", "postgres"])
def store(request, tmp_path):
    if request.param == "memory":
        yield build_store()
        return
    if request.param == "sqlite":
        sqlite = SqliteStore(str(tmp_path / "orphans.db"))
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


def _incident(incident_id: str, *, points_at: str | None, token: int = 2):
    return fault_incident(
        incident_id,
        f"event-{incident_id}",
        state=IncidentState.ACTION_PENDING,
        workflow_request_id=points_at,
        fencing_token=token,
        created_at=HOUR_AGO,
        updated_at=HOUR_AGO,
    )


def _workflow(
    request_id: str,
    incident_id: str,
    *,
    status: WorkflowStatus = WorkflowStatus.PENDING,
    token: int = 2,
    created_at: datetime = HOUR_AGO,
    **values,
):
    return workflow_request(
        request_id,
        incident_id,
        status=status,
        fencing_token=token,
        created_at=created_at,
        updated_at=created_at,
        **values,
    )


def _ids(workflows) -> list[str]:
    return [workflow.request_id for workflow in workflows]


def test_a_same_generation_twin_nobody_names_is_an_orphan(store) -> None:
    store.save_incident_and_workflow(
        _incident("inc-twin", points_at="wf-b"), _workflow("wf-b", "inc-twin")
    )
    store.save_workflow(_workflow("wf-a", "inc-twin"))

    assert _ids(store.list_orphan_workflows(created_before=CUTOFF)) == ["wf-a"]


def test_a_queued_successors_pending_predecessor_is_not_an_orphan(store) -> None:
    store.save_incident_and_workflow(
        _incident("inc-queue", points_at="wf-s"),
        _workflow("wf-s", "inc-queue", predecessor_workflow_id="wf-p"),
    )
    store.save_workflow(_workflow("wf-p", "inc-queue"))

    assert store.list_orphan_workflows(created_before=CUTOFF) == []


def test_a_workflow_whose_incident_is_gone_is_an_orphan(store) -> None:
    store.save_workflow(_workflow("wf-dangling", "inc-missing"))

    assert _ids(store.list_orphan_workflows(created_before=CUTOFF)) == ["wf-dangling"]


def test_only_the_two_waiting_statuses_and_only_past_the_window_count(store) -> None:
    store.save_incident_and_workflow(
        _incident("inc-a", points_at="wf-a-live"), _workflow("wf-a-live", "inc-a")
    )
    # Same shape as a twin, but RUNNING: the executor owns it.
    store.save_workflow(
        _workflow("wf-a-running", "inc-a", status=WorkflowStatus.RUNNING)
    )
    # Same shape as a twin, SAFETY_PENDING: still waiting, so an orphan.
    store.save_workflow(
        _workflow("wf-a-safety", "inc-a", status=WorkflowStatus.SAFETY_PENDING)
    )
    # Fresh twin inside the aggregation window: normal churn, not an orphan yet.
    store.save_workflow(_workflow("wf-a-fresh", "inc-a", created_at=NOW))
    # The incident's own workflow is never an orphan.
    store.save_incident_and_workflow(
        _incident("inc-b", points_at="wf-b"), _workflow("wf-b", "inc-b")
    )

    assert _ids(store.list_orphan_workflows(created_before=CUTOFF)) == ["wf-a-safety"]
    assert _ids(
        store.list_orphan_workflows(created_before=NOW + timedelta(seconds=1))
    ) == ["wf-a-safety", "wf-a-fresh"]


def test_orphans_come_oldest_first_and_honour_the_limit(store) -> None:
    store.save_incident_and_workflow(
        _incident("inc-c", points_at="wf-c"), _workflow("wf-c", "inc-c")
    )
    store.save_workflow(_workflow("wf-c-older", "inc-c", created_at=TWO_HOURS_AGO))
    store.save_workflow(_workflow("wf-c-newer", "inc-c", created_at=HOUR_AGO))

    assert _ids(store.list_orphan_workflows(created_before=CUTOFF)) == [
        "wf-c-older",
        "wf-c-newer",
    ]
    assert _ids(store.list_orphan_workflows(created_before=CUTOFF, limit=1)) == [
        "wf-c-older"
    ]


def test_an_incident_pointing_at_a_missing_workflow_is_reported(store) -> None:
    store.save_incident(_incident("inc-missing-wf", points_at="wf-vanished"))
    store.save_incident_and_workflow(
        _incident("inc-ok", points_at="wf-ok"), _workflow("wf-ok", "inc-ok")
    )
    store.save_incident(_incident("inc-unlinked", points_at=None))

    dangling = store.list_incidents_with_missing_workflow()

    assert [incident.incident_id for incident in dangling] == ["inc-missing-wf"]
    assert store.list_incidents_with_missing_workflow(limit=0) == []


def test_the_closed_loop_gauges_publish_both_counts(store) -> None:
    store.save_incident_and_workflow(
        _incident("inc-twin", points_at="wf-b"), _workflow("wf-b", "inc-twin")
    )
    store.save_workflow(_workflow("wf-a", "inc-twin"))
    store.save_workflow(_workflow("wf-dangling", "inc-missing"))
    store.save_incident(_incident("inc-missing-wf", points_at="wf-vanished"))

    lines = closed_loop_metric_lines(
        SimpleNamespace(context=ApplicationContext(store=store))
    )

    assert "gpu_fault_orphan_workflows 2" in lines
    assert "gpu_fault_incident_dangling_workflow_pointers 1" in lines

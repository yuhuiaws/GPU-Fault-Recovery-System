"""``list_active_workflow_incidents`` is bounded and deterministically ordered.

FINAL-建议汇总 F-J5 (P1-78C, P1-79D). The ingest path calls this jsonb
self-join twice per fault group while holding the aggregation lock; without a
LIMIT the read grows with whatever is active in the cluster. Newest first is
the order both callers already relied on implicitly.
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
from gpu_fault.store.contracts import ACTIVE_WORKFLOW_INCIDENTS_LIMIT
from tests._builders import build_store, fault_incident, workflow_request, workflow_step
from tests.store._postgres_processor_claim_support import (
    _truncate,
    postgres_store_instance,
)

NOW = datetime(2026, 9, 5, 23, 0, tzinfo=timezone.utc)


@pytest.fixture(params=["memory", "sqlite", "postgres"])
def store(request, tmp_path):
    if request.param == "memory":
        yield build_store()
        return
    if request.param == "sqlite":
        sqlite = SqliteStore(str(tmp_path / "bound.db"))
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


def _seed(store, count: int) -> list[str]:
    steps = [workflow_step(WorkflowOperation.FREEZE_EVIDENCE, node_ids=["node-a"])]
    ids: list[str] = []
    for index in range(count):
        at = NOW + timedelta(minutes=index)
        incident = fault_incident(
            f"inc-{index}",
            f"event-{index}",
            state=IncidentState.ACTION_PENDING,
            workflow_request_id=f"wf-{index}",
            created_at=at,
            updated_at=at,
        )
        workflow = workflow_request(
            f"wf-{index}",
            f"inc-{index}",
            status=WorkflowStatus.PENDING if index % 2 else WorkflowStatus.RUNNING,
            official_steps=steps,
            created_at=at,
            updated_at=at,
        )
        store.save_incident_and_workflow(incident, workflow)
        ids.append(workflow.request_id)
    return ids


def test_active_workflow_incidents_are_newest_first_and_bounded(store):
    ids = _seed(store, 5)
    cluster_id = store.get_incident("inc-0").cluster_id

    unbounded = store.list_active_workflow_incidents(cluster_id, node_ids={"node-a"})
    bounded = store.list_active_workflow_incidents(
        cluster_id, node_ids={"node-a"}, limit=2
    )

    assert [workflow.request_id for _, workflow in unbounded] == ids[::-1]
    assert [workflow.request_id for _, workflow in bounded] == ids[::-1][:2]
    assert all(
        incident.incident_id == workflow.incident_id for incident, workflow in bounded
    ), (
        "expected all( incident.incident_id == workflow.incident_id for incident, workflow in bounded ) to be true"
    )


def test_active_workflow_incidents_default_bound_is_finite(store):
    _seed(store, 3)
    cluster_id = store.get_incident("inc-0").cluster_id

    rows = store.list_active_workflow_incidents(cluster_id)

    assert len(rows) == 3
    # The default bound is large enough for any real cluster's active set but
    # is a bound nonetheless: passing it explicitly returns the same rows.
    assert (
        store.list_active_workflow_incidents(
            cluster_id, limit=ACTIVE_WORKFLOW_INCIDENTS_LIMIT
        )
        == rows
    )


def test_an_operator_blocked_workflow_is_still_active_for_node_exclusivity(store):
    """F-A4: a BLOCKED workflow that is waiting for an operator has touched
    the node and nobody has undone that; the node-exclusive check must see
    it. A settled safety plan (and a legacy row without a kind) is done."""

    steps = [workflow_step(WorkflowOperation.RESTART_NODE, node_ids=["node-a"])]
    for index, (name, kind) in enumerate(
        [
            ("operator", BlockedKind.NEEDS_OPERATOR),
            ("internal", BlockedKind.INTERNAL_ERROR),
            ("settled", BlockedKind.SAFETY_SETTLED),
            ("legacy", None),
        ]
    ):
        at = NOW + timedelta(minutes=index)
        incident = fault_incident(
            f"inc-{name}",
            f"event-{name}",
            state=IncidentState.QUARANTINED,
            workflow_request_id=f"wf-{name}",
            created_at=at,
            updated_at=at,
        )
        workflow = workflow_request(
            f"wf-{name}",
            f"inc-{name}",
            status=WorkflowStatus.BLOCKED,
            blocked_kind=kind,
            official_steps=steps,
            created_at=at,
            updated_at=at,
        )
        store.save_incident_and_workflow(incident, workflow)
    cluster_id = store.get_incident("inc-operator").cluster_id

    rows = store.list_active_workflow_incidents(cluster_id, node_ids={"node-a"})

    assert sorted(workflow.request_id for _, workflow in rows) == [
        "wf-internal",
        "wf-operator",
    ]

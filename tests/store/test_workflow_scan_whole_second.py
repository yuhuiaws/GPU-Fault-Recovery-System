"""A whole-second ``not_before`` sorts and pages where time puts it.

Store review 2026-09-07, item E. The dispatch scan orders and cuts on
``GREATEST(payload->>'created_at', payload->>'not_before')`` as TEXT so an
expression index serves it. Pydantic used to render a whole second as
``...:00Z`` and everything else as ``...:00.xxxxxxZ``; ``'Z'`` sorts after
``'.'``, so the whole second came after every fraction of its own second, the
``not_before <= now`` cut admitted fractions of the current second, and the
row-value cursor could skip or repeat at such a boundary. Externally sourced
whole-second timestamps trigger it. Runs against all three backends so the
Python-filtered stores page identically to Postgres.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from gpu_fault.models import IncidentState, WorkflowOperation, WorkflowStatus
from gpu_fault.store import SqliteStore
from gpu_fault.store.contracts import WorkflowStore
from tests._builders import build_store, fault_incident, workflow_request, workflow_step
from tests.store._postgres_processor_claim_support import (
    _truncate,
    postgres_store_instance,
)

SECOND = datetime(2026, 9, 7, 6, 0, 0, tzinfo=timezone.utc)
CREATED = SECOND - timedelta(hours=1)
EXECUTABLE = {
    WorkflowStatus.PENDING,
    WorkflowStatus.SAFETY_PENDING,
    WorkflowStatus.RUNNING,
}
# (request_id, not_before) in dispatch (time) order; the whole second first.
ROWS = [
    ("wf-whole", SECOND),
    ("wf-q1", SECOND + timedelta(microseconds=250_000)),
    ("wf-half", SECOND + timedelta(microseconds=500_000)),
    ("wf-q3", SECOND + timedelta(microseconds=750_000)),
    ("wf-next", SECOND + timedelta(seconds=1)),
]
ORDER = [request_id for request_id, _ in ROWS]


@pytest.fixture(params=["memory", "sqlite", "postgres"])
def store(request: pytest.FixtureRequest, tmp_path: Path) -> Iterator[WorkflowStore]:
    if request.param == "memory":
        yield build_store()
        return
    if request.param == "sqlite":
        sqlite = SqliteStore(str(tmp_path / "whole-second.db"))
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


def _seed(store: WorkflowStore) -> None:
    # Written in scrambled order so no backend can pass on insertion order.
    for request_id, not_before in (ROWS[2], ROWS[0], ROWS[4], ROWS[3], ROWS[1]):
        incident = fault_incident(
            f"inc-{request_id}",
            f"event-{request_id}",
            state=IncidentState.ACTION_PENDING,
            workflow_request_id=request_id,
            created_at=CREATED,
            updated_at=CREATED,
        )
        workflow = workflow_request(
            request_id,
            f"inc-{request_id}",
            official_steps=[
                workflow_step(WorkflowOperation.FREEZE_EVIDENCE, node_ids=["node-a"])
            ],
            created_at=CREATED,
            updated_at=CREATED,
            not_before=not_before,
        )
        store.save_incident_and_workflow(incident, workflow)


def test_the_whole_second_sorts_before_the_fractions_of_its_second(
    store: WorkflowStore,
) -> None:
    _seed(store)

    rows = store.list_workflows(
        EXECUTABLE, limit=100, dispatchable_at=SECOND + timedelta(seconds=5)
    )

    assert [item.request_id for item in rows] == ORDER


def test_the_not_before_cut_is_exact_at_a_whole_second(store: WorkflowStore) -> None:
    _seed(store)

    at_the_second = store.list_workflows(EXECUTABLE, limit=100, dispatchable_at=SECOND)
    just_after = store.list_workflows(
        EXECUTABLE, limit=100, dispatchable_at=SECOND + timedelta(microseconds=1)
    )
    at_half = store.list_workflows(
        EXECUTABLE, limit=100, dispatchable_at=SECOND + timedelta(microseconds=500_000)
    )

    assert [item.request_id for item in at_the_second] == ["wf-whole"]
    assert [item.request_id for item in just_after] == ["wf-whole"]
    assert [item.request_id for item in at_half] == ["wf-whole", "wf-q1", "wf-half"]


@pytest.mark.parametrize("page", [1, 2, 3])
def test_paging_across_the_whole_second_neither_skips_nor_repeats(
    store: WorkflowStore, page: int
) -> None:
    _seed(store)
    horizon = SECOND + timedelta(seconds=5)

    seen: list[str] = []
    after = None
    for _ in range(len(ROWS) + 1):
        rows = store.list_workflows(
            EXECUTABLE, limit=page, dispatchable_at=horizon, after=after
        )
        if not rows:
            break
        assert len(rows) <= page
        seen.extend(item.request_id for item in rows)
        after = rows[-1]
    else:
        pytest.fail("the cursor never ran out of rows")

    assert seen == ORDER
    assert len(set(seen)) == len(seen)

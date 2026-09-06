"""Merge and executor writers interleaving on a real Postgres row.

FINAL-建议汇总 F-B1 / F-B2. The unit tests in ``test_merge_executor_isolation``
prove the compare-and-set; these prove the row lock: while the merge builder
runs inside its transaction, the executor's leased save blocks on the row, and
when it proceeds it sees the bumped revision.
"""

from __future__ import annotations

import os
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import pytest

from gpu_fault.models import IncidentState, WorkflowOperation, WorkflowStatus
from gpu_fault.store import PostgresStore
from gpu_fault.store.shared.errors import WorkflowMergedError
from tests._builders import copy_model, fault_incident, workflow_request, workflow_step
from tests.store._postgres_processor_claim_support import _truncate

POSTGRES_URL = os.getenv("GPU_FAULT_TEST_POSTGRES_URL")
pytestmark = pytest.mark.skipif(
    not POSTGRES_URL, reason="GPU_FAULT_TEST_POSTGRES_URL is not configured"
)
NOW = datetime(2026, 9, 5, 20, 30, tzinfo=timezone.utc)
GROUP = '["cluster-a","job-a","job-a-a001"]'


@pytest.fixture(autouse=True)
def clean_tables():
    assert POSTGRES_URL is not None
    PostgresStore(POSTGRES_URL).close()
    _truncate()
    yield
    _truncate()


def _store() -> PostgresStore:
    assert POSTGRES_URL is not None
    return PostgresStore(POSTGRES_URL, initialize_schema=False)


def _create(existing_incident, existing_workflow):
    incident = fault_incident(
        "inc-m",
        "event-1",
        state=IncidentState.ACTION_PENDING,
        workflow_request_id="wf-m",
        fencing_token=1,
        node_ids=["node-a"],
        created_at=NOW,
        updated_at=NOW,
    )
    workflow = workflow_request(
        "wf-m",
        "inc-m",
        status=WorkflowStatus.PENDING,
        fencing_token=1,
        official_steps=[
            workflow_step(WorkflowOperation.RESET_GPU, node_ids=["node-a"])
        ],
        created_at=NOW,
        updated_at=NOW,
    )
    return incident, workflow


def test_postgres_executor_save_blocks_on_the_merge_row_lock_then_sees_the_merge():
    merger, executor = _store(), _store()
    in_builder = threading.Event()
    release_builder = threading.Event()
    try:
        merger.merge_attempt_fault_workflow(GROUP, "event-1", _create)
        claimed = executor.claim_workflow(
            "wf-m", "executor-a", 1, lease_duration=timedelta(minutes=5)
        )
        progressed = copy_model(claimed, completed_step_indexes=[0])

        def widen(existing_incident, existing_workflow):
            in_builder.set()
            assert release_builder.wait(timeout=10), (
                "expected release_builder.wait(timeout=10) to be true"
            )
            steps = [
                copy_model(step, node_ids=sorted({*step.node_ids, "node-b"}))
                for step in existing_workflow.official_steps
            ]
            return (
                copy_model(existing_incident, node_ids=["node-a", "node-b"]),
                copy_model(existing_workflow, official_steps=steps),
            )

        def save_progress():
            assert in_builder.wait(timeout=10), (
                "expected in_builder.wait(timeout=10) to be true"
            )
            # The merge holds the row: release it only once we are queued behind it.
            threading.Timer(0.3, release_builder.set).start()
            try:
                executor.save_workflow_if_leased(
                    progressed, "executor-a", claimed.execution_epoch
                )
            except WorkflowMergedError:
                return "rejected"
            return "accepted"

        with ThreadPoolExecutor(max_workers=2) as pool:
            merge = pool.submit(
                merger.merge_attempt_fault_workflow, GROUP, "event-2", widen
            )
            outcome = pool.submit(save_progress).result(timeout=20)
            merge.result(timeout=20)

        current = executor.get_workflow("wf-m")
        assert current.official_steps[0].node_ids == ["node-a", "node-b"]
        assert outcome == "rejected"
        assert current.merge_revision == 1

        fresh = executor.renew_workflow_lease(
            "wf-m", "executor-a", claimed.execution_epoch
        )
        executor.save_workflow_if_leased(
            copy_model(fresh, completed_step_indexes=[0]),
            "executor-a",
            claimed.execution_epoch,
        )
        final = executor.get_workflow("wf-m")
        assert final.official_steps[0].node_ids == ["node-a", "node-b"]
        assert final.completed_step_indexes == [0]
    finally:
        release_builder.set()
        merger.close()
        executor.close()


def test_postgres_claim_workflow_takes_budget_locks_before_the_row_lock():
    """F-B2: advisory locks first, then row locks, everywhere."""

    store = _store()
    try:
        store.merge_attempt_fault_workflow(GROUP, "event-1", _create)
        statements: list[str] = []
        original = store._db.cursor

        import contextlib

        class Recording:
            def __init__(self, cursor):
                self._cursor = cursor

            def execute(self, query, parameters=None):
                statements.append(query)
                return self._cursor.execute(query, parameters)

            def __getattr__(self, name):
                return getattr(self._cursor, name)

        @contextlib.contextmanager
        def recording(*args, **kwargs):
            with original(*args, **kwargs) as cursor:
                yield Recording(cursor)

        store._db.cursor = recording
        store.claim_workflow(
            "wf-m",
            "executor-a",
            1,
            lease_duration=timedelta(minutes=5),
            remediation_budget_claims={"cluster-a/node/node-a": 1},
        )
        advisory = next(
            i for i, q in enumerate(statements) if "pg_advisory_xact_lock" in q
        )
        row_lock = next(i for i, q in enumerate(statements) if "FOR UPDATE" in q)

        assert advisory < row_lock, statements
    finally:
        store.close()

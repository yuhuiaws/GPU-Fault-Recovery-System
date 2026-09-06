"""The Postgres core helpers enforce what they used to assume.

F-J4 (docs/review/FINAL-建议汇总.md). ``_get_for_update`` takes a row lock
that lives exactly as long as the enclosing transaction -- and the pool is
autocommit, so a caller outside ``transaction()`` got a lock that was gone
before the read returned, with nothing to say so. ``_put`` overwrote the
whole row unconditionally; a writer that wanted "only if unchanged" had no
way to ask.
"""

from __future__ import annotations

import pytest

from gpu_fault.models import WorkflowOperation, WorkflowStatus
from gpu_fault.store.shared.errors import StaleWriteError, TransactionRequiredError
from tests._builders import copy_model, fault_incident, workflow_request, workflow_step
from tests.store._postgres_processor_claim_support import (
    POSTGRES_URL,
    _truncate,
    postgres_store_instance,
)


@pytest.fixture
def store():
    if not POSTGRES_URL:
        pytest.skip("GPU_FAULT_TEST_POSTGRES_URL is required")
    yield from postgres_store_instance()
    _truncate()


def _seed(store):
    incident = fault_incident("inc-a", "event-a", workflow_request_id="wf-a")
    workflow = workflow_request(
        "wf-a", "inc-a", official_steps=[workflow_step(WorkflowOperation.RESET_GPU)]
    )
    store.save_incident_and_workflow(incident, workflow)
    return workflow


def test_a_locked_read_outside_a_transaction_is_refused(store):
    _seed(store)

    with pytest.raises(TransactionRequiredError):
        store._get_for_update("workflow", "wf-a")
    with store._db.transaction():
        assert store._get_for_update("workflow", "wf-a").request_id == "wf-a"
    assert store._db.in_transaction is False


def test_a_conditional_put_only_lands_on_the_row_it_was_read_from(store):
    workflow = _seed(store)
    stored = store.get_workflow("wf-a")
    changed_elsewhere = copy_model(stored, status=WorkflowStatus.RUNNING)
    store._put("workflow", "wf-a", changed_elsewhere)

    with pytest.raises(StaleWriteError):
        store._put(
            "workflow",
            "wf-a",
            copy_model(stored, status=WorkflowStatus.FAILED),
            expected=stored,  # the row moved on since this snapshot
        )
    assert store.get_workflow("wf-a").status is WorkflowStatus.RUNNING

    store._put(
        "workflow",
        "wf-a",
        copy_model(changed_elsewhere, status=WorkflowStatus.SUCCEEDED),
        expected=changed_elsewhere,
    )
    assert store.get_workflow("wf-a").status is WorkflowStatus.SUCCEEDED
    with pytest.raises(StaleWriteError):
        store._put("workflow", "wf-missing", workflow, expected=workflow)

"""The merge that creates a preempting successor also holds its predecessor,
in the same transaction (F-C1).

Before, only the incident and the successor were written; the predecessor
stayed PENDING and dispatchable until an executor claimed it and noticed the
successor. The stamp lets the dispatcher hold it immediately.
"""

from __future__ import annotations

import os

import pytest

from gpu_fault.models import WorkflowOperation, WorkflowStatus
from gpu_fault.store import SqliteStore
from tests._builders import (
    build_store,
    copy_model,
    fault_incident,
    workflow_request,
    workflow_step,
)
from tests.store._postgres_processor_claim_support import (
    _truncate,
    postgres_store_instance,
)

GROUP = "group-node-a"
RESET = WorkflowOperation.RESET_GPU
REBOOT = WorkflowOperation.RESTART_NODE


@pytest.fixture(params=["memory", "sqlite", "postgres"])
def store(request, tmp_path):
    if request.param == "memory":
        yield build_store()
        return
    if request.param == "sqlite":
        sqlite = SqliteStore(str(tmp_path / "preempt.db"))
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


def _create(existing_incident, existing_workflow):
    incident = fault_incident(
        "inc-m", "event-1", workflow_request_id="wf-old", fencing_token=1
    )
    workflow = workflow_request(
        "wf-old",
        "inc-m",
        fencing_token=1,
        official_steps=[workflow_step(RESET, node_ids=["node-a"])],
    )
    return incident, workflow


def _successor(*, preempt: bool):
    def build(existing_incident, existing_workflow):
        assert existing_incident is not None and existing_workflow is not None
        successor = workflow_request(
            "wf-new",
            "inc-m",
            fencing_token=existing_workflow.fencing_token,
            official_steps=[workflow_step(REBOOT, node_ids=["node-a"])],
            predecessor_workflow_id=existing_workflow.request_id,
            preempt_predecessor=preempt,
        )
        return copy_model(existing_incident, workflow_request_id="wf-new"), successor

    return build


def test_a_preempting_successor_holds_its_predecessor_in_the_same_merge(store):
    store.merge_attempt_fault_workflow(GROUP, "event-1", _create)

    store.merge_attempt_fault_workflow(GROUP, "event-2", _successor(preempt=True))

    predecessor = store.get_workflow("wf-old")
    assert predecessor.preemption_pending_by_workflow_id == "wf-new"
    assert predecessor.status is WorkflowStatus.PENDING, (
        "the hold is a marker, not a status change"
    )
    assert store.get_workflow("wf-new").preempt_predecessor is True


def test_a_queued_successor_leaves_its_predecessor_alone(store):
    store.merge_attempt_fault_workflow(GROUP, "event-1", _create)

    store.merge_attempt_fault_workflow(GROUP, "event-2", _successor(preempt=False))

    assert store.get_workflow("wf-old").preemption_pending_by_workflow_id is None


def test_a_terminal_predecessor_is_not_stamped(store):
    _, created = store.merge_attempt_fault_workflow(GROUP, "event-1", _create)
    store.save_workflow(copy_model(created, status=WorkflowStatus.FAILED))

    store.merge_attempt_fault_workflow(GROUP, "event-2", _successor(preempt=True))

    assert store.get_workflow("wf-old").preemption_pending_by_workflow_id is None

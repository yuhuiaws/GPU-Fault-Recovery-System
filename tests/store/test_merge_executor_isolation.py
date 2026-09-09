"""Merging into a workflow and executing it must not overwrite each other.

FINAL-建议汇总 F-B1 (P0-78A / P0-42B / P0-27A) and F-B3 (P1-80A). The merge
family read the workflow without a row lock and wrote it back wholesale; the
executor's ``save_workflow_if_leased`` checked only the lease (owner, epoch,
expiry) and then wrote its own copy wholesale. Neither noticed the other. A
target merged in between the executor's read and write was silently dropped,
and because the event's dedup link already pointed at the incident, no
redelivery could ever create a workflow for it -- the orphan of chain 3.

Every merge now bumps ``merge_revision`` under the row lock, and every leased
save compares it: a stale copy raises ``WorkflowMergedError`` (a
``WorkflowLeaseError``) so the executor stands aside and picks the merged
record up on its next lease renewal.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

import pytest

from gpu_fault.models import IncidentState, WorkflowOperation, WorkflowStatus
from gpu_fault.store import SqliteStore
from gpu_fault.store.shared.errors import (
    StaleWriteError,
    WorkflowLeaseError,
    WorkflowMergedError,
)
from gpu_fault.workflow_resolution import abandoned_generation_successor
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

NOW = datetime(2026, 9, 5, 20, 0, tzinfo=timezone.utc)
GROUP = '["cluster-a","job-a","job-a-a001"]'


@pytest.fixture(params=["memory", "sqlite", "postgres"])
def store(request, tmp_path):
    if request.param == "memory":
        yield build_store()
        return
    if request.param == "sqlite":
        sqlite = SqliteStore(str(tmp_path / "merge.db"))
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
    assert existing_incident is None and existing_workflow is None
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
            workflow_step(WorkflowOperation.RESET_GPU, node_ids=["node-a"]),
            workflow_step(WorkflowOperation.VALIDATE_GPU, node_ids=["node-a"]),
        ],
        created_at=NOW,
        updated_at=NOW,
    )
    return incident, workflow


def _widen(node: str):
    def build(existing_incident, existing_workflow):
        assert existing_incident is not None and existing_workflow is not None
        steps = [
            copy_model(step, node_ids=sorted({*step.node_ids, node}))
            for step in existing_workflow.official_steps
        ]
        return (
            copy_model(
                existing_incident, node_ids=sorted({*existing_incident.node_ids, node})
            ),
            copy_model(existing_workflow, official_steps=steps),
        )

    return build


def _nodes(workflow) -> list[str]:
    return workflow.official_steps[0].node_ids


def test_every_merge_into_an_existing_workflow_bumps_its_revision(store) -> None:
    _, created = store.merge_attempt_fault_workflow(GROUP, "event-1", _create)
    _, once = store.merge_attempt_fault_workflow(GROUP, "event-2", _widen("node-b"))
    _, twice = store.merge_attempt_fault_workflow(GROUP, "event-3", _widen("node-c"))

    assert created.merge_revision == 0
    assert once.merge_revision == 1
    assert twice.merge_revision == 2
    assert store.get_workflow("wf-m").merge_revision == 2


def test_leased_save_refuses_a_copy_read_before_a_merge(store) -> None:
    """Chain 3: the executor must not write over the target merged behind it."""

    store.merge_attempt_fault_workflow(GROUP, "event-1", _create)
    claimed = store.claim_workflow(
        "wf-m", "executor-a", 1, lease_duration=timedelta(minutes=5)
    )
    progressed = copy_model(claimed, completed_step_indexes=[0])

    store.merge_attempt_fault_workflow(GROUP, "event-2", _widen("node-b"))

    with pytest.raises(WorkflowMergedError) as raised:
        store.save_workflow_if_leased(progressed, "executor-a", claimed.execution_epoch)
    assert isinstance(raised.value, WorkflowLeaseError), (
        "expected isinstance(raised.value, WorkflowLeaseError) to be true"
    )
    assert _nodes(store.get_workflow("wf-m")) == ["node-a", "node-b"]


def test_leased_save_accepts_the_copy_taken_after_the_merge(store) -> None:
    store.merge_attempt_fault_workflow(GROUP, "event-1", _create)
    claimed = store.claim_workflow(
        "wf-m", "executor-a", 1, lease_duration=timedelta(minutes=5)
    )
    store.merge_attempt_fault_workflow(GROUP, "event-2", _widen("node-b"))

    fresh = store.renew_workflow_lease("wf-m", "executor-a", claimed.execution_epoch)
    store.save_workflow_if_leased(
        copy_model(fresh, completed_step_indexes=[0]),
        "executor-a",
        claimed.execution_epoch,
    )

    current = store.get_workflow("wf-m")
    assert _nodes(current) == ["node-a", "node-b"]
    assert current.completed_step_indexes == [0]


def test_workflow_and_incident_save_refuses_a_stale_copy_too(store) -> None:
    store.merge_attempt_fault_workflow(GROUP, "event-1", _create)
    claimed = store.claim_workflow(
        "wf-m", "executor-a", 1, lease_duration=timedelta(minutes=5)
    )
    incident = store.get_incident("inc-m")
    store.merge_attempt_fault_workflow(GROUP, "event-2", _widen("node-b"))

    with pytest.raises(WorkflowMergedError):
        store.save_workflow_and_incident_if_leased(
            copy_model(claimed, completed_step_indexes=[0]),
            incident,
            "executor-a",
            claimed.execution_epoch,
        )
    assert store.get_incident("inc-m").node_ids == ["node-a", "node-b"]


def _twin_pair(store, *, link_successor: bool):
    incident = fault_incident(
        "inc-twin",
        "event-twin",
        state=IncidentState.ACTION_PENDING,
        workflow_request_id="wf-b",
        fencing_token=2,
        created_at=NOW,
        updated_at=NOW,
    )
    a = workflow_request(
        "wf-a",
        "inc-twin",
        status=WorkflowStatus.PENDING,
        fencing_token=2,
        created_at=NOW,
        updated_at=NOW,
    )
    b = workflow_request(
        "wf-b",
        "inc-twin",
        status=WorkflowStatus.PENDING,
        fencing_token=2,
        predecessor_workflow_id="wf-a" if link_successor else None,
        created_at=NOW + timedelta(seconds=1),
        updated_at=NOW + timedelta(seconds=1),
    )
    store.save_incident_and_workflow(incident, b)
    store.save_workflow(a)
    return a, b


def test_an_unlinked_same_generation_twin_is_an_abandoned_generation(store) -> None:
    """P1-80A: same incident, same token, incident names the other, nobody links."""

    a, b = _twin_pair(store, link_successor=False)

    successor = abandoned_generation_successor(store, a, now=NOW + timedelta(hours=1))

    assert successor is not None and successor.request_id == "wf-b"


def test_a_queued_same_generation_successor_protects_its_predecessor(store) -> None:
    a, _ = _twin_pair(store, link_successor=True)

    assert (
        abandoned_generation_successor(store, a, now=NOW + timedelta(hours=1)) is None
    )


def test_has_workflow_successor_sees_any_predecessor_link(store) -> None:
    a, _ = _twin_pair(store, link_successor=True)

    assert store.has_workflow_successor("wf-a") is True
    assert store.has_workflow_successor("wf-b") is False


# ---------------------------------------------------------------- C-01


def _adopt(store, node: str):
    """A builder that finds no group link and adopts the pair another key owns.

    The shape of ``grouped_faults._active_attempt_recovery``, sxid
    ``_build_state`` and the drain incumbent join: the group has no link, the
    builder picks an already-stored (incident, workflow) and returns a widened
    copy of it.
    """

    def build(existing_incident, existing_workflow):
        assert existing_incident is None and existing_workflow is None
        incident = store.get_incident("inc-m")
        workflow = store.get_workflow("wf-m")
        steps = [
            copy_model(step, node_ids=sorted({*step.node_ids, node}))
            for step in workflow.official_steps
        ]
        return (
            copy_model(incident, node_ids=sorted({*incident.node_ids, node})),
            copy_model(workflow, official_steps=steps),
        )

    return build


def test_an_adoption_bumps_the_adopted_row_and_refuses_the_executors_old_copy(
    store,
) -> None:
    """C-01: the pair was stored under no group link (escalation, reset,
    health, node-scope faults all do this) and the executor holds it. An
    attempt-group merge adopts it. The merge must stamp the *adopted* row,
    not the group link's row (there is none), or the executor's next leased
    save overwrites the adoption while the event link already says handled."""

    incident, workflow = _create(None, None)
    store.save_incident_and_workflow(incident, workflow)
    claimed = store.claim_workflow(
        "wf-m", "executor-a", 1, lease_duration=timedelta(minutes=5)
    )
    progressed = copy_model(claimed, completed_step_indexes=[0])

    _, adopted = store.merge_attempt_fault_workflow(
        GROUP, "event-2", _adopt(store, "node-b")
    )

    assert adopted.merge_revision == 1
    assert store.get_workflow("wf-m").merge_revision == 1
    with pytest.raises(WorkflowMergedError):
        store.save_workflow_if_leased(progressed, "executor-a", claimed.execution_epoch)
    assert _nodes(store.get_workflow("wf-m")) == ["node-a", "node-b"]
    assert store.get_incident("inc-m").node_ids == ["node-a", "node-b"]
    assert store.get_incident_by_event("event-2").incident_id == "inc-m"


def test_an_adoption_through_the_replacement_group_bumps_too(store) -> None:
    incident, workflow = _create(None, None)
    store.save_incident_and_workflow(incident, workflow)
    claimed = store.claim_workflow(
        "wf-m", "executor-a", 1, lease_duration=timedelta(minutes=5)
    )

    _, adopted = store.merge_replacement_workflow(
        GROUP, "event-2", _adopt(store, "node-b")
    )

    assert adopted.merge_revision == 1
    with pytest.raises(WorkflowMergedError):
        store.save_workflow_if_leased(
            copy_model(claimed, completed_step_indexes=[0]),
            "executor-a",
            claimed.execution_epoch,
        )


def test_an_adoption_cannot_move_the_incident_back_a_generation(store) -> None:
    """The adopted incident goes through the same generation guard as
    ``save_incident``: a builder holding an older snapshot cannot write the
    previous generation back over the stored one."""

    incident, workflow = _create(None, None)
    store.save_incident_and_workflow(
        copy_model(incident, fencing_token=3), copy_model(workflow, fencing_token=3)
    )

    def build(existing_incident, existing_workflow):
        assert existing_incident is None and existing_workflow is None
        return copy_model(incident, fencing_token=2), copy_model(
            workflow, fencing_token=2
        )

    with pytest.raises(StaleWriteError):
        store.merge_attempt_fault_workflow(GROUP, "event-2", build)
    assert store.get_incident("inc-m").fencing_token == 3
    assert store.get_incident_by_event("event-2") is None


# ---------------------------------------------------------------- C-02


def test_leased_incident_write_yields_to_a_pointer_that_moved(store) -> None:
    """C-02: a QUEUE_SUCCESSOR merge moves the incident pointer to W2 and
    widens the incident; the predecessor row is untouched, so the executor's
    ``merge_revision`` still matches. Its incident snapshot predates the
    move and must not be written back over the successor's incident."""

    store.merge_attempt_fault_workflow(GROUP, "event-1", _create)
    claimed = store.claim_workflow(
        "wf-m", "executor-a", 1, lease_duration=timedelta(minutes=5)
    )
    snapshot = store.get_incident("inc-m")
    store.save_workflow(
        workflow_request(
            "wf-s",
            "inc-m",
            status=WorkflowStatus.PENDING,
            fencing_token=1,
            predecessor_workflow_id="wf-m",
            created_at=NOW + timedelta(seconds=1),
            updated_at=NOW + timedelta(seconds=1),
        )
    )
    store.save_incident(
        copy_model(snapshot, workflow_request_id="wf-s", node_ids=["node-a", "node-b"]),
        expected=snapshot,
    )

    store.save_workflow_and_incident_if_leased(
        copy_model(claimed, completed_step_indexes=[0]),
        snapshot,
        "executor-a",
        claimed.execution_epoch,
    )

    current = store.get_incident("inc-m")
    assert current.workflow_request_id == "wf-s"
    assert current.node_ids == ["node-a", "node-b"]
    assert store.get_workflow("wf-m").completed_step_indexes == [0]


def test_leased_incident_write_lands_while_the_pointer_still_agrees(store) -> None:
    store.merge_attempt_fault_workflow(GROUP, "event-1", _create)
    claimed = store.claim_workflow(
        "wf-m", "executor-a", 1, lease_duration=timedelta(minutes=5)
    )
    snapshot = store.get_incident("inc-m")

    store.save_workflow_and_incident_if_leased(
        copy_model(claimed, completed_step_indexes=[0]),
        copy_model(snapshot, state=IncidentState.RECOVERED),
        "executor-a",
        claimed.execution_epoch,
    )

    assert store.get_incident("inc-m").state is IncidentState.RECOVERED

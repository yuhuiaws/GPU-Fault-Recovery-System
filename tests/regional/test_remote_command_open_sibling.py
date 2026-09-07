"""A rewritten step never gets a second remote command while the first is open.

Architecture review 2026-09-07, item D5. ``RegionalRemoteWorkflowAdapter``
derives ``command_id`` from a digest of the whole step, and the store only ever
looked commands up by that id. A merge that rewrote a LEASED or WAITING step's
``node_ids`` or ``parameters`` therefore made the next dispatch mint a sibling
command for the same (workflow, step index) while the executor was still
running the first one on the node -- the same physical action twice. The
adapter now asks the store for an open sibling first and holds (WAITING, with
an explicit reason and a counter) instead of creating the second command.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from gpu_fault.models import WorkflowStepStatus
from gpu_fault.regional import (
    RegionalRemoteWorkflowAdapter,
    RemoteCommandResult,
    RemoteCommandStatus,
)
from gpu_fault.store import SqliteStore
from tests._builders import build_store, copy_model
from tests.regional._regional_support import TOKEN_A, registration, workflow_state


@pytest.fixture(params=["memory", "sqlite"])
def store(request, tmp_path):
    if request.param == "memory":
        instance = build_store()
    else:
        instance = SqliteStore(str(tmp_path / "sibling.db"))
    instance.save_regional_cluster(registration("cluster-a", TOKEN_A))
    try:
        yield instance
    finally:
        if request.param == "sqlite":
            instance.close()


def _adapter(store) -> RegionalRemoteWorkflowAdapter:
    return RegionalRemoteWorkflowAdapter(store, owners={"gpu-fault-kubernetes-adapter"})


def _rewritten(context):
    """The same step after a merge widened its targets: a new command digest."""

    step = copy_model(context.step, node_ids=["node-a", "node-b"])
    workflow = copy_model(context.workflow, official_steps=[step])
    return replace(context, step=step, workflow=workflow)


def test_a_rewritten_step_holds_while_its_first_command_is_leased(store) -> None:
    adapter = _adapter(store)
    context = workflow_state()
    first = adapter.execute(context)
    claimed = store.claim_remote_commands(
        "cluster-a", "executor-a", limit=1, lease_seconds=60
    )
    assert [item.status for item in claimed] == [RemoteCommandStatus.LEASED]

    held = adapter.execute(_rewritten(context))

    assert held.status is WorkflowStepStatus.WAITING
    assert held.details is not None
    assert held.details["reason"] == "OPEN_SIBLING_COMMAND"
    assert held.details["remote_command_id"] == first.details["remote_command_id"]
    assert held.details["remote_status"] == "LEASED"
    assert held.details["mutation_submitted_by_control_plane"] is False
    assert [item.command_id for item in store.list_remote_commands()] == [
        first.details["remote_command_id"]
    ], "no second command was created"
    assert adapter.open_sibling_holds_total == 1


def test_a_rewritten_step_holds_while_its_first_command_is_pending(store) -> None:
    adapter = _adapter(store)
    context = workflow_state()
    adapter.execute(context)

    held = adapter.execute(_rewritten(context))

    assert held.status is WorkflowStepStatus.WAITING
    assert held.details is not None
    assert held.details["reason"] == "OPEN_SIBLING_COMMAND"
    assert len(store.list_remote_commands()) == 1


def test_the_hold_lifts_once_the_first_command_is_terminal(store) -> None:
    adapter = _adapter(store)
    context = workflow_state()
    adapter.execute(context)
    claimed = store.claim_remote_commands(
        "cluster-a", "executor-a", limit=1, lease_seconds=60
    )
    store.complete_remote_command(
        "cluster-a",
        claimed[0].command_id,
        RemoteCommandResult(
            lease_token=claimed[0].lease_token, status=RemoteCommandStatus.SUCCEEDED
        ),
    )

    outcome = adapter.execute(_rewritten(context))

    assert outcome.status is WorkflowStepStatus.WAITING
    assert outcome.details is not None
    assert outcome.details.get("reason") is None, "this is a normal new command"
    assert len(store.list_remote_commands()) == 2
    assert adapter.open_sibling_holds_total == 0


def test_the_same_command_is_not_its_own_sibling(store) -> None:
    adapter = _adapter(store)
    context = workflow_state()
    first = adapter.execute(context)

    again = adapter.execute(context)

    assert again.details is not None
    assert again.details["remote_command_id"] == first.details["remote_command_id"]
    assert again.details.get("reason") is None
    assert adapter.open_sibling_holds_total == 0


def test_a_previous_generation_does_not_hold_the_next(store) -> None:
    """A new fencing token is the generation fence's business, not this one:
    the old command can never be claimed or completed under the new token."""

    adapter = _adapter(store)
    context = workflow_state()
    adapter.execute(context)
    incident = copy_model(
        context.incident, fencing_token=4, workflow_request_id="workflow-a"
    )
    workflow = copy_model(context.workflow, fencing_token=4)
    store.save_incident_and_workflow(incident, workflow)

    outcome = adapter.execute(replace(context, incident=incident, workflow=workflow))

    assert outcome.details is not None
    assert outcome.details.get("reason") is None
    assert len(store.list_remote_commands()) == 2

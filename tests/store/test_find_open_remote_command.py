"""``find_open_remote_command`` names the open command for one step identity.

Architecture review 2026-09-07, item D5. The remote command id is a digest of
the whole step (targets, parameters, workload ids), so a merge that rewrites a
LEASED step's ``node_ids`` makes the next dispatch compute a new id, and
``ensure_remote_command`` -- a lookup by id -- happily creates a sibling while
the old command keeps executing on the node. The adapter needs the store to
answer "is anything still open for (workflow, step_index, step space)?" so it
can hold instead. Memory and SQLite here; the Postgres half is
``test_postgres_find_open_remote_command.py``.
"""

from __future__ import annotations

from datetime import timedelta

import pytest

from gpu_fault.models import IncidentState, WorkflowOperation, WorkflowStatus
from gpu_fault.regional import RemoteActionCommand, RemoteCommandResult
from gpu_fault.remote_command_models import RemoteCommandStatus
from gpu_fault.store import SqliteStore
from tests._builders import (
    build_store,
    copy_model,
    fault_incident,
    workflow_request,
    workflow_step,
)
from tests.regional._regional_support import NOW


@pytest.fixture(params=["memory", "sqlite"])
def store(request, tmp_path):
    if request.param == "memory":
        yield build_store()
        return
    sqlite = SqliteStore(str(tmp_path / "open-command.db"))
    try:
        yield sqlite
    finally:
        sqlite.close()


def _state(store, *, safety: bool = False):
    incident = fault_incident(
        "incident-a",
        "event-a",
        state=IncidentState.ACTION_PENDING,
        workflow_request_id="workflow-a",
        fencing_token=3,
    )
    step = workflow_step(WorkflowOperation.RESET_GPU, "gpu-fault-kubernetes-adapter")
    workflow = workflow_request(
        "workflow-a",
        incident.incident_id,
        WorkflowStatus.RUNNING,
        official_steps=[step, step],
        safety_steps=[step],
        safety_only=safety,
        created_at=NOW,
        updated_at=NOW,
    )
    store.save_incident_and_workflow(incident, workflow)
    return incident, workflow


def _command(store, command_id: str, *, step_index: int = 0, created_offset=0, **over):
    incident, workflow = _state(store, safety=over.pop("safety", False))
    command = RemoteActionCommand(
        command_id=command_id,
        cluster_id=incident.cluster_id,
        workflow_request_id=workflow.request_id,
        incident_id=incident.incident_id,
        step_index=step_index,
        fencing_token=over.pop("fencing_token", 3),
        idempotency_key=f"workflow-a/{step_index}/RESET_GPU",
        step=workflow.official_steps[step_index],
        workflow=workflow,
        incident=incident,
        created_at=NOW + timedelta(seconds=created_offset),
        **over,
    )
    return store.ensure_remote_command(command)


def test_nothing_open_returns_none(store) -> None:
    _state(store)

    assert store.find_open_remote_command("workflow-a", 0, "official") is None


def test_a_pending_command_for_the_step_is_returned(store) -> None:
    command = _command(store, "remote-a")

    found = store.find_open_remote_command("workflow-a", 0, "official")

    assert found is not None, "a PENDING command is open"
    assert found.command_id == command.command_id


def test_another_step_index_is_not_a_sibling(store) -> None:
    _command(store, "remote-a", step_index=1)

    assert store.find_open_remote_command("workflow-a", 0, "official") is None


def test_the_safety_step_space_is_separate(store) -> None:
    _command(store, "remote-safety", safety=True)

    assert store.find_open_remote_command("workflow-a", 0, "official") is None
    found = store.find_open_remote_command("workflow-a", 0, "safety")
    assert found is not None, "the safety command is open in its own space"
    assert found.command_id == "remote-safety"


def test_a_leased_command_is_open_and_a_terminal_one_is_not(store) -> None:
    _command(store, "remote-a")
    claimed = store.claim_remote_commands(
        "cluster-a", "executor-a", limit=1, lease_seconds=60
    )
    assert [item.status for item in claimed] == [RemoteCommandStatus.LEASED]
    found = store.find_open_remote_command("workflow-a", 0, "official")
    assert found is not None and found.status is RemoteCommandStatus.LEASED, (
        "a LEASED command is executing somewhere and therefore open"
    )

    store.complete_remote_command(
        "cluster-a",
        "remote-a",
        RemoteCommandResult(
            lease_token=claimed[0].lease_token, status=RemoteCommandStatus.SUCCEEDED
        ),
    )

    assert store.find_open_remote_command("workflow-a", 0, "official") is None


def test_the_caller_can_exclude_its_own_command_id(store) -> None:
    _command(store, "remote-a")

    assert (
        store.find_open_remote_command(
            "workflow-a", 0, "official", exclude_command_id="remote-a"
        )
        is None
    )


def test_the_oldest_open_sibling_wins(store) -> None:
    _command(store, "remote-later", created_offset=10)
    incident, workflow = _state(store)
    older = copy_model(
        store.get_remote_command("remote-later"),
        command_id="remote-earlier",
        created_at=NOW,
    )
    store.ensure_remote_command(older)

    found = store.find_open_remote_command("workflow-a", 0, "official")

    assert found is not None and found.command_id == "remote-earlier"

from __future__ import annotations

import io
import json
import sys
from collections.abc import Iterator
from contextlib import redirect_stdout
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from gpu_fault.app import ApplicationContext
from gpu_fault.models import WorkflowStatus
from gpu_fault.orphaned_commands import cancel_orphaned_commands
from gpu_fault.regional import RemoteCommandResult, RemoteCommandStatus
from gpu_fault.store import InMemoryStore, SqliteStore, WorkflowLeaseError
from scripts.e2e.regional import run_ha001_control_plane_failover as ha001
from scripts.e2e.regional import run_ha006_executor_takeover as ha006
from scripts.e2e.regional import run_ha009_aurora_credential_rotation as ha009


@pytest.fixture(params=["memory", "sqlite"])
def seed_store(
    request: pytest.FixtureRequest, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[InMemoryStore | SqliteStore]:
    store = (
        InMemoryStore()
        if request.param == "memory"
        else SqliteStore(str(tmp_path / "ha-seeds.db"))
    )
    monkeypatch.setattr(
        ApplicationContext,
        "from_environment",
        classmethod(lambda _cls: SimpleNamespace(store=store)),
    )

    def execute_script(script: str, *arguments: str, **_kwargs: Any) -> dict[str, Any]:
        output = io.StringIO()
        with monkeypatch.context() as local, redirect_stdout(output):
            local.setattr(sys, "argv", ["unit-cpu", *arguments])
            exec(script, {})
        return json.loads(output.getvalue())

    monkeypatch.setattr(ha001, "cpu_python", execute_script)
    monkeypatch.setattr(ha006, "cpu_python", execute_script)
    monkeypatch.setattr(ha009.BASE, "cpu_python", execute_script)
    try:
        yield store
    finally:
        if isinstance(store, SqliteStore):
            store.close()


def seed_case(case: str) -> tuple[dict[str, Any], int]:
    if case == "ha001":
        return ha001.seed_closure(
            "ha001-main-unit", "unit-cluster"
        ), ha001.SEED_LEASE_SECONDS
    if case == "ha006":
        return ha006.seed_command("ha006-main-unit"), ha006.SEED_LEASE_SECONDS
    return ha009.seed_runtime_records("ha009-main-unit"), ha009.SEED_LEASE_SECONDS


@pytest.mark.parametrize("case", ["ha001", "ha006", "ha009"])
def test_seed_workflows_hold_a_bounded_foreign_lease_without_becoming_terminal(
    case: str, seed_store: InMemoryStore | SqliteStore
) -> None:
    started = datetime.now(timezone.utc)
    seed, lease_seconds = seed_case(case)
    finished = datetime.now(timezone.utc)
    workflow = seed_store.get_workflow(seed["workflow_id"])
    assert workflow.status is WorkflowStatus.PENDING
    assert workflow.blocked_reasons == []
    assert workflow.execution_owner_id and workflow.execution_owner_id.endswith("-seed")
    assert workflow.execution_epoch == 1
    expires = workflow.execution_lease_expires_at
    assert expires is not None
    assert started + timedelta(seconds=lease_seconds) <= expires
    assert expires <= finished + timedelta(seconds=lease_seconds)
    for step in workflow.official_steps:
        assert step.execution_owner != workflow.execution_owner_id
    with pytest.raises(WorkflowLeaseError, match="another executor"):
        seed_store.claim_workflow(
            workflow.request_id,
            "ordinary-dispatcher",
            workflow.fencing_token,
            now=finished,
        )
    assert cancel_orphaned_commands(seed_store, now=finished) == {}
    command_ids = seed.get("command_ids", [seed.get("command_id")])
    for command_id in command_ids:
        command = seed_store.get_remote_command(command_id)
        assert command.status is RemoteCommandStatus.PENDING
        assert command.workflow.execution_owner_id == workflow.execution_owner_id
        assert command.workflow.execution_lease_expires_at == expires
        assert command.workflow.status is WorkflowStatus.PENDING
    claimed = seed_store.claim_workflow(
        workflow.request_id,
        "ordinary-dispatcher",
        workflow.fencing_token,
        now=expires + timedelta(seconds=1),
    )
    assert claimed.execution_epoch == 2, (
        "the fixture lease must be bounded, not permanent"
    )


@pytest.mark.parametrize("case", ["ha001", "ha006", "ha009"])
def test_probe_can_complete_each_command_without_orphaning_the_remaining_steps(
    case: str, seed_store: InMemoryStore | SqliteStore
) -> None:
    seed, _lease_seconds = seed_case(case)
    command_ids = seed.get("command_ids", [seed.get("command_id")])
    workflow = seed_store.get_workflow(seed["workflow_id"])
    incident = seed_store.get_incident(seed["incident_id"])
    for index, command_id in enumerate(command_ids):
        assert (
            cancel_orphaned_commands(seed_store, now=datetime.now(timezone.utc)) == {}
        )
        claimed = seed_store.claim_remote_commands(
            incident.cluster_id,
            "unit-probe",
            limit=1,
            lease_seconds=60,
            execution_owners={step.execution_owner for step in workflow.official_steps},
        )
        assert [command.command_id for command in claimed] == [command_id]
        completed = seed_store.complete_remote_command(
            incident.cluster_id,
            command_id,
            RemoteCommandResult(
                status=RemoteCommandStatus.SUCCEEDED,
                lease_token=claimed[0].lease_token,
                details={"simulated": True, "cached": False},
            ),
        )
        assert completed.status is RemoteCommandStatus.SUCCEEDED
        assert (
            cancel_orphaned_commands(seed_store, now=datetime.now(timezone.utc)) == {}
        )
        assert all(
            seed_store.get_remote_command(pending).status is RemoteCommandStatus.PENDING
            for pending in command_ids[index + 1 :]
        ), "completing a probe step must preserve later pending commands"
    assert seed_store.get_workflow(seed["workflow_id"]) == workflow
    if case == "ha001":
        observed = ha001.closure_status(seed)
        assert observed["workflow_status_observed"] == "PENDING"
        assert len(observed["commands"]) == 3
    if case == "ha009":
        assert (
            seed_store.get_notification(seed["notification_id"]).drill_id
            == "ha009-main-unit"
        )


def test_terminal_seed_is_a_real_orphan_sweep_failure_control(
    seed_store: InMemoryStore | SqliteStore,
) -> None:
    seed, _lease_seconds = seed_case("ha001")
    workflow = seed_store.get_workflow(seed["workflow_id"])
    seed_store.save_workflow(
        workflow.model_copy(update={"status": WorkflowStatus.BLOCKED})
    )
    cancelled = cancel_orphaned_commands(seed_store, now=datetime.now(timezone.utc))
    assert seed["workflow_id"] in cancelled
    assert all(
        seed_store.get_remote_command(command_id).status is RemoteCommandStatus.FAILED
        for command_id in seed["command_ids"]
    ), "the terminal-seed control must exercise real orphan cancellation"


def test_seed_acknowledgement_stays_bound_to_predeclared_cleanup_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    def wrong_identity(*arguments: Any, **kwargs: Any) -> dict[str, Any]:
        calls.append((arguments, kwargs))
        return {"workflow_id": "foreign-workflow"}

    monkeypatch.setattr(ha001, "cpu_python", wrong_identity)
    with pytest.raises(ha001.CaseError, match="acknowledgement identity"):
        ha001.seed_closure("ha001-main-unit", "unit-cluster")
    assert len(calls) == 1 and calls[0][1] == {"attempts": 1}
    assert calls[0][0][-1] == str(ha001.SEED_LEASE_SECONDS)


def test_seed_leases_cover_the_fixture_budgets() -> None:
    assert ha001.SEED_LEASE_SECONDS >= 30 * 60
    assert ha006.SEED_LEASE_SECONDS >= 30 * 60
    assert ha009.SEED_LEASE_SECONDS == ha009.total_budget_seconds()
    assert ha009.SEED_LEASE_SECONDS > 60 * 60

"""Run the real HA seed programs against exclusively granted local PostgreSQL."""

from __future__ import annotations

import io
import json
import os
import sys
from collections.abc import Iterator
from contextlib import closing, redirect_stdout
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any

import pytest

from gpu_fault.app import ApplicationContext
from gpu_fault.models import WorkflowStatus
from gpu_fault.regional import RemoteCommandResult, RemoteCommandStatus
from gpu_fault.regional_compatibility import CURRENT_REGIONAL_EXECUTOR_PROTOCOL_VERSION
from gpu_fault.store import PostgresStore, WorkflowLeaseError
from scripts.e2e.regional import run_ha001_control_plane_failover as ha001
from scripts.e2e.regional import run_ha006_executor_takeover as ha006
from scripts.e2e.regional import run_ha009_aurora_credential_rotation as ha009
from tests.regional._cov95_notify008_postgres import isolated_database

pytestmark = [
    pytest.mark.skipif(
        not os.environ.get("GPU_FAULT_TEST_POSTGRES_URL"),
        reason="requires an exclusive owned PostgreSQL grant",
    ),
    pytest.mark.allows_cluster_binaries("docker"),
]


@pytest.fixture(params=["legacy", "dual", "dedicated"])
def stores(
    request: pytest.FixtureRequest,
) -> Iterator[tuple[PostgresStore, PostgresStore]]:
    import psycopg

    from gpu_fault.state_table_migrate import backfill_state_table, set_state_table_mode

    with isolated_database() as factory:
        if request.param != "legacy":
            with psycopg.connect(factory.url, autocommit=True) as connection:
                for kind in ("workflow", "remote_command"):
                    set_state_table_mode(
                        connection, kind, "dual", expected_mode="legacy"
                    )
                    if request.param == "dedicated":
                        backfill_state_table(connection, kind)
                        set_state_table_mode(
                            connection,
                            kind,
                            "dedicated",
                            expected_mode="dual",
                            confirm_dedicated=True,
                        )
        with closing(factory()) as first, closing(factory()) as second:
            yield first, second


@pytest.fixture(params=["ha001", "ha006", "ha009"])
def seed(
    request: pytest.FixtureRequest,
    stores: tuple[PostgresStore, PostgresStore],
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, Any]:
    module = {"ha001": ha001, "ha006": ha006, "ha009": ha009}[request.param]
    first, _ = stores
    monkeypatch.setattr(
        ApplicationContext,
        "from_environment",
        classmethod(lambda cls: SimpleNamespace(store=first)),
    )

    def cpu_python(script: str, *arguments: str, **_kwargs: Any) -> dict[str, Any]:
        output = io.StringIO()
        with monkeypatch.context() as local, redirect_stdout(output):
            local.setattr(sys, "argv", ["native-ha-seed", *arguments])
            exec(compile(script, "<native-ha-seed>", "exec"), {})
        return json.loads(output.getvalue())

    monkeypatch.setattr(
        ha009.BASE if request.param == "ha009" else module, "cpu_python", cpu_python
    )
    if request.param == "ha001":
        result = ha001.seed_closure("native-ha001", "native-ha-cluster")
    elif request.param == "ha006":
        result = ha006.seed_command("native-ha006")
    else:
        result = ha009.seed_runtime_records("native-ha009")
    return {
        **result,
        "owner": "gpu-fault-ha005-noop" if request.param == "ha009" else module.OWNER,
        "seed_owner": (
            "gpu-fault-ha009-seed"
            if request.param == "ha009"
            else module.OWNER + "-seed"
        ),
    }


def test_pending_seed_survives_orphan_sweep_and_excludes_foreign_dispatcher(
    stores: tuple[PostgresStore, PostgresStore], seed: dict[str, Any]
) -> None:
    first, second = stores
    workflow = first.get_workflow(seed["workflow_id"])
    assert workflow.status is WorkflowStatus.PENDING
    assert workflow.execution_owner_id == seed["seed_owner"]
    assert workflow.execution_epoch == 1
    expiry = workflow.execution_lease_expires_at
    assert expiry is not None and expiry > datetime.now(timezone.utc)
    commands = first.list_remote_commands(workflow_request_ids=[workflow.request_id])
    expected_ids = set(seed.get("command_ids", [seed.get("command_id")]))
    assert {command.command_id for command in commands} == expected_ids
    assert all(
        command.workflow.status is WorkflowStatus.PENDING
        and command.workflow.execution_owner_id == seed["seed_owner"]
        and command.workflow.execution_epoch == 1
        and command.workflow.execution_lease_expires_at == expiry
        for command in commands
    ), "embedded workflow snapshots must retain the same bounded seed lease"
    if "notification_id" in seed:
        notification = first.get_notification(seed["notification_id"])
        assert notification.drill_id == "native-ha009"
        assert notification.deduplication_key == seed["deduplication_key"]
        assert notification.not_before is not None
        assert notification.not_before > notification.created_at

    with pytest.raises(WorkflowLeaseError, match="another executor"):
        second.claim_workflow(
            workflow.request_id, "foreign-dispatcher", workflow.fencing_token
        )
    sweep = second.cancel_orphaned_remote_commands(
        workflow, now=datetime.now(timezone.utc), actor="native-sweep"
    )
    assert sweep.command_ids == ()
    assert sweep.cancelled == sweep.cancellation_requested == 0

    for expected_command in sorted(commands, key=lambda item: item.step_index):
        claimed = second.claim_remote_commands(
            expected_command.cluster_id,
            "native-ha-probe",
            limit=1,
            lease_seconds=60,
            execution_owners={seed["owner"]},
            executor_protocol_version=CURRENT_REGIONAL_EXECUTOR_PROTOCOL_VERSION,
        )
        assert [command.command_id for command in claimed] == [
            expected_command.command_id
        ], "the seed lease must preserve each sequential probe claim"
        command = claimed[0]
        second.complete_remote_command(
            command.cluster_id,
            command.command_id,
            RemoteCommandResult(
                lease_token=command.lease_token,
                status=RemoteCommandStatus.SUCCEEDED,
                details={"simulated": True},
            ),
        )
        current = first.get_workflow(workflow.request_id)
        assert current.status is WorkflowStatus.PENDING
        assert (
            second.cancel_orphaned_remote_commands(
                current, now=datetime.now(timezone.utc), actor="native-sweep"
            ).command_ids
            == ()
        ), "completing a probe step must not orphan later probe commands"
    assert all(
        command.status is RemoteCommandStatus.SUCCEEDED
        for command in first.list_remote_commands(
            workflow_request_ids=[workflow.request_id]
        )
    ), "every seeded native command must retain its successful result"
    taken = second.claim_workflow(
        workflow.request_id,
        "foreign-dispatcher",
        workflow.fencing_token,
        now=expiry + timedelta(seconds=1),
    )
    assert taken.execution_owner_id == "foreign-dispatcher"
    assert taken.execution_epoch == workflow.execution_epoch + 1


def test_terminal_seed_control_really_orphans_pending_commands(
    stores: tuple[PostgresStore, PostgresStore], seed: dict[str, Any]
) -> None:
    first, second = stores
    previous = first.get_workflow(seed["workflow_id"])
    first.save_workflow(
        previous.model_copy(update={"status": WorkflowStatus.BLOCKED}),
        expected=previous,
    )
    workflow = second.get_workflow(previous.request_id)
    sweep = second.cancel_orphaned_remote_commands(
        workflow, now=datetime.now(timezone.utc), actor="native-sweep"
    )
    expected_ids = set(seed.get("command_ids", [seed.get("command_id")]))
    assert set(sweep.command_ids) == expected_ids
    assert sweep.cancelled == len(expected_ids)
    assert all(
        command.status is RemoteCommandStatus.FAILED
        for command in first.list_remote_commands(
            workflow_request_ids=[workflow.request_id]
        )
    ), "the BLOCKED-seed negative control must exercise actual orphan cancellation"

from __future__ import annotations

import contextlib
import io
import json
import sys
from types import SimpleNamespace
from typing import Any

import pytest

from gpu_fault.app import ApplicationContext
from scripts.e2e.regional import net_command_fixture as network
from scripts.e2e.regional import seeded_command_fixture as seeded
from tests.execution.test_cluster_executor_lease_and_report import remote_command


@pytest.mark.parametrize("reader", [seeded, network])
def test_snapshot_script_preserves_command_identity_without_exposing_lease_authority(
    monkeypatch: pytest.MonkeyPatch, reader: Any
) -> None:
    command = remote_command("snapshot-command")
    requested: list[str] = []

    def lookup(identifier: str) -> Any:
        requested.append(identifier)
        return command

    monkeypatch.setattr(
        ApplicationContext,
        "from_environment",
        lambda: SimpleNamespace(store=SimpleNamespace(get_remote_command=lookup)),
    )

    def execute(script: str, identifier: str) -> dict[str, Any]:
        monkeypatch.setattr(sys, "argv", ["snapshot", identifier])
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            exec(compile(script, "<snapshot-script>", "exec"), {})
        return json.loads(output.getvalue())

    monkeypatch.setattr(reader, "cpu_python", execute)
    result = reader.command_snapshot(command.command_id)
    assert requested == [command.command_id]
    assert {
        key: result[key]
        for key in (
            "command_id",
            "workflow_request_id",
            "incident_id",
            "cluster_id",
            "idempotency_key",
        )
    } == {
        "command_id": command.command_id,
        "workflow_request_id": command.workflow_request_id,
        "incident_id": command.incident_id,
        "cluster_id": command.cluster_id,
        "idempotency_key": command.idempotency_key,
    }
    assert result["node_ids"] == command.step.node_ids
    assert result["operation"] == command.step.operation.value
    assert "lease_token" not in result

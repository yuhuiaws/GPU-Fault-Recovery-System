from __future__ import annotations

import io
import json
import os
from contextlib import redirect_stdout
from datetime import UTC, datetime

import pytest

from gpu_fault.remote_command_models import RemoteCommandStatus
from gpu_fault.store import PostgresStore
from scripts.e2e.regional import net_command_fixture, run_ha005_rollout_continuity
from tests._builders import workflow_request
from tests.store import test_postgres_state_tables as state_tables
from tests.store.test_postgres_workflow_state_tables import select_mode
from tests.store.test_state_table_payload import command

database = state_tables.database
migration_database = state_tables.migration_database
POSTGRES_URL = os.getenv("GPU_FAULT_TEST_POSTGRES_URL")
pytestmark = pytest.mark.skipif(
    not POSTGRES_URL, reason="GPU_FAULT_TEST_POSTGRES_URL is not configured"
)


@pytest.mark.parametrize("mode", ["legacy", "dual", "dedicated"])
@pytest.mark.parametrize("reader", ["net", "ha005"])
def test_residual_audit_counts_unlinked_workflow_and_command_in_each_mode(
    migration_database, monkeypatch, mode: str, reader: str
) -> None:
    for kind in ("remote_command", "workflow"):
        select_mode(migration_database, kind, mode)
    assert POSTGRES_URL is not None, "residual audit checks require local PostgreSQL"
    monkeypatch.setenv("GPU_FAULT_STORE_URL", POSTGRES_URL)
    store = PostgresStore(POSTGRES_URL, initialize_schema=False)
    try:
        flow = workflow_request("workflow-ha005-residual", "incident-ha005-residual")
        store.save_workflow(flow)
        store.ensure_remote_command(
            command(datetime.now(UTC)).model_copy(
                update={
                    "command_id": "command-ha005-residual",
                    "workflow_request_id": flow.request_id,
                    "workflow": flow,
                    "status": RemoteCommandStatus.SUCCEEDED,
                }
            )
        )

        def cpu_python(script: str, *arguments: str) -> dict:
            output = io.StringIO()
            with monkeypatch.context() as scoped:
                scoped.setattr("sys.argv", ["audit", *arguments])
                with redirect_stdout(output):
                    exec(compile(script, "<local-residual-audit>", "exec"), {})
            return json.loads(output.getvalue())

        module = (
            net_command_fixture if reader == "net" else run_ha005_rollout_continuity
        )
        monkeypatch.setattr(module, "cpu_python", cpu_python)
        result = (
            module.database_residuals("ha005-residual")
            if reader == "net"
            else module.database_residuals()
        )
        assert result["objects"] == 2
        assert result["total"] == 2
    finally:
        store.close()

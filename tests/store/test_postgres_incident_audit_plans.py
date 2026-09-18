"""Native plans for the actual incident-audit statements and retained snapshots."""

from __future__ import annotations

import hashlib
import json
import os
import random
from datetime import UTC, datetime
from pathlib import Path

import pytest

from gpu_fault.models import WorkflowOperation, WorkflowStatus
from gpu_fault.remote_command_models import RemoteCommandStatus
from gpu_fault.state_table_migrate import backfill_state_table
from tests._builders import fault_incident, workflow_request, workflow_step
from tests.store import test_postgres_incident_audit_scripts as audit
from tests.store.test_postgres_state_tables import plan_nodes
from tests.store.test_postgres_workflow_state_tables import select_mode
from tests.store.test_state_table_payload import command

database = audit.database
migration_database = audit.migration_database
POSTGRES_URL = os.getenv("GPU_FAULT_TEST_POSTGRES_URL")
pytestmark = pytest.mark.skipif(
    not POSTGRES_URL, reason="GPU_FAULT_TEST_POSTGRES_URL is not configured"
)
HISTORY_ROWS = 1000
PREVIEW_SECTIONS = (
    "Incident",
    "Related object counts",
    "Workflow states",
    "Open or unknown remote commands; this result must be empty before purge",
    "External successor workflows; purge these incidents first",
)


def preview_query(title: str, incident_id: str):
    source = audit.PURGE_SQL.with_name("incident-audit-preview.sql").read_text()
    opening = f"\\echo '{title}'\n"
    start = source.index(opening) + len(opening)
    statement = source[start:].split("\n\\echo ", 1)[0]
    return audit.bind_incident_query(statement, incident_id)


def seed_retained_history(connection) -> None:
    at = datetime(2026, 9, 1, tzinfo=UTC)
    blob = random.Random(4105).randbytes(4096).hex()
    incident = fault_incident("history-incident", "history-event")
    flow = workflow_request(
        "history-template", incident.incident_id, status=WorkflowStatus.SUCCEEDED
    ).model_copy(
        update={
            "official_steps": [
                workflow_step(
                    WorkflowOperation.FREEZE_EVIDENCE,
                    parameters={"retained_snapshot": blob},
                )
            ],
            "created_at": at,
            "updated_at": at,
        }
    )
    remote = command(at).model_copy(
        update={
            "cluster_id": incident.cluster_id,
            "workflow_request_id": flow.request_id,
            "incident_id": incident.incident_id,
            "workflow": flow,
            "incident": incident,
            "step": flow.official_steps[0],
            "status": RemoteCommandStatus.SUCCEEDED,
        }
    )
    connection.execute(
        "INSERT INTO gpu_fault_objects(kind,key,payload) VALUES "
        "('incident',%s,%s::jsonb)",
        (incident.incident_id, incident.model_dump_json()),
    )
    connection.execute(
        "INSERT INTO gpu_fault_objects(kind,key,payload) "
        "SELECT 'workflow', 'history-workflow-' || n, "
        "jsonb_set(%s::jsonb, '{request_id}', to_jsonb('history-workflow-' || n)) "
        "FROM generate_series(1,%s) n",
        (flow.model_dump_json(), HISTORY_ROWS),
    )
    connection.execute(
        "INSERT INTO gpu_fault_objects(kind,key,payload) "
        "SELECT 'remote_command', 'history-command-' || n, "
        "jsonb_set(jsonb_set(jsonb_set(jsonb_set(%s::jsonb, "
        "'{command_id}', to_jsonb('history-command-' || n)), "
        "'{idempotency_key}', to_jsonb('history-command-' || n || '/0')), "
        "'{workflow_request_id}', to_jsonb('history-workflow-' || n)), "
        "'{workflow,request_id}', to_jsonb('history-workflow-' || n)) "
        "FROM generate_series(1,%s) n",
        (remote.model_dump_json(), HISTORY_ROWS),
    )


def plan_summary(plan) -> dict:
    return {
        "execution_ms": plan[0]["Execution Time"],
        "scans": [
            {
                key: node[key]
                for key in (
                    "Node Type",
                    "Relation Name",
                    "Index Name",
                    "Index Cond",
                    "Filter",
                    "Actual Rows",
                    "Actual Loops",
                    "Rows Removed by Filter",
                    "Shared Hit Blocks",
                    "Shared Read Blocks",
                )
                if key in node
            }
            for node in plan_nodes(plan)
            if "Scan" in node.get("Node Type", "")
        ],
    }


def assert_typed_state_filters(plans: dict, mode: str) -> None:
    for title, plan in plans.items():
        active = [node for node in plan_nodes(plan) if node.get("Actual Loops", 0)]
        for node in active:
            predicate = node.get("Filter", "")
            assert not any(
                expression in predicate
                for expression in (
                    "gpu_fault_workflow_payload",
                    "gpu_fault_remote_command_payload",
                    "jsonb_build_object",
                )
            ), (title, node.get("Node Type"), predicate)
            rebuilds_payload = any(
                " || jsonb_build_object(" in expression
                or "gpu_fault_workflow_payload(" in expression
                or "gpu_fault_remote_command_payload(" in expression
                for expression in node.get("Output", [])
            )
            if rebuilds_payload:
                assert node.get("Actual Rows", 0) <= 1, (
                    title,
                    "state JSON was rebuilt before its identity filter",
                    node.get("Node Type"),
                    node.get("Actual Rows"),
                )
                if title == "Related object counts":
                    assert node.get("Actual Rows", 0) == 0, (
                        "counting identities must not rebuild state snapshots"
                    )
    if mode != "dedicated":
        return
    for title, index in (
        ("Workflow states", "gpu_fault_workflows_incident"),
        ("Purge workflow capture", "gpu_fault_workflows_incident"),
        ("Purge remote_command capture", "gpu_fault_remote_commands_incident"),
        (PREVIEW_SECTIONS[3], "gpu_fault_remote_commands_incident"),
    ):
        assert any(
            node.get("Index Name") == index and node.get("Actual Loops", 0)
            for node in plan_nodes(plans[title])
        ), (title, index, plan_summary(plans[title]))


@pytest.mark.parametrize("mode", ["legacy", "dual", "dedicated"])
def test_incident_audit_native_statement_plans(
    migration_database, mode: str, request: pytest.FixtureRequest, tmp_path: Path
) -> None:
    from psycopg import sql

    connection = migration_database
    flow, remote = audit.seed_terminal_pair(connection, "legacy", "legacy")
    seed_retained_history(connection)
    successor = workflow_request(
        "external-successor", "external-incident", status=WorkflowStatus.SUCCEEDED
    ).model_copy(update={"predecessor_workflow_id": flow.request_id})
    connection.execute(
        "INSERT INTO gpu_fault_objects(kind,key,payload) VALUES "
        "('workflow',%s,%s::jsonb)",
        (successor.request_id, successor.model_dump_json()),
    )
    for kind in ("remote_command", "workflow"):
        select_mode(connection, kind, mode)
        if mode == "dual":
            backfill_state_table(connection, kind, batch_size=100, max_batches=1)
    for table in (
        "gpu_fault_objects",
        "gpu_fault_workflows",
        "gpu_fault_remote_commands",
    ):
        connection.execute(sql.SQL("ANALYZE {}").format(sql.Identifier(table)))

    statements = {
        title: preview_query(title, flow.incident_id) for title in PREVIEW_SECTIONS
    }
    statements["Export"] = audit.audit_query(connection, "export", flow.incident_id)
    for kind in ("workflow", "remote_command"):
        statements[f"Purge {kind} capture"] = audit.purge_capture_query(
            kind, flow.incident_id, select_only=True
        )
    statements["Purge successor guard"] = audit.successor_guard_query(flow.incident_id)
    plans = {}
    with connection.transaction(force_rollback=True):
        audit.capture_victims(connection, flow.incident_id)
        for title, statement in statements.items():
            plans[title] = connection.execute(
                sql.SQL("EXPLAIN (ANALYZE, BUFFERS, VERBOSE, FORMAT JSON) ") + statement
            ).fetchone()[0]
        assert connection.execute(
            "SELECT key FROM purge_workflow_ids ORDER BY key"
        ).fetchall() == [(flow.request_id,)]
        assert connection.execute(
            "SELECT key FROM purge_remote_command_ids ORDER BY key"
        ).fetchall() == [(remote.command_id,)]
        assert connection.execute(statements["Purge successor guard"]).fetchone() == (
            True,
        )
        assert connection.execute(
            "SELECT gpu_fault_delete_control_state('workflow', key, payload) "
            "FROM gpu_fault_workflow_records WHERE key=%s",
            (successor.request_id,),
        ).fetchone() == (True,)
        plans["Purge successor guard absent"] = connection.execute(
            sql.SQL("EXPLAIN (ANALYZE, BUFFERS, VERBOSE, FORMAT JSON) ")
            + statements["Purge successor guard"]
        ).fetchone()[0]
        assert connection.execute(statements["Purge successor guard"]).fetchone() == (
            False,
        )

    report = {
        "mode": mode,
        "history_rows_per_state_kind": HISTORY_ROWS,
        "scripts": {
            name: hashlib.sha256(
                audit.PURGE_SQL.with_name(f"incident-audit-{name}.sql").read_bytes()
            ).hexdigest()
            for name in ("preview", "export", "purge")
        },
        "summary": {title: plan_summary(plan) for title, plan in plans.items()},
        "plans": plans,
    }
    junit = request.config.getoption("xmlpath")
    output = Path(junit).parent if junit else tmp_path
    (output / f"incident-audit-plans-{mode}.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    assert_typed_state_filters(plans, mode)

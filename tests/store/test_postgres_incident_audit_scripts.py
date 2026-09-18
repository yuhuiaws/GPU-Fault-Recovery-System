from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path

import pytest

from gpu_fault.models import WorkflowStatus
from gpu_fault.remote_command_models import RemoteCommandStatus
from gpu_fault.store import PostgresStore
from tests._builders import fault_incident, workflow_request
from tests.store import test_postgres_state_tables as state_tables
from tests.store.test_postgres_workflow_state_tables import select_mode
from tests.store.test_state_table_payload import command

database = state_tables.database
migration_database = state_tables.migration_database
POSTGRES_URL = os.getenv("GPU_FAULT_TEST_POSTGRES_URL")
pytestmark = pytest.mark.skipif(
    not POSTGRES_URL, reason="GPU_FAULT_TEST_POSTGRES_URL is not configured"
)
PURGE_SQL = (
    Path(__file__).resolve().parents[2] / "scripts/postgres/incident-audit-purge.sql"
)


def sql_block(tag: str, operation: str = "purge") -> str:
    source = PURGE_SQL.with_name(f"incident-audit-{operation}.sql").read_text()
    opening = f"DO ${tag}$"
    closing = f"${tag}$;"
    start = source.index(opening)
    end = source.index(closing, start + len(opening)) + len(closing)
    return source[start:end]


def bind_incident_query(statement: str, incident_id: str):
    from psycopg import sql

    return sql.SQL(statement.replace(":'incident_id'", "{incident_id}")).format(
        incident_id=sql.Literal(incident_id)
    )


def purge_capture_query(kind: str, incident_id: str, *, select_only: bool = False):
    source = PURGE_SQL.read_text()
    opening = f"CREATE TEMP TABLE purge_{kind}_ids ON COMMIT DROP AS\n"
    start = source.index(opening)
    statement = source[start:].split("\n\n", 1)[0]
    if select_only:
        statement = statement.removeprefix(opening)
    return bind_incident_query(statement, incident_id)


def capture_victims(connection, incident_id: str = "audit-incident") -> None:
    for kind in ("workflow", "remote_command"):
        connection.execute(purge_capture_query(kind, incident_id))


def successor_guard_query(incident_id: str):
    source = PURGE_SQL.read_text()
    start = source.index("\nSELECT EXISTS (\n", source.index("$purge_terminal_guard$;"))
    end = source.index("\n\\gset", start)
    return bind_incident_query(source[start:end], incident_id)


def seed_terminal_pair(connection, workflow_mode: str, command_mode: str):
    select_mode(connection, "remote_command", command_mode)
    select_mode(connection, "workflow", workflow_mode)
    assert POSTGRES_URL is not None, "the audit fixture requires local PostgreSQL"
    store = PostgresStore(POSTGRES_URL, initialize_schema=False)
    try:
        flow = workflow_request(
            "audit-workflow", "audit-incident", status=WorkflowStatus.SUCCEEDED
        )
        incident = fault_incident("audit-incident", "audit-event")
        store.save_incident(incident)
        store.save_workflow(flow)
        remote = command(datetime.now(UTC)).model_copy(
            update={
                "workflow_request_id": flow.request_id,
                "workflow": flow,
                "incident_id": incident.incident_id,
                "incident": incident,
                "status": RemoteCommandStatus.SUCCEEDED,
            }
        )
        store.ensure_remote_command(remote)
        return flow, remote
    finally:
        store.close()


@pytest.mark.parametrize("workflow_mode", ["legacy", "dual", "dedicated"])
@pytest.mark.parametrize("command_mode", ["legacy", "dual", "dedicated"])
def test_purge_deletes_each_authoritative_state_row(
    migration_database, workflow_mode: str, command_mode: str
) -> None:
    connection = migration_database
    seed_terminal_pair(connection, workflow_mode, command_mode)
    with connection.transaction():
        capture_victims(connection)
        connection.execute(sql_block("purge_terminal_guard"))
        connection.execute(sql_block("purge_control_state"))
    assert connection.execute(
        "SELECT kind FROM gpu_fault_control_records ORDER BY kind"
    ).fetchall() == [("incident",)]


@pytest.mark.parametrize("mode", ["legacy", "dual", "dedicated"])
def test_a_conditional_delete_miss_rolls_back_the_entire_purge(
    migration_database, mode: str
) -> None:
    import psycopg

    connection = migration_database
    flow, _ = seed_terminal_pair(connection, mode, mode)
    with pytest.raises(psycopg.errors.RaiseException, match="could not be deleted"):
        with connection.transaction():
            capture_victims(connection)
            connection.execute(
                "UPDATE purge_workflow_ids SET payload="
                "jsonb_set(payload,'{merge_revision}','99'::jsonb) WHERE key=%s",
                (flow.request_id,),
            )
            connection.execute(
                "DELETE FROM gpu_fault_objects WHERE kind='incident' AND key=%s",
                (flow.incident_id,),
            )
            connection.execute(sql_block("purge_control_state"))
    assert connection.execute(
        "SELECT kind FROM gpu_fault_control_records ORDER BY kind"
    ).fetchall() == [("incident",), ("remote_command",), ("workflow",)]


@pytest.mark.parametrize(
    "legacy_style", ["missing_revision", "missing_lease_owner", "whole_second"]
)
def test_dual_legacy_json_difference_is_deleted_atomically(
    migration_database, legacy_style: str
) -> None:
    connection = migration_database
    flow, _ = seed_terminal_pair(connection, "dual", "dual")
    payload = flow.model_dump(mode="json")
    if legacy_style == "whole_second":
        for field in ("created_at", "updated_at"):
            payload[field] = datetime.fromisoformat(payload[field]).isoformat(
                timespec="seconds"
            )
    else:
        del payload[
            "merge_revision"
            if legacy_style == "missing_revision"
            else "execution_owner_id"
        ]
    connection.execute(
        "UPDATE gpu_fault_objects SET payload=%s::jsonb "
        "WHERE kind='workflow' AND key=%s",
        (json.dumps(payload), flow.request_id),
    )
    with connection.transaction():
        capture_victims(connection)
        connection.execute(sql_block("purge_terminal_guard"))
        connection.execute(
            "DELETE FROM gpu_fault_objects WHERE kind='incident' AND key=%s",
            (flow.incident_id,),
        )
        connection.execute(sql_block("purge_control_state"))
    assert (
        connection.execute(
            "SELECT kind FROM gpu_fault_control_records ORDER BY kind"
        ).fetchall()
        == []
    ), "equivalent legacy JSON must not leave orphan audit records"


def audit_query(connection, name: str, incident_id: str):
    source = (PURGE_SQL.parent / f"incident-audit-{name}.sql").read_text()
    start = source.index("\nWITH\n" if name == "export" else "\nSELECT\n")
    statement = "\n".join(
        line for line in source[start:].splitlines() if not line.startswith("\\")
    )
    return bind_incident_query(statement, incident_id)


@pytest.mark.parametrize("workflow_mode", ["legacy", "dual", "dedicated"])
@pytest.mark.parametrize("command_mode", ["legacy", "dual", "dedicated"])
def test_incident_export_reads_each_logical_record_once(
    migration_database, workflow_mode: str, command_mode: str
) -> None:
    connection = migration_database
    flow, remote = seed_terminal_pair(connection, workflow_mode, command_mode)

    rows = connection.execute(
        audit_query(connection, "export", flow.incident_id)
    ).fetchall()
    records = [json.loads(row[0]) for row in rows]
    objects = [record for record in records if record["table"] == "gpu_fault_objects"]

    assert [(record["kind"], record["key"]) for record in objects] == [
        ("incident", flow.incident_id),
        ("remote_command", remote.command_id),
        ("workflow", flow.request_id),
    ]
    assert objects[-1]["payload"] == flow.model_dump(mode="json")
    assert objects[1]["payload"] == remote.model_dump(mode="json")
    assert objects[0]["payload"] == remote.incident.model_dump(mode="json")


@pytest.mark.parametrize("workflow_mode", ["legacy", "dual", "dedicated"])
@pytest.mark.parametrize("command_mode", ["legacy", "dual", "dedicated"])
def test_incident_preview_counts_migrated_workflow_and_command(
    migration_database, workflow_mode: str, command_mode: str
) -> None:
    connection = migration_database
    flow, _ = seed_terminal_pair(connection, workflow_mode, command_mode)

    cursor = connection.execute(audit_query(connection, "preview", flow.incident_id))
    assert cursor.fetchone()[0] == flow.incident_id
    assert cursor.nextset(), "the preview omitted related object counts"
    counts = dict(cursor.fetchall())
    assert counts["object:workflow"] == counts["object:remote_command"] == 1
    assert cursor.nextset(), "the preview omitted workflow states"
    assert cursor.fetchone()[:2] == (flow.request_id, "SUCCEEDED")
    assert cursor.nextset(), "the preview omitted the open-command safety check"
    assert cursor.fetchall() == []
    assert cursor.nextset(), "the preview omitted the external-successor safety check"
    assert cursor.fetchall() == []


@pytest.mark.parametrize(
    ("kind", "payload"),
    [
        ("workflow", {"status": "PENDING"}),
        ("workflow", {"status": "BLOCKED", "blocked_kind": "NEEDS_OPERATOR"}),
        ("workflow", {"status": "FAILED", "failure_handled_at": None}),
        ("workflow", {"status": "FAILED", "failure_handled_at": ""}),
        ("workflow", {"status": "UNKNOWN"}),
        ("workflow", {}),
        ("remote_command", {"status": "LEASED"}),
        ("remote_command", {"status": "UNKNOWN"}),
        ("remote_command", {}),
    ],
)
def test_purge_terminal_guard_rejects_unsettled_or_unknown_state(
    migration_database, kind: str, payload: dict
) -> None:
    import psycopg

    connection = migration_database
    with pytest.raises(psycopg.errors.RaiseException, match="refusing purge"):
        with connection.transaction():
            capture_victims(connection)
            connection.execute(
                f"INSERT INTO purge_{kind}_ids(key,payload) VALUES (%s,%s::jsonb)",
                ("unsettled-audit-record", json.dumps(payload)),
            )
            connection.execute(sql_block("purge_terminal_guard"))


@pytest.mark.parametrize(
    ("operation", "tag"),
    [
        ("preview", "audit_incident_argument_guard"),
        ("export", "audit_incident_argument_guard"),
        ("purge", "audit_incident_argument_guard"),
        ("purge", "audit_confirmation_argument_guard"),
        ("purge", "audit_confirmation_guard"),
        ("purge", "audit_missing_incident_guard"),
        ("purge", "audit_successor_guard"),
    ],
)
def test_psql_refusal_guards_raise_instead_of_quitting_successfully(
    migration_database, operation: str, tag: str
) -> None:
    import psycopg

    with pytest.raises(psycopg.errors.RaiseException):
        with migration_database.transaction():
            migration_database.execute(sql_block(tag, operation))


@pytest.mark.parametrize("status", [None, "UNKNOWN"])
def test_preview_includes_unknown_command_status_as_a_purge_blocker(
    migration_database, status: str | None
) -> None:
    connection = migration_database
    flow, remote = seed_terminal_pair(connection, "legacy", "legacy")
    connection.execute(
        "UPDATE gpu_fault_objects SET payload=payload || %s::jsonb "
        "WHERE kind='remote_command' AND key=%s",
        (json.dumps({"status": status}), remote.command_id),
    )
    cursor = connection.execute(audit_query(connection, "preview", flow.incident_id))
    for _ in range(3):
        assert cursor.nextset(), "preview omitted a required result set"
    assert cursor.fetchall() == [(remote.command_id, status, flow.request_id)]


@pytest.mark.parametrize("owner", ["missing", "null"])
def test_successor_with_unknown_incident_owner_blocks_purge(
    migration_database, owner: str
) -> None:
    connection = migration_database
    flow, remote = seed_terminal_pair(connection, "legacy", "legacy")
    successor = workflow_request(
        "ownerless-successor", "temporary-owner", status=WorkflowStatus.SUCCEEDED
    ).model_copy(update={"predecessor_workflow_id": flow.request_id})
    payload = successor.model_dump(mode="json")
    if owner == "missing":
        del payload["incident_id"]
    else:
        payload["incident_id"] = None
    connection.execute(
        "INSERT INTO gpu_fault_objects(kind,key,payload) VALUES "
        "('workflow',%s,%s::jsonb)",
        (successor.request_id, json.dumps(payload)),
    )
    preview = connection.execute(audit_query(connection, "preview", flow.incident_id))
    for _ in range(4):
        assert preview.nextset(), "preview omitted its successor result"
    assert preview.fetchall() == [
        (successor.request_id, None, "SUCCEEDED", flow.request_id)
    ]
    with connection.transaction(force_rollback=True):
        capture_victims(connection, flow.incident_id)
        connection.execute(sql_block("purge_terminal_guard"))
        blocked = connection.execute(
            successor_guard_query(flow.incident_id)
        ).fetchone()[0]
        if not blocked:
            connection.execute(sql_block("purge_control_state"))
        remaining = connection.execute(
            "SELECT key FROM gpu_fault_control_records "
            "WHERE kind IN ('workflow','remote_command') ORDER BY key"
        ).fetchall()
    assert blocked is True, (
        "terminal/CAS guards allowed deleting the referenced records; "
        f"remaining state keys={remaining}"
    )
    assert remaining == sorted(
        [(flow.request_id,), (remote.command_id,), (successor.request_id,)]
    )


def complete_purge(connection, incident_id: str) -> None:
    with connection.transaction():
        connection.execute("SET TRANSACTION ISOLATION LEVEL SERIALIZABLE")
        for kind in (
            "workflow",
            "remote_command",
            "notification",
            "plan",
            "decision",
            "diagnostic",
            "event",
        ):
            connection.execute(purge_capture_query(kind, incident_id))
        connection.execute(sql_block("purge_terminal_guard"))
        if connection.execute(successor_guard_query(incident_id)).fetchone()[0]:
            connection.execute(sql_block("audit_successor_guard"))
        source = PURGE_SQL.read_text()
        start = source.index("DO $purge_control_state$")
        end = source.index("\nCOMMIT;", start)
        connection.execute(bind_incident_query(source[start:end], incident_id))

from __future__ import annotations

import json
import os
import re
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
PREDICATES_SQL = PURGE_SQL.with_name("incident-audit-predicates.sql")
UNSETTLED_WORKFLOWS = (
    "Unsettled or unhandled workflows; this result must be empty before purge"
)
OPEN_REMOTE_COMMANDS = (
    "Open or unknown remote commands; this result must be empty before purge"
)


def run_includes(connection, operation: str) -> None:
    """Run what an audit script ``\\ir``-includes, as psql does in its session.

    The preview and the purge include the shared purge predicates: session
    functions their statements call, so a statement lifted out of either
    script needs them defined on the same connection first.
    """

    source = PURGE_SQL.with_name(f"incident-audit-{operation}.sql").read_text()
    for line in source.splitlines():
        if line.startswith("\\ir "):
            included = PURGE_SQL.with_name(line.removeprefix("\\ir ").strip())
            connection.execute(included.read_text())


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
    run_includes(connection, "purge")
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
    run_includes(connection, name)
    source = (PURGE_SQL.parent / f"incident-audit-{name}.sql").read_text()
    start = source.index("\nWITH\n" if name == "export" else "\nSELECT\n")
    statement = "\n".join(
        line for line in source[start:].splitlines() if not line.startswith("\\")
    )
    return bind_incident_query(statement, incident_id)


def preview_section(connection, incident_id: str, title: str) -> list[tuple]:
    """The rows the preview prints under ``title``, its result sets in order."""

    source = PURGE_SQL.with_name("incident-audit-preview.sql").read_text()
    titles = re.findall(r"^\\echo '(.*)'$", source, flags=re.MULTILINE)
    cursor = connection.execute(audit_query(connection, "preview", incident_id))
    for _ in range(titles.index(title)):
        assert cursor.nextset(), f"the preview ended before {title!r}"
    return cursor.fetchall()


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
    assert cursor.nextset(), "the preview omitted the unsettled-workflow safety check"
    assert cursor.fetchall() == []
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
    assert preview_section(connection, flow.incident_id, OPEN_REMOTE_COMMANDS) == [
        (remote.command_id, status, flow.request_id)
    ]


# A workflow shape: (status, failure_handled_at, blocks the purge, legacy JSON
# only). The empty and unknown values cannot reach the typed dedicated table
# (or its dual-mode mirror), so they exist only as legacy rows.
WORKFLOW_PURGE_SHAPES = {
    "succeeded": ("SUCCEEDED", None, False, False),
    "superseded": ("SUPERSEDED", None, False, False),
    "failed-handled": ("FAILED", "2026-09-01T00:00:00+00:00", False, False),
    "failed-unhandled": ("FAILED", None, True, False),
    "failed-empty-handled-at": ("FAILED", "", True, True),
    "pending": ("PENDING", None, True, False),
    "blocked": ("BLOCKED", None, True, False),
    "unknown-status": ("UNKNOWN", None, True, True),
}


def seed_workflow_shape(connection, mode: str, shape: str):
    """One incident with one workflow of ``shape``, written for ``mode``."""

    status, handled_at, _blocks, legacy_only = WORKFLOW_PURGE_SHAPES[shape]
    select_mode(connection, "workflow", mode)
    incident = fault_incident("shape-incident", "shape-event")
    flow = workflow_request(
        "shape-workflow",
        incident.incident_id,
        status=WorkflowStatus.SUCCEEDED if legacy_only else WorkflowStatus(status),
        failure_handled_at=(
            datetime.fromisoformat(handled_at)
            if handled_at and not legacy_only
            else None
        ),
    )
    assert POSTGRES_URL is not None, "the audit fixture requires local PostgreSQL"
    store = PostgresStore(POSTGRES_URL, initialize_schema=False)
    try:
        store.save_incident(incident)
        if not legacy_only:
            store.save_workflow(flow)
    finally:
        store.close()
    if legacy_only:
        payload = flow.model_dump(mode="json")
        payload.update(status=status, failure_handled_at=handled_at)
        connection.execute(
            "INSERT INTO gpu_fault_objects(kind,key,payload) VALUES "
            "('workflow',%s,%s::jsonb)",
            (flow.request_id, json.dumps(payload)),
        )
    return flow


def purge_guard_refusal(connection, incident_id: str) -> str | None:
    """The terminal guard's refusal over the purge's own capture, or ``None``."""

    import psycopg

    try:
        with connection.transaction(force_rollback=True):
            capture_victims(connection, incident_id)
            connection.execute(sql_block("purge_terminal_guard"))
    except psycopg.errors.RaiseException as exc:
        return str(exc)
    return None


@pytest.mark.parametrize(
    ("mode", "shape"),
    [
        (mode, shape)
        for mode in ("legacy", "dual", "dedicated")
        for shape, (_status, _handled, _blocks, legacy_only) in (
            WORKFLOW_PURGE_SHAPES.items()
        )
        if mode == "legacy" or not legacy_only
    ],
)
def test_preview_lists_exactly_the_workflows_the_purge_guard_refuses(
    migration_database, mode: str, shape: str
) -> None:
    """One predicate: what the preview lists is what the purge refuses, by
    status and by ``failure_handled_at``, in every workflow state mode."""

    connection = migration_database
    status, handled_at, blocks, _legacy_only = WORKFLOW_PURGE_SHAPES[shape]
    flow = seed_workflow_shape(connection, mode, shape)

    listed = preview_section(connection, flow.incident_id, UNSETTLED_WORKFLOWS)
    refusal = purge_guard_refusal(connection, flow.incident_id)

    assert listed == ([(flow.request_id, status, handled_at)] if blocks else []), (
        f"the preview listed {listed} for a {shape} workflow in {mode} mode"
    )
    assert (refusal is not None) is blocks, (
        f"the purge guard {'refused' if refusal else 'accepted'} a {shape} "
        f"workflow in {mode} mode: {refusal}"
    )
    assert refusal is None or "non-terminal or unhandled workflow" in refusal, refusal


@pytest.mark.parametrize(
    ("status", "blocks"),
    [
        ("SUCCEEDED", False),
        ("FAILED", False),
        ("PENDING", True),
        ("WAITING", True),
        ("LEASED", True),
        (None, True),
        ("UNKNOWN", True),
    ],
)
def test_preview_lists_exactly_the_remote_commands_the_purge_guard_refuses(
    migration_database, status: str | None, blocks: bool
) -> None:
    connection = migration_database
    flow, remote = seed_terminal_pair(connection, "legacy", "legacy")
    connection.execute(
        "UPDATE gpu_fault_objects SET payload=payload || %s::jsonb "
        "WHERE kind='remote_command' AND key=%s",
        (json.dumps({"status": status}), remote.command_id),
    )

    listed = preview_section(connection, flow.incident_id, OPEN_REMOTE_COMMANDS)
    refusal = purge_guard_refusal(connection, flow.incident_id)

    assert listed == (
        [(remote.command_id, status, flow.request_id)] if blocks else []
    ), f"the preview listed {listed} for a {status!r} remote command"
    assert (refusal is not None) is blocks, (
        f"the purge guard {'refused' if refusal else 'accepted'} a {status!r} "
        f"remote command: {refusal}"
    )
    assert refusal is None or "open or unknown remote command" in refusal, refusal


@pytest.mark.parametrize("operation", ["preview", "purge"])
def test_preview_and_purge_read_the_one_predicate_definition(operation: str) -> None:
    """Neither script spells the status vocabulary itself: both include the
    predicate file before their first statement that calls it (the purge also
    before its transaction), and that file is plain SQL psql and the tests run
    alike."""

    source = PURGE_SQL.with_name(f"incident-audit-{operation}.sql").read_text()
    include = f"\\ir {PREDICATES_SQL.name}"
    assert source.splitlines().count(include) == 1, f"{operation} must include once"
    for function in (
        "pg_temp.gpu_fault_audit_workflow_blocks_purge(",
        "pg_temp.gpu_fault_audit_remote_command_blocks_purge(",
    ):
        assert source.index(include) < source.index(function), (operation, function)
    if operation == "purge":
        assert source.index(include) < source.index("BEGIN ISOLATION LEVEL"), (
            "the purge defines its predicates before its transaction starts"
        )
    for literal in ("'SUCCEEDED'", "'SUPERSEDED'", "'FAILED'"):
        assert literal not in source, f"{operation} spells {literal} itself"
    predicates = PREDICATES_SQL.read_text()
    assert [line for line in predicates.splitlines() if line.startswith("\\")] == [], (
        "the predicate file is plain SQL"
    )


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
    assert preview_section(
        connection,
        flow.incident_id,
        "External successor workflows; purge these incidents first",
    ) == [(successor.request_id, None, "SUCCEEDED", flow.request_id)]
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
    run_includes(connection, "purge")
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

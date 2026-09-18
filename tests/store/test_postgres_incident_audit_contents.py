"""Complete incident bundles retain exclusions across independent state modes."""

from __future__ import annotations

import json
import os

import pytest

from gpu_fault.models import WorkflowStatus
from gpu_fault.store import PostgresStore
from tests._builders import fault_incident, workflow_request
from tests.store import test_postgres_incident_audit_scripts as audit

database = audit.database
migration_database = audit.migration_database
POSTGRES_URL = os.getenv("GPU_FAULT_TEST_POSTGRES_URL")
pytestmark = pytest.mark.skipif(
    not POSTGRES_URL, reason="GPU_FAULT_TEST_POSTGRES_URL is not configured"
)


def object_records(connection) -> dict:
    return {
        (kind, key): payload
        for kind, key, payload in connection.execute(
            "SELECT kind,key,payload FROM gpu_fault_control_records"
        ).fetchall()
    }


def link_records(connection) -> dict:
    return {
        (kind, key): value
        for kind, key, value in connection.execute(
            "SELECT kind,key,value FROM gpu_fault_links"
        ).fetchall()
    }


def linked_audit_records(connection, incident_id: str) -> tuple[dict, dict]:
    objects = {
        ("notification", "audit-notification"): {"incident_id": incident_id},
        ("notification_delivery", "audit-notification"): {"status": "SENT"},
        ("notification_result", "audit-notification"): {"status": "SENT"},
        ("plan", "audit-plan"): {"incident_id": incident_id},
        ("decision", "audit-completion"): {
            "recovery_plan_id": "audit-plan",
            "diagnostic_request_id": "audit-diagnostic",
        },
        ("event", "audit-completion"): {"terminal_status": "SUCCEEDED"},
        ("diagnostic", "audit-diagnostic"): {"status": "SUCCEEDED"},
        ("triage", "audit-diagnostic"): {"summary": "retained evidence"},
        ("marker", "audit-marker"): {"incident_id": incident_id},
        ("xid_correlation_event", "audit-event"): {"event_id": "audit-event"},
        ("xid_policy_decision", "audit-event"): {"event_id": "audit-event"},
        ("xid_correlation", "audit-event"): {"event_id": "audit-event"},
    }
    links = {
        ("incident_by_event", "audit-event"): incident_id,
        ("replacement_fault_group", "audit-replacement"): incident_id,
        ("sxid_fault_group", "audit-sxid"): incident_id,
        ("notification_dedup", "audit-notification-dedup"): "audit-notification",
    }
    for (kind, key), payload in objects.items():
        connection.execute(
            "INSERT INTO gpu_fault_objects(kind,key,payload) VALUES (%s,%s,%s::jsonb)",
            (kind, key, json.dumps(payload)),
        )
    for (kind, key), value in links.items():
        connection.execute(
            "INSERT INTO gpu_fault_links(kind,key,value) VALUES (%s,%s,%s) "
            "ON CONFLICT(kind,key) DO UPDATE SET value=excluded.value",
            (kind, key, value),
        )
    return objects, links


@pytest.mark.parametrize("workflow_mode", ["legacy", "dual", "dedicated"])
@pytest.mark.parametrize("command_mode", ["legacy", "dual", "dedicated"])
def test_complete_export_and_purge_preserve_unrelated_records(
    migration_database, workflow_mode: str, command_mode: str
) -> None:
    connection = migration_database
    connection.execute("DELETE FROM gpu_fault_links")
    flow, remote = audit.seed_terminal_pair(connection, workflow_mode, command_mode)
    expected_objects, expected_links = linked_audit_records(
        connection, flow.incident_id
    )
    expected_objects.update(
        {
            ("incident", flow.incident_id): remote.incident.model_dump(mode="json"),
            ("workflow", flow.request_id): flow.model_dump(mode="json"),
            ("remote_command", remote.command_id): remote.model_dump(mode="json"),
        }
    )
    excluded = {
        ("notification_result", "unlinked-result"): {"incident_id": flow.incident_id},
        ("event", "unlinked-event"): {"incident_id": flow.incident_id},
        ("diagnostic", "unlinked-diagnostic"): {"incident_id": flow.incident_id},
        ("marker", "other-marker"): {"incident_id": flow.incident_id + "-other"},
    }
    for (kind, key), payload in excluded.items():
        connection.execute(
            "INSERT INTO gpu_fault_objects(kind,key,payload) VALUES (%s,%s,%s::jsonb)",
            (kind, key, json.dumps(payload)),
        )
    assert POSTGRES_URL is not None
    other_incident = fault_incident("other-incident", "other-event")
    other_flow = workflow_request(
        "other-workflow", other_incident.incident_id, status=WorkflowStatus.SUCCEEDED
    )
    other_remote = remote.model_copy(
        update={
            "command_id": "other-command",
            "idempotency_key": "other-command/0",
            "incident_id": other_incident.incident_id,
            "workflow_request_id": other_flow.request_id,
            "incident": other_incident,
            "workflow": other_flow,
        }
    )
    store = PostgresStore(POSTGRES_URL, initialize_schema=False)
    try:
        store.save_incident(other_incident)
        store.save_workflow(other_flow)
        store.ensure_remote_command(other_remote)
    finally:
        store.close()
    before_objects = object_records(connection)
    before_links = link_records(connection)
    records = [
        json.loads(row[0])
        for row in connection.execute(
            audit.audit_query(connection, "export", flow.incident_id)
        ).fetchall()
    ]
    actual_objects = {
        (record["kind"], record["key"]): record["payload"]
        for record in records
        if record["table"] == "gpu_fault_objects"
    }
    actual_links = {
        (record["kind"], record["key"]): record["value"]
        for record in records
        if record["table"] == "gpu_fault_links"
    }
    assert actual_objects == expected_objects
    assert actual_links == expected_links
    assert len(records) == len(expected_objects) + len(expected_links)
    identities = [
        (record["table"], record["kind"], record["key"]) for record in records
    ]
    assert identities == sorted(identities)

    audit.complete_purge(connection, flow.incident_id)

    remaining_objects = object_records(connection)
    remaining_links = link_records(connection)
    assert remaining_objects == {
        key: payload
        for key, payload in before_objects.items()
        if key not in expected_objects
    }
    assert remaining_links == {
        key: value for key, value in before_links.items() if key not in expected_links
    }
    assert all(remaining_objects[key] == value for key, value in excluded.items()), (
        "purge changed or deleted an unrelated or unlinked audit record"
    )

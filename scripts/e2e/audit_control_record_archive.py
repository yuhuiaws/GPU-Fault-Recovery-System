from __future__ import annotations

import gzip
import importlib.util
import json
import os
from datetime import datetime, timezone
from urllib.parse import urlparse
from uuid import uuid4

import boto3
import psycopg

from gpu_fault.models import (
    AdvisoryNotification,
    FaultIncident,
    IncidentState,
    MarkerScope,
    NodeMarker,
    RecoveryAction,
    Severity,
    WorkflowRequest,
    WorkflowStatus,
)
from gpu_fault.store import PostgresStore


def load_module(path: str):
    spec = importlib.util.spec_from_file_location(
        "gpu_fault.control_record_archive_probe", path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def main(module_path: str, archive_uri: str) -> None:
    module = load_module(module_path)
    dsn = os.environ["GPU_FAULT_STORE_URL"]
    suffix = uuid4().hex
    incident_id = f"audit-archive-{suffix}"
    workflow_id = f"workflow-{incident_id}"
    event_id = f"event-{incident_id}"
    marker_id = f"marker-{incident_id}"
    notification_id = f"notification-{incident_id}"
    active_incident_id = f"audit-archive-active-{suffix}"
    active_workflow_id = f"workflow-{active_incident_id}"
    old = datetime(2020, 1, 1, tzinfo=timezone.utc)
    store = PostgresStore(dsn)
    s3 = boto3.client("s3")
    archive_key = None
    parsed = urlparse(archive_uri)
    try:
        incident = FaultIncident(
            incident_id=incident_id,
            event_id=event_id,
            event_type="NODE_HEALTH",
            cluster_id="audit-cluster",
            node_ids=["audit-node"],
            policy_version="audit",
            policy_source="AUDIT",
            effective_action=RecoveryAction.RUN_DIAGNOSTICS,
            state=IncidentState.RECOVERED,
            workflow_request_id=workflow_id,
            created_at=old,
            updated_at=old,
        )
        workflow = WorkflowRequest(
            request_id=workflow_id,
            incident_id=incident_id,
            status=WorkflowStatus.SUCCEEDED,
            fencing_token=1,
            created_at=old,
            updated_at=old,
        )
        store.save_incident_and_workflow(incident, workflow)
        store.add_marker(
            NodeMarker(
                marker_id=marker_id,
                source="audit",
                trusted=True,
                incident_id=incident_id,
                observed_at=old,
                expires_at=old.replace(year=2021),
                scope=MarkerScope(node_ids=["audit-node"]),
                severity=Severity.INFO,
                recommended_action=RecoveryAction.RUN_DIAGNOSTICS,
                mapping_version="audit",
                active=False,
            )
        )
        notification = AdvisoryNotification(
            notification_id=notification_id,
            deduplication_key=f"dedup-{incident_id}",
            cluster_name="audit-cluster",
            incident_id=incident_id,
            subject="audit",
            body_text="audit archive test",
            support_case_draft="audit archive test",
        )
        store.save_notification_if_absent(notification)

        active_incident = incident.model_copy(
            update={
                "incident_id": active_incident_id,
                "event_id": f"event-{active_incident_id}",
                "workflow_request_id": active_workflow_id,
                "state": IncidentState.ACTION_PENDING,
            }
        )
        active_workflow = workflow.model_copy(
            update={
                "request_id": active_workflow_id,
                "incident_id": active_incident_id,
                "status": WorkflowStatus.RUNNING,
            }
        )
        store.save_incident_and_workflow(active_incident, active_workflow)

        archiver = module.ControlRecordArchiver(dsn, archive_uri, s3_client=s3)
        try:
            archiver.archive_one(active_incident_id)
        except module.ArchiveSafetyError:
            pass
        else:
            raise AssertionError("active incident was archived")

        uri = archiver.archive_one(incident_id)
        archive_key = uri.split("/", 3)[3]
        body = gzip.decompress(
            s3.get_object(Bucket=parsed.netloc, Key=archive_key)["Body"].read()
        ).decode()
        records = [json.loads(line) for line in body.splitlines()]
        assert any(item["kind"] == "incident" for item in records)
        assert any(item["kind"] == "workflow" for item in records)
        assert any(item["kind"] == "marker" for item in records)
        assert any(item["kind"] == "notification" for item in records)
        with psycopg.connect(dsn) as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT count(*) FROM gpu_fault_objects
                    WHERE key=ANY(%s)
                    """,
                    ([incident_id, workflow_id, marker_id, notification_id],),
                )
                assert cursor.fetchone()[0] == 0
        print(
            "PASS",
            {
                "archive_uri": uri,
                "records": len(records),
                "active_guard": True,
                "database_deleted": True,
            },
        )
    finally:
        store.close()
        with psycopg.connect(dsn) as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    "DELETE FROM gpu_fault_links WHERE key LIKE %s OR value LIKE %s",
                    (f"%{suffix}%", f"%{suffix}%"),
                )
                cursor.execute(
                    "DELETE FROM gpu_fault_objects WHERE key LIKE %s "
                    "OR payload::text LIKE %s",
                    (f"%{suffix}%", f"%{suffix}%"),
                )
        if archive_key:
            s3.delete_object(Bucket=parsed.netloc, Key=archive_key)


if __name__ == "__main__":
    import sys

    main(sys.argv[1], sys.argv[2])

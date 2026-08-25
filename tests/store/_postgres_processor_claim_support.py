"""Shared fixtures and builders for split test shards."""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone

import pytest

from gpu_fault.models import IncidentState, WorkflowOperation
from gpu_fault.processor import ProcessorRequest
from gpu_fault.regional import RemoteActionCommand
from gpu_fault.store import PostgresStore
from tests._builders import (
    fault_incident,
    processor_request,
    workflow_request,
    workflow_step,
)

POSTGRES_URL = os.getenv("GPU_FAULT_TEST_POSTGRES_URL")

REQUEST_LEASE = timedelta(seconds=120)


def _truncate() -> None:
    import psycopg

    with psycopg.connect(POSTGRES_URL, autocommit=True) as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT tablename FROM pg_tables
                WHERE schemaname='public'
                  AND tablename LIKE 'gpu\\_fault%'
                  AND tablename <> 'gpu_fault_schema_version'
                  AND tablename <> 'gpu_fault_schema_migrations'
                """
            )
            tables = [row[0] for row in cursor.fetchall()]
        if tables:
            with conn.cursor() as cursor:
                cursor.execute(f"TRUNCATE {', '.join(tables)}")


@pytest.fixture
def store():
    assert POSTGRES_URL is not None
    instance = PostgresStore(POSTGRES_URL)
    _truncate()
    try:
        yield instance
    finally:
        instance.close()


def _request(
    path: str, *, body: bytes = b"{}", cluster_id: str = "cluster-a"
) -> ProcessorRequest:
    return processor_request(path, body=body, cluster_id=cluster_id)


def _telemetry(node: str, *, cluster_id: str = "cluster-a"):
    return _request(
        "/v1/collector-events/host-telemetry",
        body=('{"node_id":"' + node + '","summary":true}').encode(),
        cluster_id=cluster_id,
    )


def _evidence(node: str, *, cluster_id: str = "cluster-a"):
    return _request(
        "/v1/collector-events/host-telemetry",
        body=json.dumps(
            {
                "node_id": node,
                "edge_filter_reasons": ["threshold:synthetic-priority-50"],
                "collection_errors": [],
            }
        ).encode(),
        cluster_id=cluster_id,
    )


def _workflow_state(store, *, request_id: str, fencing_token: int = 3):
    incident = fault_incident(
        f"incident-{request_id}",
        f"event-{request_id}",
        official_action="RESTART_BM",
        state=IncidentState.ACTION_PENDING,
        fencing_token=fencing_token,
    )
    workflow = workflow_request(
        request_id,
        incident.incident_id,
        fencing_token=fencing_token,
        runtime_profile_version="active-v1",
        official_action="RESTART_BM",
        official_steps=[workflow_step(WorkflowOperation.RESTART_NODE)],
    )
    incident = incident.model_copy(update={"workflow_request_id": workflow.request_id})
    store.save_incident(incident)
    store.save_workflow(workflow)
    return incident, workflow


def _command(store, command_id: str, *, request_id: str, token: int = 3):
    incident, workflow = _workflow_state(
        store, request_id=request_id, fencing_token=token
    )
    command = RemoteActionCommand(
        command_id=command_id,
        cluster_id=incident.cluster_id,
        workflow_request_id=workflow.request_id,
        incident_id=incident.incident_id,
        step_index=0,
        fencing_token=token,
        idempotency_key=f"{request_id}/0/RESTART_NODE",
        step=workflow.official_steps[0],
        workflow=workflow,
        incident=incident,
    )
    store.ensure_remote_command(command)
    return command


def _reload(store, command_id: str):
    """Read one command back.

    There is no public single-command getter - callers either claim or
    aggregate - so the tests reach for the same decode helper the store's
    own writers use.
    """
    return store._get_optional("remote_command", command_id)


def _counter_depth(cluster_id: str) -> int:
    import psycopg

    with psycopg.connect(POSTGRES_URL, autocommit=True) as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT coalesce(incomplete_count, 0)
                FROM gpu_fault_processor_queue_counts
                WHERE cluster_id=%s
                """,
                (cluster_id,),
            )
            row = cursor.fetchone()
    return int(row[0]) if row else 0


def _fault(node: str, *, cluster_id: str = "cluster-a"):
    return _request(
        "/v1/collector-events/nvidia-kernel",
        body=('{"node_id":"' + node + '","lines":[]}').encode(),
        cluster_id=cluster_id,
    )


COMPLETION_MARKER = "locked_lanes AS ("


def _recording_cursor(store, marker: str, statements: list[str]):
    """Wrap the pool's cursor and record statements containing ``marker``."""
    import contextlib

    original = store._db.cursor

    class _Recording:
        def __init__(self, cursor):
            self._cursor = cursor

        def execute(self, query, parameters=None):
            if marker in query:
                statements.append(query)
            return self._cursor.execute(query, parameters)

        def __getattr__(self, name):
            return getattr(self._cursor, name)

    @contextlib.contextmanager
    def recording(*args, **kwargs):
        with original(*args, **kwargs) as cursor:
            yield _Recording(cursor)

    store._db.cursor = recording
    return original


def _claim_completions(store, owner: str, *, limit: int):
    claimed = store.claim_active_processor_requests(
        owner, now=datetime.now(timezone.utc), lease_duration=REQUEST_LEASE, limit=limit
    )
    return claimed, [
        {
            "request_id": entry.request_id,
            "owner_id": owner,
            "lane_epoch": entry.leader_epoch,
            "lease_token": entry.lease_token,
            "response_status": 200,
            "response_content_type": "application/json",
            "response_body_base64": "e30=",
        }
        for entry in claimed
    ]

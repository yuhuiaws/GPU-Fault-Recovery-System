from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from threading import Barrier
from uuid import uuid4

import psycopg

from gpu_fault.models import (
    FaultIncident,
    IncidentState,
    RecoveryAction,
    WorkflowRequest,
    WorkflowStatus,
)
from gpu_fault.store import PostgresStore


def store_dsn() -> str:
    path = (
        os.environ.get("GPU_FAULT_STORE_URL_FILE")
        or "/etc/gpu-fault/aurora/postgres-url"
    )
    try:
        with open(path, encoding="utf-8") as handle:
            return handle.read().strip()
    except OSError:
        return os.environ["GPU_FAULT_STORE_URL"]


def main() -> None:
    dsn = store_dsn()
    suffix = uuid4().hex
    event_ids = [
        f"audit-p12-node-health-{suffix}-a",
        f"audit-p12-node-health-{suffix}-b",
    ]
    incident_ids = [
        f"inc-{event_ids[0]}",
        f"inc-{event_ids[1]}",
    ]
    workflow_ids = [
        f"workflow-{event_ids[0]}",
        f"workflow-{event_ids[1]}",
    ]
    serialization_key = '["node-scope","audit-p12-cluster","audit-p12-node"]'
    barrier = Barrier(2)
    stores = [PostgresStore(dsn), PostgresStore(dsn)]

    def create(index: int):
        store = stores[index]
        event_id = event_ids[index]
        peer_event_id = event_ids[1 - index]

        def build():
            peer = store.get_incident_by_event(peer_event_id)
            time.sleep(0.5)
            now = datetime.now(timezone.utc)
            incident = FaultIncident(
                incident_id=incident_ids[index],
                event_id=event_id,
                event_type="NODE_HEALTH",
                cluster_id="audit-p12-cluster",
                node_ids=["audit-p12-node"],
                policy_version="audit-p12",
                policy_source="audit-p12",
                effective_action=RecoveryAction.DRAIN,
                state=IncidentState.ESCALATED,
                workflow_request_id=workflow_ids[index],
                reasons=["audit P1-2 serialization"],
                created_at=now,
                updated_at=now,
            )
            workflow = WorkflowRequest(
                request_id=workflow_ids[index],
                incident_id=incident_ids[index],
                status=WorkflowStatus.BLOCKED,
                official_action=RecoveryAction.DRAIN.value,
                fencing_token=1,
                predecessor_workflow_id=(
                    peer.workflow_request_id if peer is not None else None
                ),
                blocked_reasons=["audit object; never execute"],
                created_at=now,
                updated_at=now,
            )
            return incident, workflow

        barrier.wait(timeout=10)
        return store.create_incident_workflow_if_absent(
            event_id,
            build,
            serialization_key=serialization_key,
        )

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(create, range(2)))

        workflows = [result[1] for result in results]
        roots = [
            workflow
            for workflow in workflows
            if workflow.predecessor_workflow_id is None
        ]
        successors = [
            workflow
            for workflow in workflows
            if workflow.predecessor_workflow_id is not None
        ]
        assert len(roots) == 1, workflows
        assert len(successors) == 1, workflows
        assert successors[0].predecessor_workflow_id == roots[0].request_id
        duplicate = stores[0].create_incident_workflow_if_absent(
            event_ids[0],
            lambda: (_ for _ in ()).throw(
                AssertionError("duplicate builder must not run")
            ),
            serialization_key=serialization_key,
        )
        assert duplicate[2] is False
        print(
            "PASS",
            {
                "root": roots[0].request_id,
                "successor": successors[0].request_id,
                "predecessor": (successors[0].predecessor_workflow_id),
                "duplicate_created": duplicate[2],
            },
        )
    finally:
        for store in stores:
            store.close()
        with psycopg.connect(dsn) as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    DELETE FROM gpu_fault_links
                    WHERE key = ANY(%s)
                       OR value = ANY(%s)
                    """,
                    (
                        event_ids + incident_ids + workflow_ids,
                        incident_ids + workflow_ids,
                    ),
                )
                cursor.execute(
                    """
                    DELETE FROM gpu_fault_objects
                    WHERE (kind = 'incident' AND key = ANY(%s))
                       OR (kind = 'workflow' AND key = ANY(%s))
                    """,
                    (incident_ids, workflow_ids),
                )


if __name__ == "__main__":
    main()

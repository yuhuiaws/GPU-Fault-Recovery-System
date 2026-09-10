from __future__ import annotations

import json
import os
import time
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import psycopg

from gpu_fault.processor import ProcessorRequest


def insert_request(cursor, request: ProcessorRequest) -> None:
    payload = request.model_dump_json()
    cursor.execute(
        """
        INSERT INTO gpu_fault_processor_queue (
            request_id, status, cluster_id, correlation_key,
            ordering_key, priority, lease_owner,
            leader_epoch, lease_token, lease_expires_at,
            created_at, updated_at, payload
        )
        VALUES (
            %s, %s, %s, %s, %s, %s, NULL,
            NULL, NULL, NULL, %s, %s, %s::jsonb
        )
        """,
        (
            request.request_id,
            request.status.value,
            request.cluster_id,
            request.correlation_key,
            request.ordering_key(),
            request.queue_priority(),
            request.created_at,
            request.updated_at,
            payload,
        ),
    )
    cursor.execute(
        """
        INSERT INTO gpu_fault_objects(kind, key, payload)
        VALUES ('processor_request', %s, %s::jsonb)
        """,
        (request.request_id, payload),
    )


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
    url = store_dsn()
    runtime_profile_version = os.environ["GPU_FAULT_RUNTIME_PROFILE_VERSION"]
    suffix = uuid4().hex[:12]
    cluster_id = f"audit-order-{suffix}"
    node_id = f"audit-node-{suffix}"
    job_id = f"audit-job-{suffix}"
    attempt_id = f"{job_id}-a1"
    observed_at = datetime.now(timezone.utc)
    fault = ProcessorRequest.from_http(
        method="POST",
        path="/v1/collector-events/nvidia-kernel",
        query="",
        body=json.dumps(
            {
                "cluster_id": cluster_id,
                "node_id": node_id,
                "record_id": f"audit-kmsg-{suffix}",
                "observed_at": observed_at.isoformat(),
                "message": "NVRM: Xid (PCI:0000:59:00.0): 14",
                "product": "H200",
                "driver_branch": 580,
                "cuda_version": "13.0",
                "runtime_profile_version": runtime_profile_version,
            },
            separators=(",", ":"),
        ).encode(),
        content_type="application/json",
        cluster_id=cluster_id,
    )
    observation = ProcessorRequest.from_http(
        method="POST",
        path="/v1/workload-observations",
        query="",
        body=json.dumps(
            {
                "cluster_id": cluster_id,
                "environment": "hyperpod-eks",
                "job_id": job_id,
                "attempt_id": attempt_id,
                "workload_phase": "RUNNING",
                "observed_at": (observed_at - timedelta(seconds=1)).isoformat(),
                "started_at": (observed_at - timedelta(minutes=1)).isoformat(),
                "expected_critical_ranks": 1,
                "containers": [
                    {
                        "pod_uid": f"audit-pod-{suffix}",
                        "pod_name": f"audit-pod-{suffix}",
                        "container_name": "trainer",
                        "role": "worker",
                        "rank": 0,
                        "node_id": node_id,
                        "gpu_count": 0,
                    }
                ],
                "workload_ids": [f"kubernetes/job/audit-{suffix}"],
                "runtime_profile_version": runtime_profile_version,
                "restart_budget": 0,
            },
            separators=(",", ":"),
        ).encode(),
        content_type="application/json",
        cluster_id=cluster_id,
    )
    connection = psycopg.connect(url)
    connection.autocommit = False
    try:
        with connection.transaction():
            with connection.cursor() as cursor:
                # Insert the fault first. Both become visible at the same
                # commit, so only the claim dependency can force the
                # observation to complete first.
                insert_request(cursor, fault)
                insert_request(cursor, observation)
        deadline = time.monotonic() + 60
        completed = {}
        while time.monotonic() < deadline:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT request_id, status, updated_at, payload
                    FROM gpu_fault_processor_queue
                    WHERE request_id IN (%s, %s)
                    """,
                    (fault.request_id, observation.request_id),
                )
                completed = {
                    row[0]: {
                        "status": row[1],
                        "updated_at": row[2],
                        "payload": row[3],
                    }
                    for row in cursor.fetchall()
                }
            if len(completed) == 2 and all(
                item["status"] == "COMPLETED" for item in completed.values()
            ):
                break
            time.sleep(0.05)
        assert completed[fault.request_id]["status"] == "COMPLETED"
        assert completed[observation.request_id]["status"] == "COMPLETED"
        observation_completed_at = completed[observation.request_id]["updated_at"]
        fault_completed_at = completed[fault.request_id]["updated_at"]
        assert observation_completed_at <= fault_completed_at
        event_id = f"kernel-log-audit-kmsg-{suffix}-xid-14"
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT payload
                FROM gpu_fault_objects
                WHERE kind='xid_correlation_event' AND key=%s
                """,
                (event_id,),
            )
            event = cursor.fetchone()[0]
        assert event["job_id"] == job_id
        assert event["attempt_id"] == attempt_id
        assert event["workload_identity_source"] != ("NO_ACTIVE_MANAGED_ATTEMPT")
        print(
            json.dumps(
                {
                    "fault_inserted_first": True,
                    "observation_completed_at": (observation_completed_at.isoformat()),
                    "fault_completed_at": (fault_completed_at.isoformat()),
                    "workload_identity_source": event["workload_identity_source"],
                    "job_id": event["job_id"],
                    "attempt_id": event["attempt_id"],
                },
                indent=2,
                sort_keys=True,
            )
        )
    finally:
        connection.rollback()
        connection.autocommit = True
        with connection.cursor() as cursor:
            cursor.execute(
                """
                DELETE FROM gpu_fault_processor_queue
                WHERE request_id IN (%s, %s)
                """,
                (fault.request_id, observation.request_id),
            )
            cursor.execute(
                """
                DELETE FROM gpu_fault_objects
                WHERE key IN (%s, %s)
                   OR payload->>'cluster_id'=%s
                """,
                (
                    fault.request_id,
                    observation.request_id,
                    cluster_id,
                ),
            )
            cursor.execute(
                """
                DELETE FROM gpu_fault_links
                WHERE key LIKE %s OR value LIKE %s
                """,
                (f"%{suffix}%", f"%{suffix}%"),
            )
        connection.close()


if __name__ == "__main__":
    main()

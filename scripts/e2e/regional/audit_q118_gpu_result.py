from __future__ import annotations

import json
import os
import sys

from gpu_fault.store import PostgresStore
from gpu_fault.telemetry import EvidenceKind


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
    cluster_id, attempt_id = sys.argv[1:3]
    store = PostgresStore(
        store_dsn(),
        pool_min_size=1,
        pool_max_size=2,
        pool_timeout_seconds=5,
    )
    try:
        event = store.get_event_by_attempt(cluster_id, attempt_id)
        decision = store.get_decision_by_event(event.event_key)
        plan = store.get_plan(decision.recovery_plan_id)
        workflow = store.get_workflow(plan.workflow_request_id)
        records = store.list_raw_evidence(
            cluster_id,
            attempt_id=attempt_id,
            kind=EvidenceKind.WORKLOAD_LOG,
            limit=100,
        )
        print(
            json.dumps(
                {
                    "terminal_status": event.terminal_status.value,
                    "gpu_count": event.gpu_count,
                    "initiator": (event.termination_initiator_incident_id),
                    "decision": decision.status.value,
                    "recovery_workflow": workflow.request_id,
                    "recovery_status": workflow.status.value,
                    "restart_steps": [
                        item.operation.value for item in workflow.official_steps
                    ],
                    "restart_executions": [
                        item.status.value for item in workflow.step_executions
                    ],
                    "evidence": [
                        {
                            "record_id": record.record_id,
                            "node_id": record.node_id,
                            "expires_at": (record.expires_at.isoformat()),
                            "tail_bytes": record.payload.get("tail_bytes"),
                            "truncated": record.payload.get("truncated"),
                            "s3_uri": record.payload.get("s3_uri"),
                            "has_heartbeat": (
                                "Q118_GPU_TRAIN"
                                in str(record.payload.get("tail") or "")
                            ),
                        }
                        for record in records
                    ],
                },
                indent=2,
            )
        )
    finally:
        store.close()


if __name__ == "__main__":
    main()

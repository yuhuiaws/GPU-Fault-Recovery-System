"""Run-bound capacity data cleanup, also executable in the CPU component."""

from __future__ import annotations

import json
import os
import re
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from gpu_fault.regional import (
    RegionalRegistryRevision,
    regional_registry_content_sha256,
)
from gpu_fault.store.postgres.pool import StoreCredentials

TABLE_SCOPES = {
    "gpu_fault_processor_queue": "cluster_id",
    "gpu_fault_processor_lanes": "split_part(ordering_key, ':', 1)",
    "gpu_fault_processor_queue_counts": "cluster_id",
    "gpu_fault_gpu_metric_latest": "cluster_id",
    "gpu_fault_gpu_metrics_batches": "cluster_id",
    "gpu_fault_attempt_observations": "cluster_id",
    "gpu_fault_training_progress": "cluster_id",
}
REFERENCE_FIELDS = (
    "event_id",
    "incident_id",
    "notification_id",
    "workflow_request_id",
    "predecessor_workflow_id",
    "source_workflow_id",
)
MAX_RECORDS = 100_000


def validate_scope(run_id: str, cluster_ids: list[str]) -> None:
    if (
        not isinstance(run_id, str)
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,62}", run_id) is None
    ):
        raise RuntimeError("capacity data requires an explicit run identity")
    if (
        not cluster_ids
        or len(cluster_ids) != len(set(cluster_ids))
        or any(
            re.fullmatch(r"perf-cap-[0-9]{3}", value) is None for value in cluster_ids
        )
    ):
        raise RuntimeError("capacity data requires exact synthetic cluster identities")


def scope_records(
    cursor: Any, cluster_ids: list[str]
) -> dict[tuple[str, str], dict[str, Any]]:
    from psycopg import sql

    cursor.execute(
        "SELECT kind,key,payload FROM gpu_fault_control_records "
        "WHERE kind <> 'regional_cluster' AND "
        "(payload->>'cluster_id'=ANY(%s) OR payload->>'cluster_name'=ANY(%s))",
        (cluster_ids, cluster_ids),
    )
    records = {(kind, key): payload for kind, key, payload in cursor.fetchall()}
    while records:
        if len(records) > MAX_RECORDS:
            raise RuntimeError("capacity cleanup record bound exceeded")
        identifiers = record_identifiers(records)
        predicates = sql.SQL(" OR ").join(
            sql.SQL("payload->>{}=ANY(%s)").format(sql.Literal(field))
            for field in REFERENCE_FIELDS
        )
        successors = [
            f"workflow-reboot-after-{key}"
            for kind, key in records
            if kind == "workflow"
        ]
        cursor.execute(
            sql.SQL(
                "SELECT kind,key,payload FROM gpu_fault_control_records WHERE "
                "kind <> 'regional_cluster' AND ({predicates} OR "
                "(kind='workflow' AND key=ANY(%s)))"
            ).format(predicates=predicates),
            (*([identifiers] * len(REFERENCE_FIELDS)), successors),
        )
        additions = {(kind, key): payload for kind, key, payload in cursor.fetchall()}
        for payload in additions.values():
            for field in ("cluster_id", "cluster_name"):
                value = payload.get(field)
                if value and value not in cluster_ids:
                    raise RuntimeError(
                        "capacity cleanup dependency crosses cluster ownership"
                    )
            incident_id = payload.get("incident_id")
            if incident_id:
                cursor.execute(
                    "SELECT payload->>'cluster_id' FROM gpu_fault_control_records "
                    "WHERE kind='incident' AND key=%s",
                    (incident_id,),
                )
                owner = cursor.fetchone()
                if owner is None or owner[0] not in cluster_ids:
                    raise RuntimeError(
                        "capacity cleanup incident ownership is unknown or foreign"
                    )
        new = additions.keys() - records.keys()
        if not new:
            break
        records.update({key: additions[key] for key in new})
    return records


def record_identifiers(records: dict[tuple[str, str], dict[str, Any]]) -> list[str]:
    return sorted(
        {
            *(key for _, key in records),
            *(
                value
                for payload in records.values()
                for field in ("event_id",)
                if isinstance(value := payload.get(field), str) and value
            ),
        }
    )


def table_counts(cursor: Any, cluster_ids: list[str]) -> dict[str, int]:
    from psycopg import sql

    counts = {}
    for table, expression in TABLE_SCOPES.items():
        cursor.execute(
            sql.SQL("SELECT count(*) FROM {} WHERE {}=ANY(%s)").format(
                sql.Identifier(table), sql.SQL(expression)
            ),
            (cluster_ids,),
        )
        counts[table] = int(cursor.fetchone()[0])
    return counts


def require_registry_scope(cursor: Any, run_id: str, cluster_ids: list[str]) -> bool:
    # Hold the publication head until cleanup commits, preventing cluster-ID reuse.
    cursor.execute(
        "SELECT payload FROM gpu_fault_objects "
        "WHERE kind='regional_registry_head' AND key='current' FOR SHARE"
    )
    row = cursor.fetchone()
    if row is None:
        raise RuntimeError("capacity cleanup has no durable registry head")
    head = row[0]
    cursor.execute(
        "SELECT payload FROM gpu_fault_objects "
        "WHERE kind='regional_registry_revision' AND key=%s",
        (str(head["generation"]),),
    )
    row = cursor.fetchone()
    if row is None:
        raise RuntimeError("capacity cleanup registry revision is missing")
    revision = RegionalRegistryRevision.model_validate(row[0])
    if (
        revision.content_sha256 != head["content_sha256"]
        or regional_registry_content_sha256(revision.registrations)
        != revision.content_sha256
    ):
        raise RuntimeError("capacity cleanup registry revision changed")
    current = {item.cluster_id: item for item in revision.registrations}
    present = [current[key] for key in cluster_ids if key in current]
    if len(present) != len(cluster_ids) or any(
        not item.synthetic or item.synthetic_run_id != run_id for item in present
    ):
        raise RuntimeError(
            "capacity cleanup lacks the complete current run-owned registry"
        )
    return bool(present)


def inspect_or_cleanup(
    connection: Any, *, run_id: str, cluster_ids: list[str], cleanup: bool
) -> dict[str, Any]:
    from psycopg import sql

    validate_scope(run_id, cluster_ids)
    with connection.transaction():
        with connection.cursor() as cursor:
            cursor.execute("SET LOCAL statement_timeout = '120s'")
            registered = False
            if cleanup:
                registered = require_registry_scope(cursor, run_id, cluster_ids)
            records = scope_records(cursor, cluster_ids)
            counts = table_counts(cursor, cluster_ids)
            identifiers = record_identifiers(records)
            cursor.execute(
                "SELECT count(*) FROM gpu_fault_links WHERE key=ANY(%s) OR value=ANY(%s)",
                (identifiers, identifiers),
            )
            links = int(cursor.fetchone()[0])
            if not cleanup:
                return {
                    "records": len(records),
                    "links": links,
                    **counts,
                    "total": len(records) + links + sum(counts.values()),
                }
            if not registered and (records or links or sum(counts.values())):
                raise RuntimeError(
                    "capacity data has no current run-owned registration"
                )
            # Drained means no open work. COMPLETED rows are replay records the
            # processor keeps for its idempotency window and this cleanup deletes
            # below; counting them refused every teardown that ran right after
            # the probe stopped (HA-005 attempt 2: 26 COMPLETED rows, 3 refusals).
            cursor.execute(
                "SELECT count(*) FROM gpu_fault_processor_queue "
                "WHERE cluster_id=ANY(%s) AND status <> 'COMPLETED'",
                (cluster_ids,),
            )
            if int(cursor.fetchone()[0]):
                raise RuntimeError("capacity processor queue has not drained")
            for (kind, _), payload in records.items():
                terminal = (
                    {"SUCCEEDED", "FAILED", "CANCELLED"}
                    if kind == "remote_command"
                    else {"SUCCEEDED", "FAILED", "BLOCKED", "SUPERSEDED"}
                )
                if (
                    kind in {"workflow", "remote_command"}
                    and payload.get("status") not in terminal
                ):
                    raise RuntimeError(
                        "capacity workflow or command is still nonterminal"
                    )
            cursor.execute(
                "DELETE FROM gpu_fault_links WHERE key=ANY(%s) OR value=ANY(%s)",
                (identifiers, identifiers),
            )
            for (kind, key), payload in records.items():
                cursor.execute(
                    "SELECT gpu_fault_delete_control_state(%s,%s,%s::jsonb)",
                    (kind, key, json.dumps(payload)),
                )
                if not cursor.fetchone()[0]:
                    cursor.execute(
                        "SELECT 1 FROM gpu_fault_control_records WHERE kind=%s AND key=%s",
                        (kind, key),
                    )
                    if cursor.fetchone() is not None:
                        raise RuntimeError("capacity record changed during cleanup")
            for table, expression in TABLE_SCOPES.items():
                cursor.execute(
                    sql.SQL("DELETE FROM {} WHERE {}=ANY(%s)").format(
                        sql.Identifier(table), sql.SQL(expression)
                    ),
                    (cluster_ids,),
                )
            remaining = len(scope_records(cursor, cluster_ids)) + sum(
                table_counts(cursor, cluster_ids).values()
            )
            if remaining:
                raise RuntimeError("capacity cleanup left run-owned data")
    return {
        "run_id": run_id,
        "cluster_ids": cluster_ids,
        "total": 0,
        "deleted_records": len(records),
        "deleted_links": links,
    }


def invoke(
    control: Callable[..., str], *, run_id: str, cluster_ids: list[str], cleanup: bool
) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from scripts.e2e.regional.regional_live_fixture import component_python

    validate_scope(run_id, cluster_ids)
    pod = control(
        "get",
        "pod",
        "-l",
        "app=gpu-fault-api-ha",
        "--field-selector=status.phase=Running",
        "-o",
        "jsonpath={.items[0].metadata.name}",
    ).strip()
    if not pod:
        raise RuntimeError("capacity data probe requires a CPU API Pod")
    output = control(
        "exec",
        "-i",
        pod,
        "--",
        component_python("cpu"),
        "-",
        json.dumps({"run_id": run_id, "cluster_ids": cluster_ids, "cleanup": cleanup}),
        stdin=Path(__file__).read_bytes(),
        timeout=150,
    )
    result = json.loads(output.splitlines()[-1])
    if not isinstance(result, dict) or type(result.get("total")) is not int:
        raise RuntimeError("capacity data probe result is incomplete")
    return result


def run_request(arguments: dict[str, Any]) -> dict[str, Any]:
    import psycopg

    validate_scope(arguments["run_id"], arguments["cluster_ids"])
    if type(arguments.get("cleanup")) is not bool:
        raise RuntimeError("capacity cleanup mode must be explicit")
    path = os.getenv("GPU_FAULT_STORE_URL_FILE")
    credentials = StoreCredentials(os.getenv("GPU_FAULT_STORE_URL", ""), path=path)
    url = credentials.conninfo()
    if not url or (path and credentials.source != "file"):
        raise RuntimeError("current capacity database credentials are unavailable")
    with psycopg.connect(url, connect_timeout=10) as connection:
        return inspect_or_cleanup(connection, **arguments)


if __name__ == "__main__":
    print(json.dumps(run_request(json.loads(sys.argv[1])), sort_keys=True))

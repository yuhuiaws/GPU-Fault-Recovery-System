"""Read-only cleanup activity evidence, executed as source inside a CPU pod."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any


def connection_arguments() -> dict[str, Any]:
    from psycopg.conninfo import conninfo_to_dict

    path = os.environ.get("GPU_FAULT_STORE_URL_FILE")
    dsn = (
        Path(path).read_text(encoding="utf-8").strip()
        if path
        else os.environ.get("GPU_FAULT_STORE_URL", "").strip()
    )
    if not dsn:
        raise ValueError("database credentials unavailable")
    arguments = conninfo_to_dict(dsn)
    if arguments.get("sslmode") != "verify-full" or not arguments.get("sslrootcert"):
        raise ValueError("cleanup requires verified database TLS")
    return {
        **arguments,
        "connect_timeout": 10,
        "options": (
            "-c default_transaction_read_only=on -c statement_timeout=15000 "
            "-c lock_timeout=5000 -c idle_in_transaction_session_timeout=30000"
        ),
    }


def records_relation(cursor: Any) -> str:
    cursor.execute(
        "SELECT to_regclass('gpu_fault_control_records'), "
        "to_regclass('gpu_fault_control_state_modes'), "
        "to_regclass('gpu_fault_remote_commands'), to_regclass('gpu_fault_workflows')"
    )
    records, modes, commands, workflows = cursor.fetchone()
    if modes is not None:
        cursor.execute("SELECT kind,mode FROM gpu_fault_control_state_modes")
        values = dict(cursor.fetchall())
        if (
            set(values) != {"workflow", "remote_command"}
            or any(
                value not in {"legacy", "dual", "dedicated"}
                for value in values.values()
            )
            or None in (records, commands, workflows)
        ):
            raise ValueError("control-state storage is not known")
    elif commands is not None or workflows is not None:
        raise ValueError("control-state storage metadata is missing")
    return "gpu_fault_control_records" if records is not None else "gpu_fault_objects"


def counts(
    connection: Any, *, scope: str, cluster_ids: list[str]
) -> tuple[int, int, int, int]:
    from psycopg import sql

    if scope not in {"all", "gpu"} or scope == "gpu" and not cluster_ids:
        raise ValueError("invalid cleanup query scope")
    with connection.transaction(), connection.cursor() as cursor:
        cursor.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
        cursor.execute("SET LOCAL row_security = off")
        records = sql.Identifier(records_relation(cursor))
        # Unknown status and orphan workflow identity are blockers, including
        # during a scoped cleanup. An inner join silently discards those rows.
        workflow_scope = (
            " AND (NULLIF(incident.payload->>'cluster_id','') IS NULL "
            "OR incident.payload->>'cluster_id' = ANY(%s))"
            if scope == "gpu"
            else ""
        )
        command_scope = (
            " AND (NULLIF(payload->>'cluster_id','') IS NULL "
            "OR payload->>'cluster_id' = ANY(%s))"
            if scope == "gpu"
            else ""
        )
        queue_scope = (
            " AND (NULLIF(cluster_id,'') IS NULL OR cluster_id = ANY(%s))"
            if scope == "gpu"
            else ""
        )
        params = (cluster_ids,) if scope == "gpu" else ()
        queries = [
            (
                sql.SQL(
                    """
                    SELECT count(*) FROM {} AS workflow
                    LEFT JOIN gpu_fault_objects AS incident
                      ON incident.kind='incident'
                     AND incident.key=workflow.payload->>'incident_id'
                    WHERE workflow.kind='workflow'
                      AND coalesce(workflow.payload->>'status','')
                        NOT IN ('SUCCEEDED','FAILED','SUPERSEDED')
                    """
                    + workflow_scope
                ).format(records),
                params,
            ),
            (
                sql.SQL(
                    """
                    SELECT count(*) FROM {}
                    WHERE kind='remote_command'
                      AND coalesce(payload->>'status','') NOT IN ('SUCCEEDED','FAILED')
                    """
                    + command_scope
                ).format(records),
                params,
            ),
            (
                "SELECT count(*) FROM gpu_fault_processor_queue "
                "WHERE coalesce(status,'') <> 'COMPLETED'" + queue_scope,
                params,
            ),
        ]
        result: list[int] = []
        for statement, parameters in queries:
            cursor.execute(statement, parameters)
            row = cursor.fetchone()
            if row is None or type(row[0]) is not int or row[0] < 0:
                raise ValueError("invalid cleanup count")
            result.append(row[0])
        cursor.execute("SELECT to_regclass('gpu_fault_telemetry_spool')")
        if cursor.fetchone()[0] is None:
            if os.environ.get("GPU_FAULT_TELEMETRY_SPOOL_ENABLED", "false") != "false":
                raise ValueError("enabled telemetry spool table is missing")
            result.append(0)
        else:
            cursor.execute(
                "SELECT count(*) FROM gpu_fault_telemetry_spool WHERE true"
                + queue_scope,
                params,
            )
            row = cursor.fetchone()
            if row is None or type(row[0]) is not int or row[0] < 0:
                raise ValueError("invalid telemetry spool count")
            result.append(row[0])
        return result[0], result[1], result[2], result[3]


def fleet_inventory(connection: Any, cluster_ids: list[str]) -> list[dict[str, Any]]:
    from gpu_fault.installation_inventory import InstalledUnitInventory

    with connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT payload FROM gpu_fault_objects
            WHERE kind='agent' AND payload->>'cluster_id'=ANY(%s)
            ORDER BY key LIMIT 100001
            """,
            (cluster_ids,),
        )
        rows = cursor.fetchall()
    if len(rows) > 100000:
        raise ValueError("fleet inventory exceeds cleanup limit")
    result = []
    seen: set[tuple[str, str]] = set()
    for (agent,) in rows:
        identity = agent.get("cluster_id"), agent.get("node_id")
        if (
            identity[0] not in cluster_ids
            or not isinstance(identity[1], str)
            or not identity[1]
            or identity in seen
        ):
            raise ValueError("ambiguous fleet identity")
        seen.add(identity)
        inventory = agent.get("installed_unit_inventory")
        if inventory is not None:
            inventory = InstalledUnitInventory.model_validate(inventory).model_dump(
                mode="json"
            )
        result.append(
            {
                "cluster_id": identity[0],
                "node_id": identity[1],
                "node_instance_id": agent.get("node_instance_id"),
                "lifecycle_state": agent.get("lifecycle_state"),
                "installed_unit_inventory": inventory,
            }
        )
    return result


def main() -> int:
    try:
        import psycopg

        action, scope, *cluster_ids = sys.argv[1:]
        with psycopg.connect(**connection_arguments()) as connection:
            if action == "fleet":
                print(
                    json.dumps(fleet_inventory(connection, cluster_ids), sort_keys=True)
                )
            elif action == "counts":
                print(
                    *counts(connection, scope=scope, cluster_ids=cluster_ids), sep="\t"
                )
            else:
                raise ValueError("invalid cleanup probe")
    except Exception as exc:
        # Database exceptions can contain a DSN. Emit only the failure class.
        print(f"cleanup database probe failed: {type(exc).__name__}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

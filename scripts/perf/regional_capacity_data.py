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
# A 50-cluster run leaves ~2.5k hot-state rows per synthetic cluster (latest
# telemetry, collector status and inventory snapshots per node); the walk's
# bound scales with the run instead of refusing legitimate 50-cluster data.
RECORDS_PER_CLUSTER = 4000
# Control-state kinds are deleted one by one through the store's CAS function;
# every other kind is per-node hot state that lives in gpu_fault_objects and is
# deleted set-based, or a 50-cluster teardown outlives its exec budget.
CAS_KINDS = frozenset({"workflow", "remote_command", "incident"})
BULK_DELETE_BATCH = 5000


def record_bound(cluster_ids: list[str]) -> int:
    return max(MAX_RECORDS, RECORDS_PER_CLUSTER * len(cluster_ids))


def bulk_delete_batches(
    records: dict[tuple[str, str], dict[str, Any]],
) -> list[tuple[str, list[str]]]:
    """``(kind, keys)`` batches for the set-based deletes, CAS kinds excluded."""

    by_kind: dict[str, list[str]] = {}
    for kind, key in sorted(records):
        if kind not in CAS_KINDS:
            by_kind.setdefault(kind, []).append(key)
    return [
        (kind, keys[start : start + BULK_DELETE_BATCH])
        for kind, keys in by_kind.items()
        for start in range(0, len(keys), BULK_DELETE_BATCH)
    ]


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
        if len(records) > record_bound(cluster_ids):
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
                if owner is not None and owner[0] not in cluster_ids:
                    raise RuntimeError(
                        "capacity cleanup incident ownership is unknown or foreign"
                    )
                if owner is None and not any(
                    payload.get(field) in cluster_ids
                    for field in ("cluster_id", "cluster_name")
                ):
                    # Notifications reference derived incident ids (live 2026-09-20:
                    # `collector-silent-<node>` for perf-cap nodes) that never
                    # become incident records. The record's own cluster field is
                    # then the ownership proof; without it the record stays.
                    raise RuntimeError(
                        "capacity cleanup incident ownership is unknown or foreign"
                    )
        new = additions.keys() - records.keys()
        if not new:
            break
        records.update({key: additions[key] for key in new})
    return records


def drill_residue_only(
    records: dict[tuple[str, str], dict[str, Any]], run_id: str
) -> bool:
    """True when every scoped record is a drill notification of ``run_id``.

    The collector-silence detector keeps emitting ``[DRILL:perf-capacity]``
    notifications for synthetic nodes until their clusters are deregistered;
    a few land after the run's data purge (live 2026-09-20: four for
    perf-cap-000). They carry the drill marker and the run id in their derived
    incident id, which is ownership proof enough to sweep them without a live
    registration; anything else keeps refusing.
    """

    if not records:
        return False
    drills = {
        key
        for (kind, key), payload in records.items()
        if kind == "notification"
        and payload.get("drill_id") == "perf-capacity"
        and run_id in str(payload.get("incident_id") or "")
    }
    for (kind, key), payload in records.items():
        if kind == "notification":
            if key not in drills:
                return False
        elif kind in {"notification_delivery", "notification_result"}:
            # The delivery queue row and the provider result hang off the
            # notification by its id; they go with it.
            if str(payload.get("notification_id") or key) not in drills:
                return False
        else:
            return False
    return True


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
    if not present:
        # Deregistration already happened; the caller may sweep only the late
        # drill residue it can prove belongs to this run.
        return False
    if len(present) != len(cluster_ids) or any(
        not item.synthetic or item.synthetic_run_id != run_id for item in present
    ):
        raise RuntimeError(
            "capacity cleanup lacks the complete current run-owned registry"
        )
    return True


TERMINAL_STATUSES = {
    "remote_command": frozenset({"SUCCEEDED", "FAILED", "CANCELLED"}),
    "workflow": frozenset({"SUCCEEDED", "FAILED", "BLOCKED", "SUPERSEDED"}),
}


def nonterminal_records(records: dict[tuple[str, str], dict]) -> list[tuple[str, str]]:
    """Run-owned workflows and commands that are still open."""

    return sorted(
        (kind, key)
        for (kind, key), payload in records.items()
        if kind in TERMINAL_STATUSES
        and payload.get("status") not in TERMINAL_STATUSES[kind]
    )


def inspect_or_cleanup(
    connection: Any,
    *,
    run_id: str,
    cluster_ids: list[str],
    cleanup: bool,
    force_nonterminal: bool = False,
) -> dict[str, Any]:
    """Probe (``cleanup=False``) or delete the run's synthetic data.

    Open workflows or commands refuse the cleanup unless ``force_nonterminal``:
    a teardown runs after the run's executor Jobs are gone, so a synthetic
    command that is still LEASED or PENDING can never finish and would refuse
    every later run (live 2026-09-22: an aborted action run left 13 LEASED and
    8 PENDING RESTART_NODE commands; the next registration was refused with
    "synthetic registry entries already exist").
    """

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
            if (
                not registered
                and (records or links or sum(counts.values()))
                and (sum(counts.values()) or not drill_residue_only(records, run_id))
            ):
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
            stuck = nonterminal_records(records)
            if stuck and not force_nonterminal:
                raise RuntimeError("capacity workflow or command is still nonterminal")
            cursor.execute(
                "DELETE FROM gpu_fault_links WHERE key=ANY(%s) OR value=ANY(%s)",
                (identifiers, identifiers),
            )
            forced = set(stuck) if force_nonterminal else set()
            for (kind, key), payload in records.items():
                if kind not in CAS_KINDS:
                    continue
                if (kind, key) in forced:
                    # An open synthetic record is still being touched by the
                    # processor (lease expiry, dispatch); the CAS on the probe's
                    # snapshot would miss. Delete it unconditionally: the store
                    # function locks the row and the run owns the cluster.
                    cursor.execute(
                        "SELECT gpu_fault_delete_control_state(%s,%s)", (kind, key)
                    )
                    cursor.fetchone()
                    continue
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
            for kind, keys in bulk_delete_batches(records):
                # The store's delete function targets gpu_fault_objects for every
                # kind outside the control-state tables; this is the same delete
                # without the per-row CAS, which hot-state snapshots do not need.
                cursor.execute(
                    "DELETE FROM gpu_fault_objects WHERE kind=%s AND key=ANY(%s)",
                    (kind, keys),
                )
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
        "forced_nonterminal": len(stuck),
    }


def invoke(
    control: Callable[..., str],
    *,
    run_id: str,
    cluster_ids: list[str],
    cleanup: bool,
    force_nonterminal: bool = False,
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
        json.dumps(
            {
                "run_id": run_id,
                "cluster_ids": cluster_ids,
                "cleanup": cleanup,
                "force_nonterminal": bool(cleanup and force_nonterminal),
            }
        ),
        stdin=Path(__file__).read_bytes(),
        timeout=900 if cleanup else 150,
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
    if type(arguments.get("force_nonterminal", False)) is not bool:
        raise RuntimeError("capacity forced cleanup flag must be a boolean")
    path = os.getenv("GPU_FAULT_STORE_URL_FILE")
    credentials = StoreCredentials(os.getenv("GPU_FAULT_STORE_URL", ""), path=path)
    url = credentials.conninfo()
    if not url or (path and credentials.source != "file"):
        raise RuntimeError("current capacity database credentials are unavailable")
    with psycopg.connect(url, connect_timeout=10) as connection:
        return inspect_or_cleanup(connection, **arguments)


if __name__ == "__main__":
    print(json.dumps(run_request(json.loads(sys.argv[1])), sort_keys=True))

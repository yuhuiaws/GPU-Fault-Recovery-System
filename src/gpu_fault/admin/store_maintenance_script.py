"""Reclaim hot-table bloat through the product's own store connection.

Self-contained on purpose (standard library plus psycopg): ``gpu-fault-admin
store-maintenance`` execs this module's source into a Running CPU ingress Pod
with ``python -c`` and the request on stdin, the way every administrator verb
reaches the Store, so the verb depends on neither host-to-database network
nor the Pod's package version. Tests import it and call ``run`` against the
test database (``tests/admin/test_store_maintenance_script_postgres.py``).

Storage parameters (schema v19) keep the hot tables from bloating again; they
do not return heap or index pages a day of churn has already consumed. Two
actions do, chosen per table by what its readers tolerate:

* ``VACUUM (FULL, ANALYZE)`` for the small tables rewritten in place by every
  queue statement (lanes, queue counts, priority count shards): the rewrite
  takes an AccessExclusiveLock for well under a second at their live size,
  and nothing else returns their heap pages.
* ``REINDEX INDEX CONCURRENTLY`` for every index of the large tables (latest
  metrics, objects, the queue and its batches), then ``VACUUM (ANALYZE)``:
  no long exclusive lock on a table the admission and claim paths read
  continuously. ``REINDEX ... CONCURRENTLY`` needs PostgreSQL 12 or newer
  (``server_version_num`` is checked; Aurora PostgreSQL reports it).

The script never changes Aurora capacity, never touches a table outside the
two lists, runs under the ``gpu_fault_schema_bootstrap`` advisory lock so it
cannot overlap a schema ensure or an index build, refuses to execute a plan
whose identity differs from the one the operator reviewed, and reports the
before/after size of every table so the verb can record it as evidence.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

#: Small tables rewritten in place: ``VACUUM (FULL, ANALYZE)``.
COMPACT_TABLES = (
    "gpu_fault_processor_lanes",
    "gpu_fault_processor_queue_counts",
    "gpu_fault_processor_priority_count_shards",
)
#: Large or continuously read tables: ``REINDEX INDEX CONCURRENTLY`` per
#: index, then ``VACUUM (ANALYZE)``.
REINDEX_TABLES = (
    "gpu_fault_gpu_metric_latest",
    "gpu_fault_gpu_metric_latest_clusters",
    "gpu_fault_objects",
    "gpu_fault_processor_queue",
    "gpu_fault_processor_payloads",
    "gpu_fault_gpu_metrics_batches",
)
COMPACT_ACTION = "VACUUM (FULL, ANALYZE)"
REINDEX_ACTION = "REINDEX INDEX CONCURRENTLY + VACUUM (ANALYZE)"
ADVISORY_LOCK_NAME = "gpu_fault_schema_bootstrap"
REINDEX_CONCURRENTLY_MINIMUM_VERSION = 120000
LOCK_TIMEOUT = "10s"
STATEMENT_TIMEOUT = "840s"
DEFAULT_STORE_URL_FILE = "/etc/gpu-fault/aurora/postgres-url"


class MaintenanceRefused(RuntimeError):
    """A refusal the operator may read verbatim (no record contents)."""


def store_dsn() -> str:
    """The Pod's store DSN: the mounted Secret file first, then the env."""

    path = os.environ.get("GPU_FAULT_STORE_URL_FILE", "").strip()
    if not path and Path(DEFAULT_STORE_URL_FILE).exists():
        path = DEFAULT_STORE_URL_FILE
    if path:
        value = Path(path).read_text(encoding="utf-8").strip()
        if value:
            return value
    value = os.environ.get("GPU_FAULT_STORE_URL", "").strip()
    if not value:
        raise MaintenanceRefused("store credentials are unavailable in this Pod")
    return value


def connect(dsn: str) -> Any:
    """An autocommit session with bounded lock and statement waits."""

    import psycopg

    connection = psycopg.connect(dsn, autocommit=True, connect_timeout=10)
    connection.execute(f"SET lock_timeout = '{LOCK_TIMEOUT}'")
    connection.execute(f"SET statement_timeout = '{STATEMENT_TIMEOUT}'")
    return connection


def reindex_concurrently_supported(version_num: int) -> bool:
    return version_num >= REINDEX_CONCURRENTLY_MINIMUM_VERSION


def server_facts(cursor: Any) -> dict[str, Any]:
    cursor.execute(
        "SELECT version(), current_setting('server_version_num')::int, "
        "current_database()"
    )
    version, version_num, database = cursor.fetchone()
    return {
        "version": str(version),
        "version_num": int(version_num),
        "database": str(database),
        "reindex_concurrently_supported": reindex_concurrently_supported(
            int(version_num)
        ),
    }


def table_facts(cursor: Any, table: str) -> dict[str, Any] | None:
    """Live rows, pages and bytes of one table, or ``None`` when absent."""

    # A partitioned parent (schema v23's metric-latest twin) owns no pages:
    # its facts are summed over its partitions and its indexes are theirs.
    cursor.execute(
        """
        WITH members AS (
            SELECT c.oid FROM pg_class c
            WHERE c.oid = to_regclass(%s) AND c.relkind <> 'p'
            UNION ALL
            SELECT i.inhrelid FROM pg_inherits i WHERE i.inhparent = to_regclass(%s)
        )
        SELECT
            %s,
            COALESCE(sum(c.relpages), 0),
            COALESCE(sum(pg_relation_size(c.oid)), 0),
            COALESCE(sum(pg_indexes_size(c.oid)), 0),
            COALESCE(sum(pg_total_relation_size(c.reltoastrelid)), 0),
            COALESCE(sum(pg_total_relation_size(c.oid)), 0),
            COALESCE(
                (SELECT p.reloptions FROM pg_class p WHERE p.oid = to_regclass(%s)),
                (SELECT ch.reloptions FROM pg_inherits ih
                 JOIN pg_class ch ON ch.oid = ih.inhrelid
                 WHERE ih.inhparent = to_regclass(%s)
                 ORDER BY ch.relname LIMIT 1)
            ),
            COALESCE(
                (SELECT t.reloptions FROM pg_class p
                 JOIN pg_class t ON t.oid = p.reltoastrelid
                 WHERE p.oid = to_regclass(%s)),
                ARRAY[]::text[]
            ),
            COALESCE(sum(s.n_live_tup), 0),
            COALESCE(sum(s.n_dead_tup), 0),
            COALESCE(sum(s.n_tup_upd), 0),
            COALESCE(sum(s.n_tup_hot_upd), 0),
            COALESCE(sum(s.seq_scan), 0),
            max(s.last_vacuum),
            max(s.last_autovacuum)
        FROM members m
        JOIN pg_class c ON c.oid = m.oid
        LEFT JOIN pg_stat_user_tables s ON s.relid = c.oid
        HAVING to_regclass(%s) IS NOT NULL
        """,
        (table, table, table, table, table, table, table),
    )
    row = cursor.fetchone()
    if row is None:
        return None
    # The table's own indexes; a partitioned index (relkind I) is sized over
    # the partition indexes attached to it and REINDEX of its name rebuilds
    # every partition in turn.
    cursor.execute(
        """
        SELECT c.relname,
               CASE WHEN c.relkind = 'I' THEN (
                   SELECT COALESCE(sum(pg_relation_size(ci.oid)), 0)
                   FROM pg_inherits ih JOIN pg_class ci ON ci.oid = ih.inhrelid
                   WHERE ih.inhparent = c.oid
               ) ELSE pg_relation_size(c.oid) END,
               i.indisvalid
        FROM pg_index i
        JOIN pg_class c ON c.oid = i.indexrelid
        WHERE i.indrelid = to_regclass(%s)
        ORDER BY c.relname
        """,
        (table,),
    )
    indexes = [
        {"name": str(name), "bytes": int(size), "valid": bool(valid)}
        for name, size, valid in cursor.fetchall()
    ]
    return {
        "table": table,
        "live_rows": int(row[8]),
        "dead_rows": int(row[9]),
        "heap_pages": int(row[1]),
        "heap_bytes": int(row[2]),
        "index_bytes": int(row[3]),
        "toast_bytes": int(row[4]),
        "total_bytes": int(row[5]),
        "reloptions": sorted(str(item) for item in (row[6] or [])),
        "toast_reloptions": sorted(str(item) for item in (row[7] or [])),
        "updates": int(row[10]),
        "hot_updates": int(row[11]),
        "seq_scans": int(row[12]),
        "last_vacuum": row[13].isoformat() if row[13] is not None else None,
        "last_autovacuum": row[14].isoformat() if row[14] is not None else None,
        "indexes": indexes,
    }


def _plan_identity(tables: list[dict[str, Any]]) -> str:
    """What the operator approves: the tables and their actions, not sizes."""

    digest = hashlib.sha256()
    for item in tables:
        digest.update(f"{item['table']}\0{item['action']}\0".encode())
        for index in item.get("indexes", []):
            digest.update(f"{index['name']}\0".encode())
    return digest.hexdigest()


def plan(cursor: Any) -> dict[str, Any]:
    server = server_facts(cursor)
    tables: list[dict[str, Any]] = []
    missing: list[str] = []
    for table in COMPACT_TABLES:
        facts = table_facts(cursor, table)
        if facts is None:
            missing.append(table)
            continue
        tables.append({**facts, "action": COMPACT_ACTION})
    for table in REINDEX_TABLES:
        facts = table_facts(cursor, table)
        if facts is None:
            missing.append(table)
            continue
        tables.append({**facts, "action": REINDEX_ACTION})
    return {
        "server": server,
        "tables": tables,
        "missing_tables": missing,
        "plan_sha256": _plan_identity(tables),
    }


def _require_idle_autocommit(connection: Any) -> None:
    from psycopg.pq import TransactionStatus

    if (
        not getattr(connection, "autocommit", False)
        or connection.info.transaction_status is not TransactionStatus.IDLE
    ):
        raise MaintenanceRefused(
            "VACUUM and REINDEX CONCURRENTLY need an idle autocommit connection"
        )


def _timed(cursor: Any, statement: Any, label: str) -> dict[str, Any]:
    started = time.monotonic()
    cursor.execute(statement)
    return {"statement": label, "seconds": round(time.monotonic() - started, 3)}


def execute(connection: Any, approved_plan_sha256: str) -> dict[str, Any]:
    """Run the plan whose identity the operator approved; report before/after."""

    from psycopg import sql

    _require_idle_autocommit(connection)
    with connection.cursor() as cursor:
        current = plan(cursor)
        if current["plan_sha256"] != approved_plan_sha256:
            raise MaintenanceRefused(
                "the store changed since the plan was reviewed; plan again"
            )
        if not current["server"]["reindex_concurrently_supported"]:
            raise MaintenanceRefused(
                "REINDEX INDEX CONCURRENTLY needs PostgreSQL 12 or newer; "
                f"the store runs {current['server']['version']}"
            )
        cursor.execute(
            "SELECT pg_try_advisory_lock(hashtextextended(%s, 0))",
            (ADVISORY_LOCK_NAME,),
        )
        if not cursor.fetchone()[0]:
            raise MaintenanceRefused(
                "a schema ensure or index build holds the store maintenance lock"
            )
        actions: list[dict[str, Any]] = []
        errors: list[dict[str, str]] = []
        try:
            for item in current["tables"]:
                table = item["table"]
                steps: list[dict[str, Any]] = []
                try:
                    if item["action"] == COMPACT_ACTION:
                        steps.append(
                            _timed(
                                cursor,
                                sql.SQL("VACUUM (FULL, ANALYZE) {}").format(
                                    sql.Identifier(table)
                                ),
                                f"VACUUM (FULL, ANALYZE) {table}",
                            )
                        )
                    else:
                        for index in item["indexes"]:
                            steps.append(
                                _timed(
                                    cursor,
                                    sql.SQL("REINDEX INDEX CONCURRENTLY {}").format(
                                        sql.Identifier(index["name"])
                                    ),
                                    f"REINDEX INDEX CONCURRENTLY {index['name']}",
                                )
                            )
                        steps.append(
                            _timed(
                                cursor,
                                sql.SQL("VACUUM (ANALYZE) {}").format(
                                    sql.Identifier(table)
                                ),
                                f"VACUUM (ANALYZE) {table}",
                            )
                        )
                    status = "SUCCEEDED"
                except Exception as exc:  # noqa: BLE001 - reported per table
                    status = "FAILED"
                    errors.append({"table": table, "error": type(exc).__name__})
                after = table_facts(cursor, table)
                actions.append(
                    {
                        "table": table,
                        "action": item["action"],
                        "status": status,
                        "steps": steps,
                        "before": {
                            key: item[key]
                            for key in (
                                "live_rows",
                                "dead_rows",
                                "heap_pages",
                                "heap_bytes",
                                "index_bytes",
                                "toast_bytes",
                                "total_bytes",
                            )
                        },
                        "after": (
                            {
                                key: after[key]
                                for key in (
                                    "live_rows",
                                    "dead_rows",
                                    "heap_pages",
                                    "heap_bytes",
                                    "index_bytes",
                                    "toast_bytes",
                                    "total_bytes",
                                )
                            }
                            if after is not None
                            else None
                        ),
                    }
                )
        finally:
            cursor.execute(
                "SELECT pg_advisory_unlock(hashtextextended(%s, 0))",
                (ADVISORY_LOCK_NAME,),
            )
    return {
        "server": current["server"],
        "plan_sha256": current["plan_sha256"],
        "actions": actions,
        "errors": errors,
        "status": "FAILED" if errors else "SUCCEEDED",
    }


def run(request: dict[str, Any], connection: Any) -> dict[str, Any]:
    mode = request.get("mode")
    if mode == "plan":
        with connection.cursor() as cursor:
            return {"mode": "plan", **plan(cursor)}
    if mode == "execute":
        approved = str(request.get("plan_sha256") or "")
        if len(approved) != 64:
            raise MaintenanceRefused("execute requires the reviewed plan identity")
        return {"mode": "execute", **execute(connection, approved)}
    raise MaintenanceRefused(f"unsupported store maintenance mode: {mode!r}")


def main() -> int:
    request = json.load(sys.stdin)
    if not isinstance(request, dict):
        raise MaintenanceRefused("store maintenance request is not an object")
    try:
        connection = connect(store_dsn())
        try:
            result = run(request, connection)
        finally:
            connection.close()
    except MaintenanceRefused as exc:
        print(json.dumps({"error": str(exc), "error_kind": "refused"}))
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())

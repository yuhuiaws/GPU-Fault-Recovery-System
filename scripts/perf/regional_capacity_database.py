from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Callable

try:
    from .regional_capacity_registry import STORE_DSN_SNIPPET
except ImportError:
    from regional_capacity_registry import STORE_DSN_SNIPPET

try:
    from gpu_fault.store.postgres.hot_table_bloat import HOT_TABLES
except ImportError:  # pragma: no cover - the perf host always has the checkout
    HOT_TABLES = (
        "gpu_fault_processor_lanes",
        "gpu_fault_processor_queue_counts",
        "gpu_fault_processor_priority_count_shards",
        "gpu_fault_gpu_metric_latest",
        "gpu_fault_objects",
        "gpu_fault_processor_queue",
        "gpu_fault_gpu_metrics_batches",
    )

PAGE_BYTES = 8192


AURORA_CLUSTER_DIMENSION_METRICS = {
    "DatabaseConnections",
    "DBLoad",
}


def aurora_window(
    run_command: Callable,
    *,
    aurora_instance: str,
    aws_region: str,
    start: float,
    end: float,
) -> dict:
    cluster_id = aurora_instance.removesuffix("-writer")
    metrics = {}
    for name in (
        "ServerlessDatabaseCapacity",
        "CPUUtilization",
        "DBLoad",
        "Deadlocks",
        "CommitLatency",
        "DatabaseConnections",
    ):
        dimension = (
            f"Name=DBClusterIdentifier,Value={cluster_id}"
            if name in AURORA_CLUSTER_DIMENSION_METRICS
            else f"Name=DBInstanceIdentifier,Value={aurora_instance}"
        )
        argv = [
            "aws",
            "cloudwatch",
            "get-metric-statistics",
            "--region",
            aws_region,
            "--namespace",
            "AWS/RDS",
            "--metric-name",
            name,
            "--dimensions",
            dimension,
            "--start-time",
            datetime.fromtimestamp(start - 120, timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            ),
            "--end-time",
            datetime.fromtimestamp(end + 180, timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            ),
            "--period",
            "60",
            "--statistics",
            "Average",
            "Maximum",
            "--output",
            "json",
        ]
        try:
            payload = json.loads(run_command(argv, timeout=120).stdout)
        except Exception as exc:  # noqa: BLE001
            metrics[name] = {"error": str(exc)}
            continue
        points = sorted(
            payload.get("Datapoints", []),
            key=lambda item: item["Timestamp"],
        )
        metrics[name] = [
            {
                "timestamp": item["Timestamp"],
                "average": item.get("Average"),
                "maximum": item.get("Maximum"),
            }
            for item in points
        ]
    return metrics


def postgres_counters(
    control_command: Callable,
    pods: list[str],
) -> dict:
    if not pods:
        return {"error": "no api pod available"}
    script = POSTGRES_COUNTERS_SCRIPT.replace(
        "{STORE_DSN_SNIPPET}", STORE_DSN_SNIPPET
    ).replace("{HOT_TABLES}", repr(list(HOT_TABLES)))
    output = control_command(
        "exec",
        pods[0],
        "--",
        "python3",
        "-c",
        script,
        check=False,
        timeout=300,
    )
    for line in reversed(output.splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    return {"error": output[-2000:]}


#: Exec'd in a Running api-ha Pod with ``python3 -c``; the two placeholders
#: are substituted by ``str.replace`` (not ``format``: the body has braces).
POSTGRES_COUNTERS_SCRIPT = """
import json, os, psycopg
{STORE_DSN_SNIPPET}
tables = [
    'gpu_fault_processor_queue',
    'gpu_fault_processor_lanes',
    'gpu_fault_processor_queue_counts',
    'gpu_fault_objects',
    'gpu_fault_gpu_metric_latest',
]
hot_tables = {HOT_TABLES}
out = {}
with psycopg.connect(store_dsn(), autocommit=True) as conn:
    cur = conn.cursor()
    cur.execute(
        "SELECT deadlocks, xact_commit, xact_rollback "
        "FROM pg_stat_database WHERE datname=current_database()"
    )
    deadlocks, commits, rollbacks = cur.fetchone()
    out['deadlocks'] = deadlocks
    out['xact_commit'] = commits
    out['xact_rollback'] = rollbacks
    out['tables'] = {}
    for table in tables:
        cur.execute(
            "SELECT pg_total_relation_size(%s), n_live_tup, n_dead_tup "
            "FROM pg_stat_user_tables WHERE relname=%s",
            (table, table),
        )
        row = cur.fetchone()
        if row is not None:
            out['tables'][table] = {
                'bytes': row[0],
                'live_tuples': row[1],
                'dead_tuples': row[2],
            }
    out['hot_tables'] = {}
    for table in hot_tables:
        cur.execute(
            "SELECT COALESCE(s.n_live_tup, 0), pg_relation_size(c.oid), "
            "pg_indexes_size(c.oid), "
            "COALESCE(pg_total_relation_size(c.reltoastrelid), 0) "
            "FROM pg_class c LEFT JOIN pg_stat_user_tables s ON s.relid=c.oid "
            "WHERE c.oid=to_regclass(%s)",
            (table,),
        )
        row = cur.fetchone()
        if row is None:
            continue
        cur.execute(
            "SELECT c.relname, pg_relation_size(c.oid), c.reltuples "
            "FROM pg_index i JOIN pg_class c ON c.oid=i.indexrelid "
            "WHERE i.indrelid=to_regclass(%s) ORDER BY c.relname",
            (table,),
        )
        out['hot_tables'][table] = {
            'live_rows': int(row[0]),
            'heap_bytes': int(row[1]),
            'index_bytes': int(row[2]),
            'toast_bytes': int(row[3]),
            'indexes': {
                str(name): {'bytes': int(size), 'entries': float(entries)}
                for name, size, entries in cur.fetchall()
            },
        }
print(json.dumps(out))
"""


def hot_table_bloat_ratios(snapshot: dict) -> dict:
    """Per hot table: pages, live rows and the planner-independent ratios.

    ``snapshot`` is a ``postgres_counters`` answer; a snapshot without the
    ``hot_tables`` section (an older control plane, an exec failure) yields
    ``{}`` so the summary records the absence instead of failing the round.
    """

    tables = snapshot.get("hot_tables") if isinstance(snapshot, dict) else None
    if not isinstance(tables, dict):
        return {}
    ratios: dict[str, dict] = {}
    for table, facts in sorted(tables.items()):
        live_rows = int(facts.get("live_rows", 0))
        heap_pages = int(facts.get("heap_bytes", 0)) // PAGE_BYTES
        index_pages = int(facts.get("index_bytes", 0)) // PAGE_BYTES
        densities = []
        for index in (facts.get("indexes") or {}).values():
            pages = int(index.get("bytes", 0)) // PAGE_BYTES
            entries = float(index.get("entries", -1))
            if pages >= 64 and entries >= 0:
                densities.append(round(entries / pages, 3))
        ratios[table] = {
            "live_rows": live_rows,
            "heap_pages": heap_pages,
            "index_pages": index_pages,
            "toast_bytes": int(facts.get("toast_bytes", 0)),
            "heap_pages_per_live_row": round(heap_pages / max(live_rows, 1), 3),
            "index_pages_per_live_row": round(index_pages / max(live_rows, 1), 3),
            "index_min_entries_per_page": min(densities) if densities else None,
        }
    return ratios


def store_bloat_section(before: dict, after: dict) -> dict:
    """The ``store_bloat`` block of a round's summary: the preflight layout
    the round ran on, the layout it left behind, and the worst ratios."""

    before_ratios = hot_table_bloat_ratios(before)
    after_ratios = hot_table_bloat_ratios(after)

    def worst(ratios: dict, key: str) -> float | None:
        values = [
            item[key]
            for item in ratios.values()
            if isinstance(item.get(key), (int, float))
        ]
        return max(values) if values else None

    return {
        "before": before_ratios,
        "after": after_ratios,
        "preflight_max_heap_pages_per_live_row": worst(
            before_ratios, "heap_pages_per_live_row"
        ),
        "preflight_max_index_pages_per_live_row": worst(
            before_ratios, "index_pages_per_live_row"
        ),
        "recorded": bool(before_ratios),
    }


def processor_priority_latency(
    control_command: Callable,
    pods: list[str],
    *,
    cluster_prefix: str,
) -> dict:
    if not pods:
        return {"error": "no api pod available"}
    script = f"""
import json, os, psycopg
{STORE_DSN_SNIPPET}
prefix = {cluster_prefix!r} + '%'
with psycopg.connect(store_dsn(), autocommit=True) as conn:
    cur = conn.cursor()
    cur.execute(
        \"\"\"
        SELECT priority, status, count(*)
        FROM gpu_fault_processor_queue
        WHERE cluster_id LIKE %s
        GROUP BY priority, status
        ORDER BY priority, status
        \"\"\",
        (prefix,),
    )
    statuses = {{}}
    for priority, status, count in cur.fetchall():
        statuses.setdefault(str(priority), {{}})[status] = count
    cur.execute(
        \"\"\"
        SELECT
            priority,
            count(*),
            percentile_cont(0.50) WITHIN GROUP (
                ORDER BY extract(epoch FROM updated_at-created_at)*1000
            ),
            percentile_cont(0.95) WITHIN GROUP (
                ORDER BY extract(epoch FROM updated_at-created_at)*1000
            ),
            percentile_cont(0.99) WITHIN GROUP (
                ORDER BY extract(epoch FROM updated_at-created_at)*1000
            ),
            max(extract(epoch FROM updated_at-created_at)*1000)
        FROM gpu_fault_processor_queue
        WHERE cluster_id LIKE %s
          AND status='COMPLETED'
        GROUP BY priority
        ORDER BY priority
        \"\"\",
        (prefix,),
    )
    latency = {{
        str(priority): {{
            'count': count,
            'p50_ms': float(p50),
            'p95_ms': float(p95),
            'p99_ms': float(p99),
            'max_ms': float(maximum),
        }}
        for priority, count, p50, p95, p99, maximum
        in cur.fetchall()
    }}
    cur.execute(
        \"\"\"
        WITH per_cluster AS (
            SELECT
                priority,
                cluster_id,
                percentile_cont(0.99) WITHIN GROUP (
                    ORDER BY
                    extract(epoch FROM updated_at-created_at)*1000
                ) AS p99_ms
            FROM gpu_fault_processor_queue
            WHERE cluster_id LIKE %s
              AND status='COMPLETED'
            GROUP BY priority, cluster_id
        )
        SELECT priority, min(p99_ms), max(p99_ms), avg(p99_ms)
        FROM per_cluster
        GROUP BY priority
        ORDER BY priority
        \"\"\",
        (prefix,),
    )
    fairness = {{
        str(priority): {{
            'cluster_p99_min_ms': float(minimum),
            'cluster_p99_max_ms': float(maximum),
            'cluster_p99_mean_ms': float(mean),
        }}
        for priority, minimum, maximum, mean in cur.fetchall()
    }}
print(json.dumps({{
    'status_counts': statuses,
    'completion_latency': latency,
    'cluster_fairness': fairness,
}}))
"""
    output = control_command(
        "exec",
        pods[0],
        "--",
        "python3",
        "-c",
        script,
        check=False,
        timeout=300,
    )
    for line in reversed(output.splitlines()):
        if line.startswith("{"):
            return json.loads(line)
    return {"error": output[-2000:]}

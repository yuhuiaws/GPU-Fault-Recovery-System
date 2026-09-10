from __future__ import annotations

from datetime import datetime, timezone
import json
from typing import Callable

try:
    from .regional_capacity_registry import STORE_DSN_SNIPPET
except ImportError:
    from regional_capacity_registry import STORE_DSN_SNIPPET


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
    script = """
import json, os, psycopg
{STORE_DSN_SNIPPET}
tables = [
    'gpu_fault_processor_queue',
    'gpu_fault_processor_lanes',
    'gpu_fault_processor_queue_counts',
    'gpu_fault_objects',
    'gpu_fault_gpu_metric_latest',
]
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
print(json.dumps(out))
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
        line = line.strip()
        if line.startswith("{"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    return {"error": output[-2000:]}


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

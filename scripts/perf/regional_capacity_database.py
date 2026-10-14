from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Callable

try:
    from . import regional_capacity_priority_latency as priority_latency
    from .regional_capacity_registry import STORE_DSN_SNIPPET
except ImportError:
    import regional_capacity_priority_latency as priority_latency
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
    aurora_cluster_id: str,
    aws_region: str,
    start: float,
    end: float,
) -> dict:
    # The cluster-level metrics use the cluster the run named: the writer's
    # instance name is not "<cluster>-writer" on a managed site.
    cluster_id = aurora_cluster_id
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


#: CloudWatch's period for every Aurora reading (``aurora_window``).
PERIOD_SECONDS = 60
#: The oldest datapoint that still stands for "the reading at" a moment.
READING_STALENESS_SECONDS = 300
#: How far before its window a step's Aurora read may begin: a step resolves
#: its writer minutes before the load, and a read longer than a day of
#: one-minute datapoints is refused by CloudWatch.
STEP_LOOKBACK_SECONDS = 3 * 3600
#: ``aurora_window`` used to start this long before the load window; the
#: summary's ``aurora_capacity_max``/``aurora_cpu_max`` keep that range.
LEGACY_LEAD_SECONDS = 120
#: The writer readings a step records at its start and over its window.
STEP_READINGS = {
    "ServerlessDatabaseCapacity": "serverless_capacity_acu",
    "CPUUtilization": "cpu_utilization_percent",
}
#: The exit code of a step whose Aurora measurement failed while its own run
#: succeeded (1 is a failed run or verdict, 2 a usage error, 3 the action
#: suite's budget limit).
MEASUREMENT_FAILED_EXIT = 4
MEASUREMENT_FAILED_REASON = (
    "the Aurora measurement failed: see summary.json aurora_measurement"
)


def _epoch(value: object) -> float | None:
    if isinstance(value, datetime):
        moment = value
    elif isinstance(value, str) and value:
        try:
            moment = datetime.fromisoformat(value)
        except ValueError:
            return None
    else:
        return None
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.timestamp()


Datapoint = tuple[float, float | None]


def _datapoints(metrics: dict, name: str) -> list[tuple[float, Datapoint]]:
    """(period start, (average, maximum)) per datapoint; none for an error."""

    points = metrics.get(name)
    parsed = []
    for point in points if isinstance(points, list) else []:
        at = _epoch(point.get("timestamp")) if isinstance(point, dict) else None
        if at is not None:
            parsed.append((at, (point.get("average"), point.get("maximum"))))
    return sorted(parsed, key=lambda item: item[0])


def _reading_at(
    points: list[tuple[float, Datapoint]], moment: float | None
) -> float | None:
    """The average of the datapoint whose minute holds ``moment``, else of the
    latest earlier one (at most ``READING_STALENESS_SECONDS`` old)."""

    if moment is None:
        return None
    earlier = [
        values
        for at, values in points
        if at <= moment and moment - at < READING_STALENESS_SECONDS
    ]
    if not earlier:
        return None
    average, maximum = earlier[-1]
    value = average if average is not None else maximum
    return None if value is None else float(value)


def _peak_during(
    points: list[tuple[float, Datapoint]], start: float, end: float
) -> float | None:
    """The largest maximum of the datapoints overlapping ``[start, end]``: the
    minute holding ``start`` and every minute that begins inside the window
    (a window that ends before it starts is its first minute alone)."""

    values = [
        float(maximum if maximum is not None else average)
        for at, (average, maximum) in points
        if (at <= start < at + PERIOD_SECONDS or start < at < end)
        and (maximum if maximum is not None else average) is not None
    ]
    return max(values) if values else None


def step_started_at(placement: dict) -> float | None:
    """When the step resolved its Aurora writer (``run_placement``)."""

    return _epoch(placement.get("observed_at"))


def aurora_measurement(
    metrics: dict, *, placement: dict, window_start: float, window_end: float
) -> dict[str, Any]:
    """The step's Aurora readings against the writer it resolved at its start.

    ``serverless_capacity_acu`` and ``cpu_utilization_percent`` of that writer
    at the step start, at the window start and their maximum over the window,
    and the named cluster's peak ``DatabaseConnections`` over the window. A
    cluster whose datapoints all read zero connections is not the one the
    control plane uses (rounds named an orphan cluster at 0.5 ACU and measured
    it without notice): ``FAIL`` fails the step's measurement, never its
    product verdict. No datapoint at all is ``UNVERIFIED``.
    """

    writer = placement.get("aurora") or {}
    step_start = step_started_at(placement)
    document: dict[str, Any] = {
        "cluster_id": writer.get("cluster_id"),
        "writer_instance_id": writer.get("writer_instance_id"),
        "step_started_at_epoch": step_start,
        "window_start_epoch": window_start,
        "window_end_epoch": window_end,
    }
    for name, key in STEP_READINGS.items():
        points = _datapoints(metrics, name)
        document[key] = {
            "at_step_start": _reading_at(points, step_start),
            "at_window_start": _reading_at(points, window_start),
            "max_during_window": _peak_during(points, window_start, window_end),
        }
    connections = _peak_during(
        _datapoints(metrics, "DatabaseConnections"), window_start, window_end
    )
    document["database_connections_max_during_window"] = connections
    cluster = document["cluster_id"]
    if connections is None:
        status, reason = (
            "UNVERIFIED",
            (
                "CloudWatch returned no DatabaseConnections datapoint of Aurora "
                f"cluster {cluster} during the window"
            ),
        )
    elif connections <= 0:
        status, reason = (
            "FAIL",
            (
                f"Aurora cluster {cluster} had zero DatabaseConnections during the "
                "window: it is not the cluster the control plane uses, so its "
                "capacity and CPU readings describe another database "
                "(GPU_FAULT_AURORA_CLUSTER_ID / --aurora-cluster-id)"
            ),
        )
    else:
        status, reason = "PASS", None
    return {**document, "status": status, "reason": reason}


def aurora_step_fields(
    metrics: dict, *, placement: dict, window_start: float, window_end: float
) -> dict[str, Any]:
    """summary.json's Aurora fields: the load-window maxima over the range
    ``aurora_window`` used to read, and the step's ``aurora_measurement``."""

    since = (window_start - LEGACY_LEAD_SECONDS) // PERIOD_SECONDS * PERIOD_SECONDS

    def legacy_max(name: str) -> float | None:
        return max(
            (
                maximum or 0
                for at, (_average, maximum) in _datapoints(metrics, name)
                if at >= since
            ),
            default=None,
        )

    return {
        "aurora_capacity_max": legacy_max("ServerlessDatabaseCapacity"),
        "aurora_cpu_max": legacy_max("CPUUtilization"),
        "aurora_measurement": aurora_measurement(
            metrics,
            placement=placement,
            window_start=window_start,
            window_end=window_end,
        ),
    }


def measured_exit(code: int, summary: dict) -> int:
    """A step's own failure first; else ``MEASUREMENT_FAILED_EXIT`` when its
    Aurora measurement failed."""

    if code:
        return code
    measurement = summary.get("aurora_measurement") or {}
    return MEASUREMENT_FAILED_EXIT if measurement.get("status") == "FAIL" else 0


def step_status(
    code: int, failure: str, measurement: str | None = None
) -> dict[str, Any]:
    """status.json's ``status`` and ``reason`` for a step's exit code."""

    if code == 0:
        return {"status": "ok"}
    if code == MEASUREMENT_FAILED_EXIT:
        return {
            "status": "measurement-failed",
            "reason": measurement or MEASUREMENT_FAILED_REASON,
        }
    return {"status": "aborted", "reason": failure}


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
    # Schema v23 layout facts: the write-layout mode and the partition
    # inventories a formal round must record (performance acceptance plan
    # §3.3). An older control plane lacks the tables and records none.
    cur.execute("SELECT to_regclass('gpu_fault_hot_layout_mode')")
    if cur.fetchone()[0] is not None:
        cur.execute("SELECT mode FROM gpu_fault_hot_layout_mode WHERE singleton")
        row = cur.fetchone()
        layout = {'mode': row[0] if row else None, 'partitions': {}}
        for parent in ('gpu_fault_gpu_metric_latest_clusters', 'gpu_fault_processor_replays'):
            cur.execute(
                "SELECT count(*), COALESCE(sum(c.relpages), 0), COALESCE(sum(s.n_live_tup), 0) "
                "FROM pg_inherits i JOIN pg_class c ON c.oid=i.inhrelid "
                "LEFT JOIN pg_stat_user_tables s ON s.relid=c.oid "
                "WHERE i.inhparent=to_regclass(%s)",
                (parent,),
            )
            count, pages, live = cur.fetchone()
            layout['partitions'][parent] = {
                'count': int(count), 'pages': int(pages), 'live_rows': int(live)
            }
        out['hot_layout'] = layout
    for table in hot_tables:
        # A partitioned parent (schema v23) owns no pages: sum its partitions,
        # and size each of its partitioned indexes over the attached ones.
        cur.execute(
            "WITH members AS ("
            " SELECT c.oid FROM pg_class c WHERE c.oid=to_regclass(%s) AND c.relkind<>'p'"
            " UNION ALL SELECT i.inhrelid FROM pg_inherits i WHERE i.inhparent=to_regclass(%s))"
            " SELECT COALESCE(sum(s.n_live_tup), 0), COALESCE(sum(pg_relation_size(c.oid)), 0),"
            " COALESCE(sum(pg_indexes_size(c.oid)), 0),"
            " COALESCE(sum(pg_total_relation_size(c.reltoastrelid)), 0)"
            " FROM members m JOIN pg_class c ON c.oid=m.oid"
            " LEFT JOIN pg_stat_user_tables s ON s.relid=c.oid"
            " HAVING to_regclass(%s) IS NOT NULL",
            (table, table, table),
        )
        row = cur.fetchone()
        if row is None:
            continue
        cur.execute(
            "SELECT c.relname,"
            " CASE WHEN c.relkind='I' THEN (SELECT COALESCE(sum(pg_relation_size(ci.oid)), 0)"
            "  FROM pg_inherits ih JOIN pg_class ci ON ci.oid=ih.inhrelid WHERE ih.inhparent=c.oid)"
            " ELSE pg_relation_size(c.oid) END,"
            " CASE WHEN c.relkind='I' THEN (SELECT COALESCE(sum(ci.reltuples), 0)"
            "  FROM pg_inherits ih JOIN pg_class ci ON ci.oid=ih.inhrelid WHERE ih.inhparent=c.oid)"
            " ELSE c.reltuples END"
            " FROM pg_index i JOIN pg_class c ON c.oid=i.indexrelid"
            " WHERE i.indrelid=to_regclass(%s) ORDER BY c.relname",
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


#: The compaction policy's index rule (``store.postgres.hot_table_bloat``):
#: an index of at least this many pages holding fewer than this many entries
#: per page is rebuilt. Repeated here so a summary can state the verdict the
#: product reached on the same snapshot; ``tests/regional/
#: test_perf_store_bloat_snapshot.py`` pins both to the product's constants.
INDEX_MIN_PAGES = 64
INDEX_MIN_ENTRIES_PER_PAGE = 8.0


def hot_table_bloat_ratios(snapshot: dict) -> dict:
    """Per hot table: pages, live rows, the planner-independent ratios and
    the index rule's verdict.

    ``index_pages_per_live_row`` is a reporting ratio: with two live rows
    under 149 index pages it reads 74.5 while every one of those indexes is
    below the 64-page floor and nothing is due. ``index_max_pages`` says
    whether the floor was reached at all and ``bloated_indexes`` how many
    indexes the rule flags, so a round's record answers "should the job have
    acted" without the control plane's gauges.

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
        max_pages = 0
        bloated = 0
        for index in (facts.get("indexes") or {}).values():
            pages = int(index.get("bytes", 0)) // PAGE_BYTES
            entries = float(index.get("entries", -1))
            max_pages = max(max_pages, pages)
            if pages >= INDEX_MIN_PAGES and entries >= 0:
                density = round(entries / pages, 3)
                densities.append(density)
                if entries / pages < INDEX_MIN_ENTRIES_PER_PAGE:
                    bloated += 1
        ratios[table] = {
            "live_rows": live_rows,
            "heap_pages": heap_pages,
            "index_pages": index_pages,
            "toast_bytes": int(facts.get("toast_bytes", 0)),
            "heap_pages_per_live_row": round(heap_pages / max(live_rows, 1), 3),
            "index_pages_per_live_row": round(index_pages / max(live_rows, 1), 3),
            "index_min_entries_per_page": min(densities) if densities else None,
            "index_max_pages": max_pages,
            "bloated_indexes": bloated,
        }
    return ratios


def store_layout_section(before: dict, after: dict) -> dict:
    """The ``store_layout`` block of a round's summary (schema v23): the write
    layout the round ran on and the partition inventory before and after, so
    a round in ``inline`` mode or one whose teardown left partitions behind is
    visible in the record rather than only in a later catalog read."""

    def layout(snapshot: dict) -> dict | None:
        value = snapshot.get("hot_layout") if isinstance(snapshot, dict) else None
        return value if isinstance(value, dict) else None

    before_layout = layout(before)
    after_layout = layout(after)
    return {
        "mode": (before_layout or {}).get("mode"),
        "before": before_layout,
        "after": after_layout,
        "recorded": before_layout is not None,
    }


def store_bloat_section(before: dict, after: dict) -> dict:
    """The ``store_bloat`` block of a round's summary: the preflight layout
    the round ran on, the layout it left behind, the worst ratios, and how
    many indexes the compaction rule flagged at either end."""

    before_ratios = hot_table_bloat_ratios(before)
    after_ratios = hot_table_bloat_ratios(after)

    def worst(ratios: dict, key: str) -> float | None:
        values = [
            item[key]
            for item in ratios.values()
            if isinstance(item.get(key), (int, float))
        ]
        return max(values) if values else None

    def flagged(ratios: dict) -> int | None:
        if not ratios:
            return None
        return sum(int(item.get("bloated_indexes", 0)) for item in ratios.values())

    return {
        "before": before_ratios,
        "after": after_ratios,
        "preflight_max_heap_pages_per_live_row": worst(
            before_ratios, "heap_pages_per_live_row"
        ),
        "preflight_max_index_pages_per_live_row": worst(
            before_ratios, "index_pages_per_live_row"
        ),
        "preflight_bloated_indexes": flagged(before_ratios),
        "after_bloated_indexes": flagged(after_ratios),
        "recorded": bool(before_ratios),
    }


def processor_priority_latency(
    control_command: Callable,
    pods: list[str],
    *,
    cluster_prefix: str,
    since_epoch: float = 0.0,
) -> dict:
    """Per-tier completion latency from where the active store layout keeps
    completed requests (``regional_capacity_priority_latency``)."""

    return priority_latency.invoke(
        control_command, pods, cluster_prefix=cluster_prefix, since_epoch=since_epoch
    )

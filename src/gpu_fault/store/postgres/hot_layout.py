"""The v23 hot-table layout at run time: mode reads, partitions, the switch.

``gpu_fault_hot_layout_mode`` is the one-row fact that says which layout
writers use (``ddl_processor_layout.py`` explains why a database fact and not
a process flag: old and new Pods must never disagree while they coexist). A
writer reads it once at the start of its transaction under a shared advisory
lock on the mode key; the operator's switch takes the exclusive lock, so the
two are serialised exactly and no writer straddles the flip. Readers use a
0.25 s cached copy because every read path reads both layouts anyway.

Partition DDL lives here, not in the ``ddl*.py`` stages: a cluster's metric
partition is created when the cluster registers and dropped when it leaves,
and the replay buckets roll with the clock, so neither belongs to a schema
version. Every statement runs under the ``gpu_fault_schema_bootstrap``
advisory lock (try, never wait) with a bounded ``lock_timeout``; a held lock
means the row goes to the DEFAULT partition and the next pass tries again.
"""

from __future__ import annotations

import hashlib
import logging
import re
from datetime import datetime, timedelta, timezone
from typing import Any

from gpu_fault.store.postgres.ddl_processor_layout import (
    LAYOUT_DETACHED,
    LAYOUT_INLINE,
    LAYOUT_MODE_TABLE,
    METRIC_LATEST_CLUSTERS_DEFAULT_PARTITION,
    METRIC_LATEST_CLUSTERS_TABLE,
    METRIC_LATEST_PARTITION_OPTIONS,
    PAYLOADS_TABLE,
    REPLAYS_DEFAULT_PARTITION,
    REPLAYS_TABLE,
)

LOGGER = logging.getLogger(__name__)

METRIC_LATEST_TABLE = "gpu_fault_gpu_metric_latest"
MODE_LOCK_NAME = "gpu_fault_hot_layout_mode"
SCHEMA_LOCK_NAME = "gpu_fault_schema_bootstrap"
LAYOUT_MODES: tuple[str, ...] = (LAYOUT_INLINE, LAYOUT_DETACHED)
#: Replay rows are bucketed by completion time; a bucket is one partition.
REPLAY_BUCKET_SECONDS = 300
#: Buckets are created this far ahead of the clock so a completion always has
#: its partition; the DEFAULT partition catches a gap.
REPLAY_AHEAD_SECONDS = 900
PARTITION_LOCK_TIMEOUT = "5s"
_CLUSTER_PARTITION_PREFIX = f"{METRIC_LATEST_CLUSTERS_TABLE}_c_"
_REPLAY_PARTITION_PREFIX = f"{REPLAYS_TABLE}_p"
_REPLAY_PARTITION_NAME = re.compile(rf"^{re.escape(_REPLAY_PARTITION_PREFIX)}(\d+)$")
_PARTITION_BOUND = re.compile(r"^FOR VALUES FROM \('([^']+)'\) TO \('([^']+)'\)$")
_IDENTIFIER = re.compile(r"^[a-z_][a-z0-9_]*$")

MODE_FOR_WRITE_SQL = f"""
WITH held AS (
    SELECT pg_advisory_xact_lock_shared(hashtextextended(%s, 0))
)
SELECT mode FROM {LAYOUT_MODE_TABLE}, held WHERE singleton
"""


def normalize_layout_mode(value: Any) -> str:
    mode = str(value or LAYOUT_INLINE).strip().lower()
    if mode not in LAYOUT_MODES:
        raise ValueError("hot layout mode must be inline or detached")
    return mode


def read_layout_mode_for_write(cursor: Any) -> str:
    """The mode a writer must use for this transaction (shared lock on the key).

    Rows written by the previous release's Pods during the roll carry the
    inline layout whatever this says; readers handle both, so the lock only
    has to keep *this* writer from straddling the switch.
    """

    cursor.execute(MODE_FOR_WRITE_SQL, (MODE_LOCK_NAME,))
    row = cursor.fetchone()
    return normalize_layout_mode(row[0] if row else LAYOUT_INLINE)


def write_layout_mode(cursor: Any, mode: str) -> None:
    """Upsert the one-row fact (a table emptied by a test harness or never
    seeded gets its row back rather than a silent no-op UPDATE)."""

    cursor.execute(
        f"""
        INSERT INTO {LAYOUT_MODE_TABLE}(singleton, mode, updated_at)
        VALUES (TRUE, %s, now())
        ON CONFLICT(singleton) DO UPDATE SET
            mode=excluded.mode, updated_at=excluded.updated_at
        """,
        (normalize_layout_mode(mode),),
    )


def read_layout_mode(cursor: Any) -> str:
    """An unlocked read for readers and reports."""

    cursor.execute(f"SELECT mode FROM {LAYOUT_MODE_TABLE} WHERE singleton")
    row = cursor.fetchone()
    return normalize_layout_mode(row[0] if row else LAYOUT_INLINE)


def metric_latest_table(mode: str) -> str:
    return (
        METRIC_LATEST_CLUSTERS_TABLE if mode == LAYOUT_DETACHED else METRIC_LATEST_TABLE
    )


def cluster_partition_name(cluster_id: str) -> str:
    """Deterministic, identifier-safe: cluster ids carry characters an SQL
    identifier cannot, and the bound in the catalog keeps the real id."""

    digest = hashlib.sha256(cluster_id.encode("utf-8")).hexdigest()[:16]
    return f"{_CLUSTER_PARTITION_PREFIX}{digest}"


def replay_bucket_start(at: datetime) -> int:
    return int(at.timestamp()) // REPLAY_BUCKET_SECONDS * REPLAY_BUCKET_SECONDS


def replay_partition_name(bucket_start: int) -> str:
    return f"{_REPLAY_PARTITION_PREFIX}{bucket_start}"


def replay_bucket_bounds(name: str) -> tuple[int, int] | None:
    match = _REPLAY_PARTITION_NAME.match(name)
    if match is None:
        return None
    start = int(match.group(1))
    return start, start + REPLAY_BUCKET_SECONDS


def _safe_identifier(name: str) -> str:
    if _IDENTIFIER.fullmatch(name) is None or not name.startswith("gpu_fault_"):
        raise ValueError(f"refusing to touch an unexpected relation: {name!r}")
    return name


def _bound_timeout(cursor: Any) -> None:
    cursor.execute(f"SET LOCAL lock_timeout = '{PARTITION_LOCK_TIMEOUT}'")


def list_partitions(cursor: Any, parent: str) -> list[dict[str, Any]]:
    """Every partition of ``parent`` with its bound, pages and live rows."""

    cursor.execute(
        """
        SELECT c.relname, pg_get_expr(c.relpartbound, c.oid), c.relpages,
               COALESCE(s.n_live_tup, 0), pg_relation_size(c.oid),
               pg_indexes_size(c.oid)
        FROM pg_inherits i
        JOIN pg_class c ON c.oid = i.inhrelid
        LEFT JOIN pg_stat_user_tables s ON s.relid = c.oid
        WHERE i.inhparent = to_regclass(%s)
        ORDER BY c.relname
        """,
        (parent,),
    )
    return [
        {
            "name": str(name),
            "bound": str(bound),
            "pages": int(pages),
            "live_rows": int(live),
            "heap_bytes": int(heap),
            "index_bytes": int(indexes),
        }
        for name, bound, pages, live, heap, indexes in cursor.fetchall()
    ]


def ensure_cluster_partition(cursor: Any, cluster_id: str) -> str | None:
    """Create the cluster's metric partition inside the caller's transaction.

    Returns the partition name when it was created, ``None`` when it already
    existed. Rows that reached the DEFAULT partition first are moved into the
    new partition before it is attached (a LIST partition cannot be created
    while the default holds rows of its value).
    """

    name = _safe_identifier(cluster_partition_name(cluster_id))
    cursor.execute("SELECT to_regclass(%s)", (name,))
    if cursor.fetchone()[0] is not None:
        return None
    _bound_timeout(cursor)
    cursor.execute(
        f"SELECT EXISTS (SELECT 1 FROM {METRIC_LATEST_CLUSTERS_DEFAULT_PARTITION} "
        "WHERE cluster_id = %s)",
        (cluster_id,),
    )
    if cursor.fetchone()[0]:
        cursor.execute(
            f"CREATE TABLE {name} (LIKE {METRIC_LATEST_CLUSTERS_TABLE} INCLUDING ALL) "
            f"WITH ({METRIC_LATEST_PARTITION_OPTIONS})"
        )
        cursor.execute(
            f"WITH moved AS (DELETE FROM {METRIC_LATEST_CLUSTERS_DEFAULT_PARTITION} "
            "WHERE cluster_id = %s RETURNING *) "
            f"INSERT INTO {name} SELECT * FROM moved",
            (cluster_id,),
        )
        # DDL takes no bind parameters: the bound is a quoted literal. The
        # driver is imported here, not at module level: the store package must
        # import without the optional PostgreSQL driver (tests, in-memory and
        # SQLite deployments), and only a live cursor ever reaches this DDL.
        from psycopg import sql

        cursor.execute(
            sql.SQL(
                f"ALTER TABLE {METRIC_LATEST_CLUSTERS_TABLE} ATTACH PARTITION {name} "
                "FOR VALUES IN ({})"
            ).format(sql.Literal(cluster_id))
        )
        return name
    from psycopg import sql

    cursor.execute(
        sql.SQL(
            f"CREATE TABLE IF NOT EXISTS {name} PARTITION OF "
            f"{METRIC_LATEST_CLUSTERS_TABLE} FOR VALUES IN ({{}}) "
            f"WITH ({METRIC_LATEST_PARTITION_OPTIONS})"
        ).format(sql.Literal(cluster_id))
    )
    return name


def drop_cluster_partition(cursor: Any, cluster_id: str) -> str | None:
    """Return the cluster's metric pages wholesale; rows left in the DEFAULT
    partition for the cluster are deleted so nothing of it survives."""

    name = _safe_identifier(cluster_partition_name(cluster_id))
    _bound_timeout(cursor)
    cursor.execute("SELECT to_regclass(%s)", (name,))
    dropped = None
    if cursor.fetchone()[0] is not None:
        cursor.execute(f"DROP TABLE IF EXISTS {name}")
        dropped = name
    cursor.execute(
        f"DELETE FROM {METRIC_LATEST_CLUSTERS_DEFAULT_PARTITION} WHERE cluster_id = %s",
        (cluster_id,),
    )
    return dropped


def ensure_replay_partitions(cursor: Any, now: datetime) -> list[str]:
    """Create the replay buckets from one bucket behind the clock to
    ``REPLAY_AHEAD_SECONDS`` ahead; returns the names created.

    A bucket whose range already has rows in the DEFAULT partition cannot be
    created (Postgres refuses the default's new constraint); those rows age
    out through the default's row sweep and the bucket is skipped.
    """

    created: list[str] = []
    start = replay_bucket_start(now) - REPLAY_BUCKET_SECONDS
    end = int(now.timestamp()) + REPLAY_AHEAD_SECONDS
    _bound_timeout(cursor)
    for bucket in range(start, end, REPLAY_BUCKET_SECONDS):
        name = _safe_identifier(replay_partition_name(bucket))
        cursor.execute("SELECT to_regclass(%s)", (name,))
        if cursor.fetchone()[0] is not None:
            continue
        cursor.execute(
            f"SELECT EXISTS (SELECT 1 FROM {REPLAYS_DEFAULT_PARTITION} "
            "WHERE completed_at >= to_timestamp(%s) AND completed_at < to_timestamp(%s))",
            (bucket, bucket + REPLAY_BUCKET_SECONDS),
        )
        if cursor.fetchone()[0]:
            continue
        # Bucket bounds are integers rendered into the DDL (no bind parameters).
        cursor.execute(
            f"CREATE TABLE IF NOT EXISTS {name} PARTITION OF {REPLAYS_TABLE} "
            f"FOR VALUES FROM (to_timestamp({int(bucket)})) "
            f"TO (to_timestamp({int(bucket) + REPLAY_BUCKET_SECONDS}))"
        )
        created.append(name)
    return created


def drop_expired_replay_partitions(
    cursor: Any, older_than: datetime
) -> list[tuple[str, int]]:
    """Drop every bucket whose upper bound is past ``older_than``; returns
    ``(name, live rows)`` per dropped bucket."""

    dropped: list[tuple[str, int]] = []
    _bound_timeout(cursor)
    for partition in list_partitions(cursor, REPLAYS_TABLE):
        bounds = replay_bucket_bounds(partition["name"])
        if bounds is None:
            continue
        if datetime.fromtimestamp(bounds[1], timezone.utc) > older_than:
            continue
        name = _safe_identifier(partition["name"])
        cursor.execute(f"SELECT count(*) FROM {name}")
        rows = int(cursor.fetchone()[0])
        cursor.execute(f"DROP TABLE IF EXISTS {name}")
        dropped.append((name, rows))
    return dropped


def sweep_default_replays(cursor: Any, older_than: datetime, limit: int) -> int:
    """Row sweep of the DEFAULT replay partition (the gap catch-all)."""

    cursor.execute(
        f"""
        WITH expired AS (
            SELECT request_id, completed_at FROM {REPLAYS_DEFAULT_PARTITION}
            WHERE completed_at <= %s
            ORDER BY completed_at, request_id
            LIMIT %s
            FOR UPDATE SKIP LOCKED
        ),
        deleted AS (
            DELETE FROM {REPLAYS_DEFAULT_PARTITION} AS target
            USING expired
            WHERE target.request_id = expired.request_id
              AND target.completed_at = expired.completed_at
            RETURNING target.request_id
        )
        SELECT count(*) FROM deleted
        """,
        (older_than, limit),
    )
    return int(cursor.fetchone()[0])


def layout_status(cursor: Any) -> dict[str, Any]:
    """Mode plus the partition inventory, for the CLI and the perf runners."""

    cursor.execute("SELECT version FROM gpu_fault_schema_version WHERE singleton")
    row = cursor.fetchone()
    status: dict[str, Any] = {
        "schema_version": int(row[0]) if row else None,
        "mode": read_layout_mode(cursor),
        "metric_latest_partitions": list_partitions(
            cursor, METRIC_LATEST_CLUSTERS_TABLE
        ),
        "replay_partitions": list_partitions(cursor, REPLAYS_TABLE),
    }
    for table in ("gpu_fault_processor_queue", PAYLOADS_TABLE, METRIC_LATEST_TABLE):
        cursor.execute(
            """
            SELECT c.relpages, COALESCE(s.n_live_tup, 0), pg_indexes_size(c.oid),
                   COALESCE(pg_total_relation_size(c.reltoastrelid), 0)
            FROM pg_class c LEFT JOIN pg_stat_user_tables s ON s.relid = c.oid
            WHERE c.oid = to_regclass(%s)
            """,
            (table,),
        )
        facts = cursor.fetchone()
        status[table] = (
            None
            if facts is None
            else {
                "heap_pages": int(facts[0]),
                "live_rows": int(facts[1]),
                "index_bytes": int(facts[2]),
                "toast_bytes": int(facts[3]),
            }
        )
    return status


def seed_fresh_hot_layout(cursor: Any) -> dict[str, Any]:
    """A database that had no schema starts detached (the bootstrap proved it
    fresh before any DDL); an upgraded one keeps ``inline`` until the switch."""

    write_layout_mode(cursor, LAYOUT_DETACHED)
    cursor.execute(f"SELECT count(*) FROM {METRIC_LATEST_TABLE}")
    if int(cursor.fetchone()[0]):
        raise RuntimeError("fresh hot-layout seeding found metric latest rows")
    ensure_replay_partitions(cursor, datetime.now(timezone.utc))
    return {"seeded": True, "mode": LAYOUT_DETACHED}


def switch_to_detached(
    connection: Any, *, now: datetime | None = None
) -> dict[str, Any]:
    """``gpu-fault-store-migrate --set-hot-layout detached``: one transaction
    under the exclusive mode lock and the schema maintenance lock.

    Copies every row of the flat metric-latest table into the cluster
    partitions (creating one per cluster present or registered), creates the
    replay buckets and flips the mode. One-way, idempotent: a database already
    detached reports its status and changes nothing. ``connection`` must be
    autocommit so the maintenance lock is a session lock held for the copy.
    """

    at = now or datetime.now(timezone.utc)
    with connection.cursor() as cursor:
        cursor.execute(
            "SELECT pg_try_advisory_lock(hashtextextended(%s, 0))",
            (SCHEMA_LOCK_NAME,),
        )
        if not cursor.fetchone()[0]:
            raise RuntimeError(
                "a schema ensure, index build or store maintenance holds the lock"
            )
    try:
        with connection.transaction():
            with connection.cursor() as cursor:
                cursor.execute(
                    "SELECT version FROM gpu_fault_schema_version WHERE singleton"
                )
                row = cursor.fetchone()
                if row is None or int(row[0]) < 23:
                    raise RuntimeError(
                        "the detached hot layout needs schema version 23 or newer"
                    )
                cursor.execute(
                    "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))",
                    (MODE_LOCK_NAME,),
                )
                if read_layout_mode(cursor) == LAYOUT_DETACHED:
                    return {"changed": False, **layout_status(cursor)}
                cursor.execute(
                    f"SELECT DISTINCT cluster_id FROM {METRIC_LATEST_TABLE} "
                    "UNION SELECT key FROM gpu_fault_objects WHERE kind='regional_cluster' "
                    "ORDER BY 1"
                )
                clusters = [str(value) for (value,) in cursor.fetchall()]
                created = [
                    name
                    for cluster_id in clusters
                    if (name := ensure_cluster_partition(cursor, cluster_id))
                    is not None
                ]
                cursor.execute(
                    f"""
                    INSERT INTO {METRIC_LATEST_CLUSTERS_TABLE}
                        (key, cluster_id, node_id, observed_at, payload)
                    SELECT key, cluster_id, node_id, observed_at, payload
                    FROM {METRIC_LATEST_TABLE}
                    ON CONFLICT (cluster_id, key) DO UPDATE SET
                        node_id = excluded.node_id,
                        observed_at = excluded.observed_at,
                        payload = excluded.payload
                    WHERE excluded.observed_at > {METRIC_LATEST_CLUSTERS_TABLE}.observed_at
                    """
                )
                copied = cursor.rowcount
                buckets = ensure_replay_partitions(cursor, at)
                write_layout_mode(cursor, LAYOUT_DETACHED)
                return {
                    "changed": True,
                    "metric_latest_rows_copied": int(copied),
                    "cluster_partitions_created": created,
                    "replay_partitions_created": buckets,
                    **layout_status(cursor),
                }
    finally:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT pg_advisory_unlock(hashtextextended(%s, 0))",
                (SCHEMA_LOCK_NAME,),
            )


def maintain_replay_partitions(
    connection: Any, *, now: datetime, older_than: datetime, limit: int
) -> dict[str, Any]:
    """The processor-cleanup job's partition pass on an autocommit connection:
    create the buckets ahead, drop the expired ones, sweep the default.

    Skips (and says so) when the schema maintenance lock is held; the next
    pass picks it up. Each statement is its own transaction so one refusal
    cannot roll the others back.
    """

    result: dict[str, Any] = {
        "created": [],
        "dropped": [],
        "dropped_rows": 0,
        "default_swept": 0,
        "skipped": False,
    }
    with connection.cursor() as cursor:
        cursor.execute(
            "SELECT pg_try_advisory_lock(hashtextextended(%s, 0))",
            (SCHEMA_LOCK_NAME,),
        )
        if not cursor.fetchone()[0]:
            result["skipped"] = True
            return result
    try:
        with connection.transaction():
            with connection.cursor() as cursor:
                result["created"] = ensure_replay_partitions(cursor, now)
        with connection.transaction():
            with connection.cursor() as cursor:
                dropped = drop_expired_replay_partitions(cursor, older_than)
                result["dropped"] = [name for name, _ in dropped]
                result["dropped_rows"] = sum(rows for _, rows in dropped)
        with connection.transaction():
            with connection.cursor() as cursor:
                result["default_swept"] = sweep_default_replays(
                    cursor, older_than, limit
                )
    finally:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT pg_advisory_unlock(hashtextextended(%s, 0))",
                (SCHEMA_LOCK_NAME,),
            )
    return result


def replay_retention_cutoff(now: datetime, retention: timedelta) -> datetime:
    return now - retention

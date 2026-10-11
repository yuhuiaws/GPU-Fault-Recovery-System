"""Hot-table bloat: measure it from the catalog, compact it from the product.

Three perf rounds on one release, one database and identical admission
defaults pinned the shape of the problem. A fresh store took 1,572 req/s;
after one 32-cluster round had inserted 264k synthetic metric rows, 87k
objects and 27k queue rows and the round's teardown had deleted them again,
the same workload took 538 req/s with 49 % of the active statements waiting
on ``LWLock:BufferContent`` -- the queue INSERT (84 % of its samples), the
completion CTE (78 %) and the claim CTE (48 %); after ``REINDEX INDEX
CONCURRENTLY`` of every hot-table index and ``VACUUM FULL`` of the three
small counter/lane tables, with the large heaps and the queue's TOAST left
exactly as bloated, it took 1,141 req/s with zero BufferContent samples. The
convoy is carried by the index files (``gpu_fault_processor_queue`` 36 MiB of
indexes for 246 rows, ``gpu_fault_gpu_metric_latest`` 118 MiB for 1,233) and
the small tables (``gpu_fault_processor_queue_counts`` 280 rows in 127 pages,
read in full by every admission); the heaps cost the remaining quarter.
VACUUM returns none of those pages: a B-tree only ever recycles its deleted
pages, a heap is truncated only behind its last live row, and the rows that
survive a mass delete sit scattered across the whole file.

The policy here is the planner-independent counterpart of the operator verb
``gpu-fault-admin store-maintenance``: it reads ``pg_class``/``pg_stat_user_
tables`` for the hot tables and their indexes, derives physical ratios that
need no statistics target or cost model -- heap pages per live row, index
pages per live row, index entries per page -- and, when an index of at least
``INDEX_MIN_PAGES`` holds fewer than ``INDEX_MIN_ENTRIES_PER_PAGE`` entries
per page or one of the small in-place tables holds fewer than
``SMALL_HEAP_MIN_ROWS_PER_PAGE`` rows per page over at least
``SMALL_HEAP_MIN_PAGES`` pages, rebuilds exactly those objects: ``REINDEX
INDEX CONCURRENTLY`` per index (no long table lock; an invalid ``_ccnew``
leftover of an interrupted rebuild is dropped first) and ``VACUUM (FULL,
ANALYZE)`` for the small tables (an exclusive lock of well under a second at
their live size). The thresholds are deliberately an order of magnitude
below any compact layout -- even 500-byte index entries pack 16 to a page,
the counter rows pack dozens -- so a busy queue holding its working set is
never touched, and a store that just lost a synthetic fleet is compacted
within one job interval instead of after the next failed acceptance round.

Both halves run through the store's own autocommit pool on one pinned
connection under the ``gpu_fault_schema_bootstrap`` advisory lock the schema
ensure and the operator verb take, so none of the three can overlap; a held
lock skips the round rather than waits. Large heaps and the queue's TOAST
relation are measured and exported but never rewritten here: compacting them
needs ``VACUUM FULL``'s exclusive lock for the duration of the copy, which is
an operator decision for a maintenance window, not a background job.
"""

from __future__ import annotations

import logging
import re
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from gpu_fault.store.postgres.pool import PooledPostgresDatabase

LOGGER = logging.getLogger(__name__)

#: Small tables rewritten in place by every queue statement: compacted with
#: ``VACUUM (FULL, ANALYZE)`` when their rows thin out over many pages.
COMPACT_TABLES: tuple[str, ...] = (
    "gpu_fault_processor_lanes",
    "gpu_fault_processor_queue_counts",
    "gpu_fault_processor_priority_count_shards",
)
#: Large or continuously read tables: only their indexes are rebuilt.
REINDEX_TABLES: tuple[str, ...] = (
    "gpu_fault_gpu_metric_latest",
    "gpu_fault_objects",
    "gpu_fault_processor_queue",
    "gpu_fault_gpu_metrics_batches",
)
HOT_TABLES: tuple[str, ...] = COMPACT_TABLES + REINDEX_TABLES

#: An index smaller than this (in 8 KiB pages; 512 KiB) is never bloated.
INDEX_MIN_PAGES = 64
#: Below this many live entries per index page the index is rebuilt.
INDEX_MIN_ENTRIES_PER_PAGE = 8.0
#: A small table's heap below this many pages (128 KiB) is never bloated.
SMALL_HEAP_MIN_PAGES = 16
#: Below this many live rows per heap page a small table is rewritten.
SMALL_HEAP_MIN_ROWS_PER_PAGE = 4.0

ACTION_REINDEX = "REINDEX INDEX CONCURRENTLY"
ACTION_VACUUM_FULL = "VACUUM (FULL, ANALYZE)"
ADVISORY_LOCK_NAME = "gpu_fault_schema_bootstrap"
LOCK_TIMEOUT = "5s"
STATEMENT_TIMEOUT = "600s"
PAGE_BYTES = 8192

_TABLE_FACTS_SQL = """
SELECT
    c.oid::regclass::text,
    COALESCE(s.n_live_tup, 0),
    COALESCE(s.n_dead_tup, 0),
    pg_relation_size(c.oid),
    pg_indexes_size(c.oid),
    COALESCE(pg_total_relation_size(c.reltoastrelid), 0)
FROM pg_class c
LEFT JOIN pg_stat_user_tables s ON s.relid = c.oid
WHERE c.oid = to_regclass(%s)
"""
_INDEX_FACTS_SQL = """
SELECT c.relname, pg_relation_size(c.oid), c.reltuples, i.indisvalid, am.amname
FROM pg_index i
JOIN pg_class c ON c.oid = i.indexrelid
JOIN pg_am am ON am.oid = c.relam
WHERE i.indrelid = to_regclass(%s)
ORDER BY c.relname
"""


def _ratio(numerator: float, denominator: float) -> float:
    return round(numerator / denominator, 3) if denominator > 0 else 0.0


def index_is_bloated(pages: int, entries: float) -> bool:
    """The rebuild rule for one index (``reltuples`` -1 means never measured)."""

    if pages < INDEX_MIN_PAGES or entries < 0:
        return False
    return entries / pages < INDEX_MIN_ENTRIES_PER_PAGE


def small_heap_is_bloated(pages: int, live_rows: int) -> bool:
    """The rewrite rule for a small in-place table."""

    if pages < SMALL_HEAP_MIN_PAGES:
        return False
    return live_rows / pages < SMALL_HEAP_MIN_ROWS_PER_PAGE


def table_facts(cursor: Any, table: str) -> dict[str, Any] | None:
    """Physical facts of one hot table from the catalog, or ``None`` if absent."""

    cursor.execute(_TABLE_FACTS_SQL, (table,))
    row = cursor.fetchone()
    if row is None:
        return None
    live_rows = int(row[1])
    heap_pages = int(row[3]) // PAGE_BYTES
    index_pages = int(row[4]) // PAGE_BYTES
    cursor.execute(_INDEX_FACTS_SQL, (table,))
    indexes: list[dict[str, Any]] = []
    densities: list[float] = []
    for name, size, reltuples, valid, am in cursor.fetchall():
        pages = int(size) // PAGE_BYTES
        entries = float(reltuples)
        density = _ratio(entries, pages) if entries >= 0 else None
        if density is not None and pages >= INDEX_MIN_PAGES:
            densities.append(density)
        indexes.append(
            {
                "name": str(name),
                "pages": pages,
                "entries": entries,
                "entries_per_page": density,
                "valid": bool(valid),
                "am": str(am),
                "bloated": bool(valid) and index_is_bloated(pages, entries),
            }
        )
    return {
        "table": table,
        "small": table in COMPACT_TABLES,
        "live_rows": live_rows,
        "dead_rows": int(row[2]),
        "heap_pages": heap_pages,
        "heap_bytes": int(row[3]),
        "index_pages": index_pages,
        "index_bytes": int(row[4]),
        "toast_bytes": int(row[5]),
        "heap_pages_per_live_row": _ratio(heap_pages, max(live_rows, 1)),
        "index_pages_per_live_row": _ratio(index_pages, max(live_rows, 1)),
        "index_min_entries_per_page": min(densities) if densities else None,
        "heap_bloated": table in COMPACT_TABLES
        and small_heap_is_bloated(heap_pages, live_rows),
        "indexes": indexes,
    }


def assess_hot_tables(cursor: Any) -> dict[str, Any]:
    """Facts for every hot table plus the compaction actions they call for."""

    tables: list[dict[str, Any]] = []
    actions: list[dict[str, str]] = []
    missing: list[str] = []
    for table in HOT_TABLES:
        facts = table_facts(cursor, table)
        if facts is None:
            missing.append(table)
            continue
        tables.append(facts)
        if facts["heap_bloated"]:
            actions.append(
                {"table": table, "action": ACTION_VACUUM_FULL, "target": table}
            )
        for index in facts["indexes"]:
            if index["bloated"]:
                actions.append(
                    {"table": table, "action": ACTION_REINDEX, "target": index["name"]}
                )
    return {
        "tables": tables,
        "missing_tables": missing,
        "actions": actions,
        "measured_at": time.time(),
    }


def _timed(cursor: Any, statement: str) -> float:
    started = time.monotonic()
    cursor.execute(statement)
    return round(time.monotonic() - started, 3)


def _drop_invalid_rebuild_leftovers(cursor: Any, table: str) -> list[str]:
    """``REINDEX INDEX CONCURRENTLY`` interrupted mid-way leaves an invalid
    ``<index>_ccnew`` twin that blocks nothing but the next rebuild."""

    cursor.execute(
        """
        SELECT c.relname
        FROM pg_index i JOIN pg_class c ON c.oid = i.indexrelid
        WHERE i.indrelid = to_regclass(%s) AND NOT i.indisvalid
          AND c.relname LIKE '%%\\_ccnew%%'
        ORDER BY c.relname
        """,
        (table,),
    )
    dropped = []
    for (name,) in cursor.fetchall():
        cursor.execute(f'DROP INDEX CONCURRENTLY IF EXISTS "{name}"')
        dropped.append(str(name))
    return dropped


def _safe_identifier(name: str) -> str:
    if not name.replace("_", "").isalnum() or not name.startswith("gpu_fault_"):
        raise ValueError(f"refusing to compact an unexpected relation: {name!r}")
    return name


_TIMEOUT_SETTINGS = ("lock_timeout", "statement_timeout")
_TIMEOUT_VALUE = re.compile(r"^-?[0-9]+(?:\.[0-9]+)?(?:us|ms|s|min|h|d)?$")


def _session_timeouts(cursor: Any) -> dict[str, str]:
    """The connection's own timeouts, to put back after the maintenance ones.

    ``RESET`` would only restore the startup default; the pool's connections
    carry the store's statement timeout as a startup option, but a caller's
    explicit ``SET`` must survive the run as well, so the values are read and
    re-applied. A value that is not a Postgres interval literal is refused
    before it is interpolated into ``SET``.
    """

    saved: dict[str, str] = {}
    for setting in _TIMEOUT_SETTINGS:
        cursor.execute(f"SHOW {setting}")
        value = str(cursor.fetchone()[0])
        if _TIMEOUT_VALUE.fullmatch(value) is None:
            raise ValueError(f"unexpected {setting} value {value!r}")
        saved[setting] = value
    return saved


def compact(connection: Any, assessment: dict[str, Any]) -> dict[str, Any]:
    """Run ``assessment["actions"]`` on an autocommit ``connection``.

    The connection is pinned for the whole run (the advisory lock is a
    session lock) and its lock/statement timeouts are reset before it goes
    back to the pool. A failed action is recorded and the next one still runs;
    the result carries each touched table's facts after the work.
    """

    actions = list(assessment.get("actions") or [])
    result: dict[str, Any] = {"status": "NOOP", "actions": [], "after": {}}
    if not actions:
        return result
    with connection.cursor() as cursor:
        cursor.execute(
            "SELECT pg_try_advisory_lock(hashtextextended(%s, 0))",
            (ADVISORY_LOCK_NAME,),
        )
        if not cursor.fetchone()[0]:
            result["status"] = "SKIPPED"
            result["reason"] = "a schema ensure or store maintenance holds the lock"
            return result
        saved = _session_timeouts(cursor)
        try:
            cursor.execute(f"SET lock_timeout = '{LOCK_TIMEOUT}'")
            cursor.execute(f"SET statement_timeout = '{STATEMENT_TIMEOUT}'")
            failed = False
            for item in actions:
                table = _safe_identifier(str(item["table"]))
                target = _safe_identifier(str(item["target"]))
                record: dict[str, Any] = {**item, "status": "SUCCEEDED"}
                try:
                    if item["action"] == ACTION_VACUUM_FULL:
                        record["seconds"] = _timed(
                            cursor, f"VACUUM (FULL, ANALYZE) {table}"
                        )
                    elif item["action"] == ACTION_REINDEX:
                        record["dropped"] = _drop_invalid_rebuild_leftovers(
                            cursor, table
                        )
                        record["seconds"] = _timed(
                            cursor, f"REINDEX INDEX CONCURRENTLY {target}"
                        )
                    else:
                        raise ValueError(
                            f"unknown compaction action {item['action']!r}"
                        )
                except Exception as exc:  # noqa: BLE001 - reported per action
                    failed = True
                    record["status"] = "FAILED"
                    record["error"] = type(exc).__name__
                    LOGGER.warning(
                        "store compaction %s %s failed: %s",
                        item["action"],
                        target,
                        type(exc).__name__,
                    )
                result["actions"].append(record)
            for table in sorted({str(item["table"]) for item in actions}):
                facts = table_facts(cursor, table)
                if facts is not None:
                    result["after"][table] = facts
            result["status"] = "FAILED" if failed else "SUCCEEDED"
        finally:
            for setting, value in saved.items():
                cursor.execute(f"SET {setting} = '{value}'")
            cursor.execute(
                "SELECT pg_advisory_unlock(hashtextextended(%s, 0))",
                (ADVISORY_LOCK_NAME,),
            )
    return result


class PostgresHotTableBloatMixin:
    """``PostgresStore``'s view of the policy: measure, then compact."""

    _db: PooledPostgresDatabase

    def hot_table_bloat(self) -> dict[str, Any]:
        """Catalog facts and due compaction actions for the hot tables."""

        with self._db.cursor() as cursor:
            return assess_hot_tables(cursor)

    def compact_hot_tables(
        self, assessment: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Compact what ``assessment`` (default: a fresh one) calls for.

        Needs the pool's autocommit connections: ``REINDEX CONCURRENTLY`` and
        ``VACUUM`` cannot run inside a transaction block, so a caller holding
        a store transaction is refused instead of deadlocking on itself.
        """

        if getattr(self._db, "in_transaction", False):
            raise RuntimeError("compact_hot_tables cannot run inside a transaction")
        with self._db.pool.connection() as connection:
            if assessment is None:
                with connection.cursor() as cursor:
                    assessment = assess_hot_tables(cursor)
            return compact(connection, assessment)

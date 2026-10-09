"""Schema v19 stage: storage parameters of the hot tables.

A ``ddl*.py`` module like the others so the migration registry checksums it;
kept apart from ``ddl.py`` only because that file is at its size ratchet.

A day of campaign churn on the live store left ``gpu_fault_gpu_metric_latest``
with 1,216 live rows in 23,435 heap pages (192 MB) plus 124 MB of indexes,
``gpu_fault_processor_lanes`` with 40 rows in 1,188 pages and
``gpu_fault_processor_queue_counts`` with 3-5 rows in 62 pages read 212k
times: every admission read scanned 60-1,200 pages for a handful of rows,
every metric upsert walked bloated indexes, control-plane ingestion halved
(1428 -> 663 req/s) and the perf verdict's reserved 503 share went red. Only
``gpu_fault_objects`` had per-table autovacuum settings; the hot tables ran
on the instance defaults (scale factor 0.1), and none of their updates was
HOT (``n_tup_hot_upd`` 0), so each one left a dead tuple in every index.

Two things change here. Tables whose rows are rewritten in place get a
``fillfactor`` so a HOT update finds room on the same page, and every hot
table gets threshold-only autovacuum (``*_scale_factor=0``, ``*_threshold=50``)
with no cost delay, so a few dozen dead tuples trigger a vacuum instead of a
tenth of the table; the queue's TOAST relation gets the same vacuum
thresholds. ``ALTER TABLE ... SET`` runs only when the catalog differs (item
J: the ALTER takes ShareUpdateExclusiveLock even as a no-op), so a no-op
ensure stays lock-free. Reloptions do not return pages that are already
bloated; that is ``gpu-fault-admin store-maintenance``.
"""

from __future__ import annotations

import re
from typing import Any

from gpu_fault.store.postgres.ddl_helpers import _is_numeric

_THRESHOLD_ONLY_AUTOVACUUM = {
    "autovacuum_vacuum_scale_factor": "0",
    "autovacuum_vacuum_threshold": "50",
    "autovacuum_analyze_scale_factor": "0",
    "autovacuum_analyze_threshold": "50",
    "autovacuum_vacuum_cost_delay": "0",
}
# TOAST relations are vacuumed, never analyzed: only the vacuum options apply.
_TOAST_THRESHOLD_ONLY_AUTOVACUUM = {
    "toast.autovacuum_vacuum_scale_factor": "0",
    "toast.autovacuum_vacuum_threshold": "50",
    "toast.autovacuum_vacuum_cost_delay": "0",
}

#: Table -> storage parameters. ``fillfactor`` only where rows are updated in
#: place (the counters, the lanes, the queue and the latest-metric rows); the
#: batch idempotency records are insert-only and swept after a day.
HOT_TABLE_STORAGE_PARAMETERS: dict[str, dict[str, str]] = {
    "gpu_fault_processor_queue_counts": {
        "fillfactor": "50",
        **_THRESHOLD_ONLY_AUTOVACUUM,
    },
    "gpu_fault_processor_priority_count_shards": {
        "fillfactor": "50",
        **_THRESHOLD_ONLY_AUTOVACUUM,
    },
    "gpu_fault_processor_lanes": {
        "fillfactor": "50",
        **_THRESHOLD_ONLY_AUTOVACUUM,
    },
    "gpu_fault_processor_queue": {
        "fillfactor": "70",
        **_THRESHOLD_ONLY_AUTOVACUUM,
        **_TOAST_THRESHOLD_ONLY_AUTOVACUUM,
    },
    "gpu_fault_gpu_metric_latest": {
        "fillfactor": "70",
        **_THRESHOLD_ONLY_AUTOVACUUM,
    },
    "gpu_fault_gpu_metrics_batches": {**_THRESHOLD_ONLY_AUTOVACUUM},
}

_OPTION_NAME = re.compile(r"^(?:toast\.)?[a-z_]+$")


def _stored_options(values: Any) -> dict[str, str]:
    current: dict[str, str] = {}
    for item in values if isinstance(values, (list, tuple)) else []:
        key, _, value = str(item).partition("=")
        current[key] = value
    return current


def set_storage_parameters_if_different(
    cursor: Any, table: str, options: dict[str, str]
) -> None:
    """``ALTER TABLE ... SET (option=value, ...)`` for the options whose stored
    value differs.

    ``ddl_helpers._set_table_options_if_different`` with one difference: the
    TOAST relation's reloptions are read too, so ``toast.*`` options compare
    against where Postgres stores them (the toast table's ``pg_class`` row)
    instead of re-issuing the ALTER on every ensure. Options are validated as
    option names and numerics because they are interpolated into DDL text.
    """

    for name, value in options.items():
        if _OPTION_NAME.fullmatch(name) is None or not _is_numeric(value):
            raise ValueError(f"unsafe table option: {name}={value}")
    cursor.execute(
        """
        SELECT c.reloptions, t.reloptions
        FROM pg_class c
        LEFT JOIN pg_class t ON t.oid = c.reltoastrelid
        WHERE c.oid = to_regclass(%s)
        """,
        (table,),
    )
    row = cursor.fetchone()
    if row is None:
        return
    current = _stored_options(row[0])
    toast_current = _stored_options(row[1])
    pending: dict[str, str] = {}
    for name, value in options.items():
        stored = (
            toast_current.get(name.removeprefix("toast."))
            if name.startswith("toast.")
            else current.get(name)
        )
        if stored != value:
            pending[name] = value
    if not pending:
        return
    rendered = ", ".join(f"{name}={value}" for name, value in sorted(pending.items()))
    cursor.execute(f"ALTER TABLE {table} SET ({rendered})")


def create_storage_parameters_stage(cursor: Any) -> None:
    """Schema v19: the hot tables' storage parameters (module docstring)."""

    for table, options in HOT_TABLE_STORAGE_PARAMETERS.items():
        set_storage_parameters_if_different(cursor, table, options)

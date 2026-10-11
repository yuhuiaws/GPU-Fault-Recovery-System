"""``/metrics`` families of the hot-table bloat scan and the compaction job.

The periodic ``store-compaction`` job (``periodic_services``) measures the
hot tables' physical layout from the catalog -- heap pages per live row,
index pages per live row, the least dense index's entries per page -- and
rebuilds what crosses the thresholds in ``store.postgres.hot_table_bloat``.
A scrape only reads the job's last answer; nothing here touches the store.
The gauges carry a ``table`` label so a bloated store is visible per table
in the acceptance record and on the capacity dashboard, and the scan clock
tells an alert whether the number is fresh.
"""

from __future__ import annotations


def _escape_label(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


TABLE_GAUGES: tuple[tuple[str, str, str], ...] = (
    (
        "gpu_fault_store_hot_table_live_rows",
        "live_rows",
        "Live rows of a hot PostgreSQL table at the last bloat scan (pg_stat_user_tables.n_live_tup).",
    ),
    (
        "gpu_fault_store_hot_table_heap_pages",
        "heap_pages",
        "Heap pages (8 KiB) a hot PostgreSQL table occupies at the last bloat scan.",
    ),
    (
        "gpu_fault_store_hot_table_index_pages",
        "index_pages",
        "Index pages (8 KiB) across every index of a hot PostgreSQL table at the last bloat scan.",
    ),
    (
        "gpu_fault_store_hot_table_heap_pages_per_live_row",
        "heap_pages_per_live_row",
        "Heap pages per live row of a hot table: a compact table sits far below 1, a heap left behind by a mass delete reads in the tens.",
    ),
    (
        "gpu_fault_store_hot_table_index_pages_per_live_row",
        "index_pages_per_live_row",
        "Index pages per live row of a hot table, summed over its indexes: the index bloat ratio the compaction policy acts on.",
    ),
    (
        "gpu_fault_store_hot_table_index_min_entries_per_page",
        "index_min_entries_per_page",
        "Entries per page of the least dense index of a hot table that is at least 64 pages; below 8 the compaction job rebuilds it.",
    ),
)


def store_bloat_metric_lines(snapshot: dict[str, object]) -> list[str]:
    """Render the bloat gauges and compaction counters from the runner snapshot."""

    tables = snapshot.get("store_bloat_tables")
    lines: list[str] = []
    for family, key, help_text in TABLE_GAUGES:
        lines.append(f"# HELP {family} {help_text}")
        lines.append(f"# TYPE {family} gauge")
        if not isinstance(tables, dict):
            continue
        for table, facts in sorted(tables.items()):
            value = facts.get(key) if isinstance(facts, dict) else None
            if isinstance(value, (int, float)):
                lines.append(
                    f'{family}{{table="{_escape_label(str(table))}"}} {float(value):.3f}'
                )
    scanned = snapshot.get("store_bloat_scan_timestamp_seconds", 0.0)
    compacted = snapshot.get("store_compaction_last_timestamp_seconds", 0.0)
    lines.extend(
        [
            "# HELP gpu_fault_store_bloat_scan_timestamp_seconds Unix time of this process's last successful hot-table bloat scan; 0 before one.",
            "# TYPE gpu_fault_store_bloat_scan_timestamp_seconds gauge",
            "gpu_fault_store_bloat_scan_timestamp_seconds "
            f"{scanned if isinstance(scanned, (int, float)) else 0.0:.3f}",
            "# HELP gpu_fault_store_compaction_last_timestamp_seconds Unix time this process last rebuilt a bloated hot-table index or rewrote a small hot table; 0 if never.",
            "# TYPE gpu_fault_store_compaction_last_timestamp_seconds gauge",
            "gpu_fault_store_compaction_last_timestamp_seconds "
            f"{compacted if isinstance(compacted, (int, float)) else 0.0:.3f}",
            "# HELP gpu_fault_store_compaction_actions_total Compaction actions the periodic job ran, by action and outcome.",
            "# TYPE gpu_fault_store_compaction_actions_total counter",
        ]
    )
    actions = snapshot.get("store_compaction_actions_total")
    if isinstance(actions, dict):
        for (action, status), count in sorted(actions.items()):
            lines.append(
                "gpu_fault_store_compaction_actions_total"
                f'{{action="{_escape_label(str(action))}",'
                f'status="{_escape_label(str(status))}"}} {int(count)}'
            )
    return lines

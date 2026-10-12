"""``/metrics`` families of the hot-table bloat scan and the compaction job.

The periodic ``store-compaction`` job (``periodic_services``) measures the
hot tables' physical layout from the catalog -- heap pages per live row,
index pages per live row, the least dense index's entries per page -- and
rebuilds what crosses the thresholds in ``store.postgres.hot_table_bloat``.
A scrape only reads the job's last answer; nothing here touches the store.
The gauges carry a ``table`` label so a bloated store is visible per table
in the acceptance record and on the capacity dashboard, and the scan clock
tells an alert whether the number is fresh.

Everything here is published by the one process that holds the job's task
lease, so on a six-Pod control worker exactly one Pod carries the series:
read the gauges with ``max by (table)`` over the Deployment and the counters
with ``increase()`` -- a roll replaces the holder and its counters, Prometheus
keeps the old series. A formal perf suite once read "never ran" out of the
other Pods' zeros: the job had rebuilt the queue's indexes four times, each
time on a holder the next roll replaced, and a round that found nothing due
left no line and moved no clock. Every round now stamps
``gpu_fault_store_compaction_last_timestamp_seconds`` and counts itself in
``gpu_fault_store_compaction_rounds_total{outcome}``; the rebuild clock lives
on in ``gpu_fault_store_compaction_last_action_timestamp_seconds``.

Two per-table gauges exist so the verdict can be checked against its inputs:
``index_pages_per_live_row`` is a reporting ratio that a near-empty table
drives into the tens (two live rows under 149 index pages read 74.5) and is
never a trigger. The trigger is each index's own density, judged only on
indexes of at least 64 pages, so a table whose largest index
(``index_max_pages``) sits below the floor has nothing due by construction,
and ``due_actions`` is what the policy derived for the table on this scan.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

#: Keys of ``store.postgres.hot_table_bloat.table_facts`` copied as they are.
_FACT_KEYS = (
    "live_rows",
    "heap_pages",
    "index_pages",
    "heap_pages_per_live_row",
    "index_pages_per_live_row",
    "index_min_entries_per_page",
)

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
        "Index pages per live row of a hot table, summed over its indexes; a reporting ratio that a near-empty table drives into the tens, never the compaction trigger.",
    ),
    (
        "gpu_fault_store_hot_table_index_min_entries_per_page",
        "index_min_entries_per_page",
        "Entries per page of the least dense index of a hot table that is at least 64 pages; below 8 the compaction job rebuilds that index. Absent while no index of the table reaches 64 pages.",
    ),
    (
        "gpu_fault_store_hot_table_index_max_pages",
        "index_max_pages",
        "Pages of the largest index of a hot table at the last bloat scan; below the 64-page floor the compaction rule does not apply to any index of the table.",
    ),
    (
        "gpu_fault_store_hot_table_due_actions",
        "due_actions",
        "Compaction actions (index rebuilds, small-table rewrites) the policy derived for a hot table at the last bloat scan; 0 is the verdict that nothing crossed a threshold.",
    ),
)

#: Snapshot keys of one table, in render order.
TABLE_GAUGE_KEYS: tuple[str, ...] = tuple(key for _family, key, _help in TABLE_GAUGES)

#: ``gpu_fault_store_compaction_rounds_total{outcome}`` values, one per way a
#: round can end: nothing due, rebuilt, lock held, an action failed.
ROUND_OUTCOMES: tuple[str, ...] = ("noop", "compacted", "skipped", "failed")


def _escape_label(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def table_gauge_facts(item: Mapping[str, Any], due_actions: int) -> dict[str, Any]:
    """The gauge values of one table from its ``table_facts`` record."""

    facts: dict[str, Any] = {key: item.get(key) for key in _FACT_KEYS}
    pages = [
        int(index.get("pages", 0))
        for index in item.get("indexes") or ()
        if isinstance(index, Mapping)
    ]
    facts["index_max_pages"] = max(pages, default=0)
    facts["due_actions"] = int(due_actions)
    return facts


def scan_gauge_tables(assessment: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Per-table gauge values of one ``hot_table_bloat`` assessment."""

    due: dict[str, int] = {}
    for action in assessment.get("actions") or ():
        table = str(action.get("table", ""))
        due[table] = due.get(table, 0) + 1
    return {
        str(item["table"]): table_gauge_facts(item, due.get(str(item["table"]), 0))
        for item in assessment.get("tables") or ()
    }


def _number(value: Any, digits: int = 3) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return "n/a"
    return f"{value:.{digits}f}".rstrip("0").rstrip(".") if digits else str(value)


def describe_scan(assessment: Mapping[str, Any]) -> str:
    """One line per scan: the verdict, then every table with the ratios it
    was judged on, so a log reader can check the policy without the store."""

    actions = list(assessment.get("actions") or ())
    tables = scan_gauge_tables(assessment)
    if actions:
        verdict = f"{len(actions)} action(s) due: " + ", ".join(
            f"{action['action']} {action['target']}" for action in actions
        )
    else:
        verdict = "nothing due"
    parts = []
    for table, facts in tables.items():
        parts.append(
            f"{table} live={facts['live_rows']} heap={facts['heap_pages']}p "
            f"index={facts['index_pages']}p(max {facts['index_max_pages']}p) "
            f"index_pages/live_row={_number(facts['index_pages_per_live_row'])} "
            f"min_entries/page={_number(facts['index_min_entries_per_page'])} "
            f"due={facts['due_actions']}"
        )
    missing = list(assessment.get("missing_tables") or ())
    if missing:
        parts.append("missing " + ",".join(str(table) for table in missing))
    return verdict + "; " + "; ".join(parts) if parts else verdict


def _clock(snapshot: Mapping[str, object], key: str) -> str:
    value = snapshot.get(key, 0.0)
    return f"{value if isinstance(value, (int, float)) else 0.0:.3f}"


def store_bloat_metric_lines(snapshot: Mapping[str, object]) -> list[str]:
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
    lines.extend(
        [
            "# HELP gpu_fault_store_bloat_scan_timestamp_seconds Unix time of this process's last successful hot-table bloat scan; 0 before one.",
            "# TYPE gpu_fault_store_bloat_scan_timestamp_seconds gauge",
            "gpu_fault_store_bloat_scan_timestamp_seconds "
            + _clock(snapshot, "store_bloat_scan_timestamp_seconds"),
            "# HELP gpu_fault_store_compaction_last_timestamp_seconds Unix time this process last completed a store-compaction round (a catalog scan plus whatever it rebuilt, or nothing); 0 before one.",
            "# TYPE gpu_fault_store_compaction_last_timestamp_seconds gauge",
            "gpu_fault_store_compaction_last_timestamp_seconds "
            + _clock(snapshot, "store_compaction_last_timestamp_seconds"),
            "# HELP gpu_fault_store_compaction_last_action_timestamp_seconds Unix time this process last rebuilt a bloated hot-table index or rewrote a small hot table; 0 if never.",
            "# TYPE gpu_fault_store_compaction_last_action_timestamp_seconds gauge",
            "gpu_fault_store_compaction_last_action_timestamp_seconds "
            + _clock(snapshot, "store_compaction_last_action_timestamp_seconds"),
            "# HELP gpu_fault_store_compaction_rounds_total Store-compaction rounds this process completed, by outcome: noop (nothing due), compacted, skipped (the schema lock was held), failed (an action failed).",
            "# TYPE gpu_fault_store_compaction_rounds_total counter",
        ]
    )
    rounds = snapshot.get("store_compaction_rounds_total")
    for outcome in ROUND_OUTCOMES:
        count = rounds.get(outcome, 0) if isinstance(rounds, dict) else 0
        lines.append(
            f'gpu_fault_store_compaction_rounds_total{{outcome="{outcome}"}} {int(count)}'
        )
    lines.extend(
        [
            "# HELP gpu_fault_store_compaction_actions_total Compaction actions the periodic job ran, by table, action and outcome.",
            "# TYPE gpu_fault_store_compaction_actions_total counter",
        ]
    )
    actions = snapshot.get("store_compaction_actions_total")
    if isinstance(actions, dict):
        for (table, action, status), count in sorted(actions.items()):
            lines.append(
                "gpu_fault_store_compaction_actions_total"
                f'{{table="{_escape_label(str(table))}",'
                f'action="{_escape_label(str(action))}",'
                f'status="{_escape_label(str(status))}"}} {int(count)}'
            )
    return lines

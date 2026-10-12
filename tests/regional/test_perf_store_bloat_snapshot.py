"""The perf runner records the store's physical layout in every round.

``postgres_counters`` is exec'd into an api-ha Pod as ``python3 -c``; its
template is substituted by ``str.replace`` because the body carries braces.
The summary's ``store_bloat`` block is derived from the before/after answers
with pure functions, so a bloated store is visible in the acceptance record
whether or not the control plane already exports the same ratios.
"""

from __future__ import annotations

from pathlib import Path

from gpu_fault.store.postgres import hot_table_bloat as policy
from gpu_fault.store.postgres.hot_table_bloat import HOT_TABLES
from tests._script_loader import lazy_script_module

ROOT = Path(__file__).resolve().parents[2]
capacity_database = lazy_script_module(
    ROOT / "scripts/perf/regional_capacity_database.py"
)


def _rendered_script() -> str:
    return capacity_database.POSTGRES_COUNTERS_SCRIPT.replace(
        "{STORE_DSN_SNIPPET}", capacity_database.STORE_DSN_SNIPPET
    ).replace("{HOT_TABLES}", repr(list(capacity_database.HOT_TABLES)))


def test_the_in_pod_script_compiles_with_its_placeholders_substituted() -> None:
    script = _rendered_script()
    compile(script, "postgres_counters", "exec")
    assert "{STORE_DSN_SNIPPET}" not in script
    assert "{HOT_TABLES}" not in script
    assert "def store_dsn" in script
    for table in HOT_TABLES:
        assert f"'{table}'" in script


def test_the_hot_table_list_is_the_products_own() -> None:
    assert tuple(capacity_database.HOT_TABLES) == HOT_TABLES


def test_the_index_rule_constants_are_the_products_own() -> None:
    assert capacity_database.INDEX_MIN_PAGES == policy.INDEX_MIN_PAGES
    assert (
        capacity_database.INDEX_MIN_ENTRIES_PER_PAGE
        == policy.INDEX_MIN_ENTRIES_PER_PAGE
    )


def test_postgres_counters_exec_substitutes_the_template(monkeypatch) -> None:
    seen: dict[str, object] = {}

    def control_command(*argv, **kwargs):
        seen["argv"] = argv
        return 'noise\n{"deadlocks": 0, "xact_commit": 1, "xact_rollback": 0, "tables": {}, "hot_tables": {}}\n'

    answer = capacity_database.postgres_counters(control_command, ["api-ha-0"])

    assert answer["hot_tables"] == {}
    argv = seen["argv"]
    assert argv[:5] == ("exec", "api-ha-0", "--", "python3", "-c")
    compile(argv[5], "postgres_counters", "exec")
    assert "{STORE_DSN_SNIPPET}" not in argv[5]


def test_ratios_are_physical_pages_per_live_row_and_a_density_floor() -> None:
    snapshot = {
        "hot_tables": {
            "gpu_fault_processor_queue": {
                "live_rows": 246,
                "heap_bytes": 1044 * 8192,
                "index_bytes": 4660 * 8192,
                "toast_bytes": 54059008,
                "indexes": {
                    "gpu_fault_processor_queue_available": {
                        "bytes": 1082 * 8192,
                        "entries": 246.0,
                    },
                    "gpu_fault_processor_queue_pkey": {
                        "bytes": 8 * 8192,  # below the 64-page floor: ignored
                        "entries": 0.0,
                    },
                    "gpu_fault_processor_queue_completed": {
                        "bytes": 72 * 8192,
                        "entries": -1.0,  # never measured: ignored
                    },
                },
            },
            "gpu_fault_processor_lanes": {
                "live_rows": 0,
                "heap_bytes": 5 * 8192,
                "index_bytes": 300 * 8192,
                "toast_bytes": 0,
                "indexes": {},
            },
        }
    }
    ratios = capacity_database.hot_table_bloat_ratios(snapshot)
    assert ratios["gpu_fault_processor_queue"] == {
        "live_rows": 246,
        "heap_pages": 1044,
        "index_pages": 4660,
        "toast_bytes": 54059008,
        "heap_pages_per_live_row": 4.244,
        "index_pages_per_live_row": 18.943,
        "index_min_entries_per_page": 0.227,
        "index_max_pages": 1082,
        "bloated_indexes": 1,
    }
    # Zero live rows divide by one: the pages themselves are the signal.
    assert ratios["gpu_fault_processor_lanes"]["index_pages_per_live_row"] == 300
    assert ratios["gpu_fault_processor_lanes"]["index_min_entries_per_page"] is None
    assert ratios["gpu_fault_processor_lanes"]["index_max_pages"] == 0
    assert ratios["gpu_fault_processor_lanes"]["bloated_indexes"] == 0


def _queue_snapshot(live_rows: int, pages_and_entries: dict[str, tuple[int, float]]):
    return {
        "hot_tables": {
            "gpu_fault_processor_queue": {
                "live_rows": live_rows,
                "heap_bytes": 5 * 8192,
                "index_bytes": sum(p for p, _e in pages_and_entries.values()) * 8192,
                "toast_bytes": 0,
                "indexes": {
                    name: {"bytes": pages * 8192, "entries": entries}
                    for name, (pages, entries) in pages_and_entries.items()
                },
            }
        }
    }


def test_pages_per_live_row_is_not_the_verdict_the_floor_and_density_are() -> None:
    """The two queue layouts a formal suite recorded: two live rows under 149
    index pages read 74.5 pages per live row while every index sat below the
    64-page floor (nothing due, and the job correctly did nothing); one live
    row under three indexes of 99/158/82 pages is three rebuilds due, which
    the job then ran -- the next preflight found exactly those three at two
    pages and the three below the floor untouched."""

    idle = capacity_database.hot_table_bloat_ratios(
        _queue_snapshot(
            2,
            {
                "claim_window": (2, 3.0),
                "cluster_rows": (55, 3.0),
                "lane_rows": (55, 3.0),
                "path_claim_window": (2, 3.0),
                "pkey": (2, 3.0),
                "scope_keys": (27, 0.0),
            },
        )
    )["gpu_fault_processor_queue"]
    assert idle["index_pages"] == 143
    assert idle["index_pages_per_live_row"] == 71.5
    assert idle["index_min_entries_per_page"] is None
    assert idle["index_max_pages"] == 55
    assert idle["bloated_indexes"] == 0

    bloated = capacity_database.hot_table_bloat_ratios(
        _queue_snapshot(
            1,
            {
                "claim_window": (99, 1.0),
                "cluster_rows": (55, 1.0),
                "lane_rows": (55, 1.0),
                "path_claim_window": (158, 1.0),
                "pkey": (82, 1.0),
                "scope_keys": (27, 0.0),
            },
        )
    )["gpu_fault_processor_queue"]
    assert bloated["index_max_pages"] == 158
    assert bloated["index_min_entries_per_page"] == 0.006
    assert bloated["bloated_indexes"] == 3


def test_the_summary_section_records_the_preflight_worst_case_or_its_absence() -> None:
    before = {
        "hot_tables": {
            "gpu_fault_gpu_metric_latest": {
                "live_rows": 1233,
                "heap_bytes": 31023 * 8192,
                "index_bytes": 14459 * 8192,
                "toast_bytes": 8192,
                "indexes": {},
            },
            "gpu_fault_objects": {
                "live_rows": 690,
                "heap_bytes": 6881 * 8192,
                "index_bytes": 3924 * 8192,
                "toast_bytes": 278528,
                "indexes": {},
            },
        }
    }
    section = capacity_database.store_bloat_section(before, {"error": "exec failed"})
    assert section["recorded"] is True
    assert section["preflight_max_heap_pages_per_live_row"] == 25.161
    assert section["preflight_max_index_pages_per_live_row"] == 11.727
    assert section["preflight_bloated_indexes"] == 0
    assert section["after_bloated_indexes"] is None
    assert section["after"] == {}

    absent = capacity_database.store_bloat_section({}, {})
    assert absent == {
        "before": {},
        "after": {},
        "preflight_max_heap_pages_per_live_row": None,
        "preflight_max_index_pages_per_live_row": None,
        "preflight_bloated_indexes": None,
        "after_bloated_indexes": None,
        "recorded": False,
    }

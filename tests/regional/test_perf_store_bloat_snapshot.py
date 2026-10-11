"""The perf runner records the store's physical layout in every round.

``postgres_counters`` is exec'd into an api-ha Pod as ``python3 -c``; its
template is substituted by ``str.replace`` because the body carries braces.
The summary's ``store_bloat`` block is derived from the before/after answers
with pure functions, so a bloated store is visible in the acceptance record
whether or not the control plane already exports the same ratios.
"""

from __future__ import annotations

from pathlib import Path

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
    }
    # Zero live rows divide by one: the pages themselves are the signal.
    assert ratios["gpu_fault_processor_lanes"]["index_pages_per_live_row"] == 300
    assert ratios["gpu_fault_processor_lanes"]["index_min_entries_per_page"] is None


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
    assert section["after"] == {}

    absent = capacity_database.store_bloat_section({}, {})
    assert absent == {
        "before": {},
        "after": {},
        "preflight_max_heap_pages_per_live_row": None,
        "preflight_max_index_pages_per_live_row": None,
        "recorded": False,
    }

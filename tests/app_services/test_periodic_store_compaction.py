"""The store-compaction periodic job: scheduled, lease-guarded, measured.

The job reads the hot tables' physical layout through the store, publishes
it for ``/metrics`` and rebuilds what the policy calls for. Here the store is
a double: what matters is the schedule, that only a Postgres store (one with
the two methods) is asked, that a scan without due actions touches nothing,
that a run's outcome is counted per action, and that the gauges render with
their scan clock.
"""

from __future__ import annotations

from dataclasses import replace
from threading import Event
from types import SimpleNamespace

import pytest

from gpu_fault.app import periodic_services
from gpu_fault.app.builtin_metric_contributors import control_loop_metric_lines
from gpu_fault.app.metric_aggregation import STRATEGIES, Strategy
from gpu_fault.app.periodic_services import PeriodicServiceConfig, PeriodicServiceRunner
from gpu_fault.app.process_metrics import parse_lines
from gpu_fault.app.store_bloat_metrics import TABLE_GAUGES
from gpu_fault.telemetry import CollectorKind

QUEUE = "gpu_fault_processor_queue"
REINDEX = "REINDEX INDEX CONCURRENTLY"


def _config(**overrides) -> PeriodicServiceConfig:
    base = PeriodicServiceConfig(
        training_interval=15.0,
        spare_interval=30.0,
        identity_interval=20.0,
        cleanup_interval=60.0,
        completed_retention=600.0,
        cleanup_batch_size=10,
        cleanup_budget_seconds=0.2,
        batch_retention=86400.0,
        terminal_retention=2592000.0,
        latest_retention=2592000.0,
        finding_retention=2592000.0,
        observation_max_age=604800.0,
        remote_retention=86400.0,
        remote_claim_deadline=900.0,
        lane_retention=3600.0,
        deployment_retention=604800.0,
        archive_interval=3600.0,
        silence_interval=60.0,
        silent_after={kind: 300.0 for kind in CollectorKind},
        silent_alert_interval=3600.0,
    )
    return replace(base, **overrides)


def _facts(table: str, *, live: int, heap: int, index: int, density) -> dict:
    return {
        "table": table,
        "live_rows": live,
        "heap_pages": heap,
        "index_pages": index,
        "heap_pages_per_live_row": round(heap / max(live, 1), 3),
        "index_pages_per_live_row": round(index / max(live, 1), 3),
        "index_min_entries_per_page": density,
        "indexes": [],
    }


class _QuietJobs:
    """The other periodic jobs' store calls, answered with nothing to do."""

    def __getattr__(self, name: str):
        if name.startswith("cleanup_") or name == "reclaim_expired_processor_leases":
            return lambda *args, **kwargs: 0
        if name == "processor_queue_count_status":
            return lambda: {
                "expected_total": 0,
                "counter_total": 0,
                "mismatched_clusters": 0,
            }
        raise AttributeError(name)


class _BloatStore(_QuietJobs):
    """A Postgres-shaped store: measurable and compactable."""

    def __init__(self, actions: list[dict] | None = None) -> None:
        self.actions = actions or []
        self.scans = 0
        self.compactions: list[dict] = []
        self.compaction_status = "SUCCEEDED"
        self.fail_scan = False

    def acquire_periodic_task_lease(self, key, owner, *, now, lease_duration):
        return SimpleNamespace(owner_id=owner, lease_expires_at=now + lease_duration)

    def hot_table_bloat(self) -> dict:
        self.scans += 1
        if self.fail_scan:
            raise RuntimeError("synthetic catalog failure")
        return {
            "tables": [
                _facts(QUEUE, live=246, heap=1044, index=4660, density=0.227),
                _facts(
                    "gpu_fault_gpu_metric_latest",
                    live=1233,
                    heap=31023,
                    index=14459,
                    density=0.1,
                ),
            ],
            "missing_tables": [],
            "actions": list(self.actions),
        }

    def compact_hot_tables(self, assessment=None) -> dict:
        self.compactions.append(assessment)
        return {
            "status": self.compaction_status,
            "actions": [
                {**item, "status": self.compaction_status, "seconds": 0.5}
                for item in (assessment or {}).get("actions", [])
            ],
            "after": {},
        }


class _PlainStore(_QuietJobs):
    """The memory/SQLite shape: a task lease, no bloat methods."""

    def acquire_periodic_task_lease(self, key, owner, *, now, lease_duration):
        return SimpleNamespace(owner_id=owner, lease_expires_at=now + lease_duration)


def _runner(store, **config) -> PeriodicServiceRunner:
    return PeriodicServiceRunner(
        context=SimpleNamespace(
            store=store,
            regional_mode=False,
            spare_health_controller=None,
            control_record_archiver=None,
        ),
        processor=SimpleNamespace(
            is_healthy=lambda: True,
            active_consumers=True,
            is_leader=lambda: True,
            owner_id="pod-a:1",
        ),
        stop=Event(),
        identity_registries=[],
        ingest_node_health_findings=lambda *args, **kwargs: None,
        notify_silent_collectors=lambda *args, **kwargs: None,
        config=_config(**config),
    )


# --- configuration -----------------------------------------------------------


def test_the_interval_defaults_to_five_minutes_and_reads_the_environment(
    monkeypatch,
) -> None:
    monkeypatch.delenv("GPU_FAULT_STORE_COMPACTION_INTERVAL_SECONDS", raising=False)
    assert PeriodicServiceConfig.from_environment().store_compaction_interval == 300.0
    monkeypatch.setenv("GPU_FAULT_STORE_COMPACTION_INTERVAL_SECONDS", "45")
    assert PeriodicServiceConfig.from_environment().store_compaction_interval == 45.0


def test_zero_switches_the_job_off_and_a_negative_interval_is_refused(
    monkeypatch,
) -> None:
    monkeypatch.setenv("GPU_FAULT_STORE_COMPACTION_INTERVAL_SECONDS", "0")
    assert PeriodicServiceConfig.from_environment().store_compaction_interval == 0.0
    monkeypatch.setenv("GPU_FAULT_STORE_COMPACTION_INTERVAL_SECONDS", "-1")
    with pytest.raises(ValueError, match="GPU_FAULT_STORE_COMPACTION_INTERVAL_SECONDS"):
        PeriodicServiceConfig.from_environment()


# --- the job -----------------------------------------------------------------


def test_a_scan_without_due_actions_publishes_the_layout_and_compacts_nothing(
    monkeypatch,
) -> None:
    clock = [100.0]
    monkeypatch.setattr(
        periodic_services,
        "time",
        SimpleNamespace(monotonic=lambda: clock[0], time=lambda: 5000 + clock[0]),
    )
    store = _BloatStore()
    worker = _runner(store)

    worker.run_all_due(clock[0])

    assert store.scans == 1
    assert store.compactions == []
    snapshot = worker.metrics_snapshot()
    assert snapshot["store_bloat_tables"][QUEUE] == {
        "live_rows": 246,
        "heap_pages": 1044,
        "index_pages": 4660,
        "heap_pages_per_live_row": 4.244,
        "index_pages_per_live_row": 18.943,
        "index_min_entries_per_page": 0.227,
    }
    assert snapshot["store_bloat_scan_timestamp_seconds"] == 5100
    assert snapshot["store_compaction_last_timestamp_seconds"] == 0.0
    assert snapshot["store_compaction_actions_total"] == {}
    assert worker.job_last_run_timestamp_seconds["store_compaction"] == 5100
    # Not due again before the interval elapsed.
    clock[0] = 100.0 + 299.0
    worker.run_all_due(clock[0])
    assert store.scans == 1
    clock[0] = 100.0 + 301.0
    worker.run_all_due(clock[0])
    assert store.scans == 2


def test_due_actions_are_compacted_and_counted_per_action_and_outcome(
    monkeypatch,
) -> None:
    clock = [0.0]
    monkeypatch.setattr(
        periodic_services,
        "time",
        SimpleNamespace(monotonic=lambda: clock[0], time=lambda: 7000 + clock[0]),
    )
    actions = [
        {"table": QUEUE, "action": REINDEX, "target": "gpu_fault_processor_queue_lane"},
        {"table": QUEUE, "action": REINDEX, "target": "gpu_fault_processor_queue_pkey"},
        {
            "table": "gpu_fault_processor_queue_counts",
            "action": "VACUUM (FULL, ANALYZE)",
            "target": "gpu_fault_processor_queue_counts",
        },
    ]
    store = _BloatStore(actions)
    worker = _runner(store)

    worker.run_all_due(clock[0])

    assert worker.job_last_run_timestamp_seconds["store_compaction"] == 7000
    assert len(store.compactions) == 1
    assert store.compactions[0]["actions"] == actions
    snapshot = worker.metrics_snapshot()
    assert snapshot["store_compaction_actions_total"] == {
        (REINDEX, "SUCCEEDED"): 2,
        ("VACUUM (FULL, ANALYZE)", "SUCCEEDED"): 1,
    }
    assert snapshot["store_compaction_last_timestamp_seconds"] == 7000
    assert "store_compaction" not in worker.periodic_job_errors_total


def test_a_failed_action_counts_as_a_job_error_but_the_runner_goes_on(
    monkeypatch,
) -> None:
    clock = [0.0]
    monkeypatch.setattr(
        periodic_services,
        "time",
        SimpleNamespace(monotonic=lambda: clock[0], time=lambda: 9000 + clock[0]),
    )
    store = _BloatStore(
        [
            {
                "table": QUEUE,
                "action": REINDEX,
                "target": "gpu_fault_processor_queue_lane",
            }
        ]
    )
    store.compaction_status = "FAILED"
    worker = _runner(store)

    worker.run_all_due(clock[0])

    snapshot = worker.metrics_snapshot()
    assert snapshot["store_compaction_actions_total"] == {(REINDEX, "FAILED"): 1}
    assert worker.periodic_job_errors_total["store_compaction"] == 1
    assert "store_compaction" not in worker.job_last_run_timestamp_seconds
    # The scan itself succeeded, so the layout is still published.
    assert snapshot["store_bloat_scan_timestamp_seconds"] == 9000


def test_a_failed_scan_keeps_the_previous_layout_and_clock(monkeypatch) -> None:
    clock = [0.0]
    monkeypatch.setattr(
        periodic_services,
        "time",
        SimpleNamespace(monotonic=lambda: clock[0], time=lambda: 1000 + clock[0]),
    )
    store = _BloatStore()
    worker = _runner(store)
    worker.run_all_due(clock[0])
    store.fail_scan = True
    clock[0] = 400.0
    worker.run_all_due(clock[0])
    snapshot = worker.metrics_snapshot()
    assert snapshot["store_bloat_scan_timestamp_seconds"] == 1000
    assert QUEUE in snapshot["store_bloat_tables"]
    assert worker.periodic_job_errors_total["store_compaction"] == 1


def test_a_store_without_the_policy_is_never_asked_and_zero_disables() -> None:
    plain = _runner(_PlainStore())
    plain.run_all_due(0.0)
    assert "store_compaction" not in plain.job_last_run_timestamp_seconds
    assert "store_compaction" not in plain.periodic_job_errors_total
    store = _BloatStore()
    disabled = _runner(store, store_compaction_interval=0.0)
    disabled.run_all_due(0.0)
    assert store.scans == 0
    assert "store_compaction" not in disabled.job_last_run_timestamp_seconds


# --- /metrics ----------------------------------------------------------------


def test_the_gauges_render_per_table_with_the_scan_clock(monkeypatch) -> None:
    clock = [0.0]
    monkeypatch.setattr(
        periodic_services,
        "time",
        SimpleNamespace(monotonic=lambda: clock[0], time=lambda: 2000 + clock[0]),
    )
    store = _BloatStore(
        [
            {
                "table": QUEUE,
                "action": REINDEX,
                "target": "gpu_fault_processor_queue_lane",
            }
        ]
    )
    worker = _runner(store)
    worker.run_all_due(clock[0])

    parsed = parse_lines(
        control_loop_metric_lines(
            SimpleNamespace(context=SimpleNamespace(periodic_runner=worker))
        )
    )

    for family, key, _help in TABLE_GAUGES:
        samples = {
            dict(sample.labels)["table"]: float(sample.value)
            for sample in parsed.samples[family]
        }
        assert samples[QUEUE] == pytest.approx(
            worker.metrics_snapshot()["store_bloat_tables"][QUEUE][key]
        ), family
        assert set(samples) == {QUEUE, "gpu_fault_gpu_metric_latest"}
    assert [
        float(s.value)
        for s in parsed.samples["gpu_fault_store_bloat_scan_timestamp_seconds"]
    ] == [2000]
    assert [
        float(s.value)
        for s in parsed.samples["gpu_fault_store_compaction_last_timestamp_seconds"]
    ] == [2000]
    counters = {
        (dict(s.labels)["action"], dict(s.labels)["status"]): float(s.value)
        for s in parsed.samples["gpu_fault_store_compaction_actions_total"]
    }
    assert counters == {(REINDEX, "SUCCEEDED"): 1.0}


def test_a_scrape_before_any_scan_renders_the_families_without_table_samples() -> None:
    worker = _runner(_PlainStore())
    parsed = parse_lines(
        control_loop_metric_lines(
            SimpleNamespace(context=SimpleNamespace(periodic_runner=worker))
        )
    )
    for family, _key, _help in TABLE_GAUGES:
        assert family in parsed.families
        assert parsed.samples[family] == []
    assert [
        float(s.value)
        for s in parsed.samples["gpu_fault_store_bloat_scan_timestamp_seconds"]
    ] == [0.0]


def test_every_bloat_family_aggregates_per_process_like_the_drift_scan() -> None:
    families = [family for family, _key, _help in TABLE_GAUGES] + [
        "gpu_fault_store_bloat_scan_timestamp_seconds",
        "gpu_fault_store_compaction_last_timestamp_seconds",
        "gpu_fault_store_compaction_actions_total",
    ]
    for family in families:
        assert STRATEGIES[family] is Strategy.PER_PROCESS, family

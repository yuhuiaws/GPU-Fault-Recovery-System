"""The store-compaction periodic job: hosted, scheduled, lease-guarded, measured.

The job reads the hot tables' physical layout through the store, publishes
it for ``/metrics``, logs the verdict it reached and rebuilds what the policy
calls for. Here the store is a double: what matters is where the job is
hosted (the background-service roles' processor threads, with the shipped
default), that only a Postgres store (one with the two methods) is asked,
that every round -- including one with nothing due -- stamps its clock,
counts itself and logs the ratios it was judged on, that a run's outcome is
counted per table and action, and that the gauges render with their scan
clock before the first action.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import replace
from threading import Event
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from gpu_fault.app import ApplicationContext, create_app, periodic_services
from gpu_fault.app.builtin_metric_contributors import control_loop_metric_lines
from gpu_fault.app.lifespan_workers import start_processor_threads
from gpu_fault.app.metric_aggregation import STRATEGIES, Strategy
from gpu_fault.app.periodic_services import PeriodicServiceConfig, PeriodicServiceRunner
from gpu_fault.app.process_metrics import parse_lines
from gpu_fault.app.store_bloat_metrics import ROUND_OUTCOMES, TABLE_GAUGES
from gpu_fault.telemetry import CollectorKind
from tests._builders import build_store

QUEUE = "gpu_fault_processor_queue"
LATEST = "gpu_fault_gpu_metric_latest"
REINDEX = "REINDEX INDEX CONCURRENTLY"
VACUUM_FULL = "VACUUM (FULL, ANALYZE)"
WAIT = 5.0


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


def _facts(table: str, *, live: int, heap: int, index: int, density, indexes) -> dict:
    """A ``table_facts`` record; ``indexes`` are ``(name, pages, bloated)``."""

    return {
        "table": table,
        "live_rows": live,
        "heap_pages": heap,
        "index_pages": index,
        "heap_pages_per_live_row": round(heap / max(live, 1), 3),
        "index_pages_per_live_row": round(index / max(live, 1), 3),
        "index_min_entries_per_page": density,
        "indexes": [
            {"name": name, "pages": pages, "bloated": bloated, "am": "btree"}
            for name, pages, bloated in indexes
        ],
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
                _facts(
                    QUEUE,
                    live=246,
                    heap=1044,
                    index=4660,
                    density=0.227,
                    indexes=[
                        ("gpu_fault_processor_queue_lane", 1082, True),
                        ("gpu_fault_processor_queue_pkey", 8, False),
                    ],
                ),
                _facts(
                    LATEST,
                    live=1233,
                    heap=31023,
                    index=14459,
                    density=0.1,
                    indexes=[("gpu_fault_gpu_metric_latest_pkey", 7229, True)],
                ),
            ],
            "missing_tables": [],
            "actions": list(self.actions),
        }

    def compact_hot_tables(self, assessment=None) -> dict:
        self.compactions.append(assessment)
        if self.compaction_status == "SKIPPED":
            return {
                "status": "SKIPPED",
                "reason": "a schema ensure or store maintenance holds the lock",
                "actions": [],
                "after": {},
            }
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


def _processor(**overrides) -> SimpleNamespace:
    return SimpleNamespace(
        is_healthy=lambda: True,
        active_consumers=True,
        is_leader=lambda: True,
        owner_id="pod-a:1",
        **overrides,
    )


def _runner(store, **config) -> PeriodicServiceRunner:
    return PeriodicServiceRunner(
        context=SimpleNamespace(
            store=store,
            regional_mode=False,
            spare_health_controller=None,
            control_record_archiver=None,
        ),
        processor=_processor(),
        stop=Event(),
        identity_registries=[],
        ingest_node_health_findings=lambda *args, **kwargs: None,
        notify_silent_collectors=lambda *args, **kwargs: None,
        config=_config(**config),
    )


def _fake_clock(monkeypatch, clock: list[float], epoch: float) -> None:
    monkeypatch.setattr(
        periodic_services,
        "time",
        SimpleNamespace(monotonic=lambda: clock[0], time=lambda: epoch + clock[0]),
    )


def _render(worker: PeriodicServiceRunner):
    return parse_lines(
        control_loop_metric_lines(
            SimpleNamespace(context=SimpleNamespace(periodic_runner=worker))
        )
    )


def _actions() -> list[dict]:
    return [
        {"table": QUEUE, "action": REINDEX, "target": "gpu_fault_processor_queue_lane"},
        {"table": QUEUE, "action": REINDEX, "target": "gpu_fault_processor_queue_pkey"},
        {
            "table": "gpu_fault_processor_queue_counts",
            "action": VACUUM_FULL,
            "target": "gpu_fault_processor_queue_counts",
        },
    ]


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


# --- hosting -----------------------------------------------------------------


def test_the_worker_processor_threads_host_the_job_with_the_shipped_default(
    monkeypatch,
) -> None:
    """``start_processor_threads`` -- the control worker's thread set -- owns
    the periodic runner, and the runner scans through the store with no
    environment variable set at all: the rendered ConfigMaps carry none."""

    monkeypatch.delenv("GPU_FAULT_STORE_COMPACTION_INTERVAL_SECONDS", raising=False)
    store = _BloatStore()
    stop = Event()
    context = SimpleNamespace(
        store=store,
        regional_mode=False,
        spare_health_controller=None,
        control_record_archiver=None,
        dispatcher=SimpleNamespace(config=SimpleNamespace(enabled=False)),
        xid_correlation=SimpleNamespace(
            run_once=lambda: None, poll_interval_seconds=60.0
        ),
    )
    processor = _processor(run_processor=stop.wait, telemetry_spool_enabled=False)

    diagnostics, threads = start_processor_threads(
        context=context,
        processor=processor,
        diagnostics_publisher=SimpleNamespace(run=lambda stop_event: stop_event.wait()),
        stop=stop,
        identity_registries=[],
        ingest_node_health_findings=lambda *args, **kwargs: None,
        notify_silent_collectors=lambda *args, **kwargs: None,
    )
    try:
        runner = context.periodic_runner
        assert isinstance(runner, PeriodicServiceRunner), runner
        assert runner.config.store_compaction_interval == 300.0
        assert "gpu-fault-periodic-services" in {thread.name for thread in threads}
        deadline = time.monotonic() + WAIT
        while store.scans == 0 and time.monotonic() < deadline:
            time.sleep(0.01)
        assert store.scans >= 1, "the hosted runner never asked the store"
        assert runner.metrics_snapshot()["store_compaction_rounds_total"]["noop"] >= 1
        assert runner.job_last_run_timestamp_seconds["store_compaction"] > 0
    finally:
        stop.set()
        for thread in (diagnostics, *threads):
            thread.join(WAIT)
    assert not any(thread.is_alive() for thread in (diagnostics, *threads)), [
        thread.name for thread in (diagnostics, *threads) if thread.is_alive()
    ]


def _hosted_runner(monkeypatch, role: str):
    monkeypatch.setenv("GPU_FAULT_SERVICE_ROLE", role)
    monkeypatch.setenv("GPU_FAULT_PROCESSOR_WORKERS", "4")
    monkeypatch.setenv("GPU_FAULT_PROCESSOR_MODE", "active-active")
    monkeypatch.setenv("POD_UID", f"pod-store-compaction-hosting-{role}")
    monkeypatch.delenv("GPU_FAULT_STORE_COMPACTION_INTERVAL_SECONDS", raising=False)
    context = ApplicationContext(
        store=build_store(),
        execution_token="store-compaction-hosting-token-" + "x" * 32,
    )
    # Starlette's TestClient runs the lifespan; the lifespan is what starts
    # the role's threads and attaches the runner to the context.
    with TestClient(create_app(context)) as client:
        response = client.get("/healthz")
        assert response.status_code == 200, response.text
        return getattr(context, "periodic_runner", None)


def test_only_the_background_service_roles_host_the_periodic_runner(
    monkeypatch,
) -> None:
    """The control worker (``GPU_FAULT_SERVICE_ROLE=worker``) hosts the
    runner; the ingress Pods never do, so the job's gauges and counters are a
    control-worker fact and an api-ha scrape is the wrong place to look."""

    worker = _hosted_runner(monkeypatch, "worker")
    assert isinstance(worker, PeriodicServiceRunner), worker
    assert worker.config.store_compaction_interval == 300.0
    assert _hosted_runner(monkeypatch, "ingress") is None


# --- the job -----------------------------------------------------------------


def test_a_scan_without_due_actions_publishes_the_layout_and_stamps_the_round(
    monkeypatch,
) -> None:
    clock = [100.0]
    _fake_clock(monkeypatch, clock, 5000)
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
        "index_max_pages": 1082,
        "due_actions": 0,
    }
    assert snapshot["store_bloat_scan_timestamp_seconds"] == 5100
    # The round clock moves on every completed round, the rebuild clock only
    # on a rebuild; the round is counted by its outcome.
    assert snapshot["store_compaction_last_timestamp_seconds"] == 5100
    assert snapshot["store_compaction_last_action_timestamp_seconds"] == 0.0
    assert snapshot["store_compaction_rounds_total"] == {"noop": 1}
    assert snapshot["store_compaction_actions_total"] == {}
    assert worker.job_last_run_timestamp_seconds["store_compaction"] == 5100
    # Not due again before the interval elapsed.
    clock[0] = 100.0 + 299.0
    worker.run_all_due(clock[0])
    assert store.scans == 1
    clock[0] = 100.0 + 301.0
    worker.run_all_due(clock[0])
    assert store.scans == 2
    assert worker.metrics_snapshot()["store_compaction_rounds_total"] == {"noop": 2}


def test_every_scan_logs_its_verdict_with_the_ratios_it_was_judged_on(
    monkeypatch, caplog
) -> None:
    """A round that rebuilds nothing is still a decision: one INFO line
    carrying the job's name, the verdict and per table the live rows, pages,
    largest index, pages per live row and least density it saw."""

    _fake_clock(monkeypatch, [0.0], 5000)
    worker = _runner(_BloatStore())
    with caplog.at_level(logging.INFO, logger="gpu_fault.app.periodic_services"):
        worker.run_all_due(0.0)

    scans = [
        record
        for record in caplog.records
        if record.getMessage().startswith("store-compaction scan: ")
    ]
    assert len(scans) == 1
    assert scans[0].levelno == logging.INFO
    message = scans[0].getMessage()
    assert message.startswith("store-compaction scan: nothing due; "), message
    assert (
        f"{QUEUE} live=246 heap=1044p index=4660p(max 1082p) "
        "index_pages/live_row=18.943 min_entries/page=0.227 due=0"
    ) in message
    assert (
        f"{LATEST} live=1233 heap=31023p index=14459p(max 7229p) "
        "index_pages/live_row=11.727 min_entries/page=0.1 due=0"
    ) in message


def test_due_actions_are_compacted_counted_per_table_and_logged(
    monkeypatch, caplog
) -> None:
    clock = [0.0]
    _fake_clock(monkeypatch, clock, 7000)
    actions = _actions()
    store = _BloatStore(actions)
    worker = _runner(store)

    with caplog.at_level(logging.INFO, logger="gpu_fault.app.periodic_services"):
        worker.run_all_due(clock[0])

    assert worker.job_last_run_timestamp_seconds["store_compaction"] == 7000
    assert len(store.compactions) == 1
    assert store.compactions[0]["actions"] == actions
    snapshot = worker.metrics_snapshot()
    assert snapshot["store_compaction_actions_total"] == {
        (QUEUE, REINDEX, "SUCCEEDED"): 2,
        ("gpu_fault_processor_queue_counts", VACUUM_FULL, "SUCCEEDED"): 1,
    }
    assert snapshot["store_compaction_last_timestamp_seconds"] == 7000
    assert snapshot["store_compaction_last_action_timestamp_seconds"] == 7000
    assert snapshot["store_compaction_rounds_total"] == {"compacted": 1}
    assert snapshot["store_bloat_tables"][QUEUE]["due_actions"] == 2
    assert "store_compaction" not in worker.periodic_job_errors_total
    messages = [record.getMessage() for record in caplog.records]
    assert any(
        message.startswith(
            "store-compaction scan: 3 action(s) due: "
            f"{REINDEX} gpu_fault_processor_queue_lane, "
        )
        for message in messages
    ), messages
    assert (
        f"store-compaction {REINDEX} gpu_fault_processor_queue_lane ({QUEUE}) "
        "SUCCEEDED in 0.5s"
    ) in messages


def test_a_held_schema_lock_skips_the_round_and_says_so(monkeypatch, caplog) -> None:
    clock = [0.0]
    _fake_clock(monkeypatch, clock, 8000)
    store = _BloatStore(_actions())
    store.compaction_status = "SKIPPED"
    worker = _runner(store)

    with caplog.at_level(logging.INFO, logger="gpu_fault.app.periodic_services"):
        worker.run_all_due(clock[0])

    snapshot = worker.metrics_snapshot()
    assert snapshot["store_compaction_rounds_total"] == {"skipped": 1}
    assert snapshot["store_compaction_last_timestamp_seconds"] == 8000
    assert snapshot["store_compaction_last_action_timestamp_seconds"] == 0.0
    assert snapshot["store_compaction_actions_total"] == {}
    assert worker.job_last_run_timestamp_seconds["store_compaction"] == 8000
    assert "store_compaction" not in worker.periodic_job_errors_total
    assert (
        "store-compaction skipped: a schema ensure or store maintenance holds the lock"
    ) in [record.getMessage() for record in caplog.records]


def test_a_failed_action_counts_as_a_job_error_but_the_runner_goes_on(
    monkeypatch,
) -> None:
    clock = [0.0]
    _fake_clock(monkeypatch, clock, 9000)
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
    assert snapshot["store_compaction_actions_total"] == {(QUEUE, REINDEX, "FAILED"): 1}
    assert snapshot["store_compaction_rounds_total"] == {"failed": 1}
    assert snapshot["store_compaction_last_timestamp_seconds"] == 9000
    assert snapshot["store_compaction_last_action_timestamp_seconds"] == 9000
    assert worker.periodic_job_errors_total["store_compaction"] == 1
    assert "store_compaction" not in worker.job_last_run_timestamp_seconds
    # The scan itself succeeded, so the layout is still published.
    assert snapshot["store_bloat_scan_timestamp_seconds"] == 9000


def test_a_failed_scan_keeps_the_previous_layout_and_clocks(monkeypatch) -> None:
    clock = [0.0]
    _fake_clock(monkeypatch, clock, 1000)
    store = _BloatStore()
    worker = _runner(store)
    worker.run_all_due(clock[0])
    store.fail_scan = True
    clock[0] = 400.0
    worker.run_all_due(clock[0])
    snapshot = worker.metrics_snapshot()
    assert snapshot["store_bloat_scan_timestamp_seconds"] == 1000
    assert QUEUE in snapshot["store_bloat_tables"]
    # A scan that raised is not a completed round: the round clock and the
    # outcome counters stay, the job error is what moves.
    assert snapshot["store_compaction_last_timestamp_seconds"] == 1000
    assert snapshot["store_compaction_rounds_total"] == {"noop": 1}
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


def test_the_scan_gauges_render_before_any_action_with_the_round_clock(
    monkeypatch,
) -> None:
    """The first round of a compact store already publishes every table
    gauge -- the floor and the verdict included -- the round clock and the
    outcome counter, with the rebuild clock still at zero."""

    _fake_clock(monkeypatch, [0.0], 3000)
    worker = _runner(_BloatStore())
    worker.run_all_due(0.0)

    parsed = _render(worker)

    for family, key, _help in TABLE_GAUGES:
        samples = {
            dict(sample.labels)["table"]: float(sample.value)
            for sample in parsed.samples[family]
        }
        assert set(samples) == {QUEUE, LATEST}, family
        assert samples[QUEUE] == pytest.approx(
            worker.metrics_snapshot()["store_bloat_tables"][QUEUE][key]
        ), family
    gauges = {family for family, _key, _help in TABLE_GAUGES}
    assert "gpu_fault_store_hot_table_index_max_pages" in gauges
    assert "gpu_fault_store_hot_table_due_actions" in gauges
    assert [
        float(s.value)
        for s in parsed.samples["gpu_fault_store_bloat_scan_timestamp_seconds"]
    ] == [3000]
    assert [
        float(s.value)
        for s in parsed.samples["gpu_fault_store_compaction_last_timestamp_seconds"]
    ] == [3000]
    assert [
        float(s.value)
        for s in parsed.samples[
            "gpu_fault_store_compaction_last_action_timestamp_seconds"
        ]
    ] == [0.0]
    rounds = {
        dict(s.labels)["outcome"]: float(s.value)
        for s in parsed.samples["gpu_fault_store_compaction_rounds_total"]
    }
    assert rounds == {"noop": 1.0, "compacted": 0.0, "skipped": 0.0, "failed": 0.0}
    assert tuple(rounds) == ROUND_OUTCOMES
    assert parsed.samples["gpu_fault_store_compaction_actions_total"] == []


def test_the_action_counter_carries_table_action_and_outcome(monkeypatch) -> None:
    _fake_clock(monkeypatch, [0.0], 2000)
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
    worker.run_all_due(0.0)

    parsed = _render(worker)

    counters = {
        (
            dict(s.labels)["table"],
            dict(s.labels)["action"],
            dict(s.labels)["status"],
        ): float(s.value)
        for s in parsed.samples["gpu_fault_store_compaction_actions_total"]
    }
    assert counters == {(QUEUE, REINDEX, "SUCCEEDED"): 1.0}
    assert [
        float(s.value)
        for s in parsed.samples[
            "gpu_fault_store_compaction_last_action_timestamp_seconds"
        ]
    ] == [2000]
    rounds = {
        dict(s.labels)["outcome"]: float(s.value)
        for s in parsed.samples["gpu_fault_store_compaction_rounds_total"]
    }
    assert rounds["compacted"] == 1.0
    due = {
        dict(s.labels)["table"]: float(s.value)
        for s in parsed.samples["gpu_fault_store_hot_table_due_actions"]
    }
    assert due == {QUEUE: 1.0, LATEST: 0.0}


def test_a_scrape_before_any_scan_renders_the_families_without_table_samples() -> None:
    worker = _runner(_PlainStore())
    parsed = _render(worker)
    for family, _key, _help in TABLE_GAUGES:
        assert family in parsed.families
        assert parsed.samples[family] == []
    for family in (
        "gpu_fault_store_bloat_scan_timestamp_seconds",
        "gpu_fault_store_compaction_last_timestamp_seconds",
        "gpu_fault_store_compaction_last_action_timestamp_seconds",
    ):
        assert [float(s.value) for s in parsed.samples[family]] == [0.0], family
    assert {
        dict(s.labels)["outcome"]: float(s.value)
        for s in parsed.samples["gpu_fault_store_compaction_rounds_total"]
    } == {outcome: 0.0 for outcome in ROUND_OUTCOMES}


def test_every_bloat_family_aggregates_per_process_like_the_drift_scan() -> None:
    families = [family for family, _key, _help in TABLE_GAUGES] + [
        "gpu_fault_store_bloat_scan_timestamp_seconds",
        "gpu_fault_store_compaction_last_timestamp_seconds",
        "gpu_fault_store_compaction_last_action_timestamp_seconds",
        "gpu_fault_store_compaction_rounds_total",
        "gpu_fault_store_compaction_actions_total",
    ]
    for family in families:
        assert STRATEGIES[family] is Strategy.PER_PROCESS, family


def test_the_hosting_threads_are_daemons() -> None:
    """Guards the hosting test above: every processor thread is a daemon, so
    a runner the test could not stop cannot hold the interpreter open."""

    non_daemons = [
        thread.name
        for thread in threading.enumerate()
        if thread.name.startswith("gpu-fault-") and not thread.daemon
    ]
    assert non_daemons == []

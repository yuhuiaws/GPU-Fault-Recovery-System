"""Two reconciliation jobs whose store halves existed without a caller.

F-D5: ``reclaim_expired_processor_leases`` hands lapsed LEASED rows back to
PENDING, but only the claim window called it, and only inside its horizon.
F-D10 (P1-75F): the processor queue keeps a per-cluster counter table beside
the rows; drift between the two was only visible to an admin CLI. Each now
runs on the periodic runner, counted and exported, and the drift query -- a
full-table count -- runs on the runner's clock, never inside a ``/metrics``
scrape.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from datetime import datetime, timezone
from threading import Event
from types import SimpleNamespace

import pytest

from gpu_fault.app.builtin_metric_contributors import control_loop_metric_lines
from gpu_fault.app.periodic_services import PeriodicServiceConfig, PeriodicServiceRunner
from gpu_fault.telemetry import CollectorKind

BATCH = 10


def _config(**overrides) -> PeriodicServiceConfig:
    base = PeriodicServiceConfig(
        training_interval=15.0,
        spare_interval=30.0,
        identity_interval=20.0,
        cleanup_interval=60.0,
        completed_retention=600.0,
        cleanup_batch_size=BATCH,
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


class _Store:
    def __init__(
        self,
        *,
        reclaimed: int = 0,
        expected_total: int = 0,
        counter_total: int = 0,
        mismatched_clusters: int = 0,
    ) -> None:
        self.reclaimed = reclaimed
        self.reclaim_calls: list[dict[str, object]] = []
        self.status = {
            "expected_total": expected_total,
            "counter_total": counter_total,
            "mismatched_clusters": mismatched_clusters,
            "ready": expected_total == counter_total and mismatched_clusters == 0,
        }
        self.status_calls = 0

    def reclaim_expired_processor_leases(self, *, now: datetime, limit: int) -> int:
        self.reclaim_calls.append({"now": now, "limit": limit})
        return self.reclaimed

    def processor_queue_count_status(self) -> dict[str, object]:
        self.status_calls += 1
        return dict(self.status)


def _runner(store, **config) -> PeriodicServiceRunner:
    processor = SimpleNamespace(
        is_healthy=lambda: True,
        active_consumers=False,
        is_leader=lambda: True,
        owner_id="pod-a:1",
    )
    return PeriodicServiceRunner(
        context=SimpleNamespace(store=store, regional_mode=False, completion=None),
        processor=processor,
        stop=Event(),
        identity_registries=[],
        ingest_node_health_findings=lambda *args, **kwargs: None,
        notify_silent_collectors=lambda *args, **kwargs: None,
        config=_config(**config),
    )


# --- configuration -----------------------------------------------------------


def test_reconciliation_intervals_have_their_documented_defaults(monkeypatch):
    for name in (
        "GPU_FAULT_PROCESSOR_LEASE_RECLAIM_SECONDS",
        "GPU_FAULT_PROCESSOR_COUNTER_DRIFT_SCAN_SECONDS",
    ):
        monkeypatch.delenv(name, raising=False)

    config = PeriodicServiceConfig.from_environment()

    assert config.lease_reclaim_interval == 30.0
    assert config.counter_drift_interval == 60.0


def test_reconciliation_intervals_are_read_from_the_environment(monkeypatch):
    monkeypatch.setenv("GPU_FAULT_PROCESSOR_LEASE_RECLAIM_SECONDS", "7")
    monkeypatch.setenv("GPU_FAULT_PROCESSOR_COUNTER_DRIFT_SCAN_SECONDS", "9")

    config = PeriodicServiceConfig.from_environment()

    assert config.lease_reclaim_interval == 7.0
    assert config.counter_drift_interval == 9.0


@pytest.mark.parametrize(
    "name",
    [
        "GPU_FAULT_PROCESSOR_LEASE_RECLAIM_SECONDS",
        "GPU_FAULT_PROCESSOR_COUNTER_DRIFT_SCAN_SECONDS",
    ],
)
def test_a_non_positive_reconciliation_interval_is_refused(monkeypatch, name):
    monkeypatch.setenv(name, "0")

    with pytest.raises(ValueError, match=name):
        PeriodicServiceConfig.from_environment()


# --- F-D5 ---------------------------------------------------------------------


def test_expired_leases_are_reclaimed_counted_and_warned_about(caplog):
    store = _Store(reclaimed=3)
    runner = _runner(store)

    with caplog.at_level(logging.WARNING, logger="gpu_fault.app.periodic_services"):
        ran = runner._run_lease_reclaim(1000.0)

    assert ran is True, "the reclaim job did not run when due"
    assert len(store.reclaim_calls) == 1
    assert store.reclaim_calls[0]["limit"] == 256
    observed = store.reclaim_calls[0]["now"]
    assert isinstance(observed, datetime) and observed.tzinfo is timezone.utc, (
        "reclaim must be judged on an aware UTC clock"
    )
    assert runner.metrics_snapshot()["processor_expired_leases_reclaimed_total"] == 3
    assert any(
        record.levelno == logging.WARNING and "3" in record.getMessage()
        for record in caplog.records
    ), [record.getMessage() for record in caplog.records]


def test_a_quiet_reclaim_round_logs_nothing_and_adds_nothing(caplog):
    store = _Store(reclaimed=0)
    runner = _runner(store)

    with caplog.at_level(logging.INFO, logger="gpu_fault.app.periodic_services"):
        runner._run_lease_reclaim(1000.0)

    assert runner.metrics_snapshot()["processor_expired_leases_reclaimed_total"] == 0
    assert caplog.records == []


# --- F-G2 (4) -----------------------------------------------------------------


# --- F-D10 P1-75F -------------------------------------------------------------


def test_processor_counter_drift_is_measured_and_warned_about(caplog):
    store = _Store(expected_total=10, counter_total=7, mismatched_clusters=2)
    runner = _runner(store)

    with caplog.at_level(logging.WARNING, logger="gpu_fault.app.periodic_services"):
        ran = runner._run_counter_drift(1000.0)

    assert ran is True, "the drift job did not run when due"
    snapshot = runner.metrics_snapshot()
    assert snapshot["processor_counter_drift_abs"] == 3
    assert snapshot["processor_counter_mismatched_clusters"] == 2
    assert any(record.levelno == logging.WARNING for record in caplog.records), (
        "non-zero drift must be logged at WARNING"
    )


def test_processor_counter_drift_is_absolute_and_quiet_when_aligned(caplog):
    store = _Store(expected_total=5, counter_total=9, mismatched_clusters=0)
    runner = _runner(store)
    runner._run_counter_drift(1000.0)
    assert runner.metrics_snapshot()["processor_counter_drift_abs"] == 4

    aligned = _Store(expected_total=9, counter_total=9, mismatched_clusters=0)
    quiet = _runner(aligned)
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="gpu_fault.app.periodic_services"):
        quiet._run_counter_drift(1000.0)

    assert quiet.metrics_snapshot()["processor_counter_drift_abs"] == 0
    assert caplog.records == []


def test_the_drift_query_runs_on_the_runner_clock_not_the_scrape():
    store = _Store(expected_total=10, counter_total=7, mismatched_clusters=2)
    runner = _runner(store)
    runner._run_counter_drift(1000.0)
    runtime = SimpleNamespace(context=SimpleNamespace(periodic_runner=runner))

    lines = control_loop_metric_lines(runtime)
    control_loop_metric_lines(runtime)

    assert store.status_calls == 1, "a /metrics render must not count the table"
    assert "gpu_fault_processor_counter_drift_abs 3" in lines
    assert "gpu_fault_processor_counter_mismatched_clusters 2" in lines


# --- export and scheduling ------------------------------------------------------


def test_reconciliation_counters_reach_metrics():
    store = _Store(reclaimed=3)
    runner = _runner(store)
    runner._run_lease_reclaim(1000.0)
    runtime = SimpleNamespace(context=SimpleNamespace(periodic_runner=runner))

    lines = control_loop_metric_lines(runtime)

    assert "gpu_fault_processor_expired_leases_reclaimed_total 3" in lines
    assert "# TYPE gpu_fault_processor_counter_drift_abs gauge" in lines


def test_the_new_jobs_take_their_turn_in_every_tick(monkeypatch):
    runner = _runner(_Store())
    ran: list[str] = []
    for name in runner._JOBS:
        monkeypatch.setattr(
            runner, f"_run_{name}", lambda now, name=name: ran.append(name)
        )

    runner.run_all_due(1000.0)

    for name in ("lease_reclaim", "counter_drift"):
        assert name in ran, f"{name} is not part of the periodic tick: {ran}"

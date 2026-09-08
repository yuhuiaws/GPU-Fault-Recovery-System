"""A periodic job whose *body* raises is counted, stamped and not reported as
"just ran".

Control-plane review 2026-09-08, F-1 (CP-5). ``run_all_due`` counted a job
only when ``_run_<name>`` itself raised, but every job body runs under
``_run_scheduled``, which swallowed the exception, logged it and returned
``ran=True``. A store outage that failed ``reclaim_expired_processor_leases``
every 30 s for hours therefore left ``periodic_job_errors_total`` at zero,
``job_error_last_seen`` empty and ``job_last_run`` refreshing as if the reclaim
had succeeded -- so neither ``GpuFaultPeriodicServiceErrors`` nor
``GpuFaultPeriodicRunnerStalled`` fired. These tests drive the runner through
``run_all_due`` with real job bodies; nothing patches ``_run_*``.
"""

from __future__ import annotations

import time
from dataclasses import replace
from threading import Event
from types import SimpleNamespace

import pytest

from gpu_fault.app.periodic_services import PeriodicServiceConfig, PeriodicServiceRunner
from gpu_fault.telemetry import CollectorKind

INTERVAL = 30.0


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
        lease_reclaim_interval=INTERVAL,
        pending_triage_interval=INTERVAL,
        counter_drift_interval=INTERVAL,
    )
    return replace(base, **overrides)


class _OutageStore:
    """Every reconciliation read/write fails the way a pool outage fails;
    the cleanup sweeps succeed so only the job bodies under test raise."""

    def __init__(self) -> None:
        self.reclaim_calls = 0

    def reclaim_expired_processor_leases(self, **kwargs):
        self.reclaim_calls += 1
        raise RuntimeError("could not connect to server")

    def processor_queue_count_status(self):
        raise RuntimeError("could not connect to server")

    def cleanup_completed_processor_requests(self, **kwargs):
        return 0

    def cleanup_expired_raw_evidence(self, **kwargs):
        return 0

    def cleanup_hot_state(self, **kwargs):
        return {}

    def cleanup_processor_lanes(self, **kwargs):
        return 0

    def cleanup_inactive_markers(self, **kwargs):
        return 0

    def cleanup_terminal_notifications(self, **kwargs):
        return 0

    def cleanup_completion_records(self, **kwargs):
        return 0


def _runner(store, completion=None, **config) -> PeriodicServiceRunner:
    processor = SimpleNamespace(
        is_healthy=lambda: True,
        # Leader without consumers: no task-lease write, so the store outage
        # reaches the job bodies rather than ``_owns_task_lease``.
        active_consumers=False,
        is_leader=lambda: True,
        owner_id="pod-a:1",
    )
    context = SimpleNamespace(
        store=store,
        completion=completion,
        regional_mode=False,
        control_record_archiver=None,
        spare_health_controller=None,
    )
    return PeriodicServiceRunner(
        context=context,
        processor=processor,
        stop=Event(),
        identity_registries=[],
        ingest_node_health_findings=lambda *args, **kwargs: None,
        notify_silent_collectors=lambda *args, **kwargs: None,
        config=_config(**config),
    )


@pytest.fixture(autouse=True)
def _training_monitor_off(monkeypatch):
    monkeypatch.delenv("GPU_FAULT_ENABLE_TRAINING_HEALTH_MONITOR", raising=False)


def test_a_job_body_that_raises_is_counted_and_stamped_by_job() -> None:
    runner = _runner(_OutageStore())
    before = time.time()

    runner.run_all_due(1000.0)

    snapshot = runner.metrics_snapshot()
    failing = {"lease_reclaim", "counter_drift"}
    assert snapshot["periodic_job_errors_total"] == {name: 1 for name in failing}
    assert set(snapshot["job_error_last_seen_timestamp_seconds"]) == failing
    assert all(
        stamp >= before
        for stamp in snapshot["job_error_last_seen_timestamp_seconds"].values()
    )


def test_a_failed_job_does_not_refresh_its_last_run_stamp() -> None:
    """``job_last_run`` is what the stall alert reads; a job that fails every
    round must look stalled, not freshly run. The cleanup, which did run,
    keeps its stamp."""

    runner = _runner(_OutageStore())

    runner.run_all_due(1000.0)

    stamps = runner.metrics_snapshot()["job_last_run_timestamp_seconds"]
    assert "cleanup" in stamps
    assert not {"lease_reclaim", "counter_drift"} & set(stamps)


def test_a_failed_job_is_rescheduled_and_counted_again_next_interval() -> None:
    store = _OutageStore()
    runner = _runner(store)

    runner.run_all_due(1000.0)
    runner.run_all_due(1000.0 + 1.0)
    assert store.reclaim_calls == 1, "the failed job ran again before its interval"

    runner.run_all_due(time.monotonic() + INTERVAL + 1.0)

    assert store.reclaim_calls == 2
    assert runner.metrics_snapshot()["periodic_job_errors_total"]["lease_reclaim"] == 2


def test_a_failing_job_does_not_stop_the_jobs_behind_it_in_the_tick() -> None:
    """``lease_reclaim`` fails; ``counter_drift`` behind
    it still gets its turn (F-F1 isolation, now through the real bodies)."""

    calls: list[str] = []

    class _Store(_OutageStore):
        def processor_queue_count_status(self):
            calls.append("counter_drift")
            return {"expected_total": 0, "counter_total": 0, "mismatched_clusters": 0}

    runner = _runner(_Store())

    runner.run_all_due(1000.0)

    assert calls == ["counter_drift"]
    snapshot = runner.metrics_snapshot()
    assert snapshot["periodic_job_errors_total"] == {"lease_reclaim": 1}
    assert "counter_drift" in snapshot["job_last_run_timestamp_seconds"]

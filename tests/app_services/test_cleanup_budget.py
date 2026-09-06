"""The cleanup budget is shared fairly, squeezed jobs are visible, schedules
advance after the work, and lease losers stop knocking every 3 s.

FINAL-建议汇总 F-F2 (P1-21C, P2-77I, P1-21D, P2-21E, P3-77J, P2-21G, P3-21H,
P2-32G, P3-32I). One 20 s deadline was handed to seven jobs in a fixed order,
so the first (COMPLETED queue rows, the largest table) could spend it all and
processor-lane cleanup -- last in line -- got one batch a minute with no log
and no metric. ``next_run`` was advanced from a ``now`` taken before a 20 s
cleanup, and 24 processes re-asked for six leases every 3 s to be told "not
yours".
"""

from __future__ import annotations

import time
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from threading import Event
from types import SimpleNamespace

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


class _CleanupStore:
    """Every job returns a full batch forever (saturated) unless told otherwise,
    and each call costs a little wall clock so the budget is real."""

    def __init__(
        self,
        *,
        saturated: frozenset[str] = frozenset({"completed", "lanes"}),
        failing: frozenset[str] = frozenset(),
        cost_seconds: float = 0.01,
    ) -> None:
        self.calls: dict[str, int] = {}
        self.order: list[str] = []
        self.saturated = saturated
        self.failing = failing
        self.cost_seconds = cost_seconds

    def _job(self, name: str, *, dict_result: bool = False):
        self.calls[name] = self.calls.get(name, 0) + 1
        self.order.append(name)
        if name in self.failing:
            raise RuntimeError(f"{name} exploded")
        time.sleep(self.cost_seconds)
        count = BATCH if name in self.saturated else 0
        return {"rows": count} if dict_result else count

    def cleanup_completed_processor_requests(self, **_kwargs):
        return self._job("completed")

    def cleanup_expired_raw_evidence(self, **_kwargs):
        return self._job("raw_evidence")

    def cleanup_hot_state(self, **_kwargs):
        return self._job("hot_state", dict_result=True)

    def cleanup_processor_lanes(self, **_kwargs):
        return self._job("lanes")


def _runner(store, *, regional_mode: bool = False, **config) -> PeriodicServiceRunner:
    processor = SimpleNamespace(
        is_healthy=lambda: True,
        active_consumers=False,
        is_leader=lambda: True,
        owner_id="pod-a:1",
    )
    return PeriodicServiceRunner(
        context=SimpleNamespace(store=store, regional_mode=regional_mode),
        processor=processor,
        stop=Event(),
        identity_registries=[],
        ingest_node_health_findings=lambda *args, **kwargs: None,
        notify_silent_collectors=lambda *args, **kwargs: None,
        config=_config(**config),
    )


def test_the_last_job_gets_its_own_share_of_the_budget():
    store = _CleanupStore()
    runner = _runner(store)

    runner._run_cleanup(time.monotonic())

    # 0.2 s over four jobs is ~0.05 s each, ~5 batches at 0.01 s: lane cleanup
    # used to get exactly one batch because the first job spent everything.
    assert store.calls["lanes"] >= 3, store.calls
    assert store.calls["completed"] >= 3, store.calls


def test_squeezed_jobs_are_counted_by_name():
    store = _CleanupStore()
    runner = _runner(store)

    runner._run_cleanup(time.monotonic())
    snapshot = runner.metrics_snapshot()

    rows = snapshot["cleanup_rows_total"]
    assert rows["completed_requests"] == store.calls["completed"] * BATCH
    assert rows["processor_lanes"] == store.calls["lanes"] * BATCH
    assert rows["raw_evidence"] == 0
    exhausted = snapshot["cleanup_budget_exhausted_total"]
    assert exhausted == {"completed_requests": 1, "processor_lanes": 1}


def test_the_starting_job_rotates_between_rounds():
    store = _CleanupStore(saturated=frozenset(), cost_seconds=0.0)
    runner = _runner(store)

    runner._cleanup_round()
    first_round = list(store.order)
    store.order.clear()
    runner._cleanup_round()

    assert first_round[0] == "completed"
    assert store.order[0] == "raw_evidence"
    assert sorted(store.order) == sorted(first_round)


def test_a_failing_job_is_counted_and_the_rest_still_run():
    store = _CleanupStore(saturated=frozenset(), failing=frozenset({"completed"}))
    runner = _runner(store)

    runner._run_cleanup(time.monotonic())

    assert store.calls["lanes"] == 1
    assert runner.metrics_snapshot()["cleanup_job_errors_total"] == {
        "completed_requests": 1
    }


def test_cleanup_counters_reach_metrics():
    store = _CleanupStore()
    runner = _runner(store)
    runner._run_cleanup(time.monotonic())
    runtime = SimpleNamespace(context=SimpleNamespace(periodic_runner=runner))

    lines = control_loop_metric_lines(runtime)

    rows = runner.metrics_snapshot()["cleanup_rows_total"]["completed_requests"]
    assert (
        f'gpu_fault_periodic_cleanup_rows_total{{job="completed_requests"}} {rows}'
        in lines
    )
    assert (
        'gpu_fault_periodic_cleanup_budget_exhausted_total{job="processor_lanes"} 1'
        in lines
    )


def test_next_run_is_advanced_after_the_job_not_before():
    runner = _runner(_CleanupStore())
    finished_at: list[float] = []

    def body() -> None:
        time.sleep(0.05)
        finished_at.append(time.monotonic())

    ran = runner._run_scheduled("probe", time.monotonic(), 10.0, body, "probe failed")

    assert ran is True
    assert runner.next_run["probe"] >= finished_at[0] + 10.0 - 1e-6


def test_a_lease_loser_sleeps_until_the_lease_expires():
    now = datetime.now(timezone.utc)

    class Store:
        def __init__(self) -> None:
            self.calls = 0

        def acquire_periodic_task_lease(self, *args, **kwargs):
            self.calls += 1
            return SimpleNamespace(
                task_key="processor-cleanup",
                owner_id="pod-b:1",
                epoch=3,
                lease_expires_at=now + timedelta(seconds=30),
            )

    store = Store()
    runner = _runner(store)
    runner.processor.active_consumers = True

    assert runner._owns_task_lease("processor-cleanup", 60.0) is False
    wait = runner.next_lease_attempt["processor-cleanup"] - time.monotonic()
    assert 25.0 < wait <= 30.0, wait
    # The next tick must not ask the store again while the lease is held.
    assert runner._owns_task_lease("processor-cleanup", 60.0) is False
    assert store.calls == 1


def test_lane_retention_defaults_to_one_hour(monkeypatch):
    monkeypatch.delenv("GPU_FAULT_PROCESSOR_LANE_RETENTION_SECONDS", raising=False)

    assert PeriodicServiceConfig.from_environment().lane_retention == 3600.0

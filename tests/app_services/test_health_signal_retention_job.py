"""The periodic cleanup round runs the health-signal retention (batch 145).

The sweep's rules are ``tests/store/test_health_signal_retention.py``; here,
only its wiring: the job, its knobs, its switch and its row counter.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from threading import Event
from types import SimpleNamespace

import pytest

from gpu_fault.app.builtin_metric_contributors import control_loop_metric_lines
from gpu_fault.app.periodic_services import PeriodicServiceConfig, PeriodicServiceRunner
from gpu_fault.telemetry import CollectorKind
from gpu_fault.watcher import WorkloadPhase
from tests._builders import build_store
from tests.store._health_signal_retention_cases import (
    GPU_GATE,
    claim,
    observe,
    watch,
    window,
)

BATCH = 10


class _RecordingStore:
    def __init__(self, removed: int = 0) -> None:
        self.calls: dict[str, dict] = {}
        self.removed = removed

    def __getattr__(self, name: str):
        if not name.startswith("cleanup_"):
            raise AttributeError(name)

        def record(**kwargs):
            self.calls[name] = kwargs
            if name == "cleanup_health_signal_states":
                return {"health_signal_state": self.removed, "efa_traffic_state": 1}
            return 0

        return record


def _config(**overrides) -> PeriodicServiceConfig:
    base = PeriodicServiceConfig(
        training_interval=15.0,
        spare_interval=30.0,
        identity_interval=20.0,
        cleanup_interval=60.0,
        completed_retention=600.0,
        cleanup_batch_size=BATCH,
        cleanup_budget_seconds=0.5,
        batch_retention=86400.0,
        terminal_retention=2592000.0,
        latest_retention=2592000.0,
        finding_retention=2592000.0,
        observation_max_age=604800.0,
        remote_retention=86400.0,
        remote_claim_deadline=900.0,
        lane_retention=3600.0,
        deployment_retention=604800.0,
        archive_interval=600.0,
        silence_interval=60.0,
        silent_after={kind: 300.0 for kind in CollectorKind},
        silent_alert_interval=3600.0,
    )
    return replace(base, **overrides)


def _runner(store, **config) -> PeriodicServiceRunner:
    return PeriodicServiceRunner(
        context=SimpleNamespace(store=store, regional_mode=False),
        processor=SimpleNamespace(
            is_healthy=lambda: True,
            active_consumers=False,
            is_leader=lambda: True,
            owner_id="pod-a:1",
        ),
        stop=Event(),
        identity_registries=[],
        ingest_node_health_findings=lambda *a, **k: None,
        notify_silent_collectors=lambda *a, **k: None,
        config=_config(**config),
    )


def test_the_cleanup_round_runs_the_retention_with_its_windows() -> None:
    store = _RecordingStore()
    before = datetime.now(timezone.utc)

    _runner(store)._cleanup_round()

    call = store.calls["cleanup_health_signal_states"]
    assert abs((call["now"] - before).total_seconds()) < 5
    assert call["grace"] == timedelta(days=1)
    assert call["retention"] == timedelta(days=30)
    assert call["limit"] == BATCH


def test_a_non_positive_retention_switches_the_sweep_off() -> None:
    store = _RecordingStore()

    _runner(store, health_signal_retention=0.0)._cleanup_round()

    assert "cleanup_health_signal_states" not in store.calls


def test_removed_rows_are_counted_under_the_job_and_a_full_batch_drains_on() -> None:
    store = _RecordingStore(removed=BATCH)
    runner = _runner(store)

    runner._cleanup_round()

    # A full batch of one kind is saturation: the drain repeats the call
    # within the job's slice of the budget, and every row is counted.
    rows = runner.cleanup_rows_total["health_signal_states"]
    assert rows >= 2 * (BATCH + 1)
    assert rows % (BATCH + 1) == 0
    runtime = SimpleNamespace(context=SimpleNamespace(periodic_runner=runner))
    assert (
        f'gpu_fault_periodic_cleanup_rows_total{{periodic_job="health_signal_states"}} '
        f"{rows}"
    ) in control_loop_metric_lines(runtime)


def test_the_round_removes_an_ended_attempts_window_end_to_end() -> None:
    store = build_store()
    now = datetime.now(timezone.utc)
    watch(store, at=now)
    observe(store, "job-a", "attempt-1", WorkloadPhase.SUCCEEDED, at=now)
    claim(store, window(GPU_GATE, "job-a", "attempt-1"), now - timedelta(days=2))
    claim(store, GPU_GATE, now - timedelta(days=2))
    runner = _runner(store)

    runner._cleanup_round()

    assert store.get_health_signal_state(window(GPU_GATE, "job-a", "attempt-1")) is None
    assert store.get_health_signal_state(GPU_GATE) is not None
    assert runner.cleanup_rows_total["health_signal_states"] == 1


def test_the_windows_are_read_from_the_environment(monkeypatch) -> None:
    monkeypatch.setenv("GPU_FAULT_HEALTH_SIGNAL_RETENTION_SECONDS", "864000")
    monkeypatch.setenv("GPU_FAULT_HEALTH_SIGNAL_GRACE_SECONDS", "7200")

    config = PeriodicServiceConfig.from_environment()

    assert (config.health_signal_retention, config.health_signal_grace) == (
        864000.0,
        7200.0,
    )
    monkeypatch.setenv("GPU_FAULT_HEALTH_SIGNAL_RETENTION_SECONDS", "0")
    monkeypatch.setenv("GPU_FAULT_HEALTH_SIGNAL_GRACE_SECONDS", "1")
    assert PeriodicServiceConfig.from_environment().health_signal_retention == 0.0


@pytest.mark.parametrize(("retention", "grace"), [(86400.0, 599.0), (86400.0, 86401.0)])
def test_a_grace_that_could_remove_an_open_window_is_refused(
    retention: float, grace: float
) -> None:
    with pytest.raises(ValueError, match="GPU_FAULT_HEALTH_SIGNAL_GRACE_SECONDS"):
        _config(health_signal_retention=retention, health_signal_grace=grace)

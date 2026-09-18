"""A scan's evidence clock must not advance with metric publication or takeover."""

from __future__ import annotations

import math
from datetime import timedelta
from threading import Event
from types import SimpleNamespace

import pytest

from gpu_fault.app import periodic_services
from gpu_fault.app.builtin_metric_contributors import control_loop_metric_lines
from gpu_fault.app.periodic_services import PeriodicServiceConfig, PeriodicServiceRunner
from gpu_fault.app.process_metrics import aggregate, parse_lines

STAMP = "gpu_fault_processor_counter_drift_scan_timestamp_seconds"
AGE = "gpu_fault_processor_counter_drift_scan_max_age_seconds"
DRIFT = "gpu_fault_processor_counter_drift_abs"
MISMATCHED = "gpu_fault_processor_counter_mismatched_clusters"


class ScanStore:
    owner = "owner-a"
    drift = 7
    mismatched = 1
    fail = False

    def acquire_periodic_task_lease(self, key, _owner, *, now, lease_duration):
        return SimpleNamespace(
            owner_id=self.owner if key == "processor-counter-drift" else "other-job",
            lease_expires_at=now + lease_duration,
        )

    def processor_queue_count_status(self):
        if self.fail:
            raise RuntimeError("synthetic scan failure")
        return {
            "expected_total": self.drift,
            "counter_total": 0,
            "mismatched_clusters": self.mismatched,
        }


def runner(store: ScanStore, owner: str) -> PeriodicServiceRunner:
    return PeriodicServiceRunner(
        context=SimpleNamespace(
            store=store,
            regional_mode=False,
            spare_health_controller=None,
            control_record_archiver=None,
        ),
        processor=SimpleNamespace(active_consumers=True, owner_id=owner),
        stop=Event(),
        identity_registries=[],
        ingest_node_health_findings=lambda *_args: None,
        notify_silent_collectors=lambda *_args, **_kwargs: None,
        config=PeriodicServiceConfig.from_environment(),
    )


def metric_rows(worker: PeriodicServiceRunner):
    return parse_lines(
        control_loop_metric_lines(
            SimpleNamespace(context=SimpleNamespace(periodic_runner=worker))
        )
    )


def test_old_owner_republishes_the_original_scan_clock_after_takeover(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = [0.0]
    monkeypatch.setattr(
        periodic_services,
        "time",
        SimpleNamespace(monotonic=lambda: clock[0], time=lambda: 1000 + clock[0]),
    )
    store = ScanStore()
    first, second = runner(store, "owner-a"), runner(store, "owner-b")
    assert (
        first.metrics_snapshot()["processor_counter_drift_scan_timestamp_seconds"] == 0
    )
    first.run_all_due(clock[0])
    assert first.metrics_snapshot()["processor_counter_drift_abs"] == 7
    assert (
        first.metrics_snapshot()["processor_counter_drift_scan_timestamp_seconds"]
        == 1000
    )

    clock[0] = 121
    store.owner, store.drift, store.mismatched = "owner-b", 0, 0
    second.run_all_due(clock[0])
    clock[0] = 122
    first.run_all_due(clock[0])
    left, right = metric_rows(first), metric_rows(second)
    left.slot, right.slot = 0, 1
    result = parse_lines(aggregate(left, [right]))
    for family, expected in (
        (DRIFT, [7, 0]),
        (MISMATCHED, [1, 0]),
        (STAMP, [1000, 1121]),
        (AGE, [180, 180]),
    ):
        assert [
            (dict(sample.labels), float(sample.value))
            for sample in result.samples[family]
        ] == [({"process": "0"}, expected[0]), ({"process": "1"}, expected[1])]
    assert first.metrics_snapshot()["periodic_job_errors_total"] == {}
    assert second.metrics_snapshot()["periodic_job_errors_total"] == {}

    store.fail = True
    clock[0] = 300
    second.run_all_due(clock[0])
    failed = second.metrics_snapshot()
    assert failed["processor_counter_drift_scan_timestamp_seconds"] == 1121
    assert failed["processor_counter_drift_abs"] == 0
    assert failed["periodic_job_errors_total"] == {"counter_drift": 1}


@pytest.mark.parametrize("interval,expected", [(1, 120), (60, 180), (600, 1800)])
def test_scan_freshness_bound_comes_from_the_actual_configured_interval(
    monkeypatch: pytest.MonkeyPatch, interval: int, expected: int
) -> None:
    monkeypatch.setenv("GPU_FAULT_PROCESSOR_COUNTER_DRIFT_SCAN_SECONDS", str(interval))
    worker = runner(ScanStore(), "owner-a")
    assert worker.metrics_snapshot()[
        "processor_counter_drift_scan_max_age_seconds"
    ] == (timedelta(seconds=expected).total_seconds())
    assert (
        worker.metrics_snapshot()["processor_counter_drift_scan_timestamp_seconds"] == 0
    )


def test_all_periodic_task_families_keep_a_nonreserved_task_label() -> None:
    snapshot = {
        "periodic_job_errors_total": {"cleanup": 4},
        "job_last_run_timestamp_seconds": {"cleanup": 900.0},
        "job_error_last_seen_timestamp_seconds": {"cleanup": 1000.0},
        "cleanup_rows_total": {"cleanup": 3},
        "cleanup_budget_exhausted_total": {"cleanup": 1},
        "cleanup_job_errors_total": {"cleanup": 2},
        "processor_counter_drift_scan_timestamp_seconds": None,
        "processor_counter_drift_scan_max_age_seconds": "unknown",
    }
    lines = control_loop_metric_lines(
        SimpleNamespace(
            context=SimpleNamespace(
                periodic_runner=SimpleNamespace(metrics_snapshot=lambda: snapshot)
            )
        )
    )
    parsed = parse_lines(lines)
    for name, count in (
        ("gpu_fault_periodic_job_errors_total", 4),
        ("gpu_fault_periodic_job_last_run_timestamp_seconds", 900),
        ("gpu_fault_periodic_job_error_last_seen_timestamp_seconds", 1000),
        ("gpu_fault_periodic_cleanup_rows_total", 3),
        ("gpu_fault_periodic_cleanup_budget_exhausted_total", 1),
        ("gpu_fault_periodic_cleanup_job_errors_total", 2),
    ):
        (sample,) = parsed.samples[name]
        assert dict(sample.labels) == {"periodic_job": "cleanup"}
        assert float(sample.value) == count
    for name in (STAMP, AGE):
        assert math.isnan(float(parsed.samples[name][0].value)), (
            f"{name} must remain unknown for malformed scan metadata"
        )

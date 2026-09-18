from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import pytest

from gpu_fault.app import periodic_services
from gpu_fault.app.periodic_services import PeriodicServiceConfig, PeriodicServiceRunner
from tests._builders import build_context
from tests.app_services._cov95_runtime_workers import Stop
from tests.app_services.test_cov95_runtime_health_ingestion import finding
from tests.regional._cov95_runtime_support import offline_runtime as offline_runtime


def periodic(
    context: Any,
    *,
    registries: list[Any] | None = None,
    ingest: Any = None,
    notify: Any = None,
    config: PeriodicServiceConfig | None = None,
) -> PeriodicServiceRunner:
    return PeriodicServiceRunner(
        context=context,
        processor=SimpleNamespace(
            is_healthy=lambda: True,
            is_leader=lambda: True,
            active_consumers=False,
            owner_id="unit-periodic-owner",
        ),
        stop=Stop(),
        identity_registries=registries or [],
        ingest_node_health_findings=ingest or (lambda *args: None),
        notify_silent_collectors=notify or (lambda *args, **kwargs: None),
        config=config or PeriodicServiceConfig.from_environment(),
    )


@pytest.mark.parametrize("failures", [0, 1, 2])
def test_scheduled_identity_failure_does_not_starve_other_clusters_or_hide_the_error(
    monkeypatch: pytest.MonkeyPatch, failures: int
) -> None:
    calls = []
    clock = [1000.0]
    monkeypatch.setattr(
        periodic_services,
        "time",
        SimpleNamespace(
            monotonic=lambda: clock[0], time=lambda: 1700000000.0 + clock[0]
        ),
    )

    def refresh(index: int) -> None:
        calls.append("cluster-a" if index == 0 else "cluster-b")
        if index < failures:
            raise RuntimeError("synthetic cluster identity unavailable")

    runner = periodic(
        build_context(),
        registries=[
            SimpleNamespace(refresh=lambda: refresh(0)),
            SimpleNamespace(refresh=lambda: refresh(1)),
        ],
    )
    runner.run_all_due(clock[0])
    assert calls == ["cluster-a", "cluster-b"]
    snapshot = runner.metrics_snapshot()
    assert snapshot["periodic_job_errors_total"] == (
        {"identity": 1} if failures else {}
    )
    assert ("identity" in snapshot["job_last_run_timestamp_seconds"]) is (not failures)
    assert ("identity" in snapshot["job_error_last_seen_timestamp_seconds"]) is bool(
        failures
    )
    clock[0] += 1
    runner.run_all_due(clock[0])
    assert calls == ["cluster-a", "cluster-b"]
    clock[0] += runner.config.identity_interval
    runner.run_all_due(clock[0])
    assert calls == ["cluster-a", "cluster-b", "cluster-a", "cluster-b"]
    assert runner.metrics_snapshot()["periodic_job_errors_total"] == (
        {"identity": 2} if failures else {}
    )


@pytest.mark.parametrize("enabled", [False, True])
def test_optional_retention_jobs_respect_disabled_configuration(
    monkeypatch: pytest.MonkeyPatch, enabled: bool
) -> None:
    context = build_context()
    context.regional_mode = True
    config = PeriodicServiceConfig.from_environment()
    if not enabled:
        config = replace(
            config,
            marker_retention=0,
            notification_retention=0,
            completion_record_retention=0,
            registry_member_retention=0,
            remote_stale_fence_grace=0,
            remote_claim_deadline=0,
            observation_max_age=0,
        )
    always = {
        "cleanup_completed_processor_requests",
        "cleanup_expired_raw_evidence",
        "cleanup_hot_state",
        "cleanup_processor_lanes",
        "cleanup_terminal_remote_commands",
        "cleanup_terminal_fleet_deployments",
    }
    optional = {
        "cleanup_inactive_markers",
        "cleanup_terminal_notifications",
        "cleanup_completion_records",
        "cleanup_stale_regional_registry_members",
        "expire_stale_fenced_remote_commands",
        "expire_unclaimed_remote_commands",
    }
    calls = {}
    for name in always | optional:
        original = getattr(context.store, name)

        def capture(
            *args: Any, name: str = name, original: Any = original, **kwargs: Any
        ) -> Any:
            calls[name] = kwargs
            return original(*args, **kwargs)

        monkeypatch.setattr(context.store, name, capture)
    runner = periodic(context, config=config)
    runner.run_all_due(1000)
    assert set(calls) == always | (optional if enabled else set())
    age = calls["cleanup_hot_state"]["attempt_observation_max_age"]
    if enabled:
        assert age.total_seconds() == config.observation_max_age
    else:
        assert age is None
    assert runner.metrics_snapshot()["periodic_job_errors_total"] == {}


@pytest.mark.parametrize(
    "mode", ["quiet", "findings", "training-error", "spare-error", "archive-error"]
)
def test_periodic_job_bodies_keep_notification_order_and_isolate_failures(
    monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    monkeypatch.setenv("GPU_FAULT_ENABLE_TRAINING_HEALTH_MONITOR", "true")
    context = build_context()
    context.regional_mode = True
    calls = []
    findings = [] if mode == "quiet" else [finding("periodic", "unit_observation")]

    def invoke(name: str, result: Any = None) -> Any:
        calls.append(name)
        if mode == name + "-error":
            raise RuntimeError("synthetic job unavailable")
        return result

    def ingest(batch_id: str, actual: Any) -> None:
        assert batch_id == "training-health-monitor"
        assert actual == findings
        invoke("training")

    def notified(actual: Any) -> None:
        assert actual == findings
        invoke("training-notified")

    context.training_health = SimpleNamespace(
        scan_all=lambda: invoke("training-scan", SimpleNamespace(findings=findings)),
        mark_notified=notified,
    )
    context.spare_health_controller = SimpleNamespace(scan=lambda: invoke("spare"))
    context.control_record_archiver = SimpleNamespace(
        run_once=lambda **kwargs: invoke(
            "archive", [] if mode == "quiet" else ["unit-audit"]
        )
    )
    runner = periodic(
        context, ingest=ingest, notify=lambda *args, **kwargs: invoke("silence")
    )
    runner.run_all_due(1000)
    expected_training = (
        ["training-scan"]
        if mode == "quiet"
        else ["training-scan", "training"]
        if mode == "training-error"
        else ["training-scan", "training", "training-notified"]
    )
    assert calls == expected_training + ["spare", "archive", "silence"]
    errors = {mode.removesuffix("-error"): 1} if mode.endswith("-error") else {}
    assert runner.metrics_snapshot()["periodic_job_errors_total"] == errors
    assert all(
        name not in runner.metrics_snapshot()["job_last_run_timestamp_seconds"]
        for name in errors
    ), "failed bodies must not refresh success timestamps"

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from gpu_fault.app import lifespan_workers as workers
from gpu_fault.store.contracts import WakeupChannel
from tests._builders import build_context
from tests.app_services._cov95_runtime_workers import Stop, ThreadRecorder
from tests.regional._cov95_runtime_support import Clock
from tests.regional._cov95_runtime_support import offline_runtime as offline_runtime


@pytest.mark.parametrize("listener", [False, True])
@pytest.mark.parametrize("enabled", [False, True])
def test_dispatcher_wakeup_threads_bind_each_channel_only_when_enabled(
    monkeypatch: pytest.MonkeyPatch, listener: bool, enabled: bool
) -> None:
    recorder = ThreadRecorder()
    monkeypatch.setattr(workers, "Thread", recorder)
    calls = []
    store = SimpleNamespace()
    if listener:
        store.run_wakeup_listener = object()
    context = SimpleNamespace(
        store=store,
        dispatcher=SimpleNamespace(
            config=SimpleNamespace(enabled=enabled),
            run_wakeups=lambda *args: calls.append(args),
        ),
    )
    stop = Stop()
    threads = workers.start_dispatcher_wakeup_threads(context, stop)
    assert len(threads) == (len(WakeupChannel) if enabled and listener else 0)
    for thread in threads:
        assert thread.started is True
        thread.run()
    assert calls == (
        [(channel, stop) for channel in WakeupChannel] if enabled and listener else []
    )


@pytest.mark.parametrize("present", [False, True])
def test_registry_worker_has_explicit_stop_and_no_worker_when_absent(
    monkeypatch: pytest.MonkeyPatch, present: bool
) -> None:
    recorder = ThreadRecorder()
    monkeypatch.setattr(workers, "Thread", recorder)
    calls = []
    stop = Stop()
    runtime = (
        SimpleNamespace(run=lambda event: calls.append(event)) if present else None
    )
    worker = workers.start_regional_registry_worker(runtime, stop)
    if present:
        assert worker.name == "gpu-fault-regional-registry"
        worker.run()
        assert calls == [stop]
    else:
        assert worker is None and recorder.threads == []


@pytest.mark.parametrize("active", [False, True])
@pytest.mark.parametrize("spool", [False, True])
def test_processor_thread_composition_preserves_role_and_listener_boundaries(
    monkeypatch: pytest.MonkeyPatch, active: bool, spool: bool
) -> None:
    recorder = ThreadRecorder()
    monkeypatch.setattr(workers, "Thread", recorder)
    context = build_context()
    context.store = SimpleNamespace(
        run_wakeup_listener=object(),
        listen_processor_queue_notifications=object(),
        listen_telemetry_spool_notifications=object(),
    )
    context.dispatcher = SimpleNamespace(
        config=SimpleNamespace(enabled=True), run_wakeups=lambda *args: None
    )
    processor = SimpleNamespace(
        active_consumers=active,
        telemetry_spool_enabled=spool,
        run_processor=lambda: None,
        run_queue_notifications=lambda: None,
        run_telemetry_spool=lambda: None,
        run_telemetry_spool_notifications=lambda: None,
        run_leadership=lambda: None,
    )
    diagnostic, threads = workers.start_processor_threads(
        context=context,
        processor=processor,
        diagnostics_publisher=SimpleNamespace(run=lambda stop: None),
        stop=Stop(),
        identity_registries=[],
        ingest_node_health_findings=lambda *args: None,
        notify_silent_collectors=lambda *args: None,
    )
    names = {thread.name for thread in threads}
    assert ("gpu-fault-processor-leadership" in names) is (not active)
    assert ("gpu-fault-telemetry-spool" in names) is spool
    assert ("gpu-fault-telemetry-spool-notifications" in names) is spool
    assert "gpu-fault-processor-notifications" in names
    assert diagnostic.name == "gpu-fault-processor-diagnostics"
    assert all(thread.started for thread in [diagnostic, *threads]), (
        "every selected worker must start"
    )


@pytest.mark.parametrize(
    "defect",
    [
        "none",
        "disabled",
        "unhealthy",
        "not-leader",
        "dispatch-error",
        "correlation-error",
        "wake",
    ],
)
def test_dispatch_and_correlation_loops_recover_without_running_on_inactive_roles(
    monkeypatch: pytest.MonkeyPatch, defect: str, caplog: pytest.LogCaptureFixture
) -> None:
    recorder = ThreadRecorder()
    monkeypatch.setattr(workers, "Thread", recorder)
    monkeypatch.setattr(workers, "time", Clock())
    context = build_context()
    context.store = SimpleNamespace()
    calls = {"dispatch": 0, "correlation": 0}

    def run(name: str) -> None:
        calls[name] += 1
        if defect == name + "-error" and calls[name] == 1:
            raise RuntimeError("synthetic cycle failure")

    context.dispatcher = SimpleNamespace(
        config=SimpleNamespace(enabled=defect != "disabled", poll_interval_seconds=10),
        run_once=lambda: run("dispatch"),
        consume_wake=lambda: defect == "wake",
    )
    context.xid_correlation = SimpleNamespace(
        run_once=lambda: run("correlation"), poll_interval_seconds=10
    )
    processor = SimpleNamespace(
        active_consumers=defect != "not-leader",
        telemetry_spool_enabled=False,
        is_healthy=lambda: defect != "unhealthy",
        is_leader=lambda: False,
        run_processor=lambda: None,
        run_leadership=lambda: None,
    )
    stop = Stop(3)
    _diagnostic, threads = workers.start_processor_threads(
        context=context,
        processor=processor,
        diagnostics_publisher=SimpleNamespace(run=lambda stop: None),
        stop=stop,
        identity_registries=[],
        ingest_node_health_findings=lambda *args: None,
        notify_silent_collectors=lambda *args: None,
    )
    for name in ("gpu-fault-workflow-dispatcher", "gpu-fault-xid-correlation"):
        stop.cycles, stop.stopped = 3, False
        next(thread for thread in threads if thread.name == name).run()
    assert calls["dispatch"] == (
        0
        if defect in {"disabled", "unhealthy", "not-leader"}
        else 2
        if defect == "dispatch-error"
        else 3
        if defect == "wake"
        else 1
    )
    assert calls["correlation"] == (
        0
        if defect in {"unhealthy", "not-leader"}
        else 2
        if defect == "correlation-error"
        else 1
    )
    if defect.endswith("-error"):
        assert "cycle failed" in caplog.text


def test_one_identity_registry_failure_does_not_starve_another_cluster(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    recorder = ThreadRecorder()
    monkeypatch.setattr(workers, "Thread", recorder)
    monkeypatch.setattr(workers, "training_health_monitor_enabled", lambda: False)
    calls = []

    def failed() -> None:
        calls.append("a")
        raise RuntimeError("synthetic cluster-a provider unavailable")

    registries = [
        SimpleNamespace(refresh=failed),
        SimpleNamespace(refresh=lambda: calls.append("b")),
    ]
    context = SimpleNamespace(
        xid_correlation=SimpleNamespace(run_forever=lambda: None),
        spare_health_controller=None,
    )
    _xid, _training, _spare, identity = workers.start_nonprocessor_workers(
        context=context,
        stop=Stop(1),
        identity_registries=registries,
        ingest_node_health_findings=lambda *args: None,
    )
    identity.run()
    assert calls == ["a", "b", "a", "b"]
    assert "identity refresh failed" in caplog.text


@pytest.mark.parametrize("finding", ["none", "found", "failure"])
def test_training_health_worker_marks_notified_only_after_successful_ingestion(
    monkeypatch: pytest.MonkeyPatch, finding: str, caplog: pytest.LogCaptureFixture
) -> None:
    recorder = ThreadRecorder()
    monkeypatch.setattr(workers, "Thread", recorder)
    monkeypatch.setattr(workers, "training_health_monitor_enabled", lambda: True)
    findings = [{"node_id": "node-a"}] if finding != "none" else []
    calls = []

    def ingest(*args: Any) -> None:
        calls.append(("ingest", args))
        if finding == "failure":
            raise RuntimeError("synthetic ingestion failed")

    context = SimpleNamespace(
        xid_correlation=SimpleNamespace(run_forever=lambda: None),
        spare_health_controller=None,
        training_health=SimpleNamespace(
            scan_all=lambda: SimpleNamespace(findings=findings),
            mark_notified=lambda values: calls.append(("mark", values)),
        ),
    )
    _xid, training, spare, identity = workers.start_nonprocessor_workers(
        context=context,
        stop=Stop(1),
        identity_registries=[],
        ingest_node_health_findings=ingest,
    )
    training.run()
    assert spare is identity is None
    assert [name for name, _ in calls] == (
        []
        if finding == "none"
        else ["ingest"]
        if finding == "failure"
        else ["ingest", "mark"]
    )
    if finding == "failure":
        assert "training health scan failed" in caplog.text


def test_spare_health_worker_retries_provider_failure_without_disabling_safety(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    recorder = ThreadRecorder()
    monkeypatch.setattr(workers, "Thread", recorder)
    monkeypatch.setattr(workers, "training_health_monitor_enabled", lambda: False)
    calls = []

    def scan() -> None:
        calls.append("scan")
        if len(calls) == 1:
            raise RuntimeError("synthetic HMA health query failed")

    context = SimpleNamespace(
        xid_correlation=SimpleNamespace(run_forever=lambda: None),
        spare_health_controller=SimpleNamespace(scan=scan),
    )
    _xid, training, spare, _identity = workers.start_nonprocessor_workers(
        context=context,
        stop=Stop(1),
        identity_registries=[],
        ingest_node_health_findings=lambda *args: None,
    )
    assert training is None
    spare.run()
    assert calls == ["scan", "scan"]
    assert "HyperPod spare health scan failed" in caplog.text


def test_notification_worker_resets_backoff_after_success_and_retries_errors(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    recorder = ThreadRecorder()
    monkeypatch.setattr(workers, "Thread", recorder)
    monkeypatch.setattr(workers.random, "uniform", lambda low, high: 1.0)
    outcomes = iter([True, False, RuntimeError("unit failure")])
    calls = []
    backoffs = []

    def dispatch(owner: str, **kwargs: Any) -> Any:
        calls.append((owner, kwargs))
        value = next(outcomes)
        if isinstance(value, Exception):
            raise value
        return SimpleNamespace(throttled=value)

    def throttle(delay: float, **kwargs: Any) -> float:
        backoffs.append((delay, kwargs))
        return 4.0

    stop = Stop(2)
    worker = workers.start_notification_worker(
        context=SimpleNamespace(
            advisory_notifications=SimpleNamespace(dispatch_outbox=dispatch)
        ),
        stop=stop,
        owner="unit-owner",
        throttle_delay=throttle,
    )
    worker.run()
    assert stop.waits == [4.0, 2.0, 2.0]
    assert len(backoffs) == 1 and backoffs[0][0] == 0.0
    assert all(owner == "unit-owner" for owner, _ in calls), calls
    assert "notification dispatch cycle failed" in caplog.text


@pytest.mark.parametrize("listeners", [False, True])
def test_spool_workers_and_process_metrics_delegate_to_selected_services(
    monkeypatch: pytest.MonkeyPatch, listeners: bool
) -> None:
    recorder = ThreadRecorder()
    monkeypatch.setattr(workers, "Thread", recorder)
    calls = []
    store = SimpleNamespace()
    if listeners:
        store.listen_telemetry_spool_notifications = object()
    processor = SimpleNamespace(
        run_telemetry_spool=lambda: calls.append("spool"),
        run_telemetry_spool_notifications=lambda: calls.append("notifications"),
    )
    threads = workers.start_spool_threads(SimpleNamespace(store=store), processor)
    for thread in threads:
        thread.run()
    assert calls == (["notifications", "spool"] if listeners else ["spool"])
    stop = Stop()
    monkeypatch.setattr(
        workers.process_metrics,
        "publish_forever",
        lambda render, actual: calls.append((render(), actual)),
    )
    metrics = workers.start_process_metrics_worker(lambda: ["unit 1"], stop)
    metrics.run()
    assert calls[-1] == (["unit 1"], stop)

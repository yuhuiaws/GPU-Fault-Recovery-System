from __future__ import annotations

from types import SimpleNamespace

import pytest

from gpu_fault import completion_reconcile_loop as loop
from gpu_fault.completion_pod_parsing import CompletionControllerError
from tests.completion._cov95_runtime_support import (
    ManualTimer,
    StopLoop,
    make_controller,
)
from tests.completion._support import Clock, FakeCoreApi, FakeSink, FakeWatch, pod


@pytest.mark.parametrize("metrics_enabled", [True, False])
def test_polling_fallback_recovers_from_list_error_and_closes_metrics(
    monkeypatch, caplog, metrics_enabled
) -> None:
    class Core(FakeCoreApi):
        def list_pod_for_all_namespaces(self, **kwargs):
            if self.list_calls == 0:
                self.list_calls += 1
                raise RuntimeError("fake apiserver outage")
            return super().list_pod_for_all_namespaces(**kwargs)

    clock, core = Clock(), Core()
    controller = make_controller(core, now=clock)
    stopped, sleeps = [], []
    monkeypatch.setattr(
        loop,
        "start_completion_metrics_server",
        lambda *a, **kw: SimpleNamespace(stop=lambda: stopped.append(True))
        if metrics_enabled
        else None,
    )

    def sleep(seconds):
        sleeps.append(seconds)
        if len(sleeps) == 2:
            raise StopLoop

    monkeypatch.setattr(loop, "time", SimpleNamespace(sleep=sleep))
    with pytest.raises(StopLoop):
        controller.run(metrics_port=0)
    assert core.list_calls == 2
    assert sleeps == [5, 5]
    assert stopped == ([True] if metrics_enabled else [])
    assert controller.reconcile_runs_total == 1
    assert "using polling fallback" in caplog.text
    assert "Kubernetes completion reconciliation failed" in caplog.text


@pytest.mark.parametrize(
    "events,raises,diagnostic",
    [
        ([{"type": "ERROR", "object": {"code": 500}}], True, "Pod watch error"),
        ([{"type": "BOOKMARK", "object": None}], False, ""),
        ([{"type": "UNKNOWN", "object": None}], False, "ignoring unknown"),
        ([{"type": "ADDED", "object": {}}], True, "requires UID or name"),
        ([{"type": "DELETED", "object": {"metadata": {"uid": "unknown"}}}], False, ""),
    ],
)
def test_watch_protocol_errors_always_close_stream(events, raises, diagnostic, caplog):
    watcher = FakeWatch(events)
    controller = make_controller(
        watch_factory=lambda: watcher, reconcile_debounce_seconds=0
    )
    if raises:
        with pytest.raises(CompletionControllerError, match=diagnostic):
            controller.run_watch_cycle()
    else:
        controller.run_watch_cycle()
        if diagnostic:
            assert diagnostic in caplog.text
    assert watcher.stopped is True


def test_namespaced_watch_keys_pods_by_name_when_uid_is_missing() -> None:
    current = pod(0)
    current["metadata"].pop("uid")
    core = FakeCoreApi()

    def list_namespaced_pod(namespace, **kwargs):
        return {"items": [], "metadata": {"resourceVersion": "v1"}}

    core.list_namespaced_pod = list_namespaced_pod
    watcher = FakeWatch([{"type": "ADDED", "object": current}])
    sink = FakeSink()
    controller = make_controller(
        core,
        sink,
        namespace="training",
        watch_factory=lambda: watcher,
        reconcile_debounce_seconds=0,
    )
    controller.run_watch_cycle()
    assert watcher.arguments == (
        "list_namespaced_pod",
        {
            "namespace": "training",
            "label_selector": "gpu-fault.io/managed=true",
            "resource_version": "v1",
            "timeout_seconds": 30,
            "allow_watch_bookmarks": True,
            "_request_timeout": (5, 45),
        },
    )
    assert controller.reconciled_attempts_total == 1
    assert watcher.stopped is True
    assert any(path.endswith("workload-observations") for path, _ in sink.posts), (
        "the name-keyed Pod must reach the actual observer"
    )


@pytest.mark.parametrize("stuck", [True, False])
def test_timer_cleanup_is_bounded_and_flushes_each_affected_attempt(
    monkeypatch, caplog, stuck
):
    timers = []

    def timer(interval, callback):
        created = ManualTimer(interval, callback)
        created.stuck = stuck
        timers.append(created)
        return created

    monkeypatch.setattr(loop, "Timer", timer)
    original, changed = pod(0), pod(0, exit_code=0)
    watcher = FakeWatch([{"type": "MODIFIED", "object": changed}])
    sink = FakeSink()
    controller = make_controller(
        FakeCoreApi([original]), sink, watch_factory=lambda: watcher
    )
    controller.run_watch_cycle()
    assert len(timers) == 1
    assert timers[0].cancelled is True
    assert timers[0].daemon is True
    assert timers[0].joins == [controller.progress_stall_budget_seconds]
    assert watcher.stopped is True
    terminals = [payload for path, payload in sink.posts if path.endswith("/terminal")]
    assert [item["terminal_status"] for item in terminals] == ["SUCCEEDED"]
    assert ("still running after" in caplog.text) is stuck


def test_fired_timer_is_not_joined_and_next_event_gets_a_new_timer(monkeypatch):
    timers = []

    def timer(interval, callback):
        created = ManualTimer(interval, callback)
        timers.append(created)
        return created

    class Watch(FakeWatch):
        def stream(self, method, **kwargs):
            yield {"type": "ADDED", "object": pod(0)}
            timers[-1].fire()
            yield {"type": "MODIFIED", "object": pod(0, exit_code=0)}

    monkeypatch.setattr(loop, "Timer", timer)
    watcher = Watch([])
    controller = make_controller(watch_factory=lambda: watcher)
    controller.run_watch_cycle()
    assert len(timers) == 2
    assert timers[0].joins == []
    assert len(timers[1].joins) == 1
    assert controller.reconciled_attempts_total == 2
    assert watcher.stopped is True

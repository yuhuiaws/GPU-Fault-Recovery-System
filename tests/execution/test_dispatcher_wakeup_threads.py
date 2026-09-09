"""The worker role starts the dispatcher's two wakeup listener threads and
stops them with everything else (性能 A).

Both worker shapes scan: the processor Pod's ``run_dispatch`` loop and the
bare ``run_forever`` thread the lifespan starts when there is no processor.
Both consume the same wake event, so both get the listeners; the ingress role
never scans and never listens. Mirrors how the processor's queue listener is
started (only when the store has one) and how the notification dispatcher's
lifespan test proves shutdown by enumerating live threads.
"""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace

from fastapi.testclient import TestClient

from gpu_fault.app import ApplicationContext, create_app
from gpu_fault.app.lifespan_workers import (
    DISPATCHER_WAKEUP_THREAD_NAMES,
    start_dispatcher_wakeup_threads,
    start_processor_threads,
)
from gpu_fault.execution.config import WorkflowDispatcherConfig
from gpu_fault.execution.dispatcher import WorkflowDispatcher
from gpu_fault.store.contracts import WakeupChannel
from tests._builders import active_workflow_executor, build_store

WAIT = 5.0
TOKEN = "dispatcher-wakeup-lifespan-token-" + "x" * 32


def _live_wakeup_threads() -> dict[str, int]:
    live = {name: 0 for name in DISPATCHER_WAKEUP_THREAD_NAMES.values()}
    for thread in threading.enumerate():
        if thread.name in live:
            live[thread.name] += 1
    return live


def _dispatcher(store, *, enabled: bool = True) -> WorkflowDispatcher:
    return WorkflowDispatcher(
        store,
        active_workflow_executor(store, [], frozenset()),
        WorkflowDispatcherConfig(enabled=enabled, poll_interval_seconds=60),
    )


def _wait_connected(dispatcher: WorkflowDispatcher) -> None:
    deadline = time.monotonic() + WAIT
    while (
        not all(dispatcher.wakeup_listener_connected.values())
        and time.monotonic() < deadline
    ):
        time.sleep(0.005)
    assert all(dispatcher.wakeup_listener_connected.values()), (
        dispatcher.wakeup_listener_connected
    )


def test_one_named_thread_per_channel_stops_on_the_shared_stop_event() -> None:
    store = build_store()
    dispatcher = _dispatcher(store)
    stop = threading.Event()
    baseline = _live_wakeup_threads()

    threads = start_dispatcher_wakeup_threads(
        SimpleNamespace(store=store, dispatcher=dispatcher), stop
    )

    assert sorted(thread.name for thread in threads) == sorted(
        DISPATCHER_WAKEUP_THREAD_NAMES[channel] for channel in WakeupChannel
    )
    assert all(thread.daemon for thread in threads), (
        "wakeup threads must be daemons so they cannot block interpreter exit"
    )
    _wait_connected(dispatcher)
    live = _live_wakeup_threads()
    assert all(live[name] == baseline[name] + 1 for name in live), (live, baseline)

    stop.set()
    for thread in threads:
        thread.join(timeout=WAIT)
        assert not thread.is_alive(), f"{thread.name} outlived the stop event"
    assert _live_wakeup_threads() == baseline


def test_no_threads_for_a_disabled_dispatcher_or_a_store_without_wakeups() -> None:
    stop = threading.Event()
    store = build_store()

    assert (
        start_dispatcher_wakeup_threads(
            SimpleNamespace(store=store, dispatcher=_dispatcher(store, enabled=False)),
            stop,
        )
        == []
    )
    # Test doubles build contexts around a bare namespace; the processor's
    # queue listener is skipped for the same reason.
    assert (
        start_dispatcher_wakeup_threads(
            SimpleNamespace(store=SimpleNamespace(), dispatcher=_dispatcher(store)),
            stop,
        )
        == []
    )


def test_the_processor_worker_shape_starts_and_joins_the_listeners() -> None:
    """``start_processor_threads`` returns the listeners in the list the
    lifespan joins under the shutdown coordinator, so a listener that hung
    would be named in the shutdown failure instead of leaking silently."""

    stop = threading.Event()
    store = build_store()
    dispatcher = _dispatcher(store)
    processor = SimpleNamespace(
        is_healthy=lambda: True,
        active_consumers=True,
        is_leader=lambda: True,
        run_processor=lambda: stop.wait(),
        run_leadership=lambda: stop.wait(),
        telemetry_spool_enabled=False,
    )
    context = SimpleNamespace(
        dispatcher=dispatcher,
        xid_correlation=SimpleNamespace(
            run_once=lambda: None, poll_interval_seconds=60
        ),
        store=store,
        periodic_runner=None,
    )

    class _Runner:
        def __init__(self, **_kwargs) -> None:
            pass

        def run(self) -> None:
            stop.wait()

    import gpu_fault.app.lifespan_workers as workers

    original = workers.PeriodicServiceRunner
    workers.PeriodicServiceRunner = _Runner  # type: ignore[misc,assignment]
    try:
        _diagnostics, threads = start_processor_threads(
            context=context,
            processor=processor,
            diagnostics_publisher=SimpleNamespace(run=lambda _stop: None),
            stop=stop,
            identity_registries=[],
            ingest_node_health_findings=lambda *_args: None,
            notify_silent_collectors=lambda *_args: None,
        )
        names = {thread.name for thread in threads}
        assert set(DISPATCHER_WAKEUP_THREAD_NAMES.values()) <= names, names
        _wait_connected(dispatcher)
    finally:
        stop.set()
        workers.PeriodicServiceRunner = original  # type: ignore[misc]
    for thread in threads:
        thread.join(timeout=WAIT)
        assert not thread.is_alive(), thread.name


def test_the_worker_lifespan_without_a_processor_starts_and_stops_them(
    monkeypatch,
) -> None:
    monkeypatch.setenv("GPU_FAULT_SERVICE_ROLE", "worker")
    monkeypatch.setenv("POD_UID", "pod-dispatcher-wakeups")
    monkeypatch.delenv("GPU_FAULT_PROCESSOR_MODE", raising=False)
    monkeypatch.delenv("GPU_FAULT_TELEMETRY_SPOOL", raising=False)
    baseline = _live_wakeup_threads()
    context = ApplicationContext(
        store=build_store(),
        execution_token=TOKEN,
        dispatcher_config=WorkflowDispatcherConfig(
            enabled=True, poll_interval_seconds=60
        ),
    )
    app = create_app(context)

    with TestClient(app):
        _wait_connected(context.dispatcher)
        live = _live_wakeup_threads()
        assert all(live[name] == baseline[name] + 1 for name in live), (live, baseline)

    assert _live_wakeup_threads() == baseline, "a listener outlived its lifespan"
    assert context.dispatcher.wakeup_listener_connected == {
        "workflow_dispatch": False,
        "remote_command": False,
    }


def test_the_ingress_role_never_listens(monkeypatch) -> None:
    monkeypatch.setenv("GPU_FAULT_SERVICE_ROLE", "ingress")
    monkeypatch.setenv("POD_UID", "pod-dispatcher-wakeups-ingress")
    monkeypatch.delenv("GPU_FAULT_PROCESSOR_MODE", raising=False)
    baseline = _live_wakeup_threads()
    context = ApplicationContext(
        store=build_store(),
        execution_token=TOKEN,
        dispatcher_config=WorkflowDispatcherConfig(
            enabled=True, poll_interval_seconds=60
        ),
    )

    with TestClient(create_app(context)):
        time.sleep(0.1)
        assert _live_wakeup_threads() == baseline
        assert not any(context.dispatcher.wakeup_listener_connected.values()), (
            "the ingress role connected a wakeup listener"
        )

"""The per-cluster wakeup hub behind the long-poll claim.

One listener thread per process runs ``run_wakeup_listener`` on the
``REMOTE_COMMAND`` channel and resolves the futures of every claim request
waiting on that cluster. The hub is what lets the claim route hold a request
without holding a store I/O thread, and its liveness rule is what keeps a deaf
listener from turning a 25 s long-poll into a 25 s stall: while the listener
is not connected a wait is bounded by the old poll interval instead.
"""

from __future__ import annotations

import asyncio
import threading
import time
from typing import Any, Callable

from gpu_fault.app.remote_command_wakeups import (
    RemoteCommandWakeupHub,
    WakeupWaitOutcome,
)
from gpu_fault.store import InMemoryStore
from gpu_fault.store.contracts import WakeupChannel

WAIT = 3.0


class ScriptedStore:
    """A store whose listener the test drives by hand."""

    def __init__(self, *, connect: bool = True) -> None:
        self.connect = connect
        self.started = threading.Event()
        self.notify: Callable[[dict[str, Any]], None] | None = None
        self.state: Callable[[bool], None] | None = None
        self.channels: list[WakeupChannel] = []
        self.timeouts: list[float] = []

    def run_wakeup_listener(
        self,
        channel: WakeupChannel,
        stop_event: threading.Event,
        on_notification: Callable[[dict[str, Any]], None],
        *,
        timeout_seconds: float = 1.0,
        on_state: Callable[[bool], None] | None = None,
    ) -> None:
        self.channels.append(channel)
        self.timeouts.append(timeout_seconds)
        self.notify = on_notification
        self.state = on_state
        if self.connect and on_state is not None:
            on_state(True)
        self.started.set()
        stop_event.wait()
        if on_state is not None:
            on_state(False)

    def pending(self, cluster_id: str, command_id: str = "cmd") -> None:
        assert self.notify is not None, "listener has not started"
        self.notify(
            {
                "command_id": command_id,
                "cluster_id": cluster_id,
                "workflow_request_id": f"wf-{command_id}",
                "status": "PENDING",
            }
        )


def _hub(store, **overrides) -> RemoteCommandWakeupHub:
    return RemoteCommandWakeupHub(store, listener_timeout_seconds=0.05, **overrides)


def _run(coroutine):
    return asyncio.run(coroutine)


def test_the_hub_is_idle_until_the_first_subscription_starts_the_listener():
    store = ScriptedStore()
    hub = _hub(store)
    try:
        assert hub.listener_alive is False
        assert hub.connected is False

        async def scenario():
            async with hub.subscribe("cluster-a"):
                assert store.started.wait(WAIT), "the listener thread never started"

        _run(scenario())
        assert hub.listener_alive is True
        assert store.channels == [WakeupChannel.REMOTE_COMMAND]
        assert store.timeouts == [0.05]
    finally:
        hub.close()
    assert hub.listener_alive is False


def test_a_pending_notification_for_the_cluster_wakes_the_waiter_at_once():
    store = ScriptedStore()
    hub = _hub(store)
    try:

        async def scenario():
            async with hub.subscribe("cluster-a") as waiter:
                assert waiter.admitted is True
                assert store.started.wait(WAIT), "the listener thread never started"
                started = time.monotonic()
                threading.Timer(0.05, store.pending, ["cluster-a"]).start()
                outcome = await waiter.wait(5.0)
                return outcome, time.monotonic() - started

        outcome, elapsed = _run(scenario())
        assert outcome is WakeupWaitOutcome.WOKEN
        assert elapsed < 1.0, f"woke after {elapsed:.2f}s, not on the notification"
    finally:
        hub.close()


def test_other_clusters_and_non_pending_transitions_do_not_wake_the_waiter():
    store = ScriptedStore()
    hub = _hub(store)
    try:

        async def scenario():
            async with hub.subscribe("cluster-a") as waiter:
                assert store.started.wait(WAIT), "the listener thread never started"
                store.pending("cluster-b")
                assert store.notify is not None
                store.notify(
                    {
                        "command_id": "cmd-leased",
                        "cluster_id": "cluster-a",
                        "workflow_request_id": "wf",
                        "status": "LEASED",
                    }
                )
                store.notify({"garbage": True})
                return await waiter.wait(0.2)

        assert _run(scenario()) is WakeupWaitOutcome.TIMEOUT
    finally:
        hub.close()


def test_a_wait_is_capped_at_the_server_maximum():
    store = ScriptedStore()
    hub = _hub(store, max_wait_seconds=0.1)
    try:

        async def scenario():
            async with hub.subscribe("cluster-a") as waiter:
                assert store.started.wait(WAIT), "the listener thread never started"
                started = time.monotonic()
                outcome = await waiter.wait(30.0)
                return outcome, time.monotonic() - started

        outcome, elapsed = _run(scenario())
        assert outcome is WakeupWaitOutcome.TIMEOUT
        assert elapsed < 1.0
    finally:
        hub.close()


def test_a_disconnected_listener_bounds_the_wait_to_the_old_poll_interval():
    """Liveness rule: deaf listener degrades to polling, never to a stall."""

    store = ScriptedStore(connect=False)
    hub = _hub(store, degraded_wait_seconds=0.2)
    try:

        async def scenario():
            async with hub.subscribe("cluster-a") as waiter:
                assert store.started.wait(WAIT), "the listener thread never started"
                assert hub.connected is False
                started = time.monotonic()
                outcome = await waiter.wait(10.0)
                return outcome, time.monotonic() - started

        outcome, elapsed = _run(scenario())
        assert outcome is WakeupWaitOutcome.TIMEOUT
        assert elapsed < 1.0, f"a deaf listener held the request {elapsed:.2f}s"
    finally:
        hub.close()


def test_the_default_degraded_bound_is_the_old_poll_interval():
    hub = RemoteCommandWakeupHub(ScriptedStore())
    assert hub.degraded_wait_seconds == 2.0
    assert hub.max_wait_seconds == 25.0
    assert hub.max_waiters_per_cluster == 4


def test_a_listener_that_drops_mid_wait_releases_the_waiters():
    store = ScriptedStore()
    hub = _hub(store)
    try:

        async def scenario():
            async with hub.subscribe("cluster-a") as waiter:
                assert store.started.wait(WAIT), "the listener thread never started"
                assert store.state is not None
                threading.Timer(0.05, store.state, [False]).start()
                started = time.monotonic()
                outcome = await waiter.wait(10.0)
                return outcome, time.monotonic() - started

        outcome, elapsed = _run(scenario())
        assert outcome is WakeupWaitOutcome.DISCONNECTED
        assert elapsed < 1.0
        assert hub.connected is False
    finally:
        hub.close()


def test_waiters_beyond_the_per_cluster_cap_are_not_admitted():
    store = ScriptedStore()
    hub = _hub(store, max_waiters_per_cluster=2)
    try:

        async def scenario():
            async with (
                hub.subscribe("cluster-a") as first,
                hub.subscribe("cluster-a") as second,
                hub.subscribe("cluster-a") as third,
                hub.subscribe("cluster-b") as other,
            ):
                admitted = (
                    first.admitted,
                    second.admitted,
                    third.admitted,
                    other.admitted,
                )
                started = time.monotonic()
                outcome = await third.wait(10.0)
                return admitted, outcome, time.monotonic() - started

        admitted, outcome, elapsed = _run(scenario())
        assert admitted == (True, True, False, True)
        assert outcome is WakeupWaitOutcome.NOT_ADMITTED
        assert elapsed < 0.5

        async def after_release():
            async with hub.subscribe("cluster-a") as waiter:
                return waiter.admitted

        assert _run(after_release()) is True, "a released slot is reusable"
    finally:
        hub.close()


def test_a_store_without_a_wakeup_listener_never_connects():
    hub = RemoteCommandWakeupHub(object(), degraded_wait_seconds=0.1)
    try:

        async def scenario():
            async with hub.subscribe("cluster-a") as waiter:
                started = time.monotonic()
                outcome = await waiter.wait(10.0)
                return outcome, time.monotonic() - started

        outcome, elapsed = _run(scenario())
        assert outcome is WakeupWaitOutcome.TIMEOUT
        assert elapsed < 1.0
        assert hub.listener_alive is False
    finally:
        hub.close()


def test_close_releases_waiters_and_stops_the_thread():
    store = ScriptedStore()
    hub = _hub(store)

    async def scenario():
        async with hub.subscribe("cluster-a") as waiter:
            assert store.started.wait(WAIT), "the listener thread never started"
            threading.Timer(0.05, hub.close).start()
            return await waiter.wait(10.0)

    assert _run(scenario()) is WakeupWaitOutcome.CLOSED
    assert hub.listener_alive is False

    async def after_close():
        async with hub.subscribe("cluster-a") as waiter:
            return waiter.admitted, await waiter.wait(10.0)

    assert _run(after_close()) == (False, WakeupWaitOutcome.NOT_ADMITTED)
    assert hub.listener_alive is False, "a closed hub does not restart"


def test_the_hub_wraps_a_lifespan_and_closes_on_exit():
    store = ScriptedStore()
    hub = _hub(store)
    events: list[str] = []

    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def inner(app):
        events.append("start")
        yield
        events.append("stop")

    wrapped = hub.wrap_lifespan(inner)

    async def scenario():
        async with wrapped("app"):
            async with hub.subscribe("cluster-a"):
                assert store.started.wait(WAIT), "the listener thread never started"
            events.append("body")
        return hub.listener_alive

    assert _run(scenario()) is False
    assert events == ["start", "body", "stop"]


def test_the_real_memory_store_wakes_a_waiter_through_the_shared_hub():
    """End to end on a real backend: an ensure_remote_command wakes the wait."""

    from tests.regional._regional_support import enqueue_remote_command

    store = InMemoryStore()
    hub = _hub(store)
    try:

        async def scenario():
            async with hub.subscribe("cluster-a") as waiter:
                deadline = time.monotonic() + WAIT
                while not hub.connected and time.monotonic() < deadline:
                    await asyncio.sleep(0.01)
                assert hub.connected is True
                threading.Timer(
                    0.05, enqueue_remote_command, [store, "remote-" + "a" * 24]
                ).start()
                started = time.monotonic()
                outcome = await waiter.wait(5.0)
                return outcome, time.monotonic() - started

        outcome, elapsed = _run(scenario())
        assert outcome is WakeupWaitOutcome.WOKEN
        assert elapsed < 1.0
    finally:
        hub.close()

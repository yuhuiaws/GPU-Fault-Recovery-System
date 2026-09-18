from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from threading import Event, Lock

import pytest

from gpu_fault.store import InMemoryStore
from gpu_fault.store.contracts import WakeupChannel
from gpu_fault.store.postgres.processor_completion_runtime import (
    complete_cluster_groups,
)
from gpu_fault.store.shared.wakeups import SUBSCRIBER_BOUND, ObservedRows, WakeupHub
from tests._builders import workflow_request


@pytest.mark.parametrize("parallel", [False, True])
@pytest.mark.parametrize(
    "failures", [(), ("a",), ("a", "c")], ids=["success", "one-failure", "two-failures"]
)
def test_cluster_completion_drains_every_group_and_preserves_first_failure(
    parallel, failures
):
    visited = []
    lock = Lock()
    groups = {name: [name] for name in ("c", "a", "b")}

    def complete(batch):
        (name,) = batch
        with lock:
            visited.append(name)
        if name in failures:
            raise RuntimeError(f"failed-{name}")
        return {name: name}

    with (
        ThreadPoolExecutor(max_workers=2) if parallel else nullcontext(None) as executor
    ):
        if failures:
            with pytest.raises(RuntimeError, match="failed-a"):
                complete_cluster_groups(
                    groups,
                    concurrency=2 if parallel else 1,
                    executor=executor,
                    complete=complete,
                )
        else:
            assert complete_cluster_groups(
                groups,
                concurrency=2 if parallel else 1,
                executor=executor,
                complete=complete,
            ) == {"a": "a", "b": "b", "c": "c"}
    assert sorted(visited) == ["a", "b", "c"], (
        "a failed cluster must not strand other group work or leave futures undrained"
    )


def test_completion_without_an_executor_uses_the_serial_fallback():
    visited = []

    def complete(batch):
        visited.extend(batch)
        return {batch[0]: batch[0]}

    assert complete_cluster_groups(
        {"b": ["b"], "a": ["a"]}, concurrency=3, executor=None, complete=complete
    ) == {"a": "a", "b": "b"}
    assert visited == ["a", "b"], "serial fallback must preserve stable cluster order"
    assert (
        complete_cluster_groups({}, concurrency=3, executor=None, complete=complete)
        == {}
    )


def test_wakeup_subscriptions_are_bounded_isolated_and_removed_on_exit():
    hub = WakeupHub()
    channel = WakeupChannel.WORKFLOW_DISPATCH
    with hub.subscribe(channel) as first, hub.subscribe(channel) as second:
        with hub.subscribe(WakeupChannel.REMOTE_COMMAND) as other:
            for index in range(SUBSCRIBER_BOUND + 1):
                hub.publish(channel, {"index": index})
            expected = [{"index": index} for index in range(1, SUBSCRIBER_BOUND + 1)]
            assert first.drain() == expected, (
                "a stalled subscriber must drop its oldest hint"
            )
            assert second.drain() == expected, (
                "subscribers must not consume one another's hints"
            )
            assert other.drain() == [], "wakeup channels must remain isolated"
            assert first.drain() == [], "draining must clear the local hint queue"
    hub.publish(channel, {"index": "after-exit"})
    assert first.drain() == second.drain() == [], (
        "exited subscriptions must not keep receiving references or accumulating hints"
    )


def test_observed_rows_setdefault_and_bulk_updates_publish_only_actual_writes():
    hub = WakeupHub()
    channel = WakeupChannel.WORKFLOW_DISPATCH
    rows = ObservedRows(
        hub,
        channel,
        lambda previous, value: None if previous == value else {"value": value},
    )
    with hub.subscribe(channel) as subscription:
        assert rows.setdefault("first", 1) == 1
        assert rows.setdefault("first", 99) == 1
        rows.update({"first": 1, "second": 2}, third=3)
        assert subscription.drain() == [{"value": 1}, {"value": 2}, {"value": 3}], (
            "setdefault and update must preserve trigger-equivalent no-op suppression"
        )
        assert dict(rows) == {"first": 1, "second": 2, "third": 3}


def test_in_process_listener_deduplicates_ready_batch_and_reports_shutdown():
    store = InMemoryStore()
    stop = Event()
    states, received = [], []
    first = workflow_request("first", "incident-first")
    second = workflow_request("second", "incident-second")

    def ready(connected):
        states.append(connected)
        if connected:
            store.save_workflow(first)
            store.save_workflow(
                first.model_copy(update={"merge_revision": 1}), expected=first
            )
            store.save_workflow(second)

    def notified(payload):
        received.append(payload["request_id"])
        if payload["request_id"] == "second":
            stop.set()

    store.run_wakeup_listener(
        WakeupChannel.WORKFLOW_DISPATCH,
        stop,
        notified,
        on_state=ready,
        timeout_seconds=0.1,
    )
    assert received == ["first", "second"], (
        "one ready batch must deduplicate equivalent hints"
    )
    assert states == [True, False], "shutdown must be reported after unsubscribing"


def test_in_process_listener_can_start_stopped_without_a_state_callback():
    store = InMemoryStore()
    stop = Event()
    stop.set()
    store.run_wakeup_listener(
        WakeupChannel.REMOTE_COMMAND,
        stop,
        lambda payload: pytest.fail("a stopped listener must not deliver hints"),
    )
    with pytest.raises(ValueError, match="timeout must be positive"):
        store.run_wakeup_listener(
            WakeupChannel.REMOTE_COMMAND, stop, lambda payload: None, timeout_seconds=0
        )

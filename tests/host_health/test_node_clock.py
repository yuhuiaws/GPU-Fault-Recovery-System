"""Sustained-signal durations run on the control plane's clock, and a node
clock that steps backwards is noted, not obeyed (F-M2).

"Active for N seconds" used to be measured between two node-reported
timestamps: an NTP step or a VM migration made a five-minute rule true at
the first sample and triggered a reboot-grade action, and a backward step
made every later sample of that node's health signals vanish silently.
The store now measures on the time it received the samples when the caller
gives it, keeps the node time for display, and counts regressions.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from gpu_fault.host_health import HostTelemetryBatch
from tests._builders import build_store

KEY = "cluster-a/node-a/memory_available_percent/node"
T0 = datetime(2026, 9, 6, 12, 0, tzinfo=timezone.utc)
R0 = datetime(2026, 9, 6, 12, 0, 5, tzinfo=timezone.utc)


def _claim(store, observed_at, received_at, *, active=True, minimum=300.0):
    return store.claim_health_signal_transitions(
        [(KEY, active, observed_at, minimum)], received_at=received_at
    )[0]


def test_a_forward_clock_jump_does_not_confirm_a_sustained_rule_early():
    store = build_store()
    assert _claim(store, T0, R0) is False
    # The node's clock jumped ten minutes; the control plane saw one second.
    assert _claim(store, T0 + timedelta(minutes=10), R0 + timedelta(seconds=1)) is False
    # Once the control plane has seen the signal held for the window, it fires.
    assert (
        _claim(store, T0 + timedelta(minutes=11), R0 + timedelta(seconds=301)) is True
    )


def test_a_backward_node_clock_step_is_counted_and_the_signal_keeps_flowing():
    store = build_store()
    assert _claim(store, T0, R0) is False
    stepped_back = T0 - timedelta(minutes=2)

    emitted = _claim(store, stepped_back, R0 + timedelta(seconds=301))

    assert emitted is True
    assert store.health_signal_clock_regressions_total == 1


def test_without_a_receive_time_the_node_clock_still_drives_the_window():
    store = build_store()
    assert _claim(store, T0, None) is False
    assert _claim(store, T0 + timedelta(seconds=301), None) is True
    # And a true duplicate on that clock is still dropped.
    assert _claim(store, T0 + timedelta(seconds=301), None) is False


def test_the_batch_carries_the_time_it_was_received():
    batch = HostTelemetryBatch(
        cluster_id="cluster-a", node_id="node-a", observed_at=T0, samples=[]
    )
    assert batch.received_at is None
    stamped = batch.model_copy(update={"received_at": R0})
    assert stamped.received_at == R0

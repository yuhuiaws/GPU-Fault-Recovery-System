from __future__ import annotations

from datetime import timedelta

import pytest

from gpu_fault.models import EfaTrafficAdminAction, EfaTrafficSignal
from gpu_fault.policy import XidCorrelationRecord, XidCorrelationStatus, XidEvent
from gpu_fault.store import NotFoundError
from gpu_fault.store.shared.errors import EfaTrafficAdminConflict
from tests.store._cov95_compat_support import NOW
from tests.store._cov95_compat_support import (
    compat_store_fixture as compat_store_fixture,
)


def event(event_id, *, at=NOW, node="node-a", cluster="cluster-a", code=63, **values):
    return XidEvent(
        event_id=event_id,
        cluster_id=cluster,
        node_id=node,
        observed_at=at,
        xid=code,
        **values,
    )


def test_xid_retention_and_time_bounds_do_not_cross_node_or_cluster(compat_store):
    store = compat_store
    old = event("old", at=NOW - timedelta(seconds=10))
    foreign_node = event("foreign-node", at=old.observed_at, node="node-b")
    foreign_cluster = event("foreign-cluster", at=old.observed_at, cluster="cluster-b")
    for item in (old, foreign_node, foreign_cluster):
        assert store.save_xid_event_if_absent(item) is True
    current = event("current")
    assert (
        store.save_xid_event_if_absent(current, retain_from=NOW - timedelta(seconds=1))
        is True
    )
    assert (
        store.save_xid_event_if_absent(current.model_copy(update={"xid": 79})) is False
    )
    with pytest.raises(NotFoundError):
        store.get_xid_event("old")
    assert store.get_xid_event("current").xid == 63
    assert store.list_xid_events(
        "cluster-a", "node-a", observed_after=NOW, observed_before=NOW
    ) == [current]
    assert (
        store.list_xid_events(
            "cluster-a", "node-a", observed_after=NOW + timedelta(seconds=1)
        )
        == []
    )
    assert store.list_xid_events("cluster-a", "node-b") == [foreign_node]
    assert store.list_xid_events("cluster-b", "node-a") == [foreign_cluster]


def test_due_correlation_lease_takeover_rejects_the_previous_owner(compat_store):
    store = compat_store
    with pytest.raises(NotFoundError):
        store.get_xid_correlation("missing")
    pending = XidCorrelationRecord(event_id="pending", deadline=NOW)
    assert store.save_xid_correlation_if_absent(pending) is True
    assert (
        store.save_xid_correlation_if_absent(
            pending.model_copy(update={"deadline": NOW + timedelta(days=1)})
        )
        is False
    )
    arguments = {"lease_duration": timedelta(seconds=30), "limit": 1}
    assert (
        store.claim_due_xid_correlations(
            owner="a", now=NOW - timedelta(seconds=1), **arguments
        )
        == []
    )
    (claimed,) = store.claim_due_xid_correlations(owner="a", now=NOW, **arguments)
    assert claimed.lease_owner == "a"
    assert (
        store.claim_due_xid_correlations(
            owner="b", now=NOW + timedelta(seconds=1), **arguments
        )
        == []
    )
    (replacement,) = store.claim_due_xid_correlations(
        owner="b", now=NOW + timedelta(seconds=31), **arguments
    )
    assert replacement.event_id == pending.event_id
    with pytest.raises(ValueError, match="owner changed"):
        store.complete_xid_correlation(
            "pending", owner="a", now=NOW + timedelta(seconds=31)
        )
    completed = store.complete_xid_correlation(
        "pending", owner="b", now=NOW + timedelta(seconds=32)
    )
    assert completed.status is XidCorrelationStatus.FINALIZED
    assert completed.lease_owner is None
    assert (
        store.claim_due_xid_correlations(
            owner="c", now=NOW + timedelta(days=1), **arguments
        )
        == []
    )


def test_xid_occurrence_identity_uses_link_bit_and_event_not_arrival_count(
    compat_store,
):
    store = compat_store
    missing_link = event("missing-link", code=74, gpu_uuid="GPU-a", registers=[16])
    assert store.record_xid74_occurrences(missing_link) == {}
    assert store.record_xid74_occurrences(event("not-74")) == {}
    first = event(
        "first",
        code=74,
        gpu_uuid="GPU-a",
        registers=[16, 32],
        nvlink_link_id=2,
        nvlink_link_identity_source="compat-explicit",
    )
    assert store.record_xid74_occurrences(first) == {
        "register1.bit4": 1,
        "register2.bit5": 1,
    }
    assert store.record_xid74_occurrences(first) == {
        "register1.bit4": 1,
        "register2.bit5": 1,
    }
    second = first.model_copy(
        update={
            "event_id": "second",
            "registers": [16],
            "observed_at": NOW + timedelta(seconds=1),
        }
    )
    assert store.record_xid74_occurrences(second) == {"register1.bit4": 2}
    other_link = first.model_copy(
        update={"event_id": "other-link", "nvlink_link_id": 3}
    )
    assert store.record_xid74_occurrences(other_link) == {
        "register1.bit4": 1,
        "register2.bit5": 1,
    }


def test_xid_metric_clear_rearms_a_transition_without_replaying_stale_data(
    compat_store,
):
    store = compat_store
    observed = [
        store.observe_xid_metric(
            "cluster", "node", "GPU-a", code, NOW + timedelta(seconds=index)
        )
        for index, code in enumerate((0, 79, 79, 0, 79))
    ]
    assert observed == [False, True, False, False, True]
    assert store.observe_xid_metric("cluster", "node", "GPU-a", 94, NOW) is False
    assert store.observe_xid_metric("other", "node", "GPU-a", 79, NOW) is False
    assert store.observe_xid_metric("cluster", "other-node", "GPU-a", 79, NOW) is False


def traffic(store, *, seconds=0, bps=1000):
    return store.observe_efa_traffic(
        state_key="cluster/node/job/attempt",
        cluster_id="cluster",
        node_id="node",
        job_id="job",
        attempt_id="attempt",
        observed_at=NOW + timedelta(seconds=seconds),
        bytes_per_second=bps,
        minimum_active_bps=100,
        spike_ratio=2,
        drop_ratio=0.5,
        zero_bps=0,
        zero_warning_seconds=20,
        zero_hung_seconds=60,
        baseline_alpha=0.25,
        startup_grace_seconds=30,
        spike_event_id="compat-spike",
    )


@pytest.mark.parametrize(
    "action",
    [
        EfaTrafficAdminAction.ACKNOWLEDGE_TRANSIENT,
        EfaTrafficAdminAction.ACCEPT_NEW_BASELINE,
    ],
)
def test_efa_admin_decision_is_persisted_and_replay_keeps_the_original_actor(
    compat_store, action
):
    store = compat_store
    with pytest.raises(NotFoundError):
        store.get_efa_traffic_state("cluster/node/job/attempt")
    traffic(store)
    state, emitted = traffic(store, seconds=40, bps=3000)
    assert emitted is True
    assert state.signal is EfaTrafficSignal.SPIKE
    first = store.apply_efa_traffic_admin_action(
        state_key=state.state_key,
        event_id="compat-spike",
        action=action,
        operator="operator-a",
        reason="compatibility approval",
        decided_at=NOW + timedelta(seconds=41),
    )
    replay = store.apply_efa_traffic_admin_action(
        state_key=state.state_key,
        event_id="compat-spike",
        action=action,
        operator="operator-b",
        reason="different replay",
        decided_at=NOW + timedelta(seconds=42),
    )
    assert replay == first
    assert replay.operator == "operator-a"
    stored = store.get_efa_traffic_state(state.state_key)
    if action is EfaTrafficAdminAction.ACCEPT_NEW_BASELINE:
        assert stored.baseline_bytes_per_second == 3000
        assert stored.signal is EfaTrafficSignal.NORMAL
    else:
        assert stored.baseline_bytes_per_second == 1000
        assert stored.spike_acknowledged_by == "operator-a"


def test_efa_decision_rejects_missing_state_or_a_different_active_event(compat_store):
    store = compat_store
    arguments = {
        "event_id": "not-active",
        "action": EfaTrafficAdminAction.ACKNOWLEDGE_TRANSIENT,
        "operator": "operator",
        "reason": "unit request",
        "decided_at": NOW,
    }
    with pytest.raises(NotFoundError):
        store.apply_efa_traffic_admin_action(state_key="missing", **arguments)
    traffic(store)
    active, _ = traffic(store, seconds=40, bps=3000)
    with pytest.raises(EfaTrafficAdminConflict, match="active SPIKE"):
        store.apply_efa_traffic_admin_action(state_key=active.state_key, **arguments)
    assert store.get_efa_traffic_state(active.state_key) == active


def test_receive_clock_orders_health_signals_without_obeying_a_regressed_node_clock(
    compat_store,
):
    key = "cluster-local/node-local/link/eth0"
    assert compat_store.claim_health_signal_transitions(
        [(key, False, NOW, 10.0)], received_at=NOW
    ) == [False], "an inactive first observation cannot create a notification episode"
    assert compat_store.get_health_signal_state(key) is None, (
        "an inactive first observation must not create persistent signal state"
    )
    assert compat_store.claim_health_signal_transitions(
        [(key, True, NOW, 10.0)], received_at=NOW
    ) == [False], "a new active episode must satisfy its sustained-duration threshold"

    assert compat_store.claim_health_signal_transitions(
        [(key, True, NOW - timedelta(seconds=1), 10.0)],
        received_at=NOW + timedelta(seconds=5),
    ) == [False], (
        "a regressed node timestamp must not prematurely satisfy the receive clock"
    )
    current = compat_store.get_health_signal_state(key)
    assert current is not None and current.active_since == NOW, (
        "a backwards node clock must not restart the active episode"
    )
    assert current.clock_at == NOW + timedelta(seconds=5), (
        "the persisted ordering clock must follow control-plane receipt time"
    )
    assert compat_store.health_signal_clock_regressions_total == 1, (
        "accepted backwards node time must still be reported as a regression"
    )
    assert compat_store.claim_health_signal_transitions(
        [(key, False, NOW + timedelta(days=1), 10.0)],
        received_at=NOW + timedelta(seconds=4),
    ) == [False], "a later node timestamp cannot authorize an out-of-order delivery"
    assert compat_store.get_health_signal_state(key) == current, (
        "a rejected out-of-order clear must not end the current active episode"
    )
    assert compat_store.claim_health_signal_transitions(
        [(key, True, NOW - timedelta(seconds=2), 10.0)],
        received_at=NOW + timedelta(seconds=10),
    ) == [True], (
        "the exact sustained-duration boundary must emit despite node clock drift"
    )
    ready = compat_store.get_health_signal_state(key)
    assert ready is not None and ready.notified is not True, (
        "eligibility is not proof that the episode's notification has been delivered"
    )

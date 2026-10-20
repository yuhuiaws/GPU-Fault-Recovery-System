"""One EFA traffic row per node; its state stays one attempt's (batch 145).

``efa_traffic_bytes_per_second`` is a node-wide aggregate and the rule runs only
while exactly one managed attempt resolves on the node, so the state lives in
one row per node, ``cluster/node``, owned by the attempt in its payload. It used
to be one row per ``cluster/node/job/attempt``, never removed. The state is
still per attempt: a sample of another attempt starts it at WARMUP, never with
the previous attempt's baseline, zero streak or spike acknowledgement.

Compatibility. Upgrade: an attempt the node row does not carry continues from
its per-attempt row, which is then deleted. Rollback: the older release reads
only per-attempt rows; with those gone it starts every attempt at WARMUP --
detection waits for a first active sample and the startup grace, it never
alarms early. A per-attempt row that survived (one an older writer re-created
mid-roll) is the exception, and the retention removes it once the grace passed
without a write (``tests/store/test_health_signal_retention.py``).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from gpu_fault.efa_traffic_state import (
    efa_traffic_state_key,
    legacy_efa_traffic_state_key,
)
from gpu_fault.models import EfaTrafficAdminAction, EfaTrafficSignal
from gpu_fault.store import SqliteStore
from gpu_fault.store.shared.errors import EfaTrafficAdminConflict, NotFoundError
from tests._builders import build_store

CLUSTER = "cluster-a"
NOW = datetime(2030, 1, 1, 12, 0, tzinfo=timezone.utc)
NODE = efa_traffic_state_key(CLUSTER, "node-a")
LEGACY = legacy_efa_traffic_state_key(NODE, "job-a", "attempt-1")
T0 = NOW - timedelta(hours=3)


def efa(
    store,
    node_id: str,
    at: datetime,
    *,
    job_id: str = "job-a",
    attempt_id: str = "attempt-1",
    state_key: str | None = None,
    bps: float = 1000.0,
):
    """One sample as ``NodeHealthPolicy`` reports it; ``state_key`` names the
    row a release before batch 145 wrote (``cluster/node/job/attempt``)."""

    return store.observe_efa_traffic(
        state_key=state_key or efa_traffic_state_key(CLUSTER, node_id),
        cluster_id=CLUSTER,
        node_id=node_id,
        job_id=job_id,
        attempt_id=attempt_id,
        observed_at=at,
        bytes_per_second=bps,
        minimum_active_bps=100.0,
        spike_ratio=4.0,
        drop_ratio=0.25,
        zero_bps=1.0,
        zero_warning_seconds=60.0,
        zero_hung_seconds=600.0,
        baseline_alpha=0.2,
        startup_grace_seconds=120.0,
        spike_event_id=f"{node_id}-{at.isoformat()}-efa-traffic-spike",
    )


@pytest.fixture(params=["memory", "sqlite"])
def store(request, tmp_path):
    if request.param == "memory":
        yield build_store()
        return
    instance = SqliteStore(str(tmp_path / "efa-node-state.db"))
    try:
        yield instance
    finally:
        instance.close()


def _at(seconds: float):
    return T0 + timedelta(seconds=seconds)


def _hung_streak(store, *, state_key: str = NODE, attempt_id: str = "attempt-1"):
    """Traffic, then zero long enough to be ZERO_WARNING; returns the state."""

    for second in (0, 130, 140):
        efa(store, "node-a", _at(second), attempt_id=attempt_id, state_key=state_key)
    state, _ = efa(
        store, "node-a", _at(150), attempt_id=attempt_id, state_key=state_key, bps=0
    )
    state, emitted = efa(
        store, "node-a", _at(220), attempt_id=attempt_id, state_key=state_key, bps=0
    )
    assert (state.signal, emitted) == (EfaTrafficSignal.ZERO_WARNING, True)
    return state


def test_one_row_per_node_and_a_new_attempt_starts_at_warmup(store) -> None:
    previous = _hung_streak(store)
    assert previous.zero_since is not None and previous.baseline_bytes_per_second

    state, emitted = efa(store, "node-a", _at(230), attempt_id="attempt-2", bps=0)

    assert emitted is False
    assert (state.job_id, state.attempt_id, state.state_key) == (
        "job-a",
        "attempt-2",
        NODE,
    )
    assert state.signal is EfaTrafficSignal.WARMUP
    assert state.baseline_bytes_per_second is None
    assert state.zero_since is None and state.had_active_traffic is False
    assert state.first_observed_at == _at(230)
    assert store.get_efa_traffic_state(NODE) == state
    with pytest.raises(NotFoundError):
        store.get_efa_traffic_state(NODE, job_id="job-a", attempt_id="attempt-1")
    assert store.get_efa_traffic_state(
        NODE, job_id="job-a", attempt_id="attempt-2"
    ) == (state)


def test_a_late_sample_of_the_previous_attempt_changes_nothing(store) -> None:
    efa(store, "node-a", _at(0))
    current, _ = efa(store, "node-a", _at(60), attempt_id="attempt-2")

    late, emitted = efa(store, "node-a", _at(30), attempt_id="attempt-1", bps=0)

    assert emitted is False
    assert late == current
    assert store.get_efa_traffic_state(NODE) == current


def test_an_upgrade_carries_a_running_attempts_state_over_from_its_row(store) -> None:
    """The release before wrote ``cluster/node/job/attempt``; the attempt keeps
    its baseline and zero streak, and that row is gone afterwards."""

    before = _hung_streak(store, state_key=LEGACY)
    assert store.get_efa_traffic_state(LEGACY) == before

    state, emitted = efa(store, "node-a", _at(800), bps=0)

    assert (state.signal, emitted) == (EfaTrafficSignal.HUNG_SUSPECTED, True)
    assert state.zero_since == before.zero_since
    assert state.baseline_bytes_per_second == before.baseline_bytes_per_second
    assert state.state_key == NODE
    with pytest.raises(NotFoundError):
        store.get_efa_traffic_state(LEGACY)


def test_a_rollback_finds_no_per_attempt_row_and_starts_at_warmup(store) -> None:
    """The older release reads ``cluster/node/job/attempt`` only."""

    _hung_streak(store, state_key=LEGACY)
    efa(store, "node-a", _at(300), bps=0)  # migrated by this release

    # The older release again, on the same attempt: no row, so WARMUP; zero
    # traffic cannot escalate before a first active sample.
    for second in (310, 1000, 2000):
        state, emitted = efa(store, "node-a", _at(second), state_key=LEGACY, bps=0)
        assert (state.signal, emitted) == (EfaTrafficSignal.WARMUP, False)


def test_a_surviving_per_attempt_row_is_what_a_rollback_would_resume(store) -> None:
    """Why the retention removes superseded rows: the older release resumes a
    zero streak recorded hours ago and escalates on its first zero sample."""

    _hung_streak(store, state_key=LEGACY)
    stale = efa(store, "node-a", _at(230), state_key=LEGACY, bps=0)[0]
    assert stale.zero_since == _at(150)

    resumed, emitted = efa(store, "node-a", NOW, state_key=LEGACY, bps=0)
    assert (resumed.signal, emitted) == (EfaTrafficSignal.HUNG_SUSPECTED, True)


def test_the_hung_triage_reads_only_the_incidents_attempt(store) -> None:
    """``_hung_efa_zero_pending_times`` asks for the incident's attempt: the
    node row once another attempt took it over is not that attempt's state,
    and a state still in its per-attempt row is found there."""

    _hung_streak(store, state_key=LEGACY)
    assert store.get_efa_traffic_state(
        NODE, job_id="job-a", attempt_id="attempt-1"
    ).zero_since == (_at(150))

    efa(store, "node-a", _at(300), bps=0)
    assert store.get_efa_traffic_state(
        NODE, job_id="job-a", attempt_id="attempt-1"
    ).zero_since == (_at(150))

    efa(store, "node-a", _at(400), attempt_id="attempt-2")
    with pytest.raises(NotFoundError):
        store.get_efa_traffic_state(NODE, job_id="job-a", attempt_id="attempt-1")


def _spike(store, *, state_key: str = NODE):
    efa(store, "node-a", _at(0), state_key=state_key)
    efa(store, "node-a", _at(130), state_key=state_key)
    state, emitted = efa(store, "node-a", _at(140), state_key=state_key, bps=9000)
    assert (state.signal, emitted) == (EfaTrafficSignal.SPIKE, True)
    return state


def _decide(store, event_id: str, **identity):
    return store.apply_efa_traffic_admin_action(
        state_key=NODE,
        event_id=event_id,
        action=EfaTrafficAdminAction.ACCEPT_NEW_BASELINE,
        operator="admin@example.com",
        reason="new communication phase",
        decided_at=_at(150),
        **identity,
    )


def test_an_admin_decision_for_an_attempt_that_left_the_node_conflicts(store) -> None:
    spike = _spike(store)
    efa(store, "node-a", _at(145), attempt_id="attempt-2")

    with pytest.raises(EfaTrafficAdminConflict, match="attempt job-a/attempt-2"):
        _decide(
            store, spike.active_spike_event_id, job_id="job-a", attempt_id="attempt-1"
        )


def test_an_admin_decision_moves_a_per_attempt_state_to_the_node_row(store) -> None:
    spike = _spike(store, state_key=LEGACY)

    decision = _decide(
        store, spike.active_spike_event_id, job_id="job-a", attempt_id="attempt-1"
    )

    assert decision.resulting_signal is EfaTrafficSignal.NORMAL
    state = store.get_efa_traffic_state(NODE, job_id="job-a", attempt_id="attempt-1")
    assert (state.state_key, state.baseline_bytes_per_second) == (NODE, 9000)
    with pytest.raises(NotFoundError):
        store.get_efa_traffic_state(LEGACY)


def test_an_admin_decision_on_the_attempts_own_row(store) -> None:
    spike = _spike(store)
    with pytest.raises(NotFoundError):
        store.apply_efa_traffic_admin_action(
            state_key=efa_traffic_state_key(CLUSTER, "node-z"),
            event_id=spike.active_spike_event_id,
            action=EfaTrafficAdminAction.ACKNOWLEDGE_TRANSIENT,
            operator="admin@example.com",
            reason="no such node row",
            decided_at=_at(150),
            job_id="job-a",
            attempt_id="attempt-1",
        )
    named = _decide(
        store, spike.active_spike_event_id, job_id="job-a", attempt_id="attempt-1"
    )
    assert named.attempt_id == "attempt-1"
    # The decision is idempotent by event and action, named or not.
    assert _decide(store, spike.active_spike_event_id) == named

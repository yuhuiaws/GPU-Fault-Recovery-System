"""The ``notified`` latch is written by the deliverer, not by the claim (P0-38B).

``claim_health_signal_transitions`` used to persist ``notified=True`` the
moment it decided to emit. Delivery -- the incident, the advisory notification
-- happens afterwards; on a backend without an enclosing transaction a failed
delivery left the latch durable and the signal silent for as long as the
fault lasted. The claim now leaves the latch alone and
``mark_health_signal_notified`` sets it once delivery has succeeded.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

import pytest

from gpu_fault.store import SqliteStore
from tests._builders import build_store
from tests.store._postgres_processor_claim_support import (
    _truncate,
    postgres_store_instance,
)

KEY = "cluster-a/node-a/network_link_down/eth0"
T0 = datetime(2026, 9, 6, 12, 0, tzinfo=timezone.utc)


@pytest.fixture(params=["memory", "sqlite", "postgres"])
def store(request, tmp_path):
    if request.param == "memory":
        yield build_store()
        return
    if request.param == "sqlite":
        sqlite = SqliteStore(str(tmp_path / "notified-latch.db"))
        try:
            yield sqlite
        finally:
            sqlite.close()
        return
    if not os.getenv("GPU_FAULT_TEST_POSTGRES_URL"):
        pytest.skip("GPU_FAULT_TEST_POSTGRES_URL is required")
    for postgres in postgres_store_instance():
        yield postgres
    _truncate()


def _claim(store, observed_at: datetime, *, active: bool = True) -> bool:
    return store.claim_health_signal_transitions(
        [(KEY, active, observed_at, 0.0)], received_at=observed_at
    )[0]


def _state(store):
    return store.get_health_signal_state(KEY)


def test_notified_flag_is_written_after_delivery_not_before(store) -> None:
    assert _claim(store, T0) is True
    state = _state(store)
    assert state is not None, "the claim must persist the signal state"
    assert state.notified is not True, "the claim alone must not latch"

    # Delivery failed: nothing marked it. The next sample emits again.
    assert _claim(store, T0 + timedelta(seconds=30)) is True

    # Delivery succeeded: the deliverer latches, and the signal goes quiet.
    store.mark_health_signal_notified(KEY, notified_at=T0 + timedelta(seconds=30))
    latched = _state(store)
    assert latched is not None, "marking must keep the signal state"
    assert latched.notified is True
    assert latched.active is True
    assert _claim(store, T0 + timedelta(seconds=60)) is False


def test_marking_a_signal_that_went_inactive_is_a_no_op(store) -> None:
    assert _claim(store, T0) is True
    assert _claim(store, T0 + timedelta(seconds=30), active=False) is False

    store.mark_health_signal_notified(KEY, notified_at=T0 + timedelta(seconds=30))

    state = _state(store)
    assert state is not None, "the inactive state must survive the no-op"
    assert state.active is False
    assert state.notified is False
    # A fresh activation is a new episode and emits.
    assert _claim(store, T0 + timedelta(seconds=60)) is True


def test_marking_an_unknown_signal_creates_nothing(store) -> None:
    store.mark_health_signal_notified("cluster-a/node-z/nothing/node", notified_at=T0)

    assert store.get_health_signal_state("cluster-a/node-z/nothing/node") is None


def test_a_reactivation_after_the_delivery_is_not_latched(store) -> None:
    """The mark names the episode it delivered: an activation that began after
    ``notified_at`` is a new one and still owes a notification."""

    assert _claim(store, T0) is True
    assert _claim(store, T0 + timedelta(seconds=10), active=False) is False
    assert _claim(store, T0 + timedelta(seconds=20)) is True

    store.mark_health_signal_notified(KEY, notified_at=T0)

    state = _state(store)
    assert state is not None, "the re-activated state must remain"
    assert state.notified is not True
    assert _claim(store, T0 + timedelta(seconds=30)) is True


def test_the_single_signal_claim_leaves_the_latch_to_the_deliverer(store) -> None:
    """``claim_health_signal_transition`` is the training-progress path. It
    used to latch on the claim because its caller had no post-commit half;
    ``TrainingHealthService.mark_notified`` is that half now (log §70), so the
    single form behaves exactly like the plural one."""

    assert store.claim_health_signal_transition(KEY, True, T0) is True
    state = _state(store)
    assert state is not None, "the single claim must persist state"
    assert state.notified is not True, "the single claim alone must not latch"
    assert store.claim_health_signal_transition(KEY, True, T0 + timedelta(1)) is True

    store.mark_health_signal_notified(KEY, notified_at=T0 + timedelta(1))

    assert store.claim_health_signal_transition(KEY, True, T0 + timedelta(2)) is False

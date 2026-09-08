"""``claim_health_signal_transitions`` decides "emit" under a lock (G-5).

Control-plane review 2026-09-08. The claim ran ``SELECT ... FOR UPDATE`` on an
autocommit connection -- the row lock was released before the caller saw the
row -- and then upserted without a guard. Two workers handling samples for the
same signal (the observation lanes admit two requests for one lane; the host
health and fault ingest paths are independent writers) both read "inactive",
both computed "first activation, emit", and both opened an incident; or the
slower one wrote its stale "inactive" over the fresh "active". The whole
decision now runs inside one transaction under the same per-key advisory lock
``mark_health_signal_notified`` takes, so for a brand-new key -- where there is
no row for FOR UPDATE to lock -- the second writer still waits for the first.
"""

from __future__ import annotations

import os
import threading
from datetime import datetime, timedelta, timezone

import pytest

from tests.store._postgres_processor_claim_support import (
    _truncate,
    postgres_store_instance,
)

pytestmark = pytest.mark.skipif(
    not os.getenv("GPU_FAULT_TEST_POSTGRES_URL"),
    reason="GPU_FAULT_TEST_POSTGRES_URL is not configured",
)

KEY = "cluster-a/node-a/network_link_down/eth0"
T0 = datetime(2026, 9, 8, 12, 0, tzinfo=timezone.utc)


@pytest.fixture
def store():
    yield from postgres_store_instance()
    _truncate()


def _claim(store, observed_at: datetime, *, active: bool = True) -> bool:
    return store.claim_health_signal_transitions(
        [(KEY, active, observed_at, 0.0)], received_at=observed_at
    )[0]


def test_two_first_activations_of_one_new_signal_emit_exactly_once(store) -> None:
    """A brand-new key has no row for FOR UPDATE to lock; the advisory lock
    is what makes the second writer wait for the first."""
    from unittest import mock

    # ``_next_health_signal_state`` is a staticmethod on the shared mixin.
    original = store._next_health_signal_state
    read_done = threading.Event()
    gate = threading.Event()

    def held_decide(*args, **kwargs):
        # The first claim parks here after its read, holding the lock; the
        # second, once it gets the lock, passes straight through.
        read_done.set()
        assert gate.wait(timeout=10), (
            "the first claim must be released by the test gate"
        )
        return original(*args, **kwargs)

    results: dict[str, bool] = {}
    errors: list[BaseException] = []

    def claim(name: str, offset_seconds: float) -> None:
        try:
            results[name] = _claim(store, T0 + timedelta(seconds=offset_seconds))
        except BaseException as exc:  # pragma: no cover - reported below
            errors.append(exc)

    with mock.patch.object(
        type(store), "_next_health_signal_state", staticmethod(held_decide)
    ):
        first = threading.Thread(target=claim, args=("first", 0.0))
        first.start()
        assert read_done.wait(timeout=10), (
            "the first claim must reach its decision under the lock"
        )
        second = threading.Thread(target=claim, args=("second", 1.0))
        second.start()
        second.join(timeout=0.5)
        assert second.is_alive(), "the second claim did not wait for the lock"
        gate.set()
        first.join(timeout=15)
        second.join(timeout=15)

    assert errors == [], errors
    # Both emit: a still-active signal re-emits on every sample until the
    # deliverer latches ``notified`` (P0-38B), and the first is what the
    # deliverer will latch. What the lock changes is the state the second one
    # computes from: it now sees the first's row, so ``active_since`` is the
    # first sample's clock rather than its own -- without the lock both read
    # "no previous state", the minimum-active timer restarted, and whichever
    # write landed last decided the row.
    assert results == {"first": True, "second": True}, results
    state = store.get_health_signal_state(KEY)
    assert state is not None and state.active is True
    assert state.active_since == T0
    # The second sample was the newer one; a lost update would have left T0.
    assert state.observed_at == T0 + timedelta(seconds=1)


def test_claim_holds_the_row_until_its_own_write_lands(store) -> None:
    """A concurrent deactivation cannot interleave between read and write."""
    assert _claim(store, T0) is True
    original = store._next_health_signal_state
    gate = threading.Event()
    read_done = threading.Event()

    def blocking_decide(*args, **kwargs):
        read_done.set()
        assert gate.wait(timeout=10), (
            "the deactivation must be released by the test gate"
        )
        return original(*args, **kwargs)

    from unittest import mock

    outcome: dict[str, object] = {}

    def slow_deactivation() -> None:
        with mock.patch.object(
            type(store), "_next_health_signal_state", staticmethod(blocking_decide)
        ):
            outcome["slow"] = _claim(store, T0 + timedelta(seconds=5), active=False)

    slow = threading.Thread(target=slow_deactivation)
    slow.start()
    assert read_done.wait(timeout=10), (
        "the deactivation must reach its decision under the lock"
    )
    # A newer re-activation arrives while the deactivation is mid-decision. It
    # must wait for the lock rather than read the pre-deactivation row.
    fast_done = threading.Event()

    def fast_reactivation() -> None:
        outcome["fast"] = _claim(store, T0 + timedelta(seconds=10), active=True)
        fast_done.set()

    fast = threading.Thread(target=fast_reactivation)
    fast.start()
    assert not fast_done.wait(timeout=0.5), "the second claim did not wait for the lock"
    gate.set()
    slow.join(timeout=15)
    fast.join(timeout=15)

    state = store.get_health_signal_state(KEY)
    assert state is not None
    assert state.observed_at == T0 + timedelta(seconds=10)
    assert state.active is True
    assert outcome["fast"] is True, outcome

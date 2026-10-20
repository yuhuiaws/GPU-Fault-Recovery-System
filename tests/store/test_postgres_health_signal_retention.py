"""The health-signal retention on PostgreSQL: same rows, bounded, lock-safe.

Batch 145. The memory and SQLite halves are
``test_health_signal_retention.py``; these hold PostgreSQL to the same site
(``_health_signal_retention_cases.py``) and add what only it has: concurrent
writers on other connections. The sweep reads one primary-key page per
statement and deletes a row only while its advisory lock -- the one every claim
holds for its whole read-decide-write -- is free and the row is unchanged. A
claim mid-flight keeps its window and is never made to wait; a claim after the
delete opens a fresh one.
"""

from __future__ import annotations

import os
import threading
import time
from datetime import timedelta
from unittest import mock

import pytest

from gpu_fault.efa_traffic_state import (
    efa_traffic_state_key,
    legacy_efa_traffic_state_key,
)
from gpu_fault.models import EfaTrafficSignal
from gpu_fault.store.postgres.signal_retention import _SIGNAL_STATE_PAGE
from gpu_fault.store.shared.errors import NotFoundError
from gpu_fault.watcher import WorkloadPhase
from tests.store._health_signal_retention_cases import (
    CLUSTER,
    GPU_GATE,
    NOW,
    OLD,
    claim,
    efa,
    efa_rows,
    health_rows,
    latch,
    observe,
    seed_efa,
    seed_site,
    sweep,
    watch,
    window,
)
from tests.store._postgres_processor_claim_support import (
    POSTGRES_URL,
    _truncate,
    postgres_store_instance,
)

pytestmark = pytest.mark.skipif(
    not os.getenv("GPU_FAULT_TEST_POSTGRES_URL"),
    reason="GPU_FAULT_TEST_POSTGRES_URL is not configured",
)

ENDED = window(GPU_GATE, "job-a", "attempt-done")


@pytest.fixture
def store():
    yield from postgres_store_instance()
    _truncate()


def _connect():
    import psycopg

    return psycopg.connect(POSTGRES_URL, autocommit=False)


def _ended_window(store) -> str:
    watch(store)
    observe(store, "job-a", "attempt-done", WorkloadPhase.SUCCEEDED)
    claim(store, ENDED, OLD)
    latch(store, ENDED, OLD)
    return ENDED


def test_postgres_sweeps_the_site_to_the_same_rows(store) -> None:
    kept, removed = seed_site(store)
    efa_kept, efa_removed = seed_efa(store)

    counts = sweep(store)

    assert health_rows(store, kept) == set(kept), kept
    assert health_rows(store, removed) == set(), removed
    assert efa_rows(store, efa_kept) == efa_kept
    assert efa_rows(store, efa_removed) == set()
    assert counts == {
        "health_signal_state": len(removed),
        "efa_traffic_state": len(efa_removed),
    }
    assert sweep(store) == {"health_signal_state": 0, "efa_traffic_state": 0}


def test_the_sweep_walks_the_key_space_in_bounded_pages(store) -> None:
    watch(store)
    for index in range(12):
        observe(store, "job-a", f"attempt-{index:02d}", WorkloadPhase.SUCCEEDED)
        claim(store, window(GPU_GATE, "job-a", f"attempt-{index:02d}"), OLD)
        claim(store, f"{CLUSTER}/node-{index:02d}/network_link_down/eth0", OLD)

    removed = [sweep(store, limit=5)["health_signal_state"] for _ in range(4)]

    assert removed == [5, 5, 2, 0]
    assert len(
        health_rows(
            store,
            [
                f"{CLUSTER}/node-{index:02d}/network_link_down/eth0"
                for index in range(12)
            ],
        )
    ) == (12)


def test_the_page_is_a_primary_key_range_scan(store) -> None:
    """No index of its own: the page rides ``gpu_fault_objects_pkey``."""

    _ended_window(store)
    with _connect() as connection, connection.cursor() as cursor:
        cursor.execute("SET LOCAL enable_seqscan = off")
        cursor.execute(
            "EXPLAIN (FORMAT TEXT) " + _SIGNAL_STATE_PAGE,
            ("2030-01-01T00:00:00.000000Z", "health_signal_state", "", 1000),
        )
        plan = "\n".join(row[0] for row in cursor.fetchall())
        connection.rollback()

    assert "Limit" in plan, plan
    assert "gpu_fault_objects_pkey" in plan, plan
    assert "Sort" not in plan, plan


def test_a_key_a_claim_holds_is_skipped_without_waiting(store) -> None:
    key = _ended_window(store)
    other = window(GPU_GATE, "job-a", "attempt-other")
    claim(store, other, OLD)
    with _connect() as holder, holder.cursor() as cursor:
        # What claim_health_signal_transitions holds while it decides.
        cursor.execute(
            "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))",
            (f"health_signal_state/{key}",),
        )
        started = time.monotonic()
        counts = sweep(store)
        elapsed = time.monotonic() - started
        holder.rollback()

    # Waiting would mean waiting on this very thread's lock: it returned, and
    # the bound is far below the statement timeout that would end a wait.
    assert elapsed < 10, f"the sweep waited {elapsed:.1f}s on a held lock"
    assert counts["health_signal_state"] == 1
    assert health_rows(store, [key, other]) == {key}
    assert sweep(store)["health_signal_state"] == 1
    assert health_rows(store, [key]) == set()


def test_a_row_another_transaction_locked_is_skipped_without_waiting(store) -> None:
    key = _ended_window(store)
    with _connect() as holder, holder.cursor() as cursor:
        cursor.execute(
            """
            SELECT key FROM gpu_fault_objects
            WHERE kind='health_signal_state' AND key=%s
            FOR UPDATE
            """,
            (key,),
        )
        assert cursor.fetchone() == (key,)
        started = time.monotonic()
        assert sweep(store)["health_signal_state"] == 0
        elapsed = time.monotonic() - started
        holder.rollback()

    assert elapsed < 10, f"the sweep waited {elapsed:.1f}s on a row lock"
    assert sweep(store)["health_signal_state"] == 1


def test_a_claim_parked_mid_decision_keeps_its_window(store) -> None:
    """The sweep runs while a replica's claim of the key holds its lock: the
    window stays, the claim's write lands, and the sweep never waited."""

    key = _ended_window(store)
    original = store._next_health_signal_state
    decided = threading.Event()
    release = threading.Event()
    outcome: dict[str, object] = {}

    def parked(*args, **kwargs):
        decided.set()
        # Longer than the bound below: a sweep that waited for this claim
        # would take at least this long.
        assert release.wait(timeout=30), "the test releases the claim"
        return original(*args, **kwargs)

    def replica() -> None:
        try:
            outcome["emitted"] = store.claim_health_signal_transitions(
                [(key, False, NOW - timedelta(seconds=5), 0.0)],
                received_at=NOW - timedelta(seconds=5),
            )
        except BaseException as exc:  # pragma: no cover - reported below
            outcome["error"] = exc

    with mock.patch.object(
        type(store), "_next_health_signal_state", staticmethod(parked)
    ):
        thread = threading.Thread(target=replica)
        thread.start()
        assert decided.wait(timeout=15), "the claim reaches its decision"
        started = time.monotonic()
        counts = sweep(store)
        elapsed = time.monotonic() - started
        release.set()
        thread.join(timeout=30)

    assert "error" not in outcome, outcome
    assert elapsed < 10, f"the sweep waited {elapsed:.1f}s on the claim"
    assert counts["health_signal_state"] == 0
    state = store.get_health_signal_state(key)
    assert state is not None and state.active is False
    assert state.clock_at == NOW - timedelta(seconds=5), "the claim's write landed"
    assert sweep(store)["health_signal_state"] == 0, "a freshly written row stays"


def test_a_claim_after_the_delete_opens_a_fresh_window(store) -> None:
    key = _ended_window(store)
    assert sweep(store)["health_signal_state"] == 1

    assert claim(store, key, NOW) is True, "a new episode owes its notification"
    state = store.get_health_signal_state(key)
    assert state is not None
    assert (state.active_since, state.notified) == (NOW, False)


def test_an_efa_upgrade_moves_the_attempts_row_under_both_locks(store) -> None:
    node = efa_traffic_state_key(CLUSTER, "node-a")
    legacy = legacy_efa_traffic_state_key(node, "job-a", "attempt-1")
    for seconds in (0, 130, 140):
        efa(store, "node-a", OLD + timedelta(seconds=seconds), state_key=legacy)
    efa(store, "node-a", OLD + timedelta(seconds=150), state_key=legacy, bps=0)

    with _connect() as holder, holder.cursor() as cursor:
        # A writer of the release before still holds the attempt's row.
        cursor.execute(
            "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))",
            (f"efa_traffic_state/{legacy}",),
        )
        done = threading.Event()
        result: dict[str, object] = {}

        def observe_node_row() -> None:
            result["state"] = efa(store, "node-a", OLD + timedelta(seconds=220), bps=0)[
                0
            ]
            done.set()

        thread = threading.Thread(target=observe_node_row)
        thread.start()
        assert not done.wait(timeout=1), "the move waits for the older writer"
        holder.rollback()
        thread.join(timeout=15)

    state = result["state"]
    assert state.signal is EfaTrafficSignal.ZERO_WARNING
    assert state.zero_since == OLD + timedelta(seconds=150)
    assert store.get_efa_traffic_state(node) == state
    with pytest.raises(NotFoundError):
        store.get_efa_traffic_state(legacy)

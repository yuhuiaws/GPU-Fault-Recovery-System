"""One advisory-lock statement takes the per-key loop's keys, in its order.

Local PostgreSQL. ``take_advisory_xact_locks`` replaces a loop of single-key
``pg_advisory_xact_lock`` round trips, so it must lock exactly the keys the
loop locked -- the same ``hashtextextended`` of the same key text, which is
what still excludes a single-key ``_state_transaction`` caller -- and in the
same order. ``COLLATE "C"`` is Python's ``sorted()`` order; the database
default (``en_US.utf8`` in the postgres image) sorts ``C/1`` after ``b/1``
and would deadlock against a pod still running the loop mid-rollout.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from typing import Any

import pytest

from gpu_fault.store.postgres.core import take_advisory_xact_locks
from tests.store._cov95_compat_support import NOW
from tests.store._cov95_compat_workflows import save_pair
from tests.store._postgres_processor_claim_support import (
    _truncate,
    postgres_store_instance,
)

POSTGRES_URL = os.getenv("GPU_FAULT_TEST_POSTGRES_URL")
pytestmark = pytest.mark.skipif(
    not POSTGRES_URL, reason="GPU_FAULT_TEST_POSTGRES_URL is not configured"
)

SINGLE_KEY_LOCK = "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))"


@pytest.fixture
def store():
    yield from postgres_store_instance()
    _truncate()


def _connect():
    import psycopg

    assert POSTGRES_URL is not None
    return psycopg.connect(POSTGRES_URL, autocommit=True)


def _hashes(connection: Any, keys: Iterable[str]) -> dict[str, int]:
    rows = connection.execute(
        "SELECT key, hashtextextended(key, 0) FROM unnest(%s::text[]) AS key",
        (list(keys),),
    ).fetchall()
    return {key: value for key, value in rows}


def _advisory_locks(observer: Any, *, gate_pid: int) -> list[tuple[int, bool, int]]:
    # A bigint advisory key is split over classid (high half) and objid.
    return observer.execute(
        """
        SELECT pid, granted, (classid::bigint << 32) | objid::bigint
        FROM pg_locks
        WHERE locktype = 'advisory' AND objsubid = 1 AND pid <> %s
        """,
        (gate_pid,),
    ).fetchall()


def _wait_for(predicate: Callable[[], Any], what: str) -> Any:
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(0.02)
    raise AssertionError(f"timed out waiting until {what}")


def _held_while_blocked(
    observer: Any, gate_pid: int, gate_key: int
) -> tuple[set[int], set[int]]:
    """(granted, waiting) advisory keys of the backend blocked on ``gate_key``."""

    def blocked() -> tuple[set[int], set[int]] | None:
        rows = _advisory_locks(observer, gate_pid=gate_pid)
        pids = [pid for pid, granted, key in rows if not granted and key == gate_key]
        if not pids:
            return None
        return (
            {key for pid, granted, key in rows if pid == pids[0] and granted},
            {key for pid, granted, key in rows if pid == pids[0] and not granted},
        )

    return _wait_for(blocked, "a backend waits on the gate key")


def _locks_while_gated(
    call: Callable[[], Any], lock_keys: dict[str, str], gate: str
) -> tuple[set[str], set[str], Any]:
    """Run ``call`` while ``gate`` is held elsewhere; name what it held."""

    with _connect() as gate_connection, _connect() as observer:
        hashes = _hashes(observer, lock_keys.values())
        names = {hashes[key]: name for name, key in lock_keys.items()}
        gate_key = hashes[lock_keys[gate]]
        gate_connection.execute("SELECT pg_advisory_lock(%s)", (gate_key,))
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(call)
            try:
                granted, waiting = _held_while_blocked(
                    observer, gate_connection.info.backend_pid, gate_key
                )
            finally:
                gate_connection.execute("SELECT pg_advisory_unlock(%s)", (gate_key,))
            result = future.result(timeout=10)
    return {names[key] for key in granted}, {names[key] for key in waiting}, result


def test_a_health_signal_claim_locks_the_loop_keys_in_byte_order(store) -> None:
    signals = ["b/1", "a/1", "C/1"]

    granted, waiting, result = _locks_while_gated(
        lambda: store.claim_health_signal_transitions(
            [(signal, True, NOW, 0.0) for signal in signals]
        ),
        # The key text of mark_health_signal_notified's _state_transaction.
        {signal: f"health_signal_state/{signal}" for signal in signals},
        # Byte order is C/1, a/1, b/1: hold the last key.
        gate="b/1",
    )

    assert (granted, waiting) == ({"C/1", "a/1"}, {"b/1"}), (
        "byte order takes C/1 before a/1; the database collation would not"
    )
    assert len(result) == len(signals)


@pytest.mark.parametrize("path", ["claim", "extend"])
def test_budget_scopes_lock_the_loop_keys_in_byte_order(store, path: str) -> None:
    _, workflow = save_pair(store, f"budget-{path}")
    scopes = {"node:b": 1, "node:a": 1, "Node:c": 1}
    lease = timedelta(minutes=5)
    if path == "extend":
        store.claim_workflow(
            workflow.request_id,
            "owner",
            workflow.fencing_token,
            now=NOW,
            lease_duration=lease,
        )

    def call() -> Any:
        if path == "extend":
            return store.extend_remediation_budget(
                workflow.request_id, "owner", scopes, now=NOW
            )
        return store.claim_workflow(
            workflow.request_id,
            "owner",
            workflow.fencing_token,
            now=NOW,
            lease_duration=lease,
            remediation_budget_claims=scopes,
        )

    granted, waiting, result = _locks_while_gated(
        call, {scope: f"remediation_budget/{scope}" for scope in scopes}, gate="node:b"
    )

    assert (granted, waiting) == ({"Node:c", "node:a"}, {"node:b"}), (
        "byte order takes Node:c before node:a; the database collation would not"
    )
    assert sorted(result.remediation_budget_claims) == sorted(scopes)


def test_the_batch_order_cannot_deadlock_against_the_per_key_loop() -> None:
    names = ("a/1", "b/1", "C/1", "D/1")
    keys = {name: f"lock-order-test/{name}" for name in names}
    with (
        _connect() as gate,
        _connect() as observer,
        _connect() as batch,
        _connect() as loop,
    ):
        hashes = _hashes(observer, keys.values())
        key_names = {hashes[key]: name for name, key in keys.items()}
        gate_pid = gate.info.backend_pid
        gate.execute("SELECT pg_advisory_lock(%s)", (hashes[keys["a/1"]],))

        def locks(granted: bool) -> list[str]:
            return sorted(
                key_names[key]
                for _pid, is_granted, key in _advisory_locks(
                    observer, gate_pid=gate_pid
                )
                if is_granted == granted and key in key_names
            )

        def batch_transaction() -> None:
            with batch.transaction(), batch.cursor() as cursor:
                take_advisory_xact_locks(
                    cursor, [keys[n] for n in ("a/1", "C/1", "b/1")]
                )

        def loop_transaction() -> None:
            # An old pod: one round trip per key, in sorted() order.
            with loop.transaction(), loop.cursor() as cursor:
                for key in sorted(keys[n] for n in ("C/1", "a/1", "D/1")):
                    cursor.execute(SINGLE_KEY_LOCK, (key,))

        with ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(batch_transaction)
            _wait_for(lambda: locks(False) == ["a/1"], "the batch waits on a/1")
            assert locks(True) == ["C/1"], "the batch took C/1 first (byte order)"
            second = pool.submit(loop_transaction)
            _wait_for(lambda: locks(False) == ["C/1", "a/1"], "the loop waits on C/1")
            gate.execute("SELECT pg_advisory_unlock(%s)", (hashes[keys["a/1"]],))
            # A deadlock would surface here as DeadlockDetected.
            first.result(timeout=10)
            second.result(timeout=10)
        assert locks(True) == [] and locks(False) == []

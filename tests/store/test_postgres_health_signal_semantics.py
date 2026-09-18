from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing, contextmanager
from queue import Queue
from threading import Event

import pytest

from gpu_fault.store import PostgresStore
from tests.store._health_signal_semantics import (
    KEY,
    NOW,
    A,
    B,
    SemanticSignalContract,
    acknowledge,
    at,
    claim,
)
from tests.store._postgres_processor_claim_support import (
    _truncate,
    postgres_store_instance,
)

POSTGRES_URL = os.environ.get("GPU_FAULT_TEST_POSTGRES_URL", "")
pytestmark = pytest.mark.skipif(
    not POSTGRES_URL, reason="requires a separately allocated local PostgreSQL"
)


class SignalPostgresStore(PostgresStore):
    def seed_legacy_signal(self, payload):
        with self._state_transaction(f"health_signal_state/{KEY}"):
            with self._db.cursor() as cursor:
                cursor.execute(
                    "INSERT INTO gpu_fault_objects(kind,key,payload) "
                    "VALUES ('health_signal_state',%s,%s::jsonb)",
                    (KEY, json.dumps(payload)),
                )

    def claim_and_hold(self, fingerprint, ready, peer_pid, blocked, release):
        with self._db.transaction():
            assert claim(self, fingerprint, 2) is True
            ready.set()
            pid = peer_pid.get(timeout=5)
            deadline = time.monotonic() + 5
            with self._db.cursor() as cursor:
                while True:
                    cursor.execute(
                        "SELECT pg_backend_pid()=ANY(pg_blocking_pids(%s))", (pid,)
                    )
                    if cursor.fetchone()[0]:
                        break
                    assert time.monotonic() < deadline, (
                        "the peer did not block behind the semantic claim transaction"
                    )
                    time.sleep(0.01)
            blocked.set()
            assert release.wait(5), "the test did not release its held claim"

    def competing_update(self, peer_pid, *, stale_claim):
        with self._db.transaction():
            with self._db.cursor() as cursor:
                cursor.execute("SELECT pg_backend_pid()")
                peer_pid.put(cursor.fetchone()[0])
            if stale_claim:
                return claim(self, A, 1)
            acknowledge(self, A, 0)
            return None


@pytest.fixture
def store(monkeypatch):
    monkeypatch.setenv("GPU_FAULT_POSTGRES_HOT_STATE_MODE", "legacy")
    with contextmanager(postgres_store_instance)():
        with closing(
            SignalPostgresStore(
                POSTGRES_URL, initialize_schema=False, pool_min_size=1, pool_max_size=1
            )
        ) as instance:
            try:
                yield instance
            finally:
                _truncate()


class TestPostgresHealthSignalSemantics(SemanticSignalContract):
    pass


@pytest.mark.parametrize("later_fingerprint", [A, B])
def test_late_ack_rechecks_the_semantic_episode_after_the_claim_commits(
    store, later_fingerprint
) -> None:
    assert claim(store, A, 0) is True
    if later_fingerprint == A:
        assert claim(store, B, 1) is True
    ready, blocked, release = Event(), Event(), Event()
    peer_pid: Queue[int] = Queue()
    with closing(
        SignalPostgresStore(
            POSTGRES_URL, initialize_schema=False, pool_min_size=1, pool_max_size=1
        )
    ) as peer:
        with ThreadPoolExecutor(max_workers=2) as executor:
            pending = executor.submit(
                store.claim_and_hold,
                later_fingerprint,
                ready,
                peer_pid,
                blocked,
                release,
            )
            try:
                assert ready.wait(5), (
                    "the new meaning did not reach its uncommitted state"
                )
                ack = executor.submit(
                    peer.competing_update, peer_pid, stale_claim=False
                )
                assert blocked.wait(5), (
                    "the old acknowledgement did not wait for the claim"
                )
                assert not ack.done(), (
                    "the old acknowledgement escaped the shared key lock"
                )
            finally:
                release.set()
            pending.result(timeout=5)
            ack.result(timeout=5)
    current = store.get_health_signal_state(KEY)
    assert current.semantic_fingerprint == later_fingerprint
    assert current.semantic_since == at(2)
    assert current.active_since == NOW
    assert current.notified is False, (
        "a delayed old delivery latched the newer semantic episode"
    )
    assert claim(store, later_fingerprint, 3) is True
    acknowledge(store, later_fingerprint, 3)
    assert claim(store, later_fingerprint, 4) is False


def test_older_claim_cannot_overwrite_a_new_semantic_transition(store) -> None:
    assert claim(store, A, 0) is True
    acknowledge(store, A, 0)
    ready, blocked, release = Event(), Event(), Event()
    peer_pid: Queue[int] = Queue()
    with closing(
        SignalPostgresStore(
            POSTGRES_URL, initialize_schema=False, pool_min_size=1, pool_max_size=1
        )
    ) as peer:
        with ThreadPoolExecutor(max_workers=2) as executor:
            newer = executor.submit(
                store.claim_and_hold, B, ready, peer_pid, blocked, release
            )
            try:
                assert ready.wait(5), "the newer claim did not hold its transaction"
                stale = executor.submit(
                    peer.competing_update, peer_pid, stale_claim=True
                )
                assert blocked.wait(5), (
                    "the stale sample did not wait for the newer claim"
                )
            finally:
                release.set()
            newer.result(timeout=5)
            assert stale.result(timeout=5) is False
    current = store.get_health_signal_state(KEY)
    assert current.semantic_fingerprint == B
    assert current.semantic_since == current.clock_at == at(2)
    assert current.notified is False

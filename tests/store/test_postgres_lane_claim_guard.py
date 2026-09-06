"""One lane, two pending rows, two workers claiming at once: at most one row
of the lane may be claimed, and its completion must then succeed.

Reported live on 2026-09-05 (memory ``processor-gpu-inventory-stale-fencing-
token``): two adjacent gpu-inventory snapshots of one node enqueued 45 ms
apart were claimed by two workers, each claim bumped the lane epoch, and the
first completion failed with "stale processor lane fencing token". The claim
SQL's lane guard is supposed to make that impossible; this test pins it under
real concurrency so the guard cannot regress silently (F-D2).
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import pytest

from gpu_fault.store import PostgresStore
from tests._builders import processor_request
from tests.store._postgres_processor_claim_support import _truncate

POSTGRES_URL = os.getenv("GPU_FAULT_TEST_POSTGRES_URL")
pytestmark = pytest.mark.skipif(
    not POSTGRES_URL, reason="GPU_FAULT_TEST_POSTGRES_URL is not configured"
)
LEASE = timedelta(seconds=120)


@pytest.fixture(autouse=True)
def clean_tables():
    assert POSTGRES_URL is not None
    PostgresStore(POSTGRES_URL).close()
    _truncate()
    yield
    _truncate()


def _store() -> PostgresStore:
    assert POSTGRES_URL is not None
    return PostgresStore(POSTGRES_URL, initialize_schema=False)


def _inventory(sequence: int):
    return processor_request(
        "/v1/collector-events/gpu-inventory",
        body=('{"node_id":"node-a","gpus":[],"seq":%d}' % sequence).encode(),
        cluster_id="cluster-a",
    )


@pytest.mark.parametrize("rounds", [8])
def test_postgres_two_workers_never_both_claim_the_same_lane(rounds: int) -> None:
    first, second = _store(), _store()
    try:
        for round_index in range(rounds):
            _truncate()
            # Two rows of the same node lane: the routine coalescing merges
            # them while both are PENDING, so lease the first one out of the
            # coalescing window before the second arrives, then release it.
            a = first.enqueue_processor_request(_inventory(round_index * 2))
            leased = first.claim_active_processor_requests(
                "warmup", now=datetime.now(timezone.utc), lease_duration=LEASE, limit=1
            )
            assert [item.request_id for item in leased] == [a.request_id]
            b = first.enqueue_processor_request(_inventory(round_index * 2 + 1))
            first.release_active_processor_request(
                a.request_id, "warmup", leased[0].leader_epoch, leased[0].lease_token
            )
            assert first.processor_queue_stats()["depth"] == 2

            def claim(store, owner):
                return store.claim_active_processor_requests(
                    owner, now=datetime.now(timezone.utc), lease_duration=LEASE, limit=4
                )

            with ThreadPoolExecutor(max_workers=2) as pool:
                results = list(
                    pool.map(
                        lambda pair: claim(*pair),
                        [(first, "worker-a"), (second, "worker-b")],
                    )
                )
            claimed = [item for batch in results for item in batch]

            assert len(claimed) <= 1, [(c.request_id, c.lease_owner) for c in claimed]
            assert {b.request_id, a.request_id} >= {c.request_id for c in claimed}
            if claimed:
                item = claimed[0]
                owner_store = first if item.lease_owner == "worker-a" else second
                owner_store.complete_active_processor_request(
                    item.request_id,
                    item.lease_owner,
                    item.leader_epoch,
                    item.lease_token,
                    response_status=200,
                    response_content_type="application/json",
                    response_body_base64="e30=",
                )
    finally:
        first.close()
        second.close()

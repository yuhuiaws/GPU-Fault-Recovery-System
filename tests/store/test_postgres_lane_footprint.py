"""The lane table keeps one row per ordering key and its heap does not grow
burst over burst.

A perf burst inserted 1,224 lane rows a second while the retention sweep
deleted as many of the previous run's rows, and the heap went from 3 to 1,416
pages inside a minute. The rows are inherent: one per distinct ordering key
leased within the retention window. What must hold is that a key claimed
again updates its row in place (``INSERT ... ON CONFLICT``), that the sweep
hands the space back, and that the next burst reuses it instead of growing
the heap.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone

import pytest

from gpu_fault.store import PostgresStore
from tests._builders import processor_request
from tests.store._postgres_processor_claim_support import REQUEST_LEASE, _truncate

POSTGRES_URL = os.getenv("GPU_FAULT_TEST_POSTGRES_URL")
pytestmark = pytest.mark.skipif(
    not POSTGRES_URL, reason="GPU_FAULT_TEST_POSTGRES_URL is not configured"
)
LANES = "gpu_fault_processor_lanes"
PATH = "/v1/collector-events/host-telemetry"
KEYS_PER_BURST = 200


@pytest.fixture
def store():
    assert POSTGRES_URL is not None
    _truncate()
    PostgresStore(POSTGRES_URL).close()
    instance = PostgresStore(POSTGRES_URL, initialize_schema=False)
    try:
        yield instance
    finally:
        instance.close()
        _truncate()


def _burst(store: PostgresStore, *, keys: int) -> None:
    """Enqueue, claim and complete one request on each of ``keys`` node lanes."""

    now = datetime.now(timezone.utc)
    for index in range(keys):
        store.enqueue_processor_request(
            processor_request(
                PATH,
                body=json.dumps(
                    {"node_id": f"node-{index:04d}", "at": now.isoformat()}
                ).encode(),
                cluster_id="cluster-a",
            )
        )
    completed = 0
    while completed < keys:
        claimed = store.claim_active_processor_requests(
            "pod-a", now=now, lease_duration=REQUEST_LEASE, limit=64
        )
        assert claimed, "the backlog must drain"
        for item in claimed:
            store.complete_active_processor_request(
                item.request_id,
                "pod-a",
                item.leader_epoch,
                item.lease_token,
                response_status=200,
                response_content_type="application/json",
                response_body_base64="e30=",
                path=PATH,
            )
            completed += 1


def _lanes(store: PostgresStore) -> tuple[int, int]:
    with store._db.cursor() as cursor:
        cursor.execute(f"SELECT count(*), pg_relation_size('{LANES}') FROM {LANES}")
        count, size = cursor.fetchone()
    return int(count), int(size)


def _vacuum(store: PostgresStore) -> None:
    import psycopg

    assert POSTGRES_URL is not None
    with psycopg.connect(POSTGRES_URL, autocommit=True) as connection:
        connection.execute(f"VACUUM {LANES}")


def test_a_key_claimed_again_updates_its_lane_row_in_place(
    store: PostgresStore,
) -> None:
    _burst(store, keys=1)
    _burst(store, keys=1)
    with store._db.cursor() as cursor:
        cursor.execute(f"SELECT ordering_key, epoch FROM {LANES}")
        rows = cursor.fetchall()
    assert len(rows) == 1 and rows[0][1] == 2, rows


def test_the_heap_is_reused_burst_over_burst(store: PostgresStore, request) -> None:
    sizes: list[int] = []
    for _ in range(3):
        _burst(store, keys=KEYS_PER_BURST)
        count, size = _lanes(store)
        assert count == KEYS_PER_BURST, "one row per ordering key"
        sizes.append(size)
        # The retention sweep (lane_retention after the lease ended) hands
        # the rows back; VACUUM (autovacuum in production, tuned per schema
        # v19) makes their space reusable before the next burst.
        deleted = store.cleanup_processor_lanes(
            older_than=datetime.now(timezone.utc) + timedelta(hours=1), limit=1000
        )
        assert deleted == KEYS_PER_BURST
        assert _lanes(store)[0] == 0
        _vacuum(store)
    request.node.user_properties.append(
        ("lane_heap_bytes_per_burst", json.dumps(sizes))
    )
    assert sizes[0] > 0
    assert max(sizes) <= sizes[0] + 8192, (
        f"the heap must not grow burst over burst (one page of slack): {sizes}"
    )

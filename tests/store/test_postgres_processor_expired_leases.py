"""A LEASED row whose owner died is handed back by a reclaim job, not by luck.

FINAL-建议汇总 F-D5 (P0-33B, P1-18D). Once a lease expired the row could only
be recovered by its owner's ``release_*`` (gone) or by a claim window that
happened to cover it; a priority-100 row behind the retry horizon could sit
LEASED indefinitely with nothing counting it. The reclaim puts it back to
PENDING with ``retry_count`` bumped so the retry is booked as one.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

import pytest

from gpu_fault.processor import ProcessorRequestStatus
from gpu_fault.store import SqliteStore
from tests._builders import build_store, processor_request
from tests.store._postgres_processor_claim_support import (
    _truncate,
    postgres_store_instance,
)

DISPATCH = "/v1/workflows/dispatch"
XID = "/v1/gpu-events/xid"


@pytest.fixture(params=["memory", "sqlite", "postgres"])
def store(request, tmp_path):
    if request.param == "memory":
        yield build_store()
        return
    if request.param == "sqlite":
        sqlite = SqliteStore(str(tmp_path / "expired-leases.db"))
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


def _crashed_and_live(store):
    now = datetime.now(timezone.utc)
    crashed = store.enqueue_processor_request(
        processor_request(DISPATCH, body=b'{"node_id":"node-crashed"}')
    )
    live = store.enqueue_processor_request(
        processor_request(
            XID, body=b'{"node_id":"node-live","xid":79}', cluster_id="cluster-b"
        )
    )
    # Claimed ten minutes ago on a one-minute lease: the owner never renewed.
    dead_claim = store.claim_active_processor_requests(
        "pod-dead",
        now=now - timedelta(minutes=10),
        lease_duration=timedelta(minutes=1),
        limit=1,
        include_paths={DISPATCH},
    )
    assert [item.request_id for item in dead_claim] == [crashed.request_id], (
        "the crashed owner's claim should have leased the dispatch row"
    )
    live_claim = store.claim_active_processor_requests(
        "pod-live",
        now=now,
        lease_duration=timedelta(minutes=2),
        limit=1,
        include_paths={XID},
    )
    assert [item.request_id for item in live_claim] == [live.request_id], (
        "the live owner's claim should have leased the xid row"
    )
    return now, crashed, live


def test_expired_leases_are_reclaimed_with_a_retry_bump(store) -> None:
    now, crashed, live = _crashed_and_live(store)

    reclaimed = store.reclaim_expired_processor_leases(now=now, limit=10)

    assert reclaimed == 1
    row = store.get_processor_request(crashed.request_id)
    assert row.status is ProcessorRequestStatus.PENDING
    assert row.retry_count == 1
    assert row.lease_owner is None
    assert row.lease_token is None
    assert row.lease_expires_at is None
    kept = store.get_processor_request(live.request_id)
    assert kept.status is ProcessorRequestStatus.LEASED
    assert kept.lease_owner == "pod-live"


def test_reclaim_is_idempotent_and_leaves_the_row_claimable(store) -> None:
    now, crashed, _live = _crashed_and_live(store)
    store.reclaim_expired_processor_leases(now=now, limit=10)

    assert store.reclaim_expired_processor_leases(now=now, limit=10) == 0
    again = store.claim_active_processor_requests(
        "pod-next",
        now=now,
        lease_duration=timedelta(minutes=2),
        limit=1,
        include_paths={DISPATCH},
    )
    assert [item.request_id for item in again] == [crashed.request_id]
    assert again[0].retry_count == 1


def test_reclaim_respects_its_limit(store) -> None:
    now = datetime.now(timezone.utc)
    for index in range(3):
        store.enqueue_processor_request(
            processor_request(
                XID, body=f'{{"node_id":"node-{index}","xid":79}}'.encode()
            )
        )
    dead = store.claim_active_processor_requests(
        "pod-dead",
        now=now - timedelta(minutes=10),
        lease_duration=timedelta(minutes=1),
        limit=3,
    )
    assert len(dead) == 3, "three lanes, three leases"

    assert store.reclaim_expired_processor_leases(now=now, limit=2) == 2
    assert store.reclaim_expired_processor_leases(now=now, limit=2) == 1

"""Releasing a claimed request frees the lane and the queue row on their own
CAS each (B-2, 2026-09-08), on every backend.

``release_active_processor_request`` used to be one all-or-nothing check: if
the queue row no longer carried the owner's fencing fields it returned before
touching the lane, and if the lane CAS missed it returned before touching the
queue row. Either way something stayed leased for the full request lease
(120 s in production) with nothing counting it - the "depth reads 1 for two
minutes" tail of the gpu-inventory fencing incident. The lane row is now
released whenever ``(ordering_key, owner_id, epoch, lease_token)`` still
match, whatever the queue row says; the queue row goes back to PENDING
whenever its fencing fields match and it is LEASED, whatever the lane says.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

import pytest

from gpu_fault.processor import ProcessorRequestStatus
from gpu_fault.store import InMemoryStore, PostgresStore, SqliteStore
from tests._builders import build_store, processor_request
from tests.store._postgres_processor_claim_support import (
    _truncate,
    postgres_store_instance,
)

LEASE = timedelta(seconds=120)
HOST_TELEMETRY = "/v1/collector-events/host-telemetry"
XID = "/v1/gpu-events/xid"


@pytest.fixture(params=["memory", "sqlite", "postgres"])
def store(request, tmp_path):
    if request.param == "memory":
        yield build_store()
        return
    if request.param == "sqlite":
        sqlite = SqliteStore(str(tmp_path / "release-cas.db"))
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


def _summary(seq: int):
    return processor_request(
        HOST_TELEMETRY,
        body=('{"node_id":"node-a","summary":true,"seq":%d}' % seq).encode(),
        cluster_id="cluster-a",
    )


def _lane(store, ordering_key: str):
    if isinstance(store, InMemoryStore):
        return store._processor_lanes.get(ordering_key)
    if isinstance(store, SqliteStore):
        return store._get_optional("processor_lane", ordering_key)
    assert isinstance(store, PostgresStore), "this fixture must be the Postgres store"
    with store._db.cursor() as cursor:
        cursor.execute(
            """
            SELECT owner_id, epoch, lease_token, lease_expires_at
            FROM gpu_fault_processor_lanes
            WHERE ordering_key=%s
            """,
            (ordering_key,),
        )
        row = cursor.fetchone()
    if row is None:
        return None
    from gpu_fault.processor import ProcessorLaneLease

    return ProcessorLaneLease(
        ordering_key=ordering_key,
        owner_id=row[0],
        epoch=row[1],
        lease_token=row[2],
        lease_expires_at=row[3],
        updated_at=row[3],
    )


def _force_pending_without_lease(store, request_id: str) -> None:
    """Emulate the B-1 revert: the queue row is PENDING with no lease fields
    while the lane row still carries the owner's lease."""

    current = store.get_processor_request(request_id)
    reverted = current.model_copy(
        update={
            "status": ProcessorRequestStatus.PENDING,
            "lease_owner": None,
            "leader_epoch": None,
            "lease_token": None,
            "lease_expires_at": None,
            "updated_at": datetime.now(timezone.utc),
        }
    )
    if isinstance(store, InMemoryStore):
        store._processor_requests[request_id] = reverted
    elif isinstance(store, SqliteStore):
        store._put("processor_request", request_id, reverted)
    else:
        store._persist_processor_state(reverted)


def test_release_frees_the_lane_even_when_the_queue_row_lost_the_lease(store):
    store.enqueue_processor_request(_summary(1))
    now = datetime.now(timezone.utc)
    [claimed] = store.claim_active_processor_requests(
        "worker-a", now=now, lease_duration=LEASE, limit=1
    )
    lane_key = claimed.ordering_key()
    _force_pending_without_lease(store, claimed.request_id)
    assert _lane(store, lane_key).lease_expires_at > now, "precondition"

    store.release_active_processor_request(
        claimed.request_id,
        "worker-a",
        claimed.leader_epoch,
        claimed.lease_token,
        not_before=None,
        retry_count=1,
    )

    lane = _lane(store, lane_key)
    assert lane.lease_expires_at <= datetime.now(timezone.utc), (
        "the owner's lane lease must be released on its own CAS"
    )
    assert (lane.owner_id, lane.epoch) == ("worker-a", claimed.leader_epoch)
    row = store.get_processor_request(claimed.request_id)
    assert row.status is ProcessorRequestStatus.PENDING
    assert row.lease_owner is None
    claimed_next = store.claim_active_processor_requests(
        "worker-b", now=datetime.now(timezone.utc), lease_duration=LEASE, limit=1
    )
    assert [item.request_id for item in claimed_next] == [claimed.request_id], (
        "the lane must be claimable again right away, not after 120 s"
    )


def test_release_hands_the_queue_row_back_even_when_the_lane_moved_on(store):
    """Worker A's lease lapsed; worker B claimed a higher-priority row on the
    same node lane (epoch +1), leaving A's row LEASED with A's fields. A's
    release must still hand that row back with its retry booked."""

    summary = store.enqueue_processor_request(_summary(1))
    long_ago = datetime.now(timezone.utc) - timedelta(minutes=10)
    [stale] = store.claim_active_processor_requests(
        "worker-a", now=long_ago, lease_duration=timedelta(minutes=1), limit=1
    )
    assert stale.request_id == summary.request_id
    fault = store.enqueue_processor_request(
        processor_request(
            XID, body=b'{"node_id":"node-a","xid":79}', cluster_id="cluster-a"
        )
    )
    assert fault.ordering_key() == summary.ordering_key(), "same node lane"
    now = datetime.now(timezone.utc)
    [taken] = store.claim_active_processor_requests(
        "worker-b", now=now, lease_duration=LEASE, limit=1
    )
    assert taken.request_id == fault.request_id
    assert taken.leader_epoch == stale.leader_epoch + 1
    assert store.get_processor_request(summary.request_id).lease_owner == "worker-a"

    not_before = now + timedelta(seconds=8)
    store.release_active_processor_request(
        summary.request_id,
        "worker-a",
        stale.leader_epoch,
        stale.lease_token,
        not_before=not_before,
        retry_count=1,
    )

    row = store.get_processor_request(summary.request_id)
    assert row.status is ProcessorRequestStatus.PENDING, row.status
    assert row.lease_owner is None
    assert row.retry_count == 1
    assert row.not_before == not_before
    lane = _lane(store, summary.ordering_key())
    assert (lane.owner_id, lane.epoch) == ("worker-b", taken.leader_epoch)
    assert lane.lease_expires_at > now, "B's lane lease is not ours to release"


def test_release_still_leaves_a_completed_row_and_a_foreign_lease_alone(store):
    store.enqueue_processor_request(_summary(1))
    now = datetime.now(timezone.utc)
    [claimed] = store.claim_active_processor_requests(
        "worker-a", now=now, lease_duration=LEASE, limit=1
    )
    store.complete_active_processor_request(
        claimed.request_id,
        "worker-a",
        claimed.leader_epoch,
        claimed.lease_token,
        response_status=200,
        response_content_type="application/json",
        response_body_base64="e30=",
    )

    store.release_active_processor_request(
        claimed.request_id, "worker-a", claimed.leader_epoch, claimed.lease_token
    )
    store.release_active_processor_request(
        claimed.request_id, "worker-a", claimed.leader_epoch, "not-my-token"
    )

    assert (
        store.get_processor_request(claimed.request_id).status
        is ProcessorRequestStatus.COMPLETED
    )

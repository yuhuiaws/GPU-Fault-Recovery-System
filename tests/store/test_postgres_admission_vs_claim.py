"""Batch admission must not hand a LEASED row back to PENDING (B-1, 2026-09-08).

The live "stale processor lane fencing token" on gpu-inventory was this: two
snapshots of one node arrive 20-45 ms apart; worker A claims the first while
the ingress batch for the second has already taken its candidate snapshot
(``status='PENDING'`` in a MATERIALIZED CTE) and is waiting on A's row lock.
When A commits, ``FOR UPDATE`` re-evaluates only the ``locked`` CTE's own
predicate, returns the *LEASED* version, and the latest-wins merge writes a
blind upsert that puts the row back to PENDING with every ``lease_*`` column
cleared. A's completion then fences, A's release is a no-op (the row no longer
carries A's token), and the lane stays leased for the full 120 s.

The pin runs the three steps on real Postgres with the store's own claim
statement held open in an uncommitted transaction.
"""

from __future__ import annotations

import os
import threading
from datetime import datetime, timedelta, timezone

import pytest

from gpu_fault.processor import ProcessorRequestStatus
from gpu_fault.store import PostgresStore
from tests._builders import processor_request
from tests.store._postgres_processor_claim_support import (
    _ensure_postgres_schema,
    _truncate,
)

POSTGRES_URL = os.getenv("GPU_FAULT_TEST_POSTGRES_URL")
pytestmark = pytest.mark.skipif(
    not POSTGRES_URL, reason="GPU_FAULT_TEST_POSTGRES_URL is not configured"
)
LEASE = timedelta(seconds=120)
LIMITS = {"max_depth": 64, "max_cluster_depth": 64}
INVENTORY = "/v1/collector-events/gpu-inventory"


@pytest.fixture(autouse=True)
def clean_tables():
    assert POSTGRES_URL is not None
    _ensure_postgres_schema()
    _truncate()
    yield
    _truncate()


def _store() -> PostgresStore:
    assert POSTGRES_URL is not None
    return PostgresStore(POSTGRES_URL, initialize_schema=False)


def _inventory(sequence: str, **overrides):
    request = processor_request(
        INVENTORY,
        body=('{"node_id":"node-a","gpus":[],"seq":"%s"}' % sequence).encode(),
        cluster_id="cluster-a",
    )
    return request.model_copy(update=overrides) if overrides else request


def _row(store: PostgresStore, request_id: str) -> dict:
    with store._db.cursor() as cursor:
        cursor.execute(
            """
            SELECT status, lease_owner, leader_epoch, lease_token,
                   retry_count, not_before, payload->>'body_base64'
            FROM gpu_fault_processor_queue
            WHERE request_id=%s
            """,
            (request_id,),
        )
        row = cursor.fetchone()
    assert row is not None, request_id
    keys = (
        "status",
        "lease_owner",
        "leader_epoch",
        "lease_token",
        "retry_count",
        "not_before",
        "body_base64",
    )
    return dict(zip(keys, row, strict=True))


def _lane(store: PostgresStore, ordering_key: str, *, now: datetime) -> tuple:
    with store._db.cursor() as cursor:
        cursor.execute(
            """
            SELECT owner_id, epoch, lease_expires_at > %s
            FROM gpu_fault_processor_lanes
            WHERE ordering_key=%s
            """,
            (now, ordering_key),
        )
        return cursor.fetchone()


def _wait_for_blocked_backend(store: PostgresStore, request_id: str) -> None:
    """Block until the ingress backend is waiting on the claim's row lock."""

    deadline = datetime.now(timezone.utc) + timedelta(seconds=10)
    while datetime.now(timezone.utc) < deadline:
        with store._db.cursor() as cursor:
            cursor.execute(
                """
                SELECT count(*)
                FROM pg_stat_activity
                WHERE wait_event_type='Lock'
                  AND state='active'
                  AND query LIKE %s
                """,
                ("%FOR UPDATE OF queue%",),
            )
            if cursor.fetchone()[0] >= 1:
                return
        threading.Event().wait(0.02)
    raise AssertionError(
        f"the admission of {request_id} never blocked on the claimed row lock"
    )


def test_admission_racing_a_claim_does_not_revert_the_leased_row() -> None:
    worker_a, ingress, worker_b = _store(), _store(), _store()
    try:
        [(first, reason)] = ingress.try_enqueue_processor_requests_batch(
            [_inventory("01")], **LIMITS
        )
        assert reason is None and first is not None
        lane_key = first.ordering_key()

        # The claim's ``now`` is taken once per consumer cycle, so it is
        # systematically earlier than the second sample's construction time
        # - which is what let the ``updated_at`` upsert guard wave the
        # revert through.
        claim_now = datetime.now(timezone.utc) - timedelta(seconds=1)
        second = _inventory("02")
        assert second.updated_at > claim_now

        claimed: list = []
        claim_open = threading.Event()
        commit_claim = threading.Event()

        def hold_claim() -> None:
            sql, params = worker_a.claim_active_processor_query(
                "worker-a",
                now=claim_now,
                lease_duration=LEASE,
                limit=16,
                include_paths={INVENTORY},
            )
            with worker_a._db.transaction():
                with worker_a._db.cursor() as cursor:
                    cursor.execute(sql, params)
                    claimed.extend(
                        worker_a._decode("processor_request", row[0])
                        for row in cursor.fetchall()
                    )
                claim_open.set()
                assert commit_claim.wait(timeout=30), (
                    "the held claim must be released by the test"
                )

        admission: list = []

        def admit() -> None:
            admission.extend(
                ingress.try_enqueue_processor_requests_batch([second], **LIMITS)
            )

        claim_thread = threading.Thread(target=hold_claim, daemon=True)
        claim_thread.start()
        assert claim_open.wait(timeout=30), (
            "worker-a must open its claim before admission starts"
        )
        assert [(item.request_id, item.lease_owner) for item in claimed] == [
            (first.request_id, "worker-a")
        ]

        admit_thread = threading.Thread(target=admit, daemon=True)
        admit_thread.start()
        _wait_for_blocked_backend(worker_b, second.request_id)
        commit_claim.set()
        claim_thread.join(timeout=30)
        admit_thread.join(timeout=30)
        assert not admit_thread.is_alive(), (
            "admission must finish once the claim commits"
        )

        # The claimed row keeps A's lease untouched ...
        leased = _row(worker_b, first.request_id)
        assert leased["status"] == "LEASED", leased
        assert leased["lease_owner"] == "worker-a", leased
        assert leased["leader_epoch"] == claimed[0].leader_epoch, leased
        assert leased["lease_token"] == claimed[0].lease_token, leased
        assert leased["body_base64"] == first.body_base64, (
            "the second sample must not be written over a leased row"
        )
        # ... and the second sample is neither dropped nor merged into it:
        # it is its own PENDING row.
        [(admitted, admit_reason)] = admission
        assert admitted is not None
        assert admit_reason is None, admit_reason
        assert admitted.request_id == second.request_id
        assert admitted.request_id != first.request_id
        assert _row(worker_b, second.request_id)["status"] == "PENDING"
        assert worker_b.processor_queue_stats()["depth"] == 2

        # A's fenced completion of the first sample now succeeds.
        item = claimed[0]
        [completed] = worker_a.complete_active_processor_requests_batch(
            [
                {
                    "request_id": item.request_id,
                    "owner_id": "worker-a",
                    "lane_epoch": item.leader_epoch,
                    "lease_token": item.lease_token,
                    "response_status": 200,
                    "response_content_type": "application/json",
                    "response_body_base64": "e30=",
                }
            ]
        )
        assert completed is not None
        assert completed.status is ProcessorRequestStatus.COMPLETED
        now = datetime.now(timezone.utc)
        assert _lane(worker_b, lane_key, now=now)[2] is False, (
            "completing must release the lane for the second sample"
        )
        next_claim = worker_b.claim_active_processor_requests(
            "worker-b", now=now, lease_duration=LEASE, limit=16
        )
        assert [entry.request_id for entry in next_claim] == [second.request_id]
    finally:
        worker_a.close()
        ingress.close()
        worker_b.close()


def test_batch_admission_merge_keeps_the_retry_schedule() -> None:
    """B-5: the merged row keeps ``max(retry_count)`` and the later ``not_before``
    instead of taking the fresh sample's zero / None."""

    store = _store()
    try:
        [(first, _)] = store.try_enqueue_processor_requests_batch(
            [_inventory("01")], **LIMITS
        )
        assert first is not None
        [claimed] = store.claim_active_processor_requests(
            "worker-a", now=datetime.now(timezone.utc), lease_duration=LEASE, limit=1
        )
        not_before = datetime.now(timezone.utc) + timedelta(seconds=30)
        store.release_active_processor_request(
            claimed.request_id,
            "worker-a",
            claimed.leader_epoch,
            claimed.lease_token,
            not_before=not_before,
            retry_count=3,
        )
        before = _row(store, first.request_id)
        assert (before["status"], before["retry_count"]) == ("PENDING", 3)

        [(merged, reason)] = store.try_enqueue_processor_requests_batch(
            [_inventory("02")], **LIMITS
        )

        assert reason == "coalesced"
        assert merged is not None and merged.request_id == first.request_id
        after = _row(store, first.request_id)
        assert after["status"] == "PENDING"
        assert after["retry_count"] == 3, after
        assert after["not_before"] == not_before, after
        assert merged.retry_count == 3
        assert merged.not_before == not_before
        assert after["body_base64"] != before["body_base64"], (
            "the payload still follows the latest sample"
        )
    finally:
        store.close()

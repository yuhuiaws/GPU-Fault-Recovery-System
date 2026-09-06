"""The leader-mode release path guards a COMPLETED row like its sibling, the
legacy reconcile shell is gone, and a multi-scope batch that fails half way
says what it committed.

FINAL-建议汇总 F-D9 (P1-76D, P1-77G, P2-19E, P2-75B, P2-14C). These paths are
unreachable from the deployed active-active processor today, which is why
they were left with the bugs their neighbours had already fixed: a rollback
to leader mode, or a new caller, would step on each one immediately.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

import pytest

from gpu_fault.processor import ProcessorRequestStatus
from gpu_fault.store import InMemoryStore, SqliteStore
from gpu_fault.store.postgres.processor_storage import PostgresProcessorStorageMixin
from gpu_fault.store.shared.processor_helpers import PartialEnqueueError
from tests._builders import build_store, processor_request
from tests.store._postgres_processor_claim_support import (
    _truncate,
    postgres_store_instance,
)

LEASE = timedelta(seconds=120)
OWNER = "leader-a"


@pytest.fixture(params=["memory", "sqlite", "postgres"])
def store(request, tmp_path):
    if request.param == "memory":
        yield build_store()
        return
    if request.param == "sqlite":
        sqlite = SqliteStore(str(tmp_path / "legacy-paths.db"))
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


def _lead_and_claim(store):
    now = datetime.now(timezone.utc)
    leadership = store.acquire_processor_leadership(
        OWNER, now=now, lease_duration=LEASE
    )
    request = store.enqueue_processor_request(
        processor_request("/v1/workflows/dispatch")
    )
    claimed = store.claim_processor_requests(
        OWNER, leadership.epoch, now=now, lease_duration=LEASE, limit=1
    )
    assert [item.request_id for item in claimed] == [request.request_id], (
        "the leader should have claimed the one pending row"
    )
    return claimed[0]


def test_leader_release_does_not_reopen_a_completed_request(store) -> None:
    claimed = _lead_and_claim(store)
    store.complete_processor_request(
        claimed.request_id,
        OWNER,
        claimed.leader_epoch,
        claimed.lease_token,
        response_status=200,
        response_content_type="application/json",
        response_body_base64="e30=",
    )

    # Completion leaves the fencing fields in place, so this release carries
    # the token that still matches; only the status guard can stop it.
    store.release_processor_request(
        claimed.request_id, OWNER, claimed.leader_epoch, claimed.lease_token
    )

    current = store.get_processor_request(claimed.request_id)
    assert current.status is ProcessorRequestStatus.COMPLETED
    assert current.response_status == 200


def test_leader_release_still_hands_back_a_leased_request(store) -> None:
    claimed = _lead_and_claim(store)

    store.release_processor_request(
        claimed.request_id,
        OWNER,
        claimed.leader_epoch,
        claimed.lease_token,
        retry_count=2,
    )

    current = store.get_processor_request(claimed.request_id)
    assert current.status is ProcessorRequestStatus.PENDING
    assert current.retry_count == 2
    assert current.lease_owner is None


def test_the_legacy_reconcile_shell_is_gone() -> None:
    # The shell returned 0 without touching anything, while six call sites
    # read as if legacy ``gpu_fault_objects`` rows were being reconciled.
    assert not hasattr(
        PostgresProcessorStorageMixin, "_maybe_reconcile_legacy_processor_requests"
    ), "the no-op reconcile shell should be deleted, not kept as a decoy"
    assert not hasattr(
        PostgresProcessorStorageMixin, "_reconcile_legacy_processor_requests"
    ), "the reconcile body had no caller once the shell was gone"


@pytest.fixture(params=["memory", "postgres"])
def batch_store(request):
    if request.param == "memory":
        yield build_store()
        return
    if not os.getenv("GPU_FAULT_TEST_POSTGRES_URL"):
        pytest.skip("GPU_FAULT_TEST_POSTGRES_URL is required")
    for postgres in postgres_store_instance():
        yield postgres
    _truncate()


def test_partial_enqueue_raises_with_committed_ids(batch_store, monkeypatch) -> None:
    store = batch_store
    first = processor_request("/v1/gpu-events/xid", cluster_id="cluster-a")
    second = processor_request("/v1/gpu-events/xid", cluster_id="cluster-b")

    def refuse_cluster_b(requests) -> None:
        if any(item.cluster_id == "cluster-b" for item in requests):
            raise RuntimeError("cluster-b storage is unavailable")

    if type(store) is InMemoryStore:
        single = store.try_enqueue_processor_request

        def failing_single(request, **kwargs):
            refuse_cluster_b([request])
            return single(request, **kwargs)

        monkeypatch.setattr(store, "try_enqueue_processor_request", failing_single)
    else:
        many = store._put_processor_queue_many

        def failing_many(requests, **kwargs):
            refuse_cluster_b(list(requests))
            return many(requests, **kwargs)

        monkeypatch.setattr(store, "_put_processor_queue_many", failing_many)

    with pytest.raises(PartialEnqueueError) as raised:
        store.try_enqueue_processor_requests_batch(
            [first, second], max_depth=100, max_cluster_depth=100
        )

    error = raised.value
    assert error.committed == [first.request_id]
    assert isinstance(error.cause, RuntimeError), (
        "the original failure rides along as cause"
    )
    assert "cluster-b" in str(error.cause)
    assert (
        store.get_processor_request(first.request_id).status
        is ProcessorRequestStatus.PENDING
    )

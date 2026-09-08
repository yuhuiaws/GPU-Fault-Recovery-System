"""``complete_active_processor_requests_batch`` answers per request, on every
backend (B-8, 2026-09-08).

Postgres returns ``None`` for each request whose lease no longer validates
and completes the rest; the shared composition the memory and SQLite stores
use raised ``StaleFencingTokenError`` at the first stale entry instead. Every
coordinator test runs on memory/SQLite, so the coordinator's ``result is
None`` branch - the one production takes - was unreachable under test.
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

LEASE = timedelta(seconds=120)


@pytest.fixture(params=["memory", "sqlite", "postgres"])
def store(request, tmp_path):
    if request.param == "memory":
        yield build_store()
        return
    if request.param == "sqlite":
        sqlite = SqliteStore(str(tmp_path / "batch-completion.db"))
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


def test_batch_completion_returns_none_per_fenced_request_and_completes_the_rest(store):
    for node in ("node-a", "node-b", "node-c"):
        store.enqueue_processor_request(
            processor_request(
                "/v1/collector-events/gpu-inventory",
                body=('{"node_id":"%s","gpus":[]}' % node).encode(),
                cluster_id="cluster-a",
            )
        )
    claimed = store.claim_active_processor_requests(
        "worker-a", now=datetime.now(timezone.utc), lease_duration=LEASE, limit=3
    )
    assert len(claimed) == 3
    completions = [
        {
            "request_id": item.request_id,
            "owner_id": "worker-a",
            "lane_epoch": item.leader_epoch,
            "lease_token": item.lease_token,
            "response_status": 200,
            "response_content_type": "application/json",
            "response_body_base64": "e30=",
        }
        for item in claimed
    ]
    completions[1]["lease_token"] = "not-the-lease-token"

    results = store.complete_active_processor_requests_batch(completions)

    assert [result is None for result in results] == [False, True, False]
    assert results[0].status is ProcessorRequestStatus.COMPLETED
    assert results[2].status is ProcessorRequestStatus.COMPLETED
    assert (
        store.get_processor_request(claimed[1].request_id).status
        is ProcessorRequestStatus.LEASED
    )

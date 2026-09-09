"""Latest-wins coalescing may only replace another sample of its own channel.

FINAL-建议汇总 F-D8 (P0-35A / P1-19A). A routine sample is ``coalescable()``
when its channel declares ``latest_wins`` and it sits on the routine tier. The
admission paths used to look up "the pending row on this lane" and overwrite
it -- but a node lane is shared by every channel that resolves to the node,
so a host-telemetry heartbeat replaced a pending XID fault, or the breach
sample a detector was still counting. What a routine sample may supersede is
the previous routine sample *of the same path*, nothing else.
"""

from __future__ import annotations

import os

import pytest

from gpu_fault.store import SqliteStore
from tests._builders import build_store, processor_request
from tests.store._postgres_processor_claim_support import (
    _truncate,
    postgres_store_instance,
)

LIMITS = {"max_depth": 20, "max_cluster_depth": 20}
CLUSTER = "cluster-coalesce"


@pytest.fixture(params=["memory", "sqlite", "postgres"])
def processor_store(request, tmp_path):
    if request.param == "memory":
        yield build_store()
        return
    if request.param == "sqlite":
        store = SqliteStore(str(tmp_path / "coalesce.db"))
        try:
            yield store
        finally:
            store.close()
        return
    if not os.getenv("GPU_FAULT_TEST_POSTGRES_URL"):
        pytest.skip("GPU_FAULT_TEST_POSTGRES_URL is required")
    # Leave the shared database as we found it: the store-contract tests that
    # run after this file claim whatever is pending.
    for postgres in postgres_store_instance():
        yield postgres
    _truncate()


def _enqueue(store, path: str, body: bytes):
    request = processor_request(path, body=body, cluster_id=CLUSTER)
    accepted, reason = store.try_enqueue_processor_request(request, **LIMITS)
    assert accepted is not None, reason
    return accepted, reason


def _summary(node: str = "node-a") -> bytes:
    return ('{"node_id":"' + node + '","summary":true,"seq":1}').encode()


def _depth(store) -> int:
    return store.processor_queue_stats()["depth"]


def test_routine_sample_does_not_replace_a_pending_fault_on_its_lane(
    processor_store,
) -> None:
    fault, _ = _enqueue(
        processor_store, "/v1/gpu-events/xid", b'{"node_id":"node-a","xid":79}'
    )

    _, reason = _enqueue(
        processor_store, "/v1/collector-events/host-telemetry", _summary()
    )

    assert reason is None
    current = processor_store.get_processor_request(fault.request_id)
    assert current.path == "/v1/gpu-events/xid"
    assert current.queue_priority() == fault.queue_priority()
    assert _depth(processor_store) == 2


def test_routine_sample_does_not_replace_a_pending_breach_sample(
    processor_store,
) -> None:
    breach, _ = _enqueue(
        processor_store,
        "/v1/collector-events/host-telemetry",
        b'{"node_id":"node-a","edge_filter_reasons":["threshold:xid-rate"],'
        b'"collection_errors":[]}',
    )
    assert breach.queue_priority() == 50

    _, reason = _enqueue(
        processor_store, "/v1/collector-events/host-telemetry", _summary()
    )

    assert reason is None
    current = processor_store.get_processor_request(breach.request_id)
    assert current.queue_priority() == 50
    assert current.body_base64 == breach.body_base64


def test_routine_sample_coalesces_only_with_its_own_path(processor_store) -> None:
    first, _ = _enqueue(
        processor_store, "/v1/collector-events/host-telemetry", _summary()
    )
    other_channel, other_reason = _enqueue(
        processor_store,
        "/v1/collector-events/gpu-metrics",
        b'{"node_id":"node-a","summary":true}',
    )
    second, reason = _enqueue(
        processor_store,
        "/v1/collector-events/host-telemetry",
        b'{"node_id":"node-a","summary":true,"seq":2}',
    )

    assert other_reason is None
    assert reason == "coalesced"
    assert second.request_id == first.request_id
    assert (
        processor_store.get_processor_request(first.request_id).body_base64
        == second.body_base64
    )
    assert other_channel.request_id != first.request_id
    assert processor_store.get_processor_request(other_channel.request_id).path == (
        "/v1/collector-events/gpu-metrics"
    )


def test_coalescing_keeps_the_pending_rows_retry_schedule(processor_store) -> None:
    """B-5: a fresh sample carries ``retry_count=0`` / ``not_before=None`` and
    must not reset the backoff a failed execution left on the pending row."""

    from datetime import datetime, timedelta, timezone

    first, _ = _enqueue(
        processor_store, "/v1/collector-events/host-telemetry", _summary()
    )
    [claimed] = processor_store.claim_active_processor_requests(
        "worker-a",
        now=datetime.now(timezone.utc),
        lease_duration=timedelta(seconds=120),
        limit=1,
    )
    not_before = datetime.now(timezone.utc) + timedelta(seconds=30)
    processor_store.release_active_processor_request(
        claimed.request_id,
        "worker-a",
        claimed.leader_epoch,
        claimed.lease_token,
        not_before=not_before,
        retry_count=2,
    )

    merged, reason = _enqueue(
        processor_store,
        "/v1/collector-events/host-telemetry",
        b'{"node_id":"node-a","summary":true,"seq":2}',
    )

    assert reason == "coalesced"
    assert merged.request_id == first.request_id
    stored = processor_store.get_processor_request(first.request_id)
    assert stored.retry_count == 2
    assert stored.not_before == not_before
    assert b'"seq":2' in stored.body()

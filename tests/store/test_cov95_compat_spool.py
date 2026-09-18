from __future__ import annotations

from datetime import timedelta

import pytest

from tests.store._cov95_compat_support import HOST_PATH, NOW, request_model
from tests.store._cov95_compat_support import (
    compat_store_fixture as compat_store_fixture,
)

LEASE = timedelta(seconds=30)


@pytest.mark.parametrize(
    "limits",
    [
        {"limit": 0},
        {"limit": -1},
        {"limit": 2, "max_bytes": 0},
        {"limit": 2, "max_bytes": -1},
    ],
)
def test_nonpositive_spool_claim_budget_does_not_consume_a_sample(compat_store, limits):
    store = compat_store
    request = request_model("sample")
    store.try_spool_telemetry_requests(
        [request], max_depth=2, max_cluster_depth=2, now=NOW
    )
    assert (
        store.claim_telemetry_spool("owner", now=NOW, lease_duration=LEASE, **limits)
        == []
    )
    assert store.telemetry_spool_stats(now=NOW)["leased"] == 0
    (row,) = store.claim_telemetry_spool(
        "owner", now=NOW, lease_duration=LEASE, limit=1
    )
    assert row.attempts == 1
    assert row.request_id == request.request_id


def test_abandon_cannot_change_a_new_revision_or_a_completed_row(compat_store):
    store = compat_store
    original = request_model(
        "old", payload={"node_id": "node", "summary": True, "sequence": 1}
    )
    replacement = request_model(
        "new", payload={"node_id": "node", "summary": True, "sequence": 2}
    )
    store.try_spool_telemetry_requests(
        [original], max_depth=1, max_cluster_depth=1, now=NOW
    )
    (old,) = store.claim_telemetry_spool(
        "old-owner", now=NOW, lease_duration=LEASE, limit=1
    )
    assert store.try_spool_telemetry_requests(
        [replacement], max_depth=1, max_cluster_depth=1, now=NOW + timedelta(seconds=1)
    ) == [(replacement, "coalesced")]
    assert (
        store.abandon_telemetry_spool_claims([old], now=NOW + timedelta(seconds=1)) == 0
    )
    (current,) = store.claim_telemetry_spool(
        "current-owner", now=NOW + timedelta(seconds=1), lease_duration=LEASE, limit=1
    )
    assert current.payload["sequence"] == 2
    assert current.revision == old.revision + 1
    assert current.attempts == 2
    assert store.complete_telemetry_spool([old]) == 0
    assert store.release_telemetry_spool(
        [old], now=NOW, backoff=timedelta(hours=1)
    ) == (0, 0)
    assert (
        store.abandon_telemetry_spool_claims([current], now=NOW + timedelta(seconds=1))
        == 1
    )
    (retried,) = store.claim_telemetry_spool(
        "retry-owner", now=NOW + timedelta(seconds=1), lease_duration=LEASE, limit=1
    )
    assert retried.attempts == 2
    assert store.complete_telemetry_spool([retried]) == 1
    assert store.abandon_telemetry_spool_claims([retried], now=NOW) == 0
    assert store.telemetry_spool_stats(now=NOW)["depth"] == 0


def test_byte_limited_claim_can_make_progress_on_one_oversized_row(compat_store):
    store = compat_store
    requests = [
        request_model(f"sample-{index}", node_id=f"node-{index}") for index in range(2)
    ]
    assert [
        reason
        for _, reason in store.try_spool_telemetry_requests(
            requests, max_depth=2, max_cluster_depth=2, now=NOW
        )
    ] == [None, None]
    claimed = store.claim_telemetry_spool(
        "owner", now=NOW, lease_duration=LEASE, limit=2, max_bytes=1, path=HOST_PATH
    )
    assert len(claimed) == 1
    assert claimed[0].payload_bytes > 1
    stats = store.telemetry_spool_stats(now=NOW)
    assert stats["depth"] == 2
    assert stats["leased"] == 1
    assert stats["leased_bytes"] == claimed[0].payload_bytes
    assert store.complete_telemetry_spool(claimed) == 1
    remaining = store.claim_telemetry_spool(
        "other", now=NOW, lease_duration=LEASE, limit=2
    )
    assert len(remaining) == 1
    assert remaining[0].spool_key != claimed[0].spool_key


def test_spool_depth_is_independent_of_queue_depth_and_cluster_cap(compat_store):
    store = compat_store
    store.enqueue_processor_request(request_model("queued"))
    first = request_model("first", node_id="node-a")
    same_cluster = request_model("same-cluster", node_id="node-b")
    other_cluster = request_model("other-cluster", node_id="node-a", cluster_id="other")
    result = store.try_spool_telemetry_requests(
        [first, same_cluster, other_cluster], max_depth=2, max_cluster_depth=1, now=NOW
    )
    assert [reason for _, reason in result] == [None, "cluster", None]
    assert store.processor_queue_stats(now=NOW)["depth"] == 1
    assert store.telemetry_spool_stats(now=NOW)["depth"] == 2
    assert store.try_spool_telemetry_requests(
        [request_model("overflow", cluster_id="third")],
        max_depth=2,
        max_cluster_depth=1,
        now=NOW,
    ) == [(None, "global")]

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pytest

from gpu_fault.processor import ProcessorRequestStatus
from gpu_fault.store import NotFoundError
from tests.store._cov95_compat_support import (
    FAULT_PATH,
    GPU_PATH,
    HOST_PATH,
    NOW,
    request_model,
)
from tests.store._cov95_compat_support import (
    compat_store_fixture as compat_store_fixture,
)

LEASE = timedelta(seconds=120)


def test_duplicate_admission_preserves_existing_payload_even_at_capacity(compat_store):
    store = compat_store
    original = request_model("same-id", payload={"node_id": "node-a", "sequence": 1})
    accepted, reason = store.try_enqueue_processor_request(
        original, max_depth=1, max_cluster_depth=1
    )
    assert reason is None
    assert accepted.request_id == original.request_id
    changed = request_model("same-id", payload={"node_id": "node-a", "sequence": 2})
    repeated, reason = store.try_enqueue_processor_request(
        changed, max_depth=1, max_cluster_depth=1
    )
    assert reason is None
    assert repeated.body() == original.body()
    assert store.enqueue_processor_request(changed).body() == original.body()
    assert store.processor_queue_stats(now=NOW)["depth"] == 1


def test_routine_coalescing_preserves_slot_and_never_overwrites_a_fault(compat_store):
    store = compat_store
    original = request_model(
        "old-sample", payload={"node_id": "node-a", "summary": True, "sequence": 1}
    )
    fault = request_model(
        "fault",
        path=FAULT_PATH,
        node_id="node-a",
        payload={"node_id": "node-a", "xid": 79},
    )
    store.enqueue_processor_request(fault)
    store.enqueue_processor_request(original)
    incoming = request_model(
        "new-sample",
        payload={"node_id": "node-a", "summary": True, "sequence": 2},
        created_at=NOW + timedelta(seconds=1),
        updated_at=NOW + timedelta(seconds=1),
    )
    assert original.coalescable(), "fixture must exercise a routine latest-wins lane"
    accepted, reason = store.try_enqueue_processor_request(
        incoming, max_depth=2, max_cluster_depth=2
    )
    assert reason == "coalesced"
    assert accepted.request_id == original.request_id
    assert accepted.body() == incoming.body()
    assert store.get_processor_request(fault.request_id).body() == fault.body()
    assert store.processor_fault_backlog_depth() == 1
    assert store.processor_queue_stats(now=NOW + timedelta(seconds=2))["depth"] == 2
    with pytest.raises(NotFoundError):
        store.get_processor_request(incoming.request_id)


@pytest.mark.parametrize(
    "limits,reason",
    [
        ({"max_depth": 1, "max_cluster_depth": 5}, "global"),
        (
            {"max_depth": 2, "max_cluster_depth": 5, "reserved_fault_depth": 1},
            "global_reserved",
        ),
        ({"max_depth": 5, "max_cluster_depth": 1}, "cluster"),
        (
            {"max_depth": 5, "max_cluster_depth": 2, "reserved_cluster_fault_depth": 1},
            "cluster_reserved",
        ),
    ],
)
def test_admission_bounds_report_the_exact_budget_and_preserve_state(
    compat_store, limits, reason
):
    store = compat_store
    store.enqueue_processor_request(request_model("existing", node_id="node-a"))
    candidate = request_model("candidate", node_id="node-b")
    assert store.try_enqueue_processor_request(candidate, **limits) == (None, reason)
    assert store.processor_queue_stats(now=NOW)["depth"] == 1
    with pytest.raises(NotFoundError):
        store.get_processor_request(candidate.request_id)
    if reason.endswith("_reserved"):
        critical = request_model(
            "critical",
            path=FAULT_PATH,
            node_id="node-b",
            payload={"node_id": "node-b", "xid": 79},
        )
        accepted, why = store.try_enqueue_processor_request(critical, **limits)
        assert accepted.request_id == critical.request_id
        assert why is None
        assert store.processor_fault_backlog_depth() == 1


def test_admission_cluster_budget_does_not_count_another_cluster(compat_store):
    store = compat_store
    store.enqueue_processor_request(request_model("foreign", cluster_id="other"))
    incoming = request_model("local")
    accepted, reason = store.try_enqueue_processor_request(
        incoming, max_depth=3, max_cluster_depth=1, global_admission_guard=1
    )
    assert accepted.request_id == "local"
    assert reason is None
    assert store.processor_queue_stats(now=NOW)["by_cluster"] == {
        "other": 1,
        "cluster-local": 1,
    }


def test_queue_statistics_preserve_unscoped_work_and_bound_negative_age(compat_store):
    store = compat_store
    store.enqueue_processor_request(
        request_model("old", created_at=NOW - timedelta(seconds=20))
    )
    store.enqueue_processor_request(
        request_model(
            "future", node_id="future", created_at=NOW + timedelta(seconds=20)
        )
    )
    store.enqueue_processor_request(
        request_model(
            "unscoped", cluster_id=None, created_at=NOW - timedelta(seconds=10)
        )
    )
    assert store.processor_queue_stats(now=NOW) == {
        "depth": 3,
        "oldest_age_seconds": 20.0,
        "by_cluster": {"cluster-local": 2, "__unscoped__": 1},
        "oldest_age_by_cluster": {"cluster-local": 20.0, "__unscoped__": 10.0},
    }


def test_scope_queries_and_leadership_claims_use_real_request_identity(compat_store):
    store = compat_store
    now = datetime.now(UTC)
    request = request_model(
        "scoped",
        path=FAULT_PATH,
        payload={
            "node_id": "node-a",
            "xid": 79,
            "job_id": "train",
            "attempt_id": "attempt",
        },
        created_at=now - timedelta(seconds=1),
        updated_at=now - timedelta(seconds=1),
    )
    store.enqueue_processor_request(request)
    scope = set(request.correlation_scope_keys)
    assert scope, "real HTTP request parsing must derive correlation scopes"
    assert (
        store.has_incomplete_processor_requests_for_scopes("cluster-local", scope)
        is True
    )
    assert (
        store.has_incomplete_processor_requests_for_scopes("cluster-local", set())
        is True
    )
    assert (
        store.has_incomplete_processor_requests_for_scopes("cluster-local", {"unknown"})
        is False
    )
    assert store.has_incomplete_processor_requests_for_scopes("other", scope) is False
    assert store.get_processor_leadership() is None
    assert (
        store.claim_processor_requests(
            "leader", 1, now=now, lease_duration=LEASE, limit=1
        )
        == []
    )
    leadership = store.acquire_processor_leadership(
        "leader", now=now, lease_duration=LEASE
    )
    for owner, epoch in (("other", leadership.epoch), ("leader", leadership.epoch + 1)):
        assert (
            store.claim_processor_requests(
                owner, epoch, now=now, lease_duration=LEASE, limit=1
            )
            == []
        )
    (claimed,) = store.claim_processor_requests(
        "leader", leadership.epoch, now=now, lease_duration=LEASE, limit=1
    )
    completed = store.complete_processor_request(
        request.request_id,
        "leader",
        leadership.epoch,
        claimed.lease_token,
        response_status=200,
        response_content_type="application/json",
        response_body_base64="e30=",
    )
    assert completed.status is ProcessorRequestStatus.COMPLETED
    assert (
        store.has_incomplete_processor_requests_for_scopes("cluster-local", scope)
        is False
    )
    assert (
        store.has_incomplete_processor_requests_for_scopes("cluster-local", set())
        is False
    )


def test_leadership_claim_keeps_a_live_lane_exclusive_and_honors_limit(compat_store):
    store = compat_store
    now = datetime.now(UTC)
    for request_id, node_id in (
        ("a-first", "node-a"),
        ("a-second", "node-a"),
        ("b", "node-b"),
    ):
        store.enqueue_processor_request(
            request_model(
                request_id,
                path=FAULT_PATH,
                node_id=node_id,
                payload={"node_id": node_id, "xid": 79},
                created_at=now - timedelta(seconds=1),
                updated_at=now - timedelta(seconds=1),
            )
        )
    leadership = store.acquire_processor_leadership(
        "leader", now=now, lease_duration=LEASE
    )
    first = store.claim_processor_requests(
        "leader", leadership.epoch, now=now, lease_duration=LEASE, limit=1
    )
    assert [item.request_id for item in first] == ["a-first"]
    second = store.claim_processor_requests(
        "leader", leadership.epoch, now=now, lease_duration=LEASE, limit=5
    )
    assert [item.request_id for item in second] == ["b"]
    assert (
        store.get_processor_request("a-second").status is ProcessorRequestStatus.PENDING
    )
    assert (
        store.claim_processor_requests(
            "leader", leadership.epoch, now=now + LEASE, lease_duration=LEASE, limit=5
        )
        == []
    )


@pytest.mark.parametrize(
    "options,blocked",
    [
        ({}, True),
        ({"include_paths": {HOST_PATH}}, True),
        ({"include_paths": {GPU_PATH}}, False),
        ({"exclude_paths": {HOST_PATH}}, False),
        ({"exclude_paths": {GPU_PATH}}, True),
        ({"include_paths": set()}, False),
    ],
)
def test_lane_blocked_query_distinguishes_filtered_backlog_from_idle(
    compat_store, options, blocked
):
    store = compat_store
    now = datetime.now(UTC)
    assert store.active_backlog_is_lane_blocked(now=now, **options) is False
    store.enqueue_processor_request(
        request_model(
            "holder",
            node_id="node-a",
            created_at=now - timedelta(seconds=2),
            updated_at=now - timedelta(seconds=2),
        )
    )
    (held,) = store.claim_active_processor_requests(
        "owner", now=now, lease_duration=LEASE, limit=1
    )
    store.enqueue_processor_request(
        request_model(
            "pending",
            node_id="node-a",
            created_at=now - timedelta(seconds=1),
            updated_at=now - timedelta(seconds=1),
        )
    )
    store.enqueue_processor_request(
        request_model(
            "deferred",
            path=FAULT_PATH,
            node_id="node-b",
            payload={"node_id": "node-b", "xid": 79},
            not_before=now + LEASE * 2,
        )
    )
    assert held.ordering_key() == store.get_processor_request("pending").ordering_key()
    assert store.active_backlog_is_lane_blocked(now=now, **options) is blocked
    assert store.active_backlog_is_lane_blocked(now=now + LEASE, **options) is False
    assert (
        store.get_processor_request("pending").status is ProcessorRequestStatus.PENDING
    )


def test_active_claim_path_filters_do_not_consume_other_paths(compat_store):
    store = compat_store
    now = datetime.now(UTC)
    for index, path in enumerate((HOST_PATH, GPU_PATH, FAULT_PATH)):
        store.enqueue_processor_request(
            request_model(
                f"item-{index}",
                path=path,
                node_id=f"node-{index}",
                payload={"node_id": f"node-{index}", "summary": True, "xid": 79},
            )
        )
    claimed = store.claim_active_processor_requests(
        "owner",
        now=now,
        lease_duration=LEASE,
        limit=5,
        include_paths={HOST_PATH, GPU_PATH},
        exclude_paths={GPU_PATH},
    )
    assert [item.path for item in claimed] == [HOST_PATH]
    assert (
        store.get_processor_request("item-1").status is ProcessorRequestStatus.PENDING
    )
    assert (
        store.get_processor_request("item-2").status is ProcessorRequestStatus.PENDING
    )
    assert json.loads(claimed[0].body())["node_id"] == "node-0"

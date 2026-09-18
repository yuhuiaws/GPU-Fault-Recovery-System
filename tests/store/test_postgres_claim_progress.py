"""Bounded progress past observation-held faults on an owned PostgreSQL."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from datetime import datetime, timedelta, timezone

import pytest

from gpu_fault.channel_registry import (
    COLLECTOR_HEALTH_PATH,
    FABRIC_MANAGER_PATH,
    NVIDIA_KERNEL_PATH,
    WORKLOAD_OBSERVATIONS_PATH,
)
from gpu_fault.processor.models import ProcessorRequest
from gpu_fault.store import PostgresStore
from tests.store._postgres_processor_claim_support import (
    POSTGRES_URL,
    REQUEST_LEASE,
    _truncate,
    postgres_store_instance,
)

pytestmark = pytest.mark.skipif(
    not POSTGRES_URL, reason="requires the parent-allocated PostgreSQL"
)


@pytest.fixture
def store():
    instances = postgres_store_instance()
    try:
        yield next(instances)
    finally:
        instances.close()
        _truncate()


@pytest.fixture
def connection():
    import psycopg

    with psycopg.connect(POSTGRES_URL, autocommit=True) as connected:
        yield connected


def request(path, cluster, key, at, **body):
    return ProcessorRequest.from_http(
        method="POST",
        path=path,
        query="",
        body=json.dumps({"cluster_id": cluster, **body}).encode(),
        content_type="application/json",
        cluster_id=cluster,
    ).model_copy(update={"request_id": key, "created_at": at, "updated_at": at})


def seed_prefix(store, at, *, count=68):
    nodes = [f"held-node-{index:04d}" for index in range(count)]
    observation = request(
        WORKLOAD_OBSERVATIONS_PATH,
        "held-cluster",
        "held-observation",
        at - timedelta(seconds=2),
        job_id="held-job",
        attempt_id="held-attempt",
        containers=[{"node_id": node} for node in nodes],
    ).model_copy(update={"not_before": at + timedelta(seconds=30)})
    store.enqueue_processor_request(observation)
    faults = []
    for index, node in enumerate(nodes):
        fault = request(
            NVIDIA_KERNEL_PATH,
            "held-cluster",
            f"held-fault-{index:04d}",
            at - timedelta(seconds=1) + timedelta(microseconds=index),
            node_id=node,
        )
        store.enqueue_processor_request(fault)
        faults.append(fault)
    return observation, faults


def independent(store, at, *, key="independent", node="independent-node", path=None):
    item = request(
        path or NVIDIA_KERNEL_PATH, "independent-cluster", key, at, node_id=node
    )
    store.enqueue_processor_request(item)
    return item


def claim(store, at, *, limit=4, owner="claim-owner", **filters):
    return store.claim_active_processor_requests(
        owner,
        now=at,
        lease_duration=REQUEST_LEASE,
        limit=limit,
        **(filters or {"exclude_paths": {WORKLOAD_OBSERVATIONS_PATH}}),
    )


def complete(store, item):
    store.complete_active_processor_request(
        item.request_id,
        item.lease_owner,
        item.leader_epoch,
        item.lease_token,
        response_status=200,
        response_content_type="application/json",
        response_body_base64="e30=",
        path=item.path,
    )


@pytest.mark.parametrize("prefix_count", [68, 205])
def test_repeated_bounded_claims_reach_an_unrelated_cluster(store, prefix_count):
    at = datetime.now(timezone.utc)
    _, held = seed_prefix(store, at, count=prefix_count)
    ready = independent(store, at)

    claimed = []
    for _ in range(prefix_count // 68 + 2):
        rows = claim(store, at)
        assert len(rows) <= 4, "continuation must not enlarge the claim limit"
        claimed.extend(rows)
        if rows:
            break

    assert [row.request_id for row in claimed] == [ready.request_id], (
        "a blocked prefix must not keep the independent cluster outside every claim"
    )
    assert all(
        store.get_processor_request(item.request_id).lease_owner is None
        for item in held
    ), "progress must not bypass observation-before-fault ordering"


def test_continuation_rechecks_new_higher_priority_head_work(store):
    at = datetime.now(timezone.utc)
    seed_prefix(store, at)
    later = independent(store, at)
    assert claim(store, at, limit=1) == [], "the first bounded window is held"
    urgent = independent(
        store,
        at + timedelta(microseconds=1),
        key="urgent",
        path="/v1/attempts/terminal",
    )

    rows = claim(store, at, limit=1)

    assert [row.request_id for row in rows] == [urgent.request_id], (
        "seek progress must not hide newly arrived higher-priority work"
    )
    following = []
    for _ in range(4):
        following.extend(claim(store, at, limit=1))
        if following:
            break
    assert [row.request_id for row in following] == [later.request_id], (
        "head priority must not discard later continuation progress permanently"
    )


def test_continuation_rechecks_a_newly_unblocked_head(store):
    at = datetime.now(timezone.utc)
    observation, held = seed_prefix(store, at)
    independent(store, at)
    assert claim(store, at) == [], "the first bounded window is held"
    (observed,) = claim(
        store,
        at + timedelta(seconds=31),
        limit=1,
        include_paths={WORKLOAD_OBSERVATIONS_PATH},
    )
    assert observed.request_id == observation.request_id, (
        "the related observation must settle before its faults"
    )
    complete(store, observed)

    rows = claim(store, at + timedelta(seconds=31))

    assert [row.request_id for row in rows] == [item.request_id for item in held[:4]], (
        "newly unblocked oldest faults must precede the seek window"
    )


def test_progress_preserves_busy_lanes_and_deferred_strict_retry_barriers(store):
    at = datetime.now(timezone.utc)
    busy = independent(store, at - timedelta(seconds=3), key="busy", node="busy-node")
    (leased,) = claim(store, at, limit=1)
    assert leased.request_id == busy.request_id, "the peer must own the busy lane"
    seed_prefix(store, at)
    independent(store, at, key="busy-following", node="busy-node")
    deferred = request(
        NVIDIA_KERNEL_PATH,
        "independent-cluster",
        "deferred",
        at,
        node_id="deferred-node",
    ).model_copy(update={"not_before": at + timedelta(seconds=60)})
    store.enqueue_processor_request(deferred)
    independent(store, at, key="deferred-following", node="deferred-node")
    ready = independent(store, at, key="free")

    claimed = []
    for _ in range(4):
        claimed.extend(claim(store, at))

    assert [row.request_id for row in claimed] == [ready.request_id], (
        "continuation must retain both the live lane fence and the STRICT retry hold"
    )


def test_multiple_path_windows_do_not_skip_the_earlier_paths_unseen_rows(store):
    at = datetime.now(timezone.utc)
    _, held = seed_prefix(store, at - timedelta(seconds=2), count=205)
    for index, item in enumerate(held[:68]):
        created_at = at - timedelta(seconds=1) + timedelta(microseconds=index)
        store.enqueue_processor_request(
            item.model_copy(
                update={
                    "path": FABRIC_MANAGER_PATH,
                    "request_id": f"fabric-{index:04d}",
                    "created_at": created_at,
                    "updated_at": created_at,
                }
            )
        )
    ready = independent(store, at - timedelta(seconds=2))

    claimed = []
    for _ in range(5):
        claimed.extend(
            claim(store, at, include_paths={NVIDIA_KERNEL_PATH, FABRIC_MANAGER_PATH})
        )
        if claimed:
            break

    assert [row.request_id for row in claimed] == [ready.request_id], (
        "a later path's full-window boundary must not skip another path's backlog"
    )


def test_aging_rows_cannot_mask_a_full_observation_blocked_fault_window(store):
    at = datetime.now(timezone.utc)
    seed_prefix(store, at)
    ready = independent(store, at)
    for index in range(8):
        store.enqueue_processor_request(
            request(
                COLLECTOR_HEALTH_PATH,
                "summary-cluster",
                f"summary-{index}",
                at - timedelta(seconds=45),
                node_id=f"summary-node-{index}",
                collector="NVIDIA_KERNEL",
                edge_filter_reasons=["health-summary"],
            )
        )

    first = claim(store, at)
    second = claim(store, at)

    assert len(first) == 4 and all(
        row.path == COLLECTOR_HEALTH_PATH for row in first
    ), "the aging window should retain service while the raw prefix is held"
    assert second[0].request_id == ready.request_id, (
        "aging completions must not prevent seeking the next fault window"
    )
    assert len(second) <= 4, "continuation must preserve the original claim limit"


def test_independent_replicas_progress_without_duplicate_lane_leases(store):
    at = datetime.now(timezone.utc)
    seed_prefix(store, at)
    ready_ids = {
        independent(
            store, at, key=f"ready-{index}", node=f"ready-node-{index}"
        ).request_id
        for index in range(8)
    }
    assert POSTGRES_URL is not None, "the native fixture requires its granted URL"
    with closing(PostgresStore(POSTGRES_URL, initialize_schema=False)) as peer:
        assert claim(store, at, owner="replica-a") == [], "the first prefix is held"
        assert claim(peer, at, owner="replica-b") == [], (
            "each replica starts at the head"
        )
        seen = set()
        with ThreadPoolExecutor(max_workers=2) as workers:
            for _ in range(4):
                futures = [
                    workers.submit(claim, store, at, owner="replica-a"),
                    workers.submit(claim, peer, at, owner="replica-b"),
                ]
                for future in futures:
                    rows = future.result(timeout=10)
                    assert len(rows) <= 4, "each replica retains its own claim limit"
                    ids = {row.request_id for row in rows}
                    assert ids.isdisjoint(seen), (
                        "a leased request must not be claimed twice"
                    )
                    assert ids <= ready_ids, "no held fault may bypass its observation"
                    seen.update(ids)
                if seen == ready_ids:
                    break

    assert seen == ready_ids, "replicas must reach all unrelated ready lanes"


def test_seek_plan_bounds_interlock_input_and_uses_the_tuple_index(
    store, connection, request
):
    from tests.store.test_postgres_state_tables import plan_nodes

    at = datetime.now(timezone.utc)
    seed_prefix(store, at, count=205)
    independent(store, at)
    assert claim(store, at) == [], "the first held window establishes continuation"
    query, parameters = store.claim_active_processor_query(
        "bounded-owner",
        now=at,
        lease_duration=REQUEST_LEASE,
        limit=4,
        exclude_paths={WORKLOAD_OBSERVATIONS_PATH},
    )
    # Exclude alternative sorts to prove the seek can be served in index order,
    # following the existing order-index tests. Actual row bounds are measured.
    with connection.transaction():
        for setting in ("enable_seqscan", "enable_sort", "enable_incremental_sort"):
            connection.execute(f"SET LOCAL {setting}=off")
        plan = connection.execute(
            "EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) " + query, parameters
        ).fetchone()[0]
    request.node.user_properties.append(("bounded_seek_plan", json.dumps(plan)))
    nodes = list(plan_nodes(plan))
    windows = {
        node.get("Subplan Name"): node
        for node in nodes
        if node.get("Subplan Name") in {"CTE raw_window", "CTE claim_window"}
    }

    assert parameters["window_limit"] == 68, "the original window budget must not grow"
    assert windows["CTE raw_window"]["Actual Rows"] <= 68, (
        "the advancing raw probe must stay bounded"
    )
    assert windows["CTE claim_window"]["Actual Rows"] <= 3 * 68, (
        "head, seek and aging are the entire bounded interlock input"
    )
    assert any(
        node.get("Index Name") == "gpu_fault_processor_queue_priority_claim"
        and "ROW(priority, created_at, request_id) > ROW(" in node.get("Index Cond", "")
        for node in nodes
    ), "continuation must seek the tuple index, not discard a growing OFFSET"


def test_a_strict_retry_rearmed_behind_the_cursor_precedes_its_newer_lane_request(
    store,
):
    at = datetime.now(timezone.utc)
    seed_prefix(store, at, count=205)
    older = request(
        NVIDIA_KERNEL_PATH,
        "independent-cluster",
        "older-retry",
        at - timedelta(seconds=1) + timedelta(microseconds=100),
        node_id="rearmed-node",
    ).model_copy(update={"not_before": at + timedelta(seconds=30)})
    store.enqueue_processor_request(older)
    independent(store, at, key="newer-lane-request", node="rearmed-node")
    assert claim(store, at) == [], "the retry's future deadline must hold its lane"
    assert claim(store, at) == [], "the next bounded window is also held"

    rows = []
    for _ in range(8):
        rows = claim(store, at + timedelta(seconds=31))
        if rows:
            break

    assert [row.request_id for row in rows] == [older.request_id], (
        "a newer seek-window request must not overtake a rearmed STRICT retry"
    )

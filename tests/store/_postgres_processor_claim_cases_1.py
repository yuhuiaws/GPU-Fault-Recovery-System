"""Real-Postgres coverage for the claim, lane and remote command SQL.

``test_processor_leadership.py`` parametrizes only the memory and sqlite
stores, so the PostgresStore overrides of these paths - which are the ones
that actually run in regional mode - had no test at all. Everything here
asserts a contract that lives in SQL: the bounded claim window still
honours priority, lane exclusivity and the observation/fault interlock;
lane retention only retires idle lanes; and the batched remote command
claim keeps the fencing rules of the per-command version it replaced.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timedelta, timezone
from threading import Event, Thread

import pytest

from gpu_fault.processor import ProcessorRequestStatus
from gpu_fault.store import PostgresStore
from tests._builders import processor_request
from tests.store._postgres_processor_claim_support import (
    POSTGRES_URL,
    REQUEST_LEASE,
    _evidence,
    _fault,
    _request,
    _telemetry,
)

pytestmark = pytest.mark.skipif(
    not POSTGRES_URL, reason="GPU_FAULT_TEST_POSTGRES_URL is not configured"
)


def test_postgres_spool_claim_respects_payload_byte_budget(store) -> None:
    requests = []
    for index in range(3):
        body = json.dumps(
            {
                "cluster_id": "cluster-a",
                "node_id": f"node-{index}",
                "batch_id": f"batch-{index}",
                "edge_filter_reasons": ["health-summary"],
                "padding": "x" * 1024,
            }
        ).encode()
        requests.append(
            processor_request("/v1/collector-events/gpu-metrics", body=body)
        )
    store.try_spool_telemetry_requests(requests, max_depth=100, max_cluster_depth=100)

    claimed = store.claim_telemetry_spool(
        "spool-worker-a",
        now=datetime.now(timezone.utc),
        lease_duration=REQUEST_LEASE,
        limit=10,
        max_bytes=len(requests[0].body()) + 64,
    )

    assert len(claimed) == 1
    assert claimed[0].payload_bytes > 0


def test_postgres_spool_claim_uses_the_path_index(store) -> None:
    requests = [
        _request(
            "/v1/collector-events/gpu-metrics",
            body=b'{"node_id":"gpu-node","summary":true}',
        ),
        _request(
            "/v1/collector-events/host-telemetry",
            body=b'{"node_id":"host-node","summary":true}',
        ),
    ]
    store.try_spool_telemetry_requests(requests, max_depth=100, max_cluster_depth=100)

    claimed = store.claim_telemetry_spool(
        "spool-worker-a",
        now=datetime.now(timezone.utc),
        lease_duration=REQUEST_LEASE,
        limit=10,
        path="/v1/collector-events/host-telemetry",
    )
    with store._db.cursor() as cursor:
        cursor.execute(
            """
            SELECT to_regclass(
                'gpu_fault_telemetry_spool_path_available'
            )
            """
        )
        index_name = cursor.fetchone()[0]

    assert [item.path for item in claimed] == ["/v1/collector-events/host-telemetry"]
    assert index_name == "gpu_fault_telemetry_spool_path_available"


def test_postgres_spool_claims_and_completes_sixty_four_rows(store) -> None:
    requests = [_telemetry(f"node-batch-{index:03d}") for index in range(64)]
    results = store.try_spool_telemetry_requests(
        requests, max_depth=100, max_cluster_depth=100
    )

    claimed = store.claim_telemetry_spool(
        "spool-worker-batch",
        now=datetime.now(timezone.utc),
        lease_duration=timedelta(seconds=60),
        limit=64,
        max_bytes=8 * 1024 * 1024,
        path="/v1/collector-events/host-telemetry",
    )
    completed = store.complete_telemetry_spool(claimed)

    assert all(item[0] is not None for item in results), (
        "expected all(item[0] is not None for item in results) to be truthy"
    )
    assert len(claimed) == 64
    assert completed == 64
    assert store.telemetry_spool_stats()["depth"] == 0


def test_postgres_spool_abandon_preserves_retry_budget(store) -> None:
    request = _request(
        "/v1/collector-events/gpu-metrics",
        body=json.dumps(
            {
                "cluster_id": "cluster-a",
                "node_id": "node-a",
                "batch_id": "batch-a",
                "edge_filter_reasons": ["health-summary"],
            }
        ).encode(),
    )
    store.try_spool_telemetry_requests([request], max_depth=100, max_cluster_depth=100)
    at = datetime.now(timezone.utc)

    for _ in range(store.TELEMETRY_SPOOL_MAX_ATTEMPTS + 2):
        claimed = store.claim_telemetry_spool(
            "spool-worker-a", now=at, lease_duration=REQUEST_LEASE, limit=1
        )
        assert len(claimed) == 1
        assert claimed[0].attempts == 1
        assert store.abandon_telemetry_spool_claims(claimed, now=at) == 1

    assert store.telemetry_spool_stats()["depth"] == 1


def test_claim_window_returns_oldest_and_bounds_the_scan(store) -> None:
    """The LIMIT now lives in the innermost scan.

    Each request is on its own lane, so a window of ``limit *
    multiplier`` rows must still hand back exactly ``limit`` of them, and
    they must be the oldest - otherwise the bound would have turned
    FIFO into arbitrary order.
    """
    queued = [_telemetry(f"node-{index:03d}") for index in range(80)]
    for item in queued:
        store.enqueue_processor_request(item)

    claimed = store.claim_active_processor_requests(
        "pod-a", now=datetime.now(timezone.utc), lease_duration=REQUEST_LEASE, limit=5
    )

    assert len(claimed) == 5
    expected = {item.request_id for item in queued[:5]}
    assert {item.request_id for item in claimed} == expected


def test_claim_prefers_faults_over_telemetry_across_the_window(store) -> None:
    """Priority must survive the window's ORDER BY.

    The fault is enqueued last, so a window ordered by arrival alone
    would push it past the telemetry backlog.
    """
    for index in range(40):
        store.enqueue_processor_request(_telemetry(f"node-{index:03d}"))
    fault = _request("/v1/gpu-events/xid", body=b'{"node_id":"node-fault","xid":79}')
    store.enqueue_processor_request(fault)

    claimed = store.claim_active_processor_requests(
        "pod-a", now=datetime.now(timezone.utc), lease_duration=REQUEST_LEASE, limit=1
    )

    assert [item.request_id for item in claimed] == [fault.request_id]


def test_postgres_routine_aging_prevents_starvation(store) -> None:
    observed = datetime.now(timezone.utc)
    routine = _telemetry("node-old-routine").model_copy(
        update={
            "created_at": observed - timedelta(seconds=31),
            "updated_at": observed - timedelta(seconds=31),
        }
    )
    evidence = _evidence("node-new-evidence")
    store.enqueue_processor_request(routine)
    store.enqueue_processor_request(evidence)

    claimed = store.claim_active_processor_requests(
        "pod-aging",
        now=observed,
        lease_duration=REQUEST_LEASE,
        limit=1,
        include_paths={"/v1/collector-events/host-telemetry"},
        routine_starvation_seconds=30,
    )

    assert [item.request_id for item in claimed] == [routine.request_id]


def test_postgres_reports_priority_zero_backlog(store) -> None:
    fault = _request("/v1/gpu-events/xid", body=b'{"node_id":"node-fault","xid":79}')
    telemetry = _telemetry("node-telemetry")
    store.enqueue_processor_request(fault)
    store.enqueue_processor_request(telemetry)

    assert store.processor_fault_backlog_depth() == 1


def test_postgres_enqueue_notifies_waiting_consumers(store) -> None:
    stop = Event()
    ready = Event()
    notified = Event()
    states = []
    payloads = []

    def on_state(enabled: bool, shard: int | None) -> None:
        states.append((enabled, shard))
        if enabled:
            ready.set()

    def on_notification(payload: str) -> None:
        payloads.append(payload)
        notified.set()

    thread = Thread(
        target=store.listen_processor_queue_notifications,
        args=(stop, "spool-worker-notify-test", 1, on_notification, on_state),
    )
    thread.start()
    try:
        assert ready.wait(timeout=5), "expected ready.wait(timeout=5) to be truthy"
        store.enqueue_processor_request(
            _request("/v1/gpu-events/xid", body=b'{"node_id":"node-notify","xid":79}')
        )
        assert notified.wait(timeout=5), (
            "expected notified.wait(timeout=5) to be truthy"
        )
    finally:
        stop.set()
        thread.join(timeout=5)

    assert not thread.is_alive(), "expected thread.is_alive() to be falsy"
    assert states[0] == (True, 0)
    notification = json.loads(payloads[0])
    assert "partition" not in notification
    assert notification["priority"] == 0
    assert notification["path"] == "/v1/gpu-events/xid"
    assert notification["request_id"]


def test_postgres_spool_admission_notifies_dedicated_channel(store) -> None:
    stop = Event()
    ready = Event()
    notified = Event()
    states = []
    payloads = []

    def on_state(enabled: bool) -> None:
        states.append(enabled)
        if enabled:
            ready.set()

    def on_notification(payload: str) -> None:
        payloads.append(payload)
        notified.set()

    thread = Thread(
        target=store.listen_telemetry_spool_notifications,
        args=(stop, on_notification, on_state),
    )
    thread.start()
    try:
        assert ready.wait(timeout=5), "expected ready.wait(timeout=5) to be truthy"
        request = _telemetry("node-spool-notify")
        result = store.try_spool_telemetry_requests(
            [request], max_depth=100, max_cluster_depth=100
        )
        assert result[0][0] is not None
        assert notified.wait(timeout=5), (
            "expected notified.wait(timeout=5) to be truthy"
        )
    finally:
        stop.set()
        thread.join(timeout=5)

    assert not thread.is_alive(), "expected thread.is_alive() to be falsy"
    assert states[0] is True
    notification = json.loads(payloads[0])
    assert notification["path"] == request.path


def test_postgres_notification_shard_has_one_owner(store) -> None:
    stops = [Event(), Event()]
    ready = [Event(), Event()]
    notified = [Event(), Event()]
    shards = [None, None]

    def state(index):
        def update(enabled: bool, shard: int | None) -> None:
            if enabled:
                shards[index] = shard
                ready[index].set()

        return update

    threads = [
        Thread(
            target=store.listen_processor_queue_notifications,
            args=(
                stops[index],
                f"notification-owner-{index}",
                1,
                lambda _payload, index=index: notified[index].set(),
                state(index),
            ),
        )
        for index in range(2)
    ]
    for thread in threads:
        thread.start()
    try:
        assert all(event.wait(timeout=5) for event in ready), (
            "expected all(event.wait(timeout=5) for event in ready) to be truthy"
        )
        assert sorted(shards, key=lambda item: item is None) == [0, None]
        store.enqueue_processor_request(
            _request("/v1/gpu-events/xid", body=b'{"node_id":"node-shard","xid":79}')
        )
        assert any(event.wait(timeout=5) for event in notified), (
            "expected any(event.wait(timeout=5) for event in notified) to be truthy"
        )
        assert sum(event.is_set() for event in notified) == 1
    finally:
        for stop in stops:
            stop.set()
        for thread in threads:
            thread.join(timeout=5)

    assert all(not thread.is_alive() for thread in threads), (
        "expected all(not thread.is_alive() for thread in threads) to be truthy"
    )


def test_claim_keeps_one_request_per_lane(store) -> None:
    first = _telemetry("node-a")
    second = _telemetry("node-a")
    assert first.ordering_key() == second.ordering_key()
    store.enqueue_processor_request(first)
    store.enqueue_processor_request(second)

    claimed = store.claim_active_processor_requests(
        "pod-a", now=datetime.now(timezone.utc), lease_duration=REQUEST_LEASE, limit=8
    )

    assert [item.request_id for item in claimed] == [first.request_id]
    assert store.validate_processor_lane(
        first.ordering_key(), "pod-a", claimed[0].leader_epoch, claimed[0].lease_token
    ), (
        'expected store.validate_processor_lane( first.ordering_key(), "pod-a", claimed[0].leader_epoch, claimed[0].lease_token, ) to be truthy'
    )
    # The lane is leased, so the sibling stays queued even though a slot
    # is free.
    assert (
        store.claim_active_processor_requests(
            "pod-b",
            now=datetime.now(timezone.utc),
            lease_duration=REQUEST_LEASE,
            limit=8,
        )
        == []
    )


def test_path_filters_still_scope_the_window(store) -> None:
    telemetry = _telemetry("node-a")
    fault = _request("/v1/gpu-events/sxid", body=b'{"node_id":"node-b","sxid":13}')
    store.enqueue_processor_request(telemetry)
    store.enqueue_processor_request(fault)
    now = datetime.now(timezone.utc)

    included = store.claim_active_processor_requests(
        "pod-a",
        now=now,
        lease_duration=REQUEST_LEASE,
        limit=8,
        include_paths={"/v1/collector-events/host-telemetry"},
    )
    assert [item.request_id for item in included] == [telemetry.request_id]

    excluded = store.claim_active_processor_requests(
        "pod-b",
        now=now,
        lease_duration=REQUEST_LEASE,
        limit=8,
        exclude_paths={"/v1/collector-events/host-telemetry"},
    )
    assert [item.request_id for item in excluded] == [fault.request_id]


def test_fault_waits_for_the_correlated_observation(store) -> None:
    """The interlock moved out of the innermost scan; it must still hold.

    The observation shares a correlation scope with the fault, so the
    fault has to stay queued until the observation completes, and the
    observation has to be promoted ahead of the telemetry backlog it
    would otherwise sort behind.
    """
    for index in range(20):
        store.enqueue_processor_request(_telemetry(f"node-{index:03d}"))
    fault = _request("/v1/collector-events/nvidia-kernel", body=b'{"node_id":"node-a"}')
    observation = _request(
        "/v1/workload-observations",
        body=(
            b'{"job_id":"training-job",'
            b'"attempt_id":"training-job-a001",'
            b'"containers":[{"node_id":"node-a"}]}'
        ),
    )
    assert set(fault.correlation_scope_keys).intersection(
        observation.correlation_scope_keys
    ), (
        "expected set(fault.correlation_scope_keys).intersection( observation.correlation_scope_keys ) to be truthy"
    )
    store.enqueue_processor_request(fault)
    store.enqueue_processor_request(observation)
    now = datetime.now(timezone.utc)

    claimed = store.claim_active_processor_requests(
        "pod-a", now=now, lease_duration=REQUEST_LEASE, limit=2
    )
    claimed_ids = [item.request_id for item in claimed]
    # Promoted ahead of the telemetry backlog, and the fault it shadows
    # stays queued. The second slot goes to telemetry, which is fine.
    assert claimed_ids[0] == observation.request_id
    assert fault.request_id not in claimed_ids
    lease = claimed[0]

    store.complete_active_processor_request(
        observation.request_id,
        "pod-a",
        lease.leader_epoch,
        lease.lease_token,
        response_status=200,
        response_content_type="application/json",
        response_body_base64="e30=",
    )
    after = store.claim_active_processor_requests(
        "pod-b", now=now + timedelta(seconds=1), lease_duration=REQUEST_LEASE, limit=1
    )
    assert [item.request_id for item in after] == [fault.request_id]


def test_expired_lease_is_reclaimed(store) -> None:
    item = _telemetry("node-a")
    store.enqueue_processor_request(item)
    start = datetime.now(timezone.utc)
    first = store.claim_active_processor_requests(
        "pod-a", now=start, lease_duration=timedelta(seconds=1), limit=4
    )
    assert len(first) == 1

    later = start + timedelta(seconds=30)
    second = store.claim_active_processor_requests(
        "pod-b", now=later, lease_duration=REQUEST_LEASE, limit=4
    )

    assert [entry.request_id for entry in second] == [item.request_id]
    assert second[0].leader_epoch > first[0].leader_epoch
    assert not store.validate_processor_lane(
        item.ordering_key(), "pod-a", first[0].leader_epoch, first[0].lease_token
    ), (
        'expected store.validate_processor_lane( item.ordering_key(), "pod-a", first[0].leader_epoch, first[0].lease_token, ) to be falsy'
    )


def test_lane_cleanup_skips_lanes_that_still_hold_work(store) -> None:
    busy = _telemetry("node-busy")
    idle = _telemetry("node-idle")
    store.enqueue_processor_request(busy)
    store.enqueue_processor_request(idle)
    start = datetime.now(timezone.utc)
    claimed = store.claim_active_processor_requests(
        "pod-a", now=start, lease_duration=timedelta(seconds=1), limit=8
    )
    assert len(claimed) == 2
    for entry in claimed:
        if entry.request_id != idle.request_id:
            continue
        store.complete_active_processor_request(
            entry.request_id,
            "pod-a",
            entry.leader_epoch,
            entry.lease_token,
            response_status=200,
            response_content_type="application/json",
            response_body_base64="e30=",
        )

    deleted = store.cleanup_processor_lanes(
        older_than=start + timedelta(minutes=5), limit=100
    )

    assert deleted == 1
    assert (
        store.validate_processor_lane(idle.ordering_key(), "pod-a", 1, "gone") is False
    )
    remaining = store.claim_active_processor_requests(
        "pod-b",
        now=start + timedelta(seconds=30),
        lease_duration=REQUEST_LEASE,
        limit=8,
    )
    # The busy lane's row was left in place, so its request is still
    # claimable and its epoch still advances.
    assert [entry.request_id for entry in remaining] == [busy.request_id]
    assert remaining[0].leader_epoch == 2


def test_completed_request_is_not_reclaimed(store) -> None:
    item = _telemetry("node-a")
    store.enqueue_processor_request(item)
    now = datetime.now(timezone.utc)
    claimed = store.claim_active_processor_requests(
        "pod-a", now=now, lease_duration=REQUEST_LEASE, limit=4
    )
    store.complete_active_processor_request(
        item.request_id,
        "pod-a",
        claimed[0].leader_epoch,
        claimed[0].lease_token,
        response_status=200,
        response_content_type="application/json",
        response_body_base64="e30=",
    )

    assert (
        store.claim_active_processor_requests(
            "pod-b",
            now=now + timedelta(seconds=5),
            lease_duration=REQUEST_LEASE,
            limit=4,
        )
        == []
    )
    stored = store.get_processor_request(item.request_id)
    assert stored.status is ProcessorRequestStatus.COMPLETED


def test_queue_counters_track_bulk_statements(store) -> None:
    """The counters moved from a row trigger to a statement trigger.

    ``processor_queue_count_status`` recomputes the truth from the queue
    itself, so it is the exact oracle for the trigger: every bulk
    statement - batch admission, a multi-row claim, completion, and the
    cleanup delete - has to leave it with no mismatched cluster.
    """
    batch = [_telemetry(f"node-{index:03d}") for index in range(40)]
    batch.extend(
        _telemetry(f"node-{index:03d}", cluster_id="cluster-b") for index in range(10)
    )
    store.try_enqueue_processor_requests_batch(
        batch, max_depth=10_000, max_cluster_depth=10_000
    )
    status = store.processor_queue_count_status()
    assert status["expected_total"] == 50
    assert status["counter_total"] == 50
    assert status["mismatched_clusters"] == 0

    now = datetime.now(timezone.utc)
    claimed = store.claim_active_processor_requests(
        "pod-a", now=now, lease_duration=REQUEST_LEASE, limit=20
    )
    assert len(claimed) == 20
    # LEASED still counts as incomplete, so nothing moved.
    assert store.processor_queue_count_status() == {
        "expected_total": 50,
        "counter_total": 50,
        "mismatched_clusters": 0,
        "ready": True,
    }

    for entry in claimed:
        store.complete_active_processor_request(
            entry.request_id,
            "pod-a",
            entry.leader_epoch,
            entry.lease_token,
            response_status=200,
            response_content_type="application/json",
            response_body_base64="e30=",
        )
    status = store.processor_queue_count_status()
    assert status["expected_total"] == 30
    assert status["counter_total"] == 30
    assert status["mismatched_clusters"] == 0

    deleted = store.cleanup_completed_processor_requests(
        older_than=now + timedelta(minutes=5), limit=1000
    )
    assert deleted == 20
    status = store.processor_queue_count_status()
    assert status["expected_total"] == 30
    assert status["counter_total"] == 30
    assert status["ready"] is True


def test_partitioned_counters_separate_fault_and_low_priority(store) -> None:
    status = store.finalize_processor_counter_shards()
    assert status["mode"] == "partitioned"
    fault = _fault("node-fault")
    evidence = _evidence("node-evidence")
    routine = _telemetry("node-routine")

    for request in (fault, evidence, routine):
        accepted, reason = store.try_enqueue_processor_request(
            request, max_depth=100, max_cluster_depth=100
        )
        assert accepted is not None
        assert reason is None

    with store._db.cursor() as cursor:
        cursor.execute(
            """
            SELECT coalesce(sum(incomplete_count), 0)
            FROM gpu_fault_processor_queue_counts
            WHERE cluster_id='cluster-a'
            """
        )
        legacy_count = cursor.fetchone()[0]
        cursor.execute(
            """
            SELECT priority_bucket, sum(incomplete_count)
            FROM gpu_fault_processor_priority_count_shards
            WHERE cluster_id='cluster-a'
            GROUP BY priority_bucket
            ORDER BY priority_bucket
            """
        )
        priority_counts = dict(cursor.fetchall())

    assert legacy_count == 0
    assert priority_counts == {0: 1, 50: 1, 100: 1}
    assert store.processor_queue_stats()["depth"] == 3
    assert store.processor_queue_count_status()["ready"] is True

    restored = store.restore_legacy_processor_counters()
    assert restored["mode"] == "dual"
    assert restored["counter_total"] == 3


def test_partitioned_fault_admission_ignores_locked_evidence_shard(store) -> None:
    import psycopg

    store.finalize_processor_counter_shards()
    store.try_enqueue_processor_request(
        _evidence("node-low-seed"), max_depth=100, max_cluster_depth=100
    )
    other = PostgresStore(POSTGRES_URL)
    blocker = psycopg.connect(POSTGRES_URL, autocommit=False)
    shard_id = blocker.execute(
        """
        SELECT shard_id
        FROM gpu_fault_processor_priority_count_shards
        WHERE cluster_id='cluster-a'
          AND priority_bucket=50
        LIMIT 1
        """
    ).fetchone()[0]
    blocker.execute(
        """
        SELECT incomplete_count
        FROM gpu_fault_processor_priority_count_shards
        WHERE cluster_id='cluster-a'
          AND priority_bucket=50
          AND shard_id=%s
        FOR UPDATE
        """,
        (shard_id,),
    ).fetchone()
    started = time.monotonic()
    try:
        accepted, reason = other.try_enqueue_processor_request(
            _fault("node-fault-not-blocked"), max_depth=100, max_cluster_depth=100
        )
        elapsed = time.monotonic() - started
    finally:
        blocker.rollback()
        blocker.close()
        other.close()

    assert accepted is not None
    assert reason is None
    assert elapsed < 2


def test_partitioned_evidence_admission_ignores_fault_shard_lock(store) -> None:
    import psycopg

    store.finalize_processor_counter_shards()
    fault = _fault("node-fault-seed")
    store.try_enqueue_processor_request(fault, max_depth=100, max_cluster_depth=100)
    with store._db.cursor() as cursor:
        cursor.execute(
            """
            SELECT shard_id
            FROM gpu_fault_processor_priority_count_shards
            WHERE cluster_id='cluster-a'
              AND priority_bucket=0
              AND incomplete_count > 0
            LIMIT 1
            """
        )
        shard_id = cursor.fetchone()[0]
    other = PostgresStore(POSTGRES_URL)
    blocker = psycopg.connect(POSTGRES_URL, autocommit=False)
    blocker.execute(
        """
        SELECT incomplete_count
        FROM gpu_fault_processor_priority_count_shards
        WHERE cluster_id='cluster-a'
          AND priority_bucket=0
          AND shard_id=%s
        FOR UPDATE
        """,
        (shard_id,),
    ).fetchone()
    started = time.monotonic()
    try:
        accepted, reason = other.try_enqueue_processor_request(
            _evidence("node-evidence-not-blocked"), max_depth=100, max_cluster_depth=100
        )
        elapsed = time.monotonic() - started
    finally:
        blocker.rollback()
        blocker.close()
        other.close()

    assert accepted is not None
    assert reason is None
    assert elapsed < 2


def test_partitioned_evidence_and_routine_shards_do_not_block(store) -> None:
    import psycopg

    store.finalize_processor_counter_shards()
    store.try_enqueue_processor_request(
        _telemetry("node-routine-seed"), max_depth=100, max_cluster_depth=100
    )
    other = PostgresStore(POSTGRES_URL)
    blocker = psycopg.connect(POSTGRES_URL, autocommit=False)
    shard_id = blocker.execute(
        """
        SELECT shard_id
        FROM gpu_fault_processor_priority_count_shards
        WHERE cluster_id='cluster-a'
          AND priority_bucket=100
        LIMIT 1
        """
    ).fetchone()[0]
    blocker.execute(
        """
        SELECT incomplete_count
        FROM gpu_fault_processor_priority_count_shards
        WHERE cluster_id='cluster-a'
          AND priority_bucket=100
          AND shard_id=%s
        FOR UPDATE
        """,
        (shard_id,),
    ).fetchone()
    started = time.monotonic()
    try:
        accepted, reason = other.try_enqueue_processor_request(
            _evidence("node-evidence-not-blocked-by-routine"),
            max_depth=100,
            max_cluster_depth=100,
        )
        elapsed = time.monotonic() - started
    finally:
        blocker.rollback()
        blocker.close()
        other.close()

    assert accepted is not None
    assert reason is None
    assert elapsed < 2

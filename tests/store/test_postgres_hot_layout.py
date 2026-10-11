"""The v23 hot-table layout against the test database.

Both layouts are exercised through the real statements: the inline layout an
upgraded database keeps until the switch, and the detached layout a fresh
database starts in. What the perf evidence asked for is checked directly from
the catalog -- claims, completions and coalescing rewrites are HOT updates, a
completion leaves the queue, bodies live in the payloads table, a replay
answers a retry with the same response, replay buckets are created ahead and
dropped whole, and a cluster's metric partition comes and goes with the
cluster -- plus the switch itself, which copies the latest rows and flips the
mode once.
"""

from __future__ import annotations

import base64
import json
import os
import time
from datetime import datetime, timedelta, timezone

import pytest

from gpu_fault.regional import RegionalClusterRegistration
from gpu_fault.store import PostgresStore
from gpu_fault.store.postgres import hot_layout
from tests._builders import processor_request
from tests.store._postgres_processor_claim_support import (
    REQUEST_LEASE,
    _ensure_postgres_schema,
    _truncate,
)

POSTGRES_URL = os.getenv("GPU_FAULT_TEST_POSTGRES_URL")
pytestmark = pytest.mark.skipif(
    not POSTGRES_URL, reason="requires GPU_FAULT_TEST_POSTGRES_URL"
)
HOST_PATH = "/v1/collector-events/host-telemetry"
FAULT_PATH = "/v1/gpu-events/xid"
RESPONSE = base64.b64encode(b'{"ok":true}').decode()


def _connect():
    import psycopg

    assert POSTGRES_URL is not None
    return psycopg.connect(POSTGRES_URL, autocommit=True)


def _set_mode(mode: str) -> None:
    # The shared harness truncates every table but the schema registry, the
    # layout row included; the product upsert puts it back.
    with _connect() as connection:
        with connection.cursor() as cursor:
            hot_layout.write_layout_mode(cursor, mode)
            if mode == "detached":
                # What the switch and the cleanup job do: buckets ahead of the
                # clock, so a completion lands in its bucket, not the default.
                hot_layout.ensure_replay_partitions(cursor, datetime.now(timezone.utc))


@pytest.fixture(params=["inline", "detached"])
def store(request):
    _truncate()
    _ensure_postgres_schema()
    _set_mode(request.param)
    instance = PostgresStore(POSTGRES_URL, initialize_schema=False)
    try:
        yield instance
    finally:
        instance.close()
        _set_mode("inline")
        _truncate()


def _telemetry(node: str, value: float = 1.0, *, request_id: str | None = None):
    item = processor_request(
        HOST_PATH,
        body=json.dumps({"node_id": node, "cpu": [value] * 300}).encode(),
        cluster_id="layout-cluster",
    )
    return (
        item
        if request_id is None
        else item.model_copy(update={"request_id": request_id})
    )


def _small(node: str, value: float = 1.0):
    return processor_request(
        HOST_PATH,
        body=json.dumps({"node_id": node, "v": value}).encode(),
        cluster_id="layout-cluster",
    )


def _fault(node: str):
    return processor_request(
        FAULT_PATH,
        body=json.dumps(
            {"node_id": node, "xid": 79, "job_id": "job-1", "attempt_id": "attempt-1"}
        ).encode(),
        cluster_id="layout-cluster",
    )


def _hot_counters(connection, table: str) -> tuple[int, int]:
    return connection.execute(
        "SELECT n_tup_upd, n_tup_hot_upd FROM pg_stat_user_tables WHERE relname=%s",
        (table,),
    ).fetchone()


def _complete(store, claimed):
    return store.complete_active_processor_requests_batch(
        [
            {
                "request_id": item.request_id,
                "owner_id": "owner",
                "lane_epoch": item.leader_epoch,
                "lease_token": item.lease_token,
                "response_status": 202,
                "response_content_type": "application/json",
                "response_body_base64": RESPONSE,
            }
            for item in claimed
        ]
    )


def test_a_request_round_trips_in_both_layouts_with_its_body(store) -> None:
    requests = [_telemetry(f"node-{n}", request_id=f"rt-{n}") for n in range(3)]
    results = store.try_enqueue_processor_requests_batch(
        requests + [_fault("node-9")], max_depth=100, max_cluster_depth=100
    )
    assert all(item is not None and reason is None for item, reason in results), (
        f"every request must be admitted as a new row: {results}"
    )
    stored = store.get_processor_request("rt-0")
    assert stored.body_base64 == requests[0].body_base64
    assert stored.path == HOST_PATH
    assert stored.correlation_scope_keys == requests[0].correlation_scope_keys
    now = datetime.now(timezone.utc)
    claimed = store.claim_active_processor_requests(
        "owner", now=now, lease_duration=REQUEST_LEASE, limit=10
    )
    assert [item.path for item in claimed][0] == FAULT_PATH, "faults still lead"
    assert all(item.body_base64 for item in claimed), "a claim returns the body"
    with _connect() as connection:
        rows = connection.execute(
            "SELECT path, cardinality(scope_keys), payload ? 'body_base64' "
            "FROM gpu_fault_processor_queue ORDER BY request_id"
        ).fetchall()
    detached = store.hot_layout_mode() == "detached"
    assert {row[0] for row in rows} == {HOST_PATH, FAULT_PATH}
    assert all((row[2] is False) == detached for row in rows), (
        "the detached layout keeps the body out of the queue row"
    )
    assert [row[1] for row in rows if row[0] == FAULT_PATH] == [2], (
        "a correlated fault carries its job and node scope keys as a column"
    )
    assert {row[1] for row in rows if row[0] == HOST_PATH} == {1}, (
        "an incident-scoped telemetry row carries its node scope key"
    )
    completed = _complete(store, claimed)
    assert all(item.status.value == "COMPLETED" for item in completed), completed
    replay = store.get_processor_request("rt-1")
    assert replay.status.value == "COMPLETED"
    assert replay.response_body_base64 == RESPONSE
    assert replay.body_base64 == requests[1].body_base64
    retry = store.try_enqueue_processor_request(
        requests[1], max_depth=100, max_cluster_depth=100
    )
    assert retry == (replay, None), (
        "a retry of a completed request replays its response"
    )
    assert store.processor_queue_stats()["depth"] == 0


def test_claims_completions_and_coalescing_rewrites_are_hot_updates(store) -> None:
    """HOT needs page room as well as unindexed columns: three versions of
    three small rows fit one page at fillfactor 70, so every update here must
    be HOT or an index is being maintained."""

    store.try_enqueue_processor_requests_batch(
        [_small(f"node-{n}") for n in range(3)], max_depth=100, max_cluster_depth=100
    )
    with _connect() as connection:
        connection.execute(
            "SELECT pg_stat_reset_single_table_counters('gpu_fault_processor_queue'::regclass)"
        )
    # Coalesce: a newer sample for every lane rewrites the pending rows.
    store.try_enqueue_processor_requests_batch(
        [_small(f"node-{n}", value=2.0) for n in range(3)],
        max_depth=100,
        max_cluster_depth=100,
    )
    claimed = store.claim_active_processor_requests(
        "owner", now=datetime.now(timezone.utc), lease_duration=REQUEST_LEASE, limit=8
    )
    assert len(claimed) == 3
    assert store.renew_active_processor_request(
        claimed[0].request_id,
        "owner",
        claimed[0].leader_epoch,
        claimed[0].lease_token,
        lease_duration=REQUEST_LEASE,
    ), "the owner renews its own lease"
    _complete(store, claimed)
    store.close()  # the pool's backends flush their statistics when they exit
    with _connect() as connection:
        updates, hot = _hot_counters(connection, "gpu_fault_processor_queue")
    assert updates >= 7, updates
    assert hot == updates, (
        f"a queue update maintained an index instead of HOT ({hot}/{updates})"
    )


def test_the_detached_layout_moves_completions_to_a_replay_bucket(store) -> None:
    if store.hot_layout_mode() != "detached":
        pytest.skip("inline keeps completed rows in the queue")
    requests = [_telemetry(f"node-{n}", request_id=f"mv-{n}") for n in range(4)]
    store.try_enqueue_processor_requests_batch(
        requests, max_depth=100, max_cluster_depth=100
    )
    with _connect() as connection:
        assert connection.execute(
            "SELECT count(*) FROM gpu_fault_processor_payloads"
        ).fetchone() == (4,), "bodies live in the payloads table"
    claimed = store.claim_active_processor_requests(
        "owner", now=datetime.now(timezone.utc), lease_duration=REQUEST_LEASE, limit=4
    )
    _complete(store, claimed)
    with _connect() as connection:
        assert connection.execute(
            "SELECT count(*) FROM gpu_fault_processor_queue"
        ).fetchone() == (0,), "a completion leaves the queue"
        assert connection.execute(
            "SELECT count(*) FROM gpu_fault_processor_payloads"
        ).fetchone() == (0,), "its body row goes with it"
        rows = connection.execute(
            "SELECT tableoid::regclass::text, response_status FROM gpu_fault_processor_replays"
        ).fetchall()
    assert len(rows) == 4 and all(status == 202 for _, status in rows)
    assert all(
        hot_layout.replay_bucket_bounds(table) is not None for table, _ in rows
    ), f"completions land in a time bucket, not the DEFAULT partition: {rows}"
    # Retention drops the bucket whole once it is past the cutoff.
    removed = store.cleanup_completed_processor_requests(
        older_than=datetime.now(timezone.utc) + timedelta(seconds=600), limit=100
    )
    assert removed == 4
    with _connect() as connection:
        assert connection.execute(
            "SELECT count(*) FROM gpu_fault_processor_replays"
        ).fetchone() == (0,)
        names = [
            row[0]
            for row in connection.execute(
                "SELECT c.relname FROM pg_inherits i JOIN pg_class c ON c.oid=i.inhrelid "
                "WHERE i.inhparent='gpu_fault_processor_replays'::regclass"
            ).fetchall()
        ]
    assert "gpu_fault_processor_replays_default" in names
    assert all(
        hot_layout.replay_bucket_bounds(name) is None
        or hot_layout.replay_bucket_bounds(name)[1]
        > int(time.time()) + 600  # only buckets ahead of the cutoff survive
        for name in names
    ), names
    with pytest.raises(Exception, match="mv-0"):
        store.get_processor_request("mv-0")


def test_inline_rows_are_still_swept_by_row_after_the_switch(store) -> None:
    if store.hot_layout_mode() != "inline":
        pytest.skip("starts inline")
    requests = [_telemetry(f"node-{n}", request_id=f"sw-{n}") for n in range(2)]
    store.try_enqueue_processor_requests_batch(
        requests, max_depth=100, max_cluster_depth=100
    )
    claimed = store.claim_active_processor_requests(
        "owner", now=datetime.now(timezone.utc), lease_duration=REQUEST_LEASE, limit=2
    )
    _complete(store, claimed)
    with _connect() as connection:
        result = hot_layout.switch_to_detached(connection)
    assert result["changed"] is True and result["mode"] == "detached"
    with _connect() as connection:
        assert hot_layout.switch_to_detached(connection)["changed"] is False, (
            "one-way, idempotent"
        )
    time.sleep(0.3)
    assert store.hot_layout_mode() == "detached"
    # Inline completions written before the switch stay readable and are swept.
    assert store.get_processor_request("sw-0").status.value == "COMPLETED"
    removed = store.cleanup_completed_processor_requests(
        older_than=datetime.now(timezone.utc) + timedelta(seconds=1), limit=100
    )
    assert removed >= 2
    with pytest.raises(Exception, match="sw-0"):
        store.get_processor_request("sw-0")
    # New admissions after the switch use the detached layout.
    store.try_enqueue_processor_requests_batch(
        [_telemetry("node-after", request_id="sw-after")],
        max_depth=100,
        max_cluster_depth=100,
    )
    with _connect() as connection:
        assert connection.execute(
            "SELECT payload ? 'body_base64' FROM gpu_fault_processor_queue WHERE request_id='sw-after'"
        ).fetchone() == (False,)
        assert connection.execute(
            "SELECT count(*) FROM gpu_fault_processor_payloads WHERE request_id='sw-after'"
        ).fetchone() == (1,)


def test_metric_latest_follows_the_layout_and_cluster_partitions_come_and_go(
    store,
) -> None:
    detached = store.hot_layout_mode() == "detached"
    registration = RegionalClusterRegistration(
        cluster_id="layout-cluster",
        region="us-west-2",
        hyperpod_cluster_name="layout-cluster",
        eks_cluster_arn="arn:aws:eks:us-west-2:000000000000:cluster/layout-cluster",
        token_sha256="0" * 64,
        agent_endpoint_allowed_cidrs=["10.0.0.0/8"],
    )
    store.save_regional_cluster(registration)
    partition = hot_layout.cluster_partition_name("layout-cluster")
    with _connect() as connection:
        assert connection.execute(
            "SELECT to_regclass(%s)", (partition,)
        ).fetchone() != (None,), "registration creates the cluster's metric partition"
    from gpu_fault.gpu_metrics import GpuMetricLatest, GpuMetricSample, GpuMetricSource

    key = ("layout-cluster", "node-1", "gpu_temperature_c", "GPU-0")
    observed = datetime.now(timezone.utc)
    latest = GpuMetricLatest(
        cluster_id="layout-cluster",
        node_id="node-1",
        observed_at=observed,
        source=GpuMetricSource.DCGM_EXPORTER,
        sample=GpuMetricSample(
            metric_name="gpu_temperature_c",
            canonical_name="gpu_temperature_c",
            value=61.0,
            gpu_index="0",
            gpu_uuid="GPU-0",
        ),
    )
    store.observe_gpu_metrics([(key, latest)])
    assert [
        item.sample.value
        for item in store.list_gpu_metrics_latest("layout-cluster", "node-1")
    ] == [61.0]
    with _connect() as connection:
        flat = connection.execute(
            "SELECT count(*) FROM gpu_fault_gpu_metric_latest"
        ).fetchone()[0]
        twin = connection.execute(
            "SELECT count(*), min(tableoid::regclass::text) FROM gpu_fault_gpu_metric_latest_clusters"
        ).fetchone()
    assert (flat, twin[0]) == ((0, 1) if detached else (1, 0)), (flat, twin)
    if detached:
        assert twin[1] == partition, "the row lives in the cluster's partition"
    store.delete_regional_cluster("layout-cluster")
    with _connect() as connection:
        assert connection.execute(
            "SELECT to_regclass(%s)", (partition,)
        ).fetchone() == (None,), "removal drops the partition and its pages with it"
        assert connection.execute(
            "SELECT count(*) FROM gpu_fault_gpu_metric_latest_clusters"
        ).fetchone() == (0,)


def test_a_cluster_whose_rows_reached_the_default_partition_still_gets_its_own(
    store,
) -> None:
    with _connect() as connection:
        connection.execute(
            "INSERT INTO gpu_fault_gpu_metric_latest_clusters "
            "(key, cluster_id, node_id, observed_at, payload) VALUES "
            "('k1', 'late-cluster', 'n', now(), '{}'::jsonb)"
        )
        assert connection.execute(
            "SELECT tableoid::regclass::text FROM gpu_fault_gpu_metric_latest_clusters"
        ).fetchone() == ("gpu_fault_gpu_metric_latest_clusters_default",)
        with connection.transaction():
            with connection.cursor() as cursor:
                created = hot_layout.ensure_cluster_partition(cursor, "late-cluster")
        assert created == hot_layout.cluster_partition_name("late-cluster")
        assert connection.execute(
            "SELECT tableoid::regclass::text FROM gpu_fault_gpu_metric_latest_clusters"
        ).fetchone() == (created,), "the stray row moved into the new partition"
        with connection.transaction():
            with connection.cursor() as cursor:
                assert (
                    hot_layout.ensure_cluster_partition(cursor, "late-cluster") is None
                )
                assert (
                    hot_layout.drop_cluster_partition(cursor, "late-cluster") == created
                )


def test_the_layout_status_reports_mode_and_partitions(store) -> None:
    with _connect() as connection:
        with connection.cursor() as cursor:
            status = hot_layout.layout_status(cursor)
    assert status["schema_version"] == 23
    assert status["mode"] == store.hot_layout_mode()
    assert {item["name"] for item in status["replay_partitions"]} >= {
        "gpu_fault_processor_replays_default"
    }
    assert set(status["gpu_fault_processor_queue"]) == {
        "heap_pages",
        "live_rows",
        "index_bytes",
        "toast_bytes",
    }

"""Counter-mode switches leave every reader a correct depth, and both admission
paths serialise on the same lock.

FINAL-建议汇总 F-D10 (P2-75I, P1-75C). Switching ``dual -> partitioned`` truncated
the legacy counter table and replayed it only ``WHERE mode='dual'``, so a
process still reading the old mode from its 0.25 s cache saw depth 0 and
admitted without limit. And the single enqueue path held the request's row
lock while the batch path held an advisory lock -- two locks that never met.
"""

from __future__ import annotations

import os
import time
from threading import Thread

import pytest

from tests._builders import processor_request
from tests.store._postgres_processor_claim_support import (
    _truncate,
    postgres_store_instance,
)

pytestmark = pytest.mark.skipif(
    not os.getenv("GPU_FAULT_TEST_POSTGRES_URL"),
    reason="GPU_FAULT_TEST_POSTGRES_URL is required",
)


@pytest.fixture
def store():
    for postgres in postgres_store_instance():
        try:
            yield postgres
        finally:
            postgres.restore_legacy_processor_counters()
    _truncate()


def _legacy_counts() -> dict[str, int]:
    import psycopg

    with psycopg.connect(
        os.environ["GPU_FAULT_TEST_POSTGRES_URL"], autocommit=True
    ) as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                "SELECT cluster_id, incomplete_count FROM gpu_fault_processor_queue_counts"
            )
            return {row[0]: int(row[1]) for row in cursor.fetchall()}


def test_partitioned_switch_fills_the_legacy_counter_table(store) -> None:
    for node in ("a1", "a2"):
        store.enqueue_processor_request(
            processor_request(
                "/v1/gpu-events/xid",
                body=f'{{"node_id":"node-{node}","xid":79}}'.encode(),
                cluster_id="cluster-a",
            )
        )
    store.enqueue_processor_request(
        processor_request(
            "/v1/gpu-events/xid",
            body=b'{"node_id":"node-b1","xid":79}',
            cluster_id="cluster-b",
        )
    )

    # ``finalize_processor_counter_shards`` insists on an empty queue; the
    # cache-window hazard is about the depth readers see right after the
    # commit, so drive the switch with the queue populated.
    result = store._switch_processor_counter_mode("partitioned", require_empty=False)

    assert result["mode"] == "partitioned"
    assert _legacy_counts() == {"cluster-a": 2, "cluster-b": 1}


def test_batch_admission_waits_for_the_single_path_row_lock(store) -> None:
    import psycopg

    request = processor_request(
        "/v1/gpu-events/xid", body=b'{"node_id":"node-lock","xid":79}'
    )
    store.enqueue_processor_request(request)
    holder = psycopg.connect(os.environ["GPU_FAULT_TEST_POSTGRES_URL"])
    holder.autocommit = False
    finished: dict[str, object] = {}
    try:
        with holder.cursor() as cursor:
            cursor.execute(
                "SELECT 1 FROM gpu_fault_processor_queue WHERE request_id=%s FOR UPDATE",
                (request.request_id,),
            )

        def enqueue() -> None:
            finished["results"] = store.try_enqueue_processor_requests_batch(
                [request], max_depth=100, max_cluster_depth=100
            )

        batch = Thread(target=enqueue)
        batch.start()
        time.sleep(0.5)
        blocked = batch.is_alive()
        holder.commit()
        batch.join(timeout=10)
    finally:
        holder.close()

    assert blocked, "the batch path must queue behind the single path's row lock"
    assert not batch.is_alive(), (
        "the batch path should proceed once the lock is released"
    )
    results = finished["results"]
    assert isinstance(results, list) and len(results) == 1
    assert results[0][0] is not None and results[0][0].request_id == request.request_id

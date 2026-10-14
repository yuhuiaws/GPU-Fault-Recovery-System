"""One lane, two pending rows, two workers claiming at once: at most one row
of the lane may be claimed, and its completion must then succeed.

Reported live on 2026-09-05 (memory ``processor-gpu-inventory-stale-fencing-
token``): two adjacent gpu-inventory snapshots of one node enqueued 45 ms
apart were claimed by two workers, each claim bumped the lane epoch, and the
first completion failed with "stale processor lane fencing token". The claim
SQL's lane guard is supposed to make that impossible; this test pins it under
real concurrency so the guard cannot regress silently (F-D2).
"""

from __future__ import annotations

import json
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from gpu_fault.channel_registry import (
    GPU_METRICS_PATH,
    HOST_TELEMETRY_PATH,
    NVIDIA_KERNEL_PATH,
    ChannelPool,
    paths_for_pool,
)
from gpu_fault.store import PostgresStore
from gpu_fault.store.postgres.planner_discipline import hot_path_planner_statements
from tests._builders import processor_request
from tests.store._postgres_processor_claim_support import _truncate

POSTGRES_URL = os.getenv("GPU_FAULT_TEST_POSTGRES_URL")
pytestmark = pytest.mark.skipif(
    not POSTGRES_URL, reason="GPU_FAULT_TEST_POSTGRES_URL is not configured"
)
LEASE = timedelta(seconds=120)


@pytest.fixture(autouse=True)
def clean_tables():
    assert POSTGRES_URL is not None
    _truncate()
    PostgresStore(POSTGRES_URL).close()
    yield
    _truncate()


def _store() -> PostgresStore:
    assert POSTGRES_URL is not None
    return PostgresStore(POSTGRES_URL, initialize_schema=False)


def _inventory(sequence: int):
    return processor_request(
        "/v1/collector-events/gpu-inventory",
        body=('{"node_id":"node-a","gpus":[],"seq":%d}' % sequence).encode(),
        cluster_id="cluster-a",
    )


@pytest.mark.parametrize("rounds", [8])
def test_postgres_two_workers_never_both_claim_the_same_lane(rounds: int) -> None:
    first, second = _store(), _store()
    try:
        for round_index in range(rounds):
            _truncate()
            # Two rows of the same node lane: the routine coalescing merges
            # them while both are PENDING, so lease the first one out of the
            # coalescing window before the second arrives, then release it.
            a = first.enqueue_processor_request(_inventory(round_index * 2))
            leased = first.claim_active_processor_requests(
                "warmup", now=datetime.now(timezone.utc), lease_duration=LEASE, limit=1
            )
            assert [item.request_id for item in leased] == [a.request_id]
            b = first.enqueue_processor_request(_inventory(round_index * 2 + 1))
            first.release_active_processor_request(
                a.request_id, "warmup", leased[0].leader_epoch, leased[0].lease_token
            )
            assert first.processor_queue_stats()["depth"] == 2

            def claim(store, owner):
                return store.claim_active_processor_requests(
                    owner, now=datetime.now(timezone.utc), lease_duration=LEASE, limit=4
                )

            with ThreadPoolExecutor(max_workers=2) as pool:
                results = list(
                    pool.map(
                        lambda pair: claim(*pair),
                        [(first, "worker-a"), (second, "worker-b")],
                    )
                )
            claimed = [item for batch in results for item in batch]

            assert len(claimed) <= 1, [(c.request_id, c.lease_owner) for c in claimed]
            assert {b.request_id, a.request_id} >= {c.request_id for c in claimed}
            if claimed:
                item = claimed[0]
                owner_store = first if item.lease_owner == "worker-a" else second
                owner_store.complete_active_processor_request(
                    item.request_id,
                    item.lease_owner,
                    item.leader_epoch,
                    item.lease_token,
                    response_status=200,
                    response_content_type="application/json",
                    response_body_base64="e30=",
                )
    finally:
        first.close()
        second.close()


EVIDENCE = {"edge_filter_reasons": ["threshold"]}
# Every dedicated pool's paths: what the fault stream's claim excludes.
DEDICATED = {
    *paths_for_pool(ChannelPool.GPU),
    *paths_for_pool(ChannelPool.HOST),
    *paths_for_pool(ChannelPool.OBSERVATION),
}
HOLD_SECONDS = 2.0


def _node_row(path: str, node: str = "n1", **payload: Any):
    return processor_request(
        path, body=json.dumps({"node_id": node, **payload}).encode(), cluster_id="c1"
    )


def _complete(store: PostgresStore, item: Any) -> None:
    store.complete_active_processor_request(
        item.request_id,
        item.lease_owner,
        item.leader_epoch,
        item.lease_token,
        response_status=200,
        response_content_type="application/json",
        response_body_base64="e30=",
    )


def _claim(store: PostgresStore, owner: str, **paths: Any) -> list[Any]:
    return store.claim_active_processor_requests(
        owner, now=datetime.now(timezone.utc), lease_duration=LEASE, limit=4, **paths
    )


class _OpenClaim:
    """Another worker's claim statement, its transaction held open -- a slow
    claim that has upserted its lane rows and not yet committed."""

    def __init__(self, store: PostgresStore, **paths: Any) -> None:
        self._sql, self._params = store.claim_active_processor_query(
            "worker-a",
            now=datetime.now(timezone.utc),
            lease_duration=LEASE,
            limit=4,
            **paths,
        )
        self._holding, self._release = threading.Event(), threading.Event()
        self.rows: list[Any] = []
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        import psycopg

        assert POSTGRES_URL is not None
        with psycopg.connect(POSTGRES_URL) as connection:
            with connection.transaction():
                for statement in hot_path_planner_statements():
                    connection.execute(statement)
                self.rows = connection.execute(self._sql, self._params).fetchall()
                self._holding.set()
                self._release.wait(HOLD_SECONDS)

    def __enter__(self) -> _OpenClaim:
        self._thread.start()
        assert self._holding.wait(10), "the open claim ran"
        return self

    def __exit__(self, *_: Any) -> None:
        self._release.set()
        self._thread.join(10)


@pytest.mark.parametrize("lane", ["new", "existing"])
@pytest.mark.parametrize(
    ("held", "claimed", "later"),
    [
        pytest.param(
            HOST_TELEMETRY_PATH, GPU_METRICS_PATH, None, id="evidence-evidence"
        ),
        pytest.param(GPU_METRICS_PATH, None, NVIDIA_KERNEL_PATH, id="evidence-fault"),
    ],
)
def test_a_claim_skips_a_lane_another_claim_is_leasing_instead_of_waiting(
    lane: str, held: str, claimed: str | None, later: str | None
) -> None:
    """Two claims select different rows of one node lane -- the gpu and host
    evidence pools, or the fault stream with a P10 row queued after the
    evidence it shares the lane with. The lane upsert of the second used to
    queue on the first one's uncommitted lane row until it committed and then
    found the lane leased: 1.6 s for nothing against a claim held open 2 s,
    a P10 claim included. It skips the lane at once now, and the lane still
    goes to one lease at a time, in its priority order."""

    store = _store()
    try:
        if lane == "existing":  # an earlier row leased and finished the lane
            store.enqueue_processor_request(
                _node_row(HOST_TELEMETRY_PATH, **EVIDENCE, warm=1)
            )
            (warm,) = _claim(store, "warmup", include_paths={HOST_TELEMETRY_PATH})
            _complete(store, warm)
        held_row = store.enqueue_processor_request(_node_row(held, **EVIDENCE))
        waiting = (
            store.enqueue_processor_request(_node_row(claimed, **EVIDENCE))
            if claimed is not None
            else None
        )
        paths = (
            {"include_paths": {claimed}}
            if claimed is not None
            else {"exclude_paths": DEDICATED}
        )
        with _OpenClaim(store, include_paths={held}) as first:
            assert len(first.rows) == 1 and first.rows[0][0] is not None, (
                "the open claim leased the held row"
            )
            if later is not None:  # the P10 row arrives behind the leased evidence
                waiting = store.enqueue_processor_request(_node_row(later, xid=79))
            started = time.monotonic()
            second = _claim(store, "worker-b", **paths)
            elapsed = time.monotonic() - started
        assert waiting is not None, "a row waits on the lane"
        assert second == [], "the lane is being leased by the other claim"
        assert elapsed < 0.5, (
            f"skipped, not queued behind the open claim: {elapsed:.3f} s"
        )
        holder = store.get_processor_request(held_row.request_id)
        assert holder is not None and holder.lease_owner == "worker-a", (
            "the open claim kept its lease"
        )
        assert _claim(store, "worker-b", **paths) == [], "the lane is leased"
        _complete(store, holder)
        (taken,) = _claim(store, "worker-b", **paths)
        assert taken.request_id == waiting.request_id, "then the lane is free"
    finally:
        store.close()


def test_concurrent_claims_of_every_pool_lease_each_lane_once() -> None:
    """Sixteen workers claim the gpu, host and fault paths of 24 node lanes at
    once and finish what they get: a skipped lane is retried, never leased
    twice, and every row completes exactly once."""

    lanes = [f"n{index}" for index in range(24)]
    seed = _store()
    try:
        expected = set()
        for node in lanes:
            for path, payload in (
                (GPU_METRICS_PATH, EVIDENCE),
                (HOST_TELEMETRY_PATH, EVIDENCE),
                (NVIDIA_KERNEL_PATH, {"xid": 79}),
            ):
                item = seed.enqueue_processor_request(_node_row(path, node, **payload))
                expected.add(item.request_id)
    finally:
        seed.close()
    choices = (
        {"include_paths": {GPU_METRICS_PATH}},
        {"include_paths": {HOST_TELEMETRY_PATH}},
        {"exclude_paths": DEDICATED},
    )
    done: list[str] = []
    overlaps: list[tuple[str, str]] = []
    lock = threading.Lock()
    holders: dict[str, str] = {}

    def worker(index: int) -> None:
        store, chooser = _store(), random.Random(index)
        try:
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                with lock:
                    if len(done) >= len(expected):
                        return
                rows = _claim(store, f"worker-{index}", **chooser.choice(choices))
                for item in rows:
                    with lock:
                        other = holders.get(item.ordering_key())
                        if other is not None:
                            overlaps.append((item.request_id, other))
                        holders[item.ordering_key()] = item.request_id
                    time.sleep(0.002)
                    with lock:
                        holders.pop(item.ordering_key(), None)
                    _complete(store, item)
                    with lock:
                        done.append(item.request_id)
        finally:
            store.close()

    with ThreadPoolExecutor(max_workers=16) as pool:
        list(pool.map(worker, range(16)))

    assert overlaps == [], f"a lane leased twice at once: {overlaps[:3]}"
    assert sorted(done) == sorted(expected), "every row completes exactly once"

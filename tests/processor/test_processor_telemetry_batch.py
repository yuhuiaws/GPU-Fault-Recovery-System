"""Group-commit behaviour of /v1/internal/processor/telemetry-batch.

The batch endpoint is how every replayed telemetry request reaches the
handlers in production, and what limits the 32/50-cluster fleets is
commits per second: the Postgres pool runs autocommit, so a batch that
opened one transaction per item paid one commit per item. These tests
pin the shape that made the batch cheap -- one outer transaction, one
nested savepoint per item, one topology read per cluster -- together
with the isolation that shape must not lose.
"""

from __future__ import annotations

import asyncio
from contextlib import contextmanager
from datetime import datetime, timezone

import httpx

from gpu_fault.app import ApplicationContext, create_app
from gpu_fault.store import InMemoryStore
from tests._builders import asgi_client

TOKEN = "processor-telemetry-batch-token-" + "x" * 32
REPLAY_TOKEN = "test-processor-replay-secret-" + "r" * 32
BATCH_PATH = "/v1/internal/processor/telemetry-batch"
GPU_METRICS_PATH = "/v1/collector-events/gpu-metrics"
HOST_TELEMETRY_PATH = "/v1/collector-events/host-telemetry"
INVENTORY_PATH = "/v1/collector-events/gpu-inventory"


class RecordingStore(InMemoryStore):
    """Records the transaction shape a telemetry batch produces."""

    def __init__(self) -> None:
        super().__init__()
        self.events: list[str] = []
        self.ingestion_keys: list[tuple[str, str, str]] = []
        self.observation_reads: list[str] = []
        # Set to simulate the outer commit failing at COMMIT time, which
        # is the case the per-item retry path exists for.
        self.batch_commit_error: Exception | None = None

    @property
    def batch_transactions(self) -> int:
        return self.events.count("batch-begin")

    @contextmanager
    def processor_batch_transaction(self):
        self.events.append("batch-begin")
        with super().processor_batch_transaction():
            yield
        self.events.append("batch-commit")
        if self.batch_commit_error is not None:
            raise self.batch_commit_error

    @contextmanager
    def collector_ingestion_transaction(
        self, cluster_id: str, node_id: str, batch_id: str
    ):
        self.ingestion_keys.append((cluster_id, node_id, batch_id))
        self.events.append(f"item:{cluster_id}/{node_id}/{batch_id}")
        with super().collector_ingestion_transaction(cluster_id, node_id, batch_id):
            yield

    def list_attempt_observations(self, cluster_id: str):
        self.observation_reads.append(cluster_id)
        return super().list_attempt_observations(cluster_id)


def build_app(monkeypatch, store: RecordingStore):
    # The batch endpoint only exists with a processor, and it
    # authenticates with the processor's internal replay token.
    monkeypatch.setenv("GPU_FAULT_PROCESSOR_MODE", "active-active")
    monkeypatch.setenv("POD_UID", "pod-telemetry-batch")
    context = ApplicationContext(store=store, execution_token=TOKEN)
    return context, create_app(context)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def gpu_metrics_item(
    request_id: str, cluster_id: str, node_id: str, *, samples=None
) -> dict:
    return {
        "request_id": request_id,
        "path": GPU_METRICS_PATH,
        "payload": {
            "batch_id": f"gpu-metrics-{request_id}",
            "cluster_id": cluster_id,
            "node_id": node_id,
            "observed_at": now(),
            "source": "DCGM_EXPORTER",
            "edge_filter_reasons": ["health-summary"],
            "samples": (
                [
                    {
                        "metric_name": "DCGM_FI_DEV_SM_CLOCK",
                        "canonical_name": "sm_clock_mhz",
                        "value": 345.0,
                        "unit": "MHz",
                        "gpu_index": "0",
                        "gpu_uuid": f"GPU-{node_id}-0",
                    }
                ]
                if samples is None
                else samples
            ),
        },
    }


def host_telemetry_item(request_id: str, cluster_id: str, node_id: str) -> dict:
    return {
        "request_id": request_id,
        "path": HOST_TELEMETRY_PATH,
        "payload": {
            "batch_id": f"host-telemetry-{request_id}",
            "cluster_id": cluster_id,
            "node_id": node_id,
            "observed_at": now(),
            "collection_errors": [],
            "edge_filter_reasons": ["health-summary"],
            "samples": [{"name": "cpu_usage_percent", "value": 1.5, "unit": "percent"}],
        },
    }


def inventory_item(request_id: str, cluster_id: str, node_id: str) -> dict:
    return {
        "request_id": request_id,
        "path": INVENTORY_PATH,
        "payload": {
            "snapshot_id": f"gpu-inventory-{request_id}",
            "cluster_id": cluster_id,
            "node_id": node_id,
            "observed_at": now(),
            "source": "NVIDIA_SMI",
            "source_boot_id": "boot-a",
            "devices": [
                {
                    "gpu_index": 0,
                    "gpu_uuid": f"GPU-{node_id}-0",
                    "pci_bdf": "00000000:59:00.0",
                    "product": "H200",
                }
            ],
        },
    }


def post_batch(app, items, *, token: str = REPLAY_TOKEN) -> httpx.Response:
    async def run() -> httpx.Response:
        async with asgi_client(app) as client:
            return await client.post(
                BATCH_PATH,
                json={"items": items},
                headers={"X-GPU-Fault-Processor-Replay": token},
            )

    return asyncio.run(run())


def statuses(response: httpx.Response) -> dict[str, int]:
    return {item["request_id"]: item["status"] for item in response.json()["results"]}


def bodies(response: httpx.Response) -> dict[str, dict]:
    return {item["request_id"]: item["body"] for item in response.json()["results"]}


def test_telemetry_batch_commits_the_whole_group_once(monkeypatch) -> None:
    """One transaction, and one topology read per cluster."""

    store = RecordingStore()
    _context, app = build_app(monkeypatch, store)
    items = [
        gpu_metrics_item("req-a1", "cluster-a", "node-1"),
        host_telemetry_item("req-a2", "cluster-a", "node-2"),
        gpu_metrics_item("req-b1", "cluster-b", "node-1"),
        host_telemetry_item("req-b2", "cluster-b", "node-2"),
    ]

    response = post_batch(app, items)

    assert response.status_code == 200
    assert statuses(response) == {
        "req-a1": 200,
        "req-a2": 200,
        "req-b1": 200,
        "req-b2": 200,
    }
    assert store.batch_transactions == 1
    assert len(store.ingestion_keys) == 4
    assert store.observation_reads == ["cluster-a", "cluster-b"]
    # Every item is durable, not just the first.
    for cluster_id, node_id in (
        ("cluster-a", "node-1"),
        ("cluster-a", "node-2"),
        ("cluster-b", "node-1"),
        ("cluster-b", "node-2"),
    ):
        assert store.list_collector_statuses(cluster_id, node_id), (
            "expected store.list_collector_statuses(cluster_id, node_id) to be truthy"
        )


def test_telemetry_batch_walks_items_in_lock_order(monkeypatch) -> None:
    """A group holds several nodes' locks, so order is not claim order.

    Both telemetry paths take ``raw_evidence/<cluster>/<node>``. Two
    groups with overlapping node sets deadlock unless every group walks
    its nodes in the same order.
    """

    store = RecordingStore()
    _context, app = build_app(monkeypatch, store)
    items = [
        host_telemetry_item("req-4", "cluster-b", "node-2"),
        gpu_metrics_item("req-3", "cluster-b", "node-1"),
        host_telemetry_item("req-2", "cluster-a", "node-2"),
        gpu_metrics_item("req-1", "cluster-a", "node-1"),
    ]

    response = post_batch(app, items)

    assert set(statuses(response).values()) == {200}
    assert store.ingestion_keys == sorted(store.ingestion_keys)
    assert [key[:2] for key in store.ingestion_keys] == [
        ("cluster-a", "node-1"),
        ("cluster-a", "node-2"),
        ("cluster-b", "node-1"),
        ("cluster-b", "node-2"),
    ]


def test_telemetry_batch_rejects_one_bad_payload_alone(monkeypatch) -> None:
    """A malformed payload is 422 alone; siblings still commit."""

    store = RecordingStore()
    _context, app = build_app(monkeypatch, store)
    broken = gpu_metrics_item("req-bad", "cluster-a", "node-bad")
    broken["payload"]["samples"] = "not-a-list"
    items = [
        gpu_metrics_item("req-good", "cluster-a", "node-good"),
        broken,
        host_telemetry_item("req-host", "cluster-a", "node-host"),
    ]

    response = post_batch(app, items)

    assert statuses(response) == {"req-good": 200, "req-bad": 422, "req-host": 200}
    assert isinstance(bodies(response)["req-bad"]["detail"], list), (
        'expected isinstance(bodies(response)["req-bad"]["detail"], list) to be truthy'
    )
    # Validation runs before any transaction opens, so the bad item
    # never takes a lock and never aborts the group.
    assert store.batch_transactions == 1
    assert [key[1] for key in store.ingestion_keys] == ["node-good", "node-host"]
    assert store.list_collector_statuses("cluster-a", "node-good"), (
        'expected store.list_collector_statuses("cluster-a", "node-good") to be truthy'
    )
    assert store.list_collector_statuses("cluster-a", "node-host"), (
        'expected store.list_collector_statuses("cluster-a", "node-host") to be truthy'
    )
    assert not store.list_collector_statuses("cluster-a", "node-bad"), (
        'expected store.list_collector_statuses("cluster-a", "node-bad") to be falsy'
    )


def test_telemetry_batch_rolls_back_one_failed_item_only(monkeypatch) -> None:
    """A handler that raises rolls back its savepoint, not the group."""

    store = RecordingStore()
    context, app = build_app(monkeypatch, store)
    original = context.gpu_metrics.ingest

    def ingest(batch):
        if batch.node_id == "node-2":
            raise RuntimeError("simulated ingestion failure")
        return original(batch)

    context.gpu_metrics.ingest = ingest
    items = [
        gpu_metrics_item("req-1", "cluster-a", "node-1"),
        gpu_metrics_item("req-2", "cluster-a", "node-2"),
        host_telemetry_item("req-3", "cluster-a", "node-3"),
    ]

    response = post_batch(app, items)

    assert statuses(response) == {"req-1": 200, "req-2": 500, "req-3": 200}
    assert bodies(response)["req-2"]["detail"] == (
        "RuntimeError: simulated ingestion failure"
    )
    assert store.batch_transactions == 1
    assert store.list_collector_statuses("cluster-a", "node-1"), (
        'expected store.list_collector_statuses("cluster-a", "node-1") to be truthy'
    )
    assert store.list_collector_statuses("cluster-a", "node-3"), (
        'expected store.list_collector_statuses("cluster-a", "node-3") to be truthy'
    )


def test_telemetry_batch_retries_item_by_item_when_the_commit_fails(
    monkeypatch,
) -> None:
    """A failed group commit must not fail the whole claim batch."""

    store = RecordingStore()
    _context, app = build_app(monkeypatch, store)
    store.batch_commit_error = RuntimeError("simulated commit failure")
    items = [
        gpu_metrics_item("req-1", "cluster-a", "node-1"),
        host_telemetry_item("req-2", "cluster-a", "node-2"),
    ]

    response = post_batch(app, items)

    assert statuses(response) == {"req-1": 200, "req-2": 200}
    # The group was attempted once, then every item was replayed in its
    # own transaction: two ingestion transactions per item in total.
    assert store.batch_transactions == 1
    assert len(store.ingestion_keys) == 4
    assert store.list_collector_statuses("cluster-a", "node-1"), (
        'expected store.list_collector_statuses("cluster-a", "node-1") to be truthy'
    )
    assert store.list_collector_statuses("cluster-a", "node-2"), (
        'expected store.list_collector_statuses("cluster-a", "node-2") to be truthy'
    )


def test_telemetry_batch_runs_follow_up_after_the_commit(monkeypatch) -> None:
    """Incidents, workflows and the wake stay outside the commit.

    Holding the group transaction open across follow-up work would put
    incident and workflow writes -- and a dispatcher wake -- inside the
    same lock window as every node in the batch.
    """

    store = RecordingStore()
    context, app = build_app(monkeypatch, store)
    original_wake = context.dispatcher.wake

    def wake():
        store.events.append("wake")
        return original_wake()

    context.dispatcher.wake = wake
    items = [
        gpu_metrics_item("req-1", "cluster-a", "node-1"),
        host_telemetry_item("req-2", "cluster-a", "node-2"),
    ]

    response = post_batch(app, items)

    assert set(statuses(response).values()) == {200}
    assert store.events.count("wake") == 2
    assert store.events.index("batch-commit") < store.events.index("wake")
    assert store.events[-1] == "wake"


def test_telemetry_batch_keeps_inventory_and_group_items_together(monkeypatch) -> None:
    """GPU inventory keeps its own batch path; both still answer."""

    store = RecordingStore()
    _context, app = build_app(monkeypatch, store)
    items = [
        inventory_item("req-inv", "cluster-a", "node-1"),
        gpu_metrics_item("req-gpu", "cluster-a", "node-1"),
        host_telemetry_item("req-host", "cluster-a", "node-1"),
    ]

    response = post_batch(app, items)

    assert statuses(response) == {"req-inv": 200, "req-gpu": 200, "req-host": 200}
    assert store.batch_transactions >= 1
    assert [key[1:] for key in store.ingestion_keys] == [
        ("node-1", "gpu-metrics-req-gpu"),
        ("node-1", "host-telemetry-req-host"),
    ]


def test_telemetry_batch_rejects_unsupported_items(monkeypatch) -> None:
    store = RecordingStore()
    _context, app = build_app(monkeypatch, store)
    items = [
        {"request_id": "req-1", "path": "/v1/gpu-events/xid", "payload": {}},
        {"path": GPU_METRICS_PATH, "payload": {}},
        gpu_metrics_item("req-3", "cluster-a", "node-1"),
    ]

    response = post_batch(app, items)

    results = response.json()["results"]
    assert [item["status"] for item in results[:2]] == [422, 422]
    assert all(
        item["body"]["detail"] == "unsupported telemetry batch item"
        for item in results[:2]
    ), (
        'expected all( item["body"]["detail"] == "unsupported telemetry batch item" for item in results[:2] ) to be truthy'
    )
    assert statuses(response)["req-3"] == 200


def test_telemetry_batch_accepts_sixty_four_items(monkeypatch) -> None:
    store = RecordingStore()
    _context, app = build_app(monkeypatch, store)
    items = [
        gpu_metrics_item(f"req-{index}", "cluster-a", f"node-{index}")
        for index in range(64)
    ]

    accepted = post_batch(app, items)
    rejected = post_batch(
        app,
        [
            gpu_metrics_item(f"too-many-{index}", "cluster-a", f"node-too-many-{index}")
            for index in range(65)
        ],
    )

    assert accepted.status_code == 200
    assert len(accepted.json()["results"]) == 64
    assert all(result["status"] == 200 for result in accepted.json()["results"]), (
        'expected all(result["status"] == 200 for result in accepted.json()["results"]) to be truthy'
    )
    assert store.batch_transactions == 1
    assert rejected.status_code == 422
    assert rejected.json()["detail"] == ("telemetry batch must contain 1..64 items")


def test_telemetry_batch_requires_the_processor_replay_token(monkeypatch) -> None:
    store = RecordingStore()
    _context, app = build_app(monkeypatch, store)
    items = [gpu_metrics_item("req-1", "cluster-a", "node-1")]

    response = post_batch(app, items, token="wrong-" + "x" * 32)

    assert response.status_code == 403
    assert store.events == []

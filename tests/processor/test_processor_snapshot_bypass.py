"""Telemetry whose writes are already monotonic can skip the queue.

``gpu-inventory`` ingestion is guarded by ``observed_at`` at every write
it performs -- ``observe_gpu_inventory_snapshots`` skips a snapshot that
is not newer and repeats the test in the upsert's ``ON CONFLICT ...
WHERE``, ``save_collector_statuses_batch`` has the same shape, and the
evidence record is keyed on the snapshot id -- so the queue's lease and
its per-node ordering buy it nothing, while still charging it a queue
row, a lane row, a counter-row trigger and a share of the claim's
O(depth) scan. These tests pin the cut and, more importantly, the two
things that must not come with it: no other channel may be bypassed, and
the body must still reach the endpoint.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import httpx

from gpu_fault.app import ApplicationContext, create_app
from tests._builders import asgi_client, build_store

TOKEN = "processor-snapshot-bypass-token-" + "x" * 32
INVENTORY_PATH = "/v1/collector-events/gpu-inventory"
GPU_METRICS_PATH = "/v1/collector-events/gpu-metrics"
CLUSTER = "cluster-a"
NODE = "node-a"


def build_app(monkeypatch, store, *, bypass: str | None = None):
    monkeypatch.setenv("GPU_FAULT_PROCESSOR_MODE", "active-active")
    monkeypatch.setenv("POD_UID", "pod-snapshot-bypass")
    if bypass is None:
        monkeypatch.delenv("GPU_FAULT_PROCESSOR_SNAPSHOT_BYPASS", raising=False)
    else:
        monkeypatch.setenv("GPU_FAULT_PROCESSOR_SNAPSHOT_BYPASS", bypass)
    context = ApplicationContext(store=store, execution_token=TOKEN)
    return create_app(context)


def now() -> datetime:
    return datetime.now(timezone.utc)


def inventory_payload(
    *, snapshot_id: str = "snapshot-1", observed_at: datetime | None = None
) -> dict:
    return {
        "snapshot_id": snapshot_id,
        "cluster_id": CLUSTER,
        "node_id": NODE,
        "observed_at": (observed_at or now()).isoformat(),
        "source": "NVIDIA_SMI",
        "source_boot_id": "boot-a",
        "devices": [
            {
                "gpu_index": index,
                "gpu_uuid": f"GPU-{NODE}-{index}",
                "pci_bdf": f"00000000:{59 + index}:00.0",
                "product": "H200",
            }
            for index in range(2)
        ],
    }


def gpu_metrics_payload() -> dict:
    return {
        "batch_id": "batch-1",
        "cluster_id": CLUSTER,
        "node_id": NODE,
        "observed_at": now().isoformat(),
        "edge_filter_reasons": ["health-summary"],
        "samples": [],
    }


def post(app, path: str, payload: dict) -> httpx.Response:
    async def run() -> httpx.Response:
        async with asgi_client(app) as client:
            # The collector sends this header, and it is what gives a
            # queued request its ``cluster_id`` -- without it the queue
            # row is unscoped and the per-cluster assertions below would
            # pass whether or not a row was written.
            return await client.post(
                path, json=payload, headers={"X-GPU-Fault-Cluster-ID": CLUSTER}
            )

    return asyncio.run(run())


def get(app, path: str) -> httpx.Response:
    async def run() -> httpx.Response:
        async with asgi_client(app) as client:
            return await client.get(path)

    return asyncio.run(run())


def test_snapshot_bypass_executes_inventory_without_a_queue_row(monkeypatch) -> None:
    """The whole point: durable, and nothing left for the consumer.

    The response body is asserted device-for-device because the bypass
    hands the request on after ``await request.body()`` has already
    consumed the one-shot body shim; without installing a fresh one the
    endpoint parses an empty body and this is a 422.
    """

    store = build_store()
    app = build_app(monkeypatch, store, bypass="1")

    response = post(app, INVENTORY_PATH, inventory_payload())

    assert response.status_code == 200
    body = response.json()
    assert body["snapshot_id"] == "snapshot-1"
    assert [device["gpu_index"] for device in body["devices"]] == [0, 1]
    persisted = store.get_gpu_inventory_snapshot(CLUSTER, NODE)
    assert persisted is not None
    assert len(persisted.devices) == 2
    assert not store.has_incomplete_processor_requests(CLUSTER)


def test_snapshot_bypass_is_off_unless_it_is_asked_for(monkeypatch) -> None:
    """The control arm of the A/B has to be the real queued path."""

    store = build_store()
    app = build_app(monkeypatch, store)

    response = post(app, INVENTORY_PATH, inventory_payload())

    assert response.status_code == 202
    assert response.json()["processor_request_id"]
    assert store.has_incomplete_processor_requests(CLUSTER)
    assert store.get_gpu_inventory_snapshot(CLUSTER, NODE) is None


def test_snapshot_bypass_does_not_apply_to_gpu_metrics(monkeypatch) -> None:
    """The bypassable set is code, and gpu-metrics is not in it.

    ``collector_ingestion_transaction`` is only a named transaction --
    it does not deduplicate by ``batch_id`` -- so gpu-metrics keeps the
    lease. A future edit that widens the set by path prefix would turn
    this green and must not.
    """

    store = build_store()
    app = build_app(monkeypatch, store, bypass="1")

    response = post(app, GPU_METRICS_PATH, gpu_metrics_payload())

    assert response.status_code == 202
    assert store.has_incomplete_processor_requests(CLUSTER)


def test_snapshot_bypass_still_rejects_an_oversize_body(monkeypatch) -> None:
    """The size guard is deliberately ahead of the bypass.

    Bypassing earlier would be cheaper -- it would skip one JSON parse --
    but it would also hand an unbounded body to the endpoint.
    """

    store = build_store()
    monkeypatch.setenv("GPU_FAULT_PROCESSOR_MAX_REQUEST_BYTES", "64")
    app = build_app(monkeypatch, store, bypass="1")

    response = post(app, INVENTORY_PATH, inventory_payload())

    assert response.status_code == 413
    assert store.get_gpu_inventory_snapshot(CLUSTER, NODE) is None


def test_snapshot_bypass_is_counted_per_path_in_metrics(monkeypatch) -> None:
    """An A/B needs the arm and its volume to be readable from /metrics."""

    store = build_store()
    app = build_app(monkeypatch, store, bypass="1")

    post(app, INVENTORY_PATH, inventory_payload())
    post(
        app,
        INVENTORY_PATH,
        inventory_payload(
            snapshot_id="snapshot-2", observed_at=now() + timedelta(seconds=30)
        ),
    )
    metrics = get(app, "/metrics").text

    assert (
        "gpu_fault_processor_queue_bypass_total"
        f'{{path="{INVENTORY_PATH}"}} 2' in metrics
    )
    assert "gpu_fault_processor_queue_bypass_enabled 1" in metrics


def test_bypass_metrics_report_zero_in_the_control_arm(monkeypatch) -> None:
    """A missing series and a zero series read the same in a diff."""

    store = build_store()
    app = build_app(monkeypatch, store)

    post(app, INVENTORY_PATH, inventory_payload())
    metrics = get(app, "/metrics").text

    assert (
        "gpu_fault_processor_queue_bypass_total"
        f'{{path="{INVENTORY_PATH}"}} 0' in metrics
    )
    assert "gpu_fault_processor_queue_bypass_enabled 0" in metrics

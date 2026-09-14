"""Telemetry gets an async path that is not the processor queue.

The inline bypass in ``test_processor_snapshot_bypass.py`` can only take
gpu-inventory, because answering the other two channels in the ingress
process would move a third of the burst's execution out of six worker
replicas into three ingress replicas. The spool takes all three: it keeps
execution in the worker tier and drops what telemetry never needed from
the path in between -- the lane row, the counter-row trigger, the
O(depth) claim window and the completion record carrying a response body
no collector reads.

What these tests pin is the part that makes it correct rather than merely
cheaper: coalescing by primary key must not lose the newer sample when it
lands on a row a consumer already holds, and a failed replay must come
back rather than vanish.
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timedelta, timezone
from threading import Event, Lock, Thread

import httpx
import pytest
from fastapi.testclient import TestClient

from gpu_fault.app import ApplicationContext, _StripedAdmissionScope, create_app
from gpu_fault.processor import (
    ProcessorCoordinator,
    ProcessorLeaseSettings,
    ProcessorSpoolSettings,
)
from gpu_fault.store import InMemoryStore
from tests._builders import asgi_client, build_store, copy_model, processor_request

TOKEN = "telemetry-spool-token-" + "x" * 40
INVENTORY_PATH = "/v1/collector-events/gpu-inventory"
GPU_METRICS_PATH = "/v1/collector-events/gpu-metrics"
HOST_TELEMETRY_PATH = "/v1/collector-events/host-telemetry"
CLUSTER = "cluster-a"
NODE = "node-a"
OWNER = "pod-spool:1"


_OPEN_CLIENTS: list[Client] = []


@pytest.fixture(autouse=True)
def _close_test_clients():
    yield
    while _OPEN_CLIENTS:
        _OPEN_CLIENTS.pop().close()


class Client:
    """Every request to one app runs on one event loop.

    Admission is batched, and the batcher binds its lock, its arrival
    event and its in-flight flush task to whichever loop first submits to
    it. Uvicorn runs exactly one loop for the process lifetime, so it
    never rebinds; ``asyncio.run`` per request gives the second request a
    closed loop to wait on, and it waits forever. Multi-request tests are
    the only place that shape exists, so it lives here rather than being
    worked around in the batcher.
    """

    def __init__(self, app) -> None:
        self.app = app
        self._loop = asyncio.new_event_loop()
        _OPEN_CLIENTS.append(self)

    def close(self) -> None:
        """Drain the batchers on their own loop, as shutdown does.

        Closing the loop out from under a live flush task is what prints
        ``Task was destroyed but it is pending`` and hides a real one.
        """

        self._loop.run_until_complete(
            self.app.state.processor_admission_batcher.close()
        )
        self._loop.run_until_complete(self.app.state.fault_admission_batcher.close())
        self._loop.run_until_complete(self.app.state.evidence_admission_batcher.close())
        self._loop.run_until_complete(self.app.state.telemetry_spool_batcher.close())
        self.app.state.store_io.close()
        self.app.state.fault_store_io.close()
        self.app.state.telemetry_spool_store_io.close()
        self._loop.close()

    def post(self, path: str, payload: dict) -> httpx.Response:
        # The collector sends the cluster header, and it is what scopes a
        # spooled request -- without it the per-cluster assertions would
        # pass whether or not the row was scoped.
        return self._run(
            lambda client: client.post(
                path, json=payload, headers={"X-GPU-Fault-Cluster-ID": CLUSTER}
            )
        )

    def get(self, path: str) -> httpx.Response:
        return self._run(lambda client: client.get(path))

    def _run(self, call) -> httpx.Response:
        async def run() -> httpx.Response:
            async with asgi_client(self.app) as client:
                return await call(client)

        return self._loop.run_until_complete(run())


def build_client(monkeypatch, store, *, spool: str | None = "1") -> Client:
    monkeypatch.setenv("GPU_FAULT_PROCESSOR_MODE", "active-active")
    monkeypatch.setenv("POD_UID", "pod-telemetry-spool")
    monkeypatch.delenv("GPU_FAULT_PROCESSOR_SNAPSHOT_BYPASS", raising=False)
    if spool is None:
        monkeypatch.delenv("GPU_FAULT_TELEMETRY_SPOOL", raising=False)
    else:
        monkeypatch.setenv("GPU_FAULT_TELEMETRY_SPOOL", spool)
    context = ApplicationContext(store=store, execution_token=TOKEN)
    return Client(create_app(context))


def now() -> datetime:
    return datetime.now(timezone.utc)


def test_striped_admission_assigns_whole_batches_to_lanes() -> None:
    scope = _StripedAdmissionScope(batch_size=2, partitions=2)

    assert [scope(None) for _ in range(6)] == [
        "telemetry-0",
        "telemetry-0",
        "telemetry-1",
        "telemetry-1",
        "telemetry-0",
        "telemetry-0",
    ]


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


def gpu_metrics_payload(
    *, batch_id: str = "batch-1", node_id: str = NODE, reasons: list[str] | None = None
) -> dict:
    return {
        "batch_id": batch_id,
        "cluster_id": CLUSTER,
        "node_id": node_id,
        "observed_at": now().isoformat(),
        "edge_filter_reasons": (["health-summary"] if reasons is None else reasons),
        "samples": [],
    }


def host_telemetry_payload(
    *, batch_id: str = "host-1", node_id: str = NODE, reasons: list[str] | None = None
) -> dict:
    return {
        "batch_id": batch_id,
        "cluster_id": CLUSTER,
        "node_id": node_id,
        "observed_at": now().isoformat(),
        "edge_filter_reasons": (["health-summary"] if reasons is None else reasons),
        "collection_errors": [],
        "samples": [],
    }


def make_request(path: str, payload: dict, request_id: str):
    import json

    return copy_model(
        processor_request(path, body=json.dumps(payload).encode(), cluster_id=CLUSTER),
        request_id=request_id,
    )


def spool(store, requests, **kwargs):
    return store.try_spool_telemetry_requests(
        requests,
        max_depth=kwargs.pop("max_depth", 100),
        max_cluster_depth=kwargs.pop("max_cluster_depth", 100),
        **kwargs,
    )


def test_telemetry_is_spooled_instead_of_queued(monkeypatch) -> None:
    """All three channels leave the queue, and none of them get a receipt."""

    store = build_store()
    client = build_client(monkeypatch, store)

    for path, payload in (
        (INVENTORY_PATH, inventory_payload()),
        (GPU_METRICS_PATH, gpu_metrics_payload()),
        (HOST_TELEMETRY_PATH, host_telemetry_payload()),
    ):
        response = client.post(path, payload)
        assert response.status_code == 202, path
        body = response.json()
        assert body["spooled"] is True, path
        # The collector polls a receipt only when it is given one, and
        # the spool keeps no completion record to poll.
        assert "processor_request_id" not in body, path

    assert not store.has_incomplete_processor_requests(CLUSTER), (
        "expected store.has_incomplete_processor_requests(CLUSTER) to be falsy"
    )
    assert store.telemetry_spool_stats()["depth"] == 3


def test_spool_worker_role_starts_only_spool_consumer(monkeypatch) -> None:
    spool_started = Event()
    queue_started = Event()

    def run_spool(processor):
        with processor._state_lock:
            processor._spool_consumer_running = True
        try:
            spool_started.set()
            processor._stop.wait()
        finally:
            with processor._state_lock:
                processor._spool_consumer_running = False

    def run_queue(_processor):
        queue_started.set()

    monkeypatch.setattr(ProcessorCoordinator, "run_telemetry_spool", run_spool)
    monkeypatch.setattr(ProcessorCoordinator, "run_processor", run_queue)
    monkeypatch.setenv("GPU_FAULT_PROCESSOR_MODE", "active-active")
    monkeypatch.setenv("GPU_FAULT_SERVICE_ROLE", "spool-worker")
    monkeypatch.setenv("GPU_FAULT_TELEMETRY_SPOOL", "true")
    monkeypatch.setenv("POD_UID", "pod-spool-role")
    app = create_app(ApplicationContext(store=build_store(), execution_token=TOKEN))

    with TestClient(app) as client:
        assert spool_started.wait(timeout=2), (
            "expected spool_started.wait(timeout=2) to be truthy"
        )
        assert not queue_started.is_set(), "expected queue_started.is_set() to be falsy"
        health = client.get("/healthz").json()
        metrics = client.get("/metrics").text

    assert health["service_role"] == "spool-worker"
    assert health["processor_role"] == "spool-consumer"
    assert "gpu_fault_processor_workers 0" in metrics
    assert "gpu_fault_telemetry_spool_consumer_running 1" in metrics


def test_spool_admission_has_its_own_store_io_executor(monkeypatch) -> None:
    monkeypatch.setenv("GPU_FAULT_PROCESSOR_MODE", "active-active")
    monkeypatch.setenv("GPU_FAULT_SERVICE_ROLE", "ingress")
    monkeypatch.setenv("GPU_FAULT_TELEMETRY_SPOOL", "true")
    monkeypatch.setenv("POD_UID", "pod-spool-admission-lane")
    monkeypatch.setenv("GPU_FAULT_STORE_IO_WORKERS", "1")
    monkeypatch.setenv("GPU_FAULT_STORE_IO_MAX_IN_FLIGHT", "1")
    monkeypatch.setenv("GPU_FAULT_TELEMETRY_SPOOL_STORE_IO_WORKERS", "1")
    monkeypatch.setenv("GPU_FAULT_TELEMETRY_SPOOL_STORE_IO_MAX_IN_FLIGHT", "1")
    store = build_store()
    app = create_app(ApplicationContext(store=store, execution_token=TOKEN))
    client = Client(app)
    entered = Event()
    release = Event()

    def occupy_general_store_io():
        entered.set()
        assert release.wait(timeout=5), "expected release.wait(timeout=5) to be truthy"

    async def scenario(http_client):
        blocked = asyncio.create_task(app.state.store_io.run(occupy_general_store_io))
        assert await asyncio.to_thread(entered.wait, 2)
        response = await http_client.post(
            INVENTORY_PATH,
            json=inventory_payload(),
            headers={"X-GPU-Fault-Cluster-ID": CLUSTER},
        )
        release.set()
        await blocked
        return response

    response = client._run(scenario)

    assert response.status_code == 202
    assert response.json()["spooled"] is True
    assert store.telemetry_spool_stats()["depth"] == 1
    assert app.state.telemetry_spool_store_io.rejected_total == 0


def test_spoolable_telemetry_gets_its_own_request_budget(monkeypatch) -> None:
    store = build_store()
    original = store.try_spool_telemetry_requests

    def slow_spool(*args, **kwargs):
        time.sleep(0.5)
        return original(*args, **kwargs)

    monkeypatch.setattr(store, "try_spool_telemetry_requests", slow_spool)
    monkeypatch.setenv("GPU_FAULT_REQUEST_BUDGET_SECONDS", "0.2")
    monkeypatch.setenv("GPU_FAULT_TELEMETRY_REQUEST_BUDGET_SECONDS", "10")
    client = build_client(monkeypatch, store)

    response = client.post(GPU_METRICS_PATH, gpu_metrics_payload())

    assert response.status_code == 202, response.text
    assert float(response.headers["X-GPU-Fault-Server-Duration-Ms"]) >= 450


def test_spool_admission_batches_across_clusters(monkeypatch) -> None:
    class RecordingStore(InMemoryStore):
        def __init__(self):
            super().__init__()
            self.admission_clusters = []

        def try_spool_telemetry_requests(self, requests, **kwargs):
            self.admission_clusters.append([item.cluster_id for item in requests])
            return super().try_spool_telemetry_requests(requests, **kwargs)

    monkeypatch.setenv("GPU_FAULT_PROCESSOR_MODE", "active-active")
    monkeypatch.setenv("GPU_FAULT_SERVICE_ROLE", "ingress")
    monkeypatch.setenv("GPU_FAULT_TELEMETRY_SPOOL", "true")
    monkeypatch.setenv("GPU_FAULT_TELEMETRY_SPOOL_BATCH_GROUPS", "1")
    monkeypatch.setenv("GPU_FAULT_TELEMETRY_SPOOL_BATCH_DELAY_SECONDS", "0.01")
    monkeypatch.setenv("POD_UID", "pod-cross-cluster-spool")
    store = RecordingStore()
    app = create_app(ApplicationContext(store=store, execution_token=TOKEN))
    first = make_request(
        GPU_METRICS_PATH, gpu_metrics_payload(node_id="node-a"), "req-a"
    )
    second = copy_model(
        make_request(GPU_METRICS_PATH, gpu_metrics_payload(node_id="node-b"), "req-b"),
        cluster_id="cluster-b",
    )

    async def scenario():
        results = await asyncio.gather(
            app.state.telemetry_spool_batcher.submit(first),
            app.state.telemetry_spool_batcher.submit(second),
        )
        await app.state.telemetry_spool_batcher.close()
        return results

    results = asyncio.run(scenario())
    app.state.store_io.close()
    app.state.telemetry_spool_store_io.close()

    assert [item[1] for item in results] == [None, None]
    assert store.admission_clusters == [[CLUSTER, "cluster-b"]]


def test_spool_is_off_unless_it_is_asked_for(monkeypatch) -> None:
    """The control arm of the A/B has to be the real queued path."""

    store = build_store()
    client = build_client(monkeypatch, store, spool=None)

    response = client.post(GPU_METRICS_PATH, gpu_metrics_payload())

    assert response.status_code == 202
    assert response.json()["processor_request_id"]
    assert store.has_incomplete_processor_requests(CLUSTER), (
        "expected store.has_incomplete_processor_requests(CLUSTER) to be truthy"
    )
    assert store.telemetry_spool_stats()["depth"] == 0


def test_enabled_spool_rejects_an_item_limit_above_request_limit(monkeypatch) -> None:
    monkeypatch.setenv("GPU_FAULT_PROCESSOR_MAX_REQUEST_BYTES", "64")
    monkeypatch.setenv("GPU_FAULT_TELEMETRY_SPOOL_MAX_ITEM_BYTES", "65")

    with pytest.raises(ValueError, match="spool item byte limit"):
        build_client(monkeypatch, build_store())


def test_only_routine_samples_on_three_paths_can_be_spooled() -> None:
    """Pin the gate: routine, and on a channel whose writes are monotonic.

    The spool keeps no lane, no lease and no completion record. A fault
    that reached it would lose its ordering and its correlation scope; an
    edge-filtered batch that reports a breach would lose the completion
    record the collector polls and the lane that serialises it against the
    fault it belongs to. Node logs and rank heartbeats are routine but are
    not monotonic per node either, so being tier 100 must not be enough.

    Asserted on the request rather than over HTTP because a fault in
    active-active mode blocks on its own execution, and no consumer runs
    in this process: the end-to-end wiring of the gate is covered by
    ``test_telemetry_is_spooled_instead_of_queued`` and its control arm.
    """

    fault = make_request(
        "/v1/gpu-events/nvidia-kernel",
        {
            "event_id": "evt-1",
            "cluster_id": CLUSTER,
            "node_id": NODE,
            "observed_at": now().isoformat(),
            "source": "NVIDIA_KERNEL",
            "raw_message": "NVRM: Xid (PCI:0000:53:00): 79",
            "xid": 79,
        },
        "req-fault",
    )
    control = make_request("/v1/node-actions/act-1/claim", {}, "req-control")

    assert (fault.queue_priority(), fault.spoolable()) == (10, False)
    assert (control.queue_priority(), control.spoolable()) == (50, False)
    for path, payload in (
        (INVENTORY_PATH, inventory_payload()),
        (GPU_METRICS_PATH, gpu_metrics_payload()),
        (HOST_TELEMETRY_PATH, host_telemetry_payload()),
    ):
        request = make_request(path, payload, f"req-{path}")
        assert request.queue_priority() == 100, path
        assert request.spoolable() is True, path

        # Node logs require an explicit health-summary marker; an empty
        # or malformed batch is evidence and must not enter the spool.
        for path, payload in (
            (
                "/v1/collector-events/node-logs",
                {
                    "cluster_id": CLUSTER,
                    "node_id": NODE,
                    "collected_at": now().isoformat(),
                    "entries": [],
                },
            ),
            (
                "/v1/training-progress",
                {
                    "cluster_id": CLUSTER,
                    "attempt_id": "attempt-1",
                    "rank": 0,
                    "observed_at": now().isoformat(),
                },
            ),
        ):
            request = make_request(path, payload, f"req-{path}")
            assert request.queue_priority() == (
                50 if path == "/v1/collector-events/node-logs" else 100
            ), path
            assert request.spoolable() is False, path

        node_health = make_request(
            "/v1/collector-events/node-logs",
            {
                "cluster_id": CLUSTER,
                "node_id": NODE,
                "collected_at": now().isoformat(),
                "entries": [],
                "edge_filter_reasons": ["health-summary"],
            },
            "req-node-log-health",
        )
        assert node_health.queue_priority() == 100
        assert node_health.spoolable() is True

    # Evidence: same endpoint, one tier up, and out of the spool.
    breach = make_request(
        HOST_TELEMETRY_PATH,
        host_telemetry_payload(reasons=["threshold:x"]),
        "req-breach",
    )
    assert breach.queue_priority() == 50
    assert breach.spoolable() is False


def test_a_summary_sample_coalesces_onto_its_own_lane() -> None:
    """Latest-wins channels keep one row per lane, as the queue did."""

    store = build_store()
    first = make_request(GPU_METRICS_PATH, gpu_metrics_payload(batch_id="b1"), "req-1")
    second = make_request(GPU_METRICS_PATH, gpu_metrics_payload(batch_id="b2"), "req-2")

    assert spool(store, [first]) == [(first, None)]
    assert spool(store, [second]) == [(second, "coalesced")]
    assert store.telemetry_spool_stats()["depth"] == 1

    claimed = store.claim_telemetry_spool(
        OWNER, now=now(), lease_duration=timedelta(seconds=60), limit=10
    )
    assert len(claimed) == 1
    assert claimed[0].payload["batch_id"] == "b2"


def test_threshold_batches_stay_in_the_queue_even_with_the_spool_on(
    monkeypatch,
) -> None:
    """Edge-filtered batches feed consecutive-breach counters.

    Dropping an intermediate sample there loses a confirmation, so a
    batch reporting a breach never enters a store that collapses a lane to
    its latest write. The tier is what keeps it out: evidence is 50, and
    the spool only takes 100.
    """

    store = build_store()
    client = build_client(monkeypatch, store)

    responses = [
        client.post(
            GPU_METRICS_PATH,
            gpu_metrics_payload(
                batch_id=f"b{index}", reasons=["threshold:sm_clock_throttle"]
            ),
        )
        for index in range(3)
    ]

    assert [response.status_code for response in responses] == [202, 202, 202]
    # A receipt each, all distinct: none of the three was collapsed into
    # another, and the collector has something to poll for every one.
    receipts = {response.json()["processor_request_id"] for response in responses}
    assert len(receipts) == 3
    assert store.telemetry_spool_stats()["depth"] == 0
    assert store.processor_queue_stats()["depth"] == 3


def test_one_batch_carrying_two_samples_for_a_lane_keeps_the_newer() -> None:
    """``ON CONFLICT`` refuses the same key twice in one statement.

    Which means the collapse has to happen before the write, and the
    sample that survives has to be the last one -- it is the newer one.
    """

    store = build_store()
    first = make_request(GPU_METRICS_PATH, gpu_metrics_payload(batch_id="old"), "req-1")
    second = make_request(
        GPU_METRICS_PATH, gpu_metrics_payload(batch_id="new"), "req-2"
    )

    results = spool(store, [first, second])

    assert [reason for _, reason in results] == [None, "coalesced"]
    assert store.telemetry_spool_stats()["depth"] == 1
    claimed = store.claim_telemetry_spool(
        OWNER, now=now(), lease_duration=timedelta(seconds=60), limit=10
    )
    assert claimed[0].payload["batch_id"] == "new"


def test_a_leased_row_can_be_superseded_without_losing_the_new_sample() -> None:
    """The reason completion is fenced on ``revision``.

    Coalescing is allowed to overwrite a row another consumer already
    holds -- both replays converge, because every telemetry write keeps
    the newer ``observed_at``. What must not happen is the consumer
    holding the older payload deleting the row on success and taking the
    newer sample with it.
    """

    store = build_store()
    spool(
        store,
        [make_request(GPU_METRICS_PATH, gpu_metrics_payload(batch_id="old"), "req-1")],
    )
    claimed = store.claim_telemetry_spool(
        OWNER, now=now(), lease_duration=timedelta(seconds=60), limit=10
    )
    assert len(claimed) == 1

    spool(
        store,
        [make_request(GPU_METRICS_PATH, gpu_metrics_payload(batch_id="new"), "req-2")],
    )

    # The stale consumer reports success and deletes nothing.
    assert store.complete_telemetry_spool(claimed) == 0
    assert store.telemetry_spool_stats()["depth"] == 1
    # And the newer payload is claimable straight away rather than
    # waiting out the lease it just invalidated.
    again = store.claim_telemetry_spool(
        OWNER, now=now(), lease_duration=timedelta(seconds=60), limit=10
    )
    assert [item.payload["batch_id"] for item in again] == ["new"]


def test_a_lease_expires_without_a_reaper() -> None:
    """A lease is only ``available_at`` in the future.

    A consumer that dies holding rows stops renewing nothing, because
    there is nothing to renew: the rows fall back inside the claim window
    on their own.
    """

    store = build_store()
    spool(store, [make_request(GPU_METRICS_PATH, gpu_metrics_payload(), "req-1")])
    start = now()
    assert store.claim_telemetry_spool(
        OWNER, now=start, lease_duration=timedelta(seconds=30), limit=10
    ), (
        "expected store.claim_telemetry_spool( OWNER, now=start, lease_duration=timedelta(seconds=30), limit=10, ) to be truthy"
    )
    assert not store.claim_telemetry_spool(
        "pod-other:1",
        now=start + timedelta(seconds=10),
        lease_duration=timedelta(seconds=30),
        limit=10,
    ), (
        'expected store.claim_telemetry_spool( "pod-other:1", now=start + timedelta(seconds=10), lease_duration=timedelta(seconds=30), limit=10, ) to be falsy'
    )
    assert store.claim_telemetry_spool(
        "pod-other:1",
        now=start + timedelta(seconds=31),
        lease_duration=timedelta(seconds=30),
        limit=10,
    ), (
        'expected store.claim_telemetry_spool( "pod-other:1", now=start + timedelta(seconds=31), lease_duration=timedelta(seconds=30), limit=10, ) to be truthy'
    )


def test_spool_claim_respects_payload_byte_budget() -> None:
    store = build_store()
    requests = [
        make_request(
            GPU_METRICS_PATH,
            {
                **gpu_metrics_payload(
                    batch_id=f"batch-{index}", node_id=f"node-{index}"
                ),
                "padding": "x" * 512,
            },
            f"req-{index}",
        )
        for index in range(3)
    ]
    spool(store, requests)
    first_size = len(requests[0].body())

    claimed = store.claim_telemetry_spool(
        OWNER,
        now=now(),
        lease_duration=timedelta(seconds=60),
        limit=10,
        max_bytes=first_size + 16,
    )

    assert len(claimed) == 1
    assert claimed[0].payload_bytes <= first_size + 16


def test_spool_claim_can_select_one_telemetry_path() -> None:
    store = build_store()
    requests = [
        make_request(GPU_METRICS_PATH, gpu_metrics_payload(), "req-gpu"),
        make_request(HOST_TELEMETRY_PATH, host_telemetry_payload(), "req-host"),
    ]
    spool(store, requests, max_depth=200, max_cluster_depth=200)

    claimed = store.claim_telemetry_spool(
        OWNER,
        now=now(),
        lease_duration=timedelta(seconds=60),
        limit=10,
        path=HOST_TELEMETRY_PATH,
    )

    assert [item.path for item in claimed] == [HOST_TELEMETRY_PATH]


def test_abandoning_an_unexecuted_claim_preserves_retry_budget() -> None:
    store = build_store()
    request = make_request(GPU_METRICS_PATH, gpu_metrics_payload(), "req-abandon")
    spool(store, [request])
    at = now()

    for _ in range(store.TELEMETRY_SPOOL_MAX_ATTEMPTS + 2):
        claimed = store.claim_telemetry_spool(
            OWNER, now=at, lease_duration=timedelta(seconds=60), limit=1
        )
        assert len(claimed) == 1
        assert claimed[0].attempts == 1
        assert store.abandon_telemetry_spool_claims(claimed, now=at) == 1

    assert store.telemetry_spool_stats()["depth"] == 1


def test_spool_replay_batches_are_bounded_by_bytes() -> None:
    store = build_store()
    requests = [
        make_request(
            GPU_METRICS_PATH,
            {
                **gpu_metrics_payload(
                    batch_id=f"batch-{index}", node_id=f"node-{index}"
                ),
                "padding": "x" * 256,
            },
            f"req-{index}",
        )
        for index in range(4)
    ]
    spool(store, requests)
    claimed = store.claim_telemetry_spool(
        OWNER, now=now(), lease_duration=timedelta(seconds=60), limit=10
    )
    one_item_limit = claimed[0].payload_bytes + 300
    processor = ProcessorCoordinator(
        store,
        owner_id=OWNER,
        internal_token=TOKEN,
        active_consumers=False,
        spool=ProcessorSpoolSettings(
            telemetry_spool_enabled=True,
            telemetry_spool_max_in_flight_bytes=one_item_limit * 4,
            telemetry_spool_replay_batch_max_bytes=one_item_limit,
        ),
    )

    batches = list(processor._telemetry_spool_batches(claimed, max_items=16))

    assert [len(batch) for batch in batches] == [1, 1, 1, 1]


def test_spool_consumer_claims_only_immediately_executable_batches(monkeypatch) -> None:
    store = build_store()
    requests = [
        make_request(
            GPU_METRICS_PATH,
            gpu_metrics_payload(batch_id=f"batch-{index}", node_id=f"node-{index}"),
            f"req-{index}",
        )
        for index in range(128)
    ]
    spool(store, requests, max_depth=200, max_cluster_depth=200)
    processor = ProcessorCoordinator(
        store,
        owner_id=OWNER,
        internal_token=TOKEN,
        active_consumers=False,
        spool=ProcessorSpoolSettings(
            telemetry_spool_enabled=True, telemetry_spool_workers=2
        ),
        lease=ProcessorLeaseSettings(poll_seconds=0.01),
    )
    replay_gate = Event()
    both_started = Event()
    replay_lock = Lock()
    started = 0

    def replay(items):
        nonlocal started
        with replay_lock:
            started += 1
            if started == 2:
                both_started.set()
        assert replay_gate.wait(timeout=5), (
            "expected replay_gate.wait(timeout=5) to be truthy"
        )
        store.complete_telemetry_spool(items)

    monkeypatch.setattr(processor, "_replay_telemetry_spool", replay)
    thread = Thread(target=processor.run_telemetry_spool)
    thread.start()
    try:
        assert both_started.wait(timeout=5), (
            "expected both_started.wait(timeout=5) to be truthy"
        )
        stats = store.telemetry_spool_stats(now=now())
        assert stats["leased"] == 128
        assert processor.spool_claim_rows_total == 128
        assert processor.spool_released_total == 0
        assert processor.spool_abandoned_total == 0
        assert processor.spool_dropped_total == 0
    finally:
        processor.stop()
        replay_gate.set()
        thread.join(timeout=5)
    assert not thread.is_alive(), "expected thread.is_alive() to be falsy"


def test_spool_notification_listener_wakes_the_consumer(monkeypatch) -> None:
    store = build_store()
    processor = ProcessorCoordinator(
        store,
        owner_id=OWNER,
        internal_token=TOKEN,
        active_consumers=False,
        spool=ProcessorSpoolSettings(
            telemetry_spool_enabled=True,
            telemetry_spool_workers=1,
            telemetry_spool_notification_fallback_seconds=5,
        ),
        lease=ProcessorLeaseSettings(poll_seconds=0.01),
    )
    replayed = Event()

    def replay(items):
        store.complete_telemetry_spool(items)
        replayed.set()

    monkeypatch.setattr(processor, "_replay_telemetry_spool", replay)
    thread = Thread(target=processor.run_telemetry_spool)
    thread.start()
    try:
        deadline = time.monotonic() + 2
        while processor.spool_claim_rounds_total < 3 and time.monotonic() < deadline:
            time.sleep(0.01)
        spool(
            store,
            [make_request(GPU_METRICS_PATH, gpu_metrics_payload(), "req-notified")],
        )
        started = time.monotonic()
        processor.notify_telemetry_spool_work_available("{}")

        assert replayed.wait(timeout=1), (
            "expected replayed.wait(timeout=1) to be truthy"
        )
        assert time.monotonic() - started < 1
        runtime = processor.metrics_snapshot()["spool"]
        assert runtime["notifications_received"] == 1
        assert runtime["notification_fallback_seconds"] == 5
    finally:
        processor.stop()
        thread.join(timeout=5)
    assert not thread.is_alive(), "expected thread.is_alive() to be falsy"


def test_spool_fallback_poll_recovers_a_missed_notification(monkeypatch) -> None:
    store = build_store()
    processor = ProcessorCoordinator(
        store,
        owner_id=OWNER,
        internal_token=TOKEN,
        active_consumers=False,
        spool=ProcessorSpoolSettings(
            telemetry_spool_enabled=True,
            telemetry_spool_workers=1,
            telemetry_spool_notification_fallback_seconds=2,
        ),
        lease=ProcessorLeaseSettings(poll_seconds=0.01),
    )
    replayed = Event()

    def replay(items):
        store.complete_telemetry_spool(items)
        replayed.set()

    monkeypatch.setattr(processor, "_replay_telemetry_spool", replay)
    thread = Thread(target=processor.run_telemetry_spool)
    thread.start()
    try:
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            if processor.metrics_snapshot()["spool"]["fallback_polls"]:
                break
            time.sleep(0.01)
        time.sleep(0.05)
        spool(
            store,
            [make_request(GPU_METRICS_PATH, gpu_metrics_payload(), "req-fallback")],
        )
        started = time.monotonic()

        assert replayed.wait(timeout=3), (
            "expected replayed.wait(timeout=3) to be truthy"
        )
        assert time.monotonic() - started >= 1.5
    finally:
        processor.stop()
        thread.join(timeout=5)
    assert not thread.is_alive(), "expected thread.is_alive() to be falsy"


def test_spool_listener_reports_connection_state() -> None:
    class ListenerStore(InMemoryStore):
        def listen_telemetry_spool_notifications(self, stop, on_notification, on_state):
            on_state(True)
            on_notification('{"path":"gpu"}')
            stop.wait(5)
            on_state(False)

    processor = ProcessorCoordinator(
        ListenerStore(),
        owner_id=OWNER,
        internal_token=TOKEN,
        active_consumers=False,
        spool=ProcessorSpoolSettings(telemetry_spool_enabled=True),
    )
    thread = Thread(target=processor.run_telemetry_spool_notifications)
    thread.start()
    try:
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            runtime = processor.metrics_snapshot()["spool"]
            if runtime["notifications_received"] == 1:
                break
            time.sleep(0.01)
        assert runtime["notifications_enabled"] == 1
        assert runtime["notifications_received"] == 1
    finally:
        processor.stop()
        thread.join(timeout=5)
    assert not thread.is_alive(), "expected thread.is_alive() to be falsy"


def test_spool_consumer_weights_paths_and_borrows_idle_slots(monkeypatch) -> None:
    store = build_store()
    requests = []
    for index in range(64):
        requests.append(
            make_request(
                INVENTORY_PATH,
                {
                    **inventory_payload(snapshot_id=f"snapshot-{index}"),
                    "node_id": f"inventory-node-{index}",
                },
                f"inventory-{index}",
            )
        )
        requests.append(
            make_request(
                HOST_TELEMETRY_PATH,
                host_telemetry_payload(
                    batch_id=f"host-{index}", node_id=f"host-node-{index}"
                ),
                f"host-{index}",
            )
        )
    for index in range(128):
        requests.append(
            make_request(
                GPU_METRICS_PATH,
                gpu_metrics_payload(
                    batch_id=f"gpu-{index}", node_id=f"gpu-node-{index}"
                ),
                f"gpu-{index}",
            )
        )
    spool(store, requests, max_depth=300, max_cluster_depth=300)
    processor = ProcessorCoordinator(
        store,
        owner_id=OWNER,
        internal_token=TOKEN,
        active_consumers=False,
        spool=ProcessorSpoolSettings(
            telemetry_spool_enabled=True, telemetry_spool_workers=4
        ),
        lease=ProcessorLeaseSettings(poll_seconds=0.01),
    )
    replay_gate = Event()
    all_started = Event()
    replay_lock = Lock()
    started_paths = []

    def replay(items):
        assert len({item.path for item in items}) == 1
        with replay_lock:
            started_paths.append(items[0].path)
            if len(started_paths) == 4:
                all_started.set()
        assert replay_gate.wait(timeout=5), (
            "expected replay_gate.wait(timeout=5) to be truthy"
        )
        store.complete_telemetry_spool(items)

    monkeypatch.setattr(processor, "_replay_telemetry_spool", replay)
    thread = Thread(target=processor.run_telemetry_spool)
    thread.start()
    try:
        assert all_started.wait(timeout=5), (
            "expected all_started.wait(timeout=5) to be truthy"
        )
        assert started_paths.count(INVENTORY_PATH) == 1
        assert started_paths.count(GPU_METRICS_PATH) == 2
        assert started_paths.count(HOST_TELEMETRY_PATH) == 1
        assert processor.metrics_snapshot()["spool"]["rows_by_path"] == {
            INVENTORY_PATH: 64,
            GPU_METRICS_PATH: 128,
            HOST_TELEMETRY_PATH: 64,
        }
    finally:
        processor.stop()
        replay_gate.set()
        thread.join(timeout=5)
    assert not thread.is_alive(), "expected thread.is_alive() to be falsy"


def test_fault_backlog_throttles_spool_replay_workers(monkeypatch) -> None:
    store = build_store()
    requests = [
        make_request(
            GPU_METRICS_PATH,
            gpu_metrics_payload(batch_id=f"batch-{index}", node_id=f"node-{index}"),
            f"req-{index}",
        )
        for index in range(32)
    ]
    spool(store, requests)
    monkeypatch.setattr(store, "processor_fault_backlog_depth", lambda: 1)
    processor = ProcessorCoordinator(
        store,
        owner_id=OWNER,
        internal_token=TOKEN,
        active_consumers=False,
        spool=ProcessorSpoolSettings(
            telemetry_spool_enabled=True,
            telemetry_spool_workers=2,
            telemetry_spool_fault_pressure_workers=1,
            telemetry_spool_fault_pressure_poll_seconds=0.01,
        ),
        lease=ProcessorLeaseSettings(poll_seconds=0.01),
    )
    replay_gate = Event()
    first_started = Event()
    second_started = Event()
    replay_lock = Lock()
    started = 0

    def replay(items):
        nonlocal started
        with replay_lock:
            started += 1
            first_started.set()
            if started > 1:
                second_started.set()
        assert replay_gate.wait(timeout=5), (
            "expected replay_gate.wait(timeout=5) to be truthy"
        )
        store.complete_telemetry_spool(items)

    monkeypatch.setattr(processor, "_replay_telemetry_spool", replay)
    thread = Thread(target=processor.run_telemetry_spool)
    thread.start()
    try:
        assert first_started.wait(timeout=5), (
            "expected first_started.wait(timeout=5) to be truthy"
        )
        assert not second_started.wait(timeout=0.2), (
            "expected second_started.wait(timeout=0.2) to be falsy"
        )
        runtime = processor.metrics_snapshot()["spool"]
        assert runtime["fault_pressure_active"] == 1
        assert runtime["fault_backlog_depth"] == 1
    finally:
        processor.stop()
        replay_gate.set()
        thread.join(timeout=5)
    assert not thread.is_alive(), "expected thread.is_alive() to be falsy"


def test_spool_replay_uses_the_in_process_batch_handler(monkeypatch) -> None:
    monkeypatch.setenv("GPU_FAULT_PROCESSOR_MODE", "active-active")
    monkeypatch.setenv("GPU_FAULT_SERVICE_ROLE", "spool-worker")
    monkeypatch.setenv("GPU_FAULT_TELEMETRY_SPOOL", "true")
    monkeypatch.setenv("POD_UID", "pod-direct-spool-replay")
    store = build_store()
    app = create_app(ApplicationContext(store=store, execution_token=TOKEN))
    request = make_request(GPU_METRICS_PATH, gpu_metrics_payload(), "req-direct")
    spool(store, [request])
    claimed = store.claim_telemetry_spool(
        OWNER, now=now(), lease_duration=timedelta(seconds=60), limit=1
    )

    def fail_http(*_args, **_kwargs):
        raise AssertionError("direct replay must not use loopback HTTP")

    monkeypatch.setattr("gpu_fault.processor.coordinator.urlopen", fail_http)
    processor = app.state.processor
    assert processor.telemetry_spool_replay_handler is not None
    processor._replay_telemetry_spool(claimed)
    runtime = processor.metrics_snapshot()["spool"]
    app.state.store_io.close()
    app.state.telemetry_spool_store_io.close()

    assert store.telemetry_spool_stats()["depth"] == 0
    assert runtime["direct_replay"] == 1
    assert runtime["http_replay"] == 0


def test_a_failed_replay_comes_back_and_then_gives_up() -> None:
    """Retried, with a backoff, and bounded.

    Telemetry is a stream: a payload the endpoint cannot execute is worth
    less than the spool depth it occupies, and the next sample from the
    same node restates the state anyway. A fault would never be dropped
    this way, and no fault reaches the spool.
    """

    store = build_store()
    spool(store, [make_request(GPU_METRICS_PATH, gpu_metrics_payload(), "req-1")])
    at = now()
    for attempt in range(store.TELEMETRY_SPOOL_MAX_ATTEMPTS):
        claimed = store.claim_telemetry_spool(
            OWNER, now=at, lease_duration=timedelta(seconds=30), limit=10
        )
        assert len(claimed) == 1, attempt
        released, dropped = store.release_telemetry_spool(
            claimed, now=at, backoff=timedelta(seconds=1)
        )
        if attempt < store.TELEMETRY_SPOOL_MAX_ATTEMPTS - 1:
            assert (released, dropped) == (1, 0), attempt
        else:
            assert (released, dropped) == (0, 1), attempt
        at += timedelta(seconds=2)
    assert store.telemetry_spool_stats()["depth"] == 0


def test_a_backoff_is_not_applied_to_a_newer_sample() -> None:
    """Release carries the same fence as completion.

    A sample that arrived while the replay was failing has already pulled
    the row's availability back to now, and neither the backoff nor the
    drop is its to inherit. The attempt count is deliberately *not* reset
    by coalescing: a node whose payload the endpoint cannot execute would
    otherwise retry forever as long as it keeps sending, and the whole
    reason a telemetry row may be dropped is that the next sample restates
    the state anyway.
    """

    store = build_store()
    spool(
        store,
        [make_request(GPU_METRICS_PATH, gpu_metrics_payload(batch_id="old"), "req-1")],
    )
    at = now()
    claimed = store.claim_telemetry_spool(
        OWNER, now=at, lease_duration=timedelta(seconds=30), limit=10
    )
    # Same instant as the claim, so the assertion below is about the
    # fence and not about a coalesce clock a fraction of a millisecond
    # ahead of ``at``.
    spool(
        store,
        [make_request(GPU_METRICS_PATH, gpu_metrics_payload(batch_id="new"), "req-2")],
        now=at,
    )

    assert store.release_telemetry_spool(
        claimed, now=at, backoff=timedelta(seconds=600)
    ) == (0, 0)
    # Not parked ten minutes out behind a failure that was not its own.
    fresh = store.claim_telemetry_spool(
        OWNER, now=at, lease_duration=timedelta(seconds=30), limit=10
    )
    assert [item.payload["batch_id"] for item in fresh] == ["new"]


def test_the_spool_has_its_own_depth_cap() -> None:
    """Backpressure without a counter row.

    The processor queue keeps its depth in a single row per cluster whose
    statement trigger holds a ``FOR UPDATE`` to commit -- the convoy this
    whole change exists to leave behind. The cap here reads a grouped
    count instead, so it has to be checked at all.
    """

    store = build_store()
    requests = [
        make_request(
            GPU_METRICS_PATH,
            # One node each: four lanes, so the cap is what stops them
            # rather than the collapse.
            gpu_metrics_payload(batch_id=f"b{index}", node_id=f"node-{index}"),
            f"req-{index}",
        )
        for index in range(4)
    ]

    results = spool(store, requests, max_depth=2)

    assert [reason for _, reason in results] == [None, None, "global", "global"]
    assert store.telemetry_spool_stats()["depth"] == 2

    cluster_capped = spool(
        store,
        [make_request(HOST_TELEMETRY_PATH, host_telemetry_payload(), "req-host")],
        max_cluster_depth=2,
    )
    assert cluster_capped == [(cluster_capped[0][0], "cluster")]


def test_a_full_spool_answers_429_with_a_retry_after(monkeypatch) -> None:
    store = build_store()
    monkeypatch.setenv("GPU_FAULT_TELEMETRY_SPOOL_MAX_DEPTH", "1")
    client = build_client(monkeypatch, store)

    first = client.post(GPU_METRICS_PATH, gpu_metrics_payload())
    second = client.post(HOST_TELEMETRY_PATH, host_telemetry_payload())

    assert first.status_code == 202
    assert second.status_code == 429
    assert second.headers["Retry-After"]
    assert second.json()["scope"] == "global"


def test_spool_metrics_report_zero_in_the_control_arm(monkeypatch) -> None:
    """A missing series and a zero series read the same in a diff."""

    store = build_store()
    client = build_client(monkeypatch, store, spool=None)

    client.post(GPU_METRICS_PATH, gpu_metrics_payload())
    metrics = client.get("/metrics").text

    assert "gpu_fault_telemetry_spool_enabled 0" in metrics
    assert "gpu_fault_telemetry_spool_depth 0" in metrics
    assert "gpu_fault_telemetry_spool_admitted_total 0" in metrics


def test_spool_metrics_carry_the_arm_and_its_volume(monkeypatch) -> None:
    store = build_store()
    client = build_client(monkeypatch, store)

    client.post(GPU_METRICS_PATH, gpu_metrics_payload())
    client.post(INVENTORY_PATH, inventory_payload())
    metrics = client.get("/metrics").text

    assert "gpu_fault_telemetry_spool_enabled 1" in metrics
    assert "gpu_fault_telemetry_spool_depth 2" in metrics
    assert "gpu_fault_telemetry_spool_admitted_total 2" in metrics
    assert "gpu_fault_telemetry_spool_replay_batch_max_items 64" in metrics
    assert "gpu_fault_telemetry_spool_errors_total 0" in metrics
    assert "gpu_fault_telemetry_spool_notification_fallback_seconds 5" in metrics
    assert (
        "gpu_fault_telemetry_spool_admitted_by_path_total"
        f'{{path="{INVENTORY_PATH}"}} 1' in metrics
    )
    assert (
        "gpu_fault_telemetry_spool_cluster_depth"
        f'{{cluster_id="{CLUSTER}"}} 2' in metrics
    )


def test_the_inline_bypass_wins_over_the_spool(monkeypatch) -> None:
    """Both flags on is the intended production shape, not an accident.

    gpu-inventory is answered in the ingress process because it is cheap
    enough to be; the other two are spooled because answering them there
    would move a third of the burst's execution out of the worker tier.
    """

    store = build_store()
    monkeypatch.setenv("GPU_FAULT_PROCESSOR_SNAPSHOT_BYPASS", "1")
    monkeypatch.setenv("GPU_FAULT_TELEMETRY_SPOOL", "1")
    monkeypatch.setenv("GPU_FAULT_PROCESSOR_MODE", "active-active")
    monkeypatch.setenv("POD_UID", "pod-both-arms")
    client = Client(create_app(ApplicationContext(store=store, execution_token=TOKEN)))

    inventory = client.post(INVENTORY_PATH, inventory_payload())
    metrics_sample = client.post(GPU_METRICS_PATH, gpu_metrics_payload())

    assert inventory.status_code == 200
    assert store.get_gpu_inventory_snapshot(CLUSTER, NODE) is not None
    assert metrics_sample.status_code == 202
    assert metrics_sample.json()["spooled"] is True
    assert store.telemetry_spool_stats()["depth"] == 1
    assert not store.has_incomplete_processor_requests(CLUSTER), (
        "expected store.has_incomplete_processor_requests(CLUSTER) to be falsy"
    )


# --- E-5: the replay's bookkeeping is inside the try --------------------------


def _coordinator(store, **spool_overrides) -> ProcessorCoordinator:
    settings = {
        "telemetry_spool_enabled": True,
        "telemetry_spool_workers": 2,
        "telemetry_spool_retry_backoff_seconds": 1.0,
    }
    settings.update(spool_overrides)
    return ProcessorCoordinator(
        store,
        owner_id=OWNER,
        internal_token=TOKEN,
        active_consumers=False,
        spool=ProcessorSpoolSettings(**settings),
        lease=ProcessorLeaseSettings(poll_seconds=0.02),
    )


def _ok_handler(batch_request: dict) -> dict:
    return {
        "results": [
            {"request_id": item["request_id"], "status": 200, "body": {}}
            for item in batch_request["items"]
        ]
    }


def test_a_failed_completion_is_counted_and_releases_the_rows(monkeypatch) -> None:
    """``complete_telemetry_spool`` used to run outside the try (E-5).

    A writer failover landing on the completion left the rows leased for the
    full 60 s lease, counted nothing, and so kept
    ``GpuFaultTelemetrySpoolReplayError`` quiet while the same batch was
    replayed again and again.
    """

    store = build_store()
    spool(store, [make_request(GPU_METRICS_PATH, gpu_metrics_payload(), "req-1")])
    processor = _coordinator(store)
    processor.telemetry_spool_replay_handler = _ok_handler
    at = now()
    claimed = store.claim_telemetry_spool(
        OWNER, now=at, lease_duration=timedelta(seconds=60), limit=10
    )
    real_complete = store.complete_telemetry_spool
    failures = {"count": 0}

    def flaky_complete(items):
        if failures["count"] == 0:
            failures["count"] += 1
            raise ConnectionError("writer failover in progress")
        return real_complete(items)

    monkeypatch.setattr(store, "complete_telemetry_spool", flaky_complete)

    with pytest.raises(ConnectionError):
        processor._replay_telemetry_spool(claimed)

    runtime = processor.metrics_snapshot()["spool"]
    assert runtime["errors"] == 1, runtime
    assert runtime["completed"] == 0, runtime
    # Released with the retry backoff, not left leased for 60 s.
    stats = store.telemetry_spool_stats(now=at + timedelta(seconds=2))
    assert stats["depth"] == 1 and stats["leased"] == 0, stats
    again = store.claim_telemetry_spool(
        OWNER,
        now=at + timedelta(seconds=2),
        lease_duration=timedelta(seconds=60),
        limit=10,
    )
    assert len(again) == 1
    processor._replay_telemetry_spool(again)
    assert store.telemetry_spool_stats()["depth"] == 0
    assert processor.metrics_snapshot()["spool"]["completed"] == 1


def test_a_malformed_handler_result_is_an_error_not_a_crash_without_release() -> None:
    """``int(result.get("status"))`` on a garbage result raised outside the try."""

    store = build_store()
    spool(store, [make_request(GPU_METRICS_PATH, gpu_metrics_payload(), "req-1")])
    processor = _coordinator(store)
    processor.telemetry_spool_replay_handler = lambda _batch: {
        "results": [{"request_id": "req-1", "status": "not-a-number"}]
    }
    at = now()
    claimed = store.claim_telemetry_spool(
        OWNER, now=at, lease_duration=timedelta(seconds=60), limit=10
    )

    with pytest.raises(ValueError):
        processor._replay_telemetry_spool(claimed)

    assert processor.metrics_snapshot()["spool"]["errors"] == 1
    stats = store.telemetry_spool_stats(now=at + timedelta(seconds=2))
    assert stats["leased"] == 0, "the row stayed leased after the crash"


# --- E-6: the abandon branch must not spin ------------------------------------


def test_spool_workers_times_batch_bytes_must_fit_in_flight() -> None:
    """12 workers x 8 MiB against 64 MiB passed validation and reopened F-D8."""

    with pytest.raises(ValueError, match="in-flight"):
        ProcessorSpoolSettings(
            telemetry_spool_enabled=True,
            telemetry_spool_workers=12,
            telemetry_spool_replay_batch_max_bytes=8 * 1024 * 1024,
            telemetry_spool_max_in_flight_bytes=64 * 1024 * 1024,
        )
    # 8 x 8 MiB = 64 MiB, the production shape, still passes.
    ProcessorSpoolSettings(
        telemetry_spool_enabled=True,
        telemetry_spool_workers=8,
        telemetry_spool_replay_batch_max_bytes=8 * 1024 * 1024,
        telemetry_spool_max_in_flight_bytes=64 * 1024 * 1024,
    )


def test_an_abandoned_claim_waits_out_the_poll_interval_instead_of_spinning(
    monkeypatch,
) -> None:
    """Abandon set ``claimed_any`` and re-entered the loop with no sleep.

    The head row is always selected (``row_number=1``), so a row the byte
    budget cannot take is claimed, abandoned and claimed again at whatever
    rate six SQL statements allow (E-6 / F-D8).
    """

    store = build_store()
    oversize = gpu_metrics_payload(batch_id="huge")
    oversize["samples"] = [{"name": "x" * 64, "value": index} for index in range(900)]
    spool(store, [make_request(GPU_METRICS_PATH, oversize, "req-huge")])
    processor = _coordinator(
        store,
        telemetry_spool_workers=2,
        telemetry_spool_replay_batch_max_bytes=64 * 1024,
        telemetry_spool_max_in_flight_bytes=128 * 1024,
    )
    row_bytes = store.claim_telemetry_spool(
        "probe", now=now(), lease_duration=timedelta(seconds=0), limit=1
    )[0].payload_bytes
    assert row_bytes > 64 * 1024, row_bytes
    monkeypatch.setattr(
        processor,
        "_replay_telemetry_spool",
        lambda _items: pytest.fail("an oversize batch must never be replayed"),
    )
    thread = Thread(target=processor.run_telemetry_spool)
    thread.start()
    try:
        time.sleep(0.3)
        abandoned = processor.spool_abandoned_total
    finally:
        processor.stop()
        thread.join(timeout=5)
    assert not thread.is_alive()
    assert abandoned >= 1, "the oversize head row was never claimed"
    # 0.3 s at a 0.02 s poll is at most ~15 rounds; a spin would be hundreds.
    assert abandoned <= 30, f"the abandon branch spun {abandoned} times in 0.3 s"


# --- E-7: stale spooled samples are completed, not replayed -------------------


def test_a_stale_spooled_sample_is_completed_without_replay() -> None:
    """The queue path runs ``_complete_if_stale`` before execution; the spool
    path replayed everything, so a two-hour backlog of inventory snapshots cost
    one transaction and one advisory lock each only to be discarded on
    ``observed_at`` (E-7).
    """

    store = build_store()
    stale = inventory_payload(snapshot_id="old", observed_at=now() - timedelta(hours=2))
    fresh = inventory_payload(snapshot_id="new")
    fresh["node_id"] = "node-b"
    spool(
        store,
        [
            make_request(INVENTORY_PATH, stale, "req-old"),
            make_request(INVENTORY_PATH, fresh, "req-new"),
        ],
    )
    processor = _coordinator(store)
    replayed: list[str] = []

    def handler(batch_request: dict) -> dict:
        replayed.extend(item["request_id"] for item in batch_request["items"])
        return _ok_handler(batch_request)

    processor.telemetry_spool_replay_handler = handler
    claimed = store.claim_telemetry_spool(
        OWNER, now=now(), lease_duration=timedelta(seconds=60), limit=10
    )
    assert len(claimed) == 2

    processor._replay_telemetry_spool(claimed)

    assert replayed == ["req-new"], replayed
    assert store.telemetry_spool_stats()["depth"] == 0
    runtime = processor.metrics_snapshot()["spool"]
    assert runtime["completed"] == 2, runtime
    assert processor.spool_stale_completed_total == 1


# --- E-4: memory and postgres agree on what coalescing does to attempts -------


@pytest.fixture(params=["memory", "postgres"])
def spool_store(request):
    """The in-memory store and, when configured, the real Postgres store.

    The two backends disagreed on whether a coalescing sample resets the
    failure budget, and the test pinning the design intent ran only against
    ``InMemoryStore`` (E-4): on Postgres a node re-sending an unexecutable
    payload every second never reached the drop threshold.
    """

    if request.param == "memory":
        yield build_store()
        return
    from tests.store._postgres_processor_claim_support import (
        POSTGRES_URL,
        postgres_store_instance,
    )

    if not POSTGRES_URL:
        pytest.skip("GPU_FAULT_TEST_POSTGRES_URL is not configured")
    yield from postgres_store_instance()


@pytest.fixture
def postgres_spool_store():
    """Only the Postgres store: the depth projection is Postgres-specific, and
    a memory variant that skips would count against the CAP-005 zero-skip gate."""

    from tests.store._postgres_processor_claim_support import (
        POSTGRES_URL,
        postgres_store_instance,
    )

    if not POSTGRES_URL:
        pytest.skip("GPU_FAULT_TEST_POSTGRES_URL is not configured")
    yield from postgres_store_instance()


def test_a_failed_replay_comes_back_and_then_gives_up_on_every_store(
    spool_store,
) -> None:
    store = spool_store
    spool(store, [make_request(GPU_METRICS_PATH, gpu_metrics_payload(), "req-1")])
    at = now()
    for attempt in range(store.TELEMETRY_SPOOL_MAX_ATTEMPTS):
        claimed = store.claim_telemetry_spool(
            OWNER, now=at, lease_duration=timedelta(seconds=30), limit=10
        )
        assert len(claimed) == 1, attempt
        released, dropped = store.release_telemetry_spool(
            claimed, now=at, backoff=timedelta(seconds=1)
        )
        if attempt < store.TELEMETRY_SPOOL_MAX_ATTEMPTS - 1:
            assert (released, dropped) == (1, 0), attempt
        else:
            assert (released, dropped) == (0, 1), attempt
        at += timedelta(seconds=2)
    assert store.telemetry_spool_stats()["depth"] == 0


def test_a_backoff_is_not_applied_to_a_newer_sample_on_every_store(spool_store) -> None:
    store = spool_store
    spool(
        store,
        [make_request(GPU_METRICS_PATH, gpu_metrics_payload(batch_id="old"), "req-1")],
    )
    at = now()
    claimed = store.claim_telemetry_spool(
        OWNER, now=at, lease_duration=timedelta(seconds=30), limit=10
    )
    spool(
        store,
        [make_request(GPU_METRICS_PATH, gpu_metrics_payload(batch_id="new"), "req-2")],
        now=at,
    )

    assert store.release_telemetry_spool(
        claimed, now=at, backoff=timedelta(seconds=600)
    ) == (0, 0)
    fresh = store.claim_telemetry_spool(
        OWNER, now=at, lease_duration=timedelta(seconds=30), limit=10
    )
    assert [item.payload["batch_id"] for item in fresh] == ["new"]


def test_coalescing_keeps_the_failure_budget_on_every_store(spool_store) -> None:
    """A stream that keeps failing and keeps re-sending is still bounded.

    Postgres reset ``attempts`` to 0 on conflict "because the budget belongs
    to the payload"; the memory store did not. A node whose gpu-metrics
    payload the endpoint cannot execute re-sends every second, so on Postgres
    every claim started from zero and the row was never dropped.
    """

    store = spool_store
    at = now()
    spool(
        store,
        [make_request(GPU_METRICS_PATH, gpu_metrics_payload(batch_id="b1"), "req-1")],
        now=at,
    )
    for _ in range(2):
        claimed = store.claim_telemetry_spool(
            OWNER, now=at, lease_duration=timedelta(seconds=30), limit=10
        )
        assert len(claimed) == 1
        store.release_telemetry_spool(claimed, now=at, backoff=timedelta(0))
    assert claimed[0].attempts == 2

    spool(
        store,
        [make_request(GPU_METRICS_PATH, gpu_metrics_payload(batch_id="b2"), "req-2")],
        now=at,
    )
    again = store.claim_telemetry_spool(
        OWNER, now=at, lease_duration=timedelta(seconds=30), limit=10
    )

    assert [item.payload["batch_id"] for item in again] == ["b2"]
    assert again[0].attempts == 3, "coalescing reset the failure budget"


# --- E-2: the admission projection counts rows, it does not serialise them -----


def test_the_depth_projection_query_never_touches_the_payload() -> None:
    """Every 0.25 s each ingress process ran ``sum(octet_length(payload::text))``
    over the whole spool to decide whether one more row fits (E-2)."""

    from gpu_fault.store.postgres.telemetry_spool import PostgresTelemetrySpoolMixin

    executed: list[str] = []

    class Cursor:
        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return False

        def execute(self, sql, *_params):
            executed.append(" ".join(sql.split()))

        def fetchall(self):
            return [("cluster-a", 3), ("__unscoped__", 1)]

    class Db:
        def cursor(self):
            return Cursor()

    class Store(PostgresTelemetrySpoolMixin):
        _db = Db()
        url = "postgresql://unused"

    store = Store()
    cache = store._projected_telemetry_spool_depths(now=now())

    assert cache["depth"] == 4
    assert cache["by_cluster"] == {"cluster-a": 3, "__unscoped__": 1}
    assert len(executed) == 1
    assert "payload" not in executed[0], executed[0]
    assert "count(*)" in executed[0]
    assert "GROUP BY" in executed[0]
    assert PostgresTelemetrySpoolMixin._TELEMETRY_SPOOL_DEPTH_TTL_SECONDS >= 1.0


def test_the_depth_counts_agree_with_the_full_stats_on_postgres(
    postgres_spool_store,
) -> None:
    store = postgres_spool_store
    spool(
        store,
        [
            make_request(
                GPU_METRICS_PATH,
                gpu_metrics_payload(batch_id=f"b{index}", node_id=f"node-{index}"),
                f"req-{index}",
            )
            for index in range(3)
        ],
    )

    depths = store.telemetry_spool_depths()
    stats = store.telemetry_spool_stats()

    assert depths["depth"] == stats["depth"] == 3
    assert depths["by_cluster"] == stats["by_cluster"]

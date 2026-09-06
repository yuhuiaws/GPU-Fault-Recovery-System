"""The two collector channels that create incidents do so in the ingestion
transaction (F-M1 / P0-31A / P0-37A / P0-37B).

The gpu-metrics persist half used to commit the finding state and the batch
record, and the finish half -- after the commit -- turned the findings into
incidents. A crash or an exception between the two left a CRITICAL finding that
no incident would ever name: the next batch for that node sees the same severity,
``update_gpu_findings`` reports nothing new, and the finding is orphaned for as
long as the fault persists. These cases pin the repaired shape through the HTTP
surface only: the incident exists even when the post-commit half fails, a
same-batch replay cannot double-count, and on Postgres a failed incident write
takes the finding down with it -- per item, not per claim batch.
"""

from __future__ import annotations

import asyncio
import os
from datetime import datetime, timezone

import httpx
import pytest

from gpu_fault.app import ApplicationContext, create_app
from gpu_fault.gpu_metrics import GpuMetricBatch, GpuMetricSample, GpuMetricSource
from gpu_fault.host_health import HostMetricSample, HostTelemetryBatch
from gpu_fault.models import WorkflowStatus, WorkloadState
from gpu_fault.store import InMemoryStore
from tests._builders import asgi_client, gpu_metric_batch, host_telemetry_batch
from tests.store._postgres_processor_claim_support import (
    _truncate,
    postgres_store_instance,
)

NOW = datetime(2026, 9, 6, 9, 0, tzinfo=timezone.utc)
GPU_METRICS_PATH = "/v1/collector-events/gpu-metrics"
HOST_TELEMETRY_PATH = "/v1/collector-events/host-telemetry"
BATCH_PATH = "/v1/internal/processor/telemetry-batch"
TOKEN = "collector-ingestion-transaction-token-" + "x" * 32
REPLAY_TOKEN = "test-processor-replay-secret-" + "r" * 32


def _sample(name: str, value: float) -> GpuMetricSample:
    return GpuMetricSample(
        metric_name=f"DCGM_FI_DEV_{name.upper()}",
        canonical_name=name,
        value=value,
        gpu_index="0",
        gpu_uuid="GPU-a",
        pci_bdf="0000:b9:00",
    )


def _gpu_batch(
    batch_id: str, samples: list[GpuMetricSample], *, node_id: str = "worker-1"
) -> GpuMetricBatch:
    return gpu_metric_batch(
        batch_id,
        NOW,
        GpuMetricSource.DCGM_EXPORTER,
        samples,
        cluster_id="hp-cluster",
        node_id=node_id,
        runtime_profile_version="simulated-v1",
        workload_state=WorkloadState.IDLE,
    )


def _critical_gpu_batch(
    batch_id: str = "gpu-hot", *, node_id: str = "worker-1"
) -> GpuMetricBatch:
    return _gpu_batch(batch_id, [_sample("gpu_temperature_c", 91)], node_id=node_id)


def _critical_host_batch(batch_id: str = "host-link-down") -> HostTelemetryBatch:
    return host_telemetry_batch(
        batch_id,
        NOW,
        [HostMetricSample(name="network_link_down", value=1.0, device="eth0")],
        cluster_id="hp-cluster",
        node_id="worker-1",
        runtime_profile_version="simulated-v1",
        workload_state=WorkloadState.IDLE,
    )


def _post(context: ApplicationContext, path: str, batch) -> httpx.Response:
    async def run() -> httpx.Response:
        async with asgi_client(context) as client:
            return await client.post(path, json=batch.model_dump(mode="json"))

    return asyncio.run(run())


def _post_batch(app, batches: list[GpuMetricBatch]) -> dict[str, dict]:
    """Drive the processor's group-commit path for a list of gpu-metrics items."""

    async def run() -> httpx.Response:
        async with asgi_client(app) as client:
            return await client.post(
                BATCH_PATH,
                json={
                    "items": [
                        {
                            "request_id": f"req-{batch.batch_id}",
                            "path": GPU_METRICS_PATH,
                            "payload": batch.model_dump(mode="json"),
                        }
                        for batch in batches
                    ]
                },
                headers={"X-GPU-Fault-Processor-Replay": REPLAY_TOKEN},
            )

    response = asyncio.run(run())
    assert response.status_code == 200, response.text
    return {item["request_id"]: item for item in response.json()["results"]}


def _batch_app(monkeypatch, store):
    # The batch endpoint only exists with a processor and authenticates with
    # the processor's internal replay token (see test_processor_telemetry_batch).
    monkeypatch.setenv("GPU_FAULT_PROCESSOR_MODE", "active-active")
    monkeypatch.setenv("POD_UID", "pod-collector-ingestion-transaction")
    context = ApplicationContext(store=store, execution_token=TOKEN)
    return context, create_app(context)


def _finding_key(batch: GpuMetricBatch, canonical_name: str):
    return (batch.cluster_id, batch.node_id, "GPU-a", canonical_name)


def _fail_worker_2(context: ApplicationContext, monkeypatch) -> None:
    real_ingest = context.orchestrator.ingest_node_health

    def fail_worker_2(finding, **kwargs):
        if finding.node_id == "worker-2":
            raise RuntimeError("worker-2 exploded")
        return real_ingest(finding, **kwargs)

    monkeypatch.setattr(context.orchestrator, "ingest_node_health", fail_worker_2)


@pytest.fixture
def context() -> ApplicationContext:
    return ApplicationContext(store=InMemoryStore())


def test_critical_gpu_finding_always_yields_an_incident_or_a_counted_reason(
    context: ApplicationContext, monkeypatch
) -> None:
    # The post-commit half fails: the incident must already be there.
    def wake_fails() -> None:
        raise RuntimeError("wake failed")

    monkeypatch.setattr(context.dispatcher, "wake", wake_fails)
    with pytest.raises(RuntimeError, match="wake failed"):
        _post(context, GPU_METRICS_PATH, _critical_gpu_batch())

    state = context.store.get_gpu_finding_state(
        _finding_key(_critical_gpu_batch(), "gpu_temperature_c")
    )
    assert state is not None and state.finding is not None, "finding persisted"
    assert state.finding.severity == "CRITICAL"
    incident = context.store.get_incident_by_event(f"gpu-{state.finding.finding_id}")
    assert incident is not None, "the finding must not be committed without it"
    assert context.store.get_workflow(incident.workflow_request_id).status in {
        WorkflowStatus.PENDING,
        WorkflowStatus.SAFETY_PENDING,
    }
    (marker,) = context.store.list_markers()
    assert marker.incident_id == incident.incident_id

    # A component suppressed in favour of its composite is the one legitimate
    # way a new finding ends without an incident of its own; it is counted.
    monkeypatch.undo()
    response = _post(
        context,
        GPU_METRICS_PATH,
        _gpu_batch(
            "gpu-memory",
            [_sample("ecc_dbe_volatile_total", 1), _sample("row_remap_failure", 1)],
        ),
    )
    assert response.status_code == 200
    body = response.json()
    assert body["suppressed_finding_ids"], "the recipe must suppress a component"
    assert context.gpu_metrics.findings_without_incident == {
        "suppressed_by_composite": len(body["suppressed_finding_ids"])
    }
    for finding in body["new_findings"]:
        assert (
            context.store.get_incident_by_event(f"gpu-{finding['finding_id']}")
            is not None
        ), finding["finding_id"]


def test_critical_host_finding_is_an_incident_before_the_finish_half_runs(
    context: ApplicationContext, monkeypatch
) -> None:
    def wake_fails() -> None:
        raise RuntimeError("wake failed")

    monkeypatch.setattr(context.dispatcher, "wake", wake_fails)
    batch = _critical_host_batch()

    with pytest.raises(RuntimeError, match="wake failed"):
        _post(context, HOST_TELEMETRY_PATH, batch)

    incident = context.store.get_incident_by_event(
        f"{batch.batch_id}-network_link_down-eth0"
    )
    assert incident is not None, "host telemetry must create its incident in persist"
    assert len(context.store.list_workflows(limit=10)) == 1


def test_replayed_observation_does_not_double_count(
    context: ApplicationContext,
) -> None:
    batch = _critical_gpu_batch()

    first = _post(context, GPU_METRICS_PATH, batch)
    second = _post(context, GPU_METRICS_PATH, batch)

    assert first.status_code == 200 and second.status_code == 200
    assert first.json()["duplicate"] is False
    assert second.json()["duplicate"] is True, "the same batch id must be recognised"
    assert [item["finding_id"] for item in second.json()["new_findings"]] == [
        item["finding_id"] for item in first.json()["new_findings"]
    ]
    state = context.store.get_gpu_finding_state(
        _finding_key(batch, "gpu_temperature_c")
    )
    assert state is not None, "the finding state must exist"
    assert state.consecutive_breaches == 1
    assert len(context.store.list_workflows(limit=10)) == 1
    assert len(context.store.list_markers()) == 1


def test_a_failed_incident_write_is_recovered_by_replaying_the_batch(
    context: ApplicationContext, monkeypatch
) -> None:
    """Without a transaction (memory / SQLite) the finding is already
    persisted when the incident write fails; the collector's retry of the
    same batch must still reach the incident instead of finding nothing new."""

    batch = _critical_gpu_batch()
    real_ingest = context.orchestrator.ingest_node_health
    calls: list[str] = []

    def fail_once(finding, **kwargs):
        calls.append(finding.event_id)
        if len(calls) == 1:
            raise RuntimeError("store unavailable")
        return real_ingest(finding, **kwargs)

    monkeypatch.setattr(context.orchestrator, "ingest_node_health", fail_once)

    with pytest.raises(RuntimeError, match="store unavailable"):
        _post(context, GPU_METRICS_PATH, batch)
    response = _post(context, GPU_METRICS_PATH, batch)

    assert response.status_code == 200
    body = response.json()
    assert body["duplicate"] is True, "the retry is the same batch"
    (finding,) = body["new_findings"]
    assert (
        context.store.get_incident_by_event(f"gpu-{finding['finding_id']}") is not None
    ), "the replay must create the incident the first attempt lost"


def test_group_path_isolates_a_failed_item_and_keeps_its_siblings_incidents(
    monkeypatch,
) -> None:
    context, app = _batch_app(monkeypatch, InMemoryStore())
    _fail_worker_2(context, monkeypatch)

    results = _post_batch(
        app,
        [
            _critical_gpu_batch("gpu-1", node_id="worker-1"),
            _critical_gpu_batch("gpu-2", node_id="worker-2"),
            _critical_gpu_batch("gpu-3", node_id="worker-3"),
        ],
    )

    assert results["req-gpu-1"]["status"] == 200
    assert results["req-gpu-2"]["status"] == 500
    assert results["req-gpu-3"]["status"] == 200
    for request_id in ("req-gpu-1", "req-gpu-3"):
        finding_id = results[request_id]["body"]["new_findings"][0]["finding_id"]
        assert context.store.get_incident_by_event(f"gpu-{finding_id}") is not None, (
            request_id
        )


@pytest.mark.skipif(
    not os.getenv("GPU_FAULT_TEST_POSTGRES_URL"),
    reason="GPU_FAULT_TEST_POSTGRES_URL is not configured",
)
class TestPostgresIngestionTransaction:
    @pytest.fixture
    def store(self):
        for instance in postgres_store_instance():
            yield instance
        _truncate()

    def test_collector_channels_write_incident_and_link_atomically(
        self, store, monkeypatch
    ) -> None:
        context = ApplicationContext(store=store)
        batch = _critical_gpu_batch()
        key = _finding_key(batch, "gpu_temperature_c")

        def explode(_finding, **_kwargs):
            raise RuntimeError("incident write failed")

        monkeypatch.setattr(context.orchestrator, "ingest_node_health", explode)
        with pytest.raises(RuntimeError, match="incident write failed"):
            _post(context, GPU_METRICS_PATH, batch)

        assert (
            store.get_gpu_metrics_batch(
                (batch.cluster_id, batch.node_id, batch.batch_id)
            )
            is None
        ), "a failed incident write must roll the batch record back"
        assert store.get_gpu_finding_state(key) is None, (
            "a failed incident write must roll the finding state back"
        )
        assert store.list_markers() == []
        assert store.list_collector_statuses(batch.cluster_id) == []

        monkeypatch.undo()
        response = _post(context, GPU_METRICS_PATH, batch)

        assert response.status_code == 200
        body = response.json()
        assert body["duplicate"] is False, "nothing of the failed attempt survived"
        (finding,) = body["new_findings"]
        assert (
            store.get_incident_by_event(f"gpu-{finding['finding_id']}") is not None
        ), "the retry creates finding and incident together"
        state = store.get_gpu_finding_state(key)
        assert state is not None and state.finding is not None, "finding persisted"
        assert state.finding.finding_id == finding["finding_id"]

    def test_group_commit_rolls_back_only_the_failed_items_finding(
        self, store, monkeypatch
    ) -> None:
        context, app = _batch_app(monkeypatch, store)
        _fail_worker_2(context, monkeypatch)
        good = _critical_gpu_batch("gpu-1", node_id="worker-1")
        bad = _critical_gpu_batch("gpu-2", node_id="worker-2")

        results = _post_batch(app, [good, bad])

        assert results["req-gpu-1"]["status"] == 200
        assert results["req-gpu-2"]["status"] == 500
        assert (
            store.get_gpu_finding_state(_finding_key(good, "gpu_temperature_c"))
            is not None
        ), "the sibling's finding is committed"
        assert (
            store.get_gpu_finding_state(_finding_key(bad, "gpu_temperature_c")) is None
        ), "the failed item's finding must not outlive its incident"
        finding_id = results["req-gpu-1"]["body"]["new_findings"][0]["finding_id"]
        assert store.get_incident_by_event(f"gpu-{finding_id}") is not None

    def test_host_telemetry_claim_rolls_back_with_its_incident(
        self, store, monkeypatch
    ) -> None:
        """P0-38B: the ``notified`` latch must not outlive the notification."""

        context = ApplicationContext(store=store)
        batch = _critical_host_batch()

        def explode(_finding, **_kwargs):
            raise RuntimeError("incident write failed")

        monkeypatch.setattr(context.orchestrator, "ingest_node_health", explode)
        with pytest.raises(RuntimeError, match="incident write failed"):
            _post(context, HOST_TELEMETRY_PATH, batch)

        monkeypatch.undo()
        response = _post(context, HOST_TELEMETRY_PATH, batch)

        assert response.status_code == 200
        assert len(response.json()["incident_ids"]) == 1, (
            "the replay must re-claim the signal the rollback released"
        )
        assert (
            store.get_incident_by_event(f"{batch.batch_id}-network_link_down-eth0")
            is not None
        )

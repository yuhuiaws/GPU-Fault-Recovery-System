"""The attempt-scoped low-utilisation window through the real ingest route.

Two API replicas (two application contexts over one store) take turns with one
node's host-telemetry batches, as the load balancer spreads them. The window and
its latch are store rows, so they accumulate across replicas; the finish half
latches the attempt's own key once the incident committed, so the attempt stays
quiet afterwards; and a job that takes over the node starts its own window
instead of inheriting the old attempt's low time or its latch (batch 140).
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import httpx

from gpu_fault.app import ApplicationContext
from gpu_fault.host_health import HostMetricSample, HostTelemetryBatch
from gpu_fault.models import WorkloadState
from gpu_fault.store import InMemoryStore
from gpu_fault.watcher import WorkloadPhase
from tests._builders import (
    asgi_client,
    attempt_observation,
    container_observation,
    host_telemetry_batch,
)

NOW = datetime(2026, 9, 6, 9, 0, tzinfo=timezone.utc)
HOST_TELEMETRY_PATH = "/v1/collector-events/host-telemetry"
WORKLOAD = "training/job/shared"
A = ("job-a", "attempt-a")
B = ("job-b", "attempt-b")


def _observe(store, attempt, offset, *, ended=False) -> None:
    job_id, attempt_id = attempt
    store.save_attempt_observation(
        attempt_observation(
            job_id,
            attempt_id,
            NOW + timedelta(seconds=offset),
            cluster_id="hp-cluster",
            workload_phase=WorkloadPhase.SUCCEEDED if ended else WorkloadPhase.RUNNING,
            workload_ids=[WORKLOAD],
            containers=[
                container_observation(
                    f"pod-{attempt_id}",
                    f"worker-{attempt_id}",
                    0,
                    "worker-1",
                    gpu_uuids=["GPU-0"],
                    terminated=ended,
                    exit_code=0 if ended else None,
                )
            ],
        )
    )


def _batch(offset: int) -> HostTelemetryBatch:
    observed_at = NOW + timedelta(seconds=offset)
    return host_telemetry_batch(
        f"host-{offset}",
        observed_at,
        [
            HostMetricSample(
                name="host_gpu_utilization_percent",
                value=0.0,
                unit="percent",
                device="GPU-0",
            )
        ],
        cluster_id="hp-cluster",
        node_id="worker-1",
        runtime_profile_version="simulated-v1",
        workload_state=WorkloadState.ACTIVE,
        affected_workload_ids=[WORKLOAD],
        received_at=observed_at,
    )


def _post(context: ApplicationContext, batch: HostTelemetryBatch) -> httpx.Response:
    async def run() -> httpx.Response:
        async with asgi_client(context) as client:
            return await client.post(
                HOST_TELEMETRY_PATH, json=batch.model_dump(mode="json")
            )

    return asyncio.run(run())


def test_replicas_share_each_attempts_window_and_latch() -> None:
    store = InMemoryStore()
    replicas = (ApplicationContext(store=store), ApplicationContext(store=store))
    owners: dict[int, list[tuple[str, str]]] = {}
    timeline = [(offset, A) for offset in range(0, 421, 60)]
    timeline += [(offset, B) for offset in range(440, 801, 60)]
    for index, (offset, attempt) in enumerate(timeline):
        if attempt is B and offset == 440:
            _observe(store, A, offset - 10, ended=True)
        _observe(store, attempt, offset)
        response = _post(replicas[index % 2], _batch(offset))
        assert response.status_code == 200, response.text
        owners[offset] = [
            (item["job_id"], item["attempt_id"]) for item in response.json()["findings"]
        ]

    assert {offset: found for offset, found in owners.items() if found} == {
        300: [A],
        740: [B],
    }, owners

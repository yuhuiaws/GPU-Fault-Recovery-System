"""The live GF-GENERAL-LIVE-006 timeline replayed through host-telemetry
ingestion (batch 160): a notified device signal whose device vanished is aged
out while the node keeps posting, and the same-named device's next sustained
crossing opens a new incident.

Before: ``cluster/node/disk_io_await_ms/dm-0`` stayed active+notified for two
hours after the dm-delay device was torn down, and the new dm-0 that crossed
the threshold at 17:04Z minted nothing.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import httpx
import pytest

from gpu_fault.app import ApplicationContext
from gpu_fault.host_device_signals import DEVICE_AGE_OUT_SUMMARIES_ENV
from gpu_fault.host_health import HostMetricSample, HostTelemetryBatch
from gpu_fault.models import WorkloadState
from gpu_fault.store import InMemoryStore
from tests._builders import asgi_client, host_telemetry_batch

T0 = datetime(2026, 10, 25, 15, 1, tzinfo=timezone.utc)
HOST_TELEMETRY_PATH = "/v1/collector-events/host-telemetry"
SIGNAL_KEY = "gpu-eks-cluster/gpu-node-1/disk_io_await_ms/dm-0"
SUMMARY = 300


def _batch(
    batch_id: str, seconds: float, *, await_ms: float | None
) -> HostTelemetryBatch:
    observed_at = T0 + timedelta(seconds=seconds)
    samples = [HostMetricSample(name="cpu_usage_percent", value=12.0, unit="percent")]
    if await_ms is not None:
        samples.append(
            HostMetricSample(
                name="disk_io_await_ms",
                value=await_ms,
                unit="milliseconds",
                device="dm-0",
            )
        )
    return host_telemetry_batch(
        batch_id,
        observed_at,
        samples,
        cluster_id="gpu-eks-cluster",
        node_id="gpu-node-1",
        received_at=observed_at,
        runtime_profile_version="simulated-v1",
        workload_state=WorkloadState.IDLE,
        edge_filter_reasons=["health-summary"] if await_ms is None else ["threshold"],
    )


def _post(context: ApplicationContext, batch: HostTelemetryBatch) -> httpx.Response:
    async def run() -> httpx.Response:
        async with asgi_client(context) as client:
            return await client.post(
                HOST_TELEMETRY_PATH, json=batch.model_dump(mode="json")
            )

    return asyncio.run(run())


def _incidents(context: ApplicationContext, batch: HostTelemetryBatch) -> list[str]:
    response = _post(context, batch)
    assert response.status_code == 200, response.text
    return list(response.json()["incident_ids"])


@pytest.fixture
def context(monkeypatch) -> ApplicationContext:
    monkeypatch.delenv(DEVICE_AGE_OUT_SUMMARIES_ENV, raising=False)
    monkeypatch.delenv("GPU_FAULT_HOST_HEALTH_SUMMARY_SECONDS", raising=False)
    return ApplicationContext(store=InMemoryStore())


def test_vanished_device_is_aged_out_and_the_same_name_opens_a_new_incident(
    context: ApplicationContext,
) -> None:
    # 15:01Z the dm-delay device crosses 100 ms; 15:03Z sustained: one incident.
    assert _incidents(context, _batch("host-1", 0, await_ms=250.0)) == []
    first = _incidents(context, _batch("host-2", 120, await_ms=250.0))
    assert len(first) == 1
    latched = context.store.get_health_signal_state(SIGNAL_KEY)
    assert latched is not None and latched.active and latched.notified is True

    # The device is torn down; the node's health summaries carry no dm-0.
    at = 120
    for number in range(3, 6):
        at += SUMMARY
        assert _incidents(context, _batch(f"host-{number}", at, await_ms=None)) == []
    aged = context.store.get_health_signal_state(SIGNAL_KEY)
    assert aged is not None and aged.active is False and aged.notified is False

    # 17:04Z a new dm-0 crosses again, and is sustained: a second incident.
    assert _incidents(context, _batch("host-6", 7380, await_ms=300.0)) == []
    second = _incidents(context, _batch("host-7", 7500, await_ms=300.0))
    assert len(second) == 1 and second != first
    assert (
        context.store.get_incident_by_event("host-7-disk_io_await_ms-dm-0") is not None
    )


def test_a_device_that_stays_slow_is_still_one_incident(
    context: ApplicationContext,
) -> None:
    assert _incidents(context, _batch("host-1", 0, await_ms=250.0)) == []
    assert len(_incidents(context, _batch("host-2", 120, await_ms=250.0))) == 1
    at = 120
    for number in range(3, 30):  # two hours of summaries, dm-0 still at 250 ms
        at += SUMMARY
        assert _incidents(context, _batch(f"host-{number}", at, await_ms=250.0)) == []
    state = context.store.get_health_signal_state(SIGNAL_KEY)
    assert state is not None and state.active and state.notified is True

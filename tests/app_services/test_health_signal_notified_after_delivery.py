"""Host-telemetry ingestion latches a health signal only after delivery (P0-38B).

On a backend whose ingestion transaction is a no-op the claim and the incident
write are two separate commits. If the claim latched ``notified`` and the
incident write then failed, the CRITICAL signal was durable-marked as handled
and never emitted again for as long as the fault lasted. The latch is now
written by the finish half, after the commit that carries the incident and its
advisory notification.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import httpx
import pytest

from gpu_fault.app import ApplicationContext
from gpu_fault.host_health import HostMetricSample, HostTelemetryBatch
from gpu_fault.models import WorkloadState
from gpu_fault.store import InMemoryStore
from tests._builders import asgi_client, host_telemetry_batch

NOW = datetime(2026, 9, 6, 9, 0, tzinfo=timezone.utc)
HOST_TELEMETRY_PATH = "/v1/collector-events/host-telemetry"
SIGNAL_KEY = "hp-cluster/worker-1/network_link_down/eth0"


def _batch(batch_id: str, observed_at: datetime) -> HostTelemetryBatch:
    return host_telemetry_batch(
        batch_id,
        observed_at,
        [HostMetricSample(name="network_link_down", value=1.0, device="eth0")],
        cluster_id="hp-cluster",
        node_id="worker-1",
        runtime_profile_version="simulated-v1",
        workload_state=WorkloadState.IDLE,
    )


def _post(context: ApplicationContext, batch: HostTelemetryBatch) -> httpx.Response:
    async def run() -> httpx.Response:
        async with asgi_client(context) as client:
            return await client.post(
                HOST_TELEMETRY_PATH, json=batch.model_dump(mode="json")
            )

    return asyncio.run(run())


@pytest.fixture
def context() -> ApplicationContext:
    return ApplicationContext(store=InMemoryStore())


def test_notified_flag_is_written_after_delivery_not_before(
    context: ApplicationContext, monkeypatch
) -> None:
    def explode(_finding, **_kwargs):
        raise RuntimeError("incident write failed")

    monkeypatch.setattr(context.orchestrator, "ingest_node_health", explode)
    with pytest.raises(RuntimeError, match="incident write failed"):
        _post(context, _batch("host-1", NOW))

    state = context.store.get_health_signal_state(SIGNAL_KEY)
    assert state is not None, "the claim persisted the signal state"
    assert state.notified is not True, "a failed delivery must not latch the signal"

    # The next sample of the same, still-active fault emits again and, with
    # the incident write healthy, creates the incident.
    monkeypatch.undo()
    response = _post(context, _batch("host-2", NOW + timedelta(seconds=30)))

    assert response.status_code == 200
    assert len(response.json()["incident_ids"]) == 1, (
        "the unlatched signal must emit again on the next evaluation"
    )
    assert (
        context.store.get_incident_by_event("host-2-network_link_down-eth0") is not None
    )
    latched = context.store.get_health_signal_state(SIGNAL_KEY)
    assert latched is not None, "the signal state must still exist"
    assert latched.notified is True, "a successful delivery flips the latch"

    # And once latched, the still-active signal stays quiet.
    third = _post(context, _batch("host-3", NOW + timedelta(seconds=60)))
    assert third.status_code == 200
    assert third.json()["incident_ids"] == []


def test_a_failing_wake_does_not_unlatch_a_delivered_signal(
    context: ApplicationContext, monkeypatch
) -> None:
    """The commit is the delivery; the dispatcher wake after it is not."""

    def wake_fails() -> None:
        raise RuntimeError("wake failed")

    monkeypatch.setattr(context.dispatcher, "wake", wake_fails)
    with pytest.raises(RuntimeError, match="wake failed"):
        _post(context, _batch("host-1", NOW))

    assert (
        context.store.get_incident_by_event("host-1-network_link_down-eth0") is not None
    ), "the incident was committed before the wake"
    state = context.store.get_health_signal_state(SIGNAL_KEY)
    assert state is not None, "the signal state must exist"
    assert state.notified is True


def test_the_finish_half_derives_the_key_the_sustained_rule_claimed() -> None:
    """A sustained-rule finding carries its rule id; the key the finish half
    latches must be the one ``evaluate_metrics`` claimed, or the latch is a
    no-op and the warning repeats on every sample."""

    from gpu_fault.store.shared.health_signals import finding_health_signal_key

    context = ApplicationContext(store=InMemoryStore())
    policy = context.node_health
    duration = policy.memory_pressure_duration_seconds

    def low_memory(batch_id: str, observed_at: datetime) -> HostTelemetryBatch:
        return host_telemetry_batch(
            batch_id,
            observed_at,
            [HostMetricSample(name="memory_available_percent", value=1.0)],
            cluster_id="hp-cluster",
            node_id="worker-1",
            runtime_profile_version="simulated-v1",
            workload_state=WorkloadState.IDLE,
            received_at=observed_at,
        )

    assert policy.evaluate_metrics(low_memory("mem-1", NOW)) == []
    later = NOW + timedelta(seconds=duration + 1)
    (finding,) = policy.evaluate_metrics(low_memory("mem-2", later))

    key = finding_health_signal_key(finding)
    state = context.store.get_health_signal_state(key)
    assert state is not None, f"no claimed signal state under the derived key {key}"
    assert state.active is True
    assert state.notified is not True

    context.store.mark_health_signal_notified(key, notified_at=later)
    assert (
        policy.evaluate_metrics(low_memory("mem-3", later + timedelta(seconds=30)))
        == []
    )

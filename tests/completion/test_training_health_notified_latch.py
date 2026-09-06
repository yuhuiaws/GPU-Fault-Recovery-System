"""The training-progress path latches ``notified`` after delivery, too (P0-38B).

Batch 3 moved the host-telemetry latch out of the claim and into the
deliverer; the single-signal claim the training path uses kept latching on
emit. A finding whose incident write failed was then never re-emitted for as
long as the fault lasted. The claim now leaves the latch alone on every path
and ``TrainingHealthService.mark_notified`` sets it once the findings have
been persisted.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import pytest

from gpu_fault.training_health import (
    TrainingHealthService,
    TrainingProgressHeartbeat,
    training_health_signal_key,
)
from tests._builders import (
    asgi_client,
    attempt_observation,
    build_context,
    build_store,
    container_observation,
)

NOW = datetime(2026, 9, 6, 10, 0, tzinfo=timezone.utc)


def _observation():
    return attempt_observation(
        "training-job",
        "attempt-a",
        NOW,
        expected_critical_ranks=1,
        workload_ids=["training/job/training-job"],
        containers=[
            container_observation("pod-0", "worker-0", 0, "node-0", gpu_uuids=["GPU-0"])
        ],
    )


def _nonfinite(observed_at: datetime) -> TrainingProgressHeartbeat:
    return TrainingProgressHeartbeat(
        cluster_id="cluster-a",
        attempt_id="attempt-a",
        rank=0,
        observed_at=observed_at,
        node_id="node-0",
        step=100,
        samples_per_second=100,
        loss=1.0,
        numerical_error=True,
    )


def test_a_finding_whose_delivery_failed_is_emitted_again() -> None:
    store = build_store()
    store.save_attempt_observation(_observation())
    service = TrainingHealthService(store)

    first = service.ingest(_nonfinite(NOW))
    # Nothing marked it: the incident write raised before the latch.
    second = service.ingest(_nonfinite(NOW + timedelta(seconds=30)))

    assert [item.metric_name for item in first.findings] == ["training_nonfinite-loss"]
    state = store.get_health_signal_state(training_health_signal_key(first.findings[0]))
    assert state is not None, "the claim must persist the signal state"
    assert state.notified is not True, "the claim alone must not latch"
    assert [item.metric_name for item in second.findings] == [
        "training_nonfinite-loss"
    ], "an undelivered finding is owed again on the next heartbeat"


def test_a_delivered_finding_latches_and_goes_quiet() -> None:
    store = build_store()
    store.save_attempt_observation(_observation())
    service = TrainingHealthService(store)

    first = service.ingest(_nonfinite(NOW))
    service.mark_notified(first.findings)
    second = service.ingest(_nonfinite(NOW + timedelta(seconds=30)))

    state = store.get_health_signal_state(training_health_signal_key(first.findings[0]))
    assert state is not None, "marking must keep the signal state"
    assert state.notified is True
    assert second.findings == [], "a delivered finding is not emitted again"


def test_marking_findings_of_a_signal_that_cleared_is_a_no_op() -> None:
    store = build_store()
    store.save_attempt_observation(_observation())
    service = TrainingHealthService(store)

    first = service.ingest(_nonfinite(NOW))
    healthy = _nonfinite(NOW + timedelta(seconds=30)).model_copy(
        update={"numerical_error": False}
    )
    service.ingest(healthy)
    service.mark_notified(first.findings)

    state = store.get_health_signal_state(training_health_signal_key(first.findings[0]))
    assert state is not None, "the inactive state must survive the no-op"
    assert state.active is False
    assert state.notified is False


def test_the_training_progress_route_latches_after_the_incident_is_written() -> None:
    context = build_context()

    async def scenario() -> None:
        async with asgi_client(context) as client:
            mapped = await client.post(
                "/v1/workload-observations", json=_observation().model_dump(mode="json")
            )
            response = await client.post(
                "/v1/training-progress", json=_nonfinite(NOW).model_dump(mode="json")
            )
        assert mapped.status_code == 200
        assert response.status_code == 200
        finding = response.json()["findings"][0]
        assert context.store.get_incident_by_event(finding["event_id"]) is not None
        state = context.store.get_health_signal_state(
            "cluster-a/attempt-a/rank-0/training-nonfinite-loss"
        )
        assert state is not None, "the route must persist the signal state"
        assert state.notified is True

    asyncio.run(scenario())


def test_the_training_progress_route_leaves_the_latch_open_when_the_write_fails(
    monkeypatch,
) -> None:
    context = build_context()

    def refuse(_finding):
        raise RuntimeError("incident write refused")

    monkeypatch.setattr(context.orchestrator, "ingest_node_health", refuse)

    async def scenario() -> None:
        async with asgi_client(context) as client:
            mapped = await client.post(
                "/v1/workload-observations", json=_observation().model_dump(mode="json")
            )
            assert mapped.status_code == 200
            with pytest.raises(RuntimeError, match="incident write refused"):
                await client.post(
                    "/v1/training-progress",
                    json=_nonfinite(NOW).model_dump(mode="json"),
                )
        state = context.store.get_health_signal_state(
            "cluster-a/attempt-a/rank-0/training-nonfinite-loss"
        )
        assert state is not None, "the claim must persist the signal state"
        assert state.notified is not True, "a failed write must not latch"

    asyncio.run(scenario())

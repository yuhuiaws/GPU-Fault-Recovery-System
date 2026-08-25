from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

from gpu_fault.store import SqliteStore
from gpu_fault.telemetry import WorkloadTopologyService
from gpu_fault.training_health import (
    TrainingHealthPolicy,
    TrainingHealthService,
    TrainingProgressHeartbeat,
)
from gpu_fault.watcher import AttemptObservation
from tests._builders import (
    asgi_client,
    attempt_observation,
    build_context,
    build_store,
    container_observation,
)

NOW = datetime(2026, 7, 21, 10, 0, tzinfo=timezone.utc)


def observation(*, observed_at: datetime = NOW) -> AttemptObservation:
    return attempt_observation(
        "training-job",
        "attempt-a",
        observed_at,
        expected_critical_ranks=2,
        workload_ids=["training/job/training-job"],
        containers=[
            container_observation(
                f"pod-{rank}",
                f"worker-{rank}",
                rank,
                f"node-{rank}",
                gpu_uuids=[f"GPU-{rank}"],
            )
            for rank in range(2)
        ],
    )


def heartbeat(
    rank: int,
    *,
    observed_at: datetime = NOW,
    step: int = 100,
    loss: float = 1.0,
    numerical_error: bool = False,
) -> TrainingProgressHeartbeat:
    return TrainingProgressHeartbeat(
        cluster_id="cluster-a",
        attempt_id="attempt-a",
        rank=rank,
        observed_at=observed_at,
        node_id=f"node-{rank}",
        step=step,
        samples_per_second=100,
        loss=loss,
        numerical_error=numerical_error,
    )


def test_topology_resolves_rank_workload_and_gpu() -> None:
    store = build_store()
    topology = WorkloadTopologyService(store)
    topology.observe(observation())

    context = topology.resolve("cluster-a", "node-1", NOW + timedelta(seconds=5))

    assert context.workload_state == "ACTIVE"
    assert context.workload_ids == ["training/job/training-job"]
    assert context.attempt_ids == ["attempt-a"]
    assert context.ranks == [1]
    assert context.gpu_uuids == ["GPU-1"]
    assert context.runtime_profile_version == "simulated-v1"


def test_topology_survives_sqlite_restart(tmp_path) -> None:
    path = tmp_path / "topology.db"
    first = SqliteStore(str(path))
    first.save_attempt_observation(observation())
    first.close()

    second = SqliteStore(str(path))
    try:
        context = WorkloadTopologyService(second).resolve(
            "cluster-a", "node-0", NOW + timedelta(seconds=5)
        )
        assert context.attempt_ids == ["attempt-a"]
        assert context.ranks == [0]
    finally:
        second.close()


def test_training_health_detects_straggler_and_hang() -> None:
    store = build_store()
    store.save_attempt_observation(observation())
    service = TrainingHealthService(
        store,
        TrainingHealthPolicy(
            heartbeat_timeout_seconds=120,
            startup_grace_seconds=300,
            max_step_lag=20,
            min_throughput_ratio=0.5,
        ),
    )
    service.ingest(heartbeat(0, step=100))
    service.ingest(heartbeat(1, step=40))

    straggler = service.scan("cluster-a", now=NOW + timedelta(seconds=10))
    hung = service.scan("cluster-a", now=NOW + timedelta(seconds=121))

    assert {item.metric_name for item in straggler.findings} == {"training_straggler"}
    assert straggler.findings[0].node_id == "node-1"
    assert {item.metric_name for item in hung.findings} == {"training_hang"}
    assert {item.node_id for item in hung.findings} == {"node-0", "node-1"}


def test_training_health_detects_stalled_step_with_fresh_heartbeat() -> None:
    store = build_store()
    store.save_attempt_observation(observation())
    service = TrainingHealthService(
        store,
        TrainingHealthPolicy(
            heartbeat_timeout_seconds=120,
            startup_grace_seconds=300,
            max_step_lag=20,
            min_throughput_ratio=0.5,
        ),
    )
    service.ingest(heartbeat(0, step=10))
    service.ingest(heartbeat(0, step=10, observed_at=NOW + timedelta(seconds=60)))

    result = service.scan("cluster-a", now=NOW + timedelta(seconds=121))

    rank_zero = next(item for item in result.findings if item.node_id == "node-0")
    assert rank_zero.reason == "training step stopped advancing"


def test_training_progress_api_detects_nonfinite_loss() -> None:
    context = build_context()

    async def scenario() -> None:
        async with asgi_client(context) as client:
            mapped = await client.post(
                "/v1/workload-observations", json=observation().model_dump(mode="json")
            )
            response = await client.post(
                "/v1/training-progress",
                json=heartbeat(0, numerical_error=True).model_dump(mode="json"),
            )

        assert mapped.status_code == 200
        assert response.status_code == 200
        finding = response.json()["findings"][0]
        assert finding["metric_name"] == "training_nonfinite-loss"
        assert finding["affected_workload_ids"] == ["training/job/training-job"]
        incident = context.store.get_incident_by_event(finding["event_id"])
        assert incident is not None

    asyncio.run(scenario())

"""Pending membership can collect readiness facts without dispatch authority."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import pytest

from gpu_fault.channel_registry import (
    ATTEMPT_COVERAGE_PATH,
    COLLECTOR_HEALTH_PATH,
    WORKLOAD_OBSERVATIONS_PATH,
)
from gpu_fault.regional import RegionalClusterLifecycle
from gpu_fault.telemetry import (
    CollectorHealthSummary,
    CollectorKind,
    WorkloadCoverageHeartbeat,
)
from tests._builders import (
    asgi_client,
    attempt_observation,
    build_context,
    container_observation,
)
from tests.regional._regional_support import (
    TOKEN_A,
    enqueue_remote_command,
    registration,
)


def test_pending_cluster_persists_readiness_observations_but_cannot_claim() -> None:
    context = build_context()
    context.regional_mode = True
    context.store.save_regional_cluster(
        registration("cluster-a", TOKEN_A).model_copy(
            update={"lifecycle_state": RegionalClusterLifecycle.PENDING}
        )
    )
    queued = enqueue_remote_command(context.store, "pending-readiness")
    now = datetime.now(timezone.utc)
    health = CollectorHealthSummary(
        summary_id="pending-health",
        cluster_id="cluster-a",
        node_id="node-a",
        collector=CollectorKind.NVIDIA_KERNEL,
        observed_at=now,
        edge_filter_reasons=["health-summary"],
    )
    observation = attempt_observation(
        "pending-job",
        "pending-attempt",
        now,
        containers=[container_observation("pending-pod", "pending-job-0", 0, "node-a")],
        workload_ids=["training/job/pending-job"],
    )
    coverage = WorkloadCoverageHeartbeat(
        cluster_id="cluster-a",
        observed_at=now,
        watched_pods=1,
        watched_attempts=1,
        resource_version="pending-readiness-rv",
        watcher_instance="pending-watcher",
    )
    headers = {
        "Authorization": f"Bearer {TOKEN_A}",
        "X-GPU-Fault-Cluster-ID": "cluster-a",
    }

    async def scenario() -> None:
        async with asgi_client(context) as client:
            for path, payload, status in (
                (COLLECTOR_HEALTH_PATH, health, 200),
                (WORKLOAD_OBSERVATIONS_PATH, observation, 200),
                (ATTEMPT_COVERAGE_PATH, coverage, 202),
            ):
                response = await client.post(
                    path, headers=headers, json=payload.model_dump(mode="json")
                )
                assert response.status_code == status, response.text
            response = await client.post(
                "/v1/regional/executors/claim",
                headers=headers,
                json={"executor_id": "pending-executor"},
            )
            assert response.status_code == 423, (
                "readiness ingestion granted a PENDING executor command authority"
            )

    asyncio.run(scenario())

    statuses = context.store.list_collector_statuses("cluster-a", "node-a")
    assert any(
        status.collector is CollectorKind.NVIDIA_KERNEL
        and status.last_success_at is not None
        for status in statuses
    ), "PENDING membership must still persist its registered collector reports"
    states = context.store.list_attempt_observation_states("cluster-a")
    assert any(state.observation == observation for state in states), (
        "PENDING membership must retain workload observations for readiness"
    )
    assert context.store.get_workload_coverage_heartbeat("cluster-a") == coverage
    assert context.store.get_remote_command(queued.command_id) == queued, (
        "a refused PENDING claim changed the queued command"
    )


@pytest.mark.parametrize(
    ("state", "expected_status"),
    [
        (RegionalClusterLifecycle.FAILED, 423),
        (RegionalClusterLifecycle.DRAINING, 423),
        (RegionalClusterLifecycle.REVOKED, 403),
        (RegionalClusterLifecycle.ROLLED_BACK, 403),
    ],
)
def test_pending_ingestion_exception_does_not_open_other_lifecycle_states(
    state: RegionalClusterLifecycle, expected_status: int
) -> None:
    context = build_context()
    context.regional_mode = True
    context.store.save_regional_cluster(
        registration("cluster-a", TOKEN_A).model_copy(update={"lifecycle_state": state})
    )
    heartbeat = WorkloadCoverageHeartbeat(
        cluster_id="cluster-a",
        observed_at=datetime.now(timezone.utc),
        watched_pods=0,
        watched_attempts=0,
        resource_version="not-pending",
        watcher_instance="watcher",
    )

    async def scenario() -> None:
        async with asgi_client(context) as client:
            response = await client.post(
                ATTEMPT_COVERAGE_PATH,
                headers={
                    "Authorization": f"Bearer {TOKEN_A}",
                    "X-GPU-Fault-Cluster-ID": "cluster-a",
                },
                json=heartbeat.model_dump(mode="json"),
            )
        assert response.status_code == expected_status, response.text

    asyncio.run(scenario())
    assert context.store.get_workload_coverage_heartbeat("cluster-a") is None


def test_pending_observation_is_still_bound_to_its_cluster_token() -> None:
    context = build_context()
    context.regional_mode = True
    context.store.save_regional_cluster(
        registration("cluster-a", TOKEN_A).model_copy(
            update={"lifecycle_state": RegionalClusterLifecycle.PENDING}
        )
    )
    heartbeat = WorkloadCoverageHeartbeat(
        cluster_id="cluster-b",
        observed_at=datetime.now(timezone.utc),
        watched_pods=0,
        watched_attempts=0,
        resource_version="foreign-cluster",
        watcher_instance="watcher",
    )

    async def scenario() -> None:
        async with asgi_client(context) as client:
            response = await client.post(
                ATTEMPT_COVERAGE_PATH,
                headers={
                    "Authorization": f"Bearer {TOKEN_A}",
                    "X-GPU-Fault-Cluster-ID": "cluster-a",
                },
                json=heartbeat.model_dump(mode="json"),
            )
        assert response.status_code == 403, response.text

    asyncio.run(scenario())
    assert context.store.get_workload_coverage_heartbeat("cluster-b") is None

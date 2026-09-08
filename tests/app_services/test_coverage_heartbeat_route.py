"""``POST /v1/attempts/coverage`` is how a watcher says "I saw nothing".

An idle cluster publishes no observation, so the topology resolver read
UNKNOWN and every node-mutating plan was BLOCKED (completion-watcher F4, live
DESTR-016 attempt 4). The heartbeat carries the watcher's own view of what it
watched; it is a cluster-token write like every other data-plane ingest, so a
request without the cluster credential must be refused before it can claim
coverage for somebody else's cluster.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

from gpu_fault.app import create_app
from gpu_fault.telemetry import ATTEMPT_COVERAGE_PATH, WorkloadCoverageHeartbeat
from tests._builders import asgi_client, build_context
from tests.regional._regional_support import TOKEN_A, registration

NOW = datetime(2026, 9, 8, 12, 0, tzinfo=timezone.utc)
CLUSTER = "cluster-a"


def _heartbeat(*, watched_pods: int = 0) -> WorkloadCoverageHeartbeat:
    return WorkloadCoverageHeartbeat(
        cluster_id=CLUSTER,
        observed_at=NOW,
        watched_pods=watched_pods,
        watched_attempts=0,
        resource_version="4711",
        watcher_instance="completion-watcher-0",
    )


def test_a_coverage_heartbeat_is_accepted_and_stored() -> None:
    context = build_context()

    async def scenario() -> int:
        async with asgi_client(context) as client:
            response = await client.post(
                ATTEMPT_COVERAGE_PATH,
                json=_heartbeat(watched_pods=6).model_dump(mode="json"),
            )
        return response.status_code

    status = asyncio.run(scenario())

    stored = context.store.get_workload_coverage_heartbeat(CLUSTER)
    assert status == 202, f"the heartbeat was not accepted: {status}"
    assert stored is not None, "the accepted heartbeat was not persisted"
    assert (stored.observed_at, stored.watched_pods) == (NOW, 6), stored


def test_a_stored_heartbeat_makes_the_cluster_idle_instead_of_unknown() -> None:
    context = build_context()

    async def scenario() -> None:
        async with asgi_client(context) as client:
            await client.post(
                ATTEMPT_COVERAGE_PATH, json=_heartbeat().model_dump(mode="json")
            )

    asyncio.run(scenario())

    resolved = context.topology.resolve(
        CLUSTER, "node-1", NOW + timedelta(seconds=30)
    ).workload_state

    assert resolved == "IDLE", (
        "a heartbeat the route accepted must be the coverage the resolver reads"
    )


def test_a_heartbeat_without_a_timezone_is_refused_and_stores_nothing() -> None:
    """A naive stamp is not a time, and a stored one poisons the row.

    Every reader compares this row against an aware ``now``, which raises on a
    naive value -- on the fault ingest path, for the whole cluster. So it is
    refused here rather than guessed at.
    """

    context = build_context()

    async def scenario() -> int:
        async with asgi_client(context) as client:
            response = await client.post(
                ATTEMPT_COVERAGE_PATH,
                json={
                    "cluster_id": CLUSTER,
                    "observed_at": NOW.replace(tzinfo=None).isoformat(),
                    "watched_pods": 0,
                    "watched_attempts": 0,
                    "resource_version": "4711",
                    "watcher_instance": "completion-watcher-0",
                },
            )
        return response.status_code

    status = asyncio.run(scenario())

    assert status == 422, f"a naive observed_at must be refused: {status}"
    assert context.store.get_workload_coverage_heartbeat(CLUSTER) is None, (
        "a refused heartbeat must not leave a row the resolver cannot compare"
    )


def test_a_coverage_heartbeat_without_the_cluster_token_is_refused() -> None:
    context = build_context()
    context.regional_mode = True
    context.store.save_regional_cluster(registration(CLUSTER, TOKEN_A))
    app = create_app(context)

    async def scenario() -> tuple[int, int]:
        async with asgi_client(app) as client:
            anonymous = await client.post(
                ATTEMPT_COVERAGE_PATH, json=_heartbeat().model_dump(mode="json")
            )
            authorized = await client.post(
                ATTEMPT_COVERAGE_PATH,
                headers={
                    "Authorization": f"Bearer {TOKEN_A}",
                    "X-GPU-Fault-Cluster-ID": CLUSTER,
                },
                json=_heartbeat().model_dump(mode="json"),
            )
        return anonymous.status_code, authorized.status_code

    anonymous_status, authorized_status = asyncio.run(scenario())

    assert anonymous_status == 401, (
        f"an unauthenticated heartbeat must be refused: {anonymous_status}"
    )
    assert authorized_status == 202, (
        f"the cluster's own token must be accepted: {authorized_status}"
    )
    assert context.store.get_workload_coverage_heartbeat(CLUSTER) is not None, (
        "the authorized heartbeat was not persisted"
    )

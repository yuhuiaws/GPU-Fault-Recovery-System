from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

from gpu_fault.app import create_app
from gpu_fault.app.routes.regional_registry import _status
from gpu_fault.regional import (
    RegionalClusterLifecycle,
    RegionalRegistryMember,
    RegionalRegistryRevision,
)
from tests._builders import asgi_client, build_context
from tests.regional._regional_support import TOKEN_A, registration

EXECUTION_TOKEN = "e" * 32


def context_and_app():
    context = build_context(execution_token=EXECUTION_TOKEN)
    context.regional_mode = True
    context.store.save_regional_cluster(registration("cluster-a", TOKEN_A))
    return context, create_app(context)


def operator_headers() -> dict[str, str]:
    return {"X-GPU-Fault-Execution-Token": EXECUTION_TOKEN}


def _member(
    member_id: str,
    revision: RegionalRegistryRevision,
    *,
    generation: int,
    last_seen: datetime,
) -> RegionalRegistryMember:
    at_revision = generation == revision.generation
    return RegionalRegistryMember(
        member_id=member_id,
        service_role="control-worker",
        release_id="release-a",
        generation=generation,
        content_sha256=revision.content_sha256 if at_revision else "b" * 64,
        ready=True,
        started_at=datetime(2026, 9, 15, tzinfo=timezone.utc),
        last_seen_at=last_seen,
    )


def test_status_reports_a_departed_required_member_as_neither_acked_nor_missing() -> (
    None
):
    """A stale straggler in required is dropped from missing and never blocks.

    _status feeds describe_unconverged_publish and rotate_token, which raise on
    a non-empty missing set, so a member that has left the fleet must not appear
    there or a converged publish would still read as unfinished. A member still
    heartbeating but merely behind stays in missing and holds convergence.
    """

    now = datetime(2026, 9, 15, 0, 2, tzinfo=timezone.utc)
    revision = RegionalRegistryRevision.build(
        generation=5,
        registrations=[registration("cluster-a", TOKEN_A)],
        previous_generation=4,
        required_member_ids=["live/process", "drained/process"],
        reason="join cluster-a",
        created_at=datetime(2026, 9, 15, tzinfo=timezone.utc),
    )
    live = _member("live/process", revision, generation=5, last_seen=now)
    # Last heartbeat 100s back, past the 90s window: gone, never acked gen 5.
    drained = _member(
        "drained/process",
        revision,
        generation=4,
        last_seen=now - timedelta(seconds=100),
    )
    status = _status(revision, [live, drained], observed_at=now, stale_seconds=90)
    assert status.converged is True
    assert status.acked_member_ids == ["live/process"]
    assert status.missing_member_ids == []

    revision_two = RegionalRegistryRevision.build(
        generation=5,
        registrations=[registration("cluster-a", TOKEN_A)],
        previous_generation=4,
        required_member_ids=["behind/process", "drained/process", "live/process"],
        reason="join cluster-a",
        created_at=datetime(2026, 9, 15, tzinfo=timezone.utc),
    )
    live_two = _member("live/process", revision_two, generation=5, last_seen=now)
    behind = _member("behind/process", revision_two, generation=4, last_seen=now)
    drained_two = _member(
        "drained/process",
        revision_two,
        generation=4,
        last_seen=now - timedelta(seconds=100),
    )
    status_two = _status(
        revision_two, [live_two, behind, drained_two], observed_at=now, stale_seconds=90
    )
    assert status_two.converged is False
    assert status_two.acked_member_ids == ["live/process"]
    assert status_two.missing_member_ids == ["behind/process"]


def test_registry_api_publishes_with_cas_and_reports_convergence() -> None:
    context, app = context_and_app()
    pending = registration("cluster-a", TOKEN_A).model_copy(
        update={"lifecycle_state": RegionalClusterLifecycle.PENDING}
    )

    async def scenario() -> None:
        async with asgi_client(app) as client:
            initial = await client.get(
                "/v1/regional/registry/status", headers=operator_headers()
            )
            published = await client.post(
                "/v1/regional/registry/revisions",
                headers=operator_headers(),
                json={
                    "expected_generation": 1,
                    "registrations": [pending.model_dump(mode="json")],
                    "reason": "prepare cluster update",
                },
            )
            conflict = await client.post(
                "/v1/regional/registry/revisions",
                headers=operator_headers(),
                json={
                    "expected_generation": 1,
                    "registrations": [],
                    "reason": "stale writer",
                },
            )

        assert initial.status_code == 200
        assert initial.json()["generation"] == 1
        assert published.status_code == 200
        body = published.json()
        assert body["generation"] == 2
        assert body["cluster_states"] == {"cluster-a": "PENDING"}
        assert body["converged"] is True
        assert body["missing_member_ids"] == []
        assert conflict.status_code == 409

    asyncio.run(scenario())
    assert context.store.get_regional_registry_head().generation == 2


def test_registry_rollback_publishes_a_higher_generation() -> None:
    context, app = context_and_app()
    pending = registration("cluster-a", TOKEN_A).model_copy(
        update={"lifecycle_state": RegionalClusterLifecycle.PENDING}
    )

    async def scenario() -> None:
        async with asgi_client(app) as client:
            await client.post(
                "/v1/regional/registry/revisions",
                headers=operator_headers(),
                json={
                    "expected_generation": 1,
                    "registrations": [pending.model_dump(mode="json")],
                    "reason": "prepare cluster update",
                },
            )
            rolled_back = await client.post(
                "/v1/regional/registry/rollback",
                headers=operator_headers(),
                json={
                    "expected_generation": 2,
                    "target_generation": 1,
                    "reason": "restore previous registry content",
                },
            )

        assert rolled_back.status_code == 200
        body = rolled_back.json()
        assert body["generation"] == 3
        assert body["cluster_states"] == {"cluster-a": "ACTIVE"}
        assert body["converged"] is True

    asyncio.run(scenario())
    revision = context.store.get_regional_registry_revision(3)
    assert revision.previous_generation == 2
    assert revision.registrations[0].lifecycle_state is RegionalClusterLifecycle.ACTIVE


def test_cluster_transitions_merge_concurrent_join_membership() -> None:
    _, app = context_and_app()
    cluster_b = registration("cluster-b", "b" * 32)
    cluster_c = registration("cluster-c", "c" * 32)

    async def transition(client, item) -> None:
        response = await client.post(
            f"/v1/regional/registry/clusters/{item.cluster_id}/transition",
            headers=operator_headers(),
            json={
                "registration": item.model_dump(mode="json"),
                "lifecycle_state": "PENDING",
                "reason": f"prepare {item.cluster_id}",
            },
        )
        assert response.status_code == 200, response.text

    async def scenario() -> None:
        async with asgi_client(app) as client:
            await asyncio.gather(
                transition(client, cluster_b), transition(client, cluster_c)
            )
            status = await client.get(
                "/v1/regional/registry/status", headers=operator_headers()
            )

        assert status.status_code == 200
        assert status.json()["cluster_states"] == {
            "cluster-a": "ACTIVE",
            "cluster-b": "PENDING",
            "cluster-c": "PENDING",
        }

    asyncio.run(scenario())


def test_cluster_transition_rejects_identity_drift() -> None:
    _, app = context_and_app()
    pending = registration("cluster-b", "b" * 32).model_copy(
        update={"lifecycle_state": RegionalClusterLifecycle.PENDING}
    )
    drifted = pending.model_copy(update={"region": "us-east-1"})

    async def scenario() -> None:
        async with asgi_client(app) as client:
            first = await client.post(
                "/v1/regional/registry/clusters/cluster-b/transition",
                headers=operator_headers(),
                json={
                    "registration": pending.model_dump(mode="json"),
                    "lifecycle_state": "PENDING",
                    "reason": "prepare cluster-b",
                },
            )
            second = await client.post(
                "/v1/regional/registry/clusters/cluster-b/transition",
                headers=operator_headers(),
                json={
                    "registration": drifted.model_dump(mode="json"),
                    "lifecycle_state": "ACTIVE",
                    "reason": "activate drifted cluster-b",
                },
            )

        assert first.status_code == 200
        assert second.status_code == 409

    asyncio.run(scenario())


def test_registry_api_requires_execution_token() -> None:
    _, app = context_and_app()

    async def scenario() -> None:
        async with asgi_client(app) as client:
            response = await client.get("/v1/regional/registry/status")
        assert response.status_code == 403

    asyncio.run(scenario())

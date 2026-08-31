from __future__ import annotations

import asyncio

from gpu_fault.app import create_app
from gpu_fault.regional import RegionalClusterLifecycle
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


def test_registry_api_requires_execution_token() -> None:
    _, app = context_and_app()

    async def scenario() -> None:
        async with asgi_client(app) as client:
            response = await client.get("/v1/regional/registry/status")
        assert response.status_code == 403

    asyncio.run(scenario())

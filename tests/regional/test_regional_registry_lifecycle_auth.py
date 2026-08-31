from __future__ import annotations

import asyncio

import pytest

from gpu_fault.app import create_app
from gpu_fault.regional import RegionalClusterLifecycle
from tests._builders import asgi_client, build_context
from tests.regional._regional_support import TOKEN_A, registration


def context_for(state: RegionalClusterLifecycle):
    context = build_context()
    context.regional_mode = True
    context.store.save_regional_cluster(
        registration("cluster-a", TOKEN_A).model_copy(update={"lifecycle_state": state})
    )
    return context


def headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {TOKEN_A}", "X-GPU-Fault-Cluster-ID": "cluster-a"}


@pytest.mark.parametrize(
    ("state", "expected_status"),
    [
        (RegionalClusterLifecycle.PENDING, 423),
        (RegionalClusterLifecycle.DRAINING, 423),
        (RegionalClusterLifecycle.REVOKED, 403),
    ],
)
def test_non_active_registry_states_cannot_claim_commands(
    state: RegionalClusterLifecycle, expected_status: int
) -> None:
    async def scenario() -> None:
        async with asgi_client(context_for(state)) as client:
            response = await client.post(
                "/v1/regional/executors/claim",
                headers=headers(),
                json={"executor_id": "executor-a"},
            )
        assert response.status_code == expected_status

    asyncio.run(scenario())


def test_pending_registry_allows_read_only_verification() -> None:
    async def scenario() -> None:
        async with asgi_client(context_for(RegionalClusterLifecycle.PENDING)) as client:
            response = await client.get(
                "/v1/regional/executors/hyperpod-submissions",
                headers=headers(),
                params={"cluster_name": "hp-cluster-a", "idempotency_key": "missing"},
            )
        assert response.status_code == 200
        assert response.json() is None

    asyncio.run(scenario())


def test_draining_registry_allows_in_flight_result_path() -> None:
    async def scenario() -> None:
        async with asgi_client(
            context_for(RegionalClusterLifecycle.DRAINING)
        ) as client:
            response = await client.post(
                "/v1/regional/executors/missing/result",
                headers=headers(),
                json={"lease_token": "lease-a", "status": "SUCCEEDED"},
            )
        assert response.status_code == 404

    asyncio.run(scenario())


def test_cluster_auth_fails_closed_when_local_registry_is_stale(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = create_app(context_for(RegionalClusterLifecycle.ACTIVE))
    monkeypatch.setattr(app.state.regional_registry_runtime, "is_ready", lambda: False)

    async def scenario() -> None:
        async with asgi_client(app) as client:
            response = await client.post(
                "/v1/regional/executors/claim",
                headers=headers(),
                json={"executor_id": "executor-a"},
            )
        assert response.status_code == 503
        assert response.headers["retry-after"] == "2"

    asyncio.run(scenario())

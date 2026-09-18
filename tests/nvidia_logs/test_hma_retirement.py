from __future__ import annotations

import asyncio

import pytest

from gpu_fault.app import create_app
from gpu_fault.app.authorization import ExplicitAuthorizationRegistry
from gpu_fault.nvidia_logs import FaultIngestionResult, FaultSignalSource
from tests._builders import asgi_client, build_context


@pytest.mark.parametrize("regional", [False, True])
def test_retired_hma_routes_are_absent_and_cannot_ingest(regional: bool) -> None:
    context = build_context()
    context.regional_mode = regional
    context.execution_token = "retirement-test-only"
    app = create_app(context)
    registry = ExplicitAuthorizationRegistry()
    registry.load(app.routes)

    async def scenario() -> None:
        async with asgi_client(app) as client:
            for suffix in ("node", "kubernetes-node", "cloudwatch", "discovery"):
                path = f"/v1/provider-events/hyperpod-hma/{suffix}"
                assert not registry.route_exists(path), path
                assert path not in app.openapi()["paths"], path
                response = await client.post(path, json={"cluster_id": "cluster-a"})
                assert response.status_code == 404, response.text

    asyncio.run(scenario())
    assert not any(context.store.incident_state_counts().values())


@pytest.mark.parametrize("source", ["KUBERNETES_NODE", "CLOUDWATCH_LOG"])
def test_legacy_normalized_evidence_still_decodes(source: str) -> None:
    payload = {
        "normalized": {
            "provider_signals": [
                {
                    "signal_id": "legacy-record",
                    "source": source,
                    "cluster_id": "cluster-a",
                    "node_id": "node-a",
                    "observed_at": "2026-07-20T10:00:00.000000Z",
                    "health_status": "Unschedulable",
                    "fault_types": ["EfaError"],
                    "fault_reasons": ["InstanceUnreachable"],
                    "unschedulable_taint": True,
                    "unresolved_reasons": [
                        "hma_unschedulable_without_code: historical observation"
                    ],
                }
            ],
            "xid_events": [],
            "sxid_events": [],
        },
        "decisions": [],
        "unresolved": 1,
    }
    decoded = FaultIngestionResult.model_validate(payload)
    assert decoded.normalized.provider_signals[0].source is FaultSignalSource(source)
    assert decoded.model_dump(mode="json", exclude_unset=True) == payload

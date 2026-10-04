"""Regional routes that re-check the cluster identity and move agents.

The cluster-token middleware binds the payload ``cluster_id`` before a route
runs; the routes still refuse a mismatch themselves, so a router mounted
without that middleware cannot be talked across clusters. Node Action keys are
issued for a node the control plane knows only from its GPU metrics, a spare
with an active GPU finding is not ready, and the agent drain/revoke routes
need the registry and run its transitions.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from gpu_fault.fleet import AgentLifecycleState
from gpu_fault.gpu_metric_models import GpuHealthFinding, GpuHealthSeverity
from gpu_fault.gpu_metrics import GpuMetricLatest, GpuMetricSample, GpuMetricSource
from gpu_fault.models import MarkerScope, NodeMarker, RecoveryAction, Severity
from tests.fleet._support import NOW, SECRET
from tests.regional._cov95_runtime_routes import HEADERS, Routes
from tests.regional._cov95_runtime_routes import routes_fixture as routes_fixture

GPU = "GPU-metrics-only"
TRANSITION = {"expected_generation": 1, "transition_id": "t-1", "reason": "unit"}


@pytest.mark.parametrize(
    "route,payload",
    [
        ("/v1/regional/node-action-keys", {"node_ids": ["node-a"]}),
        ("/v1/regional/executors/spares/health", {"node_aliases": ["node-a"]}),
        (
            "/v1/regional/executors/evidence",
            {
                "record_id": "raw-1",
                "node_id": "node-a",
                "kind": "NODE_LOGS",
                "observed_at": NOW.astimezone(timezone.utc).isoformat(),
                "payload": {"text": "fixture"},
            },
        ),
    ],
)
def test_routes_refuse_a_payload_cluster_that_is_not_the_authenticated_one(
    routes: Routes, route: str, payload: dict[str, object]
) -> None:
    response = routes.client.post(
        route, json={"cluster_id": "cluster-b", **payload}, headers=HEADERS
    )

    assert response.status_code == 403, response.text
    assert response.json()["detail"] == "authenticated cluster does not match request"
    assert routes.io.calls == [], "a refused request still reached the store"


def test_node_action_keys_are_issued_for_a_node_known_only_from_gpu_metrics(
    routes: Routes,
) -> None:
    node_id = "node-metrics-only"
    routes.context.store.observe_gpu_metric(
        ("cluster-a", node_id, GPU, "temperature"),
        GpuMetricLatest(
            cluster_id="cluster-a",
            node_id=node_id,
            observed_at=NOW,
            source=GpuMetricSource.DCGM_EXPORTER,
            sample=GpuMetricSample(
                metric_name="DCGM_FI_DEV_GPU_TEMP",
                canonical_name="temperature",
                gpu_uuid=GPU,
                value=41.0,
            ),
        ),
    )

    response = routes.client.post(
        "/v1/regional/node-action-keys",
        json={"cluster_id": "cluster-a", "node_ids": [node_id]},
        headers=HEADERS,
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["cluster_id"] == "cluster-a"
    assert list(body["keys"]) == [node_id]
    assert body["keys"][node_id] != SECRET, "the fleet master left the control plane"
    assert len(body["keys"][node_id]) >= 32

    refused = routes.client.post(
        "/v1/regional/node-action-keys",
        json={"cluster_id": "cluster-a", "node_ids": ["node-never-seen"]},
        headers=HEADERS,
    )
    assert refused.status_code == 404, refused.text


def test_a_spare_with_an_active_gpu_finding_is_not_ready(routes: Routes) -> None:
    finding = GpuHealthFinding(
        finding_id="finding-a",
        cluster_id="cluster-a",
        node_id="node-a",
        gpu_uuid=GPU,
        observed_at=NOW,
        severity=GpuHealthSeverity.WARNING,
        reason="temperature above the site threshold",
        canonical_name="temperature",
        value=95.0,
    )
    routes.context.store.update_gpu_finding(
        ("cluster-a", "node-a", GPU, "temperature"), finding, NOW
    )

    response = routes.client.post(
        "/v1/regional/executors/spares/health",
        json={"cluster_id": "cluster-a", "node_aliases": ["node-a"]},
        headers=HEADERS,
    )

    assert response.status_code == 200, response.text
    report = response.json()
    assert report["ready"] is False
    assert "active GPU health finding exists" in report["reasons"]


def test_agent_drain_and_revoke_need_the_registry_and_run_its_transitions(
    routes: Routes,
) -> None:
    drained = routes.client.post(
        "/v1/regional/executors/agents/node-a/drain", json=TRANSITION, headers=HEADERS
    )
    assert drained.status_code == 200, drained.text
    assert drained.json()["lifecycle_state"] == AgentLifecycleState.DRAINING.value
    assert drained.json()["generation"] == 2

    stale = routes.client.post(
        "/v1/regional/executors/agents/node-b/revoke", json=TRANSITION, headers=HEADERS
    )
    assert stale.status_code == 409, stale.text
    assert "drained by the same transition" in stale.json()["detail"]

    revoked = routes.client.post(
        "/v1/regional/executors/agents/node-a/revoke", json=TRANSITION, headers=HEADERS
    )
    assert revoked.status_code == 200, revoked.text
    assert revoked.json()["lifecycle_state"] == AgentLifecycleState.REVOKED.value
    stored = routes.context.store.get_agent("cluster-a", "node-a")
    assert stored.lifecycle_state is AgentLifecycleState.REVOKED

    routes.context.fleet_registry = None
    disabled = routes.client.post(
        "/v1/regional/executors/agents/node-b/drain", json=TRANSITION, headers=HEADERS
    )
    assert disabled.status_code == 503, disabled.text
    assert disabled.json()["detail"] == "agent registry is disabled"


def test_a_spare_under_a_live_trusted_fault_marker_is_not_ready(routes: Routes) -> None:
    now = datetime.now(timezone.utc)
    routes.context.store.add_marker(
        NodeMarker(
            marker_id="marker-spare",
            cluster_id="cluster-a",
            source="test-agent",
            trusted=True,
            incident_id="",
            observed_at=now - timedelta(seconds=10),
            expires_at=now + timedelta(minutes=30),
            scope=MarkerScope(node_ids=["node-a"]),
            severity=Severity.CRITICAL,
            recommended_action=RecoveryAction.REBOOT_NODE,
            action_owner="simulated-runtime",
            mapping_version="test-v1",
        )
    )

    response = routes.client.post(
        "/v1/regional/executors/spares/health",
        json={"cluster_id": "cluster-a", "node_aliases": ["node-a"]},
        headers=HEADERS,
    )

    assert response.status_code == 200, response.text
    report = response.json()
    assert report["ready"] is False
    assert any(
        reason.startswith("active trusted node fault marker exists: ")
        for reason in report["reasons"]
    ), report["reasons"]

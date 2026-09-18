from __future__ import annotations

from typing import Any

import pytest

from gpu_fault.async_store import StoreIoCapacityExceeded
from gpu_fault.fleet import BarrierCoordinator
from gpu_fault.models import WorkflowOperation
from tests.fleet._support import heartbeat, signed
from tests.fleet.test_cov95_runtime_deployments import request
from tests.regional._cov95_runtime_routes import HEADERS, Routes
from tests.regional._cov95_runtime_routes import routes_fixture as routes_fixture
from tests.regional._cov95_runtime_support import offline_runtime as offline_runtime

TRANSITION = {
    "expected_generation": 1,
    "transition_id": "unit-transition",
    "reason": "unit",
}


@pytest.mark.parametrize(
    ("path", "payload"),
    [
        ("/agents/heartbeat", signed(heartbeat("node-a")).model_dump(mode="json")),
        ("/readiness", {"cluster_id": "cluster-a", "node_ids": ["node-a"]}),
        ("/agents/cluster-a/node-a/drain", TRANSITION),
        ("/deployments", request().model_dump(mode="json")),
        ("/deployments/unit/next-wave", None),
        ("/deployments/unit/nodes/node-a", {"status": "INSTALLING"}),
    ],
)
def test_disabled_registry_refuses_runtime_mutations_before_store_io(
    routes: Routes, path: str, payload: Any
) -> None:
    routes.context.fleet_registry = None
    response = routes.client.post("/v1/fleet" + path, json=payload, headers=HEADERS)
    assert response.status_code == 503, response.text
    assert response.json()["detail"] == "agent registry is disabled"
    assert routes.io.calls == []


@pytest.mark.parametrize("token", [None, "wrong"])
@pytest.mark.parametrize("action", ["drain", "revoke", "reactivate"])
def test_agent_transition_cannot_bypass_the_execution_token(
    routes: Routes, token: str | None, action: str
) -> None:
    headers = {"X-GPU-Fault-Cluster-ID": "cluster-a"}
    if token is not None:
        headers["X-GPU-Fault-Execution-Token"] = token
    before = routes.context.store.get_agent("cluster-a", "node-a")
    response = routes.client.post(
        f"/v1/fleet/agents/cluster-a/node-a/{action}", json=TRANSITION, headers=headers
    )
    assert response.status_code == 403
    assert routes.context.store.get_agent("cluster-a", "node-a") == before
    assert routes.io.calls == []


def test_missing_server_execution_secret_does_not_authorize_a_matching_header(
    routes: Routes,
) -> None:
    routes.context.execution_token = None
    response = routes.client.post(
        "/v1/fleet/deployments", json=request().model_dump(mode="json"), headers=HEADERS
    )
    assert response.status_code == 403
    assert routes.io.calls == []


@pytest.mark.parametrize(
    "path", ["/agents?cluster_id=cluster-b", "/agents/cluster-b/node-a"]
)
def test_route_layer_cluster_binding_refuses_cross_cluster_agent_reads(
    routes: Routes, path: str
) -> None:
    response = routes.client.get("/v1/fleet" + path, headers=HEADERS)
    assert response.status_code == 403
    assert routes.io.calls == []


def test_agent_list_is_header_scoped_and_disabled_registry_has_no_agents(
    routes: Routes,
) -> None:
    response = routes.client.get("/v1/fleet/agents", headers=HEADERS)
    assert response.status_code == 200
    assert {item["node_id"] for item in response.json()} == {"node-a", "node-b"}
    assert {item["cluster_id"] for item in response.json()} == {"cluster-a"}
    routes.context.fleet_registry = None
    assert routes.client.get("/v1/fleet/agents", headers=HEADERS).json() == []


@pytest.mark.parametrize("action", ["drain", "revoke", "reactivate"])
def test_invalid_agent_transition_is_a_conflict_and_preserves_generation(
    routes: Routes, action: str
) -> None:
    before = routes.context.store.get_agent("cluster-a", "node-a")
    response = routes.client.post(
        f"/v1/fleet/agents/cluster-a/node-a/{action}",
        json={**TRANSITION, "expected_generation": 99},
        headers=HEADERS,
    )
    assert response.status_code == 409
    assert routes.context.store.get_agent("cluster-a", "node-a") == before


def test_invalid_heartbeat_signature_is_rejected_without_updating_the_record(
    routes: Routes,
) -> None:
    before = routes.context.store.get_agent("cluster-a", "node-a")
    body = signed(heartbeat("node-a")).model_dump(mode="json")
    body["signature"] = "0" * 64
    response = routes.client.post(
        "/v1/fleet/agents/heartbeat", json=body, headers=HEADERS
    )
    assert response.status_code == 409
    assert routes.context.store.get_agent("cluster-a", "node-a") == before


def test_store_capacity_failure_returns_retry_after_instead_of_a_partial_result(
    routes: Routes,
) -> None:
    routes.io.error = StoreIoCapacityExceeded("synthetic saturated store lane")
    response = routes.client.get("/v1/fleet/agents", headers=HEADERS)
    assert response.status_code == 503
    assert response.headers["retry-after"] == "2"
    assert response.json()["detail"] == "store I/O capacity exceeded"


def test_deployment_routes_retain_node_ownership_and_only_heartbeat_can_make_ready(
    routes: Routes,
) -> None:
    body = request(desired_agent_version="0.10.0").model_dump(mode="json")
    created = routes.client.post("/v1/fleet/deployments", json=body, headers=HEADERS)
    assert created.status_code == 200, created.text
    assert created.json()["deployment_id"] == "unit-deployment"
    listed = routes.client.get("/v1/fleet/deployments", headers=HEADERS)
    fetched = routes.client.get(
        "/v1/fleet/deployments/unit-deployment", headers=HEADERS
    )
    assert listed.json() == [fetched.json()]
    wave = routes.client.post(
        "/v1/fleet/deployments/unit-deployment/next-wave", headers=HEADERS
    )
    assert wave.status_code == 200, wave.text
    assert wave.json()["node_ids"] == ["node-a"]
    for node in ("node-a", "foreign"):
        response = routes.client.post(
            f"/v1/fleet/deployments/unit-deployment/nodes/{node}",
            json={"status": "READY"},
            headers=HEADERS,
        )
        assert response.status_code == 409
    failed = routes.client.post(
        "/v1/fleet/deployments/unit-deployment/nodes/node-a",
        json={"status": "FAILED", "reason": "synthetic installer refused"},
        headers=HEADERS,
    )
    assert failed.status_code == 200, failed.text
    assert failed.json()["nodes"][0]["status"] == "FAILED"
    before = routes.context.store.get_fleet_deployment("unit-deployment")
    conflicting = routes.client.post(
        "/v1/fleet/deployments",
        json={**body, "desired_agent_version": "other"},
        headers=HEADERS,
    )
    assert conflicting.status_code == 409
    assert routes.context.store.get_fleet_deployment("unit-deployment") == before


def test_barrier_list_and_path_identifier_round_trip_without_triggering_an_action(
    routes: Routes,
) -> None:
    coordinator = BarrierCoordinator(routes.context.store)
    barrier = coordinator.create(
        barrier_id="workflow/step",
        cluster_id="cluster-a",
        workflow_request_id="workflow",
        incident_id="incident",
        fencing_token=1,
        operation=WorkflowOperation.RESET_GPU,
        generations={"node-a": 1},
    )
    response = routes.client.get("/v1/fleet/barriers/workflow/step", headers=HEADERS)
    assert response.status_code == 200
    assert response.json() == barrier.model_dump(mode="json")
    assert routes.client.get("/v1/fleet/barriers", headers=HEADERS).json() == [
        response.json()
    ]
    assert routes.context.store.get_barrier("workflow/step").state.value == "PREPARING"

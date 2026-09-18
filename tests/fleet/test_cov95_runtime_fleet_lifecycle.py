from __future__ import annotations

from datetime import timedelta
from typing import Any

import pytest

from gpu_fault.fleet import (
    AgentHeartbeat,
    AgentLifecycleState,
    AgentTransitionRequest,
    BarrierCoordinator,
    DeploymentNodeStatus,
    DeploymentNodeUpdate,
    FleetRegistry,
)
from gpu_fault.fleet_endpoint import node_name_address, validate_agent_endpoint
from gpu_fault.fleet_registry_deployments import reconcile_deployment
from gpu_fault.models import WorkflowOperation
from tests._builders import build_store
from tests.fleet._support import NOW, SECRET, heartbeat, registry, signed
from tests.fleet.test_cov95_runtime_deployments import request
from tests.regional._cov95_runtime_support import offline_runtime as offline_runtime


@pytest.mark.parametrize("defect", ["secret", "ports"])
def test_registry_requires_registration_credentials_and_an_endpoint_port(
    defect: str,
) -> None:
    source = registry()
    with pytest.raises(ValueError, match="registration secret|endpoint port"):
        FleetRegistry(
            build_store(),
            "short" if defect == "secret" else SECRET,
            source.policy,
            endpoint_allowed_ports=frozenset()
            if defect == "ports"
            else frozenset({9099}),
        )


def test_older_heartbeat_cannot_replace_an_accepted_agent_record() -> None:
    fleet = registry()
    saved = fleet.register(signed(heartbeat("node-a")))
    with pytest.raises(ValueError, match="older than the stored heartbeat"):
        fleet.register(
            signed(heartbeat("node-a", observed_at=NOW - timedelta(seconds=1)))
        )
    assert fleet.store.get_agent("cluster-a", "node-a") == saved


def test_revocation_is_idempotent_but_reactivation_requires_the_same_transition() -> (
    None
):
    fleet = registry()
    active = fleet.register(signed(heartbeat("node-a")))
    transition = AgentTransitionRequest(
        expected_generation=active.generation, transition_id="owned", reason="unit"
    )
    drained = fleet.drain_agent("cluster-a", "node-a", transition)
    revoked = fleet.revoke_agent("cluster-a", "node-a", transition)
    assert revoked.lifecycle_state is AgentLifecycleState.REVOKED
    assert fleet.revoke_agent("cluster-a", "node-a", transition) == revoked
    assert fleet.drain_agent("cluster-a", "node-a", transition) == revoked
    with pytest.raises(ValueError, match="revoking transition"):
        fleet.reactivate_agent(
            "cluster-a",
            "node-a",
            AgentTransitionRequest(
                expected_generation=drained.generation,
                transition_id="foreign",
                reason="unit",
            ),
        )
    assert fleet.store.get_agent("cluster-a", "node-a") == revoked
    with pytest.raises(ValueError, match="not fleet-ready"):
        fleet.endpoint("cluster-a", "node-a")


def test_repeated_deployment_progress_is_idempotent_but_retains_new_reason() -> None:
    fleet = registry()
    deployment = fleet.create_deployment(request())
    fleet.start_next_wave(deployment.deployment_id)
    update = DeploymentNodeUpdate(
        status=DeploymentNodeStatus.INSTALLING, reason="owned installer progressing"
    )
    changed = fleet.update_deployment_node(deployment.deployment_id, "node-a", update)
    assert changed.nodes[0].reason == update.reason
    assert (
        fleet.update_deployment_node(deployment.deployment_id, "node-a", update)
        == changed
    )
    assert changed.nodes[1].status is DeploymentNodeStatus.PENDING


@pytest.mark.parametrize(
    "defect", ["unrelated", "inactive", "wrong-pin", "already-ready"]
)
def test_reconciliation_cannot_ready_a_node_without_matching_active_identity(
    monkeypatch: pytest.MonkeyPatch, defect: str
) -> None:
    fleet = registry()
    deployment = fleet.create_deployment(request())
    record = fleet.register(
        signed(
            heartbeat(
                "node-c" if defect == "unrelated" else "node-a",
                version="different" if defect == "wrong-pin" else "0.9.0",
            )
        )
    )
    if defect == "inactive":
        record = fleet.drain_agent(
            "cluster-a",
            "node-a",
            AgentTransitionRequest(
                expected_generation=record.generation,
                transition_id="owned",
                reason="unit",
            ),
        )
    before = fleet.store.get_fleet_deployment(deployment.deployment_id)
    writes = []
    monkeypatch.setattr(
        fleet.store,
        "replace_fleet_deployment_if_matches",
        lambda *args: writes.append(args) or True,
    )
    reconcile_deployment(fleet, before, record)
    assert writes == []
    assert fleet.store.get_fleet_deployment(deployment.deployment_id) == before
    assert before.nodes[1].status is DeploymentNodeStatus.PENDING
    assert before.nodes[0].status is (
        DeploymentNodeStatus.READY
        if defect in {"inactive", "already-ready"}
        else DeploymentNodeStatus.PENDING
    )


@pytest.mark.parametrize("exhausted", [False, True])
def test_heartbeat_reconciliation_retries_cas_without_optimistic_ready_state(
    monkeypatch: pytest.MonkeyPatch, exhausted: bool
) -> None:
    fleet = registry()
    record = fleet.register(signed(heartbeat("node-a")))
    deployment = fleet.create_deployment(request(desired_agent_version="next"))
    matching = record.model_copy(update={"agent_version": "next"})
    fleet.store.save_agent(matching)
    replace = fleet.store.replace_fleet_deployment_if_matches
    writes = []

    def contested(value: Any, expected: Any) -> bool:
        writes.append(value)
        return not exhausted and len(writes) > 1 and replace(value, expected)

    monkeypatch.setattr(fleet.store, "replace_fleet_deployment_if_matches", contested)
    if exhausted:
        with pytest.raises(ValueError, match="conflicted with concurrent heartbeats"):
            reconcile_deployment(fleet, deployment, matching)
        assert len(writes) == 16
        assert fleet.store.get_fleet_deployment(deployment.deployment_id) == deployment
    else:
        reconcile_deployment(fleet, deployment, matching)
        assert len(writes) == 2
        current = fleet.store.get_fleet_deployment(deployment.deployment_id)
        assert current.nodes[0].status is DeploymentNodeStatus.READY
        assert current.nodes[1].status is DeploymentNodeStatus.PENDING
        reconcile_deployment(fleet, current, matching)
        assert len(writes) == 2


@pytest.mark.parametrize(
    ("body", "reason"),
    [
        ("not-a-certificate!", "PEM block is not valid base64"),
        (" \n\t", "PEM block is empty"),
    ],
)
def test_heartbeat_rejects_empty_or_unparseable_certificate_blocks(
    body: str, reason: str
) -> None:
    payload = heartbeat("node-a").model_dump(mode="python")
    payload["tls_certificate_pem"] = (
        "-----BEGIN CERTIFICATE-----\n" + body + "\n-----END CERTIFICATE-----"
    )
    with pytest.raises(ValueError, match=reason):
        AgentHeartbeat.model_validate(payload)


def test_active_agent_cannot_be_reactivated_by_a_transition_it_never_entered() -> None:
    fleet = registry()
    original = fleet.register(signed(heartbeat("node-a")))
    with pytest.raises(ValueError, match="not revoked"):
        fleet.reactivate_agent(
            "cluster-a",
            "node-a",
            AgentTransitionRequest(
                expected_generation=original.generation,
                transition_id="not-started",
                reason="unit",
            ),
        )
    assert fleet.store.get_agent("cluster-a", "node-a") == original


@pytest.mark.parametrize("prepared", [False, True])
def test_commit_evidence_is_ignored_until_the_barrier_entered_commit(
    prepared: bool,
) -> None:
    fleet = registry()
    coordinator = BarrierCoordinator(fleet.store, now=lambda: NOW)
    barrier = coordinator.create(
        barrier_id="unit-uncommitted",
        cluster_id="cluster-a",
        workflow_request_id="unit-workflow",
        incident_id="unit-incident",
        fencing_token=1,
        operation=WorkflowOperation.RESET_GPU,
        generations={"node-a": 1},
    )
    if prepared:
        barrier = coordinator.record_prepare(barrier.barrier_id, "node-a")
    assert (
        coordinator.record_commit(
            barrier.barrier_id, "node-a", details={"unapproved_commit": True}
        )
        == barrier
    )
    assert fleet.store.get_barrier(barrier.barrier_id) == barrier


def test_invalid_encoded_node_address_cannot_become_an_endpoint_ip() -> None:
    assert node_name_address("ip-256-0-0-1.internal") is None


@pytest.mark.parametrize(
    ("endpoint", "reason"),
    [
        ("https://:9099", "must name a host"),
        ("https://node-a", "port 443 is not one of"),
    ],
)
def test_endpoint_without_an_explicit_allowed_address_or_port_is_refused(
    endpoint: str, reason: str
) -> None:
    with pytest.raises(ValueError, match=reason):
        validate_agent_endpoint("node-a", endpoint)

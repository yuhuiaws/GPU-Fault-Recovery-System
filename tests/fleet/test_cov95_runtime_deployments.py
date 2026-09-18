from __future__ import annotations

from typing import Any

import pytest

from gpu_fault.fleet import (
    BarrierCoordinator,
    BarrierState,
    DeploymentNodeStatus,
    DeploymentNodeUpdate,
    DeploymentStatus,
    FleetDeploymentRequest,
)
from gpu_fault.fleet_deployment import active_deployment_wave
from gpu_fault.models import WorkflowOperation
from tests.fleet._support import ARTIFACT, CONFIG, NOW, heartbeat, registry, signed
from tests.regional._cov95_runtime_support import offline_runtime as offline_runtime


def request(**changes: Any) -> FleetDeploymentRequest:
    return FleetDeploymentRequest(
        **{
            "deployment_id": "unit-deployment",
            "cluster_id": "cluster-a",
            "node_ids": ["node-a", "node-b"],
            "max_unavailable": 1,
            "desired_agent_version": "0.9.0",
            "desired_artifact_sha256": ARTIFACT,
            "desired_policy_version": "catalog-a",
            "desired_runtime_profile_version": "profile-a",
            "desired_config_digest": CONFIG,
            **changes,
        }
    )


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"deployment_id": "../bad"}, "unsupported characters"),
        ({"node_ids": ["node-a", "node-a"]}, "unique"),
        ({"max_unavailable": 3}, "node count"),
        ({"first_wave_max_unavailable": 2}, "first_wave"),
        ({"max_unavailable_per_failure_domain": 2}, "failure_domain"),
        ({"node_failure_domains": {"node-a": "rack"}}, "exactly"),
        ({"node_failure_domains": {"node-a": "", "node-b": "rack"}}, "non-empty"),
    ],
)
def test_deployment_request_cannot_widen_its_approved_node_budget(
    changes: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        request(**changes)


@pytest.mark.parametrize(
    ("node", "status", "message"),
    [
        ("foreign", DeploymentNodeStatus.INSTALLING, "not part"),
        ("node-a", DeploymentNodeStatus.READY, "matching agent heartbeat"),
        ("node-a", DeploymentNodeStatus.FAILED, "only an INSTALLING"),
        ("node-b", DeploymentNodeStatus.INSTALLING, "active deployment wave"),
    ],
)
def test_deployment_transition_refuses_unowned_nodes_and_unproven_readiness(
    node: str, status: DeploymentNodeStatus, message: str
) -> None:
    fleet = registry()
    original = fleet.create_deployment(request())
    with pytest.raises(ValueError, match=message):
        fleet.update_deployment_node(
            original.deployment_id, node, DeploymentNodeUpdate(status=status)
        )
    assert fleet.store.get_fleet_deployment(original.deployment_id) == original


def test_installing_node_cannot_be_silently_put_back_into_pending() -> None:
    fleet = registry()
    deployment = fleet.create_deployment(request())
    fleet.start_next_wave(deployment.deployment_id)
    before = fleet.store.get_fleet_deployment(deployment.deployment_id)
    with pytest.raises(ValueError, match="invalid deployment node transition"):
        fleet.update_deployment_node(
            deployment.deployment_id,
            "node-a",
            DeploymentNodeUpdate(status=DeploymentNodeStatus.PENDING),
        )
    assert fleet.store.get_fleet_deployment(deployment.deployment_id) == before


@pytest.mark.parametrize("operation", ["update", "start", "cancel", "retry"])
def test_deployment_cas_conflicts_are_bounded_and_preserve_the_current_record(
    monkeypatch: pytest.MonkeyPatch, operation: str
) -> None:
    fleet = registry()
    deployment = fleet.create_deployment(request())
    if operation in {"update", "retry"}:
        fleet.start_next_wave(deployment.deployment_id)
    if operation == "retry":
        fleet.update_deployment_node(
            deployment.deployment_id,
            "node-a",
            DeploymentNodeUpdate(status=DeploymentNodeStatus.FAILED),
        )
    before = fleet.store.get_fleet_deployment(deployment.deployment_id)
    replacements = []
    monkeypatch.setattr(
        fleet.store,
        "replace_fleet_deployment_if_matches",
        lambda value, expected: replacements.append((value, expected)) or False,
    )
    with pytest.raises(ValueError, match="conflicted with concurrent"):
        if operation == "update":
            fleet.update_deployment_node(
                deployment.deployment_id,
                "node-a",
                DeploymentNodeUpdate(status=DeploymentNodeStatus.FAILED),
            )
        elif operation == "start":
            fleet.start_next_wave(deployment.deployment_id)
        elif operation == "cancel":
            fleet.cancel_deployment(
                deployment.deployment_id, reason="unit cancellation"
            )
        else:
            fleet.retry_failed_deployment(deployment.deployment_id)
    assert len(replacements) == 5
    assert fleet.store.get_fleet_deployment(deployment.deployment_id) == before


def test_wave_start_retries_a_transient_cas_conflict_and_is_idempotent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fleet = registry()
    deployment = fleet.create_deployment(request())
    original = fleet.store.replace_fleet_deployment_if_matches
    writes = []

    def replace(value: Any, expected: Any) -> bool:
        writes.append(value)
        return len(writes) > 1 and original(value, expected)

    monkeypatch.setattr(fleet.store, "replace_fleet_deployment_if_matches", replace)
    lease = fleet.start_next_wave(deployment.deployment_id)
    assert lease.node_ids == ["node-a"]
    assert len(writes) == 2
    assert fleet.start_next_wave(deployment.deployment_id).node_ids == ["node-a"]
    assert len(writes) == 2


@pytest.mark.parametrize("mode", ["start", "update"])
def test_corrupt_wave_cannot_exceed_max_unavailable(
    monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    fleet = registry()
    deployment = fleet.create_deployment(request())
    if mode == "update":
        fleet.start_next_wave(deployment.deployment_id)
        deployment = fleet.store.get_fleet_deployment(deployment.deployment_id)
    malformed = deployment.model_copy(update={"waves": [["node-a", "node-b"]]})
    monkeypatch.setattr(fleet.store, "get_fleet_deployment", lambda key: malformed)
    writes = []
    monkeypatch.setattr(
        fleet.store,
        "replace_fleet_deployment_if_matches",
        lambda *args: writes.append(args) or True,
    )
    with pytest.raises(ValueError, match="max_unavailable"):
        if mode == "start":
            fleet.start_next_wave(deployment.deployment_id)
        else:
            fleet.update_deployment_node(
                deployment.deployment_id,
                "node-b",
                DeploymentNodeUpdate(status=DeploymentNodeStatus.INSTALLING),
            )
    assert writes == []


def test_failed_node_retry_requires_a_fresh_matching_agent() -> None:
    fleet = registry()
    fleet.register(signed(heartbeat("node-a")))
    deployment = fleet.create_deployment(request())
    assert deployment.nodes[0].status is DeploymentNodeStatus.READY
    fleet.start_next_wave(deployment.deployment_id)
    failed = fleet.update_deployment_node(
        deployment.deployment_id,
        "node-b",
        DeploymentNodeUpdate(status=DeploymentNodeStatus.FAILED),
    )
    fleet.register(signed(heartbeat("node-b")))
    retried = fleet.retry_failed_deployment(deployment.deployment_id)
    assert failed.status is DeploymentStatus.FAILED
    assert retried.status is DeploymentStatus.SUCCEEDED
    assert all(node.status is DeploymentNodeStatus.READY for node in retried.nodes), (
        retried
    )
    assert active_deployment_wave(retried) is None
    assert fleet.retry_failed_deployment(deployment.deployment_id) == retried
    with pytest.raises(ValueError, match="already complete"):
        fleet.start_next_wave(deployment.deployment_id)


def test_failed_node_summary_is_bounded_before_any_new_wave() -> None:
    fleet = registry()
    deployment = fleet.create_deployment(
        request(node_ids=[f"node-{index:02}" for index in range(22)])
    )
    cancelled = fleet.cancel_deployment(deployment.deployment_id, reason="unit")
    with pytest.raises(ValueError, match=r"\(\+2 more\)"):
        fleet.start_next_wave(deployment.deployment_id)
    assert fleet.store.get_fleet_deployment(deployment.deployment_id) == cancelled


@pytest.mark.parametrize("failure", ["prepare", "generation", "commit"])
def test_barrier_never_commits_an_unprepared_or_refenced_participant(
    failure: str,
) -> None:
    fleet = registry()
    coordinator = BarrierCoordinator(fleet.store, now=lambda: NOW)
    barrier = coordinator.create(
        barrier_id="unit-barrier",
        cluster_id="cluster-a",
        workflow_request_id="unit-workflow",
        incident_id="unit-incident",
        fencing_token=1,
        operation=WorkflowOperation.RESET_GPU,
        generations={"node-a": 1, "node-b": 1},
    )
    with pytest.raises(ValueError, match="not a barrier participant"):
        coordinator.record_prepare(barrier.barrier_id, "foreign")
    assert (
        coordinator.begin_commit(barrier.barrier_id, {"node-a": 1, "node-b": 1}).state
        is BarrierState.PREPARING
    )
    coordinator.record_prepare(barrier.barrier_id, "node-a")
    prepared = coordinator.record_prepare(
        barrier.barrier_id,
        "node-b",
        error="unit prepare failure" if failure == "prepare" else None,
    )
    if failure == "prepare":
        assert prepared.state is BarrierState.ABORTED
        assert coordinator.record_prepare(barrier.barrier_id, "node-b") == prepared
    else:
        committing = coordinator.begin_commit(
            barrier.barrier_id,
            {"node-a": 2 if failure == "generation" else 1, "node-b": 1},
        )
        if failure == "generation":
            assert committing.state is BarrierState.ABORTED
        else:
            coordinator.record_commit(barrier.barrier_id, "node-a")
            result = coordinator.record_commit(
                barrier.barrier_id, "node-b", error="unit commit failure"
            )
            assert result.state is BarrierState.FAILED

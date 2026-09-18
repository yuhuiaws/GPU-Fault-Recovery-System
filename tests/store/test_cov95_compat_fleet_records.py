from __future__ import annotations

from datetime import timedelta

import pytest

from gpu_fault.fleet import AgentRecord
from gpu_fault.fleet_deployment import DeploymentNode, DeploymentStatus, FleetDeployment
from gpu_fault.regional import RegionalRegistryMember, RegionalRegistryRevision
from gpu_fault.store import NotFoundError
from tests.store._cov95_compat_support import NOW
from tests.store._cov95_compat_support import (
    compat_store_fixture as compat_store_fixture,
)
from tests.store.test_regional_registry_revisions import registration


def agent(cluster, node="node"):
    return AgentRecord(
        cluster_id=cluster,
        node_id=node,
        endpoint="https://192.0.2.1:9099",
        agent_protocol_version=3,
        node_action_key_version=2,
        agent_version="0.10.0",
        artifact_sha256="1" * 64,
        policy_version="compat",
        runtime_profile_version="compat",
        config_digest="2" * 64,
        allowed_operations=[],
        first_seen_at=NOW,
        last_seen_at=NOW,
        lease_expires_at=NOW + timedelta(minutes=5),
    )


def deployment(name, *, cluster="cluster", status=DeploymentStatus.PLANNED, at=NOW):
    return FleetDeployment(
        deployment_id=name,
        cluster_id=cluster,
        desired_agent_version="0.10.0",
        desired_artifact_sha256="1" * 64,
        desired_policy_version="compat",
        desired_runtime_profile_version="compat",
        desired_config_digest="2" * 64,
        max_unavailable=1,
        waves=[["node"]],
        nodes=[DeploymentNode(node_id="node", updated_at=at)],
        status=status,
        created_at=at,
        updated_at=at,
    )


@pytest.mark.parametrize(
    "method,args",
    [
        ("get_regional_cluster", ("missing",)),
        ("get_regional_registry_head", ()),
        ("get_regional_registry_revision", (1,)),
        ("get_agent", ("missing", "node")),
        ("get_fleet_deployment", ("missing",)),
        ("get_barrier", ("missing",)),
    ],
)
def test_missing_fleet_records_are_not_synthesized(compat_store, method, args):
    with pytest.raises(NotFoundError):
        getattr(compat_store, method)(*args)
    assert compat_store.list_regional_clusters() == []
    assert compat_store.list_agents() == []
    assert compat_store.list_fleet_deployments() == []
    assert compat_store.list_barriers() == []


def test_agent_compare_and_set_preserves_foreign_and_newer_records(compat_store):
    store = compat_store
    first, foreign = agent("cluster-a"), agent("cluster-b")
    assert store.replace_agent_if_matches(first, None) is True
    assert store.replace_agent_if_matches(foreign, None) is True
    replacement = first.model_copy(update={"last_seen_at": NOW + timedelta(seconds=1)})
    assert store.replace_agent_if_matches(replacement, None) is False
    assert store.replace_agent_if_matches(replacement, first) is True
    assert store.replace_agent_if_matches(first, first) is False
    cross_tenant = replacement.model_copy(update={"cluster_id": "cluster-b"})
    assert store.replace_agent_if_matches(cross_tenant, replacement) is False
    assert store.get_agent("cluster-a", "node") == replacement
    assert store.get_agent("cluster-b", "node") == foreign
    assert store.list_agents("cluster-a") == [replacement]


def test_fleet_deployment_cas_and_retention_keep_live_waves(compat_store):
    store = compat_store
    planned = deployment("planned")
    assert store.replace_fleet_deployment_if_matches(planned, None) is True
    running = planned.model_copy(update={"status": DeploymentStatus.IN_PROGRESS})
    assert store.replace_fleet_deployment_if_matches(running, planned) is True
    assert store.replace_fleet_deployment_if_matches(planned, planned) is False
    assert store.replace_fleet_deployment_if_matches(planned, None) is False
    ended = deployment(
        "ended", status=DeploymentStatus.SUCCEEDED, at=NOW - timedelta(days=3)
    )
    recent = deployment("recent", status=DeploymentStatus.FAILED, at=NOW)
    foreign = deployment("foreign", cluster="other")
    for item in (ended, recent, foreign):
        store.save_fleet_deployment(item)
    assert store.list_active_fleet_deployments("cluster") == [running]
    assert store.list_active_fleet_deployments("other") == [foreign]
    assert (
        store.cleanup_terminal_fleet_deployments(
            older_than=NOW - timedelta(days=1), limit=1
        )
        == 1
    )
    with pytest.raises(NotFoundError):
        store.get_fleet_deployment("ended")
    assert store.get_fleet_deployment("planned") == running
    assert store.get_fleet_deployment("recent") == recent


def test_registry_publish_rejects_missing_generation_without_partial_installation(
    compat_store,
):
    store = compat_store
    invalid_chain = RegionalRegistryRevision.build(
        generation=2,
        previous_generation=1,
        registrations=[registration("cluster-a")],
        required_member_ids=[],
        reason="compatibility gap",
        created_at=NOW,
    )
    with pytest.raises(ValueError, match="consecutive"):
        store.publish_regional_registry_revision(invalid_chain, expected_generation=0)
    assert store.list_regional_cluster_ids() == []
    with pytest.raises(NotFoundError):
        store.get_regional_registry_head()
    first = RegionalRegistryRevision.build(
        generation=1,
        previous_generation=None,
        registrations=[registration("cluster-a")],
        required_member_ids=[],
        reason="compatibility bootstrap",
        created_at=NOW,
    )
    head = store.publish_regional_registry_revision(first, expected_generation=0)
    assert head.generation == 1
    assert store.list_regional_cluster_ids() == ["cluster-a"]


def test_member_cleanup_is_bounded_and_preserves_new_acknowledgements(compat_store):
    store = compat_store
    for index in range(3):
        store.save_regional_registry_member(
            RegionalRegistryMember(
                member_id=f"member-{index}",
                service_role="ingress",
                release_id="compat",
                generation=1,
                content_sha256="1" * 64,
                ready=True,
                started_at=NOW - timedelta(minutes=1),
                last_seen_at=NOW + timedelta(seconds=index),
            )
        )
    assert (
        store.cleanup_stale_regional_registry_members(
            older_than=NOW + timedelta(seconds=1), limit=1
        )
        == 1
    )
    assert [item.member_id for item in store.list_regional_registry_members()] == [
        "member-1",
        "member-2",
    ]
    assert (
        store.cleanup_stale_regional_registry_members(
            older_than=NOW + timedelta(seconds=1), limit=1
        )
        == 1
    )
    assert (
        store.cleanup_stale_regional_registry_members(
            older_than=NOW + timedelta(seconds=1), limit=1
        )
        == 0
    )
    assert [item.member_id for item in store.list_regional_registry_members()] == [
        "member-2"
    ]

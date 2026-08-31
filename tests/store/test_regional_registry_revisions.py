from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from gpu_fault.fleet import AgentRecord
from gpu_fault.regional import (
    RegionalClusterLifecycle,
    RegionalClusterRegistration,
    RegionalRegistryMember,
    RegionalRegistryRevision,
)
from gpu_fault.store import InMemoryStore, SqliteStore

NOW = datetime(2026, 8, 31, tzinfo=timezone.utc)


def registration(
    cluster_id: str,
    *,
    region: str = "us-west-2",
    lifecycle_state: RegionalClusterLifecycle = RegionalClusterLifecycle.ACTIVE,
) -> RegionalClusterRegistration:
    return RegionalClusterRegistration(
        cluster_id=cluster_id,
        region=region,
        hyperpod_cluster_name=f"hyperpod-{cluster_id}",
        eks_cluster_arn=(f"arn:aws:eks:{region}:123456789012:cluster/{cluster_id}"),
        token_sha256="a" * 64,
        lifecycle_state=lifecycle_state,
        allowed_namespaces=["training"],
        agent_endpoint_allowed_cidrs=["10.0.0.0/16"],
        created_at=NOW,
        updated_at=NOW,
    )


def store_for(backend: str, tmp_path):
    if backend == "memory":
        return InMemoryStore()
    return SqliteStore(str(tmp_path / "registry.sqlite"))


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_registry_revision_publish_is_consecutive_and_idempotent(
    backend: str, tmp_path
) -> None:
    store = store_for(backend, tmp_path)
    first = RegionalRegistryRevision.build(
        generation=1,
        registrations=[registration("cluster-a")],
        previous_generation=None,
        required_member_ids=["worker-a", "ingress-a"],
        reason="bootstrap",
        created_at=NOW,
    )

    head = store.publish_regional_registry_revision(first, expected_generation=0)
    replay = store.publish_regional_registry_revision(first, expected_generation=0)

    assert replay == head
    assert store.get_regional_registry_head() == head
    assert store.get_regional_registry_revision(1) == first
    assert [item.cluster_id for item in store.list_regional_clusters()] == ["cluster-a"]

    conflicting = RegionalRegistryRevision.build(
        generation=2,
        registrations=[],
        previous_generation=1,
        required_member_ids=[],
        reason="conflicting writer",
        created_at=NOW + timedelta(seconds=1),
    )
    with pytest.raises(ValueError, match="generation conflict"):
        store.publish_regional_registry_revision(conflicting, expected_generation=0)
    if hasattr(store, "close"):
        store.close()


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_registry_revision_removal_does_not_delete_agent_until_cleanup(
    backend: str, tmp_path
) -> None:
    store = store_for(backend, tmp_path)
    first = RegionalRegistryRevision.build(
        generation=1,
        registrations=[registration("cluster-a")],
        previous_generation=None,
        required_member_ids=[],
        reason="bootstrap",
        created_at=NOW,
    )
    store.publish_regional_registry_revision(first, expected_generation=0)
    store.save_agent(
        AgentRecord(
            cluster_id="cluster-a",
            node_id="node-a",
            endpoint="https://10.0.0.10:9099",
            agent_protocol_version=3,
            node_action_key_version=2,
            agent_version="0.10.0",
            artifact_sha256="b" * 64,
            policy_version="610",
            runtime_profile_version="profile-a",
            config_digest="c" * 64,
            allowed_operations=[],
            first_seen_at=NOW,
            last_seen_at=NOW,
            lease_expires_at=NOW + timedelta(minutes=5),
        )
    )
    second = RegionalRegistryRevision.build(
        generation=2,
        registrations=[],
        previous_generation=1,
        required_member_ids=[],
        reason="revoke cluster",
        created_at=NOW + timedelta(seconds=1),
    )

    store.publish_regional_registry_revision(second, expected_generation=1)

    assert store.list_regional_clusters() == []
    assert [item.node_id for item in store.list_agents("cluster-a")] == ["node-a"]
    if hasattr(store, "close"):
        store.close()


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_registry_revision_forbids_region_drift(backend: str, tmp_path) -> None:
    store = store_for(backend, tmp_path)
    first = RegionalRegistryRevision.build(
        generation=1,
        registrations=[registration("cluster-a")],
        previous_generation=None,
        required_member_ids=[],
        reason="bootstrap",
        created_at=NOW,
    )
    store.publish_regional_registry_revision(first, expected_generation=0)
    drifted = RegionalRegistryRevision.build(
        generation=2,
        registrations=[registration("cluster-a", region="us-east-1")],
        previous_generation=1,
        required_member_ids=[],
        reason="invalid region drift",
        created_at=NOW + timedelta(seconds=1),
    )

    with pytest.raises(ValueError, match="cannot move between regions"):
        store.publish_regional_registry_revision(drifted, expected_generation=1)
    assert store.get_regional_registry_head().generation == 1
    if hasattr(store, "close"):
        store.close()


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_registry_member_ack_is_overwritten_by_member_identity(
    backend: str, tmp_path
) -> None:
    store = store_for(backend, tmp_path)
    first = RegionalRegistryMember(
        member_id="pod-a/process-a",
        service_role="ingress",
        release_id="release-a",
        generation=1,
        content_sha256="a" * 64,
        ready=True,
        started_at=NOW,
        last_seen_at=NOW,
    )
    second = first.model_copy(
        update={
            "generation": 2,
            "content_sha256": "b" * 64,
            "last_seen_at": NOW + timedelta(seconds=1),
        }
    )

    store.save_regional_registry_member(first)
    store.save_regional_registry_member(second)

    assert store.list_regional_registry_members() == [second]
    if hasattr(store, "close"):
        store.close()


def test_registry_revision_digest_and_generation_are_validated() -> None:
    with pytest.raises(ValidationError, match="content digest mismatch"):
        RegionalRegistryRevision(
            generation=1,
            content_sha256="0" * 64,
            registrations=[registration("cluster-a")],
            required_member_ids=[],
            reason="invalid digest",
            created_at=NOW,
        )
    with pytest.raises(ValidationError, match="generation must increase"):
        RegionalRegistryRevision.build(
            generation=2,
            registrations=[],
            previous_generation=2,
            required_member_ids=[],
            reason="invalid rollback",
            created_at=NOW,
        )


def test_registration_lifecycle_separates_token_recognition_from_active_use() -> None:
    token = "t" * 32
    digest = __import__("hashlib").sha256(token.encode()).hexdigest()
    pending = registration(
        "cluster-a", lifecycle_state=RegionalClusterLifecycle.PENDING
    ).model_copy(update={"token_sha256": digest})
    revoked = pending.model_copy(
        update={"lifecycle_state": RegionalClusterLifecycle.REVOKED}
    )

    assert pending.token_matches(token), pending
    assert not pending.authenticates(token), pending
    assert not revoked.token_matches(token), revoked

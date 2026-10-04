"""Fleet pins resolved per cluster from the regional registration.

A cluster that runs its own Runtime Profile registers with
``runtime_profile_version`` / ``agent_config_digest``; its Agents are held to
those pins while every other cluster keeps the site pins
(``GPU_FAULT_REQUIRED_RUNTIME_PROFILE_VERSION`` / ``..._AGENT_CONFIG_DIGEST``).
"""

from __future__ import annotations

from datetime import datetime, timezone

from gpu_fault.regional import RegionalClusterRegistration
from tests.fleet._support import CONFIG, NOW, heartbeat, registry, signed

CLUSTER_B_DIGEST = "b" * 64


def _registration(cluster_id: str, **pins: str) -> RegionalClusterRegistration:
    return RegionalClusterRegistration(
        cluster_id=cluster_id,
        region="us-east-1",
        hyperpod_cluster_name="" if pins else f"hp-{cluster_id}",
        eks_cluster_arn=f"arn:aws:eks:us-east-1:123456789012:cluster/{cluster_id}",
        token_sha256="1" * 64,
        agent_endpoint_allowed_cidrs=["10.0.0.0/16"],
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        updated_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        **pins,
    )


def _cluster_b(**updates):
    return heartbeat("node-b").model_copy(
        update={
            "cluster_id": "cluster-b",
            "endpoint": "http://node-b:9099",
            "runtime_profile_version": "profile-b",
            "config_digest": CLUSTER_B_DIGEST,
            **updates,
        }
    )


def test_an_overriding_cluster_is_held_to_its_own_pins() -> None:
    fleet = registry()
    fleet.store.save_regional_cluster(_registration("cluster-a"))
    fleet.store.save_regional_cluster(
        _registration(
            "cluster-b",
            provider="eks-managed",
            runtime_profile_version="profile-b",
            agent_config_digest=CLUSTER_B_DIGEST,
        )
    )
    fleet.register(signed(heartbeat("node-a")))
    fleet.register(signed(_cluster_b()))

    assert fleet.readiness("cluster-a", ["node-a"]).ready, (
        "the HyperPod cluster on the site profile is ready against the site pins"
    )
    assert fleet.readiness("cluster-b", ["node-b"]).ready, (
        "the plain EKS cluster is ready against its own registration pins"
    )
    policy_a = fleet.cluster_policy("cluster-a")
    policy_b = fleet.cluster_policy("cluster-b")
    assert policy_a.required_runtime_profile_version == "profile-a", (
        "a cluster without pins keeps the site profile version"
    )
    assert policy_a.required_config_digest == CONFIG, "and the site Agent config digest"
    assert policy_b.required_runtime_profile_version == "profile-b", "own version"
    assert policy_b.required_config_digest == CLUSTER_B_DIGEST, "own digest"
    assert policy_b.required_artifact_sha256 == policy_a.required_artifact_sha256, (
        "every other pin stays the site's"
    )


def test_an_agent_matching_another_cluster_pins_is_still_refused() -> None:
    fleet = registry()
    fleet.store.save_regional_cluster(_registration("cluster-a"))
    fleet.store.save_regional_cluster(
        _registration(
            "cluster-b",
            provider="eks-managed",
            runtime_profile_version="profile-b",
            agent_config_digest=CLUSTER_B_DIGEST,
        )
    )
    # A cluster-a Agent that reports cluster-b's profile version and digest.
    fleet.register(
        signed(
            heartbeat("node-a").model_copy(
                update={
                    "runtime_profile_version": "profile-b",
                    "config_digest": CLUSTER_B_DIGEST,
                }
            )
        )
    )
    report = fleet.readiness("cluster-a", ["node-a"])
    assert not report.ready, "cluster-a Agents are held to the site pins"
    reasons = " ".join(report.nodes[0].reasons)
    assert "runtime profile mismatch: expected profile-a, got profile-b" in reasons, (
        reasons
    )
    # And a cluster-b Agent on the site profile is refused the other way round.
    fleet.register(signed(_cluster_b(runtime_profile_version="profile-a")))
    report = fleet.readiness("cluster-b", ["node-b"])
    assert not report.ready, "cluster-b Agents are held to their own pins"
    assert "runtime profile mismatch: expected profile-b, got profile-a" in " ".join(
        report.nodes[0].reasons
    ), report.nodes[0].reasons


def test_pin_drift_is_classified_against_the_cluster_pins() -> None:
    fleet = registry()
    fleet.store.save_regional_cluster(
        _registration(
            "cluster-b",
            provider="eks-managed",
            runtime_profile_version="profile-b",
            agent_config_digest=CLUSTER_B_DIGEST,
        )
    )
    fleet.register(signed(_cluster_b()))
    stale = _cluster_b(runtime_profile_version="profile-old").model_copy(
        update={"node_id": "node-c", "endpoint": "http://node-c:9099"}
    )
    fleet.register(signed(stale))
    report = fleet.readiness("cluster-b", ["node-b", "node-c"])
    assert not report.ready, "the stale node blocks the wave"
    assert fleet.pin_drift_nodes["cluster-b"] == {
        "NODE_STALE": 1,
        "PIN_AHEAD_OF_FLEET": 0,
    }, "node-b already runs the cluster's pinned profile, so node-c is stale"


def test_registry_without_a_regional_entry_falls_back_to_the_site_pins() -> None:
    fleet = registry()
    fleet.register(signed(heartbeat("node-a")))
    assert fleet.readiness("cluster-a", ["node-a"]).ready, (
        "an unregistered cluster is held to the site pins as before"
    )
    assert fleet.cluster_policy("cluster-a") is fleet.policy, (
        "no override means the very same site policy object"
    )


def test_cluster_pins_are_cached_briefly_and_refreshed() -> None:
    clock = [1000.0]
    fleet = registry()
    import gpu_fault.fleet as fleet_module

    original = fleet_module.time.monotonic
    fleet_module.time.monotonic = lambda: clock[0]  # type: ignore[assignment]
    try:
        fleet.store.save_regional_cluster(_registration("cluster-a"))
        assert fleet.cluster_policy("cluster-a").required_runtime_profile_version == (
            "profile-a"
        ), "no pins yet"
        fleet.store.save_regional_cluster(
            _registration(
                "cluster-a",
                provider="eks-managed",
                runtime_profile_version="profile-b",
                agent_config_digest=CLUSTER_B_DIGEST,
            )
        )
        assert fleet.cluster_policy("cluster-a").required_runtime_profile_version == (
            "profile-a"
        ), "within the cache window the earlier answer is reused"
        clock[0] += fleet_module.CLUSTER_POLICY_CACHE_SECONDS + 1
        assert fleet.cluster_policy("cluster-a").required_runtime_profile_version == (
            "profile-b"
        ), "after the window the registry is re-read"
    finally:
        fleet_module.time.monotonic = original  # type: ignore[assignment]
    assert NOW.tzinfo is not None, "fixtures stay timezone-aware"

from __future__ import annotations

import ipaddress
from datetime import timedelta

import pytest

from gpu_fault.fleet import (
    NODE_ACTION_KEY_VERSION_DERIVED,
    AgentLifecycleState,
    AgentTransitionRequest,
    BarrierCoordinator,
    BarrierState,
    DeploymentNodeStatus,
    DeploymentNodeUpdate,
    DeploymentStatus,
    FleetCompatibilityPolicy,
    FleetDeploymentRequest,
    FleetRegistry,
    SignedAgentHeartbeat,
    derive_node_action_secret,
    parse_endpoint_networks,
    sign_agent_heartbeat,
)
from gpu_fault.models import WorkflowExecutionRequest, WorkflowOperation, WorkflowStatus
from gpu_fault.runtime_adapters import NodeActionWorkflowAdapter
from gpu_fault.store import NotFoundError, SqliteStore
from tests._builders import (
    active_workflow_executor,
    build_store,
    copy_model,
    node_action_result,
)
from tests.fleet._support import (
    ARTIFACT,
    CONFIG,
    NOW,
    SECRET,
    heartbeat,
    quiesce_then_full_reset_workflow,
    registry,
    signed,
    workflow_state,
)


def test_unknown_collector_unit_is_verified_then_discarded() -> None:
    unit = "gpu-fault-future-collector"
    value = heartbeat(
        "node-a", collector_services={unit: {"active": "active", "enabled": "enabled"}}
    )

    assert unit in value.collector_services
    record = registry().register(signed(value))
    assert unit not in record.collector_services


def test_derived_node_key_cannot_sign_for_another_node() -> None:
    fleet = registry()
    value = copy_model(
        heartbeat("node-a"), node_action_key_version=NODE_ACTION_KEY_VERSION_DERIVED
    )
    node_a_secret = derive_node_action_secret(SECRET, value.cluster_id, value.node_id)
    accepted = fleet.register(
        SignedAgentHeartbeat(
            heartbeat=value, signature=sign_agent_heartbeat(value, node_a_secret)
        )
    )

    assert accepted.node_action_key_version == NODE_ACTION_KEY_VERSION_DERIVED

    forged_secret = derive_node_action_secret(SECRET, value.cluster_id, "node-b")
    with pytest.raises(ValueError, match="signature"):
        fleet.register(
            SignedAgentHeartbeat(
                heartbeat=copy_model(
                    value, observed_at=value.observed_at + timedelta(seconds=1)
                ),
                signature=sign_agent_heartbeat(
                    copy_model(
                        value, observed_at=value.observed_at + timedelta(seconds=1)
                    ),
                    forged_secret,
                ),
            )
        )


def test_required_derived_key_rejects_shared_heartbeat() -> None:
    store = build_store()
    fleet = FleetRegistry(
        store,
        SECRET,
        FleetCompatibilityPolicy(
            required_node_action_key_version=(NODE_ACTION_KEY_VERSION_DERIVED),
            required_agent_version="0.9.0",
            required_artifact_sha256=ARTIFACT,
            required_policy_version="catalog-a",
            required_runtime_profile_version="profile-a",
            required_config_digest=CONFIG,
        ),
        now=lambda: NOW,
    )
    legacy = heartbeat("node-a")

    with pytest.raises(ValueError, match="key version is not accepted"):
        fleet.register(
            SignedAgentHeartbeat(
                heartbeat=legacy, signature=sign_agent_heartbeat(legacy, SECRET)
            )
        )
    with pytest.raises(NotFoundError):
        store.get_agent(legacy.cluster_id, legacy.node_id)


def test_node_specific_key_can_rotate_without_changing_peer() -> None:
    store = build_store()
    node_a_key = "a" * 64
    node_b_key = "b" * 64
    fleet = FleetRegistry(
        store,
        SECRET,
        FleetCompatibilityPolicy(
            required_node_action_key_version=(NODE_ACTION_KEY_VERSION_DERIVED)
        ),
        node_secrets={"node-a": node_a_key, "node-b": node_b_key},
        now=lambda: NOW,
    )
    node_a = copy_model(
        heartbeat("node-a"), node_action_key_version=NODE_ACTION_KEY_VERSION_DERIVED
    )
    node_b = copy_model(
        heartbeat("node-b", boot_id="boot-b"),
        node_action_key_version=NODE_ACTION_KEY_VERSION_DERIVED,
    )
    fleet.register(
        SignedAgentHeartbeat(
            heartbeat=node_a, signature=sign_agent_heartbeat(node_a, node_a_key)
        )
    )
    fleet.register(
        SignedAgentHeartbeat(
            heartbeat=node_b, signature=sign_agent_heartbeat(node_b, node_b_key)
        )
    )

    rotated = "c" * 64
    fleet.node_secrets["node-a"] = rotated
    next_a = copy_model(node_a, observed_at=NOW + timedelta(seconds=1))
    next_b = copy_model(node_b, observed_at=NOW + timedelta(seconds=1))

    with pytest.raises(ValueError, match="signature"):
        fleet.register(
            SignedAgentHeartbeat(
                heartbeat=next_a, signature=sign_agent_heartbeat(next_a, node_a_key)
            )
        )
    fleet.register(
        SignedAgentHeartbeat(
            heartbeat=next_a, signature=sign_agent_heartbeat(next_a, rotated)
        )
    )
    fleet.register(
        SignedAgentHeartbeat(
            heartbeat=next_b, signature=sign_agent_heartbeat(next_b, node_b_key)
        )
    )


def test_agent_endpoint_cannot_point_the_control_plane_elsewhere() -> None:
    """The advertised endpoint is where signed commands get POSTed.

    One shared secret covers the whole cluster, so any holder of it used
    to be able to register an endpoint of its choosing and have the
    control plane call it: the instance metadata service, an internal
    API on another port, or a listener that just records the signed
    command. Scheme-only validation stopped none of that.
    """
    fleet = registry()
    accepted = [
        # The node's own Kubernetes name, which is the installer default.
        "http://node-a:9099",
        # A private address, which is what a node behind a name the
        # control plane cannot resolve has to advertise.
        "http://10.1.2.3:9099",
        # EC2 private DNS: the name itself proves the address, so it is
        # accepted even though the Kubernetes node name is node-a.
        "http://ip-10-1-2-3.us-west-2.compute.internal:9099",
    ]
    rejected = {
        "http://169.254.169.254:9099": "does not address node",
        "http://metadata.internal:9099": "does not address node",
        "http://127.0.0.1:9099": "does not address node",
        "http://node-a:80": "port 80 is not one of",
        "http://node-a": "port 80 is not one of",
        "http://node-a:9099/v1/node-actions": "bare scheme",
        "http://node-a:9099?x=1": "bare scheme",
        "http://attacker@node-a:9099": "must not carry credentials",
    }

    for endpoint in accepted:
        value = copy_model(
            heartbeat("node-a"),
            heartbeat_id=f"heartbeat-{endpoint}",
            endpoint=endpoint,
            agent_incarnation_id=f"incarnation-{endpoint}",
        )
        assert registry().register(signed(value)).endpoint == endpoint, endpoint

    for endpoint, expected in rejected.items():
        value = copy_model(
            heartbeat("node-a"), heartbeat_id=f"heartbeat-{endpoint}", endpoint=endpoint
        )
        with pytest.raises(ValueError, match=expected):
            fleet.register(signed(value))

    # A registered node cannot be re-pointed later either: the endpoint
    # is checked on every heartbeat, not only on the first one.
    fleet.register(signed(heartbeat("node-a")))
    with pytest.raises(ValueError, match="does not address node"):
        fleet.register(
            signed(
                copy_model(
                    heartbeat("node-a"),
                    heartbeat_id="heartbeat-moved",
                    endpoint="http://169.254.169.254:9099",
                    observed_at=NOW + timedelta(seconds=30),
                )
            )
        )

    # An operator whose agents answer on another name says so once.
    widened = FleetRegistry(
        build_store(),
        SECRET,
        now=lambda: NOW,
        endpoint_allowed_ports=frozenset({8443}),
        endpoint_allowed_host_suffixes=(".agents.example.com",),
    )
    assert (
        widened.register(
            signed(
                copy_model(
                    heartbeat("node-a"),
                    endpoint="https://node-a.agents.example.com:8443",
                )
            )
        ).endpoint
        == "https://node-a.agents.example.com:8443"
    )


def test_agent_endpoint_can_be_confined_to_the_node_subnets() -> None:
    """Private-ness alone lets a node point at its neighbours.

    A HyperPod node id (hyperpod-i-00000000000000001) encodes no
    address, so once the scheme, port and shape are checked the only
    remaining test is "is it private" -- and every RFC1918 host in the
    VPC passes that, including another node's agent port and service
    ClusterIPs like the Kubernetes API at 172.20.0.1. Naming the node
    subnets closes that; leaving the list empty keeps the old behaviour,
    because a wrong CIDR would reject every heartbeat in the fleet.
    """
    node = "hyperpod-i-00000000000000001"

    def register(cidrs: str, endpoint: str) -> str:
        # A fresh registry per call: registering the same node twice
        # with different incarnations trips the node lease, which would
        # mask what the endpoint check itself did.
        fleet = FleetRegistry(
            build_store(),
            SECRET,
            now=lambda: NOW,
            endpoint_allowed_networks=parse_endpoint_networks(cidrs),
        )
        return fleet.register(
            signed(copy_model(heartbeat(node), endpoint=endpoint))
        ).endpoint

    # Unset: today's behaviour, so an upgrade cannot fence a fleet.
    assert register("", "http://172.20.0.1:9099") == "http://172.20.0.1:9099"

    # Set: only the node subnets are addressable.
    subnets = "10.0.0.0/16, 10.1.0.0/16"
    assert register(subnets, "http://10.0.48.46:9099") == "http://10.0.48.46:9099"
    assert register(subnets, "http://10.1.7.87:9099") == "http://10.1.7.87:9099"
    for endpoint in (
        "http://172.20.0.1:9099",  # Kubernetes API ClusterIP
        "http://10.2.0.5:9099",  # in-VPC but not a node subnet
        "http://192.168.1.1:9099",
    ):
        with pytest.raises(
            ValueError, match="outside GPU_FAULT_AGENT_ENDPOINT_ALLOWED_CIDRS"
        ):
            register(subnets, endpoint)

    # The EC2 private-DNS shortcut has to honour the same list, or the
    # name form becomes a way around it.
    with pytest.raises(
        ValueError, match="outside GPU_FAULT_AGENT_ENDPOINT_ALLOWED_CIDRS"
    ):
        register(subnets, "http://ip-172-20-0-1.us-west-2.compute.internal:9099")
    assert (
        register(subnets, "http://ip-10-0-48-47.us-west-2.compute.internal:9099")
        == "http://ip-10-0-48-47.us-west-2.compute.internal:9099"
    )

    # A single host is a legitimate allow-list entry, and a malformed
    # one has to fail at parse time -- at process start, not one
    # heartbeat at a time.
    assert parse_endpoint_networks("10.0.48.46") == (
        ipaddress.ip_network("10.0.48.46/32"),
    )
    with pytest.raises(ValueError):
        parse_endpoint_networks("10.0.0.0/16,not-a-cidr")


def test_agent_registration_is_signed_and_generation_is_fenced() -> None:
    fleet = registry()
    value = heartbeat("node-a")

    record = fleet.register(signed(value))
    repeated = fleet.register(
        signed(
            copy_model(
                value,
                heartbeat_id="heartbeat-2",
                observed_at=NOW + timedelta(seconds=10),
            )
        )
    )
    rebooted_value = copy_model(
        value,
        heartbeat_id="heartbeat-3",
        boot_id="boot-b",
        observed_at=NOW + timedelta(seconds=20),
    )
    rebooted = fleet.register(signed(rebooted_value))

    assert record.generation == 1
    assert repeated.generation == 1
    assert rebooted.generation == 2

    invalid = copy_model(signed(value), signature="0" * 64)
    try:
        fleet.register(invalid)
    except ValueError as exc:
        assert "signature" in str(exc)
    else:
        raise AssertionError("invalid signature was accepted")


def test_node_lease_fences_replaced_and_retired_agent() -> None:
    current = [NOW]
    fleet = registry(now=lambda: current[0])
    original = heartbeat("node-a")
    fleet.register(signed(original))
    replacement = heartbeat(
        "node-a",
        boot_id="boot-b",
        node_instance_id="instance-b",
        observed_at=NOW + timedelta(seconds=10),
    )

    current[0] = NOW + timedelta(seconds=10)
    with pytest.raises(ValueError, match="holds the node lease"):
        fleet.register(signed(replacement))

    current[0] = NOW + timedelta(seconds=91)
    replacement = copy_model(replacement, observed_at=current[0])
    accepted = fleet.register(signed(replacement))

    assert accepted.generation == 2
    assert accepted.agent_incarnation_id == "boot-b"
    assert "boot-a" in accepted.retired_incarnation_ids

    current[0] = NOW + timedelta(seconds=92)
    delayed_old = copy_model(original, observed_at=current[0])
    with pytest.raises(ValueError, match="retired"):
        fleet.register(signed(delayed_old))


def test_agent_drain_revoke_and_new_incarnation_activation() -> None:
    current = [NOW]
    fleet = registry(now=lambda: current[0])
    original = fleet.register(signed(heartbeat("node-a")))
    transition = AgentTransitionRequest(
        expected_generation=original.generation,
        transition_id="workflow-a/restart-node",
        reason="planned HyperPod node reboot",
    )

    drained = fleet.drain_agent("cluster-a", "node-a", transition)
    repeated = fleet.drain_agent("cluster-a", "node-a", transition)

    assert drained.lifecycle_state is AgentLifecycleState.DRAINING
    assert drained.generation == 2
    assert repeated == drained
    assert not fleet.readiness("cluster-a", ["node-a"]).ready, (
        'expected fleet.readiness("cluster-a", ["node-a"]).ready to be falsy'
    )

    current[0] = NOW + timedelta(seconds=10)
    renewed = heartbeat("node-a", observed_at=current[0])
    still_draining = fleet.register(signed(renewed))
    assert still_draining.lifecycle_state is AgentLifecycleState.DRAINING

    revoked = fleet.revoke_agent("cluster-a", "node-a", transition)
    assert revoked.lifecycle_state is AgentLifecycleState.REVOKED
    assert revoked.lease_expires_at == current[0]

    current[0] = NOW + timedelta(seconds=20)
    replacement = heartbeat("node-a", boot_id="boot-b", observed_at=current[0])
    activated = fleet.register(signed(replacement))

    assert activated.lifecycle_state is AgentLifecycleState.ACTIVE
    assert activated.generation == 3
    assert activated.transition_id is None
    assert "boot-a" in activated.retired_incarnation_ids


def test_revoked_agent_requires_explicit_matching_reactivation() -> None:
    current = [NOW]
    fleet = registry(now=lambda: current[0])
    original = fleet.register(signed(heartbeat("node-a")))
    transition = AgentTransitionRequest(
        expected_generation=original.generation,
        transition_id="workflow-a/restart-node",
        reason="planned HyperPod node reboot",
    )
    drained = fleet.drain_agent("cluster-a", "node-a", transition)
    revoked = fleet.revoke_agent(
        "cluster-a",
        "node-a",
        copy_model(transition, expected_generation=drained.generation),
    )

    with pytest.raises(ValueError, match="stale agent generation"):
        fleet.reactivate_agent("cluster-a", "node-a", transition)
    with pytest.raises(ValueError, match="revoking transition"):
        fleet.reactivate_agent(
            "cluster-a",
            "node-a",
            AgentTransitionRequest(
                expected_generation=revoked.generation,
                transition_id="workflow-b/restart-node",
                reason="unrelated workflow",
            ),
        )

    reactivated = fleet.reactivate_agent(
        "cluster-a",
        "node-a",
        AgentTransitionRequest(
            expected_generation=revoked.generation,
            transition_id=transition.transition_id,
            reason="operator verified original node is healthy",
        ),
    )

    assert reactivated.lifecycle_state is AgentLifecycleState.ACTIVE
    assert reactivated.transition_id is None
    assert "boot-a" not in reactivated.retired_incarnation_ids
    assert not fleet.readiness("cluster-a", ["node-a"]).ready, (
        'expected fleet.readiness("cluster-a", ["node-a"]).ready to be falsy'
    )

    current[0] = NOW + timedelta(seconds=1)
    renewed = fleet.register(
        signed(
            heartbeat("node-a", observed_at=current[0], agent_incarnation_id="boot-a")
        )
    )
    assert renewed.lifecycle_state is AgentLifecycleState.ACTIVE
    assert fleet.readiness("cluster-a", ["node-a"]).ready, (
        'expected fleet.readiness("cluster-a", ["node-a"]).ready to be truthy'
    )


def test_readiness_blocks_stale_and_mixed_agent_versions() -> None:
    current = [NOW]
    fleet = registry(now=lambda: current[0])
    fleet.register(signed(heartbeat("node-a")))
    fleet.register(signed(heartbeat("node-b", version="0.9.1")))

    mixed = fleet.readiness("cluster-a", ["node-a", "node-b"])
    current[0] = NOW + timedelta(seconds=100)
    stale = fleet.readiness("cluster-a", ["node-a"])

    assert not mixed.ready, "expected mixed.ready to be falsy"
    assert any(
        "version mismatch" in reason for node in mixed.nodes for reason in node.reasons
    ), (
        'expected any( "version mismatch" in reason for node in mixed.nodes for reason in node.reasons ) to be truthy'
    )
    assert not stale.ready, "expected stale.ready to be falsy"
    assert any("stale" in reason for reason in stale.nodes[0].reasons), (
        'expected any("stale" in reason for reason in stale.nodes[0].reasons) to be truthy'
    )


def test_readiness_blocks_legacy_agent_protocol() -> None:
    fleet = registry()
    legacy = copy_model(heartbeat("node-a"), agent_protocol_version=1)
    fleet.register(signed(legacy))

    report = fleet.readiness("cluster-a", ["node-a"])

    assert not report.ready, "expected report.ready to be falsy"
    assert any(
        "protocol version mismatch" in reason for reason in report.nodes[0].reasons
    ), (
        'expected any( "protocol version mismatch" in reason for reason in report.nodes[0].reasons ) to be truthy'
    )


def test_readiness_accepts_explicit_rollout_protocol_and_artifact() -> None:
    fleet = FleetRegistry(
        build_store(),
        SECRET,
        FleetCompatibilityPolicy(
            required_agent_protocol_version=3,
            compatible_agent_protocol_versions=frozenset({2}),
            required_agent_version="0.9.0",
            required_artifact_sha256="b" * 64,
            compatible_artifact_sha256s=frozenset({ARTIFACT}),
            required_policy_version="catalog-a",
            required_runtime_profile_version="profile-a",
            required_config_digest="d" * 64,
            compatible_config_digests=frozenset({CONFIG}),
        ),
        now=lambda: NOW,
    )
    old = copy_model(heartbeat("node-a"), agent_protocol_version=2)
    fleet.register(signed(old))

    report = fleet.readiness("cluster-a", ["node-a"])

    assert report.ready, "expected report.ready to be truthy"
    assert report.nodes[0].reasons == []


def test_readiness_rejects_value_outside_rollout_window() -> None:
    fleet = FleetRegistry(
        build_store(),
        SECRET,
        FleetCompatibilityPolicy(
            required_agent_protocol_version=4,
            compatible_agent_protocol_versions=frozenset({3}),
            required_artifact_sha256="d" * 64,
            compatible_artifact_sha256s=frozenset({"b" * 64}),
        ),
        now=lambda: NOW,
    )
    unsupported = copy_model(heartbeat("node-a"), agent_protocol_version=2)
    fleet.register(signed(unsupported))

    report = fleet.readiness("cluster-a", ["node-a"])

    assert not report.ready, "expected report.ready to be falsy"
    assert any(
        "expected one of 3, 4, got 2" in reason for reason in report.nodes[0].reasons
    ), (
        'expected any( "expected one of 3, 4, got 2" in reason for reason in report.nodes[0].reasons ) to be truthy'
    )
    assert any(
        "artifact SHA-256 mismatch" in reason for reason in report.nodes[0].reasons
    ), (
        'expected any( "artifact SHA-256 mismatch" in reason for reason in report.nodes[0].reasons ) to be truthy'
    )


def test_readiness_blames_the_stale_node_when_the_fleet_matches() -> None:
    """A mismatch must say which side to fix, not just expected/got.

    Two peers already run the pinned artifact, so the operator should
    upgrade the lagging node rather than second-guess the pin.
    """
    fleet = registry()
    fleet.register(signed(heartbeat("node-a")))
    fleet.register(signed(heartbeat("node-b")))
    lagging = copy_model(heartbeat("node-c"), artifact_sha256="b" * 64)
    fleet.register(signed(lagging))

    report = fleet.readiness("cluster-a", ["node-c"])

    assert not report.ready, "expected report.ready to be falsy"
    reasons = report.nodes[0].reasons
    assert any("NODE_STALE" in reason for reason in reasons), (
        'expected any("NODE_STALE" in reason for reason in reasons) to be truthy'
    )
    assert any("node-a" in reason and "node-b" in reason for reason in reasons), (
        'expected any("node-a" in reason and "node-b" in reason for reason in reasons) to be truthy'
    )
    assert not any("PIN_AHEAD_OF_FLEET" in reason for reason in reasons), (
        'expected any("PIN_AHEAD_OF_FLEET" in reason for reason in reasons) to be falsy'
    )


def test_readiness_blames_the_pin_when_no_agent_runs_it() -> None:
    """When nothing in the fleet matches, the pin itself never shipped."""
    store = build_store()
    fleet = FleetRegistry(
        store,
        SECRET,
        FleetCompatibilityPolicy(
            required_agent_version="0.9.0",
            required_artifact_sha256="d" * 64,
            required_config_digest=CONFIG,
        ),
        now=lambda: NOW,
    )
    fleet.register(signed(heartbeat("node-a")))
    fleet.register(signed(heartbeat("node-b")))

    report = fleet.readiness("cluster-a", ["node-a", "node-b"])

    assert not report.ready, "expected report.ready to be falsy"
    for node in report.nodes:
        assert any("PIN_AHEAD_OF_FLEET" in reason for reason in node.reasons), (
            'expected any("PIN_AHEAD_OF_FLEET" in reason for reason in node.reasons) to be truthy'
        )
        assert not any("NODE_STALE" in reason for reason in node.reasons), (
            'expected any("NODE_STALE" in reason for reason in node.reasons) to be falsy'
        )


def test_readiness_ignores_dead_agents_when_blaming_a_pin() -> None:
    """Records outlive machines, so only live agents prove a pin shipped.

    Reproduces a real outage: a Spot instance group was reclaimed and
    rebuilt, the control-plane pins still named the retired build, and
    every replacement node was told "NODE_STALE ... upgrade or restart
    the agent" while pointing at nodes that no longer existed. The
    replacements were current; the pin was the stale side.
    """
    current = [NOW]
    store = build_store()
    fleet = FleetRegistry(
        store,
        SECRET,
        FleetCompatibilityPolicy(
            required_agent_version="0.9.0",
            required_artifact_sha256="d" * 64,
            required_config_digest=CONFIG,
        ),
        now=lambda: current[0],
    )
    # Retired peers that happen to match the pin.
    for node_id in ("gone-a", "gone-b"):
        fleet.register(signed(copy_model(heartbeat(node_id), artifact_sha256="d" * 64)))
    # Their leases lapse; only the replacement still heartbeats.
    current[0] = NOW + timedelta(hours=9)
    fleet.register(signed(heartbeat("live-a", observed_at=current[0])))

    report = fleet.readiness("cluster-a", ["live-a"])

    assert not report.ready, "expected report.ready to be falsy"
    reasons = report.nodes[0].reasons
    assert any("PIN_AHEAD_OF_FLEET" in reason for reason in reasons), (
        'expected any("PIN_AHEAD_OF_FLEET" in reason for reason in reasons) to be truthy'
    )
    assert not any("NODE_STALE" in reason for reason in reasons), (
        'expected any("NODE_STALE" in reason for reason in reasons) to be falsy'
    )
    assert not any("gone-a" in reason for reason in reasons), (
        'expected any("gone-a" in reason for reason in reasons) to be falsy'
    )


def test_readiness_still_blames_a_node_when_a_live_peer_matches() -> None:
    """The live-only filter must not erase a genuine NODE_STALE verdict."""
    fleet = registry()
    fleet.register(signed(heartbeat("node-a")))
    lagging = copy_model(heartbeat("node-c"), artifact_sha256="b" * 64)
    fleet.register(signed(lagging))

    report = fleet.readiness("cluster-a", ["node-c"])

    reasons = report.nodes[0].reasons
    assert any("NODE_STALE" in reason for reason in reasons), (
        'expected any("NODE_STALE" in reason for reason in reasons) to be truthy'
    )
    assert any("node-a" in reason for reason in reasons), (
        'expected any("node-a" in reason for reason in reasons) to be truthy'
    )


def test_deployment_waves_reconcile_from_heartbeats() -> None:
    fleet = registry()
    deployment = fleet.create_deployment(
        FleetDeploymentRequest(
            cluster_id="cluster-a",
            node_ids=["node-c", "node-a", "node-b"],
            desired_agent_version="0.9.0",
            desired_artifact_sha256=ARTIFACT,
            desired_policy_version="catalog-a",
            desired_runtime_profile_version="profile-a",
            desired_config_digest=CONFIG,
            max_unavailable=2,
        )
    )

    assert deployment.waves == [["node-a", "node-b"], ["node-c"]]
    try:
        fleet.update_deployment_node(
            deployment.deployment_id,
            "node-a",
            DeploymentNodeUpdate(status=DeploymentNodeStatus.READY),
        )
    except ValueError as exc:
        assert "heartbeat" in str(exc)
    else:
        raise AssertionError("READY was accepted without a heartbeat")
    try:
        fleet.update_deployment_node(
            deployment.deployment_id,
            "node-c",
            DeploymentNodeUpdate(status=DeploymentNodeStatus.INSTALLING),
        )
    except ValueError as exc:
        assert "active deployment wave" in str(exc)
    else:
        raise AssertionError("future deployment wave was started")
    lease = fleet.start_next_wave(deployment.deployment_id)
    repeated_lease = fleet.start_next_wave(deployment.deployment_id)

    assert lease.wave_index == 0
    assert lease.node_ids == ["node-a", "node-b"]
    assert repeated_lease.node_ids == lease.node_ids
    for node_id in ("node-a", "node-b", "node-c"):
        fleet.register(signed(heartbeat(node_id)))
    completed = fleet.store.get_fleet_deployment(deployment.deployment_id)

    assert completed.status is DeploymentStatus.SUCCEEDED
    assert {item.status for item in completed.nodes} == {DeploymentNodeStatus.READY}


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_terminal_deployment_retention_keeps_open_rolls(tmp_path, backend) -> None:
    """Retention must not touch a roll an agent could still be in.

    Every heartbeat asks for its cluster's open deployments, so the
    finished ones are pure overhead in that lookup - but dropping one
    that is still mid-wave would strand the nodes in it.
    """
    store = (
        build_store()
        if backend == "memory"
        else SqliteStore(str(tmp_path / f"retention-{backend}.db"))
    )
    fleet = registry(store)
    # node-a already runs the desired identity, so its roll completes on
    # creation; node-b has never checked in, so its roll stays open.
    fleet.register(signed(heartbeat("node-a")))
    finished = fleet.create_deployment(
        FleetDeploymentRequest(
            cluster_id="cluster-a",
            node_ids=["node-a"],
            desired_agent_version="0.9.0",
            desired_artifact_sha256=ARTIFACT,
            desired_policy_version="catalog-a",
            desired_runtime_profile_version="profile-a",
            desired_config_digest=CONFIG,
        )
    )
    open_roll = fleet.create_deployment(
        FleetDeploymentRequest(
            cluster_id="cluster-a",
            node_ids=["node-b"],
            desired_agent_version="0.9.0",
            desired_artifact_sha256=ARTIFACT,
            desired_policy_version="catalog-a",
            desired_runtime_profile_version="profile-a",
            desired_config_digest=CONFIG,
        )
    )

    assert (
        store.get_fleet_deployment(finished.deployment_id).status
        is DeploymentStatus.SUCCEEDED
    )
    # The window has not opened yet, so nothing may go.
    assert (
        store.cleanup_terminal_fleet_deployments(
            older_than=NOW - timedelta(days=7), limit=100
        )
        == 0
    )
    assert (
        store.cleanup_terminal_fleet_deployments(
            older_than=NOW + timedelta(seconds=1), limit=100
        )
        == 1
    )
    with pytest.raises(NotFoundError):
        store.get_fleet_deployment(finished.deployment_id)
    assert [
        item.deployment_id for item in store.list_active_fleet_deployments("cluster-a")
    ] == [open_roll.deployment_id]


def test_multi_node_reset_uses_prepare_barrier_then_commit() -> None:
    store = build_store()
    fleet = registry(store)
    fleet.register(signed(heartbeat("node-a")))
    fleet.register(signed(heartbeat("node-b")))
    coordinator = BarrierCoordinator(store, now=lambda: NOW)
    sent = []

    def sender(endpoint, envelope):
        sent.append((endpoint, envelope.command))
        return node_action_result(
            envelope.command.command_id,
            envelope.command.operation,
            details={"node_id": envelope.command.node_id},
        )

    adapter = NodeActionWorkflowAdapter(
        {}, SECRET, registry=fleet, barriers=coordinator, sender=sender
    )
    workflow = workflow_state(store)
    executor = active_workflow_executor(store, [adapter], {WorkflowOperation.RESET_GPU})
    request = WorkflowExecutionRequest(expected_fencing_token=3)

    prepared = executor.execute(workflow.request_id, request)
    committed = executor.execute(workflow.request_id, request)
    barrier = store.get_barrier("workflow-a/0/RESET_GPU")

    assert prepared.status is WorkflowStatus.RUNNING
    assert prepared.waiting_step_index == 0
    assert committed.status is WorkflowStatus.SUCCEEDED
    assert barrier.state is BarrierState.COMMITTED
    assert [item.operation for _, item in sent] == [
        WorkflowOperation.VERIFY_NO_GPU_CLIENTS,
        WorkflowOperation.VERIFY_NO_GPU_CLIENTS,
        WorkflowOperation.RESET_GPU,
        WorkflowOperation.RESET_GPU,
    ]
    assert sent[0][1].gpu_uuids == ["GPU-a"]
    assert sent[1][1].gpu_uuids == ["GPU-b"]
    assert all(item.agent_generation == 1 for _, item in sent), (
        "expected all(item.agent_generation == 1 for _, item in sent) to be truthy"
    )


def test_multi_node_full_fabric_reset_uses_same_barrier() -> None:
    store = build_store()
    fleet = registry(store)
    fleet.register(signed(heartbeat("node-a")))
    fleet.register(signed(heartbeat("node-b")))
    coordinator = BarrierCoordinator(store, now=lambda: NOW)
    sent = []

    def sender(endpoint, envelope):
        sent.append((endpoint, envelope.command))
        return node_action_result(
            envelope.command.command_id,
            envelope.command.operation,
            details={
                "node_id": envelope.command.node_id,
                "reset_scope": "ALL_LOCAL_GPUS_AND_NVSWITCHES",
            },
        )

    base_workflow = workflow_state(store)
    workflow = copy_model(
        base_workflow,
        official_action="RESET_ALL_GPUS_AND_NVSWITCHES",
        official_steps=[
            copy_model(
                base_workflow.official_steps[0],
                operation=WorkflowOperation.RESET_ALL_GPUS_NVSWITCHES,
                parameters={
                    "gpu_uuids_by_node": {"node-a": ["GPU-a"], "node-b": ["GPU-b"]},
                    "fabric_partitions_by_node": {
                        "node-a": "cluster-a/node-a/local",
                        "node-b": "cluster-a/node-b/local",
                    },
                    "sxids_by_node": {"node-a": [20001], "node-b": [20001]},
                },
            )
        ],
    )
    store.save_workflow(workflow)
    adapter = NodeActionWorkflowAdapter(
        {}, SECRET, registry=fleet, barriers=coordinator, sender=sender
    )
    executor = active_workflow_executor(
        store, [adapter], {WorkflowOperation.RESET_ALL_GPUS_NVSWITCHES}
    )
    request = WorkflowExecutionRequest(expected_fencing_token=3)

    prepared = executor.execute(workflow.request_id, request)
    committed = executor.execute(workflow.request_id, request)

    assert prepared.status is WorkflowStatus.RUNNING
    assert committed.status is WorkflowStatus.SUCCEEDED
    assert [item.operation for _, item in sent] == [
        WorkflowOperation.VERIFY_NO_GPU_CLIENTS,
        WorkflowOperation.VERIFY_NO_GPU_CLIENTS,
        WorkflowOperation.RESET_ALL_GPUS_NVSWITCHES,
        WorkflowOperation.RESET_ALL_GPUS_NVSWITCHES,
    ]


def test_quiesced_barrier_allows_stale_heartbeat_in_maintenance_window() -> None:
    current = [NOW]
    store = build_store()
    fleet = registry(store, now=lambda: current[0])
    fleet.register(signed(heartbeat("node-a")))
    fleet.register(signed(heartbeat("node-b")))
    coordinator = BarrierCoordinator(store, now=lambda: current[0])
    sent = []

    def sender(_, envelope):
        sent.append(envelope.command)
        details = {"node_id": envelope.command.node_id}
        if envelope.command.operation is WorkflowOperation.QUIESCE_GPU_SERVICES:
            details["failsafe_seconds"] = 420
        return node_action_result(
            envelope.command.command_id, envelope.command.operation, details=details
        )

    adapter = NodeActionWorkflowAdapter(
        {}, SECRET, registry=fleet, barriers=coordinator, sender=sender
    )
    workflow = quiesce_then_full_reset_workflow(store)
    executor = active_workflow_executor(
        store,
        [adapter],
        {
            WorkflowOperation.QUIESCE_GPU_SERVICES,
            WorkflowOperation.RESET_ALL_GPUS_NVSWITCHES,
        },
    )
    request = WorkflowExecutionRequest(expected_fencing_token=3)

    prepared = executor.execute(workflow.request_id, request)
    current[0] = NOW + timedelta(seconds=138)
    committed = executor.execute(workflow.request_id, request)
    barrier = store.get_barrier("workflow-a/1/RESET_ALL_GPUS_NVSWITCHES")

    assert prepared.status is WorkflowStatus.RUNNING
    assert committed.status is WorkflowStatus.SUCCEEDED
    assert barrier.state is BarrierState.COMMITTED
    quiesce_execution = store.get_workflow(workflow.request_id).step_executions[0]
    assert quiesce_execution.details["agent_generations"] == {"node-a": 1, "node-b": 1}
    assert [item.operation for item in sent][-2:] == [
        WorkflowOperation.RESET_ALL_GPUS_NVSWITCHES,
        WorkflowOperation.RESET_ALL_GPUS_NVSWITCHES,
    ]

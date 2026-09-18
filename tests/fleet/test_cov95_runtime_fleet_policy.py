from __future__ import annotations

from datetime import timedelta
from types import SimpleNamespace
from typing import Any

import pytest

from gpu_fault.fleet import (
    AgentHeartbeat,
    AgentTransitionRequest,
    FleetCompatibilityPolicy,
    FleetRegistry,
)
from gpu_fault.fleet_compatibility import (
    CURRENT_AGENT_PROTOCOL_VERSION,
    pin_value_is_accepted,
    rollout_compatibility_reasons,
)
from gpu_fault.models import WorkflowOperation
from gpu_fault.store import InMemoryStore
from tests.fleet._support import NOW, SECRET, heartbeat, registry, signed
from tests.regional._cov95_runtime_support import offline_runtime as offline_runtime


@pytest.mark.parametrize(
    "changes",
    [
        {"compatible_agent_protocol_versions": {0}},
        {"compatible_artifact_sha256s": {"invalid"}},
        {"compatible_config_digests": {"invalid"}},
    ],
)
def test_compatibility_policy_refuses_invalid_accepted_versions_and_digests(
    changes: dict[str, Any],
) -> None:
    with pytest.raises(ValueError, match="positive|SHA-256"):
        FleetCompatibilityPolicy(**changes)


@pytest.mark.parametrize("count", [1, 2])
def test_compatibility_reasons_preserve_every_incompatible_pin(count: int) -> None:
    record = registry().register(signed(heartbeat("node-a")))
    policy = FleetCompatibilityPolicy(
        required_agent_protocol_version=CURRENT_AGENT_PROTOCOL_VERSION + 1,
        compatible_agent_protocol_versions=(
            {CURRENT_AGENT_PROTOCOL_VERSION + 2} if count == 2 else set()
        ),
        required_artifact_sha256="b" * 64,
        compatible_artifact_sha256s={"d" * 64} if count == 2 else set(),
        required_compatibility_digest="b" * 64,
        compatible_compatibility_digests={"d" * 64} if count == 2 else set(),
        required_config_digest="b" * 64,
        compatible_config_digests={"d" * 64} if count == 2 else set(),
    )
    reasons = rollout_compatibility_reasons(policy, record)
    assert len(reasons) == 4
    assert all(("one of" in reason) is (count == 2) for reason in reasons), reasons


@pytest.mark.parametrize(
    "attribute",
    ["artifact_sha256", "config_digest", "compatibility_digest", "policy_version"],
)
def test_compatible_pin_acceptance_does_not_leak_between_identity_fields(
    attribute: str,
) -> None:
    policy = FleetCompatibilityPolicy(
        compatible_artifact_sha256s={"a" * 64},
        compatible_config_digests={"b" * 64},
        compatible_compatibility_digests={"c" * 64},
    )
    accepted = {
        "artifact_sha256": "a" * 64,
        "config_digest": "b" * 64,
        "compatibility_digest": "c" * 64,
        "policy_version": "required",
    }
    assert (
        pin_value_is_accepted(policy, attribute, accepted[attribute], "required")
        is True
    )
    assert pin_value_is_accepted(policy, attribute, "foreign", "required") is False


@pytest.mark.parametrize(
    "when", [NOW + timedelta(seconds=31), NOW - timedelta(seconds=301)]
)
def test_heartbeat_time_window_refuses_future_and_stale_records(when: Any) -> None:
    fleet = registry()
    with pytest.raises(ValueError, match="future|too old"):
        fleet.register(signed(heartbeat("node-a", observed_at=when)))
    assert fleet.store.list_agents("cluster-a") == []


@pytest.mark.parametrize(
    "defect", ["scheme", "pem-shape", "pem-base64", "pem-gap", "pem-empty"]
)
def test_heartbeat_model_rejects_unsafe_endpoint_and_entire_certificate_chain(
    defect: str,
) -> None:
    payload = heartbeat("node-a").model_dump(mode="json")
    pem = "-----BEGIN CERTIFICATE-----\ndW5pdA==\n-----END CERTIFICATE-----"
    if defect == "scheme":
        payload["endpoint"] = "ftp://node-a:9099"
    elif defect == "pem-shape":
        payload["tls_certificate_pem"] = "not a certificate"
    elif defect == "pem-base64":
        payload["tls_certificate_pem"] = pem.replace("dW5pdA==", "a")
    elif defect == "pem-gap":
        payload["tls_certificate_pem"] = pem + "\nforeign content\n" + pem
    else:
        payload["tls_certificate_pem"] = pem.replace("dW5pdA==", "")
    with pytest.raises(ValueError, match="http or https|certificate|PEM"):
        AgentHeartbeat.model_validate(payload)


@pytest.mark.parametrize("lister", ["missing", "raises", "empty"])
def test_failed_advisory_fleet_scan_never_hides_a_pin_refusal(lister: str) -> None:
    record = registry().register(signed(heartbeat("node-a")))
    store = SimpleNamespace(get_agent=lambda cluster, node: record)
    if lister == "raises":

        def failed(cluster: str) -> Any:
            raise RuntimeError("synthetic fleet scan failed")

        store.list_agents = failed
    elif lister == "empty":
        store.list_agents = lambda cluster: []
    fleet = FleetRegistry(
        store,
        SECRET,
        FleetCompatibilityPolicy(required_artifact_sha256="b" * 64),
        now=lambda: NOW,
    )
    report = fleet.readiness("cluster-a", ["node-a"])
    assert report.ready is False
    assert fleet.pin_drift_nodes["cluster-a"] == {
        "NODE_STALE": 0,
        "PIN_AHEAD_OF_FLEET": 1,
    }
    assert any(
        "artifact SHA-256 mismatch" in reason for reason in report.nodes[0].reasons
    ), report


def test_selected_nodes_must_share_one_accepted_runtime_identity() -> None:
    fleet = FleetRegistry(InMemoryStore(), SECRET, now=lambda: NOW)
    fleet.register(signed(heartbeat("node-a")))
    fleet.register(
        signed(heartbeat("node-b").model_copy(update={"config_digest": "d" * 64}))
    )
    report = fleet.readiness("cluster-a", ["node-a", "node-b"])
    assert report.ready is False
    assert all(not node.ready for node in report.nodes), report
    assert all(
        any("within the selected node set" in reason for reason in node.reasons)
        for node in report.nodes
    ), report


@pytest.mark.parametrize("defect", ["generation", "lifecycle", "pin", "operations"])
def test_maintenance_endpoint_keeps_every_fence_except_liveness(defect: str) -> None:
    fleet = registry(now=lambda: NOW + timedelta(seconds=100))
    record = fleet.register(signed(heartbeat("node-a")))
    expected = record.generation
    if defect == "generation":
        expected += 1
    elif defect == "lifecycle":
        fleet.drain_agent(
            "cluster-a",
            "node-a",
            AgentTransitionRequest(
                expected_generation=expected, transition_id="unit", reason="unit"
            ),
        )
        expected += 1
    elif defect == "pin":
        fleet.policy = fleet.policy.model_copy(
            update={"required_artifact_sha256": "b" * 64}
        )
    else:
        fleet.policy = fleet.policy.model_copy(
            update={"required_operations": [WorkflowOperation.REMEDIATE_DRIVER]}
        )
    with pytest.raises(
        ValueError, match="generation changed|lifecycle state|maintenance-ready"
    ):
        fleet.maintenance_endpoint("cluster-a", "node-a", expected)


@pytest.mark.parametrize("operation", ["register", "drain", "revoke", "reactivate"])
def test_agent_compare_and_swap_exhaustion_never_overwrites_the_stored_record(
    monkeypatch: pytest.MonkeyPatch, operation: str
) -> None:
    fleet = registry()
    record = fleet.register(signed(heartbeat("node-a")))
    transition = AgentTransitionRequest(
        expected_generation=record.generation, transition_id="unit", reason="unit"
    )
    if operation in {"revoke", "reactivate"}:
        record = fleet.drain_agent("cluster-a", "node-a", transition)
    if operation == "reactivate":
        record = fleet.revoke_agent("cluster-a", "node-a", transition)
        transition = transition.model_copy(
            update={"expected_generation": record.generation}
        )
    before = record.model_dump(mode="json")
    writes = []
    monkeypatch.setattr(
        fleet.store,
        "replace_agent_if_matches",
        lambda replacement, expected: writes.append((replacement, expected)) or False,
    )
    with pytest.raises(ValueError, match="conflicted with concurrent"):
        if operation == "register":
            fleet.register(
                signed(heartbeat("node-a", observed_at=NOW + timedelta(seconds=1)))
            )
        else:
            getattr(fleet, operation + "_agent")("cluster-a", "node-a", transition)
    assert len(writes) == 5
    assert (
        fleet.store.get_agent("cluster-a", "node-a").model_dump(mode="json") == before
    )


def test_drain_rejects_foreign_transition_and_stale_generation_without_a_write() -> (
    None
):
    fleet = registry()
    record = fleet.register(signed(heartbeat("node-a")))
    with pytest.raises(ValueError, match="stale agent generation"):
        fleet.drain_agent(
            "cluster-a",
            "node-a",
            AgentTransitionRequest(
                expected_generation=record.generation + 1,
                transition_id="unit",
                reason="unit",
            ),
        )
    transition = AgentTransitionRequest(
        expected_generation=1, transition_id="unit", reason="unit"
    )
    drained = fleet.drain_agent("cluster-a", "node-a", transition)
    with pytest.raises(ValueError, match="another transition"):
        fleet.drain_agent(
            "cluster-a",
            "node-a",
            transition.model_copy(update={"transition_id": "foreign"}),
        )
    with pytest.raises(ValueError, match="same transition"):
        fleet.revoke_agent(
            "cluster-a",
            "node-a",
            transition.model_copy(update={"transition_id": "foreign"}),
        )
    assert fleet.store.get_agent("cluster-a", "node-a") == drained

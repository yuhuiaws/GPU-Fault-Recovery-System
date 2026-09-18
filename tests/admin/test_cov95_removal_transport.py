from __future__ import annotations

import json

import pytest

from gpu_fault.admin.bootstrap_common import BootstrapError
from tests.admin._cov95_removal_support import RemovalTransport


@pytest.fixture
def transport(tmp_path, monkeypatch):
    return RemovalTransport(tmp_path, monkeypatch)


def test_full_remove_uses_identity_checked_fake_transports_and_preserves_clusters(
    transport,
):
    result = transport.remove()
    assert result["phase"] == "COMPLETED"
    assert result["cpu_control_plane"] == result["gpu_cluster"] == "PRESERVED"
    assert result["remaining_cluster_ids"] == []
    assert transport.events.index("cleanup") < transport.events.index(
        "delete-namespace"
    )
    assert transport.events.index("delete-namespace") < transport.events.index(
        "remove-keys"
    )
    assert transport.events.index("delete-namespace") < transport.events.index(
        "delete-role"
    )
    assert transport.events.index("registry-sync") < transport.events.index(
        "sync-state"
    )
    assert transport.keys == {} and not transport.namespace and not transport.role_alive


@pytest.mark.parametrize(
    "boundary", ["drain-cluster", "cleanup", "remove-cluster", "sync-state", "verify"]
)
def test_remove_transport_failure_is_retryable_without_repeating_completed_barriers(
    transport, boundary
):
    transport.failure = boundary
    with pytest.raises(BootstrapError):
        transport.remove()
    state = transport.state()
    assert state["phase"] != "COMPLETED"
    cleanup_count = transport.events.count("cleanup")
    transport.failure = None
    assert transport.remove()["phase"] == "COMPLETED"
    if "KUBERNETES_QUIESCED" in state["completed_steps"]:
        assert transport.events.count("cleanup") == cleanup_count


@pytest.mark.parametrize(
    "nodes", [{}, {"items": [None]}, {"items": [{"metadata": {}}]}]
)
def test_remove_target_inventory_failure_stops_before_drain(transport, nodes):
    transport.nodes = nodes
    with pytest.raises(BootstrapError, match="node response"):
        transport.remove()
    assert "drain-cluster" not in transport.events
    assert "cleanup" not in transport.events


@pytest.mark.parametrize(
    "overrides",
    [
        {"NodeRecovery": "Automatic"},
        {"ClusterName": "foreign"},
        {
            "Orchestrator": {
                "Eks": {
                    "ClusterArn": "arn:aws:eks:us-east-1:123456789012:cluster/foreign"
                }
            }
        },
    ],
)
def test_remove_hyperpod_identity_drift_stops_before_drain(transport, overrides):
    transport.hyperpod_override = overrides
    with pytest.raises(BootstrapError, match="HyperPod identity|NodeRecovery"):
        transport.remove()
    assert "drain-cluster" not in transport.events


def test_remove_context_endpoint_mismatch_stops_before_drain(transport):
    transport.endpoint_override = "https://foreign.example.invalid"
    with pytest.raises(BootstrapError, match="context does not match"):
        transport.remove()
    assert "drain-cluster" not in transport.events


@pytest.mark.parametrize(
    "override",
    [
        {"scope": "cpu"},
        {"status": "FAILED"},
        {"original_resources": []},
        {"original_resources": [{"scope": "gpu:foreign", "context": "foreign"}]},
    ],
)
def test_cleanup_ack_without_bound_completion_never_deletes_namespace(
    transport, override
):
    transport.cleanup_override = override
    with pytest.raises(BootstrapError, match="evidence|target"):
        transport.remove()
    assert "delete-namespace" not in transport.events
    assert "delete-role" not in transport.events


@pytest.mark.parametrize(
    "document",
    [{}, {"kind": "Secret", "metadata": {}}, {"data": {"clusters.json": "not-base64"}}],
)
def test_final_registry_read_failure_keeps_removal_nonterminal(transport, document):
    transport.registry_secret_override = document
    with pytest.raises(BootstrapError, match="final verification failed"):
        transport.remove()
    assert transport.state()["phase"] != "COMPLETED"
    assert transport.events.count("delete-role") == 1
    transport.registry_secret_override = None
    assert transport.remove()["phase"] == "COMPLETED"
    assert transport.events.count("delete-role") == 1


def test_remove_records_bootstrap_resource_changes_after_completed_cleanup(transport):
    path = transport.directory / "bootstrap-state.json"
    associations = [{"vpc_id": "vpc-gpu-a", "vpc_region": "us-east-1"}]
    path.write_text(
        json.dumps(
            {
                "site_id": "test-site",
                "completed_tasks": ["executor_role:gpu-a", "node_keys:gpu-a"],
                "resources": {
                    "executor_role:gpu-a": {},
                    "node_keys:gpu-a": {},
                    "pki": {"vpc_associations": associations},
                    "nlb_network": {"gpu_nat_eips": ["192.0.2.20"]},
                },
            }
        )
    )
    assert transport.remove()["phase"] == "COMPLETED"
    state = json.loads(path.read_text())
    assert state["completed_tasks"] == []
    assert "executor_role:gpu-a" not in state["resources"]
    assert "node_keys:gpu-a" not in state["resources"]
    assert transport.state()["evidence"]["AWS_DETACHED"]["detached_vpc_id"] is None
    assert state["resources"]["pki"]["vpc_associations"] == associations, (
        "removal dropped a Route53 association without exact detach evidence"
    )
    assert state["resources"]["nlb_network"]["gpu_nat_eips"] == []
    assert state["removed_clusters"]["gpu-a"]["vpc_id"] == "vpc-gpu-a"
    assert state["removed_clusters"]["gpu-a"]["vpc_region"] == "us-east-1"

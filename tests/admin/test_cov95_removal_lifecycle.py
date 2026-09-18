from __future__ import annotations

from dataclasses import replace

import pytest

from gpu_fault.admin import cluster_removal as removal
from gpu_fault.admin.bootstrap_common import BootstrapError
from gpu_fault.admin.process_supervisor import ProcessSupervisionLost
from tests.admin.test_admin_cluster_removal_lifecycle import RemovalScenario


@pytest.fixture
def scenario(tmp_path, monkeypatch):
    return RemovalScenario(tmp_path, monkeypatch)


@pytest.mark.parametrize(
    "boundary",
    [
        "snapshot",
        "drain",
        "cleanup",
        "annotations",
        "namespace-request",
        "namespace-wait",
        "unregister",
        "keys",
        "aws-network",
        "aurora",
        "bootstrap",
        "failure-domains",
        "release-state",
        "verify",
    ],
)
def test_remove_retries_only_unfinished_phases(scenario, boundary):
    scenario.failure = boundary
    with pytest.raises(BootstrapError, match=boundary):
        removal.remove_cluster(scenario.request())
    before = scenario.state()
    assert before["phase"] != "COMPLETED"
    completed = set(before["completed_steps"])
    cleanup_calls = scenario.calls.count("cleanup")
    scenario.failure = None
    result = removal.remove_cluster(scenario.request())
    assert result["phase"] == "COMPLETED"
    assert completed.issubset(scenario.state()["completed_steps"]), (
        "removal resume lost completed safety barriers"
    )
    assert result["cpu_control_plane"] == result["gpu_cluster"] == "PRESERVED"
    if "KUBERNETES_QUIESCED" in completed:
        assert scenario.calls.count("cleanup") == cleanup_calls
    assert scenario.calls.index("namespace-wait") < scenario.calls.index("unregister")
    assert scenario.calls.index("namespace-wait") < scenario.calls.index("aws-network")


@pytest.mark.parametrize(
    "field,value",
    [
        ("namespace_uid", "recreated-namespace"),
        ("node_uid", "recreated-node"),
        ("incarnation", "different-cluster-incarnation"),
        ("release_identity", "f" * 64),
    ],
)
def test_remove_refuses_identity_drift_before_resumed_mutation(scenario, field, value):
    scenario.failure = "cleanup"
    with pytest.raises(BootstrapError, match="cleanup"):
        removal.remove_cluster(scenario.request())
    events = list(scenario.calls)
    scenario.failure = None
    setattr(scenario, field, value)
    with pytest.raises(BootstrapError, match="drifted|recreated"):
        removal.remove_cluster(scenario.request())
    assert scenario.calls == events, "identity drift allowed a resumed mutation"


@pytest.mark.parametrize("boundary", ["registry", "membership"])
def test_read_failure_cannot_cross_removal_barrier(scenario, monkeypatch, boundary):
    def failed(*_args, **_kwargs):
        raise BootstrapError("example read unavailable")

    name = (
        "fetch_installation_resource_registry"
        if boundary == "registry"
        else "membership_runtime_snapshot"
    )
    monkeypatch.setattr(removal, name, failed)
    with pytest.raises(BootstrapError, match="read unavailable"):
        removal.remove_cluster(scenario.request())
    if boundary == "membership":
        assert "cleanup" not in scenario.calls
        assert "namespace-request" not in scenario.calls
        assert "unregister" not in scenario.calls
        assert "aws-network" not in scenario.calls
    else:
        assert "aurora" not in scenario.calls
        assert "bootstrap" not in scenario.calls
        assert "AURORA_UPDATED" not in scenario.state()["completed_steps"]


@pytest.mark.parametrize(
    "arn",
    [
        "arn:aws:eks:us-east-1:123456789012:cluster/foreign",
        "arn:aws:sns:us-east-1:123456789012:example",
    ],
)
def test_remove_rejects_arn_not_bound_to_the_target_before_discovery(scenario, arn):
    with pytest.raises(BootstrapError, match="conflicts|cluster"):
        removal.remove_cluster(replace(scenario.request(), gpu_cluster_arn=arn))
    assert scenario.calls == []


def test_remove_requires_confirmation_before_discovery(scenario):
    with pytest.raises(BootstrapError, match="requires --confirm"):
        removal.remove_cluster(replace(scenario.request(), confirmation="WRONG"))
    assert scenario.calls == []


def test_remove_persists_supervision_loss_and_refuses_fresh_retry(
    scenario, monkeypatch
):
    original = scenario.event

    def event(name):
        if name == "cleanup":
            raise ProcessSupervisionLost("example missing completion proof")
        original(name)

    monkeypatch.setattr(scenario, "event", event)
    with pytest.raises(ProcessSupervisionLost):
        removal.remove_cluster(scenario.request())
    assert scenario.state()["phase"] == "SUPERVISION_LOST"
    before = list(scenario.calls)
    with pytest.raises(ProcessSupervisionLost, match="automatic retry"):
        removal.remove_cluster(scenario.request())
    assert scenario.calls == before


def test_completed_removal_rechecks_identity_without_repeating_cleanup(scenario):
    first = removal.remove_cluster(scenario.request())
    events = list(scenario.calls)
    second = removal.remove_cluster(scenario.request())
    assert second == first
    assert scenario.calls == events
    scenario.namespace_uid = "new-namespace"
    with pytest.raises(BootstrapError, match="recreated"):
        removal.remove_cluster(scenario.request())
    assert scenario.calls == events

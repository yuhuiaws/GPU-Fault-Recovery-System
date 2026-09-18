from __future__ import annotations

import builtins

import boto3
import pytest

from gpu_fault.hyperpod import (
    HyperPodAction,
    HyperPodAdapterConfig,
    HyperPodAdapterError,
    HyperPodAdvisoryDisposition,
    HyperPodLifecycleAdapter,
    HyperPodNode,
    HyperPodRecoveryState,
    HyperPodWorkloadRecoveryEvidence,
)
from gpu_fault.models import CapabilityMode, CapabilityName
from tests.hyperpod._cov95_provider_extra_root import (
    TARGET,
    RecordingProvider,
    RootHarness,
)
from tests.hyperpod._cov95_provider_extra_safety import (
    provider_extra_isolation as provider_extra_isolation,
)


def test_default_client_factory_uses_the_configured_region_with_a_fake_sdk(monkeypatch):
    client = RecordingProvider()
    calls = []

    def create(service, **kwargs):
        calls.append((service, kwargs))
        return client

    monkeypatch.setattr(boto3, "client", create)
    adapter = HyperPodLifecycleAdapter(
        HyperPodAdapterConfig(cluster_name="hp-cluster", region_name="unit-region")
    )
    snapshot = adapter.discover()
    assert calls == [("sagemaker", {"region_name": "unit-region"})], (
        "default construction did not bind the SageMaker client to its configured region"
    )
    assert snapshot.node_recovery == "None", (
        "discovery lost the explicit recovery policy"
    )
    assert len(snapshot.nodes) == 2, "discovery did not read both fake provider pages"
    assert client.reboot_requests == [], "discovery submitted a lifecycle action"


def test_missing_sdk_reports_installation_requirement_before_any_provider_access(
    monkeypatch,
):
    original = builtins.__import__

    def unavailable(name, *args, **kwargs):
        if name == "boto3":
            raise ModuleNotFoundError("unit missing SDK")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", unavailable)
    with pytest.raises(RuntimeError, match="install gpu-fault-control-plane"):
        HyperPodLifecycleAdapter(HyperPodAdapterConfig(cluster_name="hp-cluster"))


def test_programmatic_automatic_recovery_override_cannot_arm_mutation():
    with pytest.raises(ValueError, match="Automatic is prohibited"):
        HyperPodAdapterConfig(
            cluster_name="hp-cluster", allow_when_node_recovery_automatic=True
        )


def test_environment_requires_an_explicit_cluster_before_building_a_client(monkeypatch):
    monkeypatch.delenv("GPU_FAULT_HYPERPOD_CLUSTER", raising=False)
    with pytest.raises(ValueError, match="HYPERPOD_CLUSTER is required"):
        HyperPodAdapterConfig.from_environment()


@pytest.mark.parametrize("kind", ["summary", "detail"])
def test_node_inventory_without_logical_identity_is_refused(kind):
    h = RootHarness()
    if kind == "summary":
        h.client.summary_override = {
            "ClusterNodeSummaries": [{"InstanceId": "unit-instance"}]
        }
        operation = h.adapter.list_nodes
    else:
        h.client.detail_override = {"NodeDetails": {"InstanceId": "unit-instance"}}

        def operation():
            return h.adapter.describe_node(TARGET)

    with pytest.raises(HyperPodAdapterError, match="NodeLogicalId"):
        operation()
    assert h.client.reboot_requests == [], "malformed inventory triggered a submission"


def test_non_enriched_inventory_preserves_unknown_status_without_detail_reads():
    h = RootHarness()
    h.client.summary_override = {"ClusterNodeSummaries": [{"NodeLogicalId": TARGET}]}
    nodes = h.adapter.list_nodes(enrich=False)
    assert len(nodes) == 1, (
        "a provider node with unknown status disappeared from inventory"
    )
    assert nodes[0].status == "NotFound", (
        "missing provider status was invented as Running"
    )
    assert h.client.node_reads == [], (
        "a summary-only inventory unexpectedly fetched details"
    )
    assert nodes[0].aliases == {TARGET}, (
        "missing identifiers produced synthetic aliases"
    )


@pytest.mark.parametrize("identifiers", [[], ["missing-node"]])
def test_unresolvable_target_is_refused_without_submitting(identifiers):
    h = RootHarness()
    with pytest.raises(HyperPodAdapterError, match="target node|not part"):
        h.adapter.resolve_nodes(identifiers)
    assert h.client.reboot_requests == [], (
        "an unresolved target reached the provider mutation"
    )


def test_ambiguous_alias_cannot_select_one_of_two_real_node_models():
    h = RootHarness()
    nodes = [
        HyperPodNode(
            node_logical_id=f"logical-{number}",
            status="Running",
            kubernetes_labels={"kubernetes.io/hostname": "shared-name"},
        )
        for number in (1, 2)
    ]
    with pytest.raises(HyperPodAdapterError, match="ambiguous"):
        h.adapter.resolve_nodes(["shared-name"], nodes=nodes)
    assert h.client.list_requests == [], "resolution ignored the supplied snapshot"


def test_explicit_empty_inventory_does_not_refetch_and_resolve_from_a_different_snapshot():
    h = RootHarness()
    with pytest.raises(HyperPodAdapterError, match="not part"):
        h.adapter.resolve_nodes([TARGET], nodes=[])
    assert h.client.list_requests == [], (
        "an authoritative empty inventory must not trigger a second provider observation"
    )


@pytest.mark.parametrize(
    ("fault", "message"),
    [
        ("allowlist", "not in adapter allowlist"),
        ("cluster", "must be InService"),
        ("batch", "batch exceeds limit"),
        ("status", "non-actionable status"),
        ("isolation", "isolation evidence is missing"),
    ],
)
def test_preflight_refuses_each_unproved_gate_and_submission_stays_blocked(
    fault, message
):
    h = RootHarness(
        allowed_actions=[] if fault == "allowlist" else [HyperPodAction.REBOOT],
        max_batch_size=1 if fault == "batch" else 25,
    )
    targets = [TARGET, "worker-group-2"] if fault == "batch" else [TARGET]
    if fault == "cluster":
        h.client.cluster_status = "Updating"
    elif fault == "status":
        h.client.detail_override = {
            "NodeDetails": {
                "NodeLogicalId": TARGET,
                "InstanceStatus": {"Status": "Rebooting"},
            }
        }
    isolated = [] if fault == "isolation" else targets
    result = h.adapter.preflight(
        HyperPodAction.REBOOT, targets, isolation_verified_nodes=isolated
    )
    assert result.safe_to_submit is False, (
        "a failed provider preflight gate was ignored"
    )
    assert any(message in reason for reason in result.gate_failures), (
        f"preflight omitted the {fault} gate diagnostic"
    )
    with pytest.raises(HyperPodAdapterError, match="preflight failed"):
        h.submit(targets, isolation_verified_nodes=isolated)
    assert h.client.reboot_requests == [], "a rejected preflight submitted a reboot"


def test_unknown_recovery_ownership_stays_observe_only_and_cannot_be_advised_as_execution():
    h = RootHarness()
    h.client.node_recovery = "Unreported"
    ownership = h.adapter.resolve_recovery_ownership(
        HyperPodWorkloadRecoveryEvidence.from_eks_annotations(None)
    )
    assert ownership.node_recovery is HyperPodRecoveryState.UNKNOWN, (
        "unknown provider recovery policy was inferred"
    )
    assert all(
        item.mode is CapabilityMode.OBSERVE for item in ownership.capabilities
    ), "unknown recovery ownership created an executable capability claim"
    advisory = h.adapter.create_recovery_advisory(
        ownership,
        capability=CapabilityName.NODE_REBOOT,
        recommended_action="RESTART_NODE",
        rationale=["unit policy cannot be verified"],
    )
    assert advisory.disposition is HyperPodAdvisoryDisposition.ADVISE_AND_BLOCK, (
        "an unresolved owner was advised to mutate the node"
    )
    assert len(ownership.warnings) == 2, (
        "node and workload uncertainty were not both reported"
    )


@pytest.mark.parametrize("mismatch", ["cluster", "capability"])
def test_advisory_requires_the_same_cluster_and_a_recorded_recovery_capability(
    mismatch,
):
    h = RootHarness()
    ownership = h.adapter.resolve_recovery_ownership(
        HyperPodWorkloadRecoveryEvidence.from_eks_annotations({})
    )
    if mismatch == "cluster":
        ownership = ownership.model_copy(update={"cluster_name": "other-cluster"})
    with pytest.raises(HyperPodAdapterError, match="different cluster|not a HyperPod"):
        h.adapter.create_recovery_advisory(
            ownership,
            capability=CapabilityName.NODE_REBOOT
            if mismatch == "cluster"
            else CapabilityName.GPU_RESET,
            recommended_action="RESTART_NODE",
            rationale=["unit decision"],
        )
    assert h.client.reboot_requests == [], (
        "invalid advisory construction dispatched an action"
    )

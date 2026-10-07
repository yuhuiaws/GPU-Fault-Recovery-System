"""``EksManagedNodeProvider``: the HyperPod adapter surface over Kubernetes + EC2.

Mirrors ``tests/hyperpod/test_hyperpod.py`` for the plain-EKS provider: the
node inventory is the label-selected Kubernetes nodes joined with
``DescribeInstances`` by providerID, preflight gates on the EKS cluster and
node groups being ACTIVE, reboot is ``RebootInstances`` on exactly the bound
instance ids, replacement does not exist, and submission idempotency is the
shared durable ledger.
"""

from __future__ import annotations

import pytest

from gpu_fault.eks_provider import (
    EKS_ADAPTER_OWNER,
    EksManagedNodeProvider,
    EksProviderConfig,
    EksProviderError,
    eks_provider_from_environment,
)
from gpu_fault.hyperpod import (
    HyperPodAction,
    HyperPodAdapterError,
    HyperPodRecoveryState,
    HyperPodWorkloadRecoveryEvidence,
)
from gpu_fault.models import (
    CapabilityMode,
    CapabilityName,
    Environment,
    WorkflowOperation,
)
from gpu_fault.node_provider import EKS_REBOOT_CONFIRMATION_SOURCE
from gpu_fault.store import InMemoryStore
from tests._builders import workflow_step
from tests.eks_provider._support import (
    CLUSTER_ID,
    CPU_INSTANCE,
    CPU_NODE,
    EKS_CLUSTER,
    INSTANCE_A,
    INSTANCE_B,
    NODE_A,
    NODE_B,
    NODEGROUP,
    REGION,
    SELECTOR,
    FakeCore,
    FakeEc2,
    FakeEks,
    config,
    default_instances,
    default_nodes,
    ec2_instance,
    gpu_node,
    provider,
)

SUBMIT = {
    "isolation_verified_nodes": [NODE_A],
    "confirm_cluster_name": CLUSTER_ID,
    "workflow_fencing_token": 3,
    "expected_fencing_token": 3,
    "idempotency_key": "workflow-1/RESTART_NODE/1",
}


def test_provider_advertises_its_label_and_confirmation_source() -> None:
    adapter, *_ = provider()

    assert adapter.provider_label == "EKS"
    assert adapter.confirmation_source == EKS_REBOOT_CONFIRMATION_SOURCE
    assert adapter.config.cluster_name == CLUSTER_ID, "cluster_name is the cluster id"
    assert isinstance(EksProviderError(), HyperPodAdapterError), (
        "provider-neutral handlers catch the EKS error through the HyperPod one"
    )


def test_list_nodes_joins_selected_kubernetes_nodes_with_their_ec2_instances() -> None:
    adapter, core, ec2, _ = provider()

    nodes = {node.node_logical_id: node for node in adapter.list_nodes(enrich=True)}

    assert core.list_selectors == [SELECTOR]
    assert set(nodes) == {NODE_A, NODE_B}, "the CPU node never matches the selector"
    node = nodes[NODE_A]
    assert node.instance_id == INSTANCE_A
    assert node.status == "Running"
    assert node.instance_group_name == NODEGROUP
    assert node.instance_type == "p5en.48xlarge"
    assert node.availability_zone == "us-west-2a"
    assert node.capacity_type == "SPOT"
    assert node.private_dns_hostname == NODE_A
    assert {NODE_A, INSTANCE_A, "10.0.33.205"} <= node.aliases
    assert nodes[NODE_B].availability_zone == "us-west-2b"
    requested = {
        value
        for call in ec2.describe_calls
        for item in call["Filters"]
        for value in item["Values"]
    }
    assert requested == {INSTANCE_A, INSTANCE_B}, "EC2 is asked only about GPU nodes"
    assert CPU_INSTANCE not in requested


def test_nodes_without_a_usable_provider_id_or_instance_read_not_found() -> None:
    core = FakeCore(
        [
            gpu_node(NODE_A, INSTANCE_A, provider_id=None),
            gpu_node(NODE_B, INSTANCE_B, provider_id="aws:///us-west-2b/not-an-id"),
            gpu_node(CPU_NODE, "i-0a000003"),
        ]
    )
    ec2 = FakeEc2([ec2_instance(INSTANCE_B, state="terminated")])
    adapter, *_ = provider(core=core, ec2=ec2)

    nodes = {node.node_logical_id: node for node in adapter.list_nodes()}

    assert nodes[NODE_A].instance_id is None
    assert nodes[NODE_A].status == "NotFound"
    assert "providerID" in str(nodes[NODE_A].status_message)
    assert nodes[NODE_B].instance_id is None, "an unparseable providerID binds nothing"
    gone = nodes[CPU_NODE]
    assert gone.instance_id == "i-0a000003"
    assert gone.status == "NotFound", "an instance EC2 no longer knows is NotFound"


@pytest.mark.parametrize(
    "state,expected",
    [
        ("running", "Running"),
        ("stopped", "Stopped"),
        ("stopping", "Stopping"),
        ("shutting-down", "ShuttingDown"),
        ("terminated", "Terminated"),
        ("pending", "Pending"),
        ("weird", "Unknown"),
    ],
)
def test_ec2_states_map_onto_the_provider_status_vocabulary(
    state: str, expected: str
) -> None:
    ec2 = FakeEc2([ec2_instance(INSTANCE_A, state=state)])
    adapter, *_ = provider(core=FakeCore([gpu_node(NODE_A, INSTANCE_A)]), ec2=ec2)

    node = adapter.describe_node(NODE_A)

    assert node.status == expected
    if state == "running":
        assert node.status_message is None
    else:
        assert node.status_message == f"state {state}"


def test_resolve_nodes_accepts_node_name_or_instance_id_and_refuses_strangers() -> None:
    adapter, *_ = provider()

    resolved = adapter.resolve_nodes([INSTANCE_A, NODE_B])

    assert [node.node_logical_id for node in resolved] == [NODE_A, NODE_B]
    with pytest.raises(EksProviderError, match="not part of the selected EKS nodes"):
        adapter.resolve_nodes([CPU_NODE])
    with pytest.raises(EksProviderError, match="at least one target node"):
        adapter.resolve_nodes([])


def test_preflight_passes_on_an_active_cluster_and_node_group() -> None:
    adapter, _, _, eks = provider()

    result = adapter.preflight(
        HyperPodAction.REBOOT, [NODE_A], isolation_verified_nodes=[NODE_A]
    )

    assert result.safe_to_submit, result.gate_failures
    assert result.cluster_status == "ACTIVE"
    assert result.cluster_name == CLUSTER_ID
    assert result.node_recovery is None, "no HyperPod NodeRecovery gate on plain EKS"
    assert result.node_repair == "unset", (
        "the node group's EKS node auto repair is recorded: the opt-in field is absent"
    )
    assert result.execution_enabled is True
    assert eks.described_clusters == [EKS_CLUSTER], (
        "the EKS cluster is bound from the instances' eks:cluster-name tag"
    )
    assert eks.described_nodegroups == [(EKS_CLUSTER, NODEGROUP)]
    assert [node.node_logical_id for node in result.targets] == [NODE_A]


@pytest.mark.parametrize(
    "variant,fragment",
    [
        ("cluster-updating", "must be ACTIVE, got UPDATING"),
        (
            "nodegroup-degraded",
            "node group gpu-p5en-efa-spot-node-group must be ACTIVE",
        ),
        ("no-nodegroup-label", "carry no managed node group label"),
        (
            "node-repair-enabled",
            "EKS node auto repair is enabled on node group "
            "gpu-p5en-efa-spot-node-group; a direct reboot would create a second "
            "lifecycle trigger",
        ),
        ("instance-stopped", "non-actionable EC2 state"),
        ("instance-gone", "non-actionable EC2 state"),
        ("no-isolation", "isolation evidence is missing"),
        ("replace", "no provider node replacement"),
        ("disabled", "mutation is disabled by configuration"),
        ("no-provider-id", "nodes have no EC2 providerID"),
    ],
)
def test_preflight_fails_closed_on_each_gate(variant: str, fragment: str) -> None:
    nodes = default_nodes()
    instances = default_instances()
    eks = FakeEks()
    options: dict[str, object] = {}
    action = HyperPodAction.REBOOT
    isolation = [NODE_A]
    if variant == "cluster-updating":
        eks = FakeEks(status="UPDATING")
    elif variant == "nodegroup-degraded":
        eks = FakeEks(nodegroups={NODEGROUP: "DEGRADED"})
    elif variant == "node-repair-enabled":
        eks = FakeEks(
            nodegroups={
                NODEGROUP: {"status": "ACTIVE", "nodeRepairConfig": {"enabled": True}}
            }
        )
    elif variant == "no-nodegroup-label":
        nodes[0] = gpu_node(NODE_A, INSTANCE_A, nodegroup=None)
        instances[0] = ec2_instance(INSTANCE_A, nodegroup=None)
    elif variant == "instance-stopped":
        instances[0] = ec2_instance(INSTANCE_A, state="stopped")
    elif variant == "instance-gone":
        instances = instances[1:]
    elif variant == "no-isolation":
        isolation = []
    elif variant == "replace":
        action = HyperPodAction.REPLACE
    elif variant == "disabled":
        options["reboot_enabled"] = False
    elif variant == "no-provider-id":
        nodes[0] = gpu_node(NODE_A, INSTANCE_A, provider_id=None)
    adapter, _, ec2, _ = provider(
        core=FakeCore(nodes), ec2=FakeEc2(instances), eks=eks, **options
    )

    result = adapter.preflight(action, [NODE_A], isolation_verified_nodes=isolation)

    assert not result.safe_to_submit, variant
    assert any(fragment in failure for failure in result.gate_failures), (
        result.gate_failures
    )
    with pytest.raises(
        EksProviderError, match="EKS preflight failed|replacement is unsupported"
    ):
        adapter.submit(
            action, [NODE_A], **{**SUBMIT, "isolation_verified_nodes": isolation}
        )
    assert ec2.reboots == [], "a failed preflight never reaches RebootInstances"


def test_preflight_records_an_explicitly_disabled_node_auto_repair() -> None:
    adapter, _, _, _ = provider(
        eks=FakeEks(
            nodegroups={
                NODEGROUP: {"status": "ACTIVE", "nodeRepairConfig": {"enabled": False}}
            }
        )
    )

    result = adapter.preflight(
        HyperPodAction.REBOOT, [NODE_A], isolation_verified_nodes=[NODE_A]
    )

    assert result.safe_to_submit, result.gate_failures
    assert result.node_repair == "disabled"


def test_preflight_binds_the_eks_cluster_from_tags_or_the_configured_name() -> None:
    foreign = default_instances()
    foreign[0] = ec2_instance(INSTANCE_A, cluster="other-cluster")
    adapter, *_ = provider(ec2=FakeEc2(foreign), eks_cluster_name=EKS_CLUSTER)
    result = adapter.preflight(
        HyperPodAction.REBOOT, [NODE_A], isolation_verified_nodes=[NODE_A]
    )
    assert any(
        "tagged for EKS cluster(s) ['other-cluster']" in item
        for item in result.gate_failures
    ), result.gate_failures

    untagged = default_instances()
    untagged[0] = ec2_instance(INSTANCE_A, cluster=None)
    adapter, *_ = provider(ec2=FakeEc2(untagged))
    result = adapter.preflight(
        HyperPodAction.REBOOT, [NODE_A], isolation_verified_nodes=[NODE_A]
    )
    assert any(
        "no eks:cluster-name tag" in item and "GPU_FAULT_EKS_CLUSTER_NAME" in item
        for item in result.gate_failures
    ), result.gate_failures

    conflicting = default_instances()
    conflicting[1] = ec2_instance(INSTANCE_B, cluster="other-cluster")
    adapter, *_ = provider(ec2=FakeEc2(conflicting))
    result = adapter.preflight(
        HyperPodAction.REBOOT,
        [NODE_A, NODE_B],
        isolation_verified_nodes=[NODE_A, NODE_B],
    )
    assert any(
        "conflicting eks:cluster-name tags" in item for item in result.gate_failures
    ), result.gate_failures

    pinned, _, _, eks = provider(eks_cluster_name=EKS_CLUSTER)
    pinned_result = pinned.preflight(
        HyperPodAction.REBOOT, [NODE_A], isolation_verified_nodes=[NODE_A]
    )
    assert pinned_result.safe_to_submit, pinned_result.gate_failures
    assert eks.described_clusters == [EKS_CLUSTER]


def test_preflight_can_skip_the_execution_switch_for_read_only_callers() -> None:
    adapter, *_ = provider(reboot_enabled=False)

    result = adapter.preflight(
        HyperPodAction.REBOOT,
        [NODE_A],
        isolation_verified_nodes=[NODE_A],
        require_execution_enabled=False,
    )

    assert result.safe_to_submit, result.gate_failures
    assert result.execution_enabled is False


def test_submit_reboots_exactly_the_bound_instance_and_records_the_nodes() -> None:
    adapter, _, ec2, _ = provider()

    result = adapter.submit(HyperPodAction.REBOOT, [NODE_A], **SUBMIT)

    assert ec2.reboots == [[INSTANCE_A]]
    assert result.action is HyperPodAction.REBOOT
    assert result.cluster_name == CLUSTER_ID
    assert result.requested_node_logical_ids == [NODE_A]
    assert result.successful_node_logical_ids == [NODE_A], (
        "submitted_nodes names one logical id (the node name) per target"
    )
    assert result.failures == []
    assert result.operation_id.startswith("eks-op-"), result.operation_id
    assert result.duplicate is False


def test_submit_refuses_when_the_node_provider_id_moved_since_preflight() -> None:
    adapter, core, ec2, _ = provider()
    original_read = core.read_node

    def replaced(name: str, **kwargs: object) -> dict[str, object]:
        node = dict(original_read(name, **kwargs))
        node["spec"] = {**node["spec"], "providerID": "aws:///us-west-2a/i-0fffffff"}
        return node

    core.read_node = replaced  # type: ignore[method-assign]

    with pytest.raises(EksProviderError, match="providerID now names i-0fff"):
        adapter.submit(HyperPodAction.REBOOT, [NODE_A], **SUBMIT)
    assert ec2.reboots == [], "the new instance behind the same node name is untouched"


def test_submit_requires_cluster_confirmation_fencing_and_a_reboot_action() -> None:
    adapter, _, ec2, _ = provider()

    with pytest.raises(EksProviderError, match="cluster confirmation does not match"):
        adapter.submit(
            HyperPodAction.REBOOT, [NODE_A], **{**SUBMIT, "confirm_cluster_name": "x"}
        )
    with pytest.raises(EksProviderError, match="stale workflow fencing token"):
        adapter.submit(
            HyperPodAction.REBOOT, [NODE_A], **{**SUBMIT, "workflow_fencing_token": 2}
        )
    with pytest.raises(EksProviderError, match="replacement is unsupported"):
        adapter.submit(HyperPodAction.REPLACE, [NODE_A], **SUBMIT)
    assert ec2.reboots == []


def test_submit_is_idempotent_per_key_in_process() -> None:
    adapter, _, ec2, _ = provider()

    first = adapter.submit(HyperPodAction.REBOOT, [NODE_A], **SUBMIT)
    again = adapter.submit(HyperPodAction.REBOOT, [NODE_A], **SUBMIT)

    assert ec2.reboots == [[INSTANCE_A]]
    assert again.duplicate is True
    assert again.operation_id == first.operation_id
    with pytest.raises(EksProviderError, match="already used for a different EKS"):
        adapter.submit(HyperPodAction.REBOOT, [NODE_B], **SUBMIT)


def test_durable_submission_survives_a_provider_restart() -> None:
    store = InMemoryStore()
    first_adapter, _, first_ec2, _ = provider(store=store)
    first = first_adapter.submit(HyperPodAction.REBOOT, [NODE_A], **SUBMIT)

    second_adapter, _, second_ec2, _ = provider(store=store)
    replay = second_adapter.submit(HyperPodAction.REBOOT, [NODE_A], **SUBMIT)

    assert first_ec2.reboots == [[INSTANCE_A]]
    assert second_ec2.reboots == [], "the restarted executor replays, never resubmits"
    assert replay.duplicate is True
    assert replay.operation_id == first.operation_id
    record = store.get_hyperpod_submission(CLUSTER_ID, SUBMIT["idempotency_key"])
    assert record.state == "SUBMITTED"
    assert record.requested_node_identifiers == [NODE_A]


def test_a_failed_reboot_call_leaves_an_unknown_outcome_that_fails_closed() -> None:
    store = InMemoryStore()
    adapter, _, ec2, _ = provider(store=store)
    ec2.reboot_error = RuntimeError("RequestLimitExceeded")

    with pytest.raises(RuntimeError, match="RequestLimitExceeded"):
        adapter.submit(HyperPodAction.REBOOT, [NODE_A], **SUBMIT)

    record = store.get_hyperpod_submission(CLUSTER_ID, SUBMIT["idempotency_key"])
    assert record.state == "UNKNOWN"
    ec2.reboot_error = None
    with pytest.raises(EksProviderError, match="unknown outcome"):
        adapter.submit(HyperPodAction.REBOOT, [NODE_A], **SUBMIT)
    assert ec2.reboots == [[INSTANCE_A]], "the call is never retried blindly"


def test_execute_step_maps_restart_node_only() -> None:
    adapter, _, ec2, _ = provider()

    result = adapter.execute_step(
        workflow_step(WorkflowOperation.RESTART_NODE, node_ids=[NODE_A]), **SUBMIT
    )
    assert result.successful_node_logical_ids == [NODE_A]
    assert ec2.reboots == [[INSTANCE_A]]

    with pytest.raises(EksProviderError, match="replacement is unsupported"):
        adapter.execute_step(
            workflow_step(WorkflowOperation.REPLACE_NODE, node_ids=[NODE_A]), **SUBMIT
        )
    with pytest.raises(EksProviderError, match="unsupported EKS operation"):
        adapter.execute_step(
            workflow_step(WorkflowOperation.RESET_GPU, node_ids=[NODE_A]), **SUBMIT
        )


def test_discover_reports_the_eks_cluster_and_its_gpu_nodes() -> None:
    adapter, *_ = provider()

    snapshot = adapter.discover()

    assert snapshot.cluster_name == CLUSTER_ID
    assert snapshot.cluster_status == "ACTIVE"
    assert snapshot.node_recovery is None
    assert snapshot.cluster_arn is not None and snapshot.cluster_arn.endswith(
        EKS_CLUSTER
    )
    assert snapshot.orchestrator == {"Eks": {"ClusterArn": snapshot.cluster_arn}}
    assert sorted(node.node_logical_id for node in snapshot.nodes) == sorted(
        [NODE_A, NODE_B]
    )


def test_config_never_allows_provider_replacement() -> None:
    with pytest.raises(ValueError, match="replacement is unsupported"):
        config(replace_enabled=True)
    with pytest.raises(ValueError, match="replacement is unsupported"):
        config(allowed_actions=[HyperPodAction.REBOOT, HyperPodAction.REPLACE])
    with pytest.raises(ValueError, match="node_selector must not be blank"):
        config(node_selector=" ")
    assert config().action_enabled(HyperPodAction.REPLACE) is False
    inherited = config(reboot_enabled=None, execution_enabled=True)
    assert inherited.action_enabled(HyperPodAction.REBOOT) is True


def test_config_from_environment_reads_the_shared_contract_variables() -> None:
    environ = {
        "GPU_FAULT_HYPERPOD_CLUSTER": CLUSTER_ID,
        "GPU_FAULT_NODE_SELECTOR": SELECTOR,
        "GPU_FAULT_EKS_CLUSTER_NAME": EKS_CLUSTER,
        "GPU_FAULT_ALLOW_HYPERPOD_REBOOT": "true",
        "AWS_REGION": REGION,
        "GPU_FAULT_HYPERPOD_MAX_BATCH_SIZE": "5",
    }

    loaded = EksProviderConfig.from_environment(environ=environ)

    assert loaded.cluster_id == CLUSTER_ID
    assert loaded.node_selector == SELECTOR
    assert loaded.eks_cluster_name == EKS_CLUSTER
    assert loaded.region_name == REGION
    assert loaded.reboot_enabled is True
    assert loaded.max_batch_size == 5
    with pytest.raises(ValueError, match="GPU_FAULT_HYPERPOD_CLUSTER is required"):
        EksProviderConfig.from_environment(
            environ={
                k: v for k, v in environ.items() if k != "GPU_FAULT_HYPERPOD_CLUSTER"
            }
        )
    with pytest.raises(ValueError, match="GPU_FAULT_NODE_SELECTOR is required"):
        EksProviderConfig.from_environment(
            environ={k: v for k, v in environ.items() if k != "GPU_FAULT_NODE_SELECTOR"}
        )
    with pytest.raises(ValueError, match="ALLOW_HYPERPOD_REPLACE must remain false"):
        EksProviderConfig.from_environment(
            environ={**environ, "GPU_FAULT_ALLOW_HYPERPOD_REPLACE": "true"}
        )


def test_provider_from_environment_uses_the_executor_kubernetes_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created: list[tuple[str, str | None]] = []

    def fake_client(service: str, region_name: str | None) -> object:
        created.append((service, region_name))
        return FakeEc2(default_instances()) if service == "ec2" else FakeEks()

    monkeypatch.setattr(
        EksManagedNodeProvider, "_create_client", staticmethod(fake_client)
    )
    core = FakeCore(default_nodes())
    environ = {
        "GPU_FAULT_CLUSTER_PROVIDER": "eks-managed",
        "GPU_FAULT_HYPERPOD_CLUSTER": CLUSTER_ID,
        "GPU_FAULT_NODE_SELECTOR": SELECTOR,
        "AWS_REGION": REGION,
    }

    adapter = eks_provider_from_environment(core, environ=environ)

    assert sorted(created) == [("ec2", REGION), ("eks", REGION)], (
        "only EC2 and EKS clients; never a SageMaker client"
    )
    assert adapter.core is core
    assert [node.node_logical_id for node in adapter.list_nodes()] == [NODE_A, NODE_B]


def test_recovery_ownership_is_reboot_only_with_no_managed_recovery() -> None:
    adapter, *_ = provider()

    ownership = adapter.resolve_recovery_ownership(
        HyperPodWorkloadRecoveryEvidence.from_eks_annotations({})
    )

    assert ownership.environment is Environment.EKS
    assert ownership.node_recovery is HyperPodRecoveryState.DISABLED
    assert ownership.runtime_profile.profile_version == "eks-auto-disabled-disabled"
    by_capability = {item.capability: item for item in ownership.capabilities}
    assert by_capability[CapabilityName.NODE_REBOOT].mode is CapabilityMode.OWN
    assert by_capability[CapabilityName.NODE_REBOOT].owner == EKS_ADAPTER_OWNER
    assert by_capability[CapabilityName.NODE_REPLACE].mode is CapabilityMode.OWN, (
        "replacement is the warm-spare handover, owned like reboot"
    )
    assert by_capability[CapabilityName.NODE_REPLACE].owner == EKS_ADAPTER_OWNER
    assert by_capability[CapabilityName.NODE_REPLACE].adapter == "eks-warm-spare"
    observed = {
        item.capability: item.version for item in ownership.runtime_profile.observed
    }
    assert observed[CapabilityName.NODE_REBOOT] == "ec2-reboot-v1"
    assert observed[CapabilityName.NODE_REPLACE] == "warm-spare-v1", (
        "the profile's observed nodeReplace version is the handover, not EC2"
    )
    assert by_capability[CapabilityName.WORKLOAD_RESTART].owner == (
        "gpu-fault-restart-controller"
    )
    assert "hyperpod-managed-node-recovery" not in {
        claim.owner for claim in ownership.runtime_profile.claims
    }
    unknown = adapter.resolve_recovery_ownership(
        HyperPodWorkloadRecoveryEvidence.from_eks_annotations(None)
    )
    assert unknown.warnings == [
        "workload auto-recovery is unknown; workload restart remains observe-only"
    ]

from __future__ import annotations

import pytest

from gpu_fault.capabilities import compile_runtime_profile
from gpu_fault.hyperpod import (
    HyperPodAction,
    HyperPodAdapterConfig,
    HyperPodAdapterError,
    HyperPodAdvisoryDisposition,
    HyperPodLifecycleAdapter,
    HyperPodRecoveryOwner,
    HyperPodRecoveryState,
    HyperPodSubmissionRecord,
    HyperPodWorkflowDispatcher,
    HyperPodWorkloadRecoveryEvidence,
)
from gpu_fault.models import (
    CapabilityMode,
    CapabilityName,
    Environment,
    WorkflowOperation,
)
from tests._builders import build_store, copy_model, workflow_request, workflow_step


class FakeHyperPodClient:
    def __init__(
        self, *, cluster_status: str = "InService", node_recovery: str = "Automatic"
    ) -> None:
        self.cluster_status = cluster_status
        self.node_recovery = node_recovery
        self.reboot_requests: list[dict] = []
        self.replace_requests: list[dict] = []
        self.list_requests: list[dict] = []

    def describe_cluster(self, **kwargs):
        return {
            "ClusterArn": "arn:aws:sagemaker:us-east-1:123:cluster/hp",
            "ClusterName": kwargs["ClusterName"],
            "ClusterStatus": self.cluster_status,
            "NodeRecovery": self.node_recovery,
            "Orchestrator": {"Eks": {"ClusterArn": "arn:eks"}},
        }

    def list_cluster_nodes(self, **kwargs):
        self.list_requests.append(kwargs)
        if not kwargs.get("NextToken"):
            return {
                "NextToken": "page-2",
                "ClusterNodeSummaries": [
                    {
                        "NodeLogicalId": "worker-group-1",
                        "InstanceId": "i-00000000000000001",
                        "InstanceGroupName": "worker-group",
                        "InstanceType": "p5.48xlarge",
                        "InstanceStatus": {"Status": "Running"},
                        "PrivateDnsHostname": ("ip-10-0-0-1.ec2.internal"),
                    }
                ],
            }
        return {
            "ClusterNodeSummaries": [
                {
                    "NodeLogicalId": "worker-group-2",
                    "InstanceId": "i-00000000000000002",
                    "InstanceGroupName": "worker-group",
                    "InstanceType": "p5.48xlarge",
                    "InstanceStatus": {"Status": "Failure"},
                    "PrivateDnsHostname": ("ip-10-0-0-2.ec2.internal"),
                }
            ]
        }

    def describe_cluster_node(self, **kwargs):
        logical_id = kwargs["NodeLogicalId"]
        number = "1" if logical_id.endswith("-1") else "2"
        return {
            "NodeDetails": {
                "NodeLogicalId": logical_id,
                "InstanceId": f"i-0000000000000000{number}",
                "InstanceGroupName": "worker-group",
                "InstanceType": "p5.48xlarge",
                "InstanceStatus": {
                    "Status": ("Running" if number == "1" else "Failure")
                },
                "PrivateDnsHostname": (f"ip-10-0-0-{number}.ec2.internal"),
                "PrivatePrimaryIp": f"10.0.0.{number}",
                "Placement": {"AvailabilityZone": "us-east-1a"},
                "CapacityType": "OnDemand",
                "KubernetesConfig": {
                    "CurrentLabels": {
                        "kubernetes.io/hostname": (f"k8s-worker-{number}")
                    },
                    "CurrentTaints": [],
                },
            }
        }

    def batch_reboot_cluster_nodes(self, **kwargs):
        self.reboot_requests.append(kwargs)
        return {
            "SuccessfulNodeLogicalIds": kwargs["NodeLogicalIds"],
            "FailedNodeLogicalIds": [],
        }

    def batch_replace_cluster_nodes(self, **kwargs):
        self.replace_requests.append(kwargs)
        return {
            "SuccessfulNodeLogicalIds": [],
            "FailedNodeLogicalIds": [
                {
                    "NodeLogicalId": kwargs["NodeLogicalIds"][0],
                    "ErrorCode": "Conflict",
                    "Message": "already replacing",
                }
            ],
        }


def test_hyperpod_max_batch_size_is_configurable(monkeypatch) -> None:
    monkeypatch.setenv("GPU_FAULT_HYPERPOD_CLUSTER", "hp-cluster")
    monkeypatch.setenv("GPU_FAULT_HYPERPOD_MAX_BATCH_SIZE", "2")

    config = HyperPodAdapterConfig.from_environment()

    assert config.max_batch_size == 2


def test_hyperpod_mutation_permissions_are_action_specific(monkeypatch) -> None:
    monkeypatch.setenv("GPU_FAULT_HYPERPOD_CLUSTER", "hp-cluster")
    monkeypatch.setenv("GPU_FAULT_ALLOW_HYPERPOD_MUTATION", "false")
    monkeypatch.setenv("GPU_FAULT_ALLOW_HYPERPOD_REBOOT", "true")
    monkeypatch.setenv("GPU_FAULT_ALLOW_HYPERPOD_REPLACE", "false")

    config = HyperPodAdapterConfig.from_environment()

    assert config.action_enabled(HyperPodAction.REBOOT), (
        "expected config.action_enabled(HyperPodAction.REBOOT) to be truthy"
    )
    assert not config.action_enabled(HyperPodAction.REPLACE), (
        "expected config.action_enabled(HyperPodAction.REPLACE) to be falsy"
    )


def test_legacy_mutation_never_enables_replace(monkeypatch) -> None:
    monkeypatch.setenv("GPU_FAULT_HYPERPOD_CLUSTER", "hp-cluster")
    monkeypatch.setenv("GPU_FAULT_ALLOW_HYPERPOD_MUTATION", "true")
    monkeypatch.delenv("GPU_FAULT_ALLOW_HYPERPOD_REBOOT", raising=False)
    monkeypatch.delenv("GPU_FAULT_ALLOW_HYPERPOD_REPLACE", raising=False)

    config = HyperPodAdapterConfig.from_environment()

    assert config.action_enabled(HyperPodAction.REBOOT), (
        "expected config.action_enabled(HyperPodAction.REBOOT) to be truthy"
    )
    assert not config.action_enabled(HyperPodAction.REPLACE), (
        "expected config.action_enabled(HyperPodAction.REPLACE) to be falsy"
    )
    assert HyperPodAction.REPLACE in config.allowed_actions


def test_replace_env_is_rejected_by_the_design_invariant(monkeypatch) -> None:
    monkeypatch.setenv("GPU_FAULT_HYPERPOD_CLUSTER", "hp-cluster")
    monkeypatch.setenv("GPU_FAULT_ALLOW_HYPERPOD_REPLACE", "true")

    with pytest.raises(ValueError, match="must remain false"):
        HyperPodAdapterConfig.from_environment()


def test_programmatic_replace_enable_is_rejected() -> None:
    with pytest.raises(ValueError, match="provider node replacement is prohibited"):
        HyperPodAdapterConfig(cluster_name="hp-cluster", replace_enabled=True)


def test_automatic_node_recovery_override_is_rejected(monkeypatch) -> None:
    monkeypatch.setenv("GPU_FAULT_HYPERPOD_CLUSTER", "hp-cluster")
    monkeypatch.setenv("GPU_FAULT_ALLOW_WITH_AUTOMATIC_NODE_RECOVERY", "true")

    with pytest.raises(ValueError, match="must remain false"):
        HyperPodAdapterConfig.from_environment()


def adapter(
    *,
    execution_enabled: bool = False,
    cluster_status: str = "InService",
    node_recovery: str = "None",
) -> tuple[HyperPodLifecycleAdapter, FakeHyperPodClient]:
    client = FakeHyperPodClient(
        cluster_status=cluster_status, node_recovery=node_recovery
    )
    result = HyperPodLifecycleAdapter(
        HyperPodAdapterConfig(
            cluster_name="hp-cluster",
            region_name="us-east-1",
            execution_enabled=execution_enabled,
        ),
        client=client,
    )
    return result, client


def test_discovery_paginates_and_preserves_logical_ids() -> None:
    hp, client = adapter()

    snapshot = hp.discover()

    assert snapshot.cluster_status == "InService"
    assert [node.node_logical_id for node in snapshot.nodes] == [
        "worker-group-1",
        "worker-group-2",
    ]
    assert client.list_requests[0]["IncludeNodeLogicalIds"] is True
    assert client.list_requests[1]["NextToken"] == "page-2"
    assert snapshot.nodes[0].private_primary_ip == "10.0.0.1"


@pytest.mark.parametrize(
    ("node_recovery", "job_enabled", "node_owner", "job_owner"),
    [
        (
            "Automatic",
            True,
            HyperPodRecoveryOwner.HYPERPOD_MANAGED,
            HyperPodRecoveryOwner.HYPERPOD_MANAGED,
        ),
        (
            "Automatic",
            False,
            HyperPodRecoveryOwner.HYPERPOD_MANAGED,
            HyperPodRecoveryOwner.GPU_FAULT_ADAPTER,
        ),
        (
            "None",
            True,
            HyperPodRecoveryOwner.GPU_FAULT_ADAPTER,
            HyperPodRecoveryOwner.HYPERPOD_MANAGED,
        ),
        (
            "None",
            False,
            HyperPodRecoveryOwner.GPU_FAULT_ADAPTER,
            HyperPodRecoveryOwner.GPU_FAULT_ADAPTER,
        ),
    ],
)
def test_recovery_ownership_is_selected_independently(
    node_recovery: str,
    job_enabled: bool,
    node_owner: HyperPodRecoveryOwner,
    job_owner: HyperPodRecoveryOwner,
) -> None:
    hp, _ = adapter(node_recovery=node_recovery)
    evidence = HyperPodWorkloadRecoveryEvidence.from_slurm_auto_resume(
        job_enabled, workload_id="job-42"
    )

    ownership = hp.resolve_recovery_ownership(evidence)
    by_capability = {item.capability: item for item in ownership.capabilities}

    assert ownership.environment is Environment.HYPERPOD_EKS
    assert by_capability[CapabilityName.NODE_REBOOT].owner_type is node_owner
    assert by_capability[CapabilityName.NODE_REPLACE].owner_type is node_owner
    assert by_capability[CapabilityName.WORKLOAD_RESTART].owner_type is job_owner
    assert by_capability[CapabilityName.WORKLOAD_STOP].owner_type is job_owner
    effective = compile_runtime_profile(ownership.runtime_profile)
    assert not effective.warnings, "expected effective.warnings to be falsy"


def test_unknown_workload_recovery_blocks_custom_takeover() -> None:
    hp, _ = adapter(node_recovery="None")

    ownership = hp.resolve_recovery_ownership(
        HyperPodWorkloadRecoveryEvidence.from_eks_annotations(None)
    )
    workload = next(
        item
        for item in ownership.capabilities
        if item.capability is CapabilityName.WORKLOAD_RESTART
    )

    assert workload.state is HyperPodRecoveryState.UNKNOWN
    assert workload.owner_type is HyperPodRecoveryOwner.UNRESOLVED
    assert workload.mode is CapabilityMode.OBSERVE
    assert ownership.warnings, "expected ownership.warnings to be truthy"


def test_managed_recovery_keeps_custom_policy_in_advisory_mode() -> None:
    hp, _ = adapter(node_recovery="Automatic")
    ownership = hp.resolve_recovery_ownership(
        HyperPodWorkloadRecoveryEvidence.from_eks_annotations({})
    )

    advisory = hp.create_recovery_advisory(
        ownership,
        capability=CapabilityName.NODE_REPLACE,
        recommended_action="REPLACE_NODE",
        rationale=["repeated XID 79 on the same GPU", "DCGM diagnostic failed"],
        evidence_refs=["s3://evidence/incident-42.json"],
    )

    assert advisory.disposition is HyperPodAdvisoryDisposition.ADVISE_ONLY
    assert advisory.execution_owner == ("hyperpod-managed-node-recovery")
    assert advisory.recommended_action == "REPLACE_NODE"


def test_custom_recovery_advisory_is_executable_by_adapter() -> None:
    hp, _ = adapter(node_recovery="None")
    ownership = hp.resolve_recovery_ownership(
        HyperPodWorkloadRecoveryEvidence.from_eks_annotations({})
    )

    advisory = hp.create_recovery_advisory(
        ownership,
        capability=CapabilityName.NODE_REBOOT,
        recommended_action="RESTART_BM",
        rationale=["NVIDIA XID policy immediate action"],
    )

    assert advisory.disposition is HyperPodAdvisoryDisposition.ADVISE_AND_EXECUTE
    assert advisory.execution_owner == "gpu-fault-hyperpod-adapter"


def test_eks_annotation_controls_workload_recovery_owner() -> None:
    enabled = HyperPodWorkloadRecoveryEvidence.from_eks_annotations(
        {"sagemaker.amazonaws.com/enable-job-auto-resume": "true"},
        workload_id="pytorch-job",
    )
    disabled = HyperPodWorkloadRecoveryEvidence.from_eks_annotations({})

    assert enabled.state is HyperPodRecoveryState.ENABLED
    assert disabled.state is HyperPodRecoveryState.DISABLED


def test_resolves_logical_instance_and_dns_aliases() -> None:
    hp, _ = adapter()
    nodes = hp.list_nodes(enrich=True)

    resolved = hp.resolve_nodes(
        [
            "worker-group-1",
            "i-00000000000000002",
            "hyperpod-i-00000000000000002",
            "ip-10-0-0-1",
            "k8s-worker-2",
        ],
        nodes=nodes,
    )

    assert {node.node_logical_id for node in resolved} == {
        "worker-group-1",
        "worker-group-2",
    }


def test_read_only_preflight_reports_all_mutation_gates() -> None:
    hp, client = adapter(execution_enabled=False)

    result = hp.preflight(HyperPodAction.REBOOT, ["worker-group-1"])

    assert not result.safe_to_submit, "expected result.safe_to_submit to be falsy"
    assert "HyperPod REBOOT mutation is disabled" in " ".join(result.gate_failures)
    assert "isolation evidence is missing" in " ".join(result.gate_failures)
    assert not client.reboot_requests, "expected client.reboot_requests to be falsy"


def test_local_spare_preflight_does_not_require_provider_mutation() -> None:
    hp, client = adapter(execution_enabled=False)

    result = hp.preflight(
        HyperPodAction.REPLACE,
        ["worker-group-1"],
        isolation_verified_nodes=["worker-group-1"],
        require_execution_enabled=False,
    )

    assert result.safe_to_submit, "expected result.safe_to_submit to be truthy"
    assert result.execution_enabled is False
    assert not client.replace_requests, "expected client.replace_requests to be falsy"


def test_provider_replace_can_be_disabled_without_disabling_reboot() -> None:
    client = FakeHyperPodClient(node_recovery="None")
    hp = HyperPodLifecycleAdapter(
        HyperPodAdapterConfig(
            cluster_name="hp-cluster", reboot_enabled=True, replace_enabled=False
        ),
        client=client,
    )

    reboot = hp.preflight(
        HyperPodAction.REBOOT,
        ["worker-group-1"],
        isolation_verified_nodes=["worker-group-1"],
    )
    replace = hp.preflight(
        HyperPodAction.REPLACE,
        ["worker-group-1"],
        isolation_verified_nodes=["worker-group-1"],
    )

    assert reboot.safe_to_submit, "expected reboot.safe_to_submit to be truthy"
    assert not replace.safe_to_submit, "expected replace.safe_to_submit to be falsy"
    assert "HyperPod REPLACE mutation is disabled" in " ".join(replace.gate_failures)


def test_submit_requires_cluster_isolation_and_fencing() -> None:
    hp, client = adapter(execution_enabled=True)

    with pytest.raises(HyperPodAdapterError, match="cluster confirmation"):
        hp.submit(
            HyperPodAction.REBOOT,
            ["worker-group-1"],
            isolation_verified_nodes=["worker-group-1"],
            confirm_cluster_name="wrong-cluster",
            workflow_fencing_token=1,
            expected_fencing_token=1,
            idempotency_key="workflow/step",
        )
    with pytest.raises(HyperPodAdapterError, match="stale workflow"):
        hp.submit(
            HyperPodAction.REBOOT,
            ["worker-group-1"],
            isolation_verified_nodes=["worker-group-1"],
            confirm_cluster_name="hp-cluster",
            workflow_fencing_token=2,
            expected_fencing_token=1,
            idempotency_key="workflow/step",
        )

    assert not client.reboot_requests, "expected client.reboot_requests to be falsy"


def test_automatic_node_recovery_blocks_second_lifecycle_trigger() -> None:
    hp, client = adapter(execution_enabled=True, node_recovery="Automatic")

    preflight = hp.preflight(
        HyperPodAction.REBOOT,
        ["worker-group-1"],
        isolation_verified_nodes=["worker-group-1"],
    )

    assert not preflight.safe_to_submit, "expected preflight.safe_to_submit to be falsy"
    assert preflight.node_recovery == "Automatic"
    assert "second lifecycle trigger" in " ".join(preflight.gate_failures)
    assert not client.reboot_requests, "expected client.reboot_requests to be falsy"


def test_reboot_submission_is_idempotent() -> None:
    hp, client = adapter(execution_enabled=True)
    arguments = {
        "isolation_verified_nodes": ["ip-10-0-0-1"],
        "confirm_cluster_name": "hp-cluster",
        "workflow_fencing_token": 1,
        "expected_fencing_token": 1,
        "idempotency_key": "workflow-1/restart-node/0",
    }

    first = hp.submit(HyperPodAction.REBOOT, ["worker-group-1"], **arguments)
    second = hp.submit(HyperPodAction.REBOOT, ["worker-group-1"], **arguments)

    assert first.successful_node_logical_ids == ["worker-group-1"]
    assert second.duplicate, "expected second.duplicate to be truthy"
    assert len(client.reboot_requests) == 1
    assert client.reboot_requests[0] == {
        "ClusterName": "hp-cluster",
        "NodeLogicalIds": ["worker-group-1"],
    }


def test_provider_replace_submission_is_code_level_unreachable() -> None:
    hp, client = adapter(execution_enabled=True)

    with pytest.raises(HyperPodAdapterError, match="provider node replacement"):
        hp.submit(
            HyperPodAction.REPLACE,
            ["worker-group-2"],
            isolation_verified_nodes=["worker-group-2"],
            confirm_cluster_name="hp-cluster",
            workflow_fencing_token=4,
            expected_fencing_token=4,
            idempotency_key="workflow-2/replace-node/0",
        )

    assert not client.replace_requests, "provider replacement API was called"


def test_workflow_step_dispatches_only_supported_operations() -> None:
    hp, _ = adapter(execution_enabled=True)
    step = workflow_step(
        WorkflowOperation.RESTART_NODE,
        "gpu-fault-hyperpod-adapter",
        node_ids=["worker-group-1"],
    )

    result = hp.execute_step(
        step,
        isolation_verified_nodes=["worker-group-1"],
        confirm_cluster_name="hp-cluster",
        workflow_fencing_token=1,
        expected_fencing_token=1,
        idempotency_key="workflow-3/restart-node/0",
    )

    assert result.action is HyperPodAction.REBOOT

    unsupported = copy_model(step, operation=WorkflowOperation.RESET_GPU)
    with pytest.raises(HyperPodAdapterError, match="unsupported"):
        hp.execute_step(
            unsupported,
            isolation_verified_nodes=["worker-group-1"],
            confirm_cluster_name="hp-cluster",
            workflow_fencing_token=1,
            expected_fencing_token=1,
            idempotency_key="workflow-3/reset-gpu/1",
        )


def test_workflow_dispatcher_enforces_status_owner_and_step() -> None:
    hp, client = adapter(execution_enabled=True)
    dispatcher = HyperPodWorkflowDispatcher(hp)
    workflow = workflow_request(
        "workflow-hp",
        "incident-hp",
        fencing_token=7,
        runtime_profile_version="hyperpod-v1",
        official_action="RESTART_BM",
        official_steps=[
            workflow_step(
                WorkflowOperation.RESTART_NODE,
                "gpu-fault-hyperpod-adapter",
                node_ids=["worker-group-1"],
            )
        ],
    )

    result = dispatcher.submit(
        workflow,
        0,
        isolation_verified_nodes=["worker-group-1"],
        confirm_cluster_name="hp-cluster",
        expected_fencing_token=7,
    )

    assert result.submitted, "expected result.submitted to be truthy"
    assert len(client.reboot_requests) == 1

    wrong_owner = copy_model(
        workflow,
        request_id="workflow-other-owner",
        official_steps=[
            copy_model(workflow.official_steps[0], execution_owner="another-controller")
        ],
    )
    with pytest.raises(ValueError, match="does not match"):
        dispatcher.submit(
            wrong_owner,
            0,
            isolation_verified_nodes=["worker-group-1"],
            confirm_cluster_name="hp-cluster",
            expected_fencing_token=7,
        )


def durable_adapter(
    store, *, execution_enabled: bool = True
) -> tuple[HyperPodLifecycleAdapter, FakeHyperPodClient]:
    client = FakeHyperPodClient(node_recovery="None")
    return (
        HyperPodLifecycleAdapter(
            HyperPodAdapterConfig(
                cluster_name="hp-cluster",
                region_name="us-east-1",
                execution_enabled=execution_enabled,
            ),
            client=client,
            store=store,
        ),
        client,
    )


SUBMIT_ARGS = {
    "isolation_verified_nodes": ["worker-group-1"],
    "confirm_cluster_name": "hp-cluster",
    "workflow_fencing_token": 1,
    "expected_fencing_token": 1,
    "idempotency_key": "workflow-1/restart-node/0",
}


def test_durable_submission_survives_adapter_restart() -> None:
    store = build_store()
    first_adapter, first_client = durable_adapter(store)

    first = first_adapter.submit(
        HyperPodAction.REBOOT, ["worker-group-1"], **SUBMIT_ARGS
    )

    # A reboot restarts the node the executor runs on, so the retry
    # almost always comes from a fresh process with an empty dict.
    second_adapter, second_client = durable_adapter(store)
    second = second_adapter.submit(
        HyperPodAction.REBOOT, ["worker-group-1"], **SUBMIT_ARGS
    )

    assert first.successful_node_logical_ids == ["worker-group-1"]
    assert second.duplicate, "expected second.duplicate to be truthy"
    assert second.successful_node_logical_ids == (first.successful_node_logical_ids)
    assert len(first_client.reboot_requests) == 1
    assert not second_client.reboot_requests, (
        "expected second_client.reboot_requests to be falsy"
    )


def test_durable_submission_rejects_reused_key_for_other_request() -> None:
    store = build_store()
    first_adapter, _ = durable_adapter(store)
    first_adapter.submit(HyperPodAction.REBOOT, ["worker-group-1"], **SUBMIT_ARGS)

    second_adapter, second_client = durable_adapter(store)
    with pytest.raises(HyperPodAdapterError, match="already used for a different"):
        second_adapter.submit(HyperPodAction.REBOOT, ["worker-group-2"], **SUBMIT_ARGS)
    assert not second_client.reboot_requests, "conflicting request reached provider"


def test_interrupted_submission_fails_closed_instead_of_resubmitting() -> None:
    store = build_store()
    crashing_adapter, crashing_client = durable_adapter(store)

    def explode(**_kwargs):
        raise RuntimeError("provider call timed out")

    crashing_client.batch_reboot_cluster_nodes = explode
    with pytest.raises(RuntimeError, match="timed out"):
        crashing_adapter.submit(
            HyperPodAction.REBOOT, ["worker-group-1"], **SUBMIT_ARGS
        )

    # The provider may or may not have acted, so the retry must not
    # submit a second batch; it must demand an operator decision.
    retry_adapter, retry_client = durable_adapter(store)
    with pytest.raises(HyperPodAdapterError, match="unknown outcome"):
        retry_adapter.submit(HyperPodAction.REBOOT, ["worker-group-1"], **SUBMIT_ARGS)
    assert not retry_client.reboot_requests, (
        "expected retry_client.reboot_requests to be falsy"
    )


def test_reserved_submission_without_outcome_blocks_second_batch() -> None:
    store = build_store()
    store.reserve_hyperpod_submission(
        HyperPodSubmissionRecord(
            cluster_name="hp-cluster",
            idempotency_key=SUBMIT_ARGS["idempotency_key"],
            action=HyperPodAction.REBOOT,
            requested_node_identifiers=["worker-group-1"],
        )
    )
    blocked_adapter, blocked_client = durable_adapter(store)

    with pytest.raises(HyperPodAdapterError, match="already reserved"):
        blocked_adapter.submit(HyperPodAction.REBOOT, ["worker-group-1"], **SUBMIT_ARGS)
    assert not blocked_client.reboot_requests, (
        "expected blocked_client.reboot_requests to be falsy"
    )

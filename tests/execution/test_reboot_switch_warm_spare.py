"""A reboot the operator's reboot switch refuses still climbs to the warm spare.

``GPU_FAULT_ALLOW_HYPERPOD_REBOOT=false`` makes the provider preflight refuse
every RESTART_NODE ("HyperPod REBOOT mutation is disabled by configuration"),
before the agent revocation and the provider call. Unlike an operator-policy
refusal of a lighter action on the node itself, this one keeps the reboot
rung's successor, by design: the next rung, REPLACE_NODE, is the healthy
warm-spare failover (``HEALTHY_WARM_SPARE_ONLY``, provider replacement is
prohibited), which does not mutate the faulted node -- it moves the work to a
spare -- so its preflight skips the reboot switch
(``require_execution_enabled=not warm_spare_only``). With no labelled spare the
failover refuses with the no-fallback literal and the record ends in one
support hand-off.
"""

from __future__ import annotations

from typing import Any

from gpu_fault.adapters.hyperpod.lifecycle import HyperPodLifecycleStepAdapter
from gpu_fault.execution import WorkflowExecutionRequest
from gpu_fault.hyperpod import (
    HyperPodAction,
    HyperPodAdapterConfig,
    HyperPodLifecycleAdapter,
)
from gpu_fault.hyperpod_spares import HyperPodSpareCoordinator
from gpu_fault.models import (
    RecoveryAction,
    WorkflowOperation,
    WorkflowStatus,
    WorkflowStepStatus,
)
from gpu_fault.orchestration.escalation import HardwareEscalationService
from gpu_fault.orchestration.failure_outcome import LIFECYCLE_ISOLATION_DETAIL
from tests._builders import active_workflow_executor, build_store, copy_model
from tests.execution._support import isolated_kubernetes_adapter, workflow_state
from tests.orchestration._cov95_runtime_escalation import EscalationHarness

OP = WorkflowOperation
CLUSTER = "hp-cluster"
SWITCH_OFF = "HyperPod REBOOT mutation is disabled by configuration"
NO_SPARE = (
    "warm-spare replacement is required; provider node replacement API "
    "fallback is disabled"
)


class _SageMaker:
    """One InService cluster, NodeRecovery None, one Running node whose
    Kubernetes hostname is ``node-a``; no node carries the spare label."""

    def describe_cluster(self, **kwargs: Any) -> dict[str, Any]:
        return {
            "ClusterName": kwargs["ClusterName"],
            "ClusterStatus": "InService",
            "NodeRecovery": "None",
        }

    def list_cluster_nodes(self, **_kwargs: Any) -> dict[str, Any]:
        return {
            "ClusterNodeSummaries": [
                {
                    "NodeLogicalId": "worker-1",
                    "InstanceId": "i-00000000000000001",
                    "InstanceGroupName": "worker-group",
                    "InstanceType": "p5en.48xlarge",
                    "InstanceStatus": {"Status": "Running"},
                }
            ]
        }

    def describe_cluster_node(self, **kwargs: Any) -> dict[str, Any]:
        return {
            "NodeDetails": {
                "NodeLogicalId": kwargs["NodeLogicalId"],
                "InstanceId": "i-00000000000000001",
                "InstanceGroupName": "worker-group",
                "InstanceType": "p5en.48xlarge",
                "InstanceStatus": {"Status": "Running"},
                "KubernetesConfig": {
                    "CurrentLabels": {"kubernetes.io/hostname": "node-a"},
                    "CurrentTaints": [],
                },
            }
        }

    def batch_reboot_cluster_nodes(self, **_kwargs: Any) -> dict[str, Any]:
        raise AssertionError("a refused reboot must not reach the provider")


def _provider() -> HyperPodLifecycleAdapter:
    return HyperPodLifecycleAdapter(
        HyperPodAdapterConfig(
            cluster_name=CLUSTER, execution_enabled=False, reboot_enabled=False
        ),
        client=_SageMaker(),  # type: ignore[arg-type]
    )


def _run(operation: WorkflowOperation):
    store = build_store()
    _, workflow = workflow_state(store, [operation])
    provider = _provider()
    kubernetes = isolated_kubernetes_adapter()
    # The scheduler's one node, unlabelled: the warm-spare coordinator lists
    # the cluster's Kubernetes nodes for a local-only failover.
    kubernetes.core.node["metadata"]["name"] = "node-a"
    kubernetes.core.list_node = lambda: {"items": [kubernetes.core.node]}
    adapter = HyperPodLifecycleStepAdapter(
        provider,
        spare_coordinator=HyperPodSpareCoordinator(provider, store, kubernetes.core),
        kubernetes_adapter=kubernetes,
    )
    parameters = (
        {"replacement_strategy": "HEALTHY_WARM_SPARE_ONLY"}
        if operation is OP.REPLACE_NODE
        else {}
    )
    step = copy_model(
        workflow.official_steps[0], execution_owner=adapter.owner, parameters=parameters
    )
    store.save_workflow(copy_model(workflow, official_steps=[step]))
    active_workflow_executor(store, [adapter], {operation}).execute(
        workflow.request_id,
        WorkflowExecutionRequest(
            expected_fencing_token=3,
            isolation_verified_nodes=["node-a"],
            confirm_cluster_name=CLUSTER,
        ),
    )
    saved = store.get_workflow(workflow.request_id)
    return saved, saved.step_executions[-1]


def test_the_reboot_switch_gates_the_reboot_and_not_the_warm_spare_failover() -> None:
    provider = _provider()

    reboot = provider.preflight(
        HyperPodAction.REBOOT, ["node-a"], isolation_verified_nodes=["node-a"]
    )
    failover = provider.preflight(
        HyperPodAction.REPLACE,
        ["node-a"],
        isolation_verified_nodes=["node-a"],
        require_execution_enabled=False,
    )

    assert reboot.gate_failures == [SWITCH_OFF]
    assert (failover.safe_to_submit, failover.gate_failures) == (True, []), (
        "the warm-spare failover does not mutate the faulted node"
    )


def test_a_reboot_the_switch_refused_climbs_to_the_warm_spare_rung() -> None:
    workflow, execution = _run(OP.RESTART_NODE)

    assert workflow.status is WorkflowStatus.FAILED
    assert execution.status is WorkflowStepStatus.FAILED
    assert execution.error == f"HyperPod preflight failed: {SWITCH_OFF}"
    assert LIFECYCLE_ISOLATION_DETAIL not in execution.details
    classification = HardwareEscalationService.classify(workflow)
    assert classification is not None
    assert classification[:3] == (
        "reboot",
        RecoveryAction.REPLACE_NODE,
        OP.REPLACE_NODE,
    ), "the reboot switch keeps the warm-spare rung"


def test_with_no_labelled_spare_the_failover_ends_in_one_support_hand_off() -> None:
    workflow, execution = _run(OP.REPLACE_NODE)

    assert workflow.status is WorkflowStatus.FAILED
    assert execution.error == NO_SPARE, "the switch did not refuse the failover"
    classification = HardwareEscalationService.classify(workflow)
    assert classification is not None
    assert classification[:3] == (
        "replacement",
        RecoveryAction.ESCALATE_OPERATOR,
        OP.ESCALATE_SUPPORT,
    )
    h = EscalationHarness(
        [
            OP.FREEZE_EVIDENCE,
            OP.MARK_UNSCHEDULABLE,
            OP.QUARANTINE,
            OP.REPLACE_NODE,
            OP.VALIDATE_GPU,
            OP.RESTORE_SCHEDULING,
        ],
        failed_index=3,
        details=dict(execution.details),
    )
    result = h.escalate()
    assert result is not None
    _incident, successor = result
    assert successor.request_id == "workflow-support-after-source-workflow"
    assert [step.operation for step in successor.official_steps] == [
        OP.FREEZE_EVIDENCE,
        OP.MARK_UNSCHEDULABLE,
        OP.QUARANTINE,
        OP.ESCALATE_SUPPORT,
    ]

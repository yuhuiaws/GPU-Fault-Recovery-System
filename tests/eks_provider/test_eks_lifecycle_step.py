"""The shared lifecycle step adapter over the plain-EKS provider.

``RESTART_NODE`` submits EC2 ``RebootInstances`` for the bound instance and
waits; the poll confirms on EC2 ``running`` plus a new Node Agent incarnation
and records ``eks-running-and-new-agent-incarnation``. ``REPLACE_NODE`` fails
closed at preflight. The failure wording names the provider.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any

from gpu_fault.adapters import HyperPodLifecycleStepAdapter
from gpu_fault.eks_provider import EKS_ADAPTER_OWNER
from gpu_fault.execution import WorkflowExecutionRequest, WorkflowStepContext
from gpu_fault.fleet import AgentLifecycleState
from gpu_fault.hyperpod import HyperPodAction
from gpu_fault.models import WorkflowOperation, WorkflowStatus, WorkflowStepStatus
from gpu_fault.node_provider import EKS_REBOOT_CONFIRMATION_SOURCE
from gpu_fault.store import InMemoryStore
from tests._builders import workflow_step_execution
from tests.eks_provider._support import (
    CLUSTER_ID,
    INSTANCE_A,
    NODE_A,
    NODE_B,
    FakeCore,
    FakeEc2,
    default_instances,
    default_nodes,
    ec2_instance,
    provider,
)
from tests.execution._support import isolated_kubernetes_adapter, workflow_state
from tests.hyperpod.test_managed_recovery import agent

NOW = datetime(2026, 10, 3, 12, tzinfo=timezone.utc)


class FakeRegistry:
    """Fleet registry surface the step adapter uses: records, readiness, lifecycle."""

    def __init__(self, store: InMemoryStore) -> None:
        self.store = store
        self.clock = NOW
        self.ready = True
        self.transitions: list[tuple[str, str]] = []

    def now(self) -> datetime:
        return self.clock

    def readiness(self, *_args: Any) -> Any:
        return SimpleNamespace(ready=self.ready)

    def drain_agent(self, cluster_id: str, node_id: str, request: Any) -> Any:
        self.transitions.append(("drain", node_id))
        return self._transition(cluster_id, node_id, AgentLifecycleState.DRAINING)

    def revoke_agent(self, cluster_id: str, node_id: str, request: Any) -> Any:
        self.transitions.append(("revoke", node_id))
        return self._transition(cluster_id, node_id, AgentLifecycleState.REVOKED)

    def _transition(
        self, cluster_id: str, node_id: str, state: AgentLifecycleState
    ) -> Any:
        record = self.store.get_agent(cluster_id, node_id).model_copy(
            update={"lifecycle_state": state}
        )
        return self.store.save_agent(record)


class Harness:
    def __init__(self, operation: WorkflowOperation, **provider_options: Any) -> None:
        self.store = InMemoryStore()
        self.incident, workflow = workflow_state(self.store, [operation])
        self.incident = self.incident.model_copy(update={"node_ids": [NODE_A]})
        self.registry = FakeRegistry(self.store)
        self.record = agent(NODE_A, INSTANCE_A, "inc-before").model_copy(
            update={
                "cluster_id": self.incident.cluster_id,
                "boot_id": "boot-before",
                "first_seen_at": NOW,
                "last_seen_at": NOW,
                "lease_expires_at": NOW + timedelta(minutes=5),
            }
        )
        self.store.save_agent(self.record)
        self.provider, self.core, self.ec2, self.eks = provider(**provider_options)
        self.scheduler = isolated_kubernetes_adapter(
            self.incident.incident_id, self.incident.fencing_token
        )
        self.adapter = HyperPodLifecycleStepAdapter(
            self.provider,
            owner=EKS_ADAPTER_OWNER,
            registry=self.registry,
            kubernetes_adapter=self.scheduler,
            post_reboot_stabilization_seconds=0,
        )
        step = workflow.official_steps[0].model_copy(
            update={"execution_owner": self.adapter.owner, "node_ids": [NODE_A]}
        )
        self.workflow = workflow.model_copy(
            update={"official_steps": [step], "status": WorkflowStatus.RUNNING}
        )
        self.context = WorkflowStepContext(
            workflow=self.workflow,
            incident=self.incident,
            step=step,
            step_index=0,
            request=WorkflowExecutionRequest(
                expected_fencing_token=self.incident.fencing_token,
                confirm_cluster_name=CLUSTER_ID,
            ),
            idempotency_key="unit-eks-step",
        )

    def with_waiting(self, outcome: Any) -> WorkflowStepContext:
        execution = workflow_step_execution(
            0,
            self.context.step.operation,
            WorkflowStepStatus.WAITING,
            adapter_operation_id=outcome.adapter_operation_id,
            details=dict(outcome.details or {}),
        )
        workflow = self.workflow.model_copy(update={"step_executions": [execution]})
        return WorkflowStepContext(
            workflow=workflow,
            incident=self.incident,
            step=self.context.step,
            step_index=0,
            request=self.context.request,
            idempotency_key=self.context.idempotency_key,
        )

    def agent_rebooted(self) -> None:
        self.store.save_agent(
            self.store.get_agent(self.incident.cluster_id, NODE_A).model_copy(
                update={
                    "lifecycle_state": AgentLifecycleState.ACTIVE,
                    "boot_id": "boot-after",
                    "agent_incarnation_id": "inc-after",
                    "retired_incarnation_ids": ["boot-before"],
                }
            )
        )


def test_restart_node_submits_ec2_reboot_and_waits_for_confirmation() -> None:
    harness = Harness(WorkflowOperation.RESTART_NODE)

    outcome = harness.adapter.execute(harness.context)

    assert outcome.status is WorkflowStepStatus.WAITING, outcome
    assert harness.ec2.reboots == [[INSTANCE_A]]
    details = outcome.details or {}
    assert details["action"] == "REBOOT"
    assert details["submitted_nodes"] == [NODE_A]
    assert details["requires_external_confirmation"] is True
    assert details["activated_spare_nodes"] == []
    assert details["agent_baselines"] == {
        NODE_A: {"boot_id": "boot-before", "agent_incarnation_id": "inc-before"}
    }
    assert details["revoked_agents"] == [NODE_A]
    assert harness.registry.transitions == [("drain", NODE_A), ("revoke", NODE_A)]
    assert details["observed_isolation"][NODE_A]["unschedulable"] is True
    assert outcome.adapter_operation_id is not None
    assert outcome.adapter_operation_id.startswith("eks-op-"), (
        outcome.adapter_operation_id
    )


def test_confirmation_requires_ec2_running_and_a_new_agent_incarnation() -> None:
    harness = Harness(WorkflowOperation.RESTART_NODE)
    pending = harness.adapter.execute(harness.context)
    context = harness.with_waiting(pending)

    still_old_agent = harness.adapter.execute(context)
    assert still_old_agent.status is WorkflowStepStatus.WAITING, (
        "the same boot id is no proof of a reboot"
    )

    harness.agent_rebooted()
    harness.ec2.instances[INSTANCE_A] = ec2_instance(INSTANCE_A, state="stopping")
    not_running = harness.adapter.execute(context)
    assert not_running.status is WorkflowStepStatus.WAITING, (
        "EC2 must report running before the reboot is confirmed"
    )

    harness.ec2.instances[INSTANCE_A] = ec2_instance(INSTANCE_A, private_dns=NODE_A)
    confirmed = harness.adapter.execute(context)

    assert confirmed.status is WorkflowStepStatus.SUCCEEDED, confirmed
    details = confirmed.details or {}
    assert details["externally_confirmed"] is True
    assert details["confirmation_source"] == EKS_REBOOT_CONFIRMATION_SOURCE
    assert details["agent_observations"] == [
        {
            "node_id": NODE_A,
            "boot_id": "boot-after",
            "agent_incarnation_id": "inc-after",
        }
    ]
    assert details["provider_observations"] == [
        {
            "node_id": NODE_A,
            "node_logical_id": NODE_A,
            "instance_id": INSTANCE_A,
            "status": "Running",
        }
    ]
    assert harness.ec2.reboots == [[INSTANCE_A]], "polling never resubmits"


def test_replace_node_fails_closed_without_a_provider_replacement() -> None:
    harness = Harness(WorkflowOperation.REPLACE_NODE)

    outcome = harness.adapter.execute(harness.context)

    assert outcome.status is WorkflowStepStatus.FAILED, outcome
    assert outcome.error is not None
    assert outcome.error.startswith("EKS preflight failed: "), outcome.error
    assert "no provider node replacement" in outcome.error
    assert harness.ec2.reboots == []
    assert harness.registry.transitions == [], "no agent is revoked for a refused step"


def test_failure_wording_names_the_eks_provider() -> None:
    harness = Harness(WorkflowOperation.RESTART_NODE)
    context = WorkflowStepContext(
        workflow=harness.context.workflow,
        incident=harness.context.incident,
        step=harness.context.step,
        step_index=0,
        request=WorkflowExecutionRequest(expected_fencing_token=3),
        idempotency_key=harness.context.idempotency_key,
    )

    outcome = harness.adapter.execute(context)

    assert outcome.status is WorkflowStepStatus.FAILED
    assert outcome.error == "confirm_cluster_name is required for EKS mutation"
    assert harness.ec2.reboots == []


def test_a_degraded_node_group_blocks_the_reboot_before_submission() -> None:
    from tests.eks_provider._support import NODEGROUP, FakeEks

    harness = Harness(
        WorkflowOperation.RESTART_NODE, eks=FakeEks(nodegroups={NODEGROUP: "DEGRADED"})
    )

    outcome = harness.adapter.execute(harness.context)

    assert outcome.status is WorkflowStepStatus.FAILED, outcome
    assert outcome.error is not None
    assert "node group gpu-p5en-efa-spot-node-group must be ACTIVE" in outcome.error
    assert harness.ec2.reboots == []


def test_a_node_whose_instance_is_gone_is_not_rebooted() -> None:
    instances = [
        item for item in default_instances() if item["InstanceId"] != INSTANCE_A
    ]
    harness = Harness(
        WorkflowOperation.RESTART_NODE,
        core=FakeCore(default_nodes()),
        ec2=FakeEc2(instances),
    )

    outcome = harness.adapter.execute(harness.context)

    assert outcome.status is WorkflowStepStatus.FAILED, outcome
    assert outcome.error is not None
    assert "non-actionable EC2 state" in outcome.error
    assert f"{NODE_A}:NotFound" in outcome.error
    assert harness.ec2.reboots == []


def test_a_warm_spare_only_replace_passes_the_eks_preflight_without_a_coordinator() -> (
    None
):
    """HEALTHY_WARM_SPARE_ONLY reads the cluster, never the REPLACE allowlist.

    Without a coordinator the step still fails closed -- on the coordinator,
    not on "no provider node replacement", which would hide that the handover
    path exists on plain EKS.
    """

    harness = Harness(WorkflowOperation.REPLACE_NODE)
    step = harness.context.step.model_copy(
        update={"parameters": {"replacement_strategy": "HEALTHY_WARM_SPARE_ONLY"}}
    )
    context = WorkflowStepContext(
        workflow=harness.workflow.model_copy(update={"official_steps": [step]}),
        incident=harness.incident,
        step=step,
        step_index=0,
        request=harness.context.request,
        idempotency_key=harness.context.idempotency_key,
    )

    outcome = harness.adapter.execute(context)

    assert outcome.status is WorkflowStepStatus.FAILED, outcome
    assert outcome.error == (
        "healthy warm-spare replacement is required but the spare coordinator is disabled"
    )
    assert harness.ec2.reboots == []
    assert (
        harness.provider.preflight(
            HyperPodAction.REPLACE, [NODE_A], isolation_verified_nodes=[NODE_A]
        ).safe_to_submit
        is False
    ), "a provider replacement is still refused outright"


def test_a_warm_spare_handover_on_eks_moves_the_workload_without_ec2() -> None:
    """The HyperPod warm-spare path, with the EKS provider behind it.

    The coordinator hands the step a declared spare, the Kubernetes adapter
    isolates it, the node agent snapshot confirms it, and no EC2 call is made:
    ``provider_mutation_submitted`` is False and the confirmation source is the
    running spare.
    """
    from gpu_fault.hyperpod_spares import SpareAllocation
    from tests.execution._support import (
        FakeSpareCoordinator,
        RecordingNodeActionAdapter,
    )

    harness = Harness(WorkflowOperation.REPLACE_NODE)
    spares = FakeSpareCoordinator(
        SpareAllocation(
            applicable=True, sufficient=True, required=1, selected_node_ids=(NODE_B,)
        )
    )
    node_actions = RecordingNodeActionAdapter({NODE_B: "https://spare:9099"})
    adapter = HyperPodLifecycleStepAdapter(
        harness.provider,
        owner=EKS_ADAPTER_OWNER,
        registry=harness.registry,
        spare_coordinator=spares,
        node_action_adapter=node_actions,
        kubernetes_adapter=harness.scheduler,
        store=harness.store,
        post_reboot_stabilization_seconds=0,
    )
    step = harness.context.step.model_copy(
        update={"parameters": {"replacement_strategy": "HEALTHY_WARM_SPARE_ONLY"}}
    )
    context = WorkflowStepContext(
        workflow=harness.workflow.model_copy(update={"official_steps": [step]}),
        incident=harness.incident,
        step=step,
        step_index=0,
        request=harness.context.request,
        idempotency_key=harness.context.idempotency_key,
    )

    outcome = adapter.execute(context)

    assert outcome.status is WorkflowStepStatus.SUCCEEDED, outcome.error
    details = outcome.details or {}
    assert details["action"] == "SPARE_FAILOVER"
    assert details["activated_spare_nodes"] == [NODE_B]
    assert details["node_rebindings"] == {NODE_A: NODE_B}
    assert details["provider_mutation_submitted"] is False
    assert details["confirmation_source"] == "healthy-running-warm-spare"
    assert harness.ec2.reboots == [], "a handover never touches EC2"
    assert spares.calls[0]["local_only"] is True, "spares are read from Kubernetes"
    assert [item.step.operation for item in node_actions.contexts] == [
        WorkflowOperation.TRIGGER_HEALTH_SNAPSHOT
    ]
    assert node_actions.contexts[0].step.node_ids == [NODE_B]

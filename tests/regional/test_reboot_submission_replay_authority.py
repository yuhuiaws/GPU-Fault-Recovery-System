from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timezone

import pytest

from gpu_fault.adapters import HyperPodLifecycleStepAdapter
from gpu_fault.adapters.common import (
    ANNOTATION_FENCING,
    ANNOTATION_INCIDENT,
    QUARANTINE_TAINT,
)
from gpu_fault.adapters.kubernetes.stop_boot_transition import REBOOT_AUTHORIZATION_KEY
from gpu_fault.adapters.kubernetes.stop_ownership import (
    STOP_RECEIPT_KEY,
    stop_ownership_scope,
)
from gpu_fault.fleet import (
    CURRENT_AGENT_PROTOCOL_VERSION,
    AgentHeartbeat,
    FleetRegistry,
    SignedAgentHeartbeat,
    sign_agent_heartbeat,
)
from gpu_fault.hyperpod import HyperPodAdapterConfig, HyperPodLifecycleAdapter
from gpu_fault.models import (
    BlockedKind,
    RecoveryAction,
    WorkflowOperation,
    WorkflowStatus,
    WorkflowStepExecution,
    WorkflowStepStatus,
    workflow_is_open,
)
from gpu_fault.orchestration.escalation import HardwareEscalationService
from gpu_fault.store import InMemoryStore
from tests._builders import active_workflow_executor, workflow_step
from tests.hyperpod._cov95_provider_extra_safety import (
    provider_extra_isolation as provider_extra_isolation,
)
from tests.hyperpod.test_hyperpod import FakeHyperPodClient
from tests.regional._late_ownership_runtime import WORKLOAD, stopped_runtime

REBOOT = WorkflowOperation.RESTART_NODE
KEY = "reboot-replay-unit-" + "x" * 32


class Provider(FakeHyperPodClient):
    def describe_cluster_node(self, **kwargs):
        result = super().describe_cluster_node(**kwargs)
        if kwargs["NodeLogicalId"] == "worker-group-1":
            result["NodeDetails"]["KubernetesConfig"]["CurrentLabels"][
                "kubernetes.io/hostname"
            ] = "node-a"
        return result


@pytest.fixture
def reboot_runtime():
    state, kubernetes, validator, context = stopped_runtime()
    node = state.nodes["node-a"]
    node["spec"] = {
        "unschedulable": True,
        "taints": [
            {
                "key": QUARANTINE_TAINT,
                "value": context.incident.incident_id,
                "effect": "NoSchedule",
            }
        ],
    }
    node["metadata"]["annotations"] = {
        ANNOTATION_INCIDENT: context.incident.incident_id,
        ANNOTATION_FENCING: str(context.workflow.fencing_token),
    }
    store = InMemoryStore()
    fleet = FleetRegistry(store, KEY)
    heartbeat = AgentHeartbeat(
        cluster_id=context.incident.cluster_id,
        node_id="node-a",
        endpoint="http://node-a:9099",
        agent_protocol_version=CURRENT_AGENT_PROTOCOL_VERSION,
        agent_version="0.10.0",
        artifact_sha256="a" * 64,
        policy_version="610",
        runtime_profile_version="local-profile",
        config_digest="c" * 64,
        allowed_operations=[
            WorkflowOperation.RESET_GPU,
            WorkflowOperation.VERIFY_NO_GPU_CLIENTS,
        ],
        boot_id="node-a-boot",
        node_instance_id="i-00000000000000001",
        agent_incarnation_id="incarnation-before",
        observed_at=datetime.now(timezone.utc),
    )
    fleet.register(
        SignedAgentHeartbeat(
            heartbeat=heartbeat, signature=sign_agent_heartbeat(heartbeat, KEY)
        )
    )
    client = Provider(node_recovery="None")
    config = HyperPodAdapterConfig(
        cluster_name="hp-cluster", region_name="us-east-1", execution_enabled=True
    )

    def new_adapter():
        return HyperPodLifecycleStepAdapter(
            HyperPodLifecycleAdapter(config, client=client, store=store),
            registry=fleet,
            kubernetes_adapter=kubernetes,
            post_reboot_stabilization_seconds=0,
        )

    adapter = new_adapter()
    reboot = workflow_step(REBOOT, adapter.owner, workload_ids=[WORKLOAD])
    context = replace(
        context,
        step=reboot,
        workflow=context.workflow.model_copy(
            update={"official_steps": [context.workflow.official_steps[0], reboot]}
        ),
        request=context.request.model_copy(
            update={"confirm_cluster_name": config.cluster_name}
        ),
        idempotency_key=f"{context.workflow.request_id}/1/{REBOOT.value}",
    )
    return state, validator, context, store, client, fleet, heartbeat, new_adapter


@pytest.mark.parametrize("original_bound", [False, True], ids=["legacy", "lost-reply"])
def test_durable_reboot_replay_cannot_recreate_a_missing_submission_binding(
    reboot_runtime, original_bound
):
    state, validator, context, store, client, _fleet, _heartbeat, new_adapter = (
        reboot_runtime
    )
    receipt = deepcopy(context.workflow.step_executions[0].details[STOP_RECEIPT_KEY])
    if original_bound:
        with stop_ownership_scope(validator):
            first = new_adapter().execute(context)
    else:
        first = new_adapter().execute(context)
    assert first.status is WorkflowStepStatus.WAITING, first
    assert (REBOOT_AUTHORIZATION_KEY in first.details) is original_bound, first
    key = first.details["submission_idempotency_key"]
    original_submission = store.get_hyperpod_submission("hp-cluster", key)
    assert original_submission.state == "SUBMITTED", original_submission
    assert state.nodes["node-a"]["status"]["nodeInfo"]["bootID"] == "node-a-boot", (
        "the retry must occur before the accepted reboot changes the boot"
    )

    # The provider receipt survived, but the remote WAITING reply did not.
    store.save_incident_and_workflow(context.incident, context.workflow)
    executor = active_workflow_executor(store, [new_adapter()], [REBOOT])
    with stop_ownership_scope(validator):
        result = executor.execute(context.workflow.request_id, context.request)
    saved = store.get_workflow(context.workflow.request_id)
    assert result.status is WorkflowStatus.BLOCKED, result
    assert saved.blocked_kind is BlockedKind.NEEDS_OPERATOR, saved
    assert workflow_is_open(saved.status, saved.blocked_kind), saved
    failure = next(item for item in saved.step_executions if item.operation is REBOOT)
    assert failure.details["outcome_unknown"] is True, failure
    assert failure.details["manual_confirmation_required"] is True, failure
    assert failure.details["provider_submission_duplicate"] is True, failure
    assert failure.adapter_operation_id == first.adapter_operation_id, failure
    assert REBOOT_AUTHORIZATION_KEY not in failure.details, failure
    assert failure.details.get("node_action_not_started") is not True, failure
    classification = HardwareEscalationService.classify(saved)
    assert classification is not None, saved
    assert classification[1] is RecoveryAction.ESCALATE_OPERATOR, classification
    assert len(client.reboot_requests) == 1, client.reboot_requests
    assert store.get_hyperpod_submission("hp-cluster", key) == original_submission, (
        "a replay must not rewrite the original provider receipt"
    )
    assert saved.step_executions[0].details[STOP_RECEIPT_KEY] == receipt, saved


def test_reclaimed_confirmation_preserves_the_original_binding(reboot_runtime):
    state, validator, context, _store, client, fleet, heartbeat, new_adapter = (
        reboot_runtime
    )
    with stop_ownership_scope(validator):
        first = new_adapter().execute(context)
    binding = deepcopy(first.details[REBOOT_AUTHORIZATION_KEY])
    execution = WorkflowStepExecution(
        step_index=1,
        operation=REBOOT,
        status=first.status,
        phase="official",
        adapter_operation_id=first.adapter_operation_id,
        details=first.details,
    )
    context = replace(
        context,
        workflow=context.workflow.model_copy(
            update={"step_executions": [*context.workflow.step_executions, execution]}
        ),
    )
    heartbeat = heartbeat.model_copy(
        update={
            "boot_id": "node-a-rebooted",
            "agent_incarnation_id": "incarnation-after",
            "observed_at": datetime.now(timezone.utc),
        }
    )
    state.nodes["node-a"]["status"]["nodeInfo"]["bootID"] = heartbeat.boot_id
    fleet.register(
        SignedAgentHeartbeat(
            heartbeat=heartbeat, signature=sign_agent_heartbeat(heartbeat, KEY)
        )
    )
    assert fleet.readiness(context.incident.cluster_id, ["node-a"]).ready, (
        "the fixture must meet the real reboot-confirmation readiness gate"
    )
    with stop_ownership_scope(validator):
        confirmed = new_adapter().execute(context)
    assert confirmed.status is WorkflowStepStatus.SUCCEEDED, confirmed
    assert confirmed.details[REBOOT_AUTHORIZATION_KEY] == binding, confirmed
    assert confirmed.adapter_operation_id == first.adapter_operation_id, confirmed
    assert len(client.reboot_requests) == 1, client.reboot_requests

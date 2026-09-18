from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone

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
    node_submission_ownership_guard,
    require_mutation_ownership,
    stop_ownership_scope,
)
from gpu_fault.execution import WorkflowStepContext, WorkflowStepOutcome
from gpu_fault.execution.branch_escalation import BranchEscalator
from gpu_fault.fleet import (
    CURRENT_AGENT_PROTOCOL_VERSION,
    AgentHeartbeat,
    FleetRegistry,
    SignedAgentHeartbeat,
    sign_agent_heartbeat,
)
from gpu_fault.hyperpod import HyperPodNode
from gpu_fault.models import (
    WorkflowOperation,
    WorkflowStatus,
    WorkflowStepExecution,
    WorkflowStepStatus,
)
from gpu_fault.orchestration.arbitration import RecoveryArbiter
from gpu_fault.orchestration.dag_branching import DagBrancher
from gpu_fault.store import InMemoryStore
from tests._builders import active_workflow_executor, workflow_step
from tests.execution._support import FakeHyperPodLifecycle
from tests.regional._late_ownership_runtime import WORKLOAD, stopped_runtime

REBOOT = WorkflowOperation.RESTART_NODE
REPLACE = WorkflowOperation.REPLACE_NODE
RESET = WorkflowOperation.RESET_GPU
VALIDATE = WorkflowOperation.VALIDATE_GPU
KEY = "local-reboot-test-" + "x" * 32


class Provider(FakeHyperPodLifecycle):
    def resolve_nodes(self, identifiers):
        assert identifiers == ["node-a"]
        return [
            HyperPodNode(
                node_logical_id="node-a",
                instance_id="instance-a",
                instance_group_name="workers",
                instance_type="ml.p5.48xlarge",
                status="Running",
                kubernetes_labels={"kubernetes.io/hostname": "node-a"},
            )
        ]


def confirmed_reboot():
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
    fleet = FleetRegistry(InMemoryStore(), KEY)
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
        allowed_operations=[RESET, WorkflowOperation.VERIFY_NO_GPU_CLIENTS],
        boot_id="node-a-boot",
        node_instance_id="instance-a",
        agent_incarnation_id="incarnation-before",
        observed_at=datetime.now(timezone.utc),
    )
    fleet.register(
        SignedAgentHeartbeat(
            heartbeat=heartbeat, signature=sign_agent_heartbeat(heartbeat, KEY)
        )
    )
    provider = Provider()
    lifecycle = HyperPodLifecycleStepAdapter(
        provider,
        registry=fleet,
        kubernetes_adapter=kubernetes,
        post_reboot_stabilization_seconds=0,
    )
    steps = [
        context.workflow.official_steps[0].model_copy(update={"branch_id": "shared"}),
        workflow_step(
            REBOOT,
            lifecycle.owner,
            workload_ids=[WORKLOAD],
            branch_id="branch:node-a",
            depends_on_step_indexes=[0],
        ),
        workflow_step(
            VALIDATE,
            workload_ids=[WORKLOAD],
            branch_id="branch:node-a",
            depends_on_step_indexes=[1],
        ),
        workflow_step(
            RESET,
            node_ids=["node-b"],
            workload_ids=[WORKLOAD],
            branch_id="branch:node-b",
            depends_on_step_indexes=[0],
        ),
    ]
    context = replace(
        context,
        workflow=context.workflow.model_copy(
            update={"dag_enabled": True, "official_steps": steps}
        ),
        step=steps[1],
        request=context.request.model_copy(
            update={"confirm_cluster_name": "hp-cluster"}
        ),
        idempotency_key=f"{context.workflow.request_id}/1/RESTART_NODE",
    )
    with stop_ownership_scope(validator):
        pending = lifecycle.execute(context)
    assert pending.status is WorkflowStepStatus.WAITING, pending
    authorization = pending.details[REBOOT_AUTHORIZATION_KEY]
    assert authorization["nodes"] == {
        "node-a": {"uid": "node-a-uid", "boot_id": "node-a-boot"}
    }
    assert authorization["workflow_id"] == context.workflow.request_id
    assert authorization["fencing_token"] == context.workflow.fencing_token
    execution = WorkflowStepExecution(
        step_index=1,
        operation=REBOOT,
        phase="official",
        status=pending.status,
        adapter_operation_id=pending.adapter_operation_id,
        details=pending.details,
        # As on the CPU, the attempt is stamped after the remote submit result.
        started_at=datetime.now(timezone.utc),
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
    node["status"]["nodeInfo"]["bootID"] = heartbeat.boot_id
    fleet.register(
        SignedAgentHeartbeat(
            heartbeat=heartbeat, signature=sign_agent_heartbeat(heartbeat, KEY)
        )
    )
    with stop_ownership_scope(validator):
        confirmed = lifecycle.execute(context)
    assert confirmed.status is WorkflowStepStatus.SUCCEEDED, confirmed
    assert confirmed.details[REBOOT_AUTHORIZATION_KEY] == authorization, (
        "confirmation must retain the original pre-submit UID/boot binding"
    )
    execution = execution.model_copy(
        update={
            "status": confirmed.status,
            "details": confirmed.details,
            "updated_at": datetime.now(timezone.utc),
        }
    )
    context = replace(
        context,
        workflow=context.workflow.model_copy(
            update={
                "step_executions": [context.workflow.step_executions[0], execution],
                "completed_step_indexes": [0, 1],
                "completed_operations": [WorkflowOperation.STOP_WORKLOADS, REBOOT],
            }
        ),
        step=steps[2],
        step_index=2,
        idempotency_key=f"{context.workflow.request_id}/2/VALIDATE_GPU",
    )
    assert provider.calls == 1, "confirmation must poll, not reboot a second time"
    return state, validator, context


def test_confirmed_reboot_can_escalate_after_validation_and_does_not_block_sibling():
    state, validator, context = confirmed_reboot()
    store = InMemoryStore()
    store.save_incident_and_workflow(context.incident, context.workflow)
    calls = []

    class Actions:
        owner = "owner-a"

        def supports(self, step):
            return step.execution_owner == self.owner

        def execute(self, current: WorkflowStepContext):
            require_mutation_ownership(current)
            calls.append((current.step.operation, current.step.node_ids))
            if current.step.operation is VALIDATE and current.step_index == 2:
                return WorkflowStepOutcome.failed("GPU remains unhealthy")
            return WorkflowStepOutcome.succeeded()

    def compile_steps(_workflow, operations, node, gpus):
        return [
            workflow_step(
                operation,
                node_ids=[node],
                gpu_uuids=list(gpus),
                workload_ids=[WORKLOAD],
            )
            for operation in operations
        ]

    executor = active_workflow_executor(store, [Actions()], WorkflowOperation)
    executor.branch_escalator = BranchEscalator(
        DagBrancher(RecoveryArbiter()), compile_steps
    )
    with stop_ownership_scope(validator):
        result = executor.execute(context.workflow.request_id, context.request)
    assert result.status is WorkflowStatus.SUCCEEDED, result
    assert (RESET, ["node-b"]) in calls
    assert (REPLACE, ["node-a"]) in calls
    saved = store.get_workflow(context.workflow.request_id)
    assert saved.branch_escalation_counts == {"node-a": 1}
    receipt = saved.step_executions[0].details[STOP_RECEIPT_KEY]
    assert receipt["nodes"][0]["boot_id"] == "node-a-boot", (
        "the original STOP receipt must remain immutable"
    )
    assert state.nodes["node-a"]["status"]["nodeInfo"]["bootID"] == "node-a-rebooted"


@pytest.mark.parametrize(
    "defect",
    [
        "node-uid",
        "sibling-boot",
        "workload-uid",
        "workload-annotations",
        "late-pod",
        "unconfirmed",
        "not-completed",
        "inherited",
        "superseded",
        "safety-phase",
        "legacy-phase",
        "wrong-operation",
        "explicit-confirmation",
        "unknown-outcome",
        "foreign-workflow",
        "wrong-fence",
        "wrong-incident",
        "missing-baseline",
        "wrong-old-boot",
        "wrong-new-boot",
        "same-incarnation",
        "missing-incarnation",
        "missing-agent",
        "duplicate-agent",
        "missing-provider",
        "provider-not-running",
        "wrong-submitted-node",
        "missing-isolation",
        "wrong-isolation-node",
        "no-operation-id",
        "before-stop",
        "future-confirmation",
        "request-fence",
        "authorization-missing",
        "authorization-node-uid",
        "authorization-old-boot",
        "authorization-stop-digest",
        "authorization-workflow",
        "authorization-fence",
        "authorization-phase",
        "authorization-step",
        "authorization-epoch",
        "authorization-owner",
        "authorization-before-start",
        "authorization-target-extra",
    ],
)
def test_a_boot_change_without_complete_bound_proof_remains_fail_closed(defect):
    state, validator, context = confirmed_reboot()
    execution = context.workflow.step_executions[1]
    details = execution.details
    updates = {}
    if defect == "node-uid":
        state.nodes["node-a"]["metadata"]["uid"] = "new-uid"
    elif defect == "sibling-boot":
        state.nodes["node-b"]["status"]["nodeInfo"]["bootID"] = "unrelated-reboot"
    elif defect == "workload-uid":
        state.job["metadata"]["uid"] = "new-workload"
    elif defect == "workload-annotations":
        state.job["metadata"]["annotations"] = {}
    elif defect == "late-pod":
        state.pods.append(state.pod("node-b", "new-pod"))
    elif defect == "unconfirmed":
        execution.status = WorkflowStepStatus.WAITING
    elif defect == "not-completed":
        updates["completed_step_indexes"] = [0]
    elif defect == "inherited":
        updates["inherited_step_indexes"] = [1]
    elif defect == "superseded":
        updates["superseded_step_indexes"] = [1]
    elif defect == "safety-phase":
        execution.phase = "safety"
    elif defect == "legacy-phase":
        execution.phase = None
    elif defect == "wrong-operation":
        execution.operation = REPLACE
    elif defect == "explicit-confirmation":
        details["confirmation_source"] = "explicit-operation-id"
    elif defect == "unknown-outcome":
        details["outcome_unknown"] = True
    elif defect == "foreign-workflow":
        details["submission_idempotency_key"] = "other/RESTART_NODE/1"
    elif defect == "wrong-fence":
        details["observed_isolation"]["node-a"]["fencing_token"] = "2"
    elif defect == "wrong-incident":
        details["observed_isolation"]["node-a"]["incident"] = "other"
    elif defect == "missing-baseline":
        details["agent_baselines"] = {}
    elif defect == "wrong-old-boot":
        details["agent_baselines"]["node-a"]["boot_id"] = "unrelated-boot"
    elif defect == "wrong-new-boot":
        details["agent_observations"][0]["boot_id"] = "unrelated-boot"
    elif defect == "same-incarnation":
        details["agent_observations"][0]["agent_incarnation_id"] = "incarnation-before"
    elif defect == "missing-incarnation":
        details["agent_observations"][0]["agent_incarnation_id"] = None
    elif defect == "missing-agent":
        details["agent_observations"] = []
    elif defect == "duplicate-agent":
        details["agent_observations"] *= 2
    elif defect == "missing-provider":
        details["provider_observations"] = []
    elif defect == "provider-not-running":
        details["provider_observations"][0]["status"] = "Pending"
    elif defect == "wrong-submitted-node":
        details["submitted_nodes"] = ["node-b"]
    elif defect == "missing-isolation":
        details["observed_isolation"] = {}
    elif defect == "wrong-isolation-node":
        details["observed_isolation"]["node-a"]["kubernetes_node"] = "node-b"
    elif defect == "no-operation-id":
        execution.adapter_operation_id = None
    elif defect == "before-stop":
        execution.started_at -= timedelta(minutes=1)
    elif defect == "future-confirmation":
        execution.updated_at += timedelta(minutes=1)
    elif defect == "request-fence":
        context = replace(
            context,
            request=context.request.model_copy(update={"expected_fencing_token": 2}),
        )
    elif defect == "authorization-missing":
        details.pop(REBOOT_AUTHORIZATION_KEY)
    elif defect == "authorization-node-uid":
        details[REBOOT_AUTHORIZATION_KEY]["nodes"]["node-a"]["uid"] = "recreated"
    elif defect == "authorization-old-boot":
        details[REBOOT_AUTHORIZATION_KEY]["nodes"]["node-a"]["boot_id"] = "unproven"
    elif defect == "authorization-stop-digest":
        details[REBOOT_AUTHORIZATION_KEY]["stop_receipt_sha256"] = "0" * 64
    elif defect == "authorization-workflow":
        details[REBOOT_AUTHORIZATION_KEY]["workflow_id"] = "other"
    elif defect == "authorization-fence":
        details[REBOOT_AUTHORIZATION_KEY]["fencing_token"] = 2
    elif defect == "authorization-phase":
        details[REBOOT_AUTHORIZATION_KEY]["phase"] = "safety"
    elif defect == "authorization-step":
        details[REBOOT_AUTHORIZATION_KEY]["step_index"] = 0
    elif defect == "authorization-epoch":
        details[REBOOT_AUTHORIZATION_KEY]["execution_epoch"] = 0
    elif defect == "authorization-owner":
        details[REBOOT_AUTHORIZATION_KEY]["execution_owner"] = "other"
    elif defect == "authorization-before-start":
        details[REBOOT_AUTHORIZATION_KEY]["checked_at"] = (
            execution.started_at - timedelta(minutes=1)
        ).isoformat()
    elif defect == "authorization-target-extra":
        details[REBOOT_AUTHORIZATION_KEY]["nodes"]["node-b"] = {
            "uid": "node-b-uid",
            "boot_id": "node-b-boot",
        }
    context = replace(
        context,
        workflow=context.workflow.model_copy(update=updates),
        step=context.step.model_copy(update={"operation": REPLACE}),
    )
    with stop_ownership_scope(validator):
        outcome = node_submission_ownership_guard(context)
    assert outcome is not None, defect
    assert outcome.status is WorkflowStepStatus.FAILED
    assert outcome.details["safety_rejection"] is True
    assert outcome.details["node_action_not_started"] is True
    assert outcome.details["manual_confirmation_required"] is True

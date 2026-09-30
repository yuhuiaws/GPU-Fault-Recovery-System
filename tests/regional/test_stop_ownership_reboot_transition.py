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
from gpu_fault.execution.node_action_uncertainty import pending_remote_action_details
from gpu_fault.fleet import (
    CURRENT_AGENT_PROTOCOL_VERSION,
    AgentHeartbeat,
    FleetRegistry,
    SignedAgentHeartbeat,
    sign_agent_heartbeat,
)
from gpu_fault.hyperpod import HyperPodAction, HyperPodNode, HyperPodSubmissionResult
from gpu_fault.models import (
    WorkflowOperation,
    WorkflowStatus,
    WorkflowStepExecution,
    WorkflowStepStatus,
)
from gpu_fault.orchestration.arbitration import RecoveryArbiter
from gpu_fault.orchestration.dag_branching import DagBrancher
from gpu_fault.remote_command_models import RemoteCommandStatus
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


# --------------------------------------------------------------------------- #
# An in-flight, product-submitted sibling reboot is not ownership drift
# --------------------------------------------------------------------------- #
class TwoNodeProvider(FakeHyperPodLifecycle):
    """Resolves and reboots whichever single node a step names."""

    def resolve_nodes(self, identifiers):
        return [
            HyperPodNode(
                node_logical_id=f"logical-{node}",
                instance_id=f"instance-{node}",
                instance_group_name="workers",
                instance_type="ml.p5.48xlarge",
                status="Running",
                kubernetes_labels={"kubernetes.io/hostname": node},
            )
            for node in identifiers
        ]

    def execute_step(self, step, **kwargs):
        self.calls += 1
        assert kwargs["confirm_cluster_name"] == "hp-cluster"
        assert kwargs["isolation_verified_nodes"] == list(step.node_ids)
        return HyperPodSubmissionResult(
            operation_id=f"hyperpod-operation-{self.calls}",
            idempotency_key=kwargs["idempotency_key"],
            action=HyperPodAction.REBOOT,
            cluster_name="hp-cluster",
            requested_node_logical_ids=[f"logical-{n}" for n in step.node_ids],
            successful_node_logical_ids=[f"logical-{n}" for n in step.node_ids],
        )


def _isolate(state, node, context):
    record = state.nodes[node]
    record["spec"] = {
        "unschedulable": True,
        "taints": [
            {
                "key": QUARANTINE_TAINT,
                "value": context.incident.incident_id,
                "effect": "NoSchedule",
            }
        ],
    }
    record["metadata"]["annotations"] = {
        ANNOTATION_INCIDENT: context.incident.incident_id,
        ANNOTATION_FENCING: str(context.workflow.fencing_token),
    }


def in_flight_sibling_reboot():
    """DESTR-014 attempt 1 (live): node-a (the sibling, ``branch:initial``) has
    its RESTART_NODE submitted and still WAITING when it reboots and comes back
    with a new boot id but no new Agent; node-b (the fault node) escalates to
    its own RESTART_NODE and the pre-submit receipt recheck runs over both
    nodes of the shared STOP receipt."""

    state, kubernetes, validator, context = stopped_runtime()
    fleet = FleetRegistry(InMemoryStore(), KEY)
    for node in ("node-a", "node-b"):
        _isolate(state, node, context)
        heartbeat = AgentHeartbeat(
            cluster_id=context.incident.cluster_id,
            node_id=node,
            endpoint=f"http://{node}:9099",
            agent_protocol_version=CURRENT_AGENT_PROTOCOL_VERSION,
            agent_version="0.10.0",
            artifact_sha256="a" * 64,
            policy_version="610",
            runtime_profile_version="local-profile",
            config_digest="c" * 64,
            allowed_operations=[RESET, WorkflowOperation.VERIFY_NO_GPU_CLIENTS],
            boot_id=f"{node}-boot",
            node_instance_id=f"instance-{node}",
            agent_incarnation_id=f"{node}-incarnation-before",
            observed_at=datetime.now(timezone.utc),
        )
        fleet.register(
            SignedAgentHeartbeat(
                heartbeat=heartbeat, signature=sign_agent_heartbeat(heartbeat, KEY)
            )
        )
    provider = TwoNodeProvider()
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
            branch_id="branch:initial",
            branch_node_ids=["node-a"],
            depends_on_step_indexes=[0],
        ),
        workflow_step(
            VALIDATE,
            workload_ids=[WORKLOAD],
            branch_id="branch:initial",
            depends_on_step_indexes=[1],
        ),
        workflow_step(
            RESET,
            node_ids=["node-b"],
            workload_ids=[WORKLOAD],
            branch_id="branch:node-b",
            depends_on_step_indexes=[0],
        ),
        workflow_step(
            REBOOT,
            lifecycle.owner,
            node_ids=["node-b"],
            workload_ids=[WORKLOAD],
            branch_id="branch:node-b",
            branch_node_ids=["node-b"],
            depends_on_step_indexes=[3],
        ),
    ]
    workflow = context.workflow.model_copy(
        update={"dag_enabled": True, "official_steps": steps}
    )
    sibling = replace(
        context,
        workflow=workflow,
        step=steps[1],
        step_index=1,
        request=context.request.model_copy(
            update={"confirm_cluster_name": "hp-cluster"}
        ),
        idempotency_key=f"{workflow.request_id}/1/RESTART_NODE",
    )
    with stop_ownership_scope(validator):
        pending = lifecycle.execute(sibling)
    assert pending.status is WorkflowStepStatus.WAITING, pending
    assert pending.details["submitted_nodes"] == ["logical-node-a"]
    assert pending.details[REBOOT_AUTHORIZATION_KEY]["nodes"] == {
        "node-a": {"uid": "node-a-uid", "boot_id": "node-a-boot"}
    }
    # The CPU records the executor's WAITING result the way the remote-command
    # adapter does: the posted details plus the command pointers, with the
    # in-flight node action marked unknown until it is confirmed.
    recorded = WorkflowStepExecution(
        step_index=1,
        operation=REBOOT,
        phase="official",
        status=WorkflowStepStatus.WAITING,
        adapter_operation_id="remote/command-sibling",
        details={
            **pending_remote_action_details(
                REBOOT, RemoteCommandStatus.WAITING, pending.details
            ),
            "remote_command_id": "command-sibling",
            "remote_cluster_id": context.incident.cluster_id,
            "remote_status": RemoteCommandStatus.WAITING.value,
            "mutation_submitted_by_control_plane": False,
        },
        started_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )
    assert recorded.details["outcome_unknown"] is True
    workflow = workflow.model_copy(
        update={"step_executions": [*workflow.step_executions, recorded]}
    )
    # node-a reboots (the product's own submission) and re-registers with a
    # new boot id; its Node Agent does not come back, so nothing confirms it.
    state.nodes["node-a"]["status"]["nodeInfo"]["bootID"] = "node-a-rebooted"
    fault = replace(
        sibling,
        workflow=workflow,
        step=steps[4],
        step_index=4,
        idempotency_key=f"{workflow.request_id}/4/RESTART_NODE",
    )
    return state, validator, lifecycle, provider, fault


def test_an_in_flight_sibling_reboot_the_product_submitted_is_not_drift():
    state, validator, lifecycle, provider, fault = in_flight_sibling_reboot()
    with stop_ownership_scope(validator):
        assert node_submission_ownership_guard(fault) is None
        outcome = lifecycle.execute(fault)
    assert outcome.status is WorkflowStepStatus.WAITING, outcome
    assert provider.calls == 2, "the fault node's reboot must be submitted"
    assert outcome.details["submitted_nodes"] == ["logical-node-b"]
    assert outcome.details[REBOOT_AUTHORIZATION_KEY]["nodes"] == {
        "node-b": {"uid": "node-b-uid", "boot_id": "node-b-boot"}
    }
    receipt = fault.workflow.step_executions[0].details[STOP_RECEIPT_KEY]
    assert {node["name"]: node["boot_id"] for node in receipt["nodes"]} == {
        "node-a": "node-a-boot",
        "node-b": "node-b-boot",
    }, "the STOP receipt stays immutable; the acceptance is a read of the chain"


@pytest.mark.parametrize(
    "defect",
    [
        "no-submission-record",
        "different-receipt-digest",
        "terminal-failed",
        "terminal-succeeded-unconfirmed",
        "own-node-in-flight",
        "not-a-reboot",
        "duplicate-submission",
        "authorization-missing",
        "authorization-node-uid",
        "authorization-old-boot",
        "authorization-foreign-workflow",
        "submitted-nodes-empty",
        "not-submitted",
        "baseline-mismatch",
        "isolation-missing",
        "before-stop",
        "superseded",
        "sibling-uid-changed",
    ],
)
def test_a_boot_change_without_an_in_flight_bound_submission_is_drift(defect):
    state, validator, lifecycle, provider, fault = in_flight_sibling_reboot()
    workflow = fault.workflow
    execution = workflow.step_executions[1]
    details = execution.details
    updates = {}
    if defect == "no-submission-record":
        updates["step_executions"] = [workflow.step_executions[0]]
    elif defect == "different-receipt-digest":
        details[REBOOT_AUTHORIZATION_KEY]["stop_receipt_sha256"] = "0" * 64
    elif defect == "terminal-failed":
        # The managed-recovery cap: FAILED with the unknown-outcome flags.
        execution.status = WorkflowStepStatus.FAILED
        execution.error = "RESTART_NODE waited 600 s without confirmation"
        details["step_waiting_timeout_seconds"] = 600
    elif defect == "terminal-succeeded-unconfirmed":
        execution.status = WorkflowStepStatus.SUCCEEDED
        details.pop("outcome_unknown")
        details.pop("manual_confirmation_required")
        updates["completed_step_indexes"] = [0, 1]
    elif defect == "own-node-in-flight":
        # node-a's own next action while node-a's reboot is still in flight.
        fault = replace(
            fault,
            step=workflow.official_steps[4].model_copy(
                update={"node_ids": ["node-a"], "branch_node_ids": ["node-a"]}
            ),
        )
    elif defect == "not-a-reboot":
        details["action"] = HyperPodAction.REPLACE.value
    elif defect == "duplicate-submission":
        details["provider_submission_duplicate"] = True
    elif defect == "authorization-missing":
        details.pop(REBOOT_AUTHORIZATION_KEY)
    elif defect == "authorization-node-uid":
        details[REBOOT_AUTHORIZATION_KEY]["nodes"]["node-a"]["uid"] = "recreated"
    elif defect == "authorization-old-boot":
        details[REBOOT_AUTHORIZATION_KEY]["nodes"]["node-a"]["boot_id"] = "unproven"
    elif defect == "authorization-foreign-workflow":
        details[REBOOT_AUTHORIZATION_KEY]["workflow_id"] = "other"
    elif defect == "submitted-nodes-empty":
        details["submitted_nodes"] = []
    elif defect == "not-submitted":
        details.pop("requires_external_confirmation")
        details["node_action_not_started"] = True
    elif defect == "baseline-mismatch":
        details["agent_baselines"] = {
            "node-a": {"boot_id": "other-boot", "agent_incarnation_id": "x"}
        }
    elif defect == "isolation-missing":
        details["observed_isolation"] = {}
    elif defect == "before-stop":
        execution.started_at -= timedelta(minutes=1)
    elif defect == "superseded":
        updates["superseded_step_indexes"] = [1]
    elif defect == "sibling-uid-changed":
        state.nodes["node-a"]["metadata"]["uid"] = "node-a-recreated"
    fault = replace(fault, workflow=fault.workflow.model_copy(update=updates))
    with stop_ownership_scope(validator):
        outcome = node_submission_ownership_guard(fault)
    assert outcome is not None, defect
    assert outcome.status is WorkflowStepStatus.FAILED, defect
    assert outcome.details["reason"] == "STOP_OWNERSHIP_DRIFT", (defect, outcome)
    assert outcome.details["safety_rejection"] is True
    assert outcome.details["node_action_not_started"] is True
    assert provider.calls == 1, "nothing may be submitted after a refusal"


def test_an_unchanged_sibling_with_an_in_flight_reboot_stays_authorized():
    state, validator, lifecycle, provider, fault = in_flight_sibling_reboot()
    # The sibling has not come back yet: same boot id as the receipt.
    state.nodes["node-a"]["status"]["nodeInfo"]["bootID"] = "node-a-boot"
    with stop_ownership_scope(validator):
        assert node_submission_ownership_guard(fault) is None


def test_the_confirmed_chain_still_authorizes_after_the_in_flight_rule():
    # The previously accepted shape is unchanged: a SUCCEEDED, externally
    # confirmed reboot with the full bound proof.
    state, validator, context = confirmed_reboot()
    context = replace(
        context, step=context.step.model_copy(update={"operation": REPLACE})
    )
    with stop_ownership_scope(validator):
        assert node_submission_ownership_guard(context) is None

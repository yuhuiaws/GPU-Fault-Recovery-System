from __future__ import annotations

from copy import deepcopy
from threading import RLock

import pytest

from gpu_fault.adapters import KubernetesWorkflowAdapter
from gpu_fault.adapters.common import QUARANTINE_TAINT
from gpu_fault.app import default_simulated_profile
from gpu_fault.cluster_executor import ClusterActionExecutor
from gpu_fault.execution import WorkflowStepOutcome
from gpu_fault.models import (
    BlockedKind,
    CapabilityMode,
    CapabilityName,
    IncidentState,
    WorkflowOperation,
    WorkflowStatus,
)
from gpu_fault.orchestration.families.reset import ResetOperationService
from gpu_fault.policy import (
    DistributedXidBatch,
    GpuFaultPolicyEngine,
    load_sxid_policy,
    load_xid_policy,
)
from gpu_fault.regional import RegionalRemoteWorkflowAdapter
from tests._builders import active_workflow_executor, execute_workflow
from tests.hyperpod._cov95_provider_extra_safety import (
    provider_extra_isolation as provider_extra_isolation,
)
from tests.orchestration._cov95_runtime_builder import builder, xid
from tests.regional._late_ownership_runtime import WORKLOAD, KubernetesState
from tests.regional._regional_support import TOKEN_A, registration
from tests.regional.test_remote_step_batching_e2e import StoreClient

OWNER = "local-containment"
EVIDENCE_OWNER = "local-evidence"
SAFETY = [
    WorkflowOperation.FREEZE_EVIDENCE,
    WorkflowOperation.MARK_UNSCHEDULABLE,
    WorkflowOperation.QUARANTINE,
]


def plan(node_count, *, checkpoint=False, containment_owned=True):
    compiler = builder()
    profile = default_simulated_profile()
    compiler.store.save_profile(
        profile.model_copy(
            update={
                "capabilities": [
                    item.model_copy(
                        update={
                            "owner": EVIDENCE_OWNER
                            if item.capability is CapabilityName.EVIDENCE_CAPTURE
                            else OWNER,
                            "mode": CapabilityMode.OBSERVE
                            if not containment_owned
                            and item.capability is CapabilityName.SCHEDULER_DRAIN
                            else item.mode,
                        }
                    )
                    for item in profile.capabilities
                ]
            }
        )
    )
    nodes = [f"node-{index:03}" for index in range(node_count)]
    batch = DistributedXidBatch(
        batch_id=f"distributed-limit-{node_count}",
        job_id="job",
        attempt_id="attempt",
        restart_budget=1,
        affected_workload_ids=[WORKLOAD],
        checkpoint_manifest_ref="local-checkpoint" if checkpoint else None,
        allocation=[
            {"node_id": node, "rank": index, "gpu_uuids": [f"GPU-{node}"]}
            for index, node in enumerate([*nodes, "healthy-participant"])
        ],
        events=[
            xid(
                event_id=f"fault-{node}",
                node_id=node,
                gpu_uuid=f"GPU-{node}",
                xid=95,
                product="H200",
                runtime_profile_version=profile.profile_version,
                job_id="job",
                attempt_id="attempt",
                workload_state="ACTIVE",
                affected_workload_ids=[WORKLOAD],
            )
            for node in nodes
        ],
    )
    policy = GpuFaultPolicyEngine(load_xid_policy(), load_sxid_policy())
    incident, workflow = ResetOperationService(
        compiler.store, compiler, RLock()
    ).ingest_distributed_xids(
        batch, [policy.evaluate_xid(event) for event in batch.events]
    )
    return compiler.store, incident, workflow, nodes


class ContainmentNodes(KubernetesState):
    def __init__(self, nodes):
        super().__init__()
        self.nodes = {
            node: {
                "metadata": {
                    "name": node,
                    "uid": f"{node}-uid",
                    "resourceVersion": "1",
                    "annotations": {},
                },
                "spec": {"unschedulable": False, "taints": []},
            }
            for node in nodes
        }

    def patch_node(self, name, body):
        self.calls.append(("patch-node", name))
        current = self.nodes[name]
        assert (
            body["metadata"]["resourceVersion"]
            == current["metadata"]["resourceVersion"]
        ), "containment must retain resourceVersion preconditions"
        current["metadata"]["annotations"].update(body["metadata"]["annotations"])
        current["metadata"]["resourceVersion"] = str(
            int(current["metadata"]["resourceVersion"]) + 1
        )
        current["spec"].update(deepcopy(body["spec"]))


class Evidence:
    def supports(self, step):
        return (
            step.execution_owner == EVIDENCE_OWNER
            and step.operation is WorkflowOperation.FREEZE_EVIDENCE
        )

    def execute(self, context):
        return WorkflowStepOutcome.succeeded(operation_id=context.idempotency_key)


@pytest.mark.parametrize(
    ("nodes", "checkpoint"), [(32, False), (32, True), (64, False)]
)
def test_oversized_plan_runs_only_bounded_regional_containment(
    tmp_path, nodes, checkpoint
):
    store, incident, workflow, fault_nodes = plan(nodes, checkpoint=checkpoint)
    assert workflow.status is WorkflowStatus.SAFETY_PENDING, workflow
    assert workflow.safety_only and not workflow.dag_enabled, workflow
    assert any(
        "DISTRIBUTED_RESET_DAG_STEP_LIMIT" in reason
        for reason in workflow.blocked_reasons
    ), workflow.blocked_reasons
    assert len(workflow.official_steps) < 256, (
        "retain bounded, non-executable intent rather than materializing the oversized DAG"
    )
    assert [step.operation for step in workflow.safety_steps] == SAFETY, workflow
    assert all(step.node_ids == fault_nodes for step in workflow.safety_steps), (
        "the safe path must contain every affected node without truncation"
    )
    state = ContainmentNodes([*fault_nodes, "healthy-participant"])
    kubernetes = KubernetesWorkflowAdapter(
        owner=OWNER, core_api=state, batch_api=state, custom_api=state, store=None
    )
    store.save_regional_cluster(registration("cluster-a", TOKEN_A))
    regional = ClusterActionExecutor(
        StoreClient(store),
        [kubernetes],
        executor_id="plan-limit-unit",
        allowed_namespaces={"training"},
        claim_state_path=str(tmp_path / "claim.json"),
        liveness_state_path=str(tmp_path / "liveness.json"),
        sleep=lambda _: None,
    )
    control = active_workflow_executor(
        store,
        [Evidence(), RegionalRemoteWorkflowAdapter(store, owners={OWNER})],
        SAFETY,
    )
    for _ in range(8):
        result = execute_workflow(
            control, workflow.request_id, expected_fencing_token=1
        )
        if result.status is WorkflowStatus.BLOCKED:
            break
        regional.run_once()
    saved = store.get_workflow(workflow.request_id)
    assert saved.status is WorkflowStatus.BLOCKED, saved
    assert saved.blocked_kind is BlockedKind.SAFETY_SETTLED, saved
    assert saved.completed_operations == SAFETY, saved
    assert saved.blocked_reasons == workflow.blocked_reasons, saved
    assert (
        store.get_incident(incident.incident_id).state is IncidentState.QUARANTINED
    ), "safe containment must end with an explicit unrecovered incident"
    for node in fault_nodes:
        observed = state.nodes[node]
        assert observed["spec"]["unschedulable"] is True, observed
        assert any(
            item["key"] == QUARANTINE_TAINT for item in observed["spec"]["taints"]
        ), observed
    assert state.nodes["healthy-participant"]["spec"]["unschedulable"] is False, (
        "an unaffected allocation participant must not be quarantined"
    )
    assert all(
        command.step.operation in SAFETY for command in store.list_remote_commands()
    ), "no GPU action, provider escalation or workload restart may be dispatched"


def test_oversized_plan_without_containment_owner_is_explicitly_manual():
    store, _incident, workflow, nodes = plan(64, containment_owned=False)
    assert workflow.status is WorkflowStatus.BLOCKED, workflow
    assert workflow.blocked_kind is BlockedKind.NEEDS_OPERATOR, workflow
    assert any(
        "DISTRIBUTED_RESET_DAG_STEP_LIMIT" in reason
        for reason in workflow.blocked_reasons
    ), workflow.blocked_reasons
    assert len(nodes) == 64 and store.list_remote_commands() == [], workflow


def test_plan_just_below_the_limit_keeps_every_reset_branch():
    _store, _incident, workflow, nodes = plan(31)
    assert workflow.status is WorkflowStatus.PENDING and workflow.dag_enabled, workflow
    assert len(workflow.official_steps) == 250, workflow
    assert [
        step.node_ids
        for step in workflow.official_steps
        if step.operation is WorkflowOperation.RESET_GPU
    ] == [[node] for node in nodes], "under-limit plans must not lose a node branch"

from __future__ import annotations

from threading import RLock

import pytest

from gpu_fault.app import default_simulated_profile
from gpu_fault.models import (
    BlockedKind,
    CapabilityMode,
    CapabilityName,
    IncidentState,
    RecoveryAction,
    WorkflowOperation,
    WorkflowStatus,
)
from gpu_fault.orchestration.families.reset import ResetOperationService
from gpu_fault.policy import (
    ActionDisposition,
    DistributedXidBatch,
    GpuFaultPolicyEngine,
    load_sxid_policy,
    load_xid_policy,
)
from tests.orchestration._cov95_runtime_builder import builder, xid


@pytest.fixture(scope="module")
def policy():
    return GpuFaultPolicyEngine(load_xid_policy(), load_sxid_policy())


@pytest.fixture
def batch():
    return DistributedXidBatch(
        batch_id="reset-batch",
        job_id="training",
        attempt_id="training-a1",
        restart_budget=2,
        allocation=[
            {"node_id": "node-a", "rank": 0, "gpu_uuids": ["GPU-a"]},
            {"node_id": "node-b", "rank": 1, "gpu_uuids": ["GPU-b"]},
        ],
        affected_workload_ids=["training/pytorchjob/training"],
        events=[
            xid(
                event_id=f"member-{index}",
                xid=95,
                product="H200",
                runtime_profile_version="simulated-v1",
                job_id="training",
                attempt_id="training-a1",
                workload_state="ACTIVE",
                affected_workload_ids=["training/pytorchjob/training"],
            )
            for index in range(2)
        ],
    )


@pytest.fixture
def service():
    compiler = builder()
    compiler.store.save_profile(default_simulated_profile())
    return ResetOperationService(compiler.store, compiler, RLock())


def test_distributed_reset_deduplicates_gpu_scope_and_preserves_job_ownership(
    batch, service, policy
) -> None:
    decisions = [policy.evaluate_xid(event) for event in batch.events]
    incident, workflow = service.ingest_distributed_xids(batch, decisions)
    assert workflow.status is WorkflowStatus.PENDING, workflow
    assert incident.node_ids == ["node-a"] and incident.gpu_uuids == ["GPU-a"], incident
    assert (incident.cluster_id, incident.job_id, incident.attempt_id) == (
        "cluster-a",
        "training",
        "training-a1",
    ), "distributed recovery must remain discoverable for terminal withdrawal"
    steps = {step.operation: step for step in workflow.official_steps}
    reset = steps[WorkflowOperation.RESET_GPU]
    assert reset.parameters["gpu_uuids_by_node"] == {"node-a": ["GPU-a"]}, reset
    restart = steps[WorkflowOperation.RESTART_WORKLOAD]
    assert restart.node_ids == ["node-a", "node-b"], restart
    assert restart.parameters["source_gpu_count"] == 2, restart
    assert restart.parameters["source_attempt_id"] == "training-a1", restart
    assert restart.parameters["restart_budget"] == 2, restart
    for event in batch.events:
        linked = service.store.get_incident_by_event(event.event_id)
        assert linked is not None and linked.incident_id == incident.incident_id, (
            event.event_id,
            linked,
        )
    indexed = service.store.list_active_workflow_incidents(
        "cluster-a", job_id="training"
    )
    assert [item[1].request_id for item in indexed] == [workflow.request_id], indexed


def test_distributed_reset_replay_reuses_the_existing_pair(batch, service, policy):
    decisions = [policy.evaluate_xid(event) for event in batch.events]
    first = service.ingest_distributed_xids(batch, decisions)
    second = service.ingest_distributed_xids(batch, [])
    assert second == first, "a replay must return the committed pair without new work"
    assert len(service.store.list_workflows()) == 1, service.store.list_workflows()


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ("count", "every distributed XID event"),
        ("event-id", "do not match events"),
        ("action", "non-executable"),
        ("disposition", "non-executable"),
    ],
)
def test_distributed_reset_refuses_incomplete_or_nonexecutable_decisions(
    batch, service, policy, change, message
) -> None:
    decisions = [policy.evaluate_xid(event) for event in batch.events]
    if change == "count":
        decisions.pop()
    else:
        update = {
            "event-id": {"event_id": "unrelated-event"},
            "action": {"action": RecoveryAction.REBOOT_NODE},
            "disposition": {"disposition": ActionDisposition.BLOCKED_MISSING_EVIDENCE},
        }[change]
        decisions[0] = decisions[0].model_copy(update=update)
    with pytest.raises(ValueError, match=message):
        service.ingest_distributed_xids(batch, decisions)
    assert service.store.get_incident_by_event(batch.batch_id) is None, (
        "a rejected batch must not leave an incident or dispatchable workflow"
    )
    assert service.store.list_workflows() == [], service.store.list_workflows()


def test_distributed_reset_keeps_only_safety_dispatchable_when_reset_is_unowned(
    batch, service, policy
) -> None:
    profile = default_simulated_profile()
    service.store.save_profile(
        profile.model_copy(
            update={
                "capabilities": [
                    item.model_copy(update={"mode": CapabilityMode.OBSERVE})
                    if item.capability is CapabilityName.GPU_RESET
                    else item
                    for item in profile.capabilities
                ]
            }
        )
    )
    incident, workflow = service.ingest_distributed_xids(
        batch, [policy.evaluate_xid(event) for event in batch.events]
    )
    assert workflow.status is WorkflowStatus.SAFETY_PENDING, workflow
    assert workflow.safety_only and workflow.blocked_kind is None, workflow
    assert incident.state is IncidentState.SAFETY_PENDING, incident
    assert [step.operation for step in workflow.safety_steps] == [
        WorkflowOperation.FREEZE_EVIDENCE,
        WorkflowOperation.MARK_UNSCHEDULABLE,
        WorkflowOperation.QUARANTINE,
    ], workflow.safety_steps
    assert WorkflowOperation.RESET_GPU not in {
        step.operation for step in workflow.official_steps
    }, workflow.official_steps
    assert any("gpuReset" in reason for reason in workflow.blocked_reasons), workflow


def test_missing_profile_blocks_distributed_safety_and_recovery(batch, policy):
    compiler = builder()
    service = ResetOperationService(compiler.store, compiler, RLock())
    incident, workflow = service.ingest_distributed_xids(
        batch, [policy.evaluate_xid(event) for event in batch.events]
    )
    assert workflow.status is WorkflowStatus.BLOCKED, workflow
    assert workflow.blocked_kind is BlockedKind.NEEDS_OPERATOR, workflow
    assert incident.state is IncidentState.ESCALATED, incident
    assert workflow.official_steps == [] and workflow.safety_steps == [], workflow
    assert any(
        "profile does not exist" in reason for reason in workflow.blocked_reasons
    ), workflow

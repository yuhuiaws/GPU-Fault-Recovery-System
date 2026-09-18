from __future__ import annotations

import pytest

from gpu_fault.models import WorkflowOperation
from gpu_fault.policy import GpuFaultPolicyEngine, load_sxid_policy, load_xid_policy
from tests.orchestration._cov95_runtime_builder import builder, xid


@pytest.fixture(scope="module")
def policy():
    return GpuFaultPolicyEngine(load_xid_policy(), load_sxid_policy())


@pytest.mark.parametrize("checkpoint", [None, "s3://unit/checkpoint"])
@pytest.mark.parametrize("count", [1, 2])
def test_xid74_active_workload_is_stopped_before_reset_and_restarted_after_validation(
    policy, checkpoint, count
):
    event = xid(
        event_id=f"mechanical-{checkpoint}-{count}",
        xid=74,
        workload_state="ACTIVE",
        affected_workload_ids=["training/job/job-a"],
        checkpoint_manifest_ref=checkpoint,
        registers=[1 << 8, 0, 0, 0, 0, 0, 0],
        nvlink_link_id=2,
        nvlink_occurrence_counts={"register1.bit8": count},
    )
    decision = policy.evaluate_xid(event)
    operations = builder().catalog_operations(event, decision)
    expected = [WorkflowOperation.FREEZE_EVIDENCE, WorkflowOperation.MARK_UNSCHEDULABLE]
    if checkpoint:
        expected.append(WorkflowOperation.CHECKPOINT_WORKLOADS)
    expected.append(WorkflowOperation.STOP_WORKLOADS)
    if count == 2:
        expected.append(WorkflowOperation.COLLECT_DIAGNOSTIC_BUNDLE)
    expected.extend(
        [
            WorkflowOperation.QUIESCE_GPU_SERVICES,
            WorkflowOperation.VERIFY_NO_GPU_CLIENTS,
        ]
    )
    if count == 2:
        expected.append(WorkflowOperation.RUN_NVLINK74_WORKFLOW)
    expected.extend(
        [
            WorkflowOperation.RESET_GPU,
            WorkflowOperation.RESTORE_GPU_SERVICES,
            WorkflowOperation.VALIDATE_GPU,
            WorkflowOperation.VALIDATE_FABRIC,
            WorkflowOperation.RESTORE_SCHEDULING,
            WorkflowOperation.RESTART_WORKLOAD,
        ]
    )
    assert operations == expected, (decision, operations)


@pytest.mark.parametrize(
    ("registers", "expected"),
    [
        (
            [0, 0, 0, 1 << 18, 0, 0, 0],
            [
                WorkflowOperation.FREEZE_EVIDENCE,
                WorkflowOperation.COLLECT_DIAGNOSTIC_BUNDLE,
                WorkflowOperation.ESCALATE_SUPPORT,
            ],
        ),
        (
            [1 << 1, 0, 0, 0, 0, 0, 0],
            [WorkflowOperation.FREEZE_EVIDENCE, WorkflowOperation.ESCALATE_SUPPORT],
        ),
    ],
)
def test_xid74_unproven_fabric_or_secondary_fault_does_not_mutate_nodes(
    policy, registers, expected
):
    event = xid(
        event_id=f"support-{registers}",
        xid=74,
        workload_state="ACTIVE",
        affected_workload_ids=["training/job/job-a"],
        registers=registers,
        nvlink_link_id=2,
    )
    decision = policy.evaluate_xid(event)
    operations = builder().catalog_operations(event, decision)
    assert operations == expected, (decision, operations)

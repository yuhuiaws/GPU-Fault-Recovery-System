from __future__ import annotations

from typing import Any

import pytest

from gpu_fault.models import RecoveryAction, WorkflowOperation, WorkloadState
from gpu_fault.orchestration import IncidentOrchestrator
from gpu_fault.policy import ActionDisposition
from gpu_fault.store import InMemoryStore
from tests.orchestration._cov95_runtime_builder import builder, decision, sxid, xid

OP = WorkflowOperation


@pytest.mark.parametrize(
    ("kind", "updates", "operation", "expected"),
    [
        (
            "xid",
            {"xid": 74, "registers": [0] * 7},
            OP.RUN_NVLINK74_WORKFLOW,
            "explicit NVLink identity",
        ),
        (
            "xid",
            {"fabric_partition": None},
            OP.RESET_ALL_GPUS_NVSWITCHES,
            "fabric_partition mapping",
        ),
        (
            "xid",
            {"fabric_partition": "fabric-a"},
            OP.RESET_ALL_GPUS_NVSWITCHES,
            "complete node GPU inventory",
        ),
        (
            "sxid",
            {"participating_gpu_uuids": []},
            OP.RESET_ALL_GPUS_NVSWITCHES,
            "complete node GPU inventory",
        ),
        (
            "xid",
            {"gpu_uuid": None},
            OP.RESET_GPU,
            "RESET_GPU requires an explicit GPU UUID",
        ),
        (
            "xid",
            {"gpu_uuid": None, "pci_bdf": "0000:01:00.0"},
            OP.RESET_GPU,
            "fresh DCGM GPU inventory",
        ),
        ("sxid", {"participating_gpu_uuids": []}, OP.RESET_GPU, "explicit GPU UUID"),
        ("xid", {}, OP.RESTART_VM, "VM ownership"),
        ("xid", {}, OP.RESTART_WORKLOAD, "affected workload or job ID"),
        ("xid", {}, OP.STOP_WORKLOADS, "affected workload or job ID"),
        ("xid", {}, OP.UPDATE_SOFTWARE_FIRMWARE, "GPU_FAULT_TARGET_FIRMWARE_VERSION"),
    ],
)
def test_compiler_reports_missing_evidence_before_node_or_workload_actions(
    kind: str, updates: dict[str, Any], operation: WorkflowOperation, expected: str
) -> None:
    event = (sxid if kind == "sxid" else xid)(**updates)
    compiler = builder(target_firmware_version=None)
    errors = compiler.workflow_evidence_errors(event, [operation])
    assert any(expected in error for error in errors), errors


@pytest.mark.parametrize("official", ["UPDATE_SWFW", "RESET_ALL_GPUS_AND_NVSWITCHES"])
@pytest.mark.parametrize("active", [False, True])
@pytest.mark.parametrize("checkpoint", [False, True])
def test_catalog_repair_order_keeps_containment_before_mutation_and_restart_last(
    official: str, active: bool, checkpoint: bool
) -> None:
    event = (sxid if official.startswith("RESET_ALL") else xid)(
        workload_state=WorkloadState.ACTIVE if active else WorkloadState.IDLE,
        affected_workload_ids=["training/job/job-a"],
        checkpoint_manifest_ref="s3://unit/checkpoint" if checkpoint else None,
    )
    operations = builder().catalog_operations(
        event, decision(official_action=official, action=RecoveryAction.RESET_GPU)
    )
    assert (
        operations[0] is OP.FREEZE_EVIDENCE and operations[-1] is OP.RESTART_WORKLOAD
    ), operations
    assert operations.index(OP.MARK_UNSCHEDULABLE) < operations.index(
        OP.QUIESCE_GPU_SERVICES
    ), operations
    mutation = (
        OP.UPDATE_SOFTWARE_FIRMWARE
        if official == "UPDATE_SWFW"
        else OP.RESET_ALL_GPUS_NVSWITCHES
    )
    assert operations.index(OP.VERIFY_NO_GPU_CLIENTS) < operations.index(mutation), (
        operations
    )
    assert (
        operations.index(mutation)
        < operations.index(OP.RESTORE_GPU_SERVICES)
        < operations.index(OP.VALIDATE_GPU)
    ), operations
    assert operations.index(OP.RESTORE_SCHEDULING) < operations.index(
        OP.RESTART_WORKLOAD
    ), operations
    assert (OP.STOP_WORKLOADS in operations) is active, operations
    assert (OP.CHECKPOINT_WORKLOADS in operations) is (active and checkpoint), (
        operations
    )
    if active and checkpoint:
        assert operations.index(OP.CHECKPOINT_WORKLOADS) < operations.index(
            OP.STOP_WORKLOADS
        ), operations
    assert len(operations) == len(set(operations)), operations


@pytest.mark.parametrize("official", ["UPDATE_SWFW", "RESET_ALL_GPUS_AND_NVSWITCHES"])
def test_idle_repair_without_workload_ownership_never_invents_a_restart(
    official: str,
) -> None:
    event = sxid() if official.startswith("RESET_ALL") else xid()
    operations = builder().catalog_operations(event, decision(official_action=official))
    assert OP.RESTART_WORKLOAD not in operations, operations
    assert OP.STOP_WORKLOADS not in operations, operations


@pytest.mark.parametrize("checkpoint", [False, True])
def test_operator_escalation_keeps_workload_stop_ahead_of_quarantine(
    checkpoint: bool,
) -> None:
    event = xid(
        workload_state=WorkloadState.ACTIVE,
        checkpoint_manifest_ref="s3://unit/checkpoint" if checkpoint else None,
    )
    operations = builder().catalog_operations(
        event,
        decision(
            action=RecoveryAction.ESCALATE_OPERATOR, official_action="CONTACT_SUPPORT"
        ),
    )
    expected = [OP.FREEZE_EVIDENCE, OP.MARK_UNSCHEDULABLE]
    if checkpoint:
        expected.append(OP.CHECKPOINT_WORKLOADS)
    expected.extend([OP.STOP_WORKLOADS, OP.QUARANTINE, OP.ESCALATE_SUPPORT])
    assert operations == expected, operations


def test_missing_evidence_safety_plan_never_compiles_the_requested_reset() -> None:
    operations = builder().catalog_operations(
        xid(),
        decision(
            disposition=ActionDisposition.BLOCKED_MISSING_EVIDENCE,
            safety_action=RecoveryAction.QUARANTINE,
            action=RecoveryAction.RESET_GPU,
        ),
    )
    assert operations == [OP.FREEZE_EVIDENCE, OP.QUARANTINE], operations


@pytest.mark.parametrize("active", [False, True])
def test_pre_actions_are_inserted_once_and_only_stop_active_workloads(
    active: bool,
) -> None:
    compiler = builder()
    event = xid(
        workload_state=WorkloadState.ACTIVE if active else WorkloadState.IDLE,
        checkpoint_manifest_ref="s3://unit/checkpoint",
    )
    required = decision(
        pre_actions=[RecoveryAction.MARK_UNSCHEDULABLE, RecoveryAction.STOP_WORKLOAD]
    )
    original = [OP.FREEZE_EVIDENCE, OP.COLLECT_DIAGNOSTIC_BUNDLE]
    result = compiler.with_pre_actions(event, required, original)
    expected = [OP.FREEZE_EVIDENCE, OP.MARK_UNSCHEDULABLE]
    if active:
        expected.extend([OP.CHECKPOINT_WORKLOADS, OP.STOP_WORKLOADS])
    expected.append(OP.COLLECT_DIAGNOSTIC_BUNDLE)
    assert result == expected, result
    assert compiler.with_pre_actions(event, required, result) == result, result
    assert original == [OP.FREEZE_EVIDENCE, OP.COLLECT_DIAGNOSTIC_BUNDLE], original


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({"multi_node_aggregation_window_seconds": -1}, "must be non-negative"),
        (
            {
                "multi_node_aggregation_window_seconds": 5,
                "multi_node_aggregation_window_max_seconds": 4,
            },
            "must not be less",
        ),
        ({"processor_drain_max_wait_seconds": -1}, "must be non-negative"),
        ({"fault_action_max_age_seconds": 0}, "must be positive"),
        (
            {"sxid_driver_remediation_codes": {100}},
            "require GPU_FAULT_TARGET_DRIVER_BRANCH",
        ),
        (
            {"sxid_firmware_update_codes": {100}},
            "require GPU_FAULT_TARGET_FIRMWARE_VERSION",
        ),
    ],
)
def test_orchestrator_rejects_inconsistent_timing_and_remediation_configuration(
    kwargs: dict[str, Any], expected: str
) -> None:
    with pytest.raises(ValueError, match=expected):
        IncidentOrchestrator(InMemoryStore(), **kwargs)

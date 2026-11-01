"""Batch 233: the plans the node-health family compiles for the new findings.

A DCGM field-1523 DRAIN_P2P finding gets the XID 154 path's drain-P2P chain
(not the generic STOP_WORKLOAD hand-off), ESCALATE_OPERATOR findings freeze
evidence and hand the incident to the support template (cordoned first when
CRITICAL), the SRAM threshold flag takes the row-remap failure's RMA drain --
including the containment prefix a ``memoryDiagnostics: OBSERVE`` profile
runs before it blocks -- and the findings a reset or quiesce window raises
are that window's to judge.
"""

from __future__ import annotations

from gpu_fault.models import (
    CapabilityName,
    IncidentState,
    RecoveryAction,
    WorkflowOperation,
    WorkflowStatus,
    WorkloadState,
)
from gpu_fault.orchestration.families.conflicts import (
    INVENTORY_FINDING_VALIDATIONS,
    TRANSIENT_INVENTORY_FINDINGS,
)
from gpu_fault.policy import GpuFaultPolicyEngine, parse_xid154_action
from gpu_fault.workflow_quarantine import TERMINAL_QUARANTINE_NODES
from tests.orchestration._cov95_orch_extra_health import health_service
from tests.orchestration._cov95_orch_extra_support import finding, memory_store
from tests.orchestration._cov95_runtime_builder import builder, xid
from tests.orchestration.test_containment_before_compile_block import (
    MEMORY_REASON,
    OBSERVE_PROFILE,
    observe_profile,
)

Op = WorkflowOperation
JOB = "training/job/unit-job"


def operations(workflow) -> list[WorkflowOperation]:
    return [step.operation for step in workflow.official_steps]


def recovery_action(event_id: str, **updates):
    values = {
        "severity": "critical",
        "reason": "driver GPU recovery action Drain P2P (DCGM field 1523)",
        "recommended_action": RecoveryAction.STOP_WORKLOAD,
        "official_action": "DRAIN_P2P",
        "metric_name": "gpu_recovery_action",
        "gpu_uuids": ["GPU-a"],
        "node_id": "node-a",
        "policy_source": "NVIDIA_DCGM_GPU_RECOVERY_ACTION",
        "workload_state": WorkloadState.ACTIVE,
        "affected_workload_ids": [JOB],
        **updates,
    }
    return finding(event_id, **values)


def test_a_drain_p2p_recovery_action_compiles_the_xid154_drain_chain():
    raw = (
        "NVRM: Xid (PCI:0000:59:00): 154, GPU recovery action changed from 0x0 "
        "(None) to 0x3 (Drain P2P)"
    )
    event = xid(
        xid=154,
        product="H200",
        raw_message=raw,
        xid_154_action=parse_xid154_action(raw),
        workload_state=WorkloadState.ACTIVE,
        affected_workload_ids=[JOB],
    )
    decision = GpuFaultPolicyEngine().evaluate_xid(event, xid154_window_closed=True)
    compiler = builder()
    xid_path = compiler.with_pre_actions(
        event, decision, compiler.catalog_operations(event, decision)
    )
    store = memory_store()
    service, _absorbed = health_service(store)

    _incident, workflow = service.ingest(recovery_action("dcgm-drain-p2p"))

    assert decision.official_action == "DRAIN_P2P"
    assert operations(workflow) == xid_path
    assert operations(workflow)[:7] == [
        Op.FREEZE_EVIDENCE,
        Op.MARK_UNSCHEDULABLE,
        Op.STOP_WORKLOADS,
        Op.QUIESCE_GPU_SERVICES,
        Op.VERIFY_NO_GPU_CLIENTS,
        Op.RECOVER_DRAIN_P2P,
        Op.RESTORE_GPU_SERVICES,
    ]
    assert workflow.official_action == "DRAIN_P2P"


def test_a_plain_stop_workload_finding_keeps_its_hand_off():
    store = memory_store()
    service, _absorbed = health_service(store)

    _incident, workflow = service.ingest(
        recovery_action("plain-stop", official_action=None, metric_name="unit")
    )

    assert Op.RECOVER_DRAIN_P2P not in operations(workflow)
    assert operations(workflow)[-1] is Op.ESCALATE_SUPPORT


def test_escalate_operator_freezes_evidence_and_hands_off_to_support():
    store = memory_store()
    service, _absorbed = health_service(store)

    incident, warning = service.ingest(
        finding(
            "ecc-disabled",
            severity="warning",
            reason="ECC is disabled on this GPU (DCGM field 300)",
            recommended_action=RecoveryAction.ESCALATE_OPERATOR,
            metric_name="ecc_mode_current",
            gpu_uuids=["GPU-a"],
            node_id="node-a",
        )
    )
    _incident, critical = service.ingest(
        finding(
            "imex-recovery",
            severity="critical",
            reason="driver GPU recovery action RECOVER_IMEX_DOMAIN",
            recommended_action=RecoveryAction.ESCALATE_OPERATOR,
            official_action="RECOVER_IMEX_DOMAIN",
            metric_name="gpu_recovery_action",
            gpu_uuids=["GPU-b"],
            node_id="node-b",
        )
    )

    assert operations(warning) == [Op.FREEZE_EVIDENCE, Op.ESCALATE_SUPPORT]
    assert warning.status is WorkflowStatus.PENDING
    assert incident.state is IncidentState.ACTION_PENDING
    assert operations(critical) == [
        Op.FREEZE_EVIDENCE,
        Op.MARK_UNSCHEDULABLE,
        Op.ESCALATE_SUPPORT,
    ], "a recovery no product step does: no new work, then the operator"


def sram(**updates):
    values = {
        "severity": "critical",
        "reason": "SRAM threshold exceeded (DCGM field 346, an RMA criterion)",
        "recommended_action": RecoveryAction.DRAIN,
        "official_action": "RUN_FIELD_DIAGNOSTIC_FOR_RMA",
        "metric_name": "sram_threshold_exceeded",
        "gpu_uuids": ["GPU-0"],
        "pci_bdf": "0000:81:00.0",
        "node_id": "node-0",
        **updates,
    }
    return finding("gpu-dcgm-node-0-GPU-0-sram_threshold_exceeded", **values)


def test_the_sram_flag_takes_the_row_remap_failures_rma_drain():
    store = memory_store()
    service, _absorbed = health_service(store)

    _incident, workflow = service.ingest(sram())

    assert operations(workflow) == [
        Op.FREEZE_EVIDENCE,
        Op.MARK_UNSCHEDULABLE,
        Op.QUARANTINE,
        Op.COLLECT_DIAGNOSTIC_BUNDLE,
        Op.RUN_FIELD_DIAGNOSTIC,
        Op.VALIDATE_GPU,
        Op.ESCALATE_SUPPORT,
    ]


def test_the_sram_flag_under_memory_observe_contains_before_it_blocks():
    """b192's containment prefix, as for ``row_remap_failure``."""

    store = memory_store()
    observe_profile(store, CapabilityName.MEMORY_DIAGNOSTICS)
    service, _absorbed = health_service(store)

    incident, workflow = service.ingest(sram(runtime_profile_version=OBSERVE_PROFILE))

    assert workflow.status is WorkflowStatus.SAFETY_PENDING, workflow.blocked_reasons
    assert workflow.blocked_reasons == [MEMORY_REASON]
    assert [step.operation for step in workflow.safety_steps] == [
        Op.FREEZE_EVIDENCE,
        Op.MARK_UNSCHEDULABLE,
        Op.QUARANTINE,
        Op.COLLECT_DIAGNOSTIC_BUNDLE,
    ]
    assert workflow.safety_steps[2].parameters[TERMINAL_QUARANTINE_NODES] == ["node-0"]
    assert incident.state is IncidentState.SAFETY_PENDING


def test_an_active_workload_on_a_legacy_sram_quarantine_is_stopped():
    store = memory_store()
    service, _absorbed = health_service(store)

    _incident, workflow = service.ingest(
        sram(
            recommended_action=RecoveryAction.QUARANTINE,
            official_action="REVIEW_GPU_ARCHITECTURE",
            workload_state=WorkloadState.ACTIVE,
            affected_workload_ids=[JOB],
        )
    )

    assert Op.STOP_WORKLOADS in operations(workflow), (
        "as for row_remap_failure: an SRAM RMA flag stops the work on isolation"
    )


def test_a_reset_or_quiesce_window_owns_the_downtrain_and_recovery_findings_it_raises():
    for name in ("pcie_link_downtrain", "gpu_recovery_action"):
        assert {Op.QUIESCE_GPU_SERVICES, Op.RESET_GPU} <= TRANSIENT_INVENTORY_FINDINGS[
            name
        ]
        assert INVENTORY_FINDING_VALIDATIONS[name] == frozenset({Op.VALIDATE_GPU})

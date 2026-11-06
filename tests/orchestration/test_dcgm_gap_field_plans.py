"""Batch 233: the plans the node-health family compiles for the new findings.

A DCGM field-1523 DRAIN_P2P finding gets the XID 154 path's drain-P2P chain
(not the generic STOP_WORKLOAD hand-off), ESCALATE_OPERATOR findings freeze
evidence and hand the incident to the support template (cordoned first when
CRITICAL), the SRAM threshold flag takes the row-remap failure's RMA drain --
including the containment prefix a ``memoryDiagnostics: OBSERVE`` profile
runs before it blocks -- and the findings a reset or quiesce window raises
are that window's to judge. The advisory WARNING diagnostics -- 1423, 1424
below the brake verdict, the downtrain and 1501 -- take the read-only chain:
FREEZE_EVIDENCE -> VALIDATE_GPU, no DCGM diagnostic, no DRAIN on a failure.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from gpu_fault.app.ingest.gpu_findings import gpu_node_health_findings
from gpu_fault.execution.terminal_state import failure_incident_state
from gpu_fault.gpu_metrics import GpuMetricSample, GpuMetricSource, GpuMetricsService
from gpu_fault.health_recovery_contract import HEALTH_CONTRACT
from gpu_fault.models import (
    CapabilityName,
    IncidentState,
    RecoveryAction,
    WorkflowOperation,
    WorkflowStatus,
    WorkflowStepStatus,
    WorkloadState,
)
from gpu_fault.orchestration.escalation import HardwareEscalationService
from gpu_fault.orchestration.families.conflicts import (
    INVENTORY_FINDING_VALIDATIONS,
    TRANSIENT_INVENTORY_FINDINGS,
)
from gpu_fault.policy import GpuFaultPolicyEngine, parse_xid154_action
from gpu_fault.workflow_quarantine import TERMINAL_QUARANTINE_NODES
from tests._builders import build_store, gpu_metric_batch, workflow_step_execution
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


# --- the advisory diagnostics: read-only, never a DRAIN --------------------

NOW = datetime.now(timezone.utc).replace(microsecond=0)
BRAKE = "hw_power_brake_violation_total_us"
THERMAL_SLOWDOWN = "hw_thermal_slowdown_violation_total_us"
GEN5_X16 = {"pcie_link_gen_current": 5, "pcie_link_width_current": 16}
DOWNTRAIN_LABELS = {
    "pcie_link_gen_current": "5",
    "pcie_link_width_current": "8",
    "pcie_link_gen_max": "5",
    "pcie_link_width_max": "16",
    "consecutive_valid_samples": "3",
}


def field_finding(
    name: str,
    batches: list[dict[str, float]],
    *,
    product: str = "H200",
    labels: dict[str, str] | None = None,
    gpu: str = "GPU-a",
):
    """The node-health finding the production path books for ``name`` after
    ``batches`` (one dict of readings per scrape, 15 s apart) through the real
    ``GpuMetricsService``."""

    metrics = GpuMetricsService(store=build_store())
    result = None
    for index, values in enumerate(batches):
        result = metrics.ingest(
            gpu_metric_batch(
                f"field-{name}-{index}",
                NOW + timedelta(seconds=15 * index),
                GpuMetricSource.DCGM_EXPORTER,
                [
                    GpuMetricSample(
                        metric_name=metric,
                        canonical_name=metric,
                        value=value,
                        gpu_uuid=gpu,
                        pci_bdf="0000:59:00.0",
                        labels={
                            "modelName": product,
                            "source_boot_id": "boot-a",
                            **(labels or {}),
                        },
                    )
                    for metric, value in values.items()
                ],
                product=product,
                runtime_profile_version="simulated-v1",
                workload_state=WorkloadState.IDLE,
            )
        )
    assert result is not None
    found = [
        item
        for item in gpu_node_health_findings(None, result.new_findings)
        if item.metric_name == name
    ]
    assert len(found) == 1, (name, [item.metric_name for item in found])
    return found[0]


def failed_validation(workflow):
    """``workflow`` after its VALIDATE_GPU failed on a finding still active."""

    index = next(
        position
        for position, step in enumerate(workflow.official_steps)
        if step.operation is Op.VALIDATE_GPU
    )
    return workflow.model_copy(
        update={
            "status": WorkflowStatus.FAILED,
            "completed_step_indexes": list(range(index)),
            "completed_operations": [
                step.operation for step in workflow.official_steps[:index]
            ],
            "step_executions": [
                *(
                    workflow_step_execution(position, step.operation)
                    for position, step in enumerate(workflow.official_steps[:index])
                ),
                workflow_step_execution(
                    index,
                    Op.VALIDATE_GPU,
                    WorkflowStepStatus.FAILED,
                    error="active_gpu_health_findings",
                    details={
                        "failed_nodes": ["node-a"],
                        "node_failures": {"node-a": ["active_gpu_health_findings"]},
                    },
                ),
            ],
        }
    )


def assert_read_only_diagnostic(node_finding, why: str) -> None:
    """The finding compiles the read-only chain, and a validation that fails
    on it is a diagnostic verdict: no DCGM diagnostic, no rung, no DRAIN."""

    assert node_finding.recommended_action is RecoveryAction.RUN_DIAGNOSTICS, (
        node_finding.recommended_action
    )
    store = memory_store()
    service, _absorbed = health_service(store)

    incident, workflow = service.ingest(node_finding)

    assert operations(workflow) == [Op.FREEZE_EVIDENCE, Op.VALIDATE_GPU], (
        f"{node_finding.metric_name}: {why}; a DCGM diagnostic whose failure "
        "drains is not this finding's plan"
    )
    validation = workflow.official_steps[-1]
    assert validation.parameters[HEALTH_CONTRACT]["mode"] == "DIAGNOSTIC_ONLY", (
        validation.parameters
    )
    failed = failed_validation(workflow)
    assert HardwareEscalationService.classify(failed) is None, (
        f"{node_finding.metric_name}: a failed read-only validation climbs no rung "
        "and opens no DRAIN"
    )
    assert failure_incident_state(failed) is IncidentState.RECOVERED, (
        "the read-only chain ends 'diagnostic inconclusive', the node untouched"
    )
    assert incident.state is IncidentState.ACTION_PENDING


@pytest.mark.parametrize(
    ("name", "batches", "labels", "why"),
    [
        pytest.param(
            THERMAL_SLOWDOWN,
            [{THERMAL_SLOWDOWN: 0}, {THERMAL_SLOWDOWN: 150_000}],
            None,
            "1423 grades like the 241 thermal violation time, a read-only diagnostic",
            id="1423-hw-thermal-slowdown",
        ),
        pytest.param(
            BRAKE,
            [
                {BRAKE: 0, "clock_throttle_reasons": 0},
                {BRAKE: 15_000, "clock_throttle_reasons": 0},
            ],
            None,
            "1424 below the brake verdict grades like the 240 power violation time",
            id="1424-power-brake-duty",
        ),
        pytest.param(
            "pcie_link_downtrain",
            [{"pcie_link_downtrain": 1}],
            DOWNTRAIN_LABELS,
            "the downtrain is a load-gated advisory; its escalation is the operator "
            "once a reset did not retrain the link",
            id="235-238-downtrain",
        ),
        pytest.param(
            "pcie_correctable_error_total",
            [
                {"pcie_correctable_error_total": 0, **GEN5_X16},
                {"pcie_correctable_error_total": 10, **GEN5_X16},
                {"pcie_correctable_error_total": 20, **GEN5_X16},
            ],
            None,
            "1501 is a rate diagnostic like the 202 replay rate",
            id="1501-pcie-correctable",
        ),
    ],
)
def test_an_advisory_gap_field_diagnoses_read_only_and_never_drains(
    name, batches, labels, why
):
    assert_read_only_diagnostic(field_finding(name, batches, labels=labels), why)


def test_the_confirmed_power_brake_keeps_its_drain():
    """The read-only chain is the duty-cycle WARNING's only: the brake verdict
    (>= 1 % in two consecutive samples) still drains for the power review."""

    brake = field_finding(
        BRAKE,
        [
            {BRAKE: 0, "clock_throttle_reasons": 0},
            {BRAKE: 15_000, "clock_throttle_reasons": 0},
            {BRAKE: 165_000, "clock_throttle_reasons": 0},
            {BRAKE: 315_000, "clock_throttle_reasons": 0},
        ],
    )
    store = memory_store()
    service, _absorbed = health_service(store)

    _incident, workflow = service.ingest(brake)

    assert brake.recommended_action is RecoveryAction.DRAIN, brake.reason
    assert operations(workflow) == [
        Op.FREEZE_EVIDENCE,
        Op.MARK_UNSCHEDULABLE,
        Op.QUARANTINE,
        Op.COLLECT_DIAGNOSTIC_BUNDLE,
        Op.VALIDATE_GPU,
        Op.ESCALATE_SUPPORT,
    ]

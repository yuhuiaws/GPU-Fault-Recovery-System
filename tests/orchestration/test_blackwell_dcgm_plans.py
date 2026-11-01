"""Batch 235: the plans the Blackwell DCGM findings compile.

A failed NVLink5 link recovery (DCGM field 1212) takes the DRAIN chain; an
unrepairable memory flag takes the row-remap failure's RMA drain, including the
containment prefix a ``memoryDiagnostics: OBSERVE`` profile runs before it
blocks; a pending memory repair hands an operator the reboot; a confirmed
unhealthy fabric (174/175) takes the full GPU + NVSwitch reset over the node's
whole inventory, as an untrained one does; and the findings a product window
raises are that window's to judge. Findings are produced by the real
``GpuMetricsService`` from synthetic, documentation-derived samples.
"""

from __future__ import annotations

from datetime import datetime, timezone

from gpu_fault.app import default_simulated_profile
from gpu_fault.app.ingest.gpu_findings import gpu_node_health_findings
from gpu_fault.gpu_fabric_health import (
    FABRIC_HEALTH,
    FABRIC_HEALTH_HELD_SECONDS,
    FABRIC_HEALTH_MASK,
    FABRIC_HEALTH_SUMMARY,
)
from gpu_fault.gpu_memory_repair_policy import (
    MEMORY_CHANNEL_REPAIR_PENDING,
    MEMORY_UNREPAIRABLE,
)
from gpu_fault.gpu_metrics import GpuMetricSample, GpuMetricSource, GpuMetricsService
from gpu_fault.gpu_nvlink5_policy import NVLINK5_COUNTERS, NVLINK5_RECOVERY_FAILED
from gpu_fault.models import (
    CapabilityName,
    IncidentState,
    RecoveryAction,
    WorkflowOperation,
    WorkflowStatus,
    WorkloadState,
)
from gpu_fault.orchestration import IncidentOrchestrator
from gpu_fault.orchestration.families.conflicts import (
    FABRIC_MANAGER_OWNED_OPERATIONS,
    INVENTORY_FINDING_VALIDATIONS,
    TRANSIENT_INVENTORY_FINDINGS,
)
from tests._builders import build_store, gpu_metric_batch
from tests.orchestration._cov95_orch_extra_health import health_service
from tests.orchestration._cov95_orch_extra_support import memory_store
from tests.orchestration.test_containment_before_compile_block import (
    MEMORY_REASON,
    OBSERVE_PROFILE,
    observe_profile,
)

Op = WorkflowOperation
CLUSTER = "cluster-a"
NODE = "node-0"
NOW = datetime.now(timezone.utc).replace(microsecond=0)
GPUS = [f"GPU-b300-{index}" for index in range(8)]
B300 = "NVIDIA B300 SXM6 AC"


def node_findings(samples_by_batch, *, profile="simulated-v1", store=None):
    """The node-health findings the production path books for ``samples``."""

    metrics = GpuMetricsService(store=store or build_store())
    result = None
    for index, samples in enumerate(samples_by_batch):
        result = metrics.ingest(
            gpu_metric_batch(
                f"bw-{index}",
                NOW.replace(second=0) if index == 0 else NOW,
                GpuMetricSource.DCGM_EXPORTER,
                samples,
                cluster_id=CLUSTER,
                node_id=NODE,
                product=B300,
                runtime_profile_version=profile,
                workload_state=WorkloadState.IDLE,
            )
        )
    assert result is not None
    return gpu_node_health_findings(None, result.new_findings)


def sample(name: str, value: float, gpu: str = GPUS[0]) -> GpuMetricSample:
    return GpuMetricSample(
        metric_name=name,
        canonical_name=name,
        value=value,
        gpu_uuid=gpu,
        pci_bdf=f"00000000:4{gpu[-1]}:00.0",
        labels={"modelName": B300, "source_boot_id": "boot-a"},
    )


def operations(workflow) -> list[WorkflowOperation]:
    return [step.operation for step in workflow.official_steps]


def test_a_failed_link_recovery_compiles_the_drain_chain():
    (finding,) = node_findings(
        [[sample(NVLINK5_RECOVERY_FAILED, 0)], [sample(NVLINK5_RECOVERY_FAILED, 1)]]
    )
    store = memory_store()
    service, _absorbed = health_service(store)

    incident, workflow = service.ingest(finding)

    assert finding.recommended_action is RecoveryAction.DRAIN
    assert operations(workflow) == [
        Op.FREEZE_EVIDENCE,
        Op.MARK_UNSCHEDULABLE,
        Op.QUARANTINE,
        Op.COLLECT_DIAGNOSTIC_BUNDLE,
        Op.VALIDATE_GPU,
        Op.ESCALATE_SUPPORT,
    ]
    assert incident.official_action == "RUN_FIELD_DIAGNOSTIC"


def test_unrepairable_memory_compiles_the_rma_drain_with_the_field_diagnostic():
    (finding,) = node_findings([[sample(MEMORY_UNREPAIRABLE, 1)]])
    store = memory_store()
    service, _absorbed = health_service(store)

    _incident, workflow = service.ingest(finding)

    assert operations(workflow) == [
        Op.FREEZE_EVIDENCE,
        Op.MARK_UNSCHEDULABLE,
        Op.QUARANTINE,
        Op.COLLECT_DIAGNOSTIC_BUNDLE,
        Op.RUN_FIELD_DIAGNOSTIC,
        Op.VALIDATE_GPU,
        Op.ESCALATE_SUPPORT,
    ]


def test_unrepairable_memory_under_memory_observe_contains_before_it_blocks():
    store = memory_store()
    observe_profile(store, CapabilityName.MEMORY_DIAGNOSTICS)
    (finding,) = node_findings(
        [[sample(MEMORY_UNREPAIRABLE, 1)]], profile=OBSERVE_PROFILE
    )
    service, _absorbed = health_service(store)

    incident, workflow = service.ingest(finding)

    assert workflow.status is WorkflowStatus.SAFETY_PENDING, workflow.blocked_reasons
    assert workflow.blocked_reasons == [MEMORY_REASON]
    assert [step.operation for step in workflow.safety_steps] == [
        Op.FREEZE_EVIDENCE,
        Op.MARK_UNSCHEDULABLE,
        Op.QUARANTINE,
        Op.COLLECT_DIAGNOSTIC_BUNDLE,
    ]
    assert incident.state is IncidentState.SAFETY_PENDING


def test_a_pending_repair_hands_the_reboot_to_an_operator():
    (finding,) = node_findings(
        [[sample(MEMORY_CHANNEL_REPAIR_PENDING, 1), sample("gpu_recovery_action", 0)]]
    )
    store = memory_store()
    service, _absorbed = health_service(store)

    _incident, workflow = service.ingest(finding)

    assert operations(workflow) == [Op.FREEZE_EVIDENCE, Op.ESCALATE_SUPPORT]
    assert finding.official_action == "REBOOT_IN_SERVICE_WINDOW"


def fabric_samples(held: float) -> list[GpuMetricSample]:
    samples = []
    for gpu in GPUS:
        unhealthy = gpu == GPUS[3]
        samples += [
            sample(FABRIC_HEALTH_MASK, 0x119A if unhealthy else 0x11AA, gpu),
            sample(FABRIC_HEALTH_SUMMARY, 2 if unhealthy else 1, gpu),
        ]
        if unhealthy:
            samples.append(sample(FABRIC_HEALTH_HELD_SECONDS, held, gpu))
    return samples


def test_a_confirmed_unhealthy_fabric_plans_the_full_reset_of_the_whole_inventory():
    (finding,) = node_findings([fabric_samples(0), fabric_samples(90)])
    store = build_store()
    store.save_profile(default_simulated_profile())
    orchestrator = IncidentOrchestrator(store, multi_node_aggregation_window_seconds=30)

    _incident, workflow = orchestrator.ingest_node_health(finding)

    assert finding.metric_name == FABRIC_HEALTH
    assert workflow is not None and workflow.status is WorkflowStatus.PENDING
    assert workflow.official_action == "RESET_ALL_GPUS_AND_NVSWITCHES"
    assert operations(workflow) == [
        Op.FREEZE_EVIDENCE,
        Op.MARK_UNSCHEDULABLE,
        Op.QUIESCE_GPU_SERVICES,
        Op.VERIFY_NO_GPU_CLIENTS,
        Op.RESET_ALL_GPUS_NVSWITCHES,
        Op.RESTORE_GPU_SERVICES,
        Op.VALIDATE_GPU,
        Op.VALIDATE_FABRIC,
        Op.RESTORE_SCHEDULING,
    ]
    (reset,) = [
        step
        for step in workflow.official_steps
        if step.operation is Op.RESET_ALL_GPUS_NVSWITCHES
    ]
    assert reset.gpu_uuids == GPUS, (
        "the Node Agent refuses a fabric reset whose UUIDs are not the inventory"
    )
    assert GPUS[3] in finding.reason


def test_a_product_window_owns_what_its_links_raise():
    for name in (FABRIC_HEALTH, *sorted(NVLINK5_COUNTERS)):
        assert TRANSIENT_INVENTORY_FINDINGS[name] == FABRIC_MANAGER_OWNED_OPERATIONS
        assert INVENTORY_FINDING_VALIDATIONS[name] == frozenset({Op.VALIDATE_GPU})
    assert {
        Op.QUIESCE_GPU_SERVICES,
        Op.RESET_GPU,
        Op.RESET_ALL_GPUS_NVSWITCHES,
        Op.RESTORE_GPU_SERVICES,
        Op.RESTART_FABRIC_MANAGER,
        Op.RESTART_NODE,
    } <= FABRIC_MANAGER_OWNED_OPERATIONS

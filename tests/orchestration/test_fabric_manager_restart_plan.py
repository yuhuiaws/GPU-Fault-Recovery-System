"""RESTART_FM on a Hopper NVSwitch node is NVIDIA's H100-and-later restart (batch 128).

XID 45 alone finalizes as the catalog's RESTART_FM. On a node whose NVLink
fabric batch 127 grades (a Hopper SXM part on NVSwitches, field 170 reported),
a Fabric Manager restart alone loses the fabric state: the restarted service
needs every GPU of the fabric reset (Fabric Manager user guide), and the
documented procedure is stop the CUDA applications and GPU services, stop
Fabric Manager, ``nvidia-smi -r``, start Fabric Manager. The plan there is the
fatal-trunk SXID's chain on the node's graded GPUs; the window stops Fabric
Manager (it is a full reset), the restore starts it, both validations run.
Every other node keeps FREEZE_EVIDENCE -> RESTART_FABRIC_MANAGER.

From batch 131 a solo XID 45 reaches that chain only when the site opts in
(``GPU_FAULT_XID45_SOLO_NVSWITCH_RESTART_FM=true``); by default it is
observe-only on a graded node (``tests/orchestration/test_xid45_observe_only.py``).
The chain tests here therefore run opted in.
"""

from __future__ import annotations

import pytest

from gpu_fault.gpu_fabric_manager import FABRIC_MANAGER_STATUS, graded_fabric_gpus
from gpu_fault.gpu_fabric_window import quiesce_command_parameters
from gpu_fault.gpu_metrics import GpuMetricSample, GpuMetricSource, GpuMetricsService
from gpu_fault.orchestration.xid45_policy import XID45_SOLO_RESTART_FM_ENV
from tests._builders import build_store, gpu_metric_batch

from ._support import (
    NOW,
    ApplicationContext,
    IncidentState,
    WorkflowOperation,
    WorkflowStatus,
    WorkloadState,
    event,
)

Op = WorkflowOperation
GPUS = [f"GPU-{index}" for index in range(8)]
FABRIC_RESTART_PLAN = [
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


@pytest.fixture
def opted_in(monkeypatch: pytest.MonkeyPatch) -> ApplicationContext:
    """A control plane whose site runs the catalog's RESTART_FM on graded nodes."""

    monkeypatch.setenv(XID45_SOLO_RESTART_FM_ENV, "true")
    return ApplicationContext(store=build_store())


def _report(context: ApplicationContext, model: str, status: float) -> None:
    samples = [
        GpuMetricSample(
            metric_name="DCGM_FI_DEV_FABRIC_MANAGER_STATUS",
            canonical_name=FABRIC_MANAGER_STATUS,
            value=status,
            gpu_uuid=uuid,
            labels={"modelName": model},
        )
        for uuid in GPUS
    ]
    GpuMetricsService(store=context.store).ingest(
        gpu_metric_batch(
            "dcgm-fabric",
            NOW,
            GpuMetricSource.DCGM_EXPORTER,
            samples,
            runtime_profile_version="simulated-v1",
            workload_state=WorkloadState.IDLE,
        )
    )


def _solo_xid45(context: ApplicationContext, **values):
    xid_event = event(45, event_id=values.pop("event_id", "solo-xid45"), **values)
    decision = context.policy.evaluate_xid(xid_event, xid45_window_closed=True)
    assert decision.official_action == "RESTART_FM"
    incident, workflow = context.orchestrator.ingest(xid_event, decision)
    return incident, workflow


@pytest.mark.parametrize("model", ["NVIDIA H200", "NVIDIA B200", "NVIDIA B300 SXM6 AC"])
def test_a_hopper_nvswitch_node_plans_the_full_gpu_and_nvswitch_reset(
    opted_in: ApplicationContext, model: str
) -> None:
    context = opted_in
    _report(context, model, 3.0)

    incident, workflow = _solo_xid45(context)

    assert incident.state is IncidentState.ACTION_PENDING
    assert workflow.status is WorkflowStatus.PENDING, workflow.blocked_reasons
    assert workflow.official_action == "RESTART_FM"
    assert [step.operation for step in workflow.official_steps] == FABRIC_RESTART_PLAN
    reset = workflow.official_steps[4]
    assert reset.gpu_uuids == GPUS, "every graded GPU, not the XID's one"
    assert reset.parameters == {
        "fabric_partition": "cluster-a/node-a/local-nvswitch",
        "xid": 45,
    }
    assert {tuple(step.gpu_uuids) for step in workflow.official_steps} == {
        tuple(GPUS)
    }, "every step of the chain names the whole fabric"
    quiesce = workflow.official_steps[2]
    assert quiesce_command_parameters(workflow, 2, quiesce) == {}, (
        "a full reset: the window stops Fabric Manager and the restore starts it"
    )


def test_an_active_workload_is_stopped_and_restarted_around_it(
    opted_in: ApplicationContext,
) -> None:
    context = opted_in
    _report(context, "NVIDIA H100 80GB HBM3", 3.0)

    _incident, workflow = _solo_xid45(
        context,
        event_id="solo-xid45-active",
        workload_state=WorkloadState.ACTIVE,
        affected_workload_ids=["training/job/job-a"],
    )

    operations = [step.operation for step in workflow.official_steps]
    assert operations[:4] == [
        Op.FREEZE_EVIDENCE,
        Op.MARK_UNSCHEDULABLE,
        Op.STOP_WORKLOADS,
        Op.QUIESCE_GPU_SERVICES,
    ]
    assert operations[-1] is Op.RESTART_WORKLOAD
    assert Op.RESTART_FABRIC_MANAGER not in operations


@pytest.mark.parametrize(
    ("model", "status"),
    [
        (None, None),
        ("NVIDIA L40S", 3.0),
        ("NVIDIA GB200", 3.0),
        ("NVIDIA GB300", 3.0),
        ("NVIDIA RTX PRO 6000 Blackwell Server Edition", 3.0),
        ("NVIDIA H100 PCIe", 3.0),
        ("NVIDIA H100 80GB HBM3", 0.0),
    ],
    ids=[
        "no-field-170",
        "no-nvswitch",
        "nvl72",
        "gb300-nvl72",
        "blackwell-workstation",
        "pcie",
        "not-supported-single-gpu",
    ],
)
def test_every_other_node_keeps_the_fabric_manager_restart(
    context: ApplicationContext, model, status
) -> None:
    if model is not None:
        _report(context, model, status)

    _incident, workflow = _solo_xid45(context)

    assert workflow.status is WorkflowStatus.PENDING, workflow.blocked_reasons
    assert [step.operation for step in workflow.official_steps] == [
        Op.FREEZE_EVIDENCE,
        Op.RESTART_FABRIC_MANAGER,
    ]


def test_an_xid45_that_follows_a_companion_is_not_a_fabric_manager_restart(
    context: ApplicationContext,
) -> None:
    _report(context, "NVIDIA H200", 3.0)
    companion = event(95, event_id="companion-95")
    xid_event = event(45, event_id="not-solo-xid45")
    decision = context.policy.evaluate_xid(xid_event, companion_events=[companion])

    _incident, workflow = context.orchestrator.ingest(xid_event, decision)

    assert decision.official_action == "RESET_GPU"
    assert Op.RESET_ALL_GPUS_NVSWITCHES not in {
        step.operation for step in workflow.official_steps
    }


def test_the_graded_inventory_is_the_gpus_that_report_a_fabric_state() -> None:
    def sample(uuid: str | None, value: float, model: str = "NVIDIA H200"):
        return (
            GpuMetricSample(
                metric_name="DCGM_FI_DEV_FABRIC_MANAGER_STATUS",
                canonical_name=FABRIC_MANAGER_STATUS,
                value=value,
                gpu_uuid=uuid,
                labels={"modelName": model},
            ),
            None,
        )

    assert graded_fabric_gpus([sample("GPU-b", 3.0), sample("GPU-a", 2.0)]) == [
        "GPU-a",
        "GPU-b",
    ]
    assert graded_fabric_gpus([sample("GPU-a", 0.0)]) == [], "not supported"
    assert graded_fabric_gpus([sample("GPU-a", 3.0, "NVIDIA L40S")]) == []
    assert graded_fabric_gpus([sample(None, 3.0)]) == [], "no UUID to reset by"


#: The successor every reboot rung plans (``escalation._escalation_operations``):
#: it cordons and quarantines before it reboots, then validates and readmits.
REBOOT_SUCCESSOR = [
    Op.FREEZE_EVIDENCE,
    Op.MARK_UNSCHEDULABLE,
    Op.QUARANTINE,
    Op.RESTART_NODE,
    Op.VALIDATE_GPU,
    Op.VALIDATE_HOST,
    Op.VALIDATE_FABRIC,
    Op.RESTORE_SCHEDULING,
]


def test_a_failed_solo_fabric_manager_restart_climbs_to_the_reboot(
    context: ApplicationContext,
) -> None:
    """The solo XID 45 plan is FREEZE_EVIDENCE -> RESTART_FABRIC_MANAGER: no
    cordon, no validation. Its Fabric Manager restart failing used to end the
    record FAILED and the incident ESCALATED with no successor and no mail.
    It now climbs to the reboot the way a failed reset does; the reboot
    successor cordons and quarantines the node before it reboots it."""
    from gpu_fault.execution import WorkflowStepOutcome
    from gpu_fault.models import WorkflowExecutionRequest
    from tests._builders import active_workflow_executor

    incident, workflow = _solo_xid45(context)
    planned = [step.operation for step in workflow.official_steps]
    assert planned == [Op.FREEZE_EVIDENCE, Op.RESTART_FABRIC_MANAGER]

    class _SimulatedRuntime:
        def supports(self, step) -> bool:
            return step.execution_owner == "simulated-runtime"

        def execute(self, step_context) -> WorkflowStepOutcome:
            if step_context.step.operation is Op.RESTART_FABRIC_MANAGER:
                return WorkflowStepOutcome.failed(
                    "node agent node-a: CalledProcessError: systemctl restart "
                    "nvidia-fabricmanager returned 1"
                )
            return WorkflowStepOutcome.succeeded(
                operation_id=step_context.idempotency_key
            )

    result = active_workflow_executor(
        context.store, [_SimulatedRuntime()], planned
    ).execute(
        workflow.request_id,
        WorkflowExecutionRequest(expected_fencing_token=workflow.fencing_token),
    )

    assert result.status is WorkflowStatus.FAILED
    failed = context.store.get_workflow(workflow.request_id)
    assert context.store.get_incident(incident.incident_id).state is (
        IncidentState.ESCALATED
    )

    escalation = context.orchestrator.escalate_failed_hardware_remediation(failed)

    assert escalation is not None, "a failed Fabric Manager restart climbs nothing"
    successor_incident, successor = escalation
    assert successor.request_id == f"workflow-reboot-after-{workflow.request_id}"
    assert successor_incident.reasons[0].startswith(
        "fabric_manager_restart remediation failed; failed operation or "
        "validation: RESTART_FABRIC_MANAGER; nodes: node-a"
    ), successor_incident.reasons
    assert [step.operation for step in successor.official_steps] == (
        REBOOT_SUCCESSOR
    ), "the reboot rung isolates the node before it reboots it"
    assert {node for step in successor.official_steps for node in step.node_ids} == {
        "node-a"
    }
    assert context.orchestrator.escalate_failed_hardware_remediation(failed) == (
        escalation
    ), "one reboot successor per failed restart"


def _failed_fabric_manager_restart(operations, details=None):
    from tests.orchestration._cov95_runtime_escalation import EscalationHarness

    return EscalationHarness(
        operations,
        failed_index=operations.index(Op.RESTART_FABRIC_MANAGER),
        details=details,
    )


def test_a_failed_fabric_manager_restart_in_a_held_chain_climbs_to_the_reboot() -> None:
    """Like a failed reset in a chain that cordoned first: the reboot rung,
    no longer the support hand-off of a step no rung answered."""
    from gpu_fault.models import RecoveryAction
    from gpu_fault.orchestration.escalation import HardwareEscalationService

    h = _failed_fabric_manager_restart(
        [
            Op.FREEZE_EVIDENCE,
            Op.MARK_UNSCHEDULABLE,
            Op.RESTART_FABRIC_MANAGER,
            Op.VALIDATE_GPU,
            Op.RESTORE_SCHEDULING,
        ]
    )

    classification = HardwareEscalationService.classify(h.workflow)

    assert classification is not None
    assert classification[:3] == (
        "fabric_manager_restart",
        RecoveryAction.REBOOT_NODE,
        Op.RESTART_NODE,
    )


@pytest.mark.parametrize(
    ("details", "stage", "why"),
    [
        (
            {"remote_status_source": "executor-rejected"},
            None,
            "refused before dispatch: a stale generation, a newer one owns it",
        ),
        (
            {
                "safety_rejection": True,
                "manual_confirmation_required": True,
                "node_action_not_started": True,
            },
            "node_action_refused",
            "the agent attests it refused before acting: nothing to reboot from",
        ),
        (
            {
                "gpu_client_kind": "compute",
                "gpu_client_holders": [{"gpu_uuid": "GPU-a", "pid": "4242"}],
            },
            "node_action_refused",
            "GPU clients are live on a node nothing stopped: never a reboot",
        ),
        (
            {"outcome_unknown": True, "manual_confirmation_required": True},
            "manual_confirmation_required",
            "the restart may still be running: an operator confirms it",
        ),
    ],
    ids=["executor-rejected", "agent-refused", "gpu-clients", "unknown-outcome"],
)
def test_a_failed_fabric_manager_restart_keeps_the_endings_that_never_reboot(
    details, stage, why
) -> None:
    from gpu_fault.orchestration.escalation import HardwareEscalationService

    h = _failed_fabric_manager_restart(
        [Op.FREEZE_EVIDENCE, Op.RESTART_FABRIC_MANAGER], details
    )

    classification = HardwareEscalationService.classify(h.workflow)
    result = h.escalate()

    assert (classification and classification[0]) == stage, why
    if stage is None:
        assert result is None, why
        return
    assert result is not None, why
    _incident, successor = result
    operations = [step.operation for step in successor.official_steps]
    assert Op.RESTART_NODE not in operations, why
    assert Op.ESCALATE_SUPPORT in operations, "the operator gets one hand-off"

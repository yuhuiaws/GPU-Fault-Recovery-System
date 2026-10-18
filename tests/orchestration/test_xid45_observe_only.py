"""A solo XID 45 is observe-only on a graded NVSwitch node unless the site opts in (batch 131).

NVIDIA's Xid catalog finalizes XID 45 alone as "Solo: RESTART_FM", and its
trigger is a user or system action ("Control-C, GPU resets, sigkill"); the GPU
debug guidelines read the solo event as informational except on A100 with
Fabric Manager in the GPU's OS. On a graded NVSwitch node (Hopper SXM, HGX
B200/B300) batch 128 compiles RESTART_FM into the full GPU + NVSwitch reset, so
by default the decision there is MONITOR_ONLY / NO_ACTION: the incident, its
evidence and the investigatory notification are recorded, nothing touches the
node, RESTART_FM stays the official action, and the reasons name the evidence
and the switch. ``GPU_FAULT_XID45_SOLO_NVSWITCH_RESTART_FM=true`` restores the
catalog chain there; every other node keeps the Fabric Manager restart, and a
non-solo XID 45 follows its companion either way.
"""

from __future__ import annotations

import asyncio
from datetime import timedelta
from typing import Any

import pytest

from gpu_fault.app import ApplicationContext, create_app
from gpu_fault.gpu_fabric_manager import FABRIC_MANAGER_STATUS
from gpu_fault.gpu_metrics import GpuMetricSample, GpuMetricSource, GpuMetricsService
from gpu_fault.models import (
    IncidentState,
    RecoveryAction,
    Severity,
    WorkflowOperation,
    WorkloadState,
)
from gpu_fault.orchestration.xid45_policy import (
    NVIDIA_GPU_DEBUG_XID45_QUOTE,
    NVIDIA_GPU_DEBUG_XID45_REFERENCE,
    NVIDIA_XID_CATALOG_REFERENCE,
    NVIDIA_XID_CATALOG_SPREADSHEET,
    OPT_IN_REASON,
    XID45_SOLO_RESTART_FM_ENV,
    observe_only,
    observe_only_reference,
)
from gpu_fault.policy import ActionDisposition, XidEvent
from tests._builders import asgi_client, build_store, gpu_metric_batch

from ._support import NOW

Op = WorkflowOperation
GPUS = [f"GPU-{index}" for index in range(8)]
FABRIC_CHAIN = [
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
FABRIC_MANAGER_RESTART = [Op.FREEZE_EVIDENCE, Op.RESTART_FABRIC_MANAGER]
GRADED = ["NVIDIA H200", "NVIDIA H100 80GB HBM3", "NVIDIA B200", "NVIDIA B300 SXM6 AC"]
NOT_GRADED = ["NVIDIA A100-SXM4-80GB", "NVIDIA GB200", "NVIDIA GB300"]


class ControlPlane(ApplicationContext):
    """A control plane built with the switch as given, and its app."""

    app: Any = None

    def post(self, event: XidEvent) -> dict:
        async def post() -> dict:
            async with asgi_client(self.app) as client:
                response = await client.post(
                    "/v1/gpu-events/xid", json=event.model_dump(mode="json")
                )
            assert response.status_code == 200, response.text
            return response.json()

        return asyncio.run(post())


def _context(monkeypatch: pytest.MonkeyPatch, switch: str | None) -> ControlPlane:
    if switch is None:
        monkeypatch.delenv(XID45_SOLO_RESTART_FM_ENV, raising=False)
    else:
        monkeypatch.setenv(XID45_SOLO_RESTART_FM_ENV, switch)
    context = ControlPlane(store=build_store())
    context.app = create_app(context)
    return context


def _report(context: ControlPlane, model: str | None, status: float = 3.0) -> None:
    if model is None:
        return
    GpuMetricsService(store=context.store).ingest(
        gpu_metric_batch(
            "dcgm-fabric",
            NOW,
            GpuMetricSource.DCGM_EXPORTER,
            [
                GpuMetricSample(
                    metric_name="DCGM_FI_DEV_FABRIC_MANAGER_STATUS",
                    canonical_name=FABRIC_MANAGER_STATUS,
                    value=status,
                    gpu_uuid=uuid,
                    labels={"modelName": model},
                )
                for uuid in GPUS
            ],
            runtime_profile_version="simulated-v1",
            workload_state=WorkloadState.IDLE,
        )
    )


def _xid(code: int, event_id: str, product: str, *, seconds: int = 0) -> XidEvent:
    return XidEvent(
        event_id=event_id,
        cluster_id="cluster-a",
        node_id="node-a",
        observed_at=NOW + timedelta(seconds=seconds),
        xid=code,
        gpu_uuid="GPU-0",
        pci_bdf="0000:01:00.0",
        product=product,
        driver_branch=580,
        cuda_version="13.0",
        runtime_profile_version="simulated-v1",
        evidence_ref="kmsg://node-a/boot-1/45",
    )


def _finalize_solo(context: ControlPlane, product: str):
    """A solo XID 45 through the product's own path: posted, its companion
    window closed, finalized by the correlation coordinator."""

    clock = [NOW]
    context.xid_correlation.now = lambda: clock[0]
    pending = context.post(_xid(45, "solo-xid45", product))
    assert pending["disposition"] == "PENDING_CORRELATION", pending
    clock[0] += timedelta(seconds=31)
    assert context.xid_correlation.run_once() == 1, "the window closes once"
    decision = context.store.get_xid_policy_decision("solo-xid45")
    incident = context.store.get_incident_by_event("solo-xid45")
    workflow = (
        context.store.get_workflow(incident.workflow_request_id)
        if incident is not None and incident.workflow_request_id
        else None
    )
    return decision, incident, workflow


@pytest.mark.parametrize("product", GRADED)
@pytest.mark.parametrize("switch", [None, "false", "0"], ids=["unset", "false", "0"])
def test_a_solo_xid45_on_a_graded_node_is_observe_only_by_default(
    monkeypatch: pytest.MonkeyPatch, product: str, switch: str | None
) -> None:
    context = _context(monkeypatch, switch)
    _report(context, product)

    decision, incident, workflow = _finalize_solo(context, product)

    assert decision is not None and incident is not None, (decision, incident)
    assert decision.official_action == "RESTART_FM", "the catalog action is recorded"
    assert decision.disposition is ActionDisposition.MONITOR_ONLY, decision
    assert decision.action is RecoveryAction.NO_ACTION
    assert decision.safety_action is None and decision.pre_actions == []
    assert decision.marker.severity is Severity.INFO
    assert decision.marker.active is False, "an INFO marker, no containment"
    assert workflow is None and incident.workflow_request_id is None, (
        "no node action, no cordon: nothing is planned"
    )
    assert incident.state is IncidentState.RECOVERED, incident.state
    assert incident.official_action == "RESTART_FM"
    assert incident.effective_action is RecoveryAction.NO_ACTION
    assert incident.policy_reference == NVIDIA_GPU_DEBUG_XID45_REFERENCE, (
        "the incident carries the evidence URL"
    )
    why = [item for item in incident.reasons if item.startswith("observe-only")]
    assert len(why) == 1 and "8 GPUs report an NVLink fabric state" in why[0], why
    assert NVIDIA_GPU_DEBUG_XID45_REFERENCE in why[0], "the reason names the source"
    assert OPT_IN_REASON in incident.reasons, "and how to opt in"
    assert XID45_SOLO_RESTART_FM_ENV in OPT_IN_REASON
    assert context.store.list_workflows(limit=10) == [], "nothing anywhere"
    notifications = [
        item
        for item in context.store.list_notifications()
        if item.incident_id == incident.incident_id
    ]
    assert len(notifications) == 1, "the existing advisory path"
    body = notifications[0].body_text
    assert "Disposition: MONITOR_ONLY" in body, body
    assert "NVIDIA immediate action: RESTART_FM" in body, body
    assert "Effective action: NO_ACTION" in body, body
    assert XID45_SOLO_RESTART_FM_ENV in body, "the administrator reads the switch"


@pytest.mark.parametrize("product", GRADED)
@pytest.mark.parametrize("switch", ["true", "1", "yes"])
def test_the_site_switch_restores_the_catalog_fabric_chain(
    monkeypatch: pytest.MonkeyPatch, product: str, switch: str
) -> None:
    context = _context(monkeypatch, switch)
    _report(context, product)

    decision, incident, workflow = _finalize_solo(context, product)

    assert decision.disposition is ActionDisposition.EXECUTABLE, decision
    assert incident is not None and incident.policy_reference is None
    assert not any(item.startswith("observe-only") for item in incident.reasons)
    assert workflow is not None, "the catalog's RESTART_FM is compiled"
    assert workflow.official_action == "RESTART_FM"
    assert [step.operation for step in workflow.official_steps] == FABRIC_CHAIN, (
        "batch 128's full GPU + NVSwitch reset on the graded GPUs"
    )
    assert workflow.official_steps[4].gpu_uuids == GPUS


@pytest.mark.parametrize("product", NOT_GRADED)
@pytest.mark.parametrize("switch", [None, "true"], ids=["default", "opted-in"])
def test_every_other_node_keeps_the_fabric_manager_restart(
    monkeypatch: pytest.MonkeyPatch, product: str, switch: str | None
) -> None:
    """A100 is NVIDIA's own solo case; GB200/GB300 are not graded."""

    context = _context(monkeypatch, switch)
    _report(context, product)

    decision, incident, workflow = _finalize_solo(context, product)

    assert decision.disposition is ActionDisposition.EXECUTABLE, decision
    assert incident is not None and incident.policy_reference is None
    assert workflow is not None
    assert [step.operation for step in workflow.official_steps] == (
        FABRIC_MANAGER_RESTART
    ), product


@pytest.mark.parametrize("status", [None, 0.0], ids=["no-field-170", "not-supported"])
def test_a_hopper_node_the_product_does_not_grade_keeps_the_restart(
    monkeypatch: pytest.MonkeyPatch, status: float | None
) -> None:
    context = _context(monkeypatch, None)
    if status is not None:
        _report(context, "NVIDIA H200", status)

    _decision, _incident, workflow = _finalize_solo(context, "NVIDIA H200")

    assert workflow is not None
    assert [step.operation for step in workflow.official_steps] == (
        FABRIC_MANAGER_RESTART
    ), "no graded fabric, no observe-only rewrite"


@pytest.mark.parametrize("switch", [None, "true"], ids=["default", "opted-in"])
def test_a_non_solo_xid45_follows_its_companion_unchanged(
    monkeypatch: pytest.MonkeyPatch, switch: str | None
) -> None:
    context = _context(monkeypatch, switch)
    _report(context, "NVIDIA H200")
    clock = [NOW]
    context.xid_correlation.now = lambda: clock[0]

    context.post(_xid(45, "xid45", "NVIDIA H200"))
    companion = context.post(_xid(79, "xid79", "NVIDIA H200", seconds=5))
    clock[0] += timedelta(seconds=31)
    assert context.xid_correlation.run_once() == 1

    finalized = context.store.get_xid_policy_decision("xid45")
    assert finalized.action is RecoveryAction.REBOOT_NODE, finalized
    assert finalized.correlated_event_id == "xid79"
    assert finalized.incident_id == companion["incident_id"]
    assert not any(item.startswith("observe-only") for item in finalized.reasons)


def test_the_rewrite_is_idempotent_and_guards_a_direct_ingest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``ingest`` applies it too: a caller that skips the fence plans nothing."""

    context = _context(monkeypatch, None)
    _report(context, "NVIDIA B200")
    event = _xid(45, "direct-xid45", "NVIDIA B200")
    raw = context.policy.evaluate_xid(event, xid45_window_closed=True)
    assert raw.disposition is ActionDisposition.EXECUTABLE

    incident, workflow = context.orchestrator.ingest(event, raw)

    assert workflow is None and incident.state is IncidentState.RECOVERED
    assert incident.policy_reference == NVIDIA_GPU_DEBUG_XID45_REFERENCE
    once = observe_only(raw, GPUS)
    assert observe_only(once, GPUS).reasons == once.reasons, "no duplicate reasons"
    assert observe_only_reference(once) == NVIDIA_GPU_DEBUG_XID45_REFERENCE
    assert observe_only_reference(raw) is None, "only the rewrite carries it"


def test_the_policy_module_records_nvidia_s_evidence_verbatim() -> None:
    assert NVIDIA_GPU_DEBUG_XID45_REFERENCE == (
        "https://docs.nvidia.com/deploy/gpu-debug-guidelines/gpu-node-triage.html"
        "#understanding-xid-messages"
    )
    assert NVIDIA_XID_CATALOG_REFERENCE == (
        "https://docs.nvidia.com/deploy/xid-errors/analyzing-xid-catalog.html"
    )
    assert NVIDIA_XID_CATALOG_SPREADSHEET.endswith(
        "/4586dadb59119a55d1e93a181caa4272/Xid-Catalog.xlsx"
    ), NVIDIA_XID_CATALOG_SPREADSHEET
    assert NVIDIA_GPU_DEBUG_XID45_QUOTE == (
        "No action, informative only. Indicates channels affected by another "
        "failure. On A100, this error could be seen by itself due to unexpected "
        "Fabric Manager shutdown when FM is running in the same OS environment "
        "as the GPU. Otherwise, this error is safe to ignore as an informational "
        "message."
    )

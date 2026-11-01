"""A stale-boot fault line is recorded under its own incident (batch 236).

``reboot_window.cover_fault`` closes a kernel-log event whose boot the node has
left as a RECOVERED evidence incident. It used to build that incident from the
marker the provider-event correlation had just associated with a live incident,
so the evidence record took the live incident's id and overwrote it: the live
incident became RECOVERED, its event link moved to the stale line and its
workflow pointer was lost while the workflow itself kept running. Association
is evidence only; the stale line now keeps its own identity and names the live
incident in a reason.
"""

from __future__ import annotations

from datetime import timedelta

from gpu_fault.app import ApplicationContext
from gpu_fault.models import IncidentState
from gpu_fault.orchestration.reboot_window import STALE_BOOT_REASON
from gpu_fault.policy import FaultPolicyDecision, XidEvent
from tests._builders import build_context
from tests.orchestration._support import NOW, _active_agent

GPU = "GPU-h200-0"
EXPORTER_PCI = "00000000:B9:00.0"


def _xid48(
    event_id: str, *, source: str, pci_bdf: str, boot_id: str | None, seconds: int
) -> XidEvent:
    at = NOW + timedelta(seconds=seconds)
    return XidEvent(
        event_id=event_id,
        cluster_id="cluster-a",
        node_id="node-a",
        observed_at=at,
        source_event_time=at,
        source_boot_id=boot_id,
        event_source=source,
        xid=48,
        gpu_uuid=GPU,
        pci_bdf=pci_bdf,
        product="NVIDIA H200",
        driver_branch=580,
        cuda_version="13.0",
        runtime_profile_version="simulated-v1",
    )


def _correlate(context: ApplicationContext, event: XidEvent) -> FaultPolicyDecision:
    return context.orchestrator.correlate_provider_event(
        context.policy.evaluate_xid(event), cluster_id="cluster-a"
    )


def test_a_stale_boot_line_never_overwrites_the_incident_its_marker_matched() -> None:
    context = build_context()
    store = context.store
    store.save_agent(_active_agent(boot_id="boot-new"))
    # The live report: DCGM field 230 of the current boot (the gauge has no boot id).
    live = _xid48(
        "gauge-xid48",
        source="DCGM_EXPORTER",
        pci_bdf=EXPORTER_PCI,
        boot_id=None,
        seconds=-40,
    )
    decision = _correlate(context, live)
    live_incident, live_workflow = context.orchestrator.ingest(live, decision)
    assert live_workflow is not None, "the live XID 48 plans its GPU reset"
    store.add_marker(
        decision.marker.model_copy(update={"incident_id": live_incident.incident_id})
    )
    before = store.get_incident(live_incident.incident_id)
    # The kernel line of the previous boot, read again after the node came back.
    stale = _xid48(
        "kernel-xid48-old-boot",
        source="KERNEL_LOG",
        pci_bdf=EXPORTER_PCI,
        boot_id="boot-old",
        seconds=-60,
    )
    associated = _correlate(context, stale)
    assert associated.duplicate, "the precondition: the stale line matches the marker"
    assert associated.marker.incident_id == live_incident.incident_id

    incident, workflow = context.orchestrator.ingest(stale, associated)

    assert store.get_incident(live_incident.incident_id) == before, (
        "the stale line's evidence record overwrote the live incident"
    )
    assert store.get_workflow(live_workflow.request_id) == live_workflow
    assert workflow is None, "a line from a boot the node has left plans nothing"
    assert incident.incident_id == f"inc-{stale.event_id}", incident.incident_id
    assert incident.event_id == stale.event_id
    assert incident.state is IncidentState.RECOVERED
    assert STALE_BOOT_REASON in incident.reasons, incident.reasons
    assert any(live_incident.incident_id in reason for reason in incident.reasons), (
        "the association stays visible as evidence on the stale record"
    )
    assert store.get_incident_by_event(stale.event_id) == incident
    assert store.get_incident_by_event(live.event_id) == before

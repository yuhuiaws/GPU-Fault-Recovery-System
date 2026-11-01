"""An NVLink5 XID read from DCGM field 230 joins the kernel-sourced incident.

DCGM field 230 (``DCGM_FI_DEV_XID_ERRORS``) keeps only the last XID per GPU:
when it changes, ``GpuMetricsService._xid_event`` turns the change into an
``XidEvent`` with ``event_source=DCGM_EXPORTER`` and no kernel line. For XIDs
144-150 the policy engine decodes the IntrInfo / Error Status registers the
kernel line carries (``policy.engine._workflow_nvlink5``), so a gauge-only
event can never decode: on its own it is BLOCKED_MISSING_EVIDENCE. The kernel
line of the same fault reaches the control plane first (the exporter refreshes
on its own 15 s schedule); the gauge change must join that incident -- the
Blackwell audit's untested §5.2 assertion -- and open no second, BLOCKED one.

Synthetic fixtures derived from NVIDIA documentation: the product strings are
the driver's B-series names, the kernel line has the 2-register GSP form of
driver 580 (``nvlink5-xid-payload-formats``), and its registers are a V2
(driver >= 575) RLW_RXPIPE pattern of the pinned XID 145 decode table.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import pytest

from gpu_fault.app import default_simulated_profile
from gpu_fault.channel_registry import GPU_METRICS_PATH, NVIDIA_KERNEL_PATH
from gpu_fault.gpu_metrics import GpuMetricBatch, GpuMetricSample, GpuMetricSource
from gpu_fault.models import IncidentState
from gpu_fault.policy import XidEvent
from gpu_fault.policy.nvlink5_gauge import candidate_window, nvlink5_gauge_twin
from gpu_fault.store import InMemoryStore
from tests._builders import asgi_client, build_context

NOW = datetime(2026, 11, 1, 12, tzinfo=timezone.utc)
CLUSTER = "cluster-a"
NODE = "node-a"
GPU = "GPU-b200-0"
#: The exporter's ``pci_bus_id`` label and the kernel line's ``PCI:`` token of
#: the same device.
EXPORTER_BDF = "00000000:B9:00.0"
KERNEL_PCI = "0000:b9:00"
#: (XID, kernel line): 145 decodes under V2 to RLW_RXPIPE (RESET_GPU); 149 is
#: the captured pre-R575 NETIR_LINK_EVT shape, which matches nothing under V2
#: -- the kernel incident is BLOCKED itself and the gauge must still join it.
KERNEL_LINES = {
    145: (
        f"NVRM: Xid (PCI:{KERNEL_PCI}): 145, RLW_RXPIPE Fatal XC0 i0 Link 03 "
        "(0x00000086 0x00000001)"
    ),
    149: (
        f"NVRM: Xid (PCI:{KERNEL_PCI}): 149, NETIR_LINK_EVT Fatal XC0 i0 Link 08 "
        "(0x004505c6 0x00000000)"
    ),
}


def _batch(batch_id: str, seconds: int, xid: float) -> GpuMetricBatch:
    labels = {"gpu": "0", "UUID": GPU, "pci_bus_id": EXPORTER_BDF}
    return GpuMetricBatch(
        batch_id=batch_id,
        cluster_id=CLUSTER,
        node_id=NODE,
        observed_at=NOW + timedelta(seconds=seconds),
        source=GpuMetricSource.DCGM_EXPORTER,
        product="NVIDIA B200",
        driver_branch=580,
        cuda_version="13.0",
        runtime_profile_version="simulated-v1",
        samples=[
            GpuMetricSample(
                metric_name="DCGM_FI_DEV_XID_ERRORS",
                canonical_name="xid_last_error",
                value=xid,
                gpu_index="0",
                gpu_uuid=GPU,
                pci_bdf=EXPORTER_BDF,
                labels=dict(labels),
            ),
            GpuMetricSample(
                metric_name="DCGM_FI_DEV_GPU_TEMP",
                canonical_name="gpu_temperature_c",
                value=40,
                gpu_index="0",
                gpu_uuid=GPU,
                pci_bdf=EXPORTER_BDF,
                labels=dict(labels),
            ),
        ],
    )


def _kernel(xid: int, seconds: int) -> dict[str, object]:
    return {
        "cluster_id": CLUSTER,
        "node_id": NODE,
        "record_id": f"kmsg-xid{xid}",
        "observed_at": (NOW + timedelta(seconds=seconds)).isoformat(),
        "message": KERNEL_LINES[xid],
        "product": "NVIDIA B200",
        "driver_branch": 580,
        "cuda_version": "13.0",
        "runtime_profile_version": "simulated-v1",
    }


def _replay(xid: int) -> tuple[object, list[dict[str, object]]]:
    store = InMemoryStore()
    store.save_profile(default_simulated_profile())
    context = build_context(store=store)
    bodies: list[dict[str, object]] = []

    async def scenario() -> None:
        async with asgi_client(context) as client:
            steps = (
                (GPU_METRICS_PATH, _batch("baseline", 0, 0).model_dump(mode="json")),
                (NVIDIA_KERNEL_PATH, _kernel(xid, 10)),
                (GPU_METRICS_PATH, _batch("gauge", 30, xid).model_dump(mode="json")),
            )
            for path, payload in steps:
                response = await client.post(path, json=payload)
                assert response.status_code == 200, response.text
                bodies.append(response.json())

    asyncio.run(scenario())
    return context, bodies


@pytest.mark.parametrize("xid", sorted(KERNEL_LINES))
def test_a_field_230_nvlink5_xid_joins_the_kernel_sourced_incident(xid: int) -> None:
    context, _bodies = _replay(xid)
    store = context.store

    kernel_events = [
        event
        for event in store.list_xid_events(CLUSTER, NODE)
        if event.event_source == "KERNEL_LOG"
    ]
    gauge_events = [
        event
        for event in store.list_xid_events(CLUSTER, NODE)
        if event.event_source == "DCGM_EXPORTER"
    ]
    assert len(kernel_events) == 1 and len(gauge_events) == 1, (
        "the replay must produce one kernel event and one field-230 event: "
        f"{[(event.event_id, event.event_source) for event in store.list_xid_events(CLUSTER, NODE)]}"
    )
    (kernel,) = kernel_events
    (gauge,) = gauge_events
    assert gauge.xid == xid
    kernel_incident = store.get_incident_by_event(kernel.event_id)
    assert kernel_incident is not None, "the kernel line opens the incident"

    gauge_decision = store.get_xid_policy_decision(gauge.event_id)
    assert gauge_decision is not None, "the gauge change is decided and stored"
    assert gauge_decision.duplicate, (
        "the field-230 change must be booked as a duplicate of the kernel "
        f"incident, not a fault of its own: {gauge_decision}"
    )
    assert gauge_decision.incident_id == kernel_incident.incident_id, (
        gauge_decision.incident_id,
        kernel_incident.incident_id,
    )
    incidents = [
        incident
        for incident in store.list_incidents_by_state(
            CLUSTER, set(IncidentState), node_ids={NODE}
        )
    ]
    assert store.get_incident_by_event(gauge.event_id) == store.get_incident(
        kernel_incident.incident_id
    ), "the gauge event is linked to the kernel incident"
    assert [incident.incident_id for incident in incidents] == [
        kernel_incident.incident_id
    ], (
        "no second incident -- BLOCKED or otherwise -- for the gauge change: "
        f"{[(item.incident_id, item.state.value, item.event_source) for item in incidents]}"
    )


def _replay_steps(steps) -> object:
    store = InMemoryStore()
    store.save_profile(default_simulated_profile())
    context = build_context(store=store)

    async def scenario() -> None:
        async with asgi_client(context) as client:
            for path, payload in steps:
                response = await client.post(path, json=payload)
                assert response.status_code == 200, response.text

    asyncio.run(scenario())
    return context


def _gauge_incident(context, xid: int):
    (gauge,) = [
        event
        for event in context.store.list_xid_events(CLUSTER, NODE)
        if event.event_source == "DCGM_EXPORTER"
    ]
    return context.store.get_xid_policy_decision(gauge.event_id)


def test_a_gauge_change_without_its_kernel_line_stays_fail_closed() -> None:
    """A lost kernel line leaves the gauge the only evidence: BLOCKED as before."""

    context = _replay_steps(
        (
            (GPU_METRICS_PATH, _batch("baseline", 0, 0).model_dump(mode="json")),
            (GPU_METRICS_PATH, _batch("gauge", 30, 149).model_dump(mode="json")),
        )
    )
    decision = _gauge_incident(context, 149)

    assert decision.disposition.value == "BLOCKED_MISSING_EVIDENCE"
    assert not decision.duplicate and decision.correlated_event_id is None
    assert decision.incident_id == "inc-gauge-GPU-b200-0-xid-149"


def test_a_kernel_line_older_than_the_twin_window_is_not_joined() -> None:
    context = _replay_steps(
        (
            (GPU_METRICS_PATH, _batch("baseline", 0, 0).model_dump(mode="json")),
            (NVIDIA_KERNEL_PATH, _kernel(149, 10)),
            (GPU_METRICS_PATH, _batch("gauge", 10 + 121, 149).model_dump(mode="json")),
        )
    )

    assert _gauge_incident(context, 149).correlated_event_id is None


def test_only_the_same_xid_on_the_same_gpu_is_a_twin() -> None:
    gauge = XidEvent(
        event_id="gauge",
        cluster_id=CLUSTER,
        node_id=NODE,
        observed_at=NOW + timedelta(seconds=30),
        event_source="DCGM_EXPORTER",
        xid=149,
        gpu_uuid=GPU,
        pci_bdf=EXPORTER_BDF,
    )
    kernel = gauge.model_copy(
        update={
            "event_id": "kernel",
            "event_source": "KERNEL_LOG",
            "observed_at": NOW,
            "pci_bdf": KERNEL_PCI,
        }
    )

    assert nvlink5_gauge_twin(gauge, [kernel]) == kernel
    assert nvlink5_gauge_twin(gauge, [kernel.model_copy(update={"xid": 145})]) is None
    assert (
        nvlink5_gauge_twin(gauge, [kernel.model_copy(update={"gpu_uuid": "GPU-other"})])
        is None
    )
    assert nvlink5_gauge_twin(
        gauge, [kernel.model_copy(update={"gpu_uuid": None})]
    ) == kernel.model_copy(update={"gpu_uuid": None}), (
        "a kernel line without a UUID still names the device by its PCI address"
    )
    assert nvlink5_gauge_twin(kernel, [gauge]) is None, "only a gauge has a twin"
    assert nvlink5_gauge_twin(gauge.model_copy(update={"xid": 79}), [kernel]) is None


def test_the_candidate_window_widens_only_for_an_nvlink5_gauge() -> None:
    companion = timedelta(seconds=30)
    gauge = XidEvent(
        event_id="gauge",
        cluster_id=CLUSTER,
        node_id=NODE,
        observed_at=NOW,
        event_source="DCGM_EXPORTER",
        xid=149,
    )

    assert candidate_window(gauge, companion) == timedelta(seconds=120)
    for other in (
        gauge.model_copy(update={"xid": 79}),
        gauge.model_copy(update={"event_source": "KERNEL_LOG"}),
    ):
        assert candidate_window(other, companion) == companion, (
            "every other event keeps the companion window"
        )

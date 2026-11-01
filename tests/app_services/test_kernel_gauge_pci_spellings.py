"""The kernel line and DCGM field 230 of one H200 XID meet whatever their spelling.

A real XID reaches the control plane twice: the kernel collector posts the NVRM
line at once (``NVRM: Xid (PCI:0000:b9:00): <xid>, ...``) and DCGM field 230
(``DCGM_FI_DEV_XID_ERRORS``, shipped by ``deploy/dataplane/dcgm-counters.csv``)
changes on the exporter's next refresh, labelled ``pci_bus_id="00000000:B9:00.0"``
-- the same GPU spelled two ways. Before batch 236 the provider-event
correlation intersected the raw strings, so the second report was never booked
as a duplicate of the first, and when the store merged it into the first
report's incident its decision named an incident that was never written.

Each scenario runs twice, once with the sources' own spellings and once with
the kernel line spelled exactly like the exporter. Spelling may change nothing:
the association is identical, and so is what executes -- marker association
never merges or splits executions, the store's arbitration does.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import Iterable
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from gpu_fault.app import ApplicationContext, default_simulated_profile
from gpu_fault.channel_registry import GPU_METRICS_PATH, NVIDIA_KERNEL_PATH
from gpu_fault.gpu_metrics import GpuMetricBatch, GpuMetricSample, GpuMetricSource
from gpu_fault.models import IncidentState, WorkflowStatus
from gpu_fault.store import InMemoryStore, NotFoundError
from tests._builders import asgi_client, build_context

NOW = datetime(2026, 11, 1, 12, tzinfo=timezone.utc)
CLUSTER = "cluster-a"
NODE = "node-a"
GPU = "GPU-h200-0"
EXPORTER_PCI = "00000000:B9:00.0"
KERNEL_PCI = "0000:b9:00"
#: XID 48 resets the GPU (the idle node's two reports share one workflow); XID 13
#: restarts the workload (each report keeps its own incident).
KERNEL_TEXT = {
    48: "pid=1234, name=python, An uncorrectable double bit error was detected",
    13: "pid=1234, name=python, Graphics Exception: ESR 0x404600=0x80000002",
}
ASSOCIATION = re.compile(r"Provider marker association with incident (\S+);")


def _batch(batch_id: str, seconds: int, xid: float) -> dict[str, Any]:
    labels = {"gpu": "0", "UUID": GPU, "pci_bus_id": EXPORTER_PCI}
    samples = [
        GpuMetricSample(
            metric_name=metric,
            canonical_name=canonical,
            value=value,
            gpu_index="0",
            gpu_uuid=GPU,
            pci_bdf=EXPORTER_PCI,
            labels=dict(labels),
        )
        for metric, canonical, value in (
            ("DCGM_FI_DEV_XID_ERRORS", "xid_last_error", xid),
            ("DCGM_FI_DEV_GPU_TEMP", "gpu_temperature_c", 40),
        )
    ]
    return GpuMetricBatch(
        batch_id=batch_id,
        cluster_id=CLUSTER,
        node_id=NODE,
        observed_at=NOW + timedelta(seconds=seconds),
        source=GpuMetricSource.DCGM_EXPORTER,
        product="NVIDIA H200",
        driver_branch=580,
        cuda_version="13.0",
        runtime_profile_version="simulated-v1",
        samples=samples,
    ).model_dump(mode="json")


def _kernel(xid: int, seconds: int, pci: str) -> dict[str, Any]:
    return {
        "cluster_id": CLUSTER,
        "node_id": NODE,
        "record_id": f"kmsg-xid{xid}",
        "observed_at": (NOW + timedelta(seconds=seconds)).isoformat(),
        "source_boot_id": "boot-a",
        "message": f"NVRM: Xid (PCI:{pci}): {xid}, {KERNEL_TEXT[xid]}",
        "product": "NVIDIA H200",
        "driver_branch": 580,
        "cuda_version": "13.0",
        "runtime_profile_version": "simulated-v1",
    }


def _replay(xid: int, *, kernel_pci: str, kernel_first: bool) -> ApplicationContext:
    store = InMemoryStore()
    store.save_profile(default_simulated_profile())
    context = build_context(store=store)
    kernel_at, gauge_at = (10, 30) if kernel_first else (30, 10)
    kernel = (NVIDIA_KERNEL_PATH, _kernel(xid, kernel_at, kernel_pci))
    gauge = (GPU_METRICS_PATH, _batch("gauge", gauge_at, xid))
    steps = [
        (GPU_METRICS_PATH, _batch("baseline", 0, 0)),
        *((kernel, gauge) if kernel_first else (gauge, kernel)),
    ]

    async def scenario() -> None:
        async with asgi_client(context) as client:
            for path, body in steps:
                response = await client.post(path, json=body)
                assert response.status_code == 200, response.text

    asyncio.run(scenario())
    return context


def _reports(context: ApplicationContext) -> dict[str, Any]:
    """``event_source -> (event, decision, incident)`` of the two reports."""

    store = context.store
    reports = {
        event.event_source: (
            event,
            store.get_xid_policy_decision(event.event_id),
            store.get_incident_by_event(event.event_id),
        )
        for event in store.list_xid_events(CLUSTER, NODE)
    }
    assert sorted(reports) == ["DCGM_EXPORTER", "KERNEL_LOG"], sorted(reports)
    return reports


def _execution(context: ApplicationContext) -> tuple[Any, ...]:
    """What runs, independent of event ids and spellings."""

    store = context.store
    reports = _reports(context)
    incidents = store.list_incidents_by_state(
        CLUSTER, set(IncidentState), node_ids={NODE}
    )
    return (
        sorted(incident.state.value for incident in incidents),
        reports["KERNEL_LOG"][2].incident_id == reports["DCGM_EXPORTER"][2].incident_id,
        sorted(
            (
                workflow.status.value,
                tuple(step.operation.value for step in workflow.official_steps),
            )
            for workflow in store.list_workflows(set(WorkflowStatus), limit=50)
        ),
    )


def _associated(reasons: Iterable[str]) -> set[str]:
    return {match.group(1) for match in map(ASSOCIATION.search, reasons) if match}


def _written(context: ApplicationContext, incident_id: str) -> bool:
    try:
        context.store.get_incident(incident_id)
    except NotFoundError:
        return False
    return True


@pytest.mark.parametrize(
    "kernel_first", [True, False], ids=["kernel-first", "gauge-first"]
)
@pytest.mark.parametrize("xid", sorted(KERNEL_TEXT))
def test_the_second_report_of_one_xid_is_booked_against_the_first(
    xid: int, kernel_first: bool
) -> None:
    real = _replay(xid, kernel_pci=KERNEL_PCI, kernel_first=kernel_first)
    spelled_alike = _replay(xid, kernel_pci=EXPORTER_PCI, kernel_first=kernel_first)
    first, second = (
        ("KERNEL_LOG", "DCGM_EXPORTER")
        if kernel_first
        else ("DCGM_EXPORTER", "KERNEL_LOG")
    )

    reports = _reports(real)
    first_incident = reports[first][2]
    _, decision, incident = reports[second]
    associated = _associated([*decision.reasons, *incident.reasons])

    assert reports["KERNEL_LOG"][0].pci_bdf == KERNEL_PCI, "the kernel's own spelling"
    assert decision.duplicate, (
        f"the {second} report of XID {xid} is the {first} report's fault: "
        f"{decision.reasons}"
    )
    assert [item for item in associated if not _written(real, item)] == [], (
        f"the {second} report names an incident that was never written: "
        f"{decision.reasons}"
    )
    assert (
        incident.incident_id == first_incident.incident_id
        or first_incident.incident_id in associated
    ), "the second report's incident, or its reasons, name the first report's"
    assert _execution(real) == _execution(spelled_alike), (
        "the PCI spelling changed what executes"
    )
    assert _reports(spelled_alike)[second][1].duplicate, "the control run associates"

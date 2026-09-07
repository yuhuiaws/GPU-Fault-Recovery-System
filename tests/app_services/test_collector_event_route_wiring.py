"""The collector-event routes hand two signals on that used to stop at the door.

G7: a kernel or Fabric Manager line that names an ``Xid``/``SXid`` but carries
no readable code is normalized into an *unresolved* provider signal. The
normalizer and ``FaultIngestionService.ingest_unresolved_signals`` already turn
that into a WARNING finding and a counter; these cases pin that the HTTP routes
actually call the service, so a catalog drift is seen through the real ingress
and not only in a unit test that invokes the service by hand.

G4: the kernel collector's periodic health summary carries ``<name>:<count>``
tokens for delivery failures, kmsg overflows and boot-time re-estimates. The
route used to record every summary as a success; now the tokens land as
collector errors so ``last_error_at`` is set and the erroring-nodes gauge can
see a kernel collector that is dropping lines. An all-zero summary still only
refreshes ``last_success_at``.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import httpx

from gpu_fault.app import ApplicationContext
from gpu_fault.channel_registry import (
    COLLECTOR_HEALTH_PATH,
    FABRIC_MANAGER_PATH,
    NVIDIA_KERNEL_PATH,
)
from gpu_fault.hma import (
    UNPARSED_SXID_REASON,
    UNPARSED_XID_REASON,
    FabricManagerLogEvent,
    NvidiaKernelLogEvent,
)
from gpu_fault.telemetry import CollectorHealthSummary, CollectorKind
from tests._builders import asgi_client, build_context

NOW = datetime(2026, 9, 7, 10, 0, tzinfo=timezone.utc)


def _post(context: ApplicationContext, path: str, payload) -> httpx.Response:
    async def run() -> httpx.Response:
        async with asgi_client(context) as client:
            return await client.post(path, json=payload.model_dump(mode="json"))

    return asyncio.run(run())


def _kernel_event(record_id: str, message: str) -> NvidiaKernelLogEvent:
    return NvidiaKernelLogEvent(
        cluster_id="hp-cluster",
        node_id="worker-1",
        record_id=record_id,
        observed_at=NOW,
        message=message,
        product="H100",
    )


def _health_summary(summary_id: str, reasons: list[str]) -> CollectorHealthSummary:
    return CollectorHealthSummary(
        summary_id=summary_id,
        cluster_id="hp-cluster",
        node_id="worker-1",
        collector=CollectorKind.NVIDIA_KERNEL,
        observed_at=NOW,
        edge_filter_reasons=reasons,
    )


def test_kernel_route_reports_an_xid_line_without_a_code() -> None:
    context = build_context()

    response = _post(
        context,
        NVIDIA_KERNEL_PATH,
        _kernel_event("kmsg-drift-1", "NVRM: Xid (PCI:0000:b9:00): , pid=1234"),
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["normalized"]["xid_events"] == [], "a code was invented"
    assert body["decisions"] == [], "an unparsed line reached the fault policy"
    assert context.fault_ingestion.unresolved_signal_totals == {
        UNPARSED_XID_REASON: 1
    }, "the route did not hand the unresolved signal to the ingestion service"
    state = context.store.get_health_signal_state(
        f"hp-cluster/worker-1/{UNPARSED_XID_REASON}/node"
    )
    assert state is not None and state.active, "no unparsed-line episode opened"
    (marker,) = context.store.list_markers()
    assert marker.incident_id, "the finding did not open an incident"
    incident = context.store.get_incident(marker.incident_id)
    assert incident is not None and incident.node_ids == ["worker-1"], (
        "the finding did not open an incident on the node"
    )
    notifications = context.store.list_notifications()
    assert [item.incident_id for item in notifications] == [marker.incident_id], (
        "operator review was not notified through the route"
    )


def test_kernel_route_leaves_a_parsed_xid_line_alone() -> None:
    context = build_context()

    response = _post(
        context,
        NVIDIA_KERNEL_PATH,
        _kernel_event("kmsg-79", "NVRM: Xid (PCI:0000:b9:00): 79, pid=1234"),
    )

    assert response.status_code == 200, response.text
    assert context.fault_ingestion.unresolved_signal_totals == {}, (
        "a parsed line was counted as unresolved"
    )
    assert (
        context.store.get_health_signal_state(
            f"hp-cluster/worker-1/{UNPARSED_XID_REASON}/node"
        )
        is None
    ), "a parsed line opened an unparsed-line episode"


def test_fabric_manager_route_reports_an_sxid_line_without_a_code() -> None:
    context = build_context()

    response = _post(
        context,
        FABRIC_MANAGER_PATH,
        FabricManagerLogEvent(
            cluster_id="hp-cluster",
            node_id="worker-1",
            record_id="fm-no-code",
            observed_at=NOW,
            source="journal",
            message="nvidia-nvswitch3: SXid (PCI:0000:c1:00.0): garbled, Fatal",
            product="H200",
        ),
    )

    assert response.status_code == 200, response.text
    assert response.json()["normalized"]["sxid_events"] == [], "a code was invented"
    assert context.fault_ingestion.unresolved_signal_totals == {
        UNPARSED_SXID_REASON: 1
    }, "the route did not hand the unresolved SXid signal to the ingestion service"
    state = context.store.get_health_signal_state(
        f"hp-cluster/worker-1/{UNPARSED_SXID_REASON}/node"
    )
    assert state is not None and state.active, "no unparsed-line episode opened"


def test_kernel_health_summary_failure_counts_become_collector_errors() -> None:
    context = build_context()

    response = _post(
        context,
        COLLECTOR_HEALTH_PATH,
        _health_summary(
            "summary-1", ["health-summary", "delivery-failures:3", "kmsg-overflow:1"]
        ),
    )

    assert response.status_code == 200, response.text
    (status,) = context.store.list_collector_statuses("hp-cluster", "worker-1")
    assert status.collector is CollectorKind.NVIDIA_KERNEL
    assert status.errors == [
        "kernel collector reported delivery-failures:3",
        "kernel collector reported kmsg-overflow:1",
    ], "the summary's failure counts were not recorded as collector errors"
    assert status.last_error_at is not None, "a dropping collector looks healthy"
    assert status.last_success_at is None, "a dropping collector was marked ok"
    assert status.batch_id == "summary-1", "the status does not name the summary"


def test_all_zero_kernel_health_summary_only_refreshes_success() -> None:
    context = build_context()

    response = _post(
        context, COLLECTOR_HEALTH_PATH, _health_summary("summary-2", ["health-summary"])
    )

    assert response.status_code == 200, response.text
    (status,) = context.store.list_collector_statuses("hp-cluster", "worker-1")
    assert status.errors == [], "a clean summary produced collector errors"
    assert status.last_error_at is None, "a clean summary set last_error_at"
    assert status.last_success_at is not None, "a clean summary did not refresh success"

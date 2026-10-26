"""The two live GF-GENERAL-LIVE-015 timelines (2026-10-26) through host-telemetry ingestion.

``unsupported_bmc`` (eksb): a node without ``ipmitool`` delivers only health
summaries, which carry the steady TOOL_MISSING marker and, by design, are not
written as raw evidence (no finding, no collection error) -- the read found
zero batches. ``command_failure`` (eks): the tick that closed the replayed BMC
failure carried the NO_DEVICE marker and was delivered for the marker's own
edge. Both must leave the node's capability readable in the store: the raw
evidence where there is an edge, the latest record always.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import httpx

from gpu_fault.app import ApplicationContext
from gpu_fault.host_health import HostMetricSample, HostTelemetryBatch
from gpu_fault.models import WorkloadState
from gpu_fault.store import InMemoryStore
from gpu_fault.telemetry import EvidenceKind
from tests._builders import asgi_client, host_telemetry_batch

T0 = datetime(2026, 10, 26, 4, 24, 12, tzinfo=timezone.utc)
HOST_TELEMETRY_PATH = "/v1/collector-events/host-telemetry"
CLUSTER = "gpu-eks-cluster"
NODE = "ip-10-0-32-162.us-west-2.compute.internal"
MARKER_EDGE = "not-supported:bmc_status_unsupported:node:NO_DEVICE"


def _batch(
    batch_id: str, seconds: float, reason: str, *reasons: str
) -> HostTelemetryBatch:
    observed_at = T0 + timedelta(seconds=seconds)
    return host_telemetry_batch(
        batch_id,
        observed_at,
        [
            HostMetricSample(name="cpu_usage_percent", value=12.0, unit="percent"),
            HostMetricSample(
                name="bmc_status_unsupported", value=1.0, labels={"reason": reason}
            ),
        ],
        cluster_id=CLUSTER,
        node_id=NODE,
        received_at=observed_at,
        runtime_profile_version="simulated-v1",
        workload_state=WorkloadState.IDLE,
        edge_filter_reasons=list(reasons),
    )


def _post(context: ApplicationContext, batch: HostTelemetryBatch) -> httpx.Response:
    async def run() -> httpx.Response:
        async with asgi_client(context) as client:
            return await client.post(
                HOST_TELEMETRY_PATH, json=batch.model_dump(mode="json")
            )

    return asyncio.run(run())


def _marker_records(context: ApplicationContext) -> list[tuple[str, str]]:
    return [
        (item.labels["reason"], item.observed_at.isoformat())
        for item in context.store.list_telemetry_metrics_latest(CLUSTER, NODE)
        if item.name == "bmc_status_unsupported"
    ]


def _evidence_ids(context: ApplicationContext) -> list[str]:
    return [
        item.record_id
        for item in context.store.list_raw_evidence(
            CLUSTER, node_id=NODE, kind=EvidenceKind.HOST_TELEMETRY
        )
    ]


def test_a_steady_marker_on_a_health_summary_is_readable_without_evidence() -> None:
    context = ApplicationContext(store=InMemoryStore())

    response = _post(context, _batch("host-1", 0, "TOOL_MISSING", "health-summary"))

    assert response.status_code == 200, response.text
    assert response.json()["incident_ids"] == [], "NOT_SUPPORTED mints nothing"
    assert _evidence_ids(context) == [], (
        "a health summary without findings or collection errors is not evidence"
    )
    assert _marker_records(context) == [("TOOL_MISSING", T0.isoformat())], (
        "the latest record is the product's only witness of the steady marker"
    )


def test_a_marker_edge_is_evidence_and_moves_the_latest_record() -> None:
    context = ApplicationContext(store=InMemoryStore())
    assert (
        _post(context, _batch("host-1", 0, "TOOL_MISSING", "baseline")).status_code
        == 200
    )

    response = _post(
        context, _batch("host-2", 15, "NO_DEVICE", MARKER_EDGE, "recovered")
    )

    assert response.status_code == 200, response.text
    assert response.json()["incident_ids"] == []
    assert _evidence_ids(context) == ["host-telemetry/host-2", "host-telemetry/host-1"]
    assert _marker_records(context) == [
        ("NO_DEVICE", (T0 + timedelta(seconds=15)).isoformat())
    ]

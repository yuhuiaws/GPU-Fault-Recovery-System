"""A health summary for a node with an active GPU finding is kept as raw evidence.

The collector delivers edges on confirmation and recovery and otherwise one
health summary per node (270-330 s apart). Raw evidence used to be kept for
every edge batch and for a summary only when it minted a finding, so the
summary that re-judges an active finding -- re-grades it, clears it, or, on a
counter reset, leaves it untouched -- left nothing behind. Live (HyperPod,
2026-10-27 01:49:47Z) the summary that carried the reset and could not clear
the ``thermal_violation_total_us`` WARNING, which VALIDATE_GPU then failed on
at the grace end, was not on record. Now a summary of a node that had an
active finding when it arrived is captured, once per batch; summaries of
healthy nodes are not.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

from gpu_fault.gpu_metrics import GpuMetricBatch, GpuMetricSample, GpuMetricSource
from gpu_fault.telemetry import EvidenceKind
from tests._builders import asgi_client, build_context, gpu_metric_batch

NOW = datetime(2026, 10, 27, 1, 48, tzinfo=timezone.utc)
CLUSTER = "hp-cluster"
FAULTED = "worker-1"
HEALTHY = "worker-2"
SUMMARY = ["health-summary"]


def batch(
    batch_id: str, node_id: str, temperature: float, *, seconds: int, reasons: list[str]
) -> GpuMetricBatch:
    return gpu_metric_batch(
        batch_id,
        NOW + timedelta(seconds=seconds),
        GpuMetricSource.DCGM_EXPORTER,
        [
            GpuMetricSample(
                metric_name="DCGM_FI_DEV_GPU_TEMP",
                canonical_name="gpu_temperature_c",
                value=temperature,
                gpu_index="0",
                gpu_uuid="GPU-a",
                pci_bdf="0000:b9:00",
                labels={"modelName": "H100"},
            )
        ],
        cluster_id=CLUSTER,
        node_id=node_id,
        edge_filter_reasons=reasons,
    )


def test_summaries_of_a_node_with_an_active_finding_are_kept_as_evidence() -> None:
    context = build_context()
    batches = [
        # The confirmed edge mints a temperature WARNING: kept, as every edge is.
        batch(
            "dcgm-w1-confirmed", FAULTED, 91, seconds=0, reasons=["candidate-confirmed"]
        ),
        # A healthy node's summary: nothing to re-judge, not kept.
        batch("dcgm-w2-summary", HEALTHY, 60, seconds=5, reasons=SUMMARY),
        # The faulted node's summaries while the finding is active: the one
        # that re-grades it and the one that clears it are both kept.
        batch("dcgm-w1-summary-regrade", FAULTED, 91, seconds=300, reasons=SUMMARY),
        batch("dcgm-w1-summary-clear", FAULTED, 60, seconds=600, reasons=SUMMARY),
        # Nothing active any more: back to not keeping summaries.
        batch("dcgm-w1-summary-healthy", FAULTED, 60, seconds=900, reasons=SUMMARY),
    ]
    active_after: dict[str, list[str]] = {}

    async def scenario() -> None:
        async with asgi_client(context) as client:
            for item in batches:
                response = await client.post(
                    "/v1/collector-events/gpu-metrics",
                    json=item.model_dump(mode="json"),
                )
                assert response.status_code == 200, response.text
                active_after[item.batch_id] = sorted(
                    finding.canonical_name
                    for finding in context.gpu_metrics.findings(CLUSTER, item.node_id)
                )

    asyncio.run(scenario())

    assert active_after == {
        "dcgm-w1-confirmed": ["gpu_temperature_c"],
        "dcgm-w2-summary": [],
        "dcgm-w1-summary-regrade": ["gpu_temperature_c"],
        "dcgm-w1-summary-clear": [],
        "dcgm-w1-summary-healthy": [],
    }

    def kept(node_id: str) -> list[str]:
        return sorted(
            record.record_id
            for record in context.store.list_raw_evidence(
                CLUSTER, node_id=node_id, kind=EvidenceKind.GPU_METRICS
            )
        )

    assert kept(FAULTED) == [
        "gpu-metrics/dcgm-w1-confirmed",
        "gpu-metrics/dcgm-w1-summary-clear",
        "gpu-metrics/dcgm-w1-summary-regrade",
    ], "the edge, and every summary that arrived while a finding was active"
    assert kept(HEALTHY) == [], "a healthy node's summaries are not evidence"

"""A collection-error streak, the way every store backend must keep it.

Batch 126: the WARNING ``dcgm_field_completeness`` finding a failed scrape
books carries the streak's first error batch (its ``observed_at`` and
``finding_id``) while the finding state follows the latest batch, and it asks
for RUN_DIAGNOSTICS only once the streak has lasted the sustain window. The
backend tests drive the same dead-exporter sequence through
``GpuMetricsService`` on each store, with a fresh service -- and, where the
backend has one, a fresh store connection -- per batch, so the streak start can
only come from what the store kept.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from typing import Any

from gpu_fault.gpu_metrics import (
    GpuMetricBatch,
    GpuMetricSample,
    GpuMetricSource,
    GpuMetricsService,
)
from gpu_fault.models import RecoveryAction, WorkloadState
from tests._builders import gpu_metric_batch

NOW = datetime(2026, 10, 17, 15, 57, 1, tzinfo=timezone.utc)
CLUSTER = "cluster-a"
NODE = "node-a"
COMPLETENESS_KEY = (CLUSTER, NODE, "node", "dcgm_field_completeness")
SCRAPE_ERROR = (
    "CollectorError: cannot scrape DCGM exporter: http://127.0.0.1:9400/metrics"
)
#: Seconds after its first failure at which the 15 s DCGM collector reports a
#: dead exporter: two failures at the cadence, then the wait doubles, capped at
#: four intervals (``DcgmMetricsCollector.next_interval_seconds``).
DEAD_EXPORTER_OFFSETS = (0, 15, 30, 60, 120, 180, 240)
#: The default window: the sixth failure is the first one past it.
WINDOW_SECONDS = 180


def erroring_batch(batch_id: str, observed_at: datetime) -> GpuMetricBatch:
    return gpu_metric_batch(
        batch_id,
        observed_at,
        GpuMetricSource.DCGM_EXPORTER,
        [],
        cluster_id=CLUSTER,
        node_id=NODE,
        collection_errors=[SCRAPE_ERROR],
        edge_filter_reasons=["collection-error"],
        runtime_profile_version="simulated-v1",
        workload_state=WorkloadState.IDLE,
    )


def clean_batch(batch_id: str, observed_at: datetime) -> GpuMetricBatch:
    return gpu_metric_batch(
        batch_id,
        observed_at,
        GpuMetricSource.DCGM_EXPORTER,
        [
            GpuMetricSample(
                metric_name="DCGM_FI_DEV_GPU_TEMP",
                canonical_name="gpu_temperature_c",
                value=41.0,
                gpu_index="0",
                gpu_uuid="GPU-a",
                pci_bdf="0000:b9:00",
            )
        ],
        cluster_id=CLUSTER,
        node_id=NODE,
        runtime_profile_version="simulated-v1",
        workload_state=WorkloadState.IDLE,
    )


def assert_streak_survives_the_store(store_for_batch: Callable[[], Any]) -> None:
    """Drive a dead exporter, then its recovery, one fresh service per batch.

    ``store_for_batch`` returns the store the next batch is ingested through:
    the same object for the in-memory store, a reopened connection to the same
    database for the persistent ones.
    """

    first_at = NOW
    activated: list[str] = []
    for offset in DEAD_EXPORTER_OFFSETS:
        store = store_for_batch()
        observed_at = first_at + timedelta(seconds=offset)
        result = GpuMetricsService(store=store).ingest(
            erroring_batch(f"dcgm-{offset}", observed_at)
        )
        activated.extend(finding.finding_id for finding in result.new_findings)
        state = store.get_gpu_finding_state(COMPLETENESS_KEY)
        assert state is not None and state.finding is not None, (
            f"the failed scrape at +{offset}s was not booked"
        )
        assert state.observed_at == observed_at, (
            f"the state at +{offset}s does not follow the latest batch"
        )
        assert state.finding.observed_at == first_at, (
            f"the streak start was lost at +{offset}s: {state.finding.observed_at}"
        )
        assert state.finding.finding_id == "dcgm-0-dcgm-fields", (
            f"the streak changed identity at +{offset}s: {state.finding.finding_id}"
        )
        expected = (
            RecoveryAction.RUN_DIAGNOSTICS
            if offset >= WINDOW_SECONDS
            else RecoveryAction.NO_ACTION
        )
        assert state.finding.automatic_action == expected.value, (
            f"+{offset}s asks for {state.finding.automatic_action}, not {expected}"
        )
        assert [
            item.finding_id
            for item in store.list_gpu_findings(CLUSTER, NODE, active_only=True)
        ] == ["dcgm-0-dcgm-fields"], f"the gap is not visible at +{offset}s"
    assert activated == ["dcgm-0-dcgm-fields"], (
        f"the streak must activate exactly once, at the window: {activated}"
    )

    store = store_for_batch()
    recovered_at = first_at + timedelta(seconds=DEAD_EXPORTER_OFFSETS[-1] + 15)
    GpuMetricsService(store=store).ingest(clean_batch("dcgm-clean", recovered_at))
    state = store.get_gpu_finding_state(COMPLETENESS_KEY)
    assert state is not None and state.finding is None, "the clean scrape clears it"
    assert state.observed_at == recovered_at, "stamped with the clean batch's time"

    store = store_for_batch()
    restarted_at = recovered_at + timedelta(seconds=15)
    result = GpuMetricsService(store=store).ingest(
        erroring_batch("dcgm-again", restarted_at)
    )
    assert result.new_findings == [], "a new streak starts without action"
    state = store.get_gpu_finding_state(COMPLETENESS_KEY)
    assert state is not None and state.finding is not None, "the new streak is booked"
    assert (state.finding.finding_id, state.finding.observed_at) == (
        "dcgm-again-dcgm-fields",
        restarted_at,
    ), "a streak after a clean batch starts over"

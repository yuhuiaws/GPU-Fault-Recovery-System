"""A clean DCGM scrape clears the collection-error finding it followed.

The collector reports a failed scrape as a sample-less batch carrying
``collection_errors`` (F5), which the ingest books as a WARNING
``dcgm_field_completeness`` finding. Until 2026-09-09 nothing ever cleared it:
the WARNING a collector raised while dcgm-exporter was still starting after a
reboot stayed active for good, and VALIDATE_GPU refused every validated restore
of that node with ``active_gpu_health_findings`` (DESTR-014 attempt 10).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from gpu_fault.gpu_metrics import GpuMetricSample, GpuMetricSource, GpuMetricsService
from tests._builders import build_store, gpu_metric_batch

NOW = datetime(2026, 9, 9, 10, 43, tzinfo=timezone.utc)
COMPLETENESS_KEY = ("cluster-a", "node-a", "node", "dcgm_field_completeness")


def _erroring_batch(batch_id: str, observed_at: datetime):
    return gpu_metric_batch(
        batch_id,
        observed_at,
        GpuMetricSource.DCGM_EXPORTER,
        [],
        collection_errors=[
            "CollectorError: cannot scrape DCGM exporter: http://127.0.0.1:9400/metrics"
        ],
        edge_filter_reasons=["collection-error"],
    )


def _clean_batch(batch_id: str, observed_at: datetime):
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
    )


def _active_names(store) -> list[str]:
    return sorted(
        finding.canonical_name
        for finding in store.list_gpu_findings("cluster-a", "node-a", active_only=True)
    )


def test_a_clean_scrape_clears_the_collection_error_finding() -> None:
    store = build_store()
    service = GpuMetricsService(store=store)

    service.ingest(_erroring_batch("dcgm-1", NOW))
    assert _active_names(store) == ["dcgm_field_completeness"], (
        "the failed scrape is booked as a WARNING finding"
    )

    result = service.ingest(_clean_batch("dcgm-2", NOW + timedelta(seconds=15)))

    assert result.accepted_samples == 1
    assert _active_names(store) == [], "the clean scrape cleared the finding"
    state = store.get_gpu_finding_state(COMPLETENESS_KEY)
    assert state is not None and state.finding is None, state
    assert state.observed_at == NOW + timedelta(seconds=15), (
        "the clearing is stamped with the clean batch's time"
    )


def test_a_clean_scrape_with_no_prior_error_writes_no_completeness_state() -> None:
    store = build_store()
    service = GpuMetricsService(store=store)

    service.ingest(_clean_batch("dcgm-1", NOW))

    assert store.get_gpu_finding_state(COMPLETENESS_KEY) is None, (
        "no clearing update is written for a node that never erred"
    )
    assert _active_names(store) == []


def test_a_repeated_scrape_failure_keeps_the_finding_active() -> None:
    store = build_store()
    service = GpuMetricsService(store=store)

    service.ingest(_erroring_batch("dcgm-1", NOW))
    service.ingest(_erroring_batch("dcgm-2", NOW + timedelta(seconds=15)))

    assert _active_names(store) == ["dcgm_field_completeness"]
    state = store.get_gpu_finding_state(COMPLETENESS_KEY)
    assert state is not None and state.consecutive_breaches == 2, state

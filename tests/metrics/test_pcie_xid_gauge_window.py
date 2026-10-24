"""The DCGM last-XID gauge joins a PCIe breach only inside the window.

GF-GENERAL-LIVE-022 ``expired_xid``: ``DCGM_FI_DEV_XID_ERRORS`` holds the *last*
XID for as long as the driver stays loaded, and every scrape re-reads it, so the
gauge's latest sample is always as fresh as the batch. Bounding that sample's
``observed_at`` by the correlation window bounded nothing: a gauge that went to
32/79 long before re-armed the CRITICAL DRAIN ``PCIE_XID_LINK_FAILURE``
composite on every later PCIe replay-rate breach. The spec: only a trusted
same-GPU XID 32/79 inside the window joins; another GPU's or an expired one
does not. What dates the gauge is the XID event its change produced, held to
the window and the rules the kernel-event path already applies.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from gpu_fault.channel_registry import GPU_METRICS_PATH
from gpu_fault.gpu_metrics import (
    GpuHealthFinding,
    GpuMetricBatch,
    GpuMetricSample,
    GpuMetricsIngestionResult,
    GpuMetricSource,
    GpuMetricsService,
)
from gpu_fault.models import RecoveryAction, WorkloadState
from gpu_fault.policy import GpuFaultPolicyEngine
from gpu_fault.store import InMemoryStore, SqliteStore
from gpu_fault.xid_correlation import XidCorrelationCoordinator
from tests._builders import asgi_client, build_context

NOW = datetime(2026, 10, 24, 9, tzinfo=timezone.utc)
RULE = "PCIE_XID_LINK_FAILURE"
GPU_A = ("GPU-a", "0000:01:00.0")
GPU_B = ("GPU-b", "0000:02:00.0")


@pytest.fixture(params=["memory", "sqlite"])
def store(request: pytest.FixtureRequest, tmp_path: Path) -> Iterator[Any]:
    value = (
        InMemoryStore()
        if request.param == "memory"
        else SqliteStore(str(tmp_path / "gauge.db"))
    )
    yield value
    if isinstance(value, SqliteStore):
        value.close()


def metric(name: str, value: float, gpu: tuple[str, str] = GPU_A) -> GpuMetricSample:
    return GpuMetricSample(
        metric_name=f"DCGM_FI_DEV_{name.upper()}",
        canonical_name=name,
        value=value,
        gpu_uuid=gpu[0],
        pci_bdf=gpu[1],
    )


def batch(
    batch_id: str,
    seconds: int,
    *,
    pcie: float,
    xid: float,
    others: tuple[GpuMetricSample, ...] = (),
) -> GpuMetricBatch:
    return GpuMetricBatch(
        batch_id=batch_id,
        cluster_id="cluster-a",
        node_id="node-a",
        observed_at=NOW + timedelta(seconds=seconds),
        source=GpuMetricSource.DCGM_EXPORTER,
        samples=[metric("pcie_replay_total", pcie), metric("xid_last_error", xid)]
        + list(others),
        product="H100",
        driver_branch=575,
        cuda_version="12.9",
        runtime_profile_version="simulated-v1",
        workload_state=WorkloadState.IDLE,
        evidence_ref=f"prometheus://node-a/metrics/{batch_id}",
    )


def persist_changes(
    service: GpuMetricsService, item: GpuMetricBatch, result: GpuMetricsIngestionResult
) -> None:
    """What the API does after the service: persist the XID events the batch's
    gauge changes produced (``_ingest_gpu_findings`` -> ``ingest_xid``)."""

    coordinator = XidCorrelationCoordinator(
        service.store,
        GpuFaultPolicyEngine(),
        lambda _event, decision: decision,
        now=lambda: item.observed_at,
    )
    for event in result.xid_events:
        coordinator.ingest(event)


def ingest(
    service: GpuMetricsService, item: GpuMetricBatch
) -> GpuMetricsIngestionResult:
    result = service.ingest(item)
    persist_changes(service, item, result)
    return result


def composites(result: GpuMetricsIngestionResult) -> list[GpuHealthFinding]:
    return [
        finding
        for finding in result.new_composite_findings
        if finding.correlation_rule_id == RULE
    ]


def active_composites(service: GpuMetricsService) -> list[GpuHealthFinding]:
    return [
        finding
        for finding in service.findings("cluster-a", "node-a", active_only=True)
        if finding.correlation_rule_id == RULE
    ]


@pytest.mark.parametrize("xid", [32, 79])
def test_an_expired_gauge_does_not_join_a_later_pcie_breach(
    store: Any, xid: int
) -> None:
    """The live ``expired_xid`` plan: the gauge goes to 32 four reads before
    the PCIe rate breaches, i.e. a minute earlier than the 45 s window."""

    service = GpuMetricsService(store=store)
    ingest(service, batch("baseline", 0, pcie=1000, xid=0))
    changed = ingest(service, batch("xid", 15, pcie=1000, xid=xid))
    assert [event.xid for event in changed.xid_events] == [xid], (
        "the gauge change is the XID event that dates it"
    )
    for seconds in (30, 45, 60):
        ingest(service, batch(f"held-{seconds}", seconds, pcie=1000, xid=xid))

    result = ingest(service, batch("breach", 75, pcie=1005, xid=xid))

    assert composites(result) == [], (
        "a gauge that changed 60 s before the breach is outside the 45 s window"
    )
    (plain,) = [
        finding
        for finding in result.new_findings
        if finding.canonical_name == "pcie_replay_total"
    ]
    assert plain.automatic_action == RecoveryAction.RUN_DIAGNOSTICS.value, (
        "the breach itself is still the read-only PCIe diagnostic"
    )


@pytest.mark.parametrize(("after", "joined"), [(15, True), (45, True), (46, False)])
def test_a_gauge_change_joins_only_inside_the_window(
    store: Any, after: int, joined: bool
) -> None:
    service = GpuMetricsService(store=store)
    ingest(service, batch("baseline", 0, pcie=1000, xid=0))
    ingest(service, batch("xid", 15, pcie=1000, xid=79))

    result = ingest(service, batch("breach", 15 + after, pcie=1020, xid=79))

    found = composites(result)
    assert bool(found) is joined, (
        f"a gauge change {after} s before the breach, window 45 s: {found}"
    )
    if joined:
        assert found[0].automatic_action == RecoveryAction.DRAIN.value, found
        assert found[0].component_metrics == ["pcie_replay_total", "xid_last_error"]
        assert "xid-event://xid-GPU-a-xid-79" in found[0].component_evidence_refs, (
            "the composite names the gauge change it joined"
        )


def test_a_gauge_change_on_another_gpu_does_not_join(store: Any) -> None:
    service = GpuMetricsService(store=store)
    ingest(
        service,
        batch(
            "baseline",
            0,
            pcie=1000,
            xid=0,
            others=(
                metric("pcie_replay_total", 1000, GPU_B),
                metric("xid_last_error", 0, GPU_B),
            ),
        ),
    )
    ingest(
        service,
        batch(
            "xid-b",
            15,
            pcie=1000,
            xid=0,
            others=(
                metric("pcie_replay_total", 1000, GPU_B),
                metric("xid_last_error", 79, GPU_B),
            ),
        ),
    )

    result = ingest(
        service,
        batch(
            "breach-a",
            30,
            pcie=1005,
            xid=0,
            others=(
                metric("pcie_replay_total", 1000, GPU_B),
                metric("xid_last_error", 79, GPU_B),
            ),
        ),
    )

    assert composites(result) == [], "GPU-b's XID cannot join GPU-a's PCIe rate"


def test_a_cleared_episode_is_not_rearmed_by_the_same_gauge_change(store: Any) -> None:
    service = GpuMetricsService(store=store)
    ingest(service, batch("baseline", 0, pcie=1000, xid=0))
    first = ingest(service, batch("xid-and-breach", 15, pcie=1005, xid=79))
    assert len(composites(first)) == 1, "a change in the breaching batch joins"
    ingest(service, batch("healthy", 30, pcie=1005, xid=79))
    assert active_composites(service) == [], "a healthy PCIe sample clears it"

    rearmed = ingest(service, batch("next-breach", 45, pcie=1010, xid=79))

    assert composites(rearmed) == [], (
        "the change that formed the cleared episode cannot arm the next one"
    )
    fresh = ingest(service, batch("new-xid", 60, pcie=1015, xid=32))
    assert len(composites(fresh)) == 1, "a new gauge change joins normally"


def test_the_gauge_reads_no_history_of_its_own(
    store: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = GpuMetricsService(store=store)
    ingest(
        service,
        batch(
            "baseline",
            0,
            pcie=1000,
            xid=0,
            others=(
                metric("pcie_replay_total", 1000, GPU_B),
                metric("xid_last_error", 0, GPU_B),
            ),
        ),
    )
    reads: list[tuple[datetime, datetime]] = []
    list_events = store.list_xid_events

    def history(cluster_id, node_id, *, observed_after, observed_before):
        reads.append((observed_after, observed_before))
        return list_events(
            cluster_id,
            node_id,
            observed_after=observed_after,
            observed_before=observed_before,
        )

    monkeypatch.setattr(store, "list_xid_events", history)

    def measured(item: GpuMetricBatch) -> tuple[GpuMetricsIngestionResult, list]:
        """The composite's own history reads; the XID policy's are not its."""

        reads.clear()
        result = service.ingest(item)
        made = list(reads)
        persist_changes(service, item, result)
        return result, made

    _, made = measured(
        batch(
            "xid-without-breach",
            15,
            pcie=1000,
            xid=79,
            others=(
                metric("pcie_replay_total", 1000, GPU_B),
                metric("xid_last_error", 0, GPU_B),
            ),
        )
    )
    assert made == [], "a gauge change without a PCIe breach needs no history"

    result, made = measured(
        batch(
            "breaches",
            30,
            pcie=1005,
            xid=79,
            others=(
                metric("pcie_replay_total", 1005, GPU_B),
                metric("xid_last_error", 79, GPU_B),
            ),
        )
    )

    assert made == [(NOW - timedelta(seconds=15), NOW + timedelta(seconds=75))], (
        f"kernel events and gauge changes share one bounded query: {made}"
    )
    assert sorted(item.gpu_uuid for item in composites(result)) == ["GPU-a", "GPU-b"], (
        "GPU-a's change 15 s earlier and GPU-b's change in this batch both join"
    )


@pytest.mark.parametrize(("after", "joined"), [(15, True), (60, False)])
def test_the_api_dates_the_gauge_by_its_persisted_change(
    store: Any, after: int, joined: bool
) -> None:
    """Through the real ingestion route: the API persists the gauge change of
    one batch, and a later batch's breach joins it only inside the window."""

    context = build_context(store=store)

    async def scenario() -> None:
        async with asgi_client(context) as client:
            for item in (
                batch("baseline", 0, pcie=1000, xid=0),
                batch("xid", 15, pcie=1000, xid=32),
                batch("breach", 15 + after, pcie=1020, xid=32),
            ):
                response = await client.post(
                    GPU_METRICS_PATH, json=item.model_dump(mode="json")
                )
                assert response.status_code == 200, response.text

    asyncio.run(scenario())

    found = active_composites(context.gpu_metrics)
    assert bool(found) is joined, f"breach {after} s after the XID 32 change: {found}"
    incident = context.store.get_incident_by_event(
        "gpu-breach-GPU-a-composite-PCIE_XID_LINK_FAILURE"
    )
    assert (incident is not None) is joined, (
        "only an in-window change opens the composite's DRAIN incident"
    )

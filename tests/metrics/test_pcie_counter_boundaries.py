"""PCIe replay counter boundaries the GF-GENERAL-LIVE-022 live variants grade.

Live 2026-10-26 (plain EKS, release R13g) ``counter_reset`` and
``source_boot_change`` both FAILed with "a PCIe counter reset minted a
finding" / "a source/boot change formed a cross-generation PCIe rate". The
product's records say otherwise: the incident each run saw was minted by the
variant's own raise (+50 per 15-s read, 200/min; +5 per read, 20/min -- the
same increment ``over_rate_limit`` uses to prove the WARNING), never at the
reset tick or across the boot boundary. These tests pin, on the real collector
delivery filter and the real ingest, what the product does at the two
boundaries, so the runner's oracle grades the boundary and not the raise:

* a counter that falls below its previous accepted value is delivered as a
  ``counter-reset`` edge, graded ``delta=None`` (no finding, an active finding
  is kept rather than cleared or re-minted) and becomes the new baseline;
* a sample whose ``source_boot_id`` differs from the previous accepted one is
  graded ``delta=None`` (no cross-generation rate) and becomes the new
  generation's baseline; same-generation rates after it grade as always.

No product code changes with this module.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from gpu_fault.gpu_composites import metric_counter_delta
from gpu_fault.gpu_metrics import (
    GpuMetricBatch,
    GpuMetricLatest,
    GpuMetricSample,
    GpuMetricSource,
    GpuMetricsService,
)
from tests._builders import gpu_metric_batch
from tests.collectors._support import DcgmMetricsCollector, RecordingSink, context

NOW = datetime(2026, 10, 26, 7, 7, tzinfo=timezone.utc)
PCIE = "pcie_replay_total"
CLUSTER = "gpu-eks-cluster"
#: The cluster of ``tests.collectors._support.context()``: the collector-built
#: batches below carry it.
COLLECTOR_CLUSTER = "hp-cluster"
GPU = "GPU-053ab270-714e-0f16-46ad-f96f00134d8a"
BOOT_A = "b39c4e22-0873-4642-ba92-895dfd9e76ef"
BOOT_B = "5d2a6a0d-70f5-4a5e-9d44-2d2a8a3b1c11"


def sample(value: float, *, boot: str = BOOT_A) -> GpuMetricSample:
    return GpuMetricSample(
        metric_name="DCGM_FI_DEV_PCIE_REPLAY_COUNTER",
        canonical_name=PCIE,
        value=value,
        gpu_index="7",
        gpu_uuid=GPU,
        pci_bdf="0000:9b:00",
        labels={"modelName": "NVIDIA H200", "source_boot_id": boot},
    )


def batch(name: str, value: float, *, at: int, boot: str = BOOT_A) -> GpuMetricBatch:
    return gpu_metric_batch(
        f"dcgm-worker-1-{name}",
        NOW + timedelta(seconds=at),
        GpuMetricSource.DCGM_EXPORTER,
        [sample(value, boot=boot)],
        cluster_id=CLUSTER,
        node_id="worker-1",
        product="NVIDIA H200",
    )


def pcie_findings(result: Any) -> list[Any]:
    return [item for item in result.findings if item.canonical_name == PCIE]


def active_pcie(service: GpuMetricsService, cluster: str = CLUSTER) -> list[Any]:
    return [
        item
        for item in service.findings(cluster, "worker-1")
        if item.canonical_name == PCIE
    ]


def latest_pcie(service: GpuMetricsService, cluster: str = CLUSTER) -> GpuMetricLatest:
    (item,) = [
        item
        for item in service.latest(cluster, "worker-1")
        if item.sample.canonical_name == PCIE
    ]
    return item


# --------------------------------------------------------------------------- #
# ingest: the reset tick
# --------------------------------------------------------------------------- #
def test_a_reset_after_an_under_limit_raise_mints_nothing_and_rearms_the_baseline():
    service = GpuMetricsService()
    results = [
        service.ingest(batch("b0", 0, at=0)),
        service.ingest(batch("b1", 1, at=15)),  # 4/min: under the 8/min rule
        service.ingest(batch("b2", 2, at=30)),
        service.ingest(batch("b3", 3, at=45)),
    ]
    assert not any(pcie_findings(result) for result in results), (
        "1 per 15 s is 4/min: under the rule"
    )

    reset = service.ingest(batch("reset", 0, at=60))

    assert not pcie_findings(reset), "a counter reset is not a fault"
    assert not reset.new_findings, "nothing is minted at the reset tick"
    assert latest_pcie(service).sample.value == 0, "the baseline is the reset value"

    resumed = service.ingest(batch("b5", 1, at=75))

    assert not pcie_findings(resumed), "the next tick measures from the reset value"
    assert not service.findings(CLUSTER, "worker-1", active_only=False), (
        "no finding ever formed"
    )


def test_a_storm_then_a_reset_keeps_the_storm_finding_and_mints_nothing_at_the_reset():
    """The live counter_reset plan: +50 per 15-s read is 200/min."""

    service = GpuMetricsService()
    service.ingest(batch("b0", 0, at=0))
    storm = service.ingest(batch("b1", 50, at=15))
    (finding,) = pcie_findings(storm)
    assert finding.rate_per_minute == 200 and finding.delta == 50
    assert [item.canonical_name for item in storm.new_findings] == [PCIE], (
        "the raise, not the reset, is what mints the incident"
    )
    held = [
        service.ingest(batch("b2", 100, at=30)),
        service.ingest(batch("b3", 150, at=45)),
    ]
    assert all(pcie_findings(result) for result in held), "the storm keeps breaching"
    assert not any(result.new_findings for result in held), "one finding stays active"

    reset = service.ingest(batch("reset", 0, at=60))

    assert not pcie_findings(reset) and not reset.new_findings
    assert len(active_pcie(service)) == 1, (
        "reverting the counter is an invalid delta: the finding is kept, not "
        "cleared and not re-minted"
    )
    assert latest_pcie(service).sample.value == 0

    summary = service.ingest(batch("summary", 0, at=360))

    assert not pcie_findings(summary) and not summary.new_findings
    assert not active_pcie(service), "a zero delta over a real window recovers it"


# --------------------------------------------------------------------------- #
# ingest: the boot boundary
# --------------------------------------------------------------------------- #
def test_a_boot_change_starts_a_new_baseline_and_forms_no_cross_generation_rate():
    service = GpuMetricsService()
    service.ingest(batch("a0", 0, at=0, boot=BOOT_A))

    jump = service.ingest(batch("b0", 205, at=300, boot=BOOT_B))

    assert not pcie_findings(jump) and not jump.new_findings, (
        "205 over 300 s would be 41/min; across a boot change it is no rate"
    )
    latest = latest_pcie(service)
    assert latest.sample.value == 205
    assert latest.sample.labels["source_boot_id"] == BOOT_B, (
        "the new generation's first reading is the baseline"
    )

    held = service.ingest(batch("b1", 205, at=315, boot=BOOT_B))
    assert not pcie_findings(held), "a held value is a zero delta"

    same_generation = service.ingest(batch("b2", 210, at=330, boot=BOOT_B))

    (finding,) = pcie_findings(same_generation)
    assert finding.rate_per_minute == 20 and finding.delta == 5, (
        "the live source_boot_change plan (+5 per 15-s read) is a same-generation "
        "storm the product grades as always"
    )


def test_metric_counter_delta_is_unknown_across_boots_and_at_a_reset():
    previous = GpuMetricLatest(
        cluster_id="gpu-eks-cluster",
        node_id="worker-1",
        sample=sample(150, boot=BOOT_A),
        observed_at=NOW,
        source=GpuMetricSource.DCGM_EXPORTER,
    )
    assert (
        metric_counter_delta(sample(0, boot=BOOT_A), previous, is_counter=True) is None
    )
    assert (
        metric_counter_delta(sample(400, boot=BOOT_B), previous, is_counter=True)
        is None
    )
    assert (
        metric_counter_delta(sample(155, boot=BOOT_A), previous, is_counter=True) == 5
    )


# --------------------------------------------------------------------------- #
# collector + ingest: what the node really ships at the two boundaries
# --------------------------------------------------------------------------- #
def exporter_text(pcie: float) -> str:
    labels = f'gpu="7",UUID="{GPU}",modelName="NVIDIA H200"'
    return "\n".join(
        [
            f"DCGM_FI_DEV_GPU_TEMP{{{labels}}} 41",
            f"DCGM_FI_DEV_ECC_DBE_VOL_TOTAL{{{labels}}} 0",
            f"DCGM_FI_DEV_ROW_REMAP_FAILURE{{{labels}}} 0",
            f"DCGM_FI_DEV_ROW_REMAP_PENDING{{{labels}}} 0",
            f"DCGM_FI_DEV_PCIE_REPLAY_COUNTER{{{labels}}} {pcie}",
            f"DCGM_FI_DEV_NVLINK_ERROR_DL_CRC{{{labels}}} 0",
            f"DCGM_FI_DEV_NVLINK_ERROR_DL_RECOVERY{{{labels}}} 0",
            f"DCGM_FI_DEV_NVLINK_ERROR_DL_REPLAY{{{labels}}} 0",
            f"DCGM_FI_DEV_CLOCK_THROTTLE_REASONS{{{labels}}} 0",
            f"DCGM_FI_DEV_POWER_MGMT_LIMIT{{{labels}}} 700",
            "",
        ]
    )


def collector(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *, boot: str, times: list[int]
) -> tuple[DcgmMetricsCollector, RecordingSink]:
    path = tmp_path / f"boot-{boot}"
    path.write_text(boot + "\n", encoding="ascii")
    monkeypatch.setenv("GPU_FAULT_BOOT_ID_PATH", str(path))
    sink = RecordingSink()
    clock = iter(NOW + timedelta(seconds=at) for at in times)
    return (
        DcgmMetricsCollector(
            sink,
            context(),
            node_id="worker-1",
            now=lambda: next(clock),
            health_summary_seconds=300,
        ),
        sink,
    )


def shipped(sink: RecordingSink, start: int = 0) -> list[GpuMetricBatch]:
    return [
        GpuMetricBatch.model_validate(payload)
        for _path, payload in sink.requests[start:]
    ]


def test_the_collector_ships_the_reset_tick_as_a_reset_edge_the_ingest_grades_as_unknown(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The live counter_reset reads: 0,0,0 (baseline), 50,100,150, then 0 held."""

    dcgm, sink = collector(
        monkeypatch, tmp_path, boot=BOOT_A, times=[15 * index for index in range(10)]
    )
    for value in (0, 0, 0, 50, 100, 150, 0, 0, 0, 0):
        dcgm.collect_text(exporter_text(value))
    batches = shipped(sink)
    assert not batches[0].collection_errors, "the scrape carries every core field"
    assert [
        (
            round(next(s.value for s in b.samples if s.canonical_name == PCIE)),
            b.edge_filter_reasons,
        )
        for b in batches
    ] == [
        (0, ["initial-baseline"]),
        (50, ["counter-increased"]),
        (100, ["counter-increased"]),
        (150, ["counter-increased"]),
        (0, ["counter-reset"]),
    ], "unchanged reads are suppressed; the raise and the reset are edges"
    assert all(
        s.labels["source_boot_id"] == BOOT_A for b in batches for s in b.samples
    ), "every shipped sample names the collector's boot"

    service = GpuMetricsService()
    results = [service.ingest(item) for item in batches]

    minted = [index for index, result in enumerate(results) if result.new_findings]
    assert minted == [1], "the +50 read (200/min) mints the one incident"
    assert not pcie_findings(results[4]) and not results[4].new_findings, (
        "the reset tick is graded unknown: no finding, nothing minted"
    )
    assert len(active_pcie(service, COLLECTOR_CLUSTER)) == 1
    assert latest_pcie(service, COLLECTOR_CLUSTER).sample.value == 0


def test_a_restarted_collector_under_a_new_boot_id_starts_a_new_baseline(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The live source_boot_change run: the repointed collector restarts under
    the replayed boot id, its first scrape is pass-through (0), then +5 per
    read. The boundary (0, A) -> (0, B) is no rate; the finding forms inside B
    at the second ramp read (20/min), the plan's own storm."""

    service = GpuMetricsService()
    old, old_sink = collector(monkeypatch, tmp_path, boot=BOOT_A, times=[0])
    old.collect_text(exporter_text(0))
    service.ingest(shipped(old_sink)[0])

    new, new_sink = collector(
        monkeypatch, tmp_path, boot=BOOT_B, times=[60, 75, 90, 105, 120, 135]
    )
    for value in (0, 0, 0, 5, 10, 15):
        new.collect_text(exporter_text(value))
    batches = shipped(new_sink)
    assert [b.edge_filter_reasons for b in batches] == [
        ["initial-baseline"],
        ["counter-increased"],
        ["counter-increased"],
        ["counter-increased"],
    ]
    assert all(
        s.labels["source_boot_id"] == BOOT_B for b in batches for s in b.samples
    ), "the restarted collector labels every sample with the new boot"

    results = [service.ingest(item) for item in batches]

    assert not pcie_findings(results[0]), "(0, A) -> (0, B): no cross-generation rate"
    assert (
        latest_pcie(service, COLLECTOR_CLUSTER).sample.labels["source_boot_id"]
        == BOOT_B
    )
    assert not pcie_findings(results[1]), "5 over 45 s inside B is 6.7/min"
    (finding,) = pcie_findings(results[2])
    assert finding.rate_per_minute == 20 and finding.delta == 5, (
        "the second ramp read inside the new boot is the legitimate storm"
    )
    assert [bool(result.new_findings) for result in results] == [
        False,
        False,
        True,
        False,
    ]


def test_a_jump_at_the_boot_boundary_is_no_rate_but_the_new_generations_baseline(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The boundary the rewritten source_boot_change variant drives: the new
    boot's very first reading is already the replayed jump."""

    service = GpuMetricsService()
    old, old_sink = collector(monkeypatch, tmp_path, boot=BOOT_A, times=[0])
    old.collect_text(exporter_text(0))
    service.ingest(shipped(old_sink)[0])

    new, new_sink = collector(monkeypatch, tmp_path, boot=BOOT_B, times=[60, 75, 420])
    for value in (205, 205, 205):
        new.collect_text(exporter_text(value))
    batches = shipped(new_sink)
    assert [b.edge_filter_reasons for b in batches] == [
        ["initial-baseline"],
        ["health-summary"],
    ]

    results = [service.ingest(item) for item in batches]

    assert not any(pcie_findings(result) for result in results), (
        "(0, A) -> (205, B) is no rate; (205, B) -> (205, B) is a zero delta"
    )
    assert not any(result.new_findings for result in results), "nothing minted"
    latest = latest_pcie(service, COLLECTOR_CLUSTER)
    assert (
        latest.sample.value == 205 and latest.sample.labels["source_boot_id"] == BOOT_B
    )
    assert not service.findings(COLLECTOR_CLUSTER, "worker-1", active_only=False), (
        "no finding ever formed across or inside the new boot"
    )

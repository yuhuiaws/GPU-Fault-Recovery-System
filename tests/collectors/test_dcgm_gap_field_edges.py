"""Batch 233: what the DCGM collector reads and delivers for the audit's fields.

Each field is read under its 4.5.3 name (the one the counters CSV exports)
and its DCGM 4.6 name, an identical pair collapses to one sample, the HW
slowdown times arrive in microseconds, the slow states (1523, 346, 389, ECC
mode, InfoROM) are delivery edges on change, the PCIe link gauges are not, the
downtrain verdict is counted here under load (``collectors/gpu/pcie_link``),
field 1501 is an edge only above its link's expected rate, brake time in 1424
is delivered on every tick, and the InfoROM answer comes from nvidia-smi at the
inventory cadence (``collectors/gpu/inforom``).
"""

from __future__ import annotations

import logging
import subprocess
from datetime import datetime, timedelta, timezone

import pytest

from gpu_fault.channel_registry import GPU_METRICS_PATH
from gpu_fault.collectors.gpu.dcgm import (
    DCGM_METRICS,
    STATE_CHANGE_REASONS,
    DcgmMetricsCollector,
    parse_dcgm_samples,
)
from gpu_fault.collectors.gpu.inforom import (
    INFOROM_QUERY,
    InforomValidityProbe,
    parse_inforom_validation,
)
from gpu_fault.collectors.gpu.pcie_link import PcieLinkDowntrainTracker
from gpu_fault.collectors.models import CollectorContext
from gpu_fault.collectors.scheduling import next_stable_phase
from gpu_fault.gpu_metrics import GpuMetricBatch, GpuMetricSample, GpuMetricsService

CLUSTER = "gap-cluster"
NODE = "gap-node"
SUMMARY_SECONDS = 86_400
START = next_stable_phase(
    datetime(2026, 11, 1, 12, tzinfo=timezone.utc),
    cluster_id=CLUSTER,
    node_id=NODE,
    channel="gpu-metrics",
    interval_seconds=SUMMARY_SECONDS,
) + timedelta(seconds=1)
HEALTHY = {
    "gpu_temperature_c": 40,
    "memory_temperature_c": 45,
    "power_limit_w": 700,
    "ecc_mode_current": 1,
    "ecc_mode_pending": 1,
    "pcie_link_gen_max": 5,
    "pcie_link_width_max": 16,
    "pcie_link_gen_current": 5,
    "pcie_link_width_current": 16,
}
#: The CSV names (4.5.3) -> their DCGM 4.6 names, for the renamed fields.
RENAMED = {
    "DCGM_FI_DEV_GET_GPU_RECOVERY_ACTION": "DCGM_FI_DEV_GPU_RECOVERY_ACTION",
    "DCGM_FI_DEV_THRESHOLD_SRM": "DCGM_FI_DEV_SRAM_EXCEEDED",
    "DCGM_FI_DEV_BANKS_REMAP_ROWS_AVAIL_LOW": "DCGM_FI_DEV_BANK_REMAP_AVAIL_LOW",
    "DCGM_FI_DEV_BANKS_REMAP_ROWS_AVAIL_NONE": "DCGM_FI_DEV_BANK_REMAP_AVAIL_NONE",
    "DCGM_FI_DEV_ECC_CURRENT": "DCGM_FI_DEV_ECC_MODE",
    "DCGM_FI_DEV_PCIE_COUNT_CORRECTABLE_ERRORS": "DCGM_FI_DEV_PCIE_CORRECTABLE_ERROR_TOTAL",
}


def exporter_text(values: dict[str, float], *, gpu: str = "GPU-a") -> str:
    labels = (
        f'gpu="0",UUID="{gpu}",pci_bus_id="00000000:59:00.0",modelName="NVIDIA H200"'
    )
    return "".join(
        f"{name}{{{labels}}} {values.get(canonical, HEALTHY.get(canonical, 0))}\n"
        for name, (canonical, _unit) in DCGM_METRICS.items()
    )


class IngestingSink:
    def __init__(self) -> None:
        self.service = GpuMetricsService()
        self.batches: list[GpuMetricBatch] = []

    def post(self, path: str, payload: dict) -> dict:
        assert path == GPU_METRICS_PATH
        batch = GpuMetricBatch.model_validate(payload)
        assert not batch.collection_errors, batch.collection_errors
        self.batches.append(batch)
        return self.service.ingest(batch).model_dump(mode="json")


class Node:
    def __init__(self) -> None:
        self.sink = IngestingSink()
        self.collector = DcgmMetricsCollector(
            self.sink,
            CollectorContext(cluster_id=CLUSTER),
            node_id=NODE,
            health_summary_seconds=SUMMARY_SECONDS,
            edge_confirmation_samples=3,
        )
        self.at = START

    def tick(self, values: dict[str, float] | None = None, **kwargs) -> GpuMetricBatch:
        batch = self.collector.collect_text(
            exporter_text(values or {}), observed_at=self.at, **kwargs
        )
        self.at += timedelta(seconds=15)
        return batch

    def delivered(self) -> list[list[str]]:
        return [batch.edge_filter_reasons for batch in self.sink.batches]


def test_each_renamed_field_is_read_under_both_names_and_collapses_to_one_sample():
    for legacy, renamed in RENAMED.items():
        assert DCGM_METRICS[legacy] == DCGM_METRICS[renamed], legacy
        labels = 'gpu="0",UUID="GPU-a"'
        samples = parse_dcgm_samples(
            f"{legacy}{{{labels}}} 1\n{renamed}{{{labels}}} 1\n"
        )
        assert [sample.metric_name for sample in samples] == [legacy], (
            f"an identical {legacy}/{renamed} pair is one observation"
        )


def test_hw_slowdown_times_are_read_as_microseconds():
    samples = parse_dcgm_samples(
        'DCGM_FI_DEV_CLOCKS_EVENT_REASON_HW_POWER_BRAKE_SLOWDOWN_NS{gpu="0",UUID="GPU-a"} 2000000\n'
        'DCGM_FI_DEV_CLOCKS_EVENT_REASON_HW_THERM_SLOWDOWN_NS{gpu="0",UUID="GPU-a"} 3000\n'
    )

    assert {sample.canonical_name: sample.value for sample in samples} == {
        "hw_power_brake_violation_total_us": 2000.0,
        "hw_thermal_slowdown_violation_total_us": 3.0,
    }
    assert {sample.unit for sample in samples} == {"microseconds"}


@pytest.mark.parametrize(
    ("canonical", "value", "reason"),
    [
        ("gpu_recovery_action", 3, "gpu-recovery-action-changed"),
        ("sram_threshold_exceeded", 1, "sram-threshold-changed"),
        ("row_remap_banks_none", 1, "row-remap-headroom-changed"),
        ("ecc_mode_pending", 0, "ecc-mode-changed"),
    ],
)
def test_the_slow_states_are_delivered_when_they_change(canonical, value, reason):
    node = Node()
    node.tick()
    node.tick()

    node.tick({canonical: value})

    assert node.delivered() == [["initial-baseline"], [reason]]
    assert STATE_CHANGE_REASONS[canonical] == reason


def test_the_inforom_answer_is_a_state_edge_too():
    node = Node()

    def inforom(value: float) -> list[GpuMetricSample]:
        return [
            GpuMetricSample(
                metric_name="nvidia_smi_inforom_checksum_validation",
                canonical_name="inforom_checksum_valid",
                value=value,
                gpu_uuid="GPU-a",
            )
        ]

    node.tick(extra_samples=inforom(1))
    node.tick(extra_samples=inforom(1))
    node.tick(extra_samples=inforom(0))

    assert node.delivered() == [["initial-baseline"], ["inforom-validity-changed"]]
    finding = node.sink.service.findings(CLUSTER, NODE)
    assert [item.canonical_name for item in finding] == ["inforom_checksum_valid"]


def test_the_pcie_link_gauges_are_not_state_edges():
    """An idle GPU may downshift (nvidia-smi): a gen/width change is no edge."""

    node = Node()
    node.tick()

    node.tick({"pcie_link_gen_current": 1, "pcie_link_width_current": 4})

    assert node.delivered() == [["initial-baseline"]]


def link(gen, width, *, max_gen=5, max_width=16, util=90.0, gpu="GPU-a"):
    values = {
        "pcie_link_gen_max": max_gen,
        "pcie_link_width_max": max_width,
        "pcie_link_gen_current": gen,
        "pcie_link_width_current": width,
        "gpu_utilization_percent": util,
    }
    return [
        GpuMetricSample(
            metric_name=name,
            canonical_name=name,
            value=value,
            gpu_uuid=gpu,
            labels={"UUID": gpu},
        )
        for name, value in values.items()
        if value is not None
    ]


def verdicts(tracker: PcieLinkDowntrainTracker, *scrapes) -> list[float | None]:
    out: list[float | None] = []
    for scrape in scrapes:
        derived = tracker.observe(scrape)
        out.append(derived[0].value if derived else None)
    return out


def test_three_consecutive_valid_samples_below_the_maximum_confirm_a_downtrain():
    tracker = PcieLinkDowntrainTracker()

    assert verdicts(tracker, link(5, 8), link(5, 8), link(5, 8), link(5, 8)) == [
        None,
        None,
        1.0,
        1.0,
    ], "no verdict until the third valid sample"
    derived = tracker.observe(link(4, 16))[0]
    assert derived.value == 1.0, "Gen4 of a Gen5 maximum is a downtrain too"
    assert derived.labels["consecutive_valid_samples"] == "5"
    assert derived.labels["pcie_link_gen_current"] == "4"


def test_idle_samples_carry_the_verdict_and_never_count():
    tracker = PcieLinkDowntrainTracker()
    downtrained = link(5, 8)
    idle = link(5, 8, util=10)

    assert verdicts(
        tracker, downtrained, idle, idle, downtrained, idle, downtrained
    ) == [None, None, None, None, None, 1.0], (
        "idle neither counts nor resets: the third valid sample confirms"
    )
    assert verdicts(tracker, idle, link(5, 16, util=0)) == [1.0, 1.0], (
        "an idle full link proves nothing: the GPU is not in use"
    )


def test_a_valid_full_link_clears_the_verdict_and_the_streak():
    tracker = PcieLinkDowntrainTracker()
    verdicts(tracker, link(5, 8), link(5, 8), link(5, 8))

    assert verdicts(tracker, link(5, 16), link(5, 8), link(5, 8), link(5, 8)) == [
        0.0,
        0.0,
        0.0,
        1.0,
    ], "a new streak stands on the last verdict until it is three long"


@pytest.mark.parametrize(
    ("scrape", "verdict"),
    [
        (link(6, 16, max_gen=6), 0.0),
        (link(5, 16, max_gen=6), 1.0),
        (link(4, 16, max_gen=4), 0.0),
    ],
    ids=["gen6-full", "gen5-of-gen6", "platform-gen4"],
)
def test_the_maximum_is_the_device_and_system_one_never_a_fleet_constant(
    scrape, verdict
):
    """B300 links at Gen6; a platform offering Gen4 is not a downtrain."""

    tracker = PcieLinkDowntrainTracker()
    assert verdicts(tracker, scrape, scrape, scrape)[-1] == verdict


@pytest.mark.parametrize(
    "scrape",
    [
        link(6, 16),
        link(5, 32),
        link(0, 16),
        link(5, 16, max_gen=None),
        [
            sample
            for sample in link(5, 8)
            if sample.canonical_name != "gpu_utilization_percent"
        ],
    ],
    ids=[
        "current-above-max",
        "width-above-max",
        "zero",
        "max-missing",
        "no-utilization",
    ],
)
def test_an_invalid_sample_is_no_evidence_either_way(scrape):
    tracker = PcieLinkDowntrainTracker()
    assert verdicts(tracker, scrape, scrape, scrape) == [None, None, None]


def test_a_confirmed_downtrain_reaches_the_control_plane_as_a_finding():
    node = Node()
    downtrain = {"pcie_link_width_current": 8, "gpu_utilization_percent": 90}
    for _ in range(6):
        node.tick(downtrain)

    assert node.delivered() == [["initial-baseline"], ["candidate-confirmed"]]
    findings = node.sink.service.findings(CLUSTER, NODE)
    assert [item.canonical_name for item in findings] == ["pcie_link_downtrain"]
    assert findings[0].automatic_action == "RUN_DIAGNOSTICS"


def test_correctable_errors_are_an_edge_only_above_the_links_expected_rate():
    """Gen5 x16: DCGM expects up to 31 errors/min. 2 per 15 s is 8/min."""

    node = Node()
    total = 0
    node.tick({"pcie_correctable_error_total": total})
    for _ in range(4):
        total += 2
        node.tick({"pcie_correctable_error_total": total})
    assert node.delivered() == [["initial-baseline"]], "never on any change"

    for _ in range(3):
        total += 20  # 80/min
        node.tick({"pcie_correctable_error_total": total})

    # The first delivered window spans back to the baseline (diluted); the next
    # two are the collector's 15 s ticks above R: evidence, then the warning.
    assert node.delivered()[1:] == [["environmental-observation"]] * 3, (
        "delivered every tick while the rate holds, so two windows can confirm"
    )
    findings = {
        item.canonical_name: item for item in node.sink.service.findings(CLUSTER, NODE)
    }
    assert (
        findings["pcie_correctable_error_total"].automatic_action == "RUN_DIAGNOSTICS"
    )


def test_correctable_errors_without_the_current_link_are_never_a_candidate():
    node = Node()
    unknown = {"pcie_link_gen_current": 0, "pcie_link_width_current": 0}
    node.tick({"pcie_correctable_error_total": 0, **unknown})
    node.tick({"pcie_correctable_error_total": 1000, **unknown})

    assert node.delivered() == [["initial-baseline"]]


def test_brake_time_is_delivered_on_every_tick_while_it_accrues():
    node = Node()
    node.tick({"hw_power_brake_violation_total_us": 0})
    node.tick({"hw_power_brake_violation_total_us": 15_000})
    node.tick({"hw_power_brake_violation_total_us": 30_000})

    assert node.delivered() == [
        ["initial-baseline"],
        ["environmental-observation"],
        ["environmental-observation"],
    ]
    findings = node.sink.service.findings(CLUSTER, NODE)
    assert [item.canonical_name for item in findings] == [
        "hw_power_brake_violation_total_us"
    ]


def test_the_inforom_query_parses_valid_invalid_and_not_available():
    samples = parse_inforom_validation(
        "0, 00000000:59:00.0, GPU-a, valid\n"
        "1, 00000000:5A:00.0, GPU-b, invalid\n"
        "2, 00000000:72:00.0, GPU-c, N/A\n"
        "garbage\n"
    )

    assert [(item.gpu_uuid, item.pci_bdf, item.value) for item in samples] == [
        ("GPU-a", "00000000:59:00.0", 1.0),
        ("GPU-b", "00000000:5a:00.0", 0.0),
    ]
    assert samples[1].labels["inforom_checksum_validation"] == "invalid"
    assert "inforom.checksum_validation" in INFOROM_QUERY[1]


def test_the_inforom_probe_runs_at_its_interval_and_forgets_on_failure(caplog):
    calls: list[list[str]] = []
    answers = iter(
        [
            subprocess.CompletedProcess(
                [], 0, "0, 00000000:59:00.0, GPU-a, valid\n", ""
            ),
            subprocess.CompletedProcess([], 9, "", "driver wedged"),
        ]
    )

    def runner(command, **_kwargs):
        calls.append(command)
        return next(answers)

    probe = InforomValidityProbe(interval_seconds=60, logger=logging.getLogger("t"))
    at = START
    first = probe.refresh(at, runner)
    cached = probe.refresh(at + timedelta(seconds=15), runner)

    assert [(item.gpu_uuid, item.pci_bdf, item.value) for item in first] == [
        ("GPU-a", "00000000:59:00.0", 1.0)
    ]
    assert cached == first and len(calls) == 1, "once per interval"
    with caplog.at_level(logging.WARNING):
        failed = probe.refresh(at + timedelta(seconds=60), runner)
    assert failed == [], "a failed query keeps no stale answer"
    assert "driver wedged" in caplog.text
    assert len(calls) == 2

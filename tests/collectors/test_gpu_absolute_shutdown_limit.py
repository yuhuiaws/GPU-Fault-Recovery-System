"""The absolute shutdown limit of a T.Limit GPU comes from DCGM field 159 (batch 232).

On the HyperPod H200 (driver 580.178.04, DCGM 4.7.0) ``nvidia-smi -q -x`` prints
only T.Limit thresholds, so ``gpu_shutdown_temperature_c`` was never published
and the one-sample shutdown protection never fired. The device still reports
the absolute value: NVML's legacy threshold call and DCGM field 159 answered 95
on all 32 GPUs, and ``gpu_temp + gpu_temp_tlimit`` was 87 everywhere, so
87 - (-8) = 95. The collector publishes field 159 only where the XML has no
absolute shutdown and that arithmetic agrees within 1 C; only the emergency
reads it, the T.Limit rules keep grading the GPU.
"""

from __future__ import annotations

import logging
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from gpu_fault.channel_registry import GPU_METRICS_PATH
from gpu_fault.collectors.gpu import shutdown_limit as SHUTDOWN
from gpu_fault.collectors.gpu.dcgm import DCGM_METRICS, DcgmMetricsCollector
from gpu_fault.collectors.gpu.discovery import (
    SNAPSHOT_GPU_TEMP_LABEL,
    SNAPSHOT_TLIMIT_LABEL,
    query_nvidia_temperature_limits,
)
from gpu_fault.collectors.models import CollectorContext
from gpu_fault.gpu_metrics import (
    GpuMetricBatch,
    GpuMetricSample,
    GpuMetricsIngestionResult,
    GpuMetricsService,
    GpuMetricsThresholds,
)
from gpu_fault.gpu_temperature_policy import (
    DEVICE_SHUTDOWN_REASON,
    NVIDIA_DCGM_SHUTDOWN_FIELD_REFERENCE,
    NVIDIA_NVML_TEMPERATURE_REFERENCE,
    shutdown_from_device_field,
    temperature_decision,
)

ROOT = Path(__file__).resolve().parents[2]
NOW = datetime(2026, 11, 1, 9, 43, tzinfo=timezone.utc)
THRESHOLDS = GpuMetricsThresholds()
GPU = "GPU-4c7e0d2a-59"
#: GPU 0 of the first live HyperPod node (driver 580.178.04), in the DTD's tag
#: order: current 28 C, T.Limit 59 C, shutdown -8, slowdown -2, max operating 0.
H200_TEMPERATURE = (
    "<temperature>"
    "<gpu_temp>28 C</gpu_temp>"
    "<gpu_temp_tlimit>59 C</gpu_temp_tlimit>"
    "<gpu_temp_max_tlimit_threshold>-8 C</gpu_temp_max_tlimit_threshold>"
    "<gpu_temp_slow_tlimit_threshold>-2 C</gpu_temp_slow_tlimit_threshold>"
    "<gpu_temp_max_gpu_tlimit_threshold>0 C</gpu_temp_max_gpu_tlimit_threshold>"
    "<gpu_target_temperature>N/A</gpu_target_temperature>"
    "<memory_temp>29 C</memory_temp>"
    "<gpu_temp_max_mem_tlimit_threshold>0 C</gpu_temp_max_mem_tlimit_threshold>"
    "</temperature>"
)
#: A GPU without the T.Limit sensor: the absolute branch of every threshold.
ABSOLUTE_TEMPERATURE = (
    "<temperature>"
    "<gpu_temp>34 C</gpu_temp>"
    "<gpu_temp_tlimit>N/A</gpu_temp_tlimit>"
    "<gpu_temp_max_threshold>92 C</gpu_temp_max_threshold>"
    "<gpu_temp_slow_threshold>89 C</gpu_temp_slow_threshold>"
    "<gpu_temp_max_gpu_threshold>87 C</gpu_temp_max_gpu_threshold>"
    "<gpu_target_temperature>N/A</gpu_target_temperature>"
    "<memory_temp>38 C</memory_temp>"
    "<gpu_temp_max_mem_threshold>95 C</gpu_temp_max_mem_threshold>"
    "</temperature>"
)
H200_LIMITS = {
    "gpu_shutdown_tlimit_c": -8.0,
    "gpu_slowdown_tlimit_c": -2.0,
    "gpu_max_operating_tlimit_c": 0.0,
    "memory_max_operating_tlimit_c": 0.0,
}


def xml_limits(temperature: str = H200_TEMPERATURE) -> list[GpuMetricSample]:
    document = (
        "<nvidia_smi_log><gpu>"
        "<product_name>NVIDIA H200</product_name>"
        f"<uuid>{GPU}</uuid><minor_number>0</minor_number>"
        "<pci><pci_bus_id>00000000:59:00.0</pci_bus_id></pci>"
        f"{temperature}</gpu></nvidia_smi_log>"
    )

    def runner(argv, **_kwargs) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(argv, 0, stdout=document, stderr="")

    return query_nvidia_temperature_limits(runner)


def exporter(
    temperature: float,
    margin: float,
    shutdown: float | None = 95,
    *,
    field: str = "DCGM_FI_DEV_SHUTDOWN_TEMP",
    extra: str = "",
) -> str:
    labels = f'gpu="0",UUID="{GPU}",pci_bus_id="00000000:59:00.0"'
    lines = [
        f"DCGM_FI_DEV_GPU_TEMP{{{labels}}} {temperature}",
        f"DCGM_FI_DEV_GPU_TEMP_LIMIT{{{labels}}} {margin}",
        f"DCGM_FI_DEV_MEMORY_TEMP{{{labels}}} 29",
        f"DCGM_FI_DEV_POWER_MGMT_LIMIT{{{labels}}} 700",
    ]
    if shutdown is not None:
        lines.append(f"{field}{{{labels}}} {shutdown}")
    return "\n".join(lines) + "\n" + extra


def collector(sink=None, **options) -> DcgmMetricsCollector:
    return DcgmMetricsCollector(
        sink if sink is not None else IngestingSink(),
        CollectorContext(cluster_id="hyperpod"),
        node_id="hyperpod-node-a",
        health_summary_seconds=86_400,
        edge_confirmation_samples=3,
        **options,
    )


def by_name(batch: GpuMetricBatch) -> dict[str, list[GpuMetricSample]]:
    result: dict[str, list[GpuMetricSample]] = {}
    for sample in batch.samples:
        result.setdefault(sample.canonical_name, []).append(sample)
    return result


class IngestingSink:
    """The control plane behind the edge filter: every delivery is ingested."""

    def __init__(self) -> None:
        self.service = GpuMetricsService()
        self.results: list[GpuMetricsIngestionResult] = []

    def post(self, path: str, payload: dict) -> dict:
        assert path == GPU_METRICS_PATH, path
        result = self.service.ingest(GpuMetricBatch.model_validate(payload))
        self.results.append(result)
        return result.model_dump(mode="json")


def test_the_xml_snapshot_carries_its_reading_and_margin_on_the_shutdown_threshold() -> (
    None
):
    labels = {item.canonical_name: item.labels for item in xml_limits()}

    assert {
        key: labels["gpu_shutdown_tlimit_c"].get(key)
        for key in (SNAPSHOT_GPU_TEMP_LABEL, SNAPSHOT_TLIMIT_LABEL)
    } == {SNAPSHOT_GPU_TEMP_LABEL: "28", SNAPSHOT_TLIMIT_LABEL: "59"}
    assert all(
        SNAPSHOT_GPU_TEMP_LABEL not in item
        for name, item in labels.items()
        if name != "gpu_shutdown_tlimit_c"
    ), "only the shutdown T.Limit sample carries the snapshot"
    no_margin = xml_limits(H200_TEMPERATURE.replace("59 C", "N/A"))
    assert all(SNAPSHOT_TLIMIT_LABEL not in item.labels for item in no_margin), (
        "a snapshot whose current T.Limit is N/A carries no arithmetic input"
    )


@pytest.mark.parametrize(
    "field", ["DCGM_FI_DEV_SHUTDOWN_TEMP", "DCGM_FI_DEV_GPU_TEMP_SHUTDOWN_CELSIUS"]
)
def test_a_tlimit_gpu_publishes_field_159_when_its_tlimit_arithmetic_agrees(
    field: str,
) -> None:
    batch = collector().collect_text(
        exporter(28, 59, field=field), observed_at=NOW, extra_samples=xml_limits()
    )

    samples = by_name(batch)
    (published,) = samples["gpu_shutdown_temperature_c"]
    (raw,) = samples[SHUTDOWN.DCGM_SHUTDOWN_CANDIDATE]
    assert (published.value, published.unit, published.gpu_uuid) == (
        95.0,
        "celsius",
        GPU,
    )
    assert {
        key: published.labels.get(key)
        for key in (
            "threshold_source",
            "limit_source",
            "dcgm_field_id",
            SHUTDOWN.IMPLIED_LABEL,
        )
    } == {
        "threshold_source": "NVIDIA_DEVICE_LIMIT",
        "limit_source": field,
        "dcgm_field_id": "159",
        SHUTDOWN.IMPLIED_LABEL: "95",
    }, "(28 + 59) - (-8) = 95: the arithmetic of the live H200"
    assert raw.labels[SHUTDOWN.STATE_LABEL] == SHUTDOWN.PUBLISHED
    assert "gpu_slowdown_temperature_c" not in samples, (
        "no absolute slowdown is published: it would grade a second time"
    )
    assert "gpu_max_operating_temperature_c" not in samples


def test_an_absolute_shutdown_the_xml_reports_is_never_overridden() -> None:
    batch = collector().collect_text(
        exporter(34, 0, shutdown=95),
        observed_at=NOW,
        extra_samples=xml_limits(ABSOLUTE_TEMPERATURE),
    )

    samples = by_name(batch)
    (shutdown,) = samples["gpu_shutdown_temperature_c"]
    assert (shutdown.value, shutdown.labels["threshold_source"]) == (
        92.0,
        "nvidia-smi-xml",
    ), "the XML's 92 C stands; field 159's 95 C is not published beside it"
    (raw,) = samples[SHUTDOWN.DCGM_SHUTDOWN_CANDIDATE]
    assert raw.labels[SHUTDOWN.STATE_LABEL] == SHUTDOWN.XML_AUTHORITATIVE
    hot = GpuMetricSample(
        metric_name="DCGM_FI_DEV_GPU_TEMP",
        canonical_name="gpu_temperature_c",
        value=92,
        gpu_uuid=GPU,
    )
    decision = temperature_decision(
        THRESHOLDS, hot, {"gpu_shutdown_temperature_c": 92.0}
    )
    assert decision is not None, "92 C reaches the XML shutdown limit"
    assert (decision["reason"], decision["policy_reference"]) == (
        "GPU reached its absolute shutdown temperature",
        NVIDIA_NVML_TEMPERATURE_REFERENCE,
    ), "the XML-sourced emergency keeps its reason and reference"


@pytest.mark.parametrize(
    "temperature,reported,state,implied",
    [
        (H200_TEMPERATURE, 94, SHUTDOWN.PUBLISHED, "95"),
        (H200_TEMPERATURE, 96, SHUTDOWN.PUBLISHED, "95"),
        (H200_TEMPERATURE, 92, SHUTDOWN.MISMATCH, "95"),
        (H200_TEMPERATURE, 97, SHUTDOWN.MISMATCH, "95"),
        (H200_TEMPERATURE.replace("59 C", "N/A"), 95, SHUTDOWN.INPUT_MISSING, None),
        (H200_TEMPERATURE.replace("28 C", "N/A"), 95, SHUTDOWN.INPUT_MISSING, None),
        (H200_TEMPERATURE.replace("-8 C", "N/A"), 95, SHUTDOWN.INPUT_MISSING, None),
        (None, 95, SHUTDOWN.INPUT_MISSING, None),
    ],
    ids=[
        "minus-1",
        "plus-1",
        "minus-3",
        "plus-2",
        "margin-na",
        "reading-na",
        "no-shutdown-tlimit",
        "no-xml",
    ],
)
def test_a_mismatch_or_a_missing_input_withholds_the_limit_visibly(
    temperature: str | None, reported: float, state: str, implied: str | None
) -> None:
    batch = collector().collect_text(
        exporter(28, 59, shutdown=reported),
        observed_at=NOW,
        extra_samples=[] if temperature is None else xml_limits(temperature),
    )

    samples = by_name(batch)
    (raw,) = samples[SHUTDOWN.DCGM_SHUTDOWN_CANDIDATE]
    assert raw.value == reported
    assert raw.labels[SHUTDOWN.STATE_LABEL] == state
    assert raw.labels.get(SHUTDOWN.IMPLIED_LABEL) == implied
    published = [item.value for item in samples.get("gpu_shutdown_temperature_c", [])]
    assert published == ([reported] if state == SHUTDOWN.PUBLISHED else []), (
        f"{state}: a withheld limit is never published, 1 C of rounding is"
    )


@pytest.mark.parametrize("reported", [0, 151, 4294967295])
def test_implausible_values_are_withheld_and_blank_sentinels_dropped(
    reported: float,
) -> None:
    batch = collector().collect_text(
        exporter(28, 59, shutdown=reported), observed_at=NOW, extra_samples=xml_limits()
    )
    blank = collector().collect_text(
        exporter(28, 59, shutdown=9223372036854775794),
        observed_at=NOW,
        extra_samples=xml_limits(),
    )

    samples = by_name(batch)
    assert [
        item.labels[SHUTDOWN.STATE_LABEL]
        for item in samples[SHUTDOWN.DCGM_SHUTDOWN_CANDIDATE]
    ] == [SHUTDOWN.IMPLAUSIBLE]
    assert "gpu_shutdown_temperature_c" not in samples
    assert not {SHUTDOWN.DCGM_SHUTDOWN_CANDIDATE, "gpu_shutdown_temperature_c"} & set(
        by_name(blank)
    ), "DCGM's blank sentinel is no value at all"


def test_two_names_of_field_159_are_one_value_or_withheld() -> None:
    labels = f'gpu="0",UUID="{GPU}",pci_bus_id="00000000:59:00.0"'
    agreeing = collector().collect_text(
        exporter(
            28, 59, extra=f"DCGM_FI_DEV_GPU_TEMP_SHUTDOWN_CELSIUS{{{labels}}} 95\n"
        ),
        observed_at=NOW,
        extra_samples=xml_limits(),
    )
    disagreeing = collector().collect_text(
        exporter(
            28, 59, extra=f"DCGM_FI_DEV_GPU_TEMP_SHUTDOWN_CELSIUS{{{labels}}} 90\n"
        ),
        observed_at=NOW,
        extra_samples=xml_limits(),
    )

    assert [item.value for item in by_name(agreeing)["gpu_shutdown_temperature_c"]] == [
        95.0
    ], "an identical alias pair is deduplicated like fields 112 and 153"
    states = {
        item.labels[SHUTDOWN.STATE_LABEL]
        for item in by_name(disagreeing)[SHUTDOWN.DCGM_SHUTDOWN_CANDIDATE]
    }
    assert states == {SHUTDOWN.AMBIGUOUS}
    assert "gpu_shutdown_temperature_c" not in by_name(disagreeing)


def sample(name: str, value: float) -> GpuMetricSample:
    return GpuMetricSample(
        metric_name=name, canonical_name=name, value=value, gpu_uuid=GPU
    )


def grade(name: str, value: float, limits: dict[str, float]):
    decision = temperature_decision(THRESHOLDS, sample(name, value), limits)
    if decision is None:
        return None
    return (
        decision["severity"],
        decision["automatic_action"],
        decision["threshold_value"],
        decision["threshold_source"],
    )


@pytest.mark.parametrize("reading", [None, 59.0, 3.0, -3.0, -9.0])
def test_the_device_limit_adds_the_emergency_and_no_second_decision(
    reading: float | None,
) -> None:
    """Every grade but the emergency is the one the GPU had without the limit."""

    before = dict(H200_LIMITS)
    if reading is not None:
        before["gpu_tlimit_c"] = reading
    after = {**before, "gpu_shutdown_temperature_c": 95.0}

    assert shutdown_from_device_field(after) is True
    assert shutdown_from_device_field({"gpu_shutdown_temperature_c": 92.0}) is False
    for name, values in (
        ("gpu_temperature_c", (28, 85, 87, 90, 91, 92, 94, 94.9)),
        ("memory_temperature_c", (29, 90, 95, 99)),
        ("gpu_tlimit_c", (59, 3, -2, -9)),
    ):
        for value in values:
            assert grade(name, value, after) == grade(name, value, before), (
                f"{name}={value} with T.Limit reading {reading}: the field-159 "
                "limit changed a grade below the shutdown temperature"
            )
    for value in (95, 96, 120):
        decision = temperature_decision(
            THRESHOLDS, sample("gpu_temperature_c", value), after
        )
        assert decision is not None, value
        assert (
            decision["severity"],
            decision["automatic_action"],
            decision["threshold_value"],
            decision["threshold_source"],
            decision["reason"],
            decision["policy_reference"],
        ) == (
            "CRITICAL",
            "DRAIN",
            95.0,
            "NVIDIA_DEVICE_LIMIT",
            DEVICE_SHUTDOWN_REASON,
            NVIDIA_DCGM_SHUTDOWN_FIELD_REFERENCE,
        )


def test_the_emergency_fires_on_one_sample_at_the_dcgm_limit_and_not_below() -> None:
    sink = IngestingSink()
    node = collector(sink)
    limits = xml_limits()
    ticks = iter(range(100))

    def tick(temperature: float, margin: float, shutdown: float = 95) -> list:
        node.collect_text(
            exporter(temperature, margin, shutdown),
            observed_at=NOW + timedelta(seconds=15 * next(ticks)),
            extra_samples=limits,
        )
        return [
            item
            for item in sink.results[-1].new_findings
            if item.canonical_name == "gpu_temperature_c"
        ]

    assert tick(28, 59) == []
    assert tick(94, -7) == [], "one degree below the limit is the T.Limit rules' case"
    (finding,) = tick(95, -8)
    assert (
        finding.severity,
        finding.automatic_action,
        finding.confirmation_count,
        finding.value,
        finding.threshold_value,
        finding.threshold_source,
        finding.policy_reference,
    ) == (
        "CRITICAL",
        "DRAIN",
        1,
        95.0,
        95.0,
        "NVIDIA_DEVICE_LIMIT",
        NVIDIA_DCGM_SHUTDOWN_FIELD_REFERENCE,
    ), "one accepted sample at the device's 95 C protects, no three confirmations"
    assert "DCGM field 159" in finding.reason, finding.reason

    withheld = IngestingSink()
    # Every batch reaches the control plane: neither side may drain on it.
    other = collector(withheld, edge_filter_enabled=False)
    for index, temperature in enumerate((28, 96, 99)):
        other.collect_text(
            exporter(temperature, 59, shutdown=92),
            observed_at=NOW + timedelta(seconds=15 * index),
            extra_samples=limits,
        )
    assert len(withheld.results) == 3, "the filter is off: all three were graded"
    assert not [
        item
        for result in withheld.results
        for item in result.new_findings
        if item.canonical_name == "gpu_temperature_c"
    ], "a withheld (mismatched) limit never drains a GPU on one sample"


def test_each_gpu_says_once_why_its_limit_is_withheld(
    caplog: pytest.LogCaptureFixture,
) -> None:
    logger = logging.getLogger("tests.shutdown-limit")
    resolver = SHUTDOWN.ShutdownLimitResolver(logger)
    candidate = GpuMetricSample(
        metric_name="DCGM_FI_DEV_SHUTDOWN_TEMP",
        canonical_name=SHUTDOWN.DCGM_SHUTDOWN_CANDIDATE,
        value=92,
        unit="celsius",
        gpu_uuid=GPU,
    )

    with caplog.at_level(logging.INFO, logger=logger.name):
        for _ in range(5):
            resolver.resolve([candidate, *xml_limits()])
        for _ in range(5):
            resolver.resolve(
                [candidate.model_copy(update={"value": 95.0}), *xml_limits()]
            )

    assert [
        (record.levelname, SHUTDOWN.MISMATCH in record.getMessage())
        for record in caplog.records
    ] == [("WARNING", True), ("INFO", False)], [
        record.getMessage() for record in caplog.records
    ]
    assert "T.Limit arithmetic 95 C" in caplog.records[0].getMessage()


def test_the_vendor_evidence_is_pinned_in_code_and_docs() -> None:
    """Project rule: the NVIDIA text behind the source decision, in the repo."""

    assert NVIDIA_DCGM_SHUTDOWN_FIELD_REFERENCE == (
        "https://docs.nvidia.com/datacenter/dcgm/latest/dcgm-api/"
        "dcgm-api-field-ids.html#c.DCGM_FI_DEV_GPU_TEMP_SHUTDOWN_CELSIUS"
    )
    assert SHUTDOWN.NVIDIA_DCGM_SHUTDOWN_FIELD_QUOTE == (
        "Shutdown temperature for the device."
    )
    assert SHUTDOWN.NVIDIA_DCGM_SLOWDOWN_FIELD_QUOTE == (
        "Slowdown temperature for the device."
    )
    assert SHUTDOWN.NVIDIA_DCGM_RENAME_QUOTE == (
        "DCGM_FI_DEV_SHUTDOWN_TEMP to DCGM_FI_DEV_GPU_TEMP_SHUTDOWN_CELSIUS."
    )
    assert SHUTDOWN.NVML_THRESHOLD_REFERENCE.startswith(
        NVIDIA_NVML_TEMPERATURE_REFERENCE + "#"
    ), "the threshold API's anchor on the page the absolute findings cite"
    assert SHUTDOWN.NVML_THRESHOLD_NOTE.startswith(
        "This API is no longer the preferred interface for retrieving the "
        "following temperature thresholds on Ada and later architectures: "
        "NVML_TEMPERATURE_THRESHOLD_SHUTDOWN,"
    ), "NVML's note on the legacy call DCGM field 159 is read with"
    assert SHUTDOWN.NVML_SHUTDOWN_ENUM_QUOTE == (
        "Temperature at which the GPU will shut down for HW protection."
    )
    assert DCGM_METRICS["DCGM_FI_DEV_SHUTDOWN_TEMP"] == (
        SHUTDOWN.DCGM_SHUTDOWN_CANDIDATE,
        "celsius",
    )
    for document in (
        "docs/详细设计.md",
        "docs/en/detailed-design.md",
        "docs/故障类别与处置动作总表.md",
        "docs/en/fault-categories-and-actions.md",
    ):
        text = " ".join((ROOT / document).read_text(encoding="utf-8").split())
        for evidence in (
            "dcgm-api-field-ids.html#c.DCGM_FI_DEV_GPU_TEMP_SHUTDOWN_CELSIUS",
            SHUTDOWN.NVIDIA_DCGM_SHUTDOWN_FIELD_QUOTE,
            "group__nvmlDeviceQueries.html",
            "is no longer the preferred interface",
        ):
            assert evidence in text, f"{document} lacks {evidence!r}"

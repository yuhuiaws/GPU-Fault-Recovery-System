"""Driver 580 reports the temperature thresholds on the T.Limit scale (batch 134).

nvidia-smi 580.178.04 on the plain-EKS p5en (H200) prints
``<gpu_temp_slow_tlimit_threshold>-2 C</gpu_temp_slow_tlimit_threshold>`` where
older drivers, and GPUs without the T.Limit sensor, print the absolute
``<gpu_temp_slow_threshold>``. The parser knew only the absolute tags: it found
no limits on any H200, the probe logged a WARNING about every 64 s on each node
(28 per 30 minutes) and the GPUs were graded on the configured fallback.
NVIDIA defines those thresholds only relative to the current T.Limit, so they
grade the T.Limit reading (DCGM field 153 / ``temperature.gpu.tlimit``), never
an absolute temperature (gpu_temperature_policy.py has the vendor quotes).
"""

from __future__ import annotations

import logging
import subprocess
from datetime import datetime, timedelta, timezone

import pytest

from gpu_fault.channel_registry import GPU_METRICS_PATH
from gpu_fault.collectors.gpu.dcgm import DCGM_METRICS, DcgmMetricsCollector
from gpu_fault.collectors.gpu.discovery import query_nvidia_temperature_limits
from gpu_fault.collectors.gpu.inventory_cadence import TemperatureLimitProbe
from gpu_fault.collectors.gpu.nvidia_smi import NvidiaSmiMetricsCollector
from gpu_fault.collectors.models import CollectorContext
from gpu_fault.gpu_metrics import (
    GpuMetricBatch,
    GpuMetricsIngestionResult,
    GpuMetricsService,
)

from ._support import RecordingSink

NOW = datetime(2026, 9, 21, 6, 0, tzinfo=timezone.utc)
PROBE_LOGGER = "tests.tlimit-probe"

#: The first GPU's ``<temperature>`` element of ``nvidia-smi -q -x`` on the
#: plain-EKS p5en (H200 SXM, driver 580.178.04), verbatim, tabs included.
H200_580_TEMPERATURE = (
    "\t\t<temperature>\n"
    "\t\t\t<gpu_temp>29 C</gpu_temp>\n"
    "\t\t\t<gpu_temp_tlimit>58 C</gpu_temp_tlimit>\n"
    "\t\t\t<gpu_temp_max_tlimit_threshold>-8 C</gpu_temp_max_tlimit_threshold>\n"
    "\t\t\t<gpu_temp_slow_tlimit_threshold>-2 C</gpu_temp_slow_tlimit_threshold>\n"
    "\t\t\t<gpu_temp_max_gpu_tlimit_threshold>0 C</gpu_temp_max_gpu_tlimit_threshold>\n"
    "\t\t\t<gpu_target_temperature>N/A</gpu_target_temperature>\n"
    "\t\t\t<memory_temp>27 C</memory_temp>\n"
    "\t\t\t<gpu_temp_max_mem_tlimit_threshold>0 C</gpu_temp_max_mem_tlimit_threshold>\n"
    "\t\t</temperature>\n"
)
#: The absolute format of a GPU without the T.Limit sensor: the tags and their
#: order are the DTD that nvidia-smi 535.288.01 to 580.178.04 embeds, with the
#: absolute branch of every threshold; the values are illustrative.
ABSOLUTE_TEMPERATURE = (
    "\t\t<temperature>\n"
    "\t\t\t<gpu_temp>34 C</gpu_temp>\n"
    "\t\t\t<gpu_temp_tlimit>N/A</gpu_temp_tlimit>\n"
    "\t\t\t<gpu_temp_max_threshold>92 C</gpu_temp_max_threshold>\n"
    "\t\t\t<gpu_temp_slow_threshold>89 C</gpu_temp_slow_threshold>\n"
    "\t\t\t<gpu_temp_max_gpu_threshold>87 C</gpu_temp_max_gpu_threshold>\n"
    "\t\t\t<gpu_target_temperature>N/A</gpu_target_temperature>\n"
    "\t\t\t<memory_temp>38 C</memory_temp>\n"
    "\t\t\t<gpu_temp_max_mem_threshold>95 C</gpu_temp_max_mem_threshold>\n"
    "\t\t</temperature>\n"
)
H200_580_LIMITS = {
    "gpu_shutdown_tlimit_c": -8,
    "gpu_slowdown_tlimit_c": -2,
    "gpu_max_operating_tlimit_c": 0,
    "memory_max_operating_tlimit_c": 0,
}


def nvidia_smi_log(*temperatures: str) -> str:
    """An ``nvidia-smi -q -x`` document with one ``<gpu>`` per element.

    Only the fields the parser reads surround the elements; the bus ids and
    UUIDs are synthetic. An empty string is a GPU without ``<temperature>``.
    """

    gpus = "".join(
        f'\t<gpu id="00000000:{0x19 + index:02X}:00.0">\n'
        "\t\t<product_name>NVIDIA H200</product_name>\n"
        f"\t\t<uuid>GPU-{index}</uuid>\n"
        f"\t\t<minor_number>{index}</minor_number>\n"
        f"\t\t<pci>\n\t\t\t<pci_bus_id>00000000:{0x19 + index:02X}:00.0"
        "</pci_bus_id>\n\t\t</pci>\n"
        f"{temperature}\t</gpu>\n"
        for index, temperature in enumerate(temperatures)
    )
    return (
        '<?xml version="1.0" ?>\n<nvidia_smi_log>\n'
        "\t<driver_version>580.178.04</driver_version>\n"
        f"\t<attached_gpus>{len(temperatures)}</attached_gpus>\n"
        f"{gpus}</nvidia_smi_log>\n"
    )


def xml_runner(xml: str, calls: list[list[str]] | None = None):
    def runner(argv, **_kwargs) -> subprocess.CompletedProcess[str]:
        if calls is not None:
            calls.append(list(argv))
        return subprocess.CompletedProcess(argv, 0, stdout=xml, stderr="")

    return runner


def limits_by_gpu(xml: str) -> dict[str | None, dict[str, float]]:
    result: dict[str | None, dict[str, float]] = {}
    for item in query_nvidia_temperature_limits(xml_runner(xml)):
        result.setdefault(item.gpu_uuid, {})[item.canonical_name] = item.value
    return result


def test_driver_580_xml_yields_the_tlimit_thresholds_signed() -> None:
    samples = query_nvidia_temperature_limits(
        xml_runner(nvidia_smi_log(H200_580_TEMPERATURE))
    )

    assert {item.canonical_name: item.value for item in samples} == H200_580_LIMITS, (
        "the 580 T.Limit thresholds must parse, 0 and negative values included, "
        "and the current reading (58) is not a limit"
    )
    assert {
        (item.gpu_uuid, item.gpu_index, item.pci_bdf, item.unit) for item in samples
    } == {("GPU-0", "0", "00000000:19:00.0", "celsius")}
    assert {item.labels["threshold_source"] for item in samples} == {"nvidia-smi-xml"}


def test_the_absolute_format_keeps_its_absolute_names() -> None:
    assert limits_by_gpu(nvidia_smi_log(ABSOLUTE_TEMPERATURE)) == {
        "GPU-0": {
            "gpu_shutdown_temperature_c": 92,
            "gpu_slowdown_temperature_c": 89,
            "gpu_max_operating_temperature_c": 87,
            "memory_max_operating_temperature_c": 95,
        }
    }


def test_each_gpu_keeps_the_scale_its_driver_printed() -> None:
    absolute = limits_by_gpu(nvidia_smi_log(ABSOLUTE_TEMPERATURE))["GPU-0"]

    assert limits_by_gpu(
        nvidia_smi_log(H200_580_TEMPERATURE, ABSOLUTE_TEMPERATURE, "")
    ) == {"GPU-0": H200_580_LIMITS, "GPU-1": absolute}, (
        "a T.Limit GPU, an absolute GPU and a GPU without <temperature> on one "
        "node: each answers on its own scale, the third answers nothing"
    )


@pytest.mark.parametrize(
    "tag,raw,expected",
    [
        ("gpu_temp_slow_tlimit_threshold", "-2 C", -2.0),
        ("gpu_temp_slow_tlimit_threshold", "0 C", 0.0),
        ("gpu_temp_slow_tlimit_threshold", "N/A", None),
        ("gpu_temp_slow_tlimit_threshold", "", None),
        # A wrapped -2 and the int32 minimum are no temperature margins.
        ("gpu_temp_slow_tlimit_threshold", "4294967294 C", None),
        ("gpu_temp_slow_tlimit_threshold", "-2147483648 C", None),
        ("gpu_temp_slow_threshold", "89 C", 89.0),
        ("gpu_temp_slow_threshold", "0 C", None),
        ("gpu_temp_slow_threshold", "-2 C", None),
    ],
)
def test_missing_na_and_implausible_values_are_absent_never_zero(
    tag: str, raw: str, expected: float | None
) -> None:
    element = f"\t\t<temperature>\n\t\t\t<{tag}>{raw}</{tag}>\n\t\t</temperature>\n"

    values = [
        item.value
        for item in query_nvidia_temperature_limits(xml_runner(nvidia_smi_log(element)))
    ]

    assert values == ([] if expected is None else [expected])


def test_a_580_answer_is_a_success_and_the_probe_stays_silent(
    caplog: pytest.LogCaptureFixture,
) -> None:
    calls: list[list[str]] = []
    runner = xml_runner(nvidia_smi_log(H200_580_TEMPERATURE), calls)
    probe = TemperatureLimitProbe(
        retry_interval_seconds=60, logger=logging.getLogger(PROBE_LOGGER)
    )

    with caplog.at_level(logging.DEBUG, logger=PROBE_LOGGER):
        first = probe.refresh(NOW, runner)
        later = [
            probe.refresh(NOW + timedelta(seconds=15 * tick), runner)
            for tick in range(1, 120)
        ]

    assert {item.canonical_name: item.value for item in first} == H200_580_LIMITS
    assert all(item == first for item in later), "the limits are cached once found"
    assert len(calls) == 1, f"a found limit set is never probed again: {calls}"
    assert probe.reports_tlimit is True
    assert (probe.failures, probe.next_attempt_at) == (0, None)
    assert [record.getMessage() for record in caplog.records] == [], (
        "a node on driver 580 logged about its temperature limits; it used to "
        "warn 28 times per 30 minutes"
    )


def test_a_permanent_failure_backs_off_exponentially_and_warns_once(
    caplog: pytest.LogCaptureFixture,
) -> None:
    probe = TemperatureLimitProbe(
        retry_interval_seconds=60,
        logger=logging.getLogger(PROBE_LOGGER),
        retry_cap_seconds=600,
    )
    no_thresholds = xml_runner(nvidia_smi_log(""))
    probed_at: list[int] = []

    with caplog.at_level(logging.DEBUG, logger=PROBE_LOGGER):
        for tick in range(200):
            failures = probe.failures
            probe.refresh(NOW + timedelta(seconds=15 * tick), no_thresholds)
            if probe.failures != failures:
                probed_at.append(15 * tick)

    assert probed_at == [0, 15, 30, 90, 210, 450, 930, 1530, 2130, 2730], (
        "three quick probes, then the inventory interval doubling to the cap: "
        f"{probed_at}"
    )
    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert len(warnings) == 1, (
        "one WARNING per reason, not one per probe: "
        f"{[record.getMessage() for record in warnings]}"
    )
    assert "nvidia-smi reported no temperature thresholds" in warnings[0].getMessage()
    assert sum(r.levelno == logging.DEBUG for r in caplog.records) == 9


def test_a_new_failure_reason_warns_again_and_late_limits_are_announced(
    caplog: pytest.LogCaptureFixture,
) -> None:
    probe = TemperatureLimitProbe(
        retry_interval_seconds=60, logger=logging.getLogger(PROBE_LOGGER)
    )

    def wedged(argv, **_kwargs) -> subprocess.CompletedProcess[str]:
        raise subprocess.TimeoutExpired(list(argv), timeout=15)

    with caplog.at_level(logging.DEBUG, logger=PROBE_LOGGER):
        probe.refresh(NOW, xml_runner(nvidia_smi_log("")))
        probe.refresh(NOW + timedelta(seconds=15), wedged)
        probe.refresh(NOW + timedelta(seconds=30), wedged)
        found = probe.refresh(
            NOW + timedelta(seconds=90),
            xml_runner(nvidia_smi_log(H200_580_TEMPERATURE)),
        )

    levels = [(record.levelname, record.getMessage()[:45]) for record in caplog.records]
    assert [level for level, _message in levels] == [
        "WARNING",
        "WARNING",
        "DEBUG",
        "INFO",
    ], f"a changed reason warns once more, a repeat is DEBUG: {levels}"
    assert "available after 3 failed probe(s)" in caplog.records[-1].getMessage()
    assert {item.canonical_name for item in found} == set(H200_580_LIMITS)
    assert (probe.failures, probe.next_attempt_at, probe.last_failure) == (
        0,
        None,
        None,
    )


def test_dcgm_reads_the_tlimit_margin_as_a_signed_sample() -> None:
    collector = DcgmMetricsCollector(
        RecordingSink(),
        CollectorContext(cluster_id="tlimit"),
        node_id="node-a",
        edge_filter_enabled=False,
    )

    batch = collector.collect_text(
        'DCGM_FI_DEV_GPU_TEMP{gpu="0",UUID="GPU-0"} 29\n'
        'DCGM_FI_DEV_GPU_TEMP_LIMIT{gpu="0",UUID="GPU-0"} -3\n',
        observed_at=NOW,
    )

    assert {
        (item.canonical_name, item.value, item.unit, item.gpu_uuid)
        for item in batch.samples
    } == {
        ("gpu_temperature_c", 29.0, "celsius", "GPU-0"),
        ("gpu_tlimit_c", -3.0, "celsius", "GPU-0"),
    }


class IngestingSink:
    """The control plane behind the edge filter: every delivery is ingested."""

    def __init__(self) -> None:
        self.service = GpuMetricsService()
        self.results: list[GpuMetricsIngestionResult] = []

    def post(self, path: str, payload: dict) -> dict:
        assert path == GPU_METRICS_PATH, "temperature stays on the GPU metrics path"
        result = self.service.ingest(GpuMetricBatch.model_validate(payload))
        self.results.append(result)
        return result.model_dump(mode="json")


def exporter_text(values: dict[str, float]) -> str:
    defaults = {
        "gpu_temperature_c": 40,
        "memory_temperature_c": 45,
        "power_limit_w": 700,
        "gpu_tlimit_c": 47,
    }
    return "".join(
        f'{name}{{gpu="0",UUID="GPU-0"}} '
        f"{values.get(canonical, defaults.get(canonical, 0))}\n"
        for name, (canonical, _unit) in DCGM_METRICS.items()
    )


def test_dcgm_grades_the_margin_on_both_sides_and_not_the_fallback() -> None:
    """Producer edge and ingestion agree; the absolute reading is not graded.

    85 C and 91 C cross the configured fallback (85/90): before the fix both
    raised GPU temperature findings on boundaries that are not the GPU's.
    """

    sink = IngestingSink()
    collector = DcgmMetricsCollector(
        sink,
        CollectorContext(cluster_id="tlimit"),
        node_id="node-a",
        health_summary_seconds=86_400,
        edge_confirmation_samples=3,
    )
    limits = query_nvidia_temperature_limits(
        xml_runner(nvidia_smi_log(H200_580_TEMPERATURE))
    )
    ticks = iter(range(100))

    def tick(values: dict[str, float]) -> None:
        collector.collect_text(
            exporter_text(values),
            observed_at=NOW + timedelta(seconds=15 * next(ticks)),
            extra_samples=limits,
        )

    tick({})
    assert [result.new_findings for result in sink.results] == [[]]
    for _ in range(3):
        tick({"gpu_temperature_c": 85, "gpu_tlimit_c": 2})
    for _ in range(3):
        tick({"gpu_temperature_c": 91, "gpu_tlimit_c": -3})

    graded = [
        (
            item.canonical_name,
            item.severity,
            item.automatic_action,
            item.threshold_value,
            item.threshold_source,
        )
        for result in sink.results
        for item in result.new_findings
    ]
    assert graded == [
        ("gpu_tlimit_c", "WARNING", "RUN_DIAGNOSTICS", 3.0, "NVIDIA_DEVICE_TLIMIT"),
        ("gpu_tlimit_c", "CRITICAL", "DRAIN", -2.0, "NVIDIA_DEVICE_TLIMIT"),
    ], "slowdown -2 C is critical, 5 C before it (and max operating 0 C) warns"
    assert len(sink.results) == 3, "baseline plus one delivery per confirmed action"


def fallback_runner(xml: str, *, refuse_tlimit: bool = False):
    """nvidia-smi for the fallback collector: one H200, XML limits, CSV rows."""

    queries: list[str] = []
    values = {"temperature.gpu": "29", "temperature.gpu.tlimit": "-3"}

    def runner(argv, **_kwargs) -> subprocess.CompletedProcess[str]:
        if "-q" in argv:
            return subprocess.CompletedProcess(argv, 0, stdout=xml, stderr="")
        fields = next(
            item for item in argv if item.startswith("--query-gpu=")
        ).removeprefix("--query-gpu=")
        queries.append(fields)
        if refuse_tlimit and "temperature.gpu.tlimit" in fields.split(","):
            return subprocess.CompletedProcess(
                argv,
                2,
                stdout="",
                stderr='Field "temperature.gpu.tlimit" is not a valid field to query.',
            )
        identity = {
            "index": "0",
            "uuid": "GPU-0",
            "pci.bus_id": "00000000:19:00.0",
            "name": "NVIDIA H200",
        }
        row = ", ".join(
            identity.get(field, values.get(field, "0")) for field in fields.split(",")
        )
        return subprocess.CompletedProcess(argv, 0, stdout=row + "\n", stderr="")

    runner.queries = queries  # type: ignore[attr-defined]
    return runner


def _fallback_collector(
    monkeypatch: pytest.MonkeyPatch, tmp_path, runner
) -> NvidiaSmiMetricsCollector:
    monkeypatch.delenv("GPU_FAULT_EXPECTED_GPU_COUNT", raising=False)
    monkeypatch.delenv("GPU_FAULT_NODE_INSTANCE_TYPE", raising=False)
    boot_id = tmp_path / "boot_id"
    boot_id.write_text("boot-a\n", encoding="ascii")
    monkeypatch.setenv("GPU_FAULT_BOOT_ID_PATH", str(boot_id))
    clock = iter(NOW + timedelta(seconds=step) for step in range(1000))
    return NvidiaSmiMetricsCollector(
        RecordingSink(),
        CollectorContext(cluster_id="tlimit", product="H200", driver_branch=580),
        node_id="node-a",
        inventory_interval_seconds=3600,
        now=lambda: next(clock),
        runner=runner,
    )


def test_fallback_mode_reads_the_tlimit_once_the_limits_are_on_that_scale(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    runner = fallback_runner(nvidia_smi_log(H200_580_TEMPERATURE))
    collector = _fallback_collector(monkeypatch, tmp_path, runner)

    first = collector.collect_once()
    second = collector.collect_once()

    metric_queries = [fields for fields in runner.queries if "power.draw" in fields]
    assert ["temperature.gpu.tlimit" in fields for fields in metric_queries] == [
        False,
        True,
    ], "the field is asked only after the probe found T.Limit thresholds"
    assert "gpu_tlimit_c" not in {item.canonical_name for item in first.samples}
    readings = {
        item.canonical_name: item.value
        for item in second.samples
        if item.canonical_name in {"gpu_tlimit_c", "gpu_temperature_c"}
    }
    assert readings == {"gpu_temperature_c": 29.0, "gpu_tlimit_c": -3.0}


def test_fallback_mode_drops_a_refused_tlimit_field_after_one_round(
    monkeypatch: pytest.MonkeyPatch, tmp_path, caplog: pytest.LogCaptureFixture
) -> None:
    runner = fallback_runner(nvidia_smi_log(H200_580_TEMPERATURE), refuse_tlimit=True)
    collector = _fallback_collector(monkeypatch, tmp_path, runner)

    with caplog.at_level(logging.WARNING, logger="gpu_fault.collectors.gpu.nvidia_smi"):
        batches = [collector.collect_once() for _ in range(4)]

    asked = [fields for fields in runner.queries if "temperature.gpu.tlimit" in fields]
    assert len(asked) == 2, (
        f"one merged query and one query of the field alone, then never again: {asked}"
    )
    refusals = [
        record.getMessage()
        for record in caplog.records
        if "refused temperature.gpu.tlimit" in record.getMessage()
    ]
    assert len(refusals) == 1, refusals
    assert all(
        "gpu_temperature_c" in {item.canonical_name for item in batch.samples}
        for batch in batches
    ), "a refused T.Limit must not cost the round its other metrics"

"""GPU probe parsing and producer lifecycles over fake command responses."""

from __future__ import annotations

import logging
import subprocess
from types import SimpleNamespace

import pytest

from gpu_fault.collectors import context as context_module
from gpu_fault.collectors.gpu import discovery, nvidia_smi
from gpu_fault.collectors.gpu.inventory_cadence import (
    GpuInventoryCadence,
    TemperatureLimitProbe,
)
from gpu_fault.collectors.sinks import CollectorError
from tests.collectors import _cov95_runtime_collect as support
from tests.collectors._cov95_runtime_collect_gpu import gpu_runner

isolated_runtime = support.isolated_runtime


def command_result(stdout="", *, code=0, stderr=""):
    calls = []

    def runner(argv, **kwargs):
        calls.append((argv, kwargs))
        return subprocess.CompletedProcess(argv, code, stdout=stdout, stderr=stderr)

    return runner, calls


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("\n , , \n", "no visible GPUs"),
        ("0,GPU-a", "column count"),
        ("0,,H100", "incomplete GPU identity"),
        ("0,GPU-a,H100\n1,GPU-a,H100\n", "duplicate GPU UUIDs"),
        ("0,GPU-a,Unknown\n", "unsupported NVIDIA GPU product"),
    ],
)
def test_gpu_identity_refuses_incomplete_or_ambiguous_command_output(text, message):
    runner, calls = command_result(text)
    with pytest.raises(CollectorError, match=message):
        discovery.discover_gpu_product(runner)
    assert len(calls) == 1
    assert calls[0][1]["timeout"] == 15


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("\n , , , \n", "no GPU inventory"),
        ("0,GPU-a,H100", "column count"),
        ("bad,GPU-a,0000:01:00.0,H100", "incomplete GPU inventory"),
        ("0,,0000:01:00.0,H100", "incomplete GPU inventory"),
        ("0,GPU-a,0000:01:00.0,H100\n1,GPU-a,0000:02:00.0,H100", "failed validation"),
    ],
)
def test_inventory_parser_never_delivers_partial_identity(text, message):
    runner, _ = command_result(text)
    with pytest.raises(CollectorError, match=message):
        discovery.query_gpu_inventory(runner)


@pytest.mark.parametrize("text", ["broken", ""])
def test_unusable_driver_versions_are_reported_as_collector_errors(text):
    runner, _ = command_result(text)
    with pytest.raises(CollectorError, match="driver version"):
        discovery.discover_gpu_software_versions(runner)


@pytest.mark.parametrize(
    "summary", [OSError("summary unavailable"), ("", 1), ("no CUDA version", 0)]
)
def test_driver_identity_survives_an_unavailable_optional_cuda_summary(summary):
    calls = []

    def runner(argv, **kwargs):
        calls.append(argv)
        if len(argv) > 1:
            return subprocess.CompletedProcess(
                argv, 0, stdout="575.20\n575.21\n", stderr=""
            )
        if isinstance(summary, OSError):
            raise summary
        return subprocess.CompletedProcess(
            argv, summary[1], stdout=summary[0], stderr=""
        )

    assert discovery.discover_gpu_software_versions(runner) == (575, None)
    assert calls[-1] == ["nvidia-smi"]


@pytest.mark.parametrize("raw", ["not supported", "-3", "0", "9" * 400])
def test_temperature_probe_discards_unusable_thresholds_without_fabricating_values(raw):
    runner, _ = command_result(
        "<nvidia_smi_log><gpu><temperature>"
        f"<gpu_temp_slow_threshold>{raw}</gpu_temp_slow_threshold>"
        "</temperature></gpu><gpu/></nvidia_smi_log>"
    )
    assert discovery.query_nvidia_temperature_limits(runner) == []


def test_invalid_temperature_xml_and_empty_boot_identity_fail_closed(
    monkeypatch, tmp_path
):
    runner, _ = command_result("<broken")
    with pytest.raises(CollectorError, match="invalid XML"):
        discovery.query_nvidia_temperature_limits(runner)
    path = tmp_path / "empty-boot"
    path.write_text("")
    monkeypatch.setenv("GPU_FAULT_BOOT_ID_PATH", str(path))
    with pytest.raises(CollectorError, match="boot ID is empty"):
        discovery.read_host_boot_id()


@pytest.mark.parametrize("mode", ["auto", "required", "disabled", "invalid"])
@pytest.mark.parametrize("configured_driver", [None, "575"])
def test_context_discovery_policy_preserves_required_identity_and_optional_fallback(
    monkeypatch, mode, configured_driver
):
    monkeypatch.setenv("GPU_FAULT_CLUSTER_ID", "cluster-a")
    monkeypatch.setenv("GPU_FAULT_GPU_PRODUCT", "H100")
    monkeypatch.setenv("GPU_FAULT_GPU_PRODUCT_DISCOVERY", mode)
    if configured_driver:
        monkeypatch.setenv("GPU_FAULT_DRIVER_BRANCH", configured_driver)
    calls = []

    def runner(argv, **kwargs):
        calls.append(argv)
        if "index,uuid,name" in argv[1]:
            return subprocess.CompletedProcess(
                argv, 0, stdout="0,GPU-a,H100\n", stderr=""
            )
        raise FileNotFoundError("fake software query unavailable")

    if mode == "invalid" or (mode == "required" and configured_driver is None):
        with pytest.raises(CollectorError):
            context_module.context_from_environment(
                discover_product=True, runner=runner
            )
    else:
        context = context_module.context_from_environment(
            discover_product=True, runner=runner
        )
        assert context.product == "H100"
        assert context.driver_branch == (575 if configured_driver else None)
    if mode in {"invalid", "disabled"}:
        assert calls == []


def test_configured_driver_mismatch_refuses_the_context_before_delivery(monkeypatch):
    monkeypatch.setenv("GPU_FAULT_CLUSTER_ID", "cluster-a")
    monkeypatch.setenv("GPU_FAULT_DRIVER_BRANCH", "570")

    def runner(argv, **kwargs):
        text = (
            "0,GPU-a,H100\n"
            if "index,uuid,name" in " ".join(argv)
            else "575.22\n"
            if len(argv) > 1
            else "CUDA Version: 12.9"
        )
        return subprocess.CompletedProcess(argv, 0, stdout=text, stderr="")

    with pytest.raises(CollectorError, match="configured NVIDIA driver branch"):
        context_module.context_from_environment(discover_product=True, runner=runner)


@pytest.mark.parametrize(
    "raw,value",
    [
        ("yes", 1),
        ("true", 1),
        ("no", 0),
        ("false", 0),
        ("bad", None),
        ("[Not Supported]", None),
    ],
)
def test_nvidia_csv_accepts_boolean_health_flags_and_skips_unavailable_values(
    raw, value
):
    fields = ["index", "uuid", "name", "pci.bus_id", "remapped_rows.pending"]
    metrics = {"remapped_rows.pending": ("row_remap_pending", None)}
    samples = nvidia_smi.NvidiaSmiMetricsCollector.parse_csv(
        f"\n , , , , \n0,GPU-a,H100,0000:01:00.0,{raw}\n", fields, metrics
    )
    assert [sample.value for sample in samples] == ([] if value is None else [value])
    with pytest.raises(CollectorError, match="column count"):
        nvidia_smi.NvidiaSmiMetricsCollector.parse_csv("0,GPU-a", fields, metrics)


def test_nvidia_default_run_consumes_force_request_only_after_real_collection(
    monkeypatch, tmp_path
):
    sink = support.RecordingSink()
    marker = tmp_path / "gpu.request"
    marker.write_text("requested")
    collector = nvidia_smi.NvidiaSmiMetricsCollector(
        sink,
        support.collector_context(),
        node_id="node-a",
        force_snapshot_path=str(marker),
        now=lambda: support.NOW,
        runner=gpu_runner,
    )

    def sleep(seconds):
        assert seconds == 30
        raise support.StopLoop

    monkeypatch.setattr(nvidia_smi, "time", SimpleNamespace(sleep=sleep))
    with pytest.raises(support.StopLoop):
        collector.run()
    assert not marker.exists(), (
        "successful collection must consume its own force request"
    )
    assert [path for path, _ in sink.requests] == [
        "/v1/collector-events/gpu-inventory",
        "/v1/collector-events/gpu-metrics",
    ]
    assert sink.requests[-1][1]["samples"], (
        "the successful metrics batch must contain real parsed samples"
    )


@pytest.mark.parametrize(
    "options",
    [
        {"inventory_interval_seconds": 0},
        {"startup_spread_seconds": 0},
        {"failure_backoff_threshold": 0},
    ],
)
def test_nvidia_sampling_limits_must_remain_positive(options):
    with pytest.raises(ValueError, match="positive"):
        nvidia_smi.NvidiaSmiMetricsCollector(
            support.RecordingSink(),
            support.collector_context(),
            node_id="node-a",
            runner=gpu_runner,
            **options,
        )


def test_temperature_probe_caches_success_and_inventory_rejects_zero_cadence():
    calls = []

    def runner(argv, **kwargs):
        calls.append(argv)
        return gpu_runner(argv, **kwargs)

    probe = TemperatureLimitProbe(
        retry_interval_seconds=60, logger=logging.getLogger(__name__)
    )
    first = probe.refresh(support.NOW, runner)
    assert len(first) == 1
    assert probe.refresh(support.NOW, runner) == first
    assert len(calls) == 1
    with pytest.raises(ValueError, match="interval must be positive"):
        GpuInventoryCadence(interval_seconds=0, logger=logging.getLogger(__name__))

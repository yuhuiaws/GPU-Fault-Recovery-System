"""A collector that started before the driver stamps the product once it answers.

Live (eks-managed, DESTR-015 attempt 10 after the DESTR-025 reboot of one
node): the kernel collector came up before nvidia-smi could answer, started
without a product (auto mode, nothing configured) and stamped ``product=None``
on the injected XID 46; the policy quarantined the node as "not applicable to
product UNKNOWN" while the sibling node's XID started the reset workflow.
"""

from __future__ import annotations

import subprocess
from datetime import timedelta

import pytest

from gpu_fault.collectors import identity_refresh
from gpu_fault.collectors.context import GPU_IDENTITY_REFRESH_INTERVAL_SECONDS
from gpu_fault.collectors.gpu.discovery import (
    discover_gpu_product,
    discover_gpu_software_versions,
)
from gpu_fault.collectors.identity_refresh import GpuIdentityRefresher

from ._support import (
    NOW,
    CollectorContext,
    KernelLogCollector,
    NvidiaSmiMetricsCollector,
    RecordingSink,
    completed_nvidia_smi,
)

XID_LINE = (
    "3,1001,5000000,-;NVRM: Xid (PCI:0000:b9:00): 46, pid=4242, "
    "name=python, Ch 00000008\n"
)


class DriverComesUp:
    """nvidia-smi that fails ``failures`` times, then answers as an H200 node."""

    def __init__(self, failures: int = 1) -> None:
        self.failures = failures
        self.calls: list[list[str]] = []

    def __call__(self, command, **_kwargs) -> subprocess.CompletedProcess[str]:
        self.calls.append(list(command))
        if self.failures > 0:
            self.failures -= 1
            return completed_nvidia_smi(
                "", returncode=9, stderr="NVIDIA-SMI has failed: driver not loaded"
            )
        # Exact argv matches: the metrics round's query starts with the same
        # identity fields and must not be answered as the discovery query.
        if "--query-gpu=index,uuid,name" in command:
            return completed_nvidia_smi(
                "0, GPU-a, NVIDIA H200\n1, GPU-b, NVIDIA H200\n"
            )
        if "--query-gpu=driver_version" in command:
            return completed_nvidia_smi("580.65.06\n580.65.06\n")
        if command == ["nvidia-smi"]:
            return completed_nvidia_smi("| NVIDIA-SMI 580.65.06  CUDA Version: 13.0 |")
        if "--query-gpu=index,uuid,pci.bus_id,name" in command:
            return completed_nvidia_smi("0, GPU-a, 00000000:B9:00.0, NVIDIA H200\n")
        if any(str(item).startswith("--query-gpu=") for item in command):
            # The metrics round: one row answering exactly the fields asked for.
            fields = next(
                str(item)[len("--query-gpu=") :]
                for item in command
                if str(item).startswith("--query-gpu=")
            ).split(",")
            values = {
                "index": "0",
                "uuid": "GPU-a",
                "name": "NVIDIA H200",
                "pci.bus_id": "00000000:B9:00.0",
            }
            return completed_nvidia_smi(
                ", ".join(values.get(field, "0") for field in fields) + "\n"
            )
        return completed_nvidia_smi("", returncode=1, stderr="unsupported query")


def _productless() -> CollectorContext:
    return CollectorContext(cluster_id="hp-cluster", runtime_profile_version="v1")


def test_refresher_is_inert_when_the_product_is_known_or_discovery_disabled() -> None:
    runner = DriverComesUp(failures=0)
    known = GpuIdentityRefresher(
        CollectorContext(cluster_id="c", product="H100"), runner=runner
    )
    assert known.pending is False
    assert known.refresh(NOW).product == "H100"
    disabled = GpuIdentityRefresher(
        _productless(), runner=runner, discovery_mode="disabled"
    )
    assert disabled.pending is False
    assert disabled.refresh(NOW).product is None
    assert runner.calls == [], "neither may shell out to nvidia-smi"


def test_refresher_retries_once_per_interval_until_the_driver_answers() -> None:
    runner = DriverComesUp(failures=1)
    refresher = GpuIdentityRefresher(
        _productless(), runner=runner, discovery_mode="auto"
    )

    assert refresher.refresh(NOW).product is None, "the first attempt fails"
    assert refresher.refresh(NOW + timedelta(seconds=5)).product is None
    assert len(runner.calls) == 1, "a failed attempt is not retried inside the interval"

    later = NOW + timedelta(seconds=GPU_IDENTITY_REFRESH_INTERVAL_SECONDS)
    context = refresher.refresh(later)
    assert (context.product, context.driver_branch, context.cuda_version) == (
        "H200",
        580,
        "13.0",
    ), "the second attempt fills product and the unconfigured versions"
    assert refresher.discovered_at == later
    assert refresher.attempts == 2
    assert refresher.pending is False
    calls_after = len(runner.calls)
    assert refresher.refresh(later + timedelta(hours=1)) is context
    assert len(runner.calls) == calls_after, "a known product is never re-queried"


def test_refresher_keeps_configured_versions_and_survives_discovery_errors() -> None:
    def mixed(command, **_kwargs) -> subprocess.CompletedProcess[str]:
        if "--query-gpu=index,uuid,name" in " ".join(command):
            return completed_nvidia_smi(
                "0, GPU-a, NVIDIA H200\n1, GPU-b, NVIDIA B200\n"
            )
        raise AssertionError("only the identity query is expected")

    configured = CollectorContext(
        cluster_id="c", driver_branch=570, cuda_version="12.8"
    )
    refresher = GpuIdentityRefresher(configured, runner=mixed, discovery_mode="auto")
    assert refresher.refresh(NOW).product is None, (
        "mixed products are a CollectorError: logged, nothing stamped"
    )

    def broken(command, **_kwargs) -> subprocess.CompletedProcess[str]:
        raise RuntimeError("fake runner exploded")

    exploding = GpuIdentityRefresher(configured, runner=broken, discovery_mode="auto")
    assert exploding.refresh(NOW).product is None, (
        "an unexpected error must not escape onto the collector's read loop"
    )

    runner = DriverComesUp(failures=0)
    kept = GpuIdentityRefresher(configured, runner=runner, discovery_mode="auto")
    context = kept.refresh(NOW)
    assert (context.product, context.driver_branch, context.cuda_version) == (
        "H200",
        570,
        "12.8",
    ), "configured driver/CUDA versions are kept; only the product was missing"
    assert not any("driver_version" in " ".join(call) for call in runner.calls), (
        "fully configured versions are not re-queried"
    )
    with pytest.raises(ValueError, match="positive"):
        GpuIdentityRefresher(configured, runner=runner, interval_seconds=0)


def test_kernel_collector_stamps_the_late_product_on_the_first_xid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GPU_FAULT_GPU_PRODUCT_DISCOVERY", "auto")
    runner = DriverComesUp(failures=1)
    # The kernel collector shells out for nothing else, so the late discovery
    # uses the module's own nvidia-smi helpers; point them at the fake driver.
    monkeypatch.setattr(
        identity_refresh,
        "discover_gpu_product",
        lambda _runner=None: discover_gpu_product(runner),
    )
    monkeypatch.setattr(
        identity_refresh,
        "discover_gpu_software_versions",
        lambda _runner=None: discover_gpu_software_versions(runner),
    )
    sink = RecordingSink()
    clock = [NOW]
    collector = KernelLogCollector(
        sink, _productless(), node_id="worker-1", boot_id="boot-a", now=lambda: clock[0]
    )

    collector.collect_lines([XID_LINE])
    assert sink.requests[-1][1]["product"] is None, (
        "the driver has not answered yet: the record still leaves, product-less"
    )

    clock[0] = NOW + timedelta(seconds=GPU_IDENTITY_REFRESH_INTERVAL_SECONDS)
    collector.collect_lines([XID_LINE.replace("3,1001,", "3,1002,")])
    payload = sink.requests[-1][1]
    assert (payload["product"], payload["driver_branch"], payload["cuda_version"]) == (
        "H200",
        580,
        "13.0",
    ), payload
    assert collector.context.product == "H200"


def test_nvidia_smi_collector_batches_carry_the_late_product(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    monkeypatch.delenv("GPU_FAULT_EXPECTED_GPU_COUNT", raising=False)
    monkeypatch.delenv("GPU_FAULT_NODE_INSTANCE_TYPE", raising=False)
    boot_id = tmp_path / "boot_id"
    boot_id.write_text("boot-a\n", encoding="ascii")
    monkeypatch.setenv("GPU_FAULT_BOOT_ID_PATH", str(boot_id))
    monkeypatch.setenv("GPU_FAULT_GPU_PRODUCT_DISCOVERY", "auto")
    sink = RecordingSink()
    runner = DriverComesUp(failures=0)
    times = iter(NOW + timedelta(seconds=step) for step in range(60))
    collector = NvidiaSmiMetricsCollector(
        sink,
        _productless(),
        node_id="worker-1",
        inventory_interval_seconds=3600,
        now=lambda: next(times),
        runner=runner,
    )

    batch = collector.collect_once()

    assert (batch.product, batch.driver_branch, batch.cuda_version) == (
        "H200",
        580,
        "13.0",
    ), "the round stamps the identity discovered just before it"
    assert collector.context.product == "H200"

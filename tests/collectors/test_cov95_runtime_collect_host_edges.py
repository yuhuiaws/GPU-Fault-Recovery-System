"""Host edge transitions and accelerator discovery without host operations."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from gpu_fault.collectors.host import collector as host_module
from tests.collectors import _cov95_runtime_collect as common
from tests.collectors import _cov95_runtime_collect_host as support

isolated_runtime = common.isolated_runtime
host_case = support.host_case


@pytest.mark.parametrize("gpu,efa", [(None, None), (1, None), (None, 3), (1, 3)])
def test_explicit_inventory_counts_override_only_their_own_instance_default(
    host_case, gpu, efa
):
    collector = host_case.build(
        node_instance_type="ml.p5.48xlarge",
        expected_gpu_count=gpu,
        expected_efa_device_count=efa,
    )
    assert collector.expected_gpu_count == (8 if gpu is None else gpu)
    assert collector.expected_efa_device_count == (32 if efa is None else efa)
    assert host_case.commands == []


def add_efa(case, index, *, active=True, pci=True, driver=True):
    device = case.roots.rdma / f"efa{index}"
    if driver:
        target = case.tmp_path / "drivers/efa"
        target.mkdir(parents=True, exist_ok=True)
        (device / "device").mkdir(parents=True)
        (device / "device/driver").symlink_to(target)
    else:
        support.write_file(device / "device/uevent", "DRIVER=other")
    port = device / "ports/1"
    port.mkdir(parents=True)
    if active:
        (port / "state").write_text("4: ACTIVE")
    if pci:
        support.write_file(case.roots.pci / str(index) / "vendor", "0x1d0f")
        support.write_file(case.roots.pci / str(index) / "device", "0xefa0")
    return port


@pytest.mark.parametrize(
    "mode", ["healthy", "inactive", "unbound", "missing-pci", "excess", "unconfigured"]
)
def test_efa_inventory_keeps_failure_modes_distinct(host_case, mode):
    add_efa(
        host_case,
        0,
        active=mode != "inactive",
        pci=mode != "missing-pci",
        driver=mode != "unbound",
    )
    if mode == "excess":
        add_efa(host_case, 1)
    collector = host_case.build(
        "_efa_inventory",
        expected_efa_device_count=None if mode == "unconfigured" else 1,
        inventory_mismatch_consecutive_samples=1,
    )
    batch = collector.collect_once()
    if mode == "unconfigured":
        assert batch.samples == []
        return
    samples = {sample.name: sample for sample in batch.samples}
    expected = {
        "healthy": "HEALTHY",
        "inactive": "LINK_INACTIVE",
        "unbound": "DRIVER_UNBOUND",
        "missing-pci": "PCI_DEVICE_MISSING",
        "excess": "EXCESS_DEVICE",
    }[mode]
    assert samples["efa_inventory_active_count"].labels["failure_mode"] == expected
    assert samples["efa_inventory_mismatch"].value == int(
        mode in {"inactive", "unbound", "excess"}
    )


def test_unanswerable_driver_query_retains_context_without_claiming_missing_gpus(
    host_case,
):
    for index in range(8):
        (host_case.roots.proc / f"driver/nvidia/gpus/{index}").mkdir(parents=True)
    collector = host_case.build(
        "_gpu_inventory",
        expected_gpu_count=8,
        expected_efa_device_count=32,
        node_instance_type="ml.p5.48xlarge",
    )
    host_case.outputs[tuple(collector.GPU_QUERY_ARGV)] = (
        "",
        "NVML version mismatch",
        1,
    )
    heartbeats = []
    batch = collector.collect_once(lambda: heartbeats.append("progress"))
    values = support.values(batch)
    assert values == {
        ("gpu_inventory_expected_count", None): 8,
        ("gpu_inventory_discovered_count", None): 8,
    }
    assert batch.samples[0].labels["node_instance_type"] == "ml.p5.48xlarge"
    assert "NVML version mismatch" in batch.collection_errors[0]
    assert heartbeats == ["progress"]


@pytest.mark.parametrize("direction", ["drop", "spike", "idle"])
def test_efa_traffic_edges_require_an_active_workload_and_measured_baseline(
    host_case, direction
):
    host_case.monkeypatch.setenv("GPU_FAULT_EFA_TRAFFIC_MIN_ACTIVE_BPS", "10")
    port = add_efa(host_case, 0)
    for counter in ("rx_bytes", "tx_bytes"):
        support.write_file(port / "hw_counters" / counter, "0")
    collector = host_case.build("_rdma", edge_filter_enabled=True)
    if direction == "idle":
        collector.context = common.collector_context(workload_state="IDLE")
    collector.collect_once()
    host_case.clock.sleep(10)
    for counter in ("rx_bytes", "tx_bytes"):
        (port / "hw_counters" / counter).write_text("100")
    collector.collect_once()
    host_case.clock.sleep(10)
    for counter in ("rx_bytes", "tx_bytes"):
        (port / "hw_counters" / counter).write_text(
            "2100" if direction == "spike" else "100"
        )
    batch = collector.collect_once()
    assert ("efa-traffic-drop" in batch.edge_filter_reasons) is (direction == "drop")
    assert ("efa-traffic-spike" in batch.edge_filter_reasons) is (direction == "spike")
    same_instant = collector.collect_once()
    assert ("efa_traffic_bytes_per_second", None) not in support.values(same_instant)


def test_nonforced_host_startup_reports_readiness_and_progress_through_fake_notify(
    host_case, monkeypatch
):
    support.write_file(
        host_case.roots.proc / "meminfo",
        "MemTotal: 1000 kB\nMemAvailable: 800 kB\nMemFree: 700 kB\nCached: 100 kB\n",
    )
    collector = host_case.build("_memory")
    monkeypatch.setenv("NOTIFY_SOCKET", "private-notify")
    messages = []
    waits = []

    def sleep(seconds):
        waits.append(seconds)
        if len(waits) == 2:
            raise common.StopLoop
        host_case.clock.sleep(seconds)

    monkeypatch.setattr(host_module, "sd_notify", messages.append)
    monkeypatch.setattr(host_module, "time", SimpleNamespace(sleep=sleep))
    with pytest.raises(common.StopLoop):
        collector.run()
    assert messages == ["READY=1", "WATCHDOG=1", "WATCHDOG=1"]
    assert 0 < waits[0] <= collector.startup_spread_seconds
    assert waits[1] == collector.interval_seconds
    assert len(host_case.sink.requests) == 1

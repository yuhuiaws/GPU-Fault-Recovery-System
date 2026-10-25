"""Host edge transitions and accelerator discovery without host operations."""

from __future__ import annotations

import shutil
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


def add_pci_bus(case):
    """The bus is never only EFA: a host bridge keeps the listing realistic."""

    support.write_file(case.roots.pci / "0000:00:00.0" / "vendor", "0x8086")
    support.write_file(case.roots.pci / "0000:00:00.0" / "device", "0x1237")


def add_efa(case, index, *, active=True, pci=True, driver=True, rdma=True):
    if pci:
        support.write_file(case.roots.pci / str(index) / "vendor", "0x1d0f")
        support.write_file(case.roots.pci / str(index) / "device", "0xefa0")
    if not rdma:
        return None
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
    (port / "state").write_text("4: ACTIVE" if active else "1: DOWN")
    (port / "phys_state").write_text("5: LinkUp" if active else "3: Disabled")
    return port


@pytest.mark.parametrize(
    "mode",
    [
        "healthy",
        "inactive",
        "unbound",
        "missing-pci",
        "pci-below-bound",
        "excess",
        "unconfigured",
    ],
)
def test_efa_inventory_keeps_failure_modes_distinct(host_case, mode):
    add_pci_bus(host_case)
    # A function that left the bus took its RDMA device with it; a bus that
    # lists fewer functions than the driver has bound is a bad listing.
    add_efa(
        host_case,
        0,
        active=mode != "inactive",
        pci=mode not in {"missing-pci", "pci-below-bound"},
        driver=mode != "unbound",
        rdma=mode != "missing-pci",
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
        "pci-below-bound": "HEALTHY",
        "excess": "EXCESS_DEVICE",
    }[mode]
    labels = samples["efa_inventory_active_count"].labels
    assert labels["failure_mode"] == expected
    assert labels["pci_evidence"] == (
        "INCONSISTENT" if mode == "pci-below-bound" else "LISTED"
    )
    assert samples["efa_inventory_mismatch"].value == int(
        mode in {"inactive", "unbound", "excess", "missing-pci"}
    )
    assert batch.collection_errors == []


def efa_samples(batch):
    return {sample.name: sample for sample in batch.samples}


def test_efa_inventory_dropped_function_confirms_missing_on_the_second_sample(
    host_case,
):
    """A function that leaves the bus takes its RDMA device with it; the
    verdict is PCI_DEVICE_MISSING, and only after the configured two samples."""

    add_pci_bus(host_case)
    add_efa(host_case, 0)
    add_efa(host_case, 1)
    collector = host_case.build(
        "_efa_inventory",
        expected_efa_device_count=2,
        inventory_mismatch_consecutive_samples=2,
    )

    healthy = efa_samples(collector.collect_once())
    assert healthy["efa_inventory_mismatch"].value == 0
    assert healthy["efa_inventory_mismatch"].labels["failure_mode"] == "HEALTHY"
    assert healthy["efa_inventory_mismatch"].labels["pci_evidence"] == "LISTED"

    shutil.rmtree(host_case.roots.pci / "1")
    shutil.rmtree(host_case.roots.rdma / "efa1")

    first = efa_samples(collector.collect_once())
    assert first["efa_inventory_mismatch"].value == 0, "one sample never confirms"
    labels = first["efa_inventory_mismatch"].labels
    assert labels["failure_mode"] == "PCI_DEVICE_MISSING"
    assert labels["consecutive_mismatch_samples"] == "1"

    second = efa_samples(collector.collect_once())
    assert second["efa_inventory_mismatch"].value == 1
    labels = second["efa_inventory_mismatch"].labels
    assert labels["failure_mode"] == "PCI_DEVICE_MISSING"
    assert labels["pci_evidence"] == "LISTED"
    assert second["efa_inventory_discovered_count"].value == 1
    assert second["efa_inventory_driver_bound_count"].value == 1
    assert second["efa_inventory_missing_count"].value == 1


@pytest.mark.parametrize(
    ("pci_functions", "bound", "active", "failure_mode", "mismatch"),
    [
        # Two bound, both ACTIVE, one PCI function listed: the bus cannot
        # hold fewer functions than the driver has bound, so the listing is
        # the odd reading out and the node is healthy by its ports.
        (1, 2, 2, "HEALTHY", 0),
        # Same listing, one port DOWN: diagnose the link, do not reboot.
        (1, 2, 1, "LINK_INACTIVE", 1),
        # No EFA function listed at all while the driver has one bound: the
        # driver shortfall cannot be classified without PCI evidence.
        (0, 1, 1, "INVENTORY_UNAVAILABLE", 1),
    ],
)
def test_efa_inventory_pci_listing_below_bound_count_is_not_a_missing_function(
    host_case, pci_functions, bound, active, failure_mode, mismatch
):
    """A PCI listing below the driver's own bound count is an incomplete or
    mis-pointed listing (``GPU_FAULT_PCI_DEVICES_ROOT`` at the wrong tree),
    never a lost function: PCI_DEVICE_MISSING would reboot the node."""

    add_pci_bus(host_case)
    for index in range(bound):
        add_efa(host_case, index, pci=index < pci_functions, active=index < active)
    collector = host_case.build(
        "_efa_inventory",
        expected_efa_device_count=2,
        inventory_mismatch_consecutive_samples=1,
    )

    samples = efa_samples(collector.collect_once())

    labels = samples["efa_inventory_mismatch"].labels
    assert labels["failure_mode"] == failure_mode
    assert labels["pci_evidence"] == "INCONSISTENT"
    assert samples["efa_inventory_mismatch"].value == mismatch
    # The listing stays visible as read, next to the count it contradicts.
    assert samples["efa_inventory_discovered_count"].value == pci_functions
    assert labels["discovered_count"] == str(pci_functions)
    assert samples["efa_inventory_driver_bound_count"].value == bound


@pytest.mark.parametrize("shape", ["missing", "file", "empty"])
def test_efa_inventory_unreadable_or_empty_pci_root_is_unavailable(host_case, shape):
    """A PCI root that cannot be listed, or lists no device at all (every bus
    carries a bridge at least), is no reading: the tick reports a collection
    error (UNKNOWN) and not one EFA count, never zero functions."""

    add_efa(host_case, 0, pci=False)
    if shape != "empty":
        shutil.rmtree(host_case.roots.pci)
    if shape == "file":
        host_case.roots.pci.write_text("not a directory\n")
    collector = host_case.build(
        "_efa_inventory",
        expected_efa_device_count=2,
        inventory_mismatch_consecutive_samples=1,
    )

    batch = collector.collect_once()

    assert batch.samples == []
    assert len(batch.collection_errors) == 1, batch.collection_errors
    assert "EFA inventory observation unavailable" in batch.collection_errors[0]
    if shape == "empty":
        assert "lists no PCI device" in batch.collection_errors[0]


def test_efa_inventory_unreadable_pci_root_restarts_the_confirmation(host_case):
    """A tick without a reading breaks the streak: the two valid mismatching
    samples PCI_DEVICE_MISSING needs are counted from the next reading on."""

    add_pci_bus(host_case)
    add_efa(host_case, 0)
    collector = host_case.build(
        "_efa_inventory",
        expected_efa_device_count=2,
        inventory_mismatch_consecutive_samples=2,
    )
    first = efa_samples(collector.collect_once())
    assert first["efa_inventory_mismatch"].value == 0
    assert first["efa_inventory_mismatch"].labels["consecutive_mismatch_samples"] == (
        "1"
    )

    bus = host_case.tmp_path / "bus-aside"
    host_case.roots.pci.rename(bus)
    unreadable = collector.collect_once()
    assert unreadable.samples == []
    assert "EFA inventory observation unavailable" in unreadable.collection_errors[0]
    bus.rename(host_case.roots.pci)

    restarted = efa_samples(collector.collect_once())
    assert restarted["efa_inventory_mismatch"].value == 0, "the streak starts over"
    labels = restarted["efa_inventory_mismatch"].labels
    assert labels["consecutive_mismatch_samples"] == "1"
    assert labels["failure_mode"] == "PCI_DEVICE_MISSING"
    confirmed = efa_samples(collector.collect_once())
    assert confirmed["efa_inventory_mismatch"].value == 1


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

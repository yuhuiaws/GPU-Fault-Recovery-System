from pathlib import Path

import pytest

from gpu_fault.collectors.host.observation import (
    efa_inventory_counts,
    is_efa_device,
    is_efa_pci_device,
    read_counter,
)
from gpu_fault.collectors.host.rdma_traffic import port_traffic
from gpu_fault.host_health import NodeHealthPolicy
from gpu_fault.models import WorkloadState
from tests._builders import build_store
from tests.collectors import _cov95_runtime_collect_host as support

host_case = support.host_case
isolated_runtime = support.isolated_runtime


def efa(case):
    device = case.roots.rdma / "efa0"
    support.write_file(device / "device/uevent", "DRIVER=efa\n")
    port = device / "ports/1"
    support.write_file(port / "state", "4: ACTIVE")
    support.write_file(port / "phys_state", "5: LinkUp")
    for name in ("rx_bytes", "tx_bytes"):
        support.write_file(port / "hw_counters" / name, "100")
    pci = case.roots.pci / "0000:01:00.0"
    support.write_file(pci / "vendor", "0x1d0f")
    support.write_file(pci / "device", "0xefa0")
    return port, pci


def test_unreadable_efa_identity_never_authorizes_reboot(host_case, monkeypatch):
    _, pci = efa(host_case)
    original = Path.read_text

    def unreadable(path, *args, **kwargs):
        if path == pci / "vendor":
            raise PermissionError("test sysfs unavailable")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", unreadable)
    collector = host_case.build("_efa_inventory", expected_efa_device_count=1)
    for _ in range(3):
        batch = collector.collect_once()
        assert batch.collection_errors, "unreadable sysfs must be recorded"
        assert "efa_inventory_mismatch" not in {item.name for item in batch.samples}
        assert NodeHealthPolicy(build_store()).evaluate_metrics(batch) == []
        host_case.clock.sleep(15)


def test_inventory_read_failure_breaks_the_missing_device_streak(
    host_case, monkeypatch
):
    port, pci = efa(host_case)
    collector = host_case.build("_efa_inventory", expected_efa_device_count=2)
    assert (
        support.values(collector.collect_once())[("efa_inventory_mismatch", None)] == 0
    )
    original = Path.read_text

    def unreadable(path, *args, **kwargs):
        if path == pci / "vendor":
            raise PermissionError("test sysfs unavailable")
        return original(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "read_text", unreadable)
        host_case.clock.sleep(15)
        assert collector.collect_once().collection_errors, port
    host_case.clock.sleep(15)
    assert (
        support.values(collector.collect_once())[("efa_inventory_mismatch", None)] == 0
    )


@pytest.mark.parametrize("damage", ["invalid", "missing-direction", "reset"])
def test_incomplete_traffic_pair_is_unknown_not_zero(host_case, damage):
    port, _ = efa(host_case)
    collector = host_case.build("_rdma")
    collector.collect_once()
    host_case.clock.sleep(15)
    rx = port / "hw_counters/rx_bytes"
    if damage == "missing-direction":
        rx.unlink()
    else:
        rx.write_text("N/A" if damage == "invalid" else "1")
    batch = collector.collect_once()
    values = support.values(batch)
    assert values[("efa_traffic_observation_valid", None)] == 0
    assert ("efa_traffic_bytes_per_second", None) not in values
    assert batch.collection_errors, batch


def test_unknown_collector_context_delivers_active_and_zero_edges(host_case):
    port, _ = efa(host_case)
    collector = host_case.build("_rdma", edge_filter_enabled=True)
    collector.context = collector.context.model_copy(
        update={"workload_state": WorkloadState.UNKNOWN}
    )
    collector.collect_once()
    host_case.clock.sleep(15)
    for name in ("rx_bytes", "tx_bytes"):
        (port / "hw_counters" / name).write_text(str(32 * 1024 * 1024))
    active = collector.collect_once()
    assert "efa-traffic-active" in active.edge_filter_reasons
    host_case.clock.sleep(15)
    quiet = collector.collect_once()
    assert "efa-traffic-drop" in quiet.edge_filter_reasons


def test_required_interface_absence_is_reported(host_case):
    batch = host_case.build("_network", required_interfaces=["ens5"]).collect_once()
    assert support.values(batch)[("network_link_down", "ens5")] == 1


def test_down_link_survives_unreadable_statistics(host_case):
    interface = host_case.roots.net / "ens5"
    support.write_file(interface / "operstate", "down")
    batch = host_case.build("_network", required_interfaces=["ens5"]).collect_once()
    assert support.values(batch)[("network_link_down", "ens5")] == 1
    assert batch.collection_errors, "counter failure must remain visible"


def test_unknown_link_state_is_not_a_confirmed_link_failure(host_case):
    interface = host_case.roots.net / "ens5"
    support.write_file(interface / "operstate", "unknown")
    batch = host_case.build("_network", required_interfaces=["ens5"]).collect_once()
    assert batch.collection_errors, batch
    assert ("network_link_down", "ens5") not in support.values(batch)


def test_bmc_uses_status_not_units_column(host_case):
    support.enable_tools(host_case, "ipmitool")
    host_case.outputs[("ipmitool", "sensor")] = (
        "CPU Temp | 95.000 | degrees C | cr | na | na\n"
        "Fan | 3000.000 | RPM | ok | na | na\n",
        "",
        0,
    )
    batch = host_case.build("_bmc").collect_once()
    assert support.values(batch)[("bmc_critical_sensor", None)] == 1


@pytest.mark.parametrize("text", ["-1", "NaN", str(1 << 64)])
def test_invalid_counter_encoding_is_not_a_healthy_zero(tmp_path, text):
    path = tmp_path / "counter"
    path.write_text(text)
    with pytest.raises(ValueError):
        read_counter(path)


@pytest.mark.parametrize(
    "vendor,device,expected",
    [("", "0xefa0", None), ("0x10de", "", False), ("0x1d0f", "", None)],
)
def test_pci_identity_distinguishes_invalid_and_foreign_devices(
    tmp_path, vendor, device, expected
):
    (tmp_path / "vendor").write_text(vendor)
    (tmp_path / "device").write_text(device)
    if expected is None:
        with pytest.raises(ValueError):
            is_efa_pci_device(tmp_path)
    else:
        assert is_efa_pci_device(tmp_path) is expected


def test_ambiguous_driver_identity_is_not_silently_excluded(tmp_path):
    support.write_file(tmp_path / "device/uevent", "DRIVER=efa\nDRIVER=mlx5\n")
    with pytest.raises(ValueError, match="ambiguous"):
        is_efa_device(tmp_path)


@pytest.mark.parametrize("damage", ["absent-root", "no-ports", "invalid-state"])
def test_inventory_requires_observable_port_state(tmp_path, damage):
    root = tmp_path / "rdma"
    pci = tmp_path / "pci"
    pci.mkdir()
    if damage == "absent-root":
        assert efa_inventory_counts(root, pci) == (0, 0, 0)
        return
    device = root / "efa0"
    support.write_file(device / "device/uevent", "DRIVER=efa\n")
    (device / "ports").mkdir()
    if damage == "invalid-state":
        support.write_file(device / "ports/1/state", "N/A")
        support.write_file(device / "ports/1/phys_state", "5: LinkUp")
    with pytest.raises(ValueError):
        efa_inventory_counts(root, pci)


def test_traffic_directions_with_different_intervals_are_not_aggregated(tmp_path):
    from datetime import datetime, timezone

    port = tmp_path / "efa0/ports/1"
    for name in ("rx_bytes", "tx_bytes"):
        support.write_file(port / "hw_counters" / name, "100")
    samples, rate, error = port_traffic(
        port,
        datetime.now(timezone.utc),
        {},
        lambda key, value, at: (0, 10 if key.endswith("rx_bytes") else 20),
    )
    assert samples == []
    assert rate is None
    assert error is not None and "intervals do not match" in error

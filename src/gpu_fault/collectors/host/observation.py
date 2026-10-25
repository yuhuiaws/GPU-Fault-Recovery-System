"""Incomplete host observations preserve evidence without inventing health."""

from __future__ import annotations

from pathlib import Path

from gpu_fault.collectors.sinks import CollectorError
from gpu_fault.host_health import HostMetricSample


class HostObservationError(CollectorError):
    def __init__(self, message: str, samples: list[HostMetricSample]) -> None:
        super().__init__(message)
        self.samples = samples


def read_counter(path: Path) -> int:
    text = path.read_text().strip()
    if not text.isascii() or not text.isdigit():
        raise ValueError(f"invalid counter at {path}")
    value = int(text)
    if value > (1 << 64) - 1:
        raise ValueError(f"counter out of range at {path}")
    return value


def is_efa_pci_device(device: Path) -> bool:
    vendor = (device / "vendor").read_text().strip().lower()
    if not vendor.startswith("0x") or len(vendor) != 6:
        raise ValueError(f"invalid PCI vendor at {device}")
    int(vendor, 16)
    if vendor != "0x1d0f":
        return False
    device_id = (device / "device").read_text().strip().lower()
    if not device_id.startswith("0x") or len(device_id) != 6:
        raise ValueError(f"invalid PCI device ID at {device}")
    int(device_id, 16)
    return device_id.startswith("0xefa")


def is_efa_device(device: Path) -> bool:
    try:
        return (device / "device" / "driver").resolve(strict=True).name == "efa"
    except FileNotFoundError:
        pass
    lines = (device / "device" / "uevent").read_text().splitlines()
    drivers = [
        line.removeprefix("DRIVER=") for line in lines if line.startswith("DRIVER=")
    ]
    if len(drivers) != 1:
        raise ValueError(f"missing or ambiguous RDMA driver identity at {device}")
    return drivers[0] == "efa"


def efa_pci_function_count(pci_devices_root: Path) -> int:
    """EFA functions on the bus, counted by PCI vendor/device id, not netdev.

    An unlistable root (missing, a file, hidden by the unit's sandbox) raises
    the ``OSError``; so does a listing with no device on it at all, as a
    ``ValueError``: a PCI bus always carries at least its bridges, so an empty
    tree is a mis-pointed or hidden root, not a bus with no EFA functions.
    Either is "no reading" to the caller, never zero functions.
    """

    functions = list(pci_devices_root.iterdir())
    if not functions:
        raise ValueError(f"PCI device tree lists no PCI device: {pci_devices_root}")
    return sum(is_efa_pci_device(function) for function in functions)


def efa_inventory_counts(
    infiniband_root: Path, pci_devices_root: Path | None
) -> tuple[int | None, int, int]:
    try:
        rdma_devices = list(infiniband_root.iterdir())
    except FileNotFoundError:
        rdma_devices = []
    devices = [device for device in rdma_devices if is_efa_device(device)]
    pci_count = (
        efa_pci_function_count(pci_devices_root)
        if pci_devices_root is not None
        else None
    )
    active_count = 0
    for device in devices:
        ports = list((device / "ports").iterdir())
        if not ports:
            raise ValueError(f"RDMA device has no observable ports: {device}")
        active = False
        for port in ports:
            state = (port / "state").read_text().strip().split(":", 1)[0]
            physical = (port / "phys_state").read_text().strip().split(":", 1)[0]
            if not state.isdigit() or not physical.isdigit():
                raise ValueError(f"invalid RDMA port state at {port}")
            active = active or (state == "4" and physical == "5")
        active_count += int(active)
    return pci_count, len(devices), active_count

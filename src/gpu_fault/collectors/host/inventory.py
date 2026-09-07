from __future__ import annotations

from typing import Any, Callable

import logging
from datetime import datetime
from pathlib import Path


from gpu_fault.host_health import (
    HostMetricSample,
)


from gpu_fault.collectors.sinks import CollectorError

LOGGER = logging.getLogger(__name__)


class HostInventoryMixin:
    # Attributes supplied by the composed concrete implementation.
    _inventory_mismatch_counts: Any
    expected_efa_device_count: Any
    expected_gpu_count: Any
    infiniband_root: Any
    inventory_mismatch_consecutive_samples: Any
    node_instance_type: Any
    pci_devices_root: Any
    runner: Callable[..., Any]
    _nvidia_smi: Callable[..., Any]
    GPU_QUERY_ARGV: tuple[str, ...]

    def _inventory_samples(
        self,
        *,
        resource: str,
        expected: int,
        observed: int,
        discovered: int | None = None,
        inventory_details: dict[str, int] | None = None,
        extra_labels: dict[str, str] | None = None,
    ) -> list[HostMetricSample]:
        missing = max(0, expected - observed)
        excess = max(0, observed - expected)
        mismatch = missing > 0 or excess > 0
        self._inventory_mismatch_counts[resource] = (
            self._inventory_mismatch_counts[resource] + 1 if mismatch else 0
        )
        persistent = (
            self._inventory_mismatch_counts[resource]
            >= self.inventory_mismatch_consecutive_samples
        )
        prefix = "gpu_inventory" if resource == "gpu" else "efa_inventory"
        unit = "gpus" if resource == "gpu" else "devices"
        labels = {
            "resource": resource.upper(),
            "expected_count": str(expected),
            "observed_count": str(observed),
            "missing_count": str(missing),
            "excess_count": str(excess),
            "consecutive_mismatch_samples": str(
                self._inventory_mismatch_counts[resource]
            ),
            "required_consecutive_samples": str(
                self.inventory_mismatch_consecutive_samples
            ),
        }
        if self.node_instance_type:
            labels["node_instance_type"] = self.node_instance_type
        if self.expected_gpu_count is not None:
            labels["expected_gpu_count"] = str(self.expected_gpu_count)
        if self.expected_efa_device_count is not None:
            labels["expected_efa_device_count"] = str(self.expected_efa_device_count)
        if discovered is not None:
            labels["discovered_count"] = str(discovered)
        details = inventory_details or {}
        labels.update({name: str(value) for name, value in details.items()})
        labels.update(extra_labels or {})
        samples = [
            HostMetricSample(
                name=f"{prefix}_expected_count",
                value=expected,
                unit=unit,
                labels=labels,
            ),
            HostMetricSample(
                name=f"{prefix}_active_count",
                value=observed,
                unit=unit,
                labels=labels,
            ),
            HostMetricSample(
                name=f"{prefix}_missing_count",
                value=missing,
                unit=unit,
                labels=labels,
            ),
            HostMetricSample(
                name=f"{prefix}_excess_count",
                value=excess,
                unit=unit,
                labels=labels,
            ),
            HostMetricSample(
                name=f"{prefix}_mismatch",
                value=1 if persistent else 0,
                unit=None,
                labels=labels,
            ),
        ]
        if discovered is not None:
            samples.append(
                HostMetricSample(
                    name=f"{prefix}_discovered_count",
                    value=discovered,
                    unit=unit,
                    labels=labels,
                )
            )
        samples.extend(
            HostMetricSample(
                name=f"{prefix}_{name}",
                value=value,
                unit=unit,
                labels=labels,
            )
            for name, value in details.items()
        )
        return samples

    def _accelerator_inventory(self, _: datetime) -> list[HostMetricSample]:
        return [
            *self._gpu_inventory(_),
            *self._efa_inventory(_),
        ]

    def _gpu_inventory(self, observed_at: datetime) -> list[HostMetricSample]:
        samples = []
        if self.expected_gpu_count is not None:
            # Shares the utilization query of the same round (ARCH-G5): the
            # UUID is its first CSV column.
            completed = self._nvidia_smi(list(self.GPU_QUERY_ARGV), observed_at)
            if completed.returncode != 0:
                raise CollectorError(
                    "GPU inventory query failed: " + completed.stderr.strip()
                )
            gpu_uuids = {
                line.split(",", 1)[0].strip()
                for line in completed.stdout.splitlines()
                if line.strip()
            }
            samples.extend(
                self._inventory_samples(
                    resource="gpu",
                    expected=self.expected_gpu_count,
                    observed=len(gpu_uuids),
                )
            )
        return samples

    def _efa_inventory(self, _: datetime) -> list[HostMetricSample]:
        samples = []
        if self.expected_efa_device_count is not None:
            devices = (
                [
                    device
                    for device in self.infiniband_root.iterdir()
                    if device.is_dir() and self._is_efa_device(device)
                ]
                if self.infiniband_root.exists()
                else []
            )
            pci_devices = (
                [
                    device
                    for device in self.pci_devices_root.iterdir()
                    if device.is_dir() and self._is_efa_pci_device(device)
                ]
                if (
                    self.pci_devices_root is not None and self.pci_devices_root.exists()
                )
                else []
            )
            pci_discovered = (
                len(pci_devices) if self.pci_devices_root is not None else len(devices)
            )
            driver_bound = len(devices)
            active_devices = 0
            for device in devices:
                active = False
                for port in (device / "ports").glob("*"):
                    state_path = port / "state"
                    physical_path = port / "phys_state"
                    if not state_path.exists():
                        continue
                    state = state_path.read_text().strip().split(":", 1)[0]
                    physical = (
                        physical_path.read_text().strip().split(":", 1)[0]
                        if physical_path.exists()
                        else "5"
                    )
                    if state == "4" and physical == "5":
                        active = True
                        break
                if active:
                    active_devices += 1
            if pci_discovered < self.expected_efa_device_count:
                failure_mode = "PCI_DEVICE_MISSING"
            elif driver_bound < self.expected_efa_device_count:
                failure_mode = "DRIVER_UNBOUND"
            elif active_devices < self.expected_efa_device_count:
                failure_mode = "LINK_INACTIVE"
            elif active_devices > self.expected_efa_device_count:
                failure_mode = "EXCESS_DEVICE"
            else:
                failure_mode = "HEALTHY"
            samples.extend(
                self._inventory_samples(
                    resource="efa",
                    expected=self.expected_efa_device_count,
                    observed=active_devices,
                    discovered=pci_discovered,
                    inventory_details={
                        "driver_bound_count": driver_bound,
                    },
                    extra_labels={"failure_mode": failure_mode},
                )
            )
        return samples

    @staticmethod
    def _is_efa_pci_device(device: Path) -> bool:
        try:
            vendor = (device / "vendor").read_text().strip().lower()
            device_id = (device / "device").read_text().strip().lower()
        except OSError:
            return False
        return vendor == "0x1d0f" and device_id.startswith("0xefa")

    @staticmethod
    def _is_efa_device(device: Path) -> bool:
        driver_path = device / "device" / "driver"
        try:
            if driver_path.resolve(strict=True).name == "efa":
                return True
        except OSError:
            pass
        uevent_path = device / "device" / "uevent"
        try:
            return any(
                line.strip() == "DRIVER=efa"
                for line in uevent_path.read_text().splitlines()
            )
        except OSError:
            return False

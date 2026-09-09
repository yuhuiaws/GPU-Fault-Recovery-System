from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from gpu_fault.collectors.sinks import CollectorError
from gpu_fault.host_health import (
    HostMetricSample,
)

LOGGER = logging.getLogger(__name__)


class HostInventoryMixin:
    # Attributes supplied by the composed concrete implementation.
    _inventory_mismatch_counts: Any
    _partial_samples: list[HostMetricSample]
    expected_efa_device_count: Any
    expected_gpu_count: Any
    infiniband_root: Any
    inventory_mismatch_consecutive_samples: Any
    node_instance_type: Any
    pci_devices_root: Any
    proc_root: Path
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
        samples: list[HostMetricSample] = []
        if self.expected_gpu_count is not None:
            # Shares the utilization query of the same round (ARCH-G5): the
            # UUID is its first CSV column. A query that never answered raises
            # out of here with no samples on purpose: a timeout says nothing
            # about how many GPUs are installed, and reporting zero active GPUs
            # would mint a CRITICAL REBOOT_NODE finding for a slow driver.
            completed = self._nvidia_smi(list(self.GPU_QUERY_ARGV), observed_at)
            gpu_uuids = self._parse_gpu_uuids(completed.stdout)
            if completed.returncode != 0:
                # A GPU that falls off the bus commonly makes ``--query-gpu``
                # exit non-zero ("Unable to determine the device handle for
                # GPU ...") after listing the GPUs that still answer. Raising
                # here skipped ``_inventory_samples`` entirely, so
                # ``gpu_inventory_mismatch`` never reached the consumer and the
                # rule the query exists to feed could not fire (F-H6). The
                # samples ride out on ``_partial_samples`` and the error still
                # names the contributor.
                discovered = self._discovered_gpu_count()
                if not gpu_uuids and discovered > 0:
                    # Not one GPU listed while the driver has several bound is
                    # not "every GPU vanished", it is one broken userspace
                    # query -- "Failed to initialize NVML: Driver/library
                    # version mismatch" after a driver upgrade without a
                    # reboot, which lands on every node of the fleet at once.
                    # A mismatch here would mean fleet-wide REBOOT_NODE, so
                    # only the context ships and the streak counter is left
                    # alone.
                    self._partial_samples.extend(
                        self._driver_query_failure_samples(
                            expected=self.expected_gpu_count,
                            discovered=discovered,
                        )
                    )
                else:
                    self._partial_samples.extend(
                        self._inventory_samples(
                            resource="gpu",
                            expected=self.expected_gpu_count,
                            observed=len(gpu_uuids),
                            discovered=discovered,
                            extra_labels={"failure_mode": "DRIVER_QUERY_FAILED"},
                        )
                    )
                raise CollectorError(
                    "GPU inventory query failed: " + completed.stderr.strip()
                )
            samples.extend(
                self._inventory_samples(
                    resource="gpu",
                    expected=self.expected_gpu_count,
                    observed=len(gpu_uuids),
                )
            )
        return samples

    @staticmethod
    def _parse_gpu_uuids(stdout: str) -> set[str]:
        """The UUIDs of ``--query-gpu``, and nothing else on stdout.

        ``nvidia-smi`` prints its own diagnostics on *stdout*, interleaved with
        the GPUs that still answer ("Unable to determine the device handle for
        GPU 0000:07:00.0: Unknown Error") and fills unavailable fields with
        ``[N/A]``. Counting every non-empty first column therefore counted the
        error text as a GPU, so seven live GPUs plus one error line equalled
        the expected eight and F-H6 never fired in its most common real shape.
        A UUID is the only thing that counts as an observed GPU.
        """

        uuids = set()
        for line in stdout.splitlines():
            field = line.split(",", 1)[0].strip()
            if field.startswith(("GPU-", "MIG-")):
                uuids.add(field)
        return uuids

    def _driver_query_failure_samples(
        self, *, expected: int, discovered: int
    ) -> list[HostMetricSample]:
        """Context for a driver query that listed nothing, minus any verdict.

        The consumer's inventory rule reads ``gpu_inventory_mismatch``; these
        samples deliberately do not carry it, so the operator sees the node and
        the failure mode without a destructive action being derived from a
        number ``nvidia-smi`` never produced.
        """

        labels = {
            "resource": "GPU",
            "expected_count": str(expected),
            "discovered_count": str(discovered),
            "failure_mode": "DRIVER_QUERY_FAILED",
        }
        if self.node_instance_type:
            labels["node_instance_type"] = self.node_instance_type
        return [
            HostMetricSample(
                name="gpu_inventory_expected_count",
                value=expected,
                unit="gpus",
                labels=labels,
            ),
            HostMetricSample(
                name="gpu_inventory_discovered_count",
                value=discovered,
                unit="gpus",
                labels=labels,
            ),
        ]

    def _discovered_gpu_count(self) -> int:
        """GPUs the driver has bound, counted without ``nvidia-smi``.

        ``/proc/driver/nvidia/gpus/<pci-address>`` is one directory per GPU the
        kernel module attached, which is the same role ``pci_devices_root``
        plays for EFA: it separates "the device is gone" from "the userspace
        query failed".
        """

        root = self.proc_root / "driver" / "nvidia" / "gpus"
        try:
            return sum(1 for entry in root.iterdir() if entry.is_dir())
        except OSError:
            return 0

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

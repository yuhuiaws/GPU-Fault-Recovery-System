from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from gpu_fault.collectors.sinks import CollectorError
from gpu_fault.collectors.host.observation import (
    efa_inventory_counts,
    is_efa_device,
    is_efa_pci_device,
)
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
                discovered, unreadable = self._discovered_gpu_count()
                failure = "GPU inventory query failed: " + completed.stderr.strip()
                if unreadable is not None:
                    failure += (
                        f"; {unreadable}, so the bound GPU count is unknown as well"
                    )
                if not gpu_uuids and (discovered is None or discovered > 0):
                    # Not one GPU listed while the driver has several bound is
                    # not "every GPU vanished", it is one broken userspace
                    # query -- "Failed to initialize NVML: Driver/library
                    # version mismatch" after a driver upgrade without a
                    # reboot, which lands on every node of the fleet at once.
                    # A mismatch here would mean fleet-wide REBOOT_NODE, so
                    # only the context ships and the streak counter is left
                    # alone. Not one GPU listed while the driver's own list
                    # could not be read either is two failed reads, not zero
                    # GPUs (GF-GENERAL-LIVE-012): the procfs fallback used to
                    # report an unreadable directory as 0 bound, and the
                    # second such tick minted REBOOT_NODE.
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
                raise CollectorError(failure)
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
        self, *, expected: int, discovered: int | None
    ) -> list[HostMetricSample]:
        """Context for a driver query that listed nothing, minus any verdict.

        The consumer's inventory rule reads ``gpu_inventory_mismatch``; these
        samples deliberately do not carry it, so the operator sees the node and
        the failure mode without a destructive action being derived from a
        number ``nvidia-smi`` never produced. ``discovered`` is None when the
        driver's own list could not be read either: the inventory is then
        ``INVENTORY_UNAVAILABLE`` and no count of any kind is reported.
        """

        labels = {
            "resource": "GPU",
            "expected_count": str(expected),
            "failure_mode": "DRIVER_QUERY_FAILED"
            if discovered is not None
            else "INVENTORY_UNAVAILABLE",
        }
        if discovered is not None:
            labels["discovered_count"] = str(discovered)
        if self.node_instance_type:
            labels["node_instance_type"] = self.node_instance_type
        samples = [
            HostMetricSample(
                name="gpu_inventory_expected_count",
                value=expected,
                unit="gpus",
                labels=labels,
            )
        ]
        if discovered is not None:
            samples.append(
                HostMetricSample(
                    name="gpu_inventory_discovered_count",
                    value=discovered,
                    unit="gpus",
                    labels=labels,
                )
            )
        return samples

    def _discovered_gpu_count(self) -> tuple[int | None, str | None]:
        """GPUs the driver has bound, counted without ``nvidia-smi``.

        ``/proc/driver/nvidia/gpus/<pci-address>`` is one directory per GPU the
        kernel module attached, which is the same role ``pci_devices_root``
        plays for EFA: it separates "the device is gone" from "the userspace
        query failed". The module creates the directory when it loads, so an
        empty one is a real "no GPU bound"; one that is missing or unreadable
        is no reading at all and returns ``(None, why)`` -- never 0.
        """

        root = self.proc_root / "driver" / "nvidia" / "gpus"
        try:
            return sum(1 for entry in root.iterdir() if entry.is_dir()), None
        except OSError as exc:
            return None, f"{root} could not be read ({exc.strerror or exc})"

    def _efa_inventory(self, _: datetime) -> list[HostMetricSample]:
        samples = []
        if self.expected_efa_device_count is not None:
            try:
                pci_discovered, driver_bound, active_devices = efa_inventory_counts(
                    self.infiniband_root, self.pci_devices_root
                )
            except (OSError, ValueError) as exc:
                self._inventory_mismatch_counts["efa"] = 0
                raise CollectorError(
                    f"EFA inventory observation unavailable: {exc}"
                ) from exc
            if pci_discovered is None:
                failure_mode = (
                    "INVENTORY_UNAVAILABLE"
                    if driver_bound < self.expected_efa_device_count
                    else "LINK_INACTIVE"
                    if active_devices < self.expected_efa_device_count
                    else "EXCESS_DEVICE"
                    if active_devices > self.expected_efa_device_count
                    else "HEALTHY"
                )
            elif pci_discovered < self.expected_efa_device_count:
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
                    discovered=pci_discovered
                    if pci_discovered is not None
                    else driver_bound,
                    inventory_details={
                        "driver_bound_count": driver_bound,
                    },
                    extra_labels={"failure_mode": failure_mode},
                )
            )
        return samples

    @staticmethod
    def _is_efa_pci_device(device: Path) -> bool:
        return is_efa_pci_device(device)

    @staticmethod
    def _is_efa_device(device: Path) -> bool:
        return is_efa_device(device)

"""InfoROM checksum validity, read at the inventory cadence (batch 233).

DCGM field 84 (``DCGM_FI_DEV_INFOROM_CONFIG_VALID``) "Reads the infoROM from the
flash and verifies the checksums." (DCGM field identifiers, docs 4.6) -- on
every watch, so it stays out of the per-tick counters. The same NVML check
(``nvmlDeviceValidateInforom``) is read here once per inventory interval
(60 s by default) with nvidia-smi(1) ``--query-gpu=index,pci.bus_id,uuid,
inforom.checksum_validation`` (the brief's query plus ``uuid``, the identity the
DCGM samples carry) -- ``inforom.checksum_validation``
(``valid``, ``invalid``, ``N/A``; "Only available via --query-gpu
inforom.checksum_validation"; added in nvidia-smi v575). The answer is cached
and published beside every scrape as ``inforom_checksum_valid`` (1 valid, 0
invalid); ``N/A`` publishes nothing. A failed query keeps nothing: a wedged
driver must not keep a stale "valid" alive, and the retry waits for the next
interval. The control plane grades ``invalid`` (``gpu_field_rules``).
"""

from __future__ import annotations

import csv
import io
import logging
import subprocess
from datetime import datetime, timedelta
from typing import Callable

from gpu_fault.gpu_field_rules import INFOROM_CHECKSUM_VALID
from gpu_fault.gpu_metrics import GpuMetricSample

Runner = Callable[..., subprocess.CompletedProcess[str]]
INFOROM_QUERY = [
    "nvidia-smi",
    "--query-gpu=index,pci.bus_id,uuid,inforom.checksum_validation",
    "--format=csv,noheader",
]
INFOROM_VALUES = {"valid": 1.0, "invalid": 0.0}


def parse_inforom_validation(text: str) -> list[GpuMetricSample]:
    """One sample per GPU that answered ``valid`` or ``invalid``."""

    samples = []
    for row in csv.reader(io.StringIO(text)):
        if len(row) != 4:
            continue
        index, pci_bdf, gpu_uuid, state = (item.strip() for item in row)
        value = INFOROM_VALUES.get(state.lower())
        if value is None or not index.isdigit():
            continue
        samples.append(
            GpuMetricSample(
                metric_name="nvidia_smi_inforom_checksum_validation",
                canonical_name=INFOROM_CHECKSUM_VALID,
                value=value,
                unit=None,
                gpu_index=index,
                gpu_uuid=gpu_uuid or None,
                pci_bdf=pci_bdf.lower() or None,
                labels={
                    "gpu": index,
                    **({"UUID": gpu_uuid} if gpu_uuid else {}),
                    **({"pci_bus_id": pci_bdf} if pci_bdf else {}),
                    "inforom_checksum_validation": state.lower(),
                },
            )
        )
    return samples


class InforomValidityProbe:
    """The cached InfoROM answer, refreshed once per ``interval_seconds``."""

    def __init__(self, *, interval_seconds: int, logger: logging.Logger) -> None:
        if interval_seconds <= 0:
            raise ValueError("InfoROM probe interval must be positive")
        self.interval = timedelta(seconds=interval_seconds)
        self.logger = logger
        self.samples: list[GpuMetricSample] = []
        self.next_at: datetime | None = None
        self._warned: str | None = None

    def refresh(self, observed_at: datetime, runner: Runner) -> list[GpuMetricSample]:
        """The cached samples, re-read when the interval has passed."""

        if self.next_at is not None and observed_at < self.next_at:
            return self.samples
        self.next_at = observed_at + self.interval
        try:
            result = runner(
                INFOROM_QUERY,
                capture_output=True,
                text=True,
                timeout=15,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return self._failed(f"InfoROM validity query unavailable: {exc}")
        if result.returncode != 0:
            return self._failed(
                "InfoROM validity query failed: "
                + (result.stderr.strip() or "no error detail")
            )
        self.samples = parse_inforom_validation(result.stdout)
        self._warned = None
        return self.samples

    def _failed(self, detail: str) -> list[GpuMetricSample]:
        self.samples = []
        log = self.logger.debug if detail == self._warned else self.logger.warning
        self._warned = detail
        log("%s; no InfoROM validity until the next probe", detail)
        return self.samples

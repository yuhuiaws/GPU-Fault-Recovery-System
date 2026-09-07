from __future__ import annotations

import argparse
import csv
import io
import logging
import os
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from gpu_fault.channel_registry import GPU_METRICS_PATH
from gpu_fault.gpu_metrics import (
    GpuMetricBatch,
    GpuMetricSample,
    GpuMetricSource,
)


from gpu_fault.collectors.gpu.discovery import (
    deliver_gpu_inventory,
    query_nvidia_temperature_limits,
)
from gpu_fault.collectors.models import CollectorContext
from gpu_fault.collectors.scheduling import next_stable_phase
from gpu_fault.collectors.sinks import CollectorError, EventSink

LOGGER = logging.getLogger(__name__)

NVIDIA_SMI_CORE_FIELDS: dict[str, tuple[str, str | None]] = {
    "temperature.gpu": ("gpu_temperature_c", "celsius"),
    "power.draw": ("power_usage_w", "watts"),
    "utilization.gpu": ("gpu_utilization_percent", "percent"),
    "utilization.memory": (
        "memory_utilization_percent",
        "percent",
    ),
    "memory.used": ("framebuffer_used_mib", "MiB"),
    "memory.free": ("framebuffer_free_mib", "MiB"),
}

NVIDIA_SMI_ECC_FIELDS: dict[str, tuple[str, str | None]] = {
    "ecc.errors.corrected.volatile.total": (
        "ecc_sbe_volatile_total",
        "errors",
    ),
    "ecc.errors.uncorrected.volatile.total": (
        "ecc_dbe_volatile_total",
        "errors",
    ),
    "ecc.errors.corrected.aggregate.total": (
        "ecc_sbe_aggregate_total",
        "errors",
    ),
    "ecc.errors.uncorrected.aggregate.total": (
        "ecc_dbe_aggregate_total",
        "errors",
    ),
}

NVIDIA_SMI_RETIRED_FIELDS: dict[str, tuple[str, str | None]] = {
    "retired_pages.pending": ("retired_pages_pending", None),
}

NVIDIA_SMI_REMAP_FIELDS: dict[str, tuple[str, str | None]] = {
    "remapped_rows.correctable": (
        "row_remap_correctable_total",
        "rows",
    ),
    "remapped_rows.uncorrectable": (
        "row_remap_uncorrectable_total",
        "rows",
    ),
    "remapped_rows.pending": ("row_remap_pending", None),
    "remapped_rows.failure": ("row_remap_failure", None),
}


class NvidiaSmiMetricsCollector:
    _IDENTITY_FIELDS = ["index", "uuid", "name", "pci.bus_id"]

    def __init__(
        self,
        sink: EventSink,
        context: CollectorContext,
        *,
        node_id: str,
        interval_seconds: float = 30,
        inventory_interval_seconds: int | None = None,
        startup_spread_seconds: int | None = None,
        force_snapshot_path: str | None = None,
        now: Callable[[], datetime] | None = None,
        runner: Callable[..., subprocess.CompletedProcess[str]] = (subprocess.run),
    ) -> None:
        self.sink = sink
        self.context = context
        self.node_id = node_id
        self.interval_seconds = interval_seconds
        self.now = now or (lambda: datetime.now(timezone.utc))
        self.runner = runner
        self.inventory_interval_seconds = (
            inventory_interval_seconds
            if inventory_interval_seconds is not None
            else int(
                os.getenv(
                    "GPU_FAULT_GPU_INVENTORY_INTERVAL_SECONDS",
                    "60",
                )
            )
        )
        if self.inventory_interval_seconds <= 0:
            raise ValueError("GPU inventory interval must be positive")
        self.startup_spread_seconds = (
            startup_spread_seconds
            if startup_spread_seconds is not None
            else int(
                os.getenv(
                    "GPU_FAULT_COLLECTOR_STARTUP_SPREAD_SECONDS",
                    "15",
                )
            )
        )
        if self.startup_spread_seconds <= 0:
            raise ValueError("collector startup spread must be positive")
        self.force_snapshot_path = Path(
            force_snapshot_path
            or os.getenv(
                "GPU_FAULT_GPU_HEALTH_SNAPSHOT_REQUEST_PATH",
                ("/var/lib/gpu-fault/health-snapshot/gpu.request"),
            )
        )
        self._last_inventory_delivered_at: datetime | None = None
        self._next_inventory_at: datetime | None = None
        self._temperature_limit_samples: list[GpuMetricSample] | None = None

    def collect_once(self) -> GpuMetricBatch:
        timestamp = self.now()
        if (
            self._last_inventory_delivered_at is None
            or (
                self._next_inventory_at is not None
                and timestamp >= self._next_inventory_at
            )
            or (
                self._next_inventory_at is None
                and self._last_inventory_delivered_at is not None
                and (timestamp - self._last_inventory_delivered_at).total_seconds()
                >= self.inventory_interval_seconds
            )
        ):
            deliver_gpu_inventory(
                self.sink,
                self.context,
                node_id=self.node_id,
                observed_at=timestamp,
                runner=self.runner,
            )
            self._last_inventory_delivered_at = timestamp
            self._next_inventory_at = next_stable_phase(
                timestamp,
                cluster_id=self.context.cluster_id,
                node_id=self.node_id,
                channel="gpu-inventory",
                interval_seconds=self.inventory_interval_seconds,
            )
        samples = self._query(NVIDIA_SMI_CORE_FIELDS, required=True)
        for optional_fields in (
            NVIDIA_SMI_ECC_FIELDS,
            NVIDIA_SMI_RETIRED_FIELDS,
            NVIDIA_SMI_REMAP_FIELDS,
        ):
            samples.extend(self._query(optional_fields, required=False))
        if self._temperature_limit_samples is None:
            try:
                discovered = query_nvidia_temperature_limits(self.runner)
                if discovered:
                    self._temperature_limit_samples = discovered
            except CollectorError as exc:
                LOGGER.warning(
                    "NVIDIA temperature limits unavailable; "
                    "using configured fallback thresholds: %s",
                    exc,
                )
        samples.extend(self._temperature_limit_samples or [])
        batch = GpuMetricBatch(
            batch_id=(
                f"nvidia-smi-{self.node_id}-{int(timestamp.timestamp() * 1_000_000)}"
            ),
            cluster_id=self.context.cluster_id,
            node_id=self.node_id,
            observed_at=timestamp,
            collected_at=timestamp,
            source=GpuMetricSource.NVIDIA_SMI,
            samples=samples,
            runtime_profile_version=(self.context.runtime_profile_version),
            product=self.context.product,
            driver_branch=self.context.driver_branch,
            cuda_version=self.context.cuda_version,
            workload_state=self.context.workload_state,
            affected_workload_ids=(self.context.affected_workload_ids),
            checkpoint_manifest_ref=(self.context.checkpoint_manifest_ref),
            evidence_ref=f"nvidia-smi://{self.node_id}",
        )
        self.sink.post(
            GPU_METRICS_PATH,
            batch.model_dump(mode="json"),
        )
        return batch

    def run(self) -> None:
        force_snapshot = self.force_snapshot_path.exists()
        if not force_snapshot:
            startup_time = self.now()
            delay = (
                next_stable_phase(
                    startup_time,
                    cluster_id=self.context.cluster_id,
                    node_id=self.node_id,
                    channel="nvidia-smi-collector-startup",
                    interval_seconds=self.startup_spread_seconds,
                )
                - startup_time
            ).total_seconds()
            if delay > 0:
                time.sleep(delay)
        while True:
            succeeded = False
            try:
                self.collect_once()
                succeeded = True
            except CollectorError:
                LOGGER.exception("nvidia-smi metrics collection failed")
            if force_snapshot and succeeded:
                self.force_snapshot_path.unlink(missing_ok=True)
                force_snapshot = False
            time.sleep(self.interval_seconds)

    def _query(
        self,
        fields: dict[str, tuple[str, str | None]],
        *,
        required: bool,
    ) -> list[GpuMetricSample]:
        query_fields = [*self._IDENTITY_FIELDS, *fields]
        command = [
            "nvidia-smi",
            f"--query-gpu={','.join(query_fields)}",
            "--format=csv,noheader,nounits",
        ]
        result = self.runner(
            command,
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        if result.returncode != 0:
            if required:
                raise CollectorError(
                    f"nvidia-smi query failed: {result.stderr.strip()}"
                )
            LOGGER.warning(
                "optional nvidia-smi health fields unavailable: %s",
                result.stderr.strip(),
            )
            return []
        return self.parse_csv(result.stdout, query_fields, fields)

    @staticmethod
    def parse_csv(
        text: str,
        query_fields: list[str],
        metrics: dict[str, tuple[str, str | None]],
    ) -> list[GpuMetricSample]:
        samples = []
        for row in csv.reader(io.StringIO(text)):
            if len(row) != len(query_fields):
                raise CollectorError("unexpected nvidia-smi CSV column count")
            values = {
                name: value.strip()
                for name, value in zip(query_fields, row, strict=True)
            }
            labels = {
                "gpu": values["index"],
                "UUID": values["uuid"],
                "modelName": values["name"],
                "pci_bus_id": values["pci.bus_id"],
            }
            for field, (canonical, unit) in metrics.items():
                raw = values[field]
                if raw.lower() in {
                    "n/a",
                    "na",
                    "not supported",
                    "[not supported]",
                    "",
                }:
                    continue
                try:
                    boolean = {
                        "yes": 1.0,
                        "true": 1.0,
                        "no": 0.0,
                        "false": 0.0,
                    }.get(raw.lower())
                    value = boolean if boolean is not None else float(raw)
                except ValueError:
                    continue
                samples.append(
                    GpuMetricSample(
                        metric_name=(
                            "nvidia_smi_" + re.sub(r"[^a-z0-9]+", "_", field.lower())
                        ),
                        canonical_name=canonical,
                        value=value,
                        unit=unit,
                        gpu_index=values["index"],
                        gpu_uuid=values["uuid"],
                        pci_bdf=values["pci.bus_id"],
                        labels=labels,
                    )
                )
        return samples


def build_from_environment(
    sink: EventSink, context: CollectorContext, arguments: argparse.Namespace
) -> NvidiaSmiMetricsCollector:
    """The ``gpu-fault-collector nvidia-smi`` factory named by the registry."""

    if not arguments.node_id:
        raise SystemExit("--node-id, NODE_NAME, or HOSTNAME is required")
    return NvidiaSmiMetricsCollector(
        sink,
        context,
        node_id=arguments.node_id,
        interval_seconds=arguments.interval_seconds,
    )

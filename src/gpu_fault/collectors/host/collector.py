from __future__ import annotations

import logging
import os
import shlex
import subprocess
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from gpu_fault.channel_registry import HOST_TELEMETRY_PATH
from gpu_fault.models import WorkloadState
from gpu_fault.host_health import (
    HostMetricSample,
    HostTelemetryBatch,
    HostTelemetryHistoryPoint,
    NodeHealthPolicy,
)


from gpu_fault.collectors.models import CollectorContext
from gpu_fault.collectors.scheduling import next_stable_phase
from gpu_fault.collectors.sinks import EventSink
from gpu_fault.collectors.host.gpu_rank import HostGpuRankMixin
from gpu_fault.collectors.host.inventory import HostInventoryMixin
from gpu_fault.collectors.host.network import HostNetworkMixin
from gpu_fault.collectors.host.system_metrics import HostSystemMetricsMixin

LOGGER = logging.getLogger(__name__)


class HostTelemetryCollector(
    HostGpuRankMixin,
    HostInventoryMixin,
    HostNetworkMixin,
    HostSystemMetricsMixin,
):
    """Collects host, storage, network, RDMA, SMART and BMC signals."""

    def __init__(
        self,
        sink: EventSink,
        context: CollectorContext,
        *,
        node_id: str,
        interval_seconds: float = 15,
        filesystems: list[str] | None = None,
        required_interfaces: list[str] | None = None,
        infiniband_root: str = "/sys/class/infiniband",
        proc_root: str = "/proc",
        pci_devices_root: str | None = None,
        node_instance_type: str | None = None,
        expected_gpu_count: int | None = None,
        expected_efa_device_count: int | None = None,
        inventory_mismatch_consecutive_samples: int = 2,
        nvswitch_topology_command: list[str] | None = None,
        now: Callable[[], datetime] | None = None,
        runner: Callable[..., subprocess.CompletedProcess[str]] = (subprocess.run),
        edge_filter_enabled: bool | None = None,
        health_summary_seconds: int | None = None,
        history_max_points: int | None = None,
        startup_spread_seconds: int | None = None,
        force_snapshot_path: str | None = None,
    ) -> None:
        self.sink = sink
        self.context = context
        self.node_id = node_id
        self.interval_seconds = interval_seconds
        self.filesystems = filesystems or ["/", "/var", "/tmp"]
        self.required_interfaces = set(required_interfaces or [])
        self.infiniband_root = Path(infiniband_root)
        self.proc_root = Path(proc_root)
        self.pci_devices_root = (
            Path(pci_devices_root)
            if pci_devices_root is not None
            else (
                Path("/sys/bus/pci/devices")
                if infiniband_root == "/sys/class/infiniband"
                else None
            )
        )
        self.node_instance_type = node_instance_type
        self.expected_gpu_count = expected_gpu_count
        self.expected_efa_device_count = expected_efa_device_count
        if inventory_mismatch_consecutive_samples < 1:
            raise ValueError("inventory mismatch consecutive samples must be positive")
        self.inventory_mismatch_consecutive_samples = (
            inventory_mismatch_consecutive_samples
        )
        self._inventory_mismatch_counts = {
            "gpu": 0,
            "efa": 0,
        }
        self.nvswitch_topology_command = (
            nvswitch_topology_command
            if nvswitch_topology_command is not None
            else shlex.split(os.getenv("GPU_FAULT_NVSWITCH_TOPOLOGY_COMMAND", ""))
        )
        self.now = now or (lambda: datetime.now(timezone.utc))
        self.runner = runner
        self._previous: dict[str, tuple[float, datetime]] = {}
        self.edge_filter_enabled = (
            edge_filter_enabled
            if edge_filter_enabled is not None
            else os.getenv("GPU_FAULT_HOST_EDGE_FILTER_ENABLED", "true").strip().lower()
            == "true"
        )
        self.health_summary_seconds = (
            health_summary_seconds
            if health_summary_seconds is not None
            else int(os.getenv("GPU_FAULT_HOST_HEALTH_SUMMARY_SECONDS", "300"))
        )
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
        self.force_snapshot_path = Path(
            force_snapshot_path
            or os.getenv(
                "GPU_FAULT_HOST_HEALTH_SNAPSHOT_REQUEST_PATH",
                ("/var/lib/gpu-fault/health-snapshot/host.request"),
            )
        )
        history_points = (
            history_max_points
            if history_max_points is not None
            else int(os.getenv("GPU_FAULT_HOST_HISTORY_MAX_POINTS", "20"))
        )
        if (
            self.health_summary_seconds <= 0
            or history_points <= 0
            or self.startup_spread_seconds <= 0
        ):
            raise ValueError("host edge filter intervals and counts must be positive")
        self.low_utilization_threshold_percent = float(
            os.getenv("GPU_FAULT_LOW_UTILIZATION_THRESHOLD_PERCENT", "5")
        )
        self.memory_available_warning_percent = float(
            os.getenv("GPU_FAULT_MEMORY_AVAILABLE_WARNING_PERCENT", "5")
        )
        self.page_cache_warning_percent = float(
            os.getenv("GPU_FAULT_PAGE_CACHE_WARNING_PERCENT", "90")
        )
        self.local_filesystem_warning_percent = float(
            os.getenv("GPU_FAULT_LOCAL_FILESYSTEM_WARNING_PERCENT", "90")
        )
        self.efa_minimum_active_bps = float(
            os.getenv("GPU_FAULT_EFA_TRAFFIC_MIN_ACTIVE_BPS", "1048576")
        )
        self.efa_drop_ratio = float(
            os.getenv("GPU_FAULT_EFA_TRAFFIC_DROP_RATIO", "0.25")
        )
        self.efa_spike_ratio = float(
            os.getenv("GPU_FAULT_EFA_TRAFFIC_SPIKE_RATIO", "4")
        )
        if not all(
            0 <= value <= 100
            for value in (
                self.low_utilization_threshold_percent,
                self.memory_available_warning_percent,
                self.page_cache_warning_percent,
                self.local_filesystem_warning_percent,
            )
        ):
            raise ValueError("host edge percentage thresholds must be from 0 to 100")
        if not 0 < self.efa_drop_ratio < 1:
            raise ValueError("GPU_FAULT_EFA_TRAFFIC_DROP_RATIO must be in (0, 1)")
        if self.efa_spike_ratio <= 1:
            raise ValueError("GPU_FAULT_EFA_TRAFFIC_SPIKE_RATIO must be greater than 1")
        self.rank_liveness_enabled = (
            os.getenv("GPU_FAULT_RANK_LIVENESS_ENABLED", "true").strip().lower()
            == "true"
        )
        self.rank_progress_min_write_bps = float(
            os.getenv("GPU_FAULT_RANK_PROGRESS_MIN_WRITE_BPS", "1048576")
        )
        self.rank_progress_min_cpu_cores = float(
            os.getenv("GPU_FAULT_RANK_PROGRESS_MIN_CPU_CORES", "0.5")
        )
        self.rank_progress_gpu_idle_percent = float(
            os.getenv("GPU_FAULT_RANK_PROGRESS_GPU_IDLE_PERCENT", "5")
        )
        if (
            self.rank_progress_min_write_bps < 0
            or self.rank_progress_min_cpu_cores <= 0
            or not 0 <= self.rank_progress_gpu_idle_percent <= 100
        ):
            raise ValueError(
                "rank progress thresholds must be non-negative and the "
                "GPU idle percentage must be from 0 to 100"
            )
        self._clock_ticks = float(os.sysconf("SC_CLK_TCK") or 100)
        self._history: deque[HostTelemetryHistoryPoint] = deque(maxlen=history_points)
        self._last_delivered_at: datetime | None = None
        self._next_health_summary_at: datetime | None = None
        self._active_edge_reasons: set[str] = set()
        self._last_efa_rate: float | None = None
        self._last_gpu_utilization_percent: float | None = None
        self._rank_progress_at: datetime | None = None
        self._rank_liveness_warned = False

    def collect_once(self) -> HostTelemetryBatch:
        observed_at = self.now()
        samples: list[HostMetricSample] = []
        errors: list[str] = []
        for collector in (
            self._cpu,
            self._memory,
            self._filesystems,
            self._shared_filesystems,
            self._lustre,
            self._diskstats,
            self._network,
            self._tcp,
            self._gpu_utilization,
            self._rank_liveness,
            self._gpu_inventory,
            self._efa_inventory,
            self._rdma,
            self._efa_network,
            self._nvswitch_topology,
            self._smart,
            self._bmc,
        ):
            try:
                samples.extend(collector(observed_at))
            except Exception as exc:
                errors.append(f"{collector.__name__}: {type(exc).__name__}: {exc}")
        batch = HostTelemetryBatch(
            batch_id=(
                f"host-{self.node_id}-{int(observed_at.timestamp() * 1_000_000)}"
            ),
            cluster_id=self.context.cluster_id,
            node_id=self.node_id,
            observed_at=observed_at,
            samples=samples,
            collection_errors=errors,
            runtime_profile_version=(self.context.runtime_profile_version),
            workload_state=self.context.workload_state,
            affected_workload_ids=(self.context.affected_workload_ids),
            evidence_ref=f"host://{self.node_id}",
        )
        self._history.append(
            HostTelemetryHistoryPoint(
                observed_at=observed_at,
                samples=samples,
                collection_errors=errors,
            )
        )
        reasons = self._edge_reasons(batch)
        delivery_reasons = reasons - self._active_edge_reasons
        if not self.edge_filter_enabled:
            delivery_reasons.add("filter-disabled")
        elif self._last_delivered_at is None:
            delivery_reasons.add("baseline")
        if self._active_edge_reasons - reasons:
            delivery_reasons.add("recovered")
        if self._last_delivered_at is not None and (
            (
                self._next_health_summary_at is not None
                and observed_at >= self._next_health_summary_at
            )
            or (
                self._next_health_summary_at is None
                and (observed_at - self._last_delivered_at).total_seconds()
                >= self.health_summary_seconds
            )
        ):
            delivery_reasons.add("health-summary")
        should_deliver = not self.edge_filter_enabled or bool(delivery_reasons)
        if should_deliver:
            batch = batch.model_copy(
                update={
                    "edge_filter_reasons": sorted(delivery_reasons),
                }
            )
            self.sink.post(
                HOST_TELEMETRY_PATH,
                batch.model_dump(mode="json"),
            )
            self._last_delivered_at = observed_at
            if (
                self._next_health_summary_at is None
                or observed_at >= self._next_health_summary_at
            ):
                self._next_health_summary_at = next_stable_phase(
                    observed_at,
                    cluster_id=self.context.cluster_id,
                    node_id=self.node_id,
                    channel="host-telemetry",
                    interval_seconds=self.health_summary_seconds,
                )
        self._active_edge_reasons = reasons
        return batch

    def collect_rdma_samples(
        self,
        observed_at: datetime | None = None,
    ) -> list[HostMetricSample]:
        """Collect only RDMA/EFA counters without delivering a full batch."""
        return self._rdma(observed_at or self.now())

    def _edge_reasons(self, batch: HostTelemetryBatch) -> set[str]:
        reasons = {
            f"collection-error:{item.split(':', 1)[0]}"
            for item in batch.collection_errors
        }
        sustained = {
            "memory_available_percent": (
                "lte",
                self.memory_available_warning_percent,
            ),
            "page_cache_percent": (
                "gte",
                self.page_cache_warning_percent,
            ),
            "local_filesystem_used_percent": (
                "gte",
                self.local_filesystem_warning_percent,
            ),
        }
        if batch.workload_state is WorkloadState.ACTIVE:
            sustained.update(
                {
                    "cpu_usage_percent": (
                        "lte",
                        self.low_utilization_threshold_percent,
                    ),
                    "host_gpu_utilization_percent": (
                        "lte",
                        self.low_utilization_threshold_percent,
                    ),
                }
            )
        for sample in batch.samples:
            immediate = NodeHealthPolicy.METRIC_RULES.get(sample.name)
            if immediate is not None and sample.value >= immediate[0]:
                reasons.add(f"threshold:{sample.name}")
            candidate = sustained.get(sample.name)
            if candidate is not None:
                comparison, threshold = candidate
                if (comparison == "lte" and sample.value <= threshold) or (
                    comparison == "gte" and sample.value >= threshold
                ):
                    reasons.add(f"sustained:{sample.name}")
            if sample.name == "efa_traffic_bytes_per_second":
                previous = self._last_efa_rate
                if (
                    batch.workload_state is WorkloadState.ACTIVE
                    and previous is not None
                    and previous >= self.efa_minimum_active_bps
                ):
                    ratio = sample.value / previous
                    if ratio <= self.efa_drop_ratio:
                        reasons.add("efa-traffic-drop")
                    if ratio >= self.efa_spike_ratio:
                        reasons.add("efa-traffic-spike")
                self._last_efa_rate = sample.value
        return reasons

    def run(self) -> None:
        force_snapshot = self.force_snapshot_path.exists()
        if not force_snapshot:
            startup_time = self.now()
            delay = (
                next_stable_phase(
                    startup_time,
                    cluster_id=self.context.cluster_id,
                    node_id=self.node_id,
                    channel="host-collector-startup",
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
            except Exception:
                LOGGER.exception("host telemetry collection failed")
            if force_snapshot and succeeded:
                self.force_snapshot_path.unlink(missing_ok=True)
                force_snapshot = False
            time.sleep(self.interval_seconds)

    @staticmethod
    def _sample(
        name: str,
        value: float,
        unit: str | None = None,
        device: str | None = None,
    ) -> HostMetricSample:
        return HostMetricSample(name=name, value=value, unit=unit, device=device)

    def _delta(
        self, key: str, value: float, observed_at: datetime
    ) -> tuple[float, float] | None:
        previous = self._previous.get(key)
        self._previous[key] = (value, observed_at)
        if previous is None or observed_at <= previous[1]:
            return None
        return max(0.0, value - previous[0]), (
            observed_at - previous[1]
        ).total_seconds()

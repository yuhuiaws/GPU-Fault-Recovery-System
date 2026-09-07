from __future__ import annotations

import argparse
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
from gpu_fault.env import env_bool
from gpu_fault.models import WorkloadState
from gpu_fault.host_health import (
    HostMetricSample,
    HostTelemetryBatch,
    HostTelemetryHistoryPoint,
    NodeHealthPolicy,
)


from gpu_fault.collectors.gpu.discovery import expected_accelerator_counts
from gpu_fault.collectors.models import CollectorContext
from gpu_fault.collectors.scheduling import next_stable_phase
from gpu_fault.collectors.sinks import CollectorError, EventSink
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
        nvidia_smi_timeout_seconds: float = 15,
        nvidia_smi_breaker_rounds: int = 3,
        nvidia_smi_breaker_cooldown_rounds: int = 4,
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
        self._default_expected_counts_from_instance_type()
        if (
            nvidia_smi_timeout_seconds <= 0
            or nvidia_smi_breaker_rounds < 1
            or nvidia_smi_breaker_cooldown_rounds < 1
        ):
            raise ValueError(
                "nvidia-smi timeout must be positive and breaker rounds at least 1"
            )
        self.nvidia_smi_timeout_seconds = nvidia_smi_timeout_seconds
        self.nvidia_smi_breaker_rounds = nvidia_smi_breaker_rounds
        self.nvidia_smi_breaker_cooldown_rounds = nvidia_smi_breaker_cooldown_rounds
        self._nvidia_smi_round: dict[
            tuple[str, ...], subprocess.CompletedProcess[str] | CollectorError
        ] = {}
        self._nvidia_smi_round_at: datetime | None = None
        self._nvidia_smi_round_timed_out = False
        self._nvidia_smi_round_skipped = False
        self._nvidia_smi_consecutive_timeouts = 0
        self._nvidia_smi_breaker_skips_remaining = 0
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
            else env_bool("GPU_FAULT_HOST_EDGE_FILTER_ENABLED", True)
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
        self.rank_liveness_enabled = env_bool("GPU_FAULT_RANK_LIVENESS_ENABLED", True)
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

    #: Contributor methods, in collection order. This is a class constant rather
    #: than a literal inside :meth:`collect_once` so a caller that has to account
    #: for every contributor — a test silencing them, for instance — reads the
    #: same list the collector runs. Two host tests silenced a hand-copied subset
    #: and left ``_rank_liveness`` live, so it shelled out to ``nvidia-smi``.
    CONTRIBUTORS: tuple[str, ...] = (
        "_cpu",
        "_memory",
        "_filesystems",
        "_shared_filesystems",
        "_lustre",
        "_diskstats",
        "_network",
        "_tcp",
        "_gpu_utilization",
        "_rank_liveness",
        "_gpu_inventory",
        "_efa_inventory",
        "_rdma",
        "_efa_network",
        "_nvswitch_topology",
        "_smart",
        "_bmc",
    )

    def _default_expected_counts_from_instance_type(self) -> None:
        """Fill the expected GPU/EFA counts from the instance type.

        ``--expected-gpu-count`` was optional and the installer left it
        unset, so the ``gpu_inventory_mismatch`` finding never fired on a
        fleet whose instance type already names the count. An explicit count
        still wins; an unknown type is logged once, not guessed.
        """

        if self.expected_gpu_count is not None and (
            self.expected_efa_device_count is not None
        ):
            return
        counts = expected_accelerator_counts(self.node_instance_type)
        if counts is None:
            if self.node_instance_type:
                LOGGER.warning(
                    "instance type %s is not in the accelerator table and no "
                    "explicit expected GPU/EFA count was configured; the "
                    "inventory invariant is not checked",
                    self.node_instance_type,
                )
            return
        if self.expected_gpu_count is None:
            self.expected_gpu_count = counts["gpu"]
        if self.expected_efa_device_count is None:
            self.expected_efa_device_count = counts["efa"]
        LOGGER.info(
            "expected accelerator counts default from instance type %s: gpu=%d efa=%d",
            self.node_instance_type,
            self.expected_gpu_count,
            self.expected_efa_device_count,
        )

    def _begin_nvidia_smi_round(self) -> None:
        """Reset the per-round nvidia-smi memo and advance the circuit breaker."""

        self._nvidia_smi_round = {}
        if self._nvidia_smi_round_skipped:
            self._nvidia_smi_breaker_skips_remaining -= 1
        if self._nvidia_smi_round_timed_out:
            self._nvidia_smi_consecutive_timeouts += 1
            if self._nvidia_smi_consecutive_timeouts >= self.nvidia_smi_breaker_rounds:
                self._nvidia_smi_breaker_skips_remaining = (
                    self.nvidia_smi_breaker_cooldown_rounds
                )
                LOGGER.warning(
                    "nvidia-smi timed out %d rounds in a row; skipping GPU "
                    "queries for the next %d rounds",
                    self._nvidia_smi_consecutive_timeouts,
                    self.nvidia_smi_breaker_cooldown_rounds,
                )
        self._nvidia_smi_round_timed_out = False
        self._nvidia_smi_round_skipped = self._nvidia_smi_breaker_skips_remaining > 0

    def _nvidia_smi(
        self, argv: list[str], observed_at: datetime
    ) -> subprocess.CompletedProcess[str]:
        """Run one nvidia-smi query per round, bounded and circuit-broken.

        Three contributors used to shell out separately with 30 s timeouts,
        so a hung driver stretched a 15 s round to 90 s and the batch never
        said why. The result is memoised per round so the utilization and
        inventory readers share one query, a timeout is bounded per call,
        and after ``nvidia_smi_breaker_rounds`` consecutive timed-out rounds
        the queries are skipped for a cooldown with the reason recorded as a
        collection error rather than as a slow, quiet node.
        """

        if observed_at != self._nvidia_smi_round_at:
            # A contributor called outside ``collect_once`` with a new sample
            # time starts a fresh memo; the round bookkeeping is unchanged.
            self._nvidia_smi_round = {}
            self._nvidia_smi_round_at = observed_at
        key = tuple(argv)
        cached = self._nvidia_smi_round.get(key)
        if isinstance(cached, CollectorError):
            # The same query already hung this round; a second caller must
            # not pay the timeout again.
            raise cached
        if cached is not None:
            return cached
        if self._nvidia_smi_round_skipped:
            raise CollectorError(
                "nvidia-smi circuit breaker open "
                f"({self._nvidia_smi_consecutive_timeouts} consecutive timeouts); "
                "GPU queries skipped this round"
            )
        try:
            completed: subprocess.CompletedProcess[str] = self.runner(
                argv,
                capture_output=True,
                text=True,
                timeout=self.nvidia_smi_timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            self._nvidia_smi_round_timed_out = True
            error = CollectorError(
                f"nvidia-smi timed out after {self.nvidia_smi_timeout_seconds:g}s"
            )
            error.__cause__ = exc
            self._nvidia_smi_round[key] = error
            raise error
        self._nvidia_smi_consecutive_timeouts = 0
        self._nvidia_smi_round[key] = completed
        return completed

    def collect_once(self) -> HostTelemetryBatch:
        observed_at = self.now()
        samples: list[HostMetricSample] = []
        errors: list[str] = []
        self._begin_nvidia_smi_round()
        self._nvidia_smi_round_at = observed_at
        for name in self.CONTRIBUTORS:
            collector = getattr(self, name)
            try:
                samples.extend(collector(observed_at))
            except Exception as exc:
                errors.append(f"{name}: {type(exc).__name__}: {exc}")
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


def build_from_environment(
    sink: EventSink, context: CollectorContext, arguments: argparse.Namespace
) -> HostTelemetryCollector:
    """The ``gpu-fault-collector host`` factory named by the registry."""

    if not arguments.node_id:
        raise SystemExit("--node-id, NODE_NAME, or HOSTNAME is required")
    expected_gpu_count = os.getenv("GPU_FAULT_EXPECTED_GPU_COUNT")
    expected_efa_device_count = os.getenv("GPU_FAULT_EXPECTED_EFA_DEVICE_COUNT")
    return HostTelemetryCollector(
        sink,
        context,
        node_id=arguments.node_id,
        interval_seconds=arguments.interval_seconds,
        filesystems=[
            item
            for item in os.getenv("GPU_FAULT_FILESYSTEMS", "/,/var,/tmp").split(",")
            if item
        ],
        required_interfaces=[
            item
            for item in os.getenv("GPU_FAULT_REQUIRED_INTERFACES", "").split(",")
            if item
        ],
        pci_devices_root=(os.getenv("GPU_FAULT_PCI_DEVICES_ROOT") or None),
        node_instance_type=(os.getenv("GPU_FAULT_NODE_INSTANCE_TYPE") or None),
        expected_gpu_count=(int(expected_gpu_count) if expected_gpu_count else None),
        expected_efa_device_count=(
            int(expected_efa_device_count) if expected_efa_device_count else None
        ),
        inventory_mismatch_consecutive_samples=int(
            os.getenv(
                "GPU_FAULT_INVENTORY_MISMATCH_CONSECUTIVE_SAMPLES",
                "2",
            )
        ),
    )

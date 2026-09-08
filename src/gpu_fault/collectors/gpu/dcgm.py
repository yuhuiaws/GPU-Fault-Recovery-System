from __future__ import annotations

import argparse
import logging
import math
import os
import subprocess
import time
from collections import OrderedDict, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from gpu_fault.channel_registry import GPU_METRICS_PATH
from gpu_fault.collectors.gpu.discovery import (
    deliver_gpu_inventory,
    query_nvidia_temperature_limits,
)
from gpu_fault.collectors.models import CollectorContext
from gpu_fault.collectors.scheduling import next_stable_phase
from gpu_fault.collectors.sinks import CollectorError, EventSink
from gpu_fault.dcgm_fields import missing_dcgm_metric_groups
from gpu_fault.env import env_bool
from gpu_fault.gpu_metrics import (
    GpuMetricBatch,
    GpuMetricHistoryPoint,
    GpuMetricSample,
    GpuMetricSource,
    GpuMetricsThresholds,
)
from gpu_fault.transport.http_client import urlopen

LOGGER = logging.getLogger(__name__)

DCGM_BLANK_MAGNITUDE = 9.0e18
# A violation counter cannot accumulate more time than has elapsed. The small
# margin absorbs the skew between the exporter's own sampling and ours.
DUTY_CYCLE_MAX_PLAUSIBLE = 1.05
DCGM_NANOSECOND_DURATION_FIELDS = {
    "power_violation_total_us",
    "thermal_violation_total_us",
}

DCGM_METRICS: dict[str, tuple[str, str | None]] = {
    "DCGM_FI_DEV_GPU_TEMP": ("gpu_temperature_c", "celsius"),
    "DCGM_FI_DEV_MEMORY_TEMP": (
        "memory_temperature_c",
        "celsius",
    ),
    "DCGM_FI_DEV_POWER_USAGE": ("power_usage_w", "watts"),
    "DCGM_FI_DEV_POWER_MGMT_LIMIT": (
        "power_limit_w",
        "watts",
    ),
    "DCGM_FI_DEV_GPU_UTIL": ("gpu_utilization_percent", "percent"),
    "DCGM_FI_DEV_MEM_COPY_UTIL": (
        "memory_utilization_percent",
        "percent",
    ),
    "DCGM_FI_DEV_FB_USED": ("framebuffer_used_mib", "MiB"),
    "DCGM_FI_DEV_FB_FREE": ("framebuffer_free_mib", "MiB"),
    "DCGM_FI_DEV_SM_CLOCK": ("sm_clock_mhz", "MHz"),
    "DCGM_FI_DEV_MEM_CLOCK": ("memory_clock_mhz", "MHz"),
    "DCGM_FI_DEV_CLOCK_THROTTLE_REASONS": (
        "clock_throttle_reasons",
        "bitmask",
    ),
    "DCGM_FI_DEV_XID_ERRORS": ("xid_last_error", None),
    "DCGM_FI_DEV_ECC_SBE_VOL_TOTAL": (
        "ecc_sbe_volatile_total",
        "errors",
    ),
    "DCGM_FI_DEV_ECC_DBE_VOL_TOTAL": (
        "ecc_dbe_volatile_total",
        "errors",
    ),
    "DCGM_FI_DEV_ECC_SBE_AGG_TOTAL": (
        "ecc_sbe_aggregate_total",
        "errors",
    ),
    "DCGM_FI_DEV_ECC_DBE_AGG_TOTAL": (
        "ecc_dbe_aggregate_total",
        "errors",
    ),
    "DCGM_FI_DEV_RETIRED_SBE": (
        "retired_pages_sbe_total",
        "pages",
    ),
    "DCGM_FI_DEV_RETIRED_DBE": (
        "retired_pages_dbe_total",
        "pages",
    ),
    "DCGM_FI_DEV_RETIRED_PENDING": (
        "retired_pages_pending",
        "pages",
    ),
    "DCGM_FI_DEV_ROW_REMAP_FAILURE": (
        "row_remap_failure",
        None,
    ),
    "DCGM_FI_DEV_ROW_REMAP_PENDING": (
        "row_remap_pending",
        None,
    ),
    "DCGM_FI_DEV_CORRECTABLE_REMAPPED_ROWS": (
        "row_remap_correctable_total",
        "rows",
    ),
    "DCGM_FI_DEV_UNCORRECTABLE_REMAPPED_ROWS": (
        "row_remap_uncorrectable_total",
        "rows",
    ),
    "DCGM_FI_DEV_PCIE_REPLAY_COUNTER": (
        "pcie_replay_total",
        "replays",
    ),
    "DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_TOTAL": (
        "nvlink_crc_flit_error_total",
        "errors",
    ),
    "DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_TOTAL": (
        "nvlink_crc_data_error_total",
        "errors",
    ),
    "DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_TOTAL": (
        "nvlink_replay_error_total",
        "errors",
    ),
    "DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_TOTAL": (
        "nvlink_recovery_error_total",
        "errors",
    ),
    "DCGM_FI_DEV_NVLINK_ERROR_DL_CRC": (
        "nvlink_crc_aggregate_error_total",
        "errors",
    ),
    "DCGM_FI_DEV_NVLINK_ERROR_DL_RECOVERY": (
        "nvlink_recovery_aggregate_error_total",
        "errors",
    ),
    "DCGM_FI_DEV_NVLINK_ERROR_DL_REPLAY": (
        "nvlink_replay_aggregate_error_total",
        "errors",
    ),
    "DCGM_FI_DEV_POWER_VIOLATION": (
        "power_violation_total_us",
        "microseconds",
    ),
    "DCGM_FI_DEV_THERMAL_VIOLATION": (
        "thermal_violation_total_us",
        "microseconds",
    ),
}


class DcgmMetricsCollector:
    def __init__(
        self,
        sink: EventSink,
        context: CollectorContext,
        *,
        node_id: str,
        metrics_url: str = "http://127.0.0.1:9400/metrics",
        interval_seconds: float = 15,
        now: Callable[[], datetime] | None = None,
        runner: Callable[..., subprocess.CompletedProcess[str]] = (subprocess.run),
        edge_filter_enabled: bool | None = None,
        health_summary_seconds: int | None = None,
        edge_confirmation_samples: int | None = None,
        history_max_points: int | None = None,
        inventory_interval_seconds: int | None = None,
        state_max_keys: int | None = None,
        startup_spread_seconds: int | None = None,
        force_snapshot_path: str | None = None,
        violation_duty_cycle_threshold: float | None = None,
    ) -> None:
        self.sink = sink
        self.context = context
        self.node_id = node_id
        self.metrics_url = metrics_url
        self.interval_seconds = interval_seconds
        self.now = now or (lambda: datetime.now(timezone.utc))
        self.runner = runner
        self.edge_filter_enabled = (
            edge_filter_enabled
            if edge_filter_enabled is not None
            else env_bool("GPU_FAULT_DCGM_EDGE_FILTER_ENABLED", True)
        )
        self.health_summary_seconds = (
            health_summary_seconds
            if health_summary_seconds is not None
            else int(os.getenv("GPU_FAULT_DCGM_HEALTH_SUMMARY_SECONDS", "300"))
        )
        self.edge_confirmation_samples = (
            edge_confirmation_samples
            if edge_confirmation_samples is not None
            else int(os.getenv("GPU_FAULT_DCGM_EDGE_CONFIRMATION_SAMPLES", "3"))
        )
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
        self.state_max_keys = (
            state_max_keys
            if state_max_keys is not None
            else int(os.getenv("GPU_FAULT_DCGM_STATE_MAX_KEYS", "4096"))
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
                "GPU_FAULT_GPU_HEALTH_SNAPSHOT_REQUEST_PATH",
                ("/var/lib/gpu-fault/health-snapshot/gpu.request"),
            )
        )
        history_points = (
            history_max_points
            if history_max_points is not None
            else int(os.getenv("GPU_FAULT_DCGM_HISTORY_MAX_POINTS", "20"))
        )
        # Throttle-duration counters advance by a few microseconds on
        # every healthy sample, so treating any change as an edge makes
        # the filter deliver every batch. They are graded by duty cycle
        # instead: the fraction of the sample interval spent throttled.
        self.violation_duty_cycle_threshold = (
            violation_duty_cycle_threshold
            if violation_duty_cycle_threshold is not None
            else float(
                os.getenv(
                    "GPU_FAULT_DCGM_VIOLATION_DUTY_CYCLE_THRESHOLD",
                    "0.05",
                )
            )
        )
        if (
            self.health_summary_seconds <= 0
            or self.edge_confirmation_samples <= 0
            or history_points <= 0
            or self.inventory_interval_seconds <= 0
            or self.state_max_keys <= 0
            or self.startup_spread_seconds <= 0
        ):
            raise ValueError("DCGM edge filter intervals and counts must be positive")
        if not 0 < self.violation_duty_cycle_threshold <= 1:
            raise ValueError(
                "DCGM violation duty cycle threshold must be within (0, 1]"
            )
        self.thresholds = GpuMetricsThresholds.from_environment()
        self._history: deque[GpuMetricHistoryPoint] = deque(maxlen=history_points)
        self._previous_values: OrderedDict[str, float] = OrderedDict()
        self._candidate_streaks: dict[str, int] = {}
        self._implausible_duty_cycle_keys: set[str] = set()
        self._last_observed_at: datetime | None = None
        self._last_delivered_at: datetime | None = None
        self._last_inventory_delivered_at: datetime | None = None
        self._next_health_summary_at: datetime | None = None
        self._next_inventory_at: datetime | None = None
        self._temperature_limit_samples: list[GpuMetricSample] | None = None
        self._previous_devices: set[str] | None = None

    def collect_text(
        self,
        text: str,
        *,
        observed_at: datetime | None = None,
        extra_samples: list[GpuMetricSample] | None = None,
    ) -> GpuMetricBatch:
        try:
            from prometheus_client.parser import (
                text_string_to_metric_families,
            )
        except ImportError as exc:
            raise CollectorError(
                "install gpu-fault-control-plane[collectors] "
                "for DCGM Prometheus parsing"
            ) from exc

        samples = []
        for family in text_string_to_metric_families(text):
            for item in family.samples:
                mapping = DCGM_METRICS.get(item.name)
                if mapping is None and item.name.endswith("_total"):
                    mapping = DCGM_METRICS.get(item.name[:-6])
                if (
                    mapping is None
                    or not math.isfinite(item.value)
                    or abs(item.value) >= DCGM_BLANK_MAGNITUDE
                ):
                    continue
                labels = {str(key): str(value) for key, value in item.labels.items()}
                value = float(item.value)
                if mapping[0] in DCGM_NANOSECOND_DURATION_FIELDS:
                    value /= 1000.0
                samples.append(
                    GpuMetricSample(
                        metric_name=item.name,
                        canonical_name=mapping[0],
                        value=value,
                        unit=mapping[1],
                        gpu_index=labels.get("gpu"),
                        gpu_uuid=labels.get("UUID") or labels.get("uuid"),
                        pci_bdf=labels.get("pci_bus_id") or labels.get("pciBusId"),
                        labels=labels,
                    )
                )
        dcgm_sample_count = len(samples)
        samples.extend(extra_samples or [])
        timestamp = observed_at or self.now()
        if dcgm_sample_count == 0:
            raise CollectorError(
                "DCGM scrape contained no supported GPU metrics; "
                "verify deploy/dataplane/dcgm-counters.csv and "
                "exporter target"
            )
        observed_fields = {sample.canonical_name for sample in samples}
        missing_fields = missing_dcgm_metric_groups(observed_fields)
        batch = GpuMetricBatch(
            batch_id=(f"dcgm-{self.node_id}-{int(timestamp.timestamp() * 1_000_000)}"),
            cluster_id=self.context.cluster_id,
            node_id=self.node_id,
            observed_at=timestamp,
            collected_at=timestamp,
            source=GpuMetricSource.DCGM_EXPORTER,
            samples=samples,
            collection_errors=(
                [
                    "DCGM exporter is missing required fields: "
                    + ",".join(missing_fields)
                ]
                if missing_fields
                else []
            ),
            runtime_profile_version=(self.context.runtime_profile_version),
            product=self.context.product,
            driver_branch=self.context.driver_branch,
            cuda_version=self.context.cuda_version,
            workload_state=self.context.workload_state,
            affected_workload_ids=(self.context.affected_workload_ids),
            checkpoint_manifest_ref=(self.context.checkpoint_manifest_ref),
            evidence_ref=f"prometheus://{self.metrics_url}",
        )
        deliver, reasons = self._should_deliver(batch)
        self._history.append(
            GpuMetricHistoryPoint(
                observed_at=timestamp,
                samples=samples,
            )
        )
        if deliver:
            batch = batch.model_copy(
                update={
                    "edge_filter_reasons": reasons,
                }
            )
            self.sink.post(
                GPU_METRICS_PATH,
                batch.model_dump(mode="json"),
            )
            self._last_delivered_at = timestamp
            if (
                self._next_health_summary_at is None
                or timestamp >= self._next_health_summary_at
            ):
                self._next_health_summary_at = next_stable_phase(
                    timestamp,
                    cluster_id=self.context.cluster_id,
                    node_id=self.node_id,
                    channel="gpu-metrics",
                    interval_seconds=self.health_summary_seconds,
                )
            LOGGER.info(
                "delivered DCGM batch %s: %s",
                batch.batch_id,
                ",".join(reasons),
            )
        else:
            LOGGER.debug(
                "suppressed unchanged healthy DCGM batch %s",
                batch.batch_id,
            )
        return batch

    @staticmethod
    def _sample_key(sample: GpuMetricSample) -> str:
        device = sample.gpu_uuid or sample.pci_bdf or sample.gpu_index or "node"
        return f"{device}/{sample.canonical_name}"

    @staticmethod
    def _is_duty_cycle_counter(sample: GpuMetricSample) -> bool:
        """Counters that measure accumulated time, not events.

        ``power_violation_total_us`` and ``thermal_violation_total_us``
        creep upward on a perfectly healthy GPU, so an equality test
        against the previous sample is an edge on every batch. These are
        judged by rate instead.
        """

        return sample.canonical_name.endswith("_violation_total_us")

    @classmethod
    def _is_counter(cls, sample: GpuMetricSample) -> bool:
        if cls._is_duty_cycle_counter(sample):
            return False
        name = sample.canonical_name
        return name.endswith("_total") or name.endswith("_total_us")

    def _duty_cycle(
        self, sample: GpuMetricSample, observed_at: datetime
    ) -> float | None:
        """Fraction of the elapsed interval spent in violation.

        Returns ``None`` while no comparable previous sample exists, and
        ``0`` for a counter reset, which is not itself a fault edge.
        """

        if not self._is_duty_cycle_counter(sample):
            return None
        previous = self._previous_values.get(self._sample_key(sample))
        if previous is None or self._last_observed_at is None:
            return None
        elapsed = (observed_at - self._last_observed_at).total_seconds()
        if elapsed <= 0:
            return None
        delta = sample.value - previous
        if delta <= 0:
            return 0.0
        return delta / (elapsed * 1_000_000)

    def _duty_cycle_breached(
        self,
        sample: GpuMetricSample,
        duty_cycle: float,
    ) -> bool:
        """Whether a violation-duration counter is a candidate this sample.

        A duty cycle above 100% is not a hot GPU, it is a counter that does not
        hold the duration this code assumes. Live H200 nodes report
        ``DCGM_FI_DEV_POWER_VIOLATION`` advancing slightly *faster* than wall
        clock on a completely idle GPU, which made the duty cycle permanently
        breach the threshold: every GPU became a confirmed candidate about
        `edge_confirmation_samples` samples after the collector started, and
        because a confirmed candidate is announced once, real power throttling
        afterwards could never produce a delivery edge again. That is exactly
        the failure `GF-REGIONAL-COLLECT-002` exists to catch, reached through
        the counter rather than through the summary phase. Refusing to grade an
        impossible duty cycle keeps the noise guard without letting an unusable
        counter mask the fault it is supposed to reveal.
        """

        if duty_cycle < self.violation_duty_cycle_threshold:
            return False
        if duty_cycle <= DUTY_CYCLE_MAX_PLAUSIBLE:
            return True
        key = self._sample_key(sample)
        if key not in self._implausible_duty_cycle_keys:
            self._implausible_duty_cycle_keys.add(key)
            LOGGER.warning(
                "ignoring %s for edge detection on %s: duty cycle %.2f exceeds "
                "the elapsed interval, so the counter is not a duration in "
                "microseconds on this device",
                sample.canonical_name,
                key,
                duty_cycle,
            )
        return False

    def _candidate_keys(self, batch: GpuMetricBatch) -> set[str]:
        candidates: set[str] = set()
        values_by_device: dict[str, dict[str, float]] = {}
        for sample in batch.samples:
            device = sample.gpu_uuid or sample.pci_bdf or sample.gpu_index or "node"
            values_by_device.setdefault(device, {})[sample.canonical_name] = (
                sample.value
            )
            name = sample.canonical_name
            breached = (
                (
                    name == "gpu_temperature_c"
                    and sample.value >= self.thresholds.gpu_temperature_warning_c
                )
                or (
                    name == "memory_temperature_c"
                    and sample.value >= self.thresholds.memory_temperature_warning_c
                )
                or (
                    name
                    in {
                        "retired_pages_pending",
                        "row_remap_failure",
                        "row_remap_pending",
                    }
                    and sample.value > 0
                )
                or (name == "xid_last_error" and sample.value > 0)
            )
            duty_cycle = self._duty_cycle(sample, batch.observed_at)
            if duty_cycle is not None and self._duty_cycle_breached(sample, duty_cycle):
                breached = True
            if breached:
                candidates.add(self._sample_key(sample))
        for device, values in values_by_device.items():
            power = values.get("power_usage_w")
            limit = values.get("power_limit_w")
            utilization = values.get("gpu_utilization_percent")
            if (
                power is not None
                and limit is not None
                and limit > 0
                and utilization is not None
                and power / limit >= self.thresholds.power_limit_ratio
                and utilization
                >= self.thresholds.power_correlation_min_utilization_percent
            ):
                candidates.add(f"{device}/power_limit_correlation")
        return candidates

    @staticmethod
    def _device_keys(batch: GpuMetricBatch) -> set[str]:
        return {
            sample.gpu_uuid or sample.pci_bdf or sample.gpu_index or "node"
            for sample in batch.samples
        } - {"node"}

    def _lost_devices(self, batch: GpuMetricBatch) -> set[str]:
        """Devices the previous scrape reported and this one does not.

        A GPU that falls off the bus stops appearing in the exporter output;
        that used to register only as ``candidate-recovered`` (its confirmed
        candidate stopped being observed), which reads as good news.
        """

        devices = self._device_keys(batch)
        previous = self._previous_devices
        self._previous_devices = devices
        if previous is None:
            return set()
        return previous - devices

    def _should_deliver(self, batch: GpuMetricBatch) -> tuple[bool, list[str]]:
        lost_devices = self._lost_devices(batch)
        if not self.edge_filter_enabled:
            for sample in batch.samples:
                self._remember_value(self._sample_key(sample), sample.value)
            self._last_observed_at = batch.observed_at
            return True, ["filter-disabled"]
        reasons: list[str] = []
        if lost_devices:
            LOGGER.warning(
                "DCGM scrape lost %d device(s) since the previous sample: %s",
                len(lost_devices),
                ",".join(sorted(lost_devices)),
            )
            reasons.append("device-lost")
        if self._last_delivered_at is None:
            reasons.append("initial-baseline")
        elif (
            self._next_health_summary_at is not None
            and batch.observed_at >= self._next_health_summary_at
        ) or (
            self._next_health_summary_at is None
            and (batch.observed_at - self._last_delivered_at).total_seconds()
            >= self.health_summary_seconds
        ):
            reasons.append("health-summary")

        candidates = self._candidate_keys(batch)
        confirmed_candidates = {
            key
            for key, streak in self._candidate_streaks.items()
            if streak >= self.edge_confirmation_samples
        }
        recovered = {
            key
            for key in confirmed_candidates - candidates
            # A candidate whose device vanished has not recovered.
            if key.rsplit("/", 1)[0] not in lost_devices
        }
        if recovered:
            reasons.append("candidate-recovered")
        next_streaks: dict[str, int] = {}
        for key in sorted(candidates):
            streak = self._candidate_streaks.get(key, 0) + 1
            next_streaks[key] = streak
            if streak == self.edge_confirmation_samples:
                reasons.append("candidate-confirmed")
        self._candidate_streaks = dict(
            list(next_streaks.items())[-self.state_max_keys :]
        )

        for sample in batch.samples:
            key = self._sample_key(sample)
            previous = self._remember_value(key, sample.value)
            if (
                previous is not None
                and self._is_counter(sample)
                and sample.value != previous
            ):
                reasons.append(
                    "counter-increased" if sample.value > previous else "counter-reset"
                )
            if (
                sample.canonical_name == "xid_last_error"
                and previous is not None
                and sample.value != previous
            ):
                reasons.append("xid-changed")
        self._last_observed_at = batch.observed_at
        return bool(reasons), list(dict.fromkeys(reasons))

    def _remember_value(self, key: str, value: float) -> float | None:
        previous = self._previous_values.get(key)
        self._previous_values[key] = value
        self._previous_values.move_to_end(key)
        while len(self._previous_values) > self.state_max_keys:
            self._previous_values.popitem(last=False)
        return previous

    def collect_once(self) -> GpuMetricBatch:
        observed_at = self.now()
        if (
            self._last_inventory_delivered_at is None
            or (
                self._next_inventory_at is not None
                and observed_at >= self._next_inventory_at
            )
            or (
                self._next_inventory_at is None
                and self._last_inventory_delivered_at is not None
                and (observed_at - self._last_inventory_delivered_at).total_seconds()
                >= self.inventory_interval_seconds
            )
        ):
            try:
                snapshot = deliver_gpu_inventory(
                    self.sink,
                    self.context,
                    node_id=self.node_id,
                    observed_at=observed_at,
                    runner=self.runner,
                )
                self._last_inventory_delivered_at = observed_at
                self._next_inventory_at = next_stable_phase(
                    observed_at,
                    cluster_id=self.context.cluster_id,
                    node_id=self.node_id,
                    channel="gpu-inventory",
                    interval_seconds=self.inventory_interval_seconds,
                )
                LOGGER.info(
                    "delivered GPU inventory %s with %d devices",
                    snapshot.snapshot_id,
                    len(snapshot.devices),
                )
            except CollectorError:
                LOGGER.exception("mandatory GPU inventory delivery failed")
        try:
            with urlopen(self.metrics_url, timeout=10) as response:
                text = response.read().decode("utf-8", errors="replace")
        except OSError as exc:
            raise CollectorError(
                f"cannot scrape DCGM exporter: {self.metrics_url}"
            ) from exc
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
        return self.collect_text(
            text,
            observed_at=observed_at,
            extra_samples=self._temperature_limit_samples,
        )

    def run(self) -> None:
        force_snapshot = self.force_snapshot_path.exists()
        if not force_snapshot:
            startup_time = self.now()
            delay = (
                next_stable_phase(
                    startup_time,
                    cluster_id=self.context.cluster_id,
                    node_id=self.node_id,
                    channel="gpu-collector-startup",
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
                LOGGER.exception("DCGM metrics collection failed")
            if force_snapshot and succeeded:
                self.force_snapshot_path.unlink(missing_ok=True)
                force_snapshot = False
            time.sleep(self.interval_seconds)


def build_from_environment(
    sink: EventSink, context: CollectorContext, arguments: argparse.Namespace
) -> DcgmMetricsCollector:
    """The ``gpu-fault-collector dcgm`` factory named by the registry."""

    if not arguments.node_id:
        raise SystemExit("--node-id, NODE_NAME, or HOSTNAME is required")
    return DcgmMetricsCollector(
        sink,
        context,
        node_id=arguments.node_id,
        metrics_url=arguments.metrics_url,
        interval_seconds=arguments.interval_seconds,
    )

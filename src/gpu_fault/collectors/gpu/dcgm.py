from __future__ import annotations

import argparse
import logging
import math
import os
import subprocess
import time
from collections import OrderedDict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

from gpu_fault.channel_registry import GPU_METRICS_PATH
from gpu_fault.env import env_bool
from gpu_fault.gpu_metrics import (
    GpuMetricBatch,
    GpuMetricSample,
    GpuMetricSource,
    GpuMetricsThresholds,
)


from gpu_fault.collectors.gpu.discovery import (
    deliver_gpu_inventory,
    query_nvidia_temperature_limits,
)
from gpu_fault.transport.http_client import urlopen
from gpu_fault.collectors.models import CollectorContext
from gpu_fault.collectors.scheduling import next_stable_phase
from gpu_fault.collectors.sinks import (
    CollectorError,
    EventSink,
    deliver_or_raise,
)
from gpu_fault.dcgm_fields import missing_dcgm_metric_groups

LOGGER = logging.getLogger(__name__)

DCGM_BLANK_MAGNITUDE = 9.0e18
# A violation counter cannot accumulate more time than has elapsed. The small
# margin absorbs the skew between the exporter's own sampling and ours.
DUTY_CYCLE_MAX_PLAUSIBLE = 1.05
# How long a device's violation counter stays written off as "not a duration"
# before the collector says so again: long enough that a permanently broken
# counter cannot warn every tick, short enough that an operator who joined
# later still learns why the counter is ignored.
DUTY_CYCLE_IMPLAUSIBLE_EXPIRY_INTERVALS = 10
# A counter the exporter has not refreshed yet carries no information at all,
# so its confirmation streak is carried over instead of reset. Not forever: past
# this many collector intervals without a change the counter genuinely is not
# advancing, and a confirmed candidate must be allowed to recover.
#
# Invariant: the exporter's refresh period must stay at or below
# ``DUTY_CYCLE_STALE_CARRY_OVER_INTERVALS x interval_seconds`` (8 x 15 s = 120 s
# at the defaults, against the exporter's ``-c 15000``). Past that bound the
# carry-over expires between two refreshes, so an unchanged counter is graded as
# an idle GPU, the streak resets, and the original F2 symptom -- a sustained
# throttle that never confirms and never produces a delivery edge -- comes back
# silently: nothing in the metrics text says the exporter is slow. The collector
# checks the ratio at startup when the exporter's period is configured
# (``exporter_interval_seconds`` /
# ``GPU_FAULT_DCGM_EXPORTER_INTERVAL_SECONDS``), because DCGM does not publish
# its own collect interval.
DUTY_CYCLE_STALE_CARRY_OVER_INTERVALS = 8
# Consecutive inventory failures after which the delivery drops to the inventory
# cadence: a permanent expected-count mismatch or a wedged driver otherwise
# costs every tick one nvidia-smi subprocess and one traceback.
INVENTORY_FAILURE_BACKOFF_THRESHOLD = 3
# Consecutive temperature-limit failures after which the query drops to the
# inventory cadence instead of costing every tick a 15 s subprocess timeout.
TEMPERATURE_LIMIT_FAILURE_BACKOFF_THRESHOLD = 3
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


def _configured_exporter_interval_seconds() -> float | None:
    """The exporter's own collect period, if the deployment states it.

    ``None`` means unknown, which is not a fault: the exporter does not publish
    its collect interval, so an unset variable simply cannot be checked against
    :data:`DUTY_CYCLE_STALE_CARRY_OVER_INTERVALS`.
    """

    configured = os.getenv("GPU_FAULT_DCGM_EXPORTER_INTERVAL_SECONDS")
    if not configured or not configured.strip():
        return None
    return float(configured)


def _check_exporter_interval(
    exporter_interval_seconds: float | None, interval_seconds: float
) -> None:
    """Say once, at startup, when the carry-over cannot cover this exporter.

    See :data:`DUTY_CYCLE_STALE_CARRY_OVER_INTERVALS`: past that bound the
    carry-over expires between two refreshes, a sustained throttle stops
    confirming, and the only symptom is silence -- nothing in the metrics text
    says the exporter is slow. Startup rather than per tick because it is a
    configuration fact, and a warning repeated every 15 s for months is
    scrolled past. Unknown (no argument, no variable) is not a fault: DCGM does
    not publish its own collect interval.
    """

    configured = (
        exporter_interval_seconds
        if exporter_interval_seconds is not None
        else _configured_exporter_interval_seconds()
    )
    if configured is None:
        return
    if configured <= 0:
        raise ValueError("DCGM exporter interval must be positive")
    bound = interval_seconds * DUTY_CYCLE_STALE_CARRY_OVER_INTERVALS
    if configured <= bound:
        return
    LOGGER.warning(
        "the DCGM exporter refreshes every %.0fs, past the %.0fs the stale "
        "carry-over window covers (%d x the %.0fs collector interval): a "
        "sustained throttle can stop confirming between two refreshes and "
        "produce no delivery edge at all. Lower the exporter's -c, or raise "
        "GPU_FAULT_METRICS_INTERVAL_SECONDS so 8 x it covers the exporter",
        configured,
        bound,
        DUTY_CYCLE_STALE_CARRY_OVER_INTERVALS,
        interval_seconds,
    )


def _remember_bounded(
    table: OrderedDict[str, datetime], key: str, value: datetime, max_keys: int
) -> None:
    """Write ``key`` as the newest entry and evict the oldest past ``max_keys``.

    The write-off table is remembered only to rate-limit a warning, and it was
    unbounded: a node whose GPU UUIDs change -- a replaced device, a driver
    reload that renumbers -- added one permanent entry per broken counter in a
    process that runs for months. Bounded like the value LRU, oldest first.
    """

    table[key] = value
    table.move_to_end(key)
    if len(table) > max_keys:
        table.popitem(last=False)


def _merged_candidate_streaks(
    previous: dict[str, int],
    candidates: set[str],
    carried_over: set[str],
    confirm_at: int,
    max_keys: int,
) -> tuple[dict[str, int], int]:
    """Next tick's confirmation streaks, and how many keys just confirmed.

    A counter the exporter has not refreshed neither advances nor breaks a
    streak: rebuilding the streaks from this tick's candidates alone made a
    sustained throttle flap (confirm=1) or never confirm at all (the production
    confirm=3) whenever the exporter lagged the scrape. Carried-over keys are
    merged *first* because the bound keeps the last ``max_keys`` entries: on a
    node with more breaching keys than the bound, dropping a key this batch just
    observed breaching in favour of one it learned nothing about would keep a
    real fault from ever accumulating a streak.
    """

    merged: dict[str, int] = {}
    for key in sorted(carried_over):
        streak = previous.get(key, 0)
        if streak:
            merged[key] = streak
    confirmed = 0
    for key in sorted(candidates):
        streak = previous.get(key, 0) + 1
        merged[key] = streak
        if streak == confirm_at:
            confirmed += 1
    return dict(list(merged.items())[-max_keys:]), confirmed


class DcgmMetricsCollector:
    def __init__(
        self,
        sink: EventSink,
        context: CollectorContext,
        *,
        node_id: str,
        metrics_url: str = "http://127.0.0.1:9400/metrics",
        interval_seconds: float = 15,
        exporter_interval_seconds: float | None = None,
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
        failure_backoff_threshold: int = 3,
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
        # ``GpuMetricBatch.context_history`` has always shipped empty: the
        # control plane reads none of it, so the per-tick deque that fed it is
        # gone. The bound stays configured and validated because the installer
        # and the systemd unit still write it, and an operator who sets it to a
        # nonsense value deserves the same startup failure as before.
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
        _check_exporter_interval(exporter_interval_seconds, interval_seconds)
        self.failure_backoff_threshold = failure_backoff_threshold
        if self.failure_backoff_threshold <= 0:
            raise ValueError("DCGM failure backoff threshold must be positive")
        self.thresholds = GpuMetricsThresholds.from_environment()
        # Value *and* the time that value was first seen, per key: a counter is
        # graded against its own previous observation, never against the
        # collector's tick, so the exporter's collect interval cannot scale the
        # duty cycle (F2).
        self._previous_values: OrderedDict[str, tuple[float, datetime]] = OrderedDict()
        self._candidate_streaks: dict[str, int] = {}
        self._implausible_duty_cycle_keys: OrderedDict[str, datetime] = OrderedDict()
        self._last_delivered_at: datetime | None = None
        self._last_inventory_delivered_at: datetime | None = None
        self._next_health_summary_at: datetime | None = None
        self._next_inventory_at: datetime | None = None
        self._inventory_failures = 0
        self._next_inventory_attempt_at: datetime | None = None
        self._temperature_limit_samples: list[GpuMetricSample] | None = None
        self._temperature_limit_failures = 0
        self._next_temperature_limit_attempt_at: datetime | None = None
        self._previous_devices: set[str] | None = None
        self._consecutive_failures = 0

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
        if deliver:
            batch = batch.model_copy(
                update={
                    "edge_filter_reasons": reasons,
                }
            )
            deliver_or_raise(
                self.sink,
                GPU_METRICS_PATH,
                batch.model_dump(mode="json"),
                logger=LOGGER,
                what=f"DCGM batch {batch.batch_id}",
            )
            # The outbox replays what it took, so the edge filter advances for
            # BUFFERED exactly as for DELIVERED; only a batch that went nowhere
            # keeps the edge open (ARCH-G3).
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

    def _grade_duty_cycle(
        self, sample: GpuMetricSample, observed_at: datetime
    ) -> tuple[float | None, bool]:
        """The counter's duty cycle, and whether this tick told us nothing new.

        The window is measured from the last observation that carried a
        *different* value, not from the previous tick. The exporter refreshes
        its counters on its own schedule (30 s by default), so a collector
        scraping every 15 s reads the identical value on every other tick and
        then attributes one full exporter window of accumulated violation time
        to a single 15 s tick: a real 60% throttle graded 1.20, was written off
        as an implausible counter, and never produced a delivery edge (F2).

        An unrefreshed read is not the same observation as a counter reset, and
        conflating them cost the fix its point: grading the repeat 0.0 dropped
        the key out of the candidate set, so the confirmation streak was rebuilt
        from scratch on every other tick and at the default of three
        confirmations a sustained throttle never confirmed at all. The second
        element of the answer says "no new information": the caller carries the
        streak over and leaves the implausible write-off alone.

        Returns ``(None, False)`` when there is nothing to compare against,
        ``(0.0, False)`` for a counter reset -- not itself a fault edge -- and
        for a counter that has not moved for
        ``DUTY_CYCLE_STALE_CARRY_OVER_INTERVALS`` intervals, which is a real
        observation of an idle GPU rather than a lagging exporter.

        A window of zero or less -- two scrapes stamped alike, or a clock
        stepped backwards -- answers ``(None, True)``, the same "no new
        information" as an unrefreshed counter: calling it a reset would break a
        confirmed candidate's streak on a time correction.
        """

        if not self._is_duty_cycle_counter(sample):
            return None, False
        previous = self._previous_values.get(self._sample_key(sample))
        if previous is None:
            return None, False
        previous_value, previous_observed_at = previous
        elapsed = (observed_at - previous_observed_at).total_seconds()
        if elapsed <= 0:
            return None, True
        delta = sample.value - previous_value
        if delta < 0:
            return 0.0, False
        if delta == 0:
            stale_seconds = (
                self.interval_seconds * DUTY_CYCLE_STALE_CARRY_OVER_INTERVALS
            )
            if elapsed <= stale_seconds:
                return None, True
            return 0.0, False
        return delta / (elapsed * 1_000_000), False

    def _duty_cycle_breached(
        self,
        sample: GpuMetricSample,
        duty_cycle: float,
        observed_at: datetime,
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

        The write-off is remembered per key only to rate-limit the warning, and
        it expires after ``DUTY_CYCLE_IMPLAUSIBLE_EXPIRY_INTERVALS`` intervals:
        a counter that stays broken says so once per window rather than every
        tick, and a device whose counter becomes usable again is forgotten, so
        the next implausible value is reported immediately, and it is bounded by
        ``state_max_keys`` (see :func:`_remember_bounded`).
        """

        key = self._sample_key(sample)
        if duty_cycle < self.violation_duty_cycle_threshold:
            self._implausible_duty_cycle_keys.pop(key, None)
            return False
        if duty_cycle <= DUTY_CYCLE_MAX_PLAUSIBLE:
            self._implausible_duty_cycle_keys.pop(key, None)
            return True
        warned_at = self._implausible_duty_cycle_keys.get(key)
        expiry_seconds = self.interval_seconds * DUTY_CYCLE_IMPLAUSIBLE_EXPIRY_INTERVALS
        if (
            warned_at is None
            or (observed_at - warned_at).total_seconds() >= expiry_seconds
        ):
            _remember_bounded(
                self._implausible_duty_cycle_keys, key, observed_at, self.state_max_keys
            )
            LOGGER.warning(
                "ignoring %s for edge detection on %s: duty cycle %.2f exceeds "
                "the elapsed interval, so the counter is not a duration in "
                "microseconds on this device",
                sample.canonical_name,
                key,
                duty_cycle,
            )
        return False

    def _candidate_keys(self, batch: GpuMetricBatch) -> tuple[set[str], set[str]]:
        """The batch's candidate keys, and the keys it carries no news about.

        The second set holds duty-cycle counters the exporter has not refreshed
        since the previous scrape. They are neither candidates nor recoveries:
        their streak is carried over untouched.
        """

        candidates: set[str] = set()
        carried_over: set[str] = set()
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
            duty_cycle, no_new_value = self._grade_duty_cycle(sample, batch.observed_at)
            if no_new_value:
                carried_over.add(self._sample_key(sample))
            elif duty_cycle is not None and self._duty_cycle_breached(
                sample, duty_cycle, batch.observed_at
            ):
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
        return candidates, carried_over - candidates

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
                self._remember_value(
                    self._sample_key(sample), sample.value, batch.observed_at
                )
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

        candidates, carried_over = self._candidate_keys(batch)
        confirmed_candidates = {
            key
            for key, streak in self._candidate_streaks.items()
            if streak >= self.edge_confirmation_samples
        }
        recovered = {
            key
            for key in confirmed_candidates - candidates - carried_over
            # A candidate whose device vanished has not recovered.
            if key.rsplit("/", 1)[0] not in lost_devices
        }
        if recovered:
            reasons.append("candidate-recovered")
        self._candidate_streaks, confirmed = _merged_candidate_streaks(
            self._candidate_streaks,
            candidates,
            carried_over,
            self.edge_confirmation_samples,
            self.state_max_keys,
        )
        reasons.extend("candidate-confirmed" for _ in range(confirmed))

        for sample in batch.samples:
            key = self._sample_key(sample)
            previous = self._remember_value(key, sample.value, batch.observed_at)
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
        return bool(reasons), list(dict.fromkeys(reasons))

    def _remember_value(
        self, key: str, value: float, observed_at: datetime
    ) -> float | None:
        """Record this key's value and return the previous one.

        An unchanged value keeps the timestamp of the observation that first
        carried it, so a counter the exporter has not refreshed yet is graded
        over the window it actually accumulated in rather than over one
        collector tick (F2).
        """

        previous = self._previous_values.get(key)
        if previous is not None and previous[0] == value:
            self._previous_values[key] = (value, previous[1])
        else:
            self._previous_values[key] = (value, observed_at)
        self._previous_values.move_to_end(key)
        while len(self._previous_values) > self.state_max_keys:
            self._previous_values.popitem(last=False)
        return None if previous is None else previous[0]

    def _inventory_is_due(self, observed_at: datetime) -> bool:
        """Whether this tick owes an inventory delivery.

        A failing delivery used to be due again on the very next tick forever:
        a permanent ``--expected-gpu-count`` mismatch or a wedged driver cost
        every tick one nvidia-smi subprocess (up to its 15 s timeout) plus a
        full traceback. After ``INVENTORY_FAILURE_BACKOFF_THRESHOLD`` failures
        the retry drops to the inventory cadence, the way the temperature-limit
        probe does.
        """

        if (
            self._next_inventory_attempt_at is not None
            and observed_at < self._next_inventory_attempt_at
        ):
            return False
        if self._last_inventory_delivered_at is None:
            return True
        if self._next_inventory_at is not None:
            return observed_at >= self._next_inventory_at
        return (
            observed_at - self._last_inventory_delivered_at
        ).total_seconds() >= self.inventory_interval_seconds

    def collect_once(self) -> GpuMetricBatch:
        observed_at = self.now()
        if self._inventory_is_due(observed_at):
            try:
                snapshot = deliver_gpu_inventory(
                    self.sink,
                    self.context,
                    node_id=self.node_id,
                    observed_at=observed_at,
                    runner=self.runner,
                )
                self._last_inventory_delivered_at = observed_at
                self._inventory_failures = 0
                self._next_inventory_attempt_at = None
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
            # Every inventory failure mode is a CollectorError by contract
            # (`query_gpu_inventory`, `read_host_boot_id`, and the snapshot's own
            # pydantic validation), and none of them may cost the scrape its
            # tick: an operator `--expected-gpu-count` below the real device
            # count used to silence GPU_METRICS on the node entirely (F3). The
            # subprocess exceptions are caught as well so a future runner cannot
            # reopen that path.
            except (CollectorError, OSError, subprocess.TimeoutExpired) as exc:
                self._inventory_failures += 1
                if self._inventory_failures >= INVENTORY_FAILURE_BACKOFF_THRESHOLD:
                    self._next_inventory_attempt_at = observed_at + timedelta(
                        seconds=self.inventory_interval_seconds
                    )
                if self._inventory_failures == 1:
                    LOGGER.exception("mandatory GPU inventory delivery failed")
                else:
                    # One traceback is diagnosis; a traceback per tick is noise.
                    LOGGER.warning(
                        "mandatory GPU inventory delivery failed (%d consecutive): %s",
                        self._inventory_failures,
                        exc,
                    )
        try:
            with urlopen(self.metrics_url, timeout=10) as response:
                text = response.read().decode("utf-8", errors="replace")
        except OSError as exc:
            raise CollectorError(
                f"cannot scrape DCGM exporter: {self.metrics_url}"
            ) from exc
        self._refresh_temperature_limits(observed_at)
        return self.collect_text(
            text,
            observed_at=observed_at,
            extra_samples=self._temperature_limit_samples,
        )

    def _refresh_temperature_limits(self, observed_at: datetime) -> None:
        """Read the firmware temperature limits once, and never at the tick's cost.

        A wedged driver hangs ``nvidia-smi -q -x`` for its full 15 s timeout and
        then raises ``TimeoutExpired``, which is not a ``CollectorError``: it
        escaped the guard here *after* the DCGM scrape had already succeeded, so
        the node's telemetry went silent for the whole hang (F1). After
        ``TEMPERATURE_LIMIT_FAILURE_BACKOFF_THRESHOLD`` consecutive failures the
        retry drops to the inventory cadence, so a driver that stays wedged
        costs one probe per inventory interval instead of one per tick.
        """

        if self._temperature_limit_samples is not None:
            return
        if (
            self._next_temperature_limit_attempt_at is not None
            and observed_at < self._next_temperature_limit_attempt_at
        ):
            return
        try:
            discovered = query_nvidia_temperature_limits(self.runner)
        except (CollectorError, OSError, subprocess.TimeoutExpired) as exc:
            self._temperature_limit_unavailable(observed_at, str(exc))
            return
        if not discovered:
            # Valid XML that carries no threshold tags -- a vGPU or MIG device --
            # is not a success: treating it as one left the samples unset and
            # cleared the failure counter, so the probe ran on every tick for the
            # process's whole lifetime and no backoff could ever engage.
            self._temperature_limit_unavailable(
                observed_at, "nvidia-smi reported no temperature thresholds"
            )
            return
        self._temperature_limit_failures = 0
        self._next_temperature_limit_attempt_at = None
        self._temperature_limit_samples = discovered

    def _temperature_limit_unavailable(
        self, observed_at: datetime, detail: str
    ) -> None:
        self._temperature_limit_failures += 1
        if (
            self._temperature_limit_failures
            >= TEMPERATURE_LIMIT_FAILURE_BACKOFF_THRESHOLD
        ):
            self._next_temperature_limit_attempt_at = observed_at + timedelta(
                seconds=self.inventory_interval_seconds
            )
        LOGGER.warning(
            "NVIDIA temperature limits unavailable (%d consecutive); "
            "using configured fallback thresholds: %s",
            self._temperature_limit_failures,
            detail,
        )

    def _report_collection_error(self, exc: BaseException) -> None:
        """Deliver an erroring, sample-less batch so the node is not silent.

        DCGM mode used to log the failure and sleep: a dead exporter, a hung
        ``nvidia-smi``, or an unusable inventory reached the control plane only
        as GPU_METRICS silence, which cannot be told apart from a dead
        collector or a dead node (F5).
        """

        timestamp = self.now()
        batch = GpuMetricBatch(
            batch_id=(f"dcgm-{self.node_id}-{int(timestamp.timestamp() * 1_000_000)}"),
            cluster_id=self.context.cluster_id,
            node_id=self.node_id,
            observed_at=timestamp,
            collected_at=timestamp,
            source=GpuMetricSource.DCGM_EXPORTER,
            samples=[],
            collection_errors=[f"{type(exc).__name__}: {exc}"],
            edge_filter_reasons=["collection-error"],
            runtime_profile_version=(self.context.runtime_profile_version),
            product=self.context.product,
            driver_branch=self.context.driver_branch,
            cuda_version=self.context.cuda_version,
            workload_state=self.context.workload_state,
            affected_workload_ids=(self.context.affected_workload_ids),
            checkpoint_manifest_ref=(self.context.checkpoint_manifest_ref),
            evidence_ref=f"prometheus://{self.metrics_url}",
        )
        try:
            deliver_or_raise(
                self.sink,
                GPU_METRICS_PATH,
                batch.model_dump(mode="json"),
                logger=LOGGER,
                what=f"DCGM error batch {batch.batch_id}",
            )
        except Exception:
            LOGGER.exception("DCGM collection error report was not delivered")

    def next_interval_seconds(self) -> float:
        """The sleep before the next tick: the interval, doubled while failing.

        Mirrors the nvidia-smi collector: the first
        ``failure_backoff_threshold - 1`` failures keep the normal cadence so a
        transient blip is retried promptly; from the threshold on the wait
        doubles per failure, capped at four intervals.
        """

        excess = self._consecutive_failures - self.failure_backoff_threshold
        if excess < 0:
            return self.interval_seconds
        return min(
            self.interval_seconds * 4,
            self.interval_seconds * float(2 ** (excess + 1)),
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
                self._consecutive_failures = 0
            except Exception as exc:
                self._consecutive_failures += 1
                LOGGER.exception(
                    "DCGM metrics collection failed (%d consecutive)",
                    self._consecutive_failures,
                )
                self._report_collection_error(exc)
            if force_snapshot and succeeded:
                self.force_snapshot_path.unlink(missing_ok=True)
                force_snapshot = False
            time.sleep(self.next_interval_seconds())


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

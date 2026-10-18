"""The inventory and temperature-limit cadence both GPU collectors share.

DCGM mode grew these guards one incident at a time -- an operator
``--expected-gpu-count`` below the real device count silenced GPU_METRICS on
the node (F3), a wedged ``nvidia-smi -q -x`` escaped the guard *after* the
scrape had succeeded (F1), a vGPU with no threshold tags probed on every tick
for ever -- and the nvidia-smi fallback collector got none of them: its
inventory delivery ran outside any guard, so a wrong
``GPU_FAULT_EXPECTED_GPU_COUNT`` made every round raise and the node never
emitted one real sample in fallback mode. One implementation here, two
callers, so the two modes cannot drift apart again.
"""

from __future__ import annotations

import logging
import subprocess
from datetime import datetime, timedelta
from typing import Callable

from gpu_fault.gpu_metrics import GpuInventorySnapshot, GpuMetricSample
from gpu_fault.gpu_temperature_policy import TLIMIT_THRESHOLDS

from gpu_fault.collectors.gpu.discovery import (
    deliver_gpu_inventory,
    query_nvidia_temperature_limits,
)
from gpu_fault.collectors.models import CollectorContext
from gpu_fault.collectors.cadence import CadenceJitter, jittered_next_phase
from gpu_fault.collectors.sinks import CollectorError, EventSink

# Consecutive inventory failures after which the delivery drops to the inventory
# cadence: a permanent expected-count mismatch or a wedged driver otherwise
# costs every tick one nvidia-smi subprocess and one traceback.
INVENTORY_FAILURE_BACKOFF_THRESHOLD = 3
# Consecutive temperature-limit failures after which the query drops to the
# inventory cadence instead of costing every tick a 15 s subprocess timeout.
TEMPERATURE_LIMIT_FAILURE_BACKOFF_THRESHOLD = 3
# From the threshold on, each further failure doubles the retry interval, up to
# this cap: an answer that never carries thresholds (a vGPU, a MIG device, or an
# XML schema the parser does not know) is a property of the node, not of the
# minute, and used to cost one probe and one WARNING per minute for ever.
TEMPERATURE_LIMIT_RETRY_CAP_SECONDS = 3600

#: Every inventory failure mode is a ``CollectorError`` by contract
#: (``query_gpu_inventory``, ``read_host_boot_id`` and the snapshot's own
#: pydantic validation); the subprocess exceptions are caught as well so a
#: future runner cannot reopen the path where one took the tick down.
PROBE_FAILURES = (CollectorError, OSError, subprocess.TimeoutExpired)

Runner = Callable[..., subprocess.CompletedProcess[str]]


class GpuInventoryCadence:
    """When the GPU inventory is due, its guarded delivery, and the backoff.

    A failing delivery used to be due again on the very next tick for ever: a
    permanent ``--expected-gpu-count`` mismatch or a wedged driver cost every
    tick one nvidia-smi subprocess (up to its 15 s timeout) plus a full
    traceback. After ``INVENTORY_FAILURE_BACKOFF_THRESHOLD`` failures the
    retry drops to the inventory cadence, the way the temperature-limit probe
    does. None of the failure modes may cost the scrape its tick.
    """

    def __init__(
        self,
        *,
        interval_seconds: int,
        logger: logging.Logger,
        failure_backoff_threshold: int = INVENTORY_FAILURE_BACKOFF_THRESHOLD,
        jitter: CadenceJitter | None = None,
    ) -> None:
        if interval_seconds <= 0:
            raise ValueError("GPU inventory interval must be positive")
        self.interval_seconds = interval_seconds
        self.logger = logger
        self.jitter = jitter if jitter is not None else CadenceJitter.from_environment()
        self.failure_backoff_threshold = failure_backoff_threshold
        self.last_delivered_at: datetime | None = None
        self.next_at: datetime | None = None
        self.failures = 0
        self.next_attempt_at: datetime | None = None

    def is_due(self, observed_at: datetime) -> bool:
        """Whether this tick owes an inventory delivery."""

        if self.next_attempt_at is not None and observed_at < self.next_attempt_at:
            return False
        if self.last_delivered_at is None:
            return True
        if self.next_at is not None:
            return observed_at >= self.next_at
        return (
            observed_at - self.last_delivered_at
        ).total_seconds() >= self.interval_seconds

    def deliver_if_due(
        self,
        sink: EventSink,
        context: CollectorContext,
        *,
        node_id: str,
        observed_at: datetime,
        runner: Runner,
    ) -> GpuInventorySnapshot | None:
        """Deliver the inventory when due; ``None`` when not due or when it failed.

        A snapshot the outbox took is replayed from there, so the schedule
        advances for BUFFERED exactly as for DELIVERED; only a snapshot that
        went nowhere is a failure (ARCH-G3), and a failure never raises.
        """

        if not self.is_due(observed_at):
            return None
        try:
            snapshot = deliver_gpu_inventory(
                sink, context, node_id=node_id, observed_at=observed_at, runner=runner
            )
        except PROBE_FAILURES as exc:
            self._failed(observed_at, exc)
            return None
        self.last_delivered_at = observed_at
        self.failures = 0
        self.next_attempt_at = None
        self.next_at = jittered_next_phase(
            observed_at,
            cluster_id=context.cluster_id,
            node_id=node_id,
            channel="gpu-inventory",
            interval_seconds=self.interval_seconds,
            jitter=self.jitter,
        )
        self.logger.info(
            "delivered GPU inventory %s with %d devices",
            snapshot.snapshot_id,
            len(snapshot.devices),
        )
        return snapshot

    def _failed(self, observed_at: datetime, exc: BaseException) -> None:
        self.failures += 1
        if self.failures >= self.failure_backoff_threshold:
            self.next_attempt_at = observed_at + timedelta(
                seconds=self.interval_seconds
            )
        if self.failures == 1:
            self.logger.exception("mandatory GPU inventory delivery failed")
        else:
            # One traceback is diagnosis; a traceback per tick is noise.
            self.logger.warning(
                "mandatory GPU inventory delivery failed (%d consecutive): %s",
                self.failures,
                exc,
            )


class TemperatureLimitProbe:
    """Read the firmware temperature limits once, and never at the tick's cost.

    A wedged driver hangs ``nvidia-smi -q -x`` for its full 15 s timeout and
    then raises ``TimeoutExpired``, which is not a ``CollectorError``: it
    escaped the guard *after* the DCGM scrape had already succeeded, so the
    node's telemetry went silent for the whole hang (F1). Valid XML that
    carries no threshold tags -- a vGPU or MIG device -- is not a success
    either: treating it as one left the samples unset and cleared the failure
    counter, so the probe ran on every tick for the process's whole lifetime.
    After ``TEMPERATURE_LIMIT_FAILURE_BACKOFF_THRESHOLD`` consecutive failures
    the retry drops to ``retry_interval_seconds`` (the inventory cadence) and
    doubles with every further failure up to ``retry_cap_seconds``, so a driver
    that stays wedged costs one probe per hour at worst instead of one per tick.

    The fixed one-minute retry is what flooded the plain-EKS nodes: driver 580
    prints its thresholds as T.Limit tags the parser did not know, so every
    probe "failed", and the 60 s retry, rounded up to the next 15 s tick, logged
    one WARNING about every 64 s -- the 28 per 30 minutes of the live log. The
    counter only restarts with the process (a release roll, a quiesce or reset
    window, a TRIGGER_HEALTH_SNAPSHOT step, which restarts this unit, a reboot),
    and each restart probes three times in 30 s by design. A failure now warns
    once per distinct reason; repeats of it are DEBUG.
    """

    def __init__(
        self,
        *,
        retry_interval_seconds: int,
        logger: logging.Logger,
        failure_backoff_threshold: int = TEMPERATURE_LIMIT_FAILURE_BACKOFF_THRESHOLD,
        retry_cap_seconds: int = TEMPERATURE_LIMIT_RETRY_CAP_SECONDS,
    ) -> None:
        self.retry_interval_seconds = retry_interval_seconds
        self.retry_cap_seconds = max(retry_cap_seconds, retry_interval_seconds)
        self.logger = logger
        self.failure_backoff_threshold = failure_backoff_threshold
        self.samples: list[GpuMetricSample] | None = None
        self.failures = 0
        self.next_attempt_at: datetime | None = None
        self.last_failure: str | None = None

    @property
    def reports_tlimit(self) -> bool:
        """Whether the discovered limits are on the T.Limit scale."""

        return any(
            item.canonical_name in TLIMIT_THRESHOLDS for item in self.samples or ()
        )

    def refresh(self, observed_at: datetime, runner: Runner) -> list[GpuMetricSample]:
        """The cached limits, probing for them when unknown and not backed off."""

        if self.samples is not None:
            return self.samples
        if self.next_attempt_at is not None and observed_at < self.next_attempt_at:
            return []
        try:
            discovered = query_nvidia_temperature_limits(runner)
        except PROBE_FAILURES as exc:
            self._unavailable(observed_at, str(exc))
            return []
        if not discovered:
            self._unavailable(
                observed_at, "nvidia-smi reported no temperature thresholds"
            )
            return []
        if self.failures:
            self.logger.info(
                "NVIDIA temperature limits available after %d failed probe(s)",
                self.failures,
            )
        self.failures = 0
        self.next_attempt_at = None
        self.last_failure = None
        self.samples = discovered
        return discovered

    def retry_delay_seconds(self) -> int | None:
        """The wait after the current failure count, ``None`` below the threshold."""

        excess = self.failures - self.failure_backoff_threshold
        if excess < 0:
            return None
        # Capped exponent: the doubling is past the cap after a dozen failures.
        delay = self.retry_interval_seconds * (1 << min(excess, 32))
        return min(delay, self.retry_cap_seconds)

    def _unavailable(self, observed_at: datetime, detail: str) -> None:
        self.failures += 1
        delay = self.retry_delay_seconds()
        if delay is not None:
            self.next_attempt_at = observed_at + timedelta(seconds=delay)
        log = self.logger.debug if detail == self.last_failure else self.logger.warning
        self.last_failure = detail
        log(
            "NVIDIA temperature limits unavailable (%d consecutive); "
            "using configured fallback thresholds, next probe %s: %s",
            self.failures,
            "next tick" if delay is None else f"in {delay} s",
            detail,
        )

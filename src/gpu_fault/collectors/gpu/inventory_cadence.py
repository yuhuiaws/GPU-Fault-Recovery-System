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

from gpu_fault.collectors.gpu.discovery import (
    deliver_gpu_inventory,
    query_nvidia_temperature_limits,
)
from gpu_fault.collectors.models import CollectorContext
from gpu_fault.collectors.scheduling import next_stable_phase
from gpu_fault.collectors.sinks import CollectorError, EventSink

# Consecutive inventory failures after which the delivery drops to the inventory
# cadence: a permanent expected-count mismatch or a wedged driver otherwise
# costs every tick one nvidia-smi subprocess and one traceback.
INVENTORY_FAILURE_BACKOFF_THRESHOLD = 3
# Consecutive temperature-limit failures after which the query drops to the
# inventory cadence instead of costing every tick a 15 s subprocess timeout.
TEMPERATURE_LIMIT_FAILURE_BACKOFF_THRESHOLD = 3

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
    ) -> None:
        if interval_seconds <= 0:
            raise ValueError("GPU inventory interval must be positive")
        self.interval_seconds = interval_seconds
        self.logger = logger
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
        self.next_at = next_stable_phase(
            observed_at,
            cluster_id=context.cluster_id,
            node_id=node_id,
            channel="gpu-inventory",
            interval_seconds=self.interval_seconds,
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
    the retry drops to ``retry_interval_seconds`` (the inventory cadence), so a
    driver that stays wedged costs one probe per interval instead of one per
    tick.
    """

    def __init__(
        self,
        *,
        retry_interval_seconds: int,
        logger: logging.Logger,
        failure_backoff_threshold: int = TEMPERATURE_LIMIT_FAILURE_BACKOFF_THRESHOLD,
    ) -> None:
        self.retry_interval_seconds = retry_interval_seconds
        self.logger = logger
        self.failure_backoff_threshold = failure_backoff_threshold
        self.samples: list[GpuMetricSample] | None = None
        self.failures = 0
        self.next_attempt_at: datetime | None = None

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
        self.failures = 0
        self.next_attempt_at = None
        self.samples = discovered
        return discovered

    def _unavailable(self, observed_at: datetime, detail: str) -> None:
        self.failures += 1
        if self.failures >= self.failure_backoff_threshold:
            self.next_attempt_at = observed_at + timedelta(
                seconds=self.retry_interval_seconds
            )
        self.logger.warning(
            "NVIDIA temperature limits unavailable (%d consecutive); "
            "using configured fallback thresholds: %s",
            self.failures,
            detail,
        )

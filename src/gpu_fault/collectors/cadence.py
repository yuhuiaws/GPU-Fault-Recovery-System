"""Jitter for the node-side posting cadences.

Every periodic post from a node - the GPU inventory, the health summaries of
the metrics, host, kernel, Fabric Manager and node-log collectors, the node
Agent's heartbeat - runs on a fixed interval anchored to a per-node stable
phase (:mod:`gpu_fault.collectors.scheduling`), so in steady state the fleet
is already spread over each interval. Two things that spread does not cover:

* the phase is deterministic, so two nodes that hash to the same second post
  together on every cycle for the life of the fleet;
* after a fleet-wide restart the collectors' first posts follow the
  deterministic start-up spread (``GPU_FAULT_COLLECTOR_STARTUP_SPREAD_SECONDS``,
  15 s) and nothing else, so 12,800 nodes restarting together put their
  first posts into the same 15 s.

This module adds the random part: every repeating wait gets ``+-ratio`` of the
interval and every scheduled phase half of that
(``GPU_FAULT_COLLECTOR_CADENCE_JITTER_RATIO``, default 0.1), so either way two
consecutive posts are ``interval x (1 +- ratio)`` apart -- a 300 s summary
270-330 s after the previous one -- and a process adds a
uniform ``[0, GPU_FAULT_COLLECTOR_STARTUP_JITTER_SECONDS)`` (default 20 s) to
its start-up spread. The readiness thresholds
(``gpu_fault.collector_kinds.collector_silent_thresholds``) have to cover the
longest resulting gap; :func:`worst_case_silence_seconds` is that figure and
the registry test holds the two together.

The posts that prove a node alive are not delayed by the jitter: the kernel
collector's first health summary and the Agent's first heartbeat (its
registration) still go out at start-up.
"""

from __future__ import annotations

import os
import random
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta

from gpu_fault.collectors.scheduling import next_stable_phase

JITTER_RATIO_ENV = "GPU_FAULT_COLLECTOR_CADENCE_JITTER_RATIO"
STARTUP_JITTER_ENV = "GPU_FAULT_COLLECTOR_STARTUP_JITTER_SECONDS"
DEFAULT_JITTER_RATIO = 0.1
DEFAULT_STARTUP_JITTER_SECONDS = 20.0


@dataclass(frozen=True)
class CadenceJitter:
    """``ratio`` of an interval each cadence may move, and the random start-up offset."""

    ratio: float = DEFAULT_JITTER_RATIO
    startup_seconds: float = DEFAULT_STARTUP_JITTER_SECONDS

    @classmethod
    def from_environment(
        cls, environ: Mapping[str, str] | None = None
    ) -> CadenceJitter:
        env = os.environ if environ is None else environ
        ratio = float(env.get("GPU_FAULT_COLLECTOR_CADENCE_JITTER_RATIO", "0.1"))
        if ratio < 0:
            raise ValueError(f"{JITTER_RATIO_ENV} cannot be negative")
        if ratio > 0.5:
            raise ValueError(
                f"{JITTER_RATIO_ENV} above 0.5 would let two posts overlap"
            )
        startup = float(env.get("GPU_FAULT_COLLECTOR_STARTUP_JITTER_SECONDS", "20"))
        if startup < 0:
            raise ValueError(f"{STARTUP_JITTER_ENV} cannot be negative")
        return cls(ratio=ratio, startup_seconds=startup)

    def offset(
        self,
        interval_seconds: float,
        uniform: Callable[[float, float], float] = random.uniform,
    ) -> float:
        """A move of ``+-ratio x interval`` for one occurrence of a cadence."""

        if interval_seconds <= 0 or self.ratio <= 0:
            return 0.0
        spread = interval_seconds * self.ratio
        return min(max(uniform(-spread, spread), -spread), spread)

    def interval(
        self,
        interval_seconds: float,
        uniform: Callable[[float, float], float] = random.uniform,
    ) -> float:
        """One occurrence of a repeating wait: the interval plus its offset."""

        return max(0.0, interval_seconds + self.offset(interval_seconds, uniform))

    def max_interval(self, interval_seconds: float) -> float:
        """The longest wait :meth:`interval` can produce, and the longest spacing
        of two consecutive :func:`jittered_next_phase` posts (bar the tick)."""

        return interval_seconds * (1 + self.ratio)

    def startup_offset(
        self, uniform: Callable[[float, float], float] = random.uniform
    ) -> float:
        """The random part of a process's start-up spread, ``[0, startup_seconds)``."""

        if self.startup_seconds <= 0:
            return 0.0
        return min(max(uniform(0.0, self.startup_seconds), 0.0), self.startup_seconds)


def jittered_next_phase(
    observed_at: datetime,
    *,
    cluster_id: str,
    node_id: str,
    channel: str,
    interval_seconds: int,
    jitter: CadenceJitter,
    uniform: Callable[[float, float], float] = random.uniform,
) -> datetime:
    """The stable phase after the one ``observed_at`` served, moved by the jitter.

    The stable phase keeps the fleet spread across the interval and the node's
    schedule restart-stable; the offset breaks the lockstep of nodes that share
    a phase. Each phase moves by at most ``ratio x interval / 2``, so two
    consecutive posts are ``interval x (1 +- ratio)`` apart, the bound
    :func:`worst_case_silence_seconds` and the silence thresholds rely on.

    A post goes out on the first tick at or after its moved phase, which can be
    that half spread *before* the stable phase it serves, so the lookup starts
    half a spread past ``observed_at``. Looking up from ``observed_at`` itself
    found the phase just served a few seconds ahead and posted again on the
    next tick or two, so about every other summary went out two or three times
    and GF-REGIONAL-COLLECT-001 read the edge filter as not suppressing. The
    result is never before ``observed_at``: the offset cannot exceed the lead.
    """

    half_interval = interval_seconds / 2
    scheduled = next_stable_phase(
        observed_at + timedelta(seconds=half_interval * jitter.ratio),
        cluster_id=cluster_id,
        node_id=node_id,
        channel=channel,
        interval_seconds=interval_seconds,
    )
    return scheduled + timedelta(seconds=jitter.offset(half_interval, uniform))


def worst_case_silence_seconds(
    cadence_seconds: float,
    *,
    jitter: CadenceJitter,
    startup_spread_seconds: float,
    delivery_ladder_seconds: float,
) -> float:
    """The longest a healthy node can go without a delivered post on a cadence.

    One jittered interval at its longest, plus the start-up spread and its
    random part (a restarted collector's first post), plus the sink's retry
    ladder (a post that needed every attempt). A silence threshold below this
    alerts on a node that is merely late.
    """

    return (
        jitter.max_interval(cadence_seconds)
        + startup_spread_seconds
        + jitter.startup_seconds
        + delivery_ladder_seconds
    )

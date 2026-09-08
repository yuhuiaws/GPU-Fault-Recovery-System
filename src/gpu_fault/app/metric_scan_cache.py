"""Bounded, shared store scans for the ``/metrics`` render path.

Three metric families need the same two full-table reads: the closed-loop family
walks every persisted workflow, and both the closed-loop and attempt-ownership
families walk every registered agent. Prometheus scrapes on a fixed interval, so
without sharing, one scrape decodes the agent table twice and the whole workflow
table once -- and the workflow read was bounded only by ``limit=100_000``, which
is a memory ceiling rather than a scrape budget.

Two things are fixed here. The scans are shared behind a short TTL, so repeated
reads inside one render -- and across scrapes that arrive faster than the TTL --
cost one read instead of one per caller. And the workflow scan is bounded by a
configured limit, newest first, with the truncation published as its own metric
so a fleet that outgrows the budget says so instead of silently reporting the
newest slice as if it were everything.

The TTL is deliberately opt-outable: ``GPU_FAULT_METRICS_SCAN_TTL_SECONDS=0``
makes every call re-read, which is what a test that asserts freshness wants. The
defaults are spelled as literals in the ``os.getenv`` calls so the generated
environment reference prints a concrete value rather than a constant name.

Control-plane review 2026-09-08 (G-2, G-12, F-6, H2-3) changed three things.
The default TTL is 60 s. The ingress and worker Pods run four uvicorn
processes behind one port and ADOT scrapes a Pod every 15 s, so each process
answers about one scrape a minute and a 10 s TTL never survived between two
scrapes of the same process: every scrape was a fresh store aggregate. At 60 s
each process runs one aggregate per minute at most, however many scrapes (or
``process_metrics`` publisher ticks) it answers in between. The workflow detail
scan no longer asks for every
status newest-first -- that shape has no index (every ``updated_at`` index is
partial per status) and sorted the whole kind on disk -- but reads the open
statuses through the executable/BLOCKED partial indexes and the newest terminal
slice through ``gpu_fault_workflow_updated_all``. And the other whole-kind
aggregates a scrape used to run every time (orphan inspection, notification
outbox, remote-command backlog, spool depth) go through :meth:`shared` so they
too cost one read per TTL.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from threading import RLock
from typing import Any, Callable, TypeVar, cast

from gpu_fault.models import WorkflowStatus

T = TypeVar("T")

# The detail scan's two slices (G-2). Together they are every WorkflowStatus;
# the split is what lets each half be an index range scan.
OPEN_WORKFLOW_STATUSES: frozenset[WorkflowStatus] = frozenset(
    {
        WorkflowStatus.PENDING,
        WorkflowStatus.RUNNING,
        WorkflowStatus.SAFETY_PENDING,
        WorkflowStatus.BLOCKED,
    }
)
TERMINAL_WORKFLOW_STATUSES: frozenset[WorkflowStatus] = frozenset(
    set(WorkflowStatus) - OPEN_WORKFLOW_STATUSES
)


@dataclass(frozen=True)
class WorkflowScan:
    """A bounded slice of the workflow table plus what it left out."""

    workflows: tuple[Any, ...]
    limit: int
    truncated: bool


@dataclass(frozen=True)
class ObservationScan:
    """A bounded slice of the attempt-observation table plus what it left out."""

    states: tuple[Any, ...]
    limit: int
    truncated: bool


def scan_ttl_seconds_from_env() -> float:
    return max(
        0.0,
        float(os.getenv("GPU_FAULT_METRICS_SCAN_TTL_SECONDS", "60")),
    )


def workflow_scan_limit_from_env() -> int:
    return max(
        1,
        int(os.getenv("GPU_FAULT_METRICS_WORKFLOW_SCAN_LIMIT", "20000")),
    )


def observation_scan_limit_from_env() -> int:
    return max(
        1,
        int(os.getenv("GPU_FAULT_METRICS_OBSERVATION_SCAN_LIMIT", "20000")),
    )


class MetricScanCache:
    def __init__(
        self,
        store: Any,
        *,
        workflow_limit: int | None = None,
        observation_limit: int | None = None,
        ttl_seconds: float | None = None,
        monotonic: Callable[[], float] | None = None,
    ) -> None:
        self._store = store
        self.workflow_limit = (
            workflow_scan_limit_from_env() if workflow_limit is None else workflow_limit
        )
        self.observation_limit = (
            observation_scan_limit_from_env()
            if observation_limit is None
            else observation_limit
        )
        self.ttl_seconds = (
            scan_ttl_seconds_from_env() if ttl_seconds is None else ttl_seconds
        )
        self._monotonic = monotonic or time.monotonic
        self._lock = RLock()
        self._entries: dict[str, tuple[float, Any]] = {}

    def _cached(self, key: str, produce: Callable[[], T]) -> T:
        with self._lock:
            entry = self._entries.get(key)
            if entry is not None and self._monotonic() < entry[0]:
                return cast(T, entry[1])
        # Produced outside the lock: these are database reads, and holding the
        # lock across them would serialise concurrent scrapes behind the slowest
        # one instead of merely sharing their results.
        value = produce()
        with self._lock:
            self._entries[key] = (self._monotonic() + self.ttl_seconds, value)
        return value

    def agents(self) -> tuple[Any, ...]:
        return self._cached("agents", lambda: tuple(self._store.list_agents()))

    def workflows(self) -> WorkflowScan:
        return self._cached("workflows", self._read_workflows)

    def _read_workflows(self) -> WorkflowScan:
        # Two bounded slices rather than one all-status read (G-2). The open
        # slice is what the step-waiting, overdue and budget families describe
        # and is small by construction; the terminal slice is newest-first so
        # the duration and milestone summaries keep describing recent history
        # -- GpuFaultClosedLoopSlow takes a 6 h delta over them. One row over
        # the budget is requested on each so truncation is observed rather than
        # inferred from a full page, which cannot distinguish "exactly the
        # limit" from "more than the limit".
        limit = self.workflow_limit
        open_rows = list(
            self._store.list_workflows(
                set(OPEN_WORKFLOW_STATUSES),
                limit=limit + 1,
                newest_first=True,
            )
        )
        terminal_rows = list(
            self._store.list_workflows(
                set(TERMINAL_WORKFLOW_STATUSES),
                limit=limit + 1,
                newest_first=True,
            )
        )
        return WorkflowScan(
            workflows=(*open_rows[:limit], *terminal_rows[:limit]),
            limit=limit,
            truncated=len(open_rows) > limit or len(terminal_rows) > limit,
        )

    def shared(self, key: str, produce: Callable[[], T]) -> T:
        """One store aggregate per TTL, shared by every scrape of this process.

        For the whole-kind reads a contributor cannot bound (orphan inspection,
        notification outbox counts, remote-command backlog, spool depth). The
        key is the caller's; a contributor that reads the same aggregate with
        different arguments must use different keys.
        """

        return self._cached(f"shared:{key}", produce)

    def observation_states(self) -> ObservationScan:
        return self._cached("observation_states", self._read_observation_states)

    def _read_observation_states(self) -> ObservationScan:
        # The ownership family walked this table in full on every scrape, with no
        # cache and no bound. Attempt observations are retained for a week by
        # default (``GPU_FAULT_ATTEMPT_OBSERVATION_MAX_AGE_SECONDS``), so on a busy
        # fleet that is every training attempt from the last seven days, decoded
        # per scrape, on the request thread that serves ``/metrics``.
        #
        # Newest-first is the right slice to keep: the family reports fresh
        # attempts as ownership and observations past the freshness window as
        # staleness, and both of those live at the recent end. Dropping the oldest
        # rows first therefore loses the least interesting ones. As with
        # workflows, one row over the budget is requested so truncation is
        # observed rather than inferred.
        limit = self.observation_limit
        rows = list(
            self._store.list_attempt_observation_states(
                limit=limit + 1,
                newest_first=True,
            )
        )
        return ObservationScan(
            states=tuple(rows[:limit]),
            limit=limit,
            truncated=len(rows) > limit,
        )

    def invalidate(self) -> None:
        with self._lock:
            self._entries.clear()


def metric_scan_cache(runtime: Any) -> MetricScanCache:
    """Return the runtime's scan cache, creating an unshared one if absent.

    Plugin metric contributors and tests build an ``AppRuntime`` directly, so a
    missing cache must degrade to an uncached read rather than fail the scrape.
    """

    cache = getattr(runtime, "metric_scan_cache", None)
    if isinstance(cache, MetricScanCache):
        return cache
    return MetricScanCache(runtime.context.store)

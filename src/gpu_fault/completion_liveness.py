"""Liveness budget and progress stamping for the Completion Watcher loop.

The ``/healthz`` window is derived from one delivery through the sink chain;
``_ProgressStampingSink`` makes every delivery a progress step by
construction. Split out of ``completion_controller`` as a pure move (F6b).
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Callable

from gpu_fault.collectors import EventSink
from gpu_fault.collectors.sinks import RETRY_AFTER_CAP_SECONDS

# Borrowed on purpose: every line below used to be logged by
# ``gpu_fault.completion_controller`` and the log format prints the logger
# name, so the split must not rename what operators grep for.
LOGGER = logging.getLogger("gpu_fault.completion_controller")
# One budget answers both "how long may the loop spend inside a single
# blocking step before it counts as stuck?" (the /healthz liveness window, F3)
# and "how long does the end of a watch cycle wait for a debounce timer that is
# still reconciling?" (F5). They have to be the same number: a join that
# outlives the liveness window would guarantee a restart every time it fired,
# and a window shorter than one legal delivery would kill a working watcher
# mid-pass.
#
# The longest legal blocking step is one delivery, and one delivery is: every
# HTTP attempt at its socket timeout, a capped ``Retry-After`` sleep between
# them, and then the processor receipt poll that follows the accepted POST.
# Deliveries are what stamp progress (see ``_ProgressStampingSink``), so this
# really is the largest gap a working loop can produce; the margin covers the
# non-blocking bookkeeping around it.
PROGRESS_BUDGET_MARGIN_SECONDS = 60.0
# A liveness window may not grow without limit: a node whose telemetry stops
# reads UNKNOWN after 600 s and every node-mutating plan is then BLOCKED, so a
# watcher that is really wedged has to be replaced well inside that horizon.
# Only the delivery term is capped -- ``PROGRESS_BUDGET_RELISTS`` below is a
# floor, and lowering it under one relist would fail a healthy watch.
MAX_DELIVERY_BUDGET_SECONDS = 480.0
# A stuck watch has also missed this many relists. The floor scales with the
# watch timeout so an operator who raises
# GPU_FAULT_WATCHER_WATCH_TIMEOUT_SECONDS does not create a restart loop.
PROGRESS_BUDGET_RELISTS = 3
# Used only when the sink hides its timings (a test double, a future sink).
DEFAULT_SINK_RECEIPT_TIMEOUT_SECONDS = 120.0
DEFAULT_SINK_HTTP_TIMEOUT_SECONDS = 10.0
DEFAULT_SINK_MAX_ATTEMPTS = 4.0
# The sink may be an outbox wrapping the HTTP sink that owns the timeouts.
MAX_SINK_CHAIN_DEPTH = 5


class _ProgressStampingSink:
    """Sink proxy that stamps loop progress after every delivery returns.

    The liveness budget is expressed in *one delivery*, and one delivery is the
    only thing this loop does that can legitimately block for minutes (HTTP
    attempts, a capped ``Retry-After`` sleep, then the processor receipt poll).
    Stamping at the reconcile or attempt boundary was not enough: the outbox
    replay delivers one record per buffered event before the first attempt even
    starts, and a single failing attempt delivers an observation, a
    failure-detected event and a terminal. Putting the stamp here makes "one
    unstamped gap = at most one delivery" true by construction, so no future
    call site can reintroduce the gap.

    Everything else is delegated untouched, including the depth gauges and the
    ``replay``/``stats`` helpers, so the proxy is invisible to its callers.
    """

    #: Delivery entry points: their return -- success or failure -- is a step.
    STAMPED_METHODS = frozenset({"post", "deliver"})

    def __init__(self, sink: Any, note_progress: Callable[[], None]) -> None:
        self._sink = sink
        self._note_progress = note_progress

    @property
    def wrapped_sink(self) -> Any:
        """The sink underneath, for tests and for identity checks."""

        return self._sink

    def __getattr__(self, name: str) -> Any:
        value = getattr(self._sink, name)
        if name in self.STAMPED_METHODS and callable(value):
            return self._stamped(value)
        return value

    def _stamped(self, call: Callable[..., Any]) -> Callable[..., Any]:
        def stamped(*args: Any, **kwargs: Any) -> Any:
            try:
                return call(*args, **kwargs)
            finally:
                # ``finally``: a delivery that raised still proves the loop is
                # moving, and the retry path is the loop working (I2).
                self._note_progress()

        return stamped


def _stamping_sink(sink: Any, note_progress: Callable[[], None]) -> Any:
    """Wrap ``sink`` -- and the sink it wraps -- for progress stamping.

    ``KubernetesCompletionOutbox`` replays buffered records through the sink it
    holds, so that inner one is wrapped in place; otherwise a backlog replay is
    a single unstamped block at the top of every pass. A sink that will not
    accept the swap keeps the outer wrap alone, which is still correct for the
    live path.
    """

    inner = getattr(sink, "sink", None)
    if (
        inner is not None
        and not isinstance(inner, _ProgressStampingSink)
        and callable(getattr(inner, "post", None))
    ):
        try:
            sink.sink = _ProgressStampingSink(inner, note_progress)
        except (AttributeError, TypeError):
            LOGGER.warning(
                "cannot stamp progress on the buffered-record sink of %s; a "
                "long replay will not refresh the liveness clock",
                type(sink).__name__,
            )
    if isinstance(sink, _ProgressStampingSink):
        return sink
    return _ProgressStampingSink(sink, note_progress)


class CompletionLivenessMixin:
    """Liveness half of ``KubernetesCompletionController``."""

    # Attributes supplied by the composed concrete implementation.
    now: Callable[[], datetime]
    sink: EventSink
    watch_timeout_seconds: int

    def _sink_timing(self, attribute: str, default: float) -> float:
        """A delivery timeout read off the sink, or off the sink it wraps.

        ``KubernetesCompletionOutbox`` fronts the HTTP sink that owns the
        timeouts, so the value has to be looked up down the chain. Anything
        unreadable falls back to the shipped default: this feeds a liveness
        window, so it must never raise and never return zero.
        """

        sink: Any = self.sink
        for _ in range(MAX_SINK_CHAIN_DEPTH):
            if sink is None:
                break
            value = getattr(sink, attribute, None)
            try:
                if value is not None and float(value) > 0:
                    return float(value)
            except (TypeError, ValueError):
                pass
            sink = getattr(sink, "sink", None)
        return default

    @property
    def progress_stall_budget_seconds(self) -> float:
        """How long the loop may finish nothing before it counts as stuck.

        Derived, not configured: one whole delivery (every HTTP attempt plus
        the processor receipt poll) with a margin, and never less than
        ``PROGRESS_BUDGET_RELISTS`` watch timeouts. ``/healthz`` and the timer
        join in ``run_watch_cycle`` both read it, so a join that fires can
        never by itself push the Pod past the liveness window.
        """

        receipt = self._sink_timing(
            "processor_receipt_timeout_seconds",
            DEFAULT_SINK_RECEIPT_TIMEOUT_SECONDS,
        )
        http_timeout = self._sink_timing(
            "timeout_seconds", DEFAULT_SINK_HTTP_TIMEOUT_SECONDS
        )
        attempts = self._sink_timing("max_attempts", DEFAULT_SINK_MAX_ATTEMPTS)
        delivery = (
            receipt
            + http_timeout * attempts
            + RETRY_AFTER_CAP_SECONDS * max(attempts - 1.0, 0.0)
            + PROGRESS_BUDGET_MARGIN_SECONDS
        )
        relists = PROGRESS_BUDGET_RELISTS * float(self.watch_timeout_seconds)
        return max(min(delivery, MAX_DELIVERY_BUDGET_SECONDS), relists)

    def note_progress(self) -> None:
        """Record that the loop just finished a step (the liveness signal).

        Called around every blocking step rather than once per pass, so a slow
        pass reads as alive and only a step that never returns reads as stuck.
        """

        self.last_progress_at = self.now()

"""One bounded wait for the store at process start-up.

A control-plane process talks to Aurora before it can serve: ``PostgresStore``
opens its pool and validates the schema, the context syncs the configured
regional cluster registry and reads the registered clusters back, and the
regional registry runtime loads the durable head and writes its member row. A
Pod that starts during an Aurora writer failover finds no writer for 30-60 s,
and the pool answers ``PoolTimeout`` once its checkout timeout (10 s live) has
run out. Crashing then is the wrong answer: the restarted process needs Aurora
again to start, so the outage turns into a restart loop and the replacement
Pod comes up later than one that waited (GF-REGIONAL-HA-010 found the crash in
the context's schema check, which ran before the registry bootstrap, then the
only step with a retry).

So every one of those steps runs inside one :class:`StartupStoreWait`: one
``GPU_FAULT_STARTUP_STORE_RETRY_SECONDS`` budget (default 120 s) for the whole
start-up path, not per step -- :data:`STORE_STEP` and :data:`CONTEXT_STEP` in
``ApplicationContext.from_environment``, then :data:`REGISTRY_BOOTSTRAP_STEP`
in the factory with whatever is left. Each step retries with the registry
runtime's exponential backoff (``STARTUP_RETRY_INITIAL_SECONDS`` doubling to
``STARTUP_RETRY_MAX_INTERVAL_SECONDS``, never past the deadline), logs every
retry with the failure kind, the error class, the attempt number and the budget
left, and gives up with one :class:`StartupStoreUnavailable` once the deadline
has passed. Only an outage is waited for (:func:`startup_store_failure_kind`,
built on the store's one SQLSTATE table, ``gpu_fault.store.shared.errors``):

* the writer is unavailable -- ``psycopg_pool.PoolTimeout``, a lost or refused
  connection (``OperationalError`` without a SQLSTATE, class 08, 57P01-57P03,
  a session that landed on a reader), the server out of connections or memory
  (class 53), or a builtin ``ConnectionError``;
* a timeout -- ``ConnectionTimeout``, ``statement_timeout`` / ``lock_timeout``
  cancellations (57014, 55P03), a builtin ``TimeoutError``;
* a lost race -- serialization failure or deadlock (40001, 40P01) between
  processes that start at the same moment;
* the server refusing the credentials (SQLSTATE 28000/28P01 or libpq's FATAL
  text). This one is a decision, not a fact: during an Aurora credential
  rotation the window between RDS changing the password and the refresher
  writing the ``gpu-fault-aurora`` Secret (plus kubelet's projected-volume
  sync) refuses every new connection, the pool re-reads the mounted DSN on
  every connect, and behind the pool the refusal surfaces as the same
  ``PoolTimeout`` an outage does anyway. Waiting lets a Pod started inside that
  window come up on the refreshed password without a restart; a password that
  is simply wrong costs one budget before the same clear exit, and every retry
  line names ``authentication refused`` so it is not read as an outage.

Anything else fails at once: a schema that is not initialized or has the wrong
version, drifted triggers or indexes, a corrupt registry head, invalid
configuration (``ValueError``) -- a restart would meet the same state, and an
operator needs the message now, not after the budget. A zero budget keeps the
old fail-immediately behaviour for tools and tests: the first failure
propagates unchanged.

Messages are redacted (``gpu_fault.logging_setup.redact``) before they are
logged or quoted, and none of this module ever formats a DSN.
"""

from __future__ import annotations

import logging
import math
import os
import time
from collections.abc import Callable, Iterator, Mapping
from typing import TypeVar

from gpu_fault.logging_setup import redact
from gpu_fault.regional_registry_runtime import (
    STARTUP_RETRY_INITIAL_SECONDS,
    STARTUP_RETRY_MAX_INTERVAL_SECONDS,
)
from gpu_fault.store.postgres.pool import is_authentication_failure
from gpu_fault.store.shared.errors import operation_should_retry, writer_unavailable

LOGGER = logging.getLogger(__name__)

T = TypeVar("T")

STARTUP_STORE_RETRY_ENV = "GPU_FAULT_STARTUP_STORE_RETRY_SECONDS"

# The start-up steps that touch the store before a control-plane process
# serves, as the retry lines name them. GF-REGIONAL-HA-010 greps the last one's
# "regional registry bootstrap attempt" as supplementary evidence.
STORE_STEP = "control-plane store connection and schema check"
CONTEXT_STEP = "control-plane context"
REGISTRY_BOOTSTRAP_STEP = "regional registry bootstrap"

# The failure kinds start-up waits for, as the retry lines spell them.
UNAVAILABLE = "writer unavailable"
TIMEOUT = "timed out"
CONTENTION = "lost a race"
AUTHENTICATION = "authentication refused"

_CONTENTION_SQLSTATES = frozenset({"40001", "40P01"})
_MAX_CHAIN_DEPTH = 6
_MAX_MESSAGE_CHARS = 240


class StartupStoreUnavailable(RuntimeError):
    """Start-up spent its whole store budget and the store never answered.

    ``step`` is the start-up step that was still failing, ``attempts`` how often
    it ran, ``kind`` the last failure's kind; the last error is ``__cause__``.
    """

    def __init__(self, message: str, *, step: str, attempts: int, kind: str) -> None:
        super().__init__(message)
        self.step = step
        self.attempts = attempts
        self.kind = kind


def _chain(exc: BaseException) -> Iterator[BaseException]:
    seen: set[int] = set()
    current: BaseException | None = exc
    for _ in range(_MAX_CHAIN_DEPTH):
        if current is None or id(current) in seen:
            return
        seen.add(id(current))
        yield current
        current = current.__cause__ or current.__context__


def _sqlstate(error: BaseException) -> str | None:
    value = getattr(error, "sqlstate", None)
    return value if isinstance(value, str) else None


def _is_pool_closed(error: BaseException) -> bool:
    kind = type(error)
    return (kind.__module__ or "").startswith("psycopg_pool") and (
        kind.__name__ == "PoolClosed"
    )


def startup_store_failure_kind(exc: BaseException) -> str | None:
    """Why start-up may wait out ``exc``, or ``None`` when it must fail now.

    The answer is one of :data:`UNAVAILABLE`, :data:`TIMEOUT`,
    :data:`CONTENTION` and :data:`AUTHENTICATION` (see the module docstring for
    what each covers and why credentials count). ``PoolClosed`` is not an
    outage: start-up closed that pool itself, and retrying would use it again.
    """

    if isinstance(exc, StartupStoreUnavailable):
        return None
    chain = list(_chain(exc))
    if any(_is_pool_closed(error) for error in chain):
        return None
    if is_authentication_failure(exc):
        return AUTHENTICATION
    if writer_unavailable(exc):
        return UNAVAILABLE
    if operation_should_retry(exc):
        if any(_sqlstate(error) in _CONTENTION_SQLSTATES for error in chain):
            return CONTENTION
        return TIMEOUT
    for error in chain:
        if isinstance(error, TimeoutError):
            return TIMEOUT
        if isinstance(error, ConnectionError):
            return UNAVAILABLE
    return None


def describe_store_error(exc: BaseException) -> str:
    """``Class[SQLSTATE]: message`` on one redacted line, for a retry line."""

    sqlstate = _sqlstate(exc)
    label = type(exc).__name__ + (f"[{sqlstate}]" if sqlstate else "")
    message = " ".join(redact(str(exc)).split())
    if len(message) > _MAX_MESSAGE_CHARS:
        message = message[: _MAX_MESSAGE_CHARS - 3] + "..."
    return f"{label}: {message}" if message else label


def startup_store_budget_seconds(environ: Mapping[str, str] | None = None) -> float:
    """``GPU_FAULT_STARTUP_STORE_RETRY_SECONDS`` (default 120): finite, not negative.

    0 turns the wait off: the first store failure ends start-up, as it did
    before any step had a budget.
    """

    env = os.environ if environ is None else environ
    budget = float(env.get("GPU_FAULT_STARTUP_STORE_RETRY_SECONDS", "120"))
    if not math.isfinite(budget) or budget < 0:
        raise ValueError(
            f"{STARTUP_STORE_RETRY_ENV} must be a finite number of seconds, "
            f"0 or more; got {budget!r}"
        )
    return budget


class StartupStoreWait:
    """The start-up store budget of one process, shared by every step.

    The deadline is fixed when the wait is created; :meth:`run` retries one
    step against it, so a step that follows a long wait has only what is left.
    ``clock`` is monotonic seconds and ``sleep`` blocks for that many; both are
    injectable so a test can drive a failover without waiting through one.
    """

    def __init__(
        self,
        budget_seconds: float,
        *,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not math.isfinite(budget_seconds):
            raise ValueError("the start-up store budget must be finite")
        self.budget_seconds = max(0.0, float(budget_seconds))
        self.sleep = sleep
        self.clock = clock
        self.started_at = clock()
        self.deadline = self.started_at + self.budget_seconds

    @classmethod
    def from_environment(
        cls,
        environ: Mapping[str, str] | None = None,
        *,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> StartupStoreWait:
        return cls(startup_store_budget_seconds(environ), sleep=sleep, clock=clock)

    def remaining_seconds(self) -> float:
        return max(0.0, self.deadline - self.clock())

    def run(self, step: str, operation: Callable[[], T]) -> T:
        """``operation()``, retried while the store is out and budget is left."""

        interval = STARTUP_RETRY_INITIAL_SECONDS
        attempt = 0
        while True:
            attempt += 1
            try:
                return operation()
            except Exception as exc:
                kind = startup_store_failure_kind(exc)
                if kind is None or self.budget_seconds <= 0:
                    raise
                now = self.clock()
                remaining = self.deadline - now
                if remaining <= 0:
                    raise StartupStoreUnavailable(
                        f"start-up gave up waiting for the store after "
                        f"{now - self.started_at:.0f}s: {step} failed {attempt} "
                        f"time(s), the last one {kind} ({type(exc).__name__}), and "
                        f"the {self.budget_seconds:g}s {STARTUP_STORE_RETRY_ENV} "
                        "budget is spent",
                        step=step,
                        attempts=attempt,
                        kind=kind,
                    ) from exc
                wait = min(interval, STARTUP_RETRY_MAX_INTERVAL_SECONDS, remaining)
                LOGGER.warning(
                    "start-up store wait: %s attempt %d failed (%s, %s); "
                    "retrying in %.1fs with %.0fs of the %gs %s budget left",
                    step,
                    attempt,
                    kind,
                    describe_store_error(exc),
                    wait,
                    remaining,
                    self.budget_seconds,
                    STARTUP_STORE_RETRY_ENV,
                )
                self.sleep(wait)
                interval = min(interval * 2, STARTUP_RETRY_MAX_INTERVAL_SECONDS)

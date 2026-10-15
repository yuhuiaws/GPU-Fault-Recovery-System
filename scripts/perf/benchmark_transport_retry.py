"""The product executor's transport retries, for the perf suites' fake executors.

A formal integrated round (50 clusters) failed on one transient drop: three
executor Pods each had one claim and one result post closed by the control
plane without a response at the same moment. The claims were only counted,
but each lost result failed its Pod, the executor Job (``backoffLimit: 0``)
stopped its twenty healthy siblings, and the round waited out its timeout for
workflows nobody executed. The product executor rides out the same drop: its
client repeats a claim closed without a response, its loop backs claims off,
and it posts a result up to three times.

The fake executors (``benchmark_regional_action_executor.py``,
``benchmark_correlated_action_scenario.py``) run in plain
``python:3.12-slim`` Pods and cannot import the product, so this module, shipped
beside them, repeats that policy:

* a post that got no verdict -- the connection closed or reset before a
  status line, a read timeout, a TLS or connection error -- or one of the
  "not now" statuses is safe to repeat (:func:`retryable`); a refusal is not;
* a failed claim is repeated after the client's
  ``CLOSED_WITHOUT_RESPONSE_RETRY_SECONDS``, then on the executor loop's
  ``POLL_SECONDS * 2**n`` backoff up to ``CLAIM_BACKOFF_MAX_SECONDS``
  (:func:`claim_retry_seconds`); claims still failing after
  ``PERSISTENT_CLAIM_FAILURE_SECONDS`` (one production lease) are a persistent
  failure (:class:`ClaimRetries`);
* a result is posted at most ``RESULT_REPORT_ATTEMPTS`` times,
  ``RESULT_REPORT_BACKOFF_SECONDS`` apart with up to 50 % jitter
  (:func:`deliver`). The repeat carries the same body: the command's lease
  token is its idempotency key -- a command already terminal is returned
  unchanged -- so a result that landed before its response was lost is
  confirmed, not applied twice.

A retried failure is counted, never fatal; only a refusal, a persistent claim
failure or a result still undelivered after its attempts fails the Pod.
``tests/regional/test_perf_executor_transport.py`` pins the public constants
to ``gpu_fault.cluster_executor``.
"""

from __future__ import annotations

import random
import socket
import ssl
import time
from collections.abc import Callable
from http.client import HTTPException
from typing import Any
from urllib import error

#: ``regional_client.CLOSED_WITHOUT_RESPONSE_RETRY_SECONDS``: the client's one
#: repeat of a claim whose connection closed before any status line.
CLOSED_WITHOUT_RESPONSE_RETRY_SECONDS = 1.0
#: ``ClusterActionExecutor``'s defaults: the claim loop's poll and the cap of
#: its exponential backoff after consecutive failed claims.
POLL_SECONDS = 2.0
CLAIM_BACKOFF_MAX_SECONDS = 60.0
#: Claims failing for this long (one production lease) are a persistent
#: failure: every command the executor held has lost its lease by then.
PERSISTENT_CLAIM_FAILURE_SECONDS = 120.0
#: ``cluster_executor.lease``'s result report: three attempts, two backoffs.
RESULT_REPORT_ATTEMPTS = 3
RESULT_REPORT_BACKOFF_SECONDS = (0.5, 1.0)
RETRYABLE_STATUS_CODES = frozenset({408, 425, 429, 500, 502, 503, 504})
#: The failures that carry no verdict (``regional_client``'s transport set).
TRANSPORT_ERRORS: tuple[type[BaseException], ...] = (
    error.URLError,
    TimeoutError,
    socket.timeout,
    ssl.SSLError,
    ConnectionError,
    HTTPException,
)
SAMPLE_LIMIT = 20


def category(exc: BaseException) -> str:
    """A failure's class (and HTTP status): no URL, token or message."""

    if isinstance(exc, error.HTTPError):
        return f"{type(exc).__name__}:{exc.code}"
    if isinstance(exc, error.URLError):
        return f"{type(exc).__name__}:{type(exc.reason).__name__}"
    return type(exc).__name__


def retryable(exc: BaseException) -> bool:
    """No verdict from the control plane, or a "not now" status."""

    if isinstance(exc, error.HTTPError):
        return exc.code in RETRYABLE_STATUS_CODES
    return isinstance(exc, TRANSPORT_ERRORS)


def claim_retry_seconds(failures: int) -> float:
    """The pause before the next claim after ``failures`` consecutive failed
    claims: the client's repeat first, then the executor loop's backoff."""

    if failures <= 1:
        return CLOSED_WITHOUT_RESPONSE_RETRY_SECONDS
    return min(CLAIM_BACKOFF_MAX_SECONDS, POLL_SECONDS * 2 ** min(failures - 2, 8))


def report_backoff_seconds(
    attempt: int, jitter: Callable[[], float] = random.random
) -> float:
    """The jittered pause before result attempt ``attempt + 1``."""

    base = RESULT_REPORT_BACKOFF_SECONDS[
        min(attempt, len(RESULT_REPORT_BACKOFF_SECONDS)) - 1
    ]
    return base + jitter() * (base / 2)


class Undelivered(Exception):
    """A post that was refused, or still failed after its attempts."""

    def __init__(self, cause: BaseException, retried: list[str]) -> None:
        super().__init__(f"{category(cause)} after {len(retried)} retries")
        self.cause = cause
        self.category = category(cause)
        self.retried = retried


def deliver(
    post: Callable[[], Any],
    *,
    attempts: int = RESULT_REPORT_ATTEMPTS,
    sleep: Callable[[float], None] = time.sleep,
    jitter: Callable[[], float] = random.random,
) -> tuple[Any, list[str]]:
    """``post()`` until it lands: the answer and the failure categories it
    retried; :class:`Undelivered` when it was refused or ran out of attempts."""

    retried: list[str] = []
    for attempt in range(1, attempts + 1):
        try:
            return post(), retried
        except Exception as exc:
            if not retryable(exc) or attempt >= attempts:
                raise Undelivered(exc, retried) from exc
            retried.append(category(exc))
            sleep(report_backoff_seconds(attempt, jitter))
    raise AssertionError("deliver() must end inside its last attempt")


def count(counts: dict[str, int], key: str) -> None:
    counts[key] = counts.get(key, 0) + 1


class ClaimRetries:
    """One claim loop's failed claims: a pause while they are transient, the
    fatal verdict once one is refused or they persist."""

    def __init__(
        self,
        *,
        persistent_after_seconds: float | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.persistent_after_seconds = (
            PERSISTENT_CLAIM_FAILURE_SECONDS
            if persistent_after_seconds is None
            else persistent_after_seconds
        )
        self.clock = clock
        self.retries = 0
        self.retry_counts: dict[str, int] = {}
        self.retry_samples: list[dict[str, str | float]] = []
        self.consecutive = 0
        self.failing_since: float | None = None

    def succeeded(self) -> None:
        self.consecutive = 0
        self.failing_since = None

    def failed(self, exc: BaseException, *, elapsed: float) -> float | None:
        """The pause before the next claim, or ``None`` when this failure is
        fatal: a refusal, or claims failing for ``persistent_after_seconds``."""

        if not retryable(exc):
            return None
        now = self.clock()
        if self.failing_since is None:
            self.failing_since = now
        if now - self.failing_since >= self.persistent_after_seconds:
            return None
        self.consecutive += 1
        self.retries += 1
        kind = category(exc)
        count(self.retry_counts, kind)
        if len(self.retry_samples) < SAMPLE_LIMIT:
            self.retry_samples.append(
                {"category": kind, "elapsed_seconds": round(elapsed, 6)}
            )
        return claim_retry_seconds(self.consecutive)

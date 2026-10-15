"""An observation the engine could not make is not an unhealthy release.

``verify`` and ``stability`` are the two read-only modes a release driver rolls
back on (``scripts/release_deploy.py``), and until now any failure in them
counted. Live on 2026-10-15 both died two seconds in with ``exec credential
plugin exited 253`` -- the deploy host's ``aws eks get-token`` found no
instance-role credentials for a moment -- and the automatic rollback then took
a healthy config change back, seven minutes of forced restarts decided by an
observation that never happened.

``ApiUnavailable`` is a failure that says nothing about the release: the engine
could not authenticate to, or reach, an API it observes with. The credential
fetch raises it after its own retries (``regional_kubeconfig_cache``),
``Runner.run`` raises it for a ``kubectl`` or ``aws`` call whose client never
got an answer (``unavailable_failure``), and the AMP alert probe reports it. A
check that fails with it keeps ``{"unobservable": <reason>}`` as its evidence.

``ObservationBudget`` retries such an observation instead of judging it: verify
re-runs its report, the stability window takes its sample again and extends
itself by the time it could not see. The budget is the configured stability
window (``GPU_FAULT_RELEASE_STABILITY_SECONDS``, 120..300 s) of unobserved time
per run. Once it is spent the run still fails closed: ``HealthUnobservable``
exits with ``HEALTH_UNOBSERVABLE_EXIT_CODE`` and says that health could not be
observed, and the driver's automatic rollback follows ``spec.autoRollback`` as
for every verify or stability failure -- a release whose health was never
shown is not committed.
"""

from __future__ import annotations

import os
import re
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, TypeVar

from gpu_fault_release.regional_release_config import CheckFailure, ReleaseError
from gpu_fault_release.regional_release_narration import narrate_step

T = TypeVar("T")

# The fixed words a failure is reported with. Matched from what the client
# printed, never repeated from it: plugin output is not diagnostics.
REASONS = (
    "no-credentials",
    "expired",
    "access-denied",
    "throttled",
    "network",
    "unknown",
)
# Searched in this order, so a definitive cause (an expired session, a denied
# role) wins over the transport words a give-up message may also carry. The
# wording is the AWS CLI / botocore, client-go and urllib text for each cause.
_REASON_PATTERNS = (
    ("expired", re.compile(r"(?i)expired")),
    (
        "access-denied",
        re.compile(
            r"(?i)AccessDenied|not authorized to perform|UnauthorizedOperation"
            r"|InvalidClientTokenId|\(Forbidden\)"
        ),
    ),
    (
        "no-credentials",
        re.compile(
            r"(?i)unable to locate credentials|partial credentials found"
            r"|error when retrieving credentials|credentials are unavailable"
            r"|NoCredentialProviders"
        ),
    ),
    (
        "throttled",
        re.compile(
            r"(?i)throttl|rate exceeded|too ?many ?requests|RequestLimitExceeded"
            r"|SlowDown|ServiceUnavailable|service unavailable"
        ),
    ),
    (
        "network",
        re.compile(
            r"(?i)could not connect to the endpoint|connect timeout|read timeout"
            r"|timed out|connection (?:was )?(?:closed|reset|refused|aborted)"
            r"|network is unreachable|no route to host|name resolution"
            r"|name or service not known|no such host|i/o timeout"
            r"|tls handshake timeout|context deadline exceeded|urlopen error"
        ),
    ),
)
# The causes another attempt can outlast. An expired session or a denied role
# needs an operator, so a client call failing with one stays a failure.
TRANSIENT_REASONS = frozenset({"no-credentials", "throttled", "network"})
MAX_CLASSIFIED_CHARACTERS = 65536
# kubectl's own words for a request that never reached the API server or was
# refused before it was served. ``kubectl exec`` passes the remote process's
# stderr through and then says ``command terminated with exit code N``: that
# output is the Pod's -- a store the Pod cannot reach is a finding -- and is
# never read as the client's.
_KUBECTL_UNREACHED = re.compile(
    r"Unable to connect to the server|getting credentials: exec|\(TooManyRequests\)"
)
_KUBECTL_CREDENTIAL_PLUGIN = "getting credentials: exec"
_REMOTE_COMMAND_RAN = re.compile(r"command terminated with exit code \d+")

OBSERVATION_MODES = frozenset({"verify", "stability"})
# Pauses between attempts of one observation; the last repeats. The credential
# fetch has already retried for seconds; what is left is a blip of a minute.
OBSERVATION_RETRY_DELAYS = (5.0, 10.0, 20.0, 30.0)
STABILITY_SECONDS_ENV = "GPU_FAULT_RELEASE_STABILITY_SECONDS"
DEFAULT_OBSERVATION_SECONDS = 120
# Distinct from a report's verdicts (0 healthy, 1 unhealthy), an engine error
# (2) and the in-flight install refusal (3); the driver names it in its record
# (scripts/release_deploy.py mirrors it).
HEALTH_UNOBSERVABLE_EXIT_CODE = 4


class ApiUnavailable(CheckFailure):
    """The engine could not authenticate to, or reach, an API it observes with.

    Says nothing about the release. A ``CheckFailure`` so that a check failing
    with it keeps the reason as its evidence, which is how a verify report
    tells a check it could not observe from one that failed.
    """

    def __init__(self, message: str, *, reason: str) -> None:
        self.reason = reason if reason in REASONS else "unknown"
        super().__init__(message, {"unobservable": self.reason})


class HealthUnobservable(ReleaseError):
    """The observation budget ran out before health could be observed."""

    def __init__(self, message: str, *, last: ApiUnavailable) -> None:
        super().__init__(message)
        self.last = last


class _UnobservedReport(ApiUnavailable):
    """A verify report whose every failed check could not be observed."""

    def __init__(self, report: dict[str, Any], failed: list[tuple[str, str]]) -> None:
        super().__init__(
            "verify could not observe "
            + ", ".join(f"{name} ({reason})" for name, reason in failed),
            reason=failed[0][1],
        )
        self.report = report


def failure_reason(text: str | None) -> str:
    """The reason ``text`` names, from ``REASONS``; ``text`` itself never leaves."""

    tail = (text or "")[-MAX_CLASSIFIED_CHARACTERS:]
    for reason, pattern in _REASON_PATTERNS:
        if pattern.search(tail):
            return reason
    return "unknown"


def unavailable_reason(arguments: Sequence[str], stderr: str | None) -> str | None:
    """Why a failed ``kubectl``/``aws`` call never got an answer, or None.

    Only those two clients are read, and only for client-side causes: a
    credential plugin kubectl could not run (any reason: the engine could not
    authenticate), or a transient reason (``TRANSIENT_REASONS``) on a request
    that never got through. Everything else -- an RBAC refusal, a missing
    object, a remote command that ran and failed -- is None: a finding.
    """

    name = Path(arguments[0]).name if arguments else ""
    text = (stderr or "")[-MAX_CLASSIFIED_CHARACTERS:]
    reason = failure_reason(text)
    if name == "kubectl":
        if _REMOTE_COMMAND_RAN.search(text) or not _KUBECTL_UNREACHED.search(text):
            return None
        if reason not in TRANSIENT_REASONS and _KUBECTL_CREDENTIAL_PLUGIN not in text:
            return None
        return reason
    return reason if name == "aws" and reason in TRANSIENT_REASONS else None


def unavailable_failure(
    error: ReleaseError, arguments: Sequence[str], stderr: str | None
) -> ReleaseError:
    """``error``, or the same failure as ``ApiUnavailable`` if no API answered.

    ``unavailable_reason`` decides; the message and the exit-status attributes
    ``child_failure`` set are kept, so nothing an operator reads changes.
    """

    reason = unavailable_reason(arguments, stderr)
    if reason is None:
        return error
    unavailable = ApiUnavailable(str(error), reason=reason)
    for attribute, value in vars(error).items():
        setattr(unavailable, attribute, value)
    return unavailable


def observation_seconds(environ: Mapping[str, str] | None = None) -> int:
    """The configured stability window, which is also the observation budget.

    An invalid value is the stability window's to refuse
    (``validate_stability_window``); the budget falls back to the default
    rather than failing a verify that never reads the window.
    """

    raw = (os.environ if environ is None else environ).get(STABILITY_SECONDS_ENV, "")
    try:
        value = int(raw.strip() or DEFAULT_OBSERVATION_SECONDS)
    except ValueError:
        return DEFAULT_OBSERVATION_SECONDS
    return value if 120 <= value <= 300 else DEFAULT_OBSERVATION_SECONDS


class ObservationBudget:
    """The unobserved seconds one verify or stability run may spend.

    ``attempt`` runs an observation and, while it fails with
    ``ApiUnavailable``, pauses and runs it again. Each failed attempt and each
    pause is charged; the next pause that would overrun ``seconds`` raises
    ``HealthUnobservable`` instead. Any other failure passes through untouched:
    it is a finding, and judging findings is the caller's job.
    """

    def __init__(
        self,
        seconds: float,
        *,
        mode: str,
        sleep: Callable[[float], None] | None = None,
    ) -> None:
        self.seconds = float(seconds)
        self.mode = mode
        # Resolved per pause, so a caller's patched clock is the one it uses.
        self._sleep = sleep
        self.unobserved = 0.0
        self.retries = 0
        self.reasons: dict[str, int] = {}

    @classmethod
    def for_mode(cls, mode: str, *, dry_run: bool) -> ObservationBudget | None:
        if mode not in OBSERVATION_MODES or dry_run:
            return None
        return cls(observation_seconds(), mode=mode)

    def attempt(self, observe: Callable[[], T], *, what: str) -> T:
        delays = iter(OBSERVATION_RETRY_DELAYS)
        while True:
            # An interval, not a deadline: the window's own clock stays the
            # only monotonic reading an observation that succeeds costs.
            started = time.perf_counter()
            try:
                return observe()
            except ApiUnavailable as error:
                self._pause(
                    error,
                    what=what,
                    spent=time.perf_counter() - started,
                    delay=next(delays, OBSERVATION_RETRY_DELAYS[-1]),
                )

    def _pause(
        self, error: ApiUnavailable, *, what: str, spent: float, delay: float
    ) -> None:
        self.unobserved += max(0.0, spent)
        self.reasons[error.reason] = self.reasons.get(error.reason, 0) + 1
        if self.unobserved + delay > self.seconds:
            raise HealthUnobservable(
                f"release health could not be observed: for {self.unobserved:.0f}s "
                f"of its {self.seconds:.0f}s observation budget the {self.mode} "
                f"{what} could not authenticate to or reach an API "
                f"({error.reason}): {error}",
                last=error,
            ) from None
        self.retries += 1
        narrate_step(
            "observation-retry",
            mode=self.mode,
            what=what,
            reason=error.reason,
            unobserved=f"{self.unobserved:.0f}s/{self.seconds:.0f}s",
            next=f"{delay:.0f}s",
        )
        (self._sleep or time.sleep)(delay)
        self.unobserved += delay

    def summary(self, *, observed: bool = True) -> dict[str, Any]:
        return {
            "observed": observed,
            "budget_seconds": self.seconds,
            "unobserved_seconds": round(self.unobserved, 1),
            "retries": self.retries,
            "reasons": dict(sorted(self.reasons.items())),
        }


def observed(
    budget: ObservationBudget | None, observe: Callable[[], T], *, what: str
) -> T:
    """``observe()``, retried within ``budget`` when the run has one."""

    return observe() if budget is None else budget.attempt(observe, what=what)


def unobserved_checks(report: dict[str, Any]) -> list[tuple[str, str]] | None:
    """``(check, reason)`` when every failed check could not be observed, else None."""

    failed = [
        item
        for item in report.get("checks") or []
        if isinstance(item, dict) and item.get("status") == "FAIL"
    ]
    if report.get("healthy") or not failed:
        return None
    unobserved: list[tuple[str, str]] = []
    for item in failed:
        details = item.get("details")
        reason = details.get("unobservable") if isinstance(details, dict) else None
        if reason not in REASONS:
            return None
        unobserved.append((str(item.get("name") or "<unnamed>"), str(reason)))
    return unobserved


def observed_health_report(
    release: Any,
    budget: ObservationBudget | None,
    build: Callable[..., dict[str, Any]],
    exit_code: Callable[[dict[str, Any]], int],
) -> tuple[dict[str, Any], int]:
    """The verify report and its exit status, taken again while it is unobserved.

    A report whose only failures are checks it could not observe is not a
    verdict: it is built again within ``budget``. When the budget runs out the
    last such report is returned with ``observation.observed`` false and
    ``HEALTH_UNOBSERVABLE_EXIT_CODE``; a failure outside the checks (the state
    read) raises ``HealthUnobservable`` itself.
    """

    def observe() -> dict[str, Any]:
        release._apply_health_baseline(release._load_state())
        report = build(release, mode="verify")
        failed = unobserved_checks(report) if budget is not None else None
        if failed:
            raise _UnobservedReport(report, failed)
        return report

    if budget is None:
        report = observe()
        return report, exit_code(report)
    try:
        report = budget.attempt(observe, what="report")
    except HealthUnobservable as error:
        if not isinstance(error.last, _UnobservedReport):
            raise
        print(f"ERROR: {error}", file=sys.stderr, flush=True)
        report = {**error.last.report, "observation": budget.summary(observed=False)}
        return report, HEALTH_UNOBSERVABLE_EXIT_CODE
    if budget.retries:
        report = {**report, "observation": budget.summary()}
    return report, exit_code(report)

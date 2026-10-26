"""Bounded observation of one GF-ENV-LIVE replay, after the proxy is armed.

Arming the reviewed plan starts the experiment; it proves nothing yet. The
normally installed metrics collector reads the proxy once per collector
interval, its edge filter decides which of those reads it delivers, the control
plane grades the delivered batch, and the node-health path compiles and runs a
workflow. A verdict taken right after ``/arm`` sees none of that (live: the
GF-ENV-LIVE-004 snapshot was taken 2 s after arming and always failed "no
workflow", while the read-only cases passed on an empty store). This module
owns the wait:

* :class:`CollectorCadence` -- the node's own collector settings, as the host
  probe read them from ``collector.env``, with the product defaults: the
  interval, the edge confirmation, the health summary, and the start-up spread,
  start-up jitter and cadence jitter ratio every wait is sized from;
* :class:`ReplayProgress` -- whether the proxy served every planned read, from
  the ``/status`` ``reads`` / ``next_values`` / ``armed`` / ``aborted`` /
  ``failed_reads`` the tool reports (``/status`` never consumes a read), and --
  when no read arrived by the time the first one was due -- the re-verified
  repoint (:func:`diagnose_zero_reads`): the unit restarted, the drop-in gone,
  the endpoint reverted, the proxy refusing reads, or a collector that is
  simply silent, each named from the node's own state against the repoint
  receipt (live 2026-10-26, GF-ENV-LIVE-006: "consumed 0 of 7" said nothing
  about *why*; the proxy had answered every armed read 503);
* :func:`observation_window` -- the case's deadline: the product's own windows
  plus the collector interval and a scan margin, never beyond the proxy's
  ``max_seconds``, the collector deadman or the maintenance window, minus the
  time the runner's own cleanup needs;
* :func:`watch` -- the poll loop that feeds a judge (``env_live_judges``) until
  it reaches a verdict or the deadline.

Nothing here decides a verdict and nothing writes to a node or the store.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Protocol

from gpu_fault.collectors.cadence import (
    DEFAULT_JITTER_RATIO,
    DEFAULT_STARTUP_JITTER_SECONDS,
)
from gpu_fault.gpu_environment_policy import (
    CONFIRMATION_MAX_GAP_SECONDS,
    CONFIRMATION_SAMPLES,
)

#: The metrics collector's defaults when ``collector.env`` leaves a key unset:
#: ``collectors_cli._dcgm_arguments`` (GPU_FAULT_METRICS_INTERVAL_SECONDS "15")
#: and ``DcgmMetricsCollector`` (GPU_FAULT_DCGM_EDGE_CONFIRMATION_SAMPLES "3",
#: GPU_FAULT_DCGM_HEALTH_SUMMARY_SECONDS "300",
#: GPU_FAULT_COLLECTOR_STARTUP_SPREAD_SECONDS "15"); the random part of a
#: process's start-up and the ``+-ratio`` every repeating cadence may move are
#: ``collectors/cadence.CadenceJitter`` (GPU_FAULT_COLLECTOR_STARTUP_JITTER_
#: SECONDS "20", GPU_FAULT_COLLECTOR_CADENCE_JITTER_RATIO "0.1").
DEFAULT_INTERVAL_SECONDS = 15.0
DEFAULT_EDGE_CONFIRMATION_SAMPLES = 3
DEFAULT_HEALTH_SUMMARY_SECONDS = 300.0
STARTUP_SPREAD_SECONDS = 15.0
STARTUP_JITTER_SECONDS = float(DEFAULT_STARTUP_JITTER_SECONDS)
CADENCE_JITTER_RATIO = float(DEFAULT_JITTER_RATIO)
#: The ``collector.env`` keys the cadence is read from.
INTERVAL_KEY = "GPU_FAULT_METRICS_INTERVAL_SECONDS"
EDGE_CONFIRMATION_KEY = "GPU_FAULT_DCGM_EDGE_CONFIRMATION_SAMPLES"
HEALTH_SUMMARY_KEY = "GPU_FAULT_DCGM_HEALTH_SUMMARY_SECONDS"
STARTUP_SPREAD_KEY = "GPU_FAULT_COLLECTOR_STARTUP_SPREAD_SECONDS"
STARTUP_JITTER_KEY = "GPU_FAULT_COLLECTOR_STARTUP_JITTER_SECONDS"
JITTER_RATIO_KEY = "GPU_FAULT_COLLECTOR_CADENCE_JITTER_RATIO"
#: The constant reasons :func:`diagnose_zero_reads` names.
ZERO_READS_ENDPOINT_REVERTED = "endpoint-reverted"
ZERO_READS_DROPIN_GONE = "dropin-gone"
ZERO_READS_UNIT_RESTARTED = "unit-restarted"
ZERO_READS_PROXY_REFUSED = "proxy-refused-reads"
ZERO_READS_UNIT_INACTIVE = "unit-inactive"
ZERO_READS_COLLECTOR_SILENT = "collector-silent"
ZERO_READS_UNVERIFIED = "unverified"
#: The pass-through cycles GF-ENV-LIVE-001 needs through the proxy: "至少三次
#: 新采样", the same count the control plane confirms an environment fault on.
PASSTHROUGH_READS = CONFIRMATION_SAMPLES
#: A delivered batch's way into a record: the collector's post, the ingestion
#: queue and processor, and one store read of this runner.
PROCESSING_MARGIN_SECONDS = 30.0
#: Kept free before the earliest hard end (the proxy's replay window, the
#: collector deadman, the maintenance window) for one poll still in flight and
#: then restore-collector, stop-proxy and the probe delete, so the runner -- not
#: the deadman -- restores the unit.
CLEANUP_MARGIN_SECONDS = 90.0
#: A read is not started later than this past the cap; one already running
#: ends within its own timeout (``env_live_fixture`` bounds every poll read).
READ_GRACE_SECONDS = 30.0
MIN_POLL_SECONDS = 10.0
MAX_POLL_SECONDS = 30.0
TIMELINE_LIMIT = 80
READ_ERROR_LIMIT = 20


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


@dataclass(frozen=True)
class CollectorCadence:
    """How often the node's metrics collector reads and what it delivers."""

    interval_seconds: float = DEFAULT_INTERVAL_SECONDS
    edge_confirmation_samples: int = DEFAULT_EDGE_CONFIRMATION_SAMPLES
    health_summary_seconds: float = DEFAULT_HEALTH_SUMMARY_SECONDS
    metrics_mode: str = "dcgm"
    #: The keys taken from the node's collector.env (the rest are defaults).
    configured: tuple[str, ...] = ()
    #: A (re)started collector's first scrape waits for the deterministic
    #: start-up spread plus a random ``[0, startup_jitter_seconds)``; every
    #: repeating cadence may move by ``+-jitter_ratio`` of its interval.
    startup_spread_seconds: float = STARTUP_SPREAD_SECONDS
    startup_jitter_seconds: float = STARTUP_JITTER_SECONDS
    jitter_ratio: float = CADENCE_JITTER_RATIO

    @classmethod
    def from_probe(cls, probe_preflight: Mapping[str, Any] | None) -> CollectorCadence:
        environment = dict((probe_preflight or {}).get("collector_env") or {})
        configured: list[str] = []

        def setting(key: str, default: float, *, floor: float = 0.0) -> float:
            value = _number(environment.get(key))
            if value is None or value < floor or (floor == 0.0 and value <= 0):
                return default
            configured.append(key)
            return value

        interval = setting(INTERVAL_KEY, DEFAULT_INTERVAL_SECONDS)
        edge = setting(EDGE_CONFIRMATION_KEY, DEFAULT_EDGE_CONFIRMATION_SAMPLES)
        summary = setting(HEALTH_SUMMARY_KEY, DEFAULT_HEALTH_SUMMARY_SECONDS)
        # Zero is a valid spread, jitter or ratio (the product refuses only
        # negatives, and a ratio above 0.5); ``floor`` admits it.
        spread = setting(STARTUP_SPREAD_KEY, STARTUP_SPREAD_SECONDS, floor=-1.0)
        jitter = setting(STARTUP_JITTER_KEY, STARTUP_JITTER_SECONDS, floor=-1.0)
        ratio = setting(JITTER_RATIO_KEY, CADENCE_JITTER_RATIO, floor=-1.0)
        mode = str(environment.get("GPU_FAULT_METRICS_MODE") or "dcgm").strip()
        return cls(
            interval_seconds=interval,
            edge_confirmation_samples=max(1, int(edge)),
            health_summary_seconds=summary,
            metrics_mode=mode.lower() or "dcgm",
            configured=tuple(configured),
            startup_spread_seconds=max(0.0, spread),
            startup_jitter_seconds=max(0.0, jitter),
            jitter_ratio=min(0.5, max(0.0, ratio)),
        )

    @property
    def delivery_margin_seconds(self) -> float:
        """One more collector tick plus the ingestion path."""

        return self.interval_seconds + PROCESSING_MARGIN_SECONDS

    @property
    def poll_seconds(self) -> float:
        return min(MAX_POLL_SECONDS, max(MIN_POLL_SECONDS, self.interval_seconds))

    @property
    def max_interval_seconds(self) -> float:
        """One collector tick at its longest (``interval x (1 + ratio)``)."""

        return self.interval_seconds * (1.0 + self.jitter_ratio)

    @property
    def startup_allowance_seconds(self) -> float:
        """What a collector that (re)starts inside the window needs before its
        first scrape: the start-up spread plus the whole random start-up jitter."""

        return self.startup_spread_seconds + self.startup_jitter_seconds

    @property
    def first_read_due_seconds(self) -> float:
        """By when the first armed read must have reached the proxy.

        One tick at its longest after the arm for a running collector; a
        collector restarted right after the arm adds its start-up allowance.
        Zero reads past this point are re-verified against the repoint.
        """

        return self.max_interval_seconds + self.startup_allowance_seconds

    def replay_seconds(self, reads: int) -> float:
        """The collector consumes one proxy read per interval."""

        return max(0, reads) * self.interval_seconds

    def jitter_seconds(self, reads: int) -> float:
        """The most the cadence jitter can add over ``reads`` ticks."""

        return max(0, reads) * self.interval_seconds * self.jitter_ratio

    def confirmation_gap_ok(self) -> bool:
        """Consecutive samples can confirm only within the policy's 60 s gap."""

        return self.interval_seconds <= CONFIRMATION_MAX_GAP_SECONDS

    def evidence(self) -> dict[str, Any]:
        return {
            "interval_seconds": self.interval_seconds,
            "edge_confirmation_samples": self.edge_confirmation_samples,
            "health_summary_seconds": self.health_summary_seconds,
            "metrics_mode": self.metrics_mode,
            "startup_spread_seconds": self.startup_spread_seconds,
            "startup_jitter_seconds": self.startup_jitter_seconds,
            "jitter_ratio": self.jitter_ratio,
            "first_read_due_seconds": self.first_read_due_seconds,
            "from_collector_env": list(self.configured),
        }


def plan_reads(plan: Mapping[str, Any] | None) -> int:
    """The collector reads one armed plan consumes."""

    return sum(
        int(step.get("reads") or 0)
        for step in (plan or {}).get("steps") or []
        if isinstance(step, Mapping)
    )


def _reads_of(status: Mapping[str, Any], key: str = "reads") -> int | None:
    value = status.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def repoint_baseline(receipt: Mapping[str, Any] | None) -> dict[str, Any]:
    """What the repoint receipt pins about the collector unit.

    The zero-read re-verification compares the node's later ``snapshot``
    against these: the unit's ``InvocationID`` and ``ExecMainStartTimestamp``
    after the repoint restart, the run's drop-in path, the proxy endpoint the
    process was proven to read and the first-read proof.
    """

    value = dict(receipt or {})
    after = dict(value.get("after") or {})
    return {
        "proxy_endpoint": value.get("proxy_endpoint"),
        "original_endpoint": value.get("original_endpoint"),
        "dropin": value.get("dropin"),
        "invocation_id": after.get("InvocationID"),
        "exec_main_start": after.get("ExecMainStartTimestamp"),
        "main_pid": after.get("MainPID"),
        "first_read": value.get("first_read"),
    }


def diagnose_zero_reads(
    baseline: Mapping[str, Any],
    snapshot: Mapping[str, Any] | None,
    status: Mapping[str, Any] | None,
    *,
    proxy_endpoint: str,
) -> dict[str, Any]:
    """Name why an armed proxy counted no collector read, from the node's state.

    ``snapshot`` is the probe's ``snapshot`` (the metrics unit's ``systemctl
    show`` and the endpoint its process reads); ``status`` the latest proxy
    ``/status``. In order of decisiveness: the process no longer reads the
    proxy (``endpoint-reverted``), the run's drop-in is no longer loaded
    (``dropin-gone``), the unit restarted or is not active (``unit-restarted``,
    ``unit-inactive``), the proxy refused the reads it was asked for
    (``proxy-refused-reads``: live 2026-10-26 GF-ENV-LIVE-006, every armed read
    answered 503), or nothing changed and the collector is simply silent
    (``collector-silent``). An unreadable snapshot is ``unverified``.
    """

    status = dict(status or {})
    failed_reads = _reads_of(status, "failed_reads")
    last_error = status.get("last_error")
    facts: dict[str, Any] = {"failed_reads": failed_reads, "last_error": last_error}
    if snapshot is None:
        return {
            "reason": ZERO_READS_UNVERIFIED,
            "detail": "the node state could not be read; the repoint is unverified",
            **facts,
        }
    unit = dict(snapshot.get("metrics_unit") or {})
    effective = snapshot.get("effective_endpoint")
    dropin = baseline.get("dropin")
    loaded = str(dropin) in (unit.get("DropInPaths") or "").split() if dropin else None
    invocation = unit.get("InvocationID")
    started = unit.get("ExecMainStartTimestamp")
    restarted = bool(
        (
            baseline.get("invocation_id")
            and invocation
            and invocation != baseline.get("invocation_id")
        )
        or (
            baseline.get("exec_main_start")
            and started
            and started != baseline.get("exec_main_start")
        )
    )
    active = unit.get("ActiveState")
    facts.update(
        {
            "effective_endpoint": effective,
            "unit_active_state": active,
            "unit_invocation_id": invocation,
            "unit_exec_main_start": started,
            "dropin_loaded": loaded,
            "unit_restarted": restarted,
        }
    )
    restart_note = (
        f" (InvocationID {baseline.get('invocation_id')} -> {invocation}, "
        f"started {started})"
        if restarted
        else ""
    )
    if effective != proxy_endpoint:
        reason = ZERO_READS_ENDPOINT_REVERTED
        detail = (
            f"the collector process reads {effective!r}, not the proxy {proxy_endpoint}"
        )
        if loaded is False:
            detail += "; the repoint drop-in is no longer loaded"
        if restarted:
            detail += "; the unit restarted" + restart_note
    elif loaded is False:
        reason = ZERO_READS_DROPIN_GONE
        detail = (
            "the repoint drop-in is no longer loaded on the collector unit (its "
            "process still reads the proxy only until the next restart)"
        )
        if restarted:
            detail += "; the unit restarted" + restart_note
    elif active not in (None, "active"):
        reason = ZERO_READS_UNIT_INACTIVE
        detail = f"the collector unit is {active}, not active" + restart_note
    elif restarted:
        reason = ZERO_READS_UNIT_RESTARTED
        detail = (
            "the collector unit restarted after the repoint"
            + restart_note
            + "; the drop-in is loaded and the new process reads the proxy"
        )
    elif failed_reads:
        reason = ZERO_READS_PROXY_REFUSED
        detail = (
            f"the proxy answered {failed_reads} collector read(s) with 503 and "
            f"counted none (last error: {last_error})"
        )
    else:
        reason = ZERO_READS_COLLECTOR_SILENT
        detail = (
            "the collector unit is unchanged, the drop-in is loaded and the "
            "process reads the proxy, yet the proxy counted no read"
        )
        detail += (
            " and refused none"
            if failed_reads is not None
            else " (this proxy reports no refusal count)"
        )
    return {"reason": reason, "detail": detail, **facts}


@dataclass
class ReplayProgress:
    """Whether the proxy served every planned read (``/status`` only).

    ``start_reads`` is the ``reads`` the arm receipt's own ``/status`` reported.
    That read follows the ``POST /arm``, so it is at or after the plan's first
    read: ``reads - start_reads >= planned_reads`` can only hold once the tool
    served the whole plan (at worst one collector tick late, never early). A
    pass-through baseline never arms; there every proxy read is one collector
    scrape through the proxy (the proxy starts at zero and only the repointed
    collector reads ``/metrics``).
    """

    planned_reads: int
    armed: bool
    proxy_endpoint: str
    start_reads: int | None = None
    started_at: datetime | None = None
    reads: int | None = None
    next_values: dict[str, Any] = field(default_factory=dict)
    aborted: str | None = None
    endpoint_lost_at: datetime | None = None
    lost_endpoint: str | None = None
    consumed_at: datetime | None = None
    #: An armed proxy that reports itself unarmed (a restarted tool starts
    #: pass-through with a fresh read count) never served the rest of the plan.
    disarmed_at: datetime | None = None
    status_reads: int = 0
    status_errors: list[str] = field(default_factory=list)
    passthrough_violations: list[str] = field(default_factory=list)
    #: By when the first armed read is due (``CollectorCadence.first_read_due_
    #: seconds``); zero reads past it are re-verified against the repoint
    #: receipt (``repoint_baseline``). ``None`` disables the re-verification.
    first_read_due_seconds: float | None = None
    repoint_baseline: dict[str, Any] = field(default_factory=dict)
    #: The proxy's refused reads (``/status`` ``failed_reads`` / ``last_error``,
    #: tools that predate the field report ``None``).
    failed_reads: int | None = None
    last_error: str | None = None
    zero_reads_diagnosis: dict[str, Any] | None = None

    @property
    def required_reads(self) -> int:
        return self.planned_reads if self.armed else PASSTHROUGH_READS

    @property
    def served(self) -> int | None:
        if self.reads is None or self.start_reads is None:
            return None
        return max(0, self.reads - self.start_reads)

    @property
    def consumed(self) -> bool:
        return self.consumed_at is not None

    def begin(self, receipt: Mapping[str, Any] | None, at: datetime) -> None:
        """Start counting at the arm receipt (or at zero for the baseline)."""

        self.started_at = at
        status = dict((receipt or {}).get("status") or {})
        self.start_reads = _reads_of(status) if self.armed else 0
        self.update(receipt or {}, at)

    def update(self, receipt: Mapping[str, Any], at: datetime) -> None:
        status = dict(receipt.get("status") or {})
        if not status:
            self.status_errors.append("a /status read returned no status object")
            return
        self.status_reads += 1
        reads = _reads_of(status)
        if reads is not None:
            self.reads = reads
        failed = _reads_of(status, "failed_reads")
        if failed is not None:
            self.failed_reads = failed
        if status.get("last_error"):
            self.last_error = str(status["last_error"])
        self.next_values = dict(status.get("next_values") or {})
        if status.get("aborted"):
            self.aborted = str(status["aborted"])
        if "effective_endpoint" in receipt:
            endpoint = receipt.get("effective_endpoint")
            if endpoint != self.proxy_endpoint and self.endpoint_lost_at is None:
                self.endpoint_lost_at = at
                self.lost_endpoint = None if endpoint is None else str(endpoint)
        if not self.armed and (status.get("armed") is not False or self.next_values):
            self.passthrough_violations.append(
                f"{at.isoformat()}: pass-through proxy reported armed="
                f"{status.get('armed')} next_values={self.next_values}"
            )
        if self.armed and status.get("armed") is not True and self.consumed_at is None:
            self.disarmed_at = self.disarmed_at or at
        served = self.served
        if (
            self.consumed_at is None
            and served is not None
            and served >= self.required_reads
            and not self.next_values
            and self.endpoint_lost_at is None
            and self.disarmed_at is None
        ):
            self.consumed_at = at

    def inconclusive(self) -> list[str]:
        """Why the replay cannot be judged (empty once it was consumed)."""

        reasons: list[str] = []
        if self.aborted:
            reasons.append(
                f"the replay aborted on a real upstream safety reading: {self.aborted}"
            )
        if self.endpoint_lost_at is not None and not self.consumed:
            reasons.append(
                "the metrics collector stopped reading the proxy before the plan "
                f"was served (effective endpoint {self.lost_endpoint})"
            )
        if self.disarmed_at is not None and not self.consumed:
            reasons.append(
                "the proxy reported itself unarmed before the plan was served "
                f"({self.disarmed_at.isoformat()})"
            )
        if self.consumed:
            return reasons
        if self.start_reads is None:
            reasons.append("the arm receipt carried no proxy read count")
        elif self.status_reads == 0:
            reasons.append("no proxy /status read succeeded after arming")
        else:
            what = "planned replay reads" if self.armed else "pass-through reads"
            reasons.append(
                f"the collector consumed {self.served or 0} of {self.required_reads} "
                f"{what} before the deadline"
            )
        diagnosis = self.zero_reads_diagnosis
        if diagnosis is not None:
            reasons.append(
                f"re-verified repoint at {diagnosis.get('at')} "
                f"({diagnosis.get('reason')}): {diagnosis.get('detail')}"
            )
        if self.failed_reads and (
            diagnosis is None or diagnosis.get("reason") != ZERO_READS_PROXY_REFUSED
        ):
            reasons.append(
                f"the proxy refused {self.failed_reads} collector read(s) with 503 "
                f"(last error: {self.last_error})"
            )
        return reasons

    def needs_repoint_check(self, now: datetime) -> bool:
        """Zero reads although the first one was due, not yet re-verified."""

        if (
            not self.armed
            or self.consumed
            or self.zero_reads_diagnosis is not None
            or self.first_read_due_seconds is None
            or self.started_at is None
            or (self.served or 0) > 0
        ):
            return False
        return now >= self.started_at + timedelta(seconds=self.first_read_due_seconds)

    def diagnose(
        self, snapshot: Mapping[str, Any] | None, at: datetime
    ) -> dict[str, Any]:
        """Record the re-verified repoint (``diagnose_zero_reads``) once."""

        self.zero_reads_diagnosis = {
            "at": at.isoformat(),
            **diagnose_zero_reads(
                self.repoint_baseline,
                snapshot,
                {"failed_reads": self.failed_reads, "last_error": self.last_error},
                proxy_endpoint=self.proxy_endpoint,
            ),
        }
        return self.zero_reads_diagnosis

    def evidence(self) -> dict[str, Any]:
        return {
            "planned_reads": self.planned_reads,
            "armed": self.armed,
            "start_reads": self.start_reads,
            "reads": self.reads,
            "served": self.served,
            "required_reads": self.required_reads,
            "consumed_at": _iso(self.consumed_at),
            "aborted": self.aborted,
            "endpoint_lost_at": _iso(self.endpoint_lost_at),
            "disarmed_at": _iso(self.disarmed_at),
            "status_reads": self.status_reads,
            "status_errors": self.status_errors[-READ_ERROR_LIMIT:],
            "passthrough_violations": self.passthrough_violations[-READ_ERROR_LIMIT:],
            "failed_reads": self.failed_reads,
            "last_error": self.last_error,
            "first_read_due_seconds": self.first_read_due_seconds,
            "zero_reads_diagnosis": self.zero_reads_diagnosis,
        }


def _iso(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


@dataclass(frozen=True)
class ObservationWindow:
    """When the runner stops waiting, and why."""

    started_at: datetime
    terms: dict[str, float]
    cap_at: datetime
    caps: dict[str, str]

    @property
    def derived_seconds(self) -> float:
        return sum(self.terms.values())

    @property
    def deadline(self) -> datetime:
        return min(
            self.started_at + timedelta(seconds=self.derived_seconds), self.cap_at
        )

    @property
    def capped(self) -> bool:
        return self.cap_at < self.started_at + timedelta(seconds=self.derived_seconds)

    def evidence(self) -> dict[str, Any]:
        return {
            "started_at": self.started_at.isoformat(),
            "derived_seconds": self.derived_seconds,
            "terms": dict(self.terms),
            "deadline": self.deadline.isoformat(),
            "cap_at": self.cap_at.isoformat(),
            "caps": dict(self.caps),
            "capped": self.capped,
        }


def observation_window(
    *,
    started_at: datetime,
    terms: Mapping[str, float],
    replay_end: datetime | None,
    deadman_at: datetime,
    maintenance_end: datetime,
) -> ObservationWindow:
    """The deadline for one case: ``started_at + sum(terms)``, bounded.

    ``replay_end`` is ``arm + max_seconds`` (after it the tool passes real
    readings through again), ``deadman_at`` is ``repoint + restore_seconds``;
    the runner stops ``CLEANUP_MARGIN_SECONDS`` before the earliest of them and
    of the maintenance window so its own cleanup restores the collector.
    """

    margin = timedelta(seconds=CLEANUP_MARGIN_SECONDS)
    ends = {
        "collector_deadman": deadman_at,
        "maintenance_window": maintenance_end,
    }
    if replay_end is not None:
        ends["proxy_max_seconds"] = replay_end
    name, end = min(ends.items(), key=lambda item: item[1])
    cap = max(started_at, end - margin)
    caps = {key: value.isoformat() for key, value in ends.items()}
    caps["binding"] = name
    return ObservationWindow(
        started_at=started_at,
        terms={key: float(value) for key, value in terms.items()},
        cap_at=cap,
        caps=caps,
    )


@dataclass
class Snapshot:
    """The product's records at one poll (``None`` when a read failed).

    ``extra`` is the judge-specific fifth read a case may bind (the 009 pause
    state on the node, the 010 control node and task); ``None`` for the others.
    """

    at: datetime
    gpu_state: dict[str, Any] | None
    activity: dict[str, Any] | None
    node: dict[str, Any] | None
    extra: dict[str, Any] | None = None


@dataclass
class Judgement:
    outcome: str
    stages: dict[str, list[str]]
    inconclusive: list[str]
    evidence: dict[str, Any]


class Judge(Protocol):
    reads_node: bool

    def observe(self, snapshot: Snapshot, progress: ReplayProgress) -> None: ...

    def decide(
        self, progress: ReplayProgress, now: datetime, *, final: bool
    ) -> Judgement | None: ...

    def wait_until(
        self, progress: ReplayProgress, now: datetime
    ) -> datetime | None: ...


@dataclass
class WatchReads:
    """The reads one poll makes; the runner binds them to the live fixture.

    ``extra`` is read only when the judge sets ``reads_extra`` (the 009 pause
    state, the 010 control node and task); it fails like the other reads.
    ``repoint`` is the probe's ``snapshot`` of the collector unit, read once
    when the proxy counted no read by the time the first one was due.
    """

    status: Callable[[], dict[str, Any]]
    gpu_state: Callable[[], dict[str, Any]]
    activity: Callable[[], dict[str, Any]]
    node: Callable[[], dict[str, Any]]
    extra: Callable[[], dict[str, Any]] | None = None
    repoint: Callable[[], dict[str, Any]] | None = None


@dataclass
class WatchResult:
    judgement: Judgement
    polls: int
    ended_at: datetime
    timeline: list[dict[str, Any]]
    read_errors: list[str]
    final_snapshot: Snapshot | None


def _read(
    reader: Callable[[], dict[str, Any]],
    name: str,
    errors: list[str],
    at: datetime,
    *,
    clock: Callable[[], datetime],
    stop_at: datetime,
) -> dict[str, Any] | None:
    if clock() >= stop_at:
        errors.append(f"{at.isoformat()} {name}: skipped, past the observation cap")
        del errors[:-READ_ERROR_LIMIT]
        return None
    try:
        value = reader()
    except Exception as exc:  # noqa: BLE001 - a failed read is evidence, not a crash
        errors.append(f"{at.isoformat()} {name}: {type(exc).__name__}: {exc}")
        del errors[:-READ_ERROR_LIMIT]
        return None
    return value if isinstance(value, dict) else None


def watch(
    judge: Judge,
    progress: ReplayProgress,
    reads: WatchReads,
    window: ObservationWindow,
    *,
    poll_seconds: float,
    clock: Callable[[], datetime],
    sleep: Callable[[float], None],
) -> WatchResult:
    """Poll until ``judge`` decides or the deadline; the last poll is final.

    The proxy is read until the plan is consumed (afterwards it only passes
    real readings through), the store and the node on every poll. A judge that
    still needs a later moment -- a required workflow running inside its own
    execution deadline, the end of a hold -- keeps the loop going up to the
    window's cap, never past it.
    """

    polls = 0
    errors: list[str] = []
    timeline: list[dict[str, Any]] = []
    snapshot: Snapshot | None = None
    stop_at = window.cap_at + timedelta(seconds=READ_GRACE_SECONDS)

    def read(reader: Callable[[], dict[str, Any]], name: str) -> dict[str, Any] | None:
        return _read(reader, name, errors, now, clock=clock, stop_at=stop_at)

    while True:
        now = clock()
        if not progress.consumed:
            receipt = read(reads.status, "status")
            if receipt is not None:
                progress.update(receipt, now)
            if reads.repoint is not None and progress.needs_repoint_check(now):
                # The first armed read is overdue and none arrived: read the
                # unit back and name the reason instead of waiting it out.
                progress.diagnose(read(reads.repoint, "repoint"), now)
        snapshot = Snapshot(
            at=now,
            gpu_state=read(reads.gpu_state, "gpu_state"),
            activity=read(reads.activity, "activity"),
            node=read(reads.node, "node") if judge.reads_node else None,
            extra=(
                read(reads.extra, "extra")
                if reads.extra is not None and getattr(judge, "reads_extra", False)
                else None
            ),
        )
        judge.observe(snapshot, progress)
        polls += 1
        timeline.append(
            {
                "at": now.isoformat(),
                "served": progress.served,
                "consumed": progress.consumed,
                "reads_failed": [
                    name
                    for name, value in (
                        ("gpu_state", snapshot.gpu_state),
                        ("activity", snapshot.activity),
                    )
                    if value is None
                ],
            }
        )
        del timeline[:-TIMELINE_LIMIT]
        decided = judge.decide(progress, now, final=False)
        if decided is not None:
            return WatchResult(decided, polls, now, timeline, errors, snapshot)
        deadline = window.deadline
        needed = judge.wait_until(progress, now)
        if needed is not None and needed > deadline:
            deadline = min(needed, window.cap_at)
        if now >= deadline:
            final = judge.decide(progress, now, final=True)
            assert final is not None, "a final decision is always a judgement"
            return WatchResult(final, polls, now, timeline, errors, snapshot)
        sleep(max(0.0, min(poll_seconds, (deadline - now).total_seconds())))


__all__ = [
    "CADENCE_JITTER_RATIO",
    "CLEANUP_MARGIN_SECONDS",
    "CollectorCadence",
    "Judge",
    "Judgement",
    "ObservationWindow",
    "PASSTHROUGH_READS",
    "PROCESSING_MARGIN_SECONDS",
    "ReplayProgress",
    "STARTUP_JITTER_SECONDS",
    "STARTUP_SPREAD_SECONDS",
    "Snapshot",
    "WatchReads",
    "WatchResult",
    "ZERO_READS_COLLECTOR_SILENT",
    "ZERO_READS_DROPIN_GONE",
    "ZERO_READS_ENDPOINT_REVERTED",
    "ZERO_READS_PROXY_REFUSED",
    "ZERO_READS_UNIT_INACTIVE",
    "ZERO_READS_UNIT_RESTARTED",
    "ZERO_READS_UNVERIFIED",
    "diagnose_zero_reads",
    "observation_window",
    "plan_reads",
    "repoint_baseline",
    "watch",
]

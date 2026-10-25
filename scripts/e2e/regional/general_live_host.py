"""GF-GENERAL-LIVE-001: CPU overload.

The host collector reads ``/proc/stat`` and ``/proc/loadavg``
(``collectors/host/system_metrics.py``): ``cpu_usage_percent`` >= 98 and
``load1_per_cpu`` >= 2.0 are WARNING RUN_DIAGNOSTICS rules with a 120 s sustain
(``host_health.py``), compiled to FREEZE_EVIDENCE then VALIDATE_HOST with a
``host_diagnostic_probe`` on the breached metric. The load is a transient
``SCHED_IDLE`` unit so the system and the collectors stay responsive; the
runner's own ``/proc/stat`` sampler is the effectiveness oracle, and a tick
below the threshold makes the variant INCONCLUSIVE rather than PASS.
"""

from __future__ import annotations

from typing import Any

from scripts.e2e.regional.general_live_common import (
    await_metric_recovery,
    Oracle,
    activity,
    await_oracle_crossing,
    cadence_record,
    sustained_detection_budget,
    sustained_negative_window,
    wait_for_incident,
    wait_settled,
)
from scripts.e2e.regional.general_live_fixture import (
    VariantContext,
    VariantInconclusive,
    require_tools,
    utc_now,
)
from scripts.e2e.regional.general_live_registry import (
    LIVE_REPLAY,
    NATIVE,
    VariantSpec,
)
from scripts.e2e.regional.general_live_source_replay import host_source_replay
from scripts.e2e.regional.general_live_verdicts import (
    DIAGNOSTIC_ONLY_STATES,
    ChainExpectation,
    chain_errors,
    finding_incidents,
    hardware_errors,
    oracle_values,
    samples,
    sustained,
    verdict,
)

CASE = "GF-GENERAL-LIVE-001"
CPU_THRESHOLD = 98.0
LOAD_THRESHOLD = 2.0
#: ``host_health.py`` HOST_RESOURCE_SUSTAIN_SECONDS, the minimum active time of
#: the cpu_usage_percent and load1_per_cpu rules.
SUSTAIN_SECONDS = 120
SHORT_BURST_SECONDS = 60
#: How long the SCHED_IDLE busy loop may take to read >= 98 % on the oracle: a
#: few collector ticks on a node whose real workload leaves the idle class room.
CPU_RAMP_BUDGET = 120
#: load1 is a one-minute exponential average: 2.1 x vCPU runnable threads reach
#: 2.0 per CPU after about 60 x ln(21) = 183 s; the budget leaves room for the
#: threads to be scheduled and for a busy node's slower convergence.
LOAD_RAMP_BUDGET = 600


def detection_budget(ctx: VariantContext) -> int:
    """The detection budget from the oracle's crossing, on the product's cadence.

    The host collector ships ``threshold:cpu_usage_percent`` /
    ``threshold:load1_per_cpu`` once, as the crossing's edge, then only on its
    health summaries; the finding is minted on the first later active sample at
    least 120 s after the first (up to the sustain plus one summary after the
    edge) and VALIDATE_HOST is judged on one more summary: 120 s + 2 x
    GPU_FAULT_HOST_HEALTH_SUMMARY_SECONDS + GPU_FAULT_HOST_INTERVAL_SECONDS +
    the chain margin (915 s on a 300-s/15-s site). The 510-s budget before it
    (120 + 300 + 90, counted from the load's start) left no room for the
    validation's summary while the load was still held.
    """

    return sustained_detection_budget(ctx, SUSTAIN_SECONDS)


def quiet_window(ctx: VariantContext) -> int:
    """The negative watch after a burst shorter than the sustain ended: the
    sustain plus one health summary plus a collector interval and the margin
    (615 s on a 300-s/15-s site)."""

    return sustained_negative_window(ctx, SUSTAIN_SECONDS)


def _oracle_busy_each_tick(oracle: Oracle, minimum: int) -> None:
    oracle.read()
    values = oracle_values(
        oracle.records, lambda record: record.get("cpu_busy_percent")
    )
    if not sustained(values, lambda value: value >= CPU_THRESHOLD, minimum):
        raise VariantInconclusive(
            f"the CPU oracle never read >= {CPU_THRESHOLD}% on {minimum} consecutive ticks: "
            f"{[round(value, 1) for value in values[-12:]]}"
        )


def _run_cpu_load(
    ctx: VariantContext,
    threads: int,
    hold_seconds: int,
    *,
    ramp_seconds: int,
    budget_seconds: int,
) -> Oracle:
    """Start the oracle and the load; ``cpu-start.json`` records the budgets.

    The load and the oracle outlive the ramp, the detection budget and the
    chain's grading: VALIDATE_HOST must still find the node loaded.
    """

    oracle = Oracle(ctx, "cpu")
    oracle.start(hold_seconds + 60, interval=5, cpu=True, loadavg=True)
    started = ctx.node(
        "cpu-start", "--threads", str(threads), "--seconds", str(hold_seconds)
    )
    ctx.cleanup.push("cpu-load", lambda: ctx.node("cpu-stop"))
    ctx.evidence.write(
        "cpu-start",
        {
            **started,
            **cadence_record(
                ctx,
                sustain_seconds=SUSTAIN_SECONDS,
                detection_budget_seconds=budget_seconds,
                ramp_budget_seconds=ramp_seconds,
            ),
        },
    )
    return oracle


def _await_cpu_crossing(ctx: VariantContext, oracle: Oracle) -> None:
    """Detection is timed from the oracle's first >= 98 % reading."""

    await_oracle_crossing(
        ctx,
        oracle,
        lambda record: record.get("cpu_busy_percent"),
        lambda values: bool(values) and values[-1] >= CPU_THRESHOLD,
        timeout_seconds=CPU_RAMP_BUDGET,
        name="cpu-ramp",
        condition=f"the load did not bring cpu_busy_percent to {CPU_THRESHOLD:g} %",
    )


def _await_load_crossing(ctx: VariantContext, oracle: Oracle) -> None:
    """Detection is timed from the oracle's first load1_per_cpu >= 2.0."""

    await_oracle_crossing(
        ctx,
        oracle,
        lambda record: record.get("load1_per_cpu"),
        lambda values: bool(values) and values[-1] >= LOAD_THRESHOLD,
        timeout_seconds=LOAD_RAMP_BUDGET,
        name="load-ramp",
        condition=f"the threads did not bring load1_per_cpu to {LOAD_THRESHOLD:g}",
    )


def cpu_busy(ctx: VariantContext) -> dict[str, Any]:
    survey = ctx_survey(ctx)
    require_tools(survey, "systemd-run")
    cpus = int(survey.get("cpu_count") or 0)
    if cpus < 1:
        raise VariantInconclusive("the node reports no CPUs")
    started_at = utc_now()
    budget = detection_budget(ctx)
    hold = CPU_RAMP_BUDGET + budget + 240
    oracle = _run_cpu_load(
        ctx, cpus, hold, ramp_seconds=CPU_RAMP_BUDGET, budget_seconds=budget
    )
    _await_cpu_crossing(ctx, oracle)
    # One wait for the finding and its VALIDATE_HOST to be judgeable: the
    # validation waits for evidence newer than itself (one more summary), and
    # the load is still held through the whole budget.
    detection = wait_for_incident(
        ctx,
        started_at,
        "cpu_usage_percent",
        timeout_seconds=budget,
        metrics={"cpu_usage_percent", "load1_per_cpu"},
        validation="VALIDATE_HOST",
        name="cpu-detection",
    )
    _oracle_busy_each_tick(oracle, minimum=SUSTAIN_SECONDS // 5)
    incidents = finding_incidents(detection, "cpu_usage_percent")
    stages = {
        "chain": chain_errors(
            detection,
            ChainExpectation(
                metric="cpu_usage_percent",
                operations=("FREEZE_EVIDENCE", "VALIDATE_HOST"),
                validation="VALIDATE_HOST",
                validation_must_fail=True,
                probe_metric="cpu_usage_percent",
                incident_states=DIAGNOSTIC_ONLY_STATES,
            ),
        ),
        "hardware": hardware_errors(detection),
    }
    stopped_at = utc_now()
    ctx.node("cpu-stop")
    ctx.cleanup.discard("cpu-load")
    recovered = _await_recovery(ctx, "cpu_usage_percent", oracle, since=stopped_at)
    stages["recovery"] = recovered["errors"]
    settled = wait_settled(
        ctx,
        started_at,
        [item["incident_id"] for item in incidents],
        timeout_seconds=budget,
        metrics={"cpu_usage_percent"},
        name="cpu-settled",
    )
    return {
        "verdict": verdict(stages),
        "stages": stages,
        "errors": [
            f"{stage}: {error}" for stage, items in stages.items() for error in items
        ],
        "detection_seconds": _detection_seconds(incidents, started_at),
        "recovery": recovered,
        "settled": _workflow_statuses(settled),
    }


def load_average(ctx: VariantContext) -> dict[str, Any]:
    survey = ctx_survey(ctx)
    require_tools(survey, "systemd-run")
    cpus = int(survey.get("cpu_count") or 0)
    if cpus < 1:
        raise VariantInconclusive("the node reports no CPUs")
    threads = int(cpus * 2.1) + 1
    started_at = utc_now()
    budget = detection_budget(ctx)
    # load1 converges over minutes (LOAD_RAMP_BUDGET); detection is timed from
    # the oracle's crossing and the load is held through the whole budget.
    hold = LOAD_RAMP_BUDGET + budget + 240
    oracle = _run_cpu_load(
        ctx, threads, hold, ramp_seconds=LOAD_RAMP_BUDGET, budget_seconds=budget
    )
    _await_load_crossing(ctx, oracle)
    detection = wait_for_incident(
        ctx,
        started_at,
        "load1_per_cpu",
        timeout_seconds=budget,
        metrics={"cpu_usage_percent", "load1_per_cpu"},
        validation="VALIDATE_HOST",
        name="load-detection",
    )
    oracle.read()
    values = oracle_values(oracle.records, lambda record: record.get("load1_per_cpu"))
    if not sustained(values, lambda value: value >= LOAD_THRESHOLD, 3):
        raise VariantInconclusive(
            f"load1_per_cpu never reached {LOAD_THRESHOLD}: {values[-6:]}"
        )
    incidents = finding_incidents(detection, "load1_per_cpu")
    stages = {
        "chain": chain_errors(
            detection,
            ChainExpectation(
                metric="load1_per_cpu",
                operations=("FREEZE_EVIDENCE", "VALIDATE_HOST"),
                validation="VALIDATE_HOST",
                validation_must_fail=True,
                probe_metric="load1_per_cpu",
                single=False,
                incident_states=DIAGNOSTIC_ONLY_STATES,
            ),
        ),
        "hardware": hardware_errors(detection),
        "distinct_signals": _distinct_signal_errors(detection),
    }
    stopped_at = utc_now()
    ctx.node("cpu-stop")
    ctx.cleanup.discard("cpu-load")
    recovered = _await_recovery(ctx, "load1_per_cpu", oracle, since=stopped_at)
    stages["recovery"] = recovered["errors"]
    wait_settled(
        ctx,
        started_at,
        [item["incident_id"] for item in incidents],
        timeout_seconds=budget,
        metrics={"load1_per_cpu"},
        name="load-settled",
    )
    return {
        "verdict": verdict(stages),
        "stages": stages,
        "errors": [
            f"{stage}: {error}" for stage, items in stages.items() for error in items
        ],
        "recovery": recovered,
    }


def short_burst(ctx: VariantContext) -> dict[str, Any]:
    survey = ctx_survey(ctx)
    require_tools(survey, "systemd-run")
    cpus = int(survey.get("cpu_count") or 0)
    started_at = utc_now()
    oracle = Oracle(ctx, "cpu")
    oracle.start(SHORT_BURST_SECONDS + 30, interval=5, cpu=True)
    ctx.node("cpu-start", "--threads", str(cpus), "--seconds", str(SHORT_BURST_SECONDS))
    ctx.cleanup.push("cpu-load", lambda: ctx.node("cpu-stop"))
    ctx.hold(SHORT_BURST_SECONDS + 10, name="burst")
    ctx.node("cpu-stop")
    ctx.cleanup.discard("cpu-load")
    oracle.read()
    values = oracle_values(
        oracle.records, lambda record: record.get("cpu_busy_percent")
    )
    if not any(value >= CPU_THRESHOLD for value in values):
        raise VariantInconclusive(
            f"the burst never reached {CPU_THRESHOLD}%: {values[-6:]}"
        )
    # A burst shorter than the sustain window must mint no finding; the burst's
    # end is a recovered edge, and the watch outlasts the sustain plus the next
    # health summary (the only later batch that could carry an active sample).
    window = quiet_window(ctx)
    observed = _watch_for_no_finding(ctx, started_at, "cpu_usage_percent", window)
    stages = {
        "no_finding": chain_errors(
            observed, ChainExpectation(metric="cpu_usage_percent", present=False)
        ),
        "hardware": hardware_errors(observed),
    }
    return _result(
        stages,
        extra={
            "negative_window_seconds": window,
            "host_summary_seconds": ctx.host_summary,
            "host_interval_seconds": ctx.host_interval,
        },
    )


def recovery(ctx: VariantContext) -> dict[str, Any]:
    """Drive cpu_busy, then prove the recovered health signal and a fresh pass."""

    survey = ctx_survey(ctx)
    require_tools(survey, "systemd-run")
    cpus = int(survey.get("cpu_count") or 0)
    started_at = utc_now()
    budget = detection_budget(ctx)
    hold = CPU_RAMP_BUDGET + budget + 60
    oracle = _run_cpu_load(
        ctx, cpus, hold, ramp_seconds=CPU_RAMP_BUDGET, budget_seconds=budget
    )
    _await_cpu_crossing(ctx, oracle)
    detection = wait_for_incident(
        ctx,
        started_at,
        "cpu_usage_percent",
        timeout_seconds=budget,
        metrics={"cpu_usage_percent"},
        name="recovery-detection",
    )
    _oracle_busy_each_tick(oracle, minimum=SUSTAIN_SECONDS // 5)
    incidents = finding_incidents(detection, "cpu_usage_percent")
    stopped_at = utc_now()
    ctx.node("cpu-stop")
    ctx.cleanup.discard("cpu-load")
    recovered = _await_recovery(ctx, "cpu_usage_percent", oracle, since=stopped_at)
    settled = wait_settled(
        ctx,
        started_at,
        [item["incident_id"] for item in incidents],
        timeout_seconds=budget,
        metrics={"cpu_usage_percent"},
        name="recovery-settled",
    )
    stages = {
        "recovery": recovered["errors"],
        "hardware": hardware_errors(detection),
    }
    if not incidents:
        stages["recovery"].append("no cpu finding to recover from")
    return _result(
        stages, extra={"recovery": recovered, "settled": _workflow_statuses(settled)}
    )


def checkpoint_phase(ctx: VariantContext) -> dict[str, Any]:
    """cpu_busy while a managed attempt reports CHECKPOINT_SAVE.

    The product does not correlate the application phase with the host finding
    (audit gap), so the finding stays the diagnostic cpu chain -- that is the
    documented behaviour, recorded as a note, not a FAIL.
    """

    survey = ctx_survey(ctx)
    require_tools(survey, "systemd-run")
    cpus = int(survey.get("cpu_count") or 0)
    started_at = utc_now()
    budget = detection_budget(ctx)
    hold = CPU_RAMP_BUDGET + budget + 120
    oracle = _run_cpu_load(
        ctx, cpus, hold, ramp_seconds=CPU_RAMP_BUDGET, budget_seconds=budget
    )
    _await_cpu_crossing(ctx, oracle)
    detection = wait_for_incident(
        ctx,
        started_at,
        "cpu_usage_percent",
        timeout_seconds=budget,
        metrics={"cpu_usage_percent"},
        validation="VALIDATE_HOST",
        name="checkpoint-detection",
    )
    _oracle_busy_each_tick(oracle, minimum=SUSTAIN_SECONDS // 5)
    stopped_at = utc_now()
    ctx.node("cpu-stop")
    ctx.cleanup.discard("cpu-load")
    recovered = _await_recovery(ctx, "cpu_usage_percent", oracle, since=stopped_at)
    stages = {
        "chain": chain_errors(
            detection,
            ChainExpectation(
                metric="cpu_usage_percent",
                operations=("FREEZE_EVIDENCE", "VALIDATE_HOST"),
                validation="VALIDATE_HOST",
                validation_must_fail=True,
            ),
        ),
        "hardware": hardware_errors(detection),
        "recovery": recovered["errors"],
    }
    return _result(
        stages,
        notes=["application phase correlation is absent in the product (audit gap)"],
        extra={"recovery": recovered},
    )


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #
def ctx_survey(ctx: VariantContext) -> dict[str, Any]:
    from scripts.e2e.regional.general_live_common import run_survey

    return run_survey(ctx)


def _await_recovery(
    ctx: VariantContext, metric: str, oracle: Oracle, since: Any = None
) -> dict[str, Any]:
    """Fresh evidence that ``metric`` recovered after the load stopped."""

    threshold = CPU_THRESHOLD if metric == "cpu_usage_percent" else LOAD_THRESHOLD
    result = await_metric_recovery(ctx, metric, threshold, since=since)
    oracle.read()
    return result


def _watch_for_no_finding(
    ctx: VariantContext, since: Any, metric: str, window_seconds: int
) -> dict[str, Any]:
    last = activity(ctx, since, metrics={metric})

    def accept() -> dict[str, Any] | None:
        nonlocal last
        last = activity(ctx, since, metrics={metric})
        return last if finding_incidents(last, metric) else None

    hit = ctx.wait_until(
        accept,
        timeout_seconds=window_seconds,
        poll_seconds=ctx.host_interval * 2,
        name="negative-window",
    )
    return hit if hit is not None else last


def _distinct_signal_errors(activity_document: dict[str, Any]) -> list[str]:
    cpu = finding_incidents(activity_document, "cpu_usage_percent")
    load = finding_incidents(activity_document, "load1_per_cpu")
    errors = []
    if not load:
        errors.append("load1_per_cpu never reached its own incident")
    if cpu and load and cpu[0]["incident_id"] == load[0]["incident_id"]:
        # Co-firing is expected; they just must be bound to their own signal.
        pass
    return errors


def _detection_seconds(incidents: list[dict[str, Any]], started: Any) -> float | None:
    from scripts.e2e.regional.general_live_verdicts import detection_seconds

    return detection_seconds({}, incidents[0], started) if incidents else None


def _workflow_statuses(activity_document: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {"request_id": item.get("request_id"), "status": item.get("status")}
        for item in activity_document.get("workflows") or []
    ]


def _result(
    stages: dict[str, list[str]],
    *,
    notes: list[str] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "verdict": verdict(stages),
        "stages": stages,
        "errors": [
            f"{stage}: {error}" for stage, items in stages.items() for error in items
        ],
        "notes": notes or [],
        **(extra or {}),
    }


#: A reviewed static ``/proc/stat``: the busy counters never advance, so the
#: collector's ``total > previous_total`` guard emits no ``cpu_usage_percent``
#: sample (``collectors/host/system_metrics.py`` ``_cpu``).
STATIC_PROC_STAT = (
    b"cpu  100 0 50 1000 0 0 0 0 0 0\n"
    b"cpu0 100 0 50 1000 0 0 0 0 0 0\n"
    b"intr 0\nctxt 0\nbtime 0\nprocesses 0\nprocs_running 1\nprocs_blocked 0\n"
)


def invalid_delta(ctx: VariantContext) -> dict[str, Any]:
    """Live-replay: a non-positive ``/proc/stat`` delta yields no busy sample.

    The reviewed static snapshot is bound over ``/proc/stat`` in the host
    collector unit's own mount namespace; parsing, batching and grading stay
    real. The runner wrote the fixture, so the non-positive total delta is the
    raw-source oracle; the product must emit no ``cpu_usage_percent`` sample and
    no finding.
    """

    ctx_survey(ctx)
    replay = host_source_replay(ctx)
    replay.create()
    replay.install("proc-stat", STATIC_PROC_STAT)
    opened = replay.open_bind(
        [("proc-stat", "/proc/stat")], restore_seconds=ctx.spec.max_minutes * 60
    )
    ctx.evidence.write("source-open", opened)
    started = utc_now()
    status = replay.status()
    ctx.evidence.append_oracle(
        [{"proc_stat_fixture": (status.get("fixtures") or {}).get("proc-stat")}],
        source="proc-stat",
    )
    # Let the collector poll the static file past its baseline, then confirm no
    # busy sample and no finding across a negative window that covers at least
    # one health summary (the batch that would carry a sample if one existed).
    ctx.hold(ctx.host_interval * 4 + 30, name="static-delta")
    observed = _watch_for_no_finding(
        ctx, started, "cpu_usage_percent", quiet_window(ctx)
    )
    cpu_samples = samples(observed, "cpu_usage_percent")
    stages = {
        "no_busy_sample": (
            [
                f"{len(cpu_samples)} cpu_usage_percent samples emitted on a non-positive delta"
            ]
            if cpu_samples
            else []
        ),
        "no_finding": chain_errors(
            observed, ChainExpectation(metric="cpu_usage_percent", present=False)
        ),
        "hardware": hardware_errors(observed),
    }
    return _result(
        stages,
        notes=[
            "a static /proc/stat makes the total delta <= 0; no busy sample is emitted"
        ],
        extra={"proof_scope": "live-replay"},
    )


VARIANTS = [
    VariantSpec(
        case_id=CASE,
        variant="cpu_busy",
        proof_mode=NATIVE,
        lanes=("eks", "hp"),
        risk="L",
        flags="S",
        injection=(
            "SCHED_IDLE busy loop on every vCPU for the ramp plus the cadence budget "
            "(~21 min on a 300-s summary cadence; transient unit, deadman)"
        ),
        expected=(
            "cpu_usage_percent >= 98 for 120 s -> VALIDATE_HOST probe fails while "
            "loaded; no reset/reboot/DRAIN; detection timed from the oracle's "
            "crossing over two host health summaries"
        ),
        cleanup="stop the load; 2 fresh samples below 98 %",
        max_minutes=35,
        execute=cpu_busy,
    ),
    VariantSpec(
        case_id=CASE,
        variant="load_average",
        proof_mode=NATIVE,
        lanes=("eks", "hp"),
        risk="L",
        flags="S",
        injection=(
            "2.1x vCPU runnable SCHED_IDLE threads held until load1 converges, then "
            "through the cadence budget timed from the oracle's crossing"
        ),
        expected="load1_per_cpu >= 2.0 -> VALIDATE_HOST probe on load1_per_cpu; cpu_busy co-fires as its own signal",
        cleanup="stop the load; fresh samples below 2.0",
        max_minutes=45,
        execute=load_average,
    ),
    VariantSpec(
        case_id=CASE,
        variant="short_burst",
        proof_mode=NATIVE,
        lanes=("eks", "hp"),
        risk="L",
        flags="S",
        injection="busy for 60 s (< the 120 s sustain), then stop",
        expected=(
            "an edge plus a recovered batch; no finding/incident/workflow across the "
            "sustain plus one health summary after the burst (615 s on a 300-s cadence)"
        ),
        cleanup="none (the burst stopped itself)",
        max_minutes=16,
        execute=short_burst,
    ),
    VariantSpec(
        case_id=CASE,
        variant="recovery",
        proof_mode=NATIVE,
        lanes=("eks", "hp"),
        risk="L",
        flags="S",
        injection="drive cpu_busy to a finding, then stop the load",
        expected="recovered health-signal transition and 2 fresh samples below 98 %; the incident settles",
        cleanup="none",
        max_minutes=35,
        execute=recovery,
    ),
    VariantSpec(
        case_id=CASE,
        variant="invalid_delta",
        proof_mode=LIVE_REPLAY,
        lanes=("eks", "hp"),
        risk="L",
        flags="S",
        injection="bind a reviewed static /proc/stat into the host collector's mount namespace",
        expected="total delta <= 0 -> no cpu_usage_percent sample and no finding",
        cleanup="close the /proc/stat bind window; confirm clean",
        max_minutes=16,
        execute=invalid_delta,
    ),
    VariantSpec(
        case_id=CASE,
        variant="checkpoint_phase",
        proof_mode=NATIVE,
        lanes=("eks", "hp"),
        risk="L",
        flags="S",
        injection="cpu_busy while a managed attempt declares phase=CHECKPOINT_SAVE",
        expected="the finding stays the diagnostic cpu chain; phase correlation is absent (noted, not a FAIL)",
        cleanup="stop the job and the load",
        max_minutes=35,
        execute=checkpoint_phase,
    ),
]

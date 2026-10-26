"""GF-GENERAL-LIVE-025 DCGM collection incomplete and 031 collector silent.

Both pause a *producer* and read the product's own observation records. 025
cuts the product-owned DCGM exporter's scrape (``exporter-pause``): the metrics
collector's failed scrape becomes a sustained ``dcgm_field_completeness``
collection error that, past ``collection_error_sustain_seconds`` (180 s),
requests one bounded TRIGGER_HEALTH_SNAPSHOT + read-only validation -- never a
reset. That validation (VALIDATE_HOST under the OBSERVABILITY recovery
contract, ``health_evidence_validation.validate_observability``) passes only
on a GPU_METRICS success newer than the finding and no older than 120 s, so
while the exporter is paused it is WAITING by design (``health_evidence_pending``
"collector state is UNKNOWN", ``missing_observations``), latches
``step_waiting_slow`` at 300 s and FAILS "diagnostic inconclusive" at the 600 s
per-step cap (``execution/step_bounds.bounded_waiting_outcome``). Live
2026-10-26 ``exporter_timeout`` demanded a finished validation while the pause
(900 s) still outlasted its own 720 s wait and FAILED on the harness's order,
not the product's. It is judged in two reads now: while paused, the incident,
the planned bounded chain and a validation that has *not* passed on evidence
that cannot exist; after ``exporter-resume``, the validation's completion --
SUCCEEDED once evidence returned, how long it waited, and never through the
product's cap. 031 pauses one collector unit while the Agent lease stays valid
(``unit-pause``): the silence scanner records ``collector_observation_valid``
UNKNOWN, mints at most one snapshot-refresh workflow per node/hour, and sends
one silent advisory. A disabled/masked unit is excluded, and resuming the
producer must advance a new success time -- not merely leave the process
running. The non-native DCGM source-replay variants and the inactive-cluster /
notification-failure capability gaps are recorded NOT_RUN with their reason.

The metrics collector unit carries two channels: GPU_INVENTORY (180 s) trips
first and its refresh -- the Node Agent's TRIGGER_HEALTH_SNAPSHOT, a
``systemctl restart`` of the metrics and host collectors -- would end a bare
SIGSTOP (live 2026-10-25: the restarted process kept GPU_METRICS fresh 183 s
into ``metrics_silent``). The node probe's pause therefore survives that
restart, the advisory check names the silent channel (the inventory advisory
is not the metrics one), the hour's deduplicated advisory counts for a retry
inside the alert interval, and recovery needs a success *after* the resume.

A new success is not one poll away for every channel. GPU_METRICS posts every
scrape and GPU_INVENTORY every minute; the host and node-log collectors post
only on an edge or on their 300 s health summary, and the node log collector's
summary slot is a success only when that poll read the journal cleanly -- it
carries the collection errors otherwise, and an error-only batch does not move
``last_success_at`` (``collectors/logs/node.py`` ``collect_once``). Live
2026-10-26 (``logs_silent``, plain EKS, release R13g): the enabled collector
never succeeded in the 900 s the variant held blind before the pause, so the
UNKNOWN it then read came from a success five hours old, not from the pause;
after the resume a 600 s wait covered one summary slot and a half and none was
a success -- the R13g collector passes ``--since`` with ``--after-cursor``,
systemd refuses every poll that has a cursor, and a slot that lands on a
refused poll is an error batch (batch b190, ``journalctl-failures=72`` in 1457
s). The log variant now waits for the first success after the enable instead
of holding, the pause's UNKNOWN must be the silence's own reason
(STALE_OBSERVATION, not a refused poll's COLLECTION_ERROR), every recovery
wait is sized from the channel's cadence (``recovery_budget``), and a wait
that runs out names the refusal pattern when the channel's status shows it --
still a FAIL, never a bare one.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

from gpu_fault.collectors.cadence import (
    DEFAULT_JITTER_RATIO,
    DEFAULT_STARTUP_JITTER_SECONDS,
)
from scripts.e2e.regional.general_live_common import activity, run_survey
from scripts.e2e.regional.general_live_fixture import (
    VariantContext,
    VariantInconclusive,
    VariantNotRun,
    require_tools,
    utc_now,
)
from scripts.e2e.regional.general_live_registry import (
    CAPABILITY,
    NATIVE,
    VariantSpec,
)
from scripts.e2e.regional.general_live_verdicts import (
    DIAGNOSTIC_ONLY_STATES,
    TERMINAL_WORKFLOWS,
    ChainExpectation,
    chain_errors,
    chain_judgeable,
    executions,
    final_execution,
    finding_incidents,
    hardware_errors,
    judged_chain,
    operations,
    parse_time,
    pending_summary,
    verdict,
)

CASE_025 = "GF-GENERAL-LIVE-025"
CASE_031 = "GF-GENERAL-LIVE-031"
DCGM_SUSTAIN = 180
DCGM_BUDGET = DCGM_SUSTAIN + 420 + 120
COMPLETENESS = "dcgm_field_completeness"
VALIDATION = "VALIDATE_HOST"
#: ``exporter_timeout`` resumes the exporter itself once the product's chain
#: has reached its validation; the pause's deadman only covers a runner that
#: dies (the same 900 s the variant armed before).
EXPORTER_DEADMAN = DCGM_BUDGET + 180
#: After the resume one exporter scrape and the metrics collector's ingest put
#: a fresh GPU_METRICS success in the store (15-30 s) and the executor's next
#: 5 s poll re-reads VALIDATE_HOST: ten times that, and half the product's
#: 600 s per-step cap, so a cap-failed validation is still read as such.
VALIDATION_BUDGET = 300
#: What the waiting validation records (``health_evidence_validation.pending``)
#: and the executor's wait accounting adds (``execution/step_bounds``): kept in
#: the result as the product's own statement of why and how long it waited.
WAITING_DETAIL_KEYS = (
    "health_evidence_pending",
    "missing_observations",
    "step_waiting_seconds",
    "step_waiting_slow",
    "step_waiting_timeout_seconds",
)
#: Inventory 180 s, metrics/host 420 s, logs 900 s (collector_kinds.py).
SILENCE_THRESHOLDS = {
    "inventory_silent": ("GPU_INVENTORY", "gpu-fault-metrics-collector.service", 180),
    "metrics_silent": ("GPU_METRICS", "gpu-fault-metrics-collector.service", 420),
    "host_silent": ("HOST_TELEMETRY", "gpu-fault-host-collector.service", 420),
    "logs_silent": ("NODE_LOGS", "gpu-fault-log-collector.service", 900),
}
SILENCE_SCAN_MARGIN = 120
#: ``GPU_FAULT_COLLECTOR_SILENT_ALERT_SECONDS`` default (app/periodic_services.py):
#: one ``collector-silent/<cluster>/<node>/<channel>/<bucket>`` advisory per
#: channel per bucket; a retry inside the bucket finds no new notification.
SILENT_ALERT_INTERVAL = 3600
#: The silence's own UNKNOWN (``observation_health.observation_reason``): the
#: success aged past the threshold. A refused or failed poll reads UNKNOWN too
#: (COLLECTION_ERROR), but that is not the pause's doing.
SILENCE_REASONS = frozenset({"STALE_OBSERVATION"})
#: ``GPU_FAULT_COLLECTOR_STARTUP_SPREAD_SECONDS`` default -- the metrics and host
#: collectors' deterministic first-post spread -- and the random part every
#: collector process adds to it (``collectors/cadence.py``). A pause is a
#: SIGSTOP, not a restart, but the product restarts the paused unit from its
#: snapshot refresh and the pause drop-in re-stops the new main process: the
#: resume may start a collector from scratch.
STARTUP_SPREAD = 15
STARTUP_JITTER = int(DEFAULT_STARTUP_JITTER_SECONDS)
#: ``GPU_FAULT_COLLECTOR_SILENCE_SCAN_SECONDS`` default (app/periodic_services.py):
#: the scan that turns a new success into the HEALTHY observation.
SILENCE_SCAN = 60
#: The sink's full retry ladder, 4 x 10 s timeouts + 1 + 2 + 4 s backoff
#: (``collectors/sinks.py``; ``tests/collectors/test_cadence_jitter.py`` pins it).
DELIVERY_LADDER = 47
#: The recovery wait before this batch, kept as the floor: GPU_METRICS posts
#: every scrape (15 s) and GPU_INVENTORY every 60 s, so two of their intervals
#: and the spread are far inside it.
RECOVERY_FLOOR = 600
#: Batch b190 (not in R13g): ``collectors/logs/node.py`` ``_journal_command``
#: passes ``--since`` together with ``--after-cursor``, which systemd 252+
#: refuses with this text. The refusal drops the cursor, the next
#: ``--since``-only poll succeeds and learns one again, so every other poll
#: fails for the life of the process (``journalctl-failures`` counting up in
#: the status) and a summary slot that lands on a refused poll is an error
#: batch, not a success.
JOURNAL_REFUSAL = "Please specify only one of --since=, --cursor=, and --after-cursor="
REFUSAL_GAP = "collector alternates refusals: known product gap b190"
JOURNAL_FAILURES = re.compile(r"journalctl-failures=(\d+)")


# --------------------------------------------------------------------------- #
# 025 DCGM collection incomplete
# --------------------------------------------------------------------------- #
def _pause_exporter(ctx: VariantContext, seconds: int) -> dict[str, Any]:
    survey = run_survey(ctx)
    processes = survey.get("exporter_processes") or []
    if len(processes) != 1:
        raise VariantNotRun(
            "FIXTURE_NOT_READY",
            f"expected exactly one dcgm-exporter process, found {len(processes)}",
        )
    paused = ctx.pause("exporter-pause", "--seconds", str(seconds))
    ctx.cleanup.push("exporter", lambda: ctx.node("exporter-resume"))
    ctx.evidence.write("exporter-pause", paused)
    if not paused.get("stopped"):
        raise VariantInconclusive("the dcgm-exporter process did not stop")
    return paused


def _completion_activity(ctx: VariantContext, since: Any) -> dict[str, Any]:
    return activity(
        ctx, since, metrics={"collector_observation_valid"}, kinds=("GPU_METRICS",)
    )


def _wait_completion_error(
    ctx: VariantContext,
    since: Any,
    *,
    name: str,
    until: Callable[[dict[str, Any]], bool] | None = None,
    timeout: float = DCGM_BUDGET,
) -> dict[str, Any]:
    """The field-completeness incident -- or, with ``until``, the first read of
    the node's activity it accepts (the chain at its validation, the
    validation terminal). The last read when the budget runs out."""

    last: dict[str, Any] = {}

    def accept() -> dict[str, Any] | None:
        nonlocal last
        last = _completion_activity(ctx, since)
        ready = (
            bool(finding_incidents(last, COMPLETENESS))
            if until is None
            else until(last)
        )
        return last if ready else None

    value = ctx.wait_until(
        accept, timeout_seconds=timeout, poll_seconds=ctx.host_interval, name=name
    )
    return value if value is not None else last


def _completion_workflow(document: dict[str, Any]) -> dict[str, Any] | None:
    """The field-completeness incident's first workflow, or None."""

    chain = judged_chain(document, COMPLETENESS)
    return None if chain is None else chain[1]


def _validation_reached(document: dict[str, Any]) -> bool:
    """The chain's VALIDATE_HOST has read its evidence at least once (an
    execution record in any status), or the workflow ended before it ran."""

    workflow = _completion_workflow(document)
    if workflow is None:
        return False
    return str(workflow.get("status")) in TERMINAL_WORKFLOWS or bool(
        executions(workflow, VALIDATION)
    )


def _validation_terminal(document: dict[str, Any]) -> bool:
    return chain_judgeable(document, COMPLETENESS, validation=VALIDATION)


def _validation_execution(document: dict[str, Any]) -> dict[str, Any] | None:
    workflow = _completion_workflow(document)
    return None if workflow is None else final_execution(workflow, VALIDATION)


def _completeness_chain(*, validation: str | None) -> ChainExpectation:
    """The bounded chain: a snapshot and the observation validation, nothing
    hardware or stress; with ``validation``, that step finished SUCCEEDED and
    the incident is in a diagnostic-only state."""

    return ChainExpectation(
        metric=COMPLETENESS,
        operations=("TRIGGER_HEALTH_SNAPSHOT", VALIDATION),
        validation=validation,
        forbidden=_hardware() | {"RUN_DCGM_DIAGNOSTIC"},
        incident_states=DIAGNOSTIC_ONLY_STATES if validation else frozenset(),
    )


def _moment(value: Any) -> datetime | None:
    try:
        moment = parse_time(value)
    except ValueError:
        return None
    if moment is not None and moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment


def _paused_errors(
    document: dict[str, Any], execution: dict[str, Any] | None
) -> list[str]:
    """While the exporter is paused: the incident and its bounded chain, and a
    validation that has not passed on a success that cannot exist yet."""

    errors = chain_errors(document, _completeness_chain(validation=None))
    if execution is not None and execution.get("status") == "SUCCEEDED":
        errors.append(
            f"{VALIDATION} SUCCEEDED while the exporter was still paused: a stale "
            "success passed the observation validation"
        )
    return errors


def _validation_errors(
    execution: dict[str, Any] | None, resumed_at: datetime
) -> list[str]:
    """What the chain stage does not say about a terminal validation: a
    success older than the resume, or a wait that ran into the product's
    per-step cap instead of finding fresh evidence."""

    if execution is None:
        return []
    details = execution.get("details") or {}
    status = execution.get("status")
    if status == "SUCCEEDED":
        finished = _moment(execution.get("updated_at"))
        if finished is not None and finished < resumed_at:
            return [
                f"{VALIDATION} succeeded at {finished.isoformat()}, before the "
                f"exporter was resumed at {resumed_at.isoformat()}: a stale "
                "success passed the observation validation"
            ]
    elif status == "FAILED" and "step_waiting_timeout_seconds" in details:
        return [
            f"{VALIDATION} failed on the product's "
            f"{details['step_waiting_timeout_seconds']} s per-step cap before "
            "fresh GPU_METRICS evidence arrived: "
            + str(execution.get("error") or "")[:200]
        ]
    return []


def _validation_record(
    paused: dict[str, Any] | None,
    final: dict[str, Any] | None,
    resumed_at: datetime,
) -> dict[str, Any]:
    """The product's own account of the wait: what VALIDATE_HOST reported while
    the exporter was paused, and when and how it finished after the resume."""

    record: dict[str, Any] = {
        "resumed_at": resumed_at.isoformat(),
        "status": None if final is None else final.get("status"),
    }
    if paused is not None:
        details = paused.get("details") or {}
        record["while_paused"] = {
            "status": paused.get("status"),
            "pending": pending_summary(paused),
            **{key: details[key] for key in WAITING_DETAIL_KEYS if key in details},
        }
    if final is not None:
        started = _moment(final.get("started_at"))
        finished = _moment(final.get("updated_at"))
        record["started_at"] = final.get("started_at")
        record["finished_at"] = final.get("updated_at")
        if started is not None and finished is not None:
            record["waited_seconds"] = int((finished - started).total_seconds())
        if finished is not None:
            record["finished_after_resume"] = finished >= resumed_at
        details = final.get("details") or {}
        record.update(
            {key: details[key] for key in WAITING_DETAIL_KEYS if key in details}
        )
        if final.get("error"):
            record["error"] = str(final["error"])[:300]
    return record


def exporter_timeout(ctx: VariantContext) -> dict[str, Any]:
    """Pause the exporter until the product's chain has reached its validation,
    judge the bounded chain and the waiting validation on that read, resume the
    exporter, then judge the validation's completion on the evidence it waited
    for. The deadman only backs a runner that dies."""

    started = utc_now()
    _pause_exporter(ctx, EXPORTER_DEADMAN)
    paused = _wait_completion_error(
        ctx, started, name="dcgm-error", until=_validation_reached
    )
    paused_execution = _validation_execution(paused)
    stages = {
        "paused": _paused_errors(paused, paused_execution),
        "unknown": _observation_unknown(paused, "GPU_METRICS"),
    }
    resumed_at = ctx.clock()
    ctx.node("exporter-resume")
    ctx.cleanup.discard("exporter")
    if _completion_workflow(paused) is None:
        # No chain to wait for: one read after the resume says what is there.
        final = _completion_activity(ctx, started)
    else:
        final = _wait_completion_error(
            ctx,
            started,
            name="dcgm-validation",
            until=_validation_terminal,
            timeout=VALIDATION_BUDGET,
        )
    execution = _validation_execution(final)
    stages["chain"] = chain_errors(final, _completeness_chain(validation=VALIDATION))
    stages["validation"] = _validation_errors(execution, resumed_at)
    stages["hardware"] = hardware_errors(final)
    recovered = _await_fresh_success(ctx, "GPU_METRICS")
    stages["recovery"] = recovered["errors"]
    return _result(
        stages,
        notes=[
            "the observation validation waits for a GPU_METRICS success newer than "
            "the finding; it is judged after the exporter resumed, never while paused"
        ],
        extra={
            "validation": _validation_record(paused_execution, execution, resumed_at)
        },
    )


def sustained_unknown(ctx: VariantContext) -> dict[str, Any]:
    started = utc_now()
    hold = DCGM_BUDGET + 120
    _pause_exporter(ctx, hold + 120)
    detection = _wait_completion_error(ctx, started, name="dcgm-unknown")
    # Keep it paused past the sustain window; the observation stays UNKNOWN and
    # the chain is the bounded snapshot + validation, not a hardware step.
    ctx.hold(
        DCGM_SUSTAIN, name="sustain", check=lambda: _observation_state(ctx, started)
    )
    final = activity(
        ctx, started, metrics={"collector_observation_valid"}, kinds=("GPU_METRICS",)
    )
    stages = {
        "unknown": _observation_unknown(final, "GPU_METRICS"),
        "hardware": hardware_errors(final),
        "bounded": _snapshot_only(final),
    }
    ctx.node("exporter-resume")
    ctx.cleanup.discard("exporter")
    recovered = _await_fresh_success(ctx, "GPU_METRICS")
    stages["recovery"] = recovered["errors"]
    _ = detection
    return _result(stages)


def post_refresh_success(ctx: VariantContext) -> dict[str, Any]:
    started = utc_now()
    _pause_exporter(ctx, DCGM_BUDGET + 60)
    detection = _wait_completion_error(ctx, started, name="dcgm-refresh")
    incidents = finding_incidents(detection, "dcgm_field_completeness")
    resumed = ctx.node("exporter-resume")
    ctx.cleanup.discard("exporter")
    ctx.evidence.write("exporter-resume", resumed)
    recovered = _await_fresh_success(ctx, "GPU_METRICS")
    stages = {
        "recovery": recovered["errors"],
        "cleared": _cleared_error(ctx, started),
        "no_stale_pass": []
        if incidents
        else ["no sustained collection error was observed before the refresh"],
    }
    return _result(
        stages,
        notes=[
            "a stale success before the trigger must not pass; only a new complete batch clears it"
        ],
    )


# --------------------------------------------------------------------------- #
# 031 collector silent
# --------------------------------------------------------------------------- #
def _silence_variant(ctx: VariantContext, variant: str) -> dict[str, Any]:
    channel, unit, threshold = SILENCE_THRESHOLDS[variant]
    survey = run_survey(ctx)
    require_tools(survey, "systemctl")
    extra: dict[str, Any] = {"channel": channel, "threshold_seconds": threshold}
    if variant == "logs_silent":
        if "ENABLE_NODE_LOG_COLLECTOR" not in ctx.settings.approvals:
            raise VariantNotRun(
                "NOT_AUTHORIZED",
                "the node log collector is disabled in production; needs ENABLE_NODE_LOG_COLLECTOR",
            )
        # The success the channel had before the enable -- hours old from an
        # earlier run, or none. The pause may only begin once the enabled
        # collector has advanced it: a channel that never succeeded after the
        # enable is silent without any pause, and its UNKNOWN proves nothing
        # about the pause (live 2026-10-26: the 900 s hold ended on a success
        # five hours old and the UNKNOWN was read 20 s after the SIGSTOP).
        baseline = _channel_success(ctx, utc_now(), channel)
        enabled = ctx.pause(
            "log-collector-enable",
            "--seconds",
            str(ctx.spec.max_minutes * 60 + 300),
            "--approved",
        )
        ctx.cleanup.push("log-collector", lambda: ctx.node("log-collector-disable"))
        ctx.evidence.write("log-collector-enable", enabled)
        first = _await_fresh_success(
            ctx,
            channel,
            after=baseline,
            name="log-first-success",
            what="after the enable",
        )
        ctx.evidence.write("log-first-success", first)
        extra.update(success_before_enable=baseline, first_success=first)
        if first["errors"]:
            ctx.node("log-collector-disable")
            ctx.cleanup.discard("log-collector")
            return _result(
                {"first_success": first["errors"]},
                notes=[
                    "the unit was not paused: a channel that never succeeded after "
                    "the enable is silent without the pause, so UNKNOWN, advisory, "
                    "snapshot cap and recovery were not judged"
                ],
                extra=extra,
            )
    started = utc_now()
    hold = threshold + SILENCE_SCAN_MARGIN + 300
    paused = ctx.pause("unit-pause", "--unit", unit, "--seconds", str(hold))
    ctx.cleanup.push("pause", lambda: ctx.node("unit-resume", "--unit", unit))
    ctx.evidence.write("unit-pause", paused)
    if not paused.get("stopped"):
        raise VariantInconclusive(f"{unit} did not stop under SIGSTOP")
    budget = threshold + SILENCE_SCAN_MARGIN + 180
    observed = _wait_observation_unknown(
        ctx, started, channel, budget, reasons=SILENCE_REASONS
    )
    notes: list[str] = []
    advisory = _silent_advisory(observed, channel)
    if advisory:
        deduplicated = _deduplicated_advisory(ctx, started, channel)
        if deduplicated:
            advisory = []
            notes.append(
                f"the {channel} advisory of this alert hour was already sent and "
                f"deduplicated: {', '.join(deduplicated)}"
            )
    stages = {
        "unknown": _observation_unknown(observed, channel, reasons=SILENCE_REASONS),
        "advisory": advisory,
        "snapshot_cap": _one_snapshot_per_hour(observed),
        "hardware": hardware_errors(observed),
    }
    # The success time the channel had while paused: the resume must advance it.
    before = _channel_success(ctx, started, channel)
    ctx.node("unit-resume", "--unit", unit)
    ctx.cleanup.discard("pause")
    recovered = _await_fresh_success(ctx, channel, after=before)
    stages["recovery"] = recovered["errors"]
    if variant == "logs_silent":
        ctx.node("log-collector-disable")
        ctx.cleanup.discard("log-collector")
    extra.update(success_before_resume=before, recovery=recovered)
    return _result(stages, notes=notes, extra=extra)


def inventory_silent(ctx: VariantContext) -> dict[str, Any]:
    return _silence_variant(ctx, "inventory_silent")


def metrics_silent(ctx: VariantContext) -> dict[str, Any]:
    return _silence_variant(ctx, "metrics_silent")


def host_silent(ctx: VariantContext) -> dict[str, Any]:
    return _silence_variant(ctx, "host_silent")


def logs_silent(ctx: VariantContext) -> dict[str, Any]:
    return _silence_variant(ctx, "logs_silent")


def disabled_unit(ctx: VariantContext) -> dict[str, Any]:
    """A disabled/masked collector is excluded from the silence scan: observing
    it should raise no UNKNOWN and no advisory."""

    survey = run_survey(ctx)
    units = survey.get("units") or {}
    disabled = sorted(
        name
        for name, state in units.items()
        if str(state.get("UnitFileState")) in {"disabled", "masked"}
        and name.startswith("gpu-fault-")
        and name.endswith("-collector.service")
    )
    started = utc_now()
    ctx.hold(SILENCE_SCAN_MARGIN + ctx.host_interval * 3, name="disabled-hold")
    observed = activity(ctx, started, metrics={"collector_observation_valid"}, kinds=())
    unexpected = [
        note
        for note in _silent_advisories(observed)
        if any(unit.split(".")[0] in note for unit in disabled)
    ]
    stages = {
        "excluded": unexpected
        and [f"a disabled collector produced a silent advisory: {unexpected}"]
        or [],
        "found_disabled": []
        if disabled
        else ["no gpu-fault collector unit is disabled/masked on this node"],
    }
    return _result(
        stages,
        notes=["disabled/masked collectors are excluded, not reported silent"],
        extra={"disabled": disabled},
    )


def expired_agent(ctx: VariantContext) -> dict[str, Any]:
    """An expired Agent lease is handled apart from channel silence: no silent
    advisory is raised for a node whose Agent is not current."""

    survey = run_survey(ctx)
    started = utc_now()
    state = activity(ctx, started, metrics={"collector_observation_valid"}, kinds=())
    agent = state.get("agent") or {}
    stages = {
        "agent_present": [] if agent else ["no Agent record for the target node"],
        "no_channel_silence": []
        if not _silent_advisories(state)
        else [
            "a silent-channel advisory was raised while the Agent lease decides currency"
        ],
    }
    return _result(
        stages,
        notes=[
            "Agent expiry is a separate path (agent_is_current); not manufactured as channel silence"
        ],
        extra={
            "agent_lifecycle": agent.get("lifecycle_state"),
            "survey_units": bool(survey.get("units")),
        },
    )


def new_success(ctx: VariantContext) -> dict[str, Any]:
    """After a silence, the channel must report a new success time -- not just a
    running process -- before the node leaves UNKNOWN."""

    channel, unit, threshold = SILENCE_THRESHOLDS["host_silent"]
    run_survey(ctx)
    started = utc_now()
    before = _channel_success(ctx, started, channel)
    paused = ctx.pause("unit-pause", "--unit", unit, "--seconds", str(threshold + 300))
    ctx.cleanup.push("pause", lambda: ctx.node("unit-resume", "--unit", unit))
    if not paused.get("stopped"):
        raise VariantInconclusive(f"{unit} did not stop")
    _wait_observation_unknown(
        ctx,
        started,
        channel,
        threshold + SILENCE_SCAN_MARGIN + 180,
        reasons=SILENCE_REASONS,
    )
    ctx.node("unit-resume", "--unit", unit)
    ctx.cleanup.discard("pause")
    recovered = _await_fresh_success(ctx, channel, after=before)
    stages = {"recovery": recovered["errors"]}
    return _result(
        stages,
        extra={"channel": channel, "success_before": before, "recovery": recovered},
    )


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _hardware() -> frozenset[str]:
    from scripts.e2e.regional.general_live_registry import HARDWARE_OPERATIONS

    return HARDWARE_OPERATIONS


def _observation_metric(
    activity_document: dict[str, Any], channel: str
) -> list[dict[str, Any]]:
    return [
        item
        for item in activity_document.get("metric_latest") or []
        if item.get("name") == "collector_observation_valid"
        and item.get("device") == channel
    ]


def _observation_unknown(
    activity_document: dict[str, Any],
    channel: str,
    *,
    reasons: frozenset[str] | None = None,
) -> list[str]:
    """The channel's latest observation is UNKNOWN -- with ``reasons``, for one
    of those reasons: a paused producer's silence is STALE_OBSERVATION, while a
    collector whose polls fail reads UNKNOWN for COLLECTION_ERROR without any
    pause (025 reads that one on purpose; 031 must not take it for the pause)."""

    records = _observation_metric(activity_document, channel)
    if not records:
        return [f"no collector_observation_valid sample for {channel}"]
    latest = records[-1]
    labels = latest.get("labels") or {}
    if latest.get("value") != 0 or labels.get("state") != "UNKNOWN":
        return [f"{channel} observation is {labels}, expected UNKNOWN"]
    if reasons is not None and labels.get("reason") not in reasons:
        return [
            f"{channel} observation is UNKNOWN for {labels.get('reason')}, not the "
            f"pause's {'/'.join(sorted(reasons))}"
        ]
    return []


def _observation_state(ctx: VariantContext, since: Any) -> dict[str, Any]:
    document = activity(ctx, since, metrics={"collector_observation_valid"}, kinds=())
    return {"metric_latest": _observation_metric(document, "GPU_METRICS")}


def _wait_observation_unknown(
    ctx: VariantContext,
    since: Any,
    channel: str,
    budget: int,
    *,
    reasons: frozenset[str] | None = None,
) -> dict[str, Any]:
    last: dict[str, Any] = {}

    def accept() -> dict[str, Any] | None:
        nonlocal last
        last = activity(ctx, since, metrics={"collector_observation_valid"}, kinds=())
        return (
            last if not _observation_unknown(last, channel, reasons=reasons) else None
        )

    value = ctx.wait_until(
        accept,
        timeout_seconds=budget,
        poll_seconds=ctx.host_interval,
        name="observation-unknown",
    )
    return value if value is not None else last


def _snapshot_only(activity_document: dict[str, Any]) -> list[str]:
    for workflow in activity_document.get("workflows") or []:
        steps = set(operations(workflow))
        if steps & (_hardware() | {"RUN_DCGM_DIAGNOSTIC"}):
            return [
                f"workflow {workflow.get('request_id')} ran more than a bounded snapshot: {sorted(steps)}"
            ]
    return []


def _one_snapshot_per_hour(activity_document: dict[str, Any]) -> list[str]:
    refreshes = [
        workflow
        for workflow in activity_document.get("workflows") or []
        if "TRIGGER_HEALTH_SNAPSHOT" in operations(workflow)
    ]
    return (
        [f"{len(refreshes)} snapshot-refresh workflows in one hour (cap is one)"]
        if len(refreshes) > 1
        else []
    )


def _silent_advisories(
    activity_document: dict[str, Any], channel: str | None = None
) -> list[str]:
    """The node's ``collector-silent/<cluster>/<node>/<channel>/<bucket>``
    keys (the activity probe already keeps only this node's notifications);
    with ``channel``, only that channel's -- the metrics unit's inventory
    advisory is not its metrics advisory."""

    keys = [
        str(item.get("deduplication_key"))
        for item in activity_document.get("notifications") or []
        if str(item.get("deduplication_key") or "").startswith("collector-silent/")
    ]
    if channel is None:
        return keys
    return [key for key in keys if key.rsplit("/", 2)[-2] == channel]


def _silent_advisory(activity_document: dict[str, Any], channel: str) -> list[str]:
    if not _silent_advisories(activity_document, channel):
        return [f"no collector-silent advisory for {channel}"]
    return []


def _deduplicated_advisory(
    ctx: VariantContext, started: Any, channel: str
) -> list[str]:
    """The channel's advisory of the current alert bucket, sent before this run
    started: the product sends one per channel per bucket, so a retry inside
    the hour must accept it instead of demanding a second."""

    buckets = {
        int(moment.timestamp() // SILENT_ALERT_INTERVAL)
        for moment in (started, utc_now())
    }
    since = datetime.fromtimestamp(
        min(buckets) * SILENT_ALERT_INTERVAL, tz=timezone.utc
    )
    document = activity(ctx, since, metrics=set(), kinds=())
    return [
        key
        for key in _silent_advisories(document, channel)
        if key.rsplit("/", 1)[-1] in {str(bucket) for bucket in buckets}
    ]


def _channel_status(
    activity_document: dict[str, Any], channel: str
) -> dict[str, Any] | None:
    for status in activity_document.get("collector_statuses") or []:
        if str(status.get("collector")) == channel:
            return status
    return None


def _channel_success(ctx: VariantContext, since: Any, channel: str) -> str | None:
    status = _channel_status(activity(ctx, since, metrics=set(), kinds=()), channel)
    return status.get("last_success_at") if status else None


def recovery_budget(ctx: VariantContext, channel: str) -> int:
    """How long a resumed (or just enabled) producer may take to a success the
    harness can read: two of the channel's posts at their longest, the start-up
    spread and its random part, one poll, the silence scan that writes the
    HEALTHY observation and the sink's retry ladder -- never below the 600 s
    the wait had before.

    GPU_METRICS posts every scrape and GPU_INVENTORY every
    ``GPU_FAULT_GPU_INVENTORY_INTERVAL_SECONDS``; the host and node-log
    collectors post only on an edge or on their health summary
    (``GPU_FAULT_HOST_HEALTH_SUMMARY_SECONDS`` /
    ``GPU_FAULT_NODE_LOG_HEALTH_SUMMARY_SECONDS``, +-5 % phase move), and the
    node log collector's summary slot is a success only when that poll read
    its window cleanly (``collectors/logs/node.py`` ``collect_once``). Two
    slots is the least that makes one failed slot survivable; the figures come
    from the node's collector environment (the survey), not from fixed
    numbers. Live 2026-10-26 (``logs_silent``, release R13g): 600 s covered one
    node-log slot and a half.
    """

    cadences: dict[str, tuple[int, int]] = {
        "NODE_LOGS": (ctx.log_summary, ctx.log_interval),
        "HOST_TELEMETRY": (ctx.host_summary, ctx.host_interval),
        "GPU_METRICS": (ctx.setting("GPU_FAULT_METRICS_INTERVAL_SECONDS", 15),) * 2,
        "GPU_INVENTORY": (ctx.setting("GPU_FAULT_GPU_INVENTORY_INTERVAL_SECONDS", 60),)
        * 2,
    }
    if channel not in cadences:
        return RECOVERY_FLOOR
    cadence, poll = cadences[channel]
    derived = math.ceil(
        2 * cadence * (1 + DEFAULT_JITTER_RATIO)
        + STARTUP_SPREAD
        + STARTUP_JITTER
        + poll
        + SILENCE_SCAN
        + DELIVERY_LADDER
    )
    return max(RECOVERY_FLOOR, derived)


def journal_refusals(status: dict[str, Any] | None) -> tuple[bool, int | None]:
    """Whether the channel's status carries b190's refusal, and the collector's
    cumulative ``journalctl-failures`` count if it reports one."""

    errors = [str(item) for item in (status or {}).get("errors") or []]
    joined = " ".join(errors)
    match = JOURNAL_FAILURES.search(joined)
    return JOURNAL_REFUSAL in joined, (int(match.group(1)) if match else None)


def _no_success_detail(
    channel: str,
    budget: int,
    what: str,
    *,
    advanced: str | None,
    observation: dict[str, Any] | None,
    status: dict[str, Any] | None,
    failures: list[int],
) -> str:
    """Why the wait ran out, in the product's own words: the success that never
    came (or came without a HEALTHY observation), b190's refusal pattern when
    the status shows it, else the channel's latest collection errors."""

    if advanced:
        labels = (observation or {}).get("labels")
        detail = (
            f"{channel} reported a new success at {advanced} {what} but its "
            f"observation never read HEALTHY within {budget} s (latest {labels})"
        )
    else:
        detail = f"{channel} did not report a new success {what} within {budget} s"
    refused, _count = journal_refusals(status)
    if refused:
        counted = (
            f"journalctl-failures {failures[0]}->{failures[-1]}"
            if len(failures) > 1 and failures[0] != failures[-1]
            else f"journalctl-failures={failures[-1]}"
            if failures
            else "journalctl-failures not counted"
        )
        return f"{detail}; {REFUSAL_GAP} ({counted}: {JOURNAL_REFUSAL!r})"
    errors = [str(item) for item in (status or {}).get("errors") or []]
    if errors:
        return f"{detail}; the channel's latest errors: {errors[:2]}"
    return detail


def _await_fresh_success(
    ctx: VariantContext,
    channel: str,
    *,
    after: str | None = None,
    name: str | None = None,
    what: str = "after recovery",
) -> dict[str, Any]:
    """A success newer than ``after`` that the silence scan has read as HEALTHY,
    within ``recovery_budget``. The channel's status is kept from every read:
    when the wait runs out, the stage says what the product reported instead."""

    started = utc_now()
    budget = recovery_budget(ctx, channel)
    seen: dict[str, Any] = {"advanced": None, "observation": None, "status": None}
    failures: list[int] = []

    def accept() -> dict[str, Any] | None:
        document = activity(
            ctx, started, metrics={"collector_observation_valid"}, kinds=()
        )
        status = _channel_status(document, channel)
        seen["status"] = status
        _refused, count = journal_refusals(status)
        if count is not None and (not failures or failures[-1] != count):
            failures.append(count)
        observation = _observation_metric(document, channel)
        seen["observation"] = observation[-1] if observation else None
        success = (status or {}).get("last_success_at")
        if not success or (after is not None and str(success) <= str(after)):
            return None
        seen["advanced"] = seen["advanced"] or str(success)
        labels = (seen["observation"] or {}).get("labels") or {}
        if labels.get("state") != "HEALTHY":
            return None
        return {"last_success_at": success, "healthy": True}

    value = ctx.wait_until(
        accept,
        timeout_seconds=budget,
        poll_seconds=ctx.host_interval,
        name=name or f"{channel}-fresh-success",
    )
    record: dict[str, Any] = {
        "budget_seconds": budget,
        "waited_seconds": round((utc_now() - started).total_seconds(), 1),
        "success_before": after,
        "journalctl_failures": failures,
        "latest_status": {
            key: (seen["status"] or {}).get(key)
            for key in ("last_success_at", "last_error_at", "errors")
        }
        if seen["status"]
        else None,
    }
    if value is None:
        detail = _no_success_detail(
            channel,
            budget,
            what,
            advanced=seen["advanced"],
            observation=seen["observation"],
            status=seen["status"],
            failures=failures,
        )
        return {"errors": [detail], **record}
    return {"errors": [], **value, **record}


def _cleared_error(ctx: VariantContext, since: Any) -> list[str]:
    document = activity(
        ctx, since, metrics={"collector_observation_valid"}, kinds=("GPU_METRICS",)
    )
    for status in document.get("collector_statuses") or []:
        if str(status.get("collector")) == "GPU_METRICS":
            last_error = status.get("last_error_at")
            last_success = status.get("last_success_at")
            if last_error and last_success and str(last_success) <= str(last_error):
                return [
                    "GPU_METRICS still reports a current collection error after the refresh"
                ]
    return []


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


def _spec(
    case: str,
    variant: str,
    injection: str,
    expected: str,
    execute,
    *,
    risk="M",
    max_minutes=18,
    approvals=(),
    lanes=("eks", "hp"),
) -> VariantSpec:
    return VariantSpec(
        case_id=case,
        variant=variant,
        proof_mode=NATIVE,
        lanes=lanes,
        risk=risk,
        flags="S",
        injection=injection,
        expected=expected,
        cleanup="resume the producer; confirm a new success time advances",
        max_minutes=max_minutes,
        execute=execute,
        approvals=tuple(approvals),
    )


VARIANTS = [
    _spec(
        CASE_025,
        "exporter_timeout",
        "pause the product-owned dcgm-exporter so its scrape times out; resume it once the chain waits at its validation",
        "sustained dcgm_field_completeness error -> one bounded TRIGGER_HEALTH_SNAPSHOT + VALIDATE_HOST that waits for fresh evidence while paused and succeeds after the resume; no reset",
        exporter_timeout,
        max_minutes=20,
    ),
    _spec(
        CASE_025,
        "sustained_unknown",
        "keep the exporter paused past the 180 s sustain window",
        "the observation stays UNKNOWN; a bounded snapshot + observation validation, no GPU stress or reset",
        sustained_unknown,
        max_minutes=20,
    ),
    _spec(
        CASE_025,
        "post_refresh_success",
        "resume the exporter after the error",
        "a new complete batch clears the error; a stale success before the trigger does not pass",
        post_refresh_success,
        max_minutes=18,
    ),
    _spec(
        CASE_031,
        "inventory_silent",
        "pause the metrics collector; the GPU inventory channel goes silent (180 s)",
        "collector_observation_valid UNKNOWN, one silent advisory, one snapshot-refresh per node/hour; no arbitrary service restart",
        inventory_silent,
        max_minutes=12,
    ),
    _spec(
        CASE_031,
        "metrics_silent",
        "pause the metrics collector; the DCGM metrics channel goes silent (420 s)",
        "UNKNOWN + silent advisory + bounded refresh; no reset",
        metrics_silent,
        max_minutes=16,
    ),
    _spec(
        CASE_031,
        "host_silent",
        "pause the host collector; the host channel goes silent (420 s)",
        "UNKNOWN + silent advisory + bounded refresh",
        host_silent,
        max_minutes=16,
    ),
    _spec(
        CASE_031,
        "logs_silent",
        "enable and start the node log collector, wait for its first success, then pause it (900 s)",
        "a first success after the enable, UNKNOWN (STALE_OBSERVATION) + silent advisory under the pause, a new success within two health summaries of the resume",
        logs_silent,
        # First success (two 300 s summaries + spread), the 900 s silence with
        # its scan margin, the recovery (two summaries again) and the probe
        # round trips: ~50 minutes at the defaults.
        max_minutes=50,
        approvals=("ENABLE_NODE_LOG_COLLECTOR",),
    ),
    _spec(
        CASE_031,
        "disabled_unit",
        "observe a disabled/masked collector unit",
        "a disabled/masked unit is excluded from the silence scan: no UNKNOWN, no advisory",
        disabled_unit,
        risk="L",
        max_minutes=8,
    ),
    _spec(
        CASE_031,
        "expired_agent",
        "observe a node whose Agent lease decides currency",
        "Agent expiry is a separate path; no channel-silence advisory is manufactured for it",
        expired_agent,
        risk="L",
        max_minutes=8,
    ),
    VariantSpec(
        case_id=CASE_031,
        variant="inactive_cluster",
        proof_mode=CAPABILITY,
        lanes=("eks", "hp"),
        risk="L",
        flags="P",
        injection="not executed",
        expected="NOT_RUN(CAPABILITY_ABSENT): no inactive registered cluster on this site",
        cleanup="none",
        max_minutes=1,
        deferred="CAPABILITY_ABSENT: this site has no inactive/deregistered cluster to observe",
    ),
    VariantSpec(
        case_id=CASE_031,
        variant="notification_failure",
        proof_mode=CAPABILITY,
        lanes=("eks", "hp"),
        risk="L",
        flags="P",
        injection="not executed",
        expected="NOT_RUN(CAPABILITY_ABSENT): the live SES channel cannot be failed safely for a drill here",
        cleanup="none",
        max_minutes=1,
        deferred="CAPABILITY_ABSENT: no safe way to fail the live notification channel on this site (see NOTIFY-007 for the bounded-retry proof)",
    ),
    _spec(
        CASE_031,
        "new_success",
        "resume a paused channel and require a new success time to leave UNKNOWN",
        "recovery needs a fresh success timestamp, not merely a running process",
        new_success,
        risk="L",
        max_minutes=16,
    ),
]

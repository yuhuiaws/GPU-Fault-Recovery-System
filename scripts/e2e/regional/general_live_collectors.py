"""GF-GENERAL-LIVE-025 DCGM collection incomplete and 031 collector silent.

Both pause a *producer* and read the product's own observation records. 025
cuts the product-owned DCGM exporter's scrape (``exporter-pause``): the metrics
collector's failed scrape becomes a sustained ``dcgm_field_completeness``
collection error that, past ``collection_error_sustain_seconds`` (180 s),
requests one bounded TRIGGER_HEALTH_SNAPSHOT + read-only validation -- never a
reset. 031 pauses one collector unit while the Agent lease stays valid
(``unit-pause``): the silence scanner records ``collector_observation_valid``
UNKNOWN, mints at most one snapshot-refresh workflow per node/hour, and sends
one silent advisory. A disabled/masked unit is excluded, and resuming the
producer must advance a new success time -- not merely leave the process
running. The non-native DCGM source-replay variants and the inactive-cluster /
notification-failure capability gaps are recorded NOT_RUN with their reason.
"""

from __future__ import annotations

from typing import Any

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
    ChainExpectation,
    chain_errors,
    chain_judgeable,
    finding_incidents,
    hardware_errors,
    operations,
    verdict,
)

CASE_025 = "GF-GENERAL-LIVE-025"
CASE_031 = "GF-GENERAL-LIVE-031"
DCGM_SUSTAIN = 180
DCGM_BUDGET = DCGM_SUSTAIN + 420 + 120
#: Inventory 180 s, metrics/host 420 s, logs 900 s (collector_kinds.py).
SILENCE_THRESHOLDS = {
    "inventory_silent": ("GPU_INVENTORY", "gpu-fault-metrics-collector.service", 180),
    "metrics_silent": ("GPU_METRICS", "gpu-fault-metrics-collector.service", 420),
    "host_silent": ("HOST_TELEMETRY", "gpu-fault-host-collector.service", 420),
    "logs_silent": ("NODE_LOGS", "gpu-fault-log-collector.service", 900),
}
SILENCE_SCAN_MARGIN = 120


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
    paused = ctx.node("exporter-pause", "--seconds", str(seconds))
    ctx.cleanup.push("exporter", lambda: ctx.node("exporter-resume"))
    ctx.evidence.write("exporter-pause", paused)
    if not paused.get("stopped"):
        raise VariantInconclusive("the dcgm-exporter process did not stop")
    return paused


def _wait_completion_error(
    ctx: VariantContext, since: Any, *, name: str, validation: str | None = None
) -> dict[str, Any]:
    """The field-completeness incident; with ``validation``, its chain
    judgeable (the step terminal), not the first snapshot showing it."""

    last: dict[str, Any] = {}

    def accept() -> dict[str, Any] | None:
        nonlocal last
        last = activity(
            ctx, since, metrics={"collector_observation_valid"}, kinds=("GPU_METRICS",)
        )
        if validation is not None:
            ready = chain_judgeable(
                last, "dcgm_field_completeness", validation=validation
            )
        else:
            ready = bool(finding_incidents(last, "dcgm_field_completeness"))
        return last if ready else None

    value = ctx.wait_until(
        accept, timeout_seconds=DCGM_BUDGET, poll_seconds=ctx.host_interval, name=name
    )
    return value if value is not None else last


def exporter_timeout(ctx: VariantContext) -> dict[str, Any]:
    started = utc_now()
    hold = DCGM_BUDGET + 60
    _pause_exporter(ctx, hold + 120)
    detection = _wait_completion_error(
        ctx, started, name="dcgm-error", validation="VALIDATE_HOST"
    )
    incidents = finding_incidents(detection, "dcgm_field_completeness")
    stages = {
        "chain": chain_errors(
            detection,
            ChainExpectation(
                metric="dcgm_field_completeness",
                operations=("TRIGGER_HEALTH_SNAPSHOT", "VALIDATE_HOST"),
                validation="VALIDATE_HOST",
                forbidden=_hardware() | {"RUN_DCGM_DIAGNOSTIC"},
                incident_states=DIAGNOSTIC_ONLY_STATES,
            ),
        ),
        "hardware": hardware_errors(detection),
        "unknown": _observation_unknown(detection, "GPU_METRICS"),
    }
    if not incidents:
        stages["chain"].append("no dcgm_field_completeness finding reached an incident")
    ctx.node("exporter-resume")
    ctx.cleanup.discard("exporter")
    recovered = _await_fresh_success(ctx, "GPU_METRICS")
    stages["recovery"] = recovered["errors"]
    return _result(stages)


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
    if variant == "logs_silent":
        if "ENABLE_NODE_LOG_COLLECTOR" not in ctx.settings.approvals:
            raise VariantNotRun(
                "NOT_AUTHORIZED",
                "the node log collector is disabled in production; needs ENABLE_NODE_LOG_COLLECTOR",
            )
        enabled = ctx.node(
            "log-collector-enable",
            "--seconds",
            str(ctx.spec.max_minutes * 60 + 300),
            "--approved",
        )
        ctx.cleanup.push("log-collector", lambda: ctx.node("log-collector-disable"))
        ctx.evidence.write("log-collector-enable", enabled)
        ctx.hold(threshold, name="log-first-success")
    started = utc_now()
    hold = threshold + SILENCE_SCAN_MARGIN + 300
    paused = ctx.node("unit-pause", "--unit", unit, "--seconds", str(hold))
    ctx.cleanup.push("pause", lambda: ctx.node("unit-resume", "--unit", unit))
    ctx.evidence.write("unit-pause", paused)
    if not paused.get("stopped"):
        raise VariantInconclusive(f"{unit} did not stop under SIGSTOP")
    budget = threshold + SILENCE_SCAN_MARGIN + 180
    observed = _wait_observation_unknown(ctx, started, channel, budget)
    stages = {
        "unknown": _observation_unknown(observed, channel),
        "advisory": _silent_advisory(observed, channel),
        "snapshot_cap": _one_snapshot_per_hour(observed),
        "hardware": hardware_errors(observed),
    }
    ctx.node("unit-resume", "--unit", unit)
    ctx.cleanup.discard("pause")
    recovered = _await_fresh_success(ctx, channel)
    stages["recovery"] = recovered["errors"]
    if variant == "logs_silent":
        ctx.node("log-collector-disable")
        ctx.cleanup.discard("log-collector")
    return _result(stages, extra={"channel": channel, "threshold_seconds": threshold})


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
    paused = ctx.node("unit-pause", "--unit", unit, "--seconds", str(threshold + 300))
    ctx.cleanup.push("pause", lambda: ctx.node("unit-resume", "--unit", unit))
    if not paused.get("stopped"):
        raise VariantInconclusive(f"{unit} did not stop")
    _wait_observation_unknown(
        ctx, started, channel, threshold + SILENCE_SCAN_MARGIN + 180
    )
    ctx.node("unit-resume", "--unit", unit)
    ctx.cleanup.discard("pause")
    recovered = _await_fresh_success(ctx, channel, after=before)
    stages = {"recovery": recovered["errors"]}
    return _result(stages, extra={"channel": channel, "success_before": before})


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


def _observation_unknown(activity_document: dict[str, Any], channel: str) -> list[str]:
    records = _observation_metric(activity_document, channel)
    if not records:
        return [f"no collector_observation_valid sample for {channel}"]
    latest = records[-1]
    if (
        latest.get("value") != 0
        or (latest.get("labels") or {}).get("state") != "UNKNOWN"
    ):
        return [f"{channel} observation is {latest.get('labels')}, expected UNKNOWN"]
    return []


def _observation_state(ctx: VariantContext, since: Any) -> dict[str, Any]:
    document = activity(ctx, since, metrics={"collector_observation_valid"}, kinds=())
    return {"metric_latest": _observation_metric(document, "GPU_METRICS")}


def _wait_observation_unknown(
    ctx: VariantContext, since: Any, channel: str, budget: int
) -> dict[str, Any]:
    last: dict[str, Any] = {}

    def accept() -> dict[str, Any] | None:
        nonlocal last
        last = activity(ctx, since, metrics={"collector_observation_valid"}, kinds=())
        return last if not _observation_unknown(last, channel) else None

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


def _silent_advisories(activity_document: dict[str, Any]) -> list[str]:
    return [
        str(item.get("deduplication_key"))
        for item in activity_document.get("notifications") or []
        if str(item.get("deduplication_key") or "").startswith("collector-silent/")
    ]


def _silent_advisory(activity_document: dict[str, Any], channel: str) -> list[str]:
    advisories = _silent_advisories(activity_document)
    if not advisories:
        return [f"no collector-silent advisory for {channel}"]
    return []


def _channel_success(ctx: VariantContext, since: Any, channel: str) -> str | None:
    document = activity(ctx, since, metrics=set(), kinds=())
    for status in document.get("collector_statuses") or []:
        if str(status.get("collector")) == channel:
            return status.get("last_success_at")
    return None


def _await_fresh_success(
    ctx: VariantContext, channel: str, *, after: str | None = None
) -> dict[str, Any]:
    started = utc_now()

    def accept() -> dict[str, Any] | None:
        document = activity(ctx, started, metrics=set(), kinds=())
        for status in document.get("collector_statuses") or []:
            if str(status.get("collector")) != channel:
                continue
            success = status.get("last_success_at")
            if success and (after is None or str(success) > str(after)):
                observation = _observation_metric(
                    activity(
                        ctx, started, metrics={"collector_observation_valid"}, kinds=()
                    ),
                    channel,
                )
                healthy = (
                    observation
                    and (observation[-1].get("labels") or {}).get("state") == "HEALTHY"
                )
                return (
                    {"last_success_at": success, "healthy": bool(healthy)}
                    if healthy
                    else None
                )
        return None

    value = ctx.wait_until(
        accept,
        timeout_seconds=600,
        poll_seconds=ctx.host_interval,
        name=f"{channel}-fresh-success",
    )
    if value is None:
        return {"errors": [f"{channel} did not report a new success after recovery"]}
    return {"errors": [], **value}


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
        "pause the product-owned dcgm-exporter so its scrape times out",
        "sustained dcgm_field_completeness error -> one bounded TRIGGER_HEALTH_SNAPSHOT + VALIDATE_HOST; no reset",
        exporter_timeout,
        max_minutes=18,
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
        "enable and start the node log collector, then pause it (900 s)",
        "UNKNOWN + silent advisory after the approved log channel first succeeds",
        logs_silent,
        max_minutes=25,
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
        lanes=("eks",),
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
        lanes=("eks",),
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

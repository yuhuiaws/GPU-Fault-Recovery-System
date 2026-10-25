"""GF-GENERAL-LIVE-002 host memory and swap exhaustion.

The host collector reads ``/proc/meminfo`` (``collectors/host/system_metrics.py``):
``memory_used_percent`` >= 95 is a WARNING RUN_DIAGNOSTICS rule with a 120 s
sustain (``host_health.py`` METRIC_RULES + METRIC_RULE_SUSTAIN_SECONDS),
compiled to FREEZE_EVIDENCE then VALIDATE_HOST with a ``host_diagnostic_probe``
on ``memory_used_percent`` that fails while the pressure is held. ``swap_used_percent``
>= 80 is the swap rule.

The native pressure is the existing node probe's ``memory-start``: an allocator
in its own capped cgroup (``MemoryMax``, ``MemorySwapMax=0``, ``OOMPolicy=kill``)
that holds host ``MemAvailable`` low without exhausting the host, behind the
``HOST_MEMORY_PRESSURE`` approval and a deadman. The runner's own ``/proc/meminfo``
sampler is the effectiveness oracle; a tick that never crossed 95 % used makes
the variant INCONCLUSIVE rather than PASS, and a cgroup OOM that never moved the
host number would never be mistaken for host pressure.

``cgroup_only`` is the negative control, from the same probe's own-cgroup OOM
mode (``memory-start --cgroup-oom-bytes``): a test cgroup capped far below the
host (2 GiB; the probe refuses more than 1 % of MemTotal or the host's headroom
under the rule) is exhausted so the kernel OOM-kills the allocator *inside*
it -- systemd keeps the failed unit and ``memory-status`` reads its
``Result=oom-kill`` -- while host ``memory_used_percent`` never moves.
Repeated across a window longer than the rule's sustain plus a health summary,
the product must mint no memory finding; an OOM line the (disabled by default)
node log collector grades is a RUN_DIAGNOSTICS diagnostic at most, never a
STOP, isolation or hardware step (``log_rules.py``).

The spec's swap and read-failure variants need either a swap device this site
does not run (the spec forbids enabling business swap) or a unit-private
``/proc/meminfo`` replay that is not built in this phase; they stay in the
denominator as NOT_RUN with their reason.
"""

from __future__ import annotations

from typing import Any

from scripts.e2e.regional.general_live_common import (
    await_metric_recovery,
    Oracle,
    activity,
    run_survey,
    wait_for_incident,
    wait_settled,
)
from scripts.e2e.regional.general_live_fixture import (
    VariantBlocked,
    VariantContext,
    VariantInconclusive,
    VariantNotRun,
    require_tools,
    utc_now,
)
from scripts.e2e.regional.general_live_registry import (
    CAPABILITY,
    LIVE_REPLAY,
    NATIVE,
    OBSERVATION,
    VariantSpec,
    deferred_variant,
)
from scripts.e2e.regional.general_live_verdicts import (
    DIAGNOSTIC_ONLY_STATES,
    ChainExpectation,
    chain_errors,
    finding_incidents,
    hardware_errors,
    isolation_operations,
    minted_in_read,
    node_isolated,
    operations,
    oracle_values,
    samples,
    sustained,
    verdict,
    workflows_of,
)

CASE = "GF-GENERAL-LIVE-002"
USED_THRESHOLD = 95.0
SWAP_THRESHOLD = 80.0
SUSTAIN_SECONDS = 120
#: window + summary + a few ticks, as 001: detection lands on the next
#: delivered sample after the sustain window, up to a health-summary later.
DETECTION_BUDGET = SUSTAIN_SECONDS + 300 + 120
#: Hold host memory_used at about this percentage; below the probe's 97 % cap.
TARGET_USED_PERCENT = 96.5
#: How long the allocator may take to reach the threshold before detection is
#: timed: it touches memory at a fixed pace, so a 2 TiB p5en host needs about
#: 14 minutes (live 2026-10-24: 76 % after the 9-minute detection budget).
RAMP_BUDGET = 1080
#: cgroup_only: the test cgroup's cap, 0.1 % of a 2 TiB host (the probe refuses
#: more than 1 % of MemTotal, or more than the host's headroom under the rule).
CGROUP_OOM_CAP_BYTES = 2 << 30
#: How long one allocator may take to be OOM-killed inside its cgroup.
CGROUP_OOM_BUDGET = 300
#: How many times the cgroup is exhausted across the negative window.
CGROUP_OOM_REPEATS = 3


def _require_memory_approval(ctx: VariantContext) -> None:
    if "HOST_MEMORY_PRESSURE" not in ctx.settings.approvals:
        raise VariantNotRun(
            "NOT_AUTHORIZED", "host memory pressure needs HOST_MEMORY_PRESSURE"
        )


def _run_memory_load(ctx: VariantContext, hold_seconds: int) -> Oracle:
    oracle = Oracle(ctx, "mem")
    oracle.start(hold_seconds + 120, interval=5, meminfo=True)
    started = ctx.node(
        "memory-start",
        "--target-used-percent",
        str(TARGET_USED_PERCENT),
        "--seconds",
        str(hold_seconds + 120),
        "--approved",
    )
    ctx.cleanup.push("memory", lambda: ctx.node("memory-stop"))
    ctx.evidence.write("memory-start", started)
    return oracle


def _await_ramp(ctx: VariantContext, oracle: Oracle) -> None:
    """Wait until the oracle reads the threshold; detection is timed from there."""

    def reached() -> bool | None:
        oracle.read()
        values = oracle_values(
            oracle.records, lambda record: record.get("memory_used_percent")
        )
        return True if values and values[-1] >= USED_THRESHOLD else None

    if not ctx.wait_until(
        reached, timeout_seconds=RAMP_BUDGET, poll_seconds=15, name="memory-ramp"
    ):
        raise VariantInconclusive(
            f"the allocator did not raise memory_used to {USED_THRESHOLD}% within "
            f"{RAMP_BUDGET} s"
        )


def _oracle_used_each_tick(oracle: Oracle, minimum: int) -> None:
    oracle.read()
    values = oracle_values(
        oracle.records, lambda record: record.get("memory_used_percent")
    )
    if not sustained(values, lambda value: value >= USED_THRESHOLD, minimum):
        raise VariantInconclusive(
            f"the memory oracle never read >= {USED_THRESHOLD}% used on {minimum} "
            f"consecutive ticks: {[round(value, 1) for value in values[-12:]]}"
        )


def memory_pressure(ctx: VariantContext) -> dict[str, Any]:
    survey = run_survey(ctx)
    require_tools(survey, "systemd-run")
    _require_memory_approval(ctx)
    started_at = utc_now()
    hold = RAMP_BUDGET + DETECTION_BUDGET + 240
    oracle = _run_memory_load(ctx, hold)
    _await_ramp(ctx, oracle)
    detection = wait_for_incident(
        ctx,
        started_at,
        "memory_used_percent",
        timeout_seconds=DETECTION_BUDGET,
        metrics={"memory_used_percent", "memory_available_percent"},
        name="memory-detection",
    )
    _oracle_used_each_tick(oracle, minimum=SUSTAIN_SECONDS // 5)
    incidents = finding_incidents(detection, "memory_used_percent")

    def validated() -> dict[str, Any] | None:
        current = activity(ctx, started_at, metrics={"memory_used_percent"})
        for inc in finding_incidents(current, "memory_used_percent"):
            for workflow in current.get("workflows") or []:
                if workflow.get("incident_id") != inc["incident_id"]:
                    continue
                if any(
                    step.get("operation") == "VALIDATE_HOST"
                    and step.get("status") in {"SUCCEEDED", "FAILED"}
                    for step in workflow.get("step_executions") or []
                ):
                    return current
        return None

    observed = ctx.wait_until(
        validated,
        timeout_seconds=DETECTION_BUDGET,
        poll_seconds=20,
        name="memory-validate",
    ) or activity(ctx, started_at, metrics={"memory_used_percent"})
    stages = {
        "chain": chain_errors(
            observed,
            ChainExpectation(
                metric="memory_used_percent",
                operations=("FREEZE_EVIDENCE", "VALIDATE_HOST"),
                validation="VALIDATE_HOST",
                validation_must_fail=True,
                probe_metric="memory_used_percent",
                incident_states=DIAGNOSTIC_ONLY_STATES,
            ),
        ),
        "hardware": hardware_errors(observed),
    }
    stopped_at = utc_now()
    ctx.node("memory-stop")
    ctx.cleanup.discard("memory")
    recovered = _await_recovery(ctx, oracle, since=stopped_at)
    stages["recovery"] = recovered["errors"]
    settled = wait_settled(
        ctx,
        started_at,
        [item["incident_id"] for item in incidents],
        timeout_seconds=DETECTION_BUDGET,
        metrics={"memory_used_percent"},
        name="memory-settled",
    )
    return _result(
        stages,
        notes=["memory_available_percent (003 LOW_MEMORY_AVAILABLE) co-fires"],
        extra={"recovery": recovered, "settled": _statuses(settled)},
    )


def recovery(ctx: VariantContext) -> dict[str, Any]:
    survey = run_survey(ctx)
    require_tools(survey, "systemd-run")
    _require_memory_approval(ctx)
    started_at = utc_now()
    hold = RAMP_BUDGET + DETECTION_BUDGET + 60
    oracle = _run_memory_load(ctx, hold)
    _await_ramp(ctx, oracle)
    detection = wait_for_incident(
        ctx,
        started_at,
        "memory_used_percent",
        timeout_seconds=DETECTION_BUDGET,
        metrics={"memory_used_percent"},
        name="recovery-detection",
    )
    _oracle_used_each_tick(oracle, minimum=SUSTAIN_SECONDS // 5)
    incidents = finding_incidents(detection, "memory_used_percent")
    stopped_at = utc_now()
    ctx.node("memory-stop")
    ctx.cleanup.discard("memory")
    recovered = _await_recovery(ctx, oracle, since=stopped_at)
    settled = wait_settled(
        ctx,
        started_at,
        [item["incident_id"] for item in incidents],
        timeout_seconds=DETECTION_BUDGET,
        metrics={"memory_used_percent"},
        name="recovery-settled",
    )
    stages = {"recovery": recovered["errors"], "hardware": hardware_errors(detection)}
    if not incidents:
        stages["recovery"].append("no memory finding to recover from")
    return _result(stages, extra={"recovery": recovered, "settled": _statuses(settled)})


def no_swap(ctx: VariantContext) -> dict[str, Any]:
    """A swapless node: confirm the swap rule stays inert and record the fact.

    Read-only. ``swap_used_percent`` is 0 whenever ``SwapTotal`` is 0
    (``system_metrics.py``), so its 80 % rule can never fire -- the capability
    the spec asks to be recorded rather than worked around by enabling swap.
    """

    survey = run_survey(ctx)
    swap_total = ((survey.get("memory") or {}).get("SwapTotal")) or 0
    if swap_total:
        raise VariantBlocked(
            "CAPABILITY_ABSENT",
            f"the node has {swap_total} bytes of swap; this variant is for a "
            "swapless node (use swap_pressure instead)",
        )
    started_at = utc_now()
    ctx.hold(ctx.host_interval * 4, name="no-swap-hold")
    observed = activity(ctx, started_at, metrics={"swap_used_percent"})
    swap_samples = samples(observed, "swap_used_percent")
    stages = {
        "no_swap_finding": chain_errors(
            observed, ChainExpectation(metric="swap_used_percent", present=False)
        ),
        "swap_zero": []
        if all(
            not isinstance(item.get("value"), (int, float)) or item["value"] == 0
            for item in swap_samples
        )
        else ["swap_used_percent was non-zero on a node reporting no swap"],
    }
    return _result(
        stages,
        notes=["SwapTotal is 0; the swap_used_percent rule is inert by construction"],
        extra={"swap_total_bytes": swap_total, "swap_samples": swap_samples[-3:]},
    )


# --------------------------------------------------------------------------- #
# cgroup_only: an OOM inside a small test cgroup is not host pressure
# --------------------------------------------------------------------------- #
def _oom_cycle(ctx: VariantContext, cycle: int) -> dict[str, Any]:
    """One bounded own-cgroup OOM: start, await systemd's oom-kill result, reset.

    The allocator runs in a transient unit capped at ``CGROUP_OOM_CAP_BYTES``
    with no swap and ``OOMPolicy=kill``; the kernel's cgroup OOM killer ends it
    and systemd records ``Result=oom-kill`` on the (uncollected) failed unit,
    which ``memory-stop`` then resets. Anything but that result -- the unit ran
    out its RuntimeMaxSec, exited, or never ended within the budget -- means the
    cgroup was not exhausted, so the variant is INCONCLUSIVE, never PASS.
    """

    started = ctx.node(
        "memory-start",
        "--cgroup-oom-bytes",
        str(CGROUP_OOM_CAP_BYTES),
        "--seconds",
        str(CGROUP_OOM_BUDGET),
        "--approved",
    )
    ctx.cleanup.push("memory", lambda: ctx.node("memory-stop"))
    ctx.evidence.write(f"cgroup-oom-start-{cycle}", started)

    def ended() -> dict[str, Any] | None:
        status = ctx.node("memory-status")
        unit = status.get("unit") or {}
        if unit.get("Result") == "oom-kill" or unit.get("ActiveState") in {
            "failed",
            "inactive",
        }:
            return status
        return None

    status = ctx.wait_until(
        ended,
        timeout_seconds=CGROUP_OOM_BUDGET,
        poll_seconds=5,
        name=f"cgroup-oom-{cycle}",
    )
    stopped = ctx.node("memory-stop")
    ctx.cleanup.discard("memory")
    ctx.evidence.write(f"cgroup-oom-stop-{cycle}", stopped)
    if status is None:
        raise VariantInconclusive(
            f"cycle {cycle}: the allocator was still running after "
            f"{CGROUP_OOM_BUDGET} s; the cgroup cap was never reached"
        )
    unit = status.get("unit") or {}
    if unit.get("Result") != "oom-kill":
        raise VariantInconclusive(
            f"cycle {cycle}: the allocator ended {unit.get('ActiveState')}/"
            f"{unit.get('Result')} instead of oom-kill inside its cgroup"
        )
    record = {
        "cycle": cycle,
        "result": unit.get("Result"),
        "exec_main_code": unit.get("ExecMainCode"),
        "exec_main_status": unit.get("ExecMainStatus"),
        "memory_max_bytes": started.get("memory_max_bytes"),
        **dict((status or {}).get("memory") or {}),
    }
    ctx.evidence.append_oracle([record], source="cgroup-oom")
    return record


def _log_grading_errors(observed: dict[str, Any]) -> list[str]:
    """An OOM log finding, if any, is a RUN_DIAGNOSTICS diagnostic, never a STOP.

    The node log collector (disabled by default) grades ``out of memory`` /
    ``killed process`` lines WARNING RUN_DIAGNOSTICS from any origin
    (``log_rules.py``); the kernel collector only extracts XID/SXID records.
    """

    errors = []
    for item in observed.get("incidents") or []:
        event_id = str(item.get("event_id") or "")
        if item.get("event_type") != "NODE_HEALTH" or not event_id.startswith("log-"):
            continue
        if not minted_in_read(observed, item):
            continue
        action = item.get("effective_action")
        if action != "RUN_DIAGNOSTICS":
            errors.append(
                f"OOM log incident {item['incident_id']} was graded {action}, "
                "not RUN_DIAGNOSTICS"
            )
        for workflow in workflows_of(observed, str(item["incident_id"])):
            if "STOP_WORKLOADS" in operations(workflow):
                errors.append(
                    f"workflow {workflow.get('request_id')} stopped a workload "
                    "for a cgroup OOM line"
                )
    return errors


def cgroup_only(ctx: VariantContext) -> dict[str, Any]:
    """A cgroup-local OOM is not host pressure: no memory finding, no isolation."""

    survey = run_survey(ctx)
    require_tools(survey, "systemd-run")
    _require_memory_approval(ctx)
    total = float((survey.get("memory") or {}).get("MemTotal") or 0)
    started_at = utc_now()
    # A finding mis-graded from the OOM would have to show inside the rule's
    # sustain plus one health summary; the window outlasts both.
    window = SUSTAIN_SECONDS + ctx.host_summary + ctx.host_interval * 4
    oracle = Oracle(ctx, "cgroup")
    oracle.start(
        window + CGROUP_OOM_REPEATS * CGROUP_OOM_BUDGET + 120, interval=5, meminfo=True
    )
    kills = []
    for cycle in range(CGROUP_OOM_REPEATS):
        kills.append(_oom_cycle(ctx, cycle))
        ctx.hold(window / CGROUP_OOM_REPEATS, name=f"cgroup-window-{cycle}")
    oracle.read()
    host_used = oracle_values(
        oracle.records, lambda record: record.get("memory_used_percent")
    )
    if not host_used:
        raise VariantInconclusive("the meminfo oracle recorded no host reading")
    if max(host_used) >= USED_THRESHOLD:
        raise VariantInconclusive(
            f"host memory_used reached {max(host_used):.1f}% during the window: the "
            "host itself was under pressure, so a cgroup-local OOM cannot be told apart"
        )
    observed = activity(
        ctx,
        started_at,
        metrics={"memory_used_percent", "memory_available_percent"},
        kinds=("HOST_TELEMETRY", "NODE_LOGS"),
    )
    host_samples = [
        item
        for item in samples(observed, "memory_used_percent")
        if isinstance(item.get("value"), (int, float))
    ]
    if not host_samples:
        raise VariantInconclusive(
            "no accepted host memory sample reached the product during the window; "
            "a silent channel cannot prove how the OOM was graded"
        )
    isolation = isolation_operations(observed)
    node = ctx.regional.node_snapshot(ctx.settings.node)
    stages = {
        "no_host_pressure_finding": chain_errors(
            observed, ChainExpectation(metric="memory_used_percent", present=False)
        ),
        "no_low_memory_finding": chain_errors(
            observed,
            ChainExpectation(metric="memory_available_percent", present=False),
        ),
        "host_samples_normal": [
            f"the product accepted memory_used_percent {item['value']} at "
            f"{item['observed_at']}"
            for item in host_samples
            if item["value"] >= USED_THRESHOLD
        ],
        "log_grading": _log_grading_errors(observed),
        "hardware": hardware_errors(observed),
        "no_isolation": (
            [f"isolation operations planned: {isolation}"] if isolation else []
        )
        + (
            ["the node is isolated after a cgroup-local OOM"]
            if node_isolated(node)
            else []
        ),
    }
    return _result(
        stages,
        notes=[
            "a cgroup-local OOM proves cgroup exhaustion only, never host "
            "memory_used >= 95 %",
            "an OOM log line, when the node log collector is enabled, is a "
            "RUN_DIAGNOSTICS diagnostic at most",
        ],
        extra={
            "cgroup_oom_kills": kills,
            "cgroup_cap_percent_of_host": (
                100 * CGROUP_OOM_CAP_BYTES / total if total else None
            ),
            "host_used_percent": {
                "min": min(host_used),
                "max": max(host_used),
                "ticks": len(host_used),
            },
            "host_samples": host_samples[-3:],
        },
    )


# --------------------------------------------------------------------------- #
# shared helpers
# --------------------------------------------------------------------------- #
def _await_recovery(
    ctx: VariantContext, oracle: Oracle, since: Any = None
) -> dict[str, Any]:
    """Fresh evidence that host memory use fell back below the threshold."""

    result = await_metric_recovery(
        ctx, "memory_used_percent", USED_THRESHOLD, name="memory-recovery", since=since
    )
    oracle.read()
    return result


def _statuses(activity_document: dict[str, Any]) -> list[dict[str, Any]]:
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


VARIANTS = [
    VariantSpec(
        case_id=CASE,
        variant="memory_pressure",
        proof_mode=NATIVE,
        lanes=("eks", "hp"),
        risk="H",
        flags="S",
        injection="hold host MemAvailable low (~96.5 % used) in a capped cgroup for > 120 s",
        expected="memory_used_percent >= 95 for 120 s -> VALIDATE_HOST probe fails while loaded; no reset/reboot",
        cleanup="stop the allocator; 2 fresh samples below 95 %",
        max_minutes=45,
        approvals=("HOST_MEMORY_PRESSURE",),
        execute=memory_pressure,
    ),
    deferred_variant(
        CASE,
        "swap_pressure",
        proof_mode=CAPABILITY,
        reason=(
            "CAPABILITY_ABSENT: the p5en/HyperPod nodes run no swap device and "
            "the spec forbids enabling business swap for a test"
        ),
        lanes=("eks", "hp"),
        expected="swap_used_percent >= 80 -> VALIDATE_HOST on its own swap rule",
    ),
    VariantSpec(
        case_id=CASE,
        variant="no_swap",
        proof_mode=OBSERVATION,
        lanes=("eks", "hp"),
        risk="L",
        flags="S",
        injection="none (read-only): confirm SwapTotal is 0",
        expected="the swap_used_percent rule is inert; the swapless capability is recorded",
        cleanup="none",
        max_minutes=8,
        execute=no_swap,
    ),
    VariantSpec(
        case_id=CASE,
        variant="cgroup_only",
        proof_mode=NATIVE,
        lanes=("eks", "hp"),
        risk="M",
        flags="S",
        injection=(
            "exhaust a 2 GiB test cgroup (MemoryMax, no swap, OOMPolicy=kill) 3 times "
            "across a window longer than the memory rule's sustain plus a health "
            "summary; the kernel OOM-kills the allocator inside the cgroup and host "
            "memory_used stays normal"
        ),
        expected=(
            "no memory_used_percent / memory_available_percent finding; an OOM log "
            "line is RUN_DIAGNOSTICS at most; no STOP, isolation or hardware step; "
            "the node stays schedulable"
        ),
        cleanup=(
            "reset the OOM-killed test unit after each cycle, disarm its deadman, "
            "stop the oracle"
        ),
        max_minutes=30,
        approvals=("HOST_MEMORY_PRESSURE",),
        execute=cgroup_only,
    ),
    deferred_variant(
        CASE,
        "read_failure",
        proof_mode=LIVE_REPLAY,
        reason=(
            "FIXTURE_NOT_READY: a /proc/meminfo read failure needs a unit-private "
            "meminfo replay (the shared raw-source replay helper, not built this phase)"
        ),
        lanes=("eks",),
        expected="a meminfo read error is UNKNOWN, never a healthy or a hardware signal",
    ),
    VariantSpec(
        case_id=CASE,
        variant="recovery",
        proof_mode=NATIVE,
        lanes=("eks", "hp"),
        risk="M",
        flags="S",
        injection="drive memory_pressure to a finding, then stop the allocator",
        expected="2 fresh samples below 95 % and the incident settles; no reboot masks the OOM",
        cleanup="none",
        max_minutes=45,
        approvals=("HOST_MEMORY_PRESSURE",),
        execute=recovery,
    ),
]

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

The spec's swap, read-failure and own-cgroup-OOM variants need either a swap
device this site does not run (the spec forbids enabling business swap) or a
unit-private ``/proc/meminfo`` / own-cgroup-OOM fixture that is not built in
this phase; they stay in the denominator as NOT_RUN with their reason.
"""

from __future__ import annotations

from typing import Any

from scripts.e2e.regional.general_live_common import (
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
    ChainExpectation,
    chain_errors,
    finding_incidents,
    hardware_errors,
    oracle_values,
    samples,
    sustained,
    verdict,
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
    hold = DETECTION_BUDGET + 240
    oracle = _run_memory_load(ctx, hold)
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
                incident_states=frozenset({"ACTION_PENDING", "ESCALATED"}),
            ),
        ),
        "hardware": hardware_errors(observed),
    }
    ctx.node("memory-stop")
    ctx.cleanup.discard("memory")
    recovered = _await_recovery(ctx, oracle)
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
    hold = DETECTION_BUDGET + 60
    oracle = _run_memory_load(ctx, hold)
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
    ctx.node("memory-stop")
    ctx.cleanup.discard("memory")
    recovered = _await_recovery(ctx, oracle)
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
    return _result(
        stages, extra={"recovery": recovered, "settled": _statuses(settled)}
    )


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
# shared helpers
# --------------------------------------------------------------------------- #
def _await_recovery(ctx: VariantContext, oracle: Oracle) -> dict[str, Any]:
    started = utc_now()

    def accept() -> dict[str, Any] | None:
        current = activity(ctx, started, metrics={"memory_used_percent"})
        fresh = [
            sample
            for sample in samples(current, "memory_used_percent")
            if isinstance(sample.get("value"), (int, float))
            and sample["value"] < USED_THRESHOLD
        ]
        return {"samples": fresh} if len(fresh) >= 3 else None

    value = ctx.wait_until(
        accept,
        timeout_seconds=ctx.host_summary + ctx.host_interval * 6,
        poll_seconds=ctx.host_interval,
        name="memory-recovery",
    )
    oracle.read()
    if value is None:
        return {"errors": ["memory_used_percent did not report 3 fresh samples < 95 %"]}
    return {"errors": [], "fresh_samples": value["samples"]}


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
        cleanup="stop the allocator; 3 fresh samples below 95 %",
        max_minutes=22,
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
    deferred_variant(
        CASE,
        "cgroup_only",
        proof_mode=NATIVE,
        reason=(
            "FIXTURE_NOT_READY: a container OOM that does not move host "
            "memory_used needs a bounded own-cgroup OOM injection mode the node "
            "probe does not have yet"
        ),
        lanes=("eks", "hp"),
        expected="a cgroup OOM is not host pressure: no memory_used_percent finding",
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
        expected="3 fresh samples below 95 % and the incident settles; no reboot masks the OOM",
        cleanup="none",
        max_minutes=22,
        approvals=("HOST_MEMORY_PRESSURE",),
        execute=recovery,
    ),
]

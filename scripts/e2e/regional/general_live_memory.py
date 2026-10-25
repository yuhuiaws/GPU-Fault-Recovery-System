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

The spec's swap and read-failure variants are raw ``/proc/meminfo`` replays on
the shared host source replay (``general_live_source_replay.py``; the spec
allows a raw meminfo replay where the node has no swap device and business swap
must not be enabled): ``swap_pressure`` binds a copy of the node's real
``/proc/meminfo`` with only ``SwapTotal``/``SwapFree`` rewritten into the host
collector unit's mount namespace, so ``_memory`` computes ``swap_used_percent``
from the replayed pair and the 80 % swap rule (no sustain) fires on its own
while the memory metrics stay the node's real values; ``read_failure`` binds
the replay's unopenable node over ``/proc/meminfo`` inside that unit only (not
``InaccessiblePaths=``: the unit runs as root and reads systemd's mode-0000
node as an *empty* file), so the contributor's ``open`` fails and the tick
carries a ``_memory`` collection error and no memory sample at all -- UNKNOWN,
never a healthy value and never a hardware signal.
"""

from __future__ import annotations

from datetime import datetime
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
    LIVE_REPLAY,
    NATIVE,
    OBSERVATION,
    VariantSpec,
)
from scripts.e2e.regional.general_live_source_replay import (
    HOST_SOURCE_REPLAY,
    MEMINFO_FIXTURE,
    host_source_replay,
)
from scripts.e2e.regional.probes.general_live_node_probe import (
    MAX_SECONDS as PROBE_MAX_SECONDS,
)
from scripts.e2e.regional.regional_commands import RegionalFixtureError
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
#: The memory metrics the host collector derives from ``/proc/meminfo``
#: (``system_metrics._memory``) that carry a rule.
MEMORY_METRICS = frozenset(
    {"memory_used_percent", "memory_available_percent", "swap_used_percent"}
)
#: swap_pressure fixture: 8 GiB of swap with 1 GiB free -> 87.5 % used, above
#: the 80 % rule; every other meminfo line stays the node's real reading.
SWAP_TOTAL_KB = 8 << 20
SWAP_FREE_KB = 1 << 20
#: The swap rule has no sustain: the first delivered breach (the baseline tick
#: after the unit restart) mints the finding; the budget still allows a health
#: summary for the validation to execute.
SWAP_DETECTION_BUDGET = 300 + 120


def _require_memory_approval(ctx: VariantContext) -> None:
    if "HOST_MEMORY_PRESSURE" not in ctx.settings.approvals:
        raise VariantNotRun(
            "NOT_AUTHORIZED", "host memory pressure needs HOST_MEMORY_PRESSURE"
        )


def _probe_seconds(hold_seconds: int) -> int:
    """The unit seconds asked of the node probe, checked against its cap first.

    Live 2026-10-25 (both plain-EKS nodes, three times): memory_pressure asked
    for a 1980 s hold -- the 14-minute ramp of a 2 TiB host plus detection --
    and the probe's memory-start, then capped at 1800 s, refused it before any
    product evidence, with the message withheld by policy. The probe's cap is
    now its common load-unit cap (``MAX_SECONDS``) and this guard fails the
    variant with the numbers *before* a probe Pod touches the node.
    """

    seconds = int(hold_seconds) + 120
    if seconds > PROBE_MAX_SECONDS:
        raise RegionalFixtureError(
            f"a {seconds} s memory hold exceeds the node probe's {PROBE_MAX_SECONDS} s "
            "unit cap (memory-start); shorten the ramp/detection budgets"
        )
    return seconds


def _run_memory_load(ctx: VariantContext, hold_seconds: int) -> Oracle:
    seconds = _probe_seconds(hold_seconds)
    oracle = Oracle(ctx, "mem")
    oracle.start(seconds, interval=5, meminfo=True)
    started = ctx.node(
        "memory-start",
        "--target-used-percent",
        str(TARGET_USED_PERCENT),
        "--seconds",
        str(seconds),
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
    observed = _await_validate_host(
        ctx, started_at, "memory_used_percent", name="memory-validate"
    )
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
# read_failure / swap_pressure: raw /proc/meminfo replay in the host collector
# unit's private mount namespace (HOST_SOURCE_REPLAY)
# --------------------------------------------------------------------------- #
def memory_samples(observed: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        item
        for metric in sorted(MEMORY_METRICS)
        for item in samples(observed, metric)
        if isinstance(item.get("value"), (int, float))
    ]


def contributor_error_errors(observed: dict[str, Any], contributor: str) -> list[str]:
    """At least one delivered batch names the failed contributor."""

    for batch in observed.get("batches") or []:
        if any(
            str(error).startswith(f"{contributor}:")
            for error in batch.get("collection_errors") or []
        ):
            return []
    return [f"no delivered batch carries a {contributor} collection error"]


def _await_memory_sample(ctx: VariantContext, since: datetime) -> list[str]:
    """Fresh evidence that the memory contributor reads ``/proc/meminfo`` again."""

    def accept() -> dict[str, Any] | None:
        current = activity(ctx, since, metrics={"memory_used_percent"})
        fresh = [
            item
            for item in samples(current, "memory_used_percent")
            if isinstance(item.get("value"), (int, float))
        ]
        return {"samples": fresh} if fresh else None

    value = ctx.wait_until(
        accept,
        timeout_seconds=ctx.host_summary + ctx.host_interval * 6,
        poll_seconds=ctx.host_interval,
        name="meminfo-recovery",
    )
    if value is None:
        return ["no memory_used_percent sample reached the product after the window"]
    return []


def read_failure(ctx: VariantContext) -> dict[str, Any]:
    """An unreadable ``/proc/meminfo`` is UNKNOWN: no value, no finding, no step.

    The probe verifies through ``nsenter`` that the unit cannot read the file
    while PID 1 can (the runner's oracle); the product's records must then show
    a ``_memory`` collection error and *no* memory sample for the window -- a
    healthy value or a hardware/isolation step from a failed read is FAIL.
    """

    run_survey(ctx)
    replay = host_source_replay(ctx)
    started_at = utc_now()
    opened = replay.open_window(
        inaccessible=["/proc/meminfo"], restore_seconds=ctx.spec.max_minutes * 60
    )
    ctx.evidence.write("source-window-open", opened)
    hidden = [
        check
        for check in (opened.get("verification") or {}).get("checks") or []
        if check.get("kind") == "inaccessible" and check.get("dest") == "/proc/meminfo"
    ]
    if not hidden or not all(check.get("ok") for check in hidden):
        raise VariantInconclusive(
            "the probe did not verify that /proc/meminfo is unreadable inside the "
            f"host collector unit only: {hidden}"
        )
    ctx.evidence.append_oracle(
        [{"inaccessible_checks": hidden, **replay.evidence()}], source="meminfo"
    )
    ctx.hold(ctx.host_interval * 3 + 30, name="read-failure")
    observed = activity(ctx, started_at, metrics=MEMORY_METRICS)
    stages = {
        "unknown_observed": contributor_error_errors(observed, "_memory"),
        "no_healthy_value": [
            f"{item['name']}={item['value']} was reported at {item['observed_at']} "
            "while /proc/meminfo was unreadable"
            for item in memory_samples(observed)
        ],
        "no_finding": [
            error
            for metric in sorted(MEMORY_METRICS)
            for error in chain_errors(
                observed, ChainExpectation(metric=metric, present=False)
            )
        ],
        "hardware": hardware_errors(observed),
        "no_isolation": (
            [f"isolation operations planned: {isolation_operations(observed)}"]
            if isolation_operations(observed)
            else []
        )
        + (
            ["the node is isolated after a meminfo read failure"]
            if node_isolated(ctx.regional.node_snapshot(ctx.settings.node))
            else []
        ),
    }
    closed_at = utc_now()
    ctx.evidence.write("source-window-close", replay.close())
    stages["recovered"] = _await_memory_sample(ctx, closed_at)
    return _result(
        stages,
        notes=[
            "a failed /proc/meminfo read leaves a _memory collection error and no "
            "memory sample: UNKNOWN, never a healthy or a hardware signal"
        ],
        extra={
            "proof_scope": "live-replay",
            "batches_in_window": len(observed.get("batches") or []),
        },
    )


def swap_pressure(ctx: VariantContext) -> dict[str, Any]:
    """A replayed SwapTotal/SwapFree pair drives the swap rule on its own.

    The fixture is the node's real ``/proc/meminfo`` with only the swap pair
    rewritten (8 GiB total, 1 GiB free = 87.5 %), so ``memory_used_percent``
    and the page-cache rule keep their real inputs and only ``swap_used_percent``
    crosses its 80 % rule (``host_health.py`` METRIC_RULES; no sustain). The
    product must mint the swap finding alone, run FREEZE_EVIDENCE then
    VALIDATE_HOST with a ``host_diagnostic_probe`` on ``swap_used_percent`` that
    fails while the bind is held, and recover once the window closes.
    """

    run_survey(ctx)
    replay = host_source_replay(ctx)
    built = replay.build_meminfo(SwapTotal=SWAP_TOTAL_KB, SwapFree=SWAP_FREE_KB)
    ctx.evidence.write("meminfo-fixture", built)
    fixture_swap = built.get("swap_used_percent")
    if not isinstance(fixture_swap, (int, float)) or fixture_swap < SWAP_THRESHOLD:
        raise VariantInconclusive(
            f"the meminfo fixture computes swap_used_percent={fixture_swap}, below "
            f"the {SWAP_THRESHOLD}% rule"
        )
    fixture_memory = built.get("memory_used_percent")
    if isinstance(fixture_memory, (int, float)) and fixture_memory >= USED_THRESHOLD:
        raise VariantInconclusive(
            f"the node's real memory use is {fixture_memory:.1f}%: the swap rule "
            "cannot be isolated from the memory rule"
        )
    started_at = utc_now()
    opened = replay.open_bind(
        [(MEMINFO_FIXTURE, "/proc/meminfo")], restore_seconds=ctx.spec.max_minutes * 60
    )
    ctx.evidence.write("source-window-open", opened)
    detection = wait_for_incident(
        ctx,
        started_at,
        "swap_used_percent",
        timeout_seconds=SWAP_DETECTION_BUDGET,
        metrics=MEMORY_METRICS,
        name="swap-detection",
    )
    swap_samples = [
        item
        for item in samples(detection, "swap_used_percent")
        if isinstance(item.get("value"), (int, float))
    ]
    if not swap_samples:
        raise VariantInconclusive("no swap_used_percent sample reached the product")
    if not any(abs(item["value"] - fixture_swap) < 0.5 for item in swap_samples):
        raise VariantInconclusive(
            f"the product's swap_used_percent {[item['value'] for item in swap_samples]} "
            f"never matched the fixture's {fixture_swap:.1f}"
        )
    ctx.evidence.append_oracle(
        [{"fixture_swap_used_percent": fixture_swap, **replay.evidence()}],
        source="meminfo",
    )
    incidents = finding_incidents(detection, "swap_used_percent")
    observed = _await_validate_host(
        ctx, started_at, "swap_used_percent", name="swap-validate"
    )
    stages = {
        "chain": chain_errors(
            observed,
            ChainExpectation(
                metric="swap_used_percent",
                operations=("FREEZE_EVIDENCE", "VALIDATE_HOST"),
                validation="VALIDATE_HOST",
                validation_must_fail=True,
                probe_metric="swap_used_percent",
                incident_states=DIAGNOSTIC_ONLY_STATES,
            ),
        ),
        "no_memory_finding": chain_errors(
            observed, ChainExpectation(metric="memory_used_percent", present=False)
        ),
        "hardware": hardware_errors(observed),
    }
    stopped_at = utc_now()
    ctx.evidence.write("source-window-close", replay.close())
    recovered = await_metric_recovery(
        ctx, "swap_used_percent", SWAP_THRESHOLD, name="swap-recovery", since=stopped_at
    )
    stages["recovery"] = recovered["errors"]
    settled = wait_settled(
        ctx,
        started_at,
        [item["incident_id"] for item in incidents],
        timeout_seconds=SWAP_DETECTION_BUDGET,
        metrics=MEMORY_METRICS,
        name="swap-settled",
    )
    return _result(
        stages,
        notes=[
            "raw meminfo replay: only SwapTotal/SwapFree differ from the node's real "
            "reading; the swap rule fires on its own rule with no sustain"
        ],
        extra={
            "proof_scope": "live-replay",
            "fixture_swap_used_percent": fixture_swap,
            "swap_samples": swap_samples[-3:],
            "recovery": recovered,
            "settled": _statuses(settled),
        },
    )


# --------------------------------------------------------------------------- #
# shared helpers
# --------------------------------------------------------------------------- #
def _await_validate_host(
    ctx: VariantContext, since: datetime, metric: str, *, name: str
) -> dict[str, Any]:
    """The activity once ``metric``'s incident executed VALIDATE_HOST (or the budget)."""

    def validated() -> dict[str, Any] | None:
        current = activity(ctx, since, metrics={metric})
        for inc in finding_incidents(current, metric):
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

    return ctx.wait_until(
        validated, timeout_seconds=DETECTION_BUDGET, poll_seconds=20, name=name
    ) or activity(ctx, since, metrics={metric})


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
    VariantSpec(
        case_id=CASE,
        variant="swap_pressure",
        proof_mode=LIVE_REPLAY,
        lanes=("eks", "hp"),
        risk="L",
        flags="S",
        injection=(
            "bind a copy of the node's real /proc/meminfo with SwapTotal=8 GiB, "
            "SwapFree=1 GiB (87.5 % used) into the host collector unit's mount "
            "namespace; no swap device is created"
        ),
        expected=(
            "swap_used_percent >= 80 -> WARNING RUN_DIAGNOSTICS on its own rule (no "
            "sustain): FREEZE_EVIDENCE then VALIDATE_HOST whose swap probe fails while "
            "the bind is held; no memory finding; no reset/reboot; recovery once closed"
        ),
        cleanup="close the meminfo bind window (deadman-backed); 2 fresh samples below 80 %",
        max_minutes=20,
        approvals=(HOST_SOURCE_REPLAY,),
        execute=swap_pressure,
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
    VariantSpec(
        case_id=CASE,
        variant="read_failure",
        proof_mode=LIVE_REPLAY,
        lanes=("eks", "hp"),
        risk="L",
        flags="S",
        injection=(
            "bind an unopenable node (a driverless 0:0 character device) read-only "
            "over /proc/meminfo inside the host collector unit only -- not "
            "InaccessiblePaths=, whose mode-0000 node root reads as an empty file "
            "(verified through nsenter: open fails in the unit, PID 1 reads the "
            "real file)"
        ),
        expected=(
            "a _memory collection error and no memory sample in the window (UNKNOWN); "
            "no memory/swap finding, no hardware step, no isolation; samples resume "
            "after the window"
        ),
        cleanup="close the source window (deadman-backed); a fresh memory sample",
        max_minutes=14,
        approvals=(HOST_SOURCE_REPLAY,),
        execute=read_failure,
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

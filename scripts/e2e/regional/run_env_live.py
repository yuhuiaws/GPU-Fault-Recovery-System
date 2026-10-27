#!/usr/bin/env python3
"""GF-ENV-LIVE-001..012: live GPU environment acceptance via the replay proxy.

Plan-only by default. ``--plan`` reads the target node and its live device
limits read-only, decides which per-device variants can run on this hardware
(``env_live_plans``: a p5en H200 publishes only the T.Limit scale, so the
absolute-core and absolute-shutdown variants are NOT_RUN there), builds the
reviewed replay plans and writes ``plan.json``. ``--execute`` additionally needs
``--confirm ENVLIVE<NNN>_EXECUTE`` and, for the branches that isolate or stop a
workload, ``--confirm-isolate``/``--confirm-stop`` tokens; it ships the audited
replay tool to the node, repoints the single metrics collector at the loopback
proxy through a bounded reversible maintenance change with an independent
rollback, arms one plan, and then waits -- bounded by the product's own windows
and the proxy's replay window -- until the proxy served the plan and the
product's records show the case's expected reaction (``env_live_watch``,
``env_live_judges``), and restores everything. The tool never declares a LIVE
PASS; the verdict comes from those records: PASS only on positive evidence,
FAIL on a missing or wrong reaction, INCONCLUSIVE when the replay or its
ingestion cannot be shown. Nothing is deployed while the probe's preflight
shows a leftover of an earlier run (an open replay window, a repoint drop-in,
a transient unit, a collector reading a foreign endpoint), nothing is armed
without the repoint proof (``env_live_fixture.repoint_proof``, re-checked
against the proxy status right before the arm), and an armed proxy that counts
no read by the time the first one is due has the repoint re-verified on the
node and the reason named (``env_live_watch.diagnose_zero_reads``).

GF-ENV-LIVE-009 pauses the metrics collector read-synchronously after two
severe samples (``probes/env_replay_proxy.py pause-collector``, a deadman
resumes it), resumes it past the policy's 60 s confirmation gap, holds it a
second time right after the first post-gap read until the judge has read the
product's regraded finding state (the *recount hold*: that state lives one
collector interval, far shorter than one poll), and restarts it once at a
normal temperature (``env_live_interruption``). GF-ENV-LIVE-010
runs the 004 chain on one GPU of the target node while an independent control
task runs on the control node (``--node-b``; ``env_live_scope``); without a
control node it is NOT_RUN. Both run on plain EKS and HyperPod: a host-mode
node without an exporter Pod has its device limits read by a transient host
probe. GF-ENV-LIVE-011 (absolute shutdown) runs only where the device reports
an absolute ``gpu_shutdown_temperature_c``; GF-ENV-LIVE-012 (real physical /
vendor-lab fault) is always NOT_RUN here. The user runs ``--execute`` live later.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from collections.abc import Callable
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.e2e.regional import env_live_expectations as expectations  # noqa: E402
from scripts.e2e.regional import env_live_fixture as fixture  # noqa: E402
from scripts.e2e.regional import env_live_interruption as interruption  # noqa: E402
from scripts.e2e.regional import env_live_judges as judges  # noqa: E402
from scripts.e2e.regional import env_live_plans as plans  # noqa: E402
from scripts.e2e.regional import env_live_scope as scope  # noqa: E402
from scripts.e2e.regional import env_live_verdicts as verdicts  # noqa: E402
from scripts.e2e.regional import env_live_watch as watch  # noqa: E402
from scripts.e2e.regional.acceptance_runner_common import (  # noqa: E402
    write_json_atomic,
)
from scripts.e2e.regional.collector_action_guard import action_window  # noqa: E402
from scripts.e2e.regional.live_driver_guard import (  # noqa: E402
    CaseSurface,
    run_selected_case,
)
from scripts.e2e.regional.regional_commands import RegionalFixtureError  # noqa: E402
from scripts.e2e.regional.regional_live_fixture import (  # noqa: E402
    RegionalLiveFixture,
    run_case_main,
)

CASE_IDS = tuple(f"GF-ENV-LIVE-{number:03d}" for number in range(1, 13))
#: 012 is a real physical / vendor-lab fault: never a telemetry-replay run.
NOT_RUN_CASES = frozenset({"GF-ENV-LIVE-012"})
CASE_PATTERN = re.compile(r"^GF-ENV-LIVE-(\d{3})$")
#: Cases whose live path isolates the node and/or stops the owned workload, so
#: they demand the extra branch confirmations on top of ENVLIVE<NNN>_EXECUTE.
ISOLATING_CASES = frozenset(
    {"GF-ENV-LIVE-004", "GF-ENV-LIVE-007", "GF-ENV-LIVE-010", "GF-ENV-LIVE-011"}
)
PORT = 9411
PROXY_ENDPOINT = f"http://127.0.0.1:{PORT}/metrics"
WINDOW_SECONDS = 600
#: The store reads start this much before the runner's arm: batches and
#: findings carry the node collector's clock, incidents the control plane's.
STATE_SKEW_SECONDS = 30
ACTIVITY_SKEW_SECONDS = 5
#: The poll loop's sleep (a test drives it with a fake clock).
SLEEP = time.sleep
#: GF-ENV-LIVE-009: the on-node pause waiter gives up this long after the
#: pause read was due (collector ticks after the arm plus a margin).
PAUSE_WAIT_EXTRA_READS = 2
PAUSE_WAIT_MARGIN_SECONDS = 60


def window_seconds(expectation: expectations.Expectation) -> int:
    """The proxy's replay window and the collector deadman for one case.

    ``WINDOW_SECONDS`` plus the 009 recount hold: that hold stops the collector
    a second time inside the window, and the window's terms grow by the same
    amount (``env_live_expectations.recount_hold_seconds``).
    """

    return WINDOW_SECONDS + int(math.ceil(expectation.recount_hold_seconds))


#: GF-ENV-LIVE-010 is NOT_RUN without these.
SCOPE_NEEDS = (
    "GF-ENV-LIVE-010 needs a second GPU test node (--node-b, the site profile's "
    "node-b) for the independent control task, --site-file to submit it, and a "
    "target node with at least two GPUs"
)


class PauseNotTaken(RegionalFixtureError):
    """GF-ENV-LIVE-009's collection pause could not be scheduled: the case is
    INCONCLUSIVE (nothing was shown) and the cleanup restores the endpoint
    before the severe block can be served."""


def case_number(case_id: str) -> int:
    match = CASE_PATTERN.fullmatch(case_id)
    if match is None:
        raise RegionalFixtureError(f"not a GF-ENV-LIVE case id: {case_id}")
    return int(match.group(1))


def confirmation_for(case_id: str) -> str:
    return f"ENVLIVE{case_number(case_id):03d}_EXECUTE"


def isolate_token(case_id: str) -> str:
    return f"ENVLIVE{case_number(case_id):03d}_ISOLATE"


def stop_token(case_id: str) -> str:
    return f"ENVLIVE{case_number(case_id):03d}_STOP"


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        description="Run one guarded GF-ENV-LIVE acceptance case."
    )
    fixture.add_env_live_arguments(value, confirmation="ENVLIVE_CASE_SPECIFIC")
    return value


def configure(arguments: argparse.Namespace) -> fixture.EnvLiveSettings:
    if arguments.case not in CASE_IDS:
        raise RegionalFixtureError(
            f"--case must be one of GF-ENV-LIVE-001..012, not {arguments.case}"
        )
    return fixture.configure(arguments, confirmation_for(arguments.case))


def read_only_preflight(
    settings: fixture.EnvLiveSettings, case_dir: Path
) -> dict[str, Any]:
    preflight = fixture.read_only_preflight(settings, case_dir)
    if settings.case_id in NOT_RUN_CASES:
        # A NOT_RUN case still writes a clean, explainable plan; its "error" is
        # only that it is deliberately not executable here.
        preflight = {**preflight, "errors": list(preflight["errors"])}
    return preflight


# --------------------------------------------------------------------------- #
# Plan details
# --------------------------------------------------------------------------- #


def _target_plan(preflight: dict[str, Any]) -> dict[str, Any] | None:
    target = preflight.get("target_gpu") or {}
    uuid = target.get("gpu_uuid")
    reports = preflight.get("device_plans") or {}
    return reports.get(uuid) if uuid else None


def scope_not_run_reason(
    settings: fixture.EnvLiveSettings, preflight: dict[str, Any]
) -> str | None:
    """Why GF-ENV-LIVE-010 cannot run here, or ``None`` when it can."""

    missing: list[str] = []
    if not settings.control_node:
        missing.append("no control node (--node-b)")
    if settings.site_file is None:
        missing.append("no --site-file for the control task")
    if len(preflight.get("devices") or []) < 2:
        missing.append("the target node reports fewer than two GPUs")
    if not missing:
        return None
    return f"{SCOPE_NEEDS}; here: {', '.join(missing)}"


def replay_plans_for_case(case_id: str, preflight: dict[str, Any]) -> dict[str, Any]:
    """Build the concrete replay plans this case would arm, from live limits.

    Each plan is one ``{"steps": [...]}`` object the tool validates; a device
    that lacks a scale yields a NOT_RUN note instead of a fabricated plan.
    """

    target = preflight.get("target_gpu") or {}
    uuid = target.get("gpu_uuid")
    limits = plans.normalize_device_limits(target.get("limits") or {})
    from gpu_fault.gpu_metrics import GpuMetricsThresholds

    thresholds = GpuMetricsThresholds.from_environment()
    samples = int(preflight.get("confirmation_samples") or 3)
    if not uuid:
        return {"error": "no target GPU UUID", "plans": {}, "not_run": {}}
    built = plans.build_device_plan(
        gpu_uuid=uuid,
        gpu_index=target.get("gpu_index"),
        product=target.get("product"),
        raw_limits=target.get("limits") or {},
        thresholds=thresholds,
        confirmation_samples=samples,
    )
    runnable: dict[str, Any] = {}
    not_run: dict[str, str] = {}
    number = case_number(case_id)
    try:
        if number in {3}:
            for variant, decision in built.decisions.items():
                if decision.runnable:
                    runnable[f"{variant}/warning"] = plans.warning_plan(
                        decision, confirmation_samples=samples
                    )
                else:
                    not_run[variant] = decision.reason
        elif number in {4, 10}:
            # 010 replays the same severe plan on one GPU of a multi-GPU node;
            # the control node, task and scope checks are the runner's.
            for variant, decision in built.decisions.items():
                if decision.runnable:
                    runnable[f"{variant}/severe"] = plans.confirmation_plan(
                        decision, confirmation_samples=samples
                    )
                else:
                    not_run[variant] = decision.reason
        elif number == 5:
            decision = built.decisions[plans.CORE_TLIMIT]
            if decision.runnable:
                runnable.update(
                    {
                        f"transient/{name}": plan
                        for name, plan in plans.transient_plans(
                            decision, confirmation_samples=samples
                        ).items()
                    }
                )
            else:
                not_run[plans.CORE_TLIMIT] = decision.reason
                for variant, other in built.decisions.items():
                    if other.runnable:
                        runnable.update(
                            {
                                f"transient/{name}": plan
                                for name, plan in plans.transient_plans(
                                    other, confirmation_samples=samples
                                ).items()
                            }
                        )
                        break
        elif number == 9:
            # Interruption / source switch: two severe reads, the read-
            # synchronous collection pause (>60 s), one more severe read that
            # must count 1/N again, a clear; then one collector restart.
            decision = built.decisions[plans.CORE_TLIMIT]
            if decision.runnable:
                runnable["core_tlimit/interruption"] = plans.interruption_plan(
                    decision, confirmation_samples=samples
                )
            else:
                not_run[plans.CORE_TLIMIT] = decision.reason
        elif number == 6:
            # One rising read per confirmation sample, the tail at the confirmed
            # read's duration: the cleared read the edge filter delivers next is
            # a zero delta (live 2026-10-27, HyperPod: a rise on it re-minted
            # the WARNING and VALIDATE_GPU failed at the grace).
            runnable.update(plans.thermal_duration_plan(confirmation_samples=samples))
        elif number == 7:
            runnable.update(plans.power_brake_plans(confirmation_samples=samples))
        elif number == 8:
            decision = next(
                (d for d in built.decisions.values() if d.runnable),
                built.decisions[plans.CORE_TLIMIT],
            )
            runnable.update(plans.unavailable_field_plans(decision))
        elif number == 11:
            try:
                runnable["shutdown"] = plans.shutdown_plan(limits)
            except plans.EnvLivePlanError as exc:
                not_run["shutdown_absolute"] = str(exc)
    except plans.EnvLivePlanError as exc:
        return {"error": str(exc), "plans": {}, "not_run": not_run}
    return {"plans": runnable, "not_run": not_run}


def plan_details(
    settings: fixture.EnvLiveSettings, preflight: dict[str, Any]
) -> dict[str, Any]:
    case_id = settings.case_id
    number = case_number(case_id)
    not_run = case_id in NOT_RUN_CASES
    replay = {} if not_run else replay_plans_for_case(case_id, preflight)
    observation = observation_plan(case_id, preflight, replay.get("plans") or {})
    # 009's window carries its recount hold (the proxy window and the collector
    # deadman grow by the hold's bound, at the default cadence here).
    window = WINDOW_SECONDS + int(
        math.ceil(float((observation or {}).get("recount_hold_seconds") or 0.0))
    )
    details: dict[str, Any] = {
        "case_id": case_id,
        "risk": "live-non-destructive"
        if number in {1, 3, 6}
        else (
            "not-applicable"
            if not_run
            else (
                "live-collector-maintenance" if number == 9 else "live-workload-restart"
            )
        ),
        "target_node": settings.node,
        "target_gpu": (preflight.get("target_gpu") or {}).get("gpu_uuid"),
        "device_limits_source": preflight.get("device_limits_source"),
        "predecessor": preflight["predecessor"],
        "device_plans": preflight.get("device_plans"),
        "proxy_mechanism": {
            "tool": "tools/gpu_environment_replay.py",
            "tool_sha256": fixture.replay_tool_sha256(),
            "controller": "scripts/e2e/regional/probes/env_replay_proxy.py",
            "repoint": (
                "systemd drop-in on gpu-fault-metrics-collector.service only, "
                "GPU_FAULT_DCGM_METRICS_URL -> http://127.0.0.1:9411/metrics; the "
                "real exporter and collector.env are untouched; one producer"
            ),
            "device_limits": (
                "nvidia-smi -q -x through the product parser: in the node's "
                "exporter Pod where the exporter is a DaemonSet, else (host-mode "
                "exporter, plain EKS) by a transient host probe that is deleted again"
            ),
            "max_window_seconds": window,
            "preflight_leftovers": (
                "the probe preflight lists open replay windows, repoint drop-ins, "
                "gpu-fault-env-* transient units and the endpoint the collector "
                "process reads; any leftover of an earlier run is INCONCLUSIVE "
                "with the reason and deploys nothing"
            ),
            "repoint_proof": (
                "arm only after the repoint receipt proves the restarted collector "
                "process reads the proxy and made its first pass-through read, "
                "and the status right before the arm still shows that endpoint, "
                "an unarmed proxy and a read count at or past the proof"
            ),
            "zero_read_reverification": (
                "an armed proxy that counts no read by one jittered interval plus "
                "the collector's start-up spread and jitter after the arm has the "
                "unit read back and the reason named: unit restarted or inactive, "
                "drop-in gone, endpoint reverted, proxy refused the reads (503), or "
                "collector silent"
            ),
        },
        "replay_plans": replay.get("plans", {}),
        "not_run_variants": replay.get("not_run", {}),
        "confirmation": settings.confirmation,
        "branch_confirmations": (
            {"isolate": isolate_token(case_id), "stop": stop_token(case_id)}
            if case_id in ISOLATING_CASES
            else {}
        ),
        "stop_conditions": [
            "predecessor evidence is not PASS",
            "node not clean (workload, ownership, non-ACTIVE Agent) or <3 fresh samples",
            "a previous replay run's window, drop-in, transient unit or endpoint "
            "override is still on the node (nothing deployed)",
            "the repoint is not proven before the arm (nothing armed)",
            "device does not report the scale a variant grades on (that variant NOT_RUN)",
            "real upstream HW thermal/Power Brake, T.Limit<=0, GPU>=80C or memory>=85C",
            "collector endpoint cannot be restored or a probe resource remains",
        ],
        "rollback": {
            "collector_repoint_deadman_seconds": window,
            "runner_restores_endpoint_before_stopping_proxy": True,
            "runner_finally_deletes_probe_resources": True,
            "replay_window_max_seconds": window,
            "job_restart_after_remediation": False,
        },
        "verdict_if_not_run": "NOT_RUN" if not_run else None,
        "observation": observation,
        "preflight": preflight,
    }
    if number == 9:
        details["interruption"] = interruption_details(replay.get("plans") or {})
    if number == 10:
        reason = scope_not_run_reason(settings, preflight)
        details["scope"] = {
            "control_node": settings.control_node or None,
            "control_task": (
                f"{scope.CONTROL_TASK_MANIFEST} pinned to the control node, GPU-active "
                f"at the acceptance GPU duty ({scope.CONTROL_TASK_GPU_DUTY_PERCENT} %, "
                "never idle: an idle enrolled job is a LOW_GPU/LOW_CPU_UTILIZATION "
                "fault to the product), submitted through gpu-training-submit and "
                "removed in cleanup"
            ),
            "locks": (
                "the two-node run holds the target and the control node (both "
                "EKS lane locks on plain EKS; flag X)"
            ),
            "not_run_reason": reason,
        }
        if reason is not None:
            details["verdict_if_not_run"] = "NOT_RUN"
            details["not_run_reason"] = reason
    if not_run:
        details["not_run_reason"] = (
            "GF-ENV-LIVE-012 is a real physical or vendor-lab fault; a telemetry "
            "replay cannot stand in for it (verdict stays NOT_RUN)"
        )
    return details


def interruption_details(
    replay_plans: dict[str, Any],
    cadence: watch.CollectorCadence | None = None,
) -> dict[str, Any]:
    """How GF-ENV-LIVE-009 pauses, holds, resumes and restarts the collector."""

    plan = replay_plans.get("core_tlimit/interruption")
    pause_seconds = (
        verdicts.CONFIRMATION_MAX_GAP_SECONDS + expectations.PAUSE_MARGIN_SECONDS
    )
    pause_after = plans.pause_read_index(plan) if plan else None
    recount_hold = expectations.recount_hold_seconds(
        cadence or watch.CollectorCadence()
    )
    return {
        "pause_after_reads": pause_after,
        "pause_seconds": pause_seconds,
        "pause_deadman_seconds": pause_seconds
        + expectations.PAUSE_DEADMAN_MARGIN_SECONDS,
        "recount_hold_label": interruption.RECOUNT_HOLD_LABEL,
        "recount_hold_after_reads": None if pause_after is None else pause_after + 1,
        "recount_hold_seconds": recount_hold,
        "recount_hold_deadman_seconds": recount_hold
        + expectations.PAUSE_DEADMAN_MARGIN_SECONDS,
        "mechanism": (
            "env_replay_proxy.py pause-collector: a transient waiter unit on the "
            "node polls the proxy /status and SIGSTOPs gpu-fault-metrics-collector"
            ".service once the pause read is served and its batch is posted (the "
            "same systemctl kill pause as general_live_node_probe unit-pause); a "
            "systemd-run deadman resumes it; the runner resumes it after the "
            "pause -- scheduling first, while the unit is still stopped, a second "
            "labelled pause (the recount hold) one read later, so the product's "
            "regraded 1/N finding state (a current-state row the clearing read "
            "replaces; no history, incident or marker records a count change) "
            "stays readable until the judge has read it -- resumes the hold the "
            "moment a finding on the post-gap sample is read, and restarts the "
            "collector once (restart-collector) after the clear at a normal "
            "temperature"
        ),
        "not_driven": [
            "DCGM -> nvidia-smi source switch (normal installation only)",
            "cross-boot re-confirmation (needs an approved node maintenance reboot)",
        ],
    }


def _first_plan(
    case_id: str, replay_plans: dict[str, Any]
) -> tuple[str, dict[str, Any]]:
    """The one plan an attempt arms (001 serves a pass-through plan only)."""

    if case_number(case_id) == 1:
        return "passthrough", {"steps": [{"reads": 1, "values": {}}]}
    name, plan = sorted(replay_plans.items())[0]
    return str(name), dict(plan)


def observation_plan(
    case_id: str, preflight: dict[str, Any], replay_plans: dict[str, Any]
) -> dict[str, Any] | None:
    """The wait/verdict rule the attempt's armed plan gets, at default cadence.

    ``--execute`` recomputes it from the node's own collector.env.
    """

    number = case_number(case_id)
    # 002 is BLOCKED and 012 NOT_RUN before anything is armed.
    if number in {2, 12} or (number != 1 and not replay_plans):
        return None
    name, plan = _first_plan(case_id, replay_plans)
    try:
        expectation = expectations.expectation_for(
            case_id=case_id,
            plan_name=name,
            plan=plan,
            preflight=preflight,
            cadence=watch.CollectorCadence(),
            node=str((preflight.get("node") or {}).get("name") or ""),
        )
    except ValueError as exc:
        return {"plan": name, "error": str(exc)}
    return {
        **expectation.evidence(),
        "terms": expectation.terms,
        "derived_seconds": sum(expectation.terms.values()),
        "capped_by": (
            "min(arm + proxy max_seconds, repoint + collector deadman, maintenance "
            f"window end) - {watch.CLEANUP_MARGIN_SECONDS:.0f}s cleanup margin"
        ),
    }


# --------------------------------------------------------------------------- #
# Execute
# --------------------------------------------------------------------------- #
def _require_branch_tokens(arguments: argparse.Namespace, case_id: str) -> None:
    if case_id not in ISOLATING_CASES:
        return
    if arguments.confirm_isolate != isolate_token(case_id):
        raise RegionalFixtureError(
            f"this case isolates the node; --confirm-isolate must be "
            f"{isolate_token(case_id)}"
        )
    if arguments.confirm_stop != stop_token(case_id):
        raise RegionalFixtureError(
            f"this case stops the owned workload; --confirm-stop must be "
            f"{stop_token(case_id)}"
        )


def execute_case(
    settings: fixture.EnvLiveSettings,
    run_dir: Path,
    attempt: int,
    maintenance_window_end: datetime,
) -> int:
    case_dir = run_dir / "cases" / settings.case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    preflight = read_only_preflight(settings, case_dir)
    regional = RegionalLiveFixture(settings.regional)
    result: dict[str, Any] = {
        "schema_version": 2,
        "report_type": "fault-acceptance",
        "proof_scope": "live-telemetry-replay",
        "case_id": settings.case_id,
        "attempt": attempt,
        "verdict": "FAIL",
        "started_at": fixture.utc_now().isoformat(),
        "target_node": settings.node,
        **regional.evidence_identity(),
    }
    number = case_number(settings.case_id)

    def finish_without_run(verdict: str, **extra: Any) -> int:
        result.update(verdict=verdict, ended_at=fixture.utc_now().isoformat(), **extra)
        write_json_atomic(case_dir / f"{settings.case_id}.json", result)
        print_result(result)
        return 0

    if settings.case_id in NOT_RUN_CASES:
        return finish_without_run(
            "NOT_RUN",
            not_run_reason=(
                "GF-ENV-LIVE-012 is a real physical / vendor-lab fault; GFRS does "
                "not manufacture it and a replay cannot certify it"
            ),
        )
    if preflight["errors"]:
        raise RegionalFixtureError(
            "preflight failed: " + "; ".join(preflight["errors"])
        )
    if number == 2:
        # Real power cap reuses the COLLECT-002 controlled-load + guarded power
        # restore path, not the telemetry proxy; it needs a DCGM load generator
        # and default power limits. This runner does not drive that mechanism.
        return finish_without_run(
            "BLOCKED",
            blocked_reason=(
                "GF-ENV-LIVE-002 reuses GF-REGIONAL-COLLECT-002's controlled power "
                "cap and dcgmproftester load with its deadman restore; run "
                "run_collector_acceptance.py --case GF-REGIONAL-COLLECT-002 on a "
                "node that ships dcgmproftester and has default power limits"
            ),
        )
    if number == 10:
        reason = scope_not_run_reason(settings, preflight)
        if reason is not None:
            return finish_without_run("NOT_RUN", not_run_reason=reason)
        result["control_node"] = settings.control_node
    replay = replay_plans_for_case(settings.case_id, preflight)
    arms_a_plan = number != 1
    if arms_a_plan and not replay.get("plans"):
        return finish_without_run(
            "NOT_RUN",
            not_run_variants=replay.get("not_run", {}),
            not_run_reason=(
                "this device reports no scale the case grades on; every variant "
                "is NOT_RUN here (e.g. no absolute shutdown threshold on H200)"
            ),
        )
    if settings.case_id in ISOLATING_CASES and (
        settings.isolate_confirm != isolate_token(settings.case_id)
        or settings.stop_confirm != stop_token(settings.case_id)
    ):
        raise RegionalFixtureError(
            "this case isolates the node and stops the owned workload; pass "
            f"--confirm-isolate {isolate_token(settings.case_id)} and "
            f"--confirm-stop {stop_token(settings.case_id)}"
        )
    run_id = f"{settings.case_id.lower()}-a{attempt}-{int(time.time())}"
    controller = fixture.EnvReplayController(
        regional,
        node=settings.node,
        image=settings.host_probe_image,
        case_id=settings.case_id,
        run_id=run_id,
        case_dir=case_dir,
    )
    stages: dict[str, list[str]] = {}
    inconclusive: list[str] = []
    receipts: dict[str, Any] = {}
    #: Fixtures beyond the controller the run creates (the 010 control task);
    #: removed after the controller's own cleanup, in reverse order.
    fixtures: list[tuple[str, Callable[[], Any]]] = []
    try:
        with action_window(maintenance_window_end):
            outcome = _run_replay(
                settings,
                regional,
                controller,
                preflight,
                replay,
                receipts=receipts,
                result=result,
                case_dir=case_dir,
                maintenance_window_end=maintenance_window_end,
                run_id=run_id,
                fixtures=fixtures,
            )
            receipts["observation"] = outcome["observation"]
            stages.update(outcome["stages"])
            inconclusive.extend(outcome["inconclusive"])
    except (PauseNotTaken, fixture.EnvLiveNotShown) as exc:
        # Nothing was shown (a leftover of an earlier run, an unproven repoint,
        # a pause not taken): INCONCLUSIVE with the reason, never a FAIL the
        # product did not earn.
        result["error"] = f"{type(exc).__name__}: {exc}"
        inconclusive.append(f"{settings.case_id}: {exc}")
    except BaseException as exc:  # noqa: BLE001 - recorded; cleanup still runs
        result["error"] = f"{type(exc).__name__}: {exc}"
        stages.setdefault("execution", []).append(result["error"])
        if not isinstance(exc, Exception):
            raise
    finally:
        try:
            receipts["cleanup"] = controller.cleanup()
        except Exception as exc:  # noqa: BLE001
            receipts["cleanup_error"] = f"{type(exc).__name__}: {exc}"
        for name, action in reversed(fixtures):
            try:
                receipts[f"{name}_cleanup"] = action()
            except Exception as exc:  # noqa: BLE001 - recorded; fails the case
                receipts[f"{name}_cleanup_error"] = f"{type(exc).__name__}: {exc}"
        stages["cleanup"] = _cleanup_errors(settings.case_id, receipts)
    status = receipts.get("passthrough_status") or {}
    if "passthrough_status" in receipts:
        stages["passthrough"] = (
            []
            if (status.get("status") or {}).get("armed") is False
            else ["proxy was armed before the baseline read"]
        )
    result.update(
        verdict=verdicts.case_verdict(stages, inconclusive),
        stages=stages,
        errors=[
            f"{stage}: {item}" for stage, items in stages.items() for item in items
        ],
        inconclusive=inconclusive,
        receipts=receipts,
        ended_at=fixture.utc_now().isoformat(),
        limitations=_limitations(receipts),
    )
    write_json_atomic(case_dir / f"{settings.case_id}.json", result)
    print_result(result)
    return 0 if result["verdict"] == "PASS" else 1


def _limitations(receipts: dict[str, Any]) -> list[str]:
    expectation = (receipts.get("observation") or {}).get("expectation") or {}
    return [
        "Telemetry replay only: it proves the normally-installed collector, "
        "real control plane and scheduling run end to end, not that the "
        "sensor truly reported the fault or that hardware was repaired.",
        "The runner arms one reviewed plan per attempt; the remaining plans "
        "in replay_plans are run in follow-up attempts with fresh audit "
        "files and a fresh clean test job.",
        "The wait is bounded by the proxy's replay window and the collector "
        "deadman: a reaction the product needs longer for is INCONCLUSIVE, "
        "never PASS.",
        *(
            f"Not covered by this attempt: {item}"
            for item in expectation.get("not_covered_by_this_attempt", [])
        ),
    ]


def _control_fixtures(
    settings: fixture.EnvLiveSettings,
    regional: Any,
    *,
    case_dir: Path,
    run_id: str,
    receipts: dict[str, Any],
    fixtures: list[tuple[str, Callable[[], Any]]],
) -> dict[str, Any]:
    """GF-ENV-LIVE-010: start the control task on the control node first.

    Its Pods and the control node's state before the arm are what the scope
    judge compares against; the task is removed after the controller cleanup.
    """

    assert settings.site_file is not None, "010 runs only with --site-file"
    task = scope.ControlTask(
        regional,
        site_file=settings.site_file,
        case_dir=case_dir,
        case_id=settings.case_id,
        run_id=run_id,
        node=settings.control_node,
    )
    fixtures.append(("control_task", task.cleanup))
    receipts["control_task_start"] = task.start()
    before = task.snapshot()
    receipts["control_task_before"] = before
    receipts["control_node_before"] = fixture.node_state(
        regional, settings.control_node
    )
    return {
        "task": task,
        "task_before": before,
        "control_before": receipts["control_node_before"],
    }


def _run_replay(
    settings: fixture.EnvLiveSettings,
    regional: Any,
    controller: Any,
    preflight: dict[str, Any],
    replay: dict[str, Any],
    *,
    receipts: dict[str, Any],
    result: dict[str, Any],
    case_dir: Path,
    maintenance_window_end: datetime,
    run_id: str = "",
    fixtures: list[tuple[str, Callable[[], Any]]] | None = None,
) -> dict[str, Any]:
    """Deploy, repoint, arm one plan and observe it; the caller cleans up."""

    number = case_number(settings.case_id)
    arms_a_plan = number != 1
    control: dict[str, Any] = {}
    if number == 10:
        control = _control_fixtures(
            settings,
            regional,
            case_dir=case_dir,
            run_id=run_id,
            receipts=receipts,
            fixtures=fixtures if fixtures is not None else [],
        )
    controller.create()
    # 001 never arms (pass-through baseline). Every other case arms the first
    # reviewed plan; the rest are documented in replay_plans and run in
    # follow-up attempts with their own audit files.
    plan_name, first_plan = _first_plan(settings.case_id, replay.get("plans") or {})
    plan_path = fixture.write_plan_file(case_dir, fixture.PLAN_FILE, first_plan)
    receipts["probe_preflight"] = controller.preflight()
    # A leftover of an earlier run (its window, drop-in, deadman or waiter, or
    # a collector still reading another endpoint) would be read as this run's
    # doing: name it and deploy nothing.
    leftovers = fixture.node_leftovers(receipts["probe_preflight"], proxy_port=PORT)
    receipts["node_leftovers"] = leftovers
    if leftovers:
        raise fixture.NodeNotClean(
            "the node carries leftovers of a previous replay run; nothing was "
            "deployed: " + "; ".join(leftovers)
        )
    cadence = watch.CollectorCadence.from_probe(receipts["probe_preflight"])
    expectation = expectations.expectation_for(
        case_id=settings.case_id,
        plan_name=plan_name,
        plan=first_plan,
        preflight=preflight,
        cadence=cadence,
        node=settings.node,
    )
    window = window_seconds(expectation)
    receipts["deploy"] = controller.deploy_proxy(
        plan_path,
        str(preflight["target_gpu"]["gpu_uuid"]),
        port=PORT,
        max_seconds=window,
    )
    # What the real exporter delivered before the repoint, and the node as the
    # run found it: the baseline's unchanged-value check and the "isolated by
    # this run" checks compare against these.
    before = None
    if not arms_a_plan:
        try:
            before = fixture.gpu_records(
                regional, settings.node, fixture.utc_now(), expectation.names
            )
        except Exception as exc:  # noqa: BLE001 - the check is skipped, not faked
            receipts["before_error"] = f"{type(exc).__name__}: {exc}"
    receipts["node_before"] = regional.node_snapshot(settings.node)
    repoint_at = fixture.utc_now()
    receipts["repoint"] = controller.repoint(port=PORT, restore_seconds=window)
    # The receipt's own proof, recorded: the restarted process reads the proxy
    # and the proxy counted its first pass-through read (the controller refused
    # anything less).
    receipts["repoint_proof"] = fixture.repoint_proof(
        receipts["repoint"], PROXY_ENDPOINT
    )
    # A few pass-through cycles must reach the control plane unchanged before
    # arming (001's contract, required by every case).
    passthrough_at = fixture.utc_now()
    receipts["passthrough_status"] = controller.status(port=PORT)
    progress = watch.ReplayProgress(
        planned_reads=expectation.planned_reads,
        armed=arms_a_plan,
        proxy_endpoint=PROXY_ENDPOINT,
        first_read_due_seconds=cadence.first_read_due_seconds,
        repoint_baseline=watch.repoint_baseline(receipts["repoint"]),
    )
    if arms_a_plan:
        # Right before the arm: the collector still reads the proxy, which is
        # still pass-through and still counting from the proof.
        receipts["arm_precheck"] = fixture.require_proxy_read_before_arm(
            receipts["passthrough_status"],
            receipts["repoint_proof"],
            proxy_endpoint=PROXY_ENDPOINT,
        )
        start = fixture.utc_now()
        receipts["arm"] = controller.arm(port=PORT)
        progress.begin(receipts["arm"], start)
    else:
        start = repoint_at
        progress.begin(receipts["passthrough_status"], passthrough_at)
    result["armed_plan"] = plan_name
    if number == 9:
        receipts["pause_schedule"] = _schedule_pause(
            controller, expectation, progress, cadence
        )
    judge, extra = _judge_and_extra(
        settings,
        regional,
        controller,
        expectation,
        cadence=cadence,
        repoint_at=repoint_at,
        before=before,
        node_before=receipts["node_before"],
        control=control,
        start=start,
        progress=progress,
    )
    return _observe(
        settings,
        regional,
        controller,
        expectation,
        progress,
        judge=judge,
        extra=extra,
        cadence=cadence,
        start=start,
        repoint_at=repoint_at,
        maintenance_window_end=maintenance_window_end,
        case_dir=case_dir,
        window_seconds=window,
    )


def _schedule_pause(
    controller: Any,
    expectation: expectations.Expectation,
    progress: watch.ReplayProgress,
    cadence: watch.CollectorCadence,
) -> dict[str, Any]:
    """GF-ENV-LIVE-009: schedule the read-synchronous pause right after the arm.

    The waiter counts the proxy's absolute reads, so the target is the arm's
    read count plus the plan's pause read; it is scheduled before the first
    severe read can be served, and a schedule that does not come back with an
    active waiter refuses the case (the caller's cleanup restores the endpoint
    before the severe block starts).
    """

    if progress.start_reads is None:
        raise PauseNotTaken(
            "the arm receipt carried no proxy read count; the pause point is unknown"
        )
    target = int(progress.start_reads) + int(expectation.pause_after_reads)
    wait = int(
        (expectation.pause_after_reads + PAUSE_WAIT_EXTRA_READS)
        * cadence.interval_seconds
        + watch.STARTUP_SPREAD_SECONDS
        + PAUSE_WAIT_MARGIN_SECONDS
    )
    receipt = controller.pause_collector(
        after_reads=target,
        seconds=int(
            expectation.pause_seconds + expectations.PAUSE_DEADMAN_MARGIN_SECONDS
        ),
        wait_seconds=wait,
        port=PORT,
    )
    state = str((receipt.get("waiter") or {}).get("ActiveState") or "")
    if state not in {"active", "activating"}:
        raise PauseNotTaken(
            f"the collection pause waiter is not running ({state or 'unknown'}); "
            "refusing to serve the severe block without the pause"
        )
    return receipt


def _schedule_hold(
    controller: Any,
    expectation: expectations.Expectation,
    progress: watch.ReplayProgress,
    cadence: watch.CollectorCadence,
) -> dict[str, Any]:
    """GF-ENV-LIVE-009: schedule the recount hold right before the gap resume.

    A second labelled pause one read past the gap pause's target: the waiter
    stops the collector again as soon as it has posted the first post-gap read,
    so the product's regraded finding state -- a current-state row the next
    accepted reading overwrites, with no durable record of its own -- stays
    readable until the judge has read it (bounded by
    ``recount_hold_seconds``; the deadman adds its margin). The judge calls
    this while the collector is still stopped by the gap pause, so no read can
    slip in before the waiter runs; a schedule that comes back without a
    running waiter is refused here and recorded by the judge, and the resume
    goes ahead regardless.
    """

    if progress.start_reads is None:
        raise PauseNotTaken(
            "the arm receipt carried no proxy read count; the hold point is unknown"
        )
    target = int(progress.start_reads) + int(expectation.pause_after_reads) + 1
    # The post-gap read follows the resume within one jittered tick (the
    # collector's sleep expired during the gap); the resume exec itself may
    # take up to its own timeout.
    wait = int(
        PAUSE_WAIT_EXTRA_READS * cadence.max_interval_seconds
        + PAUSE_WAIT_MARGIN_SECONDS
    )
    receipt = controller.pause_collector(
        after_reads=target,
        seconds=int(
            expectation.recount_hold_seconds + expectations.PAUSE_DEADMAN_MARGIN_SECONDS
        ),
        wait_seconds=wait,
        port=PORT,
        label=interruption.RECOUNT_HOLD_LABEL,
    )
    state = str((receipt.get("waiter") or {}).get("ActiveState") or "")
    if state not in {"active", "activating"}:
        raise PauseNotTaken(
            f"the recount hold waiter is not running ({state or 'unknown'})"
        )
    return receipt


def _judge_and_extra(
    settings: fixture.EnvLiveSettings,
    regional: Any,
    controller: Any,
    expectation: expectations.Expectation,
    *,
    cadence: watch.CollectorCadence,
    repoint_at: datetime,
    before: dict[str, Any] | None,
    node_before: dict[str, Any] | None,
    control: dict[str, Any],
    start: datetime,
    progress: watch.ReplayProgress | None = None,
) -> tuple[Any, Callable[[], dict[str, Any]] | None]:
    """The case's judge and, for 009 / 010, the poll's fifth read."""

    if expectation.kind == "interruption":
        steps = interruption.InterruptionSteps(
            resume=controller.resume_collector,
            restart=controller.restart_collector,
        )
        if progress is not None:
            steps.hold = lambda: _schedule_hold(
                controller, expectation, progress, cadence
            )
            steps.resume_hold = lambda: controller.resume_collector(
                label=interruption.RECOUNT_HOLD_LABEL
            )
        judge = interruption.InterruptionJudge(
            expectation,
            steps=steps,
            interval_seconds=cadence.interval_seconds,
            node_before=node_before,
        )
        return judge, controller.pause_status
    if control:
        task = control["task"]
        control_node = settings.control_node
        since = start - timedelta(seconds=ACTIVITY_SKEW_SECONDS)

        def control_read() -> dict[str, Any]:
            return {
                "node": fixture.node_state(regional, control_node),
                "activity": fixture.node_records(regional, control_node, since),
                "task": task.snapshot(),
            }

        judge = scope.ScopeJudge(
            expectation,
            task_before=control["task_before"],
            control_before=control["control_before"],
            node_before=node_before,
        )
        return judge, control_read
    return (
        judges.judge_for(
            expectation,
            repoint_at=repoint_at,
            before=before,
            collector_mode=cadence.metrics_mode,
            node_before=node_before,
        ),
        None,
    )


def _observe(
    settings: fixture.EnvLiveSettings,
    regional: Any,
    controller: Any,
    expectation: expectations.Expectation,
    progress: watch.ReplayProgress,
    *,
    judge: Any,
    extra: Callable[[], dict[str, Any]] | None,
    cadence: watch.CollectorCadence,
    start: datetime,
    repoint_at: datetime,
    maintenance_window_end: datetime,
    case_dir: Path,
    window_seconds: int = WINDOW_SECONDS,
) -> dict[str, Any]:
    """Poll the proxy, the store and the node until the case is decided.

    The deadline is the expectation's product-derived terms from ``start``
    (arm, or the repoint for the pass-through baseline), never past the
    proxy's replay window, the collector deadman or the maintenance window
    (``window_seconds``: the proxy's ``max_seconds`` and the deadman's
    ``restore_seconds`` as deployed for this case).
    """

    window = watch.observation_window(
        started_at=start,
        terms=expectation.terms,
        replay_end=(
            start + timedelta(seconds=window_seconds) if progress.armed else None
        ),
        deadman_at=repoint_at + timedelta(seconds=window_seconds),
        maintenance_end=maintenance_window_end,
    )
    state_since = start - timedelta(seconds=STATE_SKEW_SECONDS)
    activity_since = start - timedelta(seconds=ACTIVITY_SKEW_SECONDS)
    reads = watch.WatchReads(
        status=lambda: controller.status(port=PORT),
        gpu_state=lambda: fixture.gpu_records(
            regional, settings.node, state_since, expectation.names
        ),
        activity=lambda: fixture.node_records(regional, settings.node, activity_since),
        node=lambda: fixture.node_state(regional, settings.node),
        extra=extra,
        # The unit read back when an armed proxy counts no read in time.
        repoint=getattr(controller, "snapshot", None),
    )
    outcome = watch.watch(
        judge,
        progress,
        reads,
        window,
        poll_seconds=cadence.poll_seconds,
        clock=fixture.utc_now,
        sleep=SLEEP,
    )
    final = outcome.final_snapshot
    if final is not None:
        write_json_atomic(
            case_dir / "observation-final.json",
            {
                "at": final.at.isoformat(),
                "gpu_state": final.gpu_state,
                "activity": final.activity,
                "node": final.node,
                "extra": final.extra,
            },
        )
    judgement = outcome.judgement
    return {
        "stages": judgement.stages,
        "inconclusive": judgement.inconclusive,
        "observation": {
            "outcome": judgement.outcome,
            "cadence": cadence.evidence(),
            "window": window.evidence(),
            "polls": outcome.polls,
            "ended_at": outcome.ended_at.isoformat(),
            "read_errors": outcome.read_errors,
            "timeline": outcome.timeline,
            **judgement.evidence,
        },
    }


def _cleanup_errors(case_id: str, receipts: dict[str, Any]) -> list[str]:
    cleanup = receipts.get("cleanup") or {}
    restore = cleanup.get("restore") or {}
    residuals = cleanup.get("probe_residuals") or {}
    # ``restored_endpoint`` is the unit's endpoint override after the restore:
    # None means no override is left, i.e. the collector runs with
    # collector.env's own endpoint again -- the state the repoint started from
    # (its ``before`` carries no override either). The probe refuses any other
    # override, so only a missing original, an inactive unit or a foreign
    # override is a failed restore.
    endpoint = restore.get("restored_endpoint")
    # A run that never repointed the collector (refused on a leftover before
    # the deploy, or failed before the repoint call) has nothing to restore:
    # the collector endpoint is untouched, and the proxy stop needs no
    # restore before it.
    repointed = (
        "repoint" in receipts or "restore" in cleanup or "restore_error" in cleanup
    )
    errors = verdicts.cleanup_errors(
        collector_endpoint_restored=(
            not repointed
            or (
                bool(restore.get("original_endpoint"))
                and (restore.get("after") or {}).get("ActiveState") == "active"
                and endpoint in (None, restore.get("original_endpoint"))
            )
        ),
        single_producer_after=True,
        fresh_batches_after=3,
        proxy_closed_after_endpoint="stop_proxy" in cleanup
        and (not repointed or "restore" in cleanup),
        probe_residuals_absent=bool(residuals)
        and all(value is False for value in residuals.values()),
        label=case_id,
    )
    # A paused collector that could not be resumed (the gap pause or the 009
    # recount hold), or a control task that could not be removed, is a failed
    # cleanup like a leftover probe.
    for key in ("resume", f"resume_{interruption.RECOUNT_HOLD_LABEL}"):
        if cleanup.get(f"{key}_error"):
            errors.append(
                f"{case_id}: the paused collector was not resumed ({key}): "
                f"{cleanup[f'{key}_error']}"
            )
        if (cleanup.get(key) or {}).get("stopped") is True:
            errors.append(
                f"{case_id}: the metrics collector is still stopped after cleanup "
                f"({key})"
            )
    for key, value in sorted(receipts.items()):
        if key.endswith("_cleanup_error"):
            errors.append(
                f"{case_id}: {key.removesuffix('_cleanup_error')} cleanup: {value}"
            )
    return errors


def print_result(result: dict[str, Any]) -> None:
    print(json.dumps(result, sort_keys=True, default=str))


CASE = CaseSurface(
    parser=parser,
    configure=configure,
    read_only_preflight=read_only_preflight,
    plan_details=plan_details,
    execute_case=execute_case,
)


def main() -> int:
    return run_selected_case(CASE)


if __name__ == "__main__":
    raise SystemExit(run_case_main(main))

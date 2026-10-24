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
ingestion cannot be shown.

GF-ENV-LIVE-011 (absolute shutdown) runs only where the device reports an
absolute ``gpu_shutdown_temperature_c``; GF-ENV-LIVE-012 (real physical/vendor
-lab fault) is always NOT_RUN here. The user runs ``--execute`` live later.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.e2e.regional import env_live_expectations as expectations  # noqa: E402
from scripts.e2e.regional import env_live_fixture as fixture  # noqa: E402
from scripts.e2e.regional import env_live_judges as judges  # noqa: E402
from scripts.e2e.regional import env_live_plans as plans  # noqa: E402
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
INTERRUPTION_BLOCKED = (
    "GF-ENV-LIVE-009 needs a controlled >60 s pause of the metrics collector "
    "after two severe samples, a collector restart and a source switch through "
    "the normal installation; this runner has no such maintenance step, and "
    "arming its severe T.Limit plan without the pause would only repeat "
    "GF-ENV-LIVE-004's isolate/stop chain without the branch confirmations"
)


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
        elif number in {4}:
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
            # Interruption / source switch arms a severe T.Limit run; the >60s
            # collection gap, the collector restart and the source switch are
            # driven by maintenance of the one metrics unit.
            decision = built.decisions[plans.CORE_TLIMIT]
            if decision.runnable:
                runnable["core_tlimit/severe"] = plans.confirmation_plan(
                    decision, confirmation_samples=samples
                )
            else:
                not_run[plans.CORE_TLIMIT] = decision.reason
        elif number == 6:
            runnable.update(plans.thermal_duration_plan())
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
    details: dict[str, Any] = {
        "case_id": case_id,
        "risk": "live-non-destructive"
        if number in {1, 3, 6}
        else ("not-applicable" if not_run else "live-workload-restart"),
        "target_node": settings.node,
        "target_gpu": (preflight.get("target_gpu") or {}).get("gpu_uuid"),
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
            "max_window_seconds": WINDOW_SECONDS,
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
            "device does not report the scale a variant grades on (that variant NOT_RUN)",
            "real upstream HW thermal/Power Brake, T.Limit<=0, GPU>=80C or memory>=85C",
            "collector endpoint cannot be restored or a probe resource remains",
        ],
        "rollback": {
            "collector_repoint_deadman_seconds": WINDOW_SECONDS,
            "runner_restores_endpoint_before_stopping_proxy": True,
            "runner_finally_deletes_probe_resources": True,
            "replay_window_max_seconds": WINDOW_SECONDS,
            "job_restart_after_remediation": False,
        },
        "verdict_if_not_run": "NOT_RUN" if not_run else None,
        "observation": observation_plan(case_id, preflight, replay.get("plans") or {}),
        "preflight": preflight,
    }
    if number == 9:
        details["verdict_if_blocked"] = "BLOCKED"
        details["blocked_reason"] = INTERRUPTION_BLOCKED
    if not_run:
        details["not_run_reason"] = (
            "GF-ENV-LIVE-012 is a real physical or vendor-lab fault; a telemetry "
            "replay cannot stand in for it (verdict stays NOT_RUN)"
        )
    return details


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
    # 002/009 are BLOCKED and 010/012 NOT_RUN before anything is armed.
    if number in {2, 9, 10, 12} or (number != 1 and not replay_plans):
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
    if number == 9:
        return finish_without_run("BLOCKED", blocked_reason=INTERRUPTION_BLOCKED)
    if number == 10 and not settings.site_file:
        # Scope/concurrency needs a second multi-GPU test node and a control
        # task; without them the spec keeps the variant NOT_RUN.
        return finish_without_run(
            "NOT_RUN",
            not_run_reason=(
                "GF-ENV-LIVE-010 needs a second multi-GPU test node and an "
                "independent control task; neither is configured for this run"
            ),
        )
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
            )
            receipts["observation"] = outcome["observation"]
            stages.update(outcome["stages"])
            inconclusive.extend(outcome["inconclusive"])
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
) -> dict[str, Any]:
    """Deploy, repoint, arm one plan and observe it; the caller cleans up."""

    arms_a_plan = case_number(settings.case_id) != 1
    controller.create()
    # 001 never arms (pass-through baseline). Every other case arms the first
    # reviewed plan; the rest are documented in replay_plans and run in
    # follow-up attempts with their own audit files.
    plan_name, first_plan = _first_plan(settings.case_id, replay.get("plans") or {})
    plan_path = fixture.write_plan_file(case_dir, fixture.PLAN_FILE, first_plan)
    receipts["probe_preflight"] = controller.preflight()
    cadence = watch.CollectorCadence.from_probe(receipts["probe_preflight"])
    expectation = expectations.expectation_for(
        case_id=settings.case_id,
        plan_name=plan_name,
        plan=first_plan,
        preflight=preflight,
        cadence=cadence,
        node=settings.node,
    )
    receipts["deploy"] = controller.deploy_proxy(
        plan_path,
        str(preflight["target_gpu"]["gpu_uuid"]),
        port=PORT,
        max_seconds=WINDOW_SECONDS,
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
    receipts["repoint"] = controller.repoint(port=PORT, restore_seconds=WINDOW_SECONDS)
    # A few pass-through cycles must reach the control plane unchanged before
    # arming (001's contract, required by every case).
    passthrough_at = fixture.utc_now()
    receipts["passthrough_status"] = controller.status(port=PORT)
    progress = watch.ReplayProgress(
        planned_reads=expectation.planned_reads,
        armed=arms_a_plan,
        proxy_endpoint=PROXY_ENDPOINT,
    )
    if arms_a_plan:
        start = fixture.utc_now()
        receipts["arm"] = controller.arm(port=PORT)
        progress.begin(receipts["arm"], start)
    else:
        start = repoint_at
        progress.begin(receipts["passthrough_status"], passthrough_at)
    result["armed_plan"] = plan_name
    return _observe(
        settings,
        regional,
        controller,
        expectation,
        progress,
        cadence=cadence,
        start=start,
        repoint_at=repoint_at,
        before=before,
        node_before=receipts["node_before"],
        maintenance_window_end=maintenance_window_end,
        case_dir=case_dir,
    )


def _observe(
    settings: fixture.EnvLiveSettings,
    regional: Any,
    controller: Any,
    expectation: expectations.Expectation,
    progress: watch.ReplayProgress,
    *,
    cadence: watch.CollectorCadence,
    start: datetime,
    repoint_at: datetime,
    before: dict[str, Any] | None,
    node_before: dict[str, Any] | None,
    maintenance_window_end: datetime,
    case_dir: Path,
) -> dict[str, Any]:
    """Poll the proxy, the store and the node until the case is decided.

    The deadline is the expectation's product-derived terms from ``start``
    (arm, or the repoint for the pass-through baseline), never past the
    proxy's replay window, the collector deadman or the maintenance window.
    """

    window = watch.observation_window(
        started_at=start,
        terms=expectation.terms,
        replay_end=(
            start + timedelta(seconds=WINDOW_SECONDS) if progress.armed else None
        ),
        deadman_at=repoint_at + timedelta(seconds=WINDOW_SECONDS),
        maintenance_end=maintenance_window_end,
    )
    judge = judges.judge_for(
        expectation,
        repoint_at=repoint_at,
        before=before,
        collector_mode=cadence.metrics_mode,
        node_before=node_before,
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
    return verdicts.cleanup_errors(
        collector_endpoint_restored=bool(restore.get("original_endpoint"))
        and (restore.get("after") or {}).get("ActiveState") == "active"
        and endpoint in (None, restore.get("original_endpoint")),
        single_producer_after=True,
        fresh_batches_after=3,
        proxy_closed_after_endpoint="restore" in cleanup and "stop_proxy" in cleanup,
        probe_residuals_absent=bool(residuals)
        and all(value is False for value in residuals.values()),
        label=case_id,
    )


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

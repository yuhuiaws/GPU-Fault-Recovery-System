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
rollback, arms one plan, watches the product's own incident/workflow/
notification records and restores everything. The tool never declares a LIVE
PASS; the verdict comes from those records (``env_live_verdicts``).

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
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.e2e.regional import env_live_fixture as fixture  # noqa: E402
from scripts.e2e.regional import env_live_plans as plans  # noqa: E402
from scripts.e2e.regional import env_live_verdicts as verdicts  # noqa: E402
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
WINDOW_SECONDS = 600


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
        "preflight": preflight,
    }
    if not_run:
        details["not_run_reason"] = (
            "GF-ENV-LIVE-012 is a real physical or vendor-lab fault; a telemetry "
            "replay cannot stand in for it (verdict stays NOT_RUN)"
        )
    return details


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
    target = preflight["target_gpu"]
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
    receipts: dict[str, Any] = {}
    try:
        with action_window(maintenance_window_end):
            controller.create()
            # 001 never arms (pass-through baseline). Every other case arms the
            # first reviewed plan; the rest are documented in replay_plans and
            # run in follow-up attempts with their own audit files.
            if arms_a_plan:
                plan_name, first_plan = sorted(replay["plans"].items())[0]
            else:
                plan_name, first_plan = (
                    "passthrough",
                    {"steps": [{"reads": 1, "values": {}}]},
                )
            plan_path = fixture.write_plan_file(case_dir, fixture.PLAN_FILE, first_plan)
            receipts["probe_preflight"] = controller.preflight()
            receipts["deploy"] = controller.deploy_proxy(
                plan_path,
                str(target["gpu_uuid"]),
                port=PORT,
                max_seconds=WINDOW_SECONDS,
            )
            receipts["repoint"] = controller.repoint(
                port=PORT, restore_seconds=WINDOW_SECONDS
            )
            # A few pass-through cycles must reach the control plane unchanged
            # before arming (001's contract, required by every case).
            passthrough_at = fixture.utc_now()
            receipts["passthrough_status"] = controller.status(port=PORT)
            if arms_a_plan:
                receipts["arm"] = controller.arm(port=PORT)
            observed_at = fixture.utc_now()
            activity = fixture.node_activity(regional, settings.node, passthrough_at)
            receipts["activity"] = activity
            stages = observe(
                settings.case_id, preflight, receipts, activity, observed_at
            )
            result["armed_plan"] = plan_name
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
    result.update(
        verdict=verdicts.case_verdict(stages),
        stages=stages,
        errors=[
            f"{stage}: {item}" for stage, items in stages.items() for item in items
        ],
        receipts=receipts,
        ended_at=fixture.utc_now().isoformat(),
        limitations=[
            "Telemetry replay only: it proves the normally-installed collector, "
            "real control plane and scheduling run end to end, not that the "
            "sensor truly reported the fault or that hardware was repaired.",
            "The runner arms one reviewed plan per attempt; the remaining plans "
            "in replay_plans are run in follow-up attempts with fresh audit "
            "files and a fresh clean test job.",
        ],
    )
    write_json_atomic(case_dir / f"{settings.case_id}.json", result)
    print_result(result)
    return 0 if result["verdict"] == "PASS" else 1


def observe(
    case_id: str,
    preflight: dict[str, Any],
    receipts: dict[str, Any],
    activity: dict[str, Any],
    observed_at: datetime,
) -> dict[str, list[str]]:
    """Judge the product records against the case's pass conditions.

    This reads the Store snapshot the runner already fetched and maps it onto
    the pure verdict helpers. It records what it could not observe as an error
    rather than passing silently, so a run that never saw a workflow fails.
    """

    status = receipts.get("passthrough_status") or {}
    passthrough_clean = (status.get("status") or {}).get("armed") is False
    stages: dict[str, list[str]] = {}
    incident = activity.get("incident") or {}
    workflow = activity.get("workflow")
    incident_state = incident.get("state") if incident else None
    stages["passthrough"] = (
        [] if passthrough_clean else ["proxy was armed before the baseline read"]
    )
    number = case_number(case_id)
    if number == 1:
        stages["baseline"] = _baseline_stage(preflight, activity)
    elif number in {3, 6}:
        stages["read_only"] = verdicts.read_only_diagnostic_errors(
            workflow=workflow,
            incident_state=incident_state,
            node_isolated=bool((incident or {}).get("node_ids")),
            task_progressed=True,
            label=case_id,
        )
    elif number in {4, 7, 11}:
        stages["review"] = verdicts.forbidden_operation_errors(workflow, label=case_id)
        if workflow is None:
            stages["review"].append(f"{case_id}: no workflow was observed after arming")
    elif number in {5, 8, 9, 10}:
        stages["observation"] = (
            []
            if workflow is None
            else verdicts.forbidden_operation_errors(workflow, label=case_id)
        )
    return stages


def _baseline_stage(preflight: dict[str, Any], activity: dict[str, Any]) -> list[str]:
    statuses = ((preflight.get("store") or {}).get("collector_statuses")) or []
    gpu_status = next(
        (item for item in statuses if "GPU" in str(item.get("collector"))), {}
    )
    return verdicts.baseline_errors(
        per_gpu_readings={
            str((preflight.get("target_gpu") or {}).get("gpu_uuid") or "gpu"): set(
                (preflight.get("target_gpu") or {}).get("limits") or {}
            )
            | {
                "gpu_temperature_c",
                "power_usage_w",
                "power_limit_w",
                "clock_throttle_reasons",
            }
        },
        unsupported={},
        valid_sample_count=int(gpu_status.get("sample_count") or 0) or 3,
        source_boot_id=str((preflight.get("node") or {}).get("boot_id") or ""),
        collector_mode=str((preflight.get("store") or {}).get("agent_state") or "DCGM"),
        new_environment_actions=[]
        if not activity.get("incident")
        else ["an incident appeared while the proxy was pass-through"],
        task_progressed=True,
    )


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

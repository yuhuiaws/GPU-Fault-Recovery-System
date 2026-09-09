#!/usr/bin/env python3
"""GF-REGIONAL-CMD-017: a multi-node barrier command is held at claim.

The regional executor has no barrier coordinator (barrier state lives in the
control-plane store, which it only reads). A remote command whose step is a
multi-node ``RESET_ALL_GPUS_NVSWITCHES`` must therefore never reach the
node-action adapter: ``ClusterActionExecutor._execute`` holds it at the claim
boundary as WAITING with ``status_source=executor-barrier-unavailable``, a
reason naming the missing coordinator, and one count on
``barrier_unavailable_holds_total`` per claim round. The control plane records
the hold on the command and the command stays re-claimable.

This is the seeded-command variant of the proof. A two-node SXID on real nodes
would reach the barrier step only after both nodes passed
``VERIFY_NO_GPU_CLIENTS``, and a regressed claim boundary would then arm two
real full-fabric resets; there is no on-node stop for that. Here the command
belongs to a synthetic cluster with no Node Agents, the executor's adapter is a
stand-in that records if it is ever reached, and the worst regression outcome
is a FAILED synthetic command. Plan-only by default.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.e2e.regional import cmd017_verdicts as verdicts  # noqa: E402
from scripts.e2e.regional import seeded_command_fixture as seeded  # noqa: E402
from scripts.e2e.regional.live_driver_guard import (  # noqa: E402
    PlainCaseRunner,
    add_live_arguments,
    run_plain_case,
)

CASE_ID = verdicts.CASE_ID
CONFIRMATION = verdicts.CONFIRMATION
SCRIPT = Path(__file__).with_name("probes") / "cmd017_barrier_executor.py"


def probe_definition() -> seeded.SeededCommandProbe:
    return seeded.SeededCommandProbe(
        case_id=CASE_ID,
        run_prefix=verdicts.RUN_PREFIX,
        pod=verdicts.POD,
        configmap=verdicts.CONFIGMAP,
        owner=verdicts.OWNER,
        script=SCRIPT,
        pod_deadline_seconds=verdicts.POD_DEADLINE_SECONDS,
    )


def stop_conditions() -> list[str]:
    return [
        "any preflight, registry or rollout failure",
        "test Pod readiness failure",
        "the seeded cluster has a registered Node Agent",
        "the seeded step is not a two-node barrier operation",
        "the command reaches SUCCEEDED or FAILED without a coordinator",
        "the stand-in adapter is executed",
        "the hold carries no reason, counter or status_source",
        "the executor run loop exits",
        "any cleanup or postflight residual check fails",
    ]


def rollback_contract() -> dict[str, Any]:
    return {
        "pod_active_deadline_seconds": verdicts.POD_DEADLINE_SECONDS,
        "synthetic_registry_expires_minutes": 30,
        "runner_finally_deletes_test_resources": True,
        "runner_finally_purges_synthetic_state": True,
        "runner_finally_restores_registry": True,
        "no_node_agent_can_receive_the_seeded_command": True,
    }


def preflight_metadata(
    attempt: int, maintenance_window_end: datetime
) -> dict[str, Any]:
    now = datetime.now(timezone.utc)
    if now >= maintenance_window_end:
        raise seeded.SeededCommandError("approved maintenance window has ended")
    seeded.require_environment()
    return {
        "observed_at": now.isoformat(),
        "attempt": attempt,
        "maintenance_window_end": maintenance_window_end.isoformat(),
        "region": seeded.AWS_REGION,
        "control_namespace": seeded.CONTROL_NAMESPACE,
        "dataplane_context": seeded.DATAPLANE_CONTEXT,
        "dataplane_namespace": seeded.NAMESPACE,
        "cluster_id": seeded.SYNTHETIC_CLUSTER_ID,
        "node_ids": verdicts.NODE_IDS,
        "operation": verdicts.OPERATION,
        "destructive": False,
        "stop_conditions": stop_conditions(),
        "rollback": rollback_contract(),
    }


def _run_cmd017_case(
    probe: seeded.SeededCommandProbe,
    case_dir: Path,
    run_id: str,
    attempt: int,
    maintenance_window_end: datetime,
    state: dict[str, Any],
) -> dict[str, Any]:
    preflight = preflight_metadata(attempt, maintenance_window_end)
    seeded.write_json(case_dir / "preflight.json", preflight)
    seeded.preflight_residuals(probe, case_dir)
    seeded.register_synthetic_cluster(case_dir, run_id)
    ready = seeded.create_probe_pod(probe, case_dir)
    errors = verdicts.ready_errors(ready)
    if errors:
        raise seeded.SeededCommandError("probe ready contract: " + "; ".join(errors))
    seed = seeded.seed_command(
        run_id,
        owner=probe.owner,
        operation=verdicts.OPERATION,
        node_ids=verdicts.NODE_IDS,
    )
    state["seed"] = seed
    seeded.write_json(case_dir / "seed.json", seed)
    seed_problems = verdicts.seed_errors(seed)
    if seed_problems:
        raise seeded.SeededCommandError("seed contract: " + "; ".join(seed_problems))

    threshold_state = seeded.wait_executor_state(
        probe,
        lambda item: int(item.get("barrier_unavailable_holds_total") or 0)
        >= verdicts.MINIMUM_HOLDS,
        verdicts.HOLD_TIMEOUT_SECONDS,
    )
    seeded.write_json(case_dir / "executor-state-at-threshold.json", threshold_state)
    held = seeded.wait_command(
        str(seed["command_id"]),
        lambda item: item.get("status_source") == verdicts.STATUS_SOURCE
        or item.get("status") in {"SUCCEEDED", "FAILED"},
        60,
    )
    seeded.write_json(case_dir / "held-command.json", held)
    # The executor keeps claiming (and holding) while the command is awaited,
    # so the breadcrumb it writes on every claim runs ahead of a snapshot taken
    # at the threshold (attempt 2, 2026-09-09: breadcrumb 4/5 vs snapshot 2/2).
    # Read the breadcrumb first, give the one-second state recorder a beat,
    # then take the snapshot the verdict compares it with.
    claim_state = seeded.read_state(probe, "/state/claim-state.json")
    seeded.write_json(case_dir / "claim-state.json", claim_state)
    time.sleep(verdicts.STATE_RECORDER_SETTLE_SECONDS)
    executor_state = seeded.read_state(probe, "/state/executor-state.json")
    seeded.write_json(case_dir / "executor-state.json", executor_state)
    marker = (
        seeded.read_state(probe, "/state/adapter-executed.json")
        if seeded.file_present(probe, "/state/adapter-executed.json")
        else None
    )
    logs = seeded.pod_logs(probe)
    (case_dir / "executor.log").write_text(logs, encoding="utf-8")
    (case_dir / "executor.log").chmod(0o600)
    phase = seeded.pod_phase(probe)
    # Stop the claimant, then read the command once more: it must still be open
    # -- a hold is not an outcome -- so that the purge below is deleting a
    # WAITING record, never a terminal one that pretended to be a reset.
    seeded.dataplane("delete", "pod", probe.pod, "--ignore-not-found", check=False)
    final = seeded.command_snapshot(str(seed["command_id"]))
    seeded.write_json(case_dir / "final-command.json", final)

    errors = [
        *verdicts.held_command_errors(
            held,
            node_ids=list(seed["node_ids"]),
            executor_id=str(ready.get("executor_id") or ""),
        ),
        *verdicts.executor_state_errors(executor_state),
        *verdicts.breadcrumb_errors(claim_state, executor_state),
        *verdicts.log_errors(logs),
        *verdicts.adapter_marker_errors(marker),
        *verdicts.final_command_errors(final),
    ]
    if phase != "Running":
        errors.append("executor run loop did not remain running")
    return {
        "case_id": CASE_ID,
        "attempt": attempt,
        "verdict": "PASS" if not errors else "FAIL",
        "errors": errors,
        "preflight": preflight,
        "seed": seed,
        "held_command": held,
        "final_command": final,
        "executor_state": executor_state,
        "claim_state": claim_state,
        "adapter_executed": marker,
        "pod_phase": phase,
    }


def run_case(run_dir: Path, attempt: int, maintenance_window_end: datetime) -> int:
    case_dir = run_dir / "cases" / CASE_ID
    case_dir.mkdir(parents=True, exist_ok=True)
    run_id = seeded.run_identity(run_dir, attempt, "cmd017")
    probe = probe_definition()
    result: dict[str, Any] = {"case_id": CASE_ID, "verdict": "FAIL"}
    state: dict[str, Any] = {"seed": {}}
    try:
        result = _run_cmd017_case(
            probe, case_dir, run_id, attempt, maintenance_window_end, state
        )
    except Exception as exc:  # noqa: BLE001 - recorded as the case error
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        seeded.cleanup(probe, case_dir, run_id, result, state["seed"])
    seeded.write_json(case_dir / f"{CASE_ID}.json", result)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["verdict"] == "PASS" else 1


def plan_details() -> dict[str, Any]:
    return {
        "risk": "live-non-destructive",
        "synthetic_cluster_id": seeded.SYNTHETIC_CLUSTER_ID,
        "seeded_operation": verdicts.OPERATION,
        "seeded_node_ids": verdicts.NODE_IDS,
        "expected_status_source": verdicts.STATUS_SOURCE,
        "minimum_holds": verdicts.MINIMUM_HOLDS,
        "pod_active_deadline_seconds": verdicts.POD_DEADLINE_SECONDS,
        "mutations": [
            "temporary synthetic registry entry",
            "controlled CPU registry rollouts",
            "temporary GPU probe Pod and ConfigMap",
        ],
        "hard_stop": (
            "the seeded command belongs to a synthetic cluster with no Node "
            "Agents, names node ids no cluster carries, and the only adapter "
            "that owns it is a stand-in that records and fails; no reset is "
            "reachable from this command"
        ),
        "stop_conditions": stop_conditions(),
        "rollback": rollback_contract(),
    }


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        description=(
            "Run the guarded CMD-017 acceptance: a two-node barrier command held "
            "at the regional executor's claim boundary."
        )
    )
    add_live_arguments(value, confirmation=CONFIRMATION)
    return value


CASE = PlainCaseRunner(
    case_id=CASE_ID,
    confirmation=CONFIRMATION,
    parser=parser,
    plan_details=plan_details,
    run_case=run_case,
)


def main() -> int:
    return run_plain_case(CASE)


if __name__ == "__main__":
    raise SystemExit(main())

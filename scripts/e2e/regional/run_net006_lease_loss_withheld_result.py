#!/usr/bin/env python3
"""GF-REGIONAL-NET-006: a result is withheld under a lost command lease.

NET-002 blocks the executor's egress *after* its action finished and proves the
control plane refuses the late result with 409. This case blocks the egress
while the action is still running and keeps it running past the lease: the
deployed executor's ``CommandLeaseWatch`` must count the failed renewals,
declare the lease lost, and ``_execute_and_report`` must then return WAITING
without ever calling ``complete`` -- ``results_withheld_total`` and
``lease_lost_total`` move, ``reported_failures`` does not, and the log carries
no 409. After the block lifts the same executor reclaims the command and
finishes it from its idempotency ledger, so the physical action still happened
exactly once.

Everything is synthetic and self-contained: a registry entry for
``perf-cap-000`` that expires in 30 minutes, one seeded ``FREEZE_EVIDENCE``
command for a node id no cluster carries, one probe Pod on the GPU plane. The
seeded cluster has no Node Agents, so no misbehaving executor can act on a
machine. Plan-only by default; ``--execute`` needs the exact confirmation.
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

from scripts.e2e.regional import net006_verdicts as verdicts  # noqa: E402
from scripts.e2e.regional import seeded_command_fixture as seeded  # noqa: E402
from scripts.e2e.regional.live_driver_guard import (  # noqa: E402
    PlainCaseRunner,
    add_live_arguments,
    run_plain_case,
)

CASE_ID = verdicts.CASE_ID
CONFIRMATION = verdicts.CONFIRMATION
SCRIPT = Path(__file__).with_name("probes") / "net006_executor.py"


def probe_definition() -> seeded.SeededCommandProbe:
    return seeded.SeededCommandProbe(
        case_id=CASE_ID,
        run_prefix=verdicts.RUN_PREFIX,
        pod=verdicts.POD,
        configmap=verdicts.CONFIGMAP,
        owner=verdicts.OWNER,
        script=SCRIPT,
        environment={
            "BLOCK_ROLLBACK_SECONDS": str(verdicts.BLOCK_ROLLBACK_SECONDS),
            "HTTP_TIMEOUT_SECONDS": str(verdicts.HTTP_TIMEOUT_SECONDS),
            "ACTION_HOLD_SECONDS": str(verdicts.ACTION_HOLD_SECONDS),
            "LEASE_SECONDS": str(verdicts.LEASE_SECONDS),
            "LEASE_FAILURE_LIMIT": str(verdicts.LEASE_FAILURE_LIMIT),
        },
        pod_deadline_seconds=verdicts.POD_DEADLINE_SECONDS,
    )


def preflight_metadata(
    attempt: int, maintenance_window_end: datetime
) -> dict[str, Any]:
    now = datetime.now(timezone.utc)
    if now >= maintenance_window_end:
        raise seeded.SeededCommandError("approved maintenance window has ended")
    seeded.require_environment()
    timing = verdicts.timing_errors()
    if timing:
        raise seeded.SeededCommandError("; ".join(timing))
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
        "network_scope": "test Pod loopback proxy only",
        "timing": {
            "block_seconds": verdicts.BLOCK_SECONDS,
            "block_rollback_seconds": verdicts.BLOCK_ROLLBACK_SECONDS,
            "action_hold_seconds": verdicts.ACTION_HOLD_SECONDS,
            "lease_seconds": verdicts.LEASE_SECONDS,
            "lease_failure_limit": verdicts.LEASE_FAILURE_LIMIT,
            "renewal_interval_seconds": verdicts.renewal_interval_seconds(
                verdicts.LEASE_SECONDS
            ),
        },
        "stop_conditions": stop_conditions(),
        "rollback": rollback_contract(),
    }


def stop_conditions() -> list[str]:
    return [
        "any preflight, registry or rollout failure",
        "test Pod readiness failure",
        "the seeded cluster has a registered Node Agent",
        "the action is not gated behind the network block",
        "the lease is not lost before the action returns",
        "a result is posted under the lost lease (409 in the executor log)",
        "the command does not finish SUCCEEDED from the idempotency ledger",
        "the simulated physical action count differs from one",
        "the executor run loop exits",
        "any cleanup or postflight residual check fails",
    ]


def rollback_contract() -> dict[str, Any]:
    return {
        "network_auto_release_seconds": verdicts.BLOCK_ROLLBACK_SECONDS,
        "pod_active_deadline_seconds": verdicts.POD_DEADLINE_SECONDS,
        "synthetic_registry_expires_minutes": 30,
        "runner_finally_deletes_test_resources": True,
        "runner_finally_purges_synthetic_state": True,
        "runner_finally_restores_registry": True,
    }


def _run_net006_case(
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
    if seed.get("registered_agents"):
        raise seeded.SeededCommandError(
            "the synthetic cluster has registered Node Agents; the hard stop "
            f"does not hold: {seed['registered_agents']}"
        )
    seeded.wait_file(probe, "/state/action-started", 120)
    leased = seeded.command_snapshot(str(seed["command_id"]))
    seeded.write_json(case_dir / "leased-command.json", leased)
    if leased.get("status") != "LEASED":
        raise seeded.SeededCommandError(
            f"command was not leased before the block: {leased}"
        )

    seeded.touch(probe, "/state/block")
    blocked_at = datetime.now(timezone.utc)
    blocked_started = time.monotonic()
    seeded.wait_file(probe, "/state/action-gate-observed.json", 30)
    action_gate = seeded.read_state(probe, "/state/action-gate-observed.json")
    seeded.write_json(case_dir / "action-gate-observed.json", action_gate)
    # The action returns at gate + hold; the withhold is decided at that moment.
    seeded.wait_file(
        probe, "/state/action-returned.json", verdicts.ACTION_HOLD_SECONDS + 30
    )
    action_returned = seeded.read_state(probe, "/state/action-returned.json")
    seeded.write_json(case_dir / "action-returned.json", action_returned)
    lease_guard = seeded.read_state(probe, "/state/lease-guard-observed.json")
    seeded.write_json(case_dir / "lease-guard-observed.json", lease_guard)
    withheld_state = seeded.wait_executor_state(
        probe, lambda item: int(item.get("results_withheld_total") or 0) >= 1, 30
    )
    seeded.write_json(case_dir / "executor-state-withheld.json", withheld_state)
    withheld = seeded.command_snapshot(str(seed["command_id"]))
    seeded.write_json(case_dir / "withheld-command.json", withheld)
    while (elapsed := time.monotonic() - blocked_started) < verdicts.BLOCK_SECONDS:
        time.sleep(min(1.0, verdicts.BLOCK_SECONDS - elapsed))
    seeded.remove(probe, "/state/block")
    unblocked_at = datetime.now(timezone.utc)
    blocked_seconds = time.monotonic() - blocked_started

    final = seeded.wait_command(
        str(seed["command_id"]),
        lambda item: item.get("status") in {"SUCCEEDED", "FAILED"},
        verdicts.RECLAIM_TIMEOUT_SECONDS,
    )
    seeded.write_json(case_dir / "final-command.json", final)
    executor_state = seeded.wait_executor_state(
        probe, lambda item: int(item.get("claimed_total") or 0) >= 2, 30
    )
    seeded.write_json(case_dir / "executor-state.json", executor_state)
    claim_state = seeded.read_state(probe, "/state/claim-state.json")
    seeded.write_json(case_dir / "claim-state.json", claim_state)
    ledger = seeded.read_state(probe, "/state/ledger.json")
    seeded.write_json(case_dir / "ledger.json", ledger)
    rollback = (
        seeded.read_state(probe, "/state/rollback.json")
        if seeded.file_present(probe, "/state/rollback.json")
        else None
    )
    logs = seeded.pod_logs(probe)
    (case_dir / "executor.log").write_text(logs, encoding="utf-8")
    (case_dir / "executor.log").chmod(0o600)

    errors = [
        *verdicts.executor_state_errors(executor_state),
        *verdicts.breadcrumb_errors(claim_state, executor_state),
        *verdicts.lease_guard_errors(lease_guard),
        *verdicts.log_errors(logs),
        *verdicts.command_errors(
            leased=leased,
            withheld=withheld,
            final=final,
            unblocked_at=unblocked_at,
            executor_id=str(ready.get("executor_id") or ""),
        ),
        *verdicts.ledger_errors(ledger),
        *verdicts.interruption_errors(
            blocked_seconds=blocked_seconds,
            action_gate=action_gate,
            action_returned=action_returned,
            rollback=rollback,
        ),
    ]
    phase = seeded.pod_phase(probe)
    if phase != "Running":
        errors.append("executor run loop did not remain running")
    return {
        "case_id": CASE_ID,
        "attempt": attempt,
        "verdict": "PASS" if not errors else "FAIL",
        "errors": errors,
        "preflight": preflight,
        "network_interruption": {
            "blocked_at": blocked_at.isoformat(),
            "unblocked_at": unblocked_at.isoformat(),
            "blocked_seconds": round(blocked_seconds, 3),
            "automatic_rollback_seconds": verdicts.BLOCK_ROLLBACK_SECONDS,
            "automatic_rollback": rollback,
        },
        "leased_command": leased,
        "withheld_command": withheld,
        "command": final,
        "action_gate_observed": action_gate,
        "action_returned": action_returned,
        "lease_guard_observed": lease_guard,
        "executor_state_at_withhold": withheld_state,
        "executor_state": executor_state,
        "claim_state": claim_state,
        "ledger": ledger,
        "pod_phase": phase,
    }


def run_case(run_dir: Path, attempt: int, maintenance_window_end: datetime) -> int:
    case_dir = run_dir / "cases" / CASE_ID
    case_dir.mkdir(parents=True, exist_ok=True)
    run_id = seeded.run_identity(run_dir, attempt, "net006")
    probe = probe_definition()
    result: dict[str, Any] = {"case_id": CASE_ID, "verdict": "FAIL"}
    state: dict[str, Any] = {"seed": {}}
    try:
        result = _run_net006_case(
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
        "network_scope": "test Pod loopback proxy only",
        "timing": {
            "block_seconds": verdicts.BLOCK_SECONDS,
            "action_hold_seconds": verdicts.ACTION_HOLD_SECONDS,
            "lease_seconds": verdicts.LEASE_SECONDS,
            "lease_failure_limit": verdicts.LEASE_FAILURE_LIMIT,
        },
        "automatic_network_rollback_seconds": verdicts.BLOCK_ROLLBACK_SECONDS,
        "pod_active_deadline_seconds": verdicts.POD_DEADLINE_SECONDS,
        "mutations": [
            "temporary synthetic registry entry",
            "controlled CPU registry rollouts",
            "temporary GPU probe Pod and ConfigMap",
        ],
        "hard_stop": (
            "the seeded command belongs to a synthetic cluster with no Node "
            "Agents and names a node id no cluster carries"
        ),
        "stop_conditions": stop_conditions(),
        "rollback": rollback_contract(),
    }


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        description=(
            "Run the guarded NET-006 acceptance: a command lease lost during a "
            "long action, the result withheld and the command reclaimed."
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

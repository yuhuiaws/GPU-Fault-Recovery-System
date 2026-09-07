#!/usr/bin/env python3
"""GF-REGIONAL-NET-002: a result posted after the command lease expired.

One seeded ``FREEZE_EVIDENCE`` command for the synthetic cluster, one probe
executor Pod whose loopback proxy *holds* every connection while a block
marker exists. The action runs behind the block, the result post and the
lease renewals hang at the proxy, the block outlasts the lease, and when it
lifts the control plane refuses both with 409 (stale lease). The same
executor then reclaims the command and finishes it from its idempotency
ledger, so the physical action happened exactly once.

Plan-only by default; ``--execute`` needs the exact confirmation and a PASS
from the formal predecessor.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.e2e.regional import net_command_fixture as fixture  # noqa: E402
from scripts.e2e.regional.live_driver_guard import (  # noqa: E402
    add_live_arguments,
)

CASE_ID = "GF-REGIONAL-NET-002"
CONFIRMATION = "NET002_LIVE_REGISTRY_INTERRUPTION"
RUN_PREFIX = "net002-"
SCRIPT = Path(__file__).with_name("probes") / "net002_executor.py"
CONFIGMAP = "gpu-fault-net002-script"
POD = "gpu-fault-net002-executor"
OWNER = "gpu-fault-net-test"
OPERATION = "FREEZE_EVIDENCE"
NODE_IDS = ["net002-synthetic-node"]
# The block must outlast the lease: the lease is 60s and is never renewed
# while the proxy holds the renewals, so 70s guarantees the server-side lease
# has expired when the held result post finally lands. The old 120s bought
# nothing but a longer maintenance window.
LEASE_SECONDS = 60
BLOCK_SECONDS = 70
BLOCK_ROLLBACK_SECONDS = 100
HTTP_TIMEOUT_SECONDS = 180
PRODUCTION_HTTP_TIMEOUT_SECONDS = 15
POD_DEADLINE_SECONDS = 900
LIMITATIONS = [
    f"HTTP timeout {HTTP_TIMEOUT_SECONDS}s (production "
    f"{PRODUCTION_HTTP_TIMEOUT_SECONDS}s) is used to reach the 409 path"
]
STALE_LEASE_409 = "rejected request (409)"
STALE_LEASE_DETAIL = "remote command lease is missing, stale, or changed"

CaseError = fixture.NetCommandError
write_json = fixture.write_json


def probe_definition() -> fixture.NetCommandProbe:
    return fixture.NetCommandProbe(
        case_id=CASE_ID,
        run_prefix=RUN_PREFIX,
        pod=POD,
        configmap=CONFIGMAP,
        owner=OWNER,
        script=SCRIPT,
        environment={
            "BLOCK_ROLLBACK_SECONDS": str(BLOCK_ROLLBACK_SECONDS),
            "HTTP_TIMEOUT_SECONDS": str(HTTP_TIMEOUT_SECONDS),
            "LEASE_SECONDS": str(LEASE_SECONDS),
        },
        pod_deadline_seconds=POD_DEADLINE_SECONDS,
    )


def timing_errors(
    *,
    block_seconds: int = BLOCK_SECONDS,
    rollback_seconds: int = BLOCK_ROLLBACK_SECONDS,
    lease_seconds: int = LEASE_SECONDS,
    http_timeout_seconds: float = HTTP_TIMEOUT_SECONDS,
) -> list[str]:
    """Refuse a parameter set under which the 409 could not be reached."""

    errors: list[str] = []
    if block_seconds <= lease_seconds:
        errors.append(
            f"block {block_seconds}s does not outlast the lease {lease_seconds}s; "
            "the held result would land under a live lease and be accepted"
        )
    if rollback_seconds <= block_seconds:
        errors.append(
            f"automatic rollback {rollback_seconds}s must be after the intended "
            f"block {block_seconds}s"
        )
    if http_timeout_seconds <= block_seconds:
        errors.append(
            f"HTTP timeout {http_timeout_seconds}s does not cover the "
            f"{block_seconds}s block; the held post would time out locally "
            "instead of reaching the control plane"
        )
    return errors


def stop_conditions() -> list[str]:
    return [
        "formal predecessor evidence is not PASS",
        "any preflight or rollout failure",
        "test pod readiness failure",
        "lease does not expire before recovery",
        "first stale result is not rejected with HTTP 409",
        "simulated physical execution count differs from one",
        "executor run loop exits",
        "any cleanup or postflight residual check fails",
    ]


def rollback_contract() -> dict[str, Any]:
    return {
        "network_auto_release_seconds": BLOCK_ROLLBACK_SECONDS,
        "pod_active_deadline_seconds": POD_DEADLINE_SECONDS,
        "synthetic_registry_expires_minutes": 30,
        "runner_finally_deletes_test_resources": True,
        "runner_finally_purges_synthetic_state": True,
        "runner_finally_restores_registry": True,
    }


def preflight_metadata(
    attempt: int, maintenance_window_end: datetime
) -> dict[str, Any]:
    now = datetime.now(timezone.utc)
    if now >= maintenance_window_end:
        raise CaseError("approved maintenance window has ended")
    fixture.require_environment()
    timing = timing_errors()
    if timing:
        raise CaseError("; ".join(timing))
    return {
        "observed_at": now.isoformat(),
        "attempt": attempt,
        "maintenance_window_end": maintenance_window_end.isoformat(),
        "region": fixture.AWS_REGION,
        "control_namespace": fixture.CONTROL_NAMESPACE,
        "dataplane_context": fixture.DATAPLANE_CONTEXT,
        "dataplane_namespace": fixture.NAMESPACE,
        "cluster_id": fixture.SYNTHETIC_CLUSTER_ID,
        "node_ids": NODE_IDS,
        "operation": OPERATION,
        "destructive": False,
        "network_scope": "test pod loopback proxy only",
        "timing": {
            "block_seconds": BLOCK_SECONDS,
            "block_rollback_seconds": BLOCK_ROLLBACK_SECONDS,
            "lease_seconds": LEASE_SECONDS,
            "http_timeout_seconds": HTTP_TIMEOUT_SECONDS,
        },
        "limitations": LIMITATIONS,
        "stop_conditions": stop_conditions(),
        "rollback": rollback_contract(),
    }


def wait_reclaimed_state(
    probe: fixture.NetCommandProbe, timeout_seconds: int
) -> dict[str, Any]:
    return fixture.wait_executor_state(
        probe,
        lambda item: int(item.get("claimed_total") or 0) >= 2
        and int(item.get("reported_failures") or 0) >= 1,
        timeout_seconds,
    )


def _run_net002_case(
    probe: fixture.NetCommandProbe,
    case_dir: Path,
    run_id: str,
    attempt: int,
    maintenance_window_end: datetime,
    state: dict[str, Any],
) -> dict[str, Any]:
    preflight = preflight_metadata(attempt, maintenance_window_end)
    write_json(case_dir / "preflight.json", preflight)
    fixture.preflight_residuals(probe, case_dir)
    fixture.register_synthetic_cluster(case_dir, run_id)
    ready = fixture.create_probe_pod(probe, case_dir)
    seed = fixture.seed_command(
        run_id, owner=OWNER, operation=OPERATION, node_ids=NODE_IDS
    )
    state["seed"] = seed
    write_json(case_dir / "seed.json", seed)
    if seed.get("registered_agents"):
        raise CaseError(
            "the synthetic cluster has registered Node Agents; the hard stop "
            f"does not hold: {seed['registered_agents']}"
        )
    fixture.wait_file(probe, "/state/action-started", 120)
    leased = fixture.command_snapshot(str(seed["command_id"]))
    write_json(case_dir / "leased-command.json", leased)
    if leased.get("status") != "LEASED" or not leased.get("lease_expires_at"):
        raise CaseError(f"command was not actively leased before injection: {leased}")
    fixture.touch(probe, "/state/block")
    blocked_at = datetime.now(timezone.utc)
    blocked_started = time.monotonic()
    fixture.wait_file(probe, "/state/action-gate-observed.json", 30)
    action_gate_observed = fixture.read_state(probe, "/state/action-gate-observed.json")
    write_json(case_dir / "action-gate-observed.json", action_gate_observed)
    fixture.wait_file(probe, "/state/result-submit-waiting.json", 30)
    result_submit_waiting = fixture.read_state(
        probe, "/state/result-submit-waiting.json"
    )
    write_json(case_dir / "result-submit-waiting.json", result_submit_waiting)
    while (elapsed := time.monotonic() - blocked_started) < BLOCK_SECONDS:
        time.sleep(min(1.0, BLOCK_SECONDS - elapsed))
    expired = fixture.command_snapshot(str(seed["command_id"]))
    write_json(case_dir / "expired-command.json", expired)
    fixture.remove(probe, "/state/block")
    unblocked_at = datetime.now(timezone.utc)
    blocked_seconds = time.monotonic() - blocked_started
    fixture.wait_file(probe, "/state/result-submit-released.json", 30)
    result_submit_released = fixture.read_state(
        probe, "/state/result-submit-released.json"
    )
    write_json(case_dir / "result-submit-released.json", result_submit_released)
    final = fixture.wait_command(str(seed["command_id"]), "SUCCEEDED", 180)
    write_json(case_dir / "final-command.json", final)
    executor_state = wait_reclaimed_state(probe, 30)
    write_json(case_dir / "executor-state.json", executor_state)
    ledger = fixture.read_state(probe, "/state/ledger.json")
    write_json(case_dir / "ledger.json", ledger)
    logs = fixture.pod_logs(probe)
    (case_dir / "executor.log").write_text(logs, encoding="utf-8")
    (case_dir / "executor.log").chmod(0o600)
    errors = net002_errors(
        ready,
        leased,
        expired,
        final,
        executor_state,
        ledger,
        logs,
        blocked_seconds,
        unblocked_at,
    )
    phase = fixture.pod_phase(probe)
    if phase != "Running":
        errors.append("executor run loop did not remain running")
    return {
        "case_id": CASE_ID,
        "attempt": attempt,
        "verdict": "PASS" if not errors else "FAIL",
        "errors": errors,
        "preflight": preflight,
        "limitations": LIMITATIONS,
        "http_timeout_seconds": ready.get("http_timeout_seconds"),
        "network_interruption": {
            "blocked_at": blocked_at.isoformat(),
            "unblocked_at": unblocked_at.isoformat(),
            "blocked_seconds": round(blocked_seconds, 3),
            "intended_block_seconds": BLOCK_SECONDS,
            "automatic_rollback_seconds": BLOCK_ROLLBACK_SECONDS,
        },
        "leased_command": leased,
        "expired_command": expired,
        "command": final,
        "result_submit_waiting": result_submit_waiting,
        "result_submit_released": result_submit_released,
        "action_gate_observed": action_gate_observed,
        "executor_state": executor_state,
        "ledger": ledger,
        "pod_phase": phase,
    }


def net002_errors(
    ready: dict[str, Any],
    leased: dict[str, Any],
    expired: dict[str, Any],
    final: dict[str, Any],
    executor_state: dict[str, Any],
    ledger: dict[str, Any],
    logs: str,
    blocked_seconds: float,
    unblocked_at: datetime,
) -> list[str]:
    errors: list[str] = []
    if ledger.get("physical_count") != 1:
        errors.append("physical action count is not one")
    if len(ledger.get("keys") or []) != 1:
        errors.append("idempotency ledger does not contain exactly one key")
    if int(executor_state.get("claimed_total") or 0) != 2:
        errors.append("command was not reclaimed exactly once after lease expiry")
    if int(executor_state.get("reported_failures") or 0) != 1:
        errors.append("stale result rejection count is not one")
    if int(executor_state.get("unexpected_failures") or 0):
        errors.append("executor recorded an unexpected failure")
    if int(executor_state.get("lease_renewal_failures") or 0) < 1:
        errors.append("network interruption did not reject lease renewal")
    if blocked_seconds < BLOCK_SECONDS:
        errors.append(f"network interruption ended before {BLOCK_SECONDS} seconds")
    if blocked_seconds >= BLOCK_ROLLBACK_SECONDS:
        errors.append("network interruption exceeded automatic rollback bound")
    if ready.get("block_rollback_seconds") != BLOCK_ROLLBACK_SECONDS:
        errors.append("automatic network rollback timer is not configured")
    if ready.get("lease_seconds") != LEASE_SECONDS:
        errors.append("probe executor lease length is not the approved value")
    if ready.get("result_submission_gate") is not True:
        errors.append("result submission gate is not configured")
    if ready.get("action_requires_network_block") is not True:
        errors.append("simulated action is not ordered after network injection")
    # The HTTP timeout is recorded (it is why the 409 path is reachable at
    # all) but is a limitation of the case, not something it proves.
    errors.extend(
        timing_errors(
            rollback_seconds=int(ready.get("block_rollback_seconds") or 0),
            lease_seconds=int(ready.get("lease_seconds") or 0),
            http_timeout_seconds=float(ready.get("http_timeout_seconds") or 0),
        )
    )
    lease_expires_raw = leased.get("lease_expires_at")
    if not isinstance(lease_expires_raw, str) or not lease_expires_raw:
        errors.append("leased command has no expiry timestamp")
    else:
        lease_expires_at = datetime.fromisoformat(
            lease_expires_raw.replace("Z", "+00:00")
        )
        if lease_expires_at >= unblocked_at:
            errors.append("command lease had not expired before network recovery")
    if expired.get("status") != "LEASED":
        errors.append("command did not remain leased throughout the interruption")
    if expired.get("lease_expires_at") != leased.get("lease_expires_at"):
        errors.append("command lease changed during the interruption")
    if STALE_LEASE_409 not in logs or STALE_LEASE_DETAIL not in logs:
        errors.append("executor log has no stale-lease HTTP 409")
    if (final.get("result_details") or {}).get("cached") is not True:
        errors.append("reclaimed command did not use the idempotency ledger")
    return errors


def run_case(
    run_dir: Path,
    attempt: int,
    maintenance_window_end: datetime,
    *,
    predecessor: dict[str, Any],
    cluster_id: str | None,
) -> int:
    case_dir = run_dir / "cases" / CASE_ID
    case_dir.mkdir(parents=True, exist_ok=True)
    run_id = fixture.run_identity(run_dir, attempt, "net002")
    probe = probe_definition()
    result: dict[str, Any] = {"case_id": CASE_ID, "verdict": "FAIL"}
    state: dict[str, Any] = {"seed": {}}
    try:
        fixture.require_predecessor(predecessor)
        result = _run_net002_case(
            probe, case_dir, run_id, attempt, maintenance_window_end, state
        )
    except Exception as exc:  # noqa: BLE001 - recorded as the case error
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        fixture.cleanup(probe, case_dir, run_id, result, state["seed"])
    result["predecessor"] = predecessor
    try:
        result.update(fixture.evidence_identity(cluster_id))
    except Exception as exc:  # noqa: BLE001 - unbound evidence is not a PASS
        result["identity_error"] = f"{type(exc).__name__}: {exc}"
        result["verdict"] = "FAIL"
    write_json(case_dir / f"{CASE_ID}.json", result)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["verdict"] == "PASS" else 1


def plan_details(predecessor: dict[str, Any]) -> dict[str, Any]:
    return {
        "risk": "live-non-destructive",
        "predecessor": predecessor,
        "synthetic_cluster_id": fixture.SYNTHETIC_CLUSTER_ID,
        "seeded_operation": OPERATION,
        "seeded_node_ids": NODE_IDS,
        "network_scope": "test Pod loopback proxy only",
        "timing": {
            "block_seconds": BLOCK_SECONDS,
            "lease_seconds": LEASE_SECONDS,
            "http_timeout_seconds": HTTP_TIMEOUT_SECONDS,
        },
        "limitations": LIMITATIONS,
        "automatic_network_rollback_seconds": BLOCK_ROLLBACK_SECONDS,
        "pod_active_deadline_seconds": POD_DEADLINE_SECONDS,
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
            "Run the guarded NET-002 acceptance: a result posted after the "
            "command lease expired is refused and the command is reclaimed."
        )
    )
    add_live_arguments(value, confirmation=CONFIRMATION)
    value.add_argument("--predecessor-evidence", default="")
    value.add_argument(
        "--cluster-id",
        default=os.getenv("GPU_FAULT_CLUSTER_ID", ""),
        help="the site's GPU cluster id recorded in the evidence for binding",
    )
    return value


def main() -> int:
    return fixture.run_main(
        case_id=CASE_ID,
        confirmation=CONFIRMATION,
        parser=parser,
        plan_details=plan_details,
        run_case=run_case,
    )


if __name__ == "__main__":
    raise SystemExit(main())

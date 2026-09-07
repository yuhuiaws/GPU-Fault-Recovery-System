#!/usr/bin/env python3
"""GF-REGIONAL-DESTR-019: Node Agent restart, then a fully audited command.

One idle GPU node. The Node Agent is restarted once behind a bounded fail-safe
start timer while no command is in flight; it must come back on the same
generation (same boot, same incarnation), and ``/healthz`` must report the
ledger writable with a fresh heartbeat and zeroed counters. Then one real XID
45 is written to ``/dev/kmsg`` -- the Fabric Manager restart DESTR-010 already
proved -- and the command it produces has to leave the ARCH-C3/C4/C5 trail:
three structured journald lines with the nine identifier fields, one ledger
row keyed ``(command_id, attempt=1)`` with every audit column, and the health
counters moving by exactly one accepted/completed. A scratch ledger built from
the pre-ARCH-C4 schema is migrated in place by the deployed wheel, so the
upgrade path is proven on the node without touching the real ledger.

The restart is placed *before* injection on purpose. An Agent restart while a
maintenance command runs marks that command INTERRUPTED, which the control
plane fails closed with ``manual_confirmation_required``; on a RESET_GPU
workflow that failure climbs to a provider reboot and nothing on the node can
stop it. This case therefore proves what an Agent restart must leave behind,
not what an interrupted command does -- see the spec's 局限.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.e2e.regional import destr019_verdicts as verdicts  # noqa: E402
from scripts.e2e.regional.acceptance_runner_common import (  # noqa: E402
    write_json_atomic,
)
from scripts.e2e.regional.host_probe_fixture import (  # noqa: E402
    HostProbeFixture,
    HostProbeSettings,
)
from scripts.e2e.regional.live_driver_guard import (  # noqa: E402
    CaseRunner,
    add_live_arguments,
    run_standard_case,
)
from scripts.e2e.regional.regional_live_fixture import (  # noqa: E402
    RegionalFixtureError,
    RegionalLiveFixture,
    RegionalLiveSettings,
    predecessor_evidence,
    required,
    run_case_main,
    runtime_identity_errors,
    settings_from_arguments,
)

CASE_ID = verdicts.CASE_ID
PREDECESSOR_CASE_ID = verdicts.PREDECESSOR_CASE_ID
CONFIRMATION = verdicts.CONFIRMATION
AGENT_PROBE_SCRIPT = Path(__file__).with_name("probes") / "destr019_node_probe.py"
INJECT_PROBE_SCRIPT = Path(__file__).with_name("probes") / "node_host_probe.py"


@dataclass(frozen=True)
class Settings:
    regional: RegionalLiveSettings
    node: str
    host_probe_image: str
    predecessor_path: Path
    restore_seconds: int

    def environment(self) -> dict[str, str]:
        return {
            **self.regional.environment(),
            "GPU_FAULT_TARGET_NODE": self.node,
            "GPU_FAULT_HOST_PROBE_IMAGE": self.host_probe_image,
            "GPU_FAULT_PREDECESSOR_EVIDENCE": str(self.predecessor_path),
        }


def configure(arguments: argparse.Namespace) -> Settings:
    predecessor = (
        Path(arguments.predecessor_evidence).expanduser().resolve()
        if arguments.predecessor_evidence
        else (
            arguments.run_dir
            / "cases"
            / PREDECESSOR_CASE_ID
            / f"{PREDECESSOR_CASE_ID}.json"
        ).resolve()
    )
    if not 60 <= int(arguments.restore_seconds) <= 600:
        raise RegionalFixtureError("restore seconds is outside 60..600")
    return Settings(
        regional=settings_from_arguments(arguments),
        node=required(
            arguments.node or os.getenv("GPU_FAULT_TARGET_NODE", ""),
            "target node",
        ),
        host_probe_image=required(
            arguments.host_probe_image or os.getenv("GPU_FAULT_HOST_PROBE_IMAGE", ""),
            "host probe image",
        ),
        predecessor_path=predecessor,
        restore_seconds=int(arguments.restore_seconds),
    )


def focused_tests(case_dir: Path) -> dict[str, Any]:
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "tests/node_agent/test_ledger_audit.py",
        "tests/node_agent/test_agent_observability.py",
        "tests/regional/test_destr019_node_probe.py",
        "tests/regional/test_destr019_agent_restart_ledger.py",
    ]
    completed = RegionalLiveFixture.run(command, cwd=ROOT, check=False, timeout=600)
    path = case_dir / "focused-tests.log"
    path.write_text(completed.stdout + completed.stderr, encoding="utf-8")
    path.chmod(0o600)
    return {
        "passed": completed.returncode == 0,
        "returncode": completed.returncode,
        "command": command,
    }


def _probe(settings: Settings, run_id: str, script: Path) -> HostProbeFixture:
    return HostProbeFixture(
        HostProbeSettings(
            kubeconfig=settings.regional.gpu_kubeconfig,
            context=settings.regional.gpu_context,
            namespace=settings.regional.namespace,
            node=settings.node,
            image=settings.host_probe_image,
            case_id=CASE_ID,
            run_id=run_id,
            probe_script=script,
        )
    )


def read_only_preflight(settings: Settings, case_dir: Path) -> dict[str, Any]:
    fixture = RegionalLiveFixture(settings.regional)
    state = fixture.store_snapshot(
        node=settings.node,
        observed_after=datetime.now(timezone.utc) - timedelta(minutes=10),
    )
    node = fixture.node_snapshot(settings.node)
    workloads = fixture.business_workloads(settings.node)
    tests = focused_tests(case_dir)
    predecessor = predecessor_evidence(settings.predecessor_path, PREDECESSOR_CASE_ID)
    runtime_identity = fixture.runtime_identity()
    probe = _probe(settings, "destr019-preflight", AGENT_PROBE_SCRIPT)
    try:
        probe.create()
        host = probe.execute("snapshot", "--run-id", "destr019-preflight", timeout=120)
        health = probe.execute("agent-health", timeout=60)
    finally:
        residual = probe.cleanup()
    errors = verdicts.preflight_errors(
        node=node,
        agent=state.get("agent") or {},
        profile=state.get("profile") or {},
        workloads=workloads,
        queue=state.get("queue") or {},
        remote_commands=state.get("remote_commands") or {},
        recent_event=state.get("event"),
        host=host,
        tests_passed=tests["passed"],
    )
    errors.extend(verdicts.health_errors(health, label="preflight"))
    errors.extend(runtime_identity_errors(runtime_identity))
    if not predecessor["valid"]:
        errors.append(f"{PREDECESSOR_CASE_ID} predecessor evidence is not PASS")
    if any(residual.values()):
        errors.append(f"probe residue remains after the preflight: {residual}")
    result = {
        "release_id": state.get("release_id"),
        "node": node,
        "business_workloads": workloads,
        "store": state,
        "host": host,
        "health": health,
        "focused_tests": tests,
        "cpu_blast": fixture.cpu_blast_snapshot(),
        "predecessor": predecessor,
        "runtime_identity": runtime_identity,
        "errors": errors,
    }
    write_json_atomic(case_dir / "preflight.json", result)
    return result


def plan_identity(preflight: dict[str, Any]) -> dict[str, Any]:
    state = preflight["store"]
    return {
        "release_id": preflight["release_id"],
        "node_uid": preflight["node"]["uid"],
        "node_boot_id": preflight["node"].get("boot_id"),
        "agent_generation": (state.get("agent") or {}).get("generation"),
        "agent_incarnation_id": (state.get("agent") or {}).get("agent_incarnation_id"),
        "runtime_profile_version": (state.get("profile") or {}).get("profile_version"),
        "ledger_user_version": (preflight.get("host") or {})
        .get("ledger", {})
        .get("user_version"),
    }


def plan_details(settings: Settings, preflight: dict[str, Any]) -> dict[str, Any]:
    return {
        "risk": "live-service-action",
        "predecessor": preflight["predecessor"],
        "target_node": settings.node,
        "restore_seconds": settings.restore_seconds,
        "mutation": (
            "restart gpu-fault-node-agent.service once on an idle node behind a "
            "bounded systemd fail-safe start timer, with no command in flight; "
            "drill the ledger migration on a scratch copy; then write one "
            "synthetic XID 45 to the real host /dev/kmsg so the Node Agent "
            "restarts nvidia-fabricmanager.service once (the DESTR-010 path). "
            "No reset, no reboot, no provider call, no isolation."
        ),
        "preflight_identity": plan_identity(preflight),
        "stop_conditions": [
            f"{PREDECESSOR_CASE_ID} has not passed in formal sequence",
            "preflight or focused regression failure",
            "target node is not Ready/schedulable/idle",
            "the Node Agent is not ACTIVE, allows no RESTART_FABRIC_MANAGER, or "
            "runs a ledger whose schema predates ARCH-C4",
            "/healthz is not 200 with a writable ledger before the restart",
            "a fail-safe restore timer from an earlier run is still armed",
            "any remote command or processor work is open when the restart is due",
            "the Agent does not return active, or returns with a new generation "
            "or a new incarnation",
            "the workflow is not the two-step Fabric Manager restart or reaches "
            "any isolation, quiesce, reset or reboot operation",
            "provider mutation appears",
            "cleanup leaves the fail-safe armed, the Agent down, Fabric Manager "
            "inactive or a probe resource behind",
        ],
        "rollback": {
            "fail_safe_start_timer_is_armed_before_the_restart": True,
            "agent_restart_never_stops_without_a_scheduled_start": True,
            "no_command_is_in_flight_when_the_agent_restarts": True,
            "migration_drill_uses_a_scratch_ledger_only": True,
            "fabric_manager_is_confirmed_active_in_finally": True,
            "runner_finally_deletes_probe_resources": True,
            "no_isolation_or_provider_action_is_authorized": True,
        },
        "preflight": preflight,
    }


def verify_plan_identity(case_dir: Path, preflight: dict[str, Any]) -> None:
    plan = json.loads((case_dir / "plan.json").read_text(encoding="utf-8"))
    planned = plan["details"]["preflight_identity"]
    current = plan_identity(preflight)
    if current != planned:
        raise RegionalFixtureError(f"{CASE_ID} plan drifted: {planned} != {current}")


@dataclass
class _LiveRun:
    settings: Settings
    regional: RegionalLiveFixture
    case_dir: Path
    preflight: dict[str, Any]
    run_id: str
    agent_probe: HostProbeFixture
    injector: HostProbeFixture
    marker: str = ""
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    restarted_at: datetime | None = None
    injected_at: datetime | None = None
    restore_armed: bool = False
    baseline_host: dict[str, Any] = field(default_factory=dict)
    baseline_node: dict[str, Any] = field(default_factory=dict)
    baseline_agent: dict[str, Any] = field(default_factory=dict)
    incident_id: str = ""
    workflow_request_id: str = ""


def _prepare_live_run(settings: Settings, run_dir: Path, attempt: int) -> _LiveRun:
    case_dir = run_dir / "cases" / CASE_ID
    case_dir.mkdir(parents=True, exist_ok=True)
    preflight = read_only_preflight(settings, case_dir)
    if preflight["errors"]:
        raise RegionalFixtureError(
            "preflight failed: " + "; ".join(preflight["errors"])
        )
    verify_plan_identity(case_dir, preflight)
    run_id = f"destr019-{run_dir.name.rsplit('-', 1)[-1].lower()}-a{attempt}"
    run = _LiveRun(
        settings=settings,
        regional=RegionalLiveFixture(settings.regional),
        case_dir=case_dir,
        preflight=preflight,
        run_id=run_id,
        agent_probe=_probe(settings, f"{run_id}-agent", AGENT_PROBE_SCRIPT),
        injector=_probe(settings, f"{run_id}-inject", INJECT_PROBE_SCRIPT),
    )
    run.marker = f"destr019-{int(time.time())}-a{attempt}"
    run.baseline_node = dict(preflight["node"])
    run.baseline_agent = dict(preflight["store"].get("agent") or {})
    return run


def _baseline(run: _LiveRun) -> dict[str, Any]:
    run.agent_probe.create()
    run.injector.create()
    host = run.agent_probe.execute("snapshot", "--run-id", run.run_id, timeout=120)
    write_json_atomic(run.case_dir / "host-baseline.json", host)
    run.baseline_host = host
    fabric = run.injector.execute("snapshot", timeout=120)
    write_json_atomic(run.case_dir / "fabric-baseline.json", fabric)
    if not fabric.get("kmsg_writable"):
        raise RegionalFixtureError("/dev/kmsg is not writable from the host probe")
    if fabric.get("compute_clients"):
        raise RegionalFixtureError("target node has active NVIDIA compute clients")
    if (fabric.get("fabric_manager") or {}).get("ActiveState") != "active":
        raise RegionalFixtureError("Fabric Manager is not active at baseline")
    health = run.agent_probe.execute("agent-health", timeout=60)
    write_json_atomic(run.case_dir / "health-before-restart.json", health)
    return health


def _migration_drill(run: _LiveRun) -> list[str]:
    report = run.agent_probe.execute(
        "migration-drill", "--run-id", run.run_id, timeout=180
    )
    write_json_atomic(run.case_dir / "migration-drill.json", report)
    return verdicts.migration_errors(report)


def _quiet_control_plane(run: _LiveRun) -> None:
    """Refuse to restart while anything is open for this cluster.

    The restart is only safe with no command in flight; the preflight said so
    minutes ago, and the moment before the restart has to say so again.
    """

    state = run.regional.store_snapshot(node=run.settings.node, queue_attempts=1)
    write_json_atomic(run.case_dir / "store-before-restart.json", state)
    if (state.get("remote_commands") or {}).get("open_by_cluster"):
        raise RegionalFixtureError(
            "remote commands are open; an Agent restart could interrupt one"
        )
    if int((state.get("queue") or {}).get("depth") or 0):
        raise RegionalFixtureError(
            "the processor queue is not empty before the restart"
        )


def _restart(run: _LiveRun) -> list[str]:
    _quiet_control_plane(run)
    run.restarted_at = datetime.now(timezone.utc)
    run.restore_armed = True
    restart = run.agent_probe.execute(
        "restart-agent",
        "--run-id",
        run.run_id,
        "--restore-seconds",
        str(run.settings.restore_seconds),
        timeout=180,
    )
    write_json_atomic(run.case_dir / "agent-restart.json", restart)
    errors = verdicts.restart_errors(
        restart, baseline_boot_id=str(run.baseline_host.get("boot_id") or "")
    )
    agent = _wait_agent_heartbeat(run)
    write_json_atomic(run.case_dir / "agent-after-restart.json", agent)
    errors.extend(
        verdicts.agent_record_errors(
            run.baseline_agent, agent, restarted_at=run.restarted_at
        )
    )
    health = run.agent_probe.execute("agent-health", timeout=60)
    write_json_atomic(run.case_dir / "health-after-restart.json", health)
    errors.extend(
        verdicts.health_errors(
            health,
            label="after restart",
            expect_counters=dict.fromkeys(verdicts.COUNTER_NAMES, 0),
            max_heartbeat_age_seconds=120,
        )
    )
    return errors


def _wait_agent_heartbeat(run: _LiveRun) -> dict[str, Any]:
    if run.restarted_at is None:
        raise RegionalFixtureError("the heartbeat wait follows the restart")
    deadline = time.monotonic() + verdicts.AGENT_HEARTBEAT_TIMEOUT_SECONDS
    agent: dict[str, Any] = {}
    while time.monotonic() < deadline:
        agent = (
            run.regional.store_snapshot(node=run.settings.node, queue_attempts=1).get(
                "agent"
            )
            or {}
        )
        seen = verdicts.parse_time(agent.get("last_seen_at"))
        if (
            agent.get("lifecycle_state") == "ACTIVE"
            and seen is not None
            and seen >= run.restarted_at
        ):
            return agent
        time.sleep(5)
    raise RegionalFixtureError(
        f"the restarted Node Agent never heartbeated to the control plane: {agent}"
    )


def _inject_and_observe(run: _LiveRun) -> dict[str, Any]:
    pci_bdf = str(
        run.injector.execute("snapshot", timeout=120).get("gpu_pci_bdf") or ""
    )
    run.injected_at = datetime.now(timezone.utc)
    injection = run.injector.execute(
        "write-xid45",
        "--marker",
        run.marker,
        "--drill-id",
        run.run_id,
        "--pci-bdf",
        pci_bdf,
    )
    write_json_atomic(run.case_dir / "injection.json", injection)
    state = run.regional.wait_for_workflow(
        node=run.settings.node,
        marker=run.marker,
        observed_after=run.injected_at,
        case_dir=run.case_dir,
        timeout_seconds=verdicts.WORKFLOW_TIMEOUT_SECONDS,
    )
    write_json_atomic(run.case_dir / "workflow-state.json", state)
    run.incident_id = str((state.get("incident") or {}).get("incident_id") or "")
    run.workflow_request_id = str((state.get("workflow") or {}).get("request_id") or "")
    return state


def _audit_errors(run: _LiveRun, state: dict[str, Any]) -> list[str]:
    if run.restarted_at is None:
        raise RegionalFixtureError("the audit follows the restart")
    errors = verdicts.workflow_errors(state, node=run.settings.node)
    commands = state.get("commands") or []
    command = commands[0] if commands else {}
    workflow = state.get("workflow") or {}
    generation = int(run.baseline_agent.get("generation") or 0)
    command_id = verdicts.expected_command_id(
        command, node=run.settings.node, generation=generation
    )
    journal = run.agent_probe.execute(
        "journal",
        "--since-epoch",
        str(run.restarted_at.timestamp()),
        "--command-id",
        command_id,
        timeout=120,
    )
    write_json_atomic(run.case_dir / "journal.json", journal)
    errors.extend(
        verdicts.journal_errors(
            journal.get("lines") or [],
            command_id=command_id,
            incident_id=run.incident_id,
            workflow_request_id=run.workflow_request_id,
            node=run.settings.node,
            fencing_token=int(workflow.get("fencing_token") or 0),
            generation=generation,
        )
    )
    audit = run.agent_probe.execute(
        "ledger-audit", "--command-id", command_id, timeout=120
    )
    write_json_atomic(run.case_dir / "ledger-audit.json", audit)
    errors.extend(
        verdicts.ledger_row_errors(
            audit,
            command_id=command_id,
            incident_id=run.incident_id,
            workflow_request_id=run.workflow_request_id,
            fencing_token=int(workflow.get("fencing_token") or 0),
            baseline_interrupted=int(
                (run.baseline_host.get("ledger") or {}).get("interrupted_count") or 0
            ),
        )
    )
    health = run.agent_probe.execute("agent-health", timeout=60)
    write_json_atomic(run.case_dir / "health-after-command.json", health)
    errors.extend(
        verdicts.health_errors(
            health,
            label="after command",
            expect_counters={"accepted": 1, "completed": 1, "failed": 0, "rejected": 0},
            max_heartbeat_age_seconds=120,
        )
    )
    return errors


def _provider_errors(run: _LiveRun) -> list[str]:
    events = run.regional.provider_events(run.started_at, datetime.now(timezone.utc))
    write_json_atomic(run.case_dir / "provider-events.json", {"events": events})
    return ["provider mutation appeared during DESTR-019"] if events else []


def execute_case(
    settings: Settings,
    run_dir: Path,
    attempt: int,
    maintenance_window_end: datetime,
) -> int:
    """Drive the live case. Not exercised by the unit suite; the verdicts it
    calls are. Every cleanup failure downgrades the verdict to FAIL."""

    run = _prepare_live_run(settings, run_dir, attempt)
    result: dict[str, Any] = {
        "case_id": CASE_ID,
        "attempt": attempt,
        "verdict": "FAIL",
        "node": settings.node,
        "maintenance_window_end": maintenance_window_end.isoformat(),
    }
    try:
        health_before = _baseline(run)
        errors = verdicts.health_errors(health_before, label="before restart")
        errors.extend(_migration_drill(run))
        if datetime.now(timezone.utc) >= maintenance_window_end:
            raise RegionalFixtureError(
                "approved maintenance window ended before restart"
            )
        errors.extend(_restart(run))
        state = _inject_and_observe(run)
        errors.extend(_audit_errors(run, state))
        errors.extend(_provider_errors(run))
        result.update(
            {
                "verdict": "PASS" if not errors else "FAIL",
                "errors": errors,
                "marker": run.marker,
                "incident_id": run.incident_id,
                "workflow_request_id": run.workflow_request_id,
                "restarted_at": run.restarted_at.isoformat()
                if run.restarted_at
                else None,
                "injected_at": run.injected_at.isoformat() if run.injected_at else None,
                "agent_generation": run.baseline_agent.get("generation"),
            }
        )
    except Exception as exc:  # noqa: BLE001 - recorded as the case error
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        cleanup = _cleanup(run)
        result["cleanup"] = cleanup
        if cleanup["errors"]:
            result["verdict"] = "FAIL"
    write_json_atomic(run.case_dir / f"{CASE_ID}.json", result)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["verdict"] == "PASS" else 1


def _refuse_residual_map(residuals: dict[str, bool]) -> dict[str, bool]:
    if any(residuals.values()):
        raise RegionalFixtureError(f"residual resources remain: {residuals}")
    return residuals


def _cleanup(run: _LiveRun) -> dict[str, Any]:
    result: dict[str, Any] = {"errors": []}

    def guard(label: str, action: Any) -> None:
        try:
            result[label] = action()
        except Exception as exc:  # noqa: BLE001 - a cleanup failure is a FAIL
            result["errors"].append(f"{label}: {type(exc).__name__}: {exc}")

    guard(
        "agent_active",
        lambda: run.agent_probe.execute(
            "ensure-agent-active", "--run-id", run.run_id, timeout=180
        ),
    )
    if run.restore_armed:
        guard(
            "disarm_restore",
            lambda: _disarm(run),
        )
    guard(
        "fabric_manager_active",
        lambda: run.injector.execute("ensure-fabric-manager-active", timeout=120),
    )
    guard("final_node", lambda: _final_node(run))
    for label, probe in (("agent", run.agent_probe), ("injector", run.injector)):
        guard(
            f"probe_cleanup_{label}",
            lambda probe=probe: _refuse_residual_map(probe.cleanup()),
        )
    guard(
        "runtime_identity",
        lambda: run.regional.verify_runtime_identity(
            run.preflight["runtime_identity"],
            evidence_path=run.case_dir / "runtime-identity-after-cleanup.json",
            stage=f"after {CASE_ID} cleanup",
        ),
    )
    return result


def _disarm(run: _LiveRun) -> dict[str, Any]:
    disarm = run.agent_probe.execute(
        "disarm-restore", "--run-id", run.run_id, timeout=120
    )
    write_json_atomic(run.case_dir / "disarm-restore.json", disarm)
    errors = verdicts.cleanup_errors(
        disarm,
        baseline_timers=list(run.baseline_host.get("gpu_fault_timers") or []),
    )
    if errors:
        raise RegionalFixtureError("; ".join(errors))
    return disarm


def _final_node(run: _LiveRun) -> dict[str, Any]:
    snapshot = run.regional.node_snapshot(run.settings.node)
    write_json_atomic(run.case_dir / "node-final.json", snapshot)
    errors = verdicts.node_errors(run.baseline_node, snapshot)
    if run.regional.business_workloads(run.settings.node):
        errors.append("target node acquired a non-system workload")
    if errors:
        raise RegionalFixtureError("; ".join(errors))
    return snapshot


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        description=(
            "Run the guarded DESTR-019 acceptance: one Node Agent restart on an "
            "idle node, the ledger migration drill, and the audit trail of the "
            "next real command."
        )
    )
    add_live_arguments(value, confirmation=CONFIRMATION)
    value.add_argument("--cpu-kubeconfig", default="")
    value.add_argument("--gpu-kubeconfig", default="")
    value.add_argument("--gpu-context", default="")
    value.add_argument("--namespace", default="gpu-fault-system")
    value.add_argument("--cluster-id", default="")
    value.add_argument("--node", default="")
    value.add_argument("--region", default="")
    value.add_argument("--host-probe-image", default="")
    value.add_argument("--predecessor-evidence", default="")
    value.add_argument(
        "--restore-seconds",
        type=int,
        default=verdicts.RESTORE_SECONDS,
        help="fail-safe start of the Node Agent unit if the restart is abandoned",
    )
    return value


CASE = CaseRunner(
    case_id=CASE_ID,
    confirmation=CONFIRMATION,
    parser=parser,
    configure=configure,
    read_only_preflight=read_only_preflight,
    plan_details=plan_details,
    execute_case=execute_case,
)


def main() -> int:
    return run_standard_case(CASE)


if __name__ == "__main__":
    raise SystemExit(run_case_main(main))

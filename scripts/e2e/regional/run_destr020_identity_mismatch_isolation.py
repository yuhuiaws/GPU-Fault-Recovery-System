#!/usr/bin/env python3
"""GF-REGIONAL-DESTR-020: isolation is observed, not remembered.

One API-replay XID is posted for an alias node id that exists nowhere -- not
as a Kubernetes node, not as a registered Node Agent, not in any inventory.
The policy answers with the site-safety quarantine, so the workflow is
safety-only (``FREEZE_EVIDENCE -> MARK_UNSCHEDULABLE -> QUARANTINE``) and
plans no physical operation at all. The deployed executor's Kubernetes adapter
must then fail ``MARK_UNSCHEDULABLE`` closed -- ``safety_rejection`` +
``absent`` -- instead of treating a 404 as "already isolated", the workflow
must fail, the incident must escalate to a human exactly once, and no real
node's scheduling state may move.

Hard stop: the alias has no Node Agent (a reset is a Node Agent command) and
names no HyperPod instance (a reboot needs one resolved from the inventory),
and the plan contains neither anyway. Nothing on this path can be armed.

The case leaves control records behind by design -- a FAILED incident and its
support escalation -- and records their ids as ``leftover_records``. There is
no node to restore.
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

from scripts.e2e.regional import destr020_verdicts as verdicts  # noqa: E402
from scripts.e2e.regional import run_collect017_efa_plugin as c017  # noqa: E402
from scripts.e2e.regional.acceptance_runner_common import (  # noqa: E402
    write_json_atomic,
)
from scripts.e2e.regional.live_driver_guard import (  # noqa: E402
    add_live_arguments,
    authorize_execution,
    build_plan,
    install_site_profile,
)
from scripts.e2e.regional.regional_live_fixture import (  # noqa: E402
    RegionalFixtureError,
    RegionalLiveFixture,
    RegionalLiveSettings,
    install_abort_signals,
    predecessor_evidence,
    required,
    run_case_main,
    runtime_identity_errors,
    settings_from_arguments,
)
from scripts.e2e.regional.run_destr009_workload_restart import (  # noqa: E402
    normalize_product,
)

CASE_ID = verdicts.CASE_ID
PREDECESSOR_CASE_ID = verdicts.PREDECESSOR_CASE_ID
CONFIRMATION = verdicts.CONFIRMATION

# The support escalation ``orchestration/escalation.py`` emits for a FAILED
# workflow is keyed ``support-after-<request_id>``; a second-order one would
# be keyed after the support workflow itself (DESTR-018 reads the same chain).
# Support terminates the chain, so the second-order pair must stay absent.
ESCALATION_CHAIN = r"""
import json
import sys

from gpu_fault.app import ApplicationContext
from gpu_fault.store import NotFoundError

store = ApplicationContext.from_environment().store
workflow_id = sys.argv[1]


def incident(incident_id):
    try:
        return store.get_incident(incident_id).model_dump(mode="json")
    except (NotFoundError, KeyError):
        return None


def workflow(request_id):
    try:
        return store.get_workflow(request_id).model_dump(mode="json")
    except (NotFoundError, KeyError):
        return None


first = f"workflow-support-after-{workflow_id}"
commands = []
for item in store.list_remote_commands(workflow_request_ids=[first]):
    payload = item.model_dump(mode="json")
    commands.append({
        "command_id": payload.get("command_id"),
        "operation": item.step.operation.value,
        "status": payload.get("status"),
        "error": payload.get("error"),
        "result_details": payload.get("result_details") or {},
    })
print(
    json.dumps(
        {
            "incident": incident(f"inc-support-after-{workflow_id}"),
            "workflow": workflow(first),
            "remote_commands": commands,
            "second_order_incident": incident(f"inc-support-after-{first}"),
            "second_order_workflow": workflow(f"workflow-support-after-{first}"),
        },
        sort_keys=True,
        default=str,
    )
)
"""

FLEET_AGENTS = r"""
import json
import sys

from gpu_fault.app import ApplicationContext

cluster_id = sys.argv[1]
store = ApplicationContext.from_environment().store
print(
    json.dumps(
        {
            "agents": [
                {
                    "node_id": item.node_id,
                    "lifecycle_state": item.lifecycle_state.value,
                }
                for item in store.list_agents(cluster_id)
            ]
        },
        sort_keys=True,
        default=str,
    )
)
"""


@dataclass(frozen=True)
class Settings:
    regional: RegionalLiveSettings
    reference_node: str
    predecessor_path: Path
    chain_watch_seconds: int

    def environment(self) -> dict[str, str]:
        return {
            **self.regional.environment(),
            "GPU_FAULT_REFERENCE_NODE": self.reference_node,
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
    if not 30 <= int(arguments.chain_watch_seconds) <= 900:
        raise RegionalFixtureError("chain watch seconds is outside 30..900")
    return Settings(
        regional=settings_from_arguments(arguments),
        reference_node=required(
            arguments.reference_node or os.getenv("GPU_FAULT_REFERENCE_NODE", ""),
            "reference node",
        ),
        predecessor_path=predecessor,
        chain_watch_seconds=int(arguments.chain_watch_seconds),
    )


def run_identity(run_dir: Path, attempt: int) -> str:
    return f"destr020-{run_dir.name.rsplit('-', 1)[-1].lower()}-a{attempt}"


def focused_tests(case_dir: Path) -> dict[str, Any]:
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "tests/execution/test_kubernetes.py::"
        "test_kubernetes_adapter_fails_closed_when_the_node_to_isolate_is_absent",
        "tests/regional/test_destr020_identity_mismatch_isolation.py",
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


def kubernetes_node_present(fixture: RegionalLiveFixture, name: str) -> bool:
    output = fixture.kubectl(
        "gpu",
        "get",
        "node",
        name,
        "--ignore-not-found",
        "-o",
        "name",
        check=False,
        timeout=60,
    )
    return bool(output.strip())


def fleet_snapshot(fixture: RegionalLiveFixture) -> dict[str, dict[str, Any]]:
    """Every GPU node's scheduling state, keyed by name."""

    result: dict[str, dict[str, Any]] = {}
    for item in fixture.gpu_nodes():
        snapshot = fixture.node_snapshot(str(item["name"]))
        result[str(item["name"])] = {
            "unschedulable": snapshot.get("unschedulable"),
            "taints": snapshot.get("taints"),
            "ownership_annotations": snapshot.get("ownership_annotations"),
        }
    return result


def alias_facts(
    fixture: RegionalLiveFixture,
    alias: str,
    *,
    fleet: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    alias_state = fixture.store_snapshot(node=alias, queue_attempts=1)
    agents = fixture.cpu_python(FLEET_AGENTS, fixture.settings.cluster_id)
    return {
        "alias": alias,
        "gpu_node_names": sorted(fleet),
        "kubernetes_node_present": kubernetes_node_present(fixture, alias),
        "agent": alias_state.get("agent"),
        "agents": agents.get("agents") or [],
        "event": alias_state.get("event"),
    }


def read_only_preflight(
    settings: Settings, case_dir: Path, *, alias: str
) -> dict[str, Any]:
    fixture = RegionalLiveFixture(settings.regional)
    state = fixture.store_snapshot(
        node=settings.reference_node,
        observed_after=datetime.now(timezone.utc) - timedelta(minutes=10),
    )
    node = fixture.node_snapshot(settings.reference_node)
    workloads = fixture.business_workloads(settings.reference_node)
    fleet = fleet_snapshot(fixture)
    facts = alias_facts(fixture, alias, fleet=fleet)
    tests = focused_tests(case_dir)
    predecessor = predecessor_evidence(settings.predecessor_path, PREDECESSOR_CASE_ID)
    runtime_identity = fixture.runtime_identity()
    errors = verdicts.preflight_errors(
        alias=alias,
        alias_facts=facts,
        reference_node=node,
        reference_agent=state.get("agent") or {},
        reference_profile=state.get("profile") or {},
        reference_workloads=workloads,
        queue=state.get("queue") or {},
        remote_commands=state.get("remote_commands") or {},
        alias_event=facts.get("event"),
        tests_passed=tests["passed"],
    )
    errors.extend(runtime_identity_errors(runtime_identity))
    if not predecessor["valid"]:
        errors.append(f"{PREDECESSOR_CASE_ID} predecessor evidence is not PASS")
    result = {
        "release_id": state.get("release_id"),
        "alias": alias,
        "alias_facts": facts,
        "reference_node": node,
        "reference_workloads": workloads,
        "store": state,
        "fleet": fleet,
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
        "alias": preflight["alias"],
        "reference_node": preflight["reference_node"]["name"],
        "reference_node_uid": preflight["reference_node"]["uid"],
        "runtime_profile_version": (state.get("agent") or {}).get(
            "runtime_profile_version"
        ),
        "gpu_node_names": preflight["alias_facts"]["gpu_node_names"],
    }


def plan_details(settings: Settings, preflight: dict[str, Any]) -> dict[str, Any]:
    return {
        "risk": "live-non-destructive",
        "predecessor": preflight["predecessor"],
        "alias": preflight["alias"],
        "reference_node": settings.reference_node,
        "injected_xid": verdicts.INJECTED_XID,
        "chain_watch_seconds": settings.chain_watch_seconds,
        "mutation": (
            "post one API-replay kernel XID absent from the NVIDIA catalog for an "
            "alias node id that is not a Kubernetes node, not a Node Agent and "
            "not a provider instance; the site-safety quarantine workflow it "
            "opens must fail closed at MARK_UNSCHEDULABLE and escalate once. "
            "No node is touched, no reset or reboot is planned or reachable."
        ),
        "preflight_identity": plan_identity(preflight),
        "stop_conditions": [
            f"{PREDECESSOR_CASE_ID} has not passed in formal sequence",
            "preflight or focused regression failure",
            "the alias resolves to a Kubernetes node, a Node Agent or a fleet entry",
            "the reference node is not Ready, idle and free of ownership",
            "any remote command or processor work is open when the XID is due",
            "the policy resolves the alias event to an executable action",
            "MARK_UNSCHEDULABLE succeeds, QUARANTINE runs or any physical "
            "operation is reached",
            "any real GPU node's scheduling state moves",
            "the support escalation is not containment_refused, plans an "
            "isolation on the alias again or does not SUCCEED",
            "the failed isolation escalates its own escalation (second-order "
            "chain, depth > 1)",
            "provider mutation appears",
        ],
        "rollback": {
            "alias_has_no_node_agent_so_no_reset_command_can_be_addressed": True,
            "alias_names_no_provider_instance_so_no_reboot_can_be_submitted": True,
            "plan_contains_no_physical_operation": True,
            "no_node_is_mutated_so_nothing_is_restored": True,
            "leftover_control_records_are_recorded_by_id": True,
        },
        "preflight": preflight,
    }


def verify_plan_identity(case_dir: Path, preflight: dict[str, Any]) -> None:
    plan = json.loads((case_dir / "plan.json").read_text(encoding="utf-8"))
    planned = plan["details"]["preflight_identity"]
    current = plan_identity(preflight)
    if current != planned:
        raise RegionalFixtureError(f"{CASE_ID} plan drifted: {planned} != {current}")


def xid_payload(
    settings: Settings,
    *,
    alias: str,
    marker: str,
    product: str,
    runtime_profile_version: str,
    observed_at: datetime,
) -> dict[str, Any]:
    return {
        "cluster_id": settings.regional.cluster_id,
        "node_id": alias,
        "record_id": marker,
        "observed_at": observed_at.isoformat(),
        "message": (
            f"NVRM: Xid (PCI:0000:b9:00): {verdicts.INJECTED_XID}, "
            f"identity-mismatch acceptance event, marker={marker}"
        ),
        "runtime_profile_version": runtime_profile_version,
        "product": product,
        "workload_state": "IDLE",
        "affected_workload_ids": [],
        "evidence_ref": f"api-replay://{CASE_ID}/{marker}",
    }


@dataclass
class _LiveRun:
    settings: Settings
    regional: RegionalLiveFixture
    case_dir: Path
    preflight: dict[str, Any]
    run_id: str
    alias: str
    marker: str = ""
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    injected_at: datetime | None = None
    fleet_before: dict[str, dict[str, Any]] = field(default_factory=dict)
    incident_id: str = ""
    workflow_request_id: str = ""
    chain: dict[str, Any] = field(default_factory=dict)
    state: dict[str, Any] = field(default_factory=dict)


def _prepare_live_run(settings: Settings, run_dir: Path, attempt: int) -> _LiveRun:
    case_dir = run_dir / "cases" / CASE_ID
    case_dir.mkdir(parents=True, exist_ok=True)
    run_id = run_identity(run_dir, attempt)
    alias = verdicts.alias_for_run(run_id)
    preflight = read_only_preflight(settings, case_dir, alias=alias)
    if preflight["errors"]:
        raise RegionalFixtureError(
            "preflight failed: " + "; ".join(preflight["errors"])
        )
    verify_plan_identity(case_dir, preflight)
    run = _LiveRun(
        settings=settings,
        regional=RegionalLiveFixture(settings.regional),
        case_dir=case_dir,
        preflight=preflight,
        run_id=run_id,
        alias=alias,
    )
    run.marker = f"destr020-{int(time.time())}-a{attempt}"
    run.fleet_before = dict(preflight["fleet"])
    write_json_atomic(case_dir / "fleet-before.json", run.fleet_before)
    return run


def _quiet_control_plane(run: _LiveRun) -> None:
    """Re-check, the moment before the XID, that nothing else is in flight."""

    state = run.regional.store_snapshot(
        node=run.settings.reference_node, queue_attempts=1
    )
    write_json_atomic(run.case_dir / "store-before-injection.json", state)
    if (state.get("remote_commands") or {}).get("open_by_cluster"):
        raise RegionalFixtureError(
            "remote commands are open; the alias workflow must not queue behind them"
        )
    if int((state.get("queue") or {}).get("depth") or 0):
        raise RegionalFixtureError(
            "the processor queue is not empty before the injection"
        )


def _inject_and_observe(run: _LiveRun) -> dict[str, Any]:
    _quiet_control_plane(run)
    metadata = run.regional.node_metadata(run.settings.reference_node)
    product = normalize_product(metadata.get("product"))
    profile_version = str(
        (run.preflight["store"].get("agent") or {}).get("runtime_profile_version") or ""
    )
    run.injected_at = datetime.now(timezone.utc)
    payload = xid_payload(
        run.settings,
        alias=run.alias,
        marker=run.marker,
        product=product,
        runtime_profile_version=profile_version,
        observed_at=run.injected_at,
    )
    injection = run.regional.post_xid_event(payload)
    write_json_atomic(
        run.case_dir / "injection.json", {"payload": payload, **injection}
    )
    errors = verdicts.injection_errors(injection)
    if errors:
        raise RegionalFixtureError("; ".join(errors))
    state = run.regional.wait_for_workflow(
        node=run.alias,
        marker=run.marker,
        observed_after=run.injected_at,
        case_dir=run.case_dir,
        timeout_seconds=verdicts.WORKFLOW_TIMEOUT_SECONDS,
    )
    write_json_atomic(run.case_dir / "workflow-state.json", state)
    run.state = state
    run.incident_id = str((state.get("incident") or {}).get("incident_id") or "")
    run.workflow_request_id = str((state.get("workflow") or {}).get("request_id") or "")
    return state


def _remote_commands(run: _LiveRun) -> list[dict[str, Any]]:
    if not run.workflow_request_id:
        return []
    bundle = run.regional.cpu_python(c017.REMOTE_COMMANDS, run.workflow_request_id)
    write_json_atomic(run.case_dir / "remote-commands.json", bundle)
    return list(bundle.get("remote_commands") or [])


TERMINAL_STATUSES = frozenset({"SUCCEEDED", "FAILED", "BLOCKED", "SUPERSEDED"})


def _escalation_chain(run: _LiveRun) -> dict[str, Any]:
    """Wait for the support escalation to finish, then watch for a second one.

    The watch is the case's most important negative assertion: an escalation
    that escalates itself is one new FAILED workflow per dispatcher pass.
    """

    if not run.workflow_request_id:
        return {}
    deadline = time.monotonic() + verdicts.ESCALATION_TIMEOUT_SECONDS
    chain: dict[str, Any] = {}
    while time.monotonic() < deadline:
        chain = run.regional.cpu_python(ESCALATION_CHAIN, run.workflow_request_id)
        support = chain.get("workflow") or {}
        if chain.get("incident") and support.get("status") in TERMINAL_STATUSES:
            break
        time.sleep(5)
    watch_until = time.monotonic() + run.settings.chain_watch_seconds
    while time.monotonic() < watch_until:
        if chain.get("second_order_workflow") or chain.get("second_order_incident"):
            break
        time.sleep(10)
        chain = run.regional.cpu_python(ESCALATION_CHAIN, run.workflow_request_id)
    chain["watched_seconds"] = run.settings.chain_watch_seconds
    write_json_atomic(run.case_dir / "escalation.json", chain)
    run.chain = chain
    return chain


def _provider_errors(run: _LiveRun) -> list[str]:
    events = run.regional.provider_events(run.started_at, datetime.now(timezone.utc))
    write_json_atomic(run.case_dir / "provider-events.json", {"events": events})
    return ["provider mutation appeared during DESTR-020"] if events else []


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
        "alias": run.alias,
        "reference_node": settings.reference_node,
        "maintenance_window_end": maintenance_window_end.isoformat(),
    }
    try:
        if datetime.now(timezone.utc) >= maintenance_window_end:
            raise RegionalFixtureError(
                "approved maintenance window ended before injection"
            )
        state = _inject_and_observe(run)
        errors = verdicts.decision_errors(
            state.get("decision"), state.get("event") or {}
        )
        errors.extend(verdicts.workflow_errors(state, alias=run.alias))
        errors.extend(
            verdicts.remote_command_errors(_remote_commands(run), alias=run.alias)
        )
        errors.extend(
            verdicts.escalation_errors(_escalation_chain(run), alias=run.alias)
        )
        errors.extend(_provider_errors(run))
        result.update(
            {
                "verdict": "PASS" if not errors else "FAIL",
                "errors": errors,
                "marker": run.marker,
                "incident_id": run.incident_id,
                "workflow_request_id": run.workflow_request_id,
                "injected_at": run.injected_at.isoformat() if run.injected_at else None,
                "leftover_records": verdicts.leftover_records(state, run.chain),
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


def _cleanup(run: _LiveRun) -> dict[str, Any]:
    """Nothing was mutated, so cleanup only proves that: fleet and identity."""

    result: dict[str, Any] = {"errors": []}

    def guard(label: str, action: Any) -> None:
        try:
            result[label] = action()
        except Exception as exc:  # noqa: BLE001 - a cleanup failure is a FAIL
            result["errors"].append(f"{label}: {type(exc).__name__}: {exc}")

    guard("fleet_after", lambda: _fleet_after(run))
    guard(
        "runtime_identity",
        lambda: run.regional.verify_runtime_identity(
            run.preflight["runtime_identity"],
            evidence_path=run.case_dir / "runtime-identity-after-cleanup.json",
            stage=f"after {CASE_ID} cleanup",
        ),
    )
    return result


def _fleet_after(run: _LiveRun) -> dict[str, dict[str, Any]]:
    after = fleet_snapshot(run.regional)
    write_json_atomic(run.case_dir / "fleet-after.json", after)
    errors = verdicts.fleet_errors(run.fleet_before, after, alias=run.alias)
    if errors:
        raise RegionalFixtureError("; ".join(errors))
    return after


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        description=(
            "Run the guarded DESTR-020 acceptance: one API-replay XID for an "
            "alias node the Kubernetes API does not know; the isolation must "
            "fail closed, nothing may be cordoned, and the incident must "
            "escalate exactly once."
        )
    )
    add_live_arguments(value, confirmation=CONFIRMATION)
    value.add_argument("--cpu-kubeconfig", default="")
    value.add_argument("--gpu-kubeconfig", default="")
    value.add_argument("--gpu-context", default="")
    value.add_argument("--namespace", default="gpu-fault-system")
    value.add_argument("--cluster-id", default="")
    value.add_argument("--region", default="")
    value.add_argument(
        "--reference-node",
        default="",
        help=(
            "a real, idle GPU node whose product and runtime profile version "
            "the alias event borrows; it is never targeted"
        ),
    )
    value.add_argument("--predecessor-evidence", default="")
    value.add_argument(
        "--chain-watch-seconds",
        type=int,
        default=verdicts.CHAIN_WATCH_SECONDS,
        help="how long to watch for a second-order escalation after the first",
    )
    return value


def main() -> int:
    install_site_profile()
    arguments = parser().parse_args()
    os.umask(0o077)
    install_abort_signals()
    settings = configure(arguments)
    case_dir = arguments.run_dir / "cases" / CASE_ID
    case_dir.mkdir(parents=True, exist_ok=True)
    if not arguments.execute:
        alias = verdicts.alias_for_run(
            run_identity(arguments.run_dir, arguments.attempt)
        )
        preflight = read_only_preflight(settings, case_dir, alias=alias)
        plan = build_plan(
            run_dir=arguments.run_dir,
            case_id=CASE_ID,
            attempt=arguments.attempt,
            confirmation=CONFIRMATION,
            environment=settings.environment(),
            details=plan_details(settings, preflight),
        )
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0 if not preflight["errors"] else 1
    deadline = authorize_execution(
        arguments,
        case_id=CASE_ID,
        confirmation=CONFIRMATION,
        environment=settings.environment(),
    )
    return execute_case(settings, arguments.run_dir, arguments.attempt, deadline)


if __name__ == "__main__":
    raise SystemExit(run_case_main(main))

#!/usr/bin/env python3
"""GF-REGIONAL-CMD-018: one open command per workflow step identity.

A remote command's id is a digest of the whole step. A merge that rewrites a
step's parameters while the command minted for the previous step is still
open would, before ARCH-D5, mint a second command for the same physical
action. ``RegionalRemoteWorkflowAdapter.execute`` now asks the store for an
open sibling of the same (workflow, step index, step space) under the same
fencing token and holds WAITING with ``reason=OPEN_SIBLING_COMMAND`` instead.

This case drives the *deployed* adapter inside a CPU API Pod against the live
store for a workflow of the synthetic cluster ``perf-cap-000``: dispatch once
(command A), rewrite the step's parameters through ``save_workflow`` with the
read copy as ``expected``, dispatch again (held, A named, a different id
withheld), then let a probe executor whose action is a local ledger claim A
and finish it, then dispatch the rewritten step once more (B is minted: the
hold was scoped to *open* siblings) and cancel B before anything can claim it.
No node can receive either command. Plan-only by default.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.e2e.regional import cmd018_verdicts as verdicts  # noqa: E402
from scripts.e2e.regional import seeded_command_fixture as seeded  # noqa: E402
from scripts.e2e.regional.live_driver_guard import (  # noqa: E402
    PlainCaseRunner,
    add_live_arguments,
    run_plain_case,
)
from scripts.perf.regional_capacity_registry import (  # noqa: E402
    STORE_DSN_SNIPPET,
)

CASE_ID = verdicts.CASE_ID
CONFIRMATION = verdicts.CONFIRMATION
SCRIPT = Path(__file__).with_name("probes") / "cmd018_ledger_executor.py"

# Runs inside the CPU API Pod: the deployed adapter, the deployed store.
_DISPATCH_AND_HOLD = r"""
import json
import sys
from datetime import datetime, timedelta, timezone

from gpu_fault.app import ApplicationContext
from gpu_fault.execution import WorkflowStepContext
from gpu_fault.models import (
    FaultIncident,
    IncidentState,
    WorkflowExecutionRequest,
    WorkflowOperation,
    WorkflowRequest,
    WorkflowStatus,
    WorkflowStepSpec,
)
from gpu_fault.regional import RegionalRemoteWorkflowAdapter

run_id, cluster_id, owner, operation_name, raw_node_ids, raw_lease_seconds = sys.argv[1:]
lease_seconds = int(raw_lease_seconds)
node_ids = [item for item in raw_node_ids.split(",") if item]
operation = WorkflowOperation(operation_name)
incident_id = f"incident-{run_id}"
workflow_id = f"workflow-actionperf-{run_id}"
step_v1 = WorkflowStepSpec(
    operation=operation, execution_owner=owner, node_ids=node_ids,
    parameters={"acceptance_revision": "v1"},
)
incident = FaultIncident(
    incident_id=incident_id,
    event_id=f"event-{run_id}",
    event_type="CMD_ACCEPTANCE",
    cluster_id=cluster_id,
    node_ids=node_ids,
    policy_version="cmd-acceptance/v1",
    policy_source="ACCEPTANCE",
    state=IncidentState.ACTION_PENDING,
    workflow_request_id=workflow_id,
    fencing_token=1,
    drill_id=run_id,
)
# The workflow is RUNNING so the deployed adapter dispatches for it, and it
# carries a live execution lease owned by the probe: an unleased RUNNING
# workflow is claimed by the deployed dispatcher, whose executor then drives
# the step itself -- attempt 1 (2026-09-09) saw the workflow FAILED and command
# A cancelled as an orphan within 46 s, before the probe had claimed once.
# Only the probe may own this step; the lease outlives the Pod deadline.
workflow = WorkflowRequest(
    request_id=workflow_id,
    incident_id=incident_id,
    status=WorkflowStatus.RUNNING,
    official_action="NO_ACTION",
    fencing_token=1,
    official_steps=[step_v1],
    execution_owner_id=f"{owner}-seed",
    execution_epoch=1,
    execution_lease_expires_at=datetime.now(timezone.utc) + timedelta(seconds=lease_seconds),
)
store = ApplicationContext.from_environment().store
store.save_incident_and_workflow(incident, workflow)
adapter = RegionalRemoteWorkflowAdapter(store, owners={owner})


def context(current, step):
    return WorkflowStepContext(
        workflow=current,
        incident=incident,
        step=step,
        step_index=0,
        request=WorkflowExecutionRequest(expected_fencing_token=1),
        idempotency_key=f"{workflow_id}/0/{operation.value}",
    )


first = adapter.execute(context(workflow, step_v1))
# The sanctioned rewrite: the same generation, new parameters, guarded write.
read_copy = store.get_workflow(workflow_id)
step_v2 = step_v1.model_copy(update={"parameters": {"acceptance_revision": "v2"}})
rewritten = read_copy.model_copy(
    update={"official_steps": [step_v2], "updated_at": datetime.now(timezone.utc)}
)
store.save_workflow(rewritten, expected=read_copy)
held = adapter.execute(context(store.get_workflow(workflow_id), step_v2))
open_ids = [
    item.command_id
    for item in store.list_remote_commands(workflow_request_ids=[workflow_id])
    if item.status.value in {"PENDING", "LEASED", "WAITING"}
]
print(json.dumps({
    "incident_id": incident_id,
    "event_id": incident.event_id,
    "workflow_id": workflow_id,
    "first": {"status": first.status.value, **(first.details or {})},
    "held": {"status": held.status.value, "details": held.details or {}},
    "open_sibling_holds_total": adapter.open_sibling_holds_total,
    "open_command_ids": open_ids,
    "all_command_ids": [
        item.command_id
        for item in store.list_remote_commands(workflow_request_ids=[workflow_id])
    ],
    "registered_agents": [item.node_id for item in store.list_agents(cluster_id)],
}, sort_keys=True, default=str))
"""

_WORKFLOW_FINAL = r"""
import json
import sys

from gpu_fault.app import ApplicationContext

workflow = ApplicationContext.from_environment().store.get_workflow(sys.argv[1])
document = workflow.model_dump(mode="json")
print(json.dumps({
    "status": document.get("status"),
    "terminal_failure_reason": document.get("terminal_failure_reason"),
    "blocked_reasons": document.get("blocked_reasons"),
    "execution_owner_id": document.get("execution_owner_id"),
    "execution_lease_expires_at": document.get("execution_lease_expires_at"),
    "step_executions": document.get("step_executions"),
    "events": (document.get("events") or [])[-20:],
}, sort_keys=True, default=str))
"""


_RELEASE_AND_CANCEL = r"""
import json
import sys

from gpu_fault.app import ApplicationContext
from gpu_fault.execution import WorkflowStepContext
from gpu_fault.models import WorkflowExecutionRequest
from gpu_fault.regional import RegionalRemoteWorkflowAdapter

workflow_id, owner, reason = sys.argv[1:]
store = ApplicationContext.from_environment().store
workflow = store.get_workflow(workflow_id)
incident = store.get_incident(workflow.incident_id)
step = workflow.official_steps[0]
adapter = RegionalRemoteWorkflowAdapter(store, owners={owner})
outcome = adapter.execute(
    WorkflowStepContext(
        workflow=workflow,
        incident=incident,
        step=step,
        step_index=0,
        request=WorkflowExecutionRequest(expected_fencing_token=workflow.fencing_token),
        idempotency_key=f"{workflow_id}/0/{step.operation.value}",
    )
)
minted = str((outcome.details or {}).get("remote_command_id") or "")
cancelled = False
cancelled_command = None
if minted and (outcome.details or {}).get("reason") != "OPEN_SIBLING_COMMAND":
    cancelled = store.cancel_remote_command(minted, reason=reason)
    command = store.get_remote_command(minted)
    cancelled_command = {
        "command_id": command.command_id,
        "status": command.status.value,
        "status_source": command.status_source,
    }
print(json.dumps({
    "outcome": {"status": outcome.status.value, "details": outcome.details or {}},
    "cancelled": cancelled,
    "cancelled_command": cancelled_command,
    "all_command_ids": [
        item.command_id
        for item in store.list_remote_commands(workflow_request_ids=[workflow_id])
    ],
}, sort_keys=True, default=str))
"""

_PURGE = (
    STORE_DSN_SNIPPET
    + r"""
import json
import sys
import psycopg

workflow_id, incident_id, event_id, *command_ids = sys.argv[1:]
deleted = {}
with psycopg.connect(store_dsn(), autocommit=True) as connection:
    cursor = connection.cursor()
    cursor.execute(
        "DELETE FROM gpu_fault_links WHERE kind='incident_by_event' AND key=%s AND value=%s",
        (event_id, incident_id),
    )
    deleted[f"incident_by_event/{event_id}"] = cursor.rowcount
    for command_id in command_ids:
        cursor.execute(
            "DELETE FROM gpu_fault_objects WHERE kind='remote_command' AND key=%s",
            (command_id,),
        )
        deleted[f"remote_command/{command_id}"] = cursor.rowcount
    for kind, key in (("workflow", workflow_id), ("incident", incident_id)):
        cursor.execute("DELETE FROM gpu_fault_objects WHERE kind=%s AND key=%s", (kind, key))
        deleted[f"{kind}/{key}"] = cursor.rowcount
    cursor.execute(
        "SELECT count(*) FROM gpu_fault_objects WHERE kind='remote_command' "
        "AND payload->>'workflow_request_id'=%s",
        (workflow_id,),
    )
    remaining = int(cursor.fetchone()[0])
print(json.dumps({"deleted": deleted, "remaining_commands": remaining}, sort_keys=True))
"""
)


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
        "the seeded cluster has a registered Node Agent",
        "the rewritten dispatch is not held with reason OPEN_SIBLING_COMMAND",
        "more than one open command exists for the step",
        "the ledger records more than one attempt",
        "the command minted after the release cannot be cancelled",
        "any cleanup or postflight residual check fails",
    ]


def rollback_contract() -> dict[str, Any]:
    return {
        "pod_active_deadline_seconds": verdicts.POD_DEADLINE_SECONDS,
        "synthetic_registry_expires_minutes": 30,
        "runner_finally_deletes_test_resources": True,
        "runner_finally_purges_synthetic_state": True,
        "runner_finally_restores_registry": True,
        "no_node_agent_can_receive_the_seeded_commands": True,
    }


def plan_details() -> dict[str, Any]:
    return {
        "risk": "live-non-destructive",
        "synthetic_cluster_id": seeded.SYNTHETIC_CLUSTER_ID,
        "seeded_operation": verdicts.OPERATION,
        "seeded_node_ids": verdicts.NODE_IDS,
        "expected_hold_reason": verdicts.HOLD_REASON,
        "mutations": [
            "temporary synthetic registry entry",
            "one incident, one workflow and two remote commands for the synthetic cluster",
            "temporary GPU probe Pod and ConfigMap",
        ],
        "hard_stop": (
            "both commands belong to a synthetic cluster with no Node Agents and name "
            "a node id no cluster carries; the only adapter that owns the step is a "
            "local ledger"
        ),
        "stop_conditions": stop_conditions(),
        "rollback": rollback_contract(),
    }


def _run_case(
    probe: seeded.SeededCommandProbe,
    case_dir: Path,
    run_id: str,
    attempt: int,
    maintenance_window_end: datetime,
    state: dict[str, Any],
) -> dict[str, Any]:
    if datetime.now(timezone.utc) >= maintenance_window_end:
        raise seeded.SeededCommandError("approved maintenance window has ended")
    seeded.require_environment()
    seeded.preflight_residuals(probe, case_dir)
    seeded.register_synthetic_cluster(case_dir, run_id)
    stages: dict[str, list[str]] = {}

    dispatch = seeded.cpu_python(
        _DISPATCH_AND_HOLD,
        run_id,
        seeded.SYNTHETIC_CLUSTER_ID,
        probe.owner,
        verdicts.OPERATION,
        ",".join(verdicts.NODE_IDS),
        str(verdicts.SEED_LEASE_SECONDS),
    )
    state["seed"] = dispatch
    seeded.write_json(case_dir / "dispatch.json", dispatch)
    stages["dispatch"] = verdicts.dispatch_errors(dispatch)
    first_id = str((dispatch.get("first") or {}).get("remote_command_id") or "")
    if stages["dispatch"]:
        raise seeded.SeededCommandError(
            "dispatch contract: " + "; ".join(stages["dispatch"])
        )

    ready = seeded.create_probe_pod(probe, case_dir)
    completed = seeded.wait_command(
        first_id,
        lambda item: item.get("status") in {"SUCCEEDED", "FAILED"},
        verdicts.COMPLETION_TIMEOUT_SECONDS,
    )
    seeded.write_json(case_dir / "completed-command.json", completed)
    # The workflow record as the control plane left it, before the purge: when
    # A does not end SUCCEEDED this is the evidence of who moved the workflow.
    seeded.write_json(
        case_dir / "workflow-final.json",
        seeded.cpu_python(_WORKFLOW_FINAL, str(dispatch["workflow_id"])),
    )
    executor_state = seeded.read_state(probe, "/state/executor-state.json")
    ledger = dict(executor_state.get("ledger") or {})
    seeded.write_json(case_dir / "executor-state.json", executor_state)
    stages["executed_once"] = verdicts.executed_once_errors(
        ledger, completed, first_id=first_id
    )
    # Stop the only claimant before minting anything else: B must never be
    # claimed, its purpose is to show the hold lifted.
    seeded.dataplane("delete", "pod", probe.pod, "--ignore-not-found", check=False)
    release = seeded.cpu_python(
        _RELEASE_AND_CANCEL,
        str(dispatch["workflow_id"]),
        probe.owner,
        f"CMD-018 acceptance cleanup {run_id}",
    )
    seeded.write_json(case_dir / "release.json", release)
    stages["release"] = verdicts.release_errors(release, first_id=first_id)
    state["command_ids"] = sorted(
        set(map(str, dispatch.get("all_command_ids") or []))
        | set(map(str, release.get("all_command_ids") or []))
    )
    stages["metric_family"] = verdicts.metric_family_errors(_control_plane_metrics())
    return {
        "case_id": CASE_ID,
        "attempt": attempt,
        "verdict": verdicts.case_verdict(stages),
        "errors": [
            f"{stage}: {error}" for stage, items in stages.items() for error in items
        ],
        "stages": stages,
        "dispatch": dispatch,
        "probe_ready": ready,
        "completed_command": completed,
        "ledger": ledger,
        "release": release,
    }


def _control_plane_metrics() -> list[str]:
    """The ``/metrics`` text of one Pod per control-plane role.

    ``open_sibling_holds_total`` is the dispatcher's counter, and since the role
    split the dispatcher runs in ``gpu-fault-control-worker`` on :8081; the
    ingress on :8080 does not carry it (attempt 1, 2026-09-09, read the ingress
    only and reported the family absent).
    """

    texts: list[str] = []
    for app, port in (("gpu-fault-control-worker", 8081), ("gpu-fault-api-ha", 8080)):
        script = (
            "import json,urllib.request\n"
            "print(json.dumps({'metrics': urllib.request.urlopen("
            f"'http://127.0.0.1:{port}/metrics', timeout=15).read().decode()}}))"
        )
        try:
            pod = seeded.control(
                "get",
                "pod",
                "-l",
                f"app={app}",
                "--field-selector=status.phase=Running",
                "-o",
                "jsonpath={.items[0].metadata.name}",
            ).strip()
            output = seeded.control(
                "exec",
                "-i",
                pod,
                "--",
                "python3",
                "-",
                stdin=script.encode(),
                timeout=120,
            )
            texts.append(str(json.loads(output.splitlines()[-1])["metrics"]))
        except Exception:  # noqa: BLE001 - the family check reports the absence
            continue
    return texts


def _purge(state: dict[str, Any], result: dict[str, Any], case_dir: Path) -> None:
    seed = state.get("seed") or {}
    if not seed:
        return
    try:
        purged = seeded.cpu_python(
            _PURGE,
            str(seed["workflow_id"]),
            str(seed["incident_id"]),
            str(seed["event_id"]),
            *state.get("command_ids", []),
        )
        seeded.write_json(case_dir / "seed-cleanup.json", purged)
        result["seed_cleanup"] = purged
        if purged.get("remaining_commands"):
            raise seeded.SeededCommandError(f"commands remain after purge: {purged}")
    except Exception as exc:  # noqa: BLE001 - recorded, verdict downgraded
        result["cleanup_error"] = f"seed cleanup: {type(exc).__name__}: {exc}"
        result["verdict"] = "FAIL"


def run_case(run_dir: Path, attempt: int, maintenance_window_end: datetime) -> int:
    case_dir = run_dir / "cases" / CASE_ID
    case_dir.mkdir(parents=True, exist_ok=True)
    run_id = seeded.run_identity(run_dir, attempt, "cmd018")
    probe = probe_definition()
    result: dict[str, Any] = {"case_id": CASE_ID, "verdict": "FAIL"}
    state: dict[str, Any] = {"seed": {}, "command_ids": []}
    try:
        result = _run_case(
            probe, case_dir, run_id, attempt, maintenance_window_end, state
        )
    except Exception as exc:  # noqa: BLE001 - recorded as the case error
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        _purge(state, result, case_dir)
        # The shared cleanup purges nothing of ours (seed={}) but tears down the
        # Pod, ConfigMap and registry entry and runs the three postflights.
        seeded.cleanup(probe, case_dir, run_id, result, {})
    seeded.write_json(case_dir / f"{CASE_ID}.json", result)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["verdict"] == "PASS" else 1


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        description=(
            "Run the guarded CMD-018 acceptance: a rewritten step is held while its "
            "sibling command is open, and the node acts exactly once."
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

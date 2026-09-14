#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.e2e.regional.acceptance_runner_common import (  # noqa: E402
    processor_queue_backlog,
    write_json_atomic,
)
from scripts.e2e.regional.host_probe_fixture import (  # noqa: E402
    HostProbeFixture,
    HostProbeSettings,
)
from scripts.e2e.regional.live_driver_guard import (  # noqa: E402
    CaseRunner,
    add_live_arguments,
    record_focused_tests,
    reusable_focused_tests,
    run_standard_case,
)
from scripts.e2e.regional.regional_live_fixture import (  # noqa: E402
    RegionalFixtureError,
    RegionalLiveFixture,
    RegionalLiveSettings,
    required,
    run_case_main,
    settings_from_arguments,
)

PROBE_SCRIPT = Path(__file__).with_name("probes") / "node_host_probe.py"
CASE_ID = "GF-REGIONAL-DESTR-010"
CONFIRMATION = "DESTR010_RESTART_FABRIC_MANAGER"
FORBIDDEN_OPERATIONS = {
    "MARK_UNSCHEDULABLE",
    "QUARANTINE",
    "QUIESCE_GPU_SERVICES",
    "RESTORE_GPU_SERVICES",
}
EXPECTED_STEPS = ["FREEZE_EVIDENCE", "RESTART_FABRIC_MANAGER"]
EXPECTED_OWNERS = ["gpu-fault-control-plane", "gpu-fault-node-agent"]
# The Node fields a Fabric Manager restart must leave alone. `ready` is judged
# on its own: a node that flickers NotReady and returns is a different finding
# from one whose cordon/taint/ownership state changed.
NODE_STATE_FIELDS = ("unschedulable", "taints", "ownership_annotations")


@dataclass(frozen=True)
class Settings:
    regional: RegionalLiveSettings
    node: str
    host_probe_image: str

    def environment(self) -> dict[str, str]:
        return {
            **self.regional.environment(),
            "GPU_FAULT_TARGET_NODE": self.node,
            "GPU_FAULT_HOST_PROBE_IMAGE": self.host_probe_image,
        }


def configure(arguments: argparse.Namespace) -> Settings:
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
    )


# What the shared STORE_PROBE does not carry and this case needs: the policy's
# companion window, the node's XID events inside it, its active workflow
# incidents, and -- at the terminal read only -- the raw NVIDIA kernel evidence
# that names the marker. The evidence scan pages through up to 1000 records,
# which is why it is not part of the per-5s wait loop.
FABRIC_PROBE = r"""
import json
import sys
from datetime import datetime, timedelta, timezone

from gpu_fault.app import ApplicationContext
from gpu_fault.policy import load_xid_policy
from gpu_fault.telemetry import EvidenceKind

cluster_id, node_id, marker, include_evidence = sys.argv[1:]
store = ApplicationContext.from_environment().store
now = datetime.now(timezone.utc)
window = int(load_xid_policy().companion_window_seconds)
recent = store.list_xid_events(
    cluster_id,
    node_id,
    observed_after=now - timedelta(seconds=window),
)
active = store.list_active_workflow_incidents(
    cluster_id,
    node_ids={node_id},
)
evidence = []
if include_evidence and marker:
    evidence = [
        {
            "record_id": item.record_id,
            "observed_at": item.observed_at.isoformat(),
            "payload": item.payload,
        }
        for item in store.list_raw_evidence(
            cluster_id,
            node_id=node_id,
            kind=EvidenceKind.NVIDIA_KERNEL,
            limit=1000,
        )
        if marker in json.dumps(item.payload, sort_keys=True, default=str)
    ]
print(json.dumps({
    "companion_window_seconds": window,
    "recent_xid_events": [
        {
            "event_id": item.event_id,
            "xid": item.xid,
            "observed_at": item.observed_at.isoformat(),
        }
        for item in recent
    ],
    "active_workflow_incidents": [
        {
            "incident_id": incident_item.incident_id,
            "workflow_request_id": workflow_item.request_id,
            "state": incident_item.state.value,
            "workflow_status": workflow_item.status.value,
        }
        for incident_item, workflow_item in active
    ],
    "evidence": evidence,
}, sort_keys=True, default=str))
"""


def fabric_probe(
    regional: RegionalLiveFixture,
    node: str,
    *,
    marker: str = "",
    include_evidence: bool = False,
) -> dict[str, Any]:
    return regional.cpu_python(
        FABRIC_PROBE,
        regional.settings.cluster_id,
        node,
        marker,
        "1" if include_evidence else "",
    )


def capability(profile: dict[str, Any] | None, name: str) -> dict[str, Any] | None:
    for item in (profile or {}).get("capabilities", []):
        if isinstance(item, dict) and item.get("capability") == name:
            return cast(dict[str, Any], item)
    return None


def normalize_notifications(items: list[Any]) -> list[dict[str, Any]]:
    """One notification shape for the verdict, whichever probe produced it.

    The shared STORE_PROBE returns ``{"notification": {...}, "result": {...}}``
    pairs; the flat ``{"notification_id", "category", "subject", "status"}``
    shape is what this case recorded before it moved onto the shared fixture.
    """

    result = []
    for item in items:
        if not isinstance(item, dict):
            continue
        if "notification" in item:
            notification = item.get("notification") or {}
            delivery = item.get("result") or {}
            result.append(
                {
                    "notification_id": notification.get("notification_id"),
                    "category": notification.get("category"),
                    "subject": notification.get("subject"),
                    "status": delivery.get("status") if delivery else None,
                    "provider_message_id_present": bool(
                        delivery and delivery.get("provider_message_id")
                    ),
                }
            )
        else:
            result.append(dict(item))
    return result


def focused_tests(case_dir: Path, *, reuse: bool = False) -> dict[str, Any]:
    """Run the focused pytest, or reuse the plan's result in ``--execute``.

    ``reuse`` consults ``reusable_focused_tests`` on the plan this case wrote:
    a passing result recorded against the same source digest is not re-run.
    """

    if reuse:
        recorded = reusable_focused_tests(case_dir / "plan.json")
        if recorded is not None:
            return {**recorded, "focused_tests_reused": True}
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "tests/node_agent/test_remediation.py::"
        "test_fabric_manager_restart_is_fenced_and_verified",
        "tests/node_agent/test_remediation.py::"
        "test_fabric_manager_restart_refuses_active_compute_client",
        "tests/node_agent/test_remediation.py::"
        "test_fabric_manager_restart_requires_explicit_enable",
        "tests/policy/_policy_cases_2.py::"
        "test_xid_45_and_48_workflow_branches_are_exact",
    ]
    completed = RegionalLiveFixture.run(
        command,
        cwd=ROOT,
        check=False,
        timeout=300,
    )
    path = case_dir / "focused-tests.log"
    path.write_text(completed.stdout + completed.stderr, encoding="utf-8")
    path.chmod(0o600)
    return {
        "passed": completed.returncode == 0,
        "returncode": completed.returncode,
        "command": command,
    }


def validate_preflight(
    state: dict[str, Any],
    fabric: dict[str, Any],
    node: dict[str, Any],
    workloads: list[dict[str, str]],
) -> list[str]:
    errors = []
    if node["ready"] != "True":
        errors.append("target node is not Ready")
    if node["unschedulable"]:
        errors.append("target node is unschedulable")
    if node["taints"]:
        errors.append("target node has taints")
    if workloads:
        errors.append("target node has non-system running Pods")
    agent = state.get("agent") or {}
    if agent.get("lifecycle_state") != "ACTIVE":
        errors.append("target Node Agent is not ACTIVE")
    if "RESTART_FABRIC_MANAGER" not in (agent.get("allowed_operations") or []):
        errors.append("target Node Agent does not allow RESTART_FABRIC_MANAGER")
    profile = state.get("profile") or {}
    restart = capability(profile, "fabricManagerRestart")
    if restart is None:
        errors.append("runtime profile has no fabricManagerRestart capability")
    elif (
        restart.get("mode") != "OWN"
        or restart.get("owner") != "gpu-fault-node-agent"
        or restart.get("adapter") != "node-action"
    ):
        errors.append("fabricManagerRestart is not OWN by the Node Agent")
    if profile.get("warnings"):
        errors.append("runtime profile has warnings")
    if fabric.get("active_workflow_incidents"):
        errors.append("target node already has an active workflow")
    if fabric.get("recent_xid_events"):
        errors.append("target node has XID events inside the companion window")
    # Routine telemetry above the fault tier must not refuse the case; only a
    # fault-tier backlog does (live 2026-09-07).
    if processor_queue_backlog(state.get("queue") or {}):
        errors.append("processor queue is not empty")
    if (state.get("remote_commands") or {}).get("open_by_cluster"):
        errors.append("remote command queue is not empty")
    return errors


def read_only_preflight(
    settings: Settings,
    case_dir: Path,
    *,
    reuse_focused_tests: bool = False,
) -> dict[str, Any]:
    regional = RegionalLiveFixture(settings.regional)
    identity = regional.evidence_identity()
    # The default drained queue read is the gate a preflight wants; the wait
    # loops below take the single sample.
    state = regional.store_snapshot(node=settings.node)
    fabric = fabric_probe(regional, settings.node)
    node = regional.node_snapshot(settings.node)
    workloads = regional.business_workloads(settings.node)
    tests = focused_tests(case_dir, reuse=reuse_focused_tests)
    errors = validate_preflight(state, fabric, node, workloads)
    if not tests["passed"]:
        errors.append("focused regression tests failed")
    result = {
        "release_id": state.get("release_id"),
        "evidence_identity": identity,
        "node": node,
        "business_workloads": workloads,
        "store": state,
        "fabric": fabric,
        "focused_tests": tests,
        "errors": errors,
    }
    write_json_atomic(case_dir / "preflight.json", result)
    return result


def workflow_errors(state: dict[str, Any]) -> list[str]:
    errors = []
    event = state.get("event") or {}
    decision = state.get("decision") or {}
    workflow = state.get("workflow") or {}
    commands = state.get("commands") or []
    evidence = state.get("evidence") or []
    notifications = [
        item
        for item in normalize_notifications(state.get("notifications") or [])
        if item.get("category") == "ACTION_COMPLETED"
        and "Fabric Manager" in str(item.get("subject") or "")
    ]
    if event.get("xid") != 45:
        errors.append("matched event is not XID 45")
    if not str(event.get("evidence_ref") or "").startswith("kmsg://"):
        errors.append("XID evidence is not backed by kmsg://")
    if decision.get("official_action") != "RESTART_FM":
        errors.append("policy did not finalize XID 45 as RESTART_FM")
    if decision.get("disposition") != "EXECUTABLE":
        errors.append("policy decision is not EXECUTABLE")
    official_steps = [
        item.get("operation") for item in workflow.get("official_steps", [])
    ]
    if official_steps != EXPECTED_STEPS:
        errors.append("workflow official steps differ from the two-step contract")
    owners = [
        item.get("execution_owner") for item in workflow.get("official_steps", [])
    ]
    if owners != EXPECTED_OWNERS:
        errors.append("workflow execution owners differ from the contract")
    completed = set(workflow.get("completed_operations") or [])
    if completed.intersection(FORBIDDEN_OPERATIONS):
        errors.append("workflow completed a forbidden isolation/quiesce operation")
    if workflow.get("status") != "SUCCEEDED":
        errors.append("workflow is not SUCCEEDED")
    if len(commands) != 1 or commands[0].get("status") != "SUCCEEDED":
        errors.append("remote Fabric Manager command is not uniquely SUCCEEDED")
    if not evidence:
        errors.append("raw NVIDIA kernel evidence was not retained")
    if len(notifications) != 1:
        errors.append("Fabric Manager action notification is not unique")
    elif notifications[0].get("status") not in {"SENT", "SKIPPED"}:
        errors.append("Fabric Manager action notification is not terminal")
    return errors


def node_state_errors(
    baseline: dict[str, Any],
    current: dict[str, Any],
    *,
    stage: str,
) -> list[str]:
    """Scheduling state must equal the baseline; readiness is its own finding.

    Comparing whole snapshots also compared ``ready``, ``boot_id`` and the
    installer annotations, so a Ready condition heartbeat or an unrelated
    annotation rewrite reported as "node state drifted" with no way to tell
    it from a cordon the case must never apply.
    """

    errors = []
    for field in NODE_STATE_FIELDS:
        if baseline.get(field) != current.get(field):
            errors.append(f"{stage}: target node {field} differs from baseline")
    if current.get("uid") != baseline.get("uid"):
        errors.append(f"{stage}: target node UID changed")
    if current.get("ready") != "True":
        errors.append(f"{stage}: target node is not Ready")
    return errors


REPLAY_SCRIPT = r"""
import json
import sys

from gpu_fault.cluster_executor import executor_from_environment
from gpu_fault.execution import WorkflowStepContext
from gpu_fault.regional import RemoteActionCommand

command = RemoteActionCommand.model_validate(json.loads(sys.argv[1]))
executor = executor_from_environment()
adapter = next(
    item for item in executor.adapters
    if getattr(item, "owner", "") == "gpu-fault-node-agent"
)
outcome = adapter.execute(
    WorkflowStepContext(
        workflow=command.workflow,
        incident=command.incident,
        step=command.step,
        step_index=command.step_index,
        # The request builder lives on the executor's dispatch layer since the
        # 2026-09-09 layer split (130be28); the private executor method it
        # replaced is gone from the deployed image (DESTR-010, 2026-09-14).
        request=executor.dispatch.execution_request(command),
        idempotency_key=command.idempotency_key,
    )
)
print(json.dumps({
    "status": outcome.status.value,
    "adapter_operation_id": outcome.adapter_operation_id,
    "details": outcome.details,
    "error": outcome.error,
}, sort_keys=True))
"""


def replayable_command(command: dict[str, Any]) -> dict[str, Any]:
    """The remote command as ``RemoteActionCommand`` will accept it.

    The store probe redacts the lease token into a digest and a length; the
    model forbids unknown fields, and the replay does not present a lease --
    it runs the adapter directly against the Node Agent, whose ledger is the
    idempotency proof -- so the token is simply absent.
    """

    value = {
        key: item
        for key, item in command.items()
        if key not in {"lease_token_sha256", "lease_token_length"}
    }
    value["lease_token"] = None
    return value


def replay_command(
    regional: RegionalLiveFixture,
    command: dict[str, Any],
) -> dict[str, Any]:
    # One attempt: the replay is expected to be a no-op on the Node Agent's
    # ledger, but that is the fact under test, not a premise to retry on.
    return regional.executor_python(
        REPLAY_SCRIPT,
        json.dumps(replayable_command(command), sort_keys=True),
        timeout=180,
        attempts=1,
    )


def host_errors(
    baseline: dict[str, Any],
    after_first: dict[str, Any],
    after_replay: dict[str, Any],
    expected_command_id: str,
) -> list[str]:
    errors = []
    before_service = baseline["fabric_manager"]
    first_service = after_first["fabric_manager"]
    replay_service = after_replay["fabric_manager"]
    if first_service.get("ActiveState") != "active":
        errors.append("Fabric Manager is not active after the first action")
    if before_service.get("MainPID") == first_service.get("MainPID"):
        errors.append("Fabric Manager MainPID did not change")
    if before_service.get("InvocationID") == first_service.get("InvocationID"):
        errors.append("Fabric Manager InvocationID did not change")
    if after_first["journal"].get("started_count") != 1:
        errors.append("journal does not show exactly one Fabric Manager start")
    baseline_ids = {item["command_id"] for item in baseline["ledger"]}
    first_ids = {item["command_id"] for item in after_first["ledger"]}
    if first_ids - baseline_ids != {expected_command_id}:
        errors.append("Node Agent ledger did not add exactly the expected command")
    if replay_service.get("MainPID") != first_service.get("MainPID"):
        errors.append("Fabric Manager MainPID changed during replay")
    if {item["command_id"] for item in after_replay["ledger"]} != first_ids:
        errors.append("Node Agent ledger grew during replay")
    if baseline["gpu_fault_timers"] != after_replay["gpu_fault_timers"]:
        errors.append("gpu-fault timer inventory changed")
    return errors


def preflight_identity(preflight: dict[str, Any]) -> dict[str, Any]:
    store = preflight["store"]
    return {
        "release_id": preflight["release_id"],
        "node_uid": preflight["node"]["uid"],
        "agent_generation": (store.get("agent") or {}).get("generation"),
        "runtime_profile_version": (store.get("profile") or {}).get("profile_version"),
    }


def execute_case(
    settings: Settings,
    run_dir: Path,
    attempt: int,
    maintenance_window_end: datetime,
) -> int:
    case_dir = run_dir / "cases" / CASE_ID
    case_dir.mkdir(parents=True, exist_ok=True)
    preflight = read_only_preflight(settings, case_dir, reuse_focused_tests=True)
    if preflight["errors"]:
        raise RegionalFixtureError(
            "preflight failed: " + "; ".join(preflight["errors"])
        )
    plan = json.loads((case_dir / "plan.json").read_text(encoding="utf-8"))
    planned = plan["details"]["preflight_identity"]
    current = preflight_identity(preflight)
    if current != planned:
        raise RegionalFixtureError(f"DESTR-010 plan drifted: {planned} != {current}")

    regional = RegionalLiveFixture(settings.regional)
    run_id = f"destr010-{run_dir.name.rsplit('-', 1)[-1].lower()}-a{attempt}"
    marker = f"destr010-{int(time.time())}-a{attempt}"
    fixture = HostProbeFixture(
        HostProbeSettings(
            kubeconfig=settings.regional.gpu_kubeconfig,
            context=settings.regional.gpu_context,
            namespace=settings.regional.namespace,
            node=settings.node,
            image=settings.host_probe_image,
            case_id=CASE_ID,
            run_id=run_id,
            probe_script=PROBE_SCRIPT,
        )
    )
    result: dict[str, Any] = {
        "case_id": CASE_ID,
        "attempt": attempt,
        "verdict": "FAIL",
        **preflight["evidence_identity"],
        "node": settings.node,
        "maintenance_window_end": maintenance_window_end.isoformat(),
        "focused_tests_reused": bool(
            preflight["focused_tests"].get("focused_tests_reused")
        ),
    }
    baseline_node = preflight["node"]
    baseline_host: dict[str, Any] | None = None
    injected_at: datetime | None = None
    try:
        fixture.create()
        baseline_host = fixture.execute("snapshot")
        write_json_atomic(case_dir / "host-baseline.json", baseline_host)
        if not baseline_host["kmsg_writable"]:
            raise RegionalFixtureError("/dev/kmsg is not writable from the host probe")
        if baseline_host["compute_clients"]:
            raise RegionalFixtureError("target node has active NVIDIA compute clients")
        if baseline_host["fabric_manager"].get("ActiveState") != "active":
            raise RegionalFixtureError("Fabric Manager is not active at baseline")
        if datetime.now(timezone.utc) >= maintenance_window_end:
            raise RegionalFixtureError(
                "approved maintenance window ended before injection"
            )

        injected_at = datetime.now(timezone.utc)
        injection = fixture.execute(
            "write-xid45",
            "--marker",
            marker,
            "--drill-id",
            run_id,
            "--pci-bdf",
            baseline_host["gpu_pci_bdf"],
        )
        write_json_atomic(case_dir / "injection.json", injection)
        timeout = int(preflight["fabric"]["companion_window_seconds"]) + 300
        # The shared wait: one cheap store read per 5s, commands filtered to
        # the marker's workflow, notifications to its incident, no raw
        # evidence scan. The evidence scan happens once, at the terminal read.
        state = regional.wait_for_workflow(
            node=settings.node,
            marker=marker,
            observed_after=injected_at,
            case_dir=case_dir,
            timeout_seconds=timeout,
        )
        terminal_fabric = fabric_probe(
            regional,
            settings.node,
            marker=marker,
            include_evidence=True,
        )
        state = {
            **state,
            "evidence": terminal_fabric.get("evidence") or [],
            "notifications": normalize_notifications(state.get("notifications") or []),
        }
        write_json_atomic(case_dir / "workflow-state.json", state)
        errors = workflow_errors(state)
        workflow_request_id = str((state.get("workflow") or {}).get("request_id") or "")
        command = state["commands"][0] if state.get("commands") else {}
        generation = int((state.get("agent") or {}).get("generation") or 0)
        expected_command_id = (
            f"{command.get('idempotency_key')}/{settings.node}/agent-{generation}"
        )
        after_first = fixture.execute(
            "snapshot",
            "--since-epoch",
            str(injected_at.timestamp()),
        )
        write_json_atomic(case_dir / "host-after-first.json", after_first)
        ledger_ids = {item["command_id"] for item in after_first["ledger"]}
        if expected_command_id not in ledger_ids:
            errors.append("expected Node Agent command ID is absent from the ledger")

        replay: dict[str, Any] = {}
        if not errors:
            replay = replay_command(regional, command)
            write_json_atomic(case_dir / "replay.json", replay)
            if replay.get("status") != "SUCCEEDED":
                errors.append("isolated Node Agent replay did not return SUCCEEDED")
        after_replay_state = regional.store_snapshot(
            node=settings.node,
            marker=marker,
            observed_after=injected_at,
            queue_attempts=1,
            workflow_request_ids=[workflow_request_id] if workflow_request_id else None,
        )
        after_replay_state["notifications"] = normalize_notifications(
            after_replay_state.get("notifications") or []
        )
        write_json_atomic(
            case_dir / "store-after-replay.json",
            after_replay_state,
        )
        first_notification_ids = {
            item.get("notification_id") for item in state.get("notifications") or []
        }
        replay_notification_ids = {
            item.get("notification_id")
            for item in after_replay_state.get("notifications") or []
        }
        if replay_notification_ids != first_notification_ids:
            errors.append("notification set changed during Node Agent replay")
        after_replay = fixture.execute(
            "snapshot",
            "--since-epoch",
            str(injected_at.timestamp()),
        )
        write_json_atomic(case_dir / "host-after-replay.json", after_replay)
        errors.extend(
            host_errors(
                baseline_host,
                after_first,
                after_replay,
                expected_command_id,
            )
        )
        post_node = regional.node_snapshot(settings.node)
        write_json_atomic(case_dir / "node-postflight.json", post_node)
        errors.extend(node_state_errors(baseline_node, post_node, stage="postflight"))
        if regional.business_workloads(settings.node):
            errors.append("target node acquired a non-system workload")
        provider_window_end = datetime.now(timezone.utc)
        events = regional.provider_events(injected_at, provider_window_end)
        # "No provider mutation" is not provable inside CloudTrail's delivery
        # window; an empty read is recorded as provisional, not as proof.
        provider_provisional = not events and regional.provider_events_provisional(
            provider_window_end
        )
        write_json_atomic(
            case_dir / "provider-events.json",
            {"events": events, "provider_events_provisional": provider_provisional},
        )
        if events:
            errors.append("provider mutation appeared during DESTR-010")
        result.update(
            {
                "verdict": "PASS" if not errors else "FAIL",
                "errors": errors,
                "marker": marker,
                "workflow": {
                    "request_id": workflow_request_id or None,
                    "status": (state.get("workflow") or {}).get("status"),
                    "official_action": (state.get("workflow") or {}).get(
                        "official_action"
                    ),
                },
                "remote_command_id": command.get("command_id"),
                "node_action_command_id": expected_command_id,
                "replay": replay,
                "notifications": after_replay_state.get("notifications") or [],
                "provider_events": events,
                "provider_events_provisional": provider_provisional,
            }
        )
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        recovery: dict[str, Any] = {}
        residuals: dict[str, bool] = {}
        try:
            recovery = fixture.execute(
                "ensure-fabric-manager-active",
                timeout=120,
            )
        except Exception as exc:
            recovery = {"error": f"{type(exc).__name__}: {exc}"}
            result["verdict"] = "FAIL"
        try:
            residuals = fixture.cleanup()
        except Exception as exc:
            residuals = {"cleanup_error": True}
            result["cleanup_error"] = f"{type(exc).__name__}: {exc}"
            result["verdict"] = "FAIL"
        result["recovery"] = recovery
        result["probe_residuals"] = residuals
        if any(residuals.values()):
            result["verdict"] = "FAIL"
        try:
            final_node = regional.node_snapshot(settings.node)
            result["final_node"] = final_node
            final_errors = node_state_errors(baseline_node, final_node, stage="final")
            if final_errors:
                result["verdict"] = "FAIL"
                errors_recorded = result.setdefault("errors", [])
                if isinstance(errors_recorded, list):
                    errors_recorded.extend(final_errors)
        except Exception as exc:
            result["postflight_error"] = f"{type(exc).__name__}: {exc}"
            result["verdict"] = "FAIL"
    write_json_atomic(case_dir / f"{CASE_ID}.json", result)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["verdict"] == "PASS" else 1


def plan_details(settings: Settings, preflight: dict[str, Any]) -> dict[str, Any]:
    details = {
        "risk": "live-service-action",
        "release_id": preflight["release_id"],
        "target_node": settings.node,
        "mutation": (
            "write one synthetic XID 45 to the real host /dev/kmsg and allow "
            "the Node Agent to restart nvidia-fabricmanager exactly once"
        ),
        "preflight_identity": preflight_identity(preflight),
        "stop_conditions": [
            "preflight or focused regression failure",
            "target node is not Ready/schedulable/idle",
            "recent XID is inside the companion window",
            "Node Agent/Profile/pin drift",
            "Fabric Manager or /dev/kmsg baseline is unhealthy",
            "workflow differs from FREEZE_EVIDENCE -> RESTART_FABRIC_MANAGER",
            "node scheduling state changes",
            "provider mutation appears",
            "probe cleanup leaves a residual",
        ],
        "rollback": {
            "probe_active_deadline_seconds": 1800,
            "runner_finally_ensures_fabric_manager_active": True,
            "runner_finally_deletes_probe_resources": True,
            "node_scheduling_state_must_equal_baseline": True,
            "node_readiness_is_judged_separately": True,
        },
        "preflight": preflight,
    }
    record_focused_tests(details, preflight["focused_tests"])
    return details


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        description="Run the guarded DESTR-010 Fabric Manager restart acceptance."
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

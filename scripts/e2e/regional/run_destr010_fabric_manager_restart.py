#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scripts.e2e.regional.acceptance_runner_common import processor_queue_backlog

if __package__:
    from .acceptance_runner_common import write_json_atomic
    from .host_probe_fixture import (
        HostProbeFixture,
        HostProbeSettings,
    )
    from .live_driver_guard import (
        CaseRunner,
        add_live_arguments,
        run_standard_case,
    )
    from .regional_live_fixture import run_case_main
else:
    from acceptance_runner_common import write_json_atomic
    from host_probe_fixture import HostProbeFixture, HostProbeSettings
    from live_driver_guard import (
        CaseRunner,
        add_live_arguments,
        run_standard_case,
    )
    from regional_live_fixture import run_case_main

ROOT = Path(__file__).resolve().parents[3]
PROBE_SCRIPT = Path(__file__).with_name("probes") / "node_host_probe.py"
CASE_ID = "GF-REGIONAL-DESTR-010"
CONFIRMATION = "DESTR010_RESTART_FABRIC_MANAGER"
SYSTEM_NAMESPACES = {
    "aws-hyperpod",
    "cert-manager",
    "gpu-fault-system",
    "hyperpod-inference-system",
    "kube-system",
    "kubeflow",
}
FORBIDDEN_OPERATIONS = {
    "MARK_UNSCHEDULABLE",
    "QUARANTINE",
    "QUIESCE_GPU_SERVICES",
    "RESTORE_GPU_SERVICES",
}
PROVIDER_MUTATIONS = {
    "BatchReplaceClusterNodes",
    "BatchRebootClusterNodes",
    "ReplaceClusterNodes",
    "RebootClusterNodes",
}


class CaseError(RuntimeError):
    pass


@dataclass(frozen=True)
class Settings:
    cpu_kubeconfig: Path
    gpu_kubeconfig: Path
    gpu_context: str
    namespace: str
    cluster_id: str
    node: str
    region: str
    host_probe_image: str

    def environment(self) -> dict[str, str]:
        return {
            "CPU_KUBECONFIG": str(self.cpu_kubeconfig),
            "GPU_KUBECONFIG": str(self.gpu_kubeconfig),
            "GPU_EKS_CONTEXT": self.gpu_context,
            "GPU_FAULT_NAMESPACE": self.namespace,
            "GPU_FAULT_CLUSTER_ID": self.cluster_id,
            "GPU_FAULT_TARGET_NODE": self.node,
            "AWS_REGION": self.region,
            "GPU_FAULT_HOST_PROBE_IMAGE": self.host_probe_image,
        }


def required(value: str, label: str) -> str:
    result = value.strip()
    if not result:
        raise CaseError(f"{label} is required")
    return result


def configure(arguments: argparse.Namespace) -> Settings:
    cpu = (
        Path(
            required(
                arguments.cpu_kubeconfig
                or os.getenv("GPU_FAULT_CONTROL_KUBECONFIG", "")
                or os.getenv("CPU_KUBECONFIG", ""),
                "CPU kubeconfig",
            )
        )
        .expanduser()
        .resolve()
    )
    gpu = (
        Path(
            required(
                arguments.gpu_kubeconfig
                or os.getenv("GPU_KUBECONFIG", "")
                or os.getenv("KUBECONFIG", ""),
                "GPU kubeconfig",
            )
        )
        .expanduser()
        .resolve()
    )
    if not cpu.is_file() or not gpu.is_file():
        raise CaseError("configured CPU/GPU kubeconfig does not exist")
    return Settings(
        cpu_kubeconfig=cpu,
        gpu_kubeconfig=gpu,
        gpu_context=required(
            arguments.gpu_context
            or os.getenv("GPU_EKS_CONTEXT", "")
            or os.getenv("GPU_FAULT_DATAPLANE_CONTEXT", ""),
            "GPU context",
        ),
        namespace=required(arguments.namespace, "namespace"),
        cluster_id=required(
            arguments.cluster_id or os.getenv("GPU_FAULT_CLUSTER_ID", ""),
            "cluster ID",
        ),
        node=required(
            arguments.node or os.getenv("GPU_FAULT_TARGET_NODE", ""),
            "target node",
        ),
        region=required(
            arguments.region
            or os.getenv("AWS_REGION", "")
            or os.getenv("AWS_DEFAULT_REGION", ""),
            "AWS Region",
        ),
        host_probe_image=required(
            arguments.host_probe_image or os.getenv("GPU_FAULT_HOST_PROBE_IMAGE", ""),
            "host probe image",
        ),
    )


def run(
    command: list[str],
    *,
    input_text: str | None = None,
    check: bool = True,
    timeout: int = 300,
) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        command,
        input=input_text,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=False,
    )
    if check and completed.returncode:
        raise CaseError(
            f"command failed ({completed.returncode}): {' '.join(command)}; "
            f"stderr={completed.stderr.strip()}"
        )
    return completed


def kubectl(
    settings: Settings,
    plane: str,
    *arguments: str,
    input_text: str | None = None,
    check: bool = True,
    timeout: int = 300,
) -> str:
    command = ["kubectl", "--kubeconfig"]
    if plane == "cpu":
        command.extend([str(settings.cpu_kubeconfig), "-n", settings.namespace])
    else:
        command.extend(
            [
                str(settings.gpu_kubeconfig),
                "--context",
                settings.gpu_context,
                "-n",
                settings.namespace,
            ]
        )
    command.extend(arguments)
    return run(
        command,
        input_text=input_text,
        check=check,
        timeout=timeout,
    ).stdout


def ready_pod(settings: Settings, plane: str, app: str) -> str:
    payload = json.loads(
        kubectl(
            settings,
            plane,
            "get",
            "pod",
            "-l",
            f"app={app}",
            "-o",
            "json",
        )
    )
    names = sorted(
        str(item["metadata"]["name"])
        for item in payload.get("items", [])
        if item.get("status", {}).get("phase") == "Running"
        and item.get("status", {}).get("containerStatuses")
        and all(
            bool(status.get("ready"))
            for status in item.get("status", {}).get("containerStatuses", [])
        )
    )
    if not names:
        raise CaseError(f"no Ready {plane} Pod for app={app}")
    return names[0]


def cpu_python(settings: Settings, script: str, *arguments: str) -> dict[str, Any]:
    last_error: Exception | None = None
    for _attempt in range(3):
        try:
            output = kubectl(
                settings,
                "cpu",
                "exec",
                "-i",
                ready_pod(settings, "cpu", "gpu-fault-api-ha"),
                "--",
                "python3",
                "-",
                *arguments,
                input_text=script,
                timeout=120,
            )
            return json.loads(output.splitlines()[-1])
        except Exception as exc:
            last_error = exc
            time.sleep(1)
    raise CaseError(f"CPU store probe failed: {last_error}")


STORE_PROBE = r"""
import json
import os
import sys
from datetime import datetime, timedelta, timezone

from gpu_fault.app import ApplicationContext
from gpu_fault.models import WorkflowStatus
from gpu_fault.policy import load_xid_policy
from gpu_fault.telemetry import EvidenceKind

cluster_id, node_id, marker, observed_after_text = sys.argv[1:]
store = ApplicationContext.from_environment().store
now = datetime.now(timezone.utc)
observed_after = (
    datetime.fromisoformat(observed_after_text.replace("Z", "+00:00"))
    if observed_after_text
    else None
)
agent = store.get_agent(cluster_id, node_id)
profile = store.get_profile(agent.runtime_profile_version)
capability = next(
    (
        item.model_dump(mode="json")
        for item in profile.capabilities
        if item.capability.value == "fabricManagerRestart"
    ),
    None,
)
window = int(load_xid_policy().companion_window_seconds)
recent = store.list_xid_events(
    cluster_id,
    node_id,
    observed_after=now - timedelta(seconds=window),
)
events = store.list_xid_events(
    cluster_id,
    node_id,
    observed_after=observed_after,
) if observed_after is not None else []
matching_events = [
    item for item in events
    if marker and marker in str(item.raw_message or "")
]
evidence = store.list_raw_evidence(
    cluster_id,
    node_id=node_id,
    kind=EvidenceKind.NVIDIA_KERNEL,
    limit=1000,
)
matching_evidence = [
    item for item in evidence
    if marker and marker in json.dumps(item.payload, sort_keys=True, default=str)
]
event = matching_events[-1] if matching_events else None
decision = store.get_xid_policy_decision(event.event_id) if event is not None else None
incident = store.get_incident_by_event(event.event_id) if event is not None else None
workflow = (
    store.get_workflow(incident.workflow_request_id)
    if incident is not None and incident.workflow_request_id
    else None
)
commands = [
    item for item in store.list_remote_commands()
    if workflow is not None and item.workflow_request_id == workflow.request_id
]
notifications = []
if incident is not None:
    for item in store.list_notifications():
        if item.incident_id != incident.incident_id:
            continue
        result = store.get_notification_result(item.notification_id)
        notifications.append({
            "notification_id": item.notification_id,
            "category": item.category,
            "subject": item.subject,
            "status": result.status.value if result is not None else None,
            "provider_message_id_present": bool(
                result is not None and result.provider_message_id
            ),
        })
active = store.list_active_workflow_incidents(
    cluster_id,
    node_ids={node_id},
)
print(json.dumps({
    "release_id": os.getenv("GPU_FAULT_RELEASE_ID"),
    "companion_window_seconds": window,
    "agent": {
        "lifecycle_state": agent.lifecycle_state.value,
        "lease_expires_at": (
            agent.lease_expires_at.isoformat()
            if agent.lease_expires_at is not None else None
        ),
        "generation": agent.generation,
        "agent_version": agent.agent_version,
        "artifact_sha256": agent.artifact_sha256,
        "compatibility_digest": (
            agent.compatibility_digest or agent.artifact_sha256
        ),
        "runtime_profile_version": agent.runtime_profile_version,
        "config_digest": agent.config_digest,
        "node_action_key_version": agent.node_action_key_version,
        "allowed_operations": [
            item.value for item in agent.allowed_operations
        ],
    },
    "profile": {
        "version": profile.profile_version,
        "warnings": profile.warnings,
        "fabric_manager_restart": capability,
    },
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
    "queue": store.processor_queue_stats(),
    "remote_commands": store.remote_command_stats(),
    "event": event.model_dump(mode="json") if event is not None else None,
    "decision": (
        decision.model_dump(mode="json") if decision is not None else None
    ),
    "incident": (
        incident.model_dump(mode="json") if incident is not None else None
    ),
    "workflow": (
        workflow.model_dump(mode="json") if workflow is not None else None
    ),
    "commands": [item.model_dump(mode="json") for item in commands],
    "notifications": notifications,
    "evidence": [
        {
            "record_id": item.record_id,
            "observed_at": item.observed_at.isoformat(),
            "payload": item.payload,
        }
        for item in matching_evidence
    ],
}, sort_keys=True, default=str))
"""


def store_probe(
    settings: Settings,
    *,
    marker: str = "",
    observed_after: datetime | None = None,
) -> dict[str, Any]:
    return cpu_python(
        settings,
        STORE_PROBE,
        settings.cluster_id,
        settings.node,
        marker,
        observed_after.isoformat() if observed_after is not None else "",
    )


def node_snapshot(settings: Settings) -> dict[str, Any]:
    value = json.loads(
        kubectl(settings, "gpu", "get", "node", settings.node, "-o", "json")
    )
    annotations = value["metadata"].get("annotations", {})
    return {
        "name": value["metadata"]["name"],
        "uid": value["metadata"]["uid"],
        "ready": next(
            (
                condition["status"]
                for condition in value["status"].get("conditions", [])
                if condition["type"] == "Ready"
            ),
            None,
        ),
        "unschedulable": value["spec"].get("unschedulable", False),
        "taints": value["spec"].get("taints", []),
        "gpu_allocatable": value["status"].get("allocatable", {}).get("nvidia.com/gpu"),
        "ownership_annotations": {
            key: value
            for key, value in annotations.items()
            if key.startswith("gpu-fault.io/")
            and not key.startswith("gpu-fault.io/installer-")
        },
    }


def business_workloads(settings: Settings) -> list[dict[str, str]]:
    value = json.loads(
        run(
            [
                "kubectl",
                "--kubeconfig",
                str(settings.gpu_kubeconfig),
                "--context",
                settings.gpu_context,
                "get",
                "pod",
                "-A",
                "--field-selector",
                f"spec.nodeName={settings.node},status.phase=Running",
                "-o",
                "json",
            ]
        ).stdout
    )
    return [
        {
            "namespace": str(item["metadata"].get("namespace", "")),
            "name": str(item["metadata"].get("name", "")),
        }
        for item in value.get("items", [])
        if item["metadata"].get("namespace") not in SYSTEM_NAMESPACES
    ]


def release_id(settings: Settings) -> str:
    value = json.loads(
        kubectl(
            settings,
            "cpu",
            "get",
            "configmap",
            "gpu-fault-regional-release-state",
            "-o",
            "json",
        )
    )
    state = json.loads(value["data"]["state.json"])
    return str(state.get("release_id") or "")


def focused_tests(case_dir: Path) -> dict[str, Any]:
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
    completed = run(command, check=False, timeout=300)
    path = case_dir / "focused-tests.log"
    path.write_text(completed.stdout + completed.stderr, encoding="utf-8")
    path.chmod(0o600)
    return {
        "passed": completed.returncode == 0,
        "returncode": completed.returncode,
        "command": command,
    }


def validate_preflight(
    settings: Settings,
    state: dict[str, Any],
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
    agent = state["agent"]
    if agent["lifecycle_state"] != "ACTIVE":
        errors.append("target Node Agent is not ACTIVE")
    if "RESTART_FABRIC_MANAGER" not in agent["allowed_operations"]:
        errors.append("target Node Agent does not allow RESTART_FABRIC_MANAGER")
    capability = state["profile"]["fabric_manager_restart"]
    if capability is None:
        errors.append("runtime profile has no fabricManagerRestart capability")
    elif (
        capability.get("mode") != "OWN"
        or capability.get("owner") != "gpu-fault-node-agent"
        or capability.get("adapter") != "node-action"
    ):
        errors.append("fabricManagerRestart is not OWN by the Node Agent")
    if state["profile"]["warnings"]:
        errors.append("runtime profile has warnings")
    if state["active_workflow_incidents"]:
        errors.append("target node already has an active workflow")
    if state["recent_xid_events"]:
        errors.append("target node has XID events inside the companion window")
    if processor_queue_backlog(state["queue"]):
        errors.append("processor queue is not empty")
    open_commands = state["remote_commands"].get("open_by_cluster") or {}
    if open_commands:
        errors.append("remote command queue is not empty")
    return errors


def read_only_preflight(settings: Settings, case_dir: Path) -> dict[str, Any]:
    state = store_probe(settings)
    node = node_snapshot(settings)
    workloads = business_workloads(settings)
    tests = focused_tests(case_dir)
    errors = validate_preflight(settings, state, node, workloads)
    if not tests["passed"]:
        errors.append("focused regression tests failed")
    result = {
        "release_id": release_id(settings),
        "node": node,
        "business_workloads": workloads,
        "store": state,
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
        for item in state.get("notifications") or []
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
    if official_steps != ["FREEZE_EVIDENCE", "RESTART_FABRIC_MANAGER"]:
        errors.append("workflow official steps differ from the two-step contract")
    owners = [
        item.get("execution_owner") for item in workflow.get("official_steps", [])
    ]
    if owners != ["gpu-fault-control-plane", "gpu-fault-node-agent"]:
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


def wait_for_workflow(
    settings: Settings,
    marker: str,
    observed_after: datetime,
    timeout_seconds: int,
    case_dir: Path,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    last: dict[str, Any] = {}
    timeline = []
    while time.monotonic() < deadline:
        last = store_probe(
            settings,
            marker=marker,
            observed_after=observed_after,
        )
        workflow = last.get("workflow") or {}
        timeline.append(
            {
                "observed_at": datetime.now(timezone.utc).isoformat(),
                "event_id": (last.get("event") or {}).get("event_id"),
                "decision": (last.get("decision") or {}).get("disposition"),
                "workflow_status": workflow.get("status"),
                "command_statuses": [
                    item.get("status") for item in last.get("commands") or []
                ],
            }
        )
        write_json_atomic(case_dir / "timeline.json", {"entries": timeline})
        if workflow.get("status") in {"SUCCEEDED", "FAILED", "BLOCKED"}:
            return last
        time.sleep(5)
    raise CaseError(f"DESTR-010 workflow did not reach a terminal state: {last}")


REPLAY_SCRIPT = r"""
import json
import sys

from gpu_fault.cluster_executor import executor_from_environment
from gpu_fault.execution import WorkflowStepContext
from gpu_fault.regional import RemoteActionCommand

command = RemoteActionCommand.model_validate(json.load(sys.stdin))
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
        request=executor._execution_request(command),
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


def replay_command(settings: Settings, command: dict[str, Any]) -> dict[str, Any]:
    encoded = json.dumps(command, sort_keys=True)
    script = REPLAY_SCRIPT.replace(
        "command = RemoteActionCommand.model_validate(json.load(sys.stdin))",
        f"command = RemoteActionCommand.model_validate(json.loads({encoded!r}))",
    )
    output = kubectl(
        settings,
        "gpu",
        "exec",
        "-i",
        ready_pod(settings, "gpu", "gpu-fault-cluster-executor"),
        "--",
        "python3",
        "-",
        input_text=script,
        timeout=180,
    )
    return json.loads(output.splitlines()[-1])


def provider_events(
    settings: Settings,
    started_at: datetime,
    ended_at: datetime,
) -> list[dict[str, str]]:
    value = json.loads(
        run(
            [
                "aws",
                "cloudtrail",
                "lookup-events",
                "--region",
                settings.region,
                "--start-time",
                started_at.isoformat(),
                "--end-time",
                ended_at.isoformat(),
                "--lookup-attributes",
                "AttributeKey=EventSource,AttributeValue=sagemaker.amazonaws.com",
                "--output",
                "json",
            ]
        ).stdout
    )
    return [
        {
            "event_name": str(item.get("EventName")),
            "event_time": str(item.get("EventTime")),
        }
        for item in value.get("Events", [])
        if item.get("EventName") in PROVIDER_MUTATIONS
    ]


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


def execute_case(
    settings: Settings,
    run_dir: Path,
    attempt: int,
    maintenance_window_end: datetime,
) -> int:
    case_dir = run_dir / "cases" / CASE_ID
    case_dir.mkdir(parents=True, exist_ok=True)
    preflight = read_only_preflight(settings, case_dir)
    if preflight["errors"]:
        raise CaseError("preflight failed: " + "; ".join(preflight["errors"]))
    plan = json.loads((case_dir / "plan.json").read_text(encoding="utf-8"))
    planned = plan["details"]["preflight_identity"]
    current = {
        "release_id": preflight["release_id"],
        "node_uid": preflight["node"]["uid"],
        "agent_generation": preflight["store"]["agent"]["generation"],
        "runtime_profile_version": preflight["store"]["profile"]["version"],
    }
    if current != planned:
        raise CaseError(f"DESTR-010 plan drifted: {planned} != {current}")

    run_id = f"destr010-{run_dir.name.rsplit('-', 1)[-1].lower()}-a{attempt}"
    marker = f"destr010-{int(time.time())}-a{attempt}"
    fixture = HostProbeFixture(
        HostProbeSettings(
            kubeconfig=settings.gpu_kubeconfig,
            context=settings.gpu_context,
            namespace=settings.namespace,
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
        "release_id": preflight["release_id"],
        "node": settings.node,
        "maintenance_window_end": maintenance_window_end.isoformat(),
    }
    baseline_node = preflight["node"]
    baseline_host: dict[str, Any] | None = None
    injected_at: datetime | None = None
    try:
        fixture.create()
        baseline_host = fixture.execute("snapshot")
        write_json_atomic(case_dir / "host-baseline.json", baseline_host)
        if not baseline_host["kmsg_writable"]:
            raise CaseError("/dev/kmsg is not writable from the host probe")
        if baseline_host["compute_clients"]:
            raise CaseError("target node has active NVIDIA compute clients")
        if baseline_host["fabric_manager"].get("ActiveState") != "active":
            raise CaseError("Fabric Manager is not active at baseline")
        if datetime.now(timezone.utc) >= maintenance_window_end:
            raise CaseError("approved maintenance window ended before injection")

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
        timeout = int(preflight["store"]["companion_window_seconds"]) + 300
        state = wait_for_workflow(
            settings,
            marker,
            injected_at,
            timeout,
            case_dir,
        )
        write_json_atomic(case_dir / "workflow-state.json", state)
        errors = workflow_errors(state)
        command = state["commands"][0] if state["commands"] else {}
        generation = int(state["agent"]["generation"])
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

        replay = {}
        if not errors:
            replay = replay_command(settings, command)
            write_json_atomic(case_dir / "replay.json", replay)
            if replay.get("status") != "SUCCEEDED":
                errors.append("isolated Node Agent replay did not return SUCCEEDED")
        after_replay_state = store_probe(
            settings,
            marker=marker,
            observed_after=injected_at,
        )
        write_json_atomic(
            case_dir / "store-after-replay.json",
            after_replay_state,
        )
        first_notification_ids = {
            item["notification_id"] for item in state.get("notifications") or []
        }
        replay_notification_ids = {
            item["notification_id"]
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
        post_node = node_snapshot(settings)
        write_json_atomic(case_dir / "node-postflight.json", post_node)
        if post_node != baseline_node:
            errors.append("target Kubernetes Node state drifted")
        if business_workloads(settings):
            errors.append("target node acquired a non-system workload")
        events = provider_events(settings, injected_at, datetime.now(timezone.utc))
        write_json_atomic(case_dir / "provider-events.json", events)
        if events:
            errors.append("provider mutation appeared during DESTR-010")
        result.update(
            {
                "verdict": "PASS" if not errors else "FAIL",
                "errors": errors,
                "marker": marker,
                "workflow": {
                    "request_id": (state.get("workflow") or {}).get("request_id"),
                    "status": (state.get("workflow") or {}).get("status"),
                    "official_action": (state.get("workflow") or {}).get(
                        "official_action"
                    ),
                },
                "remote_command_id": command.get("command_id"),
                "node_action_command_id": expected_command_id,
                "replay": replay,
                "notifications": after_replay_state.get("notifications") or [],
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
            final_node = node_snapshot(settings)
            result["final_node"] = final_node
            if final_node != baseline_node:
                result["verdict"] = "FAIL"
                result.setdefault("errors", []).append(
                    "final Kubernetes Node state differs from baseline"
                )
        except Exception as exc:
            result["postflight_error"] = f"{type(exc).__name__}: {exc}"
            result["verdict"] = "FAIL"
    write_json_atomic(case_dir / f"{CASE_ID}.json", result)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["verdict"] == "PASS" else 1


def plan_details(settings: Settings, preflight: dict[str, Any]) -> dict[str, Any]:
    return {
        "risk": "live-service-action",
        "release_id": preflight["release_id"],
        "target_node": settings.node,
        "mutation": (
            "write one synthetic XID 45 to the real host /dev/kmsg and allow "
            "the Node Agent to restart nvidia-fabricmanager exactly once"
        ),
        "preflight_identity": {
            "release_id": preflight["release_id"],
            "node_uid": preflight["node"]["uid"],
            "agent_generation": preflight["store"]["agent"]["generation"],
            "runtime_profile_version": preflight["store"]["profile"]["version"],
        },
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
            "node_state_must_equal_baseline": True,
        },
        "preflight": preflight,
    }


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

from __future__ import annotations

import json
import os
import subprocess
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
    write_json_atomic,
)
from scripts.e2e.regional.acceptance_scope import (  # noqa: E402
    FORMAL_SCOPE,
    current_acceptance_scope,
)

TERMINAL_WORKFLOW_STATUSES = {"SUCCEEDED", "FAILED", "BLOCKED"}
PROVIDER_MUTATIONS = {
    "BatchDeleteClusterNodes",
    "BatchRebootClusterNodes",
    "BatchReplaceClusterNodes",
    "DeleteClusterNodes",
    "RebootClusterNodes",
    "ReplaceClusterNodes",
}


def provider_event_actor_matches_role(
    event: dict[str, str],
    expected_role_arn: str,
) -> bool:
    expected_role_name = expected_role_arn.rsplit("/", 1)[-1]
    session_issuer_role_name = event.get("session_issuer_role_name", "")
    if session_issuer_role_name:
        return session_issuer_role_name == expected_role_name
    return expected_role_name in event.get("username", "")


class RegionalFixtureError(RuntimeError):
    pass


def required(value: str, label: str) -> str:
    result = value.strip()
    if not result:
        raise RegionalFixtureError(f"{label} is required")
    return result


def predecessor_evidence(
    path: Path,
    expected_case_id: str,
) -> dict[str, Any]:
    scope = current_acceptance_scope()
    if scope.selective:
        return {
            "path": str(path),
            "case_id": expected_case_id,
            "expected_case_id": expected_case_id,
            "verdict": "SKIPPED_BY_OPERATOR",
            "status": "SKIPPED_BY_OPERATOR",
            "valid": True,
            "execution_allowed": True,
            "evidence_valid": False,
            **scope.result_fields(),
            "error": None,
        }
    if not path.is_file():
        return {
            "path": str(path),
            "case_id": expected_case_id,
            "verdict": "MISSING",
            "valid": False,
            "execution_allowed": False,
            "evidence_valid": False,
            **scope.plan_fields(),
            "formal_sequence_satisfied": False,
            "error": "predecessor evidence does not exist",
        }
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "path": str(path),
            "case_id": expected_case_id,
            "verdict": "INVALID",
            "valid": False,
            "execution_allowed": False,
            "evidence_valid": False,
            **scope.plan_fields(),
            "formal_sequence_satisfied": False,
            "error": f"cannot read predecessor evidence: {exc}",
        }
    if not isinstance(value, dict):
        return {
            "path": str(path),
            "case_id": expected_case_id,
            "verdict": "INVALID",
            "valid": False,
            "execution_allowed": False,
            "evidence_valid": False,
            **scope.plan_fields(),
            "formal_sequence_satisfied": False,
            "error": "predecessor evidence is not a JSON object",
        }
    actual_case_id = str(value.get("case_id") or "")
    verdict = str(value.get("verdict") or "")
    evidence_scope = str(value.get("execution_scope") or FORMAL_SCOPE)
    formal_sequence_satisfied = bool(
        value.get(
            "formal_sequence_satisfied",
            evidence_scope == FORMAL_SCOPE,
        )
    )
    evidence_valid = actual_case_id == expected_case_id and verdict == "PASS"
    valid = (
        evidence_valid and evidence_scope == FORMAL_SCOPE and formal_sequence_satisfied
    )
    if not evidence_valid:
        error = "predecessor case must have verdict PASS"
    elif evidence_scope != FORMAL_SCOPE or not formal_sequence_satisfied:
        error = "selective evidence cannot satisfy a formal predecessor"
    else:
        error = None
    return {
        "path": str(path),
        "case_id": actual_case_id,
        "expected_case_id": expected_case_id,
        "verdict": verdict,
        "valid": valid,
        "execution_allowed": valid,
        "evidence_valid": evidence_valid,
        "evidence_execution_scope": evidence_scope,
        **scope.plan_fields(),
        "formal_sequence_satisfied": valid,
        "error": error,
    }


def waiting_step_executions(workflow: dict[str, Any]) -> list[dict[str, Any]]:
    result = []
    for item in workflow.get("step_executions") or []:
        if not isinstance(item, dict) or item.get("status") != "WAITING":
            continue
        result.append(
            {
                key: item.get(key)
                for key in (
                    "step_index",
                    "operation",
                    "status",
                    "adapter_operation_id",
                    "details",
                    "error",
                )
            }
        )
    return result


@dataclass(frozen=True)
class RegionalLiveSettings:
    cpu_kubeconfig: Path
    gpu_kubeconfig: Path
    gpu_context: str
    namespace: str
    cluster_id: str
    region: str

    def __post_init__(self) -> None:
        if not self.cpu_kubeconfig.is_file():
            raise ValueError("CPU kubeconfig does not exist")
        if not self.gpu_kubeconfig.is_file():
            raise ValueError("GPU kubeconfig does not exist")
        if not all(
            (
                self.gpu_context,
                self.namespace,
                self.cluster_id,
                self.region,
            )
        ):
            raise ValueError("regional live settings contain an empty identity")

    def environment(self) -> dict[str, str]:
        return {
            "CPU_KUBECONFIG": str(self.cpu_kubeconfig),
            "GPU_KUBECONFIG": str(self.gpu_kubeconfig),
            "GPU_EKS_CONTEXT": self.gpu_context,
            "GPU_FAULT_NAMESPACE": self.namespace,
            "GPU_FAULT_CLUSTER_ID": self.cluster_id,
            "AWS_REGION": self.region,
        }


def settings_from_arguments(arguments: Any) -> RegionalLiveSettings:
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
    return RegionalLiveSettings(
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
        region=required(
            arguments.region
            or os.getenv("AWS_REGION", "")
            or os.getenv("AWS_DEFAULT_REGION", ""),
            "AWS Region",
        ),
    )


STORE_PROBE = r"""
import json
import os
import sys
from datetime import datetime, timezone

from gpu_fault.app import ApplicationContext
from gpu_fault.hyperpod import hyperpod_submission_idempotency_key
from gpu_fault.store import NotFoundError

(
    cluster_id,
    node_id,
    marker,
    observed_after_text,
    job_id,
    attempt_id,
    hyperpod_cluster,
) = sys.argv[1:]
store = ApplicationContext.from_environment().store
observed_after = (
    datetime.fromisoformat(observed_after_text.replace("Z", "+00:00"))
    if observed_after_text
    else None
)
events = (
    store.list_xid_events(
        cluster_id,
        node_id,
        observed_after=observed_after,
    )
    if node_id
    else []
)
matching_events = [
    item
    for item in events
    if (
        not marker
        or marker in str(item.raw_message or "")
        or marker in str(item.event_id)
    )
]
event = matching_events[-1] if matching_events else None
decision = None
if event is not None:
    try:
        decision = store.get_xid_policy_decision(event.event_id)
    except NotFoundError:
        pass
incident = store.get_incident_by_event(event.event_id) if event is not None else None
workflow = (
    store.get_workflow(incident.workflow_request_id)
    if incident is not None and incident.workflow_request_id
    else None
)
commands = [
    item
    for item in store.list_remote_commands()
    if workflow is not None and item.workflow_request_id == workflow.request_id
]
notifications = []
if incident is not None:
    for item in store.list_notifications():
        if item.incident_id != incident.incident_id:
            continue
        result = store.get_notification_result(item.notification_id)
        notifications.append({
            "notification": item.model_dump(mode="json"),
            "result": (
                result.model_dump(mode="json")
                if result is not None else None
            ),
        })
observations = [
    item.model_dump(mode="json")
    for item in store.list_attempt_observations(cluster_id)
    if (not job_id or item.job_id == job_id)
    and (not attempt_id or item.attempt_id == attempt_id)
]
restart_budget = None
if job_id:
    try:
        restart_budget = store.get_restart_budget(cluster_id, job_id).model_dump(
            mode="json"
        )
    except NotFoundError:
        pass
agent = None
profile = None
if node_id:
    try:
        agent_model = store.get_agent(cluster_id, node_id)
        agent = agent_model.model_dump(mode="json")
        profile = store.get_profile(
            agent_model.runtime_profile_version
        ).model_dump(mode="json")
    except (KeyError, NotFoundError):
        pass
submission = None
if hyperpod_cluster and commands:
    restart_command = next(
        (
            item for item in commands
            if item.step.operation.value == "RESTART_NODE"
        ),
        None,
    )
    if restart_command is not None:
        submission_key = (
            restart_command.result_details.get("submission_idempotency_key")
            or hyperpod_submission_idempotency_key(
                workflow.request_id,
                restart_command.step_index,
                restart_command.step.operation,
            )
        )
        try:
            submission = store.get_hyperpod_submission(
                hyperpod_cluster,
                submission_key,
            )
        except NotFoundError:
            pass
print(json.dumps({
    "release_id": os.getenv("GPU_FAULT_RELEASE_ID"),
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
    "observations": observations,
    "restart_budget": restart_budget,
    "agent": agent,
    "profile": profile,
    "submission": (
        submission.model_dump(mode="json")
        if submission is not None else None
    ),
    "queue": store.processor_queue_stats(),
    "remote_commands": store.remote_command_stats(),
}, sort_keys=True, default=str))
"""


EXECUTOR_XID_POST = r"""
import json
import os
import sys
import time
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, urlopen

from gpu_fault.collectors.sinks import HttpEventSink

payload = json.loads(sys.argv[1])
sink = HttpEventSink(
    os.environ["GPU_FAULT_CONTROL_PLANE_URL"],
    bearer_token=os.environ["GPU_FAULT_CONTROL_PLANE_TOKEN"],
    timeout_seconds=15,
    max_attempts=1,
)
accepted = sink.post("/v1/collector-events/nvidia-kernel", payload)
request_id = accepted.get("processor_request_id")
receipt = None
if request_id:
    base = os.environ["GPU_FAULT_CONTROL_PLANE_URL"].rstrip("/")
    headers = {
        "Authorization": "Bearer " + os.environ["GPU_FAULT_CONTROL_PLANE_TOKEN"],
        "X-GPU-Fault-Cluster-ID": os.environ["GPU_FAULT_CLUSTER_ID"],
    }
    deadline = time.monotonic() + 180
    while time.monotonic() < deadline:
        request = Request(
            base + "/v1/processor/requests/" + quote(request_id, safe=""),
            headers=headers,
        )
        try:
            with urlopen(request, timeout=15) as response:
                body = response.read()
                receipt = {
                    "status": response.status,
                    "body": json.loads(body) if body else {},
                }
                if response.status != 202:
                    break
        except HTTPError as exc:
            body = exc.read()
            receipt = {
                "status": exc.code,
                "body": json.loads(body) if body else {},
            }
            break
        time.sleep(2)
print(json.dumps({
    "accepted": accepted,
    "receipt": receipt,
}, sort_keys=True))
"""


class RegionalLiveFixture:
    def __init__(self, settings: RegionalLiveSettings) -> None:
        self.settings = settings

    @staticmethod
    def run(
        command: list[str],
        *,
        input_text: str | None = None,
        check: bool = True,
        timeout: int = 300,
        cwd: Path | None = None,
        env: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        completed = subprocess.run(
            command,
            input=input_text,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
            cwd=cwd,
            env=env,
        )
        if check and completed.returncode:
            raise RegionalFixtureError(
                f"command failed ({completed.returncode}): "
                f"{' '.join(command)}; stderr={completed.stderr.strip()}"
            )
        return completed

    def kubectl(
        self,
        plane: str,
        *arguments: str,
        input_text: str | None = None,
        check: bool = True,
        timeout: int = 300,
        all_namespaces: bool = False,
        namespace: str | None = None,
    ) -> str:
        command = ["kubectl", "--kubeconfig"]
        if plane == "cpu":
            command.append(str(self.settings.cpu_kubeconfig))
        elif plane == "gpu":
            command.extend(
                [
                    str(self.settings.gpu_kubeconfig),
                    "--context",
                    self.settings.gpu_context,
                ]
            )
        else:
            raise ValueError(f"unknown Kubernetes plane: {plane}")
        if all_namespaces and namespace is not None:
            raise ValueError("all_namespaces and namespace are mutually exclusive")
        if all_namespaces:
            command.extend(arguments)
            command.append("--all-namespaces")
        else:
            command.extend(["-n", namespace or self.settings.namespace])
            command.extend(arguments)
        return self.run(
            command,
            input_text=input_text,
            check=check,
            timeout=timeout,
        ).stdout

    def ready_pods(self, plane: str, app: str) -> list[dict[str, Any]]:
        value = json.loads(
            self.kubectl(
                plane,
                "get",
                "pod",
                "-l",
                f"app={app}",
                "-o",
                "json",
            )
        )
        result = []
        for item in value.get("items", []):
            statuses = item.get("status", {}).get("containerStatuses", [])
            if item.get("status", {}).get("phase") != "Running":
                continue
            if not statuses or not all(
                bool(status.get("ready")) for status in statuses
            ):
                continue
            result.append(
                {
                    "name": item["metadata"]["name"],
                    "uid": item["metadata"]["uid"],
                    "node": item["spec"].get("nodeName"),
                }
            )
        return sorted(result, key=lambda item: str(item["name"]))

    def ready_pod(self, plane: str, app: str) -> str:
        pods = self.ready_pods(plane, app)
        if not pods:
            raise RegionalFixtureError(f"no Ready {plane} Pod for app={app}")
        return str(pods[0]["name"])

    def pod_python(
        self,
        plane: str,
        app: str,
        script: str,
        *arguments: str,
        timeout: int = 180,
    ) -> dict[str, Any]:
        last_error: Exception | None = None
        for _attempt in range(3):
            try:
                output = self.kubectl(
                    plane,
                    "exec",
                    "-i",
                    self.ready_pod(plane, app),
                    "--",
                    "python3",
                    "-",
                    *arguments,
                    input_text=script,
                    timeout=timeout,
                )
                value = json.loads(output.splitlines()[-1])
                if not isinstance(value, dict):
                    raise RegionalFixtureError("Pod probe did not return a JSON object")
                return cast(dict[str, Any], value)
            except Exception as exc:
                last_error = exc
                time.sleep(1)
        raise RegionalFixtureError(f"{plane} Pod probe failed: {last_error}")

    def cpu_python(self, script: str, *arguments: str) -> dict[str, Any]:
        return self.pod_python(
            "cpu",
            "gpu-fault-api-ha",
            script,
            *arguments,
        )

    def executor_python(
        self,
        script: str,
        *arguments: str,
        timeout: int = 180,
    ) -> dict[str, Any]:
        return self.pod_python(
            "gpu",
            "gpu-fault-cluster-executor",
            script,
            *arguments,
            timeout=timeout,
        )

    def store_snapshot(
        self,
        *,
        node: str = "",
        marker: str = "",
        observed_after: datetime | None = None,
        job_id: str = "",
        attempt_id: str = "",
        hyperpod_cluster: str = "",
    ) -> dict[str, Any]:
        result = self.cpu_python(
            STORE_PROBE,
            self.settings.cluster_id,
            node,
            marker,
            observed_after.isoformat() if observed_after is not None else "",
            job_id,
            attempt_id,
            hyperpod_cluster,
        )
        if not result.get("release_id"):
            result["release_id"] = self.release_id()
        return result

    def release_id(self) -> str:
        value = json.loads(
            self.kubectl(
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

    def post_xid_event(self, payload: dict[str, Any]) -> dict[str, Any]:
        if payload.get("cluster_id") != self.settings.cluster_id:
            raise RegionalFixtureError("XID payload cluster ID does not match settings")
        return self.executor_python(
            EXECUTOR_XID_POST,
            json.dumps(payload, sort_keys=True),
            timeout=240,
        )

    def wait_for_workflow(
        self,
        *,
        node: str,
        marker: str,
        observed_after: datetime,
        case_dir: Path,
        timeout_seconds: int,
        job_id: str = "",
        attempt_id: str = "",
        hyperpod_cluster: str = "",
        terminal: bool = True,
    ) -> dict[str, Any]:
        deadline = time.monotonic() + timeout_seconds
        timeline: list[dict[str, Any]] = []
        observed_waiting: dict[tuple[int, str], dict[str, Any]] = {}
        last: dict[str, Any] = {}
        while time.monotonic() < deadline:
            last = self.store_snapshot(
                node=node,
                marker=marker,
                observed_after=observed_after,
                job_id=job_id,
                attempt_id=attempt_id,
                hyperpod_cluster=hyperpod_cluster,
            )
            workflow = last.get("workflow") or {}
            current_waiting = waiting_step_executions(workflow)
            for execution in current_waiting:
                step_index = execution.get("step_index")
                observed_waiting[
                    (
                        step_index if isinstance(step_index, int) else -1,
                        str(execution.get("operation") or ""),
                    )
                ] = execution
            waiting_evidence = [
                observed_waiting[key] for key in sorted(observed_waiting)
            ]
            timeline.append(
                {
                    "observed_at": datetime.now(timezone.utc).isoformat(),
                    "event_id": (last.get("event") or {}).get("event_id"),
                    "decision": (last.get("decision") or {}).get("disposition"),
                    "workflow_status": workflow.get("status"),
                    "completed_operations": workflow.get("completed_operations") or [],
                    "command_statuses": [
                        item.get("status") for item in last.get("commands") or []
                    ],
                    "waiting_step_executions": current_waiting,
                    "submission_state": ((last.get("submission") or {}).get("state")),
                }
            )
            write_json_atomic(
                case_dir / "timeline.json",
                {
                    "entries": timeline,
                    "observed_waiting_step_executions": waiting_evidence,
                },
            )
            status = workflow.get("status")
            if status and (not terminal or status in TERMINAL_WORKFLOW_STATUSES):
                return {
                    **last,
                    "observed_waiting_step_executions": waiting_evidence,
                }
            time.sleep(5)
        raise RegionalFixtureError(
            f"workflow did not reach the requested state: {last}"
        )

    def node_snapshot(self, node: str) -> dict[str, Any]:
        value = json.loads(self.kubectl("gpu", "get", "node", node, "-o", "json"))
        annotations = value["metadata"].get("annotations", {})
        return {
            "name": value["metadata"]["name"],
            "uid": value["metadata"]["uid"],
            "boot_id": value.get("status", {}).get("nodeInfo", {}).get("bootID"),
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
            "gpu_allocatable": value["status"]
            .get("allocatable", {})
            .get("nvidia.com/gpu"),
            "ownership_annotations": {
                key: item
                for key, item in annotations.items()
                if key.startswith("gpu-fault.io/")
                and not key.startswith("gpu-fault.io/installer-")
            },
        }

    def gpu_nodes(self) -> list[dict[str, Any]]:
        value = json.loads(self.kubectl("gpu", "get", "node", "-o", "json"))
        result = []
        for item in value.get("items", []):
            gpu_count = int(
                item.get("status", {}).get("allocatable", {}).get("nvidia.com/gpu", 0)
            )
            if gpu_count <= 0:
                continue
            result.append(
                {
                    "name": item["metadata"]["name"],
                    "uid": item["metadata"]["uid"],
                    "gpu_allocatable": gpu_count,
                    "ready": next(
                        (
                            condition["status"]
                            for condition in item["status"].get("conditions", [])
                            if condition["type"] == "Ready"
                        ),
                        None,
                    ),
                    "unschedulable": item["spec"].get("unschedulable", False),
                    "taints": item["spec"].get("taints", []),
                }
            )
        return sorted(result, key=lambda item: str(item["name"]))

    def node_metadata(self, node: str) -> dict[str, Any]:
        value = json.loads(self.kubectl("gpu", "get", "node", node, "-o", "json"))
        labels = value["metadata"].get("labels", {})
        return {
            "name": value["metadata"]["name"],
            "labels": labels,
            "product": next(
                (
                    labels[key]
                    for key in (
                        "nvidia.com/gpu.product",
                        "nvidia.com/gpu.machine",
                        "beta.kubernetes.io/instance-type",
                        "node.kubernetes.io/instance-type",
                    )
                    if labels.get(key)
                ),
                None,
            ),
        }

    def gpu_workloads(self) -> list[dict[str, Any]]:
        value = json.loads(
            self.kubectl(
                "gpu",
                "get",
                "pod",
                "-o",
                "json",
                all_namespaces=True,
            )
        )
        result = []
        for item in value.get("items", []):
            phase = item.get("status", {}).get("phase")
            if phase in {"Succeeded", "Failed"}:
                continue
            gpu_count = 0
            for container in item.get("spec", {}).get("containers", []):
                resources = container.get("resources", {})
                for values in (
                    resources.get("requests", {}),
                    resources.get("limits", {}),
                ):
                    try:
                        gpu_count = max(
                            gpu_count,
                            int(values.get("nvidia.com/gpu", 0)),
                        )
                    except (TypeError, ValueError):
                        pass
            if gpu_count <= 0:
                continue
            result.append(
                {
                    "namespace": item["metadata"].get("namespace"),
                    "name": item["metadata"].get("name"),
                    "node": item.get("spec", {}).get("nodeName"),
                    "phase": phase,
                    "gpu_count": gpu_count,
                }
            )
        return sorted(
            result,
            key=lambda item: (
                str(item["namespace"]),
                str(item["name"]),
            ),
        )

    def business_workloads(self, node: str) -> list[dict[str, str]]:
        value = json.loads(
            self.kubectl(
                "gpu",
                "get",
                "pod",
                "--field-selector",
                f"spec.nodeName={node},status.phase=Running",
                "-o",
                "json",
                all_namespaces=True,
            )
        )
        system_namespaces = {
            "aws-hyperpod",
            "cert-manager",
            "hyperpod-inference-system",
            "kube-system",
            "kubeflow",
        }
        result = []
        for item in value.get("items", []):
            namespace = str(item["metadata"].get("namespace", ""))
            gpu_count = 0
            for container in item.get("spec", {}).get("containers", []):
                resources = container.get("resources", {})
                for values in (
                    resources.get("requests", {}),
                    resources.get("limits", {}),
                ):
                    try:
                        gpu_count = max(
                            gpu_count,
                            int(values.get("nvidia.com/gpu", 0)),
                        )
                    except (TypeError, ValueError):
                        pass
            if namespace in system_namespaces:
                continue
            if namespace == self.settings.namespace and gpu_count <= 0:
                continue
            result.append(
                {
                    "namespace": namespace,
                    "name": str(item["metadata"].get("name", "")),
                }
            )
        return result

    def cpu_blast_snapshot(self) -> dict[str, Any]:
        nodes = json.loads(
            self.kubectl(
                "cpu",
                "get",
                "node",
                "-o",
                "json",
            )
        )
        jobs = json.loads(
            self.kubectl(
                "cpu",
                "get",
                "job",
                "-o",
                "json",
                all_namespaces=True,
            )
        )
        events = json.loads(
            self.kubectl(
                "cpu",
                "get",
                "event",
                "-o",
                "json",
                all_namespaces=True,
            )
        )
        return {
            "nodes": {
                item["metadata"]["name"]: {
                    "taints": sorted(
                        (
                            taint["key"],
                            taint.get("value"),
                            taint["effect"],
                        )
                        for taint in item["spec"].get("taints", [])
                    ),
                    "unschedulable": item["spec"].get("unschedulable", False),
                    "labels": {
                        key: value
                        for key, value in item["metadata"].get("labels", {}).items()
                        if key.startswith("gpu-fault.io/")
                    },
                    "annotations": {
                        key: value
                        for key, value in item["metadata"]
                        .get("annotations", {})
                        .items()
                        if key.startswith("gpu-fault.io/")
                    },
                }
                for item in nodes.get("items", [])
            },
            "gpu_fault_jobs": sorted(
                (
                    str(item["metadata"].get("namespace", "")),
                    str(item["metadata"].get("name", "")),
                )
                for item in jobs.get("items", [])
                if any(
                    key.startswith("gpu-fault.io/")
                    for key in item["metadata"].get("labels", {})
                )
            ),
            "eviction_events": sorted(
                (
                    str(item["metadata"].get("uid", "")),
                    str(item.get("reason", "")),
                    str(item.get("involvedObject", {}).get("name", "")),
                )
                for item in events.get("items", [])
                if item.get("reason") in {"Evicted", "TaintManagerEviction"}
            ),
        }

    def provider_events(
        self,
        started_at: datetime,
        ended_at: datetime,
    ) -> list[dict[str, str]]:
        value = json.loads(
            self.run(
                [
                    "aws",
                    "cloudtrail",
                    "lookup-events",
                    "--region",
                    self.settings.region,
                    "--start-time",
                    started_at.isoformat(),
                    "--end-time",
                    ended_at.isoformat(),
                    "--lookup-attributes",
                    "AttributeKey=EventSource,AttributeValue=sagemaker.amazonaws.com",
                    "--output",
                    "json",
                ],
                timeout=180,
            ).stdout
        )
        result = []
        for item in value.get("Events", []):
            if item.get("EventName") not in PROVIDER_MUTATIONS:
                continue
            session_issuer_role_name = ""
            try:
                detail = json.loads(str(item.get("CloudTrailEvent") or "{}"))
            except json.JSONDecodeError:
                detail = {}
            identity = detail.get("userIdentity") or {}
            session_context = identity.get("sessionContext") or {}
            session_issuer = session_context.get("sessionIssuer") or {}
            session_issuer_arn = str(session_issuer.get("arn") or "")
            if ":role/" in session_issuer_arn:
                session_issuer_role_name = session_issuer_arn.rsplit("/", 1)[-1]
            result.append(
                {
                    "event_name": str(item.get("EventName") or ""),
                    "event_time": str(item.get("EventTime") or ""),
                    "username": str(item.get("Username") or ""),
                    "session_issuer_role_name": session_issuer_role_name,
                }
            )
        return result

    def wait_node_ready(
        self,
        node: str,
        *,
        timeout_seconds: int,
        expected_boot_id: str | None = None,
    ) -> dict[str, Any]:
        deadline = time.monotonic() + timeout_seconds
        last: dict[str, Any] = {}
        while time.monotonic() < deadline:
            try:
                last = self.node_snapshot(node)
            except Exception:
                time.sleep(10)
                continue
            if last.get("ready") == "True" and (
                expected_boot_id is None or last.get("boot_id") != expected_boot_id
            ):
                return last
            time.sleep(10)
        raise RegionalFixtureError(f"node did not return Ready: {last}")

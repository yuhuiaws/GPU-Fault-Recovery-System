from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any, cast


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.e2e.regional.acceptance_runner_common import (  # noqa: E402
    write_json_atomic,
)
from scripts.e2e.regional.host_probe_fixture import (  # noqa: E402
    HostProbeFixture,
    HostProbeSettings,
)
from scripts.e2e.regional.managed_workload_fixture import (  # noqa: E402
    TRAINING_IMAGE,
)
from scripts.e2e.regional.regional_live_fixture import (  # noqa: E402
    RegionalFixtureError,
    RegionalLiveFixture,
)


SPARE_LABEL = "gpu-fault.io/spare"
SPARE_RESERVATION_ANNOTATION = "gpu-fault.io/spare-reservation"
SPARE_POOL_STATE_ANNOTATION = "gpu-fault.io/spare-pool-state"
HYPERPOD_HEALTH_LABEL = "sagemaker.amazonaws.com/node-health-status"
INSTANCE_GROUP_LABEL = "sagemaker.amazonaws.com/instance-group-name"
INSTANCE_TYPE_LABELS = (
    "node.kubernetes.io/instance-type",
    "beta.kubernetes.io/instance-type",
)
OWNERSHIP_ANNOTATIONS = (
    "gpu-fault.io/incident-id",
    "gpu-fault.io/fencing-token",
    "gpu-fault.io/previous-unschedulable",
)
QUARANTINE_TAINT = "gpu-fault.io/quarantined"
PROVIDER_REPLACE_EVENTS = {
    "BatchDeleteClusterNodes",
    "BatchReplaceClusterNodes",
    "DeleteClusterNodes",
    "ReplaceClusterNodes",
}
SERVICE_PROBE = Path(__file__).with_name("probes") / "warm_spare_node_probe.py"


STORE_PROBE = r"""
import json
import sys

from gpu_fault.app import ApplicationContext
from gpu_fault.store import NotFoundError

cluster_id, event_id, job_id, attempt_id = sys.argv[1:]
store = ApplicationContext.from_environment().store
incident = store.get_incident_by_event(event_id) if event_id else None
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
notifications = [
    item.model_dump(mode="json")
    for item in store.list_notifications()
    if incident is not None and item.incident_id == incident.incident_id
]
notification_results = {}
for item in notifications:
    result = store.get_notification_result(item["notification_id"])
    notification_results[item["notification_id"]] = (
        result.model_dump(mode="json") if result is not None else None
    )
markers = [
    item.model_dump(mode="json")
    for item in store.list_markers()
    if incident is not None and item.incident_id == incident.incident_id
]
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
agents = [
    item.model_dump(mode="json")
    for item in store.list_agents(cluster_id)
]
print(json.dumps({
    "incident": (
        incident.model_dump(mode="json") if incident is not None else None
    ),
    "workflow": (
        workflow.model_dump(mode="json") if workflow is not None else None
    ),
    "commands": [item.model_dump(mode="json") for item in commands],
    "notifications": notifications,
    "notification_results": notification_results,
    "markers": markers,
    "observations": observations,
    "restart_budget": restart_budget,
    "agents": agents,
}, sort_keys=True, default=str))
"""


SYNTHETIC_REPLACEMENT_POST = r"""
import json
import os
import sys
from urllib.error import HTTPError
from urllib.request import Request, urlopen

payload = json.loads(sys.argv[1])
body = json.dumps(payload, separators=(",", ":")).encode()
request = Request(
    "http://127.0.0.1:8080/v1/admin/test/node-replacement",
    data=body,
    method="POST",
    headers={
        "Content-Type": "application/json",
        "X-GPU-Fault-Execution-Token": os.environ["GPU_FAULT_EXECUTION_TOKEN"],
    },
)
try:
    with urlopen(request, timeout=30) as response:
        content = response.read()
        result = {
            "status": response.status,
            "body": json.loads(content) if content else {},
        }
except HTTPError as exc:
    content = exc.read()
    result = {
        "status": exc.code,
        "body": json.loads(content) if content else {},
    }
print(json.dumps(result, sort_keys=True))
"""


RELEASE_SPARES = r"""
import json
import sys

from gpu_fault.cluster_executor import executor_from_environment

nodes = json.loads(sys.argv[1])
incident_id = sys.argv[2]
executor = executor_from_environment()
adapter = next(
    item
    for item in executor.adapters
    if getattr(item, "owner", "") == "gpu-fault-hyperpod-adapter"
)
coordinator = adapter.spare_coordinator
if coordinator is None:
    raise RuntimeError("spare coordinator is unavailable")
coordinator.release(nodes, incident_id)
print(json.dumps({
    "released_nodes": nodes,
    "incident_id": incident_id,
}, sort_keys=True))
"""


REACTIVATE_AGENT = r"""
import json
import sys

from gpu_fault.app import ApplicationContext
from gpu_fault.fleet import AgentLifecycleState, AgentTransitionRequest

cluster_id, node_id = sys.argv[1:]
context = ApplicationContext.from_environment()
record = context.store.get_agent(cluster_id, node_id)
if record.lifecycle_state is AgentLifecycleState.REVOKED:
    if context.fleet_registry is None or not record.transition_id:
        raise RuntimeError("revoked agent cannot be reactivated")
    request = AgentTransitionRequest(
        expected_generation=record.generation,
        transition_id=record.transition_id,
        reason="regional acceptance cleanup",
    )
    record = context.fleet_registry.reactivate_agent(cluster_id, node_id, request)
print(json.dumps(record.model_dump(mode="json"), sort_keys=True, default=str))
"""


CREATE_RESTORE_WORKFLOW = r"""
import json
import sys
from datetime import datetime, timezone
from uuid import uuid4

from gpu_fault.app import ApplicationContext
from gpu_fault.models import (
    IncidentState,
    WorkflowOperation,
    WorkflowRequest,
    WorkflowStatus,
    WorkflowStepSpec,
)

incident_id, node_id, profile_version, reason = sys.argv[1:]
context = ApplicationContext.from_environment()
store = context.store
incident = store.get_incident(incident_id)
if incident.workflow_request_id:
    current = store.get_workflow(incident.workflow_request_id)
    if current.status in {
        WorkflowStatus.PENDING,
        WorkflowStatus.RUNNING,
        WorkflowStatus.SAFETY_PENDING,
    }:
        raise RuntimeError("incident still has an active workflow")
now = datetime.now(timezone.utc)
workflow = WorkflowRequest(
    request_id=f"workflow-validated-restore-{uuid4()}",
    incident_id=incident.incident_id,
    runtime_profile_version=profile_version,
    status=WorkflowStatus.PENDING,
    official_action="RESTORE_SCHEDULING",
    fencing_token=incident.fencing_token,
    official_steps=[
        WorkflowStepSpec(
            operation=operation,
            execution_owner=(
                "gpu-fault-kubernetes-adapter"
                if operation is WorkflowOperation.RESTORE_SCHEDULING
                else "gpu-fault-validation-adapter"
            ),
            node_ids=[node_id],
        )
        for operation in (
            WorkflowOperation.VALIDATE_GPU,
            WorkflowOperation.VALIDATE_HOST,
            WorkflowOperation.VALIDATE_FABRIC,
            WorkflowOperation.RESTORE_SCHEDULING,
        )
    ],
    created_at=now,
    updated_at=now,
)
incident = incident.model_copy(
    update={
        "state": IncidentState.ACTION_PENDING,
        "node_ids": sorted(set(incident.node_ids) | {node_id}),
        "workflow_request_id": workflow.request_id,
        "reasons": [*incident.reasons, reason],
        "updated_at": now,
    }
)
store.save_incident_and_workflow(incident, workflow)
context.dispatcher.wake()
print(json.dumps({
    "workflow_request_id": workflow.request_id,
    "incident_id": incident.incident_id,
    "node_id": node_id,
}, sort_keys=True))
"""


WORKFLOW_BY_ID = r"""
import json
import sys

from gpu_fault.app import ApplicationContext

workflow = ApplicationContext.from_environment().store.get_workflow(sys.argv[1])
print(json.dumps(workflow.model_dump(mode="json"), sort_keys=True, default=str))
"""


FLEET_READINESS = r"""
import json
import sys

from gpu_fault.app import ApplicationContext

cluster_id, node_id = sys.argv[1:]
context = ApplicationContext.from_environment()
if context.fleet_registry is None:
    raise RuntimeError("fleet registry is unavailable")
report = context.fleet_registry.readiness(cluster_id, [node_id])
print(json.dumps(report.model_dump(mode="json"), sort_keys=True, default=str))
"""


@dataclass(frozen=True)
class NodePatch:
    labels: dict[str, str | None]
    annotations: dict[str, str | None]
    unschedulable: bool | None = None


def agent_by_node(state: dict[str, Any], node: str) -> dict[str, Any] | None:
    # Exactly one match or nothing: two Agents claiming one node is the case a
    # warm spare must never be built on, so an ambiguous fleet reads as absent
    # rather than as the first row that happened to come back.
    matches = [
        item for item in state.get("agents") or [] if item.get("node_id") == node
    ]
    return cast(dict[str, Any], matches[0]) if len(matches) == 1 else None


def instance_type(snapshot: dict[str, Any]) -> str | None:
    return next(
        (
            snapshot["labels"].get(key)
            for key in INSTANCE_TYPE_LABELS
            if snapshot["labels"].get(key)
        ),
        None,
    )


class WarmSpareLiveFixture:
    def __init__(self, regional: RegionalLiveFixture, hyperpod_cluster: str) -> None:
        self.regional = regional
        self.hyperpod_cluster = hyperpod_cluster

    def node_snapshot(self, node: str) -> dict[str, Any]:
        value = json.loads(
            self.regional.kubectl("gpu", "get", "node", node, "-o", "json")
        )
        metadata = value["metadata"]
        labels = metadata.get("labels", {})
        annotations = metadata.get("annotations", {})
        return {
            "name": metadata["name"],
            "uid": metadata["uid"],
            "resource_version": metadata.get("resourceVersion"),
            "provider_id": value.get("spec", {}).get("providerID"),
            "ready": next(
                (
                    item["status"]
                    for item in value.get("status", {}).get("conditions", [])
                    if item.get("type") == "Ready"
                ),
                None,
            ),
            "gpu_allocatable": int(
                value.get("status", {}).get("allocatable", {}).get("nvidia.com/gpu", 0)
            ),
            "unschedulable": bool(value.get("spec", {}).get("unschedulable", False)),
            "taints": value.get("spec", {}).get("taints", []),
            "labels": {
                key: labels.get(key)
                for key in (
                    SPARE_LABEL,
                    HYPERPOD_HEALTH_LABEL,
                    INSTANCE_GROUP_LABEL,
                    *INSTANCE_TYPE_LABELS,
                )
            },
            "annotations": {
                key: annotations.get(key)
                for key in (
                    SPARE_RESERVATION_ANNOTATION,
                    SPARE_POOL_STATE_ANNOTATION,
                    *OWNERSHIP_ANNOTATIONS,
                )
            },
        }

    def spare_nodes(self) -> list[str]:
        value = json.loads(
            self.regional.kubectl(
                "gpu",
                "get",
                "node",
                "-l",
                f"{SPARE_LABEL}=true",
                "-o",
                "json",
            )
        )
        return sorted(str(item["metadata"]["name"]) for item in value.get("items", []))

    def provider_inventory(self) -> dict[str, Any]:
        value = json.loads(
            self.regional.run(
                [
                    "aws",
                    "sagemaker",
                    "list-cluster-nodes",
                    "--region",
                    self.regional.settings.region,
                    "--cluster-name",
                    self.hyperpod_cluster,
                    "--output",
                    "json",
                ],
                timeout=180,
            ).stdout
        )
        rows = sorted(
            [
                {
                    "node_logical_id": item.get("NodeLogicalId"),
                    "instance_id": item.get("InstanceId"),
                    "instance_group": item.get("InstanceGroupName"),
                    "instance_type": item.get("InstanceType"),
                    "status": (item.get("InstanceStatus", {}).get("Status")),
                }
                for item in value.get("ClusterNodeSummaries", [])
            ],
            key=lambda item: (
                str(item["node_logical_id"]),
                str(item["instance_id"]),
            ),
        )
        encoded = json.dumps(rows, sort_keys=True).encode()
        return {
            "count": len(rows),
            "sha256": hashlib.sha256(encoded).hexdigest(),
            "nodes": rows,
        }

    def cluster_recovery(self) -> dict[str, Any]:
        value = json.loads(
            self.regional.run(
                [
                    "aws",
                    "sagemaker",
                    "describe-cluster",
                    "--region",
                    self.regional.settings.region,
                    "--cluster-name",
                    self.hyperpod_cluster,
                    "--output",
                    "json",
                ],
                timeout=180,
            ).stdout
        )
        return {
            "cluster_name": value.get("ClusterName"),
            "status": value.get("ClusterStatus"),
            "node_recovery": value.get("NodeRecovery"),
        }

    def executor_environment(self) -> list[dict[str, str | None]]:
        result = []
        for pod in self.regional.ready_pods("gpu", "gpu-fault-cluster-executor"):
            output = self.regional.kubectl(
                "gpu",
                "exec",
                str(pod["name"]),
                "--",
                "python3",
                "-c",
                (
                    "import json,os; print(json.dumps({"
                    "'spare_failover':os.getenv("
                    "'GPU_FAULT_ENABLE_HYPERPOD_SPARE_FAILOVER'),"
                    "'remote_state':os.getenv("
                    "'GPU_FAULT_CLUSTER_EXECUTOR_REMOTE_STATE'),"
                    "'allow_replace':os.getenv("
                    "'GPU_FAULT_ALLOW_HYPERPOD_REPLACE'),"
                    "'spare_label':os.getenv("
                    "'GPU_FAULT_HYPERPOD_SPARE_LABEL')}))"
                ),
                timeout=60,
            )
            value = json.loads(output.splitlines()[-1])
            if not isinstance(value, dict):
                raise RegionalFixtureError("executor environment is not a JSON object")
            result.append(
                {
                    "pod": str(pod["name"]),
                    **{
                        str(key): (str(item) if item is not None else None)
                        for key, item in value.items()
                    },
                }
            )
        return result

    def synthetic_replacement_gates(self) -> list[dict[str, str | None]]:
        result = []
        for pod in self.regional.ready_pods("cpu", "gpu-fault-api-ha"):
            output = self.regional.kubectl(
                "cpu",
                "exec",
                str(pod["name"]),
                "--",
                "python3",
                "-c",
                (
                    "import json,os; print(json.dumps({"
                    "'enabled':os.getenv("
                    "'GPU_FAULT_ENABLE_SYNTHETIC_REPLACEMENT_TESTS')}))"
                ),
                timeout=60,
            )
            value = json.loads(output.splitlines()[-1])
            result.append(
                {
                    "pod": str(pod["name"]),
                    "enabled": (
                        str(value.get("enabled"))
                        if value.get("enabled") is not None
                        else None
                    ),
                }
            )
        return result

    def store_snapshot(
        self,
        *,
        event_id: str = "",
        job_id: str = "",
        attempt_id: str = "",
    ) -> dict[str, Any]:
        return cast(
            dict[str, Any],
            self.regional.cpu_python(
                STORE_PROBE,
                self.regional.settings.cluster_id,
                event_id,
                job_id,
                attempt_id,
            ),
        )

    def post_synthetic_replacement(
        self,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        if payload.get("cluster_id") != self.regional.settings.cluster_id:
            raise RegionalFixtureError(
                "replacement payload cluster ID does not match settings"
            )
        return cast(
            dict[str, Any],
            self.regional.cpu_python(
                SYNTHETIC_REPLACEMENT_POST,
                json.dumps(payload, sort_keys=True),
            ),
        )

    def wait_for_workflow(
        self,
        *,
        event_id: str,
        job_id: str,
        attempt_id: str,
        case_dir: Path,
        timeout_seconds: int,
    ) -> dict[str, Any]:
        deadline = time.monotonic() + timeout_seconds
        timeline = []
        last: dict[str, Any] = {}
        while time.monotonic() < deadline:
            last = self.store_snapshot(
                event_id=event_id,
                job_id=job_id,
                attempt_id=attempt_id,
            )
            workflow = last.get("workflow") or {}
            timeline.append(
                {
                    "observed_at": datetime.now(timezone.utc).isoformat(),
                    "workflow_status": workflow.get("status"),
                    "completed_operations": workflow.get("completed_operations") or [],
                    "command_statuses": [
                        item.get("status") for item in last.get("commands") or []
                    ],
                }
            )
            write_json_atomic(case_dir / "timeline.json", {"entries": timeline})
            if workflow.get("status") in {"SUCCEEDED", "FAILED", "BLOCKED"}:
                return last
            time.sleep(5)
        raise RegionalFixtureError(
            f"warm-spare workflow did not reach a terminal state: {last}"
        )

    def release_spares(self, nodes: list[str], incident_id: str) -> dict[str, Any]:
        return cast(
            dict[str, Any],
            self.regional.executor_python(
                RELEASE_SPARES,
                json.dumps(nodes),
                incident_id,
            ),
        )

    def reactivate_agent(self, node: str) -> dict[str, Any]:
        return cast(
            dict[str, Any],
            self.regional.cpu_python(
                REACTIVATE_AGENT,
                self.regional.settings.cluster_id,
                node,
            ),
        )

    def create_restore_workflow(
        self,
        *,
        incident_id: str,
        node: str,
        profile_version: str,
        reason: str,
    ) -> dict[str, Any]:
        return cast(
            dict[str, Any],
            self.regional.cpu_python(
                CREATE_RESTORE_WORKFLOW,
                incident_id,
                node,
                profile_version,
                reason,
            ),
        )

    def wait_workflow_id(
        self,
        workflow_id: str,
        *,
        timeout_seconds: int = 900,
    ) -> dict[str, Any]:
        deadline = time.monotonic() + timeout_seconds
        last: dict[str, Any] = {}
        while time.monotonic() < deadline:
            last = self.regional.cpu_python(WORKFLOW_BY_ID, workflow_id)
            if last.get("status") in {"SUCCEEDED", "FAILED", "BLOCKED"}:
                return last
            time.sleep(5)
        raise RegionalFixtureError(
            f"restore workflow did not reach a terminal state: {last}"
        )

    def wait_agent_active(
        self,
        node: str,
        *,
        timeout_seconds: int = 180,
    ) -> dict[str, Any]:
        deadline = time.monotonic() + timeout_seconds
        last: dict[str, Any] = {}
        while time.monotonic() < deadline:
            state = self.store_snapshot()
            matches = [
                item
                for item in state.get("agents") or []
                if item.get("node_id") == node
            ]
            last = matches[0] if matches else {}
            if last.get("lifecycle_state") == "ACTIVE":
                return last
            time.sleep(5)
        raise RegionalFixtureError(f"agent did not return ACTIVE: {last}")

    def fleet_readiness(self, node: str) -> dict[str, Any]:
        return cast(
            dict[str, Any],
            self.regional.cpu_python(
                FLEET_READINESS,
                self.regional.settings.cluster_id,
                node,
            ),
        )

    def wait_fleet_readiness(
        self,
        node: str,
        *,
        ready: bool,
        timeout_seconds: int = 180,
    ) -> dict[str, Any]:
        deadline = time.monotonic() + timeout_seconds
        last: dict[str, Any] = {}
        while time.monotonic() < deadline:
            last = self.fleet_readiness(node)
            if bool(last.get("ready")) is ready:
                return last
            time.sleep(5)
        raise RegionalFixtureError(f"fleet readiness did not become {ready}: {last}")

    def wait_node_ready(
        self,
        node: str,
        *,
        ready: bool,
        timeout_seconds: int,
    ) -> dict[str, Any]:
        deadline = time.monotonic() + timeout_seconds
        last: dict[str, Any] = {}
        expected = "True" if ready else "False"
        while time.monotonic() < deadline:
            try:
                last = self.node_snapshot(node)
            except Exception:
                if not ready:
                    return {"name": node, "ready": "Unknown"}
                time.sleep(5)
                continue
            if last.get("ready") == expected:
                return last
            time.sleep(5)
        raise RegionalFixtureError(f"node did not reach Ready={expected}: {last}")


class NodeMutationFixture:
    def __init__(
        self,
        warm: WarmSpareLiveFixture,
        node: str,
        *,
        label_keys: tuple[str, ...] = (),
        annotation_keys: tuple[str, ...] = (),
        track_unschedulable: bool = False,
    ) -> None:
        self.warm = warm
        self.node = node
        self.label_keys = label_keys
        self.annotation_keys = annotation_keys
        self.track_unschedulable = track_unschedulable
        self.baseline = warm.node_snapshot(node)

    def apply(self, patch: NodePatch) -> None:
        metadata: dict[str, Any] = {}
        if patch.labels:
            metadata["labels"] = patch.labels
        if patch.annotations:
            metadata["annotations"] = patch.annotations
        body: dict[str, Any] = {"metadata": metadata}
        if patch.unschedulable is not None:
            body["spec"] = {"unschedulable": patch.unschedulable}
        self.warm.regional.kubectl(
            "gpu",
            "patch",
            "node",
            self.node,
            "--type=merge",
            "-p",
            json.dumps(body, sort_keys=True),
        )

    def restore(self) -> dict[str, Any]:
        labels = {key: self.baseline["labels"].get(key) for key in self.label_keys}
        annotations = {
            key: self.baseline["annotations"].get(key) for key in self.annotation_keys
        }
        self.apply(
            NodePatch(
                labels=labels,
                annotations=annotations,
                unschedulable=(
                    bool(self.baseline["unschedulable"])
                    if self.track_unschedulable
                    else None
                ),
            )
        )
        return self.warm.node_snapshot(self.node)


class GpuHolderFixture:
    def __init__(
        self,
        warm: WarmSpareLiveFixture,
        *,
        node: str,
        run_id: str,
        image: str = TRAINING_IMAGE,
    ) -> None:
        self.warm = warm
        self.node = node
        suffix = hashlib.sha256(f"{node}\0{run_id}".encode()).hexdigest()[:12]
        self.name = f"gpu-fault-spare-holder-{suffix}"
        self.image = image

    def manifest(self) -> dict[str, Any]:
        return {
            "apiVersion": "v1",
            "kind": "Pod",
            "metadata": {
                "name": self.name,
                "namespace": self.warm.regional.settings.namespace,
                "labels": {
                    "app": "gpu-fault-spare-holder",
                    "gpu-fault.io/acceptance-run": self.name,
                },
            },
            "spec": {
                "nodeName": self.node,
                "restartPolicy": "Never",
                "activeDeadlineSeconds": 900,
                "terminationGracePeriodSeconds": 0,
                "tolerations": [{"operator": "Exists"}],
                "containers": [
                    {
                        "name": "holder",
                        "image": self.image,
                        "imagePullPolicy": "IfNotPresent",
                        "command": [
                            "/bin/bash",
                            "-ceu",
                            (
                                "exec python3 -c 'import torch,time; "
                                'x=torch.ones(1,device="cuda:0"); '
                                "print(float(x.item()),flush=True); time.sleep(840)'"
                            ),
                        ],
                        "resources": {
                            "requests": {
                                "cpu": "100m",
                                "memory": "1Gi",
                                "nvidia.com/gpu": "1",
                            },
                            "limits": {
                                "cpu": "1",
                                "memory": "2Gi",
                                "nvidia.com/gpu": "1",
                            },
                        },
                    }
                ],
            },
        }

    def create(self) -> None:
        self.warm.regional.kubectl(
            "gpu",
            "apply",
            "-f",
            "-",
            input_text=json.dumps(self.manifest()),
        )
        self.warm.regional.kubectl(
            "gpu",
            "wait",
            "--for=condition=Ready",
            f"pod/{self.name}",
            "--timeout=300s",
            timeout=330,
        )

    def cleanup(self) -> bool:
        self.warm.regional.kubectl(
            "gpu",
            "delete",
            "pod",
            self.name,
            "--ignore-not-found",
            "--wait=true",
            check=False,
            timeout=180,
        )
        return bool(
            self.warm.regional.kubectl(
                "gpu",
                "get",
                "pod",
                self.name,
                "--ignore-not-found",
                "-o",
                "name",
                check=False,
            ).strip()
        )


class WarmSpareServiceFixture:
    def __init__(
        self,
        warm: WarmSpareLiveFixture,
        *,
        node: str,
        image: str,
        case_id: str,
        run_id: str,
    ) -> None:
        self.host = HostProbeFixture(
            HostProbeSettings(
                kubeconfig=warm.regional.settings.gpu_kubeconfig,
                context=warm.regional.settings.gpu_context,
                namespace=warm.regional.settings.namespace,
                node=node,
                image=image,
                case_id=case_id,
                run_id=run_id,
                probe_script=SERVICE_PROBE,
                active_deadline_seconds=1800,
            )
        )
        self.run_id = run_id
        self.service = ""

    def create(self) -> None:
        self.host.create()

    def stop(self, service: str, *, restore_seconds: int = 180) -> dict[str, Any]:
        self.service = service
        return cast(
            dict[str, Any],
            self.host.execute(
                "stop-with-failsafe",
                "--service",
                service,
                "--run-id",
                self.run_id,
                "--restore-seconds",
                str(restore_seconds),
                timeout=180,
            ),
        )

    def restore(self) -> dict[str, Any]:
        if not self.service:
            return {"restored": False, "reason": "service was not stopped"}
        return cast(
            dict[str, Any],
            self.host.execute(
                "restore-service",
                "--service",
                self.service,
                "--run-id",
                self.run_id,
                timeout=180,
            ),
        )

    def cleanup(self) -> dict[str, bool]:
        return cast(dict[str, bool], self.host.cleanup())

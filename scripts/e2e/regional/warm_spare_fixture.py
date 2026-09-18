from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any, Sequence, cast


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
    component_python,
)


SPARE_LABEL = "gpu-fault.io/spare"
SPARE_RESERVATION_ANNOTATION = "gpu-fault.io/spare-reservation"
SPARE_POOL_STATE_ANNOTATION = "gpu-fault.io/spare-pool-state"
SPARE_RESERVED_AT_ANNOTATION = "gpu-fault.io/spare-reserved-at"
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
# Every spare-pool annotation a snapshot reports. ``spare-reserved-at`` was
# missing until 2026-09-08: DESTR-022 wrote a back-dated reservation and read
# its own snapshot back without the key, failing "was not written" on a value
# that was on the node.
SNAPSHOT_ANNOTATIONS = (
    SPARE_RESERVATION_ANNOTATION,
    SPARE_RESERVED_AT_ANNOTATION,
    SPARE_POOL_STATE_ANNOTATION,
    *OWNERSHIP_ANNOTATIONS,
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
    "commands": [
        item.model_dump(mode="json", exclude={"lease_token"}) for item in commands
    ],
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


CLOSE_INCIDENT_POST = r"""
import json
import os
import sys
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, urlopen

incident_id, reason, operator = sys.argv[1:]
body = json.dumps(
    {"reason": reason, "operator": operator}, separators=(",", ":")
).encode()
request = Request(
    "http://127.0.0.1:8080/v1/incidents/" + quote(incident_id, safe="") + "/close",
    data=body,
    method="POST",
    headers={
        "Content-Type": "application/json",
        "X-GPU-Fault-Execution-Token": os.environ["GPU_FAULT_EXECUTION_TOKEN"],
    },
)
try:
    with urlopen(request, timeout=60) as response:
        content = response.read()
        result = {
            "status": response.status,
            "body": json.loads(content) if content else {},
        }
except HTTPError as exc:
    content = exc.read()
    try:
        decoded = json.loads(content) if content else {}
    except ValueError:
        decoded = {"detail": content.decode(errors="replace")}
    result = {"status": exc.code, "body": decoded}
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


# The workflow itself comes from the product
# (``gpu_fault.orchestration.validated_restore``), the same builder
# ``gpu-fault-admin submit-remediation --disposition restore`` uses, so the
# fixture cannot drift from what an operator gets. argv and the printed keys
# are the fixture's contract and stay as they were.
CREATE_RESTORE_WORKFLOW = r"""
import json
import sys
from datetime import datetime, timezone

from gpu_fault.app import ApplicationContext
from gpu_fault.models import WorkflowStatus, WorkflowOperation, workflow_is_open
from gpu_fault.execution.node_action_uncertainty import (
    has_unresolved_node_action, refresh_remote_action_state, quiesce_needs_restoration,
)
from gpu_fault.recovery_safety import recovery_safety_errors
from gpu_fault.orchestration.validated_restore import (
    build_validated_restore_workflow,
)

incident_id, node_id, profile_version, reason = sys.argv[1:]
context = ApplicationContext.from_environment()
store = context.store
incident = store.get_incident(incident_id)
if store.list_active_workflow_incidents(
    incident.cluster_id, node_ids=set(incident.node_ids) | {node_id}
):
    raise RuntimeError("node still has an active or operator-held workflow")
if incident.workflow_request_id:
    current = store.get_workflow(incident.workflow_request_id)
    if workflow_is_open(current.status, current.blocked_kind) or (
        current.status is WorkflowStatus.BLOCKED and current.blocked_kind is None
    ):
        raise RuntimeError("incident still has an active or unclassified blocked workflow")
    commands = store.list_remote_commands(workflow_request_ids=[current.request_id])
    current = refresh_remote_action_state(store, current)
    errors = recovery_safety_errors(
        [current.model_dump(mode="json")],
        [command.model_dump(mode="json", exclude={"lease_token"}) for command in commands],
    )
    if errors or has_unresolved_node_action(current):
        raise RuntimeError("physical action outcome requires operator reconciliation")
    if quiesce_needs_restoration(
        current,
        None,
        settling_operations={WorkflowOperation.RESTART_NODE, WorkflowOperation.REPLACE_NODE},
    ):
        raise RuntimeError("product quiesce restoration is not complete")
now = datetime.now(timezone.utc)
incident, workflow = build_validated_restore_workflow(
    incident,
    operator="acceptance-fixture",
    reference=None,
    now=now,
    node_ids=[node_id],
    runtime_profile_version=profile_version,
    reason=reason,
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


INCIDENT_BY_ID = r"""
import json
import sys

from gpu_fault.app import ApplicationContext

incident = ApplicationContext.from_environment().store.get_incident(sys.argv[1])
print(json.dumps(incident.model_dump(mode="json"), sort_keys=True, default=str))
"""


INCIDENT_CLEANUP_STATE = r"""
import json
import sys

from gpu_fault.app import ApplicationContext

cluster_id, incident_id = sys.argv[1:]
store = ApplicationContext.from_environment().store
incident = store.get_incident(incident_id)
if incident.cluster_id != cluster_id:
    raise RuntimeError("cleanup incident belongs to another cluster")
workflow = (
    store.get_workflow(incident.workflow_request_id)
    if incident.workflow_request_id else None
)
active = [
    {"request_id": item.request_id, "status": item.status.value}
    for _, item in store.list_active_workflow_incidents(
        cluster_id, node_ids=set(incident.node_ids)
    )
]
commands = [
    {"command_id": item.command_id, "status": item.status.value}
    for item in store.list_remote_commands()
    if item.cluster_id == cluster_id and item.status.value not in {"SUCCEEDED", "FAILED"}
]
print(json.dumps({
    "incident": incident.model_dump(mode="json"),
    "workflow_status": workflow.status.value if workflow else None,
    "active_workflows": active,
    "open_commands": commands,
}, sort_keys=True, default=str))
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


OPEN_WORKFLOWS = r"""
import json
import sys

from gpu_fault.app import ApplicationContext

cluster_id, job_id, nodes_text = sys.argv[1:]
nodes = {item for item in nodes_text.split(",") if item}
store = ApplicationContext.from_environment().store
matches = []
if job_id:
    matches.extend(store.list_active_workflow_incidents(cluster_id, job_id=job_id))
if nodes:
    matches.extend(store.list_active_workflow_incidents(cluster_id, node_ids=nodes))
seen = set()
workflows = []
for incident, workflow in matches:
    if workflow.request_id in seen:
        continue
    seen.add(workflow.request_id)
    workflows.append({
        "request_id": workflow.request_id,
        "status": workflow.status.value,
        "incident_id": incident.incident_id,
        "node_ids": list(incident.node_ids),
        "job_id": incident.job_id,
    })
print(json.dumps({"workflows": workflows}, sort_keys=True, default=str))
"""


ACTIVE_BUDGET_CLAIMS = r"""
import json
from datetime import datetime, timezone

from gpu_fault.app import ApplicationContext
from gpu_fault.models import WorkflowStatus

store = ApplicationContext.from_environment().store
now = datetime.now(timezone.utc)
# The same counting rule the store applies when a workflow claims its budget:
# RUNNING rows whose execution lease is still live hold their scopes; every
# other row holds nothing.
held = []
for workflow in store.list_workflows({WorkflowStatus.RUNNING}, limit=500):
    expires = workflow.execution_lease_expires_at
    if expires is None or expires <= now:
        continue
    held.append({
        "request_id": workflow.request_id,
        "claims": sorted(workflow.remediation_budget_claims),
        "limits": dict(workflow.remediation_budget_limits),
    })
print(json.dumps(
    {"observed_at": now.isoformat(), "workflows": held},
    sort_keys=True,
    default=str,
))
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

    def node_snapshot(
        self,
        node: str,
        *,
        extra_label_keys: tuple[str, ...] = (),
        extra_annotation_keys: tuple[str, ...] = (),
    ) -> dict[str, Any]:
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
                    *extra_label_keys,
                )
            },
            "annotations": {
                key: annotations.get(key)
                for key in (*SNAPSHOT_ANNOTATIONS, *extra_annotation_keys)
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
                component_python("gpu"),
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
                component_python("cpu"),
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
        return self.regional.cpu_python(
            STORE_PROBE,
            self.regional.settings.cluster_id,
            event_id,
            job_id,
            attempt_id,
        )

    def post_synthetic_replacement(
        self,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        if payload.get("cluster_id") != self.regional.settings.cluster_id:
            raise RegionalFixtureError(
                "replacement payload cluster ID does not match settings"
            )
        # One attempt: the route persists a finding and opens a workflow, so
        # a kubectl retry after a lost receipt would inject the fault twice.
        return self.regional.cpu_python(
            SYNTHETIC_REPLACEMENT_POST,
            json.dumps(payload, sort_keys=True),
            attempts=1,
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
        return self.regional.executor_python(
            RELEASE_SPARES,
            json.dumps(nodes),
            incident_id,
            attempts=1,
        )

    def reactivate_agent(self, node: str) -> dict[str, Any]:
        return self.regional.cpu_python(
            REACTIVATE_AGENT,
            self.regional.settings.cluster_id,
            node,
            attempts=1,
        )

    def incident_by_id(self, incident_id: str) -> dict[str, Any]:
        return self.regional.cpu_python(INCIDENT_BY_ID, incident_id)

    def wait_incident_idle(
        self,
        incident_id: str,
        *,
        timeout_seconds: int = 900,
        quiet_seconds: int = 15,
    ) -> dict[str, Any]:
        """Wait until the incident has no workflow a restore would race.

        The validated-restore path refuses an incident whose workflow is still
        PENDING/RUNNING/SAFETY_PENDING, and rightly so: two workflows on one
        node is exactly what fencing exists to prevent. A caller that wants to
        restore has to wait the current one out instead of forcing it.
        """

        deadline = time.monotonic() + timeout_seconds
        last: dict[str, Any] = {}
        quiet_since: float | None = None
        while time.monotonic() < deadline:
            last = self.regional.cpu_python(
                INCIDENT_CLEANUP_STATE, self.regional.settings.cluster_id, incident_id
            )
            incident = last.get("incident")
            active = last.get("active_workflows")
            commands = last.get("open_commands")
            if (
                not isinstance(incident, dict)
                or incident.get("incident_id") != incident_id
                or incident.get("cluster_id") != self.regional.settings.cluster_id
                or not isinstance(active, list)
                or not isinstance(commands, list)
                or "workflow_status" not in last
            ):
                raise RegionalFixtureError("incident cleanup state is incomplete")
            idle = (
                not active
                and not commands
                and last["workflow_status"]
                in {None, "SUCCEEDED", "FAILED", "BLOCKED", "SUPERSEDED"}
            )
            now = time.monotonic()
            quiet_since = (
                (quiet_since if quiet_since is not None else now) if idle else None
            )
            if quiet_since is not None and now - quiet_since >= quiet_seconds:
                return incident
            time.sleep(5)
        raise RegionalFixtureError(
            f"incident workflows and commands did not become quiescent: {last}"
        )

    def create_restore_workflow(
        self,
        *,
        incident_id: str,
        node: str,
        profile_version: str,
        reason: str,
    ) -> dict[str, Any]:
        return self.regional.cpu_python(
            CREATE_RESTORE_WORKFLOW,
            incident_id,
            node,
            profile_version,
            reason,
            attempts=1,
        )

    def close_incident(
        self, incident_id: str, *, reason: str, operator: str
    ) -> dict[str, Any]:
        """``POST /v1/incidents/{id}/close`` from inside the CPU ingress Pod.

        The operator exit for an ESCALATED incident whose node is back
        (DESTR-018 product gap). Returns ``{"status", "body"}``; a 409 names
        the open workflow or the state that refused it, a 404 is a release
        that predates the route -- the caller decides whether to fall back.
        Not retried: a close that landed must not be repeated blindly.
        """

        return self.regional.cpu_python(
            CLOSE_INCIDENT_POST, incident_id, reason, operator, attempts=1
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
            last = matches[0] if len(matches) == 1 else {}
            if last.get("lifecycle_state") == "ACTIVE":
                return last
            time.sleep(5)
        raise RegionalFixtureError(f"agent did not return ACTIVE: {last}")

    def fleet_readiness(self, node: str) -> dict[str, Any]:
        return self.regional.cpu_python(
            FLEET_READINESS,
            self.regional.settings.cluster_id,
            node,
        )

    def open_workflows(
        self,
        *,
        job_id: str = "",
        nodes: Sequence[str] = (),
    ) -> list[dict[str, Any]]:
        """Open workflows over ``job_id`` or any of ``nodes``, read in the store.

        Open is the store's own definition -- executable rows plus BLOCKED rows
        that still occupy their node -- so a preflight that reads ``[]`` here is
        refusing on the same rows the dispatcher would collide with.
        """

        for node in nodes:
            if "," in node or not node:
                raise ValueError("node names must be non-empty, no comma")
        value = self.regional.cpu_python(
            OPEN_WORKFLOWS,
            self.regional.settings.cluster_id,
            job_id,
            ",".join(nodes),
        )
        return cast(list[dict[str, Any]], value.get("workflows") or [])

    def active_budget_claims(self) -> list[dict[str, Any]]:
        """RUNNING workflows with a live lease and the budget scopes they hold."""

        value = self.regional.cpu_python(ACTIVE_BUDGET_CLAIMS)
        return cast(list[dict[str, Any]], value.get("workflows") or [])

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
            if last.get("ready") is ready:
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
        """Wait for the node's Ready condition to reach (or leave) ``True``.

        ``ready=False`` accepts ``Unknown`` as well, and has to: stopping
        kubelet does not make it report unhealthy, it makes it stop reporting,
        so the node lifecycle controller sets ``Ready=Unknown`` and taints the
        node unreachable. Insisting on the literal string ``False`` waited out
        the whole window while the node was already unreachable, and failed
        with "node did not reach Ready=False" over a snapshot that plainly
        showed it was not Ready. Everything downstream -- the scheduler, and
        the warm-spare candidate gate this case exercises -- treats anything
        other than ``True`` as not Ready.
        """

        deadline = time.monotonic() + timeout_seconds
        last: dict[str, Any] = {}
        while time.monotonic() < deadline:
            try:
                last = self.node_snapshot(node)
            except Exception:
                time.sleep(5)
                continue
            observed_ready = last.get("ready")
            if observed_ready in ("True", "False", "Unknown") and (
                (observed_ready == "True") is ready
            ):
                return last
            time.sleep(5)
        expected = "True" if ready else "not True"
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
        allow_reclaimed_reservation: bool = False,
    ) -> None:
        self.warm = warm
        self.node = node
        self.label_keys = label_keys
        self.annotation_keys = annotation_keys
        self.track_unschedulable = track_unschedulable
        self.allow_reclaimed_reservation = allow_reclaimed_reservation
        self.baseline = self._snapshot()
        self._written_labels: dict[str, str | None] = {}
        self._written_annotations: dict[str, str | None] = {}
        self._written_unschedulable: bool | None = None

    def _snapshot(self) -> dict[str, Any]:
        value = self.warm.node_snapshot(
            self.node,
            extra_label_keys=self.label_keys,
            extra_annotation_keys=self.annotation_keys,
        )
        if not value.get("uid") or not value.get("resource_version"):
            raise RegionalFixtureError("node UID or resourceVersion is unknown")
        if value.get("name") != self.node:
            raise RegionalFixtureError("node snapshot names another target")
        if hasattr(self, "baseline") and value["uid"] != self.baseline["uid"]:
            raise RegionalFixtureError("node UID changed; mutation and restore refused")
        return value

    def apply(self, patch: NodePatch) -> None:
        if (
            set(patch.labels) - set(self.label_keys)
            or set(patch.annotations) - set(self.annotation_keys)
            or (patch.unschedulable is not None and not self.track_unschedulable)
        ):
            raise RegionalFixtureError("node patch contains untracked fields")
        current = self._snapshot()
        for section, desired, written in (
            ("labels", patch.labels, self._written_labels),
            ("annotations", patch.annotations, self._written_annotations),
        ):
            for key in desired:
                expected = written.get(key, self.baseline[section].get(key))
                if current[section].get(key) != expected:
                    raise RegionalFixtureError(f"tracked node field changed: {key}")
        if patch.unschedulable is not None:
            expected_cordon = (
                self._written_unschedulable
                if self._written_unschedulable is not None
                else self.baseline["unschedulable"]
            )
            if current["unschedulable"] != expected_cordon:
                raise RegionalFixtureError("tracked node cordon state changed")
        # Record intent before transport: an exception can mean the ACK was lost.
        self._written_labels.update(patch.labels)
        self._written_annotations.update(patch.annotations)
        if patch.unschedulable is not None:
            self._written_unschedulable = patch.unschedulable
        self._patch(current, patch)

    def _patch(self, current: dict[str, Any], patch: NodePatch) -> None:
        for key in (
            *OWNERSHIP_ANNOTATIONS,
            SPARE_RESERVATION_ANNOTATION,
            SPARE_RESERVED_AT_ANNOTATION,
            SPARE_POOL_STATE_ANNOTATION,
        ):
            if key not in self.annotation_keys and (
                current["annotations"].get(key) != self.baseline["annotations"].get(key)
            ):
                raise RegionalFixtureError("node ownership or reservation changed")
        if current["taints"] != self.baseline["taints"]:
            raise RegionalFixtureError("node taints changed before tracked mutation")
        if not self.track_unschedulable and (
            current["unschedulable"] != self.baseline["unschedulable"]
        ):
            raise RegionalFixtureError(
                "node scheduling changed before tracked mutation"
            )
        metadata: dict[str, Any] = {
            "uid": self.baseline["uid"],
            "resourceVersion": current["resource_version"],
        }
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
        after = self._snapshot()
        for section, desired in (
            ("labels", patch.labels),
            ("annotations", patch.annotations),
        ):
            if any(after[section].get(key) != value for key, value in desired.items()):
                raise RegionalFixtureError(f"node {section} patch was not observed")
        if patch.unschedulable is not None and (
            after["unschedulable"] != patch.unschedulable
        ):
            raise RegionalFixtureError("node cordon patch was not observed")

    def restore(self) -> dict[str, Any]:
        current = self._snapshot()
        annotations = current["annotations"]
        reclaimed = (
            self.allow_reclaimed_reservation
            and bool(self._written_annotations.get(SPARE_RESERVATION_ANNOTATION))
            and self._written_annotations.get(SPARE_POOL_STATE_ANNOTATION)
            == "ALLOCATED"
            and annotations.get(SPARE_RESERVATION_ANNOTATION) is None
            and annotations.get(SPARE_RESERVED_AT_ANNOTATION) is None
            and annotations.get(SPARE_POOL_STATE_ANNOTATION) == "AVAILABLE"
            and current["unschedulable"] is True
            and current["labels"].get(SPARE_LABEL) == "true"
        )
        desired: dict[str, dict[str, str | None]] = {"labels": {}, "annotations": {}}
        for section, written in (
            ("labels", self._written_labels),
            ("annotations", self._written_annotations),
        ):
            for key, value in written.items():
                baseline = self.baseline[section].get(key)
                observed = current[section].get(key)
                if observed == baseline:
                    continue
                if observed != value and not (
                    reclaimed and key == SPARE_POOL_STATE_ANNOTATION
                ):
                    raise RegionalFixtureError(f"tracked node field changed: {key}")
                desired[section][key] = baseline
        cordon = None
        if self._written_unschedulable is not None:
            baseline_cordon = self.baseline["unschedulable"]
            if current["unschedulable"] != baseline_cordon:
                if current["unschedulable"] != self._written_unschedulable:
                    raise RegionalFixtureError("tracked node cordon state changed")
                cordon = bool(baseline_cordon)
        if desired["labels"] or desired["annotations"] or cordon is not None:
            self._patch(
                current,
                NodePatch(
                    labels=desired["labels"],
                    annotations=desired["annotations"],
                    unschedulable=cordon,
                ),
            )
        return self._snapshot()


class GpuHolderFixture:
    # How long the holder keeps its GPU. A holder that exits while the workflow
    # it was meant to block is still running turns a shortage drill into a real
    # failover, so the runner creates it as late as possible and pins its own
    # wait to end before ``deadline_at``.
    HOLD_SECONDS = 840
    DEADLINE_MARGIN_SECONDS = 60

    def __init__(
        self,
        warm: WarmSpareLiveFixture,
        *,
        node: str,
        run_id: str,
        image: str = TRAINING_IMAGE,
        hold_seconds: int = HOLD_SECONDS,
    ) -> None:
        if hold_seconds < 1:
            raise ValueError("hold_seconds must be positive")
        self.warm = warm
        self.node = node
        suffix = hashlib.sha256(f"{node}\0{run_id}".encode()).hexdigest()[:12]
        self.name = f"gpu-fault-spare-holder-{suffix}"
        self.image = image
        self.hold_seconds = hold_seconds
        self.deadline_at: datetime | None = None

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
                "activeDeadlineSeconds": (
                    self.hold_seconds + self.DEADLINE_MARGIN_SECONDS
                ),
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
                                "print(float(x.item()),flush=True); "
                                f"time.sleep({self.hold_seconds})'"
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
        # Taken before the apply: the container's sleep starts after the Pod is
        # scheduled and Ready, so this bound is conservative by construction.
        self.deadline_at = datetime.now(timezone.utc) + timedelta(
            seconds=self.hold_seconds
        )
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
            check=True,
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
                check=True,
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
        state_directory: Path,
        node_uid: str,
        plan_sha256: str,
        release_id: str,
        maintenance_expires_at: datetime,
    ) -> None:
        from scripts.e2e.regional.destr008_service_window import (
            ServiceWindowController,
        )

        self.host = HostProbeFixture(
            HostProbeSettings(
                state_directory=state_directory,
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
        self.window = ServiceWindowController(
            self.host,
            cluster_id=warm.regional.settings.cluster_id,
            node_uid=node_uid,
            plan_sha256=plan_sha256,
            release_id=release_id,
            maintenance_expires_at=maintenance_expires_at,
            read_node_uid=lambda: str(warm.node_snapshot(node)["uid"]),
        )

    @property
    def service(self) -> str:
        return self.window.service

    @property
    def failsafe_at(self) -> datetime | None:
        """The fixed restoration start, not a guarantee of restored health."""
        return self.window.failsafe_at

    @property
    def journal_path(self) -> Path:
        return self.window.path

    def create(self) -> None:
        self.window.create()

    def stop(
        self,
        service: str,
        *,
        restore_seconds: int = 180,
        delay_seconds: int = 0,
    ) -> dict[str, Any]:
        """Arm/ACK independent recovery, then schedule exactly one owned stop."""
        return self.window.stop(
            service, restore_seconds=restore_seconds, delay_seconds=delay_seconds
        )

    def restore(self) -> dict[str, Any]:
        return self.window.restore()

    def cleanup(self) -> dict[str, bool]:
        return self.window.cleanup()

    def resume_cleanup(self) -> dict[str, Any]:
        """Use the original journals; never create or schedule a new window."""
        return self.window.resume_cleanup()

    def close(self) -> None:
        """Release only the local controller lock; this is not cleanup."""
        self.window.close()

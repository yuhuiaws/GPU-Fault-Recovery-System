from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any

from scripts.e2e.regional.acceptance_runner_common import write_json_atomic
from scripts.e2e.regional.identity_acceptance_common import (
    FORBIDDEN_NAMESPACE,
    ClusterTarget,
    IdentityAcceptanceError,
    IdentitySite,
)


FLEET_CROSS_CLUSTER_PROBE = r"""
import json
import os
import sys

from gpu_fault.cluster_executor import (
    ClusterExecutorError,
    RegionalExecutorClient,
    RegionalFleetRegistry,
)
from gpu_fault.fleet import AgentTransitionRequest

other = sys.argv[1]
registry = RegionalFleetRegistry(
    RegionalExecutorClient(
        os.environ["GPU_FAULT_CONTROL_PLANE_URL"],
        os.environ["GPU_FAULT_CLUSTER_ID"],
        os.environ["GPU_FAULT_CONTROL_PLANE_TOKEN"],
        ca_file=os.environ["GPU_FAULT_CONTROL_PLANE_CA_FILE"],
    )
)
result = {}
for label, call in (
    ("list_agents", lambda: registry.list_agents(other)),
    (
        "revoke_agent",
        lambda: registry.revoke_agent(
            other,
            "probe-node",
            AgentTransitionRequest(
                expected_generation=1,
                transition_id="iso-003-probe",
                reason="iso-probe",
            ),
        ),
    ),
):
    try:
        call()
        result[label] = {"rejected": False}
    except ClusterExecutorError as exc:
        result[label] = {"rejected": True, "error": str(exc)}
print(json.dumps(result, sort_keys=True))
"""

AGENT_LIST_PROBE = r"""
import json
import sys
from gpu_fault.app import ApplicationContext
cluster_id = sys.argv[1]
agents = ApplicationContext.from_environment().store.list_agents(cluster_id)
print(json.dumps({
    "agents": [
        {
            "node_id": item.node_id,
            "generation": item.generation,
            "lifecycle_state": item.lifecycle_state.value,
        }
        for item in agents
    ]
}, sort_keys=True))
"""


def run_iso003(
    site: IdentitySite,
    primary: ClusterTarget,
    secondary: ClusterTarget,
) -> dict[str, Any]:
    regional = site.regional(primary)
    before = regional.cpu_python(AGENT_LIST_PROBE, secondary.cluster_id)
    pod = site.any_executor_pod(primary)
    result = site.pod_json(
        "gpu",
        primary,
        pod,
        FLEET_CROSS_CLUSTER_PROBE,
        secondary.cluster_id,
    )
    after = regional.cpu_python(AGENT_LIST_PROBE, secondary.cluster_id)
    checks = {
        "list_agents_rejected": result["list_agents"]["rejected"]
        and "cannot read agents for another cluster" in result["list_agents"]["error"],
        "revoke_agent_rejected": result["revoke_agent"]["rejected"]
        and "cannot transition an agent in another cluster"
        in result["revoke_agent"]["error"],
        "secondary_agent_state_unchanged": before == after,
    }
    return {
        "verdict": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "results": result,
        "limitations": [
            "The probe invokes the deployed executor client but performs no "
            "Fleet state transition."
        ],
    }


SPARE_HEALTH_PROBE = r"""
import json
import os
import ssl
import sys
import urllib.error
import urllib.request

requested = sys.argv[1]
payload = json.dumps({
    "cluster_id": requested,
    "node_aliases": ["iso-004-nonexistent-node"],
}).encode()
request = urllib.request.Request(
    os.environ["GPU_FAULT_CONTROL_PLANE_URL"].rstrip("/")
    + "/v1/regional/executors/spares/health",
    data=payload,
    method="POST",
    headers={
        "Authorization": "Bearer " + os.environ["GPU_FAULT_CONTROL_PLANE_TOKEN"],
        "Content-Type": "application/json",
        "X-GPU-Fault-Cluster-ID": os.environ["GPU_FAULT_CLUSTER_ID"],
    },
)
context = ssl.create_default_context(
    cafile=os.environ["GPU_FAULT_CONTROL_PLANE_CA_FILE"]
)
try:
    with urllib.request.urlopen(request, context=context, timeout=20) as response:
        body = json.load(response)
        print(json.dumps({"status": response.status, "body": body}, sort_keys=True))
except urllib.error.HTTPError as exc:
    print(json.dumps({
        "status": exc.code,
        "body": json.loads(exc.read() or b"{}"),
    }, sort_keys=True))
"""


def run_iso004(
    site: IdentitySite,
    primary: ClusterTarget,
    secondary: ClusterTarget,
) -> dict[str, Any]:
    pod = site.any_executor_pod(primary)
    cross = site.pod_json(
        "gpu",
        primary,
        pod,
        SPARE_HEALTH_PROBE,
        secondary.cluster_id,
    )
    local = site.pod_json(
        "gpu",
        primary,
        pod,
        SPARE_HEALTH_PROBE,
        primary.cluster_id,
    )
    detail = str((cross.get("body") or {}).get("detail") or "")
    checks = {
        "cross_cluster_rejected_403": cross["status"] == 403,
        "binding_error_is_precise": (
            "authenticated cluster does not match all payload cluster_id values"
            in detail
        ),
        "local_query_accepted": local["status"] == 200,
        "missing_node_is_not_ready": (local.get("body") or {}).get("ready") is False,
    }
    return {
        "verdict": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "cross_cluster": cross,
        "local": local,
        "limitations": [
            "The consistent-cluster branch queries a deliberately nonexistent "
            "node and performs no spare allocation."
        ],
    }


ALLOWLIST_WORKFLOW_PROBE = r"""
import json
import sys
from datetime import datetime, timezone

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

cluster_id, suffix, workload_id = sys.argv[1:]
now = datetime.now(timezone.utc)
incident = FaultIncident(
    incident_id=f"incident-{suffix}",
    event_id=f"event-{suffix}",
    event_type="ISO005_ACCEPTANCE",
    cluster_id=cluster_id,
    node_ids=[],
    policy_version="iso005/v1",
    policy_source="ACCEPTANCE",
    state=IncidentState.ACTION_PENDING,
    fencing_token=1,
    created_at=now,
    updated_at=now,
)
step = WorkflowStepSpec(
    operation=WorkflowOperation.STOP_WORKLOADS,
    execution_owner="gpu-fault-kubernetes-adapter",
    workload_ids=[workload_id],
)
workflow = WorkflowRequest(
    request_id=f"workflow-{suffix}",
    incident_id=incident.incident_id,
    status=WorkflowStatus.RUNNING,
    fencing_token=1,
    official_steps=[step],
    created_at=now,
    updated_at=now,
)
context = WorkflowStepContext(
    workflow=workflow,
    incident=incident,
    step=step,
    step_index=0,
    request=WorkflowExecutionRequest(expected_fencing_token=1),
    idempotency_key=f"{workflow.request_id}/0/STOP_WORKLOADS",
)
store = ApplicationContext.from_environment().store
adapter = RegionalRemoteWorkflowAdapter(
    store,
    owners={"gpu-fault-kubernetes-adapter"},
    operations={WorkflowOperation.STOP_WORKLOADS},
)
outcome = adapter.execute(context)
commands = [
    item.model_dump(mode="json")
    for item in store.list_remote_commands()
    if item.workflow_request_id == workflow.request_id
]
print(json.dumps({
    "outcome": outcome.model_dump(mode="json"),
    "commands": commands,
    "workflow_id": workflow.request_id,
}, sort_keys=True, default=str))
"""

REMOTE_COMMAND_PROBE = r"""
import json
import sys
from gpu_fault.app import ApplicationContext
workflow_id = sys.argv[1]
commands = [
    item.model_dump(mode="json")
    for item in ApplicationContext.from_environment().store.list_remote_commands()
    if item.workflow_request_id == workflow_id
]
print(json.dumps({"commands": commands}, sort_keys=True, default=str))
"""

REMOTE_COMMAND_DELETE_PROBE = r"""
import json
import sys
from gpu_fault.app import ApplicationContext
store = ApplicationContext.from_environment().store
deleted = []
for command_id in sys.argv[1:]:
    store._delete("remote_command", command_id)
    deleted.append(command_id)
print(json.dumps({"deleted": deleted}))
"""


def workload_snapshot(
    site: IdentitySite,
    target: ClusterTarget,
) -> dict[str, Any]:
    deployment = json.loads(
        site.regional(target).kubectl(
            "gpu",
            "get",
            "deployment",
            "regional-allowlist-probe",
            "-o",
            "json",
            namespace=FORBIDDEN_NAMESPACE,
        )
    )
    pods = json.loads(
        site.regional(target).kubectl(
            "gpu",
            "get",
            "pod",
            "-l",
            "app=regional-allowlist-probe",
            "-o",
            "json",
            namespace=FORBIDDEN_NAMESPACE,
        )
    )
    return {
        "deployment_uid": deployment["metadata"]["uid"],
        "replicas": deployment["spec"]["replicas"],
        "pod_uids": sorted(
            str(item["metadata"]["uid"]) for item in pods.get("items", [])
        ),
    }


def run_iso005(
    site: IdentitySite,
    target: ClusterTarget,
    *,
    case_dir: Path,
) -> dict[str, Any]:
    original_registry = site.registry()
    registration = next(
        item
        for item in original_registry
        if item.get("cluster_id") == target.cluster_id
    )
    original_allowlist = list(registration.get("allowed_namespaces") or [])
    if FORBIDDEN_NAMESPACE in original_allowlist:
        raise IdentityAcceptanceError(
            f"{FORBIDDEN_NAMESPACE} is already in the cluster allowlist"
        )
    manifest = {
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": {
            "name": "regional-allowlist-probe",
            "namespace": FORBIDDEN_NAMESPACE,
        },
        "spec": {
            "replicas": 1,
            "selector": {"matchLabels": {"app": "regional-allowlist-probe"}},
            "template": {
                "metadata": {"labels": {"app": "regional-allowlist-probe"}},
                "spec": {
                    "containers": [
                        {
                            "name": "pause",
                            "image": "registry.k8s.io/pause:3.10",
                            "resources": {
                                "requests": {"cpu": "1m", "memory": "4Mi"},
                                "limits": {"cpu": "10m", "memory": "16Mi"},
                            },
                        }
                    ]
                },
            },
        },
    }
    regional = site.regional(target)
    command_ids: list[str] = []
    result: dict[str, Any] = {"verdict": "FAIL"}
    try:
        site.gpu(target, "create", "namespace", FORBIDDEN_NAMESPACE)
        site.regional(target).kubectl(
            "gpu",
            "apply",
            "-f",
            "-",
            input_text=json.dumps(manifest),
            namespace=FORBIDDEN_NAMESPACE,
        )
        site.regional(target).kubectl(
            "gpu",
            "rollout",
            "status",
            "deployment/regional-allowlist-probe",
            "--timeout=180s",
            namespace=FORBIDDEN_NAMESPACE,
        )
        baseline = workload_snapshot(site, target)
        suffix = f"iso005-control-{int(time.time())}"
        control = regional.cpu_python(
            ALLOWLIST_WORKFLOW_PROBE,
            target.cluster_id,
            suffix,
            f"{FORBIDDEN_NAMESPACE}/deployment/regional-allowlist-probe",
        )
        command_ids.extend(item["command_id"] for item in control.get("commands") or [])
        updated = [dict(item) for item in original_registry]
        for item in updated:
            if item.get("cluster_id") == target.cluster_id:
                item["allowed_namespaces"] = [
                    *original_allowlist,
                    FORBIDDEN_NAMESPACE,
                ]
        site.write_registry(updated)
        site.rollout_control()
        suffix = f"iso005-executor-{int(time.time())}"
        executor = regional.cpu_python(
            ALLOWLIST_WORKFLOW_PROBE,
            target.cluster_id,
            suffix,
            f"{FORBIDDEN_NAMESPACE}/deployment/regional-allowlist-probe",
        )
        command_ids.extend(
            item["command_id"] for item in executor.get("commands") or []
        )
        workflow_id = str(executor["workflow_id"])
        deadline = time.monotonic() + 180
        final: dict[str, Any] = {"commands": []}
        while time.monotonic() < deadline:
            final = regional.cpu_python(REMOTE_COMMAND_PROBE, workflow_id)
            if final["commands"] and all(
                item["status"] in {"FAILED", "SUCCEEDED", "CANCELLED"}
                for item in final["commands"]
            ):
                break
            time.sleep(2)
        after = workload_snapshot(site, target)
        commands = final["commands"]
        checks = {
            "control_plane_rejects_outside_allowlist": (
                control["outcome"]["status"] == "FAILED"
                and "workflow targets a namespace outside"
                in str(control["outcome"].get("error") or "")
            ),
            "control_plane_created_no_command": not control["commands"],
            "executor_command_created": len(commands) == 1,
            "executor_rejects_local_allowlist": (
                len(commands) == 1
                and commands[0]["status"] == "FAILED"
                and "workload namespace is not allowed"
                in str(commands[0].get("error") or "")
            ),
            "placeholder_workload_unchanged": after == baseline,
        }
        result = {
            "verdict": "PASS" if all(checks.values()) else "FAIL",
            "checks": checks,
            "control_plane": control,
            "executor": final,
        }
    finally:
        if command_ids:
            regional.cpu_python(REMOTE_COMMAND_DELETE_PROBE, *command_ids)
        site.write_registry(original_registry)
        site.rollout_control()
        site.gpu(
            target,
            "delete",
            "namespace",
            FORBIDDEN_NAMESPACE,
            "--ignore-not-found",
            "--wait=true",
            check=False,
            timeout=300,
        )
        residual = site.gpu(
            target,
            "get",
            "namespace",
            FORBIDDEN_NAMESPACE,
            "--ignore-not-found",
            "-o",
            "name",
            check=False,
        ).strip()
        result["cleanup"] = {
            "registry_restored": site.registry() == original_registry,
            "namespace_residual": residual,
        }
        if residual or site.registry() != original_registry:
            result["verdict"] = "FAIL"
    write_json_atomic(case_dir / "iso005-details.json", result)
    result["limitations"] = [
        "The target is a one-replica pause Deployment in a dedicated namespace; "
        "the runner verifies that neither defense changes its UID or Pod set."
    ]
    return result

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from scripts.e2e.regional.acceptance_runner_common import write_json_atomic
from scripts.e2e.regional.identity_acceptance_auth import verdict
from scripts.e2e.regional.identity_acceptance_common import (
    FORBIDDEN_NAMESPACE,
    ClusterTarget,
    IdentityAcceptanceError,
    IdentityCaseFailure,
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

requests = []
class NoNetworkClient(RegionalExecutorClient):
    def _get(self, *args, **kwargs):
        requests.append("GET")
        raise AssertionError("Fleet local guard attempted an HTTP request")
    def _post(self, *args, **kwargs):
        requests.append("POST")
        raise AssertionError("Fleet local guard attempted an HTTP request")

other = sys.argv[1]
registry = RegionalFleetRegistry(
    NoNetworkClient(
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
print(json.dumps({**result, "request_count": len(requests)}, sort_keys=True))
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
        "no_http_requests": result.get("request_count") == 0,
        "secondary_agent_state_unchanged": before == after,
    }
    registered = bool(getattr(secondary, "registered", True))
    limitations = [
        "The probe invokes the deployed executor client but performs no "
        "Fleet state transition."
    ]
    if not registered:
        limitations.append(
            "The secondary cluster id is not registered on this site: the refusal "
            "rests on the executor's own cluster binding alone, and the agent-state "
            "comparison is over an empty list."
        )
    return {
        "verdict": verdict(checks),
        "checks": checks,
        "secondary_registered": registered,
        "secondary_agents_before_after_equal": before == after,
        "results": result,
        "limitations": limitations,
    }


SPARE_HEALTH_PROBE = r"""
import json
import os
import ssl
import sys
import urllib.error
import urllib.request

# One process answers for every requested cluster_id: the two queries used to
# cost two kubectl execs (two python start-ups, two TLS handshakes).
context = ssl.create_default_context(
    cafile=os.environ["GPU_FAULT_CONTROL_PLANE_CA_FILE"]
)
results = {}
for requested in sys.argv[1:]:
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
    try:
        with urllib.request.urlopen(request, context=context, timeout=20) as response:
            results[requested] = {"status": response.status, "body": json.load(response)}
    except urllib.error.HTTPError as exc:
        results[requested] = {
            "status": exc.code,
            "body": json.loads(exc.read() or b"{}"),
        }
print(json.dumps({"results": results}, sort_keys=True))
"""

MISSING_AGENT_REASON = "expected one matching agent, found 0"


def run_iso004(
    site: IdentitySite,
    primary: ClusterTarget,
    secondary: ClusterTarget,
) -> dict[str, Any]:
    pod = site.any_executor_pod(primary)
    results = site.pod_json(
        "gpu",
        primary,
        pod,
        SPARE_HEALTH_PROBE,
        secondary.cluster_id,
        primary.cluster_id,
    )["results"]
    cross = results[secondary.cluster_id]
    local = results[primary.cluster_id]
    detail = str((cross.get("body") or {}).get("detail") or "")
    local_body = local.get("body") or {}
    reasons = [str(item) for item in local_body.get("reasons") or []]
    checks = {
        "cross_cluster_rejected_403": cross["status"] == 403,
        "binding_error_is_precise": (
            "authenticated cluster does not match all payload cluster_id values"
            in detail
        ),
        "local_query_accepted": local["status"] == 200,
        "missing_node_is_not_ready": local_body.get("ready") is False,
        # The not-ready answer must be *because the node does not exist*, not
        # because the spare pool is unhealthy for some unrelated reason.
        "missing_node_reason_is_precise": any(
            MISSING_AGENT_REASON in reason for reason in reasons
        ),
    }
    registered = bool(getattr(secondary, "registered", True))
    limitations = [
        "The consistent-cluster branch queries a deliberately nonexistent "
        "node and performs no spare allocation."
    ]
    if not registered:
        limitations.append(
            "The secondary cluster id is not registered on this site: the 403 "
            "rests on the token-to-payload cluster binding alone."
        )
    return {
        "verdict": verdict(checks),
        "checks": checks,
        "secondary_registered": registered,
        "cross_cluster": cross,
        "local": local,
        "limitations": limitations,
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
from gpu_fault.store import InMemoryStore

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
registration = ApplicationContext.from_environment().store.get_regional_cluster(cluster_id)
store = InMemoryStore()
store.save_regional_cluster(registration)
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
# WorkflowStepOutcome is a dataclass, not a pydantic model.
print(json.dumps({
    "outcome": {
        "status": outcome.status.value,
        "adapter_operation_id": outcome.adapter_operation_id,
        "error": outcome.error,
        "details": outcome.details,
    },
    "commands": commands,
    "workflow_id": workflow.request_id,
    "isolated_store": True,
}, sort_keys=True, default=str))
"""

EXECUTOR_ALLOWLIST_PROBE = r"""
import json
import os
import sys
from types import SimpleNamespace
from gpu_fault.cluster_executor import ClusterActionExecutor
from gpu_fault.regional import RemoteActionCommand

expected_cluster, forbidden_workload, payload = sys.argv[1:]
cluster_id = os.environ["GPU_FAULT_CLUSTER_ID"]
if cluster_id != expected_cluster:
    raise ValueError("executor Pod cluster differs from the selected cluster")
allowed = {
    item.strip()
    for item in os.environ["GPU_FAULT_ALLOWED_WORKLOAD_NAMESPACES"].split(",")
    if item.strip()
}
executor = ClusterActionExecutor(
    SimpleNamespace(cluster_id=cluster_id), [],
    executor_id="iso005-local-guard", allowed_namespaces=allowed,
)
positive = RemoteActionCommand.model_validate(json.loads(payload))
positive = positive.model_copy(update={"lease_token": "iso005-local-probe"})
foreign_step = positive.step.model_copy(update={"workload_ids": [forbidden_workload]})
foreign_workflow = positive.workflow.model_copy(update={"official_steps": [foreign_step]})
negative = positive.model_copy(update={"step": foreign_step, "workflow": foreign_workflow})
def outcome(command):
    result = executor._execute(command)
    return {
        "status": result.status.value,
        "status_source": result.status_source,
        "error": result.error,
    }
print(json.dumps({
    "cluster_id": cluster_id,
    "allowed_namespaces": sorted(allowed),
    "negative": outcome(negative),
    "positive": outcome(positive),
    "adapter_count": len(executor.adapters),
}, sort_keys=True))
"""


def run_iso005(
    site: IdentitySite,
    target: ClusterTarget,
    *,
    case_dir: Path,
) -> dict[str, Any]:
    original_registry = site.registry()
    matching = [
        item
        for item in original_registry
        if item.get("cluster_id") == target.cluster_id
    ]
    if len(matching) != 1:
        raise IdentityAcceptanceError(
            f"registry has {len(matching)} registrations for {target.cluster_id}"
        )
    registration = matching[0]
    original_allowlist = list(registration.get("allowed_namespaces") or [])
    if not original_allowlist:
        raise IdentityAcceptanceError("cluster has no allowed fixture namespace")
    if FORBIDDEN_NAMESPACE in original_allowlist:
        raise IdentityAcceptanceError(
            f"{FORBIDDEN_NAMESPACE} is already in the cluster allowlist"
        )
    regional = site.regional(target)
    suffix = (
        "iso005-" + hashlib.sha256(str(case_dir.resolve()).encode()).hexdigest()[:12]
    )
    forbidden_workload = f"{FORBIDDEN_NAMESPACE}/deployment/{suffix}"
    allowed_namespace = (
        site.namespace
        if site.namespace in original_allowlist
        else original_allowlist[0]
    )
    result: dict[str, Any] = {
        "verdict": "FAIL",
        "checks": {},
        "fixture_scope": "isolated deployed-code guards",
        "allowed_fixture_namespace": allowed_namespace,
        "forbidden_target_namespace": FORBIDDEN_NAMESPACE,
        "cleanup": {"resources_created": False, "registry_modified": False},
        "limitations": [
            "Both guards run in deployed CPU/GPU processes using the current "
            "registration and executor namespace settings. Commands remain in an "
            "isolated in-memory Store; the local executor has no adapters.",
            "No production allowlist is widened, no foreign namespace is created, "
            "and no production command or workload is submitted or removed. This "
            "does not test Kubernetes admission or mutate a placeholder Deployment.",
        ],
    }
    try:
        control = regional.cpu_python(
            ALLOWLIST_WORKFLOW_PROBE,
            target.cluster_id,
            suffix + "-negative",
            forbidden_workload,
            attempts=1,
        )
        checks = result["checks"]
        checks["control_plane_rejects_outside_allowlist"] = (
            control["outcome"]["status"] == "FAILED"
            and control["outcome"].get("error")
            == "workflow targets a namespace outside the cluster registration: "
            + FORBIDDEN_NAMESPACE
        )
        checks["control_plane_created_no_command"] = control.get("commands") == []
        checks["control_store_is_isolated"] = control.get("isolated_store") is True
        if not all(checks.values()):
            raise IdentityAcceptanceError(
                "control-plane namespace guard did not refuse"
            )
        positive = regional.cpu_python(
            ALLOWLIST_WORKFLOW_PROBE,
            target.cluster_id,
            suffix + "-positive",
            f"{allowed_namespace}/deployment/{suffix}",
            attempts=1,
        )
        checks["allowed_namespace_passes_control_guard"] = (
            positive.get("isolated_store") is True
            and positive["outcome"]["status"] == "WAITING"
            and len(positive.get("commands") or []) == 1
        )
        if not checks["allowed_namespace_passes_control_guard"]:
            raise IdentityAcceptanceError("allowed namespace positive control failed")
        executor = regional.executor_python(
            EXECUTOR_ALLOWLIST_PROBE,
            target.cluster_id,
            forbidden_workload,
            json.dumps(positive["commands"][0], separators=(",", ":")),
            attempts=1,
        )
        negative = executor["negative"]
        local_positive = executor["positive"]
        checks.update(
            {
                "executor_cluster_matches": executor.get("cluster_id")
                == target.cluster_id,
                "executor_has_no_adapters": executor.get("adapter_count") == 0,
                "executor_rejects_local_allowlist": (
                    negative.get("status") == "FAILED"
                    and negative.get("status_source") == "executor-rejected"
                    and negative.get("error")
                    == f"ClusterExecutorError: workload namespace is not allowed: {FORBIDDEN_NAMESPACE}"
                ),
                "allowed_namespace_reaches_adapter_selection": (
                    local_positive.get("status") == "FAILED"
                    and local_positive.get("status_source") == "executor-rejected"
                    and local_positive.get("error")
                    == "ClusterExecutorError: remote command requires exactly one local adapter; found 0"
                ),
                "registry_unchanged": site.registry() == original_registry,
            }
        )
        result["control_plane"] = control["outcome"]
        result["positive_control"] = positive["outcome"]
        result["executor"] = executor
        result["verdict"] = verdict(checks)
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        raise IdentityCaseFailure(str(exc), details=result) from exc
    finally:
        write_json_atomic(case_dir / "iso005-details.json", result)
    return result

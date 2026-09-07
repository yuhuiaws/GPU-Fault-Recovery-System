from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any

from scripts.e2e.regional.acceptance_runner_common import write_json_atomic
from scripts.e2e.regional.identity_acceptance_auth import verdict
from scripts.e2e.regional.identity_acceptance_common import (
    FORBIDDEN_NAMESPACE,
    ClusterTarget,
    IdentityAcceptanceError,
    IdentityCaseFailure,
    IdentitySite,
    run_cleanup_steps,
)

TERMINAL_COMMAND_STATUSES = {"FAILED", "SUCCEEDED", "CANCELLED"}
COMMAND_SETTLE_TIMEOUT_SECONDS = 180


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
    }
    return {
        "verdict": verdict(checks),
        "checks": checks,
        # RegionalFleetRegistry raises locally, before any HTTP request, so the
        # secondary's agents cannot have changed because of this probe; the
        # before/after comparison proved nothing about the boundary and is
        # recorded as data, not as a check.
        "not_evaluated": {
            "secondary_agent_state_unchanged": (
                "n/a: the proxy rejects both calls locally before any request "
                "reaches the control plane, so the comparison is vacuous"
            )
        },
        "secondary_agents_before_after_equal": before == after,
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
    return {
        "verdict": verdict(checks),
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

# Settle, cancel, then delete. The probe commands belong to a workflow that
# was never saved, so nothing else will ever retire them; but a LEASED command
# is in the executor's hands and deleting the row under it makes its result
# POST a 404 and leaves a ledger entry for a command that no longer exists.
# Each command is therefore given a bounded wait for its terminal state, a
# public ``cancel_remote_command`` if it is still open and unleased, and only
# then the row removal (there is no public delete; ``_delete`` is the shared
# store primitive, and it is the last step, never the first).
REMOTE_COMMAND_RETIRE_PROBE = r"""
import json
import sys
import time
from gpu_fault.app import ApplicationContext
settle_seconds = float(sys.argv[1])
command_ids = sys.argv[2:]
store = ApplicationContext.from_environment().store
terminal = {"FAILED", "SUCCEEDED", "CANCELLED"}
deadline = time.monotonic() + settle_seconds
final = {}
while True:
    final = {}
    for command_id in command_ids:
        try:
            item = store.get_remote_command(command_id)
        except Exception as exc:  # absent already
            final[command_id] = {"status": None, "error": type(exc).__name__}
            continue
        final[command_id] = {
            "status": item.status.value,
            "status_source": item.status_source,
            "lease_owner_present": item.lease_owner is not None,
        }
    if all(
        value["status"] is None or value["status"] in terminal
        for value in final.values()
    ) or time.monotonic() >= deadline:
        break
    time.sleep(2)
cancelled = {}
for command_id, value in final.items():
    if value["status"] in {"PENDING", "WAITING"}:
        cancelled[command_id] = store.cancel_remote_command(
            command_id, reason="GF-REGIONAL-ISO-005 cleanup"
        )
deleted = []
skipped = []
for command_id, value in final.items():
    if value["status"] == "LEASED":
        # Still in an executor's hands after the settle window: leave the row
        # for the lease to expire rather than delete it under the executor.
        skipped.append(command_id)
        continue
    store._delete("remote_command", command_id)
    deleted.append(command_id)
print(json.dumps({
    "final": final,
    "cancelled": cancelled,
    "deleted": deleted,
    "skipped_leased": skipped,
}, sort_keys=True, default=str))
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
    result: dict[str, Any] = {"verdict": "FAIL", "checks": {}}
    failure: Exception | None = None
    try:
        site.gpu(target, "create", "namespace", FORBIDDEN_NAMESPACE)
        regional.kubectl(
            "gpu",
            "apply",
            "-f",
            "-",
            input_text=json.dumps(manifest),
            namespace=FORBIDDEN_NAMESPACE,
        )
        regional.kubectl(
            "gpu",
            "rollout",
            "status",
            "deployment/regional-allowlist-probe",
            "--timeout=180s",
            namespace=FORBIDDEN_NAMESPACE,
        )
        baseline = workload_snapshot(site, target)
        suffix = f"iso005-control-{int(time.time())}"
        # attempts=1: the probe creates a remote command; a retried exec after
        # a lost receipt would create a second one the cleanup never learns of.
        control = regional.cpu_python(
            ALLOWLIST_WORKFLOW_PROBE,
            target.cluster_id,
            suffix,
            f"{FORBIDDEN_NAMESPACE}/deployment/regional-allowlist-probe",
            attempts=1,
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
            attempts=1,
        )
        command_ids.extend(
            item["command_id"] for item in executor.get("commands") or []
        )
        workflow_id = str(executor["workflow_id"])
        deadline = time.monotonic() + COMMAND_SETTLE_TIMEOUT_SECONDS
        final: dict[str, Any] = {"commands": []}
        while time.monotonic() < deadline:
            final = regional.cpu_python(REMOTE_COMMAND_PROBE, workflow_id)
            if final["commands"] and all(
                item["status"] in TERMINAL_COMMAND_STATUSES
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
            "verdict": verdict(checks),
            "checks": checks,
            "control_plane": control,
            "executor": final,
            # The probe's workflow was never saved, so the executor's result
            # POST completes a command whose workflow does not exist. How the
            # control plane closed it is evidence in its own right.
            "executor_completion": [
                {
                    "command_id": item.get("command_id"),
                    "status": item.get("status"),
                    "status_source": item.get("status_source"),
                    "error": item.get("error"),
                    "workflow_saved": False,
                }
                for item in commands
            ],
        }
    except Exception as exc:
        failure = exc
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:

        def retire_commands() -> dict[str, Any]:
            if not command_ids:
                return {}
            return regional.cpu_python(
                REMOTE_COMMAND_RETIRE_PROBE,
                str(COMMAND_SETTLE_TIMEOUT_SECONDS),
                *command_ids,
                timeout=COMMAND_SETTLE_TIMEOUT_SECONDS + 120,
                attempts=1,
            )

        def delete_namespace() -> str:
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
            return site.gpu(
                target,
                "get",
                "namespace",
                FORBIDDEN_NAMESPACE,
                "--ignore-not-found",
                "-o",
                "name",
                check=False,
            ).strip()

        cleanup, cleanup_errors = run_cleanup_steps(
            [
                ("retire_commands", retire_commands),
                ("restore_registry", lambda: site.write_registry(original_registry)),
                ("rollout_control", site.rollout_control),
                ("delete_namespace", delete_namespace),
                ("read_registry", site.registry),
            ]
        )
        residual = str(cleanup.get("delete_namespace") or "")
        registry_restored = cleanup.get("read_registry") == original_registry
        retired = cleanup.get("retire_commands") or {}
        result["cleanup"] = {
            **cleanup,
            "registry_restored": registry_restored,
            "namespace_residual": residual,
            "commands_left_leased": list(retired.get("skipped_leased") or []),
        }
        result["cleanup_errors"] = cleanup_errors
        if (
            residual
            or not registry_restored
            or cleanup_errors
            or retired.get("skipped_leased")
        ):
            result["verdict"] = "FAIL"
        result["limitations"] = [
            "The target is a one-replica pause Deployment in a dedicated "
            "namespace; the runner verifies that neither defense changes its "
            "UID or Pod set.",
            "The probe commands belong to a workflow that is never saved; the "
            "executor-side completion of such a command is recorded under "
            "executor_completion.",
        ]
        write_json_atomic(case_dir / "iso005-details.json", result)
    if failure is not None:
        raise IdentityCaseFailure(str(failure), details=result) from failure
    return result

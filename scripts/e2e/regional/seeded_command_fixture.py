"""One seeded remote command, one throwaway executor Pod, zero residue.

``run_net002_command_recovery.py`` established the shape: register a synthetic
cluster (``perf-cap-000``) in the regional registry for the length of the case,
seed exactly one remote command for it straight into the store, run a probe
executor Pod on the GPU plane that is the *only* claimant for that cluster, and
purge every row and resource in ``finally``. NET-006 and CMD-017 need the same
scaffolding with a different probe script and a different step, so the parts
that do not depend on the case live here. NET-002 itself is left as it was.

Nothing here targets a real node: the synthetic cluster has no registered Node
Agents, its ``agent_endpoint_allowed_cidrs`` is loopback only, and the seeded
node ids are names no cluster carries. That is the hard stop every case built
on this fixture inherits -- an executor that misbehaves can at worst report a
result for a command nobody else will ever act on.
"""

from __future__ import annotations

import importlib
import json
import shlex
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, cast

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT / "scripts" / "perf") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts" / "perf"))

from scripts.e2e.regional.acceptance_scope import scoped_case_evidence  # noqa: E402

_action_capacity = importlib.import_module("regional_action_capacity_suite")
_registry = importlib.import_module("regional_capacity_registry")
_capacity_suite = importlib.import_module("regional_capacity_suite")

executor_identity = _action_capacity.executor_identity
AWS_REGION: str = _registry.AWS_REGION
CONTROL_NAMESPACE: str = _registry.CONTROL_NAMESPACE
DATAPLANE_CONTEXT: str = _registry.DATAPLANE_CONTEXT
NAMESPACE: str = _registry.NAMESPACE
control: Callable[..., str] = _registry.control
dataplane: Callable[..., str] = _registry.dataplane
load_registry: Callable[[], list[dict[str, Any]]] = _registry.load_registry
register = _registry.register
teardown = _capacity_suite.teardown
upsert_configmap = _capacity_suite.upsert_configmap

SYNTHETIC_CLUSTER_ID = "perf-cap-000"
LIVE_REGISTRY_CONFIRMATION = "ALLOW_PERF_CAPACITY_LIVE_REGISTRY"
TOKEN_SECRET = "gpu-fault-perf-clusters"
CONNECTION_SECRET = "gpu-fault-regional-connection"
DEFAULT_POD_DEADLINE_SECONDS = 900


class SeededCommandError(RuntimeError):
    pass


@dataclass(frozen=True)
class SeededCommandProbe:
    """What one case needs the executor Pod to be."""

    case_id: str
    run_prefix: str
    pod: str
    configmap: str
    owner: str
    script: Path
    environment: dict[str, str] = field(default_factory=dict)
    pod_deadline_seconds: int = DEFAULT_POD_DEADLINE_SECONDS

    def __post_init__(self) -> None:
        if not self.script.is_file():
            raise ValueError("probe script does not exist")
        if not self.run_prefix or "%" in self.run_prefix or "'" in self.run_prefix:
            raise ValueError("run prefix must be a plain identifier fragment")
        if not 60 <= self.pod_deadline_seconds <= 7200:
            raise ValueError("pod deadline is outside 60..7200 seconds")


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(scoped_case_evidence(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    path.chmod(0o600)


def require_environment() -> None:
    if not CONTROL_NAMESPACE or not NAMESPACE or not DATAPLANE_CONTEXT:
        raise SeededCommandError("GPU_FAULT_DATAPLANE_CONTEXT is required")


def run_identity(run_dir: Path, attempt: int, prefix: str) -> str:
    return f"{prefix}-{run_dir.name.rsplit('-', 1)[-1].lower()}-a{attempt}"


def cpu_python(script: str, *arguments: str) -> dict[str, Any]:
    pod = control(
        "get",
        "pod",
        "-l",
        "app=gpu-fault-api-ha",
        "-o",
        "jsonpath={.items[0].metadata.name}",
    ).strip()
    output = control(
        "exec",
        "-i",
        pod,
        "--",
        "python3",
        "-",
        *arguments,
        stdin=script.encode(),
        timeout=120,
    )
    return cast(dict[str, Any], json.loads(output.splitlines()[-1]))


# --------------------------------------------------------------------------- #
# Residual checks
# --------------------------------------------------------------------------- #
_DATABASE_RESIDUALS = r"""
import json
import os
import sys
import psycopg

prefix = sys.argv[1]
like = "%" + prefix + "%"
queries = {
    "objects": (
        "SELECT count(*) FROM gpu_fault_objects "
        "WHERE key LIKE %s OR payload->>'cluster_id' LIKE 'perf-cap-%%'"
    ),
    "links": (
        "SELECT count(*) FROM gpu_fault_links WHERE key LIKE %s OR value LIKE %s"
    ),
    "processor_queue": (
        "SELECT count(*) FROM gpu_fault_processor_queue "
        "WHERE cluster_id LIKE 'perf-cap-%%'"
    ),
    "processor_lanes": (
        "SELECT count(*) FROM gpu_fault_processor_lanes "
        "WHERE ordering_key LIKE 'perf-cap-%%'"
    ),
}
parameters = {
    "objects": (like,),
    "links": (like, like),
    "processor_queue": (),
    "processor_lanes": (),
}
result = {}
with psycopg.connect(os.environ["GPU_FAULT_STORE_URL"]) as connection:
    with connection.cursor() as cursor:
        for name, query in queries.items():
            cursor.execute(query, parameters[name])
            result[name] = int(cursor.fetchone()[0])
result["total"] = sum(result.values())
print(json.dumps(result, sort_keys=True))
"""


def database_residuals(run_prefix: str) -> dict[str, Any]:
    return cpu_python(_DATABASE_RESIDUALS, run_prefix)


def registry_residuals() -> dict[str, Any]:
    synthetic = [
        {
            "cluster_id": str(entry.get("cluster_id") or ""),
            "synthetic_run_id": entry.get("synthetic_run_id"),
        }
        for entry in load_registry()
        if bool(entry.get("synthetic"))
        or str(entry.get("cluster_id") or "").startswith("perf-cap-")
    ]
    return {"count": len(synthetic), "entries": synthetic}


def kubernetes_residuals(probe: SeededCommandProbe) -> dict[str, Any]:
    resources = {}
    for kind, name in (
        ("pod", probe.pod),
        ("configmap", probe.configmap),
        ("secret", TOKEN_SECRET),
    ):
        output = dataplane(
            "get",
            kind,
            name,
            "--ignore-not-found",
            "-o",
            "name",
            check=False,
        ).strip()
        resources[f"{kind}/{name}"] = bool(output)
    return {"count": sum(resources.values()), "resources": resources}


def preflight_residuals(probe: SeededCommandProbe, case_dir: Path) -> dict[str, Any]:
    database = database_residuals(probe.run_prefix)
    write_json(case_dir / "database-preflight.json", database)
    if database.get("total") != 0:
        raise SeededCommandError(f"database preflight found residuals: {database}")
    registry = registry_residuals()
    write_json(case_dir / "registry-verification-preflight.json", registry)
    if registry.get("count") != 0:
        raise SeededCommandError(f"registry preflight found residuals: {registry}")
    kubernetes = kubernetes_residuals(probe)
    write_json(case_dir / "kubernetes-preflight.json", kubernetes)
    if kubernetes.get("count") != 0:
        raise SeededCommandError(f"Kubernetes preflight found residuals: {kubernetes}")
    return {"database": database, "registry": registry, "kubernetes": kubernetes}


# --------------------------------------------------------------------------- #
# Store seed / snapshot / purge (run inside the CPU API Pod)
# --------------------------------------------------------------------------- #
_SEED_COMMAND = r"""
import json
import sys
from datetime import datetime, timedelta, timezone

from gpu_fault.app import ApplicationContext
from gpu_fault.models import (
    FaultIncident,
    IncidentState,
    WorkflowOperation,
    WorkflowRequest,
    WorkflowStatus,
    WorkflowStepSpec,
)
from gpu_fault.regional import RemoteActionCommand

run_id, cluster_id, owner, operation_name, raw_node_ids, raw_lease_seconds = sys.argv[1:]
lease_seconds = int(raw_lease_seconds)
node_ids = [item for item in raw_node_ids.split(",") if item]
operation = WorkflowOperation(operation_name)
incident_id = f"incident-{run_id}"
workflow_id = f"workflow-actionperf-{run_id}"
command_id = f"remote-{run_id}"
step = WorkflowStepSpec(
    operation=operation,
    execution_owner=owner,
    node_ids=node_ids,
)
incident = FaultIncident(
    incident_id=incident_id,
    event_id=f"event-{run_id}",
    event_type="NET_ACCEPTANCE",
    cluster_id=cluster_id,
    node_ids=node_ids,
    policy_version="net-acceptance/v1",
    policy_source="ACCEPTANCE",
    state=IncidentState.ACTION_PENDING,
    workflow_request_id=workflow_id,
    fencing_token=1,
    drill_id=run_id,
)
# Leased to the probe's seed identity: an unleased seeded workflow is claimed
# by the deployed dispatcher, whose executor then drives the step against a
# node no cluster carries and fails the workflow, and the orphan sweep cancels
# the seeded command under the probe (NET-006 attempt 1 and CMD-018 attempt 1,
# 2026-09-09/10). Only the probe may own these commands; the lease outlives the
# synthetic registry entry.
workflow = WorkflowRequest(
    request_id=workflow_id,
    incident_id=incident_id,
    status=WorkflowStatus.PENDING,
    official_action="NO_ACTION",
    fencing_token=1,
    official_steps=[step],
    execution_owner_id=f"{owner}-seed",
    execution_epoch=1,
    execution_lease_expires_at=datetime.now(timezone.utc) + timedelta(seconds=lease_seconds),
)
command = RemoteActionCommand(
    command_id=command_id,
    cluster_id=cluster_id,
    workflow_request_id=workflow_id,
    incident_id=incident_id,
    step_index=0,
    fencing_token=1,
    idempotency_key=f"{workflow_id}/0/{operation.value}",
    step=step,
    workflow=workflow,
    incident=incident,
)
store = ApplicationContext.from_environment().store
store.save_incident(incident)
store.save_workflow(workflow)
store.ensure_remote_command(command)
agents = [item.node_id for item in store.list_agents(cluster_id)]
print(json.dumps({
    "incident_id": incident_id,
    "event_id": incident.event_id,
    "workflow_id": workflow_id,
    "command_id": command_id,
    "cluster_id": cluster_id,
    "operation": operation.value,
    "node_ids": node_ids,
    "registered_agents": agents,
}, sort_keys=True))
"""


# The seeded workflow's execution lease: as long as the synthetic registry
# entry lives, so the deployed dispatcher never claims the workflow while a
# case's probe can still be working on its command.
SEED_LEASE_SECONDS = 30 * 60


def seed_command(
    run_id: str,
    *,
    owner: str,
    operation: str,
    node_ids: list[str],
    cluster_id: str = SYNTHETIC_CLUSTER_ID,
    lease_seconds: int = SEED_LEASE_SECONDS,
) -> dict[str, Any]:
    if not node_ids or any("," in item or not item for item in node_ids):
        raise SeededCommandError("seeded node ids must be non-empty and comma-free")
    if lease_seconds < 60:
        raise SeededCommandError(
            "the seeded workflow lease must be at least 60 seconds"
        )
    return cpu_python(
        _SEED_COMMAND,
        run_id,
        cluster_id,
        owner,
        operation,
        ",".join(node_ids),
        str(lease_seconds),
    )


_COMMAND_SNAPSHOT = r"""
import json
import sys
from gpu_fault.app import ApplicationContext
command = ApplicationContext.from_environment().store.get_remote_command(sys.argv[1])


def stamp(value):
    return value.isoformat() if value is not None else None


print(json.dumps({
    "command_id": command.command_id,
    "status": command.status.value,
    "status_source": command.status_source,
    "last_lease_owner": command.last_lease_owner,
    "lease_owner": command.lease_owner,
    "lease_expires_at": stamp(command.lease_expires_at),
    "cancellation_requested_at": stamp(
        getattr(command, "cancellation_requested_at", None)
    ),
    "updated_at": stamp(getattr(command, "updated_at", None)),
    "error": command.error,
    "result_details": command.result_details,
    "node_ids": list(command.step.node_ids),
    "operation": command.step.operation.value,
}, sort_keys=True, default=str))
"""


def command_snapshot(command_id: str) -> dict[str, Any]:
    return cpu_python(_COMMAND_SNAPSHOT, command_id)


_PURGE_SEED = r"""
import json
import os
import sys
import psycopg

command_id, workflow_id, incident_id, event_id = sys.argv[1:]
deleted = {}
with psycopg.connect(os.environ["GPU_FAULT_STORE_URL"], autocommit=True) as connection:
    cursor = connection.cursor()
    cursor.execute(
        "DELETE FROM gpu_fault_links WHERE kind='incident_by_event' "
        "AND key=%s AND value=%s",
        (event_id, incident_id),
    )
    deleted[f"incident_by_event/{event_id}"] = cursor.rowcount
    for kind, key in (
        ("remote_command", command_id),
        ("workflow", workflow_id),
        ("incident", incident_id),
    ):
        cursor.execute(
            "DELETE FROM gpu_fault_objects WHERE kind=%s AND key=%s", (kind, key)
        )
        deleted[f"{kind}/{key}"] = cursor.rowcount
    cursor.execute(
        "SELECT kind, key FROM gpu_fault_objects WHERE (kind, key) IN "
        "(('remote_command', %s), ('workflow', %s), ('incident', %s)) "
        "ORDER BY kind, key",
        (command_id, workflow_id, incident_id),
    )
    remaining = [{"kind": kind, "key": key} for kind, key in cursor.fetchall()]
    cursor.execute(
        "SELECT count(*) FROM gpu_fault_links WHERE kind='incident_by_event' "
        "AND key=%s AND value=%s",
        (event_id, incident_id),
    )
    remaining_links = int(cursor.fetchone()[0])
print(json.dumps({
    "deleted": deleted,
    "remaining": remaining,
    "remaining_links": remaining_links,
}, sort_keys=True))
"""


def purge_seed(seed: dict[str, Any]) -> dict[str, Any]:
    result = cpu_python(
        _PURGE_SEED,
        str(seed["command_id"]),
        str(seed["workflow_id"]),
        str(seed["incident_id"]),
        str(seed["event_id"]),
    )
    if result.get("remaining") or result.get("remaining_links"):
        raise SeededCommandError(f"seed cleanup left residual state: {result}")
    return result


# --------------------------------------------------------------------------- #
# Probe Pod
# --------------------------------------------------------------------------- #
def pod_manifest(
    probe: SeededCommandProbe,
    image: str,
    identity: dict[str, object],
) -> dict[str, Any]:
    environment = [
        {
            "name": "CONTROL_PLANE_URL",
            "valueFrom": {
                "secretKeyRef": {
                    "name": CONNECTION_SECRET,
                    "key": "control-plane-url",
                }
            },
        },
        {
            "name": "EXECUTOR_ARTIFACT_SHA256",
            "value": str(identity["executor_artifact_sha256"]),
        },
        {
            "name": "EXECUTOR_COMPATIBILITY_DIGEST",
            "value": str(identity["executor_compatibility_digest"]),
        },
        {"name": "EXECUTOR_OWNER", "value": probe.owner},
    ]
    for name in sorted(probe.environment):
        environment.append({"name": name, "value": probe.environment[name]})
    return {
        "apiVersion": "v1",
        "kind": "Pod",
        "metadata": {
            "name": probe.pod,
            "namespace": NAMESPACE,
            "labels": {
                "app": probe.pod,
                "gpu-fault.io/acceptance-case": probe.case_id,
            },
        },
        "spec": {
            "restartPolicy": "Never",
            "activeDeadlineSeconds": probe.pod_deadline_seconds,
            "terminationGracePeriodSeconds": 5,
            "serviceAccountName": "gpu-fault-completion-watcher",
            "tolerations": [{"operator": "Exists"}],
            "containers": [
                {
                    "name": "executor",
                    "image": image,
                    "command": [
                        "/opt/gpu-fault/executor/bin/python",
                        f"/scripts/{probe.script.name}",
                    ],
                    "env": environment,
                    "volumeMounts": [
                        {"name": "script", "mountPath": "/scripts", "readOnly": True},
                        {"name": "tokens", "mountPath": "/tokens", "readOnly": True},
                        {"name": "tls", "mountPath": "/tls", "readOnly": True},
                        {"name": "state", "mountPath": "/state"},
                    ],
                }
            ],
            "volumes": [
                {"name": "script", "configMap": {"name": probe.configmap}},
                {"name": "tokens", "secret": {"secretName": TOKEN_SECRET}},
                {
                    "name": "tls",
                    "secret": {
                        "secretName": CONNECTION_SECRET,
                        "items": [{"key": "ca.crt", "path": "ca.crt"}],
                    },
                },
                {"name": "state", "emptyDir": {}},
            ],
        },
    }


def executor_image() -> str:
    deployment = json.loads(
        dataplane("get", "deployment", "gpu-fault-cluster-executor", "-o", "json")
    )
    return str(deployment["spec"]["template"]["spec"]["containers"][0]["image"])


def register_synthetic_cluster(case_dir: Path, run_id: str) -> None:
    register(
        1,
        case_dir,
        run_id=run_id,
        expires_at=datetime.now(timezone.utc) + timedelta(minutes=30),
        allow_live_registry=True,
        live_registry_confirmation=LIVE_REGISTRY_CONFIRMATION,
    )


def create_probe_pod(probe: SeededCommandProbe, case_dir: Path) -> dict[str, Any]:
    identity = executor_identity(require_dataplane_deployment=True)
    image = executor_image()
    upsert_configmap(
        probe.configmap,
        text={probe.script.name: probe.script.read_text(encoding="utf-8")},
    )
    dataplane("delete", "pod", probe.pod, "--ignore-not-found", check=False)
    dataplane(
        "apply",
        "-f",
        "-",
        stdin=json.dumps(pod_manifest(probe, image, identity)).encode(),
    )
    dataplane("wait", "--for=condition=Ready", f"pod/{probe.pod}", "--timeout=180s")
    wait_file(probe, "/state/ready.json", 60)
    ready = read_state(probe, "/state/ready.json")
    write_json(case_dir / "probe-ready.json", ready)
    return ready


def wait_file(probe: SeededCommandProbe, path: str, timeout_seconds: int) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        result = dataplane(
            "exec",
            probe.pod,
            "--",
            "sh",
            "-c",
            f"if [ -f {shlex.quote(path)} ]; then printf present; fi",
            check=False,
        )
        if result.strip() == "present":
            return
        time.sleep(1)
    raise SeededCommandError(f"timed out waiting for {path}")


def file_present(probe: SeededCommandProbe, path: str) -> bool:
    result = dataplane(
        "exec",
        probe.pod,
        "--",
        "sh",
        "-c",
        f"if [ -f {shlex.quote(path)} ]; then printf present; fi",
        check=False,
    )
    return result.strip() == "present"


def read_state(probe: SeededCommandProbe, path: str) -> dict[str, Any]:
    output = dataplane("exec", probe.pod, "--", "cat", path)
    return cast(dict[str, Any], json.loads(output))


def touch(probe: SeededCommandProbe, path: str) -> None:
    dataplane("exec", probe.pod, "--", "touch", path)


def remove(probe: SeededCommandProbe, path: str) -> None:
    dataplane("exec", probe.pod, "--", "rm", "-f", path)


def pod_logs(probe: SeededCommandProbe) -> str:
    return dataplane("logs", probe.pod, check=False, timeout=120)


def pod_phase(probe: SeededCommandProbe) -> str:
    return dataplane(
        "get", "pod", probe.pod, "-o", "jsonpath={.status.phase}", check=False
    ).strip()


def wait_command(
    command_id: str,
    accept: Callable[[dict[str, Any]], bool],
    timeout_seconds: int,
    *,
    poll_seconds: float = 2.0,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    last: dict[str, Any] = {}
    while time.monotonic() < deadline:
        last = command_snapshot(command_id)
        if accept(last):
            return last
        time.sleep(poll_seconds)
    raise SeededCommandError(f"command did not reach the expected state: {last}")


def wait_executor_state(
    probe: SeededCommandProbe,
    accept: Callable[[dict[str, Any]], bool],
    timeout_seconds: int,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    last: dict[str, Any] = {}
    while time.monotonic() < deadline:
        last = read_state(probe, "/state/executor-state.json")
        if accept(last):
            return last
        time.sleep(1)
    return last


# --------------------------------------------------------------------------- #
# Cleanup
# --------------------------------------------------------------------------- #
def cleanup(
    probe: SeededCommandProbe,
    case_dir: Path,
    run_id: str,
    result: dict[str, Any],
    seed: dict[str, Any],
) -> None:
    """Undo everything the case created; any failure downgrades to FAIL."""

    if seed:
        try:
            seed_cleanup = purge_seed(seed)
            result["seed_cleanup"] = seed_cleanup
            write_json(case_dir / "seed-cleanup.json", seed_cleanup)
        except Exception as exc:  # noqa: BLE001 - recorded, verdict downgraded
            result["cleanup_error"] = f"seed cleanup: {type(exc).__name__}: {exc}"
            result["verdict"] = "FAIL"
    dataplane("delete", "pod", probe.pod, "--ignore-not-found", check=False)
    dataplane("delete", "configmap", probe.configmap, "--ignore-not-found", check=False)
    try:
        teardown(
            purge=True,
            deregister_clusters=True,
            allow_live_registry=True,
            live_registry_confirmation=LIVE_REGISTRY_CONFIRMATION,
            artifacts=case_dir,
            run_id=run_id,
        )
    except Exception as exc:  # noqa: BLE001 - recorded, verdict downgraded
        result["cleanup_error"] = f"registry cleanup: {type(exc).__name__}: {exc}"
        result["verdict"] = "FAIL"
    try:
        postflight = {
            "database": database_residuals(probe.run_prefix),
            "registry": registry_residuals(),
            "kubernetes": kubernetes_residuals(probe),
        }
        names = {
            "database": "database-postflight.json",
            "registry": "registry-verification-postflight.json",
            "kubernetes": "kubernetes-postflight.json",
        }
        for name, value in postflight.items():
            result[f"{name}_postflight"] = value
            write_json(case_dir / names[name], value)
            key = "total" if name == "database" else "count"
            if value.get(key) != 0:
                raise SeededCommandError(f"{name} postflight found residuals: {value}")
    except Exception as exc:  # noqa: BLE001 - recorded, verdict downgraded
        result["postflight_error"] = f"{type(exc).__name__}: {exc}"
        result["verdict"] = "FAIL"


def residual_free(result: dict[str, Any]) -> bool:
    """Whether a finished case document reports zero residue of every kind."""

    database = result.get("database_postflight") or {}
    registry = result.get("registry_postflight") or {}
    kubernetes = result.get("kubernetes_postflight") or {}
    return (
        database.get("total") == 0
        and registry.get("count") == 0
        and kubernetes.get("count") == 0
        and "cleanup_error" not in result
        and "postflight_error" not in result
    )

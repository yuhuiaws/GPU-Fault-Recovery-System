#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.e2e.regional.aurora_binding import AuroraBinding  # noqa: E402

if __package__:
    from . import ha009_refresh as refresh
    from .ha009_refresh import CaseError, stop_refresh_watchdog
    from .ha_evidence import chain_preflight, require_chain, result_identity
    from .regional_commands import run_fixture_command
    from .ha009_observation import parse_pool_metrics as parse_pool_metrics
    from .ha009_observation import (
        observation_errors as observation_errors,
        pod_observation,
        steady_deployments as steady_deployments,
    )
    from .ha009_verdicts import (
        DEPLOYMENTS,
        deployments_rolled as deployments_rolled,
        deployments_steady,
        enabled_pods,
        enabled_roles as enabled_roles,
        role_status,
        rotation_errors,
    )
    from .ha_cleanup import (
        ProcessSupervisionLost,
        attempt_cleanup,
        record_supervision_loss,
        run_cleanup,
    )
    from .ha_plan_preflight import residual_preflight
    from . import run_ha005_rollout_continuity as BASE
    from .acceptance_runner_common import write_json_atomic
    from .live_driver_guard import (
        add_live_arguments,
        authorize_execution,
        build_plan,
        environment_snapshot,
        install_site_profile,
    )
else:
    import ha009_refresh as refresh
    from ha009_refresh import CaseError, stop_refresh_watchdog
    from ha_evidence import chain_preflight, require_chain, result_identity
    from regional_commands import run_fixture_command
    from ha009_observation import parse_pool_metrics as parse_pool_metrics
    from ha009_observation import (
        observation_errors as observation_errors,
        pod_observation,
        steady_deployments as steady_deployments,
    )
    from ha009_verdicts import (
        DEPLOYMENTS,
        deployments_rolled as deployments_rolled,
        deployments_steady,
        enabled_pods,
        enabled_roles as enabled_roles,
        role_status,
        rotation_errors,
    )
    from ha_cleanup import (
        ProcessSupervisionLost,
        attempt_cleanup,
        record_supervision_loss,
        run_cleanup,
    )
    from ha_plan_preflight import residual_preflight
    import run_ha005_rollout_continuity as BASE
    from acceptance_runner_common import write_json_atomic
    from live_driver_guard import (
        add_live_arguments,
        authorize_execution,
        build_plan,
        environment_snapshot,
        install_site_profile,
    )

CASE_ID = "GF-REGIONAL-HA-009"
CONFIRMATION = "HA009_ROTATE_AURORA_CREDENTIALS"
RDS_CLUSTER_ID = ""
CRONJOB = "gpu-fault-aurora-credential-refresh"
SECRET_NAME = "gpu-fault-aurora"
AWS_REGION = ""
# Every wait the case can spend after the probe exists, in seconds. The
# synthetic registration's TTL, the probe Pod's active deadline and the
# detached refresh watchdog are all derived from these rather than guessed: a
# 45-minute registration under a worst path of ~60 minutes expired mid-case.
#
# Path A (CP-3): a rotation is a Secret write. Running Pods mount the Secret
# and their pool re-reads the DSN on every connect, so instead of a consumer
# rollout the case waits for kubelet to project the new file into every Pod
# (secret_propagation), then waits past max_idle (idle_window), then
# watches /healthz, the pool metrics and the Pod logs (post_idle_observation).
# max_idle shrinks idle pools; it does not prove all application connections recycle.
PHASE_BUDGETS = {
    "managed_rotation": 600,
    "first_refresh_job": 700,
    "secret_propagation": 300,
    "idle_window": 480,
    "post_idle_observation": 120,
    "outbox_convergence": 300,
    "runtime_records": 180,
    "processor_receipts": 180,
    "second_refresh_job": 700,
}
# GPU_FAULT_POSTGRES_POOL_MAX_IDLE_SECONDS as the base manifest ships it; the
# live value is read from the control-worker's -config-postgres ConfigMap.
DEFAULT_POOL_MAX_IDLE_SECONDS = 300
IDLE_WINDOW_MARGIN_SECONDS = 60
AUTH_FAILURE_LOG_MARKER = "password authentication failed"
DSN_FILE = "/etc/gpu-fault/aurora/postgres-url"
BUDGET_MARGIN_SECONDS = 600
# Seeded rows remain live through the last observation and owned cleanup.
SEED_LEASE_SECONDS = sum(PHASE_BUDGETS.values()) + BUDGET_MARGIN_SECONDS
# If the runner dies after rotate-secret and before the refresh Job ran, every
# new Pod fails Aurora auth until someone refreshes the Secret. The watchdog
# resumes its UID-owned Job once the runner's rotation-plus-refresh budget has
# passed unless the runner disarmed it first.
REFRESH_WATCHDOG_SECONDS = (
    PHASE_BUDGETS["managed_rotation"]
    + PHASE_BUDGETS["first_refresh_job"]
    + BUDGET_MARGIN_SECONDS
)
BASELINE_EVENT_SAMPLES = 4
BASELINE_CLAIM_SAMPLES = 8


def total_budget_seconds(
    budgets: dict[str, int] = PHASE_BUDGETS,
    *,
    margin_seconds: int = BUDGET_MARGIN_SECONDS,
) -> int:
    return sum(int(value) for value in budgets.values()) + int(margin_seconds)


def configure(arguments: argparse.Namespace) -> None:
    global AWS_REGION
    global CRONJOB
    global RDS_CLUSTER_ID
    global SECRET_NAME

    RDS_CLUSTER_ID = (
        arguments.rds_cluster_id or os.getenv("GPU_FAULT_AURORA_CLUSTER_ID", "")
    ).strip()
    AWS_REGION = (
        os.getenv("GPU_FAULT_PERF_AWS_REGION")
        or os.getenv("AWS_REGION")
        or os.getenv("AWS_DEFAULT_REGION", "")
    ).strip()
    CRONJOB = arguments.refresh_cronjob.strip()
    SECRET_NAME = arguments.aurora_secret_name.strip()
    if not RDS_CLUSTER_ID or not AWS_REGION or not CRONJOB or not SECRET_NAME:
        raise CaseError(
            "RDS cluster ID, AWS Region, refresh CronJob and Aurora Secret "
            "name are required"
        )


def environment_values() -> dict[str, str]:
    value = environment_snapshot()
    value.update(
        {
            "GPU_FAULT_AURORA_CLUSTER_ID": RDS_CLUSTER_ID,
            "GPU_FAULT_AURORA_REFRESH_CRONJOB": CRONJOB,
            "GPU_FAULT_AURORA_SECRET_NAME": SECRET_NAME,
        }
    )
    return value


def log(message: str) -> None:
    stamp = datetime.now(timezone.utc).strftime("%H:%M:%S")
    print(f"[{stamp}] {message}", flush=True)


def write_text(path: Path, value: str) -> None:
    path.write_text(value)
    path.chmod(0o600)


def aws(service: str, *args: str) -> dict:
    result = run_fixture_command(
        ["aws", service, *args, "--region", AWS_REGION, "--output", "json"],
        timeout=180,
    )
    value = json.loads(result.stdout)
    if not isinstance(value, dict):
        raise CaseError(f"aws {service} response is not an object")
    return value


def aurora_guard() -> AuroraBinding:
    return AuroraBinding(
        BASE.control,
        aws,
        AWS_REGION,
        RDS_CLUSTER_ID,
        str(BASE._registry.CONTROL_NAMESPACE),
        secret_name=SECRET_NAME,
        cronjob_name=CRONJOB,
    )


def master_secret_arn() -> str:
    return str(aurora_guard().read()["identity"]["database"]["master_secret_arn"])


def secret_versions(secret_arn: str) -> dict:
    return refresh.secret_versions(aws, secret_arn)


def kubernetes_secret_digest() -> str:
    encoded = BASE.control(
        "get",
        "secret",
        SECRET_NAME,
        "-o",
        "jsonpath={.data.postgres-url}",
    ).strip()
    if not encoded:
        raise CaseError("Aurora Kubernetes Secret has no postgres-url")
    return hashlib.sha256(encoded.encode()).hexdigest()


def kubernetes_secret_dsn_digest() -> str:
    """SHA-256 of the decoded postgres-url, comparable with the mounted file."""

    import base64

    encoded = BASE.control(
        "get",
        "secret",
        SECRET_NAME,
        "-o",
        "jsonpath={.data.postgres-url}",
    ).strip()
    if not encoded:
        raise CaseError("Aurora Kubernetes Secret has no postgres-url")
    return hashlib.sha256(base64.b64decode(encoded)).hexdigest()


_POD_PYTHON = BASE.component_python("cpu")


def pod_dsn_file_digest(pod: str) -> str:
    """SHA-256 of the projected postgres-url inside ``pod`` (never its text)."""

    return BASE.control(
        "exec",
        pod,
        "--",
        _POD_PYTHON,
        "-c",
        "import hashlib, pathlib; "
        f"print(hashlib.sha256(pathlib.Path({DSN_FILE!r}).read_bytes()).hexdigest())",
        timeout=60,
    ).strip()


def probe_pod(pod: str, port: int) -> dict:
    return pod_observation(BASE.control, pod, port, python=_POD_PYTHON)


def pool_max_idle_seconds() -> int:
    raw = BASE.control(
        "get",
        "configmap",
        "gpu-fault-control-worker-config-postgres",
        "-o",
        "jsonpath={.data.GPU_FAULT_POSTGRES_POOL_MAX_IDLE_SECONDS}",
    ).strip()
    try:
        value = float(raw) if raw else float(DEFAULT_POOL_MAX_IDLE_SECONDS)
    except ValueError:
        raise CaseError("pool max_idle is not a number") from None
    if not math.isfinite(value) or value <= 0:
        raise CaseError("pool max_idle must be finite and positive")
    return math.ceil(value)


def idle_wait_seconds(max_idle: int, *, budget: int) -> int:
    """Observe past max_idle without silently clipping the required interval."""

    required = int(max_idle) + IDLE_WINDOW_MARGIN_SECONDS
    if max_idle <= 0 or required > budget:
        raise CaseError(
            "pool max_idle cannot be observed inside the idle-window budget"
        )
    return required


def wait_secret_propagated(pods: list[str], digest: str) -> dict:
    """Every Pod's projected postgres-url matches the Secret."""

    if not pods or not digest:
        raise CaseError("Secret propagation requires Pod identities and a digest")
    deadline = time.monotonic() + PHASE_BUDGETS["secret_propagation"]
    seen: dict[str, str] = {}
    while time.monotonic() < deadline:
        seen = {pod: pod_dsn_file_digest(pod) for pod in pods}
        if all(value == digest for value in seen.values()):
            return {"digest": digest, "pods": seen}
        time.sleep(5)
    return {"digest": digest, "pods": seen}


def auth_failures_in_logs(pod: str, since: datetime) -> int:
    text = BASE.control(
        "logs",
        pod,
        "--all-containers",
        f"--since-time={since.isoformat(timespec='seconds')}",
        timeout=120,
    )
    return sum(1 for line in text.splitlines() if AUTH_FAILURE_LOG_MARKER in line)


def observe_after_idle(
    pods: list[str], *, ports: dict[str, int], samples: int = 6, interval: int = 10
) -> dict:
    """Sample Pod readiness and fresh exec-child SQL connections after max_idle."""

    started = datetime.now(timezone.utc)
    collected: dict[str, list[dict]] = {pod: [] for pod in pods}
    deadline = time.monotonic() + PHASE_BUDGETS["post_idle_observation"]
    for index in range(samples):
        for pod in pods:
            collected[pod].append(probe_pod(pod, ports[pod]))
        if index + 1 < samples and time.monotonic() < deadline:
            time.sleep(interval)
    return {
        "started_at": started.isoformat(timespec="seconds"),
        "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "samples": collected,
        "auth_failures_in_logs": {
            pod: auth_failures_in_logs(pod, started) for pod in pods
        },
    }


def deployment_snapshot() -> dict:
    deployments = json.loads(
        BASE.control("get", "deployment", *DEPLOYMENTS, "-o", "json")
    )
    result = {}
    for item in deployments.get("items", []):
        name = item["metadata"]["name"]
        ports = [
            port["containerPort"]
            for container in item["spec"]["template"]["spec"]["containers"]
            for port in container.get("ports", [])
            if port.get("name") == "http"
        ]
        if len(ports) != 1 or type(ports[0]) is not int or not 0 < ports[0] < 65536:
            raise CaseError(f"{name} has no unique HTTP port")
        pods = json.loads(
            BASE.control(
                "get",
                "pod",
                "-l",
                f"app={name}",
                "-o",
                "json",
            )
        )
        ready_names = {pod["name"] for pod in BASE.ready_pod_records(pods)}
        status = item.get("status", {})
        result[name] = {
            "name": name,
            "uid": item["metadata"]["uid"],
            "generation": item["metadata"].get("generation"),
            "observed_generation": status.get("observedGeneration"),
            "replicas": item["spec"].get("replicas", 0),
            "ready": status.get("readyReplicas", 0),
            "updated": status.get("updatedReplicas", 0),
            "available": status.get("availableReplicas", 0),
            "refresh_annotation": item["spec"]["template"]
            .get("metadata", {})
            .get("annotations", {})
            .get("gpu-fault.aws/aurora-credential-refreshed-at"),
            "pods": sorted(
                {
                    pod["metadata"]["name"]: {
                        "uid": pod["metadata"]["uid"],
                        "port": ports[0],
                        "ready": pod["metadata"]["name"] in ready_names,
                        "restarts": int(
                            (pod.get("status", {}).get("containerStatuses") or [{}])[
                                0
                            ].get("restartCount", 0)
                        ),
                    }
                    for pod in pods.get("items", [])
                }.items()
            ),
        }
    return result


def refresh_job_command(name: str, uid: str) -> list[str]:
    return [
        "kubectl",
        "--kubeconfig",
        str(BASE._registry.CONTROL_KUBECONFIG),
        "-n",
        str(BASE._registry.CONTROL_NAMESPACE),
        "patch",
        "job",
        name,
        "--type=json",
        "-p",
        json.dumps(
            [
                {"op": "test", "path": "/metadata/uid", "value": uid},
                {"op": "test", "path": "/spec/suspend", "value": True},
                {"op": "replace", "path": "/spec/suspend", "value": False},
            ]
        ),
    ]


def refresh_job_manifest(name: str, run_id: str, binding: dict) -> dict:
    return BASE.run_manifest(
        aurora_guard().refresh_job(name, binding),
        run_id,
    )


def start_refresh_watchdog(
    case_dir: Path,
    job_name: str,
    resources: BASE.OwnedProbeResources,
    *,
    run_id: str,
    binding: dict,
    delay_seconds: int = REFRESH_WATCHDOG_SECONDS,
) -> subprocess.Popen[str]:
    return refresh.start_refresh_watchdog(
        case_dir,
        refresh_job_manifest(job_name, run_id, binding),
        resources,
        resume_command=refresh_job_command,
        delay_seconds=delay_seconds,
    )


def run_refresh_job(
    case_dir: Path,
    name: str,
    resources: BASE.OwnedProbeResources,
    *,
    run_id: str,
    binding: dict,
) -> dict:
    return refresh.run_refresh_job(
        case_dir,
        refresh_job_manifest(name, run_id, binding),
        resources,
        control=BASE.control,
        timeout_seconds=PHASE_BUDGETS["first_refresh_job"],
    )


def managed_rotation_complete(
    versions: dict,
    *,
    old_current: str,
    cluster_status: str,
) -> bool:
    current = versions.get("stages", {}).get("AWSCURRENT")
    pending = versions.get("stages", {}).get("AWSPENDING")
    return bool(
        current
        and current != old_current
        and (pending is None or pending == current)
        and cluster_status == "available"
    )


def wait_rotated_secret(old_current: str, secret_arn: str) -> dict:
    deadline = time.monotonic() + PHASE_BUDGETS["managed_rotation"]
    last = {}
    while time.monotonic() < deadline:
        last = secret_versions(secret_arn)
        cluster = aws(
            "rds",
            "describe-db-clusters",
            "--db-cluster-identifier",
            RDS_CLUSTER_ID,
        )["DBClusters"][0]
        if managed_rotation_complete(
            last,
            old_current=old_current,
            cluster_status=str(cluster.get("Status") or ""),
        ):
            return last
        time.sleep(5)
    raise CaseError(f"managed secret rotation did not complete: {last}")


def seed_runtime_records(run_id: str) -> dict:
    script = r"""
import json
import sys
from datetime import datetime, timedelta, timezone
from gpu_fault.app import ApplicationContext
from gpu_fault.models import (
    AdvisoryNotification,
    FaultIncident,
    IncidentState,
    WorkflowOperation,
    WorkflowRequest,
    WorkflowStatus,
    WorkflowStepSpec,
)
from gpu_fault.regional import RemoteActionCommand

run_id, cluster_id, raw_lease_seconds = sys.argv[1:]
lease_seconds = int(raw_lease_seconds)
incident_id = f"incident-{run_id}"
event_id = f"event-{run_id}"
workflow_id = f"workflow-actionperf-{run_id}"
command_id = f"remote-{run_id}"
notification_id = f"notification-{run_id}"
dedup_key = f"{run_id}/notification"
step = WorkflowStepSpec(
    operation=WorkflowOperation.FREEZE_EVIDENCE,
    execution_owner="gpu-fault-ha005-noop",
)
incident = FaultIncident(
    incident_id=incident_id,
    event_id=event_id,
    event_type="HA_ACCEPTANCE",
    cluster_id=cluster_id,
    node_ids=[],
    policy_version="ha-acceptance/v1",
    policy_source="ACCEPTANCE",
    state=IncidentState.ACTION_PENDING,
    workflow_request_id=workflow_id,
    fencing_token=1,
    drill_id=run_id,
)
# The foreign lease protects the simulated workflow from dispatcher execution;
# PENDING keeps its open command out of the terminal-workflow orphan sweep.
workflow = WorkflowRequest(
    request_id=workflow_id,
    incident_id=incident_id,
    status=WorkflowStatus.PENDING,
    official_action="NO_ACTION",
    fencing_token=1,
    official_steps=[step],
    execution_owner_id="gpu-fault-ha009-seed",
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
    idempotency_key=f"{workflow_id}/0/FREEZE_EVIDENCE",
    step=step,
    workflow=workflow,
    incident=incident,
)
notification = AdvisoryNotification(
    notification_id=notification_id,
    deduplication_key=dedup_key,
    cluster_name=cluster_id,
    incident_id=incident_id,
    subject="HA-009 credential rotation drill",
    body_text="Synthetic notification continuity marker.",
    support_case_draft="Acceptance drill only.",
    drill_id=run_id,
    category="HA_ACCEPTANCE",
    not_before=datetime.now(timezone.utc) + timedelta(seconds=10),
)
store = ApplicationContext.from_environment().store
store.save_incident(incident)
store.save_workflow(workflow)
store.ensure_remote_command(command)
store.save_notification_if_absent(notification)
print(json.dumps({
    "incident_id": incident_id,
    "event_id": event_id,
    "workflow_id": workflow_id,
    "command_id": command_id,
    "notification_id": notification_id,
    "deduplication_key": dedup_key,
}, sort_keys=True))
"""
    return BASE.cpu_python(script, run_id, "perf-cap-000", str(SEED_LEASE_SECONDS))


def runtime_snapshot(seed: dict) -> dict:
    script = (
        BASE.STORE_DSN_SNIPPET
        + r"""
import json
import os
import sys
import psycopg
from gpu_fault.app import ApplicationContext
store = ApplicationContext.from_environment().store
command = store.get_remote_command(sys.argv[1])
notification_id = sys.argv[2]
with psycopg.connect(store_dsn()) as connection:
    cursor = connection.cursor()
    cursor.execute(
        "SELECT kind, payload->>'status' FROM gpu_fault_control_records "
        "WHERE key=%s AND kind IN "
        "('notification','notification_delivery','notification_result')",
        (notification_id,),
    )
    notification = {
        kind: {"count": 1, "status": status}
        for kind, status in cursor.fetchall()
    }
print(json.dumps({
    "command": {
        "status": command.status.value,
        "last_lease_owner": command.last_lease_owner,
        "result_details": command.result_details,
    },
    "notification": notification,
}, sort_keys=True))
"""
    )
    return BASE.cpu_python(
        script,
        str(seed["command_id"]),
        str(seed["notification_id"]),
    )


def wait_runtime_records(seed: dict, timeout_seconds: int | None = None) -> dict:
    deadline = time.monotonic() + (timeout_seconds or PHASE_BUDGETS["runtime_records"])
    last = {}
    while time.monotonic() < deadline:
        last = runtime_snapshot(seed)
        notification = last.get("notification", {})
        if (
            last.get("command", {}).get("status") == "SUCCEEDED"
            and notification.get("notification", {}).get("count") == 1
            and notification.get("notification_delivery", {}).get("count") == 1
            and notification.get("notification_result", {}).get("count") == 1
        ):
            return last
        time.sleep(2)
    raise CaseError(f"runtime records did not converge: {last}")


def cleanup_runtime_records(seed: dict) -> dict:
    script = (
        BASE.STORE_DSN_SNIPPET
        + r"""
import json
import os
import sys
import psycopg
incident_id, event_id, workflow_id, command_id, notification_id, dedup_key = sys.argv[1:]
deleted = {}
with psycopg.connect(store_dsn(), autocommit=True) as connection:
    cursor = connection.cursor()
    cursor.execute(
        "DELETE FROM gpu_fault_links WHERE "
        "(kind='incident_by_event' AND key=%s AND value=%s) OR "
        "(kind='notification_dedup' AND key=%s)",
        (event_id, incident_id, dedup_key),
    )
    deleted["links"] = cursor.rowcount
    for kind, key in [
        ("remote_command", command_id),
        ("workflow", workflow_id),
        ("incident", incident_id),
        ("notification_result", notification_id),
        ("notification_delivery", notification_id),
        ("notification", notification_id),
    ]:
        cursor.execute(
            "SELECT gpu_fault_delete_control_state(%s,%s)",
            (kind, key),
        )
        deleted[f"{kind}/{key}"] = int(cursor.fetchone()[0])
    cursor.execute(
        '''
        SELECT count(*)
        FROM gpu_fault_control_records
        WHERE (kind, key) IN (
            ('remote_command', %s),
            ('workflow', %s),
            ('incident', %s),
            ('notification_result', %s),
            ('notification_delivery', %s),
            ('notification', %s)
        )
        ''',
        (
            command_id,
            workflow_id,
            incident_id,
            notification_id,
            notification_id,
            notification_id,
        ),
    )
    objects = int(cursor.fetchone()[0])
    cursor.execute(
        '''
        SELECT count(*)
        FROM gpu_fault_links
        WHERE (kind='incident_by_event' AND key=%s AND value=%s)
           OR (kind='notification_dedup' AND key=%s)
        ''',
        (event_id, incident_id, dedup_key),
    )
    links = int(cursor.fetchone()[0])
print(json.dumps({
    "deleted": deleted,
    "remaining_objects": objects,
    "remaining_links": links,
}, sort_keys=True))
"""
    )
    result = BASE.cpu_python(
        script,
        str(seed["incident_id"]),
        str(seed["event_id"]),
        str(seed["workflow_id"]),
        str(seed["command_id"]),
        str(seed["notification_id"]),
        str(seed["deduplication_key"]),
    )
    if result["remaining_objects"] or result["remaining_links"]:
        raise CaseError(f"runtime cleanup left residuals: {result}")
    return result


def create_probe(
    image: str,
    identity: dict[str, object],
    run_id: str,
    resources: BASE.OwnedProbeResources,
) -> None:
    resources.create(
        BASE.run_manifest(
            {
                "apiVersion": "v1",
                "kind": "ConfigMap",
                "metadata": {"name": BASE.CONFIGMAP, "namespace": BASE.NAMESPACE},
                "data": {BASE.SCRIPT.name: BASE.SCRIPT.read_text()},
            },
            run_id,
        )
    )
    manifest = BASE.pod_manifest(image, identity, run_id)
    manifest["spec"]["activeDeadlineSeconds"] = total_budget_seconds()
    resources.create(BASE.run_manifest(manifest, run_id))
    BASE.dataplane(
        "wait",
        "--for=condition=Ready",
        f"pod/{BASE.POD}",
        "--timeout=180s",
    )
    BASE.wait_file("/state/ready.json", 60)
    BASE.wait_file("/state/stats.json", 60)


def request_rotation(state: dict) -> dict:
    aurora_guard().read(state["aurora_binding"])
    BASE.require_window(
        state["maintenance_window_end"],
        required_seconds=total_budget_seconds() - BUDGET_MARGIN_SECONDS,
    )
    state["rotation_started"] = True
    return aws(
        "rds",
        "modify-db-cluster",
        "--db-cluster-identifier",
        RDS_CLUSTER_ID,
        "--rotate-master-user-password",
        "--apply-immediately",
    )


def _run_rotation_case(
    case_dir: Path,
    run_id: str,
    attempt: int,
    state: dict,
) -> dict:
    database_preflight = BASE.database_residuals()
    registry_preflight = BASE.registry_residuals()
    kubernetes_preflight = BASE.kubernetes_residuals()
    write_json_atomic(case_dir / "database-preflight.json", database_preflight)
    write_json_atomic(case_dir / "registry-preflight.json", registry_preflight)
    write_json_atomic(case_dir / "kubernetes-preflight.json", kubernetes_preflight)
    if database_preflight["total"] != 0:
        raise CaseError(f"database preflight residuals: {database_preflight}")
    if registry_preflight["count"] != 0:
        raise CaseError(f"registry preflight residuals: {registry_preflight}")
    if kubernetes_preflight["count"] != 0:
        raise CaseError(f"Kubernetes preflight residuals: {kubernetes_preflight}")
    state["aurora_binding"] = aurora_guard().read(state.get("aurora_binding"))
    resources = BASE.OwnedProbeResources(
        case_dir / f"probe-resources-{run_id}.json",
        lambda args, body: BASE.dataplane(
            *args, stdin=body.encode() if body is not None else None
        ),
    )
    state["resources"] = resources
    state["job_resources"] = BASE.OwnedProbeResources(
        case_dir / f"refresh-resources-{run_id}.json",
        lambda args, body: BASE.control(
            *args, stdin=body.encode() if body is not None else None
        ),
    )
    max_idle = pool_max_idle_seconds()
    idle_wait = idle_wait_seconds(max_idle, budget=PHASE_BUDGETS["idle_window"])
    state["cleanup_armed"] = True

    secret_arn = state["aurora_binding"]["identity"]["database"]["master_secret_arn"]
    versions_before = secret_versions(secret_arn)
    current_before = str(versions_before["stages"].get("AWSCURRENT") or "")
    if not current_before:
        raise CaseError("managed secret has no AWSCURRENT version")
    digest_before = kubernetes_secret_digest()
    write_json_atomic(case_dir / "secret-versions-before.json", versions_before)

    artifacts = case_dir / f"capacity-{run_id}"
    artifacts.mkdir(exist_ok=True)
    BASE.register(
        1,
        artifacts,
        run_id=run_id,
        expires_at=datetime.now(timezone.utc)
        + timedelta(seconds=total_budget_seconds()),
        allow_live_registry=True,
        live_registry_confirmation="ALLOW_PERF_CAPACITY_LIVE_REGISTRY",
    )
    identity = BASE.executor_identity(require_dataplane_deployment=True)
    deployment = json.loads(
        BASE.dataplane("get", "deployment", "gpu-fault-cluster-executor", "-o", "json")
    )
    image = deployment["spec"]["template"]["spec"]["containers"][0]["image"]
    create_probe(image, identity, run_id, resources)
    state["probe_created"] = True
    ledger = BASE.ReceiptLedger(
        lambda: BASE.read_probe().get("accepted_request_ids", [])
    )
    state["ledger"] = ledger
    ledger.start()
    probe_baseline = BASE.wait_probe_samples(
        minimum_events=BASELINE_EVENT_SAMPLES,
        minimum_claims=BASELINE_CLAIM_SAMPLES,
    )
    write_json_atomic(case_dir / "probe-baseline.json", probe_baseline)
    # One snapshot, taken right before the rotation, is the baseline every
    # later comparison uses.
    deployments_before = deployment_snapshot()
    baseline_errors = deployments_steady(deployments_before, deployments_before)
    if baseline_errors:
        raise CaseError("CPU baseline is incomplete: " + "; ".join(baseline_errors))
    roles = role_status(deployments_before)
    write_json_atomic(
        case_dir / "baseline.json",
        {
            "kubernetes_secret_digest": digest_before,
            "deployments": deployments_before,
            "roles": roles,
        },
    )
    seed = seed_runtime_records(run_id)
    state["seed"] = seed
    write_json_atomic(case_dir / "runtime-seed.json", seed)

    watchdog_job = f"gpu-fault-{run_id}-watchdog"
    state["jobs"].append(watchdog_job)
    state["watchdog"] = start_refresh_watchdog(
        case_dir,
        watchdog_job,
        state["job_resources"],
        run_id=run_id,
        binding=state["aurora_binding"],
    )
    log("triggering RDS-managed master secret rotation")
    rotation_response = request_rotation(state)
    write_json_atomic(
        case_dir / "rotation-request.json",
        {
            "rds_cluster_id": rotation_response.get("DBCluster", {}).get(
                "DBClusterIdentifier"
            ),
            "operation": "RDS.RotateMasterUserPassword",
        },
    )
    versions_after = wait_rotated_secret(current_before, secret_arn)
    write_json_atomic(case_dir / "secret-versions-after.json", versions_after)

    job_one = f"gpu-fault-{run_id}-1"
    state["jobs"].append(job_one)
    log(f"running credential refresh Job {job_one}")
    BASE.require_window(
        state["maintenance_window_end"],
        required_seconds=PHASE_BUDGETS["first_refresh_job"],
    )
    first_job = run_refresh_job(
        case_dir,
        job_one,
        state["job_resources"],
        run_id=run_id,
        binding=state["aurora_binding"],
    )
    state["refresh_succeeded"] = True
    state["watchdog_result"] = stop_refresh_watchdog(state["watchdog"])
    if state["watchdog_result"].get("stop_error") or state["watchdog_result"].get(
        "fired"
    ):
        raise CaseError("refresh watchdog did not stop")
    state["watchdog"] = None
    digest_after = kubernetes_secret_digest()
    pods = enabled_pods(deployments_before)
    log("waiting for kubelet to project the refreshed Secret into every Pod")
    propagation = wait_secret_propagated(pods, kubernetes_secret_dsn_digest())
    write_json_atomic(case_dir / "secret-propagation.json", propagation)
    log(f"observing after {idle_wait}s (pool max_idle {max_idle}s)")
    time.sleep(idle_wait)
    idle_observation = observe_after_idle(
        pods,
        ports={
            pod: value["port"]
            for name in enabled_roles(deployments_before)
            for pod, value in deployments_before[name]["pods"]
        },
    )
    idle_observation["pool_max_idle_seconds"] = max_idle
    idle_observation["idle_wait_seconds"] = idle_wait
    write_json_atomic(case_dir / "idle-observation.json", idle_observation)
    deployments_after = deployment_snapshot()
    write_json_atomic(
        case_dir / "first-refresh.json",
        {
            "job": first_job,
            "kubernetes_secret_digest": digest_after,
            "deployments": deployments_after,
        },
    )

    attempts_at_recovery = int(BASE.read_probe()["counters"].get("event_attempts", 0))
    deadline = time.monotonic() + PHASE_BUDGETS["outbox_convergence"]
    while time.monotonic() < deadline:
        probe = BASE.read_probe()
        if (
            probe.get("outbox") == {"records": 0, "replayable": 0}
            and int(probe["counters"].get("event_attempts", 0))
            >= attempts_at_recovery + 10
        ):
            break
        time.sleep(2)
    else:
        raise CaseError("probe outbox did not converge after credential refresh")
    tail = finish_rotation_observations(case_dir, run_id, state, seed)
    result = _rotation_result(
        attempt,
        versions_before,
        versions_after,
        current_before,
        digest_before,
        digest_after,
        first_job,
        tail["second_job"],
        deployments_before,
        deployments_after,
        propagation,
        idle_observation,
        tail["final_probe"],
        tail["receipts"],
        tail["runtime"],
        tail["digest_before_noop"],
        tail["digest_after_noop"],
        tail["before_noop"],
        tail["after_noop"],
    )
    result["refresh_watchdog"] = state.get("watchdog_result")
    return result


def finish_rotation_observations(
    case_dir: Path, run_id: str, state: dict, seed: dict
) -> dict:
    runtime = wait_runtime_records(seed)
    write_json_atomic(case_dir / "runtime-final.json", runtime)
    before_noop = deployment_snapshot()
    digest_before_noop = kubernetes_secret_digest()
    job_two = f"gpu-fault-{run_id}-2"
    state["jobs"].append(job_two)
    log(f"running NOOP credential refresh Job {job_two}")
    BASE.require_window(
        state["maintenance_window_end"],
        required_seconds=PHASE_BUDGETS["second_refresh_job"],
    )
    second_job = run_refresh_job(
        case_dir,
        job_two,
        state["job_resources"],
        run_id=run_id,
        binding=state["aurora_binding"],
    )
    after_noop = deployment_snapshot()
    digest_after_noop = kubernetes_secret_digest()
    write_json_atomic(
        case_dir / "second-refresh.json",
        {
            "job": second_job,
            "kubernetes_secret_digest_before": digest_before_noop,
            "kubernetes_secret_digest_after": digest_after_noop,
            "deployments_before": before_noop,
            "deployments_after": after_noop,
        },
    )
    final_probe = BASE.stop_probe()
    write_json_atomic(case_dir / "probe-final.json", final_probe)
    accepted_ids = list(final_probe.get("accepted_request_ids", []))
    ledger = state["ledger"]
    ledger.stop()
    receipts = ledger.wait(
        accepted_ids, timeout_seconds=PHASE_BUDGETS["processor_receipts"]
    )
    receipts["telemetry"] = BASE.wait_telemetry_replay(
        final_probe, timeout_seconds=PHASE_BUDGETS["processor_receipts"]
    )
    write_json_atomic(case_dir / "processor-receipts.json", receipts)
    return {
        "runtime": runtime,
        "second_job": second_job,
        "final_probe": final_probe,
        "receipts": receipts,
        "before_noop": before_noop,
        "after_noop": after_noop,
        "digest_before_noop": digest_before_noop,
        "digest_after_noop": digest_after_noop,
    }


def _rotation_result(
    attempt: int,
    versions_before: dict,
    versions_after: dict,
    current_before: str,
    digest_before: str,
    digest_after: str,
    first_job: dict,
    second_job: dict,
    deployments_before: dict,
    deployments_after: dict,
    propagation: dict,
    idle_observation: dict,
    final_probe: dict,
    receipts: dict,
    runtime: dict,
    digest_before_noop: str,
    digest_after_noop: str,
    before_noop: dict,
    after_noop: dict,
) -> dict:
    errors = rotation_errors(
        versions_after=versions_after,
        current_before=current_before,
        digest_before=digest_before,
        digest_after=digest_after,
        first_job=first_job,
        second_job=second_job,
        deployments_before=deployments_before,
        deployments_after=deployments_after,
        propagation=propagation,
        idle_observation=idle_observation,
        final_probe=final_probe,
        receipts=receipts,
        runtime=runtime,
        digest_before_noop=digest_before_noop,
        digest_after_noop=digest_after_noop,
        before_noop=before_noop,
        after_noop=after_noop,
    )
    exercised = BASE.outbox_exercised(final_probe)
    limitations = list(BASE.KNOWN_LIMITATIONS)
    limitations.append(
        "Fresh SQL authentication is observed in an exec child in each CPU Pod; "
        "max_idle and healthz do not prove recycling of every application process pool."
    )
    if not exercised:
        limitations.append(BASE.OUTBOX_NOT_EXERCISED_LIMITATION)
    return {
        "case_id": CASE_ID,
        "attempt": attempt,
        "verdict": "PASS" if not errors else "FAIL",
        "errors": errors,
        "roles": role_status(deployments_before),
        "phase_budgets_seconds": PHASE_BUDGETS,
        "total_budget_seconds": total_budget_seconds(),
        "refresh_watchdog_seconds": REFRESH_WATCHDOG_SECONDS,
        "secret_versions_before": versions_before,
        "secret_versions_after": versions_after,
        "kubernetes_secret_digest_before": digest_before,
        "kubernetes_secret_digest_after": digest_after,
        "first_refresh": first_job,
        "second_refresh": second_job,
        "deployments_before": deployments_before,
        "deployments_after": deployments_after,
        "secret_propagation": propagation,
        "idle_observation": idle_observation,
        "probe_final": final_probe,
        "processor_receipts": receipts,
        "runtime_records": runtime,
        "outbox_exercised": exercised,
        "validation_limitations": limitations,
    }


def _cleanup_rotation(
    case_dir: Path,
    run_id: str,
    attempt: int,
    state: dict,
    result: dict,
) -> None:
    if state.get("ledger") is not None:
        state["ledger"].stop()
    if state["rotation_started"] and not state["refresh_succeeded"]:
        emergency = f"gpu-fault-{run_id}-emergency"
        state["jobs"].append(emergency)
        try:
            result["emergency_refresh"] = run_refresh_job(
                case_dir,
                emergency,
                state["job_resources"],
                run_id=run_id,
                binding=state["aurora_binding"],
            )
            state["refresh_succeeded"] = True
        except Exception as exc:
            result["emergency_refresh_error"] = f"{type(exc).__name__}: {exc}"
            result["verdict"] = "FAIL"
    if state.get("watchdog") is not None:
        # Only after the refresh succeeded (above or in the case body) may the
        # fallback be disarmed; if the emergency Job failed too, the watchdog
        # stays armed as the last line of defence and its Job is kept.
        if state["refresh_succeeded"] or not state["rotation_started"]:
            result["refresh_watchdog"] = stop_refresh_watchdog(state["watchdog"])
            if result["refresh_watchdog"].get("stop_error") or result[
                "refresh_watchdog"
            ].get("fired"):
                result["verdict"] = "FAIL"
                result["watchdog_cleanup_unverified"] = True
            else:
                state["watchdog"] = None
        else:
            result["refresh_watchdog"] = {
                "armed": True,
                "left_armed": True,
                "reason": "credential refresh never succeeded",
            }
    if state["probe_created"]:
        attempt_cleanup(
            result,
            "probe log",
            lambda: write_text(
                case_dir / "probe.log", BASE.dataplane("logs", BASE.POD, timeout=120)
            ),
        )
        attempt_cleanup(
            result,
            "stop probe",
            lambda: BASE.dataplane("exec", BASE.POD, "--", "touch", "/state/stop"),
        )
    resources = state.get("resources")
    if resources is None:
        result["verdict"] = "FAIL"
        result["cleanup_preserved"] = "probe resources have no ownership receipt"
        return
    pod_stopped = True
    for kind, name in (("Pod", BASE.POD), ("ConfigMap", BASE.CONFIGMAP)):
        deleted = attempt_cleanup(
            result,
            f"delete {kind}",
            lambda kind=kind, name=name: resources.delete(kind, name),
        )
        if kind == "Pod":
            pod_stopped = deleted
    if not pod_stopped:
        result["cleanup_preserved"] = (
            "probe shutdown unverified; retain registry and rows"
        )
        return
    jobs_stopped = state.get("watchdog") is None
    for job in state["jobs"]:
        if state.get("watchdog") is not None and job.endswith("-watchdog"):
            continue
        if not attempt_cleanup(
            result,
            f"delete job {job}",
            lambda job=job: state["job_resources"].delete("Job", job),
        ):
            jobs_stopped = False
    if not jobs_stopped:
        result["verdict"] = "FAIL"
        result["cleanup_preserved"] = (
            "refresh cleanup unverified; retain registry and rows"
        )
        return
    if state["seed"]:
        try:
            cleanup = cleanup_runtime_records(state["seed"])
            result["runtime_cleanup"] = cleanup
            write_json_atomic(case_dir / "runtime-cleanup.json", cleanup)
        except Exception as exc:
            result["runtime_cleanup_error"] = f"{type(exc).__name__}: {exc}"
            result["verdict"] = "FAIL"
            result["cleanup_preserved"] = (
                "runtime cleanup unverified; retain registry and rows"
            )
            return
    try:
        BASE.teardown(
            purge=True,
            deregister_clusters=True,
            allow_live_registry=True,
            live_registry_confirmation="ALLOW_PERF_CAPACITY_LIVE_REGISTRY",
            artifacts=case_dir / f"capacity-{run_id}",
            run_id=run_id,
        )
    except Exception as exc:
        result["registry_cleanup_error"] = f"{type(exc).__name__}: {exc}"
        result["verdict"] = "FAIL"
    try:
        postflight = {
            "database": BASE.database_residuals(),
            "registry": BASE.registry_residuals(),
            "kubernetes": BASE.kubernetes_residuals(),
            "deployments": deployment_snapshot(),
        }
        result["postflight"] = postflight
        write_json_atomic(case_dir / "postflight.json", postflight)
        if postflight["database"]["total"] != 0:
            raise CaseError(f"database residuals: {postflight}")
        if postflight["registry"]["count"] != 0:
            raise CaseError(f"registry residuals: {postflight}")
        if postflight["kubernetes"]["count"] != 0:
            raise CaseError(f"Kubernetes residuals: {postflight}")
        for name in DEPLOYMENTS:
            item = postflight["deployments"][name]
            if item["ready"] != item["replicas"]:
                raise CaseError(f"{name} is not fully Ready")
    except Exception as exc:
        result["postflight_error"] = f"{type(exc).__name__}: {exc}"
        result["verdict"] = "FAIL"


def run_case(
    run_dir: Path,
    attempt: int,
    maintenance_window_end: datetime,
    *,
    chain: dict | None = None,
    binding: dict | None = None,
) -> int:
    BASE.require_window(maintenance_window_end, required_seconds=total_budget_seconds())
    case_dir = run_dir / "cases" / CASE_ID
    case_dir.mkdir(parents=True, exist_ok=True)
    run_id = f"ha009-{run_dir.name.rsplit('-', 1)[-1].lower()}-a{attempt}"
    result: dict = {"case_id": CASE_ID, "attempt": attempt, "verdict": "FAIL"}
    state: dict = {
        "seed": {},
        "ledger": None,
        "probe_created": False,
        "rotation_started": False,
        "refresh_succeeded": False,
        "jobs": [],
        "watchdog": None,
        "cleanup_armed": False,
        "aurora_binding": binding,
        "maintenance_window_end": maintenance_window_end,
    }
    try:
        result = _run_rotation_case(case_dir, run_id, attempt, state)
    except ProcessSupervisionLost:
        record_supervision_loss(result)
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if state["cleanup_armed"]:
            run_cleanup(
                result,
                lambda: _cleanup_rotation(case_dir, run_id, attempt, state, result),
            )
    result.update(result_identity(chain))
    write_json_atomic(case_dir / f"{CASE_ID}.json", result)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["verdict"] == "PASS" else 1


def main() -> int:
    install_site_profile()
    parser = argparse.ArgumentParser(
        description="Run the HA-009 managed Aurora credential rotation case."
    )
    add_live_arguments(parser, confirmation=CONFIRMATION)
    # The regional flags the site profile fills (bind_site_profile only binds
    # flags the parser accepts); without them chain_preflight resolved an empty
    # cluster ID and the plan died on 2026-09-18.
    parser.add_argument("--cpu-kubeconfig", default="")
    parser.add_argument("--gpu-kubeconfig", default="")
    parser.add_argument("--gpu-context", default="")
    parser.add_argument("--cluster-id", default="")
    parser.add_argument("--namespace", default="gpu-fault-system")
    parser.add_argument("--region", default="")
    parser.add_argument(
        "--rds-cluster-id",
        default="",
        help="Aurora DB cluster identifier; or GPU_FAULT_AURORA_CLUSTER_ID",
    )
    parser.add_argument(
        "--refresh-cronjob",
        default="gpu-fault-aurora-credential-refresh",
    )
    parser.add_argument(
        "--aurora-secret-name",
        default="gpu-fault-aurora",
    )
    args = parser.parse_args()
    os.umask(0o077)
    configure(args)
    environment = environment_values()
    if not args.execute:
        chain = chain_preflight(args, CASE_ID)
        preflight = residual_preflight(
            BASE.database_residuals, BASE.registry_residuals, BASE.kubernetes_residuals
        )
        binding = aurora_guard().read()
        plan = build_plan(
            run_dir=args.run_dir,
            case_id=CASE_ID,
            attempt=args.attempt,
            confirmation=CONFIRMATION,
            arguments=args,
            preflight_passed=not preflight["errors"] and not chain["errors"],
            environment=environment,
            details={
                "aurora_binding": binding,
                "preflight": preflight,
                "chain": chain,
                "risk": "live-service-action",
                "mutation": "rotate RDS-managed Aurora master secret",
                "rds_cluster_id": RDS_CLUSTER_ID,
                "refresh_cronjob": CRONJOB,
                "aurora_secret_name": SECRET_NAME,
                "probe": "continuous claim, telemetry, command and notification",
                "expectation": (
                    "no Deployment rolls: running Pods reload the mounted Secret; "
                    "after the pool's max_idle every Pod still serves /healthz with "
                    "no password authentication failure in its log"
                ),
                "phase_budgets_seconds": PHASE_BUDGETS,
                "total_budget_seconds": total_budget_seconds(),
                "rollback": [
                    "candidate DSN is verified before Kubernetes Secret patch",
                    "emergency refresh Job rolls forward to AWSCURRENT",
                    "detached refresh watchdog creates the Job after "
                    f"{REFRESH_WATCHDOG_SECONDS}s if the runner dies",
                    "synthetic registry/database/Kubernetes teardown",
                ],
            },
        )
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0 if plan["preflight_passed"] is True else 1
    deadline = authorize_execution(
        args,
        case_id=CASE_ID,
        confirmation=CONFIRMATION,
        environment=environment,
    )
    chain = chain_preflight(args, CASE_ID)
    plan = json.loads((args.run_dir / "cases" / CASE_ID / "plan.json").read_text())
    require_chain(plan["details"].get("chain", {}), chain)
    return run_case(
        args.run_dir,
        args.attempt,
        deadline,
        chain=chain,
        binding=plan["details"]["aurora_binding"],
    )


if __name__ == "__main__":
    raise SystemExit(main())

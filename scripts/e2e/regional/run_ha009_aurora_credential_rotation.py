#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import signal
import subprocess
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

if __package__:
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
DEPLOYMENTS = (
    "gpu-fault-api-ha",
    "gpu-fault-control-worker",
    "gpu-fault-telemetry-spool-worker",
)
# Every wait the case can spend after the probe exists, in seconds. The
# synthetic registration's TTL, the probe Pod's active deadline and the
# detached refresh watchdog are all derived from these rather than guessed: a
# 45-minute registration under a worst path of ~60 minutes expired mid-case.
PHASE_BUDGETS = {
    "managed_rotation": 600,
    "first_refresh_job": 700,
    "consumer_rollout": 900,
    "outbox_convergence": 300,
    "runtime_records": 180,
    "processor_receipts": 180,
    "second_refresh_job": 700,
}
BUDGET_MARGIN_SECONDS = 600
# If the runner dies after rotate-secret and before the refresh Job ran, every
# new Pod fails Aurora auth until someone refreshes the Secret. The watchdog
# creates the Job itself once the runner's own rotation-plus-refresh budget has
# passed unless the runner disarmed it first.
REFRESH_WATCHDOG_SECONDS = (
    PHASE_BUDGETS["managed_rotation"]
    + PHASE_BUDGETS["first_refresh_job"]
    + BUDGET_MARGIN_SECONDS
)
BASELINE_EVENT_SAMPLES = 4
BASELINE_CLAIM_SAMPLES = 8


class CaseError(RuntimeError):
    pass


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
    result = subprocess.run(
        ["aws", service, *args, "--region", AWS_REGION, "--output", "json"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=180,
    )
    if result.returncode != 0:
        raise CaseError(f"aws {service} command failed: {result.stderr.strip()}")
    value = json.loads(result.stdout)
    if not isinstance(value, dict):
        raise CaseError(f"aws {service} response is not an object")
    return value


def master_secret_arn() -> str:
    value = aws(
        "rds",
        "describe-db-clusters",
        "--db-cluster-identifier",
        RDS_CLUSTER_ID,
    )
    clusters = value.get("DBClusters", [])
    if len(clusters) != 1:
        raise CaseError("Aurora cluster lookup did not return one item")
    secret = clusters[0].get("MasterUserSecret") or {}
    arn = str(secret.get("SecretArn") or "")
    if not arn:
        raise CaseError("Aurora cluster has no managed master secret")
    return arn


def secret_versions(secret_arn: str) -> dict:
    value = aws(
        "secretsmanager",
        "list-secret-version-ids",
        "--secret-id",
        secret_arn,
        "--include-deprecated",
    )
    versions = [
        {
            "version_id": str(item.get("VersionId") or ""),
            "stages": sorted(str(stage) for stage in item.get("VersionStages", [])),
            "created_at": str(item.get("CreatedDate") or ""),
        }
        for item in value.get("Versions", [])
    ]
    stages = {
        stage: item["version_id"] for item in versions for stage in item["stages"]
    }
    return {"versions": versions, "stages": stages}


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


def deployment_snapshot() -> dict:
    deployments = json.loads(
        BASE.control("get", "deployment", *DEPLOYMENTS, "-o", "json")
    )
    result = {}
    for item in deployments.get("items", []):
        name = item["metadata"]["name"]
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
        status = item.get("status", {})
        result[name] = {
            "name": name,
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
                        "ready": bool(
                            pod.get("status", {}).get("containerStatuses")
                            and pod["status"]["containerStatuses"][0].get("ready")
                        ),
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


def role_status(deployments: dict) -> dict[str, str]:
    """``ROLLED`` for an enabled role, ``SKIPPED_NOT_ENABLED`` for replicas=0.

    The catalog wants an unconfigured role recorded as skipped, not silently
    passed or failed; the spool-worker ships with replicas=0 on sites without
    telemetry spooling.
    """

    return {
        name: (
            "SKIPPED_NOT_ENABLED"
            if int(deployments[name].get("replicas") or 0) == 0
            else "ROLLED"
        )
        for name in DEPLOYMENTS
    }


def enabled_roles(deployments: dict) -> list[str]:
    return [
        name for name, status in role_status(deployments).items() if status == "ROLLED"
    ]


def refresh_job_command(name: str) -> list[str]:
    return [
        "kubectl",
        "--kubeconfig",
        str(BASE._registry.CONTROL_KUBECONFIG),
        "-n",
        str(BASE._registry.CONTROL_NAMESPACE),
        "create",
        "job",
        name,
        f"--from=cronjob/{CRONJOB}",
    ]


def start_refresh_watchdog(
    case_dir: Path,
    job_name: str,
    *,
    delay_seconds: int = REFRESH_WATCHDOG_SECONDS,
) -> subprocess.Popen[str]:
    """Arm a detached fallback that creates the refresh Job after ``delay_seconds``.

    ``start_new_session`` puts it in its own process group so the runner's
    death (SIGKILL, lost SSH) does not take it along; ``stop_refresh_watchdog``
    disarms it by killing that group once the runner has run the Job itself.
    """

    log_path = case_dir / "refresh-watchdog.log"
    handle = log_path.open("w", encoding="utf-8")
    os.chmod(log_path, 0o600)
    script = f"sleep {int(delay_seconds)}; exec " + shlex.join(
        refresh_job_command(job_name)
    )
    process = subprocess.Popen(
        ["/bin/bash", "-c", script],
        stdout=handle,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    handle.close()
    write_json_atomic(
        case_dir / "refresh-watchdog.json",
        {"pid": process.pid, "job": job_name, "delay_seconds": int(delay_seconds)},
    )
    return process


def stop_refresh_watchdog(process: subprocess.Popen[str] | None) -> dict[str, Any]:
    """Kill the watchdog's process group; never raise from cleanup."""

    if process is None:
        return {"armed": False}
    if process.poll() is not None:
        return {"armed": True, "fired": True, "returncode": process.returncode}
    try:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=10)
    except Exception as exc:  # pragma: no cover - platform dependent
        return {
            "armed": True,
            "fired": False,
            "stop_error": f"{type(exc).__name__}: {exc}",
        }
    return {"armed": True, "fired": False, "disarmed": True}


def run_refresh_job(case_dir: Path, name: str) -> dict:
    BASE.control("delete", "job", name, "--ignore-not-found", check=False)
    BASE.control("create", "job", name, f"--from=cronjob/{CRONJOB}")
    result = BASE.control(
        "wait",
        "--for=condition=complete",
        f"job/{name}",
        f"--timeout={PHASE_BUDGETS['first_refresh_job'] - 100}s",
        check=False,
        timeout=PHASE_BUDGETS["first_refresh_job"],
    )
    job = json.loads(BASE.control("get", "job", name, "-o", "json"))
    succeeded = int(job.get("status", {}).get("succeeded", 0)) == 1
    logs = BASE.control("logs", f"job/{name}", check=False, timeout=120)
    write_text(case_dir / f"{name}.log", logs)
    if not succeeded:
        raise CaseError(f"Aurora refresh Job failed: {name}: {result}")
    return {
        "name": name,
        "succeeded": succeeded,
        "logs": logs.strip().splitlines()[-10:],
    }


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

run_id, cluster_id = sys.argv[1:]
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
workflow = WorkflowRequest(
    request_id=workflow_id,
    incident_id=incident_id,
    status=WorkflowStatus.BLOCKED,
    official_action="NO_ACTION",
    fencing_token=1,
    official_steps=[step],
    blocked_reasons=["synthetic HA-009 credential-rotation check"],
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
    return BASE.cpu_python(script, run_id, "perf-cap-000")


def runtime_snapshot(seed: dict) -> dict:
    script = r"""
import json
import os
import sys
import psycopg
from gpu_fault.app import ApplicationContext
store = ApplicationContext.from_environment().store
command = store.get_remote_command(sys.argv[1])
notification_id = sys.argv[2]
with psycopg.connect(os.environ["GPU_FAULT_STORE_URL"]) as connection:
    cursor = connection.cursor()
    cursor.execute(
        "SELECT kind, payload->>'status' FROM gpu_fault_objects "
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
    script = r"""
import json
import os
import sys
import psycopg
incident_id, event_id, workflow_id, command_id, notification_id, dedup_key = sys.argv[1:]
deleted = {}
with psycopg.connect(os.environ["GPU_FAULT_STORE_URL"], autocommit=True) as connection:
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
            "DELETE FROM gpu_fault_objects WHERE kind=%s AND key=%s",
            (kind, key),
        )
        deleted[f"{kind}/{key}"] = cursor.rowcount
    cursor.execute(
        '''
        SELECT count(*)
        FROM gpu_fault_objects
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


def deployments_rolled(before: dict, current: dict) -> bool:
    """Every enabled role rolled to a new generation with all old Pods replaced.

    Uses HA-005's ``rollout_complete`` -- Ready, updated and available all equal
    to replicas, generation observed, old UIDs gone -- rather than a bare
    ``ready == replicas``, which is true half-way through a rollout while the
    old Pods still hold the old credentials. A replicas=0 role is skipped.
    """

    for name in enabled_roles(before):
        item = current[name]
        if int(item["generation"]) <= int(before[name]["generation"]):
            return False
        old_uids = {value["uid"] for _pod, value in before[name]["pods"]}
        if not BASE.rollout_complete(item, old_uids):
            return False
    return True


def wait_deployments(before: dict, timeout_seconds: int | None = None) -> dict:
    deadline = time.monotonic() + (timeout_seconds or PHASE_BUDGETS["consumer_rollout"])
    last = {}
    while time.monotonic() < deadline:
        last = deployment_snapshot()
        if deployments_rolled(before, last):
            return last
        time.sleep(3)
    raise CaseError(f"database consumers did not finish rollout: {last}")


def create_probe(image: str, identity: dict[str, object], run_id: str) -> None:
    BASE.upsert_configmap(
        BASE.CONFIGMAP,
        text={BASE.SCRIPT.name: BASE.SCRIPT.read_text()},
    )
    BASE.dataplane("delete", "pod", BASE.POD, "--ignore-not-found", check=False)
    manifest = BASE.pod_manifest(image, identity, run_id)
    manifest["spec"]["activeDeadlineSeconds"] = total_budget_seconds()
    BASE.dataplane(
        "apply",
        "-f",
        "-",
        stdin=json.dumps(manifest).encode(),
    )
    BASE.dataplane(
        "wait",
        "--for=condition=Ready",
        f"pod/{BASE.POD}",
        "--timeout=180s",
    )
    BASE.wait_file("/state/ready.json", 60)
    BASE.wait_file("/state/stats.json", 60)


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

    secret_arn = master_secret_arn()
    versions_before = secret_versions(secret_arn)
    current_before = str(versions_before["stages"].get("AWSCURRENT") or "")
    if not current_before:
        raise CaseError("managed secret has no AWSCURRENT version")
    digest_before = kubernetes_secret_digest()
    write_json_atomic(case_dir / "secret-versions-before.json", versions_before)

    BASE.register(
        1,
        case_dir,
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
    create_probe(image, identity, run_id)
    state["probe_created"] = True
    probe_baseline = BASE.wait_probe_samples(
        minimum_events=BASELINE_EVENT_SAMPLES,
        minimum_claims=BASELINE_CLAIM_SAMPLES,
    )
    write_json_atomic(case_dir / "probe-baseline.json", probe_baseline)
    # One snapshot, taken right before the rotation, is the baseline every
    # later comparison uses.
    deployments_before = deployment_snapshot()
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

    watchdog_job = f"gpu-fault-ha009-refresh-{attempt}-watchdog"
    state["jobs"].append(watchdog_job)
    state["watchdog"] = start_refresh_watchdog(case_dir, watchdog_job)
    log("triggering RDS-managed master secret rotation")
    rotation_response = aws(
        "secretsmanager", "rotate-secret", "--secret-id", secret_arn
    )
    state["rotation_started"] = True
    write_json_atomic(
        case_dir / "rotation-request.json",
        {
            "arn": rotation_response.get("ARN"),
            "name": rotation_response.get("Name"),
            "version_id": rotation_response.get("VersionId"),
        },
    )
    versions_after = wait_rotated_secret(current_before, secret_arn)
    write_json_atomic(case_dir / "secret-versions-after.json", versions_after)

    job_one = f"gpu-fault-ha009-refresh-{attempt}-1"
    state["jobs"].append(job_one)
    log(f"running credential refresh Job {job_one}")
    first_job = run_refresh_job(case_dir, job_one)
    state["refresh_succeeded"] = True
    state["watchdog_result"] = stop_refresh_watchdog(state["watchdog"])
    state["watchdog"] = None
    deployments_after = wait_deployments(deployments_before)
    digest_after = kubernetes_secret_digest()
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
    final_probe = BASE.read_probe()
    write_json_atomic(case_dir / "probe-final.json", final_probe)
    runtime = wait_runtime_records(seed)
    write_json_atomic(case_dir / "runtime-final.json", runtime)
    accepted_ids = sorted(set(final_probe.get("accepted_request_ids", [])))
    receipts = BASE.wait_receipts(
        accepted_ids, timeout_seconds=PHASE_BUDGETS["processor_receipts"]
    )
    write_json_atomic(case_dir / "processor-receipts.json", receipts)

    before_noop = deployment_snapshot()
    digest_before_noop = kubernetes_secret_digest()
    job_two = f"gpu-fault-ha009-refresh-{attempt}-2"
    state["jobs"].append(job_two)
    log(f"running NOOP credential refresh Job {job_two}")
    second_job = run_refresh_job(case_dir, job_two)
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
    result = _rotation_result(
        attempt,
        versions_before,
        versions_after,
        current_before,
        digest_before,
        digest_after,
        first_job,
        second_job,
        deployments_before,
        deployments_after,
        final_probe,
        receipts,
        runtime,
        digest_before_noop,
        digest_after_noop,
        before_noop,
        after_noop,
    )
    result["refresh_watchdog"] = state.get("watchdog_result")
    return result


def rotation_errors(
    *,
    versions_after: dict,
    current_before: str,
    digest_before: str,
    digest_after: str,
    first_job: dict,
    second_job: dict,
    deployments_before: dict,
    deployments_after: dict,
    final_probe: dict,
    receipts: dict,
    runtime: dict,
    digest_before_noop: str,
    digest_after_noop: str,
    before_noop: dict,
    after_noop: dict,
) -> list[str]:
    errors = []
    if versions_after["stages"].get("AWSCURRENT") == current_before:
        errors.append("AWSCURRENT did not change")
    if versions_after["stages"].get("AWSPREVIOUS") != current_before:
        errors.append("old AWSCURRENT did not become AWSPREVIOUS")
    if digest_after == digest_before:
        errors.append("Kubernetes Aurora Secret digest did not change")
    if "rotated=True restarted=True" not in "\n".join(first_job["logs"]):
        errors.append("first refresh Job did not report rotation and restart")
    roles = role_status(deployments_before)
    for name in DEPLOYMENTS:
        if roles[name] != "ROLLED":
            continue
        if int(deployments_after[name]["generation"]) <= int(
            deployments_before[name]["generation"]
        ):
            errors.append(f"{name} generation did not advance")
        old_uids = {value["uid"] for _pod, value in deployments_before[name]["pods"]}
        if not BASE.rollout_complete(deployments_after[name], old_uids):
            errors.append(
                f"{name} did not complete its rollout (ready/updated/available/uids)"
            )
        if any(value["restarts"] for _pod, value in deployments_after[name]["pods"]):
            errors.append(f"{name} replacement Pod restarted")
    accepted_ids = sorted(set(final_probe.get("accepted_request_ids", [])))
    errors.extend(
        BASE.continuity_errors(final_probe, receipts, accepted_ids=accepted_ids)
    )
    if runtime["command"].get("status") != "SUCCEEDED":
        errors.append("synthetic remote command did not succeed")
    notification = runtime["notification"]
    for kind in ("notification", "notification_delivery", "notification_result"):
        if notification.get(kind, {}).get("count") != 1:
            errors.append(f"{kind} count is not one")
    if notification.get("notification_result", {}).get("status") != "SKIPPED":
        errors.append("drill notification was not safely suppressed")
    if "rotated=False restarted=False" not in "\n".join(second_job["logs"]):
        errors.append("second refresh Job was not a NOOP")
    if digest_after_noop != digest_before_noop:
        errors.append("NOOP refresh changed the Kubernetes Secret")
    for name in DEPLOYMENTS:
        if after_noop[name]["generation"] != before_noop[name]["generation"]:
            errors.append(f"NOOP refresh changed {name} generation")
        if after_noop[name]["pods"] != before_noop[name]["pods"]:
            errors.append(f"NOOP refresh rolled {name} Pods")
    return errors


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
    if state["rotation_started"] and not state["refresh_succeeded"]:
        emergency = f"gpu-fault-ha009-refresh-{attempt}-emergency"
        state["jobs"].append(emergency)
        try:
            result["emergency_refresh"] = run_refresh_job(case_dir, emergency)
            state["refresh_succeeded"] = True
        except Exception as exc:
            result["emergency_refresh_error"] = f"{type(exc).__name__}: {exc}"
            result["verdict"] = "FAIL"
    if state.get("watchdog") is not None:
        # Only after the refresh succeeded (above or in the case body) may the
        # fallback be disarmed; if the emergency Job failed too, the watchdog
        # stays armed as the last line of defence and its Job is kept.
        if state["refresh_succeeded"]:
            result["refresh_watchdog"] = stop_refresh_watchdog(state["watchdog"])
            state["watchdog"] = None
        else:
            result["refresh_watchdog"] = {
                "armed": True,
                "left_armed": True,
                "reason": "credential refresh never succeeded",
            }
    if state["probe_created"]:
        final_log = BASE.dataplane("logs", BASE.POD, check=False, timeout=120)
        write_text(case_dir / "probe.log", final_log)
        BASE.dataplane("exec", BASE.POD, "--", "touch", "/state/stop", check=False)
    BASE.dataplane("delete", "pod", BASE.POD, "--ignore-not-found", check=False)
    BASE.dataplane(
        "delete", "configmap", BASE.CONFIGMAP, "--ignore-not-found", check=False
    )
    if state["seed"]:
        try:
            cleanup = cleanup_runtime_records(state["seed"])
            result["runtime_cleanup"] = cleanup
            write_json_atomic(case_dir / "runtime-cleanup.json", cleanup)
        except Exception as exc:
            result["runtime_cleanup_error"] = f"{type(exc).__name__}: {exc}"
            result["verdict"] = "FAIL"
    try:
        BASE.teardown(
            purge=True,
            deregister_clusters=True,
            allow_live_registry=True,
            live_registry_confirmation="ALLOW_PERF_CAPACITY_LIVE_REGISTRY",
            artifacts=case_dir,
            run_id=run_id,
        )
    except Exception as exc:
        result["registry_cleanup_error"] = f"{type(exc).__name__}: {exc}"
        result["verdict"] = "FAIL"
    for job in state["jobs"]:
        if state.get("watchdog") is not None and job.endswith("-watchdog"):
            continue
        BASE.control("delete", "job", job, "--ignore-not-found", check=False)
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
) -> int:
    if datetime.now(timezone.utc) >= maintenance_window_end:
        raise CaseError("approved maintenance window has ended")
    case_dir = run_dir / "cases" / CASE_ID
    case_dir.mkdir(parents=True, exist_ok=True)
    run_id = f"ha009-{run_dir.name.rsplit('-', 1)[-1].lower()}-a{attempt}"
    result: dict = {"case_id": CASE_ID, "attempt": attempt, "verdict": "FAIL"}
    state: dict = {
        "seed": {},
        "probe_created": False,
        "rotation_started": False,
        "refresh_succeeded": False,
        "jobs": [],
        "watchdog": None,
    }
    try:
        result = _run_rotation_case(case_dir, run_id, attempt, state)
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        _cleanup_rotation(case_dir, run_id, attempt, state, result)
    write_json_atomic(case_dir / f"{CASE_ID}.json", result)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["verdict"] == "PASS" else 1


def main() -> int:
    install_site_profile()
    parser = argparse.ArgumentParser(
        description="Run the HA-009 managed Aurora credential rotation case."
    )
    add_live_arguments(parser, confirmation=CONFIRMATION)
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
        plan = build_plan(
            run_dir=args.run_dir,
            case_id=CASE_ID,
            attempt=args.attempt,
            confirmation=CONFIRMATION,
            environment=environment,
            details={
                "risk": "live-service-action",
                "mutation": "rotate RDS-managed Aurora master secret",
                "rds_cluster_id": RDS_CLUSTER_ID,
                "refresh_cronjob": CRONJOB,
                "aurora_secret_name": SECRET_NAME,
                "probe": "continuous claim, telemetry, command and notification",
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
        return 0
    deadline = authorize_execution(
        args,
        case_id=CASE_ID,
        confirmation=CONFIRMATION,
        environment=environment,
    )
    return run_case(args.run_dir, args.attempt, deadline)


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

if __package__:
    from . import run_ha005_rollout_continuity as BASE
    from .acceptance_scope import scoped_case_evidence
    from .live_driver_guard import (
        add_live_arguments,
        authorize_execution,
        build_plan,
        environment_snapshot,
        install_site_profile,
    )
else:
    import run_ha005_rollout_continuity as BASE
    from acceptance_scope import scoped_case_evidence
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


class CaseError(RuntimeError):
    pass


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


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(scoped_case_evidence(value), indent=2, sort_keys=True) + "\n"
    )
    path.chmod(0o600)


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
        result[name] = {
            "generation": item["metadata"].get("generation"),
            "observed_generation": item.get("status", {}).get("observedGeneration"),
            "replicas": item["spec"].get("replicas", 0),
            "ready": item.get("status", {}).get("readyReplicas", 0),
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


def run_refresh_job(case_dir: Path, name: str) -> dict:
    BASE.control("delete", "job", name, "--ignore-not-found", check=False)
    BASE.control("create", "job", name, f"--from=cronjob/{CRONJOB}")
    result = BASE.control(
        "wait",
        "--for=condition=complete",
        f"job/{name}",
        "--timeout=600s",
        check=False,
        timeout=700,
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
    deadline = time.monotonic() + 600
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


def wait_runtime_records(seed: dict, timeout_seconds: int = 180) -> dict:
    deadline = time.monotonic() + timeout_seconds
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


def wait_deployments(before: dict, timeout_seconds: int = 900) -> dict:
    deadline = time.monotonic() + timeout_seconds
    last = {}
    while time.monotonic() < deadline:
        last = deployment_snapshot()
        complete = True
        for name in DEPLOYMENTS:
            current = last[name]
            previous = before[name]
            if int(current["generation"]) <= int(previous["generation"]):
                complete = False
            if current["observed_generation"] != current["generation"]:
                complete = False
            if int(current["ready"]) != int(current["replicas"]):
                complete = False
            # A rollout is not finished while a superseded Pod is still
            # draining: control-worker keeps a Terminating Pod (and its
            # Ready condition) for up to terminationGracePeriodSeconds, and
            # _rotation_result must see the Pod set with every old uid gone.
            before_uids = {value["uid"] for _pod, value in previous["pods"]}
            if any(value["uid"] in before_uids for _pod, value in current["pods"]):
                complete = False
        if complete:
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
    manifest["spec"]["activeDeadlineSeconds"] = 1800
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
    write_json(case_dir / "database-preflight.json", database_preflight)
    write_json(case_dir / "registry-preflight.json", registry_preflight)
    write_json(case_dir / "kubernetes-preflight.json", kubernetes_preflight)
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
    deployments_before = deployment_snapshot()
    write_json(case_dir / "secret-versions-before.json", versions_before)
    write_json(
        case_dir / "baseline.json",
        {
            "kubernetes_secret_digest": digest_before,
            "deployments": deployments_before,
        },
    )

    BASE.register(
        1,
        case_dir,
        run_id=run_id,
        expires_at=datetime.now(timezone.utc) + timedelta(minutes=45),
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
    time.sleep(20)
    probe_baseline = BASE.read_probe()
    write_json(case_dir / "probe-baseline.json", probe_baseline)
    deployments_before = deployment_snapshot()
    write_json(case_dir / "deployments-before-rotation.json", deployments_before)
    seed = seed_runtime_records(run_id)
    state["seed"] = seed
    write_json(case_dir / "runtime-seed.json", seed)

    log("triggering RDS-managed master secret rotation")
    rotation_response = aws(
        "secretsmanager", "rotate-secret", "--secret-id", secret_arn
    )
    state["rotation_started"] = True
    write_json(
        case_dir / "rotation-request.json",
        {
            "arn": rotation_response.get("ARN"),
            "name": rotation_response.get("Name"),
            "version_id": rotation_response.get("VersionId"),
        },
    )
    versions_after = wait_rotated_secret(current_before, secret_arn)
    write_json(case_dir / "secret-versions-after.json", versions_after)

    job_one = f"gpu-fault-ha009-refresh-{attempt}-1"
    state["jobs"].append(job_one)
    log(f"running credential refresh Job {job_one}")
    first_job = run_refresh_job(case_dir, job_one)
    state["refresh_succeeded"] = True
    deployments_after = wait_deployments(deployments_before)
    digest_after = kubernetes_secret_digest()
    write_json(
        case_dir / "first-refresh.json",
        {
            "job": first_job,
            "kubernetes_secret_digest": digest_after,
            "deployments": deployments_after,
        },
    )

    attempts_at_recovery = int(BASE.read_probe()["counters"].get("event_attempts", 0))
    deadline = time.monotonic() + 300
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
    write_json(case_dir / "probe-final.json", final_probe)
    runtime = wait_runtime_records(seed)
    write_json(case_dir / "runtime-final.json", runtime)
    accepted_ids = sorted(set(final_probe.get("accepted_request_ids", [])))
    receipts = BASE.wait_receipts(accepted_ids)
    write_json(case_dir / "processor-receipts.json", receipts)

    before_noop = deployment_snapshot()
    digest_before_noop = kubernetes_secret_digest()
    job_two = f"gpu-fault-ha009-refresh-{attempt}-2"
    state["jobs"].append(job_two)
    log(f"running NOOP credential refresh Job {job_two}")
    second_job = run_refresh_job(case_dir, job_two)
    after_noop = deployment_snapshot()
    digest_after_noop = kubernetes_secret_digest()
    write_json(
        case_dir / "second-refresh.json",
        {
            "job": second_job,
            "kubernetes_secret_digest_before": digest_before_noop,
            "kubernetes_secret_digest_after": digest_after_noop,
            "deployments_before": before_noop,
            "deployments_after": after_noop,
        },
    )
    return _rotation_result(
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
    errors = []
    if versions_after["stages"].get("AWSCURRENT") == current_before:
        errors.append("AWSCURRENT did not change")
    if versions_after["stages"].get("AWSPREVIOUS") != current_before:
        errors.append("old AWSCURRENT did not become AWSPREVIOUS")
    if digest_after == digest_before:
        errors.append("Kubernetes Aurora Secret digest did not change")
    if "rotated=True restarted=True" not in "\n".join(first_job["logs"]):
        errors.append("first refresh Job did not report rotation and restart")
    for name in DEPLOYMENTS:
        if int(deployments_after[name]["generation"]) <= int(
            deployments_before[name]["generation"]
        ):
            errors.append(f"{name} generation did not advance")
        if deployments_after[name]["ready"] != deployments_after[name]["replicas"]:
            errors.append(f"{name} did not return to declared Ready replicas")
    for name in ("gpu-fault-api-ha", "gpu-fault-control-worker"):
        before_uids = {value["uid"] for _pod, value in deployments_before[name]["pods"]}
        after_uids = {value["uid"] for _pod, value in deployments_after[name]["pods"]}
        if not before_uids.isdisjoint(after_uids):
            errors.append(f"{name} did not replace every old Pod")
        if any(value["restarts"] for _pod, value in deployments_after[name]["pods"]):
            errors.append(f"{name} replacement Pod restarted")
    counters = final_probe.get("counters", {})
    if int(counters.get("event_attempts", 0)) - int(
        counters.get("event_accepted", 0)
    ) != int(counters.get("event_failures", 0)):
        errors.append("event attempt accounting is inconsistent")
    if final_probe.get("outbox") != {"records": 0, "replayable": 0}:
        errors.append("probe outbox is not empty")
    if any(
        key in {"http-401", "http-403", "http-500"} and int(value) > 0
        for key, value in final_probe.get("error_types", {}).items()
    ):
        errors.append("probe observed 401, 403 or 500")
    if receipts.get("missing") or any(
        item["status"] != "COMPLETED" or item["response_status"] != 200
        for item in receipts.get("requests", [])
    ):
        errors.append("processor requests did not complete exactly once")
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
    return {
        "case_id": CASE_ID,
        "attempt": attempt,
        "verdict": "PASS" if not errors else "FAIL",
        "errors": errors,
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
            write_json(case_dir / "runtime-cleanup.json", cleanup)
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
        BASE.control("delete", "job", job, "--ignore-not-found", check=False)
    try:
        postflight = {
            "database": BASE.database_residuals(),
            "registry": BASE.registry_residuals(),
            "kubernetes": BASE.kubernetes_residuals(),
            "deployments": deployment_snapshot(),
        }
        result["postflight"] = postflight
        write_json(case_dir / "postflight.json", postflight)
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
    state = {
        "seed": {},
        "probe_created": False,
        "rotation_started": False,
        "refresh_succeeded": False,
        "jobs": [],
    }
    try:
        result = _run_rotation_case(case_dir, run_id, attempt, state)
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        _cleanup_rotation(case_dir, run_id, attempt, state, result)
    write_json(case_dir / f"{CASE_ID}.json", result)
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
                "rollback": [
                    "candidate DSN is verified before Kubernetes Secret patch",
                    "emergency refresh Job rolls forward to AWSCURRENT",
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

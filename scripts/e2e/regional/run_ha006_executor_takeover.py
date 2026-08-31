#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import importlib
import json
import os
from pathlib import Path
import shlex
import sys
import time

if __package__:
    from .live_driver_guard import (
        add_live_arguments,
        authorize_execution,
        build_plan,
    )
else:
    from live_driver_guard import (
        add_live_arguments,
        authorize_execution,
        build_plan,
    )

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "perf"))

_action_capacity = importlib.import_module("regional_action_capacity_suite")
_registry = importlib.import_module("regional_capacity_registry")
_capacity_suite = importlib.import_module("regional_capacity_suite")
executor_identity = _action_capacity.executor_identity
NAMESPACE = _registry.NAMESPACE
control = _registry.control
dataplane = _registry.dataplane
load_registry = _registry.load_registry
register = _registry.register
teardown = _capacity_suite.teardown
upsert_configmap = _capacity_suite.upsert_configmap

SCRIPT = Path(__file__).with_name("probes") / "ha006_executor.py"
CONFIGMAP = "gpu-fault-ha006-executor"
PODS = ("gpu-fault-ha006-a", "gpu-fault-ha006-b")
CASE_ID = "GF-REGIONAL-HA-006"
OWNER = "gpu-fault-ha006-test"
LEASE_SECONDS = 30
WINNER_SLEEP_SECONDS = 60
CONFIRMATION = "HA006_FORCE_DELETE_TEST_EXECUTOR"


class CaseError(RuntimeError):
    pass


def log(message: str) -> None:
    stamp = datetime.now(timezone.utc).strftime("%H:%M:%S")
    print(f"[{stamp}] {message}", flush=True)


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    path.chmod(0o600)


def cpu_python(script: str, *arguments: str) -> dict:
    pod = control(
        "get",
        "pod",
        "-l",
        "app=gpu-fault-api-ha",
        "--field-selector=status.phase=Running",
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
        timeout=180,
    )
    return json.loads(output.splitlines()[-1])


def database_residuals() -> dict:
    script = r"""
import json
import os
import psycopg
queries = {
    "objects": (
        "SELECT count(*) FROM gpu_fault_objects "
        "WHERE payload->>'cluster_id' LIKE 'perf-cap-%' "
        "OR key LIKE '%ha006-%'"
    ),
    "links": (
        "SELECT count(*) FROM gpu_fault_links "
        "WHERE key LIKE '%ha006-%' OR value LIKE '%ha006-%'"
    ),
    "processor_queue": (
        "SELECT count(*) FROM gpu_fault_processor_queue "
        "WHERE cluster_id LIKE 'perf-cap-%'"
    ),
    "processor_lanes": (
        "SELECT count(*) FROM gpu_fault_processor_lanes "
        "WHERE ordering_key LIKE 'perf-cap-%'"
    ),
    "processor_queue_counts": (
        "SELECT count(*) FROM gpu_fault_processor_queue_counts "
        "WHERE cluster_id LIKE 'perf-cap-%'"
    ),
}
result = {}
with psycopg.connect(os.environ["GPU_FAULT_STORE_URL"]) as connection:
    cursor = connection.cursor()
    for name, query in queries.items():
        cursor.execute(query)
        result[name] = int(cursor.fetchone()[0])
result["total"] = sum(result.values())
print(json.dumps(result, sort_keys=True))
"""
    return cpu_python(script)


def registry_residuals() -> dict:
    entries = [
        {
            "cluster_id": str(item.get("cluster_id") or ""),
            "synthetic_run_id": item.get("synthetic_run_id"),
        }
        for item in load_registry()
        if bool(item.get("synthetic"))
        or str(item.get("cluster_id") or "").startswith("perf-cap-")
    ]
    return {"count": len(entries), "entries": entries}


def kubernetes_residuals() -> dict:
    resources = {}
    for kind, name in (
        ("configmap", CONFIGMAP),
        ("secret", "gpu-fault-perf-clusters"),
        *[("pod", pod) for pod in PODS],
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


def seed_command(run_id: str) -> dict:
    script = r"""
import json
import sys
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

run_id, cluster_id, owner = sys.argv[1:]
incident_id = f"incident-{run_id}"
event_id = f"event-{run_id}"
workflow_id = f"workflow-actionperf-{run_id}"
command_id = f"remote-{run_id}"
step = WorkflowStepSpec(
    operation=WorkflowOperation.RUN_DCGM_DIAGNOSTIC,
    execution_owner=owner,
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
    blocked_reasons=["synthetic HA-006 executor takeover"],
)
command = RemoteActionCommand(
    command_id=command_id,
    cluster_id=cluster_id,
    workflow_request_id=workflow_id,
    incident_id=incident_id,
    step_index=0,
    fencing_token=1,
    idempotency_key=f"{workflow_id}/0/RUN_DCGM_DIAGNOSTIC",
    step=step,
    workflow=workflow,
    incident=incident,
)
store = ApplicationContext.from_environment().store
store.save_incident(incident)
store.save_workflow(workflow)
store.ensure_remote_command(command)
print(json.dumps({
    "incident_id": incident_id,
    "event_id": event_id,
    "workflow_id": workflow_id,
    "command_id": command_id,
    "deduplication_key": f"{run_id}/shared-action-ledger",
}, sort_keys=True))
"""
    return cpu_python(script, run_id, "perf-cap-000", OWNER)


def command_snapshot(command_id: str) -> dict:
    script = r"""
import json
import sys
from gpu_fault.app import ApplicationContext
item = ApplicationContext.from_environment().store.get_remote_command(sys.argv[1])
print(json.dumps({
    "command_id": item.command_id,
    "status": item.status.value,
    "lease_owner": item.lease_owner,
    "last_lease_owner": item.last_lease_owner,
    "lease_expires_at": (
        item.lease_expires_at.isoformat()
        if item.lease_expires_at is not None else None
    ),
    "result_details": item.result_details,
    "error": item.error,
}, sort_keys=True))
"""
    return cpu_python(script, command_id)


def notification_snapshot(deduplication_key: str) -> dict:
    script = r"""
import json
import os
import sys
import psycopg
key = sys.argv[1]
with psycopg.connect(os.environ["GPU_FAULT_STORE_URL"]) as connection:
    cursor = connection.cursor()
    cursor.execute(
        "SELECT value FROM gpu_fault_links "
        "WHERE kind='notification_dedup' AND key=%s",
        (key,),
    )
    row = cursor.fetchone()
    notification_id = row[0] if row else None
    objects = {}
    if notification_id is not None:
        cursor.execute(
            "SELECT kind, payload->>'status' FROM gpu_fault_objects "
            "WHERE key=%s AND kind IN "
            "('notification','notification_delivery','notification_result')",
            (notification_id,),
        )
        for kind, status in cursor.fetchall():
            objects[kind] = {"count": 1, "status": status}
print(json.dumps({
    "notification_id": notification_id,
    "dedup_link_count": int(notification_id is not None),
    "objects": objects,
}, sort_keys=True))
"""
    return cpu_python(script, deduplication_key)


def cleanup_seed(seed: dict, notification_id: str | None) -> dict:
    script = r"""
import json
import os
import sys
import psycopg
incident_id, event_id, workflow_id, command_id, dedup_key, notification_id = sys.argv[1:]
deleted = {}
with psycopg.connect(os.environ["GPU_FAULT_STORE_URL"], autocommit=True) as connection:
    cursor = connection.cursor()
    cursor.execute(
        "DELETE FROM gpu_fault_links "
        "WHERE (kind='incident_by_event' AND key=%s AND value=%s) "
        "OR (kind='notification_dedup' AND key=%s)",
        (event_id, incident_id, dedup_key),
    )
    deleted["links"] = cursor.rowcount
    items = [
        ("remote_command", command_id),
        ("workflow", workflow_id),
        ("incident", incident_id),
    ]
    if notification_id:
        items.extend([
            ("notification_result", notification_id),
            ("notification_delivery", notification_id),
            ("notification", notification_id),
        ])
    for kind, key in items:
        cursor.execute(
            "DELETE FROM gpu_fault_objects WHERE kind=%s AND key=%s",
            (kind, key),
        )
        deleted[f"{kind}/{key}"] = cursor.rowcount
    cursor.execute(
        "SELECT count(*) FROM gpu_fault_objects "
        "WHERE key LIKE '%ha006-%'"
    )
    objects = int(cursor.fetchone()[0])
    cursor.execute(
        "SELECT count(*) FROM gpu_fault_links "
        "WHERE key LIKE '%ha006-%' OR value LIKE '%ha006-%'"
    )
    links = int(cursor.fetchone()[0])
print(json.dumps({
    "deleted": deleted,
    "remaining_objects": objects,
    "remaining_links": links,
}, sort_keys=True))
"""
    result = cpu_python(
        script,
        str(seed["incident_id"]),
        str(seed["event_id"]),
        str(seed["workflow_id"]),
        str(seed["command_id"]),
        str(seed["deduplication_key"]),
        notification_id or "",
    )
    if result["remaining_objects"] or result["remaining_links"]:
        raise CaseError(f"seed cleanup left residuals: {result}")
    return result


def pod_manifest(
    name: str,
    image: str,
    identity: dict[str, object],
    run_id: str,
) -> dict:
    return {
        "apiVersion": "v1",
        "kind": "Pod",
        "metadata": {
            "name": name,
            "namespace": NAMESPACE,
            "labels": {"app": "gpu-fault-ha006-executor"},
        },
        "spec": {
            "restartPolicy": "Never",
            "activeDeadlineSeconds": 600,
            "terminationGracePeriodSeconds": 5,
            "serviceAccountName": "gpu-fault-completion-watcher",
            "tolerations": [{"operator": "Exists"}],
            "containers": [
                {
                    "name": "executor",
                    "image": image,
                    "command": [
                        "/opt/gpu-fault/executor/bin/python",
                        f"/scripts/{SCRIPT.name}",
                    ],
                    "env": [
                        {
                            "name": "CONTROL_PLANE_URL",
                            "valueFrom": {
                                "secretKeyRef": {
                                    "name": "gpu-fault-regional-connection",
                                    "key": "control-plane-url",
                                }
                            },
                        },
                        {"name": "RUN_ID", "value": run_id},
                        {"name": "LEASE_SECONDS", "value": str(LEASE_SECONDS)},
                        {
                            "name": "WINNER_SLEEP_SECONDS",
                            "value": str(WINNER_SLEEP_SECONDS),
                        },
                        {
                            "name": "EXECUTOR_ARTIFACT_SHA256",
                            "value": str(identity["executor_artifact_sha256"]),
                        },
                        {
                            "name": "EXECUTOR_COMPATIBILITY_DIGEST",
                            "value": str(identity["executor_compatibility_digest"]),
                        },
                    ],
                    "volumeMounts": [
                        {"name": "script", "mountPath": "/scripts", "readOnly": True},
                        {"name": "tokens", "mountPath": "/tokens", "readOnly": True},
                        {"name": "tls", "mountPath": "/tls", "readOnly": True},
                        {"name": "state", "mountPath": "/state"},
                    ],
                }
            ],
            "volumes": [
                {"name": "script", "configMap": {"name": CONFIGMAP}},
                {"name": "tokens", "secret": {"secretName": "gpu-fault-perf-clusters"}},
                {
                    "name": "tls",
                    "secret": {
                        "secretName": "gpu-fault-regional-connection",
                        "items": [{"key": "ca.crt", "path": "ca.crt"}],
                    },
                },
                {"name": "state", "emptyDir": {}},
            ],
        },
    }


def wait_file(pod: str, path: str, timeout_seconds: int) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        output = dataplane(
            "exec",
            pod,
            "--",
            "sh",
            "-c",
            f"if [ -f {shlex.quote(path)} ]; then printf present; fi",
            check=False,
        )
        if output.strip() == "present":
            return
        time.sleep(1)
    raise CaseError(f"{pod} did not create {path}")


def read_state(pod: str, path: str) -> dict:
    return json.loads(dataplane("exec", pod, "--", "cat", path))


def production_lease_seconds() -> int:
    deployment = json.loads(
        dataplane("get", "deployment", "gpu-fault-cluster-executor", "-o", "json")
    )
    for item in deployment["spec"]["template"]["spec"]["containers"][0]["env"]:
        if item.get("name") == "GPU_FAULT_CLUSTER_EXECUTOR_LEASE_SECONDS":
            return int(item["value"])
    raise CaseError("production executor lease setting is missing")


def wait_first_owner(seed: dict, timeout_seconds: int = 120) -> tuple[dict, dict]:
    deadline = time.monotonic() + timeout_seconds
    last_command = {}
    last_notification = {}
    while time.monotonic() < deadline:
        last_command = command_snapshot(str(seed["command_id"]))
        last_notification = notification_snapshot(str(seed["deduplication_key"]))
        if (
            last_command.get("status") == "LEASED"
            and last_command.get("lease_owner") in PODS
            and last_notification.get("notification_id")
        ):
            return last_command, last_notification
        time.sleep(0.5)
    raise CaseError(
        f"first owner did not acquire shared ledger: {last_command} {last_notification}"
    )


def wait_terminal(seed: dict, timeout_seconds: int = 120) -> tuple[dict, list[dict]]:
    deadline = time.monotonic() + timeout_seconds
    timeline = []
    while time.monotonic() < deadline:
        value = command_snapshot(str(seed["command_id"]))
        timeline.append(
            {
                "observed_at": datetime.now(timezone.utc).isoformat(),
                **value,
            }
        )
        if value.get("status") == "SUCCEEDED":
            return value, timeline
        time.sleep(0.5)
    raise CaseError(f"command did not reach SUCCEEDED: {timeline[-5:]}")


def run_case(
    run_dir: Path,
    attempt: int,
    maintenance_window_end: datetime,
) -> int:
    if datetime.now(timezone.utc) >= maintenance_window_end:
        raise CaseError("approved maintenance window has ended")
    case_dir = run_dir / "cases" / CASE_ID
    case_dir.mkdir(parents=True, exist_ok=True)
    run_id = f"ha006-{run_dir.name.rsplit('-', 1)[-1].lower()}-a{attempt}"
    result: dict = {"case_id": CASE_ID, "attempt": attempt, "verdict": "FAIL"}
    seed: dict = {}
    notification_id: str | None = None
    created_pods: set[str] = set()
    try:
        database_preflight = database_residuals()
        registry_preflight = registry_residuals()
        kubernetes_preflight = kubernetes_residuals()
        write_json(case_dir / "database-preflight.json", database_preflight)
        write_json(case_dir / "registry-preflight.json", registry_preflight)
        write_json(case_dir / "kubernetes-preflight.json", kubernetes_preflight)
        if database_preflight["total"] != 0:
            raise CaseError(f"database preflight residuals: {database_preflight}")
        if registry_preflight["count"] != 0:
            raise CaseError(f"registry preflight residuals: {registry_preflight}")
        if kubernetes_preflight["count"] != 0:
            raise CaseError(f"Kubernetes preflight residuals: {kubernetes_preflight}")

        register(
            1,
            case_dir,
            run_id=run_id,
            expires_at=datetime.now(timezone.utc) + timedelta(minutes=30),
            allow_live_registry=True,
            live_registry_confirmation="ALLOW_PERF_CAPACITY_LIVE_REGISTRY",
        )
        identity = executor_identity(require_dataplane_deployment=True)
        deployment = json.loads(
            dataplane("get", "deployment", "gpu-fault-cluster-executor", "-o", "json")
        )
        image = deployment["spec"]["template"]["spec"]["containers"][0]["image"]
        upsert_configmap(CONFIGMAP, text={SCRIPT.name: SCRIPT.read_text()})
        for pod in PODS:
            dataplane("delete", "pod", pod, "--ignore-not-found", check=False)
            dataplane(
                "apply",
                "-f",
                "-",
                stdin=json.dumps(pod_manifest(pod, image, identity, run_id)).encode(),
            )
            created_pods.add(pod)
        for pod in PODS:
            dataplane("wait", "--for=condition=Ready", f"pod/{pod}", "--timeout=180s")
            wait_file(pod, "/state/ready.json", 60)
        ready = {pod: read_state(pod, "/state/ready.json") for pod in PODS}
        write_json(case_dir / "executors-ready.json", ready)

        seed = seed_command(run_id)
        write_json(case_dir / "seed.json", seed)
        first, notification = wait_first_owner(seed)
        notification_id = str(notification["notification_id"])
        write_json(case_dir / "first-lease.json", first)
        write_json(case_dir / "shared-ledger-before-kill.json", notification)
        owner = str(first["lease_owner"])
        survivor = next(pod for pod in PODS if pod != owner)
        owner_state = read_state(owner, "/state/executor-state.json")
        write_json(case_dir / "owner-before-kill.json", owner_state)
        kill_requested_at = datetime.now(timezone.utc)
        log(f"force deleting lease owner {owner}; survivor={survivor}")
        dataplane(
            "delete",
            "pod",
            owner,
            "--grace-period=0",
            "--force",
        )
        killed_at = datetime.now(timezone.utc)
        created_pods.discard(owner)
        post_kill = command_snapshot(str(seed["command_id"]))
        post_kill_expiry = datetime.fromisoformat(
            str(post_kill["lease_expires_at"]).replace("Z", "+00:00")
        )
        remaining_lease = max(0.0, (post_kill_expiry - killed_at).total_seconds())
        write_json(case_dir / "post-kill-command.json", post_kill)
        final, timeline = wait_terminal(seed)
        completed_at = datetime.now(timezone.utc)
        takeover_seconds = (completed_at - killed_at).total_seconds()
        write_json(case_dir / "command-timeline.json", timeline)
        write_json(case_dir / "final-command.json", final)
        survivor_state = read_state(survivor, "/state/executor-state.json")
        write_json(case_dir / "survivor-state.json", survivor_state)
        survivor_logs = dataplane("logs", survivor, check=False, timeout=120)
        (case_dir / "survivor.log").write_text(survivor_logs)
        (case_dir / "survivor.log").chmod(0o600)
        notification_final = notification_snapshot(str(seed["deduplication_key"]))
        write_json(case_dir / "shared-ledger-final.json", notification_final)
        errors = []
        if final.get("last_lease_owner") != survivor:
            errors.append("survivor did not become the final lease owner")
        details = final.get("result_details", {})
        if details.get("cached") is not True or details.get("physical_count") != 1:
            errors.append("survivor did not reuse the shared action ledger")
        if details.get("shared_notification_id") != notification_id:
            errors.append("shared ledger identity changed across takeover")
        if takeover_seconds > remaining_lease + 5:
            errors.append("takeover exceeded remaining lease plus poll margin")
        if int(survivor_state.get("claimed_total", 0)) < 1:
            errors.append("survivor did not record a claimed command")
        if int(survivor_state.get("unexpected_failures", 0)) != 0:
            errors.append("survivor recorded an unexpected failure")
        objects = notification_final.get("objects", {})
        if notification_final.get("dedup_link_count") != 1:
            errors.append("shared action ledger dedup link count is not one")
        for kind in ("notification", "notification_delivery", "notification_result"):
            if objects.get(kind, {}).get("count") != 1:
                errors.append(f"shared ledger {kind} count is not one")
        if objects.get("notification_result", {}).get("status") != "SKIPPED":
            errors.append("shared ledger drill notification was not suppressed")
        result = {
            "case_id": CASE_ID,
            "attempt": attempt,
            "verdict": "PASS" if not errors else "FAIL",
            "errors": errors,
            "production_lease_seconds": production_lease_seconds(),
            "fixture_lease_seconds": LEASE_SECONDS,
            "winner_sleep_seconds": WINNER_SLEEP_SECONDS,
            "killed_owner": owner,
            "survivor": survivor,
            "kill_requested_at": kill_requested_at.isoformat(),
            "killed_at": killed_at.isoformat(),
            "completed_at": completed_at.isoformat(),
            "remaining_lease_at_kill_seconds": round(remaining_lease, 3),
            "takeover_seconds": round(takeover_seconds, 3),
            "first_lease": first,
            "post_kill_command": post_kill,
            "final_command": final,
            "shared_ledger": notification_final,
            "owner_state": owner_state,
            "survivor_state": survivor_state,
        }
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        for pod in PODS:
            logs = dataplane("logs", pod, check=False, timeout=120)
            path = case_dir / f"{pod}.log"
            path.write_text(logs)
            path.chmod(0o600)
            dataplane("delete", "pod", pod, "--ignore-not-found", check=False)
        dataplane(
            "delete",
            "configmap",
            CONFIGMAP,
            "--ignore-not-found",
            check=False,
        )
        if seed:
            try:
                cleanup = cleanup_seed(seed, notification_id)
                result["seed_cleanup"] = cleanup
                write_json(case_dir / "seed-cleanup.json", cleanup)
            except Exception as exc:
                result["cleanup_error"] = f"{type(exc).__name__}: {exc}"
                result["verdict"] = "FAIL"
        try:
            teardown(
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
        try:
            postflight = {
                "database": database_residuals(),
                "registry": registry_residuals(),
                "kubernetes": kubernetes_residuals(),
            }
            result["postflight"] = postflight
            write_json(case_dir / "postflight.json", postflight)
            if postflight["database"]["total"] != 0:
                raise CaseError(f"database residuals: {postflight}")
            if postflight["registry"]["count"] != 0:
                raise CaseError(f"registry residuals: {postflight}")
            if postflight["kubernetes"]["count"] != 0:
                raise CaseError(f"Kubernetes residuals: {postflight}")
        except Exception as exc:
            result["postflight_error"] = f"{type(exc).__name__}: {exc}"
            result["verdict"] = "FAIL"
    write_json(case_dir / f"{CASE_ID}.json", result)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["verdict"] == "PASS" else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    add_live_arguments(parser, confirmation=CONFIRMATION)
    args = parser.parse_args()
    os.umask(0o077)
    if not args.execute:
        plan = build_plan(
            run_dir=args.run_dir,
            case_id=CASE_ID,
            attempt=args.attempt,
            confirmation=CONFIRMATION,
            details={
                "risk": "live-non-destructive",
                "synthetic_cluster_id": "perf-cap-000",
                "fixture_lease_seconds": LEASE_SECONDS,
                "production_lease_is_recorded_at_runtime": True,
                "mutation": "force delete one of two temporary executor Pods",
                "physical_action": "simulated shared notification-dedup ledger",
                "rollback": [
                    "temporary Pods have active deadlines",
                    "synthetic registry teardown",
                    "database and Kubernetes zero-residual postflight",
                ],
            },
        )
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0
    deadline = authorize_execution(
        args,
        case_id=CASE_ID,
        confirmation=CONFIRMATION,
    )
    return run_case(args.run_dir, args.attempt, deadline)


if __name__ == "__main__":
    raise SystemExit(main())

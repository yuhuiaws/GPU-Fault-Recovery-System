#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json
import os
import shlex
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

if __package__:
    from .acceptance_scope import scoped_case_evidence
    from .live_driver_guard import (
        add_live_arguments,
        authorize_execution,
        build_plan,
        install_site_profile,
    )
else:
    from acceptance_scope import scoped_case_evidence
    from live_driver_guard import (
        add_live_arguments,
        authorize_execution,
        build_plan,
        install_site_profile,
    )

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "perf"))

_action_capacity = importlib.import_module("regional_action_capacity_suite")
_registry = importlib.import_module("regional_capacity_registry")
_capacity_suite = importlib.import_module("regional_capacity_suite")
executor_identity = _action_capacity.executor_identity
AWS_REGION = _registry.AWS_REGION
CONTROL_NAMESPACE = _registry.CONTROL_NAMESPACE
DATAPLANE_CONTEXT = _registry.DATAPLANE_CONTEXT
NAMESPACE = _registry.NAMESPACE
control = _registry.control
dataplane = _registry.dataplane
load_registry = _registry.load_registry
register = _registry.register
teardown = _capacity_suite.teardown
upsert_configmap = _capacity_suite.upsert_configmap

SCRIPT = Path(__file__).with_name("probes") / "net003_executor.py"
CONFIGMAP = "gpu-fault-net003-script"
POD = "gpu-fault-net003-executor"
OWNER = "gpu-fault-net-test"
CONFIRMATION = "NET003_RESULT_CONNECTION_RESET"
DROP_ROLLBACK_SECONDS = 10
HTTP_TIMEOUT_SECONDS = 30
POD_DEADLINE_SECONDS = 600


class CaseError(RuntimeError):
    pass


def write_json(path: Path, value) -> None:
    path.write_text(
        json.dumps(scoped_case_evidence(value), indent=2, sort_keys=True) + "\n"
    )
    path.chmod(0o600)


def preflight_metadata(attempt: int, maintenance_window_end: datetime) -> dict:
    now = datetime.now(timezone.utc)
    if now >= maintenance_window_end:
        raise CaseError("approved maintenance window has ended")
    if not CONTROL_NAMESPACE or not NAMESPACE or not DATAPLANE_CONTEXT:
        raise CaseError("GPU_FAULT_DATAPLANE_CONTEXT is required")
    return {
        "observed_at": now.isoformat(),
        "attempt": attempt,
        "maintenance_window_end": maintenance_window_end.isoformat(),
        "region": AWS_REGION,
        "control_namespace": CONTROL_NAMESPACE,
        "dataplane_context": DATAPLANE_CONTEXT,
        "dataplane_namespace": NAMESPACE,
        "cluster_id": "perf-cap-000",
        "node": None,
        "operation": "FREEZE_EVIDENCE",
        "destructive": False,
        "network_scope": "one test-pod result connection reset on loopback proxy",
        "stop_conditions": [
            "any preflight or rollout failure",
            "test pod readiness failure",
            "the first result connection is not reset",
            "the command is not reclaimed after lease expiry",
            "simulated physical execution count differs from one",
            "terminal result replay creates a second notification",
            "executor run loop exits",
            "any cleanup or postflight residual check fails",
        ],
        "rollback": {
            "drop_marker_auto_release_seconds": DROP_ROLLBACK_SECONDS,
            "pod_active_deadline_seconds": POD_DEADLINE_SECONDS,
            "runner_finally_deletes_test_resources": True,
            "runner_finally_purges_synthetic_state": True,
            "runner_finally_restores_registry": True,
        },
    }


def cpu_python(script: str, *arguments: str) -> dict:
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
    return json.loads(output.splitlines()[-1])


def database_residuals() -> dict:
    script = r"""
import json
import os
import psycopg

queries = {
    "objects": (
        "SELECT count(*) FROM gpu_fault_objects "
        "WHERE key LIKE '%net003-%' "
        "OR payload->>'cluster_id' LIKE 'perf-cap-%'"
    ),
    "links": (
        "SELECT count(*) FROM gpu_fault_links "
        "WHERE key LIKE '%net003-%' OR value LIKE '%net003-%'"
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
    "gpu_metric_latest": (
        "SELECT count(*) FROM gpu_fault_gpu_metric_latest "
        "WHERE cluster_id LIKE 'perf-cap-%'"
    ),
    "gpu_metric_batches": (
        "SELECT count(*) FROM gpu_fault_gpu_metrics_batches "
        "WHERE cluster_id LIKE 'perf-cap-%'"
    ),
    "attempt_observations": (
        "SELECT count(*) FROM gpu_fault_attempt_observations "
        "WHERE cluster_id LIKE 'perf-cap-%'"
    ),
    "training_progress": (
        "SELECT count(*) FROM gpu_fault_training_progress "
        "WHERE cluster_id LIKE 'perf-cap-%'"
    ),
}
result = {}
with psycopg.connect(os.environ["GPU_FAULT_STORE_URL"]) as connection:
    with connection.cursor() as cursor:
        for name, query in queries.items():
            cursor.execute(query)
            result[name] = int(cursor.fetchone()[0])
result["total"] = sum(result.values())
print(json.dumps(result, sort_keys=True))
"""
    return cpu_python(script)


def registry_residuals() -> dict:
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


def kubernetes_residuals() -> dict:
    resources = {}
    for kind, name in (
        ("pod", POD),
        ("configmap", CONFIGMAP),
        ("secret", "gpu-fault-perf-clusters"),
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
    return {
        "count": sum(resources.values()),
        "resources": resources,
    }


def seed_command(run_id: str) -> dict:
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

run_id, cluster_id, owner, notification_id, deduplication_key = sys.argv[1:]
incident_id = f"incident-{run_id}"
workflow_id = f"workflow-actionperf-{run_id}"
command_id = f"remote-{run_id}"
step = WorkflowStepSpec(
    operation=WorkflowOperation.FREEZE_EVIDENCE,
    execution_owner=owner,
)
incident = FaultIncident(
    incident_id=incident_id,
    event_id=f"event-{run_id}",
    event_type="NET_ACCEPTANCE",
    cluster_id=cluster_id,
    node_ids=[],
    policy_version="net-acceptance/v1",
    policy_source="ACCEPTANCE",
    state=IncidentState.ACTION_PENDING,
    workflow_request_id=workflow_id,
    fencing_token=1,
    drill_id=run_id,
)
workflow = WorkflowRequest(
    request_id=workflow_id,
    incident_id=incident_id,
    status=WorkflowStatus.PENDING,
    official_action="NO_ACTION",
    fencing_token=1,
    official_steps=[step],
)
notification = AdvisoryNotification(
    notification_id=notification_id,
    deduplication_key=deduplication_key,
    cluster_name=cluster_id,
    incident_id=incident_id,
    subject="NET-003 result retry drill",
    body_text="Synthetic non-destructive result retry acceptance notification.",
    support_case_draft="No provider action required; acceptance drill only.",
    drill_id=run_id,
    category="NET_ACCEPTANCE",
    not_before=datetime.now(timezone.utc) + timedelta(minutes=10),
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
store = ApplicationContext.from_environment().store
store.save_incident(incident)
store.save_workflow(workflow)
store.save_notification_if_absent(notification)
store.ensure_remote_command(command)
print(json.dumps({
    "incident_id": incident_id,
    "event_id": incident.event_id,
    "workflow_id": workflow_id,
    "command_id": command_id,
    "cluster_id": cluster_id,
    "notification_id": notification_id,
    "deduplication_key": deduplication_key,
}))
"""
    notification_id = f"notification-{run_id}"
    deduplication_key = f"{run_id}/result-retry"
    return cpu_python(
        script,
        run_id,
        "perf-cap-000",
        OWNER,
        notification_id,
        deduplication_key,
    )


def command_snapshot(command_id: str) -> dict:
    script = r"""
import json
import sys
from gpu_fault.app import ApplicationContext
command = ApplicationContext.from_environment().store.get_remote_command(sys.argv[1])
print(json.dumps({
    "command_id": command.command_id,
    "status": command.status.value,
    "last_lease_owner": command.last_lease_owner,
    "lease_owner": command.lease_owner,
    "lease_expires_at": (
        command.lease_expires_at.isoformat()
        if command.lease_expires_at is not None else None
    ),
    "status_source": command.status_source,
    "error": command.error,
    "result_details": command.result_details,
}, sort_keys=True))
"""
    return cpu_python(script, command_id)


def notification_snapshot(notification_id: str, deduplication_key: str) -> dict:
    script = r"""
import json
import os
import sys
import psycopg

notification_id, deduplication_key = sys.argv[1:]
objects = {}
with psycopg.connect(os.environ["GPU_FAULT_STORE_URL"]) as connection:
    with connection.cursor() as cursor:
        cursor.execute(
            '''
            SELECT kind, payload->>'status'
            FROM gpu_fault_objects
            WHERE key=%s
              AND kind IN (
                  'notification',
                  'notification_delivery',
                  'notification_result'
              )
            ORDER BY kind
            ''',
            (notification_id,),
        )
        for kind, status in cursor.fetchall():
            objects[kind] = {
                "count": objects.get(kind, {}).get("count", 0) + 1,
                "status": status,
            }
        cursor.execute(
            '''
            SELECT count(*)
            FROM gpu_fault_links
            WHERE kind='notification_dedup'
              AND key=%s
              AND value=%s
            ''',
            (deduplication_key, notification_id),
        )
        link_count = int(cursor.fetchone()[0])
print(json.dumps({"objects": objects, "dedup_link_count": link_count}, sort_keys=True))
"""
    return cpu_python(script, notification_id, deduplication_key)


def purge_seed(seed: dict) -> dict:
    script = r"""
import json
import os
import sys
import psycopg

deleted = {}
remaining = []
with psycopg.connect(os.environ["GPU_FAULT_STORE_URL"], autocommit=True) as connection:
    cursor = connection.cursor()
    cursor.execute(
        '''
        DELETE FROM gpu_fault_links
        WHERE kind='incident_by_event'
          AND key=%s
          AND value=%s
        ''',
        (sys.argv[6], sys.argv[3]),
    )
    deleted[f"incident_by_event/{sys.argv[6]}"] = cursor.rowcount
    cursor.execute(
        '''
        DELETE FROM gpu_fault_links
        WHERE kind='notification_dedup'
          AND key=%s
          AND value=%s
        ''',
        (sys.argv[5], sys.argv[4]),
    )
    deleted[f"notification_dedup/{sys.argv[5]}"] = cursor.rowcount
    items = (
        ("remote_command", sys.argv[1]),
        ("workflow", sys.argv[2]),
        ("incident", sys.argv[3]),
        ("notification_result", sys.argv[4]),
        ("notification_delivery", sys.argv[4]),
        ("notification", sys.argv[4]),
    )
    for kind, key in items:
        cursor.execute(
            "DELETE FROM gpu_fault_objects WHERE kind=%s AND key=%s",
            (kind, key),
        )
        deleted[f"{kind}/{key}"] = cursor.rowcount
    cursor.execute(
        '''
        SELECT kind, key
        FROM gpu_fault_objects
        WHERE (kind, key) IN (
            ('remote_command', %s),
            ('workflow', %s),
            ('incident', %s),
            ('notification_result', %s),
            ('notification_delivery', %s),
            ('notification', %s)
        )
        ORDER BY kind, key
        ''',
        (
            sys.argv[1],
            sys.argv[2],
            sys.argv[3],
            sys.argv[4],
            sys.argv[4],
            sys.argv[4],
        ),
    )
    remaining = [
        {"kind": kind, "key": key}
        for kind, key in cursor.fetchall()
    ]
    cursor.execute(
        '''
        SELECT count(*)
        FROM gpu_fault_links
        WHERE kind='notification_dedup'
          AND key=%s
          AND value=%s
        ''',
        (sys.argv[5], sys.argv[4]),
    )
    remaining_notification_links = int(cursor.fetchone()[0])
    cursor.execute(
        '''
        SELECT count(*)
        FROM gpu_fault_links
        WHERE kind='incident_by_event'
          AND key=%s
          AND value=%s
        ''',
        (sys.argv[6], sys.argv[3]),
    )
    remaining_incident_links = int(cursor.fetchone()[0])
print(json.dumps({
    "deleted": deleted,
    "remaining": remaining,
    "remaining_links": (
        remaining_notification_links + remaining_incident_links
    ),
}, sort_keys=True))
"""
    result = cpu_python(
        script,
        str(seed["command_id"]),
        str(seed["workflow_id"]),
        str(seed["incident_id"]),
        str(seed["notification_id"]),
        str(seed["deduplication_key"]),
        str(seed["event_id"]),
    )
    if result.get("remaining") or result.get("remaining_links"):
        raise CaseError(f"seed cleanup left residual state: {result}")
    return result


def pod_manifest(
    image: str,
    identity: dict[str, object],
    notification_id: str,
) -> dict:
    return {
        "apiVersion": "v1",
        "kind": "Pod",
        "metadata": {
            "name": POD,
            "namespace": NAMESPACE,
            "labels": {"app": "gpu-fault-net003-executor"},
        },
        "spec": {
            "restartPolicy": "Never",
            "activeDeadlineSeconds": POD_DEADLINE_SECONDS,
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
                        {
                            "name": "EXECUTOR_ARTIFACT_SHA256",
                            "value": str(identity["executor_artifact_sha256"]),
                        },
                        {
                            "name": "EXECUTOR_COMPATIBILITY_DIGEST",
                            "value": str(identity["executor_compatibility_digest"]),
                        },
                        {
                            "name": "DROP_ROLLBACK_SECONDS",
                            "value": str(DROP_ROLLBACK_SECONDS),
                        },
                        {
                            "name": "HTTP_TIMEOUT_SECONDS",
                            "value": str(HTTP_TIMEOUT_SECONDS),
                        },
                        {
                            "name": "NOTIFICATION_ID",
                            "value": notification_id,
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


def wait_file(path: str, timeout_seconds: int) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        result = dataplane(
            "exec",
            POD,
            "--",
            "sh",
            "-c",
            f"if [ -f {shlex.quote(path)} ]; then printf present; fi",
            check=False,
        )
        if result.strip() == "present":
            return
        time.sleep(1)
    raise CaseError(f"timed out waiting for {path}")


def read_state(path: str) -> dict:
    output = dataplane("exec", POD, "--", "cat", path)
    return json.loads(output)


def file_present(path: str) -> bool:
    output = dataplane(
        "exec",
        POD,
        "--",
        "sh",
        "-c",
        f"if [ -f {shlex.quote(path)} ]; then printf present; fi",
        check=False,
    )
    return output.strip() == "present"


def wait_command(command_id: str, status: str, timeout_seconds: int) -> dict:
    deadline = time.monotonic() + timeout_seconds
    last = {}
    while time.monotonic() < deadline:
        last = command_snapshot(command_id)
        if last.get("status") == status:
            return last
        time.sleep(2)
    raise CaseError(f"command did not reach {status}: {last}")


def wait_executor_state(timeout_seconds: int) -> dict:
    deadline = time.monotonic() + timeout_seconds
    last = {}
    while time.monotonic() < deadline:
        last = read_state("/state/executor-state.json")
        if int(last.get("claimed_total", 0)) >= 2:
            return last
        time.sleep(1)
    return last


def _run_net003_case(
    case_dir: Path,
    run_id: str,
    attempt: int,
    maintenance_window_end: datetime,
    state: dict,
) -> dict:
    notification_id = f"notification-{run_id}"
    deduplication_key = f"{run_id}/result-retry"
    preflight = preflight_metadata(attempt, maintenance_window_end)
    write_json(case_dir / "preflight.json", preflight)
    database_preflight = database_residuals()
    write_json(case_dir / "database-preflight.json", database_preflight)
    if database_preflight.get("total") != 0:
        raise CaseError(f"database preflight found residuals: {database_preflight}")
    registry_preflight = registry_residuals()
    write_json(case_dir / "registry-verification-preflight.json", registry_preflight)
    if registry_preflight.get("count") != 0:
        raise CaseError(f"registry preflight found residuals: {registry_preflight}")
    kubernetes_preflight = kubernetes_residuals()
    write_json(case_dir / "kubernetes-preflight.json", kubernetes_preflight)
    if kubernetes_preflight.get("count") != 0:
        raise CaseError(f"Kubernetes preflight found residuals: {kubernetes_preflight}")
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
    dataplane("delete", "pod", POD, "--ignore-not-found", check=False)
    dataplane(
        "apply",
        "-f",
        "-",
        stdin=json.dumps(pod_manifest(image, identity, notification_id)).encode(),
    )
    dataplane("wait", "--for=condition=Ready", f"pod/{POD}", "--timeout=180s")
    wait_file("/state/ready.json", 60)
    ready = read_state("/state/ready.json")
    write_json(case_dir / "probe-ready.json", ready)
    seed = seed_command(run_id)
    state["seed"] = seed
    write_json(case_dir / "seed.json", seed)
    if (
        seed.get("notification_id") != notification_id
        or seed.get("deduplication_key") != deduplication_key
    ):
        raise CaseError("seed notification identity is inconsistent")
    notification_baseline = notification_snapshot(notification_id, deduplication_key)
    write_json(case_dir / "notification-baseline.json", notification_baseline)
    wait_file("/state/action-started", 120)
    leased = command_snapshot(str(seed["command_id"]))
    write_json(case_dir / "leased-command.json", leased)
    if leased.get("status") != "LEASED" or not leased.get("lease_expires_at"):
        raise CaseError(f"command was not actively leased before injection: {leased}")
    wait_file("/state/result-submit-started.json", 30)
    result_submit_started = read_state("/state/result-submit-started.json")
    write_json(case_dir / "result-submit-started.json", result_submit_started)
    wait_file("/state/drop-observed.json", 30)
    drop_observed = read_state("/state/drop-observed.json")
    write_json(case_dir / "drop-observed.json", drop_observed)
    interrupted = command_snapshot(str(seed["command_id"]))
    write_json(case_dir / "interrupted-command.json", interrupted)
    final = wait_command(str(seed["command_id"]), "SUCCEEDED", 180)
    write_json(case_dir / "final-command.json", final)
    wait_file("/state/result-replays.json", 30)
    result_replays = read_state("/state/result-replays.json")
    write_json(case_dir / "result-replays.json", result_replays)
    notification_final = notification_snapshot(notification_id, deduplication_key)
    write_json(case_dir / "notification-final.json", notification_final)
    executor_state = wait_executor_state(30)
    write_json(case_dir / "executor-state.json", executor_state)
    ledger = read_state("/state/ledger.json")
    write_json(case_dir / "ledger.json", ledger)
    logs = dataplane("logs", POD, check=False, timeout=120)
    (case_dir / "executor.log").write_text(logs)
    (case_dir / "executor.log").chmod(0o600)
    rollback_triggered = file_present("/state/rollback.json")
    errors = _net003_errors(
        ready,
        leased,
        interrupted,
        final,
        result_replays,
        executor_state,
        ledger,
        logs,
        rollback_triggered,
        drop_observed,
        notification_id,
        notification_baseline,
        notification_final,
    )
    pod_phase = dataplane("get", "pod", POD, "-o", "jsonpath={.status.phase}").strip()
    if pod_phase != "Running":
        errors.append("executor run loop did not remain running")
    return {
        "case_id": "GF-REGIONAL-NET-003",
        "attempt": attempt,
        "verdict": "PASS" if not errors else "FAIL",
        "errors": errors,
        "preflight": preflight,
        "network_interruption": {
            "type": "single-result-connection-reset",
            "result_submit_started": result_submit_started,
            "drop_observed": drop_observed,
            "watchdog_rollback_triggered": rollback_triggered,
        },
        "leased_command": leased,
        "interrupted_command": interrupted,
        "command": final,
        "result_replays": result_replays,
        "executor_state": executor_state,
        "ledger": ledger,
        "notification_baseline": notification_baseline,
        "notification_final": notification_final,
        "pod_phase": pod_phase,
    }


def _net003_errors(
    ready: dict,
    leased: dict,
    interrupted: dict,
    final: dict,
    result_replays: dict,
    executor_state: dict,
    ledger: dict,
    logs: str,
    rollback_triggered: bool,
    drop_observed: dict,
    notification_id: str,
    notification_baseline: dict,
    notification_final: dict,
) -> list[str]:
    errors = []
    if ledger.get("physical_count") != 1:
        errors.append("physical action count is not one")
    if len(ledger.get("keys", [])) != 1:
        errors.append("idempotency ledger does not contain exactly one key")
    if int(executor_state.get("claimed_total", 0)) != 2:
        errors.append("command was not reclaimed exactly once after lease expiry")
    if int(executor_state.get("reported_failures", 0)) != 0:
        errors.append("connection reset was misclassified as a reported result")
    if int(executor_state.get("unexpected_failures", 0)):
        errors.append("executor recorded an unexpected failure")
    if int(executor_state.get("lease_renewal_failures", 0)) != 0:
        errors.append("lease renewal ran despite immediate result interruption")
    if ready.get("drop_rollback_seconds") != DROP_ROLLBACK_SECONDS:
        errors.append("connection-drop rollback timer is not configured")
    if ready.get("http_timeout_seconds") != HTTP_TIMEOUT_SECONDS:
        errors.append("executor HTTP timeout is not the approved value")
    if ready.get("result_connection_reset") is not True:
        errors.append("result connection reset is not configured")
    if ready.get("terminal_result_replays") != 2:
        errors.append("terminal result replay count is not configured")
    if drop_observed.get("connection_reset") is not True:
        errors.append("proxy did not record a connection reset")
    if rollback_triggered:
        errors.append("connection-drop marker required watchdog rollback")
    if interrupted.get("status") != "LEASED":
        errors.append("interrupted command did not remain leased")
    if interrupted.get("lease_expires_at") != leased.get("lease_expires_at"):
        errors.append("command lease changed before reclaim")
    if result_replays.get("count") != 2 or any(
        item.get("status") != "SUCCEEDED"
        for item in result_replays.get("responses", [])
    ):
        errors.append("terminal result was not replayed twice idempotently")
    transport_markers = (
        "Connection reset by peer",
        "ConnectionResetError",
        "Remote end closed connection",
        "RemoteDisconnected",
        "URLError",
    )
    if "regional cluster executor claim failed; retrying" not in logs or not any(
        marker in logs for marker in transport_markers
    ):
        errors.append("executor log has no result-submit transport interruption")
    if "rejected request (409)" in logs:
        errors.append("NET-003 unexpectedly followed the stale-result 409 path")
    if final.get("result_details", {}).get("cached") is not True:
        errors.append("reclaimed command did not use the idempotency ledger")
    if final.get("result_details", {}).get("notification_id") != notification_id:
        errors.append("final result lost the notification identity")
    baseline_objects = notification_baseline.get("objects", {})
    final_objects = notification_final.get("objects", {})
    if baseline_objects.get("notification", {}).get("count") != 1:
        errors.append("notification baseline does not contain exactly one object")
    if baseline_objects.get("notification_delivery", {}).get("count") != 1:
        errors.append("notification baseline has no single delivery row")
    if baseline_objects.get("notification_result", {}).get("count", 0) != 0:
        errors.append("notification was sent before result completion")
    if notification_baseline.get("dedup_link_count") != 1:
        errors.append("notification baseline dedup link count is not one")
    for kind in ("notification", "notification_delivery", "notification_result"):
        if final_objects.get(kind, {}).get("count") != 1:
            errors.append(f"final {kind} count is not one")
    if final_objects.get("notification_result", {}).get("status") != "SKIPPED":
        errors.append("drill notification was not safely suppressed")
    if notification_final.get("dedup_link_count") != 1:
        errors.append("terminal replays created a second dedup link")
    return errors


def _cleanup_net003(case_dir: Path, run_id: str, result: dict, seed: dict) -> None:
    if seed:
        try:
            seed_cleanup = purge_seed(seed)
            result["seed_cleanup"] = seed_cleanup
            write_json(case_dir / "seed-cleanup.json", seed_cleanup)
        except Exception as exc:
            result["cleanup_error"] = f"seed cleanup: {exc}"
            result["verdict"] = "FAIL"
    dataplane("delete", "pod", POD, "--ignore-not-found", check=False)
    dataplane("delete", "configmap", CONFIGMAP, "--ignore-not-found", check=False)
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
        result["cleanup_error"] = f"registry cleanup: {exc}"
        result["verdict"] = "FAIL"
    try:
        postflight = {
            "database": database_residuals(),
            "registry": registry_residuals(),
            "kubernetes": kubernetes_residuals(),
        }
        file_names = {
            "database": "database-postflight.json",
            "registry": "registry-verification-postflight.json",
            "kubernetes": "kubernetes-postflight.json",
        }
        for name, value in postflight.items():
            result[f"{name}_postflight"] = value
            write_json(case_dir / file_names[name], value)
            key = "total" if name == "database" else "count"
            if value.get(key) != 0:
                raise CaseError(f"{name} postflight found residuals: {value}")
    except Exception as exc:
        result["postflight_error"] = f"{type(exc).__name__}: {exc}"
        result["verdict"] = "FAIL"


def run_case(
    run_dir: Path,
    attempt: int,
    maintenance_window_end: datetime,
) -> int:
    case_dir = run_dir / "cases" / "GF-REGIONAL-NET-003"
    case_dir.mkdir(parents=True, exist_ok=True)
    run_id = f"net003-{run_dir.name.rsplit('-', 1)[-1].lower()}-a{attempt}"
    result = {"case_id": "GF-REGIONAL-NET-003", "verdict": "FAIL"}
    state: dict = {"seed": {}}
    try:
        result = _run_net003_case(
            case_dir,
            run_id,
            attempt,
            maintenance_window_end,
            state,
        )
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        _cleanup_net003(case_dir, run_id, result, state["seed"])
    write_json(case_dir / "GF-REGIONAL-NET-003.json", result)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["verdict"] == "PASS" else 1


def main() -> int:
    install_site_profile()
    parser = argparse.ArgumentParser()
    add_live_arguments(parser, confirmation=CONFIRMATION)
    args = parser.parse_args()
    os.umask(0o077)
    if not args.execute:
        plan = build_plan(
            run_dir=args.run_dir,
            case_id="GF-REGIONAL-NET-003",
            attempt=args.attempt,
            confirmation=CONFIRMATION,
            details={
                "risk": "live-non-destructive",
                "synthetic_cluster_id": "perf-cap-000",
                "network_scope": "one test-Pod result connection reset",
                "drop_marker_rollback_seconds": DROP_ROLLBACK_SECONDS,
                "pod_active_deadline_seconds": POD_DEADLINE_SECONDS,
                "terminal_result_replays": 2,
                "mutations": [
                    "temporary synthetic registry entry",
                    "controlled CPU registry rollouts",
                    "temporary GPU probe Pod and ConfigMap",
                ],
            },
        )
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0
    deadline = authorize_execution(
        args,
        case_id="GF-REGIONAL-NET-003",
        confirmation=CONFIRMATION,
    )
    return run_case(args.run_dir, args.attempt, deadline)


if __name__ == "__main__":
    raise SystemExit(main())

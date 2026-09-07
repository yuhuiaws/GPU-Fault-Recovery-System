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

SCRIPT = Path(__file__).with_name("probes") / "net002_executor.py"
CONFIGMAP = "gpu-fault-net002-script"
POD = "gpu-fault-net002-executor"
OWNER = "gpu-fault-net-test"
CONFIRMATION = "NET002_LIVE_REGISTRY_INTERRUPTION"
BLOCK_SECONDS = 120
BLOCK_ROLLBACK_SECONDS = 150
HTTP_TIMEOUT_SECONDS = 180
POD_DEADLINE_SECONDS = 900


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
        "network_scope": "test pod loopback proxy only",
        "stop_conditions": [
            "any preflight or rollout failure",
            "test pod readiness failure",
            "lease does not expire before recovery",
            "first stale result is not rejected with HTTP 409",
            "simulated physical execution count differs from one",
            "executor run loop exits",
            "any cleanup or postflight residual check fails",
        ],
        "rollback": {
            "network_auto_release_seconds": BLOCK_ROLLBACK_SECONDS,
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
        "WHERE key LIKE '%net002-%' "
        "OR payload->>'cluster_id' LIKE 'perf-cap-%'"
    ),
    "links": (
        "SELECT count(*) FROM gpu_fault_links "
        "WHERE key LIKE '%net002-%' OR value LIKE '%net002-%'"
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
from datetime import datetime, timezone

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
now = datetime.now(timezone.utc)
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
store.ensure_remote_command(command)
print(json.dumps({
    "incident_id": incident_id,
    "event_id": incident.event_id,
    "workflow_id": workflow_id,
    "command_id": command_id,
    "cluster_id": cluster_id,
}))
"""
    return cpu_python(script, run_id, "perf-cap-000", OWNER)


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
        (sys.argv[4], sys.argv[3]),
    )
    deleted[f"incident_by_event/{sys.argv[4]}"] = cursor.rowcount
    items = (
        ("remote_command", sys.argv[1]),
        ("workflow", sys.argv[2]),
        ("incident", sys.argv[3]),
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
            ('incident', %s)
        )
        ORDER BY kind, key
        ''',
        (sys.argv[1], sys.argv[2], sys.argv[3]),
    )
    remaining = [
        {"kind": kind, "key": key}
        for kind, key in cursor.fetchall()
    ]
    cursor.execute(
        '''
        SELECT count(*)
        FROM gpu_fault_links
        WHERE kind='incident_by_event'
          AND key=%s
          AND value=%s
        ''',
        (sys.argv[4], sys.argv[3]),
    )
    remaining_links = int(cursor.fetchone()[0])
print(json.dumps({
    "deleted": deleted,
    "remaining": remaining,
    "remaining_links": remaining_links,
}, sort_keys=True))
"""
    result = cpu_python(
        script,
        str(seed["command_id"]),
        str(seed["workflow_id"]),
        str(seed["incident_id"]),
        str(seed["event_id"]),
    )
    if result.get("remaining") or result.get("remaining_links"):
        raise CaseError(f"seed cleanup left residual state: {result}")
    return result


def pod_manifest(image: str, identity: dict[str, object]) -> dict:
    return {
        "apiVersion": "v1",
        "kind": "Pod",
        "metadata": {
            "name": POD,
            "namespace": NAMESPACE,
            "labels": {"app": "gpu-fault-net002-executor"},
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
                            "name": "BLOCK_ROLLBACK_SECONDS",
                            "value": str(BLOCK_ROLLBACK_SECONDS),
                        },
                        {
                            "name": "HTTP_TIMEOUT_SECONDS",
                            "value": str(HTTP_TIMEOUT_SECONDS),
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
        if (
            int(last.get("claimed_total", 0)) >= 2
            and int(last.get("reported_failures", 0)) >= 1
        ):
            return last
        time.sleep(1)
    return last


def _run_net002_case(
    case_dir: Path,
    run_id: str,
    attempt: int,
    maintenance_window_end: datetime,
    state: dict,
) -> dict:
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
        stdin=json.dumps(pod_manifest(image, identity)).encode(),
    )
    dataplane("wait", "--for=condition=Ready", f"pod/{POD}", "--timeout=180s")
    wait_file("/state/ready.json", 60)
    ready = read_state("/state/ready.json")
    write_json(case_dir / "probe-ready.json", ready)
    seed = seed_command(run_id)
    state["seed"] = seed
    write_json(case_dir / "seed.json", seed)
    wait_file("/state/action-started", 120)
    leased = command_snapshot(str(seed["command_id"]))
    write_json(case_dir / "leased-command.json", leased)
    if leased.get("status") != "LEASED" or not leased.get("lease_expires_at"):
        raise CaseError(f"command was not actively leased before injection: {leased}")
    dataplane("exec", POD, "--", "touch", "/state/block")
    blocked_at = datetime.now(timezone.utc)
    blocked_started = time.monotonic()
    wait_file("/state/action-gate-observed.json", 30)
    action_gate_observed = read_state("/state/action-gate-observed.json")
    write_json(case_dir / "action-gate-observed.json", action_gate_observed)
    wait_file("/state/result-submit-waiting.json", 30)
    result_submit_waiting = read_state("/state/result-submit-waiting.json")
    write_json(case_dir / "result-submit-waiting.json", result_submit_waiting)
    while (elapsed := time.monotonic() - blocked_started) < BLOCK_SECONDS:
        time.sleep(min(1.0, BLOCK_SECONDS - elapsed))
    expired = command_snapshot(str(seed["command_id"]))
    write_json(case_dir / "expired-command.json", expired)
    dataplane("exec", POD, "--", "rm", "-f", "/state/block")
    unblocked_at = datetime.now(timezone.utc)
    blocked_seconds = time.monotonic() - blocked_started
    wait_file("/state/result-submit-released.json", 30)
    result_submit_released = read_state("/state/result-submit-released.json")
    write_json(case_dir / "result-submit-released.json", result_submit_released)
    final = wait_command(str(seed["command_id"]), "SUCCEEDED", 180)
    write_json(case_dir / "final-command.json", final)
    executor_state = wait_executor_state(30)
    write_json(case_dir / "executor-state.json", executor_state)
    ledger = read_state("/state/ledger.json")
    write_json(case_dir / "ledger.json", ledger)
    logs = dataplane("logs", POD, check=False, timeout=120)
    (case_dir / "executor.log").write_text(logs)
    (case_dir / "executor.log").chmod(0o600)
    errors = _net002_errors(
        ready,
        leased,
        expired,
        final,
        executor_state,
        ledger,
        logs,
        blocked_seconds,
        unblocked_at,
    )
    pod_phase = dataplane("get", "pod", POD, "-o", "jsonpath={.status.phase}").strip()
    if pod_phase != "Running":
        errors.append("executor run loop did not remain running")
    return {
        "case_id": "GF-REGIONAL-NET-002",
        "attempt": attempt,
        "verdict": "PASS" if not errors else "FAIL",
        "errors": errors,
        "preflight": preflight,
        "network_interruption": {
            "blocked_at": blocked_at.isoformat(),
            "unblocked_at": unblocked_at.isoformat(),
            "blocked_seconds": round(blocked_seconds, 3),
            "automatic_rollback_seconds": BLOCK_ROLLBACK_SECONDS,
        },
        "leased_command": leased,
        "expired_command": expired,
        "command": final,
        "result_submit_waiting": result_submit_waiting,
        "result_submit_released": result_submit_released,
        "action_gate_observed": action_gate_observed,
        "executor_state": executor_state,
        "ledger": ledger,
        "pod_phase": pod_phase,
    }


def _net002_errors(
    ready: dict,
    leased: dict,
    expired: dict,
    final: dict,
    executor_state: dict,
    ledger: dict,
    logs: str,
    blocked_seconds: float,
    unblocked_at: datetime,
) -> list[str]:
    errors = []
    if ledger.get("physical_count") != 1:
        errors.append("physical action count is not one")
    if len(ledger.get("keys", [])) != 1:
        errors.append("idempotency ledger does not contain exactly one key")
    if int(executor_state.get("claimed_total", 0)) != 2:
        errors.append("command was not reclaimed exactly once after lease expiry")
    if int(executor_state.get("reported_failures", 0)) != 1:
        errors.append("stale result rejection count is not one")
    if int(executor_state.get("unexpected_failures", 0)):
        errors.append("executor recorded an unexpected failure")
    if int(executor_state.get("lease_renewal_failures", 0)) < 1:
        errors.append("network interruption did not reject lease renewal")
    if blocked_seconds < BLOCK_SECONDS:
        errors.append("network interruption ended before 120 seconds")
    if blocked_seconds >= BLOCK_ROLLBACK_SECONDS:
        errors.append("network interruption exceeded automatic rollback bound")
    if ready.get("block_rollback_seconds") != BLOCK_ROLLBACK_SECONDS:
        errors.append("automatic network rollback timer is not configured")
    if ready.get("http_timeout_seconds") != HTTP_TIMEOUT_SECONDS:
        errors.append("executor HTTP timeout does not cover the interruption")
    if ready.get("result_submission_gate") is not True:
        errors.append("result submission gate is not configured")
    if ready.get("action_requires_network_block") is not True:
        errors.append("simulated action is not ordered after network injection")
    lease_expires_raw = leased.get("lease_expires_at")
    if not isinstance(lease_expires_raw, str) or not lease_expires_raw:
        errors.append("leased command has no expiry timestamp")
    else:
        lease_expires_at = datetime.fromisoformat(
            lease_expires_raw.replace("Z", "+00:00")
        )
        if lease_expires_at >= unblocked_at:
            errors.append("command lease had not expired before network recovery")
    if expired.get("status") != "LEASED":
        errors.append("command did not remain leased throughout the interruption")
    if expired.get("lease_expires_at") != leased.get("lease_expires_at"):
        errors.append("command lease changed during the interruption")
    if "rejected request (409)" not in logs or (
        "remote command lease is missing, stale, or changed" not in logs
    ):
        errors.append("executor log has no stale-lease HTTP 409")
    if final.get("result_details", {}).get("cached") is not True:
        errors.append("reclaimed command did not use the idempotency ledger")
    return errors


def _cleanup_net002(case_dir: Path, run_id: str, result: dict, seed: dict) -> None:
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
    case_dir = run_dir / "cases" / "GF-REGIONAL-NET-002"
    case_dir.mkdir(parents=True, exist_ok=True)
    run_id = f"net002-{run_dir.name.rsplit('-', 1)[-1].lower()}-a{attempt}"
    result = {"case_id": "GF-REGIONAL-NET-002", "verdict": "FAIL"}
    state: dict = {"seed": {}}
    try:
        result = _run_net002_case(
            case_dir,
            run_id,
            attempt,
            maintenance_window_end,
            state,
        )
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        _cleanup_net002(case_dir, run_id, result, state["seed"])
    write_json(case_dir / "GF-REGIONAL-NET-002.json", result)
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
            case_id="GF-REGIONAL-NET-002",
            attempt=args.attempt,
            confirmation=CONFIRMATION,
            details={
                "risk": "live-non-destructive",
                "synthetic_cluster_id": "perf-cap-000",
                "network_scope": "test Pod loopback proxy only",
                "automatic_network_rollback_seconds": BLOCK_ROLLBACK_SECONDS,
                "pod_active_deadline_seconds": POD_DEADLINE_SECONDS,
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
        case_id="GF-REGIONAL-NET-002",
        confirmation=CONFIRMATION,
    )
    return run_case(args.run_dir, args.attempt, deadline)


if __name__ == "__main__":
    raise SystemExit(main())

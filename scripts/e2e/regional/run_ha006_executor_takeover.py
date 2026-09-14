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
from typing import Any

if __package__:
    from .acceptance_runner_common import write_json_atomic
    from .live_driver_guard import (
        add_live_arguments,
        authorize_execution,
        build_plan,
        install_site_profile,
    )
else:
    from acceptance_runner_common import write_json_atomic
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
# Lease held by the seeded workflow's seed identity; must outlive the case
# (see seed_command), cleanup deletes the rows.
SEED_LEASE_SECONDS = 30 * 60
LEASE_SECONDS = 30
WINNER_SLEEP_SECONDS = 60
# The fixture executors poll every second but back off up to this after idle
# claims, so a survivor may notice an expired lease up to poll+backoff late.
POLL_SECONDS = 1
CLAIM_BACKOFF_MAX_SECONDS = 4
SAMPLE_INTERVAL_SECONDS = 0.5
CONFIRMATION = "HA006_FORCE_DELETE_TEST_EXECUTOR"


class CaseError(RuntimeError):
    pass


def log(message: str) -> None:
    stamp = datetime.now(timezone.utc).strftime("%H:%M:%S")
    print(f"[{stamp}] {message}", flush=True)


def parse_timestamp(value: object) -> datetime | None:
    if value in (None, ""):
        return None
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def remaining_lease_seconds(post_kill: dict, killed_at: datetime) -> float | None:
    """Seconds of lease left at the kill, or None when the command held no lease.

    ``lease_expires_at`` is None outside LEASED: the kill then landed while the
    command was WAITING or already terminal, and the LEASED-branch timing bound
    has nothing to measure against. The caller records INCONCLUSIVE for that
    branch rather than crashing on ``fromisoformat(None)``.
    """

    expiry = parse_timestamp(post_kill.get("lease_expires_at"))
    if expiry is None:
        return None
    return max(0.0, (expiry - killed_at).total_seconds())


def takeover_timing(
    final: dict,
    *,
    killed_at: datetime,
    observed_completed_at: datetime,
    poll_seconds: int = POLL_SECONDS,
    claim_backoff_max_seconds: int = CLAIM_BACKOFF_MAX_SECONDS,
    sample_interval_seconds: float = SAMPLE_INTERVAL_SECONDS,
) -> dict[str, Any]:
    """Takeover duration from the store's own completion timestamp when it has one.

    ``updated_at`` on the SUCCEEDED command is written by the control plane
    when the survivor's result lands; measuring from it removes the sampler's
    ~0.5s granularity. Only when it is missing does the sampled time stand in,
    and then the sampling period is part of the tolerance. The tolerance itself
    is the executor's poll plus its idle-claim backoff -- the longest a survivor
    can legitimately take to notice an expired lease -- not a constant 5s.
    """

    store_completed_at = parse_timestamp(final.get("updated_at"))
    if store_completed_at is not None:
        completed_at = store_completed_at
        source = "store.updated_at"
        tolerance = float(poll_seconds + claim_backoff_max_seconds)
    else:
        completed_at = observed_completed_at
        source = "sampled"
        tolerance = float(poll_seconds + claim_backoff_max_seconds) + float(
            sample_interval_seconds
        )
    return {
        "completed_at": completed_at.isoformat(),
        "completion_source": source,
        "takeover_seconds": round((completed_at - killed_at).total_seconds(), 3),
        "tolerance_seconds": tolerance,
    }


def physical_action_total(*states: dict) -> int:
    """Physical actions across every executor's counter file, summed by the runner."""

    return sum(int((state or {}).get("physical_actions", 0)) for state in states)


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
    script = (
        _registry.STORE_DSN_SNIPPET
        + r"""
import json
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
with psycopg.connect(store_dsn()) as connection:
    cursor = connection.cursor()
    for name, query in queries.items():
        cursor.execute(query)
        result[name] = int(cursor.fetchone()[0])
result["total"] = sum(result.values())
print(json.dumps(result, sort_keys=True))
"""
    )
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

run_id, cluster_id, owner, raw_lease_seconds = sys.argv[1:]
lease_seconds = int(raw_lease_seconds)
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
# Leased to a seed identity, never BLOCKED. The dispatcher's orphan sweep
# cancels every open command of a terminal workflow (BLOCKED included), and
# because the completion route wakes the dispatcher it did so 50 ms after the
# probe completed step 0 of HA-001 (attempt 3, 2026-09-14), failing the
# remaining PENDING commands before the probe could claim them. A PENDING
# workflow under an unexpired foreign lease is live work only its lease holder
# may drive; the probe alone claims the commands, and cleanup deletes the rows.
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
    return cpu_python(script, run_id, "perf-cap-000", OWNER, str(SEED_LEASE_SECONDS))


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
    "updated_at": item.updated_at.isoformat(),
    "result_details": item.result_details,
    "error": item.error,
}, sort_keys=True))
"""
    return cpu_python(script, command_id)


def notification_snapshot(deduplication_key: str) -> dict:
    script = (
        _registry.STORE_DSN_SNIPPET
        + r"""
import json
import sys
import psycopg
key = sys.argv[1]
with psycopg.connect(store_dsn()) as connection:
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
    )
    return cpu_python(script, deduplication_key)


def cleanup_seed(seed: dict, notification_id: str | None) -> dict:
    script = (
        _registry.STORE_DSN_SNIPPET
        + r"""
import json
import sys
import psycopg
incident_id, event_id, workflow_id, command_id, dedup_key, notification_id = sys.argv[1:]
deleted = {}
with psycopg.connect(store_dsn(), autocommit=True) as connection:
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
    )
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
                        {"name": "POLL_SECONDS", "value": str(POLL_SECONDS)},
                        {
                            "name": "CLAIM_BACKOFF_MAX_SECONDS",
                            "value": str(CLAIM_BACKOFF_MAX_SECONDS),
                        },
                        {"name": "WAIT_FIRST_ROUND", "value": "true"},
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


def wait_first_owner(
    seed: dict, timeout_seconds: int = 120
) -> tuple[dict, dict, list[dict]]:
    """Wait for the LEASED shared-ledger winner, keeping the WAITING round on record.

    The fixture adapter reports WAITING on its first claim, so the samples taken
    here are the evidence for the WAITING branch: which replica returned
    WAITING (``last_lease_owner`` while the status is WAITING) and which one
    re-claimed and won the ledger.
    """

    deadline = time.monotonic() + timeout_seconds
    last_command: dict = {}
    last_notification: dict = {}
    timeline: list[dict] = []
    while time.monotonic() < deadline:
        last_command = command_snapshot(str(seed["command_id"]))
        last_notification = notification_snapshot(str(seed["deduplication_key"]))
        timeline.append(
            {
                "observed_at": datetime.now(timezone.utc).isoformat(),
                "status": last_command.get("status"),
                "lease_owner": last_command.get("lease_owner"),
                "last_lease_owner": last_command.get("last_lease_owner"),
                "round": (last_command.get("result_details") or {}).get("round"),
            }
        )
        if (
            last_command.get("status") == "LEASED"
            and last_command.get("lease_owner") in PODS
            and last_notification.get("notification_id")
        ):
            return last_command, last_notification, timeline
        time.sleep(SAMPLE_INTERVAL_SECONDS)
    raise CaseError(
        f"first owner did not acquire shared ledger: {last_command} {last_notification}"
    )


def waiting_branch(timeline: list[dict]) -> dict[str, Any]:
    """What the pre-kill samples show about the WAITING round.

    ``observed`` is True when a sample caught the command in WAITING (or a
    result whose ``round`` is 1); ``waiting_owner`` is the replica that returned
    it and ``reclaimed_by_other_replica`` whether a different replica took the
    second round. Store sampling at 0.5s can miss a short WAITING phase, so
    ``observed`` False is reported, not failed.
    """

    waiting_samples = [
        item
        for item in timeline
        if item.get("status") == "WAITING" or item.get("round") == 1
    ]
    waiting_owner = next(
        (
            str(item.get("last_lease_owner") or item.get("lease_owner") or "")
            for item in waiting_samples
            if item.get("last_lease_owner") or item.get("lease_owner")
        ),
        None,
    )
    leased = [item for item in timeline if item.get("status") == "LEASED"]
    second_owner = str(leased[-1].get("lease_owner") or "") if leased else None
    return {
        "observed": bool(waiting_samples),
        "waiting_owner": waiting_owner,
        "second_round_owner": second_owner,
        "reclaimed_by_other_replica": (
            bool(waiting_owner and second_owner) and waiting_owner != second_owner
        ),
        "samples": len(timeline),
    }


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
        time.sleep(SAMPLE_INTERVAL_SECONDS)
    raise CaseError(f"command did not reach SUCCEEDED: {timeline[-5:]}")


def takeover_errors(
    *,
    final: dict,
    survivor: str,
    notification_id: str,
    timing: dict[str, Any],
    remaining_lease: float,
    survivor_state: dict,
    physical_total: int,
    notification_final: dict,
) -> list[str]:
    errors = []
    if final.get("last_lease_owner") != survivor:
        errors.append("survivor did not become the final lease owner")
    details = final.get("result_details", {})
    if details.get("cached") is not True:
        errors.append("survivor did not reuse the shared action ledger")
    if physical_total != 1:
        errors.append(
            f"physical action counters across executors sum to {physical_total}, not 1"
        )
    if details.get("shared_notification_id") != notification_id:
        errors.append("shared ledger identity changed across takeover")
    if timing["takeover_seconds"] > remaining_lease + timing["tolerance_seconds"]:
        errors.append(
            "takeover exceeded remaining lease plus poll/backoff tolerance "
            f"({timing['takeover_seconds']}s > {remaining_lease}s + "
            f"{timing['tolerance_seconds']}s)"
        )
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
    return errors


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
    try:
        database_preflight = database_residuals()
        registry_preflight = registry_residuals()
        kubernetes_preflight = kubernetes_residuals()
        write_json_atomic(case_dir / "database-preflight.json", database_preflight)
        write_json_atomic(case_dir / "registry-preflight.json", registry_preflight)
        write_json_atomic(case_dir / "kubernetes-preflight.json", kubernetes_preflight)
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
        for pod in PODS:
            dataplane("wait", "--for=condition=Ready", f"pod/{pod}", "--timeout=180s")
            wait_file(pod, "/state/ready.json", 60)
        ready = {pod: read_state(pod, "/state/ready.json") for pod in PODS}
        write_json_atomic(case_dir / "executors-ready.json", ready)

        seed = seed_command(run_id)
        write_json_atomic(case_dir / "seed.json", seed)
        first, notification, pre_kill_timeline = wait_first_owner(seed)
        notification_id = str(notification["notification_id"])
        write_json_atomic(case_dir / "first-lease.json", first)
        write_json_atomic(
            case_dir / "pre-kill-timeline.json", {"entries": pre_kill_timeline}
        )
        write_json_atomic(case_dir / "shared-ledger-before-kill.json", notification)
        waiting = waiting_branch(pre_kill_timeline)
        owner = str(first["lease_owner"])
        survivor = next(pod for pod in PODS if pod != owner)
        owner_state = read_state(owner, "/state/executor-state.json")
        write_json_atomic(case_dir / "owner-before-kill.json", owner_state)
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
        post_kill = command_snapshot(str(seed["command_id"]))
        write_json_atomic(case_dir / "post-kill-command.json", post_kill)
        remaining_lease = remaining_lease_seconds(post_kill, killed_at)
        final, timeline = wait_terminal(seed)
        observed_completed_at = datetime.now(timezone.utc)
        timing = takeover_timing(
            final,
            killed_at=killed_at,
            observed_completed_at=observed_completed_at,
        )
        write_json_atomic(case_dir / "command-timeline.json", {"entries": timeline})
        write_json_atomic(case_dir / "final-command.json", final)
        survivor_state = read_state(survivor, "/state/executor-state.json")
        write_json_atomic(case_dir / "survivor-state.json", survivor_state)
        survivor_logs = dataplane("logs", survivor, check=False, timeout=120)
        (case_dir / "survivor.log").write_text(survivor_logs)
        (case_dir / "survivor.log").chmod(0o600)
        notification_final = notification_snapshot(str(seed["deduplication_key"]))
        write_json_atomic(case_dir / "shared-ledger-final.json", notification_final)
        physical_total = physical_action_total(owner_state, survivor_state)
        result = {
            "case_id": CASE_ID,
            "attempt": attempt,
            "verdict": "FAIL",
            "errors": [],
            "production_lease_seconds": production_lease_seconds(),
            "fixture_lease_seconds": LEASE_SECONDS,
            "fixture_poll_seconds": POLL_SECONDS,
            "fixture_claim_backoff_max_seconds": CLAIM_BACKOFF_MAX_SECONDS,
            "winner_sleep_seconds": WINNER_SLEEP_SECONDS,
            "killed_owner": owner,
            "survivor": survivor,
            "kill_requested_at": kill_requested_at.isoformat(),
            "killed_at": killed_at.isoformat(),
            "completed_at": timing["completed_at"],
            "completion_source": timing["completion_source"],
            "remaining_lease_at_kill_seconds": (
                round(remaining_lease, 3) if remaining_lease is not None else None
            ),
            "takeover_seconds": timing["takeover_seconds"],
            "takeover_tolerance_seconds": timing["tolerance_seconds"],
            "waiting_branch": waiting,
            "physical_actions_total": physical_total,
            "first_lease": first,
            "post_kill_command": post_kill,
            "final_command": final,
            "shared_ledger": notification_final,
            "owner_state": owner_state,
            "survivor_state": survivor_state,
        }
        if remaining_lease is None:
            # The kill did not land on a LEASED command, so the LEASED-branch
            # bound cannot be judged; the run is inconclusive, not failed.
            result["verdict"] = "INCONCLUSIVE"
            result["inconclusive_reason"] = (
                "command held no lease at the kill (lease_expires_at is null); "
                "the LEASED takeover bound could not be measured"
            )
        else:
            errors = takeover_errors(
                final=final,
                survivor=survivor,
                notification_id=notification_id,
                timing=timing,
                remaining_lease=remaining_lease,
                survivor_state=survivor_state,
                physical_total=physical_total,
                notification_final=notification_final,
            )
            result["errors"] = errors
            result["verdict"] = "PASS" if not errors else "FAIL"
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
                write_json_atomic(case_dir / "seed-cleanup.json", cleanup)
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
            write_json_atomic(case_dir / "postflight.json", postflight)
            if postflight["database"]["total"] != 0:
                raise CaseError(f"database residuals: {postflight}")
            if postflight["registry"]["count"] != 0:
                raise CaseError(f"registry residuals: {postflight}")
            if postflight["kubernetes"]["count"] != 0:
                raise CaseError(f"Kubernetes residuals: {postflight}")
        except Exception as exc:
            result["postflight_error"] = f"{type(exc).__name__}: {exc}"
            result["verdict"] = "FAIL"
    write_json_atomic(case_dir / f"{CASE_ID}.json", result)
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

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
from typing import Callable

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
STORE_DSN_SNIPPET: str = _registry.STORE_DSN_SNIPPET
executor_identity = _action_capacity.executor_identity
NAMESPACE = _registry.NAMESPACE
control = _registry.control
dataplane = _registry.dataplane
load_registry = _registry.load_registry
register = _registry.register
teardown = _capacity_suite.teardown
upsert_configmap = _capacity_suite.upsert_configmap

SCRIPT = Path(__file__).with_name("probes") / "ha005_probe.py"
CONFIGMAP = "gpu-fault-ha005-probe"
POD = "gpu-fault-ha005-probe"
CASE_ID = "GF-REGIONAL-HA-005"
CONFIRMATION = "HA005_CONTROL_PLANE_ROLLOUT"
INGRESS_DEPLOYMENT = "gpu-fault-api-ha"
ALL_DEPLOYMENTS = (
    "gpu-fault-api-ha",
    "gpu-fault-control-worker",
    "gpu-fault-telemetry-spool-worker",
)
# The probe posts one host-telemetry event every 5s and claims every 2s; the
# baseline is taken once it has produced this many of each rather than after a
# fixed sleep, so a slow first connection does not leave the baseline empty.
BASELINE_EVENT_SAMPLES = 5
BASELINE_CLAIM_SAMPLES = 10
KNOWN_LIMITATIONS = [
    "Each collector outbox remains bounded to 1000 records.",
    "An unwritable or corrupt outbox can still lose telemetry.",
    "Completion Watcher persistence remains a separate boundary.",
]
OUTBOX_NOT_EXERCISED_LIMITATION = (
    "The rollout produced no event failure, so the outbox buffer-and-replay path "
    "was not exercised in this run; its assertions are vacuous here."
)


class CaseError(RuntimeError):
    pass


def log(message: str) -> None:
    stamp = datetime.now(timezone.utc).strftime("%H:%M:%S")
    print(f"[{stamp}] {message}", flush=True)


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
        STORE_DSN_SNIPPET
        + r"""
import json
import os
import psycopg
queries = {
    "objects": (
        "SELECT count(*) FROM gpu_fault_objects "
        "WHERE payload->>'cluster_id' LIKE 'perf-cap-%' "
        "OR key LIKE '%ha005-%'"
    ),
    "links": (
        "SELECT count(*) FROM gpu_fault_links "
        "WHERE key LIKE '%ha005-%' OR value LIKE '%ha005-%'"
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
    return {"count": sum(resources.values()), "resources": resources}


def pod_manifest(image: str, identity: dict[str, object], run_id: str) -> dict:
    return {
        "apiVersion": "v1",
        "kind": "Pod",
        "metadata": {
            "name": POD,
            "namespace": NAMESPACE,
            "labels": {"app": POD},
        },
        "spec": {
            "restartPolicy": "Never",
            "activeDeadlineSeconds": 900,
            "terminationGracePeriodSeconds": 5,
            "serviceAccountName": "gpu-fault-completion-watcher",
            "tolerations": [{"operator": "Exists"}],
            "containers": [
                {
                    "name": "probe",
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
                        {"name": "SSL_CERT_FILE", "value": "/tls/ca.crt"},
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


def wait_file(path: str, timeout_seconds: int) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        output = dataplane(
            "exec",
            POD,
            "--",
            "sh",
            "-c",
            f"if [ -f {shlex.quote(path)} ]; then printf present; fi",
            check=False,
        )
        if output.strip() == "present":
            return
        time.sleep(1)
    raise CaseError(f"probe did not create {path}")


def read_probe() -> dict:
    return json.loads(dataplane("exec", POD, "--", "cat", "/state/stats.json"))


def probe_samples_ready(
    probe: dict,
    *,
    minimum_events: int = BASELINE_EVENT_SAMPLES,
    minimum_claims: int = BASELINE_CLAIM_SAMPLES,
) -> bool:
    counters = probe.get("counters") or {}
    return (
        int(counters.get("event_attempts", 0)) >= minimum_events
        and int(counters.get("claim_attempts", 0)) >= minimum_claims
    )


def wait_probe_samples(
    *,
    minimum_events: int = BASELINE_EVENT_SAMPLES,
    minimum_claims: int = BASELINE_CLAIM_SAMPLES,
    timeout_seconds: int = 180,
    read: Callable[[], dict] = read_probe,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> dict:
    """Poll the probe's counters until it has produced a usable baseline.

    Replaces the fixed ``sleep(30)``: the number of samples is what the baseline
    needs, and a probe that connected late or fast should be judged by its
    counters, not by the wall clock.
    """

    deadline = clock() + timeout_seconds
    last: dict = {}
    while True:
        last = read()
        if probe_samples_ready(
            last, minimum_events=minimum_events, minimum_claims=minimum_claims
        ):
            return last
        if clock() >= deadline:
            raise CaseError(
                f"probe did not reach {minimum_events} event / {minimum_claims} "
                f"claim samples: {last.get('counters')}"
            )
        sleep(2)


def rollout_complete(snapshot: dict, old_uids: set[str]) -> bool:
    """Whether a Deployment rollout has fully replaced its Pods.

    Every declared replica must be Ready, updated to the new template and
    available, the controller must have observed the generation it is
    reporting on, and none of the pre-rollout Pod UIDs may remain. A
    ``ready == replicas`` check alone is satisfied mid-rollout while the old
    Pods still serve.
    """

    replicas = int(snapshot.get("replicas") or 0)
    current_uids = {value["uid"] for _name, value in snapshot.get("pods", [])}
    return (
        int(snapshot.get("ready") or 0) == replicas
        and int(snapshot.get("updated") or 0) == replicas
        and int(snapshot.get("available") or 0) == replicas
        and snapshot.get("observed_generation") == snapshot.get("generation")
        and old_uids.isdisjoint(current_uids)
    )


def continuity_errors(
    final_probe: dict,
    receipts: dict,
    *,
    accepted_ids: list[str],
) -> list[str]:
    """The probe/receipt assertions HA-005 and HA-009 share.

    One place for: attempts minus 202s equals recorded failures; every failure
    was buffered; the outbox drained; no 401/403/500; every accepted
    processor_request_id completed once with an internal 200.
    """

    counters = final_probe.get("counters", {})
    attempts = int(counters.get("event_attempts", 0))
    accepted = int(counters.get("event_accepted", 0))
    failures = int(counters.get("event_failures", 0))
    buffered = int(counters.get("event_buffered", 0))
    errors = []
    if attempts - accepted != failures:
        errors.append("event attempts minus accepted does not equal failures")
    if buffered != failures:
        errors.append("not every event failure was durably buffered")
    if final_probe.get("outbox") != {"records": 0, "replayable": 0}:
        errors.append("probe outbox is not empty after recovery")
    if not accepted_ids:
        errors.append("probe captured no processor request IDs")
    if receipts.get("missing"):
        errors.append("accepted processor request IDs are missing")
    if any(
        item["status"] != "COMPLETED" or item["response_status"] != 200
        for item in receipts.get("requests", [])
    ):
        errors.append("an accepted processor request did not complete with 200")
    if any(
        key in {"http-401", "http-403", "http-500"} and int(value) > 0
        for key, value in final_probe.get("error_types", {}).items()
    ):
        errors.append("probe observed 401, 403 or 500")
    return errors


def outbox_exercised(final_probe: dict) -> bool:
    """True only if at least one event failed, i.e. the outbox path actually ran."""

    return int((final_probe.get("counters") or {}).get("event_failures", 0)) > 0


def deployment_snapshot(name: str = INGRESS_DEPLOYMENT) -> dict:
    value = json.loads(control("get", "deployment", name, "-o", "json"))
    pods = json.loads(
        control(
            "get",
            "pod",
            "-l",
            f"app={name}",
            "-o",
            "json",
        )
    )
    return {
        "name": name,
        "generation": value["metadata"].get("generation"),
        "observed_generation": value.get("status", {}).get("observedGeneration"),
        "replicas": value["spec"].get("replicas", 0),
        "ready": value.get("status", {}).get("readyReplicas", 0),
        "updated": value.get("status", {}).get("updatedReplicas", 0),
        "available": value.get("status", {}).get("availableReplicas", 0),
        "pods": sorted(
            {
                item["metadata"]["name"]: {
                    "uid": item["metadata"]["uid"],
                    "ready": bool(
                        item.get("status", {}).get("containerStatuses")
                        and item["status"]["containerStatuses"][0].get("ready")
                    ),
                    "restarts": int(
                        (item.get("status", {}).get("containerStatuses") or [{}])[
                            0
                        ].get("restartCount", 0)
                    ),
                }
                for item in pods.get("items", [])
            }.items()
        ),
    }


def processor_receipts(request_ids: list[str]) -> dict:
    if not request_ids:
        return {"requests": [], "missing": []}
    script = r"""
import json
import sys
from gpu_fault.app import ApplicationContext
store = ApplicationContext.from_environment().store
requests = []
missing = []
for request_id in sys.argv[1:]:
    try:
        item = store.get_processor_request(request_id)
    except Exception:
        missing.append(request_id)
        continue
    requests.append({
        "request_id": item.request_id,
        "status": item.status.value,
        "response_status": item.response_status,
        "path": item.path,
        "retry_count": item.retry_count,
    })
print(json.dumps({"requests": requests, "missing": missing}, sort_keys=True))
"""
    return cpu_python(script, *request_ids)


def wait_receipts(request_ids: list[str], timeout_seconds: int = 180) -> dict:
    deadline = time.monotonic() + timeout_seconds
    last = {}
    while time.monotonic() < deadline:
        last = processor_receipts(request_ids)
        if not last["missing"] and all(
            item["status"] == "COMPLETED" and item["response_status"] == 200
            for item in last["requests"]
        ):
            return last
        time.sleep(2)
    raise CaseError(f"processor receipts did not converge: {last}")


def rollout_targets(all_deployments: bool) -> list[str]:
    """Which Deployments this run restarts; a replicas=0 role is skipped, not rolled."""

    if not all_deployments:
        return [INGRESS_DEPLOYMENT]
    targets = []
    for name in ALL_DEPLOYMENTS:
        if int(deployment_snapshot(name).get("replicas") or 0) > 0:
            targets.append(name)
    return targets


def _restart_and_observe(
    case_dir: Path,
    name: str,
    *,
    timeout_seconds: int = 600,
) -> dict:
    deployment_before = deployment_snapshot(name)
    old_uids = {value["uid"] for _name, value in deployment_before["pods"]}
    requested_at = datetime.now(timezone.utc)
    log(f"starting {name} rollout restart")
    control("rollout", "restart", f"deployment/{name}")
    timeline = []
    next_log = 0.0
    started = time.monotonic()
    while time.monotonic() - started < timeout_seconds:
        elapsed = time.monotonic() - started
        snapshot = deployment_snapshot(name)
        probe = read_probe()
        timeline.append(
            {
                "elapsed_seconds": round(elapsed, 3),
                "deployment": snapshot,
                "probe": {
                    "counters": probe.get("counters", {}),
                    "statuses": probe.get("statuses", {}),
                    "error_types": probe.get("error_types", {}),
                    "outbox": probe.get("outbox", {}),
                },
            }
        )
        forbidden = {
            key: value
            for key, value in probe.get("error_types", {}).items()
            if key in {"http-401", "http-403", "http-500"} and int(value) > 0
        }
        if forbidden:
            raise CaseError(f"forbidden probe responses: {forbidden}")
        complete = rollout_complete(snapshot, old_uids)
        if elapsed >= next_log:
            log(
                f"{name} rollout t={elapsed:.0f}s ready={snapshot['ready']} "
                f"updated={snapshot['updated']} outbox={probe['outbox']}"
            )
            next_log = elapsed + 15
        if complete:
            break
        time.sleep(2)
    else:
        raise CaseError(f"{name} rollout did not complete")
    rollout_seconds = time.monotonic() - started
    write_json_atomic(case_dir / f"rollout-timeline-{name}.json", {"entries": timeline})
    return {
        "name": name,
        "requested_at": requested_at.isoformat(),
        "rollout_seconds": round(rollout_seconds, 3),
        "deployment_before": deployment_before,
        "deployment_after": deployment_snapshot(name),
    }


def _run_rollout_case(
    case_dir: Path,
    run_id: str,
    attempt: int,
    state: dict,
    *,
    all_deployments: bool = False,
) -> dict:
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
    dataplane("delete", "pod", POD, "--ignore-not-found", check=False)
    dataplane(
        "apply",
        "-f",
        "-",
        stdin=json.dumps(pod_manifest(image, identity, run_id)).encode(),
    )
    dataplane("wait", "--for=condition=Ready", f"pod/{POD}", "--timeout=180s")
    state["probe_created"] = True
    wait_file("/state/ready.json", 60)
    wait_file("/state/stats.json", 60)
    probe_baseline = wait_probe_samples()
    write_json_atomic(case_dir / "probe-baseline.json", probe_baseline)
    targets = rollout_targets(all_deployments)
    skipped = [
        name for name in ALL_DEPLOYMENTS if all_deployments and name not in targets
    ]
    rollouts = [_restart_and_observe(case_dir, name) for name in targets]

    attempts_at_recovery = int(read_probe()["counters"].get("event_attempts", 0))
    deadline = time.monotonic() + 300
    post_recovery_timeline = []
    while time.monotonic() < deadline:
        probe = read_probe()
        post_recovery_timeline.append(
            {
                "observed_at": datetime.now(timezone.utc).isoformat(),
                "counters": probe.get("counters", {}),
                "outbox": probe.get("outbox", {}),
            }
        )
        if (
            int(probe.get("outbox", {}).get("replayable", 0)) == 0
            and int(probe.get("outbox", {}).get("records", 0)) == 0
            and int(probe["counters"].get("event_attempts", 0))
            >= attempts_at_recovery + 10
        ):
            break
        time.sleep(2)
    else:
        raise CaseError("probe outbox did not converge after rollout")
    write_json_atomic(
        case_dir / "post-recovery-timeline.json", {"entries": post_recovery_timeline}
    )
    return _rollout_result(
        case_dir,
        attempt,
        rollouts,
        probe_baseline,
        skipped_deployments=skipped,
    )


def _rollout_result(
    case_dir: Path,
    attempt: int,
    rollouts: list[dict],
    probe_baseline: dict,
    *,
    skipped_deployments: list[str] | None = None,
) -> dict:
    final_probe = read_probe()
    write_json_atomic(case_dir / "probe-final.json", final_probe)
    probe_logs = dataplane("logs", POD, check=False, timeout=120)
    (case_dir / "probe.log").write_text(probe_logs)
    (case_dir / "probe.log").chmod(0o600)
    dataplane("exec", POD, "--", "touch", "/state/stop", check=False)
    accepted_ids = sorted(set(final_probe.get("accepted_request_ids", [])))
    receipts = wait_receipts(accepted_ids)
    write_json_atomic(case_dir / "processor-receipts.json", receipts)
    errors = continuity_errors(final_probe, receipts, accepted_ids=accepted_ids)
    for rollout in rollouts:
        after = deployment_snapshot(str(rollout["name"]))
        rollout["deployment_after"] = after
        if int(after["ready"]) != int(after["replicas"]):
            errors.append(
                f"{rollout['name']} did not return to its declared Ready replicas"
            )
        if any(value["restarts"] for _name, value in after["pods"]):
            errors.append(f"{rollout['name']} replacement Pod has a container restart")
    exercised = outbox_exercised(final_probe)
    limitations = list(KNOWN_LIMITATIONS)
    if not exercised:
        limitations.append(OUTBOX_NOT_EXERCISED_LIMITATION)
    primary = rollouts[0] if rollouts else {}
    return {
        "case_id": CASE_ID,
        "attempt": attempt,
        "verdict": "PASS" if not errors else "FAIL",
        "errors": errors,
        "requested_at": primary.get("requested_at"),
        "rollout_seconds": primary.get("rollout_seconds"),
        "deployment_before": primary.get("deployment_before"),
        "deployment_after": primary.get("deployment_after"),
        "rollouts": rollouts,
        "skipped_deployments": [
            {"name": name, "reason": "role not enabled (replicas=0)"}
            for name in (skipped_deployments or [])
        ],
        "probe_baseline": probe_baseline,
        "probe_final": final_probe,
        "processor_receipts": receipts,
        "outbox_exercised": exercised,
        "known_limitations": limitations,
        "validation_limitations": limitations,
    }


def _cleanup_rollout_case(
    case_dir: Path,
    run_id: str,
    result: dict,
    probe_created: bool,
) -> None:
    if probe_created:
        log_path = case_dir / "probe.log"
        if not log_path.is_file():
            probe_logs = dataplane("logs", POD, check=False, timeout=120)
            log_path.write_text(probe_logs)
            log_path.chmod(0o600)
        dataplane("exec", POD, "--", "touch", "/state/stop", check=False)
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
        result["cleanup_error"] = f"{type(exc).__name__}: {exc}"
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


def run_case(
    run_dir: Path,
    attempt: int,
    maintenance_window_end: datetime,
    *,
    all_deployments: bool = False,
) -> int:
    if datetime.now(timezone.utc) >= maintenance_window_end:
        raise CaseError("approved maintenance window has ended")
    case_dir = run_dir / "cases" / CASE_ID
    case_dir.mkdir(parents=True, exist_ok=True)
    run_id = f"ha005-{run_dir.name.rsplit('-', 1)[-1].lower()}-a{attempt}"
    result: dict = {"case_id": CASE_ID, "attempt": attempt, "verdict": "FAIL"}
    state = {"probe_created": False}
    try:
        result = _run_rollout_case(
            case_dir, run_id, attempt, state, all_deployments=all_deployments
        )
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        _cleanup_rollout_case(
            case_dir,
            run_id,
            result,
            bool(state["probe_created"]),
        )
    write_json_atomic(case_dir / f"{CASE_ID}.json", result)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["verdict"] == "PASS" else 1


def main() -> int:
    install_site_profile()
    parser = argparse.ArgumentParser()
    add_live_arguments(parser, confirmation=CONFIRMATION)
    parser.add_argument(
        "--all-deployments",
        action="store_true",
        help="rollout restart every enabled control-plane Deployment, not only ingress",
    )
    args = parser.parse_args()
    os.umask(0o077)
    mutation = (
        "rollout restart every enabled control-plane Deployment "
        f"({', '.join(ALL_DEPLOYMENTS)}; replicas=0 roles skipped)"
        if args.all_deployments
        else f"rollout restart deployment/{INGRESS_DEPLOYMENT}"
    )
    if not args.execute:
        plan = build_plan(
            run_dir=args.run_dir,
            case_id=CASE_ID,
            attempt=args.attempt,
            confirmation=CONFIRMATION,
            details={
                "risk": "live-service-action",
                "synthetic_cluster_id": "perf-cap-000",
                "mutation": mutation,
                "all_deployments": bool(args.all_deployments),
                "probe": "continuous claim and benign host telemetry",
                "rollback": [
                    "Deployment rolling strategy and PDB",
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
    return run_case(
        args.run_dir, args.attempt, deadline, all_deployments=bool(args.all_deployments)
    )


if __name__ == "__main__":
    raise SystemExit(main())

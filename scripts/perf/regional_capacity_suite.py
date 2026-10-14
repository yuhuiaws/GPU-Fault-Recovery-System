#!/usr/bin/env python3
"""One-click regional capacity harness.

Wraps every manual step that used to surround the load generators:
audit cluster registration, load-generator token secret, payload template
ConfigMap, job rendering, durable log collection, control-plane metric and
Aurora sampling, database purge and deregistration.

The load generators run in the GPU data-plane cluster (it has the spare
CPU). Registration updates the bootstrap Secret and publishes one durable
registry revision, then waits for every CPU process to ACK it.

Examples
--------
    # full 32 cluster synchronized burst, artifacts under artifacts/perf/
    scripts/perf/regional_capacity_suite.py all --case burst --clusters 32

    # keep the registration between runs
    scripts/perf/regional_capacity_suite.py register --clusters 50 \
        --suite-id run-a --keep-registration
    scripts/perf/regional_capacity_suite.py run --case burst --clusters 50 \
        --suite-id run-a --run-dir <registration-artifacts>
    scripts/perf/regional_capacity_suite.py teardown \
        --suite-id run-a --run-dir <registration-artifacts>
"""

from __future__ import annotations

import argparse
import base64
import gzip
import json
import os
import re
import secrets
import sys
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING or __package__:
    from . import regional_capacity_data as capacity_data
    from . import regional_capacity_database as capacity_database
    from . import regional_capacity_placement as capacity_placement
    from . import regional_capacity_recovery as capacity_recovery
    from . import regional_capacity_registry as capacity_registry
    from . import regional_capacity_results as capacity_results
    from . import regional_capacity_settle as capacity_settle
    from . import regional_capacity_timeseries as capacity_timeseries
    from .regional_capacity_cleanup import (
        AUDIT_PURGE_STATEMENTS as AUDIT_PURGE_STATEMENTS,
    )
    from .regional_capacity_job import build_job as _build_job
    from .regional_capacity_resources import (
        RunResources,
        ensure_load_service_account,
        run_manifest,
    )
else:
    import regional_capacity_data as capacity_data
    import regional_capacity_database as capacity_database
    import regional_capacity_placement as capacity_placement
    import regional_capacity_recovery as capacity_recovery
    import regional_capacity_registry as capacity_registry
    import regional_capacity_results as capacity_results
    import regional_capacity_settle as capacity_settle
    import regional_capacity_timeseries as capacity_timeseries
    from regional_capacity_cleanup import (
        AUDIT_PURGE_STATEMENTS as AUDIT_PURGE_STATEMENTS,
    )
    from regional_capacity_job import build_job as _build_job
    from regional_capacity_resources import (
        RunResources,
        ensure_load_service_account,
        run_manifest,
    )

AWS_REGION = capacity_registry.AWS_REGION
CONNECTION_SECRET = capacity_registry.CONNECTION_SECRET
CONTROL_NAMESPACE = capacity_registry.CONTROL_NAMESPACE
DATAPLANE_CONTEXT = capacity_registry.DATAPLANE_CONTEXT
NAMESPACE = capacity_registry.NAMESPACE
PERF_CLUSTER_PREFIX = capacity_registry.PERF_CLUSTER_PREFIX
REGISTRY_SECRET = capacity_registry.REGISTRY_SECRET
TOKEN_SECRET = capacity_registry.TOKEN_SECRET
control = capacity_registry.control
control_pods = capacity_registry.control_pods
dataplane = capacity_registry.dataplane
deregister = capacity_registry.deregister
register = capacity_registry.register
registered_cluster_ids = capacity_registry.registered_cluster_ids
run = capacity_registry.run
validate_registered_synthetic_run = capacity_registry.validate_registered_synthetic_run
validate_registry_target = capacity_registry.validate_registry_target
aggregate = capacity_results.aggregate
artifact_dir = capacity_results.artifact_dir
collect_pod_json_logs = capacity_results.collect_pod_json_logs
drain_targets = capacity_results.drain_targets
move_to_aborted = capacity_results.move_to_aborted
write_status = capacity_results.write_status
_aurora_window = capacity_database.aurora_window
_postgres_counters = capacity_database.postgres_counters
store_bloat_section = capacity_database.store_bloat_section
store_layout_section = capacity_database.store_layout_section
_processor_priority_latency = capacity_database.processor_priority_latency
CONTROL_POD_PREFIXES = capacity_placement.CONTROL_POD_PREFIXES
start_placement = capacity_placement.start_placement
placement_section = capacity_placement.placement_section

REPO_ROOT = Path(__file__).resolve().parents[2]
PERF_DIR = REPO_ROOT / "scripts" / "perf"
DEFAULT_ARTIFACT_ROOT = REPO_ROOT / "artifacts" / "perf"
INGRESS_REQUEST_RECYCLE_FLAGS = (
    "--limit-max-requests",
    "--limit-max-requests-jitter",
)

# The run's writer is resolved from this cluster (``regional_capacity_placement``).
AURORA_CLUSTER_ID = os.getenv("GPU_FAULT_AURORA_CLUSTER_ID", "")
SCRIPT_CONFIGMAP = "gpu-fault-perf-suite-scripts"
TEMPLATE_CONFIGMAP = "gpu-fault-perf-suite-templates"
START_GATE_CONFIGMAP = "gpu-fault-perf-start-gate"
START_GATE_ROLE = "gpu-fault-perf-start-gate-reader"
START_GATE_ROLE_BINDING = "gpu-fault-perf-start-gate-reader"
LOAD_SERVICE_ACCOUNT = "gpu-fault-completion-watcher"
CASES = {
    "burst": {
        "script": "benchmark_synchronized_burst.py",
        "job": "gpu-fault-perf-suite-burst",
        "deadline": 1800,
    },
    "mixed": {
        "script": "benchmark_mixed_control_plane.py",
        "job": "gpu-fault-perf-suite-mixed",
        "deadline": 1800,
    },
    "multicluster": {
        "script": "benchmark_multicluster_capacity.py",
        "job": "gpu-fault-perf-suite-multicluster",
        "deadline": 1800,
    },
}

SUPPORT_SCRIPTS = ("benchmark_mixed_control_plane.py",)


def log(message: str) -> None:
    stamp = datetime.now(timezone.utc).strftime("%H:%M:%S")
    print(f"[{stamp}] {message}", flush=True)


def upsert_configmap(
    name: str,
    *,
    resources: RunResources,
    text: dict[str, str] | None = None,
    binary: dict[str, bytes] | None = None,
) -> None:
    manifest: dict[str, Any] = {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": {"name": name, "namespace": NAMESPACE},
    }
    if text:
        manifest["data"] = text
    if binary:
        manifest["binaryData"] = {
            key: base64.b64encode(value).decode() for key, value in binary.items()
        }
    resources.create(manifest)


def prepare_start_gate(*, resources: RunResources) -> None:
    ensure_load_service_account(
        resources, name=LOAD_SERVICE_ACCOUNT, namespace=NAMESPACE
    )
    upsert_configmap(
        START_GATE_CONFIGMAP,
        resources=resources,
        text={"start_epoch": ""},
    )
    access: list[dict[str, Any]] = [
        {
            "apiVersion": "rbac.authorization.k8s.io/v1",
            "kind": "Role",
            "metadata": {
                "name": START_GATE_ROLE,
                "namespace": NAMESPACE,
            },
            "rules": [
                {
                    "apiGroups": [""],
                    "resources": ["configmaps"],
                    "resourceNames": [START_GATE_CONFIGMAP],
                    "verbs": ["get"],
                }
            ],
        },
        {
            "apiVersion": "rbac.authorization.k8s.io/v1",
            "kind": "RoleBinding",
            "metadata": {
                "name": START_GATE_ROLE_BINDING,
                "namespace": NAMESPACE,
            },
            "subjects": [
                {
                    "kind": "ServiceAccount",
                    "name": LOAD_SERVICE_ACCOUNT,
                    "namespace": NAMESPACE,
                }
            ],
            "roleRef": {
                "apiGroup": "rbac.authorization.k8s.io",
                "kind": "Role",
                "name": START_GATE_ROLE,
            },
        },
    ]
    for item in access:
        resources.create(item)


def wait_for_load_pods(
    job: str,
    expected: int,
    *,
    timeout_seconds: int,
) -> None:
    started = time.time()
    last_reported = None
    while time.time() - started < timeout_seconds:
        document = json.loads(
            dataplane(
                "get",
                "pods",
                "-l",
                f"job-name={job}",
                "-o",
                "json",
                check=False,
            )
            or '{"items":[]}'
        )
        running = 0
        ready = 0
        for item in document.get("items", []):
            if item.get("status", {}).get("phase") == "Running":
                running += 1
            conditions = item.get("status", {}).get("conditions", [])
            if any(
                condition.get("type") == "Ready" and condition.get("status") == "True"
                for condition in conditions
            ):
                ready += 1
        state = (
            len(document.get("items", [])),
            running,
            ready,
        )
        if state != last_reported:
            log(
                "load pods: "
                f"created={state[0]} running={running} "
                f"ready={ready}/{expected}"
            )
            last_reported = state
        if ready == expected:
            return
        time.sleep(2)
    raise TimeoutError(
        f"only {last_reported} load pods became ready within {timeout_seconds}s"
    )


def release_start_gate(start_epoch: float, *, resources: RunResources) -> None:
    resources.update_data(
        START_GATE_CONFIGMAP,
        {"start_epoch": f"{start_epoch:.6f}"},
    )


def publish_fixtures(case: str, *, resources: RunResources) -> None:
    script_name = CASES[case]["script"]
    payloads = {}
    for name in {script_name, *SUPPORT_SCRIPTS}:
        payloads[name] = (PERF_DIR / name).read_text()
    upsert_configmap(SCRIPT_CONFIGMAP, resources=resources, text=payloads)
    templates = json.loads((PERF_DIR / "payload-templates.json").read_text())
    upsert_configmap(
        TEMPLATE_CONFIGMAP,
        resources=resources,
        binary={
            "templates.json.gz": gzip.compress(
                json.dumps(templates).encode(),
                compresslevel=9,
            )
        },
    )


def build_job(
    case: str,
    *,
    clusters: int,
    nodes_per_cluster: int,
    xid_total: int,
    sxid_total: int,
    gpu_evidence_total: int,
    host_evidence_total: int,
    training_heartbeat_total: int,
    workload_observation_total: int,
    correlate_attempt_faults: bool,
    start_epoch: float,
    workers: int,
    duration_seconds: int,
    cpu_request: str,
    cpu_limit: str,
    include_telemetry: bool,
    prewarm_connections: bool,
) -> dict:
    return _build_job(
        case,
        spec=CASES[case],
        namespace=NAMESPACE,
        token_secret=TOKEN_SECRET,
        script_configmap=SCRIPT_CONFIGMAP,
        template_configmap=TEMPLATE_CONFIGMAP,
        start_gate_configmap=START_GATE_CONFIGMAP,
        clusters=clusters,
        nodes_per_cluster=nodes_per_cluster,
        xid_total=xid_total,
        sxid_total=sxid_total,
        gpu_evidence_total=gpu_evidence_total,
        host_evidence_total=host_evidence_total,
        training_heartbeat_total=training_heartbeat_total,
        workload_observation_total=workload_observation_total,
        correlate_attempt_faults=correlate_attempt_faults,
        start_epoch=start_epoch,
        workers=workers,
        duration_seconds=duration_seconds,
        cpu_request=cpu_request,
        cpu_limit=cpu_limit,
        include_telemetry=include_telemetry,
        prewarm_connections=prewarm_connections,
        connection_secret=CONNECTION_SECRET,
    )


# --------------------------------------------------------------------------
# control-plane observation
# --------------------------------------------------------------------------


# Anything matched here lands in the artifacts; anything else is
# dropped at scrape time. Two rounds of tuning were spent guessing at
# numbers this filter was already hiding (rejection counters, in-flight
# gauges, the admission batch), so keep it wide enough to answer "where
# did the request budget go" without another run.
METRIC_PATTERN = re.compile(
    r"pool_checkout|store_io_admission|lane_wait|queue_depth|"
    r"admission_rejections|event_loop_lag|oldest|coalesced|"
    r"queue_oldest|processor_requests|lane_rows|"
    r"admission_batch|ingress_lane|request_decode|ingress_backpressure|"
    r"rejections_total|in_flight|processor_claim|store_io_|"
    r"telemetry_spool|queue_bypass|processor_notification|"
    r"processor_consumer|request_processing|fault_pressure"
)


def control_pod_runtime_snapshot() -> dict[str, dict]:
    document = json.loads(control("get", "pod", "-o", "json"))
    snapshot: dict[str, dict] = {}
    for pod in document.get("items") or []:
        metadata = pod.get("metadata") or {}
        name = str(metadata.get("name") or "")
        if not name.startswith(CONTROL_POD_PREFIXES):
            continue
        status = pod.get("status") or {}
        ready = any(
            item.get("type") == "Ready" and item.get("status") == "True"
            for item in status.get("conditions") or []
        )
        containers = {}
        restart_count = 0
        for item in status.get("containerStatuses") or []:
            container_name = str(item.get("name") or "")
            restarts = int(item.get("restartCount") or 0)
            restart_count += restarts
            running = (item.get("state") or {}).get("running") or {}
            terminated = (item.get("lastState") or {}).get("terminated") or {}
            containers[container_name] = {
                "ready": bool(item.get("ready")),
                "restart_count": restarts,
                "running_started_at": running.get("startedAt"),
                "last_termination": (
                    {
                        "reason": terminated.get("reason"),
                        "exit_code": terminated.get("exitCode"),
                        "started_at": terminated.get("startedAt"),
                        "finished_at": terminated.get("finishedAt"),
                    }
                    if terminated
                    else None
                ),
            }
        snapshot[name] = {
            "role": (metadata.get("labels") or {}).get("app"),
            "phase": status.get("phase"),
            "ready": ready,
            "restart_count": restart_count,
            "containers": dict(sorted(containers.items())),
        }
    return dict(sorted(snapshot.items()))


def control_pod_lifecycle(
    before: dict[str, dict],
    after: dict[str, dict],
) -> dict[str, object]:
    before_names = set(before)
    after_names = set(after)
    restarted_pods: dict[str, int] = {}
    counter_regressions: dict[str, dict[str, int]] = {}
    for name in sorted(before_names & after_names):
        before_count = int(before[name].get("restart_count", 0))
        after_count = int(after[name].get("restart_count", 0))
        if after_count > before_count:
            restarted_pods[name] = after_count - before_count
        elif after_count < before_count:
            counter_regressions[name] = {
                "before": before_count,
                "after": after_count,
            }
    return {
        "restart_count_delta": sum(restarted_pods.values()),
        "restarted_pods": restarted_pods,
        "missing_pods": sorted(before_names - after_names),
        "added_pods": sorted(after_names - before_names),
        "counter_regressions": counter_regressions,
        "not_ready_after": sorted(
            name for name, item in after.items() if not item.get("ready")
        ),
    }


def ingress_process_model_preflight() -> dict[str, object]:
    deployment = json.loads(
        control(
            "get",
            "deployment",
            "gpu-fault-api-ha",
            "-o",
            "json",
        )
    )
    containers = (
        deployment.get("spec", {})
        .get("template", {})
        .get("spec", {})
        .get("containers", [])
    )
    api = next(
        (item for item in containers if item.get("name") == "api"),
        None,
    )
    if api is None:
        raise RuntimeError("live ingress Deployment has no api container")
    args = api.get("args") or []
    if len(args) != 1 or not isinstance(args[0], str):
        raise RuntimeError("live ingress command is not a single shell argument")
    command = args[0]
    forbidden = [flag for flag in INGRESS_REQUEST_RECYCLE_FLAGS if flag in command]
    return {
        "deployment": "gpu-fault-api-ha",
        "uvicorn_workers_4": "--workers 4" in command,
        "limit_concurrency_4096": "--limit-concurrency 4096" in command,
        "request_count_recycling_flags": forbidden,
        "valid": not forbidden,
    }


def scrape_metrics(pods: list[str]) -> dict[str, str]:
    snapshot = {}
    for pod in pods:
        port = capacity_timeseries.metrics_port(pod)
        try:
            body = control(
                "exec",
                pod,
                "--",
                "python3",
                "-c",
                (
                    "import urllib.request;"
                    "print(urllib.request.urlopen("
                    f"'http://127.0.0.1:{port}/metrics',timeout=15)"
                    ".read().decode())"
                ),
                timeout=90,
            )
        except Exception as exc:  # noqa: BLE001 - best effort sampling
            snapshot[pod] = f"# scrape failed: {exc}"
            continue
        snapshot[pod] = "\n".join(
            line
            for line in body.splitlines()
            if line and not line.startswith("#") and METRIC_PATTERN.search(line)
        )
    return snapshot


def scrape_cgroup(pods: list[str]) -> dict[str, dict]:
    """Read each pod's CPU accounting counters.

    `kubectl top` reports usage, which never says whether the
    container wanted more. nr_throttled/throttled_usec do: if they
    stay flat across a burst the CPU limit did not bind, and raising
    it (or promoting the pod to Guaranteed) would buy nothing.
    """
    snapshot = {}
    for pod in pods:
        try:
            body = control(
                "exec",
                pod,
                "--",
                "sh",
                "-c",
                "cat /sys/fs/cgroup/cpu.stat; echo ---; cat /sys/fs/cgroup/cpu.max",
                timeout=60,
            )
        except Exception as exc:  # noqa: BLE001 - best effort
            snapshot[pod] = {"error": str(exc)}
            continue
        stat, _, quota = body.partition("---")
        parsed: dict[str, str] = {}
        for line in stat.splitlines():
            fields = line.split()
            if len(fields) == 2:
                parsed[fields[0]] = fields[1]
        parsed["cpu.max"] = quota.strip()
        snapshot[pod] = parsed
    return snapshot


class TopSampler(threading.Thread):
    def __init__(self, path: Path, interval: float = 10.0) -> None:
        super().__init__(daemon=True)
        self.path = path
        self.interval = interval
        # Not _stop: threading.Thread._stop() is an internal
        # method join() calls, and shadowing it with an Event
        # makes join() raise instead of waiting.
        self._done = threading.Event()

    def run(self) -> None:
        with self.path.open("w") as handle:
            while not self._done.is_set():
                sample = {"epoch": time.time(), "pods": []}
                try:
                    raw = control(
                        "top",
                        "pod",
                        "--no-headers",
                        check=False,
                        timeout=60,
                    )
                    for line in raw.splitlines():
                        parts = line.split()
                        if len(parts) >= 3 and parts[0].startswith("gpu-fault-"):
                            sample["pods"].append(
                                {
                                    "name": parts[0],
                                    "cpu": parts[1],
                                    "memory": parts[2],
                                }
                            )
                except Exception as exc:  # noqa: BLE001
                    sample["error"] = str(exc)
                handle.write(json.dumps(sample) + "\n")
                handle.flush()
                self._done.wait(self.interval)

    def stop(self) -> None:
        self._done.set()


def aurora_window(start: float, end: float, *, writer: dict) -> dict:
    return _aurora_window(
        run,
        aurora_instance=writer["writer_instance_id"],
        aurora_cluster_id=writer["cluster_id"],
        aws_region=AWS_REGION,
        start=start,
        end=end,
    )


# --------------------------------------------------------------------------
# execution + collection
# --------------------------------------------------------------------------


def collect_logs(job: str, target: Path) -> list[dict]:
    raw = dataplane(
        "get",
        "pod",
        "-l",
        f"batch.kubernetes.io/job-name={job}",
        "-o",
        "jsonpath="
        "{range .items[*]}{.metadata.name} "
        "{.metadata.annotations."
        "batch\\.kubernetes\\.io/job-completion-index}"
        "{'\\n'}{end}",
    )
    entries = []
    for line in raw.splitlines():
        parts = line.split()
        if not parts:
            continue
        pod = parts[0]
        index = parts[1] if len(parts) > 1 else "?"
        entries.append((index, pod))
    return collect_pod_json_logs(
        entries,
        target,
        fetch=lambda pod: dataplane("logs", pod, check=False, timeout=180),
        on_decode_error=lambda pod: log(f"pod {pod} produced non-JSON output"),
    )


def wait_for_job(job: str, deadline_seconds: int) -> str:
    started = time.time()
    while time.time() - started < deadline_seconds:
        raw = dataplane(
            "get",
            "job",
            job,
            "-o",
            "jsonpath={.status.succeeded}|{.status.failed}|{.spec.completions}",
            check=False,
        )
        succeeded, failed, completions = (raw.split("|") + ["", "", ""])[:3]
        if succeeded and completions and succeeded == completions:
            return "Complete"
        if failed and failed not in {"", "0"}:
            return f"Failed(failed={failed})"
        time.sleep(10)
    return "Timeout"


def queue_drain(pods: list[str], **options: Any) -> dict:
    """``regional_capacity_timeseries.queue_drain`` over this suite's scrape."""
    return capacity_timeseries.queue_drain(pods, scrape=scrape_metrics, **options)


def start_processor_timeseries(pods: list[str]) -> Any:
    return capacity_timeseries.ProcessorTimeseries(pods).start()


def collect_after_load(
    pods: list[str], artifacts: Path, job: str, *, targets: tuple[float, float]
) -> tuple[dict, dict, list[dict], dict]:
    """The drain probe from the load stop, beside the after-load collection."""
    drain = capacity_timeseries.Background(
        queue_drain,
        pods,
        target_queue_depth=targets[0],
        target_spool_depth=targets[1],
    )
    drain.start()
    (artifacts / "metrics-after.json").write_text(
        json.dumps(scrape_metrics(pods), indent=1) + "\n"
    )
    cgroup_after = scrape_cgroup(pods)
    (artifacts / "cgroup-after.json").write_text(
        json.dumps(cgroup_after, indent=1) + "\n"
    )
    pg_after = postgres_counters()
    (artifacts / "postgres-after.json").write_text(
        json.dumps(pg_after, indent=1) + "\n"
    )
    documents = collect_logs(job, artifacts / "pods")
    collected_at = time.time()
    reference = capacity_timeseries.drain_reference(drain.result(), collected_at)
    return cgroup_after, pg_after, documents, reference


def release_id() -> str:
    config_map = control(
        "get",
        "deploy",
        "gpu-fault-api-ha",
        "-o",
        "jsonpath={.spec.template.spec.volumes[?(@.name=='artifact')].configMap.name}",
        check=False,
    ).strip()
    if config_map:
        return config_map.rsplit("-", 1)[-1]
    # Not the Deployment env: api-ha never carries GPU_FAULT_RELEASE_ID, so a
    # jsonpath over its container env read "" silently.
    return capacity_registry.release_state_release_id()


def release_identity() -> dict[str, str]:
    deployment = json.loads(
        control(
            "get",
            "deploy",
            "gpu-fault-api-ha",
            "-o",
            "json",
        )
    )
    template = deployment["spec"]["template"]
    annotations = template.get("metadata", {}).get("annotations") or {}
    config_map = next(
        (
            volume["configMap"]["name"]
            for volume in template["spec"].get("volumes", [])
            if volume.get("name") == "artifact"
            and volume.get("configMap", {}).get("name")
        ),
        "",
    )
    artifact_sha = str(
        annotations.get("gpu-fault.io/artifact-sha256")
        or annotations.get("gpu-fault.io/control-plane-wheel-sha256")
        or ""
    )
    module = ""
    pods = [pod for pod in control_pods() if "api-ha" in pod]
    if pods:
        module = control(
            "exec",
            pods[0],
            "--",
            "python3",
            "-c",
            "from gpu_fault import module_digest; print(module_digest())",
            check=False,
        ).strip()
    return {
        "release_id": config_map.rsplit("-", 1)[-1] if config_map else release_id(),
        "wheel_configmap": config_map,
        "wheel_sha256": artifact_sha,
        "module_digest": module,
    }


def postgres_counters() -> dict:
    pods = [pod for pod in control_pods() if "api-ha" in pod]
    return _postgres_counters(control, pods)


def processor_priority_latency(since_epoch: float = 0.0) -> dict:
    """The run's requests: created since ``since_epoch`` (the load Job's
    creation), on the synthetic clusters."""

    pods = [pod for pod in control_pods() if "api-ha" in pod]
    return _processor_priority_latency(
        control,
        pods,
        cluster_prefix=PERF_CLUSTER_PREFIX,
        since_epoch=since_epoch,
    )


def execute_case(
    case: str,
    *,
    clusters: int,
    nodes_per_cluster: int,
    xid_total: int,
    sxid_total: int,
    gpu_evidence_total: int,
    host_evidence_total: int,
    training_heartbeat_total: int,
    workload_observation_total: int,
    correlate_attempt_faults: bool,
    workers: int,
    duration_seconds: int,
    lead_seconds: int,
    cpu_request: str,
    cpu_limit: str,
    artifacts: Path,
    label: str,
    include_telemetry: bool,
    prewarm_connections: bool,
    resources: RunResources,
    placement: dict,
) -> dict:
    spec = CASES[case]
    job = spec["job"]
    publish_fixtures(case, resources=resources)
    pods = control_pods()
    log(f"observing {len(pods)} control-plane pods")
    metrics_before = scrape_metrics(pods)
    (artifacts / "metrics-before.json").write_text(
        json.dumps(metrics_before, indent=1) + "\n"
    )
    target_queue_depth, target_spool_depth = drain_targets(metrics_before)
    cgroup_before = scrape_cgroup(pods)
    (artifacts / "cgroup-before.json").write_text(
        json.dumps(cgroup_before, indent=1) + "\n"
    )
    pg_before = postgres_counters()
    (artifacts / "postgres-before.json").write_text(
        json.dumps(pg_before, indent=1) + "\n"
    )
    gated_start = case == "burst"
    start_epoch = 0.0 if gated_start else time.time() + lead_seconds
    if gated_start:
        prepare_start_gate(resources=resources)
    manifest = build_job(
        case,
        clusters=clusters,
        nodes_per_cluster=nodes_per_cluster,
        xid_total=xid_total,
        sxid_total=sxid_total,
        gpu_evidence_total=gpu_evidence_total,
        host_evidence_total=host_evidence_total,
        training_heartbeat_total=training_heartbeat_total,
        workload_observation_total=workload_observation_total,
        correlate_attempt_faults=correlate_attempt_faults,
        start_epoch=start_epoch,
        workers=workers,
        duration_seconds=duration_seconds,
        cpu_request=cpu_request,
        cpu_limit=cpu_limit,
        include_telemetry=include_telemetry,
        prewarm_connections=prewarm_connections,
    )
    manifest = run_manifest(manifest, resources.run_id)
    (artifacts / "job.json").write_text(json.dumps(manifest, indent=1) + "\n")
    load_created_at = time.time()
    resources.create(manifest)
    sampler = TopSampler(artifacts / "control-plane-top.jsonl")
    sampler.start()
    # Per Pod, once a second, through the load and the drain.
    series = start_processor_timeseries(pods)
    try:
        if gated_start:
            wait_for_load_pods(
                job,
                clusters,
                timeout_seconds=600,
            )
            start_epoch = time.time() + lead_seconds
            release_start_gate(start_epoch, resources=resources)
            log(
                f"all {clusters} load pods ready; synchronized start in {lead_seconds}s"
            )
        else:
            log(
                f"job {job} created; synchronized start in "
                f"{lead_seconds}s ({clusters} load pods)"
            )
        window_start = start_epoch
        status = wait_for_job(job, spec["deadline"] + lead_seconds + 120)
        window_end = time.time()
        log(f"job status: {status}")
        cgroup_after, pg_after, documents, drain = collect_after_load(
            pods, artifacts, job, targets=(target_queue_depth, target_spool_depth)
        )
        priority_latency = processor_priority_latency(load_created_at)
    finally:
        series.stop()
        sampler.stop()
        sampler.join(timeout=30)
    (artifacts / "processor-timeseries.json").write_text(
        json.dumps(series.document()) + "\n"
    )
    (artifacts / "queue-drain.json").write_text(json.dumps(drain, indent=1) + "\n")
    (artifacts / "processor-priority-latency.json").write_text(
        json.dumps(priority_latency, indent=1) + "\n"
    )
    aurora = aurora_window(window_start, window_end, writer=placement["aurora"])
    (artifacts / "aurora.json").write_text(
        json.dumps(aurora, indent=1, default=str) + "\n"
    )
    summary = aggregate(documents)
    summary.update(
        {
            "case": case,
            "label": label,
            "clusters": clusters,
            "nodes_per_cluster": nodes_per_cluster,
            "xid_total": xid_total,
            "sxid_total": sxid_total,
            "training_heartbeat_total": training_heartbeat_total,
            "workload_observation_total": workload_observation_total,
            "correlate_attempt_faults": correlate_attempt_faults,
            "job_status": status,
            "release_id": release_id(),
            "window_start_epoch": window_start,
            "window_end_epoch": window_end,
            "store_bloat": capacity_database.store_bloat_section(pg_before, pg_after),
            "store_layout": capacity_database.store_layout_section(pg_before, pg_after),
            # From the load stop; the *_after_collection pair is where the probe
            # used to start (after the collection), for older runs.
            "queue_drain_seconds": drain.get("drain_seconds"),
            "queue_drain_sample_interval_seconds": drain.get("sample_interval_seconds"),
            "queue_depth_at_load_stop": drain.get("depth_at_load_stop"),
            "telemetry_spool_depth_at_load_stop": drain.get(
                "telemetry_spool_depth_at_load_stop"
            ),
            "queue_drain_seconds_after_collection": drain.get(
                "drain_seconds_after_collection"
            ),
            "queue_depth_after_collection": drain.get("depth_after_collection"),
            "processor_timeseries": series.summary(),
            "cpu_throttling": {
                pod: {
                    key: int(after[key]) - int(cgroup_before.get(pod, {}).get(key, 0))
                    for key in (
                        "nr_throttled",
                        "throttled_usec",
                    )
                    if key in after
                }
                for pod, after in cgroup_after.items()
            },
            "postgres_deltas": {
                key: (pg_after.get(key) or 0) - (pg_before.get(key) or 0)
                for key in (
                    "deadlocks",
                    "xact_commit",
                    "xact_rollback",
                )
                if isinstance(pg_after.get(key), int)
                and isinstance(pg_before.get(key), int)
            },
            "queue_table_after": (pg_after.get("tables") or {}).get(
                "gpu_fault_processor_queue"
            ),
            "processor_priority_latency": priority_latency,
            "placement": placement_section(placement, AURORA_CLUSTER_ID),
            "aurora_capacity_max": max(
                (
                    point.get("maximum") or 0
                    for point in aurora.get("ServerlessDatabaseCapacity", [])
                    if isinstance(point, dict)
                ),
                default=None,
            ),
            "aurora_cpu_max": max(
                (
                    point.get("maximum") or 0
                    for point in aurora.get("CPUUtilization", [])
                    if isinstance(point, dict)
                ),
                default=None,
            ),
        }
    )
    (artifacts / "summary.json").write_text(
        json.dumps(summary, indent=1, sort_keys=True) + "\n"
    )
    return summary


def purge_audit_rows(
    *, run_id: str | None = None, artifacts: Path | None = None
) -> dict:
    """Remove only a registered run's quiescent data, never a shared prefix:
    settle, purge, then purge what lands late (``regional_capacity_settle``)."""
    return capacity_settle.purge_run_data(
        control, run_id=run_id, artifacts=artifacts, log=log
    )


def _teardown_once(
    *,
    purge: bool,
    deregister_clusters: bool,
    scope: str,
    artifacts: Path | None,
    run_id: str | None,
) -> None:
    if __package__:
        from .regional_capacity_resources import RunResources
    else:
        from regional_capacity_resources import RunResources
    from scripts.e2e.regional.seeded_command_fixture import (
        RUN_LABEL,
        delete_owned_resource,
        resource_metadata,
    )

    if (
        not isinstance(run_id, str)
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,62}", run_id) is None
    ):
        raise RuntimeError("capacity teardown requires an explicit run identity")
    if artifacts is None:
        raise RuntimeError("capacity teardown requires the run's registration intent")
    intent_path = artifacts / "registry-registration-intent.json"
    if not intent_path.exists():
        raise RuntimeError("capacity teardown has no registration ownership receipt")
    intent = json.loads(intent_path.read_text())
    if intent.get("run_id") != run_id:
        raise RuntimeError("capacity registration intent belongs to another run")
    RunResources(artifacts, run_id, NAMESPACE, dataplane).delete_all()
    token_uid = None
    metadata = resource_metadata("secret", TOKEN_SECRET, client=dataplane)
    if metadata:
        proof = json.loads((artifacts / "registry-token-proof.json").read_text())
        token_uid = proof.get("uid")
        if (
            proof.get("run_id") != run_id
            or not token_uid
            or metadata.get("uid") != token_uid
            or metadata.get("labels", {}).get(RUN_LABEL) != run_id
        ):
            raise RuntimeError("capacity token ownership or UID changed")
    if purge:
        purge_audit_rows(run_id=run_id, artifacts=artifacts)
    else:
        # --no-purge: the data must already be gone; prove it read-only first.
        remaining = capacity_data.invoke(
            control,
            run_id=run_id,
            cluster_ids=registered_cluster_ids(artifacts, run_id),
            cleanup=False,
        )
        if type(remaining.get("total")) is not int or remaining["total"] != 0:
            raise RuntimeError("capacity exact run-owned cleanup is incomplete")
    if deregister_clusters:
        # Registry, token Secret, then the gate (a failed gate must not strand the Secret).
        deregister(scope=scope, artifacts=artifacts, run_id=run_id, verify=False)
        delete_owned_resource(
            "secret",
            TOKEN_SECRET,
            run_id,
            client=dataplane,
            namespace=NAMESPACE,
            expected_uid=token_uid,
            require_uid=True,
        )
        capacity_registry.verify_registry_alignment(
            artifacts=artifacts, phase="postflight"
        )


def teardown(
    *,
    purge: bool,
    deregister_clusters: bool,
    allow_live_registry: bool = False,
    live_registry_confirmation: str | None = None,
    artifacts: Path | None = None,
    run_id: str | None = None,
    attempts: int = len(capacity_recovery.TEARDOWN_RETRY_DELAYS_SECONDS) + 1,
) -> None:
    """Retried ``_teardown_once``; every attempt repeats every ownership check."""
    scope = validate_registry_target(
        allow_live_registry=allow_live_registry,
        confirmation=live_registry_confirmation,
    )
    capacity_recovery.retry_teardown(
        lambda: _teardown_once(
            purge=purge,
            deregister_clusters=deregister_clusters,
            scope=scope,
            artifacts=artifacts,
            run_id=run_id,
        ),
        attempts=attempts,
    )


def validate_registered_resources(artifacts: Path, run_id: str, count: int) -> None:
    from scripts.e2e.regional.seeded_command_fixture import RUN_LABEL, resource_metadata

    if registered_cluster_ids(artifacts, run_id) != [
        f"{PERF_CLUSTER_PREFIX}{index:03d}" for index in range(count)
    ]:
        raise RuntimeError(
            "capacity cluster selection differs from registration intent"
        )
    proof = json.loads((artifacts / "registry-token-proof.json").read_text())
    if (
        not isinstance(proof, dict)
        or proof.get("run_id") != run_id
        or not isinstance(proof.get("uid"), str)
        or not proof["uid"]
        or not isinstance(proof.get("resource_version"), str)
        or not proof["resource_version"]
    ):
        raise RuntimeError(
            "capacity token UID proof is missing or belongs to another run"
        )
    metadata = resource_metadata("secret", TOKEN_SECRET, client=dataplane)
    if (
        metadata.get("uid") != proof["uid"]
        or metadata.get("resourceVersion") != proof["resource_version"]
        or metadata.get("namespace") != NAMESPACE
        or (metadata.get("labels") or {}).get(RUN_LABEL) != run_id
    ):
        raise RuntimeError("capacity token Secret changed since registration")


def execute_capacity_command(
    args: argparse.Namespace,
    *,
    artifacts: Path,
    suite_id: str,
    expires_at: datetime,
    scope: str,
    xid_total: int,
    sxid_total: int,
) -> int:
    failure: BaseException | None = None
    cleanup_error: Exception | None = None
    result = 0
    aborted_reason: str | None = None
    resources = RunResources(artifacts, suite_id, NAMESPACE, dataplane)
    registration_pending = False
    run_validated = False
    try:
        # Before any registration: a wrong cluster fails without registry work.
        if args.command in {"all", "run"}:
            placement = start_placement(AURORA_CLUSTER_ID, artifacts)
        if args.command in {"register", "all"}:
            registration_pending = True
            register(
                args.clusters,
                artifacts,
                run_id=suite_id,
                expires_at=expires_at,
                allow_live_registry=args.allow_live_registry,
                live_registry_confirmation=args.confirm_live_registry,
            )
            registration_pending = False
        elif args.command == "run":
            validate_registered_synthetic_run(
                count=args.clusters,
                run_id=suite_id,
                artifacts=artifacts,
                scope=scope,
            )
        if args.command in {"register", "all", "run"}:
            validate_registered_resources(artifacts, suite_id, args.clusters)
        run_validated = True
        if args.command == "register":
            write_status(artifacts, status="ok")
            return 0

        if args.command in {"all", "run"}:
            summary = execute_case(
                args.case,
                clusters=args.clusters,
                nodes_per_cluster=args.nodes_per_cluster,
                xid_total=xid_total,
                sxid_total=sxid_total,
                gpu_evidence_total=args.gpu_evidence_total,
                host_evidence_total=args.host_evidence_total,
                training_heartbeat_total=args.training_heartbeat_total,
                workload_observation_total=args.workload_observation_total,
                correlate_attempt_faults=args.correlate_attempt_faults,
                workers=args.workers,
                duration_seconds=args.duration_seconds,
                lead_seconds=args.lead_seconds,
                cpu_request=args.cpu_request,
                cpu_limit=args.cpu_limit,
                artifacts=artifacts,
                label=args.label or f"{args.case}-{args.clusters}c",
                include_telemetry=not args.fault_only,
                prewarm_connections=args.prewarm_connections,
                resources=resources,
                placement=placement,
            )
            print(json.dumps(summary, indent=1, sort_keys=True))
            if summary.get("job_status") != "Complete":
                result = 1
                aborted_reason = f"job status: {summary.get('job_status')}"
    except BaseException as exc:
        failure = exc
    finally:
        cleanup_required = registration_pending or (
            run_validated and (args.command != "register" or failure is not None)
        )
        if (
            cleanup_required
            and (artifacts / "registry-registration-intent.json").exists()
        ):
            try:
                registered_cluster_ids(artifacts, suite_id)
                teardown(
                    purge=args.command == "purge" or not args.no_purge,
                    deregister_clusters=args.command != "purge"
                    and (
                        not args.keep_registration
                        or (failure is not None and args.command == "register")
                    ),
                    allow_live_registry=args.allow_live_registry,
                    live_registry_confirmation=args.confirm_live_registry,
                    artifacts=artifacts,
                    run_id=suite_id,
                )
            except Exception as exc:
                cleanup_error = exc
                if failure is not None:
                    failure.add_note(f"capacity teardown also failed: {exc}")
                else:
                    failure = exc

    if failure is not None:
        capacity_recovery.write_aborted_status(
            artifacts,
            failure,
            cleanup_error=cleanup_error,
            run_id=suite_id,
            scope=scope,
        )
        log(f"aborted artifacts: {artifacts}")
        raise failure.with_traceback(failure.__traceback__)
    if result:
        write_status(
            artifacts,
            status="aborted",
            reason=aborted_reason or "capacity run aborted",
        )
        log(f"aborted artifacts: {artifacts}")
        return result
    write_status(artifacts, status="ok")
    return 0


def capacity_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="regional control-plane capacity suite"
    )
    parser.add_argument(
        "command",
        choices=[
            "register",
            "run",
            "teardown",
            "all",
            "purge",
        ],
    )
    parser.add_argument("--case", default="burst", choices=CASES)
    parser.add_argument("--clusters", type=int, default=32)
    parser.add_argument("--nodes-per-cluster", type=int, default=256)
    parser.add_argument("--xid-total", type=int, default=None)
    parser.add_argument("--sxid-total", type=int, default=None)
    parser.add_argument("--gpu-evidence-total", type=int, default=0)
    parser.add_argument("--host-evidence-total", type=int, default=0)
    parser.add_argument("--training-heartbeat-total", type=int, default=0)
    parser.add_argument("--workload-observation-total", type=int, default=0)
    parser.add_argument("--correlate-attempt-faults", action="store_true")
    parser.add_argument("--workers", type=int, default=256)
    parser.add_argument("--duration-seconds", type=int, default=60)
    parser.add_argument(
        "--lead-seconds",
        type=int,
        default=20,
        help=(
            "seconds between all burst load pods becoming Ready and "
            "the shared start epoch"
        ),
    )
    parser.add_argument("--cpu-request", default="2")
    parser.add_argument("--cpu-limit", default="4")
    parser.add_argument("--label", default=None)
    parser.add_argument(
        "--suite-id",
        help=(
            "shared ID for register/run phases of one capacity suite; "
            "generated when omitted"
        ),
    )
    parser.add_argument(
        "--fault-only",
        action="store_true",
        help="omit routine inventory/GPU/host telemetry",
    )
    parser.add_argument(
        "--prewarm-connections",
        action="store_true",
        help=("establish TLS connections before the synchronized start"),
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=DEFAULT_ARTIFACT_ROOT,
    )
    parser.add_argument("--keep-registration", action="store_true")
    parser.add_argument("--no-purge", action="store_true")
    parser.add_argument("--allow-live-registry", action="store_true")
    parser.add_argument("--confirm-live-registry")
    parser.add_argument("--synthetic-ttl-seconds", type=int, default=3600)
    parser.add_argument(
        "--run-dir",
        type=Path,
        help="original registration artifact directory for run, purge or teardown",
    )
    return parser


def resume_capacity_command(args: argparse.Namespace, scope: str) -> int:
    artifacts = args.run_dir
    cluster_ids = registered_cluster_ids(artifacts, args.suite_id)
    previous = json.loads((artifacts / "run.json").read_text())
    expected = capacity_recovery.resume_identity(
        suite_id=args.suite_id, scope=scope, identity=release_identity()
    )
    if not isinstance(previous, dict) or any(
        previous.get(key) != value for key, value in expected.items()
    ):
        raise RuntimeError(
            "capacity resume target or release differs from the original run"
        )
    if args.command == "run" and args.clusters != len(cluster_ids):
        raise RuntimeError(
            "capacity run cluster count differs from registration intent"
        )
    write_status(artifacts, status="running")
    return execute_capacity_command(
        args,
        artifacts=artifacts,
        suite_id=args.suite_id,
        expires_at=datetime.now(timezone.utc),
        scope=scope,
        xid_total=args.xid_total
        if args.xid_total is not None
        else round(500 * args.clusters / 32),
        sxid_total=args.sxid_total
        if args.sxid_total is not None
        else round(500 * args.clusters / 32),
    )


def main(argv: list[str] | None = None) -> int:
    parser = capacity_parser()
    args = parser.parse_args(argv)
    if not DATAPLANE_CONTEXT:
        parser.error("GPU_FAULT_DATAPLANE_CONTEXT is required")
    if args.command in {"all", "run"} and not AURORA_CLUSTER_ID:
        parser.error(
            "GPU_FAULT_AURORA_CLUSTER_ID is required: the run resolves its Aurora "
            "writer, and that writer's Availability Zone, from that cluster"
        )
    if args.synthetic_ttl_seconds < 300:
        parser.error("--synthetic-ttl-seconds must be at least 300")
    if args.command == "register" and not args.keep_registration:
        parser.error("register requires --keep-registration")
    if args.command in {"run", "purge", "teardown"}:
        if not args.suite_id or args.run_dir is None:
            parser.error(
                f"{args.command} requires the original --suite-id and --run-dir"
            )
    elif args.run_dir is not None:
        parser.error("--run-dir is only valid for run, purge or teardown")
    try:
        scope = validate_registry_target(
            allow_live_registry=args.allow_live_registry,
            confirmation=args.confirm_live_registry,
        )
    except RuntimeError as exc:
        parser.error(str(exc))
    if args.command in {"run", "purge", "teardown"}:
        return resume_capacity_command(args, scope)

    # 32 clusters carry 500 XID and 500 SXID; every other scale keeps the
    # same per-cluster fault rate so request totals stay linear.
    xid_total = (
        args.xid_total
        if args.xid_total is not None
        else round(500 * args.clusters / 32)
    )
    sxid_total = (
        args.sxid_total
        if args.sxid_total is not None
        else round(500 * args.clusters / 32)
    )
    label = args.label or f"{args.case}-{args.clusters}c"

    identity = release_identity()
    suite_id = args.suite_id or secrets.token_hex(8)
    artifacts = artifact_dir(
        args.artifact_root,
        label,
        identity["release_id"] or "unknown-release",
    )
    if any(artifacts.iterdir()):
        raise RuntimeError(
            "capacity artifacts already exist; resume with the original --run-dir"
        )
    log(f"artifacts: {artifacts}")
    started_at = datetime.now(timezone.utc).isoformat()
    expires_at = datetime.now(timezone.utc) + timedelta(
        seconds=args.synthetic_ttl_seconds
    )
    (artifacts / "run.json").write_text(
        json.dumps(
            {
                "case": args.case,
                "clusters": args.clusters,
                "nodes_per_cluster": args.nodes_per_cluster,
                "xid_total": xid_total,
                "sxid_total": sxid_total,
                "gpu_evidence_total": args.gpu_evidence_total,
                "host_evidence_total": (args.host_evidence_total),
                "training_heartbeat_total": (args.training_heartbeat_total),
                "workload_observation_total": (args.workload_observation_total),
                "correlate_attempt_faults": (args.correlate_attempt_faults),
                "workers": args.workers,
                "duration_seconds": args.duration_seconds,
                "fault_only": args.fault_only,
                "prewarm_connections": (args.prewarm_connections),
                "command": args.command,
                "suite_id": suite_id,
                "registry_scope": scope,
                "control_namespace": CONTROL_NAMESPACE,
                "dataplane_namespace": NAMESPACE,
                "registry_secret": REGISTRY_SECRET,
                "connection_secret": CONNECTION_SECRET,
                "synthetic_expires_at": expires_at.isoformat(),
                **identity,
                "started_at": started_at,
            },
            indent=1,
        )
        + "\n"
    )
    write_status(artifacts, status="running")
    return execute_capacity_command(
        args,
        artifacts=artifacts,
        suite_id=suite_id,
        expires_at=expires_at,
        scope=scope,
        xid_total=xid_total,
        sxid_total=sxid_total,
    )


if __name__ == "__main__":
    sys.exit(main())

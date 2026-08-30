#!/usr/bin/env python3
"""One-click regional capacity harness.

Wraps every manual step that used to surround the load generators:
audit cluster registration, load-generator token secret, payload template
ConfigMap, job rendering, durable log collection, control-plane metric and
Aurora sampling, database purge and deregistration.

The load generators run in the GPU data-plane cluster (it has the spare
CPU); registration happens in the regional control-plane cluster because
regional cluster tokens are read from a Secret at process start.

Examples
--------
    # full 32 cluster synchronized burst, artifacts under artifacts/perf/
    scripts/perf/regional_capacity_suite.py all --case burst --clusters 32

    # keep the registration between runs
    scripts/perf/regional_capacity_suite.py register --clusters 50
    scripts/perf/regional_capacity_suite.py run --case burst --clusters 50
    scripts/perf/regional_capacity_suite.py teardown --clusters 50
"""

from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import json
import os
import re
import secrets
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

if __package__:
    from .regional_capacity_cleanup import AUDIT_PURGE_STATEMENTS
    from .regional_capacity_database import (
        aurora_window as _aurora_window,
    )
    from .regional_capacity_database import (
        postgres_counters as _postgres_counters,
    )
    from .regional_capacity_database import (
        processor_priority_latency as _processor_priority_latency,
    )
    from .regional_capacity_job import build_job as _build_job
    from .regional_capacity_results import (
        aggregate,
        artifact_dir,
        drain_targets,
        move_to_aborted,
        write_status,
    )
else:
    from regional_capacity_cleanup import AUDIT_PURGE_STATEMENTS
    from regional_capacity_database import (
        aurora_window as _aurora_window,
    )
    from regional_capacity_database import (
        postgres_counters as _postgres_counters,
    )
    from regional_capacity_database import (
        processor_priority_latency as _processor_priority_latency,
    )
    from regional_capacity_job import build_job as _build_job
    from regional_capacity_results import (
        aggregate,
        artifact_dir,
        drain_targets,
        move_to_aborted,
        write_status,
    )

REPO_ROOT = Path(__file__).resolve().parents[2]
PERF_DIR = REPO_ROOT / "scripts" / "perf"
DEFAULT_ARTIFACT_ROOT = REPO_ROOT / "artifacts" / "perf"

NAMESPACE = "gpu-fault-system"
CONTROL_KUBECONFIG = os.getenv(
    "GPU_FAULT_CONTROL_KUBECONFIG",
    "/tmp/gpu-fault-control-plane.kubeconfig",
)
DATAPLANE_CONTEXT = os.getenv(
    "GPU_FAULT_DATAPLANE_CONTEXT",
    "",
)
AURORA_INSTANCE = os.getenv("GPU_FAULT_AURORA_INSTANCE", "gpu-fault-aurora-writer")
# The deployment's region, not the operator's. Reading AWS_REGION here
# would register the audit clusters in whatever region the shell that
# launched the harness happens to point at and would sample CloudWatch
# there too -- which returns an empty datapoint list rather than an
# error, so the Aurora section of the report would just look idle.
AWS_REGION = os.getenv("GPU_FAULT_PERF_AWS_REGION", "us-west-2")

REGISTRY_SECRET = "gpu-fault-regional-clusters"
TOKEN_SECRET = "gpu-fault-perf-clusters"
SCRIPT_CONFIGMAP = "gpu-fault-perf-suite-scripts"
TEMPLATE_CONFIGMAP = "gpu-fault-perf-suite-templates"
START_GATE_CONFIGMAP = "gpu-fault-perf-start-gate"
START_GATE_ROLE = "gpu-fault-perf-start-gate-reader"
START_GATE_ROLE_BINDING = "gpu-fault-perf-start-gate-reader"
CONTROL_DEPLOYMENTS = (
    "gpu-fault-api-ha",
    "gpu-fault-control-worker",
    "gpu-fault-telemetry-spool-worker",
)
PERF_CLUSTER_PREFIX = "perf-cap-"

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


def run(
    argv: list[str],
    *,
    stdin: bytes | None = None,
    check: bool = True,
    timeout: int = 300,
) -> subprocess.CompletedProcess:
    result = subprocess.run(
        argv,
        input=stdin,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
    )
    if check and result.returncode != 0:
        raise RuntimeError(
            f"command failed ({result.returncode}): "
            f"{' '.join(argv)}\n{result.stderr.decode()}"
        )
    return result


def control(
    *args: str,
    stdin: bytes | None = None,
    check: bool = True,
    timeout: int = 300,
) -> str:
    env_kubectl = [
        "kubectl",
        "--kubeconfig",
        CONTROL_KUBECONFIG,
    ]
    result = run(
        [*env_kubectl, "-n", NAMESPACE, *args],
        stdin=stdin,
        check=check,
        timeout=timeout,
    )
    return result.stdout.decode()


def dataplane(
    *args: str,
    stdin: bytes | None = None,
    check: bool = True,
    timeout: int = 300,
) -> str:
    result = run(
        [
            "kubectl",
            "--context",
            DATAPLANE_CONTEXT,
            "-n",
            NAMESPACE,
            *args,
        ],
        stdin=stdin,
        check=check,
        timeout=timeout,
    )
    return result.stdout.decode()


# --------------------------------------------------------------------------
# registration
# --------------------------------------------------------------------------


def load_registry() -> list[dict]:
    raw = control(
        "get",
        "secret",
        REGISTRY_SECRET,
        "-o",
        "jsonpath={.data.clusters\\.json}",
    )
    return json.loads(base64.b64decode(raw))


def validate_registry(entries: list[dict]) -> None:
    """Parse the rows with the control plane's own model.

    The registry is read in create_app(), so a malformed row does not
    surface as a failed request -- every replica crash-loops on start
    and the only way back is another patch. Checking here costs
    nothing and keeps a bad row out of the live Secret.
    """
    sys.path.insert(0, str(REPO_ROOT / "src"))
    try:
        from gpu_fault.regional import (  # noqa: PLC0415
            RegionalClusterRegistration,
            cluster_token_sha256,
        )
    except ImportError as exc:
        log(f"registry pre-validation skipped: {exc}")
        return
    for entry in entries:
        item = dict(entry)
        token = item.pop("token", None)
        if token:
            item["token_sha256"] = cluster_token_sha256(str(token))
        RegionalClusterRegistration(**item)


def write_registry(entries: list[dict]) -> None:
    validate_registry(entries)
    payload = base64.b64encode(json.dumps(entries, indent=1).encode()).decode()
    patch = json.dumps({"data": {"clusters.json": payload}})
    control("patch", "secret", REGISTRY_SECRET, "-p", patch)


def validate_notification_safety() -> None:
    """Refuse a capacity run that can mail synthetic drill notifications."""
    pods = control_pods()
    for pod in pods:
        value = (
            control(
                "exec",
                pod,
                "--",
                "python3",
                "-c",
                (
                    "import os; print("
                    "os.environ.get('GPU_FAULT_NOTIFICATION_DELIVER_DRILLS','false')"
                    ")"
                ),
            )
            .strip()
            .lower()
        )
        if value not in {"", "false", "0", "no", "off"}:
            raise RuntimeError(
                "capacity runs must not deliver drill notifications: "
                f"{pod} has GPU_FAULT_NOTIFICATION_DELIVER_DRILLS={value}"
            )


def restart_control_plane() -> None:
    for deployment in CONTROL_DEPLOYMENTS:
        replicas = control(
            "get",
            "deploy",
            deployment,
            "-o",
            "jsonpath={.spec.replicas}",
        ).strip()
        if replicas in {"", "0"}:
            log(f"skip restart of {deployment} (replicas={replicas})")
            continue
        log(f"rollout restart {deployment}")
        control("rollout", "restart", f"deploy/{deployment}")
    for deployment in CONTROL_DEPLOYMENTS:
        replicas = control(
            "get",
            "deploy",
            deployment,
            "-o",
            "jsonpath={.spec.replicas}",
        ).strip()
        if replicas in {"", "0"}:
            continue
        log(f"waiting for {deployment}")
        control(
            "rollout",
            "status",
            f"deploy/{deployment}",
            "--timeout=600s",
            timeout=700,
        )


def perf_cluster_entries(count: int) -> list[dict]:
    """Build registry rows for the synthetic audit clusters.

    RegionalClusterRegistration is strict: hyperpod_cluster_name and
    eks_cluster_arn are required, and an unknown key is rejected
    outright. A row missing either one does not fail the request that
    uses it -- it fails create_app(), so every replica crash-loops the
    moment the registry is patched. The identifiers are per-cluster
    and deliberately synthetic: nothing on the ingest path resolves
    them, and sharing one real name across 32 rows would point 32
    managed-recovery observers at the same live HyperPod cluster.
    """
    account = "000000000000"
    entries = []
    for index in range(count):
        cluster_id = f"{PERF_CLUSTER_PREFIX}{index:03d}"
        entries.append(
            {
                "cluster_id": cluster_id,
                "region": AWS_REGION,
                "hyperpod_cluster_name": cluster_id,
                "eks_cluster_arn": (
                    f"arn:aws:eks:{AWS_REGION}:{account}:cluster/{cluster_id}"
                ),
                "token": secrets.token_urlsafe(48),
                "allowed_namespaces": [
                    "default",
                    NAMESPACE,
                    "kubeflow",
                    "training",
                ],
            }
        )
    return entries


def redacted_registry_entries(entries: list[dict]) -> list[dict]:
    """Swap every live cluster token for its digest before it hits artifacts/.

    The baseline rows come straight out of the production registry Secret,
    so they carry the real regional cluster token. The artifact only ever
    has to answer "which rows did the run preserve, and did it put the same
    ones back", and a sha256 answers that as well as the token does. Same
    digest helper the API uses (``gpu_fault.regional.cluster_token_sha256``),
    inlined so redaction cannot be skipped by an import failure the way
    ``validate_registry`` legitimately is.
    """
    redacted = []
    for entry in entries:
        item = dict(entry)
        token = item.pop("token", None)
        if token is not None:
            item["token_sha256"] = hashlib.sha256(str(token).encode()).hexdigest()
        redacted.append(item)
    return redacted


def register(count: int, artifacts: Path) -> list[dict]:
    validate_notification_safety()
    existing = load_registry()
    baseline = [
        entry
        for entry in existing
        if not str(entry.get("cluster_id", "")).startswith(PERF_CLUSTER_PREFIX)
    ]
    (artifacts / "registry-baseline.json").write_text(
        json.dumps(redacted_registry_entries(baseline), indent=1) + "\n"
    )
    perf = perf_cluster_entries(count)
    log(
        f"registering {len(perf)} audit clusters "
        f"(keeping {len(baseline)} production entries)"
    )
    write_registry(baseline + perf)
    restart_control_plane()
    tokens = [
        {
            "cluster_id": entry["cluster_id"],
            "token": entry["token"],
        }
        for entry in perf
    ]
    upsert_secret(
        TOKEN_SECRET,
        {"clusters.json": json.dumps(tokens, indent=1).encode()},
    )
    return tokens


def deregister() -> None:
    existing = load_registry()
    baseline = [
        entry
        for entry in existing
        if not str(entry.get("cluster_id", "")).startswith(PERF_CLUSTER_PREFIX)
    ]
    if len(baseline) == len(existing):
        log("no audit clusters registered; nothing to deregister")
        return
    log(f"deregistering {len(existing) - len(baseline)} audit clusters")
    write_registry(baseline)
    restart_control_plane()


# --------------------------------------------------------------------------
# data-plane fixtures
# --------------------------------------------------------------------------


def upsert_secret(name: str, files: dict[str, bytes]) -> None:
    data = {key: base64.b64encode(value).decode() for key, value in files.items()}
    manifest = {
        "apiVersion": "v1",
        "kind": "Secret",
        "metadata": {"name": name, "namespace": NAMESPACE},
        "type": "Opaque",
        "data": data,
    }
    dataplane(
        "apply",
        "-f",
        "-",
        stdin=json.dumps(manifest).encode(),
    )


def upsert_configmap(
    name: str,
    *,
    text: dict[str, str] | None = None,
    binary: dict[str, bytes] | None = None,
) -> None:
    manifest = {
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
    dataplane(
        "apply",
        "-f",
        "-",
        stdin=json.dumps(manifest).encode(),
    )


def prepare_start_gate() -> None:
    upsert_configmap(
        START_GATE_CONFIGMAP,
        text={"start_epoch": ""},
    )
    manifest = {
        "apiVersion": "v1",
        "kind": "List",
        "items": [
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
                        "name": "gpu-fault-completion-watcher",
                        "namespace": NAMESPACE,
                    }
                ],
                "roleRef": {
                    "apiGroup": "rbac.authorization.k8s.io",
                    "kind": "Role",
                    "name": START_GATE_ROLE,
                },
            },
        ],
    }
    dataplane(
        "apply",
        "-f",
        "-",
        stdin=json.dumps(manifest).encode(),
    )


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


def release_start_gate(start_epoch: float) -> None:
    upsert_configmap(
        START_GATE_CONFIGMAP,
        text={"start_epoch": f"{start_epoch:.6f}"},
    )


def publish_fixtures(case: str) -> None:
    script_name = CASES[case]["script"]
    payloads = {}
    for name in {script_name, *SUPPORT_SCRIPTS}:
        payloads[name] = (PERF_DIR / name).read_text()
    upsert_configmap(SCRIPT_CONFIGMAP, text=payloads)
    templates = json.loads((PERF_DIR / "payload-templates.json").read_text())
    upsert_configmap(
        TEMPLATE_CONFIGMAP,
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
    )


# --------------------------------------------------------------------------
# control-plane observation
# --------------------------------------------------------------------------


def control_pods() -> list[str]:
    raw = control(
        "get",
        "pod",
        "-l",
        (
            "app in (gpu-fault-api-ha,gpu-fault-control-worker,"
            "gpu-fault-telemetry-spool-worker)"
        ),
        "-o",
        "jsonpath={range .items[*]}{.metadata.name} {end}",
    )
    names = [name for name in raw.split() if name]
    if names:
        return names
    raw = control(
        "get",
        "pod",
        "-o",
        "jsonpath={range .items[*]}{.metadata.name} {end}",
    )
    return [
        name
        for name in raw.split()
        if name.startswith(
            (
                "gpu-fault-api-ha-",
                "gpu-fault-control-worker-",
                "gpu-fault-telemetry-spool-worker-",
            )
        )
    ]


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
    r"telemetry_spool|queue_bypass|processor_notification"
)


def scrape_metrics(pods: list[str]) -> dict[str, str]:
    snapshot = {}
    for pod in pods:
        port = (
            "8082"
            if "telemetry-spool-worker" in pod
            else "8081"
            if "control-worker" in pod
            else "8080"
        )
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


def aurora_window(start: float, end: float) -> dict:
    return _aurora_window(
        run,
        aurora_instance=AURORA_INSTANCE,
        aws_region=AWS_REGION,
        start=start,
        end=end,
    )


# --------------------------------------------------------------------------
# execution + collection
# --------------------------------------------------------------------------


def collect_logs(job: str, target: Path) -> list[dict]:
    target.mkdir(parents=True, exist_ok=True)
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
    documents = []
    for line in raw.splitlines():
        parts = line.split()
        if not parts:
            continue
        pod = parts[0]
        index = parts[1] if len(parts) > 1 else "?"
        body = dataplane("logs", pod, check=False, timeout=180)
        (target / f"{index}-{pod}.log").write_text(body)
        try:
            documents.append(json.loads(body))
        except json.JSONDecodeError:
            log(f"pod {pod} produced non-JSON output")
    return documents


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


def queue_drain(
    pods: list[str],
    timeout_seconds: int = 900,
    interval_seconds: float = 1.0,
    target_queue_depth: float = 1.0,
    target_spool_depth: float = 1.0,
) -> dict:
    """Wait for the processor queue to return to the steady state.

    The acceptance gate is "back to steady state within 60s of the burst",
    so the sampling resolution has to be small compared to 60s. One
    ``/metrics`` scrape costs about 2s (kubectl exec), and the loop stops at
    the first empty sample, which means ``drain_seconds`` is an upper bound
    with roughly ``interval_seconds + 2s`` of granularity. Keep the interval
    at 1s: a 15s interval reported every fast drain as "about 2s" and every
    slow one as a multiple of 17s, which is not comparable across runs.
    """
    started = time.time()
    history = []
    probe = pods[0] if pods else None
    if probe is None:
        return {"samples": history}
    while time.time() - started < timeout_seconds:
        snapshot = scrape_metrics([probe]).get(probe, "")
        depth = None
        spool_depth = None
        oldest = None
        for line in snapshot.splitlines():
            if line.startswith("gpu_fault_processor_queue_depth "):
                depth = float(line.split()[-1])
            if line.startswith("gpu_fault_telemetry_spool_depth "):
                spool_depth = float(line.split()[-1])
            if line.startswith("gpu_fault_processor_queue_oldest_age_seconds "):
                oldest = float(line.split()[-1])
        history.append(
            {
                "epoch": time.time(),
                "elapsed_seconds": time.time() - started,
                "queue_depth": depth,
                "telemetry_spool_depth": spool_depth,
                "oldest_age_seconds": oldest,
            }
        )
        if (
            depth is not None
            and depth <= target_queue_depth
            and (spool_depth is None or spool_depth <= target_spool_depth)
        ):
            break
        time.sleep(interval_seconds)
    return {
        "drain_seconds": history[-1]["elapsed_seconds"] if history else None,
        "sample_interval_seconds": interval_seconds,
        "depth_at_load_stop": history[0]["queue_depth"] if history else None,
        "telemetry_spool_depth_at_load_stop": (
            history[0]["telemetry_spool_depth"] if history else None
        ),
        "target_queue_depth": target_queue_depth,
        "target_spool_depth": target_spool_depth,
        "samples": history,
    }


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
    return control(
        "get",
        "deploy",
        "gpu-fault-api-ha",
        "-o",
        "jsonpath={.spec.template.spec.containers[0].env"
        "[?(@.name=='GPU_FAULT_RELEASE_ID')].value}",
        check=False,
    ).strip()


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


def processor_priority_latency() -> dict:
    pods = [pod for pod in control_pods() if "api-ha" in pod]
    return _processor_priority_latency(
        control,
        pods,
        cluster_prefix=PERF_CLUSTER_PREFIX,
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
) -> dict:
    spec = CASES[case]
    job = spec["job"]
    publish_fixtures(case)
    dataplane(
        "delete",
        "job",
        job,
        "--ignore-not-found",
        "--wait=true",
        check=False,
        timeout=300,
    )
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
        prepare_start_gate()
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
    (artifacts / "job.json").write_text(json.dumps(manifest, indent=1) + "\n")
    dataplane(
        "apply",
        "-f",
        "-",
        stdin=json.dumps(manifest).encode(),
    )
    sampler = TopSampler(artifacts / "control-plane-top.jsonl")
    sampler.start()
    if gated_start:
        wait_for_load_pods(
            job,
            clusters,
            timeout_seconds=600,
        )
        start_epoch = time.time() + lead_seconds
        release_start_gate(start_epoch)
        log(f"all {clusters} load pods ready; synchronized start in {lead_seconds}s")
    else:
        log(
            f"job {job} applied; synchronized start in "
            f"{lead_seconds}s ({clusters} load pods)"
        )
    window_start = start_epoch
    status = wait_for_job(job, spec["deadline"] + lead_seconds + 120)
    window_end = time.time()
    log(f"job status: {status}")
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
    drain = queue_drain(
        pods,
        target_queue_depth=target_queue_depth,
        target_spool_depth=target_spool_depth,
    )
    priority_latency = processor_priority_latency()
    sampler.stop()
    sampler.join(timeout=30)
    (artifacts / "queue-drain.json").write_text(json.dumps(drain, indent=1) + "\n")
    (artifacts / "processor-priority-latency.json").write_text(
        json.dumps(priority_latency, indent=1) + "\n"
    )
    aurora = aurora_window(window_start, window_end)
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
            "queue_drain_seconds": drain.get("drain_seconds"),
            "queue_drain_sample_interval_seconds": drain.get("sample_interval_seconds"),
            "queue_depth_at_load_stop": drain.get("depth_at_load_stop"),
            "telemetry_spool_depth_at_load_stop": drain.get(
                "telemetry_spool_depth_at_load_stop"
            ),
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


def purge_audit_rows() -> None:
    pods = [pod for pod in control_pods() if "api-ha" in pod]
    if not pods:
        log("no api pod available for purge")
        return
    script = f"""
import os, psycopg
prefix = {PERF_CLUSTER_PREFIX!r} + '%'
patterns = {{
    "cluster": prefix,
    "action_workflow": "workflow-actionperf-%",
}}
statements = {AUDIT_PURGE_STATEMENTS!r}
with psycopg.connect(os.environ['GPU_FAULT_STORE_URL'], autocommit=True) as conn:
    cur = conn.cursor()
    for table, sql, pattern_name in statements:
        try:
            cur.execute(sql, (patterns[pattern_name],))
            print(f'{{table}} deleted={{cur.rowcount}}')
        except Exception as exc:
            print(f'{{table}} skipped: {{exc}}')
"""
    log("purging audit-cluster rows")
    output = control(
        "exec",
        pods[0],
        "--",
        "python3",
        "-c",
        script,
        check=False,
        timeout=900,
    )
    print(output)


def teardown(*, purge: bool, deregister_clusters: bool) -> None:
    for spec in CASES.values():
        dataplane(
            "delete",
            "job",
            spec["job"],
            "--ignore-not-found",
            check=False,
            timeout=300,
        )
    dataplane(
        "delete",
        "configmap",
        SCRIPT_CONFIGMAP,
        TEMPLATE_CONFIGMAP,
        START_GATE_CONFIGMAP,
        "--ignore-not-found",
        check=False,
    )
    dataplane(
        "delete",
        "role",
        START_GATE_ROLE,
        "--ignore-not-found",
        check=False,
    )
    dataplane(
        "delete",
        "rolebinding",
        START_GATE_ROLE_BINDING,
        "--ignore-not-found",
        check=False,
    )
    dataplane(
        "delete",
        "secret",
        TOKEN_SECRET,
        "--ignore-not-found",
        check=False,
    )
    if purge:
        purge_audit_rows()
    if deregister_clusters:
        deregister()


def main() -> int:
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
    args = parser.parse_args()
    if not DATAPLANE_CONTEXT:
        parser.error("GPU_FAULT_DATAPLANE_CONTEXT is required")

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

    if args.command == "purge":
        purge_audit_rows()
        return 0
    if args.command == "teardown":
        teardown(
            purge=not args.no_purge,
            deregister_clusters=not args.keep_registration,
        )
        return 0

    identity = release_identity()
    suite_id = args.suite_id or secrets.token_hex(8)
    artifacts = artifact_dir(
        args.artifact_root,
        label,
        identity["release_id"] or "unknown-release",
    )
    log(f"artifacts: {artifacts}")
    started_at = datetime.now(timezone.utc).isoformat()
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
                **identity,
                "started_at": started_at,
            },
            indent=1,
        )
        + "\n"
    )
    write_status(artifacts, status="running")

    try:
        if args.command in {"register", "all"}:
            register(args.clusters, artifacts)
        if args.command == "register":
            write_status(artifacts, status="ok")
            return 0

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
            label=label,
            include_telemetry=not args.fault_only,
            prewarm_connections=args.prewarm_connections,
        )
        print(json.dumps(summary, indent=1, sort_keys=True))

        if args.command == "all" and not args.keep_registration:
            teardown(
                purge=not args.no_purge,
                deregister_clusters=True,
            )
        if summary.get("job_status") == "Complete":
            write_status(artifacts, status="ok")
            return 0
        write_status(
            artifacts,
            status="aborted",
            reason=f"job status: {summary.get('job_status')}",
        )
        moved = move_to_aborted(args.artifact_root, artifacts)
        log(f"aborted artifacts: {moved}")
        return 1
    except BaseException as exc:
        write_status(
            artifacts,
            status="aborted",
            reason=f"{type(exc).__name__}: {exc}",
        )
        moved = move_to_aborted(args.artifact_root, artifacts)
        log(f"aborted artifacts: {moved}")
        raise


if __name__ == "__main__":
    sys.exit(main())

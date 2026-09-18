#!/usr/bin/env python3
from __future__ import annotations

import argparse
import functools
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.perf.regional_capacity_registry import STORE_DSN_SNIPPET  # noqa: E402

if __package__:
    from .regional_commands import run_fixture_command
    from .ha_evidence import chain_preflight, require_chain, result_identity
    from .ha_kubernetes import delete_pod
    from .regional_pod_inventory import ready_pod_records
    from .ha_probe_resources import OwnedProbeResources
    from .ha_cleanup import ProcessSupervisionLost, record_supervision_loss, run_cleanup
    from .ha001_closure import closure_errors, closure_seed, closure_summary
    from .ha001_role_health import validate_role_health
    from .ha_store_probe import store_probe_script
    from .ha_plan_preflight import require_window
    from .acceptance_runner_common import write_json_atomic
    from .live_driver_guard import (
        add_live_arguments,
        install_site_profile,
    )
    from .live_driver_guard import (
        authorize_execution as guard_authorize_execution,
        build_plan as guard_build_plan,
    )
    from .regional_live_fixture import (
        component_python,
        install_abort_signals,
        run_case_main,
    )
else:
    from regional_commands import run_fixture_command
    from ha_evidence import chain_preflight, require_chain, result_identity
    from ha_kubernetes import delete_pod
    from regional_pod_inventory import ready_pod_records
    from ha_probe_resources import OwnedProbeResources
    from ha_cleanup import ProcessSupervisionLost, record_supervision_loss, run_cleanup
    from ha001_closure import closure_errors, closure_seed, closure_summary
    from ha001_role_health import validate_role_health
    from ha_store_probe import store_probe_script
    from ha_plan_preflight import require_window
    from acceptance_runner_common import write_json_atomic

    from live_driver_guard import (
        add_live_arguments,
        install_site_profile,
    )
    from live_driver_guard import (
        authorize_execution as guard_authorize_execution,
        build_plan as guard_build_plan,
    )
    from regional_live_fixture import (
        component_python,
        install_abort_signals,
        run_case_main,
    )

PROBE_SCRIPT = Path(__file__).with_name("probes") / "ha001_probe.py"
CPU_KUBECONFIG = Path()
GPU_KUBECONFIG = Path()
GPU_CONTEXT = ""
NAMESPACE = "gpu-fault-system"
AWS_REGION = ""
CASE_ID = "GF-REGIONAL-HA-001"
CONFIGMAP = "gpu-fault-ha001-probe"
PROBE_POD = "gpu-fault-ha001-probe"
OWNER = "gpu-fault-ha-test"
SEED_LEASE_SECONDS = 30 * 60
CONFIRMATION = "HA001_DELETE_CONTROL_PODS"
INGRESS_APP = "gpu-fault-api-ha"
WORKER_APP = "gpu-fault-control-worker"
SPOOL_APP = "gpu-fault-telemetry-spool-worker"
# Each deletion is observed until the replacement Pod is Ready and the probe has
# reported no failure for SETTLE_SECONDS, but never longer than the cap.
OBSERVATION_CAP_SECONDS = 300
SETTLE_SECONDS = 60
BASELINE_SECONDS = 30
# The probe runs one claim and one health cycle every two seconds; the
# sustained-load check asks for this fraction of the cycles the elapsed time
# allows, so a run that observed for longer is held to more, not to a constant.
PROBE_CYCLE_SECONDS = 2.0
MINIMUM_CYCLE_FRACTION = 0.8
# NLB target deregistration propagates in well under this; the failure window
# a deletion may open is bounded by it plus the application's own graceful exit
# (the preStop sleep and uvicorn's --timeout-graceful-shutdown), both read from
# the live Deployment rather than assumed.
NLB_DEREGISTRATION_SECONDS = 30
LIMITATIONS = [
    "workflow closure is driven by a SimulatedAdapter; "
    "dispatcher->executor->Node Agent path is covered by E2E-001",
]


class CaseError(RuntimeError):
    pass


def environment_values() -> dict[str, str]:
    return {
        "CPU_KUBECONFIG": str(CPU_KUBECONFIG),
        "GPU_KUBECONFIG": str(GPU_KUBECONFIG),
        "GPU_EKS_CONTEXT": GPU_CONTEXT,
        "GPU_FAULT_NAMESPACE": NAMESPACE,
        "AWS_REGION": AWS_REGION,
    }


def configure(arguments: argparse.Namespace) -> None:
    global AWS_REGION
    global CPU_KUBECONFIG
    global GPU_CONTEXT
    global GPU_KUBECONFIG
    global NAMESPACE

    cpu_path = arguments.cpu_kubeconfig or os.getenv("CPU_KUBECONFIG", "")
    gpu_path = (
        arguments.gpu_kubeconfig
        or os.getenv("GPU_KUBECONFIG")
        or os.getenv("KUBECONFIG", "")
    )
    context = (
        arguments.gpu_context
        or os.getenv("GPU_EKS_CONTEXT")
        or os.getenv("GPU_FAULT_DATAPLANE_CONTEXT", "")
    )
    region = (
        arguments.region
        or os.getenv("AWS_REGION")
        or os.getenv("AWS_DEFAULT_REGION", "")
    )
    if not cpu_path or not gpu_path or not context or not region:
        raise CaseError("CPU/GPU kubeconfigs, GPU context and AWS Region are required")
    CPU_KUBECONFIG = Path(cpu_path).expanduser().resolve()
    GPU_KUBECONFIG = Path(gpu_path).expanduser().resolve()
    if not CPU_KUBECONFIG.is_file() or not GPU_KUBECONFIG.is_file():
        raise CaseError("configured CPU/GPU kubeconfig does not exist")
    GPU_CONTEXT = context
    NAMESPACE = arguments.namespace
    AWS_REGION = region


def log(message: str) -> None:
    stamp = datetime.now(timezone.utc).strftime("%H:%M:%S")
    print(f"[{stamp}] {message}", flush=True)


def run(
    argv: list[str],
    *,
    stdin: str | None = None,
    check: bool = True,
    timeout: int = 300,
) -> subprocess.CompletedProcess[str]:
    return run_fixture_command(argv, input_text=stdin, check=check, timeout=timeout)


def cpu(
    *args: str,
    stdin: str | None = None,
    check: bool = True,
    timeout: int = 300,
) -> str:
    return run(
        [
            "kubectl",
            "--kubeconfig",
            str(CPU_KUBECONFIG),
            "-n",
            NAMESPACE,
            *args,
        ],
        stdin=stdin,
        check=check,
        timeout=timeout,
    ).stdout


def gpu(
    *args: str,
    stdin: str | None = None,
    check: bool = True,
    timeout: int = 300,
) -> str:
    return run(
        [
            "kubectl",
            "--kubeconfig",
            str(GPU_KUBECONFIG),
            "--context",
            GPU_CONTEXT,
            "-n",
            NAMESPACE,
            *args,
        ],
        stdin=stdin,
        check=check,
        timeout=timeout,
    ).stdout


def ready_pods(app: str) -> list[dict]:
    value = json.loads(cpu("get", "pod", "-l", f"app={app}", "-o", "json"))
    return ready_pod_records(value)


def first_ready_cpu_pod(app: str) -> str:
    pods = ready_pods(app)
    if not pods:
        raise CaseError(f"no Ready CPU Pod for app={app}")
    return str(pods[0]["name"])


def cpu_python(script: str, *arguments: str, attempts: int = 3) -> dict:
    """Run ``script`` in a Ready ingress Pod; ``attempts=1`` for anything that writes.

    A retry after a failure that happened *after* the write performs the write
    again, so the closure seed and anything else that mutates the store gets one
    attempt only.
    """

    if attempts < 1:
        raise ValueError("attempts must be at least 1")
    last_error: Exception | None = None
    for _attempt in range(attempts):
        pod = first_ready_cpu_pod("gpu-fault-api-ha")
        try:
            output = cpu(
                "exec",
                "-i",
                pod,
                "--",
                component_python("cpu"),
                "-",
                *arguments,
                stdin=store_probe_script(script),
                timeout=120,
            )
            return json.loads(output.splitlines()[-1])
        except Exception as exc:
            last_error = exc
            time.sleep(1)
    raise CaseError(f"CPU store probe failed: {last_error}")


def queue_stats() -> dict:
    return cpu_python(
        """
import json
from gpu_fault.app import ApplicationContext
print(json.dumps(
    ApplicationContext.from_environment().store.processor_queue_stats(),
    default=str,
    sort_keys=True,
))
"""
    )


def pre_stop_sleep_seconds(pre_stop: dict | None) -> int | None:
    """The ``sleep N`` a preStop exec hook runs, or None when there is none."""

    command = ((pre_stop or {}).get("exec") or {}).get("command") or []
    for part in command:
        match = re.search(r"\bsleep\s+(\d+)\b", str(part))
        if match:
            return int(match.group(1))
    return None


def graceful_shutdown_seconds(container: dict) -> int | None:
    """uvicorn's ``--timeout-graceful-shutdown`` from the container command line."""

    text = " ".join(
        str(part)
        for part in [*container.get("command", []), *container.get("args", [])]
    )
    match = re.search(r"--timeout-graceful-shutdown\s+(\d+)\b", text)
    return int(match.group(1)) if match else None


def failure_window_limit(
    deployment: dict,
    *,
    nlb_deregistration_seconds: int = NLB_DEREGISTRATION_SECONDS,
) -> dict:
    """The failure window one Pod deletion may open, from the Deployment's own values.

    The catalog bounds the window by NLB target deregistration plus the
    application's graceful exit; the latter is the preStop sleep that keeps the
    old Pod serving while endpoints drop it, plus uvicorn's graceful-shutdown
    timeout. Both are read from the live manifest so the limit follows a
    deploy-time change instead of a constant that was true once.
    """

    pre_stop = deployment.get("pre_stop_sleep_seconds")
    graceful = deployment.get("graceful_shutdown_seconds")
    if pre_stop is None or graceful is None:
        raise CaseError(
            "cannot derive the failure window limit: preStop sleep or "
            "--timeout-graceful-shutdown is missing from the Deployment"
        )
    return {
        "pre_stop_sleep_seconds": int(pre_stop),
        "graceful_shutdown_seconds": int(graceful),
        "nlb_deregistration_seconds": int(nlb_deregistration_seconds),
        "limit_seconds": int(pre_stop)
        + int(graceful)
        + int(nlb_deregistration_seconds),
    }


def declared_replicas(topology: dict) -> dict[str, int]:
    """``spec.replicas`` per control-plane Deployment, from a topology snapshot."""

    deployments = topology.get("deployments") or {}
    result = {}
    for app in (INGRESS_APP, WORKER_APP, SPOOL_APP):
        if app not in deployments:
            raise CaseError(f"topology snapshot has no Deployment {app}")
        result[app] = int(deployments[app].get("replicas") or 0)
    return result


def minimum_ready_from_replicas(replicas: dict[str, int]) -> dict[str, int | None]:
    """Quorum floors during one deletion: every role may lose exactly one Pod."""

    return {
        "ingress": max(0, replicas[INGRESS_APP] - 1),
        "worker": max(0, replicas[WORKER_APP] - 1),
        "spool": None,
    }


def minimum_probe_cycles(
    elapsed_seconds: float,
    *,
    cycle_seconds: float = PROBE_CYCLE_SECONDS,
    fraction: float = MINIMUM_CYCLE_FRACTION,
) -> int:
    """How many probe cycles ``elapsed_seconds`` of observation must have sustained."""

    if elapsed_seconds <= 0:
        return 0
    return int((elapsed_seconds / cycle_seconds) * fraction)


def deployment_and_pdb_snapshot() -> dict:
    deployments = json.loads(
        cpu(
            "get",
            "deployment",
            INGRESS_APP,
            WORKER_APP,
            SPOOL_APP,
            "-o",
            "json",
        )
    )
    pdbs = json.loads(cpu("get", "pdb", "-o", "json"))
    return {
        "deployments": {
            item["metadata"]["name"]: {
                "replicas": item["spec"].get("replicas", 0),
                "ready_replicas": item.get("status", {}).get("readyReplicas", 0),
                "termination_grace_seconds": item["spec"]["template"]["spec"].get(
                    "terminationGracePeriodSeconds"
                ),
                "pre_stop": item["spec"]["template"]["spec"]["containers"][0]
                .get("lifecycle", {})
                .get("preStop"),
                "pre_stop_sleep_seconds": pre_stop_sleep_seconds(
                    item["spec"]["template"]["spec"]["containers"][0]
                    .get("lifecycle", {})
                    .get("preStop")
                ),
                "graceful_shutdown_seconds": graceful_shutdown_seconds(
                    item["spec"]["template"]["spec"]["containers"][0]
                ),
                "topology_spread": item["spec"]["template"]["spec"].get(
                    "topologySpreadConstraints",
                    [],
                ),
            }
            for item in deployments.get("items", [])
        },
        "pdbs": {
            item["metadata"]["name"]: {
                "min_available": item["spec"].get("minAvailable"),
                "max_unavailable": item["spec"].get("maxUnavailable"),
                "current_healthy": item.get("status", {}).get("currentHealthy", 0),
                "desired_healthy": item.get("status", {}).get("desiredHealthy", 0),
                "disruptions_allowed": item.get("status", {}).get(
                    "disruptionsAllowed",
                    0,
                ),
            }
            for item in pdbs.get("items", [])
        },
    }


def role_snapshot() -> dict:
    deployments = json.loads(
        cpu(
            "get",
            "deployment",
            "gpu-fault-api-ha",
            "gpu-fault-control-worker",
            "-o",
            "json",
        )
    )
    ports = {
        item["metadata"]["name"]: int(
            item["spec"]["template"]["spec"]["containers"][0]["ports"][0][
                "containerPort"
            ]
        )
        for item in deployments.get("items", [])
    }
    script = """
import json
import re
import sys
import urllib.request
port = int(sys.argv[1])
with urllib.request.urlopen(
    f"http://127.0.0.1:{port}/healthz", timeout=10
) as response:
    health = json.load(response)
with urllib.request.urlopen(
    f"http://127.0.0.1:{port}/metrics", timeout=10
) as response:
    metrics = response.read().decode()
match = re.search(
    r"^gpu_fault_processor_active_consumer\\s+([0-9.]+)$",
    metrics,
    re.MULTILINE,
)
print(json.dumps({
    "health": health,
    "processor_active_consumer": (
        float(match.group(1)) if match else None
    ),
}, sort_keys=True))
"""
    roles = {}
    for app in ("gpu-fault-api-ha", "gpu-fault-control-worker"):
        roles[app] = []
        for pod in ready_pods(app):
            value = json.loads(
                cpu(
                    "exec",
                    "-i",
                    str(pod["name"]),
                    "--",
                    component_python("cpu"),
                    "-",
                    str(ports[app]),
                    stdin=script,
                    timeout=30,
                ).splitlines()[-1]
            )
            roles[app].append({**pod, **value})
    return roles


def build_plan(run_dir: Path, attempt: int, *, arguments: argparse.Namespace) -> dict:
    chain = chain_preflight(arguments, CASE_ID)
    topology = deployment_and_pdb_snapshot()
    replicas = declared_replicas(topology)
    ingress = ready_pods(INGRESS_APP)
    workers = ready_pods(WORKER_APP)
    if len(ingress) != replicas[INGRESS_APP] or len(workers) != replicas[WORKER_APP]:
        raise CaseError(
            f"unexpected control-plane baseline: ingress={len(ingress)}/"
            f"{replicas[INGRESS_APP]} workers={len(workers)}/{replicas[WORKER_APP]}"
        )
    if len(workers) < 2:
        raise CaseError("the plan deletes two control-worker Pods; fewer are Ready")
    if len(ingress) < 3:
        raise CaseError("deleting ingress requires at least three Ready replicas")
    minimum_ready = minimum_ready_from_replicas(replicas)
    limits = {
        app: failure_window_limit(topology["deployments"][app])
        for app in (INGRESS_APP, WORKER_APP)
    }
    role_errors = [*chain["errors"], *validate_roles(role_snapshot(), replicas)]
    plan = {
        "risk": "live-control-plane-pod-deletion",
        "mutation": (
            "delete one ingress and two control-worker Pods in three staged "
            "phases; Pod deletion only, with no Deployment spec or node "
            "change, and the ReplicaSets recreate the deleted Pods"
        ),
        "errors": role_errors,
        "chain": chain,
        "region": AWS_REGION,
        "cpu_kubeconfig": str(CPU_KUBECONFIG),
        "gpu_context": GPU_CONTEXT,
        "cluster_id": "production cluster token, custom owner only",
        "node_mutation": None,
        "maintenance_window_required_at_execute": True,
        "targets": [
            {"phase": "ingress-1", "app": INGRESS_APP, **ingress[0]},
            {"phase": "worker-1", "app": WORKER_APP, **workers[0]},
            {"phase": "worker-2", "app": WORKER_APP, **workers[1]},
        ],
        "replicas": replicas,
        "minimum_ready": minimum_ready,
        "failure_window_limits": limits,
        "baseline": {
            "ingress_pods": ingress,
            "worker_pods": workers,
            "queue": queue_stats(),
            **topology,
        },
        "load": {
            "baseline_seconds": BASELINE_SECONDS,
            "per_deletion_observation_cap_seconds": OBSERVATION_CAP_SECONDS,
            "settle_seconds": SETTLE_SECONDS,
            "claim_interval_seconds": PROBE_CYCLE_SECONDS,
            "health_interval_seconds": PROBE_CYCLE_SECONDS,
            "minimum_cycle_fraction": MINIMUM_CYCLE_FRACTION,
            "operation_owner": OWNER,
        },
        "stop_conditions": [
            f"fewer than {minimum_ready['ingress']} Ready ingress Pods or NLB endpoints",
            f"fewer than {minimum_ready['worker']} Ready control-worker Pods",
            "probe failure window exceeds the derived limit "
            + json.dumps({app: item["limit_seconds"] for app, item in limits.items()}),
            "HTTP 401, 403 or 500 observed by the GPU probe",
            "processor queue depth exceeds baseline by 25",
            "replacement Pod does not become Ready in the observation window",
        ],
        "rollback": {
            "pod_deletion_only": True,
            "replicaset_recreates_deleted_pods": True,
            "probe_active_deadline_seconds": 1800,
            "runner_waits_for_both_deployments": True,
            "no_deployment_spec_or_node_change": True,
        },
        "limitations": LIMITATIONS,
    }
    return guard_build_plan(
        run_dir=run_dir,
        case_id=CASE_ID,
        attempt=attempt,
        confirmation=CONFIRMATION,
        arguments=arguments,
        preflight_passed=not role_errors,
        environment=environment_values(),
        details=plan,
    )


def env_value(deployment: dict, name: str) -> str:
    for item in deployment["spec"]["template"]["spec"]["containers"][0].get(
        "env",
        [],
    ):
        if item.get("name") == name and item.get("value") is not None:
            return str(item["value"])
    raise CaseError(f"executor Deployment has no literal {name}")


def probe_manifest(deployment: dict) -> dict:
    template = deployment["spec"]["template"]["spec"]
    container = template["containers"][0]
    return {
        "apiVersion": "v1",
        "kind": "Pod",
        "metadata": {
            "name": PROBE_POD,
            "namespace": NAMESPACE,
            "labels": {"app": PROBE_POD},
        },
        "spec": {
            "restartPolicy": "Never",
            "activeDeadlineSeconds": 1800,
            "terminationGracePeriodSeconds": 5,
            "serviceAccountName": template.get("serviceAccountName"),
            "tolerations": template.get("tolerations", []),
            "nodeSelector": template.get("nodeSelector", {}),
            "affinity": template.get("affinity", {}),
            "containers": [
                {
                    "name": "probe",
                    "image": container["image"],
                    "command": [
                        component_python("gpu"),
                        f"/scripts/{PROBE_SCRIPT.name}",
                    ],
                    "resources": {
                        "requests": {"cpu": "50m", "memory": "64Mi"},
                        "limits": {"cpu": "250m", "memory": "256Mi"},
                    },
                    "env": [
                        {
                            "name": "GPU_FAULT_CONTROL_PLANE_URL",
                            "valueFrom": {
                                "secretKeyRef": {
                                    "name": "gpu-fault-regional-connection",
                                    "key": "control-plane-url",
                                }
                            },
                        },
                        {
                            "name": "GPU_FAULT_CONTROL_PLANE_TOKEN",
                            "valueFrom": {
                                "secretKeyRef": {
                                    "name": "gpu-fault-regional-connection",
                                    "key": "cluster-token",
                                }
                            },
                        },
                        {
                            "name": "GPU_FAULT_CLUSTER_ID",
                            "valueFrom": {
                                "secretKeyRef": {
                                    "name": "gpu-fault-regional-connection",
                                    "key": "cluster-id",
                                }
                            },
                        },
                        {
                            "name": "GPU_FAULT_CONTROL_PLANE_CA_FILE",
                            "value": "/tls/ca.crt",
                        },
                        {
                            "name": "GPU_FAULT_EXECUTOR_ARTIFACT_SHA256",
                            "value": env_value(
                                deployment,
                                "GPU_FAULT_EXECUTOR_ARTIFACT_SHA256",
                            ),
                        },
                        {
                            "name": "GPU_FAULT_EXECUTOR_COMPATIBILITY_DIGEST",
                            "value": env_value(
                                deployment,
                                "GPU_FAULT_EXECUTOR_COMPATIBILITY_DIGEST",
                            ),
                        },
                    ],
                    "volumeMounts": [
                        {
                            "name": "script",
                            "mountPath": "/scripts",
                            "readOnly": True,
                        },
                        {"name": "ca", "mountPath": "/tls", "readOnly": True},
                        {"name": "state", "mountPath": "/state"},
                    ],
                }
            ],
            "volumes": [
                {"name": "script", "configMap": {"name": CONFIGMAP}},
                {
                    "name": "ca",
                    "secret": {
                        "secretName": "gpu-fault-regional-connection",
                        "items": [{"key": "ca.crt", "path": "ca.crt"}],
                    },
                },
                {"name": "state", "emptyDir": {}},
            ],
        },
    }


def create_probe(resources: OwnedProbeResources) -> None:
    deployment = json.loads(
        gpu("get", "deployment", "gpu-fault-cluster-executor", "-o", "json")
    )
    configmap = {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": {"name": CONFIGMAP, "namespace": NAMESPACE},
        "data": {PROBE_SCRIPT.name: PROBE_SCRIPT.read_text()},
    }
    resources.create(configmap)
    resources.create(probe_manifest(deployment))
    gpu("wait", "--for=condition=Ready", f"pod/{PROBE_POD}", "--timeout=180s")
    wait_probe_file("/state/ready.json", 60)


def wait_probe_file(path: str, timeout_seconds: int) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        output = gpu(
            "exec",
            PROBE_POD,
            "--",
            "sh",
            "-c",
            f"if [ -f {path} ]; then printf present; fi",
            check=False,
        )
        if output.strip() == "present":
            return
        time.sleep(1)
    raise CaseError(f"probe did not create {path}")


def read_probe(path: str = "/state/stats.json") -> dict:
    return json.loads(gpu("exec", PROBE_POD, "--", "cat", path))


def control_sample(include_queue: bool) -> dict:
    pods = json.loads(
        cpu(
            "get",
            "pod",
            "-l",
            (
                "app in (gpu-fault-api-ha,gpu-fault-control-worker,"
                "gpu-fault-telemetry-spool-worker)"
            ),
            "-o",
            "json",
        )
    )
    counts = {"ingress_ready": 0, "worker_ready": 0, "spool_ready": 0}
    names = {"ingress_pods": [], "worker_pods": [], "spool_pods": []}
    ready_names = {item["name"] for item in ready_pod_records(pods)}
    for item in pods["items"]:
        ready = item["metadata"]["name"] in ready_names
        app = item.get("metadata", {}).get("labels", {}).get("app")
        if app == "gpu-fault-api-ha":
            names["ingress_pods"].append(item["metadata"]["name"])
            counts["ingress_ready"] += int(ready)
        elif app == "gpu-fault-control-worker":
            names["worker_pods"].append(item["metadata"]["name"])
            counts["worker_ready"] += int(ready)
        elif app == "gpu-fault-telemetry-spool-worker":
            names["spool_pods"].append(item["metadata"]["name"])
            counts["spool_ready"] += int(ready)
    slices = json.loads(
        cpu(
            "get",
            "endpointslice",
            "-l",
            "kubernetes.io/service-name=gpu-fault-api-nlb",
            "-o",
            "json",
        )
    )
    endpoint_ready = sum(
        1
        for item in slices.get("items", [])
        for endpoint in item.get("endpoints", [])
        if endpoint.get("conditions", {}).get("ready") is not False
    )
    value = {
        "observed_at": datetime.now(timezone.utc).isoformat(),
        **counts,
        "endpoint_ready": endpoint_ready,
        "ingress_pods": sorted(names["ingress_pods"]),
        "worker_pods": sorted(names["worker_pods"]),
        "spool_pods": sorted(names["spool_pods"]),
    }
    if include_queue:
        value["queue"] = queue_stats()
    return value


def observe_phase(
    label: str,
    duration_seconds: int,
    baseline_depth: int,
    *,
    minimum_ready: dict[str, int | None],
    max_failure_window_seconds: float,
    settled: Callable[[dict, dict | None], bool] | None = None,
    settle_seconds: int = SETTLE_SECONDS,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> dict:
    """Sample the control plane every ~2s until the phase ends.

    ``duration_seconds`` is the cap. With ``settled`` given, the phase ends
    earlier: once ``settled(sample, last_probe)`` has held continuously for
    ``settle_seconds`` the recovery is complete and further sampling only costs
    maintenance-window time. ``minimum_ready`` carries the quorum floors
    (``ingress``/``worker`` required, ``spool`` optional) and
    ``max_failure_window_seconds`` the derived per-Deployment limit; both come
    from the plan, not from constants.
    """

    started = clock()
    next_queue = 0.0
    next_probe = 0.0
    next_log = 0.0
    samples = []
    queue_samples = []
    probe_samples = []
    probe: dict | None = None
    settled_since: float | None = None
    settle_reached = False
    while clock() - started < duration_seconds:
        elapsed = clock() - started
        include_queue = elapsed >= next_queue
        sample = control_sample(include_queue)
        sample["elapsed_seconds"] = round(elapsed, 3)
        samples.append(sample)
        if include_queue:
            queue_samples.append(sample["queue"])
            next_queue = elapsed + 10
        if elapsed >= next_probe:
            probe = read_probe()
            probe_samples.append(probe)
            next_probe = elapsed + 10
            if (
                float(probe.get("current_failure_window_seconds", 0))
                > max_failure_window_seconds
            ):
                raise CaseError(f"{label}: probe failure window exceeded limit")
            forbidden = {
                key: value
                for key, value in probe.get("error_counts", {}).items()
                if key in {"http-401", "http-403", "http-500"} and int(value) > 0
            }
            if forbidden:
                raise CaseError(f"{label}: forbidden probe responses: {forbidden}")
        ingress_floor = int(minimum_ready["ingress"] or 0)
        worker_floor = int(minimum_ready["worker"] or 0)
        if (
            sample["ingress_ready"] < ingress_floor
            or sample["endpoint_ready"] < ingress_floor
        ):
            raise CaseError(f"{label}: ingress availability fell below {ingress_floor}")
        if sample["worker_ready"] < worker_floor:
            raise CaseError(f"{label}: worker availability fell below {worker_floor}")
        spool_floor = minimum_ready.get("spool")
        if spool_floor is not None and sample["spool_ready"] < int(spool_floor):
            raise CaseError(f"{label}: spool availability fell below its PDB limit")
        if include_queue and int(sample["queue"]["depth"]) > baseline_depth + 25:
            raise CaseError(f"{label}: processor queue exceeded stop threshold")
        if elapsed >= next_log:
            log(
                f"{label}: t={elapsed:.0f}s ingress={sample['ingress_ready']} "
                f"workers={sample['worker_ready']} spool={sample['spool_ready']} "
                f"endpoints={sample['endpoint_ready']}"
            )
            next_log = elapsed + 30
        if settled is not None:
            if settled(sample, probe):
                settled_since = elapsed if settled_since is None else settled_since
                if elapsed - settled_since >= settle_seconds:
                    settle_reached = True
                    break
            else:
                settled_since = None
        spent = clock() - started - elapsed
        sleep(max(0.0, PROBE_CYCLE_SECONDS - spent))
    final_probe = read_probe()
    return {
        "label": label,
        "duration_seconds": round(clock() - started, 3),
        "cap_seconds": duration_seconds,
        "settle_seconds": settle_seconds if settled is not None else None,
        "settle_reached": settle_reached if settled is not None else None,
        "samples": samples,
        "queue_samples": queue_samples,
        "probe_samples": probe_samples,
        "final_probe": final_probe,
        "summary": {
            "min_ingress_ready": min(item["ingress_ready"] for item in samples),
            "min_worker_ready": min(item["worker_ready"] for item in samples),
            "min_spool_ready": min(item["spool_ready"] for item in samples),
            "min_endpoint_ready": min(item["endpoint_ready"] for item in samples),
            "max_queue_depth": max(
                (int(item["depth"]) for item in queue_samples),
                default=baseline_depth,
            ),
            "final_queue_depth": (
                int(queue_samples[-1]["depth"]) if queue_samples else baseline_depth
            ),
            "max_probe_failure_window_seconds": float(
                final_probe.get("max_failure_window_seconds", 0)
            ),
        },
    }


def replacement_settled(
    target: dict,
    desired: int,
) -> Callable[[dict, dict | None], bool]:
    """Predicate: the deleted Pod is gone, its role is back to ``desired`` Ready
    Pods, and the probe currently reports no failure window."""

    role = "ingress" if target["app"] == INGRESS_APP else "worker"
    name = str(target["name"])

    def settled(sample: dict, probe: dict | None) -> bool:
        return (
            probe is not None
            and int(sample[f"{role}_ready"]) == desired
            and name not in sample[f"{role}_pods"]
            and float(probe.get("current_failure_window_seconds", 0)) == 0.0
        )

    return settled


def delete_and_observe(
    target: dict,
    baseline_names: set[str],
    baseline_depth: int,
    *,
    desired: int,
    minimum_ready: dict[str, int | None],
    max_failure_window_seconds: float,
) -> dict:
    current = {item["name"]: item for item in ready_pods(str(target["app"]))}
    observed = current.get(str(target["name"]))
    if observed is None or observed["uid"] != target["uid"]:
        raise CaseError(f"target changed before deletion: {target['name']}")
    requested_at = datetime.now(timezone.utc)
    log(f"deleting {target['app']} Pod {target['name']} on node {target['node']}")
    delete_pod(lambda args, body: cpu(*args, stdin=body), NAMESPACE, target)
    phase = observe_phase(
        str(target["phase"]),
        OBSERVATION_CAP_SECONDS,
        baseline_depth,
        minimum_ready=minimum_ready,
        max_failure_window_seconds=max_failure_window_seconds,
        settled=replacement_settled(target, desired),
    )
    current_ready = ready_pods(str(target["app"]))
    current_names = {item["name"] for item in current_ready}
    replacements = sorted(current_names - baseline_names)
    if str(target["name"]) in current_names:
        raise CaseError(f"deleted Pod still exists: {target['name']}")
    if len(current_ready) != desired:
        raise CaseError(
            f"{target['app']} did not recover desired Ready replicas: "
            f"{len(current_ready)} != {desired}"
        )
    if not replacements:
        raise CaseError(f"no replacement Pod observed for {target['name']}")
    return {
        "target": target,
        "requested_at": requested_at.isoformat(),
        "replacements": replacements,
        "observation": phase,
        "failure_window": {
            "measured_max_seconds": phase["summary"][
                "max_probe_failure_window_seconds"
            ],
            "limit_seconds": max_failure_window_seconds,
        },
    }


def seed_closure(run_id: str, cluster_id: str) -> dict:
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

run_id, cluster_id, owner, seed_json, raw_lease_seconds = sys.argv[1:]
lease_seconds = int(raw_lease_seconds)
seed = json.loads(seed_json)
incident_id = seed["incident_id"]
event_id = seed["event_id"]
workflow_id = seed["workflow_id"]
operations = [WorkflowOperation(value) for value in seed["operations"]]
steps = [
    WorkflowStepSpec(operation=operation, execution_owner=owner)
    for operation in operations
]
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
# A terminal seed would orphan its open commands. The foreign lease keeps
# the dispatcher from driving this simulated workflow before owned cleanup.
workflow = WorkflowRequest(
    request_id=workflow_id,
    incident_id=incident_id,
    status=WorkflowStatus.PENDING,
    official_action="NO_ACTION",
    fencing_token=1,
    official_steps=steps,
    execution_owner_id=f"{owner}-seed",
    execution_epoch=1,
    execution_lease_expires_at=datetime.now(timezone.utc) + timedelta(seconds=lease_seconds),
)
store = ApplicationContext.from_environment().store
store.save_incident(incident)
store.save_workflow(workflow)
for index, step in enumerate(steps):
    store.ensure_remote_command(RemoteActionCommand(
        command_id=seed["command_ids"][index],
        cluster_id=cluster_id,
        workflow_request_id=workflow_id,
        incident_id=incident_id,
        step_index=index,
        fencing_token=1,
        idempotency_key=f"{workflow_id}/{index}/{step.operation.value}",
        step=step,
        workflow=workflow,
        incident=incident,
    ))
print(json.dumps(seed, sort_keys=True))
"""
    expected = closure_seed(run_id)
    observed = cpu_python(
        script,
        run_id,
        cluster_id,
        OWNER,
        json.dumps(expected),
        str(SEED_LEASE_SECONDS),
        attempts=1,
    )
    if observed != expected:
        raise CaseError("synthetic closure seed acknowledgement identity changed")
    return observed


def closure_status(seed: dict) -> dict:
    """What the control plane recorded from the probe's claim/complete traffic.

    Read-only: the remote commands as the executor path left them, and the
    workflow record untouched. The runner never writes the workflow's status
    itself -- an assertion on a value the runner wrote would test the runner.
    """

    script = r"""
import json
import sys
from gpu_fault.app import ApplicationContext
workflow_id, *command_ids = sys.argv[1:]
store = ApplicationContext.from_environment().store
commands = [store.get_remote_command(value) for value in command_ids]
workflow = store.get_workflow(workflow_id)
print(json.dumps({
    "workflow_id": workflow.request_id,
    "workflow_status_observed": workflow.status.value,
    "commands": [
        {
            "command_id": item.command_id,
            "step_index": item.step_index,
            "operation": item.step.operation.value,
            "status": item.status.value,
            "lease_owner": item.lease_owner,
            "last_lease_owner": item.last_lease_owner,
            "result_details": item.result_details,
            "updated_at": item.updated_at.isoformat(),
        }
        for item in commands
    ],
}, sort_keys=True))
"""
    return cpu_python(
        script,
        str(seed["workflow_id"]),
        *[str(value) for value in seed["command_ids"]],
    )


def wait_closure(seed: dict, timeout_seconds: int = 120) -> dict:
    deadline = time.monotonic() + timeout_seconds
    last = {}
    while time.monotonic() < deadline:
        last = closure_status(seed)
        if all(item["status"] == "SUCCEEDED" for item in last.get("commands", [])):
            return last
        time.sleep(2)
    raise CaseError(f"synthetic closure did not complete: {last}")


def cleanup_closure(seed: dict) -> dict:
    script = r"""
import json
import sys
import psycopg

incident_id, event_id, workflow_id, *command_ids = sys.argv[1:]
deleted = {}
with psycopg.connect(store_dsn(), autocommit=True) as connection:
    cursor = connection.cursor()
    cursor.execute(
        "DELETE FROM gpu_fault_links "
        "WHERE kind='incident_by_event' AND key=%s AND value=%s",
        (event_id, incident_id),
    )
    deleted[f"incident_by_event/{event_id}"] = cursor.rowcount
    for kind, key in [
        *[("remote_command", value) for value in command_ids],
        ("workflow", workflow_id),
        ("incident", incident_id),
    ]:
        cursor.execute(
            "SELECT gpu_fault_delete_control_state(%s,%s)",
            (kind, key),
        )
        deleted[f"{kind}/{key}"] = int(cursor.fetchone()[0])
    cursor.execute(
        "SELECT count(*) FROM gpu_fault_control_records WHERE key LIKE %s",
        ("%ha001-%",),
    )
    objects = int(cursor.fetchone()[0])
    cursor.execute(
        "SELECT count(*) FROM gpu_fault_links "
        "WHERE key LIKE %s OR value LIKE %s",
        ("%ha001-%", "%ha001-%"),
    )
    links = int(cursor.fetchone()[0])
print(json.dumps({
    "deleted": deleted,
    "remaining_objects": objects,
    "remaining_links": links,
}, sort_keys=True))
"""
    result = cpu_python(
        STORE_DSN_SNIPPET + script,
        *[str(seed[key]) for key in ("incident_id", "event_id", "workflow_id")],
        *[str(value) for value in seed["command_ids"]],
        attempts=1,
    )
    if result["remaining_objects"] or result["remaining_links"]:
        raise CaseError(f"synthetic closure cleanup left residuals: {result}")
    return result


def validate_roles(snapshot: dict, replicas: dict[str, int]) -> list[str]:
    """Every replica in its declared role, active-active, with no leadership."""

    return validate_role_health(
        snapshot, replicas, ingress_app=INGRESS_APP, worker_app=WORKER_APP
    )


def verify_plan_targets(plan: dict) -> None:
    for target in plan["targets"]:
        current = {item["name"]: item for item in ready_pods(str(target["app"]))}
        item = current.get(str(target["name"]))
        if item is None or item["uid"] != target["uid"]:
            raise CaseError(f"planned target drifted: {target['name']}")


def database_residuals() -> dict:
    return cpu_python(
        STORE_DSN_SNIPPET
        + """
import json
import psycopg
with psycopg.connect(store_dsn()) as connection:
    cursor = connection.cursor()
    cursor.execute(
        "SELECT count(*) FROM gpu_fault_control_records WHERE key LIKE %s",
        ("%ha001-%",),
    )
    objects = int(cursor.fetchone()[0])
    cursor.execute(
        "SELECT count(*) FROM gpu_fault_links "
        "WHERE key LIKE %s OR value LIKE %s",
        ("%ha001-%", "%ha001-%"),
    )
    links = int(cursor.fetchone()[0])
print(json.dumps({"objects": objects, "links": links}, sort_keys=True))
"""
    )


def probe_resources() -> dict:
    resources = {}
    for kind, name in (("pod", PROBE_POD), ("configmap", CONFIGMAP)):
        output = gpu(
            "get",
            kind,
            name,
            "--ignore-not-found",
            "-o",
            "name",
        ).strip()
        resources[f"{kind}/{name}"] = bool(output)
    return {"count": sum(resources.values()), "resources": resources}


def cleanup_steps(
    *,
    probe_created: bool,
    seed: dict,
    case_dir: Path,
    result: dict,
    expected_replicas: dict[str, int],
    owned_resources: OwnedProbeResources | None = None,
) -> list[str]:
    """Run every cleanup step, each on its own, and return what failed.

    A kubectl timeout while stopping the probe used to skip the store cleanup
    and leave the synthetic incident, workflow and commands in the production
    store. Each step now runs regardless of the ones before it; the errors are
    collected and the verdict is FAIL if any step failed.
    """

    errors: list[str] = []

    def attempt(label: str, action: Callable[[], Any]) -> Any:
        try:
            return action()
        except Exception as exc:
            errors.append(f"{label}: {type(exc).__name__}: {exc}")
            return None

    if probe_created and owned_resources is None:
        errors.append("probe resources have no ownership receipt; preserving them")
    if probe_created and owned_resources is not None:
        attempt(
            "stop probe",
            lambda: gpu("exec", PROBE_POD, "--", "touch", "/state/stop", check=False),
        )
        attempt(
            "delete probe pod",
            lambda: owned_resources.delete("Pod", PROBE_POD),
        )
        attempt(
            "delete probe configmap",
            lambda: owned_resources.delete("ConfigMap", CONFIGMAP),
        )
    resources = (
        attempt("verify probe shutdown", probe_resources) if probe_created else None
    )
    if (
        seed
        and owned_resources is not None
        and resources is not None
        and resources["count"] == 0
    ):

        def closure() -> None:
            cleanup = cleanup_closure(seed)
            result["closure_cleanup"] = cleanup
            write_json_atomic(case_dir / "synthetic-closure-cleanup.json", cleanup)

        attempt("cleanup closure", closure)
    elif seed:
        errors.append("probe shutdown unverified; preserving synthetic closure")
    for app in (INGRESS_APP, WORKER_APP):
        attempt(
            f"rollout status {app}",
            functools.partial(
                cpu,
                "rollout",
                "status",
                f"deployment/{app}",
                "--timeout=600s",
                timeout=700,
            ),
        )

    def postflight() -> None:
        residuals = {
            "database": database_residuals(),
            "kubernetes": probe_resources(),
            "ready": {app: len(ready_pods(app)) for app in expected_replicas},
        }
        result["postflight"] = residuals
        write_json_atomic(case_dir / "postflight.json", residuals)
        if residuals["database"] != {"objects": 0, "links": 0}:
            raise CaseError(f"database residuals remain: {residuals}")
        if residuals["kubernetes"]["count"] != 0:
            raise CaseError(f"probe resources remain: {residuals}")
        if residuals["ready"] != expected_replicas:
            raise CaseError(
                f"Deployments did not return to declared replicas: {residuals}"
            )

    attempt("postflight", postflight)
    return errors


def execute_case(
    run_dir: Path,
    attempt: int,
    confirmation: str,
    *,
    maintenance_window_end: datetime | None = None,
    chain: dict | None = None,
) -> int:
    if confirmation != CONFIRMATION:
        raise CaseError(f"confirmation must be exactly {CONFIRMATION}")
    case_dir = run_dir / "cases" / CASE_ID
    plan_path = case_dir / "plan.json"
    if not plan_path.is_file():
        raise CaseError("run --plan before --execute")
    envelope = json.loads(plan_path.read_text())
    if int(envelope.get("attempt", -1)) != attempt:
        raise CaseError("plan attempt does not match execute attempt")
    plan = envelope.get("details") or {}
    if "replicas" not in plan or "failure_window_limits" not in plan:
        raise CaseError(
            "plan predates the derived replica/failure-window fields; re-plan"
        )
    verify_plan_targets(plan)
    if maintenance_window_end is None:
        raise CaseError("approved maintenance window is missing")
    require_window(maintenance_window_end)
    replicas = {app: int(value) for app, value in plan["replicas"].items()}
    minimum_ready = minimum_ready_from_replicas(replicas)
    limits = {
        app: float(item["limit_seconds"])
        for app, item in plan["failure_window_limits"].items()
    }
    baseline_depth = int(plan["baseline"]["queue"]["depth"])
    baseline_names = {
        INGRESS_APP: {item["name"] for item in plan["baseline"]["ingress_pods"]},
        WORKER_APP: {item["name"] for item in plan["baseline"]["worker_pods"]},
    }
    run_id = f"ha001-{run_dir.name.rsplit('-', 1)[-1].lower()}-a{attempt}"
    seed: dict = {}
    probe_created = False
    owned_resources = None
    result: dict = {
        "case_id": CASE_ID,
        "attempt": attempt,
        "verdict": "FAIL",
        "limitations": LIMITATIONS,
    }
    try:
        if database_residuals() != {"objects": 0, "links": 0}:
            raise CaseError("HA-001 database preflight found residuals")
        if probe_resources()["count"] != 0:
            raise CaseError("HA-001 probe resources already exist")
        owned_resources = OwnedProbeResources(
            case_dir / f"probe-resources-a{attempt}.json",
            lambda args, body: gpu(*args, stdin=body),
        )
        probe_created = True
        create_probe(owned_resources)
        wait_probe_file("/state/stats.json", 60)
        ready = read_probe("/state/ready.json")
        write_json_atomic(case_dir / "probe-ready.json", ready)
        roles_before = role_snapshot()
        write_json_atomic(case_dir / "roles-before.json", roles_before)
        role_errors = validate_roles(roles_before, replicas)
        if role_errors:
            raise CaseError(f"role preflight failed: {role_errors}")

        baseline_phase = observe_phase(
            "baseline",
            BASELINE_SECONDS,
            baseline_depth,
            minimum_ready=minimum_ready,
            max_failure_window_seconds=max(limits.values()),
        )
        write_json_atomic(case_dir / "baseline-timeline.json", baseline_phase)

        deletions: list[dict] = []

        def delete_target(target: dict) -> None:
            require_window(maintenance_window_end)
            app = str(target["app"])
            phase = delete_and_observe(
                target,
                baseline_names[app],
                baseline_depth,
                desired=replicas[app],
                minimum_ready=minimum_ready,
                max_failure_window_seconds=limits[app],
            )
            deletions.append(phase)
            write_json_atomic(case_dir / f"{target['phase']}-timeline.json", phase)

        for target in plan["targets"][:2]:
            delete_target(target)

        seed = closure_seed(run_id)
        write_json_atomic(case_dir / "synthetic-closure-seed.json", seed)
        seed_closure(run_id, str(ready["cluster_id"]))
        closure = wait_closure(seed)
        write_json_atomic(case_dir / "synthetic-closure-commands.json", closure)

        delete_target(plan["targets"][2])

        roles_after = role_snapshot()
        write_json_atomic(case_dir / "roles-after.json", roles_after)
        final_probe = read_probe()
        write_json_atomic(case_dir / "probe-final.json", final_probe)

        errors = [*validate_roles(roles_after, replicas)]
        observed_seconds = float(baseline_phase["duration_seconds"])
        for deletion in deletions:
            summary = deletion["observation"]["summary"]
            phase_label = deletion["target"]["phase"]
            observed_seconds += float(deletion["observation"]["duration_seconds"])
            if summary["min_ingress_ready"] < int(minimum_ready["ingress"] or 0):
                errors.append(f"{phase_label} lost ingress quorum")
            if summary["min_worker_ready"] < int(minimum_ready["worker"] or 0):
                errors.append(f"{phase_label} lost worker quorum")
            if summary["min_endpoint_ready"] < int(minimum_ready["ingress"] or 0):
                errors.append(f"{phase_label} lost NLB endpoints")
            if summary["final_queue_depth"] > baseline_depth + 5:
                errors.append(f"{phase_label} queue did not recover")
            window = deletion["failure_window"]
            if window["measured_max_seconds"] > window["limit_seconds"]:
                errors.append(f"{phase_label} failure window exceeded limit")
            if deletion["observation"].get("settle_reached") is False:
                errors.append(
                    f"{phase_label} did not settle inside the observation cap"
                )
        counters = final_probe.get("counters", {})
        minimum_cycles = minimum_probe_cycles(observed_seconds)
        if int(counters.get("claim_success", 0)) < minimum_cycles:
            errors.append("probe did not sustain enough claim cycles")
        if int(counters.get("health_success", 0)) < minimum_cycles:
            errors.append("probe did not sustain enough health cycles")
        if any(
            key in {"http-401", "http-403", "http-500"} and int(value) > 0
            for key, value in final_probe.get("error_counts", {}).items()
        ):
            errors.append("probe observed a forbidden HTTP response")
        summary = closure_summary(seed, closure, final_probe.get("ledger", {}))
        errors.extend(
            closure_errors(seed, summary, executor_id=str(ready.get("executor_id")))
        )
        result = {
            "case_id": CASE_ID,
            "attempt": attempt,
            "verdict": "PASS" if not errors else "FAIL",
            "errors": errors,
            "limitations": LIMITATIONS,
            "plan": plan,
            "replicas": replicas,
            "roles_before": roles_before,
            "roles_after": roles_after,
            "baseline_summary": baseline_phase["summary"],
            "failure_window_limits": plan["failure_window_limits"],
            "deletions": [
                {
                    "target": item["target"],
                    "requested_at": item["requested_at"],
                    "replacements": item["replacements"],
                    "summary": item["observation"]["summary"],
                    "failure_window": item["failure_window"],
                    "observation_seconds": item["observation"]["duration_seconds"],
                    "settle_reached": item["observation"]["settle_reached"],
                }
                for item in deletions
            ],
            "probe_cycles": {
                "observed_seconds": round(observed_seconds, 3),
                "minimum": minimum_cycles,
                "claim_success": int(counters.get("claim_success", 0)),
                "health_success": int(counters.get("health_success", 0)),
            },
            "synthetic_closure": summary,
            "probe_final": final_probe,
        }
    except ProcessSupervisionLost:
        record_supervision_loss(result)
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        cleanup_errors = run_cleanup(
            result,
            lambda: cleanup_steps(
                probe_created=probe_created,
                seed=seed,
                case_dir=case_dir,
                result=result,
                expected_replicas=replicas,
                owned_resources=owned_resources,
            ),
        )
        result["cleanup_errors"] = cleanup_errors
        if cleanup_errors:
            result["verdict"] = "FAIL"
    result.update(result_identity(chain))
    write_json_atomic(case_dir / f"{CASE_ID}.json", result)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["verdict"] == "PASS" else 1


def main() -> int:
    install_site_profile()
    parser = argparse.ArgumentParser()
    add_live_arguments(parser, confirmation=CONFIRMATION)
    parser.add_argument("--cpu-kubeconfig", default="")
    parser.add_argument("--gpu-kubeconfig", default="")
    parser.add_argument("--gpu-context", default="")
    parser.add_argument("--cluster-id", default="")
    parser.add_argument("--namespace", default="gpu-fault-system")
    parser.add_argument("--region", default="")
    args = parser.parse_args()
    os.umask(0o077)
    configure(args)
    install_abort_signals()
    if not args.execute:
        plan = build_plan(args.run_dir, args.attempt, arguments=args)
        print(json.dumps(plan, sort_keys=True))
        return 0 if plan["preflight_passed"] is True else 1
    deadline = guard_authorize_execution(
        args,
        case_id=CASE_ID,
        confirmation=CONFIRMATION,
        environment=environment_values(),
    )
    chain = chain_preflight(args, CASE_ID)
    plan = json.loads((args.run_dir / "cases" / CASE_ID / "plan.json").read_text())
    require_chain(plan["details"].get("chain", {}), chain)
    return execute_case(
        args.run_dir,
        args.attempt,
        args.confirm,
        maintenance_window_end=deadline,
        chain=chain,
    )


if __name__ == "__main__":
    raise SystemExit(run_case_main(main))

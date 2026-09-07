#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

if __package__:
    from .acceptance_scope import current_acceptance_scope, scoped_case_evidence
    from .live_driver_guard import (
        add_live_arguments,
        applied_site_profile,
        install_site_profile,
    )
    from .live_driver_guard import (
        authorize_execution as guard_authorize_execution,
    )
    from .regional_live_fixture import (
        install_abort_signals,
        run_case_main,
    )
else:
    from acceptance_scope import current_acceptance_scope, scoped_case_evidence
    from live_driver_guard import (
        add_live_arguments,
        applied_site_profile,
        install_site_profile,
    )
    from live_driver_guard import (
        authorize_execution as guard_authorize_execution,
    )
    from regional_live_fixture import (
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
CONFIRMATION = "HA001_DELETE_CONTROL_PODS"
OBSERVATION_SECONDS = 300
BASELINE_SECONDS = 30
MAX_FAILURE_WINDOW_SECONDS = 55


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


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(scoped_case_evidence(value), indent=2, sort_keys=True) + "\n"
    )
    path.chmod(0o600)


def run(
    argv: list[str],
    *,
    stdin: str | None = None,
    check: bool = True,
    timeout: int = 300,
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        argv,
        input=stdin,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
    )
    if check and result.returncode != 0:
        raise CaseError(
            f"command failed ({result.returncode}): {' '.join(argv)}; "
            f"stderr={result.stderr.strip()}"
        )
    return result


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
    result = []
    for item in value.get("items", []):
        statuses = item.get("status", {}).get("containerStatuses", [])
        if item.get("status", {}).get("phase") != "Running":
            continue
        if not statuses or not all(bool(status.get("ready")) for status in statuses):
            continue
        result.append(
            {
                "name": item["metadata"]["name"],
                "uid": item["metadata"]["uid"],
                "node": item["spec"].get("nodeName"),
                "started_at": item.get("status", {}).get("startTime"),
            }
        )
    return sorted(result, key=lambda item: item["name"])


def first_ready_cpu_pod(app: str) -> str:
    pods = ready_pods(app)
    if not pods:
        raise CaseError(f"no Ready CPU Pod for app={app}")
    return str(pods[0]["name"])


def cpu_python(script: str, *arguments: str) -> dict:
    last_error: Exception | None = None
    for _attempt in range(3):
        pod = first_ready_cpu_pod("gpu-fault-api-ha")
        try:
            output = cpu(
                "exec",
                "-i",
                pod,
                "--",
                "python3",
                "-",
                *arguments,
                stdin=script,
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


def deployment_and_pdb_snapshot() -> dict:
    deployments = json.loads(
        cpu(
            "get",
            "deployment",
            "gpu-fault-api-ha",
            "gpu-fault-control-worker",
            "gpu-fault-telemetry-spool-worker",
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
                    "python3",
                    "-",
                    str(ports[app]),
                    stdin=script,
                    timeout=30,
                ).splitlines()[-1]
            )
            roles[app].append({**pod, **value})
    return roles


def build_plan(run_dir: Path, attempt: int) -> dict:
    ingress = ready_pods("gpu-fault-api-ha")
    workers = ready_pods("gpu-fault-control-worker")
    if len(ingress) != 3 or len(workers) != 6:
        raise CaseError(
            f"unexpected control-plane baseline: ingress={len(ingress)} "
            f"workers={len(workers)}"
        )
    scope = current_acceptance_scope()
    plan = {
        "schema_version": 2,
        "case_id": CASE_ID,
        "attempt": attempt,
        "confirmation": CONFIRMATION,
        "environment": environment_values(),
        "site_profile": applied_site_profile(),
        **scope.plan_fields(),
        "mutation_performed": False,
        "region": AWS_REGION,
        "cpu_kubeconfig": str(CPU_KUBECONFIG),
        "gpu_context": GPU_CONTEXT,
        "cluster_id": "production cluster token, custom owner only",
        "node_mutation": None,
        "maintenance_window_required_at_execute": True,
        "targets": [
            {"phase": "ingress-1", "app": "gpu-fault-api-ha", **ingress[0]},
            {
                "phase": "worker-1",
                "app": "gpu-fault-control-worker",
                **workers[0],
            },
            {
                "phase": "worker-2",
                "app": "gpu-fault-control-worker",
                **workers[1],
            },
        ],
        "baseline": {
            "ingress_pods": ingress,
            "worker_pods": workers,
            "queue": queue_stats(),
            **deployment_and_pdb_snapshot(),
        },
        "load": {
            "baseline_seconds": BASELINE_SECONDS,
            "per_deletion_observation_seconds": OBSERVATION_SECONDS,
            "claim_interval_seconds": 2,
            "health_interval_seconds": 2,
            "operation_owner": OWNER,
        },
        "stop_conditions": [
            "fewer than two Ready ingress Pods or NLB endpoints",
            "fewer than five Ready control-worker Pods",
            "probe failure window exceeds 55 seconds",
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
    }
    case_dir = run_dir / "cases" / CASE_ID
    case_dir.mkdir(parents=True, exist_ok=True)
    write_json(case_dir / "plan.json", plan)
    return plan


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
                        "/opt/gpu-fault/executor/bin/python",
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


def create_probe() -> None:
    deployment = json.loads(
        gpu("get", "deployment", "gpu-fault-cluster-executor", "-o", "json")
    )
    configmap = {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": {"name": CONFIGMAP, "namespace": NAMESPACE},
        "data": {PROBE_SCRIPT.name: PROBE_SCRIPT.read_text()},
    }
    gpu("apply", "-f", "-", stdin=json.dumps(configmap))
    gpu("delete", "pod", PROBE_POD, "--ignore-not-found", check=False)
    gpu("apply", "-f", "-", stdin=json.dumps(probe_manifest(deployment)))
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
    for item in pods.get("items", []):
        statuses = item.get("status", {}).get("containerStatuses", [])
        ready = (
            item.get("status", {}).get("phase") == "Running"
            and bool(statuses)
            and all(bool(status.get("ready")) for status in statuses)
        )
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
    minimum_spool_ready: int | None = None,
) -> dict:
    started = time.monotonic()
    next_queue = 0.0
    next_probe = 0.0
    next_log = 0.0
    samples = []
    queue_samples = []
    probe_samples = []
    while time.monotonic() - started < duration_seconds:
        elapsed = time.monotonic() - started
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
                > MAX_FAILURE_WINDOW_SECONDS
            ):
                raise CaseError(f"{label}: probe failure window exceeded limit")
            forbidden = {
                key: value
                for key, value in probe.get("error_counts", {}).items()
                if key in {"http-401", "http-403", "http-500"} and int(value) > 0
            }
            if forbidden:
                raise CaseError(f"{label}: forbidden probe responses: {forbidden}")
        if sample["ingress_ready"] < 2 or sample["endpoint_ready"] < 2:
            raise CaseError(f"{label}: ingress availability fell below two")
        if sample["worker_ready"] < 5:
            raise CaseError(f"{label}: worker availability fell below five")
        if (
            minimum_spool_ready is not None
            and sample["spool_ready"] < minimum_spool_ready
        ):
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
        spent = time.monotonic() - started - elapsed
        time.sleep(max(0.0, 2.0 - spent))
    final_probe = read_probe()
    return {
        "label": label,
        "duration_seconds": round(time.monotonic() - started, 3),
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


def delete_and_observe(
    target: dict,
    baseline_names: set[str],
    baseline_depth: int,
) -> dict:
    current = {item["name"]: item for item in ready_pods(str(target["app"]))}
    observed = current.get(str(target["name"]))
    if observed is None or observed["uid"] != target["uid"]:
        raise CaseError(f"target changed before deletion: {target['name']}")
    requested_at = datetime.now(timezone.utc)
    log(f"deleting {target['app']} Pod {target['name']} on node {target['node']}")
    cpu("delete", "pod", str(target["name"]), "--wait=false")
    phase = observe_phase(
        str(target["phase"]),
        OBSERVATION_SECONDS,
        baseline_depth,
    )
    current_ready = ready_pods(str(target["app"]))
    current_names = {item["name"] for item in current_ready}
    replacements = sorted(current_names - baseline_names)
    desired = 3 if target["app"] == "gpu-fault-api-ha" else 6
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
    }


def seed_closure(run_id: str, cluster_id: str) -> dict:
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
workflow_id = f"workflow-{run_id}"
operations = [
    WorkflowOperation.FREEZE_EVIDENCE,
    WorkflowOperation.STOP_WORKLOADS,
    WorkflowOperation.RESTART_WORKLOAD,
]
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
workflow = WorkflowRequest(
    request_id=workflow_id,
    incident_id=incident_id,
    status=WorkflowStatus.BLOCKED,
    official_action="NO_ACTION",
    fencing_token=1,
    official_steps=steps,
    blocked_reasons=["synthetic HA-001 executor-only closure"],
)
store = ApplicationContext.from_environment().store
store.save_incident(incident)
store.save_workflow(workflow)
command_ids = []
for index, step in enumerate(steps):
    command_id = f"remote-{run_id}-{index}"
    command_ids.append(command_id)
    store.ensure_remote_command(RemoteActionCommand(
        command_id=command_id,
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
print(json.dumps({
    "incident_id": incident_id,
    "event_id": event_id,
    "workflow_id": workflow_id,
    "command_ids": command_ids,
    "operations": [item.value for item in operations],
}, sort_keys=True))
"""
    return cpu_python(script, run_id, cluster_id, OWNER)


def closure_status(seed: dict) -> dict:
    script = r"""
import json
import sys
from gpu_fault.app import ApplicationContext
store = ApplicationContext.from_environment().store
commands = [store.get_remote_command(value) for value in sys.argv[1:]]
print(json.dumps({
    "commands": [
        {
            "command_id": item.command_id,
            "step_index": item.step_index,
            "operation": item.step.operation.value,
            "status": item.status.value,
            "result_details": item.result_details,
        }
        for item in commands
    ],
}, sort_keys=True))
"""
    return cpu_python(script, *[str(value) for value in seed["command_ids"]])


def wait_closure(seed: dict, timeout_seconds: int = 120) -> dict:
    deadline = time.monotonic() + timeout_seconds
    last = {}
    while time.monotonic() < deadline:
        last = closure_status(seed)
        if all(item["status"] == "SUCCEEDED" for item in last.get("commands", [])):
            return last
        time.sleep(2)
    raise CaseError(f"synthetic closure did not complete: {last}")


def finalize_closure(seed: dict) -> dict:
    script = r"""
import json
import sys
from gpu_fault.app import ApplicationContext
from gpu_fault.models import (
    WorkflowStatus,
    WorkflowStepExecution,
    WorkflowStepStatus,
)

workflow_id, *command_ids = sys.argv[1:]
store = ApplicationContext.from_environment().store
workflow = store.get_workflow(workflow_id)
commands = [store.get_remote_command(value) for value in command_ids]
if any(item.status.value != "SUCCEEDED" for item in commands):
    raise RuntimeError("not all synthetic closure commands succeeded")
executions = [
    WorkflowStepExecution(
        step_index=item.step_index,
        operation=item.step.operation,
        status=WorkflowStepStatus.SUCCEEDED,
        adapter_operation_id=f"remote/{item.command_id}",
        details=item.result_details,
    )
    for item in commands
]
workflow = workflow.model_copy(update={
    "status": WorkflowStatus.SUCCEEDED,
    "blocked_reasons": [],
    "completed_operations": [item.step.operation for item in commands],
    "completed_step_indexes": [item.step_index for item in commands],
    "step_executions": executions,
})
store.save_workflow(workflow)
counts = {}
for execution in workflow.step_executions:
    key = str(execution.step_index)
    counts[key] = counts.get(key, 0) + int(
        execution.status is WorkflowStepStatus.SUCCEEDED
    )
print(json.dumps({
    "workflow_id": workflow.request_id,
    "status": workflow.status.value,
    "succeeded_by_step_index": counts,
    "operations": [item.value for item in workflow.completed_operations],
}, sort_keys=True))
"""
    return cpu_python(
        script,
        str(seed["workflow_id"]),
        *[str(value) for value in seed["command_ids"]],
    )


def cleanup_closure(seed: dict) -> dict:
    script = r"""
import json
import os
import sys
import psycopg

incident_id, event_id, workflow_id, *command_ids = sys.argv[1:]
deleted = {}
with psycopg.connect(os.environ["GPU_FAULT_STORE_URL"], autocommit=True) as connection:
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
            "DELETE FROM gpu_fault_objects WHERE kind=%s AND key=%s",
            (kind, key),
        )
        deleted[f"{kind}/{key}"] = cursor.rowcount
    cursor.execute(
        "SELECT count(*) FROM gpu_fault_objects WHERE key LIKE %s",
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
        script,
        str(seed["incident_id"]),
        str(seed["event_id"]),
        str(seed["workflow_id"]),
        *[str(value) for value in seed["command_ids"]],
    )
    if result["remaining_objects"] or result["remaining_links"]:
        raise CaseError(f"synthetic closure cleanup left residuals: {result}")
    return result


def validate_roles(snapshot: dict) -> list[str]:
    errors = []
    ingress = snapshot.get("gpu-fault-api-ha", [])
    workers = snapshot.get("gpu-fault-control-worker", [])
    if len(ingress) != 3:
        errors.append("ingress role snapshot does not contain three Pods")
    if len(workers) != 6:
        errors.append("worker role snapshot does not contain six Pods")
    for item in ingress:
        health = item.get("health", {})
        if health.get("service_role") != "ingress":
            errors.append(f"{item['name']} is not ingress")
        if health.get("processor_role") != "inactive":
            errors.append(f"{item['name']} processor is not inactive")
        if item.get("processor_active_consumer") != 0.0:
            errors.append(f"{item['name']} active-consumer metric is not zero")
    for item in workers:
        health = item.get("health", {})
        if health.get("service_role") != "worker":
            errors.append(f"{item['name']} is not worker")
        if health.get("processor_role") != "active-consumer":
            errors.append(f"{item['name']} processor is not active-consumer")
        if item.get("processor_active_consumer") != 1.0:
            errors.append(f"{item['name']} active-consumer metric is not one")
    return errors


def verify_plan_targets(plan: dict) -> None:
    for target in plan["targets"]:
        current = {item["name"]: item for item in ready_pods(str(target["app"]))}
        item = current.get(str(target["name"]))
        if item is None or item["uid"] != target["uid"]:
            raise CaseError(f"planned target drifted: {target['name']}")


def database_residuals() -> dict:
    return cpu_python(
        """
import json
import os
import psycopg
with psycopg.connect(os.environ["GPU_FAULT_STORE_URL"]) as connection:
    cursor = connection.cursor()
    cursor.execute(
        "SELECT count(*) FROM gpu_fault_objects WHERE key LIKE %s",
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
            check=False,
        ).strip()
        resources[f"{kind}/{name}"] = bool(output)
    return {"count": sum(resources.values()), "resources": resources}


def execute_case(run_dir: Path, attempt: int, confirmation: str) -> int:
    if confirmation != CONFIRMATION:
        raise CaseError(f"confirmation must be exactly {CONFIRMATION}")
    case_dir = run_dir / "cases" / CASE_ID
    plan_path = case_dir / "plan.json"
    if not plan_path.is_file():
        raise CaseError("run --plan before --execute")
    plan = json.loads(plan_path.read_text())
    if int(plan.get("attempt", -1)) != attempt:
        raise CaseError("plan attempt does not match execute attempt")
    verify_plan_targets(plan)
    baseline_depth = int(plan["baseline"]["queue"]["depth"])
    baseline_names = {
        "gpu-fault-api-ha": {item["name"] for item in plan["baseline"]["ingress_pods"]},
        "gpu-fault-control-worker": {
            item["name"] for item in plan["baseline"]["worker_pods"]
        },
    }
    run_id = f"ha001-{run_dir.name.rsplit('-', 1)[-1].lower()}-a{attempt}"
    seed: dict = {}
    probe_created = False
    result: dict = {
        "case_id": CASE_ID,
        "attempt": attempt,
        "verdict": "FAIL",
    }
    try:
        if database_residuals() != {"objects": 0, "links": 0}:
            raise CaseError("HA-001 database preflight found residuals")
        create_probe()
        probe_created = True
        wait_probe_file("/state/stats.json", 60)
        ready = read_probe("/state/ready.json")
        write_json(case_dir / "probe-ready.json", ready)
        roles_before = role_snapshot()
        write_json(case_dir / "roles-before.json", roles_before)
        role_errors = validate_roles(roles_before)
        if role_errors:
            raise CaseError(f"role preflight failed: {role_errors}")

        baseline_phase = observe_phase(
            "baseline",
            BASELINE_SECONDS,
            baseline_depth,
        )
        write_json(case_dir / "baseline-timeline.json", baseline_phase)

        deletions = []
        for target in plan["targets"][:2]:
            phase = delete_and_observe(
                target,
                baseline_names[str(target["app"])],
                baseline_depth,
            )
            deletions.append(phase)
            write_json(
                case_dir / f"{target['phase']}-timeline.json",
                phase,
            )

        seed = seed_closure(run_id, str(ready["cluster_id"]))
        write_json(case_dir / "synthetic-closure-seed.json", seed)
        commands = wait_closure(seed)
        write_json(case_dir / "synthetic-closure-commands.json", commands)
        closure = finalize_closure(seed)
        write_json(case_dir / "synthetic-closure.json", closure)

        target = plan["targets"][2]
        phase = delete_and_observe(
            target,
            baseline_names[str(target["app"])],
            baseline_depth,
        )
        deletions.append(phase)
        write_json(case_dir / f"{target['phase']}-timeline.json", phase)

        roles_after = role_snapshot()
        write_json(case_dir / "roles-after.json", roles_after)
        final_probe = read_probe()
        write_json(case_dir / "probe-final.json", final_probe)

        errors = [*validate_roles(roles_after)]
        for deletion in deletions:
            summary = deletion["observation"]["summary"]
            if summary["min_ingress_ready"] < 2:
                errors.append(f"{deletion['target']['phase']} lost ingress quorum")
            if summary["min_worker_ready"] < 5:
                errors.append(f"{deletion['target']['phase']} lost worker quorum")
            if summary["min_endpoint_ready"] < 2:
                errors.append(f"{deletion['target']['phase']} lost NLB endpoints")
            if summary["final_queue_depth"] > baseline_depth + 5:
                errors.append(f"{deletion['target']['phase']} queue did not recover")
            if summary["max_probe_failure_window_seconds"] > MAX_FAILURE_WINDOW_SECONDS:
                errors.append(
                    f"{deletion['target']['phase']} failure window exceeded limit"
                )
        counters = final_probe.get("counters", {})
        if int(counters.get("claim_success", 0)) < 400:
            errors.append("probe did not sustain enough claim cycles")
        if int(counters.get("health_success", 0)) < 400:
            errors.append("probe did not sustain enough health cycles")
        if any(
            key in {"http-401", "http-403", "http-500"} and int(value) > 0
            for key, value in final_probe.get("error_counts", {}).items()
        ):
            errors.append("probe observed a forbidden HTTP response")
        ledger = final_probe.get("ledger", {})
        if ledger.get("physical_count") != 3:
            errors.append("synthetic closure physical count is not three")
        if ledger.get("operations") != seed.get("operations"):
            errors.append("synthetic closure operations are incomplete or reordered")
        if closure.get("status") != "SUCCEEDED":
            errors.append("synthetic closure workflow is not SUCCEEDED")
        if closure.get("succeeded_by_step_index") != {"0": 1, "1": 1, "2": 1}:
            errors.append("synthetic closure contains duplicate or missing steps")
        result = {
            "case_id": CASE_ID,
            "attempt": attempt,
            "verdict": "PASS" if not errors else "FAIL",
            "errors": errors,
            "plan": plan,
            "roles_before": roles_before,
            "roles_after": roles_after,
            "baseline_summary": baseline_phase["summary"],
            "deletions": [
                {
                    "target": item["target"],
                    "requested_at": item["requested_at"],
                    "replacements": item["replacements"],
                    "summary": item["observation"]["summary"],
                }
                for item in deletions
            ],
            "synthetic_closure": closure,
            "probe_final": final_probe,
        }
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if probe_created:
            gpu(
                "exec",
                PROBE_POD,
                "--",
                "touch",
                "/state/stop",
                check=False,
            )
        gpu("delete", "pod", PROBE_POD, "--ignore-not-found", check=False)
        gpu(
            "delete",
            "configmap",
            CONFIGMAP,
            "--ignore-not-found",
            check=False,
        )
        if seed:
            try:
                cleanup = cleanup_closure(seed)
                result["closure_cleanup"] = cleanup
                write_json(case_dir / "synthetic-closure-cleanup.json", cleanup)
            except Exception as exc:
                result["cleanup_error"] = f"{type(exc).__name__}: {exc}"
                result["verdict"] = "FAIL"
        try:
            cpu(
                "rollout",
                "status",
                "deployment/gpu-fault-api-ha",
                "--timeout=600s",
                timeout=700,
            )
            cpu(
                "rollout",
                "status",
                "deployment/gpu-fault-control-worker",
                "--timeout=600s",
                timeout=700,
            )
            residuals = {
                "database": database_residuals(),
                "kubernetes": probe_resources(),
            }
            result["postflight"] = residuals
            write_json(case_dir / "postflight.json", residuals)
            if residuals["database"] != {"objects": 0, "links": 0}:
                raise CaseError(f"database residuals remain: {residuals}")
            if residuals["kubernetes"]["count"] != 0:
                raise CaseError(f"probe resources remain: {residuals}")
        except Exception as exc:
            result["postflight_error"] = f"{type(exc).__name__}: {exc}"
            result["verdict"] = "FAIL"
    write_json(case_dir / f"{CASE_ID}.json", result)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["verdict"] == "PASS" else 1


def main() -> int:
    install_site_profile()
    parser = argparse.ArgumentParser()
    add_live_arguments(parser, confirmation=CONFIRMATION)
    parser.add_argument("--cpu-kubeconfig", default="")
    parser.add_argument("--gpu-kubeconfig", default="")
    parser.add_argument("--gpu-context", default="")
    parser.add_argument("--namespace", default="gpu-fault-system")
    parser.add_argument("--region", default="")
    args = parser.parse_args()
    os.umask(0o077)
    configure(args)
    install_abort_signals()
    if not args.execute:
        plan = build_plan(args.run_dir, args.attempt)
        print(json.dumps(plan, sort_keys=True))
        return 0
    deadline = guard_authorize_execution(
        args,
        case_id=CASE_ID,
        confirmation=CONFIRMATION,
        environment=environment_values(),
    )
    plan_path = args.run_dir / "cases" / CASE_ID / "plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    plan["maintenance_window_end"] = deadline.isoformat()
    write_json(plan_path, plan)
    return execute_case(args.run_dir, args.attempt, args.confirm)


if __name__ == "__main__":
    raise SystemExit(run_case_main(main))

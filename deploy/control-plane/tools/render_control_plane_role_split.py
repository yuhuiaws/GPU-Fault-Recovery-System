#!/usr/bin/env python3
"""Render the role-split control-plane Deployments from the repo.

Input is the temporary kustomize build created by
render-control-plane-role-split.sh (or, with --json, a single Deployment
as JSON, which is what the unit test feeds it).
Output is the ingress, queue-worker and telemetry-spool-worker
Deployments plus their PDBs: either written to --out-dir as the
checked-in manifests, or printed as a v1/List on stdout.

This used to read the live Deployment and pipe the result straight into
kubectl apply. Everything about the worker tier then lived only in the
cluster, so a rebuilt control plane silently came up single-role - every
request queued and nothing claimed it - and there was no file to review
or diff. render-control-plane-role-split.sh regenerates the manifests
and tests/regional/test_control_plane_role_split.py fails when they drift.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import yaml


# Everything that sizes the processor's own thread pools. These belong
# to whichever tier claims from the queue; on an ingress-only replica
# they are inert. verify_control_plane_role_split.py
# fails the deploy if any of them reappears on the ingress Deployment,
# because `kubectl set env` on both tiers is how they got there.
PROCESSOR_POOL_ENV = (
    "GPU_FAULT_PROCESSOR_WORKERS",
    "GPU_FAULT_PROCESSOR_COMPLETION_CLUSTER_CONCURRENCY",
    "GPU_FAULT_PROCESSOR_FAULT_WORKERS",
    "GPU_FAULT_PROCESSOR_FAULT_PRESSURE_EVIDENCE_WORKERS",
    "GPU_FAULT_PROCESSOR_OBSERVATION_WORKERS",
    "GPU_FAULT_PROCESSOR_GPU_TELEMETRY_WORKERS",
    "GPU_FAULT_PROCESSOR_HOST_TELEMETRY_WORKERS",
)

SENSITIVE_ENV = re.compile(r"(?:SECRET|TOKEN|PASSWORD|CREDENTIAL|PRIVATE_KEY)")


def config_domain(name: str) -> str:
    domains = (
        (
            "postgres",
            ("GPU_FAULT_POSTGRES_", "GPU_FAULT_STORE_"),
        ),
        (
            "processor",
            (
                "GPU_FAULT_PROCESSOR_",
                "GPU_FAULT_INGRESS_",
                "GPU_FAULT_FAULT_",
                "GPU_FAULT_EVIDENCE_",
            ),
        ),
        (
            "notification",
            (
                "GPU_FAULT_NOTIFICATION_",
                "GPU_FAULT_SES_",
                "GPU_FAULT_EMAIL_",
            ),
        ),
        (
            "telemetry",
            (
                "GPU_FAULT_TELEMETRY_",
                "GPU_FAULT_GPU_",
                "GPU_FAULT_HOST_",
                "GPU_FAULT_EFA_",
                "GPU_FAULT_DCGM_",
                "GPU_FAULT_TRAINING_",
                "GPU_FAULT_PCIE_",
                "GPU_FAULT_NVLINK_",
                "GPU_FAULT_THERMAL_",
                "GPU_FAULT_POWER_",
                "GPU_FAULT_MEMORY_",
            ),
        ),
        (
            "recovery",
            (
                "GPU_FAULT_HYPERPOD_",
                "GPU_FAULT_AGENT_",
                "GPU_FAULT_NODE_ACTION_",
                "GPU_FAULT_WORKFLOW_",
                "GPU_FAULT_RESTART_",
                "GPU_FAULT_SXID_",
                "GPU_FAULT_HUNG_",
                "GPU_FAULT_QUIESCE_",
            ),
        ),
    )
    for domain, prefixes in domains:
        if name.startswith(prefixes):
            return domain
    return "core"


def externalize_literal_env(
    container: dict,
    *,
    role: str,
    namespace: str,
) -> list[dict]:
    retained = []
    data_by_domain: dict[str, dict[str, str]] = defaultdict(dict)
    for item in container.get("env", []):
        name = item.get("name")
        if not name:
            raise ValueError(f"{role} has an env entry without a name")
        has_value = "value" in item
        has_source = "valueFrom" in item
        if has_value == has_source:
            raise ValueError(
                f"{role} env {name} must define exactly one of value or valueFrom"
            )
        if has_source:
            retained.append(item)
            continue
        if SENSITIVE_ENV.search(name):
            raise ValueError(f"{role} sensitive env {name} must use valueFrom")
        data_by_domain[config_domain(name)][name] = str(item["value"])
    container["env"] = retained
    config_maps = []
    env_from = []
    for domain, data in sorted(data_by_domain.items()):
        name = f"{role}-config-{domain}"
        config_maps.append(
            {
                "apiVersion": "v1",
                "kind": "ConfigMap",
                "metadata": {
                    "name": name,
                    "namespace": namespace,
                },
                "data": dict(sorted(data.items())),
            }
        )
        env_from.append({"configMapRef": {"name": name}})
    container["envFrom"] = env_from
    return config_maps


def set_env(container: dict, name: str, value: str) -> None:
    env = container.setdefault("env", [])
    for item in env:
        if item.get("name") == name:
            item.clear()
            item.update({"name": name, "value": value})
            return
    env.append({"name": name, "value": value})


def unset_env(container: dict, name: str) -> None:
    env = container.get("env")
    if not env:
        return
    container["env"] = [item for item in env if item.get("name") != name]


def configure_processor_retry(container: dict) -> None:
    set_env(container, "GPU_FAULT_PROCESSOR_RETRY_BACKOFF_SECONDS", "1")
    set_env(container, "GPU_FAULT_PROCESSOR_RETRY_BACKOFF_MAX_SECONDS", "30")


def configure_worker_queue_coordination(worker: dict) -> None:
    set_env(worker, "GPU_FAULT_PROCESSOR_NOTIFICATION_FALLBACK_SECONDS", "5")
    set_env(worker, "GPU_FAULT_PROCESSOR_NOTIFICATION_SHARDS", "24")
    set_env(worker, "GPU_FAULT_PROCESSOR_COMPLETION_CLUSTER_CONCURRENCY", "1")
    set_env(worker, "GPU_FAULT_PROCESSOR_ROUTINE_STARVATION_SECONDS", "30")
    set_env(worker, "GPU_FAULT_PROCESSOR_FAULT_PRESSURE_EVIDENCE_WORKERS", "1")


def replace_uvicorn_args(
    container: dict,
    *,
    port: int,
    workers: int,
    backlog: int,
    ingress: bool,
) -> None:
    args = container.get("args")
    if not args or len(args) != 1:
        raise ValueError("API container must use one shell args entry")
    command = args[0]
    marker = "exec uvicorn gpu_fault.app:create_app --factory"
    if marker not in command:
        raise ValueError("API container does not launch gpu_fault.app:create_app")
    prefix = command.split(marker, 1)[0].replace(
        "[collectors,postgres]",
        "[collectors,postgres,performance]",
    )
    flags = (
        f"{marker} --host 0.0.0.0 --port {port} "
        f"--workers {workers} --loop uvloop --http httptools "
        f"--backlog {backlog} --timeout-keep-alive 5 "
        "--no-proxy-headers "
        "--timeout-graceful-shutdown 60 "
        "--timeout-worker-healthcheck 20 --no-access-log"
    )
    if ingress:
        flags += " --limit-concurrency 4096 --limit-max-requests 20000 --limit-max-requests-jitter 2000"
    container["args"] = [prefix + flags]


def set_probe_port(container: dict, port: int) -> None:
    for declared in container.get("ports", []):
        if declared.get("name") == "http":
            declared["containerPort"] = port
    for name in (
        "startupProbe",
        "readinessProbe",
        "livenessProbe",
    ):
        probe = container.get(name)
        if probe and "httpGet" in probe:
            probe["httpGet"]["port"] = port
        if probe and "tcpSocket" in probe:
            probe["tcpSocket"]["port"] = port


def read_source(as_json: bool) -> dict:
    """Pull the gpu-fault-api-ha Deployment out of the input."""

    raw = sys.stdin.read()
    if as_json:
        source = json.loads(raw)
    else:
        source = next(
            (
                document
                for document in yaml.safe_load_all(raw)
                if isinstance(document, dict)
                and document.get("kind") == "Deployment"
                and document["metadata"]["name"] == "gpu-fault-api-ha"
            ),
            None,
        )
        if source is None:
            raise SystemExit("no gpu-fault-api-ha Deployment in the kustomize output")
    source["spec"]["template"]["spec"]["enableServiceLinks"] = False
    return source


HEADER = """\
# Generated by deploy/control-plane/tools/render-control-plane-role-split.sh -
# do not edit. Change deploy/control-plane/base/control-plane-deployment.yaml
# or deploy/control-plane/regional/regional-control-plane-patch.yaml and
# re-render.
"""


def consumer_copy(source: dict, role: str) -> dict:
    name, order = {
        "control": ("gpu-fault-control-worker", 10),
        "spool": ("gpu-fault-telemetry-spool-worker", 20),
    }[role]
    resource = copy.deepcopy(source)
    resource["metadata"]["name"] = name
    annotations = resource["metadata"].setdefault("annotations", {})
    annotations["gpu-fault.io/cleanup-phase"] = "consumer"
    annotations["gpu-fault.io/cleanup-order"] = str(order)
    return resource


def pod_disruption_budget(
    name: str,
    namespace: str,
    app: str,
    *,
    min_available: int | None = None,
    max_unavailable: int | None = None,
) -> dict:
    availability = (
        {"minAvailable": min_available}
        if min_available is not None
        else {"maxUnavailable": max_unavailable}
    )
    return {
        "apiVersion": "policy/v1",
        "kind": "PodDisruptionBudget",
        "metadata": {"name": name, "namespace": namespace},
        "spec": {
            **availability,
            "selector": {"matchLabels": {"app": app}},
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir")
    parser.add_argument("--json", action="store_true")
    options = parser.parse_args()

    source = read_source(options.json)
    deployment = copy.deepcopy(source)
    pod_spec = deployment["spec"]["template"]["spec"]
    containers = pod_spec["containers"]
    api = next(item for item in containers if item["name"] == "api")
    base = copy.deepcopy(api)
    configure_processor_retry(base)

    ingress = copy.deepcopy(base)
    ingress["name"] = "api"
    set_env(ingress, "GPU_FAULT_SERVICE_ROLE", "ingress")
    set_env(
        ingress,
        "GPU_FAULT_LIFESPAN_SHUTDOWN_MAX_SECONDS",
        "20",
    )
    # Dead config on this tier, so it is not carried here. The four
    # processor pools and the lane count that sizes their defaults are
    # only ever created inside ProcessorCoordinator.run_processor, and
    # that thread does not start when GPU_FAULT_SERVICE_ROLE=ingress
    # (background_services_enabled is false). Left in place they cost
    # nothing at runtime but they are read as real capacity: the ingress
    # replicas exported gpu_fault_processor_workers 24 while claiming
    # nothing, and a thread/connection budget that adds
    # 24 + 4 + 4 + 8 + 8 per ingress process is counting threads the
    # process never creates. The worker tier sets all five below.
    for consumer_only in PROCESSOR_POOL_ENV:
        unset_env(ingress, consumer_only)
    # The 40-connection process budget is split into three explicit
    # admission lanes: 24 general API, eight striped cross-cluster spool
    # batches and eight fault. The pool pre-opens sixteen connections,
    # enough for every spool and fault lane, while general traffic grows
    # into the remaining headroom on demand.
    set_env(ingress, "GPU_FAULT_POSTGRES_POOL_MIN_SIZE", "2")
    set_env(ingress, "GPU_FAULT_POSTGRES_POOL_MAX_SIZE", "40")
    set_env(ingress, "GPU_FAULT_STORE_IO_WORKERS", "28")
    set_env(ingress, "GPU_FAULT_STORE_IO_MAX_IN_FLIGHT", "1024")
    set_env(ingress, "GPU_FAULT_FAULT_STORE_IO_WORKERS", "8")
    set_env(
        ingress,
        "GPU_FAULT_EVIDENCE_STORE_IO_WORKERS",
        "4",
    )
    set_env(
        ingress,
        "GPU_FAULT_EVIDENCE_STORE_IO_MAX_IN_FLIGHT",
        "256",
    )
    set_env(
        ingress,
        "GPU_FAULT_TELEMETRY_SPOOL_STORE_IO_WORKERS",
        "8",
    )
    set_env(
        ingress,
        "GPU_FAULT_TELEMETRY_SPOOL_STORE_IO_MAX_IN_FLIGHT",
        "256",
    )
    set_env(ingress, "GPU_FAULT_FAULT_DECODE_WORKERS", "8")
    set_env(
        ingress,
        "GPU_FAULT_STORE_IO_ADMISSION_TIMEOUT_SECONDS",
        "30",
    )
    set_env(ingress, "GPU_FAULT_INGRESS_DECODE_WORKERS", "32")
    set_env(
        ingress,
        "GPU_FAULT_INGRESS_DECODE_MAX_IN_FLIGHT",
        "1024",
    )
    set_env(
        ingress,
        "GPU_FAULT_INGRESS_DECODE_TIMEOUT_SECONDS",
        "30",
    )
    set_env(
        ingress,
        "GPU_FAULT_PROCESSOR_ADMISSION_BATCH_SIZE",
        "64",
    )
    set_env(
        ingress,
        "GPU_FAULT_PROCESSOR_ADMISSION_BATCH_DELAY_SECONDS",
        "0.002",
    )
    set_env(
        ingress,
        "GPU_FAULT_PROCESSOR_FAULT_ADMISSION_BATCH_SIZE",
        "64",
    )
    set_env(
        ingress,
        "GPU_FAULT_PROCESSOR_FAULT_ADMISSION_BATCH_GROUPS",
        "8",
    )
    set_env(
        ingress,
        "GPU_FAULT_PROCESSOR_FAULT_ADMISSION_BATCH_DELAY_SECONDS",
        "0.002",
    )
    set_env(
        ingress,
        "GPU_FAULT_PROCESSOR_FAULT_ADMISSION_PROJECTION_MARGIN",
        "0",
    )
    set_env(
        ingress,
        "GPU_FAULT_PROCESSOR_EVIDENCE_ADMISSION_BATCH_SIZE",
        "32",
    )
    set_env(
        ingress,
        "GPU_FAULT_PROCESSOR_EVIDENCE_ADMISSION_BATCH_GROUPS",
        "4",
    )
    set_env(
        ingress,
        "GPU_FAULT_PROCESSOR_EVIDENCE_ADMISSION_BATCH_DELAY_SECONDS",
        "0.002",
    )
    set_env(
        ingress,
        "GPU_FAULT_PROCESSOR_EVIDENCE_ADMISSION_PROJECTION_MARGIN",
        "0",
    )
    set_env(
        ingress,
        "GPU_FAULT_TELEMETRY_REQUEST_BUDGET_SECONDS",
        "30",
    )
    set_env(ingress, "GPU_FAULT_TELEMETRY_SPOOL", "false")
    set_env(
        ingress,
        "GPU_FAULT_TELEMETRY_SPOOL_MAX_DEPTH",
        "65536",
    )
    set_env(
        ingress,
        "GPU_FAULT_TELEMETRY_SPOOL_MAX_CLUSTER_DEPTH",
        "1024",
    )
    set_env(
        ingress,
        "GPU_FAULT_TELEMETRY_SPOOL_MAX_ITEM_BYTES",
        str(4 * 1024 * 1024),
    )
    set_env(
        ingress,
        "GPU_FAULT_TELEMETRY_SPOOL_BATCH_SIZE",
        "64",
    )
    set_env(
        ingress,
        "GPU_FAULT_TELEMETRY_SPOOL_ADMISSION_PARTITIONS",
        "8",
    )
    set_env(
        ingress,
        "GPU_FAULT_TELEMETRY_SPOOL_BATCH_GROUPS",
        "8",
    )
    set_env(
        ingress,
        "GPU_FAULT_TELEMETRY_SPOOL_BATCH_DELAY_SECONDS",
        "0.01",
    )
    set_env(
        ingress,
        "GPU_FAULT_INGRESS_FAULT_CONCURRENCY",
        "256",
    )
    set_env(
        ingress,
        "GPU_FAULT_INGRESS_NORMAL_CONCURRENCY",
        "1000",
    )
    set_env(
        ingress,
        "GPU_FAULT_INGRESS_FAULT_WAIT_SECONDS",
        "30",
    )
    set_env(
        ingress,
        "GPU_FAULT_INGRESS_NORMAL_WAIT_SECONDS",
        "2",
    )
    replace_uvicorn_args(
        ingress,
        port=8080,
        workers=4,
        backlog=8192,
        ingress=True,
    )
    set_probe_port(ingress, 8080)
    # The control-plane EKS nodes do not allow these as pod-level unsafe
    # sysctls. deploy/control-plane/regional/control-plane-sysctl.yaml
    # applies the host values and re-applies them whenever a control-plane
    # node restarts.
    pod_spec.setdefault("securityContext", {}).pop("sysctls", None)

    pod_spec["containers"] = [ingress]
    ingress_pdb = pod_disruption_budget(
        "gpu-fault-api-ha-pdb",
        deployment["metadata"].get("namespace", "gpu-fault-system"),
        "gpu-fault-api-ha",
        min_available=2,
    )

    worker_deployment = consumer_copy(source, "control")
    worker_deployment["spec"]["replicas"] = int(
        os.getenv("GPU_FAULT_CONTROL_WORKER_REPLICAS", "6")
    )
    worker_deployment["spec"]["selector"]["matchLabels"] = {
        "app": "gpu-fault-control-worker"
    }
    worker_template = worker_deployment["spec"]["template"]
    worker_template["metadata"]["labels"] = {"app": "gpu-fault-control-worker"}
    worker_template["metadata"].setdefault("annotations", {})["prometheus.io/port"] = (
        "8081"
    )
    worker_spec = worker_template["spec"]
    worker_spec["terminationGracePeriodSeconds"] = 240
    worker = copy.deepcopy(base)
    worker["name"] = "control-worker"
    set_env(worker, "GPU_FAULT_SERVICE_ROLE", "worker")
    set_env(
        worker,
        "GPU_FAULT_PROCESSOR_LOCAL_URL",
        "http://127.0.0.1:8081",
    )
    set_env(
        worker,
        "GPU_FAULT_PROCESSOR_REQUEST_MAX_EXECUTION_SECONDS",
        "120",
    )
    set_env(
        worker,
        "GPU_FAULT_PROCESSOR_REQUEST_LEASE_SECONDS",
        "150",
    )
    set_env(
        worker,
        "GPU_FAULT_PROCESSOR_REQUEST_RENEW_SECONDS",
        "10",
    )
    set_env(
        worker,
        "GPU_FAULT_LIFESPAN_SHUTDOWN_MAX_SECONDS",
        "130",
    )
    set_env(
        worker,
        "GPU_FAULT_PROCESSOR_GPU_INVENTORY_STALE_SECONDS",
        "180",
    )
    set_env(
        worker,
        "GPU_FAULT_PROCESSOR_HEALTH_SUMMARY_STALE_SECONDS",
        "420",
    )
    set_env(
        worker,
        "GPU_FAULT_PROCESSOR_OBSERVATION_STALE_SECONDS",
        "120",
    )
    set_env(
        worker,
        "GPU_FAULT_PROCESSOR_TRAINING_PROGRESS_STALE_SECONDS",
        "120",
    )
    set_env(
        worker,
        "GPU_FAULT_FAULT_ACTION_MAX_AGE_SECONDS",
        "900",
    )
    # Four processes, not one. The consumer is Python: every claimed
    # request is replayed into this same container over loopback and
    # executed by interpreter code, so one process tops out at one core
    # no matter how many pool threads it has. A 50-cluster burst left
    # all six worker pods pinned at 0.88-0.94 cores of their 4-core
    # limit with nr_throttled=0 - the GIL, not the cgroup, was the
    # ceiling, and the fault backlog drained at a flat ~7 rows/s.
    # Per-process pools are divided so the per-pod totals stay close to
    # what one process used to hold. The lane count is set here rather
    # than inherited from the base manifest: this is the only tier the
    # pools exist on, so the value has to be visible next to the four
    # pools it sizes the defaults for.
    set_env(worker, "GPU_FAULT_PROCESSOR_WORKERS", "24")
    set_env(worker, "GPU_FAULT_PROCESSOR_FAULT_WORKERS", "4")
    set_env(
        worker,
        "GPU_FAULT_PROCESSOR_FAULT_IDLE_BACKOFF_MAX_SECONDS",
        "0.5",
    )
    set_env(
        worker,
        "GPU_FAULT_PROCESSOR_FAULT_BUSY_BACKOFF_MAX_SECONDS",
        "0.1",
    )
    configure_worker_queue_coordination(worker)
    set_env(
        worker,
        "GPU_FAULT_PROCESSOR_THREAD_DUMP_SIGNAL",
        "SIGUSR2",
    )
    set_env(
        worker,
        "GPU_FAULT_PROCESSOR_OBSERVATION_WORKERS",
        "2",
    )
    set_env(
        worker,
        "GPU_FAULT_PROCESSOR_GPU_TELEMETRY_WORKERS",
        "4",
    )
    set_env(
        worker,
        "GPU_FAULT_PROCESSOR_HOST_TELEMETRY_WORKERS",
        "4",
    )
    set_env(worker, "GPU_FAULT_POSTGRES_POOL_MAX_SIZE", "24")
    set_env(worker, "GPU_FAULT_STORE_IO_WORKERS", "8")
    set_env(worker, "GPU_FAULT_STORE_IO_MAX_IN_FLIGHT", "48")
    set_env(
        worker,
        "GPU_FAULT_STORE_IO_ADMISSION_TIMEOUT_SECONDS",
        "5",
    )
    set_env(worker, "GPU_FAULT_INGRESS_DECODE_WORKERS", "4")
    set_env(
        worker,
        "GPU_FAULT_INGRESS_DECODE_MAX_IN_FLIGHT",
        "32",
    )
    set_env(
        worker,
        "GPU_FAULT_INGRESS_DECODE_TIMEOUT_SECONDS",
        "5",
    )
    replace_uvicorn_args(
        worker,
        port=8081,
        workers=4,
        backlog=1024,
        ingress=False,
    )
    set_probe_port(worker, 8081)
    worker["resources"] = {
        "requests": {"cpu": "3", "memory": "4Gi"},
        "limits": {"cpu": "4", "memory": "6Gi"},
    }
    worker_spec["containers"] = [worker]
    worker_spec["affinity"] = {
        "podAntiAffinity": {
            "preferredDuringSchedulingIgnoredDuringExecution": [
                {
                    "weight": 100,
                    "podAffinityTerm": {
                        "topologyKey": "kubernetes.io/hostname",
                        "labelSelector": {
                            "matchLabels": {"app": "gpu-fault-control-worker"}
                        },
                    },
                }
            ]
        }
    }
    worker_spec["topologySpreadConstraints"] = [
        {
            "maxSkew": 1,
            "topologyKey": "kubernetes.io/hostname",
            "whenUnsatisfiable": "DoNotSchedule",
            "matchLabelKeys": ["pod-template-hash"],
            "labelSelector": {"matchLabels": {"app": "gpu-fault-control-worker"}},
        }
    ]
    worker_pdb = pod_disruption_budget(
        "gpu-fault-control-worker-pdb",
        worker_deployment["metadata"].get("namespace", "gpu-fault-system"),
        "gpu-fault-control-worker",
        max_unavailable=1,
    )
    spool_deployment = consumer_copy(source, "spool")
    spool_deployment["spec"]["replicas"] = int(
        os.getenv("GPU_FAULT_TELEMETRY_SPOOL_REPLICAS", "0")
    )
    spool_deployment["spec"]["selector"]["matchLabels"] = {
        "app": "gpu-fault-telemetry-spool-worker"
    }
    spool_template = spool_deployment["spec"]["template"]
    spool_template["metadata"]["labels"] = {"app": "gpu-fault-telemetry-spool-worker"}
    spool_template["metadata"].setdefault("annotations", {})["prometheus.io/port"] = (
        "8082"
    )
    spool_spec = spool_template["spec"]
    spool_spec["terminationGracePeriodSeconds"] = 130
    spool_worker = copy.deepcopy(base)
    spool_worker["name"] = "telemetry-spool-worker"
    for consumer_only in PROCESSOR_POOL_ENV:
        unset_env(spool_worker, consumer_only)
    set_env(
        spool_worker,
        "GPU_FAULT_SERVICE_ROLE",
        "spool-worker",
    )
    set_env(
        spool_worker,
        "GPU_FAULT_PROCESSOR_LOCAL_URL",
        "http://127.0.0.1:8082",
    )
    set_env(
        spool_worker,
        "GPU_FAULT_LIFESPAN_SHUTDOWN_MAX_SECONDS",
        "30",
    )
    set_env(
        spool_worker,
        "GPU_FAULT_TELEMETRY_SPOOL",
        "true",
    )
    set_env(
        spool_worker,
        "GPU_FAULT_TELEMETRY_SPOOL_WORKERS",
        "8",
    )
    set_env(
        spool_worker,
        "GPU_FAULT_TELEMETRY_SPOOL_FAULT_PRESSURE_WORKERS",
        "1",
    )
    set_env(
        spool_worker,
        "GPU_FAULT_TELEMETRY_SPOOL_FAULT_PRESSURE_POLL_SECONDS",
        "0.5",
    )
    set_env(
        spool_worker,
        "GPU_FAULT_TELEMETRY_SPOOL_MAX_ITEM_BYTES",
        str(4 * 1024 * 1024),
    )
    set_env(
        spool_worker,
        "GPU_FAULT_TELEMETRY_SPOOL_MAX_IN_FLIGHT_BYTES",
        str(64 * 1024 * 1024),
    )
    set_env(
        spool_worker,
        "GPU_FAULT_TELEMETRY_SPOOL_REPLAY_BATCH_MAX_ITEMS",
        "64",
    )
    set_env(
        spool_worker,
        "GPU_FAULT_TELEMETRY_SPOOL_REPLAY_BATCH_MAX_BYTES",
        str(8 * 1024 * 1024),
    )
    set_env(
        spool_worker,
        "GPU_FAULT_TELEMETRY_SPOOL_LEASE_SECONDS",
        "60",
    )
    set_env(
        spool_worker,
        "GPU_FAULT_TELEMETRY_SPOOL_RETRY_BACKOFF_SECONDS",
        "1",
    )
    set_env(
        spool_worker,
        "GPU_FAULT_TELEMETRY_SPOOL_NOTIFICATION_FALLBACK_SECONDS",
        "5",
    )
    set_env(
        spool_worker,
        "GPU_FAULT_ENABLE_HYPERPOD_MANAGED_OBSERVER",
        "false",
    )
    set_env(
        spool_worker,
        "GPU_FAULT_ENABLE_WORKFLOW_DISPATCHER",
        "false",
    )
    set_env(
        spool_worker,
        "GPU_FAULT_POSTGRES_POOL_MAX_SIZE",
        "12",
    )
    set_env(
        spool_worker,
        "GPU_FAULT_POSTGRES_STATEMENT_TIMEOUT_SECONDS",
        "20",
    )
    set_env(spool_worker, "GPU_FAULT_STORE_IO_WORKERS", "8")
    set_env(
        spool_worker,
        "GPU_FAULT_STORE_IO_MAX_IN_FLIGHT",
        "32",
    )
    set_env(
        spool_worker,
        "GPU_FAULT_STORE_IO_ADMISSION_TIMEOUT_SECONDS",
        "5",
    )
    set_env(
        spool_worker,
        "GPU_FAULT_INGRESS_DECODE_WORKERS",
        "4",
    )
    set_env(
        spool_worker,
        "GPU_FAULT_INGRESS_DECODE_MAX_IN_FLIGHT",
        "32",
    )
    replace_uvicorn_args(
        spool_worker,
        port=8082,
        workers=1,
        backlog=1024,
        ingress=False,
    )
    set_probe_port(spool_worker, 8082)
    spool_worker["resources"] = {
        "requests": {"cpu": "1", "memory": "2Gi"},
        "limits": {"cpu": "4", "memory": "4Gi"},
    }
    spool_spec["containers"] = [spool_worker]
    spool_spec["affinity"] = {
        "podAntiAffinity": {
            "preferredDuringSchedulingIgnoredDuringExecution": [
                {
                    "weight": 100,
                    "podAffinityTerm": {
                        "topologyKey": "kubernetes.io/hostname",
                        "labelSelector": {
                            "matchLabels": {"app": ("gpu-fault-telemetry-spool-worker")}
                        },
                    },
                }
            ]
        }
    }
    spool_spec["topologySpreadConstraints"] = [
        {
            "maxSkew": 1,
            "topologyKey": "kubernetes.io/hostname",
            "whenUnsatisfiable": "DoNotSchedule",
            "matchLabelKeys": ["pod-template-hash"],
            "labelSelector": {
                "matchLabels": {"app": "gpu-fault-telemetry-spool-worker"}
            },
        }
    ]
    spool_pdb = pod_disruption_budget(
        "gpu-fault-telemetry-spool-worker-pdb",
        spool_deployment["metadata"].get("namespace", "gpu-fault-system"),
        "gpu-fault-telemetry-spool-worker",
        max_unavailable=1,
    )

    for item in (
        deployment,
        worker_deployment,
        spool_deployment,
    ):
        item["metadata"].pop("resourceVersion", None)
        item["metadata"].pop("uid", None)
        item["metadata"].pop("generation", None)
        item["metadata"].pop("creationTimestamp", None)
        item["metadata"].pop("managedFields", None)
        item["metadata"].pop("ownerReferences", None)
        item.pop("status", None)
    namespace = deployment["metadata"].get("namespace", "gpu-fault-system")
    config_maps = [
        *externalize_literal_env(
            ingress,
            role="gpu-fault-api-ha",
            namespace=namespace,
        ),
        *externalize_literal_env(
            worker,
            role="gpu-fault-control-worker",
            namespace=namespace,
        ),
        *externalize_literal_env(
            spool_worker,
            role="gpu-fault-telemetry-spool-worker",
            namespace=namespace,
        ),
    ]
    if not options.out_dir:
        print(
            json.dumps(
                {
                    "apiVersion": "v1",
                    "kind": "List",
                    "items": [
                        *config_maps,
                        deployment,
                        ingress_pdb,
                        worker_deployment,
                        worker_pdb,
                        spool_deployment,
                        spool_pdb,
                    ],
                }
            )
        )
        return
    out_dir = Path(options.out_dir)
    for stale in out_dir.glob("gpu-fault-*.yaml"):
        stale.unlink()
    written: list[str] = []
    for item in config_maps:
        filename = item["metadata"]["name"] + ".yaml"
        (out_dir / filename).write_text(
            HEADER + yaml.safe_dump(item, sort_keys=False, width=72),
            encoding="utf-8",
        )
        written.append(filename)
    for item, filename in (
        (deployment, "gpu-fault-api-ha-ingress.yaml"),
        (ingress_pdb, "gpu-fault-api-ha-pdb.yaml"),
        (
            worker_deployment,
            "gpu-fault-control-worker.yaml",
        ),
        (
            worker_pdb,
            "gpu-fault-control-worker-pdb.yaml",
        ),
        (
            spool_deployment,
            "gpu-fault-telemetry-spool-worker.yaml",
        ),
        (
            spool_pdb,
            "gpu-fault-telemetry-spool-worker-pdb.yaml",
        ),
    ):
        (out_dir / filename).write_text(
            HEADER + yaml.safe_dump(item, sort_keys=False, width=72),
            encoding="utf-8",
        )
        written.append(filename)
    (out_dir / "manifest-list.txt").write_text(
        "\n".join(sorted(written)) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

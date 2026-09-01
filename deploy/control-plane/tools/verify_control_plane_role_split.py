#!/usr/bin/env python3
"""Post-deploy self-check for the regional role split.

Checks the live control plane against the two properties that decide
whether the split works at all, neither of which shows up in pod status:

  * exactly one tier serves ingress and one tier claims from the queue,
    with the worker tier scaled above zero. A control plane with no
    worker still returns 202 for every request and simply never
    processes any of them;
  * each tier's uvicorn command matches the tier - ingress on 8080 with
    several workers, worker on 8081 with one, and no request-count worker
    recycling on any tier. Ingress recycling removes serving capacity
    during bursts; worker recycling can strand leases;
  * the ingress tier carries no processor pool sizing. Those pools are
    created by run_processor, which never starts on an ingress replica,
    so a value there is inert - and inert config is worse than absent
    config here, because it is what capacity math and
    gpu_fault_processor_workers read as real threads. `kubectl set env`
    on both Deployments is how it gets there.

Exits non-zero with the reason so a deploy fails loudly.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

NAMESPACE = os.getenv("GPU_FAULT_NAMESPACE", "gpu-fault-system")
_CONFIG_MAP_CACHE: dict[str, dict[str, str]] = {}

# Kept in step with PROCESSOR_POOL_ENV in
# render_control_plane_role_split.py.
PROCESSOR_POOL_ENV = (
    "GPU_FAULT_PROCESSOR_WORKERS",
    "GPU_FAULT_PROCESSOR_FAULT_WORKERS",
    "GPU_FAULT_PROCESSOR_FAULT_PRESSURE_EVIDENCE_WORKERS",
    "GPU_FAULT_PROCESSOR_OBSERVATION_WORKERS",
    "GPU_FAULT_PROCESSOR_GPU_TELEMETRY_WORKERS",
    "GPU_FAULT_PROCESSOR_HOST_TELEMETRY_WORKERS",
)


def deployment(name: str) -> dict | None:
    result = subprocess.run(
        [
            "kubectl",
            "-n",
            NAMESPACE,
            "get",
            "deployment",
            name,
            "-o",
            "json",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    return json.loads(result.stdout)


def container(item: dict, name: str) -> dict:
    containers = item["spec"]["template"]["spec"]["containers"]
    for candidate in containers:
        if candidate["name"] == name:
            return candidate
    raise SystemExit(f"{item['metadata']['name']} has no {name} container")


def config_map_data(name: str) -> dict[str, str]:
    if name in _CONFIG_MAP_CACHE:
        return _CONFIG_MAP_CACHE[name]
    result = subprocess.run(
        [
            "kubectl",
            "-n",
            NAMESPACE,
            "get",
            "configmap",
            name,
            "-o",
            "json",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise SystemExit(f"ConfigMap {name} is missing")
    data = json.loads(result.stdout).get("data") or {}
    _CONFIG_MAP_CACHE[name] = data
    return data


def env_values(item: dict) -> dict[str, str | None]:
    values: dict[str, str | None] = {}
    for source in item.get("envFrom") or []:
        reference = source.get("configMapRef")
        if reference and reference.get("name"):
            values.update(config_map_data(reference["name"]))
    for entry in item.get("env") or []:
        name = entry.get("name")
        if not name:
            continue
        if "value" in entry:
            values[name] = entry["value"]
            continue
        reference = entry.get("valueFrom", {}).get("configMapKeyRef")
        if reference and reference.get("name"):
            values[name] = config_map_data(reference["name"]).get(reference.get("key"))
        else:
            values[name] = None
    return values


def env_value(item: dict, name: str) -> str | None:
    return env_values(item).get(name)


def env_names(item: dict) -> set[str]:
    return set(env_values(item))


def reject_request_count_recycling(
    problems: list[str],
    deployment_name: str,
    command: str,
) -> None:
    if "--limit-max-requests" in command:
        consequence = (
            "recycle removes ingress capacity"
            if deployment_name == "gpu-fault-api-ha"
            else "mid-claim recycle strands leases"
        )
        problems.append(
            f"{deployment_name} recycles uvicorn workers by request count; "
            + consequence
        )


def main() -> int:
    problems: list[str] = []
    expected_runtime_image = os.getenv("GPU_FAULT_RUNTIME_IMAGE")

    ingress = deployment("gpu-fault-api-ha")
    worker = deployment("gpu-fault-control-worker")
    spool = deployment("gpu-fault-telemetry-spool-worker")
    if ingress is None:
        problems.append("gpu-fault-api-ha is missing")
    if worker is None:
        problems.append(
            "gpu-fault-control-worker is missing: nothing claims from the processor queue"
        )
    if spool is None:
        problems.append(
            "gpu-fault-telemetry-spool-worker is missing: routine "
            "telemetry is admitted to the spool but nobody drains it"
        )
    if problems:
        for problem in problems:
            print(f"role-split check failed: {problem}")
        return 1

    api = container(ingress, "api")
    if expected_runtime_image and api.get("image") != expected_runtime_image:
        problems.append(
            "gpu-fault-api-ha runtime image does not match GPU_FAULT_RUNTIME_IMAGE"
        )
    if env_value(api, "GPU_FAULT_SERVICE_ROLE") != "ingress":
        problems.append("gpu-fault-api-ha is not GPU_FAULT_SERVICE_ROLE=ingress")
    spool_admission = env_value(api, "GPU_FAULT_TELEMETRY_SPOOL")
    if spool_admission not in {"true", "false"}:
        problems.append(
            "gpu-fault-api-ha must explicitly set GPU_FAULT_TELEMETRY_SPOOL=true or false"
        )
    try:
        ingress_pool = int(env_value(api, "GPU_FAULT_POSTGRES_POOL_MAX_SIZE") or "0")
        ingress_general_io = int(env_value(api, "GPU_FAULT_STORE_IO_WORKERS") or "0")
        ingress_fault_io = int(
            env_value(api, "GPU_FAULT_FAULT_STORE_IO_WORKERS") or "0"
        )
        ingress_evidence_io = int(
            env_value(api, "GPU_FAULT_EVIDENCE_STORE_IO_WORKERS") or "0"
        )
        ingress_spool_io = int(
            env_value(
                api,
                "GPU_FAULT_TELEMETRY_SPOOL_STORE_IO_WORKERS",
            )
            or "0"
        )
    except ValueError:
        problems.append("ingress PostgreSQL and Store I/O limits must be integers")
    else:
        required_pool = (
            ingress_general_io
            + ingress_fault_io
            + ingress_evidence_io
            + (ingress_spool_io if spool_admission == "true" else 0)
        )
        if (
            min(
                ingress_pool,
                ingress_general_io,
                ingress_fault_io,
                ingress_evidence_io,
                ingress_spool_io,
            )
            <= 0
            or ingress_pool < required_pool
        ):
            problems.append(
                "ingress PostgreSQL pool does not cover its general, "
                "fault, evidence and telemetry-spool Store I/O workers: "
                f"pool={ingress_pool}, required={required_pool}"
            )
    command = api["args"][0]
    if "--port 8080" not in command:
        problems.append("gpu-fault-api-ha does not serve on 8080")
    if "--workers 4" not in command:
        problems.append("gpu-fault-api-ha lost its uvicorn worker count")
    reject_request_count_recycling(problems, "gpu-fault-api-ha", command)
    inert = sorted(env_names(api) & set(PROCESSOR_POOL_ENV))
    if inert:
        problems.append(
            "gpu-fault-api-ha carries processor pool sizing that this "
            "tier never creates, so it is counted as capacity that "
            f"does not exist: {', '.join(inert)}. Remove it with "
            "kubectl set env deployment/gpu-fault-api-ha "
            + " ".join(f"{name}-" for name in inert)
        )

    control_worker = container(worker, "control-worker")
    if expected_runtime_image and control_worker.get("image") != expected_runtime_image:
        problems.append(
            "gpu-fault-control-worker runtime image does not match GPU_FAULT_RUNTIME_IMAGE"
        )
    if "GPU_FAULT_PROCESSOR_WORKERS" not in env_names(control_worker):
        problems.append(
            "gpu-fault-control-worker has no "
            "GPU_FAULT_PROCESSOR_WORKERS: the tier that claims from "
            "the queue fell back to the 4-lane code default"
        )
    if env_value(control_worker, "GPU_FAULT_SERVICE_ROLE") != "worker":
        problems.append("gpu-fault-control-worker is not GPU_FAULT_SERVICE_ROLE=worker")
    if env_value(control_worker, "GPU_FAULT_TELEMETRY_SPOOL") == "true":
        problems.append(
            "gpu-fault-control-worker still runs the telemetry spool "
            "consumer; it must live only in the dedicated tier"
        )
    worker_command = control_worker["args"][0]
    if "--port 8081" not in worker_command:
        problems.append("gpu-fault-control-worker does not serve on 8081")
    reject_request_count_recycling(problems, "gpu-fault-control-worker", worker_command)
    replicas = worker["spec"].get("replicas", 0)
    if not replicas:
        problems.append(
            "gpu-fault-control-worker is scaled to zero: requests are accepted and never processed"
        )
    ready = worker.get("status", {}).get("readyReplicas", 0)
    if ready != replicas:
        problems.append(f"gpu-fault-control-worker has {ready}/{replicas} ready")

    spool_worker = container(spool, "telemetry-spool-worker")
    if expected_runtime_image and spool_worker.get("image") != expected_runtime_image:
        problems.append(
            "gpu-fault-telemetry-spool-worker runtime image does not match GPU_FAULT_RUNTIME_IMAGE"
        )
    if env_value(spool_worker, "GPU_FAULT_SERVICE_ROLE") != "spool-worker":
        problems.append(
            "gpu-fault-telemetry-spool-worker is not GPU_FAULT_SERVICE_ROLE=spool-worker"
        )
    if env_value(spool_worker, "GPU_FAULT_TELEMETRY_SPOOL") != "true":
        problems.append(
            "gpu-fault-telemetry-spool-worker has its spool consumer disabled"
        )
    try:
        ingress_item_bytes = int(
            env_value(
                api,
                "GPU_FAULT_TELEMETRY_SPOOL_MAX_ITEM_BYTES",
            )
            or "0"
        )
        spool_item_bytes = int(
            env_value(
                spool_worker,
                "GPU_FAULT_TELEMETRY_SPOOL_MAX_ITEM_BYTES",
            )
            or "0"
        )
        spool_batch_bytes = int(
            env_value(
                spool_worker,
                "GPU_FAULT_TELEMETRY_SPOOL_REPLAY_BATCH_MAX_BYTES",
            )
            or "0"
        )
        spool_in_flight_bytes = int(
            env_value(
                spool_worker,
                "GPU_FAULT_TELEMETRY_SPOOL_MAX_IN_FLIGHT_BYTES",
            )
            or "0"
        )
        spool_workers = int(
            env_value(
                spool_worker,
                "GPU_FAULT_TELEMETRY_SPOOL_WORKERS",
            )
            or "0"
        )
        spool_batch_items = int(
            env_value(
                spool_worker,
                "GPU_FAULT_TELEMETRY_SPOOL_REPLAY_BATCH_MAX_ITEMS",
            )
            or "0"
        )
        spool_fallback_seconds = float(
            env_value(
                spool_worker,
                "GPU_FAULT_TELEMETRY_SPOOL_NOTIFICATION_FALLBACK_SECONDS",
            )
            or "0"
        )
    except ValueError:
        problems.append(
            "telemetry spool byte, batch, worker and fallback limits must be numeric"
        )
    else:
        if ingress_item_bytes <= 0 or ingress_item_bytes != spool_item_bytes:
            problems.append(
                "ingress and spool-worker disagree on the maximum spooled item size"
            )
        if spool_batch_bytes < spool_item_bytes + 8192:
            problems.append(
                "telemetry replay batch bytes cannot hold one maximum item plus its envelope"
            )
        if (
            spool_workers <= 0
            or spool_in_flight_bytes < spool_workers * spool_batch_bytes
        ):
            problems.append(
                "telemetry in-flight byte limit is smaller than one maximum replay batch per worker"
            )
        if spool_batch_items != 64:
            problems.append("telemetry spool replay batch item limit must be 64")
        if not 2 <= spool_fallback_seconds <= 5:
            problems.append(
                "telemetry spool notification fallback must be between 2 and 5 seconds"
            )
    spool_inert = sorted(env_names(spool_worker) & set(PROCESSOR_POOL_ENV))
    if spool_inert:
        problems.append(
            "gpu-fault-telemetry-spool-worker carries main processor "
            "pool sizing: " + ", ".join(spool_inert)
        )
    spool_command = spool_worker["args"][0]
    if "--port 8082" not in spool_command:
        problems.append("gpu-fault-telemetry-spool-worker does not serve on 8082")
    if "--workers 1" not in spool_command:
        problems.append(
            "gpu-fault-telemetry-spool-worker must run exactly one uvicorn process per Pod"
        )
    spool_replicas = spool["spec"].get("replicas", 0)
    spool_ready = spool.get("status", {}).get("readyReplicas", 0)
    if spool_admission == "true" and not spool_replicas:
        problems.append(
            "gpu-fault-telemetry-spool-worker is scaled to zero while ingress admits telemetry"
        )
    elif spool_admission == "true" and spool_ready != spool_replicas:
        problems.append(
            f"gpu-fault-telemetry-spool-worker has {spool_ready}/{spool_replicas} ready"
        )
    elif spool_admission == "false" and spool_replicas != 0:
        problems.append(
            "gpu-fault-telemetry-spool-worker must be scaled to zero "
            "while ingress admission is disabled"
        )

    for problem in problems:
        print(f"role-split check failed: {problem}")
    if problems:
        return 1
    print(
        "role-split check passed: ingress "
        f"{ingress['spec'].get('replicas')} replicas on 8080, worker "
        f"{replicas} replicas on 8081, telemetry spool "
        f"{spool_replicas} replicas on 8082"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Post-deploy self-check for the regional role split.

Checks the live control plane against the two properties that decide
whether the split works at all, neither of which shows up in pod status:

  * exactly one tier serves ingress and one tier claims from the queue,
    with the worker tier scaled above zero. A control plane with no
    worker still returns 202 for every request and simply never
    processes any of them;
  * each tier's uvicorn command matches the tier - ingress on 8080 and
    worker on 8081 with four uvicorn workers each, spool on 8082 with one,
    and no request-count worker recycling on any tier. The worker count
    is the capacity model (pools and shards are replicas x workers) and
    /metrics is aggregated over the Pod's processes by the application;
    ingress recycling removes serving capacity during bursts; worker
    recycling can strand leases;
  * the ingress tier carries no processor pool sizing. Those pools are
    created by run_processor, which never starts on an ingress replica,
    so a value there is inert - and inert config is worse than absent
    config here, because it is what capacity math and
    gpu_fault_processor_workers read as real threads. `kubectl set env`
    on both Deployments is how it gets there.

Exits non-zero with the reason so a deploy fails loudly.

Every assertion above is the *current* release's contract. A verbatim
rollback (``GPU_FAULT_ROLE_SPLIT_CONTAINER_ENV_FILE`` set) restores the
previous release's container env, which the previous release verified under
its own contract; judging it by today's would fail a correct rollback the
moment a release drops an env name or tightens a rule. In that mode the
verifier instead proves the rollback did what it promised: the three role
Deployments exist, every container's live ``env``/``envFrom`` equals the
snapshot, the containers run ``GPU_FAULT_RUNTIME_IMAGE``, and every tier is
fully ready with at least one replica. See ``verify_against_snapshot``.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from gpu_fault.container_env_snapshot import (  # noqa: E402
    ROLE_DEPLOYMENTS,
    ContainerEnvSnapshotError,
    container_env_differences,
    load_container_env_snapshot,
    pod_container_env,
)

NAMESPACE = os.getenv("GPU_FAULT_NAMESPACE", "gpu-fault-system")
# Set by the release engine on a verbatim rollback and read by the renderer
# (see CONTAINER_ENV_FILE_VARIABLE in render_control_plane_role_split.py);
# the apply script hands it on so this verifier judges the same snapshot.
CONTAINER_ENV_FILE_VARIABLE = "GPU_FAULT_ROLE_SPLIT_CONTAINER_ENV_FILE"
MISSING_ROLE_CONSEQUENCE = {
    "gpu-fault-api-ha": "",
    "gpu-fault-control-worker": ": nothing claims from the processor queue",
    "gpu-fault-telemetry-spool-worker": (
        ": routine telemetry is admitted to the spool but nobody drains it"
    ),
}
# name -> data, or None once kubectl has answered NotFound. Absence is
# cached like presence so an optional ConfigMap referenced from several
# places costs one lookup, and so a later required reference to the same
# name still fails (see config_map_data).
_CONFIG_MAP_CACHE: dict[str, dict[str, str] | None] = {}

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


def config_map_data(name: str, *, optional: bool = False) -> dict[str, str] | None:
    """Return a ConfigMap's ``data``, following kubelet's ``optional`` rule.

    A ConfigMap the API server reports as NotFound is an error for a
    required reference and ``None`` for an optional one - exactly what
    kubelet does when it builds the container environment, so the
    verifier never turns an ``optional: true`` declaration in the
    Deployment template into a required one. Any other kubectl failure
    (unreachable API server, RBAC) is fatal regardless of ``optional``:
    kubelet tolerates a ConfigMap that does not exist, not one it could
    not read, and reading "unreachable" as "absent" would silently unset
    variables the template meant to carry.
    """

    if name not in _CONFIG_MAP_CACHE:
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
        if result.returncode == 0:
            _CONFIG_MAP_CACHE[name] = json.loads(result.stdout).get("data") or {}
        elif "NotFound" in (result.stderr or ""):
            _CONFIG_MAP_CACHE[name] = None
        else:
            raise SystemExit(
                f"ConfigMap {name} could not be read: "
                + (
                    (result.stderr or "").strip()
                    or f"kubectl exited {result.returncode}"
                )
            )
    data = _CONFIG_MAP_CACHE[name]
    if data is None and not optional:
        raise SystemExit(f"ConfigMap {name} is missing")
    return data


def env_values(item: dict) -> dict[str, str | None]:
    """Resolve a container's environment the way kubelet would.

    ``envFrom.configMapRef`` and ``valueFrom.configMapKeyRef`` honour
    ``optional: true``: an absent optional ConfigMap contributes nothing
    (envFrom) or leaves the variable unset (``None``, keyRef), and so does
    a present ConfigMap that lacks the referenced key. The Deployment
    template relies on this for ``gpu-fault-failure-domain-map``, which a
    site may legitimately not have.
    """

    values: dict[str, str | None] = {}
    for source in item.get("envFrom") or []:
        reference = source.get("configMapRef")
        if reference and reference.get("name"):
            data = config_map_data(
                reference["name"], optional=bool(reference.get("optional"))
            )
            if data is not None:
                values.update(data)
    for entry in item.get("env") or []:
        name = entry.get("name")
        if not name:
            continue
        if "value" in entry:
            values[name] = entry["value"]
            continue
        reference = entry.get("valueFrom", {}).get("configMapKeyRef")
        if reference and reference.get("name"):
            data = config_map_data(
                reference["name"], optional=bool(reference.get("optional"))
            )
            values[name] = None if data is None else data.get(reference.get("key"))
        else:
            values[name] = None
    return values


def env_value(item: dict, name: str) -> str | None:
    return env_values(item).get(name)


def env_names(item: dict) -> set[str]:
    return set(env_values(item))


UVICORN_WORKERS_PER_POD = 4
SPOOL_UVICORN_WORKERS_PER_POD = 1


def require_uvicorn_workers(
    problems: list[str],
    deployment_name: str,
    command: str,
    expected: int,
) -> None:
    """The tier's process count is the capacity model.

    Pools, consumer processes and notification shards are computed as
    replicas x uvicorn workers by the renderer, so a command that says a
    different ``--workers`` (a ``kubectl edit`` is how that happens) breaks
    the shard cover or the connection budget silently. /metrics is
    aggregated over the Pod's processes by the application, so the count
    is not a scrape concern.
    """

    if (
        command.count("--workers ") != 1
        or f"--workers {expected} " not in command + " "
    ):
        problems.append(
            f"{deployment_name} must run exactly {expected} uvicorn "
            f"process{'es' if expected != 1 else ''} per Pod"
        )


AURORA_SECRET_NAME = "gpu-fault-aurora"
AURORA_VOLUME_NAME = "aurora-credentials"
AURORA_MOUNT_DIR = "/etc/gpu-fault/aurora"


def require_aurora_credential_mount(
    problems: list[str],
    deployment_name: str,
    item: dict[str, Any],
    role_container: dict[str, Any],
) -> None:
    """Every role Deployment projects the Aurora Secret as files (CP-3).

    Running Pods reload the DSN from ``/etc/gpu-fault/aurora/postgres-url``
    after a password rotation; without the mount a rotated password only
    reaches a Pod on restart and the pool degrades to PoolTimeout once
    max_idle recycles a connection. A ``subPath`` mount is rejected because
    kubelet never updates it.
    """

    spec = item.get("spec", {}).get("template", {}).get("spec", {})
    volume = next(
        (v for v in spec.get("volumes", []) if v.get("name") == AURORA_VOLUME_NAME),
        None,
    )
    if volume is None:
        problems.append(f"{deployment_name}: no {AURORA_VOLUME_NAME} volume")
        return
    secret = volume.get("secret") or {}
    if secret.get("secretName") != AURORA_SECRET_NAME:
        problems.append(
            f"{deployment_name}: {AURORA_VOLUME_NAME} must project {AURORA_SECRET_NAME}"
        )
    if secret.get("items"):
        problems.append(
            f"{deployment_name}: {AURORA_VOLUME_NAME} must project the whole Secret"
        )
    mount = next(
        (
            m
            for m in role_container.get("volumeMounts", [])
            if m.get("name") == AURORA_VOLUME_NAME
        ),
        None,
    )
    if mount is None:
        problems.append(
            f"{deployment_name}: container does not mount {AURORA_VOLUME_NAME}"
        )
        return
    if mount.get("mountPath") != AURORA_MOUNT_DIR:
        problems.append(
            f"{deployment_name}: {AURORA_VOLUME_NAME} must mount at {AURORA_MOUNT_DIR}"
        )
    if mount.get("subPath") or mount.get("subPathExpr"):
        problems.append(
            f"{deployment_name}: {AURORA_VOLUME_NAME} must not use subPath (never updated)"
        )
    if mount.get("readOnly") is not True:
        problems.append(f"{deployment_name}: {AURORA_VOLUME_NAME} must be read-only")
    url_file = env_value(role_container, "GPU_FAULT_STORE_URL_FILE")
    if url_file not in (None, f"{AURORA_MOUNT_DIR}/postgres-url"):
        problems.append(
            f"{deployment_name}: GPU_FAULT_STORE_URL_FILE must point at "
            f"{AURORA_MOUNT_DIR}/postgres-url"
        )


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


def report(problems: list[str]) -> int:
    for problem in problems:
        print(f"role-split check failed: {problem}")
    return 1 if problems else 0


def role_deployments() -> tuple[dict[str, dict[str, Any]], list[str]]:
    """The three role Deployments by name, and a problem for each missing one."""

    found: dict[str, dict[str, Any]] = {}
    problems: list[str] = []
    for name in ROLE_DEPLOYMENTS:
        item = deployment(name)
        if item is None:
            problems.append(f"{name} is missing{MISSING_ROLE_CONSEQUENCE[name]}")
        else:
            found[name] = item
    return found, problems


def spool_tier_disabled_in_snapshot(snapshot: dict[str, Any]) -> bool:
    """Whether the previous release itself ran with telemetry-spool admission off.

    Read from the snapshot's own ingress environment, not from the current
    contract: when the previous ingress carried the literal
    ``GPU_FAULT_TELEMETRY_SPOOL=false`` the spool tier was legitimately scaled
    to zero (the apply script drains it on that transition), so a rollback that
    restores that shape must not be failed for the zero.
    """

    api = (snapshot.get("gpu-fault-api-ha") or {}).get("api") or {}
    return any(
        item.get("name") == "GPU_FAULT_TELEMETRY_SPOOL"
        and str(item.get("value", "")).strip().lower() == "false"
        for item in api.get("env") or []
    )


def verify_against_snapshot(
    found: dict[str, dict[str, Any]],
    snapshot_path: str,
    expected_runtime_image: str | None,
) -> int:
    """Snapshot mode: the rollback restored what it captured, and it is up.

    The current-contract assertions in ``main`` (service roles, ports,
    uvicorn worker counts, pool arithmetic, spool limits, inert pool sizing,
    the spool/ingress admission pairing) are skipped: the previous release
    verified its own environment under its own rules. What remains is
    release-independent: live ``env``/``envFrom`` equal the snapshot
    (``gpu_fault.container_env_snapshot`` normalisation), every container runs
    the expected image, and every tier has all of its replicas ready.
    """

    print(
        "role-split check running in container env snapshot mode: "
        f"{CONTAINER_ENV_FILE_VARIABLE} is set, so the current release's "
        "contract assertions are skipped and the live Deployments are judged "
        "against the previous release's snapshot"
    )
    try:
        snapshot = load_container_env_snapshot(snapshot_path)
    except ContainerEnvSnapshotError as exc:
        raise SystemExit(
            f"{CONTAINER_ENV_FILE_VARIABLE}: previous container environment "
            f"snapshot is invalid: {exc}"
        ) from exc
    live = {name: pod_container_env(item) for name, item in found.items()}
    problems = container_env_differences(snapshot, live, actual_label="live")
    spool_may_be_zero = spool_tier_disabled_in_snapshot(snapshot)
    for name, item in found.items():
        for member in item["spec"]["template"]["spec"]["containers"]:
            if expected_runtime_image and member.get("image") != expected_runtime_image:
                problems.append(
                    f"{name} container {member['name']} runtime image does not "
                    "match GPU_FAULT_RUNTIME_IMAGE"
                )
        replicas = item["spec"].get("replicas", 0)
        ready = item.get("status", {}).get("readyReplicas", 0)
        if not replicas:
            if not (name == "gpu-fault-telemetry-spool-worker" and spool_may_be_zero):
                problems.append(f"{name} is scaled to zero")
        elif ready != replicas:
            problems.append(f"{name} has {ready}/{replicas} ready")
    if problems:
        return report(problems)
    print(
        "role-split check passed (snapshot mode): live env/envFrom equal the "
        "previous release's snapshot; "
        + ", ".join(
            f"{name} {found[name]['spec'].get('replicas')} replicas ready"
            for name in ROLE_DEPLOYMENTS
        )
    )
    return 0


def main() -> int:
    expected_runtime_image = os.getenv("GPU_FAULT_RUNTIME_IMAGE")
    snapshot_path = os.getenv(CONTAINER_ENV_FILE_VARIABLE, "").strip()
    found, problems = role_deployments()
    if problems:
        return report(problems)
    if snapshot_path:
        return verify_against_snapshot(found, snapshot_path, expected_runtime_image)
    ingress = found["gpu-fault-api-ha"]
    worker = found["gpu-fault-control-worker"]
    spool = found["gpu-fault-telemetry-spool-worker"]

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
    require_uvicorn_workers(
        problems, "gpu-fault-api-ha", command, UVICORN_WORKERS_PER_POD
    )
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
    require_uvicorn_workers(
        problems, "gpu-fault-control-worker", worker_command, UVICORN_WORKERS_PER_POD
    )
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
    require_uvicorn_workers(
        problems,
        "gpu-fault-telemetry-spool-worker",
        spool_command,
        SPOOL_UVICORN_WORKERS_PER_POD,
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

    for deployment_name, item, role_container in (
        ("gpu-fault-api-ha", ingress, api),
        ("gpu-fault-control-worker", worker, control_worker),
        ("gpu-fault-telemetry-spool-worker", spool, spool_worker),
    ):
        require_aurora_credential_mount(problems, deployment_name, item, role_container)

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

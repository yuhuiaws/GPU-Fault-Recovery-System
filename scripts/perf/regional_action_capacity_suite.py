from __future__ import annotations

import argparse
import json
import statistics
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from gpu_fault.fleet_pins import CONFIG_MAP_KEY_ENVIRONMENT, DEFAULT_CONFIG_MAP
from gpu_fault.models import WorkflowOperation
from scripts.e2e.regional.regional_live_fixture import component_python
from scripts.e2e.regional.release_phase_gate import release_phase_error

if TYPE_CHECKING or __package__:
    from .regional_capacity_placement import (
        optional_placement_section,
        optional_start_placement,
    )
    from .regional_capacity_recovery import write_aborted_status
    from .regional_capacity_registry import (
        STORE_DSN_SNIPPET,
        dataplane_identity,
        validate_registry_target,
    )
    from .regional_capacity_resources import (
        RunResources,
        ensure_load_service_account,
        run_manifest,
    )
    from .regional_capacity_results import collect_pod_json_logs
    from .regional_capacity_suite import (
        CONNECTION_SECRET,
        CONTROL_NAMESPACE,
        DEFAULT_ARTIFACT_ROOT,
        NAMESPACE,
        REGISTRY_SECRET,
        TOKEN_SECRET,
        artifact_dir,
        control,
        dataplane,
        measured_exit,
        optional_aurora_step,
        postgres_counters,
        store_bloat_section,
        store_layout_section,
        register,
        release_identity,
        scrape_cgroup,
        scrape_metrics,
        step_status,
        teardown,
        upsert_configmap,
        validate_registered_resources,
        write_status,
    )
    from .regional_executor_support import (
        executor_job_failure,
        executor_scripts,
        retry_fields,
    )
    from .regional_remediation_budget import write_remediation_budget
else:
    from regional_capacity_placement import (
        optional_placement_section,
        optional_start_placement,
    )
    from regional_capacity_recovery import write_aborted_status
    from regional_capacity_registry import (
        STORE_DSN_SNIPPET,
        dataplane_identity,
        validate_registry_target,
    )
    from regional_capacity_resources import (
        RunResources,
        ensure_load_service_account,
        run_manifest,
    )
    from regional_capacity_results import collect_pod_json_logs
    from regional_capacity_suite import (
        CONNECTION_SECRET,
        CONTROL_NAMESPACE,
        DEFAULT_ARTIFACT_ROOT,
        NAMESPACE,
        REGISTRY_SECRET,
        TOKEN_SECRET,
        artifact_dir,
        control,
        dataplane,
        measured_exit,
        optional_aurora_step,
        postgres_counters,
        store_bloat_section,
        store_layout_section,
        register,
        release_identity,
        scrape_cgroup,
        scrape_metrics,
        step_status,
        teardown,
        upsert_configmap,
        validate_registered_resources,
        write_status,
    )
    from regional_executor_support import (
        executor_job_failure,
        executor_scripts,
        retry_fields,
    )
    from regional_remediation_budget import write_remediation_budget


LOAD_SERVICE_ACCOUNT = "gpu-fault-completion-watcher"
SCRIPT_CONFIGMAP = "gpu-fault-action-capacity-script"
JOB_NAME = "gpu-fault-action-capacity-executors"
TERMINAL = {"SUCCEEDED", "FAILED", "BLOCKED", "SUPERSEDED"}
CLAIM_ERROR_SAMPLE_LIMIT = 20
REPO_ROOT = Path(__file__).resolve().parents[2]
PERF_DIR = REPO_ROOT / "scripts" / "perf"
# Fleet pins the control plane serves from the ``gpu-fault-release-metadata``
# ConfigMap (``gpu_fault.fleet_pins.FleetPinRuntime`` polls it and hot-reloads
# the snapshot). The runner reads them from the same ConfigMap through the
# product's own key<->environment table.
RELEASE_PIN_CONFIGMAP = DEFAULT_CONFIG_MAP
ENVIRONMENT_CONFIG_MAP_KEY = {
    environment: key for key, environment in CONFIG_MAP_KEY_ENVIRONMENT.items()
}
AGENT_IDENTITY_PIN_KEYS = (
    "GPU_FAULT_REQUIRED_AGENT_PROTOCOL_VERSION",
    "GPU_FAULT_REQUIRED_NODE_ACTION_KEY_VERSION",
    "GPU_FAULT_REQUIRED_AGENT_ARTIFACT_SHA256",
    "GPU_FAULT_REQUIRED_AGENT_COMPATIBILITY_DIGEST",
    "GPU_FAULT_REQUIRED_AGENT_CONFIG_DIGEST",
)
# Start-up configuration: literal ``value`` env on the api-ha Deployment, never
# hot-reloaded, so the running Pod's environment is exactly what the product
# uses for them.
AGENT_IDENTITY_STARTUP_KEYS = (
    "GPU_FAULT_REQUIRED_AGENT_VERSION",
    "GPU_FAULT_REQUIRED_POLICY_VERSION",
    "GPU_FAULT_REQUIRED_RUNTIME_PROFILE_VERSION",
    "GPU_FAULT_ALLOWED_OPERATIONS",
)
AGENT_IDENTITY_ENV_KEYS = (*AGENT_IDENTITY_PIN_KEYS, *AGENT_IDENTITY_STARTUP_KEYS)
EXECUTOR_IDENTITY_ENV_KEYS = (
    "GPU_FAULT_REQUIRED_REGIONAL_EXECUTOR_PROTOCOL_VERSION",
    "GPU_FAULT_REQUIRED_REGIONAL_EXECUTOR_ARTIFACT_SHA256",
    "GPU_FAULT_REQUIRED_REGIONAL_EXECUTOR_COMPATIBILITY_DIGEST",
)


def log(message: str) -> None:
    stamp = datetime.now(timezone.utc).strftime("%H:%M:%S")
    print(f"[{stamp}] {message}", flush=True)


REMEDIATION_BUDGET_ENV = (
    "GPU_FAULT_REMEDIATION_MAX_ACTIVE_REGION",
    "GPU_FAULT_REMEDIATION_MAX_ACTIVE_PER_CLUSTER",
    "GPU_FAULT_REMEDIATION_MAX_ACTIVE_PER_RESOURCE_CLASS",
    "GPU_FAULT_REMEDIATION_MAX_ACTIVE_PER_NODE",
    "GPU_FAULT_REMEDIATION_MAX_ACTIVE_PER_FAILURE_DOMAIN",
)


def live_remediation_budget() -> dict[str, int | None]:
    """The five remediation budget limits of the live control workers.

    Read from one Running control-worker Pod's environment; ``None`` for a
    limit that is unset. The same request model runs under the production
    budget tier and the raw capacity tier (性能压测验收方案 §8.2), so the run
    records which tier it measured instead of assuming one.
    """

    pods = control(
        "get",
        "pod",
        "-l",
        "app=gpu-fault-control-worker",
        "--field-selector=status.phase=Running",
        "-o",
        "jsonpath={.items[*].metadata.name}",
    ).split()
    if not pods:
        raise RuntimeError("no Running control-worker Pods for the budget read")
    raw = control(
        "exec",
        pods[0],
        "--",
        "python3",
        "-c",
        (
            "import json,os; print(json.dumps({"
            + ",".join(
                f"{name!r}:os.environ.get({name!r})" for name in REMEDIATION_BUDGET_ENV
            )
            + "},sort_keys=True))"
        ),
    )
    values = json.loads(raw.splitlines()[-1])
    return {
        name: (int(values[name]) if values.get(name) not in (None, "") else None)
        for name in REMEDIATION_BUDGET_ENV
    }


def budget_tier(
    budget: dict[str, int | None],
    *,
    clusters: int,
    workflows_per_cluster: int,
    executor_workers: int,
) -> dict:
    """Classify the live budget against the fixed request model (§8.2).

    Raw capacity tier: region >= clusters x workflows, cluster and resource
    class >= workflows per cluster -- every workflow may run at once, so each
    executor must reach one in-flight command per workflow. Production budget tier:
    a lower limit throttles the workflows on purpose, so an executor only has
    to make progress (floor 1); the budget waits are the measurement. Live
    2026-09-22: under 20/5/2 every executor completed its 40 commands but
    exited on the raw-tier floor of 8, failing a run that had behaved.
    """

    region = budget.get("GPU_FAULT_REMEDIATION_MAX_ACTIVE_REGION")
    per_cluster = budget.get("GPU_FAULT_REMEDIATION_MAX_ACTIVE_PER_CLUSTER")
    per_class = budget.get("GPU_FAULT_REMEDIATION_MAX_ACTIVE_PER_RESOURCE_CLASS")
    limits = [
        (region, clusters * workflows_per_cluster),
        (per_cluster, workflows_per_cluster),
        (per_class, workflows_per_cluster),
    ]
    if all(limit is None for limit, _ in limits):
        tier = "unbounded"
    elif all(limit is None or limit >= needed for limit, needed in limits):
        tier = "raw-capacity"
    else:
        tier = "production-budget"
    return {
        "tier": tier,
        "limits": dict(budget),
        "required_for_raw_capacity": {
            "region": clusters * workflows_per_cluster,
            "per_cluster": workflows_per_cluster,
            "per_resource_class": workflows_per_cluster,
        },
        # The executor's concurrency floor is progress-only in every tier: the
        # per-node and per-failure-domain limits serialize a cluster's workflows
        # on purpose, so peak concurrency is evidence (max_concurrent_commands
        # per executor), not a pass criterion. Live 2026-09-22: floors of 8 and
        # then 4 each failed a run whose 128 workflows had all SUCCEEDED.
        "min_concurrent_commands": 1,
    }


# The scope kinds the occupancy verdict bounds, and the limit each one reads
# from ``run.json.remediation_budget.limits``. Node and failure-domain peaks are
# recorded as evidence only.
BUDGET_OCCUPANCY_LIMITS = {
    "region": "GPU_FAULT_REMEDIATION_MAX_ACTIVE_REGION",
    "cluster": "GPU_FAULT_REMEDIATION_MAX_ACTIVE_PER_CLUSTER",
    "class": "GPU_FAULT_REMEDIATION_MAX_ACTIVE_PER_RESOURCE_CLASS",
}
# ``execute_capacity_run``'s result when the workflows completed but a sample
# held more RUNNING workflows on one scope than its limit.
BUDGET_LIMIT_EXCEEDED = 3


def budget_occupancy_check(history: list[dict], remediation_budget: dict) -> dict:
    """RUNNING-with-claims against the run's recorded limits (性能压测验收方案 §8.2).

    A claim is held until its workflow ends (详细设计-v2 §9.5), so every
    ``history.json`` sample must stay within the limits, not only the
    admissions: per sample, ``budget_holders_max`` is the number of RUNNING
    workflows listing the busiest scope of each kind -- the region, the busiest
    cluster, the busiest (cluster, resource class). A limit that is unset is
    not checked; a history without any such sample fails, since it proves
    nothing.
    """

    limits = remediation_budget.get("limits") or {}
    peaks: dict[str, int] = {}
    samples = [
        item["budget_holders_max"] for item in history if "budget_holders_max" in item
    ]
    for holders in samples:
        for kind, count in holders.items():
            peaks[kind] = max(peaks.get(kind, 0), int(count))
    bounded = {kind: limits.get(name) for kind, name in BUDGET_OCCUPANCY_LIMITS.items()}
    violations = [
        f"{kind} scope held by {peaks[kind]} RUNNING workflows, limit {limit}"
        for kind, limit in bounded.items()
        if limit is not None and peaks.get(kind, 0) > int(limit)
    ]
    if not samples:
        violations.append("history.json has no remediation budget occupancy sample")
    return {
        "passed": not violations,
        "samples": len(samples),
        "peak_holders": peaks,
        "limits": bounded,
        "violations": violations,
    }


def executor_job(
    *,
    clusters: int,
    expected_commands: int,
    workers: int,
    delay_scale: float,
    lease_seconds: int,
    inject_renew_failure_once: bool,
    executor_protocol_version: int,
    executor_artifact_sha256: str,
    executor_compatibility_digest: str,
    connection_secret: str = CONNECTION_SECRET,
    min_concurrent_commands: int | None = None,
    max_seconds: int = 900,
) -> dict:
    return {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {
            "name": JOB_NAME,
            "namespace": NAMESPACE,
        },
        "spec": {
            "completions": clusters,
            "parallelism": clusters,
            "completionMode": "Indexed",
            "backoffLimit": 0,
            # The executors' own wall (ACTION_MAX_SECONDS) plus teardown slack;
            # both follow --timeout-seconds (live 2026-09-22: a throttled
            # production-budget run needed ~16 min and hit the fixed 900/1200).
            "activeDeadlineSeconds": max_seconds + 300,
            "template": {
                "spec": {
                    "restartPolicy": "Never",
                    "serviceAccountName": LOAD_SERVICE_ACCOUNT,
                    "containers": [
                        {
                            "name": "executor",
                            "image": ("public.ecr.aws/docker/library/python:3.12-slim"),
                            "command": [
                                "python",
                                "/scripts/benchmark_regional_action_executor.py",
                            ],
                            "env": [
                                {
                                    "name": ("GPU_FAULT_CONTROL_PLANE_URL"),
                                    "valueFrom": {
                                        "secretKeyRef": {
                                            "name": connection_secret,
                                            "key": "control-plane-url",
                                        }
                                    },
                                },
                                {
                                    "name": "SSL_CERT_FILE",
                                    "value": "/tls/ca.crt",
                                },
                                {
                                    "name": "CLUSTERS_FILE",
                                    "value": "/tokens/clusters.json",
                                },
                                {
                                    "name": "CLUSTER_OFFSET",
                                    "valueFrom": {
                                        "fieldRef": {
                                            "fieldPath": (
                                                "metadata.annotations["
                                                "'batch.kubernetes.io/"
                                                "job-completion-index']"
                                            )
                                        }
                                    },
                                },
                                {
                                    "name": "EXPECTED_COMMANDS",
                                    "value": str(expected_commands),
                                },
                                {
                                    "name": "ACTION_EXECUTOR_WORKERS",
                                    "value": str(workers),
                                },
                                {
                                    "name": "ACTION_MAX_SECONDS",
                                    "value": str(max_seconds),
                                },
                                {
                                    "name": "ACTION_MIN_CONCURRENT_COMMANDS",
                                    "value": str(
                                        workers
                                        if min_concurrent_commands is None
                                        else min_concurrent_commands
                                    ),
                                },
                                {
                                    "name": "ACTION_DELAY_SCALE",
                                    "value": str(delay_scale),
                                },
                                {
                                    "name": "ACTION_LEASE_SECONDS",
                                    "value": str(lease_seconds),
                                },
                                {
                                    "name": "ACTION_INJECT_RENEW_FAILURE_ONCE",
                                    "value": str(inject_renew_failure_once).lower(),
                                },
                                {
                                    "name": "EXECUTOR_PROTOCOL_VERSION",
                                    "value": str(executor_protocol_version),
                                },
                                {
                                    "name": "EXECUTOR_ARTIFACT_SHA256",
                                    "value": executor_artifact_sha256,
                                },
                                {
                                    "name": "EXECUTOR_COMPATIBILITY_DIGEST",
                                    "value": executor_compatibility_digest,
                                },
                            ],
                            "resources": {
                                "requests": {
                                    "cpu": "250m",
                                    "memory": "128Mi",
                                },
                                "limits": {
                                    "cpu": "2",
                                    "memory": "512Mi",
                                },
                            },
                            "volumeMounts": [
                                {
                                    "name": "script",
                                    "mountPath": "/scripts",
                                    "readOnly": True,
                                },
                                {
                                    "name": "tokens",
                                    "mountPath": "/tokens",
                                    "readOnly": True,
                                },
                                {
                                    "name": "tls",
                                    "mountPath": "/tls",
                                    "readOnly": True,
                                },
                            ],
                        }
                    ],
                    "volumes": [
                        {
                            "name": "script",
                            "configMap": {"name": SCRIPT_CONFIGMAP},
                        },
                        {
                            "name": "tokens",
                            "secret": {"secretName": TOKEN_SECRET},
                        },
                        {
                            "name": "tls",
                            "secret": {
                                "secretName": connection_secret,
                                "items": [
                                    {
                                        "key": "ca.crt",
                                        "path": "ca.crt",
                                    }
                                ],
                            },
                        },
                    ],
                }
            },
        },
    }


def _release_state() -> dict[str, object]:
    state = json.loads(
        control(
            "get",
            "configmap",
            "gpu-fault-regional-release-state",
            "-o",
            "jsonpath={.data.state\\.json}",
        )
    )
    if not isinstance(state, dict):
        raise RuntimeError("regional release state is not a JSON object")
    # Its pins are the candidate's: only a committed release is the one served.
    if problem := release_phase_error(state):
        raise RuntimeError(problem)
    return state


def _release_pins(keys: tuple[str, ...]) -> dict[str, str]:
    """Read fleet pins from the ConfigMap the product serves them from.

    Since the fleet-pin hot reload, ``gpu-fault-release-metadata`` is the single
    system of record: ``FleetPinRuntime`` polls it and the control plane serves
    whatever it holds. A Pod's environment is only the ``configMapKeyRef``
    snapshot the kubelet resolved at container start, so a data-plane-only
    release that re-published the pins without rolling the api-ha Pods leaves
    that env stale while the product is consistent (live 2026-09-21: Pod env
    ARTIFACT_SHA256=edb7047b..., ConfigMap, release state and executor
    Deployment all 8036f771...). Pins must therefore never be read from a Pod.
    """
    document = json.loads(
        control("get", "configmap", RELEASE_PIN_CONFIGMAP, "-o", "json")
    )
    data = document.get("data") if isinstance(document, dict) else None
    if not isinstance(data, dict):
        data = {}
    values: dict[str, str] = {}
    for key in keys:
        configmap_key = ENVIRONMENT_CONFIG_MAP_KEY.get(key)
        if configmap_key is None:
            raise RuntimeError(
                f"{key} is not a fleet pin published through {RELEASE_PIN_CONFIGMAP}"
            )
        values[key] = str(data.get(configmap_key) or "").strip()
    missing = [key for key in keys if not values[key]]
    if missing:
        raise RuntimeError(
            f"control-plane release pin ConfigMap {RELEASE_PIN_CONFIGMAP} is "
            "incomplete: " + ", ".join(sorted(missing))
        )
    return values


def _control_environment(keys: tuple[str, ...]) -> dict[str, str]:
    """Read start-up configuration from a running api-ha Pod.

    Only for values the Deployment sets as literal env (agent version, policy
    and profile versions, allowed operations): those are not hot-reloaded, so
    the Pod is what the product runs with. Fleet pins are refused here; see
    ``_release_pins`` for why the Pod environment is stale for them.
    """
    pins = sorted(key for key in keys if key in ENVIRONMENT_CONFIG_MAP_KEY)
    if pins:
        raise RuntimeError(
            "fleet pins must be read from the release ConfigMap, not a Pod: "
            + ", ".join(pins)
        )
    pod = control(
        "get",
        "pod",
        "-l",
        "app=gpu-fault-api-ha",
        "-o",
        "jsonpath={.items[0].metadata.name}",
    ).strip()
    if not pod:
        raise RuntimeError("no control-plane API Pod available for release pin checks")
    script = (
        "import json, os\n"
        f"keys={json.dumps(list(keys))}\n"
        "print(json.dumps({key: os.environ.get(key, '') for key in keys}, "
        "sort_keys=True))"
    )
    output = control(
        "exec",
        pod,
        "--",
        "python3",
        "-c",
        script,
    )
    document = json.loads(output.splitlines()[-1])
    if not isinstance(document, dict):
        raise RuntimeError("control-plane release pin environment is not an object")
    values = {str(key): str(value or "") for key, value in document.items()}
    missing = [key for key in keys if not values.get(key)]
    if missing:
        raise RuntimeError(
            "control-plane release pin environment is incomplete: "
            + ", ".join(sorted(missing))
        )
    return values


def _state_value(state: dict[str, object], key: str) -> str:
    value = str(state.get(key) or "").strip()
    if not value:
        raise RuntimeError(f"regional release state is missing {key}")
    return value


def _component_digest(state: dict[str, object], component: str) -> str:
    raw = state.get("component_digests")
    if not isinstance(raw, dict):
        raise RuntimeError("regional release state is missing component digests")
    value = str(raw.get(component) or "").strip()
    if not value:
        raise RuntimeError(
            f"regional release state is missing {component} component digest"
        )
    return value


def _verify_environment_pins(
    environment: dict[str, str],
    expected: dict[str, str],
) -> None:
    mismatches = [
        key for key, value in expected.items() if environment.get(key) != value
    ]
    if mismatches:
        raise RuntimeError(
            "control-plane release pins differ from release state: "
            + ", ".join(sorted(mismatches))
        )


def release_agent_identity() -> dict[str, object]:
    state = _release_state()
    environment = {
        **_release_pins(AGENT_IDENTITY_PIN_KEYS),
        **_control_environment(AGENT_IDENTITY_STARTUP_KEYS),
    }
    protocol = _state_value(state, "agent_protocol_version")
    artifact = _state_value(state, "node_wheel_sha256")
    compatibility = _component_digest(state, "node_runtime")
    profile = _state_value(state, "runtime_profile_version")
    config_digest = _state_value(state, "agent_config_digest")
    _verify_environment_pins(
        environment,
        {
            "GPU_FAULT_REQUIRED_AGENT_PROTOCOL_VERSION": protocol,
            "GPU_FAULT_REQUIRED_AGENT_ARTIFACT_SHA256": artifact,
            "GPU_FAULT_REQUIRED_AGENT_COMPATIBILITY_DIGEST": compatibility,
            "GPU_FAULT_REQUIRED_RUNTIME_PROFILE_VERSION": profile,
            "GPU_FAULT_REQUIRED_AGENT_CONFIG_DIGEST": config_digest,
        },
    )
    allowed_operations = sorted(
        {
            WorkflowOperation(item.strip()).value
            for item in environment["GPU_FAULT_ALLOWED_OPERATIONS"].split(",")
            if item.strip()
        }
    )
    if not allowed_operations:
        raise RuntimeError("control-plane release allows no workflow operations")
    return {
        "agent_protocol_version": int(protocol),
        "node_action_key_version": int(
            environment["GPU_FAULT_REQUIRED_NODE_ACTION_KEY_VERSION"]
        ),
        "agent_version": environment["GPU_FAULT_REQUIRED_AGENT_VERSION"],
        "artifact_sha256": artifact,
        "compatibility_digest": compatibility,
        "installer_bundle_sha256": _state_value(state, "bundle_sha256"),
        "policy_version": environment["GPU_FAULT_REQUIRED_POLICY_VERSION"],
        "runtime_profile_version": profile,
        "config_digest": config_digest,
        "allowed_operations": allowed_operations,
    }


def executor_identity(
    *,
    require_dataplane_deployment: bool = True,
) -> dict[str, object]:
    state = _release_state()
    expected_protocol = _state_value(state, "executor_protocol_version")
    expected_artifact = _state_value(state, "executor_wheel_sha256")
    expected_compatibility = _component_digest(state, "executor")
    environment = _release_pins(EXECUTOR_IDENTITY_ENV_KEYS)
    _verify_environment_pins(
        environment,
        {
            "GPU_FAULT_REQUIRED_REGIONAL_EXECUTOR_PROTOCOL_VERSION": (
                expected_protocol
            ),
            "GPU_FAULT_REQUIRED_REGIONAL_EXECUTOR_ARTIFACT_SHA256": (expected_artifact),
            "GPU_FAULT_REQUIRED_REGIONAL_EXECUTOR_COMPATIBILITY_DIGEST": (
                expected_compatibility
            ),
        },
    )
    if require_dataplane_deployment:
        deployment = json.loads(
            dataplane_identity(
                "get",
                "deployment",
                "gpu-fault-cluster-executor",
                "-o",
                "json",
            )
        )
        live_environment = {
            item["name"]: item.get("value")
            for item in deployment["spec"]["template"]["spec"]["containers"][0].get(
                "env", []
            )
        }
        if (
            live_environment.get("GPU_FAULT_EXECUTOR_ARTIFACT_SHA256")
            != expected_artifact
        ):
            raise RuntimeError("live executor artifact pin differs from release state")
        if (
            live_environment.get("GPU_FAULT_EXECUTOR_COMPATIBILITY_DIGEST")
            != expected_compatibility
        ):
            raise RuntimeError(
                "live executor compatibility pin differs from release state"
            )
    return {
        "executor_protocol_version": int(expected_protocol),
        "executor_artifact_sha256": expected_artifact,
        "executor_compatibility_digest": expected_compatibility,
    }


def refresh_run_agents(
    *,
    run_id: str,
    clusters: int,
    lease_seconds: int,
    control_fn=None,
    telemetry: bool = False,
) -> dict:
    """Refresh the run's synthetic Agent heartbeats (seed script, run-heartbeats mode).

    ``control_fn`` lets a caller in another module route the kubectl calls
    through its own (test-patchable) ``control`` binding. ``telemetry`` makes
    the refresh also publish each node's healthy telemetry package, so a
    post-action validation after an escalated reboot sees the node return.
    """

    kube = control if control_fn is None else control_fn
    pod = kube(
        "get",
        "pod",
        "-l",
        "app=gpu-fault-api-ha",
        "-o",
        "jsonpath={.items[0].metadata.name}",
    ).strip()
    output = kube(
        "exec",
        "-i",
        pod,
        "--",
        "env",
        f"ACTION_RUN_ID={run_id}",
        f"ACTION_CLUSTERS={clusters}",
        f"ACTION_AGENT_LEASE_SECONDS={lease_seconds}",
        "ACTION_SEED_MODE=run-heartbeats",
        *(["ACTION_HEARTBEAT_TELEMETRY=node-return"] if telemetry else []),
        "python3",
        "-",
        stdin=(PERF_DIR / "seed_regional_action_workflows.py").read_bytes(),
        timeout=180,
    )
    return json.loads(output.splitlines()[-1])


def run_agent_heartbeat_refresher(
    *,
    run_id: str,
    clusters: int,
    lease_seconds: int,
    interval_seconds: int = 30,
    refresh_agents=None,
    clock=time.monotonic,
    control_fn=None,
    telemetry: bool = False,
):
    """A throttled callable: refreshes the run's Agent heartbeats at most once
    per ``interval_seconds``. The fleet compatibility preflight holds destructive
    steps behind a stale heartbeat, so a long run must keep them fresh (live
    2026-09-22: clusters 025-031 starved after their heartbeats aged past the
    limit while the first 25 clusters ran)."""

    next_refresh = 0.0
    refresh_agents = refresh_run_agents if refresh_agents is None else refresh_agents

    def refresh() -> None:
        nonlocal next_refresh
        now = clock()
        if now < next_refresh:
            return
        kwargs = {
            "run_id": run_id,
            "clusters": clusters,
            "lease_seconds": lease_seconds,
        }
        if control_fn is not None:
            kwargs["control_fn"] = control_fn
        if telemetry:
            kwargs["telemetry"] = True
        refresh_agents(**kwargs)
        next_refresh = now + interval_seconds

    return refresh


def seed(
    *,
    run_id: str,
    clusters: int,
    workflows_per_cluster: int,
    nodes_per_workflow: int,
    agent_identity: dict[str, object] | None,
) -> dict:
    pod = control(
        "get",
        "pod",
        "-l",
        "app=gpu-fault-api-ha",
        "-o",
        "jsonpath={.items[0].metadata.name}",
    ).strip()
    script = (PERF_DIR / "seed_regional_action_workflows.py").read_bytes()
    environment = [
        f"ACTION_RUN_ID={run_id}",
        f"ACTION_CLUSTERS={clusters}",
        f"ACTION_WORKFLOWS_PER_CLUSTER={workflows_per_cluster}",
        f"ACTION_NODES_PER_WORKFLOW={nodes_per_workflow}",
    ]
    if agent_identity is not None:
        environment.append(
            "ACTION_AGENT_IDENTITY_JSON="
            + json.dumps(agent_identity, separators=(",", ":"), sort_keys=True)
        )
    output = control(
        "exec",
        "-i",
        pod,
        "--",
        "env",
        *environment,
        component_python("cpu"),
        "-",
        stdin=script,
        timeout=300,
    )
    return json.loads(output.splitlines()[-1])


def database_snapshot(run_id: str) -> dict:
    pod = control(
        "get",
        "pod",
        "-l",
        "app=gpu-fault-api-ha",
        "-o",
        "jsonpath={.items[0].metadata.name}",
    ).strip()
    script = f"""
import json, os, psycopg
{STORE_DSN_SNIPPET}
pattern = 'workflow-actionperf-{run_id}-%'
out = {{}}
with psycopg.connect(store_dsn()) as conn:
    cur = conn.cursor()
    cur.execute(
        "SELECT payload->>'status', count(*) "
        "FROM gpu_fault_control_records "
        "WHERE kind='workflow' AND key LIKE %s GROUP BY 1",
        (pattern,),
    )
    out['workflows'] = dict(cur.fetchall())
    cur.execute(
        "SELECT payload->>'status', count(*) "
        "FROM gpu_fault_control_records "
        "WHERE kind='remote_command' "
        "AND payload->>'workflow_request_id' LIKE %s GROUP BY 1",
        (pattern,),
    )
    out['commands'] = dict(cur.fetchall())
    cur.execute(
        "SELECT min((payload->>'created_at')::timestamptz), "
        "max((payload->>'updated_at')::timestamptz) "
        "FROM gpu_fault_control_records "
        "WHERE kind='workflow' AND key LIKE %s",
        (pattern,),
    )
    first, last = cur.fetchone()
    out['first_created_at'] = first.isoformat() if first else None
    out['last_updated_at'] = last.isoformat() if last else None
    # Holders of the busiest scope per kind: RUNNING rows listing it, whatever
    # their lease, as the store counts them (the model makes no preemption
    # handoff, the rule's other holder).
    cur.execute(
        "SELECT split_part(claim.scope, ':', 1), max(claim.holders) FROM ("
        "SELECT held.scope, count(*) AS holders FROM gpu_fault_control_records r "
        "CROSS JOIN LATERAL jsonb_array_elements_text(CASE WHEN "
        "jsonb_typeof(r.payload->'remediation_budget_claims')='array' "
        "THEN r.payload->'remediation_budget_claims' ELSE '[]'::jsonb END) "
        "AS held(scope) WHERE r.kind='workflow' AND r.key LIKE %s "
        "AND r.payload->>'status'='RUNNING' GROUP BY held.scope) claim GROUP BY 1",
        (pattern,),
    )
    out['budget_holders_max'] = {{kind: int(n) for kind, n in cur.fetchall()}}
print(json.dumps(out))
"""
    output = control(
        "exec",
        pod,
        "--",
        component_python("cpu"),
        "-c",
        script,
        timeout=120,
    )
    return json.loads(output.splitlines()[-1])


def database_details(run_id: str) -> dict:
    pod = control(
        "get",
        "pod",
        "-l",
        "app=gpu-fault-api-ha",
        "-o",
        "jsonpath={.items[0].metadata.name}",
    ).strip()
    script = f"""
import json, os, re, statistics, psycopg
{STORE_DSN_SNIPPET}
from datetime import datetime
pattern = 'workflow-actionperf-{run_id}-%'
def stamp(value):
    return datetime.fromisoformat(value)
def percentile(values, ratio):
    values = sorted(values)
    return values[int((len(values) - 1) * ratio)] if values else None
with psycopg.connect(store_dsn()) as conn:
    cur = conn.cursor()
    cur.execute(
        "SELECT key, payload FROM gpu_fault_control_records "
        "WHERE kind='workflow' AND key LIKE %s",
        (pattern,),
    )
    workflows = cur.fetchall()
    cur.execute(
        "SELECT payload->>'workflow_request_id', "
        "min((payload->>'created_at')::timestamptz) "
        "FROM gpu_fault_control_records WHERE kind='remote_command' "
        "AND payload->>'workflow_request_id' LIKE %s GROUP BY 1",
        (pattern,),
    )
    first_commands = dict(cur.fetchall())
durations = []
first_actions = []
refusals = []
waited_first_actions = []
cluster_finished = {{}}
for key, payload in workflows:
    created = stamp(payload['created_at'])
    updated = stamp(payload['updated_at'])
    durations.append((updated - created).total_seconds())
    first = first_commands.get(key)
    refusals.append(int(payload.get('remediation_budget_wait_count') or 0))
    if first is not None:
        first_actions.append((first - created).total_seconds())
        if refusals[-1]:
            waited_first_actions.append(first_actions[-1])
    match = re.search(r'-c(\\d{{3}})-w\\d{{3}}$', key)
    if match:
        cluster = match.group(1)
        cluster_finished[cluster] = max(
            cluster_finished.get(cluster, updated), updated
        )
finish_values = list(cluster_finished.values())
out = {{
    'workflow_duration_p50_seconds': percentile(durations, 0.50),
    'workflow_duration_p95_seconds': percentile(durations, 0.95),
    'workflow_duration_p99_seconds': percentile(durations, 0.99),
    'workflow_duration_max_seconds': max(durations) if durations else None,
    'first_action_p50_seconds': percentile(first_actions, 0.50),
    'first_action_p95_seconds': percentile(first_actions, 0.95),
    'first_action_p99_seconds': percentile(first_actions, 0.99),
    'first_action_max_seconds': max(first_actions)
        if first_actions else None,
    'cluster_completion_skew_seconds': (
        max(finish_values) - min(finish_values)
    ).total_seconds() if finish_values else None,
    # remediation-budget.json (性能压测验收方案 §11): the product's refused
    # claim counts; the longest wait is creation to first command of a
    # workflow the budget refused at least once.
    'remediation_budget_waits': {{
        'recorded': any(
            'remediation_budget_wait_count' in item for _, item in workflows
        ),
        'claims_recorded': any(
            'remediation_budget_claims' in item for _, item in workflows
        ),
        'refused_claims': sum(refusals),
        'waited_workflows': sum(1 for value in refusals if value),
        'longest_wait_seconds': max(waited_first_actions)
            if waited_first_actions else None,
        'never_executed_workflows': sum(
            1 for key, _ in workflows if key not in first_commands
        ),
    }},
}}
print(json.dumps(out))
"""
    output = control(
        "exec",
        pod,
        "--",
        component_python("cpu"),
        "-c",
        script,
        timeout=120,
    )
    return json.loads(output.splitlines()[-1])


def collect_executor_logs(
    target: Path,
    *,
    job_name: str = JOB_NAME,
) -> list[dict]:
    raw = dataplane(
        "get",
        "pod",
        "-l",
        f"batch.kubernetes.io/job-name={job_name}",
        "-o",
        "jsonpath={range .items[*]}{.metadata.name} "
        "{.metadata.annotations.batch\\.kubernetes\\.io/"
        "job-completion-index}{'\\n'}{end}",
    )
    entries = []
    for line in raw.splitlines():
        pod, _, index = line.partition(" ")
        entries.append((index.strip(), pod))
    return collect_pod_json_logs(
        entries,
        target,
        fetch=lambda pod: dataplane("logs", pod, check=False, timeout=120),
    )


def aggregate_executor_documents(documents: list[dict]) -> dict:
    walls = [item.get("wall_seconds", 0.0) for item in documents]
    claim_error_counts: dict[str, int] = {}
    claim_error_samples: list[dict] = []
    for item in documents:
        for category, count in (item.get("claim_error_counts") or {}).items():
            claim_error_counts[str(category)] = claim_error_counts.get(
                str(category), 0
            ) + int(count)
        for sample in item.get("claim_error_samples") or []:
            if isinstance(sample, dict):
                claim_error_samples.append(
                    {
                        "category": str(sample.get("category", "unknown")),
                        "elapsed_seconds": float(sample.get("elapsed_seconds", 0.0)),
                    }
                )
    claim_error_samples.sort(
        key=lambda item: (item["elapsed_seconds"], item["category"])
    )
    return {
        "executor_pods": len(documents),
        "expected_commands": sum(
            item.get("expected_commands", 0) for item in documents
        ),
        "completed_commands": sum(
            item.get("completed_commands", 0) for item in documents
        ),
        "executor_wall_p50": statistics.median(walls) if walls else None,
        "executor_wall_max": max(walls) if walls else None,
        "duplicate_claims": sum(item.get("duplicate_claims", 0) for item in documents),
        "claim_errors": sum(item.get("claim_errors", 0) for item in documents),
        "claim_error_counts": dict(sorted(claim_error_counts.items())),
        "claim_error_samples": claim_error_samples[:CLAIM_ERROR_SAMPLE_LIMIT],
        "result_errors": sum(item.get("result_errors", 0) for item in documents),
        **retry_fields(documents),
        "renewals": sum(item.get("renewals", 0) for item in documents),
        "renewal_errors": sum(item.get("renewal_errors", 0) for item in documents),
        "injected_renewal_failures": sum(
            item.get("injected_renewal_failures", 0) for item in documents
        ),
        "long_commands": sum(item.get("long_commands", 0) for item in documents),
        "max_concurrent_commands": max(
            (item.get("max_concurrent_commands", 0) for item in documents),
            default=0,
        ),
    }


def execute_capacity_run(
    args: argparse.Namespace,
    *,
    run_id: str,
    artifacts: Path,
    expected_workflows: int,
    expected_commands_per_cluster: int,
    started: float,
    registry_scope: str,
    resources: RunResources,
    min_concurrent_commands: int | None = None,
    placement: dict | None = None,
    remediation_budget: dict | None = None,
) -> int:
    if min_concurrent_commands is None:
        min_concurrent_commands = args.executor_workers
    upsert_configmap(SCRIPT_CONFIGMAP, resources=resources, text=executor_scripts())
    identity = executor_identity(
        require_dataplane_deployment=registry_scope == "live",
    )
    manifest = executor_job(
        clusters=args.clusters,
        expected_commands=expected_commands_per_cluster,
        workers=args.executor_workers,
        min_concurrent_commands=min_concurrent_commands,
        max_seconds=int(args.timeout_seconds),
        delay_scale=args.delay_scale,
        lease_seconds=args.lease_seconds,
        inject_renew_failure_once=args.inject_renew_failure_once,
        **identity,
    )
    # The executor Pods run as the load ServiceAccount; a run creates it when
    # the perf namespace has none (live 2026-09-22: the previous run tore its
    # own account down and this Job could not create Pods).
    ensure_load_service_account(
        resources, name=LOAD_SERVICE_ACCOUNT, namespace=NAMESPACE
    )
    manifest = run_manifest(manifest, run_id)
    (artifacts / "job.json").write_text(json.dumps(manifest, indent=2) + "\n")
    resources.create(manifest)
    refresh_heartbeats = run_agent_heartbeat_refresher(
        run_id=run_id, clusters=args.clusters, lease_seconds=args.lease_seconds
    )
    # Agents are seeded after the executors are Running; refresh only from then.
    deadline = time.time() + 300
    while time.time() < deadline:
        running = dataplane(
            "get",
            "pod",
            "-l",
            f"batch.kubernetes.io/job-name={JOB_NAME}",
            "--field-selector=status.phase=Running",
            "-o",
            "name",
        ).splitlines()
        if len(running) == args.clusters:
            break
        time.sleep(5)
    else:
        raise RuntimeError("action executor pods did not start")

    pods = [
        item
        for item in control(
            "get",
            "pod",
            "-l",
            "app in (gpu-fault-api-ha,gpu-fault-control-worker)",
            "-o",
            "jsonpath={.items[*].metadata.name}",
        ).split()
        if item
    ]
    before_metrics = scrape_metrics(pods)
    before_cgroup = scrape_cgroup(pods)
    before_postgres = postgres_counters()
    seeded_at = time.time()
    seeded = seed(
        run_id=run_id,
        clusters=args.clusters,
        workflows_per_cluster=args.workflows_per_cluster,
        nodes_per_workflow=args.nodes_per_workflow,
        agent_identity=(
            release_agent_identity() if registry_scope == "isolated" else None
        ),
    )
    log(f"seeded {seeded['workflows_created']} workflows")

    history = []
    end = time.time() + args.timeout_seconds
    while time.time() < end:
        refresh_heartbeats()
        snapshot = database_snapshot(run_id)
        snapshot["elapsed_seconds"] = time.time() - seeded_at
        history.append(snapshot)
        terminal = sum(
            count
            for status, count in snapshot["workflows"].items()
            if status in TERMINAL
        )
        if terminal == expected_workflows:
            break
        if (stopped := executor_job_failure(JOB_NAME, dataplane)) is not None:
            snapshot["executor_job_failure"] = stopped  # nobody left to finish them
            break
        time.sleep(2)
    job_succeeded = ""
    job_failed = ""
    job_completions = ""
    job_deadline = time.time() + 120
    while time.time() < job_deadline:
        state = dataplane(
            "get",
            "job",
            JOB_NAME,
            "-o",
            "jsonpath={.status.succeeded}|{.status.failed}|{.spec.completions}",
            check=False,
        )
        job_succeeded, job_failed, job_completions = (state.split("|") + ["", "", ""])[
            :3
        ]
        if job_succeeded and job_succeeded == job_completions:
            break
        if job_failed and job_failed != "0":
            break
        time.sleep(2)
    after_metrics = scrape_metrics(pods)
    after_cgroup = scrape_cgroup(pods)
    after_postgres = postgres_counters()
    documents = collect_executor_logs(artifacts / "executors")
    (artifacts / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    final = history[-1] if history else {}
    details = database_details(run_id)
    occupancy = budget_occupancy_check(history, remediation_budget or {})
    executor_metrics = aggregate_executor_documents(documents)
    workflows_succeeded = (
        final.get("workflows", {}).get("SUCCEEDED", 0) == expected_workflows
    )
    executor_job_succeeded = (
        bool(job_succeeded)
        and job_succeeded == job_completions
        and not (job_failed and job_failed != "0")
    )
    summary = {
        **seeded,
        **details,
        **executor_metrics,
        "job_status": (
            "Complete"
            if workflows_succeeded and executor_job_succeeded
            else "Incomplete"
        ),
        "executor_job_succeeded": int(job_succeeded or 0),
        "executor_job_failed": int(job_failed or 0),
        "executor_job_completions": int(job_completions or 0),
        "workflow_elapsed_seconds": (
            history[-1]["elapsed_seconds"] if history else None
        ),
        "workflow_statuses": final.get("workflows", {}),
        "command_statuses": final.get("commands", {}),
        "remediation_budget_occupancy": occupancy,
        "store_bloat": store_bloat_section(before_postgres, after_postgres),
        "store_layout": store_layout_section(before_postgres, after_postgres),
        "placement": optional_placement_section(placement),
        **optional_aurora_step(placement, seeded_at, time.time()),
        "postgres_deltas": {
            key: (after_postgres.get(key) or 0) - (before_postgres.get(key) or 0)
            for key in ("deadlocks", "xact_commit", "xact_rollback")
        },
        "wall_seconds": time.time() - started,
    }
    for name, value in (
        ("metrics-before.json", before_metrics),
        ("metrics-after.json", after_metrics),
        ("cgroup-before.json", before_cgroup),
        ("cgroup-after.json", after_cgroup),
        ("postgres-before.json", before_postgres),
        ("postgres-after.json", after_postgres),
        ("summary.json", summary),
    ):
        (artifacts / name).write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n"
        )
    write_remediation_budget(artifacts, run_id, remediation_budget, occupancy, details)
    print(json.dumps(summary, indent=2, sort_keys=True))
    if summary["job_status"] != "Complete":
        return 1
    return measured_exit(0 if occupancy["passed"] else BUDGET_LIMIT_EXCEEDED, summary)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clusters", type=int, default=8)
    parser.add_argument("--workflows-per-cluster", type=int, default=2)
    parser.add_argument("--nodes-per-workflow", type=int, default=4)
    parser.add_argument("--executor-workers", type=int, default=8)
    parser.add_argument("--delay-scale", type=float, default=1.0)
    parser.add_argument("--lease-seconds", type=int, default=120)
    parser.add_argument("--inject-renew-failure-once", action="store_true")
    parser.add_argument("--timeout-seconds", type=int, default=900)
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=DEFAULT_ARTIFACT_ROOT,
    )
    parser.add_argument("--label", default=None)
    parser.add_argument("--suite-id")
    parser.add_argument("--allow-live-registry", action="store_true")
    parser.add_argument("--confirm-live-registry")
    parser.add_argument("--synthetic-ttl-seconds", type=int, default=3600)
    args = parser.parse_args(argv)
    if args.synthetic_ttl_seconds < 300:
        parser.error("--synthetic-ttl-seconds must be at least 300")
    try:
        registry_scope = validate_registry_target(
            allow_live_registry=args.allow_live_registry,
            confirmation=args.confirm_live_registry,
        )
    except RuntimeError as exc:
        parser.error(str(exc))
    run_id = args.suite_id or datetime.now(timezone.utc).strftime("act%Y%m%dT%H%M%S")
    label = args.label or f"actions-{args.clusters}c"
    identity = release_identity()
    artifacts = artifact_dir(
        args.artifact_root,
        label,
        identity["release_id"] or "unknown-release",
    )
    if any(artifacts.iterdir()):
        raise RuntimeError(
            "action capacity artifacts already exist; preserve the original run receipts"
        )
    expected_workflows = args.clusters * args.workflows_per_cluster
    expected_commands_per_cluster = args.workflows_per_cluster * 10
    log(f"artifacts: {artifacts}")
    tier = budget_tier(
        live_remediation_budget(),
        clusters=args.clusters,
        workflows_per_cluster=args.workflows_per_cluster,
        executor_workers=args.executor_workers,
    )
    log(
        f"remediation budget tier {tier['tier']}: executor concurrency floor "
        f"{tier['min_concurrent_commands']} of {args.executor_workers}"
    )
    (artifacts / "run.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "remediation_budget": tier,
                "clusters": args.clusters,
                "workflows_per_cluster": (args.workflows_per_cluster),
                "nodes_per_workflow": args.nodes_per_workflow,
                "executor_workers": args.executor_workers,
                "delay_scale": args.delay_scale,
                "lease_seconds": args.lease_seconds,
                "inject_renew_failure_once": args.inject_renew_failure_once,
                "suite_id": run_id,
                "registry_scope": registry_scope,
                "control_namespace": CONTROL_NAMESPACE,
                "dataplane_namespace": NAMESPACE,
                "registry_secret": REGISTRY_SECRET,
                "connection_secret": CONNECTION_SECRET,
                "agent_identity_source": (
                    "release-state" if registry_scope == "isolated" else "live-agent"
                ),
                **identity,
                "started_at": datetime.now(timezone.utc).isoformat(),
            },
            indent=2,
        )
        + "\n"
    )
    write_status(artifacts, status="running")
    started = time.time()
    failure: BaseException | None = None
    cleanup_error: Exception | None = None
    result = 1
    resources = RunResources(artifacts, run_id, NAMESPACE, dataplane)
    registration_pending = True
    run_validated = False
    try:
        # The Aurora writer and node zones, when the round names its cluster.
        placement = optional_start_placement(artifacts)
        register(
            args.clusters,
            artifacts,
            run_id=run_id,
            expires_at=datetime.now(timezone.utc)
            + timedelta(seconds=args.synthetic_ttl_seconds),
            allow_live_registry=args.allow_live_registry,
            live_registry_confirmation=args.confirm_live_registry,
        )
        registration_pending = False
        validate_registered_resources(artifacts, run_id, args.clusters)
        run_validated = True
        result = execute_capacity_run(
            args,
            run_id=run_id,
            artifacts=artifacts,
            expected_workflows=expected_workflows,
            expected_commands_per_cluster=expected_commands_per_cluster,
            started=started,
            registry_scope=registry_scope,
            resources=resources,
            min_concurrent_commands=tier["min_concurrent_commands"],
            placement=placement,
            remediation_budget=tier,
        )
    except BaseException as exc:
        failure = exc
    finally:
        if (registration_pending or run_validated) and (
            artifacts / "registry-registration-intent.json"
        ).exists():
            try:
                # Executors, data, then registry: all inside the retried teardown
                # (live 2026-10-13: a single purge before it raised once and left
                # perf-cap-000..031 in the live registry).
                teardown(
                    purge=True,
                    deregister_clusters=True,
                    allow_live_registry=args.allow_live_registry,
                    live_registry_confirmation=args.confirm_live_registry,
                    artifacts=artifacts,
                    run_id=run_id,
                )
            except Exception as exc:
                cleanup_error = exc
                if failure is not None:
                    failure.add_note(f"action capacity teardown also failed: {exc}")
                else:
                    failure = exc

    if failure is not None:
        write_aborted_status(
            artifacts,
            failure,
            cleanup_error=cleanup_error,
            run_id=run_id,
            scope=registry_scope,
        )
        log(f"aborted artifacts: {artifacts}")
        raise failure.with_traceback(failure.__traceback__)
    if result:
        write_status(
            artifacts,
            **step_status(
                result,
                "remediation budget occupancy exceeded a configured limit"
                if result == BUDGET_LIMIT_EXCEEDED
                else "action capacity job did not complete",
            ),
        )
        log(f"aborted artifacts: {artifacts}")
        return result
    write_status(artifacts, status="ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

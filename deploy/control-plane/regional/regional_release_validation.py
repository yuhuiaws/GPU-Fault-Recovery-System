from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any

import regional_deployment_inventory as inventory
from regional_release_config import ReleaseError
from regional_release_diff import ReleaseComponent, ReleaseExecutionPlan
from regional_release_runtime_identity import (
    CONTROL_PLANE_PYTHON,
    validate_runtime_component_identity,
)


ROOT = Path(__file__).resolve().parents[3]
TRANSIENT_CRITICAL_ALERTS = frozenset({"GpuFaultStoreIoRejected"})
CRITICAL_CLEAR_TIMEOUT_SECONDS = 420
CRITICAL_CLEAR_SAMPLE_SECONDS = 15
STORE_IO_REJECTION_METRIC = "gpu_fault_store_io_rejections_total"
CPU_METRIC_PORTS = {
    "gpu-fault-api-ha": 8080,
    "gpu-fault-control-worker": 8081,
    "gpu-fault-telemetry-spool-worker": 8082,
}


def ensure_profile_transition_safe(
    release: Any,
    previous_profile_version: str | None,
) -> None:
    desired = release.config.runtime_profile_version
    if not previous_profile_version or previous_profile_version == desired:
        return
    pod = release.runner.run(
        release._cpu(
            "-n",
            release.config.namespace,
            "get",
            "pod",
            "-l",
            f"app={inventory.CPU_INGRESS_DEPLOYMENT}",
            "--field-selector=status.phase=Running",
            "-o",
            "jsonpath={.items[0].metadata.name}",
        ),
        capture=True,
    )
    if not pod:
        raise ReleaseError("Runtime Profile transition has no running CPU ingress Pod")
    script = """
import json
import sys

from gpu_fault.app import ApplicationContext
from gpu_fault.models import WorkflowStatus
from gpu_fault.watcher import WorkloadPhase

expected = json.load(sys.stdin)
store = ApplicationContext.from_environment().store
nonterminal = {
    WorkflowStatus.PENDING,
    WorkflowStatus.SAFETY_PENDING,
    WorkflowStatus.BLOCKED,
    WorkflowStatus.RUNNING,
}
workflows = [
    item
    for item in store.list_workflows(statuses=nonterminal, limit=1001)
    if item.runtime_profile_version != expected["desired"]
]
workloads = [
    item.observation
    for item in store.list_attempt_observation_states()
    if item.observation.workload_phase
    in {WorkloadPhase.PENDING, WorkloadPhase.RUNNING}
    and item.observation.runtime_profile_version != expected["desired"]
]
print(
    json.dumps(
        {
            "workflow_count": len(workflows),
            "workload_count": len(workloads),
            "workflow_ids": [item.request_id for item in workflows[:10]],
            "attempt_ids": [item.attempt_id for item in workloads[:10]],
        },
        sort_keys=True,
    )
)
"""
    raw = release.runner.run(
        release._cpu(
            "-n",
            release.config.namespace,
            "exec",
            "-i",
            pod,
            "--",
            CONTROL_PLANE_PYTHON,
            "-c",
            script,
        ),
        input_text=json.dumps(
            {
                "previous": previous_profile_version,
                "desired": desired,
            }
        ),
        capture=True,
    )
    result = json.loads(raw)
    workflow_count = int(result.get("workflow_count", 0))
    workload_count = int(result.get("workload_count", 0))
    if workflow_count or workload_count:
        raise ReleaseError(
            "Runtime Profile finalize is blocked by old-profile activity: "
            f"workflows={workflow_count}, workloads={workload_count}"
        )


def validate_release_quick(
    release: Any,
    plan: ReleaseExecutionPlan,
) -> None:
    if plan.has(
        ReleaseComponent.CPU_STAGE,
        ReleaseComponent.CPU_FINALIZE,
    ):
        release.runner.run(
            [
                "bash",
                str(
                    ROOT
                    / "deploy/control-plane/tools/verify-control-plane-role-split.sh"
                ),
            ],
            env={
                **os.environ,
                "KUBECONFIG": release.config.cpu_kubeconfig,
                "GPU_FAULT_NAMESPACE": release.config.namespace,
            },
        )
    if plan.has(
        ReleaseComponent.EXECUTOR,
        ReleaseComponent.WATCHER,
        ReleaseComponent.COLLECTOR,
        ReleaseComponent.RECONCILER,
        ReleaseComponent.AGENT,
    ):
        for target in release.config.clusters:
            release.runner.run(
                [
                    "bash",
                    str(ROOT / "deploy/dataplane/tools/verify-dataplane-executor.sh"),
                ],
                env={
                    **os.environ,
                    "GPU_FAULT_NAMESPACE": release.config.namespace,
                    "GPU_FAULT_KUBE_CONTEXT": target.context,
                    "GPU_FAULT_CONTROL_PLANE_KUBECONFIG": (
                        release.config.cpu_kubeconfig
                    ),
                    "GPU_FAULT_EXPECTED_WHEEL_CONFIGMAP": (release.executor_wheel_cm),
                },
            )
    if plan.has(
        ReleaseComponent.CPU_STAGE,
        ReleaseComponent.CPU_FINALIZE,
        ReleaseComponent.EXECUTOR,
        ReleaseComponent.WATCHER,
        ReleaseComponent.COLLECTOR,
        ReleaseComponent.RECONCILER,
        ReleaseComponent.AGENT,
    ):
        validate_runtime_component_identity(release)


def critical_amp_alerts(release: Any) -> dict[str, Any]:
    workspace_id = release.config.health.amp_workspace_id
    if not workspace_id:
        return {"count": 0, "alerts": []}
    script = """
import json
import sys
import urllib.request

import boto3
from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest

request = json.load(sys.stdin)
region = request["region"]
workspace_id = request["workspace_id"]
url = (
    f"https://aps-workspaces.{region}.amazonaws.com/workspaces/"
    f"{workspace_id}/alertmanager/api/v2/alerts"
    "?active=true&silenced=false&inhibited=false"
)
session = boto3.Session(region_name=region)
credentials = session.get_credentials()
if credentials is None:
    raise RuntimeError("AWS credentials are unavailable for AMP alert query")
signed = AWSRequest(method="GET", url=url, headers={"Accept": "application/json"})
SigV4Auth(credentials.get_frozen_credentials(), "aps", region).add_auth(signed)
http_request = urllib.request.Request(
    url,
    headers={key: str(value) for key, value in signed.headers.items()},
)
with urllib.request.urlopen(http_request, timeout=30) as response:
    alerts = json.loads(response.read())
critical = [
    {
        "alertname": (item.get("labels") or {}).get("alertname"),
        "severity": (item.get("labels") or {}).get("severity"),
    }
    for item in alerts
    if (item.get("labels") or {}).get("severity") == "critical"
]
print(json.dumps({"count": len(critical), "alerts": critical}, sort_keys=True))
"""
    raw = release.runner.run(
        ["python3", "-c", script],
        input_text=json.dumps(
            {
                "region": release.config.aws_region,
                "workspace_id": workspace_id,
            }
        ),
        capture=True,
    )
    result = json.loads(raw)
    if not isinstance(result, dict):
        raise ReleaseError("AMP alert query returned a non-object")
    return result


def store_io_rejection_series_ready(release: Any) -> dict[str, Any]:
    reports: dict[str, dict[str, Any]] = {}
    errors: list[str] = []
    for deployment, port in CPU_METRIC_PORTS.items():
        deployed = release._get_json(
            release._cpu(
                "-n",
                release.config.namespace,
                "get",
                "deployment",
                deployment,
            )
        )
        replicas = int((deployed.get("spec") or {}).get("replicas") or 0)
        if replicas == 0:
            continue
        pods = release._get_json(
            release._cpu(
                "-n",
                release.config.namespace,
                "get",
                "pods",
                "-l",
                f"app={deployment}",
                "--field-selector=status.phase=Running",
            )
        )
        names = sorted(
            str((item.get("metadata") or {}).get("name") or "")
            for item in pods.get("items", [])
            if (item.get("metadata") or {}).get("name")
        )
        if len(names) != replicas:
            errors.append(
                f"{deployment} has {len(names)}/{replicas} Running metric Pods"
            )
        script = f"""
import json
from urllib.request import urlopen

metric = {STORE_IO_REJECTION_METRIC!r}
text = urlopen(
    "http://127.0.0.1:{port}/metrics",
    timeout=10,
).read().decode()
series = []
for line in text.splitlines():
    if not (
        line.startswith(metric + "{{")
        or line.startswith(metric + " ")
    ):
        continue
    name, raw = line.rsplit(None, 1)
    series.append(
        {{
            "labeled": 'process_id="' in name,
            "value": float(raw),
        }}
    )
print(
    json.dumps(
        {{
            "series_count": len(series),
            "all_labeled": bool(series)
            and all(item["labeled"] for item in series),
            "all_zero": bool(series)
            and all(item["value"] == 0 for item in series),
        }},
        sort_keys=True,
    )
)
"""
        for pod in names:
            raw = release.runner.run(
                release._cpu(
                    "-n",
                    release.config.namespace,
                    "exec",
                    pod,
                    "--",
                    CONTROL_PLANE_PYTHON,
                    "-c",
                    script,
                ),
                capture=True,
            )
            report = json.loads(raw)
            reports[f"{deployment}/{pod}"] = report
            if not report.get("all_labeled"):
                errors.append(f"{deployment}/{pod} has unlabeled Store I/O series")
            if not report.get("all_zero"):
                errors.append(f"{deployment}/{pod} has nonzero Store I/O rejections")
    return {
        "ready": not errors,
        "pods": reports,
        "errors": errors,
    }


def _critical_alert_names(snapshot: dict[str, Any]) -> set[str]:
    critical = snapshot.get("critical_alerts") or {}
    names = {
        str(item.get("alertname") or "").strip() or "<unknown>"
        for item in critical.get("alerts") or []
    }
    if int(critical.get("count", 0)) and not names:
        names.add("<unknown>")
    return names


def _nonterminal_remote_commands(snapshot: dict[str, Any]) -> bool:
    remote = (snapshot.get("remote_commands") or {}).get("by_status") or {}
    return any(
        int(remote.get(status, 0)) for status in ("PENDING", "LEASED", "WAITING")
    )


def wait_for_stability_baseline(
    release: Any,
    baseline: dict[str, Any],
    *,
    timeout_seconds: int = CRITICAL_CLEAR_TIMEOUT_SECONDS,
    sample_seconds: int = CRITICAL_CLEAR_SAMPLE_SECONDS,
) -> tuple[dict[str, Any], dict[str, Any]]:
    alerts = _critical_alert_names(baseline)
    report = {
        "initial_alerts": sorted(alerts),
        "wait_seconds": 0,
        "sample_count": 1,
        "metric_series": None,
    }
    if not alerts:
        return baseline, report
    unexpected = alerts - TRANSIENT_CRITICAL_ALERTS
    if unexpected:
        raise ReleaseError(
            "release stability baseline has non-settleable critical alerts: "
            + ", ".join(sorted(unexpected))
        )
    metric_series = release._store_io_rejection_series_ready()
    report["metric_series"] = metric_series
    if not metric_series.get("ready"):
        details = ", ".join(metric_series.get("errors") or [])
        raise ReleaseError(
            "release Store I/O rejection metric series are not labeled zero"
            + (f": {details}" if details else "")
        )
    if timeout_seconds < 1 or sample_seconds < 1:
        raise ReleaseError("release critical-clear wait configuration is invalid")
    initial_restarts = baseline["restarts"]
    waited = 0
    while waited < timeout_seconds:
        delay = min(sample_seconds, timeout_seconds - waited)
        time.sleep(delay)
        waited += delay
        sample = release._stability_snapshot()
        report["wait_seconds"] = waited
        report["sample_count"] = int(report["sample_count"]) + 1
        if sample["not_ready"]:
            raise ReleaseError("release critical-clear wait has non-Ready Pods")
        for key, count in sample["restarts"].items():
            if count > int(initial_restarts.get(key, 0)):
                raise ReleaseError(
                    f"release critical-clear wait observed a restart: {key}"
                )
        if _nonterminal_remote_commands(sample):
            raise ReleaseError(
                "release critical-clear wait has non-terminal remote commands"
            )
        current = _critical_alert_names(sample)
        unexpected = current - TRANSIENT_CRITICAL_ALERTS
        if unexpected:
            raise ReleaseError(
                "release critical-clear wait observed unexpected critical alerts: "
                + ", ".join(sorted(unexpected))
            )
        if not current:
            return sample, report
    raise ReleaseError(
        "release transient critical alerts did not clear within "
        f"{timeout_seconds} seconds: " + ", ".join(sorted(alerts))
    )


def stability_snapshot(release: Any) -> dict[str, Any]:
    restarts: dict[str, int] = {}
    not_ready: list[str] = []

    def collect(plane: str, kubectl: list[str]) -> None:
        value = release._get_json(
            kubectl
            + [
                "-n",
                release.config.namespace,
                "get",
                "pods",
            ]
        )
        for pod in value.get("items", []):
            metadata = pod.get("metadata", {})
            if metadata.get("deletionTimestamp"):
                continue
            owners = metadata.get("ownerReferences") or []
            if not any(
                owner.get("kind") in {"ReplicaSet", "DaemonSet"} for owner in owners
            ):
                continue
            name = str(metadata.get("name") or "")
            status = pod.get("status", {})
            ready = any(
                condition.get("type") == "Ready" and condition.get("status") == "True"
                for condition in status.get("conditions", [])
            )
            if status.get("phase") != "Running" or not ready:
                not_ready.append(f"{plane}/{name}")
            for container in status.get("containerStatuses", []):
                key = f"{plane}/{name}/{container.get('name')}"
                restarts[key] = int(container.get("restartCount") or 0)

    collect("cpu", release._cpu())
    for target in release.config.clusters:
        collect(target.cluster_id, release._gpu(target))

    pod = release.runner.run(
        release._cpu(
            "-n",
            release.config.namespace,
            "get",
            "pod",
            "-l",
            f"app={inventory.CPU_INGRESS_DEPLOYMENT}",
            "--field-selector=status.phase=Running",
            "-o",
            "jsonpath={.items[0].metadata.name}",
        ),
        capture=True,
    )
    if not pod:
        raise ReleaseError("stability window has no running CPU ingress Pod")
    script = """
import json
from gpu_fault.app import ApplicationContext

store = ApplicationContext.from_environment().store
print(
    json.dumps(
        {
            "queue": store.processor_queue_stats(),
            "remote_commands": store.remote_command_stats(),
        },
        default=float,
        sort_keys=True,
    )
)
"""
    store = json.loads(
        release.runner.run(
            release._cpu(
                "-n",
                release.config.namespace,
                "exec",
                pod,
                "--",
                CONTROL_PLANE_PYTHON,
                "-c",
                script,
            ),
            capture=True,
        )
    )
    return {
        "restarts": restarts,
        "not_ready": sorted(not_ready),
        "queue": store.get("queue") or {},
        "remote_commands": store.get("remote_commands") or {},
        "critical_alerts": release._critical_amp_alerts(),
    }


def validate_stability_window(
    release: Any,
    *,
    window_seconds: int | None = None,
    sample_seconds: int = 30,
    critical_clear_timeout_seconds: int = CRITICAL_CLEAR_TIMEOUT_SECONDS,
    critical_clear_sample_seconds: int = CRITICAL_CLEAR_SAMPLE_SECONDS,
) -> dict[str, Any]:
    configured = (
        window_seconds
        if window_seconds is not None
        else int(os.getenv("GPU_FAULT_RELEASE_STABILITY_SECONDS", "120"))
    )
    if not 120 <= configured <= 300:
        raise ReleaseError("release stability window must be within 120..300 seconds")
    if sample_seconds < 1 or sample_seconds > configured:
        raise ReleaseError("release stability sample interval is invalid")
    baseline = release._stability_snapshot()
    if baseline["not_ready"]:
        raise ReleaseError("release stability baseline has non-Ready Pods")
    baseline, critical_clear = wait_for_stability_baseline(
        release,
        baseline,
        timeout_seconds=critical_clear_timeout_seconds,
        sample_seconds=critical_clear_sample_seconds,
    )
    samples = [baseline]
    deadline = time.monotonic() + configured
    while time.monotonic() < deadline:
        time.sleep(min(sample_seconds, max(0, deadline - time.monotonic())))
        sample = release._stability_snapshot()
        if sample["not_ready"]:
            raise ReleaseError("release stability window has non-Ready Pods")
        if int(sample["critical_alerts"].get("count", 0)):
            raise ReleaseError("release stability window has critical alerts")
        for key, count in sample["restarts"].items():
            if count > int(baseline["restarts"].get(key, 0)):
                raise ReleaseError(
                    f"release stability window observed a restart: {key}"
                )
        remote = sample["remote_commands"].get("by_status") or {}
        if any(
            int(remote.get(status, 0)) for status in ("PENDING", "LEASED", "WAITING")
        ):
            raise ReleaseError(
                "release stability window has non-terminal remote commands"
            )
        samples.append(sample)
    queue_samples = [item["queue"] for item in samples]
    if len(queue_samples) >= 3:
        depths = [int(item.get("depth", 0)) for item in queue_samples[-3:]]
        ages = [
            float(item.get("oldest_age_seconds", 0.0)) for item in queue_samples[-3:]
        ]
        if depths[0] < depths[1] <= depths[2] and ages[0] < ages[1] <= ages[2]:
            raise ReleaseError(
                "release stability window observed sustained queue growth"
            )
    return {
        "mode": "stability",
        "healthy": True,
        "window_seconds": configured,
        "sample_count": len(samples),
        "baseline_queue": baseline["queue"],
        "final_queue": samples[-1]["queue"],
        "restart_total": sum(samples[-1]["restarts"].values()),
        "critical_alert_count": int(samples[-1]["critical_alerts"].get("count", 0)),
        "critical_clear": critical_clear,
    }


def _container_image(document: dict[str, Any], *path: str) -> str | None:
    value: Any = document
    for field in path:
        value = (value or {}).get(field)
    containers = (value or {}).get("containers") or []
    return str(containers[0].get("image") or "") if containers else None


def _validate_cpu_rollback(
    release: Any,
    previous: dict[str, Any],
    expected_runtime_image: str,
) -> None:
    if release._deployment_wheel(
        release._cpu(),
        inventory.CPU_INGRESS_DEPLOYMENT,
    ) != previous.get("cpu_wheel"):
        raise ReleaseError("rollback CPU wheel did not converge")
    cpu = release._get_json(
        release._cpu(
            "-n",
            release.config.namespace,
            "get",
            "deployment",
            inventory.CPU_INGRESS_DEPLOYMENT,
        )
    )
    if _container_image(cpu, "spec", "template", "spec") != expected_runtime_image:
        raise ReleaseError("rollback CPU runtime image did not converge")
    refresh_exists = (
        subprocess.run(
            release._cpu(
                "-n",
                release.config.namespace,
                "get",
                "cronjob",
                "gpu-fault-aurora-credential-refresh",
            ),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ).returncode
        == 0
    )
    if refresh_exists:
        refresh = release._get_json(
            release._cpu(
                "-n",
                release.config.namespace,
                "get",
                "cronjob",
                "gpu-fault-aurora-credential-refresh",
            )
        )
        if (
            _container_image(
                refresh,
                "spec",
                "jobTemplate",
                "spec",
                "template",
                "spec",
            )
            != expected_runtime_image
        ):
            raise ReleaseError("rollback Aurora refresh image did not converge")


def _validate_gpu_rollback(
    release: Any,
    previous: dict[str, Any],
    expected_runtime_image: str,
) -> None:
    for target in release.config.clusters:
        old = (previous.get("clusters") or {}).get(target.cluster_id, {})
        expected_wheel = old.get("wheel")
        expected_reconciler = old.get("reconciler_wheel")
        if expected_wheel and (
            release._deployment_wheel(
                release._gpu(target),
                inventory.GPU_EXECUTOR_DEPLOYMENT,
            )
            != expected_wheel
        ):
            raise ReleaseError(f"{target.cluster_id} rollback Executor wheel mismatch")
        for deployment_name in (
            inventory.GPU_EXECUTOR_DEPLOYMENT,
            inventory.GPU_RECONCILER_DEPLOYMENT,
        ):
            deployment = release._get_json(
                release._gpu(
                    target,
                    "-n",
                    release.config.namespace,
                    "get",
                    "deployment",
                    deployment_name,
                )
            )
            if (
                _container_image(deployment, "spec", "template", "spec")
                != expected_runtime_image
            ):
                raise ReleaseError(
                    f"{target.cluster_id} rollback runtime image mismatch"
                )
        if expected_reconciler and (
            release._deployment_wheel(
                release._gpu(target),
                inventory.GPU_RECONCILER_DEPLOYMENT,
            )
            != expected_reconciler
        ):
            raise ReleaseError(
                f"{target.cluster_id} rollback Reconciler wheel mismatch"
            )
        expected_template = old.get("template")
        if expected_template and (
            release._deployment_template_name(target) != expected_template
        ):
            raise ReleaseError(
                f"{target.cluster_id} rollback installer template mismatch"
            )
        if (
            expected_template
            and old.get("bundle")
            and release._template_bundle(target, expected_template) != old.get("bundle")
        ):
            raise ReleaseError(f"{target.cluster_id} rollback node bundle mismatch")
        expected_dcgm = old.get("dcgm_image")
        if expected_dcgm:
            dcgm = release._get_json(
                release._gpu(
                    target,
                    "-n",
                    release.config.namespace,
                    "get",
                    "daemonset",
                    "gpu-fault-dcgm-exporter",
                )
            )
            if _container_image(dcgm, "spec", "template", "spec") != expected_dcgm:
                raise ReleaseError(f"{target.cluster_id} rollback DCGM image mismatch")
        release.runner.run(
            [
                "bash",
                str(ROOT / "deploy/dataplane/tools/verify-dataplane-executor.sh"),
            ],
            env={
                **os.environ,
                "GPU_FAULT_NAMESPACE": release.config.namespace,
                "GPU_FAULT_KUBE_CONTEXT": target.context,
                "GPU_FAULT_CONTROL_PLANE_KUBECONFIG": (release.config.cpu_kubeconfig),
                "GPU_FAULT_EXPECTED_WHEEL_CONFIGMAP": expected_wheel or "",
            },
        )


def _validate_agent_rollback(
    release: Any,
    previous: dict[str, Any],
) -> None:
    pod = release.runner.run(
        release._cpu(
            "-n",
            release.config.namespace,
            "get",
            "pod",
            "-l",
            f"app={inventory.CPU_INGRESS_DEPLOYMENT}",
            "--field-selector=status.phase=Running",
            "-o",
            "jsonpath={.items[0].metadata.name}",
        ),
        capture=True,
    )
    script = """
import json
import sys
from datetime import datetime, timezone

from gpu_fault.app import ApplicationContext

expected = json.load(sys.stdin)
now = datetime.now(timezone.utc)
agents = [
    item
    for item in ApplicationContext.from_environment().store.list_agents()
    if getattr(item.lifecycle_state, "value", item.lifecycle_state) == "ACTIVE"
    and item.lease_expires_at is not None
    and item.lease_expires_at > now
]
mismatches = []
for item in agents:
    cluster = expected["clusters"].get(item.cluster_id) or {}
    if item.agent_protocol_version != cluster.get("protocol"):
        mismatches.append([item.node_id, "protocol"])
    if item.agent_version != cluster.get("version"):
        mismatches.append([item.node_id, "version"])
    if item.artifact_sha256 != cluster.get("artifact"):
        mismatches.append([item.node_id, "artifact"])
    if (
        item.compatibility_digest or item.artifact_sha256
    ) != cluster.get("compatibility"):
        mismatches.append([item.node_id, "compatibility"])
    if item.policy_version != cluster.get("policy"):
        mismatches.append([item.node_id, "policy"])
    if item.config_digest != cluster.get("config"):
        mismatches.append([item.node_id, "config"])
    if item.node_action_key_version != cluster.get("key_version"):
        mismatches.append([item.node_id, "key_version"])
    if item.runtime_profile_version != cluster.get("profile"):
        mismatches.append([item.node_id, "profile"])
    if (
        cluster.get("bundle") is not None
        and item.installer_bundle_sha256 != cluster["bundle"]
    ):
        mismatches.append([item.node_id, "bundle"])
    if (
        cluster.get("template") is not None
        and item.installer_template_sha256 != cluster["template"]
    ):
        mismatches.append([item.node_id, "template"])
print(json.dumps({"active": len(agents), "mismatches": mismatches}))
if not agents or mismatches:
    raise SystemExit(1)
"""
    identities = previous.get("agent_identities") or {}
    expected_cluster_ids = {target.cluster_id for target in release.config.clusters}
    if set(identities) != expected_cluster_ids:
        raise ReleaseError("rollback Agent identity snapshot is incomplete")
    expected = {
        "clusters": {
            cluster_id: {
                "protocol": identity.get("agent_protocol_version"),
                "version": identity.get("agent_version"),
                "artifact": identity.get("artifact_sha256"),
                "compatibility": identity.get("compatibility_digest"),
                "policy": identity.get("policy_version"),
                "profile": identity.get("runtime_profile_version"),
                "config": identity.get("config_digest"),
                "key_version": identity.get("node_action_key_version"),
                "bundle": identity.get("installer_bundle_sha256"),
                "template": identity.get("installer_template_sha256"),
            }
            for cluster_id, identity in identities.items()
        },
    }
    release.runner.run(
        release._cpu(
            "-n",
            release.config.namespace,
            "exec",
            "-i",
            pod,
            "--",
            CONTROL_PLANE_PYTHON,
            "-c",
            script,
        ),
        input_text=json.dumps(expected),
        capture=True,
    )


def validate_rollback(release: Any, previous: dict[str, Any]) -> None:
    if release.runner.dry_run:
        return
    metadata = dict(previous.get("metadata") or {})
    current_metadata = release._config_map_data("gpu-fault-release-metadata")
    for key, expected in metadata.items():
        if current_metadata.get(key) != expected:
            raise ReleaseError(f"rollback release metadata mismatch: {key}")
    expected_runtime_image = previous.get("runtime_image") or release.runtime_image
    _validate_cpu_rollback(release, previous, expected_runtime_image)
    expected_profile = previous.get("runtime_profile_version")
    live_profile = release._config_map_data("gpu-fault-api-ha-config-core").get(
        "GPU_FAULT_REQUIRED_RUNTIME_PROFILE_VERSION"
    )
    if live_profile != expected_profile:
        raise ReleaseError("rollback Runtime Profile did not converge")
    release.runner.run(
        [
            "bash",
            str(ROOT / "deploy/control-plane/tools/verify-control-plane-role-split.sh"),
        ],
        env={
            **os.environ,
            "KUBECONFIG": release.config.cpu_kubeconfig,
            "GPU_FAULT_NAMESPACE": release.config.namespace,
            "GPU_FAULT_RUNTIME_IMAGE": expected_runtime_image,
        },
    )
    _validate_gpu_rollback(release, previous, expected_runtime_image)
    _validate_agent_rollback(
        release,
        previous,
    )

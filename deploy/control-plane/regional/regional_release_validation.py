from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable

import regional_deployment_inventory as inventory
from regional_release_config import ReleaseError
from regional_release_diff import ReleaseComponent, ReleaseExecutionPlan
from regional_release_gpu_rollout import agents_converged, gpu_node_items
from regional_release_probes import probe_source
from regional_release_progress import GPU_COMPONENTS
from regional_release_runtime_identity import (
    CONTROL_PLANE_PYTHON,
    exec_cpu_ingress_probe,
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
QUICK_VALIDATION_EVIDENCE_ENV = "GPU_FAULT_QUICK_VALIDATION_EVIDENCE"
MAX_VALIDATION_WORKERS = 8


def _fan_out_clusters(
    release: Any,
    check: Callable[[Any], None],
    *,
    failure: str,
    targets: Any = None,
) -> None:
    """Run a read-only per-cluster check with bounded parallelism.

    Every check must stay read-only and confined to its own GPU cluster so the
    fan-out never introduces cross-cluster ordering. The first failure is
    re-raised after all in-flight checks finish, keeping the fail-closed
    contract of the sequential form.
    """

    selected = list(release.config.clusters if targets is None else targets)
    if not selected:
        return
    if len(selected) == 1 or release.runner.dry_run:
        for target in selected:
            try:
                check(target)
            except Exception as exc:
                raise ReleaseError(
                    f"{target.cluster_id} {failure} failed: {exc}"
                ) from exc
        return
    workers = min(MAX_VALIDATION_WORKERS, len(selected))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            (target.cluster_id, executor.submit(check, target)) for target in selected
        ]
        errors: list[tuple[str, Exception]] = []
        for cluster_id, future in futures:
            try:
                future.result()
            except Exception as exc:  # noqa: PERF203 - collect every cluster result
                errors.append((cluster_id, exc))
    if errors:
        cluster_id, exc = errors[0]
        raise ReleaseError(f"{cluster_id} {failure} failed: {exc}") from exc


def _fan_out_values(
    release: Any,
    items: Any,
    work: Callable[[Any], Any],
) -> list[Any]:
    """Map a read-only probe over `items` with bounded parallelism.

    Results are returned in input order so every caller keeps a deterministic
    report and error ordering regardless of completion order, and the first
    failure in input order is re-raised once the in-flight probes finish. That
    keeps the fail-closed contract of the sequential form.
    """

    selected = list(items)
    if len(selected) <= 1 or release.runner.dry_run:
        return [work(item) for item in selected]
    workers = min(MAX_VALIDATION_WORKERS, len(selected))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(work, item) for item in selected]
        results: list[Any] = []
        failure: Exception | None = None
        for future in futures:
            try:
                results.append(future.result())
            except Exception as exc:  # noqa: PERF203 - collect every probe result
                results.append(None)
                if failure is None:
                    failure = exc
    if failure is not None:
        raise failure
    return results


def _write_quick_validation_evidence(
    release: Any,
    *,
    checks: list[str],
) -> None:
    raw = os.getenv(QUICK_VALIDATION_EVIDENCE_ENV, "").strip()
    if not raw:
        return
    path = Path(raw).expanduser().resolve()
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    path.parent.chmod(0o700)
    value = {
        "schema_version": 1,
        "release_id": release.release_id,
        "release_delivery_sha256": release.config.release_delivery_sha256,
        "site_identity": {
            "site_name": release.config.site_name,
            "aws_region": release.config.aws_region,
            "cpu_eks_arn": release.config.cpu_eks_arn,
            "cluster_ids": sorted(
                target.cluster_id for target in release.config.clusters
            ),
        },
        "verified_at_epoch": int(time.time()),
        "checks": sorted(checks),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.chmod(0o600)
    os.replace(temporary, path)


def ensure_profile_transition_safe(
    release: Any,
    previous_profile_version: str | None,
) -> None:
    desired = release.config.runtime_profile_version
    if not previous_profile_version or previous_profile_version == desired:
        return
    script = probe_source("runtime_profile_transition")
    raw = exec_cpu_ingress_probe(
        release,
        script=script,
        failure="Runtime Profile transition",
        input_text=json.dumps(
            {
                "previous": previous_profile_version,
                "desired": desired,
            }
        ),
    )
    result = json.loads(raw)
    workflow_count = int(result.get("workflow_count", 0))
    workload_count = int(result.get("workload_count", 0))
    if workflow_count or workload_count:
        raise ReleaseError(
            "Runtime Profile finalize is blocked by old-profile activity: "
            f"workflows={workflow_count}, workloads={workload_count}"
        )


def validate_release_components(
    release: Any,
    *,
    cpu: bool,
    data_plane: bool,
    runtime_validator: Callable[[Any], object] = validate_runtime_component_identity,
) -> None:
    checks: list[str] = []
    if cpu:
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
                "GPU_FAULT_RUNTIME_IMAGE": release.runtime_image,
                "GPU_FAULT_SYNC_INSTALLED_RESOURCE_REGISTRY": "false",
            },
        )
        checks.append("control_plane_role_split")
    if data_plane:

        def verify_data_plane(target: Any) -> None:
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
                    "GPU_FAULT_SYNC_INSTALLED_RESOURCE_REGISTRY": "false",
                },
            )

        _fan_out_clusters(
            release,
            verify_data_plane,
            failure="data-plane executor verification",
        )
        checks.extend(
            f"data_plane_executor:{target.cluster_id}"
            for target in release.config.clusters
        )
    if cpu or data_plane:
        runtime_validator(release)
        checks.append("runtime_component_identity")
    _write_quick_validation_evidence(release, checks=checks)


def validate_release_quick(
    release: Any,
    plan: ReleaseExecutionPlan,
) -> None:
    validate_release_components(
        release,
        cpu=plan.has(
            ReleaseComponent.CPU_STAGE,
            ReleaseComponent.CPU_FINALIZE,
        ),
        data_plane=plan.has(
            ReleaseComponent.EXECUTOR,
            ReleaseComponent.WATCHER,
            ReleaseComponent.COLLECTOR,
            ReleaseComponent.RECONCILER,
            ReleaseComponent.AGENT,
        ),
    )


def critical_amp_alerts(release: Any) -> dict[str, Any]:
    workspace_id = release.config.health.amp_workspace_id
    if not workspace_id:
        return {"count": 0, "alerts": []}
    script = probe_source("critical_amp_alerts")
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


def _store_io_metric_pods(release: Any, deployment: str) -> tuple[int, tuple[str, ...]]:
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
        return 0, ()
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
    names = tuple(
        sorted(
            str((item.get("metadata") or {}).get("name") or "")
            for item in pods.get("items", [])
            if (item.get("metadata") or {}).get("name")
        )
    )
    return replicas, names


def _store_io_series_probe(release: Any, pod: str, port: int) -> dict[str, Any]:
    script = probe_source("store_io_rejection_series")
    return json.loads(
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
                json.dumps({"metric": STORE_IO_REJECTION_METRIC, "port": port}),
            ),
            capture=True,
        )
    )


def store_io_rejection_series_ready(release: Any) -> dict[str, Any]:
    reports: dict[str, dict[str, Any]] = {}
    errors: list[str] = []
    deployments = sorted(CPU_METRIC_PORTS)
    # Each role inventory is two independent reads and each metric probe is its
    # own `kubectl exec` round trip, so a settle window used to pay
    # `roles + Pods` serial round trips per sample. Results are consumed in input
    # order, so the report and the error list stay deterministic.
    inventories = _fan_out_values(
        release,
        deployments,
        lambda deployment: _store_io_metric_pods(release, deployment),
    )
    probes: list[tuple[str, str]] = []
    for deployment, (replicas, names) in zip(deployments, inventories):
        if replicas == 0:
            continue
        if len(names) != replicas:
            errors.append(
                f"{deployment} has {len(names)}/{replicas} Running metric Pods"
            )
        probes.extend((deployment, pod) for pod in names)
    samples = _fan_out_values(
        release,
        probes,
        lambda probe: _store_io_series_probe(
            release,
            probe[1],
            CPU_METRIC_PORTS[probe[0]],
        ),
    )
    for (deployment, pod), report in zip(probes, samples):
        key = f"{deployment}/{pod}"
        reports[key] = report
        if not report.get("all_labeled"):
            errors.append(f"{key} has unlabeled Store I/O series")
        if not report.get("all_zero"):
            errors.append(f"{key} has nonzero Store I/O rejections")
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
    def collect(plane: str, kubectl: list[str]) -> tuple[dict[str, int], list[str]]:
        restarts: dict[str, int] = {}
        not_ready: list[str] = []
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
        return restarts, not_ready

    def collect_store() -> dict[str, Any]:
        script = probe_source("store_stability_snapshot")
        return json.loads(
            exec_cpu_ingress_probe(
                release,
                script=script,
                failure="stability window",
                interactive=False,
            )
        )

    plane_tasks = [("cpu", release._cpu())]
    plane_tasks.extend(
        (target.cluster_id, release._gpu(target)) for target in release.config.clusters
    )
    workers = min(8, len(plane_tasks) + 2)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        pod_futures = [
            executor.submit(collect, plane, kubectl) for plane, kubectl in plane_tasks
        ]
        store_future = executor.submit(collect_store)
        alerts_future = executor.submit(release._critical_amp_alerts)
        pod_results = [future.result() for future in pod_futures]
        store = store_future.result()
        critical_alerts = alerts_future.result()
    restarts = {
        name: count
        for result, _not_ready in pod_results
        for name, count in result.items()
    }
    not_ready = [name for _restarts, result in pod_results for name in result]
    return {
        "restarts": restarts,
        "not_ready": sorted(not_ready),
        "queue": store.get("queue") or {},
        "remote_commands": store.get("remote_commands") or {},
        "critical_alerts": critical_alerts,
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
    refresh_exists = release.runner.probe(
        release._cpu(
            "-n",
            release.config.namespace,
            "get",
            "cronjob",
            "gpu-fault-aurora-credential-refresh",
        ),
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


def validate_agent_rollback_target(
    release: Any,
    previous: dict[str, Any],
    target: Any,
) -> None:
    identity = (previous.get("agent_identities") or {}).get(target.cluster_id)
    if not isinstance(identity, dict):
        raise ReleaseError(f"{target.cluster_id} rollback Agent identity is missing")
    node_names = tuple(str(value) for value in identity.get("node_ids") or [])
    if not node_names:
        raise ReleaseError(f"{target.cluster_id} rollback Agent node set is empty")
    artifact = str(identity.get("artifact_sha256") or "")
    config_digest = str(identity.get("config_digest") or "")
    if not artifact or not config_digest:
        raise ReleaseError(f"{target.cluster_id} rollback Agent pins are incomplete")
    if not agents_converged(
        gpu_node_items(release, target, fresh=True),
        target,
        artifact,
        bundle_sha=identity.get("installer_bundle_sha256"),
        template_sha=identity.get("installer_template_sha256"),
        config_digest=config_digest,
        require_node_uid=True,
        node_names=frozenset(node_names),
    ):
        raise ReleaseError(f"{target.cluster_id} rollback node annotations mismatch")
    if not release._agent_heartbeats_converged(
        target,
        node_count=len(node_names),
        node_names=tuple(sorted(node_names)),
        artifact_sha=artifact,
        config_digest=config_digest,
        runtime_profile_version=str(identity.get("runtime_profile_version") or ""),
        bundle_sha=identity.get("installer_bundle_sha256"),
        template_sha=identity.get("installer_template_sha256"),
        agent_identity=identity,
    ):
        raise ReleaseError(f"{target.cluster_id} rollback Agent heartbeat mismatch")


def validate_gpu_rollback_target(
    release: Any,
    previous: dict[str, Any],
    expected_runtime_image: str,
    target: Any,
    *,
    components: frozenset[ReleaseComponent] = GPU_COMPONENTS,
    run_verifier: bool = True,
) -> None:
    old = (previous.get("clusters") or {}).get(target.cluster_id, {})
    expected_wheel = old.get("wheel")
    expected_reconciler = old.get("reconciler_wheel")
    deployment_components = (
        (ReleaseComponent.EXECUTOR, inventory.GPU_EXECUTOR_DEPLOYMENT),
        (ReleaseComponent.WATCHER, inventory.GPU_WATCHER_DEPLOYMENT),
        (ReleaseComponent.COLLECTOR, inventory.GPU_COLLECTOR_DEPLOYMENT),
    )
    for component, deployment_name in deployment_components:
        if component not in components:
            continue
        if expected_wheel and (
            release._deployment_wheel(
                release._gpu(target),
                deployment_name,
            )
            != expected_wheel
        ):
            raise ReleaseError(
                f"{target.cluster_id} rollback {deployment_name} wheel mismatch"
            )
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
                f"{target.cluster_id} rollback {deployment_name} image mismatch"
            )
    if components.intersection({ReleaseComponent.RECONCILER, ReleaseComponent.AGENT}):
        deployment = release._get_json(
            release._gpu(
                target,
                "-n",
                release.config.namespace,
                "get",
                "deployment",
                inventory.GPU_RECONCILER_DEPLOYMENT,
            )
        )
        if (
            _container_image(deployment, "spec", "template", "spec")
            != expected_runtime_image
        ):
            raise ReleaseError(
                f"{target.cluster_id} rollback Reconciler image mismatch"
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
    if ReleaseComponent.DCGM in components:
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
    if ReleaseComponent.AGENT in components:
        validate_agent_rollback_target(release, previous, target)
    if run_verifier and components.intersection(
        {
            ReleaseComponent.EXECUTOR,
            ReleaseComponent.WATCHER,
            ReleaseComponent.COLLECTOR,
            ReleaseComponent.RECONCILER,
            ReleaseComponent.AGENT,
        }
    ):
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
                "GPU_FAULT_SYNC_INSTALLED_RESOURCE_REGISTRY": "false",
            },
        )


def _validate_gpu_rollback(
    release: Any,
    previous: dict[str, Any],
    expected_runtime_image: str,
) -> None:
    _fan_out_clusters(
        release,
        lambda target: validate_gpu_rollback_target(
            release,
            previous,
            expected_runtime_image,
            target,
        ),
        failure="rollback validation",
    )


def _validate_agent_rollback(
    release: Any,
    previous: dict[str, Any],
) -> None:
    script = probe_source("agent_rollback_identity")
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
    exec_cpu_ingress_probe(
        release,
        script=script,
        failure="rollback Agent identity validation",
        input_text=json.dumps(expected),
    )


def validate_rollback(
    release: Any,
    previous: dict[str, Any],
    *,
    restore_cpu: bool = True,
    cluster_components: dict[str, frozenset[ReleaseComponent]] | None = None,
) -> None:
    if release.runner.dry_run:
        return
    metadata = dict(previous.get("metadata") or {})
    current_metadata = release._config_map_data("gpu-fault-release-metadata")
    for key, expected in metadata.items():
        if current_metadata.get(key) != expected:
            raise ReleaseError(f"rollback release metadata mismatch: {key}")
    expected_runtime_image = previous.get("runtime_image") or release.runtime_image
    if restore_cpu:
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
                str(
                    ROOT / "deploy/control-plane/tools/"
                    "verify-control-plane-role-split.sh"
                ),
            ],
            env={
                **os.environ,
                "KUBECONFIG": release.config.cpu_kubeconfig,
                "GPU_FAULT_NAMESPACE": release.config.namespace,
                "GPU_FAULT_RUNTIME_IMAGE": expected_runtime_image,
                "GPU_FAULT_SYNC_INSTALLED_RESOURCE_REGISTRY": "false",
            },
        )
    if cluster_components is None:
        _validate_gpu_rollback(release, previous, expected_runtime_image)
        return
    _fan_out_clusters(
        release,
        lambda target: validate_gpu_rollback_target(
            release,
            previous,
            expected_runtime_image,
            target,
            components=cluster_components[target.cluster_id],
        ),
        failure="rollback validation",
        targets=[
            target
            for target in release.config.clusters
            if cluster_components.get(target.cluster_id)
        ],
    )

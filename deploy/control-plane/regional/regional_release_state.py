from __future__ import annotations

import copy
import hashlib
import json
import re
import tempfile
import threading
import time
from collections.abc import Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable

import regional_deployment_inventory as inventory
import yaml  # type: ignore[import-untyped,unused-ignore]
from regional_release_config import ClusterTarget, ReleaseError
from regional_release_diff import ReleaseComponent, ReleaseExecutionPlan
from regional_release_legacy import AGENT_IDENTITY_FIELDS
from regional_release_narration import narrate_phase
from regional_release_probes import probe_source
from regional_release_runtime_identity import exec_cpu_ingress_probe

from gpu_fault.admin.config import (
    AdminConfig,
    AdminConfigError,
    default_admin_config,
)
from gpu_fault.release_state_snapshot import (
    ReleaseStateSnapshotError,
    canonical_previous_bytes,
    encode_previous_snapshot,
    hydrate_previous_snapshot,
    validate_snapshot_config_map,
)

STATE_CONFIG_MAP = "gpu-fault-regional-release-state"
PREVIOUS_SNAPSHOT_LABEL = "gpu-fault.io/release-previous-snapshot"
PREVIOUS_SNAPSHOT_DIGEST_ANNOTATION = "gpu-fault.io/snapshot-sha256"
MAX_RELEASE_STATE_BYTES = 700 * 1024
MAX_CAPTURE_WORKERS = 8
SENSITIVE_CONFIG_KEY = re.compile(r"(?:SECRET|TOKEN|PASSWORD|CREDENTIAL|PRIVATE_KEY)")
DIGEST_IMAGE = re.compile(r"^.+@sha256:[0-9a-f]{64}$")


def remote_command_stats(release: Any) -> dict[str, Any]:
    return json.loads(
        exec_cpu_ingress_probe(
            release,
            script=probe_source("remote_command_stats"),
            failure="remote command statistics",
            sensitive=True,
            interactive=False,
        )
    )


_STATE_LOCK_GUARD = threading.Lock()


@contextmanager
def state_transaction(release: Any):
    """Serialize one read-modify-write of the release state.

    `save_state` derives its payload from `release.state` and writes it to a
    single ConfigMap, so anything that reads the state, derives an update from
    it and checkpoints the result has to hold this lock for the whole sequence:
    two GPU clusters rolling in parallel would otherwise each derive from the
    same base and drop the other's cluster entry. The lock is reentrant so a
    caller can checkpoint from inside its own transaction.
    """

    lock = getattr(release, "_state_lock", None)
    if lock is None:
        with _STATE_LOCK_GUARD:
            lock = getattr(release, "_state_lock", None)
            if lock is None:
                lock = threading.RLock()
                release._state_lock = lock
    with lock:
        yield


def cached_read(
    cache: dict[tuple[str, ...], Future[dict[str, Any]]],
    lock: threading.Lock,
    key: tuple[str, ...],
    fetch: Callable[[], dict[str, Any]],
    after: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Read `key` once per snapshot, even when several threads ask at once.

    A `Future` is published under the key *before* the read starts, so the
    second caller waits on the first read instead of issuing its own: the checks
    fan out over a thread pool and several of them ask for the same Deployment
    and the same ACM certificate, which without this arrive as duplicate
    concurrent calls rather than as cache hits.

    Every caller gets its own deepcopy. Callers mutate what they are handed --
    `capture_previous` rewrites fields on the documents it reads -- and a shared
    object would carry that mutation to whoever reads the key next.
    """

    with lock:
        future = cache.get(key)
        owner = future is None
        if owner:
            future = Future()
            cache[key] = future
    assert future is not None
    if not owner:
        return copy.deepcopy(future.result())
    try:
        value = fetch()
        if after is not None:
            after(value)
    except Exception as exc:
        future.set_exception(exc)
        raise
    future.set_result(value)
    return copy.deepcopy(value)


# Anything else -- `create-`, `put-`, `change-`, `modify-` -- must reach AWS every
# time, and a verb this list has not seen is treated as one of those until it can
# be reviewed.
AWS_READ_ONLY_VERBS = ("describe-", "get-", "list-")


def aws_read_only(arguments: Sequence[str]) -> bool:
    return len(arguments) >= 2 and arguments[1].startswith(AWS_READ_ONLY_VERBS)


def aws_json(
    release: Any,
    arguments: Sequence[str],
    *,
    region: bool = True,
    sensitive: bool = False,
    cached: bool = True,
) -> dict[str, Any]:
    """One read-only `aws` call, served from the open snapshot when there is one.

    Every AWS read in a release funnels through here so that a question asked
    twice inside one snapshot costs one round trip. Because the callers fan out
    over thread pools -- the preflight validates clusters in parallel, and one
    IAM role expansion is a tree of small reads -- the collapse has to hold for
    concurrent askers too, which is why `cached_read` publishes a promise before
    it reads.

    What this deliberately does not do is dedupe across snapshots. Two admin
    reports in one deploy ask `sesv2 get-account` again on purpose: the later
    report exists to observe what the release changed, and a cache that outlived
    the snapshot would answer it with the reading from before.

    `cached=False` is for a read inside a wait loop, where serving the previous
    answer would turn "not converged yet" into a loop that can never end.
    """

    command = ["aws", *arguments]
    if region:
        command.extend(["--region", release.config.aws_region])
    command.extend(["--output", "json"])

    def fetch() -> dict[str, Any]:
        raw = release.runner.run(command, capture=True, sensitive=sensitive)
        value: dict[str, Any] = json.loads(raw) if raw else {}
        return value

    cache = getattr(release, "_aws_read_cache", None)
    if cache is None or not cached or not aws_read_only(arguments):
        return fetch()
    return cached_read(cache, release._json_read_cache_lock, tuple(command), fetch)


def get_json(release: Any, args: list[str]) -> dict[str, Any]:
    def fetch() -> dict[str, Any]:
        raw = release.runner.run(args + ["-o", "json"], capture=True)
        return json.loads(raw) if raw else {}

    cache = getattr(release, "_json_read_cache", None)
    if cache is None or "get" not in args:
        return fetch()

    lock: threading.Lock = release._json_read_cache_lock
    return cached_read(
        cache,
        lock,
        tuple(args),
        fetch,
        after=lambda value: _cache_list_items(cache, args, value, lock),
    )


def _cache_list_items(
    cache: dict[tuple[str, ...], Future[dict[str, Any]]],
    args: list[str],
    value: dict[str, Any],
    lock: threading.Lock,
) -> None:
    try:
        get_index = args.index("get")
    except ValueError:
        return
    if len(args) != get_index + 2 or not isinstance(value.get("items"), list):
        return
    prefix = args[: get_index + 2]
    with lock:
        for item in value["items"]:
            if not isinstance(item, dict):
                continue
            name = str((item.get("metadata") or {}).get("name") or "")
            if not name:
                continue
            item_key = tuple([*prefix, name])
            if item_key in cache:
                continue
            future: Future[dict[str, Any]] = Future()
            future.set_result(item)
            cache[item_key] = future


@contextmanager
def read_snapshot(release: Any):
    """Serve every read-only `kubectl get` in the block from one observation.

    A nested block *joins* the snapshot already open instead of starting a
    second one. `status` runs the health report and the release summary inside
    one snapshot; if the health report reset the cache, the two halves of a
    single status output would describe two different observations of the
    cluster, and every read the outer block had already paid for would be paid
    for again.

    Nothing that polls for convergence may run inside a snapshot at all -- see
    `regional_release_gpu_rollout.gpu_node_items` -- so joining does not widen
    that hazard; it only stops a nested block from discarding a warm cache.
    """

    if getattr(release, "_json_read_cache", None) is not None:
        yield
        return
    release._json_read_cache = {}
    # AWS reads share the snapshot's lifetime but not its key space: `_get_json`
    # keys on a kubectl argv and `_aws_json` on an aws argv, and one dictionary
    # holding both would make a collision between the two a silent wrong answer
    # rather than a name clash. They share the lock -- it guards the two
    # dictionaries for a few instructions each, never a read.
    release._aws_read_cache = {}
    release._json_read_cache_lock = threading.Lock()
    try:
        yield
    finally:
        release._json_read_cache = None
        release._aws_read_cache = None
        release._json_read_cache_lock = None


def _plan_captures_gpu(plan: ReleaseExecutionPlan | None) -> bool:
    return plan is None or plan.has(
        ReleaseComponent.ENDPOINT,
        ReleaseComponent.DCGM,
        ReleaseComponent.EXECUTOR,
        ReleaseComponent.WATCHER,
        ReleaseComponent.COLLECTOR,
        ReleaseComponent.RECONCILER,
        ReleaseComponent.AGENT,
    )


def prime_deployment_snapshot(
    release: Any,
    plan: ReleaseExecutionPlan | None = None,
) -> None:
    if not getattr(release, "_deployment_snapshot_enabled", False):
        return
    commands = [
        release._cpu(
            "-n",
            release.config.namespace,
            "get",
            "deployment",
        )
    ]
    if _plan_captures_gpu(plan):
        commands.extend(
            (
                release._gpu(
                    target,
                    "-n",
                    release.config.namespace,
                    "get",
                    "deployment",
                )
                for target in release.config.clusters
            )
        )
    with ThreadPoolExecutor(max_workers=min(8, len(commands))) as executor:
        for future in (
            executor.submit(release._get_json, command) for command in commands
        ):
            future.result()


def config_map_data(release: Any, name: str) -> dict[str, str]:
    value = release._get_json(
        release._cpu(
            "-n",
            release.config.namespace,
            "get",
            "configmap",
            name,
        )
    )
    return dict(value.get("data") or {})


def deployment_wheel(
    release: Any,
    args: list[str],
    deployment: str,
) -> str | None:
    value = release._get_json(
        args
        + [
            "-n",
            release.config.namespace,
            "get",
            "deployment",
            deployment,
        ]
    )
    for volume in (
        value.get("spec", {}).get("template", {}).get("spec", {}).get("volumes", [])
    ):
        if volume.get("name") == "artifact":
            return (volume.get("configMap") or {}).get("name")
    return None


def deployment_image(
    release: Any,
    args: list[str],
    deployment: str,
    *,
    container_name: str | None = None,
) -> str | None:
    value = release._get_json(
        args
        + [
            "-n",
            release.config.namespace,
            "get",
            "deployment",
            deployment,
        ]
    )
    containers = (
        value.get("spec", {}).get("template", {}).get("spec", {}).get("containers", [])
    )
    for container in containers:
        if container_name is None or container.get("name") == container_name:
            image = str(container.get("image") or "").strip()
            return image or None
    return None


def deployment_env_value(
    release: Any,
    args: list[str],
    deployment: str,
    name: str,
) -> str | None:
    value = release._get_json(
        args
        + [
            "-n",
            release.config.namespace,
            "get",
            "deployment",
            deployment,
        ]
    )
    containers = (
        value.get("spec", {}).get("template", {}).get("spec", {}).get("containers", [])
    )
    for container in containers:
        for environment in container.get("env", []):
            if environment.get("name") == name:
                result = str(environment.get("value") or "").strip()
                return result or None
    return None


def template_container_image(
    text: str,
    *,
    container_name: str,
) -> str | None:
    for document in yaml.safe_load_all(text):
        containers = (
            (document or {})
            .get("spec", {})
            .get("template", {})
            .get("spec", {})
            .get("containers", [])
        )
        for container in containers:
            if container.get("name") == container_name:
                image = str(container.get("image") or "").strip()
                return image or None
    return None


def require_consistent_images(
    description: str,
    images: dict[str, str | None],
) -> str:
    missing = sorted(name for name, image in images.items() if not image)
    if missing:
        raise ReleaseError(
            f"cannot capture previous {description} image from: " + ", ".join(missing)
        )
    distinct = {str(image) for image in images.values()}
    if len(distinct) != 1:
        raise ReleaseError(
            f"previous {description} images are inconsistent across: "
            + ", ".join(sorted(images))
        )
    return distinct.pop()


def config_map_binary_key(
    release: Any,
    kubectl: list[str],
    name: str | None,
) -> str | None:
    if not name:
        return None
    value = release._get_json(
        kubectl
        + [
            "-n",
            release.config.namespace,
            "get",
            "configmap",
            name,
        ]
    )
    keys = sorted((value.get("binaryData") or {}).keys())
    return keys[0] if len(keys) == 1 else None


def cpu_role_config_maps(release: Any) -> dict[str, dict[str, str]]:
    names: set[str] = set()
    for deployment in inventory.CPU_RUNTIME_DEPLOYMENTS:
        value = release._get_json(
            release._cpu(
                "-n",
                release.config.namespace,
                "get",
                "deployment",
                deployment,
            )
        )
        pod_spec = value.get("spec", {}).get("template", {}).get("spec", {})
        for container in [
            *pod_spec.get("initContainers", []),
            *pod_spec.get("containers", []),
        ]:
            for source in container.get("envFrom", []):
                name = (source.get("configMapRef") or {}).get("name")
                if (
                    isinstance(name, str)
                    and name.startswith("gpu-fault-")
                    and "-config-" in name
                ):
                    names.add(name)

    snapshots: dict[str, dict[str, str]] = {}
    for name in sorted(names):
        data = release._config_map_data(name)
        sensitive = sorted(key for key in data if SENSITIVE_CONFIG_KEY.search(key))
        if sensitive:
            raise ReleaseError(
                f"role ConfigMap {name} contains sensitive-looking keys: "
                + ", ".join(sensitive)
            )
        snapshots[name] = data
    return snapshots


def captured_admin_config(
    release: Any,
    snapshots: dict[str, dict[str, str]],
) -> AdminConfig:
    try:
        defaults = default_admin_config()
        capacity_defaults = defaults.capacity
        worker = release._get_json(
            release._cpu(
                "-n",
                release.config.namespace,
                "get",
                "deployment",
                "gpu-fault-control-worker",
            )
        )
        spool = release._get_json(
            release._cpu(
                "-n",
                release.config.namespace,
                "get",
                "deployment",
                "gpu-fault-telemetry-spool-worker",
            )
        )
        ingress_telemetry = snapshots.get(
            "gpu-fault-api-ha-config-telemetry",
            {},
        )
        worker_core = snapshots.get("gpu-fault-control-worker-config-core", {})
        ingress_processor = snapshots.get(
            "gpu-fault-api-ha-config-processor",
            {},
        )
        ingress_recovery = snapshots.get(
            "gpu-fault-api-ha-config-recovery",
            {},
        )
        ingress_notification = snapshots.get(
            "gpu-fault-api-ha-config-notification",
            {},
        )
        spool_enabled = ingress_telemetry.get(
            "GPU_FAULT_TELEMETRY_SPOOL",
            str(capacity_defaults.telemetry_spool.enabled).lower(),
        )
        if spool_enabled not in {"true", "false"}:
            raise AdminConfigError("live ingress telemetry spool value is invalid")
        worker_replicas = (worker.get("spec") or {}).get("replicas")
        spool_replicas = (spool.get("spec") or {}).get("replicas")
        remediation = capacity_defaults.remediation
        processor = defaults.processor
        workflow = defaults.workflow
        notification = defaults.notification_delivery
        evidence = defaults.evidence
        return AdminConfig.from_mapping(
            {
                "schema_version": 1,
                "capacity": {
                    "control_worker_replicas": int(
                        capacity_defaults.control_worker_replicas
                        if worker_replicas is None
                        else worker_replicas
                    ),
                    "telemetry_spool": {
                        "enabled": spool_enabled == "true",
                        "replicas": int(
                            capacity_defaults.telemetry_spool.replicas
                            if spool_replicas is None
                            else spool_replicas
                        ),
                    },
                    "remediation": {
                        "max_active_region": int(
                            worker_core.get(
                                "GPU_FAULT_REMEDIATION_MAX_ACTIVE_REGION",
                                remediation.max_active_region,
                            )
                        ),
                        "max_active_per_cluster": int(
                            worker_core.get(
                                "GPU_FAULT_REMEDIATION_MAX_ACTIVE_PER_CLUSTER",
                                remediation.max_active_per_cluster,
                            )
                        ),
                        "max_active_per_node": int(
                            worker_core.get(
                                "GPU_FAULT_REMEDIATION_MAX_ACTIVE_PER_NODE",
                                remediation.max_active_per_node,
                            )
                        ),
                        "max_active_per_failure_domain": int(
                            worker_core.get(
                                ("GPU_FAULT_REMEDIATION_MAX_ACTIVE_PER_FAILURE_DOMAIN"),
                                remediation.max_active_per_failure_domain,
                            )
                        ),
                        "max_active_per_resource_class": int(
                            worker_core.get(
                                ("GPU_FAULT_REMEDIATION_MAX_ACTIVE_PER_RESOURCE_CLASS"),
                                remediation.max_active_per_resource_class,
                            )
                        ),
                    },
                },
                "processor": {
                    "max_queue_depth": int(
                        ingress_processor.get(
                            "GPU_FAULT_PROCESSOR_MAX_QUEUE_DEPTH",
                            processor.max_queue_depth,
                        )
                    ),
                    "max_cluster_queue_depth": int(
                        ingress_processor.get(
                            "GPU_FAULT_PROCESSOR_MAX_CLUSTER_QUEUE_DEPTH",
                            processor.max_cluster_queue_depth,
                        )
                    ),
                    "retry_after_seconds": int(
                        ingress_processor.get(
                            "GPU_FAULT_PROCESSOR_RETRY_AFTER_SECONDS",
                            processor.retry_after_seconds,
                        )
                    ),
                    "retry_backoff_seconds": int(
                        ingress_processor.get(
                            "GPU_FAULT_PROCESSOR_RETRY_BACKOFF_SECONDS",
                            processor.retry_backoff_seconds,
                        )
                    ),
                    "retry_backoff_max_seconds": int(
                        ingress_processor.get(
                            "GPU_FAULT_PROCESSOR_RETRY_BACKOFF_MAX_SECONDS",
                            processor.retry_backoff_max_seconds,
                        )
                    ),
                    "completed_retention_seconds": int(
                        ingress_processor.get(
                            ("GPU_FAULT_PROCESSOR_COMPLETED_RETENTION_SECONDS"),
                            processor.completed_retention_seconds,
                        )
                    ),
                },
                "workflow": {
                    "poll_interval_seconds": float(
                        ingress_recovery.get(
                            "GPU_FAULT_WORKFLOW_POLL_INTERVAL_SECONDS",
                            workflow.poll_interval_seconds,
                        )
                    ),
                    "dispatcher_workers": int(
                        ingress_recovery.get(
                            "GPU_FAULT_WORKFLOW_DISPATCHER_WORKERS",
                            workflow.dispatcher_workers,
                        )
                    ),
                },
                "notification_delivery": {
                    "batch_size": int(
                        ingress_notification.get(
                            "GPU_FAULT_NOTIFICATION_BATCH_SIZE",
                            notification.batch_size,
                        )
                    ),
                    "max_attempts": int(
                        ingress_notification.get(
                            "GPU_FAULT_NOTIFICATION_MAX_ATTEMPTS",
                            notification.max_attempts,
                        )
                    ),
                },
                "evidence": {
                    "retention_hours": int(
                        ingress_processor.get(
                            "GPU_FAULT_EVIDENCE_RETENTION_HOURS",
                            evidence.retention_hours,
                        )
                    ),
                    "max_records_per_node": int(
                        ingress_processor.get(
                            "GPU_FAULT_EVIDENCE_MAX_RECORDS_PER_NODE",
                            evidence.max_records_per_node,
                        )
                    ),
                },
            }
        )
    except (AdminConfigError, KeyError, TypeError, ValueError) as exc:
        raise ReleaseError("cannot capture a valid live administrator config") from exc


def capture_agent_identities(release: Any) -> dict[str, dict[str, Any]]:
    raw = exec_cpu_ingress_probe(
        release,
        script=probe_source("active_agent_identities"),
        failure="previous Agent identity capture",
    )
    records = json.loads(raw)
    result: dict[str, dict[str, Any]] = {}
    for target in release.config.clusters:
        selected = [
            item for item in records if item.get("cluster_id") == target.cluster_id
        ]
        if not selected:
            raise ReleaseError(
                f"{target.cluster_id} has no active Agent identity to capture"
            )
        identities = {
            tuple(item.get(field) for field in AGENT_IDENTITY_FIELDS)
            for item in selected
        }
        if len(identities) != 1:
            raise ReleaseError(f"{target.cluster_id} active Agent identities differ")
        identity = dict(zip(AGENT_IDENTITY_FIELDS, identities.pop(), strict=True))
        identity["node_ids"] = sorted(str(item["node_id"]) for item in selected)
        result[target.cluster_id] = identity
    return result


def capture_gpu_cluster_snapshot(
    release: Any,
    target: ClusterTarget,
) -> tuple[dict[str, Any], dict[str, str | None], str | None]:
    template = release._deployment_template_name(target)
    executor_wheel = release._deployment_wheel(
        release._gpu(target),
        inventory.GPU_EXECUTOR_DEPLOYMENT,
    )
    reconciler_wheel = release._deployment_wheel(
        release._gpu(target),
        inventory.GPU_RECONCILER_DEPLOYMENT,
    )
    bundle_name = release._template_bundle(target, template) if template else None
    template_sha256 = deployment_env_value(
        release,
        release._gpu(target),
        inventory.GPU_RECONCILER_DEPLOYMENT,
        "GPU_FAULT_INSTALLER_TEMPLATE_SHA256",
    )
    template_text = None
    if template:
        template_value = release._get_json(
            release._gpu(
                target,
                "-n",
                release.config.namespace,
                "get",
                "configmap",
                template,
            )
        )
        template_text = (template_value.get("data") or {}).get("job.yaml")
        if template_text and not template_sha256:
            template_sha256 = hashlib.sha256(template_text.encode()).hexdigest()
    installer_image = (
        template_container_image(
            template_text,
            container_name="installer",
        )
        if template_text
        else None
    )
    runtime_images = {
        f"{target.cluster_id}/{deployment}": deployment_image(
            release,
            release._gpu(target),
            deployment,
        )
        for deployment in (
            *inventory.DEPLOYMENTS,
            inventory.GPU_RECONCILER_DEPLOYMENT,
        )
    }
    bundle_key = release._config_map_binary_key(
        release._gpu(target),
        bundle_name,
    )
    bundle_sha256 = (
        release._config_map_sha(
            release._gpu(target),
            bundle_name,
            bundle_key or release.config.bundle.name,
        )
        if bundle_name
        else None
    )
    snapshot = {
        "wheel": executor_wheel,
        "wheel_key": release._config_map_binary_key(
            release._gpu(target),
            executor_wheel,
        ),
        "reconciler_wheel": reconciler_wheel,
        "reconciler_wheel_key": release._config_map_binary_key(
            release._gpu(target),
            reconciler_wheel,
        ),
        "template": template,
        "template_sha256": template_sha256,
        "bundle": bundle_name,
        "bundle_key": bundle_key,
        "bundle_sha256": bundle_sha256,
        "dcgm_image": (
            release._get_json(
                release._gpu(
                    target,
                    "-n",
                    release.config.namespace,
                    "get",
                    "daemonset",
                    "gpu-fault-dcgm-exporter",
                )
            )
            .get("spec", {})
            .get("template", {})
            .get("spec", {})
            .get("containers", [{}])[0]
            .get("image")
        ),
    }
    return snapshot, runtime_images, installer_image


def _map_clusters(
    release: Any,
    read: Any,
    *,
    targets: Any,
    failure: str,
) -> list[tuple[ClusterTarget, Any]]:
    """Run a read-only per-cluster capture with bounded parallelism.

    Results come back in the configured cluster order so the captured snapshot
    stays byte-stable, and the first failure is reported with its cluster id.
    """

    selected = list(targets)
    if len(selected) < 2:
        return [(target, read(target)) for target in selected]
    with ThreadPoolExecutor(
        max_workers=min(MAX_CAPTURE_WORKERS, len(selected))
    ) as pool:
        futures = [(target, pool.submit(read, target)) for target in selected]
        results: list[tuple[ClusterTarget, Any]] = []
        errors: list[tuple[str, Exception]] = []
        for target, future in futures:
            try:
                results.append((target, future.result()))
            except Exception as exc:  # noqa: PERF203 - drain every cluster first
                errors.append((target.cluster_id, exc))
    if errors:
        cluster_id, exc = errors[0]
        raise ReleaseError(f"{cluster_id} {failure} failed: {exc}") from exc
    return results


# State fields that describe the release the transaction was *for*. After a
# rollback that PASSED, the state still names the failed candidate (the admin
# retry path needs exactly that), while everything actually running is the
# `previous` snapshot. A snapshot taken from such a state must read these from
# `previous`, or the next upgrade's rollback target would carry the candidate's
# manifest, delivery and rendering digests (live 2026-09-07, BOOT-020 stage 2).
_ROLLED_BACK_TRUTH_KEYS = (
    "release_id",
    "component_digests",
    "release_delivery_sha256",
    "rendered_manifest_sha256",
    "node_template_sha256",
)


def _live_truth_state(state: dict[str, Any]) -> dict[str, Any]:
    """The state with candidate-descriptive fields replaced by what is live."""

    previous = state.get("previous")
    rollback_result = state.get("rollback_result")
    if (
        str(state.get("phase") or "") != "rolled-back"
        or not isinstance(previous, dict)
        or not isinstance(rollback_result, dict)
        or rollback_result.get("status") != "PASSED"
    ):
        return state
    return {
        **state,
        **{key: previous[key] for key in _ROLLED_BACK_TRUTH_KEYS if key in previous},
    }


def _capture_previous(
    release: Any,
    plan: ReleaseExecutionPlan | None = None,
) -> dict[str, Any]:
    capture_gpu = _plan_captures_gpu(plan)
    capture_cpu = plan is None or plan.has(
        ReleaseComponent.CPU_STAGE,
        ReleaseComponent.CPU_FINALIZE,
    )
    capture_observability_state = plan is None or plan.has(
        ReleaseComponent.OBSERVABILITY
    )
    capture_endpoint_state = plan is None or plan.has(ReleaseComponent.ENDPOINT)
    live_state = _live_truth_state(
        dict(release.state) if release.state else release._load_state()
    )
    probe = getattr(release, "_remote_command_stats", None)
    remote = probe() if probe is not None else remote_command_stats(release)
    metadata = release._config_map_data("gpu-fault-release-metadata")
    clusters = {}
    runtime_images = {
        f"cpu/{deployment}": deployment_image(
            release,
            release._cpu(),
            deployment,
        )
        for deployment in inventory.CPU_RUNTIME_DEPLOYMENTS
    }
    node_installer_images: dict[str, str | None] = {}
    # Every cluster snapshot is read-only and scoped to its own GPU cluster, so
    # the captures run concurrently and the results are merged in cluster order
    # to keep the snapshot byte-stable.
    for target, (snapshot, target_images, installer_image) in _map_clusters(
        release,
        lambda target: capture_gpu_cluster_snapshot(release, target),
        targets=release.config.clusters if capture_gpu else (),
        failure="previous-state capture",
    ):
        clusters[target.cluster_id] = snapshot
        runtime_images.update(target_images)
        node_installer_images[target.cluster_id] = installer_image
    live_runtime_image = require_consistent_images("runtime", runtime_images)
    runtime_image = live_runtime_image
    adopted_live_runtime_image = str(
        live_state.get("adopted_live_runtime_image") or ""
    ).strip()
    rollback_runtime_image = str(live_state.get("runtime_image") or "").strip()
    if adopted_live_runtime_image:
        if live_runtime_image != adopted_live_runtime_image:
            raise ReleaseError(
                "live runtime image drifted after legacy release-state adoption"
            )
        if not DIGEST_IMAGE.fullmatch(rollback_runtime_image):
            raise ReleaseError(
                "legacy release-state adoption has no immutable rollback runtime image"
            )
        runtime_image = rollback_runtime_image
    node_installer_image = (
        require_consistent_images("Node Installer", node_installer_images)
        if capture_gpu
        else str(live_state.get("node_installer_image") or release.node_installer_image)
    )
    adot_image = (
        deployment_image(
            release,
            release._cpu(),
            "gpu-fault-adot",
            container_name="collector",
        )
        if capture_observability_state
        else str(live_state.get("adot_image") or release.adot_image)
    )
    if not adot_image:
        raise ReleaseError(
            "cannot capture previous ADOT image from deployment/gpu-fault-adot"
        )
    agent_identities: dict[str, dict[str, Any]] = {}
    if capture_gpu:
        capture_identities = getattr(release, "_capture_agent_identities", None)
        agent_identities = (
            capture_identities()
            if capture_identities is not None
            else capture_agent_identities(release)
        )
        for target, expected_nodes in _map_clusters(
            release,
            lambda target: set(release._target_node_names(target)),
            targets=release.config.clusters,
            failure="active Agent set capture",
        ):
            captured_nodes = set(agent_identities[target.cluster_id]["node_ids"])
            if expected_nodes != captured_nodes:
                raise ReleaseError(
                    f"{target.cluster_id} active Agent set does not match HyperPod nodes"
                )
    role_config_maps = cpu_role_config_maps(release) if capture_cpu else {}
    admin_config = (
        captured_admin_config(release, role_config_maps)
        if capture_cpu
        else AdminConfig.from_mapping(live_state.get("admin_config") or {})
    )
    capture_observability_fn = getattr(
        release,
        "_capture_observability_snapshot",
        None,
    )
    observability = (
        capture_observability_fn()
        if capture_observability_fn is not None and capture_observability_state
        else None
    )
    # The endpoint has to be read before the endpoint component overwrites it:
    # the previous NLB Service only exists as a file in the previous checkout,
    # and the Route53 record is gone the moment `ensure_control_plane_dns`
    # UPSERTs the candidate's. Captured here, beside the observability snapshot,
    # for the same reason and at the same point.
    capture_endpoint_fn = getattr(release, "_capture_endpoint_snapshot", None)
    endpoint = (
        capture_endpoint_fn()
        if capture_endpoint_fn is not None and capture_endpoint_state
        else None
    )
    cpu_wheel = (
        release._deployment_wheel(
            release._cpu(),
            inventory.CPU_INGRESS_DEPLOYMENT,
        )
        if capture_cpu
        else live_state.get("wheel_config_map")
    )
    cpu_wheel_key = (
        release._config_map_binary_key(release._cpu(), cpu_wheel)
        if capture_cpu
        else None
    )
    cpu_wheel_sha256 = (
        release._config_map_sha(
            release._cpu(),
            cpu_wheel,
            cpu_wheel_key or release.config.wheel.name,
        )
        if capture_cpu and cpu_wheel
        else None
    )
    result = {
        "release_id": live_state.get("release_id"),
        "metadata": metadata,
        "component_digests": dict(live_state.get("component_digests") or {}),
        "agent_identities": agent_identities,
        "runtime_profile_version": release._config_map_data(
            "gpu-fault-api-ha-config-core"
        ).get("GPU_FAULT_REQUIRED_RUNTIME_PROFILE_VERSION"),
        "cpu_wheel": cpu_wheel,
        "cpu_wheel_sha256": cpu_wheel_sha256,
        "cpu_role_config_maps": role_config_maps,
        "admin_config": admin_config.as_dict(),
        "executor_internal_error_total": int(
            remote.get("executor_internal_error_total", 0) or 0
        ),
        "release_delivery_sha256": live_state.get("release_delivery_sha256"),
        "rendered_manifest_sha256": live_state.get("rendered_manifest_sha256"),
        "node_template_sha256": live_state.get("node_template_sha256"),
        "live_runtime_image": live_runtime_image,
        "runtime_image": runtime_image,
        "node_installer_image": node_installer_image,
        "adot_image": adot_image,
        "observability": observability,
        "endpoint": endpoint,
        "clusters": clusters,
    }
    timestamp_field = "executor_internal_error_last_seen_timestamp_seconds"
    if remote.get(timestamp_field) is not None:
        result[timestamp_field] = float(remote[timestamp_field])
    return result


def capture_previous(
    release: Any,
    plan: ReleaseExecutionPlan | None = None,
) -> dict[str, Any]:
    with read_snapshot(release):
        prime_deployment_snapshot(release, plan)
        return _capture_previous(release, plan)


def _snapshot_config_map(
    release: Any,
    name: str,
) -> dict[str, Any]:
    return release._get_json(
        release._cpu(
            "-n",
            release.config.namespace,
            "get",
            "configmap",
            name,
        )
    )


def ensure_previous_snapshot(
    release: Any,
    previous: dict[str, Any],
) -> dict[str, Any]:
    cached = getattr(release, "_previous_snapshot_reference", None)
    if (
        isinstance(cached, dict)
        and cached.get("sha256")
        == hashlib.sha256(canonical_previous_bytes(previous)).hexdigest()
    ):
        # Every save_state re-derives this reference, so the unchanged case must
        # not pay for gzip and per-chunk hashing: the canonical digest alone
        # decides whether the externalized snapshot is still the same one.
        return cached
    reference, chunks = encode_previous_snapshot(previous)
    if release.runner.dry_run:
        release._previous_snapshot_reference = reference
        return reference
    with tempfile.TemporaryDirectory(prefix="gpu-fault-release-previous-") as directory:
        root = Path(directory)
        for chunk_ref, (name, chunk) in zip(
            reference["chunks"],
            chunks,
            strict=True,
        ):
            exists = release.runner.probe(
                release._cpu(
                    "-n",
                    release.config.namespace,
                    "get",
                    "configmap",
                    name,
                ),
            )
            if exists:
                current = _snapshot_config_map(release, name)
                annotations = (current.get("metadata") or {}).get("annotations") or {}
                if (
                    current.get("immutable") is not True
                    or annotations.get(PREVIOUS_SNAPSHOT_DIGEST_ANNOTATION)
                    != reference["sha256"]
                ):
                    raise ReleaseError(
                        f"previous snapshot ConfigMap identity changed: {name}"
                    )
                try:
                    validate_snapshot_config_map(
                        current,
                        key=str(chunk_ref["key"]),
                        expected_sha256=str(chunk_ref["sha256"]),
                        expected_size=int(chunk_ref["size_bytes"]),
                    )
                except ReleaseStateSnapshotError as exc:
                    raise ReleaseError(str(exc)) from exc
                continue
            chunk_path = root / f"{name}.part"
            chunk_path.write_bytes(chunk)
            rendered = release.runner.run(
                release._cpu(
                    "-n",
                    release.config.namespace,
                    "create",
                    "configmap",
                    name,
                    f"--from-file={chunk_ref['key']}={chunk_path}",
                    "--dry-run=client",
                    "-o",
                    "yaml",
                ),
                capture=True,
            )
            document = yaml.safe_load(rendered)
            if not isinstance(document, dict):
                raise ReleaseError("cannot render previous snapshot ConfigMap")
            metadata = document.setdefault("metadata", {})
            metadata["labels"] = {
                **dict(metadata.get("labels") or {}),
                PREVIOUS_SNAPSHOT_LABEL: "true",
            }
            metadata["annotations"] = {
                **dict(metadata.get("annotations") or {}),
                PREVIOUS_SNAPSHOT_DIGEST_ANNOTATION: reference["sha256"],
            }
            document["immutable"] = True
            release.runner.run(
                release._cpu("create", "-f", "-"),
                input_text=yaml.safe_dump(document, sort_keys=True),
            )
    release._previous_snapshot_reference = reference
    return reference


def cleanup_previous_snapshots(release: Any) -> None:
    if release.runner.dry_run:
        return
    reference = release.state.get("previous_snapshot")
    keep = {
        str(item.get("config_map"))
        for item in ((reference or {}).get("chunks") or [])
        if isinstance(item, dict) and item.get("config_map")
    }
    value = release._get_json(
        release._cpu(
            "-n",
            release.config.namespace,
            "get",
            "configmap",
            "-l",
            f"{PREVIOUS_SNAPSHOT_LABEL}=true",
        )
    )
    stale = sorted(
        str((item.get("metadata") or {}).get("name") or "")
        for item in value.get("items", [])
        if str((item.get("metadata") or {}).get("name") or "")
        and str((item.get("metadata") or {}).get("name") or "") not in keep
    )
    if stale:
        release.runner.run(
            release._cpu(
                "-n",
                release.config.namespace,
                "delete",
                "configmap",
                *stale,
            )
        )


def render_persisted_state(release: Any) -> tuple[dict[str, Any], str]:
    """Externalize the previous snapshot and serialize the state exactly once.

    The document is only ever written or size-checked, and this function only
    adds or removes top-level keys, so a shallow copy is enough: deep-copying a
    state that is allowed to approach `MAX_RELEASE_STATE_BYTES` on every
    checkpoint was pure overhead.
    """

    persisted = dict(release.state)
    previous = persisted.get("previous")
    if isinstance(previous, dict):
        reference = ensure_previous_snapshot(release, previous)
        persisted["previous_snapshot"] = reference
        persisted["previous_snapshot_sha256"] = reference["sha256"]
        persisted.pop("previous", None)
        release.state["previous_snapshot"] = reference
        release.state["previous_snapshot_sha256"] = reference["sha256"]
    elif previous is None:
        persisted.pop("previous_snapshot", None)
        persisted.pop("previous_snapshot_sha256", None)
        release.state.pop("previous_snapshot", None)
        release.state.pop("previous_snapshot_sha256", None)
    text = json.dumps(persisted, indent=2, sort_keys=True)
    if len(text.encode()) > MAX_RELEASE_STATE_BYTES:
        raise ReleaseError(
            "regional release state exceeds the bounded ConfigMap payload"
        )
    return persisted, text


def persisted_state(release: Any) -> dict[str, Any]:
    return render_persisted_state(release)[0]


def deployment_template_name(
    release: Any,
    target: ClusterTarget,
) -> str | None:
    value = release._get_json(
        release._gpu(
            target,
            "-n",
            release.config.namespace,
            "get",
            "deployment",
            inventory.GPU_RECONCILER_DEPLOYMENT,
        )
    )
    for volume in (
        value.get("spec", {}).get("template", {}).get("spec", {}).get("volumes", [])
    ):
        if volume.get("name") == "installer-template":
            return (volume.get("configMap") or {}).get("name")
    return None


def template_bundle(
    release: Any,
    target: ClusterTarget,
    template_name: str,
) -> str | None:
    value = release._get_json(
        release._gpu(
            target,
            "-n",
            release.config.namespace,
            "get",
            "configmap",
            template_name,
        )
    )
    text = (value.get("data") or {}).get("job.yaml")
    if not text:
        return None
    for document in yaml.safe_load_all(text):
        for volume in (
            (document or {})
            .get("spec", {})
            .get("template", {})
            .get("spec", {})
            .get("volumes", [])
        ):
            if volume.get("name") == "installer":
                return (volume.get("configMap") or {}).get("name")
    return None


def save_state(release: Any, phase: str, **updates: Any) -> None:
    with state_transaction(release):
        _write_state(release, phase, **updates)
        narrate_phase(release, phase)


def _write_state(release: Any, phase: str, **updates: Any) -> None:
    if release.state.get("release_id") not in {None, release.release_id}:
        release.state.pop("adopted_live_runtime_image", None)
    release.state.update(
        {
            "phase": phase,
            "release_id": release.release_id,
            "wheel_sha256": release.wheel_sha,
            "executor_wheel_sha256": release.executor_wheel_sha,
            "node_wheel_sha256": release.node_wheel_sha,
            "bundle_sha256": release.bundle_sha,
            "wheel_config_map": release.wheel_cm,
            "executor_wheel_config_map": release.executor_wheel_cm,
            "bundle_config_map": release.bundle_cm,
            "database_schema_version": release.config.database_schema_version,
            "agent_protocol_version": release.config.agent_protocol_version,
            "executor_protocol_version": release.config.executor_protocol_version,
            "component_digests": release.config.component_digests,
            "runtime_profile_version": release.config.runtime_profile_version,
            "runtime_profile_sha256": release.runtime_profile_sha,
            "runtime_profile_source_sha256": release.runtime_profile_sha,
            "runtime_profile_template_sha256": release.runtime_profile_template_sha,
            "runtime_profile_policy_sha256": release.runtime_profile_policy_sha,
            "runtime_profile_registration_cluster_id": (
                release.config.runtime_profile_registration_cluster_id
            ),
            "agent_config_digest": release.config.agent_config_digest,
            "endpoint_digest": release.endpoint_digest,
            "dcgm_digest": release.dcgm_digest,
            "notification_digest": release.notification_digest,
            "cluster_ids": sorted(item.cluster_id for item in release.config.clusters),
            "cluster_registry_digest": release.cluster_registry_digest,
            "admin_config_sha256": release.admin_config_digest,
            "admin_config_role_sha256": release.admin_config_role_digests,
            "admin_config": release.config.admin_config.as_dict(),
            "release_manifest_schema_version": (
                release.config.release_manifest_schema_version
            ),
            "fleet_rollout_policy": {
                "upgrade_max_unavailable": (release.config.upgrade_max_unavailable),
                "rollback_max_unavailable": (release.config.rollback_max_unavailable),
                "upgrade_max_parallel_clusters": (
                    release.config.upgrade_max_parallel_clusters
                ),
                "first_rollback_wave_max_unavailable": 1,
                "max_unavailable_per_failure_domain": 1,
                "rollback_parallel_min_nodes": 6,
            },
            "release_delivery_sha256": (release.config.release_delivery_sha256),
            "cpu_manifest_sha256": (
                release.config.delivery_component_digests.get("cpu")
            ),
            "cpu_ingress_manifest_sha256": (
                release.config.delivery_component_digests.get("cpu_ingress")
            ),
            "cpu_worker_manifest_sha256": (
                release.config.delivery_component_digests.get("cpu_worker")
            ),
            "cpu_spool_manifest_sha256": (
                release.config.delivery_component_digests.get("cpu_spool")
            ),
            "executor_manifest_sha256": (
                release.config.delivery_component_digests.get("executor")
            ),
            "watcher_manifest_sha256": (
                release.config.delivery_component_digests.get("watcher")
            ),
            "collector_manifest_sha256": (
                release.config.delivery_component_digests.get("collector")
            ),
            "dcgm_manifest_sha256": (
                release.config.delivery_component_digests.get("dcgm")
            ),
            "node_manifest_sha256": (
                release.config.delivery_component_digests.get("node")
            ),
            "observability_manifest_sha256": (
                release.config.delivery_component_digests.get("observability")
            ),
            "observability_rules_sha256": release.observability_rules_digest,
            "observability_adot_sha256": release.observability_adot_digest,
            "schema_manifest_sha256": (
                release.config.delivery_component_digests.get("schema")
            ),
            "endpoint_manifest_sha256": (
                release.config.delivery_component_digests.get("endpoint")
            ),
            "rendered_manifest_sha256": release.rendered_manifest_digest,
            "node_template_sha256": release.node_template_sha,
            "runtime_image": release.runtime_image,
            "node_installer_image": release.node_installer_image,
            "dcgm_image": release.dcgm_exporter_image,
            "adot_image": release.adot_image,
            "updated_at_epoch": int(time.time()),
            **updates,
        }
    )
    _persisted, text = render_persisted_state(release)
    if release.runner.dry_run:
        return
    # Every component STARTED/FAILED transition checkpoints this ConfigMap, so
    # the write is one `apply` of a manifest built in-process instead of a
    # temp file plus a `create --dry-run=client` render round-trip. The manifest
    # carries its own namespace, exactly like the rendered form it replaces.
    release.runner.run(
        release._cpu("apply", "-f", "-"),
        input_text=json.dumps(
            {
                "apiVersion": "v1",
                "kind": "ConfigMap",
                "metadata": {
                    "name": STATE_CONFIG_MAP,
                    "namespace": release.config.namespace,
                },
                "data": {"state.json": text},
            }
        ),
    )


def load_state(release: Any) -> dict[str, Any]:
    value = release._get_json(
        release._cpu(
            "-n",
            release.config.namespace,
            "get",
            "configmap",
            STATE_CONFIG_MAP,
        )
    )
    raw = (value.get("data") or {}).get("state.json")
    if not raw:
        raise ReleaseError("regional release state is missing")
    try:
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            raise ReleaseStateSnapshotError("regional release state is not an object")
        release.state = hydrate_previous_snapshot(
            parsed,
            lambda name: _snapshot_config_map(release, name),
        )
    except (json.JSONDecodeError, ReleaseStateSnapshotError) as exc:
        raise ReleaseError("regional release state is invalid") from exc
    return release.state

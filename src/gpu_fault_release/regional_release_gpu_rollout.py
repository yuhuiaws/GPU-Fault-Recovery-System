from __future__ import annotations

import json
import sys
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from typing import Any

from gpu_fault_release import regional_deployment_inventory as inventory
import yaml  # type: ignore[import-untyped,unused-ignore]
from gpu_fault_release.regional_release_config import (
    ClusterTarget,
    ReleaseError,
)
from gpu_fault_release.regional_release_diff import (
    ReleaseComponent,
    ReleaseDiff,
    ReleaseExecutionPlan,
    build_execution_plan,
)
from gpu_fault_release.regional_release_rendering import render_gpu_rollout_manifests
from gpu_fault_release.regional_release_rollout_wait import wait_deployment_rollout

from gpu_fault.regional_compatibility import (
    RegionalExecutorCompatibilityPolicy,
)

ProgressSelection = ReleaseComponent | tuple[ReleaseComponent, ...]
ProgressCallback = Callable[[ProgressSelection, str, dict[str, Any] | None], None]
#: Both Completion Watcher state objects, in the order the manifest declares
#: them: the write-ahead outbox, then the routine attempt state that was split
#: out of it so a large fleet's running-attempt records could not fill the object
#: a terminal event is written to. Each is stripped from the applied manifest if
#: it already exists, because the manifest declares them empty and re-applying
#: either would drop live records.
COMPLETION_WATCHER_STATE_CONFIG_MAPS = (
    "gpu-fault-completion-watcher-outbox",
    "gpu-fault-completion-watcher-outbox-active",
)
HYPERPOD_CLUSTER_LABEL = "sagemaker.amazonaws.com/cluster-name"
NODE_INVENTORY_ATTRIBUTE = "_gpu_node_inventory"
#: Label on every Role/RoleBinding this module renders into a workload
#: namespace. Pruning selects by it, so nothing an operator created by hand in
#: the same namespace is ever touched.
WORKLOAD_NAMESPACE_RBAC_LABEL = "gpu-fault.io/workload-namespace-rbac"
#: Where the GPU/EFA device plugin DaemonSets run. RESTART_*_DEVICE_PLUGIN
#: deletes one plugin Pod there (adapters/kubernetes/node_operations.py,
#: ``plugin_namespace`` defaults to kube-system), which is the only Pod delete
#: the executor performs outside a workload namespace.
DEVICE_PLUGIN_NAMESPACES = ("kube-system",)
EXECUTOR_SERVICE_ACCOUNT = "gpu-fault-cluster-executor"
WATCHER_SERVICE_ACCOUNT = "gpu-fault-completion-watcher"
DEVICE_PLUGIN_ROLE = f"{EXECUTOR_SERVICE_ACCOUNT}-device-plugin"

#: The label every GPU training Pod carries once it has been through
#: ``gpu-fault-training-submit``/``gpu-fault-workload-annotate`` -- and the exact
#: selector the Completion Watcher lists managed workloads by (see
#: ``gpu_fault.completion_observation.MANAGED_LABEL`` /
#: ``list_completion_pods``). Reusing it means the coverage preflight sees
#: precisely the Pods the data plane will be asked to recover, nothing more.
#: Kept as a literal rather than imported so the release module stays free of
#: the watcher's Kubernetes-client import chain; a unit test asserts the two
#: never drift.
MANAGED_WORKLOAD_LABEL = "gpu-fault.io/managed"

#: Pod phases that mean the workload is finished and will never call for
#: recovery, so a training Pod sitting in one of them cannot be orphaned by a
#: missing RBAC grant.
TERMINAL_POD_PHASES = frozenset({"Succeeded", "Failed"})

#: The namespaced write verbs the executor needs to restart or stop a workload
#: (adapters/kubernetes/{workload,restart}_operations.py, primitives.py). Read
#: verbs are not repeated here: the ClusterRole already grants get/list/watch,
#: which spare activation needs across every namespace.
EXECUTOR_WORKLOAD_RULES: tuple[dict[str, Any], ...] = (
    {"apiGroups": [""], "resources": ["pods"], "verbs": ["patch", "delete"]},
    {"apiGroups": ["batch"], "resources": ["jobs"], "verbs": ["create", "patch"]},
    {
        "apiGroups": ["kubeflow.org"],
        "resources": ["pytorchjobs"],
        "verbs": ["create", "patch"],
    },
    {
        "apiGroups": ["jobset.x-k8s.io"],
        "resources": ["jobsets"],
        "verbs": ["create", "patch"],
    },
)

#: What the Completion Watcher does *to* a training Pod or its owner: annotate
#: it, suspend the owner on the passive-stop fallback, capture its log, and --
#: with GPU_FAULT_DISCOVER_POD_GPU_UUIDS -- exec ``nvidia-smi --query-gpu=uuid``
#: in the training container. ``pods/exec`` is here and nowhere else:
#: resourceNames cannot scope a subresource, so a namespaced Role is the only
#: grant that does not open a shell into every Pod on the cluster.
WATCHER_WORKLOAD_RULES: tuple[dict[str, Any], ...] = (
    {"apiGroups": [""], "resources": ["pods"], "verbs": ["patch"]},
    {"apiGroups": [""], "resources": ["pods/exec"], "verbs": ["get", "create"]},
    {"apiGroups": [""], "resources": ["pods/log"], "verbs": ["get"]},
    {"apiGroups": ["batch"], "resources": ["jobs"], "verbs": ["patch"]},
    {"apiGroups": ["kubeflow.org"], "resources": ["pytorchjobs"], "verbs": ["patch"]},
    {"apiGroups": ["jobset.x-k8s.io"], "resources": ["jobsets"], "verbs": ["patch"]},
)

DEVICE_PLUGIN_RULES: tuple[dict[str, Any], ...] = (
    {"apiGroups": [""], "resources": ["pods"], "verbs": ["delete"]},
)


def _rbac_metadata(name: str, namespace: str) -> dict[str, Any]:
    return {
        "name": name,
        "namespace": namespace,
        "labels": {WORKLOAD_NAMESPACE_RBAC_LABEL: "true"},
    }


def _role(
    name: str, namespace: str, rules: tuple[dict[str, Any], ...]
) -> dict[str, Any]:
    return {
        "apiVersion": "rbac.authorization.k8s.io/v1",
        "kind": "Role",
        "metadata": _rbac_metadata(name, namespace),
        "rules": [dict(rule) for rule in rules],
    }


def _role_binding(
    name: str,
    namespace: str,
    *,
    service_account: str,
    system_namespace: str,
) -> dict[str, Any]:
    return {
        "apiVersion": "rbac.authorization.k8s.io/v1",
        "kind": "RoleBinding",
        "metadata": _rbac_metadata(name, namespace),
        "roleRef": {
            "apiGroup": "rbac.authorization.k8s.io",
            "kind": "Role",
            "name": name,
        },
        "subjects": [
            {
                "kind": "ServiceAccount",
                "name": service_account,
                "namespace": system_namespace,
            }
        ],
    }


def _role_pair(
    name: str,
    namespace: str,
    rules: tuple[dict[str, Any], ...],
    *,
    service_account: str,
    system_namespace: str,
) -> list[dict[str, Any]]:
    return [
        _role(name, namespace, rules),
        _role_binding(
            name,
            namespace,
            service_account=service_account,
            system_namespace=system_namespace,
        ),
    ]


def render_workload_namespace_rbac(
    allowed_namespaces: tuple[str, ...] | list[str] | frozenset[str],
    *,
    system_namespace: str,
) -> dict[str, list[dict[str, Any]]]:
    """Role + RoleBinding pairs for every namespace the data plane may write to.

    The ClusterRoles in ``deploy/dataplane`` are read-only on workload kinds;
    this is where the write verbs live, one Role per ``allowed_namespaces``
    entry per identity, so the API server enforces the same boundary the
    executor checks in ``GPU_FAULT_ALLOWED_WORKLOAD_NAMESPACES``. The result
    is keyed by the Deployment whose manifest the documents travel with, so
    each pair is applied in the same ``kubectl apply`` -- and the same
    ``--dry-run=server`` preflight -- as the identity it binds.
    """

    namespaces = sorted({str(item).strip() for item in allowed_namespaces if item})
    executor: list[dict[str, Any]] = []
    watcher: list[dict[str, Any]] = []
    for namespace in namespaces:
        executor.extend(
            _role_pair(
                EXECUTOR_SERVICE_ACCOUNT,
                namespace,
                EXECUTOR_WORKLOAD_RULES,
                service_account=EXECUTOR_SERVICE_ACCOUNT,
                system_namespace=system_namespace,
            )
        )
        watcher.extend(
            _role_pair(
                WATCHER_SERVICE_ACCOUNT,
                namespace,
                WATCHER_WORKLOAD_RULES,
                service_account=WATCHER_SERVICE_ACCOUNT,
                system_namespace=system_namespace,
            )
        )
    for namespace in DEVICE_PLUGIN_NAMESPACES:
        executor.extend(
            _role_pair(
                DEVICE_PLUGIN_ROLE,
                namespace,
                DEVICE_PLUGIN_RULES,
                service_account=EXECUTOR_SERVICE_ACCOUNT,
                system_namespace=system_namespace,
            )
        )
    return {
        inventory.GPU_EXECUTOR_DEPLOYMENT: executor,
        inventory.GPU_WATCHER_DEPLOYMENT: watcher,
    }


def rbac_namespaces(allowed_namespaces: tuple[str, ...] | list[str]) -> frozenset[str]:
    """Every namespace a rendered Role may legitimately live in."""

    return frozenset(
        {str(item).strip() for item in allowed_namespaces if item}
        | set(DEVICE_PLUGIN_NAMESPACES)
    )


def managed_workload_namespaces(release: Any, target: ClusterTarget) -> list[str]:
    """Namespaces that currently hold a live GPU training Pod.

    Read-only. Lists the Pods carrying ``gpu-fault.io/managed=true`` across
    every namespace -- the same selector the Completion Watcher uses to find
    the workloads it manages -- and returns the sorted namespaces of those not
    already in a terminal phase. A Pod that has ``Succeeded`` or ``Failed`` is
    done and will never need recovery, so its namespace is not reported.
    """

    listing = release._get_json(
        release._gpu(
            target,
            "get",
            "pods",
            "--all-namespaces",
            "-l",
            f"{MANAGED_WORKLOAD_LABEL}=true",
        )
    )
    namespaces: set[str] = set()
    for item in listing.get("items", []):
        phase = str((item.get("status") or {}).get("phase") or "")
        if phase in TERMINAL_POD_PHASES:
            continue
        namespace = str((item.get("metadata") or {}).get("namespace") or "").strip()
        if namespace:
            namespaces.add(namespace)
    return sorted(namespaces)


def preflight_allowed_namespace_coverage(release: Any, target: ClusterTarget) -> None:
    """Fail closed before rollout if a live training Pod sits outside the allow-list.

    The rollout renders workload Role/RoleBindings *only* for the namespaces in
    ``allowed_namespaces`` (:func:`render_workload_namespace_rbac`), and both the
    cluster executor and the control plane reject any ``workload_id`` whose
    namespace is not in that list. A training Pod in a namespace the operator
    forgot to register therefore gets a silent 403 at recovery time -- the worst
    moment to discover it. This turns that post-rollout silence into a loud
    pre-rollout signal: it names the uncovered namespaces and stops the rollout.

    Strictly read-only by contract. It never widens ``allowed_namespaces`` --
    silently expanding RBAC would be a privilege-escalation risk -- so the
    operator must add the missing namespace to the cluster registration and
    re-run. A target without the ``allowed_namespaces`` attribute (a test
    double) is left alone, exactly as the RBAC rendering leaves it alone.
    """

    allowed = getattr(target, "allowed_namespaces", None)
    if allowed is None:
        return
    covered = {str(item).strip() for item in allowed if item}
    live = managed_workload_namespaces(release, target)
    uncovered = [namespace for namespace in live if namespace not in covered]
    # Print the coverage the rollout is about to authorise even when it passes,
    # so a green preflight still shows the operator precisely which namespaces
    # will be RBAC-authorised and which live workloads they cover.
    print(
        f"{target.cluster_id}: GPU training namespace coverage -- "
        f"allowed_namespaces={sorted(covered)}, "
        f"live managed workload namespaces={live}",
        file=sys.stderr,
        flush=True,
    )
    if uncovered:
        raise ReleaseError(
            f"{target.cluster_id}: GPU training workloads run in "
            f"namespace(s) not in allowed_namespaces: {', '.join(uncovered)}. "
            "The rendered RBAC and the executor/control-plane allow-list will "
            "not cover them, so the data plane would 403 when it tries to "
            "recover those workloads. Add the namespace(s) to this cluster's "
            "registration allowed_namespaces and re-run; the rollout will not "
            "widen the allow-list for you."
        )


def append_workload_namespace_rbac(
    manifests: dict[str, str],
    target: ClusterTarget,
    *,
    system_namespace: str,
) -> dict[str, str]:
    """Attach each identity's namespaced RBAC to its Deployment manifest text.

    A target without ``allowed_namespaces`` (the attribute, not an empty
    tuple) is a test double and is left alone; a real target with an empty
    list still gets the device-plugin Role, because RESTART_*_DEVICE_PLUGIN is
    a node operation that does not go through a workload namespace.
    """

    allowed = getattr(target, "allowed_namespaces", None)
    if allowed is None:
        return manifests
    documents = render_workload_namespace_rbac(
        tuple(allowed), system_namespace=system_namespace
    )
    for deployment, items in documents.items():
        if deployment not in manifests or not items:
            continue
        manifests[deployment] = (
            manifests[deployment].rstrip()
            + "\n---\n"
            + yaml.safe_dump_all(items, sort_keys=False)
        )
    return manifests


def prune_workload_namespace_rbac(release: Any, target: ClusterTarget) -> list[str]:
    """Delete this module's Roles from namespaces no longer in the allow-list.

    ``kubectl apply`` never removes what an earlier release created, so a
    namespace dropped from ``allowed_namespaces`` would keep its write grants
    forever. Selection is by label only, so hand-made objects in the same
    namespace are never touched. Returns the namespaces that were pruned.
    """

    allowed = getattr(target, "allowed_namespaces", None)
    if allowed is None or release.runner.dry_run:
        return []
    keep = rbac_namespaces(tuple(allowed))
    raw = release.runner.run(
        release._gpu(
            target,
            "get",
            "rolebindings",
            "--all-namespaces",
            "-l",
            f"{WORKLOAD_NAMESPACE_RBAC_LABEL}=true",
            "-o",
            "json",
        ),
        capture=True,
    )
    listing = json.loads(raw) if raw else {}
    stale = sorted(
        {
            str(((item.get("metadata") or {}).get("namespace")) or "")
            for item in listing.get("items", [])
        }
        - keep
        - {""}
    )
    for namespace in stale:
        release.runner.run(
            release._gpu(
                target,
                "-n",
                namespace,
                "delete",
                "rolebinding,role",
                "-l",
                f"{WORKLOAD_NAMESPACE_RBAC_LABEL}=true",
                "--ignore-not-found",
            )
        )
    return stale


def gpu_node_command(release: Any, target: ClusterTarget) -> list[str]:
    """Build the one node-inventory read every release path shares.

    The HyperPod cluster label is pushed to the API server so a GPU EKS cluster
    that also hosts non-HyperPod nodes never ships them over the wire, and so
    every caller produces an identical `_get_json` cache key: inside
    `read_snapshot` the whole release reads each cluster's nodes once.
    """

    return release._gpu(
        target,
        "get",
        "nodes",
        "-l",
        f"{HYPERPOD_CLUSTER_LABEL}={target.hyperpod_cluster_name}",
    )


@contextmanager
def node_inventory_scope(release: Any):
    """Pin each cluster's node inventory for one derivation step.

    Values derived from the same node list (the fleet node set and its
    failure-domain map) must describe one observation, and re-listing every
    node once per derivation is the most expensive read in the rollout. Safety
    gates stay outside this scope and read with ``fresh=True``.
    """

    previous = getattr(release, NODE_INVENTORY_ATTRIBUTE, None)
    setattr(release, NODE_INVENTORY_ATTRIBUTE, {})
    try:
        yield
    finally:
        setattr(release, NODE_INVENTORY_ATTRIBUTE, previous)


def gpu_node_items(
    release: Any,
    target: ClusterTarget,
    *,
    fresh: bool = False,
) -> list[dict[str, Any]]:
    # `fresh` opts out of the pinned inventory: convergence polling and safety
    # gates must observe live node state, never a value another step captured
    # seconds earlier. They also must not run inside `read_snapshot`, whose
    # cache would otherwise re-serve the first read until the deadline expires.
    cache = None if fresh else getattr(release, NODE_INVENTORY_ATTRIBUTE, None)
    if cache is not None and target.cluster_id in cache:
        return cache[target.cluster_id]
    value = release._get_json(gpu_node_command(release, target))
    items = list(value.get("items", []))
    if cache is not None:
        cache[target.cluster_id] = items
    return items


def preserve_completion_watcher_state(
    release: Any,
    target: ClusterTarget,
    text: str,
) -> str:
    if release.runner.dry_run:
        return text
    # Probed one object at a time so that the upgrade which introduces the
    # second one still preserves the first: the new object does not exist yet
    # and has to be applied, the outbox exists and must not be.
    existing = set()
    for name in COMPLETION_WATCHER_STATE_CONFIG_MAPS:
        returncode, _stdout, stderr = release.runner.probe_output(
            release._gpu(
                target,
                "-n",
                release.config.namespace,
                "get",
                "configmap",
                name,
            )
        )
        if not returncode:
            existing.add(name)
            continue
        if "NotFound" in stderr or "not found" in stderr:
            continue
        raise ReleaseError(
            f"{target.cluster_id} cannot inspect Completion Watcher state: "
            + stderr.strip()
        )
    if not existing:
        return text
    documents = [
        document
        for document in yaml.safe_load_all(text)
        if not (
            isinstance(document, dict)
            and document.get("kind") == "ConfigMap"
            and (document.get("metadata") or {}).get("name") in existing
        )
    ]
    return yaml.safe_dump_all(documents, sort_keys=False)


def agents_converged(
    items: list[dict[str, Any]],
    target: ClusterTarget,
    artifact_sha: str,
    *,
    bundle_sha: str | None = None,
    template_sha: str | None = None,
    config_digest: str | None = None,
    require_node_uid: bool = False,
    node_names: frozenset[str] | None = None,
) -> bool:
    nodes = [
        item
        for item in items
        if (
            item.get("metadata", {})
            .get("labels", {})
            .get("sagemaker.amazonaws.com/cluster-name")
            == target.hyperpod_cluster_name
            and (
                node_names is None
                or str(item.get("metadata", {}).get("name") or "") in node_names
            )
        )
    ]
    aligned = [
        item
        for item in nodes
        if (
            item.get("metadata", {})
            .get("annotations", {})
            .get("gpu-fault.io/installer-state")
            == "Succeeded"
            and item.get("metadata", {})
            .get("annotations", {})
            .get("gpu-fault.io/installer-artifact-sha256")
            == artifact_sha
            and (
                config_digest is None
                or item.get("metadata", {})
                .get("annotations", {})
                .get("gpu-fault.io/installer-config-digest")
                == config_digest
            )
            and (
                not require_node_uid
                or item.get("metadata", {})
                .get("annotations", {})
                .get("gpu-fault.io/installer-node-uid")
                == item.get("metadata", {}).get("uid")
            )
            and (
                bundle_sha is None
                or item.get("metadata", {})
                .get("annotations", {})
                .get("gpu-fault.io/installer-bundle-sha256")
                == bundle_sha
            )
            and (
                template_sha is None
                or item.get("metadata", {})
                .get("annotations", {})
                .get("gpu-fault.io/installer-template-sha256")
                == template_sha
            )
        )
    ]
    return bool(nodes) and len(aligned) == len(nodes)


def executor_pin_rejection(
    metadata: dict[str, str],
    *,
    protocol_version: int,
    artifact_sha: str,
    compatibility_digest: str,
) -> str | None:
    policy = RegionalExecutorCompatibilityPolicy.from_mapping(
        {
            "GPU_FAULT_REQUIRED_REGIONAL_EXECUTOR_PROTOCOL_VERSION": (
                metadata.get(
                    "required-regional-executor-protocol-version",
                    str(protocol_version),
                )
            ),
            "GPU_FAULT_COMPATIBLE_REGIONAL_EXECUTOR_PROTOCOL_VERSIONS": (
                metadata.get(
                    "compatible-regional-executor-protocol-versions",
                    "",
                )
            ),
            "GPU_FAULT_REQUIRED_REGIONAL_EXECUTOR_ARTIFACT_SHA256": (
                metadata.get(
                    "required-regional-executor-artifact-sha256",
                    "",
                )
            ),
            "GPU_FAULT_COMPATIBLE_REGIONAL_EXECUTOR_ARTIFACT_SHA256S": (
                metadata.get(
                    "compatible-regional-executor-artifact-sha256s",
                    "",
                )
            ),
            "GPU_FAULT_REQUIRED_REGIONAL_EXECUTOR_COMPATIBILITY_DIGEST": (
                metadata.get(
                    "required-regional-executor-compatibility-digest",
                    "",
                )
            ),
            "GPU_FAULT_COMPATIBLE_REGIONAL_EXECUTOR_COMPATIBILITY_DIGESTS": (
                metadata.get(
                    "compatible-regional-executor-compatibility-digests",
                    "",
                )
            ),
        }
    )
    return policy.rejection_reason(
        protocol_version,
        artifact_sha,
        compatibility_digest,
    )


def require_executor_pin(
    release: Any,
    *,
    artifact_sha: str,
    compatibility_digest: str,
) -> None:
    metadata = release._config_map_data("gpu-fault-release-metadata")
    try:
        reason = executor_pin_rejection(
            metadata,
            protocol_version=release.config.executor_protocol_version,
            artifact_sha=artifact_sha,
            compatibility_digest=compatibility_digest,
        )
    except ValueError as exc:
        raise ReleaseError(f"invalid regional executor pin metadata: {exc}") from exc
    if reason:
        raise ReleaseError(f"executor pin preflight rejected rollout: {reason}")


def _gpu_deployment_manifests(
    release: Any,
    target: ClusterTarget,
    wheel_cm: str,
    *,
    deployment_names: frozenset[str] | None = None,
    runtime_image: str | None = None,
    runtime_profile_version: str | None = None,
    executor_wheel_filename: str | None = None,
    executor_artifact_sha: str | None = None,
    executor_compatibility_digest: str | None = None,
    require_live_pin: bool,
) -> dict[str, str]:
    artifact_sha = executor_artifact_sha or release.executor_wheel_sha
    compatibility_digest = (
        executor_compatibility_digest
        or release.config.component_digests.get("executor")
        or artifact_sha
    )
    if require_live_pin:
        require_executor_pin(
            release,
            artifact_sha=artifact_sha,
            compatibility_digest=compatibility_digest,
        )
    manifests = dict(
        render_gpu_rollout_manifests(
            release,
            target,
            wheel_cm,
            deployment_names=deployment_names,
            runtime_image=runtime_image,
            runtime_profile_version=runtime_profile_version,
            executor_wheel_filename=executor_wheel_filename,
            executor_artifact_sha=artifact_sha,
            executor_compatibility_digest=compatibility_digest,
        )
    )
    if inventory.GPU_WATCHER_DEPLOYMENT in manifests:
        manifests[inventory.GPU_WATCHER_DEPLOYMENT] = preserve_completion_watcher_state(
            release,
            target,
            manifests[inventory.GPU_WATCHER_DEPLOYMENT],
        )
    return append_workload_namespace_rbac(
        manifests,
        target,
        system_namespace=release.config.namespace,
    )


def preflight_gpu_deployments(
    release: Any,
    target: ClusterTarget,
    wheel_cm: str,
    *,
    deployment_names: frozenset[str] | None = None,
    runtime_image: str | None = None,
    runtime_profile_version: str | None = None,
    executor_wheel_filename: str | None = None,
    executor_artifact_sha: str | None = None,
    executor_compatibility_digest: str | None = None,
) -> None:
    preflight_allowed_namespace_coverage(release, target)
    manifests = _gpu_deployment_manifests(
        release,
        target,
        wheel_cm,
        deployment_names=deployment_names,
        runtime_image=runtime_image,
        runtime_profile_version=runtime_profile_version,
        executor_wheel_filename=executor_wheel_filename,
        executor_artifact_sha=executor_artifact_sha,
        executor_compatibility_digest=executor_compatibility_digest,
        require_live_pin=False,
    )
    for manifest in manifests.values():
        release.runner.run(
            release._gpu(target, "apply", "--dry-run=server", "-f", "-"),
            input_text=manifest,
        )


def apply_gpu_deployments(
    release: Any,
    target: ClusterTarget,
    wheel_cm: str,
    *,
    deployment_names: frozenset[str] | None = None,
    runtime_image: str | None = None,
    runtime_profile_version: str | None = None,
    executor_wheel_filename: str | None = None,
    executor_artifact_sha: str | None = None,
    executor_compatibility_digest: str | None = None,
) -> None:
    preflight_allowed_namespace_coverage(release, target)
    manifests = _gpu_deployment_manifests(
        release,
        target,
        wheel_cm,
        deployment_names=deployment_names,
        runtime_image=runtime_image,
        runtime_profile_version=runtime_profile_version,
        executor_wheel_filename=executor_wheel_filename,
        executor_artifact_sha=executor_artifact_sha,
        executor_compatibility_digest=executor_compatibility_digest,
        require_live_pin=True,
    )
    waves = []
    if inventory.GPU_EXECUTOR_DEPLOYMENT in manifests:
        waves.append((inventory.GPU_EXECUTOR_DEPLOYMENT,))
    secondary = tuple(
        deployment
        for deployment in (
            inventory.GPU_WATCHER_DEPLOYMENT,
            inventory.GPU_COLLECTOR_DEPLOYMENT,
        )
        if deployment in manifests
    )
    if secondary:
        waves.append(secondary)
    known = {deployment for wave in waves for deployment in wave}
    waves.extend((deployment,) for deployment in sorted(set(manifests) - known))
    for manifest in manifests.values():
        release.runner.run(
            release._gpu(target, "apply", "--dry-run=server", "-f", "-"),
            input_text=manifest,
        )
    for wave in waves:
        for deployment in wave:
            manifest = manifests[deployment]
            release.runner.run(
                release._gpu(target, "apply", "-f", "-"),
                input_text=manifest,
            )
        with ThreadPoolExecutor(max_workers=len(wave)) as executor:
            futures = {
                executor.submit(
                    wait_deployment_rollout,
                    release,
                    target,
                    deployment,
                ): deployment
                for deployment in wave
            }
            for future in as_completed(futures):
                deployment = futures[future]
                try:
                    future.result()
                except Exception as exc:
                    raise ReleaseError(
                        f"{target.cluster_id} Deployment {deployment} rollout "
                        f"failed: {exc}"
                    ) from exc
    if inventory.GPU_EXECUTOR_DEPLOYMENT in manifests:
        prune_workload_namespace_rbac(release, target)


def upgrade_gpu_target(
    release: Any,
    target: ClusterTarget,
    diff: ReleaseDiff,
    plan: ReleaseExecutionPlan | None = None,
    *,
    progress: ProgressCallback | None = None,
    candidate_preflighted: bool = False,
) -> None:
    active_plan = plan or build_execution_plan(diff)

    def run_component(
        components: tuple[ReleaseComponent, ...],
        action: Callable[[], None],
    ) -> None:
        if progress is not None:
            progress(components, "STARTED", None)
        try:
            action()
        except Exception:
            if progress is not None:
                progress(components, "FAILED", None)
            raise
        if progress is not None:
            progress(components, "COMPLETED", None)

    if active_plan.has(ReleaseComponent.ENDPOINT):
        run_component(
            (ReleaseComponent.ENDPOINT,),
            lambda: (
                release._ensure_connection_secret(target),
                release._verify_gpu_control_plane_endpoint(target),
            ),
        )
    if active_plan.has(ReleaseComponent.DCGM):
        run_component(
            (ReleaseComponent.DCGM,),
            lambda: release._apply_gpu_dcgm_exporter(
                target,
            ),
        )
    deployment_components = tuple(
        (component, deployment)
        for component, deployment in (
            (
                ReleaseComponent.EXECUTOR,
                inventory.GPU_EXECUTOR_DEPLOYMENT,
            ),
            (
                ReleaseComponent.WATCHER,
                inventory.GPU_WATCHER_DEPLOYMENT,
            ),
            (
                ReleaseComponent.COLLECTOR,
                inventory.GPU_COLLECTOR_DEPLOYMENT,
            ),
        )
        if active_plan.has(component)
    )
    if deployment_components:
        run_component(
            tuple(component for component, _deployment in deployment_components),
            lambda: release._apply_gpu_deployments(
                target,
                release.executor_wheel_cm,
                deployment_names=frozenset(
                    deployment for _component, deployment in deployment_components
                ),
            ),
        )
    if active_plan.has(ReleaseComponent.AGENT):
        components = (
            (ReleaseComponent.RECONCILER, ReleaseComponent.AGENT)
            if active_plan.has(ReleaseComponent.RECONCILER)
            else (ReleaseComponent.AGENT,)
        )
        mutation_started = False

        def record_mutation_started() -> None:
            nonlocal mutation_started
            if progress is not None:
                progress(components, "STARTED", None)
            mutation_started = True

        try:
            release._roll_node_runtime(
                target,
                phase="upgrade",
                wheel_cm=release.executor_wheel_cm,
                bundle_cm=release.bundle_cm,
                artifact_sha=release.node_wheel_sha,
                config_digest=release.config.agent_config_digest,
                mutation_started=record_mutation_started,
                candidate_preflight_completed=candidate_preflighted,
            )
        except Exception:
            if progress is not None and mutation_started:
                progress(components, "FAILED", None)
            raise
        if progress is not None:
            progress(components, "COMPLETED", None)
    elif active_plan.has(ReleaseComponent.RECONCILER):
        run_component(
            (ReleaseComponent.RECONCILER,),
            lambda: release._deploy_reconciler(
                target,
                wheel_cm=release.executor_wheel_cm,
                bundle_cm=release.bundle_cm,
                artifact_sha=release.node_wheel_sha,
                config_digest=release.config.agent_config_digest,
            ),
        )


def join_target(release: Any, cluster_id: str) -> ClusterTarget:
    target = release._target(cluster_id)
    if not release._remote_commands_are_idle():
        raise ReleaseError("remote commands are PENDING/LEASED/WAITING")
    return target

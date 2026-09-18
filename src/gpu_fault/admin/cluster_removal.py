from __future__ import annotations

import base64
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextvars import copy_context
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, cast

import yaml  # type: ignore[import-untyped,unused-ignore]

from gpu_fault.admin import cluster_removal_kubernetes as kubernetes_cleanup
from gpu_fault.admin.atomic_json import write_json_atomic
from gpu_fault.admin.aws_cleanup import ResourceCleaner
from gpu_fault.admin.bootstrap_common import (
    Arn,
    BootstrapError,
    CommandRunner,
)
from gpu_fault.admin.cluster_join_evidence import membership_runtime_snapshot
from gpu_fault.admin.cluster_removal_keys import (
    validate_key_binding,
    verify_node_key_ownership,
)
from gpu_fault.admin.cluster_removal_network import (
    cluster_network as _cluster_network,
    detach_network as _detach_network,
    disassociate_vpc_from_hosted_zone as disassociate_vpc_from_hosted_zone,
    idempotent_aws as _idempotent_aws,  # noqa: F401 - compatibility export
    json_command as _json_command,
    wait_route53_change_insync as wait_route53_change_insync,
    wait_vpc_association_absent as _wait_vpc_association_absent,  # noqa: F401 - join helper
)
from gpu_fault.admin.cluster_removal_resources import (
    association_to_detach,
    deletion_waves as _deletion_waves,  # noqa: F401 - compatibility export
    detach_registered_network,
    merge_removal_resources,
    remove_target_resources,
    target_resource as _target_resource,  # noqa: F401 - compatibility export
    target_resource_plan,
)
from gpu_fault.admin.cluster_removal_state import (
    allowed_registry_lifecycles,
    canonical_digest,
    load_removal_state,
    removal_command_ownership,
    removal_result as _removal_result,
    resolve_saved_removal,
    validate_saved_site,
)
from gpu_fault.admin.diagnostics import diagnostic_text
from gpu_fault.admin.execution import (
    OVERALL_DEPLOY_SECONDS,
    deadline_scope,
    run_command,
    run_driver,
)
from gpu_fault.admin.cluster_join_state import (
    complete_step as _complete,
    step_done as _done,
)
from gpu_fault.admin.failure_domain_map import apply_failure_domain_map
from gpu_fault.admin.membership_lock import (
    membership_operation_lock,
    reload_site_for_mutation,
)
from gpu_fault.admin.node_key_proof import read_node_key_proof
from gpu_fault.admin.process_supervisor import (
    ensure_supervision_safe,
    interruption_scope,
)
from gpu_fault.admin.resource_registry import (
    fetch_installation_resource_registry,
    find_bootstrap_state,
    load_installation_resource_snapshot,
    sync_installation_resource_snapshot,
    write_installation_resource_snapshot,
)
from gpu_fault.admin.resource_registry_dns import (
    association_identity,
    network_vpc_identity,
)
from gpu_fault.admin.site import (
    RenderedSite,
    effective_environment,
    load_site,
    materialized_release_config,
)
from gpu_fault.installation_resources import (
    InstallationResourceDeletePolicy,
    InstallationResourceSnapshot,
)
from gpu_fault.node_installer_reconciler import INSTALLER_NODE_ANNOTATIONS

CONFIRMATION = "REMOVE_GPU_CLUSTER"
#: The reconciler's own list, not a copy: a hand-kept five-entry copy left
#: ``installer-attempts`` (and five others) on the nodes of a removed cluster.
INSTALLER_ANNOTATIONS = INSTALLER_NODE_ANNOTATIONS


@dataclass(frozen=True)
class RemoveClusterRequest:
    site: RenderedSite
    cluster_id: str
    confirmation: str
    gpu_cluster_arn: str | None = None


def _target(
    site: RenderedSite,
    cluster_id: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    clusters = [dict(item) for item in site.release_config["clusters"]]
    matches = [item for item in clusters if item["cluster_id"] == cluster_id]
    if len(matches) != 1:
        raise BootstrapError(f"unknown cluster_id: {cluster_id}")
    return matches[0], [item for item in clusters if item["cluster_id"] != cluster_id]


def resolve_cluster_id(
    site: RenderedSite,
    gpu_cluster_arn: str,
    *,
    discover: Callable[[str], tuple[str, str]] | None = None,
) -> str:
    """Resolve the full EKS ARN, or discover and bind the complete HyperPod ARN."""

    arn = Arn.parse(gpu_cluster_arn)
    if arn.service not in {"eks", "sagemaker"}:
        raise BootstrapError("GPU cluster ARN must use the eks or sagemaker service")
    arn.resource_name
    cpu = Arn.parse(str(site.release_config["cpu_eks_arn"]))
    if (arn.partition, arn.account, arn.region) != (
        cpu.partition,
        cpu.account,
        site.release_config["aws_region"],
    ):
        raise BootstrapError(
            "GPU cluster ARN does not belong to the managed site scope"
        )
    clusters = [dict(item) for item in site.release_config["clusters"]]
    eks_arn = gpu_cluster_arn.strip()
    hyperpod_name = ""
    if arn.service == "sagemaker":
        if discover is None:
            raise BootstrapError(
                "a HyperPod ARN needs AWS discovery to resolve its EKS cluster"
            )
        eks_arn, hyperpod_name = discover(gpu_cluster_arn)
        discovered = Arn.parse(eks_arn)
        if (
            discovered.service != "eks"
            or not discovered.resource_name
            or (discovered.partition, discovered.account, discovered.region)
            != (arn.partition, arn.account, arn.region)
            or not hyperpod_name
        ):
            raise BootstrapError(
                "HyperPod discovery returned a conflicting cluster identity"
            )
    matches = [
        item
        for item in clusters
        if item.get("eks_cluster_arn") == eks_arn
        and (not hyperpod_name or item.get("hyperpod_cluster_name") == hyperpod_name)
    ]
    if len(matches) == 1:
        return str(matches[0]["cluster_id"])
    managed = ", ".join(
        str(item.get("eks_cluster_arn") or item.get("cluster_id")) for item in clusters
    )
    raise BootstrapError(
        f"no managed GPU cluster matches {gpu_cluster_arn}; "
        f"managed clusters: {managed or 'none'}"
    )


def resolve_removal_cluster_id(
    site: RenderedSite,
    gpu_cluster_arn: str,
    *,
    discover: Callable[[str], tuple[str, str]] | None = None,
) -> str:
    """Resolve a member or an exact identity-bound, partially committed removal."""
    try:
        return resolve_cluster_id(site, gpu_cluster_arn, discover=discover)
    except BootstrapError:
        saved = resolve_saved_removal(site, gpu_cluster_arn.strip())
        if saved is None:
            raise
        return saved


def _state(
    request: RemoveClusterRequest,
) -> tuple[Path, Path, dict[str, Any]]:
    return load_removal_state(request.site, request.cluster_id)


def _gpu_kubectl(site: RenderedSite, target: dict[str, Any]) -> list[str]:
    command = ["kubectl"]
    kubeconfig = site.release_config.get("gpu_kubeconfig") or site.environment.get(
        "KUBECONFIG"
    )
    if kubeconfig:
        command.extend(["--kubeconfig", str(kubeconfig)])
    command.extend(["--context", str(target["context"])])
    return command


def _cpu_kubectl(site: RenderedSite) -> list[str]:
    return [
        "kubectl",
        "--kubeconfig",
        str(site.release_config["cpu_kubeconfig"]),
    ]


def _parallel_cluster_networks(
    runner: CommandRunner,
    *,
    region: str,
    target: dict[str, Any],
    remaining: list[dict[str, Any]],
    cpu_eks_arn: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    inputs = [
        ("target", str(target["eks_cluster_arn"])),
        ("cpu", cpu_eks_arn),
        *(
            (f"remaining:{index}", str(item["eks_cluster_arn"]))
            for index, item in enumerate(remaining)
        ),
    ]
    values: dict[str, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=min(4, len(inputs))) as executor:
        futures = {
            executor.submit(
                copy_context().run,
                _cluster_network,
                runner,
                region=region,
                eks_arn=eks_arn,
            ): key
            for key, eks_arn in inputs
        }
        try:
            for future in as_completed(futures):
                key = futures[future]
                try:
                    values[key] = future.result()
                except Exception as exc:
                    raise BootstrapError(
                        f"network discovery failed for {key}: {diagnostic_text(str(exc))}"
                    ) from exc
        except BaseException:
            for future in futures:
                future.cancel()
            raise
    return (
        values["target"],
        [values[f"remaining:{index}"] for index in range(len(remaining))],
        values["cpu"],
    )


def _target_nodes(
    site: RenderedSite,
    target: dict[str, Any],
    *,
    identities: dict[str, str] | None = None,
) -> list[str]:
    document = _json_command(
        [
            *_gpu_kubectl(site, target),
            "get",
            "nodes",
            "-l",
            "sagemaker.amazonaws.com/cluster-name="
            + str(target["hyperpod_cluster_name"]),
            "-o",
            "json",
        ],
        description="cannot list target GPU nodes",
    )
    items = document.get("items")
    if not isinstance(items, list):
        raise BootstrapError("target GPU node response has no node list")
    nodes = []
    for item in items:
        metadata = item.get("metadata") if isinstance(item, dict) else None
        if (
            not isinstance(metadata, dict)
            or not isinstance(metadata.get("name"), str)
            or not metadata["name"]
            or not metadata.get("uid")
            or (metadata.get("labels") or {}).get(
                "sagemaker.amazonaws.com/cluster-name"
            )
            != target["hyperpod_cluster_name"]
        ):
            raise BootstrapError("target GPU node response has a conflicting identity")
        nodes.append(metadata["name"])
        if identities is not None:
            identities[metadata["name"]] = str(metadata["uid"])
    if len(set(nodes)) != len(nodes):
        raise BootstrapError("target GPU node response contains duplicate names")
    return sorted(nodes)


def _removal_identity(
    request: RemoveClusterRequest,
    target: dict[str, Any],
    runner: CommandRunner,
    target_network: dict[str, Any],
    cpu_network: dict[str, Any],
) -> tuple[dict[str, str], str | None]:
    region = str(request.site.release_config["aws_region"])
    hyperpod = runner.aws_json(
        region,
        "sagemaker",
        "describe-cluster",
        "--cluster-name",
        str(target.get("expected_hyperpod_arn") or target["hyperpod_cluster_name"]),
    )
    hyperpod_arn = str(hyperpod.get("ClusterArn") or "")
    parsed = Arn.parse(hyperpod_arn)
    eks = Arn.parse(str(target["eks_cluster_arn"]))
    if (
        parsed.service != "sagemaker"
        or not parsed.resource_name
        or (parsed.partition, parsed.account, parsed.region)
        != (eks.partition, eks.account, eks.region)
        or hyperpod.get("ClusterName") != target["hyperpod_cluster_name"]
        or ((hyperpod.get("Orchestrator") or {}).get("Eks") or {}).get("ClusterArn")
        != target["eks_cluster_arn"]
        or hyperpod.get("NodeRecovery") != "None"
    ):
        raise BootstrapError("GPU HyperPod identity or NodeRecovery policy drifted")
    if request.gpu_cluster_arn and request.gpu_cluster_arn.strip() not in {
        target["eks_cluster_arn"],
        hyperpod_arn,
    }:
        raise BootstrapError("requested GPU ARN differs from the live cluster identity")
    for kubectl, network in (
        (_cpu_kubectl(request.site), cpu_network),
        (_gpu_kubectl(request.site, target), target_network),
    ):
        endpoint = str(network.get("eks_endpoint") or "")
        result = run_command(
            [
                *kubectl,
                "config",
                "view",
                "--minify",
                "-o",
                "jsonpath={.clusters[0].cluster.server}",
            ]
        )
        if (
            not endpoint.startswith("https://")
            or result.returncode
            or result.stdout.strip() != endpoint
        ):
            raise BootstrapError(
                "Kubernetes context does not match the EKS API endpoint"
            )
    namespace = str(request.site.release_config["namespace"])
    cpu_namespace = _namespace_document(_cpu_kubectl(request.site), namespace)
    if cpu_namespace is None:
        raise BootstrapError("CPU solution namespace identity is missing")
    gpu_namespace = _namespace_document(_gpu_kubectl(request.site, target), namespace)
    return {
        "eks_arn": str(target["eks_cluster_arn"]),
        "eks_created_at": str(target_network["eks_created_at"]),
        "hyperpod_arn": hyperpod_arn,
        "cpu_eks_arn": str(request.site.release_config["cpu_eks_arn"]),
        "cpu_eks_created_at": str(cpu_network["eks_created_at"]),
        "cpu_namespace_uid": str(cpu_namespace["metadata"]["uid"]),
    }, str(gpu_namespace["metadata"]["uid"]) if gpu_namespace else None


def _validate_target_registry(
    request: RemoveClusterRequest,
    target: dict[str, Any],
    snapshot: InstallationResourceSnapshot,
    identity: dict[str, str],
) -> None:
    snapshot.require_source_binding()
    if snapshot.site_id != request.site.registry_site_id:
        raise BootstrapError("installation registry belongs to another site")
    records = {resource.resource_key: resource for resource in snapshot.resources}
    eks = records.get(f"cluster/{request.cluster_id}/eks")
    hyperpod = records.get(f"cluster/{request.cluster_id}/hyperpod")
    role = records.get(f"aws/iam/executor/{request.cluster_id}/role")
    role_arn = str(target["executor_irsa_role_arn"])
    if (
        eks is None
        or (eks.resource_arn or eks.resource_id) != target["eks_cluster_arn"]
        or eks.delete_policy is not InstallationResourceDeletePolicy.PRESERVE
        or hyperpod is None
        or hyperpod.resource_id != target["hyperpod_cluster_name"]
        or hyperpod.delete_policy is not InstallationResourceDeletePolicy.PRESERVE
        or role is None
        or role.resource_type != "iam_role"
        or (
            role.resource_arn != role_arn
            if role.resource_arn
            else role.resource_id != role_arn.rsplit("/", 1)[-1]
        )
    ):
        raise BootstrapError(
            "installation registry target cluster identity is incomplete"
        )
    expected_hyperpod = hyperpod.resource_arn
    if not expected_hyperpod:
        bootstrap = find_bootstrap_state(request.site) or {}
        expected_hyperpod = (
            (bootstrap.get("resources") or {}).get("hyperpod_by_eks") or {}
        ).get(target["eks_cluster_arn"])
    if not expected_hyperpod or expected_hyperpod != identity["hyperpod_arn"]:
        raise BootstrapError(
            "installation registry GPU HyperPod ARN is missing or drifted"
        )


def _validate_token_ownership(
    site: RenderedSite,
    target: dict[str, Any],
    remaining: list[dict[str, Any]],
) -> None:
    gpu = Arn.parse(str(target["eks_cluster_arn"]))
    cpu = Arn.parse(str(site.release_config["cpu_eks_arn"]))
    if (
        gpu.service != "eks"
        or (gpu.partition, gpu.region, gpu.account)
        != (cpu.partition, cpu.region, cpu.account)
        or target["eks_cluster_arn"] == site.release_config["cpu_eks_arn"]
        or any(
            item["eks_cluster_arn"] == target["eks_cluster_arn"]
            or item["context"] == target["context"]
            for item in remaining
        )
    ):
        raise BootstrapError(
            "target GPU cluster scope overlaps or conflicts with a preserved cluster"
        )
    token_path = Path(str(target["token_file"])).resolve()
    protected = {
        site.source.resolve(),
        Path(str(site.release_config["cpu_kubeconfig"])).resolve(),
    }
    for cluster in [target, *remaining]:
        for key in ("ca_file", "fleet_master_file"):
            if cluster.get(key):
                protected.add(Path(str(cluster[key])).resolve())
    for cluster in remaining:
        protected.add(Path(str(cluster["token_file"])).resolve())
    gpu_kubeconfig = site.release_config.get("gpu_kubeconfig") or site.environment.get(
        "KUBECONFIG"
    )
    gpu_kubeconfig_paths = (
        [str(gpu_kubeconfig)]
        if gpu_kubeconfig
        else [
            path for path in os.environ.get("KUBECONFIG", "").split(os.pathsep) if path
        ]
    )
    if not gpu_kubeconfig_paths:
        gpu_kubeconfig_paths = [str(Path.home() / ".kube/config")]
    protected.update(Path(path).expanduser().resolve() for path in gpu_kubeconfig_paths)
    if token_path in protected:
        raise BootstrapError(
            "target cluster token file is shared with a preserved resource"
        )


def _export_registry(
    site: RenderedSite,
    state_dir: Path,
) -> InstallationResourceSnapshot:
    path = state_dir / "installation-resources-before.json"
    if path.exists():
        return load_installation_resource_snapshot(path)
    return fetch_installation_resource_registry(site, output=path)


def _run_kubernetes_cleanup(
    request: RemoveClusterRequest,
    target: dict[str, Any],
    state_dir: Path,
) -> Path:
    attempts = sorted(state_dir.glob("kubernetes-cleanup-*.json"))
    for path in reversed(attempts):
        if _cleanup_completed(request.site, target, path):
            return path
    path = attempts[-1] if attempts else state_dir / "kubernetes-cleanup-001.json"
    with materialized_release_config(request.site) as config:
        completed = run_driver(
            [
                str(
                    request.site.repository_root
                    / "deploy/control-plane/regional/prepare-clean-redeploy.sh"
                ),
                "--config",
                str(config),
                "--scope",
                "gpu",
                "--cluster-id",
                request.cluster_id,
                "--mode",
                "clean",
                "--node-mode",
                "uninstall",
                "--state-file",
                str(path),
                "--execute",
            ],
            cwd=request.site.repository_root,
            env=effective_environment(request.site),
            check=False,
        )
    if completed.returncode:
        if path.is_file():
            raise BootstrapError(
                f"GPU Kubernetes cleanup failed; evidence retained in {path}"
            )
        # The script refused before it wrote its state file (an unregistered
        # live resource, an invalid config); the reason is in the command log.
        raise BootstrapError(
            f"GPU Kubernetes cleanup refused before writing {path.name}; the "
            "refusal is in the command log above"
        )
    if not _cleanup_completed(request.site, target, path):
        raise BootstrapError("GPU Kubernetes cleanup has no completed, bound evidence")
    return path


def _cleanup_completed(
    site: RenderedSite,
    target: dict[str, Any],
    path: Path,
) -> bool:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise BootstrapError("cannot read GPU cleanup evidence") from exc
    if not isinstance(document, dict):
        raise BootstrapError("invalid GPU cleanup evidence")
    digest = canonical_digest(
        {key: value for key, value in document.items() if key != "content_sha256"}
    )
    expected_targets = {
        "namespace": site.release_config["namespace"],
        "cpu_kubeconfig": site.release_config["cpu_kubeconfig"],
        "clusters": [
            {"cluster_id": item["cluster_id"], "context": item["context"]}
            for item in site.release_config["clusters"]
        ],
    }
    expected_config = hashlib.sha256(
        json.dumps(site.release_config, indent=2, sort_keys=True).encode()
    ).hexdigest()
    if (
        document.get("schema_version") != 2
        or document.get("content_sha256") != digest
        or document.get("config_sha256") != expected_config
        or document.get("targets") != expected_targets
        or document.get("scope") != "gpu"
        or document.get("mode") != "clean"
        or document.get("node_mode") != "uninstall"
    ):
        raise BootstrapError("GPU cleanup evidence conflicts with the removal attempt")
    if (
        document.get("phase") != "CLEANUP_COMPLETED"
        or document.get("status") != "COMPLETED"
    ):
        return False
    captured = document.get("original_resources")
    if (
        not isinstance(captured, list)
        or not captured
        or any(
            not isinstance(item, dict)
            or item.get("scope") != f"gpu:{target['cluster_id']}"
            or item.get("context") != target["context"]
            for item in captured
        )
    ):
        raise BootstrapError("GPU cleanup evidence does not prove the selected target")
    return True


def _clear_installer_annotations(
    site: RenderedSite,
    target: dict[str, Any],
    nodes: list[str],
) -> None:
    if not nodes:
        return
    node_uids = target.get("expected_node_uids")
    if not isinstance(node_uids, dict) or set(node_uids) != set(nodes):
        raise BootstrapError("node annotation cleanup has no complete UID proof")
    kubernetes_cleanup.clear_installer_annotations(
        run_command,
        _gpu_kubectl(site, target),
        hyperpod_name=str(target["hyperpod_cluster_name"]),
        node_uids=node_uids,
        annotations=INSTALLER_ANNOTATIONS,
    )


def _request_target_namespace_deletion(
    site: RenderedSite,
    target: dict[str, Any],
) -> None:
    kubernetes_cleanup.request_namespace_deletion(
        run_command,
        _gpu_kubectl(site, target),
        str(site.release_config["namespace"]),
        target.get("expected_namespace_uid"),
    )


def _namespace_document(
    kubectl: list[str],
    namespace: str,
) -> dict[str, Any] | None:
    return kubernetes_cleanup.namespace_document(run_command, kubectl, namespace)


def _wait_target_namespace_absent(
    site: RenderedSite,
    target: dict[str, Any],
    *,
    timeout_seconds: float = 600,
    interval_seconds: float = 5,
) -> None:
    kubernetes_cleanup.wait_namespace_absent(
        run_command,
        _gpu_kubectl(site, target),
        str(site.release_config["namespace"]),
        target.get("expected_namespace_uid"),
        timeout_seconds=timeout_seconds,
        interval_seconds=interval_seconds,
    )


def _delete_target_namespace(
    site: RenderedSite,
    target: dict[str, Any],
) -> None:
    _request_target_namespace_deletion(site, target)
    _wait_target_namespace_absent(site, target)


def _remove_node_action_keys(
    site: RenderedSite,
    nodes: list[str],
    *,
    expected_key_sha256: dict[str, str] | None = None,
    expected_secret_uid: str | None = None,
) -> int:
    if not nodes:
        return 0
    runner = CommandRunner()
    proof = read_node_key_proof(
        runner, _cpu_kubectl(site), str(site.release_config["namespace"])
    )
    if proof is None:
        raise BootstrapError("CPU node-key Secret identity is unavailable")
    if expected_secret_uid is not None and proof.uid != expected_secret_uid:
        raise BootstrapError("CPU node-key Secret incarnation changed before deletion")
    present = set(proof.digests)
    removable = sorted(set(nodes) & present)
    if expected_key_sha256 is not None:
        if set(expected_key_sha256) != set(nodes) or any(
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
            for value in expected_key_sha256.values()
        ):
            raise BootstrapError("node-key rollback has an incomplete digest proof")
        if any(proof.digests[node] != expected_key_sha256[node] for node in removable):
            raise BootstrapError("CPU node-key ownership changed before deletion")
    if not removable:
        return 0
    patch = [
        {"op": "test", "path": "/metadata/uid", "value": proof.uid},
        {
            "op": "test",
            "path": "/metadata/resourceVersion",
            "value": proof.version,
        },
        *(
            {
                "op": "remove",
                "path": "/data/" + name.replace("~", "~0").replace("/", "~1"),
            }
            for name in removable
        ),
    ]
    result = run_command(
        [
            *_cpu_kubectl(site),
            "-n",
            str(site.release_config["namespace"]),
            "patch",
            "secret",
            "gpu-fault-node-action-keys",
            "--type=json",
            "-p",
            json.dumps(patch, separators=(",", ":")),
        ],
    )
    if result.returncode:
        raise BootstrapError(
            f"failed to remove target node-action keys (exit {result.returncode})"
        )
    current = read_node_key_proof(
        runner, _cpu_kubectl(site), str(site.release_config["namespace"])
    )
    if (
        current is None
        or current.uid != proof.uid
        or set(nodes).intersection(current.digests)
    ):
        raise BootstrapError("CPU node-key deletion was not confirmed")
    return len(removable)


def _run_control_plane_unregister(
    request: RemoveClusterRequest,
) -> None:
    with materialized_release_config(request.site) as config:
        completed = run_driver(
            [
                str(
                    request.site.repository_root
                    / "deploy/control-plane/regional/rollout-regional-release.sh"
                ),
                "remove-cluster",
                "--cluster-id",
                request.cluster_id,
                "--config",
                str(config),
            ],
            cwd=request.site.repository_root,
            env=effective_environment(request.site),
            check=False,
        )
    if completed.returncode:
        raise BootstrapError("control-plane cluster unregister failed")


def _run_control_plane_drain(
    request: RemoveClusterRequest,
) -> None:
    with materialized_release_config(request.site) as config:
        completed = run_driver(
            [
                str(
                    request.site.repository_root
                    / "deploy/control-plane/regional/rollout-regional-release.sh"
                ),
                "drain-cluster",
                "--cluster-id",
                request.cluster_id,
                "--config",
                str(config),
            ],
            cwd=request.site.repository_root,
            env=effective_environment(request.site),
            check=False,
        )
    if completed.returncode:
        raise BootstrapError("control-plane cluster drain failed")


def _remove_target_aws_resources(
    request: RemoveClusterRequest,
    snapshot: InstallationResourceSnapshot,
    *,
    detached_vpc_id: str | None = None,
) -> tuple[InstallationResourceSnapshot, dict[str, Any]]:
    return remove_target_resources(
        request.site,
        request.cluster_id,
        snapshot,
        cleaner_factory=ResourceCleaner,
        detached_vpc_id=detached_vpc_id,
    )


def _write_site_without_cluster(
    request: RemoveClusterRequest,
    state_dir: Path,
) -> None:
    source = request.site.source
    state_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    backup = state_dir / "site.before.yaml"
    if not backup.exists():
        backup.write_bytes(source.read_bytes())
        backup.chmod(0o600)
    document = yaml.safe_load(source.read_text(encoding="utf-8"))
    clusters = document["spec"]["clusters"]
    document["spec"]["clusters"] = [
        item for item in clusters if item.get("clusterId") != request.cluster_id
    ]
    temporary = source.with_suffix(source.suffix + ".tmp")
    temporary.write_text(
        yaml.safe_dump(document, sort_keys=False),
        encoding="utf-8",
    )
    temporary.chmod(0o600)
    temporary.replace(source)


def _update_bootstrap_state(
    request: RemoveClusterRequest,
    *,
    target_network: dict[str, Any],
    remaining_networks: list[dict[str, Any]],
    cpu_vpc_id: str,
    detached_vpc_id: str | None = None,
) -> None:
    path = request.site.source.parent / "bootstrap-state.json"
    if not path.exists():
        return
    value = cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))
    if value.get("site_id") != request.site.release_config["site_name"]:
        raise BootstrapError("bootstrap state belongs to another site")
    completed = set(value.get("completed_tasks") or [])
    resources = value.get("resources") or {}
    for name in (
        f"executor_role:{request.cluster_id}",
        f"adot_writer_role:{request.cluster_id}",
        f"node_keys:{request.cluster_id}",
    ):
        completed.discard(name)
        resources.pop(name, None)
    region = str(request.site.release_config["aws_region"])
    target_vpc = network_vpc_identity(target_network, region=region)
    if target_vpc[0] != region:
        raise BootstrapError("target network is outside the site Region")
    remaining_vpcs = {
        network_vpc_identity(item, region=region) for item in remaining_networks
    } | {(region, cpu_vpc_id)}
    if detached_vpc_id is not None and (
        detached_vpc_id != target_vpc[1] or target_vpc in remaining_vpcs
    ):
        raise BootstrapError("detached Route53 association conflicts with shared VPC")
    pki = resources.get("pki") or {}
    pki["vpc_associations"] = [
        item
        for item in pki.get("vpc_associations", [])
        if detached_vpc_id is None or association_identity(item) != target_vpc
    ]
    nlb = resources.get("nlb_network") or {}
    remaining_eips = {
        eip for item in remaining_networks for eip in item.get("nat_eips", [])
    }
    if target_network["vpc_id"] == cpu_vpc_id:
        remaining_eips.update(target_network.get("nat_eips", []))
    nlb["gpu_nat_eips"] = sorted(remaining_eips)
    resources["pki"] = pki
    resources["nlb_network"] = nlb
    value["completed_tasks"] = sorted(completed)
    value["resources"] = resources
    value.setdefault("removed_clusters", {})[request.cluster_id] = {
        "removed_at": datetime.now(timezone.utc).isoformat(),
        "vpc_id": target_network["vpc_id"],
        "vpc_region": region,
    }
    write_json_atomic(path, value)


def _delete_cluster_token(target: dict[str, Any]) -> None:
    path = Path(str(target.get("token_file") or ""))
    if path.is_file():
        path.unlink()


def _verify_target_namespace_absent(
    request: RemoveClusterRequest,
    target: dict[str, Any],
) -> None:
    namespace = str(request.site.release_config["namespace"])
    if _namespace_document(_gpu_kubectl(request.site, target), namespace) is not None:
        raise BootstrapError("target GPU solution namespace still exists")


def _verify_control_registry_absent(
    request: RemoveClusterRequest,
) -> None:
    namespace = str(request.site.release_config["namespace"])
    registry = _json_command(
        [
            *_cpu_kubectl(request.site),
            "-n",
            namespace,
            "get",
            "secret",
            "gpu-fault-regional-clusters",
            "-o",
            "json",
        ],
        description="cannot verify CPU regional registry",
    )
    metadata = registry.get("metadata") or {}
    encoded = (registry.get("data") or {}).get("clusters.json")
    if (
        registry.get("kind") != "Secret"
        or metadata.get("name") != "gpu-fault-regional-clusters"
        or metadata.get("namespace") != namespace
        or not metadata.get("uid")
        or not isinstance(encoded, str)
        or not encoded
    ):
        raise BootstrapError(
            "CPU regional registry identity or cluster data is missing"
        )
    try:
        registrations = json.loads(base64.b64decode(encoded, validate=True))
    except (TypeError, ValueError) as exc:
        raise BootstrapError("CPU regional registry cluster data is invalid") from exc
    if not isinstance(registrations, list) or any(
        not isinstance(item, dict) or not isinstance(item.get("cluster_id"), str)
        for item in registrations
    ):
        raise BootstrapError("CPU regional registry cluster data is invalid")
    if any(item["cluster_id"] == request.cluster_id for item in registrations):
        raise BootstrapError("target cluster remains in CPU regional registry")
    runtime = _runtime_membership(request.site)
    expected = {
        str(item["cluster_id"]): "ACTIVE"
        for item in request.site.release_config["clusters"]
        if item["cluster_id"] != request.cluster_id
    }
    if runtime["registry_cluster_states"] != expected:
        raise BootstrapError("durable regional registry removal has not converged")


def _runtime_membership(site: RenderedSite) -> dict[str, Any]:
    runtime = membership_runtime_snapshot(site)
    states = runtime.get("registry_cluster_states")
    generation = runtime.get("registry_generation")
    if (
        not isinstance(states, dict)
        or any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in states.items()
        )
        or type(generation) is not int
        or generation < 1
        or any(
            not isinstance(runtime.get(key), str)
            or len(runtime[key]) != 64
            or any(character not in "0123456789abcdef" for character in runtime[key])
            for key in ("registry_content_sha256", "live_release_identity_sha256")
        )
    ):
        raise BootstrapError("durable regional registry identity is incomplete")
    return runtime


def _verify_preserved_gpu_eks(
    request: RemoveClusterRequest,
    target: dict[str, Any],
) -> None:
    region = str(request.site.release_config["aws_region"])
    eks_name = Arn.parse(str(target["eks_cluster_arn"])).resource_name
    document = _json_command(
        [
            "aws",
            "eks",
            "describe-cluster",
            "--region",
            region,
            "--name",
            eks_name,
            "--output",
            "json",
        ],
        description="preserved GPU EKS cluster is missing",
    )
    cluster = document.get("cluster") or {}
    if (
        cluster.get("arn") != target["eks_cluster_arn"]
        or cluster.get("status") != "ACTIVE"
        or str(cluster.get("createdAt") or "") != target.get("expected_eks_created_at")
    ):
        raise BootstrapError("preserved GPU EKS cluster identity drifted")


def _verify_preserved_hyperpod(
    request: RemoveClusterRequest,
    target: dict[str, Any],
) -> None:
    region = str(request.site.release_config["aws_region"])
    document = _json_command(
        [
            "aws",
            "sagemaker",
            "describe-cluster",
            "--region",
            region,
            "--cluster-name",
            str(target.get("expected_hyperpod_arn") or target["hyperpod_cluster_name"]),
            "--output",
            "json",
        ],
        description="preserved GPU HyperPod cluster is missing",
    )
    if (
        not target.get("expected_hyperpod_arn")
        or document.get("ClusterArn") != target["expected_hyperpod_arn"]
        or document.get("ClusterName") != target["hyperpod_cluster_name"]
        or ((document.get("Orchestrator") or {}).get("Eks") or {}).get("ClusterArn")
        != target["eks_cluster_arn"]
        or document.get("NodeRecovery") != "None"
    ):
        raise BootstrapError(
            "preserved GPU HyperPod identity or NodeRecovery policy drifted"
        )


def _verify_target_absent(
    request: RemoveClusterRequest,
    target: dict[str, Any],
) -> None:
    _verify_target_namespace_absent(request, target)
    _verify_control_registry_absent(request)
    _verify_preserved_gpu_eks(request, target)
    _verify_preserved_hyperpod(request, target)


def _verify_remaining_site(site: RenderedSite) -> None:
    rollout = (
        site.repository_root
        / "deploy/control-plane/regional/rollout-regional-release.sh"
    )
    with materialized_release_config(site) as config:
        result = run_driver(
            [str(rollout), "verify", "--config", str(config)],
            cwd=site.repository_root,
            env=effective_environment(site),
            check=False,
        )
    if result.returncode:
        raise BootstrapError("remaining CPU control plane verification failed")


def _verify_removal_parallel(
    request: RemoveClusterRequest,
    target: dict[str, Any],
    updated_site: RenderedSite,
) -> None:
    checks: list[tuple[str, Callable[[], None]]] = [
        (
            "target namespace",
            lambda: _verify_target_namespace_absent(request, target),
        ),
        ("control registry", lambda: _verify_control_registry_absent(request)),
        ("GPU EKS", lambda: _verify_preserved_gpu_eks(request, target)),
        ("HyperPod", lambda: _verify_preserved_hyperpod(request, target)),
        ("remaining site", lambda: _verify_remaining_site(updated_site)),
    ]
    with ThreadPoolExecutor(max_workers=len(checks)) as executor:
        futures: dict[Any, str] = {
            executor.submit(copy_context().run, function): description
            for description, function in checks
        }
        failures: list[str] = []
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as exc:
                failures.append(f"{futures[future]}: {diagnostic_text(str(exc))}")
    if failures:
        raise BootstrapError(
            "remove-cluster final verification failed: " + "; ".join(sorted(failures))
        )


def _sync_release_state(site: RenderedSite) -> None:
    rollout = (
        site.repository_root
        / "deploy/control-plane/regional/rollout-regional-release.sh"
    )
    with materialized_release_config(site) as config:
        result = run_driver(
            [str(rollout), "sync-state", "--config", str(config)],
            cwd=site.repository_root,
            env=effective_environment(site),
            check=False,
        )
    if result.returncode:
        raise BootstrapError("regional release state synchronization failed")


def refresh_failure_domain_map(site: RenderedSite) -> None:
    """Re-render the control-worker's failure-domain map for the remaining clusters."""

    apply_failure_domain_map(site)


def _discover_removal(
    request: RemoveClusterRequest,
    target: dict[str, Any],
    remaining: list[dict[str, Any]],
    state_dir: Path,
    state: dict[str, Any],
    runner: CommandRunner,
) -> tuple[InstallationResourceSnapshot, dict[str, Any]]:
    saved = state["evidence"].get("DISCOVERED")
    snapshot_path = state_dir / "installation-resources-before.json"
    if saved:
        if not isinstance(saved, dict) or saved.get("registry_snapshot") != str(
            snapshot_path
        ):
            raise BootstrapError("remove-cluster registry snapshot identity is invalid")
        snapshot = load_installation_resource_snapshot(snapshot_path)
        if snapshot.source_sha256 != saved.get("registry_digest"):
            raise BootstrapError("remove-cluster registry snapshot provenance changed")
        target = {
            **target,
            "expected_hyperpod_arn": saved["provider_identity"]["hyperpod_arn"],
        }
    else:
        snapshot = _export_registry(request.site, state_dir)
    target_network, remaining_networks, cpu_network = _parallel_cluster_networks(
        runner,
        region=str(request.site.release_config["aws_region"]),
        target=target,
        remaining=remaining,
        cpu_eks_arn=str(request.site.release_config["cpu_eks_arn"]),
    )
    identity, namespace_uid = _removal_identity(
        request, target, runner, target_network, cpu_network
    )
    _validate_target_registry(request, target, snapshot, identity)
    runtime = _runtime_membership(request.site)
    cluster_states = dict(runtime["registry_cluster_states"])
    lifecycle = cluster_states.pop(request.cluster_id, None)
    expected_remaining = {str(item["cluster_id"]): "ACTIVE" for item in remaining}
    if cluster_states != expected_remaining:
        raise BootstrapError(
            "another cluster membership lifecycle is incomplete or drifted"
        )
    if lifecycle not in allowed_registry_lifecycles(state):
        raise BootstrapError("target durable registry lifecycle conflicts with removal")
    association = association_to_detach(
        request.site,
        snapshot,
        target_network=target_network,
        remaining_networks=remaining_networks,
        cpu_vpc_id=str(cpu_network["vpc_id"]),
    )
    planned_resources, _ = target_resource_plan(
        request.site,
        request.cluster_id,
        snapshot,
        detached_vpc_id=association.attributes["vpc_id"] if association else None,
    )
    ResourceCleaner(request.site).validate_supported(planned_resources)
    node_uids: dict[str, str] = {}
    nodes = _target_nodes(request.site, target, identities=node_uids)
    token = Path(str(target["token_file"]))
    try:
        token_sha256 = hashlib.sha256(token.read_bytes()).hexdigest()
    except FileNotFoundError:
        if not _done(state, "SITE_COMMIT_STARTED"):
            raise BootstrapError(
                "target cluster token is missing before site commit"
            ) from None
        token_sha256 = None
    evidence = {
        "target": state["target"],
        "nodes": nodes,
        "node_uids": node_uids,
        "target_network": target_network,
        "remaining_networks": remaining_networks,
        "cpu_network": cpu_network,
        "cpu_vpc_id": cpu_network["vpc_id"],
        "provider_identity": identity,
        "live_release_identity_sha256": runtime["live_release_identity_sha256"],
        "remaining_registry_states": cluster_states,
        "namespace_uid": namespace_uid,
        "token_sha256": token_sha256,
        "registry_snapshot": str(snapshot_path),
        "registry_digest": snapshot.source_sha256,
    }
    if saved:
        for key in (
            "target",
            "nodes",
            "node_uids",
            "target_network",
            "remaining_networks",
            "cpu_network",
            "provider_identity",
            "live_release_identity_sha256",
            "remaining_registry_states",
        ):
            if evidence[key] != saved.get(key):
                raise BootstrapError(
                    f"remove-cluster discovery identity drifted: {key}"
                )
        if namespace_uid is not None and (
            namespace_uid != saved.get("namespace_uid")
            or _done(state, "KUBERNETES_REMOVED")
        ):
            raise BootstrapError("target GPU namespace was recreated during removal")
        if token_sha256 is not None and token_sha256 != saved.get("token_sha256"):
            raise BootstrapError("target cluster credential incarnation changed")
    if namespace_uid is None and not _done(state, "KUBERNETES_QUIESCED"):
        raise BootstrapError("target GPU namespace is absent without a cleanup barrier")
    if nodes and saved:
        validate_key_binding(saved.get("node_key_ownership"), nodes)
    evidence["node_key_ownership"] = verify_node_key_ownership(
        request.site,
        request.cluster_id,
        nodes,
        runner=runner,
        gpu_kubectl=_gpu_kubectl(request.site, target),
        saved=saved.get("node_key_ownership") if saved else None,
        namespace_present=namespace_uid is not None,
        allow_missing=_done(state, "KUBERNETES_REMOVED"),
        removed=_done(state, "CONTROL_REGISTRY_REMOVED"),
    )
    return snapshot, saved or evidence


def remove_cluster(
    request: RemoveClusterRequest,
    *,
    runner: CommandRunner | None = None,
) -> dict[str, Any]:
    with membership_operation_lock(request.site):
        with (
            interruption_scope(wait_all=True),
            deadline_scope("remove-cluster", OVERALL_DEPLOY_SECONDS),
        ):
            current = reload_site_for_mutation(request.site)
            before = [
                item
                for item in request.site.release_config["clusters"]
                if item["cluster_id"] == request.cluster_id
            ]
            after = [
                item
                for item in current.release_config["clusters"]
                if item["cluster_id"] == request.cluster_id
            ]
            if before and after and before != after:
                raise BootstrapError(
                    "removal target identity changed before mutation lock"
                )
            return _remove_cluster_locked(replace(request, site=current), runner=runner)


def _remove_cluster_locked(
    request: RemoveClusterRequest,
    *,
    runner: CommandRunner | None = None,
) -> dict[str, Any]:
    if request.confirmation != CONFIRMATION:
        raise BootstrapError(f"remove-cluster requires --confirm {CONFIRMATION}")
    ensure_supervision_safe()
    state_dir, state_path, state = _state(request)
    with removal_command_ownership(state_path, state):
        return _execute_removal(request, state_dir, state_path, state, runner=runner)


def _quiesce_and_remove_gpu(
    request: RemoveClusterRequest,
    target: dict[str, Any],
    nodes: list[str],
    state_dir: Path,
    state_path: Path,
    state: dict[str, Any],
) -> None:
    if not _done(state, "CONTROL_REGISTRY_DRAINING"):
        _run_control_plane_drain(request)
        _complete(state_path, state, "CONTROL_REGISTRY_DRAINING")
    if not _done(state, "KUBERNETES_QUIESCED"):
        cleanup_path = _run_kubernetes_cleanup(request, target, state_dir)
        _clear_installer_annotations(request.site, target, nodes)
        _complete(
            state_path,
            state,
            "KUBERNETES_QUIESCED",
            {
                "cleanup_state": str(cleanup_path),
                "nodes": nodes,
            },
        )
    if not _done(state, "KUBERNETES_REMOVED"):
        _request_target_namespace_deletion(request.site, target)
        _wait_target_namespace_absent(request.site, target)
        # Namespace absence fences a terminating Reconciler's final metadata write.
        _clear_installer_annotations(request.site, target, nodes)
        _complete(state_path, state, "KUBERNETES_REMOVED", {"nodes": nodes})


def _execute_removal(
    request: RemoveClusterRequest,
    state_dir: Path,
    state_path: Path,
    state: dict[str, Any],
    *,
    runner: CommandRunner | None,
) -> dict[str, Any]:
    target = dict(state["target"])
    remaining = [
        dict(item)
        for item in request.site.release_config["clusters"]
        if item["cluster_id"] != request.cluster_id
    ]
    if request.gpu_cluster_arn:
        supplied = request.gpu_cluster_arn.strip()
        arn = Arn.parse(supplied)
        known = (state["evidence"].get("DISCOVERED") or {}).get(
            "provider_identity"
        ) or {}
        if (
            arn.service not in {"eks", "sagemaker"}
            or not arn.resource_name
            or arn.service == "eks"
            and supplied != target["eks_cluster_arn"]
            or known
            and supplied not in {known.get("eks_arn"), known.get("hyperpod_arn")}
        ):
            raise BootstrapError(
                "requested GPU ARN conflicts with the removal identity"
            )
    _validate_token_ownership(request.site, target, remaining)
    active_runner = runner or CommandRunner()
    snapshot, discovery = _discover_removal(
        request, target, remaining, state_dir, state, active_runner
    )
    if not _done(state, "DISCOVERED"):
        _complete(state_path, state, "DISCOVERED", discovery)
    nodes = list(discovery["nodes"])
    target_network = dict(discovery["target_network"])
    remaining_networks = [dict(item) for item in discovery["remaining_networks"]]
    cpu_vpc_id = str(discovery["cpu_vpc_id"])
    target["expected_hyperpod_arn"] = discovery["provider_identity"]["hyperpod_arn"]
    target["expected_eks_created_at"] = discovery["provider_identity"]["eks_created_at"]
    target["expected_namespace_uid"] = discovery["namespace_uid"]
    target["expected_node_uids"] = discovery["node_uids"]
    if state["phase"] == "COMPLETED":
        return _removal_result(
            request.site, request.cluster_id, state_path, state, remaining
        )
    _quiesce_and_remove_gpu(request, target, nodes, state_dir, state_path, state)

    if not _done(state, "CONTROL_REGISTRY_REMOVED"):
        current_node_uids: dict[str, str] = {}
        _target_nodes(request.site, target, identities=current_node_uids)
        if current_node_uids != discovery["node_uids"]:
            raise BootstrapError("node-key target node incarnation changed")
        verify_node_key_ownership(
            request.site,
            request.cluster_id,
            nodes,
            runner=active_runner,
            gpu_kubectl=_gpu_kubectl(request.site, target),
            saved=discovery["node_key_ownership"],
            namespace_present=False,
            allow_missing=True,
        )

    def remove_control_registry() -> dict[str, Any]:
        _run_control_plane_unregister(request)
        proof = discovery["node_key_ownership"]
        removed_keys = _remove_node_action_keys(
            request.site,
            nodes,
            expected_key_sha256=proof.get("expected_key_sha256"),
            expected_secret_uid=proof.get("cpu_secret_uid"),
        )
        return {"node_action_keys_removed": removed_keys}

    def detach_aws() -> dict[str, Any]:
        network = detach_registered_network(
            request,
            snapshot,
            target_network=target_network,
            remaining_networks=remaining_networks,
            cpu_vpc_id=cpu_vpc_id,
            detach=_detach_network,
        )
        updated, resources = _remove_target_aws_resources(
            request, snapshot, detached_vpc_id=network["detached_vpc_id"]
        )
        write_installation_resource_snapshot(
            request.site,
            updated,
            path=state_dir / "installation-resources-after.json",
        )
        return {**network, **resources}

    cleanup_tasks: dict[str, Any] = {}
    with ThreadPoolExecutor(max_workers=2) as executor:
        if not _done(state, "CONTROL_REGISTRY_REMOVED"):
            cleanup_tasks["CONTROL_REGISTRY_REMOVED"] = executor.submit(
                copy_context().run, remove_control_registry
            )
        if not _done(state, "AWS_DETACHED"):
            cleanup_tasks["AWS_DETACHED"] = executor.submit(
                copy_context().run, detach_aws
            )
        cleanup_failures = []
        for step, future in cleanup_tasks.items():
            try:
                evidence = future.result()
            except Exception as exc:
                cleanup_failures.append(f"{step}: {diagnostic_text(str(exc))}")
                continue
            _complete(state_path, state, step, evidence)
    if cleanup_failures:
        raise BootstrapError("; ".join(sorted(cleanup_failures)))

    if not _done(state, "AURORA_UPDATED"):
        updated = load_installation_resource_snapshot(
            state_dir / "installation-resources-after.json"
        )
        updated = merge_removal_resources(
            snapshot, updated, fetch_installation_resource_registry(request.site)
        )
        sync_installation_resource_snapshot(request.site, updated)
        _complete(
            state_path,
            state,
            "AURORA_UPDATED",
            {"snapshot_digest": updated.source_sha256},
        )

    if not _done(state, "SITE_UPDATED"):
        validate_saved_site(request.site, request.cluster_id, state)
        if not _done(state, "SITE_COMMIT_STARTED"):
            _complete(state_path, state, "SITE_COMMIT_STARTED")
        _write_site_without_cluster(request, state_dir)
        _update_bootstrap_state(
            request,
            target_network=target_network,
            remaining_networks=remaining_networks,
            cpu_vpc_id=cpu_vpc_id,
            detached_vpc_id=state["evidence"]["AWS_DETACHED"].get("detached_vpc_id"),
        )
        _delete_cluster_token(target)
        _complete(
            state_path,
            state,
            "SITE_UPDATED",
            {"remaining_cluster_ids": [item["cluster_id"] for item in remaining]},
        )

    updated_site = load_site(
        request.site.source,
        repository_root=request.site.repository_root,
    )
    if not _done(state, "RELEASE_STATE_UPDATED"):
        # Membership is final: drop the removed cluster from the failure-domain
        # map the control-worker mounts before the release state moves on.
        refresh_failure_domain_map(updated_site)
        _sync_release_state(updated_site)
        _complete(
            state_path,
            state,
            "RELEASE_STATE_UPDATED",
            {"cluster_ids": [item["cluster_id"] for item in remaining]},
        )
    if not _done(state, "VERIFIED"):
        _verify_removal_parallel(request, target, updated_site)
        _complete(
            state_path,
            state,
            "VERIFIED",
            {
                "cpu_control_plane_preserved": True,
                "gpu_cluster_preserved": True,
            },
        )

    state["phase"] = "COMPLETED"
    state["updated_at"] = datetime.now(timezone.utc).isoformat()
    write_json_atomic(state_path, state)
    ensure_supervision_safe()
    return _removal_result(
        request.site, request.cluster_id, state_path, state, remaining
    )

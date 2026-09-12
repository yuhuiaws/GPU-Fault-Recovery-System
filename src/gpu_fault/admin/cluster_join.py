from __future__ import annotations

import hashlib
import json
import os
import secrets
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Callable, cast

import yaml  # type: ignore[import-untyped,unused-ignore]

from gpu_fault.admin.atomic_json import write_json_atomic
from gpu_fault.admin.bootstrap import (
    _ensure_base_secrets,
    _gpu_nat_eips,
    discover_cluster,
)
from gpu_fault.admin.bootstrap_common import (
    Arn,
    BootstrapError,
    ClusterIdentity,
    CommandRunner,
    ensure_namespace,
    safe_name,
    write_secret,
)
from gpu_fault.admin.bootstrap_services import (
    ensure_adot_writer_role,
    ensure_executor_role,
    provision_node_action_keys,
)
from gpu_fault.admin.cluster_join_evidence import (
    JoinVerificationExpired,
    build_verified_membership_evidence,
    clear_verified_step,
    join_activation_is_irreversible,
    membership_runtime_snapshot,
    verification_is_stale,
)
from gpu_fault.admin.cluster_join_readonly import cached_network_baseline
from gpu_fault.admin.cluster_join_rollback import (
    clear_stale_installer_annotations,
    ensure_kube_context,
    nothing_installed,
    restore_current_context,
    rollback_command,
    rollback_iam_role,
)
from gpu_fault.admin.cluster_join_state import (
    complete_step as _complete,
)
from gpu_fault.admin.cluster_join_state import (
    completed_state_is_current as _completed_state_is_current,
)
from gpu_fault.admin.cluster_join_state import (
    load_join_state as _state,
)
from gpu_fault.admin.cluster_join_state import (
    reset_completed_state as _reset_completed_state,
)
from gpu_fault.admin.cluster_join_state import (
    step_done as _done,
)
from gpu_fault.admin.cluster_readiness import wait_collector_readiness
from gpu_fault.admin.cluster_removal import (
    _clear_installer_annotations,
    _cluster_network,
    _remove_node_action_keys,
    _sync_release_state,
    _wait_vpc_association_absent,
)
from gpu_fault.admin.failure_domain_map import apply_failure_domain_map
from gpu_fault.admin.membership_lock import (
    membership_operation_lock,
    reload_site_for_mutation,
)
from gpu_fault.admin.resource_registry import (
    LegacyInstallationRegistryMissing,
    fetch_installation_resource_registry,
    find_bootstrap_state,
    load_installation_resource_snapshot,
    sync_installation_resource_registry,
    sync_installation_resource_snapshot,  # noqa: F401 - commit adapter seam
    write_installation_resource_snapshot,  # noqa: F401 - commit adapter seam
)
from gpu_fault.admin.site import (
    RenderedSite,
    effective_environment,
    load_site,
    materialized_release_config,
)
from gpu_fault.installation_resources import InstallationResourceSnapshot

DEFAULT_ALLOWED_NAMESPACES = ("gpu-fault-system", "training")
_KUBECONFIG_THREAD_LOCK = Lock()


@dataclass(frozen=True)
class JoinClusterRequest:
    site: RenderedSite
    gpu_cluster_arn: str
    cluster_id: str | None = None
    allowed_namespaces: tuple[str, ...] = DEFAULT_ALLOWED_NAMESPACES
    state_dir: Path | None = None


@dataclass(frozen=True)
class JoinExecution:
    target: ClusterIdentity
    cluster_id: str
    discovery: dict[str, Any]
    local: dict[str, Any]
    prerequisites: dict[str, Any]
    candidate: RenderedSite


@dataclass
class JoinAttempt:
    request: JoinClusterRequest
    state_dir: Path
    state_path: Path
    state: dict[str, Any]
    execution: JoinExecution | None = None


def _fetch_installation_registry(
    site: RenderedSite,
    *,
    output: Path | None = None,
) -> InstallationResourceSnapshot:
    return fetch_installation_resource_registry(site, output=output)


def _load_installation_snapshot(path: Path) -> InstallationResourceSnapshot:
    return load_installation_resource_snapshot(path)


def _sync_installation_snapshot(
    site: RenderedSite,
    snapshot: InstallationResourceSnapshot,
) -> None:
    sync_installation_resource_snapshot(site, snapshot)


def _write_installation_snapshot(
    site: RenderedSite,
    snapshot: InstallationResourceSnapshot,
    *,
    path: Path | None = None,
) -> Path:
    return write_installation_resource_snapshot(site, snapshot, path=path)


def _sync_join_release_state(site: RenderedSite) -> None:
    _sync_release_state(site)


def _run_rollout(
    site: RenderedSite,
    mode: str,
    *,
    cluster_id: str | None = None,
) -> None:
    rollout = (
        site.repository_root
        / "deploy/control-plane/regional/rollout-regional-release.sh"
    )
    with materialized_release_config(site) as config:
        arguments = [str(rollout), mode, "--config", str(config)]
        if cluster_id is not None:
            arguments[2:2] = ["--cluster-id", cluster_id]
        result = subprocess.run(
            arguments,
            cwd=site.repository_root,
            env=effective_environment(site),
            check=False,
        )
    if result.returncode:
        raise BootstrapError(f"regional {mode} failed")


def _identity(value: dict[str, Any]) -> ClusterIdentity:
    normalized = dict(value)
    normalized["subnet_ids"] = tuple(normalized.get("subnet_ids") or ())
    return ClusterIdentity(**normalized)


def _join_context(cluster_arn: str) -> str:
    parsed = Arn.parse(cluster_arn)
    digest = hashlib.sha256(cluster_arn.encode()).hexdigest()[:8]
    return safe_name(f"gpu-fault-gpu-{parsed.resource_name}-{digest}")


def _discover_join_target(
    request: JoinClusterRequest,
    runner: CommandRunner,
) -> tuple[ClusterIdentity, str, dict[str, Any] | None]:
    context = _join_context(request.gpu_cluster_arn)
    target = discover_cluster(
        runner,
        cluster_arn=request.gpu_cluster_arn,
        role="gpu",
        context=context,
    )
    cluster_id = safe_name(request.cluster_id or target.hyperpod_name)
    _validate_target(request, target, cluster_id)
    existing = next(
        (
            item
            for item in request.site.release_config["clusters"]
            if item["eks_cluster_arn"] == target.eks_arn
        ),
        None,
    )
    return target, cluster_id, existing


def _validate_target(
    request: JoinClusterRequest,
    target: ClusterIdentity,
    cluster_id: str,
) -> None:
    cpu = Arn.parse(str(request.site.release_config["cpu_eks_arn"]))
    if target.region != request.site.release_config["aws_region"]:
        raise BootstrapError("GPU cluster Region does not match the existing site")
    if target.account_id != cpu.account:
        raise BootstrapError("GPU cluster account does not match the existing site")
    for item in request.site.release_config["clusters"]:
        if item["eks_cluster_arn"] == target.eks_arn:
            if item["cluster_id"] == cluster_id:
                return
            raise BootstrapError(
                f"GPU EKS cluster is already managed as {item['cluster_id']}"
            )
        if item["hyperpod_cluster_name"] == target.hyperpod_name:
            raise BootstrapError("GPU HyperPod cluster is already managed")
        if item["cluster_id"] == cluster_id:
            raise BootstrapError(f"cluster_id already exists: {cluster_id}")
        if item["context"] == target.context:
            raise BootstrapError(f"kube context already exists: {target.context}")


def _gpu_kubeconfig(site: RenderedSite) -> Path:
    configured = (
        site.environment.get("KUBECONFIG")
        or os.environ.get("KUBECONFIG")
        or str(Path.home() / ".kube/config")
    )
    return Path(configured).expanduser().resolve()


def _shared_ca_file(site: RenderedSite) -> Path:
    for cluster in site.release_config["clusters"]:
        path = Path(str(cluster.get("ca_file") or ""))
        if path.is_file():
            return path
    bootstrap = find_bootstrap_state(site)
    path = Path(
        str(
            ((bootstrap or {}).get("resources") or {}).get("pki", {}).get("ca_file", "")
        )
    )
    if path.is_file():
        return path
    raise BootstrapError(
        "cannot locate the site private CA; retain bootstrap state or one "
        "managed cluster CA file"
    )


def _control_plane_url(site: RenderedSite) -> str:
    hostname = str((site.release_config.get("dns") or {}).get("hostname") or "")
    if hostname:
        return "https://" + hostname
    for cluster in site.release_config["clusters"]:
        value = str(cluster.get("control_plane_url") or "")
        if value:
            return value
    raise BootstrapError("site has no control-plane URL")


def _list_nodes(
    runner: CommandRunner,
    *,
    kubeconfig: Path,
    target: ClusterIdentity,
) -> list[str]:
    document = json.loads(
        runner.run(
            [
                "kubectl",
                "--kubeconfig",
                str(kubeconfig),
                "--context",
                target.context,
                "get",
                "nodes",
                "-l",
                "sagemaker.amazonaws.com/cluster-name=" + target.hyperpod_name,
                "-o",
                "json",
            ]
        )
    )
    nodes = sorted(str(item["metadata"]["name"]) for item in document.get("items", []))
    if not nodes:
        raise BootstrapError("target GPU HyperPod has no Kubernetes nodes")
    return nodes


def _export_registry(site: RenderedSite, path: Path) -> None:
    if path.exists():
        load_installation_resource_snapshot(path)
        return
    try:
        fetch_installation_resource_registry(site, output=path)
    except LegacyInstallationRegistryMissing:
        sync_installation_resource_registry(site)
        fetch_installation_resource_registry(site, output=path)


def _ensure_nlb_ingress(
    *,
    region: str,
    security_group: str,
    eips: list[str],
) -> list[str]:
    created = []
    for eip in eips:
        result = subprocess.run(
            [
                "aws",
                "ec2",
                "authorize-security-group-ingress",
                "--region",
                region,
                "--group-id",
                security_group,
                "--protocol",
                "tcp",
                "--port",
                "443",
                "--cidr",
                f"{eip}/32",
            ],
            text=True,
            capture_output=True,
        )
        if result.returncode == 0:
            created.append(eip)
        elif "InvalidPermission.Duplicate" not in result.stderr:
            raise BootstrapError(
                f"cannot authorize NLB ingress for {eip}: {result.stderr.strip()}"
            )
    return created


def _wait_zone_association(
    runner: CommandRunner,
    *,
    region: str,
    hosted_zone_id: str,
    vpc_id: str,
) -> None:
    for _ in range(60):
        document = runner.aws_json(
            region,
            "route53",
            "get-hosted-zone",
            "--id",
            hosted_zone_id,
        )
        if any(
            item.get("VPCRegion") == region and item.get("VPCId") == vpc_id
            for item in document.get("VPCs", [])
        ):
            return
        time.sleep(5)
    raise BootstrapError("Route53 VPC association did not become visible")


def _existing_cluster_networks(
    runner: CommandRunner,
    site: RenderedSite,
) -> list[dict[str, Any]]:
    clusters = list(site.release_config["clusters"])
    if not clusters:
        return []
    region = str(site.release_config["aws_region"])
    values: dict[int, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=min(4, len(clusters))) as executor:
        futures = {
            executor.submit(
                _cluster_network,
                runner,
                region=region,
                eks_arn=str(item["eks_cluster_arn"]),
            ): index
            for index, item in enumerate(clusters)
        }
        for future in as_completed(futures):
            index = futures[future]
            try:
                values[index] = future.result()
            except Exception as exc:
                raise BootstrapError(
                    f"existing GPU network discovery failed: {exc}"
                ) from exc
    return [values[index] for index in range(len(clusters))]


def _ensure_network(
    runner: CommandRunner,
    *,
    site: RenderedSite,
    target: ClusterIdentity,
    existing_networks: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    region = str(site.release_config["aws_region"])
    baseline = (
        existing_networks
        if existing_networks is not None
        else _existing_cluster_networks(runner, site)
    )
    existing_eips = {
        value for network in baseline for value in network.get("nat_eips", [])
    }
    eips = list(_gpu_nat_eips(runner, target))
    created_eips = _ensure_nlb_ingress(
        region=region,
        security_group=str(site.release_config["nlb"]["security_group"]),
        eips=eips,
    )
    hosted_zone_id = str(
        (site.release_config.get("dns") or {}).get("hosted_zone_id") or ""
    )
    if not hosted_zone_id:
        raise BootstrapError("join-cluster requires the site private hosted zone")
    zone = runner.aws_json(
        region,
        "route53",
        "get-hosted-zone",
        "--id",
        hosted_zone_id,
    )
    associated = {
        (str(item.get("VPCRegion") or ""), str(item.get("VPCId") or ""))
        for item in zone.get("VPCs", [])
    }
    association_created = False
    if (target.region, target.vpc_id) not in associated:
        runner.run(
            [
                "aws",
                "route53",
                "associate-vpc-with-hosted-zone",
                "--hosted-zone-id",
                hosted_zone_id,
                "--vpc",
                f"VPCRegion={target.region},VPCId={target.vpc_id}",
                "--comment",
                "GPU fault managed data plane",
            ],
            mutate=True,
            capture=False,
        )
        association_created = True
        _wait_zone_association(
            runner,
            region=region,
            hosted_zone_id=hosted_zone_id,
            vpc_id=target.vpc_id,
        )
    return {
        "vpc_id": target.vpc_id,
        "nat_eips": eips,
        "created_ingress_eips": sorted(set(created_eips) - existing_eips),
        "existing_vpc_ids": sorted({str(item["vpc_id"]) for item in baseline}),
        "hosted_zone_id": hosted_zone_id,
        "association_created": association_created,
    }


def _parallel_prerequisites(
    request: JoinClusterRequest,
    *,
    runner: CommandRunner,
    target: ClusterIdentity,
    cluster_id: str,
    gpu_kubeconfig: Path,
    fleet_master_file: Path,
    state: dict[str, Any],
    state_path: Path,
    network_baseline: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    cached = dict((state.get("evidence") or {}).get("PREREQUISITES_READY") or {})
    tasks: dict[str, Callable[[], Any]] = {
        "executor_role": lambda: ensure_executor_role(
            runner,
            cluster=target,
            namespace=str(request.site.release_config["namespace"]),
            site_id=str(request.site.release_config["site_name"]),
        ),
        "network": lambda: _ensure_network(
            runner,
            site=request.site,
            target=target,
            existing_networks=network_baseline,
        ),
        "node_keys": lambda: provision_node_action_keys(
            runner,
            repository_root=request.site.repository_root,
            cpu_kubeconfig=Path(str(request.site.release_config["cpu_kubeconfig"])),
            gpu_kubeconfig=gpu_kubeconfig,
            namespace=str(request.site.release_config["namespace"]),
            cluster=target,
            cluster_id=cluster_id,
            fleet_master_file=fleet_master_file,
        ),
    }
    pending = {name: task for name, task in tasks.items() if name not in cached}
    failures = []
    with ThreadPoolExecutor(max_workers=max(1, len(pending))) as executor:
        futures = {executor.submit(task): name for name, task in pending.items()}
        for future in as_completed(futures):
            name = futures[future]
            try:
                cached[name] = future.result()
                state.setdefault("evidence", {})["PREREQUISITES_READY"] = cached
                write_json_atomic(state_path, state)
            except Exception as exc:
                failures.append((name, exc))
    if failures:
        names = ", ".join(sorted(name for name, _exc in failures))
        error = failures[0][1]
        error.add_note(f"join prerequisite task(s) failed: {names}")
        raise error
    # The data-plane ADOT writer role follows the executor role on purpose: both
    # sit on the cluster's OIDC provider, and two concurrent ensures would race
    # ``create-open-id-connect-provider``. No workspace, no collector, no role.
    workspace_id = request.site.release_config["health"].get("amp_workspace_id")
    if workspace_id and "adot_writer_role" not in cached:
        cached["adot_writer_role"] = ensure_adot_writer_role(
            runner,
            cluster=target,
            namespace=str(request.site.release_config["namespace"]),
            site_id=str(request.site.release_config["site_name"]),
            amp_workspace_id=str(workspace_id),
        )
        state.setdefault("evidence", {})["PREREQUISITES_READY"] = cached
        write_json_atomic(state_path, state)
    return cached


def _cluster_document(
    request: JoinClusterRequest,
    *,
    target: ClusterIdentity,
    cluster_id: str,
    role_arn: str,
    token_file: Path,
    ca_file: Path,
    fleet_master_file: Path,
    adot_irsa_role_arn: str | None = None,
) -> dict[str, Any]:
    namespaces = tuple(
        sorted(
            {
                *DEFAULT_ALLOWED_NAMESPACES,
                *request.allowed_namespaces,
            }
        )
    )
    document = {
        "clusterId": cluster_id,
        "context": target.context,
        "region": target.region,
        "hyperpodClusterName": target.hyperpod_name,
        "eksClusterArn": target.eks_arn,
        "executorIrsaRoleArn": role_arn,
        "allowedNamespaces": list(namespaces),
        "agentEndpointAllowedCidrs": list(target.subnet_cidrs),
        "controlPlaneUrl": _control_plane_url(request.site),
        "tokenFile": str(token_file),
        "caFile": str(ca_file),
        "fleetMasterFile": str(fleet_master_file),
    }
    # Absent means the release skips this cluster's data-plane collector.
    if adot_irsa_role_arn:
        document["adotIrsaRoleArn"] = adot_irsa_role_arn
    return document


def _write_candidate_site(
    request: JoinClusterRequest,
    *,
    state_dir: Path,
    attempt: int,
    gpu_kubeconfig: Path,
    cluster: dict[str, Any],
) -> Path:
    document = yaml.safe_load(request.site.source.read_text(encoding="utf-8"))
    document["spec"]["gpuKubeconfig"] = str(gpu_kubeconfig)
    clusters = list(document["spec"].get("clusters") or [])
    matches = [
        item
        for item in clusters
        if item.get("clusterId") == cluster["clusterId"]
        or item.get("eksClusterArn") == cluster["eksClusterArn"]
    ]
    if matches:
        if len(matches) != 1 or matches[0] != cluster:
            raise BootstrapError("candidate site contains a conflicting GPU cluster")
    else:
        clusters.append(cluster)
    document["spec"]["clusters"] = clusters
    path = state_dir / f"site.candidate-{attempt:03d}.yaml"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    temporary.chmod(0o600)
    temporary.replace(path)
    return path


def _bootstrap_state_path(site: RenderedSite) -> Path | None:
    site_id = str(site.release_config["site_name"])
    candidates = [site.source.parent / "bootstrap-state.json"]
    candidates.extend(
        sorted((Path.home() / ".gpu-fault/bootstrap").glob("*/bootstrap-state.json"))
    )
    matches = []
    for path in candidates:
        if not path.is_file():
            continue
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("site_id") == site_id:
            matches.append(path.resolve())
    unique = sorted(set(matches))
    if len(unique) > 1:
        raise BootstrapError(
            "multiple bootstrap states match this site: "
            + ", ".join(str(item) for item in unique)
        )
    return unique[0] if unique else None


def _update_bootstrap_state(
    site: RenderedSite,
    *,
    cluster_id: str,
    role: dict[str, Any],
    network: dict[str, Any],
    adot_writer_role: dict[str, Any] | None = None,
) -> None:
    path = _bootstrap_state_path(site)
    if path is None:
        return
    value = cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))
    resources = value.setdefault("resources", {})
    resources[f"executor_role:{cluster_id}"] = role
    resources[f"node_keys:{cluster_id}"] = {"cluster_id": cluster_id}
    completed_names = {f"executor_role:{cluster_id}", f"node_keys:{cluster_id}"}
    if adot_writer_role:
        # Recorded like the executor role so the next bootstrap re-proves it
        # and the registry/uninstall know it belongs to this cluster.
        resources[f"adot_writer_role:{cluster_id}"] = adot_writer_role
        completed_names.add(f"adot_writer_role:{cluster_id}")
    nlb = resources.setdefault("nlb_network", {})
    nlb["gpu_nat_eips"] = sorted(
        {
            *nlb.get("gpu_nat_eips", []),
            *network.get("nat_eips", []),
        }
    )
    if network["vpc_id"] not in set(network.get("existing_vpc_ids") or []):
        pki = resources.setdefault("pki", {})
        associations = list(pki.get("vpc_associations") or [])
        if not any(item.get("vpc_id") == network["vpc_id"] for item in associations):
            associations.append(
                {
                    "vpc_id": network["vpc_id"],
                    "vpc_region": site.release_config["aws_region"],
                    "ownership": "CREATED",
                }
            )
        pki["vpc_associations"] = associations
    completed = set(value.get("completed_tasks") or [])
    completed.update(completed_names)
    value["completed_tasks"] = sorted(completed)
    value.setdefault("removed_clusters", {}).pop(cluster_id, None)
    value.setdefault("joined_clusters", {})[cluster_id] = {
        "joined_at": datetime.now(timezone.utc).isoformat(),
        "vpc_id": network["vpc_id"],
    }
    write_json_atomic(path, value)


def _cleanup_candidate(
    candidate: RenderedSite,
    *,
    cluster_id: str,
    state_dir: Path,
    attempt: int,
) -> None:
    errors = []
    with materialized_release_config(candidate) as config:
        if nothing_installed(candidate, config):
            # The release failed before it installed anything (live
            # 2026-09-12: at the endpoint gate); the cleanup script refuses an
            # empty inventory, and there is nothing for it to undo.
            return
        result = subprocess.run(
            [
                str(
                    candidate.repository_root
                    / "deploy/control-plane/regional/prepare-clean-redeploy.sh"
                ),
                "--config",
                str(config),
                "--scope",
                "gpu",
                "--cluster-id",
                cluster_id,
                "--mode",
                "clean",
                "--node-mode",
                "uninstall",
                "--state-file",
                str(state_dir / f"rollback-kubernetes-{attempt:03d}.json"),
                "--execute",
            ],
            cwd=candidate.repository_root,
            env=effective_environment(candidate),
            text=True,
            capture_output=True,
            check=False,
        )
    if result.returncode:
        errors.append("GPU cleanup: " + (result.stderr or "").strip())
    if errors:
        raise BootstrapError("; ".join(errors))


def _rollback_network(network: dict[str, Any], site: RenderedSite) -> None:
    region = str(site.release_config["aws_region"])
    for eip in network.get("created_ingress_eips", []):
        rollback_command(
            [
                "aws",
                "ec2",
                "revoke-security-group-ingress",
                "--region",
                region,
                "--group-id",
                str(site.release_config["nlb"]["security_group"]),
                "--protocol",
                "tcp",
                "--port",
                "443",
                "--cidr",
                f"{eip}/32",
            ],
            not_found=("InvalidPermission.NotFound",),
        )
    if network.get("association_created"):
        rollback_command(
            [
                "aws",
                "route53",
                "disassociate-vpc-from-hosted-zone",
                "--hosted-zone-id",
                str(network["hosted_zone_id"]),
                "--vpc",
                f"VPCRegion={region},VPCId={network['vpc_id']}",
            ],
            not_found=("VPCAssociationNotFound",),
        )
        _wait_vpc_association_absent(
            hosted_zone_id=str(network["hosted_zone_id"]),
            region=region,
            vpc_id=str(network["vpc_id"]),
        )


def _rollback(
    request: JoinClusterRequest,
    *,
    state_dir: Path,
    state_path: Path,
    state: dict[str, Any],
) -> None:
    evidence = state.get("evidence") or {}
    discovery = evidence.get("DISCOVERED") or {}
    cluster_id = str(discovery.get("cluster_id") or "")
    errors = []
    local = evidence.get("LOCAL_INPUTS_READY") or {}
    nodes = list(local.get("nodes") or [])
    target = discovery.get("target") or {}
    context = str(target.get("context") or "")
    kubeconfig = Path(str(local.get("gpu_kubeconfig") or _gpu_kubeconfig(request.site)))
    if context:
        # A retried rollback finds the context its first run deleted; every
        # Kubernetes-side undo below needs it back.
        try:
            with _KUBECONFIG_THREAD_LOCK:
                ensure_kube_context(kubeconfig, target)
        except Exception as exc:
            errors.append(f"kube context restore: {exc}")
    candidate_path = Path(
        str((evidence.get("CANDIDATE_READY") or {}).get("site_file") or "")
    )
    candidate: RenderedSite | None = None
    # A release that started may have registered the cluster before it failed;
    # its registry entry is undone exactly like a joined cluster's.
    released = _done(state, "JOINED") or _done(state, "RELEASE_STARTED")
    if cluster_id and candidate_path.is_file():
        try:
            candidate = load_site(
                candidate_path,
                repository_root=request.site.repository_root,
            )
            if released:
                _run_rollout(candidate, "fail-cluster", cluster_id=cluster_id)
            _cleanup_candidate(
                candidate,
                cluster_id=cluster_id,
                state_dir=state_dir,
                attempt=int(state.get("attempt") or 1),
            )
        except Exception as exc:
            errors.append(f"kubernetes/control rollback: {exc}")
    try:
        _clear_installer_annotations(request.site, dict(target), nodes)
    except Exception as exc:
        errors.append(f"installer annotation rollback: {exc}")
    try:
        _remove_node_action_keys(request.site, nodes)
    except Exception as exc:
        errors.append(f"node key rollback: {exc}")
    if context:
        namespace = str(request.site.release_config["namespace"])
        result = subprocess.run(
            [
                "kubectl",
                "--kubeconfig",
                str(kubeconfig),
                "--context",
                context,
                "delete",
                "namespace",
                namespace,
                "--ignore-not-found",
                "--wait=true",
                "--timeout=10m",
            ],
            text=True,
            capture_output=True,
        )
        if result.returncode:
            errors.append("namespace rollback: " + (result.stderr or "").strip())
    prerequisites = evidence.get("PREREQUISITES_READY") or {}
    network = prerequisites.get("network")
    if isinstance(network, dict):
        try:
            _rollback_network(network, request.site)
        except Exception as exc:
            errors.append(f"network rollback: {exc}")
    for prerequisite, label in (
        ("executor_role", "Executor"),
        ("adot_writer_role", "ADOT writer"),
    ):
        rollback_iam_role(prerequisites.get(prerequisite), label=label, errors=errors)
    token_file = Path(str(local.get("token_file") or ""))
    secure = state_dir / "secure"
    for path in {
        token_file,
        secure / f"{cluster_id}.token",
        Path(str(local.get("fleet_master_file") or "")),
        secure / "fleet-master",
    }:
        if path.is_file():
            path.unlink()
    if context:
        result = subprocess.run(
            [
                "kubectl",
                "--kubeconfig",
                str(kubeconfig),
                "config",
                "delete-context",
                context,
            ],
            text=True,
            capture_output=True,
        )
        message = (result.stdout or "") + "\n" + (result.stderr or "")
        if result.returncode and "not found" not in message.lower():
            errors.append("kube context rollback: " + result.stderr.strip())
        else:
            restore_current_context(kubeconfig, deleted=context, errors=errors)
    if not errors and candidate is not None:
        try:
            from gpu_fault.admin.cluster_join_commit import rollback_membership

            execution = JoinExecution(
                target=_identity(dict(target)),
                cluster_id=cluster_id,
                discovery=dict(discovery),
                local=dict(local),
                prerequisites=dict(prerequisites),
                candidate=candidate,
            )
            rollback_membership(
                request,
                execution=execution,
                joined=released,
            )
        except Exception as exc:
            errors.append(f"membership rollback: {exc}")
    if errors:
        state["phase"] = "ROLLBACK_FAILED"
        state["rollback_errors"] = errors
    else:
        state["phase"] = "ROLLED_BACK"
        state["completed_steps"] = [
            item
            for item in state.get("completed_steps", [])
            if item in {"PRECHECKED", "DISCOVERED"}
        ]
        state["evidence"] = {
            key: value
            for key, value in evidence.items()
            if key in {"PRECHECKED", "DISCOVERED"}
        }
    state["updated_at"] = datetime.now(timezone.utc).isoformat()
    write_json_atomic(state_path, state)
    if errors:
        raise BootstrapError("; ".join(errors))


def _prepare_execution(
    request: JoinClusterRequest,
    *,
    runner: CommandRunner,
    state_dir: Path,
    state_path: Path,
    state: dict[str, Any],
    candidate_preflight: bool = True,
    network_baseline: list[dict[str, Any]] | None = None,
) -> JoinExecution | dict[str, Any]:
    if not _done(state, "PRECHECKED"):
        # The candidate verify after the rollout covers every cluster in the
        # site, the existing ones included; a baseline verify of the site first
        # was a second full verify that could only repeat that answer.
        _complete(state_path, state, "PRECHECKED")
    if not _done(state, "DISCOVERED"):
        target, cluster_id, existing = _discover_join_target(request, runner)
        if existing is not None:
            _complete(
                state_path,
                state,
                "DISCOVERED",
                {
                    "target": asdict(target),
                    "cluster_id": str(existing["cluster_id"]),
                    "already_managed": True,
                },
            )
            state["phase"] = "COMPLETED"
            state["updated_at"] = datetime.now(timezone.utc).isoformat()
            write_json_atomic(state_path, state)
            return {
                "site_id": request.site.release_config["site_name"],
                "cluster_id": existing["cluster_id"],
                "phase": "ALREADY_MANAGED",
                "state_file": str(state_path),
                "site_file": str(request.site.source),
            }
        attempt = int(state.get("attempt") or 1)
        registry_path = state_dir / f"installation-resources-before-{attempt:03d}.json"
        _export_registry(request.site, registry_path)
        _complete(
            state_path,
            state,
            "DISCOVERED",
            {
                "target": asdict(target),
                "cluster_id": cluster_id,
                "registry_snapshot": str(registry_path),
            },
        )
    discovery = state["evidence"]["DISCOVERED"]
    target = _identity(discovery["target"])
    cluster_id = str(discovery["cluster_id"])
    if network_baseline is None:
        network_baseline = cast(
            list[dict[str, Any]],
            cached_network_baseline(
                state_dir / "network-baseline.json",
                request.site,
                lambda: cast(
                    list[dict[str, object]],
                    _existing_cluster_networks(runner, request.site),
                ),
            ),
        )
    if not _done(state, "LOCAL_INPUTS_READY"):
        gpu_kubeconfig = _gpu_kubeconfig(request.site)
        gpu_kubeconfig.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        with _KUBECONFIG_THREAD_LOCK:
            runner.run(
                [
                    "aws",
                    "eks",
                    "update-kubeconfig",
                    "--region",
                    target.region,
                    "--name",
                    target.eks_name,
                    "--kubeconfig",
                    str(gpu_kubeconfig),
                    "--alias",
                    target.context,
                ],
                mutate=True,
                capture=False,
            )
        if gpu_kubeconfig.exists():
            gpu_kubeconfig.chmod(0o600)
        ensure_namespace(
            runner,
            kubeconfig=gpu_kubeconfig,
            context=target.context,
            namespace=str(request.site.release_config["namespace"]),
        )
        # The cluster token is site state and lives beside the tokens bootstrap
        # wrote, in the site's ``secure/``; the join directory is disposable
        # transaction state. The fleet-master copy stays in the join directory:
        # rollback deletes it, and the site's own copy has to survive that.
        secure = state_dir / "secure"
        secure.mkdir(mode=0o700, parents=True, exist_ok=True)
        site_secure = request.site.source.parent / "secure"
        site_secure.mkdir(mode=0o700, parents=True, exist_ok=True)
        token_file = site_secure / f"{cluster_id}.token"
        write_secret(token_file, secrets.token_hex(32))
        fleet_master_file = _ensure_base_secrets(
            runner,
            cpu_kubeconfig=Path(str(request.site.release_config["cpu_kubeconfig"])),
            namespace=str(request.site.release_config["namespace"]),
            secure_dir=secure,
        )
        ca_file = _shared_ca_file(request.site)
        nodes = _list_nodes(
            runner,
            kubeconfig=gpu_kubeconfig,
            target=target,
        )
        clear_stale_installer_annotations(
            request.site, {"context": target.context}, nodes
        )
        _complete(
            state_path,
            state,
            "LOCAL_INPUTS_READY",
            {
                "gpu_kubeconfig": str(gpu_kubeconfig),
                "token_file": str(token_file),
                "fleet_master_file": str(fleet_master_file),
                "ca_file": str(ca_file),
                "nodes": nodes,
            },
        )
    local = state["evidence"]["LOCAL_INPUTS_READY"]
    if not _done(state, "PREREQUISITES_READY"):
        prerequisites = _parallel_prerequisites(
            request,
            runner=runner,
            target=target,
            cluster_id=cluster_id,
            gpu_kubeconfig=Path(local["gpu_kubeconfig"]),
            fleet_master_file=Path(local["fleet_master_file"]),
            state=state,
            state_path=state_path,
            network_baseline=network_baseline,
        )
        _complete(state_path, state, "PREREQUISITES_READY", prerequisites)
    prerequisites = state["evidence"]["PREREQUISITES_READY"]
    if not _done(state, "CANDIDATE_READY"):
        cluster = _cluster_document(
            request,
            target=target,
            cluster_id=cluster_id,
            role_arn=str(prerequisites["executor_role"]["role_arn"]),
            token_file=Path(local["token_file"]),
            ca_file=Path(local["ca_file"]),
            fleet_master_file=Path(local["fleet_master_file"]),
            adot_irsa_role_arn=(prerequisites.get("adot_writer_role") or {}).get(
                "role_arn"
            ),
        )
        candidate_path = _write_candidate_site(
            request,
            state_dir=state_dir,
            attempt=int(state.get("attempt") or 1),
            gpu_kubeconfig=Path(local["gpu_kubeconfig"]),
            cluster=cluster,
        )
        candidate = load_site(
            candidate_path,
            repository_root=request.site.repository_root,
        )
        if candidate_preflight:
            _run_rollout(candidate, "preflight")
        _complete(
            state_path,
            state,
            "CANDIDATE_READY",
            {"site_file": str(candidate_path), "cluster": cluster},
        )
    candidate = load_site(
        Path(state["evidence"]["CANDIDATE_READY"]["site_file"]),
        repository_root=request.site.repository_root,
    )
    return JoinExecution(
        target=target,
        cluster_id=cluster_id,
        discovery=discovery,
        local=local,
        prerequisites=prerequisites,
        candidate=candidate,
    )


def _write_batch_candidate_site(
    site: RenderedSite,
    executions: list[JoinExecution],
    *,
    state_dir: Path,
) -> RenderedSite:
    document = yaml.safe_load(site.source.read_text(encoding="utf-8"))
    clusters = [dict(item) for item in document["spec"].get("clusters", [])]
    kubeconfigs = {str(execution.local["gpu_kubeconfig"]) for execution in executions}
    if len(kubeconfigs) != 1:
        raise BootstrapError("batch join produced inconsistent GPU kubeconfig paths")
    for execution in executions:
        candidate = yaml.safe_load(
            execution.candidate.source.read_text(encoding="utf-8")
        )
        matches = [
            dict(item)
            for item in candidate["spec"].get("clusters", [])
            if item.get("clusterId") == execution.cluster_id
        ]
        if len(matches) != 1:
            raise BootstrapError(
                f"candidate site has no unique cluster {execution.cluster_id}"
            )
        target = matches[0]
        conflicts = [
            item
            for item in clusters
            if item.get("clusterId") == target.get("clusterId")
            or item.get("eksClusterArn") == target.get("eksClusterArn")
            or item.get("hyperpodClusterName") == target.get("hyperpodClusterName")
            or item.get("context") == target.get("context")
        ]
        if conflicts:
            if len(conflicts) != 1 or conflicts[0] != target:
                raise BootstrapError(
                    f"batch join contains a conflicting cluster {execution.cluster_id}"
                )
        else:
            clusters.append(target)
    document["spec"]["gpuKubeconfig"] = next(iter(kubeconfigs))
    document["spec"]["clusters"] = sorted(
        clusters,
        key=lambda item: str(item.get("clusterId") or ""),
    )
    state_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    path = state_dir / "site.batch-candidate.yaml"
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=state_dir,
        prefix=".site.batch-candidate.",
        suffix=".tmp",
        delete=False,
    ) as temporary:
        yaml.safe_dump(document, temporary, sort_keys=False)
        temporary_path = Path(temporary.name)
    temporary_path.chmod(0o600)
    temporary_path.replace(path)
    return load_site(path, repository_root=site.repository_root)


def _deploy_cluster(
    *,
    execution: JoinExecution,
    state_path: Path,
    state: dict[str, Any],
) -> None:
    cluster_id = execution.cluster_id
    if not _done(state, "JOINED"):
        _complete(state_path, state, "RELEASE_STARTED")
        _run_rollout(execution.candidate, "join-cluster", cluster_id=cluster_id)
        _complete(state_path, state, "JOINED")
    if not _done(state, "COLLECTORS_READY"):
        readiness = wait_collector_readiness(execution.candidate, cluster_id)
        _complete(
            state_path,
            state,
            "COLLECTORS_READY",
            {
                "nodes": len(readiness.get("nodes") or []),
                "ready": readiness.get("ready"),
            },
        )


def _verify_join_candidate(
    *,
    execution: JoinExecution,
    state_path: Path,
    state: dict[str, Any],
) -> None:
    verified = (state.get("evidence") or {}).get("VERIFIED")
    if _done(state, "VERIFIED") and verification_is_stale(
        verified if isinstance(verified, dict) else {}
    ):
        # Evidence from an earlier run that is past its window: the data plane
        # is still joined and healthy, so verify it again rather than undo it.
        clear_verified_step(state_path, state)
    if not _done(state, "VERIFIED"):
        before = membership_runtime_snapshot(execution.candidate)
        _run_rollout(execution.candidate, "verify")
        after = membership_runtime_snapshot(execution.candidate)
        _complete(
            state_path,
            state,
            "VERIFIED",
            build_verified_membership_evidence(
                before,
                after,
                candidate_site_sha256=execution.candidate.source_sha256,
                source_site_sha256=str(state["source_site_sha256"]),
                source_site_non_membership_sha256=str(
                    state["source_site_non_membership_sha256"]
                ),
                candidate_cluster_ids=[
                    str(item["cluster_id"])
                    for item in execution.candidate.release_config["clusters"]
                ],
                cluster_id=execution.cluster_id,
            ),
        )


def _activate_and_commit(
    request: JoinClusterRequest,
    *,
    execution: JoinExecution,
    state_dir: Path,
    state_path: Path,
    state: dict[str, Any],
) -> None:
    commit_membership(
        request,
        execution=execution,
        state_dir=state_dir,
        state_path=state_path,
        state=state,
    )


def commit_membership(
    request: JoinClusterRequest,
    *,
    execution: JoinExecution,
    state_dir: Path,
    state_path: Path,
    state: dict[str, Any],
) -> None:
    """Activate the cluster, commit the site, then refresh what depends on membership."""

    from gpu_fault.admin.cluster_join_commit import activate_and_commit

    activate_and_commit(
        request,
        execution=execution,
        state_dir=state_dir,
        state_path=state_path,
        state=state,
    )
    # Membership is final: the failure-domain map must now cover the new
    # cluster's nodes, rendered from the committed site rather than the
    # candidate so a concurrent batch join is not narrowed to one member.
    refresh_failure_domain_map(request.site)


def refresh_failure_domain_map(site: RenderedSite) -> None:
    """Re-render the control-worker's failure-domain map from the committed site."""

    apply_failure_domain_map(
        load_site(site.source, repository_root=site.repository_root)
    )


def _deploy_and_commit(
    request: JoinClusterRequest,
    *,
    execution: JoinExecution,
    state_dir: Path,
    state_path: Path,
    state: dict[str, Any],
) -> None:
    _deploy_cluster(
        execution=execution,
        state_path=state_path,
        state=state,
    )
    for _round in range(2):
        _verify_join_candidate(
            execution=execution,
            state_path=state_path,
            state=state,
        )
        try:
            _activate_and_commit(
                request,
                execution=execution,
                state_dir=state_dir,
                state_path=state_path,
                state=state,
            )
        except JoinVerificationExpired:
            # The window closed between verify and commit. Nothing about the
            # rolled-out data plane is wrong, so re-verify instead of rolling it
            # back; one retry, so a clock that keeps producing stale evidence
            # cannot loop.
            clear_verified_step(state_path, state)
            continue
        return
    raise BootstrapError("join verification evidence expired again after a re-verify")


def _site_contains_cluster(path: Path, cluster_id: str) -> bool:
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    return any(
        item.get("clusterId") == cluster_id
        for item in document["spec"].get("clusters", [])
    )


def _record_join_failure(
    attempt: JoinAttempt,
    execution: JoinExecution | None,
) -> None:
    del execution
    activation_started = join_activation_is_irreversible(attempt.state)
    attempt.state["phase"] = (
        "FAILED_AFTER_ACTIVATION" if activation_started else "FAILED"
    )
    attempt.state["updated_at"] = datetime.now(timezone.utc).isoformat()
    write_json_atomic(attempt.state_path, attempt.state)
    if not activation_started and attempt.request.site.release_config.get(
        "auto_rollback", True
    ):
        _rollback(
            attempt.request,
            state_dir=attempt.state_dir,
            state_path=attempt.state_path,
            state=attempt.state,
        )


def _complete_join(
    attempt: JoinAttempt,
    execution: JoinExecution,
) -> dict[str, Any]:
    attempt.state["phase"] = "COMPLETED"
    attempt.state["updated_at"] = datetime.now(timezone.utc).isoformat()
    write_json_atomic(attempt.state_path, attempt.state)
    return {
        "site_id": attempt.request.site.release_config["site_name"],
        "cluster_id": execution.cluster_id,
        "gpu_eks_arn": execution.target.eks_arn,
        "gpu_hyperpod_arn": execution.target.hyperpod_arn,
        "phase": "COMPLETED",
        "state_file": str(attempt.state_path),
        "site_file": str(attempt.request.site.source),
        "cpu_control_plane": "PRESERVED",
        "gpu_cluster": "PRESERVED",
    }


def _join_cluster_locked(
    request: JoinClusterRequest,
    *,
    runner: CommandRunner | None = None,
) -> dict[str, Any]:
    state_dir, state_path, state = _state(request)
    attempt = JoinAttempt(request, state_dir, state_path, state)
    if state.get("phase") == "ROLLBACK_FAILED":
        # Finish the undo the last run could not; it raises if it still cannot.
        _rollback(request, state_dir=state_dir, state_path=state_path, state=state)
    if state.get("phase") == "ROLLED_BACK":
        # Its recorded discovery describes the world that made it fail.
        _reset_completed_state(
            request, state_dir=state_dir, state_path=state_path, state=state
        )
    if state.get("phase") == "COMPLETED":
        if _completed_state_is_current(request, state):
            if not _done(state, "RELEASE_STATE_UPDATED"):
                _sync_release_state(request.site)
                _complete(
                    state_path,
                    state,
                    "RELEASE_STATE_UPDATED",
                    {
                        "cluster_ids": [
                            item["cluster_id"]
                            for item in request.site.release_config["clusters"]
                        ]
                    },
                )
                state["phase"] = "COMPLETED"
                state["updated_at"] = datetime.now(timezone.utc).isoformat()
                write_json_atomic(state_path, state)
            return {
                "site_id": request.site.release_config["site_name"],
                "cluster_id": state["evidence"]["DISCOVERED"]["cluster_id"],
                "phase": "COMPLETED",
                "state_file": str(state_path),
            }
        _reset_completed_state(
            request,
            state_dir=state_dir,
            state_path=state_path,
            state=state,
        )
    active_runner = runner or CommandRunner()
    execution: JoinExecution | None = None
    try:
        prepared = _prepare_execution(
            request,
            runner=active_runner,
            state_dir=state_dir,
            state_path=state_path,
            state=state,
        )
        if isinstance(prepared, dict):
            return prepared
        execution = prepared
        _deploy_and_commit(
            request,
            execution=execution,
            state_dir=state_dir,
            state_path=state_path,
            state=state,
        )
    except Exception:
        _record_join_failure(attempt, execution)
        raise

    assert execution is not None
    return _complete_join(attempt, execution)


def join_cluster(
    request: JoinClusterRequest,
    *,
    runner: CommandRunner | None = None,
) -> dict[str, Any]:
    with membership_operation_lock(request.site):
        current = reload_site_for_mutation(request.site)
        return _join_cluster_locked(replace(request, site=current), runner=runner)

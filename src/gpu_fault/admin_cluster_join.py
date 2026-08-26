from __future__ import annotations

import hashlib
import json
import os
import secrets
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, cast

import yaml  # type: ignore[import-untyped]

from gpu_fault.admin_aws_cleanup import write_json_atomic
from gpu_fault.admin_bootstrap import (
    _ensure_base_secrets,
    _gpu_nat_eips,
    discover_cluster,
)
from gpu_fault.admin_bootstrap_common import (
    Arn,
    BootstrapError,
    ClusterIdentity,
    CommandRunner,
    ensure_namespace,
    safe_name,
    write_secret,
)
from gpu_fault.admin_bootstrap_services import (
    ensure_executor_role,
    provision_node_action_keys,
)
from gpu_fault.admin_cluster_readiness import wait_collector_readiness
from gpu_fault.admin_cluster_removal import (
    _clear_installer_annotations,
    _cluster_network,
    _remove_node_action_keys,
    _sync_release_state,
    _wait_vpc_association_absent,
)
from gpu_fault.admin_resource_registry import (
    LegacyInstallationRegistryMissing,
    fetch_installation_resource_registry,
    find_bootstrap_state,
    load_installation_resource_snapshot,
    sync_installation_resource_registry,
    sync_installation_resource_snapshot,
    write_installation_resource_snapshot,
)
from gpu_fault.admin_site import (
    RenderedSite,
    effective_environment,
    load_site,
    materialized_release_config,
)
from gpu_fault.installation_resources import (
    InstallationResource,
    InstallationResourceDeletePolicy,
    InstallationResourceOwnership,
    InstallationResourceSnapshot,
    InstallationResourceStatus,
)

DEFAULT_ALLOWED_NAMESPACES = ("gpu-fault-system", "training")


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


def _state(request: JoinClusterRequest) -> tuple[Path, Path, dict[str, Any]]:
    identity = hashlib.sha256(request.gpu_cluster_arn.encode()).hexdigest()[:12]
    state_dir = (
        request.state_dir or request.site.source.parent / "join-cluster" / identity
    ).expanduser()
    state_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    state_dir.chmod(0o700)
    path = state_dir / "state.json"
    expected = {
        "site_id": request.site.release_config["site_name"],
        "gpu_cluster_arn": request.gpu_cluster_arn,
        "requested_cluster_id": request.cluster_id or "",
        "allowed_namespaces": sorted(request.allowed_namespaces),
    }
    if path.exists():
        value = cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))
        for key, item in expected.items():
            if value.get(key) != item:
                raise BootstrapError(f"join-cluster state conflicts on {key}")
        return state_dir, path, value
    value = {
        "schema_version": 1,
        **expected,
        "attempt": 1,
        "source_site_sha256": request.site.source_sha256,
        "phase": "STARTED",
        "completed_steps": [],
        "evidence": {},
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json_atomic(path, value)
    return state_dir, path, value


def _done(state: dict[str, Any], step: str) -> bool:
    return step in set(state.get("completed_steps") or [])


def _completed_state_is_current(
    request: JoinClusterRequest,
    state: dict[str, Any],
) -> bool:
    discovered = (state.get("evidence") or {}).get("DISCOVERED") or {}
    target = discovered.get("target") or {}
    cluster_id = str(discovered.get("cluster_id") or "")
    eks_arn = str(target.get("eks_arn") or "")
    return any(
        item.get("cluster_id") == cluster_id and item.get("eks_cluster_arn") == eks_arn
        for item in request.site.release_config["clusters"]
    )


def _reset_completed_state(
    request: JoinClusterRequest,
    *,
    state_dir: Path,
    state_path: Path,
    state: dict[str, Any],
) -> None:
    previous_attempt = int(state.get("attempt") or 1)
    archive = state_dir / f"state.attempt-{previous_attempt:03d}.json"
    if not archive.exists():
        write_json_atomic(archive, dict(state))
    state.clear()
    state.update(
        {
            "schema_version": 1,
            "site_id": request.site.release_config["site_name"],
            "gpu_cluster_arn": request.gpu_cluster_arn,
            "requested_cluster_id": request.cluster_id or "",
            "allowed_namespaces": sorted(request.allowed_namespaces),
            "attempt": previous_attempt + 1,
            "source_site_sha256": request.site.source_sha256,
            "phase": "STARTED",
            "completed_steps": [],
            "evidence": {},
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    write_json_atomic(state_path, state)


def _complete(
    path: Path,
    state: dict[str, Any],
    step: str,
    evidence: dict[str, Any] | None = None,
) -> None:
    completed = set(state.get("completed_steps") or [])
    completed.add(step)
    state["completed_steps"] = sorted(completed)
    state["phase"] = step
    state["updated_at"] = datetime.now(timezone.utc).isoformat()
    if evidence is not None:
        state.setdefault("evidence", {})[step] = evidence
    write_json_atomic(path, state)


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


def _ensure_network(
    runner: CommandRunner,
    *,
    site: RenderedSite,
    target: ClusterIdentity,
) -> dict[str, Any]:
    region = str(site.release_config["aws_region"])
    existing_networks = [
        _cluster_network(
            runner,
            region=region,
            eks_arn=str(item["eks_cluster_arn"]),
        )
        for item in site.release_config["clusters"]
    ]
    existing_eips = {
        value for network in existing_networks for value in network.get("nat_eips", [])
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
        "existing_vpc_ids": sorted({str(item["vpc_id"]) for item in existing_networks}),
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
) -> dict[str, Any]:
    namespaces = tuple(
        sorted(
            {
                *DEFAULT_ALLOWED_NAMESPACES,
                *request.allowed_namespaces,
            }
        )
    )
    return {
        "clusterId": cluster_id,
        "context": target.context,
        "region": target.region,
        "hyperpodClusterName": target.hyperpod_name,
        "eksClusterArn": target.eks_arn,
        "executorIrsaRoleArn": role_arn,
        "allowedNamespaces": list(namespaces),
        "controlPlaneUrl": _control_plane_url(request.site),
        "tokenFile": str(token_file),
        "caFile": str(ca_file),
        "fleetMasterFile": str(fleet_master_file),
    }


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
) -> None:
    path = _bootstrap_state_path(site)
    if path is None:
        return
    value = cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))
    resources = value.setdefault("resources", {})
    resources[f"executor_role:{cluster_id}"] = role
    resources[f"node_keys:{cluster_id}"] = {"cluster_id": cluster_id}
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
    completed.update({f"executor_role:{cluster_id}", f"node_keys:{cluster_id}"})
    value["completed_tasks"] = sorted(completed)
    value.setdefault("removed_clusters", {}).pop(cluster_id, None)
    value.setdefault("joined_clusters", {})[cluster_id] = {
        "joined_at": datetime.now(timezone.utc).isoformat(),
        "vpc_id": network["vpc_id"],
    }
    write_json_atomic(path, value)


def _resource(
    *,
    site_id: str,
    key: str,
    resource_type: str,
    resource_id: str,
    ownership: InstallationResourceOwnership,
    policy: InstallationResourceDeletePolicy,
    region: str,
    account_id: str,
    arn: str | None = None,
    dependencies: list[str] | None = None,
    attributes: dict[str, str] | None = None,
) -> InstallationResource:
    now = datetime.now(timezone.utc)
    return InstallationResource(
        site_id=site_id,
        resource_key=key,
        resource_type=resource_type,
        resource_id=resource_id,
        resource_arn=arn,
        region=region,
        account_id=account_id,
        ownership=ownership,
        delete_policy=policy,
        status=InstallationResourceStatus.ACTIVE,
        dependencies=dependencies or [],
        attributes=attributes or {},
        created_at=now,
        updated_at=now,
    )


def _joined_resources(
    site: RenderedSite,
    *,
    target: ClusterIdentity,
    cluster_id: str,
    role: dict[str, Any],
    network: dict[str, Any],
) -> list[InstallationResource]:
    site_id = str(site.release_config["site_name"])
    region = str(site.release_config["aws_region"])
    account_id = target.account_id
    eks_key = f"cluster/{cluster_id}/eks"
    role_key = f"aws/iam/executor/{cluster_id}/role"
    resources = [
        _resource(
            site_id=site_id,
            key=eks_key,
            resource_type="gpu_eks",
            resource_id=target.eks_name,
            arn=target.eks_arn,
            region=region,
            account_id=account_id,
            ownership=InstallationResourceOwnership.EXTERNAL,
            policy=InstallationResourceDeletePolicy.PRESERVE,
        ),
        _resource(
            site_id=site_id,
            key=f"cluster/{cluster_id}/hyperpod",
            resource_type="gpu_hyperpod",
            resource_id=target.hyperpod_name,
            arn=target.hyperpod_arn,
            region=region,
            account_id=account_id,
            ownership=InstallationResourceOwnership.EXTERNAL,
            policy=InstallationResourceDeletePolicy.PRESERVE,
            dependencies=[eks_key],
        ),
        _resource(
            site_id=site_id,
            key=role_key,
            resource_type="iam_role",
            resource_id=str(role["role_arn"]).rsplit("/", 1)[-1],
            arn=str(role["role_arn"]),
            region=region,
            account_id=account_id,
            ownership=InstallationResourceOwnership.CREATED,
            policy=InstallationResourceDeletePolicy.DELETE,
            attributes={"inline_policy_name": str(role["inline_policy_name"])},
        ),
    ]
    provider_ownership = (
        InstallationResourceOwnership.CREATED
        if role.get("oidc_provider_ownership") == "CREATED"
        else InstallationResourceOwnership.EXTERNAL
    )
    resources.append(
        _resource(
            site_id=site_id,
            key=f"aws/iam/executor/{cluster_id}/oidc-provider",
            resource_type="iam_oidc_provider",
            resource_id=str(role["oidc_provider_arn"]),
            arn=str(role["oidc_provider_arn"]),
            region=region,
            account_id=account_id,
            ownership=provider_ownership,
            policy=(
                InstallationResourceDeletePolicy.DETACH
                if provider_ownership is InstallationResourceOwnership.CREATED
                else InstallationResourceDeletePolicy.PRESERVE
            ),
            attributes={"cluster_name": target.eks_name},
        )
    )
    if network["vpc_id"] not in set(network.get("existing_vpc_ids") or []):
        resources.append(
            _resource(
                site_id=site_id,
                key=f"aws/route53/vpc-association/{cluster_id}",
                resource_type="route53_vpc_association",
                resource_id=(
                    f"{network['hosted_zone_id']}:{region}:{network['vpc_id']}"
                ),
                region=region,
                account_id=account_id,
                ownership=InstallationResourceOwnership.CREATED,
                policy=InstallationResourceDeletePolicy.DETACH,
                dependencies=["aws/route53/zone"],
                attributes={
                    "hosted_zone_id": str(network["hosted_zone_id"]),
                    "vpc_id": str(network["vpc_id"]),
                    "vpc_region": region,
                },
            )
        )
    return resources


def _sync_registry(
    site: RenderedSite,
    *,
    before_path: Path,
    state_dir: Path,
    resources: list[InstallationResource],
) -> InstallationResourceSnapshot:
    before = load_installation_resource_snapshot(before_path)
    merged = {item.resource_key: item for item in before.resources}
    merged.update({item.resource_key: item for item in resources})
    snapshot = InstallationResourceSnapshot(
        site_id=before.site_id,
        resources=sorted(merged.values(), key=lambda item: item.resource_key),
    )
    snapshot = cast(
        InstallationResourceSnapshot,
        snapshot.model_copy(update={"source_sha256": snapshot.digest()}),
    )
    sync_installation_resource_snapshot(site, snapshot)
    write_installation_resource_snapshot(
        site,
        snapshot,
        path=state_dir
        / before_path.name.replace(
            "installation-resources-before",
            "installation-resources-after",
        ),
    )
    return snapshot


def _commit_site(
    request: JoinClusterRequest,
    *,
    state_dir: Path,
    state: dict[str, Any],
    candidate: Path,
    cluster_id: str,
) -> None:
    source = request.site.source
    current = source.read_bytes()
    current_document = yaml.safe_load(current)
    if any(
        item.get("clusterId") == cluster_id
        for item in current_document["spec"].get("clusters", [])
    ):
        return
    digest = hashlib.sha256(current).hexdigest()
    if digest != state["source_site_sha256"]:
        raise BootstrapError("site.yaml changed while join-cluster was running")
    attempt = int(state.get("attempt") or 1)
    backup = state_dir / f"site.before-{attempt:03d}.yaml"
    if not backup.exists():
        backup.write_bytes(current)
        backup.chmod(0o600)
    temporary = source.with_suffix(source.suffix + ".tmp")
    temporary.write_bytes(candidate.read_bytes())
    temporary.chmod(0o600)
    temporary.replace(source)


def _cleanup_candidate(
    candidate: RenderedSite,
    *,
    cluster_id: str,
    state_dir: Path,
    attempt: int,
) -> None:
    errors = []
    with materialized_release_config(candidate) as config:
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
    try:
        _run_rollout(candidate, "remove-cluster", cluster_id=cluster_id)
    except BootstrapError as exc:
        errors.append(f"control-plane unregister: {exc}")
    if errors:
        raise BootstrapError("; ".join(errors))


def _rollback_command(
    arguments: list[str],
    *,
    not_found: tuple[str, ...],
) -> None:
    result = subprocess.run(arguments, text=True, capture_output=True)
    if result.returncode == 0:
        return
    message = (result.stdout or "") + "\n" + (result.stderr or "")
    if any(value in message for value in not_found):
        return
    raise BootstrapError(
        f"rollback command failed ({result.returncode}): "
        f"{' '.join(arguments[:3])}: {result.stderr.strip()}"
    )


def _rollback_network(network: dict[str, Any], site: RenderedSite) -> None:
    region = str(site.release_config["aws_region"])
    for eip in network.get("created_ingress_eips", []):
        _rollback_command(
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
        _rollback_command(
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
    candidate_path = Path(
        str((evidence.get("CANDIDATE_READY") or {}).get("site_file") or "")
    )
    if cluster_id and candidate_path.is_file():
        try:
            candidate = load_site(
                candidate_path,
                repository_root=request.site.repository_root,
            )
            _cleanup_candidate(
                candidate,
                cluster_id=cluster_id,
                state_dir=state_dir,
                attempt=int(state.get("attempt") or 1),
            )
        except Exception as exc:
            errors.append(f"kubernetes/control rollback: {exc}")
    local = evidence.get("LOCAL_INPUTS_READY") or {}
    nodes = list(local.get("nodes") or [])
    target = discovery.get("target") or {}
    try:
        _clear_installer_annotations(request.site, dict(target), nodes)
    except Exception as exc:
        errors.append(f"installer annotation rollback: {exc}")
    try:
        _remove_node_action_keys(request.site, nodes)
    except Exception as exc:
        errors.append(f"node key rollback: {exc}")
    context = str(target.get("context") or "")
    kubeconfig = Path(str(local.get("gpu_kubeconfig") or _gpu_kubeconfig(request.site)))
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
    role = prerequisites.get("executor_role")
    if isinstance(role, dict) and role.get("role_arn"):
        role_name = str(role["role_arn"]).rsplit("/", 1)[-1]
        policy = str(role.get("inline_policy_name") or "")
        if policy:
            try:
                _rollback_command(
                    [
                        "aws",
                        "iam",
                        "delete-role-policy",
                        "--role-name",
                        role_name,
                        "--policy-name",
                        policy,
                    ],
                    not_found=("NoSuchEntity",),
                )
            except BootstrapError as exc:
                errors.append(f"Executor policy rollback: {exc}")
        try:
            _rollback_command(
                ["aws", "iam", "delete-role", "--role-name", role_name],
                not_found=("NoSuchEntity",),
            )
        except BootstrapError as exc:
            errors.append(f"Executor role rollback: {exc}")
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
) -> JoinExecution | dict[str, Any]:
    if not _done(state, "PRECHECKED"):
        _run_rollout(request.site, "preflight")
        _run_rollout(request.site, "verify")
        _complete(state_path, state, "PRECHECKED")
    if not _done(state, "DISCOVERED"):
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
    if not _done(state, "LOCAL_INPUTS_READY"):
        gpu_kubeconfig = _gpu_kubeconfig(request.site)
        gpu_kubeconfig.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
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
        secure = state_dir / "secure"
        secure.mkdir(mode=0o700, parents=True, exist_ok=True)
        token_file = secure / f"{cluster_id}.token"
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


def _deploy_and_commit(
    request: JoinClusterRequest,
    *,
    execution: JoinExecution,
    state_dir: Path,
    state_path: Path,
    state: dict[str, Any],
) -> None:
    cluster_id = execution.cluster_id
    if not _done(state, "JOINED"):
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
    if not _done(state, "VERIFIED"):
        _run_rollout(execution.candidate, "verify")
        _complete(state_path, state, "VERIFIED")
    if not _done(state, "SITE_UPDATED"):
        _commit_site(
            request,
            state_dir=state_dir,
            state=state,
            candidate=execution.candidate.source,
            cluster_id=cluster_id,
        )
        _update_bootstrap_state(
            request.site,
            cluster_id=cluster_id,
            role=dict(execution.prerequisites["executor_role"]),
            network=dict(execution.prerequisites["network"]),
        )
        _complete(state_path, state, "SITE_UPDATED")
    updated_site = load_site(
        request.site.source,
        repository_root=request.site.repository_root,
    )
    if not _done(state, "RELEASE_STATE_UPDATED"):
        _sync_release_state(updated_site)
        _complete(
            state_path,
            state,
            "RELEASE_STATE_UPDATED",
            {
                "cluster_ids": [
                    item["cluster_id"]
                    for item in updated_site.release_config["clusters"]
                ]
            },
        )
    if not _done(state, "REGISTRY_UPDATED"):
        snapshot = _sync_registry(
            updated_site,
            before_path=Path(execution.discovery["registry_snapshot"]),
            state_dir=state_dir,
            resources=_joined_resources(
                updated_site,
                target=execution.target,
                cluster_id=cluster_id,
                role=dict(execution.prerequisites["executor_role"]),
                network=dict(execution.prerequisites["network"]),
            ),
        )
        _complete(
            state_path,
            state,
            "REGISTRY_UPDATED",
            {"snapshot_digest": snapshot.source_sha256},
        )
    if not _done(state, "FINAL_VERIFIED"):
        _run_rollout(updated_site, "verify")
        live = fetch_installation_resource_registry(updated_site)
        required = {
            f"cluster/{cluster_id}/eks",
            f"cluster/{cluster_id}/hyperpod",
            f"aws/iam/executor/{cluster_id}/role",
        }
        active = {
            item.resource_key
            for item in live.resources
            if item.status is InstallationResourceStatus.ACTIVE
        }
        missing = sorted(required - active)
        if missing:
            raise BootstrapError(
                "joined resources are absent from Aurora registry: "
                + ", ".join(missing)
            )
        _complete(state_path, state, "FINAL_VERIFIED")


def _site_contains_cluster(path: Path, cluster_id: str) -> bool:
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    return any(
        item.get("clusterId") == cluster_id
        for item in document["spec"].get("clusters", [])
    )


def join_cluster(
    request: JoinClusterRequest,
    *,
    runner: CommandRunner | None = None,
) -> dict[str, Any]:
    state_dir, state_path, state = _state(request)
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
        cluster_id = (
            execution.cluster_id
            if execution is not None
            else str(
                ((state.get("evidence") or {}).get("DISCOVERED") or {}).get(
                    "cluster_id", ""
                )
            )
        )
        committed = bool(cluster_id) and _site_contains_cluster(
            request.site.source,
            cluster_id,
        )
        state["phase"] = "FAILED_AFTER_COMMIT" if committed else "FAILED"
        state["updated_at"] = datetime.now(timezone.utc).isoformat()
        write_json_atomic(state_path, state)
        if not committed and request.site.release_config.get("auto_rollback", True):
            _rollback(
                request,
                state_dir=state_dir,
                state_path=state_path,
                state=state,
            )
        raise

    assert execution is not None
    state["phase"] = "COMPLETED"
    state["updated_at"] = datetime.now(timezone.utc).isoformat()
    write_json_atomic(state_path, state)
    return {
        "site_id": request.site.release_config["site_name"],
        "cluster_id": execution.cluster_id,
        "gpu_eks_arn": execution.target.eks_arn,
        "gpu_hyperpod_arn": execution.target.hyperpod_arn,
        "phase": "COMPLETED",
        "state_file": str(state_path),
        "site_file": str(request.site.source),
        "cpu_control_plane": "PRESERVED",
        "gpu_cluster": "PRESERVED",
    }

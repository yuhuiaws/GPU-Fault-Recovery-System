from __future__ import annotations

import base64
import json
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, cast

import yaml  # type: ignore[import-untyped,unused-ignore]

from gpu_fault.admin.atomic_json import write_json_atomic
from gpu_fault.admin.aws_cleanup import ResourceCleaner
from gpu_fault.admin.bootstrap_common import (
    Arn,
    BootstrapError,
    CommandRunner,
    safe_name,
)
from gpu_fault.admin.failure_domain_map import apply_failure_domain_map
from gpu_fault.admin.membership_lock import (
    membership_operation_lock,
    reload_site_for_mutation,
)
from gpu_fault.admin.resource_registry import (
    fetch_installation_resource_registry,
    load_installation_resource_snapshot,
    sync_installation_resource_snapshot,
    write_installation_resource_snapshot,
)
from gpu_fault.admin.site import (
    RenderedSite,
    effective_environment,
    load_site,
    materialized_release_config,
)
from gpu_fault.installation_resources import (
    InstallationResource,
    InstallationResourceDeletePolicy,
    InstallationResourceSnapshot,
    InstallationResourceStatus,
)

CONFIRMATION = "REMOVE_GPU_CLUSTER"
INSTALLER_ANNOTATIONS = (
    "gpu-fault.io/installer-version",
    "gpu-fault.io/installer-config-digest",
    "gpu-fault.io/installer-artifact-sha256",
    "gpu-fault.io/installer-node-uid",
    "gpu-fault.io/installer-state",
)


@dataclass(frozen=True)
class RemoveClusterRequest:
    site: RenderedSite
    cluster_id: str
    confirmation: str


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
    """Map the ARN the administrator typed to the managed cluster's id.

    The administrator names GPU clusters by ARN everywhere else (``deploy``,
    ``join-cluster``); the internal ``cluster_id`` is derived at join time and
    never shown as an input. An EKS ARN is matched against the site record
    directly. A HyperPod ARN carries an opaque cluster id, not the name the
    site stores, so it is resolved through ``discover`` (the same AWS lookup
    ``join-cluster`` uses) to its EKS ARN and HyperPod name before matching.
    """

    arn = Arn.parse(gpu_cluster_arn)
    if arn.service not in {"eks", "sagemaker"}:
        raise BootstrapError("GPU cluster ARN must use the eks or sagemaker service")
    clusters = [dict(item) for item in site.release_config["clusters"]]
    eks_arn = gpu_cluster_arn.strip()
    hyperpod_name = ""
    if arn.service == "sagemaker":
        if discover is None:
            raise BootstrapError(
                "a HyperPod ARN needs AWS discovery to resolve its EKS cluster"
            )
        eks_arn, hyperpod_name = discover(gpu_cluster_arn)
    matches = [
        item
        for item in clusters
        if item.get("eks_cluster_arn") == eks_arn
        or (hyperpod_name and item.get("hyperpod_cluster_name") == hyperpod_name)
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


def _state(
    request: RemoveClusterRequest,
) -> tuple[Path, Path, dict[str, Any]]:
    state_dir = (
        request.site.source.parent / "remove-cluster" / safe_name(request.cluster_id)
    )
    state_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    path = state_dir / "state.json"
    if path.exists():
        value = cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))
        expected = {
            "site_id": request.site.release_config["site_name"],
            "cluster_id": request.cluster_id,
        }
        for key, item in expected.items():
            if value.get(key) != item:
                raise BootstrapError(f"remove-cluster state conflicts on {key}")
        return state_dir, path, value
    value = {
        "schema_version": 1,
        "site_id": request.site.release_config["site_name"],
        "cluster_id": request.cluster_id,
        "phase": "STARTED",
        "completed_steps": [],
        "evidence": {},
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json_atomic(path, value)
    return state_dir, path, value


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


def _done(state: dict[str, Any], step: str) -> bool:
    return step in set(state.get("completed_steps") or [])


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


def _json_command(arguments: list[str], *, description: str) -> dict[str, Any]:
    result = subprocess.run(arguments, text=True, capture_output=True)
    if result.returncode:
        raise BootstrapError(f"{description}: {result.stderr.strip()}")
    return cast(dict[str, Any], json.loads(result.stdout or "{}"))


def _cluster_network(
    runner: CommandRunner,
    *,
    region: str,
    eks_arn: str,
) -> dict[str, Any]:
    cluster_name = Arn.parse(eks_arn).resource_name
    cluster = runner.aws_json(
        region,
        "eks",
        "describe-cluster",
        "--name",
        cluster_name,
    )["cluster"]
    vpc_id = str(cluster["resourcesVpcConfig"]["vpcId"])
    gateways = runner.aws_json(
        region,
        "ec2",
        "describe-nat-gateways",
        "--filter",
        f"Name=vpc-id,Values={vpc_id}",
        "Name=state,Values=available",
    ).get("NatGateways", [])
    eips = sorted(
        {
            str(address["PublicIp"])
            for gateway in gateways
            for address in gateway.get("NatGatewayAddresses", [])
            if address.get("PublicIp")
        }
    )
    return {"vpc_id": vpc_id, "nat_eips": eips}


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
                _cluster_network,
                runner,
                region=region,
                eks_arn=eks_arn,
            ): key
            for key, eks_arn in inputs
        }
        for future in as_completed(futures):
            key = futures[future]
            try:
                values[key] = future.result()
            except Exception as exc:
                raise BootstrapError(
                    f"network discovery failed for {key}: {exc}"
                ) from exc
    return (
        values["target"],
        [values[f"remaining:{index}"] for index in range(len(remaining))],
        values["cpu"],
    )


def _target_nodes(
    site: RenderedSite,
    target: dict[str, Any],
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
    return sorted(str(item["metadata"]["name"]) for item in document.get("items", []))


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
        document = json.loads(path.read_text(encoding="utf-8"))
        if (
            document.get("phase") == "CLEANUP_COMPLETED"
            and document.get("status") == "COMPLETED"
        ):
            return path
    path = state_dir / f"kubernetes-cleanup-{len(attempts) + 1:03d}.json"
    with materialized_release_config(request.site) as config:
        completed = subprocess.run(
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
            env={**os.environ, **effective_environment(request.site)},
            check=False,
        )
    if completed.returncode:
        raise BootstrapError(
            f"GPU Kubernetes cleanup failed; evidence retained in {path}"
        )
    return path


def _clear_installer_annotations(
    site: RenderedSite,
    target: dict[str, Any],
    nodes: list[str],
) -> None:
    if not nodes:
        return
    result = subprocess.run(
        [
            *_gpu_kubectl(site, target),
            "annotate",
            "node",
            *nodes,
            "--overwrite",
            *(f"{name}-" for name in INSTALLER_ANNOTATIONS),
        ],
        text=True,
        capture_output=True,
    )
    if result.returncode:
        raise BootstrapError(
            "failed to clear installer annotations: " + result.stderr.strip()
        )


def _request_target_namespace_deletion(
    site: RenderedSite,
    target: dict[str, Any],
) -> None:
    result = subprocess.run(
        [
            *_gpu_kubectl(site, target),
            "delete",
            "namespace",
            str(site.release_config["namespace"]),
            "--ignore-not-found",
            "--wait=false",
        ],
        text=True,
        capture_output=True,
    )
    if result.returncode:
        raise BootstrapError(
            "failed to delete target GPU namespace: " + result.stderr.strip()
        )


def _wait_target_namespace_absent(
    site: RenderedSite,
    target: dict[str, Any],
    *,
    timeout_seconds: float = 600,
    interval_seconds: float = 5,
) -> None:
    namespace = str(site.release_config["namespace"])
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        result = subprocess.run(
            [*_gpu_kubectl(site, target), "get", "namespace", namespace],
            text=True,
            capture_output=True,
        )
        if result.returncode:
            message = (result.stdout or "") + "\n" + (result.stderr or "")
            if "NotFound" in message or "not found" in message.lower():
                return
            raise BootstrapError(
                "cannot verify target GPU namespace deletion: " + result.stderr.strip()
            )
        time.sleep(interval_seconds)
    raise BootstrapError("target GPU namespace deletion exceeded 10 minutes")


def _delete_target_namespace(
    site: RenderedSite,
    target: dict[str, Any],
) -> None:
    _request_target_namespace_deletion(site, target)
    _wait_target_namespace_absent(site, target)


def _remove_node_action_keys(
    site: RenderedSite,
    nodes: list[str],
) -> int:
    if not nodes:
        return 0
    secret = _json_command(
        [
            *_cpu_kubectl(site),
            "-n",
            str(site.release_config["namespace"]),
            "get",
            "secret",
            "gpu-fault-node-action-keys",
            "-o",
            "json",
        ],
        description="cannot read CPU node-action key Secret",
    )
    present = set((secret.get("data") or {}).keys())
    removable = sorted(set(nodes) & present)
    if not removable:
        return 0
    patch = [
        {
            "op": "remove",
            "path": "/data/" + name.replace("~", "~0").replace("/", "~1"),
        }
        for name in removable
    ]
    result = subprocess.run(
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
        text=True,
        capture_output=True,
    )
    if result.returncode:
        raise BootstrapError(
            "failed to remove target node-action keys: " + result.stderr.strip()
        )
    return len(removable)


def _run_control_plane_unregister(
    request: RemoveClusterRequest,
) -> None:
    with materialized_release_config(request.site) as config:
        completed = subprocess.run(
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
            env={**os.environ, **effective_environment(request.site)},
            check=False,
        )
    if completed.returncode:
        raise BootstrapError("control-plane cluster unregister failed")


def _run_control_plane_drain(
    request: RemoveClusterRequest,
) -> None:
    with materialized_release_config(request.site) as config:
        completed = subprocess.run(
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
            env={**os.environ, **effective_environment(request.site)},
            check=False,
        )
    if completed.returncode:
        raise BootstrapError("control-plane cluster drain failed")


def _idempotent_aws(
    arguments: list[str],
    *,
    not_found: tuple[str, ...],
    description: str,
) -> bool:
    result = subprocess.run(arguments, text=True, capture_output=True)
    if result.returncode == 0:
        return True
    message = (result.stdout or "") + "\n" + (result.stderr or "")
    if any(value in message for value in not_found):
        return False
    raise BootstrapError(f"{description}: {result.stderr.strip()}")


def _wait_vpc_association_absent(
    *,
    hosted_zone_id: str,
    region: str,
    vpc_id: str,
    timeout_seconds: float = 300,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        document = _json_command(
            [
                "aws",
                "route53",
                "get-hosted-zone",
                "--id",
                hosted_zone_id,
                "--output",
                "json",
            ],
            description="cannot verify private hosted-zone associations",
        )
        associations = {
            (str(item.get("VPCRegion") or ""), str(item.get("VPCId") or ""))
            for item in document.get("VPCs", [])
        }
        if (region, vpc_id) not in associations:
            return
        time.sleep(5)
    raise BootstrapError(f"Route53 VPC association {region}/{vpc_id} was not removed")


def wait_route53_change_insync(
    change_id: str,
    *,
    timeout_seconds: float = 300,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        document = _json_command(
            [
                "aws",
                "route53",
                "get-change",
                "--id",
                change_id,
                "--output",
                "json",
            ],
            description="cannot read Route53 VPC association change",
        )
        status = str((document.get("ChangeInfo") or {}).get("Status") or "")
        if status == "INSYNC":
            return
        if status not in {"PENDING", ""}:
            raise BootstrapError(
                f"Route53 change {change_id} has unexpected status {status}"
            )
        time.sleep(5)
    raise BootstrapError(f"Route53 change {change_id} did not reach INSYNC")


def disassociate_vpc_from_hosted_zone(
    *,
    hosted_zone_id: str,
    region: str,
    vpc_id: str,
) -> dict[str, Any]:
    result = subprocess.run(
        [
            "aws",
            "route53",
            "disassociate-vpc-from-hosted-zone",
            "--hosted-zone-id",
            hosted_zone_id,
            "--vpc",
            f"VPCRegion={region},VPCId={vpc_id}",
            "--output",
            "json",
        ],
        text=True,
        capture_output=True,
    )
    if result.returncode:
        message = (result.stdout or "") + "\n" + (result.stderr or "")
        if any(
            value in message for value in ("VPCAssociationNotFound", "NoSuchHostedZone")
        ):
            return {
                "changed": False,
                "change_id": None,
                "change_status": None,
            }
        raise BootstrapError(
            "cannot detach GPU VPC from private hosted zone: " + result.stderr.strip()
        )
    try:
        document = json.loads(result.stdout or "{}")
    except json.JSONDecodeError as exc:
        raise BootstrapError(
            "Route53 VPC disassociation returned invalid JSON"
        ) from exc
    change = document.get("ChangeInfo") or {}
    change_id = str(change.get("Id") or "")
    status = str(change.get("Status") or "")
    if not change_id or status not in {"PENDING", "INSYNC"}:
        raise BootstrapError("Route53 VPC disassociation returned invalid ChangeInfo")
    if status != "INSYNC":
        wait_route53_change_insync(change_id)
    return {
        "changed": True,
        "change_id": change_id,
        "change_status": "INSYNC",
    }


def _detach_network(
    request: RemoveClusterRequest,
    *,
    target_network: dict[str, Any],
    remaining_networks: list[dict[str, Any]],
    cpu_vpc_id: str,
) -> dict[str, Any]:
    config = request.site.release_config
    region = str(config["aws_region"])
    remaining_vpcs = {item["vpc_id"] for item in remaining_networks}
    remaining_eips = {
        eip for item in remaining_networks for eip in item.get("nat_eips", [])
    }
    revoked = sorted(set(target_network["nat_eips"]) - remaining_eips)
    if revoked:
        permissions = [
            {
                "IpProtocol": "tcp",
                "FromPort": 443,
                "ToPort": 443,
                "IpRanges": [{"CidrIp": f"{eip}/32"} for eip in revoked],
            }
        ]
        changed = _idempotent_aws(
            [
                "aws",
                "ec2",
                "revoke-security-group-ingress",
                "--region",
                region,
                "--group-id",
                str(config["nlb"]["security_group"]),
                "--ip-permissions",
                json.dumps(permissions, separators=(",", ":")),
            ],
            not_found=("InvalidPermission.NotFound",),
            description="cannot revoke target NLB ingress permissions",
        )
        if not changed:
            revoked = []
    detached_vpc = None
    route53_change_id = None
    route53_change_status = None
    if (
        target_network["vpc_id"] not in remaining_vpcs
        and target_network["vpc_id"] != cpu_vpc_id
        and config.get("dns", {}).get("hosted_zone_id")
    ):
        change = disassociate_vpc_from_hosted_zone(
            hosted_zone_id=str(config["dns"]["hosted_zone_id"]),
            region=region,
            vpc_id=str(target_network["vpc_id"]),
        )
        if change["changed"]:
            detached_vpc = target_network["vpc_id"]
            route53_change_id = change["change_id"]
            route53_change_status = change["change_status"]
            _wait_vpc_association_absent(
                hosted_zone_id=str(config["dns"]["hosted_zone_id"]),
                region=region,
                vpc_id=detached_vpc,
            )
    return {
        "revoked_nat_eips": revoked,
        "detached_vpc_id": detached_vpc,
        "route53_change_id": route53_change_id,
        "route53_change_status": route53_change_status,
    }


def _target_resource(resource: InstallationResource, cluster_id: str) -> bool:
    resource_key = str(resource.resource_key)
    return (
        resource_key.startswith(f"aws/iam/executor/{cluster_id}/")
        or resource_key.startswith(f"cluster/{cluster_id}/")
        or resource_key == f"aws/route53/vpc-association/{cluster_id}"
    )


def _deletion_waves(
    resources: list[InstallationResource],
) -> list[list[InstallationResource]]:
    remaining = {resource.resource_key: resource for resource in resources}
    waves = []
    while remaining:
        dependencies = {
            dependency
            for resource in remaining.values()
            for dependency in resource.dependencies
            if dependency in remaining
        }
        wave = [
            resource
            for key, resource in sorted(remaining.items())
            if key not in dependencies
        ]
        if not wave:
            raise BootstrapError(
                "installation registry target resources contain a dependency cycle"
            )
        waves.append(wave)
        for resource in wave:
            remaining.pop(resource.resource_key)
    return waves


def _remove_target_aws_resources(
    request: RemoveClusterRequest,
    snapshot: InstallationResourceSnapshot,
) -> tuple[InstallationResourceSnapshot, dict[str, Any]]:
    cleaner = ResourceCleaner(request.site)
    selected = [
        resource
        for resource in snapshot.resources
        if _target_resource(resource, request.cluster_id)
    ]
    roles = [resource for resource in selected if resource.resource_type == "iam_role"]
    if len(roles) != 1:
        raise BootstrapError(
            "installation registry must contain exactly one target Executor IAM role"
        )
    cleaner.validate_supported(selected)
    deleted: list[str] = []
    detached: list[str] = []
    delete_resources = [
        resource
        for resource in selected
        if resource.delete_policy is InstallationResourceDeletePolicy.DELETE
    ]
    detached.extend(
        resource.resource_key
        for resource in selected
        if resource.delete_policy is not InstallationResourceDeletePolicy.DELETE
    )
    for wave in _deletion_waves(delete_resources):
        with ThreadPoolExecutor(max_workers=min(4, len(wave))) as executor:
            futures = {
                executor.submit(
                    ResourceCleaner(request.site).delete, resource
                ): resource
                for resource in wave
            }
            for future in as_completed(futures):
                resource = futures[future]
                try:
                    future.result()
                except Exception as exc:
                    raise BootstrapError(
                        f"failed to delete {resource.resource_key}: {exc}"
                    ) from exc
                deleted.append(resource.resource_key)
    now = datetime.now(timezone.utc)
    updated = []
    for resource in snapshot.resources:
        if resource.resource_key in deleted:
            status = InstallationResourceStatus.DELETED
        elif resource.resource_key in detached:
            status = InstallationResourceStatus.DETACHED
        else:
            updated.append(resource)
            continue
        updated.append(
            resource.model_copy(
                update={
                    "status": status,
                    "updated_at": now,
                    "error": None,
                }
            )
        )
    value = InstallationResourceSnapshot(
        site_id=snapshot.site_id,
        resources=updated,
    )
    value = cast(
        InstallationResourceSnapshot,
        value.model_copy(update={"source_sha256": value.digest()}),
    )
    return value, {"deleted": sorted(deleted), "detached": sorted(detached)}


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
        f"node_keys:{request.cluster_id}",
    ):
        completed.discard(name)
        resources.pop(name, None)
    remaining_vpcs = {item["vpc_id"] for item in remaining_networks}
    pki = resources.get("pki") or {}
    pki["vpc_associations"] = [
        item
        for item in pki.get("vpc_associations", [])
        if item.get("vpc_id") != target_network["vpc_id"]
        or target_network["vpc_id"] in remaining_vpcs
    ]
    nlb = resources.get("nlb_network") or {}
    remaining_eips = {
        eip for item in remaining_networks for eip in item.get("nat_eips", [])
    }
    nlb["gpu_nat_eips"] = sorted(remaining_eips)
    resources["pki"] = pki
    resources["nlb_network"] = nlb
    value["completed_tasks"] = sorted(completed)
    value["resources"] = resources
    value.setdefault("removed_clusters", {})[request.cluster_id] = {
        "removed_at": datetime.now(timezone.utc).isoformat(),
        "vpc_id": target_network["vpc_id"],
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
    result = subprocess.run(
        [*_gpu_kubectl(request.site, target), "get", "namespace", namespace],
        text=True,
        capture_output=True,
    )
    if result.returncode == 0:
        raise BootstrapError("target GPU solution namespace still exists")
    message = (result.stdout or "") + "\n" + (result.stderr or "")
    if "NotFound" not in message and "not found" not in message.lower():
        raise BootstrapError(
            "cannot verify target GPU namespace removal: " + result.stderr.strip()
        )


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
    encoded = (registry.get("data") or {}).get("clusters.json", "")
    registrations = json.loads(base64.b64decode(encoded or "W10="))
    if any(item.get("cluster_id") == request.cluster_id for item in registrations):
        raise BootstrapError("target cluster remains in CPU regional registry")


def _verify_preserved_gpu_eks(
    request: RemoveClusterRequest,
    target: dict[str, Any],
) -> None:
    region = str(request.site.release_config["aws_region"])
    eks_name = Arn.parse(str(target["eks_cluster_arn"])).resource_name
    _json_command(
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


def _verify_preserved_hyperpod(
    request: RemoveClusterRequest,
    target: dict[str, Any],
) -> None:
    region = str(request.site.release_config["aws_region"])
    _json_command(
        [
            "aws",
            "sagemaker",
            "describe-cluster",
            "--region",
            region,
            "--cluster-name",
            str(target["hyperpod_cluster_name"]),
            "--output",
            "json",
        ],
        description="preserved GPU HyperPod cluster is missing",
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
        result = subprocess.run(
            [str(rollout), "verify", "--config", str(config)],
            cwd=site.repository_root,
            env={**os.environ, **effective_environment(site)},
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
            executor.submit(function): description for description, function in checks
        }
        failures: list[str] = []
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as exc:
                failures.append(f"{futures[future]}: {exc}")
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
        result = subprocess.run(
            [str(rollout), "sync-state", "--config", str(config)],
            cwd=site.repository_root,
            env={**os.environ, **effective_environment(site)},
            check=False,
        )
    if result.returncode:
        raise BootstrapError("regional release state synchronization failed")


def refresh_failure_domain_map(site: RenderedSite) -> None:
    """Re-render the control-worker's failure-domain map for the remaining clusters."""

    apply_failure_domain_map(site)


def remove_cluster(
    request: RemoveClusterRequest,
    *,
    runner: CommandRunner | None = None,
) -> dict[str, Any]:
    with membership_operation_lock(request.site):
        current = reload_site_for_mutation(request.site)
        return _remove_cluster_locked(replace(request, site=current), runner=runner)


def _remove_cluster_locked(
    request: RemoveClusterRequest,
    *,
    runner: CommandRunner | None = None,
) -> dict[str, Any]:
    if request.confirmation != CONFIRMATION:
        raise BootstrapError(f"remove-cluster requires --confirm {CONFIRMATION}")
    state_dir, state_path, state = _state(request)
    try:
        target, remaining = _target(request.site, request.cluster_id)
    except BootstrapError:
        discovered = (state.get("evidence") or {}).get("DISCOVERED") or {}
        saved_target = discovered.get("target")
        if not isinstance(saved_target, dict):
            raise
        target = dict(saved_target)
        remaining = [dict(item) for item in request.site.release_config["clusters"]]
    active_runner = runner or CommandRunner()
    region = str(request.site.release_config["aws_region"])

    if not _done(state, "DISCOVERED"):
        snapshot = _export_registry(request.site, state_dir)
        target_network, remaining_networks, cpu_network = _parallel_cluster_networks(
            active_runner,
            region=region,
            target=target,
            remaining=remaining,
            cpu_eks_arn=str(request.site.release_config["cpu_eks_arn"]),
        )
        evidence = {
            "target": target,
            "nodes": _target_nodes(request.site, target),
            "target_network": target_network,
            "remaining_networks": remaining_networks,
            "cpu_vpc_id": cpu_network["vpc_id"],
            "registry_snapshot": str(state_dir / "installation-resources-before.json"),
            "registry_digest": snapshot.source_sha256,
        }
        _complete(state_path, state, "DISCOVERED", evidence)
    discovery = state["evidence"]["DISCOVERED"]
    nodes = list(discovery["nodes"])
    target_network = dict(discovery["target_network"])
    remaining_networks = [dict(item) for item in discovery["remaining_networks"]]
    cpu_vpc_id = str(discovery["cpu_vpc_id"])
    snapshot = load_installation_resource_snapshot(Path(discovery["registry_snapshot"]))

    if not _done(state, "CONTROL_REGISTRY_DRAINING"):
        _run_control_plane_drain(request)
        _complete(state_path, state, "CONTROL_REGISTRY_DRAINING")

    if not _done(state, "KUBERNETES_QUIESCED") and not _done(
        state, "KUBERNETES_REMOVED"
    ):
        cleanup_path = _run_kubernetes_cleanup(request, target, state_dir)
        _clear_installer_annotations(request.site, target, nodes)
        _request_target_namespace_deletion(request.site, target)
        _complete(
            state_path,
            state,
            "KUBERNETES_QUIESCED",
            {"cleanup_state": str(cleanup_path), "nodes": nodes},
        )

    def remove_control_registry() -> dict[str, Any]:
        removed_keys = _remove_node_action_keys(request.site, nodes)
        _run_control_plane_unregister(request)
        return {"node_action_keys_removed": removed_keys}

    def detach_aws() -> dict[str, Any]:
        network = _detach_network(
            request,
            target_network=target_network,
            remaining_networks=remaining_networks,
            cpu_vpc_id=cpu_vpc_id,
        )
        updated, resources = _remove_target_aws_resources(request, snapshot)
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
                remove_control_registry
            )
        if not _done(state, "AWS_DETACHED"):
            cleanup_tasks["AWS_DETACHED"] = executor.submit(detach_aws)
        cleanup_failures = []
        for step, future in cleanup_tasks.items():
            try:
                evidence = future.result()
            except Exception as exc:
                cleanup_failures.append(f"{step}: {exc}")
                continue
            _complete(state_path, state, step, evidence)
    if cleanup_failures:
        raise BootstrapError("; ".join(sorted(cleanup_failures)))

    if not _done(state, "KUBERNETES_REMOVED"):
        _wait_target_namespace_absent(request.site, target)
        _complete(
            state_path,
            state,
            "KUBERNETES_REMOVED",
            {"nodes": nodes},
        )

    if not _done(state, "AURORA_UPDATED"):
        updated = load_installation_resource_snapshot(
            state_dir / "installation-resources-after.json"
        )
        sync_installation_resource_snapshot(request.site, updated)
        _complete(
            state_path,
            state,
            "AURORA_UPDATED",
            {"snapshot_digest": updated.source_sha256},
        )

    if not _done(state, "SITE_UPDATED"):
        _write_site_without_cluster(request, state_dir)
        _update_bootstrap_state(
            request,
            target_network=target_network,
            remaining_networks=remaining_networks,
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
    return {
        "site_id": request.site.release_config["site_name"],
        "cluster_id": request.cluster_id,
        "phase": state["phase"],
        "remaining_cluster_ids": [item["cluster_id"] for item in remaining],
        "state_file": str(state_path),
        "cpu_control_plane": "PRESERVED",
        "gpu_cluster": "PRESERVED",
    }

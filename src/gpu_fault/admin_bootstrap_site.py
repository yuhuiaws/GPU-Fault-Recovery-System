from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import replace
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import yaml  # type: ignore[import-untyped]

from gpu_fault.admin_bootstrap_common import (
    Arn,
    BootstrapError,
    BootstrapRequest,
    ClusterIdentity,
    CommandRunner,
    safe_name,
)


@lru_cache(maxsize=32)
def hyperpod_inventory(
    runner: CommandRunner,
    *,
    region: str,
) -> tuple[dict[str, Any], ...]:
    summaries = runner.aws_json(region, "sagemaker", "list-clusters").get(
        "ClusterSummaries",
        [],
    )
    names = [item.get("ClusterName") for item in summaries if item.get("ClusterName")]

    def describe(name: str) -> dict[str, Any]:
        return runner.aws_json(
            region,
            "sagemaker",
            "describe-cluster",
            "--cluster-name",
            name,
        )

    values = []
    with ThreadPoolExecutor(max_workers=min(8, max(1, len(names)))) as executor:
        futures = {executor.submit(describe, name): name for name in names}
        for future in as_completed(futures):
            values.append(future.result())
    return tuple(values)


def cluster_alias(value: str, role: str, index: int) -> str:
    parsed = Arn.parse(value)
    return safe_name(f"gpu-fault-{role}-{index}-{parsed.resource_name}")


def site_identifier(
    cpu: ClusterIdentity,
    _gpu_clusters: Sequence[ClusterIdentity],
) -> str:
    digest = hashlib.sha256(cpu.hyperpod_arn.encode()).hexdigest()[:8]
    return safe_name(f"{cpu.region}-{cpu.hyperpod_name}-{digest}", maximum=48)


def load_existing_site(state_dir: Path) -> dict[str, Any] | None:
    path = state_dir / "site.yaml"
    if not path.is_file():
        return None
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise BootstrapError(f"cannot load existing site.yaml: {exc}") from exc
    if not isinstance(value, dict) or value.get("kind") != "RegionalSite":
        raise BootstrapError("existing site.yaml is not a RegionalSite")
    return value


def load_latest_verified_site_contract(state_dir: Path) -> dict[str, Any] | None:
    release_root = state_dir / "release-deploy"
    if not release_root.is_dir():
        return None
    states = sorted(
        release_root.glob("*/state.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for state_path in states:
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise BootstrapError(
                f"cannot load release state {state_path}: {exc}"
            ) from exc
        if not isinstance(state, dict) or state.get("phase") != "COMPLETED":
            continue
        verification = state.get("verification")
        if not isinstance(verification, dict) or verification.get("status") != "PASSED":
            continue
        candidate_path = state_path.parent / "site.candidate.yaml"
        if not candidate_path.is_file():
            raise BootstrapError(
                f"completed release has no candidate site: {candidate_path}"
            )
        try:
            candidate = yaml.safe_load(candidate_path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as exc:
            raise BootstrapError(
                f"cannot load verified candidate site {candidate_path}: {exc}"
            ) from exc
        if not isinstance(candidate, dict) or candidate.get("kind") != "RegionalSite":
            raise BootstrapError(
                f"verified candidate site is not a RegionalSite: {candidate_path}"
            )
        return candidate
    return None


def recover_verified_site_contract(
    state_dir: Path,
    existing: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if existing is None:
        return None
    verified = load_latest_verified_site_contract(state_dir)
    if verified is None:
        return existing
    existing_spec = existing.get("spec")
    verified_spec = verified.get("spec")
    if not isinstance(existing_spec, dict) or not isinstance(verified_spec, dict):
        raise BootstrapError("site contract has no valid spec")
    identity_fields = (
        (
            (existing.get("metadata") or {}).get("name"),
            (verified.get("metadata") or {}).get("name"),
            "site name",
        ),
        (
            existing_spec.get("awsRegion"),
            verified_spec.get("awsRegion"),
            "AWS Region",
        ),
        (
            (existing_spec.get("cpu") or {}).get("eksArn"),
            (verified_spec.get("cpu") or {}).get("eksArn"),
            "CPU EKS ARN",
        ),
    )
    for current, stable, description in identity_fields:
        if current != stable:
            raise BootstrapError(
                f"verified release candidate {description} differs from site.yaml"
            )

    recovered = deepcopy(existing)
    recovered_spec = recovered["spec"]
    for key in ("release", "runtimeProfile"):
        value = verified_spec.get(key)
        if isinstance(value, dict):
            recovered_spec[key] = deepcopy(value)
    verified_clusters = {
        item.get("eksClusterArn"): item
        for item in verified_spec.get("clusters", [])
        if isinstance(item, dict) and item.get("eksClusterArn")
    }
    for cluster in recovered_spec.get("clusters", []):
        if not isinstance(cluster, dict):
            continue
        stable = verified_clusters.get(cluster.get("eksClusterArn"))
        if not isinstance(stable, dict):
            continue
        for key in (
            "context",
            "allowedNamespaces",
            "agentEndpointAllowedCidrs",
            "tokenFile",
            "caFile",
            "fleetMasterFile",
        ):
            if key in stable:
                cluster[key] = deepcopy(stable[key])
    return recovered


def existing_gpu_context(
    site: Mapping[str, Any] | None,
    cluster: ClusterIdentity,
) -> str | None:
    if not site:
        return None
    spec = site.get("spec")
    if not isinstance(spec, dict):
        return None
    clusters = spec.get("clusters")
    if not isinstance(clusters, list):
        return None
    for item in clusters:
        if not isinstance(item, dict):
            continue
        if (
            item.get("eksClusterArn") == cluster.eks_arn
            or item.get("hyperpodClusterName") == cluster.hyperpod_name
        ):
            context = item.get("context")
            return str(context) if context else None
    return None


def preserve_existing_site_contract(
    generated: dict[str, Any],
    existing: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if not existing:
        return generated
    existing_spec = existing.get("spec")
    generated_spec = generated.get("spec")
    if not isinstance(existing_spec, dict) or not isinstance(generated_spec, dict):
        return generated
    for key in ("release", "runtimeProfile"):
        value = existing_spec.get(key)
        if isinstance(value, dict):
            generated_spec[key] = deepcopy(value)
    existing_clusters = {
        item.get("clusterId"): item
        for item in existing_spec.get("clusters", [])
        if isinstance(item, dict) and item.get("clusterId")
    }
    for item in generated_spec.get("clusters", []):
        if not isinstance(item, dict):
            continue
        previous = existing_clusters.get(item.get("clusterId"))
        if not isinstance(previous, dict):
            continue
        for key in (
            "context",
            "allowedNamespaces",
            "agentEndpointAllowedCidrs",
            "tokenFile",
            "caFile",
            "fleetMasterFile",
        ):
            if key in previous:
                item[key] = deepcopy(previous[key])
    return generated


def validate_existing_cluster_identity(
    site: Mapping[str, Any] | None,
    *,
    cpu: ClusterIdentity,
    gpu_clusters: Sequence[ClusterIdentity],
) -> None:
    if not site:
        return
    spec = site.get("spec")
    if not isinstance(spec, dict):
        raise BootstrapError("existing site has no valid cluster identity")
    cpu_value = spec.get("cpu")
    clusters_value = spec.get("clusters")
    if not isinstance(cpu_value, dict) or not isinstance(clusters_value, list):
        raise BootstrapError("existing site has no valid cluster identity")
    existing_cpu = (
        str(cpu_value.get("eksArn") or ""),
        str(cpu_value.get("hyperpodClusterName") or ""),
    )
    requested_cpu = (cpu.eks_arn, cpu.hyperpod_name)
    existing_gpu = sorted(
        (
            str(item.get("eksClusterArn") or ""),
            str(item.get("hyperpodClusterName") or ""),
        )
        for item in clusters_value
        if isinstance(item, dict)
    )
    requested_gpu = sorted(
        (cluster.eks_arn, cluster.hyperpod_name) for cluster in gpu_clusters
    )
    if (
        existing_cpu != requested_cpu
        or len(existing_gpu) != len(clusters_value)
        or existing_gpu != requested_gpu
    ):
        raise BootstrapError(
            "requested cluster identity differs from the existing site; "
            "use join-cluster or remove-cluster for topology changes"
        )


def discover_bootstrap_scope(
    *,
    request: BootstrapRequest,
    runner: CommandRunner,
    discover: Callable[..., ClusterIdentity],
    alias: Callable[[str, str, int], str],
) -> tuple[dict[str, Any] | None, ClusterIdentity, list[ClusterIdentity]]:
    existing = recover_verified_site_contract(
        request.state_dir,
        load_existing_site(request.state_dir),
    )
    cpu = discover(
        runner,
        cluster_arn=request.cpu_cluster_arn,
        role="cpu",
        context=alias(request.cpu_cluster_arn, "cpu", 0),
    )
    cpu = replace(cpu, context=alias(cpu.eks_arn, "cpu", 0))
    with ThreadPoolExecutor(max_workers=min(8, len(request.gpu_cluster_arns))) as pool:
        futures = [
            pool.submit(
                discover,
                runner,
                cluster_arn=value,
                role="gpu",
                context=alias(value, "gpu", index),
            )
            for index, value in enumerate(request.gpu_cluster_arns, 1)
        ]
        gpu_clusters = [
            replace(
                cluster,
                context=(
                    existing_gpu_context(existing, cluster)
                    or alias(cluster.eks_arn, "gpu", index)
                ),
            )
            for index, cluster in enumerate(
                (future.result() for future in futures),
                1,
            )
        ]
    validate_existing_cluster_identity(
        existing,
        cpu=cpu,
        gpu_clusters=gpu_clusters,
    )
    return existing, cpu, gpu_clusters


def discover_subnet_cidrs(
    runner: CommandRunner,
    *,
    region: str,
    subnet_ids: tuple[str, ...],
) -> tuple[str, ...]:
    if not subnet_ids:
        return ()
    return tuple(
        sorted(
            str(item["CidrBlock"])
            for item in runner.aws_json(
                region,
                "ec2",
                "describe-subnets",
                "--subnet-ids",
                *subnet_ids,
            ).get("Subnets", [])
            if item.get("CidrBlock")
        )
    )

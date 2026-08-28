from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Mapping

import yaml  # type: ignore[import-untyped]

from gpu_fault.admin_bootstrap_common import (
    BootstrapError,
    BootstrapRequest,
    ClusterIdentity,
    CommandRunner,
)


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


def discover_bootstrap_scope(
    *,
    request: BootstrapRequest,
    runner: CommandRunner,
    discover: Callable[..., ClusterIdentity],
    alias: Callable[[str, str, int], str],
) -> tuple[dict[str, Any] | None, ClusterIdentity, list[ClusterIdentity]]:
    existing = load_existing_site(request.state_dir)
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

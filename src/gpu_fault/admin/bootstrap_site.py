from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import replace
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import yaml  # type: ignore[import-untyped,unused-ignore]

from gpu_fault.admin.bootstrap_common import (
    Arn,
    BootstrapError,
    BootstrapRequest,
    BootstrapResult,
    BootstrapState,
    ClusterIdentity,
    CommandRunner,
    safe_name,
)

INITIAL_DEPLOY_TARGET = "initial_deploy_target"


def finalize_bootstrap_site(
    site_file: Path,
    generated_site: dict[str, Any],
    existing_site: dict[str, Any] | None,
    gpu_clusters: Sequence[ClusterIdentity],
    state: Any,
    write_yaml: Callable[[Path, dict[str, Any]], None],
) -> BootstrapResult:
    write_yaml(
        site_file,
        preserve_existing_site_contract(generated_site, existing_site),
    )
    state.record("site_file", str(site_file))
    state.phase("site-ready")
    managed_keys = set(_site_gpu_keys(generated_site))
    return BootstrapResult(
        site_file=site_file,
        state_file=state.path,
        pending_gpu_cluster_arns=tuple(
            cluster.input_arn
            for cluster in gpu_clusters
            if _cluster_key(cluster) not in managed_keys
        ),
    )


def bootstrap_gpu_scope(
    existing_site: dict[str, Any] | None,
    gpu_clusters: Sequence[ClusterIdentity],
) -> list[ClusterIdentity]:
    if existing_site is None:
        return list(gpu_clusters[:1])
    existing = set(_site_gpu_keys(existing_site))
    return [cluster for cluster in gpu_clusters if _cluster_key(cluster) in existing]


def _cluster_key(cluster: ClusterIdentity) -> tuple[str, str]:
    return cluster.eks_arn, cluster.hyperpod_name


def _site_gpu_keys(site: Mapping[str, Any]) -> list[tuple[str, str]]:
    spec = site.get("spec")
    clusters = spec.get("clusters") if isinstance(spec, Mapping) else None
    if not isinstance(clusters, list):
        raise BootstrapError("existing site has no valid cluster identity")
    return [
        (
            str(item.get("eksClusterArn") or ""),
            str(item.get("hyperpodClusterName") or ""),
        )
        for item in clusters
        if isinstance(item, Mapping)
    ]


def _target_identity(
    cpu: ClusterIdentity,
    gpu_clusters: Sequence[ClusterIdentity],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "cpu": {
            "input_arn": cpu.input_arn,
            "eks_arn": cpu.eks_arn,
            "hyperpod_arn": cpu.hyperpod_arn,
            "hyperpod_name": cpu.hyperpod_name,
        },
        "gpu_clusters": [
            {
                "input_arn": cluster.input_arn,
                "eks_arn": cluster.eks_arn,
                "hyperpod_arn": cluster.hyperpod_arn,
                "hyperpod_name": cluster.hyperpod_name,
            }
            for cluster in gpu_clusters
        ],
    }


def bind_initial_deploy_target(
    state: BootstrapState,
    existing_site: Mapping[str, Any] | None,
    cpu: ClusterIdentity,
    gpu_clusters: Sequence[ClusterIdentity],
) -> list[ClusterIdentity]:
    requested = _target_identity(cpu, gpu_clusters)
    requested_keys = [_cluster_key(cluster) for cluster in gpu_clusters]
    if len(requested_keys) != len(set(requested_keys)):
        raise BootstrapError("initial deploy target contains duplicate GPU clusters")
    resources = state.value.get("resources")
    stored = (
        resources.get(INITIAL_DEPLOY_TARGET) if isinstance(resources, Mapping) else None
    )
    status = str(stored.get("status") or "") if isinstance(stored, Mapping) else ""
    if status == "COMPLETE":
        validate_existing_cluster_identity(
            existing_site,
            cpu=cpu,
            gpu_clusters=gpu_clusters,
        )
    else:
        if stored is not None:
            if not isinstance(stored, Mapping):
                raise BootstrapError("initial deploy target checkpoint is invalid")
            stored_identity = {
                key: stored.get(key)
                for key in ("schema_version", "cpu", "gpu_clusters")
            }
            if stored_identity != requested:
                raise BootstrapError(
                    "initial deploy target differs from the persisted checkpoint"
                )
        validate_existing_cluster_identity(
            existing_site,
            cpu=cpu,
            gpu_clusters=gpu_clusters,
            allow_gpu_subset=True,
        )
    existing_keys = set(_site_gpu_keys(existing_site)) if existing_site else set()
    complete = existing_site is not None and existing_keys == set(requested_keys)
    state.record(
        INITIAL_DEPLOY_TARGET,
        {
            **requested,
            "status": "COMPLETE" if complete else "PENDING",
        },
    )
    return bootstrap_gpu_scope(
        dict(existing_site) if existing_site is not None else None,
        gpu_clusters,
    )


def unique_gpu_vpcs(
    cpu: ClusterIdentity,
    gpu_clusters: Sequence[ClusterIdentity],
) -> list[tuple[str, str]]:
    cpu_vpc = (cpu.region, cpu.vpc_id)
    return sorted(
        {
            (cluster.region, cluster.vpc_id)
            for cluster in gpu_clusters
            if (cluster.region, cluster.vpc_id) != cpu_vpc
        }
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


def cluster_irsa_role_entries(
    results: Mapping[str, Any], cluster_id: str
) -> dict[str, str]:
    """The site keys naming one GPU cluster's per-cluster IAM roles.

    ``executorIrsaRoleArn`` is always there. ``adotIrsaRoleArn`` is present only
    when the ``adot_writer_role:<cluster>`` task made a role: its absence is the
    release's "skip the data-plane collector" signal, and a value the operator
    declared in an existing site wins over it (``preserve_existing_site_contract``).
    """

    entries = {
        "executorIrsaRoleArn": str(results[f"executor_role:{cluster_id}"]["role_arn"])
    }
    writer = results.get(f"adot_writer_role:{cluster_id}") or {}
    if writer.get("role_arn"):
        entries["adotIrsaRoleArn"] = str(writer["role_arn"])
    return entries


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
    # ``retention`` is operator-declared (archive-first deletion); regenerating
    # the site must not silently switch it back off.
    for key in ("release", "runtimeProfile", "retention"):
        value = existing_spec.get(key)
        if isinstance(value, dict):
            generated_spec[key] = deepcopy(value)
    # ``failureDomainLabels`` is the operator's declaration of which node labels
    # name a failure domain (EC2 topology fleets); a regenerated site must not
    # fall back to the default priority and silently re-render the map.
    labels = existing_spec.get("failureDomainLabels")
    if isinstance(labels, list) and labels:
        generated_spec["failureDomainLabels"] = list(labels)
    # ``autoRollback`` is a policy the operator owns, not an identity this
    # generator derives. The release engine refuses automatic rollback across
    # non-transactional changes (endpoint, cluster registry, ADOT manifest or
    # image) and tells the operator to fail forward instead, which is expressed
    # by declaring ``autoRollback: false`` in the site. Regenerating it as
    # ``true`` on every deploy would silently revert that decision between the
    # moment it is made and the rollout that needs it, so the declared value
    # wins. A non-boolean is left to the generated default and rejected later by
    # ``SiteSpec``, which reports the offending field.
    declared_rollback = existing_spec.get("autoRollback")
    if isinstance(declared_rollback, bool):
        generated_spec["autoRollback"] = declared_rollback
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
        # ``adotIrsaRoleArn``: a role the operator declared (hand-made before
        # bootstrap created one) wins over the created role; a cluster that
        # never declared one takes the generated value.
        for key in (
            "context",
            "allowedNamespaces",
            "agentEndpointAllowedCidrs",
            "tokenFile",
            "caFile",
            "fleetMasterFile",
            "adotIrsaRoleArn",
        ):
            if key in previous:
                item[key] = deepcopy(previous[key])
    return generated


def validate_existing_cluster_identity(
    site: Mapping[str, Any] | None,
    *,
    cpu: ClusterIdentity,
    gpu_clusters: Sequence[ClusterIdentity],
    allow_gpu_subset: bool = False,
) -> list[ClusterIdentity]:
    """Check the requested clusters against the site; return the GPU delta.

    The requested GPU set may equal the site's or be a strict superset of it:
    the extra clusters are returned so the deploy can join them after the
    release (``deploy --gpu-cluster-arn NEW`` on top of the managed set is how a
    cluster is added; ``join-cluster`` stays as an alias). A subset, or a
    different CPU, is refused: clusters leave a site only through
    ``remove-cluster``. ``allow_gpu_subset`` is kept for the initial-target
    checkpoint path and means the same thing (the site may hold fewer clusters
    than requested).
    """

    if not site:
        return []
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
    existing_gpu = sorted(_site_gpu_keys(site))
    requested_gpu = sorted(_cluster_key(cluster) for cluster in gpu_clusters)
    if len(existing_gpu) != len(clusters_value) or len(existing_gpu) != len(
        set(existing_gpu)
    ):
        raise BootstrapError("existing site has no valid cluster identity")
    if existing_cpu != requested_cpu:
        raise BootstrapError(
            "requested CPU cluster identity differs from the existing site; "
            "the CPU control plane of a site cannot change"
        )
    missing = sorted(set(existing_gpu) - set(requested_gpu))
    if missing:
        raise BootstrapError(
            "requested cluster identity differs from the existing site: the site "
            "manages GPU clusters the command omits ("
            + ", ".join(eks_arn or hyperpod for eks_arn, hyperpod in missing)
            + "); deploy accepts the managed set or a superset of it, use "
            "remove-cluster to detach a cluster"
        )
    existing_keys = set(existing_gpu)
    return [
        cluster
        for cluster in gpu_clusters
        if _cluster_key(cluster) not in existing_keys
    ]


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

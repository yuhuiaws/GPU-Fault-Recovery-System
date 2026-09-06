from __future__ import annotations

import copy
import fcntl
import os
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import TYPE_CHECKING, Any, Iterator, cast

import yaml  # type: ignore[import-untyped,unused-ignore]

from gpu_fault.admin import cluster_join as join
from gpu_fault.admin.bootstrap_common import BootstrapError
from gpu_fault.admin.cluster_join_evidence import (
    final_membership_identity,
    validate_verified_membership,
)
from gpu_fault.admin.cluster_join_state import complete_step, step_done
from gpu_fault.admin.resource_registry import LegacyInstallationRegistryMissing
from gpu_fault.admin.site import RenderedSite, load_site
from gpu_fault.installation_resources import (
    InstallationResource,
    InstallationResourceDeletePolicy,
    InstallationResourceOwnership,
    InstallationResourceSnapshot,
    InstallationResourceStatus,
)

if TYPE_CHECKING:
    from gpu_fault.admin.cluster_join import JoinClusterRequest, JoinExecution


_MEMBERSHIP_THREAD_LOCK = Lock()


@contextmanager
def _membership_lock(site: RenderedSite) -> Iterator[None]:
    path = site.source.parent / ".gpu-fault-membership.lock"
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    with _MEMBERSHIP_THREAD_LOCK, path.open("a+", encoding="utf-8") as descriptor:
        os.chmod(path, 0o600)
        fcntl.flock(descriptor.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(descriptor.fileno(), fcntl.LOCK_UN)


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
    execution: JoinExecution,
) -> list[InstallationResource]:
    target = execution.target
    cluster_id = execution.cluster_id
    role = dict(execution.prerequisites["executor_role"])
    network = dict(execution.prerequisites["network"])
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
    try:
        before = join._fetch_installation_registry(site)
    except LegacyInstallationRegistryMissing:
        before = join._load_installation_snapshot(before_path)
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
    join._sync_installation_snapshot(site, snapshot)
    join._write_installation_snapshot(
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
    candidate_document = yaml.safe_load(candidate.read_text(encoding="utf-8"))
    target_clusters = [
        dict(item)
        for item in candidate_document["spec"].get("clusters", [])
        if item.get("clusterId") == cluster_id
    ]
    if len(target_clusters) != 1:
        raise BootstrapError("candidate site has no unique joined cluster")
    expected = copy.deepcopy(candidate_document)
    observed = copy.deepcopy(current_document)
    for document in (expected, observed):
        document["spec"].pop("clusters", None)
        document["spec"].pop("gpuKubeconfig", None)
    if observed != expected:
        raise BootstrapError(
            "site.yaml non-membership fields changed while join-cluster was running"
        )
    target = target_clusters[0]
    clusters = [dict(item) for item in current_document["spec"].get("clusters", [])]
    existing = [
        item
        for item in clusters
        if item.get("clusterId") == cluster_id
        or item.get("eksClusterArn") == target.get("eksClusterArn")
        or item.get("hyperpodClusterName") == target.get("hyperpodClusterName")
        or item.get("context") == target.get("context")
    ]
    if existing:
        if len(existing) != 1 or existing[0] != target:
            raise BootstrapError("site.yaml contains a conflicting GPU cluster")
    else:
        clusters.append(target)
    current_document["spec"]["clusters"] = sorted(
        clusters,
        key=lambda item: str(item.get("clusterId") or ""),
    )
    current_document["spec"]["gpuKubeconfig"] = candidate_document["spec"][
        "gpuKubeconfig"
    ]
    attempt = int(state.get("attempt") or 1)
    backup = state_dir / f"site.before-{attempt:03d}.yaml"
    if not backup.exists():
        backup.write_bytes(current)
        backup.chmod(0o600)
    _write_site(source, current_document, prefix=f".{source.name}.")


def _write_site(source: Path, document: dict[str, Any], *, prefix: str) -> None:
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=source.parent,
        prefix=prefix,
        suffix=".tmp",
        delete=False,
    ) as temporary:
        yaml.safe_dump(document, temporary, sort_keys=False)
        temporary_path = Path(temporary.name)
    temporary_path.chmod(0o600)
    temporary_path.replace(source)


def _remove_joined_cluster_from_site(
    request: JoinClusterRequest,
    cluster_id: str,
) -> RenderedSite:
    source = request.site.source
    document = yaml.safe_load(source.read_text(encoding="utf-8"))
    document["spec"]["clusters"] = [
        item
        for item in document["spec"].get("clusters", [])
        if item.get("clusterId") != cluster_id
    ]
    _write_site(source, document, prefix=f".{source.name}.rollback.")
    return load_site(source, repository_root=request.site.repository_root)


def _remove_joined_resources(
    site: RenderedSite,
    *,
    execution: JoinExecution,
) -> None:
    try:
        live = join._fetch_installation_registry(site)
    except LegacyInstallationRegistryMissing:
        return
    keys = {item.resource_key for item in _joined_resources(site, execution=execution)}
    snapshot = InstallationResourceSnapshot(
        site_id=live.site_id,
        resources=[item for item in live.resources if item.resource_key not in keys],
    )
    snapshot = cast(
        InstallationResourceSnapshot,
        snapshot.model_copy(update={"source_sha256": snapshot.digest()}),
    )
    join._sync_installation_snapshot(site, snapshot)
    join._write_installation_snapshot(site, snapshot)


def activate_and_commit(
    request: JoinClusterRequest,
    *,
    execution: JoinExecution,
    state_dir: Path,
    state_path: Path,
    state: dict[str, Any],
) -> None:
    cluster_id = execution.cluster_id
    with _membership_lock(request.site):
        activation_started = step_done(state, "ACTIVATION_STARTED")
        verified = (state.get("evidence") or {}).get("VERIFIED")
        if not isinstance(verified, dict):
            raise BootstrapError("join candidate has no verification evidence")
        if not activation_started:
            validate_verified_membership(
                evidence=verified,
                state=state,
                current_site=request.site,
                candidate_site=execution.candidate,
                cluster_id=cluster_id,
            )
        if not step_done(state, "SITE_UPDATED"):
            _commit_site(
                request,
                state_dir=state_dir,
                state=state,
                candidate=execution.candidate.source,
                cluster_id=cluster_id,
            )
            join._update_bootstrap_state(
                request.site,
                cluster_id=cluster_id,
                role=dict(execution.prerequisites["executor_role"]),
                network=dict(execution.prerequisites["network"]),
            )
            complete_step(state_path, state, "SITE_UPDATED")
        updated_site = load_site(
            request.site.source,
            repository_root=request.site.repository_root,
        )
        if not step_done(state, "RELEASE_STATE_UPDATED"):
            join._sync_join_release_state(updated_site)
            complete_step(
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
        if not step_done(state, "REGISTRY_UPDATED"):
            snapshot = _sync_registry(
                updated_site,
                before_path=Path(execution.discovery["registry_snapshot"]),
                state_dir=state_dir,
                resources=_joined_resources(updated_site, execution=execution),
            )
            complete_step(
                state_path,
                state,
                "REGISTRY_UPDATED",
                {"snapshot_digest": snapshot.source_sha256},
            )
        if not activation_started:
            validate_verified_membership(
                evidence=verified,
                state=state,
                current_site=updated_site,
                candidate_site=execution.candidate,
                cluster_id=cluster_id,
            )
            complete_step(
                state_path,
                state,
                "ACTIVATION_STARTED",
                {"cluster_id": cluster_id},
            )
        if not step_done(state, "ACTIVATED"):
            join._run_rollout(
                execution.candidate,
                "activate-cluster",
                cluster_id=cluster_id,
            )
            complete_step(state_path, state, "ACTIVATED")
        if not step_done(state, "FINAL_VERIFIED"):
            live = join._fetch_installation_registry(updated_site)
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
            identity = final_membership_identity(
                updated_site,
                candidate_site_sha256=str(
                    verified.get("candidate_site_sha256")
                    or execution.candidate.source_sha256
                ),
                cluster_id=cluster_id,
                verified_at=str(
                    verified.get("verified_at")
                    or datetime.now(timezone.utc).isoformat()
                ),
            )
            complete_step(
                state_path,
                state,
                "FINAL_VERIFIED",
                {
                    "snapshot_digest": live.source_sha256,
                    **identity,
                },
            )


def rollback_membership(
    request: JoinClusterRequest,
    *,
    execution: JoinExecution,
    joined: bool,
) -> None:
    with _membership_lock(request.site):
        cluster_was_committed = join._site_contains_cluster(
            request.site.source,
            execution.cluster_id,
        )
        restored_site = (
            _remove_joined_cluster_from_site(request, execution.cluster_id)
            if cluster_was_committed
            else load_site(
                request.site.source,
                repository_root=request.site.repository_root,
            )
        )
        if cluster_was_committed:
            join._sync_join_release_state(restored_site)
        _remove_joined_resources(restored_site, execution=execution)
        if joined:
            join._run_rollout(
                execution.candidate,
                "rollback-cluster",
                cluster_id=execution.cluster_id,
            )

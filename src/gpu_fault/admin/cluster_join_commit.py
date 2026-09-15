from __future__ import annotations

import copy
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import yaml  # type: ignore[import-untyped,unused-ignore]

from gpu_fault.admin import cluster_join as join
from gpu_fault.admin.bootstrap_common import BootstrapError
from gpu_fault.admin.cluster_join_evidence import (
    final_membership_identity,
    validate_verified_membership,
)
from gpu_fault.admin.cluster_join_state import complete_step, step_done
from gpu_fault.admin.resource_registry import LegacyInstallationRegistryMissing
from gpu_fault.admin.resource_registry_dns import vpc_association_resource
from gpu_fault.admin.site import RenderedSite, load_site
from gpu_fault.installation_resources import (
    TERMINAL_INSTALLATION_RESOURCE_STATUSES,
    InstallationResource,
    InstallationResourceDeletePolicy,
    InstallationResourceOwnership,
    InstallationResourceSnapshot,
    InstallationResourceStatus,
)

if TYPE_CHECKING:
    from gpu_fault.admin.cluster_join import JoinClusterRequest, JoinExecution


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
    # The data-plane ADOT writer role (present only when the site has an AMP
    # workspace) is registered under the cluster so remove-cluster and
    # uninstall delete it with the executor role. The OIDC provider row stays
    # with the executor: the two roles share one provider.
    adot_writer = execution.prerequisites.get("adot_writer_role")
    if isinstance(adot_writer, dict) and adot_writer.get("role_arn"):
        resources.append(
            _resource(
                site_id=site_id,
                key=f"aws/iam/adot-writer/{cluster_id}/role",
                resource_type="iam_role",
                resource_id=str(adot_writer["role_arn"]).rsplit("/", 1)[-1],
                arn=str(adot_writer["role_arn"]),
                region=region,
                account_id=account_id,
                ownership=InstallationResourceOwnership.CREATED,
                policy=InstallationResourceDeletePolicy.DELETE,
                attributes={
                    "inline_policy_name": str(
                        adot_writer.get("inline_policy_name") or ""
                    )
                },
            )
        )
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
        # The row the registry derives from a bootstrap-created association
        # for this VPC, so a later re-sync upserts this key instead of adding one.
        resources.append(
            vpc_association_resource(
                site_id=site_id,
                owner=cluster_id,
                hosted_zone_id=str(network["hosted_zone_id"]),
                vpc_id=str(network["vpc_id"]),
                vpc_region=region,
                region=region,
                account_id=account_id,
            )
        )
    return resources


def _sealed(
    site_id: str,
    resources: list[InstallationResource],
) -> InstallationResourceSnapshot:
    snapshot = InstallationResourceSnapshot(
        site_id=site_id,
        resources=sorted(resources, key=lambda item: item.resource_key),
    )
    return cast(
        InstallationResourceSnapshot,
        snapshot.model_copy(update={"source_sha256": snapshot.digest()}),
    )


def registry_delta(
    before: InstallationResourceSnapshot,
    resources: list[InstallationResource],
    *,
    site_id: str,
) -> tuple[InstallationResourceSnapshot, InstallationResourceSnapshot]:
    """``(delta, merged)``: what this join adds, and the before snapshot with it.

    The delta is exactly ``_joined_resources``; it is what reaches Aurora, so
    the registry API upserts the joined cluster's rows instead of re-saving
    every row of the site (89 s on 2026-09-12). The consistency check against
    the ``DISCOVERED`` snapshot is cheap and local: same site, and a key the
    snapshot already holds must name the same resource, or the registry is
    describing another cluster's life under this cluster's keys.
    """

    if before.site_id != site_id:
        raise BootstrapError(
            f"registry before-snapshot belongs to site {before.site_id!r}, "
            f"not {site_id!r}"
        )
    existing = {item.resource_key: item for item in before.resources}
    # A row remove-cluster left in a terminal status (DETACHED, DELETED,
    # PRESERVED) is the cluster's previous life, not a claim on the key: a
    # re-join replaces it. A live row must name the same resource; a bootstrap
    # row that recorded no ARN (live 2026-09-13: the HyperPod row) is compared
    # on type and id alone.
    conflicts = sorted(
        item.resource_key
        for item in resources
        if (previous := existing.get(item.resource_key)) is not None
        and previous.status not in TERMINAL_INSTALLATION_RESOURCE_STATUSES
        and (
            (previous.resource_type, previous.resource_id)
            != (item.resource_type, item.resource_id)
            or (
                previous.resource_arn is not None
                and previous.resource_arn != item.resource_arn
            )
        )
    )
    if conflicts:
        raise BootstrapError(
            "registry before-snapshot already holds a different resource under: "
            + ", ".join(conflicts)
        )
    merged = {**existing, **{item.resource_key: item for item in resources}}
    return _sealed(site_id, list(resources)), _sealed(site_id, list(merged.values()))


def _sync_registry(
    site: RenderedSite,
    *,
    before_path: Path,
    state_dir: Path,
    resources: list[InstallationResource],
) -> tuple[InstallationResourceSnapshot, InstallationResourceSnapshot]:
    """Write the join's delta to Aurora; return ``(delta, merged)``.

    Idempotent: a resumed attempt re-sends the same rows (an upsert) and
    rewrites the same ``installation-resources-after`` file. The live registry
    is read once, by ``FINAL_VERIFIED``, which is where the joined rows are
    proven ACTIVE and the site's ``installation-resources.json`` is refreshed.
    """

    before = join._load_installation_snapshot(before_path)
    delta, merged = registry_delta(
        before,
        resources,
        site_id=str(site.release_config["site_name"]),
    )
    join._sync_installation_snapshot(site, delta)
    join._write_installation_snapshot(
        site,
        merged,
        path=state_dir
        / before_path.name.replace(
            "installation-resources-before",
            "installation-resources-after",
        ),
    )
    return delta, merged


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
            adot_writer_role=(
                dict(execution.prerequisites["adot_writer_role"])
                if execution.prerequisites.get("adot_writer_role")
                else None
            ),
        )
        complete_step(state_path, state, "SITE_UPDATED")
    updated_site = load_site(
        request.site.source,
        repository_root=request.site.repository_root,
    )
    if not step_done(state, "RELEASE_STATE_UPDATED"):
        synced = join._sync_join_release_state(updated_site, cluster_id=cluster_id)
        complete_step(
            state_path,
            state,
            "RELEASE_STATE_UPDATED",
            {
                "cluster_ids": [
                    item["cluster_id"]
                    for item in updated_site.release_config["clusters"]
                ],
                **(synced or {}),
            },
        )
    if not step_done(state, "REGISTRY_UPDATED"):
        delta, merged = _sync_registry(
            updated_site,
            before_path=Path(execution.discovery["registry_snapshot"]),
            state_dir=state_dir,
            resources=_joined_resources(updated_site, execution=execution),
        )
        complete_step(
            state_path,
            state,
            "REGISTRY_UPDATED",
            {
                "snapshot_digest": merged.source_sha256,
                "delta_digest": delta.source_sha256,
                "delta_keys": [item.resource_key for item in delta.resources],
                "mode": "delta",
            },
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
        if execution.prerequisites.get("adot_writer_role"):
            required.add(f"aws/iam/adot-writer/{cluster_id}/role")
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
                verified.get("verified_at") or datetime.now(timezone.utc).isoformat()
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

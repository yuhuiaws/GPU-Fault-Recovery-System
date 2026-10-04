"""The Runtime Profile a joining GPU cluster is pinned to, and its approval.

A site has one default profile (``spec.runtimeProfile``). Its ``nodeReboot``
owner is one lifecycle adapter, so a cluster whose provider that adapter does
not serve -- a plain EKS cluster joining a HyperPod site, or the reverse --
gets its own profile: the provider's example profile under its declared
version (``cluster_scope.PROVIDER_PROFILE_SOURCES`` / ``_VERSIONS``), recorded
on the cluster's ``spec.clusters[].runtimeProfile``.

Registering a new profile version is a policy change, so it goes through the
same review as a site profile change: the join writes
``release-deploy/profile-plan.json`` and stops; the operator records the plan
digest on the change request and reruns with ``--approve-profile-plan SHA
--reference REF``; the approved plan is consumed, the immutable snapshot
``profiles/<version>.yaml`` is written and the join proceeds. A later join of
another cluster needing the same version finds the snapshot and asks nothing.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped,unused-ignore]

from gpu_fault.admin.bootstrap_common import (
    BootstrapError,
    ClusterIdentity,
    CommandRunner,
    compute_agent_config_digest,
    profile_node_reboot_owner,
)
from gpu_fault.admin.profile_approval import (
    ProfileApprovalError,
    consume_profile_approval,
    profile_plan_path,
    profile_plan_sha256,
    profile_site_identity_sha256,
    resolve_profile_approval,
    write_profile_plan,
)
from gpu_fault.admin.cluster_join_types import JoinClusterRequest
from gpu_fault.admin.site import RenderedSite
from gpu_fault.cluster_scope import (
    PROVIDER_NODE_REBOOT_OWNERS,
    PROVIDER_PROFILE_SOURCES,
    PROVIDER_PROFILE_VERSIONS,
    cluster_provider,
)
from gpu_fault_release.regional_release_profile_model import (
    runtime_profile_policy_sha256,
)

__all__ = [
    "ClusterProfileOverride",
    "JoinProfileApprovalRequired",
    "cluster_profile_document",
    "cluster_profile_override",
    "cluster_runtime_profile",
    "ensure_cluster_profile_approved",
]

CLUSTER_PROFILE_CHANGE_KIND = "CLUSTER_PROFILE_NEW"
PROFILES_DIR = "profiles"


class JoinProfileApprovalRequired(BootstrapError):
    """The join stopped for Runtime Profile review; the message says how to go on."""


@dataclass(frozen=True)
class ClusterProfileOverride:
    """The profile a joining cluster needs instead of the site's."""

    cluster_id: str
    provider: str
    version: str
    #: Repository-relative example profile the override is rendered from.
    template_reference: str
    template_source: Path
    policy_digest: str
    snapshot_file: Path

    @property
    def owner(self) -> str:
        return PROVIDER_NODE_REBOOT_OWNERS[self.provider]


def cluster_profile_override(
    site: RenderedSite, target: ClusterIdentity, cluster_id: str
) -> ClusterProfileOverride | None:
    """``None`` when the site profile can reboot ``target``'s provider.

    The site profile's OWN/DELEGATE ``nodeReboot`` owner is compared with the
    adapter the provider's executor advertises. A profile with no executable
    ``nodeReboot`` claim owns nobody's reboot and needs no override.
    """

    provider = cluster_provider(target.provider).value
    profile = site.release_config.get("runtime_profile")
    source = profile.get("source") if isinstance(profile, dict) else None
    if not source:
        return None
    owner = profile_node_reboot_owner(Path(str(source)))
    if owner is None or owner == PROVIDER_NODE_REBOOT_OWNERS[provider]:
        return None
    reference = PROVIDER_PROFILE_SOURCES[provider]
    template = site.repository_root / reference
    if not template.is_file():
        raise BootstrapError(f"provider example profile is missing: {template}")
    version = PROVIDER_PROFILE_VERSIONS[provider]
    return ClusterProfileOverride(
        cluster_id=cluster_id,
        provider=provider,
        version=version,
        template_reference=reference,
        template_source=template,
        policy_digest=runtime_profile_policy_sha256(template),
        snapshot_file=site.source.parent / PROFILES_DIR / f"{version}.yaml",
    )


def _snapshot_document(override: ClusterProfileOverride) -> dict[str, Any]:
    document = yaml.safe_load(override.template_source.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise BootstrapError("provider example profile must contain one mapping")
    document["cluster_id"] = override.cluster_id
    document["profile_version"] = override.version
    return document


def _snapshot_text(override: ClusterProfileOverride) -> str:
    return str(yaml.safe_dump(_snapshot_document(override), sort_keys=False))


def _plan_document(
    site: RenderedSite, override: ClusterProfileOverride
) -> dict[str, Any]:
    """The review document, in the shape ``profile_approval`` validates."""

    site_identity = {
        "site_name": str(site.release_config["site_name"]),
        "aws_region": str(site.release_config["aws_region"]),
        "cpu_eks_arn": str(site.release_config["cpu_eks_arn"]),
    }
    site_version = str(site.release_config["runtime_profile"]["version"])
    snapshot = _snapshot_text(override)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "site_identity": site_identity,
        "site_identity_sha256": profile_site_identity_sha256(site_identity),
        "registration_cluster_id": override.cluster_id,
        "template_source": str(override.template_source),
        "active_source": str(site.release_config["runtime_profile"]["source"]),
        "snapshot_file": str(override.snapshot_file),
        "current_version": site_version,
        "desired_version": override.version,
        "current_policy_digest": None,
        "policy_digest": override.policy_digest,
        "live_profile_sha256": None,
        "active_source_sha256": None,
        "source_sha256": hashlib.sha256(
            override.template_source.read_bytes()
        ).hexdigest(),
        "snapshot_sha256": hashlib.sha256(snapshot.encode()).hexdigest(),
        "change_kind": CLUSTER_PROFILE_CHANGE_KIND,
        "changes": [
            f"cluster {override.cluster_id} ({override.provider}) runs its own "
            f"Runtime Profile {override.version}: nodeReboot owned by "
            f"{override.owner}, nodeReplace OBSERVE; the site profile "
            f"{site_version} stays on every other cluster",
        ],
        "approval_required": True,
    }
    payload["plan_sha256"] = profile_plan_sha256(payload)
    return payload


def _stop_message(state_dir: Path, plan: dict[str, Any], *, rerun: str) -> str:
    identity = plan["site_identity"]
    return "\n".join(
        [
            "Runtime Profile policy changed; the join stopped for review.",
            f"Review {profile_plan_path(state_dir)}:",
            "  site_identity:",
            f"    site_name: {identity['site_name']}",
            f"    aws_region: {identity['aws_region']}",
            f"    cpu_eks_arn: {identity['cpu_eks_arn']}",
            f"  cluster: {plan['registration_cluster_id']}",
            f"  version: {plan['current_version']} -> {plan['desired_version']}"
            " (this cluster only)",
            f"  change_kind: {plan['change_kind']}",
            "  changes:",
            *(f"    - {item}" for item in plan["changes"]),
            f"  policy_digest: {plan['policy_digest']}",
            f"  snapshot_sha256: {plan['snapshot_sha256']}",
            f"  plan_sha256: {plan['plan_sha256']}",
            "Record plan_sha256 on the change request; once it is approved, rerun:",
            f"  {rerun} --approve-profile-plan {plan['plan_sha256']} "
            "--reference CHG-<id>",
        ]
    )


def _write_snapshot(override: ClusterProfileOverride) -> None:
    path = override.snapshot_file
    content = _snapshot_text(override)
    if path.exists():
        if path.read_text(encoding="utf-8") != content:
            raise BootstrapError(f"immutable Runtime Profile snapshot differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.parent.chmod(0o700)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.chmod(0o600)
    temporary.replace(path)


def ensure_cluster_profile_approved(
    site: RenderedSite,
    override: ClusterProfileOverride,
    *,
    rerun_command: str,
) -> Path:
    """The approved, immutable snapshot of the cluster's profile.

    An existing snapshot with the same policy content is the proof of an
    earlier approval and is reused. Otherwise the plan is written (or the
    pending one re-read) and the join stops; an approval that matches the
    pending plan exactly is consumed, the snapshot written and its path
    returned.
    """

    state_dir = site.source.parent
    snapshot = override.snapshot_file
    if snapshot.is_file():
        if runtime_profile_policy_sha256(snapshot) != override.policy_digest:
            raise BootstrapError(
                f"approved Runtime Profile snapshot {snapshot} has a different "
                f"policy than {override.template_reference}; choose a new version "
                "or remove the stale snapshot after review"
            )
        return snapshot
    plan = _plan_document(site, override)
    try:
        approval = resolve_profile_approval(state_dir, current_plan=plan)
    except ProfileApprovalError as exc:
        raise BootstrapError(f"Runtime Profile approval is unusable: {exc}") from exc
    if approval is None:
        write_profile_plan(state_dir, plan)
        raise JoinProfileApprovalRequired(
            _stop_message(state_dir, plan, rerun=rerun_command)
        )
    _write_snapshot(override)
    consume_profile_approval(
        state_dir,
        expected_plan_sha256=approval.plan_sha256,
        release_id=f"join-{override.cluster_id}",
        relation=approval.relation,
    )
    return snapshot


def cluster_runtime_profile(
    request: JoinClusterRequest,
    runner: CommandRunner,
    target: ClusterIdentity,
    cluster_id: str,
) -> dict[str, str] | None:
    """The joining cluster's own ``runtimeProfile`` entry, or ``None``.

    Needed only when the site profile cannot reboot the cluster's provider;
    its new version is a reviewed policy change (see the module docstring).
    """

    override = cluster_profile_override(request.site, target, cluster_id)
    if override is None:
        return None
    rerun = (
        f"gpu-fault-admin join-cluster --state-dir {request.site.source.parent} "
        f"--gpu-cluster-arn {request.gpu_cluster_arn}"
    )
    snapshot = ensure_cluster_profile_approved(
        request.site, override, rerun_command=rerun
    )
    return cluster_profile_document(runner, request.site, override, snapshot)


def cluster_profile_document(
    runner: CommandRunner,
    site: RenderedSite,
    override: ClusterProfileOverride,
    snapshot: Path,
) -> dict[str, str]:
    """The ``runtimeProfile`` entry of the cluster's site document."""

    digest = compute_agent_config_digest(
        runner,
        repository_root=site.repository_root,
        runtime_profile_version=override.version,
    )
    return {
        "source": str(snapshot),
        "templateSource": override.template_reference,
        "version": override.version,
        "agentConfigDigest": digest,
    }


def reconcile_cluster_runtime_profiles(
    runner: CommandRunner,
    document: dict[str, Any],
    *,
    repository_root: Path,
    state_dir: Path,
) -> None:
    """Make a deploy's site document carry every per-cluster profile it needs.

    For each GPU cluster whose provider the site profile cannot reboot: keep
    the override the join recorded (its approved snapshot and version) and
    recompute ``agentConfigDigest`` for this release's repository, as the site
    level ``release.agentConfigDigest`` is on every deploy; restore an override
    that went missing from its approved snapshot (live 2026-10-04: a deploy
    regenerated the plain EKS entry without it); refuse when no approved
    snapshot exists -- the review belongs to ``join-cluster``.
    """

    spec = document.get("spec") or {}
    site_profile = (spec.get("runtimeProfile") or {}).get("source")
    owner = profile_node_reboot_owner(Path(str(site_profile))) if site_profile else None
    for entry in spec.get("clusters") or []:
        if not isinstance(entry, dict):
            continue
        provider = cluster_provider(entry.get("provider")).value
        current = entry.get("runtimeProfile")
        needs_override = (
            owner is not None and owner != PROVIDER_NODE_REBOOT_OWNERS[provider]
        )
        if isinstance(current, dict) and current.get("version"):
            version = str(current["version"])
        elif needs_override:
            version = PROVIDER_PROFILE_VERSIONS[provider]
        else:
            continue
        reference = PROVIDER_PROFILE_SOURCES[provider]
        template = repository_root / reference
        snapshot = state_dir / PROFILES_DIR / f"{version}.yaml"
        if not snapshot.is_file():
            cluster_id = entry.get("clusterId")
            raise BootstrapError(
                f"{cluster_id} ({provider}) needs Runtime Profile {version}, which "
                f"has no approved snapshot at {snapshot}; approve it with "
                f"gpu-fault-admin join-cluster --state-dir {state_dir} "
                f"--gpu-cluster-arn {entry.get('eksClusterArn')} "
                "--approve-profile-plan <sha> --reference <CHG>"
            )
        if template.is_file() and runtime_profile_policy_sha256(
            snapshot
        ) != runtime_profile_policy_sha256(template):
            raise BootstrapError(
                f"approved Runtime Profile snapshot {snapshot} has a different policy "
                f"than {reference}; choose a new version or remove the stale snapshot "
                "after review"
            )
        entry["runtimeProfile"] = {
            "source": str(snapshot),
            "templateSource": (
                str(current.get("templateSource"))
                if isinstance(current, dict) and current.get("templateSource")
                else reference
            ),
            "version": version,
            "agentConfigDigest": compute_agent_config_digest(
                runner, repository_root=repository_root, runtime_profile_version=version
            ),
        }

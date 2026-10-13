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

The plan file is one per site, so a plan is one per *version*: it names every
cluster of the operation that needs the version (a batch join, or the members
a deploy moves), anchored on the first by cluster id, so it is the same plan
whichever order lists the clusters -- the deploy preflight and the site
finalization compute it alike, and one approval covers all of them. A batch
join plans exactly as a single join does and stops before it mutates
anything. An approval covers exactly the plan it was given for: when the plan
changed before it was consumed, the approval is archived ``SUPERSEDED`` and
the new plan is written for review.

A snapshot never changes under its version. When the provider's example
profile changes policy after a cluster was pinned (live 2026-10-05: the
plain-EKS profile took ownership of ``nodeReplace`` while ``gpu-eks-cluster``
was pinned to ``eks-v1``), the cluster's profile gets a *new* version,
``regional-<provider>-<digest12>`` of the template policy. ``deploy`` plans it
the same way (``CLUSTER_PROFILE_NEW``), stops before the release work, and
``deploy --approve-profile-plan SHA --reference REF`` writes the new snapshot,
re-pins the cluster entries (version, source, templateSource, Agent config
digest) and continues; the node wave then re-pins the clusters' Agents.
``join-cluster`` derives the same version for a cluster joining after the
change. A member that never got its own version (a batch join before the
batch planned it) is planned by ``deploy`` the same way.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass, replace
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
    StaleProfileApprovalError,
    consume_profile_approval,
    profile_plan_path,
    profile_plan_sha256,
    profile_site_identity_sha256,
    resolve_profile_approval,
    supersede_profile_approval,
    write_profile_plan,
)
from gpu_fault.admin.cluster_join_types import JoinClusterRequest
from gpu_fault.admin.site import RenderedSite
from gpu_fault.cluster_scope import (
    PROVIDER_NODE_REBOOT_OWNERS,
    PROVIDER_PROFILE_SOURCES,
    PROVIDER_PROFILE_VERSIONS,
    PROVIDER_VERSION_PREFIXES,
    cluster_provider,
)
from gpu_fault_release.regional_release_profile_model import (
    runtime_profile_policy_sha256,
)

__all__ = [
    "ClusterProfileOverride",
    "JoinProfileApprovalRequired",
    "SiteFacts",
    "cluster_profile_document",
    "cluster_profile_override",
    "cluster_runtime_profile",
    "cluster_runtime_profiles",
    "ensure_cluster_profile_approved",
    "ensure_cluster_profiles_approved",
    "preflight_cluster_profile_plans",
    "provider_profile_version",
    "reconcile_cluster_runtime_profiles",
]

CLUSTER_PROFILE_CHANGE_KIND = "CLUSTER_PROFILE_NEW"
PROFILES_DIR = "profiles"


class JoinProfileApprovalRequired(BootstrapError):
    """The join stopped for Runtime Profile review; the message says how to go on."""

    def __init__(self, message: str, *, cluster_ids: Sequence[str] = ()) -> None:
        super().__init__(message)
        #: The clusters the pending plan moves to its version.
        self.cluster_ids = tuple(cluster_ids)


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


def _mapping(value: object) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def provider_profile_version(
    provider: str, policy_digest: str, *, state_dir: Path
) -> str:
    """The version the provider's template policy registers under.

    The declared version (``PROVIDER_PROFILE_VERSIONS``) while its snapshot is
    absent or carries this very policy; otherwise -- the declared snapshot is
    immutable and already names another policy -- the digest-named version
    ``regional-<provider>-<digest12>``.
    """

    declared = PROVIDER_PROFILE_VERSIONS[provider]
    snapshot = state_dir / PROFILES_DIR / f"{declared}.yaml"
    if not snapshot.is_file() or runtime_profile_policy_sha256(snapshot) == (
        policy_digest
    ):
        return declared
    return f"{PROVIDER_VERSION_PREFIXES[provider]}-{policy_digest[:12]}"


@dataclass(frozen=True)
class SiteFacts:
    """What a profile plan says about the site, from either site shape."""

    site_name: str
    aws_region: str
    cpu_eks_arn: str
    profile_version: str
    profile_source: str

    @classmethod
    def from_rendered(cls, site: RenderedSite) -> SiteFacts:
        config = site.release_config
        return cls(
            site_name=str(config["site_name"]),
            aws_region=str(config["aws_region"]),
            cpu_eks_arn=str(config["cpu_eks_arn"]),
            profile_version=str(config["runtime_profile"]["version"]),
            profile_source=str(config["runtime_profile"]["source"]),
        )

    @classmethod
    def from_document(cls, document: dict[str, Any]) -> SiteFacts:
        spec = _mapping(document.get("spec"))
        metadata = _mapping(document.get("metadata"))
        cpu = _mapping(spec.get("cpu"))
        profile = _mapping(spec.get("runtimeProfile"))
        facts = cls(
            site_name=str(metadata.get("name") or ""),
            aws_region=str(spec.get("awsRegion") or ""),
            cpu_eks_arn=str(cpu.get("eksArn") or ""),
            profile_version=str(profile.get("version") or ""),
            profile_source=str(profile.get("source") or ""),
        )
        if not (facts.site_name and facts.aws_region and facts.cpu_eks_arn):
            raise BootstrapError(
                "site document lacks the identity a Runtime Profile plan binds "
                "(metadata.name, spec.awsRegion, spec.cpu.eksArn)"
            )
        return facts


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
    policy_digest = runtime_profile_policy_sha256(template)
    version = provider_profile_version(
        provider, policy_digest, state_dir=site.source.parent
    )
    return ClusterProfileOverride(
        cluster_id=cluster_id,
        provider=provider,
        version=version,
        template_reference=reference,
        template_source=template,
        policy_digest=policy_digest,
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


@dataclass(frozen=True)
class _PlannedMember:
    """One cluster a plan moves to the plan's version."""

    cluster_id: str
    provider: str
    #: The cluster's own pinned version and its snapshot's policy when it moves
    #: from that version; ``None`` for its first own version (it runs the site
    #: profile until then).
    current_version: str | None = None
    current_policy_digest: str | None = None


def _change_line(
    facts: SiteFacts,
    override: ClusterProfileOverride,
    member: _PlannedMember,
    *,
    alone: bool,
) -> str:
    before = member.current_version
    if before is not None:
        line = (
            f"cluster {member.cluster_id} ({member.provider}) moves from its own "
            f"Runtime Profile {before} to {override.version}: the policy of "
            f"{override.template_reference} changed; {before} keeps its immutable "
            "snapshot"
        )
        return line + " and no other cluster is touched" if alone else line
    line = (
        f"cluster {member.cluster_id} ({member.provider}) runs its own Runtime "
        f"Profile {override.version}: nodeReboot owned by {override.owner}, "
        "nodeReplace OBSERVE"
    )
    if alone:
        line += (
            f"; the site profile {facts.profile_version} stays on every other cluster"
        )
    return line


def _plan_document(
    facts: SiteFacts,
    override: ClusterProfileOverride,
    members: Sequence[_PlannedMember],
) -> dict[str, Any]:
    """The review document of one version, in the shape ``profile_approval``
    validates.

    ``members`` are every cluster the operation moves to ``override.version``,
    sorted by cluster id; ``override`` is anchored on the first. A first own
    version is reviewed against the site profile, a move from a pinned
    version against that version (``current_policy_digest`` is then the
    pinned snapshot's policy). A plan for one cluster reads exactly as it did
    before plans covered several.
    """

    site_identity = {
        "site_name": facts.site_name,
        "aws_region": facts.aws_region,
        "cpu_eks_arn": facts.cpu_eks_arn,
    }
    befores = sorted(
        {member.current_version or facts.profile_version for member in members}
    )
    digests = {member.current_policy_digest for member in members}
    alone = len(members) == 1
    changes = [_change_line(facts, override, member, alone=alone) for member in members]
    if not alone:
        changes.append(
            f"{override.version} is one approval and one immutable snapshot for "
            f"these {len(members)} clusters; the site profile "
            f"{facts.profile_version} and every cluster this plan does not name "
            "stay as they are"
        )
    snapshot = _snapshot_text(override)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "site_identity": site_identity,
        "site_identity_sha256": profile_site_identity_sha256(site_identity),
        "registration_cluster_id": override.cluster_id,
        "template_source": str(override.template_source),
        "active_source": facts.profile_source,
        "snapshot_file": str(override.snapshot_file),
        "current_version": ", ".join(befores),
        "desired_version": override.version,
        "current_policy_digest": digests.pop() if len(digests) == 1 else None,
        "policy_digest": override.policy_digest,
        "live_profile_sha256": None,
        "active_source_sha256": None,
        "source_sha256": hashlib.sha256(
            override.template_source.read_bytes()
        ).hexdigest(),
        "snapshot_sha256": hashlib.sha256(snapshot.encode()).hexdigest(),
        "change_kind": CLUSTER_PROFILE_CHANGE_KIND,
        "changes": changes,
        "approval_required": True,
    }
    payload["plan_sha256"] = profile_plan_sha256(payload)
    return payload


def _stop_message(
    state_dir: Path,
    plan: dict[str, Any],
    *,
    rerun: str,
    clusters: Sequence[str],
    superseded: bool = False,
) -> str:
    identity = plan["site_identity"]
    verb = "deploy" if rerun.split()[1:2] == ["deploy"] else "join"
    alone = len(clusters) <= 1
    reason = (
        f"Runtime Profile plan changed after its approval; the {verb} stopped for "
        "review and the previous approval was archived as SUPERSEDED."
        if superseded
        else f"Runtime Profile policy changed; the {verb} stopped for review."
    )
    return "\n".join(
        [
            reason,
            f"Review {profile_plan_path(state_dir)}:",
            "  site_identity:",
            f"    site_name: {identity['site_name']}",
            f"    aws_region: {identity['aws_region']}",
            f"    cpu_eks_arn: {identity['cpu_eks_arn']}",
            f"  cluster: {plan['registration_cluster_id']}"
            if alone
            else f"  clusters: {', '.join(clusters)}",
            f"  version: {plan['current_version']} -> {plan['desired_version']}"
            + (" (this cluster only)" if alone else " (these clusters only)"),
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

    return ensure_cluster_profiles_approved(
        site, [override], rerun_command=rerun_command
    )


def ensure_cluster_profiles_approved(
    site: RenderedSite,
    overrides: Sequence[ClusterProfileOverride],
    *,
    rerun_command: str,
) -> Path:
    """The approved snapshot of one version several joining clusters need.

    ``ensure_cluster_profile_approved`` for every cluster of a batch that
    needs the version at once: one plan names them all, anchored on the first
    by cluster id, so the plan is the same whichever order the batch lists
    them in and one approval covers all of them.
    """

    group = sorted(overrides, key=lambda item: item.cluster_id)
    if not group:
        raise BootstrapError("a Runtime Profile plan needs at least one cluster")
    override = group[0]
    shared = (
        override.version,
        override.provider,
        override.policy_digest,
        override.snapshot_file,
    )
    if any(
        (item.version, item.provider, item.policy_digest, item.snapshot_file) != shared
        for item in group
    ):
        raise BootstrapError(
            "clusters planned together must need the same Runtime Profile version"
        )
    snapshot = override.snapshot_file
    if snapshot.is_file():
        if runtime_profile_policy_sha256(snapshot) != override.policy_digest:
            raise BootstrapError(
                f"approved Runtime Profile snapshot {snapshot} has a different "
                f"policy than {override.template_reference}; choose a new version "
                "or remove the stale snapshot after review"
            )
        return snapshot
    plan = _plan_document(
        SiteFacts.from_rendered(site),
        override,
        [_PlannedMember(item.cluster_id, item.provider) for item in group],
    )
    _approve_or_stop(
        site.source.parent,
        override,
        plan,
        rerun_command=rerun_command,
        release_id=f"join-{override.cluster_id}",
        clusters=[item.cluster_id for item in group],
    )
    return snapshot


def _approve_or_stop(
    state_dir: Path,
    override: ClusterProfileOverride,
    plan: dict[str, Any],
    *,
    rerun_command: str,
    release_id: str,
    clusters: Sequence[str],
    write: bool = True,
) -> None:
    """Consume a matching approval (writing the snapshot) or stop for review.

    ``write=False`` is the read-only preflight: a matching approval returns
    without consuming it (the step that writes the snapshot consumes it); no
    approval writes the plan and stops, exactly as the writing path would. An
    approval of another plan -- the clusters needing the version changed, or
    a different plan was approved meanwhile -- covers nothing here: it is
    archived ``SUPERSEDED``, this plan is written and the command stops for a
    new review, as a site profile plan does.
    """

    try:
        approval = resolve_profile_approval(state_dir, current_plan=plan)
    except StaleProfileApprovalError as exc:
        try:
            supersede_profile_approval(
                state_dir, replacement_plan=plan, reason=str(exc)
            )
        except ProfileApprovalError as error:
            raise BootstrapError(
                f"Runtime Profile approval is unusable: {error}"
            ) from error
        raise JoinProfileApprovalRequired(
            _stop_message(
                state_dir,
                plan,
                rerun=rerun_command,
                clusters=clusters,
                superseded=True,
            ),
            cluster_ids=clusters,
        ) from exc
    except ProfileApprovalError as exc:
        raise BootstrapError(f"Runtime Profile approval is unusable: {exc}") from exc
    if approval is None:
        write_profile_plan(state_dir, plan)
        raise JoinProfileApprovalRequired(
            _stop_message(state_dir, plan, rerun=rerun_command, clusters=clusters),
            cluster_ids=clusters,
        )
    if not write:
        return
    _write_snapshot(override)
    consume_profile_approval(
        state_dir,
        expected_plan_sha256=approval.plan_sha256,
        release_id=release_id,
        relation=approval.relation,
    )


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

    rerun = (
        f"gpu-fault-admin join-cluster --state-dir {request.site.source.parent} "
        f"--gpu-cluster-arn {request.gpu_cluster_arn}"
    )
    return cluster_runtime_profiles(
        request.site, runner, [(target, cluster_id)], rerun_command=rerun
    ).get(cluster_id)


def cluster_runtime_profiles(
    site: RenderedSite,
    runner: CommandRunner,
    clusters: Sequence[tuple[ClusterIdentity, str]],
    *,
    rerun_command: str,
) -> dict[str, dict[str, str]]:
    """``{cluster_id: runtimeProfile entry}`` of the joining clusters that need
    their own profile; a cluster the site profile serves is absent.

    ``cluster_runtime_profile`` for the clusters of one batch: a version is
    one plan naming every cluster of the batch that needs it (see
    ``ensure_cluster_profiles_approved``); a pending review raises
    ``JoinProfileApprovalRequired`` naming the clusters its plan covers.
    """

    versions: dict[str, list[ClusterProfileOverride]] = {}
    for target, cluster_id in clusters:
        override = cluster_profile_override(site, target, cluster_id)
        if override is not None:
            versions.setdefault(override.version, []).append(override)
    entries: dict[str, dict[str, str]] = {}
    for version in sorted(versions):
        group = versions[version]
        snapshot = ensure_cluster_profiles_approved(
            site, group, rerun_command=rerun_command
        )
        document = cluster_profile_document(runner, site, group[0], snapshot)
        entries.update({item.cluster_id: dict(document) for item in group})
    return entries


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


def _deploy_rerun(state_dir: Path) -> str:
    return f"gpu-fault-admin deploy --state-dir {state_dir}"


@dataclass(frozen=True)
class _EntryPin:
    """The profile one cluster entry pins once this deploy finalizes the site."""

    entry: dict[str, Any]
    provider: str
    #: The version the entry pins today, or derives when it pins none.
    declared: str
    version: str
    snapshot: Path
    template: Path
    reference: str
    #: Set while ``version`` has no approved snapshot: the plan's member.
    planned: _PlannedMember | None = None


def _entry_pin(
    entry: dict[str, Any],
    *,
    provider: str,
    version: str,
    repository_root: Path,
    state_dir: Path,
) -> _EntryPin:
    """What ``entry`` pins: the pinned version's approved snapshot while its
    policy is the template's; the digest-named successor when the template
    policy changed -- reused when its snapshot is approved, otherwise planned
    (``_PlannedMember`` moving from the pinned version). An entry that pins
    nothing yet (a cluster a batch join completed before the batch planned
    profiles) gets its first own version planned the way a join plans it. A
    pinned version without its snapshot, or a version without a template, is
    refused.
    """

    reference = PROVIDER_PROFILE_SOURCES[provider]
    template = repository_root / reference
    snapshot = state_dir / PROFILES_DIR / f"{version}.yaml"
    cluster_id = str(entry.get("clusterId") or "")
    pin = _EntryPin(entry, provider, version, version, snapshot, template, reference)
    current = entry.get("runtimeProfile")
    pinned = isinstance(current, dict) and bool(current.get("version"))
    if not snapshot.is_file():
        if pinned or not template.is_file():
            raise BootstrapError(
                f"{cluster_id} ({provider}) needs Runtime Profile {version}, which "
                f"has no approved snapshot at {snapshot}; approve it with "
                f"gpu-fault-admin join-cluster --state-dir {state_dir} "
                f"--gpu-cluster-arn {entry.get('eksClusterArn')} "
                "--approve-profile-plan <sha> --reference <CHG>"
            )
        return replace(pin, planned=_PlannedMember(cluster_id, provider))
    if not template.is_file():
        return pin
    pinned_digest = runtime_profile_policy_sha256(snapshot)
    policy_digest = runtime_profile_policy_sha256(template)
    if pinned_digest == policy_digest:
        return pin
    # The template policy moved on under a pinned, immutable snapshot: the
    # cluster gets the digest-named version (live 2026-10-05, gpu-eks-cluster
    # pinned to eks-v1 while the plain-EKS profile took over nodeReplace).
    successor = provider_profile_version(provider, policy_digest, state_dir=state_dir)
    pin = replace(
        pin, version=successor, snapshot=state_dir / PROFILES_DIR / f"{successor}.yaml"
    )
    if pin.snapshot.is_file():
        if runtime_profile_policy_sha256(pin.snapshot) != policy_digest:
            raise BootstrapError(
                f"Runtime Profile snapshot {pin.snapshot} does not carry "
                f"the policy its version names; review it before it is reused"
            )
        return pin
    return replace(
        pin,
        planned=_PlannedMember(
            cluster_id,
            provider,
            current_version=version,
            current_policy_digest=pinned_digest,
        ),
    )


def _approve_planned_versions(
    pins: Sequence[_EntryPin],
    *,
    document: dict[str, Any],
    state_dir: Path,
    write: bool,
) -> None:
    """One ``CLUSTER_PROFILE_NEW`` plan per version without an approved
    snapshot, naming every entry that needs it, anchored on the first by
    cluster id -- the plan the preflight and the finalization compute alike,
    whichever order the document lists the clusters in. Approved, it is
    consumed and the snapshot written (``write``); otherwise the deploy stops
    for review (``_approve_or_stop``)."""

    versions: dict[str, list[_EntryPin]] = {}
    for pin in pins:
        if pin.planned is not None:
            versions.setdefault(pin.version, []).append(pin)
    for version in sorted(versions):
        group = sorted(
            versions[version], key=lambda item: str(item.entry.get("clusterId") or "")
        )
        members = [pin.planned for pin in group if pin.planned is not None]
        first = group[0]
        override = ClusterProfileOverride(
            cluster_id=members[0].cluster_id,
            provider=first.provider,
            version=version,
            template_reference=first.reference,
            template_source=first.template,
            policy_digest=runtime_profile_policy_sha256(first.template),
            snapshot_file=first.snapshot,
        )
        plan = _plan_document(SiteFacts.from_document(document), override, members)
        _approve_or_stop(
            state_dir,
            override,
            plan,
            rerun_command=_deploy_rerun(state_dir),
            release_id=f"deploy-{override.cluster_id}",
            clusters=[member.cluster_id for member in members],
            write=write,
        )


def _entry_pins(
    document: dict[str, Any],
    *,
    repository_root: Path,
    state_dir: Path,
    strict: bool = True,
) -> list[_EntryPin]:
    return [
        _entry_pin(
            entry,
            provider=provider,
            version=version,
            repository_root=repository_root,
            state_dir=state_dir,
        )
        for entry, provider, version in _cluster_entries(
            document,
            repository_root=repository_root,
            state_dir=state_dir,
            strict=strict,
        )
    ]


def _site_reboot_owner(
    document: dict[str, Any], *, repository_root: Path, strict: bool
) -> str | None:
    """The site profile's ``nodeReboot`` owner; a relative source is read
    under ``repository_root``. The read-only preflight (``strict=False``)
    treats an unreadable site profile as unknown and leaves the verdict to the
    site finalization, which reads it with the deploy's full context."""

    spec = _mapping(document.get("spec"))
    source = _mapping(spec.get("runtimeProfile")).get("source")
    if not source:
        return None
    path = Path(str(source))
    if not path.is_absolute() and (repository_root / path).is_file():
        path = repository_root / path
    try:
        return profile_node_reboot_owner(path)
    except BootstrapError:
        if strict:
            raise
        return None


def _cluster_entries(
    document: dict[str, Any],
    *,
    repository_root: Path,
    state_dir: Path,
    strict: bool = True,
) -> list[tuple[dict[str, Any], str, str]]:
    """``(entry, provider, pinned-or-declared version)`` per cluster that needs
    its own profile: one pinned by the join, or one whose provider the site
    profile cannot reboot."""

    spec = _mapping(document.get("spec"))
    owner = _site_reboot_owner(document, repository_root=repository_root, strict=strict)
    # The base a deploy renders follows the registration cluster's provider
    # (bootstrap_discovery.regional_profile_for): judge overrides against that
    # planned base, not against a live base of another family that the same
    # deploy is about to plan back (live 2026-10-08, eks-first site under a
    # HyperPod-family base). The registration cluster itself is never an
    # override: its family is the base.
    registration = str(
        _mapping(spec.get("runtimeProfile")).get("registrationClusterId") or ""
    )
    clusters = [item for item in spec.get("clusters") or [] if isinstance(item, dict)]
    anchor = next(
        (item for item in clusters if str(item.get("clusterId")) == registration), None
    )
    if anchor is not None:
        owner = PROVIDER_NODE_REBOOT_OWNERS[
            cluster_provider(anchor.get("provider")).value
        ]
    rows: list[tuple[dict[str, Any], str, str]] = []
    for entry in clusters:
        provider = cluster_provider(entry.get("provider")).value
        current = entry.get("runtimeProfile")
        if isinstance(current, dict) and current.get("version"):
            rows.append((entry, provider, str(current["version"])))
        elif entry is anchor:
            continue
        elif owner is not None and owner != PROVIDER_NODE_REBOOT_OWNERS[provider]:
            template = repository_root / PROVIDER_PROFILE_SOURCES[provider]
            version = (
                provider_profile_version(
                    provider,
                    runtime_profile_policy_sha256(template),
                    state_dir=state_dir,
                )
                if template.is_file()
                else PROVIDER_PROFILE_VERSIONS[provider]
            )
            rows.append((entry, provider, version))
    return rows


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
    regenerated the plain EKS entry without it). A pinned version whose
    template policy changed is re-pinned to the approved digest-named
    successor; a version without an approved snapshot -- that successor, or
    the first own version of a member that never got one -- is planned once
    for every entry needing it and consumed when approved, or the deploy
    stops for review (``_approve_planned_versions``). No entry is rewritten by
    a stopped deploy.
    """

    pins = _entry_pins(document, repository_root=repository_root, state_dir=state_dir)
    _approve_planned_versions(pins, document=document, state_dir=state_dir, write=True)
    digests: dict[str, str] = {}
    for pin in pins:
        current = pin.entry.get("runtimeProfile")
        if pin.version not in digests:
            digests[pin.version] = compute_agent_config_digest(
                runner,
                repository_root=repository_root,
                runtime_profile_version=pin.version,
            )
        pin.entry["runtimeProfile"] = {
            "source": str(pin.snapshot),
            "templateSource": (
                str(current.get("templateSource"))
                if isinstance(current, dict)
                and current.get("templateSource")
                and pin.version == pin.declared
                else pin.reference
            ),
            "version": pin.version,
            "agentConfigDigest": digests[pin.version],
        }


def preflight_cluster_profile_plans(state_dir: Path, *, repository_root: Path) -> None:
    """Stop a deploy for cluster-profile review before any release work.

    Reads the deployed ``site.yaml`` (nothing else) and judges every cluster
    entry as the site finalization will: a version without an approved
    snapshot writes its ``CLUSTER_PROFILE_NEW`` plan and stops here --
    minutes in, not after the release bundle and its gates (live
    2026-10-05) -- and a matching approval is left for the finalization to
    consume. The plan is the finalization's own (one per version, every entry
    needing it named), so an approval the preflight accepts is the one the
    finalization consumes. A state dir without a site, or a site without such
    clusters, passes silently.
    """

    site_file = state_dir / "site.yaml"
    if not site_file.is_file():
        return
    try:
        document = yaml.safe_load(site_file.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return
    if not isinstance(document, dict):
        return
    pins = _entry_pins(
        document, repository_root=repository_root, state_dir=state_dir, strict=False
    )
    _approve_planned_versions(pins, document=document, state_dir=state_dir, write=False)

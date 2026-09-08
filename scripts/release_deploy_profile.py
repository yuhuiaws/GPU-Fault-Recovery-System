"""The Runtime Profile plan and approval of one release deploy.

Split from ``release_deploy.py``: how the site's Runtime Profile source is
compared with what is live, the change classification and immutable snapshot
that comparison produces, the plan digest an administrator approves, and the
stop message, resolution and archival that carry the approval through the
release.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping

import yaml  # type: ignore[import-untyped,unused-ignore]

from gpu_fault.admin.profile_approval import (
    ProfileApprovalError,
    StaleProfileApprovalError,
    clear_profile_plan,
    consume_profile_approval,
    profile_plan_path,
    profile_plan_sha256,
    profile_site_identity_sha256,
    resolve_profile_approval,
    supersede_profile_approval,
    write_profile_plan,
)
from gpu_fault.admin.site import load_site
from gpu_fault.capabilities import compile_runtime_profile
from gpu_fault.models import CapabilityMode, RuntimeProfile

if __package__:
    from scripts.release_deploy_evidence import (
        PreparedRelease,
        ReleaseDeployError,
        sha256_file,
        update_phase,
    )
else:
    from release_deploy_evidence import (
        PreparedRelease,
        ReleaseDeployError,
        sha256_file,
        update_phase,
    )


@dataclass(frozen=True)
class ProfilePlan:
    site_identity: dict[str, str]
    site_identity_sha256: str
    registration_cluster_id: str
    template_source: Path
    template_reference: str
    active_source: Path
    current_version: str
    desired_version: str
    current_policy_digest: str | None
    policy_digest: str
    live_profile_sha256: str | None
    active_source_sha256: str | None
    source_sha256: str
    snapshot_sha256: str
    change_kind: str
    changes: tuple[str, ...]
    approval: str | None
    approval_plan_sha256: str | None
    approval_relation: str | None
    approval_required: bool
    snapshot_file: Path
    snapshot_document: dict[str, Any]


@dataclass(frozen=True)
class ProfileApprovalResolution:
    profile_plan: ProfilePlan
    plan_sha256: str
    relation: str | None


def _mapping(value: object, description: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ReleaseDeployError(f"{description} must be a mapping")
    return value


def _read_site_document(path: Path) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ReleaseDeployError(f"cannot read site file {path}: {exc}") from exc
    return _mapping(value, "site document")


def _resolve_profile_path(root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else root / path).resolve()


def _profile_definition(
    path: Path,
    *,
    cluster_id: str,
    profile_version: str,
) -> tuple[RuntimeProfile, str, str, dict[str, Any]]:
    if not path.is_file():
        raise ReleaseDeployError(f"Runtime Profile source does not exist: {path}")
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ReleaseDeployError(f"cannot read Runtime Profile {path}: {exc}") from exc
    value = _mapping(document, "Runtime Profile")
    payload = {
        **value,
        "cluster_id": cluster_id,
        "profile_version": profile_version,
    }
    try:
        profile = RuntimeProfile.model_validate(payload)
        effective = compile_runtime_profile(profile)
    except ValueError as exc:
        raise ReleaseDeployError(f"invalid Runtime Profile {path}: {exc}") from exc
    if effective.warnings:
        raise ReleaseDeployError(
            "Runtime Profile has unavailable capabilities: "
            + "; ".join(effective.warnings)
        )
    normalized = profile.model_dump(mode="json")
    policy = {
        "environment": normalized["environment"],
        "claims": sorted(
            normalized["claims"],
            key=lambda item: (
                item["capability"],
                item["owner"],
                item["mode"],
                str(item.get("adapter") or ""),
            ),
        ),
        "observed": sorted(
            normalized["observed"],
            key=lambda item: (
                item["capability"],
                item["owner"],
                str(item.get("version") or ""),
            ),
        ),
    }
    digest = hashlib.sha256(
        json.dumps(policy, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return profile, digest, sha256_file(path), normalized


def _mode_rank(mode: CapabilityMode) -> int:
    return {
        CapabilityMode.DISABLED: 0,
        CapabilityMode.OBSERVE: 1,
        CapabilityMode.AUGMENT: 2,
        CapabilityMode.DELEGATE: 3,
        CapabilityMode.OWN: 3,
    }[mode]


def _classify_profile_change(
    current: RuntimeProfile,
    desired: RuntimeProfile,
) -> tuple[str, tuple[str, ...]]:
    current_claims = {item.capability: item for item in current.claims}
    desired_claims = {item.capability: item for item in desired.claims}
    changes: list[str] = []
    expansive = False
    restrictive = False
    owner_change = False
    if current.environment != desired.environment:
        changes.append(
            f"environment: {current.environment.value}->{desired.environment.value}"
        )
        owner_change = True
    for capability in sorted(
        set(current_claims) | set(desired_claims),
        key=lambda value: value.value,
    ):
        before = current_claims.get(capability)
        after = desired_claims.get(capability)
        before_mode = before.mode if before is not None else CapabilityMode.DISABLED
        after_mode = after.mode if after is not None else CapabilityMode.DISABLED
        if before_mode != after_mode:
            changes.append(
                f"{capability.value}: mode {before_mode.value}->{after_mode.value}"
            )
            expansive |= _mode_rank(after_mode) > _mode_rank(before_mode)
            restrictive |= _mode_rank(after_mode) < _mode_rank(before_mode)
        if (
            before is not None
            and after is not None
            and (before.owner != after.owner or before.adapter != after.adapter)
        ):
            changes.append(
                f"{capability.value}: owner/adapter "
                f"{before.owner}/{before.adapter}->{after.owner}/{after.adapter}"
            )
            owner_change = True

    current_observed = {
        (item.capability, item.owner): item for item in current.observed
    }
    desired_observed = {
        (item.capability, item.owner): item for item in desired.observed
    }
    for identity in sorted(
        set(current_observed) | set(desired_observed),
        key=lambda value: (value[0].value, value[1]),
    ):
        before = current_observed.get(identity)
        after = desired_observed.get(identity)
        if before != after:
            changes.append(
                f"{identity[0].value}/{identity[1]}: observed capability changed"
            )

    if expansive:
        kind = "EXPANSIVE"
    elif owner_change:
        kind = "OWNER_CHANGE"
    elif restrictive:
        kind = "RESTRICTIVE"
    else:
        kind = "IMPLEMENTATION_CHANGE"
    return kind, tuple(changes)


def plan_runtime_profile(
    site_file: Path,
    *,
    live_profile_sha256: str | None,
) -> ProfilePlan:
    site = load_site(site_file)
    document = _read_site_document(site_file)
    spec = _mapping(document.get("spec"), "site spec")
    profile_spec = _mapping(
        spec.get("runtimeProfile"),
        "site spec.runtimeProfile",
    )
    source_reference = str(profile_spec.get("source") or "").strip()
    template_reference = str(
        profile_spec.get("templateSource") or source_reference
    ).strip()
    current_version = str(profile_spec.get("version") or "").strip()
    registration_cluster_id = str(
        profile_spec.get("registrationClusterId") or ""
    ).strip()
    if not all(
        (
            source_reference,
            template_reference,
            current_version,
            registration_cluster_id,
        )
    ):
        raise ReleaseDeployError("Runtime Profile site fields are incomplete")
    active_source = _resolve_profile_path(site.repository_root, source_reference)
    template_source = _resolve_profile_path(site.repository_root, template_reference)
    candidate, policy_digest, source_sha, snapshot_document = _profile_definition(
        template_source,
        cluster_id=registration_cluster_id,
        profile_version=current_version,
    )

    current: RuntimeProfile | None = None
    current_document: dict[str, Any] | None = None
    active_sha = sha256_file(active_source) if active_source.is_file() else None
    active_is_live = live_profile_sha256 is None or active_sha == live_profile_sha256
    if not active_is_live and live_profile_sha256 is not None:
        recovered_source = (
            site_file.parent / "profiles" / f"{current_version}.yaml"
        ).resolve()
        recovered_sha = (
            sha256_file(recovered_source) if recovered_source.is_file() else None
        )
        if recovered_sha == live_profile_sha256:
            active_source = recovered_source
            active_sha = recovered_sha
            active_is_live = True
    if active_source.is_file() and active_is_live:
        current, current_digest, _source_sha, current_document = _profile_definition(
            active_source,
            cluster_id=registration_cluster_id,
            profile_version=current_version,
        )
    else:
        current_digest = None

    if current_digest == policy_digest:
        change_kind = "UNCHANGED"
        changes: tuple[str, ...] = ()
        desired_version = current_version
        if current_document is not None:
            snapshot_document = current_document
    elif current is None:
        change_kind = "UNKNOWN_BASELINE"
        changes = ("active Profile content is unavailable or differs from live SHA",)
        desired_version = f"regional-hyperpod-{policy_digest[:12]}"
    else:
        change_kind, changes = _classify_profile_change(current, candidate)
        desired_version = f"regional-hyperpod-{policy_digest[:12]}"

    approval_required = change_kind != "UNCHANGED"
    snapshot_document["profile_version"] = desired_version
    snapshot_document["cluster_id"] = registration_cluster_id
    snapshot_sha = hashlib.sha256(
        yaml.safe_dump(snapshot_document, sort_keys=False).encode()
    ).hexdigest()
    site_identity = {
        "site_name": str(site.release_config["site_name"]),
        "aws_region": str(site.release_config["aws_region"]),
        "cpu_eks_arn": str(site.release_config["cpu_eks_arn"]),
    }
    site_identity_sha = profile_site_identity_sha256(site_identity)
    return ProfilePlan(
        site_identity=site_identity,
        site_identity_sha256=site_identity_sha,
        registration_cluster_id=registration_cluster_id,
        template_source=template_source,
        template_reference=template_reference,
        active_source=active_source,
        current_version=current_version,
        desired_version=desired_version,
        current_policy_digest=current_digest,
        policy_digest=policy_digest,
        live_profile_sha256=live_profile_sha256,
        active_source_sha256=active_sha,
        source_sha256=source_sha,
        snapshot_sha256=snapshot_sha,
        change_kind=change_kind,
        changes=changes,
        approval=None,
        approval_plan_sha256=None,
        approval_relation=None,
        approval_required=approval_required,
        snapshot_file=site_file.parent / "profiles" / f"{desired_version}.yaml",
        snapshot_document=snapshot_document,
    )


def _profile_plan_payload(plan: ProfilePlan) -> dict[str, Any]:
    payload = {
        "schema_version": 1,
        "site_identity": dict(plan.site_identity),
        "site_identity_sha256": plan.site_identity_sha256,
        "registration_cluster_id": plan.registration_cluster_id,
        "template_source": str(plan.template_source),
        "active_source": str(plan.active_source),
        "snapshot_file": str(plan.snapshot_file),
        "current_version": plan.current_version,
        "desired_version": plan.desired_version,
        "current_policy_digest": plan.current_policy_digest,
        "policy_digest": plan.policy_digest,
        "live_profile_sha256": plan.live_profile_sha256,
        "active_source_sha256": plan.active_source_sha256,
        "source_sha256": plan.source_sha256,
        "snapshot_sha256": plan.snapshot_sha256,
        "change_kind": plan.change_kind,
        "changes": list(plan.changes),
        "approval_required": plan.approval_required,
    }
    payload["plan_sha256"] = profile_plan_sha256(payload)
    return payload


def _write_yaml_atomic(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        yaml.safe_dump(value, sort_keys=False),
        encoding="utf-8",
    )
    temporary.chmod(0o600)
    temporary.replace(path)


def _write_profile_snapshot(plan: ProfilePlan) -> None:
    path = plan.snapshot_file
    path.parent.mkdir(parents=True, exist_ok=True)
    path.parent.chmod(0o700)
    content = yaml.safe_dump(plan.snapshot_document, sort_keys=False)
    if path.exists():
        if path.read_text(encoding="utf-8") != content:
            raise ReleaseDeployError(
                f"immutable Runtime Profile snapshot differs: {path}"
            )
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.chmod(0o600)
    temporary.replace(path)


def profile_approval_stop_message(
    state_dir: Path,
    plan: Mapping[str, Any],
    *,
    reason: str,
) -> str:
    """The text a deploy stops with when its Runtime Profile plan needs review.

    It carries everything the review asks the administrator to check, and the
    exact rerun line with the real digest, so no ``jq`` over
    ``profile-plan.json`` is needed to continue.
    """

    identity = plan.get("site_identity")
    identity = identity if isinstance(identity, Mapping) else {}
    changes = plan.get("changes")
    change_lines = [f"    - {item}" for item in changes] if changes else ["    (none)"]
    digest = str(plan["plan_sha256"])
    lines = [
        reason,
        f"Review {profile_plan_path(state_dir)}:",
        "  site_identity:",
        f"    site_name: {identity.get('site_name')}",
        f"    aws_region: {identity.get('aws_region')}",
        f"    cpu_eks_arn: {identity.get('cpu_eks_arn')}",
        f"  version: {plan.get('current_version')} -> {plan.get('desired_version')}",
        f"  change_kind: {plan.get('change_kind')}",
        "  changes:",
        *change_lines,
        f"  live_profile_sha256: {plan.get('live_profile_sha256')}",
        f"  policy_digest: {plan.get('policy_digest')}",
        f"  snapshot_sha256: {plan.get('snapshot_sha256')}",
        f"  plan_sha256: {digest}",
        "Record plan_sha256 on the change request; once it is approved, rerun:",
        f"  gpu-fault-admin deploy --state-dir {state_dir} "
        f"--approve-profile-plan {digest} --reference CHG-<id>",
    ]
    return "\n".join(lines)


def _resolve_profile_approval_state(
    site_file: Path,
    profile_plan: ProfilePlan,
) -> ProfileApprovalResolution:
    profile_plan_payload = _profile_plan_payload(profile_plan)
    profile_plan_digest = str(profile_plan_payload["plan_sha256"])
    try:
        resolved = resolve_profile_approval(
            site_file.parent,
            current_plan=profile_plan_payload,
        )
    except StaleProfileApprovalError as exc:
        supersede_profile_approval(
            site_file.parent,
            replacement_plan=(
                profile_plan_payload if profile_plan.approval_required else None
            ),
            reason=str(exc),
        )
        if profile_plan.approval_required:
            raise ReleaseDeployError(
                profile_approval_stop_message(
                    site_file.parent,
                    profile_plan_payload,
                    reason=(
                        "Runtime Profile plan or live baseline changed; the "
                        "previous approval was archived as SUPERSEDED."
                    ),
                )
            ) from exc
        resolved = None
    if resolved is not None:
        return ProfileApprovalResolution(
            profile_plan=replace(
                profile_plan,
                approval=resolved.reference,
                approval_plan_sha256=resolved.plan_sha256,
                approval_relation=resolved.relation,
            ),
            plan_sha256=resolved.plan_sha256,
            relation=resolved.relation,
        )
    if profile_plan.approval_required:
        write_profile_plan(site_file.parent, profile_plan_payload)
        raise ReleaseDeployError(
            profile_approval_stop_message(
                site_file.parent,
                profile_plan_payload,
                reason="Runtime Profile policy changed; the deploy stopped for review.",
            )
        )
    clear_profile_plan(site_file.parent)
    return ProfileApprovalResolution(
        profile_plan=profile_plan,
        plan_sha256=profile_plan_digest,
        relation=None,
    )


def _consume_profile_approval_state(
    site_file: Path,
    prepared: PreparedRelease,
    resolution: ProfileApprovalResolution,
) -> None:
    if resolution.relation is None:
        return
    try:
        archive = consume_profile_approval(
            site_file.parent,
            expected_plan_sha256=resolution.plan_sha256,
            release_id=prepared.release_id,
            relation=resolution.relation,
        )
    except ProfileApprovalError as exc:
        update_phase(
            prepared,
            "COMPLETED",
            profile_approval_audit={
                "status": "FAILED",
                "error": str(exc),
            },
        )
        raise ReleaseDeployError(
            "release completed but Profile approval archival failed: " + str(exc)
        ) from exc
    update_phase(
        prepared,
        "COMPLETED",
        profile_approval_audit={
            "status": "CONSUMED",
            "plan_sha256": resolution.plan_sha256,
            "relation": resolution.relation,
            "archive": str(archive),
        },
    )

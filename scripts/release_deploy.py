from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml  # type: ignore[import-untyped,unused-ignore]

from gpu_fault.admin.atomic_json import write_json_atomic
from gpu_fault.admin.bootstrap_common import (
    CommandRunner,
    compute_agent_config_digest,
)
from gpu_fault.admin.operation_lock import inherited_lock_pass_fds
from gpu_fault.admin.profile_approval import (
    ProfileApprovalError,
    StaleProfileApprovalError,
    clear_profile_plan,
    consume_profile_approval,
    profile_approval_lock,
    profile_plan_sha256,
    profile_site_identity_sha256,
    resolve_profile_approval,
    supersede_profile_approval,
    write_profile_plan,
)
from gpu_fault.admin.site import (
    effective_environment,
    load_site,
    materialized_release_config,
)
from gpu_fault.capabilities import compile_runtime_profile
from gpu_fault.digests import SHA256_PATTERN
from gpu_fault.models import CapabilityMode, RuntimeProfile

if __package__:
    from scripts.release_attestation import verify_attestation
    from scripts.release_deploy_evidence import (
        PreparedRelease,
        ReleaseDeployError,
        is_clean_noop_diff,
        is_pending_commit_diff,
        release_manifest,
        release_summary_warnings,
        sha256_file,
        update_phase,
        utc_now,
        validate_stability_report,
        validate_verification_report,
        verification_metadata,
    )
    from scripts.release_failure_recovery import (
        recover_release_failure,
    )
    from scripts.release_live_state import (
        ReleaseStateNotFound as LiveReleaseStateNotFound,
    )
    from scripts.release_live_state import (
        ReleaseStateReadError as LiveReleaseStateReadError,
    )
    from scripts.release_live_state import (
        read_live_release_state as _read_live_release_state,
    )
else:
    from release_attestation import verify_attestation
    from release_deploy_evidence import (
        PreparedRelease,
        ReleaseDeployError,
        is_clean_noop_diff,
        is_pending_commit_diff,
        release_manifest,
        release_summary_warnings,
        sha256_file,
        update_phase,
        utc_now,
        validate_stability_report,
        validate_verification_report,
        verification_metadata,
    )
    from release_failure_recovery import (
        recover_release_failure,
    )
    from release_live_state import ReleaseStateNotFound as LiveReleaseStateNotFound
    from release_live_state import ReleaseStateReadError as LiveReleaseStateReadError
    from release_live_state import read_live_release_state as _read_live_release_state


ROOT = Path(__file__).resolve().parents[1]
SITE_ENV = "GPU_FAULT_SITE_FILE"
VERIFICATION_REPORT = "verification-report.json"
STABILITY_REPORT = "stability-report.json"
RELEASE_SUMMARY_REPORT = "release-summary.json"
EXPECTED_STATE_SHA256_ENV = "GPU_FAULT_EXPECTED_RELEASE_STATE_SHA256"
QUICK_VALIDATION_EVIDENCE_ENV = "GPU_FAULT_QUICK_VALIDATION_EVIDENCE"


class ReleaseStateNotFound(ReleaseDeployError):
    pass


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


def resolve_site_file(value: Path | None, environment: Mapping[str, str]) -> Path:
    raw = str(value) if value is not None else environment.get(SITE_ENV, "")
    if not raw.strip():
        raise ReleaseDeployError(
            f"provide --site or set {SITE_ENV}; the release command never guesses a site"
        )
    path = Path(raw).expanduser().resolve()
    if not path.is_file():
        raise ReleaseDeployError(f"site file does not exist: {path}")
    if path.stat().st_mode & 0o077:
        raise ReleaseDeployError(
            f"site file must not grant group/other permissions: {path}"
        )
    return path


def repository_root_from_site(path: Path) -> Path:
    document = _read_site_document(path)
    spec = _mapping(document.get("spec"), "site spec")
    raw = str(spec.get("repositoryRoot") or "").strip()
    if not raw:
        raise ReleaseDeployError("site spec.repositoryRoot is required")
    configured = Path(raw).expanduser()
    root = configured if configured.is_absolute() else path.parent / configured
    root = root.resolve()
    if root != ROOT:
        raise ReleaseDeployError(
            f"site repositoryRoot is {root}, but this release command belongs to {ROOT}"
        )
    return root


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


def read_live_release_state(
    site_file: Path,
    *,
    runner=subprocess.run,
) -> dict[str, Any]:
    try:
        return _read_live_release_state(site_file, runner=runner)
    except LiveReleaseStateNotFound as exc:
        raise ReleaseStateNotFound(str(exc)) from exc
    except LiveReleaseStateReadError as exc:
        raise ReleaseDeployError(str(exc)) from exc


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


def prepare_site_release(
    site_file: Path,
    *,
    profile_plan: ProfilePlan,
    digest_runner: CommandRunner | None = None,
) -> PreparedRelease:
    rendered = load_site(site_file)
    if rendered.repository_root != ROOT:
        raise ReleaseDeployError(
            "site repositoryRoot changed during release preparation"
        )
    manifest, release_id = release_manifest(rendered.repository_root)
    document = _read_site_document(site_file)
    spec = _mapping(document.get("spec"), "site spec")
    release = _mapping(spec.get("release"), "site spec.release")
    profile = _mapping(spec.get("runtimeProfile"), "site spec.runtimeProfile")
    images = _mapping(spec.setdefault("images", {}), "site spec.images")
    if profile_plan.approval_required and not profile_plan.approval:
        raise ReleaseDeployError(
            "Runtime Profile changed without a matching state approval"
        )
    desired_profile = profile_plan.desired_version
    digest = compute_agent_config_digest(
        digest_runner or CommandRunner(),
        repository_root=rendered.repository_root,
        runtime_profile_version=desired_profile,
    )
    if not SHA256_PATTERN.fullmatch(digest):
        raise ReleaseDeployError("computed Agent config digest is invalid")

    state_dir = site_file.parent / "release-deploy" / release_id
    state_dir.mkdir(parents=True, exist_ok=True)
    state_dir.chmod(0o700)
    backup = state_dir / "site.before.yaml"
    if not backup.exists():
        backup.write_bytes(site_file.read_bytes())
        backup.chmod(0o600)

    previous = {
        "manifest": release.get("manifest"),
        "agent_config_digest": release.get("agentConfigDigest"),
        "runtime_profile_source": profile.get("source"),
        "runtime_profile_template_source": profile.get("templateSource"),
        "runtime_profile_version": profile.get("version"),
        "images": dict(images),
    }
    _write_profile_snapshot(profile_plan)
    release["manifest"] = "dist/current-release.json"
    release["agentConfigDigest"] = digest
    profile["source"] = str(profile_plan.snapshot_file)
    profile["templateSource"] = profile_plan.template_reference
    profile["version"] = desired_profile
    if int(manifest.get("schema_version", 0)) >= 3:
        release_images = dict((manifest.get("delivery") or {}).get("images") or {})
        image_fields = {
            "runtime": "runtime",
            "nodeInstaller": "node_installer",
            "dcgmExporter": "dcgm_exporter",
            "adot": "adot",
        }
        for site_field, release_field in image_fields.items():
            reference = str(
                (release_images.get(release_field) or {}).get("reference") or ""
            )
            if not reference:
                raise ReleaseDeployError(
                    f"release manifest has no {release_field} image"
                )
            images[site_field] = reference
    candidate = state_dir / "site.candidate.yaml"
    _write_yaml_atomic(candidate, document)
    load_site(candidate, repository_root=rendered.repository_root)

    changed = previous != {
        "manifest": release["manifest"],
        "agent_config_digest": digest,
        "runtime_profile_source": profile["source"],
        "runtime_profile_template_source": profile["templateSource"],
        "runtime_profile_version": desired_profile,
        "images": dict(images),
    }
    if changed:
        _write_yaml_atomic(site_file, document)
    plan = {
        "schema_version": 1,
        "release_id": release_id,
        "site_file": str(site_file),
        "site_changed": changed,
        "previous": previous,
        "desired": {
            "manifest": release["manifest"],
            "agent_config_digest": digest,
            "runtime_profile_source": profile["source"],
            "runtime_profile_template_source": profile["templateSource"],
            "runtime_profile_version": desired_profile,
            "images": dict(images),
        },
        "profile_change": {
            "kind": profile_plan.change_kind,
            "changes": list(profile_plan.changes),
            "policy_digest": profile_plan.policy_digest,
            "source_sha256": profile_plan.source_sha256,
            "approval": profile_plan.approval,
            "approval_plan_sha256": profile_plan.approval_plan_sha256,
            "approval_relation": profile_plan.approval_relation,
        },
    }
    write_json_atomic(state_dir / "plan.json", plan)
    write_json_atomic(
        state_dir / "state.json",
        {
            **plan,
            "phase": "PREPARED",
        },
    )
    return PreparedRelease(
        site_file=site_file,
        release_id=release_id,
        runtime_profile_version=desired_profile,
        agent_config_digest=digest,
        profile_change_kind=profile_plan.change_kind,
        profile_approval=profile_plan.approval,
        state_dir=state_dir,
        site_changed=changed,
    )


def _run(
    arguments: Sequence[str], *, cwd: Path, environment: Mapping[str, str]
) -> None:
    print("+ " + " ".join(arguments), file=sys.stderr, flush=True)
    completed = subprocess.run(
        list(arguments),
        cwd=cwd,
        env=dict(environment),
        check=False,
        pass_fds=inherited_lock_pass_fds(),
    )
    if completed.returncode:
        raise ReleaseDeployError(
            f"command failed ({completed.returncode}): {arguments[0]}"
        )


def _run_json(
    arguments: Sequence[str], *, cwd: Path, environment: Mapping[str, str]
) -> dict[str, Any]:
    print("+ " + " ".join(arguments), file=sys.stderr, flush=True)
    completed = subprocess.run(
        list(arguments),
        cwd=cwd,
        env=dict(environment),
        check=False,
        stdout=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        pass_fds=inherited_lock_pass_fds(),
    )
    output = completed.stdout or ""
    if output:
        print(output, end="" if output.endswith("\n") else "\n", flush=True)
    if completed.returncode:
        raise ReleaseDeployError(
            f"command failed ({completed.returncode}): {arguments[0]}"
        )
    try:
        value = json.loads(output)
    except json.JSONDecodeError as exc:
        raise ReleaseDeployError(
            f"command returned invalid JSON: {arguments[0]}: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise ReleaseDeployError(
            f"command returned a non-object JSON document: {arguments[0]}"
        )
    return value


def _collect_release_summary(
    site_file: Path,
    *,
    root: Path,
    environment: Mapping[str, str],
) -> dict[str, Any]:
    site = load_site(site_file, repository_root=root)
    summary_environment = {
        **effective_environment(site),
        **environment,
        **site.environment,
        "GPU_FAULT_REPO_ROOT": str(root),
    }
    rollout = root / "deploy/control-plane/regional/rollout-regional-release.sh"
    with materialized_release_config(site) as config:
        return _run_json(
            [
                str(rollout),
                "release-summary",
                "--config",
                str(config),
            ],
            cwd=root,
            environment=summary_environment,
        )


def _collect_release_diff(
    site_file: Path,
    *,
    root: Path,
    environment: Mapping[str, str],
) -> dict[str, Any]:
    site = load_site(site_file, repository_root=root)
    diff_environment = {
        **effective_environment(site),
        **environment,
        **site.environment,
        "GPU_FAULT_REPO_ROOT": str(root),
    }
    rollout = root / "deploy/control-plane/regional/rollout-regional-release.sh"
    with materialized_release_config(site) as config:
        return _run_json(
            [
                str(rollout),
                "release-diff",
                "--config",
                str(config),
            ],
            cwd=root,
            environment=diff_environment,
        )


def _collect_stability_report(
    site_file: Path,
    *,
    root: Path,
    environment: Mapping[str, str],
) -> dict[str, Any]:
    site = load_site(site_file, repository_root=root)
    stability_environment = {
        **effective_environment(site),
        **environment,
        **site.environment,
        "GPU_FAULT_REPO_ROOT": str(root),
    }
    rollout = root / "deploy/control-plane/regional/rollout-regional-release.sh"
    with materialized_release_config(site) as config:
        return _run_json(
            [
                str(rollout),
                "stability",
                "--config",
                str(config),
            ],
            cwd=root,
            environment=stability_environment,
        )


def _run_release_mode(
    site_file: Path,
    *,
    mode: str,
    root: Path,
    environment: Mapping[str, str],
) -> None:
    site = load_site(site_file, repository_root=root)
    rollout_environment = {
        **effective_environment(site),
        **environment,
        **site.environment,
        "GPU_FAULT_REPO_ROOT": str(root),
    }
    rollout = root / "deploy/control-plane/regional/rollout-regional-release.sh"
    with materialized_release_config(site) as config:
        _run(
            [
                str(rollout),
                mode,
                "--config",
                str(config),
            ],
            cwd=root,
            environment=rollout_environment,
        )


def _complete_release(
    prepared: PreparedRelease,
    *,
    site_file: Path,
    root: Path,
    environment: Mapping[str, str],
    verification: dict[str, Any],
    stability: dict[str, Any],
    summary_report: dict[str, Any] | None,
    summary_generated_at: str | None,
) -> None:
    completion_warnings: list[str] = []
    try:
        if summary_report is None:
            summary_report = _collect_release_summary(
                site_file,
                root=root,
                environment=environment,
            )
            summary_generated_at = utc_now()
        summary_path = prepared.state_dir / RELEASE_SUMMARY_REPORT
        write_json_atomic(summary_path, summary_report)
        completion_warnings.extend(release_summary_warnings(summary_report))
        release_summary = {
            "status": (
                "AVAILABLE_WITH_WARNINGS" if completion_warnings else "AVAILABLE"
            ),
            "path": str(summary_path),
            "sha256": sha256_file(summary_path),
            "generated_at": summary_generated_at or utc_now(),
            "next_deploy": summary_report.get("next_deploy"),
        }
    except Exception as exc:
        warning = f"release summary unavailable: {type(exc).__name__}: {exc}"
        completion_warnings.append(warning)
        release_summary = {
            "status": "UNAVAILABLE",
            "generated_at": utc_now(),
            "error": warning,
        }
        print(f"release-deploy: warning: {warning}", file=sys.stderr)
    update_phase(
        prepared,
        "COMPLETED",
        verification=verification,
        stability=stability,
        release_summary=release_summary,
        completion_warnings=completion_warnings,
    )


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
                "Runtime Profile plan or live baseline changed; the previous "
                "approval was archived. Review the new "
                "release-deploy/profile-plan.json and run "
                "gpu-fault-admin approve-profile again"
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
            "Runtime Profile policy changed; review release-deploy/profile-plan.json "
            "and run gpu-fault-admin approve-profile --state-dir ... "
            "--plan-sha256 <plan_sha256> --reference ..."
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


def _deployment_decision(
    site_file: Path,
    *,
    root: Path,
    environment: Mapping[str, str],
) -> tuple[dict[str, Any], str | None]:
    try:
        candidate_diff = _collect_release_diff(
            site_file,
            root=root,
            environment=environment,
        )
    except Exception as exc:
        return (
            {
                "status": "APPLIED",
                "fast_path": False,
                "reason": (
                    "release diff unavailable; used full deploy path: "
                    f"{type(exc).__name__}: {exc}"
                ),
            },
            None,
        )
    state_sha256 = candidate_diff.get("state_sha256")
    if not isinstance(state_sha256, str) or len(state_sha256) != 64:
        return (
            {
                "status": "APPLIED",
                "fast_path": False,
                "reason": "release diff did not return a valid state identity",
            },
            None,
        )
    if not is_clean_noop_diff(candidate_diff):
        return (
            {
                "status": "APPLIED",
                "fast_path": False,
                "reason": "component release diff requires deployment",
                "next_deploy": candidate_diff.get("next_deploy"),
            },
            state_sha256,
        )
    _run_release_mode(
        site_file,
        mode="stage-noop",
        root=root,
        environment={
            **environment,
            EXPECTED_STATE_SHA256_ENV: state_sha256,
        },
    )
    return (
        {
            "status": "SKIPPED_NOOP",
            "fast_path": True,
            "reason": "component release diff classified desired state as NOOP",
            "next_deploy": candidate_diff["next_deploy"],
        },
        state_sha256,
    )


def _finalize_quick_validation_evidence(
    path: Path,
    *,
    prepared: PreparedRelease,
    site_file: Path,
    root: Path,
    environment: Mapping[str, str],
) -> Path | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        if (
            not isinstance(value, dict)
            or value.get("schema_version") != 1
            or value.get("release_id") != prepared.release_id
        ):
            raise ValueError("quick validation evidence identity is invalid")
        diff = _collect_release_diff(
            site_file,
            root=root,
            environment=environment,
        )
        if not (
            is_pending_commit_diff(diff, prepared.release_id)
            or is_clean_noop_diff(diff)
        ):
            raise ValueError(
                "post-deploy release state is neither a pending commit for "
                f"{prepared.release_id} nor a clean NOOP"
            )
        value["release_state_sha256"] = diff["state_sha256"]
        value["finalized_at"] = utc_now()
        write_json_atomic(path, value)
        path.chmod(0o600)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(
            "release-deploy: quick validation evidence was not reusable; "
            f"full verification will run: {exc}",
            file=sys.stderr,
        )
        return None
    return path


def _run_verification_and_stability(
    prepared: PreparedRelease,
    *,
    site_file: Path,
    root: Path,
    environment: Mapping[str, str],
    deployment: Mapping[str, Any],
    reusable_quick_evidence: Path | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Verify the fleet and watch it hold steady, then record both reports.

    Returns the recorded ``(verification, stability)`` metadata. The first
    failure is raised -- verify ahead of the stability window -- and only after
    both phases have finished, so nothing is still probing a cluster the
    caller's rollback is about to move.
    """

    def run_verification() -> dict[str, Any]:
        report = _run_json(
            [
                sys.executable,
                "-m",
                "gpu_fault.admin.cli",
                "verify",
                "-f",
                str(site_file),
            ],
            cwd=root,
            environment={
                **environment,
                **(
                    {QUICK_VALIDATION_EVIDENCE_ENV: str(reusable_quick_evidence)}
                    if reusable_quick_evidence is not None
                    else {}
                ),
            },
        )
        validate_verification_report(report)
        return report

    def run_stability() -> dict[str, Any] | None:
        if deployment["fast_path"]:
            return None
        report = _collect_stability_report(
            site_file,
            root=root,
            environment=environment,
        )
        validate_stability_report(report)
        return report

    # Both phases are read-only and neither reads anything the other writes,
    # but on production verify costs 44 s and the stability window 128 s --
    # nearly all of it sleeping between samples. Serially that is most of
    # three minutes of the release spent waiting twice.
    #
    # Every result is awaited before any is acted on: a failure in one phase
    # must not leave the other's subprocess still probing a cluster the
    # caller's rollback is about to move. A verify failure is the one raised when
    # both fail, so verify keeps reporting itself ahead of a stability window
    # that was doomed the moment verify failed.
    with ThreadPoolExecutor(max_workers=2) as executor:
        verification_future = executor.submit(run_verification)
        stability_future = executor.submit(run_stability)
        verification_report: dict[str, Any] | None = None
        stability_report: dict[str, Any] | None = None
        verification_error: Exception | None = None
        stability_error: Exception | None = None
        try:
            verification_report = verification_future.result()
        except Exception as error:
            verification_error = error
        try:
            stability_report = stability_future.result()
        except Exception as error:
            stability_error = error
    # A verify result that was produced is recorded even when the stability
    # window then fails. Serially it always was, and an operator reading the
    # failed release still needs to know whether the fleet verified before it
    # went unstable -- losing that to the concurrency would be a report the
    # release paid 44 s for and then threw away.
    if verification_report is not None:
        verification_path = prepared.state_dir / VERIFICATION_REPORT
        write_json_atomic(verification_path, verification_report)
        verification = verification_metadata(
            verification_path,
            verification_report,
        )
        update_phase(prepared, "VERIFIED", verification=verification)
    if verification_error is not None:
        raise verification_error
    if stability_error is not None:
        raise stability_error
    if stability_report is None:
        stability = {
            "status": "SKIPPED_NOOP",
            "reason": "no release mutation occurred",
        }
    else:
        stability_path = prepared.state_dir / STABILITY_REPORT
        write_json_atomic(stability_path, stability_report)
        stability = verification_metadata(
            stability_path,
            stability_report,
        )
        update_phase(
            prepared,
            "STABLE",
            verification=verification,
            stability=stability,
        )
    return verification, stability


def _commit_and_complete(
    prepared: PreparedRelease,
    *,
    site_file: Path,
    root: Path,
    environment: Mapping[str, str],
    verification: dict[str, Any],
    stability: dict[str, Any],
    summary_report: dict[str, Any] | None,
    summary_generated_at: str | None,
) -> None:
    """Commit the release transaction, then write the completion record."""

    _run_release_mode(
        site_file,
        mode="commit",
        root=root,
        environment=environment,
    )

    _complete_release(
        prepared,
        site_file=site_file,
        root=root,
        environment=environment,
        verification=verification,
        stability=stability,
        summary_report=summary_report,
        summary_generated_at=summary_generated_at,
    )


def _execute_release_locked(
    site_file: Path,
    *,
    admin_email: str | None = None,
    run_checks: bool = True,
    live_state: dict[str, Any] | None = None,
) -> PreparedRelease:
    if admin_email:
        raise ReleaseDeployError(
            "administrator email changes must be applied through IaC and site.yaml"
        )
    root = repository_root_from_site(site_file)
    environment = {
        **os.environ,
        "PYTHONPATH": str(root / "src"),
    }
    automatic_rollback = bool(
        load_site(site_file, repository_root=root).release_config["auto_rollback"]
    )
    if live_state is not None:
        current_state = live_state
    else:
        try:
            current_state = read_live_release_state(site_file)
        except ReleaseStateNotFound:
            current_state = {}
    profile_plan = plan_runtime_profile(
        site_file,
        live_profile_sha256=str(current_state.get("runtime_profile_sha256") or "")
        or None,
    )
    approval = _resolve_profile_approval_state(site_file, profile_plan)
    profile_plan = approval.profile_plan
    if run_checks:
        _run(
            ["make", f"PYTHON={sys.executable}", "check"],
            cwd=root,
            environment=environment,
        )
    prepared = prepare_site_release(
        site_file,
        profile_plan=profile_plan,
    )
    deployment_succeeded = False
    commit_started = False
    try:
        deploy_command = [
            sys.executable,
            "-m",
            "gpu_fault.admin.cli",
            "deploy",
            "-f",
            str(site_file),
        ]

        summary_report: dict[str, Any] | None = None
        summary_generated_at: str | None = None
        deployment, expected_state_sha256 = _deployment_decision(
            site_file,
            root=root,
            environment=environment,
        )

        if not deployment["fast_path"]:
            quick_evidence_path = prepared.state_dir / "quick-validation.json"
            quick_evidence_path.unlink(missing_ok=True)
            _run(
                deploy_command,
                cwd=root,
                environment={
                    **environment,
                    QUICK_VALIDATION_EVIDENCE_ENV: str(quick_evidence_path),
                    **(
                        {EXPECTED_STATE_SHA256_ENV: expected_state_sha256}
                        if expected_state_sha256 is not None
                        else {}
                    ),
                },
            )
            deployment_succeeded = True
            reusable_quick_evidence = _finalize_quick_validation_evidence(
                quick_evidence_path,
                prepared=prepared,
                site_file=site_file,
                root=root,
                environment=environment,
            )
        else:
            reusable_quick_evidence = None
        update_phase(prepared, "DEPLOYED", deployment=deployment)

        verification, stability = _run_verification_and_stability(
            prepared,
            site_file=site_file,
            root=root,
            environment=environment,
            deployment=deployment,
            reusable_quick_evidence=reusable_quick_evidence,
        )
        commit_started = True
        _commit_and_complete(
            prepared,
            site_file=site_file,
            root=root,
            environment=environment,
            verification=verification,
            stability=stability,
            summary_report=summary_report,
            summary_generated_at=summary_generated_at,
        )
    except Exception as exc:
        failure_error = f"{type(exc).__name__}: {exc}"
        failed_at = utc_now()
        rollback = recover_release_failure(
            prepared,
            site_file=site_file,
            root=root,
            environment=environment,
            failure_error=failure_error,
            failed_at=failed_at,
            deployment_succeeded=deployment_succeeded,
            commit_started=commit_started,
            automatic_rollback=automatic_rollback,
            run_release_mode=_run_release_mode,
            read_live_state=read_live_release_state,
            update_phase=update_phase,
        )
        update_phase(
            prepared,
            "FAILED",
            error=failure_error,
            failed_at=failed_at,
            rollback=rollback,
        )
        if rollback is not None and rollback["status"] == "FAILED":
            raise ReleaseDeployError(
                "release validation failed and rollback also failed: "
                f"validation={type(exc).__name__}: {exc}; "
                f"rollback={rollback['error']}"
            ) from exc
        raise
    _consume_profile_approval_state(site_file, prepared, approval)
    return prepared


def execute_release(
    site_file: Path,
    *,
    admin_email: str | None = None,
    run_checks: bool = True,
    live_state: dict[str, Any] | None = None,
) -> PreparedRelease:
    try:
        with profile_approval_lock(site_file.parent):
            return _execute_release_locked(
                site_file,
                admin_email=admin_email,
                run_checks=run_checks,
                live_state=live_state,
            )
    except ProfileApprovalError as exc:
        raise ReleaseDeployError(str(exc)) from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build, prepare and deploy one GPU fault release through the "
            "declarative site configuration."
        )
    )
    parser.add_argument(
        "--site",
        type=Path,
        help=f"RegionalSite YAML; defaults to {SITE_ENV}",
    )
    parser.add_argument(
        "--prebuilt-attestation",
        type=Path,
        help=(
            "verified CI attestation for a prebuilt schema v3 release; "
            "when set, the deployment host does not rerun make check"
        ),
    )
    parser.add_argument("--prebuilt-signature", type=Path)
    parser.add_argument("--prebuilt-bundle", type=Path)
    parser.add_argument("--prebuilt-certificate", type=Path)
    parser.add_argument("--cosign-key")
    parser.add_argument("--certificate-identity")
    parser.add_argument("--certificate-oidc-issuer")
    parser.add_argument(
        "--allow-staging-release",
        action="store_true",
        help="allow a staging-only signed release; never use for production",
    )
    parser.add_argument(
        "--profile-plan-json",
        action="store_true",
        help=(
            "read-only: print the Runtime Profile plan (template versus live "
            "Profile) as JSON and exit without building or deploying anything"
        ),
    )
    return parser


def print_profile_plan(site_file: Path) -> dict[str, Any]:
    """The Runtime Profile plan the release would act on, without acting.

    The staging fast path asks this before it calls a site UNCHANGED: the
    Profile template lives outside the source repository and the site file, so
    no other identity moves when it changes. Same planner, same live baseline
    as ``execute_release`` -- the two cannot disagree about what "unchanged"
    means. Nothing is written: no ``profile-plan.json``, no approval state.
    """

    try:
        current_state = read_live_release_state(site_file)
    except ReleaseStateNotFound:
        current_state = {}
    plan = plan_runtime_profile(
        site_file,
        live_profile_sha256=str(current_state.get("runtime_profile_sha256") or "")
        or None,
    )
    payload = _profile_plan_payload(plan)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        site_file = resolve_site_file(arguments.site, os.environ)
        if arguments.profile_plan_json:
            print_profile_plan(site_file)
            return 0
        if arguments.prebuilt_attestation is None:
            raise ReleaseDeployError(
                "release-deploy requires a signed prebuilt attestation"
            )
        attestation = verify_attestation(
            ROOT,
            arguments.prebuilt_attestation.resolve(),
            signature=(
                arguments.prebuilt_signature.resolve()
                if arguments.prebuilt_signature is not None
                else None
            ),
            bundle=(
                arguments.prebuilt_bundle.resolve()
                if arguments.prebuilt_bundle is not None
                else None
            ),
            cosign_key=arguments.cosign_key,
            certificate=(
                arguments.prebuilt_certificate.resolve()
                if arguments.prebuilt_certificate is not None
                else None
            ),
            certificate_identity=arguments.certificate_identity,
            certificate_oidc_issuer=(arguments.certificate_oidc_issuer),
            allow_staging=arguments.allow_staging_release,
        )
        _manifest, current_release_id = release_manifest(ROOT)
        subject = dict(attestation.get("subject") or {})
        current_manifest = ROOT / "dist/current-release.json"
        if subject.get("release_id") != current_release_id or subject.get(
            "manifest_sha256"
        ) != sha256_file(current_manifest):
            raise ReleaseDeployError(
                "verified attestation does not bind dist/current-release.json"
            )
        prepared = execute_release(
            site_file,
            run_checks=False,
        )
    except (OSError, ReleaseDeployError, subprocess.SubprocessError, ValueError) as exc:
        print(f"release-deploy: {exc}", file=sys.stderr)
        return 2
    state = json.loads((prepared.state_dir / "state.json").read_text(encoding="utf-8"))
    print(
        json.dumps(
            {
                "release_id": prepared.release_id,
                "site_file": str(prepared.site_file),
                "runtime_profile_version": prepared.runtime_profile_version,
                "profile_change_kind": prepared.profile_change_kind,
                "profile_approval": prepared.profile_approval,
                "agent_config_digest": prepared.agent_config_digest,
                "site_changed": prepared.site_changed,
                "state_dir": str(prepared.state_dir),
                "phase": state.get("phase"),
                "deployment": state.get("deployment"),
                "verification": state.get("verification"),
                "stability": state.get("stability"),
                "release_summary": state.get("release_summary"),
                "completion_warnings": state.get("completion_warnings", []),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

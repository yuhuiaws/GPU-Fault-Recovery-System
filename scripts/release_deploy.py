from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from gpu_fault.admin_bootstrap_common import (
    CommandRunner,
    compute_agent_config_digest,
)
from gpu_fault.admin_site import (
    effective_environment,
    load_site,
    materialized_release_config,
)
from gpu_fault.capabilities import compile_runtime_profile
from gpu_fault.models import CapabilityMode, RuntimeProfile

if __package__:
    from scripts.release_attestation import verify_attestation
else:
    from release_attestation import verify_attestation


ROOT = Path(__file__).resolve().parents[1]
SITE_ENV = "GPU_FAULT_SITE_FILE"
PROFILE_APPROVAL_ENV = "PROFILE_APPROVAL"
VERIFICATION_REPORT = "verification-report.json"
STABILITY_REPORT = "stability-report.json"
RELEASE_SUMMARY_REPORT = "release-summary.json"
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
RELEASE_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
APPROVAL_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{2,127}$")


class ReleaseDeployError(RuntimeError):
    pass


@dataclass(frozen=True)
class PreparedRelease:
    site_file: Path
    release_id: str
    runtime_profile_version: str
    agent_config_digest: str
    profile_change_kind: str
    profile_approval: str | None
    state_dir: Path
    site_changed: bool


@dataclass(frozen=True)
class ProfilePlan:
    template_source: Path
    template_reference: str
    active_source: Path
    current_version: str
    desired_version: str
    policy_digest: str
    source_sha256: str
    change_kind: str
    changes: tuple[str, ...]
    approval: str | None
    approval_required: bool
    snapshot_file: Path
    snapshot_document: dict[str, Any]


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


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
    return profile, digest, _sha256(path), normalized


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
    site = load_site(site_file)
    command = [
        "kubectl",
        "--kubeconfig",
        str(site.release_config["cpu_kubeconfig"]),
        "-n",
        str(site.release_config["namespace"]),
        "get",
        "configmap",
        "gpu-fault-regional-release-state",
        "-o",
        "json",
    ]
    completed = runner(
        command,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode:
        raise ReleaseDeployError(
            completed.stderr.strip() or "cannot read gpu-fault-regional-release-state"
        )
    try:
        config_map = json.loads(completed.stdout)
        raw = config_map["data"]["state.json"]
        value = json.loads(raw)
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ReleaseDeployError("regional release state is invalid") from exc
    return _mapping(value, "regional release state")


def plan_runtime_profile(
    site_file: Path,
    *,
    live_profile_sha256: str | None,
    approval: str | None,
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
    active_sha = _sha256(active_source) if active_source.is_file() else None
    active_is_live = live_profile_sha256 is None or active_sha == live_profile_sha256
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

    normalized_approval = (approval or "").strip() or None
    approval_required = change_kind != "UNCHANGED"
    if normalized_approval is not None and not APPROVAL_PATTERN.fullmatch(
        normalized_approval
    ):
        raise ReleaseDeployError("PROFILE_APPROVAL has an invalid format")
    snapshot_document["profile_version"] = desired_version
    snapshot_document["cluster_id"] = registration_cluster_id
    return ProfilePlan(
        template_source=template_source,
        template_reference=template_reference,
        active_source=active_source,
        current_version=current_version,
        desired_version=desired_version,
        policy_digest=policy_digest,
        source_sha256=source_sha,
        change_kind=change_kind,
        changes=changes,
        approval=normalized_approval,
        approval_required=approval_required,
        snapshot_file=site_file.parent / "profiles" / f"{desired_version}.yaml",
        snapshot_document=snapshot_document,
    )


def _profile_plan_payload(plan: ProfilePlan) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "template_source": str(plan.template_source),
        "active_source": str(plan.active_source),
        "snapshot_file": str(plan.snapshot_file),
        "current_version": plan.current_version,
        "desired_version": plan.desired_version,
        "policy_digest": plan.policy_digest,
        "source_sha256": plan.source_sha256,
        "change_kind": plan.change_kind,
        "changes": list(plan.changes),
        "approval_required": plan.approval_required,
        "approval": plan.approval,
    }


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.parent.chmod(0o700)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.chmod(0o600)
    temporary.replace(path)


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


def _release_manifest(root: Path) -> tuple[dict[str, Any], str]:
    path = root / "dist/current-release.json"
    if not path.is_file():
        raise ReleaseDeployError(
            "dist/current-release.json is missing; the build gate did not produce a release"
        )
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReleaseDeployError(f"invalid release manifest {path}: {exc}") from exc
    release_id = str(manifest.get("release_id") or "").strip()
    if not RELEASE_ID_PATTERN.fullmatch(release_id):
        raise ReleaseDeployError("release manifest has no valid release_id")
    immutable = root / "dist" / release_id / "release.json"
    if not immutable.is_file():
        raise ReleaseDeployError(
            f"content-addressed release manifest is missing: {immutable}"
        )
    if immutable.read_bytes() != path.read_bytes():
        raise ReleaseDeployError(
            "current-release.json differs from its content-addressed release.json"
        )
    return manifest, release_id


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
    manifest, release_id = _release_manifest(rendered.repository_root)
    document = _read_site_document(site_file)
    spec = _mapping(document.get("spec"), "site spec")
    release = _mapping(spec.get("release"), "site spec.release")
    profile = _mapping(spec.get("runtimeProfile"), "site spec.runtimeProfile")
    images = _mapping(spec.setdefault("images", {}), "site spec.images")
    if profile_plan.approval_required and not profile_plan.approval:
        raise ReleaseDeployError(
            "Runtime Profile changed; set PROFILE_APPROVAL to an approved change ID"
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
        },
    }
    _write_json_atomic(state_dir / "plan.json", plan)
    _write_json_atomic(
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


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _verification_metadata(path: Path, report: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": "PASSED",
        "path": str(path),
        "sha256": _sha256(path),
        "verified_at": _utc_now(),
        "summary": report.get("summary"),
    }


def _validate_verification_report(report: dict[str, Any]) -> None:
    if report.get("mode") != "verify":
        raise ReleaseDeployError("verification command returned the wrong report mode")
    if report.get("healthy") is not True:
        raise ReleaseDeployError("verification report is not healthy")
    summary = report.get("summary")
    if not isinstance(summary, dict) or summary.get("FAIL") != 0:
        raise ReleaseDeployError("verification report has an invalid summary")
    if not isinstance(report.get("checks"), list):
        raise ReleaseDeployError("verification report has no check evidence")


def _validate_stability_report(report: dict[str, Any]) -> None:
    if report.get("mode") != "stability" or report.get("healthy") is not True:
        raise ReleaseDeployError("release stability report is not healthy")
    if int(report.get("window_seconds") or 0) < 120:
        raise ReleaseDeployError("release stability report has an invalid window")


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


def _release_summary_warnings(report: dict[str, Any]) -> list[str]:
    warnings: list[str] = []
    for field in ("release_status_error", "next_deploy_error"):
        if report.get(field):
            warnings.append(f"{field}: {report[field]}")
    next_deploy = report.get("next_deploy")
    if not isinstance(next_deploy, dict):
        warnings.append("release summary has no next_deploy classification")
    elif next_deploy.get("kind") != "NOOP":
        warnings.append(
            "release summary reports a non-NOOP next deploy: "
            + json.dumps(next_deploy, sort_keys=True)
        )
    return warnings


def _is_clean_noop_summary(report: dict[str, Any]) -> bool:
    next_deploy = report.get("next_deploy")
    return (
        report.get("mode") == "release-summary"
        and isinstance(next_deploy, dict)
        and next_deploy.get("kind") == "NOOP"
        and next_deploy.get("changed") == []
        and not _release_summary_warnings(report)
    )


def _update_phase(prepared: PreparedRelease, phase: str, **values: Any) -> None:
    state_path = prepared.state_dir / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state.update({"phase": phase, **values})
    _write_json_atomic(state_path, state)


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
            summary_generated_at = _utc_now()
        summary_path = prepared.state_dir / RELEASE_SUMMARY_REPORT
        _write_json_atomic(summary_path, summary_report)
        completion_warnings.extend(_release_summary_warnings(summary_report))
        release_summary = {
            "status": (
                "AVAILABLE_WITH_WARNINGS" if completion_warnings else "AVAILABLE"
            ),
            "path": str(summary_path),
            "sha256": _sha256(summary_path),
            "generated_at": summary_generated_at or _utc_now(),
            "next_deploy": summary_report.get("next_deploy"),
        }
    except Exception as exc:
        warning = f"release summary unavailable: {type(exc).__name__}: {exc}"
        completion_warnings.append(warning)
        release_summary = {
            "status": "UNAVAILABLE",
            "generated_at": _utc_now(),
            "error": warning,
        }
        print(f"release-deploy: warning: {warning}", file=sys.stderr)
    _update_phase(
        prepared,
        "COMPLETED",
        verification=verification,
        stability=stability,
        release_summary=release_summary,
        completion_warnings=completion_warnings,
    )


def execute_release(
    site_file: Path,
    *,
    profile_approval: str | None = None,
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
    current_state = (
        live_state if live_state is not None else read_live_release_state(site_file)
    )
    profile_plan = plan_runtime_profile(
        site_file,
        live_profile_sha256=str(current_state.get("runtime_profile_sha256") or "")
        or None,
        approval=profile_approval or os.getenv(PROFILE_APPROVAL_ENV),
    )
    if profile_plan.approval_required and not profile_plan.approval:
        _write_json_atomic(
            site_file.parent / "release-deploy/profile-plan.json",
            _profile_plan_payload(profile_plan),
        )
        raise ReleaseDeployError(
            "Runtime Profile policy changed; review release-deploy/profile-plan.json "
            f"and set {PROFILE_APPROVAL_ENV} to the approved change ID"
        )
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
    try:
        deploy_command = [
            sys.executable,
            "-m",
            "gpu_fault.admin_cli",
            "deploy",
            "-f",
            str(site_file),
        ]

        summary_report: dict[str, Any] | None = None
        summary_generated_at: str | None = None
        deployment: dict[str, Any]
        same_release = current_state.get("release_id") == prepared.release_id
        if same_release:
            try:
                candidate_summary = _collect_release_summary(
                    site_file,
                    root=root,
                    environment=environment,
                )
            except Exception as exc:
                deployment = {
                    "status": "APPLIED",
                    "fast_path": False,
                    "reason": (
                        "NOOP classification unavailable; used full deploy path: "
                        f"{type(exc).__name__}: {exc}"
                    ),
                }
            else:
                if _is_clean_noop_summary(candidate_summary):
                    summary_report = candidate_summary
                    summary_generated_at = _utc_now()
                    deployment = {
                        "status": "SKIPPED_NOOP",
                        "fast_path": True,
                        "reason": "live release summary classified desired state as NOOP",
                        "next_deploy": candidate_summary["next_deploy"],
                    }
                else:
                    deployment = {
                        "status": "APPLIED",
                        "fast_path": False,
                        "reason": "live release summary requires deployment",
                        "next_deploy": candidate_summary.get("next_deploy"),
                    }
        else:
            deployment = {
                "status": "APPLIED",
                "fast_path": False,
                "reason": "live release ID differs from the desired release",
            }

        if not deployment["fast_path"]:
            _run(
                deploy_command,
                cwd=root,
                environment=environment,
            )
            deployment_succeeded = True
        _update_phase(prepared, "DEPLOYED", deployment=deployment)
        verification_report = _run_json(
            [
                sys.executable,
                "-m",
                "gpu_fault.admin_cli",
                "verify",
                "-f",
                str(site_file),
            ],
            cwd=root,
            environment=environment,
        )
        _validate_verification_report(verification_report)
        verification_path = prepared.state_dir / VERIFICATION_REPORT
        _write_json_atomic(verification_path, verification_report)
        verification = _verification_metadata(
            verification_path,
            verification_report,
        )
        _update_phase(prepared, "VERIFIED", verification=verification)
        if deployment["fast_path"]:
            stability = {
                "status": "SKIPPED_NOOP",
                "reason": "no release mutation occurred",
            }
        else:
            stability_report = _collect_stability_report(
                site_file,
                root=root,
                environment=environment,
            )
            _validate_stability_report(stability_report)
            stability_path = prepared.state_dir / STABILITY_REPORT
            _write_json_atomic(stability_path, stability_report)
            stability = _verification_metadata(
                stability_path,
                stability_report,
            )
            _update_phase(
                prepared,
                "STABLE",
                verification=verification,
                stability=stability,
            )
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
    except Exception as exc:
        rollback: dict[str, Any] | None = None
        if deployment_succeeded:
            try:
                _run_release_mode(
                    site_file,
                    mode="rollback",
                    root=root,
                    environment=environment,
                )
            except Exception as rollback_exc:
                rollback = {
                    "status": "FAILED",
                    "error": (f"{type(rollback_exc).__name__}: {rollback_exc}"),
                }
            else:
                rollback = {"status": "PASSED"}
        _update_phase(
            prepared,
            "FAILED",
            error=f"{type(exc).__name__}: {exc}",
            rollback=rollback,
        )
        if rollback is not None and rollback["status"] == "FAILED":
            raise ReleaseDeployError(
                "release validation failed and rollback also failed: "
                f"validation={type(exc).__name__}: {exc}; "
                f"rollback={rollback['error']}"
            ) from exc
        raise
    return prepared


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
        "--profile-approval",
        help=(
            "approved change ID for a Runtime Profile policy change; defaults to "
            f"{PROFILE_APPROVAL_ENV}"
        ),
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
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        site_file = resolve_site_file(arguments.site, os.environ)
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
        )
        _manifest, current_release_id = _release_manifest(ROOT)
        subject = dict(attestation.get("subject") or {})
        current_manifest = ROOT / "dist/current-release.json"
        if subject.get("release_id") != current_release_id or subject.get(
            "manifest_sha256"
        ) != _sha256(current_manifest):
            raise ReleaseDeployError(
                "verified attestation does not bind dist/current-release.json"
            )
        prepared = execute_release(
            site_file,
            profile_approval=arguments.profile_approval,
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

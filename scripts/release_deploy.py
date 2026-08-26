from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from gpu_fault.admin_bootstrap_common import (
    CommandRunner,
    compute_agent_config_digest,
)
from gpu_fault.admin_site import load_site
from gpu_fault.capabilities import compile_runtime_profile
from gpu_fault.models import CapabilityMode, RuntimeProfile


ROOT = Path(__file__).resolve().parents[1]
SITE_ENV = "GPU_FAULT_SITE_FILE"
PROFILE_APPROVAL_ENV = "PROFILE_APPROVAL"
ADMIN_EMAIL_ENV = "GPU_FAULT_ADMIN_EMAIL"
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
    _manifest, release_id = _release_manifest(rendered.repository_root)
    document = _read_site_document(site_file)
    spec = _mapping(document.get("spec"), "site spec")
    release = _mapping(spec.get("release"), "site spec.release")
    profile = _mapping(spec.get("runtimeProfile"), "site spec.runtimeProfile")
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
    }
    _write_profile_snapshot(profile_plan)
    release["manifest"] = "dist/current-release.json"
    release["agentConfigDigest"] = digest
    profile["source"] = str(profile_plan.snapshot_file)
    profile["templateSource"] = profile_plan.template_reference
    profile["version"] = desired_profile
    candidate = state_dir / "site.candidate.yaml"
    _write_yaml_atomic(candidate, document)
    load_site(candidate, repository_root=rendered.repository_root)

    changed = previous != {
        "manifest": release["manifest"],
        "agent_config_digest": digest,
        "runtime_profile_source": profile["source"],
        "runtime_profile_template_source": profile["templateSource"],
        "runtime_profile_version": desired_profile,
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


def _update_phase(prepared: PreparedRelease, phase: str, **values: Any) -> None:
    state_path = prepared.state_dir / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state.update({"phase": phase, **values})
    _write_json_atomic(state_path, state)


def execute_release(
    site_file: Path,
    *,
    profile_approval: str | None = None,
    admin_email: str | None = None,
    run_checks: bool = True,
    live_state: dict[str, Any] | None = None,
) -> PreparedRelease:
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
    try:
        deploy_command = [
            sys.executable,
            "-m",
            "gpu_fault.admin_cli",
            "deploy",
            "-f",
            str(site_file),
        ]
        effective_admin_email = admin_email or os.getenv(ADMIN_EMAIL_ENV)
        if effective_admin_email:
            deploy_command.extend(["--admin-email", effective_admin_email])
        _run(
            deploy_command,
            cwd=root,
            environment=environment,
        )
        _update_phase(prepared, "DEPLOYED")
        _run(
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
        _update_phase(prepared, "VERIFIED")
        _run(
            [
                sys.executable,
                "-m",
                "gpu_fault.admin_cli",
                "status",
                "-f",
                str(site_file),
            ],
            cwd=root,
            environment=environment,
        )
        _update_phase(prepared, "COMPLETED")
    except Exception as exc:
        _update_phase(
            prepared,
            "FAILED",
            error=f"{type(exc).__name__}: {exc}",
        )
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
        "--admin-email",
        help=(
            "administrator email for SES and SNS; defaults to "
            f"{ADMIN_EMAIL_ENV}, then the site/AWS account email"
        ),
    )
    parser.add_argument(
        "--profile-approval",
        help=(
            "approved change ID for a Runtime Profile policy change; defaults to "
            f"{PROFILE_APPROVAL_ENV}"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        site_file = resolve_site_file(arguments.site, os.environ)
        prepared = execute_release(
            site_file,
            profile_approval=arguments.profile_approval,
            admin_email=arguments.admin_email,
        )
    except (OSError, ReleaseDeployError, subprocess.SubprocessError, ValueError) as exc:
        print(f"release-deploy: {exc}", file=sys.stderr)
        return 2
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
                "phase": "COMPLETED",
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

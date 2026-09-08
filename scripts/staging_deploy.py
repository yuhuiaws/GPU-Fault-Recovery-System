from __future__ import annotations

import argparse
import json
import os
import secrets
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Mapping, Sequence, cast

import yaml  # type: ignore[import-untyped,unused-ignore]

from gpu_fault.admin.command_log import report_failure
from gpu_fault.admin.notification_precheck import WAIT_FLAG
from gpu_fault.admin.operation_lock import (
    SITE_OPERATION_LOCK_FD_ENV,
    site_operation_lock,
)

if __package__:
    from scripts.setup_deploy_host import prune_venv_versions
    from scripts.staging_deploy_inputs import (
        deploy_rerun_command,
        early_consent_refusal,
        live_release_block,
        managed_site_inputs,
        next_deploy_block,
        precheck_email_confirmations,
        resolve_deploy_inputs,
    )
    from scripts.staging_gate_caches import (
        public_release_verdict_cache,
        tool_cache_environment,
    )
    from scripts.staging_live_evidence import (
        LiveEvidenceError,
        RuntimeProfileChangePending,
        collect_live_deploy_evidence,
        live_evidence_from_release_record,
        read_live_status,
        successful_source_live_matches,
    )
    from scripts.staging_source_snapshot import _run, prepare_source_checkout
    from scripts.staging_state_hygiene import (
        DeployHostArtifacts,
        SigningMaterial,
        SourceCheckout,
        StagingDeployError,
        deploy_host_artifacts,
        deploy_host_wheelhouse_cache,
        private_state_file,
        prune_source_snapshots,
        public_state_file,
        record_source_deploy_state,
        restore_site_file,
        source_success_paths,
    )
else:
    from setup_deploy_host import prune_venv_versions
    from staging_deploy_inputs import (
        deploy_rerun_command,
        early_consent_refusal,
        live_release_block,
        managed_site_inputs,
        next_deploy_block,
        precheck_email_confirmations,
        resolve_deploy_inputs,
    )
    from staging_gate_caches import (
        public_release_verdict_cache,
        tool_cache_environment,
    )
    from staging_live_evidence import (
        LiveEvidenceError,
        RuntimeProfileChangePending,
        collect_live_deploy_evidence,
        live_evidence_from_release_record,
        read_live_status,
        successful_source_live_matches,
    )
    from staging_source_snapshot import _run, prepare_source_checkout
    from staging_state_hygiene import (
        DeployHostArtifacts,
        SigningMaterial,
        SourceCheckout,
        StagingDeployError,
        deploy_host_artifacts,
        deploy_host_wheelhouse_cache,
        private_state_file,
        prune_source_snapshots,
        public_state_file,
        record_source_deploy_state,
        restore_site_file,
        source_success_paths,
    )


ROOT = Path(__file__).resolve().parents[1]


def validate_source_checkout(
    repository_root: Path,
    *,
    verdict_cache: Path | None = None,
) -> None:
    """Refuse to deploy a tree that carries live-environment identity.

    ``verdict_cache`` lets the second call in a deploy -- the prepared snapshot,
    whose content is a copy of the live tree the first call already cleared --
    prove it is scanning identical bytes instead of scanning them again. The gate
    still walks and hashes every public file each time; only the pattern matching
    is skipped, and only on an exact content match.
    """

    _run(
        [
            sys.executable,
            str(repository_root / "scripts/check-public-release.py"),
            "--root",
            str(repository_root),
            *(("--verdict-cache", str(verdict_cache)) if verdict_cache else ()),
        ],
        cwd=repository_root,
    )


def ensure_signing_material(
    state_dir: Path,
    *,
    repository_root: Path,
) -> SigningMaterial:
    signing_dir = state_dir / "release-signing"
    signing_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    signing_dir.chmod(0o700)
    private_key = signing_dir / "cosign.key"
    public_key = signing_dir / "cosign.pub"
    password_file = signing_dir / "cosign.password"
    key_present = private_key.is_file()
    public_present = public_key.is_file()
    if key_present != public_present:
        missing = "public key" if key_present else "private key"
        raise StagingDeployError(
            "release signing material is incomplete; missing " + missing
        )
    stored_password = (
        password_file.read_text(encoding="utf-8").strip()
        if password_file.is_file()
        else ""
    )
    password = stored_password or os.getenv("COSIGN_PASSWORD", "")
    if not password:
        if key_present:
            raise StagingDeployError(
                "existing Cosign key requires release-signing/cosign.password "
                "or COSIGN_PASSWORD"
            )
        password = secrets.token_urlsafe(48)
    if not stored_password:
        password_file.write_text(password, encoding="utf-8")
        password_file.chmod(0o600)
    if not key_present:
        environment = {**os.environ, "COSIGN_PASSWORD": password}
        previous_umask = os.umask(0o077)
        try:
            _run(
                [
                    "cosign",
                    "generate-key-pair",
                    "--output-key-prefix",
                    str(signing_dir / "cosign"),
                ],
                cwd=repository_root,
                env=environment,
            )
        finally:
            os.umask(previous_umask)
        private_key.chmod(0o600)
        public_key.chmod(0o644)
    return SigningMaterial(
        private_key=private_state_file(private_key, "Cosign signing key"),
        public_key=public_state_file(public_key, "Cosign public key"),
        password_file=private_state_file(password_file, "Cosign password file"),
        password=password,
    )


def source_deploy_identity(repository_root: Path) -> dict[str, object]:
    output = _run(
        [
            sys.executable,
            str(repository_root / "scripts/deploy_source_identity.py"),
            "--root",
            str(repository_root),
        ],
        cwd=repository_root,
        capture=True,
    )
    try:
        value = json.loads(output)
    except json.JSONDecodeError as exc:
        raise StagingDeployError("source deploy identity output is invalid") from exc
    if not isinstance(value, dict):
        raise StagingDeployError("source deploy identity must be an object")
    for name in ("application", "deploy_host"):
        item = value.get(name)
        if (
            not isinstance(item, dict)
            or not isinstance(item.get("sha256"), str)
            or len(str(item["sha256"])) != 64
        ):
            raise StagingDeployError(f"source deploy {name} identity is invalid")
    deploy_host = value["deploy_host"]
    assert isinstance(deploy_host, dict)
    bundle = deploy_host.get("bundle")
    if (
        not isinstance(bundle, dict)
        or not isinstance(bundle.get("sha256"), str)
        or len(str(bundle["sha256"])) != 64
    ):
        raise StagingDeployError("deploy-host bundle identity is invalid")
    return value


def restore_trusted_ci_candidate(
    repository_root: Path,
    *,
    staging_only: bool,
) -> dict[str, object] | None:
    if staging_only:
        return None
    output = _run(
        [
            sys.executable,
            str(repository_root / "scripts/restore_ci_candidate.py"),
            "--root",
            str(repository_root),
            "--destination",
            str(repository_root / "dist"),
        ],
        cwd=repository_root,
        capture=True,
    )
    try:
        value = json.loads(output)
    except json.JSONDecodeError as exc:
        raise StagingDeployError("trusted CI candidate result is invalid") from exc
    if not isinstance(value, dict) or not isinstance(value.get("available"), bool):
        raise StagingDeployError("trusted CI candidate result is incomplete")
    return value if value["available"] is True else None


def record_trusted_ci_candidate(
    repository_root: Path,
    *,
    state_dir: Path,
    signing: SigningMaterial,
    candidate: Mapping[str, object],
) -> None:
    output = _run(
        [
            sys.executable,
            str(repository_root / "scripts/ci_candidate_receipt.py"),
            "write",
            "--root",
            str(repository_root),
            "--state-dir",
            str(state_dir),
            "--gate",
            str(candidate["ci_gate"]),
            "--repository",
            str(candidate["repository"]),
            "--run-id",
            str(candidate["run_id"]),
            "--signing-key",
            str(signing.private_key),
        ],
        cwd=repository_root,
        env={**os.environ, "COSIGN_PASSWORD": signing.password},
        capture=True,
    )
    try:
        value = json.loads(output)
    except json.JSONDecodeError as exc:
        raise StagingDeployError(
            "trusted CI candidate receipt output is invalid"
        ) from exc
    if not isinstance(value, dict) or value.get("available") is not True:
        raise StagingDeployError("trusted CI candidate receipt was not written")


def ensure_deploy_host_bundle(
    artifacts: DeployHostArtifacts,
    *,
    repository_root: Path,
    signing: SigningMaterial,
    wheelhouse_cache: Path | None = None,
) -> bool:
    present = (
        artifacts.archive.is_file(),
        artifacts.checksum.is_file(),
        artifacts.signature_bundle.is_file(),
    )
    if any(present) and not all(present):
        for path in (
            artifacts.archive,
            artifacts.checksum,
            artifacts.signature_bundle,
        ):
            path.unlink(missing_ok=True)
    if all(present):
        return True
    artifacts.archive.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    environment = {**os.environ, "COSIGN_PASSWORD": signing.password}
    command = [
        "make",
        "deploy-host-bundle",
        f"PYTHON={sys.executable}",
        f"COSIGN_SIGNING_KEY={signing.private_key}",
        f"DEPLOY_HOST_ARCHIVE={artifacts.archive}",
        f"DEPLOY_HOST_SIGNATURE_BUNDLE={artifacts.signature_bundle}",
    ]
    if wheelhouse_cache is not None:
        command.append(f"DEPLOY_HOST_WHEELHOUSE={wheelhouse_cache}")
    _run(
        command,
        cwd=repository_root,
        env=environment,
    )
    if not all(
        path.is_file()
        for path in (
            artifacts.archive,
            artifacts.checksum,
            artifacts.signature_bundle,
        )
    ):
        raise StagingDeployError("deploy-host bundle build did not publish all files")
    return False


def ensure_deploy_host_venv(
    state_dir: Path,
    *,
    repository_root: Path,
    signing: SigningMaterial,
    artifacts: DeployHostArtifacts,
    lock_fd: int,
) -> Path:
    """Install the signed bundle into ``deployer-venv`` and bind it to the state.

    ``lock_fd`` is the site operation lock the caller holds. Superseded versions
    are deleted here and only here: the setup script this shells out to is also
    hand-run with no lock, and a tree a concurrent install is writing into looks
    exactly like one it abandoned.
    """

    venv = state_dir / "deployer-venv"
    environment = {**os.environ, "PYTHON": sys.executable}
    _run(
        [
            str(repository_root / "scripts/setup-deploy-host.sh"),
            "--venv",
            str(venv),
            "--bundle",
            str(artifacts.archive),
            "--signature-bundle",
            str(artifacts.signature_bundle),
            "--cosign-key",
            str(signing.public_key),
        ],
        cwd=repository_root,
        env=environment,
    )
    admin = venv / "bin/gpu-fault-admin"
    if not admin.is_file():
        raise StagingDeployError(f"deploy-host venv has no admin CLI: {admin}")
    binding = venv.resolve() / "gpu-fault-managed-state-dir.json"
    temporary = binding.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "state_dir": str(state_dir.expanduser().resolve()),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.chmod(0o600)
    os.replace(temporary, binding)
    try:
        prune_venv_versions(venv, lock_fd=lock_fd)
    except OSError as exc:
        # The new venv is installed, active and bound. Failing the deploy over
        # the disk the superseded versions occupy would report a broken host for
        # a host that is ready, so this is reported and left for the next run.
        print(f"staging-deploy: venv versions not pruned: {exc}", file=sys.stderr)
    return venv


def load_successful_source_deploy(
    state_dir: Path,
    *,
    signing: SigningMaterial,
) -> dict[str, object] | None:
    state_path, signature_path = source_success_paths(state_dir)
    present = (state_path.is_file(), signature_path.is_file())
    if not any(present):
        return None
    if not all(present):
        raise StagingDeployError("successful source deploy authorization is incomplete")
    _run(
        [
            "cosign",
            "verify-blob",
            "--bundle",
            str(signature_path),
            "--key",
            str(signing.public_key),
            str(state_path),
        ],
        cwd=state_dir,
        capture=True,
    )
    try:
        value = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise StagingDeployError(
            "successful source deploy authorization is invalid"
        ) from exc
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise StagingDeployError(
            "successful source deploy authorization schema is invalid"
        )
    if value.get("status") != "PASSED":
        raise StagingDeployError("successful source deploy authorization is not passed")
    identities = value.get("identities")
    source = value.get("source")
    if not isinstance(identities, dict) or not isinstance(source, dict):
        raise StagingDeployError("successful source deploy authorization is incomplete")
    for name in ("application", "deploy_host"):
        item = identities.get(name)
        if (
            not isinstance(item, dict)
            or not isinstance(item.get("sha256"), str)
            or len(str(item["sha256"])) != 64
        ):
            raise StagingDeployError(
                f"successful source deploy {name} identity is invalid"
            )
    return value


def record_successful_source_deploy(
    state_dir: Path,
    *,
    source_repository_root: Path,
    source: SourceCheckout,
    identities: Mapping[str, object],
    signing: SigningMaterial,
    mode: str,
    live_evidence: Mapping[str, object],
) -> Path:
    state_path, signature_path = source_success_paths(state_dir)
    application = identities.get("application")
    deploy_host = identities.get("deploy_host")
    if not isinstance(application, Mapping) or not isinstance(deploy_host, Mapping):
        raise StagingDeployError("source deploy identities are incomplete")
    bundle = deploy_host.get("bundle")
    if not isinstance(bundle, Mapping):
        raise StagingDeployError("source deploy bundle identity is incomplete")
    value = {
        "schema_version": 1,
        "status": "PASSED",
        "mode": mode,
        "source_repository_root": str(source_repository_root),
        "prepared_repository_root": str(source.repository_root),
        "source": {
            "git_commit": source.git_commit,
            "fingerprint": source.fingerprint,
            "snapshot": source.snapshot,
            "isolated": source.isolated,
        },
        "identities": {
            "application": {"sha256": application.get("sha256")},
            "deploy_host": {
                "sha256": deploy_host.get("sha256"),
                "bundle": {"sha256": bundle.get("sha256")},
            },
        },
        "site_file": str(state_dir / "site.yaml"),
        "live": dict(live_evidence),
    }
    temporary_state = state_path.with_suffix(".tmp")
    temporary_signature = signature_path.with_suffix(".tmp")
    temporary_state.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary_state.chmod(0o600)
    temporary_signature.unlink(missing_ok=True)
    _run(
        [
            "cosign",
            "sign-blob",
            "--yes",
            "--key",
            str(signing.private_key),
            "--bundle",
            str(temporary_signature),
            str(temporary_state),
        ],
        cwd=state_dir,
        env={**os.environ, "COSIGN_PASSWORD": signing.password},
        capture=True,
    )
    temporary_signature.chmod(0o600)
    os.replace(temporary_state, state_path)
    os.replace(temporary_signature, signature_path)
    return state_path


def classify_source_deploy(
    previous: Mapping[str, object] | None,
    current: Mapping[str, object],
    *,
    source: SourceCheckout,
    site_exists: bool,
    live_matches: bool = False,
    profile_change_pending: bool = False,
) -> str:
    # A Runtime Profile template that differs from the live Profile needs the
    # release engine's plan/approve stop, whatever the source identities say:
    # DEPLOY_HOST_ONLY, QUALITY_ONLY and UNCHANGED all apply no release and
    # would record the pending change as a success.
    if profile_change_pending:
        return "APPLICATION_RELEASE"
    if previous is None or not site_exists:
        return "APPLICATION_RELEASE"
    identities = previous.get("identities")
    previous_source = previous.get("source")
    if not isinstance(identities, dict) or not isinstance(previous_source, dict):
        return "APPLICATION_RELEASE"
    application = current.get("application")
    deploy_host = current.get("deploy_host")
    previous_application = identities.get("application")
    previous_deploy_host = identities.get("deploy_host")
    if not all(
        isinstance(item, dict)
        for item in (
            application,
            deploy_host,
            previous_application,
            previous_deploy_host,
        )
    ):
        return "APPLICATION_RELEASE"
    assert isinstance(application, dict)
    assert isinstance(deploy_host, dict)
    assert isinstance(previous_application, dict)
    assert isinstance(previous_deploy_host, dict)
    if application.get("sha256") != previous_application.get("sha256"):
        return "APPLICATION_RELEASE"
    if deploy_host.get("sha256") != previous_deploy_host.get("sha256"):
        return "DEPLOY_HOST_ONLY"
    if previous_source.get("fingerprint") == source.fingerprint and live_matches:
        return "UNCHANGED"
    if previous_source.get("fingerprint") == source.fingerprint:
        return "APPLICATION_RELEASE"
    return "QUALITY_ONLY"


def _impact_base(
    repository_root: Path,
    previous: Mapping[str, object],
    fallback: str,
) -> str:
    source = previous.get("source")
    candidate = str(source.get("git_commit") or "") if isinstance(source, dict) else ""
    if candidate:
        completed = subprocess.run(
            ["git", "rev-parse", "--verify", f"{candidate}^{{commit}}"],
            cwd=repository_root,
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if completed.returncode == 0:
            return candidate
    return fallback


def run_source_impact_gate(
    *,
    repository_root: Path,
    state_dir: Path,
    source: SourceCheckout,
    previous: Mapping[str, object],
    fallback_base: str,
) -> dict[str, object]:
    output = state_dir / "source-gates" / source.fingerprint / "impact-plan.json"
    output.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    output.parent.chmod(0o700)
    base = _impact_base(repository_root, previous, fallback_base)
    selector = repository_root / "scripts/select-affected-tests.py"
    environment = tool_cache_environment(state_dir)
    raw = _run(
        [
            sys.executable,
            str(selector),
            "--base",
            base,
            "--format",
            "json",
            "--write-plan",
            str(output),
        ],
        cwd=repository_root,
        capture=True,
        env=environment,
    )
    try:
        plan = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise StagingDeployError("source impact gate plan is invalid") from exc
    if not isinstance(plan, dict):
        raise StagingDeployError("source impact gate plan must be an object")
    _run(
        [
            sys.executable,
            str(selector),
            "--base",
            base,
            "--read-plan",
            str(output),
            "--execute",
        ],
        cwd=repository_root,
        env=environment,
    )
    return plan


def run_admin_preflight(
    *,
    repository_root: Path,
    state_dir: Path,
    venv: Path,
    lock_fd: int | None = None,
) -> None:
    environment = (
        {**os.environ, SITE_OPERATION_LOCK_FD_ENV: str(lock_fd)}
        if lock_fd is not None
        else None
    )
    _run(
        [
            str(venv / "bin/gpu-fault-admin"),
            "preflight",
            "--state-dir",
            str(state_dir),
        ],
        cwd=repository_root,
        env=environment,
        pass_fds=(lock_fd,) if lock_fd is not None else (),
    )


def _link_or_copy(source: str, destination: str) -> str:
    path = Path(source)
    if path.suffix == ".whl" or path.name.endswith(".tar.gz"):
        try:
            os.link(source, destination)
            return destination
        except OSError:
            pass
    shutil.copy2(source, destination)
    return destination


def prepare_deploy_host_only_checkout(
    *,
    repository_root: Path,
    state_dir: Path,
    signing: SigningMaterial,
) -> bytes:
    site_file = state_dir / "site.yaml"
    original = site_file.read_bytes()
    try:
        document = yaml.safe_load(original)
        configured = document["spec"]["repositoryRoot"]
    except (KeyError, TypeError, yaml.YAMLError) as exc:
        raise StagingDeployError(
            "managed site has no valid repository root for host-only update"
        ) from exc
    previous_root = Path(str(configured)).expanduser()
    if not previous_root.is_absolute():
        previous_root = site_file.parent / previous_root
    previous_root = previous_root.resolve()
    previous_dist = previous_root / "dist"
    if not previous_dist.is_dir():
        raise StagingDeployError("managed site release artifacts are missing")
    current_dist = repository_root / "dist"
    if previous_root != repository_root:
        if current_dist.exists():
            shutil.rmtree(current_dist)
        shutil.copytree(
            previous_dist,
            current_dist,
            copy_function=_link_or_copy,
        )
    _run(
        [
            sys.executable,
            str(repository_root / "scripts/verify-release-attestation.py"),
            "--attestation",
            str(current_dist / "current-attestation.json"),
            "--bundle",
            str(current_dist / "current-attestation.bundle.json"),
            "--cosign-key",
            str(signing.public_key),
            "--allow-staging-release",
        ],
        cwd=repository_root,
    )
    document["spec"]["repositoryRoot"] = str(repository_root)
    temporary = site_file.with_suffix(".tmp")
    temporary.write_text(
        yaml.safe_dump(document, sort_keys=False),
        encoding="utf-8",
    )
    temporary.chmod(0o600)
    os.replace(temporary, site_file)
    return original


def run_admin_deploy(
    *,
    repository_root: Path,
    state_dir: Path,
    venv: Path,
    cpu_cluster_arn: str,
    gpu_cluster_arns: Sequence[str],
    admin_email: str,
    staging_only_release: bool,
    impact_base: str,
    lock_fd: int,
) -> None:
    command = [
        str(venv / "bin/gpu-fault-admin"),
        "deploy",
        "--cpu-cluster-arn",
        cpu_cluster_arn,
    ]
    for arn in gpu_cluster_arns:
        command.extend(("--gpu-cluster-arn", arn))
    command.extend(
        (
            "--state-dir",
            str(state_dir),
            "--admin-email",
            admin_email,
            "--repo-root",
            str(repository_root),
            "--impact-base",
            impact_base,
            "--prepared-source-release",
        )
    )
    if staging_only_release:
        command.append("--staging-only-release")
    # The admin deploy re-runs the static gates inside the prepared snapshot,
    # which by construction carries no tool cache; pointing them at the state
    # directory is what makes the second run of an identical analysis cheap.
    _run(
        command,
        cwd=repository_root,
        env={
            **tool_cache_environment(state_dir),
            SITE_OPERATION_LOCK_FD_ENV: str(lock_fd),
        },
        pass_fds=(lock_fd,),
    )


def _ensure_deploy_host(
    state_dir: Path,
    *,
    source: SourceCheckout,
    signing: SigningMaterial,
    artifacts: DeployHostArtifacts,
    lock_fd: int,
) -> tuple[bool, Path]:
    """The signed bundle and the venv installed from it, built once if missing.

    Returns whether the bundle was reused, and the venv the admin CLI runs from.
    """

    wheelhouse_cache = deploy_host_wheelhouse_cache(
        state_dir,
        repository_root=source.repository_root,
    )
    bundle_reused = ensure_deploy_host_bundle(
        artifacts,
        repository_root=source.repository_root,
        signing=signing,
        wheelhouse_cache=wheelhouse_cache,
    )
    venv = ensure_deploy_host_venv(
        state_dir,
        repository_root=source.repository_root,
        signing=signing,
        artifacts=artifacts,
        lock_fd=lock_fd,
    )
    return bundle_reused, venv


def apply_source_deploy(
    *,
    arguments: argparse.Namespace,
    repository_root: Path,
    state_dir: Path,
    source: SourceCheckout,
    identities: Mapping[str, object],
    signing: SigningMaterial,
    venv: Path,
    prepared_mode: str,
    trusted_ci_candidate: Mapping[str, object] | None,
    lock_fd: int,
    live_evidence: Mapping[str, object] | None = None,
) -> tuple[str, dict[str, object]]:
    """Apply ``prepared_mode`` and record the success that authorizes the next run.

    ``lock_fd`` is the site operation lock the caller holds across its own
    classification, and it is required. That is what removes a
    ``gpu-fault-admin status`` -- about 45 seconds -- from every deploy: nothing
    can change the site between the reading that produced ``prepared_mode`` and
    this application, so there is no re-classification to make and no second
    decision path that could disagree with the caller's.

    ``live_evidence`` is that same reading. UNCHANGED applies nothing, so it is
    still true afterwards and is recorded as the success. An APPLICATION_RELEASE
    ends with the release driver's own ``release-summary.json`` and COMPLETED
    ``state.json``, and the success record is built from those files rather
    than from a second ``status`` that would read the same facts back at the
    same cost. Only a deploy-host-only change, which runs no release driver,
    reads the site again afterwards.
    """

    mode = prepared_mode
    evidence: dict[str, object] | None = (
        dict(live_evidence) if live_evidence is not None else None
    )
    record_source_deploy_state(
        state_dir, source_repository_root=repository_root, source=source
    )
    if trusted_ci_candidate is not None and mode == "APPLICATION_RELEASE":
        record_trusted_ci_candidate(
            source.repository_root,
            state_dir=state_dir,
            signing=signing,
            candidate=trusted_ci_candidate,
        )
    if mode == "DEPLOY_HOST_ONLY":
        original_site = prepare_deploy_host_only_checkout(
            repository_root=source.repository_root,
            state_dir=state_dir,
            signing=signing,
        )
        try:
            run_admin_preflight(
                repository_root=source.repository_root,
                state_dir=state_dir,
                venv=venv,
                lock_fd=lock_fd,
            )
        except Exception:
            restore_site_file(state_dir, original_site)
            raise
    elif mode == "APPLICATION_RELEASE":
        run_admin_deploy(
            repository_root=source.repository_root,
            state_dir=state_dir,
            venv=venv,
            cpu_cluster_arn=arguments.cpu_cluster_arn,
            gpu_cluster_arns=tuple(arguments.gpu_cluster_arn),
            admin_email=arguments.admin_email,
            staging_only_release=source.snapshot,
            impact_base=arguments.base.strip(),
            lock_fd=lock_fd,
        )
    # UNCHANGED deployed nothing, so the evidence read before the apply still
    # describes the site. An APPLICATION_RELEASE just wrote the summary the
    # success record needs; anything else changed the site and reads it again.
    if mode == "APPLICATION_RELEASE":
        evidence = release_record_evidence(state_dir)
    elif mode != "UNCHANGED":
        evidence = None
    if evidence is None:
        evidence = collect_live_deploy_evidence(
            repository_root=source.repository_root,
            state_dir=state_dir,
            venv=venv,
            lock_fd=lock_fd,
        )
    record_successful_source_deploy(
        state_dir,
        source_repository_root=repository_root,
        source=source,
        identities=identities,
        signing=signing,
        mode=mode,
        live_evidence=evidence,
    )
    return mode, evidence


def release_record_evidence(state_dir: Path) -> dict[str, object] | None:
    """The success record from the release driver's files, or None to read live.

    A record that is not a committed, verified, NOOP-next release -- the driver
    completed with the summary unavailable, say -- is not an error here: the
    live reading it replaces is still there to fall back on.
    """

    try:
        return live_evidence_from_release_record(state_dir=state_dir)
    except LiveEvidenceError as exc:
        print(
            f"+ release record is not usable as evidence ({exc}); reading status",
            file=sys.stderr,
            flush=True,
        )
        return None


def pre_deploy_reading(
    *,
    previous: Mapping[str, object] | None,
    state_dir: Path,
    source: SourceCheckout,
    venv: Path,
    lock_fd: int,
) -> tuple[dict[str, object] | None, dict[str, object] | None, bool]:
    """``(status report, live evidence, profile change pending)`` under the lock.

    One quick ``status``; the report feeds the consent refusals and the
    evidence feeds the classification. A site that has no previous success, no
    ``site.yaml`` or no admin CLI has nothing to read.
    """

    if not (
        previous is not None
        and (state_dir / "site.yaml").is_file()
        and (venv / "bin/gpu-fault-admin").is_file()
    ):
        return None, None, False
    report: dict[str, object] | None = None
    try:
        report = read_live_status(
            repository_root=source.repository_root,
            state_dir=state_dir,
            venv=venv,
            lock_fd=lock_fd,
        )
        evidence = collect_live_deploy_evidence(
            repository_root=source.repository_root,
            state_dir=state_dir,
            venv=venv,
            lock_fd=lock_fd,
            report=report,
        )
    except RuntimeProfileChangePending as exc:
        print(f"+ {exc}; running the application release", file=sys.stderr)
        return report, None, True
    except (LiveEvidenceError, StagingDeployError):
        return report, None, False
    return report, evidence, False


def _prepare_deploy_host(
    *,
    mode: str,
    state_dir: Path,
    source: SourceCheckout,
    signing: SigningMaterial,
    artifacts: DeployHostArtifacts,
    venv: Path,
    lock_fd: int,
) -> tuple[str, bool, Path]:
    """``(mode, bundle reused, venv)``: the deploy host, built once if missing."""

    if mode in {"APPLICATION_RELEASE", "DEPLOY_HOST_ONLY"}:
        reused, venv = _ensure_deploy_host(
            state_dir,
            source=source,
            signing=signing,
            artifacts=artifacts,
            lock_fd=lock_fd,
        )
        return mode, reused, venv
    if not (venv / "bin/gpu-fault-admin").is_file():
        # Rebuilding the deploy host means there is no admin CLI to apply a
        # quality-only pass with, so this is an application release -- and
        # ``prepared_mode`` has to say so too, or the apply would classify
        # QUALITY_ONLY again and deploy nothing while the release path here
        # has already skipped the impact gate.
        reused, venv = _ensure_deploy_host(
            state_dir,
            source=source,
            signing=signing,
            artifacts=artifacts,
            lock_fd=lock_fd,
        )
        return "APPLICATION_RELEASE", reused, venv
    return mode, True, venv


def deploy(arguments: argparse.Namespace) -> dict[str, object]:
    repository_root = arguments.repo_root.expanduser().resolve()
    state_dir = arguments.state_dir.expanduser().resolve()
    if not arguments.base.strip():
        raise StagingDeployError("impact test base must not be empty")
    try:
        state_dir.relative_to(repository_root)
    except ValueError:
        pass
    else:
        raise StagingDeployError(
            "staging state directory must be outside the Git repository"
        )
    # Created before the gate runs so the gate has somewhere to record its
    # verdict; the directory is empty and 0700 either way, and nothing is
    # deployed from it until the gate has passed.
    state_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    state_dir.chmod(0o700)
    first_deploy = not (state_dir / "site.yaml").is_file()
    site_inputs = managed_site_inputs(state_dir) or {}
    cpu_cluster_arn, gpu_cluster_arns, admin_email = resolve_deploy_inputs(
        arguments, state_dir=state_dir
    )
    # Both confirmation mails go out and are checked here, in the first minute,
    # before the source scan, the gates and the build.
    email_confirmation = precheck_email_confirmations(
        state_dir=state_dir,
        cpu_cluster_arn=cpu_cluster_arn,
        admin_email=admin_email,
        wait_minutes=int(getattr(arguments, "wait_for_email_confirmation", 0) or 0),
        rerun_command=deploy_rerun_command(
            state_dir=state_dir,
            first_deploy=first_deploy,
            cpu_cluster_arn=cpu_cluster_arn,
            gpu_cluster_arns=gpu_cluster_arns,
            admin_email=admin_email,
        ),
    )
    verdict_cache = public_release_verdict_cache(state_dir)
    validate_source_checkout(repository_root, verdict_cache=verdict_cache)
    source = prepare_source_checkout(
        repository_root,
        state_dir=state_dir,
    )
    if source.repository_root != repository_root:
        validate_source_checkout(source.repository_root, verdict_cache=verdict_cache)
    identities = source_deploy_identity(source.repository_root)
    signing = ensure_signing_material(
        state_dir,
        repository_root=source.repository_root,
    )
    # The site lock is taken before the first ``gpu-fault-admin status`` and held
    # through the apply: the classification below is read from the live site, and
    # a classification that another operation can invalidate has to be re-read,
    # which is the 45-second call this ordering deletes.
    with site_operation_lock(state_dir, wait=True) as lock_fd:
        previous = load_successful_source_deploy(state_dir, signing=signing)
        venv = state_dir / "deployer-venv"
        report, live_evidence, profile_change_pending = pre_deploy_reading(
            previous=previous,
            state_dir=state_dir,
            source=source,
            venv=venv,
            lock_fd=lock_fd,
        )
        refusal = early_consent_refusal(
            report,
            source=source,
            auto_rollback=bool(site_inputs.get("auto_rollback", True)),
        )
        if refusal:
            raise StagingDeployError(refusal)
        mode = classify_source_deploy(
            previous,
            identities,
            source=source,
            site_exists=not first_deploy,
            live_matches=successful_source_live_matches(previous, live_evidence),
            profile_change_pending=profile_change_pending,
        )
        trusted_ci_candidate = (
            restore_trusted_ci_candidate(
                source.repository_root,
                staging_only=source.snapshot,
            )
            if mode == "APPLICATION_RELEASE"
            else None
        )
        deploy_host = identities["deploy_host"]
        assert isinstance(deploy_host, dict)
        bundle_identity = deploy_host["bundle"]
        assert isinstance(bundle_identity, dict)
        artifacts = deploy_host_artifacts(
            state_dir,
            payload_identity_sha256=str(bundle_identity["sha256"]),
        )
        mode, bundle_reused, venv = _prepare_deploy_host(
            mode=mode,
            state_dir=state_dir,
            source=source,
            signing=signing,
            artifacts=artifacts,
            venv=venv,
            lock_fd=lock_fd,
        )
        prepared_mode = mode

        gated = mode in {"DEPLOY_HOST_ONLY", "QUALITY_ONLY"}
        impact_plan: dict[str, object] | None = None
        if gated and trusted_ci_candidate is None:
            assert previous is not None
            impact_plan = run_source_impact_gate(
                repository_root=source.repository_root,
                state_dir=state_dir,
                source=source,
                previous=previous,
                fallback_base=arguments.base.strip(),
            )
        elif gated:
            impact_plan = {
                "source": "signed_main_ci_candidate",
                "run_id": trusted_ci_candidate["run_id"],
            }
        mode, live_evidence = apply_source_deploy(
            arguments=arguments,
            repository_root=repository_root,
            state_dir=state_dir,
            source=source,
            identities=identities,
            signing=signing,
            venv=venv,
            prepared_mode=prepared_mode,
            trusted_ci_candidate=trusted_ci_candidate,
            lock_fd=lock_fd,
            live_evidence=live_evidence,
        )
        try:
            prune_source_snapshots(
                state_dir,
                source_repository_root=repository_root,
                current=source.repository_root,
            )
        except (OSError, StagingDeployError) as exc:
            # The deploy is applied and its success is already signed. Failing it
            # now over disk hygiene would report a failure for a site that is
            # fully deployed, so this is reported and left for the next run.
            print(
                f"staging-deploy: source snapshots not pruned: {exc}", file=sys.stderr
            )
        return {
            "schema_version": 1,
            "git_commit": source.git_commit,
            "source_fingerprint": source.fingerprint,
            "source_checkout": str(source.repository_root),
            "source_snapshot": source.snapshot,
            "source_isolated": source.isolated,
            "deploy_mode": mode,
            "application_identity_sha256": str(
                cast(Mapping[str, object], identities["application"])["sha256"]
            ),
            "deploy_host_identity_sha256": str(deploy_host["sha256"]),
            "impact_plan": impact_plan,
            "trusted_ci_candidate": trusted_ci_candidate,
            "state_dir": str(state_dir),
            "deploy_host_bundle": str(artifacts.archive),
            "deploy_host_bundle_reused": bundle_reused,
            "deploy_host_venv": str(venv),
            "site_file": str(state_dir / "site.yaml"),
            "cpu_cluster_arn": cpu_cluster_arn,
            "gpu_cluster_arns": list(gpu_cluster_arns),
            "email_confirmation": email_confirmation.as_dict(),
            "live_release": live_release_block(report),
            "next_deploy": next_deploy_block(report),
        }


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        description=(
            "Build or reuse a signed deploy-host environment, then bootstrap "
            "or upgrade one staging site with the same command"
        )
    )
    # Optional on a managed site: ``site.yaml`` supplies them (an upgrade is
    # ``--state-dir`` alone); a first deploy must give all three.
    value.add_argument("--cpu-cluster-arn")
    value.add_argument("--gpu-cluster-arn", action="append", default=[])
    value.add_argument("--state-dir", required=True, type=Path)
    value.add_argument("--admin-email")
    value.add_argument(
        WAIT_FLAG,
        dest="wait_for_email_confirmation",
        type=int,
        default=0,
        metavar="MINUTES",
        help=(
            "poll the SES sender verification and the SNS alert subscription for "
            "up to MINUTES before giving up (default 0: check once and stop)"
        ),
    )
    value.add_argument("--base", default="origin/main")
    value.add_argument("--repo-root", type=Path, default=ROOT)
    value.add_argument("--quiet", action="store_true")
    return value


def main(arguments: Sequence[str] | None = None) -> int:
    parsed = parser().parse_args(arguments)
    try:
        result = deploy(parsed)
    except (
        LiveEvidenceError,
        OSError,
        StagingDeployError,
        subprocess.SubprocessError,
    ) as exc:
        return report_failure("staging-deploy", exc)
    if not parsed.quiet:
        print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

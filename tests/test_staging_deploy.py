from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

import pytest
import yaml

from scripts import staging_deploy


def _signing_material(root: Path) -> staging_deploy.SigningMaterial:
    signing = root / "release-signing"
    signing.mkdir(parents=True)
    private_key = signing / "cosign.key"
    public_key = signing / "cosign.pub"
    password_file = signing / "cosign.password"
    private_key.write_text("private", encoding="utf-8")
    public_key.write_text("public", encoding="utf-8")
    password_file.write_text("password", encoding="utf-8")
    private_key.chmod(0o600)
    password_file.chmod(0o600)
    public_key.chmod(0o644)
    return staging_deploy.SigningMaterial(
        private_key=private_key,
        public_key=public_key,
        password_file=password_file,
        password="password",
    )


def _source_identities() -> dict[str, object]:
    return {
        "schema_version": 1,
        "sha256": "f" * 64,
        "application": {"sha256": "a" * 64},
        "deploy_host": {"sha256": "b" * 64, "bundle": {"sha256": "c" * 64}},
    }


def _live_evidence() -> dict[str, object]:
    return {
        "site_sha256": "d" * 64,
        "release_id": "release-a",
        "state_sha256": "e" * 64,
        "phase": "complete",
        "transaction_committed": True,
        "next_deploy_kind": "NOOP",
    }


def test_signing_material_is_generated_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    commands: list[list[str]] = []
    monkeypatch.setattr(
        staging_deploy.secrets, "token_urlsafe", lambda _size: "generated-password"
    )

    def run(arguments, **_kwargs):
        command = [str(item) for item in arguments]
        commands.append(command)
        prefix = Path(command[command.index("--output-key-prefix") + 1])
        prefix.with_suffix(".key").write_text("private", encoding="utf-8")
        prefix.with_suffix(".pub").write_text("public", encoding="utf-8")
        return ""

    monkeypatch.setattr(staging_deploy, "_run", run)

    first = staging_deploy.ensure_signing_material(tmp_path, repository_root=tmp_path)
    second = staging_deploy.ensure_signing_material(tmp_path, repository_root=tmp_path)

    assert len(commands) == 1
    assert first == second
    assert first.private_key.stat().st_mode & 0o777 == 0o600
    assert first.password_file.stat().st_mode & 0o777 == 0o600
    assert first.public_key.stat().st_mode & 0o777 == 0o644
    assert first.password == "generated-password"


def test_empty_password_file_is_replaced_before_key_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    signing = tmp_path / "release-signing"
    signing.mkdir()
    password_file = signing / "cosign.password"
    password_file.write_text("", encoding="utf-8")
    password_file.chmod(0o600)
    monkeypatch.setattr(
        staging_deploy.secrets, "token_urlsafe", lambda _size: "generated-password"
    )

    def run(arguments, **_kwargs):
        command = [str(item) for item in arguments]
        prefix = Path(command[command.index("--output-key-prefix") + 1])
        prefix.with_suffix(".key").write_text("private", encoding="utf-8")
        prefix.with_suffix(".pub").write_text("public", encoding="utf-8")
        return ""

    monkeypatch.setattr(staging_deploy, "_run", run)

    material = staging_deploy.ensure_signing_material(
        tmp_path, repository_root=tmp_path
    )

    assert material.password == "generated-password"
    assert password_file.read_text(encoding="utf-8") == "generated-password"


def test_existing_key_pair_persists_prompted_password(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    signing = tmp_path / "release-signing"
    signing.mkdir()
    private_key = signing / "cosign.key"
    public_key = signing / "cosign.pub"
    password_file = signing / "cosign.password"
    private_key.write_text("private", encoding="utf-8")
    public_key.write_text("public", encoding="utf-8")
    password_file.write_text("", encoding="utf-8")
    private_key.chmod(0o600)
    public_key.chmod(0o644)
    password_file.chmod(0o600)
    monkeypatch.setenv("COSIGN_PASSWORD", "password")

    material = staging_deploy.ensure_signing_material(
        tmp_path, repository_root=tmp_path
    )

    assert material.password == "password"
    assert material.password_file.read_text(encoding="utf-8") == "password"
    assert material.password_file.stat().st_mode & 0o777 == 0o600


def test_existing_key_pair_without_password_fails_closed(tmp_path: Path) -> None:
    signing = tmp_path / "release-signing"
    signing.mkdir()
    private_key = signing / "cosign.key"
    public_key = signing / "cosign.pub"
    private_key.write_text("private", encoding="utf-8")
    public_key.write_text("public", encoding="utf-8")
    private_key.chmod(0o600)
    public_key.chmod(0o644)

    with pytest.raises(
        staging_deploy.StagingDeployError, match="existing Cosign key requires"
    ):
        staging_deploy.ensure_signing_material(tmp_path, repository_root=tmp_path)


def test_existing_bundle_is_reused_without_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifacts = staging_deploy.DeployHostArtifacts(
        archive=tmp_path / "bundle.tar.gz",
        checksum=tmp_path / "bundle.tar.gz.sha256",
        signature_bundle=tmp_path / "bundle.sigstore.json",
    )
    for path in (artifacts.archive, artifacts.checksum, artifacts.signature_bundle):
        path.write_text("value", encoding="utf-8")

    monkeypatch.setattr(
        staging_deploy,
        "_run",
        lambda *_args, **_kwargs: pytest.fail("reused bundle was rebuilt"),
    )

    assert staging_deploy.ensure_deploy_host_bundle(
        artifacts,
        repository_root=tmp_path,
        signing=_signing_material(tmp_path / "state"),
    ), "existing complete deploy-host bundle was not reused"


def test_deploy_host_artifacts_are_content_addressed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(staging_deploy, "bundle_platform_id", lambda: "test-platform")

    first = staging_deploy.deploy_host_artifacts(
        tmp_path, payload_identity_sha256="a" * 64
    )
    repeated = staging_deploy.deploy_host_artifacts(
        tmp_path, payload_identity_sha256="a" * 64
    )
    changed = staging_deploy.deploy_host_artifacts(
        tmp_path, payload_identity_sha256="b" * 64
    )

    assert first == repeated
    assert first.archive.parent.name == "a" * 64
    assert changed.archive.parent != first.archive.parent


def test_source_deploy_classification_separates_application_and_host_changes() -> None:
    source = staging_deploy.SourceCheckout(
        repository_root=Path("/repository"),
        git_commit="b" * 40,
        fingerprint="b" * 64,
        snapshot=False,
        isolated=True,
    )
    current = _source_identities()
    previous = {"identities": _source_identities(), "source": {"fingerprint": "a" * 64}}

    assert (
        staging_deploy.classify_source_deploy(
            previous, current, source=source, site_exists=True
        )
        == "QUALITY_ONLY"
    )
    host_changed = _source_identities()
    host_changed["deploy_host"] = {"sha256": "d" * 64, "bundle": {"sha256": "e" * 64}}
    assert (
        staging_deploy.classify_source_deploy(
            previous, host_changed, source=source, site_exists=True
        )
        == "DEPLOY_HOST_ONLY"
    )
    application_changed = _source_identities()
    application_changed["application"] = {"sha256": "e" * 64}
    assert (
        staging_deploy.classify_source_deploy(
            previous, application_changed, source=source, site_exists=True
        )
        == "APPLICATION_RELEASE"
    )
    assert (
        staging_deploy.classify_source_deploy(
            previous, current, source=source, site_exists=False
        )
        == "APPLICATION_RELEASE"
    )
    previous["source"] = {"fingerprint": source.fingerprint}
    assert (
        staging_deploy.classify_source_deploy(
            previous, current, source=source, site_exists=True
        )
        == "APPLICATION_RELEASE"
    )
    assert (
        staging_deploy.classify_source_deploy(
            previous, current, source=source, site_exists=True, live_matches=True
        )
        == "UNCHANGED"
    )


def test_deploy_host_wheelhouse_cache_is_lock_scoped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = tmp_path / "repo"
    requirements = repository / "requirements"
    requirements.mkdir(parents=True)
    (requirements / "build.lock").write_text("build-a\n", encoding="utf-8")
    (requirements / "deploy-host.lock").write_text("host-a\n", encoding="utf-8")
    monkeypatch.setattr(staging_deploy, "bundle_platform_id", lambda: "test-platform")

    first = staging_deploy.deploy_host_wheelhouse_cache(
        tmp_path / "state", repository_root=repository
    )
    (requirements / "deploy-host.lock").write_text("host-b\n", encoding="utf-8")
    second = staging_deploy.deploy_host_wheelhouse_cache(
        tmp_path / "state", repository_root=repository
    )

    assert first != second
    assert first.parent == second.parent
    assert first.name.startswith("test-platform-"), (
        "deploy-host wheelhouse cache key omitted the platform"
    )
    assert first.stat().st_mode & 0o777 == 0o700


def test_deploy_host_venv_is_bound_to_managed_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = tmp_path / "state"
    venv = state / "deployer-venv"
    artifacts = staging_deploy.DeployHostArtifacts(
        archive=state / "bundle.tar.gz",
        checksum=state / "bundle.tar.gz.sha256",
        signature_bundle=state / "bundle.sigstore.json",
    )

    def run(_arguments, **_kwargs):
        venv.joinpath("bin").mkdir(parents=True)
        venv.joinpath("bin/gpu-fault-admin").write_text("", encoding="utf-8")
        return ""

    monkeypatch.setattr(staging_deploy, "_run", run)

    result = staging_deploy.ensure_deploy_host_venv(
        state,
        repository_root=tmp_path,
        signing=_signing_material(state),
        artifacts=artifacts,
    )

    binding = json.loads(
        result.joinpath("gpu-fault-managed-state-dir.json").read_text(encoding="utf-8")
    )
    assert binding == {"schema_version": 1, "state_dir": str(state.resolve())}
    assert (
        result.joinpath("gpu-fault-managed-state-dir.json").stat().st_mode & 0o777
        == 0o600
    )


def test_deploy_host_only_checkout_reuses_signed_application_release(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    previous = tmp_path / "previous"
    current = tmp_path / "current"
    state = tmp_path / "state"
    for root in (previous, current, state):
        root.mkdir()
    previous_dist = previous / "dist"
    previous_dist.mkdir()
    for name in (
        "current-release.json",
        "current-attestation.json",
        "current-attestation.bundle.json",
    ):
        (previous_dist / name).write_text(name, encoding="utf-8")
    site = state / "site.yaml"
    site.write_text(
        yaml.safe_dump(
            {"kind": "RegionalSite", "spec": {"repositoryRoot": str(previous)}},
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    site.chmod(0o600)
    commands: list[list[str]] = []
    monkeypatch.setattr(
        staging_deploy,
        "_run",
        lambda arguments, **_kwargs: commands.append(list(arguments)) or "",
    )

    original = staging_deploy.prepare_deploy_host_only_checkout(
        repository_root=current, state_dir=state, signing=_signing_material(state)
    )

    document = yaml.safe_load(site.read_text(encoding="utf-8"))
    assert document["spec"]["repositoryRoot"] == str(current)
    assert (current / "dist/current-release.json").read_text(encoding="utf-8") == (
        "current-release.json"
    )
    assert "--allow-staging-release" in commands[0]
    staging_deploy.restore_site_file(state, original)
    assert yaml.safe_load(site.read_text(encoding="utf-8"))["spec"][
        "repositoryRoot"
    ] == str(previous)


def test_partial_bundle_is_removed_and_rebuilt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifacts = staging_deploy.DeployHostArtifacts(
        archive=tmp_path / "bundle.tar.gz",
        checksum=tmp_path / "bundle.tar.gz.sha256",
        signature_bundle=tmp_path / "bundle.sigstore.json",
    )
    artifacts.archive.write_text("partial", encoding="utf-8")

    commands: list[list[str]] = []

    def run(arguments, **_kwargs):
        commands.append([str(item) for item in arguments])
        for path in (artifacts.archive, artifacts.checksum, artifacts.signature_bundle):
            path.write_text("rebuilt", encoding="utf-8")
        return ""

    monkeypatch.setattr(staging_deploy, "_run", run)

    assert (
        staging_deploy.ensure_deploy_host_bundle(
            artifacts,
            repository_root=tmp_path,
            signing=_signing_material(tmp_path / "state"),
            wheelhouse_cache=tmp_path / "wheel-cache",
        )
        is False
    )
    assert artifacts.archive.read_text(encoding="utf-8") == "rebuilt"
    assert f"DEPLOY_HOST_WHEELHOUSE={tmp_path / 'wheel-cache'}" in commands[0]


def test_deploy_uses_same_orchestration_for_first_and_later_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = tmp_path / "repo"
    repository.mkdir()
    state = tmp_path / "state"
    signing = _signing_material(state)
    artifacts = staging_deploy.DeployHostArtifacts(
        archive=state / "bundle.tar.gz",
        checksum=state / "bundle.tar.gz.sha256",
        signature_bundle=state / "bundle.sigstore.json",
    )
    venv = state / "deployer-venv"
    calls: list[tuple[str, object]] = []
    source = staging_deploy.SourceCheckout(
        repository_root=repository,
        git_commit="a" * 40,
        fingerprint="a" * 64,
        snapshot=False,
        isolated=True,
    )
    monkeypatch.setattr(
        staging_deploy, "prepare_source_checkout", lambda *_args, **_kwargs: source
    )
    monkeypatch.setattr(
        staging_deploy,
        "validate_source_checkout",
        lambda root: calls.append(("scan", root)),
    )
    monkeypatch.setattr(
        staging_deploy, "ensure_signing_material", lambda *_args, **_kwargs: signing
    )
    monkeypatch.setattr(
        staging_deploy,
        "source_deploy_identity",
        lambda *_args, **_kwargs: _source_identities(),
    )
    monkeypatch.setattr(
        staging_deploy, "restore_trusted_ci_candidate", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        staging_deploy, "load_successful_source_deploy", lambda *_args, **_kwargs: None
    )

    class ApplyLock:
        def __enter__(self):
            calls.append(("lock-enter", None))
            return 17

        def __exit__(self, *_args):
            calls.append(("lock-exit", None))

    monkeypatch.setattr(
        staging_deploy, "site_operation_lock", lambda *_args, **_kwargs: ApplyLock()
    )
    monkeypatch.setattr(
        staging_deploy,
        "record_successful_source_deploy",
        lambda *_args, **_kwargs: (
            calls.append(("success", None)) or state / "source-deploy-success.json"
        ),
    )
    monkeypatch.setattr(
        staging_deploy, "deploy_host_artifacts", lambda *_args, **_kwargs: artifacts
    )
    monkeypatch.setattr(
        staging_deploy,
        "deploy_host_wheelhouse_cache",
        lambda *_args, **_kwargs: state / "wheelhouse",
    )
    monkeypatch.setattr(
        staging_deploy, "ensure_deploy_host_bundle", lambda *_args, **_kwargs: True
    )
    monkeypatch.setattr(
        staging_deploy, "ensure_deploy_host_venv", lambda *_args, **_kwargs: venv
    )
    monkeypatch.setattr(
        staging_deploy,
        "run_admin_deploy",
        lambda **kwargs: calls.append(("deploy", kwargs)),
    )
    monkeypatch.setattr(
        staging_deploy,
        "collect_live_deploy_evidence",
        lambda **_kwargs: _live_evidence(),
    )
    arguments = argparse.Namespace(
        repo_root=repository,
        state_dir=state,
        cpu_cluster_arn="arn:aws:eks:us-east-1:123456789012:cluster/cpu",
        gpu_cluster_arn=["arn:aws:eks:us-east-1:123456789012:cluster/gpu"],
        admin_email="operations@example.com",
        base="origin/main",
    )

    result = staging_deploy.deploy(arguments)

    assert result["deploy_host_bundle_reused"] is True
    assert result["source_snapshot"] is False
    assert result["source_isolated"] is True
    assert calls[0] == ("scan", repository)
    deploy_call = next(item for item in calls if item[0] == "deploy")
    assert deploy_call[1]["state_dir"] == state
    assert deploy_call[1]["gpu_cluster_arns"] == tuple(arguments.gpu_cluster_arn)
    assert deploy_call[1]["staging_only_release"] is False
    assert deploy_call[1]["impact_base"] == "origin/main"
    event_names = [name for name, _value in calls]
    assert event_names.index("lock-enter") < event_names.index("deploy"), (
        "application deploy started before the top-level apply lock"
    )
    assert event_names.index("deploy") < event_names.index("success"), (
        "source success authorization was written before application deploy"
    )
    assert event_names.index("success") < event_names.index("lock-exit"), (
        "source success authorization escaped the top-level apply lock"
    )
    state_value = json.loads(
        (state / staging_deploy.SOURCE_DEPLOY_STATE).read_text(encoding="utf-8")
    )
    assert state_value["source_repository_root"] == str(repository)
    assert state_value["release_ref"] == source.git_commit
    assert state_value["source_isolated"] is True


def test_unchanged_successful_source_does_not_query_ci_or_deploy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = tmp_path / "repo"
    repository.mkdir()
    state = tmp_path / "state"
    state.mkdir()
    (state / "site.yaml").write_text("kind: RegionalSite\n", encoding="utf-8")
    venv = state / "deployer-venv/bin"
    venv.mkdir(parents=True)
    (venv / "gpu-fault-admin").write_text("", encoding="utf-8")
    source = staging_deploy.SourceCheckout(
        repository_root=repository,
        git_commit="a" * 40,
        fingerprint="a" * 64,
        snapshot=False,
        isolated=True,
    )
    signing = _signing_material(state)
    previous = {
        "schema_version": 1,
        "status": "PASSED",
        "identities": _source_identities(),
        "source": {"fingerprint": source.fingerprint, "git_commit": source.git_commit},
        "live": _live_evidence(),
    }
    monkeypatch.setattr(
        staging_deploy, "prepare_source_checkout", lambda *_args, **_kwargs: source
    )
    monkeypatch.setattr(staging_deploy, "validate_source_checkout", lambda _root: None)
    monkeypatch.setattr(
        staging_deploy,
        "source_deploy_identity",
        lambda *_args, **_kwargs: _source_identities(),
    )
    monkeypatch.setattr(
        staging_deploy, "ensure_signing_material", lambda *_args, **_kwargs: signing
    )
    monkeypatch.setattr(
        staging_deploy,
        "load_successful_source_deploy",
        lambda *_args, **_kwargs: previous,
    )
    monkeypatch.setattr(
        staging_deploy,
        "restore_trusted_ci_candidate",
        lambda *_args, **_kwargs: pytest.fail(
            "unchanged source queried the main CI candidate"
        ),
    )
    monkeypatch.setattr(
        staging_deploy,
        "run_source_impact_gate",
        lambda *_args, **_kwargs: pytest.fail("unchanged source reran impact tests"),
    )
    monkeypatch.setattr(
        staging_deploy,
        "run_admin_deploy",
        lambda *_args, **_kwargs: pytest.fail(
            "unchanged source entered application deploy"
        ),
    )
    monkeypatch.setattr(
        staging_deploy,
        "record_successful_source_deploy",
        lambda *_args, **_kwargs: state / "source-deploy-success.json",
    )
    monkeypatch.setattr(
        staging_deploy,
        "collect_live_deploy_evidence",
        lambda **_kwargs: _live_evidence(),
    )

    result = staging_deploy.deploy(
        argparse.Namespace(
            repo_root=repository,
            state_dir=state,
            cpu_cluster_arn="cpu",
            gpu_cluster_arn=["gpu"],
            admin_email="operations@example.com",
            base="origin/main",
        )
    )

    assert result["deploy_mode"] == "UNCHANGED"
    assert result["trusted_ci_candidate"] is None


def test_staging_state_must_be_outside_repository(tmp_path: Path) -> None:
    repository = tmp_path / "repo"
    repository.mkdir()
    arguments = argparse.Namespace(
        repo_root=repository,
        state_dir=repository / "state",
        cpu_cluster_arn="cpu",
        gpu_cluster_arn=["gpu"],
        admin_email="operations@example.com",
        base="origin/main",
    )

    with pytest.raises(
        staging_deploy.StagingDeployError, match="outside the Git repository"
    ):
        staging_deploy.deploy(arguments)


def test_staging_impact_base_must_not_be_empty(tmp_path: Path) -> None:
    repository = tmp_path / "repo"
    repository.mkdir()
    arguments = argparse.Namespace(
        repo_root=repository,
        state_dir=tmp_path / "state",
        cpu_cluster_arn="cpu",
        gpu_cluster_arn=["gpu"],
        admin_email="operations@example.com",
        base=" ",
    )

    with pytest.raises(
        staging_deploy.StagingDeployError, match="impact test base must not be empty"
    ):
        staging_deploy.deploy(arguments)


def test_source_scan_runs_before_snapshot_and_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = tmp_path / "repo"
    repository.mkdir()
    state = tmp_path / "state"
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    signing = _signing_material(state)
    artifacts = staging_deploy.DeployHostArtifacts(
        archive=state / "bundle.tar.gz",
        checksum=state / "bundle.tar.gz.sha256",
        signature_bundle=state / "bundle.sigstore.json",
    )
    events: list[tuple[str, Path]] = []
    source = staging_deploy.SourceCheckout(
        repository_root=snapshot,
        git_commit="a" * 40,
        fingerprint="b" * 64,
        snapshot=True,
        isolated=True,
    )
    monkeypatch.setattr(
        staging_deploy,
        "validate_source_checkout",
        lambda root: events.append(("scan", root)),
    )
    monkeypatch.setattr(
        staging_deploy, "prepare_source_checkout", lambda *_args, **_kwargs: source
    )
    monkeypatch.setattr(
        staging_deploy, "ensure_signing_material", lambda *_args, **_kwargs: signing
    )
    monkeypatch.setattr(
        staging_deploy,
        "source_deploy_identity",
        lambda *_args, **_kwargs: _source_identities(),
    )
    monkeypatch.setattr(
        staging_deploy, "restore_trusted_ci_candidate", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        staging_deploy, "load_successful_source_deploy", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        staging_deploy,
        "record_successful_source_deploy",
        lambda *_args, **_kwargs: state / "source-deploy-success.json",
    )
    monkeypatch.setattr(
        staging_deploy, "deploy_host_artifacts", lambda *_args, **_kwargs: artifacts
    )
    monkeypatch.setattr(
        staging_deploy,
        "deploy_host_wheelhouse_cache",
        lambda *_args, **_kwargs: state / "wheelhouse",
    )

    def bundle(*_args, **_kwargs):
        events.append(("bundle", snapshot))
        return False

    monkeypatch.setattr(staging_deploy, "ensure_deploy_host_bundle", bundle)
    monkeypatch.setattr(
        staging_deploy,
        "ensure_deploy_host_venv",
        lambda *_args, **_kwargs: state / "venv",
    )
    monkeypatch.setattr(staging_deploy, "run_admin_deploy", lambda **_kwargs: None)
    monkeypatch.setattr(
        staging_deploy,
        "collect_live_deploy_evidence",
        lambda **_kwargs: _live_evidence(),
    )

    staging_deploy.deploy(
        argparse.Namespace(
            repo_root=repository,
            state_dir=state,
            cpu_cluster_arn="cpu",
            gpu_cluster_arn=["gpu"],
            admin_email="operations@example.com",
            base="origin/main",
        )
    )

    assert events == [("scan", repository), ("scan", snapshot), ("bundle", snapshot)]


def test_staging_snapshot_authorization_is_passed_to_admin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    commands: list[list[str]] = []
    venv = tmp_path / "venv"
    (venv / "bin").mkdir(parents=True)
    monkeypatch.setattr(
        staging_deploy,
        "_run",
        lambda arguments, **_kwargs: commands.append(list(arguments)) or "",
    )

    staging_deploy.run_admin_deploy(
        repository_root=tmp_path,
        state_dir=tmp_path / "state",
        venv=venv,
        cpu_cluster_arn="cpu",
        gpu_cluster_arns=("gpu",),
        admin_email="operations@example.com",
        staging_only_release=True,
        impact_base="origin/release",
        lock_fd=7,
    )

    assert "--profile-approval" not in commands[0]
    assert "--staging-only-release" in commands[0]
    assert "--prepared-source-release" in commands[0]
    assert commands[0][commands[0].index("--impact-base") + 1] == "origin/release"


def test_makefile_keeps_staging_release_build_internal() -> None:
    makefile = (staging_deploy.ROOT / "Makefile").read_text(encoding="utf-8")
    assert "\nstaging-deploy:" not in makefile
    staging_build = makefile.split("release-build-staging:\n", 1)[1].split(
        "\nrelease-deploy:", 1
    )[0]
    assert "--write-plan" in staging_build
    assert "--read-plan" in staging_build
    assert staging_build.count("scripts/select-affected-tests.py") == 2
    assert "--staging-only" in staging_build
    assert "build-release-attestation.py" in staging_build
    assert "--staging-only" in staging_build
    assert '--impact-base "$(BASE)"' in staging_build


def _git(repository: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments], cwd=repository, check=True, capture_output=True, text=True
    )
    return completed.stdout.strip()


def test_clean_source_is_isolated_without_staging_tier(tmp_path: Path) -> None:
    repository = tmp_path / "repo"
    repository.mkdir()
    _git(repository, "init")
    _git(repository, "config", "user.name", "Test")
    _git(repository, "config", "user.email", "test@example.com")
    (repository / ".gitignore").write_text("dist/\n", encoding="utf-8")
    tracked = repository / "tracked.txt"
    tracked.write_text("release\n", encoding="utf-8")
    tracked.chmod(0o664)
    _git(repository, "add", ".gitignore", "tracked.txt")
    _git(repository, "commit", "-m", "initial")
    state = tmp_path / "state"

    first = staging_deploy.prepare_source_checkout(repository, state_dir=state)
    first.repository_root.joinpath("dist").mkdir()
    first.repository_root.joinpath("dist/current-release.json").write_text(
        "signed-release\n", encoding="utf-8"
    )
    repository.joinpath("dist").mkdir()
    repository.joinpath("dist/current-release.json").write_text(
        "later-local-build\n", encoding="utf-8"
    )
    second = staging_deploy.prepare_source_checkout(repository, state_dir=state)

    assert first == second
    assert first.repository_root != repository
    assert first.snapshot is False
    assert first.isolated is True
    assert first.git_commit == _git(repository, "rev-parse", "HEAD")
    assert first.repository_root.joinpath("tracked.txt").stat().st_mode & 0o777 == 0o664
    assert (
        first.repository_root.joinpath("dist/current-release.json").read_text(
            encoding="utf-8"
        )
        == "signed-release\n"
    )


def test_dirty_source_is_snapshotted_without_changing_original(tmp_path: Path) -> None:
    repository = tmp_path / "repo"
    repository.mkdir()
    _git(repository, "init")
    _git(repository, "config", "user.name", "Test")
    _git(repository, "config", "user.email", "test@example.com")
    tracked = repository / "tracked.txt"
    tracked.write_text("before\n", encoding="utf-8")
    _git(repository, "add", "tracked.txt")
    _git(repository, "commit", "-m", "initial")
    tracked.write_text("after\n", encoding="utf-8")
    (repository / "new.txt").write_text("new\n", encoding="utf-8")

    first = staging_deploy.prepare_source_checkout(
        repository, state_dir=tmp_path / "state"
    )
    second = staging_deploy.prepare_source_checkout(
        repository, state_dir=tmp_path / "state"
    )

    assert first.snapshot is True
    assert first.isolated is True
    assert first == second
    assert (first.repository_root / "tracked.txt").read_text() == "after\n"
    assert (first.repository_root / "new.txt").read_text() == "new\n"
    assert _git(first.repository_root, "status", "--porcelain") == ""
    assert "tracked.txt" in _git(repository, "status", "--short")
    assert "new.txt" in _git(repository, "status", "--short")


def test_dirty_snapshot_preserves_tracked_modes_across_umasks(tmp_path: Path) -> None:
    repository = tmp_path / "repo"
    repository.mkdir()
    _git(repository, "init")
    _git(repository, "config", "user.name", "Test")
    _git(repository, "config", "user.email", "test@example.com")
    tracked = repository / "tracked.txt"
    tracked.write_text("before\n", encoding="utf-8")
    tracked.chmod(0o664)
    _git(repository, "add", "tracked.txt")
    _git(repository, "commit", "-m", "initial")
    tracked.write_text("after\n", encoding="utf-8")
    state = tmp_path / "state"

    previous_umask = os.umask(0o077)
    try:
        first = staging_deploy.prepare_source_checkout(repository, state_dir=state)
    finally:
        os.umask(previous_umask)

    assert first.repository_root.joinpath("tracked.txt").stat().st_mode & 0o777 == 0o664

    tracked.chmod(0o644)
    second = staging_deploy.prepare_source_checkout(repository, state_dir=state)

    assert second.fingerprint != first.fingerprint
    assert (
        second.repository_root.joinpath("tracked.txt").stat().st_mode & 0o777 == 0o644
    )


def test_snapshot_metadata_must_match_current_fingerprint(tmp_path: Path) -> None:
    repository = tmp_path / "repo"
    repository.mkdir()
    _git(repository, "init")
    _git(repository, "config", "user.name", "Test")
    _git(repository, "config", "user.email", "test@example.com")
    tracked = repository / "tracked.txt"
    tracked.write_text("before\n", encoding="utf-8")
    _git(repository, "add", "tracked.txt")
    _git(repository, "commit", "-m", "initial")
    tracked.write_text("after\n", encoding="utf-8")
    state = tmp_path / "state"
    checkout = staging_deploy.prepare_source_checkout(repository, state_dir=state)
    snapshot_dir = state / "source-snapshots" / checkout.fingerprint
    metadata = snapshot_dir / "snapshot.json"
    value = json.loads(metadata.read_text(encoding="utf-8"))
    value["fingerprint"] = "c" * 64
    metadata.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(
        staging_deploy.StagingDeployError, match="snapshot identity does not match"
    ):
        staging_deploy.prepare_source_checkout(repository, state_dir=state)


def test_snapshot_prepared_tree_must_remain_unchanged(tmp_path: Path) -> None:
    repository = tmp_path / "repo"
    repository.mkdir()
    _git(repository, "init")
    _git(repository, "config", "user.name", "Test")
    _git(repository, "config", "user.email", "test@example.com")
    tracked = repository / "tracked.txt"
    tracked.write_text("release\n", encoding="utf-8")
    tracked.chmod(0o644)
    _git(repository, "add", "tracked.txt")
    _git(repository, "commit", "-m", "initial")
    state = tmp_path / "state"
    checkout = staging_deploy.prepare_source_checkout(repository, state_dir=state)
    checkout.repository_root.joinpath("tracked.txt").chmod(0o600)

    with pytest.raises(
        staging_deploy.StagingDeployError, match="prepared tree does not match"
    ):
        staging_deploy.prepare_source_checkout(repository, state_dir=state)

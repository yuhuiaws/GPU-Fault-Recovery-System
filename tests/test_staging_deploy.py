from __future__ import annotations

import argparse
import inspect
import json
import os
from pathlib import Path

import pytest
import yaml

from gpu_fault.admin.notification_precheck import EmailConfirmation
from gpu_fault.admin.operation_lock import SITE_OPERATION_LOCK_FD_ENV
from scripts import (
    staging_deploy,
    staging_gate_caches,
    staging_live_evidence,
    staging_state_hygiene,
)


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
        "runtime_profile_sha256": "f" * 64,
        "runtime_profile_policy_digest": "c" * 64,
        "release_id": "release-a",
        "state_sha256": "e" * 64,
        "phase": "complete",
        "transaction_committed": True,
        "next_deploy_kind": "NOOP",
    }


def _status_report() -> dict[str, object]:
    """The quick ``status`` report the pre-deploy reading returns."""

    return {
        "mode": "status",
        "healthy": True,
        "health_scope": "quick",
        "live_release": {
            "release_id": "release-a",
            "phase": "complete",
            "transaction_committed": True,
            "state_sha256": "e" * 64,
        },
        "configured_release": {
            "release_id": "release-a",
            "database_schema_version": 12,
        },
        "next_deploy": {"kind": "NOOP", "changed": []},
    }


def _confirmed_email() -> EmailConfirmation:
    return EmailConfirmation(
        sender="operations@example.com",
        admin_email="operations@example.com",
        ses_verified=True,
        ses_identity_created=False,
        sns_topic_arn="arn:aws:sns:us-east-1:123456789012:gpu-fault-alerts",
        sns_status="CONFIRMED",
        sns_subscription_arn="arn:aws:sns:us-east-1:123456789012:gpu-fault-alerts:1",
    )


def test_pending_profile_change_runs_the_release_on_unchanged_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("GPU_FAULT_ADMIN_LOG", raising=False)
    repository, state = _managed_state(tmp_path)
    _install_admin_stub(state)
    signing = _signing_material(state)
    source = staging_deploy.SourceCheckout(
        repository_root=repository,
        git_commit="a" * 40,
        fingerprint="a" * 64,
        snapshot=False,
        isolated=True,
    )
    previous = {
        "schema_version": 1,
        "status": "PASSED",
        "identities": _source_identities(),
        "source": {"fingerprint": source.fingerprint, "git_commit": source.git_commit},
        "live": _live_evidence(),
    }
    events: list[str] = []
    _stub_deploy_orchestration(
        monkeypatch,
        state=state,
        source=source,
        signing=signing,
        previous=previous,
        events=events,
    )
    readings: list[str] = []

    def evidence(**_kwargs: object) -> dict[str, object]:
        readings.append("read")
        if len(readings) == 1:
            raise staging_live_evidence.RuntimeProfileChangePending("EXPANSIVE")
        events.append(EVIDENCE_EVENT)
        return _live_evidence()

    monkeypatch.setattr(staging_deploy, "collect_live_deploy_evidence", evidence)

    result = staging_deploy.deploy(_deploy_arguments(repository, state))

    assert result["deploy_mode"] == "APPLICATION_RELEASE"
    assert DEPLOY_EVENT in events, (
        "a pending Runtime Profile change on unchanged source deployed nothing"
    )
    # The success record comes from the release driver's own files, not from a
    # second reading of the site.
    assert readings == ["read"], "the pre-apply reading is the only status call"
    assert events.index(DEPLOY_EVENT) < events.index(RECORD_EVIDENCE_EVENT)
    assert events.index(RECORD_EVIDENCE_EVENT) < events.index("success")


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
    monkeypatch.setattr(
        staging_state_hygiene, "bundle_platform_id", lambda: "test-platform"
    )

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
    # A live transaction that is not committed (failed, rolled back, mid-flight)
    # needs the release engine whatever the source identities say (live
    # 2026-09-11: a foreign candidate FAILED at cpu-staged was classified
    # DEPLOY_HOST_ONLY and --supersede-failed-transaction never ran).
    assert (
        staging_deploy.classify_source_deploy(
            previous,
            current,
            source=source,
            site_exists=True,
            live_matches=True,
            live_release_pending=True,
        )
        == "APPLICATION_RELEASE"
    )


def test_live_release_transaction_pending_reads_the_status_report() -> None:
    committed = {"live_release": {"phase": "complete", "transaction_committed": True}}
    failed = {"live_release": {"phase": "failed", "transaction_committed": False}}
    rolled_back = {
        "live_release": {"phase": "rolled-back", "transaction_committed": False}
    }
    assert staging_deploy.live_release_transaction_pending(committed) is False
    assert staging_deploy.live_release_transaction_pending(failed) is True
    assert staging_deploy.live_release_transaction_pending(rolled_back) is True
    assert staging_deploy.live_release_transaction_pending(None) is False
    assert staging_deploy.live_release_transaction_pending({}) is False


def test_deploy_host_wheelhouse_cache_is_lock_scoped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = tmp_path / "repo"
    requirements = repository / "requirements"
    requirements.mkdir(parents=True)
    (requirements / "build.lock").write_text("build-a\n", encoding="utf-8")
    (requirements / "deploy-host.lock").write_text("host-a\n", encoding="utf-8")
    monkeypatch.setattr(
        staging_state_hygiene, "bundle_platform_id", lambda: "test-platform"
    )

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
    monkeypatch.setattr(staging_deploy, "prune_venv_versions", lambda *_a, **_k: ())

    result = staging_deploy.ensure_deploy_host_venv(
        state,
        repository_root=tmp_path,
        signing=_signing_material(state),
        artifacts=artifacts,
        lock_fd=STUB_LOCK_FD,
    )

    binding = json.loads(
        result.joinpath("gpu-fault-managed-state-dir.json").read_text(encoding="utf-8")
    )
    assert binding == {"schema_version": 1, "state_dir": str(state.resolve())}
    assert (
        result.joinpath("gpu-fault-managed-state-dir.json").stat().st_mode & 0o777
        == 0o600
    )


def test_deploy_host_venv_prunes_versions_under_the_held_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Old venv versions are deleted here, where the site lock is already held.

    The hand-run ``scripts/setup-deploy-host.sh`` has no lock to give, and a tree
    another install is still writing into cannot be told from one it abandoned,
    so the setup path prunes nothing: the deploy does, handing its descriptor
    down as the proof that it holds the lock.
    """

    monkeypatch.delenv("GPU_FAULT_ADMIN_LOG", raising=False)
    state = tmp_path / "state"
    venv = state / "deployer-venv"
    artifacts = staging_deploy.DeployHostArtifacts(
        archive=state / "bundle.tar.gz",
        checksum=state / "bundle.tar.gz.sha256",
        signature_bundle=state / "bundle.sigstore.json",
    )
    pruned: list[tuple[Path, int]] = []

    def run(_arguments, **_kwargs):
        venv.joinpath("bin").mkdir(parents=True)
        venv.joinpath("bin/gpu-fault-admin").write_text("", encoding="utf-8")
        return ""

    monkeypatch.setattr(staging_deploy, "_run", run)
    monkeypatch.setattr(
        staging_deploy,
        "prune_venv_versions",
        lambda target, *, lock_fd: pruned.append((target, lock_fd)) or (),
    )

    staging_deploy.ensure_deploy_host_venv(
        state,
        repository_root=tmp_path,
        signing=_signing_material(state),
        artifacts=artifacts,
        lock_fd=STUB_LOCK_FD,
    )

    assert pruned == [(venv, STUB_LOCK_FD)]


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
        lambda root, **_kwargs: calls.append(("scan", root)),
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
    monkeypatch.setattr(
        staging_deploy, "read_live_status", lambda **_k: _status_report()
    )
    monkeypatch.setattr(
        staging_deploy, "precheck_email_confirmations", lambda **_k: _confirmed_email()
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
        (state / staging_state_hygiene.SOURCE_DEPLOY_STATE).read_text(encoding="utf-8")
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
    monkeypatch.setattr(
        staging_deploy, "validate_source_checkout", lambda _root, **_kwargs: None
    )
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
    monkeypatch.setattr(
        staging_deploy, "read_live_status", lambda **_k: _status_report()
    )
    monkeypatch.setattr(
        staging_deploy, "precheck_email_confirmations", lambda **_k: _confirmed_email()
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
        lambda root, **_kwargs: events.append(("scan", root)),
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
    monkeypatch.setattr(
        staging_deploy, "read_live_status", lambda **_k: _status_report()
    )
    monkeypatch.setattr(
        staging_deploy, "precheck_email_confirmations", lambda **_k: _confirmed_email()
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


def _deploy_arguments(repository: Path, state: Path) -> argparse.Namespace:
    return argparse.Namespace(
        repo_root=repository,
        state_dir=state,
        cpu_cluster_arn="arn:aws:eks:us-east-1:123456789012:cluster/cpu",
        gpu_cluster_arn=["arn:aws:eks:us-east-1:123456789012:cluster/gpu"],
        admin_email="operations@example.com",
        base="origin/main",
    )


def _managed_state(tmp_path: Path) -> tuple[Path, Path]:
    repository = tmp_path / "repo"
    repository.mkdir()
    state = tmp_path / "state"
    state.mkdir()
    (state / "site.yaml").write_text("kind: RegionalSite\n", encoding="utf-8")
    return repository, state


def _install_admin_stub(state: Path) -> None:
    admin = state / "deployer-venv/bin"
    admin.mkdir(parents=True)
    (admin / "gpu-fault-admin").write_text("", encoding="utf-8")


# The descriptor the stubbed lock hands out. It is carried into the recorded
# event names because a child that collects live evidence without it opens the
# lock file a second time and blocks on the lock this process already holds.
STUB_LOCK_FD = 17
EVIDENCE_EVENT = f"evidence(lock_fd={STUB_LOCK_FD})"
DEPLOY_EVENT = f"deploy(lock_fd={STUB_LOCK_FD})"
RECORD_EVIDENCE_EVENT = "evidence(release-record)"


def _stub_deploy_orchestration(
    monkeypatch: pytest.MonkeyPatch,
    *,
    state: Path,
    source: staging_deploy.SourceCheckout,
    signing: staging_deploy.SigningMaterial,
    previous: dict[str, object] | None,
    events: list[str],
) -> None:
    """Everything ``deploy`` shells out to, recorded as an ordered event log.

    The log is what the ordering assertions read: the point of collecting live
    evidence once is *when* each call happens, not what it returns.
    """

    artifacts = staging_deploy.DeployHostArtifacts(
        archive=state / "bundle.tar.gz",
        checksum=state / "bundle.tar.gz.sha256",
        signature_bundle=state / "bundle.sigstore.json",
    )

    class Lock:
        def __enter__(self) -> int:
            events.append("lock-enter")
            return STUB_LOCK_FD

        def __exit__(self, *_arguments: object) -> None:
            events.append("lock-exit")

    def evidence(**kwargs: object) -> dict[str, object]:
        events.append(f"evidence(lock_fd={kwargs.get('lock_fd')})")
        return _live_evidence()

    def record_evidence(**_kwargs: object) -> dict[str, object]:
        events.append(RECORD_EVIDENCE_EVENT)
        return _live_evidence()

    def admin_deploy(**kwargs: object) -> None:
        events.append(f"deploy(lock_fd={kwargs.get('lock_fd')})")

    monkeypatch.setattr(
        staging_deploy, "prepare_source_checkout", lambda *_a, **_k: source
    )
    monkeypatch.setattr(
        staging_deploy, "validate_source_checkout", lambda _root, **_k: None
    )
    monkeypatch.setattr(
        staging_deploy, "ensure_signing_material", lambda *_a, **_k: signing
    )
    monkeypatch.setattr(
        staging_deploy, "source_deploy_identity", lambda *_a, **_k: _source_identities()
    )
    monkeypatch.setattr(
        staging_deploy, "load_successful_source_deploy", lambda *_a, **_k: previous
    )
    monkeypatch.setattr(
        staging_deploy, "restore_trusted_ci_candidate", lambda *_a, **_k: None
    )
    monkeypatch.setattr(staging_deploy, "site_operation_lock", lambda *_a, **_k: Lock())
    monkeypatch.setattr(
        staging_deploy, "deploy_host_artifacts", lambda *_a, **_k: artifacts
    )
    monkeypatch.setattr(
        staging_deploy,
        "deploy_host_wheelhouse_cache",
        lambda *_a, **_k: state / "wheelhouse",
    )
    monkeypatch.setattr(
        staging_deploy, "ensure_deploy_host_bundle", lambda *_a, **_k: True
    )
    monkeypatch.setattr(
        staging_deploy,
        "ensure_deploy_host_venv",
        lambda *_a, **_k: state / "deployer-venv",
    )
    monkeypatch.setattr(staging_deploy, "run_admin_deploy", admin_deploy)
    monkeypatch.setattr(staging_deploy, "collect_live_deploy_evidence", evidence)
    monkeypatch.setattr(
        staging_deploy, "read_live_status", lambda **_k: _status_report()
    )
    monkeypatch.setattr(
        staging_deploy, "precheck_email_confirmations", lambda **_k: _confirmed_email()
    )
    monkeypatch.setattr(
        staging_deploy, "live_evidence_from_release_record", record_evidence
    )
    monkeypatch.setattr(
        staging_deploy,
        "prune_source_snapshots",
        lambda *_a, **_k: events.append("prune") or (),
    )
    monkeypatch.setattr(
        staging_deploy,
        "record_successful_source_deploy",
        lambda *_a, **_k: (
            events.append("success") or state / "source-deploy-success.json"
        ),
    )


def test_apply_source_deploy_has_no_lockless_path() -> None:
    """One classification per deploy, and it is made under the lock.

    The apply used to re-read the live site and re-classify whenever it was
    called without a lock: a second 45-second ``status`` and a second, divergent
    decision path that no test covered. Requiring the descriptor is what keeps
    that path from growing back.
    """

    parameter = inspect.signature(staging_deploy.apply_source_deploy).parameters[
        "lock_fd"
    ]

    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert parameter.default is inspect.Parameter.empty, (
        "an apply without the caller's lock classifies the site a second time"
    )


def test_deploy_collects_live_evidence_once_under_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One pre-apply ``status`` under the lock; the success comes from the record.

    The classification the pre-apply reading feeds is only trustworthy while
    nothing else can change the site, so the lock comes first and the answer is
    reused instead of re-derived. After an application release the success
    record is built from the release driver's own ``state.json`` and
    ``release-summary.json`` -- not from a second ``status`` (about 45 seconds)
    that would read the same release back.
    """

    monkeypatch.delenv("GPU_FAULT_ADMIN_LOG", raising=False)
    repository, state = _managed_state(tmp_path)
    _install_admin_stub(state)
    signing = _signing_material(state)
    source = staging_deploy.SourceCheckout(
        repository_root=repository,
        git_commit="a" * 40,
        fingerprint="a" * 64,
        snapshot=False,
        isolated=True,
    )
    previous = {
        "schema_version": 1,
        "status": "PASSED",
        "identities": {
            "schema_version": 1,
            "sha256": "f" * 64,
            "application": {"sha256": "9" * 64},
            "deploy_host": {"sha256": "b" * 64, "bundle": {"sha256": "c" * 64}},
        },
        "source": {"fingerprint": "0" * 64, "git_commit": "b" * 40},
        "live": _live_evidence(),
    }
    events: list[str] = []
    _stub_deploy_orchestration(
        monkeypatch,
        state=state,
        source=source,
        signing=signing,
        previous=previous,
        events=events,
    )

    result = staging_deploy.deploy(_deploy_arguments(repository, state))

    assert result["deploy_mode"] == "APPLICATION_RELEASE"
    assert events == [
        "lock-enter",
        EVIDENCE_EVENT,
        DEPLOY_EVENT,
        RECORD_EVIDENCE_EVENT,
        "success",
        "prune",
        "lock-exit",
    ]


def test_release_record_that_is_not_evidence_falls_back_to_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A driver that completed without a usable summary still gets a success record."""

    def unusable(**_kwargs: object) -> dict[str, object]:
        raise staging_deploy.LiveEvidenceError("release summary unavailable")

    monkeypatch.setattr(staging_deploy, "live_evidence_from_release_record", unusable)

    assert staging_deploy.release_record_evidence(tmp_path) is None
    assert "release summary unavailable" in capsys.readouterr().err


def test_unchanged_deploy_reuses_its_single_live_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("GPU_FAULT_ADMIN_LOG", raising=False)
    repository, state = _managed_state(tmp_path)
    _install_admin_stub(state)
    signing = _signing_material(state)
    source = staging_deploy.SourceCheckout(
        repository_root=repository,
        git_commit="a" * 40,
        fingerprint="a" * 64,
        snapshot=False,
        isolated=True,
    )
    previous = {
        "schema_version": 1,
        "status": "PASSED",
        "identities": _source_identities(),
        "source": {"fingerprint": source.fingerprint, "git_commit": source.git_commit},
        "live": _live_evidence(),
    }
    events: list[str] = []
    _stub_deploy_orchestration(
        monkeypatch,
        state=state,
        source=source,
        signing=signing,
        previous=previous,
        events=events,
    )

    result = staging_deploy.deploy(_deploy_arguments(repository, state))

    assert result["deploy_mode"] == "UNCHANGED"
    assert events == ["lock-enter", EVIDENCE_EVENT, "success", "prune", "lock-exit"]


def test_run_admin_deploy_passes_tool_cache_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The real deploy path gets the same warm mypy/ruff caches the gate uses.

    The admin deploy re-runs the static gates inside a snapshot that cannot hold
    a cache, so without this the 17-second mypy analysis is paid from cold on
    every deploy while an identical answer sits in the state directory.
    """

    monkeypatch.delenv("GPU_FAULT_ADMIN_LOG", raising=False)
    # The deploy's own gate suite runs with these caches already exported (that
    # is the point of A2), and an ambient value wins over the state-dir path; the
    # assertion below is about the path the deploy derives, so start clean.
    for name, _subdirectory in staging_gate_caches.TOOL_CACHE_VARIABLES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("GPU_FAULT_TEST_AMBIENT", "inherited")
    state = tmp_path / "state"
    state.mkdir()
    captured: dict[str, object] = {}

    def run(
        arguments: object,
        *,
        cwd: Path,
        env: dict[str, str] | None = None,
        capture: bool = False,
        pass_fds: tuple[int, ...] = (),
    ) -> str:
        captured["env"] = dict(env or {})
        captured["pass_fds"] = pass_fds
        return ""

    monkeypatch.setattr(staging_deploy, "_run", run)
    lock_fd = os.open(os.devnull, os.O_RDONLY)
    try:
        staging_deploy.run_admin_deploy(
            repository_root=tmp_path / "repo",
            state_dir=state,
            venv=tmp_path / "venv",
            cpu_cluster_arn="cpu",
            gpu_cluster_arns=("gpu",),
            admin_email="operations@example.com",
            staging_only_release=False,
            impact_base="origin/main",
            lock_fd=lock_fd,
        )
    finally:
        os.close(lock_fd)

    environment = captured["env"]
    assert isinstance(environment, dict), "the admin deploy must receive an env dict"
    expected = staging_gate_caches.tool_cache_environment(state)
    for name, _subdirectory in staging_gate_caches.TOOL_CACHE_VARIABLES:
        assert environment[name] == expected[name], name
        assert Path(environment[name]).is_relative_to(state), name
    assert environment[SITE_OPERATION_LOCK_FD_ENV] == str(lock_fd), (
        "lock FD must be passed in environment"
    )
    assert captured["pass_fds"] == (lock_fd,), "lock FD must be passed to subprocess"
    assert environment["GPU_FAULT_TEST_AMBIENT"] == "inherited", (
        "ambient test variable must be inherited"
    )


def test_missing_venv_promotes_prepared_mode_and_deploys(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A rebuilt deploy host is an application release, not a silent no-op.

    Promoting only the local ``mode`` left ``prepared_mode`` at QUALITY_ONLY, so
    the impact gate was skipped as if a release were coming and the apply then
    ran no release at all -- and still recorded success.
    """

    monkeypatch.delenv("GPU_FAULT_ADMIN_LOG", raising=False)
    repository, state = _managed_state(tmp_path)
    signing = _signing_material(state)
    source = staging_deploy.SourceCheckout(
        repository_root=repository,
        git_commit="a" * 40,
        fingerprint="a" * 64,
        snapshot=False,
        isolated=True,
    )
    previous = {
        "schema_version": 1,
        "status": "PASSED",
        "identities": _source_identities(),
        "source": {"fingerprint": "0" * 64, "git_commit": "b" * 40},
        "live": _live_evidence(),
    }
    events: list[str] = []
    _stub_deploy_orchestration(
        monkeypatch,
        state=state,
        source=source,
        signing=signing,
        previous=previous,
        events=events,
    )
    monkeypatch.setattr(
        staging_deploy,
        "run_source_impact_gate",
        lambda **_kwargs: events.append("gate") or {"source": "impact"},
    )

    result = staging_deploy.deploy(_deploy_arguments(repository, state))

    assert result["deploy_mode"] == "APPLICATION_RELEASE"
    assert DEPLOY_EVENT in events, (
        "a missing deploy-host venv gated nothing and deployed nothing"
    )
    assert "gate" not in events
    assert events.index(DEPLOY_EVENT) < events.index("success")

"""Synthetic state and recorded I/O for source-deploy tests."""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from gpu_fault.admin.notification_precheck import EmailConfirmation
from scripts import staging_deploy


def signing_material(root: Path) -> staging_deploy.SigningMaterial:
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


def source_identities() -> dict[str, object]:
    return {
        "schema_version": 1,
        "sha256": "f" * 64,
        "application": {"sha256": "a" * 64},
        "deploy_host": {"sha256": "b" * 64, "bundle": {"sha256": "c" * 64}},
    }


def live_evidence() -> dict[str, object]:
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


def status_report() -> dict[str, object]:
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


def confirmed_email() -> EmailConfirmation:
    return EmailConfirmation(
        sender="operations@example.com",
        admin_email="operations@example.com",
        ses_verified=True,
        ses_identity_created=False,
        sns_topic_arn="arn:aws:sns:us-east-1:123456789012:gpu-fault-alerts",
        sns_status="CONFIRMED",
        sns_subscription_arn="arn:aws:sns:us-east-1:123456789012:gpu-fault-alerts:1",
    )


def deploy_arguments(repository: Path, state: Path) -> argparse.Namespace:
    return argparse.Namespace(
        repo_root=repository,
        state_dir=state,
        cpu_cluster_arn="arn:aws:eks:us-east-1:123456789012:cluster/cpu",
        gpu_cluster_arn=["arn:aws:eks:us-east-1:123456789012:cluster/gpu"],
        admin_email="operations@example.com",
        base="origin/main",
    )


def managed_state(tmp_path: Path) -> tuple[Path, Path]:
    repository = tmp_path / "repo"
    repository.mkdir()
    state = tmp_path / "state"
    state.mkdir()
    (state / "site.yaml").write_text("kind: RegionalSite\n", encoding="utf-8")
    return repository, state


def install_admin_stub(state: Path) -> None:
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


def stub_deploy_orchestration(
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
        return live_evidence()

    def record_evidence(**_kwargs: object) -> dict[str, object]:
        events.append(RECORD_EVIDENCE_EVENT)
        return live_evidence()

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
        staging_deploy, "source_deploy_identity", lambda *_a, **_k: source_identities()
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
        staging_deploy, "read_live_status", lambda **_k: status_report()
    )
    monkeypatch.setattr(
        staging_deploy, "precheck_email_confirmations", lambda **_k: confirmed_email()
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

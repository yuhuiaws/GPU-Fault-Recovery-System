"""``--allow-unlogged-signature``: the signers and verifiers that must still refuse.

Live 2026-11-07 the public Sigstore transparency log answered ``status 500``
and the R24 deploy stopped at the release attestation's ``cosign sign-blob``.
The switch signs with the site key only. These tests pin, for the release
attestation, the deploy-host archive, the CI candidate receipt and the
successful source deploy record, that a verifier takes a bundle without a log
entry only when the signed payload declares it *and* the running command
carries the operator's switch (or, for the release the site already runs, the
cluster's release record names it); that a bundle without the declaration, or
with a tampered one, is refused; that the default path keeps today's commands
byte for byte; and that the source preparer hands the consent on as explicit
arguments, never as an exported variable. Fake ``cosign`` calls stand in for
Sigstore. The admin CLI side is ``tests/test_unlogged_release_admin.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from gpu_fault_release import regional_release_signing as admin_signing
from scripts import (
    ci_candidate_receipt,
    release_attestation,
    setup_deploy_host,
    staging_deploy,
    staging_live_evidence,
)
from scripts import signing_transparency as transparency
from tests._unlogged_signing_support import (
    ENV,
    LOGGED_BUNDLE,
    REASON,
    ROOT,
    UNLOGGED_BUNDLE,
)
from tests._unlogged_signing_support import write_json as _write

# -- the release attestation ---------------------------------------------------


def _release(root: Path, *, declaration: dict[str, str] | None = None) -> Path:
    manifest = _write(
        root / "dist/current-release.json",
        {
            "schema_version": 3,
            "deployable": True,
            "release_id": "release-a",
            "delivery": {"sha256": "d" * 64},
        },
    )
    value: dict[str, Any] = {
        "schema_version": 1,
        "subject": {
            "release_id": "release-a",
            "manifest": "dist/current-release.json",
            "manifest_sha256": release_attestation.sha256(manifest),
            "delivery_sha256": "d" * 64,
        },
        "source": {"dirty": False},
        "quality_gates": [
            {"command": "make check", "status": "PASSED"},
            {"command": "make test-postgres-stress", "status": "PASSED"},
        ],
    }
    if declaration is not None:
        value["signing"] = declaration
    return _write(root / "dist/current-attestation.json", value)


class SigningCosign:
    """``subprocess.run`` for cosign: verifies against the bytes it "signed"."""

    def __init__(self) -> None:
        self.signed: dict[str, str] = {}
        self.commands: list[list[str]] = []

    def sign(self, path: Path) -> None:
        self.signed[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()

    def __call__(self, command: list[str], **_kwargs: Any) -> Any:
        self.commands.append(list(command))
        artifact = command[-1]
        if (
            artifact in self.signed
            and hashlib.sha256(Path(artifact).read_bytes()).hexdigest()
            != self.signed[artifact]
        ):
            return subprocess.CompletedProcess(
                command, 1, "", "Error: failed to verify signature: invalid signature"
            )
        return subprocess.CompletedProcess(command, 0, "", "Verified OK")


@pytest.fixture
def signer(monkeypatch: pytest.MonkeyPatch) -> SigningCosign:
    fake = SigningCosign()
    monkeypatch.setattr(release_attestation.subprocess, "run", fake)
    return fake


def _verify(root: Path, bundle: dict[str, Any] | None, **options: Any) -> Any:
    attestation = root / "dist/current-attestation.json"
    path = (
        _write(root / "dist/current-attestation.bundle.json", bundle)
        if bundle is not None
        else None
    )
    return release_attestation.verify_attestation_signature(
        root,
        attestation,
        signature=None,
        bundle=path,
        cosign_key=options.pop("cosign_key", "cosign.pub"),
        certificate=None,
        certificate_identity=options.pop("identity", None),
        certificate_oidc_issuer=options.pop("issuer", None),
        **options,
    )


def test_a_logged_bundle_keeps_todays_command(tmp_path: Path, signer) -> None:
    attestation = _release(tmp_path)

    _value, verification = _verify(tmp_path, LOGGED_BUNDLE)

    assert signer.commands == [
        [
            "cosign",
            "verify-blob",
            "--bundle",
            str(tmp_path / "dist/current-attestation.bundle.json"),
            "--key",
            "cosign.pub",
            str(attestation),
        ]
    ]
    assert verification.transparency_log == "logged"
    assert verification.record(attestation)["trusted_root"] == "tuf"


def test_a_bundle_without_entry_and_declaration_is_refused_even_with_consent(
    tmp_path: Path, signer
) -> None:
    _release(tmp_path)
    with pytest.raises(
        release_attestation.ReleaseAttestationError,
        match="declares no unlogged signing",
    ):
        _verify(tmp_path, UNLOGGED_BUNDLE, allow_unlogged=True)
    assert signer.commands == []


def test_a_declared_unlogged_release_still_needs_an_approval(
    tmp_path: Path, signer
) -> None:
    _release(tmp_path, declaration=transparency.unlogged_declaration(REASON))
    with pytest.raises(release_attestation.ReleaseAttestationError) as refused:
        _verify(tmp_path, UNLOGGED_BUNDLE)
    assert "backfill-transparency-log" in str(refused.value)
    assert "--allow-unlogged-signature" in str(refused.value)
    assert signer.commands == []


def test_the_switch_accepts_a_declared_unlogged_release_offline(
    tmp_path: Path, signer
) -> None:
    attestation = _release(
        tmp_path, declaration=transparency.unlogged_declaration(REASON)
    )

    _value, verification = _verify(tmp_path, UNLOGGED_BUNDLE, allow_unlogged=True)

    (command,) = signer.commands
    assert command[-2:] == [transparency.IGNORE_TLOG_FLAG, str(attestation)]
    assert "--key" in command
    record = verification.record(attestation)
    assert record["transparency_log"] == "unlogged"
    assert record["reason"] == REASON and record["approval"] == "consent"
    assert record["attestation_sha256"] == release_attestation.sha256(attestation)


@pytest.mark.parametrize(
    ("state", "approved"),
    [
        (
            {
                "release_id": "release-a",
                "release_delivery_sha256": "d" * 64,
                "transaction_committed": True,
                "phase": "complete",
            },
            True,
        ),
        (
            {
                "release_id": "release-a",
                "release_delivery_sha256": "e" * 64,
                "transaction_committed": True,
                "phase": "complete",
            },
            False,
        ),
        (
            {
                "release_id": "release-a",
                "release_delivery_sha256": "d" * 64,
                "transaction_committed": False,
                "phase": "cpu-staged",
            },
            False,
        ),
        ({"release_id": "release-b", "transaction_committed": True}, False),
    ],
    ids=["live-release", "other-delivery", "uncommitted", "other-release"],
)
def test_the_clusters_record_of_the_live_release_approves_only_itself(
    tmp_path: Path, signer, state: dict[str, Any], approved: bool
) -> None:
    _release(tmp_path, declaration=transparency.unlogged_declaration(REASON))
    if approved:
        _value, verification = _verify(
            tmp_path, UNLOGGED_BUNDLE, live_release=lambda: state
        )
        assert verification.approval == "live_release"
    else:
        with pytest.raises(release_attestation.ReleaseAttestationError):
            _verify(tmp_path, UNLOGGED_BUNDLE, live_release=lambda: state)


def test_an_unreadable_live_record_approves_nothing(tmp_path: Path, signer) -> None:
    _release(tmp_path, declaration=transparency.unlogged_declaration(REASON))

    def unreadable() -> dict[str, Any]:
        raise RuntimeError("kubectl: connection refused")

    with pytest.raises(
        release_attestation.ReleaseAttestationError, match="cannot be read"
    ):
        _verify(tmp_path, UNLOGGED_BUNDLE, live_release=unreadable)


def test_a_keyless_signature_without_entry_is_never_accepted(
    tmp_path: Path, signer
) -> None:
    _release(tmp_path, declaration=transparency.unlogged_declaration(REASON))
    with pytest.raises(release_attestation.ReleaseAttestationError, match="keyless"):
        _verify(
            tmp_path,
            UNLOGGED_BUNDLE,
            cosign_key=None,
            identity="https://github.com/o/r/.github/workflows/ci.yml@refs/heads/main",
            issuer="https://token.actions.githubusercontent.com",
            allow_unlogged=True,
        )


def test_a_tampered_declaration_breaks_the_signature(tmp_path: Path, signer) -> None:
    attestation = _release(
        tmp_path, declaration=transparency.unlogged_declaration(REASON)
    )
    signer.sign(attestation)
    value = json.loads(attestation.read_text(encoding="utf-8"))
    value["signing"]["reason"] = "nothing to see"
    attestation.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(
        release_attestation.ReleaseAttestationError, match="invalid signature"
    ):
        _verify(tmp_path, UNLOGGED_BUNDLE, allow_unlogged=True)


def test_a_malformed_declaration_is_refused_before_cosign(
    tmp_path: Path, signer
) -> None:
    _release(
        tmp_path,
        declaration={**transparency.unlogged_declaration(REASON), "scope": "all"},
    )
    with pytest.raises(release_attestation.ReleaseAttestationError, match="exactly"):
        _verify(tmp_path, UNLOGGED_BUNDLE, allow_unlogged=True)
    assert signer.commands == []


def test_the_attestation_declares_only_when_the_deploy_asks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _release(tmp_path).with_name("current-release.json")
    monkeypatch.setattr(
        release_attestation,
        "source_state",
        lambda _root: {"git_commit": "a" * 40, "dirty": False, "diff_sha256": "0"},
    )
    default = release_attestation.build_attestation(tmp_path, manifest)
    unlogged = release_attestation.build_attestation(
        tmp_path, manifest, unlogged_signature_reason=REASON
    )
    assert "signing" not in default, "the default attestation changed"
    assert {
        key: value for key, value in unlogged.items() if key != "signing"
    } == default
    assert transparency.parse_declaration(unlogged["signing"])["reason"] == REASON


def test_the_attestation_builder_requires_the_handed_down_reason() -> None:
    script = ROOT / "scripts/build-release-attestation.py"
    environment = {key: value for key, value in os.environ.items() if key != ENV}
    completed = subprocess.run(
        [sys.executable, str(script), "--unlogged-signature"],
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 2
    assert "--reason" in completed.stderr


# -- the deploy-host archive -----------------------------------------------------


def _archive(path: Path, manifest: dict[str, Any]) -> Path:
    import io
    import tarfile

    data = json.dumps(manifest).encode()
    with tarfile.open(path, "w:gz") as bundle:
        member = tarfile.TarInfo("gpu-fault-deploy-host/manifest.json")
        member.size = len(data)
        bundle.addfile(member, io.BytesIO(data))
    return path


@pytest.mark.parametrize("declared", [True, False])
@pytest.mark.parametrize("allowed", [True, False])
def test_deploy_host_setup_takes_an_unlogged_archive_only_declared_and_allowed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, declared: bool, allowed: bool
) -> None:
    archive = _archive(
        tmp_path / "bundle.tar.gz",
        {"signing": transparency.unlogged_declaration(REASON)} if declared else {},
    )
    signature = _write(tmp_path / "bundle.sigstore.json", UNLOGGED_BUNDLE)
    commands: list[list[str]] = []
    monkeypatch.setattr(
        setup_deploy_host,
        "_run",
        lambda arguments, **_kwargs: commands.append(list(arguments)) or "",
    )

    def verify() -> None:
        setup_deploy_host.verify_bundle_signature(
            archive,
            signature_bundle=signature,
            cosign_key=str(tmp_path / "cosign.pub"),
            certificate_identity=None,
            certificate_oidc_issuer=None,
            allow_unsigned=False,
            allow_unlogged=allowed,
        )

    if declared and allowed:
        verify()
        assert commands[0][-2:] == [transparency.IGNORE_TLOG_FLAG, str(archive)]
    else:
        with pytest.raises(setup_deploy_host.DeployHostSetupError):
            verify()
        assert commands == []


def test_deploy_host_setup_falls_back_to_the_pin_only_on_tuf(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = tmp_path / "state"
    directory = state / "release-signing"
    directory.mkdir(parents=True, mode=0o700)
    (directory / "cosign.key").write_text("key", encoding="utf-8")
    (directory / "cosign.pub").write_text("pub", encoding="utf-8")
    pin_files = {
        transparency.TRUSTED_ROOT_PIN: b"root",
        transparency.SIGNING_CONFIG_PIN: b"config",
    }
    for name, data in pin_files.items():
        (directory / name).write_bytes(data)
    _write(
        directory / transparency.PIN_RECORD,
        {
            "pinned_at": transparency.utc_timestamp(),
            "files": {
                name: {"sha256": hashlib.sha256(data).hexdigest()}
                for name, data in pin_files.items()
            },
        },
    )
    archive = tmp_path / "bundle.tar.gz"
    archive.write_bytes(b"archive")
    signature = _write(tmp_path / "bundle.sigstore.json", LOGGED_BUNDLE)
    commands: list[list[str]] = []

    def run(arguments: list[str], **_kwargs: Any) -> str:
        commands.append(list(arguments))
        if len(commands) == 1:
            raise setup_deploy_host.DeployHostSetupError(
                "command failed (1): cosign: Error: getting trusted root from TUF"
            )
        return ""

    monkeypatch.setattr(setup_deploy_host, "_run", run)
    setup_deploy_host.verify_bundle_signature(
        archive,
        signature_bundle=signature,
        cosign_key=str(directory / "cosign.pub"),
        certificate_identity=None,
        certificate_oidc_issuer=None,
        allow_unsigned=False,
    )
    assert "--trusted-root" not in commands[0]
    assert commands[1][commands[1].index("--trusted-root") + 1] == str(
        directory / transparency.TRUSTED_ROOT_PIN
    )


# -- the CI candidate receipt ------------------------------------------------------


@pytest.fixture
def receipt_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "repo"
    gate = _write(
        root / "dist/ci-gate.json",
        {
            "repository": "o/r",
            "run_id": 7,
            "source": {"git_commit": "c" * 40, "git_tree": "t" * 40},
        },
    )
    assert gate.is_file(), "the CI gate fixture was not written"
    monkeypatch.setattr(
        ci_candidate_receipt,
        "_git",
        lambda _root, *arguments: "t" * 40
        if arguments[-1].endswith("{tree}")
        else "c" * 40,
    )
    return root


def test_an_unlogged_receipt_is_declared_and_signed_with_the_key_only(
    tmp_path: Path, receipt_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[list[str]] = []

    def run(command: list[str], **_kwargs: Any) -> Any:
        calls.append(list(command))
        Path(command[command.index("--bundle") + 1]).write_text("{}", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(ci_candidate_receipt.subprocess, "run", run)
    value = ci_candidate_receipt.write_receipt(
        receipt_root,
        tmp_path / "state",
        gate_path=receipt_root / "dist/ci-gate.json",
        repository="o/r",
        run_id=7,
        signing_key=tmp_path / "cosign.key",
        unlogged_signature_reason=REASON,
    )
    assert transparency.parse_declaration(value["signing"])["reason"] == REASON
    assert "--unlogged" in calls[0]


@pytest.mark.parametrize("allowed", [True, False])
def test_an_unlogged_receipt_counts_only_for_the_deploy_that_allowed_it(
    tmp_path: Path, receipt_root: Path, monkeypatch: pytest.MonkeyPatch, allowed: bool
) -> None:
    state = tmp_path / "state"
    directory = state / "ci-candidates" / ("c" * 40)
    gate = receipt_root / "dist/ci-gate.json"
    _write(
        directory / "verification.json",
        {
            "schema_version": 1,
            "repository": "o/r",
            "run_id": 7,
            "source": {"git_commit": "c" * 40, "git_tree": "t" * 40},
            "ci_gate": {
                "path": str(gate.resolve()),
                "sha256": hashlib.sha256(gate.read_bytes()).hexdigest(),
            },
            "signing": transparency.unlogged_declaration(REASON),
        },
    )
    _write(directory / "verification.sigstore.json", UNLOGGED_BUNDLE)
    commands: list[list[str]] = []
    monkeypatch.setattr(
        ci_candidate_receipt.subprocess,
        "run",
        lambda command, **_kwargs: commands.append(list(command))
        or subprocess.CompletedProcess(command, 0, "", ""),
    )
    value = ci_candidate_receipt.verify_receipt(
        receipt_root, state, public_key=tmp_path / "cosign.pub", allow_unlogged=allowed
    )
    if allowed:
        assert value is not None
        assert transparency.IGNORE_TLOG_FLAG in commands[0]
    else:
        assert value is None and commands == []


# -- the source preparer (scripts/staging_deploy.py) -------------------------------


def _staging_arguments(**values: Any) -> argparse.Namespace:
    return argparse.Namespace(**values)


def test_the_preparer_takes_both_flags_or_neither() -> None:
    resolve = staging_deploy.resolve_unlogged_consent
    assert resolve(_staging_arguments()) is None
    assert (
        resolve(_staging_arguments(allow_unlogged_signature=True, reason=f" {REASON} "))
        == REASON
    )
    for arguments in (
        _staging_arguments(reason=REASON),
        _staging_arguments(allow_unlogged_signature=True),
        _staging_arguments(allow_unlogged_signature=True, reason="two\nlines"),
    ):
        with pytest.raises(staging_deploy.StagingDeployError):
            resolve(arguments)
    parsed = staging_deploy.parser().parse_args(
        ["--state-dir", "/state", "--allow-unlogged-signature", "--reason", REASON]
    )
    assert resolve(parsed) == REASON


def test_an_exported_reason_never_reaches_a_child_by_itself() -> None:
    inherited = {ENV: "stale export", "KEEP": "1"}
    assert staging_deploy.consent_environment(inherited, None) == {"KEEP": "1"}
    assert staging_deploy.consent_environment(inherited, REASON) == {
        "KEEP": "1",
        ENV: REASON,
    }


def _signing(root: Path) -> staging_deploy.SigningMaterial:
    from tests._staging_deploy_support import signing_material

    return signing_material(root)


def _success_record(state: Path, *, declared: bool) -> None:
    _write(
        state / "source-deploy-success.json",
        {
            "schema_version": 1,
            "status": "PASSED",
            "identities": {
                "application": {"sha256": "a" * 64},
                "deploy_host": {"sha256": "b" * 64},
            },
            "source": {},
            **(
                {"signing": transparency.unlogged_declaration(REASON)}
                if declared
                else {}
            ),
        },
    )
    _write(state / "source-deploy-success.sigstore.json", UNLOGGED_BUNDLE)


@pytest.mark.parametrize("allowed", [True, False])
def test_the_previous_unlogged_success_record_needs_the_switch_or_a_backfill(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, allowed: bool
) -> None:
    state = tmp_path / "state"
    signing = _signing(state)
    _success_record(state, declared=True)
    commands: list[list[str]] = []
    monkeypatch.setattr(
        staging_deploy,
        "_run",
        lambda arguments, **_kwargs: commands.append(list(arguments)) or "",
    )
    if allowed:
        value = staging_deploy.load_successful_source_deploy(
            state, signing=signing, allow_unlogged=True
        )
        assert value is not None and commands[0][-2] == transparency.IGNORE_TLOG_FLAG
    else:
        with pytest.raises(
            staging_deploy.StagingDeployError, match="backfill-transparency-log"
        ):
            staging_deploy.load_successful_source_deploy(state, signing=signing)
        assert commands == []


def test_an_undeclared_unlogged_success_record_is_refused_with_the_switch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = tmp_path / "state"
    signing = _signing(state)
    _success_record(state, declared=False)
    monkeypatch.setattr(staging_deploy, "_run", lambda *_a, **_k: "")
    with pytest.raises(staging_deploy.StagingDeployError, match="declares no"):
        staging_deploy.load_successful_source_deploy(
            state, signing=signing, allow_unlogged=True
        )


@pytest.mark.parametrize("unlogged", [True, False])
def test_the_success_record_names_the_deploys_signing_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, unlogged: bool
) -> None:
    calls: list[tuple[list[str], dict[str, str]]] = []

    def run(arguments: list[str], **options: Any) -> str:
        calls.append((list(arguments), dict(options["env"])))
        Path(arguments[arguments.index("--bundle") + 1]).write_text("{}", "utf-8")
        return ""

    monkeypatch.setattr(staging_deploy, "_run", run)
    monkeypatch.setenv(ENV, "stale export")
    state = tmp_path / "state"
    state.mkdir(mode=0o700)
    path = staging_deploy.record_successful_source_deploy(
        state,
        source_repository_root=tmp_path,
        source=staging_deploy.SourceCheckout(
            repository_root=tmp_path,
            git_commit="a" * 40,
            fingerprint="f" * 64,
            snapshot=False,
        ),
        identities={
            "application": {"sha256": "1" * 64},
            "deploy_host": {"sha256": "2" * 64, "bundle": {"sha256": "3" * 64}},
        },
        signing=_signing(tmp_path / "keys"),
        mode="APPLICATION_RELEASE",
        live_evidence={},
        unlogged_reason=REASON if unlogged else None,
        release_signing={"transparency_log": "unlogged" if unlogged else "logged"},
    )
    record = json.loads(path.read_text(encoding="utf-8"))
    ((arguments, environment),) = calls
    if unlogged:
        assert transparency.parse_declaration(record["signing"])["reason"] == REASON
        assert "--unlogged" in arguments and environment[ENV] == REASON
    else:
        assert record["signing"] == {"transparency_log": "logged"}
        assert "--unlogged" not in arguments and ENV not in environment
    assert record["release_signing"]["transparency_log"] == (
        "unlogged" if unlogged else "logged"
    )


def _artifacts(root: Path) -> staging_deploy.DeployHostArtifacts:
    return staging_deploy.DeployHostArtifacts(
        archive=root / "bundle.tar.gz",
        checksum=root / "bundle.tar.gz.sha256",
        signature_bundle=root / "bundle.sigstore.json",
    )


@pytest.mark.parametrize("consent", [True, False])
def test_a_cached_unlogged_deploy_host_bundle_is_reused_only_with_consent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, consent: bool
) -> None:
    artifacts = _artifacts(tmp_path)
    artifacts.archive.write_text("archive", encoding="utf-8")
    artifacts.checksum.write_text("sum", encoding="utf-8")
    _write(artifacts.signature_bundle, UNLOGGED_BUNDLE)
    commands: list[tuple[list[str], dict[str, str]]] = []

    def run(arguments: list[str], **options: Any) -> str:
        commands.append((list(arguments), dict(options.get("env") or {})))
        for path in (artifacts.archive, artifacts.checksum, artifacts.signature_bundle):
            path.write_text("rebuilt", encoding="utf-8")
        return ""

    monkeypatch.setattr(staging_deploy, "_run", run)
    reused = staging_deploy.ensure_deploy_host_bundle(
        artifacts,
        repository_root=tmp_path,
        signing=_signing(tmp_path / "state"),
        unlogged_reason=REASON if consent else None,
    )
    if consent:
        assert reused is True and commands == []
    else:
        assert reused is False
        ((arguments, environment),) = commands
        assert "UNLOGGED_SIGNATURE=" in arguments and ENV not in environment


def test_a_new_deploy_host_bundle_is_signed_unlogged_only_with_consent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifacts = _artifacts(tmp_path)
    commands: list[tuple[list[str], dict[str, str]]] = []

    def run(arguments: list[str], **options: Any) -> str:
        commands.append((list(arguments), dict(options.get("env") or {})))
        for path in (artifacts.archive, artifacts.checksum, artifacts.signature_bundle):
            path.write_text("built", encoding="utf-8")
        return ""

    monkeypatch.setattr(staging_deploy, "_run", run)
    staging_deploy.ensure_deploy_host_bundle(
        artifacts,
        repository_root=tmp_path,
        signing=_signing(tmp_path / "state"),
        unlogged_reason=REASON,
    )
    ((arguments, environment),) = commands
    assert "UNLOGGED_SIGNATURE=1" in arguments and environment[ENV] == REASON


def test_a_refused_deploy_host_signing_names_the_switch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifacts = _artifacts(tmp_path)

    def run(_arguments: list[str], **_options: Any) -> str:
        transparency.write_refusal_marker(
            artifacts.signature_bundle,
            artifact=artifacts.archive,
            attempts=3,
            cause="status 500",
        )
        raise staging_deploy.StagingDeployError("command failed (2): make")

    monkeypatch.setattr(staging_deploy, "_run", run)
    with pytest.raises(staging_deploy.StagingDeployError) as refused:
        staging_deploy.ensure_deploy_host_bundle(
            artifacts, repository_root=tmp_path, signing=_signing(tmp_path / "state")
        )
    assert "--allow-unlogged-signature" in str(refused.value)
    assert "nothing on the site was changed" in str(refused.value)


def test_the_inner_deploy_gets_the_consent_as_arguments_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[list[str], dict[str, str]]] = []
    monkeypatch.setattr(
        staging_deploy,
        "_run",
        lambda arguments, **options: calls.append(
            (list(arguments), dict(options["env"]))
        )
        or "",
    )
    monkeypatch.setenv(ENV, "stale export")
    venv = tmp_path / "venv"
    (venv / "bin").mkdir(parents=True)
    staging_deploy.run_admin_deploy(
        repository_root=tmp_path,
        state_dir=tmp_path / "state",
        venv=venv,
        cpu_cluster_arn="cpu",
        gpu_cluster_arns=("gpu",),
        admin_email="operations@example.com",
        staging_only_release=False,
        impact_base="origin/main",
        lock_fd=7,
        unlogged_reason=REASON,
    )
    ((arguments, environment),) = calls
    index = arguments.index("--allow-unlogged-signature")
    assert arguments[index + 1 : index + 3] == ["--reason", REASON]
    assert ENV not in environment


def test_setup_and_deploy_host_only_verification_carry_the_consent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import yaml  # type: ignore[import-untyped,unused-ignore]

    commands: list[list[str]] = []
    monkeypatch.setattr(
        staging_deploy,
        "_run",
        lambda arguments, **_kwargs: commands.append(list(arguments)) or "",
    )
    state = tmp_path / "state"
    signing = _signing(state)
    venv = state / "deployer-venv"
    (venv / "bin").mkdir(parents=True)
    (venv / "bin/gpu-fault-admin").write_text("", encoding="utf-8")
    monkeypatch.setattr(staging_deploy, "prune_venv_versions", lambda *_a, **_k: None)
    staging_deploy.ensure_deploy_host_venv(
        state,
        repository_root=tmp_path,
        signing=signing,
        artifacts=_artifacts(tmp_path),
        lock_fd=7,
        allow_unlogged=True,
    )
    assert "--allow-unlogged-signature" in commands[-1]

    previous, current = tmp_path / "previous", tmp_path / "current"
    for name in (
        "current-release.json",
        "current-attestation.json",
        "current-attestation.bundle.json",
    ):
        _write(previous / "dist" / name, {})
    current.mkdir()
    site = state / "site.yaml"
    site.write_text(
        yaml.safe_dump(
            {"kind": "RegionalSite", "spec": {"repositoryRoot": str(previous)}}
        ),
        encoding="utf-8",
    )
    for allowed, expected in (
        (True, ["--allow-unlogged-signature"]),
        (False, ["--live-release-site", str(site)]),
    ):
        commands.clear()
        original = staging_deploy.prepare_deploy_host_only_checkout(
            repository_root=current,
            state_dir=state,
            signing=signing,
            allow_unlogged=allowed,
        )
        assert commands[0][-len(expected) :] == expected
        staging_deploy.restore_site_file(state, original)


def test_only_the_overdue_signing_check_reads_as_healthy_to_a_deploy() -> None:
    def report(*failing: str, healthy: bool = False) -> dict[str, Any]:
        return {
            "mode": "status",
            "healthy": healthy,
            "health": {
                "checks": [
                    {"name": "control_api", "status": "PASS"},
                    *({"name": name, "status": "FAIL"} for name in failing),
                ]
            },
        }

    assert staging_live_evidence.deploy_healthy(report(healthy=True)), "healthy"
    assert staging_live_evidence.deploy_healthy(report(admin_signing.SIGNING_CHECK)), (
        "the overdue signing check gated a deploy"
    )
    assert not staging_live_evidence.deploy_healthy(report("control_api_ready")), (
        "an engine failure read as healthy"
    )
    assert not staging_live_evidence.deploy_healthy(
        report(admin_signing.SIGNING_CHECK, "aurora")
    ), "another failing check hid behind the signing check"
    assert not staging_live_evidence.deploy_healthy(None), "no report is not healthy"

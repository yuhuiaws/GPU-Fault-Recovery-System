from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from scripts import release_attestation


def manifest(
    root: Path, *, deployable: bool = True, staging_only: bool = False
) -> Path:
    path = root / "dist/current-release.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": 3,
                "deployable": deployable,
                "staging_only": staging_only,
                "release_id": "release-a",
                "delivery": {"sha256": "d" * 64},
            }
        ),
        encoding="utf-8",
    )
    return path


def test_attestation_rejects_non_deployable_manifest(tmp_path: Path) -> None:
    path = manifest(tmp_path, deployable=False)

    with pytest.raises(release_attestation.ReleaseAttestationError, match="deployable"):
        release_attestation.load_manifest(path)


def test_staging_manifest_requires_explicit_authorization(tmp_path: Path) -> None:
    path = manifest(tmp_path, staging_only=True)

    with pytest.raises(
        release_attestation.ReleaseAttestationError, match="staging-only"
    ):
        release_attestation.load_manifest(path)

    assert (
        release_attestation.load_manifest(path, allow_staging=True)["staging_only"]
        is True
    )


def test_manifest_rejects_non_boolean_staging_tier(tmp_path: Path) -> None:
    path = manifest(tmp_path)
    value = json.loads(path.read_text(encoding="utf-8"))
    value["staging_only"] = "false"
    path.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(
        release_attestation.ReleaseAttestationError,
        match="staging_only must be a boolean",
    ):
        release_attestation.load_manifest(path)


def test_verify_attestation_binds_manifest_and_cosign(
    tmp_path: Path, monkeypatch
) -> None:
    path = manifest(tmp_path)
    attestation = {
        "schema_version": 1,
        "subject": {
            "release_id": "release-a",
            "manifest": "dist/current-release.json",
            "manifest_sha256": release_attestation.sha256(path),
            "delivery_sha256": "d" * 64,
        },
        "source": {"dirty": False},
        "quality_gates": [
            {"command": "make check", "status": "PASSED"},
            {"command": "make test-postgres-stress", "status": "PASSED"},
        ],
    }
    attestation_path = tmp_path / "attestation.json"
    attestation_path.write_text(json.dumps(attestation), encoding="utf-8")
    signature = tmp_path / "attestation.sig"
    signature.write_text("signature", encoding="utf-8")
    commands = []

    def run(command, **_kwargs):
        commands.append(command)
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(release_attestation.subprocess, "run", run)

    verified = release_attestation.verify_attestation(
        tmp_path,
        attestation_path,
        signature=signature,
        bundle=None,
        cosign_key="cosign.pub",
        certificate=None,
        certificate_identity=None,
        certificate_oidc_issuer=None,
    )

    assert verified["subject"]["release_id"] == "release-a"
    assert commands[0][:2] == ["cosign", "verify-blob"]


def test_verify_attestation_accepts_cosign_bundle(tmp_path: Path, monkeypatch) -> None:
    path = manifest(tmp_path)
    attestation = {
        "schema_version": 1,
        "subject": {
            "release_id": "release-a",
            "manifest": "dist/current-release.json",
            "manifest_sha256": release_attestation.sha256(path),
            "delivery_sha256": "d" * 64,
        },
        "source": {"dirty": False},
        "quality_gates": [
            {"command": "make check", "status": "PASSED"},
            {"command": "make test-postgres-stress", "status": "PASSED"},
        ],
    }
    attestation_path = tmp_path / "attestation.json"
    attestation_path.write_text(json.dumps(attestation), encoding="utf-8")
    bundle = tmp_path / "attestation.bundle.json"
    bundle.write_text("{}", encoding="utf-8")
    commands = []

    def run(command, **_kwargs):
        commands.append(command)
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(release_attestation.subprocess, "run", run)

    release_attestation.verify_attestation(
        tmp_path,
        attestation_path,
        signature=None,
        bundle=bundle,
        cosign_key="cosign.pub",
        certificate=None,
        certificate_identity=None,
        certificate_oidc_issuer=None,
    )

    assert "--bundle" in commands[0]


def test_verify_staging_attestation_requires_explicit_flag(
    tmp_path: Path, monkeypatch
) -> None:
    path = manifest(tmp_path, staging_only=True)
    attestation = {
        "schema_version": 1,
        "release_tier": "staging",
        "impact_base": "origin/main",
        "subject": {
            "release_id": "release-a",
            "manifest": "dist/current-release.json",
            "manifest_sha256": release_attestation.sha256(path),
            "delivery_sha256": "d" * 64,
        },
        "source": {"dirty": False},
        "quality_gates": [
            {"command": command, "status": "PASSED"}
            for command in release_attestation.STAGING_QUALITY_GATES
        ],
    }
    attestation_path = tmp_path / "attestation.json"
    attestation_path.write_text(json.dumps(attestation), encoding="utf-8")
    bundle = tmp_path / "attestation.bundle.json"
    bundle.write_text("{}", encoding="utf-8")
    commands = []
    monkeypatch.setattr(
        release_attestation.subprocess,
        "run",
        lambda command, **_kwargs: (
            commands.append(command) or subprocess.CompletedProcess(command, 0, "", "")
        ),
    )

    with pytest.raises(
        release_attestation.ReleaseAttestationError, match="staging-only"
    ):
        release_attestation.verify_attestation(
            tmp_path,
            attestation_path,
            signature=None,
            bundle=bundle,
            cosign_key="cosign.pub",
            certificate=None,
            certificate_identity=None,
            certificate_oidc_issuer=None,
        )

    verified = release_attestation.verify_attestation(
        tmp_path,
        attestation_path,
        signature=None,
        bundle=bundle,
        cosign_key="cosign.pub",
        certificate=None,
        certificate_identity=None,
        certificate_oidc_issuer=None,
        allow_staging=True,
    )

    assert verified["release_tier"] == "staging"
    assert commands, "staging attestation did not invoke Cosign verification"


def test_build_staging_attestation_requires_and_records_impact_base(
    tmp_path: Path, monkeypatch
) -> None:
    path = manifest(tmp_path, staging_only=True)
    monkeypatch.setattr(
        release_attestation,
        "source_state",
        lambda _root: {"git_commit": "a" * 40, "dirty": False, "diff_sha256": "b" * 64},
    )

    with pytest.raises(
        release_attestation.ReleaseAttestationError, match="impact test base"
    ):
        release_attestation.build_attestation(tmp_path, path, staging_only=True)

    value = release_attestation.build_attestation(
        tmp_path, path, staging_only=True, impact_base="origin/release"
    )

    assert value["impact_base"] == "origin/release"


def test_staging_attestation_binds_impact_plan(tmp_path: Path, monkeypatch) -> None:
    path = manifest(tmp_path, staging_only=True)
    impact_plan = tmp_path / "dist/staging-impact-plan.json"
    impact_plan.write_text('{"schema_version":1}\n', encoding="utf-8")
    monkeypatch.setattr(
        release_attestation,
        "source_state",
        lambda _root: {"git_commit": "a" * 40, "dirty": False, "diff_sha256": "b" * 64},
    )

    value = release_attestation.build_attestation(
        tmp_path,
        path,
        staging_only=True,
        impact_base="origin/release",
        impact_plan_path=impact_plan,
    )

    assert value["impact_plan"] == {
        "path": "dist/staging-impact-plan.json",
        "sha256": release_attestation.sha256(impact_plan),
    }


def test_production_attestation_binds_ci_gate(tmp_path: Path, monkeypatch) -> None:
    path = manifest(tmp_path)
    gate = tmp_path / "dist/ci-gate.json"
    gate.write_text('{"schema_version":1}\n', encoding="utf-8")
    monkeypatch.setattr(
        release_attestation,
        "source_state",
        lambda _root: {"git_commit": "a" * 40, "dirty": False, "diff_sha256": "b" * 64},
    )

    value = release_attestation.build_attestation(tmp_path, path, ci_gate_path=gate)

    assert value["ci_gate"] == {
        "path": "dist/ci-gate.json",
        "sha256": release_attestation.sha256(gate),
    }
    assert {item["command"] for item in value["quality_gates"]} == set(
        release_attestation.PROMOTED_QUALITY_GATES
    )

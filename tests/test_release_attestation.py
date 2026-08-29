from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from scripts import release_attestation


def manifest(root: Path, *, deployable: bool = True) -> Path:
    path = root / "dist/current-release.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": 3,
                "deployable": deployable,
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

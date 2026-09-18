from __future__ import annotations

import json
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from scripts import release_attestation
from scripts.e2e.regional import auth015_release as release
from scripts.e2e.regional.auth015_protocol import Auth015ProofError
from scripts.release_identity import canonical_sha256
from tests.regional._cov95_auth015_release import ReleaseFiles
from tests.regional._cov95_auth015_support import KEY_A
from tests.regional._cov95_identity_support import offline_guard as offline_guard


@pytest.fixture
def files(tmp_path, monkeypatch):
    return ReleaseFiles(tmp_path, monkeypatch)


def test_signed_release_uses_existing_verifier_and_binds_all_input_bytes(files):
    inputs = files.inputs()
    result = release.verify_release_inputs(inputs)
    assert result.release_id == "release-a"
    assert result.node_digest == "b" * 64 and result.node_wheel_sha256 == "a" * 64
    assert result.runtime_image == files.delivery["images"]["runtime"]["reference"]
    assert result.input_identity == release.release_input_identity(inputs)
    assert set(result.input_identity) == {
        "descriptor",
        "attestation",
        "signature_bundle",
        "public_key",
        "manifest",
        "release_id",
    }
    assert files.commands == [
        [
            "cosign",
            "verify-blob",
            "--bundle",
            str(files.bundle_path),
            "--key",
            str(files.public_key_path),
            str(files.attestation_path),
        ]
    ], "the proof must call signature verification, not trust a file's claimed digest"
    assert "BEGIN PUBLIC KEY" not in repr(result)


@pytest.mark.parametrize(
    "defect",
    [
        "missing",
        "extra",
        "schema-bool",
        "schema-other",
        "allow-string",
        "blank-path",
        "wrong-type",
        "bad-pin",
        "missing-root",
    ],
)
def test_release_descriptor_refuses_ambiguous_or_untrusted_inputs(files, defect):
    if defect == "missing":
        files.descriptor.pop("bundle")
    elif defect == "extra":
        files.descriptor["node_secret"] = KEY_A
    elif defect == "schema-bool":
        files.descriptor["schema_version"] = True
    elif defect == "schema-other":
        files.descriptor["schema_version"] = 2
    elif defect == "allow-string":
        files.descriptor["allow_staging"] = "true"
    elif defect == "blank-path":
        files.descriptor["bundle"] = ""
    elif defect == "wrong-type":
        files.descriptor["bundle"] = []
    elif defect == "bad-pin":
        files.descriptor["cosign_public_key_sha256"] = "invalid"
    else:
        files.descriptor["repository_root"] = str(files.root / "missing")
    files.write_descriptor()
    with pytest.raises(Auth015ProofError) as caught:
        files.inputs()
    assert KEY_A not in str(caught.value)
    assert files.commands == []


@pytest.mark.parametrize("release_id", ["", "release\nother", None])
def test_expected_release_identity_is_explicit_and_well_formed(files, release_id):
    with pytest.raises(Auth015ProofError, match="descriptor"):
        files.inputs(release_id)


@pytest.mark.parametrize("payload", [b"", b"{", b"[]"])
def test_incomplete_release_inputs_are_not_evidence(files, payload):
    files.descriptor_path.write_bytes(payload)
    with pytest.raises(Auth015ProofError):
        files.inputs()


def test_release_input_size_and_file_type_are_bounded(files, monkeypatch):
    with pytest.raises(Auth015ProofError, match="regular file"):
        release.read_input(files.root)
    monkeypatch.setattr(release, "MAX_INPUT_BYTES", 1)
    with pytest.raises(Auth015ProofError, match="exceeds limit"):
        release.read_input(files.bundle_path)


def test_unreadable_input_fails_without_disclosing_values(files, monkeypatch):
    def unreadable(*args, **kwargs):
        raise PermissionError(KEY_A)

    monkeypatch.setattr(Path, "open", unreadable)
    with pytest.raises(Auth015ProofError, match="cannot be read") as caught:
        release.read_input(files.bundle_path)
    assert KEY_A not in str(caught.value)


@pytest.mark.parametrize("manifest", ["../outside.json", "/absolute.json", "", True])
def test_attestation_cannot_select_a_foreign_or_ambiguous_manifest(files, manifest):
    value = json.loads(files.attestation_path.read_text())
    value["subject"]["manifest"] = manifest
    files.attestation_path.write_text(json.dumps(value))
    with pytest.raises(Auth015ProofError, match="manifest path"):
        release.release_input_identity(files.inputs())


def test_missing_manifest_binding_is_not_a_release_proof(files):
    files.attestation_path.write_text("{}")
    with pytest.raises(Auth015ProofError, match="manifest path"):
        release.release_input_identity(files.inputs())


def test_changed_public_key_is_rejected_before_cosign(files):
    files.public_key_path.write_text("different public key", encoding="ascii")
    with pytest.raises(Auth015ProofError, match="public-key pin"):
        release.verify_release_inputs(files.inputs())
    assert files.commands == []


def test_private_key_cannot_be_used_as_a_public_key_input(files):
    files.public_key_path.write_bytes(
        Ed25519PrivateKey.generate().private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    files.public_key_path.chmod(0o600)
    import hashlib

    files.descriptor["cosign_public_key_sha256"] = hashlib.sha256(
        files.public_key_path.read_bytes()
    ).hexdigest()
    files.write_descriptor()
    with pytest.raises(Auth015ProofError, match="pinned public key"):
        release.verify_release_inputs(files.inputs())
    assert files.commands == []


def test_unsigned_manifest_changes_are_rejected_by_real_attestation_validation(files):
    files.manifest["bundle_sha256"] = "9" * 64
    files.manifest_path.write_text(json.dumps(files.manifest))
    with pytest.raises(Auth015ProofError, match="signed-release verification failed"):
        release.verify_release_inputs(files.inputs())
    assert files.commands == []


def test_signature_failure_cannot_echo_verifier_diagnostics(files, monkeypatch):
    monkeypatch.setattr(
        release_attestation.subprocess,
        "run",
        lambda args, **kwargs: subprocess.CompletedProcess(args, 1, "", KEY_A),
    )
    with pytest.raises(
        Auth015ProofError, match="signed-release verification failed"
    ) as caught:
        release.verify_release_inputs(files.inputs())
    assert KEY_A not in str(caught.value)


def test_verification_race_invalidates_the_release_receipt(files, monkeypatch):
    def changed(command, **kwargs):
        files.bundle_path.write_text('{"changed":true}')
        return files.cosign(command, **kwargs)

    monkeypatch.setattr(release_attestation.subprocess, "run", changed)
    with pytest.raises(Auth015ProofError, match="changed during verification"):
        release.verify_release_inputs(files.inputs())


@pytest.mark.parametrize(
    "defect",
    [
        "legacy-schema",
        "schema-bool",
        "foreign-release",
        "missing-component",
        "invalid-digest",
        "digest-type",
        "delivery",
        "image",
        "image-type",
        "image-component",
        "protocol-bool",
        "protocol-zero",
        "local-code-drift",
    ],
)
def test_signed_but_incompatible_release_cannot_authorize_the_probe(files, defect):
    manifest = files.manifest
    if defect == "legacy-schema":
        manifest["schema_version"] = 3
    elif defect == "schema-bool":
        manifest["schema_version"] = True
    elif defect == "foreign-release":
        manifest["release_id"] = "release-other"
    elif defect == "missing-component":
        manifest["components"].pop("node_runtime")
    elif defect in {"invalid-digest", "digest-type"}:
        manifest["components"]["node_runtime"]["wheel_sha256"] = (
            "bad" if defect == "invalid-digest" else []
        )
    elif defect == "delivery":
        manifest["delivery"]["sha256"] = "8" * 64
    elif defect in {"image", "image-type", "image-component"}:
        image = manifest["delivery"]["images"]["runtime"]
        if defect == "image-component":
            image["components"]["control_plane"]["module_digest"] = "1" * 64
        else:
            image["reference"] = "registry/image:latest" if defect == "image" else None
        manifest["delivery"]["sha256"] = canonical_sha256(
            {
                key: value
                for key, value in manifest["delivery"].items()
                if key != "sha256"
            }
        )
    elif defect in {"protocol-bool", "protocol-zero"}:
        manifest["protocol_versions"]["agent"] = (
            True if defect == "protocol-bool" else 0
        )
    else:
        manifest["components"]["node_runtime"]["module_digest"] = "9" * 64
    files.write_manifest()
    with pytest.raises(Auth015ProofError):
        release.verify_release_inputs(files.inputs())


def test_allow_staging_is_explicit_in_the_descriptor(files):
    files.descriptor["allow_staging"] = True
    files.write_descriptor()
    inputs = files.inputs()
    assert inputs.allow_staging is True
    assert release.verify_release_inputs(inputs).release_id == "release-a"
    with pytest.raises(Auth015ProofError):
        release.verify_release_inputs(
            replace(inputs, expected_release_id="release-other")
        )

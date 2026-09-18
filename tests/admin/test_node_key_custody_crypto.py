from __future__ import annotations

import hashlib
import json
import stat
import subprocess
from types import SimpleNamespace

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

from gpu_fault.admin import node_key_custody_crypto as crypto
from gpu_fault.admin.node_key_custody_models import CustodyError, Trust, canonical
from tests.deploy._node_key_custody_support import Authorities, ProvisionFixture
from tests.regional._cov95_identity_support import offline_guard as offline_guard


def test_bounded_regular_file_and_exclusive_durable_receipt(tmp_path):
    path = tmp_path / "receipt"
    crypto.write_once(path, b"synthetic-receipt")
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert crypto.read_regular(path, private=True) == b"synthetic-receipt"
    with pytest.raises(CustodyError, match="already exists"):
        crypto.write_once(path, b"replacement")
    assert path.read_bytes() == b"synthetic-receipt"


@pytest.mark.parametrize("content", [b"", b"12345"])
def test_empty_or_oversized_input_is_not_evidence(tmp_path, content):
    path = tmp_path / "input"
    path.write_bytes(content)
    with pytest.raises(CustodyError, match="empty, oversized or changed"):
        crypto.read_regular(path, limit=4)


def test_unreadable_input_and_state_directory_are_sanitized(tmp_path):
    with pytest.raises(CustodyError, match="cannot be read safely"):
        crypto.read_regular(tmp_path / "missing")
    with pytest.raises(CustodyError, match="unavailable"):
        crypto.private_directory(tmp_path / "missing")
    path = tmp_path / "regular"
    path.write_bytes(b"only-a-file")
    with pytest.raises(CustodyError, match="mode-0700"):
        crypto.private_directory(path)


@pytest.mark.parametrize("change", ["not-regular", "hardlink", "owner", "modified"])
def test_fd_metadata_must_remain_regular_private_and_stable(
    tmp_path, monkeypatch, change
):
    path = tmp_path / "private"
    path.write_bytes(b"synthetic-private-content")
    path.chmod(0o600)
    actual = path.stat()
    calls = 0

    def metadata(_fd):
        nonlocal calls
        calls += 1
        return SimpleNamespace(
            st_mode=stat.S_IFDIR | 0o600 if change == "not-regular" else actual.st_mode,
            st_nlink=2 if change == "hardlink" else 1,
            st_uid=actual.st_uid + 1 if change == "owner" else actual.st_uid,
            st_size=actual.st_size,
            st_mtime_ns=actual.st_mtime_ns + int(change == "modified" and calls > 1),
            st_ctime_ns=actual.st_ctime_ns,
        )

    monkeypatch.setattr(crypto.os, "fstat", metadata)
    with pytest.raises(CustodyError):
        crypto.read_regular(path, private=True)


def test_a_public_input_need_not_be_a_private_file(tmp_path):
    path = tmp_path / "public"
    path.write_bytes(b"public-data")
    path.chmod(0o644)
    assert crypto.read_regular(path) == b"public-data"
    with pytest.raises(CustodyError, match="controlled regular"):
        crypto.read_regular(path, private=True)


@pytest.mark.parametrize(
    "raw", [b"not-json", b"{}", b"[]", b'{"schema_version":1,"schema_version":1}']
)
def test_duplicate_and_invalid_json_never_becomes_a_trust_document(raw):
    with pytest.raises(CustodyError):
        crypto.parse(Trust, raw)


def test_external_pin_and_canonical_public_key_pins_cannot_be_self_replaced(tmp_path):
    authorities = Authorities(tmp_path)
    with pytest.raises(CustodyError, match="external trust pin"):
        crypto.CustodyCrypto(
            authorities.trust_path, "f" * 64, runner=authorities.runner
        )
    (tmp_path / "approval.pem").write_bytes((tmp_path / "witness.pem").read_bytes())
    with pytest.raises(CustodyError, match="public key differs"):
        authorities.crypto()


@pytest.mark.parametrize("same", ["public-key", "kms-key"])
def test_approval_producer_and_witness_must_be_independent_even_with_different_pem_spelling(
    tmp_path, same
):
    authorities = Authorities(tmp_path)
    trust = authorities.trust
    if same == "public-key":
        (tmp_path / "witness.pem").write_bytes(
            (tmp_path / "approval.pem").read_bytes() + b"\n"
        )
        witness = trust.witness.model_copy(
            update={"public_key_sha256": trust.approval.public_key_sha256}
        )
    else:
        witness = trust.witness.model_copy(
            update={"kms_key_arn": trust.provisioner.kms_key_arn}
        )
    authorities.trust_path.write_bytes(
        canonical(trust.model_copy(update={"witness": witness}))
    )
    pin = hashlib.sha256(authorities.trust_path.read_bytes()).hexdigest()
    with pytest.raises(CustodyError, match="independent keys"):
        crypto.CustodyCrypto(authorities.trust_path, pin, runner=authorities.runner)


def test_only_p256_public_keys_can_be_custody_authorities(tmp_path):
    authorities = Authorities(tmp_path)
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048).public_key()
    (tmp_path / "approval.pem").write_bytes(
        key.public_bytes(
            serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo
        )
    )
    with pytest.raises(CustodyError, match="P-256"):
        authorities.crypto()


@pytest.mark.parametrize("failure", ["raises", "nonzero"])
def test_external_signature_failures_never_echo_raw_diagnostics(tmp_path, failure):
    authorities = Authorities(tmp_path)
    sentinel = "synthetic-sensitive-diagnostic-" * 8

    def runner(*args, **kwargs):
        if failure == "raises":
            raise OSError(sentinel)
        return subprocess.CompletedProcess(args[0], 1, sentinel, sentinel)

    with pytest.raises(CustodyError) as caught:
        crypto.CustodyCrypto(
            authorities.trust_path, authorities.trust_pin, runner=runner
        )
    assert sentinel not in str(caught.value)


@pytest.mark.parametrize("signature", ["@" * 64, "A" * 64, "A" * 512])
def test_signature_encoding_and_decoded_size_are_bounded(tmp_path, signature):
    fixture = ProvisionFixture(tmp_path)
    signed = fixture.authorities.envelope(fixture.authorization(), "approval")
    with pytest.raises(CustodyError, match="encoding"):
        fixture.crypto.verify(
            signed.model_copy(update={"signature": signature}), "approval"
        )


def test_producer_and_unknown_roles_cannot_issue_authorizations(tmp_path):
    fixture = ProvisionFixture(tmp_path)
    authorization = fixture.authorization()
    for role in ("approval", "unknown"):
        with pytest.raises(CustodyError, match="own authorization"):
            fixture.crypto.sign(authorization, role)
    with pytest.raises(CustodyError, match="unknown"):
        fixture.crypto.authority("unknown")


@pytest.mark.parametrize(
    "failure", ["json", "identity", "algorithm", "missing", "signature"]
)
def test_kms_returned_signer_algorithm_and_signature_are_verified(tmp_path, failure):
    fixture = ProvisionFixture(tmp_path)
    original = fixture.authorities.runner

    def runner(arguments, **kwargs):
        result = original(arguments, **kwargs)
        if arguments[:3] != ["aws", "kms", "sign"]:
            return result
        if failure == "json":
            result.stdout = "not-json"
            return result
        value = json.loads(result.stdout)
        if failure == "identity":
            value["KeyId"] = fixture.authorities.trust.approval.kms_key_arn
        elif failure == "algorithm":
            value["SigningAlgorithm"] = "RSASSA_PKCS1_V1_5_SHA_256"
        elif failure == "missing":
            del value["Signature"]
        else:
            value["Signature"] = "A" * 96
        result.stdout = json.dumps(value)
        return result

    fixture.crypto.runner = runner
    with pytest.raises(CustodyError):
        fixture.crypto.sign(fixture.authorization(), "provisioner")


def test_literal_numeric_booleans_are_not_valid_custody_protocol_fields(tmp_path):
    fixture = ProvisionFixture(tmp_path)
    chain = fixture.provision(fixture.session())
    document = chain.model_dump(mode="json")
    document["transactions"][0]["started"]["statement"]["master_source_verified"] = 1
    with pytest.raises(CustodyError, match="schema"):
        crypto.parse(type(chain), json.dumps(document).encode())


def test_private_command_environment_drops_raw_material_and_keeps_role_references(
    monkeypatch,
):
    monkeypatch.setenv("GPU_FAULT_NODE_ACTION_SECRET", "synthetic-node-value")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "synthetic-aws-value")
    monkeypatch.setenv("AWS_WEB_IDENTITY_TOKEN_FILE", "/synthetic/role-file")
    environment = crypto.private_command_environment()
    assert "GPU_FAULT_NODE_ACTION_SECRET" not in environment
    assert "AWS_SECRET_ACCESS_KEY" not in environment
    assert environment["AWS_WEB_IDENTITY_TOKEN_FILE"] == "/synthetic/role-file"

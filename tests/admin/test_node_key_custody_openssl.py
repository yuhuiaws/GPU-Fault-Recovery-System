from __future__ import annotations

import subprocess

import pytest

from gpu_fault.admin.node_key_custody_crypto import CustodyCrypto
from gpu_fault.admin.node_key_custody_models import CustodyError
from tests.deploy._node_key_custody_support import ProvisionFixture


def test_real_openssl_verifies_an_independently_signed_receipt_and_rejects_forgery(
    tmp_path,
):
    fixture = ProvisionFixture(tmp_path)
    calls = []

    def openssl(arguments, **kwargs):
        assert arguments[0] == "openssl", (
            "hermetic verification may never call a cloud signer"
        )
        assert "AWS_SECRET_ACCESS_KEY" not in kwargs["environment"]
        calls.append(arguments)
        return subprocess.run(
            arguments,
            capture_output=True,
            text=True,
            check=False,
            timeout=kwargs["timeout_seconds"],
            env={
                "HOME": "/tmp",
                "PATH": "/usr/local/bin:/usr/bin:/bin",
                "AWS_CONFIG_FILE": "/dev/null",
                "AWS_SHARED_CREDENTIALS_FILE": "/dev/null",
                "AWS_EC2_METADATA_DISABLED": "true",
                "KUBECONFIG": "/dev/null",
            },
        )

    verifier = CustodyCrypto(
        fixture.authorities.trust_path, fixture.authorities.trust_pin, runner=openssl
    )
    authorization = fixture.authorization()
    signed = fixture.authorities.envelope(authorization, "approval")
    assert verifier.verify(signed, "approval") == authorization
    forged = signed.model_copy(
        update={
            "statement": authorization.model_copy(update={"producer_sha256": "f" * 64})
        }
    )
    with pytest.raises(CustodyError, match="signature command failed"):
        verifier.verify(forged, "approval")
    assert sum(call[1] == "dgst" for call in calls) == 2

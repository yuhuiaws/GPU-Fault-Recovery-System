"""Keep administrator tests off the AWS CLI when they resolve the operator.

Every mutating admin path now stamps the operator's STS identity into what it
writes. The shared guard already refuses to run ``aws`` from a test, so the
resolver is replaced here with a fixed ARN; the resolver's own tests inject
their runner directly and never go through this seam.
"""

from __future__ import annotations

import pytest

TEST_OPERATOR_ARN = "arn:aws:sts::123456789012:assumed-role/Admin/test-operator"


@pytest.fixture(autouse=True)
def _fixed_operator_identity(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    if request.module.__name__.rsplit(".", 1)[-1] == "test_operator_identity":
        # The resolver's own tests exercise the real function with an injected
        # runner; stubbing it here would make them test the stub.
        return
    try:
        from gpu_fault.admin import operator_identity
    except ImportError:  # the module is what the red run is missing
        return
    monkeypatch.setattr(
        operator_identity, "caller_identity_arn", lambda **_kwargs: TEST_OPERATOR_ARN
    )

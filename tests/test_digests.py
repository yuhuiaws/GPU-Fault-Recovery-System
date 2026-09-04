"""The digest contract every artifact pin and config digest in the system shares.

`gpu_fault.digests` replaced eight private copies of the same regex. That is only
safe if the surviving copy accepts and rejects exactly what the callers assumed, so
these tests state the contract, and the last two check that the model validators that
used to carry their own body still enforce it.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from gpu_fault.digests import DIGEST_VALUE_MESSAGE, SHA256_PATTERN, normalized_sha256
from gpu_fault.fleet_compatibility import FleetCompatibilityPolicy
from gpu_fault.fleet_deployment import FleetDeploymentRequest


def test_mixed_case_input_is_folded_not_rejected() -> None:
    """Digests are compared for equality, so one spelling has to win.

    `sha256sum` output and a digest pasted from a console differ only in case; the
    pins they produce have to compare equal, and lowercase is what the rest of the
    system stores.
    """

    assert normalized_sha256("AB" * 32) == "ab" * 32


def test_absent_pin_passes_through() -> None:
    """`None` means "no pin declared", which is not the same as an unusable pin."""

    assert normalized_sha256(None) is None


@pytest.mark.parametrize(
    "value", ["", "a" * 63, "a" * 65, "g" * 64, " " + "a" * 64, "sha256:" + "a" * 64]
)
def test_anything_that_is_not_a_bare_sha256_hex_string_is_rejected(value: str) -> None:
    with pytest.raises(ValueError, match=DIGEST_VALUE_MESSAGE):
        normalized_sha256(value)


def test_pattern_rejects_a_trailing_newline_even_under_match() -> None:
    """Callers use the pattern directly, and some of them will reach for `match`.

    A digest read from a file or from a command's stdout arrives with a newline
    attached. Under `^...$` that value matches, so the caller stores a 65-character
    "digest" that compares unequal to every real one. `\\A...\\Z` is what makes the
    two spellings behave the same.
    """

    assert SHA256_PATTERN.match("a" * 64 + "\n") is None
    assert SHA256_PATTERN.match("x" + "a" * 64) is None
    assert SHA256_PATTERN.fullmatch("a" * 64) is not None


def test_compatibility_policy_digests_fold_case_and_reject_junk() -> None:
    policy = FleetCompatibilityPolicy(required_artifact_sha256="AB" * 32)

    assert policy.required_artifact_sha256 == "ab" * 32
    with pytest.raises(ValidationError, match=DIGEST_VALUE_MESSAGE):
        FleetCompatibilityPolicy(required_config_digest="not-a-digest")


def test_deployment_request_digests_fold_case_and_reject_junk() -> None:
    request = FleetDeploymentRequest(
        cluster_id="gpu-a",
        node_ids=["node-1"],
        desired_agent_version="1.2.3",
        desired_artifact_sha256="AB" * 32,
        desired_policy_version="policy-v1",
        desired_runtime_profile_version="profile-v1",
        desired_config_digest="cd" * 32,
    )

    assert request.desired_artifact_sha256 == "ab" * 32
    with pytest.raises(ValidationError, match=DIGEST_VALUE_MESSAGE):
        FleetDeploymentRequest(
            cluster_id="gpu-a",
            node_ids=["node-1"],
            desired_agent_version="1.2.3",
            desired_artifact_sha256="a" * 63,
            desired_policy_version="policy-v1",
            desired_runtime_profile_version="profile-v1",
            desired_config_digest="cd" * 32,
        )

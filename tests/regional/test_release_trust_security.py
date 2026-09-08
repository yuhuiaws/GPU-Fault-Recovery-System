"""Trust boundary guards for the regional release pipeline.

Covers image digest pinning (H-13), IAM wildcard rejection (M-15), the NLB
security-group full-rule validation (M-17), and the plan/apply manifest pin
(M-23).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from gpu_fault_release import regional_admin_checks as admin_checks
from gpu_fault_release import regional_release_rendering as rendering
from gpu_fault_release import rollout
from gpu_fault_release.regional_release_config import ReleaseError
from gpu_fault_release.regional_release_iam import validate_executor_iam_documents
from gpu_fault_release.regional_release_progress import RollbackCompensationPlan
from gpu_fault_release.regional_release_rollback_context import (
    rollback_identity_context,
    rollback_target_arguments,
)
from gpu_fault_release.regional_release_state import (
    require_consistent_images,
    require_digest_pinned_image,
)
from gpu_fault_release.regional_release_validation import validate_rollback

DIGEST = "registry.example.com/gpu-fault@sha256:" + "a" * 64
OTHER_DIGEST = "registry.example.com/gpu-fault@sha256:" + "b" * 64


# --------------------------------------------------------------------------- #
# H-13: container images must be digest-pinned at every release/rollback sink.
# --------------------------------------------------------------------------- #


def test_require_digest_pinned_image_accepts_a_digest() -> None:
    assert require_digest_pinned_image("runtime", DIGEST) == DIGEST


@pytest.mark.parametrize(
    "image",
    [
        "registry.example.com/gpu-fault:latest",
        "registry.example.com/gpu-fault:v1",
        "registry.example.com/gpu-fault",
        "",
        None,
        "registry.example.com/gpu-fault@sha256:tooshort",
    ],
)
def test_require_digest_pinned_image_rejects_mutable_tags(image: object) -> None:
    with pytest.raises(ReleaseError):
        require_digest_pinned_image("runtime", image)  # type: ignore[arg-type]


def test_require_consistent_images_reports_the_observed_image() -> None:
    # require_consistent_images is the read-only capture of live state; it does
    # not itself pin (a forward release may legitimately observe and replace a
    # legacy tag baseline). Digest pinning is enforced at the downstream
    # rollback/resume/validation sinks that consume this value.
    assert (
        require_consistent_images(
            "runtime", {"cpu/a": "img:latest", "cpu/b": "img:latest"}
        )
        == "img:latest"
    )
    with pytest.raises(ReleaseError):
        require_consistent_images("runtime", {"cpu/a": DIGEST, "cpu/b": OTHER_DIGEST})


def _rollback_release() -> SimpleNamespace:
    return SimpleNamespace(
        config=SimpleNamespace(clusters=[]),
        runtime_image=DIGEST,
        node_installer_image=DIGEST,
    )


def test_rollback_target_arguments_rejects_tag_runtime_image() -> None:
    with pytest.raises(ReleaseError):
        rollback_target_arguments(
            _rollback_release(),
            previous={},
            metadata={},
            artifact="art",
            config_digest="cfg",
            profile="hyperpod-v1",
            executor_artifact="exec",
            executor_compatibility="compat",
            runtime_image="mutable:tag",
        )


def test_rollback_target_arguments_accepts_digest_targets() -> None:
    arguments = rollback_target_arguments(
        _rollback_release(),
        previous={"node_installer_image": OTHER_DIGEST},
        metadata={},
        artifact="art",
        config_digest="cfg",
        profile="hyperpod-v1",
        executor_artifact="exec",
        executor_compatibility="compat",
        runtime_image=DIGEST,
    )
    assert arguments["runtime_image"] == DIGEST
    assert arguments["node_installer_image"] == OTHER_DIGEST


def test_rollback_identity_context_refuses_a_non_digest_runtime() -> None:
    compensation = RollbackCompensationPlan(
        global_components=frozenset(), cluster_components={}, conservative=False
    )
    with pytest.raises(ReleaseError):
        rollback_identity_context(
            _rollback_release(), {"runtime_image": "mutable:tag"}, compensation
        )


def test_validate_rollback_refuses_a_non_digest_runtime() -> None:
    release = SimpleNamespace(
        runner=SimpleNamespace(dry_run=False),
        runtime_image=DIGEST,
        _config_map_data=lambda name: {},
    )
    with pytest.raises(ReleaseError):
        validate_rollback(release, {"metadata": {}, "runtime_image": "mutable:tag"})


# --------------------------------------------------------------------------- #
# M-15: IAM policy documents must not grant wildcard actions or resources.
# --------------------------------------------------------------------------- #

ROLE = "arn:aws:iam::123456789012:role/gpu-fault-executor"


def _scoped_document() -> dict:
    return {
        "Statement": [
            {
                "Effect": "Allow",
                "Action": ["sagemaker:DescribeCluster"],
                "Resource": ["arn:aws:sagemaker:us-east-1:123456789012:cluster/abc"],
            }
        ]
    }


def test_scoped_iam_document_passes() -> None:
    validate_executor_iam_documents(ROLE, [_scoped_document()])


@pytest.mark.parametrize(
    "statement",
    [
        {"Effect": "Allow", "Action": "*", "Resource": "*"},
        {"Effect": "Allow", "Action": ["iam:*"], "Resource": ["*"]},
        {"Effect": "Allow", "Action": ["sts:*"], "Resource": ["*"]},
        {"Effect": "Allow", "Action": ["sagemaker:DescribeCluster"], "Resource": "*"},
        {"Effect": "Allow", "NotAction": "iam:*", "Resource": "*"},
    ],
)
def test_wildcard_iam_document_is_rejected(statement: dict) -> None:
    with pytest.raises(ReleaseError):
        validate_executor_iam_documents(ROLE, [{"Statement": [statement]}])


# --------------------------------------------------------------------------- #
# M-17: the NLB security-group rule must be exactly tcp/443-443.
# --------------------------------------------------------------------------- #


def _exact_https(cidr: str) -> dict:
    return {
        "IpProtocol": "tcp",
        "FromPort": 443,
        "ToPort": 443,
        "IpRanges": [{"CidrIp": cidr}],
    }


def test_scoped_https_rule_to_internal_cidr_is_accepted() -> None:
    assert admin_checks.nlb_https_rule_violation(_exact_https("10.0.0.0/8")) is None


def test_exact_https_rule_open_to_the_internet_is_rejected() -> None:
    assert admin_checks.nlb_https_rule_violation(_exact_https("0.0.0.0/0")), (
        "0.0.0.0/0 on 443 is a violation"
    )


def test_wide_port_range_reaching_443_is_rejected() -> None:
    wide = {
        "IpProtocol": "tcp",
        "FromPort": 443,
        "ToPort": 65535,
        "IpRanges": [{"CidrIp": "0.0.0.0/0"}],
    }
    assert admin_checks.nlb_https_rule_violation(wide), wide


def test_all_protocol_rule_reaching_443_is_rejected() -> None:
    all_traffic = {"IpProtocol": "-1", "IpRanges": [{"CidrIp": "0.0.0.0/0"}]}
    assert admin_checks.permission_reaches_https(all_traffic), all_traffic
    assert admin_checks.nlb_https_rule_violation(all_traffic), all_traffic


def test_wide_range_to_internal_cidr_is_still_rejected() -> None:
    # Broader than tcp/443-443 even without an internet CIDR: opens extra ports.
    wide_internal = {
        "IpProtocol": "tcp",
        "FromPort": 400,
        "ToPort": 500,
        "IpRanges": [{"CidrIp": "10.0.0.0/8"}],
    }
    assert admin_checks.nlb_https_rule_violation(wide_internal), wide_internal


def test_rule_that_does_not_reach_443_is_ignored() -> None:
    ssh = {
        "IpProtocol": "tcp",
        "FromPort": 22,
        "ToPort": 22,
        "IpRanges": [{"CidrIp": "0.0.0.0/0"}],
    }
    assert admin_checks.nlb_https_rule_violation(ssh) is None


# --------------------------------------------------------------------------- #
# M-23: apply must render from the approved plan, not a mutated working tree.
# --------------------------------------------------------------------------- #


def test_render_release_payload_refuses_tree_drift(monkeypatch) -> None:
    payload = {"cpu": {"gpu-fault-api": [{"kind": "Deployment"}]}}
    monkeypatch.setattr(rendering, "_render_release_payload", lambda release: payload)
    # The digest a plan pins is whatever the public renderer reports for an
    # unpinned release of the same tree.
    approved = rendering.rendered_release_manifest_sha256(
        SimpleNamespace(approved_manifest_digest=None)
    )

    matched = SimpleNamespace(approved_manifest_digest=approved)
    assert rendering.render_release_payload(matched) == payload

    drifted = SimpleNamespace(approved_manifest_digest=OTHER_DIGEST)
    with pytest.raises(ReleaseError):
        rendering.render_release_payload(drifted)

    unpinned = SimpleNamespace(approved_manifest_digest=None)
    assert rendering.render_release_payload(unpinned) == payload


def test_enforce_manifest_plan_pin_is_a_noop_until_pinned(monkeypatch) -> None:
    calls: list[object] = []
    monkeypatch.setattr(
        rollout, "render_release_payload", lambda release: calls.append(release)
    )
    release = rollout.RegionalRelease.__new__(rollout.RegionalRelease)
    release.approved_manifest_digest = None

    release.enforce_manifest_plan_pin()
    assert calls == []

    release.pin_approved_manifest_plan("cafebabe")
    assert release.approved_manifest_digest == "cafebabe"
    release.enforce_manifest_plan_pin()
    assert calls == [release]

    release.pin_approved_manifest_plan(None)
    assert release.approved_manifest_digest is None

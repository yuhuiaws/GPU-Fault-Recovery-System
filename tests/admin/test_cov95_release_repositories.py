from __future__ import annotations

import json
import subprocess
from dataclasses import replace

import pytest

from gpu_fault.admin import release_repositories as repositories
from gpu_fault.admin.bootstrap_common import SITE_TAG_KEY, BootstrapError, CommandRunner
from tests.admin._cov95_join_support import target


class EcrTransport(CommandRunner):
    def __init__(self):
        super().__init__()
        self.commands = []
        self.mutations = []
        self.inventory = 1
        self.owner = "release-bootstrap"
        self.overrides = {}
        self.read_error = None
        self.missing_policy = False
        self.policy_response = None
        self.applied_policy = None
        self.policy = {
            "rules": [
                {
                    "rulePriority": 1,
                    "description": "Expire untagged BuildKit cache artifacts after 7 days",
                    "selection": {
                        "tagStatus": "untagged",
                        "countType": "sinceImagePushed",
                        "countUnit": "days",
                        "countNumber": 7,
                    },
                    "action": {"type": "expire"},
                }
            ]
        }

    def repository(self, name):
        cache = "runtime-cache" in name
        return {
            "repositoryName": name,
            "repositoryArn": f"arn:aws:ecr:us-east-1:123456789012:repository/{name}",
            "repositoryUri": "example.invalid/" + name,
            "imageTagMutability": "MUTABLE" if cache else "IMMUTABLE",
            "imageScanningConfiguration": {"scanOnPush": not cache},
            "encryptionConfiguration": {"encryptionType": "AES256"},
            **self.overrides,
        }

    def command(self, arguments, **_options):
        assert arguments[:2] == ["aws", "ecr"], "unexpected ECR transport command"
        self.commands.append(list(arguments))
        action = arguments[2]
        if self.read_error == action:
            return subprocess.CompletedProcess(arguments, 1, "", "example AccessDenied")
        if action == "describe-repositories":
            name = arguments[arguments.index("--repository-names") + 1]
            value = {"repositories": [self.repository(name)] * self.inventory}
        elif action == "get-lifecycle-policy":
            if self.missing_policy:
                return subprocess.CompletedProcess(
                    arguments, 254, "", "LifecyclePolicyNotFoundException"
                )
            value = (
                self.policy_response
                if self.policy_response is not None
                else {"lifecyclePolicyText": json.dumps(self.policy)}
            )
        else:
            raise AssertionError("unexpected ECR read")
        return subprocess.CompletedProcess(
            arguments, 0, value if isinstance(value, str) else json.dumps(value), ""
        )

    def aws_json(self, region, service, operation, *arguments, **options):
        assert (region, service) == ("us-east-1", "ecr")
        if options.get("mutate"):
            self.mutations.append(operation)
        if operation == "list-tags-for-resource":
            return {
                "tags": [
                    {"Key": SITE_TAG_KEY, "Value": "example"},
                    {"Key": "gpu-fault:owner", "Value": self.owner},
                ]
            }
        assert operation == "put-lifecycle-policy", "unexpected ECR mutation"
        return (
            self.applied_policy
            if self.applied_policy is not None
            else {
                "lifecyclePolicyText": arguments[
                    arguments.index("--lifecycle-policy-text") + 1
                ]
            }
        )


@pytest.fixture
def context(monkeypatch):
    transport = EcrTransport()
    monkeypatch.setattr(repositories, "run_command", transport.command)
    return transport, replace(target(), role="cpu")


def ensure(context):
    transport, cpu = context
    return repositories.ensure_release_repositories(
        transport, cpu=cpu, site_id="example"
    )


@pytest.mark.parametrize(
    "change", ["not-unique", "owner", "mutability", "scan", "encryption"]
)
def test_ecr_repository_identity_and_policy_drift_blocks_reuse(context, change):
    transport = context[0]
    if change == "not-unique":
        transport.inventory = 2
    elif change == "owner":
        transport.owner = "foreign"
    elif change == "mutability":
        transport.overrides["imageTagMutability"] = "UNKNOWN"
    elif change == "scan":
        transport.overrides["imageScanningConfiguration"] = {"scanOnPush": False}
    else:
        transport.overrides["encryptionConfiguration"] = {"encryptionType": "KMS"}
    with pytest.raises(BootstrapError, match="uniquely|owner|tags|scan-on-push|AES256"):
        ensure(context)
    assert transport.mutations == []


@pytest.mark.parametrize(
    "response",
    [
        "invalid-json",
        [],
        {},
        {"lifecyclePolicyText": "invalid"},
        {"lifecyclePolicyText": "[]"},
    ],
)
def test_ecr_cache_policy_must_be_a_complete_structured_response(context, response):
    transport = context[0]
    transport.policy_response = response
    with pytest.raises(BootstrapError, match="lifecycle.*invalid|policy is missing"):
        ensure(context)
    assert transport.mutations == []


@pytest.mark.parametrize("operation", ["describe-repositories", "get-lifecycle-policy"])
def test_ecr_read_failure_does_not_authorize_creation_or_policy_rewrite(
    context, operation
):
    transport = context[0]
    transport.read_error = operation
    with pytest.raises(BootstrapError, match="cannot inspect"):
        ensure(context)
    assert transport.mutations == []


def test_missing_cache_policy_is_verified_against_the_applied_response(context):
    transport = context[0]
    transport.missing_policy = True
    transport.applied_policy = {"lifecyclePolicyText": '{"rules":[]}'}
    with pytest.raises(BootstrapError, match="was not applied"):
        ensure(context)
    assert transport.mutations == ["put-lifecycle-policy"]


def test_converged_release_repositories_preserve_registry_purpose_and_digest(context):
    result = ensure(context)
    assert result["runtime"]["purpose"] == "runtime"
    assert result["cache"]["purpose"] == "build-cache"
    assert len(result["cache"]["lifecycle_policy_sha256"]) == 64
    assert context[0].mutations == []

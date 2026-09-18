from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml  # type: ignore[import-untyped,unused-ignore]

from gpu_fault.admin.bootstrap_common import ClusterIdentity
from gpu_fault.admin.bootstrap_services import pod_identity_trust
from tests.admin.test_admin_bootstrap_amp_installer import (
    ACCOUNT,
    CLUSTER,
    CLUSTER_ARN,
    NAMESPACE,
    REGION,
    ROLE_NAME,
    Installer,
)

ROLE_ARN = f"arn:aws:iam::{ACCOUNT}:role/{ROLE_NAME}"


def role_document() -> dict[str, Any]:
    cluster = ClusterIdentity(
        input_arn=CLUSTER_ARN,
        role="cpu",
        region=REGION,
        account_id=ACCOUNT,
        hyperpod_arn="",
        hyperpod_name="",
        eks_arn=CLUSTER_ARN,
        eks_name=CLUSTER,
        vpc_id="vpc-fixture",
        subnet_ids=("subnet-fixture",),
        node_recovery="None",
        context=CLUSTER_ARN,
    )
    return {
        "Role": {
            "RoleName": ROLE_NAME,
            "Arn": ROLE_ARN,
            "AssumeRolePolicyDocument": pod_identity_trust(cluster),
        }
    }


def association(namespace: str = NAMESPACE) -> dict[str, str]:
    return {
        "clusterName": CLUSTER,
        "namespace": namespace,
        "serviceAccount": "gpu-fault-adot",
        "associationId": "assoc-a",
        "roleArn": ROLE_ARN,
    }


@pytest.fixture
def installer(tmp_path: Path) -> Installer:
    value = Installer(tmp_path)
    value.arguments = ["--runtime-only"]
    value.answer("iam_get-role", json.dumps(role_document()))
    value.answer(
        "eks_list-pod-identity-associations",
        json.dumps({"associations": [association()]}),
    )
    value.answer(
        "eks_describe-pod-identity-association",
        json.dumps({"association": association()}),
    )
    return value


def assert_no_foundation_writes(installer: Installer) -> None:
    for forbidden in (
        "aws iam create-",
        "aws iam put-",
        "aws iam update-",
        "aws iam attach-",
        "aws eks create-pod-identity",
        "aws eks update-pod-identity",
        "aws sns create-",
        "aws sns set-",
        "aws sns subscribe",
    ):
        assert not installer.matching(forbidden), (
            f"runtime-only monitoring mutated foundation resource: {forbidden}"
        )


def test_runtime_only_reuses_bootstrap_identity_without_restarting_healthy_adot(
    installer: Installer,
) -> None:
    installer.run()
    assert installer.matching("apply -f"), "runtime configuration was not applied"
    assert not installer.matching("rollout restart"), (
        "an unchanged bootstrap identity forced an ADOT restart"
    )
    assert_no_foundation_writes(installer)


def test_runtime_only_still_rolls_changed_adot_configuration(
    installer: Installer,
) -> None:
    installer.apply_output = (
        "configmap/gpu-fault-adot configured\ndeployment.apps/gpu-fault-adot unchanged"
    )
    installer.run()
    assert installer.matching("rollout restart deployment/gpu-fault-adot"), (
        "changed runtime configuration did not restart its collector"
    )
    assert_no_foundation_writes(installer)


@pytest.mark.parametrize(
    "operation",
    [
        "iam_get-role",
        "iam_get-role-policy",
        "eks_list-pod-identity-associations",
        "eks_describe-pod-identity-association",
        "sns_get-topic-attributes",
    ],
)
def test_unreadable_or_missing_prerequisites_stop_before_runtime_mutation(
    installer: Installer, operation: str
) -> None:
    installer.missing(operation)
    result = installer.attempt()
    assert result.returncode != 0, "unknown prerequisite was accepted"
    assert "ERROR:" in result.stderr, "the failed prerequisite was not identified"
    assert not installer.matching("apply -f"), (
        "runtime mutation preceded prerequisite identity verification"
    )
    assert_no_foundation_writes(installer)


@pytest.mark.parametrize("field", ["Arn", "RoleName", "trust-scope", "extra-trust"])
def test_runtime_only_refuses_iam_identity_or_trust_drift(
    installer: Installer, field: str
) -> None:
    document = role_document()
    if field in {"Arn", "RoleName"}:
        document["Role"][field] = "other-identity"
    else:
        trust = document["Role"]["AssumeRolePolicyDocument"]
        if field == "trust-scope":
            trust["Statement"][0].pop("Condition")
        else:
            trust["Statement"].append(
                {"Effect": "Allow", "Principal": "*", "Action": "sts:AssumeRole"}
            )
    installer.answer("iam_get-role", json.dumps(document))
    result = installer.attempt()
    assert result.returncode != 0, "untrusted IAM role was accepted"
    assert not installer.matching("apply -f"), "untrusted IAM preceded runtime apply"
    assert_no_foundation_writes(installer)


@pytest.mark.parametrize(
    "field", ["clusterName", "namespace", "serviceAccount", "roleArn", "associationId"]
)
def test_runtime_only_refuses_association_drift(
    installer: Installer, field: str
) -> None:
    document = association()
    document[field] = "other-identity"
    installer.answer(
        "eks_describe-pod-identity-association", json.dumps({"association": document})
    )
    result = installer.attempt()
    assert result.returncode != 0, "mismatched Pod Identity was accepted"
    assert not installer.matching("apply -f"), (
        "foreign association preceded runtime apply"
    )
    assert_no_foundation_writes(installer)


@pytest.mark.parametrize("associations", [[], [{}, {}], None])
def test_runtime_only_refuses_ambiguous_associations(
    installer: Installer, associations: object
) -> None:
    installer.answer(
        "eks_list-pod-identity-associations", json.dumps({"associations": associations})
    )
    result = installer.attempt()
    assert result.returncode != 0, "ambiguous association list was accepted"
    assert not installer.matching("apply -f"), (
        "ambiguous identity reached runtime apply"
    )
    assert_no_foundation_writes(installer)


def test_runtime_only_refuses_writer_permission_drift(installer: Installer) -> None:
    installer.answer(
        "iam_get-role-policy",
        json.dumps(
            {"Version": "2012-10-17", "Statement": [{"Effect": "Allow", "Action": "*"}]}
        ),
    )
    result = installer.attempt()
    assert result.returncode != 0, "unscoped AMP permissions were accepted"
    assert not installer.matching("apply -f"), "permission drift reached runtime apply"
    assert_no_foundation_writes(installer)


def test_runtime_only_refuses_sns_policy_drift(installer: Installer) -> None:
    installer.answer(
        "sns_get-topic-attributes.text",
        json.dumps({"Version": "2012-10-17", "Statement": []}),
    )
    result = installer.attempt()
    assert result.returncode != 0, "SNS drift was repaired outside bootstrap ownership"
    assert not installer.matching("apply -f"), "SNS drift reached runtime apply"
    assert_no_foundation_writes(installer)


@pytest.mark.parametrize("scope", ["role", "trust", "policy", "association", "list"])
def test_malformed_identity_responses_are_not_echoed_or_applied(
    installer: Installer, scope: str
) -> None:
    canary = "synthetic-opaque-identity-error-canary"
    if scope == "trust":
        role = role_document()
        role["Role"]["AssumeRolePolicyDocument"] = canary
        operation, text = "iam_get-role", json.dumps(role)
    else:
        operation = {
            "role": "iam_get-role",
            "policy": "iam_get-role-policy",
            "association": "eks_describe-pod-identity-association",
            "list": "eks_list-pod-identity-associations",
        }[scope]
        text = canary
    installer.answer(operation, text)
    result = installer.attempt()
    assert result.returncode != 0, "malformed identity response was accepted"
    assert canary not in result.stdout + result.stderr
    assert "ERROR:" in result.stderr, "malformed identity has no safe failure reason"
    assert not installer.matching("apply -f"), (
        "malformed identity reached runtime apply"
    )
    assert_no_foundation_writes(installer)


def test_runtime_only_manifest_matches_the_verified_namespace(
    installer: Installer,
) -> None:
    namespace = "gpu-fault-other"
    installer.environment["NAMESPACE"] = namespace
    installer.answer(
        "eks_list-pod-identity-associations",
        json.dumps({"associations": [association(namespace)]}),
    )
    installer.answer(
        "eks_describe-pod-identity-association",
        json.dumps({"association": association(namespace)}),
    )
    captured = installer.tmp_path / "adot.yaml"
    original = installer.bin / "kubectl"
    original.rename(installer.bin / "kubectl-fixture")
    original.write_text(
        '#!/usr/bin/env bash\nset -euo pipefail\nprevious=""\n'
        'for argument in "$@"; do\n'
        '  if [[ "${previous}" == "-f" ]]; then\n'
        f"    cp -- \"${{argument}}\" '{captured}'\n"
        '  fi\n  previous="${argument}"\ndone\n'
        'exec "$(dirname "$0")/kubectl-fixture" "$@"\n'
    )
    original.chmod(0o755)
    installer.run()
    documents = list(yaml.safe_load_all(captured.read_text()))
    assert {item["metadata"]["namespace"] for item in documents} == {namespace}
    configuration = next(item for item in documents if item["kind"] == "ConfigMap")
    collector = yaml.safe_load(configuration["data"]["collector.yaml"])
    jobs = collector["receivers"]["prometheus"]["config"]["scrape_configs"]
    assert jobs[0]["kubernetes_sd_configs"][0]["namespaces"]["names"] == [namespace]
    assert_no_foundation_writes(installer)

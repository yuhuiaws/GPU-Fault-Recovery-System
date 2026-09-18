"""The Python foundation policy must satisfy the runtime-only shell verifier."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from gpu_fault.admin.bootstrap_common import ClusterIdentity
from gpu_fault.admin.bootstrap_services import ensure_monitoring_resources
from tests.admin.test_admin_bootstrap_amp_installer import (
    ACCOUNT,
    CLUSTER,
    CLUSTER_ARN,
    NAMESPACE,
    REGION,
    WORKSPACE_ID,
    Installer,
)
from tests.admin.test_amp_runtime_only import (
    assert_no_foundation_writes,
    association,
    role_document,
)
from tests.admin.test_monitoring_policy import MonitoringAWS


@pytest.mark.parametrize("namespace", [NAMESPACE, "gpu-fault-other"])
@pytest.mark.parametrize("drift", [None, "missing", "wildcard"])
def test_initial_foundation_policy_handoff_to_runtime_only(
    tmp_path: Path, namespace: str, drift: str | None
) -> None:
    cpu = ClusterIdentity(
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
    foundation = MonitoringAWS(
        cpu, initial=True, site_id="site-a", workspace_id=WORKSPACE_ID
    )
    ensure_monitoring_resources(foundation, cpu=cpu, site_id="site-a", alert_email=None)
    assert foundation.mutations() == [
        ("amp", "create-workspace"),
        ("sns", "create-topic"),
        ("sns", "set-topic-attributes"),
    ], "initial foundation returned before converging the SNS publication grant"

    policy = foundation.document()
    if drift == "missing":
        policy["Statement"] = [
            statement
            for statement in policy["Statement"]
            if statement.get("Sid") != "AllowAmpAlertmanagerPublish"
        ]
    elif drift == "wildcard":
        for statement in policy["Statement"]:
            if statement.get("Sid") == "AllowAmpAlertmanagerPublish":
                statement["Condition"]["ArnEquals"]["AWS:SourceArn"] = "*"

    runtime = Installer(tmp_path)
    runtime.arguments = ["--runtime-only"]
    runtime.environment["NAMESPACE"] = namespace
    runtime.answer("iam_get-role", json.dumps(role_document()))
    runtime.answer(
        "eks_list-pod-identity-associations",
        json.dumps({"associations": [association(namespace)]}),
    )
    runtime.answer(
        "eks_describe-pod-identity-association",
        json.dumps({"association": association(namespace)}),
    )
    runtime.answer("sns_get-topic-attributes.text", json.dumps(policy))
    result = runtime.attempt()
    assert_no_foundation_writes(runtime)
    if drift is None:
        assert result.returncode == 0, result.stderr
        assert runtime.matching("apply -f"), (
            "bootstrap-produced SNS policy did not permit versioned monitoring"
        )
    else:
        assert result.returncode != 0, (
            "runtime-only accepted a changed publication grant"
        )
        assert "SNS policy differs" in result.stderr, (
            "runtime-only failure did not identify the publication prerequisite"
        )
        assert not runtime.matching("apply -f"), (
            "runtime-only applied resources before validating the publication grant"
        )

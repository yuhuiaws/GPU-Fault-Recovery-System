from __future__ import annotations

from pathlib import Path

import pytest

from gpu_fault.admin_bootstrap_common import BootstrapError, ClusterIdentity
from gpu_fault.admin_notifications import (
    ensure_email_notifications,
    resolve_admin_email,
)


class Runner:
    dry_run = False

    def __init__(self, responses):
        self.responses = list(responses)
        self.commands = []

    def aws_json(self, _region, *arguments, **_kwargs):
        self.commands.append(("aws", arguments))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    def run(self, arguments, **kwargs):
        self.commands.append(("run", tuple(arguments), kwargs))
        if "create" in arguments and "secret" in arguments:
            return "apiVersion: v1\nkind: Secret\nmetadata:\n  name: gpu-fault-email\n"
        return ""


def _cpu() -> ClusterIdentity:
    return ClusterIdentity(
        input_arn="arn:aws:eks:us-west-2:123456789012:cluster/cpu",
        role="cpu",
        region="us-west-2",
        account_id="123456789012",
        hyperpod_arn="arn:aws:sagemaker:us-west-2:123456789012:cluster/cpu",
        hyperpod_name="cpu",
        eks_arn="arn:aws:eks:us-west-2:123456789012:cluster/cpu",
        eks_name="cpu",
        vpc_id="vpc-cpu",
        subnet_ids=("subnet-a", "subnet-b"),
        node_recovery="None",
        context="cpu",
    )


def test_configured_admin_email_does_not_query_aws() -> None:
    runner = Runner([])

    email, source = resolve_admin_email(
        runner, account_id="123456789012", configured="ops@example.com"
    )

    assert email == "ops@example.com"
    assert source == "configured"
    assert runner.commands == []


def test_admin_email_falls_back_to_organization_account() -> None:
    runner = Runner([{"Account": {"Email": "root@example.com"}}])

    email, source = resolve_admin_email(
        runner, account_id="123456789012", configured=None
    )

    assert email == "root@example.com"
    assert source == "organizations"


def test_admin_email_reports_when_aws_cannot_discover_it() -> None:
    runner = Runner(
        [BootstrapError("organizations denied"), BootstrapError("account denied")]
    )

    with pytest.raises(BootstrapError, match="--admin-email"):
        resolve_admin_email(runner, account_id="123456789012", configured=None)


def test_email_notifications_require_verified_ses_and_apply_secret(
    tmp_path: Path,
) -> None:
    runner = Runner(
        [
            {"VerifiedForSendingStatus": True, "VerificationStatus": "SUCCESS"},
            {"SendingEnabled": True, "ProductionAccessEnabled": False},
        ]
    )

    result = ensure_email_notifications(
        runner,
        cpu=_cpu(),
        cpu_kubeconfig=tmp_path / "cpu.kubeconfig",
        namespace="gpu-fault-system",
        site_id="site-a",
        admin_email="ops@example.com",
    )

    assert result["sender_email"] == "ops@example.com"
    assert result["verified"] is True
    assert result["identity_ownership"] == "EXTERNAL"
    assert any(
        command[0] == "run" and "gpu-fault-email" in command[1]
        for command in runner.commands
    ), "email Secret was not applied"


def test_unverified_ses_identity_blocks_deploy(tmp_path: Path) -> None:
    runner = Runner(
        [{"VerifiedForSendingStatus": False, "VerificationStatus": "PENDING"}]
    )

    with pytest.raises(BootstrapError, match="complete verification"):
        ensure_email_notifications(
            runner,
            cpu=_cpu(),
            cpu_kubeconfig=tmp_path / "cpu.kubeconfig",
            namespace="gpu-fault-system",
            site_id="site-a",
            admin_email="ops@example.com",
        )

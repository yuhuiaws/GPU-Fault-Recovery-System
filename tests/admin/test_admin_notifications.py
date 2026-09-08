from __future__ import annotations

import base64
import json
from pathlib import Path

import pytest

from gpu_fault.admin.bootstrap_common import BootstrapError, ClusterIdentity
from gpu_fault.admin.notifications import (
    NotificationRouting,
    ensure_email_notifications,
    ensure_ses_identity,
    resolve_admin_email,
    ses_identity_verified,
)

ROUTING = NotificationRouting(
    sender="sender@example.com",
    recipients=("ops@example.com", "oncall@example.com"),
    subject_prefix="[PROD]",
    channel="ses",
)
SNS_ROUTING = NotificationRouting(
    sender="ops@example.com",
    recipients=("ops@example.com",),
    subject_prefix="[PROD]",
    channel="sns",
)


class Runner:
    dry_run = False

    def __init__(self, responses, *, secret_data=None):
        self.responses = list(responses)
        self.commands = []
        self.secret_data = secret_data

    def aws_json(self, _region, *arguments, **_kwargs):
        self.commands.append(("aws", arguments))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    def run(self, arguments, **kwargs):
        self.commands.append(("run", tuple(arguments), kwargs))
        if "get" in arguments and "secret" in arguments:
            if self.secret_data is not None:
                return json.dumps({"data": self.secret_data})
            raise BootstrapError("NotFound")
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
        routing=ROUTING,
    )

    assert result["sender_email"] == "sender@example.com"
    assert result["email_recipients"] == ["ops@example.com", "oncall@example.com"]
    assert result["email_subject_prefix"] == "[PROD]"
    assert result["verified"] is True
    assert result["identity_ownership"] == "EXTERNAL"
    assert any(
        command[0] == "run"
        and "create" in command[1]
        and "gpu-fault-email" in command[1]
        for command in runner.commands
    ), "email Secret was not applied"
    secret_command = next(
        command[1]
        for command in runner.commands
        if command[0] == "run"
        and "create" in command[1]
        and "gpu-fault-email" in command[1]
    )
    assert "--from-literal=email-sender=sender@example.com" in secret_command
    assert (
        "--from-literal=email-recipients=ops@example.com,oncall@example.com"
        in secret_command
    )
    assert "--from-literal=email-subject-prefix=[PROD]" in secret_command
    assert "--from-literal=site-id=site-a" in secret_command
    assert "--from-literal=aws-account-id=123456789012" in secret_command


def test_sns_channel_touches_no_ses_and_writes_an_addressless_secret(
    tmp_path: Path,
) -> None:
    """One email channel per site: the default channel never creates or reads an
    SES identity, and the Secret carries only what the runtime still needs from
    it -- the subject prefix and the site context."""

    runner = Runner([])

    result = ensure_email_notifications(
        runner,
        cpu=_cpu(),
        cpu_kubeconfig=tmp_path / "cpu.kubeconfig",
        namespace="gpu-fault-system",
        site_id="site-a",
        admin_email="ops@example.com",
        routing=SNS_ROUTING,
    )

    assert [command for command in runner.commands if command[0] == "aws"] == [], (
        "the SNS channel must not call sesv2 at all"
    )
    assert result["channel"] == "sns"
    assert result["verification_status"] == "NOT_REQUIRED"
    assert result["identity_ownership"] == "NONE"
    assert "sender_email" not in result and "identity_arn" not in result
    secret_command = next(
        command[1]
        for command in runner.commands
        if command[0] == "run"
        and "create" in command[1]
        and "gpu-fault-email" in command[1]
    )
    literals = [item for item in secret_command if item.startswith("--from-literal=")]
    assert literals == [
        "--from-literal=email-subject-prefix=[PROD]",
        "--from-literal=site-id=site-a",
        "--from-literal=aws-account-id=123456789012",
    ], "no sender and no recipient list exist on the SNS channel"


def test_sns_channel_probe_reuses_a_matching_addressless_secret(tmp_path: Path) -> None:
    values = {
        "email-subject-prefix": "[PROD]",
        "site-id": "site-a",
        "aws-account-id": "123456789012",
    }
    runner = Runner(
        [],
        secret_data={
            key: base64.b64encode(value.encode()).decode()
            for key, value in values.items()
        },
    )

    ensure_email_notifications(
        runner,
        cpu=_cpu(),
        cpu_kubeconfig=tmp_path / "cpu.kubeconfig",
        namespace="gpu-fault-system",
        site_id="site-a",
        admin_email="ops@example.com",
        routing=SNS_ROUTING,
    )

    assert [
        command[2]
        for command in runner.commands
        if command[0] == "run" and command[2].get("mutate")
    ] == []


def test_unverified_ses_identity_is_recorded_not_fatal(tmp_path: Path) -> None:
    """The verification gate moved to the deploy's first minute.

    ``gpu-fault-admin deploy`` checks the sender identity before any gate or
    build (``notification_precheck``); by the time this bootstrap task runs the
    identity is verified, and if an internal hop bypassed the check the task
    records the pending status instead of failing a bootstrap minutes in.
    """

    runner = Runner(
        [
            {"VerifiedForSendingStatus": False, "VerificationStatus": "PENDING"},
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
        routing=ROUTING,
    )

    assert result["verified"] is False
    assert result["verification_status"] == "PENDING"
    assert any(
        command[0] == "run"
        and "create" in command[1]
        and "gpu-fault-email" in command[1]
        for command in runner.commands
    ), "the email Secret is applied whatever the verification status"


def test_ensure_ses_identity_sends_the_verification_once(tmp_path: Path) -> None:
    """An absent identity is created (SES mails the link); a present one is read."""

    runner = Runner(
        [
            BootstrapError("NotFoundException"),
            {"VerifiedForSendingStatus": False, "VerificationStatus": "PENDING"},
        ]
    )

    identity, created = ensure_ses_identity(
        runner, region="us-west-2", sender="sender@example.com", site_id="site-a"
    )

    assert created is True
    assert ses_identity_verified(identity) is False
    creates = [
        command
        for command in runner.commands
        if command[0] == "run" and "create-email-identity" in command[1]
    ]
    assert len(creates) == 1
    assert "Key=gpu-fault:site-id,Value=site-a" in creates[0][1]

    existing = Runner(
        [{"VerifiedForSendingStatus": True, "VerificationStatus": "SUCCESS"}]
    )
    identity, created = ensure_ses_identity(
        existing, region="us-west-2", sender="sender@example.com", site_id="site-a"
    )
    assert created is False
    assert ses_identity_verified(identity) is True
    assert not any(command[0] == "run" for command in existing.commands), (
        "a present identity is only read; no second verification mail"
    )


def test_verified_email_probe_reuses_matching_secret(tmp_path: Path) -> None:
    values = {
        "email-sender": "sender@example.com",
        "email-recipients": "ops@example.com,oncall@example.com",
        "email-subject-prefix": "[PROD]",
        "site-id": "site-a",
        "aws-account-id": "123456789012",
    }
    runner = Runner(
        [
            {"VerifiedForSendingStatus": True, "VerificationStatus": "SUCCESS"},
            {"SendingEnabled": True, "ProductionAccessEnabled": True},
        ],
        secret_data={
            key: base64.b64encode(value.encode()).decode()
            for key, value in values.items()
        },
    )

    ensure_email_notifications(
        runner,
        cpu=_cpu(),
        cpu_kubeconfig=tmp_path / "cpu.kubeconfig",
        namespace="gpu-fault-system",
        site_id="site-a",
        admin_email="ops@example.com",
        routing=ROUTING,
    )

    mutating = [
        command[2]
        for command in runner.commands
        if command[0] == "run" and command[2].get("mutate")
    ]
    assert mutating == []

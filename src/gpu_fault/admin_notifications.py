from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from gpu_fault.admin_bootstrap_common import (
    BootstrapError,
    ClusterIdentity,
    CommandRunner,
    SITE_TAG_KEY,
)

EMAIL_PATTERN = re.compile(r"^[^\s@,]+@[^\s@,]+\.[^\s@,]+$")
EMAIL_SECRET_NAME = "gpu-fault-email"


def validate_admin_email(value: str) -> str:
    normalized = value.strip()
    if not EMAIL_PATTERN.fullmatch(normalized):
        raise BootstrapError("administrator email address is invalid")
    return normalized


def _optional_aws_json(
    runner: CommandRunner,
    region: str,
    *arguments: str,
) -> dict[str, Any] | None:
    try:
        return runner.aws_json(region, *arguments, sensitive=True)
    except BootstrapError:
        return None


def resolve_admin_email(
    runner: CommandRunner,
    *,
    account_id: str,
    configured: str | None,
) -> tuple[str, str]:
    if configured:
        return validate_admin_email(configured), "configured"

    organization = _optional_aws_json(
        runner,
        "us-east-1",
        "organizations",
        "describe-account",
        "--account-id",
        account_id,
    )
    organization_email = (
        (organization or {}).get("Account", {}).get("Email")
        if organization is not None
        else None
    )
    if organization_email:
        return validate_admin_email(str(organization_email)), "organizations"

    account = _optional_aws_json(
        runner,
        "us-east-1",
        "account",
        "get-primary-email",
        "--account-id",
        account_id,
    )
    primary_email = (account or {}).get("PrimaryEmail")
    if primary_email:
        return validate_admin_email(str(primary_email)), "account"

    raise BootstrapError(
        "cannot discover the AWS account administrator email; pass "
        "--admin-email, or run from an Organizations management/delegated "
        "administrator identity with organizations:DescribeAccount or "
        "account:GetPrimaryEmail"
    )


def _ses_identity(
    runner: CommandRunner,
    *,
    region: str,
    email: str,
) -> dict[str, Any] | None:
    return _optional_aws_json(
        runner,
        region,
        "sesv2",
        "get-email-identity",
        "--email-identity",
        email,
    )


def _apply_email_secret(
    runner: CommandRunner,
    *,
    cpu_kubeconfig: Path,
    namespace: str,
    sender: str,
    recipient: str,
) -> None:
    rendered = runner.run(
        [
            "kubectl",
            "--kubeconfig",
            str(cpu_kubeconfig),
            "-n",
            namespace,
            "create",
            "secret",
            "generic",
            EMAIL_SECRET_NAME,
            f"--from-literal=email-sender={sender}",
            f"--from-literal=email-recipients={recipient}",
            "--dry-run=client",
            "-o",
            "yaml",
        ],
        sensitive=True,
    )
    runner.run(
        [
            "kubectl",
            "--kubeconfig",
            str(cpu_kubeconfig),
            "apply",
            "-f",
            "-",
        ],
        input_text=rendered,
        sensitive=True,
        mutate=True,
        capture=False,
    )


def ensure_email_notifications(
    runner: CommandRunner,
    *,
    cpu: ClusterIdentity,
    cpu_kubeconfig: Path,
    namespace: str,
    site_id: str,
    admin_email: str,
) -> dict[str, Any]:
    identity = _ses_identity(
        runner,
        region=cpu.region,
        email=admin_email,
    )
    created = identity is None
    if created:
        runner.run(
            [
                "aws",
                "sesv2",
                "create-email-identity",
                "--region",
                cpu.region,
                "--email-identity",
                admin_email,
                "--tags",
                f"Key={SITE_TAG_KEY},Value={site_id}",
            ],
            mutate=True,
            sensitive=True,
            capture=False,
        )
        identity = _ses_identity(
            runner,
            region=cpu.region,
            email=admin_email,
        )
    if identity is None:
        raise BootstrapError("SES email identity could not be inspected")
    verified = bool(identity.get("VerifiedForSendingStatus")) or (
        str(identity.get("VerificationStatus") or "").upper() == "SUCCESS"
    )
    if not verified:
        raise BootstrapError(
            "SES sent a verification request to the administrator email; "
            "complete verification and rerun the same deploy command"
        )
    account = runner.aws_json(cpu.region, "sesv2", "get-account")
    if not bool(account.get("SendingEnabled")):
        raise BootstrapError("SES sending is disabled for the AWS account")

    _apply_email_secret(
        runner,
        cpu_kubeconfig=cpu_kubeconfig,
        namespace=namespace,
        sender=admin_email,
        recipient=admin_email,
    )
    return {
        "admin_email": admin_email,
        "sender_email": admin_email,
        "identity": admin_email,
        "identity_arn": (
            f"arn:aws:ses:{cpu.region}:{cpu.account_id}:identity/{admin_email}"
        ),
        "identity_ownership": "CREATED" if created else "EXTERNAL",
        "verified": True,
        "production_access_enabled": bool(account.get("ProductionAccessEnabled")),
        "secret_name": EMAIL_SECRET_NAME,
    }

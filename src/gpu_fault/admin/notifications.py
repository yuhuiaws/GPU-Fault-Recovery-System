from __future__ import annotations

import base64
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from gpu_fault.admin.bootstrap_common import (
    SITE_TAG_KEY,
    BootstrapError,
    ClusterIdentity,
    CommandRunner,
)

EMAIL_PATTERN = re.compile(r"^[^\s@,]+@[^\s@,]+\.[^\s@,]+$")
EMAIL_SECRET_NAME = "gpu-fault-email"


@dataclass(frozen=True)
class NotificationRouting:
    sender: str
    recipients: tuple[str, ...]
    subject_prefix: str


def validate_admin_email(value: str) -> str:
    normalized = value.strip()
    if not EMAIL_PATTERN.fullmatch(normalized):
        raise BootstrapError("administrator email address is invalid")
    return normalized


def resolve_notification_routing(*, admin_email: str) -> NotificationRouting:
    """SES mail goes from the administrator address to the administrator address.

    There is no separate sender, recipient list or subject prefix on the
    command; the ``site.yaml`` fields the release engine reads are filled from
    this routing.
    """

    sender = validate_admin_email(admin_email)
    return NotificationRouting(sender=sender, recipients=(sender,), subject_prefix="")


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
    recipients: Sequence[str],
    subject_prefix: str,
    site_id: str,
    account_id: str,
) -> None:
    expected = {
        "email-sender": sender,
        "email-recipients": ",".join(recipients),
        "email-subject-prefix": subject_prefix,
        "site-id": site_id,
        "aws-account-id": account_id,
    }
    try:
        current = json.loads(
            runner.run(
                [
                    "kubectl",
                    "--kubeconfig",
                    str(cpu_kubeconfig),
                    "-n",
                    namespace,
                    "get",
                    "secret",
                    EMAIL_SECRET_NAME,
                    "-o",
                    "json",
                ],
                sensitive=True,
            )
        )
    except BootstrapError as exc:
        message = str(exc).lower()
        if "notfound" not in message and "not found" not in message:
            raise
    else:
        data = current.get("data") or {}
        decoded = {}
        try:
            decoded = {
                str(key): base64.b64decode(str(value), validate=True).decode()
                for key, value in data.items()
            }
        except (UnicodeDecodeError, ValueError):
            decoded = {}
        if decoded == expected:
            return
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
            f"--from-literal=email-sender={expected['email-sender']}",
            f"--from-literal=email-recipients={expected['email-recipients']}",
            f"--from-literal=email-subject-prefix={expected['email-subject-prefix']}",
            f"--from-literal=site-id={expected['site-id']}",
            f"--from-literal=aws-account-id={expected['aws-account-id']}",
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


def ses_identity_verified(identity: Mapping[str, Any]) -> bool:
    return bool(identity.get("VerifiedForSendingStatus")) or (
        str(identity.get("VerificationStatus") or "").upper() == "SUCCESS"
    )


def ensure_ses_identity(
    runner: CommandRunner,
    *,
    region: str,
    sender: str,
    site_id: str,
) -> tuple[dict[str, Any], bool]:
    """The SES identity for ``sender``, created (verification mail sent) if absent.

    Returns ``(identity, created)``. Idempotent: an identity that exists is only
    read, so a rerun neither re-sends the verification nor re-tags it.
    """

    identity = _ses_identity(runner, region=region, email=sender)
    created = identity is None
    if created:
        runner.run(
            [
                "aws",
                "sesv2",
                "create-email-identity",
                "--region",
                region,
                "--email-identity",
                sender,
                "--tags",
                f"Key={SITE_TAG_KEY},Value={site_id}",
            ],
            mutate=True,
            sensitive=True,
            capture=False,
        )
        identity = _ses_identity(runner, region=region, email=sender)
    if identity is None:
        raise BootstrapError("SES email identity could not be inspected")
    return identity, created


def ensure_email_notifications(
    runner: CommandRunner,
    *,
    cpu: ClusterIdentity,
    cpu_kubeconfig: Path,
    namespace: str,
    site_id: str,
    admin_email: str,
    routing: NotificationRouting,
) -> dict[str, Any]:
    """The bootstrap task: SES identity, sending account, email Secret.

    Verification is no longer gated here. ``gpu-fault-admin deploy`` checks the
    sender identity and the SNS subscription in its first minute
    (``notification_precheck``) and stops with both addresses named; by the time
    this task runs the identity is verified, and if it is not (the check was
    bypassed by an internal hop) the status is recorded for ``status`` and the
    verifier rather than failing a bootstrap that is minutes in.
    """

    identity, created = ensure_ses_identity(
        runner, region=cpu.region, sender=routing.sender, site_id=site_id
    )
    verified = ses_identity_verified(identity)
    account = runner.aws_json(cpu.region, "sesv2", "get-account")
    if not bool(account.get("SendingEnabled")):
        raise BootstrapError("SES sending is disabled for the AWS account")

    _apply_email_secret(
        runner,
        cpu_kubeconfig=cpu_kubeconfig,
        namespace=namespace,
        sender=routing.sender,
        recipients=routing.recipients,
        subject_prefix=routing.subject_prefix,
        site_id=site_id,
        account_id=cpu.account_id,
    )
    return {
        "admin_email": admin_email,
        "sender_email": routing.sender,
        "email_recipients": list(routing.recipients),
        "email_subject_prefix": routing.subject_prefix,
        "identity": routing.sender,
        "identity_arn": (
            f"arn:aws:ses:{cpu.region}:{cpu.account_id}:identity/{routing.sender}"
        ),
        "identity_ownership": "CREATED" if created else "EXTERNAL",
        "verified": verified,
        "verification_status": "VERIFIED" if verified else "PENDING",
        "production_access_enabled": bool(account.get("ProductionAccessEnabled")),
        "secret_name": EMAIL_SECRET_NAME,
    }

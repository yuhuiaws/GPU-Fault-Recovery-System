from __future__ import annotations

import hashlib
import json
from typing import Any, Callable

from gpu_fault_release.regional_release_config import (
    NOTIFICATION_CHANNEL_SES,
    NOTIFICATION_CHANNEL_SNS,
    ReleaseError,
)

EMAIL_SECRET_NAME = "gpu-fault-email"


def notification_digest(config: Any) -> str:
    payload: dict[str, Any] = {
        "schema_version": 4,
        "channel": config.channel,
        "allow_email": config.allow_email,
        "acknowledge_external_alert_channel": (
            config.acknowledge_external_alert_channel
        ),
        "admin_email": config.admin_email,
        "email_sender": config.email_sender,
        "email_recipients": list(config.email_recipients),
        "email_subject_prefix": config.email_subject_prefix,
    }
    # Added after schema_version 4 shipped, so it enters the payload only when
    # a site declares one: every existing site's digest -- compared against the
    # live Deployment annotation by the admin check, and deciding whether a
    # deploy is a NOOP -- stays what it was, and a record or double predating
    # the field hashes identically.
    configuration_set = getattr(config, "ses_configuration_set", None)
    if configuration_set:
        payload["ses_configuration_set"] = configuration_set
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _sender_literals(config: Any) -> list[str]:
    """The SES address literals; none on ``sns``, where the topic routes."""

    if config.channel != NOTIFICATION_CHANNEL_SES:
        return []
    if not config.email_sender or not config.email_recipients:
        raise ReleaseError("email notification addresses are missing")
    return [
        f"--from-literal=email-sender={config.email_sender}",
        "--from-literal=email-recipients=" + ",".join(config.email_recipients),
    ]


def ensure_notification_secret(release: Any) -> None:
    """Apply ``gpu-fault-email`` whenever the control plane may notify.

    Every channel reads ``site-id``, ``aws-account-id`` and the subject prefix
    from it; only ``ses`` also carries the sender and recipients.
    """

    config = release.config.notifications
    if not config.allow_email:
        return
    if not config.admin_email:
        raise ReleaseError("email notification addresses are missing")
    account_id = str(release.config.cpu_eks_arn).split(":")[4]
    rendered = release.runner.run(
        release._cpu(
            "-n",
            release.config.namespace,
            "create",
            "secret",
            "generic",
            EMAIL_SECRET_NAME,
            *_sender_literals(config),
            f"--from-literal=email-subject-prefix={config.email_subject_prefix}",
            f"--from-literal=site-id={release.config.site_name}",
            f"--from-literal=aws-account-id={account_id}",
            "--dry-run=client",
            "-o",
            "yaml",
        ),
        capture=True,
        sensitive=True,
    )
    release.runner.run(
        release._cpu("apply", "-f", "-"),
        input_text=rendered,
        sensitive=True,
    )


# Read by every channel from ``gpu-fault-email``; ``ses`` adds the addresses.
COMMON_SECRET_KEYS = ("email-subject-prefix", "site-id", "aws-account-id")
SES_SECRET_KEYS = ("email-sender", "email-recipients", *COMMON_SECRET_KEYS)


def _secret_text(
    secret: dict[str, Any], keys: tuple[str, ...], decode: Callable[[str], bytes]
) -> dict[str, str]:
    if missing := sorted(set(keys) - set(secret)):
        raise ReleaseError(f"{EMAIL_SECRET_NAME} is missing: " + ", ".join(missing))
    return {key: decode(secret[key]).decode() for key in keys}


def _check_site_context(release: Any, values: dict[str, str]) -> None:
    config = release.config.notifications
    expected_account_id = str(release.config.cpu_eks_arn).split(":")[4]
    if (
        values["email-subject-prefix"] != config.email_subject_prefix
        or values["site-id"] != release.config.site_name
        or values["aws-account-id"] != expected_account_id
    ):
        raise ReleaseError(
            f"{EMAIL_SECRET_NAME} differs from the declared site addresses"
        )


def _check_sns_channel(
    release: Any,
    read_secret: Callable[[str], dict[str, Any]],
    decode_secret: Callable[[str], bytes],
) -> tuple[str, dict[str, Any]]:
    topic = release.config.health.sns_topic_arn
    if not topic:
        raise ReleaseError(
            "SNS notifications require health.sns_topic_arn "
            "(site.yaml spec.health.snsTopicArn)"
        )
    values = _secret_text(
        read_secret(EMAIL_SECRET_NAME).get("data") or {},
        COMMON_SECRET_KEYS,
        decode_secret,
    )
    _check_site_context(release, values)
    return (
        "SNS administrator notification channel is configured",
        {
            "enabled": True,
            "channel": NOTIFICATION_CHANNEL_SNS,
            "sns_topic_arn": topic,
            "site_id": values["site-id"],
        },
    )


def _check_ses_channel(
    release: Any,
    aws_json: Callable[[list[str]], dict[str, Any]],
    read_secret: Callable[[str], dict[str, Any]],
    decode_secret: Callable[[str], bytes],
) -> tuple[str, dict[str, Any]]:
    config = release.config.notifications
    if not config.email_sender or not config.email_recipients:
        raise ReleaseError("email notification addresses are missing")
    identity = aws_json(
        ["sesv2", "get-email-identity", "--email-identity", config.email_sender]
    )
    verified = bool(identity.get("VerifiedForSendingStatus")) or (
        str(identity.get("VerificationStatus") or "").upper() == "SUCCESS"
    )
    if not verified:
        raise ReleaseError("SES sender identity is not verified")
    account = aws_json(["sesv2", "get-account"])
    if not bool(account.get("SendingEnabled")):
        raise ReleaseError("SES sending is disabled")
    values = _secret_text(
        read_secret(EMAIL_SECRET_NAME).get("data") or {},
        SES_SECRET_KEYS,
        decode_secret,
    )
    recipients = tuple(
        item.strip() for item in values["email-recipients"].split(",") if item.strip()
    )
    if (
        values["email-sender"] != config.email_sender
        or recipients != config.email_recipients
    ):
        raise ReleaseError(
            f"{EMAIL_SECRET_NAME} differs from the declared site addresses"
        )
    _check_site_context(release, values)
    return (
        "SES administrator notification channel is configured",
        {
            "enabled": True,
            "channel": NOTIFICATION_CHANNEL_SES,
            "sender_verified": True,
            "sending_enabled": True,
            "production_access_enabled": bool(account.get("ProductionAccessEnabled")),
            "recipient_count": len(recipients),
            "site_id": values["site-id"],
        },
    )


def check_notification_channel(
    release: Any,
    *,
    aws_json: Callable[[list[str]], dict[str, Any]],
    read_secret: Callable[[str], dict[str, Any]],
    decode_secret: Callable[[str], bytes],
) -> tuple[str, dict[str, Any]]:
    """The preflight/verify body for an enabled channel: summary and details.

    ``sns`` needs the site topic and the common Secret keys and never calls
    SES; ``ses`` still proves the verified sender, account sending and the
    address literals. The callers' cluster and AWS readers are passed in so
    the check module keeps owning (and tests keep patching) that access.
    """

    if release.config.notifications.channel == NOTIFICATION_CHANNEL_SNS:
        return _check_sns_channel(release, read_secret, decode_secret)
    return _check_ses_channel(release, aws_json, read_secret, decode_secret)

from __future__ import annotations

import hashlib
import json
from typing import Any

from regional_release_config import ReleaseError

EMAIL_SECRET_NAME = "gpu-fault-email"


def notification_digest(config: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            {
                "schema_version": 3,
                "allow_email": config.allow_email,
                "acknowledge_external_alert_channel": (
                    config.acknowledge_external_alert_channel
                ),
                "admin_email": config.admin_email,
                "email_sender": config.email_sender,
                "email_recipients": list(config.email_recipients),
                "email_subject_prefix": config.email_subject_prefix,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def ensure_notification_secret(release: Any) -> None:
    config = release.config.notifications
    if not config.allow_email:
        return
    if not config.admin_email or not config.email_sender or not config.email_recipients:
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
            f"--from-literal=email-sender={config.email_sender}",
            ("--from-literal=email-recipients=" + ",".join(config.email_recipients)),
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

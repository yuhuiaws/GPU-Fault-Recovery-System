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
                "schema_version": 2,
                "allow_email": config.allow_email,
                "acknowledge_external_alert_channel": (
                    config.acknowledge_external_alert_channel
                ),
                "admin_email": config.admin_email,
                "email_sender": config.email_sender,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def ensure_notification_secret(release: Any) -> None:
    config = release.config.notifications
    if not config.allow_email:
        return
    if not config.admin_email or not config.email_sender:
        raise ReleaseError("email notification addresses are missing")
    rendered = release.runner.run(
        release._cpu(
            "-n",
            release.config.namespace,
            "create",
            "secret",
            "generic",
            EMAIL_SECRET_NAME,
            f"--from-literal=email-sender={config.email_sender}",
            f"--from-literal=email-recipients={config.admin_email}",
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

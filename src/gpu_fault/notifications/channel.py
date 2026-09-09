"""Which delivery channel this process notifies administrators through.

``GPU_FAULT_NOTIFICATION_CHANNEL`` names it: ``sns`` (the site topic, the
default for every site the admin CLI writes), ``ses`` (the verified sender
identity, opt-in for sites that want rich mail) or ``disabled`` (persist to the
outbox only). The release engine renders the variable from ``site.yaml``
``spec.notifications.channel`` on every control-plane role, so a running
control plane always has it declared; the inference below exists for the
manifests and test fixtures that predate the variable, where the topic ARN or
the SES address pair is the only statement of intent.
"""

from __future__ import annotations

import os

from gpu_fault.notifications.ses import (
    DisabledNotificationNotifier,
    SesEmailNotifier,
    SesNotificationConfig,
)
from gpu_fault.notifications.sns import SnsNotificationConfig, SnsNotifier

NOTIFICATION_CHANNEL_SNS = "sns"
NOTIFICATION_CHANNEL_SES = "ses"
NOTIFICATION_CHANNEL_DISABLED = "disabled"
NOTIFICATION_CHANNELS = (
    NOTIFICATION_CHANNEL_SNS,
    NOTIFICATION_CHANNEL_SES,
    NOTIFICATION_CHANNEL_DISABLED,
)


def notification_channel_from_environment() -> str:
    declared = (os.getenv("GPU_FAULT_NOTIFICATION_CHANNEL") or "").strip().lower()
    if declared:
        if declared not in NOTIFICATION_CHANNELS:
            raise ValueError(
                "GPU_FAULT_NOTIFICATION_CHANNEL must be one of "
                + ", ".join(NOTIFICATION_CHANNELS)
                + f", not {declared!r}"
            )
        return declared
    if (os.getenv("GPU_FAULT_SNS_TOPIC_ARN") or "").strip():
        return NOTIFICATION_CHANNEL_SNS
    if os.getenv("GPU_FAULT_EMAIL_SENDER") or os.getenv("GPU_FAULT_EMAIL_RECIPIENTS"):
        return NOTIFICATION_CHANNEL_SES
    return NOTIFICATION_CHANNEL_DISABLED


def notification_notifier_from_environment() -> (
    SnsNotifier | SesEmailNotifier | DisabledNotificationNotifier
):
    channel = notification_channel_from_environment()
    if channel == NOTIFICATION_CHANNEL_DISABLED:
        return DisabledNotificationNotifier()
    if channel == NOTIFICATION_CHANNEL_SNS:
        return SnsNotifier(SnsNotificationConfig.from_environment())
    sender = os.getenv("GPU_FAULT_EMAIL_SENDER")
    recipients = os.getenv("GPU_FAULT_EMAIL_RECIPIENTS")
    if not sender or not recipients:
        raise ValueError(
            "GPU_FAULT_EMAIL_SENDER and "
            "GPU_FAULT_EMAIL_RECIPIENTS must be configured together"
        )
    return SesEmailNotifier(SesNotificationConfig.from_environment())

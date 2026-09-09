"""Administrator notifications over the site's SNS topic.

The topic already carries the AMP Alertmanager alerts, so one confirmed email
subscription is the whole notification setup for a site: no SES identity to
verify, no sandbox, no separate sender. The price is the medium -- SNS mails
are plain text from ``no-reply@sns.amazonaws.com`` with an ASCII subject under
100 characters -- which is why the full subject is repeated as the first line
of the message and why SES stays available as ``channel: ses``.

Same contract as :class:`gpu_fault.notifications.ses.SesEmailNotifier`: one
``send`` per deduplication key per process, ``SKIPPED`` while sending is
disabled, and any exception left to the outbox dispatcher's retry.
"""

from __future__ import annotations

import re

from gpu_fault.env import env_bool
from gpu_fault.notifications.common import (
    AdvisoryNotification,
    Any,
    NotificationResult,
    NotificationStatus,
    Protocol,
    RLock,
    StrictModel,
    model_validator,
    os,
)
from gpu_fault.notifications.delivery_context import (
    body_with_context,
    subject_with_context,
)

# The Publish API requires the subject to be ASCII text with no line breaks or
# control characters that "must be less than 100 characters long".
SNS_SUBJECT_MAX_LENGTH = 99
SNS_SUBJECT_FALLBACK = "GPU Fault Recovery notification"
SNS_TOPIC_ARN_PATTERN = re.compile(
    r"^arn:aws[a-z-]*:sns:(?P<region>[a-z0-9-]+):(?P<account>[0-9]{12})"
    r":(?P<name>[A-Za-z0-9_-]{1,256})$"
)
_NON_ASCII_RUN = re.compile(r"[^\x20-\x7e]+")


class SnsClient(Protocol):
    def publish(self, **kwargs: Any) -> dict[str, Any]: ...


def sns_subject(subject: str) -> str:
    """The deterministic ASCII form of ``subject`` the Publish API accepts.

    Line breaks and tabs become spaces, every run of characters outside
    printable ASCII (the Chinese template titles, control characters) becomes
    one ``?``, whitespace is collapsed, and the result is cut to the limit. The
    operator still reads the full subject: the notifier puts it on the first
    line of the message.
    """

    one_line = " ".join(subject.split())
    collapsed = " ".join(_NON_ASCII_RUN.sub("?", one_line).split())
    if not collapsed:
        return SNS_SUBJECT_FALLBACK
    return collapsed[:SNS_SUBJECT_MAX_LENGTH].rstrip() or SNS_SUBJECT_FALLBACK


class SnsNotificationConfig(StrictModel):
    topic_arn: str
    region_name: str | None = None
    site_id: str | None = None
    account_id: str | None = None
    subject_prefix: str = ""
    # Implied on for this channel: the subscription the administrator confirmed
    # is the whole consent. GPU_FAULT_ALLOW_EMAIL=false stays a kill switch.
    execution_enabled: bool = True

    @model_validator(mode="after")
    def validate_topic(self) -> SnsNotificationConfig:
        match = SNS_TOPIC_ARN_PATTERN.fullmatch(self.topic_arn.strip())
        if match is None:
            raise ValueError("invalid SNS topic ARN")
        self.topic_arn = self.topic_arn.strip()
        if self.region_name is None:
            self.region_name = match.group("region")
        if (
            len(self.subject_prefix) > 64
            or "\n" in self.subject_prefix
            or "\r" in self.subject_prefix
        ):
            raise ValueError(
                "email subject prefix must be a single line of at most 64 characters"
            )
        return self

    @classmethod
    def from_environment(cls) -> SnsNotificationConfig:
        topic_arn = (os.getenv("GPU_FAULT_SNS_TOPIC_ARN") or "").strip()
        if not topic_arn:
            raise ValueError(
                "GPU_FAULT_SNS_TOPIC_ARN is required when "
                "GPU_FAULT_NOTIFICATION_CHANNEL=sns"
            )
        return cls(
            topic_arn=topic_arn,
            region_name=os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION"),
            site_id=os.getenv("GPU_FAULT_SITE_ID"),
            account_id=os.getenv("GPU_FAULT_AWS_ACCOUNT_ID"),
            subject_prefix=os.getenv("GPU_FAULT_EMAIL_SUBJECT_PREFIX", "").strip(),
            execution_enabled=env_bool("GPU_FAULT_ALLOW_EMAIL", True),
        )


class SnsNotifier:
    """Publishes each notification once to the site topic."""

    def __init__(
        self,
        config: SnsNotificationConfig,
        client: SnsClient | None = None,
    ) -> None:
        self.config = config
        self.client = client or self._create_client(config)
        self._results: dict[str, NotificationResult] = {}
        self._lock = RLock()

    @staticmethod
    def _create_client(config: SnsNotificationConfig) -> Any:
        try:
            import boto3
        except ImportError as exc:
            raise RuntimeError("install gpu-fault-control-plane[hyperpod]") from exc
        return boto3.client("sns", region_name=config.region_name)

    def _subject(self, notification: AdvisoryNotification) -> str:
        return subject_with_context(
            notification,
            subject_prefix=self.config.subject_prefix,
            site_id=self.config.site_id,
            region_name=self.config.region_name,
            account_id=self.config.account_id,
        )

    def _message(self, notification: AdvisoryNotification, subject: str) -> str:
        body = body_with_context(
            notification,
            site_id=self.config.site_id,
            account_id=self.config.account_id,
            region_name=self.config.region_name,
        )
        return f"{subject}\n\n{body}"

    def send(self, notification: AdvisoryNotification) -> NotificationResult:
        with self._lock:
            existing = self._results.get(notification.deduplication_key)
            if existing is not None:
                return existing.model_copy(
                    update={"status": NotificationStatus.DUPLICATE}
                )
            if not self.config.execution_enabled:
                return NotificationResult(
                    notification_id=notification.notification_id,
                    status=NotificationStatus.SKIPPED,
                    reason="SNS notification delivery is disabled",
                )
            subject = self._subject(notification)
            response = self.client.publish(
                TopicArn=self.config.topic_arn,
                Subject=sns_subject(subject),
                Message=self._message(notification, subject),
            )
            result = NotificationResult(
                notification_id=notification.notification_id,
                status=NotificationStatus.SENT,
                provider_message_id=response.get("MessageId"),
            )
            self._results[notification.deduplication_key] = result
            return result

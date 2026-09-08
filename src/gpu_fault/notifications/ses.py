from __future__ import annotations

from gpu_fault.env import env_bool
from gpu_fault.notifications.common import (
    AdvisoryNotification,
    Any,
    Field,
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


class SesV2Client(Protocol):
    def send_email(self, **kwargs) -> dict[str, Any]: ...


class SesNotificationConfig(StrictModel):
    sender: str
    recipients: list[str] = Field(min_length=1)
    region_name: str | None = None
    site_id: str | None = None
    account_id: str | None = None
    subject_prefix: str = ""
    execution_enabled: bool = False
    configuration_set_name: str | None = None

    @model_validator(mode="after")
    def validate_addresses(self) -> SesNotificationConfig:
        addresses = [self.sender, *self.recipients]
        if any("@" not in item or "\n" in item or "\r" in item for item in addresses):
            raise ValueError("invalid email address")
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
    def from_environment(cls) -> SesNotificationConfig:
        sender = os.getenv("GPU_FAULT_EMAIL_SENDER")
        recipients = [
            item.strip()
            for item in os.getenv("GPU_FAULT_EMAIL_RECIPIENTS", "").split(",")
            if item.strip()
        ]
        if not sender or not recipients:
            raise ValueError(
                "GPU_FAULT_EMAIL_SENDER and GPU_FAULT_EMAIL_RECIPIENTS are required"
            )
        return cls(
            sender=sender,
            recipients=recipients,
            region_name=os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION"),
            site_id=os.getenv("GPU_FAULT_SITE_ID"),
            account_id=os.getenv("GPU_FAULT_AWS_ACCOUNT_ID"),
            subject_prefix=os.getenv("GPU_FAULT_EMAIL_SUBJECT_PREFIX", "").strip(),
            execution_enabled=env_bool("GPU_FAULT_ALLOW_EMAIL", False),
            configuration_set_name=os.getenv("GPU_FAULT_SES_CONFIGURATION_SET"),
        )


class SesEmailNotifier:
    """Read-only by default SES v2 notification adapter."""

    def __init__(
        self,
        config: SesNotificationConfig,
        client: SesV2Client | None = None,
    ) -> None:
        self.config = config
        self.client = client or self._create_client(config)
        self._results: dict[str, NotificationResult] = {}
        self._lock = RLock()

    @staticmethod
    def _create_client(config: SesNotificationConfig):
        try:
            import boto3
        except ImportError as exc:
            raise RuntimeError("install gpu-fault-control-plane[hyperpod]") from exc
        return boto3.client("sesv2", region_name=config.region_name)

    def _subject(self, notification: AdvisoryNotification) -> str:
        return subject_with_context(
            notification,
            subject_prefix=self.config.subject_prefix,
            site_id=self.config.site_id,
            region_name=self.config.region_name,
            account_id=self.config.account_id,
        )

    def _body(self, notification: AdvisoryNotification) -> str:
        return body_with_context(
            notification,
            site_id=self.config.site_id,
            account_id=self.config.account_id,
            region_name=self.config.region_name,
        )

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
                    reason="SES email delivery is disabled",
                )

            request: dict[str, Any] = {
                "FromEmailAddress": self.config.sender,
                "Destination": {"ToAddresses": self.config.recipients},
                "Content": {
                    "Simple": {
                        "Subject": {
                            "Data": self._subject(notification),
                            "Charset": "UTF-8",
                        },
                        "Body": {
                            "Text": {
                                "Data": self._body(notification),
                                "Charset": "UTF-8",
                            }
                        },
                    }
                },
            }
            if self.config.configuration_set_name:
                request["ConfigurationSetName"] = self.config.configuration_set_name
            response = self.client.send_email(**request)
            result = NotificationResult(
                notification_id=notification.notification_id,
                status=NotificationStatus.SENT,
                provider_message_id=response.get("MessageId"),
            )
            self._results[notification.deduplication_key] = result
            return result


class DisabledNotificationNotifier:
    """Explicit default when no delivery channel is configured."""

    def send(self, notification: AdvisoryNotification) -> NotificationResult:
        return NotificationResult(
            notification_id=notification.notification_id,
            status=NotificationStatus.SKIPPED,
            reason="no notification delivery channel is configured",
        )

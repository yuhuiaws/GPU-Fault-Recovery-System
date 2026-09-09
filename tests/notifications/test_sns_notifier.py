"""Administrator notifications over the site SNS topic (channel ``sns``).

One email channel per site: the topic the AMP alerts already use carries the
control plane's own mails, so the administrator confirms one subscription and
no SES identity is verified. The adapter keeps the SES contract the outbox
dispatcher relies on -- one publish per deduplication key, ``SKIPPED`` while
disabled, exceptions left to the retry -- and works around the Publish API's
subject rule (ASCII, one line, under 100 characters) without losing the
subject: it is the first line of the message.
"""

from __future__ import annotations

import pytest

from gpu_fault.models import NotificationStatus
from gpu_fault.notification_service import AdvisoryNotificationService
from gpu_fault.notifications import (
    HyperPodAdvisoryEmailBuilder,
    SnsNotificationConfig,
    SnsNotifier,
    notification_channel_from_environment,
    notification_notifier_from_environment,
    sns_subject,
)
from gpu_fault.notifications.ses import DisabledNotificationNotifier, SesEmailNotifier
from tests._builders import build_store
from tests.notifications._support import advisory

TOPIC_ARN = "arn:aws:sns:us-west-2:123456789012:gpu-fault-site-a-alerts"


class FakeSnsClient:
    def __init__(self) -> None:
        self.requests: list[dict] = []

    def publish(self, **kwargs):
        self.requests.append(kwargs)
        return {"MessageId": "sns-message-1"}


def _notification(incident_id: str = "incident-42"):
    return HyperPodAdvisoryEmailBuilder().build(
        advisory(),
        cluster_name="cluster-a",
        incident_id=incident_id,
        node_ids=["runtime-node-9"],
        issue_summary="GPU fault",
    )


def _notifier(client: FakeSnsClient, **overrides) -> SnsNotifier:
    values = {
        "topic_arn": TOPIC_ARN,
        "site_id": "site-a",
        "account_id": "123456789012",
        "subject_prefix": "[PROD]",
    }
    values.update(overrides)
    return SnsNotifier(SnsNotificationConfig(**values), client=client)


def _clear_channel_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "GPU_FAULT_NOTIFICATION_CHANNEL",
        "GPU_FAULT_SNS_TOPIC_ARN",
        "GPU_FAULT_EMAIL_SENDER",
        "GPU_FAULT_EMAIL_RECIPIENTS",
        "GPU_FAULT_ALLOW_EMAIL",
        "GPU_FAULT_EMAIL_SUBJECT_PREFIX",
        "GPU_FAULT_SITE_ID",
        "GPU_FAULT_AWS_ACCOUNT_ID",
    ):
        monkeypatch.delenv(name, raising=False)


def test_sns_publish_carries_the_topic_the_subject_and_the_context() -> None:
    client = FakeSnsClient()

    result = _notifier(client).send(_notification())

    assert result.status is NotificationStatus.SENT
    assert result.provider_message_id == "sns-message-1"
    (request,) = client.requests
    assert request["TopicArn"] == TOPIC_ARN
    # The region is taken from the ARN when the config does not name one.
    assert request["Subject"].startswith(
        "[PROD] [site:site-a] [region:us-west-2] [account:123456789012]"
    ), "the SNS subject carries the same site context as the SES subject"
    first_line, _blank, body = request["Message"].partition("\n\n")
    assert first_line.startswith("[PROD] [site:site-a] [region:us-west-2]"), (
        "the full subject is the first line of the message"
    )
    assert "- Site: site-a" in body
    assert "- AWS Account: 123456789012" in body
    assert "- Cluster: cluster-a" in body
    assert "runtime-node-9" in body


def test_sns_and_ses_render_the_same_subject_and_body() -> None:
    """One template, two carriers: the operator must not learn two formats."""

    from gpu_fault.notifications.ses import SesNotificationConfig

    notification = _notification()
    sns_client = FakeSnsClient()
    _notifier(sns_client, region_name="us-west-2").send(notification)

    class SesClient:
        requests: list[dict] = []

        def send_email(self, **kwargs):
            self.requests.append(kwargs)
            return {"MessageId": "ses-1"}

    SesEmailNotifier(
        SesNotificationConfig(
            sender="ops@example.com",
            recipients=["ops@example.com"],
            region_name="us-west-2",
            site_id="site-a",
            account_id="123456789012",
            subject_prefix="[PROD]",
            execution_enabled=True,
        ),
        client=SesClient(),
    ).send(notification)
    ses = SesClient.requests[0]["Content"]["Simple"]

    subject, _blank, body = sns_client.requests[0]["Message"].partition("\n\n")
    assert subject == ses["Subject"]["Data"]
    assert body == ses["Body"]["Text"]["Data"]


def test_sns_subject_is_ascii_one_line_and_under_100_characters() -> None:
    chinese = "[site:a] GPU 节点硬件 Inventory 不匹配通知"
    assert sns_subject(chinese) == "[site:a] GPU ? Inventory ?"
    assert sns_subject("line one\r\nline two\ttabbed") == "line one line two tabbed"
    assert sns_subject("   ") == "GPU Fault Recovery notification"
    assert sns_subject("节点故障") == "?"

    long = "[PROD] [site:site-a] " + "x" * 200
    truncated = sns_subject(long)
    assert len(truncated) == 99 and truncated == long[:99]
    assert sns_subject(long) == truncated, "sanitising is deterministic"
    assert all(0x20 <= ord(char) <= 0x7E for char in truncated), (
        "the Publish API rejects anything outside printable ASCII"
    )


def test_sns_delivery_is_idempotent_per_deduplication_key() -> None:
    client = FakeSnsClient()
    notifier = _notifier(client)
    notification = _notification()

    first = notifier.send(notification)
    second = notifier.send(notification)
    other = notifier.send(_notification("incident-43"))

    assert first.status is NotificationStatus.SENT
    assert second.status is NotificationStatus.DUPLICATE
    assert second.provider_message_id == "sns-message-1"
    assert other.status is NotificationStatus.SENT
    assert len(client.requests) == 2


def test_sns_delivery_honours_the_kill_switch_without_consuming_the_key() -> None:
    client = FakeSnsClient()
    config = SnsNotificationConfig(topic_arn=TOPIC_ARN, execution_enabled=False)
    notifier = SnsNotifier(config, client=client)
    notification = _notification()

    skipped = notifier.send(notification)
    assert skipped.status is NotificationStatus.SKIPPED
    assert skipped.reason == "SNS notification delivery is disabled"
    assert client.requests == []

    config.execution_enabled = True
    assert notifier.send(notification).status is NotificationStatus.SENT


def test_sns_publish_failures_propagate_to_the_dispatcher_retry() -> None:
    class FlakyClient(FakeSnsClient):
        def __init__(self) -> None:
            super().__init__()
            self.failures = 1

        def publish(self, **kwargs):
            if self.failures:
                self.failures -= 1
                raise RuntimeError("SNS unavailable")
            return super().publish(**kwargs)

    client = FlakyClient()
    notifier = SnsNotifier(SnsNotificationConfig(topic_arn=TOPIC_ARN), client=client)
    notification = _notification()
    with pytest.raises(RuntimeError, match="SNS unavailable"):
        notifier.send(notification)
    # A failed attempt records nothing, so the dispatcher's retry publishes
    # instead of being answered DUPLICATE.
    assert notifier.send(notification).status is NotificationStatus.SENT
    assert len(client.requests) == 1


@pytest.mark.parametrize(
    "topic_arn",
    [
        "gpu-fault-site-a-alerts",
        "arn:aws:sqs:us-west-2:123456789012:gpu-fault-site-a-alerts",
        "arn:aws:sns:us-west-2:12345:gpu-fault",
        "arn:aws:sns:us-west-2:123456789012:bad topic",
    ],
)
def test_sns_config_rejects_anything_but_a_topic_arn(topic_arn: str) -> None:
    with pytest.raises(ValueError, match="SNS topic ARN"):
        SnsNotificationConfig(topic_arn=topic_arn)


def test_sns_config_from_environment_is_enabled_unless_switched_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_channel_environment(monkeypatch)
    monkeypatch.setenv("GPU_FAULT_SNS_TOPIC_ARN", f" {TOPIC_ARN} ")
    monkeypatch.setenv("GPU_FAULT_SITE_ID", "site-a")
    monkeypatch.setenv("GPU_FAULT_AWS_ACCOUNT_ID", "123456789012")
    monkeypatch.setenv("GPU_FAULT_EMAIL_SUBJECT_PREFIX", " [PROD] ")
    monkeypatch.setenv("AWS_REGION", "us-east-2")

    config = SnsNotificationConfig.from_environment()

    assert config.topic_arn == TOPIC_ARN
    assert config.region_name == "us-east-2", "the declared Region wins over the ARN"
    assert config.site_id == "site-a" and config.account_id == "123456789012"
    assert config.subject_prefix == "[PROD]"
    assert config.execution_enabled is True, "the confirmed subscription is consent"

    monkeypatch.setenv("GPU_FAULT_ALLOW_EMAIL", "false")
    assert SnsNotificationConfig.from_environment().execution_enabled is False

    monkeypatch.delenv("GPU_FAULT_SNS_TOPIC_ARN")
    with pytest.raises(ValueError, match="GPU_FAULT_SNS_TOPIC_ARN is required"):
        SnsNotificationConfig.from_environment()


def test_channel_selection_follows_the_declared_variable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_channel_environment(monkeypatch)
    monkeypatch.setenv("GPU_FAULT_SNS_TOPIC_ARN", TOPIC_ARN)
    monkeypatch.setenv("GPU_FAULT_EMAIL_SENDER", "ops@example.com")
    monkeypatch.setenv("GPU_FAULT_EMAIL_RECIPIENTS", "ops@example.com")
    monkeypatch.setattr(
        SnsNotifier, "_create_client", staticmethod(lambda _config: FakeSnsClient())
    )
    monkeypatch.setattr(
        SesEmailNotifier, "_create_client", staticmethod(lambda _config: object())
    )

    monkeypatch.setenv("GPU_FAULT_NOTIFICATION_CHANNEL", "sns")
    assert isinstance(notification_notifier_from_environment(), SnsNotifier), (
        "the sns channel selects the topic publisher"
    )

    monkeypatch.setenv("GPU_FAULT_NOTIFICATION_CHANNEL", "SES")
    assert notification_channel_from_environment() == "ses"
    assert isinstance(notification_notifier_from_environment(), SesEmailNotifier), (
        "the ses channel selects the verified-sender adapter"
    )

    monkeypatch.setenv("GPU_FAULT_NOTIFICATION_CHANNEL", "disabled")
    assert isinstance(
        notification_notifier_from_environment(), DisabledNotificationNotifier
    ), "no channel means persist-only, never an implicit send"

    monkeypatch.setenv("GPU_FAULT_NOTIFICATION_CHANNEL", "smtp")
    with pytest.raises(ValueError, match="GPU_FAULT_NOTIFICATION_CHANNEL"):
        notification_notifier_from_environment()

    monkeypatch.setenv("GPU_FAULT_NOTIFICATION_CHANNEL", "sns")
    monkeypatch.delenv("GPU_FAULT_SNS_TOPIC_ARN")
    with pytest.raises(ValueError, match="GPU_FAULT_SNS_TOPIC_ARN is required"):
        notification_notifier_from_environment()


def test_an_undeclared_channel_is_inferred_from_what_the_manifest_carries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Manifests and fixtures older than the variable keep their meaning."""

    _clear_channel_environment(monkeypatch)
    monkeypatch.setattr(
        SnsNotifier, "_create_client", staticmethod(lambda _config: FakeSnsClient())
    )
    monkeypatch.setattr(
        SesEmailNotifier, "_create_client", staticmethod(lambda _config: object())
    )

    assert notification_channel_from_environment() == "disabled"
    assert isinstance(
        notification_notifier_from_environment(), DisabledNotificationNotifier
    ), "no channel means persist-only, never an implicit send"

    monkeypatch.setenv("GPU_FAULT_EMAIL_SENDER", "ops@example.com")
    with pytest.raises(ValueError, match="configured together"):
        notification_notifier_from_environment()
    monkeypatch.setenv("GPU_FAULT_EMAIL_RECIPIENTS", "ops@example.com")
    assert notification_channel_from_environment() == "ses"
    assert isinstance(notification_notifier_from_environment(), SesEmailNotifier), (
        "the ses channel selects the verified-sender adapter"
    )

    monkeypatch.setenv("GPU_FAULT_SNS_TOPIC_ARN", TOPIC_ARN)
    assert notification_channel_from_environment() == "sns", (
        "a topic outranks the SES pair when nothing declares the channel"
    )
    assert isinstance(notification_notifier_from_environment(), SnsNotifier), (
        "the sns channel selects the topic publisher"
    )


def test_sns_channel_counts_as_external_delivery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GPU_FAULT_NOTIFICATION_DISPATCHER_ENABLED", "true")
    delivered = AdvisoryNotificationService(
        build_store(), _notifier(FakeSnsClient()), async_delivery=True
    )
    disabled = AdvisoryNotificationService(
        build_store(),
        SnsNotifier(
            SnsNotificationConfig(topic_arn=TOPIC_ARN, execution_enabled=False),
            client=FakeSnsClient(),
        ),
        async_delivery=True,
    )

    assert delivered.delivers_externally() is True
    assert "NOT DELIVERED" not in delivered.describe_delivery_mode()
    assert disabled.delivers_externally() is False
    assert "GPU_FAULT_ALLOW_EMAIL" in disabled.describe_delivery_mode()

    no_channel = AdvisoryNotificationService(
        build_store(), DisabledNotificationNotifier(), async_delivery=True
    ).describe_delivery_mode()
    assert "GPU_FAULT_SNS_TOPIC_ARN" in no_channel
    assert "GPU_FAULT_EMAIL_SENDER" in no_channel

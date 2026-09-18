from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from gpu_fault.models import AdvisoryNotification, NotificationStatus
from gpu_fault.notifications import (
    SesEmailNotifier,
    SesNotificationConfig,
    SnsNotificationConfig,
    SnsNotifier,
)
from tests.notifications._support import FakeSesV2Client
from tests.notifications.test_sns_notifier import TOPIC_ARN, FakeSnsClient


def notification(index=0):
    return AdvisoryNotification(
        deduplication_key=f"local-{index}",
        cluster_name="cluster-local",
        incident_id=f"incident-{index}",
        subject="Local notification",
        body_text="Local notification body",
        support_case_draft="",
    )


def test_sns_client_has_bounded_socket_timeouts_and_retries(monkeypatch) -> None:
    calls = []

    def client(service, **kwargs):
        calls.append((service, kwargs))
        return FakeSnsClient()

    monkeypatch.setitem(sys.modules, "boto3", SimpleNamespace(client=client))
    notifier = SnsNotifier(SnsNotificationConfig(topic_arn=TOPIC_ARN))
    assert notifier.send(notification()).status is NotificationStatus.SENT
    assert len(calls) == 1
    service, kwargs = calls[0]
    assert service == "sns"
    assert kwargs["region_name"] == "us-west-2"
    config = kwargs["config"]
    assert config.connect_timeout == 5
    assert config.read_timeout == 20
    assert config.retries == {"max_attempts": 2, "mode": "standard"}


def test_sns_in_process_dedup_hint_is_bounded_without_changing_durable_truth() -> None:
    provider = FakeSnsClient()
    notifier = SnsNotifier(SnsNotificationConfig(topic_arn=TOPIC_ARN), client=provider)
    for index in range(1100):
        assert notifier.send(notification(index)).status is NotificationStatus.SENT
    assert notifier.send(notification(1099)).status is NotificationStatus.DUPLICATE
    assert len(provider.requests) == 1100
    assert notifier.send(notification(0)).status is NotificationStatus.SENT, (
        "old process hints must be evicted; the Store owns cross-process deduplication"
    )
    assert len(provider.requests) == 1101


@pytest.mark.parametrize("carrier", ["ses", "sns"])
def test_missing_sdk_is_an_actionable_constructor_error(monkeypatch, carrier) -> None:
    monkeypatch.setitem(sys.modules, "boto3", None)
    with pytest.raises(
        RuntimeError, match=r"install gpu-fault-control-plane\[hyperpod\]"
    ):
        if carrier == "ses":
            SesEmailNotifier(
                SesNotificationConfig(
                    sender="ops@example.com", recipients=["ops@example.com"]
                )
            )
        else:
            SnsNotifier(SnsNotificationConfig(topic_arn=TOPIC_ARN))


@pytest.mark.parametrize("carrier", ["ses", "sns"])
@pytest.mark.parametrize("prefix", ["x" * 65, "line\nbreak", "line\rbreak"])
def test_subject_prefix_cannot_inject_headers(carrier, prefix) -> None:
    with pytest.raises(ValueError, match="single line"):
        if carrier == "ses":
            SesNotificationConfig(
                sender="ops@example.com",
                recipients=["ops@example.com"],
                subject_prefix=prefix,
            )
        else:
            SnsNotificationConfig(topic_arn=TOPIC_ARN, subject_prefix=prefix)


@pytest.mark.parametrize(
    "address",
    ["invalid", "ops@example.com\nBcc:other@example.com", "ops@example.com\r"],
)
@pytest.mark.parametrize("field", ["sender", "recipients"])
def test_ses_address_validation_prevents_header_injection(address, field) -> None:
    arguments = {"sender": "ops@example.com", "recipients": ["ops@example.com"]}
    arguments[field] = [address] if field == "recipients" else address
    with pytest.raises(ValueError, match="invalid email"):
        SesNotificationConfig(**arguments)


@pytest.mark.parametrize(
    "sender,recipients", [("", ""), ("ops@example.com", " , "), ("", "ops@example.com")]
)
def test_ses_environment_requires_both_addresses(monkeypatch, sender, recipients):
    monkeypatch.setenv("GPU_FAULT_EMAIL_SENDER", sender)
    monkeypatch.setenv("GPU_FAULT_EMAIL_RECIPIENTS", recipients)
    with pytest.raises(ValueError, match="are required"):
        SesNotificationConfig.from_environment()


def test_ses_configuration_set_and_empty_provider_id_preserve_result_contract() -> None:
    class Provider(FakeSesV2Client):
        def send_email(self, **kwargs):
            super().send_email(**kwargs)
            return {}

    provider = Provider()
    notifier = SesEmailNotifier(
        SesNotificationConfig(
            sender="ops@example.com",
            recipients=["ops@example.com"],
            execution_enabled=True,
            configuration_set_name="audit",
        ),
        client=provider,
    )
    result = notifier.send(notification())
    assert result.status is NotificationStatus.SENT
    assert result.provider_message_id is None
    assert provider.requests[0]["ConfigurationSetName"] == "audit"
    assert notifier.send(notification()).status is NotificationStatus.DUPLICATE
    assert len(provider.requests) == 1

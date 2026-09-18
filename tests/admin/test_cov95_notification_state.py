from __future__ import annotations

import base64
from dataclasses import replace
from types import SimpleNamespace

import pytest

from gpu_fault.admin import monitoring_subscriptions as subscriptions
from gpu_fault.admin import notifications
from gpu_fault.admin.bootstrap_common import BootstrapError, BootstrapState
from tests.admin._cov95_join_support import target
from tests.admin.test_admin_monitoring_subscriptions import EMAIL, TOPIC_ARN, Runner
from tests.admin.test_admin_notifications import Runner as NotificationRunner


class SubscriptionRunner(Runner):
    def __init__(self):
        super().__init__()
        self.response = {
            "Attributes": {"TopicArn": TOPIC_ARN, "PendingConfirmation": "true"}
        }
        self.reads = 0
        self.invalid_subscribe = False

    def aws_json(self, _region, *arguments, **_options):
        assert arguments[:2] == ("sns", "get-subscription-attributes")
        self.reads += 1
        if isinstance(self.response, Exception):
            raise self.response
        return self.response

    def aws_text(self, region, *arguments, **options):
        result = super().aws_text(region, *arguments, **options)
        return "PendingConfirmation" if self.invalid_subscribe else result


@pytest.fixture
def context(tmp_path, monkeypatch):
    state = BootstrapState(tmp_path / "bootstrap.json", site_id="example")
    clock = SimpleNamespace(value=1000.0)
    monkeypatch.setattr(
        subscriptions, "time", SimpleNamespace(time=lambda: clock.value)
    )
    runner = SubscriptionRunner()
    cpu = replace(target(), role="cpu", region="us-west-2")
    return runner, state, clock, cpu


def ensure(context):
    runner, state, _clock, cpu = context
    return subscriptions.ensure_email_subscription(
        runner,
        state=state,
        cpu=cpu,
        topic_arn=TOPIC_ARN,
        topic_generation="a" * 32,
        endpoint=EMAIL,
        subscriptions=[],
    )


def remember(context, record):
    context[1].record(subscriptions.SNS_EMAIL_SUBSCRIPTION_STATE_KEY, record)


def test_confirmed_subscription_read_failure_never_authorizes_duplicate_subscribe(
    context,
):
    runner, _state, _clock, _cpu = context
    first = ensure(context)
    remember(context, {**first, "status": "CONFIRMED"})
    runner.response = BootstrapError("example AccessDenied")
    with pytest.raises(BootstrapError, match="refusing a duplicate Subscribe"):
        ensure(context)
    assert runner.subscribe_calls == 1


def test_persisted_subscription_cannot_be_rebound_to_a_different_topic(context):
    runner, _state, _clock, _cpu = context
    ensure(context)
    runner.response = {"Attributes": {"TopicArn": TOPIC_ARN + "-foreign"}}
    with pytest.raises(BootstrapError, match="belongs to another topic"):
        ensure(context)
    assert runner.subscribe_calls == 1


def test_confirmation_from_subscription_read_is_checkpointed_without_resubscribe(
    context,
):
    runner, _state, _clock, _cpu = context
    first = ensure(context)
    runner.response = {
        "Attributes": {"TopicArn": TOPIC_ARN, "PendingConfirmation": "false"}
    }
    result = ensure(context)
    assert result["status"] == "CONFIRMED"
    assert result["subscription_arn"] == first["subscription_arn"]
    assert runner.subscribe_calls == 1


@pytest.mark.parametrize(
    "response",
    [
        {"Attributes": []},
        {"Attributes": {"TopicArn": TOPIC_ARN, "PendingConfirmation": "true"}},
        BootstrapError("example temporary lookup failure"),
        BootstrapError("NotFound"),
    ],
)
@pytest.mark.parametrize("expired", [False, True])
def test_pending_subscription_suppression_survives_read_gaps_until_expiry(
    context, response, expired
):
    runner, _state, clock, _cpu = context
    first = ensure(context)
    runner.response = response
    clock.value = first["expires_at_epoch"] + 1 if expired else 1010
    result = ensure(context)
    assert runner.subscribe_calls == (2 if expired else 1)
    if not expired:
        assert result == first
    else:
        assert result["requested_at_epoch"] == clock.value


@pytest.mark.parametrize(
    "status,elapsed,requests",
    [
        ("REQUESTING", 1, 1),
        ("REQUESTING", subscriptions.SNS_EMAIL_REQUESTING_SUPPRESSION_SECONDS + 1, 2),
        ("PENDING", 1, 1),
        ("PENDING", subscriptions.SNS_EMAIL_CONFIRMATION_TTL_SECONDS + 1, 2),
    ],
)
def test_missing_subscription_arn_obeys_intent_and_confirmation_windows(
    context, status, elapsed, requests
):
    runner, _state, clock, _cpu = context
    first = ensure(context)
    previous = {**first, "status": status, "subscription_arn": None}
    remember(context, previous)
    clock.value += elapsed
    result = ensure(context)
    assert runner.subscribe_calls == requests
    assert runner.reads == 0
    if requests == 1:
        assert result == previous


def test_invalid_subscribe_response_preserves_requesting_checkpoint(context):
    runner, state, _clock, _cpu = context
    runner.invalid_subscribe = True
    with pytest.raises(BootstrapError, match="returned no subscription ARN"):
        ensure(context)
    record = state.result(subscriptions.SNS_EMAIL_SUBSCRIPTION_STATE_KEY)
    assert record["status"] == "REQUESTING"
    assert record["subscription_arn"] is None
    assert runner.subscribe_calls == 1


@pytest.mark.parametrize(
    "email",
    ["", "missing-at.example.invalid", "two@example.invalid,other@example.invalid"],
)
def test_notification_address_validation_precedes_discovery(email):
    runner = NotificationRunner([])
    with pytest.raises(BootstrapError, match="email address is invalid"):
        notifications.resolve_notification_routing(admin_email=email)
    assert runner.commands == []


def test_notification_routing_rejects_unknown_channel():
    with pytest.raises(BootstrapError, match="channel must be one of"):
        notifications.resolve_notification_routing(
            admin_email="example@example.invalid", channel="unknown"
        )


def test_account_primary_email_is_used_only_after_empty_organization_response():
    runner = NotificationRunner([{}, {"PrimaryEmail": "example@example.invalid"}])
    assert notifications.resolve_admin_email(
        runner, account_id="123456789012", configured=None
    ) == ("example@example.invalid", "account")
    assert len(runner.commands) == 2


def notify(runner, root, *, channel="sns"):
    address = "example@example.invalid"
    return notifications.ensure_email_notifications(
        runner,
        cpu=replace(target(), role="cpu"),
        cpu_kubeconfig=root / "cpu.kubeconfig",
        namespace="gpu-fault-system",
        site_id="example",
        admin_email=address,
        routing=notifications.resolve_notification_routing(
            admin_email=address, channel=channel
        ),
    )


@pytest.mark.parametrize("value", ["!not-base64!", base64.b64encode(b"\xff").decode()])
def test_malformed_notification_secret_is_reconciled_through_sensitive_transports(
    tmp_path, value
):
    runner = NotificationRunner([], secret_data={"email-subject-prefix": value})
    result = notify(runner, tmp_path)
    assert result["channel"] == "sns"
    operations = [command for command in runner.commands if command[0] == "run"]
    assert len(operations) == 3
    assert all(command[2]["sensitive"] for command in operations), (
        "notification Secret reconciliation lost sensitive capture"
    )
    assert operations[-1][2]["mutate"] is True


def test_notification_secret_permission_failure_is_not_absence(tmp_path):
    class Unavailable(NotificationRunner):
        def run(self, arguments, **options):
            self.commands.append(("run", tuple(arguments), options))
            raise BootstrapError("example permission denied")

    runner = Unavailable([])
    with pytest.raises(BootstrapError, match="permission denied"):
        notify(runner, tmp_path)
    assert len(runner.commands) == 1


def test_ses_identity_creation_requires_successful_readback(tmp_path):
    runner = NotificationRunner([None, None])
    with pytest.raises(BootstrapError, match="could not be inspected"):
        notify(runner, tmp_path, channel="ses")
    assert not any(
        command[0] == "run" and command[1][0] == "kubectl"
        for command in runner.commands
    ), "unobserved SES identity authorized notification Secret changes"


def test_disabled_ses_account_never_gets_notification_secret_changes(tmp_path):
    runner = NotificationRunner(
        [{"VerifiedForSendingStatus": True}, {"SendingEnabled": False}]
    )
    with pytest.raises(BootstrapError, match="SES sending is disabled"):
        notify(runner, tmp_path, channel="ses")
    assert [command[0] for command in runner.commands] == ["aws", "aws"]

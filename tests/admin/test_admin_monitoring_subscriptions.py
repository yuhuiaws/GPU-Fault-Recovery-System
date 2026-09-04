from __future__ import annotations

from pathlib import Path

import pytest

from gpu_fault.admin import monitoring_subscriptions as subscriptions
from gpu_fault.admin.bootstrap_common import (
    BootstrapError,
    BootstrapState,
    ClusterIdentity,
)

TOPIC_ARN = "arn:aws:sns:us-west-2:123456789012:gpu-fault-alerts"
EMAIL = "ops@example.com"


class Runner:
    dry_run = False

    def __init__(self) -> None:
        self.subscribe_calls = 0
        self.pending = True

    def aws_text(self, _region, *arguments, **_kwargs):
        assert arguments[0:2] == ("sns", "subscribe")
        self.subscribe_calls += 1
        return (
            "arn:aws:sns:us-west-2:123456789012:"
            f"gpu-fault-alerts:email-{self.subscribe_calls}"
        )

    def aws_json(self, _region, *arguments, **_kwargs):
        assert arguments[0:2] == ("sns", "get-subscription-attributes")
        return {
            "Attributes": {
                "TopicArn": TOPIC_ARN,
                "PendingConfirmation": str(self.pending).lower(),
            }
        }


class FailingLookupRunner(Runner):
    def aws_json(self, _region, *arguments, **_kwargs):
        assert arguments[0:2] == ("sns", "get-subscription-attributes")
        raise BootstrapError("temporary SNS API failure")


def _cpu() -> ClusterIdentity:
    return ClusterIdentity(
        input_arn="arn:aws:eks:us-west-2:123456789012:cluster/cpu",
        role="cpu",
        region="us-west-2",
        account_id="123456789012",
        hyperpod_arn="arn:aws:sagemaker:us-west-2:123456789012:cluster/cpu",
        hyperpod_name="cpu",
        eks_arn="arn:aws:eks:us-west-2:123456789012:cluster/cpu",
        eks_name="cpu",
        vpc_id="vpc-cpu",
        subnet_ids=("subnet-a",),
        node_recovery="None",
        context="cpu",
    )


def _ensure(
    runner: Runner,
    state: BootstrapState,
    *,
    generation: str = "a" * 32,
    visible: list[dict] | None = None,
) -> dict:
    return subscriptions.ensure_email_subscription(
        runner,
        state=state,
        cpu=_cpu(),
        topic_arn=TOPIC_ARN,
        topic_generation=generation,
        endpoint=EMAIL,
        subscriptions=visible or [],
    )


def test_pending_checkpoint_suppresses_repeated_subscribe(tmp_path: Path) -> None:
    runner = Runner()
    state = BootstrapState(tmp_path / "bootstrap-state.json", site_id="site-a")

    first = _ensure(runner, state)
    repeated = _ensure(runner, state)

    assert runner.subscribe_calls == 1
    assert first["status"] == "PENDING"
    assert repeated["subscription_arn"] == first["subscription_arn"]
    assert (
        state.value["resources"][subscriptions.SNS_EMAIL_SUBSCRIPTION_STATE_KEY][
            "status"
        ]
        == "PENDING"
    )


def test_visible_pending_subscription_is_checkpointed_without_resubscribe(
    tmp_path: Path,
) -> None:
    runner = Runner()
    state = BootstrapState(tmp_path / "bootstrap-state.json", site_id="site-a")
    visible = [
        {
            "Protocol": "email",
            "Endpoint": EMAIL,
            "SubscriptionArn": "PendingConfirmation",
        }
    ]

    result = _ensure(runner, state, visible=visible)

    assert runner.subscribe_calls == 0
    assert result["status"] == "PENDING"
    assert result["visible_pending_count"] == 1


def test_topic_generation_change_allows_one_new_confirmation_request(
    tmp_path: Path,
) -> None:
    runner = Runner()
    state = BootstrapState(tmp_path / "bootstrap-state.json", site_id="site-a")

    first = _ensure(runner, state, generation="a" * 32)
    recreated = _ensure(runner, state, generation="b" * 32)

    assert runner.subscribe_calls == 2
    assert recreated["subscription_arn"] != first["subscription_arn"]
    assert recreated["topic_generation"] == "b" * 32


def test_duplicate_confirmed_email_subscriptions_fail_closed(tmp_path: Path) -> None:
    runner = Runner()
    state = BootstrapState(tmp_path / "bootstrap-state.json", site_id="site-a")
    confirmed = [
        {
            "Protocol": "email",
            "Endpoint": EMAIL,
            "SubscriptionArn": f"{TOPIC_ARN}:confirmed-{index}",
        }
        for index in range(2)
    ]

    with pytest.raises(BootstrapError, match="duplicate confirmed"):
        _ensure(runner, state, visible=confirmed)

    assert runner.subscribe_calls == 0


def test_confirmed_and_pending_email_subscriptions_fail_closed(tmp_path: Path) -> None:
    runner = Runner()
    state = BootstrapState(tmp_path / "bootstrap-state.json", site_id="site-a")
    mixed = [
        {
            "Protocol": "email",
            "Endpoint": EMAIL,
            "SubscriptionArn": f"{TOPIC_ARN}:confirmed",
        },
        {
            "Protocol": "email",
            "Endpoint": EMAIL,
            "SubscriptionArn": "PendingConfirmation",
        },
    ]

    with pytest.raises(BootstrapError, match="both confirmed and pending"):
        _ensure(runner, state, visible=mixed)

    assert runner.subscribe_calls == 0


def test_duplicate_pending_email_requests_fail_closed(tmp_path: Path) -> None:
    runner = Runner()
    state = BootstrapState(tmp_path / "bootstrap-state.json", site_id="site-a")
    pending = [
        {
            "Protocol": "email",
            "Endpoint": EMAIL,
            "SubscriptionArn": "PendingConfirmation",
        },
        {"Protocol": "email", "Endpoint": EMAIL, "SubscriptionArn": ""},
    ]

    with pytest.raises(BootstrapError, match="duplicate pending"):
        _ensure(runner, state, visible=pending)

    assert runner.subscribe_calls == 0


def test_confirmed_checkpoint_does_not_resubscribe_on_api_failure(
    tmp_path: Path,
) -> None:
    state = BootstrapState(tmp_path / "bootstrap-state.json", site_id="site-a")
    state.record(
        subscriptions.SNS_EMAIL_SUBSCRIPTION_STATE_KEY,
        {
            "schema_version": 1,
            "status": "CONFIRMED",
            "topic_arn": TOPIC_ARN,
            "topic_generation": "a" * 32,
            "endpoint": EMAIL,
            "subscription_arn": f"{TOPIC_ARN}:confirmed",
            "requested_at_epoch": 1.0,
            "expires_at_epoch": 1.0,
        },
    )
    runner = FailingLookupRunner()

    with pytest.raises(BootstrapError, match="refusing a duplicate Subscribe"):
        _ensure(runner, state)

    assert runner.subscribe_calls == 0

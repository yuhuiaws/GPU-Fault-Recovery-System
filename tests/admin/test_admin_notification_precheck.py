"""The first-minute SES/SNS check behind ``gpu-fault-admin deploy``."""

from __future__ import annotations

from pathlib import Path

from gpu_fault.admin import notification_precheck as PRECHECK
from gpu_fault.admin.bootstrap_common import (
    BootstrapError,
    BootstrapState,
    ClusterIdentity,
)

EMAIL = "ops@example.com"
TOPIC_ARN = "arn:aws:sns:us-west-2:123456789012:gpu-fault-site-a-alerts"
GENERATION = "0" * 32


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


class Runner:
    """Just enough AWS for the check: SES identity, SNS topic, subscriptions."""

    dry_run = False

    def __init__(self, *, ses_verified: bool | None, sns_confirmed: bool) -> None:
        self.ses_verified = ses_verified  # None: no identity yet
        self.sns_confirmed = sns_confirmed
        self.commands: list[tuple[str, ...]] = []

    def aws_json(self, _region, *arguments, **_kwargs):
        self.commands.append(arguments)
        service, action = arguments[0], arguments[1]
        if (service, action) == ("sesv2", "get-email-identity"):
            if self.ses_verified is None:
                raise BootstrapError("NotFoundException")
            return {
                "VerifiedForSendingStatus": self.ses_verified,
                "VerificationStatus": "SUCCESS" if self.ses_verified else "PENDING",
            }
        if (service, action) == ("sns", "get-topic-attributes"):
            return {"Attributes": {"TopicArn": TOPIC_ARN}}
        if (service, action) == ("sns", "list-tags-for-resource"):
            return {
                "Tags": [
                    {"Key": "gpu-fault:site-id", "Value": "site-a"},
                    {"Key": "gpu-fault:topic-generation", "Value": GENERATION},
                ]
            }
        if (service, action) == ("sns", "list-subscriptions-by-topic"):
            if not self.sns_confirmed:
                return {"Subscriptions": []}
            return {
                "Subscriptions": [
                    {
                        "Protocol": "email",
                        "Endpoint": EMAIL,
                        "SubscriptionArn": TOPIC_ARN + ":confirmed",
                    }
                ]
            }
        raise AssertionError(f"unexpected aws call {arguments}")

    def aws_text(self, _region, *arguments, **_kwargs):
        self.commands.append(arguments)
        assert arguments[0:2] == ("sns", "subscribe")
        return TOPIC_ARN + ":pending"

    def run(self, arguments, **_kwargs):
        self.commands.append(tuple(arguments))
        if "create-email-identity" in arguments:
            self.ses_verified = False
        return ""


def test_first_run_sends_both_mails_and_reports_both_pending(tmp_path: Path) -> None:
    runner = Runner(ses_verified=None, sns_confirmed=False)
    state = BootstrapState(tmp_path / "bootstrap-state.json", site_id="site-a")

    result = PRECHECK.check_email_confirmations(
        runner, cpu=_cpu(), site_id="site-a", admin_email=EMAIL, state=state
    )

    assert result.confirmed is False
    assert result.ses_verified is False and result.ses_identity_created is True
    assert result.sns_status == "PENDING" and result.sns_topic_arn == TOPIC_ARN
    creates = [c for c in runner.commands if "create-email-identity" in c]
    subscribes = [c for c in runner.commands if c[0:2] == ("sns", "subscribe")]
    assert len(creates) == 1 and len(subscribes) == 1, "each mail is sent once"
    # The subscription request is recorded where the later bootstrap task reads
    # it, so that task recognises it and never subscribes a second time.
    assert state.value["resources"]["sns_email_subscription"]["status"] == "PENDING"

    message = PRECHECK.email_confirmation_refusal(
        result, rerun_command="gpu-fault-admin deploy --state-dir /secure/x"
    )
    assert f"SES sender identity {EMAIL}: verification mail sent, PENDING" in message
    assert f"SNS alert subscription {EMAIL} on {TOPIC_ARN}: confirmation mail sent" in (
        message
    )
    assert "rerun: gpu-fault-admin deploy --state-dir /secure/x" in message
    assert PRECHECK.WAIT_FLAG in message


def test_rerun_after_both_links_is_confirmed_and_sends_nothing(tmp_path: Path) -> None:
    runner = Runner(ses_verified=True, sns_confirmed=True)

    result = PRECHECK.check_email_confirmations(
        runner, cpu=_cpu(), site_id="site-a", admin_email=EMAIL, state=None
    )

    assert result.confirmed is True
    assert not any("create-email-identity" in c for c in runner.commands), (
        "a verified identity is only read"
    )
    assert not any(c[0:2] == ("sns", "subscribe") for c in runner.commands), (
        "a confirmed subscription is only read"
    )


def test_one_pending_address_is_still_a_refusal(tmp_path: Path) -> None:
    result = PRECHECK.check_email_confirmations(
        Runner(ses_verified=True, sns_confirmed=False),
        cpu=_cpu(),
        site_id="site-a",
        admin_email=EMAIL,
        state=None,
    )

    assert result.ses_verified is True and result.sns_confirmed is False
    assert result.confirmed is False
    message = PRECHECK.email_confirmation_refusal(result, rerun_command="rerun")
    assert f"SES sender identity {EMAIL}: verified" in message
    assert "confirmation mail sent" in message


def test_wait_polls_until_confirmed_and_a_zero_wait_checks_once() -> None:
    answers = iter(
        [
            _result(ses=False, sns="PENDING"),
            _result(ses=True, sns="PENDING"),
            _result(ses=True, sns="CONFIRMED"),
        ]
    )
    calls = 0
    slept: list[float] = []
    now = [0.0]

    def check() -> PRECHECK.EmailConfirmation:
        nonlocal calls
        calls += 1
        return next(answers)

    def sleep(seconds: float) -> None:
        slept.append(seconds)
        now[0] += seconds

    result = PRECHECK.await_email_confirmations(
        check, wait_minutes=5, sleep=sleep, clock=lambda: now[0], poll_seconds=30
    )

    assert result.confirmed is True
    assert calls == 3 and slept == [30, 30]

    calls = 0
    answers = iter([_result(ses=False, sns="PENDING")])
    result = PRECHECK.await_email_confirmations(
        check, wait_minutes=0, sleep=sleep, clock=lambda: now[0]
    )
    assert calls == 1 and result.confirmed is False


def test_wait_gives_up_at_the_bound() -> None:
    now = [0.0]

    def sleep(seconds: float) -> None:
        now[0] += seconds

    calls = 0

    def check() -> PRECHECK.EmailConfirmation:
        nonlocal calls
        calls += 1
        return _result(ses=False, sns="PENDING")

    result = PRECHECK.await_email_confirmations(
        check, wait_minutes=1, sleep=sleep, clock=lambda: now[0], poll_seconds=30
    )

    assert result.confirmed is False
    assert calls == 3, "one check at 0 s, then at 30 s and 60 s, then the bound"


def _result(*, ses: bool, sns: str) -> PRECHECK.EmailConfirmation:
    return PRECHECK.EmailConfirmation(
        sender=EMAIL,
        admin_email=EMAIL,
        ses_verified=ses,
        ses_identity_created=False,
        sns_topic_arn=TOPIC_ARN,
        sns_status=sns,
        sns_subscription_arn=None,
    )

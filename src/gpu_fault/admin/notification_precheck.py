"""The first-minute email check of ``gpu-fault-admin deploy``.

One address has to be confirmed by a human before a site can notify anyone:
the SNS subscription of the administrator on the site topic (SNS mails a
confirmation link). That topic carries both the AMP alerts and, with the
default ``channel: sns``, the control plane's own notifications, so it is the
whole setup. A site that opted into ``channel: ses`` also needs its SES sender
identity verified (SES mails a verification link) and is told about both.

Until now the SES check ran deep inside bootstrap -- after the source scan, the
release gates and the release build -- and the SNS check only inside ``verify``
at the very end. Twenty minutes to learn that a mailbox has a link waiting in
it. This runs before any gate or build. Every request is idempotent (an
identity or subscription that exists is only read), so the check sends each
mail once, reports the address(es) in one message with the exact rerun line,
and lets the rerun proceed once the link was clicked.
``--wait-for-email-confirmation N`` polls for up to N minutes instead.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, cast

from gpu_fault.admin.bootstrap_common import (
    BootstrapState,
    ClusterIdentity,
    CommandRunner,
)
from gpu_fault.admin.bootstrap_services import ensure_sns_topic
from gpu_fault.admin.monitoring_subscriptions import ensure_email_subscription
from gpu_fault.admin.notifications import (
    ensure_ses_identity,
    resolve_notification_routing,
    ses_identity_verified,
)
from gpu_fault.admin.site import NOTIFICATION_CHANNEL_SES, NOTIFICATION_CHANNEL_SNS

WAIT_FLAG = "--wait-for-email-confirmation"
POLL_SECONDS = 30.0


@dataclass(frozen=True)
class EmailConfirmation:
    """What the check found for the subscription and, for ``ses``, the sender."""

    sender: str
    admin_email: str
    ses_verified: bool
    ses_identity_created: bool
    sns_topic_arn: str
    sns_status: str
    sns_subscription_arn: str | None
    channel: str = NOTIFICATION_CHANNEL_SNS

    @property
    def uses_ses(self) -> bool:
        return self.channel == NOTIFICATION_CHANNEL_SES

    @property
    def sns_confirmed(self) -> bool:
        return self.sns_status == "CONFIRMED"

    @property
    def confirmed(self) -> bool:
        return self.sns_confirmed and (self.ses_verified or not self.uses_ses)

    def as_dict(self) -> dict[str, Any]:
        return {
            "channel": self.channel,
            "sender": self.sender,
            "admin_email": self.admin_email,
            "ses_verified": self.ses_verified,
            "ses_identity_created": self.ses_identity_created,
            "sns_topic_arn": self.sns_topic_arn,
            "sns_status": self.sns_status,
            "sns_subscription_arn": self.sns_subscription_arn,
        }


def check_email_confirmations(
    runner: CommandRunner,
    *,
    cpu: ClusterIdentity,
    site_id: str,
    admin_email: str,
    state: BootstrapState | None,
    channel: str = NOTIFICATION_CHANNEL_SNS,
) -> EmailConfirmation:
    """Send (once) and read the confirmation(s) the channel needs.

    The SNS topic is created here on a first deploy -- the same tagged
    ``create-topic`` bootstrap's monitoring task issues later, which then finds
    it and reuses it. ``state`` is the bootstrap state the later task reads, so
    the subscription request recorded here is the one it recognises and it
    never subscribes a second time. With ``channel="sns"`` no SES API is
    called: there is no identity to create or verify.
    """

    routing = resolve_notification_routing(admin_email=admin_email, channel=channel)
    ses_verified, created = True, False
    if routing.uses_ses:
        identity, created = ensure_ses_identity(
            runner, region=cpu.region, sender=routing.sender, site_id=site_id
        )
        ses_verified = ses_identity_verified(identity)
    topic_arn, _reused, generation = ensure_sns_topic(runner, cpu=cpu, site_id=site_id)
    subscriptions = cast(
        list[dict[str, Any]],
        runner.aws_json(
            cpu.region,
            "sns",
            "list-subscriptions-by-topic",
            "--topic-arn",
            topic_arn,
        ).get("Subscriptions", []),
    )
    subscription = ensure_email_subscription(
        runner,
        state=state,
        cpu=cpu,
        topic_arn=topic_arn,
        topic_generation=generation,
        endpoint=admin_email,
        subscriptions=subscriptions,
    )
    return EmailConfirmation(
        sender=routing.sender,
        admin_email=admin_email,
        ses_verified=ses_verified,
        ses_identity_created=created,
        channel=routing.channel,
        sns_topic_arn=topic_arn,
        sns_status=str(subscription.get("status") or ""),
        sns_subscription_arn=(
            str(subscription["subscription_arn"])
            if subscription.get("subscription_arn")
            else None
        ),
    )


def await_email_confirmations(
    check: Callable[[], EmailConfirmation],
    *,
    wait_minutes: int,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
    poll_seconds: float = POLL_SECONDS,
) -> EmailConfirmation:
    """Run ``check`` until both addresses are confirmed or the wait is over.

    ``wait_minutes=0`` (the default) checks exactly once.
    """

    result = check()
    deadline = clock() + max(0, wait_minutes) * 60
    while not result.confirmed and clock() < deadline:
        sleep(poll_seconds)
        result = check()
    return result


def email_confirmation_refusal(result: EmailConfirmation, *, rerun_command: str) -> str:
    """One message naming the address(es) the channel needs and the rerun line."""

    mails = "the two mails" if result.uses_ses else "the confirmation mail"
    links = "both links" if result.uses_ses else "the link"
    lines = [
        "email notifications are not confirmed yet; nothing was deployed. "
        f"Click the link(s) in {mails}, then rerun the same command:",
    ]
    if result.uses_ses:
        lines.append(
            f"  SES sender identity {result.sender}: "
            + ("verified" if result.ses_verified else "verification mail sent, PENDING")
        )
    lines.append(
        f"  SNS alert subscription {result.admin_email} on {result.sns_topic_arn}: "
        + (
            "confirmed"
            if result.sns_confirmed
            else f"confirmation mail sent, {result.sns_status or 'PENDING'}"
        )
    )
    lines.append(f"  rerun: {rerun_command}")
    lines.append(f"  (or add {WAIT_FLAG} MINUTES to the command to wait for {links})")
    return "\n".join(lines)

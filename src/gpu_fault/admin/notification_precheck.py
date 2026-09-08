"""The first-minute email check of ``gpu-fault-admin deploy``.

Two addresses have to be confirmed by a human before a site can notify anyone:
the SES sender identity (SES mails a verification link) and the SNS alert
subscription (SNS mails a confirmation link). Until now the SES check ran deep
inside bootstrap -- after the source scan, the release gates and the release
build -- and stopped with ``complete verification and rerun``, and the SNS
check ran only inside ``verify`` at the very end. Twenty minutes to learn that a
mailbox has a link waiting in it.

This runs before any gate or build. Both requests are idempotent (an identity
or subscription that exists is only read), so the check sends each mail once,
reports both addresses in one message with the exact rerun line, and lets the
rerun proceed once both links were clicked. ``--wait-for-email-confirmation N``
polls for up to N minutes instead of stopping.
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

WAIT_FLAG = "--wait-for-email-confirmation"
POLL_SECONDS = 30.0


@dataclass(frozen=True)
class EmailConfirmation:
    """What the check found for the two addresses."""

    sender: str
    admin_email: str
    ses_verified: bool
    ses_identity_created: bool
    sns_topic_arn: str
    sns_status: str
    sns_subscription_arn: str | None

    @property
    def sns_confirmed(self) -> bool:
        return self.sns_status == "CONFIRMED"

    @property
    def confirmed(self) -> bool:
        return self.ses_verified and self.sns_confirmed

    def as_dict(self) -> dict[str, Any]:
        return {
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
) -> EmailConfirmation:
    """Send (once) and read both confirmations.

    The SNS topic is created here on a first deploy -- the same tagged
    ``create-topic`` bootstrap's monitoring task issues later, which then finds
    it and reuses it. ``state`` is the bootstrap state the later task reads, so
    the subscription request recorded here is the one it recognises and it
    never subscribes a second time.
    """

    routing = resolve_notification_routing(admin_email=admin_email)
    identity, created = ensure_ses_identity(
        runner, region=cpu.region, sender=routing.sender, site_id=site_id
    )
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
        ses_verified=ses_identity_verified(identity),
        ses_identity_created=created,
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
    """One message naming both addresses and the rerun line."""

    lines = [
        "email notifications are not confirmed yet; nothing was deployed. "
        "Click the links in the two mails, then rerun the same command:",
    ]
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
    lines.append(
        f"  (or add {WAIT_FLAG} MINUTES to the command to wait for both links)"
    )
    return "\n".join(lines)

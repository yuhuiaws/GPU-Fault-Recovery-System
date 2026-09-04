from __future__ import annotations

import time
from collections.abc import Mapping
from typing import Any, cast

from gpu_fault.admin.bootstrap_common import (
    BootstrapError,
    BootstrapState,
    ClusterIdentity,
    CommandRunner,
)

SNS_EMAIL_SUBSCRIPTION_STATE_KEY = "sns_email_subscription"
SNS_EMAIL_CONFIRMATION_TTL_SECONDS = 48 * 60 * 60
SNS_EMAIL_REQUESTING_SUPPRESSION_SECONDS = 5 * 60


def _subscription_arn(value: object) -> str | None:
    normalized = str(value or "").strip()
    if not normalized or normalized == "PendingConfirmation":
        return None
    return normalized if normalized.startswith("arn:") else None


def _email_subscription_matches(
    item: Mapping[str, Any],
    endpoint: str,
) -> bool:
    return (
        str(item.get("Protocol") or "").lower() == "email"
        and str(item.get("Endpoint") or "").casefold() == endpoint.casefold()
    )


def _record_email_subscription(
    state: BootstrapState | None,
    value: dict[str, Any],
) -> dict[str, Any]:
    if state is not None:
        state.record(SNS_EMAIL_SUBSCRIPTION_STATE_KEY, value)
    return value


def _previous_email_subscription(
    state: BootstrapState | None,
) -> dict[str, Any]:
    if state is None:
        return {}
    value = (state.value.get("resources") or {}).get(SNS_EMAIL_SUBSCRIPTION_STATE_KEY)
    return dict(value) if isinstance(value, dict) else {}


def _checkpoint(
    *,
    status: str,
    topic_arn: str,
    topic_generation: str,
    endpoint: str,
    subscription_arn: str | None,
    requested_at_epoch: float,
    expires_at_epoch: float,
    visible_pending_count: int = 0,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": status,
        "topic_arn": topic_arn,
        "topic_generation": topic_generation,
        "endpoint": endpoint,
        "subscription_arn": subscription_arn,
        "requested_at_epoch": requested_at_epoch,
        "expires_at_epoch": expires_at_epoch,
        "visible_pending_count": visible_pending_count,
        "updated_at_epoch": time.time(),
    }


def _confirmed_checkpoint(
    *,
    state: BootstrapState | None,
    topic_arn: str,
    topic_generation: str,
    endpoint: str,
    subscription_arn: str,
    requested_at_epoch: float,
    expires_at_epoch: float,
) -> dict[str, Any]:
    return _record_email_subscription(
        state,
        _checkpoint(
            status="CONFIRMED",
            topic_arn=topic_arn,
            topic_generation=topic_generation,
            endpoint=endpoint,
            subscription_arn=subscription_arn,
            requested_at_epoch=requested_at_epoch,
            expires_at_epoch=expires_at_epoch,
        ),
    )


def ensure_email_subscription(
    runner: CommandRunner,
    *,
    state: BootstrapState | None,
    cpu: ClusterIdentity,
    topic_arn: str,
    topic_generation: str,
    endpoint: str,
    subscriptions: list[dict[str, Any]],
) -> dict[str, Any]:
    now = time.time()
    matches = [
        item for item in subscriptions if _email_subscription_matches(item, endpoint)
    ]
    confirmed = sorted(
        {
            arn
            for item in matches
            if (arn := _subscription_arn(item.get("SubscriptionArn"))) is not None
        }
    )
    pending = [
        item
        for item in matches
        if _subscription_arn(item.get("SubscriptionArn")) is None
    ]
    if len(confirmed) > 1:
        raise BootstrapError(
            "SNS topic has duplicate confirmed email subscriptions for "
            f"{endpoint}; keep one subscription and remove the duplicates"
        )
    if confirmed and pending:
        raise BootstrapError(
            "SNS topic has both confirmed and pending email subscriptions for "
            f"{endpoint}; remove or let the pending request expire"
        )
    if len(pending) > 1:
        raise BootstrapError(
            "SNS topic has duplicate pending email confirmation requests for "
            f"{endpoint}; do not confirm multiple requests and wait for expiry"
        )
    if confirmed:
        return _confirmed_checkpoint(
            state=state,
            topic_arn=topic_arn,
            topic_generation=topic_generation,
            endpoint=endpoint,
            subscription_arn=confirmed[0],
            requested_at_epoch=now,
            expires_at_epoch=now,
        )

    previous = _previous_email_subscription(state)
    same_request = (
        previous.get("topic_arn") == topic_arn
        and previous.get("topic_generation") == topic_generation
        and str(previous.get("endpoint") or "").casefold() == endpoint.casefold()
    )
    requested_at = (
        float(previous.get("requested_at_epoch") or now) if same_request else now
    )
    expires_at = (
        float(
            previous.get("expires_at_epoch")
            or requested_at + SNS_EMAIL_CONFIRMATION_TTL_SECONDS
        )
        if same_request
        else now + SNS_EMAIL_CONFIRMATION_TTL_SECONDS
    )
    if pending:
        return _record_email_subscription(
            state,
            _checkpoint(
                status="PENDING",
                topic_arn=topic_arn,
                topic_generation=topic_generation,
                endpoint=endpoint,
                subscription_arn=(
                    _subscription_arn(previous.get("subscription_arn"))
                    if same_request
                    else None
                ),
                requested_at_epoch=requested_at,
                expires_at_epoch=expires_at,
                visible_pending_count=len(pending),
            ),
        )

    if same_request:
        previous_arn = _subscription_arn(previous.get("subscription_arn"))
        if previous_arn:
            try:
                attributes = runner.aws_json(
                    cpu.region,
                    "sns",
                    "get-subscription-attributes",
                    "--subscription-arn",
                    previous_arn,
                ).get("Attributes", {})
            except BootstrapError as exc:
                missing = "notfound" in str(exc).lower() or (
                    "not found" in str(exc).lower()
                )
                if str(previous.get("status") or "") == "CONFIRMED" and not missing:
                    raise BootstrapError(
                        "cannot verify the persisted confirmed SNS email "
                        "subscription; refusing a duplicate Subscribe"
                    ) from exc
                attributes = None
            if isinstance(attributes, dict):
                if str(attributes.get("TopicArn") or topic_arn) != topic_arn:
                    raise BootstrapError(
                        "persisted SNS email subscription belongs to another topic"
                    )
                if str(attributes.get("PendingConfirmation") or "").lower() != "true":
                    return _confirmed_checkpoint(
                        state=state,
                        topic_arn=topic_arn,
                        topic_generation=topic_generation,
                        endpoint=endpoint,
                        subscription_arn=previous_arn,
                        requested_at_epoch=requested_at,
                        expires_at_epoch=expires_at,
                    )
                if now < expires_at:
                    return previous
            elif now < expires_at:
                return previous
        status = str(previous.get("status") or "")
        if (
            status == "REQUESTING"
            and now < requested_at + SNS_EMAIL_REQUESTING_SUPPRESSION_SECONDS
        ):
            return previous
        if status == "PENDING" and now < expires_at:
            return previous

    intent = _checkpoint(
        status="REQUESTING",
        topic_arn=topic_arn,
        topic_generation=topic_generation,
        endpoint=endpoint,
        subscription_arn=None,
        requested_at_epoch=now,
        expires_at_epoch=now + SNS_EMAIL_CONFIRMATION_TTL_SECONDS,
    )
    _record_email_subscription(state, intent)
    subscription_arn = runner.aws_text(
        cpu.region,
        "sns",
        "subscribe",
        "--topic-arn",
        topic_arn,
        "--protocol",
        "email",
        "--notification-endpoint",
        endpoint,
        "--return-subscription-arn",
        "--query",
        "SubscriptionArn",
        mutate=True,
    )
    normalized_arn = _subscription_arn(subscription_arn)
    if normalized_arn is None:
        raise BootstrapError("SNS email Subscribe returned no subscription ARN")
    return _record_email_subscription(
        state,
        _checkpoint(
            status="PENDING",
            topic_arn=topic_arn,
            topic_generation=topic_generation,
            endpoint=endpoint,
            subscription_arn=normalized_arn,
            requested_at_epoch=now,
            expires_at_epoch=now + SNS_EMAIL_CONFIRMATION_TTL_SECONDS,
        ),
    )


def ensure_monitoring_subscriptions(
    runner: CommandRunner,
    *,
    state: BootstrapState | None,
    cpu: ClusterIdentity,
    topic_arn: str,
    topic_generation: str,
    queue_arn: str,
    alert_email: str | None,
) -> tuple[str, str, dict[str, Any] | None]:
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
    queue_subscription = next(
        (
            item
            for item in subscriptions
            if str(item.get("Protocol") or "").lower() == "sqs"
            and item.get("Endpoint") == queue_arn
        ),
        None,
    )
    if queue_subscription is None:
        queue_subscription_arn = runner.aws_text(
            cpu.region,
            "sns",
            "subscribe",
            "--topic-arn",
            topic_arn,
            "--protocol",
            "sqs",
            "--notification-endpoint",
            queue_arn,
            "--query",
            "SubscriptionArn",
            mutate=True,
        )
        queue_subscription_ownership = "CREATED"
    else:
        queue_subscription_arn = str(queue_subscription.get("SubscriptionArn") or "")
        queue_subscription_ownership = "CREATED"
    email_subscription = (
        ensure_email_subscription(
            runner,
            state=state,
            cpu=cpu,
            topic_arn=topic_arn,
            topic_generation=topic_generation,
            endpoint=alert_email,
            subscriptions=subscriptions,
        )
        if alert_email
        else None
    )
    return queue_subscription_arn, queue_subscription_ownership, email_subscription

from __future__ import annotations

import base64
from typing import Any

from gpu_fault_release.regional_release_config import ReleaseError


def email_subscription_summary(
    subscriptions: list[dict[str, Any]],
    endpoint: str | None,
) -> dict[str, Any]:
    if not endpoint:
        return {"endpoint": None, "confirmed": 0, "pending": 0}
    matches = [
        item
        for item in subscriptions
        if str(item.get("Protocol") or "").lower() == "email"
        and str(item.get("Endpoint") or "").casefold() == endpoint.casefold()
    ]
    confirmed = [
        item
        for item in matches
        if item.get("SubscriptionArn") not in {None, "", "PendingConfirmation"}
    ]
    pending = [item for item in matches if item not in confirmed]
    if len(confirmed) > 1:
        raise ReleaseError(
            f"SNS topic has duplicate confirmed email subscriptions for {endpoint}"
        )
    if confirmed and pending:
        raise ReleaseError(
            "SNS topic has both confirmed and pending email subscriptions for "
            f"{endpoint}"
        )
    if len(pending) > 1:
        raise ReleaseError(
            "SNS topic has duplicate pending email confirmation requests for "
            f"{endpoint}"
        )
    return {
        "endpoint": endpoint,
        "confirmed": len(confirmed),
        "pending": len(pending),
    }


def decode_monitoring_configuration(
    rules_document: dict[str, Any],
    manager_document: dict[str, Any],
    topic_arn: str,
) -> tuple[str, str]:
    rules_data = (rules_document.get("ruleGroupsNamespace") or {}).get("data")
    manager_data = (manager_document.get("alertManagerDefinition") or {}).get("data")
    if not rules_data or not manager_data:
        raise ReleaseError("AMP live rules or Alertmanager definition has no data")
    try:
        rules_text = base64.b64decode(rules_data).decode()
        manager_text = base64.b64decode(manager_data).decode()
    except (ValueError, UnicodeDecodeError) as exc:
        raise ReleaseError("AMP live configuration is not valid base64") from exc
    if topic_arn not in manager_text:
        raise ReleaseError(
            "AMP Alertmanager does not reference the configured SNS topic"
        )
    return rules_text, manager_text

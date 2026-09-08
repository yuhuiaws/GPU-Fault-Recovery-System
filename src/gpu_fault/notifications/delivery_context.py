"""The site context every delivered notification carries, channel-independent.

SES and SNS render the same subject and the same body: the operator reading a
mail must see which site, account, Region and cluster it is about before the
first line of the template, whichever service carried it. Kept out of the
adapters so the two cannot drift apart.
"""

from __future__ import annotations

from gpu_fault.notifications.common import AdvisoryNotification

CONTEXT_HEADER = "通知上下文"


def subject_with_context(
    notification: AdvisoryNotification,
    *,
    subject_prefix: str,
    site_id: str | None,
    region_name: str | None,
    account_id: str | None,
) -> str:
    context = [
        value
        for value in (
            subject_prefix,
            f"[site:{site_id}]" if site_id else None,
            f"[region:{region_name}]" if region_name else None,
            f"[account:{account_id}]" if account_id else None,
        )
        if value
    ]
    return " ".join([*context, notification.subject])


def body_with_context(
    notification: AdvisoryNotification,
    *,
    site_id: str | None,
    account_id: str | None,
    region_name: str | None,
) -> str:
    context = [
        ("Site", site_id),
        ("AWS Account", account_id),
        ("Region", region_name),
        ("Cluster", notification.cluster_name),
    ]
    if not any(value for _label, value in context[:-1]):
        return notification.body_text
    header = "\n".join(
        [
            CONTEXT_HEADER,
            *[f"- {label}: {value or 'UNKNOWN'}" for label, value in context],
        ]
    )
    return f"{header}\n\n{notification.body_text}"

from gpu_fault.models import AdvisoryNotification, FaultIncident

PERFORMANCE_CLUSTER_PREFIX = "perf-cap-"
PERFORMANCE_DRILL_ID = "perf-capacity"


def bound_notifications(
    notifications: list[AdvisoryNotification],
    *,
    limit: int | None,
    newest_first: bool,
) -> list[AdvisoryNotification]:
    """Order and truncate a read of the notification table.

    Nothing ever leaves the notification table, and nothing ever leaves the set
    ``dispatch_pending`` treats as pending either: an entry qualifies when it has
    no result or a ``SKIPPED``/``FAILED`` one, and both a drill and an expired
    advisory are recorded as ``SKIPPED``. The synchronous dispatch path therefore
    read the entire history on every call and issued one extra result lookup per
    row. This is how it asks for a bounded slice instead, in the same shape
    ``list_workflows`` and ``list_attempt_observation_states`` already offer.

    Ordering is left exactly as it was for the unbounded ascending case, which is
    what every other caller uses: sorted on ``created_at`` alone, ties in
    whatever order the storage produced them. A bounded or reversed read has to
    be deterministic to be a stable page, so it breaks ties on
    ``notification_id``.
    """

    if limit is not None and limit < 0:
        raise ValueError("notification scan limit must not be negative")
    if limit is None and not newest_first:
        return notifications
    ordered = sorted(
        notifications,
        key=lambda item: (item.created_at, item.notification_id),
        reverse=newest_first,
    )
    return ordered if limit is None else ordered[:limit]


def with_incident_drill_label(
    notification: AdvisoryNotification,
    incident: FaultIncident | None,
) -> AdvisoryNotification:
    if notification.drill_id is not None:
        return notification
    drill_id = incident.drill_id if incident is not None else None
    if drill_id is None and notification.cluster_name.startswith(
        PERFORMANCE_CLUSTER_PREFIX
    ):
        drill_id = PERFORMANCE_DRILL_ID
    if drill_id is None:
        return notification
    return notification.model_copy(
        update={
            "drill_id": drill_id,
            "subject": f"[DRILL:{drill_id}] {notification.subject}",
            "body_text": (
                "【演练通知 / DRILL - 非真实故障】\n"
                f"Drill ID：{drill_id}\n\n" + notification.body_text
            ),
        }
    )

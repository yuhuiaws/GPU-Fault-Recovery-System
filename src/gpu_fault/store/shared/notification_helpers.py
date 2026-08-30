from gpu_fault.models import AdvisoryNotification, FaultIncident

PERFORMANCE_CLUSTER_PREFIX = "perf-cap-"
PERFORMANCE_DRILL_ID = "perf-capacity"


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

from gpu_fault.models import AdvisoryNotification, FaultIncident


def with_incident_drill_label(
    notification: AdvisoryNotification,
    incident: FaultIncident | None,
) -> AdvisoryNotification:
    if (
        notification.drill_id is not None
        or incident is None
        or incident.drill_id is None
    ):
        return notification
    drill_id = incident.drill_id
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

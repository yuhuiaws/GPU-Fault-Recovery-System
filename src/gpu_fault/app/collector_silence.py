from __future__ import annotations

from datetime import datetime

from gpu_fault.app.context import ApplicationContext
from gpu_fault.collector_requirements import (
    agent_is_current,
    collector_silent_thresholds,
    required_collectors_for_agent,
)
from gpu_fault.models import AdvisoryNotification
from gpu_fault.telemetry import CollectorKind, collector_producer


def notify_silent_collectors(
    context: ApplicationContext,
    *,
    observed_at: datetime,
    silent_after_seconds: float | dict[CollectorKind, float],
    alert_interval_seconds: float,
) -> int:
    bucket = int(observed_at.timestamp() // alert_interval_seconds)
    sent = 0
    defaults = collector_silent_thresholds()
    thresholds = (
        {**defaults, **silent_after_seconds}
        if isinstance(silent_after_seconds, dict)
        else {collector: silent_after_seconds for collector in defaults}
    )
    for registration in context.store.list_regional_clusters():
        if not registration.is_active(observed_at):
            continue
        statuses = {
            (item.node_id, item.collector): item
            for item in context.store.list_collector_statuses(registration.cluster_id)
        }
        for agent in context.store.list_agents(registration.cluster_id):
            if not agent_is_current(agent, observed_at=observed_at):
                continue
            for collector in required_collectors_for_agent(agent):
                status = statuses.get((agent.node_id, collector))
                last_success = status.last_success_at if status is not None else None
                if (
                    last_success is not None
                    and (observed_at - last_success).total_seconds()
                    <= thresholds[collector]
                ):
                    continue
                candidate = AdvisoryNotification(
                    deduplication_key=(
                        "collector-silent/"
                        f"{registration.cluster_id}/"
                        f"{agent.node_id}/"
                        f"{collector.value}/{bucket}"
                    ),
                    cluster_name=registration.cluster_id,
                    incident_id=(f"collector-silent-{agent.node_id}"),
                    subject=(
                        "[GPU collector warning] "
                        f"{agent.node_id} "
                        f"{collector_producer(collector)} "
                        f"channel {collector.value} is silent"
                    ),
                    body_text=(
                        f"Cluster: {registration.cluster_id}\n"
                        f"Node: {agent.node_id}\n"
                        "Collector: "
                        f"{collector_producer(collector)}\n"
                        f"Signal channel: {collector.value}\n"
                        f"Last success: {last_success}"
                    ),
                    support_case_draft=(
                        "Check systemd service, TLS/token and collector outbox."
                    ),
                )
                notification = context.store.save_notification_if_absent(candidate)
                if notification.notification_id != candidate.notification_id:
                    # Same hour bucket, same collector: the notification is
                    # already in the outbox and the dispatcher owns its
                    # retries. Sending it again every scan re-queued a
                    # RETRY/DEAD delivery and made its attempt budget
                    # unbounded (control-plane review 2026-09-08, F-2).
                    continue
                context.advisory_notifications.send(notification.notification_id)
                sent += 1
    return sent

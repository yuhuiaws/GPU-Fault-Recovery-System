from __future__ import annotations

from datetime import datetime
from typing import Any

from gpu_fault.app.context import ApplicationContext, is_placeholder_registration
from gpu_fault.collector_requirements import (
    agent_is_current,
    collector_silent_thresholds,
    required_collectors_for_agent,
)
from gpu_fault.models import AdvisoryNotification
from gpu_fault.store import NotFoundError
from gpu_fault.telemetry import CollectorKind, collector_producer


def _watched(registration: Any, observed_at: datetime) -> bool:
    """A cluster this scan alerts for: registered ACTIVE and not a placeholder.

    Every node of a synthetic registration is silent by construction, so a
    perf hour used to write thousands of drill advisories (batch 100).
    """

    return not is_placeholder_registration(registration) and registration.is_active(
        observed_at
    )


def _still_watched(
    context: ApplicationContext, cluster_id: str, node_id: str, observed_at: datetime
) -> bool:
    """Re-read the cluster and the Agent right before writing their advisory.

    The scan reads a cluster's Agents first and writes after the silence
    checks. A purge committed in between (remove-cluster, a perf teardown) got
    advisories written after it: 39 notifications with their delivery/result
    records and dedup links once made the next perf suite's registration
    refuse. A cluster no longer registered ACTIVE, or an Agent whose record is
    gone or no longer current, gets none.
    """

    try:
        registration = context.store.get_regional_cluster(cluster_id)
        agent = context.store.get_agent(cluster_id, node_id)
    except NotFoundError:
        return False
    return _watched(registration, observed_at) and agent_is_current(
        agent, observed_at=observed_at
    )


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
        if not _watched(registration, observed_at):
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
                if not _still_watched(
                    context, registration.cluster_id, agent.node_id, observed_at
                ):
                    break
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

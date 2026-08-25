"""Shared fixtures and builders for split test shards."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from gpu_fault.hyperpod import HyperPodAdvisoryDisposition, HyperPodRecoveryAdvisory
from gpu_fault.models import (
    AdvisoryNotification,
    CapabilityName,
    MarkerScope,
    NodeMarker,
    NotificationDelivery,
    NotificationResult,
    NotificationStatus,
    RecoveryAction,
    Severity,
)
from gpu_fault.notification_service import AdvisoryNotificationService
from gpu_fault.notifications import (
    HyperPodAdvisoryEmailBuilder,
    SesEmailNotifier,
    SesNotificationConfig,
)
from gpu_fault.store import InMemoryStore, SqliteStore, WorkflowLeaseError
from tests._builders import build_store, fault_incident


class FakeSesV2Client:
    def __init__(self) -> None:
        self.requests: list[dict] = []

    def send_email(self, **kwargs):
        self.requests.append(kwargs)
        return {"MessageId": "ses-message-1"}


class PartiallyFailingNotifier:
    def send(self, notification):
        if notification.incident_id == "incident-fail":
            raise RuntimeError("SES unavailable")
        return NotificationResult(
            notification_id=notification.notification_id,
            status=NotificationStatus.SENT,
            provider_message_id="message-ok",
        )


class RecordingNotifier:
    def __init__(self) -> None:
        self.notifications = []

    def send(self, notification):
        self.notifications.append(notification)
        return NotificationResult(
            notification_id=notification.notification_id,
            status=NotificationStatus.SENT,
            provider_message_id="ses-not-applicable-1",
        )


def advisory() -> HyperPodRecoveryAdvisory:
    return HyperPodRecoveryAdvisory(
        capability=CapabilityName.NODE_REPLACE,
        recommended_action="REPLACE_NODE",
        execution_owner="hyperpod-managed-node-recovery",
        disposition=HyperPodAdvisoryDisposition.ADVISE_ONLY,
        rationale=["repeated XID 79", "DCGM diagnostic failed"],
        evidence_refs=["s3://evidence/incident-42.json"],
    )


def _notification_marker(
    marker_id: str, incident_id: str, *, seconds: int
) -> NodeMarker:
    observed_at = datetime(2026, 8, 18, tzinfo=timezone.utc)
    return NodeMarker(
        marker_id=marker_id,
        source="notification-test",
        trusted=True,
        incident_id=incident_id,
        observed_at=observed_at + timedelta(seconds=seconds),
        expires_at=observed_at + timedelta(hours=1),
        scope=MarkerScope(node_ids=["node-a"]),
        severity=Severity.CRITICAL,
        recommended_action=RecoveryAction.REBOOT_NODE,
        action_owner="test",
        mapping_version="test-v1",
        raw_evidence_ref=f"s3://evidence/{marker_id}.json",
    )


def _aged_notification(
    store,
    *,
    age: timedelta,
    incident_id: str,
    category: str | None = None,
    not_before: datetime | None = None,
) -> AdvisoryNotification:
    update: dict = {"created_at": datetime.now(timezone.utc) - age}
    if category is not None:
        update["category"] = category
    if not_before is not None:
        update["not_before"] = not_before
    return store.save_notification_if_absent(
        HyperPodAdvisoryEmailBuilder()
        .build(
            advisory(),
            cluster_name="hp-cluster",
            incident_id=incident_id,
            node_ids=["worker-1"],
            issue_summary="queued while nothing was draining the outbox",
        )
        .model_copy(update=update)
    )


def _watermark_already_drawn(store) -> None:
    """Put the deployment past its one-and-only backlog suppression.

    The watermark fires the first time a dispatcher ever runs. Every gap
    after that -- a wedged dispatcher, SES denied, a worker Deployment
    scaled to zero -- accumulates an outbox that nothing suppresses, which
    is the case the shelf life exists for.
    """

    store.establish_notification_watermark(
        established_at=datetime.now(timezone.utc) - timedelta(days=30),
        established_by="pod-first-ever",
    )


class ThrottlingNotifier:
    """Answers the way SES does once the account send rate is exceeded."""

    def __init__(self, *, throttle_after: int = 0) -> None:
        self.throttle_after = throttle_after
        self.notifications = []

    def send(self, notification):
        if len(self.notifications) >= self.throttle_after:
            raise TooManyRequestsException(
                "An error occurred (TooManyRequestsException) when "
                "calling the SendEmail operation: Maximum sending rate "
                "exceeded."
            )
        self.notifications.append(notification)
        return NotificationResult(
            notification_id=notification.notification_id,
            status=NotificationStatus.SENT,
            provider_message_id=f"ses-{len(self.notifications)}",
        )


class TooManyRequestsException(Exception):
    """The botocore-generated name, which is all the classifier sees."""


class LeaseLosingStore(InMemoryStore):
    """Loses the delivery lease while the provider is being slow.

    Reproduces the shape seen in production: the send succeeds, and the
    write that records it lands after the lease has expired.
    """

    def __init__(self) -> None:
        super().__init__()
        self.completions = 0

    def complete_notification_delivery(self, *args, **kwargs):
        self.completions += 1
        raise WorkflowLeaseError("notification delivery lease is stale")


def _delivery(store, notification_id: str) -> NotificationDelivery | None:
    """Read one delivery row from either store implementation.

    There is no public single-row getter -- the dispatcher only ever sees
    deliveries through ``claim_notification_deliveries`` -- so the test
    reaches for whichever private accessor the implementation has.
    """

    if isinstance(store, SqliteStore):
        return store._get_optional("notification_delivery", notification_id)
    return store._notification_deliveries.get(notification_id)


def _drill_notification(
    store, *, incident_id: str, drill_id: str = "perf-burst"
) -> AdvisoryNotification:
    """A notification the store has labelled as belonging to a drill.

    The label is what the load suites now stamp into the synthetic Xid and
    SXid messages they inject, so this is the exact shape a pressure test
    produces.
    """

    store.save_incident(
        fault_incident(
            incident_id,
            f"event-{incident_id}",
            "TEST",
            "hp-cluster",
            node_ids=["worker-1"],
            policy_version="test",
            policy_source="test",
            drill_id=drill_id,
        )
    )
    return _aged_notification(store, age=timedelta(seconds=5), incident_id=incident_id)


def _delivery_mode(
    monkeypatch, *, email: bool, async_delivery: bool, dispatcher: bool
) -> str:
    monkeypatch.setenv(
        "GPU_FAULT_NOTIFICATION_DISPATCHER_ENABLED", "true" if dispatcher else "false"
    )
    notifier = SesEmailNotifier(
        SesNotificationConfig(
            sender="gpu@example.com",
            recipients=["admin@example.com"],
            execution_enabled=email,
        ),
        client=FakeSesV2Client(),
    )
    service = AdvisoryNotificationService(
        build_store(), notifier, async_delivery=async_delivery
    )
    return service.describe_delivery_mode()

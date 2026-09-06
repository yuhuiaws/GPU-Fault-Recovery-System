"""The outbox does not mail a notification the store already records as SENT.

FINAL-建议汇总 F-D11 (P0-38B, "发送幂等键"). ``send`` checks the recorded result
before calling the provider; ``dispatch_outbox`` did not -- it trusted the
claim query to have skipped SENT rows, but a result recorded between the
claim and the provider call (an inline ``send`` on a sibling process, or a
dispatcher whose lease lapsed after its mail went out) still reached SES a
second time. The notification id is the idempotency key; the recorded result
is the only shared truth about whether it was delivered.
"""

from __future__ import annotations

from gpu_fault.models import NotificationResult, NotificationStatus
from gpu_fault.notification_service import AdvisoryNotificationService
from gpu_fault.notifications import HyperPodAdvisoryEmailBuilder
from tests._builders import build_store
from tests.notifications._support import (
    RecordingNotifier,
    _watermark_already_drawn,
    advisory,
)


def _saved_notification(store):
    return store.save_notification_if_absent(
        HyperPodAdvisoryEmailBuilder().build(
            advisory(),
            cluster_name="hp-cluster",
            incident_id="incident-idempotent",
            node_ids=["worker-1"],
            issue_summary="delivered once, whatever path gets there first",
        )
    )


def test_notification_send_is_idempotent_by_key(monkeypatch) -> None:
    store = build_store()
    _watermark_already_drawn(store)
    notifier = RecordingNotifier()
    service = AdvisoryNotificationService(store, notifier, async_delivery=True)
    notification = _saved_notification(store)
    queued = service.send(notification.notification_id)
    assert queued.status is NotificationStatus.QUEUED

    claim = store.claim_notification_deliveries

    def claim_then_sibling_delivers(*args, **kwargs):
        deliveries = claim(*args, **kwargs)
        # Between our claim and our provider call a sibling recorded SENT.
        store.save_notification_result(
            NotificationResult(
                notification_id=notification.notification_id,
                status=NotificationStatus.SENT,
                provider_message_id="ses-sibling",
            )
        )
        return deliveries

    monkeypatch.setattr(
        store, "claim_notification_deliveries", claim_then_sibling_delivers
    )

    report = service.dispatch_outbox("pod-a")

    assert notifier.notifications == []
    assert report.attempted == 1
    assert report.sent == 1
    assert report.failed == 0
    recorded = store.get_notification_result(notification.notification_id)
    assert recorded is not None and recorded.status is NotificationStatus.SENT
    assert recorded.provider_message_id == "ses-sibling"
    # The delivery row is retired, so the next cycle does not claim it again.
    assert service.dispatch_outbox("pod-a").attempted == 0
    assert notifier.notifications == []

#!/usr/bin/env python3
"""In-Pod drill for GF-REGIONAL-NOTIFY-007: outbox delivery states.

Runs inside a control-worker Pod against an *isolated* ``InMemoryStore`` with
the Pod's real SES notifier (``notification_notifier_from_environment``), the
same way ``notification_drill.py`` does for NOTIFY-001/002 -- no production
row is written. Two labelled drill notifications are driven through the
shipped ``AdvisoryNotificationService.dispatch_outbox``:

* **retry** -- the first ``send`` is made to raise (a transient provider
  failure); the delivery row must go to RETRY with *no* result row, the
  Store's ``notification_delivery_stats`` (the data the ``/metrics`` gauges
  render) must count it under RETRY, and the second cycle must SEND it with a
  provider message id.
* **dead** -- every ``send`` raises and ``max_attempts=1``; the row must go
  DEAD, the result row must be the terminal FAILED verdict, and
  ``dead_lettered_total`` must count one.

The failing notifier wraps the real one, so the drill that does get sent is a
real, labelled DRILL email through SES. Output is one JSON object.
"""

from __future__ import annotations

import argparse
import json
import time
from typing import Any

from gpu_fault.models import AdvisoryNotification, NotificationStatus
from gpu_fault.notification_service import AdvisoryNotificationService
from gpu_fault.notifications import (
    RestartGuardEmailBuilder,
    notification_notifier_from_environment,
)
from gpu_fault.store import InMemoryStore

RETRY_BASE_SECONDS = 1


class TransientFailureNotifier:
    """Raises for the first ``failures`` sends, then delegates to the real one."""

    def __init__(self, inner: Any, *, failures: int) -> None:
        self.inner = inner
        self.remaining = failures
        self.attempts = 0

    def send(self, notification: AdvisoryNotification) -> Any:
        self.attempts += 1
        if self.remaining > 0:
            self.remaining -= 1
            raise RuntimeError("NOTIFY-007 acceptance: transient provider failure")
        return self.inner.send(notification)


class PermanentFailureNotifier:
    def __init__(self) -> None:
        self.attempts = 0

    def send(self, notification: AdvisoryNotification) -> Any:
        self.attempts += 1
        raise RuntimeError("NOTIFY-007 acceptance: permanent provider failure")


def build_drill(drill_id: str, cluster_id: str, *, label: str) -> AdvisoryNotification:
    notification = RestartGuardEmailBuilder().build_gpu_reset_completed(
        cluster_id=cluster_id,
        incident_id=f"incident-{drill_id}",
        workflow_id=f"workflow-{drill_id}",
        event_id=f"event-{drill_id}",
        event_type="NOTIFY_ACCEPTANCE_DRILL",
        policy_source="ACCEPTANCE_DRILL",
        official_action="RESET_GPU",
        reasons=[
            f"notification delivery-state drill ({label}); no GPU action occurred"
        ],
        operation_id=f"operation-{drill_id}",
        node_ids=[f"node-{drill_id}"],
        gpu_uuids=[f"GPU-{drill_id}"],
        node_results={
            f"node-{drill_id}": {
                "status": "SUCCEEDED",
                "reset_gpu_uuids": [f"GPU-{drill_id}"],
            }
        },
        workload_ids=[],
    )
    return notification.model_copy(
        update={
            "drill_id": f"{drill_id}-{label}",
            "deduplication_key": f"{notification.deduplication_key}/{label}",
            "subject": f"[DRILL:{drill_id}:{label}] {notification.subject}",
            "body_text": (
                "DRILL / 验收演练：未执行 GPU reset、任务重启或节点变更。\n\n"
                + notification.body_text
            ),
        }
    )


def _result(store: InMemoryStore, notification_id: str) -> dict[str, Any] | None:
    result = store.get_notification_result(notification_id)
    if result is None:
        return None
    return {
        "status": result.status.value,
        "provider_message_id_present": bool(result.provider_message_id),
        "reason": result.reason,
    }


def run_retry_phase(cluster_id: str, drill_id: str) -> dict[str, Any]:
    store = InMemoryStore()
    notifier = TransientFailureNotifier(
        notification_notifier_from_environment(), failures=1
    )
    service = AdvisoryNotificationService(
        store, notifier, async_delivery=True, deliver_drills=True
    )
    notification = store.save_notification_if_absent(
        build_drill(drill_id, cluster_id, label="retry")
    )
    store.enqueue_notification_delivery(notification.notification_id)
    first = service.dispatch_outbox(
        "notify007-retry-a", retry_base_seconds=RETRY_BASE_SECONDS, max_attempts=8
    )
    after_first = {
        "report": {
            "attempted": first.attempted,
            "sent": first.sent,
            "failed": first.failed,
        },
        "stats": store.notification_delivery_stats(),
        "result": _result(store, notification.notification_id),
        "dead_lettered_total": service.dead_lettered_total,
    }
    time.sleep(RETRY_BASE_SECONDS + 0.5)
    second = service.dispatch_outbox(
        "notify007-retry-b", retry_base_seconds=RETRY_BASE_SECONDS, max_attempts=8
    )
    after_second = {
        "report": {
            "attempted": second.attempted,
            "sent": second.sent,
            "failed": second.failed,
        },
        "stats": store.notification_delivery_stats(),
        "result": _result(store, notification.notification_id),
        "dead_lettered_total": service.dead_lettered_total,
    }
    return {
        "notification_id": notification.notification_id,
        "notifier_attempts": notifier.attempts,
        "after_first_cycle": after_first,
        "after_second_cycle": after_second,
        "last_cycle_timestamp_seconds": service.last_cycle_timestamp_seconds,
    }


def run_dead_phase(cluster_id: str, drill_id: str) -> dict[str, Any]:
    store = InMemoryStore()
    notifier = PermanentFailureNotifier()
    service = AdvisoryNotificationService(
        store, notifier, async_delivery=True, deliver_drills=True
    )
    notification = store.save_notification_if_absent(
        build_drill(drill_id, cluster_id, label="dead")
    )
    store.enqueue_notification_delivery(notification.notification_id)
    report = service.dispatch_outbox(
        "notify007-dead", retry_base_seconds=RETRY_BASE_SECONDS, max_attempts=1
    )
    return {
        "notification_id": notification.notification_id,
        "notifier_attempts": notifier.attempts,
        "report": {
            "attempted": report.attempted,
            "sent": report.sent,
            "failed": report.failed,
        },
        "stats": store.notification_delivery_stats(),
        "result": _result(store, notification.notification_id),
        "dead_lettered_total": service.dead_lettered_total,
        "terminal_status_value": NotificationStatus.FAILED.value,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--drill-id", required=True)
    parser.add_argument("--cluster-id", required=True)
    arguments = parser.parse_args()
    print(
        json.dumps(
            {
                "drill_id": arguments.drill_id,
                "retry": run_retry_phase(arguments.cluster_id, arguments.drill_id),
                "dead": run_dead_phase(arguments.cluster_id, arguments.drill_id),
            },
            sort_keys=True,
            default=str,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

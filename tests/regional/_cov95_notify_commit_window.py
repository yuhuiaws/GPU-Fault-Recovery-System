"""Backend-neutral notification crash contract; PostgreSQL runs belong to the parent."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any, Literal

import pytest

from gpu_fault import notification_service
from gpu_fault.models import (
    AdvisoryNotification,
    NotificationResult,
    NotificationStatus,
)
from gpu_fault.notification_service import AdvisoryNotificationService
from scripts.e2e.regional.probes.notification_drill import (
    DrillNotifier,
    build_notification,
)

CrashPoint = Literal[
    "before-provider", "accepted-before-commit", "committed-before-ack"
]


class ProcessLost(BaseException):
    """An in-process kill simulation that cannot enter ordinary exception recovery."""


class AcceptanceLog:
    def __init__(self, crash: CrashPoint) -> None:
        self.crash = crash
        self.calls = 0
        self.accepted: list[str] = []


class Provider:
    def __init__(self, ledger: AcceptanceLog) -> None:
        self.ledger = ledger

    def send(self, notification: AdvisoryNotification) -> NotificationResult:
        self.ledger.calls += 1
        if self.ledger.calls == 1 and self.ledger.crash == "before-provider":
            raise ProcessLost("unit loss before provider acceptance")
        message = f"unit-message-{len(self.ledger.accepted) + 1}"
        self.ledger.accepted.append(message)
        return NotificationResult(
            notification_id=notification.notification_id,
            status=NotificationStatus.SENT,
            provider_message_id=message,
        )


class CommitBoundary:
    """Delegate every Store operation, interrupting only the real result commit."""

    def __init__(self, store: Any, crash: CrashPoint) -> None:
        self.store, self.crash = store, crash

    def __getattr__(self, name: str) -> Any:
        return getattr(self.store, name)

    def write(self, method: str, *args: Any, **kwargs: Any) -> Any:
        if self.crash == "accepted-before-commit":
            raise ProcessLost("unit loss after acceptance, before commit")
        result = getattr(self.store, method)(*args, **kwargs)
        if self.crash == "committed-before-ack":
            raise ProcessLost("unit loss after commit, before acknowledgement")
        return result

    def save_notification_result(self, *args: Any, **kwargs: Any) -> Any:
        return self.write("save_notification_result", *args, **kwargs)

    def complete_notification_delivery(self, *args: Any, **kwargs: Any) -> Any:
        return self.write("complete_notification_delivery", *args, **kwargs)


def exercise_provider_commit_window(
    store: Any,
    monkeypatch: pytest.MonkeyPatch,
    *,
    asynchronous: bool,
    crash: CrashPoint,
    kind: str,
) -> dict[str, Any]:
    """Run unchanged assertions on a fresh real Store, without replacing its SQL."""
    drill_id = f"unit-{kind}-{crash}"
    notification = build_notification(kind, drill_id, "unit-cluster")
    store.save_notification_if_absent(notification)
    accepted = AcceptanceLog(crash)
    moment = datetime.now(timezone.utc)

    def now(tz: Any = None) -> datetime:
        return moment

    def service(backend: Any) -> AdvisoryNotificationService:
        return AdvisoryNotificationService(
            backend,
            DrillNotifier(Provider(accepted), drill_id),
            async_delivery=asynchronous,
            deliver_drills=True,
        )

    def dispatch(instance: AdvisoryNotificationService, owner: str) -> None:
        if asynchronous:
            instance.dispatch_outbox(owner, limit=1, lease_seconds=30)
        else:
            instance.send(notification.notification_id)

    with monkeypatch.context() as local:
        local.setattr(notification_service, "datetime", SimpleNamespace(now=now))
        first = service(CommitBoundary(store, crash))
        with pytest.raises(ProcessLost):
            dispatch(first, "unit-first-process")
        recorded = store.get_notification_result(notification.notification_id)
        if crash == "committed-before-ack":
            assert recorded is not None, "a committed result must survive caller loss"
            assert recorded.status is NotificationStatus.SENT
        else:
            assert recorded is None, (
                "an uncommitted provider outcome is not durable proof"
            )

        replacement = service(store)
        before_retry = len(accepted.accepted)
        if asynchronous:
            moment += timedelta(seconds=1)
            dispatch(replacement, "unit-replacement-process")
            assert len(accepted.accepted) == before_retry, (
                "a replacement must not steal an unexpired delivery lease"
            )
            moment += timedelta(seconds=31)
        dispatch(replacement, "unit-replacement-process")
        final = store.get_notification_result(notification.notification_id)
        assert final is not None, "recovery must record the final accepted result"
        assert final.status is NotificationStatus.SENT
        expected = 2 if crash == "accepted-before-commit" else 1
        assert len(accepted.accepted) == expected, (
            "acceptance before durable commit can be delivered twice without provider dedup"
        )
        dispatch(replacement, "unit-replacement-process")
        assert len(accepted.accepted) == expected, "post-commit replay must not resend"
        assert len(store.list_notifications()) == 1, (
            "one durable notification identity is retained"
        )
    return {
        "crash_point": crash,
        "asynchronous": asynchronous,
        "provider_acceptances": len(accepted.accepted),
        "provider_calls": accepted.calls,
        "duplicate_delivery_possible": crash == "accepted-before-commit",
        "final_status": final.status.value,
    }

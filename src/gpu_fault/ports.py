from __future__ import annotations

from typing import Protocol

from gpu_fault.models import AdvisoryNotification, NotificationResult


class NotificationPort(Protocol):
    def send(self, notification: AdvisoryNotification) -> NotificationResult:
        """Deliver one idempotent operator notification."""

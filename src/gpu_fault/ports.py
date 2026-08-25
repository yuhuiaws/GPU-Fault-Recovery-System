from __future__ import annotations

from typing import Protocol

from gpu_fault.models import (
    AdvisoryNotification,
    DiagnosticRequest,
    NotificationResult,
    OperationResult,
    RecoveryPlan,
)


class DiagnosticPort(Protocol):
    def submit(self, request: DiagnosticRequest) -> str:
        """Submit a quick triage request and return its operation ID."""


class RecoveryExecutorPort(Protocol):
    def execute(self, plan: RecoveryPlan) -> OperationResult:
        """Execute a recovery plan through a runtime-specific adapter."""


class NotificationPort(Protocol):
    def send(self, notification: AdvisoryNotification) -> NotificationResult:
        """Deliver one idempotent operator notification."""

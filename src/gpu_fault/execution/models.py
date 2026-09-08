from __future__ import annotations

import logging
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from gpu_fault.models import (
    FaultIncident,
    WorkflowExecutionRequest,
    WorkflowRequest,
    WorkflowStepSpec,
    WorkflowStepStatus,
)

LOGGER = logging.getLogger(__name__)


class WorkflowExecutionError(ValueError):
    pass


class WorkflowStructureError(WorkflowExecutionError):
    """The record's own shape cannot be executed: a DAG with a cycle, a
    dependency on a step that does not exist, more steps than one workflow may
    carry. Deterministic, so retrying it is a hot loop; the dispatcher writes
    the record BLOCKED for an operator instead (control-plane review
    2026-09-08, D-6)."""


class WorkflowRecordInvalidError(WorkflowExecutionError):
    """The workflow row itself could not be decoded (a ``ValidationError`` from
    ``get_workflow`` of the record being executed). The one decode failure that
    proves *this record* cannot run; a decode failure of any other row -- a
    plan, a remote command, an incident -- says nothing about it (control-plane
    review 2026-09-08, D-5)."""


def _failure_details(adapter: Any, exc: BaseException) -> dict[str, Any]:
    """Locate an unexpected adapter failure without leaking a traceback.

    A bare ``AttributeError`` says nothing about which adapter helper
    raised it, so record the innermost gpu_fault frame and the chained
    cause alongside the class name.
    """
    frames = traceback.extract_tb(exc.__traceback__)
    owned = [frame for frame in frames if "gpu_fault" in (frame.filename or "")]
    origin = (owned or frames)[-1] if frames else None
    details: dict[str, Any] = {
        "error_kind": type(exc).__name__,
        "error_adapter": type(adapter).__name__,
    }
    if origin is not None:
        details["error_site"] = (
            f"{Path(origin.filename).name}:{origin.lineno} in {origin.name}"
        )
    cause = exc.__cause__ or exc.__context__
    if cause is not None:
        details["error_cause"] = f"{type(cause).__name__}: {cause}"[:300]
    return details


@dataclass(frozen=True)
class WorkflowStepContext:
    workflow: WorkflowRequest
    incident: FaultIncident
    step: WorkflowStepSpec
    step_index: int
    request: WorkflowExecutionRequest
    idempotency_key: str


@dataclass(frozen=True)
class WorkflowStepOutcome:
    status: WorkflowStepStatus
    adapter_operation_id: str | None = None
    error: str | None = None
    details: dict[str, Any] | None = None

    @classmethod
    def succeeded(
        cls,
        *,
        operation_id: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> WorkflowStepOutcome:
        return cls(
            status=WorkflowStepStatus.SUCCEEDED,
            adapter_operation_id=operation_id,
            details=details,
        )

    @classmethod
    def waiting(
        cls,
        *,
        operation_id: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> WorkflowStepOutcome:
        return cls(
            status=WorkflowStepStatus.WAITING,
            adapter_operation_id=operation_id,
            details=details,
        )

    @classmethod
    def failed(
        cls,
        error: str,
        *,
        details: dict[str, Any] | None = None,
    ) -> WorkflowStepOutcome:
        return cls(
            status=WorkflowStepStatus.FAILED,
            error=error,
            details=details,
        )


class WorkflowStepAdapter(Protocol):
    def supports(self, step: WorkflowStepSpec) -> bool:
        """Return true only for explicitly owned operations."""

    def execute(self, context: WorkflowStepContext) -> WorkflowStepOutcome:
        """Execute or observe one idempotent workflow step."""

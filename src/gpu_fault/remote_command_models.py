from __future__ import annotations

from datetime import datetime, timedelta, timezone
from enum import StrEnum
from typing import Any

from pydantic import Field, model_validator

from gpu_fault.models import StrictModel, WorkflowStepSpec


class RemoteCommandStatus(StrEnum):
    PENDING = "PENDING"
    LEASED = "LEASED"
    WAITING = "WAITING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"


def lease_deadline(seconds: int) -> datetime:
    return datetime.now(timezone.utc) + timedelta(seconds=seconds)


# ``result_details`` key under which a compound command keeps one entry per
# covered step (``{"<step_index>": {"status", "details", "error"}}``). Lives
# here rather than in ``regional`` so the store mixins that merge progress into
# the row do not import the adapter module.
BATCHED_RESULTS_KEY = "batched_results"


class BatchedStep(StrictModel):
    """One step that rides along with the head step of a compound command.

    The executor rebuilds the exact ``WorkflowStepContext`` this step would have
    had as a command of its own, so it needs the step's own index and
    idempotency key, not just its spec.
    """

    step_index: int = Field(ge=0)
    step: WorkflowStepSpec
    idempotency_key: str = Field(min_length=1)


class BatchedStepResult(StrictModel):
    """The executor's verdict on one covered step, as reported on the progress
    route and kept under ``BATCHED_RESULTS_KEY``."""

    status: RemoteCommandStatus
    status_source: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None

    @model_validator(mode="after")
    def validate_reportable_status(self) -> BatchedStepResult:
        if self.status not in {
            RemoteCommandStatus.WAITING,
            RemoteCommandStatus.SUCCEEDED,
            RemoteCommandStatus.FAILED,
        }:
            raise ValueError(
                "batched step result must be WAITING, SUCCEEDED, or FAILED"
            )
        if self.status is RemoteCommandStatus.FAILED and not self.error:
            raise ValueError("failed batched step result requires error")
        return self

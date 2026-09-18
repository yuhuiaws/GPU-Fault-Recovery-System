"""Reset-result compatibility shared by single-node and barrier execution."""

from __future__ import annotations

from gpu_fault.models import WorkflowOperation
from gpu_fault.node_agent.protocol import NodeActionResult, NodeActionStatus
from gpu_fault.orchestration.escalation import unknown_outcome_failure


def normalize_legacy_reset_failure(result: NodeActionResult) -> NodeActionResult:
    """Older compatible Agents encode uncertainty in progress or an exception tag."""

    if (
        result.status is not NodeActionStatus.FAILED
        or result.operation
        not in {
            WorkflowOperation.RESET_GPU,
            WorkflowOperation.RESET_ALL_GPUS_NVSWITCHES,
        }
        or unknown_outcome_failure(result.details)
    ):
        return result
    raw = result.details.get("reset_outcome_unknown", [])
    invalid = not isinstance(raw, list) or any(
        not isinstance(value, str) or not value for value in raw
    )
    # NodeActionExecutor frames errors as "<exception class>: <message>".
    # Only this exact reset-specific legacy tag is recognized, never message text.
    exception_tag, separator, _ = (result.error or "").partition(": ")
    legacy_exception = bool(separator) and exception_tag == "ResetOutcomeUnknown"
    if not invalid and not raw and not legacy_exception:
        return result
    return result.model_copy(
        update={
            "retryable": False,
            "details": {
                **result.details,
                "outcome_unknown": True,
                "manual_confirmation_required": True,
                "reset_outcome_compatibility": (
                    "invalid-progress"
                    if invalid
                    else "per-gpu-progress"
                    if raw
                    else "legacy-exception-tag"
                ),
            },
        }
    )

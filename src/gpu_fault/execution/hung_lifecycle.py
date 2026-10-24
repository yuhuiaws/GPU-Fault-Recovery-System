"""Stop new hung evidence collection when ordinary completion handling takes over."""

from __future__ import annotations

from typing import Any

from gpu_fault.execution.models import WorkflowStepContext, WorkflowStepOutcome
from gpu_fault.hung_lifecycle import cleanup_reason
from gpu_fault.models import WorkflowOperation as Op


def skip_inactive_hung(
    store: Any, context: WorkflowStepContext
) -> WorkflowStepOutcome | None:
    step = context.step
    if step.operation not in {
        Op.COLLECT_HUNG_SUMMARY,
        Op.COLLECT_HUNG_TRIAGE,
        Op.COLLECT_DIAGNOSTIC_BUNDLE,
        Op.VALIDATE_FABRIC,
    }:
        return None
    parameters = step.parameters
    reason: str | None
    if parameters.get("diagnostic_reason") == "APPLICATION_NCCL_ERROR":
        reason = "APPLICATION_ERROR_OWNED_BY_WATCHER"
    elif (
        parameters.get("hung_diagnostic_node_budget") is not None
        or (parameters.get("health_recovery_contract") or {}).get("mode")
        == "HUNG_DIAGNOSTIC"
    ):
        job = parameters.get("job_id") or context.incident.job_id
        attempt = parameters.get("attempt_id") or context.incident.attempt_id
        if not job or not attempt:
            return None
        observations = [
            item
            for item in store.list_attempt_observations(context.incident.cluster_id)
            if item.job_id == job and item.attempt_id == attempt
        ]
        if len(observations) != 1:
            return None
        reason = cleanup_reason(observations[0], step.node_ids)
    else:
        return None
    if reason is None:
        return None
    return WorkflowStepOutcome.succeeded(
        details={
            "skipped": reason,
            "recovery_owner": "COMPLETION_WATCHER",
            "node_results": {},
            "hung_evidence_complete": False,
        }
    )

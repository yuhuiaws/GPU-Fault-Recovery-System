from __future__ import annotations

from datetime import datetime
from typing import cast

from gpu_fault.models import WorkflowRequest, WorkflowStatus
from gpu_fault.store.shared.errors import RemediationBudgetError


def apply_remediation_budget(
    workflow: WorkflowRequest,
    active_workflows: list[WorkflowRequest],
    claims: dict[str, int],
    *,
    now: datetime,
) -> WorkflowRequest:
    normalized = {
        str(scope): int(limit)
        for scope, limit in claims.items()
        if str(scope) and int(limit) > 0
    }
    for scope, limit in sorted(normalized.items()):
        active = sum(
            1
            for candidate in active_workflows
            if candidate.request_id != workflow.request_id
            and candidate.status is WorkflowStatus.RUNNING
            and candidate.execution_lease_expires_at is not None
            and candidate.execution_lease_expires_at > now
            and scope in candidate.remediation_budget_claims
        )
        if active < limit:
            continue
        raise RemediationBudgetError(
            "remediation concurrency budget is full: "
            f"scope={scope} active={active} limit={limit}"
        )
    return cast(
        WorkflowRequest,
        workflow.model_copy(
            update={
                "remediation_budget_claims": sorted(normalized),
                "remediation_budget_limits": normalized,
                "remediation_budget_last_blocked_reason": None,
            }
        ),
    )


def blocked_by_remediation_budget(
    workflow: WorkflowRequest,
    reason: str,
    *,
    now: datetime,
) -> WorkflowRequest:
    return cast(
        WorkflowRequest,
        workflow.model_copy(
            update={
                "execution_owner_id": None,
                "execution_lease_expires_at": None,
                "remediation_budget_wait_count": (
                    workflow.remediation_budget_wait_count + 1
                ),
                "remediation_budget_last_blocked_reason": reason,
                "updated_at": now,
            }
        ),
    )

from __future__ import annotations

from datetime import datetime

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
            f"scope={scope} active={active} limit={limit}",
            scope=scope,
        )
    return workflow.model_copy(
        update={
            "remediation_budget_claims": sorted(normalized),
            "remediation_budget_limits": normalized,
            "remediation_budget_last_blocked_reason": None,
            "remediation_budget_last_blocked_scope": None,
        }
    )


def extend_remediation_budget(
    workflow: WorkflowRequest,
    active_workflows: list[WorkflowRequest],
    claims: dict[str, int],
    *,
    now: datetime,
) -> WorkflowRequest:
    """Add ``claims`` to a leased workflow's held budget, or raise.

    Same counting rule as :func:`apply_remediation_budget`; the scopes the
    workflow already holds stay held, so a refusal changes nothing.
    """
    extended = apply_remediation_budget(workflow, active_workflows, claims, now=now)
    return workflow.model_copy(
        update={
            "remediation_budget_claims": sorted(
                set(workflow.remediation_budget_claims)
                | set(extended.remediation_budget_claims)
            ),
            "remediation_budget_limits": {
                **workflow.remediation_budget_limits,
                **extended.remediation_budget_limits,
            },
        }
    )


def blocked_by_remediation_budget(
    workflow: WorkflowRequest,
    reason: str,
    *,
    scope: str | None = None,
    now: datetime,
) -> WorkflowRequest:
    """Record a refused claim on the workflow.

    ``scope`` is the refusing budget scope (``RemediationBudgetError.scope``);
    an empty string means the caller did not know it and is stored as None.
    """
    return workflow.model_copy(
        update={
            "execution_owner_id": None,
            "execution_lease_expires_at": None,
            "remediation_budget_wait_count": (
                workflow.remediation_budget_wait_count + 1
            ),
            "remediation_budget_last_blocked_reason": reason,
            "remediation_budget_last_blocked_scope": scope or None,
            "updated_at": now,
        }
    )

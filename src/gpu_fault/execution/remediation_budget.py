from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from gpu_fault.models import (
    FaultIncident,
    WorkflowOperation,
    WorkflowRequest,
    WorkflowStepSpec,
)
from gpu_fault.operation_registry import (
    NODE_MUTATING_OPERATIONS,
    OPERATION_REGISTRY,
    OPERATION_RESOURCE_CLAIMS,
    OperationScope,
)


@dataclass(frozen=True)
class RemediationBudgetPolicy:
    region_limit: int = 20
    cluster_limit: int = 5
    node_limit: int = 1
    failure_domain_limit: int = 1
    resource_class_limit: int = 2

    @classmethod
    def from_mapping(cls, values: Mapping[str, str]) -> RemediationBudgetPolicy:
        policy = cls(
            region_limit=int(
                values.get("GPU_FAULT_REMEDIATION_MAX_ACTIVE_REGION", "20")
            ),
            cluster_limit=int(
                values.get("GPU_FAULT_REMEDIATION_MAX_ACTIVE_PER_CLUSTER", "5")
            ),
            node_limit=int(
                values.get("GPU_FAULT_REMEDIATION_MAX_ACTIVE_PER_NODE", "1")
            ),
            failure_domain_limit=int(
                values.get(
                    "GPU_FAULT_REMEDIATION_MAX_ACTIVE_PER_FAILURE_DOMAIN",
                    "1",
                )
            ),
            resource_class_limit=int(
                values.get(
                    "GPU_FAULT_REMEDIATION_MAX_ACTIVE_PER_RESOURCE_CLASS",
                    "2",
                )
            ),
        )
        if (
            min(
                policy.region_limit,
                policy.cluster_limit,
                policy.node_limit,
                policy.failure_domain_limit,
                policy.resource_class_limit,
            )
            < 1
        ):
            raise ValueError("remediation concurrency limits must be positive")
        return policy


def remediation_budget_claims(
    policy: RemediationBudgetPolicy,
    workflow: WorkflowRequest,
    incident: FaultIncident,
    steps: Sequence[WorkflowStepSpec],
) -> dict[str, int]:
    completed = set(workflow.completed_step_indexes)
    pending = [
        step
        for index, step in enumerate(steps)
        if index not in completed and _budgeted(step.operation)
    ]
    if not pending or workflow.pending_failure_step_index is not None:
        return {}
    if any(
        index in completed and _budgeted(step.operation)
        for index, step in enumerate(steps)
    ):
        return {}

    return _scopes_for_steps(policy, incident.cluster_id, pending)


def escalation_budget_claims(
    policy: RemediationBudgetPolicy,
    workflow: WorkflowRequest,
    incident: FaultIncident,
    steps: Sequence[WorkflowStepSpec],
) -> dict[str, int]:
    """Scopes an in-place branch escalation adds to a workflow's held budget.

    The workflow claimed its concurrency budget for the plan it was born
    with. A rung appended later (RESET_GPU -> RESTART_NODE -> REPLACE_NODE)
    may start a resource class or touch a failure domain that budget never
    counted; those scopes -- and only those -- have to be taken before the
    rung runs (F-N1 per-branch settlement).
    """

    mutating = [step for step in steps if _budgeted(step.operation)]
    if not mutating:
        return {}
    held = set(workflow.remediation_budget_claims)
    return {
        scope: limit
        for scope, limit in _scopes_for_steps(
            policy, incident.cluster_id, mutating
        ).items()
        if scope not in held
    }


# Isolation is budgeted like a mutation: it is containment-only for the
# scheduler, but a false-positive cascade must not quarantine nodes without
# limit (F-C5). Cordon and release alone stay free.
_ISOLATION_RESOURCE_CLASS = "NODE_ISOLATION"


def _budgeted(operation: WorkflowOperation) -> bool:
    if operation is WorkflowOperation.QUARANTINE:
        return True
    return (
        operation in NODE_MUTATING_OPERATIONS
        and operation is not WorkflowOperation.RESTORE_GPU_SERVICES
    )


def _scopes_for_steps(
    policy: RemediationBudgetPolicy,
    cluster_id: str,
    pending: Sequence[WorkflowStepSpec],
) -> dict[str, int]:
    claims = {
        "region": policy.region_limit,
        f"cluster:{cluster_id}": policy.cluster_limit,
    }
    for node_id in sorted(
        {node_id for step in pending for node_id in step.node_ids if node_id}
    ):
        claims[f"node:{cluster_id}:{node_id}"] = policy.node_limit

    failure_domains: set[str] = set()
    resource_classes: set[str] = set()
    for step in pending:
        for key in ("fabric_partition", "failure_domain", "availability_zone"):
            raw = step.parameters.get(key)
            values = raw if isinstance(raw, list) else [raw]
            failure_domains.update(str(value) for value in values if value)
        if step.operation is WorkflowOperation.QUARANTINE:
            resource_classes.add(_ISOLATION_RESOURCE_CLASS)
            continue
        resource_classes.update(OPERATION_RESOURCE_CLAIMS[step.operation])
        if not OPERATION_RESOURCE_CLAIMS[step.operation]:
            scope = OPERATION_REGISTRY[step.operation].scope
            if scope is OperationScope.WORKLOAD:
                resource_classes.add("WORKLOAD_MUTATION")
            elif scope is OperationScope.SUPPORT:
                resource_classes.add("SUPPORT_ACTION")
            else:
                resource_classes.add(f"{scope.value}_MUTATION")

    for domain in sorted(failure_domains):
        claims[f"domain:{cluster_id}:{domain}"] = policy.failure_domain_limit
    for resource_class in sorted(resource_classes):
        claims[f"class:{cluster_id}:{resource_class}"] = policy.resource_class_limit
    return claims

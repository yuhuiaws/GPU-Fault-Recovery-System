from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from gpu_fault.models import (
    FaultIncident,
    WorkflowOperation,
    WorkflowRequest,
    WorkflowStepSpec,
    resolved_step_indexes,
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


def budget_work_begun(
    workflow: WorkflowRequest, steps: Sequence[WorkflowStepSpec]
) -> bool:
    """Whether the workflow's own budgeted work has begun: its claim is a hold.

    Begun means a parked failure owes its compensation, or one of its own
    budgeted steps completed or has an execution record of any status (an
    attempted reset or reboot may still be running). Containment a preempting
    successor inherited is its predecessor's work: the successor is admitted,
    taking over the hold its predecessor reserved for it.
    """

    if workflow.pending_failure_step_index is not None:
        return True
    inherited = set(workflow.inherited_step_indexes)
    completed = set(workflow.completed_step_indexes) - inherited
    if any(
        index in completed and _budgeted(step.operation)
        for index, step in enumerate(steps)
    ):
        return True
    return any(
        execution.step_index not in inherited and _budgeted(execution.operation)
        for execution in workflow.step_executions
    )


def remediation_budget_claims(
    policy: RemediationBudgetPolicy,
    workflow: WorkflowRequest,
    incident: FaultIncident,
    steps: Sequence[WorkflowStepSpec],
) -> dict[str, int]:
    """The scopes the workflow's next claim carries (design §9.5).

    Until its budgeted work begins: the budgeted steps still to run, which the
    store admits or refuses. From then on (``budget_work_begun``): the scopes
    of its own budgeted plan, which the store adds to what it already holds
    and never refuses, so a reset, reboot or compensation in flight is never
    stranded and a record whose claims an older release dropped counts again.
    The hold ends when the workflow does.
    """

    return claims_in_cluster(policy, workflow, incident.cluster_id, steps)


def claims_in_cluster(
    policy: RemediationBudgetPolicy,
    workflow: WorkflowRequest,
    cluster_id: str,
    steps: Sequence[WorkflowStepSpec],
) -> dict[str, int]:
    """``remediation_budget_claims`` for a workflow of ``cluster_id``: what the
    dispatcher's guard asks of a row whose incident it knows only by cluster."""

    if budget_work_begun(workflow, steps):
        skipped = set(workflow.superseded_step_indexes) | set(
            workflow.inherited_step_indexes
        )
    else:
        skipped = set(resolved_step_indexes(workflow))
    budgeted = [
        step
        for index, step in enumerate(steps)
        if index not in skipped and _budgeted(step.operation)
    ]
    if not budgeted:
        return {}
    return _scopes_for_steps(policy, cluster_id, budgeted)


def scope_limit(policy: RemediationBudgetPolicy, scope: str) -> int | None:
    """The configured limit of one claimed scope, ``None`` for an unknown kind."""

    kind = scope.split(":", 1)[0]
    return {
        "region": policy.region_limit,
        "cluster": policy.cluster_limit,
        "node": policy.node_limit,
        "job": policy.failure_domain_limit,
        "domain": policy.failure_domain_limit,
        "class": policy.resource_class_limit,
    }.get(kind)


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
# The operations that act on the whole job -- its workload objects
# (``workload_ids``, one per job: ``<ns>/pytorchjob/<name>``), or the restart
# authorization's ``job_id`` -- whatever ``node_ids`` they were compiled with.
WHOLE_JOB_OPERATIONS = frozenset(
    {WorkflowOperation.STOP_WORKLOADS, WorkflowOperation.RESTART_WORKLOAD}
)


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
    # A training job, or a GPU node without one, is one workflow: its faults
    # merge into it and its nodes and GPUs are repaired in parallel inside it,
    # and `node:` keeps a second workflow off any node it touches. That alone
    # does not keep two workflows off one job: a STOP or RESTART compiled for
    # the faulted node only (an event no grouping attached, a health finding,
    # an escalation, a marker plan, ranks that already exited) still stops or
    # restarts every rank. So those two take the job's token as well.
    for token in sorted(_job_tokens(pending)):
        claims[f"job:{cluster_id}:{token}"] = policy.failure_domain_limit

    # The one failure domain left is physical: the NVSwitch fabric partition
    # the SXID planner writes into the step, on every fabric the product grades
    # the node's own baseboard. Node labels (instance group, node group, zone)
    # are no domain: read as one, they put every GPU node of a group in one
    # slot, and two jobs or two idle nodes of a node group were repaired one at
    # a time. Nor is any step key read "in case something sets it":
    # `failure_domain` and `availability_zone` once were, while nothing wrote
    # them.
    failure_domains: set[str] = set()
    resource_classes: set[str] = set()
    for step in pending:
        raw = step.parameters.get("fabric_partition")
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


def _job_tokens(pending: Sequence[WorkflowStepSpec]) -> set[str]:
    """The jobs ``pending``'s whole-job steps act on: workload ids, and the
    ``job_id`` a RESTART's authorization is reserved under."""

    tokens: set[str] = set()
    for step in pending:
        if step.operation not in WHOLE_JOB_OPERATIONS:
            continue
        tokens.update(workload for workload in step.workload_ids if workload)
        job_id = step.parameters.get("job_id")
        if step.operation is WorkflowOperation.RESTART_WORKLOAD and job_id:
            tokens.add(f"job-id/{job_id}")
    return tokens

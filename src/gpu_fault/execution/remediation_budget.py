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
    WORKLOAD_MUTATION_OPERATIONS,
)


@dataclass(frozen=True)
class RemediationBudgetPolicy:
    """The two slot limits a claim carries, both 1 as shipped (design §9.5).

    No fixed number caps how many remediations a cluster or the region runs at
    once: a correlated fault is what the cluster's remediation storm breaker
    (``gpu_fault.remediation_storm``) bounds. The region, per-cluster and
    per-resource-class limits a release before it read are retired: their
    variables are ignored (``env_validation.RETIRED_ENVIRONMENT``) and a scope
    of those kinds an older release left on a RUNNING row is no kind this
    policy knows (``scope_limit``), so it neither admits nor refuses anything.
    """

    node_limit: int = 1
    failure_domain_limit: int = 1

    @classmethod
    def from_mapping(cls, values: Mapping[str, str]) -> RemediationBudgetPolicy:
        policy = cls(
            node_limit=int(
                values.get("GPU_FAULT_REMEDIATION_MAX_ACTIVE_PER_NODE", "1")
            ),
            failure_domain_limit=int(
                values.get(
                    "GPU_FAULT_REMEDIATION_MAX_ACTIVE_PER_FAILURE_DOMAIN",
                    "1",
                )
            ),
        )
        if min(policy.node_limit, policy.failure_domain_limit) < 1:
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

    budgeted = _claimed_budgeted_steps(workflow, steps)
    if not budgeted:
        return {}
    return _scopes_for_steps(policy, cluster_id, budgeted)


def _claimed_budgeted_steps(
    workflow: WorkflowRequest, steps: Sequence[WorkflowStepSpec]
) -> list[WorkflowStepSpec]:
    """The budgeted steps this claim carries -- the single selection that both
    ``claims_in_cluster`` turns into scopes and ``storm_counted_node_ids``
    tallies nodes from. Once the workflow's budgeted work has begun: all but
    the superseded and inherited; before then: those still to run."""

    if budget_work_begun(workflow, steps):
        skipped = set(workflow.superseded_step_indexes) | set(
            workflow.inherited_step_indexes
        )
    else:
        skipped = set(resolved_step_indexes(workflow))
    return [
        step
        for index, step in enumerate(steps)
        if index not in skipped and _budgeted(step.operation)
    ]


def storm_counted_node_ids(
    workflow: WorkflowRequest, steps: Sequence[WorkflowStepSpec]
) -> frozenset[str]:
    """The nodes a claim brings into destructive NODE remediation, which the
    remediation storm breaker counts (design §9.5, ``gpu_fault.remediation_storm``).

    A node is counted when one of this claim's budgeted steps gives it a
    node-hardware mutation (``NODE_HARDWARE_MUTATING_OPERATIONS`` -- reset,
    reboot, replace, driver/firmware, fabric-manager, device-plugin, quiesce)
    or an isolation (``QUARANTINE``, a containment op the breaker counts too so
    a false-positive cascade is bounded, F-C5). It is NOT counted when its only
    budgeted steps on it are ``STOP_WORKLOADS`` / ``RESTART_WORKLOAD``: those
    stop or restart the ranks of a job the node belongs to, and the job family
    widens them across the job's whole allocation (``WHOLE_JOB_OPERATIONS``), so
    counting them would count a faulted node's healthy job peers -- serializing
    two small jobs on one cluster and false-tripping at scale (b223). The
    per-node budget exclusion still claims ``node:`` for every such node
    (``claims_in_cluster``); this governs only the storm tally.
    """

    return frozenset(
        node_id
        for step in _claimed_budgeted_steps(workflow, steps)
        if step.operation not in WORKLOAD_MUTATION_OPERATIONS
        for node_id in step.node_ids
        if node_id
    )


def scope_limit(policy: RemediationBudgetPolicy, scope: str) -> int | None:
    """The configured limit of one claimed scope, ``None`` for an unknown kind
    -- among them ``region``, ``cluster:`` and ``class:``, which an older
    release claimed and a RUNNING row it left may still list."""

    kind = scope.split(":", 1)[0]
    return {
        "node": policy.node_limit,
        "job": policy.failure_domain_limit,
        "domain": policy.failure_domain_limit,
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
    may touch a node, a job or a fabric partition that budget never counted;
    those scopes -- and only those -- have to be taken before the rung runs
    (F-N1 per-branch settlement).
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


# The operations that act on the whole job -- its workload objects
# (``workload_ids``, one per job: ``<ns>/pytorchjob/<name>``), or the restart
# authorization's ``job_id`` -- whatever ``node_ids`` they were compiled with.
WHOLE_JOB_OPERATIONS = frozenset(
    {WorkflowOperation.STOP_WORKLOADS, WorkflowOperation.RESTART_WORKLOAD}
)


def _budgeted(operation: WorkflowOperation) -> bool:
    # Isolation is budgeted like a mutation: it is containment-only for the
    # scheduler, but a false-positive cascade must not quarantine nodes without
    # limit (F-C5) -- it takes the node's slot, and its first admission counts
    # towards the cluster's storm breaker. Cordon and release alone stay free.
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
    claims: dict[str, int] = {}
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
    for step in pending:
        raw = step.parameters.get("fabric_partition")
        values = raw if isinstance(raw, list) else [raw]
        failure_domains.update(str(value) for value in values if value)
    for domain in sorted(failure_domains):
        claims[f"domain:{cluster_id}:{domain}"] = policy.failure_domain_limit
    # No region, cluster or resource-class scope: how many nodes a correlated
    # fault may take into destructive remediation at once is the cluster's
    # storm breaker's call (``gpu_fault.remediation_storm``), judged in the
    # claim's own transaction, not a fixed count that drained a whole-cluster
    # fault five nodes at a time.
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

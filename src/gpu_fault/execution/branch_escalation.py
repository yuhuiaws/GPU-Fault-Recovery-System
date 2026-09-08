"""In-place escalation of one failed node branch of a job workflow (F-N1).

A multi-node job workflow is a DAG: a shared STOP_WORKLOADS, one branch per
node, and a RESTART_WORKLOAD join that waits for every branch. When a branch
step fails, the failed *branch* walks the hardware-recovery ladder inside the
same workflow (reset -> reboot -> warm-spare replacement): its remaining
steps are retired and a fresh branch with the next rung is appended, the
join is re-wired to wait for it, and the sibling branches keep repairing.
A branch that exhausts the ladder is marked exhausted; the join must not run
and the workflow ends FAILED, which hands the outcome to the existing
whole-workflow escalation (support / drain).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Callable, Sequence

from gpu_fault.models import (
    WorkflowEventCode,
    WorkflowEventKind,
    WorkflowOperation,
    WorkflowRequest,
    WorkflowStatus,
    WorkflowStepExecution,
    WorkflowStepSpec,
    WorkflowStepStatus,
    record_workflow_event,
    resolved_step_indexes,
)
from gpu_fault.orchestration.dag_branching import branch_node_ids
from gpu_fault.orchestration.escalation import next_rung, unknown_outcome_failure

if TYPE_CHECKING:
    from gpu_fault.orchestration.dag_branching import DagBrancher

# Branch ids that are not node branches: a failure there is the job's, not one
# node's, and keeps the whole-workflow failure semantics.
JOB_LEVEL_BRANCH_IDS = frozenset({None, "shared", "join"})

# The steps an escalated branch runs after its rung: the same validation and
# release sequence the whole-workflow replacement path compiles.
ESCALATION_TAIL = (
    WorkflowOperation.VALIDATE_GPU,
    WorkflowOperation.VALIDATE_HOST,
    WorkflowOperation.VALIDATE_FABRIC,
    WorkflowOperation.RESTORE_SCHEDULING,
)

# (workflow, operations, node_id, gpu_uuids) -> steps. The workflow is passed
# so production can resolve its runtime profile; an empty list means the
# steps could not be compiled and the branch cannot be escalated in place.
CompileSteps = Callable[
    [WorkflowRequest, Sequence[WorkflowOperation], str, Sequence[str]],
    list[WorkflowStepSpec],
]


@dataclass(frozen=True)
class BranchEscalation:
    workflow: WorkflowRequest
    outcome: str  # "escalated" | "exhausted"
    node_id: str
    branch_id: str
    next_operation: WorkflowOperation | None
    reason: str


class BranchEscalator:
    def __init__(
        self,
        brancher: "DagBrancher",
        compile_steps: CompileSteps,
        *,
        max_rungs: int = 2,
    ) -> None:
        if max_rungs < 1:
            raise ValueError("branch escalation needs at least one rung")
        self.brancher = brancher
        self.compile_steps = compile_steps
        self.max_rungs = max_rungs

    def escalate_branch(
        self,
        workflow: WorkflowRequest,
        failed_index: int,
        error: str | None,
        *,
        details: Mapping[str, Any] | None = None,
    ) -> BranchEscalation | None:
        """Rewrite ``workflow`` after step ``failed_index`` FAILED.

        Returns ``None`` when the failure is not one node branch's (job-level
        step, multi-node step, non-DAG workflow); the caller then keeps the
        whole-workflow failure path. The returned workflow is not persisted.

        ``details`` is the failed outcome's: a step whose outcome is unknown
        (an INTERRUPTED reset, an abandoned mutation, a reused command id --
        ``unknown_outcome_failure``) takes no rung and is exhausted at once. The
        sibling branches still finish and release their nodes; the workflow
        then fails and the whole-workflow classifier hands the flagged step to
        an operator, exactly as off the DAG.
        """

        if not workflow.dag_enabled or not 0 <= failed_index < len(
            workflow.official_steps
        ):
            return None
        step = workflow.official_steps[failed_index]
        if step.branch_id in JOB_LEVEL_BRANCH_IDS or len(step.node_ids) != 1:
            return None
        node_id = step.node_ids[0]
        branch_id = step.branch_id
        assert branch_id is not None
        branch_indexes = set(self.brancher.node_branch_step_indexes(workflow, node_id))
        recovery_context = self._recovery_context(workflow, branch_indexes)
        unknown = unknown_outcome_failure(details)
        rung = None if unknown else next_rung(step.operation, recovery_context)
        taken = workflow.branch_escalation_counts.get(node_id, 0)
        if rung is None or taken >= self.max_rungs:
            return self._exhaust(
                workflow,
                node_id,
                branch_id,
                branch_indexes,
                reason=(
                    f"{step.operation.value} failed on {node_id}"
                    f" ({error or 'no error detail'}); "
                    + (
                        "outcome unknown, an operator confirms it; no rung"
                        if unknown
                        else "no further rung"
                        if rung is None
                        else f"{taken} rung(s) already taken"
                    )
                ),
                step_index=failed_index,
                operation=step.operation,
            )
        candidate_steps = self.compile_steps(
            workflow, [rung, *ESCALATION_TAIL], node_id, list(step.gpu_uuids)
        )
        if not candidate_steps:
            # Nothing compiled (profile gone, unsupported operation): the
            # whole-workflow failure path keeps the record honest.
            return None
        if rung is WorkflowOperation.REPLACE_NODE:
            candidate_steps = [
                (
                    item.model_copy(
                        update={
                            "parameters": {
                                **item.parameters,
                                "replacement_strategy": "HEALTHY_WARM_SPARE_ONLY",
                            }
                        }
                    )
                    if item.operation is WorkflowOperation.REPLACE_NODE
                    else item
                )
                for item in candidate_steps
            ]
        candidate = WorkflowRequest(
            request_id=f"{workflow.request_id}/escalation/{node_id}/{taken + 1}",
            incident_id=workflow.incident_id,
            status=WorkflowStatus.PENDING,
            fencing_token=workflow.fencing_token,
            runtime_profile_version=workflow.runtime_profile_version,
            official_action=rung.value,
            official_steps=candidate_steps,
        )
        replaced = self.brancher.replace_parallel_job_branch(
            workflow, candidate, node_id
        )
        replaced = replaced.model_copy(
            update={
                "branch_escalation_counts": {
                    **workflow.branch_escalation_counts,
                    node_id: taken + 1,
                },
                "updated_at": datetime.now(timezone.utc),
            }
        )
        reason = (
            f"{step.operation.value} failed on {node_id}"
            f" ({error or 'no error detail'}); escalating to {rung.value}"
        )
        # The brancher recorded the PLAN_REWRITE; this is the rung itself,
        # with both spellings of the branch's identity (RF-5, RF-7).
        replaced = record_workflow_event(
            replaced,
            WorkflowEventKind.BRANCH_ESCALATION,
            code=WorkflowEventCode.BRANCH_ESCALATED,
            step_index=failed_index,
            operation=step.operation,
            phase="official",
            status=WorkflowStepStatus.FAILED.value,
            reason=reason,
            details={
                "node_id": node_id,
                "branch_id": branch_id,
                "to_branch_id": next(
                    (
                        item.branch_id
                        for item in replaced.official_steps[
                            len(workflow.official_steps) :
                        ]
                        if item.branch_id is not None
                    ),
                    None,
                ),
                "from_operation": step.operation.value,
                "to_operation": rung.value,
                "rung_count": taken + 1,
                "max_rungs": self.max_rungs,
            },
        )
        return BranchEscalation(
            workflow=replaced,
            outcome="escalated",
            node_id=node_id,
            branch_id=branch_id,
            next_operation=rung,
            reason=reason,
        )

    def exhaust_branch(
        self,
        workflow: WorkflowRequest,
        node_id: str,
        branch_id: str,
        *,
        reason: str,
    ) -> BranchEscalation:
        """Retire ``node_id``'s branch without a further rung (caller's reason)."""

        branch_indexes = set(self.brancher.node_branch_step_indexes(workflow, node_id))
        return self._exhaust(
            workflow, node_id, branch_id, branch_indexes, reason=reason
        )

    @staticmethod
    def _recovery_context(
        workflow: WorkflowRequest, branch_indexes: set[int]
    ) -> set[WorkflowOperation]:
        completed = set(workflow.completed_step_indexes) & branch_indexes
        context = {workflow.official_steps[index].operation for index in completed}
        context.update(
            execution.operation
            for execution in workflow.step_executions
            if execution.step_index in branch_indexes
            and execution.status is WorkflowStepStatus.FAILED
        )
        return context

    @staticmethod
    def _exhaust(
        workflow: WorkflowRequest,
        node_id: str,
        branch_id: str,
        branch_indexes: set[int],
        *,
        reason: str,
        step_index: int | None = None,
        operation: WorkflowOperation | None = None,
    ) -> BranchEscalation:
        pending = branch_indexes - set(resolved_step_indexes(workflow))
        exhausted = workflow.model_copy(
            update={
                "superseded_step_indexes": sorted(
                    set(workflow.superseded_step_indexes) | pending
                ),
                "exhausted_branch_ids": [*workflow.exhausted_branch_ids, branch_id],
                "updated_at": datetime.now(timezone.utc),
            }
        )
        exhausted = record_workflow_event(
            exhausted,
            WorkflowEventKind.BRANCH_ESCALATION,
            code=WorkflowEventCode.BRANCH_EXHAUSTED,
            step_index=step_index,
            operation=operation,
            phase="official",
            reason=reason,
            details={
                "node_id": node_id,
                "branch_id": branch_id,
                "exhausted": True,
                "rung_count": workflow.branch_escalation_counts.get(node_id, 0),
                "superseded_indexes": sorted(pending),
            },
        )
        return BranchEscalation(
            workflow=exhausted,
            outcome="exhausted",
            node_id=node_id,
            branch_id=branch_id,
            next_operation=None,
            reason=reason,
        )


def exhausted_node_ids(workflow: WorkflowRequest) -> frozenset[str]:
    """The nodes whose ladder is exhausted, from ``exhausted_branch_ids``.

    ``branch_escalation_counts`` is keyed by node id and ``exhausted_branch_ids``
    by branch id; this is the join (RF-7). The parse is the inverse of the
    minting in ``dag_branching``, not a guess at the shape.
    """

    return frozenset(
        node
        for branch_id in workflow.exhausted_branch_ids
        for node in branch_node_ids(branch_id)
    )


def failed_execution(execution: WorkflowStepExecution) -> bool:
    return execution.status is WorkflowStepStatus.FAILED

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

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Callable, Sequence

from gpu_fault.models import (
    WorkflowOperation,
    WorkflowRequest,
    WorkflowStatus,
    WorkflowStepExecution,
    WorkflowStepSpec,
    WorkflowStepStatus,
    resolved_step_indexes,
)
from gpu_fault.orchestration.escalation import next_rung

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
    ) -> BranchEscalation | None:
        """Rewrite ``workflow`` after step ``failed_index`` FAILED.

        Returns ``None`` when the failure is not one node branch's (job-level
        step, multi-node step, non-DAG workflow); the caller then keeps the
        whole-workflow failure path. The returned workflow is not persisted.
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
        rung = next_rung(step.operation, recovery_context)
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
                        "no further rung"
                        if rung is None
                        else f"{taken} rung(s) already taken"
                    )
                ),
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
        return BranchEscalation(
            workflow=replaced,
            outcome="escalated",
            node_id=node_id,
            branch_id=branch_id,
            next_operation=rung,
            reason=(
                f"{step.operation.value} failed on {node_id}"
                f" ({error or 'no error detail'}); escalating to {rung.value}"
            ),
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
        return BranchEscalation(
            workflow=exhausted,
            outcome="exhausted",
            node_id=node_id,
            branch_id=branch_id,
            next_operation=None,
            reason=reason,
        )


def failed_execution(execution: WorkflowStepExecution) -> bool:
    return execution.status is WorkflowStepStatus.FAILED

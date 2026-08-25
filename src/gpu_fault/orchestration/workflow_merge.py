from __future__ import annotations

from datetime import datetime, timezone
from typing import Callable

from gpu_fault.models import (
    FaultIncident,
    WorkflowOperation,
    WorkflowRequest,
    WorkflowStatus,
    WorkflowStepExecution,
    WorkflowStepSpec,
    WorkflowStepStatus,
)
from gpu_fault.operation_registry import (
    NODE_MUTATING_OPERATIONS,
    PREEMPTION_NON_CANCELABLE_OPERATIONS,
)


class WorkflowMergeService:
    def __init__(
        self,
        arbiter,
        brancher,
        *,
        preemption_enabled: bool,
        workload_scoped_operations: set,
        node_exclusive_operations: set,
        workflow_resource_claims_by_node: Callable,
    ) -> None:
        self.arbiter = arbiter
        self.brancher = brancher
        self.preemption_enabled = preemption_enabled
        self.workload_scoped_operations = workload_scoped_operations
        self.node_exclusive_operations = node_exclusive_operations
        self.workflow_resource_claims_by_node = workflow_resource_claims_by_node

    def disposition(
        self,
        existing: WorkflowRequest,
        candidate: WorkflowRequest,
        node_id: str,
        gpu_uuids: set[str],
        *,
        allow_job_branch_merge: bool = True,
    ) -> str:
        if allow_job_branch_merge and self._read_only_branch_window_closed(
            existing, candidate
        ):
            return "QUEUE_SUCCESSOR"
        mutable = (
            existing.status is WorkflowStatus.PENDING
            and existing.execution_owner_id is None
            and not existing.completed_step_indexes
        )
        existing_rank = self.arbiter.workflow_recovery_rank(existing)
        candidate_rank = self.arbiter.workflow_recovery_rank(candidate)
        if allow_job_branch_merge and self.can_append_parallel_branch(
            existing,
            candidate,
        ):
            return "PARALLEL_BRANCH"
        if existing.dag_enabled and allow_job_branch_merge:
            result = self._dag_disposition(
                existing,
                candidate,
                node_id,
                gpu_uuids,
            )
            if result is not None:
                return result
        existing_primary = self.arbiter.workflow_primary_recovery_operations(existing)
        candidate_primary = self.arbiter.workflow_primary_recovery_operations(candidate)
        compatible = (
            candidate.status is WorkflowStatus.PENDING
            and self.arbiter.recovery_operations_dominate(
                existing_primary,
                candidate_primary,
            )
        )
        scope_covered = self.arbiter.workflow_covers_fault_scope(
            existing,
            node_id,
            gpu_uuids,
        )
        if compatible and (scope_covered or mutable):
            return "ABSORB"
        if (
            candidate_rank == existing_rank
            and candidate_primary == existing_primary
            and not self.recovery_action_has_started(
                existing,
                list(range(len(existing.official_steps))),
            )
        ):
            return "WIDEN_IN_PLACE"
        if (
            mutable
            and candidate_rank > existing_rank
            and self.arbiter.recovery_operations_dominate(
                candidate_primary,
                existing_primary,
            )
        ):
            return "REPLACE_IN_PLACE"
        if (
            allow_job_branch_merge
            and WorkflowOperation.RESTART_WORKLOAD
            in {step.operation for step in existing.official_steps}
            and self.brancher.dag_join_accepts_new_branch(existing)
        ):
            return "QUEUE_BRANCH_SUCCESSOR"
        return "REPLACE_IN_PLACE" if mutable else "QUEUE_SUCCESSOR"

    @staticmethod
    def _read_only_branch_window_closed(
        existing: WorkflowRequest,
        candidate: WorkflowRequest,
    ) -> bool:
        if any(
            step.operation in NODE_MUTATING_OPERATIONS
            for step in candidate.official_steps
        ):
            return False
        deadline = existing.aggregation_max_deadline
        if deadline is not None and datetime.now(timezone.utc) >= deadline:
            return True
        stop_indexes = {
            index
            for index, step in enumerate(existing.official_steps)
            if step.operation is WorkflowOperation.STOP_WORKLOADS
        }
        return bool(
            stop_indexes.intersection(existing.completed_step_indexes)
            or any(
                execution.step_index in stop_indexes
                and execution.status is WorkflowStepStatus.SUCCEEDED
                for execution in existing.step_executions
            )
        )

    def _dag_disposition(
        self,
        existing: WorkflowRequest,
        candidate: WorkflowRequest,
        node_id: str,
        gpu_uuids: set[str],
    ) -> str | None:
        existing_primary = self.arbiter.workflow_primary_recovery_operations(existing)
        candidate_primary = self.arbiter.workflow_primary_recovery_operations(candidate)
        if (
            not any(step.branch_id for step in existing.official_steps)
            and candidate.status is WorkflowStatus.PENDING
            and self.arbiter.recovery_operations_dominate(
                existing_primary,
                candidate_primary,
            )
            and self.arbiter.workflow_covers_fault_scope(
                existing,
                node_id,
                gpu_uuids,
            )
        ):
            return "ABSORB"
        if not self.brancher.dag_join_accepts_new_branch(existing):
            return "QUEUE_SUCCESSOR"
        indexes = self.brancher.node_branch_step_indexes(
            existing,
            node_id,
        )
        if not indexes:
            return None
        existing_rank = self.brancher.step_indexes_recovery_rank(
            existing,
            indexes,
        )
        candidate_rank = self.arbiter.workflow_recovery_rank(candidate)
        existing_primary = self.arbiter.primary_recovery_operations(
            {existing.official_steps[index].operation for index in indexes}
        )
        candidate_primary = self.arbiter.workflow_primary_recovery_operations(candidate)
        compatible = (
            candidate.status is WorkflowStatus.PENDING
            and self.arbiter.recovery_operations_dominate(
                existing_primary,
                candidate_primary,
            )
        )
        if compatible and self.brancher.node_branch_covers_fault_scope(
            existing,
            indexes,
            node_id,
            gpu_uuids,
        ):
            return "ABSORB"
        branch_started = self.brancher.branch_has_started(
            existing,
            indexes,
        )
        if (
            candidate_rank == existing_rank
            and candidate_primary == existing_primary
            and not self.recovery_action_has_started(existing, indexes)
        ):
            return "WIDEN_BRANCH"
        if (
            not branch_started
            and candidate_rank > existing_rank
            and self.arbiter.recovery_operations_dominate(
                candidate_primary,
                existing_primary,
            )
        ):
            return "REPLACE_BRANCH"
        return "QUEUE_BRANCH_SUCCESSOR"

    def can_append_parallel_branch(
        self,
        existing: WorkflowRequest,
        candidate: WorkflowRequest,
    ) -> bool:
        if (
            not self.preemption_enabled
            or existing.status not in {WorkflowStatus.PENDING, WorkflowStatus.RUNNING}
            or candidate.status is not WorkflowStatus.PENDING
        ):
            return False
        existing_operations = {step.operation for step in existing.official_steps}
        if WorkflowOperation.RESTART_WORKLOAD not in existing_operations:
            return False
        existing_nodes = self._physical_nodes(existing)
        candidate_nodes = self._physical_nodes(candidate)
        if not existing_nodes or not candidate_nodes:
            return False
        existing_claims = self.workflow_resource_claims_by_node(existing)
        candidate_claims = self.workflow_resource_claims_by_node(candidate)
        for node_id in existing_nodes & candidate_nodes:
            if self.arbiter.resource_claims_conflict(
                existing_claims.get(node_id, frozenset()),
                candidate_claims.get(node_id, frozenset()),
            ):
                return False
        return self.brancher.dag_join_accepts_new_branch(existing)

    def _physical_nodes(
        self,
        workflow: WorkflowRequest,
    ) -> set[str]:
        return {
            node_id
            for step in workflow.official_steps
            if step.operation not in self.workload_scoped_operations
            for node_id in step.node_ids
        }

    def preempt_parallel_branch(
        self,
        existing: WorkflowRequest,
        candidate: WorkflowRequest,
        node_id: str,
    ) -> WorkflowRequest:
        indexes = self.brancher.node_branch_step_indexes(
            existing,
            node_id,
        )
        if not indexes and not existing.dag_enabled:
            indexes = [
                index
                for index, step in enumerate(existing.official_steps)
                if step.operation
                not in {
                    WorkflowOperation.CHECKPOINT_WORKLOADS,
                    WorkflowOperation.STOP_WORKLOADS,
                    WorkflowOperation.RESTART_WORKLOAD,
                }
                and node_id in step.node_ids
            ]
        if not indexes:
            return self.brancher.append_parallel_job_branch_successor(
                existing,
                candidate,
                node_id,
            )
        predecessor, replaced = self._preemption_boundary(
            existing,
            indexes,
        )
        candidate = self._handoff_quiesce(
            existing,
            candidate,
            indexes,
        )
        return self.brancher.append_parallel_job_branch(
            existing,
            candidate,
            predecessor_step_index=predecessor,
            replaced_step_indexes=replaced,
        )

    @staticmethod
    def _preemption_boundary(
        workflow: WorkflowRequest,
        indexes: list[int],
    ) -> tuple[int | None, frozenset[int]]:
        completed = set(workflow.completed_step_indexes)
        blocking = [
            execution.step_index
            for execution in workflow.step_executions
            if execution.step_index in indexes
            and execution.status is WorkflowStepStatus.WAITING
            and execution.operation in PREEMPTION_NON_CANCELABLE_OPERATIONS
        ]
        safe_completed = [index for index in indexes if index in completed]
        predecessor = (
            max(blocking)
            if blocking
            else max(safe_completed)
            if safe_completed
            else None
        )
        protected = set(blocking) | completed
        return predecessor, frozenset(
            index for index in indexes if index not in protected
        )

    @staticmethod
    def _handoff_quiesce(
        existing: WorkflowRequest,
        candidate: WorkflowRequest,
        indexes: list[int],
    ) -> WorkflowRequest:
        completed = set(existing.completed_step_indexes)
        quiesced = any(
            index in completed
            and existing.official_steps[index].operation
            is WorkflowOperation.QUIESCE_GPU_SERVICES
            for index in indexes
        )
        restored = any(
            index in completed
            and existing.official_steps[index].operation
            is WorkflowOperation.RESTORE_GPU_SERVICES
            for index in indexes
        )
        steps = list(candidate.official_steps)
        if (
            not quiesced
            or restored
            or not any(
                step.operation is WorkflowOperation.RESTART_NODE for step in steps
            )
            or any(
                step.operation is WorkflowOperation.RESTORE_GPU_SERVICES
                for step in steps
            )
        ):
            return candidate
        owner = next(
            (
                step.execution_owner
                for step in existing.official_steps
                if step.operation is WorkflowOperation.RESTORE_GPU_SERVICES
            ),
            next(
                step.execution_owner
                for step in existing.official_steps
                if step.operation is WorkflowOperation.QUIESCE_GPU_SERVICES
            ),
        )
        reboot_index = next(
            index
            for index, step in enumerate(steps)
            if step.operation is WorkflowOperation.RESTART_NODE
        )
        reboot_step = steps[reboot_index]
        steps.insert(
            reboot_index + 1,
            WorkflowStepSpec(
                operation=WorkflowOperation.RESTORE_GPU_SERVICES,
                execution_owner=owner,
                node_ids=list(reboot_step.node_ids),
                gpu_uuids=list(reboot_step.gpu_uuids),
                workload_ids=list(reboot_step.workload_ids),
                parameters={
                    "preemption_quiesce_handoff_after_reboot": True,
                },
            ),
        )
        return candidate.model_copy(update={"official_steps": steps})

    def prepare_preempting_successor(
        self,
        existing: WorkflowRequest,
        candidate: WorkflowRequest,
    ) -> WorkflowRequest:
        existing_rank = self.arbiter.workflow_recovery_rank(existing)
        candidate_rank = self.arbiter.workflow_recovery_rank(candidate)
        existing_nodes = self._exclusive_nodes(existing)
        candidate_nodes = self._exclusive_nodes(candidate)
        if (
            not self.preemption_enabled
            or candidate.status is not WorkflowStatus.PENDING
            or existing.status not in {WorkflowStatus.PENDING, WorkflowStatus.RUNNING}
            or candidate_rank <= existing_rank
            or (
                existing_nodes
                and candidate_nodes
                and not existing_nodes & candidate_nodes
            )
        ):
            return candidate
        inherited = self._inherited_containment(existing, candidate)
        return candidate.model_copy(
            update={
                "preempt_predecessor": True,
                "preemption_reason": (
                    "strictly stronger recovery action: "
                    f"rank {existing_rank} -> {candidate_rank}"
                ),
                "completed_step_indexes": inherited[0],
                "completed_operations": inherited[1],
                "step_executions": inherited[2],
                "inherited_step_indexes": inherited[0],
                "updated_at": datetime.now(timezone.utc),
            }
        )

    def _exclusive_nodes(
        self,
        workflow: WorkflowRequest,
    ) -> set[str]:
        return {
            node_id
            for step in workflow.official_steps
            if step.operation in self.node_exclusive_operations
            for node_id in step.node_ids
        }

    def _inherited_containment(
        self,
        existing: WorkflowRequest,
        candidate: WorkflowRequest,
    ) -> tuple[list[int], list[WorkflowOperation], list]:
        reusable = {
            WorkflowOperation.MARK_UNSCHEDULABLE,
            WorkflowOperation.STOP_WORKLOADS,
        }
        invalidated = {
            WorkflowOperation.MARK_UNSCHEDULABLE: (
                WorkflowOperation.RESTORE_SCHEDULING in existing.completed_operations
            ),
            WorkflowOperation.STOP_WORKLOADS: (
                WorkflowOperation.RESTART_WORKLOAD in existing.completed_operations
            ),
        }
        completed = set(existing.completed_step_indexes)
        indexes: list[int] = []
        operations: list[WorkflowOperation] = []
        executions = []
        for candidate_index, candidate_step in enumerate(candidate.official_steps):
            if (
                candidate_step.operation not in reusable
                or invalidated[candidate_step.operation]
            ):
                continue
            match = next(
                (
                    (index, step)
                    for index, step in enumerate(existing.official_steps)
                    if index in completed
                    and step.operation is candidate_step.operation
                    and self.arbiter.step_scope_covers(
                        step,
                        candidate_step,
                    )
                ),
                None,
            )
            if match is None:
                continue
            existing_index, _ = match
            previous = next(
                (
                    item
                    for item in existing.step_executions
                    if item.step_index == existing_index
                    and item.status is WorkflowStepStatus.SUCCEEDED
                ),
                None,
            )
            indexes.append(candidate_index)
            operations.append(candidate_step.operation)
            executions.append(
                self._inherited_execution(
                    candidate_index,
                    candidate_step.operation,
                    existing.request_id,
                    previous,
                )
            )
        return indexes, operations, executions

    @staticmethod
    def _inherited_execution(
        index: int,
        operation: WorkflowOperation,
        workflow_id: str,
        previous,
    ) -> WorkflowStepExecution:
        now = datetime.now(timezone.utc)
        return WorkflowStepExecution(
            step_index=index,
            operation=operation,
            status=WorkflowStepStatus.SUCCEEDED,
            adapter_operation_id=(
                previous.adapter_operation_id if previous is not None else None
            ),
            details={
                **(previous.details if previous is not None else {}),
                "inherited_from_workflow_id": workflow_id,
                "preemption_reuse": True,
            },
            started_at=now,
            updated_at=now,
        )

    def recovery_action_has_started(
        self,
        workflow: WorkflowRequest,
        indexes: list[int],
    ) -> bool:
        index_set = set(indexes)
        action_operations = self.arbiter.primary_recovery_operations(
            {workflow.official_steps[index].operation for index in indexes}
        )
        action_indexes = {
            index
            for index in indexes
            if workflow.official_steps[index].operation in action_operations
        }
        if not action_indexes:
            action_indexes = index_set
        return bool(
            action_indexes.intersection(workflow.completed_step_indexes)
            or action_indexes.intersection(workflow.superseded_step_indexes)
            or any(
                execution.step_index in action_indexes
                for execution in workflow.step_executions
            )
        )

    @staticmethod
    def preemption_scope_matches(
        existing: FaultIncident,
        candidate: FaultIncident,
    ) -> bool:
        if existing.job_id is not None or candidate.job_id is not None:
            return (
                existing.job_id is not None
                and existing.job_id == candidate.job_id
                and existing.attempt_id == candidate.attempt_id
                and bool(set(existing.node_ids) & set(candidate.node_ids))
            )
        return bool(set(existing.node_ids) & set(candidate.node_ids))

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Callable

from gpu_fault.models import (
    EXECUTABLE_WORKFLOW_STATUSES,
    FaultIncident,
    WorkflowEventCode,
    WorkflowEventKind,
    WorkflowOperation,
    WorkflowRequest,
    WorkflowStatus,
    WorkflowStepExecution,
    WorkflowStepSpec,
    WorkflowStepStatus,
    lifetime_exceeded,
    record_workflow_event,
)
from gpu_fault.operation_registry import (
    NODE_MUTATING_OPERATIONS,
    NODE_WIDE_RECOVERY_OPERATIONS,
)
from gpu_fault.orchestration.disposition import Disposition
from gpu_fault.orchestration.preemption_boundary import preemption_boundary

LOGGER = logging.getLogger(__name__)


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
        # F-N1: read-only events recorded on the incident with no new steps.
        self.absorbed_record_only_total = 0
        self.lifetime_record_only_total = 0
        self.withdrawn_record_only_total = 0

    def disposition(
        self,
        existing: WorkflowRequest,
        candidate: WorkflowRequest,
        node_id: str,
        gpu_uuids: set[str],
        *,
        allow_job_branch_merge: bool = True,
        now: datetime | None = None,
    ) -> Disposition:
        if lifetime_exceeded(existing):
            # The remediation ran out of lifetime and is with an operator
            # (F-N1): the event is recorded on its incident, nothing is
            # planned or queued until the incident is closed.
            self.lifetime_record_only_total += 1
            return Disposition.ABSORB_RECORD_ONLY
        if existing.workload_withdrawn_at is not None:
            # The job this workflow repaired was stopped by someone else; it
            # is winding down (F-N1 §7). The event is recorded, not planned.
            self.withdrawn_record_only_total += 1
            return Disposition.ABSORB_RECORD_ONLY
        if existing.status not in EXECUTABLE_WORKFLOW_STATUSES:
            # Nothing merges into a record that will never execute again --
            # BLOCKED included. ABSORB / WIDEN into one silently dropped the
            # new fault and left the node's group link pinned to it (F-B4).
            # The candidate gets its own record; a terminal predecessor does
            # not hold it back.
            LOGGER.warning(
                "merge target %s is %s; queueing %s as its own record",
                existing.request_id,
                existing.status.value,
                candidate.request_id,
            )
            return Disposition.QUEUE_SUCCESSOR
        if allow_job_branch_merge and self._read_only_branch_window_closed(
            existing, candidate, now
        ):
            if self._mutating_scope_covers(existing, node_id, gpu_uuids):
                # A node-mutating action is already under way for exactly
                # this scope. Evidence collected after it finishes describes
                # a rebooted node, not the fault; the event lands on the
                # incident and schedules nothing (F-N1).
                self.absorbed_record_only_total += 1
                return Disposition.ABSORB_RECORD_ONLY
            return Disposition.QUEUE_SUCCESSOR
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
            return Disposition.PARALLEL_BRANCH
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
            candidate=candidate,
        )
        if compatible and (scope_covered or mutable):
            return Disposition.ABSORB
        if (
            candidate_rank == existing_rank
            and candidate_primary == existing_primary
            and not self.recovery_action_has_started(
                existing,
                list(range(len(existing.official_steps))),
            )
        ):
            return Disposition.WIDEN_IN_PLACE
        if (
            mutable
            and candidate_rank > existing_rank
            and self.arbiter.recovery_operations_dominate(
                candidate_primary,
                existing_primary,
            )
        ):
            return Disposition.REPLACE_IN_PLACE
        if (
            allow_job_branch_merge
            and WorkflowOperation.RESTART_WORKLOAD
            in {step.operation for step in existing.official_steps}
            and self.brancher.dag_join_accepts_new_branch(existing)
        ):
            return Disposition.QUEUE_BRANCH_SUCCESSOR
        return Disposition.REPLACE_IN_PLACE if mutable else Disposition.QUEUE_SUCCESSOR

    @staticmethod
    def _mutating_scope_covers(
        existing: WorkflowRequest,
        node_id: str,
        gpu_uuids: set[str],
    ) -> bool:
        """Does a live node-mutating step of ``existing`` cover this fault?

        Node-wide actions (reboot, replace, drain) cover every GPU of the
        node; a GPU-scoped action covers only the GPUs it names. Superseded
        steps never run again and do not count.
        """

        superseded = set(existing.superseded_step_indexes)
        for index, step in enumerate(existing.official_steps):
            if (
                index in superseded
                or step.operation not in NODE_MUTATING_OPERATIONS
                or node_id not in step.node_ids
            ):
                continue
            if (
                step.operation in NODE_WIDE_RECOVERY_OPERATIONS
                or not gpu_uuids
                or gpu_uuids <= set(step.gpu_uuids)
            ):
                return True
        return False

    @staticmethod
    def _read_only_branch_window_closed(
        existing: WorkflowRequest,
        candidate: WorkflowRequest,
        now: datetime | None = None,
    ) -> bool:
        if any(
            step.operation in NODE_MUTATING_OPERATIONS
            for step in candidate.official_steps
        ):
            return False
        deadline = existing.aggregation_max_deadline
        if deadline is not None and (now or datetime.now(timezone.utc)) >= deadline:
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
    ) -> Disposition | None:
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
                candidate=candidate,
            )
        ):
            return Disposition.ABSORB
        if not self.brancher.dag_join_accepts_new_branch(existing):
            return Disposition.QUEUE_SUCCESSOR
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
            candidate=candidate,
        ):
            return Disposition.ABSORB
        branch_started = self.brancher.branch_has_started(
            existing,
            indexes,
        )
        if (
            candidate_rank == existing_rank
            and candidate_primary == existing_primary
            and not self.recovery_action_has_started(existing, indexes)
        ):
            return Disposition.WIDEN_BRANCH
        if (
            not branch_started
            and candidate_rank > existing_rank
            and self.arbiter.recovery_operations_dominate(
                candidate_primary,
                existing_primary,
            )
        ):
            return Disposition.REPLACE_BRANCH
        return Disposition.QUEUE_BRANCH_SUCCESSOR

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
        # The same judgement the executor makes at a step boundary (F-C1),
        # restricted to this node's branch. Planning has no store to cancel a
        # remote command through, so a cancellable wait is a boundary here
        # rather than a step to mark superseded under a running command.
        boundary = preemption_boundary(
            existing,
            indexes=indexes,
            remote_cancellation_available=False,
        )
        candidate = self._handoff_quiesce(
            existing,
            candidate,
            indexes,
        )
        combined = self.brancher.append_parallel_job_branch(
            existing,
            candidate,
            predecessor_step_index=boundary.predecessor,
            replaced_step_indexes=boundary.replaceable,
        )
        if combined is existing:
            # The brancher found the branch already there; nothing was
            # preempted and nothing is recorded.
            return existing
        # Beside the brancher's PLAN_REWRITE: who preempted whom, at which
        # boundary, and why the boundary sat where it did (RF-5, RF-8). A
        # closed boundary still retires the pending tail behind it; the code
        # says a wait held the line, ``replaced_indexes`` says what went.
        return record_workflow_event(
            combined,
            WorkflowEventKind.PREEMPTION,
            code=(
                WorkflowEventCode.PREEMPTED
                if boundary.open
                else WorkflowEventCode.PREEMPTION_BOUNDARY_CLOSED
            ),
            reason=boundary.reason,
            details={
                "predecessor_workflow_id": existing.request_id,
                "successor_workflow_id": candidate.request_id,
                "node_id": node_id,
                "replaced_indexes": sorted(boundary.replaceable),
                "boundary_step_index": boundary.predecessor,
                "blocking_indexes": sorted(boundary.blocking),
            },
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
        if candidate.lifetime_deadline_at is None and existing.lifetime_deadline_at:
            # A successor shares its predecessor's lifetime (F-N1).
            candidate = candidate.model_copy(
                update={"lifetime_deadline_at": existing.lifetime_deadline_at}
            )
        existing_rank = self.arbiter.workflow_recovery_rank(existing)
        candidate_rank = self.arbiter.workflow_recovery_rank(candidate)
        existing_nodes = self._exclusive_nodes(existing)
        candidate_nodes = self._exclusive_nodes(candidate)
        if (
            not self.preemption_enabled
            or candidate.status is not WorkflowStatus.PENDING
            or existing.status not in {WorkflowStatus.PENDING, WorkflowStatus.RUNNING}
            or candidate_rank <= existing_rank
            # Fail closed on the node gate (F-C1): a candidate that holds no
            # node-exclusive step is not a stronger node action whatever its
            # rank says, and a predecessor that does hold nodes is only
            # preempted by a successor on those same nodes. An empty set no
            # longer makes the gate pass vacuously.
            or not candidate_nodes
            or (existing_nodes and not existing_nodes & candidate_nodes)
        ):
            return candidate
        inherited = self._inherited_containment(existing, candidate)
        reason = (
            "strictly stronger recovery action: "
            f"rank {existing_rank} -> {candidate_rank}"
        )
        successor = candidate.model_copy(
            update={
                "preempt_predecessor": True,
                "preemption_reason": reason,
                "completed_step_indexes": inherited[0],
                "completed_operations": inherited[1],
                "step_executions": inherited[2],
                "inherited_step_indexes": inherited[0],
                "updated_at": datetime.now(timezone.utc),
            }
        )
        # On the successor's own trail: the predecessor's record ends with the
        # executor's TERMINAL event once the preemption lands (RF-5, RF-8).
        return record_workflow_event(
            successor,
            WorkflowEventKind.PREEMPTION,
            code=WorkflowEventCode.PREEMPTED,
            reason=reason,
            details={
                "predecessor_workflow_id": existing.request_id,
                "successor_workflow_id": candidate.request_id,
                "from_rank": existing_rank,
                "to_rank": candidate_rank,
                "inherited_step_indexes": list(inherited[0]),
            },
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
        # A containment step is only a fact until its undo has started: a
        # RESTART_WORKLOAD or RESTORE_SCHEDULING with any execution record --
        # WAITING included -- means the job or the node is already on its way
        # back, so the successor must redo the containment (F-C1).
        undone = set(existing.completed_operations) | {
            execution.operation for execution in existing.step_executions
        }
        invalidated = {
            WorkflowOperation.MARK_UNSCHEDULABLE: (
                WorkflowOperation.RESTORE_SCHEDULING in undone
            ),
            WorkflowOperation.STOP_WORKLOADS: (
                WorkflowOperation.RESTART_WORKLOAD in undone
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
            existing_index, existing_step = match
            previous = next(
                (
                    item
                    for item in existing.step_executions
                    if item.step_index == existing_index
                    and item.operation is existing_step.operation
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
            phase="official",
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

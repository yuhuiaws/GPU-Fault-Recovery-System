from __future__ import annotations

from gpu_fault.models import (
    WorkflowOperation,
    WorkflowRequest,
    WorkflowStepSpec,
)
from gpu_fault.operation_registry import (
    MERGE_INTENT_OPERATIONS,
    NODE_ACTION_SCOPE_OPERATIONS,
    NODE_WIDE_RECOVERY_OPERATIONS,
    RECOVERY_OPERATION_DOMINANCE,
    RECOVERY_OPERATION_RANK,
    WORKLOAD_SCOPED_OPERATIONS,
    ZERO_RANK_ACTION_OPERATIONS,
)


class RecoveryArbiter:
    RECOVERY_OPERATION_RANK = RECOVERY_OPERATION_RANK
    MERGE_INTENT_OPERATIONS = MERGE_INTENT_OPERATIONS
    NODE_WIDE_RECOVERY_OPERATIONS = NODE_WIDE_RECOVERY_OPERATIONS
    ZERO_RANK_ACTION_OPERATIONS = ZERO_RANK_ACTION_OPERATIONS
    RECOVERY_OPERATION_DOMINANCE = RECOVERY_OPERATION_DOMINANCE
    NODE_ACTION_OPERATIONS = NODE_ACTION_SCOPE_OPERATIONS
    WORKLOAD_SCOPED_OPERATIONS = WORKLOAD_SCOPED_OPERATIONS

    def recovery_rank(self, operations: set[WorkflowOperation]) -> int:
        return max(
            (
                self.RECOVERY_OPERATION_RANK.get(operation, 0)
                for operation in operations
            ),
            default=0,
        )

    def workflow_recovery_rank(self, workflow: WorkflowRequest) -> int:
        return self.recovery_rank({step.operation for step in workflow.official_steps})

    def primary_recovery_operations(
        self,
        operations: set[WorkflowOperation],
    ) -> frozenset[WorkflowOperation]:
        node_ranked = {
            operation
            for operation in operations
            if operation is not WorkflowOperation.RESTART_WORKLOAD
            and self.RECOVERY_OPERATION_RANK.get(operation, 0) > 0
        }
        zero_ranked = operations & self.ZERO_RANK_ACTION_OPERATIONS
        ranked = {
            operation
            for operation in operations
            if self.RECOVERY_OPERATION_RANK.get(operation, 0) > 0
        }
        if node_ranked:
            ranked = node_ranked
        elif zero_ranked:
            return frozenset(zero_ranked)
        if not ranked:
            return frozenset(zero_ranked)
        return frozenset(
            operation
            for operation in ranked
            if not any(
                operation in self.RECOVERY_OPERATION_DOMINANCE.get(other, frozenset())
                for other in ranked
                if other is not operation
            )
        )

    def workflow_primary_recovery_operations(
        self, workflow: WorkflowRequest
    ) -> frozenset[WorkflowOperation]:
        return self.primary_recovery_operations(
            {step.operation for step in workflow.official_steps}
        )

    def recovery_operations_dominate(
        self,
        existing: frozenset[WorkflowOperation],
        candidate: frozenset[WorkflowOperation],
    ) -> bool:
        if not candidate:
            return not existing
        for candidate_operation in candidate:
            if candidate_operation in existing:
                continue
            if not any(
                candidate_operation
                in self.RECOVERY_OPERATION_DOMINANCE.get(
                    existing_operation, frozenset()
                )
                for existing_operation in existing
            ):
                return False
        return True

    def merge_intent(self, workflow: WorkflowRequest) -> frozenset[WorkflowOperation]:
        return frozenset(
            step.operation
            for step in workflow.official_steps
            if step.operation in self.MERGE_INTENT_OPERATIONS
        )

    def workflow_covers_fault_scope(
        self,
        workflow: WorkflowRequest,
        node_id: str,
        gpu_uuids: set[str],
    ) -> bool:
        node_ids = {
            node
            for step in workflow.official_steps
            if step.operation not in self.WORKLOAD_SCOPED_OPERATIONS
            for node in step.node_ids
        }
        if node_id not in node_ids:
            return False
        intent = self.merge_intent(workflow)
        if intent.intersection(self.NODE_WIDE_RECOVERY_OPERATIONS):
            return True
        if not gpu_uuids:
            return True
        covered: set[str] = set()
        for step in workflow.official_steps:
            if node_id not in step.node_ids:
                continue
            raw = step.parameters.get("gpu_uuids_by_node")
            if isinstance(raw, dict):
                covered.update(str(value) for value in raw.get(node_id, []))
            else:
                covered.update(step.gpu_uuids)
        return gpu_uuids <= covered

    @staticmethod
    def step_scope_covers(
        existing: WorkflowStepSpec,
        candidate: WorkflowStepSpec,
    ) -> bool:
        if not set(candidate.node_ids) <= set(existing.node_ids):
            return False
        if candidate.workload_ids and not (
            set(candidate.workload_ids) <= set(existing.workload_ids)
        ):
            return False
        return True

    @staticmethod
    def resource_claims_conflict(
        existing: frozenset[str],
        candidate: frozenset[str],
    ) -> bool:
        if not existing or not candidate:
            return False
        if (
            "NODE_LIFECYCLE_MUTATION" in existing
            or "NODE_LIFECYCLE_MUTATION" in candidate
        ):
            return True
        return bool(existing & candidate)

    def merged_gpu_scope(
        self,
        existing_workflow: WorkflowRequest | None,
        node_id: str,
        gpu_uuids: set[str],
    ) -> dict[str, list[str]]:
        mapping: dict[str, set[str]] = {}
        if existing_workflow is not None:
            for step in existing_workflow.official_steps:
                if step.operation not in self.NODE_ACTION_OPERATIONS:
                    continue
                raw = step.parameters.get("gpu_uuids_by_node")
                if isinstance(raw, dict):
                    for existing_node, values in raw.items():
                        if isinstance(values, list):
                            mapping.setdefault(str(existing_node), set()).update(
                                str(value) for value in values
                            )
                    continue
                for existing_node in step.node_ids:
                    mapping.setdefault(existing_node, set()).update(step.gpu_uuids)
        if gpu_uuids:
            mapping.setdefault(node_id, set()).update(gpu_uuids)
        return {key: sorted(values) for key, values in mapping.items() if values}

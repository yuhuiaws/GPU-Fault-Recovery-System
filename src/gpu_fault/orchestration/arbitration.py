from __future__ import annotations

from collections.abc import Iterable

from gpu_fault.models import (
    WorkflowOperation,
    WorkflowRequest,
    WorkflowStepSpec,
    resolved_step_indexes,
)
from gpu_fault.operation_registry import (
    MERGE_INTENT_OPERATIONS,
    NODE_ACTION_SCOPE_OPERATIONS,
    NODE_MUTATING_OPERATIONS,
    NODE_WIDE_RECOVERY_OPERATIONS,
    RECOVERY_OPERATION_DOMINANCE,
    RECOVERY_OPERATION_RANK,
    WORKLOAD_SCOPED_OPERATIONS,
    ZERO_RANK_ACTION_OPERATIONS,
)

# GPU-scoped recovery actions. A fault that names no GPU yet asks for one of
# these has an unresolved GPU identity; it is not a node-level fault.
GPU_SCOPED_RECOVERY_OPERATIONS = (
    NODE_ACTION_SCOPE_OPERATIONS & NODE_MUTATING_OPERATIONS
) - NODE_WIDE_RECOVERY_OPERATIONS


def gpu_identity_unresolved(
    candidate: WorkflowRequest | None,
    node_id: str,
    gpu_uuids: set[str],
) -> bool:
    if gpu_uuids or candidate is None:
        return False
    return any(
        step.operation in GPU_SCOPED_RECOVERY_OPERATIONS and node_id in step.node_ids
        for step in candidate.official_steps
    )


def fault_scope_covered(
    workflow: WorkflowRequest,
    node_id: str,
    gpu_uuids: set[str],
    *,
    indexes: Iterable[int] | None = None,
    candidate: WorkflowRequest | None = None,
) -> bool:
    """Whether the steps that will still run already act on this fault.

    Only live steps count -- completed and superseded ones are history
    (F-C2), and only steps that name ``node_id``. A node-wide recovery on
    that node covers every GPU on it. A fault with no GPU identity is
    covered when a live step names the node, unless the candidate asks for
    a GPU-scoped action: then the identity is unresolved, not node-level,
    and absorbing it would leave that GPU untouched. GPU coverage is read
    from node-mutating steps only; an evidence step that froze a GPU's
    state has not repaired it (F-B6).
    """
    resolved = resolved_step_indexes(workflow)
    selected = None if indexes is None else set(indexes)
    live = [
        step
        for index, step in enumerate(workflow.official_steps)
        if (selected is None or index in selected)
        and index not in resolved
        and node_id in step.node_ids
        and step.operation not in WORKLOAD_SCOPED_OPERATIONS
    ]
    if not live:
        return False
    if any(step.operation in NODE_WIDE_RECOVERY_OPERATIONS for step in live):
        return True
    if not gpu_uuids:
        return not gpu_identity_unresolved(candidate, node_id, gpu_uuids)
    covered: set[str] = set()
    for step in live:
        if step.operation not in NODE_MUTATING_OPERATIONS:
            continue
        raw = step.parameters.get("gpu_uuids_by_node")
        if isinstance(raw, dict):
            covered.update(str(value) for value in raw.get(node_id, []))
        else:
            covered.update(step.gpu_uuids)
    return gpu_uuids <= covered


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
        *,
        candidate: WorkflowRequest | None = None,
    ) -> bool:
        return fault_scope_covered(workflow, node_id, gpu_uuids, candidate=candidate)

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
        # A node with no GPU identity stays in scope with an empty list so
        # host-level steps still widen onto it (F-B6); GPU actions skip
        # such nodes because the agent refuses a node without GPUs.
        mapping.setdefault(node_id, set()).update(gpu_uuids)
        return {key: sorted(values) for key, values in mapping.items()}

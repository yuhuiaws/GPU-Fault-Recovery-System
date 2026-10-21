from __future__ import annotations

from collections.abc import Collection, Iterable, Mapping

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
from gpu_fault.orchestration.gpu_placement import workflow_gpu_placement
from gpu_fault.workflow_quarantine import terminal_quarantine_covered

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
    if candidate is not None and not terminal_quarantine_covered(workflow, candidate):
        return False
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


def step_gpu_shares(
    workflow: WorkflowRequest, step: WorkflowStepSpec
) -> dict[str, list[str]] | None:
    """Which of a node-action step's nodes holds which of its GPUs.

    For a step without its own ``gpu_uuids_by_node`` (a caller reads an
    explicit mapping itself). A single-node step's GPUs are its node's. A
    multi-node step's flat list does not say which node holds which GPU, so
    each GPU is placed with the workflow's own per-node lists
    (``gpu_placement.workflow_gpu_placement``: explicit mappings, single-node
    steps) restricted to the step's nodes. The shares are returned only when
    every GPU of the step lands on exactly one of its nodes; ``None`` --
    unknown -- when one lands on none of them (nothing places it, or only a
    node outside the step) or on two.
    """

    if len(step.node_ids) <= 1:
        return {
            node_id: list(dict.fromkeys(step.gpu_uuids)) for node_id in step.node_ids
        }
    placement = workflow_gpu_placement(workflow)
    shares: dict[str, list[str]] = {node_id: [] for node_id in step.node_ids}
    for gpu in dict.fromkeys(step.gpu_uuids):
        holders = [node_id for node_id in shares if gpu in placement.get(node_id, ())]
        if len(holders) != 1:
            return None
        shares[holders[0]].append(gpu)
    return shares


def widened_gpu_mapping(
    workflow: WorkflowRequest,
    step: WorkflowStepSpec,
    gpu_uuids_by_node: Mapping[str, Collection[str]],
) -> dict[str, list[str]]:
    """The ``gpu_uuids_by_node`` a merge writes on a pending node-action step.

    The step's own mapping plus every node of the merged scope
    (``RecoveryArbiter.merged_gpu_scope``) that carries GPUs; a node without
    GPU identity joins host-level steps only, as the node-action adapter
    refuses a GPU action on a node whose list is empty (P0-57C). A multi-node
    step without its own mapping whose GPUs ``step_gpu_shares`` cannot place
    gets no entry for any of its own nodes, whatever the scope holds for them:
    any of them may hold the unplaced GPU, and a partial entry would drop it
    from the action unseen. With the entry missing, the adapter refuses the
    step before any command is sent ("missing explicit GPU UUIDs for <node>").
    """

    merged: dict[str, set[str]] = {}
    withheld: set[str] = set()
    raw = step.parameters.get("gpu_uuids_by_node")
    if isinstance(raw, dict):
        for node_id, values in raw.items():
            if isinstance(values, list):
                merged.setdefault(str(node_id), set()).update(
                    str(value) for value in values
                )
    elif step_gpu_shares(workflow, step) is None:
        withheld = set(step.node_ids)
    for node_id, values in gpu_uuids_by_node.items():
        if values and node_id not in withheld:
            merged.setdefault(node_id, set()).update(values)
    return {node_id: sorted(values) for node_id, values in sorted(merged.items())}


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
        """Per node, the GPUs a merge widens the pending node actions to.

        A node-action step's own ``gpu_uuids_by_node`` says where its GPUs
        are; without one, ``step_gpu_shares`` places them. Giving every node
        of a multi-node step the step's whole list sent a sibling's GPU to
        this node's agent (batch 155): when such a step's GPUs cannot all be
        placed, none of them is attributed to a node (fail closed, never
        guess), and its nodes still join the scope.
        """
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
                shares = step_gpu_shares(existing_workflow, step) or {}
                for existing_node in step.node_ids:
                    mapping.setdefault(existing_node, set()).update(
                        shares.get(existing_node, ())
                    )
        # A node with no GPU identity stays in scope with an empty list so
        # host-level steps still widen onto it (F-B6); GPU actions skip
        # such nodes because the agent refuses a node without GPUs.
        mapping.setdefault(node_id, set()).update(gpu_uuids)
        return {key: sorted(values) for key, values in mapping.items()}

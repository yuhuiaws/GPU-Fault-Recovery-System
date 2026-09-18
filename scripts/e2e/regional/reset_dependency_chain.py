"""Validate a reset's required chain without rejecting merged read-only work."""

from __future__ import annotations

from collections import Counter
from typing import Any

from gpu_fault.execution.config import MAX_DAG_STEPS
from gpu_fault.models import WorkflowOperation
from gpu_fault.operation_registry import OPERATION_REGISTRY


def _readonly(step: dict[str, Any]) -> bool:
    try:
        operation = WorkflowOperation(step.get("operation"))
    except (TypeError, ValueError):
        return False
    semantics = OPERATION_REGISTRY[operation]
    return bool(
        not semantics.destructive
        and not semantics.planning_only
        and not semantics.workload_scoped
        and (
            semantics.zero_rank_action
            or semantics.safe_waiting_preempt
            or semantics.safe_remote_waiting_preempt
            or operation is WorkflowOperation.FREEZE_EVIDENCE
        )
        and step.get("execution_owner")
    )


def reset_chain_errors(workflow: dict[str, Any], expected: list[str]) -> list[str]:
    steps = workflow.get("official_steps")
    if (
        not isinstance(steps, list)
        or not steps
        or any(not isinstance(step, dict) for step in steps)
    ):
        return ["reset contract has no complete step inventory"]
    if len(steps) > MAX_DAG_STEPS:
        return ["reset contract exceeds the bounded workflow step inventory"]
    dependencies: list[set[int]] = []
    for index, step in enumerate(steps):
        raw = step.get("depends_on_step_indexes", [])
        if not isinstance(raw, list) or any(
            type(value) is not int or not 0 <= value < len(steps) or value == index
            for value in raw
        ):
            return ["reset contract has invalid step dependencies"]
        dependencies.append(
            set(raw) if workflow.get("dag_enabled") else set(range(index))
        )
    ancestors: dict[int, set[int]] = {}

    def visit(index: int, visiting: set[int]) -> set[int]:
        if index in visiting:
            raise ValueError("reset contract dependency graph has a cycle")
        if index not in ancestors:
            parents = dependencies[index]
            ancestors[index] = parents | set().union(
                *(visit(parent, visiting | {index}) for parent in parents)
            )
        return ancestors[index]

    try:
        for index in range(len(steps)):
            visit(index, set())
    except (ValueError, RecursionError):
        return ["reset contract dependency graph has a cycle"]

    # Keep every possible prefix: a merged VALIDATE_GPU need not be the
    # validation that authorizes this reset's RESTORE_SCHEDULING.
    prefixes: list[tuple[int, ...]] = [()]
    for operation in expected:
        prefixes = [
            (*prefix, index)
            for prefix in prefixes
            for index, step in enumerate(steps)
            if step.get("operation") == operation
            and (not prefix or prefix[-1] in ancestors[index])
        ]
        if not prefixes:
            return ["workflow is missing the required reset contract dependency chain"]
        if len(prefixes) > 256:
            return ["reset contract has an ambiguous dependency chain"]
    errors = []
    chain: tuple[int, ...] | None = None
    for candidate in prefixes:
        if all(
            index in candidate or _readonly(step) for index, step in enumerate(steps)
        ):
            chain = candidate
            break
    if chain is None:
        return ["workflow contains an unauthorized operation beyond the reset contract"]
    reset = steps[chain[expected.index("RESET_GPU")]]
    reset_nodes = set(reset.get("node_ids") or [])
    reset_gpus = set(reset.get("gpu_uuids") or [])
    for index, step in enumerate(steps):
        if index in chain:
            continue
        if (
            not set(step.get("node_ids") or []) <= reset_nodes
            or not set(step.get("gpu_uuids") or []) <= reset_gpus
        ):
            errors.append("read-only reset branch exceeds the approved reset scope")
    operations = [step.get("operation") for step in steps]
    completed = workflow.get("completed_operations") or []
    if Counter(completed) != Counter(operations):
        errors.append(
            "completed operations do not cover the reset contract and branches"
        )
    if workflow.get("dag_enabled"):
        completed_indexes = workflow.get("completed_step_indexes") or []
        if set(completed_indexes) != set(range(len(steps))):
            errors.append("reset contract DAG has incomplete or retired steps")
        executions = workflow.get("step_executions") or []
        for index, step in enumerate(steps):
            matches = [
                item
                for item in executions
                if item.get("step_index") == index
                and item.get("operation") == step.get("operation")
                and item.get("phase") in {None, "official"}
            ]
            if not matches or matches[-1].get("status") != "SUCCEEDED":
                errors.append(f"reset contract step {index} has no successful receipt")
    return errors

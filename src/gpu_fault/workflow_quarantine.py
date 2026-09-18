"""Node readmission constraints carried by terminal quarantine steps."""

from __future__ import annotations

from collections.abc import Sequence

from gpu_fault.models import (
    StepPhase,
    WorkflowOperation,
    WorkflowRequest,
    WorkflowStatus,
    WorkflowStepExecution,
    WorkflowStepSpec,
    WorkflowStepStatus,
    execution_matches_step,
    execution_phase,
    resolved_step_indexes,
)
from gpu_fault.operation_registry import WORKLOAD_SCOPED_OPERATIONS
from gpu_fault.recovery_safety import unresolved_details

TERMINAL_QUARANTINE_NODES = "terminal_quarantine_node_ids"


def fixed_quarantine_nodes(step: WorkflowStepSpec) -> frozenset[str]:
    raw = step.parameters.get(TERMINAL_QUARANTINE_NODES, [])
    if (
        not isinstance(raw, list)
        or any(not isinstance(node, str) or not node for node in raw)
        or not set(raw) <= set(step.node_ids)
    ):
        raise ValueError("terminal quarantine scope is malformed")
    return frozenset(raw)


def dependency_ancestors(workflow: WorkflowRequest, step_index: int) -> frozenset[int]:
    """DAG indexes are stable identities, not necessarily execution order."""
    if workflow.dag_enabled:
        pending = [
            (index, False)
            for index in workflow.official_steps[step_index].depends_on_step_indexes
        ]
        ancestors: set[int] = set()
        visiting: set[int] = set()
        while pending:
            index, finished = pending.pop()
            if (
                index < 0
                or index >= len(workflow.official_steps)
                or index == step_index
            ):
                return frozenset()
            if finished:
                visiting.remove(index)
                ancestors.add(index)
            elif index in visiting:
                return frozenset()
            elif index not in ancestors:
                visiting.add(index)
                pending.append((index, True))
                pending.extend(
                    (dependency, False)
                    for dependency in workflow.official_steps[
                        index
                    ].depends_on_step_indexes
                )
    else:
        ancestors = set(range(step_index))
    return frozenset(ancestors)


def replacement_ancestors(workflow: WorkflowRequest, step_index: int) -> frozenset[str]:
    """Readmission after replacement targets the spare, never the failed node."""
    superseded = set(workflow.superseded_step_indexes)
    return frozenset(
        node
        for index in dependency_ancestors(workflow, step_index) - superseded
        if workflow.official_steps[index].operation is WorkflowOperation.REPLACE_NODE
        for node in workflow.official_steps[index].node_ids
    )


def terminal_quarantine_nodes(workflow: WorkflowRequest) -> frozenset[str]:
    quarantined: set[str] = set()
    explicit: set[str] = set()
    readmitted: set[str] = set()
    superseded = set(workflow.superseded_step_indexes)
    for index, step in enumerate(workflow.official_steps):
        if index in superseded:
            continue
        if step.operation is WorkflowOperation.RESTORE_SCHEDULING:
            readmitted.update(step.node_ids)
        if step.operation is not WorkflowOperation.QUARANTINE:
            continue
        quarantined.update(step.node_ids)
        explicit.update(fixed_quarantine_nodes(step))
    # Legacy plans express the same constraint by omitting readmission.
    return frozenset(explicit | (quarantined - readmitted))


def _completed_execution(
    workflow: WorkflowRequest, phase: StepPhase, index: int, step: WorkflowStepSpec
) -> WorkflowStepExecution | None:
    if index in workflow.superseded_step_indexes:
        return None
    records = [
        item
        for item in workflow.step_executions
        if execution_matches_step(item, index, step.operation, phase)
    ]
    if not records:
        return None
    latest = records[-1]
    if (
        latest.phase != phase
        or latest.status is not WorkflowStepStatus.SUCCEEDED
        or unresolved_details(latest.details)
    ):
        return None
    return latest


def _quarantine_evidence(
    workflow: WorkflowRequest,
) -> tuple[frozenset[str], frozenset[str], bool]:
    held = set(terminal_quarantine_nodes(workflow))
    explicit: set[str] = set()
    identified: set[int] = set()
    seen_quarantine = False
    phases: tuple[tuple[StepPhase, list[WorkflowStepSpec]], ...] = (
        ("official", workflow.official_steps),
        ("safety", workflow.safety_steps),
    )
    for phase, steps in phases:
        phase_plan = workflow.model_copy(update={"official_steps": steps})
        for index, step in enumerate(steps):
            if step.operation is not WorkflowOperation.QUARANTINE:
                continue
            records = {
                position
                for position, item in enumerate(workflow.step_executions)
                if execution_matches_step(item, index, step.operation, phase)
            }
            identified.update(records)
            if not records and (
                phase != execution_phase(workflow)
                or index in workflow.superseded_step_indexes
            ):
                continue
            if not step.node_ids or any(not node for node in step.node_ids):
                raise ValueError("quarantine node scope is missing")
            seen_quarantine = True
            explicit.update(fixed_quarantine_nodes(step))
            released = {
                node
                for restore_index, restore in enumerate(steps)
                if restore.operation is WorkflowOperation.RESTORE_SCHEDULING
                and index in dependency_ancestors(phase_plan, restore_index)
                and _completed_execution(workflow, phase, restore_index, restore)
                for node in restore.node_ids
            }
            held.update(set(step.node_ids) - released)
    unknown = any(
        item.operation is WorkflowOperation.QUARANTINE and position not in identified
        for position, item in enumerate(workflow.step_executions)
    ) or (
        WorkflowOperation.QUARANTINE in workflow.completed_operations
        and not seen_quarantine
    )
    return frozenset(held | explicit), frozenset(explicit), unknown


def has_terminal_quarantine_hold(workflow: WorkflowRequest) -> bool:
    """Original-node ownership survives a sibling restore or spare recovery."""
    try:
        held, _, unknown = _quarantine_evidence(workflow)
        return bool(held) or unknown
    except ValueError:
        return True


def _validated_spare_readmission(
    workflow: WorkflowRequest, replacement_index: int, node: str
) -> bool:
    validations = {
        WorkflowOperation.VALIDATE_GPU,
        WorkflowOperation.VALIDATE_HOST,
        WorkflowOperation.VALIDATE_FABRIC,
    }
    for index, step in enumerate(workflow.official_steps):
        if (
            step.operation is not WorkflowOperation.RESTORE_SCHEDULING
            or node not in step.node_ids
            or not _completed_execution(workflow, "official", index, step)
        ):
            continue
        ancestors = dependency_ancestors(workflow, index)
        checks = [
            (parent, workflow.official_steps[parent])
            for parent in ancestors
            if workflow.official_steps[parent].operation in validations
            and node in workflow.official_steps[parent].node_ids
            and replacement_index in dependency_ancestors(workflow, parent)
        ]
        if (
            replacement_index in ancestors
            and any(
                check.operation is WorkflowOperation.VALIDATE_GPU for _, check in checks
            )
            and all(
                _completed_execution(workflow, "official", parent, check)
                for parent, check in checks
            )
        ):
            return True
    return False


def _recovered_replacements(
    workflow: WorkflowRequest, held: frozenset[str]
) -> frozenset[str]:
    recovered: set[str] = set()
    for index, step in enumerate(workflow.official_steps):
        if step.operation is not WorkflowOperation.REPLACE_NODE:
            continue
        receipt = _completed_execution(workflow, "official", index, step)
        if receipt is None or receipt.details.get("action") != "SPARE_FAILOVER":
            continue
        bindings = receipt.details.get("node_rebindings")
        if (
            not isinstance(bindings, dict)
            or not bindings
            or set(bindings) != set(step.node_ids)
            or any(not isinstance(node, str) or not node for node in bindings.values())
        ):
            continue
        targets = set(bindings.values())
        if targets & (held | set(bindings)) or len(targets) != len(bindings):
            continue
        ancestors = dependency_ancestors(workflow, index)
        if any(
            quarantine.operation is WorkflowOperation.QUARANTINE
            and set(quarantine.node_ids).intersection(bindings)
            and parent not in ancestors
            for parent, quarantine in enumerate(workflow.official_steps)
        ) or any(
            item.operation is WorkflowOperation.QUARANTINE and item.phase == "safety"
            for item in workflow.step_executions
        ):
            continue
        if all(
            _validated_spare_readmission(workflow, index, spare) for spare in targets
        ):
            recovered.update(bindings)
    return frozenset(recovered)


def has_unrecovered_quarantine(workflow: WorkflowRequest) -> bool:
    """A proven spare recovery can recover the incident, never admit its original."""
    try:
        held, explicit, unknown = _quarantine_evidence(workflow)
        if unknown or explicit:
            return True
        if (
            workflow.status is WorkflowStatus.SUCCEEDED
            and not workflow.executes_safety_steps
        ):
            held -= _recovered_replacements(workflow, held)
        return bool(held)
    except ValueError:
        return True


def terminal_quarantine_covered(
    existing: WorkflowRequest, candidate: WorkflowRequest
) -> bool:
    required = terminal_quarantine_nodes(candidate)
    if not required:
        return True
    if not required <= terminal_quarantine_nodes(existing):
        return False
    resolved = resolved_step_indexes(existing)
    return not any(
        index not in resolved
        and step.operation is WorkflowOperation.RESTORE_SCHEDULING
        and (required - replacement_ancestors(existing, index)).intersection(
            step.node_ids
        )
        for index, step in enumerate(existing.official_steps)
    )


def preserves_terminal_quarantine(workflow: WorkflowRequest, node_id: str) -> bool:
    try:
        return node_id in terminal_quarantine_nodes(workflow) and (
            terminal_quarantine_covered(workflow, workflow)
        )
    except ValueError:
        return False


def suppress_readmission(
    existing: WorkflowRequest,
    steps: Sequence[WorkflowStepSpec],
    held_nodes: frozenset[str],
) -> tuple[list[WorkflowStepSpec], set[int]]:
    """Only unsubmitted work can be narrowed or retired; history stays intact."""
    immutable = resolved_step_indexes(existing) | {
        execution.step_index for execution in existing.step_executions
    }
    updated = list(steps)
    superseded: set[int] = set()
    for index, step in enumerate(steps):
        nodes = set(step.node_ids)
        if (
            index in immutable
            or step.operation
            not in {
                WorkflowOperation.RESTORE_SCHEDULING,
                WorkflowOperation.RESTART_WORKLOAD,
            }
            or not nodes.intersection(held_nodes)
        ):
            continue
        held = held_nodes - replacement_ancestors(existing, index)
        if not nodes.intersection(held):
            continue
        if step.operation is WorkflowOperation.RESTORE_SCHEDULING:
            remaining = nodes - held
            if remaining:
                parameters = dict(step.parameters)
                mapping = parameters.get("gpu_uuids_by_node")
                if isinstance(mapping, dict):
                    parameters["gpu_uuids_by_node"] = {
                        node: values
                        for node, values in mapping.items()
                        if node in remaining
                    }
                updated[index] = step.model_copy(
                    update={"node_ids": sorted(remaining), "parameters": parameters}
                )
            else:
                superseded.add(index)
        elif step.operation is WorkflowOperation.RESTART_WORKLOAD and nodes <= held:
            superseded.add(index)
    return updated, superseded


def _insert_quarantine(
    steps: list[WorkflowStepSpec],
    superseded: set[int],
    template: WorkflowStepSpec,
    nodes: frozenset[str],
) -> None:
    insert_at = next(
        (
            index
            for index, step in enumerate(steps)
            if step.operation
            not in {
                WorkflowOperation.FREEZE_EVIDENCE,
                WorkflowOperation.MARK_UNSCHEDULABLE,
            }
        ),
        len(steps),
    )
    for index, step in enumerate(steps):
        dependencies = {
            value + (value >= insert_at) for value in step.depends_on_step_indexes
        }
        if index >= insert_at:
            dependencies.add(insert_at)
        steps[index] = step.model_copy(
            update={"depends_on_step_indexes": sorted(dependencies)}
        )
    shifted = {index + (index >= insert_at) for index in superseded}
    superseded.clear()
    superseded.update(shifted)
    steps.insert(
        insert_at,
        template.model_copy(
            update={
                "node_ids": sorted(nodes),
                "gpu_uuids": [],
                "workload_ids": [],
                "parameters": {TERMINAL_QUARANTINE_NODES: sorted(nodes)},
                "depends_on_step_indexes": list(range(insert_at)),
                "branch_id": None,
                "branch_node_ids": [],
            }
        ),
    )


def inherit_terminal_quarantine(
    existing: WorkflowRequest, candidate: WorkflowRequest
) -> WorkflowRequest:
    """A subsequent recovery cannot silently undo a node's persistent hold."""
    required = terminal_quarantine_nodes(existing) | terminal_quarantine_nodes(
        candidate
    )
    if not required:
        return candidate
    steps, retired = suppress_readmission(candidate, candidate.official_steps, required)
    superseded = set(candidate.superseded_step_indexes) | retired
    covered: set[str] = set()
    immutable = resolved_step_indexes(candidate) | {
        execution.step_index for execution in candidate.step_executions
    }
    for index, step in enumerate(steps):
        if index in superseded:
            continue
        held = required.intersection(step.node_ids)
        if step.operation is WorkflowOperation.QUARANTINE:
            covered.update(held)
        fixed = held - replacement_ancestors(candidate, index)
        if (
            fixed
            and index not in immutable
            and step.operation not in WORKLOAD_SCOPED_OPERATIONS
        ):
            steps[index] = step.model_copy(
                update={
                    "parameters": {
                        **step.parameters,
                        TERMINAL_QUARANTINE_NODES: sorted(fixed),
                    }
                }
            )
    missing = required - covered
    for index, template in enumerate(existing.official_steps):
        nodes = missing.intersection(template.node_ids)
        if (
            index in existing.superseded_step_indexes
            or template.operation is not WorkflowOperation.QUARANTINE
            or not nodes
        ):
            continue
        if candidate.completed_step_indexes or candidate.step_executions:
            raise ValueError("cannot insert quarantine into an executed candidate")
        _insert_quarantine(steps, superseded, template, nodes)
        missing -= nodes
    if missing:
        raise ValueError("terminal quarantine has no executable owner")
    if steps == candidate.official_steps and superseded == set(
        candidate.superseded_step_indexes
    ):
        return candidate
    return candidate.model_copy(
        update={
            "official_steps": steps,
            "superseded_step_indexes": sorted(superseded),
        }
    )

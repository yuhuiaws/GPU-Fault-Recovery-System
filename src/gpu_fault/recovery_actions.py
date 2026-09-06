"""One table for what each ``RecoveryAction`` means to the rest of the system.

Two consumers used to keep their own copy of this knowledge: the passive
workflow compiler's action-to-operation map (:mod:`gpu_fault.passive`) and the
warm-spare selector's "which recommended actions mean this node must not take
work" set (:mod:`gpu_fault.markers`). Each was extended by hand and each missed
the same three actions -- ``REMEDIATE_EFA_DRIVER``, ``RESTART_EFA_DEVICE_PLUGIN``
and ``RESTART_GPU_DEVICE_PLUGIN`` -- so a plan naming them failed to compile
and a node carrying such a marker was handed out as a healthy spare
(FINAL-建议汇总 F-G5).

Both are now derived from this table. Whether an action blocks spare reuse is
not a second hand-maintained flag either: it follows from the traits the
operation registry already declares for the operation the action compiles to.
"""

from __future__ import annotations

from dataclasses import dataclass

from gpu_fault.models import RecoveryAction, WorkflowOperation
from gpu_fault.operation_registry import (
    DESTRUCTIVE_OPERATIONS,
    NODE_ACTION_SCOPE_OPERATIONS,
    NODE_WIDE_RECOVERY_OPERATIONS,
)


@dataclass(frozen=True)
class RecoveryActionProfile:
    #: The workflow operation a plan step with this action compiles to;
    #: ``None`` for the one action that is not a step (``NO_ACTION``).
    operation: WorkflowOperation | None
    #: Whether a live, trusted marker recommending this action means the node
    #: must not be reused as a warm spare until the incident is recovered.
    blocks_spare: bool


def _blocks_spare(operation: WorkflowOperation | None) -> bool:
    """A node is unfit for reuse while any node-level remediation is pending.

    That is every node-wide recovery operation (reboot, replace, quarantine,
    EFA/device-plugin remediation, support escalation), every destructive
    node-scoped action (GPU reset), and the cordon itself. Advisory work
    (evidence, diagnostics, validation), releasing the cordon, and anything
    scoped to the workload rather than the node do not disqualify it.
    """
    if operation is None:
        return False
    if operation is WorkflowOperation.MARK_UNSCHEDULABLE:
        return True
    if operation in NODE_WIDE_RECOVERY_OPERATIONS:
        return True
    return (
        operation in NODE_ACTION_SCOPE_OPERATIONS
        and operation in DESTRUCTIVE_OPERATIONS
    )


_OPERATIONS: dict[RecoveryAction, WorkflowOperation | None] = {
    RecoveryAction.NO_ACTION: None,
    RecoveryAction.MARK_UNSCHEDULABLE: WorkflowOperation.MARK_UNSCHEDULABLE,
    RecoveryAction.DRAIN: WorkflowOperation.MARK_UNSCHEDULABLE,
    RecoveryAction.STOP_WORKLOAD: WorkflowOperation.STOP_WORKLOADS,
    RecoveryAction.COLLECT_EVIDENCE: WorkflowOperation.FREEZE_EVIDENCE,
    RecoveryAction.RESTART_WORKLOAD: WorkflowOperation.RESTART_WORKLOAD,
    RecoveryAction.RESET_GPU: WorkflowOperation.RESET_GPU,
    RecoveryAction.REBOOT_NODE: WorkflowOperation.RESTART_NODE,
    RecoveryAction.REPLACE_NODE: WorkflowOperation.REPLACE_NODE,
    RecoveryAction.REMEDIATE_EFA_DRIVER: WorkflowOperation.REMEDIATE_EFA_DRIVER,
    RecoveryAction.RESTART_EFA_DEVICE_PLUGIN: (
        WorkflowOperation.RESTART_EFA_DEVICE_PLUGIN
    ),
    RecoveryAction.RESTART_GPU_DEVICE_PLUGIN: (
        WorkflowOperation.RESTART_GPU_DEVICE_PLUGIN
    ),
    RecoveryAction.RUN_DIAGNOSTICS: WorkflowOperation.VALIDATE_GPU,
    RecoveryAction.VALIDATE_NODE: WorkflowOperation.VALIDATE_GPU,
    RecoveryAction.RESTORE_SCHEDULING: WorkflowOperation.RESTORE_SCHEDULING,
    RecoveryAction.QUARANTINE: WorkflowOperation.QUARANTINE,
    RecoveryAction.ESCALATE_OPERATOR: WorkflowOperation.ESCALATE_SUPPORT,
}

if set(_OPERATIONS) != set(RecoveryAction):  # pragma: no cover - import guard
    raise RuntimeError(
        "recovery action table is incomplete: "
        + ", ".join(
            sorted(action.value for action in set(RecoveryAction) - set(_OPERATIONS))
        )
    )

RECOVERY_ACTION_PROFILES: dict[RecoveryAction, RecoveryActionProfile] = {
    action: RecoveryActionProfile(
        operation=operation, blocks_spare=_blocks_spare(operation)
    )
    for action, operation in _OPERATIONS.items()
}

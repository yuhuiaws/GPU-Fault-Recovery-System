"""The rung a FAILED validation climbs, by the recovery step it judged.

A validation judges the recovery step its record ran before it. The
whole-record classifier (``escalation.HardwareEscalationService._classify``)
climbs the rung of the strongest such step (:data:`VALIDATION_RUNGS`) and
narrows the failure to the nodes that step acted on (``escalation._acted_nodes``
reads :data:`VALIDATION_JUDGED_OPERATIONS`). Every node recovery action the
registry knows (:data:`NODE_RECOVERY_OPERATIONS`) has a row; the guard in
``tests/orchestration/test_cov95_runtime_escalation.py`` fails when one is
added without it. Until it had one, a validation failing after a Fabric
Manager restart climbed nothing and the record ended silently.
"""

from __future__ import annotations

from gpu_fault.models import RecoveryAction, WorkflowOperation
from gpu_fault.operation_registry import (
    NODE_HARDWARE_MUTATING_OPERATIONS,
    RECOVERY_OPERATION_RANK,
)

#: The resets. A validation behind one climbs to the reboot, as does a failed
#: RESET_GPU or RESET_ALL_GPUS_NVSWITCHES (a failed RECOVER_DRAIN_P2P only
#: when its evidence asks for the reboot, ``escalation.next_rung``).
RESET_OPERATIONS = frozenset(
    {
        WorkflowOperation.RESET_GPU,
        WorkflowOperation.RESET_ALL_GPUS_NVSWITCHES,
        WorkflowOperation.RECOVER_DRAIN_P2P,
    }
)
#: The node recovery actions: the operations the registry calls
#: node-hardware-mutating -- they act on the node's GPUs, fabric, drivers,
#: device plugins or lifecycle (``NODE_HARDWARE_MUTATING_OPERATIONS``) -- and
#: ranks as a recovery (``RECOVERY_OPERATION_RANK``): the resets, the Fabric
#: Manager restart, the EFA driver and device-plugin restarts, the driver and
#: firmware installs, the reboot and the replacement. The GPU-service window
#: around a reset (QUIESCE / RESTORE_GPU_SERVICES, containment and release to
#: the classifier) and the planning-only RESTART_VM carry no rank. The
#: validation adapter re-reads telemetry after exactly this set
#: (``adapters/gpu_validation.TELEMETRY_READINESS_OPERATIONS``).
NODE_RECOVERY_OPERATIONS = NODE_HARDWARE_MUTATING_OPERATIONS & frozenset(
    RECOVERY_OPERATION_RANK
)
#: The stage of a FAILED step whose plan asks for the reboot
#: (``failure_escalation_action``: the EFA driver, device-plugin and EFA
#: diagnostic chains), and of a validation failing after an EFA driver or
#: device-plugin restart (:data:`VALIDATION_RUNGS`).
EFA_RECOVERY_STAGE = "efa_recovery"
#: ``(stage, operations, next_action, next_operation)``: the first row naming
#: an operation of the record's recovery context wins. The ladder is reset ->
#: reboot -> warm-spare replacement -> support (``escalation.next_rung``); a
#: driver or firmware install, the field diagnostic or the NVLink workflow goes
#: to support (the software may be half updated, the diagnostic's verdict is
#: the vendor's), a DCGM diagnostic to the drain. The runtime restarts climb
#: like a reset: the Fabric Manager restart (``RESTART_NODE`` dominates it as it
#: dominates the resets, ``operation_registry``), and the EFA driver and
#: device-plugin restarts, whose plans ask for that reboot themselves
#: (``failure_escalation_action``, ``families/health._resource_escalation``)
#: and which climb to it as well when a plan carries no request. A validation
#: with none of these before it is no recovery's verdict and has no rung.
VALIDATION_RUNGS: tuple[
    tuple[str, frozenset[WorkflowOperation], RecoveryAction, WorkflowOperation | None],
    ...,
] = (
    (
        "replacement_validation",
        frozenset({WorkflowOperation.REPLACE_NODE}),
        RecoveryAction.ESCALATE_OPERATOR,
        WorkflowOperation.ESCALATE_SUPPORT,
    ),
    (
        "reboot_validation",
        frozenset({WorkflowOperation.RESTART_NODE}),
        RecoveryAction.REPLACE_NODE,
        WorkflowOperation.REPLACE_NODE,
    ),
    (
        "reset_validation",
        RESET_OPERATIONS,
        RecoveryAction.REBOOT_NODE,
        WorkflowOperation.RESTART_NODE,
    ),
    (
        "remediation_validation",
        frozenset(
            {
                WorkflowOperation.REMEDIATE_DRIVER,
                WorkflowOperation.UPDATE_SOFTWARE_FIRMWARE,
                WorkflowOperation.RUN_FIELD_DIAGNOSTIC,
                WorkflowOperation.RUN_NVLINK74_WORKFLOW,
            }
        ),
        RecoveryAction.ESCALATE_OPERATOR,
        WorkflowOperation.ESCALATE_SUPPORT,
    ),
    (
        "temperature_validation",
        frozenset({WorkflowOperation.RUN_DCGM_DIAGNOSTIC}),
        RecoveryAction.DRAIN,
        None,
    ),
    (
        "fabric_manager_validation",
        frozenset({WorkflowOperation.RESTART_FABRIC_MANAGER}),
        RecoveryAction.REBOOT_NODE,
        WorkflowOperation.RESTART_NODE,
    ),
    (
        EFA_RECOVERY_STAGE,
        frozenset(
            {
                WorkflowOperation.REMEDIATE_EFA_DRIVER,
                WorkflowOperation.RESTART_EFA_DEVICE_PLUGIN,
                WorkflowOperation.RESTART_GPU_DEVICE_PLUGIN,
            }
        ),
        RecoveryAction.REBOOT_NODE,
        WorkflowOperation.RESTART_NODE,
    ),
)
#: Every recovery step :data:`VALIDATION_RUNGS` judges a failed validation
#: against: the node recovery actions and the diagnostics.
VALIDATION_JUDGED_OPERATIONS: frozenset[WorkflowOperation] = frozenset(
    operation
    for _stage, operations, _action, _operation in VALIDATION_RUNGS
    for operation in operations
)


def validation_rung(
    recovery_context: set[WorkflowOperation] | frozenset[WorkflowOperation],
) -> tuple[str, RecoveryAction, WorkflowOperation | None] | None:
    """``(stage, next_action, next_operation)`` of a FAILED validation whose
    record ran the recovery steps in ``recovery_context``, or ``None`` when no
    recovery step ran before it."""

    for stage, operations, next_action, next_operation in VALIDATION_RUNGS:
        if operations & recovery_context:
            return stage, next_action, next_operation
    return None

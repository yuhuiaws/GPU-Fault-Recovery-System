from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from gpu_fault.models import (
    CapabilityName,
    WorkflowOperation,
)


class OperationScope(StrEnum):
    CONTROL_PLANE = "CONTROL_PLANE"
    WORKLOAD = "WORKLOAD"
    NODE = "NODE"
    GPU = "GPU"
    FABRIC = "FABRIC"
    VALIDATION = "VALIDATION"
    SUPPORT = "SUPPORT"


class OperationAdapter(StrEnum):
    CONTROL_PLANE = "control-plane"
    KUBERNETES = "kubernetes"
    NODE_ACTION = "node-action"
    GPU_VALIDATION = "gpu-validation"
    HYPERPOD = "hyperpod"
    MANAGED_RECOVERY = "managed-recovery"
    SUPPORT = "support"


class OperationResourceClaim(StrEnum):
    GPU_RUNTIME_MUTATION = "GPU_RUNTIME_MUTATION"
    EFA_RUNTIME_MUTATION = "EFA_RUNTIME_MUTATION"
    EFA_DEVICE_PLUGIN_MUTATION = "EFA_DEVICE_PLUGIN_MUTATION"
    GPU_DEVICE_PLUGIN_MUTATION = "GPU_DEVICE_PLUGIN_MUTATION"
    NODE_LIFECYCLE_MUTATION = "NODE_LIFECYCLE_MUTATION"
    SCHEDULER_MUTATION = "SCHEDULER_MUTATION"


@dataclass(frozen=True)
class OperationSemantics:
    capability: CapabilityName
    scope: OperationScope
    adapters: frozenset[OperationAdapter]
    destructive: bool = False
    recovery_rank: int = 0
    merge_intent: bool = False
    node_wide: bool = False
    zero_rank_action: bool = False
    node_action_scope: bool = False
    workload_scoped: bool = False
    shared_dag: bool = False
    node_exclusive: bool = False
    resource_claims: frozenset[OperationResourceClaim] = field(
        default_factory=frozenset
    )
    safe_waiting_preempt: bool = False
    safe_remote_waiting_preempt: bool = False
    multi_node_barrier: bool = False
    transient_gpu_inventory_recovery: bool = False
    hardware_escalation_relevant: bool | None = None
    host_proc_root_dependent: bool = False
    maintenance_generation_scoped: bool = False
    generation_stable_command_id: bool = False
    planning_only: bool = False
    dominates: frozenset[WorkflowOperation] = field(default_factory=frozenset)


def _semantics(
    capability: CapabilityName,
    scope: OperationScope,
    *adapters: OperationAdapter,
    **values,
) -> OperationSemantics:
    return OperationSemantics(
        capability=capability,
        scope=scope,
        adapters=frozenset(adapters),
        **values,
    )


OPERATION_REGISTRY: dict[WorkflowOperation, OperationSemantics] = {
    WorkflowOperation.FREEZE_EVIDENCE: _semantics(
        CapabilityName.EVIDENCE_CAPTURE,
        OperationScope.CONTROL_PLANE,
        OperationAdapter.CONTROL_PLANE,
    ),
    WorkflowOperation.COLLECT_HUNG_TRIAGE: _semantics(
        CapabilityName.DIAGNOSTIC_BUNDLE_CAPTURE,
        OperationScope.NODE,
        OperationAdapter.NODE_ACTION,
        zero_rank_action=True,
        safe_remote_waiting_preempt=True,
        hardware_escalation_relevant=False,
    ),
    WorkflowOperation.COLLECT_DIAGNOSTIC_BUNDLE: _semantics(
        CapabilityName.DIAGNOSTIC_BUNDLE_CAPTURE,
        OperationScope.NODE,
        OperationAdapter.NODE_ACTION,
        zero_rank_action=True,
        node_action_scope=True,
        safe_remote_waiting_preempt=True,
        hardware_escalation_relevant=False,
    ),
    WorkflowOperation.RUN_DCGM_DIAGNOSTIC: _semantics(
        CapabilityName.DIAGNOSTIC_BUNDLE_CAPTURE,
        OperationScope.GPU,
        OperationAdapter.NODE_ACTION,
        recovery_rank=10,
        merge_intent=True,
        resource_claims=frozenset({OperationResourceClaim.GPU_RUNTIME_MUTATION}),
        safe_remote_waiting_preempt=True,
        hardware_escalation_relevant=True,
    ),
    WorkflowOperation.MARK_UNSCHEDULABLE: _semantics(
        CapabilityName.SCHEDULER_DRAIN,
        OperationScope.NODE,
        OperationAdapter.KUBERNETES,
        destructive=True,
        merge_intent=True,
        node_exclusive=True,
        hardware_escalation_relevant=False,
        resource_claims=frozenset({OperationResourceClaim.SCHEDULER_MUTATION}),
    ),
    WorkflowOperation.CHECKPOINT_WORKLOADS: _semantics(
        CapabilityName.CHECKPOINT_RESTORE,
        OperationScope.WORKLOAD,
        OperationAdapter.KUBERNETES,
        workload_scoped=True,
        shared_dag=True,
    ),
    WorkflowOperation.STOP_WORKLOADS: _semantics(
        CapabilityName.WORKLOAD_STOP,
        OperationScope.WORKLOAD,
        OperationAdapter.KUBERNETES,
        OperationAdapter.MANAGED_RECOVERY,
        destructive=True,
        merge_intent=True,
        workload_scoped=True,
        shared_dag=True,
        hardware_escalation_relevant=False,
    ),
    WorkflowOperation.QUIESCE_GPU_SERVICES: _semantics(
        CapabilityName.GPU_RESET,
        OperationScope.GPU,
        OperationAdapter.NODE_ACTION,
        destructive=True,
        node_action_scope=True,
        node_exclusive=True,
        hardware_escalation_relevant=False,
        resource_claims=frozenset({OperationResourceClaim.GPU_RUNTIME_MUTATION}),
        host_proc_root_dependent=True,
    ),
    WorkflowOperation.VERIFY_NO_GPU_CLIENTS: _semantics(
        CapabilityName.GPU_RESET,
        OperationScope.GPU,
        OperationAdapter.NODE_ACTION,
        node_action_scope=True,
        resource_claims=frozenset({OperationResourceClaim.GPU_RUNTIME_MUTATION}),
        safe_remote_waiting_preempt=True,
        host_proc_root_dependent=True,
        maintenance_generation_scoped=True,
        hardware_escalation_relevant=False,
    ),
    WorkflowOperation.TRIGGER_HEALTH_SNAPSHOT: _semantics(
        CapabilityName.DIAGNOSTIC_BUNDLE_CAPTURE,
        OperationScope.NODE,
        OperationAdapter.NODE_ACTION,
        zero_rank_action=True,
        hardware_escalation_relevant=False,
    ),
    WorkflowOperation.RESET_GPU: _semantics(
        CapabilityName.GPU_RESET,
        OperationScope.GPU,
        OperationAdapter.NODE_ACTION,
        destructive=True,
        recovery_rank=30,
        merge_intent=True,
        node_action_scope=True,
        node_exclusive=True,
        resource_claims=frozenset({OperationResourceClaim.GPU_RUNTIME_MUTATION}),
        multi_node_barrier=True,
        hardware_escalation_relevant=True,
        host_proc_root_dependent=True,
        maintenance_generation_scoped=True,
    ),
    WorkflowOperation.RESTORE_GPU_SERVICES: _semantics(
        CapabilityName.GPU_RESET,
        OperationScope.GPU,
        OperationAdapter.NODE_ACTION,
        destructive=True,
        node_action_scope=True,
        node_exclusive=True,
        hardware_escalation_relevant=False,
        resource_claims=frozenset({OperationResourceClaim.GPU_RUNTIME_MUTATION}),
        maintenance_generation_scoped=True,
    ),
    WorkflowOperation.RESTART_NODE: _semantics(
        CapabilityName.NODE_REBOOT,
        OperationScope.NODE,
        OperationAdapter.HYPERPOD,
        OperationAdapter.MANAGED_RECOVERY,
        destructive=True,
        recovery_rank=50,
        merge_intent=True,
        node_wide=True,
        node_exclusive=True,
        resource_claims=frozenset({OperationResourceClaim.NODE_LIFECYCLE_MUTATION}),
        transient_gpu_inventory_recovery=True,
        dominates=frozenset(
            {
                WorkflowOperation.RESET_GPU,
                WorkflowOperation.RESET_ALL_GPUS_NVSWITCHES,
                WorkflowOperation.RESTART_FABRIC_MANAGER,
                WorkflowOperation.REMEDIATE_EFA_DRIVER,
                WorkflowOperation.RESTART_EFA_DEVICE_PLUGIN,
                WorkflowOperation.RESTART_GPU_DEVICE_PLUGIN,
            }
        ),
        hardware_escalation_relevant=True,
    ),
    WorkflowOperation.REPLACE_NODE: _semantics(
        CapabilityName.NODE_REPLACE,
        OperationScope.NODE,
        OperationAdapter.HYPERPOD,
        OperationAdapter.MANAGED_RECOVERY,
        destructive=True,
        recovery_rank=75,  # strictly above the rank-70 remediations it dominates
        merge_intent=True,
        node_wide=True,
        node_exclusive=True,
        resource_claims=frozenset({OperationResourceClaim.NODE_LIFECYCLE_MUTATION}),
        transient_gpu_inventory_recovery=True,
        dominates=frozenset(
            {
                WorkflowOperation.RESET_GPU,
                WorkflowOperation.RESET_ALL_GPUS_NVSWITCHES,
                WorkflowOperation.RESTART_FABRIC_MANAGER,
                WorkflowOperation.RESTART_NODE,
                WorkflowOperation.REMEDIATE_DRIVER,
                WorkflowOperation.REMEDIATE_EFA_DRIVER,
                WorkflowOperation.RESTART_EFA_DEVICE_PLUGIN,
                WorkflowOperation.RESTART_GPU_DEVICE_PLUGIN,
                WorkflowOperation.UPDATE_SOFTWARE_FIRMWARE,
            }
        ),
        hardware_escalation_relevant=True,
    ),
    WorkflowOperation.RESTART_VM: _semantics(
        CapabilityName.VM_RESTART,
        OperationScope.NODE,
        destructive=True,
        planning_only=True,
        hardware_escalation_relevant=False,
    ),
    WorkflowOperation.RESTART_FABRIC_MANAGER: _semantics(
        CapabilityName.FABRIC_MANAGER_RESTART,
        OperationScope.GPU,
        OperationAdapter.NODE_ACTION,
        destructive=True,
        recovery_rank=10,
        merge_intent=True,
        node_exclusive=True,
        hardware_escalation_relevant=False,
        resource_claims=frozenset({OperationResourceClaim.GPU_RUNTIME_MUTATION}),
    ),
    WorkflowOperation.REMEDIATE_EFA_DRIVER: _semantics(
        CapabilityName.EFA_DRIVER_REMEDIATION,
        OperationScope.NODE,
        OperationAdapter.NODE_ACTION,
        destructive=True,
        recovery_rank=35,
        merge_intent=True,
        node_wide=True,
        node_action_scope=True,
        node_exclusive=True,
        resource_claims=frozenset({OperationResourceClaim.EFA_RUNTIME_MUTATION}),
        dominates=frozenset({WorkflowOperation.RESTART_EFA_DEVICE_PLUGIN}),
        hardware_escalation_relevant=True,
        generation_stable_command_id=True,
    ),
    WorkflowOperation.RESTART_EFA_DEVICE_PLUGIN: _semantics(
        CapabilityName.SCHEDULER_DRAIN,
        OperationScope.NODE,
        OperationAdapter.KUBERNETES,
        destructive=True,
        recovery_rank=15,
        merge_intent=True,
        node_wide=True,
        node_exclusive=True,
        resource_claims=frozenset({OperationResourceClaim.EFA_DEVICE_PLUGIN_MUTATION}),
        hardware_escalation_relevant=True,
    ),
    WorkflowOperation.RESTART_GPU_DEVICE_PLUGIN: _semantics(
        CapabilityName.SCHEDULER_DRAIN,
        OperationScope.NODE,
        OperationAdapter.KUBERNETES,
        destructive=True,
        recovery_rank=15,
        merge_intent=True,
        node_wide=True,
        node_exclusive=True,
        resource_claims=frozenset({OperationResourceClaim.GPU_DEVICE_PLUGIN_MUTATION}),
        hardware_escalation_relevant=True,
    ),
    WorkflowOperation.RUN_FIELD_DIAGNOSTIC: _semantics(
        CapabilityName.MEMORY_DIAGNOSTICS,
        OperationScope.GPU,
        OperationAdapter.NODE_ACTION,
        recovery_rank=10,
        merge_intent=True,
        safe_remote_waiting_preempt=True,
        hardware_escalation_relevant=True,
    ),
    WorkflowOperation.RUN_NVLINK74_WORKFLOW: _semantics(
        CapabilityName.NVLINK_DIAGNOSTICS,
        OperationScope.FABRIC,
        OperationAdapter.NODE_ACTION,
        safe_remote_waiting_preempt=True,
        hardware_escalation_relevant=True,
    ),
    WorkflowOperation.RESET_ALL_GPUS_NVSWITCHES: _semantics(
        CapabilityName.FABRIC_RESET,
        OperationScope.FABRIC,
        OperationAdapter.NODE_ACTION,
        destructive=True,
        recovery_rank=40,
        merge_intent=True,
        node_wide=True,
        node_action_scope=True,
        node_exclusive=True,
        resource_claims=frozenset({OperationResourceClaim.GPU_RUNTIME_MUTATION}),
        multi_node_barrier=True,
        dominates=frozenset(
            {
                WorkflowOperation.RESET_GPU,
                WorkflowOperation.RESTART_FABRIC_MANAGER,
            }
        ),
        hardware_escalation_relevant=True,
        host_proc_root_dependent=True,
        maintenance_generation_scoped=True,
    ),
    WorkflowOperation.CHECK_MECHANICALS: _semantics(
        CapabilityName.MECHANICAL_INSPECTION,
        OperationScope.NODE,
        OperationAdapter.KUBERNETES,
        recovery_rank=80,
        merge_intent=True,
        node_wide=True,
        safe_remote_waiting_preempt=True,
    ),
    WorkflowOperation.REMEDIATE_DRIVER: _semantics(
        CapabilityName.DRIVER_REMEDIATION,
        OperationScope.GPU,
        OperationAdapter.NODE_ACTION,
        destructive=True,
        recovery_rank=70,
        merge_intent=True,
        node_wide=True,
        node_action_scope=True,
        node_exclusive=True,
        resource_claims=frozenset({OperationResourceClaim.GPU_RUNTIME_MUTATION}),
        hardware_escalation_relevant=True,
        host_proc_root_dependent=True,
        generation_stable_command_id=True,
    ),
    WorkflowOperation.UPDATE_SOFTWARE_FIRMWARE: _semantics(
        CapabilityName.SOFTWARE_FIRMWARE_UPDATE,
        OperationScope.GPU,
        OperationAdapter.NODE_ACTION,
        destructive=True,
        recovery_rank=70,
        merge_intent=True,
        node_wide=True,
        node_action_scope=True,
        node_exclusive=True,
        resource_claims=frozenset({OperationResourceClaim.GPU_RUNTIME_MUTATION}),
        hardware_escalation_relevant=True,
        host_proc_root_dependent=True,
        generation_stable_command_id=True,
    ),
    WorkflowOperation.ESCALATE_SUPPORT: _semantics(
        CapabilityName.SUPPORT_ESCALATION,
        OperationScope.SUPPORT,
        OperationAdapter.SUPPORT,
        recovery_rank=80,
        merge_intent=True,
        node_wide=True,
    ),
    WorkflowOperation.VALIDATE_GPU: _semantics(
        CapabilityName.DEEP_DIAGNOSTICS,
        OperationScope.VALIDATION,
        OperationAdapter.GPU_VALIDATION,
        zero_rank_action=True,
        safe_waiting_preempt=True,
        hardware_escalation_relevant=True,
    ),
    WorkflowOperation.VALIDATE_HOST: _semantics(
        CapabilityName.DEEP_DIAGNOSTICS,
        OperationScope.VALIDATION,
        OperationAdapter.GPU_VALIDATION,
        zero_rank_action=True,
        safe_waiting_preempt=True,
    ),
    WorkflowOperation.VALIDATE_FABRIC: _semantics(
        CapabilityName.DEEP_DIAGNOSTICS,
        OperationScope.VALIDATION,
        OperationAdapter.GPU_VALIDATION,
        zero_rank_action=True,
        safe_waiting_preempt=True,
        hardware_escalation_relevant=True,
    ),
    WorkflowOperation.RESTORE_SCHEDULING: _semantics(
        CapabilityName.SCHEDULER_DRAIN,
        OperationScope.NODE,
        OperationAdapter.KUBERNETES,
        destructive=True,
        node_exclusive=True,
        hardware_escalation_relevant=False,
        resource_claims=frozenset({OperationResourceClaim.SCHEDULER_MUTATION}),
    ),
    WorkflowOperation.RESTART_WORKLOAD: _semantics(
        CapabilityName.WORKLOAD_RESTART,
        OperationScope.WORKLOAD,
        OperationAdapter.KUBERNETES,
        OperationAdapter.MANAGED_RECOVERY,
        destructive=True,
        recovery_rank=20,
        merge_intent=True,
        workload_scoped=True,
        shared_dag=True,
        hardware_escalation_relevant=False,
    ),
    WorkflowOperation.QUARANTINE: _semantics(
        CapabilityName.SCHEDULER_DRAIN,
        OperationScope.NODE,
        OperationAdapter.KUBERNETES,
        destructive=True,
        recovery_rank=60,
        merge_intent=True,
        node_wide=True,
        node_exclusive=True,
        hardware_escalation_relevant=False,
        resource_claims=frozenset({OperationResourceClaim.SCHEDULER_MUTATION}),
    ),
    # An operator's remote ``gpu-fault-collector outbox`` (ARCH-G2): reads or
    # requeues a collector's dead letters on the node. Metadata only, no GPU
    # or EFA claim, not node-exclusive, allowed while workloads run; a
    # workflow of nothing but this step is diagnostic-only to the executor.
    WorkflowOperation.COLLECTOR_OUTBOX_MAINTENANCE: _semantics(
        CapabilityName.DIAGNOSTIC_BUNDLE_CAPTURE,
        OperationScope.NODE,
        OperationAdapter.NODE_ACTION,
        zero_rank_action=True,
        safe_remote_waiting_preempt=True,
        hardware_escalation_relevant=False,
    ),
}


def validate_operation_registry() -> None:
    operations = set(WorkflowOperation)
    registered = set(OPERATION_REGISTRY)
    if operations != registered:
        missing = sorted(item.value for item in operations - registered)
        extra = sorted(item.value for item in registered - operations)
        raise RuntimeError(
            f"WorkflowOperation registry mismatch: missing={missing} extra={extra}"
        )
    for operation, semantics in OPERATION_REGISTRY.items():
        if not semantics.adapters and not semantics.planning_only:
            raise RuntimeError(
                f"{operation.value} has no adapter and is not planning-only"
            )
        if semantics.recovery_rank < 0:
            raise RuntimeError(f"{operation.value} has a negative recovery rank")
        if semantics.destructive or OperationAdapter.NODE_ACTION in semantics.adapters:
            missing_risk_fields = [
                name
                for name in ("hardware_escalation_relevant",)
                if getattr(semantics, name) is None
            ]
            if missing_risk_fields:
                raise RuntimeError(
                    f"{operation.value} must explicitly declare "
                    + ", ".join(missing_risk_fields)
                )
        if semantics.generation_stable_command_id:
            if OperationAdapter.NODE_ACTION not in semantics.adapters:
                raise RuntimeError(
                    f"{operation.value} names a generation-stable command_id "
                    "but is not a node action"
                )
            if semantics.maintenance_generation_scoped:
                raise RuntimeError(
                    f"{operation.value} is maintenance-generation scoped, which "
                    "already pins its command_id; drop generation_stable_command_id"
                )
            if not _node_mutating(semantics):
                # The stable id trades "re-run on the fresh agent" for "answer
                # from the ledger". That is only right for an action whose
                # second run damages the node; a diagnostic or a containment
                # step wants the generation suffix and the re-run.
                raise RuntimeError(
                    f"{operation.value} names a generation-stable command_id "
                    "but is not node-mutating"
                )
        unknown = semantics.dominates - operations
        if unknown:
            raise RuntimeError(
                f"{operation.value} dominates unknown operations: "
                f"{sorted(item.value for item in unknown)}"
            )
    _validate_dominance(operations)


def _node_mutating(semantics: OperationSemantics) -> bool:
    """Whether the operation acts on the node beyond scheduler containment.

    The per-semantics form of ``NODE_MUTATING_OPERATIONS``, which is derived
    from the registry after validation and so cannot be consulted by it.
    """

    return semantics.destructive and semantics.resource_claims != frozenset(
        {OperationResourceClaim.SCHEDULER_MUTATION}
    )


def _validate_dominance(operations: set[WorkflowOperation]) -> None:
    """Dominance must agree with rank, be transitively closed and acyclic.

    The arbiter compares ranks first and dominance second; a tie between the
    two ends of a declared dominance let either side win, so the incident's
    declared action and the executed action could disagree (P0-67A). Checked
    at import so a registry edit cannot reintroduce it (F-C5).
    """

    dominance = {
        operation: OPERATION_REGISTRY[operation].dominates for operation in operations
    }
    for operation, dominated_set in dominance.items():
        rank = OPERATION_REGISTRY[operation].recovery_rank
        for dominated in dominated_set:
            if not rank > OPERATION_REGISTRY[dominated].recovery_rank:
                raise RuntimeError(
                    f"{operation.value} dominates {dominated.value} without a "
                    "strictly higher recovery rank"
                )
            for grandchild in dominance[dominated]:
                if grandchild not in dominated_set:
                    raise RuntimeError(
                        f"{operation.value} dominates {dominated.value} but not "
                        f"{grandchild.value}: dominance is not transitively closed"
                    )
        frontier = set(dominated_set)
        seen: set[WorkflowOperation] = set()
        while frontier:
            current = frontier.pop()
            if current is operation:
                raise RuntimeError(f"{operation.value} dominates itself transitively")
            if current in seen:
                continue
            seen.add(current)
            frontier |= dominance[current]


def operations_for_adapter(
    adapter: OperationAdapter,
) -> frozenset[WorkflowOperation]:
    return frozenset(
        operation
        for operation, semantics in OPERATION_REGISTRY.items()
        if adapter in semantics.adapters
    )


def operations_with(
    attribute: str,
) -> frozenset[WorkflowOperation]:
    return frozenset(
        operation
        for operation, semantics in OPERATION_REGISTRY.items()
        if getattr(semantics, attribute)
    )


OPERATION_CAPABILITY = {
    operation: semantics.capability
    for operation, semantics in OPERATION_REGISTRY.items()
}
DESTRUCTIVE_OPERATIONS = operations_with("destructive")
# Containment is destructive in the audit sense -- it changes what the
# scheduler is allowed to place on the node -- but it does not touch the
# GPU runtime, the driver, the node lifecycle or a workload. Gates that
# ask "may this plan act on the node's current state?" must not count a
# cordon or a quarantine taint as such an act: refusing to isolate a node
# because its workload state is unknown, or because the evidence
# describes an earlier boot, withdraws protection instead of adding it.
CONTAINMENT_ONLY_OPERATIONS = frozenset(
    operation
    for operation, semantics in OPERATION_REGISTRY.items()
    if semantics.destructive and not _node_mutating(semantics)
)
NODE_MUTATING_OPERATIONS = DESTRUCTIVE_OPERATIONS - CONTAINMENT_ONLY_OPERATIONS
RECOVERY_OPERATION_RANK = {
    operation: semantics.recovery_rank
    for operation, semantics in OPERATION_REGISTRY.items()
    if semantics.recovery_rank > 0
}
RECOVERY_OPERATION_DOMINANCE = {
    operation: semantics.dominates
    for operation, semantics in OPERATION_REGISTRY.items()
    if semantics.dominates
}
MERGE_INTENT_OPERATIONS = operations_with("merge_intent")
NODE_WIDE_RECOVERY_OPERATIONS = operations_with("node_wide")
ZERO_RANK_ACTION_OPERATIONS = operations_with("zero_rank_action")
NODE_ACTION_SCOPE_OPERATIONS = operations_with("node_action_scope")
WORKLOAD_SCOPED_OPERATIONS = operations_with("workload_scoped")
SHARED_DAG_OPERATIONS = operations_with("shared_dag")
NODE_EXCLUSIVE_OPERATIONS = operations_with("node_exclusive")
SAFE_WAITING_PREEMPT_OPERATIONS = operations_with("safe_waiting_preempt")
SAFE_REMOTE_WAITING_PREEMPT_OPERATIONS = operations_with("safe_remote_waiting_preempt")
MULTI_NODE_BARRIER_OPERATIONS = operations_with("multi_node_barrier")
TRANSIENT_GPU_INVENTORY_OPERATIONS = operations_with("transient_gpu_inventory_recovery")
HARDWARE_ESCALATION_RELEVANT_OPERATIONS = operations_with(
    "hardware_escalation_relevant"
)
# Node-side operations that read host process state through /proc, and so
# require the agent's proc root to be the host's rather than the
# container's. Enabling one of them without the host mount produces a
# quiesce that "verifies" no GPU clients by looking at an empty namespace.
HOST_PROC_ROOT_OPERATIONS = operations_with("host_proc_root_dependent")
# Operations whose NodeAction steps must inherit the maintenance
# generation captured when the node was quiesced, instead of reading the
# live fleet endpoint.
MAINTENANCE_GENERATION_OPERATIONS = operations_with("maintenance_generation_scoped")
# Long-running mutating node actions that read the agent generation live.
# Their command_id is ``<step key>/<node>`` with no generation suffix: an
# agent restart under the action bumps the generation, and a suffix that
# moved with it formed a new command the restarted agent had never seen --
# a second driver install. Named by step and node alone, the retry lands on
# the ledger row the restart closed as INTERRUPTED (or on the finished
# result) and is answered, not executed. The live generation still rides in
# the command body for the agent's own generation check. Scoped operations
# above already pin the quiesce-time generation and need no second flag.
GENERATION_STABLE_COMMAND_OPERATIONS = operations_with("generation_stable_command_id")
OPERATION_RESOURCE_CLAIMS = {
    operation: frozenset(item.value for item in semantics.resource_claims)
    for operation, semantics in OPERATION_REGISTRY.items()
}


validate_operation_registry()

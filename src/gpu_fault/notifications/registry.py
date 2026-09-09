from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from gpu_fault.notifications.dcgm_diagnostic import (
    DcgmDiagnosticEmailBuilder,
)
from gpu_fault.notifications.diagnostic_inconclusive import (
    DiagnosticInconclusiveEmailBuilder,
)
from gpu_fault.notifications.efa_rdma import EfaRdmaEventEmailBuilder
from gpu_fault.notifications.hardware_escalation import (
    HardwareEscalationEmailBuilder,
)
from gpu_fault.notifications.hardware_inventory import (
    HardwareInventoryEmailBuilder,
)
from gpu_fault.notifications.host_resource import (
    HostResourceEventEmailBuilder,
)
from gpu_fault.notifications.hyperpod_advisory import (
    HyperPodAdvisoryEmailBuilder,
)
from gpu_fault.notifications.not_applicable import (
    NotApplicableEmailBuilder,
)
from gpu_fault.notifications.nvlink74_mechanical import (
    Nvlink74MechanicalEmailBuilder,
)
from gpu_fault.notifications.nvlink74_support import (
    Nvlink74SupportEmailBuilder,
)
from gpu_fault.notifications.restart_guard import RestartGuardEmailBuilder
from gpu_fault.notifications.sxid_event import SxidEventEmailBuilder
from gpu_fault.notifications.warm_spare import (
    WarmSpareReplacementEmailBuilder,
)
from gpu_fault.notifications.xid_investigatory import (
    XidInvestigatoryEmailBuilder,
)


class NotificationKind(StrEnum):
    HYPERPOD_ADVISORY = "HYPERPOD_ADVISORY"
    NOT_APPLICABLE = "NOT_APPLICABLE"
    XID_INVESTIGATORY = "XID_INVESTIGATORY"
    SXID_EVENT = "SXID_EVENT"
    EFA_RDMA_EVENT = "EFA_RDMA_EVENT"
    HARDWARE_INVENTORY = "HARDWARE_INVENTORY"
    HOST_RESOURCE = "HOST_RESOURCE"
    GPU_COUNT_CHANGE = "GPU_COUNT_CHANGE"
    BUDGET_EXHAUSTED = "BUDGET_EXHAUSTED"
    WORKLOAD_RESTARTED = "WORKLOAD_RESTARTED"
    NODE_RESTARTED = "NODE_RESTARTED"
    FABRIC_MANAGER_RESTARTED = "FABRIC_MANAGER_RESTARTED"
    FABRIC_RESET_COMPLETED = "FABRIC_RESET_COMPLETED"
    GPU_RESET_COMPLETED = "GPU_RESET_COMPLETED"
    DCGM_DIAGNOSTIC = "DCGM_DIAGNOSTIC"
    DIAGNOSTIC_INCONCLUSIVE = "DIAGNOSTIC_INCONCLUSIVE"
    HARDWARE_ESCALATION = "HARDWARE_ESCALATION"
    NVLINK74_SUPPORT = "NVLINK74_SUPPORT"
    NVLINK74_MECHANICAL = "NVLINK74_MECHANICAL"
    WARM_SPARE_REPLACEMENT = "WARM_SPARE_REPLACEMENT"


@dataclass(frozen=True)
class NotificationDefinition:
    builder_type: type
    method: str = "build"


NOTIFICATION_REGISTRY = {
    NotificationKind.HYPERPOD_ADVISORY: NotificationDefinition(
        HyperPodAdvisoryEmailBuilder
    ),
    NotificationKind.NOT_APPLICABLE: NotificationDefinition(NotApplicableEmailBuilder),
    NotificationKind.XID_INVESTIGATORY: NotificationDefinition(
        XidInvestigatoryEmailBuilder
    ),
    NotificationKind.SXID_EVENT: NotificationDefinition(SxidEventEmailBuilder),
    NotificationKind.EFA_RDMA_EVENT: NotificationDefinition(EfaRdmaEventEmailBuilder),
    NotificationKind.HARDWARE_INVENTORY: NotificationDefinition(
        HardwareInventoryEmailBuilder
    ),
    NotificationKind.HOST_RESOURCE: NotificationDefinition(
        HostResourceEventEmailBuilder
    ),
    NotificationKind.GPU_COUNT_CHANGE: NotificationDefinition(
        RestartGuardEmailBuilder,
        "build_gpu_count_change",
    ),
    NotificationKind.BUDGET_EXHAUSTED: NotificationDefinition(
        RestartGuardEmailBuilder,
        "build_budget_exhausted",
    ),
    NotificationKind.WORKLOAD_RESTARTED: NotificationDefinition(
        RestartGuardEmailBuilder,
        "build_workload_restarted",
    ),
    NotificationKind.NODE_RESTARTED: NotificationDefinition(
        RestartGuardEmailBuilder,
        "build_node_restarted",
    ),
    NotificationKind.FABRIC_MANAGER_RESTARTED: NotificationDefinition(
        RestartGuardEmailBuilder,
        "build_fabric_manager_restarted",
    ),
    NotificationKind.FABRIC_RESET_COMPLETED: NotificationDefinition(
        RestartGuardEmailBuilder,
        "build_fabric_reset_completed",
    ),
    NotificationKind.GPU_RESET_COMPLETED: NotificationDefinition(
        RestartGuardEmailBuilder,
        "build_gpu_reset_completed",
    ),
    NotificationKind.DCGM_DIAGNOSTIC: NotificationDefinition(
        DcgmDiagnosticEmailBuilder
    ),
    NotificationKind.DIAGNOSTIC_INCONCLUSIVE: NotificationDefinition(
        DiagnosticInconclusiveEmailBuilder
    ),
    NotificationKind.HARDWARE_ESCALATION: NotificationDefinition(
        HardwareEscalationEmailBuilder
    ),
    NotificationKind.NVLINK74_SUPPORT: NotificationDefinition(
        Nvlink74SupportEmailBuilder
    ),
    NotificationKind.NVLINK74_MECHANICAL: NotificationDefinition(
        Nvlink74MechanicalEmailBuilder
    ),
    NotificationKind.WARM_SPARE_REPLACEMENT: NotificationDefinition(
        WarmSpareReplacementEmailBuilder
    ),
}


def validate_notification_registry() -> None:
    if set(NOTIFICATION_REGISTRY) != set(NotificationKind):
        raise RuntimeError("notification registry is not exhaustive")
    for kind, definition in NOTIFICATION_REGISTRY.items():
        if not hasattr(definition.builder_type, definition.method):
            raise RuntimeError(
                f"{kind.value} builder has no method {definition.method}"
            )


class NotificationBuilderRegistry:
    def __init__(
        self,
        overrides: dict[NotificationKind, object] | None = None,
    ) -> None:
        self.overrides = overrides or {}
        self._instances: dict[type, object] = {}

    def builder(self, kind: NotificationKind) -> object:
        override = self.overrides.get(kind)
        if override is not None:
            return override
        builder_type = NOTIFICATION_REGISTRY[kind].builder_type
        return self._instances.setdefault(builder_type, builder_type())

    def build(self, kind: NotificationKind, /, *args, **kwargs) -> Any:
        definition = NOTIFICATION_REGISTRY[kind]
        method = getattr(self.builder(kind), definition.method)
        return method(*args, **kwargs)


validate_notification_registry()

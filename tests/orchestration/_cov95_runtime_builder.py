from __future__ import annotations

from datetime import datetime, timezone
from functools import lru_cache
from typing import Any

from gpu_fault.orchestration.coordinator import (
    OFFICIAL_WORKFLOW_OPERATION,
    OPERATION_CAPABILITY,
    PRE_ACTION_OPERATION,
    PRE_ACTION_ORDER,
)
from gpu_fault.orchestration.workflow_builder import WorkflowBuilder
from gpu_fault.policy import (
    FaultPolicyDecision,
    GpuFaultPolicyEngine,
    SxidClassification,
    SxidEvent,
    XidEvent,
    load_sxid_policy,
    load_xid_policy,
)
from gpu_fault.store import InMemoryStore

NOW = datetime(2026, 9, 12, 12, tzinfo=timezone.utc)


def builder(**overrides: Any) -> WorkflowBuilder:
    values = {
        "target_driver_branch": 575,
        "target_firmware_version": "firmware-unit",
        "sxid_driver_remediation_codes": {100},
        "sxid_firmware_update_codes": {100},
        "operation_capability": OPERATION_CAPABILITY,
        "official_workflow_operation": OFFICIAL_WORKFLOW_OPERATION,
        "pre_action_order": PRE_ACTION_ORDER,
        "pre_action_operation": PRE_ACTION_OPERATION,
        **overrides,
    }
    return WorkflowBuilder(InMemoryStore(), **values)


def xid(**updates: Any) -> XidEvent:
    return XidEvent(
        **{
            "event_id": "unit-xid",
            "cluster_id": "cluster-a",
            "node_id": "node-a",
            "observed_at": NOW,
            "xid": 46,
            "gpu_uuid": "GPU-a",
            "product": "H100",
            "driver_branch": 575,
            "cuda_version": "12.9",
            "workload_state": "IDLE",
            **updates,
        }
    )


def sxid(**updates: Any) -> SxidEvent:
    return SxidEvent(
        **{
            "event_id": "unit-sxid",
            "cluster_id": "cluster-a",
            "node_id": "node-a",
            "observed_at": NOW,
            "sxid": 100,
            "classification": SxidClassification.FATAL,
            "classification_source": "unit",
            "workload_state": "IDLE",
            "participating_gpu_uuids": ["GPU-a"],
            "fabric_partition": "fabric-a",
            **updates,
        }
    )


@lru_cache(maxsize=1)
def base_decision() -> FaultPolicyDecision:
    return GpuFaultPolicyEngine(load_xid_policy(), load_sxid_policy()).evaluate_xid(
        xid()
    )


def decision(**updates: Any) -> FaultPolicyDecision:
    return base_decision().model_copy(deep=True, update=updates)

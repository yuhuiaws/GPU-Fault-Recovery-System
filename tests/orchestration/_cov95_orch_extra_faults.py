from __future__ import annotations

from datetime import timedelta

from gpu_fault.fleet import AgentLifecycleState, AgentRecord
from gpu_fault.models import WorkflowOperation
from gpu_fault.policy import (
    GpuFaultPolicyEngine,
    XidEvent,
    load_sxid_policy,
    load_xid_policy,
)
from tests.orchestration._cov95_orch_extra_support import NOW


def xid(**updates):
    return XidEvent(
        **{
            "event_id": "unit-xid",
            "cluster_id": "cluster-a",
            "node_id": "node-0",
            "observed_at": NOW,
            "source_event_time": NOW,
            "event_source": "NVIDIA_KERNEL_LOG",
            "xid": 48,
            "gpu_uuid": "GPU-0",
            "product": "H100",
            "driver_branch": 575,
            "cuda_version": "12.9",
            "runtime_profile_version": "simulated-v1",
            "workload_state": "IDLE",
            **updates,
        }
    )


def policy_decision(event):
    return GpuFaultPolicyEngine(load_xid_policy(), load_sxid_policy()).evaluate_xid(
        event
    )


def agent(**updates):
    return AgentRecord(
        **{
            "cluster_id": "cluster-a",
            "node_id": "node-0",
            "endpoint": "https://node-0.example.invalid:9099",
            "agent_protocol_version": 3,
            "agent_version": "0.10.0",
            "artifact_sha256": "a" * 64,
            "policy_version": "policy-v1",
            "runtime_profile_version": "simulated-v1",
            "config_digest": "b" * 64,
            "allowed_operations": [
                WorkflowOperation.VERIFY_NO_GPU_CLIENTS,
                WorkflowOperation.RESET_GPU,
            ],
            "boot_id": "unit-boot",
            "node_instance_id": "unit-instance",
            "agent_incarnation_id": "unit-boot",
            "first_seen_at": NOW - timedelta(minutes=1),
            "last_seen_at": NOW,
            "lease_expires_at": NOW + timedelta(minutes=1),
            "lifecycle_state": AgentLifecycleState.ACTIVE,
            **updates,
        }
    )

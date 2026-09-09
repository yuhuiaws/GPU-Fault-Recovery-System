"""Shared fixtures for the compound-command (性能 C) tests.

One reset chain, built the way the planner builds it: a Kubernetes cordon, the
four node-side steps on one node under the node-agent owner, a validation and
the scheduler release. The step contexts carry the idempotency key
``ProductionWorkflowExecutor._dispatch_step`` would give them, because the
batching predicate refuses a head whose key it cannot reproduce.
"""

from __future__ import annotations

from typing import Any

from gpu_fault.execution import WorkflowStepContext
from gpu_fault.models import (
    IncidentState,
    WorkflowExecutionRequest,
    WorkflowOperation,
    WorkflowRequest,
    WorkflowStatus,
    WorkflowStepSpec,
)
from gpu_fault.remote_step_batching import (
    RemoteStepBatchingPolicy,
    step_idempotency_key,
)
from tests._builders import copy_model, fault_incident, workflow_request, workflow_step
from tests.regional._regional_support import NOW

NODE_OWNER = "gpu-fault-node-agent"
KUBERNETES_OWNER = "gpu-fault-kubernetes-adapter"
VALIDATION_OWNER = "gpu-fault-gpu-validation"
ALL_OWNERS = {NODE_OWNER, KUBERNETES_OWNER, VALIDATION_OWNER}
ACTIVE_POLICY = RemoteStepBatchingPolicy(
    enabled=True, minimum_executor_protocol_version=3
)
NODE_SIDE = (
    WorkflowOperation.QUIESCE_GPU_SERVICES,
    WorkflowOperation.VERIFY_NO_GPU_CLIENTS,
    WorkflowOperation.RESET_GPU,
    WorkflowOperation.RESTORE_GPU_SERVICES,
)


def reset_chain(node_ids: list[str] | None = None) -> list[WorkflowStepSpec]:
    nodes = ["node-a"] if node_ids is None else node_ids
    return [
        workflow_step(
            WorkflowOperation.MARK_UNSCHEDULABLE, KUBERNETES_OWNER, node_ids=nodes
        ),
        *(
            workflow_step(operation, NODE_OWNER, node_ids=nodes, gpu_uuids=["GPU-a"])
            for operation in NODE_SIDE
        ),
        workflow_step(WorkflowOperation.VALIDATE_GPU, VALIDATION_OWNER, node_ids=nodes),
        workflow_step(
            WorkflowOperation.RESTORE_SCHEDULING, KUBERNETES_OWNER, node_ids=nodes
        ),
    ]


def chain_state(
    store: Any,
    steps: list[WorkflowStepSpec],
    *,
    request_id: str = "workflow-a",
    **workflow_values: Any,
):
    incident = fault_incident(
        f"incident-{request_id}",
        f"event-{request_id}",
        official_action="RESTART_BM",
        state=IncidentState.ACTION_PENDING,
        fencing_token=3,
        created_at=NOW,
        updated_at=NOW,
    )
    workflow = workflow_request(
        request_id,
        incident.incident_id,
        workflow_values.pop("status", WorkflowStatus.PENDING),
        runtime_profile_version="active-v1",
        official_action="RESTART_BM",
        official_steps=steps,
        created_at=NOW,
        updated_at=NOW,
        **workflow_values,
    )
    incident = copy_model(incident, workflow_request_id=workflow.request_id)
    store.save_incident(incident)
    store.save_workflow(workflow)
    return incident, workflow


def step_context(
    workflow: WorkflowRequest, incident: Any, index: int
) -> WorkflowStepContext:
    step = workflow.official_steps[index]
    return WorkflowStepContext(
        workflow=workflow,
        incident=incident,
        step=step,
        step_index=index,
        request=WorkflowExecutionRequest(expected_fencing_token=3),
        idempotency_key=step_idempotency_key(workflow, index, step),
    )


def quiesce_details() -> dict[str, Any]:
    return {
        "agent_generations": {"node-a": 5},
        "maintenance_window_started_at": "2026-07-28T00:00:00+00:00",
        "maintenance_window_expires_at": "2026-07-28T00:07:00+00:00",
        "node_results": {"node-a": {"status": "SUCCEEDED", "failsafe_seconds": 420}},
    }

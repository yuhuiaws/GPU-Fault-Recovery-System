from __future__ import annotations

from dataclasses import replace
from typing import Any

from gpu_fault.models import WorkflowOperation, WorkflowStatus, WorkflowStepStatus
from tests._builders import workflow_request, workflow_step, workflow_step_execution
from tests.execution._cov95_runtime_workflows import FlowHarness

QUIESCE = WorkflowOperation.QUIESCE_GPU_SERVICES
RESET = WorkflowOperation.RESET_GPU
RESTORE = WorkflowOperation.RESTORE_GPU_SERVICES
REBOOT = WorkflowOperation.RESTART_NODE
VALIDATE = WorkflowOperation.VALIDATE_GPU


def handoff_harness(
    successor_operations: list[WorkflowOperation],
    *,
    predecessor_updates: dict[str, Any] | None = None,
) -> FlowHarness:
    h = FlowHarness(successor_operations)
    predecessor = workflow_request(
        "workflow-predecessor",
        h.incident.incident_id,
        WorkflowStatus.SUPERSEDED,
        h.workflow.fencing_token,
        preempted_by_workflow_id=h.workflow.request_id,
        official_steps=[
            workflow_step(QUIESCE),
            workflow_step(RESET),
            workflow_step(RESTORE, parameters={"services": ["nvidia-fabricmanager"]}),
        ],
        completed_step_indexes=[0],
        completed_operations=[QUIESCE],
        step_executions=[
            workflow_step_execution(
                0,
                QUIESCE,
                WorkflowStepStatus.SUCCEEDED,
                adapter_operation_id="remote/quiesce-proof",
                details={"quiesced_services": ["nvidia-fabricmanager"]},
            )
        ],
    )
    if predecessor_updates:
        predecessor = predecessor.model_copy(update=predecessor_updates)
    h.store.save_workflow(predecessor)
    h.amend(predecessor_workflow_id=predecessor.request_id)
    h.executor.config = replace(
        h.executor.config,
        allowed_operations=frozenset({*successor_operations, RESTORE}),
    )
    return h

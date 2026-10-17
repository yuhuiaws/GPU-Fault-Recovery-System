"""Which quiesce windows keep Fabric Manager running (batch 128).

A window whose only GPU-state change is RESET_GPU keeps ``nvidia-fabricmanager``
running: its QUIESCE command carries ``fabric_manager=keep`` (stopping Fabric
Manager loses the fabric state, and the restarted instance would know only the
GPUs reset in the window -- live, NVLS was gone for the whole node). Every other
window stops it as before: the full GPU + NVSwitch reset, a driver or firmware
remediation, the NVLink field diagnostic, a reboot, an operation not listed.

The policy is read off the persisted plan -- the window's steps on the QUIESCE's
own branch, whatever their status -- so a re-sent QUIESCE carries the same body
(a reused command_id with another body is 409 COMMAND_ID_REUSED on the Agent).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pytest

from gpu_fault.gpu_fabric_window import (
    fabric_manager_policy,
    quiesce_command_parameters,
    window_operations,
)
from gpu_fault.models import WorkflowRequest, WorkflowStepSpec
from gpu_fault.node_agent.ledger import canonical_digest
from tests._builders import (
    build_store,
    copy_model,
    node_action_result,
    workflow_request,
    workflow_step_execution,
)

from ._support import (
    NodeActionWorkflowAdapter,
    StubFleetRegistry,
    WorkflowExecutionRequest,
    WorkflowOperation,
    WorkflowStepContext,
    WorkflowStepStatus,
    workflow_state,
)

Op = WorkflowOperation
KEEP = {"fabric_manager": "keep"}
RESET_CHAIN = [
    Op.FREEZE_EVIDENCE,
    Op.MARK_UNSCHEDULABLE,
    Op.QUIESCE_GPU_SERVICES,
    Op.VERIFY_NO_GPU_CLIENTS,
    Op.RESET_GPU,
    Op.RESTORE_GPU_SERVICES,
    Op.VALIDATE_GPU,
    Op.RESTORE_SCHEDULING,
]


def _steps(
    operations: list[WorkflowOperation],
    *,
    node: str = "node-a",
    branch: str | None = None,
) -> list[WorkflowStepSpec]:
    return [
        WorkflowStepSpec(
            operation=operation,
            execution_owner="gpu-fault-node-agent",
            node_ids=[node],
            gpu_uuids=["GPU-a"],
            branch_id=branch,
        )
        for operation in operations
    ]


def _quiesce_index(steps: list[WorkflowStepSpec], node: str = "node-a") -> int:
    return next(
        index
        for index, step in enumerate(steps)
        if step.operation is Op.QUIESCE_GPU_SERVICES and node in step.node_ids
    )


def test_a_per_gpu_reset_window_keeps_fabric_manager() -> None:
    steps = _steps(RESET_CHAIN)

    assert window_operations(steps, 2) == [Op.VERIFY_NO_GPU_CLIENTS, Op.RESET_GPU]
    assert fabric_manager_policy(steps, 2) == "keep"


@pytest.mark.parametrize(
    "window",
    [
        [Op.VERIFY_NO_GPU_CLIENTS, Op.RESET_ALL_GPUS_NVSWITCHES],
        [Op.VERIFY_NO_GPU_CLIENTS, Op.REMEDIATE_DRIVER, Op.RESET_GPU],
        [Op.VERIFY_NO_GPU_CLIENTS, Op.UPDATE_SOFTWARE_FIRMWARE],
        [Op.VERIFY_NO_GPU_CLIENTS, Op.RUN_NVLINK74_WORKFLOW, Op.RESET_GPU],
        [Op.RESTART_NODE],
        [Op.VERIFY_NO_GPU_CLIENTS],
        [Op.VERIFY_NO_GPU_CLIENTS, Op.TRIGGER_HEALTH_SNAPSHOT, Op.RESET_GPU],
        [Op.RESET_GPU, Op.RESTART_FABRIC_MANAGER],
    ],
    ids=[
        "full-fabric-reset",
        "driver-remediation",
        "firmware-update",
        "nvlink74-field-diagnostic",
        "reboot",
        "no-reset",
        "operation-not-listed",
        "fabric-manager-restart",
    ],
)
def test_every_other_window_stops_it_as_before(window) -> None:
    steps = _steps(
        [Op.QUIESCE_GPU_SERVICES, *window, Op.RESTORE_GPU_SERVICES, Op.VALIDATE_GPU]
    )

    assert fabric_manager_policy(steps, 0) is None


def test_a_window_without_its_restore_is_not_read() -> None:
    """A window handed to a successor (no RESTORE on this branch), or a second
    QUIESCE before the RESTORE: nothing proves what runs inside it."""

    handed_off = _steps([Op.QUIESCE_GPU_SERVICES, Op.RESET_GPU, Op.VALIDATE_GPU])
    doubled = _steps(
        [
            Op.QUIESCE_GPU_SERVICES,
            Op.RESET_GPU,
            Op.QUIESCE_GPU_SERVICES,
            Op.RESTORE_GPU_SERVICES,
        ]
    )

    assert window_operations(handed_off, 0) is None
    assert fabric_manager_policy(handed_off, 0) is None
    assert fabric_manager_policy(doubled, 0) is None


def test_each_dag_branch_is_classified_on_its_own() -> None:
    """Two node branches of one job DAG: node-a resets one GPU, node-b runs the
    full fabric reset; the shared steps around them belong to neither window."""

    shared = [
        step.model_copy(update={"node_ids": ["node-a", "node-b"]})
        for step in _steps([Op.MARK_UNSCHEDULABLE, Op.STOP_WORKLOADS], branch="shared")
    ]
    node_a = _steps(RESET_CHAIN[2:6], node="node-a", branch="node:node-a")
    node_b = _steps(
        [
            Op.QUIESCE_GPU_SERVICES,
            Op.VERIFY_NO_GPU_CLIENTS,
            Op.RESET_ALL_GPUS_NVSWITCHES,
            Op.RESTORE_GPU_SERVICES,
        ],
        node="node-b",
        branch="node:node-b",
    )
    # Interleaved the way an appended branch can land.
    steps = [*shared, node_a[0], node_b[0], *node_a[1:], *node_b[1:]]

    assert fabric_manager_policy(steps, _quiesce_index(steps, "node-a")) == "keep"
    assert fabric_manager_policy(steps, _quiesce_index(steps, "node-b")) is None


def _workflow(steps: list[WorkflowStepSpec], **updates: Any) -> WorkflowRequest:
    return workflow_request("workflow-a", "incident-a", official_steps=steps, **updates)


def test_a_re_sent_quiesce_carries_the_same_parameters() -> None:
    """Execution state, superseded steps and an appended branch do not move
    the window's structure, so the body of every send is the same."""

    steps = _steps(RESET_CHAIN)
    first = quiesce_command_parameters(_workflow(steps), 2, steps[2])
    later = _workflow(
        [*steps, *_steps(RESET_CHAIN[2:6], node="node-b", branch="node:node-b")],
        completed_step_indexes=[0, 1],
        superseded_step_indexes=[4, 5],
        step_executions=[
            workflow_step_execution(
                2, Op.QUIESCE_GPU_SERVICES, WorkflowStepStatus.WAITING
            )
        ],
    )

    assert first == KEEP
    assert quiesce_command_parameters(later, 2, steps[2]) == first


def test_the_step_s_own_parameters_travel_and_a_planned_value_wins() -> None:
    steps = _steps(RESET_CHAIN)
    steps[2] = steps[2].model_copy(
        update={"parameters": {"workload_cgroup_paths_by_node": {"node-a": ["/job"]}}}
    )
    workflow = _workflow(steps)

    assert quiesce_command_parameters(workflow, 2, steps[2]) == {
        "workload_cgroup_paths_by_node": {"node-a": ["/job"]},
        **KEEP,
    }
    planned = steps[2].model_copy(update={"parameters": {"fabric_manager": "keep"}})
    assert quiesce_command_parameters(workflow, 2, planned) == KEEP
    assert quiesce_command_parameters(workflow, 4, steps[4]) == {}, "not a QUIESCE"
    assert quiesce_command_parameters(workflow, 3, steps[2]) == {
        "workload_cgroup_paths_by_node": {"node-a": ["/job"]}
    }, "an index that is not this step's says nothing"


def _context(operations: list[WorkflowOperation]) -> WorkflowStepContext:
    store = build_store()
    incident, workflow = workflow_state(store, operations)
    steps = [
        copy_model(
            step,
            execution_owner="gpu-fault-node-agent",
            node_ids=["node-a"],
            gpu_uuids=["GPU-a"],
        )
        for step in workflow.official_steps
    ]
    workflow = copy_model(workflow, official_steps=steps)
    store.save_workflow(workflow)
    return WorkflowStepContext(
        workflow=workflow,
        incident=incident,
        step=steps[0],
        step_index=0,
        request=WorkflowExecutionRequest(expected_fencing_token=3),
        idempotency_key="workflow/0/QUIESCE_GPU_SERVICES",
    )


def _adapter(sent: list[Any]) -> NodeActionWorkflowAdapter:
    def sender(_endpoint, envelope):
        sent.append(envelope)
        return node_action_result(
            envelope.command.command_id,
            envelope.command.operation,
            details={
                "quiesced": True,
                "active_services": ["nvidia-dcgm", "kubelet"],
                "kept_services": [
                    {"service": "nvidia-fabricmanager", "reason": "per-GPU reset"}
                ],
            },
        )

    registry = StubFleetRegistry({"node-a": "http://node-a:9099"}, generation=7)
    registry.now = lambda: datetime(2026, 10, 17, 18, 0, tzinfo=timezone.utc)
    return NodeActionWorkflowAdapter({}, "s" * 32, sender=sender, registry=registry)


def test_the_quiesce_command_of_a_reset_gpu_chain_says_keep() -> None:
    sent: list[Any] = []
    context = _context(
        [
            Op.QUIESCE_GPU_SERVICES,
            Op.VERIFY_NO_GPU_CLIENTS,
            Op.RESET_GPU,
            Op.RESTORE_GPU_SERVICES,
        ]
    )

    first = _adapter(sent).execute(context)
    second = _adapter(sent).execute(context)

    assert first.status is WorkflowStepStatus.SUCCEEDED, first.error
    assert sent[0].command.parameters == KEEP
    assert context.step.parameters == {}, "the persisted step is not rewritten"
    assert sent[1].command.command_id == sent[0].command.command_id
    assert canonical_digest(sent[1].command.parameters) == canonical_digest(
        sent[0].command.parameters
    ), "the digest the Agent compares for COMMAND_ID_REUSED is the same"
    assert first.details["kept_services"] == {
        "node-a": [{"service": "nvidia-fabricmanager", "reason": "per-GPU reset"}]
    }
    assert first.details["stopped_services"] == {"node-a": ["nvidia-dcgm", "kubelet"]}
    assert second.status is WorkflowStepStatus.SUCCEEDED


def test_the_quiesce_command_of_a_full_fabric_reset_says_nothing() -> None:
    sent: list[Any] = []
    context = _context(
        [
            Op.QUIESCE_GPU_SERVICES,
            Op.VERIFY_NO_GPU_CLIENTS,
            Op.RESET_ALL_GPUS_NVSWITCHES,
            Op.RESTORE_GPU_SERVICES,
        ]
    )

    _adapter(sent).execute(context)

    assert sent[0].command.parameters == {}, "an older Agent and a newer one agree"

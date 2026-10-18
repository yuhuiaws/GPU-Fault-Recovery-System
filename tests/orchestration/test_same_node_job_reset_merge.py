"""Two GPUs of one node of a running two-node job: what the control plane plans,
by when the second fault arrives (GF-REGIONAL-DESTR-026).

The idle-node merge has its own tests in ``test_merge.py``
(``test_merged_reset_targets_every_faulted_gpu``,
``test_merge_stops_once_the_reset_step_already_ran``). A node that runs a
managed job takes the attempt-scoped path instead
(``orchestration/families/grouped_faults.py``), and the live case grades that
path's shapes, pinned here:

- both faults before any step ran: one linear job workflow whose node actions
  name both GPUs with ``gpu_uuids_by_node`` -- the case's shape;
- the second after the containment steps, before the quiesce: the pending
  steps widen in place (``PLAN_WIDENED``);
- the second after the RESET_GPU command was issued: that command is left as
  it was and a successor branch resets the second GPU before the single join
  (``BRANCH_SUCCESSOR_QUEUED``) -- the late variant the case documents;
- the second after RESTART_WORKLOAD started: a successor workflow with its own
  stop and restart, i.e. a second restart of the job.
"""

from __future__ import annotations

import asyncio
from typing import Any

from gpu_fault.app import ApplicationContext
from gpu_fault.models import (
    WorkflowEventCode,
    WorkflowOperation,
    WorkflowRequest,
    WorkflowStatus,
    WorkflowStepSpec,
    WorkflowStepStatus,
)
from gpu_fault.orchestration import IncidentOrchestrator
from tests._builders import (
    build_context,
    container_observation,
    workflow_step_execution,
)
from tests.orchestration._cross_fault_support import (
    observation,
    post_faults,
    xid_payload,
)

NODE = "node-a"
PARTNER = "node-b"
GPU_A = "GPU-a0"
GPU_B = "GPU-a1"
BDF = {GPU_A: "0000:59:00.0", GPU_B: "0000:6a:00.0"}
NODE_ACTIONS = (
    WorkflowOperation.QUIESCE_GPU_SERVICES,
    WorkflowOperation.VERIFY_NO_GPU_CLIENTS,
    WorkflowOperation.RESET_GPU,
    WorkflowOperation.RESTORE_GPU_SERVICES,
)
JOB_STEPS = (WorkflowOperation.STOP_WORKLOADS, WorkflowOperation.RESTART_WORKLOAD)
LINEAR_PLAN = [
    "FREEZE_EVIDENCE",
    "MARK_UNSCHEDULABLE",
    "STOP_WORKLOADS",
    "QUIESCE_GPU_SERVICES",
    "VERIFY_NO_GPU_CLIENTS",
    "RESET_GPU",
    "RESTORE_GPU_SERVICES",
    "VALIDATE_GPU",
    "RESTORE_SCHEDULING",
    "RESTART_WORKLOAD",
]
SUCCEEDED = WorkflowStepStatus.SUCCEEDED
WAITING = WorkflowStepStatus.WAITING


def job_context() -> ApplicationContext:
    """A two-node job: the Master holds node-a's GPUs, the Worker node-b's."""

    context = build_context()
    context.orchestrator = IncidentOrchestrator(
        context.store, workflow_preemption_enabled=True
    )
    containers = [
        container_observation(
            "pod-a", "trainer-a", 0, NODE, role="master", gpu_uuids=[GPU_A, GPU_B]
        ),
        container_observation("pod-b", "trainer-b", 1, PARTNER, gpu_uuids=["GPU-b0"]),
    ]
    context.store.save_attempt_observation(
        observation().model_copy(
            update={"expected_critical_ranks": 2, "containers": containers}
        )
    )
    return context


def job_fault(context: ApplicationContext, event_id: str, gpu: str) -> dict[str, Any]:
    """One XID 46 on ``gpu`` of node-a, as the kernel collector posts it."""

    payload = xid_payload(46, event_id)
    payload.update({"gpu_uuid": gpu, "pci_bdf": BDF[gpu]})
    return asyncio.run(post_faults(context, [("/v1/gpu-events/xid", payload)]))[0]


def start_steps(
    context: ApplicationContext,
    workflow: WorkflowRequest,
    statuses: dict[WorkflowOperation, WorkflowStepStatus],
) -> WorkflowRequest:
    """Save ``workflow`` claimed by an executor with these steps executed."""

    executions = []
    completed = []
    for index, step in enumerate(workflow.official_steps):
        status = statuses.get(step.operation)
        if status is None:
            continue
        executions.append(
            workflow_step_execution(
                index, step.operation, status, adapter_operation_id=f"remote/{index}"
            )
        )
        if status is SUCCEEDED:
            completed.append(index)
    running = workflow.model_copy(
        update={
            "status": WorkflowStatus.RUNNING,
            "execution_owner_id": "executor-a",
            "not_before": None,
            "step_executions": executions,
            "completed_step_indexes": completed,
        }
    )
    context.store.save_workflow(running)
    return running


def steps_of(
    workflow: WorkflowRequest, operation: WorkflowOperation
) -> list[WorkflowStepSpec]:
    return [step for step in workflow.official_steps if step.operation is operation]


def through(last: WorkflowOperation) -> dict[WorkflowOperation, WorkflowStepStatus]:
    """Every linear step up to ``last`` succeeded."""

    operations = [WorkflowOperation(item) for item in LINEAR_PLAN]
    return {item: SUCCEEDED for item in operations[: operations.index(last) + 1]}


def test_both_faults_before_any_step_widen_one_linear_job_workflow() -> None:
    context = job_context()
    first = job_fault(context, "xid-a0", GPU_A)
    second = job_fault(context, "xid-a1", GPU_B)

    assert second["workflow_request_id"] == first["workflow_request_id"], (
        "the second fault joins the first one's workflow"
    )
    assert second["incident_id"] == first["incident_id"], "one incident for both"
    assert len(context.store.list_workflows(limit=10)) == 1, "no second workflow"
    incident = context.store.get_incident(first["incident_id"])
    assert incident.gpu_uuids == [GPU_A, GPU_B]
    assert incident.node_ids == [NODE]
    workflow = context.store.get_workflow(first["workflow_request_id"])
    assert workflow.dag_enabled is False, "one node's reset stays a linear plan"
    assert [step.operation.value for step in workflow.official_steps] == LINEAR_PLAN
    for operation in NODE_ACTIONS:
        (step,) = steps_of(workflow, operation)
        assert step.node_ids == [NODE]
        assert step.gpu_uuids == [GPU_A, GPU_B], operation
        assert step.parameters["gpu_uuids_by_node"] == {NODE: [GPU_A, GPU_B]}
    (validate,) = steps_of(workflow, WorkflowOperation.VALIDATE_GPU)
    assert validate.gpu_uuids == [GPU_A, GPU_B], "validation answers for both"
    for operation in JOB_STEPS:
        (step,) = steps_of(workflow, operation)
        assert step.node_ids == [NODE, PARTNER], (
            "the job is stopped and restarted whole"
        )
    for operation in (
        WorkflowOperation.MARK_UNSCHEDULABLE,
        WorkflowOperation.RESTORE_SCHEDULING,
    ):
        (step,) = steps_of(workflow, operation)
        assert step.node_ids == [NODE], "node B is never cordoned"


def test_a_second_fault_after_containment_widens_the_window_still_to_come() -> None:
    context = job_context()
    first = job_fault(context, "xid-a0", GPU_A)
    start_steps(
        context,
        context.store.get_workflow(first["workflow_request_id"]),
        through(WorkflowOperation.STOP_WORKLOADS),
    )

    second = job_fault(context, "xid-a1", GPU_B)

    assert second["workflow_request_id"] == first["workflow_request_id"]
    assert len(context.store.list_workflows(limit=10)) == 1, "no second workflow"
    widened = context.store.get_workflow(first["workflow_request_id"])
    assert widened.events[-1].code == WorkflowEventCode.PLAN_WIDENED
    for operation in NODE_ACTIONS:
        (step,) = steps_of(widened, operation)
        assert step.parameters["gpu_uuids_by_node"] == {NODE: [GPU_A, GPU_B]}, operation
    (mark,) = steps_of(widened, WorkflowOperation.MARK_UNSCHEDULABLE)
    assert mark.gpu_uuids == [GPU_A], "a step that ran keeps the scope it ran with"


def test_a_fault_after_the_reset_was_issued_queues_a_successor_branch() -> None:
    context = job_context()
    first = job_fault(context, "xid-a0", GPU_A)
    issued = {**through(WorkflowOperation.VERIFY_NO_GPU_CLIENTS)}
    issued[WorkflowOperation.RESET_GPU] = WAITING
    running = start_steps(
        context, context.store.get_workflow(first["workflow_request_id"]), issued
    )
    reset_index = LINEAR_PLAN.index("RESET_GPU")
    submitted = running.official_steps[reset_index]

    second = job_fault(context, "xid-a1", GPU_B)

    assert second["workflow_request_id"] == first["workflow_request_id"], (
        "the join had not started, so the record takes a branch, not a successor"
    )
    updated = context.store.get_workflow(first["workflow_request_id"])
    assert updated.dag_enabled is True
    assert updated.events[-1].code == WorkflowEventCode.BRANCH_SUCCESSOR_QUEUED
    kept = updated.official_steps[reset_index]
    assert (kept.gpu_uuids, kept.parameters) == (
        submitted.gpu_uuids,
        submitted.parameters,
    ), "the issued RESET_GPU is never rewritten"
    assert [
        step.gpu_uuids for step in steps_of(updated, WorkflowOperation.RESET_GPU)
    ] == [[GPU_A], [GPU_B]], (
        "a successor branch resets the second GPU in a window of its own"
    )
    for operation in JOB_STEPS:
        assert len(steps_of(updated, operation)) == 1, f"one {operation.value}"
    (join,) = steps_of(updated, WorkflowOperation.RESTART_WORKLOAD)
    releases = [
        index
        for index, step in enumerate(updated.official_steps)
        if step.operation is WorkflowOperation.RESTORE_SCHEDULING
    ]
    assert len(releases) == 2 and set(releases) <= set(join.depends_on_step_indexes), (
        "the single join waits for both windows to release the node"
    )


def test_a_fault_after_the_join_started_gets_its_own_workflow_and_restart() -> None:
    context = job_context()
    first = job_fault(context, "xid-a0", GPU_A)
    joined = {**through(WorkflowOperation.RESTORE_SCHEDULING)}
    joined[WorkflowOperation.RESTART_WORKLOAD] = WAITING
    start_steps(
        context, context.store.get_workflow(first["workflow_request_id"]), joined
    )

    second = job_fault(context, "xid-a1", GPU_B)

    assert second["workflow_request_id"] != first["workflow_request_id"], (
        "a join that started takes no new branch"
    )
    successor = context.store.get_workflow(second["workflow_request_id"])
    assert successor.predecessor_workflow_id == first["workflow_request_id"]
    assert [step.operation.value for step in successor.official_steps] == LINEAR_PLAN, (
        "the successor stops and restarts the job again: a second restart"
    )
    (reset,) = steps_of(successor, WorkflowOperation.RESET_GPU)
    assert reset.gpu_uuids == [GPU_B]

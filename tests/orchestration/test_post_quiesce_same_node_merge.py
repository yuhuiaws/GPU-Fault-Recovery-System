"""A second GPU of one node merged after the node's QUIESCE was dispatched.

GF-REGIONAL-PREEMPT-021 treated "only FREEZE/MARK/QUIESCE ran" as a reset not
yet submitted, so the merge widened the pending VERIFY_NO_GPU_CLIENTS, RESET_GPU
and RESTORE_GPU_SERVICES to the second GPU while the QUIESCE that had already
run kept the first one. That window never released the second GPU: the Node
Agent ended persistence and swept clients for the quiesced GPU only, the
``nvidia-persistenced`` it kept still holds every other device node, and the
Agent refuses a second QUIESCE with another target set
(``tests/node_agent/test_quiesce_window_per_gpu_set.py``). With remote step
batching the QUIESCE is the head of one compound command whose VERIFY, RESET and
RESTORE passengers keep the scope they had when it was minted
(``tests/regional/test_post_quiesce_merge_batched.py``).

Pinned here, by when the second fault arrives:

- before the QUIESCE is dispatched: the pending window widens in place, one
  RESET_GPU for both GPUs (unchanged);
- once the QUIESCE has a record (WAITING or SUCCEEDED, with or without the
  VERIFY behind it): the window keeps exactly what it was dispatched with and the
  second GPU gets a node successor branch with a quiesce window of its own,
  before the job's single join;
- an idle node (no join): a successor workflow behind the running one;
- a record-only merge (the job was withdrawn) never widens a committed window.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from gpu_fault.app import ApplicationContext
from gpu_fault.gpu_fabric_window import quiesce_command_parameters
from gpu_fault.models import (
    WorkflowEventCode,
    WorkflowOperation,
    WorkflowRequest,
    WorkflowStatus,
    WorkflowStepSpec,
    WorkflowStepStatus,
)
from gpu_fault.orchestration import IncidentOrchestrator
from gpu_fault.remote_step_batching import step_idempotency_key
from tests._builders import (
    build_context,
    container_observation,
    copy_model,
    workflow_step_execution,
)
from tests.orchestration._cross_fault_support import (
    observation,
    post_faults,
    xid_payload,
)
from tests.orchestration._support import _node_event, ingest

NODE = "node-a"
PARTNER = "node-b"
GPU_A = "GPU-a0"
GPU_B = "GPU-a1"
BDF = {GPU_A: "0000:59:00.0", GPU_B: "0000:6a:00.0"}
QUIESCE = WorkflowOperation.QUIESCE_GPU_SERVICES
VERIFY = WorkflowOperation.VERIFY_NO_GPU_CLIENTS
RESET = WorkflowOperation.RESET_GPU
RESTORE = WorkflowOperation.RESTORE_GPU_SERVICES
WINDOW = (QUIESCE, VERIFY, RESET, RESTORE)
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
#: Where the job workflow is when the second fault arrives, once its QUIESCE
#: was dispatched: the QUIESCE in flight or done, the VERIFY behind it in flight
#: or done, the RESET not recorded yet.
COMMITTED = {
    "quiesce-waiting": ("STOP_WORKLOADS", {QUIESCE: WAITING}),
    "quiesce-succeeded": ("QUIESCE_GPU_SERVICES", {}),
    "verify-waiting": ("QUIESCE_GPU_SERVICES", {VERIFY: WAITING}),
    "verify-succeeded": ("VERIFY_NO_GPU_CLIENTS", {}),
}


def job_context() -> ApplicationContext:
    """A two-node job: the Master holds node-a's two GPUs, the Worker node-b's."""

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


def run_steps(
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
            "completed_operations": [
                workflow.official_steps[i].operation for i in completed
            ],
        }
    )
    context.store.save_workflow(running)
    return running


def through(last: str) -> dict[WorkflowOperation, WorkflowStepStatus]:
    """Every linear step up to ``last`` succeeded."""

    operations = [WorkflowOperation(item) for item in LINEAR_PLAN]
    return {
        item: SUCCEEDED
        for item in operations[: operations.index(WorkflowOperation(last)) + 1]
    }


def steps_of(
    workflow: WorkflowRequest, operation: WorkflowOperation
) -> list[WorkflowStepSpec]:
    return [step for step in workflow.official_steps if step.operation is operation]


def command_identity(step: WorkflowStepSpec) -> dict[str, Any]:
    """What a command for ``step`` is named and signed by: the step without its
    DAG scheduling fields (``RegionalRemoteWorkflowAdapter`` digests the same)."""

    return step.model_copy(
        update={"branch_id": None, "depends_on_step_indexes": []}
    ).model_dump(mode="json")


def window_indexes(workflow: WorkflowRequest) -> list[int]:
    return [
        LINEAR_PLAN.index(operation.value)
        for operation in WINDOW
        if workflow.official_steps[LINEAR_PLAN.index(operation.value)].operation
        is operation
    ]


def committed_job(
    timing: str,
) -> tuple[ApplicationContext, dict[str, Any], WorkflowRequest]:
    context = job_context()
    first = job_fault(context, "xid-a0", GPU_A)
    last, extra = COMMITTED[timing]
    running = run_steps(
        context,
        context.store.get_workflow(first["workflow_request_id"]),
        {**through(last), **extra},
    )
    return context, first, running


def test_before_the_quiesce_is_dispatched_the_window_widens_in_place() -> None:
    context = job_context()
    first = job_fault(context, "xid-a0", GPU_A)
    run_steps(
        context,
        context.store.get_workflow(first["workflow_request_id"]),
        through("STOP_WORKLOADS"),
    )

    second = job_fault(context, "xid-a1", GPU_B)

    assert second["workflow_request_id"] == first["workflow_request_id"]
    widened = context.store.get_workflow(first["workflow_request_id"])
    assert widened.dag_enabled is False, "nothing ran in the window: still linear"
    assert widened.events[-1].code == WorkflowEventCode.PLAN_WIDENED
    for operation in WINDOW:
        (step,) = steps_of(widened, operation)
        assert step.gpu_uuids == [GPU_A, GPU_B], operation
        assert step.parameters["gpu_uuids_by_node"] == {NODE: [GPU_A, GPU_B]}, operation
    assert len(steps_of(widened, RESET)) == 1, "one reset for both GPUs"


@pytest.mark.parametrize("timing", sorted(COMMITTED))
def test_after_the_quiesce_is_dispatched_the_window_is_left_as_dispatched(
    timing: str,
) -> None:
    context, first, running = committed_job(timing)
    dispatched = {
        index: command_identity(running.official_steps[index])
        for index in window_indexes(running)
    }

    second = job_fault(context, "xid-a1", GPU_B)

    assert second["workflow_request_id"] == first["workflow_request_id"], (
        "the join has not started: the record takes a node branch, not a successor"
    )
    updated = context.store.get_workflow(first["workflow_request_id"])
    assert updated.events[-1].code == WorkflowEventCode.BRANCH_SUCCESSOR_QUEUED, [
        event.code for event in updated.events
    ]
    for index, identity in dispatched.items():
        assert command_identity(updated.official_steps[index]) == identity, (
            f"step {index} ({updated.official_steps[index].operation.value}) of the "
            "dispatched window was rewritten"
        )
    assert updated.step_executions == running.step_executions
    for operation in WINDOW:
        assert [step.gpu_uuids for step in steps_of(updated, operation)] == [
            [GPU_A],
            [GPU_B],
        ], f"{operation.value}: the second GPU gets a window of its own"


@pytest.mark.parametrize("timing", sorted(COMMITTED))
def test_the_second_gpu_window_runs_before_the_single_join(timing: str) -> None:
    context, first, _ = committed_job(timing)

    job_fault(context, "xid-a1", GPU_B)

    updated = context.store.get_workflow(first["workflow_request_id"])
    successor = [
        (index, step)
        for index, step in enumerate(updated.official_steps)
        if step.branch_id not in {None, "shared", "join", "branch:initial"}
    ]
    operations = [step.operation.value for _, step in successor]
    assert operations == [
        "FREEZE_EVIDENCE",
        "MARK_UNSCHEDULABLE",
        "QUIESCE_GPU_SERVICES",
        "VERIFY_NO_GPU_CLIENTS",
        "RESET_GPU",
        "RESTORE_GPU_SERVICES",
        "VALIDATE_GPU",
        "RESTORE_SCHEDULING",
    ], "the successor branch quiesces, verifies, resets and restores node-a again"
    for _, step in successor:
        if step.operation in WINDOW:
            # The node-action adapter reads ``gpu_uuids`` when a step carries
            # no ``gpu_uuids_by_node``, as a single-fault branch does.
            targets = step.parameters.get("gpu_uuids_by_node", {NODE: step.gpu_uuids})
            assert (step.node_ids, targets) == ([NODE], {NODE: [GPU_B]}), step
    first_release = LINEAR_PLAN.index("RESTORE_SCHEDULING")
    assert successor[0][1].depends_on_step_indexes == [first_release], (
        "the second window starts after the first one released the node"
    )
    for operation in (
        WorkflowOperation.STOP_WORKLOADS,
        WorkflowOperation.RESTART_WORKLOAD,
    ):
        assert len(steps_of(updated, operation)) == 1, f"one {operation.value}"
    (join,) = steps_of(updated, WorkflowOperation.RESTART_WORKLOAD)
    releases = [
        index
        for index, step in enumerate(updated.official_steps)
        if step.operation is WorkflowOperation.RESTORE_SCHEDULING
    ]
    assert len(releases) == 2 and set(releases) <= set(join.depends_on_step_indexes), (
        "the single job restart waits for both windows to release the node"
    )


def test_both_windows_keep_fabric_manager_and_send_stable_bodies() -> None:
    """Re-send stability: the dispatched window's commands keep their ids and
    bodies across the merge, and the second window's are new ones."""

    context, first, running = committed_job("quiesce-waiting")
    quiesce_index = LINEAR_PLAN.index("QUIESCE_GPU_SERVICES")
    before = quiesce_command_parameters(
        running, quiesce_index, running.official_steps[quiesce_index]
    )
    keys_before = {
        index: step_idempotency_key(running, index, running.official_steps[index])
        for index in window_indexes(running)
    }

    job_fault(context, "xid-a1", GPU_B)

    updated = context.store.get_workflow(first["workflow_request_id"])
    after = quiesce_command_parameters(
        updated, quiesce_index, updated.official_steps[quiesce_index]
    )
    assert before["fabric_manager"] == "keep", before
    assert after == before, "a re-sent QUIESCE carries the body it was sent with"
    assert {
        index: step_idempotency_key(updated, index, updated.official_steps[index])
        for index in keys_before
    } == keys_before, "the dispatched window's command ids do not move"
    (index,) = [
        index
        for index, step in enumerate(updated.official_steps)
        if step.operation is QUIESCE and step.gpu_uuids == [GPU_B]
    ]
    successor_quiesce = quiesce_command_parameters(
        updated, index, updated.official_steps[index]
    )
    assert successor_quiesce["fabric_manager"] == "keep", (
        "the second window is a per-GPU reset window too"
    )
    assert step_idempotency_key(updated, index, updated.official_steps[index]) not in (
        keys_before.values()
    ), "the second QUIESCE is a command of its own"


def test_a_fault_with_no_gpu_identity_adds_nothing_to_a_dispatched_window() -> None:
    """Only a merge that would widen the window is redirected: an event with no
    GPU names nothing to add, and the record is left as it was."""

    context, first, running = committed_job("quiesce-succeeded")
    payload = xid_payload(46, "xid-unknown-gpu")
    payload.pop("gpu_uuid")

    asyncio.run(post_faults(context, [("/v1/gpu-events/xid", payload)]))

    updated = context.store.get_workflow(first["workflow_request_id"])
    assert updated.dag_enabled is False
    assert [step.operation.value for step in updated.official_steps] == LINEAR_PLAN
    for index in window_indexes(running):
        assert command_identity(updated.official_steps[index]) == command_identity(
            running.official_steps[index]
        )


def test_a_record_only_merge_never_widens_a_dispatched_window() -> None:
    """The job was stopped by someone else: the event is recorded, and the
    scope pass that follows every flat merge leaves the window alone too."""

    context, first, running = committed_job("quiesce-succeeded")
    withdrawn = copy_model(
        running,
        workload_withdrawn_at=running.updated_at,
        workload_withdrawn_reason="job deleted by its owner",
    )
    context.store.save_workflow(withdrawn)

    job_fault(context, "xid-a1", GPU_B)

    updated = context.store.get_workflow(first["workflow_request_id"])
    for index in window_indexes(running):
        step = updated.official_steps[index]
        assert step.gpu_uuids == [GPU_A], (index, step.operation.value)
        assert step.parameters["gpu_uuids_by_node"] == {NODE: [GPU_A]}, step


@pytest.mark.parametrize("quiesce", [WAITING, SUCCEEDED], ids=["waiting", "succeeded"])
def test_an_idle_node_queues_a_successor_once_the_quiesce_is_dispatched(
    context: ApplicationContext, quiesce: WorkflowStepStatus
) -> None:
    _, _, pending = ingest(
        context, _node_event(48, event_id="idle-gpu-a", gpu_uuid="GPU-a")
    )
    quiesce_index = next(
        index
        for index, step in enumerate(pending.official_steps)
        if step.operation is QUIESCE
    )
    running = run_steps(
        context,
        pending,
        {step.operation: SUCCEEDED for step in pending.official_steps[:quiesce_index]}
        | {QUIESCE: quiesce},
    )

    _, _, second = ingest(
        context, _node_event(48, event_id="idle-gpu-b", gpu_uuid="GPU-b")
    )

    assert second.request_id != running.request_id, "no join: a workflow of its own"
    assert second.predecessor_workflow_id == running.request_id
    for operation in WINDOW:
        (step,) = steps_of(second, operation)
        assert step.gpu_uuids == ["GPU-b"], operation
    kept = context.store.get_workflow(running.request_id)
    assert kept.official_steps == running.official_steps, (
        "the running window keeps what it was dispatched with"
    )


def test_an_idle_node_still_widens_before_its_quiesce(
    context: ApplicationContext,
) -> None:
    _, _, pending = ingest(
        context, _node_event(48, event_id="idle-early-a", gpu_uuid="GPU-a")
    )
    quiesce_index = next(
        index
        for index, step in enumerate(pending.official_steps)
        if step.operation is QUIESCE
    )
    run_steps(
        context,
        pending,
        {step.operation: SUCCEEDED for step in pending.official_steps[:quiesce_index]},
    )

    _, _, second = ingest(
        context, _node_event(48, event_id="idle-early-b", gpu_uuid="GPU-b")
    )

    assert second.request_id == pending.request_id
    for operation in WINDOW:
        (step,) = steps_of(second, operation)
        assert step.gpu_uuids == ["GPU-a", "GPU-b"], operation

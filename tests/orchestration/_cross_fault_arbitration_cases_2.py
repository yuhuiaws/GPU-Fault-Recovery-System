from __future__ import annotations

import asyncio
from datetime import timedelta

import pytest

from gpu_fault.app import ApplicationContext
from gpu_fault.models import WorkflowOperation, WorkflowStatus, WorkflowStepStatus
from gpu_fault.orchestrator import IncidentOrchestrator
from gpu_fault.watcher import AttemptObservation, WorkloadPhase
from tests._builders import (
    build_context,
    container_observation,
    workflow_step_execution,
)
from tests.orchestration._cross_fault_support import (
    NOW,
    observation,
    post_faults,
    sxid_payload,
    xid_payload,
)


def two_node_observation() -> AttemptObservation:
    return observation().model_copy(
        update={
            "expected_critical_ranks": 2,
            "containers": [
                container_observation(
                    "pod-a",
                    "trainer-a",
                    0,
                    "node-a",
                    role="master",
                    gpu_uuids=["GPU-a"],
                ),
                container_observation(
                    "pod-b", "trainer-b", 1, "node-b", gpu_uuids=["GPU-b"]
                ),
            ],
        }
    )


def restarted_attempt_context(
    *,
    baseline_xid: int = 11,
    baseline_status: WorkflowStatus = WorkflowStatus.SUCCEEDED,
) -> tuple[ApplicationContext, str, str]:
    context = build_context()
    old = observation(started_at=NOW - timedelta(minutes=1))
    context.store.save_attempt_observation(old)
    baseline = asyncio.run(
        post_faults(
            context,
            [("/v1/gpu-events/xid", xid_payload(baseline_xid, "baseline-recovery"))],
        )
    )[0]
    workflow = context.store.get_workflow(baseline["workflow_request_id"])
    current_attempt_id = "train-a002"
    context.store.save_workflow(
        workflow.model_copy(
            update={
                "status": baseline_status,
                "execution_owner_id": (
                    "executor-a" if baseline_status is WorkflowStatus.RUNNING else None
                ),
                "step_executions": [
                    workflow_step_execution(
                        0,
                        WorkflowOperation.RESTART_WORKLOAD,
                        details={"restart_attempt_id": current_attempt_id},
                    )
                ],
            }
        )
    )
    context.store.save_attempt_observation(
        old.model_copy(
            update={
                "workload_phase": WorkloadPhase.STOPPED,
                "observed_at": NOW + timedelta(seconds=10),
            }
        )
    )
    context.store.save_attempt_observation(
        observation(
            attempt_id=current_attempt_id,
            started_at=NOW + timedelta(seconds=60),
            observed_at=NOW + timedelta(seconds=70),
        )
    )
    return (context, baseline["incident_id"], baseline["workflow_request_id"])


@pytest.mark.parametrize(
    "restart_status", [WorkflowStepStatus.WAITING, WorkflowStepStatus.SUCCEEDED]
)
def test_late_node_after_restart_finalization_uses_new_workflow(
    restart_status: WorkflowStepStatus,
) -> None:
    context = build_context()
    context.orchestrator = IncidentOrchestrator(
        context.store, workflow_preemption_enabled=True
    )
    observation_value = two_node_observation()
    observation_value = observation_value.model_copy(
        update={
            "expected_critical_ranks": 3,
            "containers": [
                *observation_value.containers,
                container_observation(
                    "pod-c", "trainer-c", 2, "node-c", gpu_uuids=["GPU-c"]
                ),
            ],
        }
    )
    context.store.save_attempt_observation(observation_value)
    first = asyncio.run(
        post_faults(
            context, [("/v1/gpu-events/xid", xid_payload(48, "finalization-node-a"))]
        )
    )[0]
    node_b = xid_payload(79, "finalization-node-b")
    node_b.update({"node_id": "node-b", "gpu_uuid": "GPU-b", "pci_bdf": "0000:ca:00.0"})
    asyncio.run(post_faults(context, [("/v1/gpu-events/xid", node_b)]))
    original = context.store.get_workflow(first["workflow_request_id"])
    restart_index = next(
        index
        for index, step in enumerate(original.official_steps)
        if step.operation is WorkflowOperation.RESTART_WORKLOAD
    )
    execution = workflow_step_execution(
        restart_index,
        WorkflowOperation.RESTART_WORKLOAD,
        restart_status,
        adapter_operation_id="restart/finalization",
    )
    original = original.model_copy(
        update={
            "status": WorkflowStatus.RUNNING,
            "step_executions": [execution],
            "completed_step_indexes": (
                [restart_index]
                if restart_status is WorkflowStepStatus.SUCCEEDED
                else []
            ),
            "completed_operations": (
                [WorkflowOperation.RESTART_WORKLOAD]
                if restart_status is WorkflowStepStatus.SUCCEEDED
                else []
            ),
        }
    )
    context.store.save_workflow(original)
    original_snapshot = original.model_dump(mode="json")
    node_c = xid_payload(48, f"finalization-node-c-{restart_status.value}")
    node_c.update({"node_id": "node-c", "gpu_uuid": "GPU-c", "pci_bdf": "0000:db:00.0"})

    late = asyncio.run(post_faults(context, [("/v1/gpu-events/xid", node_c)]))[0]

    assert late["workflow_request_id"] != original.request_id
    assert (
        context.store.get_workflow(original.request_id).model_dump(mode="json")
        == original_snapshot
    )
    successor = context.store.get_workflow(late["workflow_request_id"])
    assert successor.predecessor_workflow_id == original.request_id
    assert not successor.preempt_predecessor
    assert {
        node_id
        for step in successor.official_steps
        if step.operation
        not in {
            WorkflowOperation.CHECKPOINT_WORKLOADS,
            WorkflowOperation.STOP_WORKLOADS,
            WorkflowOperation.RESTART_WORKLOAD,
        }
        for node_id in step.node_ids
    } == {"node-c"}


def test_dag_compares_recovery_rank_within_the_target_node_branch() -> None:
    context = build_context()
    context.orchestrator = IncidentOrchestrator(
        context.store, workflow_preemption_enabled=True
    )
    context.store.save_attempt_observation(two_node_observation())
    low = asyncio.run(
        post_faults(
            context,
            [("/v1/gpu-events/xid", xid_payload(48, "branch-rank-node-a-reset"))],
        )
    )[0]
    node_b = xid_payload(79, "branch-rank-node-b-reboot")
    node_b.update({"node_id": "node-b", "gpu_uuid": "GPU-b", "pci_bdf": "0000:ca:00.0"})
    asyncio.run(post_faults(context, [("/v1/gpu-events/xid", node_b)]))

    upgraded = asyncio.run(
        post_faults(
            context,
            [("/v1/gpu-events/xid", xid_payload(79, "branch-rank-node-a-reboot"))],
        )
    )[0]

    workflow = context.store.get_workflow(upgraded["workflow_request_id"])
    assert workflow.request_id == low["workflow_request_id"]
    assert (
        sum(
            step.operation is WorkflowOperation.RESTART_NODE
            and step.node_ids == ["node-a"]
            for step in workflow.official_steps
        )
        == 1
    )
    assert (
        sum(
            step.operation is WorkflowOperation.RESTART_NODE
            and step.node_ids == ["node-b"]
            for step in workflow.official_steps
        )
        == 1
    )
    node_a_reset_indexes = {
        index
        for index, step in enumerate(workflow.official_steps)
        if step.operation is WorkflowOperation.RESET_GPU and step.node_ids == ["node-a"]
    }
    assert node_a_reset_indexes
    assert node_a_reset_indexes <= set(workflow.superseded_step_indexes)
    assert (
        sum(
            step.operation is WorkflowOperation.STOP_WORKLOADS
            for step in workflow.official_steps
        )
        == 1
    )
    assert (
        sum(
            step.operation is WorkflowOperation.RESTART_WORKLOAD
            for step in workflow.official_steps
        )
        == 1
    )


@pytest.mark.parametrize("completed_through_quiesce", [False, True])
def test_dag_widens_unstarted_same_rank_branch_for_another_gpu(
    completed_through_quiesce: bool,
) -> None:
    context = build_context()
    context.orchestrator = IncidentOrchestrator(
        context.store, workflow_preemption_enabled=True
    )
    observation_value = two_node_observation()
    observation_value.containers[0].gpu_uuids.append("GPU-a2")
    context.store.save_attempt_observation(observation_value)
    first = asyncio.run(
        post_faults(
            context, [("/v1/gpu-events/xid", xid_payload(48, "branch-scope-gpu-a"))]
        )
    )[0]
    node_b = xid_payload(79, "branch-scope-node-b-reboot")
    node_b.update({"node_id": "node-b", "gpu_uuid": "GPU-b", "pci_bdf": "0000:ca:00.0"})
    asyncio.run(post_faults(context, [("/v1/gpu-events/xid", node_b)]))
    before = context.store.get_workflow(first["workflow_request_id"])
    completed_indexes = []
    if completed_through_quiesce:
        completed_operations = {
            WorkflowOperation.FREEZE_EVIDENCE,
            WorkflowOperation.MARK_UNSCHEDULABLE,
            WorkflowOperation.STOP_WORKLOADS,
            WorkflowOperation.QUIESCE_GPU_SERVICES,
        }
        completed_indexes = [
            index
            for index, step in enumerate(before.official_steps)
            if step.operation in completed_operations
        ]
        context.store.save_workflow(
            before.model_copy(
                update={
                    "status": WorkflowStatus.RUNNING,
                    "completed_step_indexes": completed_indexes,
                    "completed_operations": [
                        before.official_steps[index].operation
                        for index in completed_indexes
                    ],
                    "step_executions": [
                        workflow_step_execution(
                            index, before.official_steps[index].operation
                        )
                        for index in completed_indexes
                    ],
                }
            )
        )
    second_gpu = xid_payload(48, "branch-scope-gpu-a2")
    second_gpu.update({"gpu_uuid": "GPU-a2", "pci_bdf": "0000:bb:00.0"})

    result = asyncio.run(post_faults(context, [("/v1/gpu-events/xid", second_gpu)]))[0]

    workflow = context.store.get_workflow(result["workflow_request_id"])
    assert workflow.request_id == first["workflow_request_id"]
    resets = [
        step
        for step in workflow.official_steps
        if step.operation is WorkflowOperation.RESET_GPU and step.node_ids == ["node-a"]
    ]
    assert len(resets) == 1
    assert set(resets[0].gpu_uuids) == {"GPU-a", "GPU-a2"}
    assert set(completed_indexes) <= set(workflow.completed_step_indexes)
    assert (
        sum(
            step.operation is WorkflowOperation.STOP_WORKLOADS
            for step in workflow.official_steps
        )
        == 1
    )
    assert (
        sum(
            step.operation is WorkflowOperation.RESTART_WORKLOAD
            for step in workflow.official_steps
        )
        == 1
    )


def test_dag_appends_same_node_successor_after_branch_started() -> None:
    context = build_context()
    context.orchestrator = IncidentOrchestrator(
        context.store, workflow_preemption_enabled=True
    )
    observation_value = two_node_observation()
    observation_value.containers[0].gpu_uuids.append("GPU-a2")
    context.store.save_attempt_observation(observation_value)
    first = asyncio.run(
        post_faults(
            context, [("/v1/gpu-events/xid", xid_payload(48, "branch-running-gpu-a"))]
        )
    )[0]
    node_b = xid_payload(79, "branch-running-node-b-reboot")
    node_b.update({"node_id": "node-b", "gpu_uuid": "GPU-b", "pci_bdf": "0000:ca:00.0"})
    asyncio.run(post_faults(context, [("/v1/gpu-events/xid", node_b)]))
    workflow = context.store.get_workflow(first["workflow_request_id"])
    node_a_indexes = context.orchestrator._brancher.node_branch_step_indexes(
        workflow, "node-a"
    )
    started_index = next(
        index
        for index in node_a_indexes
        if workflow.official_steps[index].operation is WorkflowOperation.RESET_GPU
    )
    original_reset = workflow.official_steps[started_index]
    started_execution = workflow_step_execution(
        started_index,
        WorkflowOperation.RESET_GPU,
        WorkflowStepStatus.WAITING,
        adapter_operation_id="remote/reset-gpu-a",
    )
    context.store.save_workflow(
        workflow.model_copy(
            update={
                "status": WorkflowStatus.RUNNING,
                "step_executions": [started_execution],
            }
        )
    )
    second_gpu = xid_payload(48, "branch-running-gpu-a2")
    second_gpu.update({"gpu_uuid": "GPU-a2", "pci_bdf": "0000:bb:00.0"})

    result = asyncio.run(post_faults(context, [("/v1/gpu-events/xid", second_gpu)]))[0]

    updated = context.store.get_workflow(result["workflow_request_id"])
    assert updated.request_id == workflow.request_id
    resets = [
        (index, step)
        for index, step in enumerate(updated.official_steps)
        if step.operation is WorkflowOperation.RESET_GPU and step.node_ids == ["node-a"]
    ]
    assert len(resets) == 2
    assert resets[0] == (started_index, original_reset)
    assert (
        next(
            execution
            for execution in updated.step_executions
            if execution.step_index == started_index
        )
        == started_execution
    )
    assert set(resets[1][1].gpu_uuids) == {"GPU-a2"}
    successor_branch_id = resets[1][1].branch_id
    successor_indexes = [
        index
        for index, step in enumerate(updated.official_steps)
        if step.branch_id == successor_branch_id
    ]
    assert (
        max(node_a_indexes)
        in updated.official_steps[min(successor_indexes)].depends_on_step_indexes
    )
    assert (
        sum(
            step.operation is WorkflowOperation.STOP_WORKLOADS
            for step in updated.official_steps
        )
        == 1
    )
    assert (
        sum(
            step.operation is WorkflowOperation.RESTART_WORKLOAD
            for step in updated.official_steps
        )
        == 1
    )
    assert len(context.store.list_workflows(limit=10)) == 1


def test_running_weaker_workflow_queues_stronger_successor() -> None:
    context = build_context()
    context.store.save_attempt_observation(observation())
    first = asyncio.run(
        post_faults(
            context, [("/v1/gpu-events/xid", xid_payload(11, "running-weaker-xid"))]
        )
    )[0]
    workflow = context.store.get_workflow(first["workflow_request_id"])
    context.store.save_workflow(
        workflow.model_copy(
            update={
                "status": WorkflowStatus.RUNNING,
                "execution_owner_id": "executor-a",
            }
        )
    )

    second = asyncio.run(
        post_faults(
            context, [("/v1/gpu-events/sxid", sxid_payload("later-stronger-sxid"))]
        )
    )[0]

    assert second["incident_id"] == first["incident_id"]
    assert second["workflow_request_id"] != first["workflow_request_id"]
    successor = context.store.get_workflow(second["workflow_request_id"])
    assert successor.predecessor_workflow_id == (first["workflow_request_id"])
    assert successor.fencing_token == workflow.fencing_token
    assert (
        context.store.get_incident(second["incident_id"]).fencing_token
        == successor.fencing_token
    )
    assert WorkflowOperation.RESET_ALL_GPUS_NVSWITCHES in {
        step.operation for step in successor.official_steps
    }
    assert len(context.store.list_workflows()) == 2


def test_preemption_marks_stronger_successor_and_reuses_containment() -> None:
    context = build_context()
    context.orchestrator = IncidentOrchestrator(
        context.store, workflow_preemption_enabled=True
    )
    context.store.save_attempt_observation(observation())
    first = asyncio.run(
        post_faults(
            context, [("/v1/gpu-events/xid", xid_payload(48, "preempt-weaker-xid"))]
        )
    )[0]
    workflow = context.store.get_workflow(first["workflow_request_id"])
    inherited_operations = {
        WorkflowOperation.MARK_UNSCHEDULABLE,
        WorkflowOperation.STOP_WORKLOADS,
    }
    completed_indexes = [
        index
        for index, step in enumerate(workflow.official_steps)
        if step.operation in inherited_operations
    ]
    assert len(completed_indexes) == 2
    context.store.save_workflow(
        workflow.model_copy(
            update={
                "status": WorkflowStatus.RUNNING,
                "execution_owner_id": "executor-a",
                "completed_step_indexes": completed_indexes,
                "completed_operations": [
                    workflow.official_steps[index].operation
                    for index in completed_indexes
                ],
                "step_executions": [
                    workflow_step_execution(
                        index, workflow.official_steps[index].operation
                    )
                    for index in completed_indexes
                ],
            }
        )
    )

    second = asyncio.run(
        post_faults(
            context, [("/v1/gpu-events/sxid", sxid_payload("preempt-stronger-sxid"))]
        )
    )[0]

    successor = context.store.get_workflow(second["workflow_request_id"])
    assert successor.predecessor_workflow_id == workflow.request_id
    assert successor.preempt_predecessor
    assert "rank 30 -> 40" in (successor.preemption_reason or "")
    inherited = {
        successor.official_steps[index].operation
        for index in successor.inherited_step_indexes
    }
    assert inherited == inherited_operations
    assert set(successor.inherited_step_indexes) == set(
        successor.completed_step_indexes
    )
    assert all(
        execution.details["inherited_from_workflow_id"] == workflow.request_id
        for execution in successor.step_executions
    )


def test_running_same_action_widens_unsubmitted_reset_for_new_gpu() -> None:
    context = build_context()
    two_gpu = observation()
    two_gpu = two_gpu.model_copy(
        update={
            "containers": [
                two_gpu.containers[0].model_copy(
                    update={"gpu_uuids": ["GPU-a", "GPU-b"]}
                )
            ]
        }
    )
    context.store.save_attempt_observation(two_gpu)
    first_payload = xid_payload(109, "running-reset-gpu-a")
    first = asyncio.run(post_faults(context, [("/v1/gpu-events/xid", first_payload)]))[
        0
    ]
    workflow = context.store.get_workflow(first["workflow_request_id"])
    context.store.save_workflow(
        workflow.model_copy(
            update={
                "status": WorkflowStatus.RUNNING,
                "execution_owner_id": "executor-a",
            }
        )
    )
    second_payload = xid_payload(109, "running-reset-gpu-b")
    second_payload["gpu_uuid"] = "GPU-b"

    second = asyncio.run(
        post_faults(context, [("/v1/gpu-events/xid", second_payload)])
    )[0]

    assert second["incident_id"] == first["incident_id"]
    assert second["workflow_request_id"] == first["workflow_request_id"]
    widened = context.store.get_workflow(second["workflow_request_id"])
    assert widened.fencing_token == workflow.fencing_token
    resets = [
        step
        for step in widened.official_steps
        if step.operation is WorkflowOperation.RESET_GPU
    ]
    assert len(resets) == 1
    assert set(resets[0].gpu_uuids) == {"GPU-a", "GPU-b"}


def test_cross_node_sxid_only_resets_its_hardware_node() -> None:
    context = build_context()
    context.store.save_attempt_observation(two_node_observation())
    sxid = sxid_payload("cross-node-sxid")
    sxid.update(
        {
            "node_id": "node-b",
            "fabric_partition": "cluster-a/node-b/local-nvswitch",
            "participating_gpu_uuids": ["GPU-b"],
        }
    )

    results = asyncio.run(
        post_faults(
            context,
            [
                ("/v1/gpu-events/xid", xid_payload(11, "cross-node-xid")),
                ("/v1/gpu-events/sxid", sxid),
            ],
        )
    )

    incident = context.store.get_incident(results[-1]["incident_id"])
    workflow = context.store.get_workflow(results[-1]["workflow_request_id"])
    reset = next(
        step
        for step in workflow.official_steps
        if step.operation is WorkflowOperation.RESET_ALL_GPUS_NVSWITCHES
    )
    stop = next(
        step
        for step in workflow.official_steps
        if step.operation is WorkflowOperation.STOP_WORKLOADS
    )
    assert incident.node_ids == ["node-a", "node-b"]
    assert reset.node_ids == ["node-b"]
    assert reset.parameters["gpu_uuids_by_node"] == {"node-b": ["GPU-b"]}
    assert stop.node_ids == ["node-a", "node-b"]


def test_stale_same_rank_xid_is_ignored_after_restart() -> None:
    context, incident_id, workflow_id = restarted_attempt_context()
    payload = xid_payload(11, "stale-same-rank-xid")
    payload.update(
        {
            "observed_at": (NOW + timedelta(seconds=40)).isoformat(),
            "collected_at": (NOW + timedelta(seconds=69)).isoformat(),
        }
    )

    result = asyncio.run(post_faults(context, [("/v1/gpu-events/xid", payload)]))[0]

    assert result["incident_id"] == incident_id
    assert result["workflow_request_id"] == workflow_id
    assert len(context.store.list_workflows()) == 1
    incident = context.store.get_incident(incident_id)
    assert any(
        "Ignored stale XID 11 action" in reason
        and "current_attempt=train-a002" in reason
        for reason in incident.reasons
    )
    assert (
        context.store.get_incident_by_event("stale-same-rank-xid").incident_id
        == incident_id
    )


def test_source_event_time_takes_precedence_over_collection_time() -> None:
    context, incident_id, workflow_id = restarted_attempt_context()
    payload = xid_payload(11, "delayed-source-clock-xid")
    payload.update(
        {
            "observed_at": (NOW + timedelta(seconds=70)).isoformat(),
            "source_event_time": (NOW + timedelta(seconds=40)).isoformat(),
            "collected_at": (NOW + timedelta(seconds=69)).isoformat(),
        }
    )

    result = asyncio.run(post_faults(context, [("/v1/gpu-events/xid", payload)]))[0]

    assert result["incident_id"] == incident_id
    assert result["workflow_request_id"] == workflow_id
    assert len(context.store.list_workflows()) == 1
    assert any(
        f"event_time={(NOW + timedelta(seconds=40)).isoformat()}" in reason
        for reason in context.store.get_incident(incident_id).reasons
    )


def test_stale_weaker_sxid_is_ignored_after_restart() -> None:
    context, incident_id, workflow_id = restarted_attempt_context(baseline_xid=79)
    payload = sxid_payload("stale-weaker-sxid")
    payload.update(
        {
            "observed_at": (NOW + timedelta(seconds=70)).isoformat(),
            "source_event_time": (NOW + timedelta(seconds=40)).isoformat(),
        }
    )

    result = asyncio.run(post_faults(context, [("/v1/gpu-events/sxid", payload)]))[0]

    assert result["incident_id"] == incident_id
    assert result["workflow_request_id"] == workflow_id
    assert len(context.store.list_workflows()) == 1
    assert any(
        "Ignored stale SXID 11001 action" in reason
        for reason in context.store.get_incident(incident_id).reasons
    )


def test_stale_stronger_action_queues_successor() -> None:
    context, incident_id, workflow_id = restarted_attempt_context()
    payload = xid_payload(79, "stale-stronger-xid")
    payload.update(
        {
            "observed_at": (NOW + timedelta(seconds=70)).isoformat(),
            "source_event_time": (NOW + timedelta(seconds=40)).isoformat(),
        }
    )

    result = asyncio.run(post_faults(context, [("/v1/gpu-events/xid", payload)]))[0]

    assert result["incident_id"] == incident_id
    assert result["workflow_request_id"] != workflow_id
    successor = context.store.get_workflow(result["workflow_request_id"])
    assert successor.predecessor_workflow_id == workflow_id
    assert WorkflowOperation.RESTART_NODE in {
        step.operation for step in successor.official_steps
    }


def test_event_after_current_attempt_start_is_processed() -> None:
    context, _, workflow_id = restarted_attempt_context()
    payload = xid_payload(11, "fresh-current-generation-xid")
    payload.update(
        {
            "observed_at": (NOW + timedelta(seconds=70)).isoformat(),
            "source_event_time": (NOW + timedelta(seconds=65)).isoformat(),
        }
    )

    result = asyncio.run(post_faults(context, [("/v1/gpu-events/xid", payload)]))[0]

    assert result["workflow_request_id"] != workflow_id
    assert len(context.store.list_workflows()) == 2


def test_fresh_weaker_diagnostic_can_run_with_previous_attempt() -> None:
    context, incident_id, workflow_id = restarted_attempt_context(
        baseline_xid=79, baseline_status=WorkflowStatus.RUNNING
    )
    payload = xid_payload(11, "fresh-weaker-while-restarting")
    payload.update(
        {
            "observed_at": (NOW + timedelta(seconds=70)).isoformat(),
            "source_event_time": (NOW + timedelta(seconds=65)).isoformat(),
        }
    )

    result = asyncio.run(post_faults(context, [("/v1/gpu-events/xid", payload)]))[0]

    assert result["incident_id"] != incident_id
    assert result["workflow_request_id"] != workflow_id
    successor = context.store.get_workflow(result["workflow_request_id"])
    assert successor.predecessor_workflow_id is None
    assert WorkflowOperation.RESTART_WORKLOAD in {
        step.operation for step in successor.official_steps
    }
    assert WorkflowOperation.RESTART_NODE not in {
        step.operation for step in successor.official_steps
    }
    assert len(context.store.list_workflows()) == 2


def test_fresh_sxid_waits_for_running_previous_attempt() -> None:
    context, incident_id, workflow_id = restarted_attempt_context(
        baseline_xid=79, baseline_status=WorkflowStatus.RUNNING
    )
    payload = sxid_payload("fresh-sxid-while-restarting")
    payload.update(
        {
            "observed_at": (NOW + timedelta(seconds=70)).isoformat(),
            "source_event_time": (NOW + timedelta(seconds=65)).isoformat(),
        }
    )

    result = asyncio.run(post_faults(context, [("/v1/gpu-events/sxid", payload)]))[0]

    assert result["incident_id"] != incident_id
    successor = context.store.get_workflow(result["workflow_request_id"])
    assert successor.predecessor_workflow_id == workflow_id
    assert WorkflowOperation.RESET_ALL_GPUS_NVSWITCHES in {
        step.operation for step in successor.official_steps
    }
    assert WorkflowOperation.RESTART_NODE not in {
        step.operation for step in successor.official_steps
    }


def test_missing_attempt_start_time_keeps_fail_safe_behavior() -> None:
    context, _, workflow_id = restarted_attempt_context()
    context.store.save_attempt_observation(
        observation(attempt_id="train-a002", observed_at=NOW + timedelta(seconds=71))
    )
    payload = xid_payload(11, "unknown-generation-xid")
    payload.update(
        {
            "observed_at": (NOW + timedelta(seconds=71)).isoformat(),
            "source_event_time": (NOW + timedelta(seconds=40)).isoformat(),
        }
    )

    result = asyncio.run(post_faults(context, [("/v1/gpu-events/xid", payload)]))[0]

    assert result["workflow_request_id"] != workflow_id
    assert len(context.store.list_workflows()) == 2

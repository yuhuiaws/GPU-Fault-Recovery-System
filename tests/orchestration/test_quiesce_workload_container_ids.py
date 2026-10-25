"""A managed job's QUIESCE carries its runtime container ids per node.

``EvidenceOperationService.quiesce_parameters`` froze only the containers'
``cgroup_path`` (a ``gpu-fault.io/cgroup-path`` Pod annotation nothing sets)
into the QUIESCE step, so the plan of every managed job named no workload
scope and the device sweep never reached the stopped job's survivors (the
reset then failed closed in VERIFY_NO_GPU_CLIENTS). The plan now also carries
``workload_container_ids_by_node``: the kubelet's ``containerID`` of every
running container without an explicit path, the same parameter the hung DAG
carries (``gpu_fault.workload_container_scope``), and only into a QUIESCE the
plan precedes with the attempt's STOP_WORKLOADS: the sweep then meets survivors
of a stopped attempt, never a job left running. It is frozen into the persisted
step, so every send of one QUIESCE carries the same body.
"""

from __future__ import annotations

import asyncio
import hashlib

import pytest

from gpu_fault.models import (
    IncidentState,
    RecoveryAction,
    WorkflowOperation,
    WorkflowStatus,
)
from gpu_fault.orchestration.arbitration import RecoveryArbiter
from gpu_fault.orchestration.dag_branching import DagBrancher
from gpu_fault.orchestration.families.evidence import (
    EvidenceOperationService,
    stops_attempt_before,
)
from gpu_fault.orchestration.families.grouped_faults import (
    GroupedFaultCallbacks,
    GroupedFaultContext,
    GroupedFaultService,
)
from gpu_fault.policy import ActionDisposition, FaultPolicyDecision
from tests._builders import (
    build_context,
    build_store,
    container_observation,
    fault_incident,
    workflow_request,
    workflow_step,
)
from tests.orchestration._cov95_runtime_evidence import EvidenceHarness
from tests.orchestration._cross_fault_support import (
    observation,
    post_faults,
    xid_payload,
)
from tests.orchestration._support import WorkloadState, event

TRAINER = hashlib.sha256(b"trainer-container").hexdigest()
PEER = hashlib.sha256(b"second-trainer-container").hexdigest()
ANNOTATED = hashlib.sha256(b"annotated-container").hexdigest()


def test_the_quiesce_plan_names_each_nodes_running_containers_by_runtime_id() -> None:
    h = EvidenceHarness()
    managed = h.observation(
        containers=[
            container_observation(
                "pod-a", "trainer", 0, "node-a", container_id=f"containerd://{TRAINER}"
            ),
            container_observation(
                "pod-b", "trainer", 1, "node-b", container_id=f"containerd://{PEER}"
            ),
            # Scoped by its own annotation already.
            container_observation(
                "pod-c",
                "trainer",
                2,
                "node-b",
                container_id=f"containerd://{ANNOTATED}",
                cgroup_path="/kubepods/pod-c",
            ),
            # Finished: nothing of it is the job's to sweep any more.
            container_observation(
                "pod-d",
                "trainer",
                3,
                "node-c",
                container_id=f"containerd://{'d' * 64}",
                terminated=True,
                exit_code=0,
            ),
            # An id no node could match proves nothing.
            container_observation(
                "pod-e", "trainer", 4, "node-c", container_id="containerd://abc123"
            ),
        ]
    )
    parameters = {"timeout_seconds": 30}

    result = h.service.quiesce_parameters(parameters, managed, attempt_stopped=True)

    assert result == {
        "timeout_seconds": 30,
        "workload_cgroup_paths_by_node": {"node-b": ["/kubepods/pod-c"]},
        "workload_container_ids_by_node": {"node-a": [TRAINER], "node-b": [PEER]},
    }, result
    assert parameters == {"timeout_seconds": 30}, "shared parameters are not mutated"
    unstopped = h.service.quiesce_parameters(parameters, managed)
    assert unstopped == {
        "timeout_seconds": 30,
        "workload_cgroup_paths_by_node": {"node-b": ["/kubepods/pod-c"]},
    }, "a plan that does not stop the attempt first never sweeps its containers"


def test_a_plan_without_any_scope_is_returned_as_it_came() -> None:
    h = EvidenceHarness()
    unknown = h.observation(
        containers=[container_observation("pod-a", "trainer", 0, "node-a")]
    )
    parameters = {"timeout_seconds": 30}

    assert (
        h.service.quiesce_parameters(parameters, unknown, attempt_stopped=True)
        is parameters
    )


def test_the_ids_need_a_stop_of_the_attempt_earlier_in_the_plan() -> None:
    stop = workflow_step(WorkflowOperation.STOP_WORKLOADS, node_ids=["node-a"])
    quiesce = workflow_step(WorkflowOperation.QUIESCE_GPU_SERVICES, node_ids=["node-a"])
    reset = workflow_step(WorkflowOperation.RESET_GPU, node_ids=["node-a"])

    assert stops_attempt_before([stop, quiesce, reset], 1) is True
    assert stops_attempt_before([quiesce, reset], 0) is False
    assert stops_attempt_before([quiesce, stop, reset], 0) is False, (
        "a stop after the quiesce does not make its survivors"
    )


def test_a_managed_jobs_reset_plan_quiesces_with_its_container_ids() -> None:
    """An XID 46 on a node of a running managed job, as the kernel collector posts it."""

    context = build_context()
    context.store.save_attempt_observation(
        observation().model_copy(
            update={
                "containers": [
                    container_observation(
                        "pod-a",
                        "trainer-a",
                        0,
                        "node-a",
                        gpu_uuids=["GPU-a"],
                        container_id=f"containerd://{TRAINER}",
                    )
                ]
            }
        )
    )

    (response,) = asyncio.run(
        post_faults(context, [("/v1/gpu-events/xid", xid_payload(46, "xid-46-a"))])
    )

    workflow = context.store.get_workflow(response["workflow_request_id"])
    operations = [step.operation for step in workflow.official_steps]
    stop = operations.index(WorkflowOperation.STOP_WORKLOADS)
    quiesce = operations.index(WorkflowOperation.QUIESCE_GPU_SERVICES)
    assert stop < quiesce, "the job is stopped before the window opens"
    parameters = workflow.official_steps[quiesce].parameters
    assert parameters["workload_container_ids_by_node"] == {"node-a": [TRAINER]}, (
        parameters
    )
    assert "workload_cgroup_paths_by_node" not in parameters, parameters


def _grouped(evidence: EvidenceOperationService) -> GroupedFaultService:
    unused = lambda *args, **kwargs: None  # noqa: E731
    arbiter = RecoveryArbiter()
    return GroupedFaultService(
        build_store(),
        arbiter,
        DagBrancher(arbiter),
        GroupedFaultCallbacks(
            active_job_recovery_workflow=unused,
            active_node_exclusive_workflow=unused,
            aggregation_deadlines=unused,
            attempt_group_key=unused,
            attempt_observation=unused,
            build_workflow=unused,
            claims_node_exclusively=unused,
            generation_fence=unused,
            incident_state_for_workflow=lambda workflow: IncidentState.ACTION_PENDING,
            merge_disposition=unused,
            prepare_preempting_successor=unused,
            preempt_parallel_job_branch=unused,
            quiesce_parameters=evidence.quiesce_parameters,
            reopen_if_terminal=unused,
            runtime_effective_action=unused,
            widen_node_action_scope=lambda workflow, scope: workflow,
        ),
        aggregation_window_seconds=30,
        workflow_preemption_enabled=True,
        attempt_group_actions=set(),
        groupable_operations=set(),
        official_operation={},
    )


@pytest.mark.parametrize("stopped", [True, False])
def test_the_grouped_plan_freezes_the_ids_only_behind_the_attempts_stop(
    stopped: bool,
) -> None:
    h = EvidenceHarness()
    managed = h.observation(
        containers=[
            container_observation(
                "pod-a",
                "trainer",
                0,
                "node-a",
                gpu_uuids=["GPU-a"],
                container_id=f"containerd://{TRAINER}",
            )
        ]
    )
    xid = event(
        48,
        event_id="xid48-a",
        workload_state=WorkloadState.ACTIVE,
        affected_workload_ids=["training/job/job-a"],
    )
    context = GroupedFaultContext(
        event=xid,
        decision=FaultPolicyDecision.model_construct(
            event_id=xid.event_id,
            action=RecoveryAction.RESET_GPU,
            official_action="RESET_GPU",
            disposition=ActionDisposition.EXECUTABLE,
            reasons=[],
        ),
        observation=managed,
        group_key='["cluster-a","job-a","attempt-a"]',
        allocation_nodes=["node-a"],
        workload_ids=["training/job/job-a"],
    )
    steps = [
        workflow_step(WorkflowOperation.MARK_UNSCHEDULABLE, node_ids=["node-a"]),
        *(
            [workflow_step(WorkflowOperation.STOP_WORKLOADS, node_ids=["node-a"])]
            if stopped
            else []
        ),
        workflow_step(WorkflowOperation.QUIESCE_GPU_SERVICES, node_ids=["node-a"]),
        workflow_step(WorkflowOperation.VERIFY_NO_GPU_CLIENTS, node_ids=["node-a"]),
        workflow_step(WorkflowOperation.RESET_GPU, node_ids=["node-a"]),
    ]
    workflow = workflow_request(
        "wf-a", "inc-a", status=WorkflowStatus.PENDING, official_steps=steps
    )
    incident = fault_incident("inc-a", "xid48-a", workflow_request_id="wf-a")

    _, scoped = _grouped(h.service)._scope_workflow(context, incident, workflow, None)

    (quiesce,) = [
        step
        for step in scoped.official_steps
        if step.operation is WorkflowOperation.QUIESCE_GPU_SERVICES
    ]
    if stopped:
        assert quiesce.parameters["workload_container_ids_by_node"] == {
            "node-a": [TRAINER]
        }, quiesce.parameters
    else:
        assert "workload_container_ids_by_node" not in quiesce.parameters, (
            "a plan that never stops the job must not sweep its running processes"
        )

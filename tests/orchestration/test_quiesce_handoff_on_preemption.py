"""Who un-quiesces a node when the reset that quiesced it is preempted by a
reboot (F-C1).

``QUIESCE_GPU_SERVICES`` stops the node's GPU services; the reset plan pairs it
with a ``RESTORE_GPU_SERVICES`` further down. Preempting that plan retires the
paired restore along with the reset, and the reboot successor has no restore of
its own -- it does not need one before the reboot, because the reboot stops
everything anyway. Without a handoff the node comes back up with its GPU
services still masked and nothing in the plan to bring them back.

``WorkflowMergeService._handoff_quiesce`` re-inserts one restore into the
successor, right after the reboot, tagged
``preemption_quiesce_handoff_after_reboot`` so the audit trail says why a
reboot plan contains a step the planner never compiles into it. Only one
incidental reference existed (``test_xid.py``, inside a much larger
end-to-end assertion), so none of the conditions were pinned.
"""

from __future__ import annotations

from gpu_fault.models import WorkflowOperation, WorkflowRequest, WorkflowStatus
from gpu_fault.operation_registry import (
    NODE_EXCLUSIVE_OPERATIONS,
    WORKLOAD_SCOPED_OPERATIONS,
)
from gpu_fault.orchestration.arbitration import RecoveryArbiter
from gpu_fault.orchestration.dag_branching import DagBrancher
from gpu_fault.orchestration.families.conflicts import NodeConflictService
from gpu_fault.orchestration.workflow_merge import WorkflowMergeService
from tests._builders import workflow_request, workflow_step

HANDOFF = "preemption_quiesce_handoff_after_reboot"
QUIESCE = WorkflowOperation.QUIESCE_GPU_SERVICES
RESTORE_SERVICES = WorkflowOperation.RESTORE_GPU_SERVICES
RESET = WorkflowOperation.RESET_GPU
REBOOT = WorkflowOperation.RESTART_NODE
REPLACE = WorkflowOperation.REPLACE_NODE

# The reset plan as the builder compiles it: quiesce, reset, restore.
RESET_PLAN = [
    WorkflowOperation.FREEZE_EVIDENCE,
    WorkflowOperation.MARK_UNSCHEDULABLE,
    QUIESCE,
    WorkflowOperation.VERIFY_NO_GPU_CLIENTS,
    RESET,
    RESTORE_SERVICES,
    WorkflowOperation.VALIDATE_GPU,
    WorkflowOperation.RESTORE_SCHEDULING,
]
# The reboot plan: no quiesce and no restore of its own.
REBOOT_PLAN = [
    WorkflowOperation.FREEZE_EVIDENCE,
    WorkflowOperation.MARK_UNSCHEDULABLE,
    REBOOT,
    WorkflowOperation.VALIDATE_GPU,
    WorkflowOperation.VALIDATE_HOST,
    WorkflowOperation.RESTORE_SCHEDULING,
]


def _merger() -> WorkflowMergeService:
    arbiter = RecoveryArbiter()
    return WorkflowMergeService(
        arbiter,
        DagBrancher(arbiter),
        preemption_enabled=True,
        workload_scoped_operations=set(WORKLOAD_SCOPED_OPERATIONS),
        node_exclusive_operations=set(NODE_EXCLUSIVE_OPERATIONS),
        workflow_resource_claims_by_node=(
            NodeConflictService.workflow_resource_claims_by_node
        ),
    )


def _quiesced_reset(*, through: WorkflowOperation = QUIESCE) -> WorkflowRequest:
    """A running reset whose steps up to and including ``through`` are done."""

    last = RESET_PLAN.index(through)
    return workflow_request(
        "wf-reset",
        "inc-a",
        WorkflowStatus.RUNNING,
        1,
        runtime_profile_version="simulated-v1",
        official_action=RESET.value,
        official_steps=[
            workflow_step(operation, "gpu-fault-node-agent", node_ids=["node-a"])
            for operation in RESET_PLAN
        ],
        completed_step_indexes=list(range(last + 1)),
        completed_operations=RESET_PLAN[: last + 1],
    )


def _successor(
    plan: list[WorkflowOperation], request_id: str = "wf-reboot"
) -> WorkflowRequest:
    return workflow_request(
        request_id,
        "inc-a",
        fencing_token=1,
        runtime_profile_version="simulated-v1",
        official_action=plan[2].value,
        official_steps=[
            workflow_step(operation, "gpu-fault-node-agent", node_ids=["node-a"])
            for operation in plan
        ],
    )


def _appended(combined: WorkflowRequest, existing: WorkflowRequest):
    return combined.official_steps[len(existing.official_steps) :]


def test_a_reboot_preempting_a_quiesced_reset_takes_over_restoring_gpu_services():
    existing = _quiesced_reset()

    combined = _merger().preempt_parallel_branch(
        existing, _successor(REBOOT_PLAN), "node-a"
    )

    appended = _appended(combined, existing)
    operations = [step.operation for step in appended]
    handoffs = [
        index
        for index, step in enumerate(appended)
        if step.parameters.get(HANDOFF) is True
    ]
    assert len(handoffs) == 1, (
        f"exactly one handoff restore belongs in the successor, got {operations}"
    )
    handoff = appended[handoffs[0]]
    assert handoff.operation is RESTORE_SERVICES, (
        "the handoff undoes the quiesce, so it is a RESTORE_GPU_SERVICES step"
    )
    assert handoffs[0] > operations.index(REBOOT), (
        "restoring services before the reboot would be undone by the reboot"
    )
    assert handoff.node_ids == ["node-a"], (
        "the handoff acts on the node the reboot acts on"
    )
    assert handoff.execution_owner == "gpu-fault-node-agent", (
        "the step needs the same executable owner the retired restore had"
    )


def test_a_reboot_successor_does_not_re_quiesce_a_node_that_is_already_quiesced():
    existing = _quiesced_reset()

    combined = _merger().preempt_parallel_branch(
        existing, _successor(REBOOT_PLAN), "node-a"
    )

    assert QUIESCE not in [step.operation for step in _appended(combined, existing)], (
        "the predecessor's completed quiesce still holds; quiescing twice would "
        "have the successor verify no GPU clients on already-stopped services"
    )


def test_a_reboot_successor_gets_no_handoff_when_the_reset_already_restored():
    """The predecessor got as far as its own restore, so the node is not left
    quiesced and the successor has nothing to take over."""

    existing = _quiesced_reset(through=RESTORE_SERVICES)

    combined = _merger().preempt_parallel_branch(
        existing, _successor(REBOOT_PLAN), "node-a"
    )

    assert not any(
        step.parameters.get(HANDOFF) for step in _appended(combined, existing)
    ), "a restore that already ran must not be re-planned into the successor"


def test_a_successor_that_plans_its_own_restore_gets_no_second_one():
    existing = _quiesced_reset()
    plan = [*REBOOT_PLAN[:3], RESTORE_SERVICES, *REBOOT_PLAN[3:]]

    combined = _merger().preempt_parallel_branch(existing, _successor(plan), "node-a")

    appended = _appended(combined, existing)
    restores = [step for step in appended if step.operation is RESTORE_SERVICES]
    assert len(restores) == 1, (
        "the successor already restores services; a handoff would run it twice"
    )
    assert restores[0].parameters.get(HANDOFF) is None, (
        "the successor's own restore is not a handoff and must not be relabelled"
    )


def test_a_replacement_successor_gets_no_handoff_for_the_node_it_discards():
    """A replacement does not bring this node back, so there is no point
    restoring services on it -- and doing so would run a node-agent step
    against a node the plan is about to take out of the cluster."""

    existing = _quiesced_reset()
    plan = [*REBOOT_PLAN[:2], REPLACE, *REBOOT_PLAN[3:]]

    combined = _merger().preempt_parallel_branch(existing, _successor(plan), "node-a")

    assert not any(
        step.parameters.get(HANDOFF) for step in _appended(combined, existing)
    ), "the handoff is the reboot's obligation, not the replacement's"

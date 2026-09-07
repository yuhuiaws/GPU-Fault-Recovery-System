"""What a same-node escalation does to the workflow *row*, seen from the API.

Two rungs of the merge write fields nothing in ``tests/orchestration``
looked at:

* ``DispositionApplier._replace_in_place`` swaps a mutable predecessor's plan
  while keeping its ``request_id``, ``created_at`` and ``lifetime_deadline_at``.
  Keeping the id is what makes the store treat the write as an in-place merge
  and bump ``merge_revision`` -- the optimistic fence an executor holding the
  old plan is checked against (F-B5, F-N1).
* ``WorkflowMergeService.prepare_preempting_successor`` marks a stronger
  successor, and the merge transaction stamps
  ``preemption_pending_by_workflow_id`` on the predecessor so the dispatcher
  holds it immediately instead of starting work a successor is about to
  supersede (F-C1).

Both are exercised here through the ingest API, so the orchestration decision
and the store write that carries it are pinned together.
"""

from __future__ import annotations

import asyncio
from datetime import timedelta

from gpu_fault.models import WorkflowOperation, WorkflowRequest, WorkflowStatus
from gpu_fault.orchestration import IncidentOrchestrator
from tests._builders import build_context, copy_model, workflow_step_execution
from tests.orchestration._cross_fault_support import (
    observation,
    post_faults,
    sxid_payload,
    xid_payload,
)

# XID 48 plans a GPU reset (rank 30); XID 79 a node reboot, and the fabric
# SXID a full fabric reset (rank 40). Both are strictly stronger on the same
# node, which is what puts the merge on the escalation rungs below.
RESET_XID = 48
REBOOT_XID = 79
CONTAINMENT = {WorkflowOperation.MARK_UNSCHEDULABLE, WorkflowOperation.STOP_WORKLOADS}


def _context():
    """A job-aware context: the observation is what makes the first fault
    plan a dispatchable PENDING workflow rather than a safety-only record."""

    context = build_context()
    context.orchestrator = IncidentOrchestrator(
        context.store, workflow_preemption_enabled=True
    )
    context.store.save_attempt_observation(observation())
    return context


def _post_xid(context, xid: int, event_id: str) -> WorkflowRequest:
    result = asyncio.run(
        post_faults(context, [("/v1/gpu-events/xid", xid_payload(xid, event_id))])
    )[0]
    return context.store.get_workflow(result["workflow_request_id"])


def _run_containment(context, workflow: WorkflowRequest) -> WorkflowRequest:
    """Hand the record to an executor that finished cordon and workload stop.

    That makes it immutable, so the next stronger fault has to become a
    preempting successor instead of replacing the plan in place.
    """

    indexes = [
        index
        for index, step in enumerate(workflow.official_steps)
        if step.operation in CONTAINMENT
    ]
    assert len(indexes) == len(CONTAINMENT), (
        f"expected a cordon and a workload stop to inherit, found {indexes}"
    )
    context.store.save_workflow(
        copy_model(
            workflow,
            status=WorkflowStatus.RUNNING,
            execution_owner_id="executor-a",
            completed_step_indexes=indexes,
            completed_operations=[
                workflow.official_steps[index].operation for index in indexes
            ],
            step_executions=[
                workflow_step_execution(index, workflow.official_steps[index].operation)
                for index in indexes
            ],
        )
    )
    return context.store.get_workflow(workflow.request_id)


def test_a_stronger_fault_on_an_unclaimed_record_replaces_its_plan_in_place():
    context = _context()
    reset = _post_xid(context, RESET_XID, "in-place-reset")

    reboot = _post_xid(context, REBOOT_XID, "in-place-reboot")

    assert reboot.request_id == reset.request_id, (
        "an unclaimed record is re-planned, not shadowed by a second row"
    )
    assert len(context.store.list_workflows()) == 1, (
        "a duplicate row would leave two records claiming the same node"
    )
    assert reboot.created_at == reset.created_at, (
        "the record's age is its own; only the plan changed"
    )
    operations = {step.operation for step in reboot.official_steps}
    assert WorkflowOperation.RESTART_NODE in operations, (
        "the stronger action is what the record now plans"
    )
    assert WorkflowOperation.RESET_GPU not in operations, (
        "the weaker action it dominates is gone from the plan"
    )


def test_an_in_place_plan_replacement_bumps_the_merge_revision():
    """``merge_revision`` is the fence a leased writer is checked against, so
    replacing the plan under an executor has to move it."""

    context = _context()
    reset = _post_xid(context, RESET_XID, "revision-reset")
    assert reset.merge_revision == 0, "a freshly planned record starts at zero"

    reboot = _post_xid(context, REBOOT_XID, "revision-reboot")

    assert reboot.merge_revision == 1, (
        "an executor still holding revision 0 must now lose its write"
    )


def test_an_in_place_plan_replacement_keeps_the_records_lifetime_deadline():
    """F-N1 caps how long one incident may hold a node. If escalating reset ->
    reboot restarted the clock, a node that keeps failing would never reach the
    operator hand-off the cap exists to force."""

    context = _context()
    reset = _post_xid(context, RESET_XID, "lifetime-reset")
    deadline = reset.created_at + timedelta(hours=6)
    context.store.save_workflow(copy_model(reset, lifetime_deadline_at=deadline))

    reboot = _post_xid(context, REBOOT_XID, "lifetime-reboot")

    assert reboot.lifetime_deadline_at == deadline, (
        "the successor plan inherits the deadline it was already under"
    )


def test_a_preempting_successor_holds_its_predecessor_on_the_way_in():
    context = _context()
    reset = _run_containment(context, _post_xid(context, RESET_XID, "hold-reset"))

    fabric = asyncio.run(
        post_faults(
            context, [("/v1/gpu-events/sxid", sxid_payload("hold-fabric-sxid"))]
        )
    )[0]

    successor = context.store.get_workflow(fabric["workflow_request_id"])
    predecessor = context.store.get_workflow(reset.request_id)
    assert successor.request_id != reset.request_id, (
        "a claimed record is not re-planned under its executor"
    )
    assert successor.preempt_predecessor is True, (
        "the successor is a strictly stronger node action, not a queued one"
    )
    assert predecessor.preemption_pending_by_workflow_id == successor.request_id, (
        "the predecessor is held in the same merge; otherwise the dispatcher "
        "starts a reset the fabric reset is about to supersede"
    )
    assert predecessor.status is WorkflowStatus.RUNNING, (
        "the hold is a marker, not a status change the executor did not make"
    )
    assert successor.preemption_pending_by_workflow_id is None, (
        "nothing is preempting the successor, so its own marker stays clear"
    )

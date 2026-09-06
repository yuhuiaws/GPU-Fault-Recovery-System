"""Which record-only counter a withdrawn workflow's absorbed event lands on
(F-N1 §7).

``WorkflowMergeService.disposition`` has three rungs that all answer
``ABSORB_RECORD_ONLY`` -- lifetime exhausted, workload withdrawn, and a live
node-mutating action already covering the scope -- and each keeps its own
total, published as a separate ``reason`` label by
``app/builtin_metric_contributors.py``. Only the disposition value was pinned
for the withdrawn rung, so an event absorbed because the job was stopped by
someone else could be counted as an operator-held lifetime timeout or as
covered evidence without any test noticing, which is exactly the attribution
an operator reads the metric for. These tests pin the rung's own total and the
fact that its siblings stay untouched.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

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

NOW = datetime.now(timezone.utc)
PAST = NOW - timedelta(minutes=5)


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


def _withdrawn(
    operation: WorkflowOperation = WorkflowOperation.RESET_GPU,
) -> WorkflowRequest:
    """A job repair whose workload someone else stopped while it ran."""

    return workflow_request(
        "wf-job",
        "inc-job",
        status=WorkflowStatus.RUNNING,
        official_steps=[
            workflow_step(operation, node_ids=["node-a"], gpu_uuids=["GPU-a"])
        ],
        workload_withdrawn_at=NOW - timedelta(seconds=1),
        aggregation_max_deadline=PAST,
    )


def _candidate(
    operation: WorkflowOperation = WorkflowOperation.RESET_GPU,
) -> WorkflowRequest:
    return workflow_request(
        "wf-new",
        "inc-new",
        official_steps=[
            workflow_step(operation, node_ids=["node-a"], gpu_uuids=["GPU-a"])
        ],
    )


def _totals(merger: WorkflowMergeService) -> tuple[int, int, int]:
    return (
        merger.withdrawn_record_only_total,
        merger.lifetime_record_only_total,
        merger.absorbed_record_only_total,
    )


def test_a_withdrawn_workflow_absorbs_the_event_under_only_the_withdrawn_counter():
    merger = _merger()

    disposition = merger.disposition(_withdrawn(), _candidate(), "node-a", {"GPU-a"})

    assert disposition == "ABSORB_RECORD_ONLY"
    assert _totals(merger) == (1, 0, 0), (
        "the withdrawn rung owns its own metric label; crediting the lifetime "
        "or covered-evidence totals would tell the operator the wrong reason"
    )


def test_a_withdrawn_workflow_still_inside_its_lifetime_is_not_counted_as_expired():
    """The lifetime rung sits above the withdrawn one, so a withdrawn workflow
    whose deadline has *not* passed must not be attributed to it."""

    merger = _merger()
    existing = _withdrawn().model_copy(
        update={"lifetime_deadline_at": NOW + timedelta(hours=1)}
    )

    disposition = merger.disposition(existing, _candidate(), "node-a", {"GPU-a"})

    assert disposition == "ABSORB_RECORD_ONLY"
    assert _totals(merger) == (1, 0, 0), (
        "a live deadline means the workflow is winding down from withdrawal, "
        "not being held by an operator after a timeout"
    )


def test_a_withdrawn_reboot_that_covers_the_fault_is_counted_as_withdrawn():
    """The withdrawn rung sits above the covered-evidence one. A withdrawn
    workflow whose running reboot also covers the candidate's scope satisfies
    both, and the reason recorded has to be the withdrawal: nothing is going to
    finish rebooting this node on the job's behalf."""

    merger = _merger()
    read_only = workflow_request(
        "wf-evidence",
        "inc-evidence",
        official_steps=[
            workflow_step(WorkflowOperation.FREEZE_EVIDENCE, node_ids=["node-a"])
        ],
    )

    disposition = merger.disposition(
        _withdrawn(WorkflowOperation.RESTART_NODE), read_only, "node-a", set()
    )

    assert disposition == "ABSORB_RECORD_ONLY"
    assert _totals(merger) == (1, 0, 0), (
        "reordering the rungs would relabel this event as covered evidence"
    )


def test_each_absorbed_event_for_a_withdrawn_workflow_advances_the_total():
    """The counter is a monotonic total the metric contributor publishes, not a
    flag for "this workflow was withdrawn"."""

    merger = _merger()
    existing = _withdrawn()

    for index in range(3):
        merger.disposition(
            existing,
            _candidate().model_copy(update={"request_id": f"wf-new-{index}"}),
            "node-a",
            {"GPU-a"},
        )

    assert _totals(merger) == (3, 0, 0), (
        "three events were recorded on the incident, so the total reads three"
    )

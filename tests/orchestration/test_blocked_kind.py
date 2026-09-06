"""BLOCKED is neither terminal nor executable, and it has three causes.

FINAL-建议汇总 F-B4 (P0-69A, P0-70B, P0-56A, P0-59A, P0-72A/B/C). A workflow
is BLOCKED because its safety steps settled (the plan finished the way it was
meant to), because compilation left an operator something to decide, or
because the dispatcher hit an internal error. The record carried none of that
after the first merge; and ``disposition()`` never looked at ``existing.status``
so a new fault could be absorbed or widened into a record nobody dispatches.
"""

from __future__ import annotations

import pytest

from gpu_fault.models import (
    EXECUTABLE_WORKFLOW_STATUSES,
    BlockedKind,
    WorkflowOperation,
    WorkflowRequest,
    WorkflowStatus,
)
from gpu_fault.orchestration.arbitration import RecoveryArbiter
from gpu_fault.orchestration.workflow_merge import WorkflowMergeService
from tests._builders import workflow_request, workflow_step


class _NoBranching:
    def dag_join_accepts_new_branch(self, _workflow) -> bool:
        return False

    def node_branch_step_indexes(self, _workflow, _node_id) -> list[int]:
        return []


def _merger() -> WorkflowMergeService:
    return WorkflowMergeService(
        RecoveryArbiter(),
        _NoBranching(),
        preemption_enabled=True,
        workload_scoped_operations=set(),
        node_exclusive_operations=set(),
        workflow_resource_claims_by_node=lambda _workflow: {},
    )


def _restart_node(request_id: str, status: WorkflowStatus) -> WorkflowRequest:
    return workflow_request(
        request_id,
        "inc-a",
        status=status,
        official_steps=[
            workflow_step(WorkflowOperation.RESTART_NODE, node_ids=["node-a"])
        ],
    )


def test_blocked_kind_is_optional_and_defaults_to_none():
    assert set(BlockedKind) == {
        BlockedKind.SAFETY_SETTLED,
        BlockedKind.NEEDS_OPERATOR,
        BlockedKind.INTERNAL_ERROR,
    }
    assert _restart_node("wf", WorkflowStatus.PENDING).blocked_kind is None


def test_executable_statuses_are_the_three_the_dispatcher_scans():
    assert EXECUTABLE_WORKFLOW_STATUSES == frozenset(
        {WorkflowStatus.PENDING, WorkflowStatus.SAFETY_PENDING, WorkflowStatus.RUNNING}
    )


def test_a_pending_incumbent_that_covers_the_fault_still_absorbs():
    disposition = _merger().disposition(
        _restart_node("wf-existing", WorkflowStatus.PENDING),
        _restart_node("wf-candidate", WorkflowStatus.PENDING),
        "node-a",
        set(),
    )

    assert disposition == "ABSORB"


@pytest.mark.parametrize(
    "status",
    [
        WorkflowStatus.BLOCKED,
        WorkflowStatus.SUCCEEDED,
        WorkflowStatus.FAILED,
        WorkflowStatus.SUPERSEDED,
    ],
)
def test_nothing_merges_into_a_record_that_will_never_execute(status):
    """The same covering incumbent, no longer executable: the candidate must
    get its own record rather than vanish into this one."""

    disposition = _merger().disposition(
        _restart_node("wf-existing", status),
        _restart_node("wf-candidate", WorkflowStatus.PENDING),
        "node-a",
        set(),
    )

    assert disposition == "QUEUE_SUCCESSOR"


# ---------------------------------------------------------------- F-A4


def test_node_exclusivity_sees_an_operator_blocked_workflow():
    """P0-42E: BLOCKED was invisible to every node-exclusive check, so a
    workflow that had already rebooted the node counted as the node being
    free. The settled kind is genuinely done and stays invisible."""

    from gpu_fault.models import IncidentState
    from gpu_fault.orchestration.families.conflicts import NodeConflictService
    from tests._builders import build_store, fault_incident

    store = build_store()
    for name, kind in (
        ("operator", BlockedKind.NEEDS_OPERATOR),
        ("settled", BlockedKind.SAFETY_SETTLED),
    ):
        workflow = workflow_request(
            f"wf-{name}",
            f"inc-{name}",
            status=WorkflowStatus.BLOCKED,
            blocked_kind=kind,
            official_steps=[
                workflow_step(WorkflowOperation.RESTART_NODE, node_ids=[f"node-{name}"])
            ],
        )
        store.save_incident_and_workflow(
            fault_incident(
                f"inc-{name}",
                f"event-{name}",
                state=IncidentState.QUARANTINED,
                workflow_request_id=workflow.request_id,
                node_ids=[f"node-{name}"],
            ),
            workflow,
        )
    conflicts = NodeConflictService(store, RecoveryArbiter())

    occupied = conflicts.active_node_exclusive_workflow("cluster-a", {"node-operator"})
    released = conflicts.active_node_exclusive_workflow("cluster-a", {"node-settled"})

    assert occupied is not None and occupied.request_id == "wf-operator"
    assert released is None


# ---------------------------------------------------------------- F-C8 (2)


def test_a_compiled_safety_pending_workflow_carries_the_safety_only_flag(context):
    """The compiler is the one place that knows the record was built to run
    its safety steps; it says so explicitly instead of leaving the executor
    to infer it from ``blocked_reasons``."""

    from gpu_fault.models import CapabilityName
    from tests._builders import copy_model
    from tests.orchestration._support import event, ingest

    default = context.store.get_profile("simulated-v1")
    limited = copy_model(
        default,
        profile_version="limited-v1",
        capabilities=[
            item
            for item in default.capabilities
            if item.capability is not CapabilityName.MECHANICAL_INSPECTION
        ],
    )
    context.store.save_profile(limited)
    xid_event = copy_model(
        event(54, event_id="mechanical-no-adapter-flag"),
        runtime_profile_version="limited-v1",
    )

    _, _, workflow = ingest(context, xid_event)

    assert workflow.status is WorkflowStatus.SAFETY_PENDING
    assert workflow.safety_only is True
    assert workflow.executes_safety_steps is True

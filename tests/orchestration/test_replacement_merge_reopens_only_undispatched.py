"""The one BLOCKED-to-PENDING writer only reaches rows the dispatcher never had.

``node_lifecycle._state`` merges a second node fault into an open replacement
workflow by building a FRESH ``WorkflowRequest`` under the same ``request_id``
with status PENDING and no ``step_executions``. It does so for a workflow in
{PENDING, BLOCKED} -- so a BLOCKED row CAN go back to PENDING through this path
-- but only while ``not_before`` is still in the future and the row has no
execution owner: the aggregation window is open and nothing has claimed it.
Once the dispatcher has claimed a row ``not_before`` is in the past for good,
so a BLOCKED row that carries execution records (the ``INTERNAL_ERROR`` shape
that keeps a WAITING install) is out of reach.

The release engine's in-flight install gate
(``regional_release_store_preflight``) leaves BLOCKED out of its scan on exactly
that ground: no DISPATCHED BLOCKED row is ever reopened, so no control plane --
old or new -- derives a command id for a step in one again. This file pins the
ground; ``tests/regional/test_release_inflight_install_gate.py`` pins the gate.
"""

from __future__ import annotations

from datetime import timedelta

from gpu_fault.models import (
    BlockedKind,
    WorkflowStatus,
    WorkflowStepExecution,
    WorkflowStepStatus,
)
from tests._builders import (
    attempt_observation,
    container_observation,
    copy_model,
    node_health_finding,
)
from tests.orchestration._support import (
    NOW,
    ApplicationContext,
    NodeHealthCategory,
    RecoveryAction,
    WorkloadState,
)


def _finding(finding_id: str, node_id: str):
    return node_health_finding(
        finding_id,
        f"event-{finding_id}",
        node_id=node_id,
        observed_at=NOW,
        category=NodeHealthCategory.GPU,
        severity="critical",
        reason="synthetic merge-ground test",
        recommended_action=RecoveryAction.REPLACE_NODE,
        runtime_profile_version="simulated-v1",
        workload_state=WorkloadState.ACTIVE,
        affected_workload_ids=["training/pytorchjob/job-a"],
        diagnostic_parameters={"replacement_strategy": "HEALTHY_WARM_SPARE_ONLY"},
    )


def _open_replacement(context: ApplicationContext):
    context.store.save_attempt_observation(
        attempt_observation(
            "job-a",
            "job-a-a001",
            NOW,
            containers=[
                container_observation(
                    "pod-a", "worker-a", 0, "node-a", gpu_uuids=["GPU-a"]
                ),
                container_observation(
                    "pod-b", "worker-b", 1, "node-b", gpu_uuids=["GPU-b"]
                ),
            ],
            workload_ids=["training/pytorchjob/job-a"],
        )
    )
    return context.orchestrator.ingest_node_health(_finding("first", "node-a"))


def test_an_undispatched_blocked_row_is_rewritten_to_pending_by_the_merge(
    context: ApplicationContext,
) -> None:
    """The rewrite exists -- pinned so the gate's ground is stated exactly: it
    reaches a BLOCKED row whose window is still open and that nothing owns."""

    _incident, workflow = _open_replacement(context)
    assert workflow.status is WorkflowStatus.PENDING
    assert workflow.not_before is not None and workflow.not_before > NOW, (
        "a fresh replacement sits in its aggregation window"
    )
    blocked = copy_model(
        workflow,
        status=WorkflowStatus.BLOCKED,
        blocked_kind=BlockedKind.NEEDS_OPERATOR,
        blocked_reasons=["operator hold"],
    )
    context.store.save_workflow(blocked, expected=workflow)

    _merged_incident, merged = context.orchestrator.ingest_node_health(
        _finding("second", "node-b")
    )

    assert merged.request_id == workflow.request_id, "merged into the same row"
    assert merged.status is WorkflowStatus.PENDING, "BLOCKED went back to PENDING"
    assert merged.step_executions == [], "a fresh request: no execution history"


def test_a_dispatched_blocked_row_is_never_reopened(
    context: ApplicationContext,
) -> None:
    """A BLOCKED row the dispatcher had -- ``not_before`` in the past, an
    execution record on a step -- is the ``INTERNAL_ERROR`` shape that can still
    carry a WAITING install. The merge does not touch it: the second fault opens
    its own workflow and the BLOCKED row keeps its status and its executions."""

    _incident, workflow = _open_replacement(context)
    dispatched_then_blocked = copy_model(
        workflow,
        status=WorkflowStatus.BLOCKED,
        blocked_kind=BlockedKind.INTERNAL_ERROR,
        blocked_reasons=["dispatcher internal error: ValidationError"],
        not_before=NOW - timedelta(hours=1),
        step_executions=[
            WorkflowStepExecution(
                step_index=0,
                operation=workflow.official_steps[0].operation,
                status=WorkflowStepStatus.WAITING,
            )
        ],
    )
    context.store.save_workflow(dispatched_then_blocked, expected=workflow)

    _merged_incident, merged = context.orchestrator.ingest_node_health(
        _finding("second", "node-b")
    )

    assert merged.request_id != workflow.request_id, (
        "a dispatched BLOCKED row is not merged into; the fault gets its own row"
    )
    untouched = context.store.get_workflow(workflow.request_id)
    assert untouched.status is WorkflowStatus.BLOCKED
    assert untouched.blocked_kind is BlockedKind.INTERNAL_ERROR
    assert [
        (execution.step_index, execution.status)
        for execution in untouched.step_executions
    ] == [(0, WorkflowStepStatus.WAITING)], "the execution history is intact"

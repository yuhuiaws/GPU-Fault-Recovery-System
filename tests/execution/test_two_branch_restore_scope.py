"""A restore answers for its own node's actions, never for a sibling branch's.

Live (gpu-eks-cluster, DESTR-015): two node branches reset their GPUs in
parallel; both resets were refused busy by the driver once and succeeded on
the retry. Branch A's RESET_GPU returned first and its RESTORE_GPU_SERVICES
was dispatched two seconds before branch B's reset returned. The refusal
judged the whole record, read B's in-flight reset as an unresolved node
action, and FAILED A's restore ``NODE_ACTION_OUTCOME_UNRESOLVED``; the branch
escalator took that for A's hardware verdict ("outcome unknown ... no rung"),
exhausted the branch, and the hand-off quarantined a node whose own reset had
succeeded. The refusal now reads the restore's own nodes only; a node action
that may still be executing there holds the restore WAITING (bounded by the
step cap) instead of refusing it; and a held or withheld restore is never a
branch's verdict.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from gpu_fault.app import default_simulated_profile
from gpu_fault.execution import WorkflowStepContext, WorkflowStepOutcome
from gpu_fault.execution.branch_escalation import BranchEscalator
from gpu_fault.execution.node_action_uncertainty import (
    RESTORE_HELD_KEY,
    UNRESOLVED_NODE_ACTION,
    restoration_refusal,
    unresolved_actions_for,
)
from gpu_fault.models import (
    IncidentState,
    WorkflowEventCode,
    WorkflowOperation,
    WorkflowStatus,
    WorkflowStepStatus,
)
from gpu_fault.orchestration.arbitration import RecoveryArbiter
from gpu_fault.orchestration.dag_branching import DagBrancher
from tests._builders import (
    active_workflow_executor,
    workflow_request,
    workflow_step,
    workflow_step_execution,
)
from tests.execution._cov95_runtime_workflows import FlowHarness

OP = WorkflowOperation
QUIESCE, RESET, RESTORE = OP.QUIESCE_GPU_SERVICES, OP.RESET_GPU, OP.RESTORE_GPU_SERVICES
NODES = ["node-a", "node-b"]


def _busy_then_reset(node: str) -> dict[str, Any]:
    """The Agent's record of a reset refused busy once and applied on retry."""

    return {
        "node_results": {
            node: {
                "node_action_command_id": f"owned/reset/{node}",
                "physical_calls": [
                    {
                        "attempt": 1,
                        "command": f"nvidia-smi --gpu-reset -i GPU-{node}",
                        "outcome": "busy",
                    },
                    {"command": "rmmod nvidia_drm", "outcome": "released"},
                    {
                        "attempt": 2,
                        "command": f"nvidia-smi --gpu-reset -i GPU-{node}",
                        "outcome": "reset",
                    },
                ],
                "reset_completed": [f"GPU-{node}"],
            }
        }
    }


def _in_flight(node: str) -> dict[str, Any]:
    """The reset the executor is still polling: PENDING on the agent."""

    return {
        "node_action_state": "PENDING",
        "node_action_command_id": f"owned/reset/{node}",
        "remote_command_id": f"remote-{node}",
        "remote_status": "LEASED",
    }


def _two_branch_steps(h: FlowHarness) -> list[Any]:
    base = h.workflow.official_steps
    steps = []
    for offset, node in enumerate(NODES):
        first = 3 * offset
        branch = f"branch:{node}"
        common = {
            "node_ids": [node],
            "gpu_uuids": [f"GPU-{node}"],
            "branch_id": branch,
            "branch_node_ids": [node],
        }
        steps.extend(
            [
                base[0].model_copy(update=common),
                base[1].model_copy(
                    update={**common, "depends_on_step_indexes": [first]}
                ),
                base[2].model_copy(
                    update={**common, "depends_on_step_indexes": [first + 1]}
                ),
            ]
        )
    return steps


def _two_branch_flow() -> tuple[FlowHarness, dict[str, int]]:
    h = FlowHarness([QUIESCE, RESET, RESTORE])
    expires = (datetime.now(timezone.utc) + timedelta(minutes=30)).isoformat()
    reset_attempts: dict[str, int] = {}

    def scripted(context: WorkflowStepContext) -> None:
        (node,) = context.step.node_ids
        operation = context.step.operation
        if operation is QUIESCE:
            h.adapter.outcomes[QUIESCE] = WorkflowStepOutcome.succeeded(
                operation_id=context.idempotency_key,
                details={
                    "agent_generations": {node: 1},
                    "maintenance_window_expires_at": expires,
                },
            )
        elif operation is RESET:
            reset_attempts[node] = reset_attempts.get(node, 0) + 1
            if node == "node-b" and reset_attempts[node] == 1:
                # Branch B's reset is in its busy-retry loop on the agent when
                # branch A's restore is dispatched.
                h.adapter.outcomes[RESET] = WorkflowStepOutcome.waiting(
                    operation_id=context.idempotency_key, details=_in_flight(node)
                )
            else:
                h.adapter.outcomes[RESET] = WorkflowStepOutcome.succeeded(
                    operation_id=context.idempotency_key, details=_busy_then_reset(node)
                )
        else:
            h.adapter.outcomes[RESTORE] = WorkflowStepOutcome.succeeded(
                operation_id=context.idempotency_key,
                details={"restored_services": {node: ["nvidia-persistenced"]}},
            )

    h.adapter.before = scripted
    h.store.save_profile(default_simulated_profile())
    h.amend(
        dag_enabled=True,
        dag_revision=1,
        official_steps=_two_branch_steps(h),
        runtime_profile_version="simulated-v1",
    )
    incident = h.store.get_incident(h.incident.incident_id)
    h.store.save_incident(
        incident.model_copy(
            update={"node_ids": NODES, "gpu_uuids": [f"GPU-{n}" for n in NODES]}
        ),
        expected=incident,
    )
    h.incident = h.store.get_incident(h.incident.incident_id)
    h.executor = active_workflow_executor(
        h.store, [h.adapter], [QUIESCE, RESET, RESTORE]
    )

    def forbidden_rung(*_args: Any) -> Any:
        raise AssertionError("a succeeded reset must not compile a hardware rung")

    h.executor.branch_escalator = BranchEscalator(
        DagBrancher(RecoveryArbiter()), forbidden_rung
    )
    return h, reset_attempts


def _calls(h: FlowHarness) -> list[tuple[str, str]]:
    return [
        (context.step.operation.value, context.step.node_ids[0])
        for context in h.adapter.calls
    ]


def test_branch_a_restores_while_branch_b_reset_is_still_in_flight() -> None:
    h, reset_attempts = _two_branch_flow()

    first = h.execute()

    saved = h.store.get_workflow(h.workflow.request_id)
    assert first.status is WorkflowStatus.RUNNING, first
    assert ("RESTORE_GPU_SERVICES", "node-a") in _calls(h), (
        "branch A's restore reached its agent with branch B's reset in flight"
    )
    restore_a = next(
        item
        for item in saved.step_executions
        if item.step_index == 2 and item.operation is RESTORE
    )
    assert restore_a.status is WorkflowStepStatus.SUCCEEDED, restore_a
    reset_b = next(
        item
        for item in saved.step_executions
        if item.step_index == 4 and item.operation is RESET
    )
    assert reset_b.status is WorkflowStepStatus.WAITING, reset_b
    assert saved.exhausted_branch_ids == []
    assert not any(
        item.code is WorkflowEventCode.BRANCH_EXHAUSTED for item in saved.events
    ), [item.reason for item in saved.events]
    assert not any(
        item.code is WorkflowEventCode.STEP_FAILED for item in saved.events
    ), [(item.step_index, item.reason) for item in saved.events]

    second = h.execute()

    saved = h.store.get_workflow(h.workflow.request_id)
    assert second.status is WorkflowStatus.SUCCEEDED, second
    assert reset_attempts == {"node-a": 1, "node-b": 2}
    assert (
        sorted(
            node for operation, node in _calls(h) if operation == "RESTORE_GPU_SERVICES"
        )
        == NODES
    ), "both nodes were restored, each once"
    assert saved.exhausted_branch_ids == []
    assert saved.blocked_kind is None
    incident = h.store.get_incident(h.incident.incident_id)
    assert incident.state is not IncidentState.QUARANTINED
    assert incident.state is not IncidentState.ESCALATED


def _chain(
    reset_status: WorkflowStepStatus,
    reset_details: dict[str, Any],
    *,
    restore_node: str = "node-a",
    reset_node: str = "node-a",
    pending_failure: int | None = None,
) -> tuple[Any, Any]:
    steps = [
        workflow_step(QUIESCE, node_ids=[reset_node]),
        workflow_step(RESET, node_ids=[reset_node]),
        workflow_step(RESTORE, node_ids=[restore_node]),
    ]
    workflow = workflow_request(
        "workflow-a",
        "incident-a",
        WorkflowStatus.RUNNING,
        official_steps=steps,
        completed_step_indexes=[0],
        pending_failure_step_index=pending_failure,
        step_executions=[
            workflow_step_execution(0, QUIESCE, phase="official"),
            workflow_step_execution(
                1, RESET, reset_status, phase="official", details=reset_details
            ),
        ],
    )
    return workflow, steps[2]


def test_a_sibling_nodes_in_flight_reset_never_holds_the_restore() -> None:
    workflow, restore = _chain(
        WorkflowStepStatus.WAITING, _in_flight("node-b"), reset_node="node-b"
    )
    assert unresolved_actions_for(workflow, ["node-a"]) == []
    assert restoration_refusal(workflow, restore) is None


def test_the_own_nodes_in_flight_reset_holds_the_restore_bounded() -> None:
    workflow, restore = _chain(WorkflowStepStatus.WAITING, _in_flight("node-a"))
    held = restoration_refusal(workflow, restore)
    assert held is not None and held.status is WorkflowStepStatus.WAITING
    assert held.details == {
        "reason": UNRESOLVED_NODE_ACTION,
        RESTORE_HELD_KEY: True,
        "awaiting_step_indexes": [1],
        "awaiting_remote_command_ids": ["remote-node-a"],
    }


@pytest.mark.parametrize(
    "gave_up",
    [
        {"step_waiting_timeout_seconds": 600},
        {"workflow_lifetime_exceeded": True},
        {"workflow_deadline_remote_command_cancellation": {"cancelled": 1}},
        {"remote_status_source": "workflow-timeout"},
    ],
)
@pytest.mark.parametrize("pending_failure", [None, 1])
def test_a_command_the_control_plane_gave_up_on_is_not_waited_for(
    gave_up: dict[str, Any], pending_failure: int | None
) -> None:
    """A deadline, a lifetime, a step cap or a cancellation landed over the open
    command: the agent may still be running it, but this record no longer owns
    the wait. The restore is withheld and the record parks for the receipt
    (``receipt_settled_resume``), as before."""

    workflow, restore = _chain(
        WorkflowStepStatus.FAILED,
        {"outcome_unknown": True, **_in_flight("node-a"), **gave_up},
        pending_failure=pending_failure,
    )
    refused = restoration_refusal(workflow, restore)
    assert refused is not None and refused.status is WorkflowStepStatus.FAILED, refused
    assert refused.details["restore_gpu_services_withheld"] is True


@pytest.mark.parametrize("pending_failure", [None, 1])
def test_an_answered_unknown_outcome_is_withheld_or_compensated(
    pending_failure: int | None,
) -> None:
    workflow, restore = _chain(
        WorkflowStepStatus.FAILED,
        {"outcome_unknown": True, "manual_confirmation_required": True},
        pending_failure=pending_failure,
    )
    refused = restoration_refusal(workflow, restore)
    if pending_failure is not None:
        assert refused is None, "the compensation restore runs once the agent answered"
    else:
        assert refused is not None and refused.status is WorkflowStepStatus.FAILED
        assert refused.details["restore_gpu_services_withheld"] is True
        assert refused.details["reason"] == UNRESOLVED_NODE_ACTION


def test_an_unresolvable_step_counts_for_every_node() -> None:
    """Fail-closed: an execution whose step index no longer resolves is held
    against any restore, as the whole-record rule did."""

    workflow, restore = _chain(
        WorkflowStepStatus.WAITING, _in_flight("node-b"), reset_node="node-b"
    )
    stray = workflow_step_execution(
        9, RESET, WorkflowStepStatus.WAITING, phase="official", details=_in_flight("x")
    )
    workflow = workflow.model_copy(
        update={"step_executions": [*workflow.step_executions, stray]}
    )
    assert [
        item.step_index for item in unresolved_actions_for(workflow, ["node-a"])
    ] == [9]
    held = restoration_refusal(workflow, restore)
    assert held is not None and held.status is WorkflowStepStatus.WAITING


def test_a_held_or_withheld_restore_is_never_a_branch_verdict() -> None:
    h, _attempts = _two_branch_flow()
    workflow = h.store.get_workflow(h.workflow.request_id)
    escalator = BranchEscalator(DagBrancher(RecoveryArbiter()), lambda *_a: [])
    withheld = {
        "reason": UNRESOLVED_NODE_ACTION,
        "outcome_unknown": True,
        "manual_confirmation_required": True,
        "restore_gpu_services_withheld": True,
        "safety_rejection": True,
    }
    assert (
        escalator.escalate_branch(
            workflow,
            2,
            "automatic GPU service restoration requires resolved node actions",
            details=withheld,
        )
        is None
    ), "a withheld restore is the whole record's uncertainty, not branch A's rung"
    held = {"reason": UNRESOLVED_NODE_ACTION, RESTORE_HELD_KEY: True}
    assert escalator.escalate_branch(workflow, 2, "held", details=held) is None
    # The guard is specific: a reset whose outcome is unknown still exhausts.
    exhausted = escalator.escalate_branch(
        workflow, 1, "node response was unreadable", details={"outcome_unknown": True}
    )
    assert exhausted is not None and exhausted.outcome == "exhausted"


def test_a_record_landing_a_bound_parks_instead_of_waiting() -> None:
    """The lifetime landed over a WAITING verify while the compound carrier
    still covered the reset: the reset's own snapshot looks in flight (the
    carrier is LEASED) but the record is landing a control-plane bound, so the
    compensation restore is withheld and the record parks for the receipt."""

    steps = [
        workflow_step(QUIESCE, node_ids=["node-a"]),
        workflow_step(OP.VERIFY_NO_GPU_CLIENTS, node_ids=["node-a"]),
        workflow_step(RESET, node_ids=["node-a"]),
        workflow_step(RESTORE, node_ids=["node-a"]),
    ]
    workflow = workflow_request(
        "workflow-a",
        "incident-a",
        WorkflowStatus.RUNNING,
        official_steps=steps,
        completed_step_indexes=[0],
        pending_failure_step_index=1,
        step_executions=[
            workflow_step_execution(0, QUIESCE, phase="official"),
            workflow_step_execution(
                1,
                OP.VERIFY_NO_GPU_CLIENTS,
                WorkflowStepStatus.FAILED,
                phase="official",
                details={"workflow_lifetime_exceeded": True},
            ),
            workflow_step_execution(
                2,
                RESET,
                WorkflowStepStatus.WAITING,
                phase="official",
                details={"outcome_unknown": True, **_in_flight("node-a")},
            ),
        ],
    )
    refused = restoration_refusal(workflow, steps[3])
    assert refused is not None and refused.status is WorkflowStepStatus.FAILED, refused
    assert refused.details["restore_gpu_services_withheld"] is True
    # The same reset under a live record (no bound landed) is waited for.
    live = workflow.model_copy(update={"pending_failure_step_index": None})
    held = restoration_refusal(live, steps[3])
    assert held is not None and held.status is WorkflowStepStatus.WAITING, held

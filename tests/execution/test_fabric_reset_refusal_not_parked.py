"""A fabric reset refused at the Agent's boundary is a verdict for the executor.

Live (gpu-eks-cluster, COLLECT-013, SXID 10003): step 5 RESET_ALL_GPUS_NVSWITCHES
FAILED with ``OwnershipRefused: OWNERSHIP_PERMIT_EXPIRED`` -- refused at the
final ownership boundary before the retry of a reset the driver had refused
busy. The Agent of that release attested nothing for a fabric reset, so the
step carried ``outcome_unknown`` and no ``reset_verdict``; the record parked
BLOCKED / NEEDS_OPERATOR, kept the node's exclusive occupancy after the
product's own validated restore had put the node back, held the next
incident's REBOOT_NODE workflow PENDING for half an hour and blocked the deploy
preflight. The boundary now attests its own refusals and the fabric reset
reports the RESET_GPU verdict keys; driven here through the real node-action
adapter (the per-node fold the live step took) with the Agent's result as the
current Agent writes it, and as the previous one wrote it.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from gpu_fault.adapters import NodeActionWorkflowAdapter
from gpu_fault.execution.node_action_uncertainty import unresolved_node_action
from gpu_fault.models import (
    BlockedKind,
    IncidentState,
    RecoveryAction,
    WorkflowOperation,
    WorkflowStatus,
    WorkflowStepStatus,
)
from gpu_fault.node_agent import NodeActionResult, NodeActionStatus, SignedNodeAction
from gpu_fault.orchestration.arbitration import RecoveryArbiter
from gpu_fault.orchestration.escalation import (
    NODE_ACTION_REFUSED_STAGE,
    HardwareEscalationService,
)
from gpu_fault.orchestration.families.conflicts import NodeConflictService
from gpu_fault.recovery_safety import unresolved_details
from tests.execution._cov95_runtime_workflows import FlowHarness
from tests.node_agent._support import SECRET

OP = WorkflowOperation
QUIESCE = OP.QUIESCE_GPU_SERVICES
RESET = OP.RESET_ALL_GPUS_NVSWITCHES
RESTORE = OP.RESTORE_GPU_SERVICES
GPUS = [f"GPU-{index}" for index in range(8)]
TIMING = {
    "pre_permit_seconds": 19.4,
    "permit_wait_seconds": 0.2,
    "post_permit_seconds": 3.6,
    "permit_lifetime_seconds": 3.0,
}
RECEIPT = {
    "boundary": "AGENT_PRE_SPAWN",
    "challenge_nonce": "1" * 64,
    "sequence": 1,
    "checked_at": "2026-01-01T00:00:00+00:00",
    "expires_at": "2026-01-01T00:00:03+00:00",
    **TIMING,
}
#: The live step's failure as the previous Agent reported it.
PREVIOUS_AGENT: dict[str, Any] = {
    "reason": "OWNERSHIP_PERMIT_EXPIRED",
    "safety_rejection": True,
    "manual_confirmation_required": True,
    "outcome_unknown": True,
    "ownership_check_boundary": "AGENT_PRE_SPAWN",
    "agent_queue_ownership_checked": True,
    "ownership_permit_timing": TIMING,
    "gpu_reset_attempts_spawned": 1,
    "physical_ownership_checks": [RECEIPT],
}
#: The same refusal as the current Agent reports it: the boundary's own
#: attestation and the fabric reset's verdict keys.
CURRENT_AGENT: dict[str, Any] = {
    **{key: value for key, value in PREVIOUS_AGENT.items() if key != "outcome_unknown"},
    "node_action_outcome_known": True,
    "reset_verdict": "not-applied",
    "reset_scope": "ALL_LOCAL_GPUS_AND_NVSWITCHES",
    "reset_completed": [],
    "reset_outcome_unknown": [],
    "reset_failed": GPUS,
    "reset_not_attempted": [],
}


def _flow(agent_details: dict[str, Any]) -> tuple[FlowHarness, list[SignedNodeAction]]:
    h = FlowHarness([QUIESCE, RESET, RESTORE])
    sent: list[SignedNodeAction] = []

    def send(_endpoint: str, envelope: SignedNodeAction) -> NodeActionResult:
        sent.append(envelope)
        command = envelope.command
        return NodeActionResult(
            command_id=command.command_id,
            operation=command.operation,
            status=NodeActionStatus.FAILED,
            details={**agent_details, "node_action_command_id": command.command_id},
            error="OwnershipRefused: OWNERSHIP_PERMIT_EXPIRED",
            retryable=False,
            completed_at=datetime.now(UTC),
        )

    adapter = NodeActionWorkflowAdapter(
        {"node-a": "http://node-a:9099"}, SECRET, sender=send
    )
    h.executor.adapters.insert(0, adapter)
    h.amend(
        official_steps=[
            step.model_copy(
                update={
                    "execution_owner": (
                        adapter.owner
                        if step.operation is RESET
                        else step.execution_owner
                    ),
                    "gpu_uuids": GPUS,
                }
            )
            for step in h.workflow.official_steps
        ]
    )
    return h, sent


def _failed_reset(h: FlowHarness) -> Any:
    saved = h.store.get_workflow(h.workflow.request_id)
    return saved, next(
        item for item in saved.step_executions if item.operation is RESET
    )


def _occupying(h: FlowHarness) -> Any:
    return NodeConflictService(
        h.store, RecoveryArbiter()
    ).active_node_exclusive_workflow(h.incident.cluster_id, set(h.incident.node_ids))


def test_the_current_agents_refusal_fails_the_chain_without_parking() -> None:
    h, sent = _flow(CURRENT_AGENT)

    result = h.execute()

    assert result.status is WorkflowStatus.FAILED, result
    saved, failure = _failed_reset(h)
    assert saved.blocked_kind is None
    assert _occupying(h) is None, "no parked record holds the node"
    assert [context.step.operation for context in h.adapter.calls] == [
        QUIESCE,
        RESTORE,
    ], "the compensation restore runs: nothing is in flight on the node"
    assert RESTORE in saved.completed_operations
    assert failure.status is WorkflowStepStatus.FAILED
    assert len(sent) == 1 and sent[0].command.operation is RESET
    details = failure.details
    assert details["node_action_outcome_known"] is True
    assert details["reset_verdict"] == "not-applied"
    assert details["reset_failed"] == GPUS and details["reset_completed"] == []
    assert details.get("outcome_unknown") is None
    assert details["safety_rejection"] is True
    assert details["failed_nodes"] == ["node-a"]
    assert details["node_failures"] == {
        "node-a": [
            "node action refused before acting: "
            "OwnershipRefused: OWNERSHIP_PERMIT_EXPIRED"
        ]
    }
    assert details["ownership_permit_timing"] == TIMING
    assert not unresolved_details(details), "nothing for an operator to confirm"
    assert not unresolved_node_action(failure), "no remote command left unresolved"
    classified = HardwareEscalationService.classify(saved)
    assert classified is not None
    stage, action, operation, flagged = classified
    assert stage == NODE_ACTION_REFUSED_STAGE
    assert action is RecoveryAction.ESCALATE_OPERATOR
    assert operation is OP.ESCALATE_SUPPORT
    assert [item.operation for item in flagged] == [RESET]
    assert h.store.get_incident(h.incident.incident_id).state is IncidentState.ESCALATED


def test_the_previous_agents_unknown_shape_still_parks_for_the_operator() -> None:
    """The record the live step wrote keeps its operator exits
    (``confirm-node-action``, ``workflow-reconcile``); only the current Agent's
    attestation ends the parking."""

    h, _sent = _flow(PREVIOUS_AGENT)

    result = h.execute()

    assert result.status is WorkflowStatus.BLOCKED
    saved, failure = _failed_reset(h)
    assert saved.blocked_kind is BlockedKind.NEEDS_OPERATOR
    assert _occupying(h) is not None and _occupying(h).request_id == saved.request_id
    assert failure.details["outcome_unknown"] is True
    assert failure.details["node_failures"]["node-a"][0].startswith(
        "node action outcome unknown: "
    ), "the unattested shape is still an unknown outcome"
    assert [context.step.operation for context in h.adapter.calls] == [
        QUIESCE,
        RESTORE,
    ], "the compensation restore still runs; only the record parks"

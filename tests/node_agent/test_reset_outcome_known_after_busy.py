"""A reset refused after a busy retry is a known verdict, not an unknown outcome.

Live (gpu-eks-cluster, XID 46, release 5377615bd785): attempt 1 of the reset
was refused "In use by another client", the display stack was unloaded, and
attempt 2 was refused at the final ownership boundary before its nvidia-smi
was spawned. The boundary's wrapper saw a granted checkpoint and a safety
rejection and wrote ``outcome_unknown: True``; the RESTORE_GPU_SERVICES
compensation was then withheld ("automatic GPU service restoration requires
resolved node actions"), the record parked BLOCKED, and ``nvidia-fabricmanager``
and ``nvidia-persistenced`` stayed stopped on an isolated node for a reset that
never happened. The reset loop now attests ``node_action_outcome_known`` when
every nvidia-smi it spawned returned; the wrapper keeps the safety refusal
(an operator, no reboot rung) but writes no unknown outcome, so the restore
runs and the record does not park.
"""

from __future__ import annotations

from typing import Any

import pytest

from gpu_fault.execution.node_action_uncertainty import (
    restoration_refusal,
    unresolved_node_action,
)
from gpu_fault.models import WorkflowOperation, WorkflowStatus, WorkflowStepStatus
from gpu_fault.node_agent.late_ownership import (
    OwnershipRefused,
    execute_with_final_ownership,
    require_physical_ownership,
)
from gpu_fault.node_agent.operations.reset import GpuResetBusy
from gpu_fault.node_agent.protocol import SignedNodeAction
from gpu_fault.recovery_safety import unresolved_details
from tests._builders import workflow_request, workflow_step, workflow_step_execution
from tests.regional.test_late_ownership_agent_gate import command as command_fixture
from tests.regional.test_late_ownership_agent_gate_edges import (
    FakeNodeAgent,
    boundary_handler,
)

OP = WorkflowOperation
REFUSAL = "OWNERSHIP_FINAL_CLIENTS_CHANGED"


def _signed() -> SignedNodeAction:
    return SignedNodeAction(command=command_fixture(), signature="unused")


def _refusing_handler(*, attested: bool):
    @boundary_handler
    def handler(_: Any) -> dict[str, Any]:
        require_physical_ownership()  # attempt 1: granted, then busy
        refused = OwnershipRefused(REFUSAL, device_clients=[])
        if attested:
            refused.action_details["node_action_outcome_known"] = True
        raise refused

    return handler


def test_an_attested_refusal_after_a_granted_checkpoint_is_not_unknown() -> None:
    agent = FakeNodeAgent()
    try:
        execute_with_final_ownership(agent, _signed(), _refusing_handler(attested=True))
    except OwnershipRefused as refused:
        details = refused.action_details
    else:  # pragma: no cover - the handler always refuses
        raise AssertionError("the refusal must propagate")

    assert details["physical_ownership_checks"], "a checkpoint was granted first"
    assert "node_action_not_started" not in details, "a reset was spawned once"
    assert details.get("outcome_unknown") is None, (
        "every physical call returned: nothing is in flight"
    )
    assert details["manual_confirmation_required"] is True, (
        "still a safety refusal: an operator, never a reboot rung"
    )
    assert details["node_action_outcome_known"] is True


def test_an_unattested_refusal_after_a_granted_checkpoint_stays_unknown() -> None:
    agent = FakeNodeAgent()
    try:
        execute_with_final_ownership(
            agent, _signed(), _refusing_handler(attested=False)
        )
    except OwnershipRefused as refused:
        details = refused.action_details
    else:  # pragma: no cover
        raise AssertionError("the refusal must propagate")

    assert details["outcome_unknown"] is True
    assert "node_action_not_started" not in details


def test_a_busy_refusal_attests_its_verdict() -> None:
    busy = GpuResetBusy("In use by another client", evidence={}, attempts=3)

    assert busy.action_details["node_action_outcome_known"] is True
    assert "manual_confirmation_required" not in busy.action_details, (
        "a busy refusal keeps its ladder; only the attestation is added"
    )


def _reset_execution(details: dict[str, Any]):
    return workflow_step_execution(
        2, OP.RESET_GPU, WorkflowStepStatus.FAILED, error=REFUSAL, details=details
    )


def test_the_attested_refusal_resolves_the_node_action_for_the_restore() -> None:
    attested = _reset_execution(
        {
            "reason": REFUSAL,
            "safety_rejection": True,
            "manual_confirmation_required": True,
            "node_action_outcome_known": True,
            "physical_ownership_checks": [{"boundary": "AGENT_PRE_SPAWN"}],
            "reset_failed": ["GPU-a"],
            "reset_outcome_unknown": [],
        }
    )
    unknown = _reset_execution(
        {
            "reason": REFUSAL,
            "safety_rejection": True,
            "manual_confirmation_required": True,
            "outcome_unknown": True,
            "reset_outcome_unknown": [],
        }
    )

    assert unresolved_node_action(attested) is False
    assert unresolved_node_action(unknown) is True
    assert unresolved_details(attested.details) is False, (
        "the operator hold reads the attestation too"
    )
    assert unresolved_details(unknown.details) is True

    steps = [
        workflow_step(OP.MARK_UNSCHEDULABLE, node_ids=["node-a"]),
        workflow_step(OP.QUIESCE_GPU_SERVICES, node_ids=["node-a"]),
        workflow_step(OP.RESET_GPU, node_ids=["node-a"]),
        workflow_step(OP.RESTORE_GPU_SERVICES, node_ids=["node-a"]),
    ]
    for execution, refused in ((attested, False), (unknown, True)):
        workflow = workflow_request(
            "wf-reset",
            "inc-reset",
            status=WorkflowStatus.RUNNING,
            official_steps=steps,
            completed_step_indexes=[0, 1],
            step_executions=[
                workflow_step_execution(0, OP.MARK_UNSCHEDULABLE),
                workflow_step_execution(1, OP.QUIESCE_GPU_SERVICES),
                execution,
            ],
        )
        outcome = restoration_refusal(workflow, steps[3])
        assert (outcome is not None) is refused, (
            "the restore is withheld only while an outcome is really unknown"
        )


def test_the_failure_compensation_restores_even_while_an_outcome_is_unknown() -> None:
    """The record is landing a failure (``pending_failure_step_index`` set): its
    node is isolated already, and withholding the restore left the host services
    stopped and the quiesce taint on -- a dark node no later validation could
    read, and every operator exit refused (live, gpu-eks-cluster)."""

    steps = [
        workflow_step(OP.MARK_UNSCHEDULABLE, node_ids=["node-a"]),
        workflow_step(OP.QUIESCE_GPU_SERVICES, node_ids=["node-a"]),
        workflow_step(OP.RESET_GPU, node_ids=["node-a"]),
        workflow_step(OP.RESTORE_GPU_SERVICES, node_ids=["node-a"]),
    ]
    unknown = _reset_execution(
        {"manual_confirmation_required": True, "outcome_unknown": True}
    )
    workflow = workflow_request(
        "wf-reset",
        "inc-reset",
        status=WorkflowStatus.RUNNING,
        official_steps=steps,
        completed_step_indexes=[0, 1],
        step_executions=[
            workflow_step_execution(0, OP.MARK_UNSCHEDULABLE),
            workflow_step_execution(1, OP.QUIESCE_GPU_SERVICES),
            unknown,
        ],
    )

    assert restoration_refusal(workflow, steps[3]) is not None, (
        "an in-chain restore still waits for the verdict"
    )
    landing = workflow.model_copy(
        update={"pending_failure_step_index": 2, "pending_failure_error": "reset"}
    )
    assert restoration_refusal(landing, steps[3]) is None, (
        "the compensation of a parked failure restores the isolated node"
    )
    assert unresolved_node_action(unknown) is True, (
        "the record still parks for confirm-node-action afterwards"
    )


@pytest.mark.parametrize(
    "details",
    [
        {"workflow_deadline_remote_command_cancellation": {"cancelled": 1}},
        {
            "workflow_deadline_remote_command_cancellation": {
                "cancelled": 0,
                "cancellation_requested": 1,
            }
        },
        {"workflow_lifetime_exceeded": True},
        {"step_waiting_timeout_seconds": 600},
        {"remote_status_source": "workflow-timeout"},
        {"remote_status": "LEASED"},
        {"node_action_state": "PENDING", "node_action_command_id": "agent/reset"},
    ],
    ids=[
        "deadline-cancelled",
        "cancellation-requested",
        "lifetime",
        "step-wait",
        "workflow-source",
        "open-remote",
        "pending-on-agent",
    ],
)
def test_the_compensation_waits_while_the_control_plane_gave_up_on_an_open_command(
    details: dict[str, Any],
) -> None:
    """A remote command the workflow deadline, lifetime or step wait cancelled
    reads FAILED in the Store, but the agent never answered: whatever it had
    started may still be running on the node, so the failure compensation does
    not restore the GPU services over it. Only an answer from the agent --
    even an unknown or refused one -- lets the compensation run."""

    steps = [
        workflow_step(OP.QUIESCE_GPU_SERVICES, node_ids=["node-a"]),
        workflow_step(OP.RESET_GPU, node_ids=["node-a"]),
        workflow_step(OP.RESTORE_GPU_SERVICES, node_ids=["node-a"]),
    ]
    gave_up = _reset_execution(
        {
            "manual_confirmation_required": True,
            "outcome_unknown": True,
            "remote_status": "FAILED",
            **details,
        }
    )
    answered = _reset_execution(
        {
            "manual_confirmation_required": True,
            "outcome_unknown": True,
            "remote_status": "FAILED",
        }
    )

    def landing(execution: Any) -> Any:
        return workflow_request(
            "wf-reset",
            "inc-reset",
            status=WorkflowStatus.RUNNING,
            official_steps=steps,
            completed_step_indexes=[0],
            step_executions=[
                workflow_step_execution(0, OP.QUIESCE_GPU_SERVICES),
                execution,
            ],
            pending_failure_step_index=1,
            pending_failure_error="reset",
        )

    assert restoration_refusal(landing(gave_up), steps[2]) is not None, (
        "the agent never answered: the reset may still be running"
    )
    assert restoration_refusal(landing(answered), steps[2]) is None, (
        "the agent answered with an unknown outcome: nothing is in flight"
    )

"""A pending client-verification poll keeps the step's continuation.

Live (eks-first site, DESTR-016): a ``VERIFY_NO_GPU_CLIENTS`` attempt the agent
had accepted but not answered within the executor's poll window came back as a
bare ``node_action_state: PENDING`` WAITING. The record replaced the previous
verdict, so the attempt counter the 60-attempt bound counts went 11 -> 3 and
the command a preemption froze mid-poll no longer carried the barrier. The
pending poll now rides on the last verdict's continuation; its own keys win.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from tests._builders import build_store, copy_model, node_action_result
from tests.execution._support import (
    NodeActionPending,
    NodeActionStatus,
    NodeActionWorkflowAdapter,
    StubFleetRegistry,
    WorkflowExecutionRequest,
    WorkflowOperation,
    WorkflowStepContext,
    WorkflowStepStatus,
    workflow_state,
    workflow_step_execution,
)

KEY = "workflow/destr016/VERIFY_NO_GPU_CLIENTS"
CLUSTERS = {
    "eks-managed": ("gpu-eks-cluster", "eks-node-a.example.internal"),
    "hyperpod": ("hp-cluster", "hyperpod-node-b"),
}
HOLDER = {"gpu_uuid": "GPU-a", "pid": "129860", "process_name": "destr016-destr0"}
REASON = "GpuClientsActive: GPU device clients are still active: GPU-a:129860"


def _parked_verify(node: str, attempt: int) -> dict:
    return {
        "gpu_client_quiesce_attempt": attempt,
        "gpu_client_kind": "device",
        "gpu_client_holders": [HOLDER],
        "gpu_client_holders_total": 1,
        "waiting_node": node,
        "reason": REASON,
    }


def _site(cluster_id: str, node: str, *, verify_record: dict | None, **parameters):
    """A quiesced node whose VERIFY step last recorded ``verify_record``."""

    store = build_store()
    incident, workflow = workflow_state(
        store,
        [
            WorkflowOperation.QUIESCE_GPU_SERVICES,
            WorkflowOperation.VERIFY_NO_GPU_CLIENTS,
        ],
    )
    incident = copy_model(incident, cluster_id=cluster_id, node_ids=[node])
    steps = [
        copy_model(
            step,
            execution_owner="gpu-fault-node-agent",
            node_ids=[node],
            gpu_uuids=["GPU-a"],
            parameters=dict(parameters) if index else {},
        )
        for index, step in enumerate(workflow.official_steps)
    ]
    executions = [
        workflow_step_execution(
            0,
            WorkflowOperation.QUIESCE_GPU_SERVICES,
            details={
                "agent_generations": {node: 7},
                "maintenance_window_expires_at": "2099-01-01T00:00:00+00:00",
            },
        )
    ]
    if verify_record is not None:
        executions.append(
            workflow_step_execution(
                1,
                WorkflowOperation.VERIFY_NO_GPU_CLIENTS,
                WorkflowStepStatus.WAITING,
                details=verify_record,
            )
        )
    workflow = copy_model(workflow, official_steps=steps, step_executions=executions)
    return incident, workflow, steps


def _execute(sender, cluster_id, node, incident, workflow, step):
    registry = StubFleetRegistry({node: f"http://{node}:9099"}, generation=7)
    # The maintenance window check reads the registry's clock; the window
    # recorded by the quiesce is open.
    registry.now = lambda: datetime.now(timezone.utc)
    adapter = NodeActionWorkflowAdapter({}, "s" * 32, sender=sender, registry=registry)
    return adapter.execute(
        WorkflowStepContext(
            workflow=workflow,
            incident=incident,
            step=step,
            step_index=1,
            request=WorkflowExecutionRequest(expected_fencing_token=3),
            idempotency_key=KEY,
        )
    )


def _still_running(node: str):
    """An agent that accepted the attempt and has not answered yet."""

    def sender(_endpoint, envelope):
        raise NodeActionPending(
            envelope.command.command_id, {"node_action_endpoint": f"http://{node}:9099"}
        )

    return sender


@pytest.mark.parametrize("provider", sorted(CLUSTERS), ids=sorted(CLUSTERS))
def test_a_pending_poll_carries_the_last_verdict_and_the_bound_continues(
    provider: str,
) -> None:
    cluster_id, node = CLUSTERS[provider]
    incident, workflow, steps = _site(
        cluster_id, node, verify_record=_parked_verify(node, 3)
    )

    pending = _execute(
        _still_running(node), cluster_id, node, incident, workflow, steps[1]
    )

    assert pending.status is WorkflowStepStatus.WAITING
    assert pending.details["node_action_state"] == "PENDING"
    assert "/attempt-4" in pending.details["node_action_command_id"], (
        "the poll belongs to attempt 4, counted from the carried verdict"
    )
    assert pending.details["gpu_client_quiesce_attempt"] == 3, (
        "the attempt the 60-attempt bound counts survives the pending poll"
    )
    assert pending.details["reason"] == REASON, "the parked reason survives"
    assert pending.details["gpu_client_holders"] == [HOLDER]
    assert pending.details["gpu_client_kind"] == "device"
    assert pending.details["waiting_node"] == node
    assert pending.details["node_action_accepted_nodes"] == [node]
    assert pending.details["node_action_endpoint"] == f"http://{node}:9099"

    # The next cycle reads the pending record as the step's last one; the
    # agent now answers attempt 4 -- the count continues, it does not restart.
    answered = copy_model(
        workflow,
        step_executions=[
            *workflow.step_executions,
            workflow_step_execution(
                1,
                WorkflowOperation.VERIFY_NO_GPU_CLIENTS,
                WorkflowStepStatus.WAITING,
                details=dict(pending.details),
            ),
        ],
    )

    def verdict(_endpoint, envelope):
        assert "/attempt-4" in envelope.command.command_id, envelope.command.command_id
        return node_action_result(
            envelope.command.command_id,
            WorkflowOperation.VERIFY_NO_GPU_CLIENTS,
            NodeActionStatus.FAILED,
            error=REASON,
            details={
                "gpu_client_kind": "device",
                "gpu_client_holders": [HOLDER],
                "gpu_client_holders_total": 1,
            },
            attempt=4,
        )

    outcome = _execute(verdict, cluster_id, node, incident, answered, steps[1])

    assert outcome.status is WorkflowStepStatus.WAITING
    assert outcome.details["gpu_client_quiesce_attempt"] == 4, (
        "the bound counts on from the carried attempt instead of restarting at 1"
    )
    assert "node_action_state" not in outcome.details, "a verdict is not a poll"
    assert "clients are still active" in outcome.details["reason"]


def test_a_pending_poll_without_a_previous_verdict_carries_nothing() -> None:
    cluster_id, node = CLUSTERS["eks-managed"]
    incident, workflow, steps = _site(cluster_id, node, verify_record=None)

    pending = _execute(
        _still_running(node), cluster_id, node, incident, workflow, steps[1]
    )

    assert pending.status is WorkflowStepStatus.WAITING
    assert pending.details["node_action_state"] == "PENDING"
    assert "/attempt-1" in pending.details["node_action_command_id"]
    assert "gpu_client_quiesce_attempt" not in pending.details
    assert "reason" not in pending.details, "nothing is invented for a first attempt"


def test_a_spare_eligibility_probe_never_carries_a_verdict() -> None:
    """The probe borrows the parent step's index; a definitive busy answer is
    FAILED where it is produced, and a pending probe must not look answered."""

    cluster_id, node = CLUSTERS["hyperpod"]
    incident, workflow, steps = _site(
        cluster_id, node, verify_record=_parked_verify(node, 3), spare_health_check=True
    )

    pending = _execute(
        _still_running(node), cluster_id, node, incident, workflow, steps[1]
    )

    assert pending.status is WorkflowStepStatus.WAITING
    assert pending.details["node_action_state"] == "PENDING"
    assert "gpu_client_quiesce_attempt" not in pending.details, (
        "a carried attempt would read as the spare's definitive busy answer"
    )
    assert "reason" not in pending.details

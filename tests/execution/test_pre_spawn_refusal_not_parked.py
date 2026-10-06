"""A reset refused at the Agent's pre-spawn boundary is a verdict, not an unknown.

Live (gpu-eks-cluster, DESTR-001 on the previous release): the final ownership
recheck refused the reset for an ``nvidia-smi`` holder, the Agent reported it
as ``outcome_unknown`` and the chain parked BLOCKED / NEEDS_OPERATOR with the
node cordoned, quarantined and quiesced. The current Agent attests the refusal
(``node_action_outcome_known``, ``reset_verdict``) and says nothing was spawned
(``node_action_not_started``); the executor then lands an ordinary failure --
the compensation restore runs, the record ends FAILED, the node is not held by a
parked record -- and the incident is the operator's as ESCALATED. A record an
older Agent wrote with the unknown shape still parks, which the operator exits
of ``confirm-node-action`` and ``workflow-reconcile`` are for.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from gpu_fault.execution import WorkflowStepOutcome
from gpu_fault.models import (
    BlockedKind,
    IncidentState,
    WorkflowExecutionRequest,
    WorkflowOperation,
    WorkflowStatus,
)
from gpu_fault.orchestration.arbitration import RecoveryArbiter
from gpu_fault.orchestration.families.conflicts import NodeConflictService
from tests._builders import active_workflow_executor, build_store
from tests.execution._cov95_runtime_workflows import RecordingAdapter
from tests.execution._support import workflow_state

OPERATIONS = [
    WorkflowOperation.QUIESCE_GPU_SERVICES,
    WorkflowOperation.RESET_GPU,
    WorkflowOperation.RESTORE_GPU_SERVICES,
]
HOLDER = {
    "gpu_uuid": "GPU-a",
    "pid": "197055",
    "device": "/dev/nvidia0",
    "process_name": "python3",
}
# What ``execute_with_final_ownership`` + ``_reset_gpu`` report for a refusal
# before the first reset of the GPU was spawned.
ATTESTED_PRE_SPAWN_REFUSAL: dict[str, Any] = {
    "reason": "OWNERSHIP_FINAL_CLIENTS_CHANGED",
    "safety_rejection": True,
    "manual_confirmation_required": True,
    "node_action_not_started": True,
    "node_action_outcome_known": True,
    "agent_queue_ownership_checked": True,
    "ownership_check_boundary": "AGENT_PRE_SPAWN",
    "physical_ownership_checks": [],
    "node_action_command_id": "workflow/1/node-a",
    "persistent_device_clients": [HOLDER],
    "gpu_reset_attempts_spawned": 0,
    "reset_verdict": "not-attempted",
    "gpu_readback": {"GPU-a": {"reset_status.reset_required": "No"}},
    "reset_completed": [],
    "reset_outcome_unknown": [],
    "reset_failed": [],
    "reset_not_attempted": ["GPU-a"],
}
# The same refusal as the previous release's Agent reported it.
UNATTESTED_UNKNOWN_REFUSAL: dict[str, Any] = {
    "reason": "OWNERSHIP_FINAL_CLIENTS_CHANGED",
    "safety_rejection": True,
    "manual_confirmation_required": True,
    "outcome_unknown": True,
    "ownership_check_boundary": "AGENT_PRE_SPAWN",
    "persistent_device_clients": [HOLDER],
    "reset_completed": [],
    "reset_not_attempted": [],
}


def _run_chain(details: dict[str, Any]) -> tuple[Any, Any, Any, Any, Any]:
    store = build_store()
    incident, workflow = workflow_state(store, OPERATIONS)
    now = datetime.now(UTC)
    store.save_incident_and_workflow(
        incident.model_copy(update={"created_at": now, "updated_at": now}),
        workflow.model_copy(update={"created_at": now, "updated_at": now}),
    )
    adapter = RecordingAdapter()
    adapter.outcomes[WorkflowOperation.RESET_GPU] = WorkflowStepOutcome.failed(
        "node execution result", details=details
    )
    executor = active_workflow_executor(store, [adapter], OPERATIONS)
    result = executor.execute(
        workflow.request_id,
        WorkflowExecutionRequest(expected_fencing_token=workflow.fencing_token),
    )
    saved = store.get_workflow(workflow.request_id)
    occupying = NodeConflictService(
        store, RecoveryArbiter()
    ).active_node_exclusive_workflow(incident.cluster_id, set(incident.node_ids))
    return store, incident, result, saved, (adapter, occupying)


def test_an_attested_pre_spawn_refusal_fails_the_chain_without_parking() -> None:
    store, incident, result, saved, (adapter, occupying) = _run_chain(
        ATTESTED_PRE_SPAWN_REFUSAL
    )

    assert result.status is WorkflowStatus.FAILED, result
    assert saved.blocked_kind is None
    assert occupying is None, "no parked record holds the node"
    assert [call.step.operation for call in adapter.calls] == OPERATIONS, (
        "the compensation restore runs: nothing is in flight on the node"
    )
    reset = [
        item
        for item in saved.step_executions
        if item.operation is WorkflowOperation.RESET_GPU
    ][-1]
    assert reset.details.get("outcome_unknown") is None
    assert reset.details["reset_not_attempted"] == ["GPU-a"]
    assert reset.details["reset_verdict"] == "not-attempted"
    assert store.get_incident(incident.incident_id).state is IncidentState.ESCALATED


def test_the_unattested_unknown_shape_of_an_older_agent_still_parks() -> None:
    _store, _incident, result, saved, (adapter, occupying) = _run_chain(
        UNATTESTED_UNKNOWN_REFUSAL
    )

    assert result.status is WorkflowStatus.BLOCKED
    assert saved.blocked_kind is BlockedKind.NEEDS_OPERATOR
    assert occupying is not None and occupying.request_id == saved.request_id
    assert [call.step.operation for call in adapter.calls] == OPERATIONS, (
        "the compensation restore still runs; only the record parks"
    )


@pytest.mark.parametrize(
    "missing", ["node_action_outcome_known", "node_action_not_started"]
)
def test_either_attestation_alone_settles_the_refusal(missing: str) -> None:
    """``node_action_outcome_known`` says every spawned call returned a verdict;
    ``node_action_not_started`` says nothing was spawned. Either one is a known
    outcome; the boundary name on its own is not what settles it."""

    details = {
        key: value
        for key, value in ATTESTED_PRE_SPAWN_REFUSAL.items()
        if key != missing
    }

    _store, _incident, result, saved, (_adapter, occupying) = _run_chain(details)

    assert result.status is WorkflowStatus.FAILED, missing
    assert saved.blocked_kind is None
    assert occupying is None

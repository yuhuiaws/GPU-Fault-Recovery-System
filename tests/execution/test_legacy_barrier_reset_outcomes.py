from __future__ import annotations

from copy import deepcopy
from datetime import timedelta
from urllib.parse import parse_qs, urlsplit

import pytest

from gpu_fault.execution import WorkflowStepOutcome
from gpu_fault.models import (
    BlockedKind,
    RecoveryAction,
    WorkflowOperation,
    WorkflowStatus,
    WorkflowStepStatus,
    workflow_is_open,
)
from gpu_fault.node_agent import (
    NodeActionExecutionState,
    NodeActionResult,
    NodeActionStatus,
)
from gpu_fault.orchestration.escalation import HardwareEscalationService
from tests.execution._cov95_runtime_workflows import FlowHarness
from tests.execution.test_legacy_reset_outcomes import FABRIC_ERROR, LEGACY_PROGRESS
from tests.execution.test_node_action_transport_retry import FakeResponse, submission
from tests.fleet._support import heartbeat, signed
from tests.hyperpod._cov95_provider_extra_safety import (
    provider_extra_isolation as provider_extra_isolation,
)
from tests.regional._cov95_runtime_adapter import fleet_adapter

RESET = WorkflowOperation.RESET_GPU
FABRIC = WorkflowOperation.RESET_ALL_GPUS_NVSWITCHES
QUIESCE = WorkflowOperation.QUIESCE_GPU_SERVICES
RESTORE = WorkflowOperation.RESTORE_GPU_SERVICES
NODES = ["node-a", "node-b", "node-c"]


def barrier_flow(monkeypatch, operation, failure_details, error):
    h = FlowHarness([QUIESCE, operation, RESTORE])
    adapter = fleet_adapter()
    fleet = adapter.registry
    fleet.register(signed(heartbeat("node-c")))
    polls = []
    submits = []

    def read_ledger(request, **_kwargs):
        if request.get_method() != "GET":
            submits.append(request)
            raise AssertionError("completed legacy results must never be resubmitted")
        url = urlsplit(request.full_url)
        node = url.hostname
        command_id = parse_qs(url.query)["command_id"][0]
        action = (
            WorkflowOperation.VERIFY_NO_GPU_CLIENTS
            if "/barrier/prepare/" in command_id
            else operation
        )
        polls.append((action, node))
        failed = action is operation and node == "node-b"
        result = NodeActionResult(
            command_id=command_id,
            operation=action,
            status=NodeActionStatus.FAILED if failed else NodeActionStatus.SUCCEEDED,
            retryable=False,
            error=error if failed else None,
            details=deepcopy(failure_details)
            if failed
            else {"reset_gpu_uuids": [f"GPU-{node}"]},
        )
        return FakeResponse(
            submission(
                command_id,
                NodeActionExecutionState.FAILED
                if failed
                else NodeActionExecutionState.SUCCEEDED,
                result,
            )
        )

    monkeypatch.setattr("gpu_fault.adapters.node_action.transport.urlopen", read_ledger)
    h.adapter.outcomes[QUIESCE] = WorkflowStepOutcome.succeeded(
        details={
            "agent_generations": dict.fromkeys(NODES, 1),
            "maintenance_window_started_at": fleet.now().isoformat(),
            "maintenance_window_expires_at": (
                fleet.now() + timedelta(minutes=5)
            ).isoformat(),
        }
    )
    h.executor.adapters.insert(0, adapter)
    h.amend(
        official_steps=[
            step.model_copy(
                update={
                    "execution_owner": adapter.owner
                    if step.operation is operation
                    else step.execution_owner,
                    "node_ids": NODES,
                    "gpu_uuids": ["GPU-a", "GPU-b", "GPU-c"],
                    "parameters": {
                        "gpu_uuids_by_node": {
                            node: ["GPU-a", "GPU-b", "GPU-c"]
                            if node == "node-b"
                            else [f"GPU-{node}"]
                            for node in NODES
                        }
                    },
                }
            )
            for step in h.workflow.official_steps
        ]
    )
    return h, adapter, polls, submits


@pytest.mark.parametrize(
    ("operation", "details", "error"),
    [
        (RESET, LEGACY_PROGRESS, "ResetProgressError: reset timed out"),
        (FABRIC, {}, FABRIC_ERROR),
        (
            RESET,
            {"reset_outcome_unknown": None},
            "ResetProgressError: malformed progress",
        ),
    ],
    ids=["per-gpu-progress", "fabric-exception", "malformed-progress"],
)
def test_legacy_barrier_uncertainty_blocks_restore_escalation_and_later_commits(
    monkeypatch, operation, details, error
):
    h, adapter, polls, submits = barrier_flow(monkeypatch, operation, details, error)
    assert h.execute().status is WorkflowStatus.RUNNING, "prepare must not reset GPUs"
    assert h.execute().status is WorkflowStatus.BLOCKED, (
        "an unknown barrier commit must require an operator, not compensation"
    )
    saved = h.store.get_workflow(h.workflow.request_id)
    assert saved.blocked_kind is BlockedKind.NEEDS_OPERATOR, saved
    assert workflow_is_open(saved.status, saved.blocked_kind), saved
    failure = next(
        item for item in saved.step_executions if item.operation is operation
    )
    assert failure.status is WorkflowStepStatus.FAILED, failure
    assert failure.details["manual_confirmation_required"] is True, failure
    assert failure.details["completed_nodes"] == ["node-a"], failure
    assert failure.details["failed_nodes"] == ["node-b"], failure
    progress = failure.details["node_failure_details"]["node-b"]
    assert progress["outcome_unknown"] is True, progress
    for key, value in details.items():
        assert progress[key] == value, (key, progress)
    if operation is FABRIC:
        assert "reset_completed" not in progress, (
            "exception-only legacy results do not prove per-GPU progress"
        )
    barrier = adapter.barriers.store.get_barrier(
        f"{saved.request_id}/1/{operation.value}"
    )
    participant = next(
        item for item in barrier.participants if item.node_id == "node-b"
    )
    assert participant.commit_details == progress, participant
    assert [item.step.operation for item in h.adapter.calls] == [QUIESCE], (
        h.adapter.calls
    )
    assert [node for action, node in polls if action is operation] == NODES[:2], polls
    assert submits == [], "a known legacy ledger row must not cause a new submission"
    classification = HardwareEscalationService.classify(saved)
    assert classification is not None, saved
    assert classification[1] is RecoveryAction.ESCALATE_OPERATOR, classification
    count = len(polls)
    assert h.execute().status is WorkflowStatus.BLOCKED, saved
    assert len(polls) == count, "operator-blocked replay must not query or reset again"


@pytest.mark.parametrize(
    "error",
    [
        "RuntimeError: reset failed cleanly",
        "RuntimeError: diagnostic mentioned ResetOutcomeUnknown: old error",
        "RuntimeError: gpu reset outcome unknown after 180s",
    ],
)
def test_barrier_ordinary_failures_are_not_classified_from_freeform_text(
    monkeypatch, error
):
    h, _adapter, _polls, submits = barrier_flow(
        monkeypatch, RESET, {"reset_outcome_unknown": []}, error
    )
    assert h.execute().status is WorkflowStatus.RUNNING, "prepare must complete first"
    assert h.execute().status is WorkflowStatus.FAILED, (
        "ordinary failures must keep the existing compensation path"
    )
    assert [item.step.operation for item in h.adapter.calls] == [QUIESCE, RESTORE], (
        h.adapter.calls
    )
    assert submits == [], "ledger replay must not submit another action"

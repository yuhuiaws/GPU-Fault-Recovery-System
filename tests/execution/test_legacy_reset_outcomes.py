from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from gpu_fault.adapters import NodeActionWorkflowAdapter
from gpu_fault.adapters.kubernetes.stop_ownership import stop_ownership_scope
from gpu_fault.execution import WorkflowStepOutcome
from gpu_fault.execution.branch_escalation import BranchEscalator
from gpu_fault.fleet import (
    CURRENT_AGENT_PROTOCOL_VERSION,
    AgentHeartbeat,
    FleetCompatibilityPolicy,
    FleetRegistry,
    SignedAgentHeartbeat,
    sign_agent_heartbeat,
)
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
from gpu_fault.orchestration.arbitration import RecoveryArbiter
from gpu_fault.orchestration.dag_branching import DagBrancher
from gpu_fault.orchestration.escalation import HardwareEscalationService
from tests.execution._cov95_runtime_workflows import FlowHarness
from tests.execution.test_node_action_transport_retry import (
    ENDPOINT,
    SECRET,
    FakeAgentWire,
    step_context,
    submission,
    wire,
)

RESET = WorkflowOperation.RESET_GPU
FABRIC = WorkflowOperation.RESET_ALL_GPUS_NVSWITCHES
QUIESCE = WorkflowOperation.QUIESCE_GPU_SERVICES
RESTORE = WorkflowOperation.RESTORE_GPU_SERVICES
GPUS = ["GPU-a", "GPU-b", "GPU-c"]
LEGACY_PROGRESS = {
    "reset_completed": ["GPU-a"],
    "reset_outcome_unknown": ["GPU-b"],
    "reset_failed": [],
    "reset_not_attempted": ["GPU-c"],
}
FABRIC_ERROR = (
    "ResetOutcomeUnknown: gpu reset outcome unknown after 180s; "
    "refusing to retry automatically"
)


def registered_legacy_agent(store, *, old_required):
    old, new = "a" * 64, "b" * 64
    required, compatible = (old, new) if old_required else (new, old)
    policy = FleetCompatibilityPolicy(
        required_artifact_sha256=required,
        compatible_artifact_sha256s=frozenset({compatible}),
        required_compatibility_digest=required,
        compatible_compatibility_digests=frozenset({compatible}),
        required_config_digest=required,
        compatible_config_digests=frozenset({compatible}),
    )
    registry = FleetRegistry(store, SECRET, policy)
    heartbeat = AgentHeartbeat(
        cluster_id="cluster-a",
        node_id="node-a",
        endpoint=ENDPOINT,
        agent_protocol_version=CURRENT_AGENT_PROTOCOL_VERSION,
        agent_version="0.10.0",
        artifact_sha256=old,
        compatibility_digest=old,
        config_digest=old,
        policy_version="610",
        runtime_profile_version="active-v1",
        allowed_operations=[
            QUIESCE,
            RESET,
            FABRIC,
            RESTORE,
            WorkflowOperation.VERIFY_NO_GPU_CLIENTS,
        ],
        boot_id="boot-a",
        node_instance_id="instance-a",
        agent_incarnation_id="incarnation-a",
        observed_at=datetime.now(timezone.utc),
    )
    registry.register(
        SignedAgentHeartbeat(
            heartbeat=heartbeat, signature=sign_agent_heartbeat(heartbeat, SECRET)
        )
    )
    assert registry.readiness("cluster-a", ["node-a"]).ready, (
        "the staged required/compatible pins legitimately admit the old Agent"
    )
    return registry


@pytest.mark.parametrize(
    "old_required", [True, False], ids=["old-required", "old-compatible"]
)
@pytest.mark.parametrize("dag", [False, True], ids=["sequential", "dag"])
@pytest.mark.parametrize("operation", [RESET, FABRIC])
def test_compatible_old_agent_ledger_failure_stays_unknown_end_to_end(
    monkeypatch, operation, dag, old_required
):
    h = FlowHarness([QUIESCE, operation, RESTORE])
    registry = registered_legacy_agent(h.store, old_required=old_required)
    now = datetime.now(timezone.utc)
    h.adapter.outcomes[QUIESCE] = WorkflowStepOutcome.succeeded(
        details={
            "agent_generations": {"node-a": 1},
            "maintenance_window_started_at": now.isoformat(),
            "maintenance_window_expires_at": (now + timedelta(minutes=5)).isoformat(),
        }
    )
    command_id = f"{h.workflow.request_id}/1/{operation.value}/node-a/agent-1"
    old_result = NodeActionResult(
        command_id=command_id,
        operation=operation,
        status=NodeActionStatus.FAILED,
        retryable=False,
        error=(
            "ResetProgressError: gpu reset outcome unknown after 120s (GPU GPU-b)"
            if operation is RESET
            else FABRIC_ERROR
        ),
        details=dict(LEGACY_PROGRESS) if operation is RESET else {},
    )
    agent = wire(
        monkeypatch,
        FakeAgentWire(
            submission(command_id, NodeActionExecutionState.FAILED, old_result)
        ),
    )
    adapter = NodeActionWorkflowAdapter({}, SECRET, registry=registry)
    h.executor.adapters.insert(0, adapter)
    h.amend(
        dag_enabled=dag,
        official_steps=[
            step.model_copy(
                update={
                    "execution_owner": adapter.owner
                    if step.operation is operation
                    else step.execution_owner,
                    "gpu_uuids": GPUS,
                    "branch_id": "branch:node-a" if dag else None,
                    "branch_node_ids": ["node-a"] if dag else [],
                    "depends_on_step_indexes": [index - 1] if dag and index else [],
                }
            )
            for index, step in enumerate(h.workflow.official_steps)
        ],
    )

    def no_rung(*_args):
        raise AssertionError("legacy unknown outcomes cannot authorize a hardware rung")

    if dag:
        h.executor.branch_escalator = BranchEscalator(
            DagBrancher(RecoveryArbiter()), no_rung
        )
    # Existing results remain queryable even when no new submission is authorized.
    with stop_ownership_scope(None):
        result = h.execute()
    assert result.status is WorkflowStatus.BLOCKED
    saved = h.store.get_workflow(h.workflow.request_id)
    assert saved.blocked_kind is BlockedKind.NEEDS_OPERATOR
    assert workflow_is_open(saved.status, saved.blocked_kind), (
        "legacy uncertainty must retain the same operator-held node occupancy"
    )
    failure = next(
        item for item in saved.step_executions if item.operation is operation
    )
    assert failure.details["outcome_unknown"] is True
    assert failure.details["manual_confirmation_required"] is True
    assert failure.details["failed_nodes"] == ["node-a"]
    assert failure.details["reset_outcome_compatibility"] == (
        "per-gpu-progress" if operation is RESET else "legacy-exception-tag"
    )
    if operation is RESET:
        for key, value in LEGACY_PROGRESS.items():
            assert failure.details[key] == value
    else:
        assert "reset_completed" not in failure.details, (
            "the legacy exception-only result does not prove any per-GPU completion"
        )
    assert [context.step.operation for context in h.adapter.calls] == [QUIESCE]
    assert saved.branch_escalation_counts == {}
    assert RESTORE not in saved.completed_operations
    classification = HardwareEscalationService.classify(saved)
    assert classification is not None
    assert classification[:3] == (
        "manual_confirmation_required",
        RecoveryAction.ESCALATE_OPERATOR,
        WorkflowOperation.ESCALATE_SUPPORT,
    )
    assert h.execute().status is WorkflowStatus.BLOCKED
    assert agent.polls == 1
    assert agent.submits == [], "a nonretryable old ledger result is never resubmitted"


@pytest.mark.parametrize(
    ("operation", "error", "details"),
    [
        (RESET, "RuntimeError: reset failed cleanly", {"reset_outcome_unknown": []}),
        (FABRIC, "RuntimeError: reset failed cleanly", {}),
        (
            FABRIC,
            "RuntimeError: diagnostic mentions ResetOutcomeUnknown: old error",
            {},
        ),
        (FABRIC, "ResetOutcomeUnknownOther: not the historical exception class", {}),
        (FABRIC, "RuntimeError: gpu reset outcome unknown after 180s", {}),
        (WorkflowOperation.REMEDIATE_DRIVER, FABRIC_ERROR, {}),
    ],
)
def test_legacy_compatibility_does_not_reclassify_other_errors(
    operation, error, details
):
    value = NodeActionResult(
        command_id="workflow/step/node-a",
        operation=operation,
        status=NodeActionStatus.FAILED,
        error=error,
        details=details,
    )
    adapter = NodeActionWorkflowAdapter(
        {"node-a": ENDPOINT}, SECRET, sender=lambda *_args: value
    )
    outcome = adapter.execute(step_context(adapter, operation))
    assert outcome.status is WorkflowStepStatus.FAILED
    assert "outcome_unknown" not in outcome.details
    assert "manual_confirmation_required" not in outcome.details


@pytest.mark.parametrize("progress", [None, False, "GPU-a", [None], [""]])
def test_malformed_legacy_progress_is_not_a_known_failure(progress):
    value = NodeActionResult(
        command_id="workflow/step/node-a",
        operation=RESET,
        status=NodeActionStatus.FAILED,
        error="ResetProgressError: incomplete legacy progress",
        details={"reset_outcome_unknown": progress},
    )
    adapter = NodeActionWorkflowAdapter(
        {"node-a": ENDPOINT}, SECRET, sender=lambda *_args: value
    )
    outcome = adapter.execute(step_context(adapter))
    assert outcome.status is WorkflowStepStatus.FAILED
    assert outcome.details["outcome_unknown"] is True
    assert outcome.details["manual_confirmation_required"] is True
    assert outcome.details["reset_outcome_compatibility"] == "invalid-progress"

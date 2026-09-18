from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from gpu_fault.adapters import NodeActionWorkflowAdapter
from gpu_fault.adapters.common import NodeActionPending
from gpu_fault.fleet import AgentTransitionRequest, BarrierState
from gpu_fault.models import WorkflowOperation, WorkflowStepStatus
from gpu_fault.node_agent import NodeActionResult, NodeActionStatus, SignedNodeAction
from tests._builders import workflow_step_execution
from tests.execution.test_node_action_transport_retry import ENDPOINT, SECRET
from tests.fleet._support import NOW
from tests.regional._cov95_runtime_adapter import context_for, fleet_adapter
from tests.regional._cov95_runtime_support import offline_runtime as offline_runtime


def success(signed: SignedNodeAction, **details: Any) -> NodeActionResult:
    return NodeActionResult(
        command_id=signed.command.command_id,
        operation=signed.command.operation,
        status=NodeActionStatus.SUCCEEDED,
        details=details,
    )


@pytest.mark.parametrize(
    ("nodes", "mapping", "error"),
    [
        ([], {}, "explicit node target"),
        (["node-a", "node-a"], {}, "duplicates"),
        (["node-a"], [], "missing explicit GPU UUIDs"),
        (["node-a"], {}, "missing explicit GPU UUIDs"),
        (["node-a"], {"node-a": []}, "missing explicit GPU UUIDs"),
        (["node-a"], {"node-a": [1]}, "missing explicit GPU UUIDs"),
    ],
)
def test_missing_or_ambiguous_target_mapping_is_refused_without_a_send(
    nodes: list[str], mapping: Any, error: str
) -> None:
    calls = []
    adapter = NodeActionWorkflowAdapter(
        {"node-a": ENDPOINT}, SECRET, sender=lambda *args: calls.append(args)
    )
    result = adapter.execute(
        context_for(
            adapter,
            WorkflowOperation.COLLECT_DIAGNOSTIC_BUNDLE,
            nodes=nodes,
            parameters={"gpu_uuids_by_node": mapping},
        )
    )
    assert result.status is WorkflowStepStatus.FAILED
    assert error in result.error
    assert calls == []


@pytest.mark.parametrize("existing_barrier", [False, True])
def test_readiness_loss_aborts_a_prepared_barrier_before_any_reset(
    existing_barrier: bool,
) -> None:
    calls = []

    def send(endpoint: str, signed: SignedNodeAction) -> NodeActionResult:
        calls.append(signed.command.operation)
        return success(signed)

    adapter = fleet_adapter(sender=send)
    context = context_for(adapter)
    if existing_barrier:
        assert adapter.execute(context).status is WorkflowStepStatus.WAITING
    adapter.registry.drain_agent(
        "cluster-a",
        "node-a",
        AgentTransitionRequest(
            expected_generation=1, transition_id="unit", reason="unit"
        ),
    )
    result = adapter.execute(context)
    assert result.status is WorkflowStepStatus.FAILED
    assert "fleet consistency gate" in result.error
    assert calls == (
        [WorkflowOperation.VERIFY_NO_GPU_CLIENTS] * 2 if existing_barrier else []
    )
    if existing_barrier:
        barrier = adapter.barriers.store.get_barrier(context.idempotency_key)
        assert barrier.state is BarrierState.ABORTED


@pytest.mark.parametrize(
    ("defect", "reason"),
    [
        ("no-generations", "incomplete"),
        ("no-expiry", "incomplete"),
        ("bad-expiry", "expiry is invalid"),
        ("naive-expiry", "timezone"),
        ("expired", "window expired"),
        ("zero-generation", "missing agent generation"),
        ("string-generation", "missing agent generation"),
        ("missing-node", "missing agent generation"),
        ("generation-drift", "maintenance agent fence"),
    ],
)
def test_quiesce_evidence_cannot_authorize_a_reset_after_scope_or_time_drift(
    defect: str, reason: str
) -> None:
    calls = []
    adapter = fleet_adapter(sender=lambda *args: calls.append(args))
    context = context_for(adapter, nodes=["node-a"], step_index=1)
    details = {
        "agent_generations": {"node-a": 1},
        "maintenance_window_expires_at": (NOW + timedelta(seconds=30)).isoformat(),
    }
    if defect == "no-generations":
        details.pop("agent_generations")
    elif defect == "no-expiry":
        details.pop("maintenance_window_expires_at")
    elif defect == "bad-expiry":
        details["maintenance_window_expires_at"] = "invalid"
    elif defect == "naive-expiry":
        details["maintenance_window_expires_at"] = NOW.replace(tzinfo=None).isoformat()
    elif defect == "expired":
        details["maintenance_window_expires_at"] = NOW.isoformat()
    elif defect == "zero-generation":
        details["agent_generations"] = {"node-a": 0}
    elif defect == "string-generation":
        details["agent_generations"] = {"node-a": "1"}
    elif defect == "missing-node":
        details["agent_generations"] = {"node-b": 1}
    else:
        adapter.registry.drain_agent(
            "cluster-a",
            "node-a",
            AgentTransitionRequest(
                expected_generation=1, transition_id="unit", reason="unit"
            ),
        )
    context = replace(
        context,
        workflow=context.workflow.model_copy(
            update={
                "step_executions": [
                    workflow_step_execution(
                        0, WorkflowOperation.QUIESCE_GPU_SERVICES, details=details
                    )
                ]
            }
        ),
    )
    result = adapter.execute(context)
    assert result.status is WorkflowStepStatus.FAILED
    assert reason in result.error
    assert calls == []


@pytest.mark.parametrize("mode", ["expired-restore", "spare", "reboot-handoff"])
def test_cleanup_and_spare_probes_use_their_own_liveness_contract(mode: str) -> None:
    sent = []

    def send(endpoint: str, signed: SignedNodeAction) -> NodeActionResult:
        sent.append(signed.command)
        return success(signed)

    adapter = fleet_adapter(sender=send)
    operation = (
        WorkflowOperation.VERIFY_NO_GPU_CLIENTS
        if mode == "spare"
        else WorkflowOperation.RESTORE_GPU_SERVICES
    )
    node = "node-b" if mode == "spare" else "node-a"
    context = context_for(
        adapter,
        operation,
        nodes=[node],
        parameters={
            "gpu_uuids_by_node": {node: ["GPU-b" if mode == "spare" else "GPU-a"]},
            **({"spare_health_check": True} if mode == "spare" else {}),
            **(
                {"preemption_quiesce_handoff_after_reboot": True}
                if mode == "reboot-handoff"
                else {}
            ),
        },
        step_index=1,
    )
    context = replace(
        context,
        workflow=context.workflow.model_copy(
            update={
                "step_executions": [
                    workflow_step_execution(
                        0,
                        WorkflowOperation.QUIESCE_GPU_SERVICES,
                        details={
                            "agent_generations": {"node-a": 1},
                            "maintenance_window_expires_at": (
                                NOW - timedelta(seconds=1)
                            ).isoformat(),
                        },
                    )
                ]
            }
        ),
    )
    assert adapter.execute(context).status is WorkflowStepStatus.SUCCEEDED
    assert [(item.operation, item.node_id, item.agent_generation) for item in sent] == [
        (operation, node, 1)
    ]


@pytest.mark.parametrize("raw_attempt", [None, "invalid"])
def test_busy_commit_retries_only_the_uncommitted_node(raw_attempt: Any) -> None:
    calls = []
    failed_once = []

    def send(endpoint: str, signed: SignedNodeAction) -> NodeActionResult:
        item = signed.command
        calls.append((item.operation, item.node_id))
        if (
            item.operation is WorkflowOperation.RESET_GPU
            and item.node_id == "node-b"
            and not failed_once
        ):
            failed_once.append(1)
            return NodeActionResult(
                command_id=item.command_id,
                operation=item.operation,
                status=NodeActionStatus.FAILED,
                error="GPU compute clients are still active",
            )
        return success(signed, verified=True)

    adapter = fleet_adapter(sender=send, verify_max_attempts=2)
    context = context_for(adapter)
    assert adapter.execute(context).status is WorkflowStepStatus.WAITING
    context = replace(
        context,
        workflow=context.workflow.model_copy(
            update={
                "step_executions": [
                    workflow_step_execution(
                        0,
                        context.step.operation,
                        details={"gpu_reset_commit_attempt": raw_attempt},
                    )
                ]
            }
        ),
    )
    waiting = adapter.execute(context)
    assert waiting.status is WorkflowStepStatus.WAITING
    assert waiting.details["waiting_nodes"] == ["node-b"]
    context = replace(
        context,
        workflow=context.workflow.model_copy(
            update={
                "step_executions": [
                    workflow_step_execution(
                        0,
                        context.step.operation,
                        WorkflowStepStatus.WAITING,
                        details=waiting.details,
                    )
                ]
            }
        ),
    )
    result = adapter.execute(context)
    assert result.status is WorkflowStepStatus.SUCCEEDED
    assert calls == [
        (WorkflowOperation.VERIFY_NO_GPU_CLIENTS, "node-a"),
        (WorkflowOperation.VERIFY_NO_GPU_CLIENTS, "node-b"),
        (WorkflowOperation.RESET_GPU, "node-a"),
        (WorkflowOperation.RESET_GPU, "node-b"),
        (WorkflowOperation.RESET_GPU, "node-b"),
    ]
    assert result.details["node_results"] == {
        "node-a": {"verified": True},
        "node-b": {"verified": True},
    }


@pytest.mark.parametrize("state", ["failed", "pending", "succeeded"])
def test_diagnostic_batch_retains_one_nodes_evidence_when_the_other_is_incomplete(
    state: str,
) -> None:
    sent = []

    def send(endpoint: str, signed: SignedNodeAction) -> NodeActionResult:
        sent.append((signed.command.node_id, signed.command.gpu_uuids))
        if signed.command.node_id == "node-b":
            if state == "pending":
                raise NodeActionPending(signed.command.command_id)
            if state == "failed":
                return NodeActionResult(
                    command_id=signed.command.command_id,
                    operation=signed.command.operation,
                    status=NodeActionStatus.FAILED,
                    error="synthetic bundle failure",
                )
        return success(signed, evidence_ref=f"unit://{signed.command.node_id}/bundle")

    adapter = NodeActionWorkflowAdapter(
        {"node-a": ENDPOINT, "node-b": "http://node-b:9099"}, SECRET, sender=send
    )
    result = adapter.execute(
        context_for(adapter, WorkflowOperation.COLLECT_DIAGNOSTIC_BUNDLE)
    )
    assert (
        result.status
        is {
            "failed": WorkflowStepStatus.FAILED,
            "pending": WorkflowStepStatus.WAITING,
            "succeeded": WorkflowStepStatus.SUCCEEDED,
        }[state]
    )
    assert result.details["node_results"]["node-a"] == {
        "evidence_ref": "unit://node-a/bundle"
    }
    assert result.details["failed_nodes"] == (["node-b"] if state == "failed" else [])
    assert result.details["waiting_nodes"] == (["node-b"] if state == "pending" else [])
    assert sorted(sent) == [("node-a", ["GPU-a"]), ("node-b", ["GPU-b"])]


@pytest.mark.parametrize("started", ["invalid", "recent", "expired"])
def test_triage_waits_only_within_its_budget_and_names_unfinished_nodes(
    started: str,
) -> None:
    now = datetime.now(timezone.utc)

    def send(endpoint: str, signed: SignedNodeAction) -> NodeActionResult:
        if signed.command.node_id == "node-b":
            raise NodeActionPending(signed.command.command_id)
        return success(signed, ranks=[{"rank": 0}])

    adapter = NodeActionWorkflowAdapter(
        {"node-a": ENDPOINT, "node-b": "http://node-b:9099"}, SECRET, sender=send
    )
    context = context_for(
        adapter,
        WorkflowOperation.COLLECT_HUNG_TRIAGE,
        parameters={
            "triage_timeout_seconds": 5,
            "not_sampled_nodes": ["node-c", "", "node-c"],
        },
    )
    context = replace(
        context,
        workflow=context.workflow.model_copy(
            update={
                "step_executions": [
                    workflow_step_execution(
                        0,
                        context.step.operation,
                        WorkflowStepStatus.WAITING,
                        details={
                            "triage_started_at": "invalid"
                            if started == "invalid"
                            else (
                                now
                                - timedelta(seconds=20 if started == "expired" else 1)
                            ).isoformat()
                        },
                    )
                ]
            }
        ),
    )
    result = adapter.execute(context)
    assert result.details["node_results"] == {"node-a": {"ranks": [{"rank": 0}]}}
    if started == "expired":
        assert result.status is WorkflowStepStatus.SUCCEEDED
        assert result.details["undetermined_nodes"] == ["node-b"]
        assert result.details["not_sampled_nodes"] == ["node-c"]
    else:
        assert result.status is WorkflowStepStatus.WAITING
        assert result.details["pending_nodes"] == ["node-b"]

from __future__ import annotations

from types import SimpleNamespace

import pytest

from gpu_fault.models import WorkflowOperation, WorkflowStatus
from tests.execution._cov95_runtime_workflows import FlowHarness


class FleetReadback:
    def __init__(self, mode):
        self.mode = mode
        self.calls = []

    def fleet_rollout_fence_deployments(self, cluster_id):
        self.calls.append(("fence", cluster_id))
        if self.mode == "fence-error":
            raise OSError("fence service unavailable")
        return ["deployment-a"] if self.mode == "fence-busy" else []

    def readiness(self, cluster_id, nodes):
        self.calls.append(("readiness", cluster_id, nodes))
        if self.mode == "readiness-error":
            raise OSError("agent registry unavailable")
        return SimpleNamespace(ready=self.mode == "ready", nodes=[])


@pytest.mark.parametrize(
    ("mode", "reason"),
    [
        ("fence-error", "rollout fence is unavailable"),
        ("fence-busy", "1 active deployment"),
        ("readiness-error", "compatibility preflight is unavailable"),
        ("readiness-false", "fleet readiness returned false"),
        ("missing-target", "no explicit node target"),
    ],
)
def test_fleet_gate_refuses_before_claim_and_retries_only_with_fresh_readiness(
    mode, reason
):
    h = FlowHarness([WorkflowOperation.RESET_GPU])
    registry = FleetReadback(mode)
    h.executor.fleet_registry = registry
    if mode == "missing-target":
        h.amend(
            official_steps=[
                h.workflow.official_steps[0].model_copy(update={"node_ids": []})
            ]
        )
    before = h.store.get_workflow(h.workflow.request_id)
    blocked = h.execute()
    assert blocked.status is WorkflowStatus.PENDING, blocked
    assert reason in (blocked.error or ""), blocked
    assert h.adapter.calls == [], "a failed fleet gate must never submit the reset"
    assert h.store.get_workflow(h.workflow.request_id) == before, (
        "preflight refusal must not claim, consume a lease, or amend the workflow"
    )
    expected_calls = [("fence", "cluster-a")]
    if mode.startswith("readiness"):
        expected_calls.append(("readiness", "cluster-a", ["node-a"]))
    assert registry.calls == expected_calls, registry.calls
    registry.mode = "ready"
    if mode == "missing-target":
        h.amend(
            official_steps=[
                h.workflow.official_steps[0].model_copy(update={"node_ids": ["node-a"]})
            ]
        )
    result = h.execute()
    assert result.status is WorkflowStatus.SUCCEEDED, result
    assert [call.step.operation for call in h.adapter.calls] == [
        WorkflowOperation.RESET_GPU
    ], h.adapter.calls
    assert registry.calls[-2:] == [
        ("fence", "cluster-a"),
        ("readiness", "cluster-a", ["node-a"]),
    ], registry.calls


def test_rollout_fence_does_not_strand_restore_after_reset_was_completed():
    h = FlowHarness(
        [
            WorkflowOperation.QUIESCE_GPU_SERVICES,
            WorkflowOperation.RESET_GPU,
            WorkflowOperation.RESTORE_GPU_SERVICES,
        ]
    )
    registry = FleetReadback("fence-error")
    h.executor.fleet_registry = registry
    h.amend(
        status=WorkflowStatus.RUNNING,
        completed_step_indexes=[0, 1],
        completed_operations=[
            WorkflowOperation.QUIESCE_GPU_SERVICES,
            WorkflowOperation.RESET_GPU,
        ],
    )
    result = h.execute()
    assert result.status is WorkflowStatus.SUCCEEDED, result
    assert registry.calls == [], "completed mutation must not strand its cleanup"
    assert [call.step.operation for call in h.adapter.calls] == [
        WorkflowOperation.RESTORE_GPU_SERVICES
    ], h.adapter.calls


def test_workload_restart_checks_the_rollout_fence_without_inventing_node_actions():
    h = FlowHarness([WorkflowOperation.RESTART_WORKLOAD])
    registry = FleetReadback("ready")
    h.executor.fleet_registry = registry
    result = h.execute()
    assert result.status is WorkflowStatus.SUCCEEDED, result
    assert registry.calls == [("fence", "cluster-a")], registry.calls
    assert [call.step.operation for call in h.adapter.calls] == [
        WorkflowOperation.RESTART_WORKLOAD
    ], h.adapter.calls

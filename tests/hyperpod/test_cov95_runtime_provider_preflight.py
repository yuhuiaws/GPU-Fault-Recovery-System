from __future__ import annotations

from typing import Any

import pytest
from botocore.exceptions import NoCredentialsError

from gpu_fault.adapters import HyperPodLifecycleStepAdapter
from gpu_fault.execution import WorkflowExecutionRequest, WorkflowStepContext
from gpu_fault.hyperpod import (
    HyperPodAction,
    HyperPodAdapterConfig,
    HyperPodAdapterError,
    HyperPodLifecycleAdapter,
)
from gpu_fault.hyperpod_spares import SpareAllocation
from gpu_fault.models import WorkflowOperation, WorkflowStepStatus
from gpu_fault.store import InMemoryStore
from tests.execution._support import (
    FakeHyperPodLifecycle,
    FakeSpareCoordinator,
    RecordingNodeActionAdapter,
    isolated_kubernetes_adapter,
    workflow_state,
)
from tests.hyperpod.test_hyperpod import FakeHyperPodClient


@pytest.mark.parametrize("recovery", [None, "", "Unknown"])
def test_provider_preflight_refuses_unknown_node_recovery_policy(recovery: Any) -> None:
    client = FakeHyperPodClient(node_recovery=recovery)
    lifecycle = HyperPodLifecycleAdapter(
        HyperPodAdapterConfig(cluster_name="hp-cluster"), client=client
    )
    result = lifecycle.preflight(
        HyperPodAction.REPLACE,
        ["worker-group-1"],
        isolation_verified_nodes=["worker-group-1"],
        require_execution_enabled=False,
    )
    assert not result.safe_to_submit, result
    assert any("NodeRecovery" in error for error in result.gate_failures), result
    assert client.reboot_requests == client.replace_requests == [], client.list_requests


@pytest.mark.parametrize(
    "error",
    [HyperPodAdapterError("fake provider policy unavailable"), NoCredentialsError()],
)
def test_failed_provider_preflight_cannot_fall_through_to_warm_spare_activation(
    error: Exception, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = InMemoryStore()
    incident, workflow = workflow_state(store, [WorkflowOperation.REPLACE_NODE])
    lifecycle = FakeHyperPodLifecycle()

    def preflight(*args: Any, **kwargs: Any) -> Any:
        raise error

    monkeypatch.setattr(lifecycle, "preflight", preflight)
    spares = FakeSpareCoordinator(
        SpareAllocation(
            applicable=True, sufficient=True, required=1, selected_node_ids=("spare-a",)
        )
    )
    node_actions = RecordingNodeActionAdapter()
    adapter = HyperPodLifecycleStepAdapter(
        lifecycle,
        spare_coordinator=spares,
        node_action_adapter=node_actions,
        kubernetes_adapter=isolated_kubernetes_adapter(),
    )
    step = workflow.official_steps[0].model_copy(
        update={
            "execution_owner": adapter.owner,
            "parameters": {"replacement_strategy": "HEALTHY_WARM_SPARE_ONLY"},
        }
    )
    context = WorkflowStepContext(
        workflow=workflow.model_copy(update={"official_steps": [step]}),
        incident=incident,
        step=step,
        step_index=0,
        request=WorkflowExecutionRequest(
            expected_fencing_token=3, confirm_cluster_name="hp-cluster"
        ),
        idempotency_key="owned/0/REPLACE_NODE",
    )
    result = adapter.execute(context)
    assert result.status is WorkflowStepStatus.FAILED, result
    assert "preflight" in (result.error or ""), result
    assert spares.calls == spares.releases == [], (spares.calls, spares.releases)
    assert node_actions.contexts == [] and lifecycle.calls == 0, node_actions.contexts

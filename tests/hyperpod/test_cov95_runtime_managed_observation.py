from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any

import pytest

from gpu_fault.execution import WorkflowStepOutcome
from gpu_fault.fleet import AgentLifecycleState
from gpu_fault.managed_recovery import RegionalHyperPodManagedRecoveryObserver
from gpu_fault.models import WorkflowOperation, WorkflowStepStatus
from tests.hyperpod._cov95_runtime_managed import ManagedHarness
from tests.hyperpod.test_managed_recovery import agent


@pytest.mark.parametrize(
    "defect", ["missing", "empty", "foreign", "duplicate", "row-type", "id-type"]
)
def test_managed_recovery_requires_complete_target_evidence_before_confirmation(
    defect: str,
) -> None:
    h = ManagedHarness()
    initial = h.start()
    h.replacement()
    details = deepcopy(initial.details)
    if defect == "missing":
        details.pop("managed_targets")
    elif defect == "empty":
        details["managed_targets"] = []
    elif defect == "foreign":
        details["managed_targets"][0]["old_node_id"] = "foreign-node"
    elif defect == "duplicate":
        details["managed_targets"] *= 2
    elif defect == "row-type":
        details["managed_targets"] = ["not-an-object"]
    else:
        details["managed_targets"][0]["old_node_id"] = ["not-an-id"]
    previous = WorkflowStepOutcome.waiting(
        operation_id=initial.adapter_operation_id, details=details
    )
    result = h.follow(previous)
    assert result.status is WorkflowStepStatus.FAILED, result
    assert "target evidence" in (result.error or ""), result
    assert (
        h.store.get_agent("hp-cluster", "k8s-old").lifecycle_state
        is AgentLifecycleState.ACTIVE
    ), result
    assert h.isolation.nodes == [] and h.lifecycle.mutations == [], h.isolation.nodes


@pytest.mark.parametrize("nodes", [[], ["k8s-old", "k8s-old"]])
def test_managed_recovery_requires_unique_requested_nodes(nodes: list[str]) -> None:
    h = ManagedHarness()
    step = h.step.model_copy(update={"node_ids": nodes})
    context = replace(
        h.context,
        step=step,
        workflow=h.workflow.model_copy(update={"official_steps": [step]}),
    )
    result = h.observer.observe(context)
    assert result.status is WorkflowStepStatus.FAILED, result
    assert "node_ids" in (result.error or ""), result
    assert h.lifecycle.mutations == [] and h.isolation.nodes == [], result


@pytest.mark.parametrize(
    "state",
    ["unchanged", "provider-pending", "provider-absent", "missing-name", "no-registry"],
)
def test_unconfirmed_managed_replacement_remains_waiting(
    state: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = ManagedHarness()
    first = h.start()
    if state == "provider-pending":
        h.lifecycle.node = h.lifecycle.node.model_copy(update={"status": "Replacing"})
    elif state == "provider-absent":
        monkeypatch.setattr(h.lifecycle, "list_nodes", lambda **_kwargs: [])
    elif state == "missing-name":
        h.lifecycle.node = h.lifecycle.node.model_copy(
            update={"instance_id": None, "kubernetes_labels": {}}
        )
    elif state == "no-registry":
        h.replacement()
        h.observer.registry = None
    result = h.follow(first)
    assert result.status is WorkflowStepStatus.WAITING, result
    assert result.details["managed_targets"] == first.details["managed_targets"], result
    assert (
        result.details["managed_recovery_started_at"]
        == first.details["managed_recovery_started_at"]
    ), result
    assert h.lifecycle.mutations == [] and h.isolation.nodes == [], result


def test_agent_only_boot_change_confirms_reboot_without_provider_submission() -> None:
    h = ManagedHarness(WorkflowOperation.RESTART_NODE)
    first = h.start()
    h.store.save_agent(agent("k8s-old", "i-old", "new-boot", generation=2))
    result = h.follow(first)
    assert result.status is WorkflowStepStatus.SUCCEEDED, result
    assert result.details["externally_confirmed"] is True, result
    assert result.details["node_rebindings"]["k8s-old"] == "k8s-old", result
    assert h.isolation.nodes == ["k8s-old"] and h.lifecycle.mutations == [], result


@pytest.mark.parametrize("state", ["waiting", "failed", "exception"])
def test_replacement_needs_successful_isolation_before_becoming_rebound(
    state: str,
) -> None:
    h = ManagedHarness()
    first = h.start()
    h.replacement()
    contexts = []

    def isolate(context: Any) -> WorkflowStepOutcome:
        contexts.append(context)
        if state == "exception":
            raise RuntimeError("fake isolation unavailable")
        if state == "failed":
            return WorkflowStepOutcome.failed("fake isolation refusal")
        return WorkflowStepOutcome.waiting(details={"admission": "pending"})

    h.observer.kubernetes_adapter = SimpleNamespace(execute=isolate)
    result = h.follow(first)
    [context] = contexts
    assert context.step.operation is WorkflowOperation.MARK_UNSCHEDULABLE, context
    assert context.step.node_ids == ["k8s-new"], context
    assert context.idempotency_key.endswith("/replacement-isolation"), context
    if state == "waiting":
        assert result.status is WorkflowStepStatus.WAITING, result
        assert (
            result.details["managed_recovery_state"] == "REPLACEMENT_ISOLATION_PENDING"
        ), result
        assert result.details["admission"] == "pending", result
    else:
        assert result.status is WorkflowStepStatus.FAILED, result
        assert "fake isolation" in (result.error or ""), result
    assert h.lifecycle.mutations == [], h.lifecycle.mutations


def test_timeout_is_clamped_to_workflow_budget_and_keeps_one_persisted_alert() -> None:
    h = ManagedHarness()
    h.workflow = h.workflow.model_copy(
        update={
            "execution_deadline": datetime.now(timezone.utc) + timedelta(seconds=10)
        }
    )
    h.context = replace(h.context, workflow=h.workflow)
    sent: list[str] = []
    h.observer.alert_sender = sent.append
    first = h.start()
    result = h.follow(first)
    repeated = h.follow(first)
    assert result.status is repeated.status is WorkflowStepStatus.FAILED, (
        result,
        repeated,
    )
    [notification] = h.store.list_notifications()
    assert notification.notification_id in (result.error or ""), result
    assert sent == [notification.notification_id, notification.notification_id], sent
    assert (
        h.store.get_agent("hp-cluster", "k8s-old").lifecycle_state
        is AgentLifecycleState.ACTIVE
    ), result
    assert h.lifecycle.mutations == [], h.lifecycle.mutations


@pytest.mark.parametrize("registered", [False, True])
def test_regional_managed_observer_never_crosses_cluster_routing_boundary(
    registered: bool,
) -> None:
    h = ManagedHarness()
    router = RegionalHyperPodManagedRecoveryObserver(
        {"hp-cluster" if registered else "foreign-cluster": h.observer}
    )
    result = router.observe(h.context)
    assert result.status is (
        WorkflowStepStatus.WAITING if registered else WorkflowStepStatus.FAILED
    ), result
    if not registered:
        assert "no managed recovery observer registered" in (result.error or ""), result
    assert h.lifecycle.mutations == [], h.lifecycle.mutations

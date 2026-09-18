from __future__ import annotations

from dataclasses import replace

import pytest

from gpu_fault.adapters.common import ANNOTATION_FENCING, ANNOTATION_INCIDENT
from gpu_fault.execution import WorkflowStepOutcome
from gpu_fault.models import WorkflowOperation, WorkflowStepStatus
from tests.execution._cov95_runtime_restart import ApiError
from tests.hyperpod._cov95_runtime_confirmation import ConfirmationHarness
from tests.hyperpod._cov95_runtime_failover import FailoverHarness

SNAPSHOT = WorkflowOperation.TRIGGER_HEALTH_SNAPSHOT
CLIENTS = WorkflowOperation.VERIFY_NO_GPU_CLIENTS


@pytest.mark.parametrize("fence", [None, "2", "4", "invalid"])
def test_warm_spare_preflight_requires_the_current_isolation_fence(
    fence: str | None,
) -> None:
    h = FailoverHarness()
    annotations = h.core.nodes["node-a"]["metadata"]["annotations"]
    if fence is None:
        annotations.pop(ANNOTATION_FENCING)
    else:
        annotations[ANNOTATION_FENCING] = fence
    result = h.execute()
    assert result.status is WorkflowStepStatus.FAILED, (
        "a stale, missing or malformed isolation fence must not authorize failover",
        result,
    )
    assert result.details["safety_rejection"] is True, result
    assert h.spares.calls == [] and h.actions.contexts == [], (
        h.spares.calls,
        h.actions.contexts,
    )
    assert h.provider.submissions == [], "provider mutation remains prohibited"


@pytest.mark.parametrize("dependency", ["scheduler", "snapshot"])
def test_pending_activation_refuses_a_missing_required_adapter_and_releases_its_reservation(
    dependency: str,
) -> None:
    h = FailoverHarness()
    h.actions.outcomes[SNAPSHOT] = WorkflowStepOutcome.waiting(
        operation_id="snapshot-pending"
    )
    first = h.execute()
    assert first.status is WorkflowStepStatus.WAITING, first
    h.follow(first)
    h.actions.outcomes[SNAPSHOT] = WorkflowStepOutcome.succeeded()
    if dependency == "scheduler":
        h.adapter.kubernetes_adapter = None
        expected = "Kubernetes"
    else:
        h.adapter.node_action_adapter = None
        expected = "health snapshot adapter"
    with pytest.raises(ValueError, match=expected):
        h.execute()
    assert h.spares.releases == [(["spare-a"], h.context.incident.incident_id)], (
        h.spares.releases
    )
    assert len(h.spares.calls) == 1, "resume must not allocate another spare"
    assert len(h.actions.contexts) == 1, (
        "missing dependencies must be refused before another snapshot"
    )
    assert h.provider.submissions == [], h.provider.submissions


def test_pending_snapshot_reuses_reserved_spare_and_preserves_alias_rebindings() -> (
    None
):
    h = FailoverHarness()
    h.actions.outcomes[SNAPSHOT] = WorkflowStepOutcome.waiting(
        details={"reason": "fresh inventory pending"}
    )
    first = h.execute()
    assert first.status is WorkflowStepStatus.WAITING, first
    first.details["provider_baselines"] = {
        "node-a": {"instance_id": "i-old", "kubernetes_node_name": "hyperpod-i-old"}
    }
    h.follow(first)
    h.actions.outcomes[SNAPSHOT] = WorkflowStepOutcome.succeeded(
        details={"inventory_generation": 2}
    )
    second = h.execute()
    assert second.status is WorkflowStepStatus.SUCCEEDED, second
    assert second.details["node_rebindings"] == {
        "node-a": "spare-a",
        "i-old": "spare-a",
        "hyperpod-i-old": "spare-a",
    }, second
    assert second.details["active_health_snapshot"] == {"inventory_generation": 2}, (
        second
    )
    assert second.details["provider_mutation_submitted"] is False, second
    assert len(h.spares.calls) == 1 and h.spares.releases == [], h.spares.calls
    assert all(call.step.node_ids == ["spare-a"] for call in h.actions.contexts), (
        h.actions.contexts
    )
    assert h.provider.submissions == [], h.provider.submissions


@pytest.mark.parametrize("error", ["snapshot rejected", ""])
def test_failed_active_snapshot_releases_unconsumed_capacity(error: str) -> None:
    h = FailoverHarness()
    h.actions.outcomes[SNAPSHOT] = WorkflowStepOutcome.failed(error)
    with pytest.raises(ValueError, match=error or "active health snapshot failed"):
        h.execute()
    assert h.spares.releases == [(["spare-a"], h.context.incident.incident_id)], (
        h.spares.releases
    )
    assert h.provider.submissions == [], h.provider.submissions


def test_foreign_spare_isolation_is_never_taken_over_without_ownership_proof() -> None:
    h = FailoverHarness()
    h.core.nodes["spare-a"]["metadata"]["annotations"][ANNOTATION_INCIDENT] = (
        "foreign-incident"
    )
    with pytest.raises(ValueError, match="already isolated by another incident"):
        h.execute()
    assert h.spares.releases == [(["spare-a"], h.context.incident.incident_id)], (
        h.spares.releases
    )
    assert h.actions.contexts == [], (
        "health snapshot cannot precede successful owned isolation"
    )
    assert (
        h.core.nodes["spare-a"]["metadata"]["annotations"][ANNOTATION_INCIDENT]
        == "foreign-incident"
    ), h.core.nodes


def test_scheduler_conflict_holds_the_reservation_then_retries_isolation() -> None:
    h = FailoverHarness()
    h.core.conflict = True
    first = h.execute()
    assert first.status is WorkflowStepStatus.WAITING, first
    assert first.details["activated_spare_nodes"] == ["spare-a"], first
    assert h.spares.releases == [] and h.actions.contexts == [], first
    h.follow(first)
    h.core.conflict = False
    second = h.execute()
    assert second.status is WorkflowStepStatus.SUCCEEDED, second
    assert len(h.spares.calls) == 1 and h.spares.releases == [], h.spares.calls
    assert h.core.nodes["spare-a"]["spec"]["unschedulable"] is True, h.core.nodes


def test_allocation_count_mismatch_releases_all_returned_nodes_before_isolation() -> (
    None
):
    h = FailoverHarness()
    h.spares.allocation = replace(
        h.spares.allocation, selected_node_ids=("spare-a", "spare-b")
    )
    with pytest.raises(ValueError, match="does not match fault node count"):
        h.execute()
    assert h.spares.releases == [
        (["spare-a", "spare-b"], h.context.incident.incident_id)
    ], h.spares.releases
    assert h.core.patches == [] and h.actions.contexts == [], (
        h.core.patches,
        h.actions.contexts,
    )


@pytest.mark.parametrize("verdict", ["pending", "busy", "failed", "ready"])
def test_node_agent_client_check_distinguishes_pending_evidence_from_a_busy_spare(
    verdict: str,
) -> None:
    h = FailoverHarness()
    h.spares.check_clients = True
    outcomes = {
        "pending": WorkflowStepOutcome.waiting(operation_id="client-check-pending"),
        "busy": WorkflowStepOutcome.waiting(
            details={
                "gpu_client_quiesce_attempt": 1,
                "reason": "compute client remains",
            }
        ),
        "failed": WorkflowStepOutcome.failed("agent check rejected"),
        "ready": WorkflowStepOutcome.succeeded(),
    }
    h.actions.outcomes[CLIENTS] = outcomes[verdict]
    result = h.execute()
    expected = {
        "pending": WorkflowStepStatus.WAITING,
        "busy": WorkflowStepStatus.FAILED,
        "failed": WorkflowStepStatus.FAILED,
        "ready": WorkflowStepStatus.SUCCEEDED,
    }[verdict]
    assert result.status is expected, result
    check = h.actions.contexts[0]
    assert check.step.operation is CLIENTS and check.step.node_ids == ["spare-a"], check
    assert check.step.parameters == {
        "compute_clients_only": True,
        "spare_health_check": True,
    }, check
    assert len(h.actions.contexts) == (2 if verdict == "ready" else 1), (
        h.actions.contexts
    )
    assert h.provider.submissions == [], h.provider.submissions


@pytest.mark.parametrize("status", [404, 500])
def test_initial_isolation_lookup_never_treats_unavailable_nodes_as_verified(
    status: int,
) -> None:
    h = FailoverHarness()
    h.core.read_error = ApiError(status)
    if status == 404:
        result = h.execute()
        assert (
            result.status is WorkflowStepStatus.FAILED
            and result.details["safety_rejection"]
        ), result
    else:
        with pytest.raises(ApiError, match="fake Kubernetes status 500"):
            h.execute()
    assert h.spares.calls == [] and h.provider.preflights == [], (
        h.spares.calls,
        h.provider.preflights,
    )


def test_provider_alias_is_resolved_before_isolation_preflight() -> None:
    h = FailoverHarness()
    del h.core.nodes["node-a"]
    result = h.execute()
    assert result.status is WorkflowStepStatus.SUCCEEDED, result
    assert h.provider.preflights[0][1]["isolation_verified_nodes"] == [
        "hyperpod-i-old",
        "node-a",
    ], h.provider.preflights
    assert h.provider.preflights[0][1]["require_execution_enabled"] is False, (
        h.provider.preflights
    )
    assert h.provider.submissions == [], h.provider.submissions


@pytest.mark.parametrize("fence", [None, "2", "3", "4", "invalid"])
def test_reboot_poll_reasserts_isolation_only_for_its_current_fence(
    fence: str | None,
) -> None:
    h = ConfirmationHarness(WorkflowOperation.RESTART_NODE)
    h.ready = False
    h.details["observed_isolation"] = {
        "node-a": {"unschedulable": True, "kubernetes_node": "node-a"}
    }
    node = h.scheduler.core.node
    node["spec"]["unschedulable"] = False
    if fence is None:
        node["metadata"]["annotations"].pop(ANNOTATION_FENCING)
    else:
        node["metadata"]["annotations"][ANNOTATION_FENCING] = fence
    result = h.execute()
    assert result.status is WorkflowStepStatus.WAITING, result
    assert node["spec"]["unschedulable"] is (fence == "3"), (
        "re-cordon must not mutate a node with an unknown or different generation",
        node,
    )
    assert result.details.get("isolation_reasserted", []) == (
        ["node-a"] if fence == "3" else []
    ), result
    assert h.provider.submissions == [], h.provider.submissions

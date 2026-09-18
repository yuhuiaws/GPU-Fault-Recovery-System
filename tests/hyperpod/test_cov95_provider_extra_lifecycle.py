from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from gpu_fault.adapters import HyperPodLifecycleStepAdapter
from gpu_fault.adapters.common import (
    ANNOTATION_FENCING,
    ANNOTATION_INCIDENT,
    QUARANTINE_TAINT,
)
from gpu_fault.execution import WorkflowStepOutcome
from gpu_fault.hyperpod import HyperPodNodeFailure
from gpu_fault.hyperpod_spares import SpareAllocation
from gpu_fault.models import WorkflowOperation, WorkflowStepStatus
from tests._builders import workflow_request, workflow_step, workflow_step_execution
from tests.execution._support import FakeSpareCoordinator
from tests.hyperpod._cov95_provider_extra_safety import (
    provider_extra_isolation as provider_extra_isolation,
)
from tests.hyperpod._cov95_provider_extra_steps import InitialStep, RecordingLifecycle
from tests.hyperpod._cov95_runtime_failover import FailoverHarness
from tests.hyperpod.test_managed_recovery import agent


def test_negative_stabilization_window_is_refused_before_any_transport():
    provider = RecordingLifecycle()
    with pytest.raises(ValueError, match="stabilization_seconds cannot be negative"):
        HyperPodLifecycleStepAdapter(provider, post_reboot_stabilization_seconds=-1)
    assert provider.resolutions == provider.preflights == [], (
        "invalid stabilization configuration reached a provider read"
    )


def test_adapter_supports_only_its_registered_operations_and_owner():
    h = InitialStep()
    assert h.adapter.supports(h.context.step), (
        "the valid provider-owned reboot was unsupported"
    )
    assert not h.adapter.supports(
        h.context.step.model_copy(update={"execution_owner": "other-owner"})
    ), "another owner's operation was accepted"
    assert not h.adapter.supports(
        h.context.step.model_copy(
            update={"operation": WorkflowOperation.FREEZE_EVIDENCE}
        )
    ), "a nonprovider operation was accepted"


@pytest.mark.parametrize(
    "fault", ["confirmation", "kubernetes", "cordon", "taint", "incident", "fence"]
)
def test_initial_step_requires_current_isolation_and_confirmation_before_preflight(
    fault,
):
    h = InitialStep()
    if fault == "confirmation":
        h.context = replace(
            h.context,
            request=h.context.request.model_copy(update={"confirm_cluster_name": None}),
        )
    elif fault == "kubernetes":
        h.adapter.kubernetes_adapter = None
    elif fault == "cordon":
        h.scheduler.core.node["spec"]["unschedulable"] = False
    elif fault == "taint":
        h.scheduler.core.node["spec"]["taints"] = [
            item
            for item in h.scheduler.core.node["spec"]["taints"]
            if item["key"] != QUARANTINE_TAINT
        ]
    else:
        key = ANNOTATION_INCIDENT if fault == "incident" else ANNOTATION_FENCING
        h.scheduler.core.node["metadata"]["annotations"][key] = "foreign"
    result = h.execute()
    assert result.status is WorkflowStepStatus.FAILED, f"{fault} did not block mutation"
    assert h.provider.preflights == [], "unproved isolation reached provider preflight"
    assert h.provider.calls == 0, (
        "a rejected isolation guard submitted a provider operation"
    )


def test_missing_agent_baseline_stays_explicit_and_requires_later_confirmation():
    h = InitialStep()
    h.adapter.registry = SimpleNamespace(store=h.store)
    result = h.execute()
    assert result.status is WorkflowStepStatus.WAITING, (
        "a reboot was marked complete at submission"
    )
    assert result.details["agent_baselines"] == {}, (
        "a missing Agent baseline was fabricated"
    )
    assert result.details["revoked_agents"] == [], (
        "a nonexistent Agent was reported revoked"
    )
    assert result.details["requires_external_confirmation"] is True, (
        "missing baseline suppressed the external confirmation requirement"
    )
    assert h.provider.calls == 1, (
        "the initial step did not make exactly one fake submission"
    )


def test_agent_revocation_failure_stops_before_provider_submission():
    h = InitialStep()
    h.store.save_agent(
        agent("node-a", "i-unit", "unit-before").model_copy(
            update={"cluster_id": h.incident.cluster_id}
        )
    )
    revoked = []
    transitions = []

    def refuse(*args):
        transitions.append(args)
        raise ValueError("unit Agent transition unavailable")

    h.adapter.registry = SimpleNamespace(
        store=h.store,
        drain_agent=refuse,
        revoke_agent=lambda *args: revoked.append(args),
    )
    result = h.execute()
    assert result.status is WorkflowStepStatus.FAILED, (
        "unconfirmed Agent revocation was ignored"
    )
    assert "failed to revoke" in result.error, (
        "Agent revocation failure lost its diagnostic"
    )
    assert h.provider.calls == 0, (
        "provider submission ran after a failed Agent transition"
    )
    assert [args[:2] for args in transitions] == [(h.incident.cluster_id, "node-a")], (
        "the failed transition was not attempted against the workflow's own Agent"
    )
    assert revoked == [], "the failed drain proceeded to revocation"


@pytest.mark.parametrize(
    "failure",
    [
        HyperPodNodeFailure(node_logical_id="logical-node-a"),
        HyperPodNodeFailure(node_id="legacy-node"),
        HyperPodNodeFailure(),
    ],
)
def test_provider_rejection_is_not_recorded_as_waiting_for_success(failure):
    h = InitialStep()
    h.provider.failures = [failure]
    result = h.execute()
    assert result.status is WorkflowStepStatus.FAILED, (
        "provider node rejection was hidden"
    )
    expected = failure.node_logical_id or failure.node_id or "unknown"
    assert expected in result.error, (
        "provider rejection lost its available node identifier"
    )
    assert h.provider.calls == 1, "a rejected submission was unexpectedly retried"


def test_reservation_cleanup_releases_pending_only_and_preserves_consumed_spares():
    h = InitialStep()
    coordinator = FakeSpareCoordinator(
        SpareAllocation(applicable=False, sufficient=False, required=1)
    )
    h.adapter.spare_coordinator = coordinator
    workflow = workflow_request(
        "cleanup",
        h.incident.incident_id,
        official_steps=[
            workflow_step(WorkflowOperation.REPLACE_NODE),
            workflow_step(WorkflowOperation.RESTART_NODE),
        ],
        step_executions=[
            workflow_step_execution(
                0,
                WorkflowOperation.REPLACE_NODE,
                WorkflowStepStatus.WAITING,
                details={"activated_spare_nodes": ["spare-a", "spare-b", "spare-a"]},
            ),
            workflow_step_execution(
                0,
                WorkflowOperation.REPLACE_NODE,
                WorkflowStepStatus.FAILED,
                details={"activated_spare_nodes": ["spare-c"]},
            ),
            workflow_step_execution(
                0,
                WorkflowOperation.REPLACE_NODE,
                WorkflowStepStatus.SUCCEEDED,
                details={"activated_spare_nodes": ["spare-b"]},
            ),
            workflow_step_execution(
                1,
                WorkflowOperation.RESTART_NODE,
                WorkflowStepStatus.FAILED,
                details={"activated_spare_nodes": ["unrelated-node"]},
            ),
        ],
    )
    released = h.adapter.release_spare_reservations(workflow, h.incident.incident_id)
    assert released == ["spare-a", "spare-c"], (
        "cleanup released a consumed or unrelated node"
    )
    assert coordinator.releases == [(released, h.incident.incident_id)], (
        "cleanup lost its incident ownership or duplicated the release"
    )


@pytest.mark.parametrize("coordinator_present", [False, True])
def test_cleanup_without_pending_reservations_does_not_release_nodes(
    coordinator_present,
):
    h = InitialStep()
    coordinator = FakeSpareCoordinator(
        SpareAllocation(applicable=False, sufficient=False, required=1)
    )
    h.adapter.spare_coordinator = coordinator if coordinator_present else None
    workflow = h.workflow.model_copy(
        update={
            "step_executions": [
                workflow_step_execution(
                    0,
                    WorkflowOperation.REPLACE_NODE,
                    WorkflowStepStatus.SUCCEEDED,
                    details={"activated_spare_nodes": ["consumed-spare"]},
                )
            ]
        }
    )
    assert (
        h.adapter.release_spare_reservations(workflow, h.incident.incident_id) == []
    ), "cleanup invented pending reservations"
    assert coordinator.releases == [], "a successfully consumed spare was released"


@pytest.mark.parametrize("invalid", ["not-applicable", "empty-selection"])
def test_unusable_allocation_never_falls_back_to_a_provider_replacement(invalid):
    h = FailoverHarness()
    h.spares.allocation = replace(
        h.spares.allocation,
        applicable=invalid != "not-applicable",
        selected_node_ids=() if invalid == "empty-selection" else ("spare-a",),
    )
    result = h.execute()
    assert result.status is WorkflowStepStatus.FAILED, (
        "an unusable allocation was activated"
    )
    assert "fallback is disabled" in result.error, (
        "the no-provider-fallback boundary was lost"
    )
    assert h.provider.submissions == [], "unusable spares reached a provider mutation"
    assert h.actions.contexts == [], "unusable spares reached a node action"


def test_pending_failover_without_its_coordinator_fails_closed():
    h = FailoverHarness()
    h.follow(
        WorkflowStepOutcome.waiting(
            details={
                "spare_failover_pending": True,
                "activated_spare_nodes": ["spare-a"],
            }
        )
    )
    h.adapter.spare_coordinator = None
    with pytest.raises(RuntimeError, match="spare coordinator is unavailable"):
        h.execute()
    assert h.provider.submissions == [], "lost coordinator caused provider fallback"
    assert h.spares.releases == [], "missing coordinator fabricated reservation cleanup"

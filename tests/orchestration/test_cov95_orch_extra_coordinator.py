from __future__ import annotations

from datetime import timedelta

import pytest

from gpu_fault.fleet import AgentLifecycleState
from gpu_fault.models import (
    EXECUTABLE_WORKFLOW_STATUSES,
    BlockedKind,
    IncidentState,
    RecoveryAction,
    WorkflowOperation,
    WorkflowStatus,
)
from gpu_fault.orchestration import IncidentOrchestrator
from gpu_fault.policy import ActionDisposition
from tests._builders import fault_incident
from tests.orchestration._cov95_orch_extra_faults import agent, policy_decision, xid
from tests.orchestration._cov95_orch_extra_safety import (
    orch_extra_isolation as orch_extra_isolation,
)
from tests.orchestration._cov95_orch_extra_support import (
    NOW,
    memory_store,
    stored_workflow,
)


@pytest.mark.parametrize(
    "variable",
    ["GPU_FAULT_SXID_DRIVER_REMEDIATION_CODES", "GPU_FAULT_SXID_FIRMWARE_UPDATE_CODES"],
)
@pytest.mark.parametrize("raw", ["word", "0", "-1"])
def test_orchestrator_constructor_refuses_unusable_sxid_remediation_codes(
    monkeypatch, variable, raw
):
    monkeypatch.setenv(variable, raw)
    store = memory_store()
    with pytest.raises(ValueError, match=variable):
        IncidentOrchestrator(
            store, target_driver_branch=575, target_firmware_version="unit-firmware"
        )
    assert store.list_workflows() == []


def test_orchestrator_constructor_normalizes_positive_code_sets_without_running_actions(
    monkeypatch,
):
    monkeypatch.setenv("GPU_FAULT_SXID_DRIVER_REMEDIATION_CODES", " 13, ,13,42 ")
    monkeypatch.setenv("GPU_FAULT_SXID_FIRMWARE_UPDATE_CODES", " 7,8,7 ")
    orchestrator = IncidentOrchestrator(
        memory_store(),
        target_driver_branch=575,
        target_firmware_version="unit-firmware",
    )
    assert orchestrator.sxid_driver_remediation_codes == {13, 42}
    assert orchestrator.sxid_firmware_update_codes == {7, 8}
    assert orchestrator.store.list_workflows() == []


@pytest.mark.parametrize(
    ("metric", "explicit_gpu", "expected"),
    [
        ("unknown-unit-metric", False, RecoveryAction.RUN_DIAGNOSTICS),
        ("row_remap_failure", False, RecoveryAction.DRAIN),
        ("retired_pages_pending", True, RecoveryAction.RESET_GPU),
        ("retired_pages_pending", False, RecoveryAction.QUARANTINE),
    ],
)
def test_gpu_metric_decision_keeps_default_or_requires_the_appropriate_hardware_scope(
    metric, explicit_gpu, expected
):
    store = memory_store()
    result = IncidentOrchestrator(store).gpu_metric_action(
        cluster_id="cluster-a",
        node_id="node-0",
        metric_name=metric,
        has_explicit_gpu=explicit_gpu,
        default=RecoveryAction.RUN_DIAGNOSTICS,
    )
    assert result is expected
    assert store.list_workflows() == []
    assert store.list_markers() == []


@pytest.mark.parametrize(
    "case", ["missing-lease", "expired-lease", "draining", "revoked", "missing-boot"]
)
def test_generation_fence_removes_automatic_actions_without_fresh_matching_agent_proof(
    case,
):
    store = memory_store()
    changes = {}
    if case == "missing-lease":
        changes["lease_expires_at"] = None
    elif case == "expired-lease":
        changes["lease_expires_at"] = NOW
    elif case in {"draining", "revoked"}:
        changes["lifecycle_state"] = {
            "draining": AgentLifecycleState.DRAINING,
            "revoked": AgentLifecycleState.REVOKED,
        }[case]
    else:
        changes["boot_id"] = None
    store.save_agent(agent(**changes))
    event = xid(source_boot_id="unit-boot")
    decision = policy_decision(event)
    result = IncidentOrchestrator(store).apply_fault_action_generation_fence(
        event, decision, now=NOW
    )
    assert result.disposition is ActionDisposition.BLOCKED_MISSING_EVIDENCE
    assert result.action is None
    assert result.safety_action is None
    assert result.pre_actions == []
    assert result.requires_operator is True
    assert result.marker.active is False
    assert result.marker.trusted is False
    assert any(
        reason.startswith("STALE_FAULT_GENERATION") for reason in result.reasons
    ), "a blocked generation lost its reason"
    assert decision.action is RecoveryAction.RESET_GPU


@pytest.mark.parametrize(
    "case", ["no-agent", "already-blocked", "readonly", "matching-boot"]
)
def test_generation_fence_preserves_decisions_outside_its_rejection_conditions(case):
    store = memory_store()
    event = xid(source_boot_id="unit-boot")
    decision = policy_decision(event)
    if case != "no-agent":
        store.save_agent(agent())
    if case == "already-blocked":
        decision = decision.model_copy(
            update={"disposition": ActionDisposition.BLOCKED_MISSING_EVIDENCE}
        )
    elif case == "readonly":
        decision = decision.model_copy(
            update={"action": RecoveryAction.NO_ACTION, "official_action": "IGNORE"}
        )
    result = IncidentOrchestrator(store).apply_fault_action_generation_fence(
        event, decision, now=NOW
    )
    assert result is decision
    assert store.list_workflows() == [], "the generation fence dispatched an action"


@pytest.mark.parametrize("seconds", [900, 901])
def test_unidentified_generation_uses_the_exact_age_boundary(seconds):
    store = memory_store()
    store.save_agent(agent())
    event = xid(source_event_time=NOW - timedelta(seconds=seconds))
    decision = policy_decision(event)
    result = IncidentOrchestrator(store).apply_fault_action_generation_fence(
        event, decision, now=NOW
    )
    assert result.disposition is (
        ActionDisposition.EXECUTABLE
        if seconds == 900
        else ActionDisposition.BLOCKED_MISSING_EVIDENCE
    )


@pytest.mark.parametrize(
    "status", [WorkflowStatus.RUNNING, WorkflowStatus.FAILED, WorkflowStatus.SUPERSEDED]
)
def test_simulation_refuses_nonexecutable_workflow_states_without_writing(status):
    store = memory_store()
    incident, workflow = stored_workflow(
        store,
        [WorkflowOperation.FREEZE_EVIDENCE],
        workflow_updates={
            "status": status,
            "execution_owner_id": None,
            "execution_lease_expires_at": None,
        },
    )
    with pytest.raises(ValueError, match="not executable"):
        IncidentOrchestrator(store).simulate(
            workflow.request_id, workflow.fencing_token
        )
    assert store.get_workflow(workflow.request_id) == workflow
    assert store.get_incident(incident.incident_id) == incident


def test_replayed_event_without_a_workflow_keeps_its_recorded_no_action_outcome():
    store = memory_store()
    event = xid()
    recorded = fault_incident(
        "inc-no-action",
        event.event_id,
        node_ids=["node-0"],
        state=IncidentState.RECOVERED,
    )
    store.save_incident(recorded)
    incident, workflow = IncidentOrchestrator(store).ingest(
        event, policy_decision(event)
    )
    assert incident == recorded
    assert workflow is None
    assert store.list_workflows() == []


@pytest.mark.parametrize("profile", [None, "unit-unreleased-profile"])
def test_fault_ingestion_cannot_emit_an_executable_reset_without_a_runtime_profile(
    profile,
):
    store = memory_store()
    event = xid(runtime_profile_version=profile)
    incident, workflow = IncidentOrchestrator(store).ingest(
        event, policy_decision(event)
    )
    assert workflow is not None, "a blocked reset lost its operator-visible record"
    assert workflow.status is WorkflowStatus.BLOCKED
    assert workflow.blocked_kind is BlockedKind.NEEDS_OPERATOR
    assert incident.event_type == "GPU_FAULT_GROUP"
    assert incident.state is IncidentState.SAFETY_PENDING
    assert any(
        "runtime" in reason.lower() and "profile" in reason.lower()
        for reason in workflow.blocked_reasons
    ), "the missing runtime-profile proof was not reported"
    assert (
        store.get_incident_by_event(event.event_id).incident_id == incident.incident_id
    )
    assert store.list_active_workflow_incidents("cluster-a") == [(incident, workflow)]
    assert (
        store.list_workflows(
            set(EXECUTABLE_WORKFLOW_STATUSES),
            dispatchable_at=workflow.updated_at + timedelta(days=1),
        )
        == []
    )

"""Passive recovery planner (FINAL-建议汇总 F-G5).

The action-to-operation map and the spare-blocking set are derived from one
table; the capability gate fails closed and does not depend on claim order;
auxiliary steps a profile cannot execute are dropped instead of rejecting the
whole plan; quick triage scopes each failed node to its own GPUs and action;
``after_incident`` no longer freezes execution-time decisions at plan time;
``avoid_node_ids`` reaches the compiled RESTART_WORKLOAD step.
"""

from __future__ import annotations

from datetime import timedelta

import pytest

from gpu_fault.app import default_simulated_profile
from gpu_fault.markers import SPARE_BLOCKING_ACTIONS
from gpu_fault.models import (
    CapabilityMode,
    CapabilityName,
    EffectiveCapability,
    IncidentState,
    MarkerScope,
    NodeMarker,
    RecoveryAction,
    RecoveryPlan,
    TerminalEvent,
    WorkflowOperation,
)
from gpu_fault.passive import ACTION_OPERATION, PassiveWorkflowCompiler
from gpu_fault.planner import PlanBuilder, UnsupportedPlanError
from gpu_fault.recovery_actions import RECOVERY_ACTION_PROFILES
from tests._builders import build_store, copy_model, fault_incident


def test_operation_map_and_spare_blocking_set_come_from_one_table() -> None:
    assert set(RECOVERY_ACTION_PROFILES) == set(RecoveryAction)
    assert ACTION_OPERATION == {
        action: profile.operation
        for action, profile in RECOVERY_ACTION_PROFILES.items()
        if profile.operation is not None
    }
    assert SPARE_BLOCKING_ACTIONS == frozenset(
        action
        for action, profile in RECOVERY_ACTION_PROFILES.items()
        if profile.blocks_spare
    )


def test_every_node_remediation_blocks_spare_reuse() -> None:
    """The three EFA/device-plugin actions were missing from both places.

    A marker asking for an EFA driver remediation names a node whose fabric
    is not usable, yet that node was handed out as a healthy warm spare.
    """
    assert SPARE_BLOCKING_ACTIONS == frozenset(
        {
            RecoveryAction.QUARANTINE,
            RecoveryAction.REPLACE_NODE,
            RecoveryAction.REBOOT_NODE,
            RecoveryAction.RESET_GPU,
            RecoveryAction.DRAIN,
            RecoveryAction.MARK_UNSCHEDULABLE,
            RecoveryAction.ESCALATE_OPERATOR,
            RecoveryAction.REMEDIATE_EFA_DRIVER,
            RecoveryAction.RESTART_EFA_DEVICE_PLUGIN,
            RecoveryAction.RESTART_GPU_DEVICE_PLUGIN,
        }
    )


def _profile(*claims: tuple[CapabilityName, CapabilityMode, str]):
    base = default_simulated_profile()
    return copy_model(
        base,
        capabilities=[
            EffectiveCapability(capability=name, mode=mode, owner=owner)
            for name, mode, owner in claims
        ],
    )


def _marker(failed_event: TerminalEvent, action: RecoveryAction) -> NodeMarker:
    return NodeMarker(
        marker_id=f"marker-{action.value.lower()}",
        source="test",
        trusted=True,
        incident_id=f"incident-{action.value.lower()}",
        observed_at=failed_event.ended_at,
        expires_at=failed_event.ended_at + timedelta(hours=1),
        scope=MarkerScope(node_ids=["node-a"], gpu_uuids=["GPU-a"]),
        severity="critical",
        recommended_action=action,
        mapping_version="test",
    )


@pytest.mark.parametrize("reverse", [False, True])
def test_capability_gate_ignores_claim_order(reverse: bool) -> None:
    claims = [
        (CapabilityName.NODE_REBOOT, CapabilityMode.AUGMENT, "observer"),
        (CapabilityName.NODE_REBOOT, CapabilityMode.OWN, "rebooter"),
    ]
    if reverse:
        claims.reverse()

    owner = PlanBuilder()._owner(_profile(*claims), RecoveryAction.REBOOT_NODE)

    assert owner == "rebooter"


def test_capability_gate_fails_closed_on_an_explicit_disable() -> None:
    profile = _profile(
        (CapabilityName.NODE_REBOOT, CapabilityMode.OWN, "rebooter"),
        (CapabilityName.NODE_REBOOT, CapabilityMode.DISABLED, "site-policy"),
    )

    with pytest.raises(UnsupportedPlanError, match="disabled"):
        PlanBuilder()._owner(profile, RecoveryAction.REBOOT_NODE)


def test_capability_gate_fails_closed_on_conflicting_owners() -> None:
    profile = _profile(
        (CapabilityName.NODE_REBOOT, CapabilityMode.OWN, "rebooter-a"),
        (CapabilityName.NODE_REBOOT, CapabilityMode.DELEGATE, "rebooter-b"),
    )

    with pytest.raises(UnsupportedPlanError, match="ambiguous"):
        PlanBuilder()._owner(profile, RecoveryAction.REBOOT_NODE)


def test_missing_evidence_capability_drops_the_evidence_step_not_the_plan(
    failed_event: TerminalEvent,
) -> None:
    profile = default_simulated_profile()
    profile = copy_model(
        profile,
        capabilities=[
            item
            for item in profile.capabilities
            if item.capability is not CapabilityName.EVIDENCE_CAPTURE
        ],
    )

    plan = PlanBuilder().from_marker(
        failed_event, _marker(failed_event, RecoveryAction.REBOOT_NODE), profile
    )

    assert [step.action for step in plan.steps] == [
        RecoveryAction.MARK_UNSCHEDULABLE,
        RecoveryAction.REBOOT_NODE,
        RecoveryAction.VALIDATE_NODE,
        RecoveryAction.RESTORE_SCHEDULING,
    ]


def test_missing_workload_stop_falls_back_to_containment(
    failed_event: TerminalEvent,
) -> None:
    """A destructive action must not run under a workload nobody can stop."""
    profile = default_simulated_profile()
    profile = copy_model(
        profile,
        capabilities=[
            item
            for item in profile.capabilities
            if item.capability is not CapabilityName.WORKLOAD_STOP
        ],
    )
    event = copy_model(failed_event, workload_ids=["training/pytorchjob/train"])

    plan = PlanBuilder().from_marker(
        event, _marker(event, RecoveryAction.REBOOT_NODE), profile
    )

    assert [step.action for step in plan.steps] == [
        RecoveryAction.MARK_UNSCHEDULABLE,
        RecoveryAction.COLLECT_EVIDENCE,
        RecoveryAction.QUARANTINE,
        RecoveryAction.ESCALATE_OPERATOR,
    ]
    assert plan.steps[-1].parameters["blocked_action"] == "REBOOT_NODE"


def test_after_incident_defers_avoidance_and_reuse_to_execution(
    failed_event: TerminalEvent,
) -> None:
    """Freezing ``incident.state`` at plan time avoided the node it repaired."""
    incident = fault_incident(
        "inc-pending",
        "event-pending",
        node_ids=["node-a"],
        state=IncidentState.ACTION_PENDING,
    )

    plan = PlanBuilder().after_incident(
        failed_event, incident, default_simulated_profile()
    )

    assert plan.avoid_node_ids == []
    step = plan.steps[0]
    assert step.action is RecoveryAction.RESTART_WORKLOAD
    assert step.parameters["reuse_allocation"] is True
    assert step.parameters["requires_incident_state"] == "RECOVERED"
    assert step.parameters["incident_id"] == "inc-pending"
    assert step.parameters["incident_node_ids"] == ["node-a"]


def test_avoid_node_ids_reach_the_compiled_restart_step(
    failed_event: TerminalEvent,
) -> None:
    store = build_store()
    event = copy_model(failed_event, workload_ids=["training/job/train"])

    def plan(avoid: list[str]) -> RecoveryPlan:
        return RecoveryPlan(
            incident_id=f"incident-avoid-{len(avoid)}",
            attempt_id=event.attempt_id,
            trigger="quick-triage:INCONCLUSIVE",
            runtime_profile_version="simulated-v1",
            avoid_node_ids=avoid,
            steps=[
                {
                    "action": RecoveryAction.RESTART_WORKLOAD,
                    "node_ids": ["node-b"],
                    "execution_owner": "simulated-runtime",
                }
            ],
        )

    avoiding = PassiveWorkflowCompiler(store).compile(plan(["node-a"]), event)
    plain = PassiveWorkflowCompiler(store).compile(plan([]), event)

    avoiding_step = store.get_workflow(avoiding.workflow_request_id).official_steps[0]
    plain_step = store.get_workflow(plain.workflow_request_id).official_steps[0]
    assert avoiding_step.operation is WorkflowOperation.RESTART_WORKLOAD
    assert avoiding_step.parameters["avoid_node_ids"] == ["node-a"]
    assert "avoid_node_ids" not in plain_step.parameters

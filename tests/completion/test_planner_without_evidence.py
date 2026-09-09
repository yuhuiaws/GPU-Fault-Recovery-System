from __future__ import annotations

from gpu_fault.app import default_simulated_profile
from gpu_fault.models import (
    CapabilityClaim,
    CapabilityMode,
    CapabilityName,
    RecoveryAction,
    TerminalEvent,
)
from gpu_fault.planner import PlanBuilder
from tests._builders import copy_model


def test_no_evidence_plans_one_restart_on_the_same_allocation(
    failed_event: TerminalEvent,
) -> None:
    plan = PlanBuilder().without_hardware_evidence(
        failed_event, default_simulated_profile()
    )

    assert plan.trigger == "no-hardware-evidence:RESTART"
    assert [step.action for step in plan.steps] == [RecoveryAction.RESTART_WORKLOAD]
    assert plan.steps[0].node_ids == ["node-a", "node-b"]
    assert plan.steps[0].parameters == {}
    assert plan.avoid_node_ids == []
    assert plan.restart_after_incident_id is None
    assert plan.checkpoint_manifest_ref == failed_event.checkpoint_manifest_ref


def test_no_evidence_without_restart_capability_escalates(
    failed_event: TerminalEvent,
) -> None:
    profile = default_simulated_profile()
    profile = copy_model(
        profile,
        capabilities=[
            claim
            for claim in profile.capabilities
            if claim.capability is not CapabilityName.WORKLOAD_RESTART
        ]
        + [
            CapabilityClaim(
                capability=CapabilityName.WORKLOAD_RESTART,
                mode=CapabilityMode.DISABLED,
                owner="nobody",
            )
        ],
    )

    plan = PlanBuilder().without_hardware_evidence(failed_event, profile)

    assert plan.trigger == "no-hardware-evidence:ESCALATE"
    assert [step.action for step in plan.steps] == [
        RecoveryAction.COLLECT_EVIDENCE,
        RecoveryAction.ESCALATE_OPERATOR,
    ]
    assert plan.steps[1].parameters["automatic_restart_blocked"] is True
    assert "WORKLOAD_RESTART" in plan.steps[1].parameters["reason"]

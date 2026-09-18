from __future__ import annotations

from copy import deepcopy

import pytest

from gpu_fault.models import RecoveryAction, WorkflowOperation, WorkflowStepStatus
from gpu_fault.orchestration.escalation import HardwareEscalationService
from tests.orchestration._cov95_runtime_escalation import EscalationHarness

DCGM = WorkflowOperation.RUN_DCGM_DIAGNOSTIC


@pytest.mark.parametrize(
    "defect",
    [
        "none",
        "missing-results",
        "missing-node",
        "scalar-result",
        "missing-recommendations",
        "non-list-recommendations",
        "unknown-step",
        "diagnostic_findings",
        "failed_checks",
        "warning_checks",
    ],
)
def test_active_workload_diagnostic_review_requires_complete_per_node_evidence(defect):
    node_results = {
        "node-a": {"recommended_actions": [{"action_code": "DCGM_EXECUTION_REVIEW"}]}
    }
    if defect == "missing-results":
        node_results = {}
    elif defect == "missing-node":
        node_results = {"node-b": node_results["node-a"]}
    elif defect == "scalar-result":
        node_results = {"node-a": "incomplete"}
    elif defect == "missing-recommendations":
        node_results["node-a"]["recommended_actions"] = None
    elif defect == "non-list-recommendations":
        node_results["node-a"]["recommended_actions"] = {
            "action_code": "DCGM_EXECUTION_REVIEW"
        }
    elif defect in {"diagnostic_findings", "failed_checks", "warning_checks"}:
        node_results["node-a"][defect] = ["memory failure"]
    h = EscalationHarness([DCGM], details={"node_results": node_results})
    h.amend(
        official_steps=[
            h.workflow.official_steps[0].model_copy(
                update={"workload_ids": ["training/job/job-a"]}
            )
        ]
    )
    if defect == "unknown-step":
        h.amend(
            step_executions=[
                h.workflow.step_executions[0].model_copy(update={"step_index": 10})
            ]
        )
    before = deepcopy(h.workflow)
    classification = HardwareEscalationService.classify(h.workflow)
    if defect == "none":
        assert classification is None, (
            "a complete execution-review-only result must not trigger hardware recovery"
        )
    else:
        assert classification is not None, (
            "missing or contradictory diagnostic evidence must not suppress the handoff",
            defect,
        )
        assert classification[:3] == (
            "temperature_diagnostic",
            RecoveryAction.DRAIN,
            None,
        ), classification
        assert classification[3] == h.workflow.step_executions, classification
    assert h.workflow == before, "classification must leave the durable failure intact"


def test_successful_diagnostic_does_not_create_a_hardware_escalation():
    h = EscalationHarness([DCGM])
    h.amend(
        official_steps=[
            h.workflow.official_steps[0].model_copy(
                update={"workload_ids": ["training/job/job-a"]}
            )
        ],
        step_executions=[
            h.workflow.step_executions[0].model_copy(
                update={"status": WorkflowStepStatus.SUCCEEDED, "error": None}
            )
        ],
    )
    assert HardwareEscalationService.classify(h.workflow) is None, h.workflow


@pytest.mark.parametrize("mapping", [[], {"node-a": "GPU-unbound"}])
def test_incomplete_step_scope_preserves_incident_nodes_without_guessing_gpu_mapping(
    mapping,
):
    h = EscalationHarness([WorkflowOperation.RESTART_NODE])
    h.amend(
        official_steps=[
            h.workflow.official_steps[0].model_copy(
                update={
                    "node_ids": [],
                    "gpu_uuids": ["GPU-b"],
                    "parameters": {"gpu_uuids_by_node": mapping},
                }
            )
        ]
    )
    scope = HardwareEscalationService.collect_scope(
        h.workflow, h.source, h.workflow.step_executions
    )
    assert scope["ordered_failed_nodes"] == ["node-a", "node-b"], scope
    assert scope["gpu_uuids"] == ["GPU-a", "GPU-b"], scope
    assert scope["gpu_uuids_by_node"] == {}, scope
    assert scope["node_reason"] == (
        "node-a: validation failed; node-b: validation failed"
    ), scope


def test_diagnostic_guidance_without_evidence_uri_does_not_fabricate_a_reference():
    h = EscalationHarness(
        [DCGM],
        details={
            "node_results": {
                "node-a": {
                    "recommended_actions": [
                        {
                            "action_code": "THERMAL_REVIEW",
                            "instruction": "Inspect cooling",
                            "trigger_tests": [],
                        }
                    ]
                }
            }
        },
    )
    scope = HardwareEscalationService.collect_scope(
        h.workflow, h.source, h.workflow.step_executions
    )
    assert scope["diagnostic_evidence"] == [], scope
    assert scope["diagnostic_guidance"] == [
        "DCGM guidance for node-a [THERMAL_REVIEW] (): Inspect cooling"
    ], scope

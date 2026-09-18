from __future__ import annotations

from copy import deepcopy

import pytest

from gpu_fault.models import (
    BlockedKind,
    RecoveryAction,
    WorkflowOperation,
    WorkflowStatus,
)
from gpu_fault.orchestration.escalation import HardwareEscalationService
from tests._builders import workflow_step
from tests.orchestration._cov95_runtime_escalation import (
    RESTART_CONTEXT,
    EscalationHarness,
)

RESET = WorkflowOperation.RESET_GPU
REBOOT = WorkflowOperation.RESTART_NODE
REPLACE = WorkflowOperation.REPLACE_NODE
VALIDATE = WorkflowOperation.VALIDATE_GPU
RESTART = WorkflowOperation.RESTART_WORKLOAD
DCGM = WorkflowOperation.RUN_DCGM_DIAGNOSTIC
SUPPORT = WorkflowOperation.ESCALATE_SUPPORT


@pytest.mark.parametrize(
    ("operations", "expected_action", "expected_operation"),
    [
        ([RESET], RecoveryAction.REBOOT_NODE, REBOOT),
        ([REBOOT], RecoveryAction.REPLACE_NODE, REPLACE),
        ([REPLACE], RecoveryAction.ESCALATE_OPERATOR, SUPPORT),
        ([DCGM], RecoveryAction.DRAIN, SUPPORT),
        (
            [WorkflowOperation.RUN_FIELD_DIAGNOSTIC],
            RecoveryAction.ESCALATE_OPERATOR,
            SUPPORT,
        ),
        (
            [WorkflowOperation.RUN_NVLINK74_WORKFLOW],
            RecoveryAction.ESCALATE_OPERATOR,
            SUPPORT,
        ),
        (
            [WorkflowOperation.REMEDIATE_DRIVER],
            RecoveryAction.ESCALATE_OPERATOR,
            SUPPORT,
        ),
        (
            [WorkflowOperation.UPDATE_SOFTWARE_FIRMWARE],
            RecoveryAction.ESCALATE_OPERATOR,
            SUPPORT,
        ),
        ([REPLACE, VALIDATE], RecoveryAction.ESCALATE_OPERATOR, SUPPORT),
        (
            [WorkflowOperation.REMEDIATE_DRIVER, VALIDATE],
            RecoveryAction.ESCALATE_OPERATOR,
            SUPPORT,
        ),
        (
            [WorkflowOperation.RUN_FIELD_DIAGNOSTIC, VALIDATE],
            RecoveryAction.ESCALATE_OPERATOR,
            SUPPORT,
        ),
        ([DCGM, VALIDATE], RecoveryAction.DRAIN, SUPPORT),
        ([RESET, VALIDATE], RecoveryAction.REBOOT_NODE, REBOOT),
    ],
)
def test_failed_recovery_emits_the_reviewed_rung_with_real_profile_compilation(
    operations, expected_action, expected_operation
) -> None:
    h = EscalationHarness(operations)
    result = h.escalate()
    assert result is not None, "classifiable recovery failure needs a durable follow-up"
    incident, successor = result
    assert incident.effective_action is expected_action, incident
    assert successor.status is WorkflowStatus.PENDING, successor.blocked_reasons
    emitted = [step.operation for step in successor.official_steps]
    assert expected_operation in emitted, emitted
    assert (incident.cluster_id, incident.job_id, incident.attempt_id) == (
        h.source.cluster_id,
        h.source.job_id,
        h.source.attempt_id,
    ), incident
    assert incident.node_ids == ["node-a"], incident
    if expected_operation is REPLACE:
        replacement = next(
            step for step in successor.official_steps if step.operation is REPLACE
        )
        assert (
            replacement.parameters["replacement_strategy"] == "HEALTHY_WARM_SPARE_ONLY"
        ), replacement
    repeated = h.escalate()
    assert repeated == result, (
        "reconciliation must reuse one escalation incident and workflow"
    )
    assert len(h.store.list_workflows()) == 2, (
        "repeated failure must not create another rung"
    )


@pytest.mark.parametrize("profile", [None, "missing-profile"])
def test_escalation_with_unknown_runtime_profile_is_blocked_for_an_operator(
    profile,
) -> None:
    h = EscalationHarness([REBOOT])
    h.amend(runtime_profile_version=profile)
    result = h.escalate()
    assert result is not None, "profile refusal still needs an operator-visible record"
    _, successor = result
    assert successor.status is WorkflowStatus.BLOCKED, successor
    assert successor.blocked_kind is BlockedKind.NEEDS_OPERATOR, successor
    assert any(
        "runtime profile" in reason or "runtime_profile_version" in reason
        for reason in successor.blocked_reasons
    ), successor


@pytest.mark.parametrize(
    "context_kind", ["absent", "partial", "inconsistent", "duplicate"]
)
def test_replacement_keeps_a_single_consistent_restart_authorization_context(
    context_kind: str,
) -> None:
    h = EscalationHarness([REBOOT])
    steps = [
        h.workflow.official_steps[0].model_copy(
            update={"workload_ids": ["training/job/job-a"]}
        )
    ]
    if context_kind != "absent":
        parameters = dict(RESTART_CONTEXT)
        if context_kind == "partial":
            parameters.pop("source_gpu_count")
        steps.append(
            workflow_step(
                RESTART,
                node_ids=["node-a", "node-b"],
                workload_ids=["training/job/job-a"],
                parameters=parameters,
            )
        )
        if context_kind in {"duplicate", "inconsistent"}:
            steps.append(
                steps[-1].model_copy(
                    update={
                        "parameters": {
                            **parameters,
                            "source_attempt_id": (
                                "other-attempt"
                                if context_kind == "inconsistent"
                                else "attempt-a"
                            ),
                        }
                    }
                )
            )
    h.amend(official_steps=steps)
    result = h.escalate()
    assert result is not None, (
        "failed reboot must produce a bounded replacement decision"
    )
    _, successor = result
    restart = next(
        step for step in successor.official_steps if step.operation is RESTART
    )
    if context_kind == "duplicate":
        assert successor.status is WorkflowStatus.PENDING, successor.blocked_reasons
        assert all(
            restart.parameters[key] == value for key, value in RESTART_CONTEXT.items()
        ), restart
        assert restart.node_ids == ["node-a", "node-b"], restart
    else:
        assert successor.status is WorkflowStatus.BLOCKED, successor
        assert any(
            "restart safety context" in reason for reason in successor.blocked_reasons
        ), successor
        assert "source_attempt_id" not in restart.parameters, restart
    replacement = next(
        step for step in successor.official_steps if step.operation is REPLACE
    )
    assert (
        replacement.parameters["replacement_strategy"] == "HEALTHY_WARM_SPARE_ONLY"
    ), replacement


def test_escalated_validation_inherits_only_the_failed_nodes_inventory_requirements() -> (
    None
):
    h = EscalationHarness([REBOOT, VALIDATE], failed_index=0)
    requirements = {
        "node-a": {"gpu_uuids": ["GPU-a"]},
        "node-b": {"gpu_uuids": ["GPU-b"]},
    }
    h.amend(
        official_steps=[
            h.workflow.official_steps[0],
            h.workflow.official_steps[1].model_copy(
                update={"parameters": {"inventory_requirements_by_node": requirements}}
            ),
        ]
    )
    result = h.escalate()
    assert result is not None, "failed reboot must retain validation context"
    _, successor = result
    validation = next(
        step for step in successor.official_steps if step.operation is VALIDATE
    )
    assert validation.parameters["inventory_requirements_by_node"] == {
        "node-a": requirements["node-a"]
    }, validation
    assert validation.node_ids == ["node-a"], validation


@pytest.mark.parametrize("reported", [[], ["foreign-node"], ["node-a", "foreign-node"]])
def test_failure_report_cannot_expand_escalation_beyond_the_failed_step(
    reported,
) -> None:
    h = EscalationHarness(
        [REBOOT],
        details={
            "failed_nodes": reported,
            "node_failures": {"node-a": "provider did not recover"},
        },
    )
    result = h.escalate()
    assert result is not None, (
        "an incomplete error report must not lose its failed step"
    )
    incident, successor = result
    assert incident.node_ids == ["node-a"], incident
    assert all(step.node_ids == ["node-a"] for step in successor.official_steps), (
        successor
    )
    assert "node-a: provider did not recover" in incident.reasons[0], incident.reasons


@pytest.mark.parametrize(
    "defect",
    [
        "none",
        "node-results-list",
        "node-result-scalar",
        "recommendations-mapping",
        "recommendation-scalar",
        "empty-code",
        "empty-instruction",
        "unknown-tests",
    ],
)
def test_diagnostic_guidance_is_scoped_and_malformed_rows_never_become_instructions(
    defect: str,
) -> None:
    recommendation = {
        "action_code": "THERMAL_REVIEW",
        "instruction": "Inspect cooling",
        "trigger_tests": ["memory"],
    }
    node_results = {
        "node-a": {
            "evidence_ref": "evidence://local-dcgm",
            "recommended_actions": [recommendation],
        },
        "foreign-node": {
            "evidence_ref": "evidence://foreign",
            "recommended_actions": [recommendation],
        },
    }
    if defect == "node-results-list":
        node_results = []
    elif defect == "node-result-scalar":
        node_results["node-a"] = "invalid"
    elif defect == "recommendations-mapping":
        node_results["node-a"]["recommended_actions"] = {}
    elif defect == "recommendation-scalar":
        node_results["node-a"]["recommended_actions"] = ["invalid"]
    elif defect == "empty-code":
        recommendation["action_code"] = ""
    elif defect == "empty-instruction":
        recommendation["instruction"] = ""
    elif defect == "unknown-tests":
        recommendation["trigger_tests"] = "not-a-list"
    h = EscalationHarness([DCGM], details={"node_results": node_results})
    original = deepcopy(h.workflow)
    scope = HardwareEscalationService.collect_scope(
        h.workflow, h.source, h.workflow.step_executions
    )
    assert scope["ordered_failed_nodes"] == ["node-a"], scope
    expected_evidence = (
        []
        if defect in {"node-results-list", "node-result-scalar"}
        else ["DCGM evidence for node-a: evidence://local-dcgm"]
    )
    assert scope["diagnostic_evidence"] == expected_evidence, scope
    expected_guidance = (
        [
            f"DCGM guidance for node-a [THERMAL_REVIEW] ({'unknown' if defect == 'unknown-tests' else 'memory'}): Inspect cooling"
        ]
        if defect in {"none", "unknown-tests"}
        else []
    )
    assert scope["diagnostic_guidance"] == expected_guidance, scope
    assert h.workflow == original, (
        "evidence interpretation must not mutate the stored failure"
    )

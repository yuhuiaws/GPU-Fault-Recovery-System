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


FREEZE = WorkflowOperation.FREEZE_EVIDENCE
CORDON = WorkflowOperation.MARK_UNSCHEDULABLE
STOP = WorkflowOperation.STOP_WORKLOADS
QUARANTINE = WorkflowOperation.QUARANTINE
BUNDLE = WorkflowOperation.COLLECT_DIAGNOSTIC_BUNDLE
DRAIN_CHAIN = [FREEZE, CORDON, QUARANTINE, BUNDLE, VALIDATE, SUPPORT]


@pytest.mark.parametrize(
    "operations",
    [DRAIN_CHAIN, [FREEZE, CORDON, STOP, QUARANTINE, BUNDLE, VALIDATE, SUPPORT]],
    ids=["drain", "drain-with-stop"],
)
def test_a_failed_validation_of_a_drain_hands_the_held_node_to_support(
    operations,
) -> None:
    """A drain (the node-health DRAIN chain, the ladder's drain-after successor)
    holds its node and hands it over with the ESCALATE_SUPPORT behind its
    VALIDATE_GPU. With no recovery action before the validation the ladder has
    no rung, but the hand-off is still owed: support, not a silent QUARANTINED."""

    h = EscalationHarness(operations, failed_index=operations.index(VALIDATE))

    classification = HardwareEscalationService.classify(h.workflow)
    assert classification is not None, (
        "a failed drain validation left the held node with nobody told"
    )
    assert classification[:3] == (
        "drain_validation",
        RecoveryAction.ESCALATE_OPERATOR,
        SUPPORT,
    )
    result = h.escalate()
    assert result is not None, "the failed drain needs a durable hand-off"
    incident, successor = result
    assert incident.reasons[0].startswith(
        "drain_validation remediation failed; failed operation or validation: "
        "VALIDATE_GPU"
    ), incident.reasons
    assert [step.operation for step in successor.official_steps] == [
        FREEZE,
        CORDON,
        QUARANTINE,
        SUPPORT,
    ], "the hold is kept and handed over; nothing stops, resets or releases"
    assert successor.lifetime_deadline_at is None, (
        "an operator hand-off starts its own clock"
    )


@pytest.mark.parametrize(
    ("operations", "why"),
    [
        (
            [FREEZE, WorkflowOperation.VALIDATE_HOST, SUPPORT],
            "a read-only chain held nothing: its failure is inconclusive",
        ),
        (
            [FREEZE, CORDON, VALIDATE, WorkflowOperation.RESTORE_SCHEDULING],
            "no hand-off was planned behind the validation",
        ),
    ],
    ids=["read-only", "no-planned-hand-off"],
)
def test_a_failed_validation_without_a_held_hand_off_climbs_nothing(
    operations, why
) -> None:
    h = EscalationHarness(operations, failed_index=1 if len(operations) == 3 else 2)

    assert HardwareEscalationService.classify(h.workflow) is None, why


def test_a_drain_whose_own_hand_off_failed_opens_no_second_one() -> None:
    """The rule is for a hand-off that never ran. A drain whose ESCALATE_SUPPORT
    itself failed keeps today's ending: no second ticket from this rule."""

    h = EscalationHarness(DRAIN_CHAIN)

    assert HardwareEscalationService.classify(h.workflow) is None


SNAPSHOT = WorkflowOperation.TRIGGER_HEALTH_SNAPSHOT
CHECKPOINT = WorkflowOperation.CHECKPOINT_WORKLOADS
DRAIN_STOP_CHAIN = [FREEZE, CORDON, STOP, QUARANTINE, BUNDLE, VALIDATE, SUPPORT]
#: The confirmed thermal DRAIN's chain; every step carries the contract.
ENV_REVIEW_CHAIN = [FREEZE, CORDON, STOP, QUARANTINE, BUNDLE, SNAPSHOT, SUPPORT]
#: The XID/SXID ESCALATE_OPERATOR chain of an active job with a checkpoint.
OPERATOR_CHAIN = [FREEZE, CORDON, CHECKPOINT, STOP, QUARANTINE, SUPPORT]
ENV_REVIEW = {
    "health_recovery_contract": {
        "mode": "ENVIRONMENT_REVIEW",
        "metric": "gpu_temperature_c",
        "detected_at": "2026-01-01T00:00:00+00:00",
        "node_id": "node-a",
    }
}
UNKNOWN_OUTCOME = {"outcome_unknown": True, "manual_confirmation_required": True}
REJECTED = {"remote_status_source": "executor-rejected"}


def held_chain(
    operations, failed_index: int, details=None, contract=None
) -> EscalationHarness:
    h = EscalationHarness(operations, failed_index=failed_index, details=details)
    if contract is not None:
        h.amend(
            official_steps=[
                step.model_copy(update={"parameters": {**step.parameters, **contract}})
                for step in h.workflow.official_steps
            ]
        )
    return h


@pytest.mark.parametrize(
    ("operations", "failed", "contract"),
    [
        (DRAIN_STOP_CHAIN, STOP, None),
        (DRAIN_CHAIN, BUNDLE, None),
        (ENV_REVIEW_CHAIN, STOP, ENV_REVIEW),
        (ENV_REVIEW_CHAIN, QUARANTINE, ENV_REVIEW),
        (ENV_REVIEW_CHAIN, BUNDLE, ENV_REVIEW),
        (ENV_REVIEW_CHAIN, SNAPSHOT, ENV_REVIEW),
        (OPERATOR_CHAIN, CHECKPOINT, None),
        (OPERATOR_CHAIN, STOP, None),
    ],
    ids=[
        "drain-stop",
        "drain-bundle",
        "environment-review-stop",
        "environment-review-quarantine",
        "environment-review-bundle",
        "environment-review-snapshot",
        "operator-checkpoint-answered",
        "operator-stop",
    ],
)
def test_a_held_chain_step_failure_hands_the_node_to_support(
    operations, failed, contract
) -> None:
    """A chain that holds its node plans its ESCALATE_SUPPORT last; a step
    before it that fails where the ladder has no rung -- the bundle, the
    snapshot, the stop, a checkpoint that answered -- still owes that hand-off
    (the ENVIRONMENT_REVIEW contract keeps hardware rungs away, not the
    operator). Before, every one ended QUARANTINED with nobody told."""

    h = held_chain(operations, operations.index(failed), contract=contract)

    classification = HardwareEscalationService.classify(h.workflow)
    assert classification is not None, (
        f"{failed.value} failed in a held chain and nobody was told"
    )
    assert classification[:3] == (
        "held_chain_step",
        RecoveryAction.ESCALATE_OPERATOR,
        SUPPORT,
    )
    result = h.escalate()
    assert result is not None, "the held node needs a durable hand-off"
    incident, successor = result
    assert incident.reasons[0].startswith(
        "held_chain_step remediation failed; failed operation or validation: "
        f"{failed.value}"
    ), incident.reasons
    assert [step.operation for step in successor.official_steps] == [
        FREEZE,
        CORDON,
        QUARANTINE,
        SUPPORT,
    ], "the hold is kept and handed over; nothing stops, resets or releases"
    assert h.escalate() == result, "one support successor per failed chain"
    assert len(h.store.list_workflows()) == 2


@pytest.mark.parametrize(
    ("operations", "failed_index", "details", "contract", "why"),
    [
        (
            DRAIN_CHAIN,
            DRAIN_CHAIN.index(SUPPORT),
            UNKNOWN_OUTCOME,
            None,
            "an abandoned support escalation: a second vendor ticket",
        ),
        (
            ENV_REVIEW_CHAIN,
            ENV_REVIEW_CHAIN.index(SUPPORT),
            UNKNOWN_OUTCOME,
            ENV_REVIEW,
            "an abandoned support escalation: a second vendor ticket",
        ),
        (
            OPERATOR_CHAIN,
            OPERATOR_CHAIN.index(CHECKPOINT),
            UNKNOWN_OUTCOME,
            None,
            "an abandoned checkpoint may still be running inside the job",
        ),
        (
            [FREEZE, CORDON, QUARANTINE, FREEZE, BUNDLE, VALIDATE, SUPPORT],
            3,
            UNKNOWN_OUTCOME,
            None,
            "an abandoned evidence freeze may still be running",
        ),
        (
            [FREEZE, CORDON, QUARANTINE, FREEZE, BUNDLE, SNAPSHOT, SUPPORT],
            3,
            UNKNOWN_OUTCOME,
            ENV_REVIEW,
            "an abandoned evidence freeze may still be running",
        ),
        (
            DRAIN_CHAIN,
            DRAIN_CHAIN.index(BUNDLE),
            REJECTED,
            None,
            "refused before dispatch: a stale generation, a newer one owns it",
        ),
        (
            ENV_REVIEW_CHAIN,
            ENV_REVIEW_CHAIN.index(SNAPSHOT),
            REJECTED,
            ENV_REVIEW,
            "refused before dispatch: a stale generation, a newer one owns it",
        ),
    ],
    ids=[
        "drain-abandoned-support",
        "environment-review-abandoned-support",
        "abandoned-checkpoint",
        "abandoned-freeze",
        "environment-review-abandoned-freeze",
        "drain-executor-rejected",
        "environment-review-executor-rejected",
    ],
)
def test_the_no_second_ticket_endings_still_open_nothing(
    operations, failed_index, details, contract, why
) -> None:
    h = held_chain(operations, failed_index, details=details, contract=contract)

    assert HardwareEscalationService.classify(h.workflow) is None, why
    assert h.escalate() is None, why
    assert len(h.store.list_workflows()) == 1, "no successor was opened"


#: The steps a job family compiles for the job's whole allocation
#: (``families/grouped_health.py``, ``grouped_faults.py``): a STOP or a
#: CHECKPOINT names every node of the job, the cordon only the faulty one.
JOB_WIDE = {STOP, CHECKPOINT}


@pytest.mark.parametrize(
    ("operations", "failed", "contract"),
    [
        (DRAIN_STOP_CHAIN, STOP, None),
        (ENV_REVIEW_CHAIN, STOP, ENV_REVIEW),
        (OPERATOR_CHAIN, STOP, None),
        (OPERATOR_CHAIN, CHECKPOINT, None),
    ],
    ids=[
        "drain-stop",
        "environment-review-stop",
        "operator-stop",
        "operator-checkpoint",
    ],
)
def test_the_hand_off_holds_only_the_nodes_the_chain_held(
    operations, failed, contract
) -> None:
    """A job-wide step that fails names the whole allocation; the hand-off
    keeps the chain's hold and must not cordon the job's healthy nodes."""

    h = held_chain(operations, operations.index(failed), contract=contract)
    h.amend(
        official_steps=[
            step.model_copy(update={"node_ids": ["node-a", "node-b"]})
            if step.operation in JOB_WIDE
            else step
            for step in h.workflow.official_steps
        ]
    )

    result = h.escalate()

    assert result is not None, "the held node still needs its hand-off"
    incident, successor = result
    assert incident.node_ids == ["node-a"], incident.node_ids
    assert {step.operation: step.node_ids for step in successor.official_steps} == {
        FREEZE: ["node-a"],
        CORDON: ["node-a"],
        QUARANTINE: ["node-a"],
        SUPPORT: ["node-a"],
    }, "node-b was never held by the chain: the hand-off does not cordon it"
    assert "node-b" not in incident.reasons[0], incident.reasons


def test_a_row_without_its_completed_executions_hands_over_its_completed_hold() -> None:
    """A record written before completions were recorded keeps only its FAILED
    execution; its completed isolation steps still name the held node."""

    h = held_chain(DRAIN_STOP_CHAIN, DRAIN_STOP_CHAIN.index(STOP))
    h.amend(
        step_executions=[
            item for item in h.workflow.step_executions if item.operation is STOP
        ],
        official_steps=[
            step.model_copy(update={"node_ids": ["node-a", "node-b"]})
            if step.operation in JOB_WIDE
            else step
            for step in h.workflow.official_steps
        ],
    )

    incident, successor = h.escalate()

    assert incident.node_ids == ["node-a"], incident.node_ids
    assert {step.node_ids[0] for step in successor.official_steps} == {"node-a"}


def test_a_failed_support_hand_off_is_terminated_not_handed_off_again() -> None:
    """A support-after successor that holds its node and failed before its own
    ESCALATE_SUPPORT is the chain terminator's: no support-after-support."""

    h = held_chain(
        [FREEZE, CORDON, QUARANTINE, SUPPORT], failed_index=2, details={"error": "x"}
    )
    h.store.save_incident(
        h.source.model_copy(update={"event_id": "support-after-earlier-workflow"})
    )

    assert h.escalate() is None
    assert len(h.store.list_workflows()) == 1, "the terminator opens no successor"


def test_a_cooldown_plan_without_a_hand_off_still_opens_nothing() -> None:
    """A legacy THERMAL_COOLDOWN plan cordons but plans no ESCALATE_SUPPORT:
    nothing is owed by this rule, whatever step failed."""

    cooldown = {"health_recovery_contract": {"mode": "THERMAL_COOLDOWN"}}
    operations = [FREEZE, CORDON, BUNDLE, SNAPSHOT, VALIDATE]
    h = held_chain(operations, operations.index(SNAPSHOT), contract=cooldown)

    assert HardwareEscalationService.classify(h.workflow) is None


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

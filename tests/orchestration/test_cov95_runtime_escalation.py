from __future__ import annotations

from copy import deepcopy

import pytest

from gpu_fault.adapters.gpu_validation import TELEMETRY_READINESS_OPERATIONS
from gpu_fault.models import (
    BlockedKind,
    RecoveryAction,
    WorkflowOperation,
    WorkflowStatus,
)
from gpu_fault.operation_registry import (
    NODE_HARDWARE_MUTATING_OPERATIONS,
    RECOVERY_OPERATION_RANK,
)
from gpu_fault.orchestration import validation_rungs
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
FM_RESTART = WorkflowOperation.RESTART_FABRIC_MANAGER
EFA_DRIVER = WorkflowOperation.REMEDIATE_EFA_DRIVER
EFA_PLUGIN = WorkflowOperation.RESTART_EFA_DEVICE_PLUGIN
GPU_PLUGIN = WorkflowOperation.RESTART_GPU_DEVICE_PLUGIN


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
        ([FM_RESTART, VALIDATE], RecoveryAction.REBOOT_NODE, REBOOT),
        ([EFA_DRIVER, VALIDATE], RecoveryAction.REBOOT_NODE, REBOOT),
        ([EFA_PLUGIN, VALIDATE], RecoveryAction.REBOOT_NODE, REBOOT),
        ([GPU_PLUGIN, VALIDATE], RecoveryAction.REBOOT_NODE, REBOOT),
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


#: The node recovery actions: the operations the registry calls
#: node-hardware-mutating and ranks as a recovery. Pinned: a recovery action
#: added to the registry (or one that loses its rank) fails here until the
#: escalation ladder says which rung a validation failing after it climbs to.
NODE_RECOVERY_ACTIONS = frozenset(
    {
        RESET,
        WorkflowOperation.RESET_ALL_GPUS_NVSWITCHES,
        WorkflowOperation.RECOVER_DRAIN_P2P,
        FM_RESTART,
        EFA_DRIVER,
        EFA_PLUGIN,
        GPU_PLUGIN,
        WorkflowOperation.REMEDIATE_DRIVER,
        WorkflowOperation.UPDATE_SOFTWARE_FIRMWARE,
        REBOOT,
        REPLACE,
    }
)
_REBOOT_RUNG = (RecoveryAction.REBOOT_NODE, REBOOT)
_SUPPORT_RUNG = (RecoveryAction.ESCALATE_OPERATOR, SUPPORT)
#: The rung a validation failing after each node recovery action climbs to:
#: reset -> reboot -> warm-spare replacement -> support; the installs go to
#: support (their software may be half updated), and the runtime restarts --
#: Fabric Manager, the EFA driver, the device plugins -- climb like a reset.
VALIDATION_RUNG_AFTER = {
    RESET: ("reset_validation", *_REBOOT_RUNG),
    WorkflowOperation.RESET_ALL_GPUS_NVSWITCHES: ("reset_validation", *_REBOOT_RUNG),
    WorkflowOperation.RECOVER_DRAIN_P2P: ("reset_validation", *_REBOOT_RUNG),
    FM_RESTART: ("fabric_manager_validation", *_REBOOT_RUNG),
    EFA_DRIVER: ("efa_recovery", *_REBOOT_RUNG),
    EFA_PLUGIN: ("efa_recovery", *_REBOOT_RUNG),
    GPU_PLUGIN: ("efa_recovery", *_REBOOT_RUNG),
    WorkflowOperation.REMEDIATE_DRIVER: ("remediation_validation", *_SUPPORT_RUNG),
    WorkflowOperation.UPDATE_SOFTWARE_FIRMWARE: (
        "remediation_validation",
        *_SUPPORT_RUNG,
    ),
    REBOOT: ("reboot_validation", RecoveryAction.REPLACE_NODE, REPLACE),
    REPLACE: ("replacement_validation", *_SUPPORT_RUNG),
}


def _node_recovery_actions() -> frozenset[WorkflowOperation]:
    return NODE_HARDWARE_MUTATING_OPERATIONS & frozenset(RECOVERY_OPERATION_RANK)


def test_every_node_recovery_action_is_a_recovery_step_before_a_validation() -> None:
    """The guard. The node recovery actions come from the registry -- the
    node-hardware-mutating operations that carry a recovery rank; the
    GPU-service window around a reset (QUIESCE / RESTORE_GPU_SERVICES) and the
    planning-only RESTART_VM carry none -- and the validation adapter re-reads
    telemetry after exactly the same set. Every one of them must be a recovery
    step the ladder judges a failed validation against: RESTART_FABRIC_MANAGER
    was not, so a validation failing after a Fabric Manager restart had no
    rung and the record ended silently."""

    derived = _node_recovery_actions()
    assert derived == NODE_RECOVERY_ACTIONS, (
        "the registry's recovery actions changed: classify the new one in "
        "validation_rungs.VALIDATION_RUNGS, then update this pin"
    )
    assert derived == TELEMETRY_READINESS_OPERATIONS, (
        "the validation adapter re-reads telemetry after exactly the recovery actions"
    )
    assert validation_rungs.NODE_RECOVERY_OPERATIONS == derived
    unjudged = derived - validation_rungs.VALIDATION_JUDGED_OPERATIONS
    assert not unjudged, "no rung for a validation failing after " + ", ".join(
        sorted(operation.value for operation in unjudged)
    )
    assert set(VALIDATION_RUNG_AFTER) == derived


@pytest.mark.parametrize("held", [True, False], ids=["held", "unheld"])
@pytest.mark.parametrize(
    "operation", sorted(_node_recovery_actions()), ids=lambda operation: operation.value
)
def test_a_validation_failing_after_a_recovery_action_climbs_its_rung(
    operation, held
) -> None:
    """A validation that fails after a node recovery action is a failed
    recovery, whether or not the plan cordoned first: it climbs the rung of
    that action. Before, a validation after a Fabric Manager restart -- or
    after the EFA driver or a device-plugin restart whose plan carried no
    reboot request -- climbed nothing."""

    operations = (
        [FREEZE, CORDON, operation, VALIDATE, WorkflowOperation.RESTORE_SCHEDULING]
        if held
        else [FREEZE, operation, VALIDATE]
    )
    h = EscalationHarness(
        operations,
        failed_index=operations.index(VALIDATE),
        details={"failed_nodes": ["node-a"], "node_failures": {"node-a": ["x"]}},
    )

    classification = HardwareEscalationService.classify(h.workflow)

    assert classification is not None, (
        f"a validation failed after {operation.value} and climbed nothing"
    )
    assert classification[:3] == VALIDATION_RUNG_AFTER[operation]


def test_a_validation_after_a_fabric_manager_restart_escalates_the_restarted_node() -> (
    None
):
    """Counted as the recovery step, the Fabric Manager restart also scopes
    the failed validation: the reboot rung acts on the node it restarted, not
    on a sibling the validation merely checked."""

    operations = [FREEZE, CORDON, FM_RESTART, VALIDATE]
    h = EscalationHarness(
        operations,
        failed_index=operations.index(VALIDATE),
        details={"failed_nodes": ["node-a", "node-b"]},
    )
    h.amend(
        official_steps=[
            step
            if step.operation is FM_RESTART
            else step.model_copy(update={"node_ids": ["node-a", "node-b"]})
            for step in h.workflow.official_steps
        ]
    )

    result = h.escalate()

    assert result is not None, "the failed validation needs its rung"
    incident, successor = result
    assert incident.effective_action is RecoveryAction.REBOOT_NODE
    assert incident.reasons[0].startswith(
        "fabric_manager_validation remediation failed; failed operation or "
        "validation: VALIDATE_GPU"
    ), incident.reasons
    assert incident.node_ids == ["node-a"], incident.node_ids
    assert {
        step.operation: step.node_ids
        for step in successor.official_steps
        if step.operation in {CORDON, REBOOT}
    } == {CORDON: ["node-a"], REBOOT: ["node-a"]}


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
        (ENV_REVIEW_CHAIN, BUNDLE, ENV_REVIEW),
        (ENV_REVIEW_CHAIN, SNAPSHOT, ENV_REVIEW),
        (OPERATOR_CHAIN, CHECKPOINT, None),
        (OPERATOR_CHAIN, STOP, None),
    ],
    ids=[
        "drain-stop",
        "drain-bundle",
        "environment-review-stop",
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


#: The node-health QUARANTINE plan of a BMC sensor, a host signal or a severe
#: kernel line (``families/health._operations``): it plans no hand-off and
#: ends with the node held -- its QUARANTINE carries the terminal hold.
QUARANTINE_PLAN = [FREEZE, CORDON, STOP, QUARANTINE]
TERMINAL = {"terminal_quarantine_node_ids": ["node-a"]}


def terminal_hold(h: EscalationHarness, *, job_wide: bool = False) -> None:
    """Give ``h``'s QUARANTINE its terminal hold (and its STOP the job's
    whole allocation)."""

    h.amend(
        official_steps=[
            step.model_copy(update={"parameters": {**step.parameters, **TERMINAL}})
            if step.operation is QUARANTINE
            else step.model_copy(update={"node_ids": ["node-a", "node-b"]})
            if job_wide and step.operation in JOB_WIDE
            else step
            for step in h.workflow.official_steps
        ]
    )


@pytest.mark.parametrize(
    ("explicit", "job_wide"),
    [(True, False), (False, False), (True, True)],
    ids=["terminal-quarantine", "legacy-no-readmission", "job-wide-stop"],
)
def test_a_quarantine_plan_whose_stop_fails_hands_the_held_node_to_support(
    explicit, job_wide
) -> None:
    """The QUARANTINE plan plans no ESCALATE_SUPPORT: its hand-off is the end
    state itself (QUARANTINED, the terminal notice). A STOP_WORKLOADS that
    fails after the cordon ended the record FAILED + QUARANTINED with no
    ticket and no notice; the held node is now handed to support."""

    h = held_chain(QUARANTINE_PLAN, QUARANTINE_PLAN.index(STOP))
    if explicit:
        terminal_hold(h, job_wide=job_wide)

    classification = HardwareEscalationService.classify(h.workflow)
    assert classification is not None, "the held node was left with nobody told"
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
        "STOP_WORKLOADS"
    ), incident.reasons
    assert incident.node_ids == ["node-a"], incident.node_ids
    assert [(step.operation, step.node_ids) for step in successor.official_steps] == [
        (FREEZE, ["node-a"]),
        (CORDON, ["node-a"]),
        (QUARANTINE, ["node-a"]),
        (SUPPORT, ["node-a"]),
    ], "the hold is kept and handed over: no stop again, no release, no node-b"
    assert h.escalate() == result, "one support successor per failed plan"
    assert len(h.store.list_workflows()) == 2


def test_a_failed_containment_prefix_hands_its_held_node_to_support() -> None:
    """A plan refused only for a missing owner runs its containment prefix as
    safety steps; a prefix step failing after the hold (the diagnostic
    bundle) leaves a FAILED record holding the node with neither the
    contained hand-off notice (that is a settled record's) nor a ticket."""

    prefix = [FREEZE, CORDON, QUARANTINE, BUNDLE]
    h = held_chain(prefix, prefix.index(BUNDLE))
    terminal_hold(h)
    h.amend(
        safety_steps=h.workflow.official_steps,
        safety_only=True,
        official_steps=[
            *h.workflow.official_steps,
            workflow_step(VALIDATE, node_ids=["node-a"]),
            workflow_step(SUPPORT, node_ids=["node-a"]),
        ],
        step_executions=[
            item.model_copy(update={"phase": "safety"})
            for item in h.workflow.step_executions
        ],
    )

    result = h.escalate()

    assert result is not None, "the held node needs a durable hand-off"
    incident, successor = result
    assert incident.reasons[0].startswith("held_chain_step remediation failed"), (
        incident.reasons
    )
    assert [step.operation for step in successor.official_steps] == [
        FREEZE,
        CORDON,
        QUARANTINE,
        SUPPORT,
    ]


@pytest.mark.parametrize(
    ("details", "expected", "why"),
    [
        (
            REJECTED,
            None,
            "refused before dispatch: a stale generation, a newer one owns it",
        ),
        (
            UNKNOWN_OUTCOME,
            "manual_confirmation_required",
            "an operator confirms the stop first: one ticket, not a second one",
        ),
    ],
    ids=["executor-rejected", "unknown-outcome"],
)
def test_a_quarantine_plan_keeps_its_deliberate_stop_endings(
    details, expected, why
) -> None:
    h = held_chain(QUARANTINE_PLAN, QUARANTINE_PLAN.index(STOP), details=details)
    terminal_hold(h)

    classification = HardwareEscalationService.classify(h.workflow)

    assert (classification and classification[0]) == expected, why
    h.escalate()
    assert len(h.store.list_workflows()) == (1 if expected is None else 2), why


COOLDOWN = {"health_recovery_contract": {"mode": "THERMAL_COOLDOWN"}}
REFUSED = {"safety_rejection": True}
RESTORE = WorkflowOperation.RESTORE_SCHEDULING
COOLDOWN_CHAIN = [FREEZE, CORDON, BUNDLE, SNAPSHOT, VALIDATE, RESTORE]


@pytest.mark.parametrize(
    ("operations", "failed", "details", "contract", "stage", "successor"),
    [
        (
            ENV_REVIEW_CHAIN,
            CORDON,
            None,
            ENV_REVIEW,
            "containment_or_release",
            [FREEZE, CORDON, QUARANTINE, SUPPORT],
        ),
        (
            ENV_REVIEW_CHAIN,
            CORDON,
            REFUSED,
            ENV_REVIEW,
            "containment_refused",
            [FREEZE, SUPPORT],
        ),
        (
            ENV_REVIEW_CHAIN,
            QUARANTINE,
            None,
            ENV_REVIEW,
            "containment_or_release",
            [FREEZE, CORDON, QUARANTINE, SUPPORT],
        ),
        (
            COOLDOWN_CHAIN,
            RESTORE,
            None,
            COOLDOWN,
            "containment_or_release",
            [FREEZE, CORDON, QUARANTINE, SUPPORT],
        ),
    ],
    ids=[
        "environment-review-cordon",
        "environment-review-cordon-refused",
        "environment-review-quarantine",
        "cooldown-restore",
    ],
)
def test_a_contract_chain_whose_isolation_fails_gets_the_containment_hand_off(
    operations, failed, details, contract, stage, successor
) -> None:
    """A diagnostic contract keeps the hardware rungs away, never the operator:
    its cordon, quarantine or restore that fails is handed over like in every
    other chain (F-H1). The ENVIRONMENT_REVIEW chain whose cordon failed held
    nothing and ended ESCALATED with no mail; it now opens the containment
    hand-off, and ESCALATED names a real escalation."""

    h = held_chain(
        operations, operations.index(failed), details=details, contract=contract
    )

    classification = HardwareEscalationService.classify(h.workflow)
    assert classification is not None, f"{failed.value} failed and nobody was told"
    assert classification[:3] == (stage, RecoveryAction.ESCALATE_OPERATOR, SUPPORT)
    result = h.escalate()
    assert result is not None, "the failed isolation needs a durable hand-off"
    incident, successor_workflow = result
    assert incident.reasons[0].startswith(f"{stage} remediation failed"), (
        incident.reasons
    )
    assert [step.operation for step in successor_workflow.official_steps] == (
        successor
    ), "a refused isolation is not planned again; no hardware rung either way"
    assert h.escalate() == result, "one hand-off per failed chain"
    assert len(h.store.list_workflows()) == 2


@pytest.mark.parametrize(
    ("details", "expected", "why"),
    [
        (
            REJECTED,
            None,
            "refused before dispatch: a stale generation, a newer one owns it",
        ),
        (
            UNKNOWN_OUTCOME,
            "manual_confirmation_required",
            "an operator confirms the cordon first: one ticket, not a second one",
        ),
    ],
    ids=["executor-rejected", "unknown-outcome"],
)
def test_a_contract_chains_cordon_endings_match_every_other_chain(
    details, expected, why
) -> None:
    """A cordon refused before dispatch opens nothing (no second ticket); one
    whose outcome nobody knows gets the manual-confirmation ticket, the only
    one -- the endings a cordon of any other chain has."""

    h = held_chain(
        ENV_REVIEW_CHAIN,
        ENV_REVIEW_CHAIN.index(CORDON),
        details=details,
        contract=ENV_REVIEW,
    )

    classification = HardwareEscalationService.classify(h.workflow)

    assert (classification and classification[0]) == expected, why
    h.escalate()
    assert len(h.store.list_workflows()) == (1 if expected is None else 2), why


QUIESCE = WorkflowOperation.QUIESCE_GPU_SERVICES
VERIFY = WorkflowOperation.VERIFY_NO_GPU_CLIENTS
RESTORE_SERVICES = WorkflowOperation.RESTORE_GPU_SERVICES
#: The release chains: they cordon in order to readmit the node.
RESET_CHAIN = [
    FREEZE,
    CORDON,
    CHECKPOINT,
    STOP,
    QUIESCE,
    VERIFY,
    RESET,
    RESTORE_SERVICES,
    VALIDATE,
    RESTORE,
    RESTART,
]
REBOOT_CHAIN = [
    FREEZE,
    CORDON,
    QUARANTINE,
    CHECKPOINT,
    STOP,
    REBOOT,
    VALIDATE,
    WorkflowOperation.VALIDATE_HOST,
    WorkflowOperation.VALIDATE_FABRIC,
    RESTORE,
    RESTART,
]
#: The older persisted THERMAL_COOLDOWN plan (``families/health._operations``).
COOLDOWN_PLAN = [FREEZE, CORDON, STOP, BUNDLE, SNAPSHOT, VALIDATE, RESTORE, RESTART]


@pytest.mark.parametrize(
    ("operations", "failed", "contract", "job_wide"),
    [
        (RESET_CHAIN, STOP, None, False),
        (RESET_CHAIN, STOP, None, True),
        (RESET_CHAIN, VERIFY, None, False),
        (RESET_CHAIN, CHECKPOINT, None, True),
        (REBOOT_CHAIN, STOP, None, True),
        (COOLDOWN_PLAN, VALIDATE, COOLDOWN, False),
        (COOLDOWN_PLAN, SNAPSHOT, COOLDOWN, False),
        (COOLDOWN_PLAN, BUNDLE, COOLDOWN, False),
        (COOLDOWN_PLAN, STOP, COOLDOWN, True),
    ],
    ids=[
        "reset-stop",
        "reset-job-wide-stop",
        "reset-verify",
        "reset-checkpoint-answered",
        "reboot-job-wide-stop",
        "cooldown-validate",
        "cooldown-snapshot",
        "cooldown-bundle",
        "cooldown-job-wide-stop",
    ],
)
def test_a_release_chain_whose_step_fails_after_the_cordon_hands_the_node_over(
    operations, failed, contract, job_wide
) -> None:
    """A reset, a reboot and the older cooldown plan cordon in order to
    readmit the node. A step after the cordon that fails where no rung
    answers -- the job stop, the GPU-client check, a checkpoint that
    answered; the cooldown's validation, snapshot or bundle -- left the record
    FAILED + QUARANTINED with no ticket and no notice. The held node is now
    handed to support, and only the held node."""

    h = held_chain(operations, operations.index(failed), contract=contract)
    if job_wide:
        h.amend(
            official_steps=[
                step.model_copy(update={"node_ids": ["node-a", "node-b"]})
                if step.operation in JOB_WIDE
                else step
                for step in h.workflow.official_steps
            ]
        )

    classification = HardwareEscalationService.classify(h.workflow)
    assert classification is not None, (
        f"{failed.value} failed after the cordon and nobody was told"
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
    assert incident.node_ids == ["node-a"], incident.node_ids
    assert [(step.operation, step.node_ids) for step in successor.official_steps] == [
        (FREEZE, ["node-a"]),
        (CORDON, ["node-a"]),
        (QUARANTINE, ["node-a"]),
        (SUPPORT, ["node-a"]),
    ], "the hold is kept and handed over: no hardware rung, no release, no node-b"
    assert h.escalate() == result, "one support successor per failed chain"
    assert len(h.store.list_workflows()) == 2


@pytest.mark.parametrize(
    ("operations", "failed_index", "details", "contract", "why"),
    [
        (
            RESET_CHAIN,
            RESET_CHAIN.index(STOP),
            REJECTED,
            None,
            "refused before dispatch: a stale generation, a newer one owns it",
        ),
        (
            RESET_CHAIN,
            RESET_CHAIN.index(CHECKPOINT),
            UNKNOWN_OUTCOME,
            None,
            "an abandoned checkpoint may still be running inside the job",
        ),
        (
            COOLDOWN_PLAN,
            COOLDOWN_PLAN.index(SNAPSHOT),
            REJECTED,
            COOLDOWN,
            "refused before dispatch: a stale generation, a newer one owns it",
        ),
        (
            RESET_CHAIN,
            RESET_CHAIN.index(RESTART),
            {"error": "the job did not come back"},
            None,
            "the node was released before the restart: nothing is held",
        ),
    ],
    ids=[
        "reset-stop-executor-rejected",
        "reset-abandoned-checkpoint",
        "cooldown-snapshot-executor-rejected",
        "reset-restart-after-release",
    ],
)
def test_a_release_chain_keeps_its_deliberate_no_ticket_endings(
    operations, failed_index, details, contract, why
) -> None:
    h = held_chain(operations, failed_index, details=details, contract=contract)

    assert HardwareEscalationService.classify(h.workflow) is None, why
    assert h.escalate() is None, why
    assert len(h.store.list_workflows()) == 1, "no successor was opened"


@pytest.mark.parametrize(
    ("operations", "contract"),
    [
        (RESET_CHAIN, None),
        (DRAIN_STOP_CHAIN, None),
        (QUARANTINE_PLAN, None),
        (ENV_REVIEW_CHAIN, ENV_REVIEW),
        (COOLDOWN_PLAN, COOLDOWN),
    ],
    ids=["reset", "drain", "quarantine-plan", "environment-review", "cooldown"],
)
def test_an_unknown_outcome_job_wide_stop_holds_the_whole_allocation(
    operations, contract
) -> None:
    """The one hand-off that is not narrowed to the held nodes: a job-wide
    STOP whose outcome nobody knows. Until an operator confirms the job's
    state, isolating its whole allocation is the safe choice, so the
    manual-confirmation hand-off cordons, quarantines and names every node of
    the stop -- in every chain, the diagnostic-contract chains included."""

    h = held_chain(
        operations, operations.index(STOP), details=UNKNOWN_OUTCOME, contract=contract
    )
    h.amend(
        official_steps=[
            step.model_copy(update={"node_ids": ["node-a", "node-b"]})
            if step.operation in JOB_WIDE
            else step
            for step in h.workflow.official_steps
        ]
    )

    classification = HardwareEscalationService.classify(h.workflow)
    assert classification is not None, "an operator must confirm the stop"
    assert classification[:3] == (
        "manual_confirmation_required",
        RecoveryAction.ESCALATE_OPERATOR,
        SUPPORT,
    )
    incident, successor = h.escalate()
    assert incident.node_ids == ["node-a", "node-b"], incident.node_ids
    assert {
        step.operation: step.node_ids
        for step in successor.official_steps
        if step.operation in {CORDON, QUARANTINE}
    } == {CORDON: ["node-a", "node-b"], QUARANTINE: ["node-a", "node-b"]}, (
        "the job's state is unknown on every node of the stop"
    )


def test_an_unknown_outcome_in_a_read_only_chain_cordons_nothing() -> None:
    """A read-only hung diagnostic holds nothing: its evidence step that the
    executor abandoned (outcome unknown) opens no manual-confirmation
    hand-off, so the diagnostic never cordons or quarantines a node."""

    hung = {"health_recovery_contract": {"mode": "HUNG_DIAGNOSTIC"}}
    operations = [
        FREEZE,
        WorkflowOperation.COLLECT_HUNG_SUMMARY,
        WorkflowOperation.COLLECT_HUNG_TRIAGE,
        BUNDLE,
        WorkflowOperation.VALIDATE_FABRIC,
        SUPPORT,
    ]
    h = held_chain(
        operations,
        operations.index(WorkflowOperation.COLLECT_HUNG_TRIAGE),
        details=UNKNOWN_OUTCOME,
        contract=hung,
    )

    assert HardwareEscalationService.classify(h.workflow) is None
    assert h.escalate() is None
    assert len(h.store.list_workflows()) == 1, "no successor was opened"


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

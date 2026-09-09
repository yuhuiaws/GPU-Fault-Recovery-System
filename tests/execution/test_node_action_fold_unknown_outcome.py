"""The node-action fold of a FAILED whose outcome the agent does not know.

An install the Node Agent killed at its deadline (``TimeoutExpired``) is a
non-retryable FAILED whose ``details`` carry ``outcome_unknown`` and
``manual_confirmation_required``: the installer may still be running, or may
have half-written the driver. ``_fold_result`` used to copy the agent's details
into the step record only for a transport retry that ran out of attempts, so
these flags stopped at the adapter and both ladders read the step as a
*definite* failure: the whole-workflow classifier filed a driver install under
``software_or_firmware_remediation`` with the default "validation failed"
reason, and an EFA driver install -- whose step carries
``failure_escalation_action=REBOOT_NODE`` -- was promoted to RESTART_NODE over
a node whose installer may still have been running.
"""

from __future__ import annotations

from gpu_fault.adapters import NodeActionWorkflowAdapter
from gpu_fault.models import (
    RecoveryAction,
    WorkflowOperation,
    WorkflowStatus,
    WorkflowStepStatus,
)
from gpu_fault.node_agent import NodeActionResult, NodeActionStatus
from gpu_fault.orchestration.escalation import HardwareEscalationService
from tests._builders import workflow_request, workflow_step, workflow_step_execution
from tests.execution.test_node_action_transport_retry import (
    ENDPOINT,
    SECRET,
    step_context,
)

INSTALL = WorkflowOperation.REMEDIATE_DRIVER
AGENT_ERROR = (
    "InstallOutcomeUnknownError: driver install did not finish within 1800s and "
    "was killed (TimeoutExpired); the node state is unknown and manual "
    "confirmation is required before any retry"
)
AGENT_DETAILS = {
    "outcome_unknown": True,
    "manual_confirmation_required": True,
    "install": "driver",
    "install_timeout_seconds": 1800,
}


def _install_timeout_fold(operation: WorkflowOperation = INSTALL):
    """Fold one install-timeout result through the real adapter."""

    agent_result = NodeActionResult(
        command_id="workflow/step/node-a",
        operation=operation,
        status=NodeActionStatus.FAILED,
        error=AGENT_ERROR,
        retryable=False,
        details=dict(AGENT_DETAILS),
    )
    adapter = NodeActionWorkflowAdapter(
        {"node-a": ENDPOINT}, SECRET, sender=lambda _endpoint, _envelope: agent_result
    )
    return adapter.execute(step_context(adapter, operation))


def test_an_install_timeout_keeps_its_unknown_outcome_flags_in_the_step_details():
    outcome = _install_timeout_fold()

    assert outcome.status is WorkflowStepStatus.FAILED, outcome
    assert outcome.details.get("outcome_unknown") is True, outcome.details
    assert outcome.details.get("manual_confirmation_required") is True, outcome.details
    # What the agent knew about the install travels with the flags: an
    # operator confirming the node reads which installer and which deadline.
    assert outcome.details.get("install") == "driver", outcome.details
    assert outcome.details.get("install_timeout_seconds") == 1800, outcome.details


def test_an_install_timeout_names_the_node_and_the_cause_like_an_interruption():
    """``failed_nodes`` / ``node_failures`` are what the support reason quotes."""

    outcome = _install_timeout_fold()

    assert outcome.details.get("failed_nodes") == ["node-a"], outcome.details
    assert outcome.details.get("node_failures") == {
        "node-a": [f"node action outcome unknown: {AGENT_ERROR}"]
    }, outcome.details
    # The step's own accounting still comes last and is not the agent's to set.
    assert outcome.details.get("completed_nodes") == [], outcome.details
    assert outcome.details.get("node_results") == {}, outcome.details


def test_a_plain_non_retryable_failure_does_not_borrow_the_unknown_outcome_shape():
    """A definite failure (clean non-zero exit) still climbs the ladder."""

    agent_result = NodeActionResult(
        command_id="workflow/step/node-a",
        operation=INSTALL,
        status=NodeActionStatus.FAILED,
        error="RuntimeError: driver install exited 1",
        retryable=False,
        details={"install": "driver"},
    )
    adapter = NodeActionWorkflowAdapter(
        {"node-a": ENDPOINT}, SECRET, sender=lambda _endpoint, _envelope: agent_result
    )

    outcome = adapter.execute(step_context(adapter, INSTALL))

    assert outcome.status is WorkflowStepStatus.FAILED, outcome
    assert "outcome_unknown" not in outcome.details, outcome.details
    assert "manual_confirmation_required" not in outcome.details, outcome.details
    assert "node_failures" not in outcome.details, outcome.details


def _failed_after_fold(folded, install_step):
    steps = [
        workflow_step(WorkflowOperation.MARK_UNSCHEDULABLE, node_ids=["node-a"]),
        install_step,
    ]
    return workflow_request(
        "wf",
        "inc",
        status=WorkflowStatus.FAILED,
        official_steps=steps,
        completed_step_indexes=[0],
        completed_operations=[steps[0].operation],
        step_executions=[
            workflow_step_execution(0, steps[0].operation),
            workflow_step_execution(
                1,
                install_step.operation,
                WorkflowStepStatus.FAILED,
                error=folded.error,
                details=dict(folded.details),
            ),
        ],
    )


def test_the_folded_install_timeout_is_classified_for_manual_confirmation():
    """End to end from the fold: the details the adapter wrote decide the stage.

    The driver install already ended at an operator, but under a stage that
    says the remediation definitely failed; the unknown-outcome stage is what
    tells the operator to go and read the node before anything is retried.
    """

    folded = _install_timeout_fold()
    workflow = _failed_after_fold(
        folded, workflow_step(INSTALL, node_ids=["node-a"], gpu_uuids=["GPU-a"])
    )

    classification = HardwareEscalationService.classify(workflow)

    assert classification is not None, "an install timeout must be escalated"
    stage, action, operation, failures = classification
    assert action is RecoveryAction.ESCALATE_OPERATOR, (
        "an install whose outcome is unknown must reach an operator, not the "
        f"reboot rung: {stage} -> {action}"
    )
    assert operation is WorkflowOperation.ESCALATE_SUPPORT, operation
    assert stage == "manual_confirmation_required", stage
    assert [item.step_index for item in failures] == [1], failures


def test_a_folded_efa_install_timeout_is_not_promoted_to_a_reboot():
    """The EFA remediation step asks for REBOOT_NODE on failure (``health.py``).

    That request is right for a definite failure and wrong for an installer
    that may still be running; the flags the fold now carries take precedence.
    """

    folded = _install_timeout_fold(WorkflowOperation.REMEDIATE_EFA_DRIVER)
    efa_step = workflow_step(
        WorkflowOperation.REMEDIATE_EFA_DRIVER,
        node_ids=["node-a"],
        gpu_uuids=["GPU-a"],
        parameters={"failure_escalation_action": RecoveryAction.REBOOT_NODE.value},
    )

    classification = HardwareEscalationService.classify(
        _failed_after_fold(folded, efa_step)
    )

    assert classification is not None, "an EFA install timeout must be escalated"
    stage, action, operation, _failures = classification
    assert action is RecoveryAction.ESCALATE_OPERATOR, (
        f"an EFA installer that may still be running must not be rebooted over: "
        f"{stage} -> {action}"
    )
    assert operation is WorkflowOperation.ESCALATE_SUPPORT, operation
    assert stage == "manual_confirmation_required", stage

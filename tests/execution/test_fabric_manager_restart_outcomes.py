"""Which failed RESTART_FABRIC_MANAGER climbs to the reboot.

The solo XID 45 plan on a node whose fabric is not graded is FREEZE_EVIDENCE ->
RESTART_FABRIC_MANAGER: no cordon, nothing stopped. A failed restart climbs to
the reboot (stage ``fabric_manager_restart``) only when the Node Agent reports
that the Fabric Manager unit failed -- the node-action fold writes
``fabric_manager_restart_outcome`` UNIT_FAILED for a ``systemctl`` call on the
unit that exited non-zero once the agent's own checks had passed. Every other
failure never ran the restart -- the node's configuration or the GPU-client
check refused it, a read before the unit failed, the command was refused or
lost before the agent's handler answered -- or says nothing about the unit, and
gets the refused hand-off (``node_action_refused``: FREEZE_EVIDENCE ->
ESCALATE_SUPPORT, no isolation, no reboot), its reason naming why. The endings
judged before the ladder are kept: a restart the executor refused before
dispatch opens nothing, one the safety boundary refused gets the refused
hand-off, one whose outcome is unknown the operator's confirmation.

Each failure with a producer here runs the real agent handler, the real
node-action fold and the real control-plane executor.
"""

from __future__ import annotations

import io
import json
from dataclasses import dataclass
from email.message import Message
from pathlib import Path
from subprocess import CalledProcessError, CompletedProcess, TimeoutExpired
from typing import Any
from urllib import error as urllib_error

import pytest

import gpu_fault.node_agent.operations.reset as reset_operations
from gpu_fault.adapters import NodeActionWorkflowAdapter
from gpu_fault.adapters.node_action.fabric_manager_outcome import (
    fabric_manager_restart_outcome,
)
from gpu_fault.execution.node_action_uncertainty import has_unresolved_node_action
from gpu_fault.models import (
    BlockedKind,
    WorkflowOperation,
    WorkflowStatus,
    WorkflowStepStatus,
)
from gpu_fault.node_agent import NodeActionResult, NodeActionStatus
from gpu_fault.node_agent.late_ownership import OwnershipRefused
from gpu_fault.orchestration.escalation import HardwareEscalationService
from gpu_fault.orchestration.failure_outcome import (
    FABRIC_MANAGER_RESTART_OUTCOME_DETAIL,
    FABRIC_MANAGER_RESTART_OUTCOMES,
    FM_RESTART_NO_UNIT_VERDICT,
    FM_RESTART_REFUSED_BY_NODE_CONFIGURATION,
    FM_RESTART_REFUSED_GPU_CLIENTS,
    FM_RESTART_UNIT_FAILED,
)
from tests._builders import active_workflow_executor, node_action_result
from tests.execution._cov95_runtime_workflows import FlowHarness
from tests.execution._support import StubFleetRegistry
from tests.node_agent._support import SECRET, node_action_executor
from tests.orchestration._cov95_runtime_escalation import EscalationHarness

OP = WorkflowOperation
FM = "nvidia-fabricmanager"
SOLO = [OP.FREEZE_EVIDENCE, OP.RESTART_FABRIC_MANAGER]
ENDPOINT = "http://node-a:9099"
REBOOT = ("fabric_manager_restart", OP.RESTART_NODE)
SUPPORT = ("node_action_refused", OP.ESCALATE_SUPPORT)
#: What the refused hand-off plans: evidence and the ticket, nothing on the node.
HAND_OFF = [OP.FREEZE_EVIDENCE, OP.ESCALATE_SUPPORT]
#: The calls the agent's restart makes on its success path, in order. Each is a
#: failure point with its own case in AGENT_CASES.
RESTART_CALLS = [
    ("nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name"),
    ("systemctl", "show"),
    ("systemctl", "restart"),
    ("systemctl", "is-active"),
    ("systemctl", "show"),
]


class HostCalls:
    """nvidia-smi and systemctl as a node with a running Fabric Manager answers
    them. ``fail_call`` fails the call at that position of the restart's call
    sequence, ``fail_command`` every call that starts with it."""

    def __init__(
        self,
        *,
        fail_call: int | None = None,
        fail_command: tuple[str, ...] | None = None,
        failure: tuple[int, str] | BaseException = (1, "failed"),
        clients: str = "",
    ) -> None:
        self.fail_call = fail_call
        self.fail_command = fail_command
        self.failure = failure
        self.clients = clients
        self.commands: list[list[str]] = []

    def __call__(self, value: list[str], *, check: bool = False, **_: Any):
        position = len(self.commands)
        self.commands.append(list(value))
        failing = position == self.fail_call or (
            self.fail_command is not None
            and tuple(value[: len(self.fail_command)]) == self.fail_command
        )
        returncode, stderr = 0, ""
        if failing:
            if isinstance(self.failure, BaseException):
                raise self.failure
            returncode, stderr = self.failure
        stdout = (
            self.clients
            if value[0] == "nvidia-smi"
            else "101\n"
            if value[:2] == ["systemctl", "show"]
            else ""
        )
        if check and returncode:
            raise CalledProcessError(returncode, value, stderr=stderr)
        return CompletedProcess(value, returncode, stdout=stdout, stderr=stderr)


def solo_restart(
    tmp_path: Path,
    host: HostCalls,
    *,
    enabled: bool = True,
    send: Any = None,
    registry: Any = None,
    endpoints: dict[str, str] | None = None,
    allowed: list[WorkflowOperation] = SOLO,
    cap_seconds: int | None = None,
    ticks: int = 1,
) -> FlowHarness:
    """The solo XID 45 plan, its restart sent to a real agent through the real
    node-action adapter and run by the real control-plane executor."""

    agent = node_action_executor(
        tmp_path,
        "fm-actions.db",
        allowed_operations={OP.RESTART_FABRIC_MANAGER},
        fabric_manager_restart_enabled=enabled,
        runner=host,
        now=None,
    )

    def to_agent(_endpoint: str, envelope: Any) -> NodeActionResult:
        result = agent.execute(envelope)
        return NodeActionResult.model_validate_json(result.model_dump_json())

    h = FlowHarness(SOLO)
    if allowed != SOLO or cap_seconds is not None:
        h.executor = active_workflow_executor(
            h.store,
            [h.adapter],
            allowed,
            step_waiting_timeout_overrides=(
                None
                if cap_seconds is None
                else {OP.RESTART_FABRIC_MANAGER: cap_seconds}
            ),
        )
    adapter = NodeActionWorkflowAdapter(
        {"node-a": ENDPOINT} if endpoints is None else endpoints,
        SECRET,
        sender=send or to_agent,
        registry=registry,
    )
    h.executor.adapters.insert(0, adapter)
    h.amend(
        official_steps=[
            step.model_copy(update={"execution_owner": adapter.owner})
            if step.operation is OP.RESTART_FABRIC_MANAGER
            else step
            for step in h.workflow.official_steps
        ]
    )
    for _ in range(ticks):
        h.execute()
    return h


def restart_record(h: FlowHarness):
    workflow = h.store.get_workflow(h.workflow.request_id)
    restart = [
        item
        for item in workflow.step_executions
        if item.operation is OP.RESTART_FABRIC_MANAGER
    ][-1]
    return workflow, restart


def ending(h: FlowHarness) -> tuple[tuple[str, WorkflowOperation], str]:
    """The rung the failed record climbs to and what its hand-off says."""

    workflow, restart = restart_record(h)
    assert workflow.status is WorkflowStatus.FAILED, workflow.status
    assert restart.status is WorkflowStepStatus.FAILED, restart.status
    classification = HardwareEscalationService.classify(workflow)
    assert classification is not None, "a failed restart never ends silently"
    stage, _action, operation, failed = classification
    scope = HardwareEscalationService.collect_scope(
        workflow, h.store.get_incident(workflow.incident_id), failed
    )
    return (stage, operation), scope["node_reason"]


@dataclass(frozen=True)
class AgentCase:
    name: str
    outcome: str | None
    ending: tuple[str, WorkflowOperation]
    reason: str
    fail_call: int | None = None
    failure: tuple[int, str] = (1, "failed")
    enabled: bool = True
    clients: str = ""
    refuse_checkpoint: bool = False


AGENT_CASES = (
    AgentCase(
        "node-configuration",
        FM_RESTART_REFUSED_BY_NODE_CONFIGURATION,
        SUPPORT,
        "node-a: RESTART_FABRIC_MANAGER refused by the node configuration (Fabric "
        "Manager restart not enabled); the restart never ran",
        enabled=False,
    ),
    AgentCase(
        "compute-apps-query",
        FM_RESTART_NO_UNIT_VERDICT,
        SUPPORT,
        "node-a: RESTART_FABRIC_MANAGER failed with no verdict on the unit: node "
        "agent node-a: RuntimeError: nvidia-smi exited with status 9: driver gone",
        fail_call=0,
        failure=(9, "driver gone"),
    ),
    AgentCase(
        "gpu-clients",
        FM_RESTART_REFUSED_GPU_CLIENTS,
        SUPPORT,
        "node-a: RESTART_FABRIC_MANAGER refused: GPU compute clients active on a "
        "node nothing stopped; the restart never ran",
        clients="GPU-a, 4242, python\n",
    ),
    AgentCase(
        "unit-read-before",
        FM_RESTART_UNIT_FAILED,
        REBOOT,
        "node-a: RESTART_FABRIC_MANAGER failed",
        fail_call=1,
        failure=(1, "Failed to get properties: Connection timed out"),
    ),
    AgentCase(
        "ownership-checkpoint",
        None,
        SUPPORT,
        "node-a: node action refused before acting: OwnershipRefused: "
        "OWNERSHIP_PERMIT_EXPIRED",
        refuse_checkpoint=True,
    ),
    AgentCase(
        "unit-restart",
        FM_RESTART_UNIT_FAILED,
        REBOOT,
        "node-a: RESTART_FABRIC_MANAGER failed",
        fail_call=2,
        failure=(1, "Job for nvidia-fabricmanager.service failed"),
    ),
    # The record cannot tell a unit that is not installed from one that failed:
    # the agent's read before the restart does not ask for the unit's LoadState.
    AgentCase(
        "unit-not-installed",
        FM_RESTART_UNIT_FAILED,
        REBOOT,
        "node-a: RESTART_FABRIC_MANAGER failed",
        fail_call=2,
        failure=(5, f"Failed to restart {FM}.service: Unit {FM}.service not found."),
    ),
    AgentCase(
        "unit-active-check",
        FM_RESTART_UNIT_FAILED,
        REBOOT,
        "node-a: RESTART_FABRIC_MANAGER failed",
        fail_call=3,
        failure=(3, ""),
    ),
    AgentCase(
        "unit-read-after",
        FM_RESTART_UNIT_FAILED,
        REBOOT,
        "node-a: RESTART_FABRIC_MANAGER failed",
        fail_call=4,
        failure=(1, "Failed to get properties: Connection timed out"),
    ),
)


def test_every_call_of_the_agent_s_restart_is_a_failure_point_pinned_here(
    tmp_path: Path,
) -> None:
    """The guard: a call the restart gains is a failure nobody classified."""

    host = HostCalls()

    h = solo_restart(tmp_path, host)

    workflow, restart = restart_record(h)
    assert workflow.status is WorkflowStatus.SUCCEEDED, restart.error
    assert [tuple(call[:2]) for call in host.commands] == RESTART_CALLS, (
        "the agent's restart calls changed: give the new call's failure its "
        "outcome and ending in AGENT_CASES"
    )
    assert {
        case.fail_call for case in AGENT_CASES if case.fail_call is not None
    } == set(range(len(RESTART_CALLS))), "every call of the restart fails once"


@pytest.mark.parametrize("case", AGENT_CASES, ids=lambda case: case.name)
def test_each_failure_of_the_agent_s_restart_has_its_rung(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, case: AgentCase
) -> None:
    if case.refuse_checkpoint:

        def refuse() -> None:
            # What the final ownership boundary raises at the checkpoint
            # before the restart (``late_ownership.physical_check``).
            refusal = OwnershipRefused("OWNERSHIP_PERMIT_EXPIRED")
            refusal.action_details["node_action_outcome_known"] = True
            raise refusal

        monkeypatch.setattr(reset_operations, "require_physical_ownership", refuse)
    host = HostCalls(
        fail_call=case.fail_call, failure=case.failure, clients=case.clients
    )

    h = solo_restart(tmp_path, host, enabled=case.enabled)

    _workflow, restart = restart_record(h)
    assert restart.details.get(FABRIC_MANAGER_RESTART_OUTCOME_DETAIL) == case.outcome
    rung, reason = ending(h)
    assert rung == case.ending, case.name
    assert reason == case.reason
    restarted = ["systemctl", "restart", FM] in host.commands
    assert restarted is (case.fail_call is not None and case.fail_call >= 2), (
        "the unit is restarted only once every check before it passed"
    )


def _http_403(_endpoint: str, _envelope: Any) -> NodeActionResult:
    """The agent's answer when its allowlist lacks the operation."""

    body = {
        "detail": {
            "code": "OPERATION_NOT_ALLOWED",
            "message": "operation RESTART_FABRIC_MANAGER is not allowed",
            "retryable": False,
            "requires_new_command": False,
        }
    }
    raise urllib_error.HTTPError(
        f"{ENDPOINT}/v1/node-actions/submit",
        403,
        "Forbidden",
        Message(),
        io.BytesIO(json.dumps(body).encode()),
    )


@pytest.mark.parametrize(
    ("setup", "cause"),
    [
        pytest.param(
            lambda: {"registry": StubFleetRegistry({"node-a": ENDPOINT}, ready=False)},
            "fleet consistency gate failed: node-a: heartbeat is stale",
            id="fleet-gate",
        ),
        pytest.param(
            lambda: {"allowed": [OP.FREEZE_EVIDENCE]},
            "operation RESTART_FABRIC_MANAGER is not in GPU_FAULT_ALLOWED_OPERATIONS",
            id="control-plane-allowlist",
        ),
        pytest.param(
            lambda: {"send": _http_403},
            "node agent node-a rejected request: HTTP 403: operation "
            "RESTART_FABRIC_MANAGER is not allowed",
            id="agent-allowlist",
        ),
        pytest.param(
            lambda: {"endpoints": {"node-b": "http://node-b:9099"}},
            "no node action endpoint for node-a",
            id="no-endpoint",
        ),
    ],
)
def test_a_restart_refused_before_the_agent_s_handler_never_reboots(
    tmp_path: Path, setup: Any, cause: str
) -> None:
    """The fleet gate, either allowlist (the operator's policy) or a node the
    adapter cannot address: the restart never reached the agent's handler."""

    host = HostCalls()

    h = solo_restart(tmp_path, host, **setup())

    _workflow, restart = restart_record(h)
    assert FABRIC_MANAGER_RESTART_OUTCOME_DETAIL not in restart.details
    rung, reason = ending(h)
    assert rung == SUPPORT
    assert reason == (
        f"node-a: RESTART_FABRIC_MANAGER failed with no verdict on the unit: {cause}"
    )
    assert host.commands == [], "nothing ran on the node"


def test_a_restart_that_timed_out_until_its_step_cap_never_reboots(
    tmp_path: Path,
) -> None:
    """``systemctl restart`` killed at its deadline is a retryable failure the
    agent runs again; the step cap then fails a step whose systemd restart job
    may still be running. The previous rung rebooted over it."""

    host = HostCalls(
        fail_command=("systemctl", "restart"),
        failure=TimeoutExpired(["systemctl", "restart", FM], 780),
    )

    h = solo_restart(tmp_path, host, cap_seconds=0, ticks=2)

    _workflow, restart = restart_record(h)
    assert restart.details["retryable_node_action"] is True
    assert FABRIC_MANAGER_RESTART_OUTCOME_DETAIL not in restart.details
    rung, reason = ending(h)
    assert rung == SUPPORT
    assert reason.startswith(
        "node-a: RESTART_FABRIC_MANAGER failed with no verdict on the unit: step "
        "1/RESTART_FABRIC_MANAGER stayed non-terminal"
    ), reason


def test_an_interrupted_restart_parks_for_the_operator_and_climbs_nothing(
    tmp_path: Path,
) -> None:
    """Unknown-outcome handling is kept: the restart may have run."""

    def interrupted(_endpoint: str, envelope: Any) -> NodeActionResult:
        return node_action_result(
            envelope.command.command_id,
            OP.RESTART_FABRIC_MANAGER,
            NodeActionStatus.INTERRUPTED,
            error="node action attempt was interrupted",
            retryable=False,
        )

    h = solo_restart(tmp_path, HostCalls(), send=interrupted)

    workflow, restart = restart_record(h)
    assert workflow.status is WorkflowStatus.BLOCKED, workflow.status
    assert workflow.blocked_kind is BlockedKind.NEEDS_OPERATOR
    assert has_unresolved_node_action(workflow), "parked for the confirmation"
    assert restart.details["node_action_interrupted"] is True
    assert FABRIC_MANAGER_RESTART_OUTCOME_DETAIL not in restart.details


def _restart_failure(details: dict[str, Any], **values: Any) -> EscalationHarness:
    h = EscalationHarness(SOLO, failed_index=1, details=details)
    if values:
        current = h.workflow
        executions = list(current.step_executions)
        executions[-1] = executions[-1].model_copy(update=values)
        h.amend(step_executions=executions)
    return h


@pytest.mark.parametrize(
    ("details", "error", "stage"),
    [
        pytest.param(
            {"remote_status_source": "unclaimed-deadline-exceeded"},
            "no cluster executor claimed the command before its deadline",
            SUPPORT[0],
            id="unclaimed",
        ),
        pytest.param(
            {"remote_status_source": "workflow-timeout"},
            "workflow execution deadline exceeded",
            SUPPORT[0],
            id="cancelled-before-any-report",
        ),
        pytest.param(
            {
                "workflow_execution_deadline": "2026-11-07T00:00:00+00:00",
                "workflow_deadline_overdue_seconds": 4,
                "workflow_lifetime_exceeded": False,
            },
            "workflow execution deadline exceeded before step 1",
            SUPPORT[0],
            id="execution-deadline",
        ),
        pytest.param(
            {
                "node_action_retry_exhausted": True,
                "node_action_attempts": 4,
                FABRIC_MANAGER_RESTART_OUTCOME_DETAIL: FM_RESTART_NO_UNIT_VERDICT,
            },
            "node agent node-a: node action failed after 4 attempts (re-submit "
            "limit 3): TimeoutExpired: ...",
            SUPPORT[0],
            id="re-submits-ran-out",
        ),
        pytest.param(
            {"remote_status_source": "executor-rejected"},
            "command fencing token does not match workflow/incident",
            None,
            id="executor-rejected-kept",
        ),
        pytest.param(
            {"workflow_lifetime_exceeded": True},
            "workflow lifetime exceeded",
            "lifetime_exceeded",
            id="lifetime-kept",
        ),
        pytest.param(
            {
                FABRIC_MANAGER_RESTART_OUTCOME_DETAIL: FM_RESTART_UNIT_FAILED,
                "outcome_unknown": True,
                "manual_confirmation_required": True,
            },
            "executor abandoned RESTART_FABRIC_MANAGER; outcome unknown",
            "manual_confirmation_required",
            id="unknown-outcome-kept",
        ),
        pytest.param(
            {
                "node_action_error_code": "OPERATION_NOT_ALLOWED",
                "http_status": 403,
                "safety_rejection": True,
                "manual_confirmation_required": True,
            },
            "node agent node-a rejected request: HTTP 403: not allowed",
            "manual_confirmation_required",
            id="guarded-agent-rejection-kept",
        ),
    ],
)
def test_a_restart_with_no_unit_verdict_never_reboots(
    details: dict[str, Any], error: str, stage: str | None
) -> None:
    """Failures whose producers live in the regional Store and executor, as the
    step record carries them; and the endings judged before the ladder."""

    h = _restart_failure(details, error=error)

    classification = HardwareEscalationService.classify(h.workflow)
    result = h.escalate()

    assert (classification and classification[0]) == stage
    if stage is None:
        assert result is None, "nothing reached the node: no ticket either"
        return
    assert result is not None, "never a silent FAILED"
    _incident, successor = result
    operations = [step.operation for step in successor.official_steps]
    assert OP.RESTART_NODE not in operations, operations
    assert OP.ESCALATE_SUPPORT in operations, operations


#: The rung each outcome the fold writes climbs to. The guard below fails when
#: an outcome is added without one.
OUTCOME_RUNGS = {
    FM_RESTART_UNIT_FAILED: REBOOT,
    FM_RESTART_REFUSED_BY_NODE_CONFIGURATION: SUPPORT,
    FM_RESTART_REFUSED_GPU_CLIENTS: SUPPORT,
    FM_RESTART_NO_UNIT_VERDICT: SUPPORT,
}


@pytest.mark.parametrize("outcome", sorted(OUTCOME_RUNGS))
def test_every_restart_outcome_has_its_rung_and_successor(outcome: str) -> None:
    assert set(OUTCOME_RUNGS) == FABRIC_MANAGER_RESTART_OUTCOMES, (
        "a new restart outcome needs its rung here and in the escalation"
    )
    clients = (
        {"gpu_client_kind": "compute"}
        if outcome == FM_RESTART_REFUSED_GPU_CLIENTS
        else {}
    )
    h = _restart_failure({FABRIC_MANAGER_RESTART_OUTCOME_DETAIL: outcome, **clients})

    classification = HardwareEscalationService.classify(h.workflow)
    result = h.escalate()

    assert classification is not None
    assert (classification[0], classification[2]) == OUTCOME_RUNGS[outcome]
    assert result is not None
    _incident, successor = result
    operations = [step.operation for step in successor.official_steps]
    if OUTCOME_RUNGS[outcome] == REBOOT:
        assert successor.request_id == "workflow-reboot-after-source-workflow"
        assert operations[:4] == [
            OP.FREEZE_EVIDENCE,
            OP.MARK_UNSCHEDULABLE,
            OP.QUARANTINE,
            OP.RESTART_NODE,
        ], "the reboot rung isolates the node before it reboots it"
    else:
        assert successor.request_id == "workflow-support-after-source-workflow"
        assert operations == HAND_OFF, operations


def test_a_record_reboots_only_when_every_failed_restart_s_unit_failed() -> None:
    """Fail-closed across a record: a node whose restart never ran is never
    rebooted because another node's unit failed."""

    h = EscalationHarness(
        [OP.FREEZE_EVIDENCE, OP.RESTART_FABRIC_MANAGER, OP.RESTART_FABRIC_MANAGER],
        failed_index=2,
        details={
            FABRIC_MANAGER_RESTART_OUTCOME_DETAIL: (
                FM_RESTART_REFUSED_BY_NODE_CONFIGURATION
            )
        },
    )
    executions = list(h.workflow.step_executions)
    executions[1] = executions[1].model_copy(
        update={
            "status": WorkflowStepStatus.FAILED,
            "details": {FABRIC_MANAGER_RESTART_OUTCOME_DETAIL: FM_RESTART_UNIT_FAILED},
        }
    )
    h.amend(step_executions=executions)

    classification = HardwareEscalationService.classify(h.workflow)

    assert classification is not None
    assert (classification[0], classification[2]) == SUPPORT


@pytest.mark.parametrize(
    ("values", "outcome"),
    [
        pytest.param(
            {"error": "RuntimeError: systemctl exited with status 1: failed"},
            FM_RESTART_UNIT_FAILED,
            id="unit-call",
        ),
        pytest.param(
            {
                "error": "node action failed after 4 attempts (re-submit limit 3): "
                "TimeoutExpired: Command '['systemctl', 'restart', "
                "'nvidia-fabricmanager']' timed out after 780 seconds",
                "details": {"node_action_retry_exhausted": True},
            },
            FM_RESTART_NO_UNIT_VERDICT,
            id="re-submits-ran-out",
        ),
        pytest.param(
            {
                "error": "RuntimeError: systemctl exited with status 1: failed",
                "details": {"node_action_retry_exhausted": True},
            },
            FM_RESTART_NO_UNIT_VERDICT,
            id="re-submits-ran-out-on-a-unit-call",
        ),
        pytest.param(
            {"error": "ValueError: an error that names no unit call"},
            FM_RESTART_NO_UNIT_VERDICT,
            id="unknown-wording",
        ),
        pytest.param(
            {
                "error": "OwnershipRefused: OWNERSHIP_PERMIT_EXPIRED",
                "details": {
                    "safety_rejection": True,
                    "manual_confirmation_required": True,
                    "node_action_not_started": True,
                },
            },
            None,
            id="safety-refusal-has-its-own-flags",
        ),
        pytest.param({"status": NodeActionStatus.SUCCEEDED}, None, id="succeeded"),
        pytest.param({"operation": OP.RESET_GPU}, None, id="another-operation"),
    ],
)
def test_the_fold_reads_the_agent_s_verdict_of_the_unit(
    values: dict[str, Any], outcome: str | None
) -> None:
    fields: dict[str, Any] = {
        "operation": OP.RESTART_FABRIC_MANAGER,
        "status": NodeActionStatus.FAILED,
        "error": "RuntimeError: systemctl exited with status 1: failed",
        "retryable": False,
        **values,
    }
    result = node_action_result(
        "workflow/1/RESTART_FABRIC_MANAGER/node-a",
        fields.pop("operation"),
        fields.pop("status"),
        **fields,
    )

    assert fabric_manager_restart_outcome(result) == outcome

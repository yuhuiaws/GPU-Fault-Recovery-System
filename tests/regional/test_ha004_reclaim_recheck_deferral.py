"""GF-REGIONAL-HA-004's reclaim boundary: an inconclusive final recheck defers.

The failing live sequence: the replica that had dispatched RESET_GPU handed
the WAITING carrier back, a sibling reclaimed it while the Agent's
``AGENT_PRE_SPAWN`` challenge was pending, its idle-node recheck raised inside
``check_idle`` and came back as a *returned* ``STOP_OWNERSHIP_UNVERIFIABLE``
refusal -- which the transport signed as a denial. A denial is irrevocable on
the Agent (a later grant is ``OWNERSHIP_PERMIT_STALE``) and terminal for the
reset (``retryable=False``), so one Kubernetes read that failed on one replica
quarantined a healthy node.

These tests pin the corrected contract end to end: only a violation the
validator observed is signed as a denial; a recheck that could not be completed
(a raised read, a missing validator, a truncated list, a lease this executor
can no longer vouch for) leaves the challenge unanswered so the next lease
owner re-runs it; and the Agent still fails closed on its own deadline when
nobody ever answers.
"""

from __future__ import annotations

import io
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from typing import Any

import pytest
from fastapi.testclient import TestClient

from gpu_fault.adapters import NodeActionWorkflowAdapter
from gpu_fault.adapters.kubernetes.stop_ownership import (
    CONCLUSIVE_REFUSAL_REASONS,
    refusal,
    refusal_is_conclusive,
    stop_ownership_scope,
)
from gpu_fault.adapters.node_action import transport
from gpu_fault.adapters.node_action.lease_guard import active_lease_guard
from gpu_fault.execution import WorkflowStepContext, WorkflowStepOutcome
from gpu_fault.models import (
    WorkflowOperation,
    WorkflowStepExecution,
    WorkflowStepSpec,
    WorkflowStepStatus,
)
from gpu_fault.node_agent import create_node_agent_app
from gpu_fault.node_agent.late_ownership import (
    NodeOwnershipGate,
    OwnershipRefused,
    sign_permit,
)
from gpu_fault.node_agent.protocol import (
    NodeActionExecutionState,
    NodeActionSubmission,
    SignedNodeAction,
)
from tests.node_agent._support import (
    SECRET,
    FakeRunner,
    no_device_clients,
    node_action_executor,
    result_params,
)
from tests.regional._late_ownership_runtime import runtime, stopped_runtime
from tests.regional.test_late_ownership_agent_boundary import (
    LocalRegistry,
    settled,
    transport_bridge,
)
from tests.regional.test_late_ownership_agent_gate import KEY, Clock
from tests.regional.test_late_ownership_agent_gate import command as owned_command
from tests.regional.test_late_ownership_transport import exchange, inputs

SHUTDOWN_HOLD = "executor shutdown requested: SIGTERM received"

# What the guard answered at the final checkpoint, and the deferral reason the
# transport must record instead of signing anything.
INCONCLUSIVE_ANSWERS: dict[str, tuple[Any, str]] = {
    "unverifiable": (
        lambda: refusal("STOP_OWNERSHIP_UNVERIFIABLE"),
        "STOP_OWNERSHIP_UNVERIFIABLE",
    ),
    "validator-unavailable": (None, "STOP_OWNERSHIP_VALIDATOR_UNAVAILABLE"),
    "participants-unknown": (
        lambda: refusal("STOP_PARTICIPANTS_UNKNOWN"),
        "STOP_PARTICIPANTS_UNKNOWN",
    ),
    "lease-lost": (
        lambda: WorkflowStepOutcome.waiting(
            details={
                "node_action_state": "LEASE_LOST",
                "node_action_not_started": True,
                "reason": SHUTDOWN_HOLD,
            }
        ),
        "OWNERSHIP_LEASE_LOST",
    ),
    "unlisted-code": (
        lambda: refusal("STOP_SOMETHING_NEWER_THAN_THIS_BUILD"),
        "STOP_SOMETHING_NEWER_THAN_THIS_BUILD",
    ),
    "malformed": (
        lambda: WorkflowStepOutcome.failed(
            "not public", details={"reason": "invalid text"}
        ),
        "STOP_OWNERSHIP_UNVERIFIABLE",
    ),
}


def connection_aborted(*_args: Any, **_kwargs: Any) -> Any:
    """A Kubernetes client failure: the class is diagnostic, the text is not."""

    raise RuntimeError("('Connection aborted.', RemoteDisconnected())")


def idle_reset_runtime() -> tuple[Any, Any, WorkflowStepContext]:
    """An idle node: no STOP step, no workload, one RESET_GPU step.

    HA-004 injects on an idle node, so its guard path is ``check_idle`` (node
    identity plus a GPU-Pod inventory), not the receipt-bound ``check``.
    """

    state, _adapter, validator, context = runtime()
    state.pods = []
    reset = WorkflowStepSpec(
        operation=WorkflowOperation.RESET_GPU,
        execution_owner="gpu-fault-node-agent",
        node_ids=["node-a"],
        gpu_uuids=["GPU-a"],
        workload_ids=[],
    )
    workflow = context.workflow.model_copy(update={"official_steps": [reset]})
    idle = WorkflowStepContext(
        workflow=workflow,
        incident=context.incident,
        step=reset,
        step_index=0,
        request=context.request,
        idempotency_key="workflow-local/0/RESET_GPU",
    )
    return state, validator, idle


def recording_agent(
    monkeypatch: pytest.MonkeyPatch, command_id: str
) -> list[SignedNodeAction]:
    """Record every submission the transport posts; answer PENDING to each."""

    posted: list[SignedNodeAction] = []

    def send(request: Any, **_kwargs: Any) -> io.BytesIO:
        posted.append(SignedNodeAction.model_validate_json(request.data))
        return io.BytesIO(
            NodeActionSubmission(
                command_id=command_id, state=NodeActionExecutionState.PENDING
            )
            .model_dump_json()
            .encode()
        )

    monkeypatch.setattr(transport, "urlopen", send)
    return posted


@pytest.mark.parametrize("answer", sorted(INCONCLUSIVE_ANSWERS))
def test_an_inconclusive_final_recheck_is_deferred_never_signed(
    monkeypatch: pytest.MonkeyPatch, answer: str
) -> None:
    adapter, envelope, pending = inputs()
    posted = recording_agent(monkeypatch, envelope.command.command_id)
    before, reason = INCONCLUSIVE_ANSWERS[answer]

    state, outcome = exchange(adapter, envelope, pending, before)

    assert state is pending, "the Agent's pending state is left as it was"
    assert posted == [], "nothing may be signed on a check that did not complete"
    assert outcome.status is WorkflowStepStatus.WAITING
    assert outcome.details["node_action_state"] == "OWNERSHIP_RECHECK_DEFERRED"
    assert outcome.details["ownership_recheck_deferred"] is True
    assert outcome.details["reason"] == reason
    assert outcome.details["ownership_check_boundary"] == "AGENT_PRE_SPAWN"
    assert outcome.details["ownership_challenge_sequence"] == 1
    assert outcome.details["node_action_command_id"] == envelope.command.command_id
    assert outcome.details["node_action_accepted_nodes"] == ["node-a"]
    assert "outcome_unknown" not in outcome.details
    assert "manual_confirmation_required" not in outcome.details
    assert "node_action_not_started" not in outcome.details
    assert "invalid text" not in str(outcome)
    assert SHUTDOWN_HOLD not in str(outcome), "hold text is not a public reason"


@pytest.mark.parametrize("reason", sorted(CONCLUSIVE_REFUSAL_REASONS))
def test_an_observed_violation_is_still_signed_as_a_denial(
    monkeypatch: pytest.MonkeyPatch, reason: str
) -> None:
    adapter, envelope, pending = inputs()
    posted = recording_agent(monkeypatch, envelope.command.command_id)

    _state, outcome = exchange(adapter, envelope, pending, lambda: refusal(reason))

    assert len(posted) == 1, "an observed violation is answered at once"
    permit = posted[0].ownership_permit
    assert permit is not None
    assert permit.allowed is False and permit.reason == reason
    assert outcome.status is WorkflowStepStatus.WAITING
    assert outcome.details["reason"] == "OWNERSHIP_DENIAL_UNCONFIRMED"
    assert outcome.details["node_action_command_id"] == envelope.command.command_id


def test_the_conclusive_table_names_observed_violations_only() -> None:
    assert CONCLUSIVE_REFUSAL_REASONS == {
        "STOP_OWNERSHIP_SCOPE_MISMATCH",
        "STOP_OWNERSHIP_IDENTITY_UNKNOWN",
        "STOP_OWNERSHIP_DRIFT",
        "STOP_PARTICIPANTS_ACTIVE",
        "STOP_PARTICIPANTS_CHANGED",
        "STOP_OWNERSHIP_RECEIPT_MISSING",
    }
    for reason in (
        "STOP_OWNERSHIP_UNVERIFIABLE",
        "STOP_OWNERSHIP_VALIDATOR_UNAVAILABLE",
        "STOP_PARTICIPANTS_UNKNOWN",
        "OWNERSHIP_LEASE_LOST",
        "OK",
        None,
        7,
    ):
        assert refusal_is_conclusive(reason) is False, reason


def test_check_idle_names_the_exception_class_it_failed_closed_on(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    state, validator, context = idle_reset_runtime()
    assert validator.check_idle(context) is None, "the idle node passes when read"
    monkeypatch.setattr(state, "read_node", connection_aborted)

    with caplog.at_level(logging.WARNING):
        outcome = validator.check_idle(context)

    assert outcome is not None and outcome.status is WorkflowStepStatus.FAILED
    assert outcome.details is not None
    assert outcome.details["reason"] == "STOP_OWNERSHIP_UNVERIFIABLE"
    assert outcome.details["refusal_cause"] == "RuntimeError"
    assert "Connection aborted" not in str(outcome), "no private diagnostic"
    assert any("RuntimeError" in record.getMessage() for record in caplog.records), (
        "the class of the failed read must reach the log"
    )
    assert "Connection aborted" not in caplog.text


def test_a_reclaiming_replica_whose_recheck_fails_leaves_the_challenge_pending(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The live sequence, end to end through the real Agent and adapter.

    Replica B dispatches RESET_GPU on the idle node. Replica A reclaims the
    WAITING command while the Agent's pre-spawn challenge is up and its node
    read raises: it must record a deferral, deliver no permit and leave the
    challenge with the Agent. Replica B takes the command back, grants on
    evidence, and the reset runs exactly once.
    """

    _healthy_state, healthy, context = idle_reset_runtime()
    broken_state, broken, _context = idle_reset_runtime()
    monkeypatch.setattr(broken_state, "read_node", connection_aborted)
    hardware = FakeRunner()
    agent = node_action_executor(
        tmp_path,
        "ha004-reclaim.db",
        allowed_operations={WorkflowOperation.RESET_GPU},
        reset_enabled=True,
        now=None,
        runner=hardware,
        require_final_ownership=True,
        agent_generation=1,
        device_client_finder=no_device_clients,
        gpu_device_path_finder=lambda: {"GPU-a": "/dev/nvidia0"},
        device_client_samples=1,
    )
    gate = agent.ownership_gate
    assert gate is not None, "the owned Agent must enforce the final checkpoint"
    adapter = NodeActionWorkflowAdapter({}, SECRET, registry=LocalRegistry())
    with TestClient(create_node_agent_app(agent)) as client:
        monkeypatch.setattr(transport, "urlopen", transport_bridge(client))
        with stop_ownership_scope(healthy):
            dispatched = adapter.execute(context)
        assert dispatched.status is WorkflowStepStatus.WAITING, dispatched
        assert dispatched.details is not None
        command_id = dispatched.details["node_action_command_id"]

        with stop_ownership_scope(broken):
            deferred = settled(
                lambda: adapter.execute(context),
                lambda value: value.status is not WorkflowStepStatus.WAITING
                or value.details.get("ownership_recheck_deferred") is True,
            )
        assert deferred.status is WorkflowStepStatus.WAITING, deferred
        assert deferred.details is not None
        assert deferred.details["node_action_state"] == "OWNERSHIP_RECHECK_DEFERRED"
        assert deferred.details["reason"] == "STOP_OWNERSHIP_UNVERIFIABLE"
        assert deferred.details["refusal_cause"] == "RuntimeError"
        assert deferred.details["node_action_command_id"] == command_id
        assert deferred.details["node_action_accepted_nodes"] == ["node-a"]
        assert "outcome_unknown" not in deferred.details
        assert "manual_confirmation_required" not in deferred.details
        assert gate.challenge(command_id) is not None, (
            "the challenge must stay with the Agent for the next lease owner"
        )
        assert not any("--gpu-reset" in call for call in hardware.commands), (
            "no physical action before a granted permit"
        )

        with stop_ownership_scope(healthy):
            final = settled(
                lambda: adapter.execute(context),
                lambda value: value.status is not WorkflowStepStatus.WAITING,
            )
        assert final.status is WorkflowStepStatus.SUCCEEDED, final

        def read_result() -> Any:
            response = client.get(
                "/v1/node-actions/result", params=result_params(command_id)
            )
            assert response.status_code == 200, response.text
            return response.json()

        result = settled(read_result, lambda value: value["state"] != "PENDING")
    assert result["state"] == "SUCCEEDED", result
    assert len(result["result"]["details"]["physical_ownership_checks"]) == 1
    assert [call for call in hardware.commands if "--gpu-reset" in call] == [
        ["nvidia-smi", "--gpu-reset", "-i", "GPU-a"]
    ], "exactly one physical reset across the reclaim"


@pytest.mark.parametrize("accepted", [True, False])
def test_a_lease_hold_keeps_the_agents_pointer_once_the_node_accepted(
    monkeypatch: pytest.MonkeyPatch, accepted: bool
) -> None:
    """A hold outcome is posted as the command's record when a replica stops,
    so it must not claim ``node_action_not_started`` over a recorded acceptance:
    the control plane reads that key as proof that no mutation began."""

    _state, _kube, validator, context = stopped_runtime()
    command_id = f"{context.idempotency_key}/node-a/agent-1"
    if accepted:
        prior = WorkflowStepExecution(
            step_index=1,
            operation=WorkflowOperation.RESET_GPU,
            status=WorkflowStepStatus.WAITING,
            phase="official",
            details={
                "node_action_command_id": command_id,
                "node_action_state": "PENDING",
                "node_action_accepted_nodes": ["node-a"],
            },
        )
        context = replace(
            context,
            workflow=context.workflow.model_copy(
                update={"step_executions": [*context.workflow.step_executions, prior]}
            ),
        )
    adapter = NodeActionWorkflowAdapter({}, SECRET, registry=LocalRegistry())
    monkeypatch.setattr(
        transport,
        "urlopen",
        lambda *_a, **_kw: pytest.fail("no node action may start under a hold"),
    )
    token = active_lease_guard.set(lambda: SHUTDOWN_HOLD)
    try:
        with stop_ownership_scope(validator):
            held = adapter.execute(context)
    finally:
        active_lease_guard.reset(token)

    assert held.status is WorkflowStepStatus.WAITING
    assert held.details is not None
    assert held.details["node_action_state"] == "LEASE_LOST"
    assert held.details["reason"] == SHUTDOWN_HOLD
    if accepted:
        assert held.details["node_action_command_id"] == command_id
        assert held.details["node_action_accepted_nodes"] == ["node-a"]
        assert "node_action_not_started" not in held.details
    else:
        assert held.details["node_action_not_started"] is True
        assert "node_action_command_id" not in held.details


def test_an_unanswered_challenge_still_fails_closed_on_the_agents_deadline() -> None:
    """Deferring executors leave the wait to the Agent, which bounds it itself."""

    clock = Clock()
    ticks = iter([0.0])
    gate = NodeOwnershipGate(
        secret=KEY,
        boot_id="boot",
        now=lambda: clock.now,
        monotonic=lambda: next(ticks, 10_000.0),
    )

    with pytest.raises(OwnershipRefused) as refused:
        gate.require(owned_command())

    assert refused.value.action_details["reason"] == "OWNERSHIP_RECHECK_TIMEOUT"
    assert refused.value.action_details["node_action_not_started"] is True
    assert refused.value.action_details["safety_rejection"] is True
    assert gate.challenge("owned-command") is None, "the timed-out challenge is gone"


def test_a_signed_denial_is_irrevocable_which_is_why_inconclusive_must_defer() -> None:
    clock = Clock()
    gate = NodeOwnershipGate(
        secret=KEY, boot_id="boot", now=lambda: clock.now, monotonic=clock.monotonic
    )
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(gate.require, owned_command())
        assert clock.ready.wait(5), "the Agent must reach its wait"
        challenge = settled(
            lambda: gate.challenge("owned-command"), lambda value: value is not None
        )
        assert challenge is not None
        denial = sign_permit(
            challenge,
            KEY,
            allowed=False,
            reason="STOP_OWNERSHIP_UNVERIFIABLE",
            now=clock.now,
        )
        gate.authorize(denial)
        grant = sign_permit(challenge, KEY, allowed=True, reason="OK", now=clock.now)
        with pytest.raises(OwnershipRefused, match="OWNERSHIP_PERMIT_STALE"):
            gate.authorize(grant)
        with pytest.raises(OwnershipRefused, match="STOP_OWNERSHIP_UNVERIFIABLE"):
            future.result(timeout=5)

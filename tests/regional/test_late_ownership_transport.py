from __future__ import annotations

import io
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from urllib.error import HTTPError, URLError

import pytest

from gpu_fault.adapters import NodeActionWorkflowAdapter
from gpu_fault.adapters.kubernetes.stop_ownership import refusal, stop_ownership_scope
from gpu_fault.adapters.node_action import adapter as adapter_module
from gpu_fault.adapters.node_action import transport
from gpu_fault.execution import WorkflowStepOutcome
from gpu_fault.models import WorkflowStepStatus
from gpu_fault.node_agent.late_ownership import (
    OWNERSHIP_PROTOCOL,
    OwnershipChallenge,
    command_identity,
    current_ownership_challenge,
    permit_signature,
)
from gpu_fault.node_agent.protocol import (
    NodeActionCommand,
    NodeActionExecutionState,
    NodeActionResult,
    NodeActionStatus,
    NodeActionSubmission,
    SignedNodeAction,
    sign_node_action,
)
from tests.node_agent._support import SECRET
from tests.regional._late_ownership_runtime import stopped_runtime
from tests.regional.test_late_ownership_agent_boundary import LocalRegistry


def inputs():
    context = stopped_runtime()[-1]
    now = datetime.now(timezone.utc)
    command = NodeActionCommand(
        command_id=context.idempotency_key + "/node-a/agent-1",
        workflow_request_id=context.workflow.request_id,
        incident_id=context.incident.incident_id,
        fencing_token=1,
        operation=context.step.operation,
        node_id="node-a",
        agent_generation=1,
        gpu_uuids=context.step.gpu_uuids,
        issued_at=now,
        expires_at=now + timedelta(seconds=90),
        ownership_guard=OWNERSHIP_PROTOCOL,
    )
    envelope = SignedNodeAction(
        command=command, signature=sign_node_action(command, SECRET)
    )
    challenge = OwnershipChallenge(
        command_id=command.command_id,
        workflow_id=command.workflow_request_id,
        incident_id=command.incident_id,
        node_id=command.node_id,
        boot_id="boot",
        agent_generation=1,
        fencing_token=1,
        command_sha256=command_identity(command),
        nonce="a" * 64,
        sequence=1,
        boundary="AGENT_PRE_SPAWN",
        expires_at=command.expires_at,
    )
    pending = NodeActionSubmission(
        command_id=command.command_id,
        state=NodeActionExecutionState.PENDING,
        ownership_challenge=challenge,
    )
    adapter = NodeActionWorkflowAdapter({}, SECRET, registry=LocalRegistry())
    return adapter, envelope, pending


def exchange(adapter, envelope, pending, before):
    context = stopped_runtime()[-1]

    class Validator:
        def check(self, current):
            if current_ownership_challenge() is None:
                return None
            return (
                before()
                if before is not None
                else refusal("STOP_OWNERSHIP_VALIDATOR_UNAVAILABLE")
            )

    send = transport.urlopen

    def io_boundary(request, **kwargs):
        if request.get_method() == "GET":
            return io.BytesIO(pending.model_dump_json().encode())
        return send(request, **kwargs)

    with pytest.MonkeyPatch.context() as patched, stop_ownership_scope(Validator()):
        patched.setattr(transport, "urlopen", io_boundary)
        outcome = adapter.execute(context)
    return pending, outcome


@pytest.mark.parametrize(
    "field,value",
    [
        ("command_id", "other"),
        ("workflow_id", "other"),
        ("incident_id", "other"),
        ("node_id", "other"),
        ("agent_generation", 2),
        ("fencing_token", 2),
        ("command_sha256", "f" * 64),
        ("boundary", "AGENT_HANDLER_ENTRY"),
    ],
)
def test_wrong_native_challenge_identity_never_gets_a_permit(monkeypatch, field, value):
    adapter, envelope, pending = inputs()
    pending = pending.model_copy(
        update={
            "ownership_challenge": pending.ownership_challenge.model_copy(
                update={field: value}
            )
        }
    )
    calls = []
    monkeypatch.setattr(transport, "urlopen", lambda *a, **kw: calls.append(a))
    state, outcome = exchange(
        adapter, envelope, pending, lambda: calls.append("checked")
    )
    assert state is pending and calls == []
    assert outcome.status is WorkflowStepStatus.WAITING
    assert outcome.details["reason"] == "OWNERSHIP_CHALLENGE_MISMATCH"
    assert outcome.details["manual_confirmation_required"]
    assert outcome.details["outcome_unknown"] is True
    assert "node_action_not_started" not in outcome.details


@pytest.mark.parametrize("decision", ["allow", "deny", "lease", "unknown", "missing"])
def test_only_a_fresh_guard_result_is_signed_and_refusal_preserves_accepted_pointer(
    monkeypatch, decision
):
    adapter, envelope, pending = inputs()
    sent = []

    def send(request, **kwargs):
        signed = SignedNodeAction.model_validate_json(request.data)
        sent.append(signed)
        return io.BytesIO(
            NodeActionSubmission(
                command_id=envelope.command.command_id,
                state=NodeActionExecutionState.PENDING,
            )
            .model_dump_json()
            .encode()
        )

    monkeypatch.setattr(transport, "urlopen", send)

    def before():
        assert current_ownership_challenge() == pending.ownership_challenge
        if decision == "allow":
            return None
        if decision == "lease":
            return WorkflowStepOutcome.waiting(
                details={"node_action_state": "LEASE_LOST"}
            )
        if decision == "unknown":
            return WorkflowStepOutcome.failed(
                "not public", details={"reason": "invalid text"}
            )
        return refusal("STOP_OWNERSHIP_DRIFT")

    state, outcome = exchange(
        adapter, envelope, pending, None if decision == "missing" else before
    )
    assert state.command_id == envelope.command.command_id
    assert current_ownership_challenge() is None
    assert outcome.status is WorkflowStepStatus.WAITING
    assert outcome.details["node_action_command_id"] == envelope.command.command_id
    assert outcome.details["node_action_accepted_nodes"] == ["node-a"]
    if decision in {"allow", "deny"}:
        # A fresh pass and an observed violation are the two answers a
        # replica may sign; both carry the accepted pointer forward.
        assert len(sent) == 1
        permit = sent[0].ownership_permit
        assert permit.signature == permit_signature(permit, SECRET)
        assert permit.allowed is (decision == "allow")
    else:
        # A lost lease, a malformed verdict and a missing validator are not
        # evidence about the node: nothing is signed, and the challenge stays
        # with the Agent for the next lease owner to answer on evidence.
        assert sent == []
        assert outcome.details["node_action_state"] == "OWNERSHIP_RECHECK_DEFERRED"
        assert outcome.details["ownership_recheck_deferred"] is True
        assert (
            outcome.details["reason"]
            == {
                "lease": "OWNERSHIP_LEASE_LOST",
                "unknown": "STOP_OWNERSHIP_UNVERIFIABLE",
                "missing": "STOP_OWNERSHIP_VALIDATOR_UNAVAILABLE",
            }[decision]
        )
        assert "manual_confirmation_required" not in outcome.details
    if decision == "deny":
        assert outcome.details["reason"] == "OWNERSHIP_DENIAL_UNCONFIRMED"
        assert outcome.details["outcome_unknown"] is True
        assert outcome.details["manual_confirmation_required"] is True
    else:
        assert "outcome_unknown" not in outcome.details
    assert "node_action_not_started" not in outcome.details
    assert "invalid text" not in str(outcome)


@pytest.mark.parametrize("defect", ["http", "transport", "wrong-response"])
def test_permit_delivery_failure_never_becomes_permission_or_hardware_failure(
    monkeypatch, defect
):
    adapter, envelope, pending = inputs()

    def send(request, **kwargs):
        if defect == "http":
            raise HTTPError(
                request.full_url, 409, "local refusal", {}, io.BytesIO(b"{}")
            )
        if defect == "transport":
            raise URLError("local timeout")
        return io.BytesIO(
            NodeActionSubmission(
                command_id="other", state=NodeActionExecutionState.PENDING
            )
            .model_dump_json()
            .encode()
        )

    monkeypatch.setattr(transport, "urlopen", send)
    if defect == "http":
        _state, outcome = exchange(adapter, envelope, pending, lambda: None)
        assert outcome.status is WorkflowStepStatus.WAITING
        assert outcome.details["http_status"] == 409
        assert outcome.details["manual_confirmation_required"]
        assert outcome.details["ownership_permit_delivery_unknown"] is True
        assert "node_action_not_started" not in outcome.details
    elif defect == "transport":
        _state, outcome = exchange(adapter, envelope, pending, lambda: None)
        assert outcome.status is WorkflowStepStatus.WAITING
        assert outcome.details["ownership_permit_delivery_unknown"] is True
        assert outcome.details["node_action_command_id"] == envelope.command.command_id
    else:
        _state, outcome = exchange(adapter, envelope, pending, lambda: None)
        assert outcome.details["reason"] == "OWNERSHIP_RESPONSE_MISMATCH"
        assert outcome.details["manual_confirmation_required"]


@pytest.mark.parametrize("lifetime", ["future", "expired", "naive"])
def test_guarded_command_lifetime_cannot_extend_the_approved_workflow(lifetime):
    _state, _kube, validator, context = stopped_runtime()
    now = datetime.now(timezone.utc)
    deadline = (
        now + timedelta(seconds=15)
        if lifetime == "future"
        else now - timedelta(seconds=1)
    )
    if lifetime == "naive":
        deadline = now.replace(tzinfo=None)
    context = replace(
        context,
        workflow=context.workflow.model_copy(update={"lifetime_deadline_at": deadline}),
    )
    sent = []

    def sender(endpoint, envelope):
        sent.append(envelope)
        return NodeActionResult(
            command_id=envelope.command.command_id,
            operation=envelope.command.operation,
            status=NodeActionStatus.SUCCEEDED,
        )

    adapter = NodeActionWorkflowAdapter(
        {}, SECRET, registry=LocalRegistry(), sender=sender
    )
    with stop_ownership_scope(validator):
        outcome = adapter.execute(context)
    if lifetime == "future":
        assert len(sent) == 1 and sent[0].command.expires_at == deadline
        assert outcome.status is WorkflowStepStatus.SUCCEEDED
    else:
        assert sent == []
        assert outcome.status is WorkflowStepStatus.FAILED
        assert outcome.details["reason"] == (
            "OWNERSHIP_LIFETIME_UNKNOWN"
            if lifetime == "naive"
            else "OWNERSHIP_WORKFLOW_EXPIRED"
        )
        assert outcome.details["manual_confirmation_required"]


@pytest.mark.parametrize(
    "shape",
    [
        None,
        [],
        {},
        {"ownership_guard_protocol": "legacy"},
        {"ownership_guard_protocol": OWNERSHIP_PROTOCOL},
    ],
)
def test_deployed_capability_must_be_explicit_and_result_reads_never_submit(
    monkeypatch, shape
):
    import json

    adapter = NodeActionWorkflowAdapter({"node-a": "http://owned.invalid"}, SECRET)
    calls = []
    monkeypatch.setattr(
        adapter_module,
        "urlopen",
        lambda request, **kw: calls.append(request.get_method())
        or io.BytesIO(json.dumps(shape).encode()),
    )
    assert adapter.read_ownership_capability("local", "node-a") is (
        shape == {"ownership_guard_protocol": OWNERSHIP_PROTOCOL}
    )

    def absent(request, **kwargs):
        assert request.get_method() == "GET"
        raise HTTPError(request.full_url, 404, "local absence", {}, io.BytesIO(b"{}"))

    monkeypatch.setattr(transport, "urlopen", absent)
    assert adapter.read_action_result("local", "node-a", "unseen") is None
    assert calls == ["GET"]
    assert adapter.read_ownership_capability("local", "unknown") is False
    with pytest.raises(ValueError, match="unavailable"):
        adapter.read_action_result("local", "unknown", "unseen")

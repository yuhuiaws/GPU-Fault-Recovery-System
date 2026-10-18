"""Node-action transport edges under the ownership protocol and outside it.

A SUCCEEDED ledger row that claims the action never started is an unknown
outcome, not a success; a rejected submission whose body is not an object (or
cannot be read) is unconfirmed under ownership and propagates without it; a
transport exception is a plain failure without a guard and an unknown
outcome with one; a lost lease keeps the accepted pointer and intent of the
action already running; and the step's own history decides whether a recorded
action is resumed, mismatched or ignored.
"""

from __future__ import annotations

import io
from dataclasses import replace
from typing import Any
from urllib.error import HTTPError

import pytest

from gpu_fault.adapters import NodeActionWorkflowAdapter
from gpu_fault.adapters.common import NODE_ACTION_ACCEPTED_NODES_KEY
from gpu_fault.adapters.kubernetes.stop_ownership import stop_ownership_scope
from gpu_fault.adapters.node_action import transport
from gpu_fault.adapters.node_action.lease_guard import active_lease_guard
from gpu_fault.execution import WorkflowStepContext
from gpu_fault.models import (
    WorkflowOperation,
    WorkflowStepExecution,
    WorkflowStepStatus,
)
from gpu_fault.node_agent.protocol import (
    NodeActionExecutionState,
    NodeActionResult,
    NodeActionStatus,
    NodeActionSubmission,
)
from tests.node_agent._support import SECRET
from tests.regional._late_ownership_runtime import stopped_runtime
from tests.regional.test_late_ownership_agent_boundary import LocalRegistry

NODE = "node-a"


class _Validator:
    """A stop-ownership validator that has nothing to object to."""

    def check(self, _current: Any) -> None:
        return None


def _adapter(**overrides: Any) -> NodeActionWorkflowAdapter:
    settings: dict[str, Any] = {"registry": LocalRegistry(), **overrides}
    return NodeActionWorkflowAdapter({}, SECRET, **settings)


def _context() -> WorkflowStepContext:
    return stopped_runtime()[-1]


def _command_id(context: WorkflowStepContext) -> str:
    return f"{context.idempotency_key}/{NODE}/agent-1"


def _body(submission: NodeActionSubmission) -> io.BytesIO:
    return io.BytesIO(submission.model_dump_json().encode())


def _http_error(status: int, body: bytes | None) -> HTTPError:
    return HTTPError(
        "http://node-local", status, "rejected", {}, io.BytesIO(body or b"")
    )


class _UnreadableHTTPError(HTTPError):
    def read(self, *_args: Any) -> bytes:
        raise OSError("connection reset while reading the rejection")


def _wire(monkeypatch: pytest.MonkeyPatch, on_get: Any, on_post: Any) -> list[str]:
    methods: list[str] = []

    def urlopen(request: Any, **_kwargs: Any) -> Any:
        methods.append(request.get_method())
        return on_get() if request.get_method() == "GET" else on_post()

    monkeypatch.setattr(transport, "urlopen", urlopen)
    return methods


def _not_found() -> Any:
    raise _http_error(404, b"")


def _execute_owned(adapter: NodeActionWorkflowAdapter, context: WorkflowStepContext):
    with stop_ownership_scope(_Validator()):
        return adapter.execute(context)


def test_a_succeeded_row_that_says_it_never_started_is_an_unknown_outcome(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context()
    command_id = _command_id(context)
    row = NodeActionSubmission(
        command_id=command_id,
        state=NodeActionExecutionState.SUCCEEDED,
        result=NodeActionResult(
            command_id=command_id,
            operation=WorkflowOperation.RESET_GPU,
            status=NodeActionStatus.SUCCEEDED,
            details={"node_action_not_started": True},
        ),
    )
    methods = _wire(monkeypatch, lambda: _body(row), _not_found)

    outcome = _execute_owned(_adapter(), context)

    assert outcome.status is WorkflowStepStatus.WAITING, outcome
    assert outcome.details["reason"] == "OWNERSHIP_RESPONSE_MISMATCH"
    assert outcome.details["node_action_response_unknown"] is True
    assert methods == ["GET"], "a contradictory row was followed by a submission"


@pytest.mark.parametrize("body", [b"[]", b"42", b'{"detail": "not an object"}', None])
def test_an_unconfirmed_rejection_without_an_object_body_is_an_unknown_response(
    monkeypatch: pytest.MonkeyPatch, body: bytes | None
) -> None:
    context = _context()

    def rejected() -> Any:
        if body is None:
            raise _UnreadableHTTPError("http://node-local", 409, "rejected", {}, None)
        raise _http_error(409, body)

    methods = _wire(monkeypatch, _not_found, rejected)

    outcome = _execute_owned(_adapter(), context)

    assert outcome.status is WorkflowStepStatus.WAITING, outcome
    assert outcome.details["reason"] == "OWNERSHIP_RESPONSE_UNAVAILABLE"
    assert outcome.details["http_status"] == 409
    assert outcome.details["node_action_response_unknown"] is True
    assert methods == ["GET", "POST"]


def test_an_unreadable_rejection_propagates_when_the_submission_is_confirmed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without the ownership guard the rejection is the agent's verdict, and a
    verdict that cannot be read is not reinterpreted as anything else."""

    context = _context()

    def rejected() -> Any:
        raise _UnreadableHTTPError("http://node-local", 409, "rejected", {}, None)

    _wire(monkeypatch, _not_found, rejected)

    with pytest.raises(OSError, match="connection reset while reading"):
        _adapter().execute(context)


def test_a_transport_exception_fails_the_step_without_a_guard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context()

    def exploded() -> Any:
        raise RuntimeError("socket closed mid-handshake")

    _wire(monkeypatch, _not_found, exploded)

    outcome = _adapter().execute(context)

    assert outcome.status is WorkflowStepStatus.FAILED, outcome
    assert outcome.error == (
        f"node agent {NODE} request failed: RuntimeError: socket closed mid-handshake"
    )


def test_a_sender_exception_under_the_guard_is_an_unverifiable_outcome() -> None:
    context = _context()

    def sender(_endpoint: str, _envelope: Any) -> NodeActionResult:
        raise RuntimeError("sender crashed after the envelope left")

    outcome = _execute_owned(_adapter(sender=sender), context)

    assert outcome.status is WorkflowStepStatus.WAITING, outcome
    assert outcome.details["reason"] == "OWNERSHIP_PROTOCOL_UNVERIFIABLE"
    assert outcome.details["node_action_response_unknown"] is True
    assert outcome.details["manual_confirmation_required"] is True


def _with_history(
    context: WorkflowStepContext, *executions: WorkflowStepExecution
) -> WorkflowStepContext:
    workflow = context.workflow.model_copy(
        update={"step_executions": [*context.workflow.step_executions, *executions]}
    )
    return replace(context, workflow=workflow)


def _reset_execution(
    status: WorkflowStepStatus, details: dict[str, Any], *, phase: str = "official"
) -> WorkflowStepExecution:
    return WorkflowStepExecution(
        step_index=1,
        operation=WorkflowOperation.RESET_GPU,
        status=status,
        phase=phase,
        details=details,
    )


def test_a_lost_lease_keeps_the_pointer_and_intent_of_the_accepted_action(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """First the agent accepts the action (PENDING); then the lease is lost
    while the step is re-dispatched. The hold must name the command that is
    running and its intent so the next lease holder resumes exactly it."""

    context = _context()
    command_id = _command_id(context)
    accepted = NodeActionSubmission(
        command_id=command_id, state=NodeActionExecutionState.PENDING
    )
    _wire(monkeypatch, lambda: _body(accepted), _not_found)
    first = _execute_owned(_adapter(), context)
    assert first.status is WorkflowStepStatus.WAITING, first
    assert first.details[NODE_ACTION_ACCEPTED_NODES_KEY] == [NODE]
    intent = first.details["node_action_intent_sha256"]
    assert isinstance(intent, str) and len(intent) == 64

    resumed = _with_history(
        context, _reset_execution(WorkflowStepStatus.WAITING, dict(first.details))
    )
    token = active_lease_guard.set(lambda: "lease renewal failed 3 times")
    try:
        held = _execute_owned(_adapter(), resumed)
    finally:
        active_lease_guard.reset(token)

    assert held.status is WorkflowStepStatus.WAITING, held
    assert held.details["node_action_state"] == "LEASE_LOST"
    assert held.details["node_action_command_id"] == command_id
    assert held.details[NODE_ACTION_ACCEPTED_NODES_KEY] == [NODE]
    assert held.details["node_action_intent_sha256"] == intent
    assert "node_action_not_started" not in held.details


def test_an_unknown_response_recorded_in_another_phase_is_an_intent_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context()
    command_id = _command_id(context)
    methods = _wire(monkeypatch, _not_found, _not_found)
    history = _with_history(
        context,
        _reset_execution(
            WorkflowStepStatus.WAITING,
            {
                "node_action_command_id": command_id,
                "node_action_response_unknown": True,
                "waiting_node": NODE,
            },
            phase="safety",
        ),
    )

    outcome = _execute_owned(_adapter(), history)

    assert outcome.status is WorkflowStepStatus.WAITING, outcome
    assert outcome.details["reason"] == "OWNERSHIP_INTENT_MISMATCH"
    assert outcome.details["node_action_intent_mismatch"] is True
    assert methods == [], "a mismatched intent must not reach the agent at all"


def test_an_accepted_action_recorded_in_another_phase_is_an_intent_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context()
    command_id = _command_id(context)
    methods = _wire(monkeypatch, _not_found, _not_found)
    history = _with_history(
        context,
        _reset_execution(
            WorkflowStepStatus.WAITING,
            {
                "node_action_command_id": command_id,
                NODE_ACTION_ACCEPTED_NODES_KEY: [NODE],
            },
            phase="safety",
        ),
    )

    outcome = _execute_owned(_adapter(), history)

    assert outcome.status is WorkflowStepStatus.WAITING, outcome
    assert outcome.details["reason"] == "OWNERSHIP_INTENT_MISMATCH"
    assert outcome.details["node_action_intent_mismatch"] is True
    assert methods == [], "a mismatched intent must not reach the agent at all"


def _succeeded_row(command_id: str) -> NodeActionSubmission:
    return NodeActionSubmission(
        command_id=command_id,
        state=NodeActionExecutionState.SUCCEEDED,
        result=NodeActionResult(
            command_id=command_id,
            operation=WorkflowOperation.RESET_GPU,
            status=NodeActionStatus.SUCCEEDED,
            details={"physical_ownership_checks": []},
        ),
    )


def test_a_waiting_record_of_another_command_does_not_resume_an_action(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context()
    done = _succeeded_row(_command_id(context))
    methods = _wire(monkeypatch, _not_found, lambda: _body(done))
    history = _with_history(
        context,
        _reset_execution(
            WorkflowStepStatus.WAITING,
            {"node_action_command_id": "workflow-other/1/RESET_GPU/node-a"},
        ),
    )

    outcome = _execute_owned(_adapter(), history)

    assert outcome.status is WorkflowStepStatus.SUCCEEDED, outcome
    assert methods == ["GET", "POST"], methods


def test_a_settled_phase_hides_the_older_records_behind_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A newer SUCCEEDED record settles the phase, so the accepted WAITING
    record behind it is not resumed; the step submits afresh."""

    context = _context()
    command_id = _command_id(context)
    methods = _wire(monkeypatch, _not_found, lambda: _body(_succeeded_row(command_id)))
    history = _with_history(
        context,
        _reset_execution(
            WorkflowStepStatus.WAITING,
            {
                "node_action_command_id": command_id,
                NODE_ACTION_ACCEPTED_NODES_KEY: [NODE],
            },
        ),
        _reset_execution(WorkflowStepStatus.SUCCEEDED, {}),
    )

    outcome = _execute_owned(_adapter(), history)

    assert outcome.status is WorkflowStepStatus.SUCCEEDED, outcome
    assert methods == ["GET", "POST"], methods


class _UnreachableRegistry(LocalRegistry):
    def endpoint(self, cluster_id: str, node_id: str) -> tuple[None, int]:
        return None, 1


def test_an_unknown_delivery_whose_agent_has_no_endpoint_stays_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The first dispatch left the delivery unknown; on re-dispatch the agent
    has no endpoint any more, so nothing can be read and the step stays an
    unknown outcome rather than being resubmitted or failed."""

    context = _context()

    def rejected() -> Any:
        raise _http_error(409, b"[]")

    _wire(monkeypatch, _not_found, rejected)
    first = _execute_owned(_adapter(), context)
    assert first.details["node_action_response_unknown"] is True, first
    history = _with_history(
        context, _reset_execution(WorkflowStepStatus.WAITING, dict(first.details))
    )
    methods = _wire(monkeypatch, _not_found, _not_found)

    outcome = _execute_owned(_adapter(registry=_UnreachableRegistry()), history)

    assert outcome.status is WorkflowStepStatus.WAITING, outcome
    assert outcome.details["reason"] == "OWNERSHIP_RESULT_UNAVAILABLE"
    assert outcome.details["node_action_response_unknown"] is True
    assert outcome.details["node_action_command_id"] == _command_id(context)
    assert methods == [], "an agent without an endpoint was still contacted"

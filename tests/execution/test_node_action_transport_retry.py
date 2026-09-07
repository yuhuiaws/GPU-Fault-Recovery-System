"""Control-plane side of a node action that did not finish on the first try.

``NodeActionTransportMixin._send`` polls the agent ledger and only submits when
the agent has never seen the command. A ``FAILED retryable=True`` row therefore
used to be polled forever: the agent supports ``attempt + 1`` on re-submit, but
nothing ever re-submitted, so the step spun until its 600 s bound.

The HTTP layer is faked at ``urlopen`` because the retry decision is made on
what came back over the wire; the adapter, the envelope signing and the result
folding are the real ones.
"""

from __future__ import annotations

import io
import json
from typing import Any
from urllib.error import HTTPError

import pytest

from gpu_fault.adapters import NodeActionWorkflowAdapter
from gpu_fault.adapters.node_action.lease_guard import active_lease_guard
from gpu_fault.execution import WorkflowStepContext
from gpu_fault.models import (
    WorkflowExecutionRequest,
    WorkflowOperation,
    WorkflowStepStatus,
)
from gpu_fault.node_agent import (
    NodeActionExecutionState,
    NodeActionResult,
    NodeActionStatus,
    NodeActionSubmission,
)
from tests._builders import fault_incident, workflow_request, workflow_step

SECRET = "s" * 32
ENDPOINT = "http://node-a:9099"
FENCING_TOKEN = 3


def step_context(
    adapter: NodeActionWorkflowAdapter,
    operation: WorkflowOperation = WorkflowOperation.RESET_GPU,
    *,
    idempotency_key: str = "workflow/step",
) -> WorkflowStepContext:
    step = workflow_step(
        operation, adapter.owner, node_ids=["node-a"], gpu_uuids=["GPU-a"]
    )
    return WorkflowStepContext(
        workflow=workflow_request(
            "workflow-a",
            "incident-a",
            fencing_token=FENCING_TOKEN,
            official_steps=[step],
        ),
        incident=fault_incident("incident-a", "event-a", fencing_token=FENCING_TOKEN),
        step=step,
        step_index=0,
        request=WorkflowExecutionRequest(expected_fencing_token=FENCING_TOKEN),
        idempotency_key=idempotency_key,
    )


class FakeResponse:
    def __init__(self, body: bytes) -> None:
        self.body = body

    def read(self) -> bytes:
        return self.body

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, *_exc: object) -> None:
        return None


def submission(
    command_id: str,
    state: NodeActionExecutionState,
    result: NodeActionResult | None = None,
) -> bytes:
    return (
        NodeActionSubmission(command_id=command_id, state=state, result=result)
        .model_dump_json()
        .encode()
    )


def failed_result(command_id: str, *, attempt: int, retryable: bool = True) -> bytes:
    return submission(
        command_id,
        NodeActionExecutionState.FAILED,
        NodeActionResult(
            command_id=command_id,
            operation=WorkflowOperation.RESET_GPU,
            status=NodeActionStatus.FAILED,
            error="TimeoutExpired: nvidia-smi --gpu-reset timed out",
            retryable=retryable,
            attempt=attempt,
        ),
    )


class FakeAgentWire:
    """Answers the poll with a canned ledger row and records every submit."""

    def __init__(self, poll_body: bytes, submit_body: bytes | None = None) -> None:
        self.poll_body = poll_body
        self.submit_body = submit_body
        self.polls = 0
        self.submits: list[Any] = []

    def __call__(
        self, request: Any, timeout: float | None = None, *, ssl_context: Any = None
    ) -> FakeResponse:
        if request.get_method() == "GET":
            self.polls += 1
            return FakeResponse(self.poll_body)
        self.submits.append(request)
        if self.submit_body is None:
            raise AssertionError("the agent was not expected to receive a submit")
        return FakeResponse(self.submit_body)


def wire(monkeypatch: pytest.MonkeyPatch, fake: FakeAgentWire) -> FakeAgentWire:
    monkeypatch.setattr("gpu_fault.adapters.node_action.transport.urlopen", fake)
    return fake


def test_a_retryable_ledger_failure_is_resubmitted_with_the_same_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The agent keeps the attempt counter; the control plane must ask again.

    Same ``command_id`` and ``Idempotency-Key``: the agent's ledger row is the
    idempotency record, and the re-submit is what moves it to ``attempt + 1``.
    """

    command_id = "workflow/step/node-a"
    fake = wire(
        monkeypatch,
        FakeAgentWire(
            failed_result(command_id, attempt=1),
            submission(command_id, NodeActionExecutionState.PENDING),
        ),
    )
    adapter = NodeActionWorkflowAdapter({"node-a": ENDPOINT}, SECRET)

    outcome = adapter.execute(step_context(adapter))

    assert outcome.status is WorkflowStepStatus.WAITING
    assert outcome.details["node_action_state"] == "PENDING"
    assert outcome.details["node_action_resubmitted"] is True
    assert outcome.details["node_action_attempt"] == 2
    assert len(fake.submits) == 1
    submitted = fake.submits[0]
    assert submitted.full_url == ENDPOINT + "/v1/node-actions/submit"
    assert submitted.get_header("Idempotency-key") == command_id
    assert json.loads(submitted.data)["command"]["command_id"] == command_id


def test_resubmits_stop_at_the_retry_limit_with_a_terminal_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Three re-submits is the default; the fourth attempt's failure is final."""

    command_id = "workflow/step/node-a"
    fake = wire(monkeypatch, FakeAgentWire(failed_result(command_id, attempt=4)))
    adapter = NodeActionWorkflowAdapter({"node-a": ENDPOINT}, SECRET)

    outcome = adapter.execute(step_context(adapter))

    assert outcome.status is WorkflowStepStatus.FAILED
    assert fake.submits == []
    assert outcome.details["node_action_retry_exhausted"] is True
    assert outcome.details["node_action_attempts"] == 4
    assert "TimeoutExpired" in outcome.details["node_action_last_error"]
    assert "4 attempt" in (outcome.error or "")


def test_the_retry_limit_is_read_from_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GPU_FAULT_NODE_ACTION_RETRY_LIMIT", "1")
    monkeypatch.setenv(
        "GPU_FAULT_NODE_AGENT_ENDPOINTS", json.dumps({"node-a": ENDPOINT})
    )
    monkeypatch.setenv("GPU_FAULT_NODE_ACTION_SECRET", SECRET)
    command_id = "workflow/step/node-a"
    fake = wire(monkeypatch, FakeAgentWire(failed_result(command_id, attempt=2)))

    adapter = NodeActionWorkflowAdapter.from_environment()
    outcome = adapter.execute(step_context(adapter))

    assert adapter.node_action_retry_limit == 1
    assert outcome.status is WorkflowStepStatus.FAILED
    assert outcome.details["node_action_attempts"] == 2
    assert fake.submits == []


def test_a_non_retryable_ledger_failure_is_not_resubmitted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    command_id = "workflow/step/node-a"
    fake = wire(
        monkeypatch,
        FakeAgentWire(failed_result(command_id, attempt=1, retryable=False)),
    )
    adapter = NodeActionWorkflowAdapter({"node-a": ENDPOINT}, SECRET)

    outcome = adapter.execute(step_context(adapter))

    assert outcome.status is WorkflowStepStatus.FAILED
    assert fake.submits == []
    assert "node_action_retry_exhausted" not in outcome.details


def http_error(status: int, detail: dict[str, Any] | None = None) -> HTTPError:
    body = json.dumps({"detail": detail} if detail is not None else {}).encode()
    return HTTPError(
        ENDPOINT + "/v1/node-actions/submit", status, "error", {}, io.BytesIO(body)
    )


def adapter_raising(error: Exception) -> NodeActionWorkflowAdapter:
    def sender(_endpoint: str, _envelope: Any) -> NodeActionResult:
        raise error

    return NodeActionWorkflowAdapter({"node-a": ENDPOINT}, SECRET, sender=sender)


@pytest.mark.parametrize("status", [500, 502, 503, 408, 429])
def test_server_side_and_throttling_statuses_keep_the_step_waiting(status: int) -> None:
    """A 5xx after the operation ran is not the operation failing.

    The agent's ledger write can fail after a successful reset; the next poll
    finds the row or re-submits. Failing the step here would mark the GPU
    unrecoverable for a transient agent-side error.
    """

    adapter = adapter_raising(http_error(status))

    outcome = adapter.execute(step_context(adapter))

    assert outcome.status is WorkflowStepStatus.WAITING
    assert outcome.details["node_action_transport_retry"] is True
    assert outcome.details["http_status"] == status
    assert outcome.details["node_action_command_id"] == "workflow/step/node-a"


@pytest.mark.parametrize(
    ("status", "code"),
    [
        (409, "STALE_AGENT_GENERATION"),
        (410, "COMMAND_EXPIRED"),
        (409, "STALE_FENCING_TOKEN"),
    ],
)
def test_a_rejection_that_wants_a_new_command_holds_and_says_so(
    status: int, code: str
) -> None:
    adapter = adapter_raising(
        http_error(
            status,
            {
                "code": code,
                "message": "rejected",
                "retryable": True,
                "requires_new_command": True,
            },
        )
    )

    outcome = adapter.execute(step_context(adapter))

    assert outcome.status is WorkflowStepStatus.WAITING
    assert outcome.details["node_action_state"] == "NEW_COMMAND_REQUIRED"
    assert outcome.details["node_action_requires_new_command"] is True
    assert outcome.details["node_action_error_code"] == code
    assert outcome.details["http_status"] == status


def test_a_retryable_rejection_without_a_new_command_is_a_transport_retry() -> None:
    """The agent has not heard its generation yet; the same command is fine."""

    adapter = adapter_raising(
        http_error(
            409,
            {
                "code": "AGENT_GENERATION_UNKNOWN",
                "message": "agent generation is not known yet",
                "retryable": True,
                "requires_new_command": False,
            },
        )
    )

    outcome = adapter.execute(step_context(adapter))

    assert outcome.status is WorkflowStepStatus.WAITING
    assert outcome.details["node_action_transport_retry"] is True
    assert outcome.details["node_action_error_code"] == "AGENT_GENERATION_UNKNOWN"


@pytest.mark.parametrize(
    ("status", "code"),
    [
        (401, "INVALID_SIGNATURE"),
        (403, "OPERATION_NOT_ALLOWED"),
        (422, "TARGET_NODE_MISMATCH"),
    ],
)
def test_other_client_errors_still_fail_the_step(status: int, code: str) -> None:
    adapter = adapter_raising(
        http_error(
            status,
            {
                "code": code,
                "message": "refused",
                "retryable": False,
                "requires_new_command": False,
            },
        )
    )

    outcome = adapter.execute(step_context(adapter))

    assert outcome.status is WorkflowStepStatus.FAILED
    assert outcome.details["node_action_error_code"] == code
    assert outcome.details["http_status"] == status


def test_a_lost_lease_stops_new_node_actions_from_being_sent() -> None:
    """The regional executor sets the guard; the adapter must consult it."""

    sent: list[str] = []

    def sender(_endpoint: str, envelope: Any) -> NodeActionResult:
        sent.append(envelope.command.command_id)
        raise AssertionError("no node action may start under a lost lease")

    adapter = NodeActionWorkflowAdapter({"node-a": ENDPOINT}, SECRET, sender=sender)
    token = active_lease_guard.set(lambda: "lease renewal failed 3 times")
    try:
        outcome = adapter.execute(step_context(adapter))
    finally:
        active_lease_guard.reset(token)

    assert outcome.status is WorkflowStepStatus.WAITING
    assert outcome.details["node_action_state"] == "LEASE_LOST"
    assert outcome.details["reason"] == "lease renewal failed 3 times"
    assert sent == []

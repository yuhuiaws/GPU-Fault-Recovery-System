from __future__ import annotations

import io
import json
import ssl
from typing import Any
from urllib.error import HTTPError

import pytest

from gpu_fault.adapters import NodeActionWorkflowAdapter
from gpu_fault.adapters.node_action import transport
from gpu_fault.models import WorkflowOperation, WorkflowStepStatus
from gpu_fault.node_agent import (
    NodeActionExecutionState,
    NodeActionResult,
    NodeActionStatus,
    NodeActionSubmission,
    SignedNodeAction,
)
from tests.execution.test_node_action_round_trips import CountingFleetRegistry
from tests.execution.test_node_action_transport_retry import (
    ENDPOINT,
    SECRET,
    FakeResponse,
    step_context,
)
from tests.regional._cov95_runtime_support import offline_runtime as offline_runtime


@pytest.mark.parametrize("status", [403, 503])
@pytest.mark.parametrize("body", [None, [], True, "proxy refused", 7])
def test_nonobject_json_rejection_keeps_http_status_and_never_submits(
    monkeypatch: pytest.MonkeyPatch, status: int, body: Any
) -> None:
    calls = []

    def reject(request: Any, **kwargs: Any) -> Any:
        calls.append((request.get_method(), request.full_url))
        raise HTTPError(
            request.full_url,
            status,
            "synthetic proxy refusal",
            {},
            io.BytesIO(json.dumps(body).encode()),
        )

    monkeypatch.setattr(transport, "urlopen", reject)
    adapter = NodeActionWorkflowAdapter({"node-a": ENDPOINT}, SECRET)
    outcome = adapter.execute(step_context(adapter))
    assert outcome.status is (
        WorkflowStepStatus.FAILED if status == 403 else WorkflowStepStatus.WAITING
    )
    assert outcome.details["http_status"] == status
    assert outcome.details["node_action_error_code"] == "HTTP_REJECTION"
    assert len(calls) == 1
    assert calls[0][0] == "GET"
    assert calls[0][1].startswith(ENDPOINT + "/v1/node-actions/result?"), calls


@pytest.mark.parametrize(
    "state",
    [
        NodeActionExecutionState.SUCCEEDED,
        NodeActionExecutionState.FAILED,
        NodeActionExecutionState.INTERRUPTED,
    ],
)
def test_terminal_envelope_without_a_result_is_not_success_or_permission_to_resubmit(
    monkeypatch: pytest.MonkeyPatch, state: NodeActionExecutionState
) -> None:
    calls = []

    def response(request: Any, **kwargs: Any) -> FakeResponse:
        calls.append(request.get_method())
        return FakeResponse(
            NodeActionSubmission(command_id="workflow/step/node-a", state=state)
            .model_dump_json()
            .encode()
        )

    monkeypatch.setattr(transport, "urlopen", response)
    adapter = NodeActionWorkflowAdapter({"node-a": ENDPOINT}, SECRET)
    outcome = adapter.execute(step_context(adapter))
    assert outcome.status is WorkflowStepStatus.FAILED
    assert "completed without a result" in outcome.error
    assert calls == ["GET"]


def test_missing_endpoint_fails_before_any_transport_call() -> None:
    calls = []
    adapter = NodeActionWorkflowAdapter(
        {"node-b": "http://node-b:9099"},
        SECRET,
        sender=lambda *args: calls.append(args),
    )
    outcome = adapter.execute(step_context(adapter))
    assert outcome.status is WorkflowStepStatus.FAILED
    assert "no node action endpoint for node-a" in outcome.error
    assert calls == []


def test_nonfleet_https_reuses_verified_system_trust(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contexts = []
    calls = []

    def default_context(**kwargs: Any) -> ssl.SSLContext:
        assert kwargs == {}
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        contexts.append(context)
        return context

    def send(endpoint: str, signed: SignedNodeAction) -> NodeActionResult:
        calls.append(endpoint)
        return NodeActionResult(
            command_id=signed.command.command_id,
            operation=signed.command.operation,
            status=NodeActionStatus.SUCCEEDED,
        )

    monkeypatch.setattr(transport.ssl, "create_default_context", default_context)
    adapter = NodeActionWorkflowAdapter(
        {"node-a": "https://node-a:9099"}, SECRET, sender=send
    )
    for index in range(2):
        result = adapter.execute(
            step_context(adapter, idempotency_key=f"workflow/step-{index}")
        )
        assert result.status is WorkflowStepStatus.SUCCEEDED
    assert len(contexts) == 1
    assert contexts[0].verify_mode == ssl.CERT_REQUIRED
    assert contexts[0].check_hostname is True
    assert calls == ["https://node-a:9099"] * 2


def test_missing_pinned_certificate_cannot_fall_back_to_system_trust(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = CountingFleetRegistry({"node-a": "https://node-a:9099"})
    registry.certificates.clear()
    calls = []
    monkeypatch.setattr(
        transport.ssl, "create_default_context", lambda **kwargs: calls.append(kwargs)
    )
    adapter = NodeActionWorkflowAdapter(
        {}, SECRET, registry=registry, sender=lambda *args: calls.append(args)
    )
    outcome = adapter.execute(
        step_context(adapter, WorkflowOperation.TRIGGER_HEALTH_SNAPSHOT)
    )
    assert outcome.status is WorkflowStepStatus.FAILED
    assert "TLS certificate is unavailable" in outcome.error
    assert calls == []

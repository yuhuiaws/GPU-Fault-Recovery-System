from __future__ import annotations

import io
import json
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from email.message import Message
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlsplit
from urllib.request import Request

import pytest
from fastapi.testclient import TestClient

from gpu_fault.adapters import NodeActionWorkflowAdapter
from gpu_fault.adapters.common import node_action_accepted_nodes
from gpu_fault.adapters.kubernetes.stop_ownership import stop_ownership_scope
from gpu_fault.adapters.node_action import transport
from gpu_fault.execution import WorkflowStepContext, WorkflowStepOutcome, step_bounds
from gpu_fault.execution.config import ProductionExecutorConfig
from gpu_fault.models import (
    RecoveryAction,
    WorkflowExecutionRequest,
    WorkflowOperation,
    WorkflowRequest,
    WorkflowStatus,
    WorkflowStepStatus,
)
from gpu_fault.node_agent.app import create_node_agent_app
from gpu_fault.node_agent.executor import NodeActionExecutor
from gpu_fault.node_agent.protocol import (
    NodeActionExecutionState,
    NodeActionResult,
    NodeActionStatus,
    NodeActionSubmission,
    SignedNodeAction,
)
from gpu_fault.orchestration.escalation import HardwareEscalationService
from tests._builders import (
    active_workflow_executor,
    build_store,
    workflow_step,
    workflow_step_execution,
)
from tests.execution._support import FakeAdapter, workflow_state
from tests.node_agent._support import FakeRunner, no_device_clients
from tests.node_agent._support import (
    node_action_executor as node_action_executor_fixture,
)
from tests.regional._late_ownership_runtime import (
    stopped_runtime as stopped_runtime_fixture,
)
from tests.regional.test_late_ownership_agent_boundary import (
    transport_bridge as transport_bridge_fixture,
)
from tests.regional.test_late_ownership_agent_gate_edges import (
    owned_agent_io as owned_agent_io,
)
from tests.regional.test_late_ownership_transport import inputs as inputs_fixture

stopped_runtime: Callable[[], tuple[Any, Any, Any, WorkflowStepContext]] = (
    stopped_runtime_fixture
)
inputs: Callable[
    [], tuple[NodeActionWorkflowAdapter, SignedNodeAction, NodeActionSubmission]
] = inputs_fixture
node_action_executor: Callable[..., NodeActionExecutor] = node_action_executor_fixture
transport_bridge: Callable[[TestClient], Callable[..., io.BytesIO]] = (
    transport_bridge_fixture
)
Reply = bytes | Exception


class Wire:
    def __init__(self, replies: list[Reply]) -> None:
        self.replies = deque(replies)
        self.requests: list[Request] = []

    def __call__(self, request: Request, **kwargs: Any) -> io.BytesIO:
        self.requests.append(request)
        assert self.replies, "an unresolved command must not cause an extra request"
        reply = self.replies.popleft()
        if isinstance(reply, Exception):
            raise reply
        return io.BytesIO(reply)

    def methods(self) -> list[str]:
        return [request.get_method() for request in self.requests]


@dataclass
class Target:
    adapter: NodeActionWorkflowAdapter
    envelope: SignedNodeAction
    pending: NodeActionSubmission
    context: WorkflowStepContext
    validator: Any
    state: Any

    @property
    def command_id(self) -> str:
        return self.envelope.command.command_id

    def execute(self) -> WorkflowStepOutcome:
        with stop_ownership_scope(self.validator):
            return self.adapter.execute(self.context)

    def remember(self, outcome: WorkflowStepOutcome) -> None:
        self.context = replace(
            self.context,
            workflow=step_bounds.record_attempt(
                self.context.workflow,
                self.context.step,
                self.context.step_index,
                outcome,
            ),
        )

    def terminal(
        self,
        status: NodeActionStatus = NodeActionStatus.SUCCEEDED,
        *,
        retryable: bool = False,
        details: dict[str, Any] | None = None,
    ) -> NodeActionSubmission:
        return NodeActionSubmission(
            command_id=self.command_id,
            state=NodeActionExecutionState(status.value),
            result=NodeActionResult(
                command_id=self.command_id,
                operation=self.envelope.command.operation,
                status=status,
                retryable=retryable,
                details=details or {},
            ),
        )


@pytest.fixture
def target() -> Target:
    adapter, envelope, pending = inputs()
    state, _kube, validator, context = stopped_runtime()
    return Target(adapter, envelope, pending, context, validator, state)


def encoded(state: NodeActionSubmission) -> bytes:
    return state.model_dump_json().encode()


def submitted(request: Request) -> SignedNodeAction:
    assert isinstance(request.data, bytes), (
        "Node Action bodies must be serialized bytes"
    )
    return SignedNodeAction.model_validate_json(request.data)


def http_error(status: int, detail: dict[str, Any] | None = None) -> HTTPError:
    return HTTPError(
        "http://node-local/v1/node-actions/submit",
        status,
        "hermetic response",
        Message(),
        io.BytesIO(json.dumps({"detail": detail or {}}).encode()),
    )


def assert_unknown(
    target: Target, outcome: WorkflowStepOutcome, *, accepted: bool, permit: bool
) -> None:
    assert outcome.status is WorkflowStepStatus.WAITING
    assert outcome.adapter_operation_id == target.context.idempotency_key
    assert outcome.details["node_action_state"] == "PENDING"
    assert outcome.details["node_action_command_id"] == target.command_id
    assert outcome.details["node_action_response_unknown"] is True
    assert outcome.details["outcome_unknown"] is True
    assert outcome.details["manual_confirmation_required"] is True
    assert node_action_accepted_nodes(outcome.details) == (
        {"node-a"} if accepted else set()
    )
    assert outcome.details.get("ownership_permit_delivery_unknown", False) is permit
    assert "node_action_not_started" not in outcome.details
    assert "node_action_requires_new_command" not in outcome.details
    assert "failed_nodes" not in outcome.details


def defective_response(target: Target, defect: str) -> Reply:
    if defect == "json":
        return b"{broken"
    if defect == "schema":
        return b'{"command_id": "untrusted", "state": "unknown"}'
    if defect == "read":
        return OSError("response stream ended")
    if defect == "timeout":
        return TimeoutError("post-delivery timeout")
    if defect == "url":
        return URLError("post-delivery disconnect")
    if defect.startswith("http-"):
        return http_error(int(defect.removeprefix("http-")))
    terminal = target.terminal().model_dump(mode="json")
    if defect == "submission-id":
        terminal["command_id"] = "other-command"
    elif defect == "result-id":
        terminal["result"]["command_id"] = "other-command"
    elif defect == "operation":
        terminal["result"]["operation"] = WorkflowOperation.RESTART_NODE.value
    elif defect == "status":
        terminal["state"] = NodeActionExecutionState.FAILED.value
    elif defect == "missing-result":
        terminal["result"] = None
    elif defect == "pending-result":
        terminal["state"] = NodeActionExecutionState.PENDING.value
    else:
        raise AssertionError(f"unknown response defect: {defect}")
    return json.dumps(terminal).encode()


@pytest.mark.parametrize(
    "defect",
    [
        "json",
        "schema",
        "read",
        "timeout",
        "url",
        "submission-id",
        "result-id",
        "operation",
        "status",
        "missing-result",
        "pending-result",
        "http-401",
        "http-403",
        "http-404",
        "http-409",
        "http-410",
        "http-422",
        "http-429",
        "http-500",
        "http-503",
    ],
)
def test_ambiguous_permit_reply_keeps_the_accepted_command_pending(
    target: Target, monkeypatch: pytest.MonkeyPatch, defect: str
) -> None:
    wire = Wire([encoded(target.pending), defective_response(target, defect)])
    monkeypatch.setattr(transport, "urlopen", wire)

    outcome = target.execute()

    assert_unknown(target, outcome, accepted=True, permit=True)
    assert wire.methods() == ["GET", "POST"]
    sent = submitted(wire.requests[-1])
    assert sent.ownership_permit is not None
    assert sent.ownership_permit.allowed is True
    assert sent.command.command_id == target.command_id
    if defect.startswith("http-"):
        assert outcome.details["http_status"] == int(defect.removeprefix("http-"))


@pytest.mark.parametrize(
    "defect",
    [
        "json",
        "schema",
        "read",
        "timeout",
        "submission-id",
        "result-id",
        "operation",
        "status",
        "missing-result",
        "pending-result",
        "http-401",
        "http-409",
        "http-422",
        "http-408",
        "http-429",
        "http-503",
    ],
)
def test_ambiguous_initial_reply_is_unknown_without_inventing_acceptance(
    target: Target, monkeypatch: pytest.MonkeyPatch, defect: str
) -> None:
    wire = Wire([http_error(404), defective_response(target, defect)])
    monkeypatch.setattr(transport, "urlopen", wire)

    outcome = target.execute()

    assert_unknown(target, outcome, accepted=False, permit=False)
    assert wire.methods() == ["GET", "POST"]
    sent = submitted(wire.requests[-1])
    assert sent.ownership_permit is None


@pytest.mark.parametrize("accepted", [False, True])
@pytest.mark.parametrize(
    "later",
    ["absent", "json", "wrong-id", "http-403", "http-503", "timeout", "pending"],
)
def test_fresh_adapter_keeps_unknown_delivery_poll_only_until_a_valid_terminal_reply(
    target: Target, monkeypatch: pytest.MonkeyPatch, accepted: bool, later: str
) -> None:
    first = encoded(target.pending) if accepted else http_error(404)
    wire = Wire([first, b"unreadable"])
    monkeypatch.setattr(transport, "urlopen", wire)
    target.remember(target.execute())
    # Reconstruct from the serialized workflow, not state retained by this adapter.
    target.adapter = inputs()[0]
    target.context = replace(
        target.context,
        workflow=WorkflowRequest.model_validate_json(
            target.context.workflow.model_dump_json()
        ),
    )
    reply: Reply
    if later == "absent":
        reply = http_error(404)
    elif later == "wrong-id":
        reply = defective_response(target, "submission-id")
    elif later == "pending":
        reply = encoded(target.pending)
    else:
        reply = defective_response(target, later)
    wire.replies.extend([reply, encoded(target.terminal())])

    uncertain = target.execute()
    assert_unknown(
        target, uncertain, accepted=accepted or later == "pending", permit=accepted
    )
    target.remember(uncertain)
    completed = target.execute()

    assert completed.status is WorkflowStepStatus.SUCCEEDED
    assert completed.details["node_results"] == {"node-a": {}}
    assert "outcome_unknown" not in completed.details
    assert wire.methods() == ["GET", "POST", "GET", "GET"]
    queries = [
        parse_qs(urlsplit(request.full_url).query)
        for request in wire.requests
        if request.get_method() == "GET"
    ]
    assert all(query["command_id"] == [target.command_id] for query in queries), (
        "all recovery polls must retain the possibly accepted command identity"
    )


@pytest.mark.parametrize("defect", ["json", "result-id", "operation", "missing-result"])
def test_later_malformed_poll_preserves_previously_confirmed_acceptance(
    target: Target, monkeypatch: pytest.MonkeyPatch, defect: str
) -> None:
    target.remember(
        WorkflowStepOutcome.waiting(
            details={
                "node_action_command_id": target.command_id,
                "node_action_state": "PENDING",
                "node_action_accepted_nodes": ["node-a"],
            }
        )
    )
    wire = Wire([defective_response(target, defect)])
    monkeypatch.setattr(transport, "urlopen", wire)

    outcome = target.execute()

    assert_unknown(target, outcome, accepted=True, permit=False)
    assert wire.methods() == ["GET"]


def test_missing_previously_accepted_command_does_not_authorize_a_new_submission(
    target: Target, monkeypatch: pytest.MonkeyPatch
) -> None:
    target.remember(
        WorkflowStepOutcome.waiting(
            details={
                "node_action_command_id": target.command_id,
                "node_action_state": "PENDING",
                "node_action_accepted_nodes": ["node-a"],
            }
        )
    )
    wire = Wire([http_error(404)])
    monkeypatch.setattr(transport, "urlopen", wire)

    assert_unknown(target, target.execute(), accepted=True, permit=False)
    assert wire.methods() == ["GET"]


@pytest.mark.parametrize("phase", ["lookup", "submission", "permit"])
@pytest.mark.parametrize("boundary", ["read", "exit"])
def test_response_read_and_context_exit_failures_never_reach_not_started_fallback(
    target: Target, monkeypatch: pytest.MonkeyPatch, phase: str, boundary: str
) -> None:
    calls: list[str] = []

    class BrokenResponse(io.BytesIO):
        def read(self, size: int | None = -1) -> bytes:
            if boundary == "read":
                raise ValueError("response decoder lost its buffer")
            return super().read(size)

        def __exit__(self, *args: object) -> None:
            self.close()
            if boundary == "exit":
                raise RuntimeError("response context could not close")

    def exchange(request: Request, **kwargs: Any) -> io.BytesIO:
        calls.append(request.get_method())
        if request.get_method() == "GET" and phase != "lookup":
            if phase == "submission":
                raise http_error(404)
            return io.BytesIO(encoded(target.pending))
        return BrokenResponse(encoded(target.terminal()))

    monkeypatch.setattr(transport, "urlopen", exchange)

    assert_unknown(
        target, target.execute(), accepted=phase == "permit", permit=phase == "permit"
    )
    assert calls == (["GET"] if phase == "lookup" else ["GET", "POST"])


def test_actual_agent_completion_can_be_polled_after_a_corrupted_permit_ack(
    target: Target, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    target.context = replace(
        target.context,
        step=target.context.step.model_copy(update={"gpu_uuids": ["GPU-a"]}),
    )
    hardware = FakeRunner()
    agent = node_action_executor(
        tmp_path,
        "unknown-permit.db",
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
    posted: list[SignedNodeAction] = []
    corrupted = False
    with TestClient(create_node_agent_app(agent)) as client:
        bridge = transport_bridge(client)

        def exchange(request: Request, **kwargs: Any) -> io.BytesIO:
            nonlocal corrupted
            reply = bridge(request, **kwargs)
            if request.get_method() == "POST":
                signed = submitted(request)
                posted.append(signed)
                if signed.ownership_permit is not None:
                    corrupted = True
                    reply.close()
                    return io.BytesIO(b"lost accepted permit response")
            return reply

        monkeypatch.setattr(transport, "urlopen", exchange)
        for _ in range(200):
            outcome = target.execute()
            if corrupted:
                break
            target.remember(outcome)
        else:
            pytest.fail(
                "the owned Agent did not reach its physical permission boundary"
            )
        assert_unknown(target, outcome, accepted=True, permit=True)
        target.remember(outcome)
        for _ in range(200):
            outcome = target.execute()
            if outcome.status is not WorkflowStepStatus.WAITING:
                break
            target.remember(outcome)
        else:
            pytest.fail("the accepted command did not return a terminal ledger result")

    assert outcome.status is WorkflowStepStatus.SUCCEEDED
    assert len(posted) == 2
    assert posted[0].ownership_permit is None
    assert posted[1].ownership_permit is not None
    assert posted[1].ownership_permit.allowed is True
    assert [command for command in hardware.commands if "--gpu-reset" in command] == [
        ["nvidia-smi", "--gpu-reset", "-i", "GPU-a"]
    ]


@pytest.mark.parametrize(
    ("status", "code"),
    [(401, "INVALID_SIGNATURE"), (403, "OPERATION_NOT_ALLOWED"), (422, "INVALID_TTL")],
)
def test_initial_explicit_http_refusal_still_fails_without_claiming_acceptance(
    target: Target, monkeypatch: pytest.MonkeyPatch, status: int, code: str
) -> None:
    wire = Wire(
        [
            http_error(404),
            http_error(
                status,
                {
                    "code": code,
                    "message": "pre-dispatch rejection",
                    "retryable": False,
                    "requires_new_command": False,
                },
            ),
        ]
    )
    monkeypatch.setattr(transport, "urlopen", wire)

    outcome = target.execute()

    assert outcome.status is WorkflowStepStatus.FAILED
    assert outcome.details["node_action_error_code"] == code
    assert outcome.details["http_status"] == status
    assert outcome.details["safety_rejection"] is True
    assert node_action_accepted_nodes(outcome.details) == set()
    assert "node_action_response_unknown" not in outcome.details
    assert wire.methods() == ["GET", "POST"]


def test_pre_submit_owner_refusal_still_prevents_delivery(
    target: Target, monkeypatch: pytest.MonkeyPatch
) -> None:
    target.state.job["metadata"]["uid"] = "recreated-before-submit"
    wire = Wire([http_error(404)])
    monkeypatch.setattr(transport, "urlopen", wire)

    outcome = target.execute()

    assert outcome.status is WorkflowStepStatus.FAILED
    assert outcome.details["node_action_not_started"] is True
    assert outcome.details["manual_confirmation_required"] is True
    assert "node_action_response_unknown" not in outcome.details
    assert wire.methods() == ["GET"]


def test_workflow_expiry_does_not_convert_an_uncertain_delivery_into_not_started(
    target: Target, monkeypatch: pytest.MonkeyPatch
) -> None:
    wire = Wire([encoded(target.pending), b"unreadable", encoded(target.terminal())])
    monkeypatch.setattr(transport, "urlopen", wire)
    target.remember(target.execute())
    target.context = replace(
        target.context,
        workflow=target.context.workflow.model_copy(
            update={
                "lifetime_deadline_at": datetime.now(timezone.utc)
                - timedelta(seconds=1)
            }
        ),
    )

    outcome = target.execute()

    assert outcome.status is WorkflowStepStatus.SUCCEEDED
    assert "node_action_not_started" not in outcome.details
    assert wire.methods() == ["GET", "POST", "GET"]


@pytest.mark.parametrize(
    "change",
    [
        "phase",
        "unknown-phase",
        "fencing",
        "parameters",
        "gpu-scope",
        "cluster",
        "prefix",
    ],
)
def test_uncertain_history_cannot_be_adopted_after_phase_or_intent_changes(
    target: Target, monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    wire = Wire([encoded(target.pending), b"unreadable", encoded(target.terminal())])
    monkeypatch.setattr(transport, "urlopen", wire)
    unknown = target.execute()
    target.remember(unknown)
    digest = unknown.details["node_action_intent_sha256"]
    assert isinstance(digest, str) and len(digest) == 64
    if change == "phase":
        target.context = replace(
            target.context,
            workflow=target.context.workflow.model_copy(update={"safety_only": True}),
        )
    elif change == "unknown-phase":
        target.context = replace(
            target.context,
            workflow=target.context.workflow.model_copy(
                update={
                    "step_executions": [
                        item.model_copy(update={"phase": None})
                        if item.step_index == 1
                        else item
                        for item in target.context.workflow.step_executions
                    ]
                }
            ),
        )
    elif change == "fencing":
        target.context = replace(
            target.context,
            workflow=target.context.workflow.model_copy(update={"fencing_token": 2}),
        )
    elif change in {"parameters", "gpu-scope"}:
        changed_step: dict[str, Any] = (
            {"parameters": {"changed_intent": True}}
            if change == "parameters"
            else {"gpu_uuids": ["GPU-other"]}
        )
        target.context = replace(
            target.context, step=target.context.step.model_copy(update=changed_step)
        )
    elif change == "cluster":
        target.context = replace(
            target.context,
            incident=target.context.incident.model_copy(update={"cluster_id": "other"}),
        )
    else:
        target.context = replace(
            target.context, idempotency_key="merged-workflow/1/RESET_GPU"
        )

    for _ in range(2):
        refused = target.execute()
        assert_unknown(target, refused, accepted=True, permit=True)
        assert refused.details["reason"] == "OWNERSHIP_INTENT_MISMATCH"
        assert refused.details["node_action_intent_sha256"] == digest
        target.remember(refused)
    assert wire.methods() == ["GET", "POST"]
    assert len(wire.replies) == 1


def test_unbound_legacy_uncertainty_never_borrows_the_current_intent(
    target: Target, monkeypatch: pytest.MonkeyPatch
) -> None:
    wire = Wire([encoded(target.pending), b"unreadable", encoded(target.terminal())])
    monkeypatch.setattr(transport, "urlopen", wire)
    unknown = target.execute()
    unknown.details.pop("node_action_intent_sha256")
    target.remember(unknown)

    for _ in range(2):
        refused = target.execute()
        assert_unknown(target, refused, accepted=True, permit=True)
        assert refused.details["reason"] == "OWNERSHIP_INTENT_MISMATCH"
        assert "node_action_intent_sha256" not in refused.details
        target.remember(refused)
    assert wire.methods() == ["GET", "POST"]


def test_new_phase_without_a_command_receipt_cannot_hide_prior_uncertainty(
    target: Target, monkeypatch: pytest.MonkeyPatch
) -> None:
    wire = Wire([encoded(target.pending), b"unreadable", encoded(target.terminal())])
    monkeypatch.setattr(transport, "urlopen", wire)
    target.remember(target.execute())
    workflow = target.context.workflow
    target.context = replace(
        target.context,
        workflow=workflow.model_copy(
            update={
                "safety_only": True,
                "step_executions": [
                    *workflow.step_executions,
                    workflow_step_execution(
                        1,
                        target.context.step.operation,
                        WorkflowStepStatus.WAITING,
                        phase="safety",
                        details={},
                    ),
                ],
            }
        ),
    )

    refused = target.execute()

    assert_unknown(target, refused, accepted=True, permit=True)
    assert refused.details["reason"] == "OWNERSHIP_INTENT_MISMATCH"
    assert wire.methods() == ["GET", "POST"]


def test_waiting_exhaustion_retains_unknown_flags_for_the_hardware_classifier(
    target: Target, monkeypatch: pytest.MonkeyPatch
) -> None:
    wire = Wire([encoded(target.pending), b"unreadable"])
    monkeypatch.setattr(transport, "urlopen", wire)
    unknown = target.execute()
    target.remember(unknown)
    executions = target.context.workflow.step_executions
    old = datetime.now(timezone.utc) - timedelta(seconds=601)
    workflow = target.context.workflow.model_copy(
        update={
            "step_executions": [
                item.model_copy(update={"started_at": old})
                if item.step_index == target.context.step_index
                else item
                for item in executions
            ]
        }
    )
    bounded = step_bounds.bounded_waiting_outcome(
        SimpleNamespace(config=ProductionExecutorConfig.from_mapping({})),
        workflow,
        target.context.step,
        target.context.step_index,
        unknown,
    )
    saved = step_bounds.record_attempt(
        workflow, target.context.step, target.context.step_index, bounded
    ).model_copy(update={"status": WorkflowStatus.FAILED})

    assert bounded.status is WorkflowStepStatus.FAILED
    assert bounded.details is not None, (
        "waiting exhaustion must retain response uncertainty"
    )
    assert bounded.details["step_waiting_timeout_seconds"] == 600
    assert bounded.details["outcome_unknown"] is True
    assert bounded.details["manual_confirmation_required"] is True
    assert bounded.details["node_action_command_id"] == target.command_id
    assert node_action_accepted_nodes(bounded.details) == {"node-a"}
    assert "node_action_not_started" not in bounded.details
    classification = HardwareEscalationService.classify(saved)
    assert classification is not None, "unknown physical outcomes need operator review"
    assert classification[0] == "manual_confirmation_required"
    assert classification[1] is RecoveryAction.ESCALATE_OPERATOR
    assert classification[2] is WorkflowOperation.ESCALATE_SUPPORT
    assert wire.methods() == ["GET", "POST"]


@pytest.mark.parametrize("exhausted", [False, True])
def test_unknown_reset_cannot_release_services_or_scheduling_on_timeout(
    target: Target, monkeypatch: pytest.MonkeyPatch, exhausted: bool
) -> None:
    wire = Wire([encoded(target.pending), b"unreadable"])
    monkeypatch.setattr(transport, "urlopen", wire)
    unknown = target.execute()
    store = build_store()
    operations = [
        WorkflowOperation.QUIESCE_GPU_SERVICES,
        WorkflowOperation.RESET_GPU,
        WorkflowOperation.RESTORE_GPU_SERVICES,
        WorkflowOperation.RESTORE_SCHEDULING,
    ]
    _incident, workflow = workflow_state(store, operations)
    started = datetime.now(timezone.utc) - timedelta(seconds=601 if exhausted else 1)
    workflow = workflow.model_copy(
        update={
            "execution_deadline": started + timedelta(seconds=1800),
            "completed_step_indexes": [0],
            "completed_operations": [WorkflowOperation.QUIESCE_GPU_SERVICES],
            "step_executions": [
                workflow_step_execution(0, operations[0]),
                workflow_step_execution(
                    1,
                    operations[1],
                    WorkflowStepStatus.WAITING,
                    details=unknown.details,
                    started_at=started,
                ),
            ],
        }
    )
    store.save_workflow(workflow)
    adapter = FakeAdapter(
        {
            operation: unknown
            if operation is WorkflowOperation.RESET_GPU
            else WorkflowStepOutcome.succeeded()
            for operation in operations
        }
    )
    executor = active_workflow_executor(store, [adapter], operations)

    result = executor.execute(
        workflow.request_id,
        WorkflowExecutionRequest(expected_fencing_token=workflow.fencing_token),
    )

    assert adapter.calls == ["workflow-active/1/RESET_GPU"], (
        "an unconfirmed reset must not release services or scheduling, even at its cap"
    )
    saved = store.get_workflow(workflow.request_id)
    assert saved.branch_escalation_counts == {}
    if not exhausted:
        assert result.status is WorkflowStatus.RUNNING
    else:
        classification = HardwareEscalationService.classify(saved)
        assert classification is not None, (
            "timed-out unknown actions need operator review"
        )
        assert classification[1] is RecoveryAction.ESCALATE_OPERATOR


@pytest.mark.parametrize("exhausted", [False, True])
def test_real_adapter_unknown_lookup_cannot_authorize_compensation(
    target: Target, monkeypatch: pytest.MonkeyPatch, exhausted: bool
) -> None:
    monkeypatch.setattr(
        target.adapter.registry,
        "maintenance_endpoint",
        lambda cluster, node, generation: "http://node-local",
        raising=False,
    )
    now = datetime.now(timezone.utc)
    quiesce = workflow_step(
        WorkflowOperation.QUIESCE_GPU_SERVICES,
        target.adapter.owner,
        node_ids=["node-a"],
    )
    restore = workflow_step(
        WorkflowOperation.RESTORE_GPU_SERVICES,
        target.adapter.owner,
        node_ids=["node-a"],
    )
    steps = [
        target.context.workflow.official_steps[0],
        quiesce,
        target.context.step,
        restore,
    ]
    quiesced = workflow_step_execution(
        1,
        quiesce.operation,
        phase="official",
        details={
            "agent_generations": {"node-a": 1},
            "maintenance_window_expires_at": (now + timedelta(hours=1)).isoformat(),
        },
    )
    target.context = replace(
        target.context,
        step_index=2,
        idempotency_key="workflow-local/2/RESET_GPU",
        workflow=target.context.workflow.model_copy(
            update={
                "official_steps": steps,
                "completed_step_indexes": [0, 1],
                "completed_operations": [steps[0].operation, quiesce.operation],
                "step_executions": [*target.context.workflow.step_executions, quiesced],
            }
        ),
    )
    posted: list[SignedNodeAction] = []
    polled: list[str] = []
    first_lookup = True

    def exchange(request: Request, **kwargs: Any) -> io.BytesIO:
        nonlocal first_lookup
        if request.get_method() == "GET":
            polled.append(parse_qs(urlsplit(request.full_url).query)["command_id"][0])
            if first_lookup:
                first_lookup = False
                return io.BytesIO(b"unreadable earlier command lookup")
            raise http_error(404)
        signed = submitted(request)
        posted.append(signed)
        return io.BytesIO(
            encoded(
                NodeActionSubmission(
                    command_id=signed.command.command_id,
                    state=NodeActionExecutionState.SUCCEEDED,
                    result=NodeActionResult(
                        command_id=signed.command.command_id,
                        operation=signed.command.operation,
                        status=NodeActionStatus.SUCCEEDED,
                    ),
                )
            )
        )

    monkeypatch.setattr(transport, "urlopen", exchange)
    unknown = target.execute()
    assert unknown.status is WorkflowStepStatus.WAITING
    assert unknown.details["outcome_unknown"] is True
    assert node_action_accepted_nodes(unknown.details) == set()
    target.remember(unknown)
    started = now - timedelta(seconds=601 if exhausted else 1)
    workflow = target.context.workflow.model_copy(
        update={
            "execution_deadline": started + timedelta(seconds=1800),
            "step_executions": [
                item.model_copy(update={"started_at": started})
                if item.step_index == 2
                else item
                for item in target.context.workflow.step_executions
            ],
        }
    )
    store = build_store()
    store.save_incident(target.context.incident)
    store.save_workflow(workflow)
    executor = active_workflow_executor(
        store, [target.adapter], [step.operation for step in steps]
    )

    with stop_ownership_scope(target.validator):
        executor.execute(
            workflow.request_id,
            WorkflowExecutionRequest(expected_fencing_token=workflow.fencing_token),
        )

    assert posted == [], (
        "unknown lookup is not permission to restore services or retry reset"
    )
    assert polled == [unknown.details["node_action_command_id"]] * 2
    saved = store.get_workflow(workflow.request_id)
    assert saved.branch_escalation_counts == {}
    reset = next(item for item in saved.step_executions if item.step_index == 2)
    assert reset.details["outcome_unknown"] is True
    assert "node_action_not_started" not in reset.details


@pytest.mark.parametrize(
    "reply",
    [
        "pending",
        "no-action",
        "legacy-no-action",
        "later-refusal",
        "success",
        "other-failure",
    ],
)
def test_negative_permit_never_substitutes_intended_denial_for_agent_evidence(
    target: Target, monkeypatch: pytest.MonkeyPatch, reply: str
) -> None:
    target.state.job["metadata"]["uid"] = "changed-before-final-recheck"
    details: dict[str, Any] = {
        "reason": "OWNED_NATIVE_REFUSAL",
        "node_action_not_started": True,
        "safety_rejection": True,
        "manual_confirmation_required": True,
        "agent_queue_ownership_checked": True,
        "ownership_check_boundary": "AGENT_PRE_SPAWN",
        "node_action_command_id": target.command_id,
    }
    if reply == "no-action":
        details["physical_ownership_checks"] = []
    elif reply == "later-refusal":
        details["physical_ownership_checks"] = [{"sequence": 1}]
    elif reply in {"success", "other-failure"}:
        details = {}
    terminal = target.terminal(
        NodeActionStatus.SUCCEEDED if reply == "success" else NodeActionStatus.FAILED,
        details=details,
    )
    assert terminal.result is not None
    terminal.result.error = "original native result"
    wire = Wire(
        [
            encoded(target.pending),
            encoded(target.pending if reply == "pending" else terminal),
        ]
    )
    monkeypatch.setattr(transport, "urlopen", wire)

    outcome = target.execute()

    sent = submitted(wire.requests[-1])
    assert sent.ownership_permit is not None
    assert sent.ownership_permit.allowed is False
    assert wire.methods() == ["GET", "POST"]
    if reply == "pending":
        assert_unknown(target, outcome, accepted=True, permit=False)
        assert outcome.details["reason"] == "OWNERSHIP_DENIAL_UNCONFIRMED"
    elif reply == "success":
        assert outcome.status is WorkflowStepStatus.SUCCEEDED
        assert "node_action_not_started" not in outcome.details
    else:
        assert outcome.status is WorkflowStepStatus.FAILED
        assert "original native result" in str(outcome.error)
        if reply == "no-action":
            assert outcome.details["node_action_not_started"] is True
            assert outcome.details["physical_ownership_checks"] == []
            assert "outcome_unknown" not in outcome.details
        elif reply in {"legacy-no-action", "later-refusal"}:
            assert "node_action_not_started" not in outcome.details
            assert outcome.details["outcome_unknown"] is True
            assert outcome.details["manual_confirmation_required"] is True
            assert outcome.details["reason"] == "OWNED_NATIVE_REFUSAL"
        else:
            assert "node_action_not_started" not in outcome.details
            assert "reason" not in outcome.details


@pytest.mark.parametrize(
    ("field", "value"),
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
def test_mismatched_pending_challenge_is_unknown_and_never_gets_a_permit(
    target: Target, monkeypatch: pytest.MonkeyPatch, field: str, value: Any
) -> None:
    challenge = target.pending.ownership_challenge
    assert challenge is not None
    state = target.pending.model_copy(
        update={"ownership_challenge": challenge.model_copy(update={field: value})}
    )
    wire = Wire([encoded(state)])
    monkeypatch.setattr(transport, "urlopen", wire)

    outcome = target.execute()

    assert_unknown(target, outcome, accepted=True, permit=False)
    assert outcome.details["reason"] == "OWNERSHIP_CHALLENGE_MISMATCH"
    assert wire.methods() == ["GET"]


def test_accepted_challenge_recheck_exception_is_not_a_pre_dispatch_refusal(
    target: Target, monkeypatch: pytest.MonkeyPatch
) -> None:
    def unavailable(*args: Any) -> None:
        raise RuntimeError("late recheck failed")

    monkeypatch.setattr(target.validator, "check", unavailable)
    wire = Wire([encoded(target.pending)])
    monkeypatch.setattr(transport, "urlopen", wire)

    outcome = target.execute()

    assert_unknown(target, outcome, accepted=True, permit=False)
    assert outcome.details["reason"] == "OWNERSHIP_RECHECK_UNVERIFIABLE"
    assert wire.methods() == ["GET"]


@pytest.mark.parametrize("flag", ["outcome_unknown", "manual_confirmation_required"])
def test_unknown_terminal_failure_cannot_request_a_retry_or_lose_uncertainty(
    target: Target, monkeypatch: pytest.MonkeyPatch, flag: str
) -> None:
    wire = Wire(
        [
            encoded(
                target.terminal(
                    NodeActionStatus.FAILED, retryable=True, details={flag: True}
                )
            )
        ]
    )
    monkeypatch.setattr(transport, "urlopen", wire)

    outcome = target.execute()

    assert outcome.status is WorkflowStepStatus.FAILED
    assert outcome.details[flag] is True
    assert "node_action_not_started" not in outcome.details
    assert wire.methods() == ["GET"]

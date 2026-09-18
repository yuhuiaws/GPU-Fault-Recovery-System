from __future__ import annotations

import io
import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from urllib.parse import parse_qs, urlsplit

import pytest

from gpu_fault.models import WorkflowOperation
from gpu_fault.node_agent.protocol import SignedNodeAction, sign_node_action
from scripts.e2e.regional import auth015_protocol as protocol
from tests.regional._cov95_auth015_support import KEY_A, KEY_B, AgentPair, agent
from tests.regional._cov95_identity_support import offline_guard as offline_guard


@pytest.fixture
def pair(tmp_path, monkeypatch):
    value = AgentPair(tmp_path, monkeypatch)
    try:
        yield value
    finally:
        value.close()


def test_deployed_challenges_use_real_signature_gates_without_dispatch(pair):
    proof = protocol.prove_signatures(pair.targets["node-a"], pair.targets["node-b"])
    assert [item["status"] for item in proof["observations"]] == [
        404,
        404,
        403,
        401,
        401,
        404,
    ]
    assert all(item["matched"] is True for item in proof["observations"]), (
        "every control and negative response must match the exact protocol"
    )
    assert [node for node, _ in pair.reads] == ["node-a", "node-b", "node-b"], (
        "wrong-key commands and queries must not reach ledger reads"
    )
    assert all(command_id == proof["command_id"] for _, command_id in pair.reads), (
        "all result lookups must use this proof's fresh command identity"
    )
    assert proof["rotated_key_activation_proved"] is False
    assert KEY_A not in repr(proof) and KEY_B not in repr(proof)
    commands = [
        SignedNodeAction.model_validate_json(request.data)
        for request in pair.requests
        if request.data is not None
    ]
    assert len(commands) == 2 and commands[0].command == commands[1].command
    command = commands[0].command
    assert command.operation is WorkflowOperation.FREEZE_EVIDENCE
    assert command.gpu_uuids == [] and command.parameters == {}
    assert command.expires_at < command.issued_at - timedelta(seconds=30)
    assert commands[0].signature == sign_node_action(command, KEY_B)
    assert commands[1].signature == sign_node_action(command, KEY_A)
    for request in pair.requests:
        if request.data is None:
            query = parse_qs(urlsplit(request.full_url).query)
            stamp = datetime.fromisoformat(query["issued_at"][0])
            assert abs((datetime.now(timezone.utc) - stamp).total_seconds()) < 5


def test_each_protocol_proof_uses_a_new_command_identity(pair):
    first = protocol.prove_signatures(pair.targets["node-a"], pair.targets["node-b"])
    second = protocol.prove_signatures(pair.targets["node-a"], pair.targets["node-b"])
    assert first["command_id"] != second["command_id"], (
        "a prior result cannot satisfy a new challenge"
    )


@pytest.mark.parametrize("clock_offset", [-31536000, -31, -30, -29, 0, 31536000])
def test_impossible_time_window_stops_future_handler_support_at_every_clock(
    pair, monkeypatch, clock_offset
):
    executor = pair.executors["node-b"]
    executor.allowed_operations = {WorkflowOperation.FREEZE_EVIDENCE}
    now = datetime.now(timezone.utc)
    executor.now = lambda: now + timedelta(seconds=clock_offset)
    monkeypatch.setattr(
        executor,
        "execute",
        lambda _: pytest.fail("future handler support bypassed the time guard"),
    )
    command = protocol.rejected_command(pair.records["node-b"], "auth015-future", now)
    envelope = SignedNodeAction(
        command=command, signature=sign_node_action(command, KEY_B)
    )
    result = pair.clients["node-b"].post(
        "/v1/node-actions/submit", json=envelope.model_dump(mode="json")
    )
    assert result.status_code in {410, 422}
    assert result.json()["detail"]["code"] in {"COMMAND_EXPIRED", "INVALID_ISSUED_AT"}
    assert pair.reads == [], (
        "even the correctly signed future operation must remain non-dispatching"
    )


@pytest.mark.parametrize(
    "defect",
    [
        "same-node",
        "foreign-cluster",
        "shared-key",
        "short-key",
        "spaced-key",
        "http",
        "no-cert",
        "key-v1",
        "allowed-challenge",
    ],
)
def test_ambiguous_or_executable_target_is_refused_before_any_request(pair, defect):
    source, target = pair.targets["node-a"], pair.targets["node-b"]
    if defect == "same-node":
        target = protocol.SignatureTarget(source.agent, KEY_B)
    elif defect == "foreign-cluster":
        target = protocol.SignatureTarget(
            agent("node-b", cluster_id="cluster-b"), KEY_B
        )
    elif defect in {"shared-key", "short-key", "spaced-key"}:
        key = {"shared-key": KEY_A, "short-key": "short", "spaced-key": KEY_B + " "}[
            defect
        ]
        target = protocol.SignatureTarget(target.agent, key)
    else:
        changes = {
            "http": {"endpoint": "http://10.0.1.2:9099"},
            "no-cert": {"tls_certificate_pem": None},
            "key-v1": {"node_action_key_version": 1},
            "allowed-challenge": {
                "allowed_operations": [WorkflowOperation.FREEZE_EVIDENCE]
            },
        }[defect]
        target = protocol.SignatureTarget(agent("node-b", **changes), KEY_B)
    with pytest.raises(protocol.Auth015ProofError, match="targets or key scope"):
        protocol.prove_signatures(source, target)
    assert pair.requests == []
    assert KEY_A not in repr(source) and KEY_B not in repr(target)


def test_invalid_certificate_stops_before_the_request(pair):
    record = agent(
        "node-b",
        tls_certificate_pem="-----BEGIN CERTIFICATE-----\ninvalid\n-----END CERTIFICATE-----",
    )
    with pytest.raises(protocol.Auth015ProofError, match="TLS certificate"):
        protocol.prove_signatures(
            pair.targets["node-a"], protocol.SignatureTarget(record, KEY_B)
        )
    assert pair.requests == []


@pytest.mark.parametrize("failure_at", range(6))
def test_wrong_response_stops_the_challenge_sequence(pair, monkeypatch, failure_at):
    original = pair.bounded_request
    calls = []

    def response(request, **kwargs):
        calls.append(request)
        if len(calls) == failure_at + 1:
            return 200, json.dumps({"unexpected": KEY_A}).encode()
        return original(request, **kwargs)

    monkeypatch.setattr(protocol, "bounded_request", response)
    with pytest.raises(
        protocol.Auth015ProofError, match="unexpected response"
    ) as caught:
        protocol.prove_signatures(pair.targets["node-a"], pair.targets["node-b"])
    assert len(calls) == failure_at + 1, (
        "no later request is sent after a failed control or rejection"
    )
    assert KEY_A not in str(caught.value) and KEY_B not in str(caught.value)


@pytest.mark.parametrize("defect", ["network", "json", "oversize"])
def test_transport_and_unbounded_responses_fail_without_echoing_payloads(
    pair, monkeypatch, defect
):
    def response(*args, **kwargs):
        if defect == "network":
            raise OSError(KEY_A)
        return 404, b"{" if defect == "json" else b"x" * (
            protocol.MAX_RESPONSE_BYTES + 1
        )

    monkeypatch.setattr(protocol, "bounded_request", response)
    with pytest.raises(protocol.Auth015ProofError) as caught:
        protocol.prove_signatures(pair.targets["node-a"], pair.targets["node-b"])
    assert KEY_A not in str(caught.value), (
        "transport diagnostics cannot disclose private key material"
    )


def test_expired_challenge_budget_prevents_the_first_request(pair, monkeypatch):
    ticks = iter([100.0, 131.0])
    monkeypatch.setattr(
        protocol, "time", SimpleNamespace(monotonic=lambda: next(ticks))
    )
    with pytest.raises(protocol.Auth015ProofError, match="deadline"):
        protocol.prove_signatures(pair.targets["node-a"], pair.targets["node-b"])
    assert pair.requests == []


def test_response_arriving_after_the_budget_cannot_be_used_as_proof(pair, monkeypatch):
    ticks = iter([100.0, 100.0, 131.0])
    monkeypatch.setattr(
        protocol, "time", SimpleNamespace(monotonic=lambda: next(ticks))
    )
    with pytest.raises(protocol.Auth015ProofError, match="deadline"):
        protocol.prove_signatures(pair.targets["node-a"], pair.targets["node-b"])
    assert len(pair.requests) == 1, "a late response must stop subsequent requests"


@pytest.mark.parametrize("field", ["retryable", "requires_new_command"])
def test_rejection_flags_must_be_literal_booleans(pair, monkeypatch, field):
    original = pair.bounded_request

    def response(request, **kwargs):
        status, payload = original(request, **kwargs)
        if kwargs["method"] == "POST":
            document = json.loads(payload)
            document["detail"][field] = 0
            payload = json.dumps(document).encode()
        return status, payload

    monkeypatch.setattr(protocol, "bounded_request", response)
    with pytest.raises(protocol.Auth015ProofError, match="unexpected response"):
        protocol.prove_signatures(pair.targets["node-a"], pair.targets["node-b"])
    assert len(pair.requests) == 3


def test_future_streaming_request_body_is_refused_before_transport(pair, monkeypatch):
    original = protocol.Request

    def streaming(*args, **kwargs):
        request = original(*args, **kwargs)
        request.data = io.BytesIO(b"unbounded source")
        return request

    monkeypatch.setattr(protocol, "Request", streaming)
    with pytest.raises(protocol.Auth015ProofError, match="bounded byte body"):
        protocol.prove_signatures(pair.targets["node-a"], pair.targets["node-b"])
    assert pair.requests == []


def test_worker_deadline_stops_the_signature_sequence_without_private_diagnostics(
    pair, monkeypatch
):
    def expired(*args, **kwargs):
        raise protocol.HttpDeadlineExceeded(KEY_A)

    monkeypatch.setattr(protocol, "bounded_request", expired)
    with pytest.raises(protocol.Auth015ProofError, match="deadline") as caught:
        protocol.prove_signatures(pair.targets["node-a"], pair.targets["node-b"])
    assert KEY_A not in str(caught.value)
    assert pair.requests == []

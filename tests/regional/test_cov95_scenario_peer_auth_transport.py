from __future__ import annotations

import json
import time
from http.client import HTTPResponse

import pytest

from scripts.e2e.regional import auth015_protocol as protocol
from tests.regional._cov95_scenario_peer_safety import (
    scenario_peer_transport_guard_fixture as scenario_peer_transport_guard_fixture,
)
from tests.regional._cov95_scenario_peer_transport import response_server


def test_signature_proof_does_not_consume_an_oversized_tls_response_before_refusing_it(
    monkeypatch, tmp_path, scenario_peer_transport_guard
):
    body = json.dumps({"detail": "x" * (protocol.MAX_RESPONSE_BYTES * 16)}).encode()
    consumed = []
    original_read = HTTPResponse.read

    def observed_read(response, amount=None):
        value = original_read(response, amount)
        consumed.append(len(value))
        return value

    monkeypatch.setattr(HTTPResponse, "read", observed_read)
    with response_server(tmp_path, scenario_peer_transport_guard, body) as server:
        with pytest.raises(protocol.Auth015ProofError):
            protocol.prove_signatures(*server.targets)
        assert server.requests == [("GET", "/v1/node-actions/result")], (
            "the size-limit test did not reach exactly the intended TLS response"
        )
        assert sum(consumed) <= protocol.MAX_RESPONSE_BYTES + 1, (
            f"proof consumed {sum(consumed)} response bytes before applying its "
            f"{protocol.MAX_RESPONSE_BYTES}-byte limit"
        )


def test_signature_proof_wall_clock_budget_stops_a_trickled_tls_body(
    monkeypatch, tmp_path, scenario_peer_transport_guard
):
    monkeypatch.setattr(protocol, "REQUEST_TIMEOUT_SECONDS", 0.10)
    monkeypatch.setattr(protocol, "PROOF_TIMEOUT_SECONDS", 0.20)
    body = json.dumps({"detail": "node action command is unknown"}).encode()
    with response_server(
        tmp_path, scenario_peer_transport_guard, body, interval=0.04
    ) as server:
        started = time.monotonic()
        with pytest.raises(protocol.Auth015ProofError):
            protocol.prove_signatures(*server.targets)
        elapsed = time.monotonic() - started
        assert server.requests == [("GET", "/v1/node-actions/result")], (
            "the deadline test did not reach the trickled TLS response"
        )
        assert elapsed <= protocol.PROOF_TIMEOUT_SECONDS + 0.40, (
            f"trickled input held the proof for {elapsed:.3f}s despite its "
            f"{protocol.PROOF_TIMEOUT_SECONDS:.2f}s whole-proof deadline"
        )

from __future__ import annotations

import json

import pytest

from scripts.e2e.regional import auth015_agent_probe as probe
from tests.regional._cov95_auth015_live import API_TOKEN
from tests.regional._cov95_auth015_support import KEY_A, agent
from tests.regional._cov95_identity_support import offline_guard as offline_guard


@pytest.fixture
def transport(monkeypatch):
    monkeypatch.setenv("GPU_FAULT_RELEASE_ID", "release-a")
    monkeypatch.setenv("GPU_FAULT_EXECUTION_TOKEN", API_TOKEN)
    calls = []

    def query(**kwargs):
        calls.append(kwargs)
        assert kwargs.get("method", "GET") == "GET", (
            "the collector must never write through the CPU API"
        )
        assert kwargs["headers"]["X-GPU-Fault-Execution-Token"] == API_TOKEN
        assert kwargs["timeout"] == 5
        value = (
            {"module_digest": "f" * 64}
            if kwargs["url"].endswith("/v1/version")
            else agent(kwargs["url"].rsplit("/", 1)[-1]).model_dump(mode="json")
        )
        return 200, json.dumps(value).encode()

    monkeypatch.setattr(probe, "http_exchange", query)
    return calls


def test_readonly_probe_serializes_actual_agent_models_without_credentials(
    transport, capsys
):
    assert probe.main(["cluster-a", "node-a", "node-b", "release-a"]) == 0
    output = capsys.readouterr().out
    value = json.loads(output)
    assert set(value["agents"]) == {"node-a", "node-b"}
    assert value["release_id"] == "release-a"
    assert len(transport) == 3
    assert API_TOKEN not in output and KEY_A not in output


@pytest.mark.parametrize(
    "arguments",
    [
        [],
        ["cluster-a", "node-a", "node-b"],
        ["", "node-a", "node-b", "release-a"],
        ["cluster-a", "node-a", "node-b", ""],
        ["cluster-a", "node-a", "node-a", "release-a"],
        ["cluster-a", "bad/node", "node-b", "release-a"],
        ["cluster-a", "node-a", "node-b", "other-release"],
    ],
)
def test_bad_snapshot_identity_cannot_reach_the_api(transport, arguments, capsys):
    assert probe.main(arguments) == 1
    assert json.loads(capsys.readouterr().out) == {
        "error": "AUTH015 read-only agent snapshot failed"
    }
    assert transport == []


@pytest.mark.parametrize("token", [None, ""])
def test_missing_cpu_authentication_is_not_an_anonymous_fallback(
    transport, monkeypatch, token
):
    if token is None:
        monkeypatch.delenv("GPU_FAULT_EXECUTION_TOKEN")
    else:
        monkeypatch.setenv("GPU_FAULT_EXECUTION_TOKEN", token)
    assert probe.main(["cluster-a", "node-a", "node-b", "release-a"]) == 1
    assert transport == []


@pytest.mark.parametrize(
    "defect", ["http", "oversize", "non-object", "json", "transport"]
)
def test_invalid_cpu_responses_never_echo_the_body_or_exception(
    transport, monkeypatch, capsys, defect
):
    def query(*args, **kwargs):
        if defect == "transport":
            raise OSError(API_TOKEN)
        status, payload = {
            "http": (403, API_TOKEN.encode()),
            "oversize": (200, b"x" * (probe.MAX_RESPONSE_BYTES + 1)),
            "non-object": (200, b"[]"),
            "json": (200, b"{"),
        }[defect]
        return status, payload

    monkeypatch.setattr(probe, "http_exchange", query)
    assert probe.main(["cluster-a", "node-a", "node-b", "release-a"]) == 1
    output = capsys.readouterr().out
    assert API_TOKEN not in output
    assert json.loads(output) == {"error": "AUTH015 read-only agent snapshot failed"}

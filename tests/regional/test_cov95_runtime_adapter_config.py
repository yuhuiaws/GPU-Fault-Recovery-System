from __future__ import annotations

import json
from typing import Any

import pytest

from gpu_fault.adapters import NodeActionWorkflowAdapter
from tests.execution.test_node_action_transport_retry import ENDPOINT, SECRET
from tests.regional._cov95_runtime_support import offline_runtime as offline_runtime


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"secret": "short"}, "shared secret"),
        ({"secret": ""}, "secrets are required"),
        ({"node_secrets": {"": "a" * 32}}, "node IDs"),
        ({"node_secrets": {"node-a": "short"}}, "at least 32"),
        ({"node_action_key_version": 9}, "key version"),
        ({"endpoints": {}}, "endpoints or a fleet registry"),
        ({"verify_max_attempts": 0}, "must be positive"),
        ({"maintenance_window_seconds": 29}, "between 30 and 3600"),
        ({"maintenance_window_seconds": 3601}, "between 30 and 3600"),
        ({"submit_timeout_seconds": 0}, "submit timeout"),
        ({"submit_timeout_seconds": 61}, "submit timeout"),
        ({"poll_timeout_seconds": 0}, "poll timeout"),
        ({"poll_timeout_seconds": 61}, "poll timeout"),
        ({"max_parallel_node_actions": 0}, "parallelism"),
        ({"max_parallel_node_actions": 65}, "parallelism"),
        ({"node_action_retry_limit": -1}, "retry limit"),
        ({"node_action_retry_limit": 21}, "retry limit"),
    ],
)
def test_adapter_rejects_unbounded_or_unowned_configuration(
    changes: dict[str, Any], message: str
) -> None:
    calls = []
    settings = {
        "endpoints": {"node-a": ENDPOINT},
        "secret": SECRET,
        "sender": lambda *args: calls.append(args),
        **changes,
    }
    with pytest.raises(ValueError, match=message):
        NodeActionWorkflowAdapter(**settings)
    assert calls == []


@pytest.mark.parametrize("raw", ["{", "[]", '{"node-a":1}', '{"node-a":null}'])
def test_endpoint_environment_requires_a_json_node_to_url_mapping(
    monkeypatch: pytest.MonkeyPatch, raw: str
) -> None:
    monkeypatch.setenv("GPU_FAULT_NODE_AGENT_ENDPOINTS", raw)
    with pytest.raises(ValueError, match="must be JSON|must map node IDs to URLs"):
        NodeActionWorkflowAdapter.from_environment()


def test_node_scoped_key_map_works_without_a_shared_secret(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "GPU_FAULT_NODE_AGENT_ENDPOINTS", json.dumps({"node-a": ENDPOINT})
    )
    monkeypatch.setenv(
        "GPU_FAULT_NODE_ACTION_KEYS_JSON", json.dumps({"node-a": "a" * 32})
    )
    monkeypatch.setenv("GPU_FAULT_NODE_ACTION_KEY_VERSION", "2")
    adapter = NodeActionWorkflowAdapter.from_environment()
    assert adapter.node_action_key_version == 2
    assert adapter.secret == ""
    assert adapter.knows_node("cluster-a", "node-a") is True
    assert adapter.knows_node("cluster-a", "node-b") is False

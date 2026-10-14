"""A node key added after the executor started reaches its node-action adapter.

The executor signs every node action with the node's key from the mounted
``gpu-fault-node-action-keys`` Secret, which the adapter used to read once, at
start. A replacement node keyed later -- by the node-key sync, between deploys
-- was therefore unknown to the executor until it restarted: its actions waited
out the ``KEY_PENDING`` grace and then failed closed with "node action secret
is unavailable", although the kubelet had long projected the new key file.
``PropagatingNodeSecrets`` re-reads the source on a miss, rate-limited, and
takes in only names it lacks: a key it holds is never replaced underneath a
running action (a rotation restarts the executor).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from gpu_fault.adapters.node_action.adapter import NodeActionWorkflowAdapter
from gpu_fault.adapters.node_action.key_propagation import (
    NODE_ACTION_KEY_RELOAD_SECONDS,
    PropagatingNodeSecrets,
)
from gpu_fault.fleet_compatibility import NODE_ACTION_KEY_VERSION_DERIVED
from gpu_fault.node_action_keys import (
    NodeActionKeyUnavailableError,
    resolve_node_action_secret,
)

CLUSTER = "gpu-eks-cluster"
KNOWN = "ip-10-0-33-205.us-west-2.compute.internal"
NEW = "ip-10-0-36-21.us-west-2.compute.internal"
KEY_KNOWN = "k" * 64
KEY_NEW = "n" * 64


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def _adapter(monkeypatch: pytest.MonkeyPatch, keys: Path) -> NodeActionWorkflowAdapter:
    monkeypatch.setenv("GPU_FAULT_NODE_ACTION_KEYS_DIR", str(keys))
    monkeypatch.delenv("GPU_FAULT_NODE_ACTION_KEYS_JSON", raising=False)
    monkeypatch.delenv("GPU_FAULT_NODE_ACTION_SECRET", raising=False)
    monkeypatch.setenv(
        "GPU_FAULT_NODE_AGENT_ENDPOINTS", json.dumps({KNOWN: f"https://{KNOWN}:9099"})
    )
    return NodeActionWorkflowAdapter.from_environment()


def _signing_key(adapter: NodeActionWorkflowAdapter, node_id: str) -> str:
    # What the adapter signs a version-2 action for ``node_id`` with.
    return resolve_node_action_secret(
        adapter.secret,
        adapter.node_secrets,
        CLUSTER,
        node_id,
        NODE_ACTION_KEY_VERSION_DERIVED,
    )


def test_a_key_projected_after_start_is_signed_with_without_a_restart(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    keys = tmp_path / "node-action-keys"
    keys.mkdir()
    (keys / KNOWN).write_text(KEY_KNOWN + "\n", encoding="utf-8")
    adapter = _adapter(monkeypatch, keys)
    assert isinstance(adapter.node_secrets, PropagatingNodeSecrets), (
        "from_environment must keep the re-reading key view"
    )
    clock = Clock()
    adapter.node_secrets.clock = clock

    with pytest.raises(NodeActionKeyUnavailableError):
        _signing_key(adapter, NEW)

    # The sync patched the Secret; the kubelet projects the new file.
    (keys / NEW).write_text(KEY_NEW, encoding="utf-8")
    clock.now += NODE_ACTION_KEY_RELOAD_SECONDS
    with caplog.at_level(logging.INFO, logger="gpu_fault.adapters.node_action"):
        assert _signing_key(adapter, NEW) == KEY_NEW

    assert _signing_key(adapter, KNOWN) == KEY_KNOWN
    messages = [record.getMessage() for record in caplog.records]
    assert any(NEW in message for message in messages), messages
    assert not any(KEY_NEW in message for message in messages), (
        "a key value leaked into a log line"
    )


def test_misses_reread_the_source_at_most_once_per_interval() -> None:
    clock = Clock()
    loads: list[float] = []

    def load() -> dict[str, str]:
        loads.append(clock.now)
        return {}

    secrets = PropagatingNodeSecrets({KNOWN: KEY_KNOWN}, load=load, clock=clock)

    assert secrets.get(KNOWN) == KEY_KNOWN
    assert loads == [], "a hit never touches the source"
    assert secrets.get(NEW) is None
    assert secrets.get(f"{CLUSTER}/{NEW}") is None
    clock.now += NODE_ACTION_KEY_RELOAD_SECONDS - 1
    assert secrets.get(NEW) is None
    assert loads == [1000.0], "misses inside the interval share one re-read"
    clock.now += 1
    assert secrets.get(NEW) is None
    assert loads == [1000.0, 1000.0 + NODE_ACTION_KEY_RELOAD_SECONDS]


def test_a_reread_never_replaces_a_key_it_already_holds() -> None:
    clock = Clock()
    secrets = PropagatingNodeSecrets(
        {KNOWN: KEY_KNOWN}, load=lambda: {KNOWN: "r" * 64, NEW: KEY_NEW}, clock=clock
    )

    assert secrets.get(NEW) == KEY_NEW
    assert secrets[KNOWN] == KEY_KNOWN, "a rotation restarts the executor instead"


def test_a_source_that_cannot_be_read_keeps_the_miss_a_miss(
    caplog: pytest.LogCaptureFixture,
) -> None:
    def load() -> dict[str, str]:
        raise ValueError("node action node secrets must be at least 32 characters")

    secrets = PropagatingNodeSecrets({KNOWN: KEY_KNOWN}, load=load, clock=Clock())

    with caplog.at_level(logging.WARNING, logger="gpu_fault.adapters.node_action"):
        assert secrets.get(NEW) is None

    assert secrets.get(KNOWN) == KEY_KNOWN
    assert "could not re-read the node action keys" in caplog.text


def test_a_plain_mapping_still_works_for_a_hand_built_adapter() -> None:
    adapter = NodeActionWorkflowAdapter(
        {KNOWN: f"https://{KNOWN}:9099"}, "", node_secrets={KNOWN: KEY_KNOWN}
    )

    assert not isinstance(adapter.node_secrets, PropagatingNodeSecrets), (
        "a hand-given mapping is copied as before"
    )
    assert _signing_key(adapter, KNOWN) == KEY_KNOWN

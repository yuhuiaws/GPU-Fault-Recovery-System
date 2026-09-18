"""Fake delivery I/O around real activation, Agent startup and heartbeat logic."""

from __future__ import annotations

import base64
import copy
import uuid
from datetime import datetime, timezone

from fastapi.testclient import TestClient

from gpu_fault.admin.node_key_custody_activation import CustodyActivation
from gpu_fault.fleet import (
    AgentHeartbeat,
    AgentRecord,
    AgentTransitionRequest,
    FleetRegistry,
    SignedAgentHeartbeat,
    sign_agent_heartbeat,
)
from gpu_fault.node_agent import app as agent_app
from gpu_fault.node_agent import config as agent_config
from gpu_fault.store import InMemoryStore
from tests.admin._security_activation_support import ActivationDouble


def start_node(runtime, node: str, *, rotated: bool) -> None:
    """An installer delivers a private file, then a new process reads its env."""
    encoded = runtime.provisioning.api.state["secrets"]["gpu"]["data"][node]
    runtime_dir = runtime.root / ("node-process-" + uuid.uuid4().hex)
    runtime_dir.mkdir(mode=0o700)
    delivered = runtime_dir / "node-key"
    delivered.write_bytes(base64.b64decode(encoded))
    delivered.chmod(0o600)
    pair = runtime.site.pair
    old = runtime.site.raw["agent_snapshot"]["agents"][node]
    current = copy.deepcopy(old)
    current["last_seen_at"] = datetime.now(timezone.utc).isoformat()
    if rotated:
        current["agent_incarnation_id"] = "activated-" + uuid.uuid4().hex
        current["generation"] += 1
    with runtime.site.monkeypatch.context() as patch:
        patch.setenv("GPU_FAULT_NODE_ACTION_SECRET", delivered.read_text())
        patch.setenv("GPU_FAULT_NODE_ACTION_KEY_VERSION", "2")
        patch.setenv("GPU_FAULT_NODE_ALLOWED_OPERATIONS", "COLLECT_DIAGNOSTIC_BUNDLE")
        patch.setenv("GPU_FAULT_NODE_ACTION_DB", str(runtime_dir / "actions.db"))
        patch.setenv("NODE_NAME", node)
        executor = agent_config.executor_from_environment()
    pair.stack.callback(executor.ledger.close)
    executor.agent_generation = current["generation"]
    pair.clients[node].__exit__(None, None, None)
    pair.executors[node] = executor
    pair.clients[node] = pair.stack.enter_context(
        TestClient(agent_app.create_node_agent_app(executor))
    )
    runtime.pending_heartbeats[node] = current


def publish_heartbeats(runtime) -> None:
    store = InMemoryStore()
    for record in runtime.site.raw["agent_snapshot"]["agents"].values():
        store.save_agent(AgentRecord.model_validate(record))
    keys = {
        node: base64.b64decode(value).decode()
        for node, value in runtime.provisioning.api.state["secrets"]["cpu"][
            "data"
        ].items()
    }
    registry = FleetRegistry(
        store, "local-unused-shared-key-" + "x" * 32, node_secrets=keys
    )
    pending_records = {
        **runtime.site.raw["agent_snapshot"]["agents"],
        **runtime.pending_heartbeats,
    }
    for node, pending in pending_records.items():
        now = datetime.now(timezone.utc)
        heartbeat = AgentHeartbeat.model_validate(
            {
                **{
                    key: value
                    for key, value in pending.items()
                    if key in AgentHeartbeat.model_fields
                },
                "observed_at": now,
            }
        )
        accepted = registry.register(
            SignedAgentHeartbeat(
                heartbeat=heartbeat,
                signature=sign_agent_heartbeat(
                    heartbeat, runtime.site.pair.executors[node].secret
                ),
            )
        )
        runtime.site.pair.executors[node].agent_generation = accepted.generation
        runtime.site.raw["agent_snapshot"]["agents"][node] = accepted.model_dump(
            mode="json"
        )
        runtime.site.pair.records[node] = accepted
    runtime.pending_heartbeats.clear()


class RuntimeActivation(ActivationDouble):
    def __init__(self, runtime):
        super().__init__(
            keys=lambda: runtime.provisioning.api.state["secrets"]["gpu"]["data"]
        )
        self.runtime = runtime

    def guard(self, snapshot):
        evidence = super().guard(snapshot)
        store = InMemoryStore()
        current = AgentRecord.model_validate(
            self.runtime.site.raw["agent_snapshot"]["agents"]["node-a"]
        )
        store.save_agent(current)
        registry = FleetRegistry(store, "local-unused-shared-key-" + "x" * 32)
        drained = registry.drain_agent(
            current.cluster_id,
            current.node_id,
            AgentTransitionRequest(
                expected_generation=current.generation,
                transition_id="node-key-"
                + self.runtime.chain.transactions[
                    -1
                ].authorization.statement.transaction_id,
                reason="authorized local activation fixture",
            ),
        )
        self.runtime.site.raw["agent_snapshot"]["agents"]["node-a"] = (
            drained.model_dump(mode="json")
        )
        return evidence

    def install(self, snapshot):
        evidence = super().install(snapshot)
        start_node(self.runtime, "node-a", rotated=True)
        return evidence

    def refresh_cpu(self, snapshot):
        evidence = super().refresh_cpu(snapshot)
        publish_heartbeats(self.runtime)
        return evidence


def activate_runtime(runtime) -> None:
    runtime.pending_heartbeats = {}
    head = runtime.chain.transactions[-1]
    if head.authorization.statement.purpose == "install":
        for node in runtime.provisioning.api.state["secrets"]["gpu"]["data"]:
            start_node(runtime, node, rotated=False)
        publish_heartbeats(runtime)
        return
    io = RuntimeActivation(runtime)
    state_dir = runtime.root / "activation-admin"
    state_dir.mkdir(mode=0o700, exist_ok=True)
    # The state engine only uses state_dir; actual cloud bindings are the I/O
    # boundary double, while the witness below uses the real signed chain.
    from types import SimpleNamespace

    handle = CustodyActivation(
        io, SimpleNamespace(state_dir=state_dir), head.authorization
    )
    io.bind_completed(head.completed.statement)
    handle.prepare()
    result = handle.finish()
    assert result["runtime_activation"] == "DEPLOYED_NOT_WITNESSED"
    runtime.activation_events = io.events

"""A refused reboot reservation gives the agent back with every real ledger.

Live 2026-10-05 (gpu-eks-cluster DESTR-001): the control plane refused the
RESTART_NODE reservation (403), the step failed, and the agent the lifecycle
adapter had revoked for the coming reboot stayed REVOKED. The fix gives the
agents back when the provider-submission ledger provably holds no record for
the step -- but it read "provably no record" as a lookup that returns ``None``,
and no real store does: the memory store and the shared primitives (SQLite,
PostgreSQL) raise ``NotFoundError``, the regional client raises ``KeyError``
for the control plane's empty answer. Every real store fell into the
failing-lookup branch, so the agents were never given back outside the test
stub. The ledger is write-ahead (the INTENDED record is reserved before
``BatchRebootClusterNodes`` / ``RebootInstances``), so the store's
``LookupError`` proves nothing was submitted; a record in any state, a failing
lookup or no durable ledger still keep the agents revoked.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from gpu_fault.cluster_executor.regional_client import RegionalHyperPodSubmissionStore
from gpu_fault.fleet_compatibility import CURRENT_AGENT_PROTOCOL_VERSION
from gpu_fault.hyperpod import (
    HyperPodAction,
    HyperPodSubmissionRecord,
    hyperpod_submission_idempotency_key,
)
from gpu_fault.store import SqliteStore
from tests._builders import active_workflow_executor, build_store, copy_model
from tests.execution._support import (
    NOW,
    AgentHeartbeat,
    AgentLifecycleState,
    FakeHyperPodLifecycle,
    FleetRegistry,
    HyperPodLifecycleStepAdapter,
    SignedAgentHeartbeat,
    WorkflowExecutionRequest,
    WorkflowOperation,
    WorkflowStatus,
    WorkflowStepStatus,
    isolated_kubernetes_adapter,
    sign_agent_heartbeat,
    workflow_state,
)

CLUSTER = "hp-cluster"
KEY = hyperpod_submission_idempotency_key(
    "workflow-active", 0, WorkflowOperation.RESTART_NODE
)
RECORD = HyperPodSubmissionRecord(
    cluster_name=CLUSTER,
    idempotency_key=KEY,
    action=HyperPodAction.REBOOT,
    requested_node_identifiers=["node-a"],
)


class _ControlPlane:
    """The regional client's transport: the control plane answers ``null``
    for a key it holds no record of (``get_remote_hyperpod_submission``)."""

    def __init__(self, record: HyperPodSubmissionRecord | None) -> None:
        self.record = record

    def _get(self, _path: str) -> Any:
        return None if self.record is None else self.record.model_dump(mode="json")


class _FailingLedger:
    def get_hyperpod_submission(self, _cluster_name: str, _key: str) -> Any:
        raise RuntimeError("regional control plane request failed: timeout")


def _memory(record: HyperPodSubmissionRecord | None, _tmp_path: Path) -> Any:
    store = build_store()
    if record is not None:
        store.save_hyperpod_submission(record)
    return store


def _sqlite(record: HyperPodSubmissionRecord | None, tmp_path: Path) -> Any:
    store = SqliteStore(str(tmp_path / "ledger.db"))
    if record is not None:
        store.save_hyperpod_submission(record)
    return store


def _regional(record: HyperPodSubmissionRecord | None, _tmp_path: Path) -> Any:
    return RegionalHyperPodSubmissionStore(_ControlPlane(record))  # type: ignore[arg-type]


def _refused_reservation(tmp_path: Path, ledger: Any):
    """The lifecycle adapter revokes the agent, the reservation is refused
    before any provider call, and the step fails; returns the agent record."""

    class RefusingLifecycle(FakeHyperPodLifecycle):
        store = ledger

        def execute_step(self, step, **kwargs):
            self.calls += 1
            raise RuntimeError(
                "HTTP 403: submission cluster_name does not match the "
                "registered provider cluster for this cluster_id"
            )

    store = build_store()
    secret = "agent-secret-" + "x" * 32
    fleet = FleetRegistry(store, secret, now=lambda: NOW)
    heartbeat = AgentHeartbeat(
        cluster_id="cluster-a",
        node_id="node-a",
        endpoint="http://node-a:9099",
        agent_protocol_version=CURRENT_AGENT_PROTOCOL_VERSION,
        agent_version="0.10.0",
        artifact_sha256="a" * 64,
        policy_version="610",
        runtime_profile_version="active-v1",
        config_digest="c" * 64,
        allowed_operations=[WorkflowOperation.RESET_GPU],
        boot_id="boot-a",
        node_instance_id="instance-a",
        agent_incarnation_id="incarnation-a",
        observed_at=NOW,
    )
    fleet.register(
        SignedAgentHeartbeat(
            heartbeat=heartbeat, signature=sign_agent_heartbeat(heartbeat, secret)
        )
    )
    _, workflow = workflow_state(store, [WorkflowOperation.RESTART_NODE])
    lifecycle = RefusingLifecycle()
    adapter = HyperPodLifecycleStepAdapter(
        lifecycle, registry=fleet, kubernetes_adapter=isolated_kubernetes_adapter()
    )
    step = copy_model(workflow.official_steps[0], execution_owner=adapter.owner)
    store.save_workflow(copy_model(workflow, official_steps=[step]))
    result = active_workflow_executor(
        store, [adapter], {WorkflowOperation.RESTART_NODE}
    ).execute(
        workflow.request_id,
        WorkflowExecutionRequest(
            expected_fencing_token=3,
            isolation_verified_nodes=["node-a"],
            confirm_cluster_name=CLUSTER,
        ),
    )
    assert result.status is WorkflowStatus.FAILED, result
    assert lifecycle.calls == 1
    execution = store.get_workflow(workflow.request_id).step_executions[0]
    assert execution.status is WorkflowStepStatus.FAILED, execution
    return store.get_agent("cluster-a", "node-a")


@pytest.mark.parametrize(
    "ledger", [_memory, _sqlite, _regional], ids=["memory", "sqlite", "regional"]
)
def test_a_store_that_has_no_record_gives_the_agent_back(
    tmp_path: Path, ledger: Any
) -> None:
    record = _refused_reservation(tmp_path, ledger(None, tmp_path))

    assert record.lifecycle_state is AgentLifecycleState.ACTIVE, (
        "the reservation was refused before any provider call: nothing will "
        "reboot the node, its agent heartbeats again"
    )
    assert record.transition_id is None and record.transition_reason is None
    assert "incarnation-a" not in record.retired_incarnation_ids


@pytest.mark.parametrize(
    "ledger", [_memory, _sqlite, _regional], ids=["memory", "sqlite", "regional"]
)
def test_a_store_that_holds_the_record_keeps_the_agent_revoked(
    tmp_path: Path, ledger: Any
) -> None:
    record = _refused_reservation(tmp_path, ledger(RECORD, tmp_path))

    assert record.lifecycle_state is AgentLifecycleState.REVOKED, (
        "a recorded reservation may still be submitted: the agent stays revoked"
    )
    assert record.transition_id == "workflow-active/0/RESTART_NODE/hyperpod-lifecycle"


@pytest.mark.parametrize(
    "ledger", [_FailingLedger(), None], ids=["failing-lookup", "no-durable-ledger"]
)
def test_an_unanswered_lookup_keeps_the_agent_revoked(
    tmp_path: Path, ledger: Any
) -> None:
    record = _refused_reservation(tmp_path, ledger)

    assert record.lifecycle_state is AgentLifecycleState.REVOKED, (
        "nothing proves the reservation was refused: fail closed"
    )

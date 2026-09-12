"""What the regional executor does when it can no longer vouch for a command.

Three situations the lease heartbeat used to only count, never act on:

* the control plane refuses to renew (``N`` times in a row) -- another executor
  may already own the command, so this one must not start more node actions and
  must not post a result under a lease it no longer holds;
* the local view of the lease has expired without a renewal ever landing;
* the renew response says the command was cancelled -- the executor should stop
  starting work and still report, because the control plane records a
  post-cancellation result.

Plus the wiring gap that made every multi-node barrier step fail: the regional
executor has no barrier coordinator, so a barrier-requiring operation on more
than one node must be held with a reason an operator can read, not a silent
FAILED from deep inside the adapter.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from typing import Any

import pytest

from gpu_fault.adapters.node_action.lease_guard import lease_hold_reason
from gpu_fault.cluster_executor import (
    ClusterActionExecutor,
    ClusterExecutorError,
    executor_from_environment,
)
from gpu_fault.execution.models import WorkflowStepOutcome
from gpu_fault.models import WorkflowOperation
from gpu_fault.operation_registry import MULTI_NODE_BARRIER_OPERATIONS
from gpu_fault.regional import RemoteActionCommand, RemoteCommandStatus
from tests._builders import fault_incident, workflow_request, workflow_step
from tests.execution.test_cluster_executor_lease_and_report import (
    EXECUTOR,
    FakeExecutorClient,
    RecordingAdapter,
    stop_events,
)

CLUSTER = "cluster-a"
FENCING_TOKEN = 3
NODE_ACTION_OWNER = "gpu-fault-node-agent"


@pytest.fixture(autouse=True)
def claim_state_outside_shared_tmp(tmp_path, monkeypatch) -> str:
    path = str(tmp_path / "claim-state.json")
    monkeypatch.setenv("GPU_FAULT_CLUSTER_EXECUTOR_CLAIM_STATE_PATH", path)
    return path


def remote_command(
    command_id: str,
    *,
    operation: WorkflowOperation = WorkflowOperation.VALIDATE_HOST,
    node_ids: list[str] | None = None,
    execution_owner: str = "owner-a",
) -> RemoteActionCommand:
    return RemoteActionCommand(
        command_id=command_id,
        cluster_id=CLUSTER,
        workflow_request_id="workflow-a",
        incident_id="incident-a",
        step_index=0,
        fencing_token=FENCING_TOKEN,
        idempotency_key=f"idem-{command_id}",
        lease_token=f"lease-{command_id}",
        step=workflow_step(operation, execution_owner, node_ids=node_ids),
        workflow=workflow_request(
            "workflow-a", "incident-a", fencing_token=FENCING_TOKEN
        ),
        incident=fault_incident("incident-a", "event-a", fencing_token=FENCING_TOKEN),
    )


class GatedAdapter(RecordingAdapter):
    """Waits for the lease heartbeat to act before finishing the step.

    The renewer runs on its own thread, so without a gate the adapter would
    usually finish before the first renewal and the test would race.
    """

    def __init__(self, gate: dict[str, Any]) -> None:
        super().__init__()
        self.gate = gate
        self.hold_reasons: list[str | None] = []

    def execute(self, context: Any) -> WorkflowStepOutcome:
        deadline = time.monotonic() + 5
        while not self.gate["ready"]() and time.monotonic() < deadline:
            time.sleep(0.005)
        self.hold_reasons.append(lease_hold_reason())
        return super().execute(context)


class CancellingClient(FakeExecutorClient):
    def renew(
        self, command: RemoteActionCommand, executor_id: str, lease_seconds: int
    ) -> RemoteActionCommand:
        self.renewals.append((command.command_id, executor_id, lease_seconds))
        return command.model_copy(
            update={
                "cancellation_requested_at": datetime.now(timezone.utc),
                "cancellation_reason": "superseded by a later incident",
            }
        )


def build_executor(
    client: FakeExecutorClient, adapters: list[Any], **overrides: Any
) -> ClusterActionExecutor:
    return ClusterActionExecutor(
        client,
        adapters,
        executor_id=EXECUTOR,
        allowed_namespaces={"training"},
        **overrides,
    )


def test_consecutive_renewal_failures_release_the_command_locally(monkeypatch) -> None:
    """After the limit the lease is treated as lost: no report, a counter."""

    stop_events(monkeypatch)
    client = FakeExecutorClient(
        [remote_command("command-a")],
        renew_error=ClusterExecutorError("stale lease token", status_code=409),
    )
    gate: dict[str, Any] = {"ready": lambda: False}
    adapter = GatedAdapter(gate)
    executor = build_executor(client, [adapter], lease_renewal_failure_limit=1)
    gate["ready"] = lambda: executor.lease_lost_total >= 1

    executor.run_once()

    assert client.completed == []
    assert executor.lease_renewal_failures == 1
    assert executor.lease_lost_total == 1
    assert executor.results_withheld_total == 1
    assert adapter.hold_reasons[0] is not None
    assert "renewal" in adapter.hold_reasons[0]
    assert executor.last_cycle_advanced is False


def test_one_renewal_failure_below_the_limit_still_reports(monkeypatch) -> None:
    stop_events(monkeypatch)
    client = FakeExecutorClient(
        [remote_command("command-a")],
        renew_error=ClusterExecutorError("stale lease token", status_code=409),
    )
    gate: dict[str, Any] = {"ready": lambda: False}
    adapter = GatedAdapter(gate)
    executor = build_executor(client, [adapter], lease_renewal_failure_limit=3)
    gate["ready"] = lambda: executor.lease_renewal_failures >= 1

    executor.run_once()

    assert client.reported("command-a").status is RemoteCommandStatus.SUCCEEDED
    assert executor.lease_lost_total == 0
    assert adapter.hold_reasons == [None]


def test_a_locally_expired_lease_withholds_the_result(monkeypatch) -> None:
    """No renewal ever landed and the lease window passed: another executor
    may hold the command now, so the result stays with the agent ledger."""

    class ImmediateStop:
        """A stop event that is already set: the renewer never runs."""

        def wait(self, _timeout: float | None = None) -> bool:
            return True

        def set(self) -> None:
            return None

    monkeypatch.setattr("gpu_fault.cluster_executor.lease.Event", ImmediateStop)
    clock = {"now": 1000.0}
    client = FakeExecutorClient([remote_command("command-a")])

    class SlowAdapter(RecordingAdapter):
        def execute(self, context: Any) -> WorkflowStepOutcome:
            clock["now"] += 121.0
            return super().execute(context)

    executor = build_executor(
        client, [SlowAdapter()], lease_seconds=120, clock=lambda: clock["now"]
    )

    executor.run_once()

    assert client.completed == []
    assert executor.lease_lost_total == 1
    assert executor.results_withheld_total == 1


def test_cancellation_in_the_renew_response_is_seen_and_still_reported(
    monkeypatch,
) -> None:
    stop_events(monkeypatch)
    client = CancellingClient([remote_command("command-a")])
    gate: dict[str, Any] = {"ready": lambda: False}
    adapter = GatedAdapter(gate)
    executor = build_executor(client, [adapter])
    gate["ready"] = lambda: executor.cancellations_observed_total >= 1

    executor.run_once()

    assert executor.cancellations_observed_total == 1
    assert adapter.hold_reasons[0] is not None
    assert "superseded" in adapter.hold_reasons[0]
    assert client.reported("command-a").status is RemoteCommandStatus.SUCCEEDED
    assert executor.results_withheld_total == 0


def test_the_failure_limit_is_validated_and_read_from_the_environment(
    monkeypatch,
) -> None:
    with pytest.raises(ClusterExecutorError, match="failure limit"):
        build_executor(FakeExecutorClient(), [], lease_renewal_failure_limit=0)
    monkeypatch.setenv("GPU_FAULT_CLUSTER_EXECUTOR_LEASE_FAILURE_LIMIT", "5")
    monkeypatch.setenv("GPU_FAULT_CONTROL_PLANE_URL", "https://control.example")
    monkeypatch.setenv("GPU_FAULT_CONTROL_PLANE_TOKEN", "token")
    monkeypatch.setenv("GPU_FAULT_CLUSTER_ID", CLUSTER)
    monkeypatch.delenv("GPU_FAULT_ENABLE_HYPERPOD_ADAPTER", raising=False)
    monkeypatch.delenv("GPU_FAULT_ENABLE_NODE_ACTION_ADAPTER", raising=False)

    class FakeKubernetesAdapter:
        owner = "gpu-fault-kubernetes-adapter"

        def __init__(self, **_kwargs: Any) -> None:
            pass

    monkeypatch.setattr(
        "gpu_fault.cluster_executor.bootstrap.KubernetesWorkflowAdapter",
        FakeKubernetesAdapter,
    )

    executor = executor_from_environment()

    assert executor.lease_renewal_failure_limit == 5


class BarrierlessAdapter(RecordingAdapter):
    barriers = None


def test_a_multi_node_barrier_step_is_held_when_no_coordinator_is_wired() -> None:
    client = FakeExecutorClient(
        [
            remote_command(
                "command-a",
                operation=WorkflowOperation.RESET_GPU,
                node_ids=["node-a", "node-b"],
            )
        ]
    )
    adapter = BarrierlessAdapter()
    executor = build_executor(client, [adapter])

    executor.run_once()

    result = client.reported("command-a")
    assert result.status is RemoteCommandStatus.WAITING
    assert result.status_source == "executor-barrier-unavailable"
    assert result.details["multi_node_barrier_unavailable"] is True
    assert result.details["node_ids"] == ["node-a", "node-b"]
    assert "barrier coordinator" in result.details["reason"]
    assert adapter.contexts == []
    assert executor.barrier_unavailable_holds_total == 1


def test_a_single_node_barrier_operation_runs_without_a_coordinator() -> None:
    client = FakeExecutorClient(
        [remote_command("command-a", operation=WorkflowOperation.RESET_GPU)]
    )
    adapter = BarrierlessAdapter()
    executor = build_executor(client, [adapter])

    executor.run_once()

    assert client.reported("command-a").status is RemoteCommandStatus.SUCCEEDED
    assert len(adapter.contexts) == 1
    assert executor.barrier_unavailable_holds_total == 0


@pytest.mark.parametrize(
    "operation", sorted(MULTI_NODE_BARRIER_OPERATIONS, key=lambda item: item.value)
)
def test_regional_wiring_either_coordinates_or_refuses_barrier_operations(
    monkeypatch, operation: WorkflowOperation
) -> None:
    """The adapter set ``executor_from_environment`` builds must never let a
    barrier operation reach ``_execute_multi_node_reset`` unwired."""

    monkeypatch.setenv("GPU_FAULT_CONTROL_PLANE_URL", "https://control.example")
    monkeypatch.setenv("GPU_FAULT_CONTROL_PLANE_TOKEN", "token")
    monkeypatch.setenv("GPU_FAULT_CLUSTER_ID", CLUSTER)
    monkeypatch.setenv("GPU_FAULT_ENABLE_NODE_ACTION_ADAPTER", "true")
    monkeypatch.setenv("GPU_FAULT_NODE_ACTION_SECRET", "n" * 32)
    monkeypatch.delenv("GPU_FAULT_ENABLE_HYPERPOD_ADAPTER", raising=False)
    monkeypatch.delenv("GPU_FAULT_NODE_AGENT_ENDPOINTS", raising=False)

    class FakeKubernetesAdapter:
        owner = "gpu-fault-kubernetes-adapter"

        def __init__(self, **_kwargs: Any) -> None:
            pass

        def supports(self, _step: Any) -> bool:
            return False

    monkeypatch.setattr(
        "gpu_fault.cluster_executor.bootstrap.KubernetesWorkflowAdapter",
        FakeKubernetesAdapter,
    )
    # The fleet preflight is a control-plane round trip; it is not what this
    # test is about and it must not decide the outcome here.
    monkeypatch.setattr(
        "gpu_fault.cluster_executor.dispatch.fleet_preflight_reason",
        lambda *_args, **_kwargs: None,
    )
    executor = executor_from_environment()
    node_action = next(
        adapter for adapter in executor.adapters if adapter.owner == NODE_ACTION_OWNER
    )
    client = FakeExecutorClient(
        [
            remote_command(
                "command-a",
                operation=operation,
                node_ids=["node-a", "node-b"],
                execution_owner=NODE_ACTION_OWNER,
            )
        ]
    )
    executor.client = client

    executor.run_once()

    result = client.reported("command-a")
    if node_action.barriers is None:
        assert result.status is RemoteCommandStatus.WAITING
        assert result.details["multi_node_barrier_unavailable"] is True
    else:
        assert "multi_node_barrier_unavailable" not in result.details


def test_counters_are_exposed_as_a_snapshot_and_in_the_claim_breadcrumb(
    claim_state_outside_shared_tmp: str,
) -> None:
    client = FakeExecutorClient([remote_command("command-a")])
    executor = build_executor(client, [RecordingAdapter()])

    executor.run_once()
    snapshot = executor.metrics_snapshot()

    assert snapshot["claimed_total"] == 1
    assert {
        "reported_failures",
        "unexpected_failures",
        "lease_renewal_failures",
        "lease_lost_total",
        "results_withheld_total",
        "cancellations_observed_total",
        "barrier_unavailable_holds_total",
        "last_successful_claim_at",
    } <= set(snapshot)
    with open(claim_state_outside_shared_tmp, encoding="utf-8") as handle:
        breadcrumb = json.load(handle)
    assert breadcrumb["counters"]["claimed_total"] == 1


def test_the_breadcrumb_advertises_owners_before_the_first_claim(
    claim_state_outside_shared_tmp: str,
) -> None:
    """A cluster still PENDING in the registry refuses every claim with 423,
    so a breadcrumb written only by a successful claim left the readiness
    probe advertising no owners and the join's executor rollout never became
    Ready (live, 2026-09-12). Owners are configuration; the loop writes them
    before it claims anything, with no claim timestamp yet."""

    client = FakeExecutorClient([])
    executor = build_executor(client, [RecordingAdapter()])
    executor.request_stop("test: advertise only")

    executor.run()

    with open(claim_state_outside_shared_tmp, encoding="utf-8") as handle:
        breadcrumb = json.load(handle)
    assert breadcrumb["execution_owners"] == executor.execution_owners, (
        "the pre-claim breadcrumb must advertise the configured owners"
    )
    assert breadcrumb["last_successful_claim_at"] is None, (
        "no claim has happened, so the breadcrumb must not fake one"
    )

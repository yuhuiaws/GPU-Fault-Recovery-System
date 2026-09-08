"""What the executor re-pays on every poll of a command that already started.

cluster-executor review F6. A node action that answers WAITING is re-claimed
about every two seconds until the Agent finishes, and each of those cycles used
to redo the whole destructive fleet preflight: a rollout-fence GET plus a fleet
readiness POST, both control-plane round trips, for a mutation that has already
been handed to the Agent and cannot be called back.

Skipping is only safe in that exact shape. The preflight is the fail-closed gate
in front of every destructive action, so it may be skipped only when
``result_details`` proves a mutation already began (``node_action_command_id``,
written by the adapter when the Agent accepted the command) -- never for a fresh
command, where the fence is the only thing standing between a node reset and a
cluster-wide rollout.
"""

from __future__ import annotations

from typing import Any

from gpu_fault.cluster_executor import ClusterActionExecutor
from gpu_fault.models import WorkflowOperation
from gpu_fault.regional import RemoteCommandStatus
from tests.execution.test_cluster_executor_lease_and_report import (
    EXECUTOR,
    FakeExecutorClient,
    RecordingAdapter,
    remote_command,
)


class CountingFenceRegistry:
    """Answers the rollout fence closed and counts every fence read.

    The regional proxy owns no store and answers the fence with a control-plane
    round trip, which ``fleet_rollout_fence`` resolves by this method's
    presence. Answering closed makes each read observable in the result too.
    """

    def __init__(self) -> None:
        self.fence_reads = 0

    def fleet_rollout_fence_deployments(self, cluster_id: str) -> list[str]:
        self.fence_reads += 1
        return ["deployment-1"]


def build_executor(
    client: FakeExecutorClient, adapters: list[Any]
) -> ClusterActionExecutor:
    return ClusterActionExecutor(
        client, adapters, executor_id=EXECUTOR, allowed_namespaces={"training"}
    )


def destructive_command(
    command_id: str, *, result_details: dict[str, Any] | None = None
):
    return remote_command(
        command_id,
        operation=WorkflowOperation.REMEDIATE_DRIVER,
        node_ids=["node-a"],
        result_details=result_details,
        workflow_steps=True,
    )


def test_the_fleet_preflight_is_skipped_once_the_node_action_has_begun() -> None:
    """A mutation already accepted by the Agent cannot be fenced back.

    Re-running the fence and readiness checks costs two control-plane round
    trips per poll and, worse, can flip a running REMEDIATE_DRIVER to WAITING
    because an unrelated rollout started -- stranding the poll that was the only
    way to learn the driver install's outcome.
    """

    registry = CountingFenceRegistry()
    adapter = RecordingAdapter()
    adapter.registry = registry  # type: ignore[attr-defined]
    client = FakeExecutorClient(
        [
            destructive_command(
                "command-a",
                result_details={"node_action_command_id": "idem-command-a/node-a"},
            )
        ]
    )
    executor = build_executor(client, [adapter])

    assert executor.run_once() == 1, "the command must still be executed"

    result = client.reported("command-a")
    assert result.status is RemoteCommandStatus.SUCCEEDED, result
    assert registry.fence_reads == 0, (
        "the rollout fence was read again for a mutation that had already "
        f"started: {registry.fence_reads} read(s)"
    )
    assert adapter.contexts != [], (
        "the adapter must be reached so the in-flight node action can be polled"
    )


def test_a_fresh_destructive_command_is_still_fenced() -> None:
    """Without the pointer nothing has started, so the fence still decides."""

    registry = CountingFenceRegistry()
    adapter = RecordingAdapter()
    adapter.registry = registry  # type: ignore[attr-defined]
    client = FakeExecutorClient([destructive_command("command-a")])
    executor = build_executor(client, [adapter])

    assert executor.run_once() == 1, "the hold must be reported, not raised"

    result = client.reported("command-a")
    assert result.status is RemoteCommandStatus.WAITING, result
    assert result.details["fleet_preflight_blocked"] is True, result.details
    assert registry.fence_reads == 1, (
        "a fresh destructive command must still be checked against the fence"
    )
    assert adapter.contexts == [], "the adapter must not run behind a closed fence"


def test_an_unrelated_continuation_key_does_not_open_the_fence() -> None:
    """Only the node-action pointer proves a mutation began.

    ``result_details`` carries whatever the previous cycle left behind --
    baselines, spare-failover state, retry counts. Treating any non-empty
    details as "already started" would open the fence for a command that has
    merely been held once.
    """

    registry = CountingFenceRegistry()
    adapter = RecordingAdapter()
    adapter.registry = registry  # type: ignore[attr-defined]
    client = FakeExecutorClient(
        [
            destructive_command(
                "command-a", result_details={"agent_baselines": {"node-a": 1}}
            )
        ]
    )
    executor = build_executor(client, [adapter])

    executor.run_once()

    result = client.reported("command-a")
    assert result.status is RemoteCommandStatus.WAITING, result
    assert registry.fence_reads == 1, (
        "a replayed hold is not a started mutation; the fence must still run"
    )

"""What the executor re-pays on every poll of a command that already started.

cluster-executor review F6. A node action that answers WAITING is re-claimed
about every two seconds until the Agent finishes, and each of those cycles used
to redo the whole destructive fleet preflight: a rollout-fence GET plus a fleet
readiness POST, both control-plane round trips, for a mutation that has already
been handed to the Agent and cannot be called back.

Skipping is only safe in that exact shape. The preflight is the fail-closed gate
in front of every destructive action, so it may be skipped only on evidence that
the *Agent accepted* the command. The pointer alone is not that evidence: the
adapter stamps ``node_action_command_id`` just as readily on a connection that
was refused (the normal state of a faulty node -- nothing was ever submitted),
on an agent that answered 503, and on an envelope the agent will never accept
again. In all three the mutation has not begun, and the fence is the only thing
standing between a node reset and a cluster-wide rollout.

The details fed to the executor here are produced by the real transport rather
than written by hand, because that is exactly the contract that drifted.
"""

from __future__ import annotations

import io
from email.message import Message
from typing import Any, Callable
from urllib import error as urllib_error

import pytest

from gpu_fault.adapters import NodeActionWorkflowAdapter
from gpu_fault.cluster_executor import ClusterActionExecutor
from gpu_fault.models import WorkflowOperation
from gpu_fault.node_agent import NodeActionExecutionState, NodeActionSubmission
from gpu_fault.regional import RemoteCommandResult, RemoteCommandStatus
from tests.execution.test_cluster_executor_lease_and_report import (
    EXECUTOR,
    FakeExecutorClient,
    RecordingAdapter,
    remote_command,
)
from tests.execution.test_node_action_round_trips import SECRET, step_context


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


def poll_a_destructive_command(
    result_details: dict[str, Any],
) -> tuple[CountingFenceRegistry, RecordingAdapter, RemoteCommandResult]:
    """Run one executor cycle over a REMEDIATE_DRIVER carrying these details."""

    registry = CountingFenceRegistry()
    adapter = RecordingAdapter()
    adapter.registry = registry  # type: ignore[attr-defined]
    client = FakeExecutorClient(
        [destructive_command("command-a", result_details=result_details)]
    )
    executor = build_executor(client, [adapter])

    assert executor.run_once() == 1, "the outcome must be reported, not raised"

    return registry, adapter, client.reported("command-a")


def refused_connection(*_args: Any, **_kwargs: Any) -> Any:
    """The faulty node's normal state: the submit never left this process."""

    raise urllib_error.URLError(ConnectionRefusedError(111, "Connection refused"))


def rejecting_agent(status: int, body: bytes) -> Callable[..., Any]:
    """An agent that answered, and answered no."""

    def wire(request: Any, *_args: Any, **_kwargs: Any) -> Any:
        raise urllib_error.HTTPError(
            request.full_url, status, "rejected", Message(), io.BytesIO(body)
        )

    return wire


def accepting_agent(request: Any, *_args: Any, **_kwargs: Any) -> Any:
    """The steady-state poll: the agent's ledger holds the command PENDING."""

    command_id = request.full_url.split("command_id=")[1].split("&")[0]
    body = (
        NodeActionSubmission(
            command_id=command_id, state=NodeActionExecutionState.PENDING
        )
        .model_dump_json()
        .encode()
    )

    class Response:
        def read(self) -> bytes:
            return body

        def __enter__(self) -> Any:
            return self

        def __exit__(self, *_exc: object) -> None:
            return None

    return Response()


def node_action_details(
    monkeypatch: pytest.MonkeyPatch, wire: Callable[..., Any]
) -> dict[str, Any]:
    """The details the real transport writes for one REMEDIATE_DRIVER send.

    Hand-written details drift away from the adapter, and the drift is the
    defect: these guards are about what the adapter actually stamps.
    """

    monkeypatch.setattr("gpu_fault.adapters.node_action.transport.urlopen", wire)
    adapter = NodeActionWorkflowAdapter({"node-a": "http://node-a:9099"}, SECRET)

    outcome = adapter.execute(
        step_context(adapter, operation=WorkflowOperation.REMEDIATE_DRIVER)
    )

    details = dict(outcome.details or {})
    assert details.get("node_action_command_id"), (
        "the trap these guards exist for is gone: the adapter no longer stamps "
        f"a node-action pointer for this send: {details}"
    )
    return details


def test_the_fleet_preflight_is_skipped_once_the_agent_accepted_the_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A mutation already accepted by the Agent cannot be fenced back.

    Re-running the fence and readiness checks costs two control-plane round
    trips per poll and, worse, can flip a running REMEDIATE_DRIVER to WAITING
    because an unrelated rollout started -- stranding the poll that was the only
    way to learn the driver install's outcome.
    """

    details = node_action_details(monkeypatch, accepting_agent)
    assert details["node_action_state"] == "PENDING", details
    assert details["node_action_accepted"] is True, (
        "acceptance has to be stamped by the one path that parsed an agent "
        f"acceptance, so the executor can tell it from a pointer: {details}"
    )

    registry, adapter, result = poll_a_destructive_command(details)

    assert result.status is RemoteCommandStatus.SUCCEEDED, result
    assert registry.fence_reads == 0, (
        "the rollout fence was read again for a mutation that had already "
        f"started: {registry.fence_reads} read(s)"
    )
    assert adapter.contexts != [], (
        "the adapter must be reached so the in-flight node action can be polled"
    )


@pytest.mark.parametrize(
    "wire",
    [
        pytest.param(refused_connection, id="connection-refused"),
        pytest.param(
            rejecting_agent(
                503,
                b'{"detail": {"code": "AGENT_BUSY", "retryable": true, '
                b'"message": "another action is running"}}',
            ),
            id="agent-answered-503",
        ),
        pytest.param(
            rejecting_agent(
                409,
                b'{"detail": {"code": "COMMAND_EXPIRED", '
                b'"requires_new_command": true, "message": "expired"}}',
            ),
            id="agent-requires-a-new-command",
        ),
    ],
)
def test_a_pointer_the_agent_never_accepted_does_not_open_the_fence(
    monkeypatch: pytest.MonkeyPatch, wire: Callable[..., Any]
) -> None:
    """Three ways the pointer exists while nothing has been mutated.

    A refused connection means the submit never left the executor; a 503 and a
    COMMAND_EXPIRED are the agent explicitly refusing the envelope. Treating the
    pointer alone as "the mutation began" opened the destructive fence for a
    REMEDIATE_DRIVER that had not started -- on a node that is unreachable,
    which is precisely when a concurrent fleet rollout is most likely to be the
    reason.
    """

    details = node_action_details(monkeypatch, wire)
    assert details["node_action_state"] != "PENDING", (
        f"this wire was supposed to model a send the agent never took: {details}"
    )
    assert "node_action_accepted" not in details, (
        f"acceptance must never be stamped for a refused send: {details}"
    )

    registry, adapter, result = poll_a_destructive_command(details)

    assert result.status is RemoteCommandStatus.WAITING, result
    assert result.details["fleet_preflight_blocked"] is True, result.details
    assert registry.fence_reads == 1, (
        "a destructive command the agent never accepted must still be checked "
        f"against the rollout fence: {registry.fence_reads} read(s)"
    )
    assert adapter.contexts == [], "the adapter must not run behind a closed fence"


def test_a_fresh_destructive_command_is_still_fenced() -> None:
    """Without the pointer nothing has started, so the fence still decides."""

    registry, adapter, result = poll_a_destructive_command({})

    assert result.status is RemoteCommandStatus.WAITING, result
    assert result.details["fleet_preflight_blocked"] is True, result.details
    assert registry.fence_reads == 1, (
        "a fresh destructive command must still be checked against the fence"
    )
    assert adapter.contexts == [], "the adapter must not run behind a closed fence"


def test_an_unrelated_continuation_key_does_not_open_the_fence() -> None:
    """Only proven agent acceptance opens it.

    ``result_details`` carries whatever the previous cycle left behind --
    baselines, spare-failover state, retry counts. Treating any non-empty
    details as "already started" would open the fence for a command that has
    merely been held once.
    """

    registry, _adapter, result = poll_a_destructive_command(
        {"agent_baselines": {"node-a": 1}}
    )

    assert result.status is RemoteCommandStatus.WAITING, result
    assert registry.fence_reads == 1, (
        "a replayed hold is not a started mutation; the fence must still run"
    )

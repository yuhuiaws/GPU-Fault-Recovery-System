"""What one failed agent-record read does to a node-action step.

The transport reads the agent record (key version, pinned certificate) from the
control plane before every send. A record that is *missing* is a verdict about
the node; a control plane that did not answer, or answered 5xx, is not -- yet
both used to be wrapped into the same ``ValueError`` and folded into a terminal
FAILED step. For a single-node RESET_GPU that meant one 503 from the control
plane climbed the hardware ladder to a reboot.
"""

from __future__ import annotations

from typing import Any
from urllib.error import URLError

import pytest

from gpu_fault.adapters import NodeActionWorkflowAdapter
from gpu_fault.cluster_executor import ClusterExecutorError
from gpu_fault.models import WorkflowOperation
from gpu_fault.regional import RemoteCommandResult, RemoteCommandStatus
from tests.execution._support import StubFleetRegistry
from tests.execution.test_cluster_executor_lease_and_report import (
    FakeExecutorClient,
    build_executor,
    remote_command,
)

ENDPOINT = "http://node-a:9099"
SECRET = "s" * 32


class FailingRecordRegistry(StubFleetRegistry):
    """Addresses node-a normally; the agent-record read itself fails."""

    def __init__(self, error: BaseException) -> None:
        super().__init__({"node-a": ENDPOINT})
        self.error = error

    def get_agent(self, cluster_id: str, node_id: str) -> Any:
        raise self.error


def run_one_health_snapshot(error: BaseException) -> RemoteCommandResult:
    def sender(_endpoint: str, _envelope: Any) -> Any:
        pytest.fail("nothing may be sent without the agent record")

    adapter = NodeActionWorkflowAdapter(
        {}, SECRET, registry=FailingRecordRegistry(error), sender=sender
    )
    command = remote_command(
        "command-a",
        operation=WorkflowOperation.TRIGGER_HEALTH_SNAPSHOT,
        execution_owner=adapter.owner,
    )
    client = FakeExecutorClient([command])
    executor = build_executor(client, [adapter])

    assert executor.run_once() == 1, "the outcome must be reported, not raised"

    return client.reported("command-a")


def unanswered_request() -> ClusterExecutorError:
    """The client's shape for a request that never got an answer: no status
    code, the transport failure as the cause (``ClusterExecutorClient._send``)."""

    error = ClusterExecutorError(
        "regional control plane request failed: URLError: connection refused"
    )
    error.__cause__ = URLError(ConnectionRefusedError(111, "Connection refused"))
    return error


@pytest.mark.parametrize(
    ("error", "status_source"),
    [
        pytest.param(
            ClusterExecutorError("control plane answered 503", status_code=503),
            "executor-retryable-control-plane",
            id="503",
        ),
        pytest.param(
            unanswered_request(), "executor-retryable-transport", id="no-answer"
        ),
    ],
)
def test_an_unanswered_agent_record_read_holds_the_step(
    error: BaseException, status_source: str
) -> None:
    """No answer from the control plane is not a verdict about the GPU."""

    result = run_one_health_snapshot(error)

    assert result.status is RemoteCommandStatus.WAITING, (
        "a control plane that did not answer failed the step terminally, which "
        f"is what climbs RESET_GPU to a reboot on one 503: {result}"
    )
    assert result.status_source == status_source, result


@pytest.mark.parametrize(
    "error",
    [
        pytest.param(KeyError(("cluster-a", "node-a")), id="404-as-keyerror"),
        pytest.param(
            ClusterExecutorError("forbidden", status_code=403), id="4xx-refusal"
        ),
    ],
)
def test_a_missing_or_refused_agent_record_still_fails_the_step(
    error: BaseException,
) -> None:
    result = run_one_health_snapshot(error)

    assert result.status is RemoteCommandStatus.FAILED, result
    assert "agent key metadata is unavailable for node-a" in (result.error or ""), (
        result.error
    )

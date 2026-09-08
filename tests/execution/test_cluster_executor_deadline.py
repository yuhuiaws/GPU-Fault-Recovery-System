"""The regional executor's execution deadline, liveness breadcrumb and SIGTERM.

Three ways a data-plane executor used to hold work forever (cluster-executor
review F5/F8):

* nothing bounded ``_execute``. The renewal thread runs until the command
  returns, so a wedged boto3 call, a proxy read that never times out or an
  adapter defect kept the command LEASED indefinitely -- and because ``claim``
  excludes it, no other replica could ever take it over;
* the Deployment had a startup and a readiness probe but no liveness probe, so
  a claim loop that stopped making progress was never restarted. The readiness
  breadcrumb cannot answer that question: it is only written after a
  *successful* claim, so a control-plane outage makes it stale while the loop
  is perfectly alive;
* SIGTERM was not handled at all. A rollout killed the process mid-command and
  the lease then sat for a full window before the sibling re-claimed it.

The deadline must not turn a fail-closed path into a fail-open one: for a node
action the executor cannot know whether the Agent finished, so the timeout
result says the outcome is unknown (``manual_confirmation_required``, exactly
what an INTERRUPTED node action reports) instead of a plain FAILED.
"""

from __future__ import annotations

import json
import os
import signal
import time
from pathlib import Path
from threading import Event
from typing import Any

import pytest
import yaml

from gpu_fault.cluster_executor import (
    DEFAULT_LIVENESS_STATE_PATH,
    LIVENESS_STALE_AFTER_SECONDS,
    ClusterActionExecutor,
    ClusterExecutorError,
    executor_from_environment,
)
from gpu_fault.execution.models import WorkflowStepOutcome
from gpu_fault.models import WorkflowOperation
from gpu_fault.regional import RemoteCommandStatus
from tests.execution.test_cluster_executor_lease_and_report import (
    EXECUTOR,
    FakeExecutorClient,
    RecordingAdapter,
    remote_command,
)

ROOT = Path(__file__).resolve().parents[2]
EXECUTOR_MANIFEST = ROOT / "deploy/dataplane/cluster-action-executor.yaml"
CLUSTER = "cluster-a"


@pytest.fixture(autouse=True)
def breadcrumbs_outside_shared_tmp(tmp_path, monkeypatch) -> str:
    """Keep both breadcrumbs out of the shared ``/tmp`` defaults.

    The liveness path is derived from the claim-state path, so pointing the
    claim state at ``tmp_path`` moves both under this test's own directory
    instead of one fixed name every xdist worker would rewrite.
    """

    monkeypatch.setenv(
        "GPU_FAULT_CLUSTER_EXECUTOR_CLAIM_STATE_PATH",
        str(tmp_path / "claim-state.json"),
    )
    return str(tmp_path / "executor-loop-alive")


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


class StuckAdapter(RecordingAdapter):
    """An adapter that never returns until the test releases it.

    Models the case the deadline exists for: a call inside the adapter that has
    no timeout of its own (a boto3 describe, a control-plane proxy read), which
    no amount of cancellation from the control plane can interrupt.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.entered = Event()
        self.released = Event()
        self.returned = Event()

    def execute(self, context: Any) -> WorkflowStepOutcome:
        self.entered.set()
        self.released.wait(10)
        try:
            return super().execute(context)
        finally:
            self.returned.set()


def renew_stop_marks(monkeypatch, client: FakeExecutorClient) -> list[int]:
    """Record how many results were posted each time a renewer was stopped.

    The deadline has to stop the heartbeat *before* it posts, or the executor
    would be reporting a verdict for a command it is still (invisibly)
    executing while telling the control plane the lease is alive.
    """

    marks: list[int] = []

    class MarkingEvent(Event):
        def set(self) -> None:
            marks.append(len(client.completed))
            super().set()

    monkeypatch.setattr("gpu_fault.cluster_executor.Event", MarkingEvent)
    return marks


def test_a_stuck_adapter_stops_being_renewed_after_the_execution_cap(
    monkeypatch,
) -> None:
    """Past the cap the command is given up on, not held under a live lease."""

    client = FakeExecutorClient([remote_command("command-a")])
    marks = renew_stop_marks(monkeypatch, client)
    adapter = StuckAdapter()
    executor = build_executor(client, [adapter], max_execution_seconds=0.2)

    try:
        assert executor.run_once() == 1, "the claim itself must still succeed"
        assert adapter.entered.is_set(), "the adapter never started"
        result = client.reported("command-a")
        assert result.status is RemoteCommandStatus.FAILED, result
        assert result.status_source == "executor-execution-timeout", result
        assert result.details["execution_timeout"] is True, result.details
        assert result.details["outcome_unknown"] is True, result.details
        assert executor.execution_timeouts_total == 1, (
            "the abandoned command was not counted"
        )
        assert executor.metrics_snapshot()["stuck_executions"] == 1, (
            "the abandoned thread must be visible to the operator"
        )
        assert client.renewals == [], (
            "a command past the execution cap must not be renewed"
        )
        assert marks[0] == 0, (
            "the renewer must be stopped before the timeout result is posted, "
            f"but the first stop happened after {marks[0]} post(s)"
        )
    finally:
        adapter.released.set()
    assert adapter.returned.wait(10), "the stuck adapter never returned"
    assert [command_id for command_id, _ in client.completed] == ["command-a"], (
        "a late result was posted under a lease this executor stopped renewing"
    )


def test_a_stuck_node_action_reports_an_unknown_outcome_not_a_plain_failure() -> None:
    """A RESET_GPU may still be running on the node; FAILED alone would lie.

    The Agent holds its own ledger row and the executor never saw the answer,
    so the timeout mirrors INTERRUPTED semantics: FAILED, with the outcome
    marked unknown and manual confirmation demanded.
    """

    command = remote_command(
        "command-a",
        operation=WorkflowOperation.RESET_GPU,
        result_details={"node_action_command_id": "idem-command-a/node-a"},
    )
    client = FakeExecutorClient([command])
    adapter = StuckAdapter()
    executor = build_executor(client, [adapter], max_execution_seconds=0.2)

    try:
        executor.run_once()
        result = client.reported("command-a")
        assert result.status_source == "executor-execution-timeout", result
        assert result.details["manual_confirmation_required"] is True, (
            "a node action whose outcome is unknown must not close silently: "
            f"{result.details}"
        )
        assert result.details["node_action_command_id"] == ("idem-command-a/node-a"), (
            "the operator needs the ledger row to confirm by hand"
        )
        assert "unknown" in (result.error or ""), result.error
    finally:
        adapter.released.set()
    assert adapter.returned.wait(10), "the stuck adapter never returned"


def test_a_non_mutating_timeout_does_not_demand_manual_confirmation() -> None:
    """VALIDATE_HOST changes nothing, so the flag would only cost attention."""

    client = FakeExecutorClient([remote_command("command-a")])
    adapter = StuckAdapter()
    executor = build_executor(client, [adapter], max_execution_seconds=0.2)

    try:
        executor.run_once()
        result = client.reported("command-a")
        assert "manual_confirmation_required" not in result.details, result.details
    finally:
        adapter.released.set()
    assert adapter.returned.wait(10), "the stuck adapter never returned"


def test_the_execution_cap_is_validated_and_read_from_the_environment(
    monkeypatch,
) -> None:
    with pytest.raises(ClusterExecutorError, match="execution seconds"):
        build_executor(FakeExecutorClient(), [], max_execution_seconds=0)
    monkeypatch.setenv("GPU_FAULT_CLUSTER_EXECUTOR_MAX_EXECUTION_SECONDS", "900")
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
        "gpu_fault.cluster_executor.KubernetesWorkflowAdapter", FakeKubernetesAdapter
    )

    executor = executor_from_environment()

    assert executor.max_execution_seconds == 900, (
        "the cap must be operator-settable without a code change"
    )


def test_sigterm_stops_claiming_and_stops_renewing(monkeypatch) -> None:
    """A rollout must release the lease instead of parking it for a window.

    Two halves: ``run()`` stops claiming, and a command already in flight stops
    being renewed so the local lease lapses and the sibling replica can take
    the command over as soon as the control plane's window passes.
    """

    class OneRenewalStopEvent(Event):
        """Lets the renew loop reach its body exactly once, without sleeping."""

        def __init__(self) -> None:
            super().__init__()
            self.waits = 0

        def wait(self, timeout: float | None = None) -> bool:
            self.waits += 1
            return self.waits > 1

    monkeypatch.setattr("gpu_fault.cluster_executor.Event", OneRenewalStopEvent)
    client = FakeExecutorClient([remote_command("command-a")])
    executor = build_executor(client, [RecordingAdapter()])
    previous = signal.getsignal(signal.SIGTERM)
    try:
        executor.install_signal_handlers()
        signal.raise_signal(signal.SIGTERM)
    finally:
        signal.signal(signal.SIGTERM, previous)

    assert executor.stop_requested is True, "SIGTERM did not reach the executor"
    assert "SIGTERM" in (executor.stop_reason or ""), executor.stop_reason

    executor.run_once()

    assert client.renewals == [], (
        "an in-flight command must stop being renewed so the lease lapses fast"
    )
    assert client.reported("command-a").status is RemoteCommandStatus.SUCCEEDED, (
        "the work already done still has to be reported"
    )

    claims_before = len(client.claims)
    executor.run()

    assert len(client.claims) == claims_before, (
        "run() must not claim anything more after a stop was requested"
    )


class BreadcrumbWatchingAdapter(RecordingAdapter):
    """Blocks until the poll loop *refreshes* the liveness breadcrumb.

    Existence is not enough: the loop writes the breadcrumb before it claims,
    so the file is already there when this adapter starts. What has to be true
    is that it keeps being rewritten while the command runs.
    """

    def __init__(self, path: str) -> None:
        super().__init__()
        self.path = path
        self.saw_refresh = False

    def execute(self, context: Any) -> WorkflowStepOutcome:
        before = os.stat(self.path).st_mtime_ns
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if os.stat(self.path).st_mtime_ns != before:
                self.saw_refresh = True
                break
            time.sleep(0.005)
        return super().execute(context)


def test_the_poll_loop_refreshes_its_liveness_breadcrumb_while_a_command_runs(
    breadcrumbs_outside_shared_tmp: str,
) -> None:
    """A long-but-live command must not look like a wedged loop.

    The liveness probe reads this file's age, so the poll loop keeps writing it
    while it waits for its batch. If only the claim wrote it, a legitimate
    twenty-minute REPLACE_NODE would restart the Pod mid-mutation.
    """

    client = FakeExecutorClient([remote_command("command-a")])
    adapter = BreadcrumbWatchingAdapter(breadcrumbs_outside_shared_tmp)
    executor = build_executor(client, [adapter], liveness_interval_seconds=0.01)

    executor.run_once()

    assert adapter.saw_refresh is True, (
        "the poll loop did not refresh the liveness breadcrumb while the "
        "command was still executing"
    )
    with open(breadcrumbs_outside_shared_tmp, encoding="utf-8") as handle:
        breadcrumb = json.load(handle)
    assert breadcrumb["executor_id"] == EXECUTOR, breadcrumb


def test_the_liveness_breadcrumb_is_refreshed_even_when_the_claim_fails(
    breadcrumbs_outside_shared_tmp: str,
) -> None:
    """A control-plane outage is not a reason to restart every executor Pod.

    The readiness breadcrumb is only written after a successful claim, so it
    cannot carry liveness: it goes stale for a loop that is running perfectly
    and merely cannot reach the control plane.
    """

    class FailingClaimClient(FakeExecutorClient):
        def claim(self, *args: Any, **kwargs: Any) -> list:
            raise ClusterExecutorError("control plane unreachable")

    executor = build_executor(FailingClaimClient(), [RecordingAdapter()])

    with pytest.raises(ClusterExecutorError):
        executor.run_once()

    assert os.path.exists(breadcrumbs_outside_shared_tmp), (
        "the poll loop reached the control plane and came back, which is "
        "liveness, but wrote no breadcrumb"
    )


def test_the_executor_manifest_probes_the_poll_loop_liveness() -> None:
    """The Deployment restarts a wedged loop, and only a wedged loop.

    The probe reads the breadcrumb's age locally: it must not call the control
    plane, or a control-plane outage would restart both replicas of every
    regional cluster at once.
    """

    documents = [
        item
        for item in yaml.safe_load_all(EXECUTOR_MANIFEST.read_text(encoding="utf-8"))
        if item
    ]
    deployment = next(item for item in documents if item["kind"] == "Deployment")
    container = deployment["spec"]["template"]["spec"]["containers"][0]
    liveness = container["livenessProbe"]
    probe = " ".join(liveness["exec"]["command"])

    assert DEFAULT_LIVENESS_STATE_PATH in probe, (
        f"the probe must read the path the poll loop writes: {probe}"
    )
    assert str(LIVENESS_STALE_AFTER_SECONDS) in probe, (
        f"the probe must use the documented staleness bound: {probe}"
    )
    assert "readiness" not in probe, (
        f"liveness must not depend on the control plane: {probe}"
    )
    assert liveness["periodSeconds"] * liveness["failureThreshold"] >= 60, liveness
    assert "initialDelaySeconds" not in liveness, (
        f"the startup probe already gates liveness: {liveness}"
    )
    assert (
        container["startupProbe"]["exec"]["command"] != liveness["exec"]["command"]
    ), "the startup probe proves the process started, not that the loop turns"

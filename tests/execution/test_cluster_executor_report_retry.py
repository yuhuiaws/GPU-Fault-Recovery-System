"""Reporting a result across a broken wire, and a poison command in a claim.

``test_cluster_executor_lease_and_report.py`` covers the batch that reports
cleanly and the batch whose result the control plane *refuses*. What it never
covered is the batch whose result never reaches the control plane at all: on
2026-09-08 a review reproduced a ``URLError`` on ``complete`` escaping the
worker thread, so ``run_once`` raised, the whole batch's remaining reports were
lost, and the command stayed LEASED until its 120 s lease expired -- then it was
re-claimed and **re-executed**, which for a RESET_GPU or a RESTART_WORKLOAD
means running a destructive action twice for one workflow step.

The two paths asserted here are therefore:

* the result post survives a transport failure while the lease is still ours,
  is bounded (three attempts), stops the moment the lease is gone, and never
  escapes ``_execute_and_report`` whatever it raises;
* a single unparseable command in a claim response is failed on its own lease
  instead of dropping the whole batch, which the control plane had already
  committed leases for.

The helpers are imported from the sibling module rather than copied: the fake
control-plane client and the recording adapter are the same boundary, and two
divergent copies of them would be worse than one shared one.
"""

from __future__ import annotations

import json
import logging
import socket
import ssl
from http.client import IncompleteRead
from io import BytesIO
from typing import Any
from urllib.error import HTTPError, URLError

import pytest

from gpu_fault.cluster_executor import (
    ClusterActionExecutor,
    ClusterExecutorError,
    RegionalExecutorClient,
    RegionalFleetRegistry,
)
from gpu_fault.regional import (
    RemoteActionCommand,
    RemoteCommandResult,
    RemoteCommandStatus,
)
from tests.execution.test_cluster_executor_lease_and_report import (
    CLUSTER,
    CONTINUATION_STATE,
    EXECUTOR,
    FakeExecutorClient,
    RecordingAdapter,
    remote_command,
)

CONTROL_PLANE = "https://control-plane.example"
TOKEN = "token-value-" + "t" * 32


class FakeClock:
    """A monotonic clock the test moves by hand, in seconds."""

    def __init__(self) -> None:
        self.seconds = 0.0

    def __call__(self) -> float:
        return self.seconds


class RecordingSleep:
    """The executor's injected sleep: records the delay, never really waits.

    ``advance`` lets a test make time pass *because* the executor backed off,
    which is how the lease expires between two report attempts.
    """

    def __init__(self, *, clock: FakeClock | None = None, advance: float = 0.0) -> None:
        self.delays: list[float] = []
        self.clock = clock
        self.advance = advance

    def __call__(self, seconds: float) -> None:
        self.delays.append(seconds)
        if self.clock is not None:
            self.clock.seconds += self.advance


class FlakyReportClient(FakeExecutorClient):
    """Control-plane stand-in whose ``complete`` fails a scripted few times.

    ``FakeExecutorClient.complete_errors`` raises the same error forever, which
    cannot express "the connection dropped once". ``attempts`` records every
    post, ``completed`` only the ones that landed.
    """

    def __init__(
        self,
        *batches: list[RemoteActionCommand],
        complete_failures: dict[str, list[BaseException]] | None = None,
    ) -> None:
        super().__init__(*batches)
        self.failures = {
            command_id: list(errors)
            for command_id, errors in (complete_failures or {}).items()
        }
        self.attempts: list[str] = []

    def complete(
        self, command: RemoteActionCommand, result: RemoteCommandResult
    ) -> RemoteActionCommand:
        self.attempts.append(command.command_id)
        queued = self.failures.get(command.command_id)
        if queued:
            raise queued.pop(0)
        self.completed.append((command.command_id, result))
        return command


def build(
    client: Any, adapters: list[Any], tmp_path: Any, **overrides: Any
) -> ClusterActionExecutor:
    """A real executor whose readiness breadcrumb stays inside this test's dir."""

    overrides.setdefault("claim_state_path", str(tmp_path / "claim-state.json"))
    return ClusterActionExecutor(
        client,
        adapters,
        executor_id=EXECUTOR,
        allowed_namespaces={"training"},
        **overrides,
    )


class _CannedResponse:
    """One urlopen response body, for the client-level claim tests."""

    def __init__(self, body: bytes) -> None:
        self.body = body

    def __enter__(self) -> _CannedResponse:
        return self

    def __exit__(self, *_args: Any) -> None:
        return None

    def read(self) -> bytes:
        return self.body


def transport_client(monkeypatch, bodies: list[bytes]) -> tuple[Any, list[Any]]:
    """A real ``RegionalExecutorClient`` over a scripted urlopen.

    Returns the client and the list of ``(url, payload)`` requests it sent, so
    a test can assert what the executor put on the wire without a live control
    plane.
    """

    sent: list[Any] = []
    remaining = list(bodies)

    def fake_urlopen(request: Any, **_kwargs: Any) -> _CannedResponse:
        sent.append(
            (
                request.full_url,
                json.loads(request.data or b"null") if request.data else None,
            )
        )
        return _CannedResponse(remaining.pop(0) if remaining else b"{}")

    monkeypatch.setattr("gpu_fault.cluster_executor.urlopen", fake_urlopen)
    return RegionalExecutorClient(CONTROL_PLANE, CLUSTER, TOKEN), sent


def raising_client(monkeypatch, error: BaseException) -> Any:
    """A real client whose every request raises ``error``."""

    def fake_urlopen(*_args: Any, **_kwargs: Any) -> None:
        raise error

    monkeypatch.setattr("gpu_fault.cluster_executor.urlopen", fake_urlopen)
    return RegionalExecutorClient(CONTROL_PLANE, CLUSTER, TOKEN)


def test_a_transport_error_on_complete_is_retried_under_a_live_lease(tmp_path) -> None:
    """The action already ran; the report has to survive one dropped connection.

    The baseline control plane closes ~1 connection a minute ("Remote end closed
    connection"), so a single ``URLError`` on the result post is expected
    traffic, not an incident. Dropping the result there costs a full lease of
    latency and then re-runs the action, so the post is retried while the lease
    is still ours -- and the action itself is not run again.
    """

    client = FlakyReportClient(
        [remote_command("command-a"), remote_command("command-b")],
        complete_failures={"command-a": [URLError(socket.timeout("timed out"))]},
    )
    adapter = RecordingAdapter()
    sleeper = RecordingSleep()
    executor = build(
        client, [adapter], tmp_path, max_concurrent_commands=2, sleep=sleeper
    )

    assert executor.run_once() == 2, "both claimed commands must be handled"
    assert adapter.executed_command_ids() == ["command-a", "command-b"], (
        "a failed report must not re-run the action"
    )
    assert client.attempts.count("command-a") == 2, (
        f"the dropped report must be retried once: {client.attempts}"
    )
    assert sorted(command_id for command_id, _ in client.completed) == [
        "command-a",
        "command-b",
    ], "every result must reach the control plane, the sibling's included"
    assert executor.reported_failures == 0, (
        "a report that landed on retry is not a reporting failure"
    )
    assert len(sleeper.delays) == 1, f"one backoff, not a spin: {sleeper.delays}"
    assert 0.5 <= sleeper.delays[0] < 1.0, (
        f"the first backoff is 0.5s plus jitter: {sleeper.delays}"
    )


def test_a_report_failure_never_escapes_run_once(tmp_path) -> None:
    """A report that can never land must be counted, not raised.

    An exception escaping the worker thread comes back out of
    ``future.result()`` in ``run_once``, which loses the other commands' reports
    and sends ``run()`` into claim backoff. The bound is three attempts: past
    that the command is left for the next lease holder, and this cycle counts
    as not advanced so the loop takes the idle poll instead of spinning.
    """

    client = FakeExecutorClient(
        [remote_command("command-a")],
        complete_errors={
            "command-a": ssl.SSLError("EOF occurred in violation of protocol")
        },
    )
    adapter = RecordingAdapter()
    sleeper = RecordingSleep()
    executor = build(client, [adapter], tmp_path, sleep=sleeper)

    assert executor.run_once() == 1, "run_once must return, not raise"
    assert executor.reported_failures == 1, (
        "the unreportable result must be counted exactly once"
    )
    assert [command_id for command_id, _ in client.completed] == ["command-a"] * 3, (
        f"the retry must stop at three attempts: {client.completed}"
    )
    assert executor.last_cycle_advanced is False, (
        "a result that never landed is not progress"
    )
    assert executor.unexpected_failures == 0, "a broken wire is not an executor defect"

    assert executor.run_once() == 0, "the next claim returns nothing new"
    assert adapter.executed_command_ids() == ["command-a"], (
        "the command must not be executed a second time"
    )


def test_a_lease_lost_during_the_report_backoff_withholds_the_retry(tmp_path) -> None:
    """The lease can lapse *while* we back off, so re-check before re-posting.

    Checking the hold only where the failure is caught is not enough: the sleep
    is where a 120 s lease actually runs out, and a POST sent after that races
    the result of whichever replica re-claimed the command -- exactly what the
    withheld-result rule in ``_execute_under_lease`` exists to prevent. A
    withheld retry is not a reporting failure either: nothing was refused and
    nothing was lost that the next lease holder will not redo.
    """

    clock = FakeClock()
    client = FakeExecutorClient(
        [remote_command("command-a")],
        complete_errors={"command-a": URLError("connection reset by peer")},
    )
    sleeper = RecordingSleep(clock=clock, advance=1000.0)
    executor = build(client, [RecordingAdapter()], tmp_path, clock=clock, sleep=sleeper)

    assert executor.run_once() == 1, "run_once must return, not raise"
    assert [command_id for command_id, _ in client.completed] == ["command-a"], (
        f"no result may be posted once the lease is gone: {client.completed}"
    )
    assert executor.results_withheld_total == 1, (
        "abandoning the retry under a lost lease is a withheld result"
    )
    assert executor.reported_failures == 0, (
        "a withheld retry is not the control plane refusing the result"
    )


def test_a_truncated_response_on_complete_is_retried(tmp_path) -> None:
    """A half-read response carries no verdict either, so it must be retried.

    ``http.client`` raises its own family (``IncompleteRead``,
    ``BadStatusLine``) when the connection dies mid-response; it is not a
    ``URLError`` and used to be neither wrapped nor retried, so it escaped the
    worker exactly like F1's timeout did.
    """

    client = FlakyReportClient(
        [remote_command("command-a")],
        complete_failures={"command-a": [IncompleteRead(b"partial", 12)]},
    )
    sleeper = RecordingSleep()
    executor = build(client, [RecordingAdapter()], tmp_path, sleep=sleeper)

    assert executor.run_once() == 1, "run_once must return, not raise"
    assert client.attempts.count("command-a") == 2, (
        f"a truncated response must be retried: {client.attempts}"
    )
    assert executor.reported_failures == 0, (
        "the retry landed, so nothing failed to report"
    )


def test_a_command_without_a_lease_token_never_sinks_its_batch(tmp_path) -> None:
    """The refusal is right; taking the batch down with it is not.

    ``_execute`` raises before its own try block when a claimed command carries
    no lease token, so the exception used to leave the worker thread and fail
    ``run_once`` -- taking the siblings' reports with it.
    """

    unleased = remote_command("command-a").model_copy(update={"lease_token": None})
    client = FakeExecutorClient([unleased, remote_command("command-b")])
    adapter = RecordingAdapter()
    executor = build(client, [adapter], tmp_path, max_concurrent_commands=2)

    assert executor.run_once() == 2, "run_once must return, not raise"
    assert [command_id for command_id, _ in client.completed] == ["command-b"], (
        "the sibling's result must still be reported"
    )
    assert executor.unexpected_failures == 1, (
        "a command the executor cannot even answer is an executor-side defect"
    )


def test_a_malformed_command_in_a_claim_is_failed_without_dropping_its_siblings(
    monkeypatch,
) -> None:
    """The control plane already committed the leases before we parsed anything.

    Validating the batch as one document meant a single command this build
    cannot parse (a newer enum value, a control-plane-only field) dropped every
    command leased with it. They then expired together and were re-claimed
    together with the poison command, forever. Failing the one command on its
    own lease is what lets its siblings run and lets the workflow that owns the
    poison command see a verdict instead of hanging.
    """

    good = remote_command("command-good").model_dump(mode="json")
    broken = remote_command("command-broken").model_dump(mode="json")
    broken["step"]["operation"] = "TELEPORT_THE_GPU"
    client, sent = transport_client(
        monkeypatch, [json.dumps({"commands": [good, broken]}).encode(), b"{}"]
    )

    commands = client.claim(EXECUTOR, max_commands=5, lease_seconds=120)

    assert [command.command_id for command in commands] == ["command-good"], (
        "the parsable command must still be executed"
    )
    assert len(sent) == 2, f"the poison command needs its own result post: {sent}"
    url, payload = sent[1]
    assert url == (CONTROL_PLANE + "/v1/regional/executors/command-broken/result"), (
        f"the rejection must be posted for the malformed command: {url}"
    )
    assert payload["status"] == RemoteCommandStatus.FAILED.value, (
        f"an unparseable command cannot be held WAITING: {payload}"
    )
    assert payload["status_source"] == "executor-rejected", (
        f"this is the executor refusing, not the action failing: {payload}"
    )
    assert payload["lease_token"] == "lease-command-broken", (
        "the rejection must use the lease the control plane just handed out"
    )
    assert "step.operation" in payload["error"], (
        f"the operator needs the field that could not be parsed: {payload['error']}"
    )


def test_a_claim_response_of_the_wrong_shape_still_fails_the_cycle(monkeypatch) -> None:
    """Per-command tolerance must not become tolerance of a broken response.

    A response with no ``commands`` list at all is not a poison command, it is
    a control plane the executor cannot talk to; silently claiming nothing
    would look exactly like an empty queue.
    """

    client, _sent = transport_client(monkeypatch, [b'{"commands":{"a":1}}'])

    with pytest.raises(Exception) as raised:
        client.claim(EXECUTOR, max_commands=5, lease_seconds=120)

    assert "commands" in str(raised.value), (
        f"the failure must name the field the control plane got wrong: {raised.value}"
    )


@pytest.mark.parametrize(
    ("error", "label"),
    [
        (URLError(socket.timeout("timed out")), "urlerror"),
        (TimeoutError("timed out"), "timeout"),
        (ssl.SSLError("record layer failure"), "ssl"),
        (ConnectionResetError("reset by peer"), "connection-reset"),
        (IncompleteRead(b"partial", 12), "incomplete-read"),
    ],
    ids=["urlerror", "timeout", "ssl", "connection-reset", "incomplete-read"],
)
def test_a_transport_failure_to_the_control_plane_is_one_exception_type(
    monkeypatch, error: BaseException, label: str
) -> None:
    """Callers classify by ``status_code``, so they must get the same type.

    Only ``HTTPError`` used to be converted, so every caller of the regional
    proxies had a second, unhandled exception family to know about -- which is
    how a timeout on a proxy read escaped the command worker entirely.
    """

    registry = RegionalFleetRegistry(raising_client(monkeypatch, error))

    with pytest.raises(ClusterExecutorError) as raised:
        registry.list_agents(CLUSTER)

    assert raised.value.status_code is None, (
        f"a transport failure has no HTTP status ({label})"
    )
    assert type(error).__name__ in str(raised.value), (
        f"the wrapped error must name its cause ({label}): {raised.value}"
    )


def test_a_control_plane_transport_failure_keeps_the_command_retryable(
    tmp_path, monkeypatch
) -> None:
    """Wrapping the transport error must not turn a hold into a failure.

    The adapters reach the control plane through the regional proxies, so a
    timeout there says nothing about the GPU. It has to stay WAITING: reporting
    FAILED would mark a healthy node unrecoverable because a read timed out.
    """

    proxy = RegionalFleetRegistry(
        raising_client(monkeypatch, URLError(socket.timeout("timed out")))
    )

    class ProxyReadingAdapter(RecordingAdapter):
        def execute(self, context: Any) -> Any:
            self.contexts.append(context)
            proxy.list_agents(CLUSTER)
            raise AssertionError("the proxy read must have failed")

    client = FakeExecutorClient([remote_command("command-a")])
    executor = build(client, [ProxyReadingAdapter()], tmp_path)

    assert executor.run_once() == 1, "run_once must return, not raise"
    result = client.reported("command-a")
    assert result.status is RemoteCommandStatus.WAITING, (
        f"a control-plane timeout is not the action failing: {result}"
    )
    assert result.details.get("retryable_transport_error") is True, (
        "the operator must be able to tell a transport hold from a refusal: "
        f"{result.details}"
    )
    assert executor.unexpected_failures == 0, "a timeout is not an executor defect"


def test_a_transport_hold_keeps_the_previous_cycles_details(
    tmp_path, monkeypatch
) -> None:
    """The transport hold is manufactured by the executor, so it owes the merge.

    ``complete_remote_command`` *replaces* ``result_details``, so this WAITING
    result becomes the whole record of the step: whatever the adapter parked
    there on an earlier cycle (the agent generations it validated against, a
    pending spare failover, the quiesce attempt counter) is gone unless the
    hold carries it forward. The hold's own keys still win -- ``reason`` must
    name the transport failure, not the stale wait.
    """

    proxy = RegionalFleetRegistry(
        raising_client(monkeypatch, URLError(socket.timeout("timed out")))
    )

    class ProxyReadingAdapter(RecordingAdapter):
        def execute(self, context: Any) -> Any:
            self.contexts.append(context)
            proxy.list_agents(CLUSTER)
            raise AssertionError("the proxy read must have failed")

    client = FakeExecutorClient(
        [remote_command("command-a", result_details=dict(CONTINUATION_STATE))]
    )
    executor = build(client, [ProxyReadingAdapter()], tmp_path)

    assert executor.run_once() == 1, "run_once must return, not raise"
    result = client.reported("command-a")
    assert result.status is RemoteCommandStatus.WAITING, result
    assert result.status_source == "executor-retryable-transport", result
    assert result.details["agent_baselines"] == CONTINUATION_STATE["agent_baselines"], (
        f"the transport hold dropped the adapter's continuation state: {result.details}"
    )
    assert result.details["spare_failover_pending"] is True, result.details
    assert result.details["gpu_client_quiesce_attempt"] == 12, result.details
    assert result.details["retryable_transport_error"] is True, result.details
    assert result.details["reason"] != CONTINUATION_STATE["reason"], (
        f"the hold's own reason must win over the replayed one: {result.details}"
    )


def test_a_missing_agent_is_still_reported_as_a_missing_key(monkeypatch) -> None:
    """404 on an agent read is a KeyError, which is what callers catch."""

    registry = RegionalFleetRegistry(
        raising_client(
            monkeypatch,
            HTTPError(
                CONTROL_PLANE + "/v1/fleet/agents/cluster-a/node-a",
                404,
                "Not Found",
                {},  # type: ignore[arg-type]
                BytesIO(b'{"detail":"no such agent"}'),
            ),
        )
    )

    with pytest.raises(KeyError) as raised:
        registry.get_agent(CLUSTER, "node-a")

    assert raised.value.args[0] == (CLUSTER, "node-a"), (
        f"the missing key names the cluster and the node: {raised.value}"
    )


def test_a_control_plane_error_that_merely_mentions_404_is_not_a_missing_agent(
    monkeypatch,
) -> None:
    """A 503 whose body quotes an upstream 404 is still a 503.

    Reading the status out of the message text made any error body containing
    "(404)" look like an absent agent, and an absent agent is what the fleet
    code treats as "this node has no agent record" -- a permanent verdict built
    out of a transient one.
    """

    registry = RegionalFleetRegistry(
        raising_client(
            monkeypatch,
            HTTPError(
                CONTROL_PLANE + "/v1/fleet/agents/cluster-a/node-a",
                503,
                "Service Unavailable",
                {},  # type: ignore[arg-type]
                BytesIO(b'{"detail":"upstream returned (404) for /probe"}'),
            ),
        )
    )

    with pytest.raises(ClusterExecutorError) as raised:
        registry.get_agent(CLUSTER, "node-a")

    assert raised.value.status_code == 503, (
        f"the status must come from the response, not the text: {raised.value}"
    )


def test_a_claim_failure_and_a_post_claim_failure_are_logged_apart(
    monkeypatch, caplog, tmp_path
) -> None:
    """ "claim failed" used to be printed for failures nowhere near the claim.

    ``run()`` logged every ``run_once`` exception as a claim failure, so a
    result that could not be posted and a claim response the executor could not
    use both read as "the control plane is unreachable" -- and the operator
    looked at the wrong side of the wire.
    """

    class BadShapeClient(FakeExecutorClient):
        def claim(self, executor_id: str, **kwargs: Any) -> Any:
            super().claim(executor_id, **kwargs)
            return None

    def stop_after_one_backoff(_seconds: float) -> None:
        raise StopIteration

    monkeypatch.setattr("gpu_fault.cluster_executor.time.sleep", stop_after_one_backoff)
    executor = build(BadShapeClient(), [RecordingAdapter()], tmp_path)

    with caplog.at_level(logging.ERROR):
        with pytest.raises(StopIteration):
            executor.run()

    messages = [record.getMessage() for record in caplog.records]
    assert messages, "the failure must be logged at all"
    assert not any("claim failed" in message for message in messages), (
        f"nothing here failed the claim round trip: {messages}"
    )

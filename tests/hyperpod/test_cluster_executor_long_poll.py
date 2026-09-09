"""The executor side of the long-poll claim.

``GPU_FAULT_CLUSTER_EXECUTOR_CLAIM_WAIT_SECONDS`` (default 20) rides in the
claim body as ``wait_seconds``; the control plane holds an empty claim that
long, so ``run()`` loops straight back into the next claim instead of sleeping
``poll_seconds``. 0 disables the field entirely -- it is left out of the body,
so the executor still speaks to a control plane that predates it -- and
restores pure polling. The claim request's HTTP timeout grows by the wait so
a held request is not a timeout.
"""

from __future__ import annotations

import io
import json
from contextlib import contextmanager

import pytest

from gpu_fault.cluster_executor import (
    ClusterActionExecutor,
    ClusterExecutorError,
    RegionalExecutorClient,
    executor_from_environment,
)


class EmptyClient:
    cluster_id = "cluster-a"

    def __init__(self) -> None:
        self.claims: list[dict] = []

    def claim(self, executor_id, **kwargs):
        self.claims.append({"executor_id": executor_id, **kwargs})
        return []


class Adapter:
    owner = "gpu-fault-kubernetes-adapter"


class Stop(BaseException):
    """Escapes ``run()``, which swallows every ``Exception`` as a claim failure."""


def _stop_after(monkeypatch, sleeps: list[float], count: int):
    def sleep(seconds):
        sleeps.append(seconds)
        if len(sleeps) == count:
            raise StopIteration

    monkeypatch.setattr("gpu_fault.cluster_executor.time.sleep", sleep)


def _executor(client, **overrides) -> ClusterActionExecutor:
    return ClusterActionExecutor(
        client,
        [Adapter()],
        executor_id="executor-a",
        allowed_namespaces={"training"},
        poll_seconds=2,
        **overrides,
    )


def test_a_long_poll_executor_does_not_sleep_after_an_empty_claim(monkeypatch):
    client = EmptyClient()
    executor = _executor(client, claim_wait_seconds=20)
    stopped = {"claims": 0}
    original = executor.run_once

    def run_once():
        stopped["claims"] += 1
        if stopped["claims"] == 5:
            raise Stop
        return original()

    monkeypatch.setattr(executor, "run_once", run_once)
    sleeps: list[float] = []
    monkeypatch.setattr(
        "gpu_fault.cluster_executor.time.sleep", lambda s: sleeps.append(s)
    )

    with pytest.raises(Stop):
        executor.run()

    assert sleeps == [], "the server already waited; the loop must not"
    assert len(client.claims) == 4
    assert all(claim["wait_seconds"] == 20 for claim in client.claims), (
        "a long-polling executor must pass its configured wait on every claim"
    )
    assert executor.last_successful_claim_at is not None, (
        "an empty long-poll is still a successful claim"
    )


def test_a_polling_executor_still_sleeps_poll_seconds(monkeypatch) -> None:
    client = EmptyClient()
    executor = _executor(client, claim_wait_seconds=0)
    sleeps: list[float] = []
    _stop_after(monkeypatch, sleeps, 3)

    with pytest.raises(StopIteration):
        executor.run()

    assert sleeps == [2, 2, 2]
    assert all(claim["wait_seconds"] == 0 for claim in client.claims), (
        "a polling executor must never ask the server to hold the claim"
    )


def test_a_held_command_still_takes_the_idle_path_under_long_poll(monkeypatch):
    """A WAITING command is re-claimable at once, so without the idle sleep a
    long-poll would spin on it exactly as pure polling did on 2026-09-04."""

    executor = _executor(EmptyClient(), claim_wait_seconds=20)

    def held_run_once():
        executor.last_cycle_advanced = False
        return 1

    monkeypatch.setattr(executor, "run_once", held_run_once)
    sleeps: list[float] = []
    _stop_after(monkeypatch, sleeps, 2)

    with pytest.raises(StopIteration):
        executor.run()

    assert sleeps == [2, 2]


def test_claim_failures_back_off_from_poll_seconds_under_long_poll(monkeypatch):
    class Failing(EmptyClient):
        def claim(self, *_args, **_kwargs):
            raise ClusterExecutorError("control plane unreachable")

    executor = _executor(Failing(), claim_wait_seconds=20, claim_backoff_max_seconds=8)
    sleeps: list[float] = []
    _stop_after(monkeypatch, sleeps, 4)

    with pytest.raises(StopIteration):
        executor.run()

    assert sleeps == [2, 4, 8, 8], "the transport backoff is untouched"


def test_claim_wait_seconds_is_bounded_like_the_protocol_field() -> None:
    with pytest.raises(ClusterExecutorError):
        _executor(EmptyClient(), claim_wait_seconds=-1)
    with pytest.raises(ClusterExecutorError):
        _executor(EmptyClient(), claim_wait_seconds=31)
    assert _executor(EmptyClient(), claim_wait_seconds=30).claim_wait_seconds == 30


class _CapturingTransport:
    def __init__(self) -> None:
        self.requests: list[tuple[dict, float]] = []

    @contextmanager
    def __call__(self, request, *, timeout, ssl_context=None):
        self.requests.append((json.loads(request.data), timeout))
        yield io.BytesIO(b'{"commands": []}')


def _client() -> RegionalExecutorClient:
    return RegionalExecutorClient(
        "https://control-plane.example", "cluster-a", "token", timeout_seconds=15
    )


def test_the_claim_body_carries_wait_seconds_and_the_timeout_grows(monkeypatch):
    transport = _CapturingTransport()
    monkeypatch.setattr("gpu_fault.cluster_executor.urlopen", transport)

    _client().claim(
        "executor-a",
        execution_owners=["gpu-fault-kubernetes-adapter"],
        max_commands=5,
        lease_seconds=120,
        wait_seconds=20,
    )

    ((body, timeout),) = transport.requests
    assert body["wait_seconds"] == 20
    assert timeout == 35, "request timeout = wait + the ordinary timeout"


def test_a_disabled_wait_is_left_out_of_the_body_and_keeps_the_timeout(
    monkeypatch,
) -> None:
    transport = _CapturingTransport()
    monkeypatch.setattr("gpu_fault.cluster_executor.urlopen", transport)

    _client().claim(
        "executor-a",
        execution_owners=["gpu-fault-kubernetes-adapter"],
        max_commands=5,
        lease_seconds=120,
        wait_seconds=0,
    )
    _client().claim(
        "executor-a",
        execution_owners=["gpu-fault-kubernetes-adapter"],
        max_commands=5,
        lease_seconds=120,
    )

    for body, timeout in transport.requests:
        assert "wait_seconds" not in body, "an old control plane must accept it"
        assert timeout == 15


def _executor_environment(monkeypatch) -> None:
    monkeypatch.setenv("GPU_FAULT_CONTROL_PLANE_URL", "https://control-plane.example")
    monkeypatch.setenv("GPU_FAULT_CONTROL_PLANE_TOKEN", "token")
    monkeypatch.setenv("GPU_FAULT_CLUSTER_ID", "cluster-a")
    monkeypatch.delenv("GPU_FAULT_STORE_URL", raising=False)
    monkeypatch.delenv("GPU_FAULT_ENABLE_HYPERPOD_ADAPTER", raising=False)
    monkeypatch.delenv("GPU_FAULT_ENABLE_NODE_ACTION_ADAPTER", raising=False)

    class FakeKubernetesAdapter:
        owner = "gpu-fault-kubernetes-adapter"

        def __init__(self, **kwargs) -> None:
            self.owner = kwargs.get("owner", self.owner)

    monkeypatch.setattr(
        "gpu_fault.cluster_executor.KubernetesWorkflowAdapter", FakeKubernetesAdapter
    )


def test_the_environment_defaults_the_claim_wait_to_twenty_seconds(monkeypatch):
    _executor_environment(monkeypatch)
    monkeypatch.delenv("GPU_FAULT_CLUSTER_EXECUTOR_CLAIM_WAIT_SECONDS", raising=False)

    assert executor_from_environment().claim_wait_seconds == 20.0


def test_the_environment_can_disable_the_long_poll(monkeypatch) -> None:
    _executor_environment(monkeypatch)
    monkeypatch.setenv("GPU_FAULT_CLUSTER_EXECUTOR_CLAIM_WAIT_SECONDS", "0")

    assert executor_from_environment().claim_wait_seconds == 0.0

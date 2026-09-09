"""``POST /v1/regional/executors/claim`` with ``wait_seconds``: the long-poll.

An empty first claim with ``wait_seconds > 0`` is held on the per-cluster
wakeup hub until a PENDING command lands for that cluster (or the bound runs
out), then claimed again. ``wait_seconds`` omitted or 0 is today's single
claim, which is also what an executor that predates the field sends. The hub
itself is covered in ``tests/app/test_remote_command_wakeups.py``; these tests
pin what the route does with its answers.
"""

from __future__ import annotations

import asyncio
import threading
import time
from dataclasses import replace

from gpu_fault.app.remote_command_wakeups import (
    RemoteCommandWakeupHub,
    WakeupWaitOutcome,
)
from gpu_fault.app.routes.regional import get_regional_dependencies
from gpu_fault.regional import RemoteCommandClaimRequest
from tests._builders import asgi_client, build_context, create_app
from tests.regional._regional_support import (
    TOKEN_A,
    TOKEN_B,
    enqueue_remote_command,
    registration,
)

CLAIM = "/v1/regional/executors/claim"
HEADERS_A = {
    "Authorization": f"Bearer {TOKEN_A}",
    "X-GPU-Fault-Cluster-ID": "cluster-a",
}
HEADERS_B = {
    "Authorization": f"Bearer {TOKEN_B}",
    "X-GPU-Fault-Cluster-ID": "cluster-b",
}


def _context():
    context = build_context()
    context.regional_mode = True
    context.store.save_regional_cluster(registration("cluster-a", TOKEN_A))
    context.store.save_regional_cluster(registration("cluster-b", TOKEN_B))
    return context


class FakeWaiter:
    def __init__(self, hub, cluster_id: str) -> None:
        self.hub = hub
        self.cluster_id = cluster_id
        self.admitted = hub.admit

    async def wait(self, wait_seconds: float) -> WakeupWaitOutcome:
        self.hub.waits.append((self.cluster_id, wait_seconds))
        if not self.admitted:
            return WakeupWaitOutcome.NOT_ADMITTED
        await asyncio.sleep(self.hub.hold_seconds)
        return self.hub.outcome


class FakeHub:
    """Records the subscriptions and answers every wait the same way."""

    def __init__(
        self,
        outcome: WakeupWaitOutcome = WakeupWaitOutcome.TIMEOUT,
        *,
        hold_seconds: float = 0.0,
        admit: bool = True,
    ) -> None:
        self.outcome = outcome
        self.hold_seconds = hold_seconds
        self.admit = admit
        self.subscriptions: list[str] = []
        self.waits: list[tuple[str, float]] = []

    def subscribe(self, cluster_id: str):
        hub = self

        class _Subscription:
            async def __aenter__(self):
                hub.subscriptions.append(cluster_id)
                return FakeWaiter(hub, cluster_id)

            async def __aexit__(self, *_exc):
                return False

        return _Subscription()


def _app_with_hub(context, hub):
    app = create_app(context)
    real = app.dependency_overrides[get_regional_dependencies]()
    app.dependency_overrides[get_regional_dependencies] = lambda: replace(
        real, remote_command_wakeups=hub
    )
    return app


def _counting_claims(context):
    calls: list[str] = []
    original = context.store.claim_remote_commands

    def counted(cluster_id, *args, **kwargs):
        calls.append(cluster_id)
        return original(cluster_id, *args, **kwargs)

    context.store.claim_remote_commands = counted
    return calls


def test_wait_seconds_is_an_optional_bounded_additive_field() -> None:
    """Old executors omit it and get today's behaviour; the bound is 0..30."""

    assert RemoteCommandClaimRequest(executor_id="e").wait_seconds == 0
    assert RemoteCommandClaimRequest(executor_id="e", wait_seconds=30).wait_seconds, (
        "the upper bound of 30 s must be accepted as-is"
    )
    for illegal in (-1, 30.5, "soon"):
        try:
            RemoteCommandClaimRequest(executor_id="e", wait_seconds=illegal)
        except ValueError:
            continue
        raise AssertionError(f"wait_seconds={illegal!r} was accepted")


def test_no_wait_seconds_is_a_single_claim_without_a_subscription() -> None:
    context = _context()
    hub = FakeHub()
    calls = _counting_claims(context)

    async def scenario():
        async with asgi_client(_app_with_hub(context, hub)) as client:
            plain = await client.post(
                CLAIM, headers=HEADERS_A, json={"executor_id": "executor-a"}
            )
            zero = await client.post(
                CLAIM,
                headers=HEADERS_A,
                json={"executor_id": "executor-a", "wait_seconds": 0},
            )
            return plain, zero

    plain, zero = asyncio.run(scenario())
    assert (plain.status_code, plain.json()) == (200, {"commands": []})
    assert (zero.status_code, zero.json()) == (200, {"commands": []})
    assert hub.subscriptions == []
    assert calls == ["cluster-a", "cluster-a"]


def test_an_empty_claim_waits_on_its_cluster_and_claims_again_when_woken():
    context = _context()
    hub = FakeHub(WakeupWaitOutcome.WOKEN)
    calls = _counting_claims(context)

    async def scenario():
        async with asgi_client(_app_with_hub(context, hub)) as client:
            return await client.post(
                CLAIM,
                headers=HEADERS_A,
                json={"executor_id": "executor-a", "wait_seconds": 5},
            )

    response = asyncio.run(scenario())
    assert response.status_code == 200
    assert hub.subscriptions == ["cluster-a"]
    assert hub.waits == [("cluster-a", 5.0)]
    assert calls == ["cluster-a", "cluster-a"], "woken claims exactly once more"


def test_a_timed_out_or_disconnected_wait_still_claims_again() -> None:
    """A lost NOTIFY must cost one wait, not a command."""

    for outcome in (WakeupWaitOutcome.TIMEOUT, WakeupWaitOutcome.DISCONNECTED):
        context = _context()
        hub = FakeHub(outcome)
        calls = _counting_claims(context)

        async def scenario():
            async with asgi_client(_app_with_hub(context, hub)) as client:
                return await client.post(
                    CLAIM,
                    headers=HEADERS_A,
                    json={"executor_id": "executor-a", "wait_seconds": 5},
                )

        response = asyncio.run(scenario())
        assert response.status_code == 200, outcome
        assert calls == ["cluster-a", "cluster-a"], outcome


def test_a_waiter_the_hub_did_not_admit_answers_with_the_first_claim() -> None:
    context = _context()
    hub = FakeHub(admit=False)
    calls = _counting_claims(context)

    async def scenario():
        async with asgi_client(_app_with_hub(context, hub)) as client:
            return await client.post(
                CLAIM,
                headers=HEADERS_A,
                json={"executor_id": "executor-a", "wait_seconds": 5},
            )

    response = asyncio.run(scenario())
    assert (response.status_code, response.json()) == (200, {"commands": []})
    assert calls == ["cluster-a"], "no second claim for a request that never waited"


def test_a_non_empty_first_claim_returns_without_waiting() -> None:
    context = _context()
    enqueue_remote_command(context.store, "remote-" + "1" * 24)
    hub = FakeHub(WakeupWaitOutcome.WOKEN)

    async def scenario():
        async with asgi_client(_app_with_hub(context, hub)) as client:
            return await client.post(
                CLAIM,
                headers=HEADERS_A,
                json={"executor_id": "executor-a", "wait_seconds": 5},
            )

    response = asyncio.run(scenario())
    assert [c["command_id"] for c in response.json()["commands"]] == [
        "remote-" + "1" * 24
    ]
    assert hub.waits == []


def test_the_protocol_gate_still_runs_before_any_wait() -> None:
    context = _context()
    hub = FakeHub(WakeupWaitOutcome.WOKEN)

    async def scenario():
        async with asgi_client(_app_with_hub(context, hub)) as client:
            return await client.post(
                CLAIM,
                headers=HEADERS_A,
                json={
                    "executor_id": "executor-a",
                    "executor_protocol_version": 99,
                    "wait_seconds": 5,
                },
            )

    response = asyncio.run(scenario())
    assert response.status_code == 503
    assert hub.subscriptions == []


def test_a_real_hub_wakes_the_held_claim_on_a_command_for_its_cluster_only():
    """End to end on the memory store: the app's own hub, a real listener
    thread, a command for another cluster that must not wake it."""

    context = _context()
    app = create_app(context)
    hub = app.state.remote_command_wakeups
    assert isinstance(hub, RemoteCommandWakeupHub), (
        "create_app must install a real wakeup hub on app.state"
    )
    try:

        async def scenario():
            async with asgi_client(app) as client:
                # A wait on cluster-b first, so the listener is up and the
                # cluster-a wait below can be observed to ignore it.
                warm = await client.post(
                    CLAIM,
                    headers=HEADERS_B,
                    json={"executor_id": "executor-b", "wait_seconds": 0.2},
                )
                assert warm.json() == {"commands": []}
                assert hub.connected is True

                def later():
                    enqueue_remote_command(
                        context.store, "remote-" + "b" * 24, cluster_id="cluster-b"
                    )
                    time.sleep(0.2)
                    enqueue_remote_command(
                        context.store, "remote-" + "a" * 24, cluster_id="cluster-a"
                    )

                threading.Timer(0.05, later).start()
                started = time.monotonic()
                response = await client.post(
                    CLAIM,
                    headers=HEADERS_A,
                    json={"executor_id": "executor-a", "wait_seconds": 10},
                )
                return response, time.monotonic() - started

        response, elapsed = asyncio.run(scenario())
        assert response.status_code == 200
        assert [c["command_id"] for c in response.json()["commands"]] == [
            "remote-" + "a" * 24
        ]
        assert 0.2 <= elapsed < 3.0, f"held {elapsed:.2f}s"
    finally:
        hub.close()
    assert hub.listener_alive is False

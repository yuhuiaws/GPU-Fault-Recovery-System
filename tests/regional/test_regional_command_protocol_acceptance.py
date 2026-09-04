"""Local proxies for the manual remote-command protocol acceptance cases.

``GF-REGIONAL-CMD-*`` walk the claim/result contract value by value:
``max_commands`` at 0/1/25/26, ``lease_seconds`` at 9/10/7200/7201, the exact
status code for a foreign ``command_id``, whether a terminal result can be
overwritten. Every one of those answers is produced by a model constraint or a
store transition that exists in this process, which makes the manual run an
expensive way to re-read the model definitions -- and an easy one to skip, since
nothing about it looks dangerous.

What the live case still adds is the executor binary, the network and the real
clock. What CI can own is the contract itself, so a widened bound or a lease
check that stops being enforced fails here first.

Time is constructed rather than waited for: an expired lease is a command written
with a past ``lease_expires_at``, not a fifteen-second sleep, because a suite that
sleeps is a suite that gets skipped.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from gpu_fault.regional import (
    RegionalRemoteWorkflowAdapter,
    RemoteCommandResult,
    RemoteCommandStatus,
)
from gpu_fault.store import WorkflowLeaseError
from tests._builders import asgi_client, build_context, copy_model
from tests.regional._regional_support import (
    NOW,
    TOKEN_A,
    TOKEN_B,
    registration,
    remote_context,
)
from tests.regional._regional_support import enqueue_remote_command as enqueue

CLAIM = "/v1/regional/executors/claim"
KUBERNETES_OWNER = "gpu-fault-kubernetes-adapter"
HYPERPOD_OWNER = "gpu-fault-hyperpod-adapter"
NODE_AGENT_OWNER = "gpu-fault-node-agent"
COMMAND_ID = re.compile(r"\Aremote-[0-9a-f]{24}\Z")
CLUSTER_HEADERS = {
    "Authorization": f"Bearer {TOKEN_A}",
    "X-GPU-Fault-Cluster-ID": "cluster-a",
}


def regional_context():
    context = build_context()
    context.regional_mode = True
    context.execution_token = "e" * 32
    context.store.save_regional_cluster(registration("cluster-a", TOKEN_A))
    context.store.save_regional_cluster(registration("cluster-b", TOKEN_B))
    return context


def result(lease_token: str, status: str, details: dict) -> RemoteCommandResult:
    return RemoteCommandResult(
        lease_token=lease_token, status=RemoteCommandStatus(status), details=details
    )


def seconds_until(deadline: str) -> float:
    parsed = datetime.fromisoformat(deadline)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return (parsed - datetime.now(timezone.utc)).total_seconds()


def test_cmd001_max_commands_is_enforced_at_both_ends_of_its_range() -> None:
    """``ge=1, le=25`` is a batch bound, not a suggestion.

    An unbounded batch lets one executor lease the entire regional backlog in a
    single poll, and every command it then fails to run waits out a full lease
    before anyone else can take it.
    """

    context = regional_context()
    for index in range(26):
        enqueue(context.store, f"remote-{index:024x}")

    async def scenario() -> None:
        async with asgi_client(context) as client:
            answers = {}
            for label, value in (
                ("zero", 0),
                ("one", 1),
                ("twenty-five", 25),
                ("twenty-six", 26),
                ("negative", -1),
                ("numeric-string", "5"),
                ("fractional", 5.5),
            ):
                response = await client.post(
                    CLAIM,
                    headers=CLUSTER_HEADERS,
                    json={"executor_id": "executor-a", "max_commands": value},
                )
                claimed = (
                    response.json()["commands"] if response.status_code == 200 else []
                )
                answers[label] = (response.status_code, len(claimed))
                # Hand every leased command back, so the next boundary starts from
                # the same 26-command backlog instead of a shrinking one.
                for command in claimed:
                    returned = await client.post(
                        f"/v1/regional/executors/{command['command_id']}/result",
                        headers=CLUSTER_HEADERS,
                        json={
                            "lease_token": command["lease_token"],
                            "status": "WAITING",
                        },
                    )
                    assert returned.status_code == 200, returned.text

            assert answers == {
                "zero": (422, 0),
                "one": (200, 1),
                "twenty-five": (200, 25),
                "twenty-six": (422, 0),
                "negative": (422, 0),
                # JSON has one number type and a form field has none, so a numeric
                # string is a legitimate integer; 5.5 is not an integer at all.
                "numeric-string": (200, 5),
                "fractional": (422, 0),
            }

    asyncio.run(scenario())


def test_cmd002_lease_seconds_bounds_and_the_deadline_it_returns() -> None:
    """The lease is the only thing keeping two executors off one command.

    A lease shorter than the poll interval expires under a healthy executor and
    hands its command to a second one mid-action; a lease longer than two hours
    parks a destructive command behind a dead pod for the rest of the shift. And
    once a lease is gone the result has to be refused, or the executor that lost
    it can still close a command somebody else is now running.
    """

    context = regional_context()
    for index in range(4):
        enqueue(context.store, f"remote-{index:024x}")
    expired = enqueue(
        context.store,
        "remote-" + "f" * 24,
        status=RemoteCommandStatus.LEASED,
        lease_owner="executor-a",
        lease_token="lease-token-that-has-expired",
        lease_expires_at=NOW - timedelta(seconds=5),
    )

    async def scenario() -> None:
        async with asgi_client(context) as client:
            statuses = {}
            granted = {}
            for value in (9, 0, 7201, 10, 600, 601, 7200):
                response = await client.post(
                    CLAIM,
                    headers=CLUSTER_HEADERS,
                    json={
                        "executor_id": "executor-a",
                        "lease_seconds": value,
                        "max_commands": 1,
                    },
                )
                statuses[value] = response.status_code
                if response.status_code != 200:
                    continue
                for command in response.json()["commands"]:
                    deadline = command["lease_expires_at"]
                    assert deadline is not None, command
                    # The deadline is the executor's only clock for renewals, so it
                    # has to be the window that was asked for rather than a
                    # server-side default.
                    granted[value] = seconds_until(deadline)

            stale = await client.post(
                f"/v1/regional/executors/{expired.command_id}/result",
                headers=CLUSTER_HEADERS,
                json={"lease_token": expired.lease_token, "status": "SUCCEEDED"},
            )

            assert statuses == {
                9: 422,
                0: 422,
                7201: 422,
                10: 200,
                600: 200,
                601: 200,
                7200: 200,
            }
            assert sorted(granted) == [10, 600, 601, 7200]
            for window, deadline in granted.items():
                assert abs(deadline - window) <= 3, (window, deadline)
            assert stale.status_code == 409, stale.text
            assert (
                context.store.get_remote_command(expired.command_id).status
                is RemoteCommandStatus.LEASED
            )

    asyncio.run(scenario())


def test_cmd004_a_claim_only_takes_commands_for_the_owners_it_asked_for() -> None:
    """An adapter that leases another adapter's command cannot execute it.

    It will lease it, fail to run it, and hold the lease until it expires -- which
    reads as a stuck recovery rather than a routing bug. The three owners here are
    the ones the dispatcher actually assigns.
    """

    context = regional_context()
    enqueue(context.store, "remote-" + "1" * 24, owner=KUBERNETES_OWNER)
    enqueue(context.store, "remote-" + "2" * 24, owner=HYPERPOD_OWNER)
    enqueue(context.store, "remote-" + "3" * 24, owner=NODE_AGENT_OWNER)

    async def scenario() -> None:
        async with asgi_client(context) as client:
            first = await client.post(
                CLAIM,
                headers=CLUSTER_HEADERS,
                json={
                    "executor_id": "executor-a",
                    "execution_owners": [NODE_AGENT_OWNER],
                    "max_commands": 25,
                },
            )
            rest = await client.post(
                CLAIM,
                headers=CLUSTER_HEADERS,
                json={
                    "executor_id": "executor-a",
                    "execution_owners": [KUBERNETES_OWNER, HYPERPOD_OWNER],
                    "max_commands": 25,
                },
            )

            node_agent = first.json()["commands"]
            others = rest.json()["commands"]

            assert [item["step"]["execution_owner"] for item in node_agent] == [
                NODE_AGENT_OWNER
            ]
            assert sorted(item["step"]["execution_owner"] for item in others) == [
                HYPERPOD_OWNER,
                KUBERNETES_OWNER,
            ]
            claimed_ids = [item["command_id"] for item in node_agent + others]
            assert len(set(claimed_ids)) == 3, claimed_ids

    asyncio.run(scenario())


def test_cmd005_the_claim_order_is_fifo_and_ties_break_deterministically() -> None:
    """Otherwise the oldest destructive command starves behind newer arrivals.

    A backlog that grows faster than one executor drains it will keep serving the
    newest commands forever if the order is anything but creation order, and the
    starved command is by construction the one that has already waited longest on
    a fenced GPU. The tie-break matters for the same reason: two executors polling
    the same instant have to agree about what comes next.
    """

    context = regional_context()
    for index in range(5):
        enqueue(
            context.store,
            f"remote-{index:024x}",
            created_at=NOW + timedelta(seconds=5 * index),
        )
    tie = NOW + timedelta(seconds=100)
    enqueue(context.store, "remote-" + "b" * 24, created_at=tie)
    enqueue(context.store, "remote-" + "a" * 24, created_at=tie)

    async def scenario() -> None:
        async with asgi_client(context) as client:
            response = await client.post(
                CLAIM,
                headers=CLUSTER_HEADERS,
                json={"executor_id": "executor-a", "max_commands": 25},
            )

            commands = response.json()["commands"]
            created = [item["created_at"] for item in commands]

            assert len(commands) == 7, commands
            assert created == sorted(created)
            assert [item["command_id"] for item in commands[-2:]] == [
                "remote-" + "a" * 24,
                "remote-" + "b" * 24,
            ]

    asyncio.run(scenario())


def test_cmd007_a_terminal_result_is_idempotent_and_cannot_be_rewritten() -> None:
    """A retry must not walk a finished recovery back to FAILED.

    The executor retries on any network error, including one that ate a 200, so a
    second SUCCEEDED submission is ordinary traffic. Honouring a later FAILED
    would resurrect a closed incident; rejecting the retry because the lease token
    has already been cleared would make the executor retry forever.
    """

    context = regional_context()
    command = enqueue(context.store, "remote-" + "7" * 24)

    async def scenario() -> None:
        async with asgi_client(context) as client:
            claim = await client.post(
                CLAIM,
                headers=CLUSTER_HEADERS,
                json={"executor_id": "executor-a", "max_commands": 1},
            )
            lease_token = claim.json()["commands"][0]["lease_token"]
            path = f"/v1/regional/executors/{command.command_id}/result"
            first = await client.post(
                path,
                headers=CLUSTER_HEADERS,
                json={
                    "lease_token": lease_token,
                    "status": "SUCCEEDED",
                    "details": {"node": "node-a"},
                },
            )
            repeats = []
            for body in (
                {"lease_token": lease_token, "status": "SUCCEEDED"},
                {"lease_token": "a-token-nobody-issued", "status": "SUCCEEDED"},
                {
                    "lease_token": lease_token,
                    "status": "FAILED",
                    "error": "executor changed its mind",
                },
            ):
                response = await client.post(path, headers=CLUSTER_HEADERS, json=body)
                repeats.append((response.status_code, response.json()))

            assert first.status_code == 200, first.text
            assert [status for status, _ in repeats] == [200, 200, 200]
            assert [payload for _, payload in repeats] == [first.json()] * 3
            assert first.json()["status"] == "SUCCEEDED"
            assert first.json()["error"] is None
            assert first.json()["result_details"] == {"node": "node-a"}

    asyncio.run(scenario())


def test_cmd008_an_illegal_result_is_rejected_and_changes_nothing() -> None:
    """The result body is the one place an executor can write control-plane state.

    ``PENDING``/``LEASED`` are claim-side statuses; accepting one would leave a
    command that no executor holds and no reaper will ever expire. A ``FAILED``
    with no error text produces an incident nobody can triage. An unknown field
    means the executor and the control plane disagree about the protocol, which is
    exactly when guessing is worst. Each of these has to be refused with the
    command left exactly as it was.
    """

    context = regional_context()
    command = enqueue(context.store, "remote-" + "8" * 24)

    async def scenario() -> None:
        async with asgi_client(context) as client:
            claim = await client.post(
                CLAIM,
                headers=CLUSTER_HEADERS,
                json={"executor_id": "executor-a", "max_commands": 1},
            )
            lease_token = claim.json()["commands"][0]["lease_token"]
            path = f"/v1/regional/executors/{command.command_id}/result"
            answers = {}
            for label, body in (
                ("pending", {"lease_token": lease_token, "status": "PENDING"}),
                ("leased", {"lease_token": lease_token, "status": "LEASED"}),
                (
                    "failed-without-error",
                    {"lease_token": lease_token, "status": "FAILED"},
                ),
                (
                    "failed-empty-error",
                    {"lease_token": lease_token, "status": "FAILED", "error": ""},
                ),
                ("missing-lease-token", {"status": "SUCCEEDED"}),
                ("unknown-status", {"lease_token": lease_token, "status": "DONE"}),
                (
                    "unknown-field",
                    {
                        "lease_token": lease_token,
                        "status": "SUCCEEDED",
                        "surprise": "extra",
                    },
                ),
            ):
                response = await client.post(path, headers=CLUSTER_HEADERS, json=body)
                stored = context.store.get_remote_command(command.command_id)
                answers[label] = (response.status_code, stored.status.value)
                assert stored.lease_token == lease_token, label

            accepted = await client.post(
                path,
                headers=CLUSTER_HEADERS,
                json={"lease_token": lease_token, "status": "WAITING"},
            )

            assert answers == {
                "pending": (422, "LEASED"),
                "leased": (422, "LEASED"),
                "failed-without-error": (422, "LEASED"),
                "failed-empty-error": (422, "LEASED"),
                "missing-lease-token": (422, "LEASED"),
                "unknown-status": (422, "LEASED"),
                "unknown-field": (422, "LEASED"),
            }
            # The same route accepts a legal status, so the sweep above is
            # rejecting these bodies rather than the endpoint.
            assert accepted.status_code == 200, accepted.text
            assert accepted.json()["status"] == "WAITING"

    asyncio.run(scenario())


def test_cmd009_an_unknown_and_a_foreign_command_are_the_same_404() -> None:
    """Existence must not be observable across the cluster boundary.

    If a foreign ``command_id`` answered 403 while an invented one answered 404,
    cluster A could enumerate what cluster B is currently recovering -- one probe
    per id -- while holding none of B's credentials.
    """

    context = regional_context()
    foreign = enqueue(context.store, "remote-" + "b" * 24, cluster_id="cluster-b")

    async def scenario() -> None:
        async with asgi_client(context) as client:
            answers = {}
            for label, command_id in (
                ("unknown", "remote-" + "0" * 24),
                ("foreign", foreign.command_id),
            ):
                response = await client.post(
                    f"/v1/regional/executors/{command_id}/result",
                    headers=CLUSTER_HEADERS,
                    json={"lease_token": "any-token", "status": "SUCCEEDED"},
                )
                answers[label] = (response.status_code, response.json()["detail"])

            # Both details are the same function of what the caller already sent,
            # so neither one says whether the command exists.
            assert answers == {
                "unknown": (404, "resource not found: cluster-a/remote-" + "0" * 24),
                "foreign": (404, f"resource not found: cluster-a/{foreign.command_id}"),
            }
            assert (
                context.store.get_remote_command(foreign.command_id).status
                is RemoteCommandStatus.PENDING
            )

    asyncio.run(scenario())


def test_cmd012_the_command_identity_is_reused_on_retry_and_new_on_a_fence() -> None:
    """The command id is a digest, which is what makes dispatch retry-safe.

    A dispatcher retry must not enqueue the same GPU reset twice; a workflow whose
    fencing token has moved on must not reuse the command the stale generation
    left behind. So an identical ``(request_id, step_index, fencing_token, step)``
    is the same command, and a bumped fence is a different one.
    """

    context = regional_context()
    store = context.store
    adapter = RegionalRemoteWorkflowAdapter(store, owners={KUBERNETES_OWNER})
    state = remote_context("cmd012", KUBERNETES_OWNER)

    for _ in range(3):
        adapter.execute(state)
    retried = store.list_remote_commands()

    fenced = copy_model(state.workflow, fencing_token=state.workflow.fencing_token + 1)
    adapter.execute(replace(state, workflow=fenced))
    after_fence = store.list_remote_commands()

    assert len(retried) == 1, [item.command_id for item in retried]
    assert len(after_fence) == 2, [item.command_id for item in after_fence]
    assert len({item.command_id for item in after_fence}) == 2
    assert {item.fencing_token for item in after_fence} == {3, 4}
    for item in after_fence:
        assert COMMAND_ID.match(item.command_id) is not None, item.command_id
    # The retries did not rewrite the queued command either: an in-flight action
    # keeps its original creation time and its PENDING status.
    assert (
        retried[0].created_at
        == store.get_remote_command(retried[0].command_id).created_at
    )
    assert retried[0].status is RemoteCommandStatus.PENDING


def test_cmd012_a_reused_command_id_with_different_content_is_refused() -> None:
    """Two different actions cannot share one command id.

    The id is a digest of the action, so a clash means either a hash collision or
    a caller reusing an id for other content. Either way it has to stop at the
    store rather than silently replace a queued mutation with a different one.
    """

    context = regional_context()
    enqueue(context.store, "remote-" + "c" * 24, owner=KUBERNETES_OWNER)

    with pytest.raises(ValueError, match="remote command identity conflict"):
        enqueue(context.store, "remote-" + "c" * 24, owner=HYPERPOD_OWNER)

    stored = context.store.get_remote_command("remote-" + "c" * 24)
    assert stored.step.execution_owner == KUBERNETES_OWNER


def test_cmd013_waiting_can_be_reclaimed_and_carries_the_previous_details() -> None:
    """A long action reports progress by handing the command back as WAITING.

    Each round has to see what the last round left, or a multi-stage action
    (cordon, wait for the pods to leave, then reset) restarts from the beginning
    on every poll and never finishes. Nothing caps the number of rounds, which is
    what the live case exercises: twenty rounds later the command is still
    claimable and still carries its progress.
    """

    context = regional_context()
    store = context.store
    command = enqueue(store, "remote-" + "d" * 24)
    seen = []

    for round_number in range(20):
        claimed = store.claim_remote_commands(
            "cluster-a",
            "executor-a",
            limit=1,
            lease_seconds=60,
            execution_owners={KUBERNETES_OWNER},
        )
        assert len(claimed) == 1, f"round {round_number} could not reclaim the command"
        seen.append(claimed[0].result_details.get("round"))
        store.complete_remote_command(
            "cluster-a",
            command.command_id,
            result(claimed[0].lease_token, "WAITING", {"round": round_number}),
        )

    # A command handed back as WAITING belongs to nobody: it has to be claimed
    # again before any result is accepted, or an executor that already let go of
    # it could still close it.
    with pytest.raises(WorkflowLeaseError):
        store.complete_remote_command(
            "cluster-a",
            command.command_id,
            result("a-token-nobody-issued", "WAITING", {}),
        )

    final = store.claim_remote_commands(
        "cluster-a",
        "executor-a",
        limit=1,
        lease_seconds=60,
        execution_owners={KUBERNETES_OWNER},
    )
    store.complete_remote_command(
        "cluster-a",
        command.command_id,
        result(final[0].lease_token, "SUCCEEDED", {"round": "done"}),
    )

    assert seen == [None] + list(range(19))
    assert store.get_remote_command(command.command_id).status is (
        RemoteCommandStatus.SUCCEEDED
    )


def test_cmd014_the_delegating_step_states_the_control_plane_did_not_mutate() -> None:
    """The audit chain needs the negative claim written down, not inferred.

    "The CPU control plane holds no GPU kubeconfig" is an architecture statement;
    the workflow step is where an auditor can see it per action -- which cluster
    the mutation went to, under which command id, and that this process did not
    perform it itself.
    """

    context = regional_context()
    adapter = RegionalRemoteWorkflowAdapter(context.store, owners={KUBERNETES_OWNER})
    state = remote_context("cmd014", KUBERNETES_OWNER)

    outcome = adapter.execute(state)

    assert outcome.status.value == "WAITING"
    assert outcome.details["mutation_submitted_by_control_plane"] is False
    assert outcome.details["remote_cluster_id"] == "cluster-a"
    assert outcome.details["remote_status"] == "PENDING"
    assert COMMAND_ID.match(outcome.details["remote_command_id"]) is not None, (
        outcome.details["remote_command_id"]
    )

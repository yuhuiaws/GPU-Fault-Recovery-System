"""``X-GPU-Fault-Command-ID`` on the executor's lease-fenced posts, server side.

The executor's result, lease renewal and progress posts carry the command id as
the transport's licence to repeat them once on a fresh connection. That repeat
is only safe because the control plane answers a repeated post under the same
lease token exactly as it answered the first, which is what these tests pin
against the real routes and store. The header is optional both ways during a
rolling upgrade: an executor that predates it is served unchanged, and a
control plane that predates it never reads it (a route ignores headers it does
not declare, as the claim and readiness routes still do).
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from gpu_fault.app.routes import regional as regional_routes
from gpu_fault.regional import RemoteCommandStatus
from gpu_fault.regional_compatibility import CURRENT_REGIONAL_EXECUTOR_PROTOCOL_VERSION
from gpu_fault.store import InMemoryStore, SqliteStore
from gpu_fault.transport.http_client import COMMAND_ID_HEADER, REQUEST_ID_HEADER
from tests._builders import asgi_client, build_context
from tests.regional._regional_support import (
    TOKEN_A,
    enqueue_remote_command,
    registration,
)

CLUSTER = "cluster-a"
EXECUTOR = "cluster-a/executor-0"
OWNER = "gpu-fault-kubernetes-adapter"
COMMAND_ID = "remote-" + "c" * 24
BASE = "/v1/regional/executors"


@pytest.fixture(params=["memory", "sqlite"])
def site(request: pytest.FixtureRequest, tmp_path: Path) -> Iterator[SimpleNamespace]:
    """One leased command; SQLite runs the shared transitions PostgreSQL runs."""

    store = (
        InMemoryStore()
        if request.param == "memory"
        else SqliteStore(str(tmp_path / "commands.db"))
    )
    context = build_context(store=store)
    context.regional_mode = True
    store.save_regional_cluster(registration(CLUSTER, TOKEN_A))
    enqueue_remote_command(store, COMMAND_ID, owner=OWNER)
    [leased] = store.claim_remote_commands(
        CLUSTER, EXECUTOR, limit=1, lease_seconds=60, execution_owners={OWNER}
    )
    try:
        yield SimpleNamespace(
            context=context, store=store, lease_token=leased.lease_token
        )
    finally:
        close = getattr(store, "close", None)
        if close is not None:
            close()


def posts(
    site: SimpleNamespace, *calls: tuple[str, dict[str, Any], dict[str, str]]
) -> list[tuple[int, Any]]:
    """Each ``(path, body, headers)`` in order, through one real application."""

    async def run() -> list[tuple[int, Any]]:
        answers = []
        async with asgi_client(site.context) as client:
            for path, body, headers in calls:
                response = await client.post(
                    path,
                    json=body,
                    headers={
                        "Authorization": f"Bearer {TOKEN_A}",
                        "X-GPU-Fault-Cluster-ID": CLUSTER,
                        **headers,
                    },
                )
                answers.append((response.status_code, response.json()))
        return answers

    return asyncio.run(run())


def renewal(site: SimpleNamespace) -> dict[str, Any]:
    return {
        "executor_id": EXECUTOR,
        "lease_token": site.lease_token,
        "lease_seconds": 60,
    }


def progress(site: SimpleNamespace) -> dict[str, Any]:
    return {
        "executor_id": EXECUTOR,
        "lease_token": site.lease_token,
        "batched_results": {"0": {"status": "SUCCEEDED", "details": {"seen": 1}}},
    }


def result(site: SimpleNamespace, status: str = "SUCCEEDED") -> dict[str, Any]:
    body: dict[str, Any] = {
        "lease_token": site.lease_token,
        "status": status,
        "details": {"verdict": status.lower()},
    }
    if status == "FAILED":
        body["error"] = "a late duplicate that says otherwise"
    return body


IDENTITY = {COMMAND_ID_HEADER: COMMAND_ID}


def test_a_repeated_lease_fenced_post_is_answered_as_the_first(
    site: SimpleNamespace,
) -> None:
    """What makes the transport's repeat safe: renewal and progress under the
    same lease token are idempotent, and a result for a command that is already
    terminal returns it unchanged -- whatever the repeat says."""

    (
        (renewed, first_renewal),
        (renewed_again, second_renewal),
        (progressed, first_progress),
        (progressed_again, second_progress),
        (completed, first_result),
        (completed_again, repeated_result),
        (contradicted, contradicting_result),
    ) = posts(
        site,
        (f"{BASE}/{COMMAND_ID}/renew", renewal(site), IDENTITY),
        (f"{BASE}/{COMMAND_ID}/renew", renewal(site), IDENTITY),
        (f"{BASE}/{COMMAND_ID}/progress", progress(site), IDENTITY),
        (f"{BASE}/{COMMAND_ID}/progress", progress(site), IDENTITY),
        (f"{BASE}/{COMMAND_ID}/result", result(site), IDENTITY),
        (f"{BASE}/{COMMAND_ID}/result", result(site), IDENTITY),
        (f"{BASE}/{COMMAND_ID}/result", result(site, "FAILED"), IDENTITY),
    )

    assert (renewed, renewed_again) == (200, 200)
    assert second_renewal["lease_token"] == first_renewal["lease_token"]
    assert second_renewal["status"] == first_renewal["status"] == "LEASED"
    assert second_renewal["lease_expires_at"] >= first_renewal["lease_expires_at"], (
        "a repeated renewal only moves the same lease's expiry"
    )
    assert (progressed, progressed_again) == (200, 200)
    entry = {"status": "SUCCEEDED", "status_source": None, "details": {"seen": 1}}
    assert (
        second_progress["result_details"]
        == first_progress["result_details"]
        == {"batched_results": {"0": {**entry, "error": None}}}
    ), "a repeated progress post merges entries it leaves equal"
    assert (completed, completed_again, contradicted) == (200, 200, 200)
    assert first_result["status"] == "SUCCEEDED"
    assert repeated_result == first_result == contradicting_result, (
        "an already-terminal command is returned unchanged"
    )
    stored = site.store.get_remote_command(COMMAND_ID)
    assert stored.status is RemoteCommandStatus.SUCCEEDED
    assert stored.result_details == {"verdict": "succeeded"}


@pytest.mark.parametrize("route", ["renew", "progress", "result"])
def test_a_header_naming_another_command_is_refused_before_the_store(
    site: SimpleNamespace, route: str
) -> None:
    body = {"renew": renewal, "progress": progress, "result": result}[route](site)
    before = site.store.get_remote_command(COMMAND_ID)

    [(status, answer)] = posts(
        site,
        (
            f"{BASE}/{COMMAND_ID}/{route}",
            body,
            {COMMAND_ID_HEADER: "remote-" + "d" * 24},
        ),
    )

    assert status == 400
    assert COMMAND_ID_HEADER in answer["detail"]
    assert site.store.get_remote_command(COMMAND_ID) == before, "nothing was written"


def test_an_executor_that_predates_the_header_is_served_unchanged(
    site: SimpleNamespace,
) -> None:
    answers = posts(
        site,
        (f"{BASE}/{COMMAND_ID}/renew", renewal(site), {}),
        (f"{BASE}/{COMMAND_ID}/progress", progress(site), {}),
        (f"{BASE}/{COMMAND_ID}/result", result(site), {}),
    )

    assert [status for status, _answer in answers] == [200, 200, 200]
    assert answers[-1][1]["status"] == "SUCCEEDED"


def test_a_control_plane_that_predates_the_header_ignores_it(
    site: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An older control plane declares the header on no route. The claim and
    readiness routes still do not, and every route behaves like them once the
    check is gone: the identity changes nothing about the answer."""

    monkeypatch.setattr(regional_routes, "require_command_identity", lambda *_: None)
    executor = {
        "executor_id": EXECUTOR,
        "executor_protocol_version": CURRENT_REGIONAL_EXECUTOR_PROTOCOL_VERSION,
        "execution_owners": [OWNER],
    }
    foreign = {COMMAND_ID_HEADER: "remote-" + "d" * 24, REQUEST_ID_HEADER: "a" * 32}
    answers = posts(
        site,
        (
            f"{BASE}/claim",
            {**executor, "max_commands": 1, "lease_seconds": 60},
            foreign,
        ),
        (f"{BASE}/readiness", executor, foreign),
        (f"{BASE}/{COMMAND_ID}/renew", renewal(site), foreign),
        (f"{BASE}/{COMMAND_ID}/progress", progress(site), foreign),
        (f"{BASE}/{COMMAND_ID}/result", result(site), foreign),
    )

    statuses = [status for status, _answer in answers]
    assert statuses == [200, 200, 200, 200, 200], answers
    assert answers[0][1] == {"commands": []}, "the only command is already leased"
    assert answers[1][1]["ready"] is True
    assert answers[-1][1]["status"] == "SUCCEEDED"

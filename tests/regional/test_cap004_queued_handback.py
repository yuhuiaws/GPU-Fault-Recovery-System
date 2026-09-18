"""CAP004 queue handbacks use the real local API and Executor lifecycle."""

from __future__ import annotations

import hashlib
import json
import threading
from collections.abc import Iterator
from datetime import datetime, timedelta, timezone
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.response import addinfourl

import pytest
from anyio.from_thread import start_blocking_portal

from gpu_fault.app import create_app
from gpu_fault.cluster_executor import ClusterActionExecutor, ClusterExecutorError
from gpu_fault.regional import (
    RemoteActionCommand,
    RemoteCommandResult,
    RemoteCommandStatus,
)
from gpu_fault.store import InMemoryStore
from scripts.e2e.regional import capacity_queued_lease as proof
from scripts.e2e.regional.probes import cap004_commands as commands
from tests._builders import asgi_client, build_context
from tests.regional._regional_support import registration

RUN_ID = "cap004queueack"
URL = "http://127.0.0.1:18765"
TOKEN = "q" * 32
TIME_SCALE = 0.2


@pytest.fixture
def local_api(monkeypatch: pytest.MonkeyPatch) -> Iterator[SimpleNamespace]:
    store = InMemoryStore()
    context = build_context(store=store, execution_token="e" * 32)
    context.regional_mode = True
    store.save_regional_cluster(registration(commands.CLUSTER_ID, TOKEN))
    monkeypatch.setenv("GPU_FAULT_SERVICE_ROLE", "ingress")
    monkeypatch.setenv("GPU_FAULT_TELEMETRY_SPOOL", "false")
    monkeypatch.setenv("GPU_FAULT_STORE_IO_WORKERS", "4")
    monkeypatch.setenv(
        "GPU_FAULT_REGIONAL_CLUSTERS_JSON",
        json.dumps(
            [
                {
                    "cluster_id": commands.CLUSTER_ID,
                    "token": TOKEN,
                    "hyperpod_cluster_name": f"hp-{commands.CLUSTER_ID}",
                }
            ]
        ),
    )
    app = create_app(context)
    state = SimpleNamespace(
        store=store,
        before=lambda _path, _payload: None,
        after=lambda _path, _payload, status, body: (status, body),
    )

    class FastRenewalEvent(threading.Event):
        def wait(self, timeout: float | None = None) -> bool:
            return super().wait(None if timeout is None else timeout * TIME_SCALE)

    def lease_deadline(seconds: int) -> datetime:
        return datetime.now(timezone.utc) + timedelta(seconds=seconds * TIME_SCALE)

    monkeypatch.setattr("gpu_fault.cluster_executor.lease.Event", FastRenewalEvent)
    monkeypatch.setattr(
        "gpu_fault.store.memory.remote_commands.lease_deadline", lease_deadline
    )
    monkeypatch.setattr(
        "gpu_fault.store.shared.remote_commands.lease_deadline", lease_deadline
    )
    with start_blocking_portal() as portal:

        async def request_api(request: Any, path: str) -> tuple[int, bytes]:
            async with asgi_client(app) as api:
                response = await api.request(
                    request.get_method(),
                    path,
                    headers=dict(request.header_items()),
                    content=request.data,
                )
            return response.status_code, response.content

        def urlopen(request: Any, **_kwargs: Any) -> addinfourl:
            assert urlsplit(request.full_url).netloc == "127.0.0.1:18765", (
                "the production client must only address the isolated local API"
            )
            path = urlsplit(request.full_url).path
            payload = json.loads(request.data)
            state.before(path, payload)
            status, body = portal.call(request_api, request, path)
            status, body = state.after(path, payload, status, body)
            stream = BytesIO(body)
            if status >= 400:
                raise HTTPError(request.full_url, status, "local API", {}, stream)
            return addinfourl(stream, {}, request.full_url, status)

        monkeypatch.setattr(
            "gpu_fault.cluster_executor.regional_client.urlopen", urlopen
        )
        try:
            with portal.wrap_async_context_manager(app.router.lifespan_context(app)):
                commands.command_snapshot(store, RUN_ID, "seed")
                yield state
        finally:
            app.state.store_io.close()


@pytest.mark.parametrize("race", ["none", "before-ack", "after-ack"])
def test_queue_accepts_only_acknowledged_waiting_renewal_races(
    race: str,
    local_api: SimpleNamespace,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_id = commands.commands_for_run(RUN_ID)[0].command_id
    completing = threading.Event()
    renewal_waiting = threading.Event()
    committed = threading.Event()
    acknowledged = threading.Event()
    rejected = threading.Event()
    events: list[str] = []
    executors: list[ClusterActionExecutor] = []
    threads_before = set(threading.enumerate())

    def is_handback(path: str, payload: dict[str, Any]) -> bool:
        return (
            path.endswith(f"/{first_id}/result")
            and payload["status"] == "WAITING"
            and payload.get("status_source") is None
        )

    def before(path: str, payload: dict[str, Any]) -> None:
        if race == "none":
            return
        if is_handback(path, payload):
            completing.set()
            assert renewal_waiting.wait(5), (
                "the WAITING handback must overlap an in-flight renewal"
            )
        elif path.endswith(f"/{first_id}/renew") and completing.is_set():
            renewal_waiting.set()
            gate = committed if race == "before-ack" else acknowledged
            assert gate.wait(5), "the renewal must follow the selected handback event"

    def after(
        path: str, payload: dict[str, Any], status: int, body: bytes
    ) -> tuple[int, bytes]:
        if race != "none" and is_handback(path, payload):
            assert status == 200, "the WAITING result must really commit"
            saved = next(
                item
                for item in local_api.store.list_remote_commands()
                if item.command_id == first_id
            )
            assert saved.status is RemoteCommandStatus.WAITING, (
                "the ACK must correspond to a durable WAITING handback"
            )
            assert saved.lease_token is None, "the handback must release its lease"
            events.append("committed")
            committed.set()
            if race == "before-ack":
                assert rejected.wait(5), "hold the ACK until the renewal is rejected"
        elif (
            race != "none"
            and path.endswith(f"/{first_id}/renew")
            and completing.is_set()
        ):
            assert status == 409, "the API must reject the released lease"
            events.append("renewal-rejected")
            rejected.set()
        return status, body

    def executor(client: Any, *args: Any, **kwargs: Any) -> ClusterActionExecutor:
        complete = client.complete

        def acknowledge(command: Any, result: Any) -> Any:
            saved = complete(command, result)
            if command.command_id == first_id and race != "none":
                events.append("acknowledged")
                acknowledged.set()
            return saved

        monkeypatch.setattr(client, "complete", acknowledge)
        value = ClusterActionExecutor(client, *args, **kwargs)
        executors.append(value)
        return value

    local_api.before = before
    local_api.after = after
    monkeypatch.setattr(proof, "ClusterActionExecutor", executor)
    try:
        result = proof.run_queued_lease_proof(
            URL, TOKEN, RUN_ID, tmp_path, timeout_seconds=10
        )
    finally:
        assert len(executors) == 1, "the proof must use one production Executor"
        counters = executors[0].metrics_snapshot()
        assert counters["lease_lost_total"] == (1 if race == "none" else 2), (
            "raw losses must retain both the queued loss and any handback race"
        )
        assert counters["lease_renewal_failures"] == (0 if race == "none" else 1), (
            "only the handback's overlapping renewal may fail inside the Executor"
        )
        assert counters["results_withheld_total"] == 1, (
            "the expired queued command must still withhold its result"
        )
        assert (
            events
            == {
                "none": [],
                "before-ack": ["committed", "renewal-rejected", "acknowledged"],
                "after-ack": ["committed", "acknowledged", "renewal-rejected"],
            }[race]
        ), "the regression must exercise the selected real commit/ACK order"
        assert not [
            thread
            for thread in threading.enumerate()
            if thread not in threads_before
            and thread.name.startswith(
                (RUN_ID, "gpu-fault-command", "cmd-command-", "lease-command-")
            )
        ], "no proof thread may survive cleanup"
    assert result["passed"] is True, "an acknowledged handback is not a queued loss"
    assert result["executor_counters"] == counters, "raw evidence must be preserved"
    assert result["adapter_calls"] == 1 and result["queued_adapter_calls"] == 0, (
        "only the first nonphysical command may reach an adapter"
    )
    assert result["competitor_claims"] == 1 and result["stale_renewal_status"] == 409, (
        "the queued lease must actually be stolen and its old authority rejected"
    )
    original = executors[0].client.claimed[0]
    digest = hashlib.sha256(original.lease_token.encode()).hexdigest()
    assert result["acknowledged_waiting_handbacks"] == [
        {"command_id": first_id, "lease_sha256": digest, "count": 1}
    ], "the ACK evidence must identify the exact original lease, without its token"
    assert result["handback_renewal_rejections"] == (
        {} if race == "none" else {first_id: 1}
    ), "only the first command's acknowledged handback may explain a renewal"
    assert result["unconfirmed_renewal_rejections"] == 0, (
        "all rejected renewals must be attributed"
    )
    assert result["renewal_rejections"] == (
        []
        if race == "none"
        else [
            {
                "command_id": first_id,
                "lease_sha256": digest,
                "count": 1,
                "acknowledged_waiting": True,
            }
        ]
    ), "raw renewal evidence must retain the original command, lease and count"
    assert original.lease_token not in json.dumps(result), (
        "the evidence must not expose a lease token"
    )
    assert all(
        item.status is RemoteCommandStatus.WAITING and item.lease_token is None
        for item in local_api.store.list_remote_commands()
    ), "all 25 commands must be handed back without claiming execution success"


@pytest.fixture
def queue_client(local_api: SimpleNamespace) -> proof.QueueClient:
    client = proof.QueueClient(
        URL,
        TOKEN,
        executor_id=f"{RUN_ID}-queued-executor",
        selected=commands.commands_for_run(RUN_ID)[:2],
    )
    client.claim(
        client.executor_id,
        execution_owners=[commands.OWNER],
        max_commands=2,
        lease_seconds=10,
    )
    return client


def waiting_result(command: RemoteActionCommand) -> RemoteCommandResult:
    assert command.lease_token, "the fixture must hold the original lease"
    return RemoteCommandResult(
        lease_token=command.lease_token,
        status=RemoteCommandStatus.WAITING,
        details={"nonphysical": True, "queue_admission_only": True},
    )


def assert_rejected_renewal(
    client: proof.QueueClient, command: RemoteActionCommand, owner: str
) -> None:
    with pytest.raises(ClusterExecutorError) as error:
        client.renew(command, owner, 10)
    assert error.value.status_code == 409, "the real Store must reject this lease"


@pytest.mark.parametrize("boundary", ["wrong-command", "wrong-token", "wrong-owner"])
def test_handback_does_not_explain_another_renewal_authority(
    boundary: str, queue_client: proof.QueueClient
) -> None:
    first, queued = queue_client.claimed
    queue_client.complete(first, waiting_result(first))
    if boundary == "wrong-command":
        command = first.model_copy(update={"command_id": queued.command_id})
    elif boundary == "wrong-token":
        command = first.model_copy(update={"lease_token": "different-test-lease"})
    else:
        command = first
    assert_rejected_renewal(
        queue_client,
        command,
        "another-executor" if boundary == "wrong-owner" else queue_client.executor_id,
    )
    if boundary == "wrong-owner":
        assert queue_client.problems == ["unexpected queued renewal failure"], (
            "a wrong owner must fail the proof rather than enter ACK reconciliation"
        )
        assert queue_client.renewal_evidence() == ({}, 0)
    else:
        assert queue_client.renewal_evidence() == ({}, 1), (
            "an ACK for another command or token must leave the rejection unexplained"
        )


def test_a_later_lease_ack_cannot_explain_the_original_rejection(
    queue_client: proof.QueueClient,
) -> None:
    first = queue_client.claimed[0]
    wire = proof.CapacityWireClient(URL, TOKEN, cluster_id=commands.CLUSTER_ID)
    wire.complete(first, waiting_result(first))
    assert_rejected_renewal(queue_client, first, queue_client.executor_id)
    reclaimed = wire.claim(
        queue_client.executor_id,
        execution_owners=[commands.OWNER],
        max_commands=1,
        lease_seconds=10,
    )[0]
    assert reclaimed.command_id == first.command_id, "reclaim the same command"
    assert reclaimed.lease_token != first.lease_token, "reclaim must change authority"
    with pytest.raises(proof.Cap004Error, match="original authority"):
        queue_client.complete(reclaimed, waiting_result(reclaimed))
    assert queue_client.renewal_evidence() == ({}, 1), (
        "a new lease cannot provide the original lease's missing ACK"
    )


@pytest.mark.parametrize(
    "boundary", ["command", "command-token", "result-token", "succeeded", "details"]
)
def test_invalid_handback_request_is_refused_before_any_effect(
    boundary: str, queue_client: proof.QueueClient, local_api: SimpleNamespace
) -> None:
    first, queued = queue_client.claimed
    command = queued if boundary == "command" else first
    result = waiting_result(command)
    if boundary == "command-token":
        command = first.model_copy(update={"lease_token": "different-test-lease"})
    elif boundary == "result-token":
        result = result.model_copy(update={"lease_token": "different-test-lease"})
    elif boundary == "succeeded":
        result = result.model_copy(update={"status": RemoteCommandStatus.SUCCEEDED})
    elif boundary == "details":
        result = result.model_copy(update={"details": {"nonphysical": False}})
    before = local_api.store.list_remote_commands()
    with pytest.raises(proof.Cap004Error, match="original authority"):
        queue_client.complete(command, result)
    assert local_api.store.list_remote_commands() == before, (
        "invalid handbacks must not change even the nonphysical Store state"
    )
    assert queue_client.completed_leases == {}, "no invalid request may count as an ACK"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("command_id", "command-cap004queueack-cap004-001"),
        ("cluster_id", "another-cluster"),
        ("last_lease_owner", "another-executor"),
        ("status", "SUCCEEDED"),
        ("status", "FAILED"),
        ("lease_token", "not-released"),
        ("lease_token", ""),
        ("lease_owner", "not-released"),
        ("lease_expires_at", "2099-01-01T00:00:00Z"),
        ("result_details", {"nonphysical": False}),
        ("error", "different outcome"),
        ("status_source", "different outcome"),
    ],
    ids=[
        "wrong-command",
        "wrong-cluster",
        "wrong-owner",
        "succeeded",
        "failed",
        "lease-token",
        "empty-lease-token",
        "lease-owner",
        "lease-expiry",
        "wrong-details",
        "error",
        "wrong-source",
    ],
)
def test_invalid_waiting_ack_cannot_explain_a_renewal_rejection(
    field: str, value: Any, queue_client: proof.QueueClient, local_api: SimpleNamespace
) -> None:
    first = queue_client.claimed[0]

    def after(
        path: str, _payload: dict[str, Any], status: int, body: bytes
    ) -> tuple[int, bytes]:
        if path.endswith(f"/{first.command_id}/result"):
            assert status == 200, "the real Store must commit before corrupting the ACK"
            document = json.loads(body)
            document[field] = value
            body = json.dumps(document).encode()
        return status, body

    local_api.after = after
    with pytest.raises((proof.Cap004Error, ValueError)):
        queue_client.complete(first, waiting_result(first))
    assert_rejected_renewal(queue_client, first, queue_client.executor_id)
    assert queue_client.renewal_evidence() == ({}, 1), (
        "a malformed or non-WAITING ACK must never explain a stale renewal"
    )
    assert queue_client.completed_leases == {}, "invalid ACKs are not durable evidence"


def test_committed_waiting_without_an_ack_is_not_reconciled(
    queue_client: proof.QueueClient, local_api: SimpleNamespace
) -> None:
    first = queue_client.claimed[0]

    def after(
        path: str, _payload: dict[str, Any], status: int, body: bytes
    ) -> tuple[int, bytes]:
        if path.endswith(f"/{first.command_id}/result"):
            assert status == 200, "the transport failure must follow the real commit"
            raise ClusterExecutorError("controlled loss of the WAITING ACK")
        return status, body

    local_api.after = after
    with pytest.raises(ClusterExecutorError, match="loss of the WAITING ACK"):
        queue_client.complete(first, waiting_result(first))
    saved = next(
        item
        for item in local_api.store.list_remote_commands()
        if item.command_id == first.command_id
    )
    assert saved.status is RemoteCommandStatus.WAITING and saved.lease_token is None, (
        "this must exercise a committed but unacknowledged handback"
    )
    assert_rejected_renewal(queue_client, first, queue_client.executor_id)
    assert queue_client.renewal_evidence() == ({}, 1), (
        "durable state alone cannot replace the missing same-lease ACK"
    )


@pytest.mark.parametrize("loss_delta", [-1, 1])
def test_queue_proof_rejects_missing_or_unexplained_lease_losses(
    loss_delta: int,
    local_api: SimpleNamespace,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executors: list[ClusterActionExecutor] = []

    class ChangedLossCountExecutor(ClusterActionExecutor):
        def run_once(self) -> int:
            result = super().run_once()
            self.increment("lease_lost_total", loss_delta)
            return result

    def executor(*args: Any, **kwargs: Any) -> ClusterActionExecutor:
        value = ChangedLossCountExecutor(*args, **kwargs)
        executors.append(value)
        return value

    monkeypatch.setattr(proof, "ClusterActionExecutor", executor)
    with pytest.raises(proof.Cap004Error, match="queued lease authority proof failed"):
        proof.run_queued_lease_proof(URL, TOKEN, RUN_ID, tmp_path, timeout_seconds=10)
    counters = executors[0].metrics_snapshot()
    assert counters["lease_lost_total"] == 1 + loss_delta, (
        "the proof must not rewrite raw counts to fit its expectation"
    )
    assert counters["results_withheld_total"] == 1, (
        "this refusal must retain the real queued command's withheld result"
    )
    assert all(
        item.status is RemoteCommandStatus.WAITING and item.lease_token is None
        for item in local_api.store.list_remote_commands()
    ), "proof failure must still release the owned nonphysical fixture leases"

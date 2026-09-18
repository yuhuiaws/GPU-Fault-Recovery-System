"""Serial PostgreSQL proofs of CAP004; no cloud or acceptance runner calls."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import threading
from collections.abc import Iterator
from contextlib import redirect_stdout
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from io import BytesIO, StringIO
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import Request
from urllib.response import addinfourl
from uuid import uuid4

import pytest
from anyio.from_thread import start_blocking_portal
from fastapi import FastAPI

from gpu_fault.app import create_app
from gpu_fault.cluster_executor import ClusterExecutorError, regional_client
from gpu_fault.regional import (
    RemoteActionCommand,
    RemoteCommandResult,
    RemoteCommandStatus,
)
from gpu_fault.store import PostgresStore
from gpu_fault.store.shared.errors import WorkflowLeaseError
from scripts.e2e.regional import capacity_acceptance_executor as proof
from scripts.e2e.regional.capacity_acceptance_cases import cap004_progression_errors
from scripts.e2e.regional.probes import cap004_commands as probe
from scripts.e2e.regional.run_cap005_postgres_suite import database_url, validate_server
from tests._builders import asgi_client, build_context
from tests.regional._regional_support import registration

POSTGRES_URL = os.environ.get("GPU_FAULT_TEST_POSTGRES_URL", "")
pytestmark = pytest.mark.skipif(
    not POSTGRES_URL, reason="requires an allocated serial PostgreSQL 16 slot"
)
API_URL = "http://127.0.0.1:18765"
TOKEN = "t" * 32
TIME_SCALE = 0.05


@dataclass(frozen=True)
class ProbeDatabase:
    run_id: str
    name: str
    credentials: Path
    identity: Path


@pytest.fixture
def cap004_database(tmp_path: Path) -> Iterator[ProbeDatabase]:
    import psycopg
    from psycopg import sql

    if os.environ.get("PYTEST_XDIST_WORKER"):
        pytest.fail("CAP004 PostgreSQL must run serially, without xdist")
    validate_server(POSTGRES_URL)
    run_id = f"cap004pg{uuid4().hex[:16]}"
    name = f"gpu_fault_{run_id}_cap004"
    work = tmp_path / "probe-work"
    work.mkdir(mode=0o700)
    credentials = work / "store-url"
    credentials.touch(mode=0o600)
    credentials.write_text(database_url(POSTGRES_URL, name), encoding="utf-8")
    identity = work / "database-name"
    identity.write_text(name, encoding="utf-8")
    attempted_create = False
    try:
        with psycopg.connect(
            POSTGRES_URL,
            autocommit=True,
            connect_timeout=5,
            options="-c statement_timeout=10000",
        ) as connection:
            assert (
                connection.execute(
                    "SELECT 1 FROM pg_database WHERE datname=%s", (name,)
                ).fetchone()
                is None
            ), "a pre-existing database is not owned by this test"
            # Track ownership before CREATE can commit and lose its acknowledgement.
            attempted_create = True
            connection.execute(
                sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name))
            )
        yield ProbeDatabase(run_id, name, credentials, identity)
    finally:
        try:
            if attempted_create:
                with psycopg.connect(
                    POSTGRES_URL,
                    autocommit=True,
                    connect_timeout=5,
                    options="-c statement_timeout=10000",
                ) as connection:
                    connection.execute(
                        sql.SQL("DROP DATABASE IF EXISTS {} WITH (FORCE)").format(
                            sql.Identifier(name)
                        )
                    )
                    assert (
                        connection.execute(
                            "SELECT 1 FROM pg_database WHERE datname=%s", (name,)
                        ).fetchone()
                        is None
                    ), "the test-owned database survived teardown"
        finally:
            credentials.unlink(missing_ok=True)


@pytest.fixture
def postgres_api(
    cap004_database: ProbeDatabase, monkeypatch: pytest.MonkeyPatch
) -> Iterator[FastAPI]:
    store = PostgresStore(
        cap004_database.credentials.read_text(encoding="utf-8"),
        initialize_schema=True,
        pool_min_size=0,
        pool_max_size=8,
        pool_timeout_seconds=5,
    )
    app: FastAPI | None = None
    try:
        context = build_context(store=store, execution_token="e" * 32)
        context.regional_mode = True
        store.save_regional_cluster(registration(probe.CLUSTER_ID, TOKEN))
        monkeypatch.setenv("GPU_FAULT_SERVICE_ROLE", "ingress")
        monkeypatch.setenv("GPU_FAULT_TELEMETRY_SPOOL", "false")
        monkeypatch.setenv("GPU_FAULT_STORE_IO_WORKERS", "8")
        monkeypatch.setenv(
            "GPU_FAULT_REGIONAL_CLUSTERS_JSON",
            json.dumps(
                [
                    {
                        "cluster_id": probe.CLUSTER_ID,
                        "token": TOKEN,
                        "hyperpod_cluster_name": f"hp-{probe.CLUSTER_ID}",
                    }
                ]
            ),
        )
        app = create_app(context)

        class FastRenewalEvent(threading.Event):
            def wait(self, timeout: float | None = None) -> bool:
                return super().wait(None if timeout is None else timeout * TIME_SCALE)

        def lease_deadline(seconds: int) -> datetime:
            return datetime.now(timezone.utc) + timedelta(seconds=seconds * TIME_SCALE)

        # Keep the real claim SQL, locks, renewals and completion transactions.
        monkeypatch.setattr("gpu_fault.cluster_executor.lease.Event", FastRenewalEvent)
        monkeypatch.setattr(
            "gpu_fault.store.postgres.remote_commands.lease_deadline", lease_deadline
        )
        monkeypatch.setattr(
            "gpu_fault.store.shared.remote_commands.lease_deadline", lease_deadline
        )
        with start_blocking_portal() as portal:

            async def request_api(request: Request) -> tuple[int, bytes]:
                async with asgi_client(app) as api:
                    response = await api.request(
                        request.get_method(),
                        urlsplit(request.full_url).path,
                        headers=dict(request.header_items()),
                        content=request.data,
                    )
                return response.status_code, response.content

            def urlopen(request: Request, **_kwargs: Any) -> addinfourl:
                assert urlsplit(request.full_url).netloc == "127.0.0.1:18765", (
                    "the Executor may only address this isolated ASGI API"
                )
                status, body = portal.call(request_api, request)
                stream = BytesIO(body)
                if status >= 400:
                    raise HTTPError(
                        request.full_url, status, "isolated PostgreSQL API", {}, stream
                    )
                return addinfourl(stream, {}, request.full_url, status)

            monkeypatch.setattr(
                "gpu_fault.cluster_executor.regional_client.urlopen", urlopen
            )
            with portal.wrap_async_context_manager(app.router.lifespan_context(app)):
                yield app
    finally:
        if app is not None:
            app.state.store_io.close()
        store.close()


def invoke_probe(
    mode: str, database: ProbeDatabase, monkeypatch: pytest.MonkeyPatch
) -> dict[str, Any]:
    paths = {
        "/work/store-url": database.credentials,
        "/work/database-name": database.identity,
    }
    output = StringIO()
    with monkeypatch.context() as patched:
        patched.setattr(probe, "Path", paths.__getitem__)
        patched.setattr(
            sys, "argv", ["cap004_commands.py", mode, database.run_id, database.name]
        )
        with redirect_stdout(output):
            assert probe.main() == 0, "the actual CAP004 database probe failed"
    return json.loads(output.getvalue())


@pytest.mark.usefixtures("postgres_api")
def test_cleanup_reclaims_an_expired_lease_through_real_postgres(
    cap004_database: ProbeDatabase, monkeypatch: pytest.MonkeyPatch
) -> None:
    invoke_probe("seed", cap004_database, monkeypatch)
    deadlines = 0
    rejected: list[str] = []
    attempted: dict[str, set[str]] = {}
    complete = PostgresStore.complete_remote_command

    def expires_once(seconds: int) -> datetime:
        nonlocal deadlines
        deadlines += 1
        return datetime.now(timezone.utc) + timedelta(
            seconds=-1 if deadlines == 1 else seconds
        )

    def observe_completion(store, cluster_id, command_id, result):
        fingerprint = hashlib.sha256(result.lease_token.encode()).hexdigest()
        attempted.setdefault(command_id, set()).add(fingerprint)
        try:
            return complete(store, cluster_id, command_id, result)
        except WorkflowLeaseError:
            rejected.append(command_id)
            raise

    monkeypatch.setattr(
        "gpu_fault.store.postgres.remote_commands.lease_deadline", expires_once
    )
    monkeypatch.setattr(PostgresStore, "complete_remote_command", observe_completion)
    closed = invoke_probe("cleanup", cap004_database, monkeypatch)
    assert rejected, "the regression must exercise the real expired-lease rejection"
    assert all(len(attempted[command_id]) >= 2 for command_id in rejected), (
        "expired commands must complete under a newly claimed lease"
    )
    assert closed["status_counts"] == {"FAILED": probe.COMMAND_COUNT}
    assert all(not row["lease_present"] for row in closed["commands"]), (
        "cleanup must release every real PostgreSQL command lease"
    )


@pytest.mark.parametrize("late_renewal", ["none", "before-ack", "after-ack"])
def test_cap004_production_executor_and_probe_use_real_postgres(
    late_renewal: str,
    cap004_database: ProbeDatabase,
    postgres_api: FastAPI,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = cap004_database
    assert postgres_api.state.store_io.workers == 8, (
        "PostgreSQL API must allow concurrency"
    )
    before = set(threading.enumerate())
    seed = invoke_probe("seed", database, monkeypatch)
    assert seed["commands_created"] == 25, (
        "the real probe must seed exactly 25 commands"
    )
    assert seed["status_counts"] == {"PENDING": 25}, (
        "seed did not reach the owned database"
    )
    first_id = probe.commands_for_run(database.run_id)[0].command_id
    completing = threading.Event()
    renewal_waiting = threading.Event()
    committed = threading.Event()
    renewed = threading.Event()
    completion_events: list[str] = []
    acknowledged: set[str] = set()

    class DelayedRenewalClient(proof.Cap004Client):
        def renew(
            self, command: RemoteActionCommand, executor_id: str, lease_seconds: int
        ) -> RemoteActionCommand:
            delayed = (
                late_renewal != "none"
                and command.command_id == first_id
                and completing.is_set()
            )
            if delayed:
                renewal_waiting.set()
                assert committed.wait(5), (
                    "the delayed renewal must follow the PostgreSQL terminal commit"
                )
            try:
                return super().renew(command, executor_id, lease_seconds)
            except ClusterExecutorError as exc:
                if delayed:
                    assert exc.status_code == 409, (
                        "PostgreSQL must reject renewal of the completed lease"
                    )
                    completion_events.append("renewal-rejected")
                raise
            finally:
                if delayed:
                    renewed.set()

        def complete(
            self, command: RemoteActionCommand, result: RemoteCommandResult
        ) -> RemoteActionCommand:
            if command.command_id == first_id and late_renewal != "none":
                completing.set()
                assert renewal_waiting.wait(5), (
                    "completion must overlap a renewal without pausing earlier heartbeats"
                )
            saved = super().complete(command, result)
            acknowledged.add(command.command_id)
            if command.command_id == first_id and late_renewal != "none":
                completion_events.append("acknowledged")
                if late_renewal == "after-ack":
                    committed.set()
            return saved

    urlopen = regional_client.urlopen

    def result_ack(request: Request, **kwargs: Any) -> addinfourl:
        response: addinfourl = urlopen(request, **kwargs)
        if (
            late_renewal != "none"
            and urlsplit(request.full_url).path.endswith(f"/{first_id}/result")
            and isinstance(request.data, bytes)
            and json.loads(request.data)["status"] == "SUCCEEDED"
        ):
            assert response.getcode() == 200, (
                "the terminal result must really commit before the renewal rejection"
            )
            completion_events.append("committed")
            if late_renewal == "before-ack":
                committed.set()
                assert renewed.wait(5), (
                    "the renewal rejection must precede the terminal result ACK"
                )
        return response

    monkeypatch.setattr(proof, "Cap004Client", DelayedRenewalClient)
    monkeypatch.setattr(regional_client, "urlopen", result_ack)

    result = proof.run_executor_proof(
        API_URL,
        TOKEN,
        database.run_id,
        tmp_path,
        timeout_seconds=30,
        poll_seconds=0.05,
        minimum_hold_seconds=0.02,
        action_timeout_seconds=5,
    )
    terminal = invoke_probe("inspect", database, monkeypatch)
    closed = invoke_probe("cleanup", database, monkeypatch)

    assert result["problems"] == [], "PostgreSQL protocol or Executor proof failed"
    assert result["commands_claimed"] == 25, "the proof lost a unique command"
    assert result["max_concurrent_commands"] == 5, "the real scheduler cap changed"
    assert result["executor_batch_size"] == 5 and result["lease_seconds"] == 10, (
        "only five long actions may be leased to the production Executor at once"
    )
    assert (
        result["bulk_api"]["max_commands"]
        == result["bulk_api"]["waiting_handbacks"]
        == 25
    ), "the 25-command API measurement must hand back all leases before execution"
    assert (
        cap004_progression_errors(seed["commands"], result["lease_progressions"]) == []
    ), "all 25 PostgreSQL lease expiries must advance"
    assert result["injected_renewal_failure_status"] == 409, (
        "PostgreSQL must reject the controlled wrong lease owner"
    )
    counters = result["executor_counters"]
    assert counters["claimed_total"] == 26, (
        "one command must be reclaimed after lease loss"
    )
    terminal_rejections = result["terminal_renewal_rejections"]
    assert acknowledged == {row["command_id"] for row in seed["commands"]}, (
        "all 25 terminal results must be acknowledged by the production client"
    )
    assert all(
        count == 1 and command_id in acknowledged
        for command_id, count in terminal_rejections.items()
    ), "only one terminal renewal rejection per acknowledged command is explained"
    assert result["unconfirmed_renewal_rejections"] == 0, (
        "every terminal rejection must match the same acknowledged completion lease"
    )
    if late_renewal != "none":
        assert terminal_rejections[first_id] == 1, (
            "the coordinated command must receive a real PostgreSQL terminal rejection"
        )
    assert (
        completion_events
        == {
            "none": [],
            "before-ack": ["committed", "renewal-rejected", "acknowledged"],
            "after-ack": ["committed", "acknowledged", "renewal-rejected"],
        }[late_renewal]
    ), "the test must exercise the selected terminal commit/ACK ordering"
    expected_renewal_failures = 1 + sum(terminal_rejections.values())
    assert (
        counters["lease_renewal_failures"]
        == counters["lease_lost_total"]
        == expected_renewal_failures
    ), "only the injected failure and acknowledged terminal rejections are explained"
    assert counters["results_withheld_total"] == 1, (
        "only the original lost-lease holder must withhold its result"
    )
    assert result["ledger_exactly_once"] is True and len(result["ledger"]) == 25, (
        "each unique command must insert one nonphysical ledger entry"
    )
    assert sorted(row["calls"] for row in result["ledger"]) == [1] * 24 + [2], (
        "the new lease holder must reuse the durable ledger"
    )
    long_actions = [
        row
        for row in result["action_timings"]
        if not row["cached"]
        and row["adapter_succeeded"]
        and not row["lease_guard_held"]
        and row["elapsed_seconds"] > row["original_lease_seconds"]
    ]
    assert len({row["command_id"] for row in long_actions}) >= 5, (
        "at least five successful actions must outlive their own original PostgreSQL lease"
    )
    assert result["long_commands"] == len(long_actions), (
        "long-action evidence is inconsistent"
    )
    assert result["adapter_types"] == ["NonphysicalLedgerAdapter"], (
        "no NodeAgent, Kubernetes or provider adapter is allowed"
    )
    assert probe.terminal_errors(terminal, database.run_id) == [], (
        "actual PostgreSQL readback did not prove terminal, lease-free commands"
    )
    assert terminal == closed, "cleanup must preserve successful terminal results"
    assert result["background_threads_stopped"] is True, "Executor threads did not stop"
    assert not [
        thread
        for thread in threading.enumerate()
        if thread not in before
        and thread.name.startswith(
            (database.run_id, "gpu-fault-command", "cmd-command-", "lease-command-")
        )
    ], "no Executor thread may survive database teardown"


@pytest.mark.parametrize(
    "boundary", ["wrong-owner", "unacknowledged-result", "other-lease"]
)
def test_cap004_postgres_refuses_unexplained_renewals(
    boundary: str,
    cap004_database: ProbeDatabase,
    postgres_api: FastAPI,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = cap004_database
    seed = invoke_probe("seed", database, monkeypatch)
    assert seed["status_counts"] == {"PENDING": 25}, (
        "refusal checks must use the full real PostgreSQL command inventory"
    )
    client = proof.Cap004Client(API_URL, TOKEN, database.run_id)
    wire = proof.Cap004WireClient(API_URL, TOKEN)
    try:
        with pytest.raises(ClusterExecutorError, match="controlled claim"):
            client.claim(
                "local",
                execution_owners=[probe.OWNER],
                max_commands=5,
                lease_seconds=10,
            )
        command = client.claim(
            "local", execution_owners=[probe.OWNER], max_commands=5, lease_seconds=10
        )[0]
        if boundary != "wrong-owner":
            wire.complete(
                command,
                RemoteCommandResult(
                    lease_token=str(command.lease_token),
                    status=RemoteCommandStatus.WAITING
                    if boundary == "other-lease"
                    else RemoteCommandStatus.SUCCEEDED,
                ),
            )
        with pytest.raises(ClusterExecutorError) as rejection:
            client.renew(
                command, "foreign-owner" if boundary == "wrong-owner" else "local", 10
            )
        assert rejection.value.status_code == 409, (
            "the real PostgreSQL API must reject the invalid renewal"
        )
        if boundary == "other-lease":
            reclaimed = next(
                item
                for item in wire.claim(
                    "next-holder",
                    execution_owners=[probe.OWNER],
                    max_commands=25,
                    lease_seconds=10,
                )
                if item.command_id == command.command_id
            )
            assert reclaimed.lease_token != command.lease_token, (
                "the later result ACK must belong to a different lease"
            )
            client.complete(
                reclaimed,
                RemoteCommandResult(
                    lease_token=str(reclaimed.lease_token),
                    status=RemoteCommandStatus.SUCCEEDED,
                ),
            )
        if boundary == "wrong-owner":
            assert client.renewal_evidence() == ({}, 0), (
                "a wrong-owner rejection cannot enter terminal race reconciliation"
            )
            assert client.problems == ["unexpected renewal failure"], (
                "incorrect renewal identity must always fail the proof"
            )
        else:
            assert client.renewal_evidence() == ({}, 1), (
                "unacknowledged results and later leases cannot explain a rejection"
            )
    finally:
        closed = invoke_probe("cleanup", database, monkeypatch)
        assert closed["status_counts"] == (
            {"FAILED": 25}
            if boundary == "wrong-owner"
            else {"SUCCEEDED": 1, "FAILED": 24}
        ), "refusal cleanup must preserve real outcomes without fabricating success"
        assert all(row["lease_present"] is False for row in closed["commands"]), (
            "refusal cleanup must release every PostgreSQL command lease"
        )

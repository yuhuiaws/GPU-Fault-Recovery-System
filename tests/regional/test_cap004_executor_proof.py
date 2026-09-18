"""CAP004 exercises production scheduling/protocol with no external services."""

from __future__ import annotations

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
from gpu_fault.regional import RemoteCommandResult, RemoteCommandStatus
from gpu_fault.store import InMemoryStore, SqliteStore
from scripts.e2e.regional import capacity_acceptance_cases as cases
from scripts.e2e.regional import capacity_acceptance_executor as proof
from scripts.e2e.regional.probes import cap004_commands as commands
from tests._builders import asgi_client, build_context
from tests.regional._regional_support import registration

RUN_ID = "cap004localtest"
URL = "http://127.0.0.1:18765"
TOKEN = "t" * 32
TIME_SCALE = 0.2


@pytest.fixture
def local_api(monkeypatch: pytest.MonkeyPatch) -> Iterator[SimpleNamespace]:
    store = InMemoryStore()
    context = build_context(store=store, execution_token="e" * 32)
    context.regional_mode = True
    store.save_regional_cluster(registration(commands.CLUSTER_ID, TOKEN))
    monkeypatch.setenv("GPU_FAULT_SERVICE_ROLE", "ingress")
    monkeypatch.setenv("GPU_FAULT_TELEMETRY_SPOOL", "false")
    monkeypatch.setenv("GPU_FAULT_STORE_IO_WORKERS", "1")
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
    calls: list[tuple[str, str | None]] = []
    api_state = SimpleNamespace(store=store, calls=calls, transform=None)

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
                "the production client must only address the isolated API"
            )
            payload = json.loads(request.data)
            path = urlsplit(request.full_url).path
            calls.append((path, payload.get("executor_id")))
            status, body = portal.call(request_api, request, path)
            if api_state.transform is not None:
                status, body = api_state.transform(path, payload, status, body)
            stream = BytesIO(body)
            if status >= 400:
                raise HTTPError(request.full_url, status, "local API", {}, stream)
            return addinfourl(stream, {}, request.full_url, status)

        monkeypatch.setattr(
            "gpu_fault.cluster_executor.regional_client.urlopen", urlopen
        )
        try:
            with portal.wrap_async_context_manager(app.router.lifespan_context(app)):
                yield api_state
        finally:
            app.state.store_io.close()


def harness(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> cases.CapacityAcceptanceCases:
    value = cases.CapacityAcceptanceCases.__new__(cases.CapacityAcceptanceCases)
    value.run_dir = tmp_path
    value.run_id = RUN_ID
    value.tokens = [TOKEN]
    value.active_probe = None
    monkeypatch.setattr(
        value,
        "deploy_probe",
        lambda *_a: SimpleNamespace(
            url=URL, pod="local-probe", database=f"gpu_fault_{RUN_ID}_cap004"
        ),
    )
    return value


@pytest.mark.parametrize("late_renewal", ["none", "before-ack", "after-ack"])
def test_real_executor_api_and_ledger_complete_cap004(
    late_renewal: str,
    local_api: SimpleNamespace,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    value = harness(tmp_path, monkeypatch)
    events: list[str] = []
    executors: list[ClusterActionExecutor] = []
    before = set(threading.enumerate())
    first_id = commands.commands_for_run(RUN_ID)[0].command_id
    completing = threading.Event()
    renewal_waiting = threading.Event()
    committed = threading.Event()
    renewed = threading.Event()

    class DelayedRenewalClient(proof.Cap004Client):
        def renew(self, command: Any, executor_id: str, lease_seconds: int) -> Any:
            delayed = (
                late_renewal != "none"
                and command.command_id == first_id
                and completing.is_set()
            )
            if delayed:
                renewal_waiting.set()
                assert committed.wait(5), (
                    "the delayed renewal must follow the API's terminal commit"
                )
            try:
                return super().renew(command, executor_id, lease_seconds)
            finally:
                if delayed:
                    renewed.set()

        def complete(self, command: Any, result: RemoteCommandResult) -> Any:
            if command.command_id == first_id and late_renewal != "none":
                completing.set()
                assert renewal_waiting.wait(5), (
                    "the result must overlap an in-flight renewal without pausing earlier heartbeats"
                )
            saved = super().complete(command, result)
            if command.command_id == first_id and late_renewal == "after-ack":
                committed.set()
            return saved

    def result_ack(
        path: str, _payload: Any, status: int, body: bytes
    ) -> tuple[int, bytes]:
        if (
            late_renewal == "before-ack"
            and path.endswith(f"/{first_id}/result")
            and _payload["status"] == "SUCCEEDED"
        ):
            assert status == 200, "the terminal result must really commit first"
            committed.set()
            assert renewed.wait(5), "the renewal rejection must precede the result ACK"
        return status, body

    def executor(*args: Any, **kwargs: Any) -> ClusterActionExecutor:
        result = ClusterActionExecutor(*args, **kwargs)
        executors.append(result)
        return result

    def store_probe(_probe: Any, mode: str) -> dict[str, Any]:
        events.append(mode)
        return commands.command_snapshot(local_api.store, RUN_ID, mode)

    def run(*args: Any) -> dict[str, Any]:
        events.append("executor")
        return proof.run_executor_proof(
            *args,
            timeout_seconds=30,
            poll_seconds=0.02,
            minimum_hold_seconds=0.01,
            action_timeout_seconds=8,
        )

    def cleanup(_probe: Any) -> dict[str, Any]:
        assert value.cap004_executor_stopped is True, (
            "cleanup preceded Executor shutdown"
        )
        assert (
            commands.terminal_errors(
                commands.command_snapshot(local_api.store, RUN_ID, "inspect"), RUN_ID
            )
            == []
        ), "resource deletion preceded terminal readback"
        events.append("delete")
        return {"database_dropped": True, "residual_probe_pods": []}

    monkeypatch.setattr(proof, "ClusterActionExecutor", executor)
    monkeypatch.setattr(proof, "Cap004Client", DelayedRenewalClient)
    local_api.transform = result_ack
    monkeypatch.setattr(cases, "run_executor_proof", run)
    monkeypatch.setattr(value, "cap004_store", store_probe)
    monkeypatch.setattr(value, "cleanup_probe", cleanup)
    result = value.case_004()

    assert result["status"] == "PASS", "the production proof did not pass"
    assert result["production_executor_verified"] is True, "the simulator gap remains"
    assert result["problems"] == result["lease_progression_errors"] == [], (
        "protocol failures cannot be hidden by cleanup"
    )
    assert result["max_concurrent_commands"] == 5, (
        "the real scheduler did not hit its cap"
    )
    assert result["executor_batch_size"] == 5 and result["lease_seconds"] == 10, (
        "long actions must not queue behind 25 short-lived leases"
    )
    assert (
        result["bulk_api"]["max_commands"]
        == result["bulk_api"]["waiting_handbacks"]
        == 25
    ), "the separate 25-command API measurement must release every lease"
    long_actions = [
        row
        for row in result["action_timings"]
        if not row["cached"]
        and row["adapter_succeeded"]
        and not row["lease_guard_held"]
        and row["elapsed_seconds"] > row["original_lease_seconds"]
    ]
    assert len(long_actions) == result["long_commands"] == 24, (
        "successful actions must actually outlive their original lease; the lost lease is excluded"
    )
    assert len(result["lease_progressions"]) == 25, "all queued commands must renew"
    assert result["executor_counters"]["claimed_total"] == 26, (
        "25 unique commands plus one real lost-lease recovery must be claimed"
    )
    assert result["executor_counters"]["results_withheld_total"] == 1, (
        "the first lease holder must withhold its result"
    )
    assert result["injected_renewal_failure_status"] == 409, (
        "the API must reject the owner"
    )
    terminal_rejections = result["terminal_renewal_rejections"]
    if late_renewal != "none":
        assert terminal_rejections[first_id] == 1, (
            "the test must exercise a real API rejection after completion"
        )
    assert result["unconfirmed_renewal_rejections"] == 0, (
        "every rejected renewal must match its acknowledged completion lease"
    )
    assert result["executor_counters"]["lease_renewal_failures"] == (
        1 + sum(terminal_rejections.values())
    ), "only the injected failure and acknowledged terminal races are explained"
    assert result["claim_retry_backoff_verified"] is True, "run() retry was bypassed"
    assert len(result["ledger"]) == 25 and all(
        row["mutation_count"] == 1 for row in result["ledger"]
    ), "each unique command must mutate only the nonphysical ledger once"
    assert sorted(row["calls"] for row in result["ledger"]) == [1] * 24 + [2], (
        "the new lease holder must replay the existing entry"
    )
    assert result["bulk_api"]["competitor_samples"] > 0, (
        "no successful bulk API competitor observation"
    )
    assert result["adapter_types"] == ["NonphysicalLedgerAdapter"], (
        "no NodeAgent, Kubernetes or provider adapter may be constructed"
    )
    assert len(executors) == 1 and executors[0].fleet_registry is None, (
        "the proof must reuse one real loop with no physical registry"
    )
    assert executors[0].spare_reservation_sweep is None, (
        "no provider housekeeping is allowed"
    )
    assert events == ["seed", "executor", "inspect", "cleanup", "delete"], (
        "proof, terminal readback and cleanup ordering changed"
    )
    assert not [
        thread
        for thread in threading.enumerate()
        if thread not in before
        and thread.name.startswith(
            (RUN_ID, "gpu-fault-command", "cmd-command-", "lease-command-")
        )
    ], "Executor or probe threads survived cleanup"
    assert (
        tmp_path / "CAP-004/nonphysical-ledger.sqlite"
    ).stat().st_mode & 0o777 == 0o600, "the ledger must remain private"


@pytest.mark.parametrize("bad", ["duplicate", "missing", "foreign", "physical_owner"])
def test_production_client_refuses_an_unbound_claim(
    bad: str, local_api: SimpleNamespace
) -> None:
    commands.command_snapshot(local_api.store, RUN_ID, "seed")

    def change(path: str, _payload: Any, status: int, body: bytes) -> tuple[int, bytes]:
        if path.endswith("/claim"):
            document = json.loads(body)
            rows = document["commands"]
            if bad == "duplicate":
                rows[-1] = rows[0]
            elif bad == "missing":
                rows.pop()
            elif bad == "foreign":
                rows[0]["cluster_id"] = "foreign"
            else:
                rows[0]["step"]["execution_owner"] = "gpu-fault-node-agent"
            body = json.dumps(document).encode()
        return status, body

    local_api.transform = change
    client = proof.Cap004Client(URL, TOKEN, RUN_ID)
    kwargs = dict(execution_owners=[commands.OWNER], max_commands=5, lease_seconds=10)
    with pytest.raises(ClusterExecutorError, match="controlled claim"):
        client.claim("local", **kwargs)
    with pytest.raises((proof.Cap004Error, ValueError), match="identity|command set"):
        client.claim("local", **kwargs)
    assert client.first_claimed.is_set() is False, (
        "invalid claims cannot reach the adapter"
    )


def test_active_lease_rejection_cannot_be_classified_as_terminal(
    local_api: SimpleNamespace,
) -> None:
    commands.command_snapshot(local_api.store, RUN_ID, "seed")
    client = proof.Cap004Client(URL, TOKEN, RUN_ID)
    kwargs = dict(execution_owners=[commands.OWNER], max_commands=5, lease_seconds=10)
    with pytest.raises(ClusterExecutorError, match="controlled claim"):
        client.claim("local", **kwargs)
    claimed = client.claim("local", **kwargs)
    with pytest.raises(ClusterExecutorError):
        client.renew(claimed[0], "foreign-owner", 10)
    assert client.renewal_evidence() == ({}, 0), (
        "a wrong-owner rejection must not enter terminal race reconciliation"
    )
    assert client.problems == ["unexpected renewal failure"], (
        "incorrect renewal identity must always fail the proof"
    )


@pytest.mark.parametrize("boundary", ["unacknowledged-result", "other-lease"])
def test_terminal_renewal_requires_the_same_acknowledged_lease(
    boundary: str, local_api: SimpleNamespace
) -> None:
    commands.command_snapshot(local_api.store, RUN_ID, "seed")
    client = proof.Cap004Client(URL, TOKEN, RUN_ID)
    kwargs = dict(execution_owners=[commands.OWNER], max_commands=5, lease_seconds=10)
    with pytest.raises(ClusterExecutorError, match="controlled claim"):
        client.claim("local", **kwargs)
    command = client.claim("local", **kwargs)[0]
    wire = proof.Cap004WireClient(URL, TOKEN)
    wire.complete(
        command,
        RemoteCommandResult(
            lease_token=str(command.lease_token),
            status=RemoteCommandStatus.WAITING
            if boundary == "other-lease"
            else RemoteCommandStatus.SUCCEEDED,
        ),
    )
    with pytest.raises(ClusterExecutorError):
        client.renew(command, "local", 10)
    if boundary == "other-lease":
        reclaimed = next(
            item
            for item in wire.claim(
                "next-holder",
                execution_owners=[commands.OWNER],
                max_commands=25,
                lease_seconds=10,
            )
            if item.command_id == command.command_id
        )
        client.complete(
            reclaimed,
            RemoteCommandResult(
                lease_token=str(reclaimed.lease_token),
                status=RemoteCommandStatus.SUCCEEDED,
            ),
        )
    assert client.renewal_evidence() == ({}, 1), (
        "unacknowledged results and later lease holders cannot explain an old rejection"
    )


def test_renewal_requires_a_strictly_advancing_server_lease(
    local_api: SimpleNamespace,
) -> None:
    commands.command_snapshot(local_api.store, RUN_ID, "seed")
    client = proof.Cap004Client(URL, TOKEN, RUN_ID)
    kwargs = dict(execution_owners=[commands.OWNER], max_commands=5, lease_seconds=10)
    with pytest.raises(ClusterExecutorError):
        client.claim("local", **kwargs)
    claimed = client.claim("local", **kwargs)

    def unchanged(
        path: str, _payload: Any, status: int, body: bytes
    ) -> tuple[int, bytes]:
        if path.endswith("/renew"):
            document = json.loads(body)
            document["lease_expires_at"] = claimed[0].lease_expires_at.isoformat()
            body = json.dumps(document).encode()
        return status, body

    local_api.transform = unchanged
    with pytest.raises(proof.Cap004Error, match="progression"):
        client.renew(claimed[0], "local", 10)
    assert client.renewed_once(claimed[0].command_id) is False, (
        "HTTP 200 alone does not prove an advancing lease"
    )


def test_competitor_errors_cannot_prove_no_duplicate_claims(
    local_api: SimpleNamespace, tmp_path: Path
) -> None:
    commands.command_snapshot(local_api.store, RUN_ID, "seed")

    def unavailable(
        _path: str, payload: Any, status: int, body: bytes
    ) -> tuple[int, bytes]:
        if payload.get("executor_id") == f"{RUN_ID}-competitor":
            return 503, b'{"detail":"controlled competitor outage"}'
        return status, body

    local_api.transform = unavailable
    with pytest.raises(ClusterExecutorError, match="503"):
        proof.run_executor_proof(URL, TOKEN, RUN_ID, tmp_path)
    assert not (tmp_path / "nonphysical-ledger.sqlite").exists(), (
        "a failed bulk API observation must prevent all adapter execution"
    )


def test_claim_payload_budget_rejects_before_execution(
    local_api: SimpleNamespace,
) -> None:
    commands.command_snapshot(local_api.store, RUN_ID, "seed")

    def oversized(
        _path: str, _payload: Any, status: int, body: bytes
    ) -> tuple[int, bytes]:
        return status, body + b" " * proof.PAYLOAD_BUDGET_BYTES

    local_api.transform = oversized
    client = proof.Cap004Client(URL, TOKEN, RUN_ID)
    stops: list[str] = []
    client.stop_executor = stops.append
    kwargs = dict(execution_owners=[commands.OWNER], max_commands=5, lease_seconds=10)
    with pytest.raises(ClusterExecutorError):
        client.claim("local", **kwargs)
    with pytest.raises(proof.Cap004Error, match="payload budget"):
        client.claim("local", **kwargs)
    assert stops == ["CAP004 payload budget"], (
        "oversized claims must stop the real loop"
    )
    assert client.first_claimed.is_set() is False, (
        "oversized commands reached execution"
    )


def test_deadline_stops_real_workers_and_preserves_failed_outcomes(
    local_api: SimpleNamespace, tmp_path: Path
) -> None:
    commands.command_snapshot(local_api.store, RUN_ID, "seed")
    result = proof.run_executor_proof(
        URL,
        TOKEN,
        RUN_ID,
        tmp_path,
        timeout_seconds=0.15,
        poll_seconds=0.01,
        minimum_hold_seconds=0.01,
        action_timeout_seconds=1,
    )
    assert "CAP004 overall deadline exceeded" in result["problems"], (
        "a truncated run must not be accepted as capacity evidence"
    )
    assert result["background_threads_stopped"] is True, "deadline left active workers"
    closed = commands.command_snapshot(local_api.store, RUN_ID, "cleanup")
    assert all(row["lease_present"] is False for row in closed["commands"]), (
        "deadline cleanup must release every remaining lease"
    )
    assert closed["status_counts"].get("SUCCEEDED", 0) < 25, (
        "cleanup cannot manufacture successful actions after interruption"
    )


def test_renewal_alone_does_not_prove_an_action_outlived_its_lease(
    local_api: SimpleNamespace, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    commands.command_snapshot(local_api.store, RUN_ID, "seed")
    original = proof.NonphysicalLedgerAdapter.wait_for_lease_evidence

    def only_one_renewal(
        adapter: proof.NonphysicalLedgerAdapter, command_id: str, started: float
    ) -> None:
        if command_id == adapter.client.failure_command_id:
            original(adapter, command_id, started)
            return
        for _attempt in range(500):
            if adapter.client.renewed_once(command_id):
                return
            adapter.stop.wait(0.002)
        raise AssertionError("the local renewal did not arrive")

    monkeypatch.setattr(
        proof.NonphysicalLedgerAdapter, "wait_for_lease_evidence", only_one_renewal
    )
    result = proof.run_executor_proof(
        URL,
        TOKEN,
        RUN_ID,
        tmp_path,
        timeout_seconds=10,
        poll_seconds=0.02,
        minimum_hold_seconds=0.01,
        action_timeout_seconds=3,
    )
    assert result["long_commands"] < 5, "renewal alone must not count as a long action"
    assert "successful_actions_outlive_initial_lease" in result["problems"], (
        "lease renewal and a lost-lease replay cannot substitute for five long successes"
    )


def test_interrupted_join_does_not_claim_verified_shutdown(
    local_api: SimpleNamespace, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    commands.command_snapshot(local_api.store, RUN_ID, "seed")
    join = threading.Thread.join

    def interrupted(thread: threading.Thread, timeout: float | None = None) -> None:
        join(thread, timeout)
        if thread.name == f"{RUN_ID}-deadline":
            raise KeyboardInterrupt

    monkeypatch.setattr(threading.Thread, "join", interrupted)
    with pytest.raises(proof.Cap004ThreadsRunning, match="shutdown"):
        proof.run_executor_proof(
            URL,
            TOKEN,
            RUN_ID,
            tmp_path,
            timeout_seconds=0.15,
            poll_seconds=0.01,
            minimum_hold_seconds=0.01,
            action_timeout_seconds=1,
        )
    assert not any(
        thread.name.startswith(RUN_ID) for thread in threading.enumerate()
    ), "the test must not leave a deadline or competitor thread running"


@pytest.mark.parametrize(
    "mode", ["seed_ack_lost", "executor_error", "live_threads", "cleanup_error"]
)
def test_case_failure_never_keeps_pass_or_deletes_under_live_threads(
    mode: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    value = harness(tmp_path, monkeypatch)
    events: list[str] = []
    case_dir = tmp_path / "CAP-004"
    case_dir.mkdir()
    (case_dir / "summary.json").write_text('{"status":"PASS"}', encoding="utf-8")

    def store_probe(_probe: Any, operation: str) -> dict[str, Any]:
        events.append(operation)
        if operation == "seed" and mode == "seed_ack_lost":
            raise OSError("seed acknowledgement lost")
        return {"commands": []}

    def run(*_args: Any) -> dict[str, Any]:
        events.append("executor")
        if mode == "live_threads":
            raise proof.Cap004ThreadsRunning("still running")
        raise OSError("executor transport failed")

    def cleanup(_probe: Any) -> dict[str, Any]:
        events.append("delete")
        if mode == "cleanup_error":
            raise OSError("database deletion failed")
        return {"database_dropped": True}

    monkeypatch.setattr(value, "cap004_store", store_probe)
    monkeypatch.setattr(cases, "run_executor_proof", run)
    monkeypatch.setattr(value, "cleanup_probe", cleanup)
    with pytest.raises((OSError, proof.Cap004ThreadsRunning)):
        value.case_004()
    summary = json.loads((case_dir / "summary.json").read_text())
    assert summary["status"] == "FAIL", "old PASS must be invalidated before work"
    assert summary["production_executor_verified"] is False, (
        "failed work cannot prove execution"
    )
    if mode == "live_threads":
        assert events == ["seed", "executor"], "no cleanup can race a live Executor"
        assert value.cleanup_all(), (
            "outer cleanup must retain resources and report failure"
        )
        with pytest.raises(cases.CapError, match="shutdown"):
            cases.CapHarnessBase.cleanup_probe(value, object())
    else:
        assert events[-2:] == ["cleanup", "delete"], (
            "failure still requires owned cleanup"
        )


def test_seed_and_cleanup_use_real_store_transitions(tmp_path: Path) -> None:
    store = SqliteStore(str(tmp_path / "cleanup.sqlite"))
    try:
        seeded = commands.command_snapshot(store, RUN_ID, "seed")
        assert len(seeded["commands"]) == 25, "seed inventory is incomplete"
        with pytest.raises(ValueError, match="empty isolated"):
            commands.command_snapshot(store, RUN_ID, "seed")
        claimed = store.claim_remote_commands(
            commands.CLUSTER_ID,
            "local",
            limit=5,
            lease_seconds=240,
            execution_owners={commands.OWNER},
        )
        store.complete_remote_command(
            commands.CLUSTER_ID,
            claimed[0].command_id,
            RemoteCommandResult(
                lease_token=claimed[0].lease_token,
                status=RemoteCommandStatus.SUCCEEDED,
                details={"nonphysical": True, "ledger_count": 1},
            ),
        )
        closed = commands.command_snapshot(store, RUN_ID, "cleanup")
        assert closed["status_counts"] == {"SUCCEEDED": 1, "FAILED": 24}, (
            "cleanup must not fabricate successful execution"
        )
        assert all(row["lease_present"] is False for row in closed["commands"]), (
            "cleanup left an active command lease"
        )
        assert commands.command_snapshot(store, RUN_ID, "cleanup") == closed, (
            "cleanup must be repeatable after acknowledgement loss"
        )
    finally:
        store.close()


@pytest.mark.parametrize(
    "field", ["commands", "lease_present", "status", "ledger_count", "nonphysical"]
)
def test_terminal_readback_is_not_inferred_from_an_empty_claim(field: str) -> None:
    snapshot = {
        "run_id": RUN_ID,
        "cluster_id": commands.CLUSTER_ID,
        "node_agent_records": 0,
        "commands": [
            {
                "command_id": item.command_id,
                "idempotency_key": item.idempotency_key,
                "status": "SUCCEEDED",
                "lease_present": False,
                "nonphysical": True,
                "ledger_count": 1,
            }
            for item in commands.commands_for_run(RUN_ID)
        ],
    }
    assert commands.terminal_errors(snapshot, RUN_ID) == [], (
        "complete control must pass"
    )
    if field == "commands":
        snapshot["commands"].pop()
    else:
        snapshot["commands"][0][field] = None
    assert commands.terminal_errors(snapshot, RUN_ID), (
        "incomplete closure must fail closed"
    )


@pytest.mark.parametrize(
    "url",
    [
        "https://production.invalid",
        "http://localhost:18080",
        "http://127.0.0.1",
        "http://127.0.0.1:18080/production",
        "http://user@127.0.0.1:18080",
    ],
)
def test_proof_cannot_address_an_unbound_control_plane(url: str) -> None:
    with pytest.raises(proof.Cap004Error, match="loopback"):
        proof.Cap004Client(url, TOKEN, RUN_ID)

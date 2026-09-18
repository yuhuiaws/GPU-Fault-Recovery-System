"""Probe execution, transport and watchdog behavior without host effects."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts.e2e.regional.probes import (
    net002_executor,
    net003_executor,
    net006_executor,
)
from tests.regional._cov95_collect_net import (
    Clock,
    FakeSocket,
    StopLoop,
    isolate_paths,
    local_socket_module,
    no_external_effects,  # noqa: F401
    stop_sleep,
)

PROBES = (net002_executor, net003_executor, net006_executor)


@pytest.fixture(params=PROBES, ids=("net002", "net003", "net006"))
def probe(request: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Any:
    module = request.param
    isolate_paths(monkeypatch, module, tmp_path)
    module.STATE.mkdir(parents=True)
    clock = Clock()
    monkeypatch.setattr(module, "time", clock)
    return module, clock


@pytest.mark.parametrize("concurrent_commit", [False, True])
def test_action_ledger_records_once_and_replays_cached(
    probe: Any, monkeypatch: pytest.MonkeyPatch, concurrent_commit: bool
) -> None:
    module, clock = probe
    if module is net003_executor:
        adapter = module.LedgerAdapter("notification-a")
    elif module is net006_executor:
        adapter = module.HoldingLedgerAdapter()
        monkeypatch.setattr(
            module, "lease_hold_reason", lambda: "lease expired locally"
        )
    else:
        adapter = module.LedgerAdapter()
    assert adapter.supports(SimpleNamespace(execution_owner=module.OWNER)), (
        "the probe adapter must claim its configured execution owner"
    )
    assert not adapter.supports(SimpleNamespace(execution_owner="another-owner")), (
        "foreign execution owners must not be claimed"
    )
    context = SimpleNamespace(idempotency_key="workflow/0/FREEZE_EVIDENCE")
    if hasattr(module, "BLOCK"):
        module.BLOCK.touch()

    def commit_while_sleeping(_seconds: float) -> None:
        if concurrent_commit:
            module.LEDGER.write_text(
                json.dumps({"physical_count": 1, "keys": [context.idempotency_key]})
            )

    clock.on_sleep = commit_while_sleeping
    first = adapter.execute(context)
    second = adapter.execute(context)
    assert first.details["cached"] is concurrent_commit
    assert second.details["cached"] is True
    assert second.details["physical_count"] == 1
    assert json.loads(module.LEDGER.read_text()) == {
        "keys": [context.idempotency_key],
        "physical_count": 1,
    }
    assert module.ACTION_STARTED.read_text() == context.idempotency_key
    if module is net006_executor:
        observed = json.loads(module.LEASE_GUARD_OBSERVED.read_text())
        assert observed["reason"] == "lease expired locally"
        assert json.loads(module.ACTION_RETURNED.read_text())["cached"] is True


@pytest.mark.parametrize("module", [net002_executor, net006_executor])
@pytest.mark.parametrize("arm", [False, True])
def test_action_waits_for_block_and_refuses_unarmed_execution(
    module: Any, arm: bool, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    isolate_paths(monkeypatch, module, tmp_path)
    module.STATE.mkdir(parents=True)
    clock = Clock()
    monkeypatch.setattr(module, "time", clock)
    if arm:
        clock.on_sleep = lambda _seconds: module.BLOCK.touch()
    else:
        clock.on_sleep = lambda _seconds: setattr(clock, "now", clock.now + 31)
    if module is net006_executor:
        monkeypatch.setattr(module, "lease_hold_reason", lambda: None)
        adapter = module.HoldingLedgerAdapter()
    else:
        adapter = module.LedgerAdapter()
    context = SimpleNamespace(idempotency_key="command-a")
    if not arm:
        with pytest.raises(RuntimeError, match="not armed"):
            adapter.execute(context)
        assert not module.LEDGER.exists(), "unarmed action must leave no ledger commit"
    else:
        result = adapter.execute(context)
        assert result.details["physical_count"] == 1
        assert (
            json.loads(module.ACTION_GATE_OBSERVED.read_text())["idempotency_key"]
            == "command-a"
        )


@pytest.mark.parametrize("mode", ["relay", "error", "blocked", "drop"])
def test_proxy_listener_relays_or_refuses_without_real_sockets(
    probe: Any, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    module, clock = probe
    client = FakeSocket((b"request", b""))
    upstream = FakeSocket((b"response",))
    listener = FakeSocket()
    listener.accepted.append(client)
    if mode == "blocked":
        if module is net003_executor:
            module.DROP_NEXT.touch()
        else:
            module.BLOCK.touch()
            clock.on_sleep = lambda _seconds: module.BLOCK.unlink(missing_ok=True)
    if mode == "drop" and module is net003_executor:
        module.DROP_NEXT.touch()
    connections: list[Any] = []

    def connect(address: Any, timeout: float) -> FakeSocket:
        connections.append((address, timeout))
        if mode == "error":
            raise OSError("offline")
        return upstream

    monkeypatch.setattr(
        module,
        "socket",
        local_socket_module(socket=lambda *_args: listener, create_connection=connect),
    )
    reads = iter([[], [client], [upstream], [client]])
    monkeypatch.setattr(
        module, "select", SimpleNamespace(select=lambda *args: (next(reads), [], []))
    )
    dropped: list[Any] = []
    if module is net003_executor:

        def lose_response(left: Any, right: Any) -> dict[str, Any]:
            dropped.append((left, right))
            return {"connection_reset": True, "upstream_response_bytes": 8}

        monkeypatch.setattr(module, "relay_losing_response", lose_response)

    class InlineThread:
        def __init__(self, *, target: Any, args: tuple[Any, ...], daemon: bool) -> None:
            self.target, self.args = target, args
            assert daemon is True

        def start(self) -> None:
            self.target(*self.args)

    monkeypatch.setattr(module, "Thread", InlineThread)
    proxy_class = (
        module.RefusingProxy if module is net006_executor else module.GateProxy
    )
    proxy = proxy_class("192.0.2.1", 18443)
    with pytest.raises(StopLoop):
        proxy.run()
    assert listener.bound == ("127.0.0.1", 18443)
    assert listener.backlog == 32
    assert client.closed, "every accepted client must close even on transport failure"
    assert listener.closed, "listener ownership must release on worker exit"
    if mode == "blocked" and module is net006_executor:
        assert proxy.refused_total == 1
        assert connections == []
    elif mode in {"blocked", "drop"} and module is net003_executor:
        assert dropped == [(client, upstream)]
        assert json.loads(module.DROP_OBSERVED.read_text())["connection_reset"] is True
        assert not module.DROP_NEXT.exists(), "a drop marker must be consumed once"
    elif mode != "error":
        assert client.sent == [b"response"]
        assert upstream.sent == [b"request"]
    else:
        assert client.sent == [], "connection errors cannot manufacture response bytes"


@pytest.mark.parametrize("blocked_at", ["connected", "selected", "received"])
def test_net006_existing_tunnel_stops_forwarding_when_block_is_armed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, blocked_at: str
) -> None:
    module = net006_executor
    isolate_paths(monkeypatch, module, tmp_path)
    module.STATE.mkdir(parents=True)

    class Client(FakeSocket):
        def recv(self, size: int) -> bytes:
            data = super().recv(size)
            if blocked_at == "received":
                module.BLOCK.touch()
            return data

    client = Client((b"private-request", b""))
    upstream = FakeSocket((b"response",))
    listener = FakeSocket()
    listener.accepted.append(client)

    def connect(address: Any, timeout: float) -> FakeSocket:
        if blocked_at == "connected":
            module.BLOCK.touch()
        return upstream

    def readable(*args: Any) -> tuple[list[Any], list[Any], list[Any]]:
        if blocked_at == "selected":
            module.BLOCK.touch()
        return [client], [], []

    class InlineThread:
        def __init__(self, *, target: Any, args: tuple[Any, ...], daemon: bool) -> None:
            self.target, self.args = target, args

        def start(self) -> None:
            self.target(*self.args)

    monkeypatch.setattr(
        module,
        "socket",
        local_socket_module(socket=lambda *_args: listener, create_connection=connect),
    )
    monkeypatch.setattr(module, "select", SimpleNamespace(select=readable))
    monkeypatch.setattr(module, "Thread", InlineThread)
    with pytest.raises(StopLoop):
        module.RefusingProxy("192.0.2.1", 18443).run()
    assert upstream.sent == [], "blocked buffered data must not reach the control plane"
    assert client.sent == []
    assert client.closed and upstream.closed and listener.closed


@pytest.mark.parametrize("age", [None, 1, 300])
def test_watchdog_removes_only_expired_marker(
    probe: Any, monkeypatch: pytest.MonkeyPatch, age: int | None
) -> None:
    module, clock = probe
    marker = module.DROP_NEXT if module is net003_executor else module.BLOCK
    if age is not None:
        marker.touch()
        os.utime(marker, (clock.now - age, clock.now - age))
    clock.on_sleep = stop_sleep
    worker = (
        module.rollback_stale_drop
        if module is net003_executor
        else module.rollback_stale_block
    )
    with pytest.raises(StopLoop):
        worker()
    if age == 300:
        assert not marker.exists(), "expired marker must be removed"
        proof = json.loads(module.ROLLBACK_STATE.read_text())
        assert proof["automatic"] is True
        assert 300 in proof.values(), "watchdog must retain the observed marker age"
    else:
        assert marker.exists() is (age is not None)
        assert not module.ROLLBACK_STATE.exists(), "no false rollback proof"


@pytest.mark.parametrize(
    "last_claim", [None, datetime(2026, 9, 1, tzinfo=timezone.utc)]
)
def test_executor_metrics_are_atomically_recorded(probe: Any, last_claim: Any) -> None:
    module, clock = probe
    clock.on_sleep = stop_sleep
    executor = SimpleNamespace(
        claimed_total=2,
        reported_failures=0,
        unexpected_failures=0,
        lease_renewal_failures=3,
        transport_retries_total=1,
        last_successful_claim_at=last_claim,
        metrics_snapshot=lambda: {"claimed_total": 2, "results_withheld_total": 1},
    )
    with pytest.raises(StopLoop):
        if module is net006_executor:
            module.record_executor_state(executor, SimpleNamespace(refused_total=3))
        else:
            module.record_executor_state(executor)
    state = json.loads(module.EXECUTOR_STATE.read_text())
    assert state["claimed_total"] == 2
    assert not module.EXECUTOR_STATE.with_suffix(".tmp").exists(), (
        "atomic metrics publication must consume its temporary file"
    )
    if module is net006_executor:
        assert state["proxy_refused_total"] == 3
    else:
        assert state["last_successful_claim_at"] == (
            last_claim.isoformat() if last_claim else None
        )


@pytest.mark.parametrize("valid_url", [True, False])
def test_main_uses_pinned_client_and_local_proxy(
    probe: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, valid_url: bool
) -> None:
    module, _clock = probe
    monkeypatch.setenv(
        "CONTROL_PLANE_URL",
        "https://control.invalid/" if valid_url else "/missing-host",
    )
    monkeypatch.setenv("EXECUTOR_ARTIFACT_SHA256", "a" * 64)
    monkeypatch.setenv("EXECUTOR_COMPATIBILITY_DIGEST", "b" * 64)
    monkeypatch.setenv("NOTIFICATION_ID", "notification-a")
    tokens = tmp_path / "tokens/clusters.json"
    tokens.parent.mkdir()
    tokens.write_text(
        json.dumps([{"cluster_id": "cluster-a", "token": "example-only"}])
    )
    calls: list[Any] = []

    def resolve(host: str, port: int, *args: Any, **kwargs: Any) -> list[Any]:
        calls.append(("resolve", host, port))
        return [(None, None, None, None, ("192.0.2.1", port))]

    network = local_socket_module(getaddrinfo=resolve)
    monkeypatch.setattr(module, "socket", network)

    class Thread:
        def __init__(self, **kwargs: Any) -> None:
            calls.append(("thread", kwargs))

        def start(self) -> None:
            calls.append(("thread-start",))

    def client(*args: Any, **kwargs: Any) -> object:
        calls.append(("client", args, kwargs))
        return object()

    class Executor:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            calls.append(("executor", args, kwargs))

        def run(self) -> None:
            calls.append(("run",))
            network.getaddrinfo("control.invalid", 18443)
            network.getaddrinfo("other.invalid", 443)

    monkeypatch.setattr(module, "Thread", Thread)
    monkeypatch.setattr(module, "ClusterActionExecutor", Executor)
    client_name = {
        net002_executor: "GatedRegionalExecutorClient",
        net003_executor: "InterruptingRegionalExecutorClient",
        net006_executor: "RegionalExecutorClient",
    }[module]
    monkeypatch.setattr(module, client_name, client)
    if not valid_url:
        with pytest.raises(RuntimeError, match="no hostname"):
            module.main()
        assert calls == [], "invalid URL must fail before DNS, client or thread startup"
        return
    module.main()
    ready = json.loads(module.READY.read_text())
    assert ready["cluster_id"] == "cluster-a"
    assert ready["proxy_port"] == 18443
    assert ("resolve", "127.0.0.1", 18443) in calls
    assert ("resolve", "other.invalid", 443) in calls
    client_call = next(call for call in calls if call[0] == "client")
    assert client_call[1][:2] == ("https://control.invalid:18443", "cluster-a")
    assert client_call[2]["ca_file"] == "/tls/ca.crt"
    assert client_call[2]["executor_artifact_sha256"] == "a" * 64
    executor_call = next(call for call in calls if call[0] == "executor")
    assert executor_call[2]["batch_size"] == 1
    assert executor_call[2]["max_concurrent_commands"] == 1
    assert executor_call[2]["lease_seconds"] == module.LEASE_SECONDS
    assert len([call for call in calls if call[0] == "thread-start"]) == 3


@pytest.mark.parametrize("blocked", [False, True])
def test_gated_completion_only_posts_after_release(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, blocked: bool
) -> None:
    module = net002_executor
    isolate_paths(monkeypatch, module, tmp_path)
    module.STATE.mkdir(parents=True)
    if blocked:
        module.BLOCK.touch()
    clock = Clock()
    clock.on_sleep = lambda _seconds: module.BLOCK.unlink()
    monkeypatch.setattr(module, "time", clock)
    monkeypatch.setattr(
        module.RegionalExecutorClient, "__init__", lambda self, *args, **kwargs: None
    )
    posted: list[Any] = []

    def complete(self: Any, command: Any, result: Any) -> str:
        assert not module.BLOCK.exists(), "result post must not cross an active gate"
        posted.append((command, result))
        return "committed"

    monkeypatch.setattr(module.RegionalExecutorClient, "complete", complete)
    client = module.GatedRegionalExecutorClient()
    command = SimpleNamespace(command_id="command-a")
    assert client.complete(command, "result") == "committed"
    assert posted == [(command, "result")]
    assert module.RESULT_SUBMIT_WAITING.exists() is blocked
    assert module.RESULT_SUBMIT_RELEASED.exists() is blocked


@pytest.mark.parametrize("end", ["quiet", "max-wait", "no-request", "upstream-close"])
def test_response_loss_relay_has_bounded_exit_and_never_leaks_response(
    monkeypatch: pytest.MonkeyPatch, end: str
) -> None:
    module = net003_executor
    request = b"\x17\x03\x03\x00\x01x"
    client = FakeSocket((request,))
    upstream = FakeSocket((b"response", b""))
    ticks = iter(
        [0, 0.1, 0.2, 0.3, 1, 1.1]
        if end != "max-wait"
        else [0, 0.05, 0.1, 0.15, 0.21, 0.22]
    )
    events = iter(
        [[client], [upstream], [], []]
        if end not in {"no-request", "upstream-close"}
        else (
            [[], [], [], []]
            if end == "no-request"
            else [[client], [upstream], [upstream]]
        )
    )
    monkeypatch.setattr(
        module, "select", SimpleNamespace(select=lambda *args: (next(events), [], []))
    )
    observed = module.relay_losing_response(
        client,
        upstream,
        quiet_seconds=0.5 if end != "max-wait" else 10,
        max_wait_seconds=0.1 if end in {"max-wait", "no-request"} else 10,
        clock=lambda: next(ticks),
    )
    assert client.closed, "relay must reset and close the client at every bounded exit"
    assert client.sent == [], "encrypted response bytes must never reach the client"
    assert observed["request_forwarded"] is (end != "no-request")
    assert observed["upstream_response_bytes"] == (0 if end == "no-request" else 8)
    assert observed["upstream_closed_first"] is (end == "upstream-close")


@pytest.mark.parametrize("first", ["accepted", "rejected", "transport"])
@pytest.mark.parametrize("same_command", [False, True])
def test_lost_result_response_is_recorded_but_retry_belongs_to_executor(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, first: str, same_command: bool
) -> None:
    module = net003_executor
    isolate_paths(monkeypatch, module, tmp_path)
    module.STATE.mkdir(parents=True)
    monkeypatch.setattr(module, "time", Clock())
    monkeypatch.setattr(
        module.RegionalExecutorClient, "__init__", lambda self, *a, **k: None
    )
    calls: list[Any] = []
    terminal = SimpleNamespace(
        command_id="command-a",
        status=SimpleNamespace(value="SUCCEEDED"),
        status_source="executor",
        updated_at=datetime(2026, 9, 1, tzinfo=timezone.utc),
    )

    def complete(self: Any, command: Any, result: Any) -> Any:
        calls.append(command.command_id)
        if len(calls) == 1:
            assert module.DROP_NEXT.exists(), "arm response loss before the first POST"
            assert pool_closes == [True], (
                "the one-shot proxy must receive a fresh result connection"
            )
            if first == "rejected":
                raise module.ClusterExecutorError("rejected (409)", status_code=409)
            if first == "transport":
                raise ConnectionResetError("lost response")
        return terminal

    pool_closes: list[bool] = []
    monkeypatch.setattr(
        module,
        "CONNECTION_POOL",
        SimpleNamespace(close=lambda: pool_closes.append(True)),
    )
    monkeypatch.setattr(module.RegionalExecutorClient, "complete", complete)
    client = module.InterruptingRegionalExecutorClient()
    command = SimpleNamespace(command_id="command-a")
    if first == "accepted":
        assert client.complete(command, None) is terminal
    else:
        error = (
            module.ClusterExecutorError if first == "rejected" else ConnectionResetError
        )
        with pytest.raises(error):
            client.complete(command, None)
    assert calls == ["command-a"], "probe must not implement a retry loop"
    second = command if same_command else SimpleNamespace(command_id="command-b")
    assert client.complete(second, None) is terminal
    assert pool_closes == [True], "the product's retry must retain its normal pool"
    assert module.RESULT_REPLAYS.exists() is (first == "transport" and same_command)
    if first == "transport":
        interrupted = json.loads(module.RESULT_INTERRUPTED.read_text())
        assert interrupted["first_post_succeeded"] is False
        assert interrupted["exception"] == "ConnectionResetError"
        if same_command:
            replay = json.loads(module.RESULT_REPLAYS.read_text())
            assert replay["count"] == 1
            assert replay["retry_owner"] == "product-executor"
            assert (
                replay["responses"][0]["updated_at"] == terminal.updated_at.isoformat()
            )
    elif first == "accepted":
        assert (
            json.loads(module.RESULT_INTERRUPTED.read_text())["first_post_succeeded"]
            is True
        )
    else:
        assert not module.RESULT_INTERRUPTED.exists(), (
            "received 409 is not response loss"
        )


def test_tls_scanner_and_handshake_forwarding_before_client_close(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = net003_executor
    scanner = module.TlsRecordScanner()
    record = b"\x17\x03\x03\x00\x03abc"
    scanner.feed(record[:2])
    scanner.feed(record[2:6])
    scanner.feed(record[6:])
    scanner.feed(b"\x16\x03\x03\x00\x00" + record)
    assert scanner.application_records == 2
    client = FakeSocket((b"",))
    upstream = FakeSocket((b"handshake", b"more-handshake"))
    reads = iter([[upstream], [upstream], [client]])
    monkeypatch.setattr(
        module, "select", SimpleNamespace(select=lambda *args: (next(reads), [], []))
    )
    result = module.relay_losing_response(client, upstream, clock=lambda: 1.0)
    assert client.sent == [b"handshake", b"more-handshake"]
    assert result["client_closed_first"] is True
    assert result["request_forwarded"] is False

from __future__ import annotations

import io
import json
import os
import socket
import struct
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace, TracebackType
from typing import Any, Self

import pytest

from scripts.e2e.regional.late_ownership_barrier import BoundaryDenied, process_identity
from scripts.e2e.regional.late_ownership_contract import AcceptanceScope, NodeIdentity
from scripts.e2e.regional.probes import late_ownership_node_probe as probe
from tests.regional import test_late_ownership_node_probe as shared
from tests.regional._late_ownership_support import scope
from tests.regional.test_late_ownership_node_probe import WireReceipt

daemon = shared.daemon
stale_mailbox = shared.stale_mailbox


@dataclass
class RpcSocket:
    peer_pid: int = field(default_factory=os.getpid)
    peer_uid: int = field(default_factory=os.geteuid)
    flags: int = 0
    raw: bytes | None = None
    changes: dict[str, Any] = field(default_factory=dict)
    failure: str | None = None
    sent: list[bytes] = field(default_factory=list)
    connected: str | None = None
    closed: bool = False
    timeout: int | None = None

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.closed = True

    def settimeout(self, timeout: int) -> None:
        self.timeout = timeout

    def connect(self, path: str) -> None:
        self.connected = path
        if self.failure == "connect":
            raise ConnectionRefusedError("local fake: unavailable")

    def getsockopt(self, level: int, option: int, length: int) -> bytes:
        assert (level, option, length) == (
            socket.SOL_SOCKET,
            socket.SO_PEERCRED,
            struct.calcsize("3i"),
        ), "RPC must obtain kernel peer credentials"
        return struct.pack("3i", self.peer_pid, self.peer_uid, os.getegid())

    def sendall(self, message: bytes) -> None:
        self.sent.append(message)
        if self.failure == "send":
            raise BrokenPipeError("local fake: peer closed")

    def recvmsg(self, maximum: int) -> tuple[bytes, list[Any], int, None]:
        assert maximum == probe.MAX_MESSAGE_BYTES, "RPC reads must have a packet bound"
        if self.failure == "receive":
            raise TimeoutError("local fake: reply lost")
        if self.raw is not None:
            return self.raw, [], self.flags, None
        request = json.loads(self.sent[-1])
        value = {
            "scope_sha256": request["scope_sha256"],
            "request_id": request["request_id"],
            "action": request["action"],
            "phase": "ATTACHED",
            **self.changes,
        }
        return json.dumps(value).encode(), [], self.flags, None


@pytest.fixture
def rpc(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[AcceptanceScope, NodeIdentity, RpcSocket, Path]:
    binding = scope()
    node = binding.nodes[0].model_copy(
        update={"boot_id": process_identity(os.getpid()).boot_id}
    )
    binding = binding.model_copy(update={"nodes": (node, binding.nodes[1])})
    directory = tmp_path / "mailbox" / "node"
    directory.mkdir(mode=0o700, parents=True)
    directory.parent.chmod(0o700)
    endpoint = RpcSocket()
    monkeypatch.setattr(probe, "mailbox_path", lambda *args: directory)
    monkeypatch.setattr(
        probe,
        "socket",
        SimpleNamespace(
            AF_UNIX=socket.AF_UNIX,
            SOCK_SEQPACKET=socket.SOCK_SEQPACKET,
            SOL_SOCKET=socket.SOL_SOCKET,
            SO_PEERCRED=socket.SO_PEERCRED,
            MSG_TRUNC=socket.MSG_TRUNC,
            socket=lambda *_: endpoint,
        ),
    )
    return binding, node, endpoint, directory


def test_rpc_uses_private_sequenced_packet_and_actual_peer_incarnation(
    rpc: Any,
) -> None:
    binding, node, endpoint, directory = rpc
    response = probe.daemon_request(binding, node, "status", {})
    assert response["peer"] == process_identity(os.getpid()).model_dump(mode="json"), (
        "peer identity must come from the kernel and current process incarnation"
    )
    assert endpoint.connected == str(directory / "control.sock"), (
        "RPC scope must be exact"
    )
    assert endpoint.timeout == 30 and endpoint.closed, "RPC must be bounded and closed"
    assert len(endpoint.sent) == 1, "one invocation must issue one request only"
    request = json.loads(endpoint.sent[0])
    assert set(request) == {"scope_sha256", "request_id", "action", "payload"}, (
        "RPC must not disclose unrelated process or environment data"
    )
    assert request["scope_sha256"] == binding.digest(), (
        "the entire source scope must be bound"
    )
    assert request["action"] == "status" and request["payload"] == {}, (
        "RPC cannot change the requested command"
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"scope_sha256": "0" * 64},
        {"request_id": "0" * 32},
        {"action": "exit"},
        {"error_kind": "BoundaryDenied"},
        {"error_kind": None},
    ],
)
def test_rpc_rejects_unbound_or_failed_responses(
    rpc: Any, changes: dict[str, Any]
) -> None:
    binding, node, endpoint, _ = rpc
    endpoint.changes = changes
    with pytest.raises(BoundaryDenied, match="unbound"):
        probe.daemon_request(binding, node, "status", {})
    assert endpoint.closed and len(endpoint.sent) == 1, (
        "failure cannot retry or leak a socket"
    )


@pytest.mark.parametrize(
    ("raw", "error"),
    [
        (b"[]", BoundaryDenied),
        (b"", ValueError),
        (b"{", ValueError),
        (b"\xff", UnicodeError),
    ],
)
def test_rpc_closed_or_invalid_response_is_not_completion(
    rpc: Any, raw: bytes, error: type[Exception]
) -> None:
    binding, node, endpoint, _ = rpc
    endpoint.raw = raw
    with pytest.raises(error):
        probe.daemon_request(binding, node, "finish", {})
    assert endpoint.closed, "a closed or malformed response must close the local socket"


@pytest.mark.parametrize(
    ("failure", "error"),
    [
        ("connect", ConnectionRefusedError),
        ("send", BrokenPipeError),
        ("receive", TimeoutError),
    ],
)
def test_lost_rpc_never_fabricates_a_receipt_or_retries(
    rpc: Any, failure: str, error: type[Exception]
) -> None:
    binding, node, endpoint, _ = rpc
    endpoint.failure = failure
    with pytest.raises(error):
        probe.daemon_request(binding, node, "exit", {})
    assert endpoint.closed and len(endpoint.sent) <= 1, (
        "unknown outcomes cannot resubmit"
    )


@pytest.mark.parametrize("identity", ["uid", "boot", "unavailable"])
def test_rpc_peer_guard_precedes_command_send(
    rpc: Any, monkeypatch: pytest.MonkeyPatch, identity: str
) -> None:
    binding, node, endpoint, _ = rpc
    if identity == "uid":
        endpoint.peer_uid += 1
    elif identity == "boot":
        node = node.model_copy(update={"boot_id": "replaced-boot"})
    else:

        def absent(*args: Any, **kwargs: Any) -> None:
            raise BoundaryDenied("local fake: identity is unavailable")

        monkeypatch.setattr(probe, "process_identity", absent)
    with pytest.raises(BoundaryDenied, match="identity"):
        probe.daemon_request(binding, node, "calibrated", {})
    assert not endpoint.sent and endpoint.closed, (
        "unproven peer cannot receive a command"
    )


def test_rpc_truncation_never_becomes_a_partial_completion_receipt(rpc: Any) -> None:
    binding, node, endpoint, _ = rpc
    endpoint.flags = socket.MSG_TRUNC
    with pytest.raises(BoundaryDenied, match="truncated"):
        probe.daemon_request(binding, node, "finish", {})
    assert endpoint.closed, "truncated replies must close the client"


@pytest.mark.parametrize("excess", [0, 1])
def test_rpc_request_byte_limit_is_exact(
    rpc: Any, monkeypatch: pytest.MonkeyPatch, excess: int
) -> None:
    binding, node, endpoint, _ = rpc
    nonce = "1" * 32
    monkeypatch.setattr(
        probe, "secrets", SimpleNamespace(token_hex=lambda length: nonce)
    )
    empty = json.dumps(
        {
            "scope_sha256": binding.digest(),
            "action": "status",
            "payload": {"padding": ""},
            "request_id": nonce,
        },
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    payload = {"padding": "x" * (probe.MAX_MESSAGE_BYTES - len(empty) + excess)}
    if excess:
        with pytest.raises(BoundaryDenied, match="oversized"):
            probe.daemon_request(binding, node, "status", payload)
        assert not endpoint.sent, "one byte over the limit must be refused before send"
    else:
        probe.daemon_request(binding, node, "status", payload)
        assert len(endpoint.sent[0]) == probe.MAX_MESSAGE_BYTES, (
            "the exact limit is permitted"
        )
    assert endpoint.closed, "both size-bound paths must close the client"


def send_packet(daemon: Any, value: Any = None, *, raw: bytes | None = None) -> bytes:
    with socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET) as endpoint:
        endpoint.settimeout(2)
        endpoint.connect(str(daemon.directory / "control.sock"))
        endpoint.sendall(json.dumps(value).encode() if raw is None else raw)
        return endpoint.recv(probe.MAX_MESSAGE_BYTES)


def request_for(daemon: Any) -> dict[str, Any]:
    return {
        "scope_sha256": daemon.binding.digest(),
        "request_id": "1" * 32,
        "action": "calibrated",
        "payload": {},
    }


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("scope_sha256", None),
        ("request_id", None),
        ("request_id", "short"),
        ("request_id", "x" * 33),
        ("payload", []),
        ("payload", None),
        ("extra", "unapproved"),
    ],
)
def test_daemon_rejects_request_shape_before_calibration(
    daemon: Any, key: str, value: Any
) -> None:
    request = request_for(daemon)
    request[key] = value
    assert send_packet(daemon, request) == b"", (
        "refused requests cannot receive success"
    )
    with pytest.raises(BoundaryDenied, match="unbound"):
        daemon.future.result(2)
    assert "calibrate" not in daemon.calls and daemon.calls[-1] == "close", (
        "shape failure must close the witness before calibration"
    )
    assert not daemon.directory.exists(), "owned mailbox must be removed after refusal"


@pytest.mark.parametrize("missing", ["scope_sha256", "request_id", "action", "payload"])
def test_daemon_does_not_guess_missing_command_identity(
    daemon: Any, missing: str
) -> None:
    request = request_for(daemon)
    request.pop(missing)
    assert send_packet(daemon, request) == b"", "missing identity must not get an ACK"
    with pytest.raises(BoundaryDenied, match="unbound"):
        daemon.future.result(2)
    assert "calibrate" not in daemon.calls, (
        "missing command identity cannot start observation"
    )


def test_daemon_rejects_non_object_request(daemon: Any) -> None:
    assert send_packet(daemon, []) == b"", "non-object commands cannot get a receipt"
    with pytest.raises(BoundaryDenied, match="unbound"):
        daemon.future.result(2)


@pytest.mark.parametrize("defect", ["uid", "oversize"])
def test_daemon_rejects_foreign_peer_or_truncated_request(
    daemon: Any, monkeypatch: pytest.MonkeyPatch, defect: str
) -> None:
    if defect == "uid":
        monkeypatch.setattr(
            probe,
            "struct",
            SimpleNamespace(
                calcsize=struct.calcsize,
                unpack=lambda *args: (os.getpid(), os.geteuid() + 1, os.getegid()),
            ),
        )
        raw = json.dumps(request_for(daemon)).encode()
    else:
        raw = b"x" * (probe.MAX_MESSAGE_BYTES + 1)
    assert send_packet(daemon, raw=raw) == b"", (
        "invalid callers must not receive an ACK"
    )
    with pytest.raises(BoundaryDenied, match="caller"):
        daemon.future.result(2)
    assert "calibrate" not in daemon.calls, "invalid caller must not calibrate"


def test_daemon_oversized_receipt_is_never_sent(
    daemon: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        getattr(probe, "AttachedExecWitness"),
        "start_receipt",
        lambda *args, **kwargs: WireReceipt({"padding": "x" * probe.MAX_MESSAGE_BYTES}),
    )
    with pytest.raises(ValueError):
        daemon.request("calibrated")
    with pytest.raises(BoundaryDenied, match="response exceeds"):
        daemon.future.result(2)
    assert daemon.calls[-1] == "close" and not daemon.directory.exists(), (
        "oversized responses must close the witness and owned mailbox"
    )


@pytest.mark.parametrize("action", ["calibrated", "clients"])
def test_completed_witness_cannot_restart_calibration_or_client_observation(
    daemon: Any, action: str
) -> None:
    daemon.request("calibrated")
    daemon.request("finish", daemon.quiet.model_dump(mode="json"))
    with pytest.raises(ValueError):
        daemon.request(action, {"pod_uids": ["owned"]})
    with pytest.raises(BoundaryDenied, match="out of order"):
        daemon.future.result(2)
    assert daemon.calls.count("calibrate") == 1 and daemon.calls.count("finish") == 1, (
        "a completed witness cannot start a second measurement"
    )


@pytest.mark.parametrize("field", ["source_sha256", "release_id", "execution_epoch"])
def test_scope_source_release_and_command_epoch_are_bound(
    daemon: Any, field: str
) -> None:
    value: Any = 99 if field == "execution_epoch" else "e" * 64
    changed = daemon.binding.model_copy(update={field: value})
    request = request_for(daemon)
    request["scope_sha256"] = changed.digest()
    assert send_packet(daemon, request) == b"", (
        "changed source authority cannot get an ACK"
    )
    with pytest.raises(BoundaryDenied, match="unbound"):
        daemon.future.result(2)
    assert "calibrate" not in daemon.calls, "source drift must precede measurement"


@pytest.fixture
def startup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    binding = scope()
    identity = process_identity(os.getpid())
    node = binding.nodes[0].model_copy(update={"boot_id": identity.boot_id})
    binding = binding.model_copy(update={"nodes": (node, binding.nodes[1])})
    directory = tmp_path / "mailboxes" / "node"
    calls: list[str] = []
    values = SimpleNamespace(
        executable="/usr/bin/true", pid=str(os.getpid()), failure=None
    )
    monkeypatch.setenv("LATE_OWNERSHIP_WITNESS_POD_UID", "owned-local-pod")
    monkeypatch.setattr(probe, "mailbox_path", lambda *args: directory)
    monkeypatch.setattr(
        probe, "shutil", SimpleNamespace(which=lambda _: values.executable)
    )

    def pid_read(command: list[str], **kwargs: Any) -> SimpleNamespace:
        assert command == [
            "systemctl",
            "show",
            "--property=MainPID",
            "--value",
            "gpu-fault-node-agent.service",
        ], "startup must issue only the exact read-only Agent identity query"
        assert kwargs == {
            "text": True,
            "capture_output": True,
            "check": True,
            "timeout": 15,
        }, "Agent identity reads must be checked and bounded"
        calls.append("pid-read")
        return SimpleNamespace(stdout=values.pid)

    class Witness:
        def __init__(self, path: Path, tracee: Any, **kwargs: Any) -> None:
            assert path == directory and tracee == identity, (
                "witness must bind the original Agent"
            )
            assert kwargs["executable"] == Path("/usr/bin/true").resolve(), (
                "the witness must pin the resolved executable"
            )
            calls.append("construct")
            if values.failure == "construct":
                raise BoundaryDenied("local fake: construct failed")

        def start(self) -> None:
            calls.append("start")
            if values.failure == "start":
                raise BoundaryDenied("local fake: start failed")

        def check(self) -> None:
            calls.append("check")
            if values.failure == "check":
                raise BoundaryDenied("local fake: check failed")

        def close(self) -> None:
            calls.append("close")

    monkeypatch.setattr(probe, "subprocess", SimpleNamespace(run=pid_read))
    monkeypatch.setattr(probe, "AttachedExecWitness", Witness)
    return SimpleNamespace(
        binding=binding,
        node=node,
        directory=directory,
        calls=calls,
        values=values,
        probe=probe.NodeProbe(binding, node),
    )


@pytest.mark.parametrize(
    "failure", ["executable", "pid", "zero-pid", "boot", "construct"]
)
def test_startup_identity_failure_never_attaches_or_calibrates(
    startup: Any, failure: str
) -> None:
    if failure == "executable":
        startup.values.executable = None
    elif failure == "pid":
        startup.values.pid = "unknown"
    elif failure == "zero-pid":
        startup.values.pid = "0"
    elif failure == "boot":
        startup.probe.node = startup.node.model_copy(update={"boot_id": "wrong-boot"})
    else:
        startup.values.failure = "construct"
    with pytest.raises(BoundaryDenied):
        startup.probe.run_daemon()
    assert "start" not in startup.calls, "startup failure must not attach a tracer"
    assert not (startup.directory / "control.sock").exists(), (
        "failed startup must expose no RPC"
    )
    receipt = json.loads((startup.directory / "owner.json").read_text())
    assert receipt["pod_uid"] == "owned-local-pod", (
        "partial startup must retain exact cleanup custody"
    )
    assert receipt["scope_sha256"] == startup.binding.digest(), (
        "partial custody must remain scoped"
    )


@pytest.mark.parametrize("failure", ["start", "check", "deadline", "wait"])
def test_daemon_termination_always_closes_witness_without_terminal_success(
    startup: Any, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    if failure in {"start", "check"}:
        startup.values.failure = failure
    elif failure == "deadline":
        startup.probe.deadline = 0.0
    else:

        def wait(*args: Any) -> tuple[list[Any], list[Any], list[Any]]:
            startup.probe.deadline = 0.0
            return [], [], []

        monkeypatch.setattr(probe, "select", SimpleNamespace(select=wait))
    with pytest.raises(BoundaryDenied, match="failed|deadline expired"):
        startup.probe.run_daemon()
    assert startup.calls[-1] == "close", (
        "every post-construction exit must close the witness"
    )
    assert not startup.directory.exists(), (
        "owned daemon termination must remove its mailbox"
    )


def test_daemon_lost_reply_closes_both_endpoints_and_the_witness(
    startup: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = {
        "scope_sha256": startup.binding.digest(),
        "request_id": "1" * 32,
        "action": "status",
        "payload": {},
    }
    endpoint = RpcSocket(raw=json.dumps(request).encode(), failure="send")

    class Server:
        closed = False

        def __enter__(self) -> Self:
            return self

        def __exit__(
            self,
            exc_type: type[BaseException] | None,
            exc: BaseException | None,
            traceback: TracebackType | None,
        ) -> None:
            self.closed = True

        def bind(self, path: str) -> None:
            assert path == str(startup.directory / "control.sock"), (
                "the daemon must bind only its private owned mailbox"
            )
            Path(path).touch(mode=0o600)

        def listen(self, backlog: int) -> None:
            assert backlog == 2, "the daemon listener must remain bounded"

        def accept(self) -> tuple[RpcSocket, None]:
            return endpoint, None

    server = Server()
    monkeypatch.setattr(
        probe,
        "socket",
        SimpleNamespace(
            AF_UNIX=socket.AF_UNIX,
            SOCK_SEQPACKET=socket.SOCK_SEQPACKET,
            SOL_SOCKET=socket.SOL_SOCKET,
            SO_PEERCRED=socket.SO_PEERCRED,
            MSG_TRUNC=socket.MSG_TRUNC,
            socket=lambda *_: server,
        ),
    )
    monkeypatch.setattr(
        probe, "select", SimpleNamespace(select=lambda *args: ([server], [], []))
    )
    with pytest.raises(BrokenPipeError):
        startup.probe.run_daemon()
    assert len(endpoint.sent) == 1, "an undelivered response must not be retried"
    assert endpoint.closed and server.closed, "lost replies must close both sockets"
    assert startup.calls[-1] == "close", (
        "lost replies must close the independent witness"
    )
    assert not startup.directory.exists(), (
        "lost replies must remove only the owned mailbox"
    )


@pytest.mark.parametrize("kind", ["file", "public"])
def test_private_directory_guard_refuses_non_directory_and_public_access(
    tmp_path: Path, kind: str
) -> None:
    target = tmp_path / "mailbox"
    if kind == "file":
        target.write_text("local fixture")
    else:
        target.mkdir(mode=0o755)
        target.chmod(0o755)
    with pytest.raises(BoundaryDenied, match="privately owned"):
        probe.private_directory(target)
    assert target.exists(), "identity refusal must not delete the untrusted path"


@pytest.mark.parametrize("target", ["directory", "owner", "child"])
def test_cleanup_does_not_touch_foreign_uid_paths(
    stale_mailbox: Any, monkeypatch: pytest.MonkeyPatch, target: str
) -> None:
    binding, directory, _ = stale_mailbox
    selected = (
        directory
        if target == "directory"
        else directory / ("owner.json" if target == "owner" else "control.sock")
    )
    if target == "child":
        with socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET) as endpoint:
            endpoint.bind(str(selected))
    original = Path.lstat

    def metadata(path: Path) -> os.stat_result:
        result = original(path)
        if path == selected:
            values = list(result)
            values[4] = os.geteuid() + 1
            return os.stat_result(values)
        return result

    monkeypatch.setattr(Path, "lstat", metadata)
    with pytest.raises(BoundaryDenied, match="private|unowned"):
        probe.cleanup_mailbox(binding, binding.nodes[0], "owned")
    assert (directory / "owner.json").exists(), (
        "foreign ownership cannot authorize deletion"
    )


def test_cleanup_reused_pid_does_not_signal_the_new_incarnation(
    stale_mailbox: Any,
) -> None:
    binding, directory, owner = stale_mailbox
    current = process_identity(os.getpid())
    recorded = current.model_copy(update={"start_ticks": current.start_ticks + 1})
    owner["producer"] = recorded.model_dump(mode="json")
    (directory / "owner.json").write_text(json.dumps(owner))
    probe.cleanup_mailbox(binding, binding.nodes[0], "owned")
    assert not directory.exists(), (
        "a dead original incarnation can release its owned mailbox"
    )
    assert process_identity(os.getpid()) == current, (
        "cleanup must never signal a reused PID"
    )


def entry_request(binding: AcceptanceScope, **options: Any) -> dict[str, Any]:
    return {
        "scope": binding.model_dump(mode="json"),
        "node": binding.nodes[0].name,
        **options,
    }


def test_entry_rpc_failure_reports_only_error_type(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    binding = scope()
    request = entry_request(binding, action="status", payload={})
    monkeypatch.setattr(
        probe, "sys", SimpleNamespace(stdin=io.StringIO(json.dumps(request)))
    )

    def fail(*args: Any) -> None:
        raise TimeoutError("test-only private diagnostic")

    monkeypatch.setattr(probe, "daemon_request", fail)
    assert probe.main() == 1, "lost replies must not report successful completion"
    assert json.loads(capsys.readouterr().out) == {"error_kind": "TimeoutError"}, (
        "errors must not echo credentials, commands, models, or raw diagnostics"
    )


@pytest.mark.parametrize("node", ["unowned-node", None, 5])
def test_entry_unknown_node_never_starts_or_contacts_a_witness(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], node: Any
) -> None:
    request = entry_request(scope(), action="status", payload={})
    request["node"] = node
    calls: list[str] = []
    monkeypatch.setattr(
        probe, "sys", SimpleNamespace(stdin=io.StringIO(json.dumps(request)))
    )
    monkeypatch.setattr(probe, "daemon_request", lambda *args: calls.append("rpc"))
    monkeypatch.setattr(
        probe.NodeProbe, "run_daemon", lambda self: calls.append("daemon")
    )
    assert probe.main() == 1, "unknown node identity must fail closed"
    assert set(json.loads(capsys.readouterr().out)) == {"error_kind"}, (
        "failure evidence must be sanitized"
    )
    assert not calls, "unowned node must not open any host channel"


@pytest.mark.parametrize("seconds_after_end", [-60, 60, 181])
def test_daemon_deadline_never_extends_the_original_cleanup_window(
    monkeypatch: pytest.MonkeyPatch, seconds_after_end: int
) -> None:
    binding = scope()
    now = binding.maintenance_end + timedelta(seconds=seconds_after_end)
    monkeypatch.setattr(probe, "datetime", SimpleNamespace(now=lambda _: now))
    monkeypatch.setattr(probe, "time", SimpleNamespace(monotonic=lambda: 50.0))
    child = probe.NodeProbe(binding, binding.nodes[0])
    remaining = max(
        0.0, (binding.maintenance_end + timedelta(seconds=180) - now).total_seconds()
    )
    assert child.deadline <= 50.0 + remaining, (
        "reconstruction must not turn an expired action window into a fresh 180-second grant"
    )


def test_cleanup_rpc_shape_cannot_rearm_a_daemon_after_maintenance(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    now = datetime.now(timezone.utc)
    binding = scope().model_copy(
        update={
            "maintenance_start": now - timedelta(seconds=120),
            "maintenance_end": now - timedelta(seconds=60),
        }
    )
    request = entry_request(binding, action="status", payload={}, daemon=True)
    calls: list[str] = []
    monkeypatch.setattr(
        probe, "sys", SimpleNamespace(stdin=io.StringIO(json.dumps(request)))
    )
    monkeypatch.setattr(
        probe.NodeProbe, "run_daemon", lambda self: calls.append("daemon")
    )
    monkeypatch.setattr(probe, "daemon_request", lambda *args: calls.append("rpc"))
    assert probe.main() == 1, "cleanup RPC fields cannot authorize a new daemon"
    assert set(json.loads(capsys.readouterr().out)) == {"error_kind"}, (
        "ambiguous mode must fail closed"
    )
    assert not calls, "no daemon or RPC may start from an ambiguous cleanup request"


@pytest.mark.parametrize("excess", [0, 1])
@pytest.mark.parametrize("encoding", ["ascii", "utf8"])
def test_entry_enforces_the_message_limit_before_host_io(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    excess: int,
    encoding: str,
) -> None:
    binding = scope()
    request = entry_request(binding, action="status", payload={"padding": ""})
    empty = json.dumps(request, ensure_ascii=False)
    remaining = probe.MAX_MESSAGE_BYTES + excess - len(empty.encode())
    character = "x" if encoding == "ascii" else "\u00e9"
    count, tail = divmod(remaining, len(character.encode()))
    request["payload"]["padding"] = character * count + "x" * tail
    raw = json.dumps(request, ensure_ascii=False)
    assert len(raw.encode()) == probe.MAX_MESSAGE_BYTES + excess, (
        "the boundary control must measure bytes, not Unicode characters"
    )
    calls: list[str] = []

    def status(*args: Any) -> dict[str, Any]:
        calls.append("status")
        return {"scope_sha256": binding.digest(), "phase": "ATTACHED"}

    monkeypatch.setattr(probe, "sys", SimpleNamespace(stdin=io.StringIO(raw)))
    monkeypatch.setattr(probe, "daemon_request", status)
    assert probe.main() == excess, (
        "one byte above the input bound must fail before parsing"
    )
    response = json.loads(capsys.readouterr().out)
    if excess:
        assert set(response) == {"error_kind"} and not calls, (
            "oversized input cannot reach RPC"
        )
    else:
        assert response["phase"] == "ATTACHED" and calls == ["status"], (
            "the exact size boundary may carry one scoped RPC"
        )


@pytest.mark.parametrize(
    "mode",
    [
        {"daemon": False},
        {"daemon": 1},
        {"daemon": "true"},
        {"daemon": None},
        {"daemon": True, "extra": "unapproved"},
        {"daemon": True, "payload": {}},
        {"daemon": False, "action": "status", "payload": {}},
        {"daemon": True, "action": "status", "payload": {}},
        {"action": "status"},
        {"action": "status", "payload": [], "extra": "unapproved"},
        {"action": "status", "payload": []},
        {"action": "status", "payload": None},
        {"action": [], "payload": {}},
        {"action": "unsupported", "payload": {}},
        {"action": None, "payload": {}},
        {},
    ],
)
def test_entry_requires_one_exact_mode_before_any_host_io(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    mode: dict[str, Any],
) -> None:
    request = entry_request(scope(), **mode)
    calls: list[str] = []
    monkeypatch.setattr(
        probe, "sys", SimpleNamespace(stdin=io.StringIO(json.dumps(request)))
    )
    monkeypatch.setattr(
        probe.NodeProbe, "run_daemon", lambda self: calls.append("daemon")
    )
    monkeypatch.setattr(probe, "daemon_request", lambda *args: calls.append("rpc"))
    monkeypatch.setattr(probe, "cleanup_mailbox", lambda *args: calls.append("cleanup"))
    assert probe.main() == 1, "ambiguous or malformed modes must fail closed"
    assert json.loads(capsys.readouterr().out) == {"error_kind": "BoundaryDenied"}, (
        "request-shape refusal must not disclose the request"
    )
    assert not calls, "mode validation must precede all host I/O"


@pytest.mark.parametrize("raw", ["[]", "null", "false", '"status"'])
def test_entry_rejects_non_object_input(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], raw: str
) -> None:
    monkeypatch.setattr(probe, "sys", SimpleNamespace(stdin=io.StringIO(raw)))
    assert probe.main() == 1, "non-object input must fail closed"
    assert json.loads(capsys.readouterr().out) == {"error_kind": "BoundaryDenied"}, (
        "a primitive cannot carry command authority"
    )


@pytest.mark.parametrize("encoding", ["ascii", "utf8"])
def test_entry_checks_bytes_before_json_or_scope_decode(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], encoding: str
) -> None:
    raw = (
        "x" * (probe.MAX_MESSAGE_BYTES + 1)
        if encoding == "ascii"
        else "\u00e9" * (probe.MAX_MESSAGE_BYTES // 2 + 1)
    )
    calls: list[str] = []

    def decode(*args: Any, **kwargs: Any) -> None:
        calls.append("decode")
        raise AssertionError("oversized input reached decoding")

    monkeypatch.setattr(probe, "sys", SimpleNamespace(stdin=io.StringIO(raw)))
    monkeypatch.setattr(probe, "json", SimpleNamespace(loads=decode, dumps=json.dumps))
    assert probe.main() == 1, "oversized input must fail before decoding"
    assert json.loads(capsys.readouterr().out) == {"error_kind": "BoundaryDenied"}, (
        "byte refusal must happen before any parsing exception"
    )
    assert not calls, "neither JSON nor the embedded scope may be decoded"


@pytest.mark.parametrize(
    "payload", ['{"a":1,"a":2}', '{"a":NaN}', '{"a":Infinity}', '{"a":-Infinity}']
)
def test_entry_rejects_ambiguous_or_nonstandard_json(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], payload: str
) -> None:
    binding = scope()
    raw = (
        '{"scope":'
        + json.dumps(binding.model_dump(mode="json"))
        + ',"node":'
        + json.dumps(binding.nodes[0].name)
        + ',"action":"status","payload":'
        + payload
        + "}"
    )
    calls: list[str] = []
    monkeypatch.setattr(probe, "sys", SimpleNamespace(stdin=io.StringIO(raw)))
    monkeypatch.setattr(probe, "daemon_request", lambda *args: calls.append("rpc"))
    assert probe.main() == 1, "ambiguous JSON cannot carry source or command authority"
    assert json.loads(capsys.readouterr().out) == {"error_kind": "BoundaryDenied"}, (
        "strict JSON refusal must be sanitized"
    )
    assert not calls, "ambiguous payloads must not reach a host channel"


def test_duplicate_daemon_selector_is_not_accepted(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    raw = json.dumps(entry_request(scope(), daemon=True))
    raw = raw[:-1] + ',"daemon":true}'
    calls: list[str] = []
    monkeypatch.setattr(probe, "sys", SimpleNamespace(stdin=io.StringIO(raw)))
    monkeypatch.setattr(
        probe.NodeProbe, "run_daemon", lambda self: calls.append("daemon")
    )
    assert probe.main() == 1, "even identical duplicate selectors must be rejected"
    assert json.loads(capsys.readouterr().out) == {"error_kind": "BoundaryDenied"}, (
        "selector ambiguity cannot be resolved by last-field-wins"
    )
    assert not calls, "duplicate selectors cannot start a daemon"


def test_entry_preserves_request_prefetched_by_the_pinned_source_loader(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    binding = scope()
    prefix = "local-pinned-source-prefix\n"
    raw = json.dumps(entry_request(binding, action="status", payload={})) + "\n"
    calls: list[str] = []

    def status(*args: Any) -> dict[str, Any]:
        calls.append("status")
        return {"scope_sha256": binding.digest(), "phase": "ATTACHED"}

    with io.TextIOWrapper(
        io.BytesIO((prefix + raw).encode()), encoding="utf-8"
    ) as source:
        assert source.read(len(prefix)) == prefix, (
            "the source loader consumes only its prefix"
        )
        monkeypatch.setattr(probe, "sys", SimpleNamespace(stdin=source))
        monkeypatch.setattr(probe, "daemon_request", status)
        assert probe.main() == 0, (
            "the request must remain on the same buffered text stream"
        )
    assert json.loads(capsys.readouterr().out)["scope_sha256"] == binding.digest(), (
        "buffering must not change the source binding"
    )
    assert calls == ["status"], "the prefetched request must dispatch exactly once"


def test_daemon_start_rechecks_action_window_before_creating_mailbox(
    startup: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    expired = startup.binding.maintenance_end + timedelta(seconds=1)
    monkeypatch.setattr(probe, "datetime", SimpleNamespace(now=lambda _: expired))
    with pytest.raises(ValueError, match="approved maintenance window"):
        startup.probe.run_daemon()
    assert not startup.calls, "delayed startup must not query or attach to the Agent"
    assert not startup.directory.exists(), (
        "expired authority must not create a new mailbox"
    )

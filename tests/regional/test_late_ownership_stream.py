from __future__ import annotations

import json
import time
from types import SimpleNamespace

import pytest

from scripts.e2e.regional import late_ownership_stream as stream
from scripts.e2e.regional.late_ownership_barrier import BoundaryDenied


class Socket:
    def __init__(self, chunks=(), *, returncode=0):
        self.chunks = list(chunks)
        self.stdout = ""
        self.stderr = ""
        self.open = True
        self.returncode = returncode
        self.writes = []
        self.updates = []
        self.close_count = 0

    def is_open(self):
        return self.open

    def update(self, *, timeout):
        self.updates.append(timeout)
        if self.chunks:
            self.stdout, self.stderr = self.chunks.pop(0)
        else:
            self.open = False

    def read_stdout(self):
        value, self.stdout = self.stdout, ""
        return value

    def read_stderr(self):
        value, self.stderr = self.stderr, ""
        return value

    def write_stdin(self, value):
        self.writes.append(value)

    def close(self):
        self.close_count += 1
        self.open = False


def channel(socket=None, *, check=lambda: None):
    return stream.ProbeStream(
        socket or Socket(),
        scope_sha256="a" * 64,
        check_identity=check,
        deadline=time.monotonic() + 30,
    )


def message(**changes):
    return (
        json.dumps(
            {
                "kind": "ready",
                "scope_sha256": "a" * 64,
                "sequence": 1,
                "payload": {"phase": "ready"},
                **changes,
            }
        )
        + "\n"
    )


def test_split_frames_are_bound_and_both_sides_recheck_uid():
    checks = []
    text = message()
    socket = Socket([(text[:10], ""), (text[10:], "")])
    pipe = channel(socket, check=lambda: checks.append("uid"))
    pipe.send("begin", {"owned": True})
    assert json.loads(socket.writes[0]) == {
        "kind": "begin",
        "scope_sha256": "a" * 64,
        "payload": {"owned": True},
    }
    assert pipe.receive("ready") == {"phase": "ready"}
    assert len(checks) == 5
    assert all(0 < timeout <= 1 for timeout in socket.updates), (
        "frame I/O must stay within the remaining one-second deadline"
    )
    pipe.finish()
    assert pipe.closed and socket.close_count == 1


@pytest.mark.parametrize("deadline", [float("nan"), float("inf"), -1, 0])
def test_no_unbounded_or_expired_stream(deadline):
    with pytest.raises(BoundaryDenied, match="deadline"):
        stream.ProbeStream(
            Socket(),
            scope_sha256="a" * 64,
            check_identity=lambda: None,
            deadline=deadline,
        )


@pytest.mark.parametrize(
    "data",
    [
        "not-json\n",
        "[]\n",
        message(kind="late-callback"),
        message(scope_sha256="b" * 64),
        message(sequence=True),
        message(sequence=2),
        message(payload=[]),
        message(extra="field"),
    ],
)
def test_stale_or_malformed_acknowledgement_is_never_a_receipt(data):
    pipe = channel(Socket([(data, "")]))
    with pytest.raises(BoundaryDenied, match="protocol|stale"):
        pipe.receive("ready")
    assert pipe.sequence == 0


@pytest.mark.parametrize(
    "defect", ["stdout", "stderr", "request", "exit", "closed", "deadline"]
)
def test_stream_size_exit_and_window_guards(defect):
    socket = Socket()
    pipe = channel(socket)
    if defect == "stdout":
        socket.chunks = [("x" * (stream.MAX_MESSAGE_BYTES + 1), "")]
    elif defect == "stderr":
        socket.chunks = [("", "private" * stream.MAX_MESSAGE_BYTES)]
    elif defect == "request":
        with pytest.raises(BoundaryDenied, match="bound"):
            pipe.send("begin", {"value": "x" * stream.MAX_MESSAGE_BYTES})
        assert socket.writes == []
        return
    elif defect == "exit":
        socket.open = False
    elif defect == "closed":
        pipe.close()
    else:
        pipe.deadline = 0
    with pytest.raises(BoundaryDenied):
        pipe.receive("ready")


@pytest.mark.parametrize("defect", ["trailing", "stderr", "nonzero", "uid"])
def test_terminal_json_without_verified_process_exit_cannot_finish(defect):
    socket = Socket()
    pipe = channel(socket)
    if defect == "trailing":
        socket.chunks = [("late action\n", "")]
    elif defect == "stderr":
        socket.chunks = [("", "x" * (stream.MAX_MESSAGE_BYTES + 1))]
    elif defect == "nonzero":
        socket.returncode = 1
    else:

        def changed():
            raise BoundaryDenied("UID replaced")

        pipe.check_identity = changed
    with pytest.raises(BoundaryDenied):
        pipe.finish()
    assert not pipe.closed, (
        "failed process verification must not mark the stream finished"
    )
    pipe.close()


def test_replaced_owner_between_send_and_ack_is_refused():
    checks = iter([None, BoundaryDenied("owner replaced")])

    def check():
        result = next(checks)
        if result is not None:
            raise result

    pipe = channel(check=check)
    with pytest.raises(BoundaryDenied, match="replaced"):
        pipe.send("recheck", {})
    assert len(pipe.stream.writes) == 1, "lost acknowledgement is not proof of no send"


@pytest.mark.parametrize("tls,host", [(False, "https://local"), (True, "http://local")])
def test_kubernetes_transport_rejects_unverified_tls(monkeypatch, tls, host):
    from kubernetes import config

    closed = []
    client = SimpleNamespace(
        configuration=SimpleNamespace(verify_ssl=tls, host=host),
        close=lambda: closed.append(True),
    )
    monkeypatch.setattr(config, "new_client_from_config", lambda **kw: client)
    with pytest.raises(BoundaryDenied, match="TLS"):
        stream.open_executor_stream(
            kubeconfig="/dev/null",
            context="local",
            namespace="owned",
            pod="executor",
            python="/python",
            program="pass",
        )
    assert closed == [True]


@pytest.mark.parametrize("chroot", [None, "/host"])
@pytest.mark.parametrize("failure", [False, True])
def test_kubernetes_stream_is_pinned_and_always_closes_its_api_client(
    monkeypatch, chroot, failure
):
    import kubernetes.stream as kube_stream
    from kubernetes import client, config

    closed, calls = [], []
    api = SimpleNamespace(
        configuration=SimpleNamespace(verify_ssl=True, host="https://local"),
        close=lambda: closed.append(True),
    )
    socket = Socket()
    marker = object()
    monkeypatch.setattr(config, "new_client_from_config", lambda **kw: api)
    monkeypatch.setattr(
        client,
        "CoreV1Api",
        lambda value: SimpleNamespace(connect_get_namespaced_pod_exec=marker),
    )

    def connect(*args, **kwargs):
        calls.append((args, kwargs))
        if failure:
            raise OSError("local fake connection")
        return socket

    monkeypatch.setattr(kube_stream, "stream", connect)
    kwargs = dict(
        kubeconfig="/dev/null",
        context="local",
        namespace="owned",
        pod="executor",
        python="/python",
        program="pass",
        chroot=chroot,
    )
    if failure:
        with pytest.raises(OSError):
            stream.open_executor_stream(**kwargs)
    else:
        managed = stream.open_executor_stream(**kwargs)
        assert managed.returncode == 0
        managed.close()
        assert socket.close_count == 1
    assert closed == [True]
    args, options = calls[0]
    assert args == (marker, "executor", "owned")
    assert options["command"] == (["chroot", "/host"] if chroot else []) + [
        "/python",
        "-I",
        "-u",
        "-c",
        stream.stdin_loader("pass"),
    ]
    if not failure:
        assert socket.writes == ["pass"]
    assert options["_preload_content"] is False and options["tty"] is False


def test_api_client_is_closed_even_when_websocket_close_fails():
    closed = []

    def failed():
        raise OSError("close")

    managed = stream.ManagedExecStream(
        SimpleNamespace(close=failed),
        SimpleNamespace(close=lambda: closed.append(True)),
    )
    with pytest.raises(OSError):
        managed.close()
    assert closed == [True]


def test_cpu_holder_request_is_serviced_before_continuing_the_queued_boundary():
    check = message(kind="holder-check", payload={"nonce": "owned"})
    socket = Socket([(check + message(sequence=2), "")])
    calls = []
    pipe = channel(socket)
    pipe.holder_check = lambda payload: calls.append(payload) or {
        "holder_valid": True,
        **payload,
    }
    assert pipe.receive("ready") == {"phase": "ready"}
    assert calls == [{"nonce": "owned"}]
    assert json.loads(socket.writes[0]) == {
        "kind": "holder-check-result",
        "scope_sha256": "a" * 64,
        "payload": {"holder_valid": True, "nonce": "owned"},
    }
    assert pipe.sequence == 2


def test_failed_cpu_check_never_returns_a_successful_holder_acknowledgement():
    pipe = channel(Socket([(message(kind="holder-check"), "")]))

    def failed(payload):
        raise BoundaryDenied("CPU owner was replaced")

    pipe.holder_check = failed
    with pytest.raises(BoundaryDenied, match="replaced"):
        pipe.receive("ready")
    assert pipe.stream.writes == []

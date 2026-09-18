from __future__ import annotations

import base64
import io
import json
import os
import select
import signal
import subprocess
import sys
import time
from contextlib import redirect_stdout
from types import SimpleNamespace

import pytest

from scripts.e2e.regional import auth015_http as http
from tests.regional._cov95_auth015_http import (
    loopback_server,
    owned_subreaper,
    server_certificate,
)

PRIVATE_MARKER = "synthetic-private-http-marker"


@pytest.fixture
def workers(monkeypatch):
    original = subprocess.Popen
    records = []

    def launch(*args, **kwargs):
        process = original(*args, **kwargs)
        records.append((process, args[0], kwargs["env"]))
        return process

    monkeypatch.setattr(http.subprocess, "Popen", launch)
    yield records
    for process, _arguments, _environment in records:
        assert process.poll() is not None, (
            "the caller must kill and reap its own worker"
        )
        assert process.stdout.closed and process.stderr.closed, (
            "private worker streams must be closed"
        )


@pytest.mark.parametrize("tls", [False, True])
def test_body_limit_is_enforced_on_the_stream_before_eof(tmp_path, workers, tls):
    with loopback_server(tmp_path, tls=tls) as server:
        started = time.monotonic()
        with pytest.raises(http.HttpResponseTooLarge, match="byte limit"):
            http.bounded_request(
                server.url + "/oversize",
                certificate=server.certificate,
                timeout=2,
                max_response_bytes=server.byte_limit,
            )
        assert time.monotonic() - started < 2, (
            "the byte cap must reject the prefix without waiting for the declared billion-byte body"
        )
        assert len(server.requests) == 1
    assert len(workers) == 1


@pytest.mark.parametrize("phase", ["headers", "body"])
def test_trickling_response_cannot_reset_the_wall_clock_deadline(
    tmp_path, workers, phase
):
    with loopback_server(tmp_path) as server:
        started = time.monotonic()
        with pytest.raises(http.HttpDeadlineExceeded):
            http.bounded_request(
                server.url + "/" + phase,
                timeout=0.4,
                max_response_bytes=server.byte_limit,
            )
        elapsed = time.monotonic() - started
        assert elapsed < 1.4, (
            "continuous bytes must not extend the worker's absolute deadline"
        )
        assert len(server.requests) == 1
    assert len(workers) == 1


def test_private_request_data_never_enters_worker_argv_or_environment(
    tmp_path, workers
):
    with loopback_server(tmp_path, tls=True) as server:
        url = server.url + "/result?signature=" + PRIVATE_MARKER
        status, body = http.bounded_request(
            url,
            method="POST",
            headers={"Authorization": PRIVATE_MARKER},
            data=PRIVATE_MARKER.encode(),
            certificate=server.certificate,
            timeout=2,
        )
        assert (status, body) == (200, b'{"ok":true}')
        assert server.requests[0][1].endswith(PRIVATE_MARKER), (
            "the private query must reach only the selected loopback server"
        )
        assert server.requests[0][2]["Authorization"] == PRIVATE_MARKER
        assert server.requests[0][3] == PRIVATE_MARKER.encode()
    _process, arguments, environment = workers[0]
    assert arguments[1:5] == ["-I", "-S", "-B", "-c"]
    assert PRIVATE_MARKER not in repr(arguments) and url not in repr(arguments)
    assert PRIVATE_MARKER not in repr(environment)
    assert (
        environment["AWS_CONFIG_FILE"]
        == environment["AWS_SHARED_CREDENTIALS_FILE"]
        == "/dev/null"
    )


def test_worker_does_not_follow_redirects(tmp_path, workers):
    with loopback_server(tmp_path) as server:
        assert http.bounded_request(server.url + "/redirect", timeout=2) == (302, b"")
        assert len(server.requests) == 1, (
            "a response must not redirect a signed query to another target"
        )


def test_wrong_tls_authority_is_rejected_without_raw_diagnostics(tmp_path, workers):
    other = tmp_path / "other"
    other.mkdir()
    wrong_certificate = server_certificate(other)
    with loopback_server(tmp_path, tls=True) as server:
        with pytest.raises(http.BoundedHttpError, match="could not complete") as caught:
            http.bounded_request(
                server.url + "/?signature=" + PRIVATE_MARKER,
                certificate=wrong_certificate,
                timeout=2,
            )
        assert PRIVATE_MARKER not in str(caught.value)
        assert server.requests == []


def packet(base_url, **updates):
    return {
        "url": base_url,
        "method": "GET",
        "headers": {},
        "seconds": 1.0,
        "limit": 1024,
        "body": None,
        "certificate": None,
        **updates,
    }


@pytest.mark.parametrize("tls", [False, True])
def test_worker_stream_reader_and_private_serialization_use_real_http(
    tmp_path, monkeypatch, tls
):
    with loopback_server(tmp_path, tls=tls) as server:
        request = packet(server.url, certificate=server.certificate, seconds=10)
        value = http.worker_exchange(request)
        assert value["status"] == 200
        assert base64.b64decode(value["body"]) == b'{"ok":true}'
        monkeypatch.setattr(
            http.sys,
            "stdin",
            SimpleNamespace(buffer=io.BytesIO(json.dumps(request).encode())),
        )
        output = io.StringIO()
        with redirect_stdout(output):
            assert http.worker_main() == 0
        assert json.loads(output.getvalue()) == value


def test_local_cpu_probe_fallback_uses_the_bounded_stdlib_worker(tmp_path, workers):
    from scripts.e2e.regional.auth015_agent_probe import http_exchange

    with loopback_server(tmp_path) as server:
        assert http_exchange(url=server.url, timeout=2) == (200, b'{"ok":true}')
    assert len(workers) == 1


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("url", None),
        ("url", "http://127.0.0.1/\n"),
        ("url", "file:///tmp/not-http"),
        ("url", "http://user:secret@127.0.0.1"),
        ("url", "http://127.0.0.1/#fragment"),
        ("url", "http://192.0.2.1"),
        ("url", "https://127.0.0.1"),
        ("method", "DELETE"),
        ("headers", []),
        ("headers", {"test": False}),
        ("seconds", True),
        ("seconds", 0),
        ("seconds", float("nan")),
        ("limit", True),
        ("limit", 0),
        ("limit", http.MAX_RESPONSE_BYTES + 1),
    ],
)
def test_invalid_worker_inputs_fail_before_any_connection(field, value):
    with pytest.raises(http.BoundedHttpError):
        http.worker_exchange(packet("http://127.0.0.1:1", **{field: value}))


@pytest.mark.parametrize("raw", [b"[]", b"{", b"x" * (http.MAX_REQUEST_BYTES + 1)])
def test_bad_private_input_has_a_constant_error_without_payload(
    raw, monkeypatch, capsys
):
    monkeypatch.setattr(http.sys, "stdin", SimpleNamespace(buffer=io.BytesIO(raw)))
    assert http.worker_main() == 1
    assert json.loads(capsys.readouterr().out) == {"error": "transport"}


@pytest.mark.parametrize(
    ("error", "code", "reason"),
    [
        (http.HttpResponseTooLarge("private"), 65, "body_limit"),
        (TimeoutError(PRIVATE_MARKER), 124, "timeout"),
    ],
)
def test_worker_failure_output_is_small_and_does_not_echo_input(
    error, code, reason, monkeypatch, capsys
):
    monkeypatch.setattr(
        http.sys, "stdin", SimpleNamespace(buffer=io.BytesIO(b'{"seconds":1}'))
    )

    def fail(_):
        raise error

    monkeypatch.setattr(http, "worker_exchange", fail)
    assert http.worker_main() == code
    assert json.loads(capsys.readouterr().out) == {"error": reason}


@pytest.mark.parametrize("timeout", [0, -1, True, float("nan"), float("inf"), 31])
def test_invalid_limits_are_rejected_without_spawning_a_worker(monkeypatch, timeout):
    monkeypatch.setattr(
        http.subprocess,
        "run",
        lambda *a, **k: pytest.fail("invalid limits spawned a worker"),
    )
    with pytest.raises(http.BoundedHttpError, match="limits"):
        http.bounded_request("http://127.0.0.1", timeout=timeout)


def test_oversized_private_request_is_rejected_before_process_creation(monkeypatch):
    monkeypatch.setattr(
        http.subprocess,
        "run",
        lambda *a, **k: pytest.fail("oversized input spawned a worker"),
    )
    with pytest.raises(http.BoundedHttpError, match="input exceeds"):
        http.bounded_request("http://127.0.0.1", data=b"x" * http.MAX_REQUEST_BYTES)


@pytest.mark.parametrize("returncode", [1, 65, 124])
def test_worker_return_codes_never_echo_private_stderr(monkeypatch, returncode):
    monkeypatch.setattr(
        http.subprocess,
        "run",
        lambda args, **kwargs: subprocess.CompletedProcess(
            args, returncode, b"", PRIVATE_MARKER.encode()
        ),
    )
    with pytest.raises(http.BoundedHttpError) as caught:
        http.bounded_request("http://127.0.0.1")
    assert PRIVATE_MARKER not in str(caught.value)


@pytest.mark.parametrize(
    "stdout",
    [
        b"[]",
        b"{",
        b"{}",
        b'{"status":true,"body":""}',
        b'{"status":99,"body":""}',
        b'{"status":200,"body":null}',
        b'{"status":200,"body":"!"}',
        b"x" * 12000,
    ],
)
def test_malformed_worker_receipts_cannot_become_http_results(monkeypatch, stdout):
    monkeypatch.setattr(
        http.subprocess,
        "run",
        lambda args, **kwargs: subprocess.CompletedProcess(args, 0, stdout, b""),
    )
    with pytest.raises(http.BoundedHttpError, match="invalid bounded response"):
        http.bounded_request("http://127.0.0.1")


def test_start_failure_is_sanitized(monkeypatch):
    def fail(*args, **kwargs):
        raise OSError(PRIVATE_MARKER)

    monkeypatch.setattr(http.subprocess, "run", fail)
    with pytest.raises(http.BoundedHttpError, match="could not complete") as caught:
        http.bounded_request("http://127.0.0.1")
    assert PRIVATE_MARKER not in str(caught.value)


def test_decoded_worker_body_cannot_exceed_the_requested_cap(monkeypatch):
    output = json.dumps(
        {"status": 200, "body": base64.b64encode(b"ab").decode()}
    ).encode()
    monkeypatch.setattr(
        http.subprocess,
        "run",
        lambda args, **kwargs: subprocess.CompletedProcess(args, 0, output, b""),
    )
    with pytest.raises(http.BoundedHttpError, match="invalid bounded response"):
        http.bounded_request("http://127.0.0.1", max_response_bytes=1)


@pytest.mark.parametrize(
    ("url", "port"),
    [
        ("http://[::1]", 80),
        ("https://[::1]", 443),
        ("http://[::1]:8080", 8080),
        ("https://[::1]:8443", 8443),
    ],
)
def test_ipv6_default_ports_are_bound_by_the_real_http_connection_constructor(
    tmp_path, monkeypatch, url, port
):
    selected = "HTTPSConnection" if url.startswith("https:") else "HTTPConnection"
    original = getattr(http.http.client, selected)
    initialize = original.__init__
    connections = []
    ca = server_certificate(tmp_path)

    class Reply:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read(self, amount):
            assert amount == 1025
            return b"{}"

    def observed_init(connection, *args, **kwargs):
        initialize(connection, *args, **kwargs)
        connections.append(connection)
        monkeypatch.setattr(connection, "request", lambda *a, **k: None)
        monkeypatch.setattr(connection, "getresponse", lambda: Reply())

    monkeypatch.setattr(original, "__init__", observed_init)
    assert http.worker_exchange(packet(url, certificate=ca))["status"] == 200
    assert (connections[0].host, connections[0].port) == ("::1", port), (
        "None must not let http.client split the last IPv6 component into a port"
    )


@pytest.mark.parametrize("host", ["fe80::1", "2001:db8::1"])
def test_nonloopback_ipv6_cleartext_is_rejected_before_connecting(monkeypatch, host):
    monkeypatch.setattr(
        http.http.client,
        "HTTPConnection",
        lambda *a, **k: pytest.fail("cleartext IPv6 escaped loopback"),
    )
    with pytest.raises(http.BoundedHttpError, match="loopback"):
        http.worker_exchange(packet(f"http://[{host}]"))


@pytest.mark.parametrize("termination", [signal.SIGKILL, signal.SIGHUP])
def test_http_worker_expires_and_is_reaped_after_its_controller_dies(
    tmp_path, termination
):
    controller_source = (
        "import json, subprocess, sys\n"
        f"namespace = {{'__name__': 'owned_controller', '__file__': {http.__file__!r}}}\n"
        f"exec(compile({http.worker_source()!r}, {http.__file__!r}, 'exec'), namespace)\n"
        "original = subprocess.Popen\n"
        "def tracked(*args, **kwargs):\n"
        "    child = original(*args, **kwargs)\n"
        "    print(child.pid, flush=True)\n"
        "    return child\n"
        "subprocess.Popen = tracked\n"
        "namespace['bounded_request'](**json.load(sys.stdin))\n"
    )
    worker_pid = None
    reaped = False
    with owned_subreaper(), loopback_server(tmp_path) as server:
        controller = subprocess.Popen(
            [sys.executable, "-I", "-S", "-B", "-c", controller_source],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            # A runner started under nohup inherits SIGHUP=SIG_IGN and passes
            # it on; the modelled controller must die of the signal it is sent.
            preexec_fn=lambda: signal.signal(signal.SIGHUP, signal.SIG_DFL),
            env={
                "HOME": "/tmp",
                "PATH": os.defpath,
                "AWS_CONFIG_FILE": os.devnull,
                "AWS_SHARED_CREDENTIALS_FILE": os.devnull,
                "AWS_EC2_METADATA_DISABLED": "true",
                "KUBECONFIG": os.devnull,
            },
        )
        try:
            # Bounds are generous against a loaded 16-worker suite: the worker
            # must still be trickling when the controller dies, otherwise its
            # own timeout reaps it first and the orphan check below has no
            # child to wait for (seen once in a full-suite run, 2026-09-17).
            controller.stdin.write(
                json.dumps({"url": server.url + "/body", "timeout": 6.0}).encode()
            )
            controller.stdin.close()
            assert select.select([controller.stdout], [], [], 10)[0], (
                "the owned controller did not start its worker"
            )
            worker_pid = int(controller.stdout.readline())
            assert server.seen.wait(10), (
                "the worker must reach the owned trickling response"
            )
            assert controller.poll() is None, (
                "the controller must still own an in-flight request"
            )
            controller.send_signal(termination)
            # This process is the subreaper: the dying controller and its
            # orphaned worker both become our children and may be collectable
            # in either order. Reap whatever exits and attribute it, instead
            # of waiting on one pid at a time -- under a loaded suite the
            # per-pid waits raced each other (ChildProcessError / a stolen
            # exit status, 2026-09-17).
            deadline = time.monotonic() + 8
            status = None
            controller_exited = False
            while time.monotonic() < deadline and not (controller_exited and reaped):
                try:
                    pid, result = os.waitpid(-1, os.WNOHANG)
                except ChildProcessError:
                    break
                if pid == 0:
                    time.sleep(0.01)
                    continue
                if pid == controller.pid:
                    controller_exited = True
                    controller.returncode = os.waitstatus_to_exitcode(result)
                elif pid == worker_pid:
                    reaped, status = True, result
            assert controller_exited, (
                "the controller must exit once it receives the termination signal"
            )
            assert reaped, (
                "loss of the controller must not leave a trickling worker alive"
            )
            assert os.waitstatus_to_exitcode(status) == 124, (
                "the worker's independent deadline must terminate it"
            )
        finally:
            if controller.poll() is None:
                controller.kill()
            controller.wait(timeout=2)
            if worker_pid is not None and not reaped:
                try:
                    pid, _ = os.waitpid(worker_pid, os.WNOHANG)
                except ChildProcessError:
                    pid = worker_pid
                if pid == 0:
                    os.kill(worker_pid, signal.SIGKILL)
                    os.waitpid(worker_pid, 0)
            for stream in (controller.stdin, controller.stdout, controller.stderr):
                stream.close()


@pytest.mark.parametrize("state", ["unknown-handler", "existing-timer"])
def test_worker_refuses_to_replace_an_unowned_alarm_state(monkeypatch, state):
    if state == "unknown-handler":
        monkeypatch.setattr(http.signal, "getsignal", lambda _: None)
    else:
        monkeypatch.setattr(http.signal, "getitimer", lambda _: (1.0, 0.0))
    monkeypatch.setattr(
        http.signal,
        "setitimer",
        lambda *args: pytest.fail("unowned signal state was changed"),
    )
    with pytest.raises(http.BoundedHttpError, match="alarm state"):
        with http.worker_deadline(1):
            pytest.fail("unowned alarm state admitted network work")


def test_failed_alarm_install_restores_the_original_signal_handler(monkeypatch):
    previous = signal.getsignal(signal.SIGALRM)
    calls = []

    def timer(_kind, seconds):
        calls.append(seconds)
        if seconds:
            raise OSError("synthetic alarm setup failure")
        return 0.0, 0.0

    monkeypatch.setattr(http.signal, "setitimer", timer)
    with pytest.raises(OSError, match="alarm setup failure"):
        with http.worker_deadline(1):
            pytest.fail("failed alarm installation admitted network work")
    assert calls == [1, 0]
    assert signal.getsignal(signal.SIGALRM) is previous

from __future__ import annotations

import base64
import io
import json
import signal
import subprocess
import urllib.error
from collections import deque
from types import SimpleNamespace

import pytest

from gpu_fault.admin import execution
from gpu_fault.admin import native_http as native
from gpu_fault.admin.deadlines import (
    DeploymentDeadlineExceeded,
    HttpResponseTooLarge,
    deadline_scope,
)
from tests.admin._cov95_api_support import ShimTransport


def request_document(**changes):
    return {
        "backend": "http",
        "method": "POST",
        "url": "https://example.invalid/api",
        "headers": {"X-Example": "fixture"},
        "body": base64.b64encode(b"example-body").decode(),
        "proxies": {},
        **changes,
    }


@pytest.fixture
def opener(monkeypatch):
    calls = []

    class Opener:
        error = None
        status = 201
        content = b"example-response"

        def open(self, request, timeout):
            calls.append((request, timeout))
            if self.error is not None:
                raise self.error
            response = io.BytesIO(self.content)
            response.status = self.status
            return response

    transport = Opener()
    monkeypatch.setattr(native.urllib.request, "build_opener", lambda *_args: transport)
    return transport, calls


@pytest.mark.parametrize("backend", ["http", "aws"])
@pytest.mark.parametrize("body", [None, b"", b"example-body"])
def test_worker_request_preserves_payload_and_status(opener, backend, body):
    _transport, calls = opener
    result = native.worker_request(
        request_document(
            backend=backend,
            body=base64.b64encode(body).decode() if body is not None else None,
        )
    )
    assert result == {"status": 201, "body": "example-response"}
    request, timeout = calls[0]
    assert request.full_url == "https://example.invalid/api"
    assert request.method == "POST"
    assert request.data == body
    assert request.get_header("X-example") == "fixture"
    assert 0 < timeout <= 30


@pytest.mark.parametrize("status", [400, 403, 429, 500, 503])
def test_worker_http_errors_are_responses_not_transport_success(opener, status):
    transport, calls = opener
    transport.error = urllib.error.HTTPError(
        "https://example.invalid/api", status, "example-error", {}, io.BytesIO(b"error")
    )
    assert native.worker_request(request_document()) == {
        "status": status,
        "body": "error",
    }
    assert len(calls) == 1


@pytest.mark.parametrize(
    "changes",
    [
        {"backend": "other"},
        {"method": None},
        {"url": None},
        {"url": "file:///example"},
        {"headers": []},
        {"headers": {1: "value"}},
        {"headers": {"key": 1}},
        {"proxies": None},
        {"body": 1},
        {"body": "not-valid-base64"},
    ],
)
def test_invalid_worker_requests_do_not_open_network(opener, changes):
    _transport, calls = opener
    with pytest.raises((native.NativeHttpError, ValueError)):
        native.worker_request(request_document(**changes))
    assert calls == []


@pytest.fixture
def credentials(monkeypatch):
    transport = ShimTransport()
    transport.result = 0
    transport.reads[901] = deque([b'{"Version":1}', b""])
    transport.reads[902] = deque([b"example-diagnostic", b""])
    monkeypatch.setattr(native, "subprocess", transport.subprocess)
    monkeypatch.setattr(native, "os", transport.os)
    monkeypatch.setattr(native, "selectors", transport.selectors)
    return transport


def test_worker_credentials_drains_both_pipes_and_retries_nonblocking_read(credentials):
    credentials.reads[901].appendleft(BlockingIOError())
    assert native.worker_credentials() == {"Version": 1}
    command, options = credentials.calls[0]
    assert command == ["aws", "configure", "export-credentials", "--format", "process"]
    assert options["env"]["AWS_PAGER"] == ""
    assert options["env"]["AWS_CLI_AUTO_PROMPT"] == "off"
    assert credentials.stdout.closed and credentials.stderr.closed
    assert credentials.signals == []


@pytest.mark.parametrize(
    "scenario", ["exit", "malformed", "missing-pipe", "oversized", "timeout"]
)
def test_worker_credential_failures_discard_output_and_close_pipes(
    credentials, scenario
):
    expected = native.NativeHttpError
    if scenario == "exit":
        credentials.result = 19
    elif scenario == "malformed":
        credentials.reads[901] = deque([b"invalid-json", b""])
    elif scenario == "missing-pipe":
        credentials.missing_pipes = True
    elif scenario == "oversized":
        credentials.reads[902] = deque([b"x" * 65536] * 129 + [b""])
        expected = HttpResponseTooLarge
    else:
        credentials.wait_failures.extend(
            [
                subprocess.TimeoutExpired(["fake"], 30),
                subprocess.TimeoutExpired(["fake"], 0.25),
            ]
        )
        expected = subprocess.TimeoutExpired
    with pytest.raises(expected):
        native.worker_credentials()
    if scenario != "missing-pipe":
        assert credentials.stdout.closed and credentials.stderr.closed
    assert credentials.poll() is not None
    if scenario == "timeout":
        assert credentials.signals == [signal.SIGTERM, signal.SIGKILL]
    elif scenario in {"missing-pipe", "oversized"}:
        assert credentials.signals == [signal.SIGTERM]


@pytest.mark.parametrize("mode", ["request", "credentials", "unknown"])
def test_native_main_dispatches_only_known_modes(
    monkeypatch, opener, credentials, mode
):
    handlers = {}
    output = io.StringIO()
    monkeypatch.setattr(
        native,
        "signal",
        SimpleNamespace(
            SIGTERM=signal.SIGTERM,
            SIGINT=signal.SIGINT,
            signal=lambda signum, handler: handlers.update({signum: handler}),
        ),
    )
    monkeypatch.setattr(
        native,
        "sys",
        SimpleNamespace(
            argv=["native-worker", mode],
            stdin=io.TextIOWrapper(io.BytesIO(json.dumps(request_document()).encode())),
            stdout=output,
        ),
    )
    with deadline_scope("native fixture", 5):
        code = native.main()
    assert code == (1 if mode == "unknown" else 0)
    if mode == "request":
        assert json.loads(output.getvalue()) == {
            "status": 201,
            "body": "example-response",
        }
        assert credentials.calls == []
    elif mode == "credentials":
        assert json.loads(output.getvalue()) == {"Version": 1}
        assert opener[1] == []
    else:
        assert output.getvalue() == ""
        assert not credentials.calls and not opener[1]
    with pytest.raises(DeploymentDeadlineExceeded, match="interrupted"):
        handlers[signal.SIGTERM](signal.SIGTERM, None)


@pytest.mark.parametrize(
    "scenario",
    ["no-deadline", "oversized", "invalid-json", "expired", "transport-timeout"],
)
def test_native_main_errors_have_no_private_output(monkeypatch, opener, scenario):
    output = io.StringIO()
    raw = json.dumps(request_document()).encode()
    if scenario == "oversized":
        raw = b"x" * (native.MAX_MESSAGE_BYTES + 1)
    elif scenario == "invalid-json":
        raw = b"example-invalid"
    elif scenario == "transport-timeout":
        opener[0].error = subprocess.TimeoutExpired(
            ["fake"], 1, output="example-output"
        )
    monkeypatch.setattr(
        native,
        "signal",
        SimpleNamespace(SIGTERM=15, SIGINT=2, signal=lambda *_args: None),
    )
    monkeypatch.setattr(
        native,
        "sys",
        SimpleNamespace(
            argv=["worker", "request"],
            stdin=io.TextIOWrapper(io.BytesIO(raw)),
            stdout=output,
        ),
    )
    if scenario == "no-deadline":
        code = native.main()
    elif scenario == "expired":
        monkeypatch.setenv("GPU_FAULT_DEPLOY_DEADLINE_MONOTONIC", "1")
        code = native.main()
    else:
        with deadline_scope("worker fixture", 5):
            code = native.main()
    assert code == {"oversized": 65, "expired": 124, "transport-timeout": 124}.get(
        scenario, 1
    )
    assert output.getvalue() == ""
    if scenario != "transport-timeout":
        assert opener[1] == []


@pytest.mark.parametrize("document", [None, [], 1, "example", True])
def test_native_message_decoder_requires_mapping(document):
    with pytest.raises(native.NativeHttpError, match="invalid message"):
        native.decode_message(json.dumps(document))


def test_native_messages_enforce_encoded_size():
    with pytest.raises(HttpResponseTooLarge):
        native.encode_message({"body": "x" * native.MAX_MESSAGE_BYTES})
    with pytest.raises(HttpResponseTooLarge):
        native.decode_message("x" * (native.MAX_MESSAGE_BYTES + 1))


@pytest.mark.parametrize(
    "response",
    [
        {"status": True, "body": ""},
        {"status": 99, "body": ""},
        {"status": 600, "body": ""},
        {"status": 200, "body": []},
    ],
)
def test_parent_validates_http_worker_response(monkeypatch, response):
    monkeypatch.setattr(
        execution,
        "run_command",
        lambda command, **_kwargs: subprocess.CompletedProcess(
            command, 0, json.dumps(response), ""
        ),
    )
    with pytest.raises(native.NativeHttpError, match="invalid response"):
        native.http_request("GET", "https://example.invalid", {}, None)


@pytest.mark.parametrize(
    "changes",
    [
        {"Version": True},
        {"Version": 2},
        {"AccessKeyId": ""},
        {"AccessKeyId": 1},
        {"SecretAccessKey": ""},
        {"SecretAccessKey": None},
        {"SessionToken": 5},
    ],
)
def test_export_rejects_invalid_credential_shapes(monkeypatch, changes):
    response = {
        "Version": 1,
        "AccessKeyId": "example-access",
        "SecretAccessKey": "example-secret",
        **changes,
    }
    monkeypatch.setattr(
        execution,
        "run_command",
        lambda command, **_kwargs: subprocess.CompletedProcess(
            command, 0, json.dumps(response), ""
        ),
    )
    with pytest.raises(native.NativeHttpError, match="invalid response"):
        native.export_aws_credentials()

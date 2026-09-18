from __future__ import annotations

import base64
import contextlib
import json
import os
import ssl
import subprocess
import sys
import threading
import time
import traceback
import urllib.parse
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from gpu_fault.admin import api_budget, execution, grafana, native_http
from gpu_fault.admin.bootstrap_common import BootstrapError
from gpu_fault.admin.deadlines import DeploymentDeadlineExceeded, HttpResponseTooLarge
from gpu_fault.admin.process_supervisor import run_owned_command
from gpu_fault_release import regional_adot_self_metrics as adot
from gpu_fault_release.regional_release_config import ReleaseError

PRIVATE_MARKER = "fixture-private-http-value"


@contextlib.contextmanager
def server_fixture(mode: str) -> Iterator[tuple[str, list[dict[str, Any]]]]:
    stopped = threading.Event()
    requests: list[dict[str, Any]] = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            self.respond()

        def do_POST(self) -> None:
            self.respond()

        def respond(self) -> None:
            requests.append(
                {
                    "headers": dict(self.headers),
                    "body": self.rfile.read(
                        int(self.headers.get("Content-Length", "0"))
                    ),
                }
            )
            try:
                if mode == "headers":
                    self.wfile.write(b"HTTP/1.1 200 OK\r\nX-Drip: ")
                else:
                    self.send_response(503 if mode == "large-error" else 200)
                    if mode in {"trailers", "chunk-size"}:
                        self.send_header("Transfer-Encoding", "chunked")
                    self.end_headers()
                if mode in {"headers", "trailers", "chunk-size"}:
                    if mode == "trailers":
                        self.wfile.write(b"0\r\n")
                    elif mode == "chunk-size":
                        self.wfile.write(b"1;extension=")
                    while not stopped.is_set():
                        self.wfile.write(
                            b"X-Trailer: x\r\n" if mode == "trailers" else b"x"
                        )
                        self.wfile.flush()
                        stopped.wait(0.01)
                elif mode.startswith("large"):
                    chunk = (PRIVATE_MARKER.encode() * 4096)[:65536]
                    remaining = native_http.MAX_MESSAGE_BYTES + 1
                    while remaining and not stopped.is_set():
                        piece = chunk[:remaining]
                        self.wfile.write(piece)
                        remaining -= len(piece)
                else:
                    self.wfile.write(b'{"status":"success","data":{"result":[]}}')
            except OSError:
                pass

        def log_message(self, _format: str, *args: object) -> None:
            pass

    with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
        thread = threading.Thread(
            target=server.serve_forever, kwargs={"poll_interval": 0.01}
        )
        thread.start()
        try:
            yield f"http://127.0.0.1:{server.server_port}/fixture", requests
        finally:
            stopped.set()
            server.shutdown()
            thread.join(timeout=2)
            assert not thread.is_alive(), "owned loopback server did not stop"


@pytest.mark.parametrize("mode", ["headers", "chunk-size", "trailers"])
def test_supervised_http_stops_inside_framing_reads(mode: str) -> None:
    with api_budget.deployment_api_budget(), server_fixture(mode) as (url, requests):
        started = time.monotonic()
        with execution.deadline_scope("isolated framing", 0.6):
            with pytest.raises(DeploymentDeadlineExceeded, match="deadline"):
                grafana.urllib_transport("GET", url, {}, None)
        assert requests, "loopback fixture never entered the blocking HTTP parser"
        assert time.monotonic() - started < 2, (
            "HTTP framing outlived the task and supervised cleanup budget"
        )


def test_request_headers_and_body_travel_only_over_private_stdin(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    calls: list[tuple[list[str], dict[str, Any]]] = []

    def tracked(
        arguments: list[str], **kwargs: Any
    ) -> subprocess.CompletedProcess[str]:
        calls.append((arguments, kwargs))
        return run_owned_command(arguments, **kwargs)

    monkeypatch.setattr(execution, "run_owned_command", tracked)
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", PRIVATE_MARKER)
    monkeypatch.setenv("GPU_FAULT_EXECUTION_TOKEN", PRIVATE_MARKER)
    with api_budget.deployment_api_budget(), server_fixture("ok") as (url, requests):
        result = grafana.urllib_transport(
            "POST", url, {"Authorization": PRIVATE_MARKER}, PRIVATE_MARKER.encode()
        )
        assert result.status == 200, "valid isolated HTTP request failed"
    assert requests[0]["headers"]["Authorization"] == PRIVATE_MARKER
    assert requests[0]["body"] == PRIVATE_MARKER.encode()
    arguments, options = calls[0]
    assert PRIVATE_MARKER not in str(arguments), "request credentials entered argv"
    assert PRIVATE_MARKER not in str(options["environment"]), (
        "request credentials entered the HTTP worker environment"
    )
    assert PRIVATE_MARKER in options["input_text"], "request did not use private stdin"
    output = capsys.readouterr()
    assert PRIVATE_MARKER not in output.out + output.err, "request data entered logs"


def test_authenticated_proxy_configuration_is_private_stdin_not_http_worker_environment(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    calls: list[dict[str, Any]] = []

    def tracked(
        arguments: list[str], **kwargs: Any
    ) -> subprocess.CompletedProcess[str]:
        calls.append(kwargs)
        return run_owned_command(arguments, **kwargs)

    monkeypatch.setattr(execution, "run_owned_command", tracked)
    with server_fixture("ok") as (proxy, requests):
        authority = urllib.parse.urlsplit(proxy).netloc
        proxy_url = f"http://fixture-user:{PRIVATE_MARKER}@{authority}"
        monkeypatch.setenv("HTTP_PROXY", proxy_url)
        monkeypatch.setenv("NO_PROXY", "")
        result = grafana.urllib_transport(
            "GET", "http://origin.invalid/fixture", {}, None
        )
    assert result.status == 200, "HTTP helper did not use the configured proxy"
    headers = {name.lower(): value for name, value in requests[0]["headers"].items()}
    expected = base64.b64encode(f"fixture-user:{PRIVATE_MARKER}".encode()).decode()
    assert headers["proxy-authorization"] == "Basic " + expected, (
        "proxy authentication did not reach the loopback proxy"
    )
    assert json.loads(calls[0]["input_text"])["proxies"]["http"] == proxy_url, (
        "proxy configuration was not carried in private stdin"
    )
    assert PRIVATE_MARKER not in str(calls[0]["environment"]), (
        "proxy credentials entered the HTTP worker environment"
    )
    output = capsys.readouterr()
    assert PRIVATE_MARKER not in output.out + output.err, (
        "proxy credentials entered logs"
    )


@pytest.mark.parametrize("mode", ["large", "large-error"])
def test_real_http_output_limit_discards_private_partial_responses(
    mode: str, capsys: pytest.CaptureFixture[str]
) -> None:
    with server_fixture(mode) as (url, requests):
        with pytest.raises(
            BootstrapError, match="response exceeds its size limit"
        ) as error:
            grafana.urllib_transport("GET", url, {}, None)
    assert requests, "loopback response-size fixture was not reached"
    assert PRIVATE_MARKER not in "".join(traceback.format_exception(error.value)), (
        "oversized response data escaped through an exception"
    )
    output = capsys.readouterr()
    assert PRIVATE_MARKER not in output.out + output.err, (
        "partial response entered logs"
    )


def test_invalid_private_headers_never_escape_in_exception_text(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with server_fixture("ok") as (url, requests):
        with pytest.raises(BootstrapError, match="HTTP request failed") as error:
            grafana.urllib_transport(
                "GET", url, {"Authorization": PRIVATE_MARKER + "\r\ninvalid"}, None
            )
    assert requests == [], "invalid headers reached the HTTP server"
    assert PRIVATE_MARKER not in "".join(traceback.format_exception(error.value)), (
        "private request headers escaped through a transport exception"
    )
    output = capsys.readouterr()
    assert PRIVATE_MARKER not in output.out + output.err, "private header entered logs"


def test_https_worker_rejects_an_untrusted_loopback_certificate(tmp_path: Path) -> None:
    key, certificate = tmp_path / "key.pem", tmp_path / "certificate.pem"
    subprocess.run(
        [
            "openssl",
            "req",
            "-x509",
            "-newkey",
            "rsa:2048",
            "-nodes",
            "-keyout",
            str(key),
            "-out",
            str(certificate),
            "-days",
            "1",
            "-subj",
            "/CN=localhost",
        ],
        capture_output=True,
        check=True,
        timeout=10,
    )
    received: list[bool] = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            received.append(True)
            self.send_response(200)
            self.end_headers()

        def log_message(self, _format: str, *args: object) -> None:
            pass

    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(certificate, key)
    with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
        server.socket = context.wrap_socket(server.socket, server_side=True)
        thread = threading.Thread(
            target=server.serve_forever, kwargs={"poll_interval": 0.01}
        )
        thread.start()
        try:
            with pytest.raises(BootstrapError, match="HTTP request failed"):
                grafana.urllib_transport(
                    "GET", f"https://127.0.0.1:{server.server_port}/fixture", {}, None
                )
            assert received == [], "HTTP helper disabled TLS certificate validation"
        finally:
            server.shutdown()
            thread.join(timeout=2)
            assert not thread.is_alive(), "owned TLS fixture did not stop"


@pytest.fixture
def credential_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path]:
    binary = tmp_path / "bin"
    binary.mkdir()
    provider = tmp_path / "provider.py"
    record = tmp_path / "provider.json"
    provider.write_text(
        "import json, os, sys, time\n"
        "from pathlib import Path\n"
        "record, mode = sys.argv[1:]\n"
        "start = Path(f'/proc/{os.getpid()}/stat').read_text().rsplit(')', 1)[1].split()[19]\n"
        "Path(record).write_text(json.dumps({'pid': os.getpid(), 'start': start}))\n"
        "if mode == 'hang':\n"
        "    time.sleep(60)\n"
        "elif mode == 'noisy':\n"
        f"    sys.stdout.write({PRIVATE_MARKER!r} * 400000)\n"
        "elif mode == 'malformed':\n"
        f"    print({PRIVATE_MARKER!r})\n"
        "elif mode in ('static', 'static-null'):\n"
        "    value = {'Version': 1, 'AccessKeyId': 'fixture-access',\n"
        f"        'SecretAccessKey': {PRIVATE_MARKER!r}}}\n"
        "    if mode == 'static-null':\n"
        "        value['SessionToken'] = None\n"
        "    print(json.dumps(value))\n"
        "else:\n"
        "    print(json.dumps({'Version': 1, 'AccessKeyId': 'fixture-access',\n"
        f"        'SecretAccessKey': {PRIVATE_MARKER!r}, 'SessionToken': 'fixture-session'}}))\n"
    )
    aws = binary / "aws"
    aws.write_text(
        f"#!{sys.executable}\n"
        "import configparser, os, shlex, subprocess, sys\n"
        "if sys.argv[1:] != ['configure', 'export-credentials', '--format', 'process']:\n"
        "    raise SystemExit('unexpected fixture AWS command')\n"
        "configuration = configparser.RawConfigParser()\n"
        "configuration.read(os.environ['AWS_CONFIG_FILE'])\n"
        "with subprocess.Popen(shlex.split(configuration['default']['credential_process']),\n"
        "    stdout=subprocess.PIPE, stderr=subprocess.PIPE) as provider:\n"
        "    output, errors = provider.communicate()\n"
        "    sys.stdout.buffer.write(output)\n"
        "    sys.stderr.buffer.write(errors)\n"
        "    raise SystemExit(provider.returncode)\n"
    )
    aws.chmod(0o700)
    configuration = tmp_path / "aws-config"
    monkeypatch.setenv("PATH", str(binary) + os.pathsep + os.environ["PATH"])
    monkeypatch.setenv("AWS_CONFIG_FILE", str(configuration))
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", "/dev/null")
    monkeypatch.setenv("AWS_EC2_METADATA_DISABLED", "true")
    monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
    monkeypatch.delenv("AWS_SECRET_ACCESS_KEY", raising=False)
    monkeypatch.delenv("AWS_SESSION_TOKEN", raising=False)
    return provider, record


def configure_provider(
    provider: Path, record: Path, mode: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    import shlex

    configuration = provider.parent / "aws-config"
    configuration.write_text(
        "[default]\ncredential_process = "
        + shlex.join([sys.executable, str(provider), str(record), mode])
        + "\n"
    )
    configuration.chmod(0o600)
    monkeypatch.setenv("AWS_CONFIG_FILE", str(configuration))


def amp_query() -> list[dict[str, Any]]:
    return adot.amp_instant_query(
        SimpleNamespace(
            config=SimpleNamespace(
                aws_region="us-west-2",
                health=SimpleNamespace(amp_workspace_id="fixture-workspace"),
            )
        ),
        "up",
    )


def test_hung_credential_process_is_reaped_before_http_can_start(
    credential_provider: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    provider, record = credential_provider
    configure_provider(provider, record, "hang", monkeypatch)
    requests: list[object] = []
    monkeypatch.setattr(
        adot, "http_request", lambda *args, **kwargs: requests.append(args)
    )
    with api_budget.deployment_api_budget():
        started = time.monotonic()
        with execution.deadline_scope("hung credential process", 1):
            with pytest.raises(DeploymentDeadlineExceeded, match="deadline"):
                amp_query()
        assert record.is_file(), "the owned credential process never started"
        identity = json.loads(record.read_text())
        stat = Path(f"/proc/{identity['pid']}/stat")
        if stat.exists():
            actual = stat.read_text().rsplit(")", 1)[1].split()
            assert actual[19] != identity["start"], (
                "supervised credential resolution left its provider alive"
            )
        assert time.monotonic() - started < 3, (
            "credential resolution ignored its deadline"
        )
        assert requests == [], "AMP started HTTP after credential resolution timed out"


@pytest.mark.parametrize("mode", ["noisy", "malformed"])
def test_credential_capture_errors_are_bounded_and_opaque(
    credential_provider: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    mode: str,
) -> None:
    provider, record = credential_provider
    configure_provider(provider, record, mode, monkeypatch)
    requests: list[object] = []
    monkeypatch.setattr(
        adot, "http_request", lambda *args, **kwargs: requests.append(args)
    )
    with api_budget.deployment_api_budget():
        with pytest.raises(
            HttpResponseTooLarge if mode == "noisy" else ReleaseError
        ) as error:
            amp_query()
    assert record.is_file(), "credential fixture was not invoked through the CLI shim"
    assert requests == [], "invalid credentials reached the HTTP request"
    assert PRIVATE_MARKER not in "".join(traceback.format_exception(error.value)), (
        "private credential output escaped through the error chain"
    )
    output = capsys.readouterr()
    assert PRIVATE_MARKER not in output.out + output.err, (
        "credential output entered logs"
    )


@pytest.mark.parametrize("mode", ["ok", "static", "static-null"])
def test_amp_signs_frozen_exported_credentials_before_http_admission(
    credential_provider: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    provider, record = credential_provider
    configure_provider(provider, record, mode, monkeypatch)
    original = native_http.http_request
    with api_budget.deployment_api_budget(), server_fixture("ok") as (url, requests):

        def loopback(
            method: str, _url: str, headers: dict[str, str], body: bytes, **kwargs: Any
        ) -> tuple[int, str]:
            assert record.is_file(), "HTTP admission preceded credential resolution"
            return original(method, url, headers, body, **kwargs)

        monkeypatch.setattr(adot, "http_request", loopback)
        assert amp_query() == [], "AMP response vector changed at the helper boundary"
        statistics = api_budget.statistics()
        backends = cast(dict[str, dict[str, object]], statistics["backends"])
        assert backends["aws"]["finished_commands"] == 2, (
            "credential export and HTTP did not use separate bounded admissions"
        )
    assert requests[0]["headers"]["Authorization"].startswith("AWS4-HMAC-SHA256 "), (
        "exported frozen credentials were not used for SigV4"
    )
    headers = {name.lower(): value for name, value in requests[0]["headers"].items()}
    assert ("x-amz-security-token" in headers) == (mode == "ok"), (
        "static credentials fabricated a session token or session credentials lost it"
    )


@pytest.mark.allows_cluster_binaries("aws")
@pytest.mark.parametrize("mode", ["ok", "static", "static-null"])
def test_public_credential_worker_captures_cli_output_without_a_native_provider(
    credential_provider: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    provider, record = credential_provider
    configure_provider(provider, record, mode, monkeypatch)
    with (
        api_budget.deployment_api_budget(),
        execution.deadline_scope("worker export", 5),
    ):
        assert api_budget.resolve_tool("aws") == str(provider.parent / "bin" / "aws"), (
            "credential worker fixture did not bind the fake AWS executable"
        )
        document = native_http.worker_credentials()
    assert document["Version"] == 1, (
        "credential worker changed the process-format version"
    )
    assert document["AccessKeyId"] == "fixture-access", (
        "credential worker changed identity"
    )
    assert document["SecretAccessKey"] == PRIVATE_MARKER, (
        "credential worker lost private output"
    )
    assert bool(document.get("SessionToken")) == (mode == "ok"), (
        "credential worker changed optional session-token semantics"
    )

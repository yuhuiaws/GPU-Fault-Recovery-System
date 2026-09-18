"""Private stdio transport for supervised HTTP and AWS credential resolution."""

from __future__ import annotations

import base64
import json
import os
import selectors
import signal
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping
from pathlib import Path
from typing import Literal, cast

if not __package__:
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from gpu_fault.admin.api_budget import ApiBudgetError, PARENT_ENV, api_slot
from gpu_fault.admin.deadlines import (
    MAX_HTTP_RESPONSE_BYTES,
    DeploymentDeadlineExceeded,
    HttpResponseTooLarge,
    current_deadline,
    deadline_scope,
    read_http_response,
    recovery_active,
    remaining_timeout,
)

MAX_MESSAGE_BYTES = MAX_HTTP_RESPONSE_BYTES
HTTP_SECONDS = 30
_TIMEOUT_EXIT = 124
_OVERSIZE_EXIT = 65
_WorkerMode = Literal["request", "credentials"]


class NativeHttpError(RuntimeError):
    """Captured payloads are never attached to this error."""


def encode_message(document: object) -> str:
    text = json.dumps(document, ensure_ascii=True, separators=(",", ":"))
    if len(text) > MAX_MESSAGE_BYTES:
        raise HttpResponseTooLarge("native HTTP message exceeds its size limit")
    return text


def decode_message(text: str) -> dict[str, object]:
    if len(text.encode("utf-8")) > MAX_MESSAGE_BYTES:
        raise HttpResponseTooLarge("native HTTP message exceeds its size limit")
    try:
        document = json.loads(text)
    except (ValueError, UnicodeError):
        document = None
    if not isinstance(document, dict):
        raise NativeHttpError("native HTTP helper returned an invalid message")
    return cast(dict[str, object], document)


def _worker_environment(*, credentials: bool) -> dict[str, str]:
    names = {
        "HOME",
        "PATH",
        "LANG",
        "LC_ALL",
        "SSL_CERT_FILE",
        "SSL_CERT_DIR",
        PARENT_ENV,
    }
    if credentials:
        # Only the existing CLI credential chain receives AWS environment inputs.
        # Newly exported credentials are never copied into any environment.
        names.update(name for name in os.environ if name.startswith("AWS_"))
        names.update(
            name
            for name in os.environ
            if name.lower() in {"http_proxy", "https_proxy", "all_proxy", "no_proxy"}
        )
    else:
        names.update({"NO_PROXY", "no_proxy"})
    return {name: os.environ[name] for name in names if name in os.environ}


def _run_worker(
    mode: _WorkerMode, payload: object, *, seconds: float
) -> dict[str, object]:
    from gpu_fault.admin.execution import run_command
    from gpu_fault.admin.process_supervisor import ensure_supervision_safe

    ensure_supervision_safe(allow_interrupted=recovery_active())
    deadline = current_deadline()
    label = deadline.label if deadline is not None else f"native {mode}"
    result: subprocess.CompletedProcess[str] | None = None
    timed_out = False
    try:
        result = run_command(
            [sys.executable, "-I", "-S", "-B", str(Path(__file__).resolve()), mode],
            input_text=encode_message(payload),
            capture=True,
            environment=_worker_environment(credentials=mode == "credentials"),
            timeout_seconds=remaining_timeout(seconds),
        )
    except (subprocess.TimeoutExpired, DeploymentDeadlineExceeded):
        timed_out = True
    except HttpResponseTooLarge:
        raise
    except ApiBudgetError:
        raise
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError):
        pass
    # Raise outside the handler so private subprocess output is not retained
    # through exception chaining, including partial output on timeout.
    if timed_out or result is not None and result.returncode == _TIMEOUT_EXIT:
        raise DeploymentDeadlineExceeded(f"deployment deadline exceeded: {label}")
    if result is not None and result.returncode == _OVERSIZE_EXIT:
        raise HttpResponseTooLarge("native HTTP response exceeds its size limit")
    if result is None or result.returncode:
        raise NativeHttpError(f"native {mode} helper failed")
    ensure_supervision_safe(allow_interrupted=recovery_active())
    remaining_timeout(seconds)
    return decode_message(result.stdout)


def http_request(
    method: str,
    url: str,
    headers: Mapping[str, str],
    body: bytes | None,
    *,
    backend: Literal["http", "aws"] = "http",
    seconds: float = HTTP_SECONDS,
    label: str = "native HTTP request",
) -> tuple[int, str]:
    with deadline_scope(label, seconds):
        response = _run_worker(
            "request",
            {
                "backend": backend,
                "method": method,
                "url": url,
                "headers": dict(headers),
                "body": base64.b64encode(body).decode("ascii")
                if body is not None
                else None,
                "proxies": urllib.request.getproxies(),
            },
            seconds=seconds,
        )
    status, text = response.get("status"), response.get("body")
    if type(status) is not int or not 100 <= status <= 599 or not isinstance(text, str):
        raise NativeHttpError("native HTTP helper returned an invalid response")
    return status, text


def export_aws_credentials(*, seconds: float = HTTP_SECONDS) -> dict[str, str]:
    with deadline_scope("AWS credential resolution", seconds):
        response = _run_worker("credentials", {}, seconds=seconds)
    access = response.get("AccessKeyId")
    secret = response.get("SecretAccessKey")
    token = response.get("SessionToken")
    if (
        type(response.get("Version")) is not int
        or response["Version"] != 1
        or not isinstance(access, str)
        or not access
        or not isinstance(secret, str)
        or not secret
        or token is not None
        and not isinstance(token, str)
    ):
        raise NativeHttpError("AWS credential export returned an invalid response")
    return {
        "AccessKeyId": access,
        "SecretAccessKey": secret,
        "SessionToken": token or "",
    }


def _string_mapping(value: object) -> dict[str, str]:
    if not isinstance(value, dict) or any(
        not isinstance(key, str) or not isinstance(item, str)
        for key, item in value.items()
    ):
        raise NativeHttpError("native HTTP request has an invalid mapping")
    return cast(dict[str, str], value)


def worker_request(document: Mapping[str, object]) -> dict[str, object]:
    """Read one response while the actual worker owns its admission lease."""
    backend, method, url = (
        document.get("backend"),
        document.get("method"),
        document.get("url"),
    )
    if (
        backend not in {"http", "aws"}
        or not isinstance(method, str)
        or not isinstance(url, str)
    ):
        raise NativeHttpError("native HTTP request identity is invalid")
    if urllib.parse.urlsplit(url).scheme not in {"http", "https"}:
        raise NativeHttpError("native HTTP request scheme is invalid")
    headers = _string_mapping(document.get("headers"))
    proxies = _string_mapping(document.get("proxies"))
    encoded = document.get("body")
    if encoded is not None and not isinstance(encoded, str):
        raise NativeHttpError("native HTTP request body is invalid")
    body = base64.b64decode(encoded, validate=True) if encoded is not None else None
    request = urllib.request.Request(url, data=body, method=method, headers=headers)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler(proxies))
    with api_slot(str(backend), parent_id=os.environ.get(PARENT_ENV)):
        try:
            with opener.open(
                request, timeout=remaining_timeout(HTTP_SECONDS)
            ) as response:
                status = int(response.status)
                content = read_http_response(response)
        except urllib.error.HTTPError as error:
            with error:
                status = int(error.code)
                content = read_http_response(error)
        remaining_timeout(HTTP_SECONDS)
    return {"status": status, "body": content.decode("utf-8", "replace")}


def worker_credentials() -> dict[str, object]:
    """Drain private AWS CLI output without allowing a credential helper to hang."""
    process = subprocess.Popen(
        ["aws", "configure", "export-credentials", "--format", "process"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env={**os.environ, "AWS_PAGER": "", "AWS_CLI_AUTO_PROMPT": "off"},
    )
    output = bytearray()
    size = 0
    try:
        if process.stdout is None or process.stderr is None:
            raise NativeHttpError("AWS credential export has no output pipes")
        with selectors.DefaultSelector() as selector:
            for stream in (process.stdout, process.stderr):
                os.set_blocking(stream.fileno(), False)
                selector.register(stream, selectors.EVENT_READ)
            while selector.get_map():
                remaining = remaining_timeout(HTTP_SECONDS)
                for event, _mask in selector.select(timeout=min(0.1, remaining)):
                    try:
                        chunk = os.read(event.fd, 65536)
                    except BlockingIOError:
                        continue
                    if not chunk:
                        selector.unregister(event.fileobj)
                        continue
                    size += len(chunk)
                    if size > MAX_MESSAGE_BYTES:
                        raise HttpResponseTooLarge(
                            "AWS credential output exceeds its size limit"
                        )
                    if event.fileobj is process.stdout:
                        output.extend(chunk)
        code = process.wait(timeout=remaining_timeout(HTTP_SECONDS))
        if code:
            raise NativeHttpError("AWS credential export failed")
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=0.25)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=0.25)
        for cleanup_stream in (process.stdout, process.stderr):
            if cleanup_stream is not None:
                cleanup_stream.close()
    return decode_message(output.decode("utf-8"))


def main() -> int:
    # Signals are confined to this single-threaded, independently supervised worker.
    def interrupted(_signum: int, _frame: object) -> None:
        raise DeploymentDeadlineExceeded("native worker interrupted")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        if current_deadline() is None:
            raise NativeHttpError("native HTTP helper requires an inherited deadline")
        remaining_timeout(HTTP_SECONDS)
        raw = sys.stdin.buffer.read(MAX_MESSAGE_BYTES + 1)
        if len(raw) > MAX_MESSAGE_BYTES:
            raise HttpResponseTooLarge("native HTTP request exceeds its size limit")
        document = decode_message(raw.decode("utf-8"))
        if sys.argv[1:] == ["request"]:
            response = worker_request(document)
        elif sys.argv[1:] == ["credentials"]:
            response = worker_credentials()
        else:
            raise NativeHttpError("native HTTP helper mode is invalid")
        sys.stdout.write(encode_message(response))
        return 0
    except (DeploymentDeadlineExceeded, subprocess.TimeoutExpired):
        return _TIMEOUT_EXIT
    except HttpResponseTooLarge:
        return _OVERSIZE_EXIT
    except Exception:
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

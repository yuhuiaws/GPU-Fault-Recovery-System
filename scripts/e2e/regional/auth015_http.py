"""Bounded private HTTP worker, usable on deploy hosts and inside CPU Pods."""

from __future__ import annotations

import base64
import http.client
import ipaddress
import json
import math
import os
import signal
import ssl
import subprocess
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

MAX_REQUEST_BYTES = 1024 * 1024
MAX_RESPONSE_BYTES = 1024 * 1024


class BoundedHttpError(RuntimeError):
    pass


class HttpDeadlineExceeded(BoundedHttpError):
    pass


class HttpResponseTooLarge(BoundedHttpError):
    pass


def worker_source() -> str:
    embedded = globals().get("AUTH015_HTTP_SOURCE")
    return (
        embedded
        if isinstance(embedded, str)
        else Path(__file__).read_text(encoding="utf-8")
    )


def worker_timeout(document: dict[str, Any]) -> float:
    seconds = document["seconds"]
    if (
        type(seconds) not in {int, float}
        or not math.isfinite(seconds)
        or not 0 < seconds <= 30
    ):
        raise BoundedHttpError("HTTP worker timeout is invalid")
    return float(seconds)


@contextmanager
def worker_deadline(seconds: float) -> Iterator[None]:
    previous = signal.getsignal(signal.SIGALRM)
    if previous is None or signal.getitimer(signal.ITIMER_REAL) != (0.0, 0.0):
        raise BoundedHttpError("HTTP worker alarm state is unavailable")
    # A lost parent cannot cancel this deadline. _exit also avoids blocked or
    # broken private output pipes; the kernel closes the sole worker's sockets.
    signal.signal(signal.SIGALRM, lambda _signum, _frame: os._exit(124))
    try:
        signal.setitimer(signal.ITIMER_REAL, seconds)
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def worker_exchange(document: dict[str, Any]) -> dict[str, Any]:
    url, method, headers = document["url"], document["method"], document["headers"]
    seconds, limit = worker_timeout(document), document["limit"]
    if (
        not isinstance(url, str)
        or any(ord(character) < 32 for character in url)
        or method not in {"GET", "POST"}
        or not isinstance(headers, dict)
        or any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in headers.items()
        )
        or type(limit) is not int
        or not 0 < limit <= MAX_RESPONSE_BYTES
    ):
        raise BoundedHttpError("HTTP worker request is invalid")
    parts = urlsplit(url)
    if (
        parts.scheme not in {"http", "https"}
        or parts.username
        or parts.password
        or parts.fragment
    ):
        raise BoundedHttpError("HTTP worker target is invalid")
    address = ipaddress.ip_address(parts.hostname or "")
    port = (
        parts.port
        if parts.port is not None
        else (443 if parts.scheme == "https" else 80)
    )
    certificate = document.get("certificate")
    connection: http.client.HTTPConnection
    if parts.scheme == "https":
        if not isinstance(certificate, str) or not certificate:
            raise BoundedHttpError("HTTP worker requires a pinned TLS certificate")
        context = ssl.create_default_context(cadata=certificate)
        connection = http.client.HTTPSConnection(
            str(address), port, timeout=seconds, context=context
        )
    else:
        if not address.is_loopback:
            raise BoundedHttpError("cleartext HTTP is restricted to loopback")
        connection = http.client.HTTPConnection(str(address), port, timeout=seconds)
    encoded = document.get("body")
    body = None if encoded is None else base64.b64decode(encoded, validate=True)
    path = (parts.path or "/") + ("?" + parts.query if parts.query else "")
    try:
        connection.request(method, path, body=body, headers=headers)
        with connection.getresponse() as response:
            # http.client returns a streaming response; no pooled pre-buffer is used.
            content = response.read(limit + 1)
            if len(content) > limit:
                raise HttpResponseTooLarge("HTTP response exceeds its byte limit")
            return {
                "status": response.status,
                "body": base64.b64encode(content).decode("ascii"),
            }
    finally:
        connection.close()


def worker_main() -> int:
    try:
        raw = sys.stdin.buffer.read(MAX_REQUEST_BYTES + 1)
        if len(raw) > MAX_REQUEST_BYTES:
            raise BoundedHttpError("HTTP worker input exceeds its byte limit")
        document = json.loads(raw)
        if not isinstance(document, dict):
            raise BoundedHttpError("HTTP worker input is invalid")
        with worker_deadline(worker_timeout(document)):
            result = worker_exchange(document)
    except HttpResponseTooLarge:
        print('{"error":"body_limit"}')
        return 65
    except TimeoutError:
        print('{"error":"timeout"}')
        return 124
    except Exception:
        # URLs, request headers and response bodies are private stdio only.
        print('{"error":"transport"}')
        return 1
    print(json.dumps(result, separators=(",", ":")))
    return 0


def bounded_request(
    url: str,
    *,
    method: str = "GET",
    headers: dict[str, str] | None = None,
    data: bytes | None = None,
    certificate: str | None = None,
    timeout: float = 5,
    max_response_bytes: int = 8192,
) -> tuple[int, bytes]:
    if (
        type(timeout) not in {int, float}
        or not math.isfinite(timeout)
        or not 0 < timeout <= 30
        or type(max_response_bytes) is not int
        or not 0 < max_response_bytes <= MAX_RESPONSE_BYTES
    ):
        raise BoundedHttpError("HTTP worker limits are invalid")
    payload = json.dumps(
        {
            "url": url,
            "method": method,
            "headers": headers or {},
            "body": base64.b64encode(data).decode("ascii")
            if data is not None
            else None,
            "certificate": certificate,
            "seconds": timeout,
            "limit": max_response_bytes,
        },
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("ascii")
    if len(payload) > MAX_REQUEST_BYTES:
        raise BoundedHttpError("HTTP worker input exceeds its byte limit")
    try:
        completed = subprocess.run(
            [sys.executable, "-I", "-S", "-B", "-c", worker_source()],
            input=payload,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env={
                "HOME": "/tmp",
                "PATH": os.defpath,
                "AWS_CONFIG_FILE": os.devnull,
                "AWS_SHARED_CREDENTIALS_FILE": os.devnull,
                "AWS_EC2_METADATA_DISABLED": "true",
                "KUBECONFIG": os.devnull,
            },
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        # run() kills and waits for this sole worker; it never starts descendants.
        raise HttpDeadlineExceeded("HTTP worker deadline expired") from None
    except (OSError, subprocess.SubprocessError):
        raise BoundedHttpError("HTTP worker could not complete") from None
    if completed.returncode == 124:
        raise HttpDeadlineExceeded("HTTP worker deadline expired")
    if completed.returncode == 65:
        raise HttpResponseTooLarge("HTTP response exceeds its byte limit")
    if completed.returncode:
        raise BoundedHttpError("HTTP worker could not complete")
    try:
        if len(completed.stdout) > 4 * ((max_response_bytes + 2) // 3) + 256:
            raise ValueError
        result = json.loads(completed.stdout)
        if (
            not isinstance(result, dict)
            or set(result) != {"status", "body"}
            or type(result["status"]) is not int
            or not 100 <= result["status"] <= 599
            or not isinstance(result["body"], str)
        ):
            raise ValueError
        body = base64.b64decode(result["body"], validate=True)
        if len(body) > max_response_bytes:
            raise ValueError
    except (TypeError, ValueError):
        raise BoundedHttpError(
            "HTTP worker returned an invalid bounded response"
        ) from None
    return result["status"], body


if __name__ == "__main__":
    raise SystemExit(worker_main())

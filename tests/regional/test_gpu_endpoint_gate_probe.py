"""The GPU endpoint gate keeps retrying on resolver failures, request by request.

Live 2026-09-13: right after the join associated the GPU VPC with the private
hosted zone, one CoreDNS replica already resolved the control-plane name while
another still served the cached NXDOMAIN. The probe's resolve loop passed and
its very next request failed "Name or service not known", which rolled the join
back. Every request now retries on a resolver failure until the DNS wait ends.
"""

from __future__ import annotations

import json
import pathlib
import socket
import ssl
import time
import urllib.error
import urllib.request

import pytest

from gpu_fault_release import regional_release_probes as PROBES


class _Response:
    status = 200

    def __enter__(self) -> "_Response":
        return self

    def __exit__(self, *_exc: object) -> None:
        return None


def _run_probe(
    monkeypatch,
    *,
    resolver_failures: int,
    request_failures: int,
    request_error: Exception | None = None,
) -> dict:
    # Read the probe before Path.read_text is stubbed for the mounted files.
    source = compile(
        PROBES.probe_source("gpu_endpoint_gate"), "gpu_endpoint_gate", "exec"
    )
    lookups: list[int] = [0]
    requests: list[int] = [0]

    def getaddrinfo(host, port, *_args):
        lookups[0] += 1
        if lookups[0] <= resolver_failures:
            raise socket.gaierror(-2, "Name or service not known")
        return [(None, None, None, None, ("10.0.0.7", port))]

    def urlopen(target, *, context=None, timeout=None):
        requests[0] += 1
        if requests[0] <= request_failures:
            raise urllib.error.URLError(
                request_error
                if request_error is not None
                else socket.gaierror(-2, "Name or service not known")
            )
        return _Response()

    clock = [0.0]
    monkeypatch.setattr(socket, "getaddrinfo", getaddrinfo)
    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(ssl, "create_default_context", lambda cafile=None: object())
    monkeypatch.setattr(pathlib.Path, "read_text", lambda self, *a, **k: "value")
    monkeypatch.setattr(
        time, "sleep", lambda seconds: clock.__setitem__(0, clock[0] + seconds)
    )
    monkeypatch.setattr(time, "monotonic", lambda: clock[0])
    monkeypatch.setenv("CONTROL_PLANE_URL", "https://api.example.internal")
    monkeypatch.setenv("EXPECTED_HOSTNAME", "api.example.internal")
    monkeypatch.setenv("PROBE_INCIDENT_ID", "incident-a")
    monkeypatch.setenv("DNS_RESOLVE_WAIT_SECONDS", "60")
    printed: list[str] = []
    namespace = {
        "__name__": "__main__",
        "print": lambda *a, **k: printed.append(" ".join(map(str, a))),
    }
    exec(source, namespace)
    return {
        "output": json.loads(printed[-1]),
        "lookups": lookups[0],
        "requests": requests[0],
    }


def test_a_request_that_fails_to_resolve_after_the_lookup_passed_is_retried(
    monkeypatch,
) -> None:
    result = _run_probe(monkeypatch, resolver_failures=2, request_failures=1)

    assert result["output"]["cluster_token"] == "accepted", "the gate passed"
    assert result["lookups"] == 3, "the lookup retried past two NXDOMAIN answers"
    assert result["requests"] == 3, (
        "healthz retried once after its own resolver failure, then the token probe ran"
    )


def test_a_name_that_never_resolves_still_fails_when_the_wait_runs_out(
    monkeypatch,
) -> None:
    with pytest.raises(urllib.error.URLError):
        # 60 s wait, 5 s per retry: the lookup passes at once, healthz never resolves.
        _run_probe(monkeypatch, resolver_failures=0, request_failures=100)


def test_a_handshake_that_times_out_on_a_brand_new_nlb_is_retried(monkeypatch) -> None:
    """Live 2026-09-13 (fresh bootstrap, NLB created five minutes earlier): the
    name resolved and the targets were healthy, but the TLS handshake through the
    new NLB timed out -- the load balancer accepted the connection before it
    forwarded anything. That is "not there yet", the same as an unresolved name."""

    result = _run_probe(
        monkeypatch,
        resolver_failures=0,
        request_failures=2,
        request_error=TimeoutError("_ssl.c:981: The handshake operation timed out"),
    )

    assert result["output"]["cluster_token"] == "accepted", "the gate passed"
    assert result["requests"] == 4, "two timeouts were retried before healthz answered"


def test_a_rejected_certificate_is_never_retried(monkeypatch) -> None:
    with pytest.raises(urllib.error.URLError, match="certificate verify failed"):
        _run_probe(
            monkeypatch,
            resolver_failures=0,
            request_failures=100,
            request_error=ssl.SSLCertVerificationError("certificate verify failed"),
        )

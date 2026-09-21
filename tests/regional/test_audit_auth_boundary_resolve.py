"""The AUTH matrix can reach the control plane through a pinned local forward."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import audit_auth_boundary as audit


class _Response:
    def __init__(self, status: int, body: bytes) -> None:
        self.status = status
        self._body = body

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *exc: object) -> None:
        return None


@pytest.fixture
def pins() -> Any:
    yield audit.pin_resolution
    audit.pin_resolution([])


def test_pinned_host_connects_to_the_forward_with_the_production_host_header(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, pins: Any
) -> None:
    seen: list[Any] = []

    def urlopen(request: Any, context: Any = None, timeout: float = 0) -> _Response:
        seen.append((request, context))
        return _Response(403, b'{"detail": "denied", "lease_token": "x"}')

    monkeypatch.setattr(audit.urllib.request, "urlopen", urlopen)
    pins(["api.example.internal:18080:127.0.0.1"])
    result = audit.post(
        "http://api.example.internal:18080",
        tmp_path / "absent-ca.crt",
        "/v1/regional/executors/claim",
        cluster_id="cluster-a",
        token="secret",
        payload={"executor_id": "auth-probe"},
    )
    assert result == {"status": 403, "body": {"detail": "denied"}}, (
        "the pinned request must return the upstream verdict with the usual redaction"
    )
    request, context = seen[0]
    assert request.full_url == "http://127.0.0.1:18080/v1/regional/executors/claim", (
        "the connection must go to the forwarded local address"
    )
    assert request.get_header("Host") == "api.example.internal:18080", (
        "the production endpoint host must stay in the Host header"
    )
    assert request.get_header("Authorization") == "Bearer secret", (
        "pinning must not touch the credentials under test"
    )
    assert context is None, "a plaintext forward must not build a TLS context"


def test_unpinned_https_requests_keep_their_real_host(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, pins: Any
) -> None:
    seen: list[Any] = []

    def urlopen(request: Any, context: Any = None, timeout: float = 0) -> _Response:
        seen.append((request, context))
        return _Response(200, b"{}")

    ca = tmp_path / "ca.crt"
    ca.write_text("fixture CA; never parsed because the context builder is replaced")
    monkeypatch.setattr(audit.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(
        audit.ssl, "create_default_context", lambda cafile: {"cafile": cafile}
    )
    pins(["other.example.internal:443:127.0.0.1"])
    audit.get("https://api.example.internal", ca, "/healthz")
    request, context = seen[0]
    assert request.full_url == "https://api.example.internal/healthz", (
        "a host without a pin must be dialled as written"
    )
    assert request.get_header("Host") is None and context == {"cafile": str(ca)}, (
        "an unpinned TLS request keeps urllib's Host handling and a CA-backed context"
    )


@pytest.mark.parametrize(
    "rule", ["api.example.internal", "host:port:1.2.3.4", ":80:1.2.3.4"]
)
def test_malformed_pins_are_refused(rule: str, pins: Any) -> None:
    with pytest.raises(ValueError, match="HOST:PORT:ADDRESS"):
        pins([rule])


def test_pinning_a_tls_endpoint_is_refused_instead_of_sending_a_wrong_sni(
    tmp_path: Path, pins: Any
) -> None:
    pins(["api.example.internal:443:127.0.0.1"])
    with pytest.raises(ValueError, match="plaintext"):
        audit.get("https://api.example.internal", tmp_path / "ca.crt", "/healthz")

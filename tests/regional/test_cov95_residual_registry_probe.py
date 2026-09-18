"""Registry probe result/error protocol with fake HTTP and context boundaries."""

from __future__ import annotations

import io
import json
import sys
from types import SimpleNamespace
from urllib.error import HTTPError, URLError

import pytest

from scripts.e2e.regional.probes import boot023_registry_probe as probe
from tests.regional import _cov95_residual_support as support

residual_isolation = support.residual_isolation


class Response:
    def __init__(self, body):
        self.body = body
        self.status = 200
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.closed = True

    def read(self):
        return self.body


@pytest.mark.parametrize("body,payload", [(b"", {}), (b'{"ok":true}', {"ok": True})])
def test_fetch_reads_json_and_closes_successful_response(monkeypatch, body, payload):
    response = Response(body)
    calls = []
    monkeypatch.setattr(
        probe, "urlopen", lambda url, **kwargs: calls.append((url, kwargs)) or response
    )
    assert probe.fetch("http://probe.invalid/healthz") == {
        "status": 200,
        "payload": payload,
    }
    assert response.closed, "response context must close even for an empty body"
    assert calls[0][1] == {"timeout": 10}


@pytest.mark.parametrize(
    "body",
    [b"", b'{"detail":"busy"}', b"invalid" * 200, b"\xff" * 600],
    ids=["empty", "json", "non-json", "non-utf8"],
)
def test_http_error_body_is_bounded_even_when_not_utf8_json(monkeypatch, body):
    def fail(*args, **kwargs):
        raise HTTPError(
            "http://probe.invalid/healthz", 503, "busy", {}, io.BytesIO(body)
        )

    monkeypatch.setattr(probe, "urlopen", fail)
    result = probe.fetch("http://probe.invalid/healthz")
    assert result["status"] == 503
    if body.startswith(b"invalid") or body.startswith(b"\xff"):
        assert len(result["payload"]["raw"]) == 500
    else:
        assert result["payload"] == ({"detail": "busy"} if body else {})


@pytest.mark.parametrize(
    "error",
    [
        URLError("fake unavailable"),
        OSError("fake unavailable"),
        ValueError("fake invalid response"),
    ],
)
def test_transport_failure_is_an_unknown_status_not_a_health_proof(monkeypatch, error):
    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(probe, "urlopen", fail)
    result = probe.fetch("http://probe.invalid/healthz")
    assert result["status"] is None
    assert type(error).__name__ in result["error"]


def test_main_reads_the_selected_generation_and_real_registry_digest(
    monkeypatch, capsys
):
    from gpu_fault.app import ApplicationContext
    from gpu_fault.regional_registry import regional_registry_config_sha256

    generations = []
    digest = regional_registry_config_sha256([])
    context = SimpleNamespace(
        regional_registry_secret_sha256=digest,
        store=SimpleNamespace(
            get_regional_registry_head=lambda: SimpleNamespace(
                generation=7, content_sha256="a" * 64
            ),
            get_regional_registry_revision=lambda generation: generations.append(
                generation
            )
            or SimpleNamespace(registrations=[]),
        ),
    )
    monkeypatch.setattr(
        ApplicationContext, "from_environment", classmethod(lambda cls: context)
    )
    monkeypatch.setattr(probe.socket, "gethostname", lambda: "private-cpu-pod")
    monkeypatch.setattr(
        probe, "urlopen", lambda *args, **kwargs: Response(b'{"ok":true}')
    )
    monkeypatch.setattr(sys, "argv", ["registry-probe", "8181"])
    assert probe.main() == 0
    report = json.loads(capsys.readouterr().out)
    assert generations == [7]
    assert report["head_generation"] == 7
    assert report["secret_config_sha256"] == report["durable_config_sha256"] == digest
    assert report["hostname"] == "private-cpu-pod"
    assert (
        report["healthz"] == report["livez"] == {"status": 200, "payload": {"ok": True}}
    )

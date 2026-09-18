from __future__ import annotations

import json
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from scripts.e2e.regional.probes import notify008_provider as module
from scripts.e2e.regional.probes.notify008_protocol import ProbeError, Variant
from tests.regional._cov95_notify008_support import RUN_ID


class ReadyPipe:
    def __init__(self):
        self.messages = []
        self.closed = False

    def send(self, message):
        self.messages.append(message)

    def close(self):
        self.closed = True


@pytest.mark.parametrize("seconds", [0, -1, 601, True, "5", float("inf"), float("nan")])
def test_provider_startup_refuses_invalid_lifetime_before_opening_a_ledger(
    tmp_path, seconds
):
    ready = ReadyPipe()
    ledger = tmp_path / "provider.jsonl"
    module.serve_provider(ledger, RUN_ID, ready, seconds)
    assert ready.messages == [{"error": "ProbeError"}] and ready.closed, (
        "invalid lifetime must report failure and close the readiness endpoint"
    )
    assert not ledger.exists(), "refused startup must not create an acceptance ledger"


def test_provider_normal_deadline_closes_real_server_and_private_ledger(tmp_path):
    ready = ReadyPipe()
    path = tmp_path / "provider.jsonl"
    module.serve_provider(path, RUN_ID, ready, 0.01)
    assert len(ready.messages) == 1 and set(ready.messages[0]) == {"pid", "port"}, (
        "a local bounded provider must publish one explicit readiness acknowledgement"
    )
    assert ready.closed and path.read_bytes() == b"", (
        "a no-request provider must terminate with no accepted notifications"
    )
    with path.open("ab") as stream:
        assert stream.tell() == 0, (
            "the stopped provider must leave a readable empty ledger"
        )


def test_provider_server_failure_closes_the_owned_ledger_and_sanitizes_error(
    tmp_path, monkeypatch
):
    ready = ReadyPipe()
    path = tmp_path / "provider.jsonl"
    opened = []
    ledger_type = module.ReceiptLog

    def tracked(*args):
        value = ledger_type(*args)
        opened.append(value)
        return value

    def unavailable(*args):
        raise OSError("private-bind-detail")

    monkeypatch.setattr(module, "ReceiptLog", tracked)
    monkeypatch.setattr(module, "HTTPServer", unavailable)
    module.serve_provider(path, RUN_ID, ready, 2)
    assert ready.messages == [{"error": "OSError"}] and ready.closed, (
        "socket startup refusal must report only the error class"
    )
    assert opened[0].stream.closed, "server failure must close the newly owned ledger"


@pytest.mark.parametrize(
    "fault", ["status", "oversized", "not-object", "run", "notification", "message-id"]
)
def test_provider_client_rejects_unbounded_or_unbound_ack_without_proxying(
    monkeypatch, fault
):
    notification_id = Variant("inline", "before-provider", "gpu-reset").notification_id(
        RUN_ID
    )
    value = {
        "run_id": RUN_ID,
        "notification_id": notification_id,
        "message_id": "simulated-" + "1" * 32,
    }
    if fault == "not-object":
        value = []
    elif fault == "run":
        value["run_id"] = "foreign"
    elif fault == "notification":
        value["notification_id"] = "foreign"
    elif fault == "message-id":
        value["message_id"] = None
    payload = b"x" * 4097 if fault == "oversized" else json.dumps(value).encode()
    requests = []

    @contextmanager
    def opened(request, timeout):
        requests.append((request, timeout))
        yield SimpleNamespace(
            status=503 if fault == "status" else 200, read=lambda count: payload[:count]
        )

    def opener(proxy):
        assert proxy.proxies == {}, (
            "the local provider client must disable ambient proxies"
        )
        return SimpleNamespace(open=opened)

    monkeypatch.setattr(module, "build_opener", opener)
    with pytest.raises(ProbeError, match="acknowledge|identity"):
        module.accept_notification(32800, RUN_ID, notification_id)
    ((request, timeout),) = requests
    assert request.full_url == "http://127.0.0.1:32800/accept" and timeout == 3, (
        "failure testing must retain the fixed loopback destination and finite transport timeout"
    )

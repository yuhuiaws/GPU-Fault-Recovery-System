from __future__ import annotations

import json
import os
from urllib.error import HTTPError
from urllib.request import ProxyHandler, Request, build_opener

import pytest

from scripts.e2e.regional.probes import notify008_provider as module
from scripts.e2e.regional.probes.notify008_protocol import ProbeError, Variant
from tests.regional._cov95_notify008_support import RUN_ID
from tests.regional._cov95_notify008_support import provider_fixture as provider_fixture


def test_provider_process_records_distinct_durable_acceptances_independently(provider):
    notification_id = Variant(
        "inline", "accepted-before-commit", "gpu-reset"
    ).notification_id(RUN_ID)
    first = module.accept_notification(provider.port, RUN_ID, notification_id)
    second = module.accept_notification(provider.port, RUN_ID, notification_id)

    assert provider.pid != os.getpid(), (
        "the accepting provider must not be the test caller"
    )
    assert first["provider_pid"] == second["provider_pid"] == provider.pid, (
        "both receipts must name the independent spawned provider"
    )
    assert first["message_id"] != second["message_id"], (
        "a non-idempotent simulated provider must expose repeated acceptance"
    )
    assert module.read_receipts(provider.ledger, run_id=RUN_ID) == [first, second], (
        "an independent file read must observe both receipts before the provider exits"
    )
    assert (
        module.read_receipts(
            provider.ledger, run_id=RUN_ID, notification_id="unrelated"
        )
        == []
    ), "receipt selection must not attribute another notification's acceptance"
    assert provider.ledger.stat().st_mode & 0o777 == 0o600, (
        "the local receipt ledger must be private"
    )


@pytest.mark.parametrize(
    ("path", "body", "content_type"),
    [
        ("/wrong", b"{}", "application/json"),
        ("/accept", b"", "application/json"),
        ("/accept", b"x" * 4097, "application/json"),
        ("/accept", b"not-json", "application/json"),
        ("/accept", b"{}", "text/plain"),
        ("/accept", b"[]", "application/json"),
        (
            "/accept",
            b'{"run_id":"foreign","notification_id":"foreign"}',
            "application/json",
        ),
    ],
)
def test_provider_refuses_unknown_or_unbounded_requests_without_accepting(
    provider, path, body, content_type
):
    request = Request(
        f"http://127.0.0.1:{provider.port}{path}",
        data=body,
        headers={"Content-Type": content_type},
        method="POST",
    )
    with pytest.raises(HTTPError) as raised:
        build_opener(ProxyHandler({})).open(request, timeout=3)
    assert raised.value.code == 400, "malformed fixture requests must be refused"
    assert module.read_receipts(provider.ledger, run_id=RUN_ID) == [], (
        "refused input must not create acceptance evidence"
    )


def test_receipt_sync_failure_is_not_acknowledged(tmp_path, monkeypatch):
    ledger = module.ReceiptLog(tmp_path / "receipts.jsonl", RUN_ID)
    request = {
        "run_id": RUN_ID,
        "notification_id": Variant(
            "inline", "before-provider", "gpu-reset"
        ).notification_id(RUN_ID),
    }

    def unavailable(_fd):
        raise OSError("local receipt sync unavailable")

    monkeypatch.setattr(module.os, "fsync", unavailable)
    try:
        with pytest.raises(OSError, match="sync unavailable"):
            ledger.accept(request)
        assert ledger.records == [], "a failed durable append must not be acknowledged"
    finally:
        ledger.close()


def test_provider_budget_cannot_be_extended_by_repeating_a_valid_notification(tmp_path):
    ledger = module.ReceiptLog(tmp_path / "receipts.jsonl", RUN_ID)
    request = {
        "run_id": RUN_ID,
        "notification_id": Variant(
            "inline", "before-provider", "gpu-reset"
        ).notification_id(RUN_ID),
    }
    try:
        for _ in range(16):
            ledger.accept(request)
        with pytest.raises(ProbeError, match="budget"):
            ledger.accept(request)
        assert len(ledger.records) == 16, "the acceptance budget must remain fixed"
    finally:
        ledger.close()


@pytest.mark.parametrize("port", [0, 65536, True, "80"])
def test_provider_client_refuses_invalid_ports_before_any_transport(port):
    with pytest.raises(ProbeError, match="port"):
        module.accept_notification(port, RUN_ID, "unused")


@pytest.mark.parametrize(
    "payload", [b"partial", b"{}\n", b'{"run_id":"foreign","sequence":1}\n']
)
def test_observer_refuses_partial_or_foreign_receipt_ledger(tmp_path, payload):
    path = tmp_path / "invalid.jsonl"
    path.write_bytes(payload)
    with pytest.raises(ProbeError):
        module.read_receipts(path, run_id=RUN_ID)


def test_observer_refuses_missing_symlink_and_oversized_ledger(tmp_path):
    missing = tmp_path / "missing"
    with pytest.raises(ProbeError, match="absent"):
        module.read_receipts(missing, run_id=RUN_ID)
    real = tmp_path / "real"
    real.write_bytes(b"")
    missing.symlink_to(real)
    with pytest.raises(ProbeError, match="absent"):
        module.read_receipts(missing, run_id=RUN_ID)
    real.write_bytes(b"x" * (4096 * 16 + 1))
    with pytest.raises(ProbeError, match="oversized"):
        module.read_receipts(real, run_id=RUN_ID)
    real.write_bytes(
        b"".join(
            json.dumps({"run_id": RUN_ID, "sequence": index}).encode() + b"\n"
            for index in range(1, 18)
        )
    )
    with pytest.raises(ProbeError, match="budget"):
        module.read_receipts(real, run_id=RUN_ID)

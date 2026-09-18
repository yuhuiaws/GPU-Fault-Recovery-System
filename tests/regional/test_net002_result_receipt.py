"""NET002 first-result evidence is immutable, scoped and free of raw responses."""

from __future__ import annotations

import json
import os
import ssl
import stat
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts.e2e.regional import run_net002_command_recovery as runner
from scripts.e2e.regional.probes import net002_executor as probe
from tests.regional._cov95_collect_net import Clock


@pytest.fixture
def gated_client(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Any:
    for name, filename in (
        ("BLOCK", "block"),
        ("RESULT_SUBMIT_WAITING", "waiting.json"),
        ("RESULT_SUBMIT_RELEASED", "released.json"),
    ):
        monkeypatch.setattr(probe, name, tmp_path / filename)
    clock = Clock()
    calls: list[str] = []

    def release(_seconds: float) -> None:
        assert calls == [], (
            "no transport closure or submission is allowed while blocked"
        )
        probe.BLOCK.unlink()

    def close_pool() -> None:
        assert not probe.BLOCK.exists(), (
            "gated_client: expected no probe.BLOCK.exists()"
        )
        calls.append("close-caller-pool")

    clock.on_sleep = release
    monkeypatch.setattr(probe, "time", clock)
    monkeypatch.setattr(probe, "CONNECTION_POOL", SimpleNamespace(close=close_pool))
    client = probe.GatedRegionalExecutorClient(
        "https://control.invalid", "net002-synthetic", "example-only"
    )
    client.ssl_context = ssl.create_default_context()
    return SimpleNamespace(
        client=client,
        clock=clock,
        calls=calls,
        path=tmp_path / "first-result-submission.json",
        command=SimpleNamespace(command_id="remote-net002"),
    )


def rejected(body: str, code: Any = 409) -> probe.ClusterExecutorError:
    return probe.ClusterExecutorError(
        f"regional control plane rejected request ({code}): {body}", status_code=code
    )


@pytest.mark.parametrize("detail", runner.STALE_LEASE_DETAILS)
@pytest.mark.parametrize("json_body", [False, True], ids=["legacy-text", "json"])
def test_first_stale_result_is_recorded_once_without_response_secrets(
    gated_client: Any, monkeypatch: pytest.MonkeyPatch, detail: str, json_body: bool
) -> None:
    harness = gated_client
    body = (
        json.dumps({"detail": detail, "untrusted": "do-not-persist"})
        if json_body
        else detail
    )
    original = rejected(body)
    context = harness.client.ssl_context
    completed = object()

    def complete(client: Any, command: Any, result: Any) -> Any:
        assert not probe.BLOCK.exists(), (
            "test_first_stale_result_is_recorded_once_without_response_secrets: expected no probe.BLOCK.exists()"
        )
        assert client.ssl_context is context
        assert context.check_hostname and context.verify_mode == ssl.CERT_REQUIRED
        harness.calls.append("complete")
        if harness.calls.count("complete") == 1:
            raise original
        return completed

    monkeypatch.setattr(probe.RegionalExecutorClient, "complete", complete)
    probe.BLOCK.touch()
    with pytest.raises(probe.ClusterExecutorError) as raised:
        harness.client.complete(harness.command, None)
    assert raised.value is original, "the actual failure must propagate unchanged"
    assert harness.calls == ["close-caller-pool", "complete"]
    first_bytes = harness.path.read_bytes()
    first = json.loads(first_bytes)
    released = json.loads(probe.RESULT_SUBMIT_RELEASED.read_text())
    assert first == {
        "command_id": "remote-net002",
        "submission_index": 1,
        "status_code": 409,
        "stale_lease_reason": detail,
        "submitted_at_epoch": harness.clock.now,
        "observed_at_epoch": harness.clock.now,
        "gate_released_at_epoch": released["observed_at_epoch"],
        "caller_transport_pool_closed": True,
    }
    assert b"do-not-persist" not in first_bytes
    assert harness.path.stat().st_mode & 0o777 == 0o600
    assert harness.client.complete(harness.command, {"cached": True}) is completed
    assert (
        harness.client.complete(SimpleNamespace(command_id="unrelated"), None)
        is completed
    )
    assert harness.path.read_bytes() == first_bytes, (
        "later results must not rewrite proof"
    )
    assert harness.calls.count("close-caller-pool") == 1


@pytest.mark.parametrize(
    ("error", "code"),
    [
        (rejected("unrelated conflict"), 409),
        (rejected(json.dumps({"detail": runner.STALE_LEASE_DETAIL + " extra"})), 409),
        (rejected(json.dumps({"detail": [runner.STALE_LEASE_DETAIL]})), 409),
        (rejected(json.dumps({"other": runner.STALE_LEASE_DETAIL})), 409),
        (rejected(runner.STALE_LEASE_DETAIL, 403), 403),
        (rejected(runner.STALE_LEASE_DETAIL, "409"), None),
        (rejected(runner.STALE_LEASE_DETAIL, 409.0), None),
        (rejected(runner.STALE_LEASE_DETAIL, True), None),
        (rejected(runner.STALE_LEASE_DETAIL, 999), None),
        (
            probe.ClusterExecutorError(
                "renewal " + runner.STALE_LEASE_DETAIL, status_code=409
            ),
            409,
        ),
        (probe.ClusterExecutorError("transport: do-not-persist"), None),
        (ConnectionResetError("do-not-persist"), None),
    ],
    ids=[
        "unrelated",
        "extra-text",
        "list",
        "other-key",
        "forbidden",
        "string-code",
        "float-code",
        "bool-code",
        "invalid-code",
        "no-response-envelope",
        "wrapped-transport",
        "raw-transport",
    ],
)
def test_only_closed_stale_reasons_and_safe_status_codes_are_persisted(
    gated_client: Any, monkeypatch: pytest.MonkeyPatch, error: Exception, code: Any
) -> None:
    harness = gated_client

    def complete(*args: Any) -> None:
        raise error

    monkeypatch.setattr(probe.RegionalExecutorClient, "complete", complete)
    with pytest.raises(type(error)) as raised:
        harness.client.complete(harness.command, None)
    assert raised.value is error
    first_bytes = harness.path.read_bytes()
    receipt = json.loads(first_bytes)
    assert receipt["status_code"] == code
    assert receipt["stale_lease_reason"] is None
    assert receipt["caller_transport_pool_closed"] is False
    assert b"do-not-persist" not in first_bytes
    monkeypatch.setattr(
        probe.RegionalExecutorClient, "complete", lambda *args: "cached"
    )
    assert harness.client.complete(harness.command, None) == "cached"
    assert harness.path.read_bytes() == first_bytes


def test_existing_receipt_is_not_replaced_by_a_new_client(
    gated_client: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = gated_client
    original = {"command_id": "earlier-command", "status_code": None}
    harness.path.write_text(json.dumps(original))
    monkeypatch.setattr(
        probe.RegionalExecutorClient, "complete", lambda *args: "cached"
    )
    assert harness.client.complete(harness.command, None) == "cached"
    assert json.loads(harness.path.read_text()) == original
    assert set(harness.path.parent.iterdir()) == {harness.path}


def test_deleted_first_receipt_is_not_recreated_from_a_later_success(
    gated_client: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = gated_client
    monkeypatch.setattr(
        probe.RegionalExecutorClient, "complete", lambda *args: "cached"
    )
    assert harness.client.complete(harness.command, None) == "cached"
    assert json.loads(harness.path.read_text())["status_code"] == 200
    harness.path.unlink()
    assert harness.client.complete(harness.command, None) == "cached"
    assert not harness.path.exists(), (
        "test_deleted_first_receipt_is_not_recreated_from_a_later_success: expected no harness.path.exists()"
    )


def test_receipt_is_fsynced_before_publication_and_directory_is_synced(
    gated_client: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = gated_client
    fsync, link = os.fsync, os.link
    events: list[str] = []

    def sync(fd: int) -> None:
        events.append(
            "directory-sync" if stat.S_ISDIR(os.fstat(fd).st_mode) else "file-sync"
        )
        fsync(fd)

    def publish(source: Path, destination: Path) -> None:
        assert destination == harness.path
        assert not destination.exists(), (
            "test_receipt_is_fsynced_before_publication_and_directory_is_synced: expected no destination.exists()"
        )
        assert (
            json.loads(source.read_text())["command_id"] == harness.command.command_id
        )
        assert events == ["file-sync"], (
            "complete bytes must be durable before publication"
        )
        events.append("publish")
        link(source, destination)

    monkeypatch.setattr(os, "fsync", sync)
    monkeypatch.setattr(os, "link", publish)
    monkeypatch.setattr(
        probe.RegionalExecutorClient, "complete", lambda *args: "cached"
    )
    assert harness.client.complete(harness.command, None) == "cached"
    assert events == ["file-sync", "publish", "directory-sync"]
    assert set(harness.path.parent.iterdir()) == {harness.path}

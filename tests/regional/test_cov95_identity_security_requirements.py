from __future__ import annotations

import hashlib
import ssl
import urllib.request
from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import audit_auth_boundary as audit
from tests.regional._cov95_identity_support import ASGIBridge
from tests.regional._cov95_identity_support import offline_guard as offline_guard
from tests.regional._regional_support import TOKEN_A, enqueue_remote_command
from tests.regional.test_regional_command_protocol_acceptance import regional_context


def command_api(monkeypatch: pytest.MonkeyPatch) -> Any:
    context = regional_context()
    bridge = ASGIBridge(context)
    monkeypatch.setattr(urllib.request, "urlopen", bridge.urlopen)
    monkeypatch.setattr(ssl, "create_default_context", lambda **kwargs: object())
    return context


def test_invalid_results_preserve_the_entire_leased_command(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    context = command_api(monkeypatch)
    command = enqueue_remote_command(
        context.store, "remote-" + "8" * 24, owner=audit.ACCEPTANCE_PROBE_OWNER
    )
    ca = tmp_path / "public-ca"
    claimed = audit.post(
        "https://unit.invalid",
        ca,
        "/v1/regional/executors/claim",
        cluster_id="cluster-a",
        token=TOKEN_A,
        payload=audit.claim_payload(
            executor_id="unit-executor",
            artifact_sha256="a" * 64,
            compatibility_digest="b" * 64,
        ),
    )
    assert claimed["status"] == 200
    stored = context.store.get_remote_command(command.command_id)
    lease = stored.lease_token
    assert stored.status.value == "LEASED" and lease

    def fingerprint() -> str:
        return hashlib.sha256(
            context.store.get_remote_command(command.command_id)
            .model_dump_json()
            .encode()
        ).hexdigest()

    baseline = fingerprint()
    path = f"/v1/regional/executors/{command.command_id}/result"
    bodies = [
        {"lease_token": lease, "status": "PENDING"},
        {"lease_token": lease, "status": "LEASED"},
        {"lease_token": lease, "status": "FAILED"},
        {"lease_token": lease, "status": "FAILED", "error": ""},
        {"status": "SUCCEEDED"},
        {"lease_token": lease, "status": "DONE"},
        {"lease_token": lease, "status": "SUCCEEDED", "unexpected": True},
    ]
    for payload in bodies:
        response = audit.post(
            "https://unit.invalid",
            ca,
            path,
            cluster_id="cluster-a",
            token=TOKEN_A,
            payload=payload,
        )
        assert response["status"] == 422
        assert fingerprint() == baseline, (
            "rejected result changed the full command record"
        )
    accepted = audit.post(
        "https://unit.invalid",
        ca,
        path,
        cluster_id="cluster-a",
        token=TOKEN_A,
        payload={"lease_token": lease, "status": "WAITING"},
    )
    assert accepted["status"] == 200
    assert accepted["body"]["status"] == "WAITING"
    assert fingerprint() != baseline


def test_foreign_and_unknown_command_lookup_are_non_enumerating_and_nonmutating(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    context = command_api(monkeypatch)
    foreign = enqueue_remote_command(
        context.store, "remote-" + "b" * 24, cluster_id="cluster-b"
    )
    baseline = hashlib.sha256(foreign.model_dump_json().encode()).hexdigest()
    answers = []
    for command_id in ("remote-" + "0" * 24, foreign.command_id):
        response = audit.post(
            "https://unit.invalid",
            tmp_path / "public-ca",
            f"/v1/regional/executors/{command_id}/result",
            cluster_id="cluster-a",
            token=TOKEN_A,
            payload={"lease_token": "example-irrelevant-lease", "status": "SUCCEEDED"},
        )
        assert response == {
            "status": 404,
            "body": {"detail": f"resource not found: cluster-a/{command_id}"},
        }
        answers.append(response["body"]["detail"].removesuffix(command_id))
    assert answers == ["resource not found: cluster-a/"] * 2
    after = context.store.get_remote_command(foreign.command_id)
    assert hashlib.sha256(after.model_dump_json().encode()).hexdigest() == baseline
    assert after.status.value == "PENDING"

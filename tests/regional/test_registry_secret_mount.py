"""``secret_drift`` compares the registry Secret as it is now (batch 245).

join-cluster and remove-cluster rewrite the ``gpu-fault-regional-clusters``
Secret and publish the durable head, and they never restart the CPU
Deployments: registry membership is hot (review H4). The drift flag compared
the head with the digest of the Secret each process was STARTED with, so after
every join or remove each replica that predated it reported
``secret_drift=True`` although the Secret and the head agreed (live
2026-11-03: three api-ha replicas, Secret and head both at generation 13, the
perf registry-alignment preflight refused). The CPU Deployments now also mount
the Secret, kubelet keeps that file current, and the runtime re-reads it on
every refresh; the start-up digest stays the record and the fallback where
nothing is mounted. A Secret-only rewrite (review H3) still drifts -- and is
now seen without a restart.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pytest

from gpu_fault import regional_registry_runtime as runtime_module
from gpu_fault.regional import RegionalClusterLifecycle, RegionalRegistryRevision
from gpu_fault.regional_registry import (
    configured_regional_registrations,
    regional_registry_config_sha256,
    regional_registry_document_sha256,
)
from gpu_fault.regional_registry_runtime import (
    SECRET_SOURCE_MOUNTED,
    SECRET_SOURCE_STARTUP,
    RegionalRegistryRuntime,
)
from gpu_fault.store import InMemoryStore
from tests.regional._regional_support import NOW, TOKEN_A, TOKEN_B, registration

LEAKED_TOKEN = "plaintext-token-" + "x" * 24


def _entry(cluster_id: str, token: str) -> dict[str, Any]:
    """One ``clusters.json`` entry the way the release tooling writes it."""

    item = registration(cluster_id, token).model_dump(mode="json")
    for field in ("token_sha256", "created_at", "updated_at", "lifecycle_state"):
        item.pop(field)
    return {**item, "token": token}


def _digest(entries: list[dict[str, Any]]) -> str:
    return regional_registry_config_sha256(configured_regional_registrations(entries))


def _write(path: Path, entries: list[dict[str, Any]]) -> None:
    path.write_text(json.dumps(entries, sort_keys=True), encoding="utf-8")


def _publish(store: InMemoryStore, registrations: list[Any], *, reason: str) -> None:
    generation = store.get_regional_registry_head().generation
    store.publish_regional_registry_revision(
        RegionalRegistryRevision.build(
            generation=generation + 1,
            registrations=registrations,
            previous_generation=generation,
            required_member_ids=[],
            reason=reason,
            created_at=NOW,
        ),
        expected_generation=generation,
    )


def _runtime(
    store: InMemoryStore, secret_file: Path | None, startup: list[dict[str, Any]]
) -> RegionalRegistryRuntime:
    return RegionalRegistryRuntime.bootstrap(
        store,
        member_id="api-ha-0",
        service_role="ingress",
        release_id="release-a",
        poll_seconds=1,
        stale_seconds=5,
        now=lambda: NOW,
        secret_config_sha256=_digest(startup),
        secret_file=str(secret_file) if secret_file is not None else None,
    )


def _site(tmp_path: Path) -> tuple[InMemoryStore, Path, list[dict[str, Any]]]:
    store = InMemoryStore()
    store.save_regional_cluster(registration("cluster-a", TOKEN_A))
    secret = tmp_path / "clusters.json"
    startup = [_entry("cluster-a", TOKEN_A)]
    _write(secret, startup)
    return store, secret, startup


def test_a_replica_started_before_a_join_reports_no_drift_once_secret_and_head_agree(
    tmp_path: Path,
) -> None:
    store, secret, startup = _site(tmp_path)
    runtime = _runtime(store, secret, startup)
    assert runtime.status()["secret_drift"] is False

    # join-cluster writes the Secret first (update_registry), then publishes
    # the cluster as PENDING; for that moment the two really disagree.
    joined = [*startup, _entry("cluster-b", TOKEN_B)]
    _write(secret, joined)
    assert runtime.refresh_once() is True
    assert runtime.secret_drift() is True, "Secret ahead of the head is a drift"

    pending = registration("cluster-b", TOKEN_B).model_copy(
        update={"lifecycle_state": RegionalClusterLifecycle.PENDING}
    )
    _publish(store, [registration("cluster-a", TOKEN_A), pending], reason="join")
    assert runtime.refresh_once() is True

    status = runtime.status()
    assert status["secret_drift"] is False, status
    assert status["secret_source"] == SECRET_SOURCE_MOUNTED
    assert status["secret_config_sha256"] == _digest(joined)
    assert status["secret_config_sha256"] == status["durable_config_sha256"]
    assert status["secret_startup_config_sha256"] == _digest(startup), (
        "the start-up copy is kept as the record; it no longer decides drift"
    )
    assert status["secret_error"] is None


def test_a_replica_started_before_a_remove_reports_no_drift_after_the_purge(
    tmp_path: Path,
) -> None:
    store, secret, _ = _site(tmp_path)
    store.save_regional_cluster(registration("cluster-b", TOKEN_B))
    startup = [_entry("cluster-a", TOKEN_A), _entry("cluster-b", TOKEN_B)]
    _write(secret, startup)
    runtime = _runtime(store, secret, startup)
    assert runtime.secret_drift() is False

    # remove-cluster: REVOKED first (lifecycle is not configuration), then the
    # Secret loses the entry, then the purge publishes the head without it.
    revoked = registration("cluster-b", TOKEN_B).model_copy(
        update={"lifecycle_state": RegionalClusterLifecycle.REVOKED}
    )
    _publish(store, [registration("cluster-a", TOKEN_A), revoked], reason="revoked")
    assert runtime.refresh_once() is True
    assert runtime.secret_drift() is False
    _write(secret, startup[:1])
    _publish(store, [registration("cluster-a", TOKEN_A)], reason="purged")
    assert runtime.refresh_once() is True

    status = runtime.status()
    assert status["secret_drift"] is False, status
    assert status["secret_startup_config_sha256"] != status["durable_config_sha256"]


def test_a_secret_only_rewrite_drifts_without_waiting_for_a_restart(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Review H3: a release that rewrote only the Secret never reached the
    running registry. The start-up digest saw that only in a Pod started after
    the rewrite; the mounted copy shows it in every running replica."""

    store, secret, startup = _site(tmp_path)
    runtime = _runtime(store, secret, startup)
    rewritten = [{**startup[0], "agent_endpoint_allowed_cidrs": ["10.9.0.0/16"]}]
    _write(secret, rewritten)

    with caplog.at_level(logging.WARNING, logger=runtime_module.__name__):
        assert runtime.refresh_once() is True
        assert runtime.refresh_once() is True

    status = runtime.status()
    assert status["secret_drift"] is True, status
    assert status["secret_config_sha256"] == _digest(rewritten)
    assert status["secret_startup_config_sha256"] == status["durable_config_sha256"], (
        "the start-up copy still agrees: only the mounted copy sees this drift"
    )
    warnings = [r.getMessage() for r in caplog.records if "drifts" in r.getMessage()]
    assert len(warnings) == 1, "one warning per transition, not per refresh"
    assert _digest(rewritten) in warnings[0] and SECRET_SOURCE_MOUNTED in warnings[0]
    assert status["durable_config_sha256"] in warnings[0]


def test_without_a_mount_the_start_up_copy_still_decides(tmp_path: Path) -> None:
    """A Pod of a release that does not mount the Secret (and every tool that
    passes no file) behaves as before: only a restart clears the drift."""

    store, _, startup = _site(tmp_path)
    runtime = _runtime(store, tmp_path / "not-mounted" / "clusters.json", startup)
    assert runtime.secret_source() == SECRET_SOURCE_STARTUP
    _publish(
        store,
        [registration("cluster-a", TOKEN_A), registration("cluster-b", TOKEN_B)],
        reason="join",
    )
    assert runtime.refresh_once() is True

    status = runtime.status()
    assert status["secret_drift"] is True, status
    assert status["secret_source"] == SECRET_SOURCE_STARTUP
    assert status["secret_config_sha256"] == status["secret_startup_config_sha256"]


def test_a_mounted_secret_that_does_not_load_drifts_and_never_leaks_it(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    store, secret, startup = _site(tmp_path)
    runtime = _runtime(store, secret, startup)

    secret.write_text('[{"token": "' + LEAKED_TOKEN + '", ', encoding="utf-8")
    with caplog.at_level(logging.WARNING, logger=runtime_module.__name__):
        assert runtime.refresh_once() is True, "a bad Secret never fails the refresh"
    status = runtime.status()
    assert status["secret_drift"] is True
    assert status["secret_config_sha256"] is None
    assert status["secret_error"] == "regional registry Secret is not valid JSON"
    assert runtime.is_ready() is True, "drift is a report, not a readiness gate"

    invalid_entry = [{**startup[0], "token": LEAKED_TOKEN, "unexpected": LEAKED_TOKEN}]
    _write(secret, invalid_entry)
    with caplog.at_level(logging.WARNING, logger=runtime_module.__name__):
        assert runtime.refresh_once() is True
    status = runtime.status()
    assert status["secret_error"] == "regional registry Secret entry is invalid"
    assert LEAKED_TOKEN not in json.dumps(status, default=str)
    assert LEAKED_TOKEN not in caplog.text
    assert caplog.text.count("Secret drifts from the durable head") == 2, (
        "a changed error is a new line; an unchanged one stays quiet"
    )

    _write(secret, startup)
    assert runtime.refresh_once() is True
    assert runtime.status()["secret_drift"] is False
    assert runtime.status()["secret_error"] is None


def test_a_mounted_secret_that_disappears_drifts_until_it_is_back(
    tmp_path: Path,
) -> None:
    store, secret, startup = _site(tmp_path)
    runtime = _runtime(store, secret, startup)

    secret.unlink()
    assert runtime.refresh_once() is True
    status = runtime.status()
    assert status["secret_drift"] is True
    assert status["secret_source"] == SECRET_SOURCE_MOUNTED, "no silent fallback"
    assert status["secret_error"] == (
        "registry Secret file is unreadable (FileNotFoundError)"
    )

    _write(secret, startup)
    assert runtime.refresh_once() is True
    assert runtime.status()["secret_drift"] is False


def test_an_unchanged_secret_is_read_but_not_parsed_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[int] = []

    def counting(raw: bytes | str) -> str:
        calls.append(len(raw))
        return regional_registry_document_sha256(raw)

    monkeypatch.setattr(runtime_module, "regional_registry_document_sha256", counting)
    store, secret, startup = _site(tmp_path)
    runtime = _runtime(store, secret, startup)
    for _ in range(5):
        assert runtime.refresh_once() is True
    assert len(calls) == 1, "bootstrap parsed it; equal bytes are not re-parsed"

    _write(secret, [*startup, _entry("cluster-b", TOKEN_B)])
    assert runtime.refresh_once() is True
    assert len(calls) == 2


def test_the_document_digest_matches_the_start_up_digest() -> None:
    """One normalisation for both copies, so a quiet site compares equal."""

    entries = [_entry("cluster-a", TOKEN_A), _entry("cluster-b", TOKEN_B)]
    for text in (json.dumps(entries), json.dumps(entries, indent=2) + "\n"):
        assert regional_registry_document_sha256(text) == _digest(entries)
        assert regional_registry_document_sha256(text.encode()) == _digest(entries)
    with pytest.raises(ValueError, match="not a list of objects"):
        regional_registry_document_sha256(json.dumps({"cluster_id": "cluster-a"}))
    with pytest.raises(ValueError, match="not valid JSON"):
        regional_registry_document_sha256(b"\xff\xfe" + LEAKED_TOKEN.encode())

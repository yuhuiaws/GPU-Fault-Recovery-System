"""Registry dual truth (architecture review H3).

The running control plane serves the durable Aurora registry head and ignores
the ``GPU_FAULT_REGIONAL_CLUSTERS_JSON`` Secret once a head exists, but the
release engine's upgrade path only rewrote the Secret. Two fixes: the runtime
reports ``secret_drift`` when the Secret and the durable head disagree, and the
upgrade REGISTRY component publishes a durable revision through the same
publish-and-converge path join/remove use.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from gpu_fault.regional import RegionalRegistryRevision
from gpu_fault.regional_registry import (
    configured_regional_registrations,
    regional_registry_config_sha256,
)
from gpu_fault.regional_registry_runtime import RegionalRegistryRuntime
from gpu_fault.store import InMemoryStore
from gpu_fault_release import regional_release_online_registry as ONLINE_REGISTRY
from gpu_fault_release import regional_release_orchestration as ORCHESTRATION
from tests.regional._regional_support import TOKEN_A, TOKEN_B, registration
from tests.regional._release_orchestrator_support import phase_release

NOW = datetime(2026, 9, 7, tzinfo=timezone.utc)


def _runtime(store: InMemoryStore, *, secret_config_sha256: str | None):
    return RegionalRegistryRuntime.bootstrap(
        store,
        member_id="pod-a",
        service_role="ingress",
        release_id="release-a",
        poll_seconds=1,
        stale_seconds=5,
        now=lambda: NOW,
        secret_config_sha256=secret_config_sha256,
    )


def _secret_entry(cluster_id: str, token: str) -> dict:
    item = registration(cluster_id, token).model_dump(mode="json")
    item.pop("token_sha256")
    item.pop("created_at")
    item.pop("updated_at")
    item.pop("lifecycle_state")
    item["token"] = token
    return item


def test_secret_matching_the_durable_head_reports_no_drift() -> None:
    store = InMemoryStore()
    store.save_regional_cluster(registration("cluster-a", TOKEN_A))
    secret_digest = regional_registry_config_sha256(
        configured_regional_registrations([_secret_entry("cluster-a", TOKEN_A)])
    )

    runtime = _runtime(store, secret_config_sha256=secret_digest)

    status = runtime.status()
    assert status["secret_drift"] is False, status
    assert status["secret_config_sha256"] == secret_digest


def test_secret_behind_the_durable_head_reports_drift_and_warns(
    caplog: pytest.LogCaptureFixture,
) -> None:
    store = InMemoryStore()
    store.save_regional_cluster(registration("cluster-a", TOKEN_A))
    seed = _runtime(store, secret_config_sha256=None)
    # A join published cluster-b durably; the Secret still only knows cluster-a.
    store.publish_regional_registry_revision(
        RegionalRegistryRevision.build(
            generation=2,
            registrations=[
                registration("cluster-a", TOKEN_A),
                registration("cluster-b", TOKEN_B),
            ],
            previous_generation=1,
            required_member_ids=[],
            reason="join cluster-b",
            created_at=NOW,
        ),
        expected_generation=seed.snapshot().generation,
    )
    secret_digest = regional_registry_config_sha256(
        configured_regional_registrations([_secret_entry("cluster-a", TOKEN_A)])
    )

    with caplog.at_level(logging.WARNING, logger="gpu_fault.regional_registry_runtime"):
        runtime = _runtime(store, secret_config_sha256=secret_digest)

    status = runtime.status()
    assert status["secret_drift"] is True, status
    warnings = [record for record in caplog.records if "drifts" in record.getMessage()]
    assert len(warnings) == 1, "start-up must warn exactly once about the drift"
    message = warnings[0].getMessage()
    assert secret_digest in message, "the warning must name the Secret digest"
    assert runtime.durable_config_sha256() in message, (
        "the warning must name the durable digest"
    )


def test_secret_digest_ignores_runtime_owned_fields() -> None:
    """Lifecycle and timestamps belong to the durable head, not the Secret."""

    active = registration("cluster-a", TOKEN_A)
    later = active.model_copy(
        update={
            "lifecycle_state": active.lifecycle_state.__class__("DRAINING"),
            "updated_at": datetime(2030, 1, 1, tzinfo=timezone.utc),
        }
    )

    assert regional_registry_config_sha256([active]) == regional_registry_config_sha256(
        [later]
    ), "a lifecycle transition alone must not look like Secret drift"
    assert regional_registry_config_sha256([active]) != regional_registry_config_sha256(
        [active.model_copy(update={"agent_endpoint_allowed_cidrs": ["10.9.0.0/16"]})]
    ), "a configuration change must change the digest"


def _upgrade(release, *, calls: list) -> None:
    ORCHESTRATION.run_upgrade_phases(
        release,
        diff=ORCHESTRATION.ReleaseDiff(
            kind=ORCHESTRATION.ReleaseChangeKind.FULL,
            changed=frozenset({"cluster_registry"}),
        ),
        plan=ORCHESTRATION.ReleaseExecutionPlan(
            nodes=(
                ORCHESTRATION.ReleaseComponent.REGISTRY,
                ORCHESTRATION.ReleaseComponent.CPU_STAGE,
                ORCHESTRATION.ReleaseComponent.VERIFY,
            )
        ),
        previous={},
        completed_phases=set(),
        completed_clusters=set(),
        registry_staged=False,
    )


def test_staged_registry_is_published_durably_before_the_cpu_rolls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        ORCHESTRATION, "preflight_upgrade_mutations", lambda _self, _plan: None
    )
    calls: list[str] = []
    release = phase_release(
        calls, _stage_registry=lambda: calls.append("registry") or True
    )

    _upgrade(release, calls=calls)

    assert calls.index("registry") < calls.index("publish-registry"), (
        "the durable publish must follow the Secret write"
    )
    assert calls.index("publish-registry") < calls.index("cpu-stage"), (
        "the CPU tier must roll onto a registry the fleet already converged on"
    )


def test_unchanged_registry_is_not_republished(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        ORCHESTRATION, "preflight_upgrade_mutations", lambda _self, _plan: None
    )
    calls: list[str] = []
    release = phase_release(calls)

    _upgrade(release, calls=calls)

    assert "registry" in calls
    assert "publish-registry" not in calls, (
        "a registry the Secret already matched must not mint a new generation"
    )


def _online_release(calls: list) -> SimpleNamespace:
    class Runner:
        dry_run = False

        def run(self, arguments, **kwargs):
            calls.append((list(arguments), kwargs))
            if "get" in arguments and "pod" in arguments:
                return "ingress-0"
            return json.dumps(
                {"generation": 7, "content_sha256": "b" * 64, "converged": True}
            )

    return SimpleNamespace(
        runner=Runner(),
        release_id="rel-42",
        config=SimpleNamespace(namespace="gpu-fault-system"),
        _cpu=lambda *arguments: list(arguments),
    )


def test_publish_staged_registry_uses_the_join_publish_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list = []
    release = _online_release(calls)
    monkeypatch.setattr(
        ONLINE_REGISTRY,
        "registry",
        lambda _release: [_secret_entry("cluster-a", TOKEN_A)],
    )

    status = ONLINE_REGISTRY.publish_staged_registry(release)

    assert status["converged"] is True
    publishes = [kwargs for arguments, kwargs in calls if "input_text" in kwargs]
    assert len(publishes) == 1, "exactly one publish-and-converge exec"
    transaction = json.loads(publishes[0]["input_text"])
    assert transaction["path"] == "/v1/regional/registry/revisions"
    assert transaction["use_current_generation"] is True
    assert "rel-42" in transaction["payload"]["reason"]
    registrations = transaction["payload"]["registrations"]
    assert [item["cluster_id"] for item in registrations] == ["cluster-a"]
    assert "token" not in registrations[0], "plaintext tokens never leave the host"
    assert (
        registrations[0]["token_sha256"]
        == registration("cluster-a", TOKEN_A).token_sha256
    )


def test_publish_staged_registry_is_a_no_op_in_dry_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list = []
    release = _online_release(calls)
    release.runner.dry_run = True
    monkeypatch.setattr(
        ONLINE_REGISTRY,
        "registry",
        lambda _release: pytest.fail("dry run must not read the Secret"),
    )

    assert ONLINE_REGISTRY.publish_staged_registry(release) == {}
    assert calls == [], "dry run must not exec into a Pod"

"""Isolated lifecycle publication using real in-memory registry CAS."""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any

import pytest

from gpu_fault.regional import (
    RegionalClusterLifecycle,
    RegionalRegistryPublishRequest,
    RegionalRegistryRevision,
)
from gpu_fault.store import InMemoryStore
from gpu_fault_release import regional_release_online_registry as REGISTRY
from gpu_fault_release.regional_release_config import ReleaseError
from tests.regional._regional_support import TOKEN_A, registration


class DrainWire:
    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        states = {
            "gpu-a": RegionalClusterLifecycle.ACTIVE,
            "gpu-b": RegionalClusterLifecycle.ACTIVE,
            **{
                f"other-{state.value.lower()}": state
                for state in RegionalClusterLifecycle
            },
        }
        self.original = [
            registration(cluster_id, TOKEN_A).model_copy(
                update={
                    "lifecycle_state": state,
                    "created_at": datetime(2026, 7, 1, tzinfo=timezone.utc),
                    "updated_at": datetime(2026, 7, 2, tzinfo=timezone.utc),
                }
            )
            for cluster_id, state in states.items()
        ]
        self.store = InMemoryStore()
        self.store.publish_regional_registry_revision(
            RegionalRegistryRevision.build(
                generation=1,
                previous_generation=None,
                registrations=self.original,
                required_member_ids=[],
                reason="cleanup fixture",
            ),
            expected_generation=0,
        )
        self.secret = [
            item.model_dump(
                mode="json", exclude={"lifecycle_state", "created_at", "updated_at"}
            )
            for item in self.original
        ]
        self.responses: dict[str, Any] = {}
        self.transactions: list[dict[str, Any]] = []
        self.before_publish: Callable[[], None] | None = None
        self.publish_response: Callable[[dict[str, Any]], Any] | None = None
        self.targets: list[str] = []
        self.release = SimpleNamespace(
            _target=self.targets.append, _remote_commands_are_idle=lambda: True
        )
        monkeypatch.setattr(REGISTRY, "registry", lambda _release: self.secret)
        monkeypatch.setattr(REGISTRY, "exec_cpu_ingress_command", self.execute)

    def status(self) -> dict[str, Any]:
        head = self.store.get_regional_registry_head()
        return {
            "generation": head.generation,
            "content_sha256": head.content_sha256,
            "cluster_states": {
                item.cluster_id: item.lifecycle_state.value
                for item in self.store.list_regional_clusters()
            },
            "required_member_ids": [],
            "acked_member_ids": [],
            "missing_member_ids": [],
            "active_member_ids": [],
            "members": [],
            "converged": True,
        }

    def redacted(self) -> list[dict[str, Any]]:
        result = []
        for entry in self.store.list_regional_clusters():
            item = entry.model_dump(mode="json")
            for field in ("token_sha256", "retiring_token_sha256"):
                digest = item.pop(field) or ""
                item[f"{field}_present"] = bool(digest)
                item[f"{field}_length"] = len(digest)
            result.append(item)
        return result

    def execute(
        self,
        _release: Any,
        *,
        arguments: tuple[str, ...],
        input_text: str,
        **kwargs: Any,
    ) -> str:
        assert kwargs["sensitive"] is True, "registry transport lost redaction"
        if arguments[-2] == "GET":
            path = arguments[-1]
            if path in self.responses:
                return json.dumps(self.responses[path])
            if path == "/v1/regional/registry/status":
                return json.dumps(self.status())
            assert path == "/v1/regional/clusters", "unexpected registry read"
            return json.dumps(self.redacted())
        transaction = json.loads(input_text)
        self.transactions.append(transaction)
        assert transaction["path"] == "/v1/regional/registry/revisions"
        assert transaction["use_current_generation"] is False, (
            "publisher replaced the generation bound to the lifecycle evidence"
        )
        assert transaction["timeout_seconds"] == 300
        assert kwargs["timeout_seconds"] == 360
        request = RegionalRegistryPublishRequest.model_validate(transaction["payload"])
        assert request.required_member_ids is None, "caller bypassed current-fleet ACKs"
        if self.before_publish is not None:
            self.before_publish()
        try:
            self.store.publish_regional_registry_revision(
                RegionalRegistryRevision.build(
                    generation=request.expected_generation + 1,
                    previous_generation=request.expected_generation,
                    registrations=request.registrations,
                    required_member_ids=[],
                    reason=request.reason,
                ),
                expected_generation=request.expected_generation,
            )
        except ValueError:
            raise ReleaseError("registry generation conflict") from None
        result = self.status()
        if self.publish_response is not None:
            return json.dumps(self.publish_response(result))
        return json.dumps(result)


def test_batch_drain_preserves_every_unselected_durable_lifecycle_and_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wire = DrainWire(monkeypatch)

    REGISTRY.drain_registry_clusters(wire.release, ["gpu-b", "gpu-a", "gpu-b"])

    assert wire.store.get_regional_registry_head().generation == 2
    assert len(wire.transactions) == 1
    assert wire.transactions[0]["payload"]["expected_generation"] == 1
    for original in wire.original:
        actual = wire.store.get_regional_cluster(original.cluster_id)
        if original.cluster_id in {"gpu-a", "gpu-b"}:
            assert actual.lifecycle_state is RegionalClusterLifecycle.DRAINING
            assert actual.created_at == original.created_at
        else:
            assert actual.model_dump() == original.model_dump(), (
                "drain rewrote an unrelated durable registration"
            )

    REGISTRY.drain_registry_cluster(wire.release, "gpu-a")

    assert wire.store.get_regional_registry_head().generation == 3
    assert wire.store.get_regional_cluster("gpu-b").lifecycle_state is (
        RegionalClusterLifecycle.DRAINING
    ), "a later single-cluster drain reactivated its sibling"


def test_batch_drain_cas_refuses_a_concurrent_registry_revision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wire = DrainWire(monkeypatch)

    def concurrent_transition() -> None:
        entries = [
            item.model_copy(update={"lifecycle_state": RegionalClusterLifecycle.FAILED})
            if item.cluster_id == "other-pending"
            else item
            for item in wire.original
        ]
        wire.store.publish_regional_registry_revision(
            RegionalRegistryRevision.build(
                generation=2,
                previous_generation=1,
                registrations=entries,
                required_member_ids=[],
                reason="concurrent join failure",
            ),
            expected_generation=1,
        )

    wire.before_publish = concurrent_transition
    with pytest.raises(ReleaseError, match="generation conflict"):
        REGISTRY.drain_registry_clusters(wire.release, ["gpu-a", "gpu-b"])

    assert len(wire.transactions) == 1, "CAS failure retried with stale lifecycle data"
    assert wire.store.get_regional_registry_head().generation == 2
    assert wire.store.get_regional_cluster("gpu-a").lifecycle_state is (
        RegionalClusterLifecycle.ACTIVE
    )
    assert wire.store.get_regional_cluster("other-pending").lifecycle_state is (
        RegionalClusterLifecycle.FAILED
    ), "drain overwrote the concurrent lifecycle transition"


@pytest.mark.parametrize(
    "defect",
    [
        "status-shape",
        "unknown-state",
        "missing-state",
        "extra-state",
        "secret-missing",
        "secret-duplicate",
        "secret-token-drift",
        "secret-config-drift",
        "digest-mismatch",
        "entries-shape",
        "entries-incomplete",
        "entries-duplicate",
        "credential-shape",
        "timestamp-drift",
    ],
)
def test_batch_drain_requires_complete_matching_durable_evidence(
    monkeypatch: pytest.MonkeyPatch, defect: str
) -> None:
    wire = DrainWire(monkeypatch)
    status = wire.status()
    entries = wire.redacted()
    if defect == "status-shape":
        wire.responses["/v1/regional/registry/status"] = []
    elif defect == "unknown-state":
        status["cluster_states"]["gpu-a"] = "UNKNOWN"
    elif defect == "missing-state":
        status["cluster_states"].pop("gpu-a")
    elif defect == "extra-state":
        status["cluster_states"]["unexpected"] = "ACTIVE"
    elif defect == "secret-missing":
        wire.secret.pop()
    elif defect == "secret-duplicate":
        wire.secret.append(dict(wire.secret[0]))
    elif defect == "secret-token-drift":
        wire.secret[0]["token_sha256"] = "c" * 64
    elif defect == "secret-config-drift":
        wire.secret[0]["region"] = "us-east-1"
    elif defect == "digest-mismatch":
        status["content_sha256"] = "f" * 64
    elif defect == "entries-shape":
        wire.responses["/v1/regional/clusters"] = [None]
    elif defect == "entries-incomplete":
        entries.pop()
    elif defect == "entries-duplicate":
        entries[0] = dict(entries[1])
    elif defect == "credential-shape":
        entries[0]["token_sha256_present"] = 1
    else:
        entries[0]["updated_at"] = "2026-08-01T00:00:00Z"
    wire.responses.setdefault("/v1/regional/registry/status", status)
    wire.responses.setdefault("/v1/regional/clusters", entries)

    with pytest.raises(ReleaseError):
        REGISTRY.drain_registry_clusters(wire.release, ["gpu-a", "gpu-b"])

    assert wire.transactions == [], "invalid evidence reached the mutation transport"
    assert wire.store.get_regional_registry_head().generation == 1


@pytest.mark.parametrize(
    "cluster_id", ["missing", "other-revoked", "other-failed", "other-pending"]
)
def test_batch_drain_does_not_reopen_an_unmanaged_or_failed_target(
    monkeypatch: pytest.MonkeyPatch, cluster_id: str
) -> None:
    wire = DrainWire(monkeypatch)

    with pytest.raises(ReleaseError):
        REGISTRY.drain_registry_clusters(wire.release, [cluster_id])

    assert wire.transactions == [], "cleanup widened a target's authentication state"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("generation", 3),
        ("content_sha256", "d" * 64),
        ("cluster_states", {}),
        ("converged", False),
        ("missing_member_ids", ["new-worker"]),
        ("members", "unknown"),
    ],
)
def test_batch_drain_rejects_missing_ack_or_another_publish_result(
    monkeypatch: pytest.MonkeyPatch, field: str, value: Any
) -> None:
    wire = DrainWire(monkeypatch)
    wire.publish_response = lambda status: {**status, field: value}

    with pytest.raises(ReleaseError, match="publish evidence|did not converge"):
        REGISTRY.drain_registry_clusters(wire.release, ["gpu-a", "gpu-b"])

    assert len(wire.transactions) == 1, "uncertain publication was replayed"
    assert wire.store.get_regional_registry_head().generation == 2
    assert wire.store.get_regional_cluster("gpu-a").lifecycle_state is (
        RegionalClusterLifecycle.DRAINING
    ), "a convergence failure reopened a potentially published drain"

"""fail-cluster and rollback-cluster settle the registry for a failed join."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from gpu_fault_release import regional_release_online_registry as online
from gpu_fault_release.regional_release_config import ReleaseError


def _release(calls: list[tuple[Any, ...]]) -> Any:
    return SimpleNamespace(
        _update_registry=lambda target, *, remove: calls.append(
            ("update_registry", target.cluster_id, remove)
        )
    )


@pytest.mark.parametrize(
    "reason",
    [
        "command failed (1): kubectl",
        "regional cluster identity differs",
        "regional registry generation changed",
    ],
)
def test_purge_preserves_the_entry_when_the_transition_is_refused(
    monkeypatch: pytest.MonkeyPatch, reason: str
) -> None:
    """A refused transition cannot authorize deleting another attempt's entry."""

    calls: list[tuple[Any, ...]] = []
    failure = ReleaseError(reason)

    def refuse(_release: Any, _cluster_id: str, _state: str, *, reason: str) -> None:
        calls.append(("transition", _cluster_id, _state))
        raise failure

    monkeypatch.setattr(online, "transition_join_registry", refuse)
    monkeypatch.setattr(
        online,
        "purge_registry_cluster",
        lambda _release, cluster_id: calls.append(("purge", cluster_id)),
    )

    with pytest.raises(ReleaseError) as raised:
        online.purge_failed_join(_release(calls), SimpleNamespace(cluster_id="gpu-b"))

    assert raised.value is failure
    assert calls == [("transition", "gpu-b", "ROLLED_BACK")], (
        "a refused rollback transition must leave the Secret and durable revision intact"
    )


def test_purge_updates_the_secret_and_revision_only_after_rollback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[Any, ...]] = []
    monkeypatch.setattr(
        online,
        "transition_join_registry",
        lambda _release, cluster_id, state, *, reason: calls.append(
            ("transition", cluster_id, state)
        ),
    )
    monkeypatch.setattr(
        online,
        "purge_registry_cluster",
        lambda _release, cluster_id: calls.append(("purge", cluster_id)),
    )

    online.purge_failed_join(_release(calls), SimpleNamespace(cluster_id="gpu-b"))

    assert calls == [
        ("transition", "gpu-b", "ROLLED_BACK"),
        ("update_registry", "gpu-b", True),
        ("purge", "gpu-b"),
    ], "purge must follow a successful identity-bound rollback transition"


def test_fail_join_refuses_an_unproven_missing_entry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def refuse(_release: Any, _cluster_id: str, _state: str, *, reason: str) -> None:
        raise ReleaseError("new regional cluster must enter PENDING first")

    monkeypatch.setattr(online, "transition_join_registry", refuse)

    with pytest.raises(ReleaseError, match="PENDING first"):
        online.fail_join_registry(SimpleNamespace(), "gpu-b")


def test_fail_join_marks_a_live_entry_failed(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[tuple[str, str]] = []
    monkeypatch.setattr(
        online,
        "transition_join_registry",
        lambda _release, cluster_id, state, *, reason: seen.append((cluster_id, state)),
    )

    online.fail_join_registry(SimpleNamespace(), "gpu-b")

    assert seen == [("gpu-b", "FAILED")], "a live entry must be marked FAILED"
